"""Read tensor metadata (name, dtype, shape, exact byte size) from weight-file headers.

safetensors: an 8-byte little-endian header length, then a JSON header with each
tensor's dtype, shape and byte offsets.

GGUF: a binary header with metadata key/values and tensor infos; byte sizes come from
the gaps between tensor data offsets, so no per-type size table is needed.
"""

from __future__ import annotations

import json
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .sources import Source, SourceError

MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024
MAX_GGUF_HEADER = 256 * 1024 * 1024


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


# ---------------------------------------------------------------- safetensors


def safetensors_files(files: list[str], index: dict | None) -> list[str]:
    """The safetensors shards vLLM would load from a repo's root."""
    if index is not None:
        return sorted(set(index.get("weight_map", {}).values()))
    root = [f for f in files if "/" not in f and f.endswith(".safetensors")]
    hf_style = [f for f in root if not f.startswith("consolidated")]
    return sorted(hf_style or root)


def parse_safetensors_header(blob: bytes) -> list[Tensor]:
    header = json.loads(blob)
    out = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        out.append(Tensor(name, info["dtype"], tuple(info["shape"]), end - start))
    return out


def read_safetensors_header(source: Source, filename: str) -> list[Tensor]:
    head, _ = source.read_range(filename, 0, 7)
    if len(head) < 8:
        raise SourceError(f"{source.name}: {filename} is too short to be a safetensors file")
    n = struct.unpack("<Q", head)[0]
    if n == 0 or n > MAX_SAFETENSORS_HEADER:
        raise SourceError(f"{source.name}: {filename} has an invalid safetensors header length ({n})")
    blob, _ = source.read_range(filename, 8, 8 + n - 1)
    if len(blob) != n:
        raise SourceError(f"{source.name}: short read of the {filename} header")
    return parse_safetensors_header(blob)


def read_safetensors(source: Source, shards: list[str], workers: int = 8) -> list[Tensor]:
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(shards)))) as ex:
        parts = list(ex.map(lambda f: read_safetensors_header(source, f), shards))
    return [t for part in parts for t in part]


# ---------------------------------------------------------------------- GGUF

_GGUF_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_GGUF_STRING, _GGUF_ARRAY = 8, 9
GGUF_DEFAULT_ALIGNMENT = 32


class _RangeReader:
    """Sequential reader over a remote file that fetches more bytes as needed."""

    def __init__(self, source: Source, filename: str, chunk: int = 1 << 20):
        self.source, self.filename, self.chunk = source, filename, chunk
        self.buf = b""
        self.pos = 0
        self.size: int | None = None

    def _ensure(self, n: int) -> None:
        while len(self.buf) < self.pos + n:
            start = len(self.buf)
            if start >= MAX_GGUF_HEADER:
                raise SourceError(f"{self.source.name}: {self.filename} header is larger than expected")
            want = max(self.chunk, self.pos + n - start)
            data, total = self.source.read_range(self.filename, start, start + want - 1)
            if total is not None:
                self.size = total
            if not data:
                raise SourceError(f"{self.source.name}: unexpected end of {self.filename}")
            self.buf += data
            self.chunk = min(self.chunk * 2, 32 << 20)

    def take(self, n: int) -> bytes:
        self._ensure(n)
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.take(self.unpack("<Q")).decode("utf-8", "replace")


def _gguf_value(r: _RangeReader, vtype: int):
    if vtype in _GGUF_SCALAR:
        return r.unpack(_GGUF_SCALAR[vtype])
    if vtype == _GGUF_STRING:
        return r.string()
    if vtype == _GGUF_ARRAY:
        etype, n = r.unpack("<I"), r.unpack("<Q")
        if etype in _GGUF_SCALAR and n > 4096:
            # Large numeric arrays (token scores, types) are skipped; keep only the length.
            r.take(struct.calcsize(_GGUF_SCALAR[etype]) * n)
            return _ArrayLen(n)
        if etype == _GGUF_STRING and n > 4096:
            for _ in range(n):
                r.take(r.unpack("<Q"))
            return _ArrayLen(n)
        return [_gguf_value(r, etype) for _ in range(n)]
    raise SourceError(f"unknown GGUF value type {vtype}")


class _ArrayLen(int):
    """Stands in for a long GGUF array whose contents were skipped."""


@dataclass
class GGUFFile:
    metadata: dict
    tensors: list[Tensor]


def read_gguf(source: Source, filename: str) -> GGUFFile:
    r = _RangeReader(source, filename)
    if r.take(4) != b"GGUF":
        raise SourceError(f"{source.name}: {filename} is not a GGUF file")
    version = r.unpack("<I")
    if version < 2:
        raise SourceError(f"{source.name}: GGUF version {version} is not supported")
    n_tensors, n_kv = r.unpack("<Q"), r.unpack("<Q")
    meta = {}
    for _ in range(n_kv):
        key = r.string()
        meta[key] = _gguf_value(r, r.unpack("<I"))
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.unpack("<I")
        dims = tuple(r.unpack("<Q") for _ in range(n_dims))
        ttype = r.unpack("<I")
        offset = r.unpack("<Q")
        infos.append((name, dims, ttype, offset))
    alignment = int(meta.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
    data_start = -(-r.pos // alignment) * alignment
    if r.size is None:
        _, r.size = source.read_range(filename, 0, 0)
    if r.size is None:
        raise SourceError(f"{source.name}: cannot determine the size of {filename}")
    data_len = r.size - data_start
    order = sorted(range(len(infos)), key=lambda i: infos[i][3])
    tensors: list[Tensor | None] = [None] * len(infos)
    for k, i in enumerate(order):
        name, dims, ttype, offset = infos[i]
        nxt = infos[order[k + 1]][3] if k + 1 < len(order) else data_len
        # GGUF stores dims innermost-first; report them outermost-first like PyTorch.
        tensors[i] = Tensor(name, f"ggml_type_{ttype}", tuple(reversed(dims)), nxt - offset)
    return GGUFFile(meta, [t for t in tensors if t is not None])


def gguf_files(files: list[str], prefer: str | None = None) -> list[str]:
    """Pick one GGUF model (all of its split parts) from a repo's files.

    `prefer` matches a quantization tag such as "Q4_K_M" in the file name.
    """
    ggufs = [f for f in files if f.endswith(".gguf") and "mmproj" not in f.lower()]
    if not ggufs:
        return []
    groups: dict[str, list[str]] = {}
    for f in ggufs:
        key = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", f)
        groups.setdefault(key, []).append(f)
    keys = sorted(groups)
    if prefer:
        matched = [k for k in keys if prefer.lower() in k.lower()]
        if not matched:
            raise SourceError(f"no GGUF file matching '{prefer}' (have: {', '.join(keys)})")
        keys = matched
    else:
        q4 = [k for k in keys if "q4_k_m" in k.lower()]
        keys = q4 or keys
    return sorted(groups[keys[0]])
