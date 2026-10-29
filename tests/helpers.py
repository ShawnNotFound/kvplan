"""Build small model files (config.json, safetensors, GGUF) for tests."""

from __future__ import annotations

import json
import struct
from pathlib import Path

_ST_SIZES = {"F32": 4, "F16": 2, "BF16": 2, "U8": 1, "I8": 1, "F8_E4M3": 1}


def write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...]]]) -> None:
    """Write a safetensors file with zero-filled data."""
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = _ST_SIZES[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + n]}
        offset += n
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\0" * offset)


def llama_like_tensors(layers: int, hidden: int, heads: int, kv_heads: int, inter: int, vocab: int,
                       tied: bool = False, dtype: str = "BF16") -> dict:
    hd = hidden // heads
    t = {"model.embed_tokens.weight": (dtype, (vocab, hidden)), "model.norm.weight": (dtype, (hidden,))}
    for i in range(layers):
        p = f"model.layers.{i}."
        t |= {
            p + "self_attn.q_proj.weight": (dtype, (heads * hd, hidden)),
            p + "self_attn.k_proj.weight": (dtype, (kv_heads * hd, hidden)),
            p + "self_attn.v_proj.weight": (dtype, (kv_heads * hd, hidden)),
            p + "self_attn.o_proj.weight": (dtype, (hidden, heads * hd)),
            p + "mlp.gate_proj.weight": (dtype, (inter, hidden)),
            p + "mlp.up_proj.weight": (dtype, (inter, hidden)),
            p + "mlp.down_proj.weight": (dtype, (hidden, inter)),
            p + "input_layernorm.weight": (dtype, (hidden,)),
            p + "post_attention_layernorm.weight": (dtype, (hidden,)),
        }
    if not tied:
        t["lm_head.weight"] = (dtype, (vocab, hidden))
    return t


def write_model_dir(path: Path, config: dict, tensors: dict) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))
    write_safetensors(path / "model.safetensors", tensors)
    return path


# GGUF ----------------------------------------------------------------------

def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, value) -> bytes:
    if isinstance(value, bool):
        return _s(key) + struct.pack("<I?", 7, value)
    if isinstance(value, int):
        return _s(key) + struct.pack("<IQ", 10, value) if value > 2**31 else _s(key) + struct.pack("<II", 4, value)
    if isinstance(value, float):
        return _s(key) + struct.pack("<If", 6, value)
    if isinstance(value, str):
        return _s(key) + struct.pack("<I", 8) + _s(value)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return _s(key) + struct.pack("<IIQ", 9, 8, len(value)) + b"".join(_s(v) for v in value)
    if isinstance(value, list):
        return _s(key) + struct.pack("<IIQ", 9, 4, len(value)) + b"".join(struct.pack("<I", v) for v in value)
    raise TypeError(value)


def write_gguf(path: Path, metadata: dict, tensors: list[tuple[str, tuple[int, ...], int, int]],
               alignment: int = 32) -> None:
    """tensors: (name, dims innermost-first, ggml type id, nbytes)."""
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    out += b"".join(_kv(k, v) for k, v in metadata.items())
    offset = 0
    infos = b""
    for name, dims, ttype, nbytes in tensors:
        infos += _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<IQ", ttype, offset)
        offset += -(-nbytes // alignment) * alignment
    out += infos
    out += b"\0" * (-len(out) % alignment)
    out += b"\0" * offset
    path.write_bytes(out)
