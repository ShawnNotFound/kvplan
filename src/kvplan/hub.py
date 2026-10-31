"""Hugging Face Hub helpers beyond reading one model: search, model lists, login and download.

Everything goes through the public HTTP API (https://huggingface.co/docs/hub/api), so kvplan
keeps no dependencies. `download` uses huggingface_hub when it is installed, so the files land
in the cache vLLM reads; without it, kvplan downloads the files itself into a directory.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .sources import DEFAULT_ENDPOINT, USER_AGENT, GatedRepoError, SourceError, find_token, token_path

# Files vLLM reads besides the weights: configs, tokenizers, chat templates, processors, remote code.
SUPPORT_FILE_PATTERNS = ("*.json", "*.jinja", "*.model", "*.txt", "*.py", "*.tiktoken")


@dataclass
class HubModel:
    """One entry of the Hub's model list, with the fields kvplan uses."""
    id: str
    downloads: int = 0
    likes: int = 0
    gated: bool = False
    pipeline_tag: str | None = None
    architecture: str | None = None
    model_type: str | None = None
    parameters: dict[str, int] = field(default_factory=dict)   # safetensors parameter count per dtype
    tags: list[str] = field(default_factory=list)

    @property
    def total_parameters(self) -> int:
        return sum(self.parameters.values())

    @property
    def checkpoint_bytes(self) -> int:
        """Bytes of the safetensors weights, from the parameter count per dtype."""
        size = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "U32": 4, "BF16": 2, "F16": 2, "I16": 2,
                "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "F4": 0.5}
        return int(sum(n * size.get(d, 2) for d, n in self.parameters.items()))

    @classmethod
    def from_api(cls, d: dict) -> HubModel:
        cfg = d.get("config") or {}
        archs = cfg.get("architectures")
        st = d.get("safetensors") or {}
        return cls(id=d["id"], downloads=d.get("downloads") or 0, likes=d.get("likes") or 0,
                   gated=bool(d.get("gated")), pipeline_tag=d.get("pipeline_tag"),
                   architecture=archs[0] if isinstance(archs, list) and archs else None,
                   model_type=cfg.get("model_type"), parameters=dict(st.get("parameters") or {}),
                   tags=list(d.get("tags") or []))


def _get_json(url: str, token: str | None = None, timeout: float = 30.0):
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise SourceError("Hugging Face rejected the token (401); check HF_TOKEN or run `kvplan login` "
                              "again") from e
        raise SourceError(f"Hugging Face API: HTTP {e.code} for {url}") from e
    except urllib.error.URLError as e:
        raise SourceError(f"cannot reach Hugging Face ({e.reason})") from e


_EXPAND = ("downloads", "likes", "gated", "pipeline_tag", "config", "safetensors", "tags")


def list_models(*, search: str | None = None, pipeline_tag: str | None = None, sort: str = "downloads",
                limit: int = 50, token: str | None = None, endpoint: str = DEFAULT_ENDPOINT) -> list[HubModel]:
    """Models from the Hub's model list, most downloaded first."""
    params = [("sort", sort), ("direction", "-1"), ("limit", str(limit))]
    if search:
        params.append(("search", search))
    if pipeline_tag:
        params.append(("pipeline_tag", pipeline_tag))
    params += [("expand[]", e) for e in _EXPAND]
    url = f"{endpoint}/api/models?{urllib.parse.urlencode(params)}"
    return [HubModel.from_api(d) for d in _get_json(url, token or find_token())]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def search_models(query: str, limit: int = 8, token: str | None = None,
                  endpoint: str = DEFAULT_ENDPOINT) -> list[HubModel]:
    """Repos that match a loose name such as 'llama 3.1 8b instruct' or 'Qwen2.5-7B', most downloaded first.

    The Hub searches for the query as a substring of the repo id, so a query with spaces or a
    different separator finds nothing. kvplan searches for the longest word instead and keeps the
    repos whose id contains every word, ignoring case and punctuation.
    """
    words = [w for w in re.split(r"[\s/]+", query.strip()) if w]
    if not words:
        return []
    queries = list(dict.fromkeys([query.strip() if len(words) == 1 else "-".join(words), max(words, key=len)]))
    keys = [_norm(w) for w in words]
    found: dict[str, HubModel] = {}
    for q in queries:
        for m in list_models(search=q, limit=100, token=token, endpoint=endpoint):
            if all(k in _norm(m.id) for k in keys):
                found.setdefault(m.id, m)
        if len(found) >= limit:
            break
    return sorted(found.values(), key=lambda m: -m.downloads)[:limit]


def similar_models(name: str, limit: int = 8, token: str | None = None,
                   endpoint: str = DEFAULT_ENDPOINT) -> list[HubModel]:
    """Popular repos whose name is close to a misspelled or partial one ('Qwen2.5-7B-Instrct')."""
    import difflib
    owner, _, repo = name.rpartition("/")
    pool: dict[str, HubModel] = {m.id: m for m in search_models(repo, limit=limit, token=token, endpoint=endpoint)}
    parts = re.split(r"(?<=[^-_.\s])(?=[-_.\s])", repo)
    # Search for ever shorter prefixes of the name ("Qwen2.5-7B", then "Qwen2.5") to find the family.
    for k in range(len(parts) - 1, 0, -1):
        if len(pool) >= 60:
            break
        prefix = "".join(parts[:k]).strip("-_. ")
        if len(prefix) >= 3:
            for m in list_models(search=prefix, limit=50, token=token, endpoint=endpoint):
                pool.setdefault(m.id, m)
    target = _norm(repo)

    def score(m: HubModel) -> float:
        mo, _, mr = m.id.partition("/")
        s = difflib.SequenceMatcher(None, target, _norm(mr)).ratio()
        return s + (0.1 if owner and mo.lower() == owner.lower() else 0)

    ranked = [(score(m), m) for m in pool.values()]
    ranked = [(s, m) for s, m in ranked if s >= 0.6]
    ranked.sort(key=lambda sm: (-round(sm[0], 1), -sm[1].downloads))
    return [m for _, m in ranked[:limit]]


def resolve_name(name: str, token: str | None = None) -> tuple[str | None, list[HubModel]]:
    """For a name without an owner ('Qwen2.5-7B-Instruct'), the most downloaded repo with exactly
    that name, and the other matches. (None, matches) when no repo has that exact name."""
    matches = search_models(name, limit=8, token=token)
    exact = [m for m in matches if "/" not in name and m.id.split("/", 1)[1].lower() == name.lower()]
    if exact:
        return exact[0].id, matches
    return None, matches or similar_models(name, token=token)


def whoami(token: str, endpoint: str = DEFAULT_ENDPOINT) -> str:
    """The user name a token belongs to; raises SourceError for a bad token."""
    try:
        info = _get_json(f"{endpoint}/api/whoami-v2", token)
    except SourceError as e:
        if "401" in str(e):
            raise SourceError("Hugging Face rejected this token (401); copy it again from "
                              "https://huggingface.co/settings/tokens") from e
        raise
    return info.get("name") or "unknown"


def save_token(token: str) -> Path:
    """Store a token where huggingface_hub (and so vLLM and huggingface-cli) also look for it."""
    path = token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token.strip())
    os.chmod(path, 0o600)
    return path


def download(repo_id: str, weight_files: list[str], all_files: list[str], *, revision: str = "main",
             token: str | None = None, local_dir: str | None = None, endpoint: str = DEFAULT_ENDPOINT,
             log=print) -> str:
    """Download the weight files kvplan sized plus the configs and tokenizer vLLM needs.

    Returns where the files are: the Hugging Face cache snapshot, which `vllm serve REPO_ID` reads,
    or the directory.
    """
    import fnmatch
    support = [f for f in all_files if "/" not in f and any(fnmatch.fnmatch(f, p) for p in SUPPORT_FILE_PATTERNS)]
    wanted = sorted(set(weight_files) | set(support))
    token = token or find_token()
    try:
        import huggingface_hub
    except ImportError:
        huggingface_hub = None
    if huggingface_hub is not None:
        log(f"downloading {len(wanted)} files with huggingface_hub"
            + (f" into {local_dir}" if local_dir else " into the Hugging Face cache"))
        try:
            path = huggingface_hub.snapshot_download(repo_id, revision=revision, allow_patterns=wanted,
                                                     token=token or None, local_dir=local_dir)
        except Exception as e:  # huggingface_hub raises many error types; report them the same way
            raise SourceError(f"download of {repo_id} failed: {e}") from e
        return str(local_dir or path)
    dest = Path(local_dir or repo_id.split("/", 1)[1])
    log(f"downloading {len(wanted)} files into {dest} (install huggingface_hub to use the shared cache)")
    for name in wanted:
        _fetch(f"{endpoint}/{repo_id}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(name)}",
               dest / name, token, repo_id, log)
    return str(dest)


def _fetch(url: str, dest: Path, token: str | None, repo_id: str, log) -> None:
    """Stream one file to dest, resuming a partial download left in dest.part."""
    from .sources import _opener
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        resp = _opener.open(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:          # the partial file is already complete
            part.rename(dest)
            return
        if e.code in (401, 403):
            raise GatedRepoError(f"{repo_id} is gated or private ({e.code}): run `kvplan login` or set "
                                 "HF_TOKEN, and accept the model's licence on its Hugging Face page") from e
        raise SourceError(f"{repo_id}: HTTP {e.code} downloading {dest.name}") from e
    except urllib.error.URLError as e:
        raise SourceError(f"cannot reach Hugging Face ({e.reason})") from e
    with resp:
        if have and resp.status != 206:
            have = 0               # the server ignored the range; start again
        total = resp.headers.get("Content-Length")
        total = int(total) + have if total else None
        done, last = have, -1
        with part.open("ab" if have else "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and sys.stderr.isatty():
                    pct = done * 100 // total
                    if pct != last:
                        print(f"\r  {dest.name}: {pct}% of {total / 2**30:.2f} GiB", end="", file=sys.stderr)
                        last = pct
        if total and sys.stderr.isatty():
            print(file=sys.stderr)
    part.rename(dest)
    log(f"  {dest.name}")
