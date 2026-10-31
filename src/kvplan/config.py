"""Config files with default options, and model lists for checking many models at once.

A config file is TOML (Python 3.11+) or JSON. Top-level keys apply to every command that has the
option; a table named after a command applies to that command only:

    gpu = "A6000"
    format = "csv"
    output = "results/"

    [check]
    max_model_len = 8192
    tp = 2

    [recommend]
    task = "code"
    priority = "speed"

Keys are the long option names, with dashes or underscores. Options on the command line win.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path

ALIASES = {"tp": "tensor_parallel_size"}
CONFIG_ENV = "KVPLAN_CONFIG"


def default_config_path() -> Path | None:
    if os.environ.get(CONFIG_ENV):
        return Path(os.environ[CONFIG_ENV]).expanduser()
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "kvplan"
    for name in ("config.toml", "config.json"):
        if (base / name).is_file():
            return base / name
    return None


def _parse(path: Path, text: str) -> dict:
    if path.suffix.lower() == ".toml":
        try:
            import tomllib
        except ImportError:
            raise ValueError(f"{path}: reading TOML needs Python 3.11 or newer; use a JSON config file") from None
        try:
            return tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"{path} is not valid TOML: {e}") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold an object of option names and values")
    return data


def load_config(path: str | None) -> dict:
    """The settings in path, or in the default config file if there is one; {} when there is none."""
    p = Path(path).expanduser() if path else default_config_path()
    if p is None:
        return {}
    if not p.is_file():
        if path or os.environ.get(CONFIG_ENV):
            raise ValueError(f"config file {p} not found")
        return {}
    return _parse(p, p.read_text())


def _key(name: str) -> str:
    name = name.replace("-", "_")
    return ALIASES.get(name, name)


def apply(parser: argparse.ArgumentParser, command: str, settings: dict) -> None:
    """Make the settings for this command the parser's defaults."""
    actions = {a.dest: a for a in parser._actions if a.dest not in ("help", "model", "query", argparse.SUPPRESS)}
    commands = {"check", "inspect", "search", "download", "login", "recommend"}
    shared = {_key(k): v for k, v in settings.items() if k not in commands and not isinstance(v, dict)}
    own = settings.get(command) or {}
    if not isinstance(own, dict):
        raise ValueError(f"config: [{command}] must be a table of options")
    own = {_key(k): v for k, v in own.items()}
    unknown = sorted(k for k in own if k not in actions)
    if unknown:
        raise ValueError(f"config: [{command}] has unknown option(s) {', '.join(unknown)}; "
                         f"valid: {', '.join(sorted(actions))}")
    defaults = {k: v for k, v in shared.items() if k in actions} | own
    for k, v in defaults.items():
        a = actions[k]
        if a.choices is not None and v not in a.choices:
            raise ValueError(f"config: {k} = {v!r} is not one of {', '.join(map(str, a.choices))}")
    parser.set_defaults(**defaults)


# Options a model list may set per model, with their types.
_ENTRY_TYPES = {"gpu": str, "gpu_mem_gib": float, "gpu_index": str, "tp": int, "max_model_len": int,
                "max_num_seqs": int, "max_num_batched_tokens": int, "gpu_memory_utilization": float,
                "kv_cache_dtype": str, "block_size": int, "enforce_eager": bool, "other_gpu_mem_gib": float,
                "overhead_gib": float, "revision": str, "gguf": str, "skip_mm_profiling": bool}


def _convert(key: str, value, where: str):
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    t = _ENTRY_TYPES[key]
    try:
        if t is bool:
            if isinstance(value, bool):
                return value
            if str(value).lower() in ("1", "true", "yes", "y"):
                return True
            if str(value).lower() in ("0", "false", "no", "n"):
                return False
            raise ValueError
        if t is int and isinstance(value, float) and value.is_integer():
            value = int(value)
        return t(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: {key} = {value!r} is not a valid {t.__name__}") from None


def _entry(item, where: str) -> dict:
    if isinstance(item, str):
        item = {"model": item}
    if not isinstance(item, dict) or not str(item.get("model") or "").strip():
        raise ValueError(f"{where}: each entry needs a model")
    entry = {"model": str(item["model"]).strip()}
    for k, v in item.items():
        if k == "model":
            continue
        key = k.strip().replace("-", "_")
        if key == "tensor_parallel_size":
            key = "tp"
        if key not in _ENTRY_TYPES:
            raise ValueError(f"{where}: unknown option '{k}'; use model and any of {', '.join(_ENTRY_TYPES)}")
        v = _convert(key, v, where)
        if v is not None:
            entry[key] = v
    return entry


def read_model_list(path: str) -> list[dict]:
    """Models to check, each a dict with 'model' and optional per-model options.

    A .json file holds a list of names or objects; a .csv file has a 'model' column and option
    columns; anything else (or '-' for standard input) has one model per line, # for comments.
    """
    if path == "-":
        text, suffix, name = sys.stdin.read(), "", "standard input"
    else:
        p = Path(path).expanduser()
        if not p.is_file():
            raise ValueError(f"model list {p} not found")
        text, suffix, name = p.read_text(), p.suffix.lower(), str(p)
    stripped = text.lstrip()
    if suffix == ".json" or (suffix == "" and stripped.startswith("[")):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"{name} is not valid JSON: {e}") from None
        if not isinstance(data, list):
            raise ValueError(f"{name} must hold a list of models")
        entries = [_entry(x, f"{name} item {i}") for i, x in enumerate(data, 1)]
    elif suffix == ".csv" or (suffix == "" and stripped.lower().startswith("model,")):
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames or "model" not in [f.strip() for f in reader.fieldnames]:
            raise ValueError(f"{name}: a CSV model list needs a 'model' column")
        entries = [_entry({k.strip(): v for k, v in row.items() if k}, f"{name} line {i}")
                   for i, row in enumerate(reader, 2) if (row.get("model") or "").strip()]
    else:
        entries = [{"model": line.split("#", 1)[0].strip()} for line in text.splitlines()
                   if line.split("#", 1)[0].strip()]
    if not entries:
        raise ValueError(f"{name} lists no models")
    return entries
