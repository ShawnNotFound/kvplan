"""Render results as text tables, JSON, JSON Lines, CSV or Markdown, and write them to files."""

from __future__ import annotations

import csv
import io
import json
import os
import re
from pathlib import Path

FORMATS = ("text", "json", "jsonl", "csv", "markdown")
EXTENSIONS = {"text": ".txt", "json": ".json", "jsonl": ".jsonl", "csv": ".csv", "markdown": ".md"}
_BY_EXTENSION = {".txt": "text", ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl", ".csv": "csv",
                 ".md": "markdown", ".markdown": "markdown"}


def _is_dir_target(path: str) -> bool:
    return os.path.isdir(path) or path.endswith(("/", os.sep))


def choose_format(fmt: str | None, json_flag: bool = False, output: str | None = None) -> str:
    """--format wins, then --json, then the extension of the --output file, then text."""
    if fmt:
        return fmt
    if json_flag:
        return "json"
    if output and not _is_dir_target(output):
        return _BY_EXTENSION.get(Path(output).suffix.lower(), "text")
    return "text"


def _columns(rows: list[dict]) -> list[str]:
    cols: dict[str, None] = {}
    for r in rows:
        cols.update(dict.fromkeys(r))
    return list(cols)


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    return str(v)


def table(rows: list[dict], columns: list[str] | None = None) -> str:
    """A plain aligned text table."""
    cols = columns or _columns(rows)
    cells = [[_cell(r.get(c)) for c in cols] for r in rows]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(cols)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(cols, widths)).rstrip()]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip() for row in cells]
    return "\n".join(lines)


def render(fmt: str, data, rows: list[dict]) -> str:
    """data is the full result (for JSON); rows are flat records (for CSV, Markdown and JSON Lines)."""
    if fmt == "json":
        return json.dumps(data, indent=2, default=str) + "\n"
    if fmt == "jsonl":
        return "".join(json.dumps(r, default=str) + "\n" for r in rows)
    cols = _columns(rows)
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: _cell(r.get(c)) for c in cols})
        return buf.getvalue()
    if fmt == "markdown":
        esc = lambda s: s.replace("|", "\\|").replace("\n", " ")
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(esc(_cell(r.get(c))) for c in cols) + " |" for r in rows]
        return "\n".join(lines) + "\n"
    raise ValueError(f"unknown format '{fmt}'; choose one of {', '.join(FORMATS)}")


def slug(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name.strip("/")).strip("_") or "kvplan"


def write(path: str, body: str, fmt: str, name: str, append: bool = False) -> str:
    """Write body to path, or to <path>/<name>.<ext> when path is a directory. Returns the file written.

    With append, CSV skips its header when the file already has content and JSON Lines just adds
    lines; other formats cannot be appended to.
    """
    target = Path(path) / (slug(name) + EXTENSIONS[fmt]) if _is_dir_target(path) else Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if append and fmt not in ("csv", "jsonl"):
        raise ValueError(f"--append works with csv and jsonl, not {fmt}")
    if append and target.exists() and target.stat().st_size > 0:
        if fmt == "csv":
            header, _, rest = body.partition("\n")
            with target.open(newline="") as f:
                existing = f.readline().rstrip("\r\n")
            if existing != header:
                raise ValueError(f"{target} has different columns ({existing}); write to a new file instead")
            body = rest
        with target.open("a", newline="") as f:
            f.write(body)
    else:
        with target.open("w", newline="") as f:
            f.write(body)
    return str(target)
