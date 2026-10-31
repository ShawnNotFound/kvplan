"""Colours for terminal output.

Colour is used only when writing to a terminal, never in files, pipes or the JSON/CSV/Markdown
formats. `--color never` or the NO_COLOR environment variable (https://no-color.org) turns it off;
`--color always` or FORCE_COLOR turns it on even when the output is not a terminal.
"""

from __future__ import annotations

import os

MODES = ("auto", "always", "never")
_CODES = {"red": "31", "green": "32", "yellow": "33", "cyan": "36", "bold": "1", "dim": "2"}
_mode = "auto"


def set_mode(mode: str) -> None:
    global _mode
    if mode not in MODES:
        raise ValueError(f"--color must be one of {', '.join(MODES)}, not '{mode}'")
    _mode = mode


def enabled(stream) -> bool:
    """Whether to colour text written to stream."""
    if _mode == "never":
        return False
    if _mode == "always":
        return True
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty() and os.environ.get("TERM") != "dumb"


def paint(text: str, *styles: str, on: bool = True) -> str:
    if not on or not styles or not text:
        return text
    return "\033[" + ";".join(_CODES[s] for s in styles) + "m" + text + "\033[0m"


def verdict_color(starts) -> str:
    """green: vLLM should start; red: it will not; yellow: start not confirmed or an error."""
    return {True: "green", False: "red"}.get(starts, "yellow")


SEVERITY_COLOR = {"error": "red", "warning": "yellow", "info": "dim"}
