"""Shared I/O helpers."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path, strict: bool = False) -> Iterator[dict]:
    """Yield one dict per JSONL line.

    A truncated FINAL line (interrupted run: kill/OOM mid-append) is skipped
    with a warning instead of crashing every downstream consumer — runners
    append one line per trial, so Ctrl+C routinely creates exactly that state.
    A malformed line that is NOT the final one still raises (that's corruption,
    not truncation), as does any malformed line under strict=True.
    """
    import warnings

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    last_idx = len(lines) - 1
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if strict or i != last_idx:
                raise
            warnings.warn(f"read_jsonl: skipping truncated final line of {path} "
                          f"({line[:60]!r}...) — interrupted run tail", stacklevel=2)


def encode_image_b64(path: str | Path) -> str:
    """Base64-encoded image bytes (no data: prefix)."""
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("ascii")
