"""Load per-trial JSONL result files into a tidy pandas DataFrame."""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from ..utils.io import read_jsonl


def load_results_jsonl(path: str | Path) -> pd.DataFrame:
    rows = list(read_jsonl(path))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Coerce dtypes
    for c in ["set_size", "changed_idx", "prompt_tokens", "completion_tokens"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["change", "parsed_yes", "parsable", "correct"]:
        if c in df.columns:
            df[c] = df[c].astype("boolean")
    return df


def default_results_dir(requested: str | Path = "results/raw") -> Path:
    """The publication source of truth: prefer results/raw_reparsed (rows
    re-parsed with the FIXED parsers, see results/reparse_report.md) whenever
    it exists and `requested` is the plain results/raw default. Prints which
    directory is used so a silent fallback can't misreport 50 CoT rows."""
    req = Path(requested)
    reparsed = req.parent / "raw_reparsed"
    if req.name == "raw" and reparsed.is_dir():
        print(f"[loader] using re-parsed rows: {reparsed} (see reparse_report.md)")
        return reparsed
    return req


def load_all_results(results_dir: str | Path, pattern: str = "*.jsonl",
                     include_smoke: bool = False) -> pd.DataFrame:
    """Concat every JSONL in `results_dir` (recursive) into one DataFrame.

    Smoke files (basename starting with '_smoke') are EXCLUDED by default —
    they share provider/model/condition keys with real runs and previously
    merged silently into published K cells. Duplicate (trial_id, model,
    condition) groups get a loud warning: they indicate a rerun that appended
    onto stale rows (driver output is append-mode).
    """
    import warnings

    paths = sorted(glob.glob(str(Path(results_dir) / "**" / pattern), recursive=True))
    if not include_smoke:
        skipped = [p for p in paths if Path(p).name.startswith("_smoke")]
        if skipped:
            warnings.warn(f"load_all_results: excluding {len(skipped)} smoke file(s); "
                          f"pass include_smoke=True to keep them", stacklevel=2)
        paths = [p for p in paths if not Path(p).name.startswith("_smoke")]
    frames = [load_results_jsonl(p) for p in paths]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    key_cols = [c for c in ("trial_id", "model", "condition") if c in df.columns]
    if len(key_cols) == 3:
        dup = df.duplicated(subset=key_cols).sum()
        if dup:
            warnings.warn(f"load_all_results: {dup} duplicate (trial_id, model, "
                          f"condition) rows — a rerun appended onto stale output; "
                          f"downstream K/figures will double-count until deduped",
                          stacklevel=2)
    return df
