"""Experiment runner for conditions C1, C2, and (open-only) C3.

For C1: feed the model the COMBINED image (sample | divider | test) in one turn.
For C2: feed sample image -> ack -> test image, both turns kept in context.
For C3: hand off to `vwm.local.c3_runner` which uses KV-cache truncation (open models only).

Outputs JSONL with one row per trial:
{
  "trial_id": "...", "set_size": 4, "change": true, "changed_idx": 2, ...
  "model": "gpt-4o-2024-11-20", "provider": "openai", "condition": "C1",
  "raw": "YES", "parsed_yes": true, "parsable": true,
  "correct": true, "latency_s": 1.42, "prompt_tokens": 1100, "completion_tokens": 1,
  "error": null, "timestamp": "..."
}
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from tqdm import tqdm

from ..api import get_client
from ..utils import (
    load_manifest,
    append_jsonl,
    ensure_dir,
    SYSTEM_INSTRUCTION,
    PROMPT_C1_SIMULTANEOUS,
    PROMPT_C2_SEQUENTIAL_TURN1,
    PROMPT_C2_SEQUENTIAL_TURN2,
)
from ..utils.parsing import normalize_response


@dataclass
class ExperimentConfig:
    provider: str                          # "openai" | "anthropic" | "gemini" | (local handled elsewhere)
    condition: str                         # "C1" | "C2"
    manifest_path: Path
    out_path: Path
    n_trials: int | None = None            # limit (smoke test); None = all
    set_sizes: tuple[int, ...] | None = None
    seed: int = 0


def _resolve_paths(trial: dict, manifest_root: Path) -> dict:
    """Resolve manifest image paths regardless of the process's CWD.

    Manifests store repo-root-relative POSIX paths ("data/stimuli/C1/x.png").
    Candidates, in order: as-is (CWD == repo root); anchored at the manifest's
    directory after stripping the shared prefix; anchored at the repo root
    inferred from the manifest location (<root>/data/stimuli/manifest.json).
    Unresolvable paths raise IMMEDIATELY — the old code silently kept a bad
    path and burned ~15s of client retries per trial before erroring.
    """
    # `manifest_root` is the manifest FILE path at the call site — anchor at
    # its directory (cross-model review caught the earlier fix resolving
    # against the file itself: <...>/manifest_C1.json/C1/... never exists).
    root_dir = manifest_root.parent if manifest_root.suffix else manifest_root
    for key in ("sample_path", "test_path", "combined_path"):
        if key not in trial:
            continue
        p = Path(trial[key])
        if p.exists():
            continue
        candidates = []
        try:
            candidates.append(root_dir / p.relative_to("data/stimuli"))
        except ValueError:
            pass
        candidates.append(root_dir.parent.parent / p)
        for alt in candidates:
            if alt.exists():
                trial[key] = str(alt)
                break
        else:
            raise FileNotFoundError(
                f"trial {trial.get('trial_id')!r}: {key}={trial[key]!r} not found "
                f"as-is (CWD={Path.cwd()}) nor under {root_dir} — run from "
                f"the repo root or fix the manifest paths")
    return trial


def _row_template(trial: dict, cfg: ExperimentConfig, model_id: str) -> dict:
    return {
        "trial_id": trial["trial_id"],
        "set_size": trial["set_size"],
        "change": trial["change"],
        "changed_idx": trial["changed_idx"],
        "new_color": trial["new_color"],
        "colors": trial["colors"],
        "positions": trial["positions"],
        "condition": cfg.condition,
        "provider": cfg.provider,
        "model": model_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_condition(cfg: ExperimentConfig, *, dry_run: bool = False, verbose: bool = True) -> dict:
    """Execute one (provider, condition) sweep. Appends to JSONL incrementally."""
    load_dotenv()

    manifest = load_manifest(cfg.manifest_path)
    if manifest["condition"] != cfg.condition:
        raise ValueError(
            f"manifest condition={manifest['condition']!r} != cfg.condition={cfg.condition!r}"
        )
    trials: list[dict] = manifest["trials"]
    if cfg.set_sizes:
        trials = [t for t in trials if t["set_size"] in cfg.set_sizes]
    if cfg.n_trials is not None:
        trials = trials[: cfg.n_trials]

    if dry_run:
        print(f"[DRY] {cfg.provider}:{cfg.condition} would run {len(trials)} trials")
        return {"dry_run": True, "n_trials": len(trials)}

    client = get_client(cfg.provider)
    ensure_dir(cfg.out_path.parent)

    n_ok = 0
    n_err = 0
    t_start = time.monotonic()

    bar = tqdm(trials, disable=not verbose, desc=f"{cfg.provider}/{cfg.condition}")
    for trial in bar:
        trial = _resolve_paths(trial, Path(cfg.manifest_path))
        row = _row_template(trial, cfg, client.model)

        if cfg.condition == "C1":
            resp = client.call_single(
                images=[Path(trial["combined_path"])],
                prompt=PROMPT_C1_SIMULTANEOUS,
                system=SYSTEM_INSTRUCTION,
            )
        elif cfg.condition == "C2":
            resp = client.call_two_turn(
                sample_image=Path(trial["sample_path"]),
                sample_prompt=PROMPT_C2_SEQUENTIAL_TURN1,
                test_image=Path(trial["test_path"]),
                test_prompt=PROMPT_C2_SEQUENTIAL_TURN2,
                system=SYSTEM_INSTRUCTION,
            )
        else:
            raise ValueError(f"runner.py only handles C1/C2 (got {cfg.condition!r}); "
                             f"use vwm.local.c3_runner for C3.")

        norm = normalize_response(resp.raw)
        correct = (norm["parsed_yes"] == trial["change"]) if norm["parsable"] else None
        row.update({
            "raw": resp.raw,
            "parsed_yes": norm["parsed_yes"],
            "parsable": norm["parsable"],
            "correct": correct,
            "latency_s": resp.latency_s,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "error": resp.error,
        })
        append_jsonl(cfg.out_path, row)
        if resp.error:
            n_err += 1
        else:
            n_ok += 1
        bar.set_postfix(ok=n_ok, err=n_err)

    dur = time.monotonic() - t_start
    summary = {
        "provider": cfg.provider,
        "model": client.model,
        "condition": cfg.condition,
        "n_trials": len(trials),
        "n_ok": n_ok,
        "n_err": n_err,
        "err_rate": (n_err / len(trials)) if trials else 0.0,
        "duration_s": dur,
        "out_path": str(cfg.out_path),
    }
    if verbose:
        print(f"[{cfg.provider}/{cfg.condition}] done: "
              f"{n_ok} ok / {n_err} err in {dur:.1f}s -> {cfg.out_path}")
    return summary


# ---------- CLI ----------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single (provider, condition) experiment.")
    p.add_argument("--provider", required=True,
                   choices=["openai", "anthropic", "gemini"])
    p.add_argument("--condition", required=True, choices=["C1", "C2"])
    p.add_argument("--manifest", type=Path, required=True,
                   help="Path to manifest_<condition>.json from vwm.stimuli.generate")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL path (e.g. results/raw/openai_C1.jsonl)")
    p.add_argument("--n-trials", type=int, default=None,
                   help="Limit number of trials (smoke test)")
    p.add_argument("--set-sizes", type=str, default=None,
                   help="Comma-separated set sizes to include, e.g. '1,2,4'")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    set_sizes = None
    if args.set_sizes:
        set_sizes = tuple(int(s) for s in args.set_sizes.split(","))
    cfg = ExperimentConfig(
        provider=args.provider,
        condition=args.condition,
        manifest_path=args.manifest,
        out_path=args.out,
        n_trials=args.n_trials,
        set_sizes=set_sizes,
    )
    summary = run_condition(cfg, dry_run=args.dry_run)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
