"""Comprehension control (defeats the "task-misunderstanding" critique).

Skeptics argue an LLM's apparent capacity limit may just be that it does not
perceive or understand the array, not that it cannot remember it (cf. Ivanova et
al., "Do Language Models Understand the Cognitive Tasks Given to Them?",
arXiv:2412.18120). This probe verifies perception WITHOUT any memory demand: it
shows the SAMPLE array and asks the model to COUNT the squares.

Interpretation: a model that cannot count N squares cannot be meaningfully scored
for *remembering* N squares. We gate the capacity (K) estimates on per-set-size
comprehension accuracy — a C3 collapse at N=8 only supports the memory claim if
the model demonstrably perceives 8 squares here.

Output JSONL: per trial, predicted_count vs set_size, condition="comprehension".
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from ..utils import (
    load_manifest,
    append_jsonl,
    ensure_dir,
    SYSTEM_INSTRUCTION,
)
from ..utils.prompts import PROMPT_COMPREHENSION_COUNT
from ..utils.parsing import parse_count
from .qwen3_vl import Qwen3VLClient
from .qwen25_vl import Qwen25VLClient
from .llava_15 import LLaVA15Client


LOCAL_REGISTRY = {
    "qwen3-vl-4b": Qwen3VLClient,
    "qwen2.5-vl-7b": Qwen25VLClient,
    "llava-1.5-7b": LLaVA15Client,
}


@dataclass
class ComprehensionConfig:
    model_key: str
    manifest_path: Path
    out_path: Path
    n_trials: int | None = None
    set_sizes: tuple[int, ...] | None = None
    max_new_tokens: int = 16


def _sample_path(trial: dict) -> str:
    """The single array whose perception we test.

    `sample_path` and `test_path` are each single arrays of N squares (either is a
    valid count probe). `combined_path` holds TWO arrays side-by-side (2N squares),
    so using it would double the count and corrupt the comprehension gate -- refuse it.
    """
    for key in ("sample_path", "test_path"):
        if trial.get(key):
            return trial[key]
    raise ValueError(
        f"trial {trial.get('trial_id')!r}: comprehension needs a single-array image "
        "(sample_path or test_path). Pass a C2/C3 manifest, not a C1 (combined) manifest."
    )


def run_comprehension(cfg: ComprehensionConfig) -> dict:
    manifest = load_manifest(cfg.manifest_path)
    trials = manifest["trials"]
    if cfg.set_sizes:
        trials = [t for t in trials if t["set_size"] in cfg.set_sizes]
    if cfg.n_trials is not None:
        trials = trials[: cfg.n_trials]

    cls = LOCAL_REGISTRY.get(cfg.model_key.lower())
    if cls is None:
        raise ValueError(f"Unknown local model key {cfg.model_key!r}")
    client = cls()
    ensure_dir(cfg.out_path.parent)

    n_ok = 0
    n_err = 0
    t0 = time.monotonic()
    bar = tqdm(trials, desc=f"comp/{cfg.model_key}")
    for trial in bar:
        resp = client.call_single(
            images=[Path(_sample_path(trial))],
            prompt=PROMPT_COMPREHENSION_COUNT,
            system=SYSTEM_INSTRUCTION,
            max_new_tokens=cfg.max_new_tokens,
        )
        pred = parse_count(resp.raw)
        correct = (pred == trial["set_size"]) if pred is not None else None
        row = {
            "trial_id": trial["trial_id"],
            "set_size": trial["set_size"],
            "condition": "comprehension",
            "mode": "count",
            "provider": "local",
            "model": client.model_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw": resp.raw,
            "predicted_count": pred,
            "parsable": pred is not None,
            "correct": correct,
            "latency_s": resp.latency_s,
            "error": resp.error,
        }
        append_jsonl(cfg.out_path, row)
        if resp.error:
            n_err += 1
        else:
            n_ok += 1
        bar.set_postfix(ok=n_ok, err=n_err)

    client.unload()
    dur = time.monotonic() - t0
    print(f"[comp/{cfg.model_key}] {n_ok} ok / {n_err} err in {dur:.1f}s")
    return {
        "model_key": cfg.model_key,
        "model": client.model_id,
        "condition": "comprehension",
        "n_trials": len(trials),
        "n_ok": n_ok,
        "n_err": n_err,
        "duration_s": dur,
        "out_path": str(cfg.out_path),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Comprehension control (count-the-squares probe).")
    p.add_argument("--model", required=True, choices=list(LOCAL_REGISTRY))
    p.add_argument("--manifest", type=Path, required=True,
                   help="Any manifest with sample images (e.g. data/stimuli/manifest_C2.json)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--set-sizes", type=str, default=None)
    p.add_argument("--max-new-tokens", type=int, default=16)
    args = p.parse_args(argv)
    set_sizes = tuple(int(s) for s in args.set_sizes.split(",")) if args.set_sizes else None
    cfg = ComprehensionConfig(
        model_key=args.model,
        manifest_path=args.manifest,
        out_path=args.out,
        n_trials=args.n_trials,
        set_sizes=set_sizes,
        max_new_tokens=args.max_new_tokens,
    )
    print(run_comprehension(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
