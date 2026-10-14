"""Local-model runner for C1 and C2 conditions.

Mirrors `vwm.experiments.runner` but uses the open-weight VLM clients defined
in `vwm.local`. C3 lives in `c3_runner.py` because it has special semantics.
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
    PROMPT_C1_SIMULTANEOUS,
    PROMPT_C2_SEQUENTIAL_TURN1,
    PROMPT_C2_SEQUENTIAL_TURN2,
)
from ..utils.parsing import normalize_response
from .qwen3_vl import Qwen3VLClient
from .qwen25_vl import Qwen25VLClient
from .llava_15 import LLaVA15Client


LOCAL_REGISTRY = {
    "qwen3-vl-4b": Qwen3VLClient,
    "qwen2.5-vl-7b": Qwen25VLClient,
    "llava-1.5-7b": LLaVA15Client,
}


@dataclass
class LocalConfig:
    model_key: str
    condition: str             # "C1" or "C2"
    manifest_path: Path
    out_path: Path
    n_trials: int | None = None
    set_sizes: tuple[int, ...] | None = None


def run_local(cfg: LocalConfig) -> dict:
    manifest = load_manifest(cfg.manifest_path)
    if manifest["condition"] != cfg.condition:
        raise ValueError(f"manifest condition={manifest['condition']!r} != {cfg.condition!r}")

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
    bar = tqdm(trials, desc=f"local/{cfg.model_key}/{cfg.condition}")
    for trial in bar:
        row = {
            "trial_id": trial["trial_id"],
            "set_size": trial["set_size"],
            "change": trial["change"],
            "changed_idx": trial["changed_idx"],
            "new_color": trial["new_color"],
            "colors": trial["colors"],
            "positions": trial["positions"],
            "condition": cfg.condition,
            "provider": "local",
            "model": client.model_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if cfg.condition == "C1":
            resp = client.call_single(
                images=[Path(trial["combined_path"])],
                prompt=PROMPT_C1_SIMULTANEOUS,
                system=SYSTEM_INSTRUCTION,
            )
        else:
            resp = client.call_two_turn(
                sample_image=Path(trial["sample_path"]),
                sample_prompt=PROMPT_C2_SEQUENTIAL_TURN1,
                test_image=Path(trial["test_path"]),
                test_prompt=PROMPT_C2_SEQUENTIAL_TURN2,
                system=SYSTEM_INSTRUCTION,
            )
        norm = normalize_response(resp.raw)
        correct = (norm["parsed_yes"] == trial["change"]) if norm["parsable"] else None
        row.update({
            "raw": resp.raw,
            "parsed_yes": norm["parsed_yes"],
            "parsable": norm["parsable"],
            "correct": correct,
            "latency_s": resp.latency_s,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": resp.error,
        })
        append_jsonl(cfg.out_path, row)
        if resp.error:
            n_err += 1
        else:
            n_ok += 1
        bar.set_postfix(ok=n_ok, err=n_err)

    client.unload()
    dur = time.monotonic() - t0
    print(f"[local/{cfg.model_key}/{cfg.condition}] {n_ok} ok / {n_err} err in {dur:.1f}s")
    return {
        "model_key": cfg.model_key,
        "model": client.model_id,
        "condition": cfg.condition,
        "n_trials": len(trials),
        "n_ok": n_ok,
        "n_err": n_err,
        "duration_s": dur,
        "out_path": str(cfg.out_path),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(LOCAL_REGISTRY))
    p.add_argument("--condition", required=True, choices=["C1", "C2"])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--set-sizes", type=str, default=None)
    args = p.parse_args(argv)
    set_sizes = tuple(int(s) for s in args.set_sizes.split(",")) if args.set_sizes else None
    cfg = LocalConfig(
        model_key=args.model,
        condition=args.condition,
        manifest_path=args.manifest,
        out_path=args.out,
        n_trials=args.n_trials,
        set_sizes=set_sizes,
    )
    print(run_local(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
