"""Comprehension control for proprietary VLMs (OpenAI, Anthropic, Gemini).

Mirror of `vwm/local/comprehension_runner.py` but uses the proprietary API clients via
`vwm.api.get_client`. Output JSONL is schema-identical to the local version so the
downstream count-accuracy gate analysis works on both.

Defeats the "task-misunderstanding" critique (arXiv:2412.18120) for the frontier cells
by verifying the model perceives the array before its capacity is scored.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from ..api import get_client
from ..utils import load_manifest, append_jsonl, ensure_dir, SYSTEM_INSTRUCTION
from ..utils.prompts import PROMPT_COMPREHENSION_COUNT
from ..utils.parsing import parse_count


PROVIDERS = ("openai", "anthropic", "gemini")


def _sample_path(trial: dict) -> str:
    """Use sample or test image (each a single array of N squares); refuse combined."""
    for key in ("sample_path", "test_path"):
        if trial.get(key):
            return trial[key]
    raise ValueError(
        f"trial {trial.get('trial_id')!r}: needs sample_path or test_path; "
        "combined_path holds two arrays and would double the count."
    )


@dataclass
class ComprehensionAPIConfig:
    provider: str
    manifest_path: Path
    out_path: Path
    n_trials: int | None = None
    set_sizes: tuple[int, ...] | None = None
    max_tokens: int = 16


def run_comprehension_api(cfg: ComprehensionAPIConfig) -> dict:
    load_dotenv()
    if cfg.provider not in PROVIDERS:
        raise ValueError(f"unknown provider {cfg.provider!r}; expected one of {PROVIDERS}")

    manifest = load_manifest(cfg.manifest_path)
    trials = manifest["trials"]
    if cfg.set_sizes:
        trials = [t for t in trials if t["set_size"] in cfg.set_sizes]
    if cfg.n_trials is not None:
        trials = trials[: cfg.n_trials]

    client = get_client(cfg.provider)
    ensure_dir(cfg.out_path.parent)

    n_ok = 0
    n_err = 0
    t0 = time.monotonic()
    bar = tqdm(trials, desc=f"comp-api/{cfg.provider}")
    for trial in bar:
        resp = client.call_single(
            images=[Path(_sample_path(trial))],
            prompt=PROMPT_COMPREHENSION_COUNT,
            system=SYSTEM_INSTRUCTION,
            max_tokens=cfg.max_tokens,
        )
        pred = parse_count(resp.raw)
        correct = (pred == trial["set_size"]) if pred is not None else None
        row = {
            "trial_id": trial["trial_id"],
            "set_size": trial["set_size"],
            "condition": "comprehension",
            "mode": "count",
            "provider": cfg.provider,
            "model": client.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw": resp.raw,
            "predicted_count": pred,
            "parsable": pred is not None,
            "correct": correct,
            "latency_s": resp.latency_s,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "error": resp.error,
        }
        append_jsonl(cfg.out_path, row)
        if resp.error:
            n_err += 1
        else:
            n_ok += 1
        bar.set_postfix(ok=n_ok, err=n_err)

    dur = time.monotonic() - t0
    print(f"[comp-api/{cfg.provider}] {n_ok} ok / {n_err} err in {dur:.1f}s -> {cfg.out_path}")
    return {
        "provider": cfg.provider,
        "model": client.model,
        "n_trials": len(trials),
        "n_ok": n_ok,
        "n_err": n_err,
        "duration_s": dur,
        "out_path": str(cfg.out_path),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Comprehension probe for proprietary VLMs.")
    p.add_argument("--provider", required=True, choices=PROVIDERS)
    p.add_argument("--manifest", type=Path, required=True,
                   help="Any manifest with sample images (e.g. data/stimuli/manifest_C2.json)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--set-sizes", type=str, default=None)
    p.add_argument("--max-tokens", type=int, default=16)
    args = p.parse_args(argv)
    set_sizes = tuple(int(s) for s in args.set_sizes.split(",")) if args.set_sizes else None
    cfg = ComprehensionAPIConfig(
        provider=args.provider,
        manifest_path=args.manifest,
        out_path=args.out,
        n_trials=args.n_trials,
        set_sizes=set_sizes,
        max_tokens=args.max_tokens,
    )
    print(run_comprehension_api(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
