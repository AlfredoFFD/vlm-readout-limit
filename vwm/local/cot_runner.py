"""Chain-of-thought variants of C1/C2/C3 (Remi Megret's "reasoning step" bottleneck).

The standard conditions force a single YES/NO token (temperature 0, max_new_tokens=8).
This runner instead lets the model REASON before answering: it appends a
think-step-by-step instruction, raises the token budget, and parses the final
"Answer: YES/NO".

Scientific question: does chain-of-thought act as an external memory scratchpad that
lets the model route around the visual-working-memory bottleneck? The diagnostic
contrasts are:
- C2-CoT vs C2  -> does reasoning raise capacity when the sample is still in context?
- C3-CoT vs C3  -> with the sample absent, reasoning has nothing to recover, so C3-CoT
  must still collapse to chance. If it does not, the model is leaking sample info.

Output: a JSONL with condition tagged "<C>-CoT" and the full reasoning kept in `raw`.
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
from ..utils.prompts import (
    PROMPT_C1_SIMULTANEOUS_COT,
    PROMPT_C2_SEQUENTIAL_TURN1,
    PROMPT_C2_SEQUENTIAL_TURN2_COT,
)
from ..utils.parsing import parse_final_yes_no
from .qwen3_vl import Qwen3VLClient
from .qwen25_vl import Qwen25VLClient
from .llava_15 import LLaVA15Client


LOCAL_REGISTRY = {
    "qwen3-vl-4b": Qwen3VLClient,
    "qwen2.5-vl-7b": Qwen25VLClient,
    "llava-1.5-7b": LLaVA15Client,
}

COT_MAX_NEW_TOKENS = 256


@dataclass
class CoTConfig:
    model_key: str
    condition: str             # "C1" | "C2" | "C3"
    manifest_path: Path
    out_path: Path
    n_trials: int | None = None
    set_sizes: tuple[int, ...] | None = None
    max_new_tokens: int = COT_MAX_NEW_TOKENS


def run_cot(cfg: CoTConfig) -> dict:
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
    n_unparsable = 0
    t0 = time.monotonic()
    bar = tqdm(trials, desc=f"cot/{cfg.model_key}/{cfg.condition}")
    for trial in bar:
        row = {
            "trial_id": trial["trial_id"],
            "set_size": trial["set_size"],
            "change": trial["change"],
            "changed_idx": trial["changed_idx"],
            "new_color": trial["new_color"],
            "colors": trial["colors"],
            "positions": trial["positions"],
            "condition": f"{cfg.condition}-CoT",
            "mode": "cot",
            "provider": "local",
            "model": client.model_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if cfg.condition == "C1":
            resp = client.call_single(
                images=[Path(trial["combined_path"])],
                prompt=PROMPT_C1_SIMULTANEOUS_COT,
                system=SYSTEM_INSTRUCTION,
                max_new_tokens=cfg.max_new_tokens,
            )
        elif cfg.condition == "C2":
            resp = client.call_two_turn(
                sample_image=Path(trial["sample_path"]),
                sample_prompt=PROMPT_C2_SEQUENTIAL_TURN1,
                test_image=Path(trial["test_path"]),
                test_prompt=PROMPT_C2_SEQUENTIAL_TURN2_COT,
                system=SYSTEM_INSTRUCTION,
                max_new_tokens=cfg.max_new_tokens,
            )
        else:  # C3: test-only (sample withheld) + CoT
            # INTENTIONAL: we reuse the C2-turn2 prompt (which says "the SAMPLE array I
            # showed you earlier") on a test-only call, exactly as c3_runner.py does. The
            # point of C3 is "C2 minus the sample tokens", so the prompt must be identical;
            # the absent sample IS the KV-truncation manipulation. With CoT the model may
            # reason about a sample it never saw -> watch n_unparsable for this condition.
            resp = client.call_single(
                images=[Path(trial["test_path"])],
                prompt=PROMPT_C2_SEQUENTIAL_TURN2_COT,
                system=SYSTEM_INSTRUCTION,
                max_new_tokens=cfg.max_new_tokens,
            )

        parsed = parse_final_yes_no(resp.raw)
        parsable = parsed is not None
        correct = (parsed == trial["change"]) if parsable else None
        row.update({
            "raw": resp.raw,
            "parsed_yes": parsed,
            "parsable": parsable,
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
            if not parsable:
                n_unparsable += 1
        bar.set_postfix(ok=n_ok, err=n_err, unparsable=n_unparsable)

    client.unload()
    dur = time.monotonic() - t0
    print(f"[cot/{cfg.model_key}/{cfg.condition}] {n_ok} ok / {n_err} err / "
          f"{n_unparsable} unparsable in {dur:.1f}s")
    return {
        "model_key": cfg.model_key,
        "model": client.model_id,
        "condition": f"{cfg.condition}-CoT",
        "n_trials": len(trials),
        "n_ok": n_ok,
        "n_err": n_err,
        "n_unparsable": n_unparsable,
        "duration_s": dur,
        "out_path": str(cfg.out_path),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Chain-of-thought C1/C2/C3 runner (local VLMs).")
    p.add_argument("--model", required=True, choices=list(LOCAL_REGISTRY))
    p.add_argument("--condition", required=True, choices=["C1", "C2", "C3"])
    p.add_argument("--manifest", type=Path, required=True,
                   help="Manifest matching --condition (data/stimuli/manifest_<C>.json)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--set-sizes", type=str, default=None)
    p.add_argument("--max-new-tokens", type=int, default=COT_MAX_NEW_TOKENS)
    args = p.parse_args(argv)
    set_sizes = tuple(int(s) for s in args.set_sizes.split(",")) if args.set_sizes else None
    cfg = CoTConfig(
        model_key=args.model,
        condition=args.condition,
        manifest_path=args.manifest,
        out_path=args.out,
        n_trials=args.n_trials,
        set_sizes=set_sizes,
        max_new_tokens=args.max_new_tokens,
    )
    print(run_cot(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
