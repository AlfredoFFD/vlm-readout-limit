"""C3 test-only (sample-withheld) experiment runner (open models only).

What C3 actually does: it presents ONLY the test image with the C2 turn-2
prompt -- the sample image is never shown to the model. This is the
information-theoretic floor of context ablation: it measures whether
change-detection performance survives at all when the sample is absent from the
active context. Empirically both open Qwens collapse to K ~ 0, i.e. their
apparent C1/C2 capacity is in-context retrieval over the sample tokens, not a
persistent store.

IMPORTANT -- this is NOT a literal KV-cache truncation. This runner does no
cache manipulation: it issues a single `client.call_single([test_image], ...)`
forward pass that simply omits the sample. (And note the open-model clients
re-run a full forward pass per turn, so no KV cache persists across turns for
there to be anything to "truncate".) Withholding the sample is not the same as
encoding the sample and then dropping its cached positions: the latter would
additionally reveal whether sample information is smuggled forward into the
retained acknowledgment tokens. That genuine post-encoding truncation is the
decisive storage-vs-retrieval control, implemented (pending GPU validation) in
`vwm/local/kv_truncation_runner.py`; do not conflate the two.

The companion `tests/test_c3_forget.py` is a construction-time guard that this
code path never passes the sample image to the model -- it documents the
test-only contract and is not evidence for any cache-level interpretation.
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

from ..utils import (
    load_manifest,
    append_jsonl,
    ensure_dir,
    SYSTEM_INSTRUCTION,
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
class C3Config:
    model_key: str
    manifest_path: Path
    out_path: Path
    n_trials: int | None = None
    set_sizes: tuple[int, ...] | None = None


def _row_template(trial: dict, cfg: C3Config, model_id: str) -> dict:
    return {
        "trial_id": trial["trial_id"],
        "set_size": trial["set_size"],
        "change": trial["change"],
        "changed_idx": trial["changed_idx"],
        "new_color": trial["new_color"],
        "colors": trial["colors"],
        "positions": trial["positions"],
        "condition": "C3",
        "provider": "local",
        "model": model_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_c3(cfg: C3Config, client=None) -> dict:
    """Run a C3 sweep on a single local model. Appends to JSONL incrementally.

    C3 == test-only inference: the model never sees the sample (a
    sample-withheld baseline — NOT a cache manipulation; the literal
    post-encoding ablation lives in kv_truncation_runner.py).

    ``client`` is injectable so tests can drive the PRODUCTION loop with a spy
    and assert the sample is never sent (tests/test_c3_forget.py).
    """
    load_dotenv()

    manifest = load_manifest(cfg.manifest_path)
    if manifest["condition"] != "C3":
        raise ValueError(f"Expected C3 manifest, got {manifest['condition']!r}")
    trials: list[dict] = manifest["trials"]
    if cfg.set_sizes:
        trials = [t for t in trials if t["set_size"] in cfg.set_sizes]
    if cfg.n_trials is not None:
        trials = trials[: cfg.n_trials]

    if client is None:
        cls = LOCAL_REGISTRY.get(cfg.model_key.lower())
        if cls is None:
            raise ValueError(f"Unknown local model key {cfg.model_key!r}. "
                             f"Known: {list(LOCAL_REGISTRY)}")
        client = cls()
    ensure_dir(cfg.out_path.parent)

    n_ok = 0
    n_err = 0
    t_start = time.monotonic()
    bar = tqdm(trials, desc=f"local/{cfg.model_key}/C3")
    for trial in bar:
        row = _row_template(trial, cfg, client.model_id)

        # KV-truncated == no sample turn: feed only system + test prompt + test image.
        # This is exactly what would remain if we discarded the KV positions of the
        # sample turn after the ack.
        resp = client.call_single(
            images=[Path(trial["test_path"])],
            prompt=PROMPT_C2_SEQUENTIAL_TURN2,
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
    dur = time.monotonic() - t_start
    summary = {
        "model_key": cfg.model_key,
        "model": client.model_id,
        "condition": "C3",
        "n_trials": len(trials),
        "n_ok": n_ok,
        "n_err": n_err,
        "duration_s": dur,
        "out_path": str(cfg.out_path),
    }
    print(f"[local/{cfg.model_key}/C3] {n_ok} ok / {n_err} err in {dur:.1f}s")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help=f"One of {list(LOCAL_REGISTRY)}")
    p.add_argument("--manifest", type=Path, required=True,
                   help="C3 manifest path (data/stimuli/manifest_C3.json)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--set-sizes", type=str, default=None)
    args = p.parse_args(argv)
    set_sizes = tuple(int(s) for s in args.set_sizes.split(",")) if args.set_sizes else None
    cfg = C3Config(
        model_key=args.model,
        manifest_path=args.manifest,
        out_path=args.out,
        n_trials=args.n_trials,
        set_sizes=set_sizes,
    )
    print(run_c3(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
