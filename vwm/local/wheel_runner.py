"""Runner for the CIELAB-wheel design (manifest schema v2, design="wheel").

Two tasks share the same stimuli:
- ``change`` (M4): sequential YES/NO change detection. If the client exposes
  ``two_turn_yes_prob`` it also records a graded P(YES) so a curved SDT ROC can be
  swept downstream; otherwise it records only the parsed YES/NO.
- ``report`` (M3): continuous report. Turn 1 shows the sample; turn 2 shows the
  color-free probe (cued square + 12-anchor legend ring); the model answers a legend
  index 1..M which is scored as circular error on the wheel.

The legacy ``local_runner.py`` reads palette-only trial fields (`new_color`,`colors`),
so the wheel needs this separate path. The model client is dependency-injected so the
whole pipeline is CPU-testable with a fake (see tests/test_wheel_runner.py).
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
    PROMPT_WHEEL_REPORT_TURN1,
    PROMPT_WHEEL_REPORT_TURN2,
)
from ..utils.parsing import parse_yes_no, parse_report_index
from ..analysis.roc import roc_auc
from ..stimuli.generate import score_report, WHEEL_REPORT_M

# "simultaneous" (2026-08-07 assembly review): same trials via the already-
# rendered combined sample|test image, both arrays visible -> a PERCEPTION
# baseline with no memory demand. Tier-2 memory claims are made on the
# sequential-vs-simultaneous contrast (or gated on near-ceiling simultaneous
# performance per delta-by-N cell) — a set-size-dependent encoder bottleneck
# would otherwise mimic the resource-rational d'(N) decline.
TASKS = ("change", "report", "simultaneous")


@dataclass
class WheelRunConfig:
    model_key: str
    task: str                          # "change" | "report"
    manifest_path: Path
    out_path: Path
    n_trials: int | None = None
    set_sizes: tuple[int, ...] | None = None
    deltas: tuple[float, ...] | None = None
    dtype: str = "bfloat16"
    prompt_id: int = 0                 # index into CHANGE_PROMPT_VARIANTS
                                       # (v1 = binding-fair, REQUIRED for swap manifests)


def _resolve_client(model_key: str, dtype: str = "bfloat16"):
    # Imported lazily so importing this module does not pull in torch/transformers.
    from .local_runner import LOCAL_REGISTRY
    cls = LOCAL_REGISTRY.get(model_key.lower())
    if cls is None:
        raise ValueError(f"Unknown local model key {model_key!r}")
    return cls(dtype=dtype)


def _base_row(tr: dict, client, task: str, condition_tag: str | None = None) -> dict:
    return {
        "trial_id": tr["trial_id"],
        "set_size": tr["set_size"],
        "design": "wheel",
        "task": task,
        # Every other runner writes `condition`; the K pipeline groups on it.
        # Wheel rows carry a distinct condition so they can never be silently
        # pooled into a palette-design K cell (and loader/k_estimation filter
        # on design=="wheel" besides).
        # includes manifest condition + prompt variant so v0/v1 and swap/dense
        # runs never collide in the loader's (trial_id, model, condition) dedupe
        "condition": condition_tag or f"wheel-{task}",
        "change": tr["change"],
        "delta_deg": tr["delta_deg"],
        "changed_idx": tr["changed_idx"],
        "wheel_zero_offset": tr["wheel_zero_offset"],
        "probe_idx": tr["probe_idx"],
        "provider": getattr(client, "provider", "local"),
        "model": getattr(client, "model_id", "unknown"),
        # precision provenance: Tier-2's closed-loop claim needs behavior and
        # geometry from the SAME effective model (bf16 vs 4-bit differ)
        "dtype": getattr(client, "effective_dtype", None),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _run_change(client, tr: dict, row: dict, prompt_id: int = 0) -> tuple[dict, bool]:
    from ..utils import CHANGE_PROMPT_VARIANTS
    t1, t2 = CHANGE_PROMPT_VARIANTS[prompt_id]
    sample, test = Path(tr["sample_path"]), Path(tr["test_path"])
    p_yes = None
    error = None
    raw = ""
    try:
        if hasattr(client, "two_turn_yes_prob"):
            raw, p_yes = client.two_turn_yes_prob(
                sample, t1, test, t2, SYSTEM_INSTRUCTION)
        else:
            resp = client.call_two_turn(
                sample, t1, test, t2, SYSTEM_INSTRUCTION)
            raw, error = resp.raw, resp.error
    except Exception as e:                                          # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    parsed = parse_yes_no(raw)
    correct = (parsed == tr["change"]) if parsed is not None else None
    row.update({
        "raw": raw, "parsed_yes": parsed, "parsable": parsed is not None,
        "correct": correct, "p_yes": p_yes, "error": error,
        "prompt_id": prompt_id,
    })
    return row, error is None


def _run_simultaneous(client, tr: dict, row: dict) -> tuple[dict, bool]:
    combined = Path(tr["combined_path"])
    error = None
    raw = ""
    try:
        resp = client.call_single(images=[combined], prompt=PROMPT_C1_SIMULTANEOUS,
                                  system=SYSTEM_INSTRUCTION)
        raw, error = resp.raw, resp.error
    except Exception as e:                                          # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    parsed = parse_yes_no(raw)
    correct = (parsed == tr["change"]) if parsed is not None else None
    row.update({
        "raw": raw, "parsed_yes": parsed, "parsable": parsed is not None,
        "correct": correct, "p_yes": None, "error": error,
    })
    return row, error is None


def _run_report(client, tr: dict, row: dict) -> tuple[dict, bool]:
    sample, report_img = Path(tr["sample_path"]), Path(tr["report_path"])
    error = None
    raw = ""
    try:
        resp = client.call_two_turn(
            sample, PROMPT_WHEEL_REPORT_TURN1, report_img, PROMPT_WHEEL_REPORT_TURN2,
            SYSTEM_INSTRUCTION)
        raw, error = resp.raw, resp.error
    except Exception as e:                                          # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    idx = parse_report_index(raw, hi=WHEEL_REPORT_M)
    err_deg = (score_report(idx, tr["report_hue_deg"], M=WHEEL_REPORT_M)
               if idx is not None else None)
    row.update({
        "raw": raw, "report_hue_deg": tr["report_hue_deg"], "reported_index": idx,
        "report_parsable": idx is not None, "report_error_deg": err_deg, "error": error,
        # `parsable` is the pipeline-wide filter key (every consumer checks it);
        # keep report_parsable as the task-specific alias.
        "parsable": idx is not None,
    })
    return row, error is None


def run_wheel(cfg: WheelRunConfig, client=None) -> dict:
    if cfg.task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {cfg.task!r}")
    manifest = load_manifest(cfg.manifest_path)
    if manifest.get("design") != "wheel":
        raise ValueError(f"manifest design={manifest.get('design')!r} != 'wheel' (need a wheel dataset)")

    trials = manifest["trials"]
    if cfg.set_sizes:
        trials = [t for t in trials if t["set_size"] in cfg.set_sizes]
    if cfg.deltas:
        trials = [t for t in trials if t["delta_deg"] in cfg.deltas]
    if cfg.n_trials is not None:
        trials = trials[: cfg.n_trials]

    mcond = manifest.get("condition", "wheel")
    cond_tag = f"{mcond}-{cfg.task}-p{cfg.prompt_id}"
    client = client or _resolve_client(cfg.model_key, cfg.dtype)
    client.ensure_loaded()
    ensure_dir(cfg.out_path.parent)

    n_ok = n_err = 0
    acc_list: list[float] = []
    p_list: list[float] = []
    chg_list: list[bool] = []
    err_list: list[float] = []

    t0 = time.monotonic()
    bar = tqdm(trials, desc=f"wheel/{cfg.model_key}/{cfg.task}")
    for tr in bar:
        row = _base_row(tr, client, cfg.task, cond_tag)
        if cfg.task == "change":
            row, ok = _run_change(client, tr, row, cfg.prompt_id)
            if row["correct"] is not None:
                acc_list.append(1.0 if row["correct"] else 0.0)
            if row["p_yes"] is not None:
                p_list.append(row["p_yes"])
                chg_list.append(bool(tr["change"]))
        elif cfg.task == "simultaneous":
            row, ok = _run_simultaneous(client, tr, row)
            if row["correct"] is not None:
                acc_list.append(1.0 if row["correct"] else 0.0)
        else:
            row, ok = _run_report(client, tr, row)
            if row["report_error_deg"] is not None:
                err_list.append(row["report_error_deg"])
        append_jsonl(cfg.out_path, row)
        n_ok += int(ok)
        n_err += int(not ok)
        bar.set_postfix(ok=n_ok, err=n_err)

    client.unload()
    dur = time.monotonic() - t0
    summary = {
        "model_key": cfg.model_key,
        "model": getattr(client, "model_id", "unknown"),
        "task": cfg.task,
        "n_trials": len(trials),
        "n_ok": n_ok,
        "n_err": n_err,
        "duration_s": dur,
        "out_path": str(cfg.out_path),
    }
    summary["dtype"] = getattr(client, "effective_dtype", None)
    if cfg.task in ("change", "simultaneous"):
        summary["mean_acc"] = (sum(acc_list) / len(acc_list)) if acc_list else None
        summary["auc"] = (roc_auc(p_list, chg_list)
                          if p_list and any(chg_list) and not all(chg_list) else None)
    else:
        summary["mean_error_deg"] = (sum(err_list) / len(err_list)) if err_list else None
    print(f"[wheel/{cfg.model_key}/{cfg.task}] {n_ok} ok / {n_err} err in {dur:.1f}s")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the CIELAB-wheel design (M3 report / M4 change).")
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--set-sizes", type=str, default=None)
    p.add_argument("--deltas", type=str, default=None)
    p.add_argument("--prompt-id", type=int, default=0,
                   help="CHANGE_PROMPT_VARIANTS index; 1 = binding-fair (required for swap)")
    p.add_argument("--dtype", choices=["bfloat16", "4bit"], default="bfloat16",
                   help="model precision; bf16 is the A100-validated regime and the "
                        "one the Tier-2 geometry must match")
    args = p.parse_args(argv)
    set_sizes = tuple(int(s) for s in args.set_sizes.split(",")) if args.set_sizes else None
    deltas = tuple(float(s) for s in args.deltas.split(",")) if args.deltas else None
    cfg = WheelRunConfig(
        model_key=args.model, task=args.task, manifest_path=args.manifest,
        out_path=args.out, n_trials=args.n_trials, set_sizes=set_sizes, deltas=deltas,
        dtype=args.dtype, prompt_id=args.prompt_id,
    )
    print(run_wheel(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
