"""Top-level orchestration: run one (provider, condition) and log to the ledger.

Wraps `vwm.experiments.runner.run_condition` and `vwm.local.local_runner.run_local`
(plus `vwm.local.c3_runner.run_c3`) so any execution path produces an identical
provenance row in `results/ledger.tsv`.

Usage (from a shell):

    python -m vwm.experiments.driver --provider openai     --condition C1
    python -m vwm.experiments.driver --provider anthropic  --condition C2
    python -m vwm.experiments.driver --provider qwen3-vl-4b --condition C3   # local

The driver enforces the autoresearch-pattern discipline:
  - records `commit` hash before launching so a crash is still attributable
  - computes mean_acc and K(N=8) from the resulting JSONL
  - assigns status: keep | discard | crash
  - description is free-form (override via --note)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .ledger import LedgerRow, append, now_iso, short_commit
from .runner import ExperimentConfig, run_condition
from ..analysis.loader import load_results_jsonl
from ..analysis.k_estimation import per_model_k_table


PROPRIETARY = {"openai", "anthropic", "gemini"}
LOCAL_KEYS = {"qwen3-vl-4b", "qwen2.5-vl-7b", "llava-1.5-7b"}


def _is_mock(name: str) -> bool:
    return name == "mock" or name.startswith("mock:")


def _summarize(out_path: Path) -> tuple[float, float, float, int]:
    """Return (mean_acc, K@N=8, fail_rate, n_trials) from a per-trial JSONL."""
    df = load_results_jsonl(out_path)
    if df.empty:
        return 0.0, float("nan"), 1.0, 0
    n = int(len(df))
    parsable = df[df["parsable"] == True]                                # noqa: E712
    fail_rate = 1.0 - (len(parsable) / n if n else 0.0)
    if parsable.empty:
        return 0.0, float("nan"), fail_rate, n
    mean_acc = float((parsable["correct"] == True).mean())               # noqa: E712
    try:
        tab = per_model_k_table(df, n_boot=500)
        row8 = tab[tab["set_size"] == 8]
        k8 = float(row8.iloc[0]["K"]) if not row8.empty else float("nan")
    except Exception:                                                    # noqa: BLE001
        k8 = float("nan")
    return mean_acc, k8, fail_rate, n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", required=True,
                   help=f"Proprietary: {sorted(PROPRIETARY)} | Local: {sorted(LOCAL_KEYS)}")
    p.add_argument("--condition", required=True, choices=["C1", "C2", "C3"])
    p.add_argument("--manifest", type=Path, default=None,
                   help="Override; defaults to data/stimuli/manifest_<condition>.json")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSONL path; defaults to results/raw/<provider>_<condition>.jsonl")
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--set-sizes", type=str, default=None)
    p.add_argument("--note", type=str, default="",
                   help="Free-form description for the ledger row.")
    p.add_argument("--ledger", type=Path, default=Path("results/ledger.tsv"))
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing output JSONL instead of refusing.")
    p.add_argument("--append", action="store_true",
                   help="Append to an existing output JSONL (rows from BOTH runs "
                        "will pool downstream — only for deliberate resumes).")
    args = p.parse_args(argv)

    manifest = args.manifest or Path(f"data/stimuli/manifest_{args.condition}.json")
    out_path = args.out or Path(f"results/raw/{args.provider}_{args.condition}.jsonl")

    # Output rows are appended one per trial; silently appending a rerun onto a
    # stale file double-counts every trial in the ledger and K tables.
    if out_path.exists() and not (args.overwrite or args.append):
        raise SystemExit(
            f"{out_path} already exists ({out_path.stat().st_size} bytes). "
            f"Rerunning would append and double-count downstream. Pass "
            f"--overwrite to replace it, --append to deliberately extend it, "
            f"or --out to write elsewhere.")
    if args.overwrite and out_path.exists():
        out_path.unlink()

    commit = short_commit()
    t0 = time.monotonic()
    status: str
    note = args.note or f"{args.provider} | {args.condition}"
    try:
        if args.provider in PROPRIETARY or _is_mock(args.provider):
            if args.condition == "C3":
                raise ValueError("C3 is open-models-only")
            cfg = ExperimentConfig(
                provider=args.provider, condition=args.condition,
                manifest_path=manifest, out_path=out_path,
                n_trials=args.n_trials,
                set_sizes=(tuple(int(s) for s in args.set_sizes.split(",")) if args.set_sizes else None),
            )
            run_condition(cfg)
        elif args.provider in LOCAL_KEYS:
            if args.condition == "C3":
                from ..local.c3_runner import C3Config, run_c3
                run_c3(C3Config(
                    model_key=args.provider, manifest_path=manifest, out_path=out_path,
                    n_trials=args.n_trials,
                    set_sizes=(tuple(int(s) for s in args.set_sizes.split(",")) if args.set_sizes else None),
                ))
            else:
                from ..local.local_runner import LocalConfig, run_local
                run_local(LocalConfig(
                    model_key=args.provider, condition=args.condition,
                    manifest_path=manifest, out_path=out_path,
                    n_trials=args.n_trials,
                    set_sizes=(tuple(int(s) for s in args.set_sizes.split(",")) if args.set_sizes else None),
                ))
        else:
            raise ValueError(f"Unknown provider {args.provider!r}")

        mean_acc, k8, fail_rate, n = _summarize(out_path)
        status = "keep" if (n > 0 and fail_rate < 0.05) else "discard"
        if n == 0:
            status = "crash"
        model_name = args.provider          # the JSONL knows the canonical model id; the row is provider-scoped
        rec = LedgerRow(
            commit=commit, timestamp=now_iso(),
            provider=args.provider, model=model_name, condition=args.condition,
            n_trials=n, mean_acc=mean_acc, k_at_8=k8, fail_rate=fail_rate,
            duration_s=time.monotonic() - t0,
            status=status, description=note,
        )
    except Exception as e:                                                # noqa: BLE001
        rec = LedgerRow(
            commit=commit, timestamp=now_iso(),
            provider=args.provider, model=args.provider, condition=args.condition,
            n_trials=0, mean_acc=0.0, k_at_8=float("nan"), fail_rate=1.0,
            duration_s=time.monotonic() - t0,
            status="crash",
            description=(args.note + " | " if args.note else "") + f"{type(e).__name__}: {e}",
        )
        append(rec, args.ledger)
        raise

    append(rec, args.ledger)
    print(rec.to_tsv())
    return 0


if __name__ == "__main__":
    sys.exit(main())
