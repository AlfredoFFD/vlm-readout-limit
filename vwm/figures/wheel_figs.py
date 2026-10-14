"""Tier-2 behavioral figures for the CIELAB-wheel design.

Renders the falsifier-bearing plots from `vwm.analysis.wheel_analysis` summaries:
- miss-rate vs change magnitude per set size (resource/TCC smooth decline vs slot step),
- empirical SDT ROC per set size with the high-threshold chord for contrast,
- report precision (circular SD) vs set size,
- signed report-error histograms per set size.

TCC model-prediction overlays are added once the parameter-free engine lands; these
plot the DATA only. Matplotlib Agg backend; each figure also saved as PDF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..analysis.wheel_analysis import (
    miss_rate_by_delta,
    detection_by_set_size,
    roc_points_by_set_size,
    report_error_by_set_size,
    report_signed_errors,
)
from ..utils.io import read_jsonl

_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def _save(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_miss_rate_by_delta(by_delta: dict, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, n in enumerate(sorted(by_delta)):
        deltas = sorted(by_delta[n])
        ax.plot(deltas, [by_delta[n][d] for d in deltas], "-o",
                color=_COLORS[i % len(_COLORS)], lw=2, label=f"N={n}")
    ax.set_xlabel("Change magnitude δ (deg)")
    ax.set_ylabel("Miss rate  P(\"no\" | change)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Miss rate vs change magnitude\n(smooth decline → resource/TCC; flat → slot/all-or-none)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    return _save(fig, out_path)


def plot_roc(roc_points: dict, detection: dict, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="chance")
    for i, n in enumerate(sorted(roc_points)):
        rp = roc_points[n]
        c = _COLORS[i % len(_COLORS)]
        det = detection.get(n, {})
        auc = det.get("auc")
        lbl = f"N={n}" + (f" (AUC={auc:.2f})" if auc is not None else "")
        ax.plot(rp["fpr"], rp["tpr"], "-", color=c, lw=2, label=lbl)
        # high-threshold (slot) chord through the (F,H) operating point, for contrast
        hit, fa = det.get("hit"), det.get("fa")
        if hit is not None and fa is not None:
            ax.plot([0, fa, 1], [0, hit, 1], ":", color=c, lw=1, alpha=0.7)
    ax.set_xlabel("False-alarm rate")
    ax.set_ylabel("Hit rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("SDT ROC per set size\n(solid = graded P(yes); dotted = high-threshold/slot chord)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_aspect("equal")
    return _save(fig, out_path)


def plot_precision(report_by_set_size: dict, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ns = sorted(report_by_set_size)
    sds = [report_by_set_size[n].get("circular_sd_deg") for n in ns]
    ax.plot(ns, sds, "-o", color="#d62728", lw=2)
    ax.set_xlabel("Set size N")
    ax.set_ylabel("Report circular SD (deg)")
    ax.set_title("Report imprecision vs set size\n(rising SD → precision falls with load)", fontsize=10)
    ax.set_xticks(ns)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)
    return _save(fig, out_path)


def plot_report_error_hist(signed_errors_by_set_size: dict, out_path: Path) -> Path:
    ns = sorted(signed_errors_by_set_size)
    fig, axes = plt.subplots(1, len(ns), figsize=(3.2 * len(ns), 3.4), sharey=True, squeeze=False)
    bins = list(range(-180, 181, 15))
    for ax, n in zip(axes[0], ns):
        ax.hist(signed_errors_by_set_size[n], bins=bins, color="#1f77b4", alpha=0.85)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_title(f"N={n}", fontsize=10)
        ax.set_xlabel("Report error (deg)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0][0].set_ylabel("count")
    fig.suptitle("Signed report-error distribution (central peak + swap tails)", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_path)


def make_wheel_figures(change_rows, report_rows, out_dir: Path, manifest: dict | None = None) -> dict:
    """Render all wheel behavioral figures from runner rows. Returns {name: path}."""
    out_dir = Path(out_dir)
    paths = {}
    if change_rows:
        paths["miss_rate"] = plot_miss_rate_by_delta(
            miss_rate_by_delta(change_rows), out_dir / "wheel_miss_rate_by_delta.png")
        paths["roc"] = plot_roc(
            roc_points_by_set_size(change_rows), detection_by_set_size(change_rows),
            out_dir / "wheel_roc.png")
    if report_rows:
        paths["precision"] = plot_precision(
            report_error_by_set_size(report_rows), out_dir / "wheel_precision.png")
        paths["report_hist"] = plot_report_error_hist(
            report_signed_errors(report_rows), out_dir / "wheel_report_hist.png")
    return paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render wheel behavioral figures from runner JSONL.")
    p.add_argument("--change", type=Path, default=None, help="change-task JSONL")
    p.add_argument("--report", type=Path, default=None, help="report-task JSONL")
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    args = p.parse_args(argv)
    change_rows = list(read_jsonl(args.change)) if args.change else []
    report_rows = list(read_jsonl(args.report)) if args.report else []
    paths = make_wheel_figures(change_rows, report_rows, args.out)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
