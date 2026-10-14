"""Autoresearch-style progress chart from `results/ledger.tsv`.

X-axis: experiment number (chronological).
Y-axis: K at set size 8 (the canonical capacity estimate).
Gray dots: discarded or crashed runs.
Colored dots: kept runs, colored by condition.
Annotation: short note from the ledger description.

Pattern lifted from `karpathy/autoresearch/progress.png` (March 2026):
visualization of incremental research progress over many small experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..experiments.ledger import read_ledger


CONDITION_COLORS = {
    "C1": "#1f77b4",
    "C2": "#dd8452",
    "C3": "#4c72b0",
}


def plot_progress(ledger_path: Path = Path("results/ledger.tsv"),
                  out_path: Path = Path("results/figures/progress.png")) -> Path:
    rows = read_ledger(ledger_path)
    if not rows:
        raise SystemExit(f"No ledger rows in {ledger_path!r}")

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot all rows
    for i, r in enumerate(rows):
        x = i
        y = r.k_at_8 if np.isfinite(r.k_at_8) else float("nan")
        if r.status in ("discard", "crash"):
            ax.plot(x, y, "o", color="lightgray", markersize=7, markeredgecolor="gray")
        else:
            ax.plot(x, y, "o", color=CONDITION_COLORS.get(r.condition, "black"),
                    markersize=9, markeredgecolor="black")
            # Annotate kept runs
            label = f"{r.provider}/{r.condition}"
            ax.annotate(label, xy=(x, y), xytext=(5, 5),
                        textcoords="offset points", fontsize=7)

    # Running best line (max K_at_8 over kept runs)
    best = []
    cur = -np.inf
    for r in rows:
        if r.status == "keep" and np.isfinite(r.k_at_8) and r.k_at_8 > cur:
            cur = r.k_at_8
        best.append(cur if np.isfinite(cur) else np.nan)
    ax.plot(range(len(rows)), best, "-", color="black", lw=1, alpha=0.5, label="running best K@N=8")

    # Human reference band
    ax.axhspan(2.1, 4.1, color="gray", alpha=0.15, label="human (Rouder 2008)")
    ax.axhline(3.1, color="black", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xlabel("Experiment #")
    ax.set_ylabel("K at N=8")
    ax.set_title(f"Autoresearch progress — {len(rows)} experiments, "
                 f"{sum(1 for r in rows if r.status == 'keep')} kept")
    # Conditions legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=col,
               markeredgecolor="black", markersize=8, label=cond)
        for cond, col in CONDITION_COLORS.items()
    ]
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray",
                          markeredgecolor="gray", markersize=8, label="discarded / crashed"))
    handles.append(Line2D([0], [0], color="black", linestyle="--", alpha=0.5, label="human mean"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ledger", type=Path, default=Path("results/ledger.tsv"))
    p.add_argument("--out", type=Path, default=Path("results/figures/progress.png"))
    args = p.parse_args(argv)
    print(f"wrote {plot_progress(args.ledger, args.out)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
