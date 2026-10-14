"""Per-set-size memory-vs-retrieval delta curves.

While Figure 3 shows the headline N=8 ΔK = K_C1 - K_C2 as a single bar per
model, this figure shows the FULL profile: ΔK as a function of N.

For the headline Qwen3-VL-4B finding:
    N    K_C1   K_C2   K_C3   dK(C1-C2)   dK(C2-C3)
    1   1.00   1.00   -0.12        0.00       1.12
    2   1.48   1.96    0.00       -0.48       1.96
    3   2.52   3.00   -0.06       -0.48       3.06
    4   3.44   3.92    0.08       -0.48       3.84
    6   5.76   5.40    0.00        0.36       5.40
    8   7.84   4.48    0.00        3.36       4.48

Reveals that the memory cost (dK(C1-C2)) only becomes substantial at N=8,
while the cache-truncation cost (dK(C2-C3)) is large at every N. This
visualizes the difference between "memory bottleneck" and "retrieval-only
bottleneck" much more clearly than the N=8 snapshot alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_delta_curves(k_table: pd.DataFrame, out_path: Path) -> Path:
    df = k_table.copy()
    # Filter out mocks
    df = df[~df["provider"].astype(str).str.startswith("mock")]
    if df.empty:
        raise SystemExit("No non-mock rows in k_table")

    def _short(name: str) -> str:
        s = name.split("/", 1)[-1]
        return s.removesuffix("-Instruct")

    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, m in zip(axes, models):
        sub = df[df["model"] == m]
        pivot = sub.pivot_table(index="set_size", columns="condition", values="K", aggfunc="first")
        pivot = pivot.sort_index()
        if "C1" in pivot.columns and "C2" in pivot.columns:
            ax.plot(pivot.index, pivot["C1"] - pivot["C2"], "-o",
                    color="#d62728", lw=2,
                    label=r"K$_{C1}$ − K$_{C2}$ (memory cost; +ve = C2 hurts)")
        if "C2" in pivot.columns and "C3" in pivot.columns:
            ax.plot(pivot.index, pivot["C2"] - pivot["C3"], "-s",
                    color="#1f77b4", lw=2,
                    label=r"K$_{C2}$ − K$_{C3}$ (in-context retrieval value)")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("Set size N")
        ax.set_title(_short(m), fontsize=11)
        ax.set_xticks([1, 2, 3, 4, 6, 8])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)

        # Annotate the sign convention so a reader of the slide can interpret it cold.
        ax.axhspan(-10, 0, color="#d62728", alpha=0.04)
        ax.axhspan(0, 10, color="#1f77b4", alpha=0.04)

    axes[0].set_ylabel(r"$\Delta$K")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    # Explanatory subtitle that survives the slide-reader's quick glance.
    fig.suptitle("Figure 4. Per-set-size memory and retrieval gaps\n"
                 r"Red > 0: model loses capacity when forced to remember (C2).  "
                 r"Red < 0: model gains capacity from the sequential format.",
                 fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/figures/k_table.csv")
    p.add_argument("--out", default="results/figures/fig4_delta_curves.png")
    args = p.parse_args(argv)
    df = pd.read_csv(args.csv)
    print(f"wrote {plot_delta_curves(df, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
