"""Publication-quality figures for the VWM-MLLM preprint.

Fig 1 - Psychometric curves: accuracy vs set size, one line per model, faceted by
        condition. The human reference is a horizontal dashed line at K_human/N
        marked on each panel.

Fig 2 - K bar chart: K per model with bootstrap CIs, grouped by condition,
        with the human reference band drawn behind.

Fig 3 - C1 vs C2 delta: per-model bar of (K_C1 - K_C2). Positive = retrieval is
        the only bottleneck (memory adds nothing). Near-zero or negative =
        memory is the operative bottleneck.

Style: matplotlib only (no seaborn dependency at figure time). Vector PDF + 300dpi PNG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.k_estimation import per_model_k_table
from ..analysis.loader import load_all_results


# --- Human reference K from Rouder 2008 (n=170) and Adam/Vogel/Awh 2017 ---
HUMAN_K_REFERENCE: dict[str, float] = {
    "mean": 3.1,        # ~3 items, consistent across studies
    "sd": 1.0,
    "source": "Rouder et al. 2008 (n=170); Adam, Vogel & Awh 2017",
}

# Set sizes used across the experiment.
SET_SIZES: tuple[int, ...] = (1, 2, 3, 4, 6, 8)


# ---------- shared helpers ----------

def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)


def _model_palette(models: Sequence[str]) -> dict[str, tuple]:
    """Stable color palette across figures."""
    cmap = plt.get_cmap("tab10")
    return {m: cmap(i % 10) for i, m in enumerate(models)}


# ---------- Figure 1: psychometric curves ----------

def plot_psychometric_curves(
    k_table: pd.DataFrame,
    out_path: Path,
    conditions: Sequence[str] = ("C1", "C2"),
) -> Path:
    """Accuracy vs set size, one line per model, one panel per condition."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(conditions), figsize=(4.5 * len(conditions), 4), sharey=True)
    if len(conditions) == 1:
        axes = [axes]
    models = sorted(k_table["model"].unique())
    pal = _model_palette(models)

    for ax, cond in zip(axes, conditions):
        sub = k_table[k_table["condition"] == cond].copy()
        for m, g in sub.groupby("model"):
            g = g.sort_values("set_size")
            ax.plot(g["set_size"], g["accuracy"], "-o", color=pal[m], label=m, lw=2, ms=5)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance" if cond == conditions[0] else None)
        # Human reference at typical accuracy (~0.83 across set sizes 1-8 from Rouder 2008)
        ax.axhline(0.83, color="black", linestyle="--", linewidth=1, alpha=0.5,
                   label="human (Rouder 2008)" if cond == conditions[0] else None)
        ax.set_xlabel("Set size N")
        ax.set_title(f"Condition {cond}")
        ax.set_xticks(SET_SIZES)
        ax.set_ylim(0.4, 1.05)
        _style_axis(ax)

    axes[0].set_ylabel("Accuracy (P correct)")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Figure 1. Psychometric curves: accuracy vs set size")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------- Figure 2: K bar chart with human ref ----------

def _short_model_name(name: str) -> str:
    """Strip the HF org prefix and the '-Instruct' suffix for cleaner axis labels."""
    s = name.split("/", 1)[-1]
    s = s.removesuffix("-Instruct")
    return s


def plot_k_bar_chart(
    k_table: pd.DataFrame,
    out_path: Path,
    conditions: Sequence[str] = ("C1", "C2", "C3"),
    summary_set_size: int = 8,
) -> Path:
    """One bar per (model, condition) at the largest set size, with bootstrap CIs.

    The summary set size determines where K asymptotes -- Pashler's K is defined
    per set size but the largest (N=8) is the canonical capacity estimate.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sub = k_table[k_table["set_size"] == summary_set_size].copy()
    sub = sub.sort_values(["condition", "model"])
    if sub.empty:
        raise ValueError(f"No rows at set_size={summary_set_size} in k_table")
    models = sorted(sub["model"].unique())
    pal = _model_palette(models)

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(models) * len(conditions) + 2), 5))
    n_models = len(models)
    bar_w = 0.8 / len(conditions)
    for ci, cond in enumerate(conditions):
        sc = sub[sub["condition"] == cond].set_index("model").reindex(models)
        x = np.arange(n_models) + (ci - (len(conditions) - 1) / 2) * bar_w
        heights = sc["K"].values.astype(float)
        lo = sc["K_low"].values.astype(float)
        hi = sc["K_high"].values.astype(float)
        err_lo = np.maximum(heights - lo, 0)
        err_hi = np.maximum(hi - heights, 0)
        # Alpha grades 0.5 -> 1.0 across conditions so visually the bar tone tracks condition order.
        alpha = 0.5 + 0.5 * (ci / max(len(conditions) - 1, 1))
        bars = ax.bar(
            x, heights, width=bar_w, label=cond,
            color=[pal[m] for m in models],
            edgecolor="black",
            alpha=alpha,
        )
        ax.errorbar(x, heights, yerr=[err_lo, err_hi], fmt="none",
                    ecolor="black", capsize=3, lw=1)
        # condition label above each cluster -- placed *above the top of the error bar*
        # so it doesn't overlap the bars or CIs.
        for xi, h, hi in zip(x, heights, hi):
            top = max(h, hi)
            ax.text(xi, top + 0.18, cond, ha="center", fontsize=9, color="black")

    # Human reference band
    hm, hs = HUMAN_K_REFERENCE["mean"], HUMAN_K_REFERENCE["sd"]
    ax.axhspan(hm - hs, hm + hs, color="gray", alpha=0.18, label=f"human K = {hm:.1f}±{hs:.1f}")
    ax.axhline(hm, color="black", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xticks(np.arange(n_models))
    ax.set_xticklabels([_short_model_name(m) for m in models],
                       rotation=0, ha="center", fontsize=10)
    ax.set_ylabel(f"Cowan's K = N(H − F)  (at N={summary_set_size})")
    ax.set_title(f"Figure 2. K vs human reference (N={summary_set_size})")
    # Put the legend OUTSIDE the plot to avoid overlapping the bars.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------- Figure 3: C1 - C2 delta ----------

def plot_c1_c2_delta(
    k_table: pd.DataFrame,
    out_path: Path,
    summary_set_size: int = 8,
) -> Path:
    """Per-model bar of K(C1) - K(C2). Reveals whether *memory* (not just retrieval)
    limits performance: a large positive delta means the model loses capacity when
    forced to remember rather than directly compare.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub = k_table[k_table["set_size"] == summary_set_size].copy()
    pivot = sub.pivot_table(index="model", columns="condition", values="K", aggfunc="first")
    if "C1" not in pivot.columns or "C2" not in pivot.columns:
        raise ValueError("Need both C1 and C2 in k_table to plot delta")
    pivot["delta"] = pivot["C1"] - pivot["C2"]
    pivot = pivot.sort_values("delta", ascending=False)

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(pivot) + 2), 4.5))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in pivot["delta"]]
    ax.bar(range(len(pivot)), pivot["delta"].values, color=colors, edgecolor="black")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels(pivot.index, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("ΔK (C1 − C2)")
    ax.set_title(f"Figure 3. Memory-vs-retrieval gap (N={summary_set_size})\n"
                 f"Δ>0 → loses capacity when forced to remember; Δ≈0 → retrieval is the only bottleneck")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------- driver ----------

def make_all(
    results_dir: str | Path = "results/raw",
    figures_dir: str | Path = "results/figures",
    n_boot: int = 2000,
    exclude_providers: tuple[str, ...] = (),
) -> dict:
    """Load all results, compute K-table, emit figures 1-3. Returns paths.

    Set `exclude_providers` to drop e.g. "mock:perfect" from publication figures
    while still keeping them in the K-table CSV (which is auditable).
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    from ..analysis.loader import default_results_dir
    df = load_all_results(default_results_dir(results_dir))
    if df.empty:
        raise SystemExit(f"No results found under {results_dir!r}")
    table = per_model_k_table(df, n_boot=n_boot)
    table.to_csv(figures_dir / "k_table.csv", index=False)
    if exclude_providers:
        table = table[~table["provider"].isin(exclude_providers)].reset_index(drop=True)
    out: dict[str, Path] = {}
    # Only include conditions that actually appear in the table (don't render an empty C3 column).
    present_conditions = tuple(c for c in ("C1", "C2", "C3") if c in set(table["condition"]))
    out["fig1"] = plot_psychometric_curves(table, figures_dir / "fig1_psychometric.png",
                                           conditions=present_conditions[:2] or ("C1",))
    out["fig2"] = plot_k_bar_chart(table, figures_dir / "fig2_k_vs_human.png",
                                   conditions=present_conditions or ("C1", "C2"))
    try:
        out["fig3"] = plot_c1_c2_delta(table, figures_dir / "fig3_c1_c2_delta.png")
    except ValueError as e:
        # Only C1 results present (e.g. partial run) -> skip Fig 3
        print(f"[fig3 skipped]: {e}")
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--figures-dir", default="results/figures")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--exclude", type=str, default="",
                   help="Comma-separated provider names to drop from figures (e.g. 'mock:perfect,mock:chance')")
    args = p.parse_args()
    excl = tuple(x.strip() for x in args.exclude.split(",") if x.strip())
    paths = make_all(args.results_dir, args.figures_dir, n_boot=args.n_boot,
                     exclude_providers=excl)
    for name, p in paths.items():
        print(f"{name}: {p}")
