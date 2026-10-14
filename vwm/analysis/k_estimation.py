"""K estimation with bootstrap CIs and an optional psychometric (psignifit) fit.

We expose two estimators, both standard in the VWM literature:

    Hit rate (H)        = P("change" response | change trial)
    False-alarm rate (F)= P("change" response | no-change trial)

    K_cowan(N)    = N * (H - F)                 # Cowan 2001 simplification
    K_pashler(N)  = N * (H - F) / (1 - F)       # Pashler 1988 high-threshold model

Pashler (1988) is the original formula and corrects for guessing bias when F is
high; Cowan (2001) is the more common simplification used when F is near 0.
The two converge when F is small. Per Rouder et al. (2008), the Cowan form is
preferred for population averages because it is well-defined even when F = 1
(the Pashler form diverges), but the Pashler form is more accurate per-cell.

We report the Cowan form as the primary K in figures and tables (it stays
well-defined even when F -> 1, as in C3 cache ablation), and the Pashler-corrected
form as a sanity-check column (`K_pashler`). Both are computed in `per_model_k_table`.
NOTE: the function `pashler_k` below computes the *Cowan* form N*(H-F); the name is
retained for backwards test compatibility. The Pashler-corrected form is
`pashler_k_corrected`. The `K` and `K_cowan` columns are therefore identical (Cowan),
and the paper's tables use that column.

Bootstrap CIs:
    For each set size, resample trials WITH replacement, recompute H and F, derive K,
    then report the 2.5 / 97.5 percentile. Default 2000 resamples.

Psychometric fit (optional, requires `psignifit`):
    Fit a Weibull/logistic to "1 - accuracy" vs set size to estimate the set size
    where performance falls to a target level (e.g. 75%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


# ---------------- core formulas ----------------

@dataclass
class HitsFA:
    n_change: int
    n_no_change: int
    n_hit: int
    n_fa: int

    @property
    def hit_rate(self) -> float:
        return (self.n_hit / self.n_change) if self.n_change > 0 else float("nan")

    @property
    def fa_rate(self) -> float:
        return (self.n_fa / self.n_no_change) if self.n_no_change > 0 else float("nan")


def compute_hits_fa(df: pd.DataFrame) -> HitsFA:
    """Compute hits and false alarms from a tidy results frame for one (model,condition,set_size)."""
    if df.empty:
        return HitsFA(0, 0, 0, 0)
    # Drop unparseable rows -- those are not "no" responses, they're missing data.
    df = df[df["parsable"] == True]                                     # noqa: E712
    is_change = df["change"] == True                                    # noqa: E712
    pred_yes = df["parsed_yes"] == True                                 # noqa: E712
    n_change = int(is_change.sum())
    n_no_change = int((~is_change).sum())
    n_hit = int((is_change & pred_yes).sum())
    n_fa = int((~is_change & pred_yes).sum())
    return HitsFA(n_change=n_change, n_no_change=n_no_change, n_hit=n_hit, n_fa=n_fa)


def pashler_k(set_size: int, hits_fa: HitsFA) -> float:
    """Cowan (2001) simplification: K = N * (H - F).

    Name kept for backwards compatibility with existing tests; the Pashler-corrected
    form is `pashler_k_corrected`.
    """
    return set_size * (hits_fa.hit_rate - hits_fa.fa_rate)


def cowan_k(set_size: int, hits_fa: HitsFA) -> float:
    """Cowan (2001) simplification, alias of `pashler_k`."""
    return pashler_k(set_size, hits_fa)


def pashler_k_corrected(set_size: int, hits_fa: HitsFA) -> float:
    """Pashler (1988) high-threshold form: K = N * (H - F) / (1 - F).

    Diverges as F -> 1 (we return NaN in that case). For F < 1 this corrects for
    guessing bias in the participant's "yes" responses.
    """
    denom = 1.0 - hits_fa.fa_rate
    if denom <= 0.0:
        return float("nan")
    return set_size * (hits_fa.hit_rate - hits_fa.fa_rate) / denom


# ---------------- bootstrap ----------------

def bootstrap_ci(
    df: pd.DataFrame,
    set_size: int,
    n_boot: int = 2000,
    rng_seed: int = 0,
    ci: tuple[float, float] = (2.5, 97.5),
) -> tuple[float, float, float]:
    """Return (K_point, K_low, K_high) from non-parametric bootstrap.

    Resamples trials with replacement. Unparseable trials are dropped from the
    bootstrap sample as well — same denominator as the point estimate.
    """
    base = df[df["parsable"] == True]                                   # noqa: E712
    if base.empty:
        return float("nan"), float("nan"), float("nan")
    point = pashler_k(set_size, compute_hits_fa(base))
    rng = np.random.default_rng(rng_seed)
    idx = base.index.to_numpy()
    ks = np.empty(n_boot, dtype=np.float64)
    is_change = base["change"].values
    pred_yes = base["parsed_yes"].values
    for b in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        loc = base.index.get_indexer(sample)
        # H and FA on the bootstrap sample
        ic = is_change[loc]
        py = pred_yes[loc]
        nc = ic.sum()
        nnc = (~ic).sum()
        if nc == 0 or nnc == 0:
            ks[b] = float("nan")
            continue
        h = (ic & py).sum() / nc
        f = (~ic & py).sum() / nnc
        ks[b] = set_size * (h - f)
    low, high = np.nanpercentile(ks, ci)
    return float(point), float(low), float(high)


# ---------------- per-model K table ----------------

def per_model_k_table(
    df: pd.DataFrame,
    n_boot: int = 2000,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """One row per (provider, model, condition, set_size) with both K forms + CIs, H, FA, N."""
    out: list[dict] = []
    # K estimation is defined for the palette change-detection design only.
    # Wheel rows (design=="wheel", incl. report rows with no parsed_yes) must
    # never pool into K cells; require a condition key rather than KeyError-ing.
    if "condition" not in df.columns:
        raise ValueError("per_model_k_table: dataframe has no 'condition' column — "
                         "are these wheel/report rows? K applies to palette designs.")
    if "design" in df.columns:
        df = df[df["design"] != "wheel"]
        if df.empty:
            return pd.DataFrame()
    grouped = df.groupby(["provider", "model", "condition", "set_size"], dropna=False)
    for (provider, model, condition, n), g in grouped:
        hf = compute_hits_fa(g)
        k_point, k_low, k_high = bootstrap_ci(g, set_size=n, n_boot=n_boot, rng_seed=rng_seed)
        k_corr = pashler_k_corrected(n, hf)
        out.append({
            "provider": provider,
            "model": model,
            "condition": condition,
            "set_size": int(n),
            "n_trials": int(len(g)),
            "n_parsable": int(g["parsable"].sum()) if "parsable" in g else int(len(g)),
            "hit_rate": hf.hit_rate,
            "fa_rate": hf.fa_rate,
            "K": k_point,                                # Cowan form (compat name)
            "K_cowan": k_point,
            "K_pashler": k_corr,                         # corrected form
            "K_low": k_low,
            "K_high": k_high,
            "accuracy": float((g["correct"] == True).sum() / max(int(g["parsable"].sum()), 1)),  # noqa: E712
        })
    return pd.DataFrame(out).sort_values(
        ["condition", "provider", "model", "set_size"]
    ).reset_index(drop=True)


# ---------------- optional: psignifit ----------------

def try_psignifit_fit(set_sizes: Sequence[int], accuracies: Sequence[float], n_trials_per_size: Sequence[int]):
    """Best-effort psychometric fit. Returns the psignifit result dict or None on failure."""
    try:
        import psignifit                                                 # noqa: WPS433
    except Exception:
        return None
    # psignifit expects an Nx3 array [stim, correct, total]
    n_correct = [int(round(acc * tot)) for acc, tot in zip(accuracies, n_trials_per_size)]
    data = np.array(list(zip(set_sizes, n_correct, n_trials_per_size)), dtype=np.float64)
    options = {
        "sigmoidName": "logistic",
        "expType": "2AFC",
    }
    try:
        return psignifit.psignifit(data, **options)
    except Exception:
        return None


if __name__ == "__main__":  # pragma: no cover
    import argparse
    from .loader import load_all_results, default_results_dir
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/raw")
    p.add_argument("--out", default="results/k_table.csv")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()
    df = load_all_results(default_results_dir(args.results_dir))
    if df.empty:
        raise SystemExit(f"No results found under {args.results_dir!r}")
    table = per_model_k_table(df, n_boot=args.n_boot)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {args.out}")
