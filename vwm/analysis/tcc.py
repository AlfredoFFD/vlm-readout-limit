"""TCC engine (Tier-2): the Target Confusability Competition forward model,
exact channel likelihood, d' fitting, and the max-rule Monte-Carlo ideal
observer used to CALIBRATE the change-detection falsifiers.

Source of truth for every equation: results/TCC_RESEARCH.md (each card there is
verified against Schurgin, Wixted & Brady 2020, NHB 4:1156-1172, or labeled OUR
DERIVATION). Conventions (pinned by tests/test_tcc.py):

  - 360 channels; degrees_grid() = -179..180; the studied target sits at 0
    (index 179). (The equations doc quoted the paper's MATLAB with 180
    channels for orientation-like spaces; for the 360-deg color wheel we use
    360 — one channel per wheel degree — which is the paper's color setup.)
  - f_vec is aligned to degrees_grid(), f(0)=1, min-max normalized to [0,1]
    (TCC_RESEARCH.md Sec 4/4.1).
  - Noise SD = 1 per channel, iid (the canonical 2-line uncorrelated model;
    TCC_RESEARCH.md Sec 1.3 note: correlation structure is not critical for
    continuous-report fits, d' rescales by ~0.65).

Falsifier-calibration context (assembly review 2026-08-07): the closed-form
yes/no ROC (Sec 3.3, OUR DERIVATION: d_change = d'(1-f(delta)),
AUC = Phi(d_change/sqrt(2))) is single-probe equal-variance SDT. The actual M4
task shows the WHOLE N-item test array, so the ideal observer is a max-rule
(extreme-value) detector whose ROC deviates from the closed form at N>1.
`simulate_change_auc_maxrule` is the calibrated replacement; quote the closed
form only where the calibration table (scripts/tcc_calibrate.py ->
results/tcc_calibration.md) shows it holds.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "degrees_grid", "synthetic_f", "f_from_cosine",
    "simulate_report_errors", "response_pmf", "fit_dprime",
    "change_dprime", "change_auc_closed_form", "simulate_change_auc_maxrule",
    "simulate_change_auc_lr",
    "anchor_channels", "simulate_report_12afc", "fit_dprime_nafc",
    "predict_all",
]

_N_CHANNELS = 360
_TARGET_IDX = 179          # degrees_grid()[179] == 0


def degrees_grid() -> np.ndarray:
    """Signed circular distances of every wheel color from the target."""
    return np.arange(-179, 181)


# ------------------------------------------------------------------- f(x) ---

def synthetic_f(tau: float = 50.0) -> np.ndarray:
    """Exponential-similarity f for simulations/calibration (Shepard-like;
    descriptive stand-in for a measured f — TCC_RESEARCH.md Sec 4.1)."""
    f = np.exp(-np.abs(degrees_grid()) / tau)
    return (f - f.min()) / (f.max() - f.min())


def f_from_cosine(similarities: np.ndarray, target_idx: int = _TARGET_IDX) -> np.ndarray:
    """Min-max normalize a raw similarity curve into TCC's f convention
    (f(target)=1, min 0). OUR DERIVATION (Sec 4): the model-geometry drop-in
    for the human Likert f. Raises if the maximum is not at the target — a
    non-self-maximal similarity curve means the embedding/readout is broken,
    and silently renormalizing it would poison every downstream prediction."""
    s = np.asarray(similarities, dtype=float)
    if s.argmax() != target_idx:
        raise ValueError(f"similarity max at index {s.argmax()}, expected the "
                         f"target index {target_idx}")
    return (s - s.min()) / (s.max() - s.min())


# ------------------------------------------------- continuous report (Sec 1) --

def simulate_report_errors(dprime: float, f_vec: np.ndarray, rng: np.random.Generator,
                           n_trials: int = 10000) -> np.ndarray:
    """Monte-Carlo continuous report: m_x ~ N(d'*f(x), 1), respond argmax.
    Returns signed errors in degrees (paper's 2-line generative model)."""
    f_vec = np.asarray(f_vec, dtype=float)
    signals = rng.standard_normal((n_trials, f_vec.size)) + dprime * f_vec
    return degrees_grid()[signals.argmax(axis=1)].astype(float)


def response_pmf(dprime: float, f_vec: np.ndarray, channels: np.ndarray | None = None,
                 n_grid: int = 3001) -> np.ndarray:
    """EXACT response distribution of the argmax rule (iid unit-variance):

        P(resp=j) = ∫ φ(m - d'f_j) · Π_{k≠j} Φ(m - d'f_k) dm

    computed by numerical integration (no Monte Carlo). `channels` restricts
    the competition to a subset of channel indices (n-AFC); default = all 360
    (continuous report)."""
    f_vec = np.asarray(f_vec, dtype=float)
    f = f_vec if channels is None else f_vec[np.asarray(channels)]
    lo = min(-8.0, dprime * f.min() - 8.0)
    hi = dprime * f.max() + 8.0
    m = np.linspace(lo, hi, n_grid)
    z = m[:, None] - dprime * f[None, :]           # (grid, channels)
    logcdf = norm.logcdf(z)
    total_logcdf = logcdf.sum(axis=1)
    log_integrand = norm.logpdf(z) + (total_logcdf[:, None] - logcdf)
    pmf = np.trapz(np.exp(log_integrand), m, axis=0)
    return pmf / pmf.sum()


def _nll_from_counts(dprime: float, f_vec: np.ndarray, counts: np.ndarray,
                     channels: np.ndarray | None = None) -> float:
    pmf = response_pmf(dprime, f_vec, channels=channels)
    return -float(np.sum(counts * np.log(np.clip(pmf, 1e-300, None))))


def fit_dprime(errors_deg: np.ndarray, f_vec: np.ndarray,
               bounds: tuple[float, float] = (0.0, 12.0)) -> float:
    """MLE of the single free parameter from signed continuous-report errors
    (degrees). Uses the exact channel likelihood; 1-D bounded optimization
    (TCC_RESEARCH.md Sec 5 STEP 1)."""
    from scipy.optimize import minimize_scalar
    errors = np.asarray(errors_deg)
    idx = np.round(errors).astype(int) + _TARGET_IDX
    counts = np.bincount(np.clip(idx, 0, _N_CHANNELS - 1), minlength=_N_CHANNELS)
    res = minimize_scalar(lambda d: _nll_from_counts(d, f_vec, counts),
                          bounds=bounds, method="bounded")
    return float(res.x)


# ----------------------------------- change detection (Sec 3.3 + calibration) --

def _f_at_delta(f_vec: np.ndarray, delta_deg: float) -> float:
    idx = int(round(delta_deg)) + _TARGET_IDX
    if not 0 <= idx < _N_CHANNELS:
        # wrap onto the circle
        idx = (int(round(delta_deg)) + 180) % 360 - 180 + _TARGET_IDX
    return float(np.asarray(f_vec)[idx])


def change_dprime(dprime: float, f_vec: np.ndarray, delta_deg: float) -> float:
    """OUR DERIVATION (Sec 3.3): d_change(delta) = d' * (1 - f(delta))."""
    return dprime * (1.0 - _f_at_delta(f_vec, delta_deg))


def change_auc_closed_form(dprime: float, f_vec: np.ndarray, delta_deg: float) -> float:
    """Single-probe equal-variance SDT closed form: AUC = Phi(d_change/sqrt(2)).
    VALID AT N=1 ONLY unless the calibration table says otherwise — the real
    task is an N-item max-rule (see simulate_change_auc_maxrule)."""
    return float(norm.cdf(change_dprime(dprime, f_vec, delta_deg) / np.sqrt(2.0)))


def _rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC = P(pos > neg) + 0.5*P(tie), rank-based, O(n log n)."""
    scores = np.concatenate([pos, neg])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1)
    # midranks for ties
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def _change_signals(dprime: float, f_d: float, set_size: int,
                    rng: np.random.Generator, n_trials: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(no-change, change) memory-match signal matrices for the N-item task:
    m_i ~ N(d'*f(x_i), 1); x_i = 0 for unchanged items, delta for the changed."""
    noise = rng.standard_normal((n_trials, set_size)) + dprime
    sig = rng.standard_normal((n_trials, set_size)) + dprime
    sig[:, 0] = rng.standard_normal(n_trials) + dprime * f_d
    return noise, sig


def simulate_change_auc_maxrule(dprime: float, f_vec: np.ndarray, delta_deg: float,
                                set_size: int, rng: np.random.Generator,
                                n_trials: int = 30000) -> float:
    """Min-rule HEURISTIC observer for the N-item change task: respond CHANGE
    when the least familiar item is suspicious enough (max-of-unfamiliarity).

    NOT the ideal observer (2026-08-07 assembly review, independently
    reproduced): the likelihood-ratio observer (`simulate_change_auc_lr`)
    beats this rule by ~0.01 AUC in small-delta/high-N cells — exactly where
    slot-vs-resource is adjudicated. Keep this as a plausible suboptimal
    decision model for comparison; pre-registered ideal-observer predictions
    use the LR rule."""
    noise, sig = _change_signals(dprime, _f_at_delta(f_vec, delta_deg),
                                 set_size, rng, n_trials)
    return _rank_auc(-sig.min(axis=1), -noise.min(axis=1))


def simulate_change_auc_lr(dprime: float, f_vec: np.ndarray, delta_deg: float,
                           set_size: int, rng: np.random.Generator,
                           n_trials: int = 30000) -> float:
    """Likelihood-ratio IDEAL observer for the N-item change task (the
    pre-registered baseline). With equal priors over which item changed,

        LR ∝ Σ_i p(m | item i changed)/p(m | no change)
           ∝ Σ_i exp(-d'(1-f(delta)) · m_i)
        dv = logsumexp(-d_change · m_i)

    — a softmin over familiarity, weighting every suspicious item, which is
    why it dominates the min-rule (that only looks at the single worst item).
    At N=1 both rules are monotone transforms of m_1 and agree with the
    closed form. At delta=0, d_change=0 makes dv constant -> AUC 0.5."""
    from scipy.special import logsumexp
    d_c = change_dprime(dprime, f_vec, delta_deg)
    noise, sig = _change_signals(dprime, _f_at_delta(f_vec, delta_deg),
                                 set_size, rng, n_trials)
    return _rank_auc(logsumexp(-d_c * sig, axis=1),
                     logsumexp(-d_c * noise, axis=1))


# --------------------------------------------- 12-AFC anchor report (M3 task) --

def anchor_channels(phase_deg: float, m: int = 12) -> list[int]:
    """Channel indices of the M legend anchors for a given per-trial phase
    (anchors at phase + k*360/M, mapped onto degrees_grid indices)."""
    step = 360.0 / m
    out = []
    for k in range(m):
        d = (phase_deg + k * step + 180.0) % 360.0 - 180.0   # wrap to (-180,180]
        d = 180.0 if d == -180.0 else d
        # modular guard: round-half-even can push -179.5 -> -180 -> index -1
        out.append((int(round(d)) + _TARGET_IDX) % _N_CHANNELS)
    return out


def simulate_report_12afc(dprime: float, f_vec: np.ndarray, rng: np.random.Generator,
                          n_trials: int = 10000, m: int = 12
                          ) -> list[tuple[float, int]]:
    """Simulate the M3 anchor-legend task: per trial a random phase, the
    observer argmaxes familiarity over the M anchor channels only. Returns
    [(phase_deg, chosen_channel_index), ...] — enough to fit d' with the
    matched n-AFC likelihood."""
    f_vec = np.asarray(f_vec, dtype=float)
    out: list[tuple[float, int]] = []
    phases = rng.uniform(0.0, 360.0 / m, size=n_trials)
    for phase in phases:
        ch = anchor_channels(float(phase), m=m)
        f = f_vec[ch]
        sig = rng.standard_normal(len(ch)) + dprime * f
        out.append((float(phase), ch[int(sig.argmax())]))
    return out


def fit_dprime_nafc(responses: list[tuple[float, int]], f_vec: np.ndarray,
                    m: int = 12, bounds: tuple[float, float] = (0.0, 12.0),
                    phase_bins: int = 30) -> float:
    """MLE of d' from quantized (M-AFC) reports with per-trial phase, using the
    MATCHED likelihood: for each (binned) phase, P(choose anchor j) from the
    exact n-AFC channel integral. This is the correct-model fit the assembly
    review asked to check against continuous-likelihood shortcuts."""
    from scipy.optimize import minimize_scalar
    f_vec = np.asarray(f_vec, dtype=float)
    step = (360.0 / m) / phase_bins
    binned: dict[int, np.ndarray] = {}
    for b in range(phase_bins):
        ch = anchor_channels((b + 0.5) * step, m=m)
        binned[b] = np.asarray(ch)

    # counts[bin][slot] over responses
    counts = np.zeros((phase_bins, m))
    for phase, chosen in responses:
        b = min(int(phase / step), phase_bins - 1)
        ch = binned[b]
        # CIRCULAR index distance: channel 0 (-179 deg) is 2 deg from channel
        # 358 (179 deg), not 358 — a linear argmin mis-slots wrap-adjacent picks
        lin = np.abs(ch - chosen)
        slot = int(np.minimum(lin, _N_CHANNELS - lin).argmin())
        counts[b, slot] += 1

    def nll(d: float) -> float:
        total = 0.0
        for b in range(phase_bins):
            if counts[b].sum() == 0:
                continue
            pmf = response_pmf(d, f_vec, channels=binned[b])
            total -= float(np.sum(counts[b] * np.log(np.clip(pmf, 1e-300, None))))
        return total

    res = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(res.x)


# ------------------------------------------------------ the prediction spine --

def predict_all(dprime: float, f_vec: np.ndarray) -> dict:
    """TCC_RESEARCH.md Sec 5 STEP 2: everything predictable at this set size
    from (d', f) with ZERO additional parameters. The signature is the
    contract — tests enforce that nothing else can be passed as a required
    argument. N>1 change-detection AUCs are NOT included here on purpose:
    they require the max-rule MC (set size is a design variable, not a fitted
    parameter — call simulate_change_auc_maxrule explicitly)."""
    f_vec = np.asarray(f_vec, dtype=float)
    pmf = response_pmf(dprime, f_vec)
    deg = degrees_grid().astype(float)
    # circular SD from the predicted pmf
    ang = np.deg2rad(deg)
    R = np.hypot((pmf * np.sin(ang)).sum(), (pmf * np.cos(ang)).sum())
    circ_sd = float(np.rad2deg(np.sqrt(-2.0 * np.log(max(R, 1e-12)))))
    deltas = (15, 30, 60, 90, 135, 180)
    return {
        "error_pmf": pmf,
        "auc_by_delta_closed": {d: change_auc_closed_form(dprime, f_vec, d)
                                for d in deltas},
        "d_change_by_delta": {d: change_dprime(dprime, f_vec, d) for d in deltas},
        "report_circ_sd": circ_sd,
    }
