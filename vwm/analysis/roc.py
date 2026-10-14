"""M4: continuous YES/NO decision signal -> SDT ROC.

The 1-token YES/NO answer collapses the decision to a hard bit. To trace a *curved*
SDT ROC (and contrast it with the 2-point linear high-threshold ROC that the slot
model predicts), pull a graded P(YES) from the first generated step's logits,
restricted to the YES/NO token subset:

    P(YES) = sum_y exp(z[y]) / ( sum_y exp(z[y]) + sum_n exp(z[n]) )

Then sweep a criterion over P(YES) across trials. Everything here is pure NumPy +
stdlib; the only model-touching helper (`resolve_yes_no_token_ids`) is unit-tested
against a fake tokenizer so the GPU extraction path stays a thin, validated seam.

Inverse-normal (z) uses ``statistics.NormalDist`` so there is no SciPy dependency.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Sequence

import numpy as np

# Surface forms a greedy first token may take for each answer. The leading-space
# variants matter for BPE/SentencePiece tokenizers where " YES" != "YES".
YES_WORDS: tuple[str, ...] = ("YES", "Yes", "yes", " YES", " Yes", " yes")
NO_WORDS: tuple[str, ...] = ("NO", "No", "no", " NO", " No", " no")

_Z = NormalDist().inv_cdf


def yes_prob_from_scores(
    step0_logits, yes_ids: Sequence[int], no_ids: Sequence[int]
) -> float:
    """P(YES) from the first generated step's logits over the YES/NO token subset."""
    z = np.asarray(step0_logits, dtype=np.float64).ravel()
    yz = z[list(yes_ids)]
    nz = z[list(no_ids)]
    m = float(max(yz.max(), nz.max()))           # log-sum-exp stability
    sy = float(np.exp(yz - m).sum())
    sn = float(np.exp(nz - m).sum())
    return sy / (sy + sn)


def resolve_yes_no_token_ids(
    tokenizer,
    yes_words: Sequence[str] = YES_WORDS,
    no_words: Sequence[str] = NO_WORDS,
) -> tuple[list[int], list[int]]:
    """Map YES/NO surface forms to the FIRST token id each encodes to (deduped).

    Raises if either set is empty, or if the two sets share their only token (the
    readout would be ill-defined). Shared tokens are dropped from both sides first.
    """
    def first_ids(words: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for w in words:
            enc = tokenizer.encode(w, add_special_tokens=False)
            if enc:
                ids.append(int(enc[0]))
        return sorted(set(ids))

    yes = first_ids(yes_words)
    no = first_ids(no_words)
    overlap = set(yes) & set(no)
    if overlap:
        yes = [i for i in yes if i not in overlap]
        no = [i for i in no if i not in overlap]
    if not yes or not no:
        raise ValueError("could not resolve disjoint YES/NO first-token ids")
    return yes, no


def _as_pos_neg(scores, labels) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    if s.shape != y.shape:
        raise ValueError("scores and labels must have the same length")
    pos = s[y]
    neg = s[~y]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("need at least one positive (change) and one negative (no-change) trial")
    return pos, neg


def roc_auc(scores, labels) -> float:
    """Area under the ROC = P(score(change) > score(no-change)), ties counted as 0.5
    (Mann-Whitney U). Equals the trapezoidal area under ``roc_curve``."""
    pos, neg = _as_pos_neg(scores, labels)
    diff = pos[:, None] - neg[None, :]
    wins = float(np.count_nonzero(diff > 0))
    ties = float(np.count_nonzero(diff == 0))
    return (wins + 0.5 * ties) / (pos.size * neg.size)


def roc_curve(scores, labels) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step ROC. Returns (fpr, tpr, thresholds) including the (0,0) origin; the final
    point is (1,1). Thresholds are descending; predict positive iff score >= threshold.

    Tied scores are collapsed into a single threshold step (one ROC vertex per distinct
    score), so a tie block spanning both classes yields a diagonal whose trapezoid area
    matches ``roc_auc`` (Mann-Whitney) exactly -- without this, ties produce spurious
    right-angle staircases and the area diverges from the AUC."""
    pos, neg = _as_pos_neg(scores, labels)
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(float)
    order = np.argsort(-s, kind="mergesort")     # stable, descending
    s_sorted = s[order]
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1.0 - y_sorted)
    # keep only the last index of each run of equal scores (true threshold boundaries)
    bnd = np.where(np.diff(s_sorted) != 0)[0]
    idx = np.append(bnd, s_sorted.size - 1)
    tpr = np.concatenate([[0.0], tps[idx] / pos.size])
    fpr = np.concatenate([[0.0], fps[idx] / neg.size])
    thr = np.concatenate([[np.inf], s_sorted[idx]])
    return fpr, tpr, thr


def _clamp_rate(r: float, n: int | None) -> float:
    if n:
        lo, hi = 0.5 / n, 1.0 - 0.5 / n          # standard 1/(2N) edge correction
    else:
        lo, hi = 1e-6, 1.0 - 1e-6
    return min(max(float(r), lo), hi)


def dprime(
    hit_rate: float, fa_rate: float,
    n_signal: int | None = None, n_noise: int | None = None,
) -> float:
    """SDT d' = z(hit) - z(false-alarm), with 1/(2N) clamping (or epsilon if N unknown)
    so perfect/zero rates do not blow up to +/-inf."""
    return _Z(_clamp_rate(hit_rate, n_signal)) - _Z(_clamp_rate(fa_rate, n_noise))


def high_threshold_roc(hit_rate: float, fa_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """The linear high-threshold ROC predicted by an all-or-none slot model: the
    straight chords (0,0)->(F,H)->(1,1). Contrast this against the curved SDT ROC."""
    f = float(fa_rate)
    h = float(hit_rate)
    return np.array([0.0, f, 1.0]), np.array([0.0, h, 1.0])
