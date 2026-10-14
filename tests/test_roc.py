"""CPU tests for M4: continuous YES/NO decision signal -> SDT ROC primitives.

Phase 0.5(c). The model-touching part (generate(output_scores=True)) is GPU-only and
validated on Colab; everything science-bearing -- pulling P(YES) from step-0 logits,
resolving YES/NO token ids, the ROC/AUC/d' math -- is pure and tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from vwm.analysis.roc import (
    yes_prob_from_scores,
    resolve_yes_no_token_ids,
    roc_curve,
    roc_auc,
    dprime,
)


# ---------------- P(YES) from logits ----------------

def test_yes_prob_high_when_yes_logit_dominates():
    z = np.full(10, -5.0)
    z[3] = 8.0     # a YES token
    z[7] = -1.0    # a NO token
    assert yes_prob_from_scores(z, yes_ids=[3], no_ids=[7]) > 0.99


def test_yes_prob_half_when_balanced():
    z = np.zeros(10)
    z[3] = 2.0
    z[7] = 2.0
    assert yes_prob_from_scores(z, yes_ids=[3], no_ids=[7]) == pytest.approx(0.5)


def test_yes_prob_ignores_non_yes_no_tokens():
    z = np.full(10, -10.0)
    z[0] = 100.0   # some unrelated token with huge logit -> must be ignored
    z[3] = 1.0     # YES
    z[7] = 0.0     # NO
    p = yes_prob_from_scores(z, yes_ids=[3], no_ids=[7])
    assert p == pytest.approx(np.exp(1.0) / (np.exp(1.0) + np.exp(0.0)))


def test_yes_prob_sums_multiple_surface_forms():
    z = np.full(12, -8.0)
    z[3] = 1.0
    z[4] = 1.0     # two YES surface forms
    z[7] = 1.0     # one NO
    p = yes_prob_from_scores(z, yes_ids=[3, 4], no_ids=[7])
    assert p == pytest.approx(2 * np.exp(1.0) / (2 * np.exp(1.0) + np.exp(1.0)))


# ---------------- YES/NO token id resolution ----------------

class _FakeTokenizer:
    def __init__(self, table):
        self._table = table   # word -> list[int]

    def encode(self, text, add_special_tokens=False):
        return list(self._table.get(text, []))


def test_resolve_yes_no_uses_first_token_id_and_dedups():
    tok = _FakeTokenizer({
        "YES": [101, 5], "Yes": [101, 6], " yes": [102],
        "NO": [200], "No": [200], " no": [201],
    })
    yes, no = resolve_yes_no_token_ids(
        tok, yes_words=("YES", "Yes", " yes"), no_words=("NO", "No", " no")
    )
    assert yes == [101, 102]      # first ids, sorted, deduped
    assert no == [200, 201]


def test_resolve_raises_when_yes_no_share_only_token():
    tok = _FakeTokenizer({"YES": [42], "NO": [42]})
    with pytest.raises(ValueError):
        resolve_yes_no_token_ids(tok, yes_words=("YES",), no_words=("NO",))


# ---------------- ROC / AUC ----------------

def test_roc_auc_perfect_separation_is_one():
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    labels = np.array([True, True, True, False, False, False])
    assert roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_reversed_is_zero():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    labels = np.array([True, True, True, False, False, False])
    assert roc_auc(scores, labels) == pytest.approx(0.0)


def test_roc_auc_all_tied_is_half():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([True, True, False, False])
    assert roc_auc(scores, labels) == pytest.approx(0.5)


def test_roc_curve_endpoints_and_monotone():
    scores = np.array([0.9, 0.6, 0.4, 0.1])
    labels = np.array([True, True, False, False])
    fpr, tpr, thr = roc_curve(scores, labels)
    assert len(fpr) == len(tpr) == len(thr)
    assert fpr[0] == pytest.approx(0.0) and tpr[0] == pytest.approx(0.0)
    assert fpr[-1] == pytest.approx(1.0) and tpr[-1] == pytest.approx(1.0)
    assert np.all(np.diff(fpr) >= -1e-9)        # non-decreasing
    assert np.all(np.diff(tpr) >= -1e-9)


def _trapezoid(fpr, tpr):
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def test_roc_auc_matches_trapezoid_of_curve():
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    labels = rng.random(200) < (0.3 + 0.4 * scores)   # correlated -> auc > 0.5
    fpr, tpr, _ = roc_curve(scores, labels)
    assert roc_auc(scores, labels) == pytest.approx(_trapezoid(fpr, tpr), abs=1e-9)


def test_roc_curve_trapezoid_equals_auc_all_ties():
    # the documented contract (AUC == trapezoid of the curve) must hold even with ties
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([True, True, False, False])
    fpr, tpr, _ = roc_curve(scores, labels)
    assert roc_auc(scores, labels) == pytest.approx(0.5)
    assert _trapezoid(fpr, tpr) == pytest.approx(0.5)


def test_roc_curve_trapezoid_equals_auc_heavy_ties():
    rng = np.random.default_rng(1)
    scores = np.round(rng.random(300), 2)             # quantized -> many tied scores
    labels = rng.random(300) < (0.3 + 0.4 * scores)
    fpr, tpr, _ = roc_curve(scores, labels)
    assert roc_auc(scores, labels) == pytest.approx(_trapezoid(fpr, tpr), abs=1e-9)


# ---------------- d' ----------------

def test_dprime_zero_when_hit_equals_fa():
    assert dprime(0.5, 0.5) == pytest.approx(0.0)


def test_dprime_symmetric_value():
    # z(0.84134) - z(0.15866) ~= 1.0 - (-1.0) = 2.0
    assert dprime(0.84134, 0.15866) == pytest.approx(2.0, abs=1e-3)


def test_dprime_clamps_extremes_with_counts():
    d = dprime(1.0, 0.0, n_signal=100, n_noise=100)
    assert np.isfinite(d) and d > 0
