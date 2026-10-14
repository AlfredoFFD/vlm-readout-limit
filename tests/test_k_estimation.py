"""Unit tests for K-estimation math."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vwm.analysis.k_estimation import (
    HitsFA,
    compute_hits_fa,
    pashler_k,
    pashler_k_corrected,
    cowan_k,
    bootstrap_ci,
    per_model_k_table,
)


def test_pashler_corrected_diverges_when_fa_is_one():
    """Pashler-corrected form returns NaN when F = 1 (denominator 1 - F = 0)."""
    import math
    hf = HitsFA(n_change=10, n_no_change=10, n_hit=10, n_fa=10)  # H=1, F=1
    assert hf.hit_rate == 1.0 and hf.fa_rate == 1.0
    k = pashler_k_corrected(4, hf)
    assert math.isnan(k)


def test_pashler_corrected_equals_n_at_perfect():
    """When H=1 and F=0, both K forms equal N."""
    hf = HitsFA(n_change=10, n_no_change=10, n_hit=10, n_fa=0)
    assert pashler_k(4, hf) == 4.0
    assert pashler_k_corrected(4, hf) == 4.0
    assert cowan_k(4, hf) == 4.0


def test_pashler_corrected_larger_when_fa_nonzero():
    """When F > 0, Pashler-corrected K > Cowan K (because dividing by (1-F) < 1)."""
    hf = HitsFA(n_change=100, n_no_change=100, n_hit=90, n_fa=30)
    cowan = pashler_k(4, hf)
    pashler = pashler_k_corrected(4, hf)
    assert pashler > cowan


def _mk(change: list[bool], parsed_yes: list[bool], set_size: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "provider": ["x"] * len(change),
        "model": ["m"] * len(change),
        "condition": ["C1"] * len(change),
        "set_size": [set_size] * len(change),
        "change": change,
        "parsed_yes": parsed_yes,
        "parsable": [True] * len(change),
        "correct": [c == p for c, p in zip(change, parsed_yes)],
    })


def test_perfect_observer_k_equals_n():
    """If H=1 and FA=0, K = N."""
    df = _mk([True, True, True, False, False, False], [True, True, True, False, False, False], set_size=4)
    hf = compute_hits_fa(df)
    assert hf.hit_rate == 1.0
    assert hf.fa_rate == 0.0
    assert pashler_k(4, hf) == 4.0


def test_chance_observer_k_zero():
    """If H = FA = 0.5, K should be ~0."""
    df = _mk(
        [True] * 100 + [False] * 100,
        [True, False] * 50 + [True, False] * 50,
        set_size=4,
    )
    hf = compute_hits_fa(df)
    assert abs(hf.hit_rate - 0.5) < 1e-9
    assert abs(hf.fa_rate - 0.5) < 1e-9
    assert abs(pashler_k(4, hf)) < 1e-9


def test_k_can_be_negative_when_below_chance():
    df = _mk(
        [True, True, True, True, False, False, False, False],
        [False, False, False, False, True, True, True, True],
        set_size=2,
    )
    hf = compute_hits_fa(df)
    assert hf.hit_rate == 0.0
    assert hf.fa_rate == 1.0
    assert pashler_k(2, hf) == -2.0


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(0)
    n = 200
    change = rng.choice([True, False], size=n).tolist()
    # H = 0.8, FA = 0.2
    parsed_yes = []
    for c in change:
        if c:
            parsed_yes.append(rng.random() < 0.8)
        else:
            parsed_yes.append(rng.random() < 0.2)
    df = _mk(change, parsed_yes, set_size=4)
    k, lo, hi = bootstrap_ci(df, set_size=4, n_boot=500, rng_seed=1)
    assert lo <= k <= hi
    assert (hi - lo) > 0


def test_per_model_table_produces_one_row_per_group():
    df = pd.concat([
        _mk([True, True, False, False], [True, True, False, False], set_size=2),
        _mk([True, True, False, False], [True, False, False, True], set_size=4),
    ], ignore_index=True)
    out = per_model_k_table(df, n_boot=100)
    assert len(out) == 2
    assert set(out["set_size"]) == {2, 4}
    # K=2: H=1, FA=0, K=2*(1-0)=2
    # K=4: H=0.5, FA=0.5, K=4*0=0
    row2 = out[out.set_size == 2].iloc[0]
    row4 = out[out.set_size == 4].iloc[0]
    assert row2["K"] == 2.0
    assert row4["K"] == 0.0
