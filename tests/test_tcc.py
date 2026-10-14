"""TDD for the Tier-2 TCC engine (vwm/analysis/tcc.py).

Implements results/TCC_RESEARCH.md. Conventions pinned here (the equations doc
had a 360-vs-180 ambiguity — resolved as 360 channels, degrees -179..180,
target at 0):
  - degrees_grid() = np.arange(-179, 181); index 179 <-> 0 deg (the target).
  - f_vec aligned to degrees_grid, f(0)=1, f symmetric, min 0 after norm.
Property tests validate the generative core against known TCC behavior, the
exact channel-likelihood against Monte Carlo, and d' recovery; the max-rule
change-detection MC is the falsifier-calibration machinery required by the
assembly review (closed-form AUC is only valid where MC says it is).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from vwm.analysis.tcc import (
    degrees_grid,
    synthetic_f,
    f_from_cosine,
    simulate_report_errors,
    response_pmf,
    fit_dprime,
    change_dprime,
    change_auc_closed_form,
    simulate_change_auc_maxrule,
    anchor_channels,
    fit_dprime_nafc,
    simulate_report_12afc,
    predict_all,
)


RNG = np.random.default_rng


# ---------------------------------------------------------------- convention

def test_convention_pinned():
    deg = degrees_grid()
    assert len(deg) == 360
    assert deg[0] == -179 and deg[-1] == 180
    assert deg[179] == 0
    f = synthetic_f(tau=50.0)
    assert len(f) == 360
    assert f[179] == pytest.approx(1.0)
    assert f.min() == pytest.approx(0.0)
    # symmetry where both signs exist
    assert f[179 + 90] == pytest.approx(f[179 - 90], abs=1e-9)


def test_f_from_cosine_normalizes():
    deg = degrees_grid()
    s = 0.3 + 0.7 * np.cos(np.deg2rad(deg))          # cosine-similarity-like
    f = f_from_cosine(s, target_idx=179)
    assert f[179] == pytest.approx(1.0)
    assert f.min() == pytest.approx(0.0)
    assert np.all(f >= 0) and np.all(f <= 1)
    with pytest.raises(ValueError):
        f_from_cosine(np.roll(s, 5), target_idx=179)  # max not at target


# ------------------------------------------------------------ generative core

def test_dprime_zero_gives_uniform_errors():
    errs = simulate_report_errors(0.0, synthetic_f(), rng=RNG(0), n_trials=20000)
    # uniform circular errors: E|e| = 90 deg
    assert abs(np.mean(np.abs(errs)) - 90.0) < 3.0


def test_high_dprime_concentrates_errors():
    errs = simulate_report_errors(4.0, synthetic_f(), rng=RNG(1), n_trials=5000)
    assert np.median(np.abs(errs)) < 30.0


def test_error_magnitude_monotone_in_dprime():
    means = [np.mean(np.abs(simulate_report_errors(d, synthetic_f(),
                                                   rng=RNG(2), n_trials=8000)))
             for d in (1.0, 2.0, 3.0)]
    assert means[0] > means[1] > means[2]


def test_response_pmf_matches_monte_carlo():
    f = synthetic_f()
    pmf = response_pmf(2.0, f)
    assert pmf.shape == (360,)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-6)
    errs = simulate_report_errors(2.0, f, rng=RNG(3), n_trials=200000)
    hist = np.bincount((errs + 179).astype(int), minlength=360) / len(errs)
    assert 0.5 * np.abs(pmf - hist).sum() < 0.02      # total-variation distance


def test_fit_dprime_recovers_truth():
    f = synthetic_f()
    errs = simulate_report_errors(2.0, f, rng=RNG(4), n_trials=20000)
    d_hat = fit_dprime(errs, f)
    assert d_hat == pytest.approx(2.0, abs=0.1)


# ------------------------------------------- change detection + calibration

def test_change_dprime_limits():
    f = synthetic_f()
    assert change_dprime(2.0, f, delta_deg=0) == pytest.approx(0.0)
    assert change_dprime(2.0, f, delta_deg=180) == pytest.approx(2.0)


def test_closed_form_auc_matches_mc_at_n1():
    f = synthetic_f()
    for delta in (30, 90, 180):
        closed = change_auc_closed_form(2.0, f, delta_deg=delta)
        mc = simulate_change_auc_maxrule(2.0, f, delta_deg=delta, set_size=1,
                                         rng=RNG(5), n_trials=60000)
        assert abs(closed - mc) < 0.015, f"delta={delta}: {closed} vs {mc}"


def test_lr_observer_dominates_minrule_where_it_matters():
    """The LR (softmin) rule is the true ideal observer; the min-rule is a
    heuristic. In the small-delta/high-N cells the gap is ~0.01 AUC (assembly
    review, independently reproduced)."""
    from vwm.analysis.tcc import simulate_change_auc_lr
    f = synthetic_f()
    lr = simulate_change_auc_lr(2.0, f, delta_deg=30, set_size=6,
                                rng=RNG(20), n_trials=120000)
    mn = simulate_change_auc_maxrule(2.0, f, delta_deg=30, set_size=6,
                                     rng=RNG(21), n_trials=120000)
    assert lr > mn + 0.004, f"LR {lr} should beat min-rule {mn}"


def test_lr_observer_matches_closed_form_at_n1():
    from vwm.analysis.tcc import simulate_change_auc_lr
    f = synthetic_f()
    for delta in (30, 90, 180):
        closed = change_auc_closed_form(2.0, f, delta_deg=delta)
        mc = simulate_change_auc_lr(2.0, f, delta_deg=delta, set_size=1,
                                    rng=RNG(22), n_trials=60000)
        assert abs(closed - mc) < 0.015


def test_maxrule_auc_monotone_in_delta_and_setsize():
    f = synthetic_f()
    aucs_d = [simulate_change_auc_maxrule(2.0, f, delta_deg=d, set_size=4,
                                          rng=RNG(6), n_trials=30000)
              for d in (15, 60, 180)]
    assert aucs_d[0] < aucs_d[1] < aucs_d[2]
    aucs_n = [simulate_change_auc_maxrule(2.0, f, delta_deg=90, set_size=n,
                                          rng=RNG(7), n_trials=30000)
              for n in (1, 4, 6)]
    assert aucs_n[0] > aucs_n[1] > aucs_n[2]


# --------------------------------------------------- 12-AFC report (M3 task)

def test_anchor_channels_shape():
    idx = anchor_channels(phase_deg=7.0, m=12)
    assert len(idx) == 12
    assert len(set(idx)) == 12
    assert all(0 <= i < 360 for i in idx)


def test_fit_dprime_nafc_recovers_truth_from_quantized_reports():
    f = synthetic_f()
    resp = simulate_report_12afc(2.0, f, rng=RNG(8), n_trials=15000)
    d_hat = fit_dprime_nafc(resp, f)
    assert d_hat == pytest.approx(2.0, abs=0.15)


# --------------------------------------------------------------- the contract

def test_predict_all_is_parameter_free():
    sig = inspect.signature(predict_all)
    required = [p for p in sig.parameters.values() if p.default is p.empty]
    assert [p.name for p in required] == ["dprime", "f_vec"], (
        "predict_all must accept nothing beyond (dprime, f_vec) — the "
        "parameter-free contract from TCC_RESEARCH.md Sec 5")
    out = predict_all(2.0, synthetic_f())
    assert set(out) >= {"error_pmf", "auc_by_delta_closed", "report_circ_sd"}
