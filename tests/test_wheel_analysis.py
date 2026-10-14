"""CPU tests for the wheel behavioral analysis (Tier-2 SDT/precision metrics).

Operates on the runner's JSONL rows. Pins the plan's falsifier-bearing summaries:
miss-rate vs delta (resource/TCC smooth decline vs slot step), curved SDT ROC vs the
linear high-threshold chord, d'(N)/K(N), and report circular precision(N) + swaps.
"""

from __future__ import annotations

import math

import pytest

from vwm.analysis.roc import dprime
from vwm.analysis.wheel_analysis import (
    miss_rate_by_delta,
    detection_by_set_size,
    report_error_by_set_size,
    report_swap_rate,
    summarize_change,
    summarize_report,
)
from vwm.stimuli.generate import legend_anchor_canonical_hue, WHEEL_REPORT_M


def _chg(set_size, change, delta, said_yes, p_yes=None):
    return {
        "task": "change", "set_size": set_size, "change": change, "delta_deg": delta,
        "parsed_yes": said_yes, "parsable": said_yes is not None, "p_yes": p_yes,
        "error": None,
    }


def _rep(set_size, reported_index, report_hue_deg, trial_id="t"):
    return {
        "task": "report", "set_size": set_size, "trial_id": trial_id,
        "reported_index": reported_index, "report_hue_deg": report_hue_deg,
        "report_parsable": reported_index is not None, "error": None,
    }


# ---------------- change: miss-rate vs delta ----------------

def test_miss_rate_declines_with_delta():
    rows = [_chg(2, True, 15, False) for _ in range(10)]   # near foils -> all missed
    rows += [_chg(2, True, 180, True) for _ in range(10)]  # far foils -> all caught
    mr = miss_rate_by_delta(rows)
    assert mr[2][15] == pytest.approx(1.0)
    assert mr[2][180] == pytest.approx(0.0)


def test_miss_rate_ignores_catch_and_unparsable():
    rows = [_chg(2, False, 0, True), _chg(2, True, 90, None)]   # catch + unparsable only
    mr = miss_rate_by_delta(rows)
    assert mr.get(2, {}).get(90) is None or 90 not in mr.get(2, {})


# ---------------- change: hit/fa/d'/K/AUC/curvature ----------------

def test_detection_hit_fa_dprime_k():
    rows = [_chg(4, True, 90, i < 6) for i in range(8)]    # hit 6/8 = .75
    rows += [_chg(4, False, 0, i < 2) for i in range(8)]   # fa 2/8 = .25
    d = detection_by_set_size(rows)[4]
    assert d["hit"] == pytest.approx(0.75)
    assert d["fa"] == pytest.approx(0.25)
    assert d["k"] == pytest.approx(4 * (0.75 - 0.25))
    assert d["d_prime"] == pytest.approx(dprime(0.75, 0.25, 8, 8))


def test_detection_auc_and_curvature():
    # graded p_yes perfectly orders the classes (AUC 1) while the hard YES/NO operating
    # point is intermediate (H=.7, F=.3) -> the curved ROC bows above the slot chord.
    rows = [_chg(2, True, 90, said_yes=(i < 7), p_yes=0.55 + 0.04 * i) for i in range(10)]
    rows += [_chg(2, False, 0, said_yes=(i < 3), p_yes=0.05 + 0.04 * i) for i in range(10)]
    d = detection_by_set_size(rows)[2]
    assert d["hit"] == pytest.approx(0.7)
    assert d["fa"] == pytest.approx(0.3)
    assert d["auc"] == pytest.approx(1.0)
    assert d["auc_ht"] == pytest.approx(0.7, abs=1e-9)        # trapezoid of (0,0)->(.3,.7)->(1,1)
    assert d["curvature"] == pytest.approx(0.3, abs=1e-9)     # auc - auc_ht > 0


def test_detection_auc_none_without_pyes():
    rows = [_chg(3, True, 90, True) for _ in range(4)] + [_chg(3, False, 0, False) for _ in range(4)]
    d = detection_by_set_size(rows)[3]
    assert d["auc"] is None and d["curvature"] is None       # no graded signal recorded


# ---------------- report: precision + swaps ----------------

def test_report_error_bias_and_circular_sd():
    # signed errors of +30 and -30 about true hue 0: bias ~0, circular SD ~30.7 deg
    rows = [_rep(3, reported_index=2, report_hue_deg=0.0),     # legend 2 -> 30 deg -> +30
            _rep(3, reported_index=12, report_hue_deg=0.0)]    # legend 12 -> 330 deg -> -30
    st = report_error_by_set_size(rows)[3]
    assert st["n"] == 2
    assert st["bias_deg"] == pytest.approx(0.0, abs=1e-6)
    assert st["circular_sd_deg"] == pytest.approx(30.7, abs=0.5)
    assert st["rmse_deg"] == pytest.approx(30.0, abs=1e-6)


def test_report_error_zero_when_exact():
    rows = [_rep(1, reported_index=1, report_hue_deg=0.0),      # legend 1 -> 0 deg, true 0
            _rep(1, reported_index=4, report_hue_deg=90.0)]     # legend 4 -> 90 deg, true 90
    st = report_error_by_set_size(rows)[1]
    assert st["circular_sd_deg"] == pytest.approx(0.0, abs=1e-6)
    assert st["rmse_deg"] == pytest.approx(0.0, abs=1e-6)


def test_report_swap_rate_counts_nontarget_matches():
    # trial has items at canonical {0, 100}; probed = item 0 (true 0). A report of legend 4
    # (90 deg) is far from the target (0) but near the OTHER item (100) -> a swap.
    manifest = {"trials": [
        {"trial_id": "s1", "set_size": 2, "probe_idx": 0, "target_hues_deg": [0.0, 100.0]},
        {"trial_id": "s2", "set_size": 2, "probe_idx": 0, "target_hues_deg": [0.0, 100.0]},
    ]}
    rows = [_rep(2, reported_index=4, report_hue_deg=0.0, trial_id="s1"),   # ~90 -> swap to item@100
            _rep(2, reported_index=1, report_hue_deg=0.0, trial_id="s2")]   # ~0  -> correct (no swap)
    sr = report_swap_rate(rows, manifest)
    assert sr[2] == pytest.approx(0.5)


# ---------------- bundles ----------------

def test_summaries_bundle_keys():
    chg = [_chg(2, True, 90, True, p_yes=0.9), _chg(2, False, 0, False, p_yes=0.1)]
    sc = summarize_change(chg)
    assert "by_delta" in sc and "by_set_size" in sc
    rep = [_rep(2, 1, 0.0, "t")]
    sr = summarize_report(rep)
    assert "by_set_size" in sr


def test_analysis_task_parameter_for_simultaneous_rows():
    """Regression (cross-model review): simultaneous rows were silently
    filtered out by the task=='change' hardcode, making the decisive
    perception-baseline diagnostics empty."""
    from vwm.analysis.wheel_analysis import miss_rate_by_delta, detection_by_set_size
    rows = [
        {"task": "simultaneous", "set_size": 2, "change": True, "delta_deg": 30,
         "parsed_yes": True, "p_yes": None},
        {"task": "simultaneous", "set_size": 2, "change": True, "delta_deg": 30,
         "parsed_yes": False, "p_yes": None},
        {"task": "simultaneous", "set_size": 2, "change": False, "delta_deg": 0,
         "parsed_yes": False, "p_yes": None},
    ]
    assert miss_rate_by_delta(rows) == {}                      # default excludes
    mr = miss_rate_by_delta(rows, task="simultaneous")
    assert mr[2][30] == 0.5
    det = detection_by_set_size(rows, task="simultaneous")
    assert det[2]["hit"] == 0.5 and det[2]["fa"] == 0.0
