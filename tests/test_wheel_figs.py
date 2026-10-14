"""Smoke tests for wheel behavioral figures (render to a temp dir, assert files exist).

Deterministic synthetic rows (fixed seeds, no hash()) and existence-only assertions, so
these do not inherit the statistical flakiness of test_figures.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vwm.figures.wheel_figs import (
    plot_miss_rate_by_delta,
    plot_roc,
    plot_precision,
    plot_report_error_hist,
    make_wheel_figures,
)
from vwm.analysis.wheel_analysis import (
    miss_rate_by_delta,
    detection_by_set_size,
    roc_points_by_set_size,
    report_error_by_set_size,
    report_signed_errors,
)

DELTAS = (15, 30, 60, 90, 135, 180)


def _change_rows():
    rows = []
    for n in (2, 4):
        for d in DELTAS:
            # hit prob rises with delta; record graded p_yes too
            phit = min(0.98, 0.2 + d / 200.0)
            for i in range(10):
                said = i < round(phit * 10)
                rows.append({"task": "change", "set_size": n, "change": True, "delta_deg": float(d),
                             "parsed_yes": said, "parsable": True,
                             "p_yes": 0.55 + 0.04 * i if said else 0.45 - 0.04 * i, "error": None})
        for i in range(20):                                   # catch trials
            said = i < 2
            rows.append({"task": "change", "set_size": n, "change": False, "delta_deg": 0,
                         "parsed_yes": said, "parsable": True,
                         "p_yes": 0.1 + 0.01 * i, "error": None})
    return rows


def _report_rows():
    rows = []
    for n in (2, 4):
        for t in range(20):
            true = (t * 17) % 360
            idx = 1 + (round(true / 30) % 12)                # near-correct legend pick
            rows.append({"task": "report", "set_size": n, "trial_id": f"n{n}_t{t}",
                         "reported_index": idx, "report_hue_deg": float(true),
                         "report_parsable": True, "error": None})
    return rows


def test_plot_miss_rate_by_delta(tmp_path: Path):
    out = plot_miss_rate_by_delta(miss_rate_by_delta(_change_rows()), tmp_path / "miss.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_roc(tmp_path: Path):
    rows = _change_rows()
    out = plot_roc(roc_points_by_set_size(rows), detection_by_set_size(rows), tmp_path / "roc.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_precision(tmp_path: Path):
    out = plot_precision(report_error_by_set_size(_report_rows()), tmp_path / "prec.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_report_error_hist(tmp_path: Path):
    out = plot_report_error_hist(report_signed_errors(_report_rows()), tmp_path / "hist.png")
    assert out.exists() and out.stat().st_size > 0


def test_make_wheel_figures_bundles(tmp_path: Path):
    paths = make_wheel_figures(_change_rows(), _report_rows(), tmp_path)
    for key in ("miss_rate", "roc", "precision", "report_hist"):
        assert key in paths and Path(paths[key]).exists() and Path(paths[key]).stat().st_size > 0
