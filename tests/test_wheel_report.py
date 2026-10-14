"""CPU tests for the M3 continuous-report probe (anchor-legend ring + scorer).

Phase 0.5(b). Durable spec (RESEARCH_PLAN.md): render an M=12 anchor-legend ring +
cue the probed square with a double black/white border ring (NO answer leak); the
model answers by INDEX 1..M (>=8-AFC); score circular error offset-invariantly on
the wheel.

Design decision (deviates from the parenthetical "(offset 0)" note): the legend is
rendered with the SAME per-trial wheel_zero_offset as the squares, so (a) the legend
is not itself a fixed categorical anchor across trials, and (b) circular-error scoring
in the canonical frame is exactly offset-invariant (the offset cancels).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from PIL import Image

from vwm.stimuli.colorwheel import wheel_rgb
from vwm.stimuli.palette import BACKGROUND_RGB
from vwm.stimuli.geometry import CANVAS_W, CANVAS_H, SQUARE_PX
from vwm.stimuli.generate import (
    _build_wheel_trial,
    render_report_probe,
    score_report,
    legend_ring_positions,
    legend_anchor_canonical_hue,
    WHEEL_REPORT_M,
)


def _trial():
    return _build_wheel_trial("rpt_n4", set_size=4, change=True,
                              delta_deg=60.0, rng=random.Random(5))


# ---------------- scorer (pure) ----------------

def test_score_report_zero_error_at_exact_anchor():
    assert score_report(1, 0.0) == pytest.approx(0.0)
    assert score_report(7, 0.0) == pytest.approx(180.0)        # M=12 -> index 7 == 180 deg


def test_score_report_is_circular():
    # index 12 -> canonical 330 deg; true 0 deg -> wrap-around distance is 30, not 330
    assert score_report(12, 0.0) == pytest.approx(30.0)


def test_score_report_between_anchors():
    assert score_report(1, 15.0) == pytest.approx(15.0)        # anchor 0 vs true 15


def test_legend_anchor_canonical_hue_even_spacing():
    assert legend_anchor_canonical_hue(0, 12) == pytest.approx(0.0)
    assert legend_anchor_canonical_hue(6, 12) == pytest.approx(180.0)
    assert legend_anchor_canonical_hue(3, 12) == pytest.approx(90.0)


def test_score_report_rejects_out_of_range_index():
    with pytest.raises(ValueError):
        score_report(0, 0.0)
    with pytest.raises(ValueError):
        score_report(WHEEL_REPORT_M + 1, 0.0)


# ---------------- renderer ----------------

def test_report_probe_writes_png_with_legend_region(tmp_path: Path):
    out = render_report_probe(_trial(), tmp_path)
    p = Path(out.report_path)
    assert p.exists() and p.stat().st_size > 0
    img = Image.open(out.report_path)
    assert img.size[0] == CANVAS_W
    assert img.size[1] > CANVAS_H                              # legend region appended below


def test_report_probe_does_not_leak_answer_color(tmp_path: Path):
    tr = _trial()
    out = render_report_probe(tr, tmp_path)
    img = Image.open(out.report_path).convert("RGB")
    cx, cy = tr.positions[tr.probe_idx]
    px = img.getpixel((cx, cy))
    leaked = wheel_rgb((tr.target_hues_deg[tr.probe_idx] + tr.wheel_zero_offset) % 360)
    assert px == BACKGROUND_RGB                                # neutral box, not the color
    assert px != leaked


def test_report_probe_all_squares_neutral(tmp_path: Path):
    tr = _trial()
    out = render_report_probe(tr, tmp_path)
    img = Image.open(out.report_path).convert("RGB")
    for cx, cy in tr.positions:
        assert img.getpixel((cx, cy)) == BACKGROUND_RGB       # no item color anywhere in the array


def test_report_probe_has_double_border_ring_on_cued_square(tmp_path: Path):
    tr = _trial()
    out = render_report_probe(tr, tmp_path)
    img = Image.open(out.report_path).convert("RGB")
    cx, cy = tr.positions[tr.probe_idx]
    half = SQUARE_PX // 2
    box = img.crop((cx - half - 4, cy - half - 4, cx + half + 4, cy + half + 4))
    pix = set(box.getdata())
    assert (0, 0, 0) in pix and (255, 255, 255) in pix        # double black/white ring


def test_report_legend_discs_match_offset_colors(tmp_path: Path):
    tr = _trial()
    out = render_report_probe(tr, tmp_path)
    img = Image.open(out.report_path).convert("RGB")
    pos = legend_ring_positions(WHEEL_REPORT_M)
    assert len(pos) == WHEEL_REPORT_M
    for k in (0, 3, 6, 9):
        cx, cy = pos[k]
        px = img.getpixel((cx, cy))
        expect = wheel_rgb((legend_anchor_canonical_hue(k, WHEEL_REPORT_M)
                            + tr.wheel_zero_offset) % 360)     # legend rotated WITH the trial offset
        assert max(abs(a - b) for a, b in zip(px, expect)) <= 8


def test_build_trial_sets_probe_idx_in_range(tmp_path: Path):
    tr = _trial()
    assert 0 <= tr.probe_idx < tr.set_size
