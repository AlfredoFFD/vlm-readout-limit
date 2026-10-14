"""Smoke tests for the Luck-Vogel stimulus generator."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from PIL import Image

from vwm.stimuli.palette import COLORS, COLOR_NAMES, BACKGROUND_RGB
from vwm.stimuli.geometry import sample_positions, MIN_DIST_PX, CANVAS_W, CANVAS_H
from vwm.stimuli.generate import (
    Trial,
    make_trial,
    render_trial,
    generate_dataset,
    SET_SIZES,
    DIVIDER_PX,
)


# ---------------- palette ----------------

def test_palette_has_nine_unique_colors():
    assert len(COLORS) == 9
    assert len(set(COLORS.values())) == 9
    assert "red" in COLORS and "white" in COLORS and "black" in COLORS


def test_background_is_neutral_gray():
    assert BACKGROUND_RGB == (128, 128, 128)


# ---------------- geometry ----------------

@pytest.mark.parametrize("n", [1, 2, 3, 4, 6, 8])
def test_sample_positions_respects_min_distance_and_quadrant_cap(n):
    rng = random.Random(0)
    pts = sample_positions(n, rng)
    assert len(pts) == n
    # min distance
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            assert (dx * dx + dy * dy) >= MIN_DIST_PX * MIN_DIST_PX
    # quadrant cap (max 2/quadrant)
    cx, cy = CANVAS_W // 2, CANVAS_H // 2
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for x, y in pts:
        q = (0 if x < cx and y < cy else
             1 if x >= cx and y < cy else
             2 if x < cx and y >= cy else 3)
        counts[q] += 1
    for q, c in counts.items():
        assert c <= 2, f"quadrant {q} has {c} > 2 items"


def test_geometry_rejects_oversized_n():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        sample_positions(9, rng)


# ---------------- trial construction ----------------

def test_make_trial_change_picks_unused_color():
    rng = random.Random(123)
    t = make_trial("test_1", set_size=4, change=True, rng=rng)
    assert t.set_size == 4
    assert len(t.colors) == 4
    assert len(set(t.colors)) == 4                  # no dup colors in sample
    assert t.change is True
    assert 0 <= t.changed_idx < 4
    assert t.new_color not in t.colors              # change is detectable
    assert t.new_color in COLOR_NAMES


def test_make_trial_no_change_has_sentinel_fields():
    rng = random.Random(456)
    t = make_trial("test_2", set_size=3, change=False, rng=rng)
    assert t.change is False
    assert t.changed_idx == -1
    assert t.new_color == ""


# ---------------- rendering ----------------

def test_render_trial_writes_three_pngs(tmp_path: Path):
    rng = random.Random(789)
    trial = make_trial("ren_1", set_size=4, change=True, rng=rng)
    out = render_trial(trial, tmp_path, "C1")
    for key in ("sample_path", "test_path", "combined_path"):
        p = Path(getattr(out, key))
        assert p.exists() and p.stat().st_size > 0
    sample = Image.open(out.sample_path)
    test = Image.open(out.test_path)
    combined = Image.open(out.combined_path)
    assert sample.size == (CANVAS_W, CANVAS_H)
    assert test.size == (CANVAS_W, CANVAS_H)
    assert combined.size == (CANVAS_W * 2 + DIVIDER_PX, CANVAS_H)


def test_change_trial_differs_in_pixels(tmp_path: Path):
    """If we say it's a change trial, the rendered sample and test PNGs must differ."""
    rng = random.Random(42)
    trial = make_trial("diff_1", set_size=4, change=True, rng=rng)
    trial = render_trial(trial, tmp_path, "C1")
    s = Image.open(trial.sample_path).convert("RGB").tobytes()
    t = Image.open(trial.test_path).convert("RGB").tobytes()
    assert s != t


def test_no_change_trial_is_pixel_identical(tmp_path: Path):
    rng = random.Random(99)
    trial = make_trial("same_1", set_size=4, change=False, rng=rng)
    trial = render_trial(trial, tmp_path, "C1")
    s = Image.open(trial.sample_path).convert("RGB").tobytes()
    t = Image.open(trial.test_path).convert("RGB").tobytes()
    assert s == t


# ---------------- determinism + manifest ----------------

def test_generate_dataset_deterministic(tmp_path: Path):
    """Same seed -> identical manifest contents."""
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    generate_dataset(out1, "C1", seed=20260521, trials_per_set_size=4)
    generate_dataset(out2, "C1", seed=20260521, trials_per_set_size=4)
    m1 = json.loads((out1 / "manifest_C1.json").read_text())
    m2 = json.loads((out2 / "manifest_C1.json").read_text())
    # Compare trial specs (ignore paths because they include tmpdirs)
    def _strip(m):
        for tr in m["trials"]:
            for k in ("sample_path", "test_path", "combined_path"):
                tr.pop(k, None)
        return m
    assert _strip(m1) == _strip(m2)


def test_generate_dataset_change_rate_is_50pct(tmp_path: Path):
    info = generate_dataset(tmp_path, "C1", seed=1, trials_per_set_size=100)
    m = json.loads(Path(info["manifest_path"]).read_text())
    by_n = {}
    for tr in m["trials"]:
        by_n.setdefault(tr["set_size"], []).append(tr["change"])
    for n, flags in by_n.items():
        change_rate = sum(flags) / len(flags)
        assert change_rate == 0.5, f"n={n}: change rate {change_rate} != 0.5"


def test_generate_dataset_count_matches_six_set_sizes(tmp_path: Path):
    info = generate_dataset(tmp_path, "C2", seed=1, trials_per_set_size=10)
    assert info["n_trials"] == len(SET_SIZES) * 10
