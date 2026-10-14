"""CPU tests for the CIELAB color-wheel DATASET generator (manifest schema v2).

Phase 0.5(a): wire `colorwheel.make_wheel_trial` into a full graded-magnitude
change-detection dataset with a schema-v2, `design:"wheel"` manifest. These tests
pin the durable spec from research `wf_f41b1976-1e4` (RESEARCH_PLAN.md):
  - per-trial set_size hues, pairwise circular sep >= 40 deg
  - change = rotate ONE square by delta in {15,30,60,90,135,180}
  - delta=0 catch ~= 30% of trials
  - per-trial wheel_zero_offset ~ U{0..359} (anti-categorical-anchoring)
  - manifest v2 fields: target_hues_deg, changed_idx, delta_deg, wheel_zero_offset, test_hue_deg
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from vwm.stimuli.colorwheel import angular_distance, FOIL_DISTANCES_DEG
from vwm.stimuli.generate import (
    generate_wheel_dataset,
    WHEEL_FOIL_DISTANCES,
    WHEEL_CATCH_RATE,
    WHEEL_MIN_SAMPLE_SEP_DEG,
)

# v2 manifest fields every wheel trial must carry (the durable contract)
_REQUIRED_V2_FIELDS = (
    "target_hues_deg",
    "changed_idx",
    "delta_deg",
    "wheel_zero_offset",
    "test_hue_deg",
)


def _small_manifest(tmp_path: Path, **kw) -> dict:
    kw.setdefault("set_sizes", (2, 4))
    kw.setdefault("trials_per_cell", 2)
    info = generate_wheel_dataset(out_root=tmp_path, seed=20260604, **kw)
    return json.loads(Path(info["manifest_path"]).read_text())


def test_wheel_manifest_has_v2_schema_and_design_tag(tmp_path: Path):
    m = _small_manifest(tmp_path)
    assert m["schema_version"] == 2
    assert m["design"] == "wheel"
    assert len(m["trials"]) > 0
    for tr in m["trials"]:
        for f in _REQUIRED_V2_FIELDS:
            assert f in tr, f"trial missing v2 field {f!r}"


def test_wheel_change_trials_use_only_foil_deltas(tmp_path: Path):
    m = _small_manifest(tmp_path)
    for tr in m["trials"]:
        if tr["change"]:
            assert tr["delta_deg"] in WHEEL_FOIL_DISTANCES
        else:
            assert tr["delta_deg"] == 0
            assert tr["changed_idx"] == -1
            assert tr["test_hue_deg"] is None


def test_wheel_grid_covers_every_set_size_x_delta_cell(tmp_path: Path):
    m = _small_manifest(tmp_path)
    cells = {(tr["set_size"], tr["delta_deg"]) for tr in m["trials"] if tr["change"]}
    expected = {(n, d) for n in (2, 4) for d in WHEEL_FOIL_DISTANCES}
    assert cells == expected


def test_wheel_catch_rate_is_about_30pct(tmp_path: Path):
    # enough trials per cell that the fraction is stable
    m = _small_manifest(tmp_path, trials_per_cell=10)
    n_catch = sum(1 for tr in m["trials"] if not tr["change"])
    frac = n_catch / len(m["trials"])
    assert WHEEL_CATCH_RATE - 0.06 <= frac <= WHEEL_CATCH_RATE + 0.06, f"catch frac {frac}"


def test_wheel_sample_hues_pairwise_separated(tmp_path: Path):
    m = _small_manifest(tmp_path)
    for tr in m["trials"]:
        hues = tr["target_hues_deg"]
        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                assert angular_distance(hues[i], hues[j]) >= WHEEL_MIN_SAMPLE_SEP_DEG - 1e-9


def test_wheel_change_trials_test_array_pairwise_separated(tmp_path: Path):
    """Every CHANGE trial's TEST array (sample with one square rotated) must keep all
    items >= 40 deg apart -- not just the sample array."""
    m = _small_manifest(tmp_path, trials_per_cell=3)
    for tr in m["trials"]:
        if not tr["change"]:
            continue
        hues = tr["test_hues_deg"]
        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                assert angular_distance(hues[i], hues[j]) >= WHEEL_MIN_SAMPLE_SEP_DEG - 1e-9


def test_wheel_zero_offset_in_range_and_varies(tmp_path: Path):
    m = _small_manifest(tmp_path, trials_per_cell=4)
    offs = [tr["wheel_zero_offset"] for tr in m["trials"]]
    assert all(0 <= o < 360 for o in offs)
    assert len(set(offs)) > 1, "wheel_zero_offset must vary across trials"


def test_wheel_test_hue_consistent_with_delta(tmp_path: Path):
    m = _small_manifest(tmp_path)
    for tr in m["trials"]:
        if tr["change"]:
            base = tr["target_hues_deg"][tr["changed_idx"]]
            assert angular_distance(base, tr["test_hue_deg"]) == pytest.approx(
                tr["delta_deg"], abs=1e-6
            )


def test_wheel_render_change_differs_catch_identical(tmp_path: Path):
    m = _small_manifest(tmp_path)
    seen_change = seen_catch = False
    for tr in m["trials"]:
        s = Image.open(tr["sample_path"]).convert("RGB").tobytes()
        t = Image.open(tr["test_path"]).convert("RGB").tobytes()
        if tr["change"]:
            assert s != t
            seen_change = True
        else:
            assert s == t
            seen_catch = True
    assert seen_change and seen_catch


def test_wheel_render_uses_offset_not_canonical(tmp_path: Path):
    """The rendered color must reflect (hue + wheel_zero_offset), not the raw canonical
    hue -- otherwise the anti-anchoring offset is recorded but never applied."""
    from vwm.stimuli.colorwheel import wheel_rgb

    m = _small_manifest(tmp_path)
    tr = next(t for t in m["trials"] if not t["change"])  # catch: sample==test, simple
    off = tr["wheel_zero_offset"]
    img = Image.open(tr["sample_path"]).convert("RGB")
    # sample a pixel at the first square's center; it must match the OFFSET color
    cx, cy = tr["positions"][0]
    px = img.getpixel((cx, cy))
    expect_off = wheel_rgb((tr["target_hues_deg"][0] + off) % 360)
    expect_raw = wheel_rgb(tr["target_hues_deg"][0] % 360)
    # allow JPEG-free PNG exactness, small tol for antialias at center
    assert max(abs(a - b) for a, b in zip(px, expect_off)) <= 6
    if off % 360 != 0:
        assert px != expect_raw or expect_off == expect_raw


def test_wheel_dataset_deterministic(tmp_path: Path):
    m1 = _small_manifest(tmp_path / "a")
    m2 = _small_manifest(tmp_path / "b")

    def _strip(m):
        for tr in m["trials"]:
            for k in ("sample_path", "test_path", "combined_path", "report_path"):
                tr.pop(k, None)
        return m

    assert _strip(m1) == _strip(m2)
