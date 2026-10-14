"""CPU tests for the CIELAB color-wheel graded-magnitude stimuli."""

import random

import pytest

from vwm.stimuli.colorwheel import (
    wheel_rgb, angular_distance, delta_e, WHEEL_RADIUS,
    make_wheel_trial, FOIL_DISTANCES_DEG,
)

POS = [(10, 10), (100, 10), (10, 100), (100, 100), (200, 10), (200, 100)]


def test_wheel_rgb_valid():
    for ang in (0, 45, 137, 359):
        r, g, b = wheel_rgb(ang)
        assert all(0 <= c <= 255 and isinstance(c, int) for c in (r, g, b))


def test_angular_distance_wraps():
    assert angular_distance(350, 10) == 20
    assert angular_distance(10, 350) == 20
    assert angular_distance(0, 180) == 180
    assert angular_distance(90, 90) == 0


def test_delta_e_monotone_and_diameter():
    # near hues are more similar than far hues (CIE76 on the equal-L wheel)
    assert delta_e(100, 105) < delta_e(100, 140) < delta_e(100, 190)
    # opposite hues are a full diameter apart: chord = 2 * radius
    assert delta_e(0, 180) == pytest.approx(2 * WHEEL_RADIUS, abs=1e-6)


def test_change_trial_rotates_exactly_one_by_the_requested_distance():
    rng = random.Random(0)
    t = make_wheel_trial("t", 4, change=True, rng=rng, positions=POS[:4],
                         change_distance_deg=40.0)
    diffs = [angular_distance(a, b) for a, b in zip(t.angles, t.test_angles)]
    moved = [d for d in diffs if d > 1e-6]
    assert len(moved) == 1                      # exactly one square changed
    assert moved[0] == pytest.approx(40.0, abs=1e-6)
    assert t.change_distance_deg == 40.0
    assert t.changed_idx >= 0


def test_no_change_trial_is_identical():
    rng = random.Random(1)
    t = make_wheel_trial("t", 4, change=False, rng=rng, positions=POS[:4])
    assert t.test_angles == t.angles
    assert t.changed_idx == -1 and t.change_distance_deg == 0.0


def test_sample_hues_separated():
    rng = random.Random(2)
    t = make_wheel_trial("t", 6, change=True, rng=rng, positions=POS[:6],
                         change_distance_deg=80.0, min_sample_sep_deg=40.0)
    for i in range(len(t.angles)):
        for j in range(i + 1, len(t.angles)):
            assert angular_distance(t.angles[i], t.angles[j]) >= 40.0 - 1e-9


def test_deterministic_from_seed():
    a = make_wheel_trial("t", 4, True, random.Random(7), POS[:4], 20.0).to_dict()
    b = make_wheel_trial("t", 4, True, random.Random(7), POS[:4], 20.0).to_dict()
    assert a == b


def test_foil_distances_drawn_when_unspecified():
    rng = random.Random(3)
    for _ in range(20):
        t = make_wheel_trial("t", 3, True, rng, POS[:3])
        assert t.change_distance_deg in FOIL_DISTANCES_DEG


def test_change_trial_test_hue_separated_from_other_items():
    """On a change trial the rotated square must remain >= sep from every OTHER item,
    so it stays individuable and the change magnitude is not confounded by a near-collision.
    (Regression for the silent collision-fallback found in adversarial review.)"""
    rng = random.Random(0)
    for _ in range(400):
        n = rng.choice([2, 3, 4, 6])
        d = float(rng.choice(FOIL_DISTANCES_DEG))
        t = make_wheel_trial("t", n, True, rng, POS[:n], change_distance_deg=d)
        ci = t.changed_idx
        for j in range(n):
            if j != ci:
                assert angular_distance(t.test_angles[ci], t.angles[j]) >= 40.0 - 1e-9
