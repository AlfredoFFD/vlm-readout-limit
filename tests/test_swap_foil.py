"""TDD for the SWAP-FOIL condition (PLAN_TOPTIER P2 keystone).

A swap trial exchanges the colors of two squares: the color SET is preserved
exactly, so any observer relying on set membership sees "no change"; only an
observer that binds color to location can detect it. The behavioral verdict on
swap trials decides the paper's thesis (bag-of-features vs binding).
"""

from __future__ import annotations

import json
import random

import pytest

from vwm.stimuli.colorwheel import make_swap_trial, angular_distance
from vwm.stimuli.generate import generate_swap_dataset


POSITIONS_4 = [(60, 60), (200, 80), (120, 200), (320, 220)]


def test_swap_trial_preserves_color_set():
    rng = random.Random(0)
    t = make_swap_trial("s0", 4, rng, POSITIONS_4)
    assert t.change is True and t.swap is True
    i, j = t.swapped_pair
    assert i != j and 0 <= i < 4 and 0 <= j < 4
    # test array = sample array with colors at i and j exchanged
    assert t.test_angles_deg[i] == t.angles_deg[j]
    assert t.test_angles_deg[j] == t.angles_deg[i]
    for k in range(4):
        if k not in (i, j):
            assert t.test_angles_deg[k] == t.angles_deg[k]
    # multiset of colors identical
    assert sorted(t.test_angles_deg) == sorted(t.angles_deg)


def test_swap_pair_is_discriminable():
    """The two swapped colors must differ by >= the min separation, else the
    swap is undetectable by ANY observer and the trial is uninformative."""
    rng = random.Random(1)
    for k in range(50):
        t = make_swap_trial(f"s{k}", 3, rng, POSITIONS_4[:3], min_swap_sep_deg=40.0)
        i, j = t.swapped_pair
        assert angular_distance(t.angles_deg[i], t.angles_deg[j]) >= 40.0


def test_swap_requires_at_least_two_items():
    with pytest.raises(ValueError):
        make_swap_trial("s", 1, random.Random(0), POSITIONS_4[:1])


def test_generate_swap_dataset(tmp_path):
    info = generate_swap_dataset(tmp_path, seed=7, set_sizes=(2, 3, 4),
                                 trials_per_cell=6, catch_rate=0.5, render=True)
    m = json.loads((tmp_path / "manifest_swap.json").read_text())
    trials = m["trials"]
    assert m["design"] == "wheel"
    assert m["condition"] == "swap"
    swaps = [t for t in trials if t["change"]]
    catches = [t for t in trials if not t["change"]]
    assert len(swaps) == 3 * 6
    assert len(catches) >= 1
    for t in swaps:
        assert t["swap"] is True and len(t["swapped_pair"]) == 2
        # runner-compatible fields
        for k in ("trial_id", "set_size", "sample_path", "test_path",
                  "delta_deg", "wheel_zero_offset", "positions"):
            assert k in t, k
        # swap trials carry delta_deg = angular distance between swapped colors
        i, j = t["swapped_pair"]
        assert t["delta_deg"] == pytest.approx(
            angular_distance(t["target_hues_deg"][i], t["target_hues_deg"][j]), abs=0.01)
    # rendered files exist
    from pathlib import Path
    assert Path(swaps[0]["sample_path"]).exists()
    assert Path(swaps[0]["test_path"]).exists()
    assert info["n_trials"] == len(trials)


def test_generate_swap_dataset_deterministic(tmp_path):
    a = generate_swap_dataset(tmp_path / "a", seed=9, set_sizes=(3,),
                              trials_per_cell=4, render=False)
    b = generate_swap_dataset(tmp_path / "b", seed=9, set_sizes=(3,),
                              trials_per_cell=4, render=False)
    ma = json.loads((tmp_path / "a" / "manifest_swap.json").read_text())["trials"]
    mb = json.loads((tmp_path / "b" / "manifest_swap.json").read_text())["trials"]
    for ta, tb in zip(ma, mb):
        assert ta["target_hues_deg"] == tb["target_hues_deg"]
        assert ta.get("swapped_pair") == tb.get("swapped_pair")
