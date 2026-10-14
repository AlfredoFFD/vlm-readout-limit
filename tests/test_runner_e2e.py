"""End-to-end test of the experiment runner using the MockClient.

This validates the entire (manifest -> driver -> JSONL -> ledger -> K-table)
pipeline without needing any external API keys or model weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vwm.api import get_client
from vwm.experiments.runner import ExperimentConfig, run_condition
from vwm.experiments.ledger import append, read_ledger, LedgerRow, now_iso, short_commit
from vwm.experiments.driver import _summarize
from vwm.stimuli.generate import generate_dataset


@pytest.fixture(scope="module")
def small_manifests(tmp_path_factory):
    """Generate a tiny stimulus set in a temp dir (4 trials per set size)."""
    out = tmp_path_factory.mktemp("stim")
    info_c1 = generate_dataset(out, "C1", seed=42, trials_per_set_size=4)
    info_c2 = generate_dataset(out, "C2", seed=42, trials_per_set_size=4)
    return {"C1": Path(info_c1["manifest_path"]), "C2": Path(info_c2["manifest_path"])}


def test_mock_perfect_runs_to_completion_with_high_accuracy(small_manifests, tmp_path):
    out = tmp_path / "mock_perfect_C1.jsonl"
    cfg = ExperimentConfig(
        provider="mock:perfect",
        condition="C1",
        manifest_path=small_manifests["C1"],
        out_path=out,
    )
    run_condition(cfg)
    mean_acc, k8, fail_rate, n = _summarize(out)
    # 6 set sizes * 4 trials = 24 trials
    assert n == 24
    assert fail_rate == 0.0
    assert mean_acc >= 0.95          # perfect mock + pixel-diff -> ceiling
    # K at N=8 should be high (close to 8 if change/no-change perfectly separated)
    assert k8 >= 4.0


def test_mock_chance_runs_to_completion(small_manifests, tmp_path):
    out = tmp_path / "mock_chance_C1.jsonl"
    cfg = ExperimentConfig(
        provider="mock:chance",
        condition="C1",
        manifest_path=small_manifests["C1"],
        out_path=out,
    )
    run_condition(cfg)
    mean_acc, k8, fail_rate, n = _summarize(out)
    assert n == 24
    assert fail_rate == 0.0
    # Chance accuracy ~= 0.5 +- 0.2 on 24 trials (binomial(24, 0.5) 95% range)
    assert 0.25 < mean_acc < 0.75


def test_two_turn_path_works_for_c2(small_manifests, tmp_path):
    out = tmp_path / "mock_perfect_C2.jsonl"
    cfg = ExperimentConfig(
        provider="mock:perfect",
        condition="C2",
        manifest_path=small_manifests["C2"],
        out_path=out,
    )
    run_condition(cfg)
    mean_acc, k8, fail_rate, n = _summarize(out)
    assert n == 24
    assert fail_rate == 0.0
    assert mean_acc >= 0.95
