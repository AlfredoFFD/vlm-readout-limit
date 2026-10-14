"""CPU end-to-end tests for the wheel runner (M3 report + M4 change-detection).

Uses a dependency-injected PERFECT-ORACLE fake client (reads the trial id from the
image filename and answers correctly) so the runner's dispatch, parsing, scoring, and
JSONL output are exercised without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vwm.stimuli.colorwheel import angular_distance
from vwm.stimuli.generate import (
    generate_wheel_dataset,
    legend_anchor_canonical_hue,
    WHEEL_REPORT_M,
)
from vwm.local.base import LocalResponse
from vwm.local.wheel_runner import run_wheel, WheelRunConfig
from vwm.utils.parsing import parse_report_index


# ---------------- parsing ----------------

def test_parse_report_index_basic():
    assert parse_report_index("7") == 7
    assert parse_report_index("Answer: 11") == 11
    assert parse_report_index("I think it is number 3.") == 3
    assert parse_report_index("12", hi=12) == 12


def test_parse_report_index_out_of_range_and_garbage():
    assert parse_report_index("99", hi=12) is None
    assert parse_report_index("none of them") is None
    assert parse_report_index("", hi=12) is None


def test_parse_report_index_takes_concluding_number():
    # a model that reasons before answering concludes with its choice -> take the LAST
    # in-range value (mirrors parse_count), not the first stray number.
    assert parse_report_index("comparing 2 and 5, I choose 9") == 9
    assert parse_report_index("between 4 and 11 ... my answer: 8") == 8


# ---------------- fake client ----------------

class PerfectWheelClient:
    model_id = "fake-perfect"
    provider = "local"

    def __init__(self, trials):
        self._by_id = {t["trial_id"]: t for t in trials}
        self.loaded = False

    def ensure_loaded(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    @staticmethod
    def _tid(sample_image) -> str:
        name = Path(sample_image).name            # sample_<tid>.png
        return name[len("sample_"):-len(".png")]

    def call_two_turn(self, sample_image, sample_prompt, test_image, test_prompt,
                      system, max_new_tokens=8):
        tr = self._by_id[self._tid(sample_image)]
        if "legend" in test_prompt.lower():
            true = tr["report_hue_deg"]
            best = min(range(WHEEL_REPORT_M),
                       key=lambda k: angular_distance(legend_anchor_canonical_hue(k), true))
            raw = str(best + 1)
        else:
            raw = "YES" if tr["change"] else "NO"
        return LocalResponse(raw=raw, parsed_yes=None, model=self.model_id,
                             provider=self.provider, latency_s=0.0)

    def two_turn_yes_prob(self, sample_image, sample_prompt, test_image, test_prompt,
                          system, max_new_tokens=4):
        tr = self._by_id[self._tid(sample_image)]
        return ("YES" if tr["change"] else "NO", 0.99 if tr["change"] else 0.01)


def _dataset(tmp_path: Path):
    info = generate_wheel_dataset(tmp_path, seed=20260605, set_sizes=(2,), trials_per_cell=2)
    m = json.loads(Path(info["manifest_path"]).read_text())
    return info, m


# ---------------- change-detection (M4) ----------------

def test_run_wheel_change_perfect_acc_and_auc(tmp_path: Path):
    info, m = _dataset(tmp_path)
    out = tmp_path / "change.jsonl"
    cfg = WheelRunConfig(model_key="fake", task="change",
                         manifest_path=Path(info["manifest_path"]), out_path=out)
    summary = run_wheel(cfg, client=PerfectWheelClient(m["trials"]))
    assert summary["n_err"] == 0
    assert summary["mean_acc"] == pytest.approx(1.0)
    assert summary["auc"] == pytest.approx(1.0)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == m["n_trials"]
    assert all("p_yes" in r and r["task"] == "change" for r in rows)


# ---------------- continuous report (M3) ----------------

def test_run_wheel_report_low_circular_error(tmp_path: Path):
    info, m = _dataset(tmp_path)
    out = tmp_path / "report.jsonl"
    cfg = WheelRunConfig(model_key="fake", task="report",
                         manifest_path=Path(info["manifest_path"]), out_path=out)
    summary = run_wheel(cfg, client=PerfectWheelClient(m["trials"]))
    assert summary["n_err"] == 0
    # a perfect responder can only be off by the legend quantization (+/- half a step)
    assert summary["mean_error_deg"] <= 360.0 / WHEEL_REPORT_M / 2 + 1e-6
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == m["n_trials"]
    for r in rows:
        assert r["task"] == "report"
        assert 1 <= r["reported_index"] <= WHEEL_REPORT_M
        assert r["report_error_deg"] is not None


def test_run_wheel_respects_filters(tmp_path: Path):
    info, m = _dataset(tmp_path)
    out = tmp_path / "subset.jsonl"
    cfg = WheelRunConfig(model_key="fake", task="change",
                         manifest_path=Path(info["manifest_path"]), out_path=out, n_trials=5)
    summary = run_wheel(cfg, client=PerfectWheelClient(m["trials"]))
    assert summary["n_trials"] == 5
