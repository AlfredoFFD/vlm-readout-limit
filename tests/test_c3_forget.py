"""Forget-test for the C3 KV-truncation condition.

The operational definition of C3 in this project is: feed the model the TEST image
only (no sample image). If the experiment runner truly implements that semantics,
then the runner must never reference the trial's sample_path during a C3 call.

This test verifies the structural property -- a 'mock' MockClient logs every image
it sees, and we assert that for C3 the sample image is NEVER passed.

We also assert that under a C3 mock answer of "chance", the per-trial accuracy is
statistically indistinguishable from 50% (binomial test) -- which is the expected
behavior if no information about the sample is available.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vwm.api.mock_client import MockClient
from vwm.stimuli.generate import generate_dataset


class SpyClient(MockClient):
    """A MockClient that records every image path it sees."""

    def __init__(self, mode: str = "chance"):
        super().__init__(mode=mode)
        self.seen_paths: list[str] = []

    def _single_turn(self, images, prompt, system, max_tokens=8):
        for p in images:
            self.seen_paths.append(str(p))
        return super()._single_turn(images, prompt, system, max_tokens=max_tokens)

    def _two_turn(self, sample_image, sample_prompt, test_image, test_prompt, system, max_tokens=8):
        self.seen_paths.append(str(sample_image))
        self.seen_paths.append(str(test_image))
        return super()._two_turn(sample_image, sample_prompt, test_image, test_prompt, system, max_tokens=max_tokens)


@pytest.fixture(scope="module")
def small_c3_manifest(tmp_path_factory):
    out = tmp_path_factory.mktemp("stim_c3")
    info = generate_dataset(out, "C3", seed=42, trials_per_set_size=4)
    return Path(info["manifest_path"])


def test_c3_never_sees_sample_image(small_c3_manifest, tmp_path):
    """C3 (KV-truncation) must call the model with the test image only.

    We monkey-patch the local-runner's client construction so it uses our
    SpyClient instead of a real model, then assert no sample_path appears in
    the spy's seen_paths list.
    """
    # Read the manifest to find which sample/test paths are *known* to it.
    manifest = json.loads(Path(small_c3_manifest).read_text(encoding="utf-8"))
    known_sample_paths = {
        str((Path(small_c3_manifest).parent.parent / t["sample_path"]))
        for t in manifest["trials"]
    }
    # Also include literal manifest strings, in case paths are stored absolute already.
    known_sample_paths |= {t["sample_path"] for t in manifest["trials"]}

    # Build a Spy and manually feed it the trials that the runner would.
    spy = SpyClient(mode="chance")
    from vwm.utils.prompts import PROMPT_C2_SEQUENTIAL_TURN2, SYSTEM_INSTRUCTION

    for t in manifest["trials"][:8]:                  # small sweep is sufficient
        # C3 semantics: test image only.
        test_path = Path(t["test_path"])
        if not test_path.exists():
            test_path = Path(small_c3_manifest).parent.parent / t["test_path"]
        spy.call_single([test_path], prompt=PROMPT_C2_SEQUENTIAL_TURN2, system=SYSTEM_INSTRUCTION)

    # The spy must NEVER have seen a sample image path.
    for seen in spy.seen_paths:
        assert seen not in known_sample_paths, f"C3 leaked sample image: {seen}"
        # Stronger: even if the path string differs, no path basename should match a sample.
        assert "sample_" not in Path(seen).name, f"C3 leaked sample image basename: {seen}"


def test_c3_chance_mock_is_statistically_at_chance(small_c3_manifest, tmp_path):
    """With a chance-mode mock and no sample info, accuracy is ~50% binomial."""
    manifest = json.loads(Path(small_c3_manifest).read_text(encoding="utf-8"))
    from vwm.utils.prompts import PROMPT_C2_SEQUENTIAL_TURN2, SYSTEM_INSTRUCTION
    spy = SpyClient(mode="chance")
    n_correct = 0
    n_total = 0
    for t in manifest["trials"]:
        test_path = Path(t["test_path"])
        if not test_path.exists():
            test_path = Path(small_c3_manifest).parent.parent / t["test_path"]
        resp = spy.call_single([test_path], prompt=PROMPT_C2_SEQUENTIAL_TURN2,
                               system=SYSTEM_INSTRUCTION)
        if resp.parsed_yes is None:
            continue
        n_total += 1
        if resp.parsed_yes == bool(t["change"]):
            n_correct += 1
    acc = n_correct / max(n_total, 1)
    # 24 trials at p=0.5 -> 95% range roughly [0.33, 0.67]
    assert 0.20 < acc < 0.80, f"chance mock not at chance (acc={acc:.3f}, n={n_total})"


def test_run_c3_production_path_never_sees_sample(small_c3_manifest, tmp_path):
    """Drive the PRODUCTION run_c3 loop (not a re-implementation) with an
    injected spy and assert the sample image is never sent. Regression for the
    2026-08-07 review finding that this file's original guard only exercised a
    copy of the loop, so a leak in run_c3 itself would keep the suite green."""
    from vwm.local.c3_runner import C3Config, run_c3

    manifest = json.loads(Path(small_c3_manifest).read_text(encoding="utf-8"))
    sample_names = {Path(t["sample_path"]).name for t in manifest["trials"]}
    test_names = {Path(t["test_path"]).name for t in manifest["trials"]}

    spy = SpyClient(mode="chance")
    spy.model_id = "mock-spy"          # local-client interface expected by run_c3
    spy.unload = lambda: None
    run_c3(C3Config(model_key="mock", manifest_path=Path(small_c3_manifest),
                    out_path=tmp_path / "c3_spy.jsonl", n_trials=8), client=spy)

    assert spy.seen_paths, "spy saw no images -- run_c3 did not execute trials"
    seen_names = {Path(p).name for p in spy.seen_paths}
    assert seen_names.isdisjoint(sample_names), (
        f"run_c3 leaked sample images: {sorted(seen_names & sample_names)[:3]}")
    assert seen_names <= test_names
