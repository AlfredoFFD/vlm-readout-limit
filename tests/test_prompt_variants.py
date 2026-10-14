"""Unit tests for the prompt-robustness sweep (PROMPT_VARIANTS + runner wiring).

Verifies the variant registry is well-formed and that the 'default' variant is
byte-identical to the canonical prompts (so a default sweep reproduces the original
C1/C2 behavior). No GPU / model required.
"""

from __future__ import annotations

import pytest

from vwm.utils.prompts import (
    PROMPT_VARIANTS,
    PROMPT_C1_SIMULTANEOUS,
    PROMPT_C2_SEQUENTIAL_TURN1,
    PROMPT_C2_SEQUENTIAL_TURN2,
)

EXPECTED_VARIANTS = {"default", "terse", "explicit", "neutral"}
REQUIRED_KEYS = {"c1", "c2_turn1", "c2_turn2"}


def test_registry_shape():
    assert EXPECTED_VARIANTS <= set(PROMPT_VARIANTS), "missing expected prompt variants"
    for name, v in PROMPT_VARIANTS.items():
        assert REQUIRED_KEYS <= set(v), f"variant {name!r} missing keys"
        for key in REQUIRED_KEYS:
            assert isinstance(v[key], str) and v[key].strip(), f"{name}.{key} empty"


def test_default_variant_is_canonical():
    d = PROMPT_VARIANTS["default"]
    assert d["c1"] == PROMPT_C1_SIMULTANEOUS
    assert d["c2_turn1"] == PROMPT_C2_SEQUENTIAL_TURN1
    assert d["c2_turn2"] == PROMPT_C2_SEQUENTIAL_TURN2


def test_every_variant_requests_yes_no():
    # The decision prompts (c1, c2_turn2) must elicit a YES/NO answer.
    for name, v in PROMPT_VARIANTS.items():
        for key in ("c1", "c2_turn2"):
            text = v[key].upper()
            assert "YES" in text and "NO" in text, f"{name}.{key} does not request YES/NO"


def test_variants_are_distinct_wordings():
    # The non-default variants must actually differ from default (else the sweep is a no-op).
    for name in EXPECTED_VARIANTS - {"default"}:
        assert PROMPT_VARIANTS[name]["c1"] != PROMPT_VARIANTS["default"]["c1"], \
            f"variant {name} c1 is identical to default"


def test_runner_imports_and_rejects_unknown_variant():
    from vwm.local.prompt_sweep_runner import run_sweep, SweepConfig
    from pathlib import Path
    cfg = SweepConfig(
        model_key="qwen3-vl-4b", condition="C1", variant="does-not-exist",
        manifest_path=Path("nope.json"), out_path=Path("out.jsonl"),
    )
    with pytest.raises(ValueError, match="Unknown prompt variant"):
        run_sweep(cfg)
