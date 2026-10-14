"""Unit tests for the chain-of-thought layer (parse_final_yes_no + CoT prompts).

CoT outputs reason before answering, so the standard first-word parser fails on
them. These cases lock in that we extract the FINAL verdict correctly, and that
the CoT prompts reuse the exact task wording of the single-token prompts (so any
behavioral difference is attributable to reasoning, not to a reworded task).
No GPU / model is required.
"""

from __future__ import annotations

import pytest

from vwm.utils.parsing import parse_final_yes_no, parse_yes_no
from vwm.utils.prompts import (
    PROMPT_C1_SIMULTANEOUS,
    PROMPT_C1_SIMULTANEOUS_COT,
    PROMPT_C2_SEQUENTIAL_TURN2,
    PROMPT_C2_SEQUENTIAL_TURN2_COT,
)


@pytest.mark.parametrize("text,expected", [
    ("Let me check each square. The top-left went blue to red. Answer: YES", True),
    ("All four squares match across both arrays. Answer: NO", False),
    ("Square 1 same, square 2 same. Hmm. Actually yes. Answer: YES", True),
    ("First I thought yes, but on reflection no change. Answer: NO", False),
    ("answer: yes", True),
    ("Answer - NO", False),
    ("The colors are identical, so they are the same", False),   # no Answer line -> 'same'
    ("One square clearly changed color", True),                  # 'changed' -> yes
    ("YES", True),
    ("NO", False),
    ("", None),
    (None, None),
    # regression (code-review CRITICAL #1): a body word like "identical" after the real
    # verdict must NOT flip the score. Explicit Answer line wins; bare verdict on the
    # final line wins; body chatter on earlier lines is ignored.
    ("Checking each square.\nThe rest are identical.\nAnswer: YES", True),
    ("Square 1 same.\nSquare 2 same.\nNO", False),
    ("Square 2 changed.\nSquare 3 same.\nYES", True),
    ("The top-left changed. The others are identical.\nAnswer: NO", False),
])
def test_parse_final_yes_no(text, expected):
    assert parse_final_yes_no(text) == expected


def test_explicit_final_answer_overrides_earlier_mentions():
    # An earlier 'yes' must not win over the explicit final 'Answer: NO'.
    assert parse_final_yes_no("yes yes yes ... but Answer: NO") is False
    assert parse_final_yes_no("no, no, no ... Answer: YES") is True


def test_single_token_matches_base_parser():
    for tok in ["YES", "no", "Yes.", "NO!"]:
        assert parse_final_yes_no(tok) == parse_yes_no(tok)


def test_cot_prompts_preserve_task_wording():
    # CoT variant must differ from the single-token prompt ONLY in the answer-format
    # tail -- the task description (everything before it) must be byte-identical.
    for base, cot in [
        (PROMPT_C1_SIMULTANEOUS, PROMPT_C1_SIMULTANEOUS_COT),
        (PROMPT_C2_SEQUENTIAL_TURN2, PROMPT_C2_SEQUENTIAL_TURN2_COT),
    ]:
        head = base.split("Answer with a single token")[0]
        assert cot.startswith(head), "CoT prompt changed the task description, not just the answer format"
        assert "single token" not in cot, "CoT prompt should not still demand a single token"
        assert "Answer: YES" in cot and "Answer: NO" in cot
