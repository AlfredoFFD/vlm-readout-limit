"""Unit tests for the comprehension-control parser (parse_count) and runner wiring.

The comprehension control verifies a model perceives the array (counts squares)
before its capacity is scored. These cases lock in robust integer extraction from
noisy model replies. No GPU / model required.
"""

from __future__ import annotations

import pytest

from vwm.utils.parsing import parse_count


@pytest.mark.parametrize("text,expected", [
    ("4", 4),
    ("There are 4 colored squares.", 4),
    ("I count four squares", 4),
    ("eight", 8),
    ("The array contains 8 squares on a gray background.", 8),
    ("Six squares are visible.", 6),
    ("1", 1),
    ("There are no squares? Actually three.", 3),   # digit absent -> word 'three'
    ("", None),
    (None, None),
    ("I cannot tell", None),
    # regression: layout chatter must not hijack the count (code-review CRITICAL #3)
    ("I can see 4 squares in a 2x2 grid arrangement.", 4),
    ("2 rows of 4 squares each, so 8 total.", 8),
    ("Based on the 3-column layout, I see 6 squares.", 6),
])
def test_parse_count(text, expected):
    assert parse_count(text) == expected


def test_digit_beats_word_when_both_present():
    # If a digit is present it wins (it is the more explicit answer).
    assert parse_count("four, i.e. 4 squares") == 4


def test_sample_path_refuses_combined_image():
    # sample/test are single arrays (valid); combined holds two arrays -> must refuse,
    # else the count doubles and silently corrupts the comprehension gate.
    from vwm.local.comprehension_runner import _sample_path
    assert _sample_path({"sample_path": "a/s.png", "test_path": "a/t.png"}) == "a/s.png"
    assert _sample_path({"test_path": "a/t.png"}) == "a/t.png"
    with pytest.raises(ValueError):
        _sample_path({"combined_path": "a/c.png", "trial_id": 7})
    with pytest.raises(ValueError):
        _sample_path({"trial_id": 1})
