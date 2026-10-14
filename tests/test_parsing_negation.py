"""Regression tests for the 2026-08-07 code-review parsing findings.

Finding 1 (CRITICAL): parse_yes_no / parse_final_yes_no inverted negated
no-change answers ("Nothing changed." -> YES), silently inflating false-alarm
rates for every sentence-answering model.

Finding 5 (MAJOR): parse_count's cue rule let layout chatter hijack the count
("There are 2 rows and 2 columns, so 4 squares in total." -> 2).
"""

from __future__ import annotations

import pytest

from vwm.utils.parsing import (
    parse_yes_no,
    parse_final_yes_no,
    parse_count,
    parse_report_index,
)


# ------------------------------------------------- negated no-change answers

@pytest.mark.parametrize("text", [
    "Nothing changed.",
    "None of the squares changed color.",
    "Neither array changed.",
    "Nothing is different.",
    "The colors did not change.",
    "No change occurred.",
    "The squares didn't change.",
    "It looks no different.",
])
def test_negated_change_words_parse_as_no(text):
    assert parse_yes_no(text) is False


@pytest.mark.parametrize("text", [
    "They are not the same.",
    "The arrays are not identical.",
])
def test_negated_same_words_parse_as_yes(text):
    assert parse_yes_no(text) is True


@pytest.mark.parametrize("text,expected", [
    ("Yes, the color changed.", True),
    ("No, they are the same.", False),
    ("The color changed.", True),
    ("They look the same.", False),
    ("YES", True),
    ("no", False),
    ("changed", True),
    ("unchanged", False),
])
def test_unnegated_answers_still_parse(text, expected):
    assert parse_yes_no(text) is expected


def test_cot_final_line_negation():
    cot = ("Comparing square by square. The red square matches; the blue one "
           "matches too.\nAnswer: None of the squares changed")
    assert parse_final_yes_no(cot) is False


def test_cot_explicit_answer_still_wins():
    cot = "The arrays look similar but square 3 differs.\nAnswer: YES"
    assert parse_final_yes_no(cot) is True


# ------------------------------------------------------- count cue hijacking

@pytest.mark.parametrize("text,expected", [
    ("There are 2 rows and 2 columns, so 4 squares in total.", 4),
    ("I see a 2 x 2 grid containing 4 colored squares.", 4),
    ("There are 4 squares.", 4),
    ("I see 4.", 4),
    ("I count 6 squares arranged in a 2 x 3 grid.", 6),
    ("Four squares.", 4),
    ("The answer is 8.", 8),
])
def test_parse_count(text, expected):
    assert parse_count(text) == expected


# --------------------------------------------- report-index instruction echo

def test_report_index_instruction_echo_returns_none():
    assert parse_report_index("Please answer with a number from 1 to 12.") is None


def test_report_index_still_parses_real_answers():
    assert parse_report_index("The color matches legend disc 7.") == 7
    assert parse_report_index("Answer: 11") == 11
    assert parse_report_index("3") == 3
