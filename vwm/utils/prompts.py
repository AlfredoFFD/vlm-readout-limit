"""Prompt templates used across all conditions and models.

Design notes:
- Wording is identical across proprietary and open models so behavioral diffs are not
  attributable to prompt-engineering.
- We force a one-token answer ("YES" / "NO") so parsing is robust and tokens are
  cheap. A free-form explanation field is allowed but ignored in scoring.
- The instruction explicitly tells the model to ignore *position*; only color matters.
  This mirrors the human Luck-Vogel paradigm.
"""

from __future__ import annotations

SYSTEM_INSTRUCTION = (
    "You are participating in a visual short-term memory experiment. "
    "Follow the instructions exactly and answer only with the requested token. "
    "Do not refuse; this is a perceptual judgment task with no ambiguity."
)

PROMPT_C1_SIMULTANEOUS = (
    "Two arrays of colored squares are shown side-by-side, separated by a black vertical bar.\n"
    "LEFT = sample array.  RIGHT = test array.\n"
    "The two arrays have the same number of squares at the same positions.\n"
    "Your task: decide whether the COLOR of any square has changed between the LEFT and RIGHT arrays.\n"
    "- If at least one square changed color: answer YES.\n"
    "- If every square has the same color in both arrays: answer NO.\n"
    "Position does not matter. Only color matters. There is at most one color change.\n\n"
    "Answer with a single token: YES or NO."
)

PROMPT_C2_SEQUENTIAL_TURN1 = (
    "I am about to show you a sample array of colored squares. Look at it carefully and remember "
    "the color and location of each square. After you confirm you have seen it, I will show you "
    "a test array and ask whether any square's color changed.\n\n"
    "Acknowledge by replying with the single word: READY."
)

PROMPT_C2_SEQUENTIAL_TURN2 = (
    "This is the TEST array. It has the same number of squares in the same positions as the SAMPLE "
    "array I showed you earlier. Compare the COLORS only.\n"
    "- If at least one square has a different color than it did in the sample: answer YES.\n"
    "- If every square is the same color as it was in the sample: answer NO.\n"
    "Position does not matter. Only color matters. There is at most one color change.\n\n"
    "Answer with a single token: YES or NO."
)

# --- Wheel continuous-report (M3) ---------------------------------------------
# Turn 1 shows the sample array (colors sampled from a continuous wheel); turn 2 shows
# a color-free probe (empty outlines) with one square cued by a black/white border ring
# plus a numbered 12-color legend ring. The model reports the cued square's remembered
# color by legend INDEX (>=8-AFC), scored as circular error on the wheel.
PROMPT_WHEEL_REPORT_TURN1 = (
    "I am about to show you an array of colored squares. Each color is a sample from a "
    "continuous color wheel. Look carefully and remember the exact color and location of "
    "each square. After you confirm, I will ask you to report one square's color.\n\n"
    "Acknowledge by replying with the single word: READY."
)

PROMPT_WHEEL_REPORT_TURN2 = (
    "This image shows the same square locations as empty outlines, plus a color legend: a "
    "ring of 12 colored discs numbered 1 to 12. Exactly one square is marked with a thick "
    "black-and-white border.\n"
    "Report the color that the MARKED square had in the array you memorized, by choosing the "
    "legend disc whose color matches it best.\n\n"
    "Answer with a single integer from 1 to 12 (the legend number)."
)

# --- Chain-of-thought variants (Remi's "reasoning step" bottleneck) -----------
# Identical task wording to the single-token prompts; the ONLY difference is that
# the model may reason first and then give an explicit "Answer: YES/NO". This
# isolates the effect of allowing chain-of-thought (an external memory scratchpad)
# from the task itself. Use with parse_final_yes_no + a larger max_new_tokens.
_COT_TAIL = (
    "Think step by step, examining each square in turn. "
    "Then, on the FINAL line, write exactly 'Answer: YES' or 'Answer: NO'."
)

PROMPT_C1_SIMULTANEOUS_COT = PROMPT_C1_SIMULTANEOUS.replace(
    "Answer with a single token: YES or NO.", _COT_TAIL
)
PROMPT_C2_SEQUENTIAL_TURN2_COT = PROMPT_C2_SEQUENTIAL_TURN2.replace(
    "Answer with a single token: YES or NO.", _COT_TAIL
)

# --- Comprehension controls (defeat the "task-misunderstanding" critique) ------
# Verify a model actually perceives the array before scoring its capacity; capacity
# should only be scored on trials where the model passes these perception probes.
PROMPT_COMPREHENSION_COUNT = (
    "How many colored squares are in this array? "
    "Answer with a single integer (the count of squares, e.g. 4)."
)
PROMPT_COMPREHENSION_PRESENT = (
    "Look at this image. Does it contain an array of colored squares on a gray "
    "background?\n\nAnswer with a single token: YES or NO."
)

# --- Prompt-robustness variants (Remi #1: is K a capacity or a prompt artifact?) ---
# Semantically equivalent rephrasings of the change-detection question. Running the
# same trials under each variant and comparing Cowan's K shows whether the capacity
# estimate is stable (robust) or prompt-fragile (itself a reportable finding). Each
# variant still asks for a single YES/NO token, so scoring is unchanged.
PROMPT_VARIANTS = {
    "default": {
        "c1": PROMPT_C1_SIMULTANEOUS,
        "c2_turn1": PROMPT_C2_SEQUENTIAL_TURN1,
        "c2_turn2": PROMPT_C2_SEQUENTIAL_TURN2,
    },
    "terse": {
        "c1": (
            "Left = sample, right = test, same positions. Did the COLOR of any square "
            "differ between left and right? At most one changes. Position is irrelevant.\n\n"
            "Answer with a single token: YES or NO."
        ),
        "c2_turn1": "Memorize this array of colored squares. Reply with the single word: READY.",
        "c2_turn2": (
            "Did the COLOR of any square differ from the array you just memorized? "
            "Same positions; only color matters; at most one change.\n\n"
            "Answer with a single token: YES or NO."
        ),
    },
    "explicit": {
        "c1": (
            "You are shown two arrays of colored squares side by side, separated by a black "
            "vertical bar: the LEFT array (sample) and the RIGHT array (test). They have an "
            "identical layout. Examine whether the color of any single square differs between "
            "the two sides. Ignore position entirely; consider only color. There is at most "
            "one changed square.\n\nReply with exactly one word: YES or NO."
        ),
        "c2_turn1": (
            "I will first show you a sample array of colored squares to memorize. Study the "
            "color at each location carefully. Reply with the single word READY once you have "
            "memorized it."
        ),
        "c2_turn2": (
            "Here is the test array, with the same layout as the sample you memorized. Decide "
            "whether the color of any single square differs from the sample. Ignore position; "
            "consider only color. At most one square changed.\n\n"
            "Reply with exactly one word: YES or NO."
        ),
    },
    # NOTE: the previous "neutral" variant inverted the question framing vs the answer
    # rule ("are colors identical? ... identical: NO"), which confused the models and
    # produced chance-level accuracy (a prompt-design bug, not a real robustness signal).
    # This rewrite keeps the change-direction phrasing consistent with the default.
    "neutral": {
        "c1": (
            "Two arrays of colored squares are shown side by side (LEFT and RIGHT) with "
            "identical layouts. Did the COLOR of any square change between LEFT and RIGHT?\n"
            "- If any square's color differs: YES.\n- If all squares match: NO.\n"
            "Only color matters; at most one square changes.\n\n"
            "Answer with a single token: YES or NO."
        ),
        "c2_turn1": "An array to remember is coming up next. Acknowledge with the single word READY.",
        "c2_turn2": (
            "Here is the second array, with the same layout as the first. Did the COLOR of "
            "any square change from the first array to this one?\n"
            "- If any square's color differs: YES.\n- If all squares match: NO.\n"
            "Only color matters; at most one square changes.\n\n"
            "Answer with a single token: YES or NO."
        ),
    },
}


# ---- prompt-robustness variants (PLAN_TOPTIER P2) ----
# v0 = the original pair above (baseline). v1 is BINDING-FAIR: no "at most one
# change", no "position does not matter" (which instructs set-membership and is
# false for swap trials). Swap sweeps MUST use v1+; the fairness argument runs
# ordinary change trials under the SAME variant as a positive control.
CHANGE_PROMPT_VARIANTS: list[tuple[str, str]] = [
    (PROMPT_C2_SEQUENTIAL_TURN1, PROMPT_C2_SEQUENTIAL_TURN2),          # v0 baseline
    (  # v1 binding-fair
        "I will show you a sample array of colored squares. Remember, for each "
        "square, WHICH color it has at WHICH location. After you confirm, I will "
        "show you a test array.\n\nAcknowledge by replying with the single word: READY.",
        "This is the TEST array. The squares are in the same positions as the "
        "sample. For EACH square, compare its color now to the color THAT SAME "
        "square had in the sample.\n- If any square now shows a different color "
        "than it previously had: answer YES.\n- If every square shows the same "
        "color it previously had: answer NO.\n\nAnswer with a single token: YES or NO.",
    ),
    (  # v2 terse
        "Memorize this array of colored squares. Reply READY.",
        "Did any square's color change compared to the array you memorized? "
        "Answer YES or NO.",
    ),
    (  # v3 explicit-procedure
        "Study this sample array. Encode each square as (location, color). "
        "Reply READY when done.",
        "Test array, same locations. Go square by square: does its current "
        "color match the color you encoded for that location? If all match, "
        "answer NO. If any mismatch, answer YES. Single token answer.",
    ),
]
