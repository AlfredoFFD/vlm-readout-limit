"""Robust YES/NO parsing for model responses.

Model outputs are noisy. We accept:
- Exact "YES" / "NO" (with or without punctuation)
- First word of the first sentence, lowercased
- "y" / "n" as last resort
- Phrases like "yes, the color changed" / "no, they are the same"
- "changed" -> yes, "same" / "unchanged" / "identical" -> no (last-ditch)

Returns:
- True for YES
- False for NO
- None if unparseable
"""

from __future__ import annotations

import re

_YES_TOKENS = {"yes", "y", "yeah", "yep", "yup", "true", "1", "changed", "different"}
_NO_TOKENS = {"no", "n", "nope", "nah", "false", "0", "same", "unchanged", "identical", "no_change"}

# Sentence-level verdict vocabulary (negation-aware; see _verdict_from_tokens).
_PURE_YES = {"yes", "yeah", "yep", "yup", "true"}
_PURE_NO = {"no", "nope", "nah", "false"}
_CHANGE_WORDS = {"changed", "change", "changes", "different", "differs", "differ", "recolored"}
_SAME_WORDS = {"same", "unchanged", "identical", "no_change", "match", "matches"}
# Apostrophes are stripped upstream, so contractions appear without them.
_NEGATORS = {
    "no", "not", "nothing", "none", "neither", "never", "nor", "without",
    "didnt", "doesnt", "dont", "isnt", "arent", "wasnt", "werent",
    "hasnt", "havent", "cannot", "cant",
}


def _verdict_from_tokens(tokens: set[str]) -> bool | None:
    """Negation-aware YES/NO verdict from a sentence's token set.

    'changed'-family words assert change unless negated ("nothing changed" ->
    NO); 'same'-family words assert no-change unless negated ("not the same" ->
    YES). A sentence mixing both families is ambiguous. Bare yes/no words are
    the last resort so that "no change occurred" resolves through the negation
    logic, not the bare "no".
    """
    neg = bool(tokens & _NEGATORS)
    chg = bool(tokens & _CHANGE_WORDS)
    sam = bool(tokens & _SAME_WORDS)
    if chg and not sam:
        return not neg
    if sam and not chg:
        return neg
    if chg and sam:
        return None
    yes = bool(tokens & _PURE_YES)
    no = bool(tokens & _PURE_NO)
    if yes and not no:
        return True
    if no and not yes:
        return False
    return None


def parse_yes_no(text: str | None) -> bool | None:
    if text is None:
        return None
    s = text.strip().lower()
    if not s:
        return None
    # Strip code fences / quotes / punctuation surrounding the answer.
    s = re.sub(r"[\"'`*_]", "", s)
    # Common chat artifacts.
    s = s.lstrip(":-> \t")
    # Direct token match on first word — but only when the sentence carries no
    # negator (else "No, ..." wins while "Nothing changed" would misparse).
    head = re.split(r"[.!?\n]", s, maxsplit=1)[0]
    head_tokens = set(t for t in re.split(r"[\s,;:]+", head) if t)
    first_word = re.split(r"[\s,.!?:;]+", s, maxsplit=1)[0] if s else ""
    if not (head_tokens & _NEGATORS):
        if first_word in _YES_TOKENS:
            return True
        if first_word in _NO_TOKENS:
            return False
    elif first_word in _PURE_YES | {"y"}:
        return True
    elif first_word in _PURE_NO | {"n"}:
        return False
    # Negation-aware sentence-level verdict on the first sentence.
    return _verdict_from_tokens(head_tokens)


def normalize_response(text: str | None) -> dict:
    """Return a structured response: {raw, parsed_yes, parsable}."""
    parsed = parse_yes_no(text)
    return {
        "raw": text if text is not None else "",
        "parsed_yes": parsed,
        "parsable": parsed is not None,
    }


_ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*(yes|no|y|n|true|false)\b", re.IGNORECASE)


def parse_final_yes_no(text: str | None) -> bool | None:
    """Extract a YES/NO from a chain-of-thought (reasoning) response.

    CoT outputs reason *before* answering, so first-word parsing (`parse_yes_no`)
    fails. Resolution order:
    1. explicit "Answer: YES/NO" -- take the LAST such match (the final verdict);
    2. else the last yes/no-ish token anywhere in the text (the conclusion);
    3. else fall back to the first-word parser for single-token replies.
    """
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    # 1. explicit "Answer: YES/NO" -- the LAST such match is the model's final verdict.
    matches = _ANSWER_RE.findall(s)
    if matches:
        return matches[-1].lower() in {"yes", "y", "true"}
    # 2. the verdict should be on the FINAL non-empty line. Honor it only if that line
    #    is unambiguous (not a body sentence that mixes "changed" and "identical");
    #    negation-aware so "Answer: None of the squares changed" resolves to NO.
    last_line = next((ln for ln in reversed(s.splitlines()) if ln.strip()), "")
    line_tokens = set(re.findall(r"[a-z_]+", re.sub(r"[\"'`*_]", "", last_line.lower())))
    verdict = _verdict_from_tokens(line_tokens)
    if verdict is not None:
        return verdict
    # 3. ambiguous/empty final line -> fall back to the first-word/first-sentence parser.
    return parse_yes_no(text)


_NUMWORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# A digit that directly follows counting/answer language is the model's actual count,
# even when layout chatter ("a 2x2 grid", "3 columns") introduces stray numbers.
_COUNT_CUE = re.compile(
    r"(?:\bcount(?:ed)?\b|\bsee\b|\bsaw\b|there\s+(?:are|is)|\bare\b|\btotal\b|\banswer\b|[=:])"
    r"[^\d]{0,12}(\d{1,2})\b",
    re.IGNORECASE,
)


# The unit rule beats every cue: a number attached to the word "square(s)" IS
# the count of squares, however much layout chatter surrounds it.
_COUNT_UNIT = re.compile(r"\b(\d{1,2})\s+(?:[a-z-]+\s+){0,2}squares?\b", re.IGNORECASE)


def parse_count(text: str | None) -> int | None:
    """Extract an integer square-count from a comprehension-probe response.

    Resolution order, chosen so layout chatter ("a 2x2 grid", "3 columns", "2
    rows") does not hijack the answer:
    1. a number attached to the unit ("4 squares", "4 colored squares") — the
       LAST such match (a reasoning reply concludes with its answer);
    2. a digit right after counting/answer language ("I see 4", "there are 4", "= 4");
    3. else the LAST standalone number;
    4. else the last number word ("four").
    Returns None if no count is found.
    """
    if not text:
        return None
    s = text.lower()
    units = []
    for m in _COUNT_UNIT.finditer(s):
        pre = s[max(0, m.start() - 4):m.start()]
        post = s[m.end():m.end() + 8]
        # "rows of 4 squares each": distributive/partitive counts are per-group
        # chatter, not the total — leave those to the later rules.
        if re.search(r"\bof\s*$", pre) or re.match(r"\s*each\b", post):
            continue
        units.append(int(m.group(1)))
    if units:
        return units[-1]
    cues = _COUNT_CUE.findall(s)
    if cues:
        return int(cues[-1])
    nums = re.findall(r"\b(\d{1,2})\b", s)
    if nums:
        return int(nums[-1])
    word_hits = [
        (m.start(), n)
        for word, n in _NUMWORDS.items()
        for m in re.finditer(rf"\b{word}\b", s)
    ]
    if word_hits:
        return max(word_hits)[1]   # last number word by position
    return None


_REPORT_NUMWORDS = {
    **_NUMWORDS, "eleven": 11, "twelve": 12,
}
# A digit that follows answer/index/legend language is the report index, even when
# the response carries chatter ("the square marked with border", "square 2").
_REPORT_CUE = re.compile(
    r"(?:answer|index|legend|colou?r|number|disc|#|=|:)\s*[#:=]?\s*(\d{1,2})\b",
    re.IGNORECASE,
)


def parse_report_index(text: str | None, lo: int = 1, hi: int = 12) -> int | None:
    """Extract a 1..hi legend index from a continuous-report response.

    Resolution order: (1) a digit right after answer/index/legend language; (2) the
    first standalone integer within [lo, hi]; (3) a number word in range. Returns None
    if nothing in range is found.
    """
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    # Drop instruction echoes of the answer range ("a number from 1 to 12",
    # "between 1 and 12", "1-12") so they cannot be mistaken for an answer.
    s = re.sub(r"\b(?:from\s+)?\d{1,2}\s*(?:to|-|–|through|and)\s*\d{1,2}\b", " ", s)
    if not s.strip():
        return None

    def _in(n: int) -> bool:
        return lo <= n <= hi

    # 1. cued matches -> the LAST in-range one (a reasoning reply concludes with its answer)
    cued = [n for n in (int(m) for m in _REPORT_CUE.findall(s)) if _in(n)]
    if cued:
        return cued[-1]
    # 2. bare digits -> the LAST in-range one (mirrors parse_count)
    bare = [n for n in (int(m) for m in re.findall(r"\b(\d{1,2})\b", s)) if _in(n)]
    if bare:
        return bare[-1]
    # 3. number words -> the last by position
    low = s.lower()
    hits = [
        (m.start(), n)
        for word, n in _REPORT_NUMWORDS.items() if _in(n)
        for m in re.finditer(rf"\b{word}\b", low)
    ]
    if hits:
        return max(hits)[1]
    return None
