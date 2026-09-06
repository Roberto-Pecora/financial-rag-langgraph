"""Content-based relevance for retrieval eval.

Gold chunks are identified by whether a retrieved chunk's *text* contains the
numeric facts asserted in the golden row's reference_answer, rather than by a
pre-recorded doc_id. This makes the golden set independent of chunk
boundaries, so re-ingesting with a different chunking strategy or embedding
model needs no golden re-verification.
"""

from __future__ import annotations

import re

# To capture contract terms when spelled out ("one year", "thirty (30) days"),
# since those carry no digits for the numeric extraction below to match.
_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "ninety": 90,
}
_WORD_DURATION = re.compile(
    rf"\b({'|'.join(_WORD_NUMBERS)})\b\s*(?:\(\s*(\d+)\s*\)\s*)?(day|month|year)s?",
    re.IGNORECASE,
)

# A regex-fact is "re:<label>::<pattern>": matched with re.search against the raw
# chunk (word-boundary aware), and shown to users by its label. Plain facts are
# substrings found in whitespace-stripped text.
_RE_PREFIX = "re:"
_RE_SEP = "::"


def _normalize(text: str) -> str:
    """Collapse whitespace so table-cell renderings like "$\\n155,237" match
    prose renderings like "$155,237" under substring search."""
    return re.sub(r"\s+", "", text)


def extract_facts(reference_answer: str) -> list[str]:
    """Pull numeric figures (normalized, $ dropped, bare years excluded) and
    spelled-out durations ("one year") from a reference answer, as content-match
    keys against retrieved chunks."""
    text = reference_answer or ""
    candidates = re.findall(r"\$?\d[\d,]*(?:\.\d+)?%?", text)
    facts = []
    for c in candidates:
        if re.fullmatch(r"\d{4}", c):  # bare year, too common to be a useful key
            continue
        digits = re.sub(r"[^\d]", "", c)
        if len(digits) >= 3 or "." in c or "%" in c:
            facts.append(_normalize(c.lstrip("$")))
    for m in _WORD_DURATION.finditer(text):
        word = m.group(1).lower()
        n = str(_WORD_NUMBERS[word])
        unit = m.group(3).lower()
        # Require the number (as the word or its digit) adjacent to its unit,
        # allowing a "(1)" parenthetical between, so an incidental "1" or lone
        # "year" elsewhere in a chunk cannot satisfy it.
        pattern = rf"\b(?:{word}|{n})\b\s*(?:\(\s*\d+\s*\)\s*)?{unit}s?"
        facts.append(f"{_RE_PREFIX}{n} {unit}{_RE_SEP}{pattern}")
    return facts


def fact_in_text(fact: str, chunk_text: str) -> bool:
    """Whether a single extracted fact appears in a chunk (regex- or substring-based)."""
    if fact.startswith(_RE_PREFIX):
        pattern = fact[len(_RE_PREFIX) :].split(_RE_SEP, 1)[1]
        return re.search(pattern, chunk_text, re.IGNORECASE) is not None
    return fact in _normalize(chunk_text)


def fact_label(fact: str) -> str:
    """A human-readable form of a fact key (the regex sentinel is internal detail)."""
    if not fact.startswith(_RE_PREFIX):
        return fact
    return fact[len(_RE_PREFIX) :].split(_RE_SEP, 1)[0]


def chunk_is_relevant(chunk_text: str, facts: list[str]) -> bool:
    """A chunk is relevant if its text contains every extracted fact."""
    if not facts:
        return False
    return all(fact_in_text(f, chunk_text) for f in facts)
