"""Headwords and parts of speech from Webster's Unabridged Dictionary (1913)."""

import re
from collections import defaultdict

# A headword sits alone on a line in capitals, e.g. "ABACUS" or "SEA SERPENT".
_HEADWORD = re.compile(r"^[A-Z][A-Z' -]*[A-Z]$")
# The line after it starts with the syllabified word and its part of speech,
# e.g. 'Ab"a*cus, n.; pl.' or 'Brave, a. [Compar. Braver ...'.
_POS = re.compile(r",\s*(n|a|adv|v\.\s*[ti]|v)\.")

_POS_NAMES = {"n": "noun", "a": "adjective", "adv": "adverb"}


def _pos_name(raw: str) -> str:
    raw = raw.replace(" ", "")
    return "verb" if raw.startswith("v") else _POS_NAMES[raw]


def parse_headwords(body: str) -> dict[str, set[str]]:
    """Map each single-word lowercase headword to the parts of speech it is listed under."""
    parts: dict[str, set[str]] = defaultdict(set)
    lines = body.split("\n")
    for i, line in enumerate(lines[:-1]):
        word = line.strip()
        if not _HEADWORD.match(word) or " " in word or "-" in word:
            continue
        nxt = lines[i + 1].strip()
        match = _POS.search(nxt[:80])
        if match:
            parts[word.lower()].add(_pos_name(match.group(1)))
    return dict(parts)
