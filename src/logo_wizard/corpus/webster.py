"""Headwords and parts of speech from Webster's Unabridged Dictionary (1913)."""

import re
from collections import defaultdict

# A headword sits alone on a line in capitals, e.g. "ABACUS" or "SEA SERPENT".
_HEADWORD = re.compile(r"^[A-Z][A-Z' -]*[A-Z]$")
# The line after it starts with the syllabified word and its part of speech,
# e.g. 'Ab"a*cus, n.; pl.' or 'Brave, a. [Compar. Braver ...'. Only the first
# label counts, so 'That, pron., a., conj., & adv.' is a pronoun, not an adjective.
_POS = re.compile(r"^[^,]*,\s*(n|a|adv|v\.\s*[ti]|v|prep|conj|pron|interj|imp|p\.\s*p|p\.\s*pr|definite article|indefinite article)\.")

# An inflected form leaves the label empty and explains it below: 'Came,' then
# 'Defn: imp. of Come.'
_INFLECTED = re.compile(r"^[^,]*,$")
_INFLECTED_DEFN = re.compile(r"^Defn: (imp|p\. p|p\. pr)\.")

_POS_NAMES = {"n": "noun", "a": "adjective", "adv": "adverb"}


def _pos_name(raw: str) -> str:
    """Content-word labels by name; prepositions, pronouns, participles and the like as "function"."""
    raw = raw.replace(" ", "")
    if raw in _POS_NAMES:
        return _POS_NAMES[raw]
    return "verb" if raw.startswith("v") else "function"


def parse_headwords(body: str) -> dict[str, set[str]]:
    """Map each single-word lowercase headword to the parts of speech it is listed under.

    Words with any entry as a function word (preposition, pronoun, participle...)
    carry "function", which keeps "For" and "That" out of the noun and adjective lists.
    """
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
        elif _INFLECTED.match(nxt) and any(_INFLECTED_DEFN.match(l.strip()) for l in lines[i + 2 : i + 5]):
            parts[word.lower()].add("function")
    return dict(parts)
