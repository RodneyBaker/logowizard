"""Words and proper nouns from the King James Bible."""

import re
from collections import Counter

_VERSE = re.compile(r"^\d+:\d+\s+")
_TOKEN = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
_SENTENCE_END = re.compile(r"[.!?:;]\s*$")


def verses(body: str) -> list[str]:
    """Join the wrapped lines of each verse into one string."""
    out: list[str] = []
    current: list[str] = []
    for line in body.split("\n"):
        line = line.strip()
        if _VERSE.match(line):
            if current:
                out.append(" ".join(current))
            current = [_VERSE.sub("", line)]
        elif line and current:
            current.append(line)
        elif not line and current:
            out.append(" ".join(current))
            current = []
    if current:
        out.append(" ".join(current))
    return out


def word_counts(verse_list: list[str]) -> Counter:
    """Lowercase word frequencies, used to keep dictionary words that people actually use."""
    return Counter(t.lower() for v in verse_list for t in _TOKEN.findall(v))


def proper_nouns(verse_list: list[str], min_count: int = 3) -> Counter:
    """Capitalised words that never appear in lowercase and are not sentence starts."""
    capital: Counter = Counter()
    lower: set[str] = set()
    for verse in verse_list:
        tokens = list(_TOKEN.finditer(verse))
        for idx, match in enumerate(tokens):
            word = match.group(0)
            if word[0].islower():
                lower.add(word)
                continue
            if idx == 0 or word.isupper():
                continue
            before = verse[: match.start()]
            if _SENTENCE_END.search(before):
                continue
            capital[word] += 1
    return Counter(
        {w: c for w, c in capital.items() if c >= min_count and w.lower() not in lower and len(w) > 2}
    )
