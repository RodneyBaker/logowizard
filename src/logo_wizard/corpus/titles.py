"""Turn word lists into comic-book and novel-cover title phrases.

Templates fill slots from the public-domain word lists. Real genre-fiction
titles from the Gutenberg catalog are mixed in as they are, so the layout
model also sees the shape of genuine titles.
"""

import random
import re
from dataclasses import dataclass

SMALL_WORDS = {"a", "an", "and", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"}

SERIES = ["Tales", "Chronicles", "Adventures", "Legends", "Saga", "Secrets", "Mysteries", "Return", "Rise", "Fall", "Curse", "Shadow", "Quest", "Voyage", "Song", "Book"]
KIN = ["Daughter", "Son", "Bride", "Children", "Heir", "King", "Queen", "Prince", "Princess", "Keeper", "Lord", "Lady", "Master", "Ghost", "Prophet"]
HERO_PREFIX = ["Captain", "Doctor", "Professor", "The Mighty", "The Amazing", "The Incredible", "The Astonishing", "Agent", "Commander", "Lady", "Sergeant", "Kid", "Mister", "Madame"]
HERO_SUFFIX = ["Man", "Woman", "Girl", "Boy", "Kid", "Squad", "Legion", "Patrol", "Force", "League"]
NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

# (template, style). Slots: {adj} {noun} {noun2} {proper} {first} {last} {series} {kin} {hero} {suffix}
TEMPLATES = [
    ("The {adj} {noun}", "novel"),
    ("The {noun} of {proper}", "novel"),
    ("{kin} of {proper}", "novel"),
    ("{kin} of the {noun}", "novel"),
    ("The {noun} and the {noun2}", "novel"),
    ("{series} of the {adj} {noun}", "novel"),
    ("{series} of {proper}", "novel"),
    ("The {last} {noun}", "novel"),
    ("{first} of {proper}", "novel"),
    ("A {noun} in {proper}", "novel"),
    ("Beyond the {adj} {noun}", "novel"),
    ("{hero} {noun}", "comic"),
    ("{hero} {proper}", "comic"),
    ("{noun} {suffix}", "comic"),
    ("The {adj} {noun} {suffix}", "comic"),
    ("{first} {last} and the {noun} of {noun2}", "comic"),
    ("{series} of the {noun} {suffix}", "comic"),
    ("{adj} {noun} Comics", "comic"),
    ("Attack of the {adj} {noun}", "comic"),
    ("{noun}!", "comic"),
]


def title_case(text: str) -> str:
    words = text.split(" ")
    out = []
    for i, word in enumerate(words):
        if i > 0 and word.lower() in SMALL_WORDS:
            out.append(word.lower())
        elif word[:1].islower():
            out.append(word[:1].upper() + word[1:])
        else:
            out.append(word)
    return " ".join(out)


def fix_articles(text: str) -> str:
    """'A Adventuress' -> 'An Adventuress' after the slots are filled."""
    return re.sub(r"\b(A|a) (?=[AEIOUaeiou])", lambda m: m.group(1) + "n ", text)


@dataclass(frozen=True)
class Phrase:
    text: str
    style: str
    subtitle: str
    origin: str


@dataclass
class WordLists:
    nouns: list[str]
    adjectives: list[str]
    proper: list[str]
    first: list[str]
    last: list[str]


def _subtitle(rng: random.Random, style: str, words: WordLists) -> str:
    roll = rng.random()
    if roll < 0.5:
        return ""
    if style == "comic":
        return rng.choice([f"No. {rng.randint(1, 99)}", f"Issue #{rng.randint(1, 250)}", f"Vol. {rng.choice(NUMERALS)}"])
    if roll < 0.85:
        return f"{rng.choice(words.first)} {rng.choice(words.last)}"
    return f"Book {rng.choice(NUMERALS)}"


def generate(words: WordLists, count: int, seed: int = 1890, real_titles: list[str] | None = None) -> list[Phrase]:
    """Deterministic list of unique title phrases, real titles first."""
    for name in ("nouns", "adjectives", "proper", "first", "last"):
        needed = 2 if name == "nouns" else 1
        if len(getattr(words, name)) < needed:
            raise ValueError(f"word list '{name}' needs at least {needed} entries; check the raw sources")
    rng = random.Random(seed)
    seen: set[str] = set()
    phrases: list[Phrase] = []

    for title in real_titles or []:
        # The catalog mixes "The war of the worlds" with "The War of the Worlds".
        title = title_case(title)
        style = "novel"
        key = title.lower()
        if key not in seen:
            seen.add(key)
            phrases.append(Phrase(title, style, _subtitle(rng, style, words), "gutenberg-title"))

    attempts = 0
    while len(phrases) < count and attempts < count * 20:
        attempts += 1
        template, style = rng.choice(TEMPLATES)
        noun, noun2 = rng.sample(words.nouns, 2)
        text = template.format(
            adj=rng.choice(words.adjectives),
            noun=noun,
            noun2=noun2,
            proper=rng.choice(words.proper),
            first=rng.choice(words.first),
            last=rng.choice(words.last),
            series=rng.choice(SERIES),
            kin=rng.choice(KIN),
            hero=rng.choice(HERO_PREFIX),
            suffix=rng.choice(HERO_SUFFIX),
        )
        text = fix_articles(re.sub(r"\s+", " ", title_case(text)).strip())
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(Phrase(text, style, _subtitle(rng, style, words), "template"))
    return phrases
