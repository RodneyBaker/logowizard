"""Build the word lists and title phrases from the raw sources.

Usage (from the repo root):
    python -m logo_wizard.corpus            # download if needed, then build
    python -m logo_wizard.corpus --offline  # build from files already in local/raw
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from .. import paths
from . import catalog, census, kjv, webster
from .fetch import fetch_all
from .gutenberg import strip_boilerplate
from .sources import SOURCES
from .titles import Phrase, WordLists, generate

RAW_DIR = paths.RAW_DIR
OUT_DIR = paths.CORPORA


def load_blocklist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# Grammar words and everyday verbs the dictionary also lists as nouns or
# adjectives ("Will, n.", "Every, a.", "Make, n."); in a title slot they read as mistakes.
FUNCTION_WORDS = set(
    """
    all also am an and any are as at be been but by can could did do does each either even ever every few
    for from had has have he her here him his how if in into is it its let may me might more most much must
    my neither no nor not now of off on once one only or other our out over own said same shall she should
    so some such than that the thee their them then there these they thine this those thou thus thy to too
    unto upon us very was we were what when where which while who whom whose why will with would ye yea yet
    you your
    come go make say put take know set give get see like pass bring seek hear cast cut deliver live
    none many toward
    """.split()
)


def pick_vocabulary(
    parts: dict[str, set[str]], usage: Counter, blocklist: set[str], min_usage: int = 2
) -> tuple[list[str], list[str]]:
    """Dictionary nouns and adjectives that are also in common use, most used first."""

    def keep(word: str) -> bool:
        return (
            3 <= len(word) <= 12
            and word.isalpha()
            and usage[word] >= min_usage
            and word not in blocklist
            and "function" not in parts[word]
            and word not in FUNCTION_WORDS
        )

    ranked = sorted(parts, key=lambda w: (-usage[w], w))
    nouns = [w.capitalize() for w in ranked if "noun" in parts[w] and keep(w)]
    adjectives = [w.capitalize() for w in ranked if "adjective" in parts[w] and keep(w)]
    return nouns, adjectives


def write_list(name: str, items: list[str], out_dir: Path) -> None:
    (out_dir / name).write_text("\n".join(items) + "\n", encoding="utf-8")


def write_phrases(phrases: list[Phrase], out_dir: Path) -> None:
    with open(out_dir / "phrases.tsv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["text", "style", "subtitle", "origin"])
        for p in phrases:
            writer.writerow([p.text, p.style, p.subtitle, p.origin])


def build(raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR, count: int = 120_000, seed: int = 1890) -> dict[str, int]:
    paths = {key: raw_dir / source.filename for key, source in SOURCES.items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing raw sources (run without --offline to download): " + ", ".join(missing))
    blocklist = load_blocklist(out_dir / "blocklist.txt")

    verse_list = kjv.verses(strip_boilerplate(_read(paths["kjv"])))
    titles = [t for t in catalog.genre_titles(_read(paths["catalog"])) if not set(t.lower().split()) & blocklist]
    usage = kjv.word_counts(verse_list) + Counter(w.lower() for t in titles for w in t.split())

    parts = webster.parse_headwords(strip_boilerplate(_read(paths["webster"])))
    nouns, adjectives = pick_vocabulary(parts, usage, blocklist)
    proper = [w for w, _ in kjv.proper_nouns(verse_list).most_common() if w.lower() not in blocklist]
    last = census.surnames(paths["surnames"])
    first = sorted(
        set(census.first_names(_read(paths["first_female"])) + census.first_names(_read(paths["first_male"]))),
        key=str.lower,
    )

    words = WordLists(nouns=nouns, adjectives=adjectives, proper=proper, first=first, last=last)
    phrases = generate(words, count=count, seed=seed, real_titles=titles)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_list("nouns.txt", nouns, out_dir)
    write_list("adjectives.txt", adjectives, out_dir)
    write_list("proper_nouns.txt", proper, out_dir)
    write_list("first_names.txt", first, out_dir)
    write_list("surnames.txt", last, out_dir)
    write_list("genre_titles.txt", titles, out_dir)
    write_phrases(phrases, out_dir)
    return {
        "nouns": len(nouns),
        "adjectives": len(adjectives),
        "proper_nouns": len(proper),
        "first_names": len(first),
        "surnames": len(last),
        "genre_titles": len(titles),
        "phrases": len(phrases),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Logo Wizard word lists and title phrases.")
    parser.add_argument("--offline", action="store_true", help="do not download; use files in local/raw")
    parser.add_argument("--count", type=int, default=120_000, help="number of phrases to produce")
    parser.add_argument("--seed", type=int, default=1890)
    args = parser.parse_args(argv)
    if not args.offline:
        fetch_all(RAW_DIR)
    try:
        counts = build(count=args.count, seed=args.seed)
    except FileNotFoundError as err:
        raise SystemExit(str(err))
    for name, n in counts.items():
        print(f"{name:>14}: {n:,}")
