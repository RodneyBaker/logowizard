import zipfile

import pytest
from collections import Counter

from logo_wizard.corpus import build, catalog, census, kjv, webster
from logo_wizard.corpus.gutenberg import strip_boilerplate
from logo_wizard.corpus.sources import SOURCES
from logo_wizard.corpus.titles import WordLists, fix_articles, generate, title_case

KJV_SAMPLE = """﻿The Project Gutenberg eBook of The King James Bible
License text here.
*** START OF THE PROJECT GUTENBERG EBOOK THE KING JAMES BIBLE ***

1:1 In the beginning God created the heaven and the earth. And Moses went
down to Babylon with the iron sword.

1:2 And the sword of Moses was in Babylon; and the king of Babylon saw
Moses at the river.

1:3 Then Moses spake unto the king of Babylon by the dark river: Light the lamp.

*** END OF THE PROJECT GUTENBERG EBOOK THE KING JAMES BIBLE ***
More license text.
"""

WEBSTER_SAMPLE = """*** START OF THE PROJECT GUTENBERG EBOOK WEBSTER ***
SWORD
Sword, n. Etym: [OE. swerd.]
A weapon with a long blade.

DARK
Dark, a. [OE. derk.]
Destitute of light.

DARK
Dark, n.
Absence of light.

SPAKE
Spake, v. t.
Old past tense of speak.

FOR
For, prep. Etym: [AS. for.]
In the place of.

FOR
For, n.
One who takes the affirmative side.

THAT
That, pron., a., conj., & adv.
A demonstrative.

WENT
Went,

Defn: imp. & p. p. of Wend.

SEA SERPENT
Sea serpent, n.
A huge sea monster.
*** END OF THE PROJECT GUTENBERG EBOOK WEBSTER ***
"""

CATALOG_SAMPLE = '''Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves
1,Text,1990-01-01,"Treasure Island",en,"Stevenson, Robert Louis","Adventure stories; Pirates -- Fiction",PR,Adventure
2,Text,1990-01-01,"The Time Machine: An Invention",en,"Wells, H. G.","Science fiction; Time travel -- Fiction",PR,Science Fiction
3,Text,1990-01-01,"A Treatise on Soil Chemistry",en,"Someone","Soils",S,
4,Text,1990-01-01,"Le Fantôme",fr,"Leroux","Horror tales",PQ,
5,Sound,1990-01-01,"Dracula",en,"Stoker","Horror tales; Vampires",PR,Gothic Fiction
6,Text,1990-01-01,"Dracula's Guest",en,"Stoker","Horror tales; Vampires",PR,Gothic Fiction
'''


def test_strip_boilerplate_keeps_only_body():
    body = strip_boilerplate(KJV_SAMPLE)
    assert body.startswith("1:1 In the beginning")
    assert "License" not in body and "END OF" not in body


def test_verses_join_wrapped_lines():
    verses = kjv.verses(strip_boilerplate(KJV_SAMPLE))
    assert len(verses) == 3
    assert verses[0].endswith("with the iron sword.")


def test_proper_nouns_skip_sentence_starts_and_common_words():
    verses = kjv.verses(strip_boilerplate(KJV_SAMPLE))
    names = kjv.proper_nouns(verses, min_count=2)
    assert set(names) == {"Moses", "Babylon"}
    assert kjv.word_counts(verses)["river"] == 2


def test_webster_headwords_and_parts_of_speech():
    parts = webster.parse_headwords(strip_boilerplate(WEBSTER_SAMPLE))
    assert parts["sword"] == {"noun"}
    assert parts["dark"] == {"adjective", "noun"}
    assert parts["spake"] == {"verb"}
    assert "sea serpent" not in parts
    assert parts["for"] == {"function", "noun"}
    assert parts["that"] == {"function"}
    assert parts["went"] == {"function"}


def test_catalog_keeps_english_genre_text_titles_without_subtitles():
    assert catalog.genre_titles(CATALOG_SAMPLE) == ["Treasure Island", "The Time Machine", "Dracula's Guest"]


def test_clean_title_drops_alternate_titles():
    assert catalog.clean_title("Frankenstein; Or, The Modern Prometheus") == "Frankenstein"
    assert catalog.clean_title("Ivanhoe, or, The Jew and his Daughter") == "Ivanhoe"


def test_census_parsers(tmp_path):
    archive = tmp_path / "names.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Names_2010Census.csv", "name,rank,count\nSMITH,1,2442977\nJOHNSON,2,1932812\nALL OTHER NAMES,0,29312001\n")
    assert census.surnames(archive) == ["Smith", "Johnson"]
    assert census.first_names("MARY           2.629  2.629      1\nPATRICIA       1.073  3.702      2\n") == ["Mary", "Patricia"]


def test_title_case():
    assert title_case("the sword of the dark river") == "The Sword of the Dark River"


def test_fix_articles():
    assert fix_articles("A Adventuress in Heber") == "An Adventuress in Heber"
    assert fix_articles("A Sword in Babylon") == "A Sword in Babylon"


def test_generate_is_unique_and_deterministic():
    words = WordLists(
        nouns=["Sword", "River", "Lamp", "Tower", "Comet"],
        adjectives=["Dark", "Iron", "Golden"],
        proper=["Babylon", "Moses"],
        first=["Mary", "John"],
        last=["Smith", "Johnson"],
    )
    first = generate(words, count=200, seed=7, real_titles=["Treasure Island"])
    again = generate(words, count=200, seed=7, real_titles=["Treasure Island"])
    assert first == again
    assert first[0].text == "Treasure Island" and first[0].origin == "gutenberg-title"
    texts = [p.text.lower() for p in first]
    assert len(texts) == len(set(texts)) == 200
    assert {p.style for p in first} == {"novel", "comic"}


def test_generate_rejects_too_few_words():
    words = WordLists(nouns=["Sword"], adjectives=["Dark"], proper=["Babylon"], first=["Mary"], last=["Smith"])
    with pytest.raises(ValueError, match="nouns"):
        generate(words, count=10)


def test_pick_vocabulary_filters_rare_and_blocked_words():
    parts = {
        "sword": {"noun"},
        "dark": {"adjective", "noun"},
        "zymurgy": {"noun"},
        "harlot": {"noun"},
        "for": {"function", "noun"},
        "will": {"verb", "noun"},
    }
    usage = Counter({"sword": 10, "dark": 5, "harlot": 9, "zymurgy": 0, "for": 99, "will": 50})
    nouns, adjectives = build.pick_vocabulary(parts, usage, blocklist={"harlot"})
    assert nouns == ["Sword", "Dark"]
    assert adjectives == ["Dark"]


def test_build_end_to_end_offline(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (raw / SOURCES["kjv"].filename).write_text(
        KJV_SAMPLE.replace("*** END", "1:4 The dark iron lamp burned by the river.\n\n*** END"), encoding="utf-8"
    )
    (raw / SOURCES["webster"].filename).write_text(WEBSTER_SAMPLE.replace("*** END", "RIVER\nRiver, n.\nA stream.\n\nIRON\nIron, a.\nOf iron.\n*** END"), encoding="utf-8")
    (raw / SOURCES["catalog"].filename).write_text(CATALOG_SAMPLE, encoding="utf-8")
    with zipfile.ZipFile(raw / SOURCES["surnames"].filename, "w") as zf:
        zf.writestr("Names_2010Census.csv", "name,rank,count\nSMITH,1,2\nJONES,2,1\n")
    (raw / SOURCES["first_female"].filename).write_text("MARY 2.6 2.6 1\n", encoding="utf-8")
    (raw / SOURCES["first_male"].filename).write_text("JAMES 3.3 3.3 1\n", encoding="utf-8")

    counts = build.build(raw_dir=raw, out_dir=out, count=50, seed=1)
    assert counts["phrases"] == 50
    assert (out / "phrases.tsv").read_text(encoding="utf-8").startswith("text\tstyle\tsubtitle\torigin\n")
    assert "Sword" in (out / "nouns.txt").read_text(encoding="utf-8")
