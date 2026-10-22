"""Real titles of public-domain genre fiction from the Project Gutenberg catalog."""

import csv
import io
import re

GENRES = re.compile(
    r"adventure|science fiction|fantasy|horror|ghost|gothic|detective|mystery|"
    r"pirate|western stories|sea stories|superhero|occult|vampire|war stories",
    re.I,
)
_SUBTITLE = re.compile(r"\s*(?:[:;\n\r]|\s--\s|\s—\s|,\s*or,?\s).*$", re.I | re.S)
_ALLOWED = re.compile(r"^[A-Za-z][A-Za-z' .&!?-]*$")


def clean_title(raw: str, max_words: int = 6) -> str | None:
    title = _SUBTITLE.sub("", raw).strip(" .,")
    title = re.sub(r"\s+", " ", title)
    if not title or not _ALLOWED.match(title):
        return None
    if len(title.split()) > max_words:
        return None
    return title


def genre_titles(catalog_csv: str) -> list[str]:
    """Clean, de-duplicated English genre-fiction titles, in catalog order."""
    seen: set[str] = set()
    titles: list[str] = []
    for row in csv.DictReader(io.StringIO(catalog_csv)):
        if row.get("Type") != "Text" or row.get("Language") != "en":
            continue
        tags = f"{row.get('Subjects', '')} {row.get('Bookshelves', '')}"
        if not GENRES.search(tags):
            continue
        title = clean_title(row.get("Title", ""))
        if title and title.lower() not in seen:
            seen.add(title.lower())
            titles.append(title)
    return titles
