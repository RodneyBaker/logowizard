"""First names and surnames from US Census Bureau frequency lists."""

import csv
import io
import zipfile
from pathlib import Path


def surnames(zip_path: Path, limit: int = 5000) -> list[str]:
    """Most common 2010 surnames, title-cased, most frequent first."""
    with zipfile.ZipFile(zip_path) as archive:
        member = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        text = archive.read(member).decode("utf-8", errors="replace")
    names = []
    for row in csv.DictReader(io.StringIO(text)):
        name = (row.get("name") or "").strip()
        if name.isalpha():
            names.append(name.title())
        if len(names) >= limit:
            break
    return names


def first_names(text: str, limit: int = 1000) -> list[str]:
    """1990 first-name list: whitespace-separated name, frequency, cumulative, rank."""
    names = []
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0].isalpha():
            names.append(fields[0].title())
        if len(names) >= limit:
            break
    return names
