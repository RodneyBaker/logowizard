"""Download sources once into the local raw cache."""

import shutil
import urllib.request
from pathlib import Path

from .sources import SOURCES, Source

USER_AGENT = "logo-wizard-corpus/0.1 (personal research; one download per file)"


def fetch(source: Source, raw_dir: Path, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / source.filename
    if target.exists() and not force:
        return target
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(source.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out)
    partial.replace(target)
    return target


def fetch_all(raw_dir: Path, force: bool = False) -> dict[str, Path]:
    return {key: fetch(source, raw_dir, force) for key, source in SOURCES.items()}
