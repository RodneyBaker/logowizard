"""Download the curated families and write the font manifest.

Usage (from the repo root):
    logo-wizard-fonts              # download into local/fonts, write data/fonts/manifest.csv
    logo-wizard-fonts --offline    # rebuild the manifest from files already downloaded

Fonts come from the Google Fonts repository. Only the ofl/, apache/ and ufl/
folders are tried, and the license named in each family's METADATA.pb must
match its folder, so nothing outside SIL OFL 1.1, Apache 2.0 and the Ubuntu
Font License gets in. Each face is checked for the characters titles use.
"""

import argparse
import csv
import hashlib
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import uharfbuzz as hb

from .. import paths
from .catalog import FamilyEntry, unique_families

FONT_DIR = paths.LOCAL_ROOT / "fonts"
MANIFEST = paths.FONT_MANIFEST

BASE_URL = "https://raw.githubusercontent.com/google/fonts/main"
LICENSE_DIRS = {"ofl": "OFL", "apache": "APACHE2", "ufl": "UFL"}
LICENSE_NAMES = {"OFL": "SIL OFL 1.1", "APACHE2": "Apache 2.0", "UFL": "Ubuntu Font License 1.0"}
LICENSE_FILES = {"OFL": "OFL.txt", "APACHE2": "LICENSE.txt", "UFL": "UFL.txt"}
USER_AGENT = "logo-wizard-fonts/0.1 (personal research; one download per file)"

UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER = UPPER.lower()
REQUIRED = UPPER + "0123456789"
PUNCTUATION = "'-!?.,:&"


@dataclass
class Face:
    face_id: str
    family: str
    dirname: str
    file: str
    weight: int
    variation: str
    license: str
    source_url: str
    sha256: str
    font_class: str
    moods: str
    has_lower: bool
    has_punct: bool


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def parse_metadata(text: str) -> dict:
    """Pull the fields we need out of a METADATA.pb text proto."""
    name = re.search(r'^name: "([^"]+)"', text, re.M)
    license_ = re.search(r'^license: "([^"]+)"', text, re.M)
    fonts = []
    for block in re.findall(r"^fonts \{(.*?)^\}", text, re.M | re.S):
        style = re.search(r'style: "([^"]+)"', block)
        weight = re.search(r"weight: (\d+)", block)
        filename = re.search(r'filename: "([^"]+)"', block)
        if style and weight and filename:
            fonts.append({"style": style.group(1), "weight": int(weight.group(1)), "filename": filename.group(1)})
    axes = {}
    for block in re.findall(r"^axes \{(.*?)^\}", text, re.M | re.S):
        tag = re.search(r'tag: "([^"]+)"', block)
        lo = re.search(r"min_value: ([\d.]+)", block)
        hi = re.search(r"max_value: ([\d.]+)", block)
        if tag and lo and hi:
            axes[tag.group(1)] = (float(lo.group(1)), float(hi.group(1)))
    return {
        "name": name.group(1) if name else "",
        "license": license_.group(1) if license_ else "",
        "fonts": fonts,
        "axes": axes,
    }


def pick_faces(meta: dict) -> list[tuple[str, int, str]]:
    """(filename, weight, variation) for a regular and a heavy face of the family."""
    upright = [f for f in meta["fonts"] if f["style"] == "normal"]
    if not upright:
        return []
    variable = [f for f in upright if "[" in f["filename"]]
    if variable:
        filename = variable[0]["filename"]
        lo, hi = meta["axes"].get("wght", (400, 400))
        weights = sorted({int(min(max(400, lo), hi)), int(min(hi, 800))})
        return [(filename, w, f"wght={w}" if "wght" in meta["axes"] else "") for w in weights]
    by_weight = sorted(upright, key=lambda f: f["weight"])
    regular = min(by_weight, key=lambda f: abs(f["weight"] - 400))
    heavy = max((f for f in by_weight if f["weight"] <= 900), key=lambda f: f["weight"])
    picks = [regular] if regular is heavy else [regular, heavy]
    return [(f["filename"], f["weight"], "") for f in picks]


def coverage(path: Path) -> tuple[bool, bool, bool]:
    """(has the required characters, has lowercase, has title punctuation)."""
    face = hb.Face(path.read_bytes())
    cmap = hb.Font(face).face.unicodes
    def has(chars: str) -> bool:
        return all(ord(c) in cmap for c in chars)
    return has(REQUIRED), has(LOWER), has(PUNCTUATION)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_family(entry: FamilyEntry, font_dir: Path, offline: bool) -> tuple[list[Face], str]:
    """Download one family. Returns its faces and a note for the log."""
    family_dir = font_dir / entry.dirname
    meta_path = family_dir / "METADATA.pb"
    license_dir = None
    if meta_path.exists():
        license_dir = (family_dir / "LICENSE_DIR").read_text().strip()
    elif not offline:
        for candidate in LICENSE_DIRS:
            try:
                data = _get(f"{BASE_URL}/{candidate}/{entry.dirname}/METADATA.pb")
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    continue
                raise
            family_dir.mkdir(parents=True, exist_ok=True)
            meta_path.write_bytes(data)
            (family_dir / "LICENSE_DIR").write_text(candidate)
            license_dir = candidate
            break
    if license_dir is None:
        return [], f"{entry.dirname}: not found in ofl/, apache/ or ufl/"

    meta = parse_metadata(meta_path.read_text(encoding="utf-8"))
    if meta["license"] != LICENSE_DIRS[license_dir]:
        return [], f"{entry.dirname}: METADATA license {meta['license']!r} does not match {license_dir}/"

    base = f"{BASE_URL}/{license_dir}/{entry.dirname}"
    license_file = LICENSE_FILES[meta["license"]]
    targets = [license_file] + sorted({filename for filename, _, _ in pick_faces(meta)})
    for filename in targets:
        target = family_dir / filename
        if target.exists() or offline:
            continue
        try:
            target.write_bytes(_get(f"{base}/{urllib.request.quote(filename)}"))
        except urllib.error.HTTPError as err:
            if filename != license_file:
                return [], f"{entry.dirname}: {filename} download failed ({err.code})"

    faces = []
    for filename, weight, variation in pick_faces(meta):
        path = family_dir / filename
        if not path.exists():
            return [], f"{entry.dirname}: {filename} missing (run without --offline)"
        required, has_lower, has_punct = coverage(path)
        if not required:
            return [], f"{entry.dirname}: lacks A-Z or 0-9"
        faces.append(
            Face(
                face_id=f"{entry.dirname}-{weight}",
                family=meta["name"],
                dirname=entry.dirname,
                file=f"{entry.dirname}/{filename}",
                weight=weight,
                variation=variation,
                license=LICENSE_NAMES[meta["license"]],
                source_url=f"{base}/{filename}",
                sha256=sha256(path),
                font_class=entry.font_class,
                moods=" ".join(entry.moods),
                has_lower=has_lower,
                has_punct=has_punct,
            )
        )
    return faces, ""


def write_manifest(faces: list[Face], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=[f.name for f in fields(Face)])
        writer.writeheader()
        for face in sorted(faces, key=lambda f: f.face_id):
            writer.writerow(asdict(face))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="only use files already in local/fonts")
    parser.add_argument("--font-dir", type=Path, default=FONT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    families = unique_families()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda e: fetch_family(e, args.font_dir, args.offline), families))
    faces = [face for got, _ in results for face in got]
    for _, note in results:
        if note:
            print("skipped", note, file=sys.stderr)
    write_manifest(faces, args.manifest)
    kept = len({f.dirname for f in faces})
    print(f"{kept} of {len(families)} families, {len(faces)} faces -> {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
