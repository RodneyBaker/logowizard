"""Generate the synthetic training set: render + glyph mask + spec per sample.

Usage (from the repo root):
    logo-wizard-synth --count 250000                 # about a day on 6 cores
    logo-wizard-synth --count 2000 --out local/dataset-small
    logo-wizard-synth --preview 48                   # contact sheet of the first kept samples
    logo-wizard-synth --benchmark 200                # time the generator on this machine

Samples are written in shards (1000 by default) under local/dataset/:
    shard_00012/00012345.png        finished logo, RGBA, transparent background
    shard_00012/00012345_mask.png   plain glyph mask (L)
    shard_00012/specs.jsonl         one record per kept sample
    shard_00012/DONE                written last; finished shards are skipped on a re-run

Every sample is seeded by (seed, index), so the same command rebuilds the
same dataset. Samples that fail a filter are counted in shard stats, not kept.
"""

import argparse
import csv
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .. import paths
from ..render.draw import Renderer
from ..render.grammar import sample_spec
from . import filters

PHRASES = paths.PHRASES
OUT_DIR = paths.DATASET_DIR
SHARD_SIZE = 1000
WIDTH, HEIGHT = 256, 160

STYLE_WORDS = {
    "comic": ["comic", "comics", "pulp", "superhero", "cartoon"],
    "novel": ["novel", "book", "classic", "literary", "victorian"],
}
MOOD_STYLE_WORDS = {
    "horror": ["horror", "gothic", "haunted"],
    "scifi": ["sci-fi", "science fiction", "space"],
    "western": ["western", "frontier"],
    "fantasy": ["fantasy", "magic", "myth"],
    "elegant": ["romance", "elegant"],
    "pulp": ["pulp", "noir", "mystery"],
    "playful": ["fun", "cartoon"],
    "retro": ["retro", "golden age"],
}


@dataclass
class Settings:
    out: Path
    seed: int
    width: int
    height: int
    min_contrast: float
    min_coverage: float
    ocr: bool
    min_ocr: float
    shard_size: int = SHARD_SIZE


def load_phrases(path: Path = PHRASES) -> list[tuple[str, str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["text"], r["style"], r["subtitle"]) for r in csv.DictReader(f, delimiter="\t")]


def style_words_for(style: str, rng: np.random.Generator) -> str:
    """Optional words a user might type next to the title."""
    roll = rng.random()
    if roll < 0.35:
        return ""
    style_word = STYLE_WORDS[style][int(rng.integers(len(STYLE_WORDS[style])))]
    if roll < 0.55:
        return style_word
    mood = list(MOOD_STYLE_WORDS)[int(rng.integers(len(MOOD_STYLE_WORDS)))]
    mood_word = MOOD_STYLE_WORDS[mood][int(rng.integers(len(MOOD_STYLE_WORDS[mood])))]
    return mood_word if roll < 0.8 else f"{mood_word} {style_word}"


def make_sample(index: int, phrases, renderer: Renderer, settings: Settings):
    rng = np.random.default_rng([settings.seed, index])
    text, style, subtitle = phrases[int(rng.integers(len(phrases)))]
    if rng.random() < 0.15:
        style = "novel" if style == "comic" else "comic"
    if rng.random() < 0.5:
        subtitle = ""
    words = style_words_for(style, rng)
    spec = sample_spec(text, renderer.faces, renderer.palettes, rng, style=style, style_words=words, subtitle=subtitle)
    result = renderer.render(spec, settings.width, settings.height)
    return text, subtitle, style, words, spec, result


def check(result, text: str, subtitle: str, settings: Settings) -> tuple[str, dict]:
    """'' if the sample passes, else the name of the failed filter. Also returns the scores."""
    scores = {
        "contrast": round(filters.edge_contrast(result.image, result.mask), 2),
        "coverage": round(filters.coverage(result.image, result.mask), 3),
    }
    if scores["coverage"] < settings.min_coverage:
        return "coverage", scores
    if scores["contrast"] < settings.min_contrast:
        return "contrast", scores
    if settings.ocr:
        scores["ocr"] = round(filters.ocr_score(result.image, f"{text} {subtitle}"), 3)
        if scores["ocr"] < settings.min_ocr:
            return "ocr", scores
    return "", scores


_worker: dict = {}


def _init_worker(settings: Settings) -> None:
    # One Tesseract thread per worker; otherwise its OpenMP threads swamp the CPU.
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    _worker["renderer"] = Renderer()
    _worker["phrases"] = load_phrases()
    _worker["settings"] = settings


def build_shard(shard: int, count: int) -> dict:
    settings: Settings = _worker["settings"]
    shard_dir = settings.out / f"shard_{shard:05d}"
    if (shard_dir / "DONE").exists():
        return json.loads((shard_dir / "DONE").read_text())
    shard_dir.mkdir(parents=True, exist_ok=True)
    stats = {"shard": shard, "kept": 0, "coverage": 0, "contrast": 0, "ocr": 0, "error": 0, "hashes": []}
    start = shard * settings.shard_size
    with open(shard_dir / "specs.jsonl", "w", encoding="utf-8") as out:
        for index in range(start, min(start + settings.shard_size, count)):
            try:
                text, subtitle, style, words, spec, result = make_sample(index, _worker["phrases"], _worker["renderer"], settings)
            except Exception as err:  # a font that cannot shape a rare character, etc.
                stats["error"] += 1
                print(f"sample {index}: {err}", file=sys.stderr)
                continue
            failed, scores = check(result, text, subtitle, settings)
            if failed:
                stats[failed] += 1
                continue
            name = f"{index:08d}"
            result.to_pil().save(shard_dir / f"{name}.png", optimize=False, compress_level=6)
            result.mask_pil().save(shard_dir / f"{name}_mask.png")
            h = filters.dhash(result.image)
            stats["hashes"].append([name, f"{h:016x}"])
            record = {"id": name, "text": text, "subtitle": subtitle, "style": style, "words": words, "spec": spec.to_dict(), **scores}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["kept"] += 1
    (shard_dir / "DONE").write_text(json.dumps(stats))
    return stats


def write_duplicates(out: Path, all_stats: list[dict]) -> int:
    """List near-duplicates (equal dhash) so loaders can skip them; the first copy is kept."""
    seen: set[str] = set()
    dupes = []
    for stats in sorted(all_stats, key=lambda s: s["shard"]):
        for name, h in stats["hashes"]:
            if h in seen:
                dupes.append(name)
            seen.add(h)
    (out / "duplicates.txt").write_text("\n".join(dupes) + ("\n" if dupes else ""), encoding="utf-8")
    return len(dupes)


def iter_records(root: Path, skip_duplicates: bool = True):
    """Yield (shard_dir, record) for every kept sample, in order."""
    dupes = set()
    dupes_file = root / "duplicates.txt"
    if skip_duplicates and dupes_file.exists():
        dupes = set(dupes_file.read_text(encoding="utf-8").split())
    for shard_dir in sorted(root.glob("shard_*")):
        specs = shard_dir / "specs.jsonl"
        if not (shard_dir / "DONE").exists() or not specs.exists():
            continue
        with open(specs, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record["id"] not in dupes:
                    yield shard_dir, record


def preview(root: Path, n: int, target: Path) -> Path:
    from ..render.cli import contact_sheet

    images = []
    for shard_dir, record in iter_records(root):
        images.append(np.asarray(Image.open(shard_dir / f"{record['id']}.png").convert("RGBA")))
        if len(images) >= n:
            break
    if not images:
        raise SystemExit(f"No samples under {root}")
    sheet = contact_sheet(images, columns=8)
    sheet.save(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=250_000, help="samples to attempt (kept ones are fewer)")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", default=f"{WIDTH}x{HEIGHT}")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    parser.add_argument("--min-contrast", type=float, default=1.5)
    parser.add_argument("--min-coverage", type=float, default=0.75)
    parser.add_argument("--min-ocr", type=float, default=0.35, help="Tesseract similarity needed to keep a sample")
    parser.add_argument("--no-ocr", action="store_true", help="skip the Tesseract check")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE, help="keep the same value when resuming a run")
    parser.add_argument("--preview", type=int, metavar="N", help="write a contact sheet of N kept samples and exit")
    parser.add_argument("--benchmark", type=int, metavar="N", help="time N samples in one process and exit")
    args = parser.parse_args(argv)

    if args.preview:
        print(preview(args.out, args.preview, args.out / "preview.png"))
        return 0

    width, _, height = args.size.partition("x")
    ocr = not args.no_ocr
    if ocr and not filters.tesseract_available():
        print("Tesseract not found; running without the OCR check (install it or pass --no-ocr to silence this).", file=sys.stderr)
        ocr = False
    settings = Settings(args.out, args.seed, int(width), int(height or width), args.min_contrast, args.min_coverage, ocr, args.min_ocr, args.shard_size)

    if args.benchmark:
        _init_worker(settings)
        t0 = time.perf_counter()
        render_time = 0.0
        outcome = {"kept": 0}
        for i in range(args.benchmark):
            r0 = time.perf_counter()
            text, subtitle, _, _, _, result = make_sample(i, _worker["phrases"], _worker["renderer"], settings)
            render_time += time.perf_counter() - r0
            failed, _ = check(result, text, subtitle, settings)
            outcome[failed or "kept"] = outcome.get(failed or "kept", 0) + 1
        total = time.perf_counter() - t0
        per = total / args.benchmark
        print(f"{per * 1000:.0f} ms per sample ({render_time / args.benchmark * 1000:.0f} ms render), OCR {'on' if ocr else 'off'}; outcomes {outcome}")
        print(f"Estimate with {args.workers} workers: {args.workers / per * 3600:,.0f} samples per hour")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    shards = (args.count + args.shard_size - 1) // args.shard_size
    t0 = time.perf_counter()
    all_stats = []
    # spawn, as on Windows, so a run here behaves like a run on the training PC.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context, initializer=_init_worker, initargs=(settings,)) as pool:
        for stats in pool.map(build_shard, range(shards), [args.count] * shards):
            all_stats.append(stats)
            kept = sum(s["kept"] for s in all_stats)
            elapsed = time.perf_counter() - t0
            print(f"shard {stats['shard']:05d}: kept {stats['kept']}  (total {kept:,}, {len(all_stats)}/{shards} shards, {elapsed / 60:.1f} min)", flush=True)
    dupes = write_duplicates(args.out, all_stats)
    summary = {k: sum(s[k] for s in all_stats) for k in ("kept", "coverage", "contrast", "ocr", "error")}
    summary["duplicates"] = dupes
    summary["settings"] = {"seed": args.seed, "size": args.size, "ocr": ocr, "min_ocr": args.min_ocr, "min_contrast": args.min_contrast}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
