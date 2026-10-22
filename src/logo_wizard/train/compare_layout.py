"""Blind side-by-side test: trained layout model vs the hand-written grammar.

Usage:
    logo-wizard-layout-compare --n 40                 # writes local/runs/layout/compare/
    logo-wizard-layout-compare --score A,B,B,A,...    # after picking, prints the tally

Each row of compare.png shows the same title drawn by both, in random
order. Write down A (left) or B (right) for the one you prefer, row by row,
then pass the picks to --score. key.json says which side was the model.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..models.layout.predict import DEFAULT_WEIGHTS, LayoutPredictor
from ..render.draw import Renderer, on_background
from ..render.grammar import sample_spec
from ..synth.generate import load_phrases, style_words_for
from .common import RUNS_DIR

OUT = RUNS_DIR / "layout" / "compare"


def build(n: int, weights: Path, out: Path, seed: int) -> Path:
    renderer = Renderer()
    predictor = LayoutPredictor(weights, set(renderer.faces), set(renderer.palettes))
    phrases = load_phrases()
    rng = np.random.default_rng(seed)
    w, h = 384, 240
    sheet = Image.new("RGB", (2 * w + 40, n * h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    key = []
    for row in range(n):
        text, style, subtitle = phrases[int(rng.integers(len(phrases)))]
        words = style_words_for(style, rng)
        rules = sample_spec(text, renderer.faces, renderer.palettes, rng, style=style, style_words=words)
        model = predictor.suggest(text, words, n=1, seed=int(rng.integers(2**31)))[0]
        model_left = bool(rng.random() < 0.5)
        left, right = (model, rules) if model_left else (rules, model)
        for col, spec in enumerate((left, right)):
            image = renderer.render(spec, w, h).image
            sheet.paste(on_background(image), (40 + col * w, row * h))
        draw.text((8, row * h + h // 2), str(row + 1), fill=(0, 0, 0))
        key.append({"row": row + 1, "text": text, "words": words, "model": "A" if model_left else "B"})
    out.mkdir(parents=True, exist_ok=True)
    sheet.save(out / "compare.png")
    (out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    return out / "compare.png"


def score(picks: str, out: Path) -> str:
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    choices = [p.strip().upper() for p in picks.replace(" ", ",").split(",") if p.strip()]
    model_wins = sum(1 for k, c in zip(key, choices) if c == k["model"])
    total = min(len(key), len(choices))
    verdict = "model is preferred" if model_wins > total / 2 else "tie" if model_wins * 2 == total else "rules are preferred"
    return f"Model picked {model_wins} of {total} times ({verdict})."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--score", metavar="PICKS", help="comma-separated A/B picks, one per row")
    args = parser.parse_args(argv)
    if args.score:
        print(score(args.score, args.out))
    else:
        print(build(args.n, args.weights, args.out, args.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
