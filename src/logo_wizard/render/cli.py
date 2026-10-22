"""Render title logos from the command line with the procedural grammar.

Usage:
    logo-wizard-render "The Iron Comet"                      # 8 comic candidates
    logo-wizard-render "Daughter of Babylon" --style novel --subtitle "A Novel"
    logo-wizard-render "Crypt Tales" --words "horror pulp" -n 12 --size 768x512
    logo-wizard-render --spec local/out/logo_03.json         # re-render a saved spec

Writes logo_NN.png (transparent), logo_NN.json (the spec) and sheet.png.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .. import paths
from .draw import Renderer, on_background
from .grammar import read_style, sample_spec
from .spec import DesignSpec



def contact_sheet(images: list[np.ndarray], columns: int = 4, background=None) -> Image.Image:
    h, w = images[0].shape[:2]
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * w, rows * h), (255, 255, 255))
    for i, image in enumerate(images):
        sheet.paste(on_background(image, background), ((i % columns) * w, (i // columns) * h))
    return sheet


def parse_size(text: str) -> tuple[int, int]:
    w, _, h = text.lower().partition("x")
    return int(w), int(h or w)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?", help="the title to draw")
    parser.add_argument("--style", choices=["comic", "novel"], help="default: read from --words, else comic")
    parser.add_argument("--words", default="", help="style words, e.g. 'gothic novel' or 'sci-fi comic'")
    parser.add_argument("--subtitle", default="")
    parser.add_argument("-n", "--count", type=int, default=8)
    parser.add_argument("--size", default="768x512", help="WIDTHxHEIGHT in pixels")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--spec", type=Path, help="render a saved spec JSON instead of sampling")
    parser.add_argument("--out", type=Path, default=paths.OUT_DIR)
    args = parser.parse_args(argv)

    width, height = parse_size(args.size)
    renderer = Renderer()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.spec:
        spec = DesignSpec.from_json(args.spec.read_text(encoding="utf-8"))
        result = renderer.render(spec, width, height)
        target = args.out / (args.spec.stem + f"_{width}x{height}.png")
        result.to_pil().save(target)
        print(target)
        return 0
    if not args.text:
        parser.error("give a title, or --spec")

    rng = np.random.default_rng(args.seed)
    style = args.style or read_style(args.words)
    images = []
    for i in range(args.count):
        spec = sample_spec(args.text, renderer.faces, renderer.palettes, rng, style=style, style_words=args.words, subtitle=args.subtitle)
        result = renderer.render(spec, width, height)
        result.to_pil().save(args.out / f"logo_{i:02d}.png")
        (args.out / f"logo_{i:02d}.json").write_text(spec.to_json(), encoding="utf-8")
        images.append(result.image)
    contact_sheet(images).save(args.out / "sheet.png")
    print(f"{args.count} logos -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
