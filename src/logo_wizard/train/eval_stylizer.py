"""Check the stylizer: OCR readability of its output, and a side-by-side sheet.

Usage:
    logo-wizard-eval-stylizer --n 200                  # uses local/runs/stylizer/ema.pt and local/dataset

Prints the share of stylized logos Tesseract can still read (the milestone
target is above 95%), next to the same share for the renderer's own
samples, and writes eval.png with columns: plain fill, stylized, renderer.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from ..infer.stylize import DEFAULT_WEIGHTS, Stylizer
from ..render.draw import Renderer, on_background
from ..render.spec import DesignSpec
from ..synth import filters
from ..synth.generate import OUT_DIR, iter_records


def plain(spec: DesignSpec) -> DesignSpec:
    """The same letters with no effects: what the stylizer has to beat."""
    return DesignSpec(text=spec.text, style=spec.style, subtitle=spec.subtitle, font=spec.font, subtitle_font=spec.subtitle_font,
                      palette=spec.palette, case=spec.case, arrangement=spec.arrangement, warp=spec.warp,
                      warp_amount=spec.warp_amount, tracking=spec.tracking, tilt=spec.tilt, fill_mode="flat", seed=spec.seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=OUT_DIR)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--min-ocr", type=float, default=0.35)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None, help="default: next to the weights")
    args = parser.parse_args(argv)

    renderer = Renderer()
    stylizer = Stylizer(args.weights, renderer, device=args.device)
    records = [rec for _, rec in iter_records(args.data)]
    # The last records are the least likely to have been seen in a short run.
    records = records[-args.n:]
    if not records:
        print(f"No samples under {args.data}", file=sys.stderr)
        return 1
    if not filters.tesseract_available():
        print("Tesseract is needed for this check.", file=sys.stderr)
        return 1
    out_size = (stylizer.size[0] * 2, stylizer.size[1] * 2)
    read_model = read_render = 0
    rows = []
    for i in range(0, len(records), args.batch):
        batch = records[i : i + args.batch]
        specs = [DesignSpec.from_dict(r["spec"]) for r in batch]
        images = stylizer.stylize(specs, out_size, steps=args.steps, guidance=args.guidance, seed=i)
        for rec, spec, image in zip(batch, specs, images):
            expected = f"{rec['text']} {rec['subtitle']}"
            reference = renderer.render(spec, *out_size).image
            read_model += filters.ocr_score(image, expected) >= args.min_ocr
            read_render += filters.ocr_score(reference, expected) >= args.min_ocr
            if len(rows) < 24:
                rows.append((renderer.render(plain(spec), *out_size).image, image, reference))
        print(f"{min(i + args.batch, len(records))}/{len(records)}", flush=True)
    n = len(records)
    result = {"samples": n, "stylized_readable": round(read_model / n, 3), "renderer_readable": round(read_render / n, 3), "target": 0.95}
    out = args.out or args.weights.parent
    w, h = out_size
    sheet = Image.new("RGB", (3 * w, len(rows) * h), (255, 255, 255))
    for r, row in enumerate(rows):
        for c, image in enumerate(row):
            sheet.paste(on_background(image), (c * w, r * h))
    sheet.save(out / "eval.png")
    (out / "eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Sheet: {out / 'eval.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
