# Logo Wizard

A local Windows app that turns text into comic-book and novel-cover style title logos (raster PNG), using small models trained from scratch on synthetic data.

- Letters are always drawn by real, openly licensed fonts, so spelling is exact.
- A small layout model picks font, warp, palette and effects; a small diffusion "stylizer" adds the finish.
- Text sources are public domain (King James Bible, Webster's 1913, Census name lists, Gutenberg genre-fiction titles).
- Generated datasets and trained weights stay on the local machine in `local/` (gitignored).

Target hardware: Windows, NVIDIA GTX 1080 (8 GB), 16 GB RAM. Runs without a GPU using the plain render.

See `docs/design-plan.md` for the full plan.

## Build the text corpus (milestone 1)

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
logo-wizard-corpus
```

This downloads the sources once into `local/raw/`, then writes word lists and `phrases.tsv` (about 120,000 comic and novel-cover titles) to `data/corpora/`. Run `pytest` to check the parsers.

## Fonts and the procedural renderer (milestone 2)

```
logo-wizard-fonts                                  # downloads 264 open-license families into local/fonts (about 70 MB)
logo-wizard-render "The Iron Comet"                # 8 comic candidates in local/out, plus sheet.png
logo-wizard-render "Daughter of Babylon" --style novel --subtitle "A Novel"
logo-wizard-render "Crypt Tales" --words "horror pulp" -n 12 --size 1024x640
logo-wizard-app                                    # local web app at http://127.0.0.1:8765
```

The renderer shapes text with HarfBuzz, bends the letter outlines (arch, valley, bulge, wave, perspective, fan, stagger), and draws drop shadows, 3D extrusion, stacked outlines, gradient or foil fills, an inline highlight, halftone dots and distressed or aged textures with Skia. The title grammar (`render/grammar.py`) reads moods from the title and style words (horror, sci-fi, western, fantasy, pulp...) and picks fonts, palettes, warps and effects to suit. This works on CPU with no trained model, and it is the same code that will generate the training data.

## Synthetic dataset (milestone 3)

Install Tesseract for the OCR check: the Windows installer from UB Mannheim (https://github.com/UB-Mannheim/tesseract/wiki), default folder `C:\Program Files\Tesseract-OCR`. Then:

```
pip install -e .[train]
logo-wizard-synth --benchmark 100          # time per sample on this PC
logo-wizard-synth --count 250000           # writes local/dataset (about 10 GB); safe to stop and re-run
logo-wizard-synth --preview 64             # local/dataset/preview.png for a visual spot-check
```

Each sample is a 256x160 transparent PNG, its plain glyph mask and the spec that drew it, seeded by its index so the same command rebuilds the same set. Samples are dropped when Tesseract cannot read the title back (`--min-ocr`, character similarity, default 0.35), when the letter fill has too little contrast against its outlines (`--min-contrast`, WCAG ratio, default 1.5), or when a texture erased too much of the letters. Exact visual duplicates are listed in `duplicates.txt` and skipped by the loaders. With OCR on, a sample takes about 0.6 s of one core, so 250k attempts on 6 cores is roughly 7 hours; `--no-ocr` is about 10 times faster. Run `--benchmark` to measure it on your PC.
