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
