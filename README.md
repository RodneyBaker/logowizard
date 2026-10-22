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

## Layout model (milestone 4)

Install the PyTorch build that still has kernels for the GTX 1080 (Pascal, sm_61) before the rest of the training extras. Newer CUDA 12.8+ builds dropped Pascal, so the project pins the CUDA 12.6 build:

```
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -e .[train,app,dev]
python scripts/check_gpu.py        # confirms the GPU works and measures fp32 speed, VRAM and step times
```

Then, once `local/dataset` exists:

```
logo-wizard-train-layout                      # about 17M parameters, batch 256, 12 epochs; resumable
logo-wizard-train-layout --max-minutes 180    # stop after 3 hours, run the same command to carry on
logo-wizard-layout-compare --n 40             # blind sheet: model vs hand-written rules
logo-wizard-layout-compare --score A,B,A,...  # your picks, one per row
```

The model reads the style words and the title as characters and writes the spec one field at a time (style, font, palette, case, arrangement, warp, effects...). At each step it can only choose that field's values, so every output is a valid spec, and fonts or palettes missing on the machine are never proposed. Checkpoints and `log.csv` (loss and per-field accuracy each epoch) go to `local/runs/layout/`; the app loads `best.pt`. Training runs in fp32; the training loop saves on Ctrl+C.

## Stylizer (milestone 5)

A 35M parameter pixel-space diffusion UNet that paints the finished logo around the plain glyph mask, conditioned on the spec's look (style, palette, fill, texture, halftone, outlines, inline, 3D). It trains at 128x80 in fp32 with gradient checkpointing, which keeps batch 16 inside the GTX 1080's 8 GB.

```
logo-wizard-train-stylizer --max-minutes 480     # one overnight session; run again to continue
logo-wizard-eval-stylizer --n 200                # OCR readability of stylized output vs the renderer
```

Every 2,000 steps it saves `local/runs/stylizer/last.pt` (full state) and `ema.pt` (the smoothed weights the app uses); every 10,000 steps it writes `preview_<step>.png` with mask, stylized and target columns. `log.csv` records loss, samples per second and peak VRAM, which gives the measured hardware numbers for the plan. The milestone is done when `eval.json` shows `stylized_readable` above 0.95 and the sheet looks richer than the plain fill. For the 256x160 stretch goal: `--size 256x160 --batch-size 6 --run local/runs/stylizer-256`.

At export the stylizer output is upscaled with a light sharpen, and letter interiors are re-cut from a full-size glyph mask so the shapes stay crisp.

## The app, feedback and ONNX export (milestone 6)

```
logo-wizard-app                    # opens http://127.0.0.1:8765 (or the next free port)
```

The Create tab takes a title, optional style words ("western comic", "gothic novel") and an optional subtitle, and shows 12 candidates. "Ideas from" picks where they come from: the trained layout model, the hand-written rules, or a mix of both (the default when a model is found). Click a candidate to refine every field (font, palette, warp and amount, fill, outlines, 3D depth, shadow, halftone, tilt...), tick "Finish with the stylizer" for the diffusion finish, and export PNG at 512, 1024 or 2048 px wide, or the spec as JSON.

Thumbs up and down, exports and Blind test picks are appended to `local/feedback/feedback.jsonl`. The Blind test tab shows one take from the model and one from the rules side by side, in random order, and keeps a running tally of how often the model wins. Once you have 20 or more votes:

```
logo-wizard-finetune-layout        # liked specs are trained toward, disliked ones pushed away
```

It keeps the previous weights as `best.before-feedback.pt`, so a bad round is one file rename to undo.

To run the app without PyTorch (the Windows build does this), export the models to ONNX:

```
logo-wizard-export-onnx            # local/models/layout.onnx and stylizer.onnx, plus their .json configs
```

The app prefers `local/models/*.onnx` (onnxruntime, DirectML on Windows so the GTX 1080 is used) and falls back to the PyTorch checkpoints in `local/runs/`. Set `LOGO_WIZARD_HOME` to keep fonts, datasets, runs, models and feedback somewhere other than `local/`.
