# Design plan

The living design plan is the Claude doc "Textual Logo Model: Design Plan":
https://claude.ai/artifact/DACQTGD5ahQZtnK6Lpg5Q8

## Key decisions (2026-10-22)

- **Approach:** a small layout model (text -> JSON design spec) + an exact font renderer + a small glyph-conditioned diffusion stylizer, all trained from scratch on synthetic data. Fonts draw the letters, so spelling is always exact.
- **Logos:** text only, no emblems. The letters themselves can be warped, reshaped, outlined and extruded, like comic book titles.
- **Styles:** comic book titles and novel covers first.
- **Data:** public-domain text (King James Bible, Webster's 1913, Census names, Moby word lists, Gutenberg fiction titles) and OFL/Apache/UFL fonts. Generated datasets and trained weights stay local in `local/` (gitignored).
- **Hardware:** Windows only; NVIDIA GTX 1080 (8 GB, Pascal), i5-9400F, 16 GB RAM. Train in fp32, stylizer at 128 px, pin a PyTorch/CUDA build that still supports Pascal (sm_61).

## Milestones

1. Repo and corpus
2. Fonts and renderer (procedural-only app works on CPU)
3. Synthetic dataset (250k samples, OCR and contrast filters)
4. Layout model
5. Stylizer
6. App and feedback loop
7. Windows packaging
