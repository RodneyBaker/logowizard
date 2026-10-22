# Logo Wizard

A local Windows app that turns text into comic-book and novel-cover style title logos (raster PNG), using small models trained from scratch on synthetic data.

- Letters are always drawn by real, openly licensed fonts, so spelling is exact.
- A small layout model picks font, warp, palette and effects; a small diffusion "stylizer" adds the finish.
- Text sources are public domain (King James Bible, Webster's 1913, Census name lists, Moby word lists, Gutenberg fiction titles).
- Generated datasets and trained weights stay on the local machine in `local/` (gitignored).

Target hardware: Windows, NVIDIA GTX 1080 (8 GB), 16 GB RAM. Runs without a GPU using the plain render.

See `docs/design-plan.md` for the full plan.
