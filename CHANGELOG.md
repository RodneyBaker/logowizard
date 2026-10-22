# Changelog

## 0.1.0 (first release)

Logo Wizard turns a title into comic-book and novel-cover title logos on your own Windows PC. Everything runs locally; nothing is sent anywhere.

- **App**: type a title, optional style words ("western comic", "gothic novel") and a subtitle, and get 12 candidates. Refine any of them (font, palette, warp, fill, outlines, 3D, shadow, halftone, tilt) and export PNG at 512, 1024 or 2048 px, or the design as JSON.
- **Exact lettering**: every letter is drawn from one of 264 openly licensed font families (SIL OFL 1.1, Apache 2.0, Ubuntu Font License), bundled with their licenses.
- **Comic and cover effects**: arch, valley, bulge, wave, perspective, fan and stagger warps; stacked outlines, 3D extrusion, drop shadows, gradient and foil fills, inline highlights, halftone dots and distressed or aged textures.
- **Your own models**: the layout model and the diffusion stylizer are trained on your PC from a synthetic dataset (see INSTALL.md). Drop the exported `.onnx` files into `%LOCALAPPDATA%\LogoWizard\models` and the app uses them, on the GPU through DirectML. Without them it uses its hand-written title rules.
- **Learns from you**: thumbs up/down and the Blind test tab (model against rules) are saved locally, and `logo-wizard-finetune-layout` folds them back into the model.
- **Windows package**: a one-folder build with no Python or PyTorch needed, as a setup.exe (per-user, no admin) or a zip with `install.ps1`.

Known limits: the stylizer works at 128x80 and is upscaled for export; CPU-only stylizing takes several seconds per image.
