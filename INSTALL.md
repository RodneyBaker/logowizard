# Installing and training Logo Wizard on Windows

There are two parts. The **app** is a normal Windows program you install from the release. The **training kit** is this repository, used on the same PC to build the dataset and train the models the app uses. You can run the app straight away; it gets better once your models are in place.

## 1. Install the app

From the [Releases page](https://github.com/RodneyBaker/logowizard/releases), download either

- `LogoWizard-0.1.0-setup.exe`: run it. It installs for your user only (no admin prompt) and adds Start menu entries, or
- `LogoWizard-0.1.0-win64.zip`: unzip it anywhere, then right-click `install.ps1` > Run with PowerShell.

Start **Logo Wizard** from the Start menu. A console window opens and the app appears in your browser at http://127.0.0.1:8765 (or the next free port). Close the console window to stop it.

Windows SmartScreen may warn that the program is unrecognised, because the build is not code-signed. Choose "More info" > "Run anyway".

Your files live in `%LOCALAPPDATA%\LogoWizard` (`models\` and `feedback\`). Uninstalling or upgrading never touches them.

## 2. Set up the training kit

Install [Python 3.11](https://www.python.org/downloads/windows/) (tick "Add python.exe to PATH") and [Git](https://git-scm.com/download/win). For OCR filtering of the dataset, also install [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki) and add it to PATH. Then, in PowerShell:

```
git clone https://github.com/RodneyBaker/logowizard.git
cd logowizard
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -e .[train,app,dev]
python scripts\check_gpu.py
```

The PyTorch pin matters: CUDA 12.8 and newer builds dropped the GTX 1080 (Pascal). `check_gpu.py` should print the GPU name and a speed test.

## 3. Build the dataset and train

Each step is resumable: stop it any time and run the same command again to carry on. Times are estimates for a GTX 1080 and i5-9400F.

```
logo-wizard-fonts                                  # about 1 minute, local\fonts
logo-wizard-synth --count 250000                   # about 7 hours with OCR (overnight), local\dataset
logo-wizard-train-layout                           # a few hours, local\runs\layout
logo-wizard-train-stylizer --max-minutes 480       # one or more overnight sessions, local\runs\stylizer
logo-wizard-eval-stylizer --n 200                  # done when stylized_readable is above 0.95
logo-wizard-export-onnx                            # local\models\layout.onnx and stylizer.onnx
```

## 4. Give the models to the app

```
Copy-Item local\models\* $env:LOCALAPPDATA\LogoWizard\models\
```

Restart the app. The top right corner shows "layout model (ONNX)" when it found them.

## 5. Teach it your taste

Use thumbs up/down in the Create tab and pick winners in the Blind test tab; the tally there shows how often the model beats the rules. After 20 or more votes:

```
logo-wizard-finetune-layout --feedback $env:LOCALAPPDATA\LogoWizard\feedback\feedback.jsonl
Copy-Item local\models\layout.* $env:LOCALAPPDATA\LogoWizard\models\
```

The previous weights are kept as `local\runs\layout\best.before-feedback.pt` in case you prefer them.

## Building the installer yourself

```
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
```

This makes `dist\LogoWizard\`, the zip, and the setup.exe if [Inno Setup 6](https://jrsoftware.org/isinfo.php) is installed. Pushing a tag like `v0.1.0` makes GitHub Actions build the same thing and publish a release.
