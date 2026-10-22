"""Where Logo Wizard reads its bundled data and keeps its local files.

Two roots:
  DATA_ROOT  read-only files that ship with the code: font manifest,
             palettes, phrase lists, the web page. The repo in a
             development install, the unpacked bundle in the Windows build.
  LOCAL_ROOT everything made or downloaded on this machine: fonts,
             datasets, training runs, exported models, feedback. The
             repo's local/ folder in a development install, and
             %LOCALAPPDATA%\\LogoWizard in the Windows build.

Set LOGO_WIZARD_HOME to put LOCAL_ROOT anywhere else, for example to point
the packaged app at the local/ folder of your training checkout.
"""

import os
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
_REPO = Path(__file__).resolve().parents[2]

if FROZEN:
    DATA_ROOT = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
else:
    DATA_ROOT = _REPO


def _local_root() -> Path:
    if os.environ.get("LOGO_WIZARD_HOME"):
        return Path(os.environ["LOGO_WIZARD_HOME"]).expanduser()
    if FROZEN:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "LogoWizard"
    return _REPO / "local"


LOCAL_ROOT = _local_root()

FONT_MANIFEST = DATA_ROOT / "data" / "fonts" / "manifest.csv"
PALETTES = DATA_ROOT / "data" / "palettes" / "palettes.yaml"
CORPORA = DATA_ROOT / "data" / "corpora"
PHRASES = CORPORA / "phrases.tsv"

# The Windows build carries its own copy of the fonts; a dev install downloads them.
BUNDLED_FONTS = DATA_ROOT / "fonts"
FONT_DIR = BUNDLED_FONTS if FROZEN and BUNDLED_FONTS.exists() else LOCAL_ROOT / "fonts"

RAW_DIR = LOCAL_ROOT / "raw"
DATASET_DIR = LOCAL_ROOT / "dataset"
RUNS_DIR = LOCAL_ROOT / "runs"
MODELS_DIR = LOCAL_ROOT / "models"
FEEDBACK_DIR = LOCAL_ROOT / "feedback"
OUT_DIR = LOCAL_ROOT / "out"
