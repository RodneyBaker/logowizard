Logo Wizard for Windows
=======================

Start: double-click LogoWizard.exe (or the Start menu shortcut after running
install.ps1). A console window opens and your browser shows the app at
http://127.0.0.1:8765. Close the console window to stop it.

Without trained models the app uses its hand-written title rules. To use
your own trained models, copy layout.onnx, layout.json, stylizer.onnx and
stylizer.json (made by logo-wizard-export-onnx) into

    %LOCALAPPDATA%\LogoWizard\models

and restart the app. Thumbs up/down and blind-test picks are saved in
%LOCALAPPDATA%\LogoWizard\feedback for fine-tuning.

Fonts: 264 openly licensed families (SIL OFL 1.1, Apache 2.0, Ubuntu Font
License) are included in the fonts folder with their license files; see
DATA_LICENSES.md.
