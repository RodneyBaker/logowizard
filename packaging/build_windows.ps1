# Build the one-folder Windows release of Logo Wizard.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
#
# Produces dist\LogoWizard\ (the app), dist\LogoWizard-<version>-win64.zip and,
# if Inno Setup 6 is installed, dist\LogoWizard-<version>-setup.exe.
# The PyTorch training stack is not included: the app runs ONNX models
# (CPU, or the GPU through DirectML) exported with logo-wizard-export-onnx.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$version = (Select-String -Path pyproject.toml -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
Write-Host "Building Logo Wizard $version"

# Stop on a failed native command too, not only on PowerShell errors.
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }

# Python 3.11 through the py launcher, or set LW_PYTHON to a python.exe (CI does).
if (-not (Test-Path .build-venv)) {
    if ($env:LW_PYTHON) { & $env:LW_PYTHON -m venv .build-venv } else { py -3.11 -m venv .build-venv }
    Check "Creating the build venv"
}
$py = Join-Path $root ".build-venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -e ".[app]" pyinstaller; Check "pip install"
# DirectML build of ONNX Runtime: same API, and it can use any DirectX 12 GPU, the GTX 1080 included.
& $py -m pip uninstall -y onnxruntime
& $py -m pip install onnxruntime-directml; Check "pip install onnxruntime-directml"

if (-not (Get-ChildItem local\fonts -Recurse -Filter *.ttf -ErrorAction SilentlyContinue)) {
    & $py -m logo_wizard.fonts.fetch; Check "Font download"
}

& $py -m PyInstaller packaging\logowizard.spec --noconfirm --distpath dist --workpath build; Check "PyInstaller"
Copy-Item packaging\install.ps1, packaging\README-windows.txt dist\LogoWizard\

$zip = "dist\LogoWizard-$version-win64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path dist\LogoWizard -DestinationPath $zip
Write-Host "Wrote $zip"

$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (Test-Path $iscc) {
    & $iscc "/DAppVersion=$version" packaging\installer.iss; Check "Inno Setup"
    Write-Host "Wrote dist\LogoWizard-$version-setup.exe"
} else {
    Write-Host "Inno Setup 6 not found; skipped the setup.exe (the zip has install.ps1)."
}
