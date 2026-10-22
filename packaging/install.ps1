# Install Logo Wizard from the unzipped release folder for the current user.
#
#   Right-click install.ps1 > Run with PowerShell
#
# Copies the app to %LOCALAPPDATA%\Programs\LogoWizard and adds a Start menu
# shortcut. Your fonts, models and feedback live in %LOCALAPPDATA%\LogoWizard
# and are not touched by installing or uninstalling.

$ErrorActionPreference = "Stop"
$source = $PSScriptRoot
$target = Join-Path $env:LOCALAPPDATA "Programs\LogoWizard"
$data = Join-Path $env:LOCALAPPDATA "LogoWizard"

if (Test-Path $target) { Remove-Item -Recurse -Force $target }
Copy-Item -Recurse $source $target
New-Item -ItemType Directory -Force -Path (Join-Path $data "models"), (Join-Path $data "feedback") | Out-Null

$menu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut((Join-Path $menu "Logo Wizard.lnk"))
$link.TargetPath = Join-Path $target "LogoWizard.exe"
$link.WorkingDirectory = $target
$link.IconLocation = (Join-Path $target "LogoWizard.exe") + ",0"
$link.Description = "Comic and novel title logos, made on this PC"
$link.Save()

Write-Host "Installed to $target"
Write-Host "Put exported models (layout.onnx, stylizer.onnx and their .json files) in $data\models"
Write-Host "Start it from the Start menu: Logo Wizard"
