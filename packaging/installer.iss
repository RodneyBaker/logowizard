; Inno Setup 6 script for Logo Wizard. Built by build_windows.ps1:
;   ISCC.exe /DAppVersion=0.1.0 packaging\installer.iss

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{8E0B6A51-3C55-4C0E-9A55-6E2F0B7C1A21}
AppName=Logo Wizard
AppVersion={#AppVersion}
AppPublisher=Rodney Baker
DefaultDirName={localappdata}\Programs\LogoWizard
DefaultGroupName=Logo Wizard
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=LogoWizard-{#AppVersion}-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=logowizard.ico
UninstallDisplayIcon={app}\LogoWizard.exe

[Files]
Source: "..\dist\LogoWizard\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Dirs]
Name: "{localappdata}\LogoWizard\models"
Name: "{localappdata}\LogoWizard\feedback"

[Icons]
Name: "{group}\Logo Wizard"; Filename: "{app}\LogoWizard.exe"
Name: "{group}\Logo Wizard models folder"; Filename: "{localappdata}\LogoWizard\models"
Name: "{group}\Uninstall Logo Wizard"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\LogoWizard.exe"; Description: "Start Logo Wizard"; Flags: nowait postinstall skipifsilent
