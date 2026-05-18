#define MyAppName "CueMate"
#ifndef MyAppVersion
#define MyAppVersion "0.1.0"
#endif
#ifndef SourceDir
#define SourceDir "..\..\dist\windows-installer\stage\CueMate"
#endif
#ifndef OutputDir
#define OutputDir "..\..\dist\windows-installer\output"
#endif

[Setup]
AppId={{D6BE2729-845A-47AF-B892-9D47EE217136}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=CueMate
DefaultDirName={localappdata}\Programs\CueMate
DefaultGroupName=CueMate
DisableProgramGroupPage=yes
OutputDir={#OutputDir}
OutputBaseFilename=CueMateSetup
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#MyAppName}
WizardStyle=modern

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\CueMate"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\Start-CueMate.ps1"""; WorkingDir: "{app}"
Name: "{autodesktop}\CueMate"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\Start-CueMate.ps1"""; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: checkedonce
Name: "mobileaccess"; Description: "Prepare optional phone access with Tailscale"; GroupDescription: "Mobile access:"; Flags: checkedonce

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\Bootstrap-CueMate.ps1"" -InstallDir ""{app}"""; WorkingDir: "{app}"; StatusMsg: "Preparing CueMate runtime, models, and local services..."; Check: WizardIsTaskSelected('mobileaccess')
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\Bootstrap-CueMate.ps1"" -InstallDir ""{app}"" -SkipTailscaleInstall"; WorkingDir: "{app}"; StatusMsg: "Preparing CueMate runtime, models, and local services..."; Check: not WizardIsTaskSelected('mobileaccess')
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\Start-CueMate.ps1"" -InstallDir ""{app}"""; WorkingDir: "{app}"; Description: "Launch CueMate"; Flags: postinstall nowait skipifsilent
