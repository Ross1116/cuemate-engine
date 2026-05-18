# CueMate Windows Installer

This folder contains the private-beta Windows packaging path.

## Build

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\windows\build-installer.ps1
```

The build script:

- builds `web/dist`
- compiles `go/cmd/apiserver` to `apiserver.exe`
- stages the runtime under `dist/windows-installer/stage/CueMate`
- invokes Inno Setup to write `dist/windows-installer/output/CueMateSetup.exe`

For a fast staging-only smoke check:

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\windows\build-installer.ps1 -SkipInstaller
```

## Installed Runtime

The installer writes immutable app files to:

```text
%LOCALAPPDATA%\Programs\CueMate
```

Mutable data and logs go to:

```text
%LOCALAPPDATA%\CueMate
```

`Bootstrap-CueMate.ps1` is run after installation and may be rerun safely. `Start-CueMate.ps1` is used by the Start Menu and Desktop shortcuts.

This first packaging version is Windows/private-beta oriented and unsigned by default. Add a signing step around `CueMateSetup.exe` when a certificate is available.
