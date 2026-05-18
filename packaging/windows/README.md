# CueMate Windows Packaging

This folder builds the Windows private-beta installer: `CueMateSetup.exe`.

The installer is per-user and unsigned by default. It installs immutable app files under `%LOCALAPPDATA%\Programs\CueMate` and stores mutable data under `%LOCALAPPDATA%\CueMate`.

## Developer Prerequisites

- Windows 10/11
- PowerShell
- Node.js/npm
- Go 1.24+
- Inno Setup 6, unless building with `-SkipInstaller`
- Internet access for dependency restore and optional prerequisite installation

The build script can install Inno Setup through `winget` when available. It does not install Go or Node because those are developer-machine prerequisites.

## Build Commands

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\windows\build-installer.ps1
```

The installer is written to:

```text
dist\windows-installer\output\CueMateSetup.exe
```

Build a staged runtime without invoking Inno Setup:

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\windows\build-installer.ps1 -SkipInstaller
```

Use a custom version:

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\windows\build-installer.ps1 -Version 0.2.0
```

## What Gets Staged

The staged runtime lives at:

```text
dist\windows-installer\stage\CueMate
```

It includes:

- `apiserver.exe`
- `web/dist`
- `python/`
- `docker/`
- `config/`
- `db/`
- `scripts/`
- `.env.example`
- `README.md`
- `Bootstrap-CueMate.ps1`
- `Start-CueMate.ps1`
- `docs/Decision_Engine_Plan.md`
- `VERSION`

The docs sentinel is intentionally included because the Python runtime uses the repo-like layout for path discovery.

## Smoke Checks

After a staging build:

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\windows\Test-PackagingSmoke.ps1
```

After a full installer build:

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\windows\Test-PackagingSmoke.ps1 -RequireInstaller
```

The smoke test checks that required staged files exist and that the PowerShell scripts parse.

## Installed Runtime

App files:

```text
%LOCALAPPDATA%\Programs\CueMate
```

User data:

```text
%LOCALAPPDATA%\CueMate
```

Important files:

```text
%LOCALAPPDATA%\CueMate\data\cuemate.db
%LOCALAPPDATA%\CueMate\data\inference-cache.db
%LOCALAPPDATA%\CueMate\.venv
%LOCALAPPDATA%\CueMate\setup-state.json
%LOCALAPPDATA%\CueMate\remote.json
%LOCALAPPDATA%\CueMate\logs
```

Common logs:

- `bootstrap.log`
- `launcher.log`
- `apiserver.out.log`
- `apiserver.err.log`
- `scorer.out.log`
- `scorer.err.log`
- `tailscale-serve.log`

## Bootstrap And Launcher

`Bootstrap-CueMate.ps1` runs after install and is safe to rerun. It:

- installs Python 3.12, Docker Desktop, and optionally Tailscale via `winget`
- creates CueMate's private Python virtual environment
- initializes SQLite from `db/schema.sql`
- builds Docker model-service images
- downloads and prewarms model assets
- writes resumable setup state to `%LOCALAPPDATA%\CueMate\setup-state.json`

`Start-CueMate.ps1` is the Start Menu/Desktop shortcut target. It:

- resumes bootstrap if core setup is incomplete
- sets runtime environment variables
- starts the Python scorer if port `47834` is not already open
- starts the Go API if `/healthz` is not already ready
- configures Tailscale Serve when possible
- opens `http://127.0.0.1:8080`

## Clean VM Acceptance Checklist

On a clean Windows VM:

1. Run `CueMateSetup.exe`.
2. Keep the default desktop shortcut selected.
3. Choose whether to prepare optional Tailscale mobile access.
4. Let bootstrap run until complete or blocked.
5. If Docker prompts for sign-in, WSL setup, or restart, finish that prompt and launch CueMate again.
6. Confirm CueMate opens in the browser.
7. Confirm logs exist under `%LOCALAPPDATA%\CueMate\logs`.
8. Import a local folder or DJ library.
9. Run Smart refresh and the analysis worker.
10. Confirm recommendations render.
11. If testing mobile access, sign in to Tailscale on phone and PC, then scan the QR code from Full Mode.

## Signing

The private-beta installer is unsigned. When a certificate is available, add signing after Inno Setup produces:

```text
dist\windows-installer\output\CueMateSetup.exe
```

Suggested future hook:

```powershell
signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /a .\dist\windows-installer\output\CueMateSetup.exe
```

Do not block local private-beta builds on signing.
