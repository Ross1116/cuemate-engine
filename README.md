# CueMate

CueMate is a local-first DJ recommendation app that helps you choose the next track while you are building a set.

It imports playlists, analyzes tracks, learns the shape of a playlist, and gives transition suggestions grouped by intent: maintain, build, reset, jump, or contrast. It runs on your own machine, keeps your library data local, and can optionally expose the app to your phone through Tailscale for QR-based mobile access.

## What It Does

- Imports local folders and DJ library exports from Rekordbox, Traktor, and Serato.
- Analyzes BPM, key, loudness, bass, energy, groove, danceability, and mood signals.
- Builds playlist-relative context so recommendations fit the specific set, not just the individual songs.
- Scores next-track candidates into practical DJ lanes: maintain, build, reset, jump, and contrast.
- Records played choices and uses feedback to tune playlist-level scoring weights.
- Runs as a local web app with a Windows installer, desktop/start-menu launcher, and optional phone pairing.

## Quick Start For Users

### Windows Installer

Build or download `CueMateSetup.exe`, then run it.

The installer sets up CueMate under:

```text
%LOCALAPPDATA%\Programs\CueMate
```

CueMate stores your local data under:

```text
%LOCALAPPDATA%\CueMate
```

After install, open CueMate from the Start Menu or Desktop shortcut. The launcher starts the local services and opens the app in your browser.

### Mobile Access

Mobile access is optional.

For phone access, install and sign in to Tailscale on both your computer and phone. CueMate will use Tailscale Serve to create a private HTTPS URL, then the app can generate a QR code from **Full Mode -> Connect phone**.

Without Tailscale, CueMate still works locally on your computer. Only phone access is unavailable.

## Developer Setup

This repository has three main runtimes:

- `web/`: React/Vite client
- `go/`: local HTTP API and scoring client
- `python/`: analysis engine and gRPC scoring service

### Prerequisites

- Windows 10/11 recommended
- Go 1.24+
- Python 3.12+
- Node.js/npm
- Docker Desktop for model-backed analysis
- Tailscale for optional mobile access

### Install Python Package

```powershell
python -m pip install --user -e ".\python[dev]"
```

### Initialize Database

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dbmate.ps1 up
```

### Start Development Services

Terminal 1, scoring service:

```powershell
$env:DATABASE_URL="sqlite:data/cuemate.db"
$env:CUEMATE_INFERENCE_CACHE_PATH="data/inference-cache.db"
python -m cuemate_analysis serve-scoring --host 127.0.0.1 --port 47834
```

Terminal 2, web client:

```powershell
npm --prefix web install
npm --prefix web run dev
```

Terminal 3, Go API:

```powershell
$env:DATABASE_URL="sqlite:data/cuemate.db"
$env:WEB_DIST_DIR="web/dist"
$env:SCORING_GRPC_ADDR="127.0.0.1:47834"
$env:GO_API_ADDR="127.0.0.1:8080"
go run ./go/cmd/apiserver
```

Open:

```text
http://127.0.0.1:8080
```

For mobile development with Tailscale:

```powershell
tailscale serve --bg http://127.0.0.1:8080
$env:CUEMATE_REMOTE_URL="https://your-machine.your-tailnet.ts.net"
```

Restart the Go API after setting `CUEMATE_REMOTE_URL`.

## Building The Windows Installer

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\windows\build-installer.ps1
```

Output:

```text
dist\windows-installer\output\CueMateSetup.exe
```

The build script:

- builds `web/dist`
- compiles `apiserver.exe`
- stages Python, Docker, config, database schema, scripts, and web assets
- runs packaging smoke checks
- invokes Inno Setup to create the installer

## Technical Overview

CueMate is intentionally local-first. The web app talks to a Go API on localhost. The Go API hydrates data from SQLite and calls a Python gRPC scorer. Python owns analysis, feature extraction, and recommendation semantics.

```text
React PWA
  -> Go HTTP API
    -> SQLite catalog/features/events
    -> Python gRPC scoring service
      -> DSP + Docker-backed model workers
```

### Data Flow

1. Import tracks from folders or DJ library exports.
2. Store tracks, playlists, and playlist order in SQLite.
3. Run fast analysis for immediate BPM/key availability.
4. Run full or staged analysis for richer absolute features.
5. Compute playlist-relative features and playlist stats.
6. Score candidates against current track, history, target lane, and tuned playlist weights.
7. Capture played outcomes and feed them back into playlist-specific tuning.

### Core Engineering Choices

- **Local-first SQLite** keeps user music metadata and feedback on the machine.
- **Go API boundary** gives the UI a small, stable HTTP surface with health/readiness handling.
- **Python scoring service** keeps analysis and recommendation math close to the audio feature pipeline.
- **gRPC/protobuf contract** makes the scorer boundary typed and testable.
- **Docker model workers** isolate heavier ML dependencies for BPM, key, and semantic analysis.
- **Tailscale remote mode** avoids public exposure while still enabling phone use.
- **Resumable Windows bootstrap** lets setup recover after Docker, Python, or Tailscale prerequisite prompts.

## Important Commands

```powershell
# Run tests
go test ./go/...
npm --prefix web run lint
npm --prefix web run build

# Build installer
powershell -ExecutionPolicy Bypass -File .\packaging\windows\build-installer.ps1

# Packaging smoke check
powershell -ExecutionPolicy Bypass -File .\packaging\windows\Test-PackagingSmoke.ps1 -RequireInstaller
```

## Project Layout

```text
config/              runtime configuration
db/                  SQLite migrations and schema snapshot
docker/              model-service Docker assets
go/                  local API, gRPC client, and Go tests
packaging/windows/   Windows installer, bootstrap, and launcher scripts
proto/               scoring protobuf contract
python/              analysis engine, scorer, CLI, and Python tests
scripts/             developer and model-service helpers
web/                 React/Vite app
```

## Current Status

CueMate is a Windows-first private beta. The installer builds and the local app flow works, including optional QR-based phone pairing over Tailscale.

The remaining production-readiness items are code signing, clean-VM acceptance testing, and broader packaging hardening for machines with unusual Python, Docker, or network policies.
