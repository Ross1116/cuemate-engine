# CueMate

CueMate is a local, playlist-specific track selection helper for DJs. It imports your playlists, analyzes your tracks, and helps you pick the next song based on what you are trying to do in the mix: maintain, build, reset, jump, or contrast.

It is a passion-built product for the practical, messy, fun part of DJing: deciding what belongs next.

It runs on your own computer, stores data locally, and opens in your browser. Optional phone access is available through Tailscale, so you can scan a QR code and use the same CueMate session from your mobile device.

## Who It Is For

- DJs who want playlist-aware help choosing the next track.
- Bedroom, club, and radio workflows where playlist context matters.
- Curious music people who want a private, local tool built around real mixing habits.

## One-Click Windows Install

Download `CueMateSetup.exe` from the latest GitHub pre-release, then double-click it. If you are building from source, use the installer build command below.

The installer puts app files here:

```text
%LOCALAPPDATA%\Programs\CueMate
```

CueMate stores your database, logs, Python environment, and setup state here:

```text
%LOCALAPPDATA%\CueMate
```

After installation, open **CueMate** from the Start Menu or Desktop shortcut. The launcher starts the local scorer and API, then opens:

```text
http://127.0.0.1:8080
```

The first launch can take a while because CueMate may install Python, install Docker Desktop, create its private Python environment, build model-service images, and download model assets. If Docker asks you to sign in, update WSL, or restart Windows, finish that prompt and launch CueMate again. Setup resumes from its saved state.

## First Use

1. Open CueMate.
2. Switch to **Full** mode.
3. Import music from a local folder, audio files, or a DJ library export.
4. Select a playlist from the Library panel.
5. Use **Smart refresh playlist** to queue missing or stale analysis.
6. Run the analysis worker if work is queued and not moving.
7. Return to the main recommendation view and choose your current track.

CueMate never deletes your audio files. Removing a playlist from CueMate only removes CueMate's local playlist state.

## Phone Access

Phone access is optional. Local desktop use does not need it.

To use CueMate on your phone:

1. Install and sign in to Tailscale on your computer.
2. Install and sign in to Tailscale on your phone using the same tailnet.
3. Open CueMate on your computer.
4. Go to **Full -> Connect phone**.
5. Scan the QR code.

CueMate uses Tailscale Serve so the app stays private to your tailnet instead of being exposed publicly.

## Troubleshooting

**CueMate opens but analysis is unavailable**

Docker Desktop may not be ready. Open Docker Desktop, finish any login, WSL, update, or restart prompt, then launch CueMate again.

**The installer or launcher says Python is missing**

Install Python 3.12 or rerun CueMate setup with internet access so it can install Python through `winget`. If Python was just installed, close and reopen CueMate so Windows refreshes PATH.

**A playlist looks stale or recommendations are weak**

Open **Full -> Playlist Health** and click **Smart refresh playlist**. This queues only tracks that are missing, stale, or signature-mismatched.

**Where are logs?**

```text
%LOCALAPPDATA%\CueMate\logs
```

Useful files include `bootstrap.log`, `launcher.log`, `apiserver.err.log`, `apiserver.out.log`, `scorer.err.log`, and `scorer.out.log`.

**Phone QR is not available**

CueMate still works locally. For mobile access, make sure Tailscale is installed, signed in, and allowed to use Serve on your tailnet.

## Developer Setup

Prerequisites:

- Windows 10/11 recommended
- Go 1.25+
- Python 3.12+
- Node.js/npm
- Docker Desktop
- Tailscale for optional mobile testing

Install dependencies:

```powershell
npm --prefix web install
python -m pip install --user -e ".\python[dev]"
powershell -ExecutionPolicy Bypass -File .\scripts\dbmate.ps1 up
```

Start the scorer:

```powershell
$env:DATABASE_URL="sqlite:data/cuemate.db"
$env:CUEMATE_INFERENCE_CACHE_PATH="data/inference-cache.db"
python -m cuemate_analysis serve-scoring --host 127.0.0.1 --port 47834
```

Start the web app:

```powershell
npm --prefix web run dev
```

Start the Go API:

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

## Build The Installer

Build the full Windows installer:

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\windows\build-installer.ps1
```

Output:

```text
dist\windows-installer\output\CueMateSetup.exe
```

Fast staging-only check:

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\windows\build-installer.ps1 -SkipInstaller
```

More packaging details live in `packaging/windows/README.md`.

## Validation

```powershell
go test ./go/...
npm --prefix web run lint
npm --prefix web run build
powershell -ExecutionPolicy Bypass -File .\packaging\windows\Test-PackagingSmoke.ps1
```

## Technical Architecture

CueMate is local-first:

```text
React/Vite UI
  -> Go HTTP API
    -> SQLite database
    -> Python gRPC scoring service
      -> audio analysis + Docker-backed model services
```

Key engineering choices:

- **SQLite local data model** for imported tracks, playlist membership, analysis signatures, queued analysis jobs, recommendation events, and feedback history.
- **Go API** for the browser-facing HTTP surface, playlist browsing, recommendation hydration, health checks, remote pairing, and controlled local tool execution.
- **Python analysis service** for imports, audio feature extraction, model-backed BPM/key/semantic analysis, relative transition features, and scoring metadata.
- **gRPC/protobuf boundary** between Go and Python so recommendation requests, scoring explanations, and signature metadata stay typed across the process boundary.
- **Signature-aware analysis refresh** so CueMate can tell whether a track is missing, stale, or analyzed with an older config/model contract before queueing work.
- **Docker model services** for heavier BPM, key, and semantic analysis dependencies that are easier to isolate outside the main Python runtime.
- **React/Vite client** for the local browser UI, including playlist health, analysis controls, recommendation lanes, candidate details, and optional phone pairing.
- **Tailscale remote mode** for private phone access without opening public ports.
- **Windows installer bootstrap** for resumable setup on non-developer machines, including prerequisite checks, runtime environment setup, and local service startup.

## Project Layout

```text
config/              runtime configuration
db/                  SQLite migrations and schema snapshot
docker/              model-service Docker assets
go/                  local API, gRPC client, and Go tests
packaging/windows/   Windows installer, bootstrap, launcher, smoke checks
proto/               scoring protobuf contract
python/              analysis engine, scorer, CLI, and Python tests
scripts/             developer and model-service helpers
web/                 React/Vite app
```
