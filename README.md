# CueMate Engine

CueMate Engine is a monorepo with a hard split between the Python analysis plane and the Go decision plane. The repository now includes the first Milestone 1 slice from the decision engine plan: local playlist import, absolute feature analysis on PC, and SQLite persistence. Scoring, relative features, windowed features, and the Go API are still deferred.

## Repository layout

```text
/
|- python/   # Analysis Plane: audio ingest, DSP, artifact production, scoring service
|- go/       # Decision Plane: API, orchestration, sync, session surfaces
|- proto/    # Shared protobuf scoring contract
|- db/       # SQL migrations and checked-in schema snapshot
|- data/     # Local-only SQLite database, descriptors, and scratch artifacts
|- scripts/  # Utility scripts for validation, migrations, and local ops
```

## Current baseline

- Shared protobuf contract lives at `proto/djengine/scoring/v1/scoring.proto`
- `buf.yaml` defines `proto/` as the protobuf module root
- `go/` is its own Go module and is included from the root `go.work`
- `python/` now exposes a Milestone 1 CLI for local ingest and absolute analysis
- `config/default.json` is the checked-in runtime config baseline for analysis settings
- `dbmate` is the migration tool for forward-only SQL schema changes
- `db/` now includes Milestone 1 tables for `tracks`, `playlists`, `playlist_tracks`, `track_features_abs`, and `analysis_jobs`
- `compose.yaml` currently provides an operations-only migration service
- Generated artifacts and local env files are kept out of git by default

## Start here on a fresh machine

For a clean Windows bootstrap from a fresh clone, follow:

- [Bootstrap on Windows](d:/Personal%20Projects/CueMate/cuemate-engine/docs/bootstrap-windows.md)

## Environment setup

Before running migrations or services, copy `.env.example` to `.env` and adjust any values needed for your machine:

```powershell
Copy-Item .env.example .env
```

This is required because [compose.yaml](d:/Personal%20Projects/CueMate/cuemate-engine/compose.yaml) uses `env_file: - .env`. Make sure `.env` exists before running commands like:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\docker-compose.ps1 --profile ops run --rm migrate
```

## Useful commands

Validate the protobuf contract:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\compile-proto.ps1
```

Check local prerequisites:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-prereqs.ps1
```

Run dbmate from the repo root with the correct migration paths:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dbmate.ps1 status
```

Run the Docker Compose migration profile with an isolated local Docker config:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\docker-compose.ps1 --profile ops run --rm migrate
```

Install the Python analysis CLI in editable mode:

```powershell
python -m pip install --user -e ".\python[dev]"
```

Build the local TempoCNN Docker image used by the primary BPM backend:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-tempocnn-image.ps1
```

Build the local MusicalKeyCNN Docker image used by the primary key backend:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-musicalkeycnn-image.ps1
```

Optional: warm-start the persistent TempoCNN service container yourself. The CLI will auto-start it on demand too.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-tempocnn-service.ps1
```

Optional: warm-start the persistent MusicalKeyCNN service container yourself. The CLI will auto-start it on demand too.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-musicalkeycnn-service.ps1
```

## Milestone 1 CLI

Import a local playlist or crate:

```powershell
python -m cuemate_analysis import-playlist --name "My Playlist" .\path\to\audio
```

Analyze imported tracks with absolute features only:

```powershell
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode full
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode full --tempo-backend tempocnn --tempocnn-accelerator auto --force
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode full --key-backend musicalkeycnn --musicalkeycnn-device auto --force
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode full --key-backend musicalkeycnn --musicalkeycnn-policy full_track --force
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode full --tempo-backend baseline --force
```

Inspect playlist analysis state:

```powershell
python -m cuemate_analysis list-playlist --name "My Playlist"
```

Inspect one analyzed track:

```powershell
python -m cuemate_analysis show-track --track-id trk_example123
```

Analyze BPM only for one file or an imported playlist:

```powershell
python -m cuemate_analysis analyze-bpm "D:\path\to\track.wav"
python -m cuemate_analysis analyze-bpm "D:\path\to\track.wav" --backend baseline
python -m cuemate_analysis analyze-bpm-playlist --playlist "Fred again"
python -m cuemate_analysis analyze-bpm-playlist --playlist "Fred again" --limit 5 --output .\data\benchmarks\fred-again-bpm.csv
```

Compare the current fallback baseline against the primary TempoCNN backend on one file:

```powershell
python -m cuemate_analysis compare-bpm "D:\path\to\track.wav"
python -m cuemate_analysis compare-bpm "D:\path\to\track.wav" --json
python -m cuemate_analysis compare-bpm "D:\path\to\track.wav" --tempocnn-accelerator auto
```

Compare MusicalKeyCNN policies against tagged keys on one file, or benchmark those policies across a playlist:

```powershell
python -m cuemate_analysis compare-key "D:\path\to\track.wav"
python -m cuemate_analysis compare-key "D:\path\to\track.wav" --policies single_excerpt,full_track
python -m cuemate_analysis benchmark-key --playlist "Fred again"
python -m cuemate_analysis benchmark-key --playlist "Fred again" --limit 10 --output .\data\benchmarks\fred-again-key.csv
```

Benchmark tempo backends across an imported playlist:

```powershell
python -m cuemate_analysis benchmark-bpm --playlist "Fred again"
python -m cuemate_analysis benchmark-bpm --playlist "Fred again" --backends baseline,tempocnn --limit 5 --output .\data\benchmarks\fred-again.csv
```

TempoCNN is now the primary tempo path for comparison, benchmarking, and persisted tempo analysis. The default TempoCNN model path is `python/models/essentia/deepsquare-k16-3.pb`, and the default local Docker image name is `cuemate-tempocnn:local`.

Speed notes:

- TempoCNN is now used for BPM only; key extraction is no longer part of the TempoCNN Docker path
- MusicalKeyCNN is now the primary key backend for playlist analysis
- `compare-key` and `benchmark-key` are the intended commands for measuring the MusicalKeyCNN policy tradeoff against tagged keys
- `analyze-bpm` and `analyze-bpm-playlist` are the intended BPM-only commands
- `compare-bpm` is the debugging/benchmark command when you want to compare baseline vs TempoCNN directly
- single-track comparisons now use a warm TempoCNN service container when available, so repeated checks avoid model/container cold starts
- playlist analysis and `benchmark-bpm` batch TempoCNN tracks into the warm service path, so the model only loads once and stays hot
- MusicalKeyCNN runs through its own warm Docker worker, separate from the TempoCNN BPM worker
- the host-side analyzer still computes the remaining absolute features locally with `librosa`

Manual Docker debug for one track:

```powershell
docker run --rm --gpus all `
  -v "${PWD}:/workspace:ro" `
  -v "D:\Personal Projects\Music:/audio:ro" `
  cuemate-tempocnn:local `
  python /workspace/docker/tempocnn/run_tempocnn.py `
  "/audio/Fred again/Fred again.. - ..FEISTY.flac" `
  "/workspace/python/models/essentia/deepsquare-k16-3.pb"
```

GPU notes:

- TempoCNN now defaults to `--tempocnn-accelerator auto` and will try GPU instead of forcing CPU
- if TempoCNN is unavailable for a track, the analyzer falls back to the current librosa baseline automatically and records `baseline_fallback` as the BPM source
- the primary TempoCNN runtime now uses Docker rather than WSL
- if Docker cannot expose a usable GPU cleanly, TempoCNN will retry on CPU and the notes will say so
- the biggest speed gain comes from batched TempoCNN runs and removing unnecessary key extraction from the TempoCNN container path
- the librosa baseline remains CPU-bound
- MusicalKeyCNN now defaults to `full_track`; `balanced` is no longer part of the normal workflow because it did not show a useful speed advantage
- automatic chroma fallback is disabled for the primary MusicalKeyCNN path; if the model is unavailable, analysis now falls back to a tagged key only when one exists

## Intent for the next commits

- add Milestone 2 relative-feature and windowed-feature tables and refresh logic
- add the Python gRPC scoring service on top of the persisted analysis data
- add the Go API and orchestration packages under `go/cmd/` and `go/internal/`
- add service Dockerfiles only when real app entrypoints exist
