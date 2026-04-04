# CueMate Engine

CueMate Engine is a monorepo with a hard split between the Python analysis plane and the Go decision plane. The repository now includes the first Milestone 1 slice from the decision engine plan: local playlist import, absolute feature analysis on PC, and SQLite persistence.

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
- [Bootstrap on Windows](./docs/bootstrap-windows.md)

## Environment setup

Before running migrations or services, copy `.env.example` to `.env` and adjust any values needed for your machine:

```powershell
Copy-Item .env.example .env
```

This is required because [compose.yaml](./compose.yaml) uses `env_file: - .env`. Make sure `.env` exists before running commands like:

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

List playlists available from an exported DJ library:

```powershell
python -m cuemate_analysis list-dj-playlists --source rekordbox --library "D:\Exports\rekordbox.xml"
python -m cuemate_analysis list-dj-playlists --source traktor --library "D:\Exports\collection.nml"
python -m cuemate_analysis list-dj-playlists --source serato --library "D:\Music\_Serato_\Subcrates"
```

Import one playlist from Rekordbox, Traktor, or Serato:

```powershell
python -m cuemate_analysis import-dj-playlist --source rekordbox --library "D:\Exports\rekordbox.xml" --playlist "Main Room"
python -m cuemate_analysis import-dj-playlist --source traktor --library "D:\Exports\collection.nml" --playlist "Warmup" --name "Warmup"
python -m cuemate_analysis import-dj-playlist --source serato --library "D:\Music\_Serato_\Subcrates" --playlist "Club Set"
```

Analyze imported tracks with absolute features only:

```powershell
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode full
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode full --force
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
python -m cuemate_analysis analyze-bpm-playlist --playlist "Fred again"
python -m cuemate_analysis analyze-bpm-playlist --playlist "Fred again" --limit 5 --output .\data\benchmarks\fred-again-bpm.csv
```

Analyze just BPM + key with the production TempoCNN and MusicalKeyCNN workers:

```powershell
python -m cuemate_analysis analyze-bpm-key "D:\path\to\track.wav"
python -m cuemate_analysis analyze-bpm-key-playlist --playlist "Fred again"
python -m cuemate_analysis analyze-bpm-key-playlist --playlist "Fred again" --limit 5 --output .\data\benchmarks\fred-again-bpm-key.csv
```

Purge persisted TempoCNN and MusicalKeyCNN caches, and clear the warm service state so the next run is fully fresh:

```powershell
python -m cuemate_analysis purge-model-cache
python -m cuemate_analysis purge-model-cache --backend tempocnn
python -m cuemate_analysis purge-model-cache --playlist "Fred again"
```

Speed notes:

- TempoCNN is now used for BPM only
- MusicalKeyCNN is the primary key backend for playlist analysis
- exported Rekordbox/Traktor BPM and key metadata are now imported into the local track catalog and used during analysis resolution
- Serato crate import currently contributes playlist membership and local file paths only; it does not provide BPM/key metadata yet
- `analyze-bpm` and `analyze-bpm-playlist` are the intended BPM-only commands
- `analyze-bpm-key` and `analyze-bpm-key-playlist` are the intended fast paths when you only want BPM + key
- TempoCNN and MusicalKeyCNN both use warm Docker workers so repeated analysis avoids model cold starts
- both warm workers now cache results for unchanged files, so rerunning the same BPM/key playlist pass is dramatically faster
- those persistent caches live in `data/inference-cache.db` and can be purged on demand with `purge-model-cache`
- playlist analysis batches TempoCNN and MusicalKeyCNN work so the models stay loaded
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

- if TempoCNN is unavailable for a track, the analyzer falls back to the current librosa baseline automatically and records `baseline_fallback` as the BPM source
- the primary TempoCNN runtime now uses Docker rather than WSL
- if Docker cannot expose a usable GPU cleanly, TempoCNN will retry on CPU and the notes will say so
- the biggest speed gains come from batched TempoCNN runs and keeping both model workers warm
- the librosa baseline remains CPU-bound
- MusicalKeyCNN now uses `full_track` as the settled production path
- automatic chroma fallback is disabled; if MusicalKeyCNN is unavailable, analysis falls back to a tagged key only when one exists

## Intent for the next commits

- add Milestone 2 relative-feature and windowed-feature tables and refresh logic
- add the Python gRPC scoring service on top of the persisted analysis data
- add the Go API and orchestration packages under `go/cmd/` and `go/internal/`
- add service Dockerfiles only when real app entrypoints exist
