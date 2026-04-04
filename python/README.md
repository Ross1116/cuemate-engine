# Analysis Plane

The Python analysis plane is responsible for offline ingest, feature extraction, artifact generation, and the authoritative scoring service runtime.

## Intended layout

```text
python/
|- src/        # Hand-written analysis and service packages
|- tests/      # Analysis-plane tests
|- generated/  # Generated protobuf/grpc Python artifacts
|- models/     # Local model weights and checkpoints (git-ignored)
|- pyproject.toml
```

## Current Milestone 1 scope

The current implementation covers the first milestone from the decision engine plan:

- import local files and directories as playlists/crates
- read embedded tags with `mutagen`
- decode audio and extract absolute features on PC
- persist imported tracks, playlists, analysis jobs, and `track_features_abs` into SQLite
- inspect imported and analyzed tracks from a local CLI

This package does not yet expose scoring, relative features, windowed features, or the gRPC scoring service.

## Install

From the repository root:

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

The default TempoCNN graph expected by the CLI lives at `python/models/essentia/deepsquare-k16-3.pb`. Override it with `CUEMATE_TEMPOCNN_MODEL` or `--tempocnn-model` if you want to compare a different `.pb` model.
The default MusicalKeyCNN checkpoint expected by the CLI lives at `python/models/musicalkeycnn/keynet.pt`. Override it with `CUEMATE_MUSICALKEYCNN_MODEL` or `--musicalkeycnn-model` if you want to compare a different checkpoint.

## CLI entrypoints

Use the module form so the commands work even when the user-site Scripts directory is not on `PATH`:

```powershell
python -m cuemate_analysis import-playlist --name "My Playlist" .\path\to\audio
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode full
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode full --force
python -m cuemate_analysis analyze-bpm "D:\path\to\track.wav"
python -m cuemate_analysis analyze-bpm-playlist --playlist "My Playlist"
python -m cuemate_analysis analyze-bpm-key "D:\path\to\track.wav"
python -m cuemate_analysis analyze-bpm-key-playlist --playlist "My Playlist"
python -m cuemate_analysis purge-model-cache
python -m cuemate_analysis list-playlist --name "My Playlist"
python -m cuemate_analysis show-track --track-id trk_example123
```

Important notes:

- `tempocnn` is now the primary BPM backend used by `analyze-playlist`
- `musicalkeycnn` is now the primary key backend used by `analyze-playlist`
- `analyze-bpm` and `analyze-bpm-playlist` are the intended BPM-only commands
- `analyze-bpm-key` and `analyze-bpm-key-playlist` are the intended fast paths when you only want BPM + key
- if TempoCNN is unavailable for a track, analysis falls back to the current librosa baseline automatically and records `baseline_fallback` as the source
- TempoCNN now runs through Docker and will try GPU before falling back to CPU
- TempoCNN now handles BPM only; key extraction is no longer part of the TempoCNN container path
- MusicalKeyCNN now runs through its own warm Docker service and is independent from the TempoCNN worker
- MusicalKeyCNN now defaults to `full_track`
- if MusicalKeyCNN is unavailable for a track, analysis now falls back to a tagged key only when one exists
- repeated requests now go through warm TempoCNN and MusicalKeyCNN service containers when possible
- repeated requests for unchanged files are cached inside those warm services, so reruns are much faster than the first pass
- those persistent model caches are also stored in `data/inference-cache.db`, and `purge-model-cache` clears both the persistent rows and the warm service state
- playlist analysis batches TempoCNN tracks through the warm service so the model stays loaded
- the default TempoCNN model shipped in the repo is `deepsquare-k16-3.pb`
- the default MusicalKeyCNN checkpoint shipped in the local repo cache is `keynet.pt`
- the default local Docker image name is `cuemate-tempocnn:local`, and you can override it with `CUEMATE_TEMPOCNN_IMAGE`
- the default local Docker image name for key detection is `cuemate-musicalkeycnn:local`, and you can override it with `CUEMATE_MUSICALKEYCNN_IMAGE`
- if Docker cannot expose a usable GPU cleanly, the TempoCNN notes will say it retried on CPU
- `librosa`-based baseline analysis remains CPU-bound

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
