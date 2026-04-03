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

The default TempoCNN graph expected by the CLI lives at `python/models/essentia/deepsquare-k16-3.pb`. Override it with `CUEMATE_TEMPOCNN_MODEL` or `--tempocnn-model` if you want to compare a different `.pb` model.

## CLI entrypoints

Use the module form so the commands work even when the user-site Scripts directory is not on `PATH`:

```powershell
python -m cuemate_analysis import-playlist --name "My Playlist" .\path\to\audio
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode full
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode full --tempo-backend tempocnn --tempocnn-accelerator auto --force
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode full --tempo-backend baseline --force
python -m cuemate_analysis list-playlist --name "My Playlist"
python -m cuemate_analysis show-track --track-id trk_example123
```

## Experimental BPM comparison

You can compare the current fallback baseline against the primary TempoCNN backend on a single file:

```powershell
python -m cuemate_analysis compare-bpm "D:\path\to\track.wav"
python -m cuemate_analysis compare-bpm "D:\path\to\track.wav" --json
python -m cuemate_analysis benchmark-bpm --playlist "Fred again"
python -m cuemate_analysis compare-bpm "D:\path\to\track.wav" --tempocnn-accelerator auto
python -m cuemate_analysis benchmark-bpm --playlist "Fred again" --backends baseline,tempocnn --limit 5 --output .\data\benchmarks\fred-again.csv
python -m cuemate_analysis compare-bpm "D:\path\to\track.wav" --tempocnn-model "D:\path\to\deeptemp-k16-3.pb"
```

Important notes:

- `tempocnn` is now the primary BPM backend used by `analyze-playlist`
- if TempoCNN is unavailable for a track, analysis falls back to the current librosa baseline automatically and records `baseline_fallback` as the source
- `benchmark-bpm` now defaults to `baseline,tempocnn`
- TempoCNN now runs through Docker, and `--tempocnn-accelerator auto` will try GPU before falling back to CPU
- TempoCNN now handles BPM only; key extraction is no longer part of the TempoCNN container path
- playlist analysis and benchmarking batch TempoCNN tracks into one container run so the model only loads once per batch
- the default TempoCNN model shipped in the repo is `deepsquare-k16-3.pb`
- the default local Docker image name is `cuemate-tempocnn:local`, and you can override it with `CUEMATE_TEMPOCNN_IMAGE`
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
