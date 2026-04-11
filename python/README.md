# Analysis Plane

The Python analysis plane is responsible for offline ingest, feature extraction, artifact generation, and the authoritative scoring service runtime.

## Intended layout

```text
python/
|- src/        # Hand-written analysis and service packages
|  |- djengine/ # Generated protobuf/grpc Python artifacts
|- tests/      # Analysis-plane tests
|- models/     # Local model weights and checkpoints (git-ignored)
|- pyproject.toml
```

## Current scope

The current implementation covers the shipped Python analysis and scoring core:

- import local files and directories as playlists/crates
- import playlists from exported Rekordbox XML, Traktor NML, and Serato crate files
- read embedded tags with `mutagen`
- decode audio and extract absolute features on PC
- persist imported tracks, playlists, analysis jobs, and `track_features_abs` into SQLite
- inspect imported and analyzed tracks from a local CLI
- score and recommend transitions from persisted absolute + relative context
- expose the scorer through a local gRPC service after protobuf compilation

This package now exposes the local recommendation/scoring CLI, but a few signals are still intentionally incomplete:

- `transition_support`
- `vocal_transition`
- `rhythmic_continuity`

Those components are explicit stubs and are excluded from weighted scoring until they are implemented. Windowed intro/outro features are still deferred, and `vocals_abs` / `vocals_rel` are not populated by the current analysis pipeline yet.

The shipped local feedback loop is also part of the current scope:

- recommendation event item capture for returned candidates
- playlist-level feedback summaries
- per-playlist tuned weights that override heuristic playlist adaptation
- a local worker that applies tuned weights from queued `feedback_tuning_jobs`

## Install

From the repository root:

```powershell
python -m pip install --user -e ".\python[dev]"
```

Build the shared TensorFlow/Essentia Docker image used by the primary BPM backend. The TempoCNN build helper now delegates to this shared image:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-tempocnn-image.ps1
```

Build the local MusicalKeyCNN Docker image used by the primary key backend:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-musicalkeycnn-image.ps1
```

Build the same shared TensorFlow/Essentia Docker image through the Essentia-specific helper:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-essentia-semantics-image.ps1
```

Optional: warm-start the shared TensorFlow/Essentia service through the TempoCNN alias helper. The CLI will auto-start it on demand too.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-tempocnn-service.ps1
```

Optional: warm-start the persistent MusicalKeyCNN service container yourself. The CLI will auto-start it on demand too.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-musicalkeycnn-service.ps1
```

Optional: warm-start the same shared TensorFlow/Essentia service through the Essentia-specific helper. The CLI will auto-start it on demand too.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-essentia-semantics-service.ps1
```

The default TempoCNN graph expected by the CLI lives at `python/models/essentia/deepsquare-k16-3.pb`. Override it with `CUEMATE_TEMPOCNN_MODEL` or `--tempocnn-model` if you want to compare a different `.pb` model.
The default MusicalKeyCNN checkpoint expected by the CLI lives at `python/models/musicalkeycnn/keynet.pt`. Override it with `CUEMATE_MUSICALKEYCNN_MODEL` or `--musicalkeycnn-model` if you want to compare a different checkpoint.

Current runtime topology:

- one shared TensorFlow/Essentia service for TempoCNN BPM and Essentia semantic inference
- one separate PyTorch service for MusicalKeyCNN key inference

The `tempocnn` image/start helpers remain for compatibility and discoverability, but they no longer represent a third independent warm service container.

## CLI entrypoints

Use the module form so the commands work even when the user-site Scripts directory is not on `PATH`:

```powershell
python -m cuemate_analysis import-playlist --name "My Playlist" .\path\to\audio
python -m cuemate_analysis list-dj-playlists --source rekordbox --library "D:\Exports\rekordbox.xml"
python -m cuemate_analysis import-dj-playlist --source rekordbox --library "D:\Exports\rekordbox.xml" --playlist "Main Room"
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode full
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode full --force
python -m cuemate_analysis analyze-bpm "D:\path\to\track.wav"
python -m cuemate_analysis analyze-bpm-playlist --playlist "My Playlist"
python -m cuemate_analysis analyze-bpm-key "D:\path\to\track.wav"
python -m cuemate_analysis analyze-bpm-key-playlist --playlist "My Playlist"
python -m cuemate_analysis analyze-relative-playlist --playlist "My Playlist"
python -m cuemate_analysis analyze-energy-playlist --playlist "My Playlist"
python -m cuemate_analysis download-essentia-semantic-models
python -m cuemate_analysis analyze-essentia-playlist --playlist "My Playlist"
python -m cuemate_analysis recommend-next --playlist "My Playlist" --current-track trk_example123 --target maintain
python -m cuemate_analysis score-pair --playlist "My Playlist" --current trk_example123 --candidate trk_example456
python -m cuemate_analysis inspect-scoring-weights --playlist "My Playlist"
python -m cuemate_analysis feedback-summary --playlist "My Playlist"
python -m cuemate_analysis feedback-tune --playlist "My Playlist" --preview-only
python -m cuemate_analysis run-feedback-worker --limit 10
python -m cuemate_analysis inspect-scoring-metadata
python -m cuemate_analysis serve-scoring --host 127.0.0.1 --port 47834
python -m cuemate_analysis purge-model-cache
python -m cuemate_analysis list-playlist --name "My Playlist"
python -m cuemate_analysis show-track --track-id trk_example123
```

Important notes:

- `tempocnn` is now the primary BPM backend used by `analyze-playlist`
- `musicalkeycnn` is now the primary key backend used by `analyze-playlist`
- Rekordbox and Traktor imports can contribute BPM/key metadata that is stored on `tracks` and folded into the final analysis resolution
- Serato crate imports currently provide playlist membership and file-path discovery only; BPM/key metadata is not available from the current parser
- `analyze-bpm` and `analyze-bpm-playlist` are the intended BPM-only commands
- `analyze-bpm-key` and `analyze-bpm-key-playlist` are the intended fast paths when you only want BPM + key
- `analyze-relative-playlist` is the read-only inspection surface for playlist-relative context and playlist stats previews
- `analyze-energy-playlist` is the read-only diagnostics surface for comparing absolute-energy formulas against the current production analyzer
- `download-essentia-semantic-models` and `analyze-essentia-playlist` are the model-acquisition and read-only inspection surfaces for Essentia semantic absolute features
- `recommend-next` organizes scored suggestions into `maintain`, `build`, `reset`, `jump`, and `contrast` lanes
- `score-pair` is the main diagnostics surface for auditing one current->candidate transition
- `inspect-scoring-weights` exposes the static/base/tuned/effective layers plus the active weight source
- `feedback-summary` reports playlist-level recommendation outcomes and current feedback-tuning state
- `feedback-tune` previews or applies per-playlist tuned weights from recorded outcomes
- `run-feedback-worker` claims queued `feedback_tuning_jobs` and applies tuned weights when thresholds are met
- `inspect-scoring-metadata` exposes the active scoring contract and runtime metadata
- `serve-scoring` runs the local gRPC scorer after the protobuf contract has been compiled
- treat vocal-related weights and diagnostics as placeholders for now: `vocal_transition` is stubbed, and `vocals_abs` / `vocals_rel` are currently unavailable in analysis output
- if TempoCNN is unavailable for a track, analysis falls back to the current librosa baseline automatically and records `baseline_fallback` as the source
- TempoCNN now runs through the shared TensorFlow/Essentia Docker service and will try GPU before falling back to CPU
- TempoCNN now handles BPM only; key extraction is no longer part of the TempoCNN container path
- MusicalKeyCNN now runs through its own warm Docker service and is independent from the TempoCNN worker
- MusicalKeyCNN now defaults to `full_track`
- if MusicalKeyCNN is unavailable for a track, analysis now falls back to a tagged key only when one exists
- repeated requests now go through the shared TensorFlow/Essentia service and the separate MusicalKeyCNN service when possible
- the Essentia semantic lane uses the same shared TensorFlow/Essentia warm Docker service as TempoCNN; there is a single shared container for those two paths, not two separate warm services
- repeated requests for unchanged files are cached inside those warm services, so reruns are much faster than the first pass
- those persistent model caches are also stored in `data/inference-cache.db`, and `purge-model-cache` clears both the persistent rows and the warm service state
- playlist analysis batches TempoCNN tracks through the warm service so the model stays loaded
- the default TempoCNN model shipped in the repo is `deepsquare-k16-3.pb`
- the default MusicalKeyCNN checkpoint shipped in the local repo cache is `keynet.pt`
- the default local Docker image name is `cuemate-tempocnn:local`, and you can override it with `CUEMATE_TEMPOCNN_IMAGE`
- the default local Docker image name for key detection is `cuemate-musicalkeycnn:local`, and you can override it with `CUEMATE_MUSICALKEYCNN_IMAGE`
- if Docker cannot expose a usable GPU cleanly, the TempoCNN notes will say it retried on CPU
- `librosa`-based baseline analysis remains CPU-bound

Feedback-tuning weight precedence:

- `feedback_tuned_weights`
- `adapted_weights`
- static scoring weights

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

## Protobuf / gRPC workflow

Generate the scoring contract artifacts before using the gRPC service:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\compile-proto.ps1
```

This runs `buf lint`, writes `data/scoring.pb`, and generates Python gRPC stubs into `python/src/djengine/`.
