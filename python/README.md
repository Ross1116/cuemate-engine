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

## CLI entrypoints

Use the module form so the commands work even when the user-site Scripts directory is not on `PATH`:

```powershell
python -m cuemate_analysis import-playlist --name "My Playlist" .\path\to\audio
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode full
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode full --tempo-backend beatnet --force
python -m cuemate_analysis list-playlist --name "My Playlist"
python -m cuemate_analysis show-track --track-id trk_example123
```

## Experimental BPM comparison

Optional local installs for the BeatNet comparison path:

```powershell
python -m pip install --user BeatNet madmom-prebuilt PyAudio
```

You can compare the current Milestone 1 BPM detector against an experimental BeatNet-backed estimate on a single file:

```powershell
python -m cuemate_analysis compare-bpm "D:\path\to\track.wav"
python -m cuemate_analysis compare-bpm "D:\path\to\track.wav" --json
python -m cuemate_analysis benchmark-bpm --playlist "Fred again"
python -m cuemate_analysis benchmark-bpm --playlist "Fred again" --backends baseline,essentia_wsl,beatnet --limit 5 --output .\data\benchmarks\fred-again.csv
```

Important notes:

- BeatNet is being used as an experimental tempo backend only
- `analyze-playlist --tempo-backend beatnet` now uses BeatNet for BPM estimation in the persisted analysis path
- it does not help with key detection
- the current wrapper uses a local compatibility shim for `madmom-prebuilt` on Windows
- Essentia is currently reached through WSL for comparison and benchmarking
- `benchmark-bpm` defaults to `baseline,essentia_wsl` because BeatNet is much slower
