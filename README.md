# CueMate Engine

CueMate Engine is the local analysis and recommendation foundation for CueMate.

The repository is centered on a **Python-first analysis engine** with:

- DJ-library playlist import
- staged BPM/key analysis for fast feedback
- background enrichment for absolute features
- persisted relative playlist context
- live recommendation/scoring lanes and pair diagnostics
- Docker-backed model workers for BPM, key, and semantic mood/intensity analysis, with TempoCNN and Essentia semantics sharing one TensorFlow/Essentia service

The Go and protobuf layers now define the shipped local API and scorer boundary, while the scoring semantics still live in the Python analysis plane.

## Current State

Implemented:

- Milestone 1 absolute analysis
- Milestone 2 Phase 1 relative-context logic
- Milestone 2 Phase 2 persisted relative context + refresh/orchestration
- Milestone 3 recommendation/scoring core:
  - target-aware candidate scoring
  - lane organization (`maintain`, `build`, `reset`, `jump`, `contrast`)
  - recommendation confidence
  - pair scoring diagnostics
  - scoring metadata and compatibility checks
- Milestone 4 Python scoring service slice:
  - protobuf scoring contract revised to match the live scorer
  - local gRPC scoring service runtime
  - Python proto/codegen workflow
- Milestone 4 Go bootstrap slice:
  - Go protobuf/gRPC client bootstrap
  - `scoringctl` smoke CLI for metadata and fixture-driven score calls
  - Go proto/codegen workflow
- Milestone 4 Go live API slice:
  - Go HTTP API server for `/recommendations`, `/scoring/metadata`, `/healthz`, and `/readyz`
  - SQLite hydration of live scoring inputs
  - scorer readiness/circuit-breaker handling in the decision plane
- Milestone 5 feedback loop slice:
  - immutable recommendation event item capture for returned candidates
  - `/events/played` outcome capture plus queued `feedback_tuning_jobs`
  - playlist-level feedback summaries and per-playlist tuned weights
  - local feedback worker for automatic per-playlist retuning
- staged analysis pipeline:
  - `fast_pass`
  - `staged` (default)
  - `full`

Deferred for now:

- windowed intro/outro analysis

Broader follow-on work:

- multi-device/authenticated sync on top of the shipped local-first recommendation and snapshot/outbox surfaces

## Architecture

The current pipeline is split into 5 main layers:

1. Import/catalog
- imports local playlists or DJ-library playlists from Rekordbox XML, Traktor NML, and Serato crates
- stores tracks, playlists, and playlist membership in SQLite

2. Fast analysis
- computes BPM + key quickly for immediate UI/CLI feedback
- persists results into `track_features_fast`

3. Enrichment analysis
- computes canonical absolute features into `track_features_abs`
- uses local DSP + model-backed semantics
- runs inline for `full` or queued in `analysis_jobs` for `staged`

4. Relative context
- computes playlist-relative features and playlist stats
- persists canonical rows into:
  - `track_features_rel`
  - `playlist_stats`

5. Recommendation/scoring
- ranks next-track candidates from precomputed absolute + relative context
- organizes results into move lanes
- exposes the local HTTP API for live recommendations, event capture, feedback summaries, and sync surfaces
- keeps CLI inspection surfaces for recommendations, score breakdowns, weights, scoring metadata, and local operator workflows

## Repository Layout

```text
/
|- config/
|  |- default.json                         # Runtime defaults
|  |- essentia_semantic_calibration.json  # Semantic calibration config
|
|- data/                                  # Local-only runtime data
|  |- cuemate.db                          # Main SQLite DB
|  |- inference-cache.db                  # Persistent model inference cache
|  |- benchmarks/                         # Local benchmark outputs
|
|- db/
|  |- migrations/                         # SQL migrations
|  |- schema.sql                          # Checked-in schema snapshot
|  |- README.md
|
|- docker/
|  |- essentia_semantics/                 # Shared TF/Essentia service
|  |- musicalkeycnn/                      # PyTorch key service image
|  |- tempocnn/                           # TempoCNN image + service files
|
|- docs/
|  |- bootstrap-windows.md
|  |- Decision_Engine_Plan.md
|  |- stack-decisions.md
|
|- go/
|  |- cmd/                               # Go API + smoke/debug entrypoints
|  |- internal/                          # Go client/bootstrap packages
|  |- gen/                               # Generated Go artifacts
|  |- README.md
|
|- proto/
|  |- djengine/scoring/v1/scoring.proto  # Shared scoring contract
|  |- README.md
|
|- python/
|  |- models/                            # Local model artifacts
|  |  |- essentia/
|  |  |- essentia_semantics/
|  |  |- musicalkeycnn/
|  |- src/cuemate_analysis/
|  |  |- cli.py                          # Main CLI entrypoint
|  |  |- analysis.py                     # Absolute analysis + resolution logic
|  |  |- relative_context.py             # Relative layer + refresh logic
|  |  |- database.py                     # SQLite access layer
|  |  |- dj_import.py                    # Rekordbox/Traktor/Serato importers
|  |  |- tempo_backend.py                # TempoCNN backend client/runtime helpers
|  |  |- key_backend.py                  # MusicalKeyCNN backend client/runtime helpers
|  |  |- essentia_semantic_backend.py    # Essentia semantic backend client/runtime helpers
|  |  |- models.py                       # Dataclasses / result shapes
|  |  |- config.py                       # Runtime config loading + signatures
|  |  |- dsp_benchmark.py                # DSP benchmark harness
|  |- src/djengine/                      # Generated Python protobuf/gRPC artifacts
|  |- tests/                             # Python test suite
|  |- pyproject.toml
|  |- README.md
|
|- scripts/
|  |- build-*.ps1                        # Docker image build helpers
|  |- start-*.ps1                        # Service start helpers
|  |- dbmate.ps1
|  |- docker-compose.ps1
|  |- check-prereqs.ps1
|  |- compile-proto.ps1
|  |- README.md
|
|- compose.yaml
|- buf.yaml
|- go.work
|- .env.example
|- README.md
```

## Canonical Data Model

Current important tables:

- `tracks`
  - imported/local track catalog
- `playlists`
  - playlist catalog
- `playlist_tracks`
  - playlist membership and order
- `track_features_fast`
  - fast-stage BPM/key results for immediate feedback
- `track_features_abs`
  - canonical absolute analysis rows
  - includes `analysis_signature`, `config_signature`, and `scoring_contract_id_at_analysis`
- `track_features_rel`
  - canonical persisted relative playlist rows
- `playlist_stats`
  - persisted playlist-level relative stats/adaptation info
- `analysis_jobs`
  - local staged-analysis/enrichment queue

## Feature Contract

### Fast layer

Persisted in `track_features_fast`:

- resolved BPM
- resolved key
- confidence + provenance/source fields

### Canonical absolute layer

Persisted in `track_features_abs`.

Current split:

- DSP-native canonical primitives:
  - `loudness_lufs`
  - `loudness_norm`
  - `bass_abs`
  - `time_signature`
  - `time_signature_confidence`

- DSP-native support fields:
  - `energy_heuristic_abs`
  - `energy_sustained`
  - `energy_peak`
  - optional support descriptors:
    - `drums_abs`
    - `harmonic_abs`
    - `groove_abs`

- model-backed semantic fields:
  - `danceability_abs`
  - `arousal_abs`
  - `valence_abs`
  - `mood_aggressive_abs`
  - `mood_party_abs`
  - `mood_relaxed_abs`

- canonical fused intensity:
  - `energy_abs`

- explicit Essentia lane outputs:
  - `energy_essentia_fused`
  - `energy_essentia_bucket`

- scoring compatibility metadata:
  - `analysis_signature`
  - `config_signature`
  - `scoring_contract_id_at_analysis`

### Canonical relative layer

Persisted in:

- `track_features_rel`
- `playlist_stats`

The canonical relative read path is persisted-first. Relative rows are refreshed after successful enrichment or via `refresh-relative-playlist`.

### Recommendation/scoring layer

Current recommendation output is lane-based and target-aware:

- `maintain`
- `build`
- `reset`
- `jump`
- `contrast`

Current scoring metadata also exposes:

- active scoring contract id
- compatible analysis/config signatures
- component availability/active state
- capability flags for known gaps such as unavailable vocals/window features

## Model Runtime Topology

Current runtime split:

- shared **TensorFlow/Essentia** service
  - TempoCNN BPM
  - Essentia semantic inference
- separate **PyTorch** service
  - MusicalKeyCNN key inference

Why this split:

- TempoCNN and Essentia share a TensorFlow/Essentia stack
- MusicalKeyCNN is PyTorch and is kept separate for stability and simpler CUDA/runtime management

## Analysis Modes

### `fast_pass`

- computes BPM + key only
- writes `track_features_fast`
- intended for immediate feedback

### `staged` (default)

- computes fast BPM + key immediately
- queues enrichment jobs in `analysis_jobs`
- returns without waiting for absolute/relative enrichment

### `full`

- uses the same staged pipeline
- waits for enrichment and relative refresh to finish
- intended for explicit full analysis/backfill runs

## BPM / Key Resolution Behavior

Important nuance:

- `imported`
  - metadata imported from a DJ library source such as Rekordbox XML
- `tag`
  - metadata embedded directly in the audio file
- `tempocnn`
  - BPM model estimate
- `musicalkeycnn`
  - key model estimate

Combined source labels such as `imported+tempocnn` or `tag+musicalkeycnn` mean:

- one source won resolution
- the model agreed closely enough to boost confidence
- values are **not averaged**

## Setup

### 1. Bootstrap

On Windows, start here:

- [Bootstrap on Windows](./docs/bootstrap-windows.md)

### 2. Create `.env`

```powershell
Copy-Item .env.example .env
```

This is required because [compose.yaml](./compose.yaml) uses `.env`.

### 3. Install the Python package

```powershell
python -m pip install --user -e ".\python[dev]"
```

### 4. Run prerequisite checks

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-prereqs.ps1
```

### 5. Run migrations

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dbmate.ps1 status
powershell -ExecutionPolicy Bypass -File .\scripts\docker-compose.ps1 --profile ops run --rm migrate
```

## Building Local Model Services

The active runtime topology uses two warm model-service containers:

- one shared TensorFlow/Essentia service for TempoCNN BPM + Essentia semantic inference
- one separate PyTorch service for MusicalKeyCNN key inference

The `tempocnn` folder and helper scripts still exist, but they now alias into the shared TensorFlow/Essentia image/service rather than starting an independent third container.

Build images:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-tempocnn-image.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\build-musicalkeycnn-image.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\build-essentia-semantics-image.ps1
```

Warm-start services manually if desired:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-tempocnn-service.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start-musicalkeycnn-service.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start-essentia-semantics-service.ps1
```

Notes:

- `build-tempocnn-image.ps1` delegates to the shared Essentia/TensorFlow image build
- `start-tempocnn-service.ps1` delegates to the shared Essentia/TensorFlow service startup
- starting either the TempoCNN or Essentia semantics service path results in the same shared container

The CLI also starts them on demand.

## Common API Workflows

Start the local scorer and API:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\compile-proto.ps1
```

Run the scorer and the API in separate terminals because `python -m cuemate_analysis serve-scoring` is a long-running server process.

Terminal 1:

```powershell
python -m cuemate_analysis serve-scoring --host 127.0.0.1 --port 47834
```

Terminal 2:

```powershell
go run ./go/cmd/apiserver
```

PowerShell background alternative:

```powershell
Start-Process powershell -ArgumentList '-NoExit','-Command','python -m cuemate_analysis serve-scoring --host 127.0.0.1 --port 47834'
go run ./go/cmd/apiserver
```

Check service health and scoring metadata:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8080/healthz
Invoke-RestMethod -Uri http://127.0.0.1:8080/readyz
Invoke-RestMethod -Uri http://127.0.0.1:8080/scoring/metadata | ConvertTo-Json -Depth 10
```

Request live recommendations:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8080/recommendations `
  -ContentType "application/json" `
  -Body '{"playlist_name":"Fred again","current_track_id":"trk_example123","target":"build"}' |
  ConvertTo-Json -Depth 10
```

Record a played outcome:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8080/events/played `
  -ContentType "application/json" `
  -Body '{"recommendation_event_id":"evt_example","chosen_track_id":"trk_example456"}'
```

Read playlist feedback summary:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8080/feedback/summary `
  -ContentType "application/json" `
  -Body '{"playlist_name":"Fred again"}' |
  ConvertTo-Json -Depth 10
```

Local mobile/bootstrap sync primitives:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8080/sync/playlists/snapshot `
  -ContentType "application/json" `
  -Body '{"playlist_name":"Fred again"}' |
  ConvertTo-Json -Depth 12

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8080/sync/outbox/pull `
  -ContentType "application/json" `
  -Body '{"limit":100}' |
  ConvertTo-Json -Depth 12
```

## Common CLI Workflows

### Import local folders

```powershell
python -m cuemate_analysis import-playlist --name "My Playlist" .\path\to\audio
```

### Inspect DJ-library playlists

```powershell
python -m cuemate_analysis list-dj-playlists --source rekordbox --library "D:\Exports\rekordbox.xml"
python -m cuemate_analysis list-dj-playlists --source traktor --library "D:\Exports\collection.nml"
python -m cuemate_analysis list-dj-playlists --source serato --library "D:\Music\_Serato_\Subcrates"
```

### Import DJ-library playlists

```powershell
python -m cuemate_analysis import-dj-playlist --source rekordbox --library "D:\Exports\rekordbox.xml" --playlist "Main Room"
python -m cuemate_analysis import-dj-playlist --source traktor --library "D:\Exports\collection.nml" --playlist "Warmup" --name "Warmup"
python -m cuemate_analysis import-dj-playlist --source serato --library "D:\Music\_Serato_\Subcrates" --playlist "Club Set"
```

Notes:

- Rekordbox support expects the **XML library export**
- Serato import currently gives playlist membership and local paths only; it does not import BPM/key metadata
- if a playlist name already exists locally, import with `--name` to avoid name collisions

### Analyze playlists

```powershell
python -m cuemate_analysis analyze-playlist --playlist "My Playlist"
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode staged --force
python -m cuemate_analysis run-analysis-worker --limit 25
python -m cuemate_analysis analyze-playlist --playlist "My Playlist" --analysis-mode full --force
```

### Inspect tracks and playlists

```powershell
python -m cuemate_analysis list-playlist --name "My Playlist"
python -m cuemate_analysis show-track --track-id trk_example123
```

### BPM / key-only workflows

```powershell
python -m cuemate_analysis analyze-bpm "D:\path\to\track.wav"
python -m cuemate_analysis analyze-bpm-playlist --playlist "Fred again"
python -m cuemate_analysis analyze-bpm-key "D:\path\to\track.wav"
python -m cuemate_analysis analyze-bpm-key-playlist --playlist "Fred again"
```

### Relative context workflows

```powershell
python -m cuemate_analysis analyze-relative-playlist --playlist "Fred again"
python -m cuemate_analysis analyze-relative-playlist --playlist "Fred again" --energy-source canonical
python -m cuemate_analysis analyze-relative-playlist --playlist "Fred again" --energy-source heuristic_legacy
python -m cuemate_analysis refresh-relative-playlist --playlist "Fred again"
```

### Recommendation/scoring workflows

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\compile-proto.ps1
python -m cuemate_analysis recommend-next --playlist "Fred again" --current-track trk_example123 --target maintain
python -m cuemate_analysis score-pair --playlist "Fred again" --current trk_example123 --candidate trk_example456 --target reset
python -m cuemate_analysis inspect-scoring-weights --playlist "Fred again"
python -m cuemate_analysis inspect-scoring-metadata
python -m cuemate_analysis inspect-scoring-metadata --json
python -m cuemate_analysis serve-scoring --host 127.0.0.1 --port 47834
go run ./go/cmd/scoringctl metadata
go run ./go/cmd/scoringctl score --fixture .\go\testdata\score_candidate.json
python -m cuemate_analysis feedback-summary --playlist "Fred again" --json
python -m cuemate_analysis feedback-tune --playlist "Fred again" --preview-only --json
python -m cuemate_analysis run-feedback-worker --limit 10
```

### Essentia semantic workflows

```powershell
python -m cuemate_analysis download-essentia-semantic-models
python -m cuemate_analysis analyze-essentia-playlist --playlist "Fred again"
python -m cuemate_analysis prewarm-model-services
```

### Cache and service maintenance

```powershell
python -m cuemate_analysis clear-analysis-queue
python -m cuemate_analysis clear-analysis-queue --job-kind enrichment
python -m cuemate_analysis clear-analysis-queue --include-running
python -m cuemate_analysis purge-model-cache
python -m cuemate_analysis purge-model-cache --backend tempocnn
python -m cuemate_analysis purge-model-cache --backend essentia_semantics
python -m cuemate_analysis purge-model-cache --backend essentia_semantics --clear-warm-services
```

## Performance Notes

- staged analysis is the intended interactive mode
- `track_features_fast` exists to make BPM/key available quickly
- enrichment then fills in canonical absolute data and refreshes relative data
- keeping warm model services alive matters a lot for repeat performance
- persistent model caches live in `data/inference-cache.db`
- local DSP remains CPU-bound
- TempoCNN and Essentia semantics use the shared TensorFlow/Essentia runtime
- MusicalKeyCNN uses its own warm PyTorch worker

Current Essentia behavior:

- default semantic mode is a **middle excerpt**
- tracks can escalate to **multi-sample** semantic inference when configured triggers fire
- excerpt-only decode is used in the Essentia service for semantic windows

## Development Commands

Run lint:

```powershell
python -m ruff check python/src python/tests
```

Run tests:

```powershell
python -m pytest python/tests
```

Compile protobuf contract:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\compile-proto.ps1
```

The compile helper runs `buf lint`, writes `data/scoring.pb`, and generates Python and Go gRPC stubs into `python/src/djengine/` and `go/gen/`.

For the shipped Milestone 5 operator flow, see [docs/feedback-loop-local.md](docs/feedback-loop-local.md).

Benchmark local DSP:

```powershell
python -m cuemate_analysis benchmark-dsp --playlist "Fred again"
python -m cuemate_analysis benchmark-dsp --path "D:\Music\track.flac"
```

## Known Boundaries

- Go decision plane now serves the shipped local recommendations API, event capture, feedback summary surfaces, and single-consumer snapshot/outbox sync primitives that a local mobile/bootstrap client can consume
- windowed intro/outro analysis is intentionally deferred
- `transition_support`, `vocal_transition`, and `rhythmic_continuity` are still explicit stubs and are excluded from weighted scoring
- `vocals_abs` / `vocals_rel` are not populated by the current analysis pipeline yet, so vocal-dependent recommendation logic remains limited
- some metadata imported from DJ libraries or file tags can still be wrong; the current resolver uses provenance + confidence heuristics rather than treating any source as perfect
- Essentia semantic calibration infrastructure exists, and the current semantic lane is live; future listening-test calibration can still refine those heuristics

## Roadmap

Done:

- Milestone 1 absolute analysis
- Milestone 2 persisted relative context
- Milestone 3 Python recommendation/scoring core
- Milestone 4 local service/API bootstrap, write-side cleanup, and explicit-ack sync protocol
- Milestone 5 per-playlist feedback loop and auto-tuned weights

Later:

- multi-device/authenticated sync surfaces
- optional advanced enrichments
