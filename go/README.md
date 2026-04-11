# Decision Plane

The Go decision plane is responsible for API surfaces, request orchestration, sync policy, session state, and translating external requests into the canonical scoring RPCs.

## Intended layout

```text
go/
|- cmd/       # Entrypoints such as the API server or smoke CLIs
|- internal/  # Private decision-plane packages such as scoring clients
|- gen/       # Generated protobuf/grpc Go artifacts
|- go.mod
```

Current scope:

- a thin Go gRPC client for the Python scoring service
- a local HTTP API server for live recommendations and scorer metadata
- SQLite hydration for playlist/current/history scoring inputs
- scorer readiness/circuit-breaker handling in the Go decision plane
- recommendation event capture for successful `/recommendations` responses
- recommendation outcome capture via `/events/played`
- read-only feedback summary via `/feedback/summary`
- manual BPM/key corrections that mark playlists stale and queue full reanalysis
- playlist-scoped JSON snapshot export for sync/bootstrap scenarios
- explicit-ack snapshot delivery state via `playlist_sync_state`
- pull/ack `sync_outbox` processing for played events and corrections
- generated Go protobuf/grpc artifacts under `go/gen/`
- a small `scoringctl` smoke CLI for metadata and fixture-driven score calls

The Go layer still does not own scoring semantics. Python remains the authoritative scorer, while Go now owns the local API/orchestration path on top of SQLite + gRPC.

## Smoke workflow

From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\compile-proto.ps1
python -m cuemate_analysis serve-scoring
go run ./go/cmd/apiserver
go run ./go/cmd/scoringctl metadata
go run ./go/cmd/scoringctl score --fixture .\go\testdata\score_candidate.json
```

With the API server running:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/recommendations -ContentType "application/json" -Body '{"playlist_name":"Fred again","current_track_id":"trk_example123","target":"build"}'
```

Additional write/snapshot surfaces:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/events/played -ContentType "application/json" -Body '{"recommendation_event_id":"evt_123","chosen_track_id":"trk_next"}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/feedback/summary -ContentType "application/json" -Body '{"playlist_name":"Fred again"}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/corrections -ContentType "application/json" -Body '{"track_id":"trk_example123","field":"bpm","new_value":128.0}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/sync/playlists/snapshot -ContentType "application/json" -Body '{"playlist_name":"Fred again"}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/sync/playlists/snapshot/ack -ContentType "application/json" -Body '{"snapshot_id":"snap_123"}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/sync/outbox/pull -ContentType "application/json" -Body '{"limit":100}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/sync/outbox/ack -ContentType "application/json" -Body '{"ack_through_id":42}'
```

Weight precedence in live recommendations:

- `feedback_tuned_weights`
- `adapted_weights`
- static scoring weights

Configuration defaults:

- `GO_API_ADDR=127.0.0.1:8080`
- `SCORING_GRPC_ADDR=127.0.0.1:47834`
- `SCORING_RPC_TIMEOUT_MS=250`
