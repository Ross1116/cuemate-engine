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

Current scope in this slice:

- a thin Go gRPC client for the Python scoring service
- a local HTTP API server for live recommendations and scorer metadata
- SQLite hydration for playlist/current/history scoring inputs
- scorer readiness/circuit-breaker handling in the Go decision plane
- generated Go protobuf/grpc artifacts under `go/gen/`
- a small `scoringctl` smoke CLI for metadata and fixture-driven score calls

The Go layer still does not own scoring semantics. Python remains the authoritative scorer, while Go now owns the read-only live recommendation API path on top of SQLite + gRPC.

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

Configuration defaults:

- `GO_API_ADDR=127.0.0.1:8080`
- `SCORING_GRPC_ADDR=127.0.0.1:47834`
- `SCORING_RPC_TIMEOUT_MS=250`
