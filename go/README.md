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
- generated Go protobuf/grpc artifacts under `go/gen/`
- a small `scoringctl` smoke CLI for metadata and fixture-driven score calls

The Go layer still does not own scoring semantics or playlist hydration. Python remains the authoritative scorer.

## Smoke workflow

From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\compile-proto.ps1
python -m cuemate_analysis serve-scoring
go run ./go/cmd/scoringctl metadata
go run ./go/cmd/scoringctl score --fixture .\go\testdata\score_candidate.json
```

Configuration defaults:

- `SCORING_GRPC_ADDR=127.0.0.1:47834`
- `SCORING_RPC_TIMEOUT_MS=250`
