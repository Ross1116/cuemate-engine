# CueMate Engine

CueMate Engine starts as a monorepo with a hard split between the Python analysis plane and the Go decision plane. This repository is intentionally at a setup-only stage: contract, tooling, workspace config, and folder scaffolding are in place, but service and DSP implementation code has not started yet.

## Repository layout

```text
/
|- python/   # Analysis Plane: audio ingest, DSP, artifact production, scoring service
|- go/       # Decision Plane: API, orchestration, sync, session surfaces
|- proto/    # Shared protobuf scoring contract
|- data/     # Local-only SQLite database, descriptors, and scratch artifacts
|- scripts/  # Utility scripts for validation and code generation
```

## Current baseline

- Shared protobuf contract lives at `proto/djengine/scoring/v1/scoring.proto`
- `buf.yaml` defines `proto/` as the protobuf module root
- `go/` is its own Go module and is included from the root `go.work`
- `python/` is configured with a `pyproject.toml` for future analysis-plane packaging
- Generated artifacts are kept out of git by default

## Useful commands

Validate the protobuf contract with the locally installed protobuf compiler:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\compile-proto.ps1
```

## Intent for the next commits

- add Python scoring-service and analysis-plane packages under `python/src/`
- add Go API and orchestration packages under `go/cmd/` and `go/internal/`
- add generated protobuf stubs into dedicated generated-code directories, not hand-written package paths
