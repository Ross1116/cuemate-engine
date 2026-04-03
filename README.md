# CueMate Engine

CueMate Engine starts as a monorepo with a hard split between the Python analysis plane and the Go decision plane. This repository is intentionally at a setup-only stage: contract, tooling, workspace config, migration scaffolding, and deployment groundwork are in place, but service and DSP implementation code has not started yet.

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
- `python/` is configured with a `pyproject.toml` for future analysis-plane packaging
- `dbmate` is the migration tool for forward-only SQL schema changes
- `compose.yaml` currently provides an operations-only migration service
- Generated artifacts and local env files are kept out of git by default

## Useful commands

Validate the protobuf contract:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\compile-proto.ps1
```

Run dbmate from the repo root with the correct migration paths:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dbmate.ps1 status
```

Run the Docker Compose migration profile with an isolated local Docker config:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\docker-compose.ps1 --profile ops run --rm migrate
```

## Intent for the next commits

- add Python scoring-service and analysis-plane packages under `python/src/`
- add Go API and orchestration packages under `go/cmd/` and `go/internal/`
- add the first SQLite migration files in `db/migrations/`
- add service Dockerfiles only when real app entrypoints exist
