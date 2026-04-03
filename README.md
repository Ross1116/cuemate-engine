# cuemate-engine

CueMate Engine is organized as a small monorepo with a clean split between the analysis and decision planes.

## Repository layout

```text
/
|- python/   # Analysis Plane: decoding, DSP, and gRPC scoring service
|- go/       # Decision Plane: API, sync orchestration, and session state
|- proto/    # Shared scoring contract definitions
|- data/     # Local-only SQLite storage
|- scripts/  # Setup and code generation helpers
```

The Python side handles the heavy analysis work, while the Go side owns orchestration and API-facing decision logic.
