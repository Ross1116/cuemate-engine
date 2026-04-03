# Decision Plane

The Go decision plane is responsible for API surfaces, request orchestration, sync policy, session state, and translating external requests into the canonical scoring RPCs.

## Intended layout

```text
go/
|- cmd/       # Entrypoints such as the API server
|- internal/  # Private decision-plane packages
|- gen/       # Generated protobuf/grpc Go artifacts
|- go.mod
```

No implementation code is committed yet. This directory currently exists as a module and layout baseline.
