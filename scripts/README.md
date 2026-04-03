# Scripts

This directory holds setup and contract-maintenance helpers. No product logic lives here.

## Available helpers

- `check-prereqs.ps1`: checks whether the core local tools are available from the current shell and whether Docker/Tailscale look reachable
- `compile-proto.ps1`: validates the protobuf contract with the local `protoc` compiler and writes a descriptor set to `data/scoring.pb`
- `build-tempocnn-image.ps1`: builds the local Docker image that backs the primary TempoCNN BPM analyzer
- `start-tempocnn-service.ps1`: starts the warm TempoCNN service container for faster repeated BPM analysis
- `dbmate.ps1`: runs `dbmate` from the repo root with the repository migration and schema paths preconfigured, preferring the repo-local fallback binary when needed
- `docker-compose.ps1`: runs `docker-compose` with a repo-local `DOCKER_CONFIG` to avoid machine-specific Docker CLI config issues

## Usage

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-prereqs.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\compile-proto.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\build-tempocnn-image.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start-tempocnn-service.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\dbmate.ps1 status
powershell -ExecutionPolicy Bypass -File .\scripts\docker-compose.ps1 --profile ops run --rm migrate
```
