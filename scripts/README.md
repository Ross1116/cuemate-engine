# Scripts

This directory holds setup and contract-maintenance helpers. No product logic lives here.

## Available helpers

- `check-prereqs.ps1`: checks whether the core local tools are available from the current shell and whether Docker/Tailscale look reachable
- `compile-proto.ps1`: runs `buf lint`, writes a descriptor set to `data/scoring.pb`, and generates Python protobuf/gRPC stubs into `python/src/djengine/`
- `build-tempocnn-image.ps1`: compatibility alias that builds the shared TensorFlow/Essentia image used by TempoCNN and Essentia semantics
- `build-essentia-semantics-image.ps1`: builds the shared TensorFlow/Essentia image directly (same image as `build-tempocnn-image.ps1`)
- `build-musicalkeycnn-image.ps1`: builds the local Docker image that backs the primary MusicalKeyCNN key analyzer
- `start-tempocnn-service.ps1`: compatibility alias that starts the shared TensorFlow/Essentia service used by TempoCNN and Essentia semantics
- `start-essentia-semantics-service.ps1`: starts the shared TensorFlow/Essentia service directly
- `start-musicalkeycnn-service.ps1`: starts the warm MusicalKeyCNN service container for faster repeated key analysis
- `dbmate.ps1`: runs `dbmate` from the repo root with the repository migration and schema paths preconfigured, preferring the repo-local fallback binary when needed
- `docker-compose.ps1`: runs `docker-compose` with a repo-local `DOCKER_CONFIG` to avoid machine-specific Docker CLI config issues

## Usage

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-prereqs.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\compile-proto.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\build-tempocnn-image.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\build-essentia-semantics-image.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\build-musicalkeycnn-image.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start-tempocnn-service.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start-essentia-semantics-service.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start-musicalkeycnn-service.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\dbmate.ps1 status
powershell -ExecutionPolicy Bypass -File .\scripts\docker-compose.ps1 --profile ops run --rm migrate
```
