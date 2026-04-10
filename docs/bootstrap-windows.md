# Bootstrap on Windows

This repository currently has a **Windows-first local bootstrap**. The app itself is not implemented yet, but these instructions get a fresh Windows machine to the same infrastructure/setup state as the current repo baseline.

## Scope of this guide

After completing this guide, you should have:

- the repo cloned locally
- local environment defaults copied into `.env`
- protobuf validation working
- Docker Desktop and Docker Compose working
- Tailscale connected
- SQL migrations runnable through Docker Compose
- optional local `dbmate` CLI access

## Assumptions

- Windows 11 or recent Windows 10
- PowerShell available
- You can install desktop software on the machine
- The repository has already been cloned locally

## 1. Install required tools

Install these before doing anything else.

### Required

- Git
- Go 1.24+
- Python 3.12+
- Protocol Buffers compiler (`protoc`)
- Buf CLI
- Docker Desktop

### Recommended

- Tailscale
- dbmate
- `protoc-gen-go`
- `protoc-gen-go-grpc`

### Suggested install commands

These are the commands we expect to work on a fresh Windows machine.

```powershell
winget install Git.Git
winget install GoLang.Go
winget install Python.Python.3.12
winget install bufbuild.buf
winget install Docker.DockerDesktop
winget install Tailscale.Tailscale
winget install amacneil.dbmate
```

For `protoc`, install the latest official Windows release from the protobuf GitHub releases page and make sure its `bin` directory is on your user `Path`.

Official releases:

- https://github.com/protocolbuffers/protobuf/releases/latest

Install the Go protobuf plugins after Go is available:

```powershell
go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.11
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5.1
```

## 2. Open a fresh PowerShell

After installs, close your terminal and open a new PowerShell so updated PATH entries are picked up.

## 3. Verify prerequisites

From the repo root, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-prereqs.ps1
```

This checks the core CLI tools, `.env`, Docker reachability, and Tailscale visibility.

## 4. Copy local environment defaults

Create a local `.env` file from the checked-in example.

```powershell
Copy-Item .env.example .env
```

If `.env` already exists, leave it alone.

## 5. Start Docker Desktop

Open Docker Desktop and wait until it is fully ready.

Verify with:

```powershell
docker info
```

If this fails, do not continue until Docker is healthy.

## 6. Authenticate Tailscale

Bring the device onto your tailnet:

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" up
```

Then verify:

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" status
```

## 7. Validate the protobuf contract

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\compile-proto.ps1
```

Expected result: the contract validates and a descriptor file is written under `data/`.
The script also generates Python stubs under `python/src/djengine/` and Go stubs under `go/gen/`.

## 8. Validate Docker Compose wiring

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\docker-compose.ps1 --profile ops config
```

Expected result: Compose renders the `migrate` service configuration without errors.

## 9. Check migration status

Preferred command:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dbmate.ps1 status
```

If the repository already contains applied migrations, you should see them listed as applied after the database is created and migrated.

## 10. Apply migrations

Use the Docker-based migration path as the **default supported workflow** on a fresh machine:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\docker-compose.ps1 --profile ops run --rm migrate
```

This is the most reproducible path because it does not depend on the local `dbmate` binary behaving perfectly.

## 11. Optional: create a new migration locally

If local `dbmate` works on your machine, you can create a new migration file with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dbmate.ps1 new create_schema_metadata
```

## Supported local workflows

### Most reproducible

- `compile-proto.ps1`
- `docker-compose.ps1 --profile ops run --rm migrate`

### Optional local convenience

- `dbmate.ps1`

The repository includes a wrapper for `dbmate`, but on some Windows machines WinGet command aliases may be flaky. If that happens, use the Docker Compose migration workflow instead.

## Troubleshooting

### `buf` or `dbmate` is installed but will not run

Open a fresh PowerShell first. If the CLI still fails, prefer the checked-in wrappers and the Docker-based migration path.

### `docker info` fails

Docker Desktop is not ready yet, or first-run setup has not been completed.

### Tailscale command opens but `status` fails

Finish the browser auth flow first, then rerun `tailscale status`.

### PowerShell says `Unexpected token 'up'`

Use the call operator when invoking a quoted executable path:

```powershell
& "C:\Program Files\Tailscale\tailscale.exe" up
```

## What this guide does not promise yet

This repo is currently documented for a **fresh Windows machine**. It is not yet fully documented as a zero-friction bootstrap for macOS or Linux.
