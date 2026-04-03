# Stack Decisions

This repository is intentionally local-first.

## Runtime shape

- Go hosts the public API and orchestration surface.
- Python hosts the authoritative scoring service over gRPC.
- SQLite is the canonical local database on PC.
- Mobile consumes playlist/crate snapshots rather than mutating canonical analysis tables.

## Operational choices

- CI/CD: GitHub Actions, CodeRabbit, GitHub Releases.
- Migrations: dbmate with forward-only SQL files in `db/migrations/`.
- Local deployment: native processes first, Docker Compose once service images exist.
- Private remote access: Tailscale.
- Optional always-on hobby host: Oracle Cloud Always Free VM, only after the local-first flow is stable.

## What we are explicitly not doing yet

- No hosted primary Postgres.
- No Kubernetes.
- No cloud-first deployment assumption.
- No separate scoring implementation outside the Python service.
