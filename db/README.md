# Database and migrations

This directory holds forward-only SQL migrations and the checked-in schema snapshot.

## Rules

- SQLite is the authoritative local store on PC.
- Schema changes go through SQL migrations only.
- Migrations are forward-only in normal workflow.
- `db/schema.sql` is a generated snapshot owned by `dbmate`.
- Avoid having more than one application service perform automatic database migrations against the same database; designate a single migration owner such as a migration job or one startup path.
- On a fresh machine, the Docker Compose migration path is the most reproducible way to apply migrations.

## Local commands

Create a migration:

```powershell
.\scripts\dbmate.ps1 new create_tracks_table
```

Apply pending migrations through Docker Compose:

```powershell
.\scripts\docker-compose.ps1 --profile ops run --rm migrate
```

Show migration status:

```powershell
.\scripts\dbmate.ps1 status
```
