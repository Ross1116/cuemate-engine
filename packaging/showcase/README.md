# CueMate Showcase Container

This package builds a hosted, read-only CueMate demo. It serves the built web app, the Go API, a sanitized SQLite snapshot, and the lightweight Python scoring service in one container.

## Build A Showcase DB

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\export-showcase-db.ps1
```

The default output is:

```text
dist\showcase\cuemate-showcase.db
```

The exporter keeps real track titles, artists, playlists, Spotify playlist links, analysis values, playlist stats, recommendation events, and feedback history. It strips local file paths, file hashes, sessions, remote tokens, analysis jobs, sync state, manual corrections, and worker state.

Spotify playlist buttons come from the `spotify_url` value saved on each playlist in the local dev app. Attach or clear those links in the dev UI, then export the showcase DB so the read-only demo uses the same values.

## Build And Run Locally

```powershell
docker build -f .\packaging\showcase\Dockerfile -t cuemate-showcase:local .
docker run --rm -p 8080:8080 cuemate-showcase:local
```

Open:

```text
http://127.0.0.1:8080
```

Useful checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/healthz
Invoke-RestMethod http://127.0.0.1:8080/readyz
Invoke-RestMethod http://127.0.0.1:8080/setup/status
Invoke-RestMethod http://127.0.0.1:8080/playlists
```

## Deployment Notes

Any host that can run a single Docker container should work: Fly.io, Render, Railway, or a small VPS.

Required runtime settings are already defaulted in the image:

```text
CUEMATE_SHOWCASE_MODE=1
DATABASE_URL=sqlite:file:/app/data/cuemate-showcase.db?mode=ro&immutable=1
WEB_DIST_DIR=/app/web/dist
SCORING_GRPC_ADDR=127.0.0.1:47834
```

Set `PORT` if the host injects a dynamic port. The entrypoint maps it to `GO_API_ADDR=0.0.0.0:$PORT`.

To refresh the public demo, run analysis locally, export a new `cuemate-showcase.db`, rebuild the image, and redeploy.
