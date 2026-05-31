param(
    [string]$SourceDb = (Join-Path $env:LOCALAPPDATA "CueMate\data\cuemate.db"),
    [string]$OutputDb = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..")) "dist\showcase\cuemate-showcase.db"),
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$SourceDb = [System.IO.Path]::GetFullPath($SourceDb)
$OutputDb = [System.IO.Path]::GetFullPath($OutputDb)

if (-not (Test-Path $SourceDb -PathType Leaf)) {
    throw "Source CueMate database not found at $SourceDb"
}

$outputDir = Split-Path -Parent $OutputDb
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
Copy-Item -Path $SourceDb -Destination $OutputDb -Force

$sanitizer = Join-Path ([System.IO.Path]::GetTempPath()) ("cuemate-showcase-sanitize-{0}.py" -f ([guid]::NewGuid().ToString("N")))
$code = @'
from __future__ import annotations

import pathlib
import sqlite3
import sys

db_path = pathlib.Path(sys.argv[1])

private_tables = [
    "analysis_jobs",
    "sync_outbox",
    "playlist_sync_state",
    "remote_pairing_tokens",
    "remote_sessions",
    "manual_corrections",
    "feedback_tuning_jobs",
]

feature_hash_tables = [
    "track_features_abs",
    "track_features_fast",
]


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


with sqlite3.connect(db_path) as conn:
    conn.execute("PRAGMA foreign_keys = OFF")
    if not table_exists(conn, "playlists") or not table_exists(conn, "tracks"):
        raise SystemExit("Input database is missing CueMate playlists/tracks tables.")

    playlist_count = int(conn.execute("SELECT COUNT(*) FROM playlists").fetchone()[0])
    if playlist_count < 1:
        raise SystemExit("Showcase export requires at least one playlist.")

    if "spotify_url" not in columns(conn, "playlists"):
        conn.execute("ALTER TABLE playlists ADD COLUMN spotify_url TEXT")

    if "file_path" in columns(conn, "tracks"):
        conn.execute("UPDATE tracks SET file_path = 'showcase://tracks/' || id")
    if "file_hash" in columns(conn, "tracks"):
        conn.execute("UPDATE tracks SET file_hash = NULL")

    for table in feature_hash_tables:
        if table_exists(conn, table) and "source_file_hash" in columns(conn, table):
            conn.execute(f"""UPDATE "{table}" SET source_file_hash = 'showcase'""")

    for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'"):
        table = str(row[0])
        if "user_id" in columns(conn, table):
            conn.execute(f"""UPDATE "{table}" SET user_id = 'showcase'""")

    for table in private_tables:
        if table_exists(conn, table):
            conn.execute(f'DELETE FROM "{table}"')

    raw_path_count = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM tracks
            WHERE file_path GLOB '[A-Za-z]:*'
               OR file_path GLOB '/*'
               OR file_path GLOB '\\*'
            """
        ).fetchone()[0]
    )
    if raw_path_count:
        raise SystemExit(f"Showcase export still contains {raw_path_count} raw file paths.")

    conn.commit()
    conn.execute("VACUUM")

print(f"Showcase database written to {db_path}")
'@

try {
    Set-Content -Path $sanitizer -Value $code -Encoding UTF8
    & $Python $sanitizer $OutputDb
    if ($LASTEXITCODE -ne 0) {
        throw "Showcase database sanitization failed with exit code $LASTEXITCODE"
    }
} finally {
    Remove-Item -Path $sanitizer -Force -ErrorAction SilentlyContinue
}
