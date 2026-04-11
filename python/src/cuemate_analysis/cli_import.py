"""CLI handlers for import/ingest commands."""
from __future__ import annotations

import argparse
from pathlib import Path

import cuemate_analysis.cli as _cli
from cuemate_analysis.analysis import utc_now
from cuemate_analysis.dj_import import list_dj_playlists, load_dj_playlist
from cuemate_analysis.ingest import (
    discover_audio_files,
    make_playlist_id,
    read_track_metadata,
    read_track_metadata_with_overrides,
)


def handle_import_playlist(args: argparse.Namespace) -> int:
    settings = _cli.load_runtime_settings()
    playlist_id = make_playlist_id(args.name)
    paths = discover_audio_files(args.paths)
    if not paths:
        raise SystemExit("No supported audio files were found in the provided paths.")
    timestamp = utc_now()
    tracks = [read_track_metadata(path) for path in paths]

    with _cli.Database(settings.database_path) as database:
        database.upsert_playlist(playlist_id, args.name, len(tracks), timestamp)
        for track in tracks:
            database.upsert_track(track, timestamp)
        database.replace_playlist_tracks(playlist_id, [track.id for track in tracks], timestamp)

    print(
        f"Imported playlist '{args.name}' ({playlist_id}) with {len(tracks)} track(s) into {settings.database_path}"
    )
    return 0


def handle_list_dj_playlists(args: argparse.Namespace) -> int:
    library_path = Path(args.library).expanduser().resolve()
    if not library_path.exists():
        raise FileNotFoundError(f"DJ library path was not found: {library_path}")

    playlist_names = list_dj_playlists(args.source, library_path)
    if not playlist_names:
        print(f"No playlists were found in {library_path}")
        return 0

    print(f"Found {len(playlist_names)} playlist(s) in {library_path}")
    for name in playlist_names:
        print(f"- {name}")
    return 0


def handle_import_dj_playlist(args: argparse.Namespace) -> int:
    settings = _cli.load_runtime_settings()
    library_path = Path(args.library).expanduser().resolve()
    if not library_path.exists():
        raise FileNotFoundError(f"DJ library path was not found: {library_path}")

    source_playlist_name = args.playlist
    local_playlist_name = args.name or source_playlist_name
    playlist_id = make_playlist_id(local_playlist_name)
    timestamp = utc_now()
    imported_entries = load_dj_playlist(args.source, library_path, source_playlist_name)
    if not imported_entries:
        raise SystemExit(
            f"Playlist '{source_playlist_name}' from {args.source} did not yield any usable track entries."
        )

    tracks = []
    skipped: list[str] = []
    for entry in imported_entries:
        if not entry.file_path.is_file():
            skipped.append(f"{entry.file_path} (file not found)")
            continue
        try:
            tracks.append(
                read_track_metadata_with_overrides(
                    entry.file_path,
                    bpm_imported=entry.bpm_imported,
                    key_imported=entry.key_imported,
                    title_override=entry.title,
                    artist_override=entry.artist,
                    genre_override=entry.genre,
                    import_source=entry.import_source,
                )
            )
        except Exception as exc:
            skipped.append(f"{entry.file_path} ({exc})")

    if not tracks:
        raise SystemExit(
            f"Playlist '{source_playlist_name}' from {args.source} did not contain any importable local audio files."
        )

    with _cli.Database(settings.database_path) as database:
        database.upsert_playlist(playlist_id, local_playlist_name, len(tracks), timestamp)
        for track in tracks:
            database.upsert_track(track, timestamp)
        database.replace_playlist_tracks(playlist_id, [track.id for track in tracks], timestamp)

    print(
        f"Imported DJ playlist '{source_playlist_name}' from {args.source} as "
        f"'{local_playlist_name}' ({playlist_id}) with {len(tracks)} track(s)."
    )
    if skipped:
        print(f"Skipped {len(skipped)} track(s) that could not be resolved locally.")
        for detail in skipped[:10]:
            print(f"- {detail}")
        if len(skipped) > 10:
            print(f"- ... and {len(skipped) - 10} more")
    return 0
