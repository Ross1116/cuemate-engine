from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

from cuemate_analysis.analysis import analyze_track, utc_now
from cuemate_analysis.config import load_runtime_settings
from cuemate_analysis.database import Database
from cuemate_analysis.dj_import import list_dj_playlists, load_dj_playlist
from cuemate_analysis.ingest import (
    discover_audio_files,
    make_playlist_id,
    read_track_metadata,
    read_track_metadata_with_overrides,
)
from cuemate_analysis.key_backend import (
    KeyEstimate,
    MUSICALKEYCNN_POLICY_FULL_TRACK,
    estimate_musicalkeycnn_keys,
    purge_musicalkeycnn_cache,
    resolve_musicalkeycnn_model_path,
)
from cuemate_analysis.persistent_inference_cache import resolve_inference_cache_path
from cuemate_analysis.tempo_backend import (
    TempoEstimate,
    estimate_tempocnn_bpms,
    purge_tempocnn_cache,
    resolve_tempocnn_model_path,
)

TEMPOCNN_PROGRESS_BATCH_SIZE = 8


def hash_file_identity(path: Path) -> str:
    if not path.is_file():
        return f"missing-{path.name}"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def build_effective_analysis_signature(
    base_signature: str,
    *,
    tempocnn_model: str | None = None,
    tempocnn_accelerator: str = "auto",
    musicalkeycnn_model: str | None = None,
    musicalkeycnn_device: str = "auto",
    musicalkeycnn_policy: str = MUSICALKEYCNN_POLICY_FULL_TRACK,
) -> str:
    tempo_model_hash = hash_file_identity(resolve_tempocnn_model_path(tempocnn_model))
    key_model_hash = hash_file_identity(resolve_musicalkeycnn_model_path(musicalkeycnn_model))
    return (
        f"{base_signature}"
        f"-tempo-tempocnn-{tempo_model_hash}-{tempocnn_accelerator}"
        f"-key-musicalkeycnn-{key_model_hash}-{musicalkeycnn_device}-{musicalkeycnn_policy}"
    )


def summarize_estimate(estimate: TempoEstimate) -> str:
    if not estimate.available or estimate.bpm is None:
        return f"unavailable ({estimate.elapsed_ms:.1f} ms)" if estimate.elapsed_ms is not None else "unavailable"
    return (
        f"{estimate.details.get('display_bpm', round(estimate.bpm, 1))} BPM "
        f"in {estimate.elapsed_ms:.1f} ms"
    )


def summarize_key_estimate(estimate: KeyEstimate) -> str:
    if not estimate.available or estimate.key is None:
        return f"unavailable ({estimate.elapsed_ms:.1f} ms)" if estimate.elapsed_ms is not None else "unavailable"
    return f"{estimate.details.get('display_key', estimate.key)} in {estimate.elapsed_ms:.1f} ms"


def build_bpm_payload(path: Path, metadata, estimate: TempoEstimate) -> dict[str, object]:
    return {
        "file_path": path.as_posix(),
        "title": metadata.title,
        "artist": metadata.artist,
        "tagged_bpm": metadata.bpm_tag,
        "estimate": estimate.to_payload(),
    }


def prefetch_bpm_and_key_estimates(paths: list[Path], settings) -> tuple[dict[Path, TempoEstimate], dict[Path, KeyEstimate]]:
    if not paths:
        return {}, {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        tempo_future = executor.submit(estimate_tempocnn_bpms, paths)
        key_future = executor.submit(
            estimate_musicalkeycnn_keys,
            paths,
            model_path=settings.analysis.key_model_path,
            device=settings.analysis.key_device,
            policy=settings.analysis.key_policy,
        )
        return tempo_future.result(), key_future.result()


def build_bpm_key_payload(
    path: Path,
    metadata,
    bpm_estimate: TempoEstimate,
    key_estimate: KeyEstimate,
) -> dict[str, object]:
    return {
        "file_path": path.as_posix(),
        "title": metadata.title,
        "artist": metadata.artist,
        "tagged_bpm": metadata.bpm_tag,
        "tagged_key": metadata.key_tag,
        "bpm": bpm_estimate.to_payload(),
        "key": key_estimate.to_payload(),
    }


def build_fast_playlist_bpm_payload(row, path: Path, estimate: TempoEstimate) -> dict[str, object]:
    return {
        "file_path": path.as_posix(),
        "title": row["title"],
        "artist": row["artist"],
        "tagged_bpm": None,
        "estimate": estimate.to_payload(),
    }


def build_fast_playlist_bpm_key_payload(
    row,
    path: Path,
    bpm_estimate: TempoEstimate,
    key_estimate: KeyEstimate,
) -> dict[str, object]:
    return {
        "file_path": path.as_posix(),
        "title": row["title"],
        "artist": row["artist"],
        "tagged_bpm": None,
        "tagged_key": None,
        "bpm": bpm_estimate.to_payload(),
        "key": key_estimate.to_payload(),
    }


def chunk_items(items: list[object], chunk_size: int = TEMPOCNN_PROGRESS_BATCH_SIZE) -> list[list[object]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [items[index:index + chunk_size] for index in range(0, len(items), chunk_size)]


def track_from_playlist_row(row) -> object:
    return read_track_metadata_with_overrides(
        Path(row["file_path"]),
        bpm_imported=row["imported_bpm"],
        key_imported=row["imported_key"],
        title_override=row["title"],
        artist_override=row["artist"],
        genre_override=row["genre"],
        import_source=row["import_source"] or "local_files",
    )


def should_skip_analysis(
    row,
    track,
    *,
    effective_analysis_signature: str,
    config_signature: str,
    analysis_mode: str,
    force: bool,
) -> bool:
    if force:
        return False
    return (
        row["source_file_hash"] == track.file_hash
        and row["analysis_signature"] == effective_analysis_signature
        and row["analysis_mode"] == analysis_mode
        and row["config_signature"] == config_signature
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cuemate-analysis",
        description="CueMate Milestone 1 ingest and absolute-analysis CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import-playlist",
        help="Import a local playlist or crate from files and directories.",
    )
    import_parser.add_argument("--name", required=True, help="Playlist name to create or refresh.")
    import_parser.add_argument(
        "paths",
        nargs="+",
        help="One or more audio files or directories to import.",
    )

    list_dj_playlists_parser = subparsers.add_parser(
        "list-dj-playlists",
        help="List playlists available from a Rekordbox, Traktor, or Serato library export.",
    )
    list_dj_playlists_parser.add_argument(
        "--source",
        required=True,
        choices=["rekordbox", "traktor", "serato"],
        help="DJ library source type.",
    )
    list_dj_playlists_parser.add_argument(
        "--library",
        required=True,
        help="Path to the exported Rekordbox XML, Traktor NML, or Serato crate folder/file.",
    )

    import_dj_playlist_parser = subparsers.add_parser(
        "import-dj-playlist",
        help="Import one playlist from a Rekordbox, Traktor, or Serato library export.",
    )
    import_dj_playlist_parser.add_argument(
        "--source",
        required=True,
        choices=["rekordbox", "traktor", "serato"],
        help="DJ library source type.",
    )
    import_dj_playlist_parser.add_argument(
        "--library",
        required=True,
        help="Path to the exported Rekordbox XML, Traktor NML, or Serato crate folder/file.",
    )
    import_dj_playlist_parser.add_argument(
        "--playlist",
        required=True,
        help="Playlist/crate name inside the source library export.",
    )
    import_dj_playlist_parser.add_argument(
        "--name",
        help="Optional local CueMate playlist name. Defaults to the source playlist name.",
    )

    analyze_parser = subparsers.add_parser(
        "analyze-playlist",
        help="Analyze imported playlist tracks and persist absolute features.",
    )
    analyze_parser.add_argument("--playlist", required=True, help="Playlist name to analyze.")
    analyze_parser.add_argument(
        "--analysis-mode",
        choices=["fast_pass", "full"],
        default="full",
        help="Analysis depth to run.",
    )
    analyze_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-analyze tracks even when the stored analysis signature and file hash match.",
    )

    list_parser = subparsers.add_parser(
        "list-playlist",
        help="List playlist membership and current analysis state.",
    )
    list_parser.add_argument("--name", required=True, help="Playlist name to inspect.")

    show_parser = subparsers.add_parser(
        "show-track",
        help="Show imported metadata and absolute features for a track.",
    )
    show_parser.add_argument("--track-id", required=True, help="Track identifier to inspect.")

    analyze_bpm_parser = subparsers.add_parser(
        "analyze-bpm",
        help="Estimate BPM for one file with the production TempoCNN path.",
    )
    analyze_bpm_parser.add_argument("path", help="Audio file to analyze.")
    analyze_bpm_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the BPM payload as JSON.",
    )

    analyze_bpm_key_parser = subparsers.add_parser(
        "analyze-bpm-key",
        help="Estimate BPM and key for one file with TempoCNN and MusicalKeyCNN only.",
    )
    analyze_bpm_key_parser.add_argument("path", help="Audio file to analyze.")
    analyze_bpm_key_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the BPM+key payload as JSON.",
    )

    analyze_bpm_playlist_parser = subparsers.add_parser(
        "analyze-bpm-playlist",
        help="Estimate BPM for an imported playlist with the production TempoCNN path.",
    )
    analyze_bpm_playlist_parser.add_argument("--playlist", required=True, help="Playlist name to analyze.")
    analyze_bpm_playlist_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional track limit for a faster pass.",
    )
    analyze_bpm_playlist_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the playlist BPM payload as JSON.",
    )
    analyze_bpm_playlist_parser.add_argument(
        "--output",
        help="Optional output path for a CSV report.",
    )

    analyze_bpm_key_playlist_parser = subparsers.add_parser(
        "analyze-bpm-key-playlist",
        help="Estimate BPM and key for an imported playlist with TempoCNN and MusicalKeyCNN only.",
    )
    analyze_bpm_key_playlist_parser.add_argument("--playlist", required=True, help="Playlist name to analyze.")
    analyze_bpm_key_playlist_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional track limit for a faster pass.",
    )
    analyze_bpm_key_playlist_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the playlist BPM+key payload as JSON.",
    )
    analyze_bpm_key_playlist_parser.add_argument(
        "--output",
        help="Optional output path for a CSV report.",
    )

    purge_cache_parser = subparsers.add_parser(
        "purge-model-cache",
        help="Purge persisted TempoCNN and MusicalKeyCNN caches and clear warm service state.",
    )
    purge_cache_parser.add_argument(
        "--backend",
        choices=["all", "tempocnn", "musicalkeycnn"],
        default="all",
        help="Limit the purge to one backend.",
    )
    purge_cache_parser.add_argument(
        "--playlist",
        help="Optional playlist name to scope the purge to imported track file paths.",
    )
    purge_cache_parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Optional file path to purge. Can be passed multiple times.",
    )

    return parser


def handle_import_playlist(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    playlist_id = make_playlist_id(args.name)
    paths = discover_audio_files(args.paths)
    if not paths:
        raise SystemExit("No supported audio files were found in the provided paths.")
    timestamp = utc_now()
    tracks = [read_track_metadata(path) for path in paths]

    with Database(settings.database_path) as database:
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
    settings = load_runtime_settings()
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

    with Database(settings.database_path) as database:
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


def handle_analyze_playlist(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    playlist_name = args.playlist
    effective_analysis_signature = build_effective_analysis_signature(
        settings.analysis_signature,
        tempocnn_accelerator="auto",
        musicalkeycnn_model=settings.analysis.key_model_path,
        musicalkeycnn_device=settings.analysis.key_device,
        musicalkeycnn_policy=settings.analysis.key_policy,
    )

    with Database(settings.database_path) as database:
        rows = database.get_playlist_tracks(playlist_name)
        if not rows:
            raise SystemExit(f"Playlist '{playlist_name}' was not found. Import it first.")

        total = len(rows)
        processed = 0
        for chunk in chunk_items(rows):
            prepared_tracks = []
            for row in chunk:
                index = int(row["position"])
                try:
                    track = track_from_playlist_row(row)
                except Exception as exc:
                    print(f"[{index}/{total}] failed {row['track_id']}: {exc}", file=sys.stderr)
                    continue
                prepared_tracks.append({"row": row, "track": track})

            pending_paths = [
                item["track"].file_path
                for item in prepared_tracks
                if not should_skip_analysis(
                    item["row"],
                    item["track"],
                    effective_analysis_signature=effective_analysis_signature,
                    config_signature=settings.config_signature,
                    analysis_mode=args.analysis_mode,
                    force=args.force,
                )
            ]
            prefetched_tempocnn_estimates: dict[Path, TempoEstimate] = {}
            prefetched_musicalkeycnn_estimates: dict[Path, KeyEstimate] = {}
            if pending_paths:
                prefetched_tempocnn_estimates, prefetched_musicalkeycnn_estimates = prefetch_bpm_and_key_estimates(
                    pending_paths,
                    settings,
                )

            for item in prepared_tracks:
                index = int(item["row"]["position"])
                row = item["row"]
                track = item["track"]
                start_time = time.perf_counter()
                job_id: int | None = None
                try:
                    created_at = utc_now()
                    job_id = database.create_analysis_job(
                        playlist_id=row["playlist_id"],
                        track_id=row["track_id"],
                        track_path=row["file_path"],
                        analysis_mode=args.analysis_mode,
                        analysis_signature=effective_analysis_signature,
                        config_signature=settings.config_signature,
                        source_file_hash=track.file_hash,
                        priority=total - index,
                        created_at=created_at,
                    )
                    database.mark_analysis_job_started(job_id, created_at)
                    database.upsert_track(track, utc_now())
                    if should_skip_analysis(
                        row,
                        track,
                        effective_analysis_signature=effective_analysis_signature,
                        config_signature=settings.config_signature,
                        analysis_mode=args.analysis_mode,
                        force=args.force,
                    ):
                        duration_seconds = round(time.perf_counter() - start_time, 3)
                        database.mark_analysis_job_completed(
                            job_id,
                            duration_seconds,
                            {
                                "analysis_mode": args.analysis_mode,
                                "analysis_signature": effective_analysis_signature,
                                "config_signature": settings.config_signature,
                                "tempo_backend": "tempocnn",
                                "tempocnn_accelerator": "auto",
                                "key_backend": "musicalkeycnn",
                                "musicalkeycnn_model": settings.analysis.key_model_path,
                                "musicalkeycnn_device": settings.analysis.key_device,
                                "musicalkeycnn_policy": settings.analysis.key_policy,
                                "skipped": True,
                            },
                            utc_now(),
                        )
                        print(
                            f"[{index}/{total}] skipped {track.id} ({track.title}) - analysis is already current"
                        )
                        continue

                    result = analyze_track(
                        track,
                        settings,
                        args.analysis_mode,
                        tempo_backend="tempocnn",
                        tempocnn_accelerator="auto",
                        prefetched_tempocnn_estimate=prefetched_tempocnn_estimates.get(track.file_path.resolve()),
                        key_backend="musicalkeycnn",
                        musicalkeycnn_model=settings.analysis.key_model_path,
                        musicalkeycnn_device=settings.analysis.key_device,
                        musicalkeycnn_policy=settings.analysis.key_policy,
                        prefetched_musicalkeycnn_estimate=prefetched_musicalkeycnn_estimates.get(track.file_path.resolve()),
                        analysis_signature=effective_analysis_signature,
                    )
                    database.upsert_track_features(result)
                    duration_seconds = round(time.perf_counter() - start_time, 3)
                    database.mark_analysis_job_completed(
                        job_id,
                        duration_seconds,
                        {
                            "analysis_mode": args.analysis_mode,
                            "analysis_signature": effective_analysis_signature,
                            "config_signature": settings.config_signature,
                            "tempo_backend": "tempocnn",
                            "tempocnn_accelerator": "auto",
                            "key_backend": "musicalkeycnn",
                            "musicalkeycnn_model": settings.analysis.key_model_path,
                            "musicalkeycnn_device": settings.analysis.key_device,
                            "musicalkeycnn_policy": settings.analysis.key_policy,
                        },
                        utc_now(),
                    )
                    processed += 1
                    print(
                        f"[{index}/{total}] analyzed {track.id} ({track.title}) -> "
                        f"{result.bpm:.1f} BPM ({result.bpm_source}), {result.key} ({result.key_source})"
                    )
                except Exception as exc:
                    duration_seconds = round(time.perf_counter() - start_time, 3)
                    if job_id is not None:
                        database.mark_analysis_job_failed(job_id, str(exc), duration_seconds, utc_now())
                    print(f"[{index}/{total}] failed {track.id}: {exc}", file=sys.stderr)

    print(f"Completed playlist analysis for '{playlist_name}'. Updated {processed} track(s).")
    return 0


def handle_list_playlist(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    with Database(settings.database_path) as database:
        rows = database.get_playlist_tracks(args.name)
        if not rows:
            raise SystemExit(f"Playlist '{args.name}' was not found.")

        print(f"Playlist '{args.name}' ({len(rows)} track(s))")
        for row in rows:
            title = row["title"] or Path(row["file_path"]).stem
            artist = row["artist"] or "Unknown artist"
            if row["analyzed_at"]:
                summary = f"{row['analysis_mode']} | {row['bpm']:.1f} BPM | {row['key']}"
            else:
                summary = "not analyzed"
            print(f"{row['position']:02d}. {artist} - {title} [{row['track_id']}] :: {summary}")
    return 0


def handle_show_track(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    with Database(settings.database_path) as database:
        details = database.get_track_details(args.track_id)
        if details is None:
            raise SystemExit(f"Track '{args.track_id}' was not found.")
    print(json.dumps(details, indent=2, sort_keys=True))
    return 0


def handle_analyze_bpm(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file was not found: {path}")

    metadata = read_track_metadata(path)
    estimate = estimate_tempocnn_bpms([path])[path.resolve()]
    payload = build_bpm_payload(path, metadata, estimate)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"File: {payload['file_path']}")
    print(f"Track: {metadata.artist or 'Unknown artist'} - {metadata.title or path.stem}")
    print(f"Tagged BPM: {metadata.bpm_tag if metadata.bpm_tag is not None else 'none'}")
    print("Backend: tempocnn")
    print(f"BPM: {summarize_estimate(estimate)}")
    if estimate.confidence is not None:
        print(f"Confidence: {estimate.confidence:.2f}")
    for note in estimate.notes:
        print(f"- {note}")
    return 0


def handle_analyze_bpm_key(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file was not found: {path}")

    metadata = read_track_metadata(path)
    bpm_estimate = estimate_tempocnn_bpms([path])[path.resolve()]
    key_estimate = estimate_musicalkeycnn_keys([path])[path.resolve()]
    payload = build_bpm_key_payload(path, metadata, bpm_estimate, key_estimate)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"File: {payload['file_path']}")
    print(f"Track: {metadata.artist or 'Unknown artist'} - {metadata.title or path.stem}")
    print(f"BPM: {summarize_estimate(bpm_estimate)}")
    if bpm_estimate.confidence is not None:
        print(f"BPM Confidence: {bpm_estimate.confidence:.2f}")
    print(f"Key: {summarize_key_estimate(key_estimate)}")
    if key_estimate.confidence is not None:
        print(f"Key Confidence: {key_estimate.confidence:.2f}")
    return 0


def handle_analyze_bpm_playlist(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    with Database(settings.database_path) as database:
        rows = database.get_playlist_tracks(args.playlist)
        if not rows:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    payload_rows: list[dict[str, object]] = []
    total = len(rows)
    if not args.json:
        print(f"Playlist '{args.playlist}' BPM pass with backend tempocnn")

    for chunk in chunk_items(rows):
        prefetched_tempocnn_estimates = estimate_tempocnn_bpms([Path(row["file_path"]) for row in chunk])

        for row in chunk:
            index = int(row["position"])
            path = Path(row["file_path"]).resolve()
            estimate = prefetched_tempocnn_estimates[path]
            payload = build_fast_playlist_bpm_payload(row, path, estimate)
            payload["track_id"] = row["track_id"]
            payload["position"] = index
            payload_rows.append(payload)
            if not args.json:
                title = payload["title"] or Path(str(payload["file_path"])).stem
                print(
                    f"[{index}/{total}] {title} [{payload['track_id']}] :: "
                    f"{summarize_estimate(estimate)}"
                )

    if args.json:
        print(
            json.dumps(
                {
                    "playlist": args.playlist,
                    "backend": "tempocnn",
                    "tracks": payload_rows,
                },
                indent=2,
                sort_keys=True,
            )
        )
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows_for_csv: list[dict[str, object]] = []
        for item in payload_rows:
            estimate = TempoEstimate(**item["estimate"])
            rows_for_csv.append(
                {
                    "position": item["position"],
                    "track_id": item["track_id"],
                    "title": item["title"],
                    "artist": item["artist"],
                    "file_path": item["file_path"],
                    "tagged_bpm": item["tagged_bpm"],
                    "backend": "tempocnn",
                    "available": estimate.available,
                    "bpm": estimate.bpm,
                    "confidence": estimate.confidence,
                    "elapsed_ms": estimate.elapsed_ms,
                    "notes": " | ".join(estimate.notes),
                }
            )
        fieldnames = list(rows_for_csv[0].keys()) if rows_for_csv else [
            "position", "track_id", "title", "artist", "file_path",
            "tagged_bpm", "backend", "available", "bpm", "confidence", "elapsed_ms", "notes",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_for_csv)
        if not args.json:
            print(f"Wrote CSV report to {output_path}")
    return 0


def handle_analyze_bpm_key_playlist(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    with Database(settings.database_path) as database:
        rows = database.get_playlist_tracks(args.playlist)
        if not rows:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    payload_rows: list[dict[str, object]] = []
    total = len(rows)
    if not args.json:
        print(f"Playlist '{args.playlist}' BPM+key pass with TempoCNN + MusicalKeyCNN")

    for chunk in chunk_items(rows):
        chunk_paths = [Path(row["file_path"]) for row in chunk]
        prefetched_tempocnn_estimates, prefetched_musicalkeycnn_estimates = prefetch_bpm_and_key_estimates(
            chunk_paths,
            settings,
        )

        for row in chunk:
            index = int(row["position"])
            path = Path(row["file_path"]).resolve()
            bpm_estimate = prefetched_tempocnn_estimates[path]
            key_estimate = prefetched_musicalkeycnn_estimates[path]
            payload = build_fast_playlist_bpm_key_payload(row, path, bpm_estimate, key_estimate)
            payload["track_id"] = row["track_id"]
            payload["position"] = index
            payload_rows.append(payload)
            if not args.json:
                title = payload["title"] or Path(str(payload["file_path"])).stem
                print(
                    f"[{index}/{total}] {title} [{payload['track_id']}] :: "
                    f"{summarize_estimate(bpm_estimate)} :: {summarize_key_estimate(key_estimate)}"
                )

    if args.json:
        print(
            json.dumps(
                {
                    "playlist": args.playlist,
                    "tracks": payload_rows,
                },
                indent=2,
                sort_keys=True,
            )
        )
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows_for_csv: list[dict[str, object]] = []
        for item in payload_rows:
            bpm_estimate = TempoEstimate(**item["bpm"])
            key_estimate = KeyEstimate(**item["key"])
            rows_for_csv.append(
                {
                    "position": item["position"],
                    "track_id": item["track_id"],
                    "title": item["title"],
                    "artist": item["artist"],
                    "file_path": item["file_path"],
                    "bpm": bpm_estimate.bpm,
                    "bpm_confidence": bpm_estimate.confidence,
                    "bpm_elapsed_ms": bpm_estimate.elapsed_ms,
                    "key": key_estimate.key,
                    "key_confidence": key_estimate.confidence,
                    "key_elapsed_ms": key_estimate.elapsed_ms,
                }
            )
        fieldnames = list(rows_for_csv[0].keys()) if rows_for_csv else [
            "position", "track_id", "title", "artist", "file_path",
            "bpm", "bpm_confidence", "bpm_elapsed_ms", "key", "key_confidence", "key_elapsed_ms",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_for_csv)
        if not args.json:
            print(f"Wrote CSV report to {output_path}")
    return 0


def handle_purge_model_cache(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    scoped_paths: list[str] = []

    if args.playlist:
        with Database(settings.database_path) as database:
            rows = database.get_playlist_tracks(args.playlist)
            if not rows:
                raise SystemExit(f"Playlist '{args.playlist}' was not found.")
            scoped_paths.extend([str(Path(row["file_path"]).resolve()) for row in rows])

    if args.path:
        scoped_paths.extend([str(Path(path).expanduser().resolve()) for path in args.path])

    deduped_paths = sorted({str(Path(path).resolve()) for path in scoped_paths})
    deleted_tempocnn = 0
    deleted_musicalkeycnn = 0

    if args.backend in {"all", "tempocnn"}:
        deleted_tempocnn = purge_tempocnn_cache(file_paths=deduped_paths or None)
    if args.backend in {"all", "musicalkeycnn"}:
        deleted_musicalkeycnn = purge_musicalkeycnn_cache(file_paths=deduped_paths or None)

    scope_label = "all cached tracks"
    if deduped_paths:
        scope_label = f"{len(deduped_paths)} scoped file(s)"

    print(f"Purged model inference cache for {scope_label}.")
    if args.backend in {"all", "tempocnn"}:
        print(f"- TempoCNN rows removed: {deleted_tempocnn}")
        print("- TempoCNN warm service state cleared")
    if args.backend in {"all", "musicalkeycnn"}:
        print(f"- MusicalKeyCNN rows removed: {deleted_musicalkeycnn}")
        print("- MusicalKeyCNN warm service state cleared")
    print(f"- Persistent cache DB: {resolve_inference_cache_path()}")
    print("- Re-run analysis with --force if you want stored playlist analysis rows refreshed too.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "import-playlist":
            return handle_import_playlist(args)
        if args.command == "list-dj-playlists":
            return handle_list_dj_playlists(args)
        if args.command == "import-dj-playlist":
            return handle_import_dj_playlist(args)
        if args.command == "analyze-playlist":
            return handle_analyze_playlist(args)
        if args.command == "list-playlist":
            return handle_list_playlist(args)
        if args.command == "show-track":
            return handle_show_track(args)
        if args.command == "analyze-bpm":
            return handle_analyze_bpm(args)
        if args.command == "analyze-bpm-key":
            return handle_analyze_bpm_key(args)
        if args.command == "analyze-bpm-playlist":
            return handle_analyze_bpm_playlist(args)
        if args.command == "analyze-bpm-key-playlist":
            return handle_analyze_bpm_key_playlist(args)
        if args.command == "purge-model-cache":
            return handle_purge_model_cache(args)
    except sqlite3.OperationalError as exc:
        message = str(exc)
        if "no such table" in message.lower():
            print(
                "The local database schema is not ready. Apply migrations first with "
                "powershell -ExecutionPolicy Bypass -File .\\scripts\\docker-compose.ps1 --profile ops run --rm migrate",
                file=sys.stderr,
            )
        else:
            print(message, file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        print(str(exc), file=sys.stderr)
        return 1

    parser.print_help()
    return 1
