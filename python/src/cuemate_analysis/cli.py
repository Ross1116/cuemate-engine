from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sqlite3
import sys
import time

from cuemate_analysis.analysis import analyze_track, utc_now
from cuemate_analysis.config import load_runtime_settings
from cuemate_analysis.database import Database
from cuemate_analysis.ingest import discover_audio_files, make_playlist_id, read_track_metadata
from cuemate_analysis.key_backend import (
    MUSICALKEYCNN_POLICY_FULL_TRACK,
    estimate_musicalkeycnn_keys,
    resolve_musicalkeycnn_model_path,
)
from cuemate_analysis.tempo_backend import (
    TempoEstimate,
    estimate_tempocnn_bpms,
    resolve_tempocnn_model_path,
)

TEMPOCNN_PROGRESS_BATCH_SIZE = 8


def build_effective_analysis_signature(
    base_signature: str,
    *,
    tempocnn_model: str | None = None,
    tempocnn_accelerator: str = "auto",
    musicalkeycnn_model: str | None = None,
    musicalkeycnn_device: str = "auto",
    musicalkeycnn_policy: str = MUSICALKEYCNN_POLICY_FULL_TRACK,
) -> str:
    tempo_model_name = resolve_tempocnn_model_path(tempocnn_model).stem
    key_model_name = resolve_musicalkeycnn_model_path(musicalkeycnn_model).stem
    return (
        f"{base_signature}"
        f"-tempo-tempocnn-{tempo_model_name}-{tempocnn_accelerator}"
        f"-key-musicalkeycnn-{key_model_name}-{musicalkeycnn_device}-{musicalkeycnn_policy}"
    )


def summarize_estimate(estimate: TempoEstimate) -> str:
    if not estimate.available or estimate.bpm is None:
        return f"unavailable ({estimate.elapsed_ms:.1f} ms)" if estimate.elapsed_ms is not None else "unavailable"
    return (
        f"{estimate.details.get('display_bpm', round(estimate.bpm, 1))} BPM "
        f"in {estimate.elapsed_ms:.1f} ms"
    )


def build_bpm_payload(path: Path, metadata, estimate: TempoEstimate) -> dict[str, object]:
    return {
        "file_path": path.as_posix(),
        "title": metadata.title,
        "artist": metadata.artist,
        "tagged_bpm": metadata.bpm_tag,
        "estimate": estimate.to_payload(),
    }


def chunk_items(items: list[object], chunk_size: int = TEMPOCNN_PROGRESS_BATCH_SIZE) -> list[list[object]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [items[index:index + chunk_size] for index in range(0, len(items), chunk_size)]


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

        prepared_tracks = [
            {
                "row": row,
                "track": read_track_metadata(Path(row["file_path"])),
            }
            for row in rows
        ]
        total = len(rows)
        processed = 0
        for chunk in chunk_items(prepared_tracks):
            pending_paths = [
                item["track"].file_path
                for item in chunk
                if args.force
                or item["row"]["source_file_hash"] != item["track"].file_hash
                or item["row"]["analysis_signature"] != effective_analysis_signature
                or item["row"]["analysis_mode"] != args.analysis_mode
            ]
            prefetched_tempocnn_estimates: dict[Path, TempoEstimate] = {}
            prefetched_musicalkeycnn_estimates = {}
            if pending_paths:
                prefetched_tempocnn_estimates = estimate_tempocnn_bpms(pending_paths)
                prefetched_musicalkeycnn_estimates = estimate_musicalkeycnn_keys(
                    pending_paths,
                    model_path=settings.analysis.key_model_path,
                    device=settings.analysis.key_device,
                    policy=settings.analysis.key_policy,
                )

            for item in chunk:
                index = int(item["row"]["position"])
                row = item["row"]
                track = item["track"]
                created_at = utc_now()
                job_id = database.create_analysis_job(
                    playlist_id=row["playlist_id"],
                    track_id=row["track_id"],
                    track_path=row["file_path"],
                    analysis_mode=args.analysis_mode,
                    analysis_signature=effective_analysis_signature,
                    config_signature=settings.config_signature,
                    source_file_hash=row["file_hash"],
                    priority=total - index,
                    created_at=created_at,
                )
                start_time = time.perf_counter()
                database.mark_analysis_job_started(job_id, created_at)
                try:
                    database.upsert_track(track, utc_now())
                    if (
                        not args.force
                        and row["source_file_hash"] == track.file_hash
                        and row["analysis_signature"] == effective_analysis_signature
                        and row["analysis_mode"] == args.analysis_mode
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
            metadata = read_track_metadata(path)
            estimate = prefetched_tempocnn_estimates[path]
            payload = build_bpm_payload(path, metadata, estimate)
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "import-playlist":
            return handle_import_playlist(args)
        if args.command == "analyze-playlist":
            return handle_analyze_playlist(args)
        if args.command == "list-playlist":
            return handle_list_playlist(args)
        if args.command == "show-track":
            return handle_show_track(args)
        if args.command == "analyze-bpm":
            return handle_analyze_bpm(args)
        if args.command == "analyze-bpm-playlist":
            return handle_analyze_bpm_playlist(args)
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
