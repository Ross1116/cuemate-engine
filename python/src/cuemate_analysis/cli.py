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
from cuemate_analysis.tempo_experiments import (
    TempoEstimate,
    estimate_baseline_bpm,
    estimate_beatnet_bpm,
    estimate_essentia_wsl_bpm,
)


def build_effective_analysis_signature(
    base_signature: str,
    *,
    tempo_backend: str,
    beatnet_model: int,
) -> str:
    if tempo_backend == "baseline":
        return base_signature
    return f"{base_signature}-tempo-{tempo_backend}-m{beatnet_model}"


def parse_backend_list(raw: str) -> list[str]:
    backends = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"baseline", "beatnet", "essentia_wsl"}
    invalid = [item for item in backends if item not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported backends: {', '.join(invalid)}")
    if not backends:
        raise argparse.ArgumentTypeError("At least one backend must be selected.")
    return backends


def summarize_estimate(estimate: TempoEstimate) -> str:
    if not estimate.available or estimate.bpm is None:
        return f"unavailable ({estimate.elapsed_ms:.1f} ms)" if estimate.elapsed_ms is not None else "unavailable"
    return (
        f"{estimate.details.get('display_bpm', round(estimate.bpm, 1))} BPM "
        f"in {estimate.elapsed_ms:.1f} ms"
    )


def run_selected_backend(
    backend: str,
    path: Path,
    settings,
    *,
    beatnet_model: int,
) -> TempoEstimate:
    if backend == "baseline":
        return estimate_baseline_bpm(path, settings)
    if backend == "beatnet":
        return estimate_beatnet_bpm(path, model=beatnet_model)
    if backend == "essentia_wsl":
        return estimate_essentia_wsl_bpm(path)
    raise ValueError(f"Unsupported backend: {backend}")


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
    analyze_parser.add_argument(
        "--tempo-backend",
        choices=["baseline", "beatnet"],
        default="baseline",
        help="Tempo detector backend to use for BPM estimation.",
    )
    analyze_parser.add_argument(
        "--beatnet-model",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="BeatNet model to use when --tempo-backend beatnet is selected.",
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

    compare_parser = subparsers.add_parser(
        "compare-bpm",
        help="Compare the current BPM detector against experimental backends on one file.",
    )
    compare_parser.add_argument("path", help="Audio file to analyze.")
    compare_parser.add_argument(
        "--beatnet-model",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="BeatNet model to use when BeatNet is available.",
    )
    compare_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the comparison payload as JSON.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark-bpm",
        help="Benchmark tempo backends across an imported playlist.",
    )
    benchmark_parser.add_argument("--playlist", required=True, help="Playlist name to benchmark.")
    benchmark_parser.add_argument(
        "--backends",
        type=parse_backend_list,
        default=["baseline", "essentia_wsl"],
        help="Comma-separated backend list. Default: baseline,essentia_wsl",
    )
    benchmark_parser.add_argument(
        "--beatnet-model",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="BeatNet model to use when beatnet is included.",
    )
    benchmark_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional track limit for a faster benchmark pass.",
    )
    benchmark_parser.add_argument(
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
        tempo_backend=args.tempo_backend,
        beatnet_model=args.beatnet_model,
    )

    with Database(settings.database_path) as database:
        rows = database.get_playlist_tracks(playlist_name)
        if not rows:
            raise SystemExit(f"Playlist '{playlist_name}' was not found. Import it first.")

        total = len(rows)
        processed = 0
        for index, row in enumerate(rows, start=1):
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
                track = read_track_metadata(Path(row["file_path"]))
                database.upsert_track(track, utc_now())
                existing = database.get_existing_analysis(track.id)
                if (
                    not args.force
                    and existing is not None
                    and existing["source_file_hash"] == track.file_hash
                    and existing["analysis_signature"] == effective_analysis_signature
                    and existing["analysis_mode"] == args.analysis_mode
                ):
                    duration_seconds = round(time.perf_counter() - start_time, 3)
                    database.mark_analysis_job_completed(
                        job_id,
                        duration_seconds,
                        {
                            "analysis_mode": args.analysis_mode,
                            "analysis_signature": effective_analysis_signature,
                            "config_signature": settings.config_signature,
                            "tempo_backend": args.tempo_backend,
                            "beatnet_model": args.beatnet_model,
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
                    tempo_backend=args.tempo_backend,
                    beatnet_model=args.beatnet_model,
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
                        "tempo_backend": args.tempo_backend,
                        "beatnet_model": args.beatnet_model,
                    },
                    utc_now(),
                )
                processed += 1
                print(
                    f"[{index}/{total}] analyzed {track.id} ({track.title}) -> "
                    f"{result.bpm:.1f} BPM ({result.bpm_source}), {result.key}"
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


def handle_compare_bpm(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    path = Path(args.path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file was not found: {path}")

    metadata = read_track_metadata(path)
    baseline = estimate_baseline_bpm(path, settings)
    beatnet = estimate_beatnet_bpm(path, model=args.beatnet_model)
    essentia_wsl = estimate_essentia_wsl_bpm(path)

    payload = {
        "file_path": path.as_posix(),
        "title": metadata.title,
        "artist": metadata.artist,
        "tagged_bpm": metadata.bpm_tag,
        "baseline": baseline.to_payload(),
        "beatnet": beatnet.to_payload(),
        "essentia_wsl": essentia_wsl.to_payload(),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"File: {payload['file_path']}")
    print(f"Track: {metadata.artist or 'Unknown artist'} - {metadata.title or path.stem}")
    print(f"Tagged BPM: {metadata.bpm_tag if metadata.bpm_tag is not None else 'none'}")
    print(f"Baseline: {summarize_estimate(baseline)} (confidence {baseline.confidence:.2f})")
    if beatnet.available and beatnet.bpm is not None and beatnet.confidence is not None:
        print(
            f"BeatNet: {summarize_estimate(beatnet)} "
            f"(confidence {beatnet.confidence:.2f}, beats {beatnet.details['beat_count']})"
        )
    else:
        print(f"BeatNet: {summarize_estimate(beatnet)}")
    if essentia_wsl.available and essentia_wsl.bpm is not None and essentia_wsl.confidence is not None:
        print(
            f"Essentia WSL: {summarize_estimate(essentia_wsl)} "
            f"(score {essentia_wsl.confidence:.2f}, "
            f"key {essentia_wsl.details['key']} {essentia_wsl.details['scale']})"
        )
    else:
        print(f"Essentia WSL: {summarize_estimate(essentia_wsl)}")

    for note in beatnet.notes:
        print(f"- {note}")
    for note in essentia_wsl.notes:
        print(f"- {note}")
    return 0


def handle_benchmark_bpm(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    with Database(settings.database_path) as database:
        rows = database.get_playlist_tracks(args.playlist)
        if not rows:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    report_rows: list[dict[str, object]] = []
    print(
        f"Benchmarking {len(rows)} track(s) from '{args.playlist}' with backends: {', '.join(args.backends)}"
    )

    for index, row in enumerate(rows, start=1):
        path = Path(row["file_path"])
        metadata = read_track_metadata(path)
        line_parts = [f"[{index}/{len(rows)}] {metadata.title or path.stem}"]
        row_payload: dict[str, object] = {
            "track_id": row["track_id"],
            "title": metadata.title,
            "artist": metadata.artist,
            "file_path": path.as_posix(),
            "tagged_bpm": metadata.bpm_tag,
        }

        for backend in args.backends:
            estimate = run_selected_backend(
                backend,
                path,
                settings,
                beatnet_model=args.beatnet_model,
            )
            row_payload[f"{backend}_available"] = estimate.available
            row_payload[f"{backend}_bpm"] = estimate.bpm
            row_payload[f"{backend}_confidence"] = estimate.confidence
            row_payload[f"{backend}_elapsed_ms"] = estimate.elapsed_ms
            row_payload[f"{backend}_notes"] = " | ".join(estimate.notes)
            if backend == "essentia_wsl":
                row_payload[f"{backend}_key"] = estimate.details.get("key")
                row_payload[f"{backend}_scale"] = estimate.details.get("scale")
                row_payload[f"{backend}_key_strength"] = estimate.details.get("key_strength")
            line_parts.append(f"{backend}={summarize_estimate(estimate)}")

        report_rows.append(row_payload)
        print(" :: ".join(line_parts))

    summary: list[str] = []
    for backend in args.backends:
        successful = [
            row for row in report_rows if row.get(f"{backend}_available") and row.get(f"{backend}_elapsed_ms") is not None
        ]
        if not successful:
            summary.append(f"{backend}: no successful runs")
            continue
        avg_elapsed = sum(float(row[f"{backend}_elapsed_ms"]) for row in successful) / len(successful)
        avg_bpm = sum(float(row[f"{backend}_bpm"]) for row in successful if row.get(f"{backend}_bpm") is not None) / len(successful)
        summary.append(
            f"{backend}: {len(successful)}/{len(report_rows)} ok, avg {avg_bpm:.1f} BPM, avg {avg_elapsed:.1f} ms"
        )

    print("Summary:")
    for line in summary:
        print(f"- {line}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = sorted({key for row in report_rows for key in row.keys()})
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report_rows)
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
        if args.command == "compare-bpm":
            return handle_compare_bpm(args)
        if args.command == "benchmark-bpm":
            return handle_benchmark_bpm(args)
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
