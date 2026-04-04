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
from cuemate_analysis.energy_experiments import analyze_energy_path
from cuemate_analysis.energy_model import (
    EnergyFeatureVector,
    FEATURE_NAMES,
    energy_consensus,
    evaluate_energy_model,
    load_energy_dataset_rows,
    predict_energy_from_features,
    resolve_energy_model_meta_path,
    resolve_energy_model_path,
    save_energy_bundle,
    train_energy_bundle,
)
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
from cuemate_analysis.relative_context import (
    compute_relative_playlist_preview,
    preview_to_json,
    row_to_relative_track_input,
)
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


def resolve_expected_energy_model_signature(settings) -> str:
    if not settings.analysis.energy_parallel_enabled:
        return "disabled"
    model_path = resolve_energy_model_path(settings.analysis.energy_model_path, settings.repo_root)
    meta_path = resolve_energy_model_meta_path(settings.analysis.energy_model_meta_path, settings.repo_root)
    if not model_path.is_file() or not meta_path.is_file():
        return "missing"
    return f"{hash_file_identity(model_path)}-{hash_file_identity(meta_path)}"


def build_effective_analysis_signature(
    base_signature: str,
    *,
    tempocnn_model: str | None = None,
    tempocnn_accelerator: str = "auto",
    musicalkeycnn_model: str | None = None,
    musicalkeycnn_device: str = "auto",
    musicalkeycnn_policy: str = MUSICALKEYCNN_POLICY_FULL_TRACK,
    energy_model_signature: str = "missing",
) -> str:
    tempo_model_hash = hash_file_identity(resolve_tempocnn_model_path(tempocnn_model))
    key_model_hash = hash_file_identity(resolve_musicalkeycnn_model_path(musicalkeycnn_model))
    return (
        f"{base_signature}"
        f"-tempo-tempocnn-{tempo_model_hash}-{tempocnn_accelerator}"
        f"-key-musicalkeycnn-{key_model_hash}-{musicalkeycnn_device}-{musicalkeycnn_policy}"
        f"-energy-{energy_model_signature}"
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


def build_energy_dataset_row(
    *,
    row,
    features: EnergyFeatureVector,
    playlist_name: str,
) -> dict[str, object]:
    return {
        "track_id": row["track_id"],
        "file_hash": row["file_hash"],
        "file_path": str(Path(row["file_path"]).resolve().as_posix()),
        "title": row["title"] or "",
        "artist": row["artist"] or "",
        "playlist_name": playlist_name,
        "position": int(row["position"]),
        "stored_energy_abs": row["energy_abs"],
        **features.to_payload(),
        "teacher_energy": "",
        "teacher_source": "",
        "teacher_confidence": "",
        "manual_bucket": "",
        "manual_score": "",
        "manual_notes": "",
    }


def load_energy_label_lookup(dataset_path: Path | None) -> dict[str, dict[str, str]]:
    if dataset_path is None:
        return {}
    resolved = dataset_path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            (row.get("track_id") or "").strip(): row
            for row in reader
            if (row.get("track_id") or "").strip()
        }


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
    expected_energy_model_signature: str,
) -> bool:
    if force:
        return False
    energy_current = True
    if analysis_mode == "full" and expected_energy_model_signature not in {"disabled", "missing"}:
        energy_current = row["energy_model_signature"] == expected_energy_model_signature and row["energy_learned"] is not None
    return (
        row["source_file_hash"] == track.file_hash
        and row["analysis_signature"] == effective_analysis_signature
        and row["analysis_mode"] == analysis_mode
        and row["config_signature"] == config_signature
        and energy_current
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

    analyze_relative_playlist_parser = subparsers.add_parser(
        "analyze-relative-playlist",
        help="Compute experimental playlist-relative context from persisted absolute features.",
    )
    analyze_relative_playlist_parser.add_argument("--playlist", required=True, help="Playlist name to analyze.")
    analyze_relative_playlist_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional track limit for a faster experimental subset pass.",
    )
    analyze_relative_playlist_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the relative-context payload as JSON.",
    )
    analyze_relative_playlist_parser.add_argument(
        "--output",
        help="Optional output path for a CSV report.",
    )
    analyze_relative_playlist_parser.add_argument(
        "--energy-source",
        choices=["heuristic", "learned"],
        default=None,
        help="Choose which absolute-energy source to use for relative scaling.",
    )

    analyze_energy_playlist_parser = subparsers.add_parser(
        "analyze-energy-playlist",
        help="Compare experimental absolute-energy candidates for a playlist without persisting results.",
    )
    analyze_energy_playlist_parser.add_argument("--playlist", required=True, help="Playlist name to analyze.")
    analyze_energy_playlist_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional track limit for a faster experimental subset pass.",
    )
    analyze_energy_playlist_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the energy-workbench payload as JSON.",
    )
    analyze_energy_playlist_parser.add_argument(
        "--output",
        help="Optional output path for a CSV report.",
    )
    analyze_energy_playlist_parser.add_argument(
        "--dataset",
        help="Optional labeled dataset CSV to merge teacher/manual labels into the report.",
    )

    export_energy_dataset_parser = subparsers.add_parser(
        "export-energy-dataset",
        help="Export an energy training dataset for an imported playlist.",
    )
    export_energy_dataset_parser.add_argument("--playlist", required=True, help="Playlist name to export.")
    export_energy_dataset_parser.add_argument("--output", required=True, help="CSV path to write.")
    export_energy_dataset_parser.add_argument("--limit", type=int, default=0, help="Optional track limit.")

    train_energy_model_parser = subparsers.add_parser(
        "train-energy-model",
        help="Train the teacher-first energy model from a labeled CSV dataset.",
    )
    train_energy_model_parser.add_argument("--dataset", required=True, help="Labeled dataset CSV path.")
    train_energy_model_parser.add_argument("--model-out", required=True, help="Joblib artifact output path.")
    train_energy_model_parser.add_argument("--meta-out", required=True, help="JSON metadata output path.")

    benchmark_energy_model_parser = subparsers.add_parser(
        "benchmark-energy-model",
        help="Benchmark baseline, hybrid, and learned energy scorers from a labeled dataset.",
    )
    benchmark_energy_model_parser.add_argument("--dataset", required=True, help="Labeled dataset CSV path.")
    benchmark_energy_model_parser.add_argument("--json", action="store_true", help="Emit benchmark JSON.")
    benchmark_energy_model_parser.add_argument("--output", help="Optional CSV output path.")
    benchmark_energy_model_parser.add_argument("--model-path", help="Optional learned model override path.")
    benchmark_energy_model_parser.add_argument("--meta-path", help="Optional learned model metadata override path.")

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
    expected_energy_model_signature = resolve_expected_energy_model_signature(settings)
    effective_analysis_signature = build_effective_analysis_signature(
        settings.analysis_signature,
        tempocnn_accelerator="auto",
        musicalkeycnn_model=settings.analysis.key_model_path,
        musicalkeycnn_device=settings.analysis.key_device,
        musicalkeycnn_policy=settings.analysis.key_policy,
        energy_model_signature=expected_energy_model_signature,
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
                    expected_energy_model_signature=expected_energy_model_signature,
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
                        expected_energy_model_signature=expected_energy_model_signature,
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
                                "energy_parallel_enabled": settings.analysis.energy_parallel_enabled,
                                "energy_model_signature": expected_energy_model_signature,
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
                            "energy_parallel_enabled": settings.analysis.energy_parallel_enabled,
                            "energy_model_signature": expected_energy_model_signature,
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
    details["energy_summary"] = {
        "heuristic": details.get("energy_abs"),
        "hybrid": details.get("energy_hybrid"),
        "learned": details.get("energy_learned"),
        "learned_bucket": details.get("energy_learned_bucket"),
        "model_signature": details.get("energy_model_signature"),
        "model_source": details.get("energy_model_source"),
        "model_inferred_at": details.get("energy_model_inferred_at"),
    }
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
    settings = load_runtime_settings()
    path = Path(args.path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file was not found: {path}")

    metadata = read_track_metadata(path)
    bpm_estimate = estimate_tempocnn_bpms([path])[path.resolve()]
    key_estimate = estimate_musicalkeycnn_keys(
        [path],
        model_path=settings.analysis.key_model_path,
        device=settings.analysis.key_device,
        policy=settings.analysis.key_policy,
    )[path.resolve()]
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
            estimate = TempoEstimate.from_payload(item["estimate"])
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
            bpm_estimate = TempoEstimate.from_payload(item["bpm"])
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


def handle_analyze_relative_playlist(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    with Database(settings.database_path) as database:
        rows = database.get_playlist_relative_inputs(args.playlist)
        if not rows:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    energy_source = args.energy_source or settings.analysis.energy_source_default

    preview = compute_relative_playlist_preview(
        [row_to_relative_track_input(row) for row in rows],
        settings,
        playlist_name=args.playlist,
        is_limited=bool(args.limit and args.limit > 0),
        energy_source=energy_source,
    )

    if args.json:
        print(preview_to_json(preview))
    else:
        stats = preview.playlist_stats
        print(
            f"Playlist '{args.playlist}' relative-context preview :: "
            f"status={stats.status} :: analyzed={stats.track_count_analyzed}/{stats.track_count_total} :: "
            f"eligible={stats.eligible_track_count}"
        )
        print("Relative scores are computed against the eligible analyzed tracks in this playlist slice.")
        print(f"Energy source: {energy_source}")
        print(f"Relative signature: {stats.relative_signature}")
        if stats.adapted_weights is not None and stats.adaptation_strength is not None:
            print(f"Adapted weights enabled :: adaptation_strength={stats.adaptation_strength:.2f}")
        else:
            print("Adapted weights skipped")
        for note in stats.weight_adaptation_notes:
            print(f"- {note}")
        if energy_source == "learned":
            fallback_count = sum(1 for track in preview.tracks if track.energy_source_used != "learned")
            if fallback_count:
                print(f"- learned energy unavailable for {fallback_count} track(s); heuristic fallback was used per-track")
        for track in preview.tracks:
            title = track.title or Path(track.file_path).stem
            print(
                f"[{track.position}/{stats.track_count_total}] {title} :: "
                f"energy_rel={track.energy_rel:.3f} ({track.energy_source_used}) :: {track.intensity_band} :: "
                f"{', '.join(track.role_hints)}"
            )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows_for_csv: list[dict[str, object]] = []
        for track in preview.tracks:
            rows_for_csv.append(
                {
                    "position": track.position,
                    "track_id": track.track_id,
                    "playlist_id": track.playlist_id,
                    "title": track.title,
                    "artist": track.artist,
                    "file_path": track.file_path,
                    "energy_source_used": track.energy_source_used,
                    "energy_rel": track.energy_rel,
                    "bass_rel": track.bass_rel,
                    "drums_rel": track.drums_rel,
                    "vocals_rel": track.vocals_rel,
                    "groove_rel": track.groove_rel,
                    "energy_spread": track.energy_spread,
                    "bass_spread": track.bass_spread,
                    "drums_spread": track.drums_spread,
                    "vocals_spread": track.vocals_spread,
                    "groove_spread": track.groove_spread,
                    "intensity_band": track.intensity_band,
                    "intensity_membership": json.dumps(track.intensity_membership, sort_keys=True),
                    "role_hints": json.dumps(track.role_hints),
                    "valid_as_of_track_count": track.valid_as_of_track_count,
                    "analysis_signature": track.analysis_signature,
                    "config_signature": track.config_signature,
                }
            )
        fieldnames = list(rows_for_csv[0].keys()) if rows_for_csv else [
            "position", "track_id", "playlist_id", "title", "artist", "file_path",
            "energy_source_used", "energy_rel", "bass_rel", "drums_rel", "vocals_rel", "groove_rel",
            "energy_spread", "bass_spread", "drums_spread", "vocals_spread", "groove_spread",
            "intensity_band", "intensity_membership", "role_hints", "valid_as_of_track_count",
            "analysis_signature", "config_signature",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_for_csv)
        if not args.json:
            print(f"Wrote CSV report to {output_path}")
    return 0


def handle_analyze_energy_playlist(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    with Database(settings.database_path) as database:
        rows = database.get_playlist_relative_inputs(args.playlist)
        if not rows:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    payload_rows: list[dict[str, object]] = []
    total = len(rows)
    label_lookup = load_energy_label_lookup(Path(args.dataset)) if args.dataset else {}
    model_path = resolve_energy_model_path(settings.analysis.energy_model_path, settings.repo_root)
    meta_path = resolve_energy_model_meta_path(settings.analysis.energy_model_meta_path, settings.repo_root)
    model_available = model_path.is_file() and meta_path.is_file()

    if not args.json:
        print(f"Playlist '{args.playlist}' energy workbench :: experimental absolute-energy candidates")

    for row in rows:
        index = int(row["position"])
        path = Path(row["file_path"]).resolve()
        candidates = analyze_energy_path(
            path,
            sample_rate=settings.analysis.sample_rate,
            mono=settings.analysis.mono,
        )
        features = EnergyFeatureVector(
            baseline=candidates.baseline,
            loudness_fusion=candidates.loudness_fusion,
            club_fusion=candidates.club_fusion,
            pressure_fusion=candidates.pressure_fusion,
            energy_sustained=float(candidates.energy_sustained if candidates.energy_sustained is not None else 0.5),
            energy_peak=float(candidates.energy_peak if candidates.energy_peak is not None else 0.5),
            loudness_norm=candidates.loudness_norm,
            loudness_lufs=candidates.loudness_lufs,
            bass_abs=candidates.bass_abs,
            drums_abs=float(candidates.drums_abs if candidates.drums_abs is not None else 0.5),
            harmonic_abs=float(candidates.harmonic_abs if candidates.harmonic_abs is not None else 0.5),
            groove_abs=float(candidates.groove_abs if candidates.groove_abs is not None else 0.5),
        )
        hybrid_blended: float | None = None
        learned: float | None = None
        learned_bucket: str | None = None
        if model_available:
            inference = predict_energy_from_features(
                features,
                model_path=model_path,
                meta_path=meta_path,
            )
            hybrid_blended = inference.hybrid
            learned = inference.learned
            learned_bucket = inference.bucket
        payload = {
            "position": index,
            "track_id": row["track_id"],
            "title": row["title"],
            "artist": row["artist"],
            "file_path": path.as_posix(),
            "stored_energy_abs": row["energy_abs"],
            "baseline": candidates.baseline,
            "loudness_fusion": candidates.loudness_fusion,
            "club_fusion": candidates.club_fusion,
            "pressure_fusion": candidates.pressure_fusion,
            "hybrid_blended": hybrid_blended,
            "learned": learned,
            "learned_bucket": learned_bucket,
            "consensus": energy_consensus(features),
            "energy_sustained": candidates.energy_sustained,
            "energy_peak": candidates.energy_peak,
            "loudness_norm": candidates.loudness_norm,
            "loudness_lufs": candidates.loudness_lufs,
            "bass_abs": candidates.bass_abs,
            "drums_abs": candidates.drums_abs,
            "harmonic_abs": candidates.harmonic_abs,
            "groove_abs": candidates.groove_abs,
        }
        if row["energy_learned"] is not None:
            payload["stored_energy_learned"] = row["energy_learned"]
            payload["stored_energy_learned_bucket"] = row["energy_learned_bucket"]
        labels = label_lookup.get(str(row["track_id"]))
        if labels:
            payload["teacher_energy"] = labels.get("teacher_energy") or None
            payload["teacher_source"] = labels.get("teacher_source") or None
            payload["teacher_confidence"] = labels.get("teacher_confidence") or None
            payload["manual_bucket"] = labels.get("manual_bucket") or None
            payload["manual_score"] = labels.get("manual_score") or None
        payload_rows.append(payload)
        if not args.json:
            title = row["title"] or path.stem
            stored_label = (
                f"{float(payload['stored_energy_abs']):.3f}"
                if payload["stored_energy_abs"] is not None
                else "none"
            )
            learned_label = f"{float(learned):.3f}" if learned is not None else "none"
            hybrid_label = f"{float(hybrid_blended):.3f}" if hybrid_blended is not None else "none"
            teacher_label = payload.get("teacher_energy")
            print(
                f"[{index}/{total}] {title} :: "
                f"stored={stored_label} :: "
                f"baseline={payload['baseline']:.3f} :: "
                f"hybrid={hybrid_label} :: "
                f"learned={learned_label} :: "
                f"club={payload['club_fusion']:.3f} :: "
                f"pressure={payload['pressure_fusion']:.3f} :: "
                f"consensus={payload['consensus']:.3f}"
                + (f" :: teacher={teacher_label}" if teacher_label not in (None, "") else "")
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
        fieldnames = list(payload_rows[0].keys()) if payload_rows else [
            "position", "track_id", "title", "artist", "file_path", "stored_energy_abs",
            "baseline", "loudness_fusion", "club_fusion", "pressure_fusion", "hybrid_blended", "learned", "learned_bucket", "consensus",
            "energy_sustained", "energy_peak", "loudness_norm", "loudness_lufs",
            "bass_abs", "drums_abs", "harmonic_abs", "groove_abs",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(payload_rows)
        if not args.json:
            print(f"Wrote CSV report to {output_path}")
    return 0


def handle_export_energy_dataset(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    with Database(settings.database_path) as database:
        rows = database.get_playlist_tracks(args.playlist)
        if not rows:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_rows: list[dict[str, object]] = []
    total = len(rows)
    print(f"Exporting energy dataset for '{args.playlist}'")
    for row in rows:
        index = int(row["position"])
        path = Path(row["file_path"]).resolve()
        candidates = analyze_energy_path(
            path,
            sample_rate=settings.analysis.sample_rate,
            mono=settings.analysis.mono,
        )
        features = EnergyFeatureVector(
            baseline=candidates.baseline,
            loudness_fusion=candidates.loudness_fusion,
            club_fusion=candidates.club_fusion,
            pressure_fusion=candidates.pressure_fusion,
            energy_sustained=float(candidates.energy_sustained if candidates.energy_sustained is not None else 0.5),
            energy_peak=float(candidates.energy_peak if candidates.energy_peak is not None else 0.5),
            loudness_norm=candidates.loudness_norm,
            loudness_lufs=candidates.loudness_lufs,
            bass_abs=candidates.bass_abs,
            drums_abs=float(candidates.drums_abs if candidates.drums_abs is not None else 0.5),
            harmonic_abs=float(candidates.harmonic_abs if candidates.harmonic_abs is not None else 0.5),
            groove_abs=float(candidates.groove_abs if candidates.groove_abs is not None else 0.5),
        )
        dataset_rows.append(build_energy_dataset_row(row=row, features=features, playlist_name=args.playlist))
        print(f"[{index}/{total}] {row['title'] or path.stem}")

    fieldnames = [
        "track_id", "file_hash", "file_path", "title", "artist", "playlist_name", "position", "stored_energy_abs",
        *FEATURE_NAMES,
        "teacher_energy", "teacher_source", "teacher_confidence", "manual_bucket", "manual_score", "manual_notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset_rows)
    print(f"Wrote energy dataset to {output_path}")
    return 0


def handle_train_energy_model(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset).expanduser().resolve()
    model_out = Path(args.model_out).expanduser().resolve()
    meta_out = Path(args.meta_out).expanduser().resolve()

    rows = load_energy_dataset_rows(dataset_path)
    bundle, metrics = train_energy_bundle(rows)
    metadata = save_energy_bundle(bundle, metrics, model_out=model_out, meta_out=meta_out)

    print(f"Trained energy model from {dataset_path}")
    print(f"- model artifact: {model_out}")
    print(f"- metadata: {meta_out}")
    print(f"- model signature: {metadata['artifact_signature']}")
    print(f"- teacher rows: {metadata['dataset_counts']['teacher_total']}")
    print(f"- manual rows: {metadata['dataset_counts']['manual_total']}")
    return 0


def handle_benchmark_energy_model(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    dataset_path = Path(args.dataset).expanduser().resolve()
    model_path = resolve_energy_model_path(args.model_path or settings.analysis.energy_model_path, settings.repo_root)
    meta_path = resolve_energy_model_meta_path(args.meta_path or settings.analysis.energy_model_meta_path, settings.repo_root)
    if not model_path.is_file() or not meta_path.is_file():
        raise SystemExit(f"Energy model artifacts were not found: {model_path} / {meta_path}")

    rows = load_energy_dataset_rows(dataset_path)
    benchmark = evaluate_energy_model(rows, model_path=model_path, meta_path=meta_path)

    if args.json:
        print(json.dumps(benchmark, indent=2, sort_keys=True))
    else:
        print(f"Energy benchmark :: model_signature={benchmark['model_signature']}")
        print(
            f"Validation rows :: teacher={benchmark['dataset_counts']['validation_teacher']} :: "
            f"manual={benchmark['dataset_counts']['validation_manual']}"
        )
        for comparator_name, comparator in benchmark["comparators"].items():
            teacher_metrics = comparator["teacher_metrics"] or {}
            manual_metrics = comparator["manual_metrics"] or {}
            print(
                f"- {comparator_name}: "
                f"teacher_mae={teacher_metrics.get('mae', 0.0):.4f}, "
                f"teacher_rmse={teacher_metrics.get('rmse', 0.0):.4f}, "
                f"teacher_spearman={teacher_metrics.get('spearman', 0.0):.4f}, "
                f"manual_macro_f1={manual_metrics.get('macro_f1', 0.0):.4f}, "
                f"manual_bucket_accuracy={manual_metrics.get('bucket_accuracy', 0.0):.4f}"
            )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows_for_csv: list[dict[str, object]] = []
        for comparator_name, comparator in benchmark["comparators"].items():
            rows_for_csv.append(
                {
                    "comparator": comparator_name,
                    "teacher_mae": comparator["teacher_metrics"]["mae"] if comparator["teacher_metrics"] else None,
                    "teacher_rmse": comparator["teacher_metrics"]["rmse"] if comparator["teacher_metrics"] else None,
                    "teacher_spearman": comparator["teacher_metrics"]["spearman"] if comparator["teacher_metrics"] else None,
                    "manual_bucket_accuracy": comparator["manual_metrics"]["bucket_accuracy"] if comparator["manual_metrics"] else None,
                    "manual_macro_f1": comparator["manual_metrics"]["macro_f1"] if comparator["manual_metrics"] else None,
                    "manual_weighted_kappa": comparator["manual_metrics"]["weighted_kappa"] if comparator["manual_metrics"] else None,
                    "manual_score_mae": comparator["manual_metrics"]["score_mae"] if comparator["manual_metrics"] else None,
                }
            )
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_for_csv[0].keys()) if rows_for_csv else [
                "comparator", "teacher_mae", "teacher_rmse", "teacher_spearman",
                "manual_bucket_accuracy", "manual_macro_f1", "manual_weighted_kappa", "manual_score_mae",
            ])
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
        if args.command == "analyze-relative-playlist":
            return handle_analyze_relative_playlist(args)
        if args.command == "analyze-energy-playlist":
            return handle_analyze_energy_playlist(args)
        if args.command == "export-energy-dataset":
            return handle_export_energy_dataset(args)
        if args.command == "train-energy-model":
            return handle_train_energy_model(args)
        if args.command == "benchmark-energy-model":
            return handle_benchmark_energy_model(args)
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
