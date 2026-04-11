"""CLI handlers for worker/admin commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import time

import cuemate_analysis.cli as _cli
from cuemate_analysis.analysis import utc_now
from cuemate_analysis.essentia_semantic_backend import (
    download_essentia_semantic_models,
    ensure_essentia_semantic_service,
    estimate_essentia_semantic_batch,
    resolve_essentia_semantic_model_root,
    resolve_essentia_semantic_service_name,
    resolve_essentia_semantic_service_port,
)
from cuemate_analysis.ingest import read_track_metadata_with_overrides
from cuemate_analysis.key_backend import (
    ensure_musicalkeycnn_service,
    estimate_musicalkeycnn_keys,
    resolve_musicalkeycnn_image_name,
    resolve_musicalkeycnn_model_path,
    resolve_musicalkeycnn_service_name,
    resolve_musicalkeycnn_service_port,
)
from cuemate_analysis.tempo_backend import (
    ensure_tempocnn_service,
    estimate_tempocnn_bpms,
    resolve_tempocnn_image_name,
    resolve_tempocnn_service_name,
    resolve_tempocnn_service_port,
)


def handle_purge_model_cache(args: argparse.Namespace) -> int:
    settings = _cli.load_runtime_settings()
    scoped_paths: list[str] = []

    if args.playlist:
        with _cli.Database(settings.database_path) as database:
            rows = database.get_playlist_tracks(args.playlist)
            if not rows:
                raise SystemExit(f"Playlist '{args.playlist}' was not found.")
            scoped_paths.extend([str(Path(row["file_path"]).resolve()) for row in rows])

    if args.path:
        scoped_paths.extend([str(Path(path).expanduser().resolve()) for path in args.path])

    deduped_paths = sorted({str(Path(path).resolve()) for path in scoped_paths})
    deleted_tempocnn = 0
    deleted_musicalkeycnn = 0
    deleted_essentia_semantics = 0

    if args.backend in {"all", "tempocnn"}:
        deleted_tempocnn = _cli.purge_tempocnn_cache(
            file_paths=deduped_paths or None,
            clear_warm_service=args.clear_warm_services,
        )
    if args.backend in {"all", "musicalkeycnn"}:
        deleted_musicalkeycnn = _cli.purge_musicalkeycnn_cache(
            file_paths=deduped_paths or None,
            clear_warm_service=args.clear_warm_services,
        )
    if args.backend in {"all", "essentia_semantics"}:
        deleted_essentia_semantics = _cli.purge_essentia_semantic_cache(
            file_paths=deduped_paths or None,
            clear_warm_service=args.clear_warm_services,
        )

    scope_label = "all cached tracks"
    if deduped_paths:
        scope_label = f"{len(deduped_paths)} scoped file(s)"

    print(f"Purged model inference cache for {scope_label}.")
    if args.backend in {"all", "tempocnn"}:
        print(f"- TempoCNN rows removed: {deleted_tempocnn}")
        print(
            "- TempoCNN warm service state cleared"
            if args.clear_warm_services
            else "- TempoCNN warm service state preserved"
        )
    if args.backend in {"all", "musicalkeycnn"}:
        print(f"- MusicalKeyCNN rows removed: {deleted_musicalkeycnn}")
        print(
            "- MusicalKeyCNN warm service state cleared"
            if args.clear_warm_services
            else "- MusicalKeyCNN warm service state preserved"
        )
    if args.backend in {"all", "essentia_semantics"}:
        print(f"- Essentia semantics rows removed: {deleted_essentia_semantics}")
        print(
            "- Essentia semantics warm service state cleared"
            if args.clear_warm_services
            else "- Essentia semantics warm service state preserved"
        )
    print(f"- Persistent cache DB: {_cli.resolve_inference_cache_path()}")
    print("- Re-run analysis with --force if you want stored playlist analysis rows refreshed too.")
    return 0


def handle_clear_analysis_queue(args: argparse.Namespace) -> int:
    settings = _cli.load_runtime_settings()
    statuses = ["pending"]
    if args.include_running:
        statuses.append("running")
    job_kind = None if args.job_kind == "all" else args.job_kind

    with _cli.Database(settings.database_path) as database:
        deleted = database.clear_analysis_jobs(statuses=statuses, job_kind=job_kind)

    scope = f"job kind '{job_kind}'" if job_kind is not None else "all job kinds"
    status_label = ", ".join(statuses)
    print(f"Removed {deleted} analysis job(s) from the local queue.")
    print(f"- Statuses cleared: {status_label}")
    print(f"- Scope: {scope}")
    print(f"- Database: {settings.database_path}")
    if args.include_running:
        print("- Running jobs were included; make sure no active analysis worker is still using those rows.")
    return 0


def handle_run_analysis_worker(args: argparse.Namespace) -> int:
    settings = _cli.load_runtime_settings()
    expected_essentia_semantic_signature = _cli.resolve_expected_essentia_semantic_signature(settings)
    effective_analysis_signature = _cli.build_effective_analysis_signature(
        settings.analysis_signature,
        tempocnn_accelerator="auto",
        musicalkeycnn_model=settings.analysis.key_model_path,
        musicalkeycnn_device=settings.analysis.key_device,
        musicalkeycnn_policy=settings.analysis.key_policy,
        essentia_semantic_signature=expected_essentia_semantic_signature,
    )

    with _cli.Database(settings.database_path) as database:
        jobs = database.claim_pending_analysis_jobs(
            job_kind="enrichment",
            limit=args.limit,
            started_at=utc_now(),
        )
        if not jobs:
            print("No pending enrichment jobs were found.")
            return 0

        pending_items: list[dict[str, object]] = []
        for offset, job in enumerate(jobs, start=1):
            track_row = database.get_track_row(str(job["track_id"]))
            if track_row is None:
                database.mark_analysis_job_failed(
                    int(job["id"]),
                    "Track metadata was missing.",
                    0.0,
                    utc_now(),
                )
                continue

            try:
                track = read_track_metadata_with_overrides(
                    Path(str(track_row["file_path"])),
                    bpm_imported=track_row["imported_bpm"],
                    key_imported=track_row["imported_key"],
                    title_override=track_row["title"],
                    artist_override=track_row["artist"],
                    genre_override=track_row["genre"],
                    import_source=track_row["import_source"] or "local_files",
                )
            except Exception as exc:
                database.mark_analysis_job_failed(
                    int(job["id"]),
                    f"Metadata reconstruction failed: {str(exc)}",
                    0.0,
                    utc_now(),
                )
                continue
            prepared = _cli.PreparedTrack(
                row={
                    "playlist_id": job["playlist_id"],
                    "track_id": job["track_id"],
                    "file_path": track.file_path.as_posix(),
                },
                track=track,
                position=offset,
                resolved_path=track.file_path.resolve(),
            )
            pending_items.append(
                {
                    "prepared": prepared,
                    "job_id": int(job["id"]),
                    "start_time": time.perf_counter(),
                    "display_index": offset,
                    "display_total": len(jobs),
                }
            )

        if not pending_items:
            print("No runnable enrichment jobs were found.")
            return 0

        (
            processed,
            updated_track_ids,
            dsp_diagnostics,
            tempo_diagnostics,
            key_diagnostics,
            essentia_diagnostics,
        ) = _cli._process_enrichment_batch(
            pending_items=pending_items,
            settings=settings,
            database=database,
            effective_analysis_signature=effective_analysis_signature,
            expected_essentia_semantic_signature=expected_essentia_semantic_signature,
            print_progress=True,
        )

        claimed_job_ids = [int(job["id"]) for job in jobs]
        final_jobs = database.get_analysis_jobs_by_ids(claimed_job_ids)

        jobs_by_playlist: dict[str, list[sqlite3.Row]] = {}
        for job in final_jobs:
            playlist_id = job["playlist_id"]
            if playlist_id:
                jobs_by_playlist.setdefault(str(playlist_id), []).append(job)

        touched_playlists = {
            playlist_id
            for playlist_id, playlist_jobs in jobs_by_playlist.items()
            if all(str(job["status"]) == "completed" for job in playlist_jobs)
        }

        if touched_playlists:
            _cli._refresh_relative_for_playlists(database, settings, touched_playlists)

        if updated_track_ids and hasattr(database, "get_playlists_containing_tracks") and hasattr(database, "mark_playlists_stale"):
            linked = database.get_playlists_containing_tracks(updated_track_ids)
            stale_targets = [pid for pid in linked if pid not in touched_playlists]
            if stale_targets:
                database.mark_playlists_stale(stale_targets, "absolute_track_changed", utc_now())
                print(f"Marked {len(stale_targets)} linked playlist(s) stale.")

        print(f"Processed {processed} enrichment job(s).")

        if args.print_backend_diagnostics:
            _cli.print_backend_diagnostics("DSP local lane", dsp_diagnostics, requested_device="cpu")
            _cli.print_backend_diagnostics("TempoCNN", tempo_diagnostics, requested_device="auto")
            _cli.print_backend_diagnostics("MusicalKeyCNN", key_diagnostics, requested_device=settings.analysis.key_device)
            _cli.print_backend_diagnostics(
                "Essentia semantics",
                essentia_diagnostics,
                requested_device=settings.analysis.essentia_semantic_device,
            )

        return 0


def handle_prewarm_model_services(args: argparse.Namespace) -> int:
    settings = _cli.load_runtime_settings()
    warm_path = None
    if args.path:
        warm_path = Path(args.path).expanduser().resolve()
        if not warm_path.is_file():
            raise FileNotFoundError(f"Warmup path was not found: {warm_path}")

    if warm_path is not None:
        estimate_tempocnn_bpms([warm_path])
        estimate_musicalkeycnn_keys(
            [warm_path],
            model_path=settings.analysis.key_model_path,
            device=settings.analysis.key_device,
            policy=settings.analysis.key_policy,
        )
        if settings.analysis.essentia_semantics_enabled:
            estimate_essentia_semantic_batch(
                [warm_path],
                model_root=settings.analysis.essentia_semantic_model_root,
                image_name=settings.analysis.essentia_semantic_image,
                device=settings.analysis.essentia_semantic_device,
                family_policy=settings.analysis.essentia_semantic_model_family_policy,
            )
    else:
        tempocnn_ready, tempocnn_notes = ensure_tempocnn_service(
            drive_letters=[],
            image_name=resolve_tempocnn_image_name(None),
            accelerator="auto",
            service_name=resolve_tempocnn_service_name(None),
            service_port=resolve_tempocnn_service_port(None),
        )
        if not tempocnn_ready:
            raise SystemExit("\n".join(tempocnn_notes) or "TempoCNN service warmup failed.")
        key_ready, key_notes = ensure_musicalkeycnn_service(
            model_path=resolve_musicalkeycnn_model_path(settings.analysis.key_model_path),
            image_name=resolve_musicalkeycnn_image_name(None),
            device=settings.analysis.key_device,
            service_name=resolve_musicalkeycnn_service_name(None),
            service_port=resolve_musicalkeycnn_service_port(None),
            drive_letters=[],
        )
        if not key_ready:
            raise SystemExit("\n".join(key_notes) or "MusicalKeyCNN service warmup failed.")
        if settings.analysis.essentia_semantics_enabled:
            semantic_ready, semantic_notes = ensure_essentia_semantic_service(
                model_root=resolve_essentia_semantic_model_root(
                    settings.analysis.essentia_semantic_model_root
                ),
                image_name=settings.analysis.essentia_semantic_image,
                device=settings.analysis.essentia_semantic_device,
                family_policy=settings.analysis.essentia_semantic_model_family_policy,
                service_name=resolve_essentia_semantic_service_name(None),
                service_port=resolve_essentia_semantic_service_port(None),
                drive_letters=[],
            )
            if not semantic_ready:
                raise SystemExit(
                    "\n".join(semantic_notes) or "Essentia semantic service warmup failed."
                )
    print("Model services prewarmed.")
    return 0


def handle_download_essentia_semantic_models(args: argparse.Namespace) -> int:
    settings = _cli.load_runtime_settings()
    model_root = args.model_root or settings.analysis.essentia_semantic_model_root
    family_policy = args.family_policy or settings.analysis.essentia_semantic_model_family_policy
    downloaded = download_essentia_semantic_models(model_root=model_root, family_policy=family_policy)
    resolved_root = resolve_essentia_semantic_model_root(model_root)
    print(f"Downloaded Essentia semantic model bundle to {resolved_root}")
    for path in downloaded:
        print(f"- {path}")
    return 0


def handle_benchmark_dsp(args: argparse.Namespace) -> int:
    from cuemate_analysis.dsp_benchmark import (
        benchmark_dsp_paths,
        summarize_dsp_benchmark,
        write_benchmark_csv,
        write_benchmark_json,
    )

    settings = _cli.load_runtime_settings()
    paths: list[Path] = []

    if args.playlist:
        with _cli.Database(settings.database_path) as database:
            rows = database.get_playlist_relative_inputs(args.playlist)
            if not rows:
                raise SystemExit(f"Playlist '{args.playlist}' was not found.")
            paths.extend(Path(row["file_path"]) for row in rows)

    if args.path:
        paths.extend(Path(p).expanduser().resolve() for p in args.path)

    if not paths:
        raise SystemExit("Provide --playlist or --path to specify files to benchmark.")

    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    print(f"Benchmarking DSP pipeline for {len(paths)} track(s)...")
    samples = benchmark_dsp_paths(paths, settings)
    summary = summarize_dsp_benchmark(samples)

    if args.json:
        print(json.dumps({"track_count": len(samples), "summary": summary}, indent=2, sort_keys=True))
    else:
        print(f"\nDSP Benchmark Summary ({len(samples)} tracks):")
        for key, value in sorted(summary.items()):
            print(f"  {key}: {value:.4f}s")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        if output_path.suffix == ".json":
            write_benchmark_json(samples, summary, output_path)
        else:
            write_benchmark_csv(samples, output_path)
        if not args.json:
            print(f"\nReport written to {output_path}")

    return 0
