"""CLI handlers for analysis commands (absolute, relative, energy, essentia)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
import cuemate_analysis.cli as _cli
from cuemate_analysis.analysis import (
    DspLaneResult,
    build_fast_analysis_result,
    utc_now,
)
from cuemate_analysis.config import build_fast_analysis_signature
from cuemate_analysis.energy_experiments import analyze_energy_path
from cuemate_analysis.energy_features import EnergyFeatureVector, energy_consensus
from cuemate_analysis.essentia_semantic_backend import EssentiaSemanticEstimate
from cuemate_analysis.ingest import read_track_metadata
from cuemate_analysis.key_backend import KeyEstimate
from cuemate_analysis.relative_context import row_to_relative_track_input
from cuemate_analysis.tempo_backend import TempoEstimate


def handle_analyze_playlist(args: argparse.Namespace) -> int:
    settings = _cli.load_runtime_settings()
    playlist_name = args.playlist
    expected_essentia_semantic_signature = _cli.resolve_expected_essentia_semantic_signature(settings)
    effective_analysis_signature = _cli.build_effective_analysis_signature(
        settings.analysis_signature,
        tempocnn_accelerator="auto",
        musicalkeycnn_model=settings.analysis.key_model_path,
        musicalkeycnn_device=settings.analysis.key_device,
        musicalkeycnn_policy=settings.analysis.key_policy,
        essentia_semantic_signature=expected_essentia_semantic_signature,
    )
    fast_analysis_signature = build_fast_analysis_signature(settings)

    with _cli.Database(settings.database_path) as database:
        if hasattr(database, "get_table_columns"):
            track_feature_columns = database.get_table_columns("track_features_abs")
            required_columns = {"energy_heuristic_abs"}
            missing_columns = sorted(required_columns - track_feature_columns)
            if missing_columns:
                raise SystemExit(
                    "The local database schema is out of date and is missing track_features_abs column(s): "
                    f"{', '.join(missing_columns)}. Apply migrations first with "
                    "powershell -ExecutionPolicy Bypass -File .\\scripts\\docker-compose.ps1 --profile ops run --rm migrate"
                )
            fast_columns = database.get_table_columns("track_features_fast")
            if "track_id" not in fast_columns:
                raise SystemExit(
                    "The local database schema is out of date and is missing the track_features_fast table. Apply migrations first with "
                    "powershell -ExecutionPolicy Bypass -File .\\scripts\\docker-compose.ps1 --profile ops run --rm migrate"
                )

        rows = database.get_playlist_tracks(playlist_name)
        if not rows:
            raise SystemExit(f"Playlist '{playlist_name}' was not found. Import it first.")

        total = len(rows)
        processed = 0
        playlist_id: str = str(rows[0]["playlist_id"]) if rows else ""
        updated_track_ids: list[str] = []
        dsp_diagnostics: list[DspLaneResult] = []
        tempo_diagnostics: list[TempoEstimate] = []
        key_diagnostics: list[KeyEstimate] = []
        essentia_diagnostics: list[EssentiaSemanticEstimate] = []
        pending_enrichment_items: list[dict[str, object]] = []

        for chunk in _cli.chunk_items(rows, settings.analysis.full_chunk_size):
            prepared_tracks, prepare_failures = _cli.prepare_analysis_chunk(chunk)
            for row, error_message in prepare_failures:
                index = int(row["position"])
                print(f"[{index}/{total}] failed {row['track_id']}: {error_message}", file=sys.stderr)

            if not prepared_tracks:
                continue

            prefetched_tempocnn_estimates = _cli.run_tempo_lane(prepared_tracks, settings)
            prefetched_musicalkeycnn_estimates = _cli.run_key_lane(prepared_tracks, settings)

            for prepared in prepared_tracks:
                row = prepared.row
                track = prepared.track
                index = prepared.position

                database.upsert_track(track, utc_now())

                if not _cli.should_skip_fast_analysis(
                    row,
                    track,
                    effective_analysis_signature=fast_analysis_signature,
                    config_signature=settings.config_signature,
                    force=args.force,
                ):
                    fast_job_id = database.create_analysis_job_with_kind(
                        playlist_id=row["playlist_id"],
                        track_id=row["track_id"],
                        track_path=row["file_path"],
                        job_kind="fast_pass",
                        analysis_mode="fast_pass",
                        analysis_signature=fast_analysis_signature,
                        config_signature=settings.config_signature,
                        source_file_hash=track.file_hash,
                        priority=total - index,
                        created_at=utc_now(),
                    )
                    fast_started = time.perf_counter()
                    database.mark_analysis_job_started(fast_job_id, utc_now())
                    try:
                        fast_result = build_fast_analysis_result(
                            track,
                            settings,
                            prefetched_tempocnn_estimate=prefetched_tempocnn_estimates.get(prepared.resolved_path),
                            prefetched_musicalkeycnn_estimate=prefetched_musicalkeycnn_estimates.get(prepared.resolved_path),
                            analysis_signature=fast_analysis_signature,
                        )
                        database.upsert_track_fast_features(fast_result)
                        database.mark_analysis_job_completed(
                            fast_job_id,
                            round(time.perf_counter() - fast_started, 3),
                            _cli.build_fast_job_timing_breakdown(
                                settings=settings,
                                effective_analysis_signature=fast_analysis_signature,
                            ),
                            utc_now(),
                        )
                        if args.analysis_mode in {"fast_pass", "staged"}:
                            print(
                                f"[{index}/{total}] fast_ready {track.id} ({track.title}) -> "
                                f"{fast_result.bpm:.1f} BPM ({fast_result.bpm_source}), {fast_result.key} ({fast_result.key_source})"
                            )
                    except Exception as exc:
                        database.mark_analysis_job_failed(
                            fast_job_id,
                            str(exc),
                            round(time.perf_counter() - fast_started, 3),
                            utc_now(),
                        )
                        print(f"[{index}/{total}] failed {track.id}: {exc}", file=sys.stderr)
                        if args.analysis_mode in {"fast_pass", "staged"}:
                            continue
                elif args.analysis_mode == "fast_pass":
                    print(f"[{index}/{total}] skipped {track.id} ({track.title}) - fast analysis is already current")

                if args.analysis_mode == "fast_pass":
                    continue

                if _cli.should_skip_analysis(
                    row,
                    track,
                    effective_analysis_signature=effective_analysis_signature,
                    config_signature=settings.config_signature,
                    analysis_mode="full",
                    force=args.force,
                    expected_essentia_semantic_signature=expected_essentia_semantic_signature,
                ):
                    if args.analysis_mode == "full":
                        print(f"[{index}/{total}] skipped {track.id} ({track.title}) - analysis is already current")
                    continue

                job_id = database.create_analysis_job_with_kind(
                    playlist_id=row["playlist_id"],
                    track_id=row["track_id"],
                    track_path=row["file_path"],
                    job_kind="enrichment",
                    analysis_mode="full",
                    analysis_signature=effective_analysis_signature,
                    config_signature=settings.config_signature,
                    source_file_hash=track.file_hash,
                    priority=total - index,
                    created_at=utc_now(),
                )
                pending_enrichment_items.append(
                    {
                        "prepared": prepared,
                        "job_id": job_id,
                        "start_time": time.perf_counter(),
                        "display_index": index,
                        "display_total": total,
                    }
                )

        if args.analysis_mode == "staged":
            print(f"Queued {len(pending_enrichment_items)} enrichment job(s) for '{playlist_name}'.")
            return 0

        if args.analysis_mode == "full":
            (
                processed,
                updated_track_ids,
                dsp_diagnostics,
                tempo_diagnostics,
                key_diagnostics,
                essentia_diagnostics,
            ) = _cli._process_enrichment_batch(
                pending_items=pending_enrichment_items,
                settings=settings,
                database=database,
                effective_analysis_signature=effective_analysis_signature,
                expected_essentia_semantic_signature=expected_essentia_semantic_signature,
                print_progress=True,
            )

    print(f"Completed playlist analysis for '{playlist_name}'. Updated {processed} track(s).")
    if args.analysis_mode == "full":
        _cli.print_backend_diagnostics("DSP local lane", dsp_diagnostics, requested_device="cpu")
        _cli.print_backend_diagnostics("TempoCNN", tempo_diagnostics, requested_device="auto")
        _cli.print_backend_diagnostics("MusicalKeyCNN", key_diagnostics, requested_device=settings.analysis.key_device)
        _cli.print_backend_diagnostics(
            "Essentia semantics",
            essentia_diagnostics,
            requested_device=settings.analysis.essentia_semantic_device,
        )

    if args.analysis_mode == "full" and processed > 0 and playlist_id:
        with _cli.Database(settings.database_path) as post_db:
            try:
                if hasattr(post_db, "get_playlist_name_by_id"):
                    _cli._refresh_relative_for_playlists(post_db, settings, {playlist_id})
            except Exception as exc:
                print(f"Warning: canonical relative refresh failed for '{playlist_name}': {exc}", file=sys.stderr)

            if updated_track_ids and hasattr(post_db, "get_playlists_containing_tracks") and hasattr(post_db, "mark_playlists_stale"):
                linked = post_db.get_playlists_containing_tracks(updated_track_ids)
                stale_targets = [pid for pid in linked if pid != playlist_id]
                if stale_targets:
                    post_db.mark_playlists_stale(stale_targets, "absolute_track_changed", utc_now())
                    print(f"Marked {len(stale_targets)} linked playlist(s) stale.")

    return 0


def handle_analyze_bpm(args: argparse.Namespace) -> int:
    from cuemate_analysis.tempo_backend import estimate_tempocnn_bpms

    path = Path(args.path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file was not found: {path}")

    metadata = read_track_metadata(path)
    estimate = estimate_tempocnn_bpms([path])[path.resolve()]
    payload = _cli.build_bpm_payload(path, metadata, estimate)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"File: {payload['file_path']}")
    print(f"Track: {metadata.artist or 'Unknown artist'} - {metadata.title or path.stem}")
    print(f"Tagged BPM: {metadata.bpm_tag if metadata.bpm_tag is not None else 'none'}")
    print("Backend: tempocnn")
    print(f"BPM: {_cli.summarize_estimate(estimate)}")
    if estimate.confidence is not None:
        print(f"Confidence: {estimate.confidence:.2f}")
    _cli.print_backend_diagnostics("TempoCNN", [estimate], requested_device="auto")
    for note in estimate.notes:
        print(f"- {note}")
    return 0


def handle_analyze_bpm_key(args: argparse.Namespace) -> int:
    from cuemate_analysis.key_backend import estimate_musicalkeycnn_keys
    from cuemate_analysis.tempo_backend import estimate_tempocnn_bpms

    settings = _cli.load_runtime_settings()
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
    payload = _cli.build_bpm_key_payload(path, metadata, bpm_estimate, key_estimate)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"File: {payload['file_path']}")
    print(f"Track: {metadata.artist or 'Unknown artist'} - {metadata.title or path.stem}")
    print(f"BPM: {_cli.summarize_estimate(bpm_estimate)}")
    if bpm_estimate.confidence is not None:
        print(f"BPM Confidence: {bpm_estimate.confidence:.2f}")
    print(f"Key: {_cli.summarize_key_estimate(key_estimate)}")
    if key_estimate.confidence is not None:
        print(f"Key Confidence: {key_estimate.confidence:.2f}")
    _cli.print_backend_diagnostics("TempoCNN", [bpm_estimate], requested_device="auto")
    _cli.print_backend_diagnostics("MusicalKeyCNN", [key_estimate], requested_device=settings.analysis.key_device)
    return 0


def handle_analyze_bpm_playlist(args: argparse.Namespace) -> int:
    from cuemate_analysis.tempo_backend import estimate_tempocnn_bpms

    settings = _cli.load_runtime_settings()
    with _cli.Database(settings.database_path) as database:
        rows = database.get_playlist_tracks(args.playlist)
        if not rows:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    payload_rows: list[dict[str, object]] = []
    diagnostics_estimates: list[TempoEstimate] = []
    total = len(rows)
    if not args.json:
        print(f"Playlist '{args.playlist}' BPM pass with backend tempocnn")

    for chunk in _cli.chunk_items(rows, settings.analysis.tempo_chunk_size):
        prefetched_tempocnn_estimates = estimate_tempocnn_bpms([Path(row["file_path"]) for row in chunk])

        for row in chunk:
            index = int(row["position"])
            path = Path(row["file_path"]).resolve()
            estimate = prefetched_tempocnn_estimates[path]
            diagnostics_estimates.append(estimate)
            payload = _cli.build_fast_playlist_bpm_payload(row, path, estimate)
            payload["track_id"] = row["track_id"]
            payload["position"] = index
            payload_rows.append(payload)
            if not args.json:
                title = payload["title"] or Path(str(payload["file_path"])).stem
                print(
                    f"[{index}/{total}] {title} [{payload['track_id']}] :: "
                    f"{_cli.summarize_estimate(estimate)}"
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
    else:
        _cli.print_backend_diagnostics("TempoCNN", diagnostics_estimates, requested_device="auto")
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
    settings = _cli.load_runtime_settings()
    with _cli.Database(settings.database_path) as database:
        rows = database.get_playlist_tracks(args.playlist)
        if not rows:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    payload_rows: list[dict[str, object]] = []
    bpm_diagnostics: list[TempoEstimate] = []
    key_diagnostics: list[KeyEstimate] = []
    total = len(rows)
    if not args.json:
        print(f"Playlist '{args.playlist}' BPM+key pass with TempoCNN + MusicalKeyCNN")

    bpm_key_chunk_size = min(settings.analysis.tempo_chunk_size, settings.analysis.key_chunk_size)
    for chunk in _cli.chunk_items(rows, bpm_key_chunk_size):
        chunk_paths = [Path(row["file_path"]) for row in chunk]
        prefetched_tempocnn_estimates, prefetched_musicalkeycnn_estimates = _cli.prefetch_bpm_and_key_estimates(
            chunk_paths,
            settings,
        )

        for row in chunk:
            index = int(row["position"])
            path = Path(row["file_path"]).resolve()
            bpm_estimate = prefetched_tempocnn_estimates[path]
            key_estimate = prefetched_musicalkeycnn_estimates[path]
            bpm_diagnostics.append(bpm_estimate)
            key_diagnostics.append(key_estimate)
            payload = _cli.build_fast_playlist_bpm_key_payload(row, path, bpm_estimate, key_estimate)
            payload["track_id"] = row["track_id"]
            payload["position"] = index
            payload_rows.append(payload)
            if not args.json:
                title = payload["title"] or Path(str(payload["file_path"])).stem
                print(
                    f"[{index}/{total}] {title} [{payload['track_id']}] :: "
                    f"{_cli.summarize_estimate(bpm_estimate)} :: {_cli.summarize_key_estimate(key_estimate)}"
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
    else:
        _cli.print_backend_diagnostics("TempoCNN", bpm_diagnostics, requested_device="auto")
        _cli.print_backend_diagnostics("MusicalKeyCNN", key_diagnostics, requested_device=settings.analysis.key_device)
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
    settings = _cli.load_runtime_settings()

    energy_source = args.energy_source or settings.analysis.energy_source_default
    if energy_source == "essentia_fused":
        energy_source = "canonical"
    if energy_source == "heuristic":
        energy_source = "heuristic_legacy"

    with _cli.Database(settings.database_path) as database:
        abs_rows = database.get_playlist_relative_inputs(args.playlist)
        if not abs_rows:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")

        playlist_id = str(abs_rows[0]["playlist_id"])

        if energy_source == "canonical":
            from cuemate_analysis.config import build_relative_experiment_signature

            stats_row = database.get_playlist_stats(playlist_id)
            current_relative_signature = build_relative_experiment_signature(
                settings,
                energy_source="canonical",
            )

            needs_refresh = (
                stats_row is None
                or bool(stats_row["is_stale"])
                or str(stats_row["relative_signature"]) != current_relative_signature
            )

            if needs_refresh:
                if stats_row is None:
                    reason = "missing"
                elif bool(stats_row["is_stale"]):
                    reason = str(stats_row["stale_reason"] or "stale")
                else:
                    reason = "relative_signature_changed"

                preview = _cli._refresh_canonical_relative_preview(
                    database,
                    playlist_name=args.playlist,
                    playlist_id=playlist_id,
                    settings=settings,
                    abs_rows=abs_rows,
                    announce_reason=reason,
                    json_mode=args.json,
                )
            else:
                preview = _cli._load_persisted_canonical_preview(
                    database, args.playlist, playlist_id, abs_rows, settings
                )

            if args.limit and args.limit > 0:
                preview = _cli._limit_relative_preview(preview, limit=args.limit)
        else:
            if args.limit and args.limit > 0:
                abs_rows = abs_rows[: args.limit]
            rel_inputs = [row_to_relative_track_input(row) for row in abs_rows]
            preview = _cli.compute_relative_playlist_preview(
                rel_inputs,
                settings,
                playlist_name=args.playlist,
                is_limited=bool(args.limit and args.limit > 0),
                energy_source="heuristic_legacy",
            )

    _cli._run_canonical_relative_preview_and_print(
        preview,
        playlist_name=args.playlist,
        energy_source=energy_source,
        args=args,
        abs_rows=abs_rows,
    )

    if args.output:
        _cli._write_relative_csv(preview, Path(args.output).expanduser().resolve(), args)

    return 0


def handle_refresh_relative_playlist(args: argparse.Namespace) -> int:
    settings = _cli.load_runtime_settings()
    with _cli.Database(settings.database_path) as database:
        abs_rows = database.get_playlist_relative_inputs(args.playlist)
        if not abs_rows:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")
        playlist_id = str(abs_rows[0]["playlist_id"])
        preview = _cli._refresh_canonical_relative_preview(
            database,
            playlist_name=args.playlist,
            playlist_id=playlist_id,
            settings=settings,
            abs_rows=abs_rows,
            announce_reason="manual_refresh",
            json_mode=args.json,
        )

    if args.limit and args.limit > 0:
        preview = _cli._limit_relative_preview(preview, limit=args.limit)

    _cli._run_canonical_relative_preview_and_print(
        preview,
        playlist_name=args.playlist,
        energy_source="canonical",
        args=args,
        abs_rows=abs_rows,
    )

    if args.output:
        _cli._write_relative_csv(preview, Path(args.output).expanduser().resolve(), args)

    return 0


def handle_analyze_energy_playlist(args: argparse.Namespace) -> int:
    settings = _cli.load_runtime_settings()
    with _cli.Database(settings.database_path) as database:
        rows = database.get_playlist_relative_inputs(args.playlist)
        if not rows:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    payload_rows: list[dict[str, object]] = []
    fieldname_order: list[str] = []
    fieldname_seen: set[str] = set()
    total = len(rows)

    if not args.json:
        print(f"Playlist '{args.playlist}' energy diagnostics :: absolute-energy comparisons")

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
        if row["energy_essentia_fused"] is not None:
            payload["stored_energy_essentia_fused"] = row["energy_essentia_fused"]
            payload["stored_energy_essentia_bucket"] = row["energy_essentia_bucket"]
            payload["danceability_abs"] = row["danceability_abs"]
            payload["arousal_abs"] = row["arousal_abs"]
            payload["valence_abs"] = row["valence_abs"]
            payload["mood_aggressive_abs"] = row["mood_aggressive_abs"]
            payload["mood_party_abs"] = row["mood_party_abs"]
            payload["mood_relaxed_abs"] = row["mood_relaxed_abs"]
        for key in payload:
            if key not in fieldname_seen:
                fieldname_seen.add(key)
                fieldname_order.append(key)
        payload_rows.append(payload)
        if not args.json:
            title = row["title"] or path.stem
            stored_label = (
                f"{float(payload['stored_energy_abs']):.3f}"
                if payload["stored_energy_abs"] is not None
                else "none"
            )
            message = (
                f"[{index}/{total}] {title} :: "
                f"stored={stored_label} :: "
                f"baseline={payload['baseline']:.3f} :: "
                f"loudness={payload['loudness_fusion']:.3f} :: "
            )
            if payload.get("stored_energy_essentia_fused") is not None:
                message += f"essentia={float(payload['stored_energy_essentia_fused']):.3f} :: "
            message += (
                f"club={payload['club_fusion']:.3f} :: "
                f"pressure={payload['pressure_fusion']:.3f} :: "
                f"consensus={payload['consensus']:.3f}"
            )
            print(message)

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
        fieldnames = fieldname_order if payload_rows else [
            "position", "track_id", "title", "artist", "file_path", "stored_energy_abs",
            "baseline", "loudness_fusion", "club_fusion", "pressure_fusion", "consensus",
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


def handle_analyze_essentia_playlist(args: argparse.Namespace) -> int:
    settings = _cli.load_runtime_settings()
    with _cli.Database(settings.database_path) as database:
        rows = database.get_playlist_tracks(args.playlist)
        if not rows:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    payload_rows: list[dict[str, object]] = []
    fieldname_order: list[str] = []
    fieldname_seen: set[str] = set()
    diagnostics_estimates: list[EssentiaSemanticEstimate] = []
    total = len(rows)
    if not args.json:
        print(f"Playlist '{args.playlist}' Essentia semantic preview")

    for chunk in _cli.chunk_items(rows, settings.analysis.essentia_chunk_size):
        estimates = _cli.prefetch_essentia_semantic_estimates(chunk, settings)
        for row in chunk:
            index = int(row["position"])
            path = Path(row["file_path"]).resolve()
            estimate = estimates.get(path)
            if estimate is not None:
                diagnostics_estimates.append(estimate)
            payload = {
                "position": index,
                "track_id": row["track_id"],
                "title": row["title"],
                "artist": row["artist"],
                "file_path": path.as_posix(),
                "available": estimate.available if estimate is not None else False,
                "danceability_abs": estimate.danceability_abs if estimate is not None else None,
                "arousal_abs": estimate.arousal_abs if estimate is not None else None,
                "valence_abs": estimate.valence_abs if estimate is not None else None,
                "mood_aggressive_abs": estimate.mood_aggressive_abs if estimate is not None else None,
                "mood_party_abs": estimate.mood_party_abs if estimate is not None else None,
                "mood_relaxed_abs": estimate.mood_relaxed_abs if estimate is not None else None,
                "energy_essentia_fused": estimate.energy_essentia_fused if estimate is not None else None,
                "energy_essentia_bucket": estimate.energy_essentia_bucket if estimate is not None else None,
                "notes": [] if estimate is None else estimate.notes,
            }
            for key in payload:
                if key not in fieldname_seen:
                    fieldname_seen.add(key)
                    fieldname_order.append(key)
            payload_rows.append(payload)
            if not args.json:
                title = row["title"] or path.stem
                fused_label = "none" if payload["energy_essentia_fused"] is None else f"{float(payload['energy_essentia_fused']):.3f}"
                print(
                    f"[{index}/{total}] {title} :: "
                    f"danceability={payload['danceability_abs'] if payload['danceability_abs'] is not None else 'none'} :: "
                    f"arousal={payload['arousal_abs'] if payload['arousal_abs'] is not None else 'none'} :: "
                    f"party={payload['mood_party_abs'] if payload['mood_party_abs'] is not None else 'none'} :: "
                    f"relaxed={payload['mood_relaxed_abs'] if payload['mood_relaxed_abs'] is not None else 'none'} :: "
                    f"fused={fused_label} :: "
                    f"bucket={payload['energy_essentia_bucket'] or 'none'}"
                )

    if args.json:
        print(json.dumps({"playlist": args.playlist, "tracks": payload_rows}, indent=2, sort_keys=True))
    else:
        _cli.print_backend_diagnostics(
            "Essentia semantics",
            diagnostics_estimates,
            requested_device=settings.analysis.essentia_semantic_device,
        )
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = fieldname_order if payload_rows else [
            "position", "track_id", "title", "artist", "file_path", "available",
            "danceability_abs", "arousal_abs", "valence_abs", "mood_aggressive_abs",
            "mood_party_abs", "mood_relaxed_abs", "energy_essentia_fused", "energy_essentia_bucket", "notes",
        ]
        rows_for_csv = [{**item, "notes": " | ".join(item["notes"])} for item in payload_rows]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_for_csv)
        if not args.json:
            print(f"Wrote CSV report to {output_path}")
    return 0
