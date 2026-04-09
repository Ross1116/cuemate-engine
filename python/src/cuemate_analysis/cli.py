from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any
import numpy as np

from cuemate_analysis.analysis import (
    DspLaneResult,
    build_analysis_result,
    build_fast_analysis_result,
    compute_dsp_lane_result,
    utc_now,
)
from cuemate_analysis.config import RuntimeSettings, build_fast_analysis_signature, load_runtime_settings
from cuemate_analysis.database import Database
from cuemate_analysis.dj_import import list_dj_playlists, load_dj_playlist
from cuemate_analysis.essentia_semantic_backend import (
    EssentiaSemanticEstimate,
    build_essentia_semantic_manifest_signature,
    build_essentia_semantic_model_manifest,
    download_essentia_semantic_models,
    ensure_essentia_semantic_service,
    estimate_essentia_semantic_batch,
    purge_essentia_semantic_cache,
    resolve_essentia_semantic_model_root,
    resolve_essentia_semantic_service_name,
    resolve_essentia_semantic_service_port,
)
from cuemate_analysis.energy_experiments import analyze_energy_path
from cuemate_analysis.energy_features import EnergyFeatureVector, energy_consensus
from cuemate_analysis.ingest import (
    discover_audio_files,
    make_playlist_id,
    read_track_metadata,
    read_track_metadata_with_overrides,
)
from cuemate_analysis.key_backend import (
    KeyEstimate,
    MUSICALKEYCNN_POLICY_FULL_TRACK,
    ensure_musicalkeycnn_service,
    estimate_musicalkeycnn_keys,
    purge_musicalkeycnn_cache,
    resolve_musicalkeycnn_image_name,
    resolve_musicalkeycnn_model_path,
    resolve_musicalkeycnn_service_name,
    resolve_musicalkeycnn_service_port,
)
from cuemate_analysis.persistent_inference_cache import resolve_inference_cache_path
from cuemate_analysis.relative_context import (
    RelativePlaylistPreview,
    compute_relative_playlist_preview,
    preview_to_json,
    refresh_canonical_relative_playlist,
    row_to_relative_track_input,
)
from cuemate_analysis.tempo_backend import (
    TempoEstimate,
    ensure_tempocnn_service,
    estimate_tempocnn_bpms,
    purge_tempocnn_cache,
    resolve_tempocnn_image_name,
    resolve_tempocnn_model_path,
    resolve_tempocnn_service_name,
    resolve_tempocnn_service_port,
)

TEMPOCNN_PROGRESS_BATCH_SIZE = 8


@dataclass(frozen=True)
class PreparedTrack:
    row: object
    track: object
    position: int
    resolved_path: Path


def hash_file_identity(path: Path) -> str:
    if not path.is_file():
        return f"missing-{path.name}"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def resolve_expected_essentia_semantic_signature(settings) -> str:
    if not settings.analysis.essentia_semantics_enabled:
        return "disabled"
    try:
        manifest = build_essentia_semantic_model_manifest(
            settings.analysis.essentia_semantic_model_root,
            family_policy=settings.analysis.essentia_semantic_model_family_policy,
        )
        return build_essentia_semantic_manifest_signature(
            manifest,
            device=settings.analysis.essentia_semantic_device,
        )
    except Exception:
        return "missing"


def build_effective_analysis_signature(
    base_signature: str,
    *,
    tempocnn_model: str | None = None,
    tempocnn_accelerator: str = "auto",
    musicalkeycnn_model: str | None = None,
    musicalkeycnn_device: str = "auto",
    musicalkeycnn_policy: str = MUSICALKEYCNN_POLICY_FULL_TRACK,
    essentia_semantic_signature: str = "missing",
) -> str:
    tempo_model_hash = hash_file_identity(resolve_tempocnn_model_path(tempocnn_model))
    key_model_hash = hash_file_identity(resolve_musicalkeycnn_model_path(musicalkeycnn_model))
    return (
        f"{base_signature}"
        f"-tempo-tempocnn-{tempo_model_hash}-{tempocnn_accelerator}"
        f"-key-musicalkeycnn-{key_model_hash}-{musicalkeycnn_device}-{musicalkeycnn_policy}"
        f"-essentia-{essentia_semantic_signature}"
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


def estimate_has_persistent_cache_hit(estimate) -> bool:
    return any(str(note).startswith("Persistent inference cache hit.") for note in getattr(estimate, "notes", []))


def estimate_runner_device(estimate) -> str | None:
    details = getattr(estimate, "details", {}) or {}
    explicit = details.get("runner_device")
    if explicit:
        return str(explicit)
    tf_logical = details.get("tf_logical_gpu_count")
    if tf_logical is not None:
        try:
            return "cuda" if int(tf_logical) > 0 else "cpu"
        except (TypeError, ValueError):
            return None
    torch_logical = details.get("torch_logical_gpu_count")
    if torch_logical is not None:
        try:
            return "cuda" if int(torch_logical) > 0 else "cpu"
        except (TypeError, ValueError):
            return None
    return None


def estimate_gpu_summary(estimate) -> str | None:
    details = getattr(estimate, "details", {}) or {}
    tf_physical = details.get("tf_physical_gpu_count")
    tf_logical = details.get("tf_logical_gpu_count")
    if tf_physical is not None or tf_logical is not None:
        return f"tensorflow_gpus={tf_physical if tf_physical is not None else '?'} physical / {tf_logical if tf_logical is not None else '?'} logical"
    torch_physical = details.get("torch_physical_gpu_count")
    torch_logical = details.get("torch_logical_gpu_count")
    if torch_physical is not None or torch_logical is not None:
        return f"torch_gpus={torch_physical if torch_physical is not None else '?'} physical / {torch_logical if torch_logical is not None else '?'} logical"
    return None


def print_backend_diagnostics(label: str, estimates: list[object], *, requested_device: str | None = None) -> None:
    if not estimates:
        return
    total = len(estimates)
    available = sum(1 for estimate in estimates if getattr(estimate, "available", False))
    cache_hits = sum(1 for estimate in estimates if estimate_has_persistent_cache_hit(estimate))
    elapsed_values = [
        float(estimate.elapsed_ms)
        for estimate in estimates
        if getattr(estimate, "elapsed_ms", None) is not None
    ]
    runner_devices = sorted({device for estimate in estimates if (device := estimate_runner_device(estimate))})
    gpu_summaries = sorted({summary for estimate in estimates if (summary := estimate_gpu_summary(estimate))})
    sampling_mode_counts: dict[str, int] = {}
    sampling_trigger_counts: dict[str, int] = {}
    for estimate in estimates:
        details = getattr(estimate, "details", {}) or {}
        mode = details.get("sampling_mode")
        if mode:
            clean_mode = str(mode)
            sampling_mode_counts[clean_mode] = sampling_mode_counts.get(clean_mode, 0) + 1
        for trigger in details.get("sampling_triggers") or []:
            clean_trigger = str(trigger)
            sampling_trigger_counts[clean_trigger] = sampling_trigger_counts.get(clean_trigger, 0) + 1

    print(f"{label} diagnostics:")
    if requested_device:
        print(f"- requested_device: {requested_device}")
    print(f"- results: {available}/{total} available")
    print(f"- persistent_cache_hits: {cache_hits}/{total}")
    if elapsed_values:
        print(f"- avg_elapsed_ms: {sum(elapsed_values) / len(elapsed_values):.1f}")
    if runner_devices:
        print(f"- runner_device(s): {', '.join(runner_devices)}")
    for summary in gpu_summaries:
        print(f"- {summary}")
    if sampling_mode_counts:
        mode_summary = ", ".join(f"{name}={count}" for name, count in sorted(sampling_mode_counts.items()))
        print(f"- sampling_modes: {mode_summary}")
    if sampling_trigger_counts:
        trigger_summary = ", ".join(
            f"{name}={count}" for name, count in sorted(sampling_trigger_counts.items())
        )
        print(f"- sampling_triggers: {trigger_summary}")


def _fmt_float(value, digits: int = 3) -> str:
    if value is None:
        return "None"
    return f"{float(value):.{digits}f}"


def format_full_analysis_summary(result) -> str:
    parts: list[str] = []

    # Core resolved outputs
    parts.append(f"bpm={_fmt_float(result.bpm, 1)}")
    parts.append(f"bpm_conf={_fmt_float(result.bpm_confidence, 2)}")
    parts.append(f"bpm_source={result.bpm_source}")

    parts.append(f"key={result.key}")
    parts.append(f"key_conf={_fmt_float(result.key_confidence, 2)}")
    parts.append(f"key_source={result.key_source}")

    parts.append(f"time_sig={result.time_signature}")
    parts.append(f"time_sig_conf={_fmt_float(result.time_signature_confidence, 2)}")

    # Energy / DSP
    parts.append(f"energy_abs={_fmt_float(result.energy_abs)}")
    parts.append(f"energy_heuristic_abs={_fmt_float(result.energy_heuristic_abs)}")
    parts.append(f"energy_sustained={_fmt_float(result.energy_sustained)}")
    parts.append(f"energy_peak={_fmt_float(result.energy_peak)}")
    parts.append(f"loudness_lufs={_fmt_float(result.loudness_lufs, 2)}")
    parts.append(f"loudness_norm={_fmt_float(result.loudness_norm)}")
    parts.append(f"bass_abs={_fmt_float(result.bass_abs)}")
    # deferred (dsp full analysis) for now
    # parts.append(f"drums_abs={_fmt_float(result.drums_abs)}")
    # parts.append(f"harmonic_abs={_fmt_float(result.harmonic_abs)}")
    # parts.append(f"groove_abs={_fmt_float(result.groove_abs)}")

    # Essentia semantics, only when present
    if result.danceability_abs is not None:
        parts.append(f"danceability_abs={_fmt_float(result.danceability_abs)}")
    if result.arousal_abs is not None:
        parts.append(f"arousal_abs={_fmt_float(result.arousal_abs)}")
    if result.valence_abs is not None:
        parts.append(f"valence_abs={_fmt_float(result.valence_abs)}")
    if result.mood_aggressive_abs is not None:
        parts.append(f"mood_aggressive_abs={_fmt_float(result.mood_aggressive_abs)}")
    if result.mood_party_abs is not None:
        parts.append(f"mood_party_abs={_fmt_float(result.mood_party_abs)}")
    if result.mood_relaxed_abs is not None:
        parts.append(f"mood_relaxed_abs={_fmt_float(result.mood_relaxed_abs)}")
    if result.energy_essentia_fused is not None:
        parts.append(f"energy_essentia_fused={_fmt_float(result.energy_essentia_fused)}")
    if result.energy_essentia_bucket is not None:
        parts.append(f"energy_essentia_bucket={result.energy_essentia_bucket}")
    if result.essentia_semantic_source is not None:
        parts.append(f"essentia_source={result.essentia_semantic_source}")

    return " :: ".join(parts)

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


def prefetch_essentia_semantic_estimates(rows, settings) -> dict[Path, EssentiaSemanticEstimate]:
    if not rows or not settings.analysis.essentia_semantics_enabled:
        return {}
    paths, auxiliary_features_by_path = build_essentia_auxiliary_features_by_path(rows)
    return estimate_essentia_semantic_batch(
        paths,
        model_root=settings.analysis.essentia_semantic_model_root,
        image_name=settings.analysis.essentia_semantic_image,
        device=settings.analysis.essentia_semantic_device,
        family_policy=settings.analysis.essentia_semantic_model_family_policy,
        auxiliary_features_by_path=auxiliary_features_by_path,
        default_excerpt_seconds=settings.analysis.essentia_semantic_default_excerpt_seconds,
        multisample_excerpt_seconds=settings.analysis.essentia_semantic_multisample_excerpt_seconds,
        mismatch_threshold=settings.analysis.essentia_semantic_trigger_mismatch_threshold,
        confidence_threshold=settings.analysis.essentia_semantic_trigger_confidence_threshold,
        structure_rms_cv_threshold=settings.analysis.essentia_semantic_trigger_structure_rms_cv,
        outlier_zscore_threshold=settings.analysis.essentia_semantic_trigger_outlier_zscore,
    )


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


def build_failed_tempo_estimate(error_message: str) -> TempoEstimate:
    return TempoEstimate(
        backend="tempocnn",
        bpm=None,
        confidence=None,
        elapsed_ms=None,
        details={},
        notes=[error_message],
        available=False,
    )


def build_failed_key_estimate(error_message: str) -> KeyEstimate:
    return KeyEstimate(
        backend="musicalkeycnn",
        key=None,
        key_number=None,
        key_letter=None,
        confidence=None,
        elapsed_ms=None,
        details={},
        notes=[error_message],
        available=False,
    )


def build_failed_essentia_estimate(error_message: str) -> EssentiaSemanticEstimate:
    return EssentiaSemanticEstimate(
        backend="essentia_semantics",
        danceability_abs=None,
        arousal_abs=None,
        valence_abs=None,
        mood_aggressive_abs=None,
        mood_party_abs=None,
        mood_relaxed_abs=None,
        semantic_confidence=None,
        energy_essentia_fused=None,
        energy_essentia_bucket=None,
        elapsed_ms=None,
        details={},
        notes=[error_message],
        available=False,
    )


def prepare_analysis_chunk(rows) -> tuple[list[PreparedTrack], list[tuple[object, str]]]:
    prepared_tracks: list[PreparedTrack] = []
    failures: list[tuple[object, str]] = []
    for row in rows:
        try:
            track = track_from_playlist_row(row)
            prepared_tracks.append(
                PreparedTrack(
                    row=row,
                    track=track,
                    position=int(row["position"]),
                    resolved_path=track.file_path.resolve(),
                )
            )
        except Exception as exc:
            failures.append((row, str(exc)))
    return prepared_tracks, failures


def build_essentia_auxiliary_features_by_path(
    items,
    *,
    dsp_results: dict[Path, DspLaneResult] | None = None,
) -> tuple[list[Path], dict[Path, dict[str, object]]]:
    auxiliary_features_by_path: dict[Path, dict[str, object]] = {}
    paths: list[Path] = []
    proxy_values: dict[Path, float] = {}
    structure_values: dict[Path, float] = {}
    peak_time_ratios: dict[Path, float] = {}
    for item in items:
        row = item.row if isinstance(item, PreparedTrack) else item
        path = (item.resolved_path if isinstance(item, PreparedTrack) else Path(row["file_path"]).resolve())
        paths.append(path)
        row_keys = set(row.keys()) if hasattr(row, "keys") else set()

        def optional_value(key: str):
            if row_keys and key not in row_keys:
                return None
            return row[key]

        dsp_result = (dsp_results or {}).get(path)
        loudness_norm = optional_value("loudness_norm")
        bass_abs = optional_value("bass_abs")
        duration_seconds = optional_value("duration_seconds")
        if dsp_result is not None and dsp_result.available:
            if loudness_norm is None and dsp_result.loudness is not None:
                loudness_norm = dsp_result.loudness.get("loudness_norm")
            if bass_abs is None:
                bass_abs = dsp_result.bass_abs
            if duration_seconds is None and dsp_result.artifacts is not None:
                duration_seconds = dsp_result.artifacts.duration_seconds
            if dsp_result.energy is not None and dsp_result.loudness is not None:
                proxy_values[path] = float(
                    (0.75 * float(dsp_result.energy.get("energy_abs", 0.0)))
                    + (0.15 * float(dsp_result.loudness.get("loudness_norm", 0.0)))
                    + (0.10 * float(dsp_result.bass_abs or 0.0))
                )
            if dsp_result.artifacts is not None and dsp_result.artifacts.rms.size:
                # Use coarse section-level RMS variation instead of frame-level jitter.
                # Frame-level CV fires on almost every energetic rhythmic track and
                # makes adaptive semantics effectively multisample-by-default.
                section_frames = max(
                    8,
                    int(round((3.0 * float(dsp_result.artifacts.sr)) / float(dsp_result.artifacts.hop_length))),
                )
                rms = dsp_result.artifacts.rms
                pad = (-len(rms)) % section_frames
                coarse = np.pad(rms, (0, pad), mode="edge") if pad else rms
                coarse = coarse.reshape(-1, section_frames).mean(axis=1)
                coarse_mean = float(coarse.mean())
                if coarse_mean > 1e-9:
                    structure_values[path] = float(coarse.std() / coarse_mean)
                if dsp_result.artifacts.duration_seconds > 1e-9:
                    coarse_index = int(np.argmax(coarse))
                    coarse_count = max(len(coarse), 1)
                    peak_time_ratios[path] = float((coarse_index + 0.5) / coarse_count)

        auxiliary_features_by_path[path] = {
            "loudness_norm": loudness_norm,
            "drums_abs": optional_value("drums_abs"),
            "groove_abs": optional_value("groove_abs"),
            "bass_abs": bass_abs,
            "duration_seconds": duration_seconds,
        }

    if proxy_values:
        proxy_mean = sum(proxy_values.values()) / len(proxy_values)
        proxy_variance = sum((value - proxy_mean) ** 2 for value in proxy_values.values()) / max(len(proxy_values), 1)
        proxy_std = proxy_variance ** 0.5
        for path, payload in auxiliary_features_by_path.items():
            payload["dsp_proxy_energy"] = proxy_values.get(path)
            payload["structure_rms_cv"] = structure_values.get(path)
            payload["peak_time_ratio"] = peak_time_ratios.get(path)
            value = proxy_values.get(path)
            if value is not None and proxy_std > 1e-9:
                payload["playlist_outlier_score"] = abs(value - proxy_mean) / proxy_std
            else:
                payload["playlist_outlier_score"] = 0.0
    return paths, auxiliary_features_by_path


def run_dsp_lane(prepared_tracks: list[PreparedTrack], settings, analysis_mode: str) -> dict[Path, DspLaneResult]:
    if not prepared_tracks:
        return {}
    worker_count = max(1, min(settings.analysis.dsp_workers, len(prepared_tracks)))
    results: dict[Path, DspLaneResult] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(compute_dsp_lane_result, item.track, settings, analysis_mode): item
            for item in prepared_tracks
        }
        for future, item in future_map.items():
            try:
                results[item.resolved_path] = future.result()
            except Exception as exc:
                results[item.resolved_path] = DspLaneResult.failure(track_id=item.track.id, error=exc)
    return results


def run_tempo_lane(prepared_tracks: list[PreparedTrack], settings) -> dict[Path, TempoEstimate]:
    if not prepared_tracks:
        return {}
    results: dict[Path, TempoEstimate] = {}
    for chunk in chunk_items(prepared_tracks, settings.analysis.tempo_chunk_size):
        chunk_paths = [item.resolved_path for item in chunk]
        try:
            results.update(estimate_tempocnn_bpms(chunk_paths))
        except Exception as exc:
            failure = f"TempoCNN lane failed: {exc}"
            for path in chunk_paths:
                results[path] = build_failed_tempo_estimate(failure)
    return results


def run_key_lane(prepared_tracks: list[PreparedTrack], settings) -> dict[Path, KeyEstimate]:
    if not prepared_tracks:
        return {}
    results: dict[Path, KeyEstimate] = {}
    for chunk in chunk_items(prepared_tracks, settings.analysis.key_chunk_size):
        chunk_paths = [item.resolved_path for item in chunk]
        try:
            results.update(
                estimate_musicalkeycnn_keys(
                    chunk_paths,
                    model_path=settings.analysis.key_model_path,
                    device=settings.analysis.key_device,
                    policy=settings.analysis.key_policy,
                )
            )
        except Exception as exc:
            failure = f"MusicalKeyCNN lane failed: {exc}"
            for path in chunk_paths:
                results[path] = build_failed_key_estimate(failure)
    return results


def run_essentia_lane(
    prepared_tracks: list[PreparedTrack],
    settings,
    analysis_mode: str,
    *,
    dsp_results: dict[Path, DspLaneResult] | None = None,
) -> dict[Path, EssentiaSemanticEstimate]:
    if analysis_mode != "full" or not settings.analysis.essentia_semantics_enabled or not prepared_tracks:
        return {}
    results: dict[Path, EssentiaSemanticEstimate] = {}
    for chunk in chunk_items(prepared_tracks, settings.analysis.essentia_chunk_size):
        chunk_paths, auxiliary_features_by_path = build_essentia_auxiliary_features_by_path(
            chunk,
            dsp_results=dsp_results,
        )
        try:
            results.update(
                estimate_essentia_semantic_batch(
                    chunk_paths,
                    model_root=settings.analysis.essentia_semantic_model_root,
                    image_name=settings.analysis.essentia_semantic_image,
                    device=settings.analysis.essentia_semantic_device,
                    family_policy=settings.analysis.essentia_semantic_model_family_policy,
                    auxiliary_features_by_path=auxiliary_features_by_path,
                    default_excerpt_seconds=settings.analysis.essentia_semantic_default_excerpt_seconds,
                    multisample_excerpt_seconds=settings.analysis.essentia_semantic_multisample_excerpt_seconds,
                    mismatch_threshold=settings.analysis.essentia_semantic_trigger_mismatch_threshold,
                    confidence_threshold=settings.analysis.essentia_semantic_trigger_confidence_threshold,
                    structure_rms_cv_threshold=settings.analysis.essentia_semantic_trigger_structure_rms_cv,
                    outlier_zscore_threshold=settings.analysis.essentia_semantic_trigger_outlier_zscore,
                )
            )
        except Exception as exc:
            failure = f"Essentia semantics lane failed: {exc}"
            for path in chunk_paths:
                results[path] = build_failed_essentia_estimate(failure)
    return results


def estimate_lane_status(estimate, *, disabled: bool = False) -> str:
    if disabled:
        return "disabled"
    if estimate is None:
        return "failed"
    if getattr(estimate, "available", False):
        return "cached" if estimate_has_persistent_cache_hit(estimate) else "succeeded"
    return "failed"


def build_job_timing_breakdown(
    *,
    args,
    settings,
    effective_analysis_signature: str,
    expected_essentia_semantic_signature: str,
    lane_status: dict[str, str] | None = None,
    degraded: bool = False,
    missing_lanes: list[str] | None = None,
    skipped: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "analysis_mode": args.analysis_mode,
        "analysis_signature": effective_analysis_signature,
        "config_signature": settings.config_signature,
        "tempo_backend": "tempocnn",
        "tempocnn_accelerator": "auto",
        "key_backend": "musicalkeycnn",
        "musicalkeycnn_model": settings.analysis.key_model_path,
        "musicalkeycnn_device": settings.analysis.key_device,
        "musicalkeycnn_policy": settings.analysis.key_policy,
        "essentia_semantics_enabled": settings.analysis.essentia_semantics_enabled,
        "essentia_semantic_signature": expected_essentia_semantic_signature,
        "skipped": skipped,
    }
    if lane_status is not None:
        payload["lane_status"] = lane_status
        payload["degraded"] = degraded
        payload["missing_lanes"] = missing_lanes or []
    return payload


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
    expected_essentia_semantic_signature: str,
) -> bool:
    if force:
        return False
    essentia_current = True
    if analysis_mode == "full" and expected_essentia_semantic_signature not in {"disabled", "missing"}:
        essentia_current = (
            row["essentia_semantic_signature"] == expected_essentia_semantic_signature
            and row["energy_essentia_fused"] is not None
        )
    return (
        row["source_file_hash"] == track.file_hash
        and row["analysis_signature"] == effective_analysis_signature
        and row["analysis_mode"] == analysis_mode
        and row["config_signature"] == config_signature
        and essentia_current
    )


def should_skip_fast_analysis(
    row,
    track,
    *,
    effective_analysis_signature: str,
    config_signature: str,
    force: bool,
) -> bool:
    if force:
        return False

    row_keys = set(row.keys()) if hasattr(row, "keys") else set()

    def optional_value(key: str):
        if row_keys and key not in row_keys:
            return None
        return row[key]

    return (
        optional_value("fast_source_file_hash") == track.file_hash
        and optional_value("fast_analysis_signature") == effective_analysis_signature
        and optional_value("fast_config_signature") == config_signature
        and optional_value("fast_bpm") is not None
        and optional_value("fast_key") is not None
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
        choices=["fast_pass", "staged", "full"],
        default="staged",
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
        help="Inspect canonical playlist-relative context from persisted relative tables.",
    )
    analyze_relative_playlist_parser.add_argument("--playlist", required=True, help="Playlist name to analyze.")
    analyze_relative_playlist_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional output limit for a smaller playlist slice preview.",
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
        choices=["canonical", "heuristic_legacy", "essentia_fused"],
        default=None,
        help="Choose which absolute-energy source to use for relative scaling. "
             "'canonical' uses the fused intensity score (energy_abs). "
             "'heuristic_legacy' uses the old DSP-only heuristic score. "
             "'essentia_fused' is a deprecated alias for 'canonical'.",
    )

    refresh_relative_playlist_parser = subparsers.add_parser(
        "refresh-relative-playlist",
        help="Refresh and persist canonical playlist-relative context for one playlist.",
    )
    refresh_relative_playlist_parser.add_argument("--playlist", required=True, help="Playlist name to refresh.")
    refresh_relative_playlist_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional output limit for the printed preview after refresh.",
    )
    refresh_relative_playlist_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the refreshed relative-context payload as JSON.",
    )
    refresh_relative_playlist_parser.add_argument(
        "--output",
        help="Optional output path for a CSV report.",
    )

    run_worker_parser = subparsers.add_parser(
        "run-analysis-worker",
        help="Process pending staged enrichment jobs from the local analysis queue.",
    )
    run_worker_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum pending jobs to process in one pass.",
    )
    run_worker_parser.add_argument(
        "--print-backend-diagnostics",
        action="store_true",
        help="Print backend diagnostics after processing jobs.",
    )

    prewarm_parser = subparsers.add_parser(
        "prewarm-model-services",
        help="Start and prewarm the shared TF/Essentia and MusicalKeyCNN services.",
    )
    prewarm_parser.add_argument(
        "--path",
        help="Optional audio file to use for warmup requests. Falls back to service health warmup when omitted.",
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

    download_essentia_models_parser = subparsers.add_parser(
        "download-essentia-semantic-models",
        help="Download the Essentia semantic model bundle into the local model root.",
    )
    download_essentia_models_parser.add_argument(
        "--model-root",
        help="Optional model root override.",
    )
    download_essentia_models_parser.add_argument(
        "--family-policy",
        choices=["best_per_task", "musicnn_only"],
        help="Optional family policy override for the manifest download.",
    )

    analyze_essentia_playlist_parser = subparsers.add_parser(
        "analyze-essentia-playlist",
        help="Inspect Essentia semantic predictions for a playlist without persisting results.",
    )
    analyze_essentia_playlist_parser.add_argument("--playlist", required=True, help="Playlist name to analyze.")
    analyze_essentia_playlist_parser.add_argument("--limit", type=int, default=0, help="Optional track limit.")
    analyze_essentia_playlist_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    analyze_essentia_playlist_parser.add_argument("--output", help="Optional output CSV path.")

    purge_cache_parser = subparsers.add_parser(
        "purge-model-cache",
        help="Purge persisted model inference caches; keep warm services unless --clear-warm-services is set.",
    )
    purge_cache_parser.add_argument(
        "--backend",
        choices=["all", "tempocnn", "musicalkeycnn", "essentia_semantics"],
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
    purge_cache_parser.add_argument(
        "--clear-warm-services",
        action="store_true",
        help="Also remove the warm Docker model services so the next run is a true cold start.",
    )

    recommend_next_parser = subparsers.add_parser(
        "recommend-next",
        help="Recommend next tracks from a playlist given a current track.",
    )
    recommend_next_parser.add_argument("--playlist", required=True, help="Playlist name.")
    recommend_next_parser.add_argument(
        "--current-track",
        default=None,
        help="Track ID to recommend from. Omit to use the first analyzed track in the playlist.",
    )
    recommend_next_parser.add_argument(
        "--target",
        choices=["maintain", "build", "reset", "jump", "contrast"],
        default="maintain",
        help="Desired move type / energy direction (default: maintain).",
    )
    recommend_next_parser.add_argument(
        "--max-per-lane",
        type=int,
        default=3,
        help="Maximum candidates per lane (default: 3).",
    )
    recommend_next_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON to stdout.",
    )

    score_pair_parser = subparsers.add_parser(
        "score-pair",
        help="Score a specific current→candidate track pair and show the full breakdown.",
    )
    score_pair_parser.add_argument("--playlist", required=True, help="Playlist name.")
    score_pair_parser.add_argument("--current", required=True, help="Current track ID.")
    score_pair_parser.add_argument("--candidate", required=True, help="Candidate track ID.")
    score_pair_parser.add_argument(
        "--target",
        choices=["maintain", "build", "reset", "jump", "contrast"],
        default="maintain",
        help="Move target for weight resolution (default: maintain).",
    )
    score_pair_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON to stdout.",
    )

    inspect_weights_parser = subparsers.add_parser(
        "inspect-scoring-weights",
        help="Show effective scoring weights for a playlist.",
    )
    inspect_weights_parser.add_argument("--playlist", required=True, help="Playlist name.")
    inspect_weights_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON to stdout.",
    )

    inspect_metadata_parser = subparsers.add_parser(
        "inspect-scoring-metadata",
        help="Show active scoring metadata and optional artifact compatibility status.",
    )
    inspect_metadata_parser.add_argument(
        "--analysis-signature",
        help="Optional artifact analysis_signature to compare against the active scoring metadata.",
    )
    inspect_metadata_parser.add_argument(
        "--config-signature",
        help="Optional artifact config_signature to compare against the active scoring metadata.",
    )
    inspect_metadata_parser.add_argument(
        "--scoring-contract-id-at-analysis",
        help="Optional artifact scoring_contract_id_at_analysis to compare against the active scoring metadata.",
    )
    inspect_metadata_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON to stdout.",
    )

    serve_scoring_parser = subparsers.add_parser(
        "serve-scoring",
        help="Run the Python scoring gRPC service.",
    )
    serve_scoring_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1).",
    )
    serve_scoring_parser.add_argument(
        "--port",
        type=int,
        default=47834,
        help="Port to bind (default: 47834).",
    )
    serve_scoring_parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="gRPC worker thread count (default: 8).",
    )

    benchmark_dsp_parser = subparsers.add_parser(
        "benchmark-dsp",
        help="Profile DSP pipeline substep timings for a playlist or explicit file paths.",
    )
    benchmark_dsp_parser.add_argument(
        "--playlist",
        default=None,
        help="Playlist name to benchmark (reads file paths from DB).",
    )
    benchmark_dsp_parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Explicit file path to benchmark. Can be passed multiple times.",
    )
    benchmark_dsp_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional track limit for a faster subset pass.",
    )
    benchmark_dsp_parser.add_argument(
        "--output",
        help="Output path for CSV or JSON report (extension determines format).",
    )
    benchmark_dsp_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit summary as JSON to stdout.",
    )

    return parser


def build_fast_job_timing_breakdown(
    *,
    settings,
    effective_analysis_signature: str,
) -> dict[str, object]:
    return {
        "job_kind": "fast_pass",
        "analysis_signature": effective_analysis_signature,
        "config_signature": settings.config_signature,
        "tempo_backend": "tempocnn",
        "key_backend": "musicalkeycnn",
        "stage_state": "fast_ready",
    }


def _refresh_relative_for_playlists(database: Database, settings, playlist_ids: set[str]) -> None:
    for playlist_id in sorted(playlist_ids):
        playlist_name = database.get_playlist_name_by_id(playlist_id)
        if not playlist_name:
            continue
        rel_rows = database.get_playlist_relative_inputs(playlist_name)
        rel_inputs = [row_to_relative_track_input(row) for row in rel_rows]
        refresh_canonical_relative_playlist(
            rel_inputs,
            settings,
            playlist_name=playlist_name,
            playlist_id=playlist_id,
            database=database,
            timestamp=utc_now(),
        )
        print(f"Refreshed canonical relative data for '{playlist_name}'.")


def _process_enrichment_batch(
    *,
    pending_items: list[dict[str, object]],
    settings,
    database: Database,
    effective_analysis_signature: str,
    expected_essentia_semantic_signature: str,
    print_progress: bool = True,
) -> tuple[int, list[str], list[DspLaneResult], list[TempoEstimate], list[KeyEstimate], list[EssentiaSemanticEstimate]]:
    if not pending_items:
        return 0, [], [], [], [], []

    pending_prepared = [item["prepared"] for item in pending_items]
    with ThreadPoolExecutor(max_workers=2) as executor:
        dsp_future = executor.submit(run_dsp_lane, pending_prepared, settings, "full")
        prefetched_tempocnn_estimates = run_tempo_lane(pending_prepared, settings)
        prefetched_musicalkeycnn_estimates = run_key_lane(pending_prepared, settings)
        prefetched_dsp_results = dsp_future.result()
        prefetched_essentia_semantic_estimates = run_essentia_lane(
            pending_prepared,
            settings,
            "full",
            dsp_results=prefetched_dsp_results,
        )

    processed = 0
    updated_track_ids: list[str] = []
    dsp_diagnostics: list[DspLaneResult] = []
    tempo_diagnostics: list[TempoEstimate] = []
    key_diagnostics: list[KeyEstimate] = []
    essentia_diagnostics: list[EssentiaSemanticEstimate] = []

    for item in pending_items:
        prepared = item["prepared"]
        track = prepared.track
        index = int(item.get("display_index", prepared.position))
        total = int(item.get("display_total", len(pending_items)))
        start_time = float(item["start_time"])
        job_id = int(item["job_id"])
        resolved_path = prepared.resolved_path

        dsp_result = prefetched_dsp_results.get(resolved_path)
        tempo_estimate = prefetched_tempocnn_estimates.get(resolved_path)
        key_estimate = prefetched_musicalkeycnn_estimates.get(resolved_path)
        essentia_estimate = prefetched_essentia_semantic_estimates.get(resolved_path)

        if dsp_result is not None:
            dsp_diagnostics.append(dsp_result)
        if tempo_estimate is not None:
            tempo_diagnostics.append(tempo_estimate)
        if key_estimate is not None:
            key_diagnostics.append(key_estimate)
        if essentia_estimate is not None:
            essentia_diagnostics.append(essentia_estimate)

        try:
            if dsp_result is None or not dsp_result.available:
                raise ValueError(dsp_result.error if dsp_result is not None and dsp_result.error else "DSP lane failed.")

            result = build_analysis_result(
                track,
                settings,
                "full",
                dsp_result=dsp_result,
                tempo_backend="tempocnn",
                tempocnn_accelerator="auto",
                prefetched_tempocnn_estimate=tempo_estimate,
                key_backend="musicalkeycnn",
                musicalkeycnn_model=settings.analysis.key_model_path,
                musicalkeycnn_device=settings.analysis.key_device,
                musicalkeycnn_policy=settings.analysis.key_policy,
                prefetched_musicalkeycnn_estimate=key_estimate,
                prefetched_essentia_semantic_estimate=essentia_estimate,
                analysis_signature=effective_analysis_signature,
            )
            lane_status = {
                "dsp": estimate_lane_status(dsp_result),
                "tempocnn": estimate_lane_status(tempo_estimate),
                "musicalkeycnn": estimate_lane_status(key_estimate),
                "essentia_semantics": estimate_lane_status(
                    essentia_estimate,
                    disabled=not settings.analysis.essentia_semantics_enabled,
                ),
            }
            missing_lanes = [lane for lane, status in lane_status.items() if status == "failed"]
            degraded = bool(missing_lanes)
            database.upsert_track_features(result)
            updated_track_ids.append(track.id)
            duration_seconds = round(time.perf_counter() - start_time, 3)
            database.mark_analysis_job_completed(
                job_id,
                duration_seconds,
                build_job_timing_breakdown(
                    args=argparse.Namespace(analysis_mode="full"),
                    settings=settings,
                    effective_analysis_signature=effective_analysis_signature,
                    expected_essentia_semantic_signature=expected_essentia_semantic_signature,
                    lane_status=lane_status,
                    degraded=degraded,
                    missing_lanes=missing_lanes,
                ),
                utc_now(),
            )
            processed += 1
            if essentia_estimate is not None and not getattr(essentia_estimate, "available", False):
                print(
                    f"[{index}/{total}] essentia_debug {track.id}: "
                    + " | ".join(str(note) for note in (essentia_estimate.notes or [])),
                    file=sys.stderr,
                )
            if print_progress:
                degraded_suffix = f" (degraded: missing {', '.join(missing_lanes)})" if degraded else ""
                print(
                    f"[{index}/{total}] analyzed {track.id} ({track.title}){degraded_suffix} :: "
                    f"{format_full_analysis_summary(result)}"
                )
        except Exception as exc:
            duration_seconds = round(time.perf_counter() - start_time, 3)
            database.mark_analysis_job_failed(job_id, str(exc), duration_seconds, utc_now())
            if print_progress:
                print(f"[{index}/{total}] failed {track.id}: {exc}", file=sys.stderr)

    return processed, updated_track_ids, dsp_diagnostics, tempo_diagnostics, key_diagnostics, essentia_diagnostics


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
    expected_essentia_semantic_signature = resolve_expected_essentia_semantic_signature(settings)
    effective_analysis_signature = build_effective_analysis_signature(
        settings.analysis_signature,
        tempocnn_accelerator="auto",
        musicalkeycnn_model=settings.analysis.key_model_path,
        musicalkeycnn_device=settings.analysis.key_device,
        musicalkeycnn_policy=settings.analysis.key_policy,
        essentia_semantic_signature=expected_essentia_semantic_signature,
    )
    fast_analysis_signature = build_fast_analysis_signature(settings)

    with Database(settings.database_path) as database:
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

        for chunk in chunk_items(rows, settings.analysis.full_chunk_size):
            prepared_tracks, prepare_failures = prepare_analysis_chunk(chunk)
            for row, error_message in prepare_failures:
                index = int(row["position"])
                print(f"[{index}/{total}] failed {row['track_id']}: {error_message}", file=sys.stderr)

            if not prepared_tracks:
                continue

            prefetched_tempocnn_estimates = run_tempo_lane(prepared_tracks, settings)
            prefetched_musicalkeycnn_estimates = run_key_lane(prepared_tracks, settings)

            for prepared in prepared_tracks:
                row = prepared.row
                track = prepared.track
                index = prepared.position

                database.upsert_track(track, utc_now())

                if not should_skip_fast_analysis(
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
                            build_fast_job_timing_breakdown(
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

                if should_skip_analysis(
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
            ) = _process_enrichment_batch(
                pending_items=pending_enrichment_items,
                settings=settings,
                database=database,
                effective_analysis_signature=effective_analysis_signature,
                expected_essentia_semantic_signature=expected_essentia_semantic_signature,
                print_progress=True,
            )

    print(f"Completed playlist analysis for '{playlist_name}'. Updated {processed} track(s).")
    if args.analysis_mode == "full":
        print_backend_diagnostics("DSP local lane", dsp_diagnostics, requested_device="cpu")
        print_backend_diagnostics("TempoCNN", tempo_diagnostics, requested_device="auto")
        print_backend_diagnostics("MusicalKeyCNN", key_diagnostics, requested_device=settings.analysis.key_device)
        print_backend_diagnostics(
            "Essentia semantics",
            essentia_diagnostics,
            requested_device=settings.analysis.essentia_semantic_device,
        )

    if args.analysis_mode == "full" and processed > 0 and playlist_id:
        with Database(settings.database_path) as post_db:
            try:
                if hasattr(post_db, "get_playlist_name_by_id"):
                    _refresh_relative_for_playlists(post_db, settings, {playlist_id})
            except Exception as exc:
                print(f"Warning: canonical relative refresh failed for '{playlist_name}': {exc}", file=sys.stderr)

            if updated_track_ids and hasattr(post_db, "get_playlists_containing_tracks") and hasattr(post_db, "mark_playlists_stale"):
                linked = post_db.get_playlists_containing_tracks(updated_track_ids)
                stale_targets = [pid for pid in linked if pid != playlist_id]
                if stale_targets:
                    post_db.mark_playlists_stale(stale_targets, "absolute_track_changed", utc_now())
                    print(f"Marked {len(stale_targets)} linked playlist(s) stale.")

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
            elif row["fast_analyzed_at"]:
                summary = f"fast_ready | {row['fast_bpm']:.1f} BPM | {row['fast_key']}"
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
    details["analysis_state"] = (
        "full_ready" if details.get("analyzed_at")
        else "fast_ready" if details.get("fast_analyzed_at")
        else "metadata_only"
    )
    details["energy_summary"] = {
        "heuristic": details.get("energy_abs"),
        "heuristic_legacy": details.get("energy_heuristic_abs"),
        "essentia_fused": details.get("energy_essentia_fused"),
        "essentia_bucket": details.get("energy_essentia_bucket"),
    }
    details["essentia_semantics"] = {
        "danceability_abs": details.get("danceability_abs"),
        "arousal_abs": details.get("arousal_abs"),
        "valence_abs": details.get("valence_abs"),
        "mood_aggressive_abs": details.get("mood_aggressive_abs"),
        "mood_party_abs": details.get("mood_party_abs"),
        "mood_relaxed_abs": details.get("mood_relaxed_abs"),
        "energy_essentia_fused": details.get("energy_essentia_fused"),
        "energy_essentia_bucket": details.get("energy_essentia_bucket"),
        "essentia_semantic_signature": details.get("essentia_semantic_signature"),
        "essentia_semantic_source": details.get("essentia_semantic_source"),
        "essentia_semantic_inferred_at": details.get("essentia_semantic_inferred_at"),
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
    print_backend_diagnostics("TempoCNN", [estimate], requested_device="auto")
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
    print_backend_diagnostics("TempoCNN", [bpm_estimate], requested_device="auto")
    print_backend_diagnostics("MusicalKeyCNN", [key_estimate], requested_device=settings.analysis.key_device)
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
    diagnostics_estimates: list[TempoEstimate] = []
    total = len(rows)
    if not args.json:
        print(f"Playlist '{args.playlist}' BPM pass with backend tempocnn")

    for chunk in chunk_items(rows, settings.analysis.tempo_chunk_size):
        prefetched_tempocnn_estimates = estimate_tempocnn_bpms([Path(row["file_path"]) for row in chunk])

        for row in chunk:
            index = int(row["position"])
            path = Path(row["file_path"]).resolve()
            estimate = prefetched_tempocnn_estimates[path]
            diagnostics_estimates.append(estimate)
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
    else:
        print_backend_diagnostics("TempoCNN", diagnostics_estimates, requested_device="auto")
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
    bpm_diagnostics: list[TempoEstimate] = []
    key_diagnostics: list[KeyEstimate] = []
    total = len(rows)
    if not args.json:
        print(f"Playlist '{args.playlist}' BPM+key pass with TempoCNN + MusicalKeyCNN")

    bpm_key_chunk_size = min(settings.analysis.tempo_chunk_size, settings.analysis.key_chunk_size)
    for chunk in chunk_items(rows, bpm_key_chunk_size):
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
            bpm_diagnostics.append(bpm_estimate)
            key_diagnostics.append(key_estimate)
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
    else:
        print_backend_diagnostics("TempoCNN", bpm_diagnostics, requested_device="auto")
        print_backend_diagnostics("MusicalKeyCNN", key_diagnostics, requested_device=settings.analysis.key_device)
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


def _run_canonical_relative_preview_and_print(
    preview: "RelativePlaylistPreview",
    *,
    playlist_name: str,
    energy_source: str,
    args: argparse.Namespace,
    abs_rows: list,
) -> None:
    """Print relative playlist results in text or JSON, shared by canonical and legacy paths."""
    if args.json:
        print(preview_to_json(preview))
        return

    stats = preview.playlist_stats
    source_label = "persisted canonical" if energy_source == "canonical" else energy_source
    print(
        f"Playlist '{playlist_name}' relative-context :: "
        f"status={stats.status} :: analyzed={stats.track_count_analyzed}/{stats.track_count_total} :: "
        f"eligible={stats.eligible_track_count}"
    )
    print(f"Energy source: {source_label}")
    print(f"Relative signature: {stats.relative_signature}")
    if stats.adapted_weights is not None and stats.adaptation_strength is not None:
        print(f"Adapted weights enabled :: adaptation_strength={stats.adaptation_strength:.2f}")
    else:
        print("Adapted weights skipped")
    for note in stats.weight_adaptation_notes:
        print(f"- {note}")
    if energy_source == "canonical" and abs_rows:
        essentia_count = sum(1 for row in abs_rows if row["energy_essentia_fused"] is not None)
        if essentia_count:
            print(f"Canonical energy diagnostics: essentia_fused_rows={essentia_count}/{len(abs_rows)}")
    for track in preview.tracks:
        title = track.title or Path(track.file_path).stem
        print(
            f"[{track.position}/{stats.track_count_total}] {title} :: "
            f"energy_rel={track.energy_rel:.3f} :: {track.intensity_band} :: "
            f"{', '.join(track.role_hints)}"
        )


def _write_relative_csv(preview: "RelativePlaylistPreview", output_path: Path, args: argparse.Namespace) -> None:
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
        "energy_rel", "bass_rel", "drums_rel", "vocals_rel", "groove_rel",
        "energy_spread", "bass_spread", "drums_spread", "vocals_spread", "groove_spread",
        "intensity_band", "intensity_membership", "role_hints", "valid_as_of_track_count",
        "analysis_signature", "config_signature",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_for_csv)
    if not args.json:
        print(f"Wrote CSV report to {output_path}")


def _limit_relative_preview(
    preview: RelativePlaylistPreview,
    *,
    limit: int,
) -> RelativePlaylistPreview:
    if limit <= 0 or len(preview.tracks) <= limit:
        return preview
    return RelativePlaylistPreview(
        playlist=preview.playlist,
        playlist_id=preview.playlist_id,
        is_limited=True,
        limited_track_count=limit,
        playlist_stats=preview.playlist_stats,
        tracks=preview.tracks[:limit],
    )


def _refresh_canonical_relative_preview(
    database: Database,
    *,
    playlist_name: str,
    playlist_id: str,
    settings: RuntimeSettings,
    abs_rows: list,
    announce_reason: str | None = None,
    json_mode: bool = False,
) -> RelativePlaylistPreview:
    if announce_reason and not json_mode:
        print(f"Refreshing canonical relative data ({announce_reason})...")
    rel_inputs = [row_to_relative_track_input(row) for row in abs_rows]
    return refresh_canonical_relative_playlist(
        rel_inputs,
        settings,
        playlist_name=playlist_name,
        playlist_id=playlist_id,
        database=database,
        timestamp=utc_now(),
    )


def _ensure_scoring_relative_freshness(
    playlist_name: str,
    playlist_stats: dict[str, Any] | None,
    settings: RuntimeSettings,
) -> None:
    """Fail fast when canonical relative artifacts are missing or stale for scoring."""
    from cuemate_analysis.config import build_relative_experiment_signature

    expected_relative_signature = build_relative_experiment_signature(
        settings,
        energy_source="canonical",
    )

    if playlist_stats is None:
        raise SystemExit(
            f"Playlist '{playlist_name}' has no persisted relative features. "
            f"Run `python -m cuemate_analysis refresh-relative-playlist --playlist \"{playlist_name}\"` first."
        )

    if bool(playlist_stats.get("is_stale")):
        reason = str(playlist_stats.get("stale_reason") or "stale")
        raise SystemExit(
            f"Playlist relative features are stale ({reason}); run "
            f"`python -m cuemate_analysis refresh-relative-playlist --playlist \"{playlist_name}\"`."
        )

    if str(playlist_stats.get("relative_signature") or "") != expected_relative_signature:
        raise SystemExit(
            f"Playlist relative features are out of date for '{playlist_name}'; run "
            f"`python -m cuemate_analysis refresh-relative-playlist --playlist \"{playlist_name}\"`."
        )


def handle_analyze_relative_playlist(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()

    energy_source = args.energy_source or settings.analysis.energy_source_default
    # Normalize deprecated / legacy aliases
    if energy_source == "essentia_fused":
        energy_source = "canonical"
    if energy_source == "heuristic":
        energy_source = "heuristic_legacy"

    with Database(settings.database_path) as database:
        abs_rows = database.get_playlist_relative_inputs(args.playlist)
        if not abs_rows:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")

        playlist_id = str(abs_rows[0]["playlist_id"])

        if energy_source == "canonical":
            # Persisted-first: refresh when rows are missing, stale, or the current
            # relative signature no longer matches the persisted signature.
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

                preview = _refresh_canonical_relative_preview(
                    database,
                    playlist_name=args.playlist,
                    playlist_id=playlist_id,
                    settings=settings,
                    abs_rows=abs_rows,
                    announce_reason=reason,
                    json_mode=args.json,
                )
            else:
                # Read from persisted tables and reconstruct preview shape.
                preview = _load_persisted_canonical_preview(
                    database, args.playlist, playlist_id, abs_rows, settings
                )

            if args.limit and args.limit > 0:
                preview = _limit_relative_preview(preview, limit=args.limit)
        else:
            # heuristic_legacy: always compute in-memory, never persist.
            if args.limit and args.limit > 0:
                abs_rows = abs_rows[: args.limit]
            rel_inputs = [row_to_relative_track_input(row) for row in abs_rows]
            preview = compute_relative_playlist_preview(
                rel_inputs,
                settings,
                playlist_name=args.playlist,
                is_limited=bool(args.limit and args.limit > 0),
                energy_source="heuristic_legacy",
            )

    _run_canonical_relative_preview_and_print(
        preview,
        playlist_name=args.playlist,
        energy_source=energy_source,
        args=args,
        abs_rows=abs_rows,
    )

    if args.output:
        _write_relative_csv(preview, Path(args.output).expanduser().resolve(), args)

    return 0


def handle_refresh_relative_playlist(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    with Database(settings.database_path) as database:
        abs_rows = database.get_playlist_relative_inputs(args.playlist)
        if not abs_rows:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")
        playlist_id = str(abs_rows[0]["playlist_id"])
        preview = _refresh_canonical_relative_preview(
            database,
            playlist_name=args.playlist,
            playlist_id=playlist_id,
            settings=settings,
            abs_rows=abs_rows,
            announce_reason="manual_refresh",
            json_mode=args.json,
        )

    if args.limit and args.limit > 0:
        preview = _limit_relative_preview(preview, limit=args.limit)

    _run_canonical_relative_preview_and_print(
        preview,
        playlist_name=args.playlist,
        energy_source="canonical",
        args=args,
        abs_rows=abs_rows,
    )

    if args.output:
        _write_relative_csv(preview, Path(args.output).expanduser().resolve(), args)

    return 0


def _load_persisted_canonical_preview(
    database: "Database",
    playlist_name: str,
    playlist_id: str,
    abs_rows: list,
    settings: "RuntimeSettings",
) -> "RelativePlaylistPreview":
    """Reconstruct a RelativePlaylistPreview from persisted track_features_rel + playlist_stats."""
    from cuemate_analysis.relative_context import (
        RelativePlaylistPreview,
        RelativeTrackPreview,
        PlaylistStatsPreview,
    )
    from cuemate_analysis.config import build_relative_experiment_signature

    stats_row = database.get_playlist_stats(playlist_id)
    track_rows = database.get_persisted_relative_rows(playlist_id)

    tracks: list[RelativeTrackPreview] = []
    for row in track_rows:
        tracks.append(RelativeTrackPreview(
            track_id=str(row["track_id"]),
            playlist_id=playlist_id,
            position=int(row["position"]),
            title=row["title"] if row["title"] else None,
            artist=row["artist"] if row["artist"] else None,
            file_path=str(row["file_path"]),
            energy_source_used="canonical",
            energy_rel=float(row["energy_rel"]),
            bass_rel=float(row["bass_rel"]),
            drums_rel=float(row["drums_rel"]),
            vocals_rel=float(row["vocals_rel"]) if row["vocals_rel"] is not None else None,
            groove_rel=float(row["groove_rel"]),
            energy_spread=float(row["energy_spread"]),
            bass_spread=float(row["bass_spread"]),
            drums_spread=float(row["drums_spread"]),
            vocals_spread=float(row["vocals_spread"]),
            groove_spread=float(row["groove_spread"]),
            intensity_band=str(row["intensity_band"]),
            intensity_membership=json.loads(str(row["intensity_membership"])),
            role_hints=json.loads(str(row["role_hints"])),
            valid_as_of_track_count=int(row["valid_as_of_track_count"]),
            analyzed_at=None,
            analysis_signature=str(row["analysis_signature"]),
            config_signature=str(row["config_signature"]),
        ))

    def _opt_float(v: object) -> float | None:
        return float(v) if v is not None else None

    relative_sig = (
        str(stats_row["relative_signature"])
        if stats_row
        else build_relative_experiment_signature(settings, energy_source="canonical")
    )
    stats = PlaylistStatsPreview(
        playlist_id=playlist_id,
        track_count_total=int(stats_row["track_count_total"]) if stats_row else len(abs_rows),
        track_count_analyzed=int(stats_row["track_count_analyzed"]) if stats_row else 0,
        eligible_track_count=int(stats_row["eligible_track_count"]) if stats_row else 0,
        avg_harmonic=_opt_float(stats_row["avg_harmonic"]) if stats_row else None,
        key_diversity=_opt_float(stats_row["key_diversity"]) if stats_row else None,
        bpm_range=_opt_float(stats_row["bpm_range"]) if stats_row else None,
        energy_spread=_opt_float(stats_row["energy_spread"]) if stats_row else None,
        bass_spread=_opt_float(stats_row["bass_spread"]) if stats_row else None,
        drums_spread=_opt_float(stats_row["drums_spread"]) if stats_row else None,
        vocals_spread=_opt_float(stats_row["vocals_spread"]) if stats_row else None,
        harmonic_spread=_opt_float(stats_row["harmonic_spread"]) if stats_row else None,
        groove_spread=_opt_float(stats_row["groove_spread"]) if stats_row else None,
        adapted_weights=json.loads(str(stats_row["adapted_weights"])) if stats_row and stats_row["adapted_weights"] else None,
        adaptation_strength=_opt_float(stats_row["adaptation_strength"]) if stats_row else None,
        weight_adaptation_notes=json.loads(str(stats_row["weight_adaptation_notes"])) if stats_row and stats_row["weight_adaptation_notes"] else [],
        status=str(stats_row["status"]) if stats_row else "insufficient_tracks",
        energy_source_used="canonical",
        relative_signature=relative_sig,
    )
    return RelativePlaylistPreview(
        playlist=playlist_name,
        playlist_id=playlist_id,
        is_limited=False,
        limited_track_count=stats.track_count_total,
        playlist_stats=stats,
        tracks=tracks,
    )


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
        fieldnames = list(payload_rows[0].keys()) if payload_rows else [
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


def handle_download_essentia_semantic_models(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    model_root = args.model_root or settings.analysis.essentia_semantic_model_root
    family_policy = args.family_policy or settings.analysis.essentia_semantic_model_family_policy
    downloaded = download_essentia_semantic_models(model_root=model_root, family_policy=family_policy)
    resolved_root = resolve_essentia_semantic_model_root(model_root)
    print(f"Downloaded Essentia semantic model bundle to {resolved_root}")
    for path in downloaded:
        print(f"- {path}")
    return 0


def handle_analyze_essentia_playlist(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    with Database(settings.database_path) as database:
        rows = database.get_playlist_tracks(args.playlist)
        if not rows:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    payload_rows: list[dict[str, object]] = []
    diagnostics_estimates: list[EssentiaSemanticEstimate] = []
    total = len(rows)
    if not args.json:
        print(f"Playlist '{args.playlist}' Essentia semantic preview")

    for chunk in chunk_items(rows, settings.analysis.essentia_chunk_size):
        estimates = prefetch_essentia_semantic_estimates(chunk, settings)
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
        print_backend_diagnostics(
            "Essentia semantics",
            diagnostics_estimates,
            requested_device=settings.analysis.essentia_semantic_device,
        )
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(payload_rows[0].keys()) if payload_rows else [
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
    deleted_essentia_semantics = 0

    if args.backend in {"all", "tempocnn"}:
        deleted_tempocnn = purge_tempocnn_cache(
            file_paths=deduped_paths or None,
            clear_warm_service=args.clear_warm_services,
        )
    if args.backend in {"all", "musicalkeycnn"}:
        deleted_musicalkeycnn = purge_musicalkeycnn_cache(
            file_paths=deduped_paths or None,
            clear_warm_service=args.clear_warm_services,
        )
    if args.backend in {"all", "essentia_semantics"}:
        deleted_essentia_semantics = purge_essentia_semantic_cache(
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
    print(f"- Persistent cache DB: {resolve_inference_cache_path()}")
    print("- Re-run analysis with --force if you want stored playlist analysis rows refreshed too.")
    return 0


def handle_run_analysis_worker(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
    expected_essentia_semantic_signature = resolve_expected_essentia_semantic_signature(settings)
    effective_analysis_signature = build_effective_analysis_signature(
        settings.analysis_signature,
        tempocnn_accelerator="auto",
        musicalkeycnn_model=settings.analysis.key_model_path,
        musicalkeycnn_device=settings.analysis.key_device,
        musicalkeycnn_policy=settings.analysis.key_policy,
        essentia_semantic_signature=expected_essentia_semantic_signature,
    )
    
    with Database(settings.database_path) as database:
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

            track = read_track_metadata_with_overrides(
                Path(str(track_row["file_path"])),
                bpm_imported=track_row["imported_bpm"],
                key_imported=track_row["imported_key"],
                title_override=track_row["title"],
                artist_override=track_row["artist"],
                genre_override=track_row["genre"],
                import_source=track_row["import_source"] or "local_files",
            )
            prepared = PreparedTrack(
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
        ) = _process_enrichment_batch(
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
            _refresh_relative_for_playlists(database, settings, touched_playlists)

        if updated_track_ids and hasattr(database, "get_playlists_containing_tracks") and hasattr(database, "mark_playlists_stale"):
            linked = database.get_playlists_containing_tracks(updated_track_ids)
            stale_targets = [pid for pid in linked if pid not in touched_playlists]
            if stale_targets:
                database.mark_playlists_stale(stale_targets, "absolute_track_changed", utc_now())
                print(f"Marked {len(stale_targets)} linked playlist(s) stale.")

        print(f"Processed {processed} enrichment job(s).")

        if args.print_backend_diagnostics:
            print_backend_diagnostics("DSP local lane", dsp_diagnostics, requested_device="cpu")
            print_backend_diagnostics("TempoCNN", tempo_diagnostics, requested_device="auto")
            print_backend_diagnostics("MusicalKeyCNN", key_diagnostics, requested_device=settings.analysis.key_device)
            print_backend_diagnostics(
                "Essentia semantics",
                essentia_diagnostics,
                requested_device=settings.analysis.essentia_semantic_device,
            )

        return 0


def handle_prewarm_model_services(args: argparse.Namespace) -> int:
    settings = load_runtime_settings()
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


def handle_benchmark_dsp(args: argparse.Namespace) -> int:
    from cuemate_analysis.dsp_benchmark import (
        benchmark_dsp_paths,
        summarize_dsp_benchmark,
        write_benchmark_csv,
        write_benchmark_json,
    )

    settings = load_runtime_settings()
    paths: list[Path] = []

    if args.playlist:
        with Database(settings.database_path) as database:
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


def handle_recommend_next(args: argparse.Namespace) -> int:
    from cuemate_analysis.config import build_scoring_config
    from cuemate_analysis.scoring import get_recommendations, row_to_scoring_track_context

    settings = load_runtime_settings()
    config = build_scoring_config(settings, target=args.target)

    with Database(settings.database_path) as db:
        playlist_row = db.get_playlist(args.playlist)
        if playlist_row is None:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")
        playlist_id = str(playlist_row["id"])

        playlist_stats = db.get_playlist_stats_for_scoring(playlist_id)
        _ensure_scoring_relative_freshness(args.playlist, playlist_stats, settings)

        candidate_rows = db.get_scoring_candidates(playlist_id)
        if not candidate_rows:
            raise SystemExit(f"No analyzed tracks found for playlist '{args.playlist}'.")

        current_id = args.current_track
        if current_id is None:
            current_id = str(candidate_rows[0]["track_id"])

        current_row = db.get_track_scoring_context(current_id, playlist_id)
        if current_row is None:
            raise SystemExit(f"Track '{current_id}' not found in playlist '{args.playlist}'.")
        current = row_to_scoring_track_context(current_row)
        candidates = [row_to_scoring_track_context(r) for r in candidate_rows]

    if args.current_track is None and not args.json:
        current_label = f"{current.artist} - {current.title}" if current.artist or current.title else current.track_id
        print(f"No --current-track given; using first analyzed track: {current_label} [{current.track_id}]")

    result = get_recommendations(
        current,
        candidates,
        history=[],
        config=config,
        playlist_stats=playlist_stats,
        target=args.target,
        max_per_lane=args.max_per_lane,
    )

    if args.json:
        # Serialize: replace ScoringTrackContext objects with their track_id
        def _serialize(obj):
            from cuemate_analysis.scoring import ScoringTrackContext
            if isinstance(obj, ScoringTrackContext):
                return obj.track_id
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialize(i) for i in obj]
            return obj

        print(json.dumps(_serialize(result), indent=2))
        return 0

    # Human-readable output
    conf = result["recommendation_confidence"]
    meta = result["meta"]
    current_label = f"{current.artist} - {current.title}" if current.artist or current.title else current.track_id
    print(f"\nRecommendations for '{args.playlist}' | current: {current_label} [{current.track_id}]")
    print(f"Target: {args.target}  |  Confidence: {conf:.2f}  |  Scored: {meta['scored_candidates']} tracks\n")
    fallback_note = meta.get("fallback_note")
    if fallback_note:
        print(f"  Note: {fallback_note}\n")

    for lane in result["lane_order"]:
        items = result["lanes"].get(lane, [])
        if not items:
            continue
        print(f"  [{lane.upper()}]")
        for item in items:
            cand = item["candidate"]
            score = item["score"]
            move = item["move"]
            risk = item["risk"]
            bpm_rel = item["transition_features"].get("effective_bpm_distance", 0.0)
            key_label = item["transition_features"].get("key_compat_label", "—")
            secondary = " (contrast)" if item.get("secondary_lane") else ""
            cand_label = f"{cand.artist} - {cand.title}" if cand.artist or cand.title else cand.track_id
            print(
                f"    {cand_label} [{cand.track_id}]\n"
                f"      score={score:.3f}  move={move:<8}  risk={risk:<6}  bpm_dist={bpm_rel:.1f}  key={key_label}{secondary}"
            )
        print()

    return 0


def handle_score_pair(args: argparse.Namespace) -> int:
    from cuemate_analysis.config import build_scoring_config
    from cuemate_analysis.scoring import row_to_scoring_track_context, score_candidate

    settings = load_runtime_settings()
    config = build_scoring_config(settings, target=args.target)

    with Database(settings.database_path) as db:
        playlist_row = db.get_playlist(args.playlist)
        if playlist_row is None:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")
        playlist_id = str(playlist_row["id"])

        playlist_stats = db.get_playlist_stats_for_scoring(playlist_id)
        _ensure_scoring_relative_freshness(args.playlist, playlist_stats, settings)

        current_row = db.get_track_scoring_context(args.current, playlist_id)
        if current_row is None:
            raise SystemExit(f"Track '{args.current}' not found in playlist '{args.playlist}'.")
        candidate_row = db.get_track_scoring_context(args.candidate, playlist_id)
        if candidate_row is None:
            raise SystemExit(f"Track '{args.candidate}' not found in playlist '{args.playlist}'.")

    current = row_to_scoring_track_context(current_row)
    candidate = row_to_scoring_track_context(candidate_row)

    result = score_candidate(
        current=current,
        candidate=candidate,
        history=[],
        config=config,
        playlist_stats=playlist_stats,
    )

    if args.json:
        def _serialize(obj):
            from cuemate_analysis.scoring import ScoringTrackContext
            if isinstance(obj, ScoringTrackContext):
                return obj.track_id
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialize(i) for i in obj]
            return obj

        print(json.dumps(_serialize(result), indent=2))
        return 0

    # Human-readable breakdown
    def _track_label(t) -> str:
        if t.artist or t.title:
            return f"{t.artist} - {t.title} [{t.track_id}]"
        return t.track_id

    print(f"\nScore pair: {_track_label(current)}  ->  {_track_label(candidate)}")
    print(f"  Final score:      {result['score']:.4f}  (raw: {result['raw_score']:.4f})")
    print(f"  Penalty:          {result['penalty_multiplier']:.4f}")
    print(f"  Move:             {result['move']} (confidence={result['move_confidence']:.2f})")
    print(f"  Risk:             {result['risk']} (score={result['risk_score']:.3f})")
    print(f"  Contrast score:   {result['contrast_score']:.3f}")

    print("\n  Transition features:")
    for k, v in sorted(result["transition_features"].items()):
        print(f"    {k:<32}  {v}")

    print("\n  Component scores:")
    for k, v in sorted(result["component_scores"].items()):
        w = result["weights_used"].get(k, 0.0)
        c = result["confidences"].get(k, 1.0)
        if v is None:
            print(f"    {k:<24}  {'stub':<8}  weight={w:.4f}  (not implemented)")
        else:
            print(f"    {k:<24}  {v:.4f}  weight={w:.4f}  conf={c:.4f}")

    if result["penalty_factors"]:
        print("\n  Penalty factors:")
        for factor in result["penalty_factors"]:
            name = factor.get("factor", "unknown")
            penalty = factor.get("effective_penalty", factor.get("raw_penalty", ""))
            severity = factor.get("severity", "")
            print(f"    {name:<28}  penalty={penalty}  severity={severity}")

    if result["risk_factors"]:
        print("\n  Risk factors:")
        for note in result["risk_factors"]:
            print(f"    {note}")

    stub_components = sorted(
        name for name, value in result["component_scores"].items() if value is None
    )
    current_vocals_rel = result["transition_features"].get("current_vocals_rel")
    candidate_vocals_rel = result["transition_features"].get("candidate_vocals_rel")
    if stub_components or current_vocals_rel is None or candidate_vocals_rel is None:
        print("\n  Notes:")
        if stub_components:
            joined = ", ".join(stub_components)
            print(f"    Stubbed and excluded from weighted scoring: {joined}.")
        if current_vocals_rel is None or candidate_vocals_rel is None:
            print(
                "    vocals_abs / vocals_rel are not populated yet; missing vocal fields are unknown, not silence."
            )

    return 0


def handle_inspect_scoring_weights(args: argparse.Namespace) -> int:
    from cuemate_analysis.config import build_scoring_config
    from cuemate_analysis.scoring import resolve_effective_weights

    settings = load_runtime_settings()
    config = build_scoring_config(settings, target="maintain")

    with Database(settings.database_path) as db:
        playlist_row = db.get_playlist(args.playlist)
        if playlist_row is None:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")
        playlist_id = str(playlist_row["id"])
        playlist_stats = db.get_playlist_stats_for_scoring(playlist_id)
        _ensure_scoring_relative_freshness(args.playlist, playlist_stats, settings)

    static_weights = config["static_weights"]
    adapted_weights = (playlist_stats or {}).get("adapted_weights") if playlist_stats else None
    effective_weights = resolve_effective_weights(playlist_stats, config)

    if args.json:
        print(json.dumps({
            "playlist": args.playlist,
            "static_weights": static_weights,
            "adapted_weights": adapted_weights,
            "effective_weights": effective_weights,
            "weight_floors": config["weight_floors"],
        }, indent=2))
        return 0

    print(f"\nScoring weights for '{args.playlist}'\n")
    print(f"  {'Component':<24}  {'Static':>8}  {'Adapted':>8}  {'Effective':>9}")
    print(f"  {'-'*24}  {'-'*8}  {'-'*8}  {'-'*9}")
    for key in sorted(static_weights):
        s = static_weights.get(key, 0.0)
        a = (adapted_weights or {}).get(key) if adapted_weights else None
        e = effective_weights.get(key, s)
        a_str = f"{a:.4f}" if a is not None else "    —   "
        print(f"  {key:<24}  {s:>8.4f}  {a_str:>8}  {e:>9.4f}")

    if adapted_weights:
        print("\n  Adaptation is active (playlist has adapted_weights).")
    else:
        print("\n  No adaptation — using static weights.")

    return 0


def handle_inspect_scoring_metadata(args: argparse.Namespace) -> int:
    from cuemate_analysis.scoring import check_analysis_compatibility, get_scoring_metadata

    settings = load_runtime_settings()
    metadata = get_scoring_metadata(settings)

    compatibility = None
    if (
        args.analysis_signature is not None
        or args.config_signature is not None
        or args.scoring_contract_id_at_analysis is not None
    ):
        compatibility = check_analysis_compatibility(
            args.analysis_signature,
            args.config_signature,
            args.scoring_contract_id_at_analysis,
            scoring_metadata=metadata,
        )

    if args.json:
        payload: dict[str, object] = {"metadata": metadata}
        if compatibility is not None:
            payload["compatibility"] = compatibility
        print(json.dumps(payload, indent=2))
        return 0

    active = metadata["active_signatures"]
    print("\nScoring metadata\n")
    print(f"  analysis_signature:   {active['analysis_signature']}")
    print(f"  config_signature:     {active['config_signature']}")
    print(f"  scoring_contract_id:  {active['scoring_contract_id']}")
    print(f"  engine_version:       {metadata['engine_version']}")
    print(f"  healthy:              {metadata['healthy']}")
    print(f"  status_note:          {metadata['status_note']}")

    print("\n  Capability flags:")
    for key, value in sorted(metadata["capability_flags"].items()):
        print(f"    {key:<32} {value}")

    print("\n  Supported lanes:")
    for lane in metadata["supported_lane_groups"]:
        print(f"    {lane['lane_id']:<10} {lane['summary']}")

    print("\n  Components:")
    for component in metadata["components"]:
        available = component.get("available")
        active = component.get("active")
        if available is False:
            state = "stubbed"
        elif active is False:
            state = "inactive"
        else:
            state = "active"
        print(
            f"    {component['component_id']:<24} weight={component['weight']:.4f}  state={state:<8}  {component['description']}"
        )

    if compatibility is not None:
        print("\n  Compatibility:")
        print(f"    exact_match:         {compatibility['exact_match']}")
        print(f"    compatible:          {compatibility['compatible']}")
        print(f"    requires_reanalysis: {compatibility['requires_reanalysis']}")
        print(f"    reason:              {compatibility['reason']}")
        notes = compatibility.get("notes", [])
        if notes:
            print("    notes:")
            for note in notes:
                print(f"      {note}")

    return 0


def handle_serve_scoring(args: argparse.Namespace) -> int:
    from cuemate_analysis.scoring_service import serve_scoring_grpc

    settings = load_runtime_settings()
    return serve_scoring_grpc(
        host=args.host,
        port=args.port,
        settings=settings,
        max_workers=args.max_workers,
    )


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
        if args.command == "refresh-relative-playlist":
            return handle_refresh_relative_playlist(args)
        if args.command == "run-analysis-worker":
            return handle_run_analysis_worker(args)
        if args.command == "prewarm-model-services":
            return handle_prewarm_model_services(args)
        if args.command == "analyze-energy-playlist":
            return handle_analyze_energy_playlist(args)
        if args.command == "download-essentia-semantic-models":
            return handle_download_essentia_semantic_models(args)
        if args.command == "analyze-essentia-playlist":
            return handle_analyze_essentia_playlist(args)
        if args.command == "purge-model-cache":
            return handle_purge_model_cache(args)
        if args.command == "benchmark-dsp":
            return handle_benchmark_dsp(args)
        if args.command == "recommend-next":
            return handle_recommend_next(args)
        if args.command == "score-pair":
            return handle_score_pair(args)
        if args.command == "inspect-scoring-weights":
            return handle_inspect_scoring_weights(args)
        if args.command == "inspect-scoring-metadata":
            return handle_inspect_scoring_metadata(args)
        if args.command == "serve-scoring":
            return handle_serve_scoring(args)
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
