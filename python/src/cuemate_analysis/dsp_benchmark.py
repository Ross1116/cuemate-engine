from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from time import perf_counter

import librosa
import numpy as np

from cuemate_analysis.analysis import (
    TrackDspArtifacts,
    extract_bass_ratio,
    extract_energy,
    extract_full_features,
    extract_loudness,
    detect_time_signature,
    DEFAULT_N_FFT,
    DEFAULT_HOP_LENGTH,
)
from cuemate_analysis.config import RuntimeSettings


@dataclass(frozen=True)
class DspBenchmarkSample:
    file_path: str
    decode_seconds: float
    # Artifact substeps
    stft_seconds: float
    rms_centroid_seconds: float
    onset_envelope_seconds: float
    beat_tracking_seconds: float
    artifact_seconds: float
    # Feature extraction stages
    energy_seconds: float
    loudness_seconds: float
    bass_seconds: float
    time_signature_seconds: float
    full_features_seconds: float
    total_dsp_seconds: float
    total_with_decode_seconds: float

    def to_payload(self) -> dict[str, float | str]:
        return asdict(self)


def _build_artifacts_with_substeps(
    y: np.ndarray,
    sr: int,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> tuple[TrackDspArtifacts, dict[str, float]]:
    """Build TrackDspArtifacts while timing each substep."""
    timings: dict[str, float] = {}

    t0 = perf_counter()
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    timings["stft_seconds"] = perf_counter() - t0

    t0 = perf_counter()
    rms = librosa.feature.rms(S=magnitude, hop_length=hop_length)[0]
    spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    timings["rms_centroid_seconds"] = perf_counter() - t0

    t0 = perf_counter()
    onset_env = librosa.onset.onset_strength(S=librosa.feature.melspectrogram(S=magnitude**2, sr=sr), sr=sr)
    timings["onset_envelope_seconds"] = perf_counter() - t0

    # Beat tracking is lazy on TrackDspArtifacts; force it here for timing.
    artifacts = TrackDspArtifacts(
        y=y,
        sr=sr,
        duration_seconds=max(len(y) / float(sr), 1e-6),
        stft=stft,
        magnitude=magnitude,
        frequencies=frequencies,
        rms=rms,
        spectral_centroid=spectral_centroid,
        onset_env=onset_env,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    t0 = perf_counter()
    _ = artifacts.beat_frames
    _ = artifacts.beat_times
    timings["beat_tracking_seconds"] = perf_counter() - t0

    return artifacts, timings


def benchmark_dsp_single(
    path: Path,
    settings: RuntimeSettings,
) -> DspBenchmarkSample:
    """Profile the full DSP pipeline for a single file with substep timings."""
    resolved = path.expanduser().resolve()

    decode_start = perf_counter()
    y, sr = librosa.load(
        resolved.as_posix(),
        sr=settings.analysis.sample_rate,
        mono=settings.analysis.mono,
    )
    decode_end = perf_counter()

    artifact_start = perf_counter()
    artifacts, substeps = _build_artifacts_with_substeps(y, sr)
    artifact_end = perf_counter()

    energy_start = perf_counter()
    extract_energy(artifacts=artifacts)
    energy_end = perf_counter()

    loudness_start = perf_counter()
    extract_loudness(artifacts=artifacts)
    loudness_end = perf_counter()

    bass_start = perf_counter()
    extract_bass_ratio(artifacts=artifacts)
    bass_end = perf_counter()

    time_signature_start = perf_counter()
    detect_time_signature(artifacts=artifacts)
    time_signature_end = perf_counter()

    full_start = perf_counter()
    extract_full_features(artifacts=artifacts)
    full_end = perf_counter()

    return DspBenchmarkSample(
        file_path=resolved.as_posix(),
        decode_seconds=decode_end - decode_start,
        stft_seconds=substeps["stft_seconds"],
        rms_centroid_seconds=substeps["rms_centroid_seconds"],
        onset_envelope_seconds=substeps["onset_envelope_seconds"],
        beat_tracking_seconds=substeps["beat_tracking_seconds"],
        artifact_seconds=artifact_end - artifact_start,
        energy_seconds=energy_end - energy_start,
        loudness_seconds=loudness_end - loudness_start,
        bass_seconds=bass_end - bass_start,
        time_signature_seconds=time_signature_end - time_signature_start,
        full_features_seconds=full_end - full_start,
        total_dsp_seconds=full_end - artifact_start,
        total_with_decode_seconds=full_end - decode_start,
    )


def benchmark_dsp_paths(paths: list[Path], settings: RuntimeSettings) -> list[DspBenchmarkSample]:
    """Profile the DSP pipeline for multiple files."""
    return [benchmark_dsp_single(path, settings) for path in paths]


SUMMARY_FIELDS = [
    "decode_seconds",
    "stft_seconds",
    "rms_centroid_seconds",
    "onset_envelope_seconds",
    "beat_tracking_seconds",
    "artifact_seconds",
    "energy_seconds",
    "loudness_seconds",
    "bass_seconds",
    "time_signature_seconds",
    "full_features_seconds",
    "total_dsp_seconds",
    "total_with_decode_seconds",
]


def summarize_dsp_benchmark(samples: list[DspBenchmarkSample]) -> dict[str, float]:
    if not samples:
        return {}
    return {
        f"median_{field}": median(getattr(sample, field) for sample in samples)
        for field in SUMMARY_FIELDS
    }


def write_benchmark_csv(samples: list[DspBenchmarkSample], output_path: Path) -> None:
    """Write per-track benchmark samples to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(DspBenchmarkSample.__dataclass_fields__.keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.to_payload())


def write_benchmark_json(
    samples: list[DspBenchmarkSample],
    summary: dict[str, float],
    output_path: Path,
) -> None:
    """Write full benchmark report (per-track + summary) to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "track_count": len(samples),
        "summary": summary,
        "tracks": [sample.to_payload() for sample in samples],
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
