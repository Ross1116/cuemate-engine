from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from time import perf_counter

import librosa

from cuemate_analysis.analysis import (
    build_track_dsp_artifacts,
    extract_bass_ratio,
    extract_energy,
    extract_full_features,
    extract_loudness,
    detect_time_signature,
)
from cuemate_analysis.config import RuntimeSettings


@dataclass(frozen=True)
class DspBenchmarkSample:
    file_path: str
    decode_seconds: float
    artifact_seconds: float
    energy_seconds: float
    loudness_seconds: float
    bass_seconds: float
    time_signature_seconds: float
    full_features_seconds: float
    total_dsp_seconds: float
    total_with_decode_seconds: float

    def to_payload(self) -> dict[str, float | str]:
        return asdict(self)


def benchmark_dsp_paths(paths: list[Path], settings: RuntimeSettings) -> list[DspBenchmarkSample]:
    samples: list[DspBenchmarkSample] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        decode_start = perf_counter()
        y, sr = librosa.load(
            resolved.as_posix(),
            sr=settings.analysis.sample_rate,
            mono=settings.analysis.mono,
        )
        decode_end = perf_counter()

        artifact_start = perf_counter()
        artifacts = build_track_dsp_artifacts(y, sr)
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

        samples.append(
            DspBenchmarkSample(
                file_path=resolved.as_posix(),
                decode_seconds=decode_end - decode_start,
                artifact_seconds=artifact_end - artifact_start,
                energy_seconds=energy_end - energy_start,
                loudness_seconds=loudness_end - loudness_start,
                bass_seconds=bass_end - bass_start,
                time_signature_seconds=time_signature_end - time_signature_start,
                full_features_seconds=full_end - full_start,
                total_dsp_seconds=full_end - artifact_start,
                total_with_decode_seconds=full_end - decode_start,
            )
        )
    return samples


def summarize_dsp_benchmark(samples: list[DspBenchmarkSample]) -> dict[str, float]:
    if not samples:
        return {}
    fields = [
        "decode_seconds",
        "artifact_seconds",
        "energy_seconds",
        "loudness_seconds",
        "bass_seconds",
        "time_signature_seconds",
        "full_features_seconds",
        "total_dsp_seconds",
        "total_with_decode_seconds",
    ]
    return {f"median_{field}": median(getattr(sample, field) for sample in samples) for field in fields}
