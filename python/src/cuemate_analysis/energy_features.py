from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


FEATURE_NAMES = [
    "baseline",
    "loudness_fusion",
    "club_fusion",
    "pressure_fusion",
    "energy_sustained",
    "energy_peak",
    "loudness_norm",
    "loudness_lufs",
    "bass_abs",
    "drums_abs",
    "harmonic_abs",
    "groove_abs",
]


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    sanitized = _safe_float(value, fallback=minimum)
    return float(max(minimum, min(maximum, sanitized)))


def _safe_float(value: float | None, fallback: float = 0.5) -> float:
    if value is None:
        return fallback
    numeric = float(value)
    if not math.isfinite(numeric):
        return fallback
    return numeric


@dataclass(frozen=True)
class EnergyFeatureVector:
    baseline: float
    loudness_fusion: float
    club_fusion: float
    pressure_fusion: float
    energy_sustained: float
    energy_peak: float
    loudness_norm: float
    loudness_lufs: float
    bass_abs: float
    drums_abs: float
    harmonic_abs: float
    groove_abs: float

    def to_payload(self) -> dict[str, float]:
        return asdict(self)

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in FEATURE_NAMES], dtype=float)


def build_energy_feature_vector(
    *,
    energy_abs: float,
    energy_sustained: float | None,
    energy_peak: float | None,
    loudness_norm: float,
    loudness_lufs: float,
    bass_abs: float,
    drums_abs: float | None,
    harmonic_abs: float | None,
    groove_abs: float | None,
) -> EnergyFeatureVector:
    baseline = clamp(float(energy_abs))
    sustained = clamp(_safe_float(energy_sustained))
    peak = clamp(_safe_float(energy_peak))
    drums_value = clamp(_safe_float(drums_abs))
    harmonic_value = clamp(_safe_float(harmonic_abs))
    groove_value = clamp(_safe_float(groove_abs))
    bass_value = clamp(float(bass_abs))
    loudness_norm_value = clamp(float(loudness_norm))
    loudness_lufs_value = float(loudness_lufs)

    loudness_fusion = clamp(
        (0.40 * loudness_norm_value)
        + (0.28 * baseline)
        + (0.18 * peak)
        + (0.14 * sustained)
    )
    club_fusion = clamp(
        (0.28 * baseline)
        + (0.22 * loudness_norm_value)
        + (0.18 * drums_value)
        + (0.18 * groove_value)
        + (0.14 * bass_value)
    )
    pressure_fusion = clamp(
        (0.24 * peak)
        + (0.24 * drums_value)
        + (0.20 * loudness_norm_value)
        + (0.16 * groove_value)
        + (0.10 * bass_value)
        + (0.06 * (1.0 - harmonic_value))
    )
    return EnergyFeatureVector(
        baseline=baseline,
        loudness_fusion=loudness_fusion,
        club_fusion=club_fusion,
        pressure_fusion=pressure_fusion,
        energy_sustained=sustained,
        energy_peak=peak,
        loudness_norm=loudness_norm_value,
        loudness_lufs=loudness_lufs_value,
        bass_abs=bass_value,
        drums_abs=drums_value,
        harmonic_abs=harmonic_value,
        groove_abs=groove_value,
    )


def energy_consensus(features: EnergyFeatureVector) -> float:
    return clamp(
        float(
            np.mean(
                [
                    features.baseline,
                    features.loudness_fusion,
                    features.club_fusion,
                    features.pressure_fusion,
                ]
            )
        )
    )
