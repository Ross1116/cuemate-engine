from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from cuemate_analysis.analysis import (
    build_track_dsp_artifacts,
    extract_bass_ratio,
    extract_energy,
    extract_full_features,
    extract_loudness,
)
from cuemate_analysis.energy_features import EnergyFeatureVector, build_energy_feature_vector, energy_consensus


@dataclass(frozen=True)
class EnergyCandidateSet:
    baseline: float
    loudness_fusion: float
    club_fusion: float
    pressure_fusion: float
    consensus: float
    energy_sustained: float | None
    energy_peak: float | None
    loudness_norm: float
    loudness_lufs: float
    bass_abs: float
    drums_abs: float | None
    harmonic_abs: float | None
    groove_abs: float | None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def build_energy_candidate_set_from_vector(
    features: EnergyFeatureVector,
) -> EnergyCandidateSet:
    return EnergyCandidateSet(
        baseline=features.baseline,
        loudness_fusion=features.loudness_fusion,
        club_fusion=features.club_fusion,
        pressure_fusion=features.pressure_fusion,
        consensus=energy_consensus(features),
        energy_sustained=features.energy_sustained,
        energy_peak=features.energy_peak,
        loudness_norm=features.loudness_norm,
        loudness_lufs=features.loudness_lufs,
        bass_abs=features.bass_abs,
        drums_abs=features.drums_abs,
        harmonic_abs=features.harmonic_abs,
        groove_abs=features.groove_abs,
    )


def build_energy_candidate_set(y: np.ndarray, sr: int) -> EnergyCandidateSet:
    artifacts = build_track_dsp_artifacts(y, sr)
    energy = extract_energy(artifacts=artifacts)
    loudness = extract_loudness(artifacts=artifacts)
    bass_abs = extract_bass_ratio(artifacts=artifacts)
    full = extract_full_features(artifacts=artifacts)
    features = build_energy_feature_vector(
        energy_abs=float(energy["energy_abs"]),
        energy_sustained=energy["energy_sustained"],
        energy_peak=energy["energy_peak"],
        loudness_norm=float(loudness["loudness_norm"]),
        loudness_lufs=float(loudness["loudness_lufs"]),
        bass_abs=bass_abs,
        drums_abs=full["drums_abs"],
        harmonic_abs=full["harmonic_abs"],
        groove_abs=full["groove_abs"],
    )
    return build_energy_candidate_set_from_vector(features)


def analyze_energy_path(path: Path, *, sample_rate: int = 22050, mono: bool = True) -> EnergyCandidateSet:
    resolved = path.expanduser().resolve()
    y, sr = librosa.load(resolved.as_posix(), sr=sample_rate, mono=mono)
    if y.size == 0:
        raise ValueError(f"No audio samples decoded for {resolved}")
    return build_energy_candidate_set(y, sr)
