from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import re
import time

import librosa
import numpy as np
import pyloudnorm as pyln

from cuemate_analysis.config import RuntimeSettings
from cuemate_analysis.essentia_semantic_backend import EssentiaSemanticEstimate
from cuemate_analysis.models import AnalysisResult, FastAnalysisResult, ImportedTrack


PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
IMPORTED_BPM_CONFIDENCE = 0.88
FILE_TAG_BPM_CONFIDENCE = 0.78
FILE_TAG_KEY_CONFIDENCE = 0.85
IMPORTED_KEY_CONFIDENCE = 0.80
MUSICALKEYCNN_CONFIRMED_THRESHOLD = 0.70
MUSICALKEYCNN_UNCERTAIN_THRESHOLD = 0.40
BPM_CORROBORATION_BOOST = 0.06
BPM_CONFLICT_PENALTY = 0.10
BPM_MODEL_PRIORITY_THRESHOLD = 0.85
MIN_DETECTED_BPM_CONFIDENCE = 0.55
MUSICALKEYCNN_METADATA_CONFLICT_PENALTY = 0.85

CAMELOT_BY_PITCH_MODE = {
    ("A", "minor"): ("8A", 8, "A"),
    ("E", "minor"): ("9A", 9, "A"),
    ("B", "minor"): ("10A", 10, "A"),
    ("F#", "minor"): ("11A", 11, "A"),
    ("C#", "minor"): ("12A", 12, "A"),
    ("G#", "minor"): ("1A", 1, "A"),
    ("D#", "minor"): ("2A", 2, "A"),
    ("A#", "minor"): ("3A", 3, "A"),
    ("F", "minor"): ("4A", 4, "A"),
    ("C", "minor"): ("5A", 5, "A"),
    ("G", "minor"): ("6A", 6, "A"),
    ("D", "minor"): ("7A", 7, "A"),
    ("B", "major"): ("1B", 1, "B"),
    ("F#", "major"): ("2B", 2, "B"),
    ("C#", "major"): ("3B", 3, "B"),
    ("G#", "major"): ("4B", 4, "B"),
    ("D#", "major"): ("5B", 5, "B"),
    ("A#", "major"): ("6B", 6, "B"),
    ("F", "major"): ("7B", 7, "B"),
    ("C", "major"): ("8B", 8, "B"),
    ("G", "major"): ("9B", 9, "B"),
    ("D", "major"): ("10B", 10, "B"),
    ("A", "major"): ("11B", 11, "B"),
    ("E", "major"): ("12B", 12, "B"),
}

ENHARMONIC_NOTES = {
    "AB": "G#",
    "BB": "A#",
    "CB": "B",
    "DB": "C#",
    "EB": "D#",
    "FB": "E",
    "GB": "F#",
    "B#": "C",
    "E#": "F",
}
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return float(max(minimum, min(maximum, value)))


def normalize_loudness(lufs: float) -> float:
    return clamp((lufs + 24.0) / 18.0)


def log_normalize(value: float, scale: float) -> float:
    if scale <= 0.0:
        return clamp(value)
    return clamp(math.log1p(max(value, 0.0) * scale) / math.log1p(scale))


@dataclass
class TrackDspArtifacts:
    y: np.ndarray
    sr: int
    duration_seconds: float
    stft: np.ndarray
    magnitude: np.ndarray
    frequencies: np.ndarray
    rms: np.ndarray
    spectral_centroid: np.ndarray
    onset_env: np.ndarray
    n_fft: int = DEFAULT_N_FFT
    hop_length: int = DEFAULT_HOP_LENGTH
    _beat_frames: np.ndarray | None = None
    _beat_times: np.ndarray | None = None
    _hpss: tuple[np.ndarray, np.ndarray] | None = None
    _percussive_onset_env: np.ndarray | None = None
    _pulse: np.ndarray | None = None
    _harmonic_chroma: np.ndarray | None = None
    _harmonic_waveform: np.ndarray | None = None
    _percussive_waveform: np.ndarray | None = None

    @classmethod
    def from_audio(
        cls,
        y: np.ndarray,
        sr: int,
        *,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
    ) -> "TrackDspArtifacts":
        y_array = np.asarray(y, dtype=np.float32)

        if y_array.ndim > 1:
            y_mono = librosa.to_mono(y_array)
        else:
            y_mono = y_array

        y_mono = np.ascontiguousarray(y_mono, dtype=np.float32)
        duration_seconds = max(y_mono.shape[-1] / float(sr), 1e-6)

        stft = librosa.stft(y=y_mono, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        # Reuse magnitude spectrogram instead of recomputing from waveform.
        rms = librosa.feature.rms(S=magnitude, hop_length=hop_length)[0]
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        onset_env = librosa.onset.onset_strength(
            S=librosa.feature.melspectrogram(S=magnitude**2, sr=sr),
            sr=sr,
        )
        return cls(
            y=y_mono,
            sr=sr,
            duration_seconds=duration_seconds,
            stft=stft,
            magnitude=magnitude,
            frequencies=frequencies,
            rms=rms,
            spectral_centroid=spectral_centroid,
            onset_env=onset_env,
            n_fft=n_fft,
            hop_length=hop_length,
        )

    @property
    def beat_frames(self) -> np.ndarray:
        if self._beat_frames is None:
            _, frames = librosa.beat.beat_track(onset_envelope=self.onset_env, sr=self.sr)
            self._beat_frames = np.asarray(frames, dtype=int)
        return self._beat_frames

    @property
    def beat_times(self) -> np.ndarray:
        if self._beat_times is None:
            self._beat_times = librosa.frames_to_time(
                self.beat_frames, sr=self.sr, hop_length=self.hop_length,
            )
        return self._beat_times

    @property
    def hpss(self) -> tuple[np.ndarray, np.ndarray]:
        if self._hpss is None:
            self._hpss = librosa.decompose.hpss(self.magnitude)
        return self._hpss

    @property
    def harmonic_magnitude(self) -> np.ndarray:
        return self.hpss[0]

    @property
    def percussive_magnitude(self) -> np.ndarray:
        return self.hpss[1]

    @property
    def harmonic_waveform(self) -> np.ndarray:
        if self._harmonic_waveform is None:
            self._harmonic_waveform = librosa.istft(
                self.harmonic_magnitude * np.exp(1j * np.angle(self.stft)),
                hop_length=self.hop_length,
                length=len(self.y),
            )
        return self._harmonic_waveform

    @property
    def percussive_waveform(self) -> np.ndarray:
        if self._percussive_waveform is None:
            self._percussive_waveform = librosa.istft(
                self.percussive_magnitude * np.exp(1j * np.angle(self.stft)),
                hop_length=self.hop_length,
                length=len(self.y),
            )
        return self._percussive_waveform

    @property
    def percussive_onset_env(self) -> np.ndarray:
        if self._percussive_onset_env is None:
            self._percussive_onset_env = librosa.onset.onset_strength(y=self.percussive_waveform, sr=self.sr)
        return self._percussive_onset_env

    @property
    def pulse(self) -> np.ndarray:
        if self._pulse is None:
            self._pulse = librosa.beat.plp(onset_envelope=self.onset_env, sr=self.sr)
        return self._pulse

    @property
    def harmonic_chroma(self) -> np.ndarray:
        if self._harmonic_chroma is None:
            try:
                self._harmonic_chroma = librosa.feature.chroma_stft(S=self.harmonic_magnitude, sr=self.sr)
            except Exception:
                try:
                    self._harmonic_chroma = librosa.feature.chroma_stft(y=self.harmonic_waveform, sr=self.sr)
                except Exception:
                    self._harmonic_chroma = librosa.feature.chroma_cqt(y=self.harmonic_waveform, sr=self.sr)
        return self._harmonic_chroma

@dataclass
class DspLaneResult:
    track_id: str
    y: np.ndarray | None
    sr: int | None
    artifacts: TrackDspArtifacts | None
    energy: dict[str, float] | None
    loudness: dict[str, float] | None
    bass_abs: float | None
    time_signature: dict[str, str | float] | None
    full_features: dict[str, float | None]
    elapsed_ms: float | None
    details: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    available: bool = True
    error: str | None = None

    @classmethod
    def success(
        cls,
        *,
        track_id: str,
        y: np.ndarray,
        sr: int,
        artifacts: TrackDspArtifacts,
        energy: dict[str, float],
        loudness: dict[str, float],
        bass_abs: float,
        time_signature: dict[str, str | float],
        full_features: dict[str, float | None],
        elapsed_ms: float,
    ) -> "DspLaneResult":
        return cls(
            track_id=track_id,
            y=y,
            sr=sr,
            artifacts=artifacts,
            energy=energy,
            loudness=loudness,
            bass_abs=bass_abs,
            time_signature=time_signature,
            full_features=full_features,
            elapsed_ms=elapsed_ms,
            details={"runner_device": "cpu"},
            notes=["Shared local DSP lane."],
            available=True,
            error=None,
        )

    @classmethod
    def failure(cls, *, track_id: str, error: Exception | str, elapsed_ms: float | None = None) -> "DspLaneResult":
        message = str(error)
        return cls(
            track_id=track_id,
            y=None,
            sr=None,
            artifacts=None,
            energy=None,
            loudness=None,
            bass_abs=None,
            time_signature=None,
            full_features={"drums_abs": None, "harmonic_abs": None, "groove_abs": None},
            elapsed_ms=elapsed_ms,
            details={"runner_device": "cpu"},
            notes=[message],
            available=False,
            error=message,
        )


def build_track_dsp_artifacts(
    y: np.ndarray,
    sr: int,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> TrackDspArtifacts:
    return TrackDspArtifacts.from_audio(y, sr, n_fft=n_fft, hop_length=hop_length)


def parse_key_label(raw_value: str | None) -> dict[str, str | int] | None:
    if not raw_value:
        return None

    clean = raw_value.strip().replace("♭", "b").replace("♯", "#")
    camelot_match = re.fullmatch(r"(1[0-2]|[1-9])([ABab])", clean)
    if camelot_match:
        number = int(camelot_match.group(1))
        letter = camelot_match.group(2).upper()
        return {"key": f"{number}{letter}", "key_number": number, "key_letter": letter}

    compact = re.sub(r"\s+", "", clean)
    lower = compact.lower()
    mode = "major"
    note = compact

    if lower.endswith("minor"):
        mode = "minor"
        note = compact[:-5]
    elif lower.endswith("major"):
        mode = "major"
        note = compact[:-5]
    elif lower.endswith("min"):
        mode = "minor"
        note = compact[:-3]
    elif lower.endswith("maj"):
        mode = "major"
        note = compact[:-3]
    elif compact.endswith("m") and len(compact) > 1:
        mode = "minor"
        note = compact[:-1]

    if not note:
        return None

    note = note[0].upper() + note[1:]
    note = note.replace("b", "B")
    note = ENHARMONIC_NOTES.get(note.upper(), note.upper()).replace("B", "b")
    if len(note) > 1 and note[1] == "b":
        note = ENHARMONIC_NOTES.get(note.upper(), note.upper())

    normalized_note = note[0].upper() + note[1:]
    lookup = CAMELOT_BY_PITCH_MODE.get((normalized_note, mode))
    if lookup is None:
        return None

    key, key_number, key_letter = lookup
    return {"key": key, "key_number": key_number, "key_letter": key_letter}


def bayesian_key_confidence(
    conf_a: float,
    conf_b: float,
    *,
    provenance_independent: bool = False,
) -> float:
    conf_a = clamp(conf_a)
    conf_b = clamp(conf_b)
    if provenance_independent:
        p_both_right = conf_a * conf_b
        p_both_wrong_and_agree = (1.0 - conf_a) * (1.0 - conf_b) * (1.0 / 24.0)
        combined = p_both_right / max(p_both_right + p_both_wrong_and_agree, 1e-9)
        return clamp(combined, maximum=0.98)
    stronger = max(conf_a, conf_b)
    weaker = min(conf_a, conf_b)
    correlation_lift = 0.15 * weaker
    return clamp(stronger + correlation_lift, maximum=0.95)


def detect_bpm(
    y: np.ndarray | None = None,
    sr: int | None = None,
    *,
    artifacts: TrackDspArtifacts | None = None,
) -> dict[str, float]:
    if artifacts is None:
        if y is None or sr is None:
            raise ValueError("detect_bpm requires either raw audio or TrackDspArtifacts.")
        artifacts = build_track_dsp_artifacts(y, sr)
    onset_env = artifacts.onset_env
    tempo_value, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=artifacts.sr)
    tempo = float(np.atleast_1d(tempo_value)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=artifacts.sr, hop_length=artifacts.hop_length)

    if len(beat_times) >= 2:
        ibis = np.diff(beat_times)
        ibi_mean = float(np.mean(ibis)) if ibis.size else 0.0
        ibi_std = float(np.std(ibis)) if ibis.size else 0.0
        bpm_confidence = clamp(1.0 - ((ibi_std / max(ibi_mean, 1e-6)) * 5.0))
    else:
        bpm_confidence = 0.3

    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=artifacts.sr)
    peak_sharpness = float(np.max(tempogram) / (np.mean(tempogram) + 1e-10)) if tempogram.size else 0.0
    tempogram_confidence = clamp(peak_sharpness / 20.0)
    combined_confidence = clamp((0.5 * bpm_confidence) + (0.5 * tempogram_confidence))

    return {"bpm": max(tempo, 0.0), "bpm_confidence": combined_confidence}

def bpm_agreement_delta(reference_bpm: float, candidate_bpm: float) -> tuple[bool, float, float]:
    """
    Returns:
        (agrees, matched_candidate_bpm, absolute_delta)

    Agreement preserves the existing half/double-time logic:
    ratios [1.0, 2.0, 0.5]
    tolerance 1.0 for 1x, 2.0 for half/double matches
    """
    ratios = [1.0, 2.0, 0.5]
    best_ratio = min(ratios, key=lambda ratio: abs(reference_bpm - (candidate_bpm * ratio)))
    matched = candidate_bpm * best_ratio
    tolerance = 1.0 if math.isclose(best_ratio, 1.0) else 2.0
    delta = abs(reference_bpm - matched)
    return (delta <= tolerance, matched, delta)


def make_bpm_source_label(winner_label: str, corroborators: list[str]) -> str:
    ordered = [winner_label, *[label for label in corroborators if label != winner_label]]
    seen: set[str] = set()
    unique = []
    for label in ordered:
        if label not in seen:
            seen.add(label)
            unique.append(label)
    return "+".join(unique)


def build_bpm_candidates(
    imported_bpm: float | None,
    tagged_bpm: float | None,
    detected_bpm: float,
    detected_confidence: float,
    *,
    detected_source: str,
) -> list[dict[str, float | str | list[str]]]:
    candidates: list[dict[str, float | str | list[str]]] = []

    if imported_bpm and imported_bpm > 0:
        candidates.append(
            {
                "label": "imported",
                "bpm": float(imported_bpm),
                "base_confidence": IMPORTED_BPM_CONFIDENCE,
                "score": IMPORTED_BPM_CONFIDENCE,
                "corroborators": [],
                "agreements": 0,
                "conflicts": 0,
                "delta_to_detected": float("inf"),
            }
        )

    if tagged_bpm and tagged_bpm > 0:
        candidates.append(
            {
                "label": "tag",
                "bpm": float(tagged_bpm),
                "base_confidence": FILE_TAG_BPM_CONFIDENCE,
                "score": FILE_TAG_BPM_CONFIDENCE,
                "corroborators": [],
                "agreements": 0,
                "conflicts": 0,
                "delta_to_detected": float("inf"),
            }
        )

    if detected_bpm > 0:
        candidates.append(
            {
                "label": detected_source,
                "bpm": float(detected_bpm),
                "base_confidence": float(detected_confidence),
                "score": float(detected_confidence),
                "corroborators": [],
                "agreements": 0,
                "conflicts": 0,
                "delta_to_detected": 0.0,
            }
        )

    return candidates

def resolve_bpm(
    imported_bpm: float | None,
    tagged_bpm: float | None,
    detected: dict[str, float],
    *,
    detected_source: str = "detected",
) -> dict[str, float | str]:
    detected_bpm = float(detected.get("bpm", 0.0))
    detected_confidence = float(detected.get("bpm_confidence", 0.0))

    candidates = build_bpm_candidates(
        imported_bpm,
        tagged_bpm,
        detected_bpm,
        detected_confidence,
        detected_source=detected_source,
    )

    if not candidates:
        raise ValueError("Unable to resolve BPM from tags or detection.")

    # Detection-only case keeps the minimum floor behavior from current logic.
    if len(candidates) == 1 and str(candidates[0]["label"]) == detected_source:
        return {
            "bpm": float(candidates[0]["bpm"]),
            "bpm_confidence": max(float(candidates[0]["base_confidence"]), MIN_DETECTED_BPM_CONFIDENCE),
            "bpm_source": detected_source,
        }

    # Pairwise corroboration / conflict scoring.
    for i, left in enumerate(candidates):
        for j, right in enumerate(candidates):
            if i >= j:
                continue

            left_bpm = float(left["bpm"])
            right_bpm = float(right["bpm"])
            agrees, _, delta = bpm_agreement_delta(left_bpm, right_bpm)

            if str(left["label"]) == detected_source:
                left["delta_to_detected"] = 0.0
                right["delta_to_detected"] = min(float(right["delta_to_detected"]), float(delta))
            elif str(right["label"]) == detected_source:
                right["delta_to_detected"] = 0.0
                left["delta_to_detected"] = min(float(left["delta_to_detected"]), float(delta))

            if agrees:
                left["score"] = min(0.98, float(left["score"]) + BPM_CORROBORATION_BOOST)
                right["score"] = min(0.98, float(right["score"]) + BPM_CORROBORATION_BOOST)
                left["agreements"] = int(left["agreements"]) + 1
                right["agreements"] = int(right["agreements"]) + 1
                cast_left = list(left["corroborators"])
                cast_right = list(right["corroborators"])
                cast_left.append(str(right["label"]))
                cast_right.append(str(left["label"]))
                left["corroborators"] = cast_left
                right["corroborators"] = cast_right
            else:
                left["score"] = max(0.0, float(left["score"]) - BPM_CONFLICT_PENALTY)
                right["score"] = max(0.0, float(right["score"]) - BPM_CONFLICT_PENALTY)
                left["conflicts"] = int(left["conflicts"]) + 1
                right["conflicts"] = int(right["conflicts"]) + 1

    # Model-priority modifier:
    # strong detector confidence should usually beat stale/conflicting metadata.
    for candidate in candidates:
        label = str(candidate["label"])
        if label == detected_source:
            if float(candidate["base_confidence"]) >= BPM_MODEL_PRIORITY_THRESHOLD:
                candidate["score"] = min(0.98, float(candidate["score"]) + 0.07)
        else:
            detected_candidate = next((item for item in candidates if str(item["label"]) == detected_source), None)
            if detected_candidate is not None and float(detected_candidate["base_confidence"]) >= BPM_MODEL_PRIORITY_THRESHOLD:
                agrees, _, _ = bpm_agreement_delta(float(candidate["bpm"]), float(detected_candidate["bpm"]))
                if not agrees:
                    candidate["score"] = max(0.0, float(candidate["score"]) - 0.05)

    # Deterministic winner selection:
    # 1) highest score
    # 2) most agreements
    # 3) fewer conflicts
    # 4) if near tie, prefer detected over tag
    # 5) smaller delta to detected
    def winner_key(item: dict[str, float | str | list[str]]):
        label = str(item["label"])
        score = float(item["score"])
        agreements = int(item["agreements"])
        conflicts = int(item["conflicts"])
        detected_bonus = 1 if label == detected_source else 0
        tag_penalty = -1 if label == "tag" else 0
        delta_to_detected = float(item["delta_to_detected"])
        return (
            score,
            agreements,
            -conflicts,
            detected_bonus,
            tag_penalty,
            -delta_to_detected,
        )

    winner = max(candidates, key=winner_key)
    winner_label = str(winner["label"])
    winner_bpm = float(winner["bpm"])
    winner_confidence = float(winner["score"])

    # Preserve old behavior that detected-alone should not fall below a floor,
    # but do not artificially floor detection in mixed-source arbitration.
    if winner_label == detected_source and len(candidates) == 1:
        winner_confidence = max(winner_confidence, MIN_DETECTED_BPM_CONFIDENCE)

    corroborators = list(winner["corroborators"])
    source = make_bpm_source_label(winner_label, corroborators)

    return {
        "bpm": winner_bpm,
        "bpm_confidence": winner_confidence,
        "bpm_source": source,
    }


def resolve_bpm_with_backend(
    track: ImportedTrack,
    y: np.ndarray,
    sr: int,
    *,
    tempo_backend: str,
    tempocnn_model: str | None = None,
    tempocnn_accelerator: str = "auto",
    prefetched_tempocnn_estimate=None,
    artifacts: TrackDspArtifacts | None = None,
) -> dict[str, float | str]:
    if tempo_backend == "baseline":
        return resolve_bpm(
            track.bpm_imported,
            track.bpm_tag,
            detect_bpm(artifacts=artifacts or build_track_dsp_artifacts(y, sr)),
            detected_source="detected",
        )

    from cuemate_analysis.tempo_backend import (
        TEMPO_BACKEND_TEMPOCNN,
        estimate_tempocnn_bpm,
        normalize_tempo_backend,
    )

    normalized_backend = normalize_tempo_backend(tempo_backend)
    if normalized_backend != TEMPO_BACKEND_TEMPOCNN:
        raise ValueError(f"Unsupported tempo backend: {tempo_backend}")

    estimate = prefetched_tempocnn_estimate or estimate_tempocnn_bpm(
        track.file_path,
        model_path=tempocnn_model,
        accelerator=tempocnn_accelerator,
    )

    if estimate.available and estimate.bpm is not None:
        return resolve_bpm(
            track.bpm_imported,
            track.bpm_tag,
            {
                "bpm": float(estimate.bpm),
                "bpm_confidence": clamp(float(estimate.confidence or 0.0)),
            },
            detected_source="tempocnn",
        )

    baseline = detect_bpm(y, sr) if artifacts is None else detect_bpm(artifacts=artifacts)
    return resolve_bpm(
        track.bpm_imported,
        track.bpm_tag,
        baseline,
        detected_source="baseline_fallback",
    )


def detect_key(y: np.ndarray, sr: int) -> dict[str, str | int | float]:
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    except Exception:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    chroma_mean = np.mean(chroma, axis=1)
    if not np.any(chroma_mean):
        raise ValueError("Unable to derive chroma for key estimation.")

    chroma_vector = chroma_mean / np.sum(chroma_mean)
    scores: list[tuple[float, str, str]] = []
    for pitch_index, pitch_name in enumerate(PITCH_CLASSES):
        major_score = float(np.dot(chroma_vector, np.roll(MAJOR_PROFILE, pitch_index)))
        minor_score = float(np.dot(chroma_vector, np.roll(MINOR_PROFILE, pitch_index)))
        scores.append((major_score, pitch_name, "major"))
        scores.append((minor_score, pitch_name, "minor"))

    scores.sort(key=lambda item: item[0], reverse=True)
    best_score, best_pitch, best_mode = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else 0.0
    confidence = clamp(0.45 + ((best_score - second_score) / max(abs(best_score), 1e-6)))

    key, key_number, key_letter = CAMELOT_BY_PITCH_MODE[(best_pitch, best_mode)]
    return {
        "key": key,
        "key_number": key_number,
        "key_letter": key_letter,
        "key_confidence": confidence,
        "pitch": best_pitch,
        "mode": best_mode,
    }

def classify_musicalkeycnn_band(
    confidence: float,
    *,
    has_metadata: bool,
    has_conflict: bool,
) -> str:
    confidence = clamp(confidence)
    if has_conflict:
        if confidence >= MUSICALKEYCNN_CONFIRMED_THRESHOLD:
            return "uncertain"
        return "conflicted"

    if confidence >= MUSICALKEYCNN_CONFIRMED_THRESHOLD:
        return "confirmed"
    if confidence >= MUSICALKEYCNN_UNCERTAIN_THRESHOLD:
        return "uncertain"
    return "conflicted"


def merge_key_sources(primary: str, agreeing_sources: list[str]) -> str:
    ordered = [primary, *agreeing_sources]
    seen: set[str] = set()
    merged: list[str] = []
    for item in ordered:
        if item not in seen:
            seen.add(item)
            merged.append(item)
    return "+".join(merged)


def resolve_key(
    tagged_key: str | None,
    imported_key: str | None,
    detected: dict[str, str | int | float],
) -> dict[str, str | int | float | None]:
    parsed_tag = parse_key_label(tagged_key)
    parsed_imported = parse_key_label(imported_key)
    detected_source = str(detected.get("key_source", "chroma"))
    detected_confidence = clamp(float(detected.get("key_confidence", 0.0) or 0.0))

    detected_payload = {
        "key": str(detected["key"]),
        "key_number": int(detected["key_number"]),
        "key_letter": str(detected["key_letter"]),
    }

    # MusicalKeyCNN-primary policy:
    # the model is primary when present; metadata is corroboration/disagreement evidence.
    if detected_source == "musicalkeycnn":
        agreeing_sources: list[str] = []
        metadata_confidences: list[float] = []
        has_conflict = False

        if parsed_tag is not None:
            if parsed_tag["key"] == detected_payload["key"]:
                agreeing_sources.append("tag")
                metadata_confidences.append(FILE_TAG_KEY_CONFIDENCE)
            else:
                has_conflict = True

        if parsed_imported is not None:
            if parsed_imported["key"] == detected_payload["key"]:
                agreeing_sources.append("imported")
                metadata_confidences.append(IMPORTED_KEY_CONFIDENCE)
            else:
                has_conflict = True

        final_confidence = detected_confidence
        if metadata_confidences:
            for meta_conf in metadata_confidences:
                final_confidence = bayesian_key_confidence(
                    final_confidence,
                    meta_conf,
                    provenance_independent=False,
                )
        elif has_conflict:
            final_confidence = clamp(final_confidence * MUSICALKEYCNN_METADATA_CONFLICT_PENALTY)

        key_band = classify_musicalkeycnn_band(
            final_confidence,
            has_metadata=(parsed_tag is not None or parsed_imported is not None),
            has_conflict=has_conflict and not bool(agreeing_sources),
        )

        return {
            **detected_payload,
            "key_confidence": final_confidence,
            "key_source": merge_key_sources("musicalkeycnn", agreeing_sources),
            "key_imported": imported_key,
            "key_tagged": tagged_key,
            "key_agreement": 1 if agreeing_sources else (0 if (parsed_tag is not None or parsed_imported is not None) else None),
            "key_band": key_band,
        }

    # Non-CNN fallback path (e.g. chroma): preserve existing metadata-first behavior.
    if parsed_tag is not None:
        if parsed_tag["key"] == detected_payload["key"]:
            return {
                **parsed_tag,
                "key_confidence": bayesian_key_confidence(
                    detected_confidence,
                    FILE_TAG_KEY_CONFIDENCE,
                    provenance_independent=False,
                ),
                "key_source": f"tag+{detected_source}",
                "key_imported": imported_key,
                "key_tagged": tagged_key,
                "key_agreement": 1,
                "key_band": "confirmed",
            }

        return {
            **parsed_tag,
            "key_confidence": clamp(min(detected_confidence, FILE_TAG_KEY_CONFIDENCE) * 0.6),
            "key_source": "tag_conflicted",
            "key_imported": imported_key,
            "key_tagged": tagged_key,
            "key_agreement": 0,
            "key_band": "conflicted",
        }

    if parsed_imported is not None:
        if parsed_imported["key"] == detected_payload["key"]:
            return {
                **parsed_imported,
                "key_confidence": bayesian_key_confidence(
                    detected_confidence,
                    IMPORTED_KEY_CONFIDENCE,
                    provenance_independent=False,
                ),
                "key_source": f"imported+{detected_source}",
                "key_imported": imported_key,
                "key_tagged": tagged_key,
                "key_agreement": 1,
                "key_band": "confirmed",
            }

        return {
            **parsed_imported,
            "key_confidence": clamp(min(detected_confidence, IMPORTED_KEY_CONFIDENCE) * 0.6),
            "key_source": "imported_conflicted",
            "key_imported": imported_key,
            "key_tagged": tagged_key,
            "key_agreement": 0,
            "key_band": "conflicted",
        }

    # No metadata at all: return detected result directly.
    standalone_band = (
        "confirmed"
        if detected_confidence >= MUSICALKEYCNN_CONFIRMED_THRESHOLD
        else "uncertain"
        if detected_confidence >= MUSICALKEYCNN_UNCERTAIN_THRESHOLD
        else "conflicted"
    )

    return {
        **detected_payload,
        "key_confidence": detected_confidence,
        "key_source": detected_source,
        "key_imported": imported_key,
        "key_tagged": tagged_key,
        "key_agreement": None,
        "key_band": standalone_band if detected_source == "musicalkeycnn" else "uncertain",
    }


def resolve_tag_only_key(tagged_key: str | None, imported_key: str | None) -> dict[str, str | int | float | None]:
    parsed_tag = parse_key_label(tagged_key)
    if parsed_tag is not None:
        return {
            **parsed_tag,
            "key_confidence": FILE_TAG_KEY_CONFIDENCE,
            "key_source": "tag_only_fallback",
            "key_imported": imported_key,
            "key_tagged": tagged_key,
            "key_agreement": None,
            "key_band": "uncertain",
        }

    parsed_imported = parse_key_label(imported_key)
    if parsed_imported is not None:
        return {
            **parsed_imported,
            "key_confidence": IMPORTED_KEY_CONFIDENCE,
            "key_source": "import_only_fallback",
            "key_imported": imported_key,
            "key_tagged": tagged_key,
            "key_agreement": None,
            "key_band": "uncertain",
        }

    raise RuntimeError(
        "MusicalKeyCNN was unavailable and no usable tagged or imported key was found; chroma fallback is disabled."
    )
    

def resolve_key_with_backend(
    track: ImportedTrack,
    y: np.ndarray,
    sr: int,
    *,
    key_backend: str,
    musicalkeycnn_model: str | None = None,
    musicalkeycnn_image: str | None = None,
    musicalkeycnn_device: str = "auto",
    musicalkeycnn_policy: str = "full_track",
    prefetched_musicalkeycnn_estimate=None,
) -> dict[str, str | int | float | None]:
    if key_backend == "chroma":
        return resolve_key(
            track.key_tag,
            track.key_imported,
            {
                **detect_key(y, sr),
                "key_source": "chroma",
            },
        )

    from cuemate_analysis.key_backend import (
        KEY_BACKEND_MUSICALKEYCNN,
        estimate_musicalkeycnn_key,
        normalize_key_backend,
    )

    normalized_backend = normalize_key_backend(key_backend)
    if normalized_backend != KEY_BACKEND_MUSICALKEYCNN:
        raise ValueError(f"Unsupported key backend: {key_backend}")

    estimate = prefetched_musicalkeycnn_estimate or estimate_musicalkeycnn_key(
        track.file_path,
        model_path=musicalkeycnn_model,
        image_name=musicalkeycnn_image,
        device=musicalkeycnn_device,
        policy=musicalkeycnn_policy,
    )

    if (
        estimate.available
        and estimate.key
        and estimate.key_number is not None
        and estimate.key_letter is not None
    ):
        return resolve_key(
            track.key_tag,
            track.key_imported,
            {
                "key": estimate.key,
                "key_number": estimate.key_number,
                "key_letter": estimate.key_letter,
                "key_confidence": clamp(float(estimate.confidence or 0.0)),
                "pitch": estimate.details.get("pitch"),
                "mode": estimate.details.get("mode"),
                "key_source": "musicalkeycnn",
            },
        )

    try:
        chroma_detected = detect_key(y, sr)
        if (
            chroma_detected.get("key")
            and chroma_detected.get("key_number") is not None
            and chroma_detected.get("key_letter") is not None
        ):
            return resolve_key(
                track.key_tag,
                track.key_imported,
                {
                    "key": str(chroma_detected["key"]),
                    "key_number": int(chroma_detected["key_number"]),
                    "key_letter": str(chroma_detected["key_letter"]),
                    "key_confidence": clamp(float(chroma_detected.get("key_confidence", 0.0) or 0.0)),
                    "pitch": chroma_detected.get("pitch"),
                    "mode": chroma_detected.get("mode"),
                    "key_source": "chroma",
                },
            )
    except Exception:
        pass

    return resolve_tag_only_key(track.key_tag, track.key_imported)

def extract_energy(
    y: np.ndarray | None = None,
    sr: int | None = None,
    *,
    artifacts: TrackDspArtifacts | None = None,
) -> dict[str, float]:
    if artifacts is None:
        if y is None or sr is None:
            raise ValueError("extract_energy requires either raw audio or TrackDspArtifacts.")
        artifacts = build_track_dsp_artifacts(y, sr)
    rms = artifacts.rms
    sustained = float(np.percentile(rms, 75))
    peak = float(np.percentile(rms, 95))
    brightness = clamp(float(np.mean(artifacts.spectral_centroid)) / max(artifacts.sr / 2.0, 1.0))
    onset_env = artifacts.onset_env
    onset_peak = float(np.percentile(onset_env, 85)) if onset_env.size else 0.0

    # Use log compression so modern mastered tracks still separate meaningfully
    # instead of saturating at 1.0 across an entire crate.
    sustained_norm = log_normalize(sustained, scale=30.0)
    peak_norm = log_normalize(peak, scale=24.0)
    onset_norm = log_normalize(onset_peak, scale=8.0)
    raw_energy = (
        (0.42 * sustained_norm)
        + (0.18 * peak_norm)
        + (0.22 * brightness)
        + (0.18 * onset_norm)
    )
    return {
        "energy_abs": clamp(raw_energy),
        "energy_sustained": sustained_norm,
        "energy_peak": peak_norm,
    }


def detect_time_signature(
    y: np.ndarray | None = None,
    sr: int | None = None,
    *,
    artifacts: TrackDspArtifacts | None = None,
) -> dict[str, str | float]:
    if artifacts is None:
        if y is None or sr is None:
            raise ValueError("detect_time_signature requires either raw audio or TrackDspArtifacts.")
        artifacts = build_track_dsp_artifacts(y, sr)

    onset_env = artifacts.onset_env
    beat_frames = artifacts.beat_frames

    if len(beat_frames) < 16:
        return {"time_signature": "4/4", "time_signature_confidence": 0.20}

    beat_strengths = onset_env[beat_frames]
    if beat_strengths.size < 16 or float(np.mean(beat_strengths)) <= 1e-6:
        return {"time_signature": "4/4", "time_signature_confidence": 0.20}

    candidates = ["3/4", "4/4", "5/4"]
    scores: dict[str, float] = {}
    normalized_strengths = beat_strengths / max(float(np.mean(beat_strengths)), 1e-6)

    for signature in candidates:
        meter = int(signature.split("/")[0])
        grouped = [normalized_strengths[index::meter] for index in range(meter)]
        means = np.array(
            [float(np.mean(group)) if group.size else 0.0 for group in grouped],
            dtype=float,
        )
        accent_contrast = float(np.max(means) - np.min(means))
        downbeat_strength = float(np.max(means))
        regularity = clamp(len(beat_frames) / float(meter * 12))
        scores[signature] = (accent_contrast * 0.5) + (downbeat_strength * 0.2) + (regularity * 0.3)

    best_signature = max(scores, key=scores.get)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_score = ordered[0][1]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = best_score - second_score

    # Be conservative: only emit non-4/4 when evidence is unusually clear.
    if best_signature != "4/4" and margin < 0.18:
        return {"time_signature": "4/4", "time_signature_confidence": 0.25}

    confidence = clamp(0.20 + (margin * 0.35) + (0.10 if best_signature == "4/4" else 0.0))
    return {"time_signature": best_signature, "time_signature_confidence": confidence}


def extract_loudness(
    y: np.ndarray | None = None,
    sr: int | None = None,
    *,
    artifacts: TrackDspArtifacts | None = None,
) -> dict[str, float]:
    if artifacts is None:
        if y is None or sr is None:
            raise ValueError("extract_loudness requires either raw audio or TrackDspArtifacts.")
        artifacts = build_track_dsp_artifacts(y, sr)
    meter = pyln.Meter(artifacts.sr)
    loudness_lufs = float(meter.integrated_loudness(artifacts.y.astype(np.float64)))
    return {
        "loudness_lufs": loudness_lufs,
        "loudness_norm": normalize_loudness(loudness_lufs),
    }


def extract_bass_ratio(
    y: np.ndarray | None = None,
    sr: int | None = None,
    *,
    artifacts: TrackDspArtifacts | None = None,
) -> float:
    if artifacts is None:
        if y is None or sr is None:
            raise ValueError("extract_bass_ratio requires either raw audio or TrackDspArtifacts.")
        artifacts = build_track_dsp_artifacts(y, sr)
    bass_mask = (artifacts.frequencies >= 30.0) & (artifacts.frequencies <= 150.0)
    total_mask = (artifacts.frequencies >= 30.0) & (artifacts.frequencies <= 8000.0)
    bass_energy = float(np.sum(artifacts.magnitude[bass_mask]))
    total_energy = float(np.sum(artifacts.magnitude[total_mask])) or 1e-6
    return clamp(bass_energy / total_energy)


def extract_full_features(
    y: np.ndarray | None = None,
    sr: int | None = None,
    *,
    artifacts: TrackDspArtifacts | None = None,
) -> dict[str, float]:
    if artifacts is None:
        if y is None or sr is None:
            raise ValueError("extract_full_features requires either raw audio or TrackDspArtifacts.")
        artifacts = build_track_dsp_artifacts(y, sr)
    onset_env = artifacts.onset_env
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=artifacts.sr)
    onset_rate = len(onset_frames) / artifacts.duration_seconds
    pulse = artifacts.pulse
    chroma = artifacts.harmonic_chroma
    chroma_focus = float(np.mean(np.max(chroma, axis=0))) if chroma.size else 0.0
    flatness = librosa.feature.spectral_flatness(S=artifacts.magnitude)[0]
    flatness_mean = clamp(float(np.mean(flatness)) / 0.5 if flatness.size else 0.0)
    onset_norm = log_normalize(float(np.percentile(onset_env, 85)) if onset_env.size else 0.0, scale=40.0)
    spectral_contrast = librosa.feature.spectral_contrast(S=artifacts.magnitude, sr=artifacts.sr)
    contrast_mean = clamp(float(np.mean(spectral_contrast)) / 30.0 if spectral_contrast.size else 0.0)

    drums_abs = clamp((0.55 * onset_norm) + (0.45 * clamp(onset_rate / 6.0)))
    harmonic_abs = clamp((0.60 * chroma_focus) + (0.25 * (1.0 - flatness_mean)) + (0.15 * contrast_mean))
    groove_abs = clamp(float(np.percentile(pulse, 75)) / 0.9 if pulse.size else 0.0)
    return {
        "drums_abs": drums_abs,
        "harmonic_abs": harmonic_abs,
        "groove_abs": groove_abs,
    }


def load_track_audio(track: ImportedTrack, settings: RuntimeSettings) -> tuple[np.ndarray, int, TrackDspArtifacts]:
    y, sr = librosa.load(
        track.file_path.as_posix(),
        sr=settings.analysis.sample_rate,
        mono=settings.analysis.mono,
    )
    if y.size == 0:
        raise ValueError(f"No audio samples decoded for {track.file_path}")
    return y, sr, build_track_dsp_artifacts(y, sr)


def compute_dsp_lane_result(
    track: ImportedTrack,
    settings: RuntimeSettings,
    analysis_mode: str,
) -> DspLaneResult:
    started = time.perf_counter()
    try:
        y, sr, artifacts = load_track_audio(track, settings)
        energy = extract_energy(artifacts=artifacts)
        loudness = extract_loudness(artifacts=artifacts)
        bass_abs = extract_bass_ratio(artifacts=artifacts)
        time_signature = detect_time_signature(artifacts=artifacts)

        # Removed from the critical path for throughput:
        # extract_full_features(artifacts=artifacts)
        full_features = {
            "drums_abs": None,
            "harmonic_abs": None,
            "groove_abs": None,
        }

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        return DspLaneResult.success(
            track_id=track.id,
            y=y,
            sr=sr,
            artifacts=artifacts,
            energy=energy,
            loudness=loudness,
            bass_abs=bass_abs,
            time_signature=time_signature,
            full_features=full_features,
            elapsed_ms=elapsed_ms,
        )
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        return DspLaneResult.failure(track_id=track.id, error=exc, elapsed_ms=elapsed_ms)


def build_fast_analysis_result(
    track: ImportedTrack,
    settings: RuntimeSettings,
    *,
    prefetched_tempocnn_estimate=None,
    prefetched_musicalkeycnn_estimate=None,
    analysis_signature: str | None = None,
) -> FastAnalysisResult:
    if prefetched_tempocnn_estimate is not None and prefetched_tempocnn_estimate.available and prefetched_tempocnn_estimate.bpm is not None:
        bpm = resolve_bpm(
            track.bpm_imported,
            track.bpm_tag,
            {
                "bpm": float(prefetched_tempocnn_estimate.bpm),
                "bpm_confidence": float(prefetched_tempocnn_estimate.confidence or 0.0),
            },
            detected_source="tempocnn",
        )
    elif track.bpm_imported or track.bpm_tag:
        bpm = resolve_bpm(
            track.bpm_imported,
            track.bpm_tag,
            {"bpm": 0.0, "bpm_confidence": 0.0},
            detected_source="metadata_only",
        )
    else:
        raise ValueError("Fast analysis requires either TempoCNN BPM or imported/tagged BPM.")

    if (
        prefetched_musicalkeycnn_estimate is not None
        and prefetched_musicalkeycnn_estimate.available
        and prefetched_musicalkeycnn_estimate.key
        and prefetched_musicalkeycnn_estimate.key_number is not None
        and prefetched_musicalkeycnn_estimate.key_letter is not None
    ):
        key = resolve_key(
            track.key_tag,
            track.key_imported,
            {
                "key": prefetched_musicalkeycnn_estimate.key,
                "key_number": prefetched_musicalkeycnn_estimate.key_number,
                "key_letter": prefetched_musicalkeycnn_estimate.key_letter,
                "key_confidence": float(prefetched_musicalkeycnn_estimate.confidence or 0.0),
                "pitch": prefetched_musicalkeycnn_estimate.details.get("pitch"),
                "mode": prefetched_musicalkeycnn_estimate.details.get("mode"),
                "key_source": "musicalkeycnn",
            },
        )
    else:
        key = resolve_tag_only_key(track.key_tag, track.key_imported)

    return FastAnalysisResult(
        track_id=track.id,
        source_file_hash=track.file_hash,
        bpm=float(bpm["bpm"]),
        bpm_confidence=float(bpm["bpm_confidence"]),
        bpm_source=str(bpm["bpm_source"]),
        key=str(key["key"]),
        key_number=int(key["key_number"]),
        key_letter=str(key["key_letter"]),
        key_confidence=float(key["key_confidence"]),
        key_source=str(key["key_source"]),
        key_imported=key["key_imported"],
        key_tagged=key["key_tagged"],
        key_agreement=key["key_agreement"],
        analyzed_at=utc_now(),
        analysis_signature=analysis_signature or settings.analysis_signature,
        config_signature=settings.config_signature,
    )


def build_analysis_result(
    track: ImportedTrack,
    settings: RuntimeSettings,
    analysis_mode: str,
    *,
    dsp_result: DspLaneResult,
    tempo_backend: str = "tempocnn",
    tempocnn_model: str | None = None,
    tempocnn_accelerator: str = "auto",
    prefetched_tempocnn_estimate=None,
    key_backend: str | None = None,
    musicalkeycnn_model: str | None = None,
    musicalkeycnn_image: str | None = None,
    musicalkeycnn_device: str | None = None,
    musicalkeycnn_policy: str | None = None,
    prefetched_musicalkeycnn_estimate=None,
    prefetched_essentia_semantic_estimate: EssentiaSemanticEstimate | None = None,
    analysis_signature: str | None = None,
) -> AnalysisResult:
    if not dsp_result.available or dsp_result.y is None or dsp_result.sr is None or dsp_result.artifacts is None:
        raise ValueError(dsp_result.error or f"DSP lane was unavailable for {track.file_path}")

    bpm = resolve_bpm_with_backend(
        track,
        dsp_result.y,
        dsp_result.sr,
        tempo_backend=tempo_backend,
        tempocnn_model=tempocnn_model,
        tempocnn_accelerator=tempocnn_accelerator,
        prefetched_tempocnn_estimate=prefetched_tempocnn_estimate,
        artifacts=dsp_result.artifacts,
    )
    effective_key_backend = key_backend or settings.analysis.key_backend
    key = resolve_key_with_backend(
        track,
        dsp_result.y,
        dsp_result.sr,
        key_backend=effective_key_backend,
        musicalkeycnn_model=musicalkeycnn_model or settings.analysis.key_model_path,
        musicalkeycnn_image=musicalkeycnn_image,
        musicalkeycnn_device=musicalkeycnn_device or settings.analysis.key_device,
        musicalkeycnn_policy=musicalkeycnn_policy or settings.analysis.key_policy,
        prefetched_musicalkeycnn_estimate=prefetched_musicalkeycnn_estimate,
    )

    danceability_abs: float | None = None
    arousal_abs: float | None = None
    valence_abs: float | None = None
    mood_aggressive_abs: float | None = None
    mood_party_abs: float | None = None
    mood_relaxed_abs: float | None = None
    energy_essentia_fused: float | None = None
    energy_essentia_bucket: str | None = None
    essentia_semantic_signature: str | None = None
    essentia_semantic_source: str | None = None
    essentia_semantic_inferred_at: str | None = None
    if analysis_mode == "full" and prefetched_essentia_semantic_estimate is not None and prefetched_essentia_semantic_estimate.available:
        danceability_abs = prefetched_essentia_semantic_estimate.danceability_abs
        arousal_abs = prefetched_essentia_semantic_estimate.arousal_abs
        valence_abs = prefetched_essentia_semantic_estimate.valence_abs
        mood_aggressive_abs = prefetched_essentia_semantic_estimate.mood_aggressive_abs
        mood_party_abs = prefetched_essentia_semantic_estimate.mood_party_abs
        mood_relaxed_abs = prefetched_essentia_semantic_estimate.mood_relaxed_abs
        if None not in (
            danceability_abs,
            arousal_abs,
            valence_abs,
            mood_aggressive_abs,
            mood_party_abs,
            mood_relaxed_abs,
        ):
            cal = settings.semantic_calibration
            cal_arousal = cal.calibrate("arousal_abs", float(arousal_abs))
            cal_danceability = cal.calibrate("danceability_abs", float(danceability_abs))
            cal_party = cal.calibrate("mood_party_abs", float(mood_party_abs))
            cal_relaxed = cal.calibrate("mood_relaxed_abs", float(mood_relaxed_abs))
            cal_aggressive = cal.calibrate("mood_aggressive_abs", float(mood_aggressive_abs))
            energy_essentia_fused = clamp(
                (0.34 * cal_arousal)
                + (0.24 * cal_danceability)
                + (0.18 * cal_party)
                + (0.14 * (1.0 - cal_relaxed))
                + (0.05 * cal_aggressive)
                + (0.03 * float(dsp_result.loudness["loudness_norm"]))
                + (0.02 * float(dsp_result.bass_abs or 0.0))
            )
            if energy_essentia_fused < 0.30:
                energy_essentia_bucket = "low"
            elif energy_essentia_fused < 0.55:
                energy_essentia_bucket = "groove"
            elif energy_essentia_fused < 0.78:
                energy_essentia_bucket = "drive"
            else:
                energy_essentia_bucket = "peak"
        essentia_semantic_signature = str(
            prefetched_essentia_semantic_estimate.details.get("model_signature") or ""
        ) or None
        essentia_semantic_source = str(
            prefetched_essentia_semantic_estimate.details.get("semantic_source") or ""
        ) or None
        essentia_semantic_inferred_at = utc_now()

    return AnalysisResult(
        track_id=track.id,
        source_file_hash=track.file_hash,
        bpm=float(bpm["bpm"]),
        bpm_confidence=float(bpm["bpm_confidence"]),
        bpm_source=str(bpm["bpm_source"]),
        time_signature=str(dsp_result.time_signature["time_signature"]),
        time_signature_confidence=float(dsp_result.time_signature["time_signature_confidence"]),
        key=str(key["key"]),
        key_number=int(key["key_number"]),
        key_letter=str(key["key_letter"]),
        key_confidence=float(key["key_confidence"]),
        key_source=str(key["key_source"]),
        key_imported=key["key_imported"],
        key_tagged=key["key_tagged"],
        key_agreement=key["key_agreement"],
        energy_abs=float(energy_essentia_fused) if energy_essentia_fused is not None else float(dsp_result.energy["energy_abs"]),
        energy_heuristic_abs=float(dsp_result.energy["energy_abs"]),
        energy_sustained=dsp_result.energy["energy_sustained"],
        energy_peak=dsp_result.energy["energy_peak"],
        danceability_abs=danceability_abs,
        arousal_abs=arousal_abs,
        valence_abs=valence_abs,
        mood_aggressive_abs=mood_aggressive_abs,
        mood_party_abs=mood_party_abs,
        mood_relaxed_abs=mood_relaxed_abs,
        energy_essentia_fused=energy_essentia_fused,
        energy_essentia_bucket=energy_essentia_bucket,
        essentia_semantic_signature=essentia_semantic_signature,
        essentia_semantic_source=essentia_semantic_source,
        essentia_semantic_inferred_at=essentia_semantic_inferred_at,
        loudness_lufs=dsp_result.loudness["loudness_lufs"],
        loudness_norm=dsp_result.loudness["loudness_norm"],
        bass_abs=float(dsp_result.bass_abs),
        drums_abs=dsp_result.full_features["drums_abs"],
        harmonic_abs=dsp_result.full_features["harmonic_abs"],
        groove_abs=dsp_result.full_features["groove_abs"],
        vocals_abs=None,
        vocals_confidence=None,
        analysis_mode=analysis_mode,
        analyzed_at=utc_now(),
        analysis_signature=analysis_signature or settings.analysis_signature,
        config_signature=settings.config_signature,
    )


def analyze_track(
    track: ImportedTrack,
    settings: RuntimeSettings,
    analysis_mode: str,
    *,
    tempo_backend: str = "tempocnn",
    tempocnn_model: str | None = None,
    tempocnn_accelerator: str = "auto",
    prefetched_tempocnn_estimate=None,
    key_backend: str | None = None,
    musicalkeycnn_model: str | None = None,
    musicalkeycnn_image: str | None = None,
    musicalkeycnn_device: str | None = None,
    musicalkeycnn_policy: str | None = None,
    prefetched_musicalkeycnn_estimate=None,
    prefetched_essentia_semantic_estimate: EssentiaSemanticEstimate | None = None,
    analysis_signature: str | None = None,
) -> AnalysisResult:
    dsp_result = compute_dsp_lane_result(track, settings, analysis_mode)
    return build_analysis_result(
        track,
        settings,
        analysis_mode,
        dsp_result=dsp_result,
        tempo_backend=tempo_backend,
        tempocnn_model=tempocnn_model,
        tempocnn_accelerator=tempocnn_accelerator,
        prefetched_tempocnn_estimate=prefetched_tempocnn_estimate,
        key_backend=key_backend,
        musicalkeycnn_model=musicalkeycnn_model,
        musicalkeycnn_image=musicalkeycnn_image,
        musicalkeycnn_device=musicalkeycnn_device,
        musicalkeycnn_policy=musicalkeycnn_policy,
        prefetched_musicalkeycnn_estimate=prefetched_musicalkeycnn_estimate,
        prefetched_essentia_semantic_estimate=prefetched_essentia_semantic_estimate,
        analysis_signature=analysis_signature,
    )
