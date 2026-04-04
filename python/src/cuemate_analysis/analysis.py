from __future__ import annotations

from datetime import datetime, timezone
import math
import re

import librosa
import numpy as np
import pyloudnorm as pyln

from cuemate_analysis.config import RuntimeSettings
from cuemate_analysis.energy_model import build_energy_feature_vector, predict_energy_from_features, resolve_energy_model_meta_path, resolve_energy_model_path
from cuemate_analysis.models import AnalysisResult, ImportedTrack


PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
FILE_TAG_BPM_CONFIDENCE = 0.90
FILE_TAG_KEY_CONFIDENCE = 0.85

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
MUSICALKEYCNN_OVERRIDE_CONFIDENCE = 0.5


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


def detect_bpm(y: np.ndarray, sr: int) -> dict[str, float]:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_value, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = float(np.atleast_1d(tempo_value)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) >= 2:
        ibis = np.diff(beat_times)
        ibi_mean = float(np.mean(ibis)) if ibis.size else 0.0
        ibi_std = float(np.std(ibis)) if ibis.size else 0.0
        bpm_confidence = clamp(1.0 - ((ibi_std / max(ibi_mean, 1e-6)) * 5.0))
    else:
        bpm_confidence = 0.3

    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    peak_sharpness = float(np.max(tempogram) / (np.mean(tempogram) + 1e-10)) if tempogram.size else 0.0
    tempogram_confidence = clamp(peak_sharpness / 20.0)
    combined_confidence = clamp((0.5 * bpm_confidence) + (0.5 * tempogram_confidence))

    return {"bpm": max(tempo, 0.0), "bpm_confidence": combined_confidence}


def resolve_bpm(
    imported_bpm: float | None,
    tagged_bpm: float | None,
    detected: dict[str, float],
    *,
    detected_source: str = "detected",
) -> dict[str, float | str]:
    detected_bpm = float(detected.get("bpm", 0.0))
    detected_confidence = float(detected.get("bpm_confidence", 0.0))

    if imported_bpm and imported_bpm > 0:
        source = "imported"
        confidence = 0.95
        if detected_bpm > 0:
            ratios = [1.0, 2.0, 0.5]
            best_ratio = min(ratios, key=lambda ratio: abs(imported_bpm - (detected_bpm * ratio)))
            matched = detected_bpm * best_ratio
            tolerance = 1.0 if math.isclose(best_ratio, 1.0) else 2.0
            if abs(imported_bpm - matched) <= tolerance:
                source = f"imported+{detected_source}"
                confidence = 0.98
        return {"bpm": float(imported_bpm), "bpm_confidence": confidence, "bpm_source": source}

    if tagged_bpm and tagged_bpm > 0:
        source = "tag"
        confidence = FILE_TAG_BPM_CONFIDENCE
        if detected_bpm > 0:
            ratios = [1.0, 2.0, 0.5]
            best_ratio = min(ratios, key=lambda ratio: abs(tagged_bpm - (detected_bpm * ratio)))
            matched = detected_bpm * best_ratio
            tolerance = 1.0 if math.isclose(best_ratio, 1.0) else 2.0
            if abs(tagged_bpm - matched) <= tolerance:
                source = f"tag+{detected_source}"
                confidence = 0.98
        return {"bpm": float(tagged_bpm), "bpm_confidence": confidence, "bpm_source": source}

    if detected_bpm > 0:
        return {
            "bpm": detected_bpm,
            "bpm_confidence": max(detected_confidence, 0.55),
            "bpm_source": detected_source,
        }

    raise ValueError("Unable to resolve BPM from tags or detection.")


def resolve_bpm_with_backend(
    track: ImportedTrack,
    y: np.ndarray,
    sr: int,
    *,
    tempo_backend: str,
    tempocnn_model: str | None = None,
    tempocnn_accelerator: str = "auto",
    prefetched_tempocnn_estimate=None,
) -> dict[str, float | str]:
    if tempo_backend == "baseline":
        return resolve_bpm(track.bpm_imported, track.bpm_tag, detect_bpm(y, sr), detected_source="detected")

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
            {"bpm": float(estimate.bpm), "bpm_confidence": float(estimate.confidence or 0.0)},
            detected_source="tempocnn",
        )

    baseline = detect_bpm(y, sr)
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


def resolve_key(
    tagged_key: str | None,
    imported_key: str | None,
    detected: dict[str, str | int | float],
) -> dict[str, str | int | float | None]:
    parsed_tag = parse_key_label(tagged_key)
    parsed_imported = parse_key_label(imported_key)
    detected_source = str(detected.get("key_source", "chroma"))
    detected_confidence = float(detected.get("key_confidence", 0.0) or 0.0)

    if parsed_tag is not None:
        if parsed_tag["key"] == detected["key"]:
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
            }

        if (
            detected_source == "musicalkeycnn"
            and detected_confidence >= MUSICALKEYCNN_OVERRIDE_CONFIDENCE
        ):
            return {
                "key": detected["key"],
                "key_number": detected["key_number"],
                "key_letter": detected["key_letter"],
                "key_confidence": clamp(min(detected_confidence, FILE_TAG_KEY_CONFIDENCE) * 0.6),
                "key_source": "musicalkeycnn_override_tag",
                "key_imported": imported_key,
                "key_tagged": tagged_key,
                "key_agreement": 0,
            }

        return {
            **parsed_tag,
            "key_confidence": clamp(min(detected_confidence, FILE_TAG_KEY_CONFIDENCE) * 0.6),
            "key_source": "tag_conflicted",
            "key_imported": imported_key,
            "key_tagged": tagged_key,
            "key_agreement": 0,
        }

    if parsed_imported is not None:
        if parsed_imported["key"] == detected["key"]:
            return {
                **parsed_imported,
                "key_confidence": bayesian_key_confidence(
                    detected_confidence,
                    0.80,
                    provenance_independent=False,
                ),
                "key_source": f"imported+{detected_source}",
                "key_imported": imported_key,
                "key_tagged": tagged_key,
                "key_agreement": 1,
            }

        if (
            detected_source == "musicalkeycnn"
            and detected_confidence >= MUSICALKEYCNN_OVERRIDE_CONFIDENCE
        ):
            return {
                "key": detected["key"],
                "key_number": detected["key_number"],
                "key_letter": detected["key_letter"],
                "key_confidence": clamp(min(detected_confidence, 0.80) * 0.6),
                "key_source": "musicalkeycnn_override_imported",
                "key_imported": imported_key,
                "key_tagged": tagged_key,
                "key_agreement": 0,
            }

        return {
            **parsed_imported,
            "key_confidence": clamp(min(detected_confidence, 0.80) * 0.6),
            "key_source": "imported_conflicted",
            "key_imported": imported_key,
            "key_tagged": tagged_key,
            "key_agreement": 0,
        }

    return {
        "key": detected["key"],
        "key_number": detected["key_number"],
        "key_letter": detected["key_letter"],
        "key_confidence": detected["key_confidence"],
        "key_source": detected_source,
        "key_imported": imported_key,
        "key_tagged": tagged_key,
        "key_agreement": None,
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
        }

    parsed_imported = parse_key_label(imported_key)
    if parsed_imported is not None:
        return {
            **parsed_imported,
            "key_confidence": 0.80,
            "key_source": "import_only_fallback",
            "key_imported": imported_key,
            "key_tagged": tagged_key,
            "key_agreement": None,
        }

    raise RuntimeError("MusicalKeyCNN was unavailable and no usable tagged or imported key was found; chroma fallback is disabled.")


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
    if estimate.available and estimate.key and estimate.key_number is not None and estimate.key_letter is not None:
        return resolve_key(
            track.key_tag,
            track.key_imported,
            {
                "key": estimate.key,
                "key_number": estimate.key_number,
                "key_letter": estimate.key_letter,
                "key_confidence": float(estimate.confidence or 0.0),
                "pitch": estimate.details.get("pitch"),
                "mode": estimate.details.get("mode"),
                "key_source": "musicalkeycnn",
            },
        )

    return resolve_tag_only_key(track.key_tag, track.key_imported)


def extract_energy(y: np.ndarray, sr: int) -> dict[str, float]:
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    sustained = float(np.percentile(rms, 75))
    peak = float(np.percentile(rms, 95))
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    brightness = clamp(float(np.mean(centroid)) / max(sr / 2.0, 1.0))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
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


def detect_time_signature(y: np.ndarray, sr: int) -> dict[str, str | float]:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    if len(beat_frames) < 8:
        return {"time_signature": "4/4", "time_signature_confidence": 0.35}

    beat_strengths = onset_env[beat_frames]
    if beat_strengths.size < 8 or float(np.mean(beat_strengths)) <= 1e-6:
        return {"time_signature": "4/4", "time_signature_confidence": 0.4}

    candidates = ["3/4", "4/4", "5/4"]
    scores: dict[str, float] = {}
    normalized_strengths = beat_strengths / max(float(np.mean(beat_strengths)), 1e-6)

    for signature in candidates:
        meter = int(signature.split("/")[0])
        grouped = [normalized_strengths[index::meter] for index in range(meter)]
        means = np.array([float(np.mean(group)) if group.size else 0.0 for group in grouped], dtype=float)
        accent_contrast = float(np.max(means) - np.min(means))
        downbeat_strength = float(np.max(means))
        regularity = clamp(len(beat_frames) / float(meter * 8))
        scores[signature] = (accent_contrast * 0.5) + (downbeat_strength * 0.25) + (regularity * 0.25)

    best_signature = max(scores, key=scores.get)
    ordered_scores = sorted(scores.values(), reverse=True)
    best_score = ordered_scores[0]
    second_score = ordered_scores[1] if len(ordered_scores) > 1 else 0.0
    confidence = clamp(0.35 + ((best_score - second_score) * 0.45) + (0.20 if best_signature == "4/4" else 0.0))
    return {"time_signature": best_signature, "time_signature_confidence": confidence}


def extract_loudness(y: np.ndarray, sr: int) -> dict[str, float]:
    meter = pyln.Meter(sr)
    loudness_lufs = float(meter.integrated_loudness(y.astype(np.float64)))
    return {
        "loudness_lufs": loudness_lufs,
        "loudness_norm": normalize_loudness(loudness_lufs),
    }


def extract_bass_ratio(y: np.ndarray, sr: int) -> float:
    spectrum = np.abs(librosa.stft(y=y, n_fft=2048, hop_length=512))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bass_mask = (frequencies >= 30.0) & (frequencies <= 150.0)
    total_mask = (frequencies >= 30.0) & (frequencies <= 8000.0)
    bass_energy = float(np.sum(spectrum[bass_mask]))
    total_energy = float(np.sum(spectrum[total_mask])) or 1e-6
    return clamp(bass_energy / total_energy)


def extract_full_features(y: np.ndarray, sr: int) -> dict[str, float]:
    harmonic, percussive = librosa.effects.hpss(y)
    total_rms = float(np.sqrt(np.mean(np.square(y)))) or 1e-6
    percussive_rms = float(np.sqrt(np.mean(np.square(percussive))))
    harmonic_rms = float(np.sqrt(np.mean(np.square(harmonic))))
    onset_env = librosa.onset.onset_strength(y=percussive, sr=sr)
    duration_seconds = max(len(y) / float(sr), 1e-6)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_rate = len(onset_frames) / duration_seconds
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    try:
        chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
    except Exception:
        chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr)
    chroma_focus = float(np.mean(np.max(chroma, axis=0))) if chroma.size else 0.0

    drums_abs = clamp((0.7 * (percussive_rms / total_rms)) + (0.3 * clamp(onset_rate / 6.0)))
    harmonic_abs = clamp((0.6 * (harmonic_rms / total_rms)) + (0.4 * chroma_focus))
    groove_abs = clamp(float(np.percentile(pulse, 75)) / 0.9 if pulse.size else 0.0)
    return {
        "drums_abs": drums_abs,
        "harmonic_abs": harmonic_abs,
        "groove_abs": groove_abs,
    }


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
    analysis_signature: str | None = None,
) -> AnalysisResult:
    y, sr = librosa.load(
        track.file_path.as_posix(),
        sr=settings.analysis.sample_rate,
        mono=settings.analysis.mono,
    )
    if y.size == 0:
        raise ValueError(f"No audio samples decoded for {track.file_path}")

    bpm = resolve_bpm_with_backend(
        track,
        y,
        sr,
        tempo_backend=tempo_backend,
        tempocnn_model=tempocnn_model,
        tempocnn_accelerator=tempocnn_accelerator,
        prefetched_tempocnn_estimate=prefetched_tempocnn_estimate,
    )
    effective_key_backend = key_backend or settings.analysis.key_backend
    key = resolve_key_with_backend(
        track,
        y,
        sr,
        key_backend=effective_key_backend,
        musicalkeycnn_model=musicalkeycnn_model or settings.analysis.key_model_path,
        musicalkeycnn_image=musicalkeycnn_image,
        musicalkeycnn_device=musicalkeycnn_device or settings.analysis.key_device,
        musicalkeycnn_policy=musicalkeycnn_policy or settings.analysis.key_policy,
        prefetched_musicalkeycnn_estimate=prefetched_musicalkeycnn_estimate,
    )
    energy = extract_energy(y, sr)
    loudness = extract_loudness(y, sr)
    bass_abs = extract_bass_ratio(y, sr)
    time_signature = detect_time_signature(y, sr)

    if analysis_mode == "full":
        full_features = extract_full_features(y, sr)
    else:
        full_features = {
            "drums_abs": None,
            "harmonic_abs": None,
            "groove_abs": None,
        }

    energy_features = build_energy_feature_vector(
        energy_abs=float(energy["energy_abs"]),
        energy_sustained=energy["energy_sustained"],
        energy_peak=energy["energy_peak"],
        loudness_norm=float(loudness["loudness_norm"]),
        loudness_lufs=float(loudness["loudness_lufs"]),
        bass_abs=bass_abs,
        drums_abs=full_features["drums_abs"],
        harmonic_abs=full_features["harmonic_abs"],
        groove_abs=full_features["groove_abs"],
    )
    energy_hybrid: float | None = None
    energy_learned: float | None = None
    energy_learned_bucket: str | None = None
    energy_model_signature: str | None = None
    energy_model_source: str | None = None
    energy_model_inferred_at: str | None = None
    if analysis_mode == "full" and settings.analysis.energy_parallel_enabled:
        model_path = resolve_energy_model_path(settings.analysis.energy_model_path, settings.repo_root)
        meta_path = resolve_energy_model_meta_path(settings.analysis.energy_model_meta_path, settings.repo_root)
        if model_path.is_file() and meta_path.is_file():
            try:
                energy_inference = predict_energy_from_features(
                    energy_features,
                    model_path=model_path,
                    meta_path=meta_path,
                )
            except Exception:
                energy_inference = None
            if energy_inference is not None:
                energy_hybrid = float(energy_inference.hybrid)
                energy_learned = float(energy_inference.learned)
                energy_learned_bucket = str(energy_inference.bucket)
                energy_model_signature = str(energy_inference.model_signature)
                energy_model_source = str(energy_inference.model_source)
                energy_model_inferred_at = utc_now()

    return AnalysisResult(
        track_id=track.id,
        source_file_hash=track.file_hash,
        bpm=float(bpm["bpm"]),
        bpm_confidence=float(bpm["bpm_confidence"]),
        bpm_source=str(bpm["bpm_source"]),
        time_signature=str(time_signature["time_signature"]),
        time_signature_confidence=float(time_signature["time_signature_confidence"]),
        key=str(key["key"]),
        key_number=int(key["key_number"]),
        key_letter=str(key["key_letter"]),
        key_confidence=float(key["key_confidence"]),
        key_source=str(key["key_source"]),
        key_imported=key["key_imported"],
        key_tagged=key["key_tagged"],
        key_agreement=key["key_agreement"],
        energy_abs=float(energy["energy_abs"]),
        energy_sustained=energy["energy_sustained"],
        energy_peak=energy["energy_peak"],
        energy_hybrid=energy_hybrid,
        energy_learned=energy_learned,
        energy_learned_bucket=energy_learned_bucket,
        energy_model_signature=energy_model_signature,
        energy_model_source=energy_model_source,
        energy_model_inferred_at=energy_model_inferred_at,
        loudness_lufs=loudness["loudness_lufs"],
        loudness_norm=loudness["loudness_norm"],
        bass_abs=bass_abs,
        drums_abs=full_features["drums_abs"],
        harmonic_abs=full_features["harmonic_abs"],
        groove_abs=full_features["groove_abs"],
        vocals_abs=None,
        vocals_confidence=None,
        analysis_mode=analysis_mode,
        analyzed_at=utc_now(),
        analysis_signature=analysis_signature or settings.analysis_signature,
        config_signature=settings.config_signature,
    )
