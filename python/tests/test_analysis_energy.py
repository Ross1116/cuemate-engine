from dataclasses import replace
from pathlib import Path

import numpy as np

from cuemate_analysis.analysis import analyze_track, extract_energy
from cuemate_analysis.config import load_runtime_settings
from cuemate_analysis.energy_model import EnergyInferenceResult
from cuemate_analysis.models import ImportedTrack


def test_extract_energy_discriminates_without_saturating() -> None:
    sr = 22050
    duration_seconds = 4.0
    timeline = np.linspace(0.0, duration_seconds, int(sr * duration_seconds), endpoint=False, dtype=np.float32)

    quiet = 0.05 * np.sin(2.0 * np.pi * 220.0 * timeline)

    lively = (0.32 * np.sin(2.0 * np.pi * 220.0 * timeline)) + (0.18 * np.sin(2.0 * np.pi * 2200.0 * timeline))
    pulse_centers = np.arange(0.4, duration_seconds, 0.5)
    click_envelope = np.zeros_like(timeline)
    width = int(sr * 0.02)
    for center in pulse_centers:
        start = max(int(center * sr) - (width // 2), 0)
        stop = min(start + width, click_envelope.size)
        ramp = np.hanning(max(stop - start, 2))[: stop - start]
        click_envelope[start:stop] += ramp
    lively = lively + (0.35 * click_envelope.astype(np.float32))

    quiet_energy = extract_energy(quiet.astype(np.float32), sr)
    lively_energy = extract_energy(lively.astype(np.float32), sr)

    assert 0.0 <= quiet_energy["energy_abs"] < lively_energy["energy_abs"] < 1.0
    assert 0.0 <= quiet_energy["energy_sustained"] <= 1.0
    assert 0.0 <= quiet_energy["energy_peak"] <= 1.0
    assert 0.0 <= lively_energy["energy_sustained"] <= 1.0
    assert 0.0 <= lively_energy["energy_peak"] <= 1.0


def test_analyze_track_populates_parallel_learned_energy_in_full_mode(monkeypatch, tmp_path: Path) -> None:
    settings = load_runtime_settings()
    settings = replace(
        settings,
        analysis=replace(
            settings.analysis,
            energy_parallel_enabled=True,
            energy_model_path="python/models/energy/teacher_first_v1.joblib",
            energy_model_meta_path="python/models/energy/teacher_first_v1.meta.json",
        ),
    )
    model_path = tmp_path / "teacher_first_v1.joblib"
    meta_path = tmp_path / "teacher_first_v1.meta.json"
    model_path.write_bytes(b"model")
    meta_path.write_text("{}", encoding="utf-8")
    track_path = tmp_path / "track.wav"
    track_path.write_bytes(b"audio")
    track = ImportedTrack(
        id="trk_energy",
        file_path=track_path,
        file_hash="hash-energy",
        title="Track",
        artist="Artist",
        genre=None,
        duration_seconds=180.0,
        bpm_imported=None,
        bpm_tag=None,
        key_imported=None,
        key_tag=None,
    )

    monkeypatch.setattr("cuemate_analysis.analysis.librosa.load", lambda *args, **kwargs: (np.zeros(22050, dtype=float), 22050))
    monkeypatch.setattr("cuemate_analysis.analysis.resolve_bpm_with_backend", lambda *args, **kwargs: {"bpm": 128.0, "bpm_confidence": 0.9, "bpm_source": "tempocnn"})
    monkeypatch.setattr(
        "cuemate_analysis.analysis.resolve_key_with_backend",
        lambda *args, **kwargs: {
            "key": "8A",
            "key_number": 8,
            "key_letter": "A",
            "key_confidence": 0.8,
            "key_source": "musicalkeycnn",
            "key_imported": None,
            "key_tagged": None,
            "key_agreement": None,
        },
    )
    monkeypatch.setattr("cuemate_analysis.analysis.extract_energy", lambda *args, **kwargs: {"energy_abs": 0.6, "energy_sustained": 0.55, "energy_peak": 0.7})
    monkeypatch.setattr("cuemate_analysis.analysis.extract_loudness", lambda *args, **kwargs: {"loudness_lufs": -9.0, "loudness_norm": 0.8})
    monkeypatch.setattr("cuemate_analysis.analysis.extract_bass_ratio", lambda *args, **kwargs: 0.42)
    monkeypatch.setattr("cuemate_analysis.analysis.detect_time_signature", lambda *args, **kwargs: {"time_signature": "4/4", "time_signature_confidence": 0.75})
    monkeypatch.setattr("cuemate_analysis.analysis.extract_full_features", lambda *args, **kwargs: {"drums_abs": 0.7, "harmonic_abs": 0.4, "groove_abs": 0.6})
    monkeypatch.setattr("cuemate_analysis.analysis.resolve_energy_model_path", lambda *args, **kwargs: model_path)
    monkeypatch.setattr("cuemate_analysis.analysis.resolve_energy_model_meta_path", lambda *args, **kwargs: meta_path)
    monkeypatch.setattr(
        "cuemate_analysis.analysis.predict_energy_from_features",
        lambda *args, **kwargs: EnergyInferenceResult(
            hybrid=0.64,
            learned=0.68,
            bucket="drive",
            model_signature="sig123",
            model_source="teacher_first_hgbr",
        ),
    )

    result = analyze_track(track, settings, "full")

    assert result.energy_abs == 0.6
    assert result.energy_hybrid == 0.64
    assert result.energy_learned == 0.68
    assert result.energy_learned_bucket == "drive"
    assert result.energy_model_signature == "sig123"
    assert result.energy_model_source == "teacher_first_hgbr"


def test_analyze_track_leaves_parallel_learned_energy_empty_in_fast_pass(monkeypatch, tmp_path: Path) -> None:
    settings = load_runtime_settings()
    settings = replace(
        settings,
        analysis=replace(settings.analysis, energy_parallel_enabled=True),
    )
    track_path = tmp_path / "track.wav"
    track_path.write_bytes(b"audio")
    track = ImportedTrack(
        id="trk_fast",
        file_path=track_path,
        file_hash="hash-fast",
        title="Track",
        artist="Artist",
        genre=None,
        duration_seconds=180.0,
        bpm_imported=None,
        bpm_tag=None,
        key_imported=None,
        key_tag=None,
    )

    monkeypatch.setattr("cuemate_analysis.analysis.librosa.load", lambda *args, **kwargs: (np.zeros(22050, dtype=float), 22050))
    monkeypatch.setattr("cuemate_analysis.analysis.resolve_bpm_with_backend", lambda *args, **kwargs: {"bpm": 128.0, "bpm_confidence": 0.9, "bpm_source": "tempocnn"})
    monkeypatch.setattr(
        "cuemate_analysis.analysis.resolve_key_with_backend",
        lambda *args, **kwargs: {
            "key": "8A",
            "key_number": 8,
            "key_letter": "A",
            "key_confidence": 0.8,
            "key_source": "musicalkeycnn",
            "key_imported": None,
            "key_tagged": None,
            "key_agreement": None,
        },
    )
    monkeypatch.setattr("cuemate_analysis.analysis.extract_energy", lambda *args, **kwargs: {"energy_abs": 0.6, "energy_sustained": 0.55, "energy_peak": 0.7})
    monkeypatch.setattr("cuemate_analysis.analysis.extract_loudness", lambda *args, **kwargs: {"loudness_lufs": -9.0, "loudness_norm": 0.8})
    monkeypatch.setattr("cuemate_analysis.analysis.extract_bass_ratio", lambda *args, **kwargs: 0.42)
    monkeypatch.setattr("cuemate_analysis.analysis.detect_time_signature", lambda *args, **kwargs: {"time_signature": "4/4", "time_signature_confidence": 0.75})

    result = analyze_track(track, settings, "fast_pass")

    assert result.energy_hybrid is None
    assert result.energy_learned is None
    assert result.energy_learned_bucket is None
