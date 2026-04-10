import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace

from cuemate_analysis.analysis import build_analysis_result, resolve_bpm, resolve_bpm_with_backend, resolve_key_with_backend
from cuemate_analysis.config import build_relative_experiment_signature, load_runtime_settings
from cuemate_analysis.cli import build_effective_analysis_signature
from cuemate_analysis.analysis import parse_key_label
from cuemate_analysis.ingest import discover_audio_files, make_playlist_id, make_track_id
from cuemate_analysis.models import ImportedTrack
from cuemate_analysis.scoring import SCORING_CONTRACT_ID
from cuemate_analysis.tempo_backend import TempoEstimate
from cuemate_analysis.key_backend import KeyEstimate


def test_parse_key_label_supports_camelot_and_note_names() -> None:
    assert parse_key_label("8A") == {"key": "8A", "key_number": 8, "key_letter": "A"}
    assert parse_key_label("Am") == {"key": "8A", "key_number": 8, "key_letter": "A"}
    assert parse_key_label("C") == {"key": "8B", "key_number": 8, "key_letter": "B"}


def test_make_ids_are_stable_for_same_input(tmp_path: Path) -> None:
    track_path = tmp_path / "track.wav"
    track_path.write_bytes(b"test")

    assert make_track_id(track_path) == make_track_id(track_path)
    assert make_playlist_id("Smoke Test") == make_playlist_id("Smoke Test")


def test_discover_audio_files_filters_supported_extensions(tmp_path: Path) -> None:
    audio_file = tmp_path / "track.wav"
    ignored_file = tmp_path / "notes.txt"
    audio_file.write_bytes(b"audio")
    ignored_file.write_text("ignore me", encoding="utf-8")

    discovered = discover_audio_files([tmp_path])

    assert discovered == [audio_file.resolve()]


def test_resolve_bpm_uses_backend_name_in_source() -> None:
    # tag=124.0 and detected=124.1 (0.1 BPM apart) agree via bpm_agreement_delta.
    # Model priority boost (conf=0.88 >= 0.85) gives detected the edge → winner is detected.
    # Source label reflects winner (tempocnn) + corroborator (tag).
    resolved = resolve_bpm(None, 124.0, {"bpm": 124.1, "bpm_confidence": 0.88}, detected_source="tempocnn")

    assert resolved["bpm"] == 124.1
    assert resolved["bpm_source"] == "tempocnn+tag"


def test_effective_analysis_signature_includes_production_models(tmp_path: Path) -> None:
    tempo_model = tmp_path / "deepsquare-k16-3.pb"
    key_model = tmp_path / "keynet.pt"
    tempo_model.write_bytes(b"tempo")
    key_model.write_bytes(b"key")

    signature = build_effective_analysis_signature(
        "m1-stable",
        tempocnn_model=str(tempo_model),
        tempocnn_accelerator="auto",
        musicalkeycnn_model=str(key_model),
        musicalkeycnn_device="auto",
    )

    assert signature.startswith("m1-stable-tempo-tempocnn-")
    assert "-auto-key-musicalkeycnn-" in signature
    assert "-auto-full_track-essentia-" in signature


def test_build_analysis_result_tags_scoring_contract(monkeypatch, tmp_path: Path) -> None:
    settings = load_runtime_settings()
    track_path = tmp_path / "track.wav"
    track_path.write_bytes(b"test")
    track = ImportedTrack(
        id="trk_test",
        file_path=track_path,
        file_hash="hash",
        title="Track",
        artist="Artist",
        genre=None,
        duration_seconds=10.0,
        bpm_imported=None,
        bpm_tag=None,
        key_imported=None,
        key_tag=None,
    )

    monkeypatch.setattr(
        "cuemate_analysis.analysis.resolve_bpm_with_backend",
        lambda *args, **kwargs: {
            "bpm": 128.0,
            "bpm_confidence": 0.9,
            "bpm_source": "tempocnn",
        },
    )
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

    dsp_result = SimpleNamespace(
        available=True,
        y=np.zeros(22050, dtype=float),
        sr=22050,
        artifacts=object(),
        energy={"energy_abs": 0.6, "energy_sustained": 0.5, "energy_peak": 0.7},
        loudness={"loudness_lufs": -8.5, "loudness_norm": 0.72},
        bass_abs=0.4,
        time_signature={"time_signature": "4/4", "time_signature_confidence": 0.8},
        full_features={"drums_abs": 0.5, "harmonic_abs": 0.4, "groove_abs": 0.3},
    )

    result = build_analysis_result(
        track,
        settings,
        "full",
        dsp_result=dsp_result,
    )

    assert result.scoring_contract_id_at_analysis == SCORING_CONTRACT_ID


def test_relative_signature_changes_with_energy_source() -> None:
    settings = load_runtime_settings()

    canonical = build_relative_experiment_signature(settings, energy_source="canonical")
    legacy = build_relative_experiment_signature(settings, energy_source="heuristic_legacy")

    assert canonical != legacy


def test_resolve_bpm_with_backend_falls_back_to_baseline_when_tempocnn_is_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    track_path = tmp_path / "track.wav"
    track_path.write_bytes(b"test")
    track = ImportedTrack(
        id="trk_test",
        file_path=track_path,
        file_hash="hash",
        title="Track",
        artist="Artist",
        genre=None,
        duration_seconds=10.0,
        bpm_imported=None,
        bpm_tag=None,
        key_imported=None,
        key_tag=None,
    )

    monkeypatch.setattr(
        "cuemate_analysis.tempo_backend.estimate_tempocnn_bpm",
        lambda *args, **kwargs: TempoEstimate(
            backend="tempocnn",
            bpm=None,
            confidence=None,
            elapsed_ms=1.0,
            details={},
            notes=["TempoCNN unavailable"],
            available=False,
        ),
    )
    monkeypatch.setattr(
        "cuemate_analysis.analysis.detect_bpm",
        lambda y, sr: {"bpm": 128.0, "bpm_confidence": 0.81},
    )

    resolved = resolve_bpm_with_backend(
        track,
        np.zeros(22050, dtype=float),
        22050,
        tempo_backend="tempocnn",
    )

    assert resolved["bpm"] == 128.0
    assert resolved["bpm_source"] == "baseline_fallback"


def test_resolve_key_with_backend_falls_back_to_tag_when_musicalkeycnn_is_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    track_path = tmp_path / "track.wav"
    track_path.write_bytes(b"test")
    track = ImportedTrack(
        id="trk_test",
        file_path=track_path,
        file_hash="hash",
        title="Track",
        artist="Artist",
        genre=None,
        duration_seconds=10.0,
        bpm_imported=None,
        bpm_tag=None,
        key_imported=None,
        key_tag="8A",
    )

    monkeypatch.setattr(
        "cuemate_analysis.key_backend.estimate_musicalkeycnn_key",
        lambda *args, **kwargs: KeyEstimate(
            backend="musicalkeycnn",
            key=None,
            key_number=None,
            key_letter=None,
            confidence=None,
            elapsed_ms=1.0,
            details={},
            notes=["MusicalKeyCNN unavailable"],
            available=False,
        ),
    )
    resolved = resolve_key_with_backend(
        track,
        np.zeros(22050, dtype=float),
        22050,
        key_backend="musicalkeycnn",
    )

    assert resolved["key"] == "8A"
    assert resolved["key_source"] == "tag_only_fallback"


def test_resolve_key_prefers_high_confidence_musicalkeycnn_over_conflicting_tag() -> None:
    resolved = resolve_key_with_backend(
        ImportedTrack(
            id="trk_test",
            file_path=Path("D:/fake/track.wav"),
            file_hash="hash",
            title="Track",
            artist="Artist",
            genre=None,
            duration_seconds=10.0,
            bpm_imported=None,
            bpm_tag=None,
            key_imported=None,
            key_tag="2A",
        ),
        np.zeros(22050, dtype=float),
        22050,
        key_backend="musicalkeycnn",
        prefetched_musicalkeycnn_estimate=KeyEstimate(
            backend="musicalkeycnn",
            key="3A",
            key_number=3,
            key_letter="A",
            confidence=0.78,
            elapsed_ms=10.0,
            details={"pitch": "A#", "mode": "minor"},
            notes=[],
            available=True,
        ),
    )

    assert resolved["key"] == "3A"
    # key_source is "musicalkeycnn" when the model wins but doesn't append _override_tag suffix
    assert resolved["key_source"] == "musicalkeycnn"
    assert resolved["key_tagged"] == "2A"
    assert resolved["key_agreement"] == 0


def test_resolve_bpm_prefers_imported_bpm_when_present() -> None:
    # imported=128.0 and detected=128.1 agree; tag=124.0 conflicts with both.
    # Model priority (conf=0.90 >= 0.85) gives detected the final edge.
    # Winner is detected (tempocnn); imported is its corroborator.
    resolved = resolve_bpm(128.0, 124.0, {"bpm": 128.1, "bpm_confidence": 0.90}, detected_source="tempocnn")

    assert resolved["bpm"] == pytest.approx(128.1)
    assert resolved["bpm_source"] == "tempocnn+imported"
