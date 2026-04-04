import numpy as np
from pathlib import Path

from cuemate_analysis.analysis import resolve_bpm, resolve_bpm_with_backend, resolve_key_with_backend
from cuemate_analysis.cli import build_effective_analysis_signature
from cuemate_analysis.analysis import parse_key_label
from cuemate_analysis.ingest import discover_audio_files, make_playlist_id, make_track_id
from cuemate_analysis.models import ImportedTrack
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
    resolved = resolve_bpm(None, 124.0, {"bpm": 124.1, "bpm_confidence": 0.88}, detected_source="tempocnn")

    assert resolved["bpm"] == 124.0
    assert resolved["bpm_source"] == "tag+tempocnn"


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
    assert signature.endswith("-auto-full_track")


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
    assert resolved["key_source"] == "musicalkeycnn_override_tag"
    assert resolved["key_tagged"] == "2A"
    assert resolved["key_agreement"] == 0


def test_resolve_bpm_prefers_imported_bpm_when_present() -> None:
    resolved = resolve_bpm(128.0, 124.0, {"bpm": 128.1, "bpm_confidence": 0.90}, detected_source="tempocnn")

    assert resolved["bpm"] == 128.0
    assert resolved["bpm_source"] == "imported+tempocnn"
