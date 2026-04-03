import numpy as np
from pathlib import Path

from cuemate_analysis.analysis import resolve_bpm, resolve_bpm_with_backend
from cuemate_analysis.cli import build_effective_analysis_signature, parse_backend_list
from cuemate_analysis.analysis import parse_key_label
from cuemate_analysis.ingest import discover_audio_files, make_playlist_id, make_track_id
from cuemate_analysis.models import ImportedTrack
from cuemate_analysis.tempo_experiments import TempoEstimate


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
    resolved = resolve_bpm(124.0, {"bpm": 124.1, "bpm_confidence": 0.88}, detected_source="tempocnn")

    assert resolved["bpm"] == 124.0
    assert resolved["bpm_source"] == "tag+tempocnn"


def test_effective_analysis_signature_stays_plain_for_baseline() -> None:
    baseline = build_effective_analysis_signature("m1-stable", tempo_backend="baseline")

    assert baseline == "m1-stable"


def test_effective_analysis_signature_changes_for_essentia_tempocnn() -> None:
    signature = build_effective_analysis_signature(
        "m1-stable",
        tempo_backend="tempocnn",
        tempocnn_model="D:/models/deepsquare-k16-3.pb",
        tempocnn_accelerator="auto",
    )

    assert signature == "m1-stable-tempo-tempocnn-deepsquare-k16-3-auto"


def test_parse_backend_list_deduplicates_entries() -> None:
    parsed = parse_backend_list("baseline,tempocnn,tempocnn")

    assert parsed == ["baseline", "tempocnn"]


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
        bpm_tag=None,
        key_tag=None,
    )

    monkeypatch.setattr(
        "cuemate_analysis.tempo_experiments.estimate_tempocnn_bpm",
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
