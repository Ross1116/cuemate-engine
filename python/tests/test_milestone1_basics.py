from pathlib import Path

from cuemate_analysis.analysis import resolve_bpm
from cuemate_analysis.cli import build_effective_analysis_signature
from cuemate_analysis.analysis import parse_key_label
from cuemate_analysis.ingest import discover_audio_files, make_playlist_id, make_track_id


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
    resolved = resolve_bpm(124.0, {"bpm": 124.1, "bpm_confidence": 0.88}, detected_source="beatnet")

    assert resolved["bpm"] == 124.0
    assert resolved["bpm_source"] == "tag+beatnet"


def test_effective_analysis_signature_changes_for_beatnet() -> None:
    baseline = build_effective_analysis_signature("m1-stable", tempo_backend="baseline", beatnet_model=1)
    beatnet = build_effective_analysis_signature("m1-stable", tempo_backend="beatnet", beatnet_model=2)

    assert baseline == "m1-stable"
    assert beatnet == "m1-stable-tempo-beatnet-m2"
