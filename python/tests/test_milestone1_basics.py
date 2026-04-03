from pathlib import Path

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
