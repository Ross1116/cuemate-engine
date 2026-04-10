from __future__ import annotations

import os
from pathlib import Path

import pytest

from cuemate_analysis.path_safety import (
    resolve_allowed_roots,
    resolve_existing_directory_path,
    resolve_existing_file_path,
)


def test_resolve_existing_file_path_accepts_existing_file(tmp_path: Path) -> None:
    track_path = tmp_path / "track.wav"
    track_path.write_bytes(b"wav")

    resolved = resolve_existing_file_path(track_path, "track_path")

    assert resolved == track_path.resolve()


def test_resolve_existing_file_path_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_existing_file_path(tmp_path / "missing.wav", "track_path")


@pytest.mark.parametrize("raw_path", ["", "   "])
def test_resolve_existing_file_path_rejects_empty_values(raw_path: str) -> None:
    with pytest.raises(ValueError, match="is required"):
        resolve_existing_file_path(raw_path, "track_path")


def test_resolve_existing_file_path_rejects_outside_allowed_roots(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_file = tmp_path / "outside.wav"
    outside_file.write_bytes(b"wav")

    with pytest.raises(ValueError, match="allowed roots"):
        resolve_existing_file_path(
            outside_file,
            "track_path",
            allowed_roots=[allowed_root.resolve()],
        )


def test_resolve_existing_directory_path_accepts_directory(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    model_root.mkdir()

    resolved = resolve_existing_directory_path(model_root, "model_root")

    assert resolved == model_root.resolve()


def test_resolve_existing_directory_path_rejects_outside_allowed_roots(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    with pytest.raises(ValueError, match="allowed roots"):
        resolve_existing_directory_path(
            outside_dir,
            "model_root",
            allowed_roots=[allowed_root.resolve()],
        )


def test_resolve_allowed_roots_splits_env_style_list(tmp_path: Path) -> None:
    root_one = tmp_path / "one"
    root_two = tmp_path / "two"
    root_one.mkdir()
    root_two.mkdir()

    resolved = resolve_allowed_roots(f"{root_one}{os.pathsep}{root_two}")

    assert resolved == [root_one.resolve(), root_two.resolve()]


def test_resolve_allowed_roots_defaults_empty_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "name", "nt")

    resolved = resolve_allowed_roots(None)

    assert resolved == []


def test_resolve_existing_file_path_rejects_symlink_escaping_allowed_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_file = tmp_path / "outside.wav"
    outside_file.write_bytes(b"wav")
    inside_link = allowed_root / "linked.wav"
    try:
        inside_link.symlink_to(outside_file)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(ValueError, match="allowed roots"):
        resolve_existing_file_path(
            inside_link,
            "track_path",
            allowed_roots=[allowed_root.resolve()],
        )
