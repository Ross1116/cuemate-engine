import os
from pathlib import Path

from cuemate_analysis.tempo_experiments import (
    DEFAULT_TEMPOCNN_MODEL,
    TEMPO_BACKEND_TEMPOCNN,
    normalize_tempo_backend,
    resolve_tempocnn_model_path,
    windows_path_to_wsl,
)


def test_windows_path_to_wsl_converts_drive_paths() -> None:
    path = Path("D:/Personal Projects/Music/example.wav")

    assert windows_path_to_wsl(path) == "/mnt/d/Personal Projects/Music/example.wav"


def test_normalize_tempo_backend_maps_legacy_tempocnn_alias() -> None:
    assert normalize_tempo_backend("essentia_wsl_tempocnn") == TEMPO_BACKEND_TEMPOCNN
    assert normalize_tempo_backend("tempocnn") == TEMPO_BACKEND_TEMPOCNN


def test_resolve_tempocnn_model_path_prefers_explicit_value(tmp_path: Path) -> None:
    model_path = tmp_path / "custom.pb"
    model_path.write_bytes(b"pb")

    assert resolve_tempocnn_model_path(model_path) == model_path.resolve()


def test_resolve_tempocnn_model_path_uses_env_var(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "env-model.pb"
    model_path.write_bytes(b"pb")
    monkeypatch.setenv("CUEMATE_TEMPOCNN_MODEL", os.fspath(model_path))

    assert resolve_tempocnn_model_path() == model_path.resolve()


def test_resolve_tempocnn_model_path_uses_repo_default_when_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("CUEMATE_TEMPOCNN_MODEL", raising=False)

    assert resolve_tempocnn_model_path() == DEFAULT_TEMPOCNN_MODEL.resolve()
