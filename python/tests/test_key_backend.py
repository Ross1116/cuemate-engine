from pathlib import Path, PureWindowsPath

from cuemate_analysis.key_backend import (
    DEFAULT_MUSICALKEYCNN_MODEL,
    KEY_BACKEND_MUSICALKEYCNN,
    MUSICALKEYCNN_POLICY_FULL_TRACK,
    build_musicalkeycnn_service_run_command,
    normalize_key_backend,
    normalize_musicalkeycnn_policy_choice,
    resolve_musicalkeycnn_image_name,
    resolve_musicalkeycnn_model_path,
    resolve_musicalkeycnn_service_name,
    resolve_musicalkeycnn_service_port,
    windows_path_to_container_path,
)


def test_normalize_key_backend_accepts_musicalkeycnn() -> None:
    assert normalize_key_backend("musicalkeycnn") == KEY_BACKEND_MUSICALKEYCNN


def test_normalize_musicalkeycnn_policy_defaults_to_full_track() -> None:
    assert normalize_musicalkeycnn_policy_choice(None) == MUSICALKEYCNN_POLICY_FULL_TRACK


def test_resolve_musicalkeycnn_model_path_prefers_explicit_value(tmp_path: Path) -> None:
    model_path = tmp_path / "keynet.pt"
    model_path.write_bytes(b"pt")

    assert resolve_musicalkeycnn_model_path(model_path) == model_path.resolve()


def test_resolve_musicalkeycnn_model_path_uses_repo_default(monkeypatch) -> None:
    monkeypatch.delenv("CUEMATE_MUSICALKEYCNN_MODEL", raising=False)

    assert resolve_musicalkeycnn_model_path() == DEFAULT_MUSICALKEYCNN_MODEL.resolve()


def test_resolve_musicalkeycnn_service_defaults(monkeypatch) -> None:
    monkeypatch.delenv("CUEMATE_MUSICALKEYCNN_IMAGE", raising=False)
    monkeypatch.delenv("CUEMATE_MUSICALKEYCNN_SERVICE_NAME", raising=False)
    monkeypatch.delenv("CUEMATE_MUSICALKEYCNN_SERVICE_PORT", raising=False)

    assert resolve_musicalkeycnn_image_name() == "cuemate-musicalkeycnn:local"
    assert resolve_musicalkeycnn_service_name() == "cuemate-musicalkeycnn-service"
    assert resolve_musicalkeycnn_service_port() == 47832


def test_windows_path_to_container_path_maps_drive_root() -> None:
    path = PureWindowsPath("D:/Personal Projects/Music/example.wav")

    assert windows_path_to_container_path(path) == "/host/d/Personal Projects/Music/example.wav"


def test_build_musicalkeycnn_service_run_command_includes_drive_mounts(monkeypatch) -> None:
    monkeypatch.setattr("cuemate_analysis.key_backend.host_gpu_available", lambda: False)
    command = build_musicalkeycnn_service_run_command(
        ["d", "e"],
        image_name="cuemate-musicalkeycnn:test",
        service_name="cuemate-musicalkeycnn-service-test",
        service_port=49001,
        model_path="D:/models/keynet.pt",
        device="auto",
    )

    command_text = " ".join(command)
    assert "cuemate-musicalkeycnn:test" in command
    assert "--gpus" not in command
    assert "127.0.0.1:49001:49001" in command_text
    assert "target=/host/d" in command_text
    assert "target=/host/e" in command_text
    assert "PYTHONPATH=/workspace/python/src" in command_text
    assert "/model/keynet.pt" in command_text
