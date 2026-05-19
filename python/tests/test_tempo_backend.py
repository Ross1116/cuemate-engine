import os
from pathlib import Path, PureWindowsPath

from cuemate_analysis.tempo_backend import (
    DEFAULT_TEMPOCNN_MODEL,
    TEMPO_BACKEND_TEMPOCNN,
    build_tempocnn_batch_docker_command,
    build_tempocnn_docker_command,
    build_tempocnn_service_run_command,
    normalize_tempo_backend,
    resolve_tempocnn_image_name,
    resolve_tempocnn_service_name,
    resolve_tempocnn_service_port,
    resolve_tempocnn_model_path,
    windows_path_to_container_path,
)


def test_build_tempocnn_docker_command_uses_workspace_model_for_repo_default(tmp_path: Path) -> None:
    track_path = tmp_path / "example.wav"
    track_path.write_bytes(b"wav")

    command = build_tempocnn_docker_command(
        track_path,
        DEFAULT_TEMPOCNN_MODEL,
        image_name="cuemate-tempocnn:test",
        accelerator="cpu",
    )

    assert command[0:3] == ["docker", "run", "--rm"]
    assert "cuemate-tempocnn:test" in command
    assert "/workspace/python/models/essentia/deepsquare-k16-3.pb" in command
    assert "/workspace/docker/tempocnn/run_tempocnn.py" in command
    assert f"/input/{track_path.name}" in command
    assert "--gpus" not in command
    assert "/model/" not in " ".join(command)


def test_normalize_tempo_backend_accepts_tempocnn() -> None:
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


def test_resolve_tempocnn_image_name_prefers_env_var(monkeypatch) -> None:
    monkeypatch.setenv("CUEMATE_TEMPOCNN_IMAGE", "cuemate-tempocnn:gpu")

    assert resolve_tempocnn_image_name() == "cuemate-tempocnn:gpu"


def test_resolve_tempocnn_service_defaults(monkeypatch) -> None:
    monkeypatch.delenv("CUEMATE_TEMPOCNN_SERVICE_NAME", raising=False)
    monkeypatch.delenv("CUEMATE_TEMPOCNN_SERVICE_PORT", raising=False)

    assert resolve_tempocnn_service_name() == "cuemate-essentia-semantics-service"
    assert resolve_tempocnn_service_port() == 47833


def test_build_tempocnn_docker_command_mounts_external_model(tmp_path: Path) -> None:
    track_path = tmp_path / "track.flac"
    model_path = tmp_path / "custom.pb"
    track_path.write_bytes(b"audio")
    model_path.write_bytes(b"pb")

    command = build_tempocnn_docker_command(
        track_path,
        model_path,
        image_name="cuemate-tempocnn:test",
        accelerator="auto",
    )

    assert "--gpus" in command
    assert "/model/custom.pb" in command


def test_build_tempocnn_batch_docker_command_mounts_multiple_track_directories(tmp_path: Path) -> None:
    track_a_dir = tmp_path / "A"
    track_b_dir = tmp_path / "B"
    track_a_dir.mkdir()
    track_b_dir.mkdir()
    track_a = track_a_dir / "one.flac"
    track_b = track_b_dir / "two.flac"
    track_a.write_bytes(b"a")
    track_b.write_bytes(b"b")

    command, container_paths = build_tempocnn_batch_docker_command(
        [track_a, track_b],
        DEFAULT_TEMPOCNN_MODEL,
        image_name="cuemate-tempocnn:test",
        accelerator="auto",
    )

    assert "--gpus" in command
    assert "/workspace/docker/tempocnn/run_tempocnn_batch.py" in command
    assert container_paths[track_a.resolve()].startswith("/input/")
    assert container_paths[track_b.resolve()].startswith("/input/")
    assert container_paths[track_a.resolve()] != container_paths[track_b.resolve()]


def test_windows_path_to_container_path_maps_drive_root() -> None:
    path = PureWindowsPath("D:/Personal Projects/Music/example.wav")

    assert windows_path_to_container_path(path) == "/host/d/Personal Projects/Music/example.wav"


def test_build_tempocnn_service_run_command_includes_drive_mounts() -> None:
    command = build_tempocnn_service_run_command(
        ["d", "e"],
        image_name="cuemate-tempocnn:test",
        service_name="cuemate-tempocnn-service-test",
        service_port=49000,
        accelerator="auto",
    )

    command_text = " ".join(command)
    assert "cuemate-tempocnn:test" in command
    assert "--gpus" in command
    assert "127.0.0.1:49000:49000" in command_text
    assert "ESSENTIA_SEMANTIC_SERVICE_PORT=49000" in command
    assert "target=/host/d" in command_text
    assert "target=/host/e" in command_text
    assert "/workspace/docker/essentia_semantics/service.py" in command


def test_runtime_root_can_come_from_installed_env(monkeypatch, tmp_path: Path) -> None:
    install_root = tmp_path / "CueMate"
    install_root.mkdir()
    monkeypatch.setenv("CUEMATE_REPO_ROOT", os.fspath(install_root))

    from cuemate_analysis import tempo_backend

    assert tempo_backend.resolve_runtime_root() == install_root.resolve()
