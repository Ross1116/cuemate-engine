from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import librosa

from cuemate_analysis.analysis import detect_bpm
from cuemate_analysis.config import RuntimeSettings
from cuemate_analysis.persistent_inference_cache import (
    ModelInferenceCacheEntry,
    PersistentInferenceCache,
    resolve_inference_cache_path,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPOCNN_MODEL = REPO_ROOT / "python" / "models" / "essentia" / "deepsquare-k16-3.pb"
DEFAULT_TEMPOCNN_IMAGE = "cuemate-tempocnn:local"
DEFAULT_TEMPOCNN_SERVICE_NAME = "cuemate-tempocnn-service"
DEFAULT_TEMPOCNN_SERVICE_PORT = 47831
TEMPO_BACKEND_BASELINE = "baseline"
TEMPO_BACKEND_TEMPOCNN = "tempocnn"
TEMPO_BACKEND_CHOICES = {TEMPO_BACKEND_BASELINE, TEMPO_BACKEND_TEMPOCNN}
TEMPOCNN_ACCELERATOR_CHOICES = {"auto", "cpu"}
GPU_FALLBACK_MARKERS = (
    "could not select device driver",
    "capabilities: [[gpu]]",
    "nvidia-container-runtime",
    "unknown flag: --gpus",
)
TEMPOCNN_PERSISTED_CACHE_VERSION = "tempocnn-cache-v1"


@dataclass(frozen=True)
class TempoEstimate:
    backend: str
    bpm: float | None
    confidence: float | None
    elapsed_ms: float | None
    details: dict[str, Any]
    notes: list[str]
    available: bool = True

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def normalize_tempo_backend(backend: str) -> str:
    clean = backend.strip().lower()
    if clean in TEMPO_BACKEND_CHOICES:
        return clean
    raise ValueError(f"Unsupported tempo backend: {backend}")


def resolve_tempocnn_model_path(model_path: str | Path | None = None) -> Path:
    raw_value = model_path or os.getenv("CUEMATE_TEMPOCNN_MODEL")
    candidate = Path(raw_value).expanduser() if raw_value else DEFAULT_TEMPOCNN_MODEL
    return candidate.resolve()


def resolve_tempocnn_image_name(image_name: str | None = None) -> str:
    raw_value = image_name or os.getenv("CUEMATE_TEMPOCNN_IMAGE") or DEFAULT_TEMPOCNN_IMAGE
    clean = raw_value.strip()
    if not clean:
        raise ValueError("TempoCNN Docker image name cannot be empty.")
    return clean


def resolve_tempocnn_service_name(service_name: str | None = None) -> str:
    raw_value = service_name or os.getenv("CUEMATE_TEMPOCNN_SERVICE_NAME") or DEFAULT_TEMPOCNN_SERVICE_NAME
    clean = raw_value.strip()
    if not clean:
        raise ValueError("TempoCNN service name cannot be empty.")
    return clean


def resolve_tempocnn_service_port(port: int | str | None = None) -> int:
    raw_value = port or os.getenv("CUEMATE_TEMPOCNN_SERVICE_PORT") or DEFAULT_TEMPOCNN_SERVICE_PORT
    value = int(raw_value)
    if value <= 0:
        raise ValueError("TempoCNN service port must be a positive integer.")
    return value


def normalize_accelerator_choice(value: str | None) -> str:
    choice = (value or "auto").strip().lower()
    if choice not in TEMPOCNN_ACCELERATOR_CHOICES:
        allowed = ", ".join(sorted(TEMPOCNN_ACCELERATOR_CHOICES))
        raise ValueError(f"Unsupported accelerator '{value}'. Expected one of: {allowed}")
    return choice


def estimate_baseline_bpm(path: Path, settings: RuntimeSettings) -> TempoEstimate:
    started = time.perf_counter()
    y, sr = librosa.load(
        path.as_posix(),
        sr=settings.analysis.sample_rate,
        mono=settings.analysis.mono,
    )
    result = detect_bpm(y, sr)
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
    return TempoEstimate(
        backend=TEMPO_BACKEND_BASELINE,
        bpm=float(result["bpm"]),
        confidence=float(result["bpm_confidence"]),
        elapsed_ms=elapsed_ms,
        details={
            "sample_rate": sr,
            "duration_seconds": round(len(y) / sr, 3),
            "display_bpm": round(float(result["bpm"]), 1),
        },
        notes=["Current Milestone 1 librosa-based detector."],
    )


def summarize_stderr(stderr: str, *, max_lines: int = 8) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    trimmed = lines[:max_lines]
    trimmed.append(f"... ({len(lines) - max_lines} more lines omitted)")
    return "\n".join(trimmed)


def windows_path_to_container_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"Expected a Windows drive path, got: {resolved}")
    posix_path = resolved.as_posix()
    tail = posix_path[2:] if len(posix_path) >= 2 and posix_path[1] == ":" else posix_path
    return f"/host/{drive}{tail}"


def docker_volume_spec(path: Path, container_path: str, *, read_only: bool = True) -> str:
    spec = f"{os.fspath(path.resolve())}:{container_path}"
    if read_only:
        spec = f"{spec}:ro"
    return spec


def container_path_for_model(model_path: Path) -> tuple[list[str], str]:
    try:
        relative = model_path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        mount = docker_volume_spec(model_path.parent, "/model", read_only=True)
        return (["-v", mount], f"/model/{model_path.name}")
    return ([], f"/workspace/{relative.as_posix()}")


def build_track_mounts(track_paths: list[Path]) -> tuple[list[str], dict[Path, str]]:
    mount_args: list[str] = []
    container_paths: dict[Path, str] = {}
    directory_mounts: dict[Path, str] = {}

    for track_path in sorted({path.resolve() for path in track_paths}, key=os.fspath):
        parent = track_path.parent
        if parent not in directory_mounts:
            mount_point = f"/input/{len(directory_mounts)}"
            directory_mounts[parent] = mount_point
            mount_args.extend(["-v", docker_volume_spec(parent, mount_point, read_only=True)])
        container_paths[track_path] = f"{directory_mounts[parent]}/{track_path.name}"

    return mount_args, container_paths


def build_tempocnn_docker_command(
    track_path: Path,
    model_path: Path,
    *,
    image_name: str | None = None,
    accelerator: str = "auto",
) -> list[str]:
    resolved_accelerator = normalize_accelerator_choice(accelerator)
    resolved_image_name = resolve_tempocnn_image_name(image_name)
    resolved_track_path = track_path.resolve()
    resolved_model_path = model_path.resolve()
    extra_mounts, container_model_path = container_path_for_model(resolved_model_path)

    command = [
        "docker",
        "run",
        "--rm",
        "-e",
        "TF_CPP_MIN_LOG_LEVEL=3",
        "-v",
        docker_volume_spec(REPO_ROOT, "/workspace", read_only=True),
        "-v",
        docker_volume_spec(resolved_track_path.parent, "/input", read_only=True),
    ]
    if resolved_accelerator == "auto":
        command.extend(["--gpus", "all"])
    else:
        command.extend(["-e", "CUDA_VISIBLE_DEVICES=-1"])
    command.extend(extra_mounts)
    command.extend(
        [
            resolved_image_name,
            "python",
            "/workspace/docker/tempocnn/run_tempocnn.py",
            f"/input/{resolved_track_path.name}",
            container_model_path,
        ]
    )
    return command


def build_tempocnn_batch_docker_command(
    track_paths: list[Path],
    model_path: Path,
    *,
    image_name: str | None = None,
    accelerator: str = "auto",
) -> tuple[list[str], dict[Path, str]]:
    resolved_accelerator = normalize_accelerator_choice(accelerator)
    resolved_image_name = resolve_tempocnn_image_name(image_name)
    resolved_track_paths = [path.resolve() for path in track_paths]
    resolved_model_path = model_path.resolve()
    extra_mounts, container_model_path = container_path_for_model(resolved_model_path)
    track_mount_args, container_paths = build_track_mounts(resolved_track_paths)

    command = [
        "docker",
        "run",
        "--rm",
        "-e",
        "TF_CPP_MIN_LOG_LEVEL=3",
        "-v",
        docker_volume_spec(REPO_ROOT, "/workspace", read_only=True),
    ]
    if resolved_accelerator == "auto":
        command.extend(["--gpus", "all"])
    else:
        command.extend(["-e", "CUDA_VISIBLE_DEVICES=-1"])
    command.extend(track_mount_args)
    command.extend(extra_mounts)
    command.extend(
        [
            resolved_image_name,
            "python",
            "/workspace/docker/tempocnn/run_tempocnn_batch.py",
            container_model_path,
            *[container_paths[path] for path in resolved_track_paths],
        ]
    )
    return command, container_paths


def build_tempocnn_service_run_command(
    drive_letters: list[str],
    *,
    image_name: str | None = None,
    service_name: str | None = None,
    service_port: int | str | None = None,
    accelerator: str = "auto",
) -> list[str]:
    resolved_image_name = resolve_tempocnn_image_name(image_name)
    resolved_service_name = resolve_tempocnn_service_name(service_name)
    resolved_service_port = resolve_tempocnn_service_port(service_port)
    resolved_accelerator = normalize_accelerator_choice(accelerator)

    command = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        resolved_service_name,
        "--publish",
        f"127.0.0.1:{resolved_service_port}:{resolved_service_port}",
        "--env",
        f"TEMPOCNN_SERVICE_PORT={resolved_service_port}",
        "--env",
        "TF_CPP_MIN_LOG_LEVEL=3",
        "--env",
        "CUEMATE_TEMPOCNN_DEFAULT_MODEL=/workspace/python/models/essentia/deepsquare-k16-3.pb",
        "--volume",
        f"{os.fspath(REPO_ROOT.resolve())}:/workspace:ro",
    ]
    if resolved_accelerator == "auto":
        command.extend(["--gpus", "all"])
    else:
        command.extend(["--env", "CUDA_VISIBLE_DEVICES=-1"])
    for drive in sorted({letter.lower() for letter in drive_letters}):
        source = f"{drive.upper()}:\\"
        command.extend(
            [
                "--mount",
                f"type=bind,source={source},target=/host/{drive},readonly",
            ]
        )
    command.extend(
        [
            resolved_image_name,
            "python",
            "/workspace/docker/tempocnn/service.py",
        ]
    )
    return command


def run_docker_command(command: list[str], *, timeout: int = 240) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def inspect_docker_container(name: str) -> dict[str, Any] | None:
    completed = run_docker_command(["docker", "inspect", name], timeout=30)
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
    except Exception:
        return None
    if not payload:
        return None
    return payload[0]


def remove_docker_container(name: str) -> None:
    run_docker_command(["docker", "rm", "-f", name], timeout=30)


def service_container_matches(
    details: dict[str, Any],
    *,
    drive_letters: list[str],
    image_name: str,
) -> bool:
    if not details.get("State", {}).get("Running"):
        return False
    image = str(details.get("Config", {}).get("Image") or "")
    if image != image_name:
        return False
    mounts = details.get("Mounts", [])
    targets = {str(item.get("Destination") or "") for item in mounts}
    required_targets = {"/workspace", *{f"/host/{drive.lower()}" for drive in drive_letters}}
    return required_targets.issubset(targets)


def wait_for_tempocnn_service_health(service_port: int, *, timeout_seconds: float = 45.0) -> tuple[bool, str]:
    health_url = f"http://127.0.0.1:{service_port}/health"
    deadline = time.time() + timeout_seconds
    last_error = "service health check timed out"
    while time.time() < deadline:
        try:
            with urllib_request.urlopen(health_url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") == "ok":
                return True, ""
            last_error = f"unexpected health payload: {payload}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.5)
    return False, last_error


def ensure_tempocnn_service(
    *,
    drive_letters: list[str],
    image_name: str,
    accelerator: str,
    service_name: str,
    service_port: int,
) -> tuple[bool, list[str]]:
    details = inspect_docker_container(service_name)
    if details is not None and service_container_matches(details, drive_letters=drive_letters, image_name=image_name):
        healthy, health_error = wait_for_tempocnn_service_health(service_port)
        if healthy:
            return True, []
        remove_docker_container(service_name)
    if details is not None:
        remove_docker_container(service_name)

    command = build_tempocnn_service_run_command(
        drive_letters,
        image_name=image_name,
        service_name=service_name,
        service_port=service_port,
        accelerator=accelerator,
    )
    completed = run_docker_command(command, timeout=120)
    if completed.returncode != 0:
        notes = [f"TempoCNN service start failed with exit code {completed.returncode}."]
        if completed.stderr.strip():
            notes.append(summarize_stderr(completed.stderr))
        return False, notes

    healthy, health_error = wait_for_tempocnn_service_health(service_port)
    if healthy:
        return True, []
    return False, [f"TempoCNN service did not become healthy: {health_error}"]


def request_tempocnn_service(
    *,
    service_port: int,
    track_paths: list[str],
    model_path: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    request_body = json.dumps({"tracks": track_paths, "model_path": model_path}).encode("utf-8")
    request = urllib_request.Request(
        f"http://127.0.0.1:{service_port}/analyze-bpm",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=max(30, len(track_paths) * 15)) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload, []
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return None, [f"TempoCNN service request failed with HTTP {exc.code}.", body]
    except Exception as exc:
        return None, [f"TempoCNN service request failed: {exc}"]


def should_retry_on_cpu(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(marker in lowered for marker in GPU_FALLBACK_MARKERS)


def parse_docker_json_payload(completed: subprocess.CompletedProcess[str]) -> tuple[dict[str, Any] | None, list[str]]:
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0 or not stdout:
        notes = [f"Docker TempoCNN failed with exit code {completed.returncode}."]
        if stderr:
            notes.append(summarize_stderr(stderr))
            if "no such image" in stderr.lower() or "pull access denied" in stderr.lower():
                notes.append(
                    "Build the local TempoCNN image first with "
                    "powershell -ExecutionPolicy Bypass -File .\\scripts\\build-tempocnn-image.ps1"
                )
        return None, notes

    try:
        return json.loads(stdout.splitlines()[-1]), []
    except Exception as exc:
        notes = [f"Docker TempoCNN returned unparsable output: {exc}"]
        if stdout:
            notes.append(stdout)
        if stderr:
            notes.append(summarize_stderr(stderr))
        return None, notes


def build_tempocnn_unavailable_estimate(
    *,
    elapsed_ms: float,
    notes: list[str],
) -> TempoEstimate:
    return TempoEstimate(
        backend=TEMPO_BACKEND_TEMPOCNN,
        bpm=None,
        confidence=None,
        elapsed_ms=elapsed_ms,
        details={},
        notes=notes,
        available=False,
    )


def build_tempocnn_success_estimate(
    payload: dict[str, Any],
    *,
    elapsed_ms: float,
    model_path: Path,
    image_name: str,
    notes: list[str] | None = None,
    batch_size: int = 1,
) -> TempoEstimate:
    details = {
        "display_bpm": round(float(payload["bpm"]), 1),
        "model_path": str(model_path),
        "model_name": model_path.name,
        "docker_image": image_name,
        "batch_size": batch_size,
        "local_count": int(payload["local_count"]),
        "tempo_spread": payload["tempo_spread"],
        "agreement_with_global": payload["agreement_with_global"],
        "probability_peak": payload["probability_peak"],
        "tf_physical_gpu_count": payload.get("tf_physical_gpu_count"),
        "tf_logical_gpu_count": payload.get("tf_logical_gpu_count"),
    }
    estimate_notes = list(notes or [])
    estimate_notes.insert(0, "Primary TempoCNN estimate from Docker Essentia.")
    estimate_notes.append(f"TempoCNN model: {details['model_name']}")
    estimate_notes.append(f"Docker image: {image_name}")
    if details["tf_logical_gpu_count"] == 0:
        estimate_notes.append("Docker TensorFlow did not register a usable GPU, so TempoCNN likely ran on CPU.")
    return TempoEstimate(
        backend=TEMPO_BACKEND_TEMPOCNN,
        bpm=float(payload["bpm"]),
        confidence=float(payload["confidence"]),
        elapsed_ms=elapsed_ms,
        details=details,
        notes=estimate_notes,
    )


def build_tempocnn_cache_descriptor(
    track_path: Path,
    *,
    model_path: Path,
    accelerator: str,
) -> dict[str, object]:
    stat_result = track_path.stat()
    descriptor_payload = {
        "version": TEMPOCNN_PERSISTED_CACHE_VERSION,
        "track_path": track_path.resolve().as_posix(),
        "file_mtime_ns": int(stat_result.st_mtime_ns),
        "file_size": int(stat_result.st_size),
        "model_path": model_path.resolve().as_posix(),
        "accelerator": accelerator,
    }
    cache_key = hashlib.sha1(json.dumps(descriptor_payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "cache_key": cache_key,
        "file_path": descriptor_payload["track_path"],
        "file_mtime_ns": descriptor_payload["file_mtime_ns"],
        "file_size": descriptor_payload["file_size"],
        "model_signature": f"{TEMPOCNN_PERSISTED_CACHE_VERSION}:{model_path.name}:{accelerator}",
    }


def load_cached_tempocnn_estimates(
    paths: list[Path],
    *,
    model_path: Path,
    accelerator: str,
) -> tuple[dict[Path, TempoEstimate], dict[Path, dict[str, object]]]:
    if not paths:
        return {}, {}
    started = time.perf_counter()
    descriptors = {
        path: build_tempocnn_cache_descriptor(path, model_path=model_path, accelerator=accelerator)
        for path in paths
    }
    try:
        with PersistentInferenceCache(resolve_inference_cache_path()) as cache:
            payloads = cache.fetch_payloads(
                TEMPO_BACKEND_TEMPOCNN,
                [str(item["cache_key"]) for item in descriptors.values()],
            )
    except Exception:
        return {}, descriptors

    elapsed_ms = round((time.perf_counter() - started) * 1000.0 / max(len(paths), 1), 1)
    estimates: dict[Path, TempoEstimate] = {}
    for path, descriptor in descriptors.items():
        payload = payloads.get(str(descriptor["cache_key"]))
        if payload is None:
            continue
        notes = ["Persistent inference cache hit.", *payload.get("notes", [])]
        estimates[path] = TempoEstimate(
            **{
                **payload,
                "elapsed_ms": elapsed_ms,
                "notes": notes,
            }
        )
    return estimates, descriptors


def persist_tempocnn_estimates(
    estimates: dict[Path, TempoEstimate],
    descriptors: dict[Path, dict[str, object]],
) -> None:
    entries: list[ModelInferenceCacheEntry] = []
    for path, estimate in estimates.items():
        if not estimate.available or estimate.bpm is None:
            continue
        descriptor = descriptors.get(path)
        if descriptor is None:
            continue
        entries.append(
            ModelInferenceCacheEntry(
                backend=TEMPO_BACKEND_TEMPOCNN,
                cache_key=str(descriptor["cache_key"]),
                file_path=str(descriptor["file_path"]),
                file_mtime_ns=int(descriptor["file_mtime_ns"]),
                file_size=int(descriptor["file_size"]),
                model_signature=str(descriptor["model_signature"]),
                payload=estimate.to_payload(),
            )
        )
    if not entries:
        return
    try:
        with PersistentInferenceCache(resolve_inference_cache_path()) as cache:
            cache.upsert_entries(entries)
    except Exception:
        return


def purge_tempocnn_cache(file_paths: list[str] | None = None) -> int:
    deleted = 0
    try:
        with PersistentInferenceCache(resolve_inference_cache_path()) as cache:
            deleted = cache.purge(backend=TEMPO_BACKEND_TEMPOCNN, file_paths=file_paths)
    except Exception:
        deleted = 0
    remove_docker_container(resolve_tempocnn_service_name(None))
    return deleted


def estimate_tempocnn_bpms(
    paths: list[Path],
    *,
    model_path: str | Path | None = None,
    accelerator: str | None = None,
    image_name: str | None = None,
) -> dict[Path, TempoEstimate]:
    started = time.perf_counter()
    try:
        resolved_paths = [path.resolve() for path in paths]
        missing = [path for path in resolved_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Audio file was not found: {missing[0]}")
        resolved_model_path = resolve_tempocnn_model_path(model_path)
        if not resolved_model_path.is_file():
            raise FileNotFoundError(
                f"TempoCNN model was not found at {resolved_model_path}. "
                "Set CUEMATE_TEMPOCNN_MODEL or place the default model in python/models/essentia/."
            )
        resolved_image_name = resolve_tempocnn_image_name(image_name)
        resolved_accelerator = normalize_accelerator_choice(
            accelerator or os.getenv("CUEMATE_TEMPOCNN_ACCELERATOR", "auto")
        )
        if not resolved_paths:
            return {}
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        return {
            path.resolve(): build_tempocnn_unavailable_estimate(
                elapsed_ms=elapsed_ms,
                notes=[f"Could not prepare TempoCNN Docker run: {exc}"],
            )
            for path in paths
        }

    cached_estimates, cache_descriptors = load_cached_tempocnn_estimates(
        resolved_paths,
        model_path=resolved_model_path,
        accelerator=resolved_accelerator,
    )
    pending_paths = [path for path in resolved_paths if path not in cached_estimates]
    if not pending_paths:
        return cached_estimates

    command, container_paths = build_tempocnn_batch_docker_command(
        pending_paths,
        resolved_model_path,
        image_name=resolved_image_name,
        accelerator=resolved_accelerator,
    )

    service_notes: list[str] = []
    model_service_mounts, container_model_path = container_path_for_model(resolved_model_path)
    use_service = resolved_accelerator == "auto" and not model_service_mounts
    if use_service:
        try:
            service_started = time.perf_counter()
            container_track_paths = {
                path: windows_path_to_container_path(path)
                for path in pending_paths
            }
            service_name = resolve_tempocnn_service_name(None)
            service_port = resolve_tempocnn_service_port(None)
            drive_letters = [path.drive.rstrip(":").lower() for path in [*pending_paths, resolved_model_path]]
            service_ready, startup_notes = ensure_tempocnn_service(
                drive_letters=drive_letters,
                image_name=resolved_image_name,
                accelerator=resolved_accelerator,
                service_name=service_name,
                service_port=service_port,
            )
            if service_ready:
                service_payload, service_error_notes = request_tempocnn_service(
                    service_port=service_port,
                    track_paths=[container_track_paths[path] for path in pending_paths],
                    model_path=container_model_path,
                )
                if service_payload is not None:
                    service_elapsed_ms = round((time.perf_counter() - service_started) * 1000.0, 1)
                    tf_physical_gpu_count = service_payload.get("tf_physical_gpu_count")
                    tf_logical_gpu_count = service_payload.get("tf_logical_gpu_count")
                    batch_results = {
                        item.get("track_path"): item for item in service_payload.get("results", [])
                    }
                    uncached_results = {
                        path: (
                            build_tempocnn_unavailable_estimate(
                                elapsed_ms=service_elapsed_ms / max(len(pending_paths), 1),
                                notes=["TempoCNN service did not return a result for this track."],
                            )
                            if batch_results.get(container_track_paths[path]) is None
                            else (
                                build_tempocnn_unavailable_estimate(
                                    elapsed_ms=service_elapsed_ms / max(len(pending_paths), 1),
                                    notes=[f"TempoCNN service failed for this track: {batch_results[container_track_paths[path]]['error']}"],
                                )
                                if batch_results[container_track_paths[path]].get("error")
                                else build_tempocnn_success_estimate(
                                    {
                                        **dict(batch_results[container_track_paths[path]]),
                                        "tf_physical_gpu_count": tf_physical_gpu_count,
                                        "tf_logical_gpu_count": tf_logical_gpu_count,
                                    },
                                    elapsed_ms=service_elapsed_ms / max(len(pending_paths), 1),
                                    model_path=resolved_model_path,
                                    image_name=resolved_image_name,
                                    notes=["Warm Docker service path."],
                                    batch_size=len(pending_paths),
                                )
                            )
                        )
                        for path in pending_paths
                    }
                    persist_tempocnn_estimates(uncached_results, cache_descriptors)
                    return {**cached_estimates, **uncached_results}
                service_notes.extend(service_error_notes)
            else:
                service_notes.extend(startup_notes)
        except Exception as exc:
            service_notes.append(f"TempoCNN service path was unavailable: {exc}")

    fallback_note: str | None = None
    try:
        completed = run_docker_command(command, timeout=max(240, len(resolved_paths) * 30))
        if (
            resolved_accelerator == "auto"
            and completed.returncode != 0
            and should_retry_on_cpu(completed.stderr)
        ):
            retry_command, retry_container_paths = build_tempocnn_batch_docker_command(
                pending_paths,
                resolved_model_path,
                image_name=resolved_image_name,
                accelerator="cpu",
            )
            completed = run_docker_command(
                retry_command,
                timeout=max(240, len(pending_paths) * 30),
            )
            container_paths = retry_container_paths
            fallback_note = "Docker GPU runtime was unavailable, so TempoCNN retried in CPU mode."
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        return {
            path: build_tempocnn_unavailable_estimate(
                elapsed_ms=elapsed_ms,
                notes=[f"TempoCNN Docker invocation failed: {exc}"],
            )
            for path in pending_paths
        }

    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
    payload, notes = parse_docker_json_payload(completed)
    if payload is None:
        if fallback_note:
            notes.insert(0, fallback_note)
        return {
            path: build_tempocnn_unavailable_estimate(
                elapsed_ms=elapsed_ms,
                notes=notes,
            )
            for path in pending_paths
        }

    tf_physical_gpu_count = payload.get("tf_physical_gpu_count")
    tf_logical_gpu_count = payload.get("tf_logical_gpu_count")
    batch_results = {item.get("track_path"): item for item in payload.get("results", [])}
    per_track_elapsed_ms = round(elapsed_ms / max(len(pending_paths), 1), 1)
    result_map: dict[Path, TempoEstimate] = {}

    for path in pending_paths:
        container_track_path = container_paths[path]
        item = batch_results.get(container_track_path)
        if item is None:
            result_map[path] = build_tempocnn_unavailable_estimate(
                elapsed_ms=per_track_elapsed_ms,
                notes=["TempoCNN batch run did not return a result for this track."],
            )
            continue
        if item.get("error"):
            result_map[path] = build_tempocnn_unavailable_estimate(
                elapsed_ms=per_track_elapsed_ms,
                notes=[f"TempoCNN batch run failed for this track: {item['error']}"],
            )
            continue

        item_payload = dict(item)
        item_payload["tf_physical_gpu_count"] = tf_physical_gpu_count
        item_payload["tf_logical_gpu_count"] = tf_logical_gpu_count
        item_notes: list[str] = []
        if fallback_note:
            item_notes.append(fallback_note)
        item_notes.extend(service_notes)
        item_notes.append(f"Batch run size: {len(pending_paths)} track(s).")
        result_map[path] = build_tempocnn_success_estimate(
            item_payload,
            elapsed_ms=per_track_elapsed_ms,
            model_path=resolved_model_path,
            image_name=resolved_image_name,
            notes=item_notes,
            batch_size=len(pending_paths),
        )
    persist_tempocnn_estimates(result_map, cache_descriptors)
    return {**cached_estimates, **result_map}


def estimate_tempocnn_bpm(
    path: Path,
    *,
    model_path: str | Path | None = None,
    accelerator: str | None = None,
    image_name: str | None = None,
) -> TempoEstimate:
    return estimate_tempocnn_bpms(
        [path],
        model_path=model_path,
        accelerator=accelerator,
        image_name=image_name,
    )[path.resolve()]
