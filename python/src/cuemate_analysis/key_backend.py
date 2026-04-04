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

from cuemate_analysis.analysis import detect_key
from cuemate_analysis.persistent_inference_cache import (
    ModelInferenceCacheEntry,
    PersistentInferenceCache,
    resolve_inference_cache_path,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MUSICALKEYCNN_MODEL = REPO_ROOT / "python" / "models" / "musicalkeycnn" / "keynet.pt"
DEFAULT_MUSICALKEYCNN_IMAGE = "cuemate-musicalkeycnn:local"
DEFAULT_MUSICALKEYCNN_SERVICE_NAME = "cuemate-musicalkeycnn-service"
DEFAULT_MUSICALKEYCNN_SERVICE_PORT = 47832
KEY_BACKEND_CHROMA = "chroma"
KEY_BACKEND_MUSICALKEYCNN = "musicalkeycnn"
KEY_BACKEND_CHOICES = {KEY_BACKEND_CHROMA, KEY_BACKEND_MUSICALKEYCNN}
MUSICALKEYCNN_DEVICE_CHOICES = {"auto", "cpu", "cuda"}
MUSICALKEYCNN_POLICY_SINGLE_EXCERPT = "single_excerpt"
MUSICALKEYCNN_POLICY_BALANCED = "balanced"
MUSICALKEYCNN_POLICY_FULL_TRACK = "full_track"
MUSICALKEYCNN_POLICY_CHOICES = {
    MUSICALKEYCNN_POLICY_SINGLE_EXCERPT,
    MUSICALKEYCNN_POLICY_BALANCED,
    MUSICALKEYCNN_POLICY_FULL_TRACK,
}
MUSICALKEYCNN_PERSISTED_CACHE_VERSION = "musicalkeycnn-cache-v1"


@dataclass(frozen=True)
class KeyEstimate:
    backend: str
    key: str | None
    key_number: int | None
    key_letter: str | None
    confidence: float | None
    elapsed_ms: float | None
    details: dict[str, Any]
    notes: list[str]
    available: bool = True

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def normalize_key_backend(backend: str) -> str:
    clean = backend.strip().lower()
    if clean in KEY_BACKEND_CHOICES:
        return clean
    raise ValueError(f"Unsupported key backend: {backend}")


def resolve_musicalkeycnn_model_path(model_path: str | Path | None = None) -> Path:
    raw_value = model_path or os.getenv("CUEMATE_MUSICALKEYCNN_MODEL")
    candidate = Path(raw_value).expanduser() if raw_value else DEFAULT_MUSICALKEYCNN_MODEL
    return candidate.resolve()


def resolve_musicalkeycnn_image_name(image_name: str | None = None) -> str:
    raw_value = image_name or os.getenv("CUEMATE_MUSICALKEYCNN_IMAGE") or DEFAULT_MUSICALKEYCNN_IMAGE
    clean = raw_value.strip()
    if not clean:
        raise ValueError("MusicalKeyCNN Docker image name cannot be empty.")
    return clean


def resolve_musicalkeycnn_service_name(service_name: str | None = None) -> str:
    raw_value = service_name or os.getenv("CUEMATE_MUSICALKEYCNN_SERVICE_NAME") or DEFAULT_MUSICALKEYCNN_SERVICE_NAME
    clean = raw_value.strip()
    if not clean:
        raise ValueError("MusicalKeyCNN service name cannot be empty.")
    return clean


def resolve_musicalkeycnn_service_port(port: int | str | None = None) -> int:
    raw_value = port or os.getenv("CUEMATE_MUSICALKEYCNN_SERVICE_PORT") or DEFAULT_MUSICALKEYCNN_SERVICE_PORT
    value = int(raw_value)
    if value <= 0:
        raise ValueError("MusicalKeyCNN service port must be positive.")
    return value


def normalize_musicalkeycnn_device_choice(value: str | None) -> str:
    choice = (value or "auto").strip().lower()
    if choice not in MUSICALKEYCNN_DEVICE_CHOICES:
        allowed = ", ".join(sorted(MUSICALKEYCNN_DEVICE_CHOICES))
        raise ValueError(f"Unsupported MusicalKeyCNN device '{value}'. Expected one of: {allowed}")
    return choice


def normalize_musicalkeycnn_policy_choice(value: str | None) -> str:
    choice = (value or MUSICALKEYCNN_POLICY_FULL_TRACK).strip().lower()
    if choice not in MUSICALKEYCNN_POLICY_CHOICES:
        allowed = ", ".join(sorted(MUSICALKEYCNN_POLICY_CHOICES))
        raise ValueError(f"Unsupported MusicalKeyCNN policy '{value}'. Expected one of: {allowed}")
    return choice


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
        return (["--volume", mount], f"/model/{model_path.name}")
    return ([], f"/workspace/{relative.as_posix()}")


def build_musicalkeycnn_service_run_command(
    drive_letters: list[str],
    *,
    image_name: str | None = None,
    service_name: str | None = None,
    service_port: int | str | None = None,
    model_path: str | Path | None = None,
    device: str = "auto",
) -> list[str]:
    resolved_image_name = resolve_musicalkeycnn_image_name(image_name)
    resolved_service_name = resolve_musicalkeycnn_service_name(service_name)
    resolved_service_port = resolve_musicalkeycnn_service_port(service_port)
    resolved_device = normalize_musicalkeycnn_device_choice(device)
    resolved_model_path = resolve_musicalkeycnn_model_path(model_path)
    extra_mounts, container_model_path = container_path_for_model(resolved_model_path)

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
        f"MUSICALKEYCNN_SERVICE_PORT={resolved_service_port}",
        "--env",
        f"CUEMATE_MUSICALKEYCNN_DEFAULT_MODEL={container_model_path}",
        "--env",
        f"CUEMATE_MUSICALKEYCNN_DEFAULT_DEVICE={resolved_device}",
        "--env",
        "PYTHONPATH=/workspace/python/src",
        "--volume",
        f"{os.fspath(REPO_ROOT.resolve())}:/workspace:ro",
    ]
    if resolved_device in {"auto", "cuda"}:
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
    command.extend(extra_mounts)
    command.extend(
        [
            resolved_image_name,
            "python",
            "-m",
            "cuemate_analysis.musicalkey_service",
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
    requires_external_model_mount: bool,
) -> bool:
    if not details.get("State", {}).get("Running"):
        return False
    image = str(details.get("Config", {}).get("Image") or "")
    if image != image_name:
        return False
    mounts = details.get("Mounts", [])
    targets = {str(item.get("Destination") or "") for item in mounts}
    required_targets = {"/workspace", *{f"/host/{drive.lower()}" for drive in drive_letters}}
    if requires_external_model_mount:
        required_targets.add("/model")
    return required_targets.issubset(targets)


def wait_for_musicalkeycnn_service_health(service_port: int, *, timeout_seconds: float = 45.0) -> tuple[bool, str]:
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


def ensure_musicalkeycnn_service(
    *,
    model_path: Path,
    image_name: str,
    device: str,
    service_name: str,
    service_port: int,
    drive_letters: list[str],
) -> tuple[bool, list[str]]:
    extra_model_mounts, _ = container_path_for_model(model_path)
    details = inspect_docker_container(service_name)
    if details is not None and service_container_matches(
        details,
        drive_letters=drive_letters,
        image_name=image_name,
        requires_external_model_mount=bool(extra_model_mounts),
    ):
        healthy, health_error = wait_for_musicalkeycnn_service_health(service_port)
        if healthy:
            return True, []
        remove_docker_container(service_name)
    if details is not None:
        remove_docker_container(service_name)

    command = build_musicalkeycnn_service_run_command(
        drive_letters,
        image_name=image_name,
        service_name=service_name,
        service_port=service_port,
        model_path=model_path,
        device=device,
    )
    completed = run_docker_command(command, timeout=120)
    if completed.returncode != 0:
        notes = [f"MusicalKeyCNN service start failed with exit code {completed.returncode}."]
        if completed.stderr.strip():
            notes.append(completed.stderr.strip())
        return False, notes

    healthy, health_error = wait_for_musicalkeycnn_service_health(service_port)
    if healthy:
        return True, []
    return False, [f"MusicalKeyCNN service did not become healthy: {health_error}"]


def request_musicalkeycnn_service(
    *,
    service_port: int,
    track_paths: list[str],
    model_path: str,
    device: str,
    policy: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    request_body = json.dumps(
        {"tracks": track_paths, "model_path": model_path, "device": device, "policy": policy}
    ).encode("utf-8")
    request = urllib_request.Request(
        f"http://127.0.0.1:{service_port}/analyze-key",
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
        return None, [f"MusicalKeyCNN service request failed with HTTP {exc.code}.", body]
    except Exception as exc:
        return None, [f"MusicalKeyCNN service request failed: {exc}"]


def build_musicalkeycnn_unavailable_estimate(
    *,
    elapsed_ms: float | None,
    notes: list[str],
) -> KeyEstimate:
    return KeyEstimate(
        backend=KEY_BACKEND_MUSICALKEYCNN,
        key=None,
        key_number=None,
        key_letter=None,
        confidence=None,
        elapsed_ms=elapsed_ms,
        details={},
        notes=notes,
        available=False,
    )


def build_musicalkeycnn_success_estimate(
    payload: dict[str, Any],
    *,
    elapsed_ms: float,
    model_path: Path,
    image_name: str,
    device: str,
    policy: str,
    notes: list[str] | None = None,
    batch_size: int = 1,
) -> KeyEstimate:
    payload_notes = [
        "Primary MusicalKeyCNN estimate from warm Docker PyTorch service.",
        f"MusicalKeyCNN model: {model_path.name}",
        f"Docker image: {image_name}",
        f"MusicalKeyCNN requested device: {device}",
        f"MusicalKeyCNN policy: {policy}",
    ]
    if notes:
        payload_notes = [*notes, *payload_notes]
    return KeyEstimate(
        backend=KEY_BACKEND_MUSICALKEYCNN,
        key=str(payload["key"]),
        key_number=int(payload["key_number"]),
        key_letter=str(payload["key_letter"]),
        confidence=float(payload["confidence"]),
        elapsed_ms=elapsed_ms,
        details={
            "pitch": payload.get("pitch"),
            "mode": payload.get("mode"),
            "predicted_class": payload.get("predicted_class"),
            "top_probability": payload.get("top_probability"),
            "top_margin": payload.get("top_margin"),
            "runner_device": payload.get("runner_device"),
            "second_choice": payload.get("second_choice"),
            "torch_physical_gpu_count": payload.get("torch_physical_gpu_count"),
            "torch_logical_gpu_count": payload.get("torch_logical_gpu_count"),
            "display_key": payload.get("key"),
            "batch_size": batch_size,
            "policy": payload.get("policy", policy),
            "excerpt_seconds": payload.get("excerpt_seconds"),
            "excerpt_count": payload.get("excerpt_count"),
        },
        notes=payload_notes,
        available=True,
    )


def build_musicalkeycnn_cache_descriptor(
    track_path: Path,
    *,
    model_path: Path,
    device: str,
    policy: str,
) -> dict[str, object]:
    stat_result = track_path.stat()
    descriptor_payload = {
        "version": MUSICALKEYCNN_PERSISTED_CACHE_VERSION,
        "track_path": track_path.resolve().as_posix(),
        "file_mtime_ns": int(stat_result.st_mtime_ns),
        "file_size": int(stat_result.st_size),
        "model_path": model_path.resolve().as_posix(),
        "device": device,
        "policy": policy,
    }
    cache_key = hashlib.sha1(json.dumps(descriptor_payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "cache_key": cache_key,
        "file_path": descriptor_payload["track_path"],
        "file_mtime_ns": descriptor_payload["file_mtime_ns"],
        "file_size": descriptor_payload["file_size"],
        "model_signature": f"{MUSICALKEYCNN_PERSISTED_CACHE_VERSION}:{model_path.name}:{device}:{policy}",
    }


def load_cached_musicalkeycnn_estimates(
    paths: list[Path],
    *,
    model_path: Path,
    device: str,
    policy: str,
) -> tuple[dict[Path, KeyEstimate], dict[Path, dict[str, object]]]:
    if not paths:
        return {}, {}
    started = time.perf_counter()
    descriptors = {
        path: build_musicalkeycnn_cache_descriptor(
            path,
            model_path=model_path,
            device=device,
            policy=policy,
        )
        for path in paths
    }
    try:
        with PersistentInferenceCache(resolve_inference_cache_path()) as cache:
            payloads = cache.fetch_payloads(
                KEY_BACKEND_MUSICALKEYCNN,
                [str(item["cache_key"]) for item in descriptors.values()],
            )
    except Exception:
        return {}, descriptors

    elapsed_ms = round((time.perf_counter() - started) * 1000.0 / max(len(paths), 1), 1)
    estimates: dict[Path, KeyEstimate] = {}
    for path, descriptor in descriptors.items():
        payload = payloads.get(str(descriptor["cache_key"]))
        if payload is None:
            continue
        notes = ["Persistent inference cache hit.", *payload.get("notes", [])]
        estimates[path] = KeyEstimate(
            **{
                **payload,
                "elapsed_ms": elapsed_ms,
                "notes": notes,
            }
        )
    return estimates, descriptors


def persist_musicalkeycnn_estimates(
    estimates: dict[Path, KeyEstimate],
    descriptors: dict[Path, dict[str, object]],
) -> None:
    entries: list[ModelInferenceCacheEntry] = []
    for path, estimate in estimates.items():
        if not estimate.available or not estimate.key:
            continue
        descriptor = descriptors.get(path)
        if descriptor is None:
            continue
        entries.append(
            ModelInferenceCacheEntry(
                backend=KEY_BACKEND_MUSICALKEYCNN,
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


def purge_musicalkeycnn_cache(file_paths: list[str] | None = None) -> int:
    deleted = 0
    try:
        with PersistentInferenceCache(resolve_inference_cache_path()) as cache:
            deleted = cache.purge(backend=KEY_BACKEND_MUSICALKEYCNN, file_paths=file_paths)
    except Exception:
        deleted = 0
    remove_docker_container(resolve_musicalkeycnn_service_name(None))
    return deleted


def estimate_musicalkeycnn_keys(
    track_paths: list[Path | str],
    *,
    model_path: str | Path | None = None,
    image_name: str | None = None,
    device: str = "auto",
    policy: str = MUSICALKEYCNN_POLICY_FULL_TRACK,
) -> dict[Path, KeyEstimate]:
    resolved_paths = [Path(path).expanduser().resolve() for path in track_paths]
    if not resolved_paths:
        return {}

    resolved_model_path = resolve_musicalkeycnn_model_path(model_path)
    resolved_image_name = resolve_musicalkeycnn_image_name(image_name)
    normalized_device = normalize_musicalkeycnn_device_choice(device)
    normalized_policy = normalize_musicalkeycnn_policy_choice(policy)
    cached_estimates, cache_descriptors = load_cached_musicalkeycnn_estimates(
        resolved_paths,
        model_path=resolved_model_path,
        device=normalized_device,
        policy=normalized_policy,
    )
    pending_paths = [path for path in resolved_paths if path not in cached_estimates]
    if not pending_paths:
        return cached_estimates

    service_name = resolve_musicalkeycnn_service_name(None)
    service_port = resolve_musicalkeycnn_service_port(None)

    if not resolved_model_path.is_file():
        notes = [
            f"MusicalKeyCNN model checkpoint was not found: {resolved_model_path}",
            "Download or copy keynet.pt into python/models/musicalkeycnn/keynet.pt.",
        ]
        return {
            path: build_musicalkeycnn_unavailable_estimate(elapsed_ms=None, notes=notes)
            for path in pending_paths
        }

    started = time.perf_counter()
    container_track_paths = {path: windows_path_to_container_path(path) for path in pending_paths}
    service_ready, startup_notes = ensure_musicalkeycnn_service(
        model_path=resolved_model_path,
        image_name=resolved_image_name,
        device=normalized_device,
        service_name=service_name,
        service_port=service_port,
        drive_letters=[path.drive.rstrip(":").lower() for path in [*pending_paths, resolved_model_path]],
    )
    if not service_ready:
        uncached_unavailable = {
            path: build_musicalkeycnn_unavailable_estimate(elapsed_ms=None, notes=startup_notes)
            for path in pending_paths
        }
        return {**cached_estimates, **uncached_unavailable}

    service_payload, request_notes = request_musicalkeycnn_service(
        service_port=service_port,
        track_paths=[container_track_paths[path] for path in pending_paths],
        model_path=(
            container_path_for_model(resolved_model_path)[1]
        ),
        device=normalized_device,
        policy=normalized_policy,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000.0 / max(len(pending_paths), 1), 1)
    if service_payload is None:
        notes = [*startup_notes, *request_notes]
        uncached_unavailable = {
            path: build_musicalkeycnn_unavailable_estimate(elapsed_ms=elapsed_ms, notes=notes)
            for path in pending_paths
        }
        return {**cached_estimates, **uncached_unavailable}

    results_by_track = {str(item.get("track_path")): item for item in service_payload.get("results", [])}
    estimates: dict[Path, KeyEstimate] = {}
    for path in pending_paths:
        item = results_by_track.get(container_track_paths[path])
        if item is None:
            estimates[path] = build_musicalkeycnn_unavailable_estimate(
                elapsed_ms=elapsed_ms,
                notes=["MusicalKeyCNN service did not return a result for this track."],
            )
            continue
        if item.get("error"):
            estimates[path] = build_musicalkeycnn_unavailable_estimate(
                elapsed_ms=elapsed_ms,
                notes=[f"MusicalKeyCNN failed for this track: {item['error']}"],
            )
            continue

        estimates[path] = build_musicalkeycnn_success_estimate(
            {
                **dict(item),
                "torch_physical_gpu_count": service_payload.get("torch_physical_gpu_count"),
                "torch_logical_gpu_count": service_payload.get("torch_logical_gpu_count"),
            },
            elapsed_ms=elapsed_ms,
            model_path=resolved_model_path,
            image_name=resolved_image_name,
            device=normalized_device,
            policy=normalized_policy,
            notes=["Warm Docker service path."],
            batch_size=len(pending_paths),
        )
    persist_musicalkeycnn_estimates(estimates, cache_descriptors)
    return {**cached_estimates, **estimates}


def estimate_musicalkeycnn_key(
    track_path: Path | str,
    *,
    model_path: str | Path | None = None,
    image_name: str | None = None,
    device: str = "auto",
    policy: str = MUSICALKEYCNN_POLICY_FULL_TRACK,
) -> KeyEstimate:
    return estimate_musicalkeycnn_keys(
        [track_path],
        model_path=model_path,
        image_name=image_name,
        device=device,
        policy=policy,
    )[Path(track_path).expanduser().resolve()]


def estimate_chroma_key(
    track_path: Path | str,
    *,
    sample_rate: int = 22050,
    mono: bool = True,
) -> KeyEstimate:
    resolved_path = Path(track_path).expanduser().resolve()
    started = time.perf_counter()
    try:
        y, sr = librosa.load(resolved_path.as_posix(), sr=sample_rate, mono=mono)
        if y.size == 0:
            raise ValueError(f"No audio samples decoded for {resolved_path}")
        detected = detect_key(y, sr)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        return KeyEstimate(
            backend=KEY_BACKEND_CHROMA,
            key=str(detected["key"]),
            key_number=int(detected["key_number"]),
            key_letter=str(detected["key_letter"]),
            confidence=float(detected["key_confidence"]),
            elapsed_ms=elapsed_ms,
            details={
                "pitch": detected.get("pitch"),
                "mode": detected.get("mode"),
                "display_key": detected.get("key"),
            },
            notes=["Deterministic chroma fallback on the host CPU."],
            available=True,
        )
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        return KeyEstimate(
            backend=KEY_BACKEND_CHROMA,
            key=None,
            key_number=None,
            key_letter=None,
            confidence=None,
            elapsed_ms=elapsed_ms,
            details={},
            notes=[f"Chroma key estimation failed: {exc}"],
            available=False,
        )
