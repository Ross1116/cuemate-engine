from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import os
from pathlib import Path, PurePath
import shutil
import subprocess
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from cuemate_analysis.persistent_inference_cache import (
    ModelInferenceCacheEntry,
    PersistentInferenceCache,
    resolve_inference_cache_path,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ESSENTIA_SEMANTIC_IMAGE = "cuemate-essentia-semantics:local"
DEFAULT_ESSENTIA_SEMANTIC_SERVICE_NAME = "cuemate-essentia-semantics-service"
DEFAULT_ESSENTIA_SEMANTIC_SERVICE_PORT = 47833
DEFAULT_ESSENTIA_SEMANTIC_MODEL_ROOT = REPO_ROOT / "python" / "models" / "essentia_semantics"
ESSENTIA_SEMANTIC_BACKEND = "essentia_semantics"
ESSENTIA_SEMANTIC_CACHE_VERSION = "essentia-semantics-cache-v2"
ESSENTIA_SEMANTIC_DEVICE_CHOICES = {"auto", "cpu", "cuda"}
ESSENTIA_SEMANTIC_FAMILY_POLICIES = {"best_per_task", "musicnn_only"}
ESSENTIA_SEMANTIC_URLS = {
    "musicnn_embedding_pb": "https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb",
    "musicnn_embedding_json": "https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.json",
    "deam_head_pb": "https://essentia.upf.edu/models/classification-heads/deam/deam-msd-musicnn-2.pb",
    "deam_head_json": "https://essentia.upf.edu/models/classification-heads/deam/deam-msd-musicnn-2.json",
    "danceability_pb": "https://essentia.upf.edu/models/classifiers/danceability/danceability-musicnn-msd-2.pb",
    "danceability_json": "https://essentia.upf.edu/models/classifiers/danceability/danceability-musicnn-msd-2.json",
    "mood_aggressive_pb": "https://essentia.upf.edu/models/classifiers/mood_aggressive/mood_aggressive-musicnn-msd-1.pb",
    "mood_aggressive_json": "https://essentia.upf.edu/models/classifiers/mood_aggressive/mood_aggressive-musicnn-msd-1.json",
    "mood_party_pb": "https://essentia.upf.edu/models/classifiers/mood_party/mood_party-musicnn-msd-1.pb",
    "mood_party_json": "https://essentia.upf.edu/models/classifiers/mood_party/mood_party-musicnn-msd-1.json",
    "mood_relaxed_pb": "https://essentia.upf.edu/models/classifiers/mood_relaxed/mood_relaxed-musicnn-msd-1.pb",
    "mood_relaxed_json": "https://essentia.upf.edu/models/classifiers/mood_relaxed/mood_relaxed-musicnn-msd-1.json",
}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EssentiaSemanticEstimate:
    backend: str
    danceability_abs: float | None
    arousal_abs: float | None
    valence_abs: float | None
    mood_aggressive_abs: float | None
    mood_party_abs: float | None
    mood_relaxed_abs: float | None
    energy_essentia_fused: float | None
    energy_essentia_bucket: str | None
    elapsed_ms: float | None
    details: dict[str, Any]
    notes: list[str]
    available: bool = True

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return float(max(minimum, min(maximum, value)))


def bucket_from_score(score: float) -> str:
    if score < 0.30:
        return "low"
    if score < 0.55:
        return "groove"
    if score < 0.78:
        return "drive"
    return "peak"


def resolve_essentia_semantic_model_root(model_root: str | Path | None = None) -> Path:
    raw_value = model_root or os.getenv("CUEMATE_ESSENTIA_SEMANTIC_MODEL_ROOT")
    candidate = Path(raw_value).expanduser() if raw_value else DEFAULT_ESSENTIA_SEMANTIC_MODEL_ROOT
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def resolve_essentia_semantic_image_name(image_name: str | None = None) -> str:
    raw_value = image_name or os.getenv("CUEMATE_ESSENTIA_SEMANTIC_IMAGE") or DEFAULT_ESSENTIA_SEMANTIC_IMAGE
    clean = raw_value.strip()
    if not clean:
        raise ValueError("Essentia semantic Docker image name cannot be empty.")
    return clean


def resolve_essentia_semantic_service_name(service_name: str | None = None) -> str:
    raw_value = service_name or os.getenv("CUEMATE_ESSENTIA_SEMANTIC_SERVICE_NAME") or DEFAULT_ESSENTIA_SEMANTIC_SERVICE_NAME
    clean = raw_value.strip()
    if not clean:
        raise ValueError("Essentia semantic service name cannot be empty.")
    return clean


def resolve_essentia_semantic_service_port(port: int | str | None = None) -> int:
    raw_value = port or os.getenv("CUEMATE_ESSENTIA_SEMANTIC_SERVICE_PORT") or DEFAULT_ESSENTIA_SEMANTIC_SERVICE_PORT
    value = int(raw_value)
    if value <= 0:
        raise ValueError("Essentia semantic service port must be positive.")
    return value


def normalize_essentia_semantic_device_choice(value: str | None) -> str:
    choice = (value or "auto").strip().lower()
    if choice not in ESSENTIA_SEMANTIC_DEVICE_CHOICES:
        allowed = ", ".join(sorted(ESSENTIA_SEMANTIC_DEVICE_CHOICES))
        raise ValueError(f"Unsupported Essentia semantic device '{value}'. Expected one of: {allowed}")
    return choice


def normalize_essentia_semantic_family_policy(value: str | None) -> str:
    choice = (value or "best_per_task").strip().lower()
    if choice not in ESSENTIA_SEMANTIC_FAMILY_POLICIES:
        allowed = ", ".join(sorted(ESSENTIA_SEMANTIC_FAMILY_POLICIES))
        raise ValueError(f"Unsupported Essentia semantic family policy '{value}'. Expected one of: {allowed}")
    return choice


def windows_path_to_container_path(path: Path | PurePath) -> str:
    resolved = path.resolve() if isinstance(path, Path) else path
    posix_path = resolved.as_posix()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        if resolved.is_absolute():
            return f"/host{posix_path}"
        raise ValueError(f"Expected a Windows drive path or POSIX absolute path, got: {resolved}")
    tail = posix_path[2:] if len(posix_path) >= 2 and posix_path[1] == ":" else posix_path
    return f"/host/{drive}{tail}"


def docker_volume_spec(path: Path, container_path: str, *, read_only: bool = True) -> str:
    spec = f"{os.fspath(path.resolve())}:{container_path}"
    if read_only:
        spec = f"{spec}:ro"
    return spec


def host_gpu_available() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        completed = subprocess.run([nvidia_smi, "-L"], capture_output=True, text=True, timeout=10, check=False)
    except Exception:
        return False
    return completed.returncode == 0 and bool((completed.stdout or "").strip())


def build_essentia_semantic_model_manifest(
    model_root: str | Path | None = None,
    *,
    family_policy: str = "best_per_task",
) -> dict[str, Any]:
    resolved_root = resolve_essentia_semantic_model_root(model_root)
    normalized_policy = normalize_essentia_semantic_family_policy(family_policy)
    tasks = {
        "musicnn_embedding": {
            "family": "musicnn",
            "pb_path": resolved_root / "musicnn" / "msd-musicnn-1.pb",
            "json_path": resolved_root / "musicnn" / "msd-musicnn-1.json",
        },
        "deam_head": {
            "family": "musicnn",
            "pb_path": resolved_root / "musicnn" / "deam-msd-musicnn-2.pb",
            "json_path": resolved_root / "musicnn" / "deam-msd-musicnn-2.json",
        },
        "danceability": {
            "family": "musicnn",
            "pb_path": resolved_root / "musicnn" / "danceability-musicnn-msd-2.pb",
            "json_path": resolved_root / "musicnn" / "danceability-musicnn-msd-2.json",
        },
        "mood_aggressive": {
            "family": "musicnn",
            "pb_path": resolved_root / "musicnn" / "mood_aggressive-musicnn-msd-1.pb",
            "json_path": resolved_root / "musicnn" / "mood_aggressive-musicnn-msd-1.json",
        },
        "mood_party": {
            "family": "musicnn",
            "pb_path": resolved_root / "musicnn" / "mood_party-musicnn-msd-1.pb",
            "json_path": resolved_root / "musicnn" / "mood_party-musicnn-msd-1.json",
        },
        "mood_relaxed": {
            "family": "musicnn",
            "pb_path": resolved_root / "musicnn" / "mood_relaxed-musicnn-msd-1.pb",
            "json_path": resolved_root / "musicnn" / "mood_relaxed-musicnn-msd-1.json",
        },
    }
    return {
        "family_policy": normalized_policy,
        "model_root": resolved_root,
        "tasks": tasks,
    }


def build_essentia_semantic_manifest_signature(manifest: dict[str, Any], *, device: str) -> str:
    task_identity = {}
    for name, info in manifest["tasks"].items():
        pb_path = Path(info["pb_path"])
        json_path = Path(info["json_path"])
        if not pb_path.is_file() or not json_path.is_file():
            raise FileNotFoundError(f"Essentia semantic model artifact missing for {name}: {pb_path} / {json_path}")
        pb_stat = pb_path.stat()
        json_stat = json_path.stat()
        task_identity[name] = {
            "family": info["family"],
            "pb_path": pb_path.resolve().as_posix(),
            "pb_mtime_ns": int(pb_stat.st_mtime_ns),
            "pb_size": int(pb_stat.st_size),
            "json_path": json_path.resolve().as_posix(),
            "json_mtime_ns": int(json_stat.st_mtime_ns),
            "json_size": int(json_stat.st_size),
        }
    payload = {
        "version": ESSENTIA_SEMANTIC_CACHE_VERSION,
        "family_policy": manifest["family_policy"],
        "device": normalize_essentia_semantic_device_choice(device),
        "tasks": task_identity,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def download_essentia_semantic_models(
    *,
    model_root: str | Path | None = None,
    family_policy: str = "best_per_task",
) -> list[Path]:
    manifest = build_essentia_semantic_model_manifest(model_root, family_policy=family_policy)
    resolved_root = Path(manifest["model_root"])
    musicnn_dir = resolved_root / "musicnn"
    musicnn_dir.mkdir(parents=True, exist_ok=True)
    download_map = {
        musicnn_dir / "msd-musicnn-1.pb": ESSENTIA_SEMANTIC_URLS["musicnn_embedding_pb"],
        musicnn_dir / "msd-musicnn-1.json": ESSENTIA_SEMANTIC_URLS["musicnn_embedding_json"],
        musicnn_dir / "deam-msd-musicnn-2.pb": ESSENTIA_SEMANTIC_URLS["deam_head_pb"],
        musicnn_dir / "deam-msd-musicnn-2.json": ESSENTIA_SEMANTIC_URLS["deam_head_json"],
        musicnn_dir / "danceability-musicnn-msd-2.pb": ESSENTIA_SEMANTIC_URLS["danceability_pb"],
        musicnn_dir / "danceability-musicnn-msd-2.json": ESSENTIA_SEMANTIC_URLS["danceability_json"],
        musicnn_dir / "mood_aggressive-musicnn-msd-1.pb": ESSENTIA_SEMANTIC_URLS["mood_aggressive_pb"],
        musicnn_dir / "mood_aggressive-musicnn-msd-1.json": ESSENTIA_SEMANTIC_URLS["mood_aggressive_json"],
        musicnn_dir / "mood_party-musicnn-msd-1.pb": ESSENTIA_SEMANTIC_URLS["mood_party_pb"],
        musicnn_dir / "mood_party-musicnn-msd-1.json": ESSENTIA_SEMANTIC_URLS["mood_party_json"],
        musicnn_dir / "mood_relaxed-musicnn-msd-1.pb": ESSENTIA_SEMANTIC_URLS["mood_relaxed_pb"],
        musicnn_dir / "mood_relaxed-musicnn-msd-1.json": ESSENTIA_SEMANTIC_URLS["mood_relaxed_json"],
    }
    downloaded: list[Path] = []
    for target, url in download_map.items():
        if target.is_file():
            downloaded.append(target)
            continue
        urllib_request.urlretrieve(url, target)
        downloaded.append(target)
    return downloaded


def container_path_for_model_root(model_root: Path) -> tuple[list[str], str]:
    try:
        relative = model_root.resolve().relative_to(REPO_ROOT)
    except ValueError:
        mount = docker_volume_spec(model_root, "/models", read_only=True)
        return (["--volume", mount], "/models")
    return ([], f"/workspace/{relative.as_posix()}")


def build_essentia_semantic_service_run_command(
    drive_letters: list[str],
    *,
    image_name: str | None = None,
    service_name: str | None = None,
    service_port: int | str | None = None,
    model_root: str | Path | None = None,
    device: str = "auto",
    family_policy: str = "best_per_task",
) -> list[str]:
    resolved_image_name = resolve_essentia_semantic_image_name(image_name)
    resolved_service_name = resolve_essentia_semantic_service_name(service_name)
    resolved_service_port = resolve_essentia_semantic_service_port(service_port)
    resolved_device = normalize_essentia_semantic_device_choice(device)
    normalized_family_policy = normalize_essentia_semantic_family_policy(family_policy)
    resolved_model_root = resolve_essentia_semantic_model_root(model_root)
    extra_mounts, container_model_root = container_path_for_model_root(resolved_model_root)
    command = [
        "docker", "run", "-d", "--rm",
        "--name", resolved_service_name,
        "--publish", f"127.0.0.1:{resolved_service_port}:{resolved_service_port}",
        "--env", f"ESSENTIA_SEMANTIC_SERVICE_PORT={resolved_service_port}",
        "--env", f"CUEMATE_ESSENTIA_SEMANTIC_MODEL_ROOT={container_model_root}",
        "--env", f"CUEMATE_ESSENTIA_SEMANTIC_DEVICE={resolved_device}",
        "--env", f"CUEMATE_ESSENTIA_SEMANTIC_MODEL_FAMILY_POLICY={normalized_family_policy}",
        "--env", "PYTHONPATH=/workspace/python/src",
        "--volume", f"{os.fspath(REPO_ROOT.resolve())}:/workspace:ro",
    ]
    if resolved_device == "cuda":
        command.extend(["--gpus", "all"])
    elif resolved_device == "auto":
        if host_gpu_available():
            command.extend(["--gpus", "all"])
        else:
            command.extend(["--env", "CUDA_VISIBLE_DEVICES=-1"])
    else:
        command.extend(["--env", "CUDA_VISIBLE_DEVICES=-1"])
    for drive in sorted({letter.lower() for letter in drive_letters}):
        source = f"{drive.upper()}:\\"
        command.extend(["--mount", f"type=bind,source={source},target=/host/{drive},readonly"])
    command.extend(extra_mounts)
    command.extend([resolved_image_name, "python", "/workspace/docker/essentia_semantics/service.py"])
    return command


def run_docker_command(command: list[str], *, timeout: int = 240) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def inspect_docker_container(name: str) -> dict[str, Any] | None:
    completed = run_docker_command(["docker", "inspect", name], timeout=30)
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
    except Exception:
        return None
    return payload[0] if payload else None


def remove_docker_container(name: str) -> None:
    run_docker_command(["docker", "rm", "-f", name], timeout=30)


def service_container_matches(
    details: dict[str, Any],
    *,
    drive_letters: list[str],
    image_name: str,
    requires_external_model_mount: bool,
    expected_model_source: str | None = None,
) -> bool:
    if not details.get("State", {}).get("Running"):
        return False
    if str(details.get("Config", {}).get("Image") or "") != image_name:
        return False
    mounts = details.get("Mounts", [])
    targets = {str(item.get("Destination") or "") for item in mounts}
    required_targets = {"/workspace", *{f"/host/{drive.lower()}" for drive in drive_letters}}
    if requires_external_model_mount:
        required_targets.add("/models")
        model_mount = next((item for item in mounts if str(item.get("Destination") or "") == "/models"), None)
        if model_mount is None:
            return False
        if expected_model_source and str(model_mount.get("Source") or "") != expected_model_source:
            return False
    return required_targets.issubset(targets)


def wait_for_essentia_semantic_service_health(service_port: int, *, timeout_seconds: float = 45.0) -> tuple[bool, str]:
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


def ensure_essentia_semantic_service(
    *,
    model_root: Path,
    image_name: str,
    device: str,
    family_policy: str,
    service_name: str,
    service_port: int,
    drive_letters: list[str],
) -> tuple[bool, list[str]]:
    extra_mounts, _ = container_path_for_model_root(model_root)
    expected_model_source = os.fspath(model_root.resolve()) if extra_mounts else None
    details = inspect_docker_container(service_name)
    if details is not None and service_container_matches(
        details,
        drive_letters=drive_letters,
        image_name=image_name,
        requires_external_model_mount=bool(extra_mounts),
        expected_model_source=expected_model_source,
    ):
        healthy, health_error = wait_for_essentia_semantic_service_health(service_port)
        if healthy:
            return True, []
        remove_docker_container(service_name)
    if details is not None:
        remove_docker_container(service_name)
    command = build_essentia_semantic_service_run_command(
        drive_letters,
        image_name=image_name,
        service_name=service_name,
        service_port=service_port,
        model_root=model_root,
        device=device,
        family_policy=family_policy,
    )
    completed = run_docker_command(command, timeout=120)
    if completed.returncode != 0:
        notes = [f"Essentia semantic service start failed with exit code {completed.returncode}."]
        if completed.stderr.strip():
            notes.append(completed.stderr.strip())
        return False, notes
    healthy, health_error = wait_for_essentia_semantic_service_health(service_port)
    if healthy:
        return True, []
    try:
        remove_docker_container(service_name)
    except Exception as exc:
        logger.debug("Failed to clean up unhealthy Essentia semantic service container '%s'.", service_name, exc_info=exc)
    return False, [f"Essentia semantic service did not become healthy: {health_error}"]


def request_essentia_semantic_service(
    *,
    service_port: int,
    track_paths: list[str],
    model_root: str,
    device: str,
    family_policy: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    request_body = json.dumps(
        {"tracks": track_paths, "model_root": model_root, "device": device, "family_policy": family_policy}
    ).encode("utf-8")
    request = urllib_request.Request(
        f"http://127.0.0.1:{service_port}/analyze-semantics",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    request_timeout = max(180, len(track_paths) * 75)
    try:
        with urllib_request.urlopen(request, timeout=request_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload, []
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return None, [f"Essentia semantic service request failed with HTTP {exc.code}.", body]
    except Exception as exc:
        return None, [f"Essentia semantic service request failed: {exc}"]


def build_essentia_semantic_unavailable_estimate(
    *,
    elapsed_ms: float | None,
    notes: list[str],
) -> EssentiaSemanticEstimate:
    return EssentiaSemanticEstimate(
        backend=ESSENTIA_SEMANTIC_BACKEND,
        danceability_abs=None,
        arousal_abs=None,
        valence_abs=None,
        mood_aggressive_abs=None,
        mood_party_abs=None,
        mood_relaxed_abs=None,
        energy_essentia_fused=None,
        energy_essentia_bucket=None,
        elapsed_ms=elapsed_ms,
        details={},
        notes=notes,
        available=False,
    )


def build_essentia_semantic_success_estimate(
    payload: dict[str, Any],
    *,
    elapsed_ms: float,
    model_signature: str,
    image_name: str,
    device: str,
    family_policy: str,
    notes: list[str] | None = None,
) -> EssentiaSemanticEstimate:
    payload_notes = [
        "Primary Essentia semantic estimate from warm Docker TensorFlow service.",
        f"Docker image: {image_name}",
        f"Essentia semantic requested device: {device}",
        f"Essentia family policy: {family_policy}",
    ]
    if notes:
        payload_notes = [*notes, *payload_notes]
    semantics = {
        "danceability_abs": payload.get("danceability_abs"),
        "arousal_abs": payload.get("arousal_abs"),
        "valence_abs": payload.get("valence_abs"),
        "mood_aggressive_abs": payload.get("mood_aggressive_abs"),
        "mood_party_abs": payload.get("mood_party_abs"),
        "mood_relaxed_abs": payload.get("mood_relaxed_abs"),
    }
    fused = None
    bucket = None
    if all(semantics.get(key) is not None for key in semantics):
        fused = clamp(
            (0.34 * float(semantics["arousal_abs"]))
            + (0.18 * float(semantics["danceability_abs"]))
            + (0.16 * float(semantics["mood_party_abs"]))
            + (0.14 * (1.0 - float(semantics["mood_relaxed_abs"])))
            + (0.08 * float(semantics["mood_aggressive_abs"]))
            + (0.04 * float(payload.get("loudness_norm") or 0.0))
            + (0.03 * float(payload.get("drums_abs") or 0.0))
            + (0.02 * float(payload.get("groove_abs") or 0.0))
            + (0.01 * float(payload.get("bass_abs") or 0.0))
        )
        bucket = bucket_from_score(fused)
    return EssentiaSemanticEstimate(
        backend=ESSENTIA_SEMANTIC_BACKEND,
        danceability_abs=float(semantics["danceability_abs"]) if semantics["danceability_abs"] is not None else None,
        arousal_abs=float(semantics["arousal_abs"]) if semantics["arousal_abs"] is not None else None,
        valence_abs=float(semantics["valence_abs"]) if semantics["valence_abs"] is not None else None,
        mood_aggressive_abs=float(semantics["mood_aggressive_abs"]) if semantics["mood_aggressive_abs"] is not None else None,
        mood_party_abs=float(semantics["mood_party_abs"]) if semantics["mood_party_abs"] is not None else None,
        mood_relaxed_abs=float(semantics["mood_relaxed_abs"]) if semantics["mood_relaxed_abs"] is not None else None,
        energy_essentia_fused=fused,
        energy_essentia_bucket=bucket,
        elapsed_ms=elapsed_ms,
        details={
            "model_signature": model_signature,
            "semantic_source": payload.get("semantic_source"),
            "runner_device": payload.get("runner_device"),
            "tf_physical_gpu_count": payload.get("tf_physical_gpu_count"),
            "tf_logical_gpu_count": payload.get("tf_logical_gpu_count"),
            "family_map": payload.get("family_map"),
        },
        notes=payload_notes,
        available=True,
    )


def build_essentia_semantic_cache_descriptor(
    track_path: Path,
    *,
    model_signature: str,
    device: str,
    family_policy: str,
) -> dict[str, object]:
    stat_result = track_path.stat()
    descriptor_payload = {
        "version": ESSENTIA_SEMANTIC_CACHE_VERSION,
        "track_path": track_path.resolve().as_posix(),
        "file_mtime_ns": int(stat_result.st_mtime_ns),
        "file_size": int(stat_result.st_size),
        "model_signature": model_signature,
        "device": device,
        "family_policy": family_policy,
    }
    cache_key = hashlib.sha1(json.dumps(descriptor_payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "cache_key": cache_key,
        "file_path": descriptor_payload["track_path"],
        "file_mtime_ns": descriptor_payload["file_mtime_ns"],
        "file_size": descriptor_payload["file_size"],
        "model_signature": f"{ESSENTIA_SEMANTIC_CACHE_VERSION}:{model_signature}:{device}:{family_policy}",
    }


def load_cached_essentia_semantic_estimates(
    paths: list[Path],
    *,
    model_signature: str,
    device: str,
    family_policy: str,
) -> tuple[dict[Path, EssentiaSemanticEstimate], dict[Path, dict[str, object]]]:
    if not paths:
        return {}, {}
    started = time.perf_counter()
    descriptors = {
        path: build_essentia_semantic_cache_descriptor(path, model_signature=model_signature, device=device, family_policy=family_policy)
        for path in paths
    }
    try:
        with PersistentInferenceCache(resolve_inference_cache_path()) as cache:
            payloads = cache.fetch_payloads(
                ESSENTIA_SEMANTIC_BACKEND,
                [str(item["cache_key"]) for item in descriptors.values()],
            )
    except Exception:
        return {}, descriptors
    elapsed_ms = round((time.perf_counter() - started) * 1000.0 / max(len(paths), 1), 1)
    estimates: dict[Path, EssentiaSemanticEstimate] = {}
    for path, descriptor in descriptors.items():
        payload = payloads.get(str(descriptor["cache_key"]))
        if payload is None:
            continue
        notes = ["Persistent inference cache hit.", *payload.get("notes", [])]
        estimates[path] = EssentiaSemanticEstimate(**{**payload, "elapsed_ms": elapsed_ms, "notes": notes})
    return estimates, descriptors


def persist_essentia_semantic_estimates(
    estimates: dict[Path, EssentiaSemanticEstimate],
    descriptors: dict[Path, dict[str, object]],
) -> None:
    entries: list[ModelInferenceCacheEntry] = []
    for path, estimate in estimates.items():
        if not estimate.available:
            continue
        descriptor = descriptors.get(path)
        if descriptor is None:
            continue
        entries.append(
            ModelInferenceCacheEntry(
                backend=ESSENTIA_SEMANTIC_BACKEND,
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
    except Exception as exc:
        logger.debug("Failed to persist Essentia semantic inference cache entries.", exc_info=exc)


def purge_essentia_semantic_cache(file_paths: list[str] | None = None) -> int:
    deleted = 0
    try:
        with PersistentInferenceCache(resolve_inference_cache_path()) as cache:
            deleted = cache.purge(
                backend=ESSENTIA_SEMANTIC_BACKEND,
                file_paths=file_paths,
                purge_all=file_paths is None,
            )
    except Exception:
        deleted = 0
    remove_docker_container(resolve_essentia_semantic_service_name(None))
    return deleted


def estimate_essentia_semantic_batch(
    track_paths: list[Path | str],
    *,
    model_root: str | Path | None = None,
    image_name: str | None = None,
    device: str = "auto",
    family_policy: str = "best_per_task",
    auxiliary_features_by_path: dict[Path, dict[str, float | None]] | None = None,
) -> dict[Path, EssentiaSemanticEstimate]:
    resolved_paths = [Path(path).expanduser().resolve() for path in track_paths]
    if not resolved_paths:
        return {}
    resolved_model_root = resolve_essentia_semantic_model_root(model_root)
    resolved_image_name = resolve_essentia_semantic_image_name(image_name)
    normalized_device = normalize_essentia_semantic_device_choice(device)
    normalized_family_policy = normalize_essentia_semantic_family_policy(family_policy)
    manifest = build_essentia_semantic_model_manifest(resolved_model_root, family_policy=normalized_family_policy)
    try:
        model_signature = build_essentia_semantic_manifest_signature(manifest, device=normalized_device)
    except Exception as exc:
        notes = [f"Essentia semantic model manifest is incomplete: {exc}"]
        return {path: build_essentia_semantic_unavailable_estimate(elapsed_ms=None, notes=notes) for path in resolved_paths}

    cached_estimates, cache_descriptors = load_cached_essentia_semantic_estimates(
        resolved_paths,
        model_signature=model_signature,
        device=normalized_device,
        family_policy=normalized_family_policy,
    )
    pending_paths = [path for path in resolved_paths if path not in cached_estimates]
    if not pending_paths:
        return cached_estimates

    service_name = resolve_essentia_semantic_service_name(None)
    service_port = resolve_essentia_semantic_service_port(None)
    started = time.perf_counter()
    container_track_paths: dict[Path, str] = {}
    unavailable_estimates: dict[Path, EssentiaSemanticEstimate] = {}
    service_candidate_paths: list[Path] = []
    for path in pending_paths:
        try:
            container_track_paths[path] = windows_path_to_container_path(path)
            service_candidate_paths.append(path)
        except Exception as exc:
            unavailable_estimates[path] = build_essentia_semantic_unavailable_estimate(
                elapsed_ms=None,
                notes=[f"Could not map track path into the Essentia semantic container: {exc}"],
            )
    if not service_candidate_paths:
        return {**cached_estimates, **unavailable_estimates}

    drive_letters = sorted({path.drive.rstrip(":").lower() for path in service_candidate_paths if path.drive})
    try:
        service_ready, startup_notes = ensure_essentia_semantic_service(
            model_root=resolved_model_root,
            image_name=resolved_image_name,
            device=normalized_device,
            family_policy=normalized_family_policy,
            service_name=service_name,
            service_port=service_port,
            drive_letters=drive_letters,
        )
    except Exception as exc:
        service_ready = False
        startup_notes = [f"Essentia semantic service startup failed: {exc}"]
    if not service_ready:
        uncached_unavailable = {
            path: build_essentia_semantic_unavailable_estimate(elapsed_ms=None, notes=startup_notes)
            for path in service_candidate_paths
        }
        return {**cached_estimates, **unavailable_estimates, **uncached_unavailable}

    _, container_model_root = container_path_for_model_root(resolved_model_root)
    service_payload, request_notes = request_essentia_semantic_service(
        service_port=service_port,
        track_paths=[container_track_paths[path] for path in service_candidate_paths],
        model_root=container_model_root,
        device=normalized_device,
        family_policy=normalized_family_policy,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000.0 / max(len(service_candidate_paths), 1), 1)
    if service_payload is None:
        notes = [*startup_notes, *request_notes]
        uncached_unavailable = {
            path: build_essentia_semantic_unavailable_estimate(elapsed_ms=elapsed_ms, notes=notes)
            for path in service_candidate_paths
        }
        return {**cached_estimates, **unavailable_estimates, **uncached_unavailable}

    results_by_track = {str(item.get("track_path")): item for item in service_payload.get("results", [])}
    estimates: dict[Path, EssentiaSemanticEstimate] = {}
    for path in service_candidate_paths:
        item = results_by_track.get(container_track_paths[path])
        if item is None:
            estimates[path] = build_essentia_semantic_unavailable_estimate(
                elapsed_ms=elapsed_ms,
                notes=["Essentia semantic service did not return a result for this track."],
            )
            continue
        if item.get("error"):
            estimates[path] = build_essentia_semantic_unavailable_estimate(
                elapsed_ms=elapsed_ms,
                notes=[f"Essentia semantic service failed for this track: {item['error']}"],
            )
            continue
        aux = (auxiliary_features_by_path or {}).get(path.resolve(), {})
        payload = {
            **dict(item),
            "loudness_norm": aux.get("loudness_norm"),
            "drums_abs": aux.get("drums_abs"),
            "groove_abs": aux.get("groove_abs"),
            "bass_abs": aux.get("bass_abs"),
            "tf_physical_gpu_count": service_payload.get("tf_physical_gpu_count"),
            "tf_logical_gpu_count": service_payload.get("tf_logical_gpu_count"),
        }
        estimates[path] = build_essentia_semantic_success_estimate(
            payload,
            elapsed_ms=elapsed_ms,
            model_signature=model_signature,
            image_name=resolved_image_name,
            device=normalized_device,
            family_policy=normalized_family_policy,
            notes=["Warm Docker service path."],
        )
    persist_essentia_semantic_estimates(estimates, cache_descriptors)
    return {**cached_estimates, **unavailable_estimates, **estimates}
