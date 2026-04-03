from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import librosa

from cuemate_analysis.analysis import detect_bpm
from cuemate_analysis.config import RuntimeSettings


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPOCNN_MODEL = REPO_ROOT / "python" / "models" / "essentia" / "deepsquare-k16-3.pb"
DEFAULT_TEMPOCNN_IMAGE = "cuemate-tempocnn:local"
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


def run_docker_command(command: list[str], *, timeout: int = 240) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


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

    command, container_paths = build_tempocnn_batch_docker_command(
        resolved_paths,
        resolved_model_path,
        image_name=resolved_image_name,
        accelerator=resolved_accelerator,
    )

    fallback_note: str | None = None
    try:
        completed = run_docker_command(command, timeout=max(240, len(resolved_paths) * 30))
        if (
            resolved_accelerator == "auto"
            and completed.returncode != 0
            and should_retry_on_cpu(completed.stderr)
        ):
            retry_command, retry_container_paths = build_tempocnn_batch_docker_command(
                resolved_paths,
                resolved_model_path,
                image_name=resolved_image_name,
                accelerator="cpu",
            )
            completed = run_docker_command(
                retry_command,
                timeout=max(240, len(resolved_paths) * 30),
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
            for path in resolved_paths
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
            for path in resolved_paths
        }

    tf_physical_gpu_count = payload.get("tf_physical_gpu_count")
    tf_logical_gpu_count = payload.get("tf_logical_gpu_count")
    batch_results = {item.get("track_path"): item for item in payload.get("results", [])}
    per_track_elapsed_ms = round(elapsed_ms / max(len(resolved_paths), 1), 1)
    result_map: dict[Path, TempoEstimate] = {}

    for path in resolved_paths:
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
        item_notes.append(f"Batch run size: {len(resolved_paths)} track(s).")
        result_map[path] = build_tempocnn_success_estimate(
            item_payload,
            elapsed_ms=per_track_elapsed_ms,
            model_path=resolved_model_path,
            image_name=resolved_image_name,
            notes=item_notes,
            batch_size=len(resolved_paths),
        )

    return result_map


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
