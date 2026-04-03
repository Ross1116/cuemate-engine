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
TEMPO_BACKEND_BASELINE = "baseline"
TEMPO_BACKEND_TEMPOCNN = "tempocnn"
TEMPO_BACKEND_CHOICES = {TEMPO_BACKEND_BASELINE, TEMPO_BACKEND_TEMPOCNN}
LEGACY_TEMPOCNN_ALIASES = {"essentia_wsl_tempocnn"}
TEMPOCNN_ACCELERATOR_CHOICES = {"auto", "cpu"}


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
    if clean in LEGACY_TEMPOCNN_ALIASES:
        return TEMPO_BACKEND_TEMPOCNN
    if clean in TEMPO_BACKEND_CHOICES:
        return clean
    raise ValueError(f"Unsupported tempo backend: {backend}")


def resolve_tempocnn_model_path(model_path: str | Path | None = None) -> Path:
    raw_value = model_path or os.getenv("CUEMATE_TEMPOCNN_MODEL")
    candidate = Path(raw_value).expanduser() if raw_value else DEFAULT_TEMPOCNN_MODEL
    return candidate.resolve()


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


def windows_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"Expected a Windows drive path, got: {resolved}")
    posix_path = resolved.as_posix()
    tail = posix_path[2:] if len(posix_path) >= 2 and posix_path[1] == ":" else posix_path
    return f"/mnt/{drive}{tail}"


def summarize_stderr(stderr: str, *, max_lines: int = 8) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    trimmed = lines[:max_lines]
    trimmed.append(f"... ({len(lines) - max_lines} more lines omitted)")
    return "\n".join(trimmed)


def run_wsl_python(
    script: str,
    *args: str,
    timeout: int = 240,
    accelerator: str = "auto",
) -> subprocess.CompletedProcess[str]:
    resolved_accelerator = normalize_accelerator_choice(accelerator)
    command = ["wsl.exe", "env", "TF_CPP_MIN_LOG_LEVEL=3"]
    if resolved_accelerator == "cpu":
        command.append("CUDA_VISIBLE_DEVICES=-1")
    command.extend(["python3", "-c", script, *args])
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def parse_wsl_json_payload(completed: subprocess.CompletedProcess[str]) -> tuple[dict[str, Any] | None, list[str]]:
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0 or not stdout:
        notes = [f"WSL TempoCNN failed with exit code {completed.returncode}."]
        if stderr:
            notes.append(summarize_stderr(stderr))
        return None, notes

    try:
        return json.loads(stdout.splitlines()[-1]), []
    except Exception as exc:
        notes = [f"WSL TempoCNN returned unparsable output: {exc}"]
        if stdout:
            notes.append(stdout)
        if stderr:
            notes.append(summarize_stderr(stderr))
        return None, notes


def estimate_tempocnn_bpm(
    path: Path,
    *,
    model_path: str | Path | None = None,
    accelerator: str | None = None,
) -> TempoEstimate:
    started = time.perf_counter()
    try:
        wsl_path = windows_path_to_wsl(path)
        resolved_model_path = resolve_tempocnn_model_path(model_path)
        if not resolved_model_path.is_file():
            raise FileNotFoundError(
                f"TempoCNN model was not found at {resolved_model_path}. "
                "Set CUEMATE_TEMPOCNN_MODEL or place the default model in python/models/essentia/."
            )
        wsl_model_path = windows_path_to_wsl(resolved_model_path)
    except Exception as exc:
        return TempoEstimate(
            backend=TEMPO_BACKEND_TEMPOCNN,
            bpm=None,
            confidence=None,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
            details={},
            notes=[f"Could not prepare TempoCNN paths: {exc}"],
            available=False,
        )

    script = (
        "import json, numpy as np, sys\n"
        "import essentia.standard as es\n"
        "try:\n"
        "    import tensorflow as tf\n"
        "    gpu_count = len(tf.config.list_physical_devices('GPU'))\n"
        "except Exception:\n"
        "    gpu_count = None\n"
        "tempo_audio = es.MonoLoader(filename=sys.argv[1], sampleRate=11025, resampleQuality=4)()\n"
        "key_audio = es.MonoLoader(filename=sys.argv[1], sampleRate=22050)()\n"
        "global_tempo, local_tempi, local_probs = es.TempoCNN(graphFilename=sys.argv[2])(tempo_audio)\n"
        "local_tempi = np.asarray(local_tempi, dtype=float)\n"
        "local_probs = np.asarray(local_probs, dtype=float)\n"
        "spread = float(np.median(np.abs(local_tempi - float(global_tempo)))) if local_tempi.size else None\n"
        "agreement = float(np.mean(np.abs(local_tempi - float(global_tempo)) <= 2.0)) if local_tempi.size else 0.0\n"
        "stability = max(0.0, min(1.0, 1.0 - ((spread or 0.0) / max(float(global_tempo) * 0.05, 1.0)))) if local_tempi.size else 0.0\n"
        "confidence = (agreement + stability) / 2.0 if local_tempi.size else 0.0\n"
        "key, scale, strength = es.KeyExtractor()(key_audio)\n"
        "print(json.dumps({"
        "'bpm': float(global_tempo), "
        "'confidence': float(confidence), "
        "'local_count': int(local_tempi.size), "
        "'tempo_spread': spread, "
        "'agreement_with_global': agreement, "
        "'probability_peak': float(np.max(local_probs)) if local_probs.size else None, "
        "'tf_gpu_count': gpu_count, "
        "'key': key, "
        "'scale': scale, "
        "'key_strength': float(strength)"
        "}))\n"
    )

    try:
        completed = run_wsl_python(
            script,
            wsl_path,
            wsl_model_path,
            accelerator=accelerator or os.getenv("CUEMATE_TEMPOCNN_ACCELERATOR", "auto"),
        )
    except Exception as exc:
        return TempoEstimate(
            backend=TEMPO_BACKEND_TEMPOCNN,
            bpm=None,
            confidence=None,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
            details={},
            notes=[f"TempoCNN invocation failed: {exc}"],
            available=False,
        )

    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
    payload, notes = parse_wsl_json_payload(completed)
    if payload is None:
        return TempoEstimate(
            backend=TEMPO_BACKEND_TEMPOCNN,
            bpm=None,
            confidence=None,
            elapsed_ms=elapsed_ms,
            details={},
            notes=notes,
            available=False,
        )

    details = {
        "display_bpm": round(float(payload["bpm"]), 1),
        "model_path": str(resolve_tempocnn_model_path(model_path)),
        "model_name": resolve_tempocnn_model_path(model_path).name,
        "local_count": int(payload["local_count"]),
        "tempo_spread": payload["tempo_spread"],
        "agreement_with_global": payload["agreement_with_global"],
        "probability_peak": payload["probability_peak"],
        "tf_gpu_count": payload.get("tf_gpu_count"),
        "key": str(payload["key"]),
        "scale": str(payload["scale"]),
        "key_strength": float(payload["key_strength"]),
    }
    notes = [
        "Primary TempoCNN estimate from WSL Essentia + KeyExtractor.",
        f"TempoCNN model: {details['model_name']}",
    ]
    if details["tf_gpu_count"] == 0:
        notes.append("WSL TensorFlow did not expose a usable GPU, so TempoCNN fell back to CPU.")
    return TempoEstimate(
        backend=TEMPO_BACKEND_TEMPOCNN,
        bpm=float(payload["bpm"]),
        confidence=float(payload["confidence"]),
        elapsed_ms=elapsed_ms,
        details=details,
        notes=notes,
    )
