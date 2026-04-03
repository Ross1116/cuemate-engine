from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import time
from typing import Any
import warnings

import librosa
import numpy as np

from cuemate_analysis.analysis import clamp, detect_bpm
from cuemate_analysis.config import RuntimeSettings


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
        backend="baseline",
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


def bpm_from_beat_times(beat_times: np.ndarray) -> tuple[float, float, dict[str, Any]]:
    if beat_times.size < 2:
        raise ValueError("BeatNet returned too few beat positions to derive tempo.")

    intervals = np.diff(beat_times)
    intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
    if intervals.size == 0:
        raise ValueError("BeatNet beat intervals were not usable for tempo estimation.")

    median_interval = float(np.median(intervals))
    bpm = 60.0 / median_interval
    mad = float(np.median(np.abs(intervals - median_interval)))
    confidence = clamp(1.0 - (mad / max(median_interval, 1e-6)))
    return bpm, confidence, {
        "beat_count": int(beat_times.size),
        "median_interval_seconds": round(median_interval, 4),
        "interval_mad_seconds": round(mad, 4),
        "display_bpm": round(bpm, 1),
    }


@contextmanager
def patched_madmom_distribution() -> Any:
    original_distribution = metadata.distribution

    def patched(name: str):
        if name == "madmom":
            try:
                return original_distribution("madmom")
            except metadata.PackageNotFoundError:
                return original_distribution("madmom-prebuilt")
        return original_distribution(name)

    metadata.distribution = patched
    try:
        yield
    finally:
        metadata.distribution = original_distribution


def estimate_beatnet_bpm(
    path: Path,
    *,
    model: int = 1,
    inference_model: str = "PF",
    device: str = "cpu",
) -> TempoEstimate:
    started = time.perf_counter()
    try:
        with patched_madmom_distribution():
            from BeatNet.BeatNet import BeatNet
    except Exception as exc:
        return TempoEstimate(
            backend="beatnet",
            bpm=None,
            confidence=None,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
            details={},
            notes=[f"BeatNet unavailable: {exc}"],
            available=False,
        )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`",
                category=FutureWarning,
            )
            estimator = BeatNet(
                model,
                mode="online",
                inference_model=inference_model,
                plot=[],
                thread=False,
                device=device,
            )
            output = estimator.process(path.as_posix())
        beat_times = np.asarray(output)[:, 0]
        bpm, confidence, details = bpm_from_beat_times(beat_times)
        details["model"] = model
        details["inference_model"] = inference_model
        details["first_beats_seconds"] = [round(float(value), 3) for value in beat_times[:8]]
        return TempoEstimate(
            backend="beatnet",
            bpm=bpm,
            confidence=confidence,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
            details=details,
            notes=[
                "Experimental BeatNet-derived BPM from beat intervals.",
                "BeatNet is tempo-focused only; it does not improve key detection.",
            ],
        )
    except Exception as exc:
        return TempoEstimate(
            backend="beatnet",
            bpm=None,
            confidence=None,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
            details={"model": model, "inference_model": inference_model},
            notes=[f"BeatNet failed on this file: {exc}"],
            available=False,
        )


def windows_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"Expected a Windows drive path, got: {resolved}")
    posix_path = resolved.as_posix()
    tail = posix_path[2:] if len(posix_path) >= 2 and posix_path[1] == ":" else posix_path
    return f"/mnt/{drive}{tail}"


def estimate_essentia_wsl_bpm(path: Path) -> TempoEstimate:
    started = time.perf_counter()
    try:
        wsl_path = windows_path_to_wsl(path)
    except Exception as exc:
        return TempoEstimate(
            backend="essentia_wsl",
            bpm=None,
            confidence=None,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
            details={},
            notes=[f"Could not translate path for WSL: {exc}"],
            available=False,
        )

    script = (
        "import essentia.standard as es, json, sys\n"
        "audio = es.MonoLoader(filename=sys.argv[1], sampleRate=22050)()\n"
        "bpm, beats, confidence, _, _ = es.RhythmExtractor2013(method='multifeature')(audio)\n"
        "key, scale, strength = es.KeyExtractor()(audio)\n"
        "print(json.dumps({"
        "'bpm': float(bpm), "
        "'beat_count': len(beats), "
        "'confidence': float(confidence), "
        "'key': key, "
        "'scale': scale, "
        "'key_strength': float(strength)"
        "}))\n"
    )

    try:
        completed = subprocess.run(
            ["wsl.exe", "python3", "-c", script, wsl_path],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except Exception as exc:
        return TempoEstimate(
            backend="essentia_wsl",
            bpm=None,
            confidence=None,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
            details={},
            notes=[f"WSL Essentia invocation failed: {exc}"],
            available=False,
        )

    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0 or not stdout:
        notes = [f"WSL Essentia failed with exit code {completed.returncode}."]
        if stderr:
            notes.append(stderr)
        return TempoEstimate(
            backend="essentia_wsl",
            bpm=None,
            confidence=None,
            elapsed_ms=elapsed_ms,
            details={},
            notes=notes,
            available=False,
        )

    try:
        payload = json.loads(stdout.splitlines()[-1])
    except Exception as exc:
        notes = [f"WSL Essentia returned unparsable output: {exc}"]
        if stdout:
            notes.append(stdout)
        if stderr:
            notes.append(stderr)
        return TempoEstimate(
            backend="essentia_wsl",
            bpm=None,
            confidence=None,
            elapsed_ms=elapsed_ms,
            details={},
            notes=notes,
            available=False,
        )

    details = {
        "beat_count": int(payload["beat_count"]),
        "display_bpm": round(float(payload["bpm"]), 1),
        "key": str(payload["key"]),
        "scale": str(payload["scale"]),
        "key_strength": float(payload["key_strength"]),
    }
    notes = ["Experimental WSL Essentia estimate using RhythmExtractor2013 + KeyExtractor."]
    notes.append("Essentia tempo confidence is a backend-specific score, not a normalized 0-1 probability.")
    if stderr:
        notes.append(stderr)
    return TempoEstimate(
        backend="essentia_wsl",
        bpm=float(payload["bpm"]),
        confidence=float(payload["confidence"]),
        elapsed_ms=elapsed_ms,
        details=details,
        notes=notes,
    )
