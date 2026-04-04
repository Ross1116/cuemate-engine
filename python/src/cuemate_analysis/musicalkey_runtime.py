from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn


MUSICALKEYCNN_DEVICE_CHOICES = {"auto", "cpu", "cuda"}
MUSICALKEYCNN_SAMPLE_RATE = 44100
MUSICALKEYCNN_N_BINS = 105
MUSICALKEYCNN_HOP_LENGTH = 8820
MUSICALKEYCNN_EXCERPT_SECONDS = 20.0
MUSICALKEYCNN_MAX_EXCERPTS = 2
MUSICALKEYCNN_POLICY_SINGLE_EXCERPT = "single_excerpt"
MUSICALKEYCNN_POLICY_BALANCED = "balanced"
MUSICALKEYCNN_POLICY_FULL_TRACK = "full_track"
MUSICALKEYCNN_POLICY_CHOICES = {
    MUSICALKEYCNN_POLICY_SINGLE_EXCERPT,
    MUSICALKEYCNN_POLICY_BALANCED,
    MUSICALKEYCNN_POLICY_FULL_TRACK,
}
MUSICALKEYCNN_PREPROCESS_WORKERS = max(1, min(4, os.cpu_count() or 1))


@dataclass(frozen=True)
class MusicalKeyClassInfo:
    class_index: int
    key: str
    key_number: int
    key_letter: str
    pitch: str
    mode: str


CLASS_TO_KEY_INFO: dict[int, MusicalKeyClassInfo] = {
    0: MusicalKeyClassInfo(0, "1A", 1, "A", "G#", "minor"),
    1: MusicalKeyClassInfo(1, "2A", 2, "A", "D#", "minor"),
    2: MusicalKeyClassInfo(2, "3A", 3, "A", "A#", "minor"),
    3: MusicalKeyClassInfo(3, "4A", 4, "A", "F", "minor"),
    4: MusicalKeyClassInfo(4, "5A", 5, "A", "C", "minor"),
    5: MusicalKeyClassInfo(5, "6A", 6, "A", "G", "minor"),
    6: MusicalKeyClassInfo(6, "7A", 7, "A", "D", "minor"),
    7: MusicalKeyClassInfo(7, "8A", 8, "A", "A", "minor"),
    8: MusicalKeyClassInfo(8, "9A", 9, "A", "E", "minor"),
    9: MusicalKeyClassInfo(9, "10A", 10, "A", "B", "minor"),
    10: MusicalKeyClassInfo(10, "11A", 11, "A", "F#", "minor"),
    11: MusicalKeyClassInfo(11, "12A", 12, "A", "C#", "minor"),
    12: MusicalKeyClassInfo(12, "1B", 1, "B", "B", "major"),
    13: MusicalKeyClassInfo(13, "2B", 2, "B", "F#", "major"),
    14: MusicalKeyClassInfo(14, "3B", 3, "B", "C#", "major"),
    15: MusicalKeyClassInfo(15, "4B", 4, "B", "G#", "major"),
    16: MusicalKeyClassInfo(16, "5B", 5, "B", "D#", "major"),
    17: MusicalKeyClassInfo(17, "6B", 6, "B", "A#", "major"),
    18: MusicalKeyClassInfo(18, "7B", 7, "B", "F", "major"),
    19: MusicalKeyClassInfo(19, "8B", 8, "B", "C", "major"),
    20: MusicalKeyClassInfo(20, "9B", 9, "B", "G", "major"),
    21: MusicalKeyClassInfo(21, "10B", 10, "B", "D", "major"),
    22: MusicalKeyClassInfo(22, "11B", 11, "B", "A", "major"),
    23: MusicalKeyClassInfo(23, "12B", 12, "B", "E", "major"),
}

MODEL_CACHE: dict[tuple[str, str], tuple[nn.Module, torch.device]] = {}


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int]) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu(self.bn(self.conv(x)))


class KeyNet(nn.Module):
    def __init__(self, num_classes: int = 24, in_channels: int = 1, nf: int = 20, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = BasicConv2d(in_channels, nf, kernel_size=5)
        self.conv2 = BasicConv2d(nf, nf, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(p=dropout)

        self.conv3 = BasicConv2d(nf, 2 * nf, kernel_size=3)
        self.conv4 = BasicConv2d(2 * nf, 2 * nf, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(p=dropout)

        self.conv5 = BasicConv2d(2 * nf, 4 * nf, kernel_size=3)
        self.conv6 = BasicConv2d(4 * nf, 4 * nf, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout2d(p=dropout)

        self.conv7 = BasicConv2d(4 * nf, 8 * nf, kernel_size=3)
        self.dropout4 = nn.Dropout2d(p=dropout)
        self.conv8 = BasicConv2d(8 * nf, 8 * nf, kernel_size=3)
        self.dropout5 = nn.Dropout2d(p=dropout)

        self.conv9 = BasicConv2d(8 * nf, num_classes, kernel_size=1)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.pool1(self.conv2(self.conv1(x))))
        x = self.dropout2(self.pool2(self.conv4(self.conv3(x))))
        x = self.dropout3(self.pool3(self.conv6(self.conv5(x))))
        x = self.dropout4(self.conv7(x))
        x = self.dropout5(self.conv8(x))
        x = self.conv9(x)
        x = self.global_avgpool(x)
        return torch.flatten(x, 1)


def normalize_device_choice(choice: str | None) -> str:
    clean = (choice or "auto").strip().lower()
    if clean not in MUSICALKEYCNN_DEVICE_CHOICES:
        allowed = ", ".join(sorted(MUSICALKEYCNN_DEVICE_CHOICES))
        raise ValueError(f"Unsupported MusicalKeyCNN device '{choice}'. Expected one of: {allowed}")
    return clean


def normalize_policy_choice(choice: str | None) -> str:
    clean = (choice or MUSICALKEYCNN_POLICY_FULL_TRACK).strip().lower()
    if clean not in MUSICALKEYCNN_POLICY_CHOICES:
        allowed = ", ".join(sorted(MUSICALKEYCNN_POLICY_CHOICES))
        raise ValueError(f"Unsupported MusicalKeyCNN policy '{choice}'. Expected one of: {allowed}")
    return clean


def detect_gpu_counts() -> tuple[int, int]:
    physical = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logical = physical
    return physical, logical


def resolve_device(choice: str | None) -> torch.device:
    normalized_choice = normalize_device_choice(choice)
    if normalized_choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("MusicalKeyCNN requested CUDA, but no CUDA device is available.")
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_path: str | Path, device_choice: str | None = None) -> tuple[nn.Module, torch.device]:
    resolved_model_path = Path(model_path).expanduser().resolve()
    resolved_device = resolve_device(device_choice)
    cache_key = (resolved_model_path.as_posix(), str(resolved_device))
    cached = MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    model = KeyNet().to(resolved_device)
    state_dict = torch.load(resolved_model_path, map_location=resolved_device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros((1, 1, MUSICALKEYCNN_N_BINS, 99), dtype=torch.float32, device=resolved_device)
        _ = model(dummy)
    cached = (model, resolved_device)
    MODEL_CACHE[cache_key] = cached
    return cached


def load_audio_excerpt(
    track_path: str | Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int = MUSICALKEYCNN_SAMPLE_RATE,
) -> np.ndarray:
    with sf.SoundFile(Path(track_path)) as handle:
        native_sample_rate = int(handle.samplerate)
        total_frames = int(handle.frames)
        start_frame = max(0, min(total_frames, int(start_seconds * native_sample_rate)))
        frame_count = max(1, int(duration_seconds * native_sample_rate))
        handle.seek(start_frame)
        audio = handle.read(frames=frame_count, dtype="float32", always_2d=True)

    if audio.size == 0:
        raise ValueError(f"No audio samples decoded for {track_path}")

    waveform = np.mean(audio, axis=1, dtype=np.float32)
    if native_sample_rate != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=native_sample_rate, target_sr=sample_rate)
    return waveform.astype(np.float32, copy=False)


def load_audio_full_track(
    track_path: str | Path,
    *,
    sample_rate: int = MUSICALKEYCNN_SAMPLE_RATE,
) -> np.ndarray:
    with sf.SoundFile(Path(track_path)) as handle:
        native_sample_rate = int(handle.samplerate)
        audio = handle.read(dtype="float32", always_2d=True)

    if audio.size == 0:
        raise ValueError(f"No audio samples decoded for {track_path}")

    waveform = np.mean(audio, axis=1, dtype=np.float32)
    if native_sample_rate != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=native_sample_rate, target_sr=sample_rate)
    return waveform.astype(np.float32, copy=False)


def build_cqt_tensor(
    waveform: np.ndarray,
    *,
    sample_rate: int = MUSICALKEYCNN_SAMPLE_RATE,
    n_bins: int = MUSICALKEYCNN_N_BINS,
    hop_length: int = MUSICALKEYCNN_HOP_LENGTH,
) -> torch.Tensor:
    cqt = librosa.cqt(
        waveform,
        sr=sample_rate,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=24,
        fmin=65,
    )
    spec = np.log1p(np.abs(cqt))
    chunk = spec[:, 0:-2]
    return torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)


def select_excerpt_starts(
    track_path: str | Path,
    *,
    excerpt_seconds: float = MUSICALKEYCNN_EXCERPT_SECONDS,
    max_excerpts: int = MUSICALKEYCNN_MAX_EXCERPTS,
) -> list[float]:
    with sf.SoundFile(Path(track_path)) as handle:
        duration_seconds = float(handle.frames) / float(handle.samplerate)

    usable_duration = max(0.0, duration_seconds - excerpt_seconds)
    if usable_duration <= 0.0:
        return [0.0]
    if duration_seconds <= excerpt_seconds * 2.0 or max_excerpts <= 1:
        return [usable_duration / 2.0]

    anchors = [0.35, 0.65] if max_excerpts == 2 else np.linspace(0.2, 0.8, num=max_excerpts).tolist()
    starts: list[float] = []
    for anchor in anchors[:max_excerpts]:
        starts.append(max(0.0, min(usable_duration, duration_seconds * float(anchor) - (excerpt_seconds / 2.0))))
    return starts


def preprocess_audio_excerpts(
    track_path: str | Path,
    *,
    sample_rate: int = MUSICALKEYCNN_SAMPLE_RATE,
    n_bins: int = MUSICALKEYCNN_N_BINS,
    hop_length: int = MUSICALKEYCNN_HOP_LENGTH,
    excerpt_seconds: float = MUSICALKEYCNN_EXCERPT_SECONDS,
    max_excerpts: int = MUSICALKEYCNN_MAX_EXCERPTS,
) -> torch.Tensor:
    starts = select_excerpt_starts(track_path, excerpt_seconds=excerpt_seconds, max_excerpts=max_excerpts)
    excerpt_tensors: list[torch.Tensor] = []
    for start_seconds in starts:
        waveform = load_audio_excerpt(
            track_path,
            start_seconds=start_seconds,
            duration_seconds=excerpt_seconds,
            sample_rate=sample_rate,
        )
        excerpt_tensors.append(
            build_cqt_tensor(
                waveform,
                sample_rate=sample_rate,
                n_bins=n_bins,
                hop_length=hop_length,
            )
        )

    if not excerpt_tensors:
        raise ValueError(f"No excerpts prepared for {track_path}")
    return torch.stack(excerpt_tensors, dim=0)


def preprocess_audio_for_policy(
    track_path: str | Path,
    *,
    policy: str = MUSICALKEYCNN_POLICY_FULL_TRACK,
    sample_rate: int = MUSICALKEYCNN_SAMPLE_RATE,
    n_bins: int = MUSICALKEYCNN_N_BINS,
    hop_length: int = MUSICALKEYCNN_HOP_LENGTH,
    excerpt_seconds: float = MUSICALKEYCNN_EXCERPT_SECONDS,
) -> torch.Tensor:
    normalized_policy = normalize_policy_choice(policy)
    if normalized_policy == MUSICALKEYCNN_POLICY_FULL_TRACK:
        waveform = load_audio_full_track(track_path, sample_rate=sample_rate)
        return build_cqt_tensor(
            waveform,
            sample_rate=sample_rate,
            n_bins=n_bins,
            hop_length=hop_length,
        ).unsqueeze(0)

    max_excerpts = 1 if normalized_policy == MUSICALKEYCNN_POLICY_SINGLE_EXCERPT else MUSICALKEYCNN_MAX_EXCERPTS
    return preprocess_audio_excerpts(
        track_path,
        sample_rate=sample_rate,
        n_bins=n_bins,
        hop_length=hop_length,
        excerpt_seconds=excerpt_seconds,
        max_excerpts=max_excerpts,
    )


def warm_pipeline(model: nn.Module, device: torch.device) -> None:
    waveform = np.zeros(int(MUSICALKEYCNN_SAMPLE_RATE * MUSICALKEYCNN_EXCERPT_SECONDS), dtype=np.float32)
    spec_tensor = build_cqt_tensor(waveform).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(spec_tensor)
        _ = torch.softmax(logits, dim=1)


def build_prediction_payload(
    track_path: str | Path,
    device: torch.device,
    spec_tensor: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    policy: str,
) -> dict[str, Any]:
    top_probabilities, top_indices = torch.topk(probabilities, k=2, dim=1)
    top_index = int(top_indices[0, 0].item())
    second_index = int(top_indices[0, 1].item())
    top_probability = float(top_probabilities[0, 0].item())
    second_probability = float(top_probabilities[0, 1].item())
    top_margin = max(0.0, top_probability - second_probability)
    confidence = max(0.0, min(1.0, top_probability))

    top_info = CLASS_TO_KEY_INFO[top_index]
    second_info = CLASS_TO_KEY_INFO[second_index]
    return {
        "track_path": Path(track_path).resolve().as_posix(),
        "key": top_info.key,
        "key_number": top_info.key_number,
        "key_letter": top_info.key_letter,
        "pitch": top_info.pitch,
        "mode": top_info.mode,
        "confidence": confidence,
        "predicted_class": top_info.class_index,
        "top_probability": top_probability,
        "top_margin": top_margin,
        "runner_device": str(device),
        "sample_rate": MUSICALKEYCNN_SAMPLE_RATE,
        "policy": policy,
        "excerpt_seconds": None if policy == MUSICALKEYCNN_POLICY_FULL_TRACK else MUSICALKEYCNN_EXCERPT_SECONDS,
        "excerpt_count": int(spec_tensor.shape[0]),
        "second_choice": {
            "class_index": second_info.class_index,
            "key": second_info.key,
            "pitch": second_info.pitch,
            "mode": second_info.mode,
            "probability": second_probability,
        },
    }


def preprocess_tracks_for_policy(
    track_paths: list[str | Path],
    *,
    policy: str,
) -> list[tuple[str, torch.Tensor]]:
    normalized_policy = normalize_policy_choice(policy)
    worker_count = max(1, min(MUSICALKEYCNN_PREPROCESS_WORKERS, len(track_paths)))

    def preprocess_one(track_path: str | Path) -> tuple[str, torch.Tensor]:
        return (
            Path(track_path).resolve().as_posix(),
            preprocess_audio_for_policy(track_path, policy=normalized_policy),
        )

    if worker_count == 1:
        return [preprocess_one(track_path) for track_path in track_paths]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(preprocess_one, track_paths))


def predict_keys(
    model: nn.Module,
    device: torch.device,
    track_paths: list[str | Path],
    *,
    policy: str = MUSICALKEYCNN_POLICY_FULL_TRACK,
) -> list[dict[str, Any]]:
    normalized_policy = normalize_policy_choice(policy)
    preprocessed_tracks = preprocess_tracks_for_policy(track_paths, policy=normalized_policy)
    results: list[dict[str, Any]] = []

    with torch.no_grad():
        for resolved_track_path, spec_tensor in preprocessed_tracks:
            logits = model(spec_tensor.to(device))
            probabilities = torch.softmax(logits, dim=1).mean(dim=0, keepdim=True)
            results.append(
                build_prediction_payload(
                    resolved_track_path,
                    device,
                    spec_tensor,
                    probabilities,
                    policy=normalized_policy,
                )
            )
    return results


def predict_key(
    model: nn.Module,
    device: torch.device,
    track_path: str | Path,
    *,
    policy: str = MUSICALKEYCNN_POLICY_FULL_TRACK,
) -> dict[str, Any]:
    return predict_keys(model, device, [track_path], policy=policy)[0]
