import numpy as np
from pathlib import Path

from cuemate_analysis.tempo_experiments import bpm_from_beat_times, windows_path_to_wsl


def test_bpm_from_beat_times_uses_median_interval() -> None:
    beat_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    bpm, confidence, details = bpm_from_beat_times(beat_times)

    assert round(bpm, 2) == 120.0
    assert confidence == 1.0
    assert details["beat_count"] == 5
    assert details["display_bpm"] == 120.0


def test_windows_path_to_wsl_converts_drive_paths() -> None:
    path = Path("D:/Personal Projects/Music/example.wav")

    assert windows_path_to_wsl(path) == "/mnt/d/Personal Projects/Music/example.wav"
