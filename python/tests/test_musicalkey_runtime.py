from pathlib import Path

import torch
import torch.nn as nn

from cuemate_analysis.musicalkey_runtime import predict_keys


class FakeKeyModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((x.shape[0], 24), dtype=torch.float32, device=x.device)
        logits[:, 7] = 5.0
        logits[:, 19] = 1.0
        return logits


def test_predict_keys_returns_ordered_results_with_excerpt_counts(monkeypatch) -> None:
    track_one = Path("D:/music/one.flac")
    track_two = Path("D:/music/two.flac")

    monkeypatch.setattr(
        "cuemate_analysis.musicalkey_runtime.preprocess_tracks_for_policy",
        lambda track_paths, policy: [
            (Path(track_paths[0]).resolve().as_posix(), torch.zeros((2, 1, 105, 12), dtype=torch.float32)),
            (Path(track_paths[1]).resolve().as_posix(), torch.zeros((1, 1, 105, 8), dtype=torch.float32)),
        ],
    )

    results = predict_keys(
        FakeKeyModel(),
        torch.device("cpu"),
        [track_one, track_two],
        policy="full_track",
    )

    assert [item["track_path"] for item in results] == [
        track_one.resolve().as_posix(),
        track_two.resolve().as_posix(),
    ]
    assert [item["key"] for item in results] == ["8A", "8A"]
    assert [item["excerpt_count"] for item in results] == [2, 1]
