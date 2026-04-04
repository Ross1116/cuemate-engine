from pathlib import Path

from cuemate_analysis.essentia_semantic_backend import (
    build_essentia_semantic_manifest_signature,
    build_essentia_semantic_model_manifest,
    build_essentia_semantic_success_estimate,
)


def _write_artifact(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def test_manifest_signature_changes_when_model_artifact_changes(tmp_path: Path) -> None:
    root = tmp_path / "models"
    for relative_name in [
        "musicnn/msd-musicnn-1.pb",
        "musicnn/msd-musicnn-1.json",
        "musicnn/deam-msd-musicnn-2.pb",
        "musicnn/deam-msd-musicnn-2.json",
        "musicnn/danceability-musicnn-msd-2.pb",
        "musicnn/danceability-musicnn-msd-2.json",
        "musicnn/mood_aggressive-musicnn-msd-1.pb",
        "musicnn/mood_aggressive-musicnn-msd-1.json",
        "musicnn/mood_party-musicnn-msd-1.pb",
        "musicnn/mood_party-musicnn-msd-1.json",
        "musicnn/mood_relaxed-musicnn-msd-1.pb",
        "musicnn/mood_relaxed-musicnn-msd-1.json",
    ]:
        _write_artifact(root / relative_name, f"fixture:{relative_name}")

    manifest = build_essentia_semantic_model_manifest(root)
    first = build_essentia_semantic_manifest_signature(manifest, device="cpu")
    _write_artifact(root / "musicnn" / "mood_party-musicnn-msd-1.pb", "fixture:changed")
    second = build_essentia_semantic_manifest_signature(manifest, device="cpu")

    assert first != second


def test_success_estimate_computes_bounded_fused_score_and_bucket() -> None:
    estimate = build_essentia_semantic_success_estimate(
        {
            "danceability_abs": 0.9,
            "arousal_abs": 0.8,
            "valence_abs": 0.6,
            "mood_aggressive_abs": 0.7,
            "mood_party_abs": 0.85,
            "mood_relaxed_abs": 0.1,
            "loudness_norm": 0.75,
            "drums_abs": 0.8,
            "groove_abs": 0.7,
            "bass_abs": 0.65,
            "semantic_source": "best_per_task:musicnn_stable",
            "runner_device": "cpu",
            "family_map": {"danceability": "musicnn"},
        },
        elapsed_ms=12.5,
        model_signature="abc123",
        image_name="cuemate-essentia-semantics:local",
        device="cpu",
        family_policy="best_per_task",
    )

    assert estimate.available is True
    assert estimate.energy_essentia_fused is not None
    assert 0.0 <= estimate.energy_essentia_fused <= 1.0
    assert estimate.energy_essentia_bucket in {"low", "groove", "drive", "peak"}


def test_success_estimate_leaves_fused_score_empty_when_primary_semantics_missing() -> None:
    estimate = build_essentia_semantic_success_estimate(
        {
            "danceability_abs": 0.9,
            "arousal_abs": None,
            "valence_abs": 0.6,
            "mood_aggressive_abs": 0.7,
            "mood_party_abs": 0.85,
            "mood_relaxed_abs": 0.1,
            "loudness_norm": 0.75,
            "drums_abs": 0.8,
            "groove_abs": 0.7,
            "bass_abs": 0.65,
            "semantic_source": "best_per_task:musicnn_stable",
            "runner_device": "cpu",
            "family_map": {"danceability": "musicnn"},
        },
        elapsed_ms=12.5,
        model_signature="abc123",
        image_name="cuemate-essentia-semantics:local",
        device="cpu",
        family_policy="best_per_task",
    )

    assert estimate.energy_essentia_fused is None
    assert estimate.energy_essentia_bucket is None
