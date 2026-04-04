import csv
from pathlib import Path

from cuemate_analysis.energy_model import (
    bucket_from_score,
    evaluate_energy_model,
    load_energy_dataset_rows,
    predict_energy_from_features,
    save_energy_bundle,
    train_energy_bundle,
)


def _write_energy_dataset(path: Path, *, rows: int = 60) -> Path:
    fieldnames = [
        "track_id", "file_hash", "file_path", "title", "artist", "playlist_name", "position", "stored_energy_abs",
        "baseline", "loudness_fusion", "club_fusion", "pressure_fusion", "energy_sustained", "energy_peak",
        "loudness_norm", "loudness_lufs", "bass_abs", "drums_abs", "harmonic_abs", "groove_abs",
        "teacher_energy", "teacher_source", "teacher_confidence", "manual_bucket", "manual_score", "manual_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(rows):
            baseline = 0.15 + (0.012 * index)
            loudness_fusion = min(1.0, baseline + 0.04)
            club_fusion = min(1.0, baseline + 0.02)
            pressure_fusion = min(1.0, baseline + 0.03)
            teacher_energy = min(1.0, (0.35 * baseline) + (0.25 * loudness_fusion) + (0.20 * club_fusion) + (0.20 * pressure_fusion))
            if teacher_energy < 0.30:
                bucket = "low"
            elif teacher_energy < 0.55:
                bucket = "groove"
            elif teacher_energy < 0.78:
                bucket = "drive"
            else:
                bucket = "peak"
            writer.writerow(
                {
                    "track_id": f"trk_{index:03d}",
                    "file_hash": f"hash-{index:03d}",
                    "file_path": f"D:/Music/trk_{index:03d}.wav",
                    "title": f"Track {index}",
                    "artist": "Artist",
                    "playlist_name": "Synthetic",
                    "position": index + 1,
                    "stored_energy_abs": round(baseline, 6),
                    "baseline": round(baseline, 6),
                    "loudness_fusion": round(loudness_fusion, 6),
                    "club_fusion": round(club_fusion, 6),
                    "pressure_fusion": round(pressure_fusion, 6),
                    "energy_sustained": round(min(1.0, baseline + 0.01), 6),
                    "energy_peak": round(min(1.0, baseline + 0.05), 6),
                    "loudness_norm": round(min(1.0, baseline + 0.06), 6),
                    "loudness_lufs": round(-18.0 + (index * 0.1), 6),
                    "bass_abs": round(min(1.0, 0.20 + (index * 0.008)), 6),
                    "drums_abs": round(min(1.0, 0.22 + (index * 0.009)), 6),
                    "harmonic_abs": round(min(1.0, 0.60 - (index * 0.003)), 6),
                    "groove_abs": round(min(1.0, 0.24 + (index * 0.007)), 6),
                    "teacher_energy": round(teacher_energy, 6),
                    "teacher_source": "offline_teacher",
                    "teacher_confidence": 0.9,
                    "manual_bucket": bucket,
                    "manual_score": "",
                    "manual_notes": "",
                }
            )
    return path


def test_load_energy_dataset_rows_rejects_duplicate_track_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "duplicate.csv"
    _write_energy_dataset(dataset_path, rows=3)
    with dataset_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trk_000", "hash-x", "D:/x.wav", "Dup", "Artist", "Synthetic", 4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -10.0, 0.4, 0.4, 0.4, 0.4, 0.5, "teacher", 1.0, "drive", "", ""])
    try:
        load_energy_dataset_rows(dataset_path)
        assert False, "expected duplicate track_id validation failure"
    except ValueError as exc:
        assert "duplicate track_id" in str(exc)


def test_train_and_predict_energy_model_is_deterministic(tmp_path: Path) -> None:
    dataset_path = _write_energy_dataset(tmp_path / "dataset.csv")
    rows = load_energy_dataset_rows(dataset_path)

    bundle_a, metrics_a = train_energy_bundle(rows)
    bundle_b, metrics_b = train_energy_bundle(rows)

    model_out_a = tmp_path / "energy-a.joblib"
    meta_out_a = tmp_path / "energy-a.json"
    model_out_b = tmp_path / "energy-b.joblib"
    meta_out_b = tmp_path / "energy-b.json"
    metadata_a = save_energy_bundle(bundle_a, metrics_a, model_out=model_out_a, meta_out=meta_out_a)
    metadata_b = save_energy_bundle(bundle_b, metrics_b, model_out=model_out_b, meta_out=meta_out_b)

    sample = rows[10].features
    prediction_a = predict_energy_from_features(sample, model_path=model_out_a, meta_path=meta_out_a)
    prediction_b = predict_energy_from_features(sample, model_path=model_out_b, meta_path=meta_out_b)

    assert metadata_a["feature_order"] == metadata_b["feature_order"]
    assert metadata_a["calibrated_thresholds"] == metadata_b["calibrated_thresholds"]
    assert prediction_a.hybrid == prediction_b.hybrid
    assert prediction_a.learned == prediction_b.learned
    assert prediction_a.bucket == prediction_b.bucket


def test_benchmark_energy_model_compares_all_scorers(tmp_path: Path) -> None:
    dataset_path = _write_energy_dataset(tmp_path / "dataset.csv")
    rows = load_energy_dataset_rows(dataset_path)
    bundle, metrics = train_energy_bundle(rows)
    model_out = tmp_path / "energy.joblib"
    meta_out = tmp_path / "energy.json"
    save_energy_bundle(bundle, metrics, model_out=model_out, meta_out=meta_out)

    benchmark = evaluate_energy_model(rows, model_path=model_out, meta_path=meta_out)

    assert set(benchmark["comparators"]) == {"baseline", "hybrid_blended", "learned"}
    assert benchmark["comparators"]["learned"]["teacher_metrics"]["mae"] >= 0.0


def test_bucket_from_score_uses_expected_ordering() -> None:
    assert bucket_from_score(0.1) == "low"
    assert bucket_from_score(0.4) == "groove"
    assert bucket_from_score(0.7) == "drive"
    assert bucket_from_score(0.9) == "peak"
