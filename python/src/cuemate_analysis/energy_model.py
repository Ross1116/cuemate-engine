from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error


FEATURE_NAMES = [
    "baseline",
    "loudness_fusion",
    "club_fusion",
    "pressure_fusion",
    "energy_sustained",
    "energy_peak",
    "loudness_norm",
    "loudness_lufs",
    "bass_abs",
    "drums_abs",
    "harmonic_abs",
    "groove_abs",
]
HYBRID_FEATURE_NAMES = [
    "baseline",
    "loudness_fusion",
    "club_fusion",
    "pressure_fusion",
]
MANUAL_BUCKETS = ["low", "groove", "drive", "peak"]
MANUAL_BUCKET_ANCHORS = {
    "low": 0.125,
    "groove": 0.375,
    "drive": 0.625,
    "peak": 0.875,
}
FALLBACK_THRESHOLDS = {
    "low": 0.30,
    "groove": 0.55,
    "drive": 0.78,
}
DEFAULT_MODEL_SOURCE = "teacher_first_hgbr"
MIN_MANUAL_ROWS_FOR_CALIBRATION = 40
MIN_MANUAL_ROWS_FOR_THRESHOLD_SEARCH = 20


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return float(max(minimum, min(maximum, value)))


def resolve_energy_model_path(model_path: str | None, repo_root: Path) -> Path:
    raw = model_path or "python/models/energy/teacher_first_v1.joblib"
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def resolve_energy_model_meta_path(meta_path: str | None, repo_root: Path) -> Path:
    raw = meta_path or "python/models/energy/teacher_first_v1.meta.json"
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


@dataclass(frozen=True)
class EnergyFeatureVector:
    baseline: float
    loudness_fusion: float
    club_fusion: float
    pressure_fusion: float
    energy_sustained: float
    energy_peak: float
    loudness_norm: float
    loudness_lufs: float
    bass_abs: float
    drums_abs: float
    harmonic_abs: float
    groove_abs: float

    def to_payload(self) -> dict[str, float]:
        return asdict(self)

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in FEATURE_NAMES], dtype=float)

    def hybrid_array(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in HYBRID_FEATURE_NAMES], dtype=float)


@dataclass(frozen=True)
class EnergyDatasetRow:
    track_id: str
    file_hash: str
    file_path: str
    title: str | None
    artist: str | None
    playlist_name: str | None
    position: int | None
    stored_energy_abs: float | None
    features: EnergyFeatureVector
    teacher_energy: float | None
    teacher_source: str | None
    teacher_confidence: float
    manual_bucket: str | None
    manual_score: float | None
    manual_notes: str | None


@dataclass(frozen=True)
class EnergyInferenceResult:
    hybrid: float
    learned: float
    bucket: str
    model_signature: str
    model_source: str


def _safe_float(value: float | None, fallback: float = 0.5) -> float:
    if value is None:
        return fallback
    return float(value)


def build_energy_feature_vector(
    *,
    energy_abs: float,
    energy_sustained: float | None,
    energy_peak: float | None,
    loudness_norm: float,
    loudness_lufs: float,
    bass_abs: float,
    drums_abs: float | None,
    harmonic_abs: float | None,
    groove_abs: float | None,
) -> EnergyFeatureVector:
    baseline = clamp(float(energy_abs))
    sustained = clamp(_safe_float(energy_sustained))
    peak = clamp(_safe_float(energy_peak))
    drums_value = clamp(_safe_float(drums_abs))
    harmonic_value = clamp(_safe_float(harmonic_abs))
    groove_value = clamp(_safe_float(groove_abs))
    bass_value = clamp(float(bass_abs))
    loudness_norm_value = clamp(float(loudness_norm))
    loudness_lufs_value = float(loudness_lufs)

    loudness_fusion = clamp(
        (0.40 * loudness_norm_value)
        + (0.28 * baseline)
        + (0.18 * peak)
        + (0.14 * sustained)
    )
    club_fusion = clamp(
        (0.28 * baseline)
        + (0.22 * loudness_norm_value)
        + (0.18 * drums_value)
        + (0.18 * groove_value)
        + (0.14 * bass_value)
    )
    pressure_fusion = clamp(
        (0.24 * peak)
        + (0.24 * drums_value)
        + (0.20 * loudness_norm_value)
        + (0.16 * groove_value)
        + (0.10 * bass_value)
        + (0.06 * (1.0 - harmonic_value))
    )
    return EnergyFeatureVector(
        baseline=baseline,
        loudness_fusion=loudness_fusion,
        club_fusion=club_fusion,
        pressure_fusion=pressure_fusion,
        energy_sustained=sustained,
        energy_peak=peak,
        loudness_norm=loudness_norm_value,
        loudness_lufs=loudness_lufs_value,
        bass_abs=bass_value,
        drums_abs=drums_value,
        harmonic_abs=harmonic_value,
        groove_abs=groove_value,
    )


def energy_consensus(features: EnergyFeatureVector) -> float:
    return clamp(float(np.mean(features.hybrid_array())))


def bucket_from_score(score: float, thresholds: dict[str, float] | None = None) -> str:
    bounds = thresholds or FALLBACK_THRESHOLDS
    if score < float(bounds["low"]):
        return "low"
    if score < float(bounds["groove"]):
        return "groove"
    if score < float(bounds["drive"]):
        return "drive"
    return "peak"


def _hash_track_id(track_id: str) -> float:
    digest = hashlib.sha1(track_id.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def split_dataset_rows(rows: list[EnergyDatasetRow], train_ratio: float = 0.8) -> tuple[list[EnergyDatasetRow], list[EnergyDatasetRow]]:
    train_rows: list[EnergyDatasetRow] = []
    validation_rows: list[EnergyDatasetRow] = []
    for row in rows:
        if _hash_track_id(row.track_id) < train_ratio:
            train_rows.append(row)
        else:
            validation_rows.append(row)
    if not train_rows or not validation_rows:
        midpoint = max(1, int(len(rows) * train_ratio))
        return rows[:midpoint], rows[midpoint:]
    return train_rows, validation_rows


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return int(stripped)


def load_energy_dataset_rows(dataset_path: Path) -> list[EnergyDatasetRow]:
    resolved = dataset_path.expanduser().resolve()
    seen_track_ids: set[str] = set()
    rows: list[EnergyDatasetRow] = []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, raw in enumerate(reader, start=2):
            track_id = (raw.get("track_id") or "").strip()
            if not track_id:
                raise ValueError(f"{resolved}:{index}: missing track_id")
            if track_id in seen_track_ids:
                raise ValueError(f"{resolved}:{index}: duplicate track_id '{track_id}'")
            seen_track_ids.add(track_id)

            teacher_energy = _parse_optional_float(raw.get("teacher_energy"))
            if teacher_energy is not None and not 0.0 <= teacher_energy <= 1.0:
                raise ValueError(f"{resolved}:{index}: teacher_energy must be between 0 and 1")

            teacher_confidence = _parse_optional_float(raw.get("teacher_confidence"))
            teacher_confidence_value = 1.0 if teacher_confidence is None else teacher_confidence
            if not 0.0 <= teacher_confidence_value <= 1.0:
                raise ValueError(f"{resolved}:{index}: teacher_confidence must be between 0 and 1")

            manual_bucket = (raw.get("manual_bucket") or "").strip().lower() or None
            if manual_bucket is not None and manual_bucket not in MANUAL_BUCKETS:
                raise ValueError(f"{resolved}:{index}: invalid manual_bucket '{manual_bucket}'")

            manual_score = _parse_optional_float(raw.get("manual_score"))
            if manual_score is not None and not 0.0 <= manual_score <= 1.0:
                raise ValueError(f"{resolved}:{index}: manual_score must be between 0 and 1")
            if manual_score is None and manual_bucket is not None:
                manual_score = MANUAL_BUCKET_ANCHORS[manual_bucket]

            features = EnergyFeatureVector(
                baseline=float(raw["baseline"]),
                loudness_fusion=float(raw["loudness_fusion"]),
                club_fusion=float(raw["club_fusion"]),
                pressure_fusion=float(raw["pressure_fusion"]),
                energy_sustained=float(raw["energy_sustained"]),
                energy_peak=float(raw["energy_peak"]),
                loudness_norm=float(raw["loudness_norm"]),
                loudness_lufs=float(raw["loudness_lufs"]),
                bass_abs=float(raw["bass_abs"]),
                drums_abs=float(raw["drums_abs"]),
                harmonic_abs=float(raw["harmonic_abs"]),
                groove_abs=float(raw["groove_abs"]),
            )
            rows.append(
                EnergyDatasetRow(
                    track_id=track_id,
                    file_hash=(raw.get("file_hash") or "").strip(),
                    file_path=(raw.get("file_path") or "").strip(),
                    title=(raw.get("title") or "").strip() or None,
                    artist=(raw.get("artist") or "").strip() or None,
                    playlist_name=(raw.get("playlist_name") or "").strip() or None,
                    position=_parse_optional_int(raw.get("position")),
                    stored_energy_abs=_parse_optional_float(raw.get("stored_energy_abs")),
                    features=features,
                    teacher_energy=teacher_energy,
                    teacher_source=(raw.get("teacher_source") or "").strip() or None,
                    teacher_confidence=teacher_confidence_value,
                    manual_bucket=manual_bucket,
                    manual_score=manual_score,
                    manual_notes=(raw.get("manual_notes") or "").strip() or None,
                )
            )
    return rows


def _weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    absolute_error = np.abs(y_true - y_pred)
    if weights is None:
        return float(np.mean(absolute_error))
    return float(np.average(absolute_error, weights=weights))


def _weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    squared_error = np.square(y_true - y_pred)
    if weights is None:
        return float(np.sqrt(np.mean(squared_error)))
    return float(np.sqrt(np.average(squared_error, weights=weights)))


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return 0.0
    corr = spearmanr(y_true, y_pred).statistic
    if corr is None or np.isnan(corr):
        return 0.0
    return float(corr)


def train_hybrid_weights(rows: list[EnergyDatasetRow]) -> dict[str, float]:
    teacher_rows = [row for row in rows if row.teacher_energy is not None]
    if not teacher_rows:
        raise ValueError("Hybrid training requires at least one teacher-labeled row.")
    x = np.vstack([row.features.hybrid_array() for row in teacher_rows])
    y = np.array([float(row.teacher_energy) for row in teacher_rows], dtype=float)
    sample_weight = np.array([float(row.teacher_confidence) for row in teacher_rows], dtype=float)

    def objective(weights: np.ndarray) -> float:
        prediction = np.clip(x @ weights, 0.0, 1.0)
        return float(np.average(np.square(prediction - y), weights=sample_weight))

    initial = np.full(len(HYBRID_FEATURE_NAMES), 1.0 / len(HYBRID_FEATURE_NAMES), dtype=float)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(HYBRID_FEATURE_NAMES),
        constraints=[{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}],
    )
    if not result.success:
        weights = initial
    else:
        weights = np.clip(result.x, 0.0, 1.0)
        total = float(np.sum(weights)) or 1.0
        weights = weights / total
    return {name: float(value) for name, value in zip(HYBRID_FEATURE_NAMES, weights, strict=True)}


def hybrid_score(features: EnergyFeatureVector, weights: dict[str, float]) -> float:
    return clamp(sum(getattr(features, name) * float(weights.get(name, 0.0)) for name in HYBRID_FEATURE_NAMES))


def _rows_to_matrix(rows: list[EnergyDatasetRow]) -> np.ndarray:
    return np.vstack([row.features.to_array() for row in rows]).astype(float)


def _teacher_arrays(rows: list[EnergyDatasetRow]) -> tuple[np.ndarray, np.ndarray]:
    y = np.array([float(row.teacher_energy) for row in rows], dtype=float)
    sample_weight = np.array([float(row.teacher_confidence) for row in rows], dtype=float)
    return y, sample_weight


def _manual_rows(rows: list[EnergyDatasetRow]) -> list[EnergyDatasetRow]:
    return [row for row in rows if row.manual_bucket is not None and row.manual_score is not None]


def _candidate_thresholds(scores: np.ndarray) -> list[float]:
    unique = sorted({round(float(score), 6) for score in scores})
    if len(unique) < 2:
        return []
    return [(unique[index] + unique[index + 1]) / 2.0 for index in range(len(unique) - 1)]


def optimize_bucket_thresholds(scores: np.ndarray, buckets: list[str]) -> dict[str, float]:
    candidates = _candidate_thresholds(scores)
    if len(candidates) < 3:
        return dict(FALLBACK_THRESHOLDS)

    best_thresholds = dict(FALLBACK_THRESHOLDS)
    best_score = -1.0
    y_true = buckets

    for low_index in range(len(candidates) - 2):
        low = candidates[low_index]
        for groove_index in range(low_index + 1, len(candidates) - 1):
            groove = candidates[groove_index]
            for drive_index in range(groove_index + 1, len(candidates)):
                drive = candidates[drive_index]
                thresholds = {"low": low, "groove": groove, "drive": drive}
                y_pred = [bucket_from_score(float(score), thresholds) for score in scores]
                score = float(f1_score(y_true, y_pred, labels=MANUAL_BUCKETS, average="macro", zero_division=0))
                if score > best_score:
                    best_score = score
                    best_thresholds = thresholds
    return best_thresholds


def _manual_metrics(scores: np.ndarray, rows: list[EnergyDatasetRow], thresholds: dict[str, float]) -> dict[str, float] | None:
    manual_rows = _manual_rows(rows)
    if not manual_rows:
        return None
    truth_buckets = [str(row.manual_bucket) for row in manual_rows]
    truth_scores = np.array([float(row.manual_score) for row in manual_rows], dtype=float)
    predicted_scores = scores
    predicted_buckets = [bucket_from_score(float(score), thresholds) for score in predicted_scores]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        weighted_kappa = cohen_kappa_score(truth_buckets, predicted_buckets, labels=MANUAL_BUCKETS, weights="quadratic")
    if np.isnan(weighted_kappa):
        weighted_kappa = 0.0
    return {
        "bucket_accuracy": float(accuracy_score(truth_buckets, predicted_buckets)),
        "macro_f1": float(f1_score(truth_buckets, predicted_buckets, labels=MANUAL_BUCKETS, average="macro", zero_division=0)),
        "weighted_kappa": float(weighted_kappa),
        "score_mae": float(mean_absolute_error(truth_scores, predicted_scores)),
    }


def _teacher_metrics(rows: list[EnergyDatasetRow], scores: np.ndarray) -> dict[str, float] | None:
    teacher_rows = [row for row in rows if row.teacher_energy is not None]
    if not teacher_rows:
        return None
    y_true = np.array([float(row.teacher_energy) for row in teacher_rows], dtype=float)
    sample_weight = np.array([float(row.teacher_confidence) for row in teacher_rows], dtype=float)
    return {
        "mae": _weighted_mae(y_true, scores, sample_weight),
        "rmse": _weighted_rmse(y_true, scores, sample_weight),
        "spearman": _spearman(y_true, scores),
    }


def train_energy_bundle(rows: list[EnergyDatasetRow]) -> tuple[dict[str, Any], dict[str, Any]]:
    teacher_rows = [row for row in rows if row.teacher_energy is not None]
    if len(teacher_rows) < 8:
        raise ValueError("Training requires at least 8 teacher-labeled rows.")

    train_rows, validation_rows = split_dataset_rows(teacher_rows)
    x_train = _rows_to_matrix(train_rows)
    y_train, sample_weight = _teacher_arrays(train_rows)
    x_validation = _rows_to_matrix(validation_rows)
    y_validation, validation_weight = _teacher_arrays(validation_rows)

    hybrid_weights = train_hybrid_weights(train_rows)
    hybrid_validation = np.array([hybrid_score(row.features, hybrid_weights) for row in validation_rows], dtype=float)

    model = HistGradientBoostingRegressor(
        random_state=42,
        learning_rate=0.05,
        max_depth=4,
        max_iter=250,
        min_samples_leaf=4,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    raw_validation = np.clip(model.predict(x_validation), 0.0, 1.0)

    manual_train_rows = _manual_rows(train_rows)
    calibrator: IsotonicRegression | None = None
    calibrator_applied = False
    if len(manual_train_rows) >= MIN_MANUAL_ROWS_FOR_CALIBRATION:
        calibration_inputs = np.clip(
            model.predict(np.vstack([row.features.to_array() for row in manual_train_rows]).astype(float)),
            0.0,
            1.0,
        )
        calibration_targets = np.array([float(row.manual_score) for row in manual_train_rows], dtype=float)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(calibration_inputs, calibration_targets)
        final_validation = np.clip(calibrator.predict(raw_validation), 0.0, 1.0)
        calibrator_applied = True
    else:
        final_validation = raw_validation

    validation_manual_rows = _manual_rows(validation_rows)
    if len(validation_manual_rows) >= MIN_MANUAL_ROWS_FOR_THRESHOLD_SEARCH:
        validation_manual_scores = np.array(
            [float(final_validation[index]) for index, row in enumerate(validation_rows) if row.manual_bucket is not None and row.manual_score is not None],
            dtype=float,
        )
        validation_manual_buckets = [str(row.manual_bucket) for row in validation_rows if row.manual_bucket is not None and row.manual_score is not None]
        thresholds = optimize_bucket_thresholds(validation_manual_scores, validation_manual_buckets)
    else:
        thresholds = dict(FALLBACK_THRESHOLDS)

    final_bundle = {
        "model": model,
        "calibrator": calibrator,
        "hybrid_weights": hybrid_weights,
        "thresholds": thresholds,
        "feature_names": FEATURE_NAMES,
        "model_source": DEFAULT_MODEL_SOURCE,
    }
    metrics = {
        "dataset_counts": {
            "teacher_total": len(teacher_rows),
            "train_teacher": len(train_rows),
            "validation_teacher": len(validation_rows),
            "manual_total": len(_manual_rows(rows)),
            "train_manual": len(manual_train_rows),
            "validation_manual": len(validation_manual_rows),
        },
        "teacher_metrics": {
            "baseline": _teacher_metrics(validation_rows, np.array([row.features.baseline for row in validation_rows], dtype=float)),
            "hybrid_blended": _teacher_metrics(validation_rows, hybrid_validation),
            "learned": _teacher_metrics(validation_rows, final_validation),
        },
        "manual_metrics": {
            "baseline": _manual_metrics(
                np.array([row.features.baseline for row in validation_manual_rows], dtype=float),
                validation_manual_rows,
                thresholds,
            ) if validation_manual_rows else None,
            "hybrid_blended": _manual_metrics(
                np.array([hybrid_score(row.features, hybrid_weights) for row in validation_manual_rows], dtype=float),
                validation_manual_rows,
                thresholds,
            ) if validation_manual_rows else None,
            "learned": _manual_metrics(
                np.array(
                    [
                        float(
                            calibrator.predict([np.clip(model.predict(row.features.to_array().reshape(1, -1))[0], 0.0, 1.0)])[0]
                            if calibrator is not None
                            else np.clip(model.predict(row.features.to_array().reshape(1, -1))[0], 0.0, 1.0)
                        )
                        for row in validation_manual_rows
                    ],
                    dtype=float,
                ),
                validation_manual_rows,
                thresholds,
            ) if validation_manual_rows else None,
        },
        "thresholds": thresholds,
        "calibrator_applied": calibrator_applied,
        "manual_threshold_rows": len(validation_manual_rows),
    }
    return final_bundle, metrics


def save_energy_bundle(bundle: dict[str, Any], metrics: dict[str, Any], *, model_out: Path, meta_out: Path) -> dict[str, Any]:
    model_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_out)
    artifact_hash = hashlib.sha256(model_out.read_bytes()).hexdigest()
    metadata = {
        "model_type": "HistGradientBoostingRegressor",
        "feature_order": FEATURE_NAMES,
        "hybrid_features": HYBRID_FEATURE_NAMES,
        "trained_at": utc_now(),
        "dataset_counts": metrics["dataset_counts"],
        "teacher_metrics": metrics["teacher_metrics"],
        "manual_metrics": metrics["manual_metrics"],
        "calibrated_thresholds": metrics["thresholds"],
        "artifact_signature": artifact_hash[:16],
        "artifact_sha256": artifact_hash,
        "model_source": DEFAULT_MODEL_SOURCE,
        "calibrator_applied": metrics["calibrator_applied"],
    }
    meta_out.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


@lru_cache(maxsize=8)
def _load_energy_bundle_cached(
    model_path_str: str,
    meta_path_str: str,
    model_mtime_ns: int,
    meta_mtime_ns: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    del model_mtime_ns, meta_mtime_ns
    bundle = joblib.load(model_path_str)
    metadata = json.loads(Path(meta_path_str).read_text(encoding="utf-8"))
    return bundle, metadata


def load_energy_bundle(model_path: Path, meta_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_model = model_path.expanduser().resolve()
    resolved_meta = meta_path.expanduser().resolve()
    return _load_energy_bundle_cached(
        str(resolved_model),
        str(resolved_meta),
        resolved_model.stat().st_mtime_ns,
        resolved_meta.stat().st_mtime_ns,
    )


def predict_energy_from_features(
    features: EnergyFeatureVector,
    *,
    model_path: Path,
    meta_path: Path,
) -> EnergyInferenceResult:
    bundle, metadata = load_energy_bundle(model_path, meta_path)
    hybrid_weights = bundle["hybrid_weights"]
    thresholds = bundle["thresholds"]
    hybrid = hybrid_score(features, hybrid_weights)
    raw_prediction = float(np.clip(bundle["model"].predict(features.to_array().reshape(1, -1))[0], 0.0, 1.0))
    calibrator = bundle.get("calibrator")
    if calibrator is not None:
        learned = float(np.clip(calibrator.predict([raw_prediction])[0], 0.0, 1.0))
    else:
        learned = raw_prediction
    return EnergyInferenceResult(
        hybrid=round(hybrid, 6),
        learned=round(learned, 6),
        bucket=bucket_from_score(learned, thresholds),
        model_signature=str(metadata["artifact_signature"]),
        model_source=str(metadata.get("model_source", DEFAULT_MODEL_SOURCE)),
    )


def evaluate_energy_model(
    rows: list[EnergyDatasetRow],
    *,
    model_path: Path,
    meta_path: Path,
) -> dict[str, Any]:
    teacher_rows = [row for row in rows if row.teacher_energy is not None]
    if len(teacher_rows) < 8:
        raise ValueError("Benchmarking requires at least 8 teacher-labeled rows.")
    train_rows, validation_rows = split_dataset_rows(teacher_rows)
    del train_rows

    bundle, metadata = load_energy_bundle(model_path, meta_path)
    hybrid_weights = bundle["hybrid_weights"]
    thresholds = bundle["thresholds"]
    calibrator = bundle.get("calibrator")

    baseline_scores = np.array([row.features.baseline for row in validation_rows], dtype=float)
    hybrid_scores = np.array([hybrid_score(row.features, hybrid_weights) for row in validation_rows], dtype=float)
    raw_scores = np.array(
        [float(np.clip(bundle["model"].predict(row.features.to_array().reshape(1, -1))[0], 0.0, 1.0)) for row in validation_rows],
        dtype=float,
    )
    learned_scores = np.array(
        [float(np.clip(calibrator.predict([score])[0], 0.0, 1.0)) if calibrator is not None else float(score) for score in raw_scores],
        dtype=float,
    )

    validation_manual_rows = _manual_rows(validation_rows)
    manual_baseline_scores = np.array([row.features.baseline for row in validation_manual_rows], dtype=float)
    manual_hybrid_scores = np.array([hybrid_score(row.features, hybrid_weights) for row in validation_manual_rows], dtype=float)
    manual_learned_scores = np.array(
        [
            float(np.clip(calibrator.predict([raw])[0], 0.0, 1.0)) if calibrator is not None else float(raw)
            for raw, row in zip(raw_scores, validation_rows, strict=True)
            if row.manual_bucket is not None and row.manual_score is not None
        ],
        dtype=float,
    )

    return {
        "model_signature": metadata["artifact_signature"],
        "thresholds": thresholds,
        "dataset_counts": {
            "teacher_total": len(teacher_rows),
            "validation_teacher": len(validation_rows),
            "validation_manual": len(validation_manual_rows),
        },
        "comparators": {
            "baseline": {
                "teacher_metrics": _teacher_metrics(validation_rows, baseline_scores),
                "manual_metrics": _manual_metrics(manual_baseline_scores, validation_manual_rows, thresholds) if validation_manual_rows else None,
            },
            "hybrid_blended": {
                "teacher_metrics": _teacher_metrics(validation_rows, hybrid_scores),
                "manual_metrics": _manual_metrics(manual_hybrid_scores, validation_manual_rows, thresholds) if validation_manual_rows else None,
            },
            "learned": {
                "teacher_metrics": _teacher_metrics(validation_rows, learned_scores),
                "manual_metrics": _manual_metrics(manual_learned_scores, validation_manual_rows, thresholds) if validation_manual_rows else None,
            },
        },
    }
