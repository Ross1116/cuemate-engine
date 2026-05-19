from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_STATIC_WEIGHTS = {
    "target_energy": 0.22,
    "transition_support": 0.18,
    "bass_transition": 0.15,
    "vocal_transition": 0.13,
    "harmonic": 0.12,
    "tempo": 0.10,
    "history_fit": 0.06,
    "rhythmic_continuity": 0.04,
}

DEFAULT_WEIGHT_FLOORS = {
    "target_energy": 0.08,
    "transition_support": 0.05,
    "bass_transition": 0.04,
    "vocal_transition": 0.03,
    "harmonic": 0.04,
    "tempo": 0.03,
    "history_fit": 0.03,
    "rhythmic_continuity": 0.02,
}

DEFAULT_SCORING_THRESHOLDS = {
    "bpm_hard": 8.0,
    "bpm_hard_by_target": {
        "maintain": 8.0,
        "build": 12.0,
        "jump": 20.0,
        "reset": 20.0,
    },
    "bpm_ratio_pass": ["half", "double", "three_two", "two_three"],
    "bpm_soft": 3.0,
    "cooldown_window": 5,
}

DEFAULT_SCORING_MOVE_TYPES = {
    "jump_threshold": 0.12,
    "build_threshold": 0.05,
    "maintain_range": 0.05,
    "reset_energy_threshold": -0.08,
    "reset_vocal_threshold": 0.50,
    "drop_threshold": -0.05,
}

DEFAULT_SCORING_PENALTIES = {
    "max_total_penalty": 0.80,
    "bpm_over_soft": 0.30,
    "key_mismatch": 0.45,
    "vocal_clash": 0.35,
}


@dataclass(frozen=True)
class AnalysisSettings:
    sample_rate: int
    mono: bool
    key_backend: str
    key_model_path: str | None
    key_device: str
    key_policy: str
    parallel_workers: int
    max_workers_auto: bool
    per_track_timeout_seconds: int
    model_preload: bool
    fast_pass_enabled: bool
    analysis_signature_seed: str
    energy_source_default: str
    essentia_semantics_enabled: bool
    essentia_semantic_image: str
    essentia_semantic_device: str
    essentia_semantic_model_family_policy: str
    essentia_semantic_model_root: str
    essentia_semantic_default_excerpt_seconds: float
    essentia_semantic_multisample_excerpt_seconds: float
    essentia_semantic_trigger_mismatch_threshold: float
    essentia_semantic_trigger_confidence_threshold: float
    essentia_semantic_trigger_structure_rms_cv: float
    essentia_semantic_trigger_outlier_zscore: float
    full_chunk_size: int
    tempo_chunk_size: int
    key_chunk_size: int
    essentia_chunk_size: int
    dsp_workers: int


@dataclass(frozen=True)
class ThresholdSettings:
    small_playlist_limit: int
    min_playlist_for_relative: int


@dataclass(frozen=True)
class ScoringSettings:
    """Scoring config payload for build_scoring_config().

    __post_init__ uses object.__setattr__ because this frozen dataclass still needs
    per-instance mutable defaults for thresholds, move_types, and penalties without
    sharing a single dict across instances.
    """
    static_weights: dict[str, float]
    weight_floors: dict[str, float]
    harmonic_confidence_floor: float = 0.15
    key_confidence_threshold: float = 0.5
    thresholds: dict[str, Any] = None  # type: ignore[assignment]
    move_types: dict[str, Any] = None  # type: ignore[assignment]
    penalties: dict[str, Any] = None  # type: ignore[assignment]
    contrast_threshold: float = 0.45
    secondary_contrast_threshold: float = 0.65
    max_per_lane: int = 3

    def __post_init__(self) -> None:
        # Provide mutable defaults after frozen dataclass construction
        if self.thresholds is None:
            object.__setattr__(self, "thresholds", dict(DEFAULT_SCORING_THRESHOLDS))
        if self.move_types is None:
            object.__setattr__(self, "move_types", dict(DEFAULT_SCORING_MOVE_TYPES))
        if self.penalties is None:
            object.__setattr__(self, "penalties", dict(DEFAULT_SCORING_PENALTIES))


@dataclass(frozen=True)
class WeightAdaptationSettings:
    mode: str
    adaptation_strength: float


@dataclass(frozen=True)
class FeedbackSettings:
    learning_rate: float = 0.35
    max_component_shift: float = 0.25
    min_contributory_events: int = 20
    min_pairwise_comparisons: int = 40
    min_new_events_since_last_tune: int = 5


@dataclass(frozen=True)
class SemanticHeadCalibration:
    offset: float
    scale: float


@dataclass(frozen=True)
class SemanticCalibrationSettings:
    calibration_version: str
    heads: dict[str, SemanticHeadCalibration]

    def calibrate(self, head_name: str, raw_value: float) -> float:
        entry = self.heads.get(head_name)
        if entry is None:
            return max(0.0, min(1.0, raw_value))
        return max(0.0, min(1.0, (raw_value - entry.offset) * entry.scale))


DEFAULT_SEMANTIC_CALIBRATION = SemanticCalibrationSettings(
    calibration_version="identity",
    heads={},
)


@dataclass(frozen=True)
class RuntimeSettings:
    repo_root: Path
    env_path: Path
    config_path: Path
    database_path: Path
    database_url: str
    config_signature: str
    analysis_signature: str
    analysis: AnalysisSettings
    thresholds: ThresholdSettings
    scoring: ScoringSettings
    weight_adaptation: WeightAdaptationSettings
    feedback: FeedbackSettings
    semantic_calibration: SemanticCalibrationSettings


def find_repo_root(start: Path | None = None) -> Path:
    candidates: list[Path] = []
    if start is not None:
        resolved = start.resolve()
        candidates.extend([resolved, *resolved.parents])

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    candidates.extend(Path(__file__).resolve().parents)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "python" / "pyproject.toml").is_file() and (
            candidate / "db" / "schema.sql"
        ).is_file():
            return candidate

    raise FileNotFoundError("Unable to locate the CueMate repository root.")


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def resolve_database_path(database_url: str, repo_root: Path) -> Path:
    if not database_url.startswith("sqlite:"):
        raise ValueError(f"Unsupported DATABASE_URL for Milestone 1: {database_url}")

    raw_path = database_url[len("sqlite:") :]
    while raw_path.startswith("//"):
        raw_path = raw_path[1:]

    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_semantic_calibration(repo_root: Path) -> SemanticCalibrationSettings:
    calibration_path = repo_root / "config" / "essentia_semantic_calibration.json"
    if not calibration_path.is_file():
        return DEFAULT_SEMANTIC_CALIBRATION
    payload = _load_json(calibration_path)
    heads: dict[str, SemanticHeadCalibration] = {}
    for head_name, entry in payload.get("heads", {}).items():
        heads[head_name] = SemanticHeadCalibration(
            offset=float(entry.get("offset", 0.0)),
            scale=float(entry.get("scale", 1.0)),
        )
    return SemanticCalibrationSettings(
        calibration_version=str(payload.get("calibration_version", "unknown")),
        heads=heads,
    )


def _serialize_semantic_calibration(calibration: SemanticCalibrationSettings) -> dict[str, Any]:
    return {
        "calibration_version": calibration.calibration_version,
        "heads": {
            head_name: {"offset": entry.offset, "scale": entry.scale}
            for head_name, entry in sorted(calibration.heads.items())
        },
    }


def _relative_path_for_fingerprint(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _fingerprint_file(path: Path, *, repo_root: Path) -> dict[str, Any]:
    return {
        "relative_path": _relative_path_for_fingerprint(path, repo_root),
        "size": path.stat().st_size,
        "sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
    }


def _fingerprint_directory(path: Path, *, repo_root: Path) -> str:
    if not path.exists():
        return "missing"

    if path.is_file():
        payload = _fingerprint_file(path, repo_root=repo_root)
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    entries: list[dict[str, Any]] = []
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        entries.append(
            {
                "relative_path": _relative_path_for_fingerprint(child, repo_root),
                "size": child.stat().st_size,
                "sha1": hashlib.sha1(child.read_bytes()).hexdigest(),
            }
        )

    return hashlib.sha1(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()


def resolve_image_digest(image_name: str) -> str:
    """
    Best-effort immutable image identity.

    Returns:
    - repo digest form like 'image@sha256:...'
    - otherwise the original image name if digest resolution is unavailable
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name, "--format", "{{json .RepoDigests}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return image_name

        raw = result.stdout.strip()
        if not raw:
            return image_name

        digests = json.loads(raw)
        if isinstance(digests, list) and digests:
            for entry in digests:
                if isinstance(entry, str) and "@sha256:" in entry:
                    return entry
        return image_name
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError, ValueError):
        return image_name


def _analysis_signature(
    config_signature: str,
    analysis_config: AnalysisSettings,
    *,
    repo_root: Path,
    semantic_calibration: SemanticCalibrationSettings,
) -> str:
    model_root = Path(analysis_config.essentia_semantic_model_root)
    if not model_root.is_absolute():
        model_root = (repo_root / model_root).resolve()

    payload = {
        "config_signature": config_signature,
        "seed": analysis_config.analysis_signature_seed,
        "sample_rate": analysis_config.sample_rate,
        "mono": analysis_config.mono,
        "key_backend": analysis_config.key_backend,
        "key_model_path": analysis_config.key_model_path,
        "key_device": analysis_config.key_device,
        "key_policy": analysis_config.key_policy,
        "fast_pass_enabled": analysis_config.fast_pass_enabled,
        "essentia_semantics_enabled": analysis_config.essentia_semantics_enabled,
        "essentia_semantic_image_digest": resolve_image_digest(analysis_config.essentia_semantic_image),
        "essentia_semantic_device": analysis_config.essentia_semantic_device,
        "essentia_semantic_model_family_policy": analysis_config.essentia_semantic_model_family_policy,
        "essentia_semantic_model_root_fingerprint": _fingerprint_directory(model_root, repo_root=repo_root),
        "essentia_semantic_default_excerpt_seconds": analysis_config.essentia_semantic_default_excerpt_seconds,
        "essentia_semantic_multisample_excerpt_seconds": analysis_config.essentia_semantic_multisample_excerpt_seconds,
        "essentia_semantic_trigger_mismatch_threshold": analysis_config.essentia_semantic_trigger_mismatch_threshold,
        "essentia_semantic_trigger_confidence_threshold": analysis_config.essentia_semantic_trigger_confidence_threshold,
        "essentia_semantic_trigger_structure_rms_cv": analysis_config.essentia_semantic_trigger_structure_rms_cv,
        "essentia_semantic_trigger_outlier_zscore": analysis_config.essentia_semantic_trigger_outlier_zscore,
        "semantic_calibration": _serialize_semantic_calibration(semantic_calibration),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"m1-{digest[:12]}"

def _fast_analysis_signature(
    config_signature: str,
    analysis_config: AnalysisSettings,
) -> str:
    payload = {
        "config_signature": config_signature,
        "seed": analysis_config.analysis_signature_seed,
        "sample_rate": analysis_config.sample_rate,
        "mono": analysis_config.mono,
        "key_backend": analysis_config.key_backend,
        "key_model_path": analysis_config.key_model_path,
        "key_device": analysis_config.key_device,
        "key_policy": analysis_config.key_policy,
        "fast_pass_enabled": analysis_config.fast_pass_enabled,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"m1fast-{digest[:12]}"

def _relative_signature(
    config_signature: str,
    thresholds: ThresholdSettings,
    scoring: ScoringSettings,
    weight_adaptation: WeightAdaptationSettings,
    energy_source: str,
) -> str:
    payload = {
        "config_signature": config_signature,
        "small_playlist_limit": thresholds.small_playlist_limit,
        "min_playlist_for_relative": thresholds.min_playlist_for_relative,
        "static_weights": scoring.static_weights,
        "weight_floors": scoring.weight_floors,
        "weight_adaptation_mode": weight_adaptation.mode,
        "adaptation_strength": weight_adaptation.adaptation_strength,
        "energy_source": energy_source,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"m2exp-{digest[:12]}"


def load_runtime_settings(repo_root: Path | None = None) -> RuntimeSettings:
    root = find_repo_root(repo_root)
    env_path = root / ".env"
    env_example_path = root / ".env.example"

    env_values = load_env_file(env_example_path)
    env_values.update(load_env_file(env_path))
    env_values.update(
        {
            key: value
            for key, value in os.environ.items()
            if key in env_values or key.startswith("CUEMATE_")
        }
    )

    config_path_value = env_values.get("CUEMATE_CONFIG_PATH", "config/default.json")
    config_path = (root / config_path_value).resolve()
    config_payload = _load_json(config_path)
    analysis_payload = config_payload.get("analysis", {})
    thresholds_payload = config_payload.get("thresholds", {})
    scoring_payload = config_payload.get("scoring", {})
    weight_adaptation_payload = config_payload.get("weight_adaptation", {})
    feedback_payload = config_payload.get("feedback", {})

    parallel_workers = int(analysis_payload.get("parallel_workers", 4))
    max_workers_auto = bool(analysis_payload.get("max_workers_auto", True))
    auto_dsp_workers = parallel_workers
    if max_workers_auto:
        cpu_count = os.cpu_count() or parallel_workers
        auto_dsp_workers = max(1, min(cpu_count, 8))

    analysis_settings = AnalysisSettings(
        sample_rate=int(analysis_payload.get("sample_rate", 22050)),
        mono=bool(analysis_payload.get("mono", True)),
        key_backend=str(analysis_payload.get("key_backend", analysis_payload.get("key_model", "musicalkeycnn"))),
        key_model_path=analysis_payload.get("musicalkeycnn_model"),
        key_device=str(analysis_payload.get("musicalkeycnn_device", "auto")),
        key_policy=str(analysis_payload.get("musicalkeycnn_policy", "full_track")),
        parallel_workers=parallel_workers,
        max_workers_auto=max_workers_auto,
        per_track_timeout_seconds=int(analysis_payload.get("per_track_timeout_seconds", 120)),
        model_preload=bool(analysis_payload.get("model_preload", True)),
        fast_pass_enabled=bool(analysis_payload.get("fast_pass_enabled", True)),
        analysis_signature_seed=str(analysis_payload.get("analysis_signature_seed", "m1-absolute-v1")),
        energy_source_default=str(analysis_payload.get("energy_source_default", "heuristic")),
        essentia_semantics_enabled=bool(analysis_payload.get("essentia_semantics_enabled", True)),
        essentia_semantic_image=str(analysis_payload.get("essentia_semantic_image", "cuemate-essentia-semantics:local")),
        essentia_semantic_device=str(analysis_payload.get("essentia_semantic_device", "auto")),
        essentia_semantic_model_family_policy=str(analysis_payload.get("essentia_semantic_model_family_policy", "best_per_task")),
        essentia_semantic_model_root=str(analysis_payload.get("essentia_semantic_model_root", "python/models/essentia_semantics")),
        essentia_semantic_default_excerpt_seconds=float(analysis_payload.get("essentia_semantic_default_excerpt_seconds", 60.0)),
        essentia_semantic_multisample_excerpt_seconds=float(analysis_payload.get("essentia_semantic_multisample_excerpt_seconds", 30.0)),
        essentia_semantic_trigger_mismatch_threshold=float(analysis_payload.get("essentia_semantic_trigger_mismatch_threshold", 0.22)),
        essentia_semantic_trigger_confidence_threshold=float(analysis_payload.get("essentia_semantic_trigger_confidence_threshold", 0.58)),
        essentia_semantic_trigger_structure_rms_cv=float(analysis_payload.get("essentia_semantic_trigger_structure_rms_cv", 0.45)),
        essentia_semantic_trigger_outlier_zscore=float(analysis_payload.get("essentia_semantic_trigger_outlier_zscore", 1.35)),
        full_chunk_size=int(analysis_payload.get("full_chunk_size", 4)),
        tempo_chunk_size=int(analysis_payload.get("tempo_chunk_size", 8)),
        key_chunk_size=int(analysis_payload.get("key_chunk_size", 8)),
        essentia_chunk_size=int(analysis_payload.get("essentia_chunk_size", 4)),
        dsp_workers=int(analysis_payload.get("dsp_workers", auto_dsp_workers)),
    )
    thresholds = ThresholdSettings(
        small_playlist_limit=int(thresholds_payload.get("small_playlist_limit", 12)),
        min_playlist_for_relative=int(thresholds_payload.get("min_playlist_for_relative", 5)),
    )
    scoring = ScoringSettings(
        static_weights={
            key: float(value)
            for key, value in scoring_payload.get("static_weights", DEFAULT_STATIC_WEIGHTS).items()
        },
        weight_floors={
            key: float(value)
            for key, value in scoring_payload.get("weight_floors", DEFAULT_WEIGHT_FLOORS).items()
        },
        harmonic_confidence_floor=float(scoring_payload.get("harmonic_confidence_floor", 0.15)),
        key_confidence_threshold=float(scoring_payload.get("key_confidence_threshold", 0.5)),
        thresholds={**DEFAULT_SCORING_THRESHOLDS, **scoring_payload.get("thresholds", {})},
        move_types={**DEFAULT_SCORING_MOVE_TYPES, **scoring_payload.get("move_types", {})},
        penalties={**DEFAULT_SCORING_PENALTIES, **scoring_payload.get("penalties", {})},
        contrast_threshold=float(scoring_payload.get("contrast_threshold", 0.45)),
        secondary_contrast_threshold=float(scoring_payload.get("secondary_contrast_threshold", 0.65)),
        max_per_lane=int(scoring_payload.get("max_per_lane", 3)),
    )
    weight_adaptation = WeightAdaptationSettings(
        mode=str(weight_adaptation_payload.get("mode", "auto")),
        adaptation_strength=float(weight_adaptation_payload.get("adaptation_strength", 0.7)),
    )
    feedback = FeedbackSettings(
        learning_rate=float(feedback_payload.get("learning_rate", 0.35)),
        max_component_shift=float(feedback_payload.get("max_component_shift", 0.25)),
        min_contributory_events=int(feedback_payload.get("min_contributory_events", 20)),
        min_pairwise_comparisons=int(feedback_payload.get("min_pairwise_comparisons", 40)),
        min_new_events_since_last_tune=int(feedback_payload.get("min_new_events_since_last_tune", 5)),
    )

    config_signature = str(config_payload.get("config_signature", "default"))
    database_url = env_values.get("DATABASE_URL", "sqlite:data/cuemate.db")
    database_path = resolve_database_path(database_url, root)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    semantic_calibration = _load_semantic_calibration(root)

    return RuntimeSettings(
        repo_root=root,
        env_path=env_path if env_path.is_file() else env_example_path,
        config_path=config_path,
        database_path=database_path,
        database_url=database_url,
        config_signature=config_signature,
        analysis_signature=_analysis_signature(
            config_signature,
            analysis_settings,
            repo_root=root,
            semantic_calibration=semantic_calibration,
        ),
        analysis=analysis_settings,
        thresholds=thresholds,
        scoring=scoring,
        weight_adaptation=weight_adaptation,
        feedback=feedback,
        semantic_calibration=semantic_calibration,
    )


def build_relative_experiment_signature(settings: RuntimeSettings, *, energy_source: str = "canonical") -> str:
    return _relative_signature(
        settings.config_signature,
        settings.thresholds,
        settings.scoring,
        settings.weight_adaptation,
        energy_source,
    )
    

def build_fast_analysis_signature(settings: RuntimeSettings) -> str:
    return _fast_analysis_signature(
        settings.config_signature,
        settings.analysis,
    )


def build_scoring_config(settings: RuntimeSettings, *, target: str = "maintain") -> dict[str, Any]:
    """Merge RuntimeSettings scoring section into the flat dict shape consumed by scoring functions.

    The `target` parameter (maintain/build/reset/jump) is request-time state and
    is NOT persisted in config — it's injected per recommendation call.
    """
    s = settings.scoring
    return {
        "target": target,
        "static_weights": s.static_weights,
        "weight_floors": s.weight_floors,
        "harmonic_confidence_floor": s.harmonic_confidence_floor,
        "key_confidence_threshold": s.key_confidence_threshold,
        "thresholds": s.thresholds,
        "move_types": s.move_types,
        "penalties": s.penalties,
        "contrast_threshold": s.contrast_threshold,
        "secondary_contrast_threshold": s.secondary_contrast_threshold,
        "max_per_lane": s.max_per_lane,
    }
