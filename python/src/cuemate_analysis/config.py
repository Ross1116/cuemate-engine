from __future__ import annotations

import hashlib
import json
import os
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
    static_weights: dict[str, float]
    weight_floors: dict[str, float]


@dataclass(frozen=True)
class WeightAdaptationSettings:
    mode: str
    adaptation_strength: float


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
            candidate / "docs" / "Decision_Engine_Plan.md"
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


def _analysis_signature(config_signature: str, analysis_config: AnalysisSettings) -> str:
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
        "essentia_semantic_image": analysis_config.essentia_semantic_image,
        "essentia_semantic_device": analysis_config.essentia_semantic_device,
        "essentia_semantic_model_family_policy": analysis_config.essentia_semantic_model_family_policy,
        "essentia_semantic_model_root": analysis_config.essentia_semantic_model_root,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"m1-{digest[:12]}"


def _relative_signature(
    config_signature: str,
    thresholds: ThresholdSettings,
    scoring: ScoringSettings,
    weight_adaptation: WeightAdaptationSettings,
) -> str:
    payload = {
        "config_signature": config_signature,
        "small_playlist_limit": thresholds.small_playlist_limit,
        "min_playlist_for_relative": thresholds.min_playlist_for_relative,
        "static_weights": scoring.static_weights,
        "weight_floors": scoring.weight_floors,
        "weight_adaptation_mode": weight_adaptation.mode,
        "adaptation_strength": weight_adaptation.adaptation_strength,
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
    )
    weight_adaptation = WeightAdaptationSettings(
        mode=str(weight_adaptation_payload.get("mode", "auto")),
        adaptation_strength=float(weight_adaptation_payload.get("adaptation_strength", 0.7)),
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
        analysis_signature=_analysis_signature(config_signature, analysis_settings),
        analysis=analysis_settings,
        thresholds=thresholds,
        scoring=scoring,
        weight_adaptation=weight_adaptation,
        semantic_calibration=semantic_calibration,
    )


def build_relative_experiment_signature(settings: RuntimeSettings) -> str:
    return _relative_signature(
        settings.config_signature,
        settings.thresholds,
        settings.scoring,
        settings.weight_adaptation,
    )
