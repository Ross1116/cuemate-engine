from __future__ import annotations

import json
from typing import Any

from cuemate_analysis.config import RuntimeSettings, build_scoring_config
from cuemate_analysis.scoring import resolve_effective_weights, resolve_weight_source


def decode_json_object(raw_value: Any) -> dict[str, Any]:
    if isinstance(raw_value, dict):
        return raw_value
    if not raw_value:
        return {}
    if isinstance(raw_value, str):
        try:
            decoded = json.loads(raw_value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def decode_json_array(raw_value: Any) -> list[Any]:
    if isinstance(raw_value, list):
        return raw_value
    if not raw_value:
        return []
    if isinstance(raw_value, str):
        try:
            decoded = json.loads(raw_value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return []
        return decoded if isinstance(decoded, list) else []
    return []


def normalize_weight_map(payload: dict[str, Any] | None, static_weights: dict[str, float]) -> dict[str, float]:
    weights = dict(static_weights)
    if payload:
        for key, value in payload.items():
            if key in static_weights and value is not None:
                weights[key] = float(value)
    total = sum(weights.values())
    if total > 0:
        weights = {key: float(value) / total for key, value in weights.items()}
    return weights


def canonicalize_event_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_candidate: dict[str, dict[str, Any]] = {}
    for item in items:
        candidate_track_id = str(item.get("candidate_track_id") or "")
        if not candidate_track_id:
            continue
        existing = by_candidate.get(candidate_track_id)
        final_score = float(item.get("final_score", 0.0) or 0.0)
        if existing is None or final_score > float(existing.get("final_score", 0.0) or 0.0):
            by_candidate[candidate_track_id] = dict(item)
    return sorted(
        by_candidate.values(),
        key=lambda item: (-float(item.get("final_score", 0.0) or 0.0), str(item.get("candidate_track_id") or "")),
    )


def build_feedback_weight_layers(
    playlist_stats: dict[str, Any] | None,
    settings: RuntimeSettings | Any,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scoring_config = config if config is not None else build_scoring_config(settings, target="maintain")
    static_source = dict(scoring_config["static_weights"])
    static_weights = normalize_weight_map(None, static_source)
    base_weights = normalize_weight_map(
        (playlist_stats or {}).get("adapted_weights"),
        static_source,
    )
    tuned_weights = (
        normalize_weight_map((playlist_stats or {}).get("feedback_tuned_weights"), static_source)
        if (playlist_stats or {}).get("feedback_tuned_weights")
        else None
    )
    effective_weights = resolve_effective_weights(playlist_stats, scoring_config)
    weight_source = resolve_weight_source(playlist_stats)
    return {
        "source": weight_source,
        "static": static_weights,
        "base": base_weights,
        "tuned": tuned_weights,
        "effective": {key: float(value) for key, value in sorted(effective_weights.items())},
    }
