from __future__ import annotations

from concurrent import futures
from typing import Any

import grpc

from cuemate_analysis.config import build_scoring_config, load_runtime_settings
from cuemate_analysis.explanations import (
    build_live_candidate_explanation,
    compute_set_trend,
    generate_session_notes,
)
from cuemate_analysis.feedback import build_feedback_summary_from_payload
from cuemate_analysis.scoring import (
    SUPPORTED_LANE_GROUPS,
    ScoringTrackContext,
    check_analysis_compatibility,
    compute_ranking_strength,
    get_recommendations,
    get_scoring_metadata,
    resolve_effective_weights,
    resolve_weight_source,
    score_candidate,
)
from cuemate_analysis.scoring_proto import load_scoring_proto_modules


DEFAULT_SCORING_SERVICE_HOST = "127.0.0.1"
DEFAULT_SCORING_SERVICE_PORT = 47834
_VALID_TARGETS = {"maintain", "build", "reset", "jump", "contrast"}
_PUBLIC_LANE_ORDER = ["maintain", "build", "reset", "jump", "contrast"]


def _lane_display(lane_id: str) -> dict[str, str]:
    for lane in SUPPORTED_LANE_GROUPS:
        if lane["lane_id"] == lane_id:
            return lane
    return {
        "lane_id": lane_id,
        "display_name": lane_id.title(),
        "summary": "",
    }


def _optional_float(message: Any, field_name: str) -> float | None:
    return float(getattr(message, field_name)) if message.HasField(field_name) else None


def _optional_int(message: Any, field_name: str) -> int | None:
    return int(getattr(message, field_name)) if message.HasField(field_name) else None


def _get_track_signatures(message: Any) -> tuple[str | None, str | None, str | None]:
    if not message.HasField("signatures"):
        return None, None, None
    sig = message.signatures
    return (
        sig.analysis_signature or None,
        sig.config_signature or None,
        sig.scoring_contract_id or None,
    )


def _track_from_proto(message: Any) -> ScoringTrackContext:
    return ScoringTrackContext(
        track_id=message.track_id,
        bpm=float(message.bpm),
        key=message.musical_key or None,
        key_confidence=_optional_float(message, "key_confidence"),
        key_source=message.key_source or None,
        key_agreement=_optional_int(message, "key_agreement"),
        energy_rel=_optional_float(message, "energy_rel"),
        bass_rel=_optional_float(message, "bass_rel"),
        drums_rel=_optional_float(message, "drums_rel"),
        vocals_rel=_optional_float(message, "vocals_rel"),
        groove_rel=_optional_float(message, "groove_rel"),
        intensity_band=message.intensity_band or None,
        role_hints=list(message.role_hints),
        title=message.title or None,
        artist=message.artist or None,
    )


def _history_from_proto(items: Any) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for item in items:
        history.append(
            {
                "track_id": item.track_id,
                "id": item.track_id,
                "key": item.musical_key or None,
                "energy_rel": _optional_float(item, "energy_rel"),
                "relation": item.relation or None,
                "plays_ago": int(item.plays_ago) if item.HasField("plays_ago") else None,
                "elapsed_since_play_seconds": (
                    float(item.elapsed_since_play_seconds)
                    if item.HasField("elapsed_since_play_seconds")
                    else None
                ),
            }
        )
    return history


def _playlist_stats_from_proto(message: Any) -> dict[str, Any] | None:
    if message is None:
        return None
    payload: dict[str, Any] = {}
    if message.HasField("energy_spread"):
        payload["energy_spread"] = float(message.energy_spread)
    if message.adapted_weights:
        payload["adapted_weights"] = dict(message.adapted_weights)
    weight_source = _weight_source_from_proto(message)
    if weight_source:
        payload["weight_source"] = weight_source
    return payload or None


def _weight_source_from_proto(message: Any) -> str:
    enum_value = int(getattr(message, "weight_source_enum", 0) or 0)
    if enum_value == 3:
        return "feedback_tuned_weights"
    if enum_value == 2:
        return "adapted_weights"
    if enum_value == 1:
        return "static"
    return ""


def _active_signatures_payload(settings: Any) -> dict[str, str]:
    metadata = get_scoring_metadata(settings)
    return dict(metadata["active_signatures"])


def _aggregate_compatibility(track_messages: list[Any], settings: Any) -> dict[str, Any]:
    exact_match = True
    compatible = True
    requires_reanalysis = False
    notes: list[str] = []
    reason = "exact_match"

    for track_message in track_messages:
        if not track_message.track_id:
            raise ValueError("track_id is required for all scoring tracks.")
        analysis_sig, config_sig, scoring_contract_id = _get_track_signatures(track_message)
        status = check_analysis_compatibility(
            analysis_sig,
            config_sig,
            scoring_contract_id,
            settings=settings,
        )
        if not status["compatible"]:
            return status
        exact_match = exact_match and bool(status["exact_match"])
        compatible = compatible and bool(status["compatible"])
        requires_reanalysis = requires_reanalysis or bool(status["requires_reanalysis"])
        if status["reason"] != "exact_match" and reason == "exact_match":
            reason = status["reason"]
        notes.extend(status.get("notes", []))

    return {
        "exact_match": exact_match,
        "compatible": compatible,
        "requires_reanalysis": requires_reanalysis,
        "reason": reason,
        "notes": notes,
    }


def _set_signature_fields(target: Any, payload: dict[str, Any]) -> None:
    target.analysis_signature = str(payload.get("analysis_signature", "") or "")
    target.config_signature = str(payload.get("config_signature", "") or "")
    target.scoring_contract_id = str(payload.get("scoring_contract_id", "") or "")


def _set_track_message(
    target: Any,
    candidate: ScoringTrackContext,
    *,
    source_message: Any | None = None,
) -> None:
    target.track_id = candidate.track_id
    target.bpm = float(candidate.bpm)
    if candidate.key is not None:
        target.musical_key = candidate.key
    if candidate.key_confidence is not None:
        target.key_confidence = float(candidate.key_confidence)
    if candidate.key_source is not None:
        target.key_source = candidate.key_source
    if candidate.key_agreement is not None:
        target.key_agreement = int(candidate.key_agreement)
    if candidate.energy_rel is not None:
        target.energy_rel = float(candidate.energy_rel)
    if candidate.bass_rel is not None:
        target.bass_rel = float(candidate.bass_rel)
    if candidate.drums_rel is not None:
        target.drums_rel = float(candidate.drums_rel)
    if candidate.vocals_rel is not None:
        target.vocals_rel = float(candidate.vocals_rel)
    if candidate.groove_rel is not None:
        target.groove_rel = float(candidate.groove_rel)
    if candidate.intensity_band is not None:
        target.intensity_band = candidate.intensity_band
    if candidate.role_hints:
        target.role_hints.extend(candidate.role_hints)
    if candidate.title is not None:
        target.title = candidate.title
    if candidate.artist is not None:
        target.artist = candidate.artist
    if source_message is not None and source_message.HasField("signatures"):
        _set_signature_fields(target.signatures, _signature_payload_from_proto(source_message.signatures))


def _signature_payload_from_proto(message: Any) -> dict[str, Any]:
    return {
        "analysis_signature": message.analysis_signature,
        "config_signature": message.config_signature,
        "scoring_contract_id": message.scoring_contract_id,
    }


def _copy_map(target: Any, payload: dict[str, float | None]) -> None:
    for key, value in payload.items():
        if value is None:
            continue
        target[key] = float(value)


def _copy_int_map(target: Any, payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        if value is None:
            continue
        try:
            target[key] = int(value)
            continue
        except (TypeError, ValueError):
            pass
        try:
            target[key] = int(float(value))
        except (TypeError, ValueError):
            continue


def _struct_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        return dict(value)
    except TypeError:
        pass
    if hasattr(value, "items"):
        return {str(key): item for key, item in value.items()}
    return {}


def _track_payload(track: ScoringTrackContext) -> dict[str, Any]:
    return {
        "track_id": track.track_id,
        "bpm": track.bpm,
        "key": track.key,
        "key_confidence": track.key_confidence,
        "key_source": track.key_source,
        "key_agreement": track.key_agreement,
        "energy_rel": track.energy_rel,
        "bass_rel": track.bass_rel,
        "drums_rel": track.drums_rel,
        "vocals_rel": track.vocals_rel,
        "groove_rel": track.groove_rel,
        "intensity_band": track.intensity_band,
        "role_hints": list(track.role_hints),
        "title": track.title,
        "artist": track.artist,
    }


def _set_advisory(target: Any, payload: dict[str, Any] | None) -> None:
    if payload is None:
        return
    target.level = str(payload.get("level", "") or "")
    target.notes.extend(str(item) for item in payload.get("notes", []) if item)


def _set_tempo_key_summary(target: Any, payload: dict[str, Any] | None) -> None:
    if payload is None:
        return
    target.tempo_text = str(payload.get("tempo_text", "") or "")
    target.key_text = str(payload.get("key_text", "") or "")
    target.key_state = str(payload.get("key_state", "") or "")


def _set_explanation_block(target: Any, payload: dict[str, Any] | None) -> None:
    if payload is None:
        return
    target.summary.extend(str(item) for item in payload.get("summary", []) if item)
    target.why.extend(str(item) for item in payload.get("why", []) if item)
    target.watch.extend(str(item) for item in payload.get("watch", []) if item)
    _set_advisory(target.handoff, payload.get("handoff"))
    _set_tempo_key_summary(target.tempo_key, payload.get("tempo_key"))
    target.character_shift.extend(str(item) for item in payload.get("character_shift", []) if item)


def _lane_order_for_response(target_lane: str) -> list[str]:
    ordered = [lane for lane in _PUBLIC_LANE_ORDER if lane != target_lane]
    return [target_lane, *ordered] if target_lane in _PUBLIC_LANE_ORDER else list(_PUBLIC_LANE_ORDER)


def _lane_empty_reason(lane_name: str) -> str:
    return f"No viable {lane_name} options after current scoring filters."


def _set_transition_features(target: Any, payload: dict[str, Any]) -> None:
    target.effective_bpm_distance = float(payload.get("effective_bpm_distance", 0.0) or 0.0)
    target.raw_bpm_distance = float(payload.get("raw_bpm_distance", 0.0) or 0.0)
    target.bpm_relationship = str(payload.get("bpm_relationship", "") or "")
    target.key_distance = int(payload.get("key_distance", 0) or 0)
    target.key_compat_label = str(payload.get("key_compat_label", "") or "")
    target.key_confidence_current = float(payload.get("key_confidence_current", 0.0) or 0.0)
    target.key_confidence_candidate = float(payload.get("key_confidence_candidate", 0.0) or 0.0)
    target.delta_energy_rel = float(payload.get("delta_energy_rel", 0.0) or 0.0)
    target.delta_bass_rel = float(payload.get("delta_bass_rel", 0.0) or 0.0)
    if payload.get("current_vocals_rel") is not None:
        target.current_vocals_rel = float(payload["current_vocals_rel"])
    if payload.get("candidate_vocals_rel") is not None:
        target.candidate_vocals_rel = float(payload["candidate_vocals_rel"])
    target.current_outro_low_end = float(payload.get("current_outro_low_end", 0.0) or 0.0)
    target.candidate_intro_low_end = float(payload.get("candidate_intro_low_end", 0.0) or 0.0)


def _set_penalty_factor(target: Any, payload: dict[str, Any]) -> None:
    target.factor = str(payload.get("factor", "") or "")
    target.severity = float(payload.get("severity", 0.0) or 0.0)
    target.raw_penalty = float(payload.get("raw_penalty", 0.0) or 0.0)
    if payload.get("gate") is not None:
        target.gate = str(payload["gate"])


def _applied_weight_adaptation(
    playlist_stats: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    effective_weights = resolve_effective_weights(playlist_stats, config)
    source = str((playlist_stats or {}).get("weight_source") or resolve_weight_source(playlist_stats))
    if source == "feedback_tuned_weights":
        return {
            "adaptation_id": "feedback_tuned_weights",
            "component_weights": effective_weights,
            "explanation": "Playlist feedback_tuned_weights were applied.",
        }
    if source == "adapted_weights":
        return {
            "adaptation_id": "adapted_weights",
            "component_weights": effective_weights,
            "explanation": "Playlist adapted_weights were applied.",
        }
    return {
        "adaptation_id": "static_weights",
        "component_weights": effective_weights,
        "explanation": "Using static scoring weights.",
    }


def _feedback_playlist_stats_from_proto(message: Any) -> dict[str, Any]:
    return {
        "adapted_weights": dict(message.adapted_weights),
        "feedback_tuned_weights": dict(message.feedback_tuned_weights),
        "feedback_tuning_notes": list(message.feedback_tuning_notes),
        "feedback_event_count": int(message.feedback_event_count),
        "feedback_last_tuned_at": str(message.feedback_last_tuned_at or "") or None,
        "feedback_tuning_metrics": _struct_to_dict(message.feedback_tuning_metrics),
    }


def _feedback_events_from_proto(items: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for event in items:
        events.append(
            {
                "id": str(event.event_id or ""),
                "timestamp": str(event.timestamp or ""),
                "track_chosen": str(event.track_chosen or ""),
                "chosen_was_recommended": bool(event.chosen_was_recommended),
                "items": [
                    {
                        "candidate_track_id": str(item.candidate_track_id or ""),
                        "final_score": float(item.final_score),
                        "lane_id": str(item.lane_id or ""),
                        "primary_lane": str(item.primary_lane or ""),
                    }
                    for item in event.items
                ],
            }
        )
    return events


def _set_feedback_summary_response(target: Any, payload: dict[str, Any]) -> None:
    target.playlist_id = str(payload.get("playlist_id", "") or "")
    target.playlist_name = str(payload.get("playlist_name", "") or "")
    window = payload.get("window", {})
    target.window.since = str(window.get("since", "") or "")
    target.window.until = str(window.get("until", "") or "")

    metrics = payload.get("metrics", {})
    target.metrics.total_events = int(metrics.get("total_events", 0) or 0)
    target.metrics.contributory_events = int(metrics.get("contributory_events", 0) or 0)
    target.metrics.ranked_events = int(metrics.get("ranked_events", 0) or 0)
    target.metrics.pairwise_comparison_count = int(metrics.get("pairwise_comparison_count", 0) or 0)
    target.metrics.chosen_top1_rate = float(metrics.get("chosen_top1_rate", 0.0) or 0.0)
    target.metrics.chosen_top3_rate = float(metrics.get("chosen_top3_rate", 0.0) or 0.0)
    target.metrics.chosen_top5_rate = float(metrics.get("chosen_top5_rate", 0.0) or 0.0)
    if metrics.get("mean_chosen_rank") is not None:
        target.metrics.mean_chosen_rank = float(metrics["mean_chosen_rank"])
    _copy_int_map(target.metrics.lane_acceptance_counts, metrics.get("lane_acceptance_counts", {}))
    _copy_int_map(target.metrics.higher_scored_lane_skip_counts, metrics.get("higher_scored_lane_skip_counts", {}))

    weights = payload.get("weights", {})
    target.weights.source = str(weights.get("source", "") or "")
    _copy_map(target.weights.static_weights, weights.get("static", {}))
    _copy_map(target.weights.base_weights, weights.get("base", {}))
    _copy_map(target.weights.tuned_weights, weights.get("tuned") or {})
    _copy_map(target.weights.effective_weights, weights.get("effective", {}))

    tuning = payload.get("tuning", {})
    target.tuning.last_tuned_at = str(tuning.get("last_tuned_at", "") or "")
    target.tuning.feedback_event_count = int(tuning.get("feedback_event_count", 0) or 0)
    target.tuning.notes.extend(str(item) for item in tuning.get("notes", []) if item)
    if tuning.get("metrics"):
        target.tuning.metrics.update(tuning.get("metrics", {}))


def _populate_scored_candidate(
    target: Any,
    payload: dict[str, Any],
    *,
    current_track: ScoringTrackContext | None = None,
    lane_scores: list[float] | None = None,
    source_message: Any | None = None,
) -> None:
    _set_track_message(target.candidate, payload["candidate"], source_message=source_message)
    target.raw_score = float(payload.get("raw_score", 0.0) or 0.0)
    target.final_score = float(payload.get("score", 0.0) or 0.0)
    target.penalty_multiplier = float(payload.get("penalty_multiplier", 1.0) or 1.0)
    for factor in payload.get("penalty_factors", []):
        _set_penalty_factor(target.penalty_factors.add(), factor)
    target.risk = str(payload.get("risk", "") or "")
    target.risk_score = float(payload.get("risk_score", 0.0) or 0.0)
    target.risk_factors.extend(str(item) for item in payload.get("risk_factors", []) if item)
    target.move = str(payload.get("move", "") or "")
    target.move_confidence = float(payload.get("move_confidence", 0.0) or 0.0)
    target.move_note = str(payload.get("move_note", "") or "")
    target.contrast_score = float(payload.get("contrast_score", 0.0) or 0.0)
    _copy_map(target.component_scores, payload.get("component_scores", {}))
    _copy_map(target.component_confidences, payload.get("confidences", {}))
    _copy_map(target.weights_used, payload.get("weights_used", {}))
    _set_transition_features(target.transition_features, payload.get("transition_features", {}))
    target.primary_lane = str(payload.get("primary_lane", "") or "")
    target.secondary_lane = bool(payload.get("secondary_lane", False))
    if lane_scores is not None:
        target.ranking_strength = float(
            compute_ranking_strength(float(payload.get("score", 0.0) or 0.0), lane_scores)
        )
    if current_track is not None:
        explanation = build_live_candidate_explanation(
            current_track=_track_payload(current_track),
            candidate_track={**_track_payload(payload["candidate"]), "move": payload.get("move", "")},
            transition_features=payload.get("transition_features", {}),
            scores=payload,
            current_outro_window=None,
            candidate_intro_window=None,
            history_context=None,
        )
        _set_tempo_key_summary(target.tempo_key, explanation.get("tempo_key"))
        target.advisory_hints.extend(str(item) for item in explanation.get("summary", []) if item)
        target.reasons.extend(str(item) for item in explanation.get("summary", []) if item)
        target.watchouts.extend(str(item) for item in explanation.get("watch", [])[:2] if item)
        _set_explanation_block(target.explanation, explanation)


class ScoringServiceServicer:
    def __init__(self, settings: Any | None = None):
        self.settings = settings or load_runtime_settings()

    def GetRecommendations(self, request: Any, context: grpc.ServicerContext) -> Any:
        pb2, _ = load_scoring_proto_modules()
        if request.target_lane and request.target_lane not in _VALID_TARGETS:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "target_lane must be a supported lane.")
        if not request.current_track.track_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "current_track.track_id is required.")

        track_messages = [request.current_track, *list(request.candidates)]
        compatibility = _aggregate_compatibility(track_messages, self.settings)
        if not compatibility["compatible"]:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                compatibility.get("reason", "incompatible_signatures"),
            )

        target_lane = request.target_lane or "maintain"
        scoring_config = build_scoring_config(self.settings, target=target_lane)
        current_track = _track_from_proto(request.current_track)
        candidates = [_track_from_proto(item) for item in request.candidates]
        history = _history_from_proto(request.history)
        playlist_stats = _playlist_stats_from_proto(request.playlist_stats)
        results = get_recommendations(
            current_track=current_track,
            candidates=candidates,
            history=history,
            config=scoring_config,
            playlist_stats=playlist_stats,
            target=target_lane,
            max_per_lane=request.max_per_lane or None,
        )

        response = pb2.GetRecommendationsResponse()
        response.recommendations_status = "available"
        response.status_note = "Scoring available."
        lane_order = results.get("lane_order")
        if isinstance(lane_order, list) and lane_order:
            response.lane_order.extend(str(item) for item in lane_order if str(item))
        else:
            response.lane_order.extend(_lane_order_for_response(target_lane))
        lane_iteration_order = list(response.lane_order)
        for lane_name in _lane_order_for_response(target_lane):
            if lane_name not in lane_iteration_order:
                lane_iteration_order.append(lane_name)
        response.recommendation_confidence = float(results.get("recommendation_confidence", 0.0) or 0.0)
        _set_track_message(response.current_track, current_track, source_message=request.current_track)
        meta = results.get("meta", {})
        response.meta.target = str(meta.get("target", "") or "")
        response.meta.total_candidates = int(meta.get("total_candidates", 0) or 0)
        response.meta.filtered_candidates = int(meta.get("filtered_candidates", 0) or 0)
        response.meta.scored_candidates = int(meta.get("scored_candidates", 0) or 0)
        response.meta.current_track_id = str(meta.get("current_track_id", "") or "")
        response.meta.requested_lane_available = bool(meta.get("requested_lane_available", False))
        response.meta.best_alternative_lanes.extend(meta.get("best_alternative_lanes", []))
        response.meta.fallback_note = str(meta.get("fallback_note", "") or "")
        trend = compute_set_trend(history)
        response.set_context.trend.label = trend["label"]
        response.set_context.trend.direction = trend["direction"]
        response.set_context.history_length = len(history)
        response.set_context.has_gaps = any(item.get("energy_rel") is None for item in history)
        response.set_context.session_notes.extend(
            generate_session_notes(
                history_length=len(history),
                has_gaps=bool(response.set_context.has_gaps),
                last_recommendation_outcome=None,
            )
        )

        metadata = get_scoring_metadata(self.settings)
        response.capabilities.flags.update(metadata.get("capability_flags", {}))
        candidate_sources = {item.track_id: item for item in request.candidates}
        lane_payloads = results.get("lanes") or {}
        for lane_name in lane_iteration_order:
            items = lane_payloads.get(lane_name, [])
            lane_message = response.lanes.add()
            lane_meta = _lane_display(lane_name)
            lane_message.lane_group.lane_id = lane_meta["lane_id"]
            lane_message.lane_group.display_name = lane_meta["display_name"]
            lane_message.lane_group.summary = lane_meta["summary"]
            lane_message.availability = "available" if items else "empty"
            lane_message.empty_reason = "" if items else _lane_empty_reason(lane_name)
            lane_scores = [float(item.get("score", 0.0) or 0.0) for item in items]
            for item in items:
                source_message = candidate_sources.get(item["candidate"].track_id)
                _populate_scored_candidate(
                    lane_message.items.add(),
                    item,
                    current_track=current_track,
                    lane_scores=lane_scores,
                    source_message=source_message,
                )

        _set_signature_fields(response.active_signatures, _active_signatures_payload(self.settings))
        response.compatibility.exact_match = bool(compatibility["exact_match"])
        response.compatibility.compatible = bool(compatibility["compatible"])
        response.compatibility.requires_reanalysis = bool(compatibility["requires_reanalysis"])
        response.compatibility.reason = str(compatibility["reason"])
        response.compatibility.notes.extend(compatibility.get("notes", []))

        applied = _applied_weight_adaptation(playlist_stats, scoring_config)
        response.applied_weight_adaptation.adaptation_id = applied["adaptation_id"]
        response.applied_weight_adaptation.component_weights.update(applied["component_weights"])
        response.applied_weight_adaptation.explanation = applied["explanation"]
        return response

    def ScoreCandidate(self, request: Any, context: grpc.ServicerContext) -> Any:
        pb2, _ = load_scoring_proto_modules()
        if request.target_lane and request.target_lane not in _VALID_TARGETS:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "target_lane must be a supported lane.")
        if not request.current_track.track_id or not request.candidate.track_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "current_track and candidate track_id are required.")

        track_messages = [request.current_track, request.candidate]
        compatibility = _aggregate_compatibility(track_messages, self.settings)
        if not compatibility["compatible"]:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                compatibility.get("reason", "incompatible_signatures"),
            )

        target_lane = request.target_lane or "maintain"
        scoring_config = build_scoring_config(self.settings, target=target_lane)
        current_track = _track_from_proto(request.current_track)
        candidate_track = _track_from_proto(request.candidate)
        history = _history_from_proto(request.history)
        playlist_stats = _playlist_stats_from_proto(request.playlist_stats)
        result = score_candidate(
            current=current_track,
            candidate=candidate_track,
            history=history,
            config=scoring_config,
            playlist_stats=playlist_stats,
        )

        response = pb2.ScoreCandidateResponse()
        _populate_scored_candidate(
            response.scored_candidate,
            result,
            current_track=current_track,
            lane_scores=[float(result.get("score", 0.0) or 0.0)],
            source_message=request.candidate,
        )
        _set_signature_fields(response.active_signatures, _active_signatures_payload(self.settings))
        response.compatibility.exact_match = bool(compatibility["exact_match"])
        response.compatibility.compatible = bool(compatibility["compatible"])
        response.compatibility.requires_reanalysis = bool(compatibility["requires_reanalysis"])
        response.compatibility.reason = str(compatibility["reason"])
        response.compatibility.notes.extend(compatibility.get("notes", []))

        applied = _applied_weight_adaptation(playlist_stats, scoring_config)
        response.applied_weight_adaptation.adaptation_id = applied["adaptation_id"]
        response.applied_weight_adaptation.component_weights.update(applied["component_weights"])
        response.applied_weight_adaptation.explanation = applied["explanation"]
        return response

    def GetScoringMetadata(self, request: Any, context: grpc.ServicerContext) -> Any:
        pb2, _ = load_scoring_proto_modules()
        metadata = get_scoring_metadata(self.settings)
        response = pb2.GetScoringMetadataResponse()
        _set_signature_fields(response.active_signatures, metadata["active_signatures"])
        response.compatible_analysis_signatures.extend(metadata.get("compatible_analysis_signatures", []))
        response.compatible_config_signatures.extend(metadata.get("compatible_config_signatures", []))
        for item in metadata.get("components", []):
            component = response.components.add()
            component.component_id = str(item.get("component_id", "") or "")
            component.description = str(item.get("description", "") or "")
            component.weight = float(item.get("weight", 0.0) or 0.0)
            component.available = bool(item.get("available", False))
            component.active = bool(item.get("active", False))
            component.status = str(item.get("status", "") or "")
        for lane in metadata.get("supported_lane_groups", []):
            lane_group = response.supported_lane_groups.add()
            lane_group.lane_id = str(lane.get("lane_id", "") or "")
            lane_group.display_name = str(lane.get("display_name", "") or "")
            lane_group.summary = str(lane.get("summary", "") or "")
        response.capability_flags.update(metadata.get("capability_flags", {}))
        response.healthy = bool(metadata.get("healthy", False))
        response.engine_version = str(metadata.get("engine_version", "") or "")
        response.status_note = str(metadata.get("status_note", "") or "")
        response.expected_relative_signature = str(metadata.get("expected_relative_signature", "") or "")
        return response

    def GetFeedbackSummary(self, request: Any, context: grpc.ServicerContext) -> Any:
        pb2, _ = load_scoring_proto_modules()
        if not request.playlist_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "playlist_id is required.")

        summary = build_feedback_summary_from_payload(
            settings=self.settings,
            playlist_id=request.playlist_id,
            playlist_name=request.playlist_name or request.playlist_id,
            playlist_stats=_feedback_playlist_stats_from_proto(request.playlist_stats),
            events=_feedback_events_from_proto(request.events),
            since=str(request.window.since or "") or None,
            until=str(request.window.until or "") or None,
        )
        response = pb2.GetFeedbackSummaryResponse()
        _set_feedback_summary_response(response, summary)
        return response


def build_grpc_server(settings: Any | None = None, *, max_workers: int = 8) -> grpc.Server:
    _, pb2_grpc = load_scoring_proto_modules()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb2_grpc.add_ScoringServiceServicer_to_server(ScoringServiceServicer(settings), server)
    return server


def serve_scoring_grpc(
    *,
    host: str = DEFAULT_SCORING_SERVICE_HOST,
    port: int = DEFAULT_SCORING_SERVICE_PORT,
    settings: Any | None = None,
    max_workers: int = 8,
) -> int:
    server = build_grpc_server(settings, max_workers=max_workers)
    bind_address = f"{host}:{port}"
    bound_port = server.add_insecure_port(bind_address)
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind scoring gRPC service to {bind_address}")
    server.start()
    print(f"Scoring gRPC service listening on {host}:{port}")
    server.wait_for_termination()
    return 0
