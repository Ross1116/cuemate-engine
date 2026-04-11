from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cuemate_analysis.config import RuntimeSettings
from cuemate_analysis.database import Database
from cuemate_analysis.feedback_shared import (
    build_feedback_weight_layers,
    canonicalize_event_items,
    decode_json_array,
    decode_json_object,
)


LEARNING_RATE = 0.35
MAX_COMPONENT_SHIFT = 0.25
MIN_CONTRIBUTORY_EVENTS = 20
MIN_PAIRWISE_COMPARISONS = 40
MIN_NEW_EVENTS_SINCE_LAST_TUNE = 5


def build_feedback_summary_from_payload(
    *,
    settings: RuntimeSettings,
    playlist_id: str,
    playlist_name: str,
    playlist_stats: dict[str, Any] | None,
    events: list[dict[str, Any]],
    since: str | None = None,
    until: str | None = None,
) -> dict[str, Any]:
    normalized_events = normalize_feedback_summary_events(events)
    weight_layers = build_feedback_weight_layers(playlist_stats, settings)

    total_events = len(normalized_events)
    contributory_events = 0
    ranked_events = 0
    chosen_top1 = 0
    chosen_top3 = 0
    chosen_top5 = 0
    chosen_rank_total = 0.0
    pairwise_comparison_count = 0
    lane_acceptance_counts: dict[str, int] = defaultdict(int)
    higher_scored_lane_skips: dict[str, int] = defaultdict(int)

    for event in normalized_events:
        canonical_items = list(event["canonical_items"])
        chosen_track_id = str(event.get("track_chosen") or "")
        chosen_item = next(
            (item for item in canonical_items if str(item.get("candidate_track_id") or "") == chosen_track_id),
            None,
        )
        if chosen_item is None:
            continue
        ranked_events += 1
        chosen_rank = next(
            (
                index
                for index, item in enumerate(canonical_items, start=1)
                if str(item.get("candidate_track_id") or "") == chosen_track_id
            ),
            None,
        )
        if chosen_rank is not None:
            chosen_rank_total += float(chosen_rank)
            if chosen_rank <= 1:
                chosen_top1 += 1
            if chosen_rank <= 3:
                chosen_top3 += 1
            if chosen_rank <= 5:
                chosen_top5 += 1

        if not bool(event.get("chosen_was_recommended")):
            continue
        contributory_events += 1
        chosen_lane = str(chosen_item.get("primary_lane") or chosen_item.get("lane_id") or "unknown")
        lane_acceptance_counts[chosen_lane] += 1
        chosen_score = float(chosen_item.get("final_score", 0.0) or 0.0)
        pairwise_comparison_count += sum(
            1
            for item in canonical_items
            if float(item.get("final_score", 0.0) or 0.0) > chosen_score
        )

        best_score_by_lane: dict[str, float] = {}
        for item in event["items"]:
            lane_id = str(item.get("lane_id") or item.get("primary_lane") or "unknown")
            score = float(item.get("final_score", 0.0) or 0.0)
            best_score_by_lane[lane_id] = max(best_score_by_lane.get(lane_id, score), score)
        for lane_id, lane_score in best_score_by_lane.items():
            if lane_id != chosen_lane and lane_score > chosen_score:
                higher_scored_lane_skips[lane_id] += 1

    top_denominator = total_events if total_events > 0 else 1
    mean_chosen_rank = (chosen_rank_total / ranked_events) if ranked_events > 0 else None
    feedback_metrics = (playlist_stats or {}).get("feedback_tuning_metrics") or {}
    return {
        "playlist_id": playlist_id,
        "playlist_name": playlist_name,
        "window": {"since": since, "until": until},
        "metrics": {
            "total_events": total_events,
            "contributory_events": contributory_events,
            "ranked_events": ranked_events,
            "pairwise_comparison_count": pairwise_comparison_count,
            "chosen_top1_rate": round(chosen_top1 / top_denominator, 4),
            "chosen_top3_rate": round(chosen_top3 / top_denominator, 4),
            "chosen_top5_rate": round(chosen_top5 / top_denominator, 4),
            "mean_chosen_rank": None if mean_chosen_rank is None else round(mean_chosen_rank, 4),
            "lane_acceptance_counts": dict(sorted(lane_acceptance_counts.items())),
            "higher_scored_lane_skip_counts": dict(sorted(higher_scored_lane_skips.items())),
        },
        "weights": {
            **weight_layers,
        },
        "tuning": {
            "last_tuned_at": (playlist_stats or {}).get("feedback_last_tuned_at"),
            "feedback_event_count": int((playlist_stats or {}).get("feedback_event_count") or 0),
            "notes": list((playlist_stats or {}).get("feedback_tuning_notes") or []),
            "metrics": feedback_metrics,
        },
        "events": normalized_events,
    }


def normalize_feedback_summary_events(events: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized_events: list[dict[str, Any]] = []
    for event in events or []:
        items_payload = event.get("items")
        items: list[dict[str, Any]] = []
        if isinstance(items_payload, list):
            for item in items_payload:
                if isinstance(item, Mapping):
                    items.append(dict(item))
        canonical_payload = event.get("canonical_items")
        if isinstance(canonical_payload, list):
            canonical_items = [dict(item) for item in canonical_payload if isinstance(item, Mapping)]
        else:
            canonical_items = canonicalize_event_items(items)
        normalized_events.append(
            {
                **dict(event),
                "items": items,
                "canonical_items": canonical_items,
                "track_chosen": None if event.get("track_chosen") is None else str(event.get("track_chosen") or ""),
                "chosen_was_recommended": bool(event.get("chosen_was_recommended")),
                "timestamp": None if event.get("timestamp") is None else str(event.get("timestamp") or ""),
            }
        )
    return normalized_events


def _load_feedback_events(
    database: Database,
    playlist_id: str,
    *,
    since: str | None = None,
    until: str | None = None,
) -> list[dict[str, Any]]:
    params: list[Any] = [playlist_id]
    where = ["playlist_id = ?"]
    if since:
        where.append("timestamp >= ?")
        params.append(since)
    if until:
        where.append("timestamp <= ?")
        params.append(until)
    event_rows = database.connection.execute(
        f"""
        SELECT
          id,
          playlist_id,
          current_track_id,
          target,
          recommendations_status,
          track_chosen,
          chosen_was_recommended,
          skipped_over,
          adapted_weights,
          timestamp
        FROM recommendation_events
        WHERE {' AND '.join(where)}
        ORDER BY timestamp ASC, id ASC
        """,
        params,
    ).fetchall()
    if not event_rows:
        return []

    event_ids = [str(row["id"]) for row in event_rows]
    placeholders = ", ".join("?" for _ in event_ids)
    item_rows = database.connection.execute(
        f"""
        SELECT
          event_id,
          lane_id,
          lane_rank,
          candidate_track_id,
          final_score,
          raw_score,
          penalty_multiplier,
          move,
          move_confidence,
          risk,
          risk_score,
          primary_lane,
          secondary_lane,
          component_scores_json,
          confidences_json,
          weights_used_json,
          transition_features_json
        FROM recommendation_event_items
        WHERE event_id IN ({placeholders})
        ORDER BY event_id ASC, lane_id ASC, lane_rank ASC
        """,
        event_ids,
    ).fetchall()
    items_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in item_rows:
        items_by_event[str(row["event_id"])].append(
            {
                "event_id": str(row["event_id"]),
                "lane_id": str(row["lane_id"]),
                "lane_rank": int(row["lane_rank"]),
                "candidate_track_id": str(row["candidate_track_id"]),
                "final_score": float(row["final_score"]),
                "raw_score": float(row["raw_score"]),
                "penalty_multiplier": float(row["penalty_multiplier"]),
                "move": str(row["move"]),
                "move_confidence": float(row["move_confidence"]),
                "risk": str(row["risk"]),
                "risk_score": float(row["risk_score"]),
                "primary_lane": str(row["primary_lane"] or row["lane_id"]),
                "secondary_lane": bool(row["secondary_lane"]),
                "component_scores": decode_json_object(row["component_scores_json"]),
                "confidences": decode_json_object(row["confidences_json"]),
                "weights_used": decode_json_object(row["weights_used_json"]),
                "transition_features": decode_json_object(row["transition_features_json"]),
            }
        )

    events: list[dict[str, Any]] = []
    for row in event_rows:
        event_id = str(row["id"])
        items = items_by_event.get(event_id, [])
        events.append(
            {
                "id": event_id,
                "playlist_id": str(row["playlist_id"]),
                "current_track_id": str(row["current_track_id"]),
                "target": str(row["target"]),
                "recommendations_status": str(row["recommendations_status"]),
                "track_chosen": str(row["track_chosen"]) if row["track_chosen"] is not None else None,
                "chosen_was_recommended": bool(row["chosen_was_recommended"]) if row["chosen_was_recommended"] is not None else False,
                "skipped_over": decode_json_array(row["skipped_over"]),
                "adapted_weights": decode_json_object(row["adapted_weights"]),
                "timestamp": str(row["timestamp"]),
                "items": items,
                "canonical_items": canonicalize_event_items(items),
            }
        )
    return events


def build_feedback_summary(
    database: Database,
    settings: RuntimeSettings,
    *,
    playlist_id: str,
    playlist_name: str,
    since: str | None = None,
    until: str | None = None,
) -> dict[str, Any]:
    playlist_stats = database.get_playlist_stats_for_scoring(playlist_id)
    events = _load_feedback_events(database, playlist_id, since=since, until=until)
    return build_feedback_summary_from_payload(
        settings=settings,
        playlist_id=playlist_id,
        playlist_name=playlist_name,
        playlist_stats=playlist_stats,
        events=events,
        since=since,
        until=until,
    )


def compute_feedback_tuning(
    summary: dict[str, Any],
    settings: RuntimeSettings,
    *,
    force: bool = False,
) -> dict[str, Any]:
    feedback_cfg = getattr(settings, "feedback", None)
    learning_rate = feedback_cfg.learning_rate if feedback_cfg else LEARNING_RATE
    max_component_shift = feedback_cfg.max_component_shift if feedback_cfg else MAX_COMPONENT_SHIFT
    min_contributory_events = feedback_cfg.min_contributory_events if feedback_cfg else MIN_CONTRIBUTORY_EVENTS
    min_pairwise_comparisons = feedback_cfg.min_pairwise_comparisons if feedback_cfg else MIN_PAIRWISE_COMPARISONS
    min_new_events_since_last_tune = feedback_cfg.min_new_events_since_last_tune if feedback_cfg else MIN_NEW_EVENTS_SINCE_LAST_TUNE

    weights_payload = summary["weights"]
    base_weights = dict(weights_payload["base"])
    static_weights = dict(weights_payload["static"])
    metrics = dict(summary["metrics"])
    events = list(summary["events"])
    last_tuned_at = summary["tuning"].get("last_tuned_at")

    component_sums: dict[str, float] = defaultdict(float)
    component_counts: dict[str, int] = defaultdict(int)
    contributory_events = 0
    pairwise_count = 0
    new_contributory_events = 0

    for event in events:
        if not bool(event.get("chosen_was_recommended")):
            continue
        canonical_items = list(event.get("canonical_items") or [])
        chosen_track_id = str(event.get("track_chosen") or "")
        chosen_item = next(
            (item for item in canonical_items if str(item.get("candidate_track_id") or "") == chosen_track_id),
            None,
        )
        if chosen_item is None:
            continue
        contributory_events += 1
        if last_tuned_at and str(event.get("timestamp") or "") > str(last_tuned_at):
            new_contributory_events += 1
        chosen_score = float(chosen_item.get("final_score", 0.0) or 0.0)
        chosen_components = decode_json_object(chosen_item.get("component_scores"))
        for skipped in canonical_items:
            skipped_score = float(skipped.get("final_score", 0.0) or 0.0)
            if skipped_score <= chosen_score:
                continue
            skipped_components = decode_json_object(skipped.get("component_scores"))
            pairwise_count += 1
            for component, base_weight in static_weights.items():
                if component not in chosen_components or component not in skipped_components:
                    continue
                chosen_value = chosen_components.get(component)
                skipped_value = skipped_components.get(component)
                if chosen_value is None or skipped_value is None:
                    continue
                signal = float(chosen_value) - float(skipped_value)
                component_sums[component] += signal
                component_counts[component] += 1

    thresholds_met = (
        contributory_events >= min_contributory_events
        and pairwise_count >= min_pairwise_comparisons
        and (
            last_tuned_at is None
            or new_contributory_events >= min_new_events_since_last_tune
        )
    )
    should_apply = force or thresholds_met
    tuned_weights = dict(base_weights)
    component_signals: dict[str, float] = {}
    if should_apply and pairwise_count > 0:
        for component, base_weight in base_weights.items():
            mean_signal = component_sums[component] / component_counts[component] if component_counts[component] else 0.0
            clipped_signal = max(-1.0, min(1.0, mean_signal))
            component_signals[component] = round(clipped_signal, 6)
            proposed = base_weight * (1.0 + (learning_rate * clipped_signal))
            lower = base_weight * (1.0 - max_component_shift)
            upper = base_weight * (1.0 + max_component_shift)
            tuned_weights[component] = max(lower, min(upper, proposed))
        floors = settings.scoring.weight_floors
        for component, floor in floors.items():
            tuned_weights[component] = max(tuned_weights.get(component, 0.0), float(floor))
        total = sum(tuned_weights.values())
        if total > 0:
            tuned_weights = {component: value / total for component, value in tuned_weights.items()}
    notes = [
        f"contributory_events={contributory_events}",
        f"pairwise_comparison_count={pairwise_count}",
        f"new_contributory_events={new_contributory_events}",
    ]
    if not thresholds_met and not force:
        notes.append(
            "Thresholds not met for auto-apply "
            f"(needs {min_contributory_events} contributory events, {min_pairwise_comparisons} pairwise comparisons, "
            f"and {min_new_events_since_last_tune} new contributory events since last tune)."
        )
    if force:
        notes.append("Force mode bypassed automatic apply thresholds.")
    tuning_metrics = {
        "contributory_event_count": contributory_events,
        "pairwise_comparison_count": pairwise_count,
        "new_contributory_events": new_contributory_events,
        "learning_rate": learning_rate,
        "max_component_shift": max_component_shift,
        "component_mean_signal": component_signals,
        "applied": bool(should_apply and pairwise_count > 0),
    }
    metrics["contributory_events"] = contributory_events
    metrics["pairwise_comparison_count"] = pairwise_count
    return {
        "should_apply": bool(should_apply and pairwise_count > 0),
        "thresholds_met": thresholds_met,
        "force": force,
        "base_weights": base_weights,
        "tuned_weights": tuned_weights,
        "notes": notes,
        "feedback_event_count": contributory_events,
        "new_contributory_events": new_contributory_events,
        "pairwise_comparison_count": pairwise_count,
        "metrics": tuning_metrics,
        "summary_metrics": metrics,
    }


def apply_feedback_tuning(
    database: Database,
    *,
    playlist_id: str,
    tuning_result: dict[str, Any],
    applied_at: str,
) -> None:
    database.update_playlist_feedback_tuning(
        playlist_id=playlist_id,
        feedback_tuned_weights=tuning_result["tuned_weights"],
        feedback_tuning_notes=list(tuning_result["notes"]),
        feedback_event_count=int(tuning_result["feedback_event_count"]),
        feedback_last_tuned_at=applied_at,
        feedback_tuning_metrics=dict(tuning_result["metrics"]),
    )


def run_feedback_worker(
    database: Database,
    settings: RuntimeSettings,
    *,
    limit: int,
    started_at: str,
) -> list[dict[str, Any]]:
    claimed_jobs = database.claim_pending_feedback_tuning_jobs(limit=limit, started_at=started_at)
    results: list[dict[str, Any]] = []
    for job in claimed_jobs:
        job_id = int(job["id"])
        playlist_id = str(job["playlist_id"])
        try:
            playlist_name = database.get_playlist_name_by_id(playlist_id) or playlist_id
            summary = build_feedback_summary(
                database,
                settings,
                playlist_id=playlist_id,
                playlist_name=playlist_name,
            )
            tuning_result = compute_feedback_tuning(summary, settings, force=False)
            if tuning_result["should_apply"]:
                apply_feedback_tuning(
                    database,
                    playlist_id=playlist_id,
                    tuning_result=tuning_result,
                    applied_at=started_at,
                )
            finished_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            database.mark_feedback_tuning_job_completed(job_id, finished_at=finished_at)
            results.append(
                {
                    "job_id": job_id,
                    "playlist_id": playlist_id,
                    "playlist_name": playlist_name,
                    "applied": bool(tuning_result["should_apply"]),
                    "pairwise_comparison_count": int(tuning_result["pairwise_comparison_count"]),
                    "feedback_event_count": int(tuning_result["feedback_event_count"]),
                    "notes": list(tuning_result["notes"]),
                }
            )
        except (KeyError, ValueError, TypeError, sqlite3.Error) as exc:
            finished_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            database.mark_feedback_tuning_job_failed(job_id, error_message=str(exc), finished_at=finished_at)
            results.append(
                {
                    "job_id": job_id,
                    "playlist_id": playlist_id,
                    "applied": False,
                    "error": str(exc),
                }
            )
    return results


def open_database(settings: RuntimeSettings) -> Database:
    db_path = Path(settings.database_path)
    return Database(db_path)
