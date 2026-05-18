"""CLI handlers for feedback commands."""
from __future__ import annotations

import argparse
import json

import cuemate_analysis.cli as _cli


def handle_feedback_summary(args: argparse.Namespace) -> int:
    from cuemate_analysis.feedback import build_feedback_summary, open_database

    settings = _cli.load_runtime_settings()
    with open_database(settings) as database:
        playlist_row = database.get_playlist(args.playlist)
        if playlist_row is None:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")
        summary = build_feedback_summary(
            database,
            settings,
            playlist_id=str(playlist_row["id"]),
            playlist_name=str(playlist_row["name"]),
            since=args.since,
            until=args.until,
        )
    payload = _cli._feedback_public_payload(summary)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    metrics = payload["metrics"]
    weights = payload["weights"]
    tuning = payload["tuning"]
    print(f"\nFeedback summary for '{args.playlist}'\n")
    print(f"  Total events:            {metrics['total_events']}")
    print(f"  Contributory events:     {metrics['contributory_events']}")
    print(f"  Pairwise comparisons:    {metrics['pairwise_comparison_count']}")
    print(f"  Chosen in top 1:         {metrics['chosen_top1_rate']:.2%}")
    print(f"  Chosen in top 3:         {metrics['chosen_top3_rate']:.2%}")
    print(f"  Chosen in top 5:         {metrics['chosen_top5_rate']:.2%}")
    print(f"  Mean chosen rank:        {metrics['mean_chosen_rank'] if metrics['mean_chosen_rank'] is not None else 'n/a'}")
    print(f"  Active weight source:    {weights['source']}")
    print(f"  Last tuned at:           {tuning['last_tuned_at'] or 'never'}")
    print(f"  Tuned event count:       {tuning['feedback_event_count']}")

    print("\n  Lane acceptance counts:")
    lane_counts = metrics["lane_acceptance_counts"] or {}
    if not lane_counts:
        print("    none")
    for lane_name, count in lane_counts.items():
        print(f"    {lane_name:<12} {count}")

    print("\n  Higher-scored lane skips:")
    skip_counts = metrics["higher_scored_lane_skip_counts"] or {}
    if not skip_counts:
        print("    none")
    for lane_name, count in skip_counts.items():
        print(f"    {lane_name:<12} {count}")

    for label in ("static", "base", "tuned", "effective"):
        values = weights.get(label)
        print(f"\n  {label.title()} weights:")
        if not values:
            print("    none")
            continue
        for key, value in sorted(values.items()):
            print(f"    {key:<24} {value:.4f}")
    return 0


def handle_feedback_tune(args: argparse.Namespace) -> int:
    from cuemate_analysis.analysis import utc_now
    from cuemate_analysis.feedback import (
        apply_feedback_tuning,
        build_feedback_summary,
        compute_feedback_tuning,
        open_database,
    )

    settings = _cli.load_runtime_settings()
    applied_at = utc_now()
    with open_database(settings) as database:
        playlist_row = database.get_playlist(args.playlist)
        if playlist_row is None:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")
        playlist_id = str(playlist_row["id"])
        summary = build_feedback_summary(
            database,
            settings,
            playlist_id=playlist_id,
            playlist_name=str(playlist_row["name"]),
        )
        tuning_result = compute_feedback_tuning(summary, settings, force=args.force)
        if tuning_result["should_apply"] and not args.preview_only:
            apply_feedback_tuning(
                database,
                playlist_id=playlist_id,
                tuning_result=tuning_result,
                applied_at=applied_at,
            )
            summary["tuning"]["last_tuned_at"] = applied_at
            summary["tuning"]["feedback_event_count"] = tuning_result["feedback_event_count"]
            summary["tuning"]["notes"] = list(tuning_result["notes"])
            summary["tuning"]["metrics"] = dict(tuning_result["metrics"])
            summary["weights"]["tuned"] = dict(tuning_result["tuned_weights"])
            summary["weights"]["effective"] = dict(tuning_result["tuned_weights"])
            summary["weights"]["source"] = "feedback_tuned_weights"
    payload = {
        "summary": _cli._feedback_public_payload(summary),
        "tuning_result": {
            "should_apply": bool(tuning_result["should_apply"]),
            "thresholds_met": bool(tuning_result["thresholds_met"]),
            "force": bool(tuning_result["force"]),
            "preview_only": bool(args.preview_only),
            "feedback_event_count": int(tuning_result["feedback_event_count"]),
            "pairwise_comparison_count": int(tuning_result["pairwise_comparison_count"]),
            "new_contributory_events": int(tuning_result["new_contributory_events"]),
            "notes": list(tuning_result["notes"]),
            "base_weights": dict(tuning_result["base_weights"]),
            "tuned_weights": dict(tuning_result["tuned_weights"]),
            "metrics": dict(tuning_result["metrics"]),
        },
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    action = "Previewed" if args.preview_only else ("Applied" if tuning_result["should_apply"] else "Skipped")
    print(f"\n{action} feedback tuning for '{args.playlist}'\n")
    print(f"  Thresholds met:         {tuning_result['thresholds_met']}")
    print(f"  Pairwise comparisons:   {tuning_result['pairwise_comparison_count']}")
    print(f"  Contributory events:    {tuning_result['feedback_event_count']}")
    print(f"  New contributory events:{tuning_result['new_contributory_events']}")
    print("\n  Notes:")
    for note in tuning_result["notes"]:
        print(f"    {note}")
    print("\n  Tuned weights:")
    for key, value in sorted(tuning_result["tuned_weights"].items()):
        print(f"    {key:<24} {value:.4f}")
    return 0


def handle_run_feedback_worker(args: argparse.Namespace) -> int:
    from cuemate_analysis.analysis import utc_now
    from cuemate_analysis.feedback import open_database, run_feedback_worker

    settings = _cli.load_runtime_settings()
    with open_database(settings) as database:
        results = run_feedback_worker(
            database,
            settings,
            limit=max(1, int(args.limit)),
            started_at=utc_now(),
        )

    print(f"Processed {len(results)} feedback tuning job(s).")
    exit_code = 0
    for result in results:
        if result.get("error"):
            exit_code = 1
            print(f"- job {result['job_id']} playlist={result['playlist_id']} error={result['error']}")
            continue
        print(
            f"- job {result['job_id']} playlist={result['playlist_name']} "
            f"applied={result['applied']} events={result['feedback_event_count']} "
            f"pairwise={result['pairwise_comparison_count']}"
        )
    return exit_code
