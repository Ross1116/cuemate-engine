"""CLI handlers for inspection commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cuemate_analysis.cli as _cli


def handle_list_playlist(args: argparse.Namespace) -> int:
    settings = _cli.load_runtime_settings()
    with _cli.Database(settings.database_path) as database:
        rows = database.get_playlist_tracks(args.name)
        if not rows:
            raise SystemExit(f"Playlist '{args.name}' was not found.")

        print(f"Playlist '{args.name}' ({len(rows)} track(s))")
        for row in rows:
            title = row["title"] or Path(row["file_path"]).stem
            artist = row["artist"] or "Unknown artist"
            if row["analyzed_at"]:
                summary = f"{row['analysis_mode']} | {row['bpm']:.1f} BPM | {row['key']}"
            elif row["fast_analyzed_at"]:
                summary = f"fast_ready | {row['fast_bpm']:.1f} BPM | {row['fast_key']}"
            else:
                summary = "not analyzed"
            print(f"{row['position']:02d}. {artist} - {title} [{row['track_id']}] :: {summary}")
    return 0


def handle_show_track(args: argparse.Namespace) -> int:
    settings = _cli.load_runtime_settings()
    with _cli.Database(settings.database_path) as database:
        details = database.get_track_details(args.track_id)
        if details is None:
            raise SystemExit(f"Track '{args.track_id}' was not found.")
    details["analysis_state"] = (
        "full_ready" if details.get("analyzed_at")
        else "fast_ready" if details.get("fast_analyzed_at")
        else "metadata_only"
    )
    details["energy_summary"] = {
        "heuristic": details.get("energy_abs"),
        "heuristic_legacy": details.get("energy_heuristic_abs"),
        "essentia_fused": details.get("energy_essentia_fused"),
        "essentia_bucket": details.get("energy_essentia_bucket"),
    }
    details["essentia_semantics"] = {
        "danceability_abs": details.get("danceability_abs"),
        "arousal_abs": details.get("arousal_abs"),
        "valence_abs": details.get("valence_abs"),
        "mood_aggressive_abs": details.get("mood_aggressive_abs"),
        "mood_party_abs": details.get("mood_party_abs"),
        "mood_relaxed_abs": details.get("mood_relaxed_abs"),
        "energy_essentia_fused": details.get("energy_essentia_fused"),
        "energy_essentia_bucket": details.get("energy_essentia_bucket"),
        "essentia_semantic_signature": details.get("essentia_semantic_signature"),
        "essentia_semantic_source": details.get("essentia_semantic_source"),
        "essentia_semantic_inferred_at": details.get("essentia_semantic_inferred_at"),
    }
    print(json.dumps(details, indent=2, sort_keys=True))
    return 0
