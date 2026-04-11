from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from cuemate_analysis.cli import main
from cuemate_analysis.config import build_relative_experiment_signature
from cuemate_analysis.config import load_runtime_settings
from cuemate_analysis.database import Database
from cuemate_analysis.feedback import build_feedback_summary, compute_feedback_tuning
from cuemate_analysis.feedback_shared import (
    build_feedback_weight_layers,
    canonicalize_event_items,
    decode_json_array,
    decode_json_object,
)


def _seed_feedback_db(tmp_path: Path) -> Database:
    repo_root = Path(__file__).resolve().parents[2]
    schema_sql = (repo_root / "db" / "schema.sql").read_text(encoding="utf-8")
    db_path = tmp_path / "feedback.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(schema_sql)
        connection.executescript(
            """
            INSERT INTO playlists (id, name, track_count, created_at, updated_at)
            VALUES ('pl_1', 'Feedback Playlist', 2, '2026-04-10T00:00:00Z', '2026-04-10T00:00:00Z');

            INSERT INTO tracks (id, file_path, file_hash, title, artist, imported_at, updated_at)
            VALUES
              ('trk_current', '/music/current.flac', 'hash_current', 'Current', 'Tester', '2026-04-10T00:00:00Z', '2026-04-10T00:00:00Z'),
              ('trk_a', '/music/a.flac', 'hash_a', 'A', 'Tester', '2026-04-10T00:00:00Z', '2026-04-10T00:00:00Z'),
              ('trk_b', '/music/b.flac', 'hash_b', 'B', 'Tester', '2026-04-10T00:00:00Z', '2026-04-10T00:00:00Z'),
              ('trk_outside', '/music/out.flac', 'hash_out', 'Outside', 'Tester', '2026-04-10T00:00:00Z', '2026-04-10T00:00:00Z');

            INSERT INTO playlist_stats (
              playlist_id, track_count_total, track_count_analyzed, eligible_track_count,
              energy_spread, bass_spread, drums_spread, vocals_spread, harmonic_spread, groove_spread,
              avg_harmonic, key_diversity, bpm_range, adapted_weights, adaptation_strength,
              weight_adaptation_notes, feedback_tuned_weights, feedback_tuning_notes, feedback_event_count,
              feedback_last_tuned_at, feedback_tuning_metrics, status, energy_source_used,
              relative_signature, refreshed_at, is_stale, stale_reason, stale_marked_at
            ) VALUES (
              'pl_1', 2, 2, 2,
              0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
              0.5, 0.3, 4.0, '{"harmonic":0.12,"target_energy":0.22,"bass_transition":0.15,"tempo":0.1,"history_fit":0.06,"transition_support":0.18,"vocal_transition":0.13,"rhythmic_continuity":0.04}', 0.7,
              '[]', NULL, '[]', 0, NULL, '{}', 'ok', 'canonical',
              'rel_sig_current', '2026-04-10T00:00:00Z', 0, NULL, NULL
            );
            """
        )
        connection.commit()
    finally:
        connection.close()
    return Database(db_path)


def test_build_feedback_summary_counts_out_of_set_choices_in_total_metrics(tmp_path: Path):
    settings = load_runtime_settings()
    with _seed_feedback_db(tmp_path) as database:
        database.connection.executescript(
            """
            INSERT INTO recommendation_events (
              id, playlist_id, current_track_id, target, candidate_count, recommendation_confidence,
              recommendations_status, lanes_returned, track_chosen, chosen_was_recommended,
              skipped_over, adapted_weights, scoring_contract_id, timestamp
            ) VALUES
              (
                'evt_1', 'pl_1', 'trk_current', 'maintain', 2, 0.8,
                'available', '{}', 'trk_a', 1, '[]', '{"harmonic":0.12}', 'm3-v1', '2026-04-10T01:00:00Z'
              ),
              (
                'evt_2', 'pl_1', 'trk_current', 'maintain', 2, 0.7,
                'available', '{}', 'trk_outside', 0, '[]', '{"harmonic":0.12}', 'm3-v1', '2026-04-10T02:00:00Z'
              );

            INSERT INTO recommendation_event_items (
              event_id, lane_id, lane_rank, candidate_track_id, final_score, raw_score, penalty_multiplier,
              move, move_confidence, risk, risk_score, primary_lane, secondary_lane,
              component_scores_json, confidences_json, weights_used_json, transition_features_json
            ) VALUES
              ('evt_1', 'maintain', 1, 'trk_a', 0.80, 0.80, 1.0, 'maintain', 0.9, 'low', 0.1, 'maintain', 0, '{"harmonic":0.9}', '{"harmonic":1.0}', '{"harmonic":0.12}', '{"effective_bpm_distance":1.0}'),
              ('evt_1', 'build', 1, 'trk_b', 0.90, 0.90, 1.0, 'build', 0.9, 'low', 0.1, 'build', 0, '{"harmonic":0.3}', '{"harmonic":1.0}', '{"harmonic":0.12}', '{"effective_bpm_distance":1.0}'),
              ('evt_2', 'maintain', 1, 'trk_a', 0.80, 0.80, 1.0, 'maintain', 0.9, 'low', 0.1, 'maintain', 0, '{"harmonic":0.9}', '{"harmonic":1.0}', '{"harmonic":0.12}', '{"effective_bpm_distance":1.0}'),
              ('evt_2', 'build', 1, 'trk_b', 0.90, 0.90, 1.0, 'build', 0.9, 'low', 0.1, 'build', 0, '{"harmonic":0.3}', '{"harmonic":1.0}', '{"harmonic":0.12}', '{"effective_bpm_distance":1.0}');
            """
        )
        database.connection.commit()
        summary = build_feedback_summary(
            database,
            settings,
            playlist_id="pl_1",
            playlist_name="Feedback Playlist",
        )

    assert summary["metrics"]["total_events"] == 2
    assert summary["metrics"]["contributory_events"] == 1
    assert summary["metrics"]["chosen_top1_rate"] == 0.0
    assert summary["metrics"]["chosen_top3_rate"] == 0.5


def test_compute_feedback_tuning_increases_positive_signal_component_and_normalizes():
    settings = load_runtime_settings()
    summary = {
        "weights": {
            "base": {
                "target_energy": 0.22,
                "transition_support": 0.18,
                "bass_transition": 0.15,
                "vocal_transition": 0.13,
                "harmonic": 0.12,
                "tempo": 0.10,
                "history_fit": 0.06,
                "rhythmic_continuity": 0.04,
            },
            "static": settings.scoring.static_weights,
        },
        "metrics": {
            "total_events": 20,
            "contributory_events": 20,
            "pairwise_comparison_count": 40,
        },
        "tuning": {
            "last_tuned_at": None,
        },
        "events": [
            {
                "timestamp": f"2026-04-10T00:{idx:02d}:00Z",
                "track_chosen": "trk_a",
                "chosen_was_recommended": True,
                "canonical_items": [
                    {
                        "candidate_track_id": "trk_b",
                        "final_score": 0.9,
                        "component_scores": {"harmonic": 0.2, "bass_transition": 0.8},
                    },
                    {
                        "candidate_track_id": "trk_a",
                        "final_score": 0.8,
                        "component_scores": {"harmonic": 1.0, "bass_transition": 0.2},
                    },
                ],
            }
            for idx in range(20)
        ],
    }
    result = compute_feedback_tuning(summary, settings, force=True)

    assert result["should_apply"] is True
    assert result["tuned_weights"]["harmonic"] > result["base_weights"]["harmonic"]
    assert result["tuned_weights"]["bass_transition"] < result["base_weights"]["bass_transition"]
    assert 0.99 <= sum(result["tuned_weights"].values()) <= 1.01


def test_compute_feedback_tuning_skips_apply_when_thresholds_not_met():
    settings = load_runtime_settings()
    summary = {
        "weights": {
            "base": settings.scoring.static_weights,
            "static": settings.scoring.static_weights,
        },
        "metrics": {
            "total_events": 1,
            "contributory_events": 1,
            "pairwise_comparison_count": 1,
        },
        "tuning": {
            "last_tuned_at": None,
        },
        "events": [],
    }
    result = compute_feedback_tuning(summary, settings, force=False)

    assert result["should_apply"] is False
    assert result["thresholds_met"] is False


def test_feedback_shared_helpers_decode_and_canonicalize():
    assert decode_json_object('{"harmonic": 0.8}') == {"harmonic": 0.8}
    assert decode_json_array('["build", "jump"]') == ["build", "jump"]
    canonical = canonicalize_event_items(
        [
            {"candidate_track_id": "trk_b", "final_score": 0.6},
            {"candidate_track_id": "trk_a", "final_score": 0.7},
            {"candidate_track_id": "trk_a", "final_score": 0.8},
        ]
    )
    assert [item["candidate_track_id"] for item in canonical] == ["trk_a", "trk_b"]
    assert canonical[0]["final_score"] == 0.8


def test_cli_feedback_summary_and_inspect_weights_report_weight_layers(tmp_path: Path, monkeypatch, capsys):
    settings = replace(load_runtime_settings(), database_path=tmp_path / "feedback.db")
    expected_relative_signature = build_relative_experiment_signature(settings, energy_source="canonical")

    with _seed_feedback_db(tmp_path) as database:
        database.connection.execute(
            """
            UPDATE playlist_stats
            SET relative_signature = ?,
                feedback_tuned_weights = ?,
                feedback_tuning_notes = ?,
                feedback_event_count = ?,
                feedback_last_tuned_at = ?,
                feedback_tuning_metrics = ?
            WHERE playlist_id = 'pl_1'
            """,
            (
                expected_relative_signature,
                json.dumps({
                    "harmonic": 0.20,
                    "target_energy": 0.18,
                    "bass_transition": 0.14,
                    "tempo": 0.08,
                    "history_fit": 0.06,
                    "transition_support": 0.16,
                    "vocal_transition": 0.12,
                    "rhythmic_continuity": 0.06,
                }),
                json.dumps(["applied tuned weights"]),
                24,
                "2026-04-10T03:00:00Z",
                json.dumps({"pairwise_comparison_count": 44}),
            ),
        )
        database.connection.executescript(
            """
            INSERT INTO recommendation_events (
              id, playlist_id, current_track_id, target, candidate_count, recommendation_confidence,
              recommendations_status, lanes_returned, track_chosen, chosen_was_recommended,
              skipped_over, adapted_weights, scoring_contract_id, timestamp
            ) VALUES (
              'evt_cli_1', 'pl_1', 'trk_current', 'reset', 2, 0.75,
              'available', '{}', 'trk_a', 1, '[]', '{"harmonic":0.12}', 'm3-v1', '2026-04-10T01:00:00Z'
            );

            INSERT INTO recommendation_event_items (
              event_id, lane_id, lane_rank, candidate_track_id, final_score, raw_score, penalty_multiplier,
              move, move_confidence, risk, risk_score, primary_lane, secondary_lane,
              component_scores_json, confidences_json, weights_used_json, transition_features_json
            ) VALUES
              ('evt_cli_1', 'reset', 1, 'trk_a', 0.70, 0.70, 1.0, 'reset', 0.9, 'low', 0.1, 'reset', 0, '{"harmonic":0.9}', '{"harmonic":1.0}', '{"harmonic":0.12}', '{"effective_bpm_distance":1.0}'),
              ('evt_cli_1', 'build', 1, 'trk_b', 0.80, 0.80, 1.0, 'build', 0.9, 'low', 0.1, 'build', 0, '{"harmonic":0.3}', '{"harmonic":1.0}', '{"harmonic":0.12}', '{"effective_bpm_distance":1.0}');
            """
        )
        database.connection.commit()

    monkeypatch.setattr("cuemate_analysis.cli.load_runtime_settings", lambda: settings)

    assert main(["feedback-summary", "--playlist", "Feedback Playlist", "--json"]) == 0
    summary_payload = json.loads(capsys.readouterr().out)
    assert summary_payload["weights"]["source"] == "feedback_tuned_weights"
    assert set(summary_payload["weights"]) == {"source", "static", "base", "tuned", "effective"}
    assert summary_payload["metrics"]["pairwise_comparison_count"] == 1

    assert main(["inspect-scoring-weights", "--playlist", "Feedback Playlist", "--json"]) == 0
    inspect_payload = json.loads(capsys.readouterr().out)
    assert inspect_payload["weight_source"] == "feedback_tuned_weights"
    assert inspect_payload["base_weights"]["harmonic"] == 0.12
    assert inspect_payload["tuned_weights"]["harmonic"] == 0.20
    assert inspect_payload["effective_weights"]["harmonic"] == 0.20


def test_feedback_weight_layers_match_expected_precedence(tmp_path: Path):
    settings = replace(load_runtime_settings(), database_path=tmp_path / "feedback.db")
    layers = build_feedback_weight_layers(
        {
            "adapted_weights": {"harmonic": 0.12, "target_energy": 0.22},
            "feedback_tuned_weights": {"harmonic": 0.20, "target_energy": 0.18},
        },
        settings,
    )
    assert layers["source"] == "feedback_tuned_weights"
    assert layers["base"]["harmonic"] == 0.12
    assert layers["tuned"]["harmonic"] > layers["base"]["harmonic"]
    assert layers["effective"]["harmonic"] == 0.20
