from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from cuemate_analysis.cli import main
from cuemate_analysis.scoring import ScoringTrackContext, organize_into_lanes


def _ctx(
    *,
    track_id: str,
    title: str,
    artist: str = "Test Artist",
    vocals_rel: float | None = None,
) -> ScoringTrackContext:
    return ScoringTrackContext(
        track_id=track_id,
        bpm=128.0,
        key="8A",
        key_confidence=0.9,
        key_source="musicalkeycnn",
        key_agreement=1,
        energy_rel=0.5,
        bass_rel=0.5,
        drums_rel=0.5,
        vocals_rel=vocals_rel,
        groove_rel=0.5,
        intensity_band="Drive",
        role_hints=[],
        title=title,
        artist=artist,
    )


def _lane_item(
    candidate: ScoringTrackContext,
    *,
    score: float,
    move: str,
    contrast_score: float = 0.0,
) -> dict:
    return {
        "candidate": candidate,
        "score": score,
        "raw_score": score,
        "penalty_multiplier": 1.0,
        "penalty_factors": [],
        "risk": "low",
        "risk_score": 0.1,
        "risk_factors": [],
        "move": move,
        "move_confidence": 0.8,
        "move_note": None,
        "contrast_score": contrast_score,
        "component_scores": {},
        "transition_features": {
            "effective_bpm_distance": 0.0,
            "key_compat_label": "perfect",
        },
        "confidences": {},
        "weights_used": {},
        "secondary_lane": False,
    }


class _FakeDB:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_playlist(self, name: str):
        return {"id": "plt_test", "name": name}

    def get_scoring_candidates(self, playlist_id: str):
        return [
            {"track_id": "trk_current"},
            {"track_id": "trk_reset"},
            {"track_id": "trk_jump"},
            {"track_id": "trk_candidate"},
        ]

    def get_track_scoring_context(self, track_id: str, playlist_id: str):
        return {"track_id": track_id}

    def get_playlist_stats_for_scoring(self, playlist_id: str):
        return {"is_stale": False, "relative_signature": "sig-current"}


class _FakeWeightsDB(_FakeDB):
    def get_playlist_stats_for_scoring(self, playlist_id: str):
        return {
            "adapted_weights": {
                "target_energy": 0.31,
                "harmonic": 0.08,
            }
        }


def test_organize_into_lanes_promotes_best_lane_when_target_missing():
    reset_item = _lane_item(_ctx(track_id="trk_reset", title="Reset"), score=0.82, move="reset")
    jump_item = _lane_item(_ctx(track_id="trk_jump", title="Jump"), score=0.64, move="jump")
    out = organize_into_lanes(
        [jump_item, reset_item],
        target_lane="maintain",
        config={
            "max_per_lane": 3,
            "contrast_threshold": 0.45,
            "secondary_contrast_threshold": 0.65,
        },
    )
    assert out["lane_order"][:2] == ["reset", "jump"]


def test_recommend_next_prints_fallback_note_when_target_lane_is_empty(monkeypatch, capsys):
    current = _ctx(track_id="trk_current", title="Current")
    reset_candidate = _ctx(track_id="trk_reset", title="Reset Pick")
    jump_candidate = _ctx(track_id="trk_jump", title="Jump Pick")
    contexts = {
        "trk_current": current,
        "trk_reset": reset_candidate,
        "trk_jump": jump_candidate,
        "trk_candidate": _ctx(track_id="trk_candidate", title="Unused"),
    }

    monkeypatch.setattr(
        "cuemate_analysis.cli.load_runtime_settings",
        lambda: SimpleNamespace(database_path=Path("fake.db")),
    )
    monkeypatch.setattr(
        "cuemate_analysis.config.build_relative_experiment_signature",
        lambda settings, energy_source="canonical": "sig-current",
    )
    monkeypatch.setattr("cuemate_analysis.cli.Database", lambda _: _FakeDB())
    monkeypatch.setattr("cuemate_analysis.config.build_scoring_config", lambda settings, target: {"target": target})
    monkeypatch.setattr(
        "cuemate_analysis.scoring.row_to_scoring_track_context",
        lambda row: contexts[row["track_id"]],
    )
    monkeypatch.setattr(
        "cuemate_analysis.scoring.get_recommendations",
        lambda *args, **kwargs: {
            "lane_order": ["reset", "jump"],
            "lanes": {
                "reset": [_lane_item(reset_candidate, score=0.82, move="reset")],
                "jump": [_lane_item(jump_candidate, score=0.64, move="jump")],
            },
            "target_lane": "maintain",
            "recommendation_confidence": 0.58,
            "meta": {
                "scored_candidates": 2,
                "fallback_note": "No strong maintain candidates found; best alternatives are reset and jump.",
            },
        },
    )

    assert main(
        [
            "recommend-next",
            "--playlist",
            "Test Playlist",
            "--current-track",
            "trk_current",
            "--target",
            "maintain",
        ]
    ) == 0
    out = capsys.readouterr().out
    assert "No strong maintain candidates found; best alternatives are reset and jump." in out
    assert out.index("[RESET]") < out.index("[JUMP]")


def test_score_pair_prints_stub_and_missing_vocal_notes(monkeypatch, capsys):
    current = _ctx(track_id="trk_current", title="Current", vocals_rel=None)
    candidate = _ctx(track_id="trk_candidate", title="Candidate", vocals_rel=None)
    contexts = {
        "trk_current": current,
        "trk_reset": _ctx(track_id="trk_reset", title="Reset Pick"),
        "trk_jump": _ctx(track_id="trk_jump", title="Jump Pick"),
        "trk_candidate": candidate,
    }

    monkeypatch.setattr(
        "cuemate_analysis.cli.load_runtime_settings",
        lambda: SimpleNamespace(database_path=Path("fake.db")),
    )
    monkeypatch.setattr(
        "cuemate_analysis.config.build_relative_experiment_signature",
        lambda settings, energy_source="canonical": "sig-current",
    )
    monkeypatch.setattr("cuemate_analysis.cli.Database", lambda _: _FakeDB())
    monkeypatch.setattr("cuemate_analysis.config.build_scoring_config", lambda settings, target: {"target": target})
    monkeypatch.setattr(
        "cuemate_analysis.scoring.row_to_scoring_track_context",
        lambda row: contexts[row["track_id"]],
    )
    monkeypatch.setattr(
        "cuemate_analysis.scoring.score_candidate",
        lambda **kwargs: {
            "candidate": candidate,
            "raw_score": 0.74,
            "score": 0.74,
            "penalty_multiplier": 1.0,
            "penalty_factors": [],
            "risk": "medium",
            "risk_score": 0.25,
            "risk_factors": [],
            "move": "reset",
            "move_confidence": 0.9,
            "move_note": None,
            "contrast_score": 0.31,
            "component_scores": {
                "target_energy": 0.85,
                "transition_support": None,
                "bass_transition": 0.72,
                "vocal_transition": None,
                "harmonic": 0.5,
                "tempo": 1.0,
                "history_fit": 1.0,
                "rhythmic_continuity": None,
            },
            "transition_features": {
                "current_vocals_rel": None,
                "candidate_vocals_rel": None,
            },
            "confidences": {
                "target_energy": 1.0,
                "transition_support": 1.0,
                "bass_transition": 1.0,
                "vocal_transition": 1.0,
                "harmonic": 1.0,
                "tempo": 1.0,
                "history_fit": 1.0,
                "rhythmic_continuity": 1.0,
            },
            "weights_used": {
                "target_energy": 0.22,
                "transition_support": 0.18,
                "bass_transition": 0.15,
                "vocal_transition": 0.13,
                "harmonic": 0.12,
                "tempo": 0.10,
                "history_fit": 0.06,
                "rhythmic_continuity": 0.04,
            },
        },
    )

    assert main(
        [
            "score-pair",
            "--playlist",
            "Test Playlist",
            "--current",
            "trk_current",
            "--candidate",
            "trk_candidate",
            "--target",
            "reset",
        ]
    ) == 0
    out = capsys.readouterr().out
    assert "Score pair: Test Artist - Current [trk_current]  ->  Test Artist - Candidate [trk_candidate]" in out
    assert "Stubbed and excluded from weighted scoring" in out
    assert "vocals_abs / vocals_rel are not populated yet" in out


def test_inspect_scoring_metadata_json(monkeypatch, capsys):
    monkeypatch.setattr(
        "cuemate_analysis.cli.load_runtime_settings",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "cuemate_analysis.scoring.get_scoring_metadata",
        lambda settings=None: {
            "active_signatures": {
                "analysis_signature": "m1-test",
                "config_signature": "default",
                "scoring_contract_id": "m3-v1",
            },
            "compatible_analysis_signatures": ["m1-test"],
            "compatible_config_signatures": ["default"],
            "components": [],
            "supported_lane_groups": [],
            "capability_flags": {"vocals_available": False},
            "healthy": True,
            "engine_version": "0.1.0",
            "status_note": "ok",
        },
    )

    assert main(["inspect-scoring-metadata", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["active_signatures"]["scoring_contract_id"] == "m3-v1"
    assert payload["metadata"]["capability_flags"]["vocals_available"] is False


def test_inspect_scoring_metadata_text_with_compatibility(monkeypatch, capsys):
    monkeypatch.setattr(
        "cuemate_analysis.cli.load_runtime_settings",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "cuemate_analysis.scoring.get_scoring_metadata",
        lambda settings=None: {
            "active_signatures": {
                "analysis_signature": "m1-test",
                "config_signature": "default",
                "scoring_contract_id": "m3-v1",
            },
            "compatible_analysis_signatures": ["m1-test"],
            "compatible_config_signatures": ["default"],
            "components": [
                {
                    "component_id": "target_energy",
                    "description": "Energy direction",
                    "weight": 0.22,
                    "available": True,
                    "active": True,
                }
            ],
            "supported_lane_groups": [
                {"lane_id": "maintain", "summary": "Keep the room on its current frame."}
            ],
            "capability_flags": {"vocals_available": False},
            "healthy": True,
            "engine_version": "0.1.0",
            "status_note": "ok",
        },
    )
    monkeypatch.setattr(
        "cuemate_analysis.scoring.check_analysis_compatibility",
        lambda *args, **kwargs: {
            "exact_match": False,
            "compatible": True,
            "requires_reanalysis": False,
            "reason": "compatible_but_not_exact",
            "notes": ["legacy artifact is still allowed"],
        },
    )

    assert main(
        [
            "inspect-scoring-metadata",
            "--analysis-signature",
            "legacy-analysis",
            "--config-signature",
            "default",
            "--scoring-contract-id-at-analysis",
            "m3-v1",
        ]
    ) == 0
    out = capsys.readouterr().out
    assert "Scoring metadata" in out
    assert "scoring_contract_id:  m3-v1" in out
    assert "state=active" in out
    assert "compatible_but_not_exact" in out
    assert "legacy artifact is still allowed" in out


def test_inspect_scoring_weights_json(monkeypatch, capsys):
    monkeypatch.setattr(
        "cuemate_analysis.cli.load_runtime_settings",
        lambda: SimpleNamespace(database_path=Path("fake.db")),
    )
    monkeypatch.setattr("cuemate_analysis.cli.Database", lambda _: _FakeWeightsDB())
    monkeypatch.setattr(
        "cuemate_analysis.config.build_scoring_config",
        lambda settings, target: {
            "static_weights": {"target_energy": 0.22, "harmonic": 0.12},
            "weight_floors": {"target_energy": 0.08, "harmonic": 0.04},
        },
    )

    assert main(["inspect-scoring-weights", "--playlist", "Test Playlist", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["effective_weights"]["target_energy"] == 0.31
    assert payload["effective_weights"]["harmonic"] == 0.08
    assert payload["weight_floors"]["target_energy"] == 0.08


def test_inspect_scoring_weights_text(monkeypatch, capsys):
    monkeypatch.setattr(
        "cuemate_analysis.cli.load_runtime_settings",
        lambda: SimpleNamespace(database_path=Path("fake.db")),
    )
    monkeypatch.setattr("cuemate_analysis.cli.Database", lambda _: _FakeWeightsDB())
    monkeypatch.setattr(
        "cuemate_analysis.config.build_scoring_config",
        lambda settings, target: {
            "static_weights": {"target_energy": 0.22, "harmonic": 0.12},
            "weight_floors": {"target_energy": 0.08, "harmonic": 0.04},
        },
    )

    assert main(["inspect-scoring-weights", "--playlist", "Test Playlist"]) == 0
    out = capsys.readouterr().out
    assert "Scoring weights for 'Test Playlist'" in out
    assert "target_energy" in out
    assert "0.3100" in out
    assert "Adaptation is active" in out


def test_recommend_next_requires_fresh_relative_artifacts(monkeypatch, capsys):
    class _StaleDB(_FakeDB):
        def get_playlist_stats_for_scoring(self, playlist_id: str):
            return {
                "is_stale": True,
                "stale_reason": "playlist_membership_changed",
                "relative_signature": "sig-current",
            }

    monkeypatch.setattr(
        "cuemate_analysis.cli.load_runtime_settings",
        lambda: SimpleNamespace(database_path=Path("fake.db")),
    )
    monkeypatch.setattr(
        "cuemate_analysis.config.build_relative_experiment_signature",
        lambda settings, energy_source="canonical": "sig-current",
    )
    monkeypatch.setattr("cuemate_analysis.cli.Database", lambda _: _StaleDB())
    monkeypatch.setattr(
        "cuemate_analysis.config.build_scoring_config",
        lambda settings, target: {"target": target},
    )
    monkeypatch.setattr(
        "cuemate_analysis.scoring.row_to_scoring_track_context",
        lambda row: _ctx(track_id=row["track_id"], title=row["track_id"]),
    )

    assert main(["recommend-next", "--playlist", "Test Playlist", "--target", "maintain"]) == 1
    assert "Playlist relative features are stale" in capsys.readouterr().err
