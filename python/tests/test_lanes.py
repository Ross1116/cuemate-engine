"""
Phase 3 lane organization tests.

Covers:
- compute_ranking_strength
- organize_into_lanes: lane assignment, contrast dual-membership, empty lanes, target ordering
- compute_recommendation_confidence
- get_recommendations: end-to-end orchestration with synthetic tracks
"""
from __future__ import annotations

import pytest

from cuemate_analysis.scoring import (
    ScoringTrackContext,
    compute_ranking_strength,
    compute_recommendation_confidence,
    get_recommendations,
    organize_into_lanes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(
    track_id: str = "trk_01",
    bpm: float = 128.0,
    key: str = "8B",
    key_confidence: float = 0.90,
    key_source: str = "musicalkeycnn",
    key_agreement: int = 1,
    energy_rel: float = 0.0,
    bass_rel: float = 0.0,
    drums_rel: float = 0.0,
    vocals_rel: float = 0.0,
    groove_rel: float = 0.0,
    intensity_band: str = "mid",
    role_hints: list[str] | None = None,
) -> ScoringTrackContext:
    return ScoringTrackContext(
        track_id=track_id,
        bpm=bpm,
        key=key,
        key_confidence=key_confidence,
        key_source=key_source,
        key_agreement=key_agreement,
        energy_rel=energy_rel,
        bass_rel=bass_rel,
        drums_rel=drums_rel,
        vocals_rel=vocals_rel,
        groove_rel=groove_rel,
        intensity_band=intensity_band,
        role_hints=role_hints or [],
    )


def _result(
    score: float = 0.7,
    move: str = "maintain",
    contrast_score: float = 0.2,
    track_id: str = "trk_01",
) -> dict:
    ctx = _ctx(track_id=track_id)
    return {
        "candidate": ctx,
        "score": score,
        "raw_score": score,
        "penalty_multiplier": 1.0,
        "penalty_factors": {},
        "risk": "low",
        "risk_score": 0.1,
        "risk_factors": {},
        "move": move,
        "move_confidence": 0.8,
        "move_note": None,
        "contrast_score": contrast_score,
        "component_scores": {},
        "transition_features": {},
        "confidences": {"harmonic": 1.0, "tempo": 1.0},
        "weights_used": {},
    }


def _default_config() -> dict:
    return {
        "target": "maintain",
        "max_per_lane": 3,
        "contrast_threshold": 0.45,
        "secondary_contrast_threshold": 0.65,
        "static_weights": {
            "target_energy": 0.22,
            "transition_support": 0.18,
            "bass_transition": 0.15,
            "vocal_transition": 0.13,
            "harmonic": 0.12,
            "tempo": 0.10,
            "history_fit": 0.06,
            "rhythmic_continuity": 0.04,
        },
        "weight_floors": {
            "target_energy": 0.08,
            "transition_support": 0.05,
            "bass_transition": 0.04,
            "vocal_transition": 0.03,
            "harmonic": 0.04,
            "tempo": 0.03,
            "history_fit": 0.03,
            "rhythmic_continuity": 0.02,
        },
        "harmonic_confidence_floor": 0.15,
        "thresholds": {"bpm_hard": 8.0, "bpm_soft": 3.0, "cooldown_window": 5},
        "move_types": {
            "jump_threshold": 0.12,
            "build_threshold": 0.05,
            "maintain_range": 0.05,
            "reset_energy_threshold": -0.08,
            "reset_vocal_threshold": 0.50,
            "drop_threshold": -0.05,
        },
        "penalties": {
            "max_total_penalty": 0.80,
            "bpm_over_soft": 0.30,
            "key_mismatch": 0.45,
            "vocal_clash": 0.35,
        },
    }


# ---------------------------------------------------------------------------
# compute_ranking_strength
# ---------------------------------------------------------------------------

class TestComputeRankingStrength:
    def test_empty_lane_scores_returns_one(self):
        assert compute_ranking_strength(0.8, []) == 1.0

    def test_single_score_returns_one(self):
        assert compute_ranking_strength(0.8, [0.8]) == 1.0

    def test_top_rank_returns_one(self):
        # Highest score in a 3-item lane → rank 0 → strength 1.0
        strength = compute_ranking_strength(0.9, [0.9, 0.7, 0.5])
        assert strength == pytest.approx(1.0)

    def test_bottom_rank_returns_zero(self):
        # Lowest score in a 3-item lane → rank 2 → strength 0.0
        strength = compute_ranking_strength(0.5, [0.9, 0.7, 0.5])
        assert strength == pytest.approx(0.0)

    def test_middle_rank_returns_half(self):
        strength = compute_ranking_strength(0.7, [0.9, 0.7, 0.5])
        assert strength == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# organize_into_lanes
# ---------------------------------------------------------------------------

class TestOrganizeIntoLanes:
    def test_target_lane_first_in_order(self):
        results = [
            _result(score=0.9, move="build", track_id="trk_01"),
            _result(score=0.8, move="maintain", track_id="trk_02"),
        ]
        out = organize_into_lanes(results, target_lane="maintain", config=_default_config())
        assert out["lane_order"][0] == "maintain"

    def test_build_move_goes_to_build_lane(self):
        results = [_result(score=0.9, move="build", track_id="trk_01")]
        out = organize_into_lanes(results, target_lane="build", config=_default_config())
        assert "build" in out["lanes"]
        assert out["lanes"]["build"][0]["candidate"].track_id == "trk_01"

    def test_maintain_move_goes_to_maintain_lane(self):
        results = [_result(score=0.7, move="maintain", track_id="trk_01")]
        out = organize_into_lanes(results, target_lane="maintain", config=_default_config())
        assert "maintain" in out["lanes"]

    def test_reset_move_goes_to_reset_lane(self):
        results = [_result(score=0.6, move="reset", track_id="trk_01")]
        out = organize_into_lanes(results, target_lane="reset", config=_default_config())
        assert "reset" in out["lanes"]

    def test_drop_move_goes_to_reset_lane(self):
        results = [_result(score=0.6, move="drop", track_id="trk_01")]
        out = organize_into_lanes(results, target_lane="maintain", config=_default_config())
        assert "reset" in out["lanes"]

    def test_jump_move_goes_to_jump_lane(self):
        results = [_result(score=0.9, move="jump", track_id="trk_01")]
        out = organize_into_lanes(results, target_lane="jump", config=_default_config())
        assert "jump" in out["lanes"]
        assert out["lanes"]["jump"][0]["candidate"].track_id == "trk_01"

    def test_unknown_move_goes_to_wildcard(self):
        results = [_result(score=0.5, move="unknown_move", track_id="trk_01")]
        out = organize_into_lanes(results, target_lane="maintain", config=_default_config())
        assert "wildcard" in out["lanes"]

    def test_empty_lanes_not_in_output(self):
        results = [_result(score=0.7, move="maintain", track_id="trk_01")]
        out = organize_into_lanes(results, target_lane="build", config=_default_config())
        # build lane is empty (maintain move → maintain lane)
        assert "build" not in out["lanes"]
        assert "build" not in out["lane_order"]

    def test_empty_results(self):
        out = organize_into_lanes([], target_lane="maintain", config=_default_config())
        assert out["lanes"] == {}
        assert out["lane_order"] == []

    def test_max_per_lane_cap(self):
        # 5 maintain results — only 3 should be in the lane
        results = [
            _result(score=0.9 - i * 0.05, move="maintain", track_id=f"trk_{i:02d}")
            for i in range(5)
        ]
        cfg = {**_default_config(), "max_per_lane": 3}
        out = organize_into_lanes(results, target_lane="maintain", config=cfg)
        assert len(out["lanes"]["maintain"]) == 3

    def test_contrast_dual_membership_high_score(self):
        # contrast_score >= 0.65 → secondary contrast member
        results = [
            _result(score=0.8, move="build", contrast_score=0.80, track_id="trk_01"),
        ]
        out = organize_into_lanes(results, target_lane="build", config=_default_config())
        assert "contrast" in out["lanes"]
        contrast_entry = out["lanes"]["contrast"][0]
        assert contrast_entry["secondary_lane"] is True
        assert contrast_entry["candidate"].track_id == "trk_01"

    def test_contrast_dual_membership_primary_false(self):
        # primary lane member has secondary_lane=False
        results = [_result(score=0.8, move="build", contrast_score=0.3, track_id="trk_01")]
        out = organize_into_lanes(results, target_lane="build", config=_default_config())
        assert out["lanes"]["build"][0]["secondary_lane"] is False

    def test_low_contrast_score_not_in_contrast_lane(self):
        # contrast_score=0.3 < threshold 0.65 → not in contrast lane
        results = [_result(score=0.8, move="build", contrast_score=0.30, track_id="trk_01")]
        out = organize_into_lanes(results, target_lane="maintain", config=_default_config())
        assert "contrast" not in out["lanes"]

    def test_contrast_lane_capped_at_max_per_lane(self):
        # 5 high-contrast results — contrast lane capped at max_per_lane
        results = [
            _result(score=0.9 - i * 0.05, move="maintain", contrast_score=0.90, track_id=f"trk_{i:02d}")
            for i in range(5)
        ]
        cfg = {**_default_config(), "max_per_lane": 3}
        out = organize_into_lanes(results, target_lane="maintain", config=cfg)
        assert len(out["lanes"].get("contrast", [])) == 3

    def test_target_lane_missing_from_results(self):
        # target=reset but only build/maintain candidates → reset absent from lane_order
        results = [
            _result(score=0.9, move="build", track_id="trk_01"),
            _result(score=0.8, move="maintain", track_id="trk_02"),
        ]
        out = organize_into_lanes(results, target_lane="reset", config=_default_config())
        assert "reset" not in out["lane_order"]

    def test_lane_order_canonical_after_target(self):
        results = [
            _result(score=0.9, move="build", track_id="trk_01"),
            _result(score=0.85, move="jump", track_id="trk_04"),
            _result(score=0.8, move="maintain", track_id="trk_02"),
            _result(score=0.7, move="reset", track_id="trk_03"),
        ]
        out = organize_into_lanes(results, target_lane="maintain", config=_default_config())
        order = out["lane_order"]
        assert order[0] == "maintain"
        # canonical order after target: build < jump < reset
        assert order.index("build") < order.index("jump")
        assert order.index("jump") < order.index("reset")

    def test_primary_lane_field_set_correctly(self):
        results = [_result(score=0.8, move="build", track_id="trk_01")]
        out = organize_into_lanes(results, target_lane="build", config=_default_config())
        assert out["lanes"]["build"][0]["primary_lane"] == "build"


# ---------------------------------------------------------------------------
# compute_recommendation_confidence
# ---------------------------------------------------------------------------

class TestComputeRecommendationConfidence:
    def test_empty_results_returns_zero(self):
        assert compute_recommendation_confidence([]) == 0.0

    def test_single_result_high_confidence(self):
        results = [_result(score=0.9)]
        conf = compute_recommendation_confidence(results)
        assert 0.0 < conf <= 1.0

    def test_high_separation_boosts_confidence(self):
        # Large gap between 1st and 2nd → higher separation factor
        high_sep = [_result(score=0.95), _result(score=0.60)]
        low_sep = [_result(score=0.95), _result(score=0.90)]
        c_high = compute_recommendation_confidence(high_sep)
        c_low = compute_recommendation_confidence(low_sep)
        assert c_high > c_low

    def test_more_candidates_boosts_confidence(self):
        few = [_result(score=0.8 - i * 0.05) for i in range(2)]
        many = [_result(score=0.8 - i * 0.02) for i in range(10)]
        assert compute_recommendation_confidence(many) >= compute_recommendation_confidence(few)

    def test_low_feature_confidence_reduces_score(self):
        results = [_result(score=0.8), _result(score=0.6)]
        high_conf = compute_recommendation_confidence(results, avg_feature_conf=1.0)
        low_conf = compute_recommendation_confidence(results, avg_feature_conf=0.2)
        assert high_conf > low_conf

    def test_low_analysis_coverage_reduces_score(self):
        results = [_result(score=0.8), _result(score=0.6)]
        full = compute_recommendation_confidence(results, analysis_coverage=1.0)
        partial = compute_recommendation_confidence(results, analysis_coverage=0.0)
        assert full > partial

    def test_output_clamped_to_zero_one(self):
        results = [_result(score=0.9)]
        conf = compute_recommendation_confidence(results, avg_feature_conf=1.0, analysis_coverage=1.0)
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# get_recommendations (end-to-end orchestration)
# ---------------------------------------------------------------------------

class TestGetRecommendations:
    def _make_current(self) -> ScoringTrackContext:
        return _ctx(track_id="trk_current", bpm=128.0, key="8B", energy_rel=0.0)

    def _make_candidates(self) -> list[ScoringTrackContext]:
        return [
            # close maintain candidate (same key, same BPM)
            _ctx(track_id="trk_A", bpm=128.0, key="8B", energy_rel=0.02),
            # build candidate (slight energy boost, compatible key)
            _ctx(track_id="trk_B", bpm=128.0, key="9B", energy_rel=0.08),
            # reset candidate (energy drop)
            _ctx(track_id="trk_C", bpm=128.0, key="8B", energy_rel=-0.12),
            # distant BPM candidate (outside hard limit)
            _ctx(track_id="trk_D", bpm=145.0, key="1A", energy_rel=0.0),
        ]

    def test_returns_required_keys(self):
        current = self._make_current()
        candidates = self._make_candidates()
        out = get_recommendations(current, candidates, [], _default_config())
        assert "lane_order" in out
        assert "lanes" in out
        assert "recommendation_confidence" in out
        assert "meta" in out

    def test_meta_fields(self):
        current = self._make_current()
        candidates = self._make_candidates()
        out = get_recommendations(current, candidates, [], _default_config())
        meta = out["meta"]
        assert meta["current_track_id"] == "trk_current"
        assert meta["total_candidates"] == 4
        assert "scored_candidates" in meta
        assert "filtered_candidates" in meta

    def test_current_track_excluded(self):
        current = self._make_current()
        candidates = [current] + self._make_candidates()
        out = get_recommendations(current, candidates, [], _default_config())
        all_ids = [
            r["candidate"].track_id
            for lane_items in out["lanes"].values()
            for r in lane_items
        ]
        assert "trk_current" not in all_ids

    def test_target_lane_first(self):
        current = self._make_current()
        candidates = self._make_candidates()
        out = get_recommendations(
            current, candidates, [], _default_config(), target="maintain"
        )
        if out["lane_order"]:
            assert out["lane_order"][0] == "maintain"

    def test_target_maintain_default(self):
        current = self._make_current()
        candidates = self._make_candidates()
        out = get_recommendations(current, candidates, [], _default_config())
        assert out["meta"]["target"] == "maintain"

    def test_recommendation_confidence_between_zero_and_one(self):
        current = self._make_current()
        candidates = self._make_candidates()
        out = get_recommendations(current, candidates, [], _default_config())
        assert 0.0 <= out["recommendation_confidence"] <= 1.0

    def test_no_candidates_returns_empty_lanes(self):
        current = self._make_current()
        out = get_recommendations(current, [], [], _default_config())
        assert out["lanes"] == {}
        assert out["lane_order"] == []
        assert out["meta"]["scored_candidates"] == 0

    def test_max_per_lane_respected(self):
        current = self._make_current()
        # 6 maintain-like candidates (same BPM, compatible key, small energy delta)
        candidates = [
            _ctx(track_id=f"trk_{i:02d}", bpm=128.0, key="8B", energy_rel=0.01 * i)
            for i in range(6)
        ]
        cfg = {**_default_config(), "max_per_lane": 2}
        out = get_recommendations(current, candidates, [], cfg, max_per_lane=2)
        for lane_items in out["lanes"].values():
            assert len(lane_items) <= 2

    def test_uses_configured_max_per_lane_when_arg_omitted(self):
        current = self._make_current()
        candidates = [
            _ctx(track_id=f"trk_{i:02d}", bpm=128.0, key="8B", energy_rel=0.01 * i)
            for i in range(6)
        ]
        cfg = {**_default_config(), "max_per_lane": 2}
        out = get_recommendations(current, candidates, [], cfg)
        for lane_items in out["lanes"].values():
            assert len(lane_items) <= 2

    def test_playlist_stats_passed_through(self):
        """get_recommendations accepts playlist_stats without crashing."""
        current = self._make_current()
        candidates = self._make_candidates()
        playlist_stats = {
            "energy_spread": 0.15,
            "adapted_weights": None,
        }
        out = get_recommendations(
            current, candidates, [], _default_config(), playlist_stats=playlist_stats
        )
        assert "lanes" in out

    def test_history_cooldown_excludes_recent_tracks(self):
        current = self._make_current()
        # trk_A is in recent history → should be filtered out by cooldown
        # history items are dicts (as consumed by filter_candidates)
        history = [{"track_id": "trk_A"}] * 5
        candidates = [
            _ctx(track_id="trk_A", bpm=128.0, key="8B", energy_rel=0.02),
            _ctx(track_id="trk_B", bpm=128.0, key="9B", energy_rel=0.08),
        ]
        out = get_recommendations(current, candidates, history, _default_config())
        all_ids = [
            r["candidate"].track_id
            for lane_items in out["lanes"].values()
            for r in lane_items
        ]
        assert "trk_A" not in all_ids
        assert "trk_B" in all_ids

    def test_history_accepts_scoring_track_context_items(self):
        current = self._make_current()
        history = [
            _ctx(track_id="trk_hist_1", bpm=128.0, key="8A", energy_rel=0.30),
            _ctx(track_id="trk_hist_2", bpm=128.0, key="8B", energy_rel=0.35),
        ]
        candidates = [
            _ctx(track_id="trk_A", bpm=128.0, key="9A", energy_rel=0.02),
        ]
        out = get_recommendations(current, candidates, history, _default_config())
        all_ids = [
            r["candidate"].track_id
            for lane_items in out["lanes"].values()
            for r in lane_items
        ]
        assert "trk_A" in all_ids

    def test_avg_feature_conf_ignores_stubbed_components(self, monkeypatch):
        current = self._make_current()
        candidates = [_ctx(track_id="trk_A", bpm=128.0, key="8B", energy_rel=0.02)]
        captured: dict[str, float] = {}

        def fake_score_candidate(*args, **kwargs):
            return {
                "candidate": candidates[0],
                "score": 0.8,
                "raw_score": 0.8,
                "penalty_multiplier": 1.0,
                "penalty_factors": [],
                "risk": "low",
                "risk_score": 0.1,
                "risk_factors": [],
                "move": "maintain",
                "move_confidence": 0.9,
                "move_note": None,
                "contrast_score": 0.1,
                "component_scores": {
                    "target_energy": 0.9,
                    "transition_support": None,
                    "bass_transition": 0.8,
                    "vocal_transition": None,
                    "harmonic": 0.9,
                    "tempo": 1.0,
                    "history_fit": 1.0,
                    "rhythmic_continuity": None,
                },
                "transition_features": {},
                "confidences": {
                    "target_energy": 0.5,
                    "transition_support": 1.0,
                    "bass_transition": 0.5,
                    "vocal_transition": 1.0,
                    "harmonic": 0.5,
                    "tempo": 0.5,
                    "history_fit": 0.5,
                    "rhythmic_continuity": 1.0,
                },
                "weights_used": {},
                "secondary_lane": False,
            }

        def fake_recommendation_confidence(ranked_results, analysis_coverage=1.0, avg_feature_conf=1.0):
            captured["avg_feature_conf"] = avg_feature_conf
            return 0.42

        monkeypatch.setattr("cuemate_analysis.scoring.score_candidate", fake_score_candidate)
        monkeypatch.setattr(
            "cuemate_analysis.scoring.compute_recommendation_confidence",
            fake_recommendation_confidence,
        )

        out = get_recommendations(current, candidates, [], _default_config())
        assert out["recommendation_confidence"] == 0.42
        assert captured["avg_feature_conf"] == pytest.approx(0.5)
