"""
Phase 1 scoring tests.

Covers:
- Key parsing and compatibility
- BPM distance with ratio awareness
- All v1 score components
- Key trust / harmonic confidence policy
- Confidence modulation (weight floors, corroborated vs weak key)
- Penalty system
- Candidate filtering
- Move classification
- Risk computation
- Integration: score_candidate with synthetic data
"""
from __future__ import annotations

import pytest

from cuemate_analysis.scoring import (
    LABEL_CONFIG,
    SCORING_CONTRACT_ID,
    HARMONIC_CONFIDENCE_FLOOR,
    HARMONIC_SCORE_MAP,
    KEY_CONFIDENCE_THRESHOLD,
    STATIC_WEIGHTS,
    ScoringTrackContext,
    bass_transition_score,
    build_confidence_map,
    camelot_compatibility,
    classify_move,
    classify_label,
    classify_track_labels,
    check_analysis_compatibility,
    compute_penalties,
    compute_risk,
    compute_transition_features,
    compute_weighted_score,
    contrast_score,
    effective_bpm_distance,
    filter_candidates,
    get_scoring_metadata,
    harmonic_score,
    history_fit_score,
    parse_camelot,
    resolve_effective_weights,
    score_candidate,
    sigmoid_normalize,
    target_energy_score,
    tempo_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _track(
    track_id: str = "trk_test",
    bpm: float = 128.0,
    key: str | None = "8A",
    key_confidence: float | None = 0.80,
    key_source: str | None = "musicalkeycnn",
    key_agreement: int | None = None,
    energy_rel: float | None = 0.5,
    bass_rel: float | None = 0.5,
    drums_rel: float | None = 0.5,
    vocals_rel: float | None = 0.2,
    groove_rel: float | None = 0.5,
    intensity_band: str | None = "Drive",
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


_DEFAULT_CONFIG: dict = {
    "target": "maintain",
    "thresholds": {
        "bpm_hard": 8.0,
        "bpm_soft": 3.0,
        "cooldown_window": 5,
    },
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
# parse_camelot
# ---------------------------------------------------------------------------


class TestParseCamelot:
    def test_valid_minor(self):
        assert parse_camelot("8A") == (8, "A")

    def test_valid_major(self):
        assert parse_camelot("11B") == (11, "B")

    def test_boundary_1(self):
        assert parse_camelot("1A") == (1, "A")

    def test_boundary_12(self):
        assert parse_camelot("12B") == (12, "B")

    def test_out_of_range(self):
        assert parse_camelot("13A") == (None, None)

    def test_zero(self):
        assert parse_camelot("0A") == (None, None)

    def test_bad_mode(self):
        assert parse_camelot("8C") == (None, None)

    def test_empty(self):
        assert parse_camelot("") == (None, None)

    def test_none(self):
        assert parse_camelot(None) == (None, None)

    def test_non_numeric(self):
        assert parse_camelot("XA") == (None, None)


# ---------------------------------------------------------------------------
# Phase 6: labels & metadata
# ---------------------------------------------------------------------------


class TestAbsoluteLabels:
    def test_classify_label_uses_boundaries(self):
        cfg = LABEL_CONFIG["energy"]
        assert classify_label(0.54, cfg["boundaries"], cfg["labels"]) == "Low"
        assert classify_label(0.88, cfg["boundaries"], cfg["labels"]) == "Max"

    def test_classify_label_none_returns_none(self):
        cfg = LABEL_CONFIG["vocals"]
        assert classify_label(None, cfg["boundaries"], cfg["labels"]) is None

    def test_classify_track_labels_from_dict(self):
        labels = classify_track_labels(
            {
                "energy_abs": 0.80,
                "bass_abs": 0.71,
                "drums_abs": 0.76,
                "harmonic_abs": 0.70,
                "vocals_abs": 0.20,
                "groove_abs": 0.86,
            }
        )
        assert labels == {
            "energy": "Peak",
            "bass": "Punch",
            "drums": "Strong",
            "harmonic": "Full",
            "vocals": "Instrumental",
            "groove": "Swing",
        }


class TestScoringMetadata:
    def test_metadata_exposes_contract_and_version(self):
        metadata = get_scoring_metadata()
        assert metadata["active_signatures"]["scoring_contract_id"] == SCORING_CONTRACT_ID
        assert metadata["engine_version"]
        assert metadata["capability_flags"]["vocals_available"] is False
        assert "stubbed and excluded from weighted scoring" in metadata["status_note"]

    def test_metadata_components_follow_weight_table(self):
        metadata = get_scoring_metadata()
        component_ids = [component["component_id"] for component in metadata["components"]]
        assert component_ids == list(STATIC_WEIGHTS.keys())

    def test_metadata_marks_stubbed_components_unavailable(self):
        metadata = get_scoring_metadata()
        by_id = {component["component_id"]: component for component in metadata["components"]}
        assert by_id["transition_support"]["available"] is False
        assert by_id["transition_support"]["active"] is False
        assert by_id["transition_support"]["status"] == "stubbed"
        assert by_id["target_energy"]["available"] is True
        assert by_id["target_energy"]["active"] is True
        assert by_id["target_energy"]["status"] == "live"

    def test_check_analysis_compatibility_exact(self):
        metadata = get_scoring_metadata()
        active = metadata["active_signatures"]
        status = check_analysis_compatibility(
            active["analysis_signature"],
            active["config_signature"],
            active["scoring_contract_id"],
            scoring_metadata=metadata,
        )
        assert status["exact_match"] is True
        assert status["compatible"] is True
        assert status["requires_reanalysis"] is False

    def test_check_analysis_compatibility_compatible_but_not_exact(self):
        metadata = get_scoring_metadata(
            compatible_analysis_signatures=["legacy-analysis"],
            compatible_config_signatures=["legacy-config"],
        )
        status = check_analysis_compatibility(
            "legacy-analysis",
            "legacy-config",
            SCORING_CONTRACT_ID,
            scoring_metadata=metadata,
        )
        assert status["exact_match"] is False
        assert status["compatible"] is True
        assert status["requires_reanalysis"] is False
        assert status["reason"] == "compatible_but_not_exact"

    def test_check_analysis_compatibility_accepts_active_family_prefix(self):
        metadata = get_scoring_metadata()
        active = metadata["active_signatures"]
        detailed_signature = f"{active['analysis_signature']}-tempo-tempocnn-abc123"
        status = check_analysis_compatibility(
            detailed_signature,
            active["config_signature"],
            SCORING_CONTRACT_ID,
            scoring_metadata=metadata,
        )
        assert status["exact_match"] is False
        assert status["compatible"] is True
        assert status["requires_reanalysis"] is False
        assert status["reason"] == "compatible_but_not_exact"

    def test_check_analysis_compatibility_rejects_missing_contract(self):
        metadata = get_scoring_metadata()
        active = metadata["active_signatures"]
        status = check_analysis_compatibility(
            active["analysis_signature"],
            active["config_signature"],
            None,
            scoring_metadata=metadata,
        )
        assert status["compatible"] is False
        assert status["requires_reanalysis"] is True
        assert status["reason"] == "missing_scoring_contract_id"


# ---------------------------------------------------------------------------
# camelot_compatibility
# ---------------------------------------------------------------------------


class TestCamelotCompatibility:
    def test_perfect(self):
        dist, label = camelot_compatibility(8, "A", 8, "A")
        assert dist == 0
        assert label == "perfect"

    def test_relative_key(self):
        dist, label = camelot_compatibility(8, "A", 8, "B")
        assert dist == 1
        assert label == "relative_key"

    def test_adjacent_same_mode(self):
        dist, label = camelot_compatibility(8, "A", 9, "A")
        assert dist == 1
        assert label == "adjacent"

    def test_cross_adjacent(self):
        dist, label = camelot_compatibility(8, "A", 9, "B")
        assert dist == 2
        assert label == "cross_adjacent"

    def test_energy_boost(self):
        dist, label = camelot_compatibility(8, "A", 10, "A")
        assert dist == 2
        assert label == "energy_boost"

    def test_energy_key_change(self):
        dist, label = camelot_compatibility(1, "A", 6, "A")
        assert dist == 2
        assert label == "energy_key_change"

    def test_mismatch(self):
        dist, label = camelot_compatibility(1, "A", 6, "B")
        assert dist == 3
        assert label == "mismatch"

    def test_wheel_wraps(self):
        # 12 and 1 are adjacent on the wheel (distance 1)
        dist, label = camelot_compatibility(12, "A", 1, "A")
        assert dist == 1
        assert label == "adjacent"

    def test_none_inputs(self):
        dist, label = camelot_compatibility(None, None, 8, "A")
        assert dist == 3
        assert label == "mismatch"

    def test_harmonic_score_map_coverage(self):
        for d in (0, 1, 2, 3):
            assert d in HARMONIC_SCORE_MAP


# ---------------------------------------------------------------------------
# effective_bpm_distance
# ---------------------------------------------------------------------------


class TestEffectiveBpmDistance:
    def test_direct_match(self):
        dist, matched, rel, raw = effective_bpm_distance(128.0, 128.0)
        assert dist == 0.0
        assert rel == "direct"
        assert raw == 0.0

    def test_direct_small_delta(self):
        dist, _, rel, raw = effective_bpm_distance(128.0, 130.0)
        assert rel == "direct"
        assert pytest.approx(dist, abs=0.01) == 2.0

    def test_double_relationship(self):
        dist, matched, rel, raw = effective_bpm_distance(128.0, 64.0)
        # 64*2 = 128 → raw=0, but floor for "double" is 0.5
        assert rel == "double"
        assert dist >= 0.5

    def test_half_relationship(self):
        dist, _, rel, _ = effective_bpm_distance(64.0, 128.0)
        assert rel == "half"
        assert dist >= 0.5

    def test_three_two_floor(self):
        # From a=128, b=85.33: b*3/2 ≈ 128 → matched as "three_two" (floor=1.5)
        dist, _, rel, _ = effective_bpm_distance(128.0, 85.33)
        assert rel == "three_two"
        assert dist >= 1.5

    def test_hard_bpm_gap(self):
        # 128 vs 140 — direct, raw = 12
        dist, _, rel, _ = effective_bpm_distance(128.0, 140.0)
        assert rel == "direct"
        assert dist == pytest.approx(12.0, abs=0.1)

    def test_creative_ratio_never_beats_close_direct(self):
        # A close direct match (delta=0.5) vs double with same raw delta (64.25*2=128.5 → raw=0.5)
        # double floor = 0.5, so effective_double = max(0.5, 0.5) = 0.5 → tie.
        # Use a direct delta < 0.5 to guarantee it wins over the double floor.
        dist_direct, _, rel_direct, _ = effective_bpm_distance(128.0, 128.3)
        dist_double, _, rel_double, _ = effective_bpm_distance(128.0, 64.0)  # exact half → floor 0.5
        assert dist_direct < dist_double
        assert rel_direct == "direct"
        assert rel_double == "double"


# ---------------------------------------------------------------------------
# harmonic_score
# ---------------------------------------------------------------------------


class TestHarmonicScore:
    def test_perfect_match(self):
        assert harmonic_score("8A", "8A") == 1.00

    def test_adjacent(self):
        assert harmonic_score("8A", "9A") == 0.85

    def test_mismatch(self):
        assert harmonic_score("1A", "6B") == 0.15

    def test_none_key(self):
        assert harmonic_score(None, "8A") == 0.5
        assert harmonic_score("8A", None) == 0.5
        assert harmonic_score(None, None) == 0.5


# ---------------------------------------------------------------------------
# tempo_score
# ---------------------------------------------------------------------------


class TestTempoScore:
    def test_exact_match_is_1(self):
        s = tempo_score(128.0, 128.0, _DEFAULT_CONFIG)
        assert s == pytest.approx(1.0, abs=0.01)

    def test_within_soft_threshold(self):
        s = tempo_score(128.0, 130.0, _DEFAULT_CONFIG)
        assert s > 0.5

    def test_at_hard_limit(self):
        s = tempo_score(128.0, 136.0, _DEFAULT_CONFIG)
        assert s >= 0.15  # floor

    def test_floor_not_zero(self):
        s = tempo_score(128.0, 200.0, _DEFAULT_CONFIG)
        assert s >= 0.15


# ---------------------------------------------------------------------------
# target_energy_score
# ---------------------------------------------------------------------------


class TestTargetEnergyScore:
    def test_maintain_no_delta(self):
        assert target_energy_score(0.0, "maintain") == 1.0

    def test_maintain_large_delta(self):
        assert target_energy_score(0.5, "maintain") == 0.0

    def test_build_positive_delta(self):
        s = target_energy_score(0.15, "build")
        assert s > 0.5

    def test_build_negative_delta(self):
        s = target_energy_score(-0.15, "build")
        assert s < 0.5

    def test_reset_negative_delta(self):
        s = target_energy_score(-0.15, "reset")
        assert s > 0.5

    def test_reset_near_neutral_delta_still_scores_as_viable(self):
        s = target_energy_score(0.01, "reset")
        assert s > 0.5

    def test_jump_large_delta(self):
        s = target_energy_score(0.3, "jump")
        assert s > 0.7

    def test_unknown_target(self):
        assert target_energy_score(0.1, "unknown_target") == 0.5


# ---------------------------------------------------------------------------
# bass_transition_score
# ---------------------------------------------------------------------------


class TestBassTransitionScore:
    def test_maintain_no_delta(self):
        s = bass_transition_score(0.5, 0.5, "maintain")
        assert s == pytest.approx(1.0, abs=0.01)

    def test_maintain_large_delta(self):
        s = bass_transition_score(0.0, 1.0, "maintain")
        assert s == 0.0

    def test_build_positive_delta(self):
        s = bass_transition_score(0.3, 0.7, "build")
        assert s > 0.5

    def test_reset_small_bass_rise_is_tolerated(self):
        s = bass_transition_score(0.5, 0.55, "reset")
        assert 0.2 <= s < 1.0

    def test_none_inputs_fallback(self):
        s = bass_transition_score(None, None, "maintain")
        assert s == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# sigmoid_normalize
# ---------------------------------------------------------------------------


class TestSigmoidNormalize:
    def test_center_is_half(self):
        assert sigmoid_normalize(0.0, 0.0, 0.1) == pytest.approx(0.5, abs=0.01)

    def test_above_center_is_above_half(self):
        assert sigmoid_normalize(0.2, 0.0, 0.1) > 0.5

    def test_below_center_is_below_half(self):
        assert sigmoid_normalize(-0.2, 0.0, 0.1) < 0.5

    def test_output_between_0_and_1(self):
        for x in (-10.0, -1.0, 0.0, 1.0, 10.0):
            v = sigmoid_normalize(x, 0.0, 1.0)
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# history_fit_score
# ---------------------------------------------------------------------------


class TestHistoryFitScore:
    def test_empty_history(self):
        c = _track(track_id="t1")
        assert history_fit_score(c, []) == 1.0

    def test_key_repetition_penalised(self):
        c = _track(track_id="t2", key="8A")
        history = [{"key": "8A", "energy_rel": 0.5}] * 3
        score = history_fit_score(c, history)
        assert score < 1.0

    def test_energy_stagnation_penalised(self):
        c = _track(track_id="t3", key="9A")
        history = [{"key": "5B", "energy_rel": 0.5}] * 4
        score = history_fit_score(c, history)
        assert score < 1.0

    def test_no_repeat_full_score(self):
        c = _track(track_id="t4", key="1A")
        history = [
            {"key": "3B", "energy_rel": 0.3},
            {"key": "6A", "energy_rel": 0.6},
            {"key": "9B", "energy_rel": 0.7},
        ]
        score = history_fit_score(c, history)
        assert score == 1.0

    def test_score_non_negative(self):
        c = _track(track_id="t5", key="8A")
        history = [{"key": "8A", "energy_rel": 0.5}] * 10
        assert history_fit_score(c, history) >= 0.0

    def test_unknown_energies_do_not_trigger_stagnation(self):
        c = _track(track_id="t6", key="9A")
        history = [
            {"key": "5B", "energy_rel": None},
            {"key": "6A", "energy_rel": None},
            {"key": "7B", "energy_rel": None},
            {"key": "8A", "energy_rel": 0.3},
        ]
        assert history_fit_score(c, history) == 1.0

    def test_unknown_candidate_key_does_not_trigger_repeat_penalty(self):
        c = _track(track_id="t7", key=None)
        history = [{"key": None, "energy_rel": 0.5}] * 3
        assert history_fit_score(c, history) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# contrast_score
# ---------------------------------------------------------------------------


class TestContrastScore:
    def test_identical_low_contrast(self):
        r = {"energy_rel": 0.5, "bass_rel": 0.5, "vocals_rel": 0.2,
             "groove_rel": 0.5, "role_hints": ["safe_continuation"]}
        s = contrast_score(r, r)
        assert s < 0.15  # very low — all deltas are zero

    def test_opposite_high_contrast(self):
        r_a = {"energy_rel": 0.1, "bass_rel": 0.1, "vocals_rel": 0.9,
               "groove_rel": 0.1, "role_hints": ["opener"]}
        r_b = {"energy_rel": 0.9, "bass_rel": 0.9, "vocals_rel": 0.0,
               "groove_rel": 0.9, "role_hints": ["peak_tool"]}
        s = contrast_score(r_a, r_b)
        assert s > 0.5

    def test_output_between_0_and_1(self):
        r_a = {"energy_rel": 0.3, "bass_rel": 0.4, "vocals_rel": 0.1,
               "groove_rel": 0.3, "role_hints": []}
        r_b = {"energy_rel": 0.7, "bass_rel": 0.6, "vocals_rel": 0.8,
               "groove_rel": 0.7, "role_hints": ["vocal_feature"]}
        s = contrast_score(r_a, r_b)
        assert 0.0 <= s <= 1.0

    def test_preserves_zero_values(self):
        r_a = {"energy_rel": 0.0, "bass_rel": 0.0, "vocals_rel": 0.0,
               "groove_rel": 0.0, "role_hints": []}
        r_b = {"energy_rel": 0.0, "bass_rel": 0.0, "vocals_rel": 0.0,
               "groove_rel": 0.0, "role_hints": []}
        assert contrast_score(r_a, r_b) == 0.0

    def test_missing_vocals_do_not_count_as_instrumental_contrast(self):
        r_a = {"energy_rel": 0.5, "bass_rel": 0.5, "vocals_rel": None,
               "groove_rel": 0.5, "role_hints": []}
        r_b = {"energy_rel": 0.5, "bass_rel": 0.5, "vocals_rel": None,
               "groove_rel": 0.5, "role_hints": []}
        assert contrast_score(r_a, r_b) == 0.0


# ---------------------------------------------------------------------------
# Key trust / build_confidence_map
# ---------------------------------------------------------------------------


class TestKeyTrust:
    def test_corroborated_full_confidence(self):
        current = _track(key_confidence=0.45, key_agreement=1)
        candidate = _track(key_confidence=0.45, key_agreement=2)
        conf = build_confidence_map(current, candidate)
        assert conf["harmonic"] == pytest.approx(1.0)

    def test_high_confidence_standalone_medium(self):
        current = _track(key_confidence=0.75, key_agreement=None)
        candidate = _track(key_confidence=0.80, key_agreement=None)
        conf = build_confidence_map(current, candidate)
        # Should equal min(0.75, 0.80) = 0.75
        assert conf["harmonic"] == pytest.approx(0.75)

    def test_weak_standalone_floor(self):
        current = _track(key_confidence=0.30, key_agreement=None)
        candidate = _track(key_confidence=0.25, key_agreement=None)
        conf = build_confidence_map(current, candidate)
        # Both are weak: max(floor, kc*0.5) → max(0.15, 0.15)=0.15 for current, max(0.15,0.125)=0.15 for candidate
        assert conf["harmonic"] == pytest.approx(HARMONIC_CONFIDENCE_FLOOR, abs=0.01)

    def test_min_of_both_sides(self):
        # corroborated current, weak candidate → min = weak
        current = _track(key_confidence=0.90, key_agreement=1)
        candidate = _track(key_confidence=0.20, key_agreement=None)
        conf = build_confidence_map(current, candidate)
        assert conf["harmonic"] < KEY_CONFIDENCE_THRESHOLD

    def test_other_components_full_confidence(self):
        current = _track(key_confidence=0.20)
        candidate = _track(key_confidence=0.20)
        conf = build_confidence_map(current, candidate)
        for key in ("target_energy", "tempo", "bass_transition", "history_fit"):
            assert conf[key] == 1.0

    def test_custom_harmonic_floor_from_config(self):
        current = _track(key_confidence=0.10, key_agreement=None)
        candidate = _track(key_confidence=0.10, key_agreement=None)
        conf = build_confidence_map(
            current,
            candidate,
            {"harmonic_confidence_floor": 0.25},
        )
        assert conf["harmonic"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# compute_weighted_score
# ---------------------------------------------------------------------------


class TestComputeWeightedScore:
    def test_uniform_scores_return_value(self):
        scores = {k: 0.5 for k in STATIC_WEIGHTS}
        confs = {k: 1.0 for k in STATIC_WEIGHTS}
        result = compute_weighted_score(scores, STATIC_WEIGHTS, confs)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_weight_floor_prevents_zero_weight(self):
        scores = {k: 1.0 for k in STATIC_WEIGHTS}
        # Set harmonic confidence to near-zero
        confs = {k: 1.0 for k in STATIC_WEIGHTS}
        confs["harmonic"] = 0.001
        result_low = compute_weighted_score(scores, STATIC_WEIGHTS, confs)
        confs["harmonic"] = 1.0
        result_high = compute_weighted_score(scores, STATIC_WEIGHTS, confs)
        assert 0.0 <= result_low <= 1.0
        assert 0.0 <= result_high <= 1.0
        assert result_low < result_high

    def test_harmonic_weight_reduced_for_weak_key(self):
        scores = {k: 0.5 for k in STATIC_WEIGHTS}
        scores["harmonic"] = 1.0  # harmonic is great
        confs_full = {k: 1.0 for k in STATIC_WEIGHTS}
        confs_weak = {k: 1.0 for k in STATIC_WEIGHTS}
        confs_weak["harmonic"] = HARMONIC_CONFIDENCE_FLOOR  # very weak

        score_full = compute_weighted_score(scores, STATIC_WEIGHTS, confs_full)
        score_weak = compute_weighted_score(scores, STATIC_WEIGHTS, confs_weak)
        # Harmonic contribution should be lower when confidence is low
        assert score_full > score_weak

    def test_output_between_0_and_1(self):
        scores = {k: 0.8 for k in STATIC_WEIGHTS}
        confs = {k: 0.5 for k in STATIC_WEIGHTS}
        weaker = compute_weighted_score(
            scores,
            STATIC_WEIGHTS,
            confs,
            weight_floors={"harmonic": 0.05},
            harmonic_confidence_floor=0.05,
        )
        stronger = compute_weighted_score(
            scores,
            STATIC_WEIGHTS,
            confs,
            weight_floors={"harmonic": 0.35},
            harmonic_confidence_floor=0.35,
        )
        assert 0.0 <= weaker <= 1.0
        assert 0.0 <= stronger <= 1.0
        assert weaker != stronger

    def test_uses_configured_weight_floors_and_confidence_floor(self):
        scores = {k: 0.5 for k in STATIC_WEIGHTS}
        confs = {k: 0.01 for k in STATIC_WEIGHTS}
        result = compute_weighted_score(
            scores,
            STATIC_WEIGHTS,
            confs,
            weight_floors={"harmonic": 0.20},
            harmonic_confidence_floor=0.30,
        )
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# resolve_effective_weights
# ---------------------------------------------------------------------------


class TestResolveEffectiveWeights:
    def test_falls_back_to_static(self):
        w = resolve_effective_weights(None)
        assert w == STATIC_WEIGHTS

    def test_uses_adapted_weights(self):
        adapted = {"target_energy": 0.30, "harmonic": 0.20}
        stats = {"adapted_weights": adapted}
        w = resolve_effective_weights(stats)
        assert w == adapted

    def test_falls_back_when_no_adapted(self):
        stats = {"energy_spread": 0.3}
        w = resolve_effective_weights(stats)
        assert w == STATIC_WEIGHTS

    def test_uses_config_static_weights_when_provided(self):
        stats = {"energy_spread": 0.3}
        config = {"static_weights": {"target_energy": 0.40, "harmonic": 0.10}}
        w = resolve_effective_weights(stats, config)
        assert w == config["static_weights"]


# ---------------------------------------------------------------------------
# compute_transition_features
# ---------------------------------------------------------------------------


class TestComputeTransitionFeatures:
    def test_missing_vocals_stay_unknown(self):
        current = _track(track_id="t1", vocals_rel=None)
        candidate = _track(track_id="t2", vocals_rel=None)
        tf = compute_transition_features(current, candidate)
        assert tf["current_vocals_rel"] is None
        assert tf["candidate_vocals_rel"] is None


# ---------------------------------------------------------------------------
# compute_penalties
# ---------------------------------------------------------------------------


class TestComputePenalties:
    def _tf(self, **kwargs):
        defaults = {
            "effective_bpm_distance": 0.0,
            "raw_bpm_distance": 0.0,
            "bpm_relationship": "direct",
            "key_distance": 0,
            "key_compat_label": "perfect",
            "key_confidence_current": 0.9,
            "key_confidence_candidate": 0.9,
            "delta_energy_rel": 0.0,
            "delta_bass_rel": 0.0,
            "current_vocals_rel": 0.0,
            "candidate_vocals_rel": 0.0,
            "current_outro_low_end": 0.0,
            "candidate_intro_low_end": 0.0,
        }
        defaults.update(kwargs)
        return defaults

    def test_no_penalty_for_clean_transition(self):
        tf = self._tf()
        scores = {"harmonic": 1.0}
        confs = {k: 1.0 for k in STATIC_WEIGHTS}
        mult, factors = compute_penalties(tf, scores, confs, _DEFAULT_CONFIG)
        assert mult == 1.0
        assert factors == []

    def test_bpm_penalty_over_soft(self):
        tf = self._tf(effective_bpm_distance=6.0)
        scores = {"harmonic": 0.5}
        confs = {k: 1.0 for k in STATIC_WEIGHTS}
        mult, factors = compute_penalties(tf, scores, confs, _DEFAULT_CONFIG)
        assert mult < 1.0
        assert any(f["factor"] == "bpm_over_soft" for f in factors)

    def test_key_mismatch_penalty_gated_on_confidence_and_weak_harmonic(self):
        # Gate: key_dist >= 3, key_conf >= 0.60, harmonic < 0.35
        tf = self._tf(
            key_distance=3,
            key_confidence_current=0.80,
            key_confidence_candidate=0.80,
        )
        scores = {"harmonic": 0.15}  # weak harmonic
        confs = {k: 1.0 for k in STATIC_WEIGHTS}
        mult, factors = compute_penalties(tf, scores, confs, _DEFAULT_CONFIG)
        assert any(f["factor"] == "key_mismatch" for f in factors)

    def test_key_mismatch_not_applied_for_high_harmonic_score(self):
        tf = self._tf(
            key_distance=3,
            key_confidence_current=0.80,
            key_confidence_candidate=0.80,
        )
        scores = {"harmonic": 0.85}  # strong harmonic — gate should block
        confs = {k: 1.0 for k in STATIC_WEIGHTS}
        mult, factors = compute_penalties(tf, scores, confs, _DEFAULT_CONFIG)
        assert not any(f["factor"] == "key_mismatch" for f in factors)

    def test_vocal_clash_penalty(self):
        tf = self._tf(current_vocals_rel=0.8, candidate_vocals_rel=0.8)
        scores = {"harmonic": 0.5}
        confs = {k: 1.0 for k in STATIC_WEIGHTS}
        mult, factors = compute_penalties(tf, scores, confs, _DEFAULT_CONFIG)
        assert any(f["factor"] == "vocal_clash" for f in factors)

    def test_penalty_multiplier_bounded(self):
        tf = self._tf(
            effective_bpm_distance=7.0,
            key_distance=3,
            key_confidence_current=0.9,
            key_confidence_candidate=0.9,
            current_vocals_rel=0.9,
            candidate_vocals_rel=0.9,
        )
        scores = {"harmonic": 0.15}
        confs = {k: 1.0 for k in STATIC_WEIGHTS}
        mult, _ = compute_penalties(tf, scores, confs, _DEFAULT_CONFIG)
        max_total = _DEFAULT_CONFIG["penalties"]["max_total_penalty"]
        assert mult >= 1.0 - max_total

    def test_single_penalty_reduced(self):
        tf = self._tf(effective_bpm_distance=6.0)
        scores = {"harmonic": 0.5}
        confs = {k: 1.0 for k in STATIC_WEIGHTS}
        mult, factors = compute_penalties(tf, scores, confs, _DEFAULT_CONFIG)
        # Single-factor penalty is multiplied by 0.6
        assert "compound_note" in factors[0]


# ---------------------------------------------------------------------------
# filter_candidates
# ---------------------------------------------------------------------------


class TestFilterCandidates:
    def test_removes_same_track(self):
        current = _track(track_id="t1", bpm=128)
        candidates = [_track(track_id="t1", bpm=128)]
        result = filter_candidates(current, candidates, [], _DEFAULT_CONFIG)
        assert result == []

    def test_removes_bpm_over_hard_limit(self):
        current = _track(track_id="t1", bpm=128)
        far = _track(track_id="t2", bpm=140)  # dist = 12 > 8
        result = filter_candidates(current, [far], [], _DEFAULT_CONFIG)
        assert result == []

    def test_passes_bpm_within_hard_limit(self):
        current = _track(track_id="t1", bpm=128)
        close = _track(track_id="t2", bpm=133)  # dist = 5 < 8
        result = filter_candidates(current, [close], [], _DEFAULT_CONFIG)
        assert len(result) == 1

    def test_removes_cooldown_repeat(self):
        current = _track(track_id="t1", bpm=128)
        candidate = _track(track_id="t2", bpm=128)
        history = [{"track_id": "t2", "energy_rel": 0.5}] * 3
        result = filter_candidates(current, [candidate], history, _DEFAULT_CONFIG)
        assert result == []

    def test_cooldown_zero_disables_recent_history_filter(self):
        current = _track(track_id="t1", bpm=128)
        candidate = _track(track_id="t2", bpm=128)
        history = [{"track_id": "t2", "energy_rel": 0.5}] * 3
        cfg = {
            **_DEFAULT_CONFIG,
            "thresholds": {**_DEFAULT_CONFIG["thresholds"], "cooldown_window": 0},
        }
        result = filter_candidates(current, [candidate], history, cfg)
        assert len(result) == 1

    def test_bypass_bpm_filter(self):
        current = _track(track_id="t1", bpm=128)
        far = _track(track_id="t2", bpm=160)
        result = filter_candidates(
            current, [far], [], _DEFAULT_CONFIG, bypass_filters={"bpm"}
        )
        assert len(result) == 1

    def test_key_not_filtered(self):
        current = _track(track_id="t1", bpm=128, key="1A")
        candidate = _track(track_id="t2", bpm=128, key="6B")  # mismatch key
        result = filter_candidates(current, [candidate], [], _DEFAULT_CONFIG)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# classify_move
# ---------------------------------------------------------------------------


class TestClassifyMove:
    def test_maintain(self):
        name, conf, _ = classify_move(0.0, 0.0, 0.2, 0.2, None, _DEFAULT_CONFIG)
        assert name == "maintain"
        assert conf == 0.90

    def test_build(self):
        name, conf, _ = classify_move(0.08, 0.0, 0.2, 0.2, None, _DEFAULT_CONFIG)
        assert name == "build"

    def test_jump(self):
        name, conf, _ = classify_move(0.20, 0.0, 0.2, 0.2, None, _DEFAULT_CONFIG)
        assert name == "jump"

    def test_reset(self):
        name, conf, _ = classify_move(-0.20, 0.0, 0.2, 0.65, None, _DEFAULT_CONFIG)
        assert name == "reset"

    def test_reset_reframe_path(self):
        name, conf, note = classify_move(-0.06, -0.10, 0.2, 0.2, None, _DEFAULT_CONFIG)
        assert name == "reset"
        assert conf == 0.75
        assert note == "reframe"

    def test_reset_vocal_reframe_path_uses_config_threshold(self):
        name, conf, note = classify_move(-0.06, 0.0, 0.2, 0.65, None, _DEFAULT_CONFIG)
        assert name == "reset"
        assert conf == 0.75
        assert note == "vocal reframe"

    def test_reset_vocal_reframe_requires_actual_vocal_increase(self):
        name, conf, note = classify_move(-0.06, 0.0, 0.7, 0.72, None, _DEFAULT_CONFIG)
        assert name != "reset" or note != "vocal reframe"

    def test_drop(self):
        name, conf, _ = classify_move(-0.08, 0.0, 0.2, 0.2, None, _DEFAULT_CONFIG)
        assert name == "drop"

    def test_slight_dip_reduced_confidence(self):
        # The "slight dip" fallback fires when delta is in the dead zone between maintain_t and
        # drop_t. With custom thresholds where drop_t > maintain_t, the zone (-maintain_t, -drop_t)
        # is reachable. Use config with drop_t = -0.10 and maintain_t = 0.05 → delta=-0.07 falls
        # in (-0.05, -0.10) = slight dip zone.
        cfg = {**_DEFAULT_CONFIG, "move_types": {**_DEFAULT_CONFIG["move_types"], "drop_threshold": -0.10}}
        name, conf, note = classify_move(-0.07, 0.0, 0.2, 0.2, None, cfg)
        assert name == "maintain"
        assert conf == 0.55
        assert note == "slight dip"

    def test_threshold_scales_with_spread(self):
        # Wide-energy playlist should raise jump threshold
        name_narrow, _, _ = classify_move(0.13, 0.0, 0.2, 0.2, None, _DEFAULT_CONFIG)
        name_wide, _, _ = classify_move(0.13, 0.0, 0.2, 0.2, 0.6, _DEFAULT_CONFIG)
        # With wide spread (scale=2 clamped to 1.5), jump_threshold = 0.18 > 0.13 → not jump
        assert name_narrow == "jump"
        assert name_wide != "jump"


# ---------------------------------------------------------------------------
# compute_risk
# ---------------------------------------------------------------------------


class TestComputeRisk:
    def _tf(self, **kwargs):
        defaults = {
            "effective_bpm_distance": 0.0,
            "key_distance": 0,
            "bpm_relationship": "direct",
            "key_confidence_current": 0.9,
            "key_confidence_candidate": 0.9,
            "delta_energy_rel": 0.0,
            "current_vocals_rel": 0.2,
            "candidate_vocals_rel": 0.2,
        }
        defaults.update(kwargs)
        return defaults

    def test_clean_transition_low_risk(self):
        level, score, factors = compute_risk(self._tf(), _DEFAULT_CONFIG)
        assert level == "low"
        assert score < 0.2
        assert factors == []

    def test_high_bpm_raises_risk(self):
        tf = self._tf(effective_bpm_distance=5.0)
        level, score, factors = compute_risk(tf, _DEFAULT_CONFIG)
        assert any("BPM" in f for f in factors)
        assert score > 0.0

    def test_key_mismatch_raises_risk(self):
        tf = self._tf(key_distance=3)
        level, score, factors = compute_risk(tf, _DEFAULT_CONFIG)
        assert any("harmonic" in f.lower() for f in factors)

    def test_dual_vocal_raises_risk(self):
        tf = self._tf(current_vocals_rel=0.8, candidate_vocals_rel=0.8)
        level, score, factors = compute_risk(tf, _DEFAULT_CONFIG)
        assert any("vocal" in f.lower() for f in factors)

    def test_multiple_factors_compound(self):
        tf = self._tf(
            effective_bpm_distance=5.0,
            key_distance=3,
            current_vocals_rel=0.8,
            candidate_vocals_rel=0.8,
        )
        level, score, _ = compute_risk(tf, _DEFAULT_CONFIG)
        assert score > 0.5
        assert level == "high"

    def test_uncertain_key_adds_risk(self):
        tf = self._tf(key_confidence_current=0.3, key_confidence_candidate=0.5)
        _, _, factors = compute_risk(tf, _DEFAULT_CONFIG)
        assert any("uncertain" in f.lower() for f in factors)

    def test_creative_tempo_pivot_adds_risk(self):
        tf = self._tf(bpm_relationship="three_two")
        _, _, factors = compute_risk(tf, _DEFAULT_CONFIG)
        assert any("pivot" in f.lower() for f in factors)


# ---------------------------------------------------------------------------
# score_candidate — integration
# ---------------------------------------------------------------------------


class TestScoreCandidate:
    def test_basic_returns_expected_keys(self):
        current = _track(track_id="t1", bpm=128, key="8A", energy_rel=0.5)
        candidate = _track(track_id="t2", bpm=129, key="8A", energy_rel=0.52)
        result = score_candidate(current, candidate, [], _DEFAULT_CONFIG, None)
        for key in (
            "candidate", "raw_score", "score", "penalty_multiplier", "penalty_factors",
            "risk", "risk_score", "risk_factors", "move", "move_confidence",
            "move_note", "contrast_score", "component_scores", "transition_features",
            "confidences", "weights_used",
        ):
            assert key in result

    def test_score_between_0_and_1(self):
        current = _track(track_id="t1", bpm=128, key="8A", energy_rel=0.5)
        candidate = _track(track_id="t2", bpm=130, key="3B", energy_rel=0.7)
        result = score_candidate(current, candidate, [], _DEFAULT_CONFIG, None)
        assert 0.0 <= result["score"] <= 1.0

    def test_perfect_match_scores_high(self):
        current = _track(track_id="t1", bpm=128, key="8A", key_confidence=0.90,
                         key_agreement=1, energy_rel=0.5, bass_rel=0.5)
        candidate = _track(track_id="t2", bpm=128, key="8A", key_confidence=0.90,
                           key_agreement=1, energy_rel=0.5, bass_rel=0.5)
        result = score_candidate(current, candidate, [], _DEFAULT_CONFIG, None)
        assert result["score"] >= 0.70

    def test_key_mismatch_with_high_confidence_scores_lower(self):
        # Same BPM and energy, but mismatch key with high confidence
        base = dict(bpm=128, energy_rel=0.5, bass_rel=0.5, key_confidence=0.90,
                    key_agreement=1)
        current = _track(track_id="t1", key="1A", **base)
        good = _track(track_id="t2", key="1A", **base)
        bad = _track(track_id="t3", key="6B", **base)
        good_result = score_candidate(current, good, [], _DEFAULT_CONFIG, None)
        bad_result = score_candidate(current, bad, [], _DEFAULT_CONFIG, None)
        assert good_result["score"] > bad_result["score"]

    def test_weak_key_reduces_harmonic_weight(self):
        # Same pair, one with corroborated key, one with weak key
        curr_strong = _track(track_id="t1", bpm=128, key="1A", energy_rel=0.5,
                             bass_rel=0.5, key_confidence=0.90, key_agreement=1)
        curr_weak = _track(track_id="t1", bpm=128, key="1A", energy_rel=0.5,
                           bass_rel=0.5, key_confidence=0.20, key_agreement=None)
        cand = _track(track_id="t2", bpm=128, key="6B", energy_rel=0.5,
                      bass_rel=0.5, key_confidence=0.90, key_agreement=1)

        conf_strong = score_candidate(curr_strong, cand, [], _DEFAULT_CONFIG, None)["confidences"]["harmonic"]
        conf_weak = score_candidate(curr_weak, cand, [], _DEFAULT_CONFIG, None)["confidences"]["harmonic"]
        assert conf_strong > conf_weak

    def test_contrast_score_not_in_weighted_score_path(self):
        # contrast_score is returned but not listed in component_scores (which feed weighted score)
        current = _track(track_id="t1", bpm=128)
        candidate = _track(track_id="t2", bpm=128)
        result = score_candidate(current, candidate, [], _DEFAULT_CONFIG, None)
        assert "contrast_score" in result
        assert "contrast_score" not in result["component_scores"]

    def test_corroborated_key_moves_harmonic_used_in_scoring(self):
        # Confirm corroborated key leads to higher harmonic confidence
        current = _track(track_id="t1", bpm=128, key="8A",
                         key_confidence=0.40, key_agreement=1)
        candidate = _track(track_id="t2", bpm=128, key="8A",
                           key_confidence=0.40, key_agreement=1)
        result = score_candidate(current, candidate, [], _DEFAULT_CONFIG, None)
        assert result["confidences"]["harmonic"] == pytest.approx(1.0)

    def test_adapted_weights_used_when_available(self):
        current = _track(track_id="t1", bpm=128)
        candidate = _track(track_id="t2", bpm=128)
        adapted = {k: 1.0 / len(STATIC_WEIGHTS) for k in STATIC_WEIGHTS}
        playlist_stats = {"adapted_weights": adapted, "energy_spread": 0.3}
        result = score_candidate(
            current, candidate, [], _DEFAULT_CONFIG, playlist_stats
        )
        assert result["weights_used"] == adapted

    def test_move_classify_in_result(self):
        current = _track(track_id="t1", bpm=128, energy_rel=0.4)
        candidate = _track(track_id="t2", bpm=128, energy_rel=0.6)
        result = score_candidate(current, candidate, [], _DEFAULT_CONFIG, None)
        assert result["move"] in ("maintain", "build", "jump", "reset", "drop")

    def test_no_history_history_fit_is_1(self):
        current = _track(track_id="t1", bpm=128)
        candidate = _track(track_id="t2", bpm=128)
        result = score_candidate(current, candidate, [], _DEFAULT_CONFIG, None)
        assert result["component_scores"]["history_fit"] == pytest.approx(1.0)
