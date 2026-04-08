"""
Phase 5 explanation engine tests.

Covers:
- generate_reasons: language policy, budget, all condition branches
- generate_window_advisory: level escalation, null-window grace
- generate_handoff_narrative: low-end stacking, vocal clash, null-window grace
- generate_outro_summary: part assembly, null-window grace
- compute_set_trend: all direction labels
- describe_character_shift: role-hint transitions
- generate_relative_context_notes: all dimensions
- generate_key_neighborhood_text: all compat labels
- generate_session_notes: history depth, outcome tracking
- generate_tempo_key_summary: direct/ratio/unknown paths
- track_recommendation_outcome: found/not-found, position, higher-scored-lanes
- build_live_candidate_explanation: budget caps, None-window grace
- build_track_detail_explanation: None-window grace
- Language policy: no forbidden phrases in any output
"""
from __future__ import annotations

import re

import pytest

from cuemate_analysis.explanations import (
    COMPACT_MAX_REASONS,
    COMPACT_MAX_WATCHOUTS,
    EXPANDED_MAX_WATCH,
    EXPANDED_MAX_WHY,
    build_live_candidate_explanation,
    build_track_detail_explanation,
    compute_set_trend,
    describe_character_shift,
    generate_handoff_narrative,
    generate_key_neighborhood_text,
    generate_outro_summary,
    generate_reasons,
    generate_relative_context_notes,
    generate_session_notes,
    generate_tempo_key_summary,
    generate_window_advisory,
    track_recommendation_outcome,
)

# ---------------------------------------------------------------------------
# Language policy enforcement
# ---------------------------------------------------------------------------

FORBIDDEN_PHRASES = [
    r"\bguaranteed smooth blend\b",
    r"\bwrong key\b",
    r"\bunsafe\b",
    r"\bdo not mix\b",
    r"\bbest track\b",
    r"\bcorrect transition\b",
]


def _all_text(obj) -> str:
    """Recursively extract all string values from nested dicts/lists."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return " ".join(_all_text(i) for i in obj)
    if isinstance(obj, dict):
        return " ".join(_all_text(v) for v in obj.values())
    return ""


def _assert_language_policy(obj) -> None:
    text = _all_text(obj)
    for pattern in FORBIDDEN_PHRASES:
        assert not re.search(pattern, text, re.IGNORECASE), (
            f"Forbidden phrase matched by pattern '{pattern}' in: {text!r}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tf(**kwargs) -> dict:
    """Build a transition_features dict with sensible defaults."""
    return {
        "delta_energy_rel": 0.0,
        "delta_bass_rel": 0.0,
        "bpm_distance": 0.0,
        "bpm_relationship": "direct",
        "key_compat_label": "adjacent",
        **kwargs,
    }


def _scores(**kwargs) -> dict:
    return {
        "harmonic": 0.85,
        "tempo": 0.80,
        "target_energy": 0.70,
        "bass_transition": 0.60,
        "history_fit": 0.50,
        "transition_support": 0.50,
        "rhythmic_continuity": 0.50,
        **kwargs,
    }


def _window(**kwargs) -> dict:
    return {
        "cleanliness_abs": 0.80,
        "bass_abs": 0.30,
        "vocals_abs": 0.10,
        "early_vocal_entry_seconds": None,
        "low_end_occupancy": 0.20,
        **kwargs,
    }


def _track(**kwargs) -> dict:
    return {
        "bpm": 128.0,
        "key": "8B",
        "key_confidence": 0.90,
        "role_hints": [],
        "vocals_rel": 0.3,
        "bass_rel": 0.3,
        "energy_rel": 0.0,
        "move": "maintain",
        **kwargs,
    }


# ---------------------------------------------------------------------------
# generate_reasons
# ---------------------------------------------------------------------------

class TestGenerateReasons:
    def test_jump_move_adds_contrast_note(self):
        reasons = generate_reasons(_tf(), _scores(), "jump")
        assert any("contrast" in r.lower() for r in reasons)

    def test_high_harmonic_score_friendly(self):
        reasons = generate_reasons(_tf(key_compat_label="adjacent"), _scores(harmonic=0.90), "maintain")
        assert any("harmonically friendly" in r.lower() for r in reasons)

    def test_low_harmonic_score_tension(self):
        reasons = generate_reasons(_tf(), _scores(harmonic=0.20), "maintain")
        assert any("harmonic tension" in r.lower() for r in reasons)

    def test_mid_harmonic_score_no_harmonic_note(self):
        reasons = generate_reasons(_tf(), _scores(harmonic=0.55), "maintain")
        harmonic_notes = [r for r in reasons if "harmonic" in r.lower()]
        assert len(harmonic_notes) == 0

    def test_big_energy_uplift(self):
        reasons = generate_reasons(_tf(delta_energy_rel=0.15), _scores(), "maintain")
        assert any("big energy uplift" in r.lower() for r in reasons)

    def test_moderate_build(self):
        reasons = generate_reasons(_tf(delta_energy_rel=0.07), _scores(), "maintain")
        assert any("builds momentum" in r.lower() for r in reasons)

    def test_energy_drop_breathing_room(self):
        reasons = generate_reasons(_tf(delta_energy_rel=-0.10), _scores(), "maintain")
        assert any("breathing room" in r.lower() for r in reasons)

    def test_steady_energy(self):
        reasons = generate_reasons(_tf(delta_energy_rel=0.02), _scores(), "maintain")
        assert any("steady" in r.lower() for r in reasons)

    def test_heavy_bass(self):
        reasons = generate_reasons(_tf(delta_bass_rel=0.20), _scores(), "maintain")
        assert any("heavier low end" in r.lower() for r in reasons)

    def test_light_bass(self):
        reasons = generate_reasons(_tf(delta_bass_rel=-0.20), _scores(), "maintain")
        assert any("lighter bass" in r.lower() for r in reasons)

    def test_creative_tempo_pivot(self):
        reasons = generate_reasons(_tf(bpm_relationship="three_two"), _scores(), "maintain")
        assert any("creative tempo pivot" in r.lower() for r in reasons)

    def test_direct_tempo_no_pivot_note(self):
        reasons = generate_reasons(_tf(bpm_relationship="direct"), _scores(), "maintain")
        assert not any("creative tempo pivot" in r.lower() for r in reasons)

    def test_double_tempo_no_pivot_note(self):
        reasons = generate_reasons(_tf(bpm_relationship="double"), _scores(), "maintain")
        assert not any("creative tempo pivot" in r.lower() for r in reasons)

    def test_clean_intro_transition_support(self):
        reasons = generate_reasons(_tf(), _scores(transition_support=0.80), "maintain")
        assert any("cleaner incoming intro profile" in r.lower() for r in reasons)

    def test_dense_transition_support(self):
        reasons = generate_reasons(_tf(), _scores(transition_support=0.20), "maintain")
        assert any("denser handoff" in r.lower() for r in reasons)

    def test_returns_list(self):
        result = generate_reasons(_tf(), _scores(), "maintain")
        assert isinstance(result, list)

    def test_language_policy(self):
        result = generate_reasons(_tf(bpm_relationship="four_three"), _scores(harmonic=0.10), "jump")
        _assert_language_policy(result)


# ---------------------------------------------------------------------------
# generate_window_advisory
# ---------------------------------------------------------------------------

class TestGenerateWindowAdvisory:
    def test_none_window_returns_none(self):
        assert generate_window_advisory(None, "intro_32") is None

    def test_open_section_green(self):
        result = generate_window_advisory(_window(cleanliness_abs=0.90), "intro_32")
        assert result["level"] == "green"
        assert any("open" in n.lower() for n in result["notes"])

    def test_dense_section_yellow(self):
        result = generate_window_advisory(_window(cleanliness_abs=0.50), "outro_32")
        assert result["level"] == "yellow"

    def test_crowded_section_orange(self):
        result = generate_window_advisory(_window(cleanliness_abs=0.20), "intro_32")
        assert result["level"] == "orange"

    def test_bass_carrying_outro_escalates(self):
        result = generate_window_advisory(
            _window(cleanliness_abs=0.90, bass_abs=0.70), "outro_32"
        )
        assert result["level"] in ("yellow", "orange")
        assert any("bass" in n.lower() for n in result["notes"])

    def test_bass_intro_escalates(self):
        result = generate_window_advisory(
            _window(cleanliness_abs=0.90, bass_abs=0.70), "intro_32"
        )
        assert result["level"] in ("yellow", "orange")

    def test_light_bass_intro_note(self):
        result = generate_window_advisory(
            _window(cleanliness_abs=0.90, bass_abs=0.10), "intro_32"
        )
        assert any("light bass" in n.lower() for n in result["notes"])

    def test_early_vocal_entry_seconds(self):
        result = generate_window_advisory(
            _window(cleanliness_abs=0.90, vocals_abs=0.70, early_vocal_entry_seconds=4.0),
            "intro_32",
        )
        assert any("enters early" in n.lower() for n in result["notes"])

    def test_vocal_no_seconds(self):
        result = generate_window_advisory(
            _window(cleanliness_abs=0.90, vocals_abs=0.70, early_vocal_entry_seconds=None),
            "intro_32",
        )
        assert any("vocal content present" in n.lower() for n in result["notes"])

    def test_long_intro_bass_note(self):
        result = generate_window_advisory(
            _window(cleanliness_abs=0.90, bass_abs=0.50), "intro_64"
        )
        assert any("extended overlap" in n.lower() for n in result["notes"])

    def test_language_policy(self):
        result = generate_window_advisory(_window(cleanliness_abs=0.20, bass_abs=0.80), "intro_32")
        _assert_language_policy(result)


# ---------------------------------------------------------------------------
# generate_handoff_narrative
# ---------------------------------------------------------------------------

class TestGenerateHandoffNarrative:
    def test_both_none_returns_none(self):
        assert generate_handoff_narrative(None, None) is None

    def test_current_none_returns_none(self):
        assert generate_handoff_narrative(None, _window()) is None

    def test_candidate_none_returns_none(self):
        assert generate_handoff_narrative(_window(), None) is None

    def test_heavy_low_end_stacking_orange(self):
        outro = _window(low_end_occupancy=0.80)
        intro = _window(bass_abs=0.70)
        result = generate_handoff_narrative(outro, intro)
        assert result["level"] == "orange"
        assert result["low_end_stacking"] > 0.4

    def test_medium_low_end_yellow(self):
        outro = _window(low_end_occupancy=0.60)
        intro = _window(bass_abs=0.45)
        result = generate_handoff_narrative(outro, intro)
        assert result["level"] in ("yellow", "orange")

    def test_high_vocal_clash(self):
        outro = _window(vocals_abs=0.70)
        intro = _window(vocals_abs=0.70)
        result = generate_handoff_narrative(outro, intro)
        assert result["vocal_clash"] > 0.25
        assert any("vocal content" in n.lower() for n in result["notes"])

    def test_vocal_fade_note(self):
        outro = _window(vocals_abs=0.70)
        intro = _window(vocals_abs=0.05)
        result = generate_handoff_narrative(outro, intro)
        assert any("outgoing vocals" in n.lower() for n in result["notes"])

    def test_vocal_arrives_note(self):
        outro = _window(vocals_abs=0.05)
        intro = _window(vocals_abs=0.70)
        result = generate_handoff_narrative(outro, intro)
        assert any("vocal content arrives" in n.lower() for n in result["notes"])

    def test_clean_handoff_both_open(self):
        outro = _window(cleanliness_abs=0.90, low_end_occupancy=0.10, vocals_abs=0.05)
        intro = _window(cleanliness_abs=0.90, bass_abs=0.10, vocals_abs=0.05)
        result = generate_handoff_narrative(outro, intro)
        assert result["level"] == "green"
        assert any("open" in n.lower() or "clean" in n.lower() for n in result["notes"])

    def test_blend_space_computed(self):
        outro = _window(cleanliness_abs=0.80)
        intro = _window(cleanliness_abs=0.60)
        result = generate_handoff_narrative(outro, intro)
        assert result["blend_space"] == pytest.approx(0.70)

    def test_language_policy(self):
        outro = _window(low_end_occupancy=0.90, vocals_abs=0.80)
        intro = _window(bass_abs=0.90, vocals_abs=0.80)
        result = generate_handoff_narrative(outro, intro)
        _assert_language_policy(result)


# ---------------------------------------------------------------------------
# generate_outro_summary
# ---------------------------------------------------------------------------

class TestGenerateOutroSummary:
    def test_none_returns_none(self):
        assert generate_outro_summary(None) is None

    def test_bass_heavy(self):
        result = generate_outro_summary(_window(bass_abs=0.80))
        assert "bass heavy" in result["text"]

    def test_bass_present(self):
        result = generate_outro_summary(_window(bass_abs=0.50))
        assert "bass present" in result["text"]

    def test_light_bass(self):
        result = generate_outro_summary(_window(bass_abs=0.20))
        assert "light bass" in result["text"]

    def test_no_vocals(self):
        result = generate_outro_summary(_window(vocals_abs=0.05))
        assert "no vocals" in result["text"]

    def test_vocals_present(self):
        result = generate_outro_summary(_window(vocals_abs=0.70))
        assert "vocals present" in result["text"]

    def test_vocals_unknown(self):
        w = _window()
        w["vocals_abs"] = None
        result = generate_outro_summary(w)
        assert "vocals unknown" in result["text"]

    def test_open_section(self):
        result = generate_outro_summary(_window(cleanliness_abs=0.80))
        assert "open" in result["text"]

    def test_busy_section(self):
        result = generate_outro_summary(_window(cleanliness_abs=0.20))
        assert "busy" in result["text"]

    def test_cleanliness_abs_in_output(self):
        result = generate_outro_summary(_window(cleanliness_abs=0.75))
        assert result["cleanliness_abs"] == pytest.approx(0.75)

    def test_separator(self):
        result = generate_outro_summary(_window())
        assert " · " in result["text"]


# ---------------------------------------------------------------------------
# compute_set_trend
# ---------------------------------------------------------------------------

class TestComputeSetTrend:
    def test_single_item_just_started(self):
        result = compute_set_trend([{"energy_rel": 0.5}])
        assert result["direction"] == "unknown"

    def test_empty_just_started(self):
        result = compute_set_trend([])
        assert result["direction"] == "unknown"

    def test_two_items_building(self):
        result = compute_set_trend([{"energy_rel": 0.3}, {"energy_rel": 0.5}])
        assert result["direction"] == "up"

    def test_two_items_dropping(self):
        result = compute_set_trend([{"energy_rel": 0.7}, {"energy_rel": 0.4}])
        assert result["direction"] == "down"

    def test_two_items_steady(self):
        result = compute_set_trend([{"energy_rel": 0.5}, {"energy_rel": 0.52}])
        assert result["direction"] == "flat"

    def test_building_steadily(self):
        history = [{"energy_rel": v} for v in [0.3, 0.4, 0.5, 0.6, 0.7]]
        result = compute_set_trend(history)
        assert result["direction"] == "up"
        assert "building" in result["label"]

    def test_winding_down(self):
        history = [{"energy_rel": v} for v in [0.8, 0.7, 0.6, 0.5, 0.4]]
        result = compute_set_trend(history)
        assert result["direction"] in ("down", "down_from_peak")

    def test_peaked_easing(self):
        history = [{"energy_rel": v} for v in [0.3, 0.5, 0.9, 0.8, 0.7]]
        result = compute_set_trend(history)
        assert result["direction"] in ("down", "down_from_peak")

    def test_steady_long(self):
        history = [{"energy_rel": 0.5 + i * 0.005} for i in range(6)]
        result = compute_set_trend(history)
        assert result["direction"] == "flat"


# ---------------------------------------------------------------------------
# describe_character_shift
# ---------------------------------------------------------------------------

class TestDescribeCharacterShift:
    def _shift(self, cur_roles=None, cand_roles=None, **kwargs):
        defaults = dict(
            current_vocals_rel=0.3,
            candidate_vocals_rel=0.3,
            current_bass_rel=0.3,
            candidate_bass_rel=0.3,
            current_energy_rel=0.0,
            candidate_energy_rel=0.0,
        )
        defaults.update(kwargs)
        return describe_character_shift(
            cur_roles or [], cand_roles or [], **defaults
        )

    def test_vocal_to_vocal(self):
        notes = self._shift(["vocal_feature"], ["vocal_feature"])
        assert any("vocal → vocal" in n for n in notes)

    def test_vocal_to_instrumental(self):
        notes = self._shift(["vocal_feature"], [], candidate_vocals_rel=0.05)
        assert any("instrumental" in n for n in notes)

    def test_instrumental_to_vocal(self):
        notes = self._shift([], ["vocal_feature"])
        assert any("instrumental → vocal" in n for n in notes)

    def test_both_instrumental_low_vocals(self):
        notes = self._shift([], [], current_vocals_rel=0.05, candidate_vocals_rel=0.05)
        assert any("both mostly instrumental" in n for n in notes)

    def test_bass_driver_continues(self):
        notes = self._shift(["bass_driver"], ["bass_driver"])
        assert any("bass-driver character continues" in n for n in notes)

    def test_bass_driver_incoming(self):
        notes = self._shift([], ["bass_driver"])
        assert any("bass-driver character incoming" in n for n in notes)

    def test_bass_pressure_steps_back(self):
        notes = self._shift(["bass_driver"], [])
        assert any("bass pressure steps back" in n for n in notes)

    def test_stepping_up_to_peak(self):
        notes = self._shift([], ["peak_tool"])
        assert any("stepping up to peak pressure" in n for n in notes)

    def test_peak_to_breather(self):
        notes = self._shift(["peak_tool"], ["opener"])
        assert any("peak → breather" in n for n in notes)

    def test_opener_to_peak_stepping_up(self):
        # opener → peak_tool: cur_peak=False, cand_peak=True → "stepping up to peak pressure"
        # (the "big jump" branch is unreachable because the prior elif fires first)
        notes = self._shift(["opener"], ["peak_tool"])
        assert any("stepping up to peak pressure" in n for n in notes)

    def test_empty_roles_no_crash(self):
        notes = self._shift([], [])
        assert isinstance(notes, list)

    def test_language_policy(self):
        notes = self._shift(["vocal_feature", "peak_tool"], ["bass_driver"])
        _assert_language_policy(notes)


# ---------------------------------------------------------------------------
# generate_relative_context_notes
# ---------------------------------------------------------------------------

class TestGenerateRelativeContextNotes:
    def test_high_energy(self):
        notes = generate_relative_context_notes({"energy_rel": 0.90})
        assert any("higher energy" in n.lower() for n in notes)

    def test_above_average_energy(self):
        notes = generate_relative_context_notes({"energy_rel": 0.75})
        assert any("above-average energy" in n.lower() for n in notes)

    def test_low_energy(self):
        notes = generate_relative_context_notes({"energy_rel": 0.10})
        assert any("lower energy" in n.lower() for n in notes)

    def test_below_average_energy(self):
        notes = generate_relative_context_notes({"energy_rel": 0.22})
        assert any("below-average energy" in n.lower() for n in notes)

    def test_mid_range_no_note(self):
        notes = generate_relative_context_notes({"energy_rel": 0.50})
        energy_notes = [n for n in notes if "energy" in n.lower()]
        assert len(energy_notes) == 0

    def test_mostly_instrumental_note(self):
        notes = generate_relative_context_notes({"vocals_rel": 0.10})
        assert any("mostly instrumental" in n.lower() for n in notes)

    def test_vocal_heavy_note(self):
        notes = generate_relative_context_notes({"vocals_rel": 0.85})
        assert any("vocal-heavy" in n.lower() for n in notes)

    def test_none_values_skipped(self):
        notes = generate_relative_context_notes({"energy_rel": None, "bass_rel": None})
        assert isinstance(notes, list)

    def test_all_dimensions_covered(self):
        notes = generate_relative_context_notes({
            "energy_rel": 0.95,
            "bass_rel": 0.95,
            "drums_rel": 0.05,
            "vocals_rel": 0.95,
            "groove_rel": 0.05,
        })
        assert len(notes) > 3

    def test_language_policy(self):
        notes = generate_relative_context_notes({"energy_rel": 0.95, "vocals_rel": 0.85})
        _assert_language_policy(notes)


# ---------------------------------------------------------------------------
# generate_key_neighborhood_text
# ---------------------------------------------------------------------------

class TestGenerateKeyNeighborhoodText:
    @pytest.mark.parametrize("label,expected_short_fragment", [
        ("perfect", "same key"),
        ("relative_key", "relative major/minor"),
        ("adjacent", "adjacent on the wheel"),
        ("cross_adjacent", "cross-adjacent"),
        ("energy_boost", "energy-boost"),
        ("mismatch", "harmonic tension"),
    ])
    def test_short_text(self, label, expected_short_fragment):
        result = generate_key_neighborhood_text(label, 2, "8B", "9B")
        assert expected_short_fragment.lower() in result["short"].lower()

    def test_mismatch_direction_includes_distance(self):
        result = generate_key_neighborhood_text("mismatch", 4, "1A", "5B")
        assert "4" in result["direction"]

    def test_unknown_label_returns_empty_strings(self):
        result = generate_key_neighborhood_text("completely_unknown", 0, "1A", "1A")
        assert result["short"] == ""
        assert result["direction"] == ""

    def test_language_policy(self):
        for label in ("perfect", "relative_key", "adjacent", "cross_adjacent",
                      "energy_boost", "mismatch"):
            result = generate_key_neighborhood_text(label, 3, "8B", "9A")
            _assert_language_policy(result)


# ---------------------------------------------------------------------------
# generate_session_notes
# ---------------------------------------------------------------------------

class TestGenerateSessionNotes:
    def test_no_history(self):
        notes = generate_session_notes(0, False)
        assert any("no history" in n.lower() for n in notes)

    def test_first_track(self):
        notes = generate_session_notes(1, False)
        assert any("first track" in n.lower() for n in notes)

    def test_gaps_in_history(self):
        notes = generate_session_notes(5, True)
        assert any("gaps" in n.lower() for n in notes)

    def test_no_gaps_no_gap_note(self):
        notes = generate_session_notes(5, False)
        assert not any("gaps" in n.lower() for n in notes)

    def test_outcome_was_recommended(self):
        outcome = {"was_recommended": True, "position": 2, "lane": "build", "higher_scored_lanes": []}
        notes = generate_session_notes(5, False, outcome)
        assert any("picked" in n.lower() for n in notes)

    def test_outcome_not_recommended(self):
        outcome = {"was_recommended": False, "position": None, "lane": None, "higher_scored_lanes": ["maintain"]}
        notes = generate_session_notes(5, False, outcome)
        assert any("not in recommendations" in n.lower() for n in notes)

    def test_skipped_lanes_noted(self):
        outcome = {"was_recommended": True, "position": 1, "lane": "build",
                   "higher_scored_lanes": ["maintain"]}
        notes = generate_session_notes(5, False, outcome)
        assert any("skipped" in n.lower() for n in notes)

    def test_language_policy(self):
        outcome = {"was_recommended": False, "position": None, "lane": None, "higher_scored_lanes": []}
        notes = generate_session_notes(3, True, outcome)
        _assert_language_policy(notes)


# ---------------------------------------------------------------------------
# generate_tempo_key_summary
# ---------------------------------------------------------------------------

class TestGenerateTempoKeySummary:
    def test_exact_bpm_match(self):
        result = generate_tempo_key_summary(128.0, 128.0, "8B", "8B")
        assert "0 BPM" in result["tempo_text"]
        assert result["key_state"] == "normal"

    def test_small_bpm_diff_easy(self):
        result = generate_tempo_key_summary(128.0, 129.5, "8B", "8B")
        assert "easy" in result["tempo_text"]

    def test_moderate_bpm_diff_push(self):
        result = generate_tempo_key_summary(128.0, 131.0, "8B", "8B")
        assert "push" in result["tempo_text"]

    def test_large_bpm_diff_shift(self):
        result = generate_tempo_key_summary(128.0, 136.0, "8B", "8B")
        assert "shift" in result["tempo_text"]

    def test_double_time(self):
        result = generate_tempo_key_summary(64.0, 128.0, "8B", "8B")
        assert "double" in result["tempo_text"].lower() or "time" in result["tempo_text"]

    def test_half_time(self):
        result = generate_tempo_key_summary(128.0, 64.0, "8B", "8B")
        assert "half" in result["tempo_text"].lower() or "time" in result["tempo_text"]

    def test_creative_pivot(self):
        result = generate_tempo_key_summary(128.0, 85.3, "8B", "8B")
        assert "pivot" in result["tempo_text"].lower()

    def test_unknown_key(self):
        result = generate_tempo_key_summary(128.0, 128.0, None, "8B")
        assert result["key_state"] == "unknown"
        assert "unknown" in result["key_text"]

    def test_low_confidence_uncertain(self):
        result = generate_tempo_key_summary(128.0, 128.0, "8B", "8B",
                                            current_key_confidence=0.4,
                                            candidate_key_confidence=0.9)
        assert result["key_state"] == "uncertain"
        assert "uncertain" in result["key_text"]

    def test_high_confidence_normal_state(self):
        result = generate_tempo_key_summary(128.0, 128.0, "8B", "8B",
                                            current_key_confidence=0.9,
                                            candidate_key_confidence=0.9)
        assert result["key_state"] == "normal"

    def test_language_policy(self):
        result = generate_tempo_key_summary(128.0, 85.0, "1A", "7B",
                                            current_key_confidence=0.3,
                                            candidate_key_confidence=0.3)
        _assert_language_policy(result)


# ---------------------------------------------------------------------------
# track_recommendation_outcome
# ---------------------------------------------------------------------------

class TestTrackRecommendationOutcome:
    def _lane(self, track_ids_scores: list[tuple[str, float]]) -> list[dict]:
        return [{"track_id": tid, "score": s} for tid, s in track_ids_scores]

    def test_found_first_position(self):
        lanes = {"maintain": self._lane([("trk_A", 0.9), ("trk_B", 0.7)])}
        result = track_recommendation_outcome(lanes, "trk_A")
        assert result["was_recommended"] is True
        assert result["position"] == 1
        assert result["lane"] == "maintain"

    def test_found_second_position(self):
        lanes = {"maintain": self._lane([("trk_A", 0.9), ("trk_B", 0.7)])}
        result = track_recommendation_outcome(lanes, "trk_B")
        assert result["position"] == 2

    def test_not_found(self):
        lanes = {"maintain": self._lane([("trk_A", 0.9)])}
        result = track_recommendation_outcome(lanes, "trk_Z")
        assert result["was_recommended"] is False
        assert result["position"] is None
        assert result["lane"] is None
        assert "maintain" in result["higher_scored_lanes"]

    def test_higher_scored_lanes(self):
        lanes = {
            "build": self._lane([("trk_X", 0.95)]),
            "maintain": self._lane([("trk_A", 0.70)]),
        }
        result = track_recommendation_outcome(lanes, "trk_A")
        # build lane top score 0.95 > trk_A score 0.70 → build is higher-scored
        assert "build" in result["higher_scored_lanes"]

    def test_no_higher_scored_lanes(self):
        lanes = {
            "build": self._lane([("trk_A", 0.90)]),
            "maintain": self._lane([("trk_B", 0.50)]),
        }
        result = track_recommendation_outcome(lanes, "trk_A")
        # trk_A score 0.90 > maintain top 0.50 → no higher-scored lanes
        assert result["higher_scored_lanes"] == []

    def test_empty_lanes(self):
        result = track_recommendation_outcome({}, "trk_A")
        assert result["was_recommended"] is False


# ---------------------------------------------------------------------------
# build_live_candidate_explanation
# ---------------------------------------------------------------------------

class TestBuildLiveCandidateExplanation:
    def test_returns_required_keys(self):
        result = build_live_candidate_explanation(
            _track(), _track(move="maintain"), _tf(), _scores(), None, None
        )
        for key in ("summary", "why", "watch", "handoff", "tempo_key", "character_shift"):
            assert key in result

    def test_none_windows_no_crash(self):
        result = build_live_candidate_explanation(
            _track(), _track(), _tf(), _scores(), None, None
        )
        assert result["handoff"] is None

    def test_summary_budget_cap(self):
        result = build_live_candidate_explanation(
            _track(), _track(move="jump"), _tf(bpm_relationship="three_two", delta_energy_rel=0.20),
            _scores(harmonic=0.20, transition_support=0.10), None, None
        )
        assert len(result["summary"]) <= COMPACT_MAX_REASONS

    def test_why_budget_cap(self):
        result = build_live_candidate_explanation(
            _track(), _track(move="jump"), _tf(bpm_relationship="three_two", delta_energy_rel=0.20),
            _scores(harmonic=0.20, transition_support=0.10), None, None
        )
        assert len(result["why"]) <= EXPANDED_MAX_WHY

    def test_watch_budget_cap(self):
        outro = _window(low_end_occupancy=0.90, vocals_abs=0.90, cleanliness_abs=0.20)
        intro = _window(bass_abs=0.90, vocals_abs=0.90, cleanliness_abs=0.20)
        result = build_live_candidate_explanation(
            _track(), _track(), _tf(), _scores(), outro, intro
        )
        assert len(result["watch"]) <= EXPANDED_MAX_WATCH

    def test_high_handoff_risk_adds_watch(self):
        outro = _window(low_end_occupancy=0.90)
        intro = _window(bass_abs=0.90)
        result = build_live_candidate_explanation(
            _track(), _track(), _tf(), _scores(), outro, intro
        )
        assert len(result["watch"]) > 0

    def test_language_policy(self):
        result = build_live_candidate_explanation(
            _track(role_hints=["vocal_feature"]),
            _track(move="jump", role_hints=["bass_driver"]),
            _tf(bpm_relationship="four_three", delta_energy_rel=-0.15),
            _scores(harmonic=0.10),
            _window(low_end_occupancy=0.90, vocals_abs=0.80),
            _window(bass_abs=0.90, vocals_abs=0.80),
        )
        _assert_language_policy(result)


# ---------------------------------------------------------------------------
# build_track_detail_explanation
# ---------------------------------------------------------------------------

class TestBuildTrackDetailExplanation:
    def test_returns_required_keys(self):
        result = build_track_detail_explanation({}, {"energy_rel": 0.5}, {})
        for key in ("relative_context", "intro_advisory", "outro_advisory", "outro_summary"):
            assert key in result

    def test_no_windows_all_none(self):
        result = build_track_detail_explanation({}, {}, {})
        assert result["intro_advisory"] is None
        assert result["outro_advisory"] is None
        assert result["outro_summary"] is None

    def test_windows_populated(self):
        windows = {"intro_32": _window(), "outro_32": _window()}
        result = build_track_detail_explanation({}, {"energy_rel": 0.5}, windows)
        assert result["intro_advisory"] is not None
        assert result["outro_advisory"] is not None
        assert result["outro_summary"] is not None

    def test_relative_context_notes(self):
        result = build_track_detail_explanation({}, {"energy_rel": 0.95}, {})
        assert len(result["relative_context"]) > 0

    def test_language_policy(self):
        windows = {"intro_32": _window(cleanliness_abs=0.20, bass_abs=0.80),
                   "outro_32": _window(low_end_occupancy=0.90)}
        result = build_track_detail_explanation(
            {}, {"energy_rel": 0.95, "vocals_rel": 0.90}, windows
        )
        _assert_language_policy(result)
