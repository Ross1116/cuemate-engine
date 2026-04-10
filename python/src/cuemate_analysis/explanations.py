"""
Milestone 3 - Phase 5: Explanation and Deterministic Text Generation Layer

All functions are deterministic. No LLMs, no UI logic, no formatting.
Reads only from precomputed features, scores, and live session state.

Language policy (section 0.4):
  Preferred: "harmonically friendly," "higher clash risk," "easy tempo match,"
             "strong contrast option," "cleaner intro," "heavier low end,"
             "trust your ears."
  Forbidden: "guaranteed smooth blend," "wrong key," "unsafe," "avoid,"
             "best track," "correct transition," "do not mix."

Windows are always optional. Every window-dependent path returns None when the
window is unavailable. Callers must handle None gracefully.
"""
from __future__ import annotations

from typing import Any

from cuemate_analysis.scoring import (
    camelot_compatibility,
    effective_bpm_distance,
    parse_camelot,
)

# ---------------------------------------------------------------------------
# Budget constants (section 20.3.1)
# ---------------------------------------------------------------------------

COMPACT_MAX_REASONS = 3
COMPACT_MAX_WATCHOUTS = 2
EXPANDED_MAX_WHY = 5
EXPANDED_MAX_WATCH = 3
EXPANDED_MAX_HANDOFF = 1  # one handoff block per candidate


def _coalesce_numeric(value: float | None, default: float) -> float:
    return default if value is None else value


# ---------------------------------------------------------------------------
# 20.4  Core candidate reasons
# ---------------------------------------------------------------------------

def generate_reasons(
    transition_features: dict[str, Any],
    scores: dict[str, Any],
    move_type: str,
) -> list[str]:
    """Return a list of human-readable reason strings for this candidate.

    Follows the language policy. Reasons are ordered: move hint first, then
    harmonic, energy, vocal, bass, transition-support, tempo-relationship.
    The caller is responsible for applying the compact/expanded budget cap.
    """
    reasons: list[str] = []

    if move_type == "jump":
        reasons.append("Deliberate contrast move")

    harmonic = scores.get("harmonic", 0.5)
    key_label = transition_features.get("key_compat_label", "")
    if harmonic >= 0.85:
        label_text = f" ({key_label})" if key_label else ""
        reasons.append(f"Harmonically friendly{label_text}")
    elif harmonic <= 0.30:
        reasons.append("Higher harmonic tension - trust your ears on the blend")

    delta_e = transition_features.get("delta_energy_rel", 0.0) or 0.0
    if delta_e > 0.12:
        reasons.append(f"Big energy uplift (+{delta_e:.0%}) - bold move")
    elif delta_e > 0.05:
        reasons.append("Builds momentum")
    elif delta_e < -0.08:
        reasons.append("Creates breathing room")
    elif abs(delta_e) <= 0.05:
        reasons.append("Keeps pressure steady")

    vocal_score = scores.get("vocal_transition", None)
    if vocal_score is not None:
        if vocal_score < 0.3:
            reasons.append("Both tracks carry strong vocal content - watch the overlap")
        elif vocal_score > 0.85:
            reasons.append("Good vocal contrast - cleaner blend space")

    delta_b = transition_features.get("delta_bass_rel", 0.0) or 0.0
    if delta_b > 0.15:
        reasons.append("Heavier low end incoming")
    elif delta_b < -0.15:
        reasons.append("Lighter bass - opens space")

    ts = scores.get("transition_support", None)
    if ts is not None:
        if ts > 0.75:
            reasons.append("Cleaner incoming intro profile")
        elif ts < 0.3:
            reasons.append("Denser handoff - more demanding overlap")

    bpm_rel = transition_features.get("bpm_relationship", "direct")
    if bpm_rel not in ("direct", "double", "half"):
        reasons.append(f"Creative tempo pivot ({bpm_rel}) - trust your ears")

    return reasons


# ---------------------------------------------------------------------------
# 20.5  Window and handoff text primitives
# ---------------------------------------------------------------------------

def generate_window_advisory(
    window_features: dict[str, Any] | None,
    window_name: str,
) -> dict[str, Any] | None:
    """Return a single-window advisory dict or None when the window is unavailable.

    Output: {"level": "green"|"yellow"|"orange", "notes": [str]}
    """
    if window_features is None:
        return None

    clean = _coalesce_numeric(window_features.get("cleanliness_abs"), 0.5)
    bass = _coalesce_numeric(window_features.get("bass_abs"), 0.5)
    vocals = window_features.get("vocals_abs")
    early_vocal = window_features.get("early_vocal_entry_seconds")

    notes: list[str] = []
    level = "green"

    is_intro = window_name.startswith("intro")
    is_outro = window_name.startswith("outro")
    is_long = "64" in window_name

    if clean > 0.75:
        notes.append("Open section - plenty of space")
    elif clean > 0.55:
        notes.append("Reasonably open section")
    elif clean > 0.35:
        level = "yellow"
        notes.append("Denser section - more going on here")
    else:
        level = "orange"
        notes.append("Crowded section - lots happening early")

    if level == "orange" and notes and notes[-1] == "Crowded section - lots happening early" and not is_intro:
        notes[-1] = "Crowded section - lots happening late" if is_outro else "Crowded section - many events"

    if level == "orange" and notes and not is_intro:
        if is_outro:
            notes[0] = "Crowded section - lots happening late"
        else:
            notes[0] = "Crowded section - many events"

    if level == "orange" and notes and notes[-1].startswith("Crowded section"):
        if is_intro:
            notes[-1] = "Crowded section - lots happening early"
        elif is_outro:
            notes[-1] = "Crowded section - lots happening late"
        else:
            notes[-1] = "Crowded section - many events"

    if bass > 0.6 and is_outro:
        notes.append("Bass still carrying")
        if level == "green":
            level = "yellow"
    elif bass > 0.6 and is_intro:
        notes.append("Bass present from early on")
        if level == "green":
            level = "yellow"
    elif bass < 0.2 and is_intro:
        notes.append("Light bass - more overlap space")

    if vocals is not None and vocals > 0.5 and is_intro:
        if early_vocal is not None:
            notes.append(f"Vocal content enters early (~{int(early_vocal)}s)")
        else:
            notes.append("Vocal content present in this section")
        if level == "green":
            level = "yellow"

    if is_long and is_intro and bass > 0.3:
        notes.append("Bass arrives deeper into the intro - watch extended overlap")

    return {"level": level, "notes": notes}


def generate_handoff_narrative(
    current_outro: dict[str, Any] | None,
    candidate_intro: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a pairwise handoff narrative or None when either window is unavailable.

    Output: {"level": str, "notes": [str], "low_end_stacking": float,
             "vocal_clash": float, "blend_space": float}
    """
    if current_outro is None or candidate_intro is None:
        return None

    cur_vocal = current_outro.get("vocals_abs")
    cur_clean = _coalesce_numeric(current_outro.get("cleanliness_abs"), 0.5)
    cur_low = _coalesce_numeric(current_outro.get("low_end_occupancy"), 0.5)
    cand_bass = _coalesce_numeric(candidate_intro.get("bass_abs"), 0.5)
    cand_vocal = candidate_intro.get("vocals_abs")
    cand_clean = _coalesce_numeric(candidate_intro.get("cleanliness_abs"), 0.5)

    notes: list[str] = []
    level = "green"

    low_end_product = cur_low * cand_bass
    if low_end_product > 0.4:
        notes.append("Both carry heavy low end - watch the stacking")
        level = "orange"
    elif low_end_product > 0.2:
        notes.append("Some low-end overlap - still manageable")
        level = "yellow"

    vocal_product = 0.0 if cur_vocal is None or cand_vocal is None else cur_vocal * cand_vocal
    if cur_vocal is not None and cand_vocal is not None and vocal_product > 0.25:
        notes.append("Both have vocal content in the overlap zone - demanding handoff")
        if level != "orange":
            level = "yellow"
    elif cur_vocal is not None and cand_vocal is not None and cur_vocal > 0.5 and cand_vocal < 0.15:
        notes.append("Outgoing vocals fade while the incoming intro stays mostly instrumental")
    elif cur_vocal is not None and cand_vocal is not None and cur_vocal < 0.15 and cand_vocal > 0.5:
        notes.append("Vocal content arrives with the incoming track")

    if cur_clean < 0.4 and cand_clean > 0.7:
        notes.append("Incoming section is open while the current groove still carries")
    elif cur_clean > 0.7 and cand_clean < 0.4:
        notes.append("Incoming track is busy from the start over an open current outro")
    elif cur_clean < 0.4 and cand_clean < 0.4:
        notes.append("Both sections are busy - denser overlap")
        if level == "green":
            level = "yellow"
    elif cur_clean > 0.7 and cand_clean > 0.7:
        notes.append("Both sections are open - plenty of space")

    if level == "green" and not notes:
        notes.append("Clean handoff conditions")
    elif level == "orange" and len(notes) > 1:
        notes.append("More demanding overlap - trust your ears")

    return {
        "level": level,
        "notes": notes,
        "low_end_stacking": round(low_end_product, 2),
        "vocal_clash": round(vocal_product, 2),
        "blend_space": round((cur_clean + cand_clean) / 2, 2),
    }


def generate_outro_summary(
    outro_window: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a compact outro-character dict or None when unavailable.

    Output: {"text": str, "cleanliness_abs": float}
    e.g. "bass heavy · no vocals · busy"
    """
    if outro_window is None:
        return None

    parts: list[str] = []
    bass = _coalesce_numeric(outro_window.get("bass_abs"), 0.5)
    vocals = outro_window.get("vocals_abs")
    clean = _coalesce_numeric(outro_window.get("cleanliness_abs"), 0.5)

    if bass > 0.65:
        parts.append("bass heavy")
    elif bass > 0.4:
        parts.append("bass present")
    else:
        parts.append("light bass")

    if vocals is None:
        parts.append("vocals unknown")
    elif vocals > 0.5:
        parts.append("vocals present")
    elif vocals > 0.2:
        parts.append("light vocals")
    else:
        parts.append("no vocals")

    if clean > 0.65:
        parts.append("open")
    elif clean > 0.4:
        parts.append("moderate")
    else:
        parts.append("busy")

    return {
        "text": " · ".join(parts),
        "cleanliness_abs": round(clean, 2),
    }


# ---------------------------------------------------------------------------
# 20.6  Relative and session context text
# ---------------------------------------------------------------------------

def compute_set_trend(history: list[dict[str, Any]]) -> dict[str, str]:
    """Return a trend label and direction for the session history.

    Input: list of dicts each with at minimum "energy_rel" (float).
    Output: {"label": str, "direction": str}
    """
    if len(history) < 2:
        return {"label": "just started", "direction": "unknown"}

    energies = [
        _coalesce_numeric(h.get("energy_rel"), 0.5)
        for h in history
        if h.get("energy_rel") is not None
    ]
    if len(energies) < 2:
        return {"label": "just started", "direction": "unknown"}

    if len(energies) < 3:
        delta = energies[-1] - energies[0]
        if delta > 0.08:
            return {"label": "building", "direction": "up"}
        elif delta < -0.08:
            return {"label": "dropping", "direction": "down"}
        else:
            return {"label": "steady", "direction": "flat"}

    recent_3 = energies[-3:]
    deltas = [recent_3[i + 1] - recent_3[i] for i in range(len(recent_3) - 1)]
    avg_delta = sum(deltas) / len(deltas)
    overall_delta = energies[-1] - energies[0]

    if avg_delta > 0.05 and overall_delta > 0.1:
        return {"label": "building steadily", "direction": "up"}
    elif avg_delta > 0.05:
        return {"label": "building", "direction": "up"}
    elif avg_delta < -0.05 and overall_delta < -0.1:
        return {"label": "winding down", "direction": "down"}
    elif avg_delta < -0.05:
        return {"label": "easing", "direction": "down"}
    elif max(energies) - energies[-1] > 0.15 and energies[-1] < max(energies):
        return {"label": "peaked, easing", "direction": "down_from_peak"}
    elif abs(avg_delta) < 0.03:
        return {"label": "steady", "direction": "flat"}
    else:
        return {"label": "mixed", "direction": "mixed"}


def describe_character_shift(
    current_roles: list[str],
    candidate_roles: list[str],
    current_vocals_rel: float | None,
    candidate_vocals_rel: float | None,
    current_bass_rel: float | None,
    candidate_bass_rel: float | None,
    current_energy_rel: float | None,
    candidate_energy_rel: float | None,
) -> list[str]:
    """Return character-shift notes comparing current and candidate track roles.

    Input: role_hints lists and _rel features for both tracks.
    Output: list of str - each is a character shift note.
    """
    notes: list[str] = []
    cur = set(current_roles or [])
    cand = set(candidate_roles or [])

    cur_vocal_feat = "vocal_feature" in cur
    cand_vocal_feat = "vocal_feature" in cand

    if cur_vocal_feat and cand_vocal_feat:
        notes.append("vocal → vocal transition")
    elif cur_vocal_feat and not cand_vocal_feat:
        if candidate_vocals_rel is not None and candidate_vocals_rel < 0.15:
            notes.append("vocal → instrumental shift")
        else:
            notes.append("vocal content drops off")
    elif not cur_vocal_feat and cand_vocal_feat:
        notes.append("instrumental → vocal shift")
    else:
        if (
            current_vocals_rel is not None
            and candidate_vocals_rel is not None
            and current_vocals_rel < 0.15
            and candidate_vocals_rel < 0.15
        ):
            notes.append("both mostly instrumental")

    cur_bass_driver = "bass_driver" in cur
    cand_bass_driver = "bass_driver" in cand
    if cur_bass_driver and cand_bass_driver:
        notes.append("bass-driver character continues")
    elif not cur_bass_driver and cand_bass_driver:
        notes.append("bass-driver character incoming")
    elif cur_bass_driver and not cand_bass_driver:
        notes.append("bass pressure steps back")

    cur_peak = "peak_tool" in cur
    cand_peak = "peak_tool" in cand
    cand_opener = "opener" in cand or "relief_track" in cand
    cur_opener = "opener" in cur or "relief_track" in cur

    if cur_opener and cand_peak:
        notes.append("big jump from low to peak")
    elif not cur_peak and cand_peak:
        notes.append("stepping up to peak pressure")
    elif cur_peak and cand_opener:
        notes.append("peak → breather shift")

    return notes


def generate_relative_context_notes(track_rel_features: dict[str, Any]) -> list[str]:
    """Return playlist-relative placement notes for one track.

    Input: dict of relative features (energy_rel, bass_rel, drums_rel, vocals_rel, groove_rel).
    Output: list of str.
    """
    notes: list[str] = []
    DIMENSIONS = {
        "energy_rel": "energy",
        "bass_rel": "bass",
        "drums_rel": "drum energy",
        "vocals_rel": "vocal content",
        "groove_rel": "groove",
    }
    for key, name in DIMENSIONS.items():
        val = track_rel_features.get(key)
        if val is None:
            continue
        if val > 0.85:
            notes.append(f"Higher {name} than most of your playlist")
        elif val > 0.70:
            notes.append(f"Above-average {name} for this playlist")
        elif val < 0.15:
            notes.append(f"Lower {name} than most of your playlist")
        elif val < 0.30:
            notes.append(f"Below-average {name} for this playlist")

    vocals_rel = track_rel_features.get("vocals_rel")
    if vocals_rel is not None:
        if vocals_rel < 0.15:
            notes.append("Mostly instrumental - helpful contrast against vocal-heavy material")
        elif vocals_rel > 0.80:
            notes.append("Vocal-heavy - watch overlap with other vocal tracks")

    return notes


def generate_key_neighborhood_text(
    compat_label: str,
    distance: int,
    current_key: str,
    candidate_key: str,
) -> dict[str, str]:
    """Return short and direction key-relationship descriptions.

    Output: {"short": str, "direction": str}
    """
    SHORT: dict[str, str] = {
        "perfect": "Same key - tonally matched",
        "relative_key": "Relative major/minor - harmonically friendly",
        "adjacent": "Adjacent on the wheel - harmonically friendly",
        "cross_adjacent": "Cross-adjacent - usually works well",
        "energy_boost": "Energy-boost key change - two steps on the wheel",
        "energy_key_change": "Dominant key change - can work for energy shifts",
        "mismatch": "Higher harmonic tension - trust your ears on the blend",
    }
    DIRECTION: dict[str, str] = {
        "perfect": "Same position on the Camelot wheel",
        "relative_key": "Same number, different mode (major/minor)",
        "adjacent": "One step on the wheel",
        "cross_adjacent": "One step, crossing major/minor",
        "energy_boost": "Two steps on the wheel",
        "energy_key_change": "Dominant relationship",
        "mismatch": f"Distance {distance} on the wheel",
    }
    return {
        "short": SHORT.get(compat_label, ""),
        "direction": DIRECTION.get(compat_label, ""),
    }


def generate_session_notes(
    history_length: int,
    has_gaps: bool,
    last_recommendation_outcome: dict[str, Any] | None = None,
) -> list[str]:
    """Return session-state notes based on history depth and last outcome.

    Output: list of str.
    """
    notes: list[str] = []

    if history_length == 0:
        notes.append(
            "No history yet - recommendations are based on this track and playlist character only"
        )
    elif history_length == 1:
        notes.append("First track of the set")
    elif has_gaps:
        notes.append("History has gaps - cooldown and trend are less certain")

    if last_recommendation_outcome is not None:
        was_rec = last_recommendation_outcome.get("was_recommended", False)
        pos = last_recommendation_outcome.get("position")
        lane = last_recommendation_outcome.get("lane")
        skipped = last_recommendation_outcome.get("higher_scored_lanes", [])

        if was_rec and pos is not None and lane:
            notes.append(f"Picked #{pos} from {lane.upper()} lane")
        elif was_rec:
            notes.append("Was a recommended option")
        else:
            notes.append("Was not in recommendations")

        if skipped:
            lane_names = ", ".join(lane.upper() for lane in skipped)
            notes.append(f"You skipped higher-scored {lane_names} options")

    return notes


def generate_tempo_key_summary(
    current_bpm: float,
    candidate_bpm: float,
    current_key: str | None,
    candidate_key: str | None,
    current_key_confidence: float | None = None,
    candidate_key_confidence: float | None = None,
) -> dict[str, str]:
    """Return compact tempo and key relationship summary.

    Output: {"tempo_text": str, "key_text": str, "key_state": "normal|uncertain|unknown"}
    """
    bpm_diff = candidate_bpm - current_bpm
    _bpm_dist, _matched, relationship, _raw = effective_bpm_distance(current_bpm, candidate_bpm)

    if relationship == "direct":
        if abs(bpm_diff) < 0.5:
            tempo_text = "0 BPM (match)"
        else:
            sign = "+" if bpm_diff > 0 else ""
            if abs(bpm_diff) <= 2:
                tempo_text = f"{sign}{bpm_diff:.0f} BPM (easy)"
            elif abs(bpm_diff) <= 4:
                tempo_text = f"{sign}{bpm_diff:.0f} BPM (push)"
            else:
                tempo_text = f"{sign}{bpm_diff:.0f} BPM (shift)"
    elif relationship in ("double", "half"):
        tempo_text = f"{relationship} time"
    else:
        tempo_text = f"{relationship} pivot"

    key_state = "normal"
    if not current_key or not candidate_key:
        key_text = "key unknown"
        key_state = "unknown"
    else:
        min_conf = min(
            current_key_confidence if current_key_confidence is not None else 1.0,
            candidate_key_confidence if candidate_key_confidence is not None else 1.0,
        )
        c_num, c_mode = parse_camelot(current_key)
        d_num, d_mode = parse_camelot(candidate_key)
        _dist, label = camelot_compatibility(c_num, c_mode, d_num, d_mode)
        KEY_LABEL: dict[str, str] = {
            "perfect": "same",
            "relative_key": "relative",
            "adjacent": "adjacent",
            "cross_adjacent": "cross-adjacent",
            "energy_boost": "energy boost",
            "energy_key_change": "dominant",
            "mismatch": "tension",
        }
        key_label = KEY_LABEL.get(label, "unknown")
        key_text = f"{current_key}→{candidate_key} ({key_label})"
        if min_conf < 0.7:
            key_state = "uncertain"
        if min_conf < 0.5:
            key_text = f"{current_key}→{candidate_key} ({key_label}, uncertain)"

    return {
        "tempo_text": tempo_text,
        "key_text": key_text,
        "key_state": key_state,
    }


def track_recommendation_outcome(
    lanes: dict[str, list[dict[str, Any]]],
    chosen_track_id: str,
) -> dict[str, Any]:
    """Compute telemetry outcome for what the DJ actually played.

    Input:
        lanes: dict of lane_name → list of recommendation dicts,
               each with "track_id" (or "candidate" having .track_id) and "score".
        chosen_track_id: what the DJ selected.
    Output: {"chosen_track_id": str, "was_recommended": bool, "position": int|None,
             "lane": str|None, "higher_scored_lanes": [str]}
    """
    def _track_id(rec: dict[str, Any]) -> str:
        # Support both raw track_id string and nested candidate object
        if "track_id" in rec:
            return str(rec["track_id"])
        cand = rec.get("candidate")
        if cand is not None:
            return str(getattr(cand, "track_id", "") or rec.get("candidate", ""))
        return ""

    for lane_name, lane_tracks in lanes.items():
        for i, rec in enumerate(lane_tracks):
            if _track_id(rec) == chosen_track_id:
                higher_scored_lanes = []
                for other_lane, other_tracks in lanes.items():
                    if other_lane == lane_name:
                        continue
                    if other_tracks and other_tracks[0].get("score", 0.0) > rec.get("score", 0.0):
                        higher_scored_lanes.append(other_lane)
                return {
                    "chosen_track_id": chosen_track_id,
                    "was_recommended": True,
                    "position": i + 1,
                    "lane": lane_name,
                    "higher_scored_lanes": higher_scored_lanes,
                }

    return {
        "chosen_track_id": chosen_track_id,
        "was_recommended": False,
        "position": None,
        "lane": None,
        "higher_scored_lanes": [],
    }


# ---------------------------------------------------------------------------
# 20.7  Composition layer
# ---------------------------------------------------------------------------

def build_live_candidate_explanation(
    current_track: dict[str, Any],
    candidate_track: dict[str, Any],
    transition_features: dict[str, Any],
    scores: dict[str, Any],
    current_outro_window: dict[str, Any] | None,
    candidate_intro_window: dict[str, Any] | None,
    history_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose the full live explanation block for one candidate.

    Budget: summary ≤ 3 reasons; why ≤ 5; watch ≤ 3; at most 1 handoff block.

    Input:
        current_track / candidate_track: dicts with at minimum bpm, key, role_hints,
            vocals_rel, bass_rel, energy_rel, and a "move" key on candidate_track.
        transition_features: output of compute_transition_features()
        scores: score_candidate() result payload or component-score-like dict.
            Risk notes are read from top-level `risk_factors` when present.
        current_outro_window / candidate_intro_window: windowed feature dicts or None
        history_context: unused in v1 (reserved for session-level notes)
    Output:
        {"summary": [str], "why": [str], "watch": [str],
         "handoff": dict|None, "tempo_key": dict, "character_shift": [str]}
    """
    move = scores.get("move") or candidate_track.get("move", "")
    component_scores = scores.get("component_scores") if isinstance(scores.get("component_scores"), dict) else {}
    normalized_scores = {**component_scores, **scores}
    reasons = generate_reasons(transition_features, normalized_scores, move)

    handoff = generate_handoff_narrative(current_outro_window, candidate_intro_window)

    tempo_key = generate_tempo_key_summary(
        current_bpm=current_track.get("bpm", 0.0),
        candidate_bpm=candidate_track.get("bpm", 0.0),
        current_key=current_track.get("key"),
        candidate_key=candidate_track.get("key"),
        current_key_confidence=(
            current_track.get("key_confidence")
            if current_track.get("key_confidence") is not None
            else transition_features.get("key_confidence_current")
        ),
        candidate_key_confidence=(
            candidate_track.get("key_confidence")
            if candidate_track.get("key_confidence") is not None
            else transition_features.get("key_confidence_candidate")
        ),
    )

    character_shift = describe_character_shift(
        current_roles=current_track.get("role_hints") or [],
        candidate_roles=candidate_track.get("role_hints") or [],
        current_vocals_rel=current_track.get("vocals_rel"),
        candidate_vocals_rel=candidate_track.get("vocals_rel"),
        current_bass_rel=current_track.get("bass_rel"),
        candidate_bass_rel=candidate_track.get("bass_rel"),
        current_energy_rel=current_track.get("energy_rel"),
        candidate_energy_rel=candidate_track.get("energy_rel"),
    )

    # Watchouts: handoff warnings first, then scorer risk factors
    watch: list[str] = []
    if handoff and handoff["level"] in ("yellow", "orange"):
        watch.extend(handoff["notes"][:COMPACT_MAX_WATCHOUTS])
    risk_factors = scores.get("risk_factors") or scores.get("risk_flags") or []
    for flag in risk_factors:
        if len(watch) >= EXPANDED_MAX_WATCH:
            break
        watch.append(flag)

    # Apply budget caps
    why = reasons[:EXPANDED_MAX_WHY]
    watch = watch[:EXPANDED_MAX_WATCH]

    return {
        "summary": reasons[:COMPACT_MAX_REASONS],
        "why": why,
        "watch": watch,
        "handoff": handoff,
        "tempo_key": tempo_key,
        "character_shift": character_shift,
    }


def build_track_detail_explanation(
    track_abs: dict[str, Any],
    track_rel: dict[str, Any],
    windows: dict[str, Any],
) -> dict[str, Any]:
    """Compose the full-analysis explanation block for a single track.

    Input:
        track_abs: absolute feature dict (unused directly - reserved for future fields)
        track_rel: relative feature dict with energy_rel, bass_rel, etc.
        windows: dict of window_name → feature dict (e.g. "intro_32", "outro_32")
    Output:
        {"relative_context": [str], "intro_advisory": dict|None,
         "outro_advisory": dict|None, "outro_summary": dict|None}
    """
    intro_32 = windows.get("intro_32")
    outro_32 = windows.get("outro_32")

    return {
        "relative_context": generate_relative_context_notes(track_rel),
        "intro_advisory": generate_window_advisory(intro_32, "intro_32"),
        "outro_advisory": generate_window_advisory(outro_32, "outro_32"),
        "outro_summary": generate_outro_summary(outro_32),
    }
