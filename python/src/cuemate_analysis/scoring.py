"""
Milestone 3 — Phase 1: Scoring Primitives (Narrow v1)

Pure functions only. No DB, no CLI. All functions are importable and testable in
isolation. DB integration, config loading, and CLI surface come in later phases.

Deferred (add after v1 ranking is validated):
  - vocal_transition_score
  - groove_score
  - transition_support_score (returns neutral 0.5 always in v1)
  - Any window-shaped interfaces
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HARMONIC_SCORE_MAP: dict[int, float] = {0: 1.00, 1: 0.85, 2: 0.50, 3: 0.15}

RATIO_DISTANCE_FLOOR: dict[str, float] = {
    "direct": 0.0,
    "double": 0.5,
    "half": 0.5,
    "three_two": 1.5,
    "two_three": 1.5,
    "four_three": 2.0,
    "three_four": 2.0,
}

# Loaded from config/default.json in Phase 2; kept here as module-level fallback
# so scoring functions work without any config wiring in tests.
STATIC_WEIGHTS: dict[str, float] = {
    "target_energy": 0.22,
    "transition_support": 0.18,
    "bass_transition": 0.15,
    "vocal_transition": 0.13,
    "harmonic": 0.12,
    "tempo": 0.10,
    "history_fit": 0.06,
    "rhythmic_continuity": 0.04,
}

WEIGHT_FLOORS: dict[str, float] = {
    "target_energy": 0.08,
    "transition_support": 0.05,
    "bass_transition": 0.04,
    "vocal_transition": 0.03,
    "harmonic": 0.04,
    "tempo": 0.03,
    "history_fit": 0.03,
    "rhythmic_continuity": 0.02,
}

# Explicit floor for harmonic contribution when a key is weak and unverified.
# Tunable here rather than buried in scoring logic. Also stored in config/default.json.
HARMONIC_CONFIDENCE_FLOOR: float = 0.15

# Threshold below which a standalone key estimate is not considered corroborated.
KEY_CONFIDENCE_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ScoringTrackContext:
    """Flat feature bundle consumed by all scoring functions.

    Populated from DB rows (Phase 2) or constructed directly in tests.
    The `id` attribute is also stored as `track_id` for compatibility with
    history dicts that reference `candidate.id` from the plan.
    """

    track_id: str
    bpm: float
    key: str | None
    key_confidence: float | None
    key_source: str | None
    key_agreement: int | None
    energy_rel: float | None
    bass_rel: float | None
    drums_rel: float | None
    vocals_rel: float | None
    groove_rel: float | None
    intensity_band: str | None
    role_hints: list[str] = field(default_factory=list)

    # Convenience alias so code copied from the plan spec (`candidate.id`) works
    # without patching every call site.
    @property
    def id(self) -> str:
        return self.track_id


# ---------------------------------------------------------------------------
# Key compatibility
# ---------------------------------------------------------------------------


def parse_camelot(key: str | None) -> tuple[int | None, str | None]:
    """Parse a Camelot notation key string into (number, mode).

    Returns (None, None) for any unparseable input.
    """
    if not key or len(key) < 2:
        return None, None
    mode = key[-1]
    number_part = key[:-1]
    if mode not in ("A", "B") or not number_part.isdigit():
        return None, None
    num = int(number_part)
    if num < 1 or num > 12:
        return None, None
    return num, mode


def camelot_compatibility(
    a_num: int | None,
    a_mode: str | None,
    b_num: int | None,
    b_mode: str | None,
) -> tuple[int, str]:
    """Return (distance, label) for two Camelot positions.

    Distance maps to HARMONIC_SCORE_MAP. Falls back to (3, 'mismatch') when
    either position is None.
    """
    if a_num is None or b_num is None or a_mode is None or b_mode is None:
        return 3, "mismatch"
    wheel_dist = min(abs(a_num - b_num), 12 - abs(a_num - b_num))
    same_mode = a_mode == b_mode
    if wheel_dist == 0 and same_mode:
        return 0, "perfect"
    if wheel_dist == 0 and not same_mode:
        return 1, "relative_key"
    if wheel_dist == 1 and same_mode:
        return 1, "adjacent"
    if wheel_dist == 1 and not same_mode:
        return 2, "cross_adjacent"
    if wheel_dist == 2 and same_mode:
        return 2, "energy_boost"
    if wheel_dist == 7 and same_mode:
        return 2, "energy_key_change"
    return 3, "mismatch"


# ---------------------------------------------------------------------------
# BPM distance
# ---------------------------------------------------------------------------


def effective_bpm_distance(
    a_bpm: float, b_bpm: float
) -> tuple[float, float, str, float]:
    """Return (effective_distance, matched_bpm, relationship, raw_distance).

    Considers direct, double, half, and creative ratio relationships and applies
    per-relationship distance floors so creative matches do not score as well as
    direct matches even when the raw BPM delta happens to be tiny.
    """
    candidates = [
        (b_bpm, "direct"),
        (b_bpm * 2, "double"),
        (b_bpm / 2, "half"),
        (b_bpm * 3 / 2, "three_two"),
        (b_bpm * 2 / 3, "two_three"),
        (b_bpm * 4 / 3, "four_three"),
        (b_bpm * 3 / 4, "three_four"),
    ]
    distances: list[tuple[float, float, str, float]] = []
    for candidate_bpm, label in candidates:
        raw_distance = abs(a_bpm - candidate_bpm)
        effective_distance = max(raw_distance, RATIO_DISTANCE_FLOOR.get(label, 0.0))
        distances.append((effective_distance, candidate_bpm, label, raw_distance))
    return min(distances, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Sigmoid helper
# ---------------------------------------------------------------------------


def sigmoid_normalize(x: float, center: float = 0.0, spread: float = 0.1) -> float:
    z = (x - center) / max(spread, 1e-6)
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# Score components — v1 set
# ---------------------------------------------------------------------------


def harmonic_score(current_key: str | None, candidate_key: str | None) -> float:
    """Score harmonic compatibility in [0, 1]. Returns 0.5 when either key is None."""
    if not current_key or not candidate_key:
        return 0.5
    c_num, c_mode = parse_camelot(current_key)
    d_num, d_mode = parse_camelot(candidate_key)
    dist, _ = camelot_compatibility(c_num, c_mode, d_num, d_mode)
    return HARMONIC_SCORE_MAP[dist]


def tempo_score(current_bpm: float, candidate_bpm: float, config: dict[str, Any]) -> float:
    """Advisory tempo compatibility score in [0.15, 1].

    Uses a gentle curve so exact matches score highest but soft-threshold
    matches do not collapse to zero.
    """
    dist, _, _, _ = effective_bpm_distance(current_bpm, candidate_bpm)
    soft = max(config.get("thresholds", {}).get("bpm_soft", 3.0), 0.5)
    ratio = min(1.5, dist / soft)
    return max(0.15, 1.0 - 0.85 * (ratio**1.6))


def target_energy_score(delta_energy_rel: float, target: str) -> float:
    """Score how well the energy delta matches the DJ's intent."""
    if target == "build":
        return sigmoid_normalize(delta_energy_rel, 0.08, 0.10)
    if target == "reset":
        return sigmoid_normalize(-delta_energy_rel, 0.08, 0.10)
    if target == "maintain":
        return 1.0 - min(abs(delta_energy_rel) * 5, 1.0)
    if target == "jump":
        return sigmoid_normalize(abs(delta_energy_rel), 0.15, 0.10)
    return 0.5


def bass_transition_score(
    current_bass_rel: float | None,
    candidate_bass_rel: float | None,
    target: str,
) -> float:
    """Score bass transition smoothness relative to target intent."""
    current_bass_rel = current_bass_rel if current_bass_rel is not None else 0.5
    candidate_bass_rel = candidate_bass_rel if candidate_bass_rel is not None else 0.5
    delta = candidate_bass_rel - current_bass_rel
    if target == "build":
        return sigmoid_normalize(delta, 0.1, 0.15)
    if target == "reset":
        return sigmoid_normalize(-delta, 0.1, 0.15)
    return 1.0 - min(abs(delta) * 3, 1.0)


def history_fit_score(
    candidate: ScoringTrackContext,
    history: list[dict[str, Any]],
    window: int = 5,
) -> float:
    """Baseline history signal: short-term key repetition + energy stagnation.

    This is a conservative placeholder. Expansion (artist repetition, BPM
    staleness, intensity-band reuse) is deferred to Milestone 5.
    """
    recent = history[-window:]
    if not recent:
        return 1.0
    key_penalty = sum(
        0.15 * (0.8**i)
        for i, h in enumerate(reversed(recent))
        if h.get("key") == candidate.key
    )
    energies = [h.get("energy_rel", 0.5) for h in recent]
    stagnation = (
        0.2
        if len(set(round(e, 1) for e in energies)) <= 2 and len(energies) >= 3
        else 0.0
    )
    return max(0.0, 1.0 - key_penalty - stagnation)


def contrast_score(
    current_rel: dict[str, Any],
    candidate_rel: dict[str, Any],
) -> float:
    """Measure how much of a deliberate character shift the candidate represents.

    NOT included in the weighted score. Used exclusively for:
    - contrast lane population
    - secondary_lane dual-membership
    - risk advisory notes
    - wildcard eligibility

    Will be added to weighted scoring only after v1 ranking is validated.
    """
    dims: list[tuple[str, float, float]] = []
    energy_delta = abs(
        (current_rel.get("energy_rel") or 0.5) - (candidate_rel.get("energy_rel") or 0.5)
    )
    dims.append(("energy", energy_delta, 0.30))
    vocal_delta = abs(
        (current_rel.get("vocals_rel") or 0.0) - (candidate_rel.get("vocals_rel") or 0.0)
    )
    dims.append(("vocal", vocal_delta, 0.20))
    bass_delta = abs(
        (current_rel.get("bass_rel") or 0.5) - (candidate_rel.get("bass_rel") or 0.5)
    )
    dims.append(("bass", bass_delta, 0.20))
    groove_delta = abs(
        (current_rel.get("groove_rel") or 0.5) - (candidate_rel.get("groove_rel") or 0.5)
    )
    dims.append(("groove", groove_delta, 0.10))
    cur_roles = set(current_rel.get("role_hints") or [])
    cand_roles = set(candidate_rel.get("role_hints") or [])
    if cur_roles or cand_roles:
        union = cur_roles | cand_roles
        intersection = cur_roles & cand_roles
        role_dissimilarity = 1.0 - (len(intersection) / max(len(union), 1))
    else:
        role_dissimilarity = 0.0
    dims.append(("role", role_dissimilarity, 0.20))
    raw = sum(delta * weight for _, delta, weight in dims)
    breadth_bonus = min(0.15, sum(1 for _, delta, _ in dims if delta > 0.25) * 0.05)
    return round(min(1.0, raw + breadth_bonus), 4)


# ---------------------------------------------------------------------------
# Weight resolution
# ---------------------------------------------------------------------------


def resolve_effective_weights(
    playlist_stats: dict[str, Any] | None,
) -> dict[str, float]:
    """Return effective per-component weights for live scoring.

    Reads precomputed `adapted_weights` from playlist_stats when available.
    Falls back to STATIC_WEIGHTS so scoring works without a DB-populated crate.
    """
    if playlist_stats and playlist_stats.get("adapted_weights"):
        return dict(playlist_stats["adapted_weights"])
    return dict(STATIC_WEIGHTS)


# ---------------------------------------------------------------------------
# Key trust / confidence modulation
# ---------------------------------------------------------------------------


def _harmonic_confidence(track: ScoringTrackContext) -> float:
    """Derive harmonic confidence from key trust policy.

    - Corroborated (key_agreement >= 1): full weight (1.0)
    - High-confidence standalone (key_confidence >= 0.5): medium (key_confidence)
    - Weak standalone: floor (max(HARMONIC_CONFIDENCE_FLOOR, key_confidence * 0.5))
    """
    # Corroborated: multiple independent sources agree
    if track.key_agreement is not None and track.key_agreement >= 1:
        return 1.0
    # High-confidence standalone
    kc = track.key_confidence or 0.0
    if kc >= KEY_CONFIDENCE_THRESHOLD:
        return kc
    # Weak standalone — clamped to floor
    return max(HARMONIC_CONFIDENCE_FLOOR, kc * 0.5)


def build_confidence_map(
    current: ScoringTrackContext,
    candidate: ScoringTrackContext,
) -> dict[str, float]:
    """Assemble per-component confidence values for the current→candidate pair.

    Harmonic confidence is the *minimum* of both track key trusts — a weak key
    on either side reduces confidence for the pair.

    All other components default to 1.0 (fully trusted) in v1. Groove and vocal
    confidence modulation will be added when those score components are wired in.
    """
    harmonic_conf = min(
        _harmonic_confidence(current),
        _harmonic_confidence(candidate),
    )
    return {
        "target_energy": 1.0,
        "transition_support": 1.0,
        "bass_transition": 1.0,
        "vocal_transition": 1.0,
        "harmonic": harmonic_conf,
        "tempo": 1.0,
        "history_fit": 1.0,
        "rhythmic_continuity": 1.0,
    }


def compute_weighted_score(
    feature_scores: dict[str, float],
    weights: dict[str, float],
    confidences: dict[str, float],
) -> float:
    """Weight-modulated score in [0, 1].

    Confidence values reduce the effective weight of uncertain signals.
    Weight floors prevent any component from vanishing entirely.
    This is weight modulation, NOT calibrated statistical probability.
    """
    adjusted: dict[str, float] = {}
    for feature, weight in weights.items():
        conf = confidences.get(feature, 1.0)
        effective_conf = max(conf, HARMONIC_CONFIDENCE_FLOOR)
        adjusted[feature] = weight * effective_conf
    for feature in list(adjusted):
        if feature in WEIGHT_FLOORS:
            adjusted[feature] = max(adjusted[feature], WEIGHT_FLOORS[feature])
    total = sum(adjusted.values())
    if total < 1e-6:
        return 0.5
    normalized = {k: v / total for k, v in adjusted.items()}
    return sum(
        normalized[f] * feature_scores[f]
        for f in feature_scores
        if f in normalized
    )


# ---------------------------------------------------------------------------
# Transition features
# ---------------------------------------------------------------------------


def compute_transition_features(
    current: ScoringTrackContext,
    candidate: ScoringTrackContext,
) -> dict[str, Any]:
    """Canonical per-pair feature bundle used by penalties, risk, and explanations.

    Window parameters (current_outro, candidate_intro) are deferred — transition_support
    scores a neutral 0.5 when windows are absent.
    """
    bpm_dist, _, bpm_relationship, raw_bpm_dist = effective_bpm_distance(
        current.bpm, candidate.bpm
    )
    key_distance = 0
    key_compat_label = "unknown"
    if current.key and candidate.key:
        c_num, c_mode = parse_camelot(current.key)
        d_num, d_mode = parse_camelot(candidate.key)
        key_distance, key_compat_label = camelot_compatibility(c_num, c_mode, d_num, d_mode)
    return {
        "effective_bpm_distance": bpm_dist,
        "raw_bpm_distance": raw_bpm_dist,
        "bpm_relationship": bpm_relationship,
        "key_distance": key_distance,
        "key_compat_label": key_compat_label,
        "key_confidence_current": current.key_confidence or 0.0,
        "key_confidence_candidate": candidate.key_confidence or 0.0,
        "delta_energy_rel": (
            (candidate.energy_rel if candidate.energy_rel is not None else 0.5)
            - (current.energy_rel if current.energy_rel is not None else 0.5)
        ),
        "delta_bass_rel": (
            (candidate.bass_rel if candidate.bass_rel is not None else 0.5)
            - (current.bass_rel if current.bass_rel is not None else 0.5)
        ),
        "current_vocals_rel": current.vocals_rel if current.vocals_rel is not None else 0.0,
        "candidate_vocals_rel": candidate.vocals_rel if candidate.vocals_rel is not None else 0.0,
        # Window-based low-end fields — always 0.0 in v1 (no window data)
        "current_outro_low_end": 0.0,
        "candidate_intro_low_end": 0.0,
    }


# ---------------------------------------------------------------------------
# Penalty system
# ---------------------------------------------------------------------------


def compute_penalties(
    transition_features: dict[str, Any],
    feature_scores: dict[str, float],
    confidences: dict[str, float],
    config: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    """Return (penalty_multiplier, penalty_factors) where multiplier ∈ [1-max_total, 1.0].

    Penalties model compounding difficulty not captured by individual score components.
    They reduce ranking position but never eliminate candidates.
    """
    penalty_config = config.get("penalties", {})
    max_total = penalty_config.get("max_total_penalty", 0.80)
    raw_penalties: list[float] = []
    penalty_factors: list[dict[str, Any]] = []

    # BPM soft-threshold penalty
    bpm_dist = transition_features.get("effective_bpm_distance", 0.0)
    bpm_soft = config.get("thresholds", {}).get("bpm_soft", 3.0)
    if bpm_dist > bpm_soft:
        severity = min(1.0, (bpm_dist - bpm_soft) / max(bpm_soft, 1e-6))
        base_penalty = penalty_config.get("bpm_over_soft", 0.30) * severity
        raw_penalties.append(base_penalty)
        penalty_factors.append(
            {
                "factor": "bpm_over_soft",
                "severity": round(severity, 3),
                "raw_penalty": round(base_penalty, 3),
            }
        )

    # Key mismatch penalty — only when confidently poor AND harmonic score is already weak
    key_distance = transition_features.get("key_distance", 0)
    key_conf = min(
        transition_features.get("key_confidence_current", 1.0),
        transition_features.get("key_confidence_candidate", 1.0),
    )
    harmonic_component = feature_scores.get("harmonic", 0.5)
    if key_distance >= 3 and key_conf >= 0.60 and harmonic_component < 0.35:
        severity = min(1.0, (key_distance - 2) / 2.0)
        conf_factor = max(0.25, key_conf)
        base_penalty = penalty_config.get("key_mismatch", 0.45) * severity * conf_factor
        raw_penalties.append(base_penalty)
        penalty_factors.append(
            {
                "factor": "key_mismatch",
                "severity": round(severity, 3),
                "raw_penalty": round(base_penalty, 3),
                "gate": "Only applied when harmonic score is already weak and key confidence is high",
            }
        )

    # Vocal clash penalty
    vocal_product = (transition_features.get("current_vocals_rel") or 0.0) * (
        transition_features.get("candidate_vocals_rel") or 0.0
    )
    if vocal_product > 0.25:
        severity = min(1.0, (vocal_product - 0.25) / 0.5)
        vocal_conf = max(0.25, confidences.get("vocal_transition", 1.0))
        base_penalty = penalty_config.get("vocal_clash", 0.35) * severity * vocal_conf
        raw_penalties.append(base_penalty)
        penalty_factors.append(
            {
                "factor": "vocal_clash",
                "severity": round(severity, 3),
                "raw_penalty": round(base_penalty, 3),
            }
        )

    # Low-end stacking penalty (window data only — always 0 in v1, gate will never fire)
    low_end_product = (transition_features.get("current_outro_low_end") or 0.0) * (
        transition_features.get("candidate_intro_low_end") or 0.0
    )
    if low_end_product > 0.4:
        severity = min(1.0, (low_end_product - 0.4) / 0.4)
        base_penalty = 0.20 * severity
        raw_penalties.append(base_penalty)
        penalty_factors.append(
            {
                "factor": "low_end_stacking",
                "severity": round(severity, 3),
                "raw_penalty": round(base_penalty, 3),
            }
        )

    if not raw_penalties:
        return 1.0, []

    if len(raw_penalties) == 1:
        single_effective = raw_penalties[0] * 0.6
        penalty_factors[0]["effective_penalty"] = round(single_effective, 3)
        penalty_factors[0]["compound_note"] = (
            "Single factor — reduced penalty because the component score already captures part of this signal"
        )
        return round(max(1.0 - max_total, 1.0 - single_effective), 4), penalty_factors

    sorted_pairs = sorted(
        zip(raw_penalties, penalty_factors), key=lambda x: x[0], reverse=True
    )
    compound = 0.0
    for i, (penalty_value, factor) in enumerate(sorted_pairs):
        diminishing_factor = 0.7**i
        effective = penalty_value * diminishing_factor
        compound += effective
        factor["effective_penalty"] = round(effective, 3)
        factor["diminishing_factor"] = round(diminishing_factor, 3)
    compound = min(compound, max_total)
    return round(max(1.0 - max_total, 1.0 - compound), 4), [
        factor for _, factor in sorted_pairs
    ]


# ---------------------------------------------------------------------------
# Candidate filtering
# ---------------------------------------------------------------------------


def filter_candidates(
    current: ScoringTrackContext,
    candidates: list[ScoringTrackContext],
    history: list[dict[str, Any]],
    config: dict[str, Any],
    bypass_filters: set[str] | None = None,
) -> list[ScoringTrackContext]:
    """Remove candidates that fail hard filters.

    Hard filters (permissive assistant-mode defaults):
    - same track as current
    - recently played cooldown
    - BPM distance beyond hard limit

    Key is NOT a hard filter. It contributes to score and risk only.
    """
    if bypass_filters is None:
        bypass_filters = set()
    thresholds = config.get("thresholds", {})
    bpm_hard = thresholds.get("bpm_hard", 8.0)
    cooldown_window = thresholds.get("cooldown_window", 5)
    recent_ids = {h.get("id") or h.get("track_id") for h in history[-cooldown_window:]}

    filtered: list[ScoringTrackContext] = []
    for candidate in candidates:
        # Same-track exclusion (always applied)
        if candidate.track_id == current.track_id:
            continue
        # BPM hard limit
        if "bpm" not in bypass_filters:
            bpm_dist, _, _, _ = effective_bpm_distance(current.bpm, candidate.bpm)
            if bpm_dist > bpm_hard:
                continue
        # Cooldown
        if "cooldown" not in bypass_filters and candidate.track_id in recent_ids:
            continue
        filtered.append(candidate)
    return filtered


# ---------------------------------------------------------------------------
# Move classification
# ---------------------------------------------------------------------------


def classify_move(
    delta_energy_rel: float,
    delta_bass_rel: float,
    vocal_current_rel: float,
    vocal_candidate_rel: float,
    energy_spread: float | None,
    config: dict[str, Any] | None,
) -> tuple[str, float, str | None]:
    """Classify the primary move type → (name, confidence, note).

    Thresholds scale with playlist energy spread so classifications feel
    proportional across narrow-BPM crates and wide-energy crates.
    """
    thresholds = config.get("move_types", {}) if config else {}
    jump_t = thresholds.get("jump_threshold", 0.12)
    build_t = thresholds.get("build_threshold", 0.05)
    maintain_t = thresholds.get("maintain_range", 0.05)
    reset_e_t = thresholds.get("reset_energy_threshold", -0.08)
    reset_v_t = thresholds.get("reset_vocal_threshold", 0.50)
    drop_t = thresholds.get("drop_threshold", -0.05)

    if energy_spread and energy_spread > 0.05:
        scale = max(0.5, min(1.5, energy_spread / 0.3))
        jump_t *= scale
        build_t *= scale
        maintain_t *= scale
        reset_e_t /= scale
        drop_t /= scale

    if delta_energy_rel > jump_t:
        return "jump", 0.95, None
    if delta_energy_rel > build_t:
        return "build", 0.85, None
    if delta_energy_rel < reset_e_t and (
        vocal_candidate_rel > reset_v_t or delta_energy_rel < -0.15
    ):
        return "reset", 0.85, None
    if delta_energy_rel < drop_t:
        return "drop", 0.80, None
    if abs(delta_energy_rel) <= maintain_t:
        return "maintain", 0.90, None
    # Slight negative dip that doesn't qualify as reset/drop — reduced confidence
    return "maintain", 0.55, "slight dip"


# ---------------------------------------------------------------------------
# Risk computation
# ---------------------------------------------------------------------------


def compute_risk(
    transition_features: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, float, list[str]]:
    """Return (level, risk_score, factors) where level ∈ {'low', 'medium', 'high'}.

    Risk is informational only — it does not gate candidates.
    """
    factors: list[str] = []
    factor_weights: list[float] = []
    thresholds = config.get("thresholds", {})

    bpm_dist = transition_features["effective_bpm_distance"]
    bpm_soft = thresholds.get("bpm_soft", 3.0)
    if bpm_dist > bpm_soft:
        factors.append(f"BPM difference: {bpm_dist:.1f}")
        factor_weights.append(0.3)

    key_dist = transition_features["key_distance"]
    if key_dist >= 3:
        factors.append(f"Higher harmonic tension (distance {key_dist})")
        factor_weights.append(0.25)

    cur_voc = transition_features.get("current_vocals_rel", 0.0)
    cand_voc = transition_features.get("candidate_vocals_rel", 0.0)
    if cur_voc > 0.6 and cand_voc > 0.6:
        factors.append("Both tracks carry strong vocal content")
        factor_weights.append(0.2)

    delta_e = abs(transition_features.get("delta_energy_rel", 0.0))
    if delta_e > 0.15:
        factors.append(f"Big energy shift ({delta_e:.0%}) — bold move")
        factor_weights.append(0.15)

    bpm_rel = transition_features.get("bpm_relationship", "direct")
    if bpm_rel not in ("direct", "double", "half"):
        factors.append(f"Creative tempo pivot ({bpm_rel})")
        factor_weights.append(0.2)

    key_conf = min(
        transition_features.get("key_confidence_current", 1.0),
        transition_features.get("key_confidence_candidate", 1.0),
    )
    if key_conf < 0.6:
        factors.append(f"Key detection uncertain ({key_conf:.0%}) — trust your ears")
        factor_weights.append(0.1)

    risk_score = min(1.0, sum(factor_weights))
    if len(factors) > 1:
        risk_score = min(1.0, risk_score * (1.0 + (len(factors) - 1) * 0.1))
    level = "high" if risk_score > 0.5 else "medium" if risk_score > 0.2 else "low"
    return level, round(risk_score, 3), factors


# ---------------------------------------------------------------------------
# Full candidate scoring
# ---------------------------------------------------------------------------


def score_candidate(
    current: ScoringTrackContext,
    candidate: ScoringTrackContext,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    playlist_stats: dict[str, Any] | None,
    confidences: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Orchestrate all scoring components for one current→candidate pair.

    Returns a result dict suitable for ranking and lane organization.
    `confidences` is computed from key trust when not supplied externally.
    """
    if confidences is None:
        confidences = build_confidence_map(current, candidate)

    transition_features = compute_transition_features(current, candidate)

    target = config.get("target", "maintain")
    feature_scores: dict[str, float] = {
        "target_energy": target_energy_score(
            transition_features["delta_energy_rel"], target
        ),
        # transition_support returns neutral 0.5 in v1 — no window data yet
        "transition_support": 0.5,
        "bass_transition": bass_transition_score(
            current.bass_rel, candidate.bass_rel, target
        ),
        # vocal_transition deferred — returns neutral 0.5 in v1
        "vocal_transition": 0.5,
        "harmonic": harmonic_score(current.key, candidate.key),
        "tempo": tempo_score(current.bpm, candidate.bpm, config),
        "history_fit": history_fit_score(candidate, history),
        # rhythmic_continuity (groove_score) deferred — neutral 0.5 in v1
        "rhythmic_continuity": 0.5,
    }

    weights = resolve_effective_weights(playlist_stats)
    raw_score = compute_weighted_score(feature_scores, weights, confidences)

    penalty_multiplier, penalty_factors = compute_penalties(
        transition_features, feature_scores, confidences, config
    )
    final_score = round(raw_score * penalty_multiplier, 4)

    risk_level, risk_score, risk_factors = compute_risk(transition_features, config)

    move_name, move_confidence, move_note = classify_move(
        transition_features["delta_energy_rel"],
        transition_features.get("delta_bass_rel", 0.0),
        current.vocals_rel or 0.0,
        candidate.vocals_rel or 0.0,
        (playlist_stats or {}).get("energy_spread"),
        config,
    )

    current_rel = {
        "energy_rel": current.energy_rel,
        "bass_rel": current.bass_rel,
        "vocals_rel": current.vocals_rel,
        "groove_rel": current.groove_rel,
        "role_hints": current.role_hints,
    }
    candidate_rel = {
        "energy_rel": candidate.energy_rel,
        "bass_rel": candidate.bass_rel,
        "vocals_rel": candidate.vocals_rel,
        "groove_rel": candidate.groove_rel,
        "role_hints": candidate.role_hints,
    }

    return {
        "candidate": candidate,
        "raw_score": raw_score,
        "score": final_score,
        "penalty_multiplier": penalty_multiplier,
        "penalty_factors": penalty_factors,
        "risk": risk_level,
        "risk_score": risk_score,
        "risk_factors": risk_factors,
        "move": move_name,
        "move_confidence": move_confidence,
        "move_note": move_note,
        # contrast_score is computed here but NOT included in weighted ranking
        "contrast_score": contrast_score(current_rel, candidate_rel),
        "component_scores": feature_scores,
        "transition_features": transition_features,
        "confidences": confidences,
        "weights_used": weights,
    }
