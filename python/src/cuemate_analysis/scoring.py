"""
Milestone 3 — Phase 1: Scoring Primitives (Narrow v1)

Pure functions only. No DB, no CLI. All functions are importable and testable in
isolation. DB integration, config loading, and CLI surface come in later phases.

Deferred (add after v1 ranking is validated):
  - vocal_transition_score
  - groove_score / rhythmic_continuity
  - transition_support_score (requires window data)

Stub components use STUB_SCORE = None. compute_weighted_score skips None scores
and renormalizes over active components — weight table is unchanged for reporting.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from cuemate_analysis import __version__

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

# Canonical source is config.py; aliased here for backward compatibility with
# tests and code that import from scoring directly.
from cuemate_analysis.config import DEFAULT_STATIC_WEIGHTS as STATIC_WEIGHTS  # noqa: E402
from cuemate_analysis.config import DEFAULT_WEIGHT_FLOORS as WEIGHT_FLOORS  # noqa: E402

# Sentinel for scoring components that are not yet implemented.
# compute_weighted_score skips these entirely and renormalizes over active components.
# The weight table is left unchanged for reporting / contract stability.
STUB_SCORE: None = None

# Explicit floor for harmonic contribution when a key is weak and unverified.
# Tunable here rather than buried in scoring logic. Also stored in config/default.json.
HARMONIC_CONFIDENCE_FLOOR: float = 0.15

# Threshold below which a standalone key estimate is not considered corroborated.
KEY_CONFIDENCE_THRESHOLD: float = 0.5

# Phase 6 metadata/versioning contract for the Python scoring core.
SCORING_CONTRACT_ID: str = "m3-v1"

# Applied to absolute features only.
LABEL_CONFIG: dict[str, dict[str, list[float] | list[str]]] = {
    "energy": {
        "boundaries": [0.55, 0.68, 0.78, 0.88],
        "labels": ["Low", "Groove", "High", "Peak", "Max"],
    },
    "bass": {
        "boundaries": [0.55, 0.70, 0.82],
        "labels": ["Light", "Groove", "Punch", "Heavy"],
    },
    "drums": {
        "boundaries": [0.60, 0.75, 0.90],
        "labels": ["Soft", "Drive", "Strong", "Max"],
    },
    "harmonic": {
        "boundaries": [0.65, 0.75],
        "labels": ["Mid", "Full", "Rich"],
    },
    "vocals": {
        "boundaries": [0.35, 0.60],
        "labels": ["Instrumental", "Mixed", "Vocal"],
    },
    "groove": {
        "boundaries": [0.65, 0.75, 0.85],
        "labels": ["Flat", "Groove", "Drive", "Swing"],
    },
}

COMPONENT_DESCRIPTIONS: dict[str, str] = {
    "target_energy": "How well the candidate matches the requested pressure direction.",
    "transition_support": "Intro/outro handoff support from windowed analysis.",
    "bass_transition": "Low-end continuity or relief across the transition.",
    "vocal_transition": "Blend-space risk/opportunity from vocal overlap or contrast.",
    "harmonic": "Camelot/key compatibility, confidence-modulated by key trust.",
    "tempo": "BPM proximity after ratio-aware tempo matching.",
    "history_fit": "Penalty for repetition or stagnant recent sequencing.",
    "rhythmic_continuity": "Groove/rhythmic feel continuity across the handoff.",
}

STUBBED_COMPONENTS: set[str] = {
    "transition_support",
    "vocal_transition",
    "rhythmic_continuity",
}

SUPPORTED_LANE_GROUPS: list[dict[str, str]] = [
    {
        "lane_id": "maintain",
        "display_name": "Maintain",
        "summary": "Keep the room on its current frame.",
    },
    {
        "lane_id": "build",
        "display_name": "Build",
        "summary": "Raise momentum without making it a full jump.",
    },
    {
        "lane_id": "reset",
        "display_name": "Reset",
        "summary": "Create space or reframe the room before the next move.",
    },
    {
        "lane_id": "jump",
        "display_name": "Jump",
        "summary": "Make a bigger energy or character move.",
    },
    {
        "lane_id": "contrast",
        "display_name": "Contrast",
        "summary": "Exploratory, higher-contrast alternatives.",
    },
]


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
    title: str | None = None
    artist: str | None = None

    # Convenience alias so code copied from the plan spec (`candidate.id`) works
    # without patching every call site.
    @property
    def id(self) -> str:
        return self.track_id


# ---------------------------------------------------------------------------
# DB row → ScoringTrackContext
# ---------------------------------------------------------------------------


def row_to_scoring_track_context(row: Any) -> ScoringTrackContext:
    """Convert a DB row (from get_scoring_candidates / get_track_scoring_context) to a ScoringTrackContext.

    Accepts sqlite3.Row, dict, or any mapping-like object. JSON-encoded
    role_hints columns are decoded here so the rest of the scoring layer
    never has to deal with raw strings.
    """
    # Support both sqlite3.Row (key access) and plain dict
    def _get(key: str, default: Any = None) -> Any:
        try:
            return row[key]
        except (KeyError, IndexError):
            return default

    raw_role_hints = _get("role_hints")
    if isinstance(raw_role_hints, str):
        try:
            role_hints: list[str] = json.loads(raw_role_hints)
        except (json.JSONDecodeError, TypeError):
            role_hints = []
    elif isinstance(raw_role_hints, list):
        role_hints = raw_role_hints
    else:
        role_hints = []

    return ScoringTrackContext(
        track_id=str(_get("track_id")),
        bpm=float(_get("bpm") or 0.0),
        key=_get("key") or None,
        key_confidence=float(_get("key_confidence")) if _get("key_confidence") is not None else None,
        key_source=_get("key_source") or None,
        key_agreement=int(_get("key_agreement")) if _get("key_agreement") is not None else None,
        energy_rel=float(_get("energy_rel")) if _get("energy_rel") is not None else None,
        bass_rel=float(_get("bass_rel")) if _get("bass_rel") is not None else None,
        drums_rel=float(_get("drums_rel")) if _get("drums_rel") is not None else None,
        vocals_rel=float(_get("vocals_rel")) if _get("vocals_rel") is not None else None,
        groove_rel=float(_get("groove_rel")) if _get("groove_rel") is not None else None,
        intensity_band=_get("intensity_band") or None,
        role_hints=role_hints,
        title=_get("title") or None,
        artist=_get("artist") or None,
    )


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
    if wheel_dist == 5 and same_mode:
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
# Phase 6: Labels & scoring metadata
# ---------------------------------------------------------------------------


def classify_label(
    value: float | None,
    boundaries: list[float],
    labels: list[str],
) -> str | None:
    """Bucket an absolute feature value into a calibrated label."""
    if value is None:
        return None
    for i, threshold in enumerate(boundaries):
        if value < threshold:
            return labels[i]
    return labels[-1]


def classify_track_labels(track_abs: Any) -> dict[str, str | None]:
    """Return absolute feature labels for a dict/dataclass/object with *_abs fields."""

    def _get(name: str) -> float | None:
        if isinstance(track_abs, dict):
            value = track_abs.get(name)
        else:
            value = getattr(track_abs, name, None)
        return float(value) if value is not None else None

    return {
        feature_name: classify_label(
            _get(f"{feature_name}_abs"),
            spec["boundaries"],  # type: ignore[arg-type]
            spec["labels"],      # type: ignore[arg-type]
        )
        for feature_name, spec in LABEL_CONFIG.items()
    }


def get_scoring_metadata(
    settings: Any | None = None,
    *,
    compatible_analysis_signatures: list[str] | None = None,
    compatible_config_signatures: list[str] | None = None,
    healthy: bool = True,
    status_note: str | None = None,
) -> dict[str, Any]:
    """Return scoring metadata in a proto-aligned dict shape."""
    if settings is None:
        from cuemate_analysis.config import load_runtime_settings

        settings = load_runtime_settings()
    from cuemate_analysis.config import build_relative_experiment_signature

    active_analysis_signature = str(getattr(settings, "analysis_signature"))
    active_config_signature = str(getattr(settings, "config_signature"))
    expected_relative_signature = build_relative_experiment_signature(
        settings,
        energy_source="canonical",
    )
    active_weights = dict(getattr(settings, "scoring").static_weights)
    metadata_status_note = status_note or (
        "Python scoring core metadata. "
        "transition_support, vocal_transition, and rhythmic_continuity are currently stubbed and excluded from weighted scoring."
    )

    return {
        "active_signatures": {
            "analysis_signature": active_analysis_signature,
            "scoring_contract_id": SCORING_CONTRACT_ID,
            "config_signature": active_config_signature,
        },
        "compatible_analysis_signatures": compatible_analysis_signatures or [active_analysis_signature],
        "compatible_config_signatures": compatible_config_signatures or [active_config_signature],
        "components": [
            {
                "component_id": component_id,
                "description": COMPONENT_DESCRIPTIONS.get(component_id, ""),
                "weight": active_weights.get(component_id, STATIC_WEIGHTS.get(component_id, 0.0)),
                "available": component_id not in STUBBED_COMPONENTS,
                "active": component_id not in STUBBED_COMPONENTS,
                "status": "live" if component_id not in STUBBED_COMPONENTS else "stubbed",
            }
            for component_id in STATIC_WEIGHTS
        ],
        "supported_lane_groups": list(SUPPORTED_LANE_GROUPS),
        "capability_flags": {
            "vocals_available": False,
            "window_features_available": False,
            "transition_support_available": False,
            "vocal_transition_available": False,
            "rhythmic_continuity_available": False,
            "explanations_available": True,
            "label_classification_available": True,
        },
        "healthy": healthy,
        "engine_version": __version__,
        "status_note": metadata_status_note,
        "expected_relative_signature": expected_relative_signature,
    }


def check_analysis_compatibility(
    track_analysis_signature: str | None,
    track_config_signature: str | None,
    track_scoring_contract_id_at_analysis: str | None = None,
    *,
    scoring_metadata: dict[str, Any] | None = None,
    settings: Any | None = None,
) -> dict[str, Any]:
    """Compare artifact signatures against the active scoring metadata."""
    metadata = scoring_metadata or get_scoring_metadata(settings)
    active = metadata["active_signatures"]
    active_analysis_signature = active["analysis_signature"]
    active_config_signature = active["config_signature"]
    active_scoring_contract_id = active["scoring_contract_id"]
    compatible_analysis_signatures = set(metadata.get("compatible_analysis_signatures", []))
    compatible_config_signatures = set(metadata.get("compatible_config_signatures", []))

    if not track_analysis_signature or not track_config_signature:
        return {
            "exact_match": False,
            "compatible": False,
            "requires_reanalysis": True,
            "reason": "missing_signature_metadata",
            "notes": ["Artifact is missing analysis_signature or config_signature."],
        }

    if track_scoring_contract_id_at_analysis is None:
        return {
            "exact_match": False,
            "compatible": False,
            "requires_reanalysis": True,
            "reason": "missing_scoring_contract_id",
            "notes": ["Artifact predates scoring contract tagging; reanalysis is required."],
        }

    if track_scoring_contract_id_at_analysis != active_scoring_contract_id:
        return {
            "exact_match": False,
            "compatible": False,
            "requires_reanalysis": True,
            "reason": "scoring_contract_mismatch",
            "notes": [
                f"Artifact scoring contract {track_scoring_contract_id_at_analysis} does not match active {active_scoring_contract_id}."
            ],
        }

    def _analysis_family_match(track_sig: str, active_sig: str) -> bool:
        return track_sig == active_sig or track_sig.startswith(f"{active_sig}-")

    analysis_exact = track_analysis_signature == active_analysis_signature
    config_exact = track_config_signature == active_config_signature
    if analysis_exact and config_exact:
        return {
            "exact_match": True,
            "compatible": True,
            "requires_reanalysis": False,
            "reason": "exact_match",
            "notes": [],
        }

    notes: list[str] = []
    analysis_compatible = (
        track_analysis_signature in compatible_analysis_signatures
        or any(_analysis_family_match(track_analysis_signature, sig) for sig in compatible_analysis_signatures)
    )
    config_compatible = track_config_signature in compatible_config_signatures
    if not analysis_compatible:
        return {
            "exact_match": False,
            "compatible": False,
            "requires_reanalysis": True,
            "reason": "analysis_signature_incompatible",
            "notes": [
                f"Artifact analysis signature {track_analysis_signature} is not in the active compatibility set."
            ],
        }
    if not config_compatible:
        return {
            "exact_match": False,
            "compatible": False,
            "requires_reanalysis": True,
            "reason": "config_signature_incompatible",
            "notes": [
                f"Artifact config signature {track_config_signature} is not in the active compatibility set."
            ],
        }

    if not analysis_exact:
        notes.append(
            f"Artifact analysis signature {track_analysis_signature} is compatible with active {active_analysis_signature}."
        )
    if not config_exact:
        notes.append(
            f"Artifact config signature {track_config_signature} is compatible with active {active_config_signature}."
        )
    return {
        "exact_match": False,
        "compatible": True,
        "requires_reanalysis": False,
        "reason": "compatible_but_not_exact",
        "notes": notes,
    }


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
        # Reset prefers pressure drop but tolerates near-neutral energy if
        # the track reframes the room (captured by other components).
        # Strongly negative → great, slightly negative → good, neutral → okay,
        # positive → poor.
        if delta_energy_rel <= -0.03:
            return sigmoid_normalize(-delta_energy_rel, 0.06, 0.12)
        # Near-neutral: gentle falloff rather than cliff
        return max(0.25, 1.0 - delta_energy_rel * 4)
    if target == "maintain":
        return 1.0 - min(abs(delta_energy_rel) * 5, 1.0)
    if target == "jump":
        return sigmoid_normalize(abs(delta_energy_rel), 0.15, 0.10)
    if target == "contrast":
        return sigmoid_normalize(abs(delta_energy_rel), 0.10, 0.12)
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
        # Reset prefers lower/flatter bass (pressure release) but tolerates
        # neutral bass if energy or character is shifting.
        if delta <= 0:
            return sigmoid_normalize(-delta, 0.05, 0.15)
        # Rising bass during reset is mildly penalized, not banned
        return max(0.2, 1.0 - delta * 3)
    if target == "contrast":
        return 1.0 - min(abs(delta) * 2, 0.8)
    return 1.0 - min(abs(delta) * 3, 1.0)


def history_fit_score(
    candidate: ScoringTrackContext,
    history: list[dict[str, Any]],
    window: int = 5,
) -> float:
    """Baseline history signal: short-term key repetition + energy stagnation.

    This is a conservative placeholder. Expansion can still add artist
    repetition, BPM staleness, and intensity-band reuse on top of the current
    shipped Milestone 5 behavior.
    """
    recent = history[-window:]
    if not recent:
        return 1.0
    key_penalty = sum(
        0.15 * (0.8**i)
        for i, h in enumerate(reversed(recent))
        if candidate.key is not None and h.get("key") is not None and h.get("key") == candidate.key
    )
    measured_energies = [h.get("energy_rel") for h in recent if h.get("energy_rel") is not None]
    stagnation = (
        0.2
        if len(set(round(e, 1) for e in measured_energies)) <= 2 and len(measured_energies) >= 3
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
    current_energy = current_rel.get("energy_rel") if current_rel.get("energy_rel") is not None else 0.5
    candidate_energy = candidate_rel.get("energy_rel") if candidate_rel.get("energy_rel") is not None else 0.5
    energy_delta = abs(current_energy - candidate_energy)
    dims.append(("energy", energy_delta, 0.30))
    current_vocals = current_rel.get("vocals_rel")
    candidate_vocals = candidate_rel.get("vocals_rel")
    vocal_delta = (
        None
        if current_vocals is None or candidate_vocals is None
        else abs(current_vocals - candidate_vocals)
    )
    dims.append(("vocal", vocal_delta, 0.20))
    current_bass = current_rel.get("bass_rel") if current_rel.get("bass_rel") is not None else 0.5
    candidate_bass = candidate_rel.get("bass_rel") if candidate_rel.get("bass_rel") is not None else 0.5
    bass_delta = abs(current_bass - candidate_bass)
    dims.append(("bass", bass_delta, 0.20))
    current_groove = current_rel.get("groove_rel") if current_rel.get("groove_rel") is not None else 0.5
    candidate_groove = candidate_rel.get("groove_rel") if candidate_rel.get("groove_rel") is not None else 0.5
    groove_delta = abs(current_groove - candidate_groove)
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
    raw = sum((delta or 0.0) * weight for _, delta, weight in dims)
    breadth_bonus = min(0.15, sum(1 for _, delta, _ in dims if delta is not None and delta > 0.25) * 0.05)
    return round(min(1.0, raw + breadth_bonus), 4)


# ---------------------------------------------------------------------------
# Weight resolution
# ---------------------------------------------------------------------------


def resolve_effective_weights(
    playlist_stats: dict[str, Any] | None,
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return effective per-component weights for live scoring.

    Weight precedence:
    1. feedback_tuned_weights
    2. adapted_weights
    3. config-provided static weights
    4. module STATIC_WEIGHTS fallback
    """
    if playlist_stats and playlist_stats.get("feedback_tuned_weights"):
        return dict(playlist_stats["feedback_tuned_weights"])
    if playlist_stats and playlist_stats.get("adapted_weights"):
        return dict(playlist_stats["adapted_weights"])
    if config and config.get("static_weights"):
        return dict(config["static_weights"])
    return dict(STATIC_WEIGHTS)


def resolve_weight_source(
    playlist_stats: dict[str, Any] | None,
) -> str:
    if playlist_stats and playlist_stats.get("feedback_tuned_weights"):
        return "feedback_tuned_weights"
    if playlist_stats and playlist_stats.get("adapted_weights"):
        return "adapted_weights"
    return "static"


# ---------------------------------------------------------------------------
# Key trust / confidence modulation
# ---------------------------------------------------------------------------


def _harmonic_confidence(
    track: ScoringTrackContext,
    harmonic_confidence_floor: float | None = None,
) -> float:
    """Derive harmonic confidence from key trust policy.

    - Corroborated (key_agreement >= 1): full weight (1.0)
    - High-confidence standalone (key_confidence >= 0.5): medium (key_confidence)
    - Weak standalone: floor (max(HARMONIC_CONFIDENCE_FLOOR, key_confidence * 0.5))
    """
    floor = (
        HARMONIC_CONFIDENCE_FLOOR
        if harmonic_confidence_floor is None
        else float(harmonic_confidence_floor)
    )
    # Corroborated: multiple independent sources agree
    if track.key_agreement is not None and track.key_agreement >= 1:
        return 1.0
    # High-confidence standalone
    kc = track.key_confidence or 0.0
    if kc >= KEY_CONFIDENCE_THRESHOLD:
        return kc
    # Weak standalone — clamped to floor
    return max(floor, kc * 0.5)


def build_confidence_map(
    current: ScoringTrackContext,
    candidate: ScoringTrackContext,
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Assemble per-component confidence values for the current→candidate pair.

    Harmonic confidence is the *minimum* of both track key trusts — a weak key
    on either side reduces confidence for the pair.

    All other components default to 1.0 (fully trusted) in v1. Groove and vocal
    confidence modulation will be added when those score components are wired in.
    """
    harmonic_floor = None if config is None else config.get("harmonic_confidence_floor")
    harmonic_conf = min(
        _harmonic_confidence(current, harmonic_floor),
        _harmonic_confidence(candidate, harmonic_floor),
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
    feature_scores: dict[str, float | None],
    weights: dict[str, float],
    confidences: dict[str, float],
    *,
    weight_floors: dict[str, float] | None = None,
    harmonic_confidence_floor: float | None = None,
) -> float:
    """Weight-modulated score in [0, 1].

    Confidence values reduce the effective weight of uncertain signals.
    Weight floors prevent any component from vanishing entirely.
    Components with a score of None (STUB_SCORE) are skipped entirely and the
    remaining weights are renormalized — the weight table is left unchanged.
    This is weight modulation, NOT calibrated statistical probability.
    """
    floors = WEIGHT_FLOORS if weight_floors is None else weight_floors
    confidence_floor = (
        HARMONIC_CONFIDENCE_FLOOR
        if harmonic_confidence_floor is None
        else float(harmonic_confidence_floor)
    )
    adjusted: dict[str, float] = {}
    for feature, weight in weights.items():
        if feature_scores.get(feature) is None:
            continue  # stub — excluded from this scoring pass
        conf = confidences.get(feature, 1.0)
        effective_conf = max(conf, confidence_floor)
        adjusted[feature] = weight * effective_conf
    for feature in list(adjusted):
        if feature in floors:
            adjusted[feature] = max(adjusted[feature], floors[feature])
    total = sum(adjusted.values())
    if total < 1e-6:
        return 0.5
    normalized = {k: v / total for k, v in adjusted.items()}
    return sum(
        normalized[f] * feature_scores[f]  # type: ignore[operator]
        for f in feature_scores
        if f in normalized
    )


# ---------------------------------------------------------------------------
# Transition features
# ---------------------------------------------------------------------------


def compute_transition_features(
    current: ScoringTrackContext,
    candidate: ScoringTrackContext,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical per-pair feature bundle used by penalties, risk, and explanations.

    Window parameters (current_outro, candidate_intro) are deferred.
    Vocal-relative fields stay as None when the analysis pipeline has not populated
    them yet so diagnostics can distinguish "unknown" from "measured silence".
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
    harmonic_floor = None if config is None else config.get("harmonic_confidence_floor")
    trusted_current_key_conf = _harmonic_confidence(current, harmonic_floor)
    trusted_candidate_key_conf = _harmonic_confidence(candidate, harmonic_floor)
    return {
        "effective_bpm_distance": bpm_dist,
        "raw_bpm_distance": raw_bpm_dist,
        "bpm_relationship": bpm_relationship,
        "key_distance": key_distance,
        "key_compat_label": key_compat_label,
        "key_confidence_current": trusted_current_key_conf,
        "key_confidence_candidate": trusted_candidate_key_conf,
        "delta_energy_rel": (
            (candidate.energy_rel if candidate.energy_rel is not None else 0.5)
            - (current.energy_rel if current.energy_rel is not None else 0.5)
        ),
        "delta_bass_rel": (
            (candidate.bass_rel if candidate.bass_rel is not None else 0.5)
            - (current.bass_rel if current.bass_rel is not None else 0.5)
        ),
        "current_vocals_rel": current.vocals_rel,
        "candidate_vocals_rel": candidate.vocals_rel,
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
    cur_vocals = transition_features.get("current_vocals_rel")
    cand_vocals = transition_features.get("candidate_vocals_rel")
    if cur_vocals is not None and cand_vocals is not None:
        vocal_product = cur_vocals * cand_vocals
    else:
        vocal_product = None
    if vocal_product is not None and vocal_product > 0.25:
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
    target: str = "maintain",
) -> list[ScoringTrackContext]:
    """Remove candidates that fail hard filters.

    Hard filters (permissive assistant-mode defaults):
    - same track as current
    - recently played cooldown
    - BPM distance beyond lane-specific hard limit

    The BPM hard limit varies by target lane (bpm_hard_by_target). Ratio-based
    relationships listed in bpm_ratio_pass (e.g. half/double, 3:2/2:3) bypass
    the hard limit entirely — they are intentional tempo pivots, not wide jumps.

    Key is NOT a hard filter. It contributes to score and risk only.
    """
    if bypass_filters is None:
        bypass_filters = set()
    thresholds = config.get("thresholds", {})
    bpm_hard_by_target: dict[str, float] = thresholds.get("bpm_hard_by_target", {})
    bpm_hard = bpm_hard_by_target.get(target, thresholds.get("bpm_hard", 8.0))
    bpm_ratio_pass: set[str] = set(thresholds.get("bpm_ratio_pass", []))
    cooldown_window = int(thresholds.get("cooldown_window", 5))
    if cooldown_window <= 0:
        recent_ids: set[str | None] = set()
    else:
        recent_ids = {h.get("id") or h.get("track_id") for h in history[-cooldown_window:]}

    filtered: list[ScoringTrackContext] = []
    for candidate in candidates:
        # Same-track exclusion (always applied)
        if candidate.track_id == current.track_id:
            continue
        # BPM hard limit — wildcard has no distance cap; ratio relationships in
        # bpm_ratio_pass bypass the distance check.  Non-direct relationships
        # that are NOT in bpm_ratio_pass (e.g. four_three, three_four) are
        # always blocked — they are too adventurous for default recommendations.
        if "bpm" not in bypass_filters and target not in ("wildcard", "contrast"):
            bpm_dist, _, bpm_relationship, _ = effective_bpm_distance(current.bpm, candidate.bpm)
            if bpm_relationship != "direct" and bpm_relationship not in bpm_ratio_pass:
                continue
            if bpm_dist > bpm_hard and bpm_relationship not in bpm_ratio_pass:
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
    vocal_current_rel: float | None,
    vocal_candidate_rel: float | None,
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
    # Reset = room reframe / pressure drop.  Two paths:
    #   (a) clear energy drop — the classic pressure release
    #   (b) moderate dip with bass relief or vocal reframing
    candidate_vocal_reset = (
        vocal_candidate_rel is not None
        and vocal_candidate_rel >= reset_v_t
        and (
            vocal_current_rel is None
            or vocal_candidate_rel >= vocal_current_rel + 0.05
        )
    )
    if delta_energy_rel < reset_e_t:
        return "reset", 0.90, None
    if delta_energy_rel < drop_t and delta_bass_rel < -0.05:
        return "reset", 0.75, "reframe"
    if delta_energy_rel < drop_t and candidate_vocal_reset:
        return "reset", 0.75, "vocal reframe"
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

    cur_voc = transition_features.get("current_vocals_rel")
    cand_voc = transition_features.get("candidate_vocals_rel")
    if cur_voc is not None and cand_voc is not None and cur_voc > 0.6 and cand_voc > 0.6:
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
        confidences = build_confidence_map(current, candidate, config)

    transition_features = compute_transition_features(current, candidate, config)

    target = config.get("target", "maintain")
    feature_scores: dict[str, float] = {
        "target_energy": target_energy_score(
            transition_features["delta_energy_rel"], target
        ),
        # transition_support, vocal_transition, rhythmic_continuity are not yet
        # implemented — STUB_SCORE (None) causes compute_weighted_score to skip
        # them and renormalize over active components only.
        "transition_support": STUB_SCORE,
        "bass_transition": bass_transition_score(
            current.bass_rel, candidate.bass_rel, target
        ),
        "vocal_transition": STUB_SCORE,
        "harmonic": harmonic_score(current.key, candidate.key),
        "tempo": tempo_score(current.bpm, candidate.bpm, config),
        "history_fit": history_fit_score(candidate, history),
        "rhythmic_continuity": STUB_SCORE,
    }

    weights = resolve_effective_weights(playlist_stats, config)
    raw_score = compute_weighted_score(
        feature_scores,
        weights,
        confidences,
        weight_floors=config.get("weight_floors"),
        harmonic_confidence_floor=config.get("harmonic_confidence_floor"),
    )

    penalty_multiplier, penalty_factors = compute_penalties(
        transition_features, feature_scores, confidences, config
    )
    final_score = round(raw_score * penalty_multiplier, 4)

    risk_level, risk_score, risk_factors = compute_risk(transition_features, config)

    move_name, move_confidence, move_note = classify_move(
        transition_features["delta_energy_rel"],
        transition_features.get("delta_bass_rel", 0.0),
        current.vocals_rel,
        candidate.vocals_rel,
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


# ---------------------------------------------------------------------------
# Phase 3: Lane Organization & Top-Level Orchestration
# ---------------------------------------------------------------------------

# Move-type → lane mapping. Contrast is a virtual lane populated via dual-membership.
_MOVE_TO_LANE: dict[str, str] = {
    "jump": "jump",
    "build": "build",
    "maintain": "maintain",
    "reset": "reset",
    "drop": "reset",
}


def compute_ranking_strength(candidate_score: float, lane_scores: list[float]) -> float:
    """Return a [0, 1] strength value showing how this candidate stands against lane peers.

    When lane_scores is empty or has only this candidate, returns 1.0 (top by default).
    The score is position-weighted: 1st place gets 1.0, last place (in a lane of
    max_per_lane entries) gets 0.0, with linear interpolation in between.
    """
    if not lane_scores:
        return 1.0
    sorted_scores = sorted(lane_scores, reverse=True)
    rank = 0
    for i, s in enumerate(sorted_scores):
        if s <= candidate_score:
            rank = i
            break
    n = len(sorted_scores)
    if n == 1:
        return 1.0
    return 1.0 - rank / (n - 1)


def organize_into_lanes(
    ranked_results: list[dict[str, Any]],
    target_lane: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Organize scored candidates into move lanes.

    Lane names: "build", "jump", "maintain", "reset", "wildcard", "contrast".
    - Each candidate's primary lane is derived from its move type via _MOVE_TO_LANE.
    - Candidates with contrast_score >= secondary_contrast_threshold (0.65) that are
      NOT already in the contrast lane get added to "contrast" as secondary members
      (secondary_lane=True).
    - "contrast" is the user-facing exploratory lane.
    - Candidates that don't map to any lane via move type land in "wildcard",
      which is mainly an internal fallback.
    - Each lane is capped at max_per_lane (default 3).
    - lane_order: target_lane first, then remaining lanes in canonical order, then wildcard/contrast.

    Returns dict:
        {
            "lane_order": [...],
            "lanes": {lane_name: [result_with_lane_fields, ...]},
            "target_lane": target_lane,
        }
    """
    cfg = config or {}
    max_per_lane: int = cfg.get("max_per_lane", 3)
    contrast_threshold: float = cfg.get("contrast_threshold", 0.45)
    secondary_contrast_threshold: float = cfg.get("secondary_contrast_threshold", 0.65)

    # Canonical ordering of non-target lanes
    _CANONICAL_ORDER = ["build", "jump", "maintain", "reset", "wildcard", "contrast"]

    lanes: dict[str, list[dict[str, Any]]] = {name: [] for name in _CANONICAL_ORDER}

    # First pass: assign primary lanes.
    # When targeting contrast, route high-contrast tracks to the contrast lane directly.
    for result in ranked_results:
        move = result.get("move", "")
        cs = result.get("contrast_score", 0.0) or 0.0
        if target_lane == "contrast" and cs >= contrast_threshold:
            primary = "contrast"
        else:
            primary = _MOVE_TO_LANE.get(move, "wildcard")
        if len(lanes[primary]) < max_per_lane:
            entry = {**result, "primary_lane": primary, "secondary_lane": False}
            lanes[primary].append(entry)

    # Second pass: contrast dual-membership (only when not already targeting contrast).
    # Iterate ranked_results again (in score order) to fill contrast lane.
    if target_lane != "contrast":
        for result in ranked_results:
            if len(lanes["contrast"]) >= max_per_lane:
                break
            cs = result.get("contrast_score", 0.0) or 0.0
            if cs >= secondary_contrast_threshold:
                move = result.get("move", "")
                primary = _MOVE_TO_LANE.get(move, "wildcard")
                # Add as secondary member (even if primary lane is full — already appended above)
                entry = {**result, "primary_lane": primary, "secondary_lane": True}
                lanes["contrast"].append(entry)

    lane_top_scores = {
        name: max((item.get("score", 0.0) or 0.0) for item in items)
        for name, items in lanes.items()
        if items
    }

    # Build lane_order: target first when available; otherwise promote the best
    # non-empty non-contrast lane so empty-target fallbacks are obvious in the CLI.
    present_lanes = [name for name in _CANONICAL_ORDER if lanes[name]]
    if target_lane in present_lanes:
        ordered = [target_lane] + [lane for lane in present_lanes if lane != target_lane]
    else:
        ordered = sorted(
            [lane for lane in present_lanes if lane != "contrast"],
            key=lambda lane: (-lane_top_scores[lane], _CANONICAL_ORDER.index(lane)),
        )
        if "contrast" in present_lanes:
            ordered.append("contrast")

    return {
        "lane_order": ordered,
        "lanes": {name: items for name, items in lanes.items() if items},
        "target_lane": target_lane,
    }


def compute_recommendation_confidence(
    ranked_results: list[dict[str, Any]],
    analysis_coverage: float = 1.0,
    avg_feature_conf: float = 1.0,
) -> float:
    """Return a [0, 1] recommendation confidence for the full result set.

    v1 formula (simple):
    - Score separation between top-2 candidates (wider = more decisive = higher confidence)
    - Number of candidates available (more candidates = more options = higher confidence)
    - Average feature confidence (quality of underlying analysis)
    - analysis_coverage fraction (what proportion of candidates have rel features)

    All factors blended with fixed weights; result clamped to [0, 1].
    """
    if not ranked_results:
        return 0.0

    scores = [r.get("score", 0.0) or 0.0 for r in ranked_results]
    top_score = scores[0]

    # Separation factor: gap between 1st and 2nd (or 0 if only one candidate)
    if len(scores) >= 2:
        gap = top_score - scores[1]
        # A gap of 0.3+ is considered decisive; normalize to [0, 1]
        separation_factor = min(gap / 0.30, 1.0)
    else:
        separation_factor = 1.0  # only one candidate — decisive by default

    # Depth factor: ≥10 candidates → 1.0
    depth_factor = min(len(ranked_results) / 10.0, 1.0)

    # Combine: separation 40%, depth 20%, feature conf 25%, coverage 15%
    confidence = (
        0.40 * separation_factor
        + 0.20 * depth_factor
        + 0.25 * avg_feature_conf
        + 0.15 * analysis_coverage
    )
    return max(0.0, min(1.0, confidence))


def get_recommendations(
    current_track: ScoringTrackContext,
    candidates: list[ScoringTrackContext],
    history: list[dict[str, Any] | ScoringTrackContext],
    config: dict[str, Any],
    playlist_stats: dict[str, Any] | None = None,
    target: str = "maintain",
    max_per_lane: int | None = None,
) -> dict[str, Any]:
    """Top-level recommendation entry point.

    Filters candidates, scores each one, organizes into lanes, computes
    recommendation confidence, and returns the structured result.

    Returns:
        {
            "lane_order": [...],
            "lanes": {lane_name: [scored_result, ...]},
            "recommendation_confidence": float,
            "meta": {
                "target": str,
                "total_candidates": int,
                "scored_candidates": int,
                "current_track_id": str,
            }
        }
    """
    normalized_history: list[dict[str, Any]] = []
    for item in history:
        if isinstance(item, dict):
            normalized_history.append(item)
        elif is_dataclass(item):
            normalized_history.append(asdict(item))
        else:
            normalized_history.append(dict(item))

    effective_max_per_lane = config.get("max_per_lane", 3) if max_per_lane is None else max_per_lane
    cfg = {**config, "max_per_lane": effective_max_per_lane, "target": target}

    # Filter candidates (removes current track, cooldown, BPM hard limit)
    filtered = filter_candidates(current_track, candidates, normalized_history, cfg, target=target)

    # Score each candidate
    scored: list[dict[str, Any]] = []
    for candidate in filtered:
        result = score_candidate(
            current=current_track,
            candidate=candidate,
            history=normalized_history,
            config=cfg,
            playlist_stats=playlist_stats,
        )
        scored.append(result)

    # Sort by final score descending
    ranked = sorted(scored, key=lambda r: r.get("score", 0.0), reverse=True)

    # Lane organization
    lane_result = organize_into_lanes(ranked, target_lane=target, config=cfg)

    # Average feature confidence across all scored candidates (for recommendation confidence)
    if ranked:
        all_confs: list[float] = []
        for r in ranked:
            component_scores = r.get("component_scores") or {}
            for feature, value in (r.get("confidences") or {}).items():
                if component_scores.get(feature) is None:
                    continue
                all_confs.append(float(value))
        avg_feature_conf = sum(all_confs) / len(all_confs) if all_confs else 1.0
    else:
        avg_feature_conf = 1.0

    # Analysis coverage: fraction of candidates with rel features (non-None energy_rel)
    total_input = len(candidates)
    if total_input > 0:
        rel_count = sum(1 for c in candidates if c.energy_rel is not None)
        analysis_coverage = rel_count / total_input
    else:
        analysis_coverage = 0.0

    rec_confidence = compute_recommendation_confidence(
        ranked,
        analysis_coverage=analysis_coverage,
        avg_feature_conf=avg_feature_conf,
    )

    target_lane_items = lane_result["lanes"].get(target, [])
    requested_lane_available = bool(target_lane_items)
    lane_order = lane_result["lane_order"]
    best_alternative_lanes = lane_order[:2] if not requested_lane_available else []

    fallback_note: str | None = None
    if not requested_lane_available:
        if len(best_alternative_lanes) >= 2:
            fallback_note = (
                f"No strong {target} candidates found; best alternatives are "
                f"{best_alternative_lanes[0]} and {best_alternative_lanes[1]}."
            )
        elif len(best_alternative_lanes) == 1:
            fallback_note = (
                f"No strong {target} candidates found; best alternative is "
                f"{best_alternative_lanes[0]}."
            )

    return {
        **lane_result,
        "recommendation_confidence": rec_confidence,
        "meta": {
            "target": target,
            "total_candidates": total_input,
            "filtered_candidates": len(filtered),
            "scored_candidates": len(ranked),
            "current_track_id": current_track.track_id,
            "requested_lane_available": requested_lane_available,
            "best_alternative_lanes": best_alternative_lanes,
            "fallback_note": fallback_note,
        },
    }
