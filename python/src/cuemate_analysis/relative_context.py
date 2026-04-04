from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any

import numpy as np

from cuemate_analysis.config import RuntimeSettings, build_relative_experiment_signature


LOW_DISCRIMINATION_THRESHOLDS = {
    "energy_abs": 0.08,
    "bass_abs": 0.06,
    "drums_abs": 0.06,
    "vocals_abs": 0.10,
    "groove_abs": 0.05,
    "harmonic_abs": 0.05,
}

DIMENSION_TO_SCORER = {
    "energy_abs": "target_energy",
    "bass_abs": "bass_transition",
    "drums_abs": "rhythmic_continuity",
    "vocals_abs": "vocal_transition",
    "groove_abs": "rhythmic_continuity",
    "harmonic_abs": "harmonic",
}

VOCAL_CONFIDENCE_THRESHOLD = 0.5


@dataclass(frozen=True)
class RelativeTrackInput:
    playlist_id: str
    playlist_name: str
    track_id: str
    position: int
    file_path: str
    title: str | None
    artist: str | None
    has_absolute_analysis: bool
    bpm: float | None
    key: str | None
    energy_abs: float | None
    energy_learned: float | None
    energy_learned_bucket: str | None
    bass_abs: float | None
    drums_abs: float | None
    harmonic_abs: float | None
    groove_abs: float | None
    vocals_abs: float | None
    vocals_confidence: float | None
    analyzed_at: str | None
    analysis_signature: str | None
    config_signature: str | None


@dataclass(frozen=True)
class RelativeTrackPreview:
    track_id: str
    playlist_id: str
    position: int
    title: str | None
    artist: str | None
    file_path: str
    energy_source_used: str
    energy_rel: float
    bass_rel: float
    drums_rel: float
    vocals_rel: float | None
    groove_rel: float
    energy_spread: float
    bass_spread: float
    drums_spread: float
    vocals_spread: float
    groove_spread: float
    intensity_band: str
    intensity_membership: dict[str, float]
    role_hints: list[str]
    valid_as_of_track_count: int
    analyzed_at: str | None
    analysis_signature: str | None
    config_signature: str | None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlaylistStatsPreview:
    playlist_id: str
    track_count_total: int
    track_count_analyzed: int
    eligible_track_count: int
    avg_harmonic: float | None
    key_diversity: float | None
    bpm_range: float | None
    energy_spread: float | None
    bass_spread: float | None
    drums_spread: float | None
    vocals_spread: float | None
    harmonic_spread: float | None
    groove_spread: float | None
    adapted_weights: dict[str, float] | None
    adaptation_strength: float | None
    weight_adaptation_notes: list[str]
    status: str
    relative_signature: str

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelativePlaylistPreview:
    playlist: str
    playlist_id: str | None
    is_limited: bool
    limited_track_count: int
    playlist_stats: PlaylistStatsPreview
    tracks: list[RelativeTrackPreview]

    def to_payload(self) -> dict[str, Any]:
        return {
            "playlist": self.playlist,
            "playlist_id": self.playlist_id,
            "is_limited": self.is_limited,
            "limited_track_count": self.limited_track_count,
            "playlist_stats": self.playlist_stats.to_payload(),
            "tracks": [track.to_payload() for track in self.tracks],
        }


def row_to_relative_track_input(row) -> RelativeTrackInput:
    row_keys = set(row.keys()) if hasattr(row, "keys") else set()

    def read_optional(key: str):
        if row_keys and key not in row_keys:
            return None
        return row[key]

    return RelativeTrackInput(
        playlist_id=str(row["playlist_id"]),
        playlist_name=str(row["playlist_name"]),
        track_id=str(row["track_id"]),
        position=int(row["position"]),
        file_path=str(row["file_path"]),
        title=read_optional("title"),
        artist=read_optional("artist"),
        has_absolute_analysis=bool(row["has_absolute_analysis"]),
        bpm=float(read_optional("bpm")) if read_optional("bpm") is not None else None,
        key=str(read_optional("key")) if read_optional("key") is not None else None,
        energy_abs=float(read_optional("energy_abs")) if read_optional("energy_abs") is not None else None,
        energy_learned=float(read_optional("energy_learned")) if read_optional("energy_learned") is not None else None,
        energy_learned_bucket=str(read_optional("energy_learned_bucket")) if read_optional("energy_learned_bucket") is not None else None,
        bass_abs=float(read_optional("bass_abs")) if read_optional("bass_abs") is not None else None,
        drums_abs=float(read_optional("drums_abs")) if read_optional("drums_abs") is not None else None,
        harmonic_abs=float(read_optional("harmonic_abs")) if read_optional("harmonic_abs") is not None else None,
        groove_abs=float(read_optional("groove_abs")) if read_optional("groove_abs") is not None else None,
        vocals_abs=float(read_optional("vocals_abs")) if read_optional("vocals_abs") is not None else None,
        vocals_confidence=float(read_optional("vocals_confidence")) if read_optional("vocals_confidence") is not None else None,
        analyzed_at=read_optional("analyzed_at"),
        analysis_signature=read_optional("analysis_signature"),
        config_signature=read_optional("config_signature"),
    )


def robust_scale(value: float, p10: float, p90: float, fallback: float = 0.5) -> float:
    spread = p90 - p10
    if spread < 1e-6:
        return fallback
    return max(0.0, min(1.0, (value - p10) / spread))


def percentile_spread(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, 90) - np.percentile(values, 10))


def assign_intensity_band(
    energy_rel: float,
    vocal_rel: float | None,
    drums_rel: float,
    previous_band: str | None = None,
    previous_energy_rel: float | None = None,
) -> str:
    rules = [
        ("Peak", lambda e, v, d: e > 0.75),
        ("Drive", lambda e, v, d: e > 0.50),
        ("Groove", lambda e, v, d: e > 0.25),
        ("Low", lambda e, v, d: True),
    ]
    candidate = "Low"
    for band_name, rule in rules:
        if rule(energy_rel, vocal_rel, drums_rel):
            candidate = band_name
            break
    if (
        previous_band is not None
        and previous_energy_rel is not None
        and candidate != previous_band
        and abs(energy_rel - previous_energy_rel) < 0.08
    ):
        return previous_band
    return candidate


def compute_intensity_membership(
    energy_rel: float,
    vocal_rel: float | None,
    drums_rel: float,
) -> dict[str, float]:
    def gaussian(x: float, center: float, width: float) -> float:
        return math.exp(-0.5 * ((x - center) / max(width, 1e-6)) ** 2)

    def sigmoid(x: float, center: float, spread: float) -> float:
        z = (x - center) / max(spread, 1e-6)
        return 1.0 / (1.0 + math.exp(-z))

    return {
        "low": round(max(0.0, 1.0 - energy_rel * 3.0), 3),
        "groove": round(gaussian(energy_rel, center=0.40, width=0.18), 3),
        "drive": round(sigmoid(energy_rel, center=0.55, spread=0.12), 3),
        "peak": round(sigmoid(energy_rel, center=0.80, spread=0.10), 3),
    }


def assign_role_hints(
    energy_rel: float,
    vocal_rel: float | None,
    bass_rel: float | None,
    drums_rel: float | None,
    *,
    vocals_confidence: float | None,
) -> list[str]:
    roles: list[str] = []
    vocals_known = vocal_rel is not None and vocals_confidence is not None and vocals_confidence >= VOCAL_CONFIDENCE_THRESHOLD
    bass_value = 0.5 if bass_rel is None else bass_rel
    drums_value = 0.5 if drums_rel is None else drums_rel

    if vocals_known and vocal_rel is not None and vocal_rel > 0.75:
        roles.append("vocal_feature")
    if vocals_known and vocal_rel is not None and bass_value > 0.85 and vocal_rel < 0.3:
        roles.append("bass_driver")
    if energy_rel < 0.20:
        roles.append("opener")
    if energy_rel < 0.30:
        roles.append("relief_track")
    if 0.40 <= energy_rel <= 0.60:
        roles.append("steady_energy")
    if energy_rel > 0.70 and drums_value > 0.80:
        roles.append("pressure_builder")
    if energy_rel > 0.80:
        roles.append("peak_tool")
    if not roles:
        roles.append("wildcard_candidate")
    return roles


def apply_low_discrimination_adjustment(
    weights: dict[str, float],
    playlist_stats: dict[str, float | None],
    *,
    weight_floors: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    adjusted = dict(weights)
    notes: list[str] = []
    for dimension, threshold in LOW_DISCRIMINATION_THRESHOLDS.items():
        spread_key = dimension.replace("_abs", "_spread")
        spread = playlist_stats.get(spread_key)
        scorer_key = DIMENSION_TO_SCORER.get(dimension)
        if spread is None or scorer_key is None or scorer_key not in adjusted:
            continue
        if float(spread) < threshold:
            old_value = adjusted[scorer_key]
            new_value = max(old_value * 0.5, weight_floors.get(scorer_key, 0.02))
            adjusted[scorer_key] = new_value
            notes.append(
                f"{dimension} spread {float(spread):.3f} < {threshold:.3f} -> "
                f"{scorer_key} {old_value:.3f} -> {new_value:.3f}"
            )
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {key: value / total for key, value in adjusted.items()}
    return adjusted, notes


def build_weight_profile_from_playlist_features(
    playlist_features: list[RelativeTrackInput],
    *,
    static_weights: dict[str, float],
    weight_floors: dict[str, float],
    adaptation_strength: float,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    if len(playlist_features) < 12:
        return None, {}

    avg_harmonic = float(np.mean([track.harmonic_abs for track in playlist_features if track.harmonic_abs is not None]))
    harmonic_significance = min(1.0, avg_harmonic / 0.6)
    unique_keys = len({track.key for track in playlist_features if track.key})
    key_diversity = min(1.0, unique_keys / min(len(playlist_features), 12))
    drums_values = [float(track.drums_abs) for track in playlist_features if track.drums_abs is not None]
    bass_values = [float(track.bass_abs) for track in playlist_features if track.bass_abs is not None]
    energy_values = [float(track.energy_abs) for track in playlist_features if track.energy_abs is not None]
    harmonic_values = [float(track.harmonic_abs) for track in playlist_features if track.harmonic_abs is not None]
    groove_values = [float(track.groove_abs) for track in playlist_features if track.groove_abs is not None]
    vocal_values = [float(track.vocals_abs) for track in playlist_features if track.vocals_abs is not None]
    bpm_values = [float(track.bpm) for track in playlist_features if track.bpm is not None]

    drums_spread = percentile_spread(drums_values)
    bass_spread = percentile_spread(bass_values)
    energy_spread = percentile_spread(energy_values)
    harmonic_spread = percentile_spread(harmonic_values)
    vocal_spread = percentile_spread(vocal_values) if len(vocal_values) > 5 else 0.3
    groove_spread = percentile_spread(groove_values) if len(groove_values) > 5 else 0.2
    bpm_range = (max(bpm_values) - min(bpm_values)) if bpm_values else 0.0
    tempo_significance = min(1.0, bpm_range / 20.0)

    raw = {
        "target_energy": 0.22 * (0.7 + energy_spread * 0.3),
        "transition_support": 0.18,
        "bass_transition": 0.15 * (0.6 + bass_spread * 0.8 + drums_spread * 0.4),
        "vocal_transition": 0.13 * (0.5 + vocal_spread * 1.0),
        "harmonic": 0.12 * harmonic_significance * (0.5 + key_diversity * 0.5),
        "tempo": 0.10 * tempo_significance,
        "history_fit": 0.06,
        "rhythmic_continuity": 0.04,
    }
    for key in raw:
        raw[key] = max(raw[key], weight_floors.get(key, 0.02))
    total = sum(raw.values())
    adapted_base = {key: value / total for key, value in raw.items()} if total > 0 else dict(raw)
    adaptation_strength = max(0.0, min(1.0, adaptation_strength))
    base = {
        key: ((1.0 - adaptation_strength) * static_weights.get(key, 0.0))
        + (adaptation_strength * adapted_base.get(key, 0.0))
        for key in set(static_weights) | set(adapted_base)
    }
    base_total = sum(base.values())
    if base_total > 0:
        base = {key: value / base_total for key, value in base.items()}

    stats = {
        "avg_harmonic": avg_harmonic,
        "key_diversity": key_diversity,
        "bpm_range": bpm_range,
        "energy_spread": energy_spread,
        "bass_spread": bass_spread,
        "drums_spread": drums_spread,
        "vocals_spread": vocal_spread,
        "harmonic_spread": harmonic_spread,
        "groove_spread": groove_spread,
    }
    adjusted, notes = apply_low_discrimination_adjustment(base, stats, weight_floors=weight_floors)
    stats["adapted_weights"] = adjusted
    stats["adaptation_strength"] = adaptation_strength
    stats["weight_adaptation_notes"] = [
        f"adaptation_strength blend = {adaptation_strength:.2f}",
        "adapted weights are blended with STATIC_WEIGHTS before low-discrimination adjustment",
        *notes,
    ]
    return adjusted, stats


def compute_relative_playlist_preview(
    rows: list[RelativeTrackInput],
    settings: RuntimeSettings,
    *,
    playlist_name: str,
    is_limited: bool,
    energy_source: str = "heuristic",
) -> RelativePlaylistPreview:
    relative_signature = build_relative_experiment_signature(settings)
    playlist_id = rows[0].playlist_id if rows else None
    track_count_total = len(rows)
    analyzed_tracks = [row for row in rows if row.has_absolute_analysis]
    eligible_tracks = [
        row
        for row in analyzed_tracks
        if (row.energy_learned if energy_source == "learned" and row.energy_learned is not None else row.energy_abs) is not None
        and row.bass_abs is not None
        and row.drums_abs is not None
        and row.harmonic_abs is not None
        and row.groove_abs is not None
        and row.bpm is not None
        and row.key is not None
    ]
    track_count_analyzed = len(analyzed_tracks)
    eligible_track_count = len(eligible_tracks)

    if eligible_track_count < settings.thresholds.min_playlist_for_relative:
        stats = PlaylistStatsPreview(
            playlist_id=playlist_id or "",
            track_count_total=track_count_total,
            track_count_analyzed=track_count_analyzed,
            eligible_track_count=eligible_track_count,
            avg_harmonic=None,
            key_diversity=None,
            bpm_range=None,
            energy_spread=None,
            bass_spread=None,
            drums_spread=None,
            vocals_spread=None,
            harmonic_spread=None,
            groove_spread=None,
            adapted_weights=None,
            adaptation_strength=None,
            weight_adaptation_notes=[
                f"Relative analysis requires at least {settings.thresholds.min_playlist_for_relative} eligible tracks.",
            ],
            status="insufficient_tracks",
            relative_signature=relative_signature,
        )
        return RelativePlaylistPreview(
            playlist=playlist_name,
            playlist_id=playlist_id,
            is_limited=is_limited,
            limited_track_count=track_count_total,
            playlist_stats=stats,
            tracks=[],
        )

    def bounds(values: list[float]) -> tuple[float, float]:
        return float(np.percentile(values, 10)), float(np.percentile(values, 90))

    energy_values = [
        float(track.energy_learned if energy_source == "learned" and track.energy_learned is not None else track.energy_abs)
        for track in eligible_tracks
        if (track.energy_learned if energy_source == "learned" and track.energy_learned is not None else track.energy_abs) is not None
    ]
    bass_values = [float(track.bass_abs) for track in eligible_tracks if track.bass_abs is not None]
    drums_values = [float(track.drums_abs) for track in eligible_tracks if track.drums_abs is not None]
    groove_values = [float(track.groove_abs) for track in eligible_tracks if track.groove_abs is not None]
    harmonic_values = [float(track.harmonic_abs) for track in eligible_tracks if track.harmonic_abs is not None]
    vocal_values = [float(track.vocals_abs) for track in eligible_tracks if track.vocals_abs is not None]
    bpm_values = [float(track.bpm) for track in eligible_tracks if track.bpm is not None]

    energy_bounds = bounds(energy_values)
    bass_bounds = bounds(bass_values)
    drums_bounds = bounds(drums_values)
    groove_bounds = bounds(groove_values)
    vocal_bounds = bounds(vocal_values) if vocal_values else (0.0, 1.0)

    energy_spread = percentile_spread(energy_values)
    bass_spread = percentile_spread(bass_values)
    drums_spread = percentile_spread(drums_values)
    harmonic_spread = percentile_spread(harmonic_values)
    groove_spread = percentile_spread(groove_values) if len(groove_values) > 5 else 0.2
    vocals_spread = percentile_spread(vocal_values) if len(vocal_values) > 5 else 0.3
    avg_harmonic = float(np.mean(harmonic_values)) if harmonic_values else None
    key_diversity = min(1.0, len({track.key for track in eligible_tracks if track.key}) / min(eligible_track_count, 12))
    bpm_range = (max(bpm_values) - min(bpm_values)) if bpm_values else None

    previews: list[RelativeTrackPreview] = []
    for track in eligible_tracks:
        effective_energy = float(
            track.energy_learned if energy_source == "learned" and track.energy_learned is not None else track.energy_abs
        )
        energy_rel = robust_scale(effective_energy, *energy_bounds)
        bass_rel = robust_scale(float(track.bass_abs), *bass_bounds)
        drums_rel = robust_scale(float(track.drums_abs), *drums_bounds)
        groove_rel = robust_scale(float(track.groove_abs), *groove_bounds)
        vocals_rel = robust_scale(float(track.vocals_abs), *vocal_bounds) if track.vocals_abs is not None and vocal_values else None
        previews.append(
            RelativeTrackPreview(
                track_id=track.track_id,
                playlist_id=track.playlist_id,
                position=track.position,
                title=track.title,
                artist=track.artist,
                file_path=track.file_path,
                energy_source_used="learned" if energy_source == "learned" and track.energy_learned is not None else "heuristic",
                energy_rel=round(energy_rel, 4),
                bass_rel=round(bass_rel, 4),
                drums_rel=round(drums_rel, 4),
                vocals_rel=None if vocals_rel is None else round(vocals_rel, 4),
                groove_rel=round(groove_rel, 4),
                energy_spread=round(energy_spread, 4),
                bass_spread=round(bass_spread, 4),
                drums_spread=round(drums_spread, 4),
                vocals_spread=round(vocals_spread, 4),
                groove_spread=round(groove_spread, 4),
                intensity_band=assign_intensity_band(energy_rel, vocals_rel, drums_rel, previous_band=None, previous_energy_rel=None),
                intensity_membership=compute_intensity_membership(energy_rel, vocals_rel, drums_rel),
                role_hints=assign_role_hints(
                    energy_rel,
                    vocals_rel,
                    bass_rel,
                    drums_rel,
                    vocals_confidence=track.vocals_confidence,
                ),
                valid_as_of_track_count=eligible_track_count,
                analyzed_at=track.analyzed_at,
                analysis_signature=track.analysis_signature,
                config_signature=track.config_signature,
            )
        )

    adapted_weights: dict[str, float] | None = None
    adaptation_strength: float | None = settings.weight_adaptation.adaptation_strength
    weight_adaptation_notes: list[str] = []
    if settings.weight_adaptation.mode == "auto" and eligible_track_count >= settings.thresholds.small_playlist_limit:
        adapted_weights, weight_stats = build_weight_profile_from_playlist_features(
            eligible_tracks,
            static_weights=settings.scoring.static_weights,
            weight_floors=settings.scoring.weight_floors,
            adaptation_strength=settings.weight_adaptation.adaptation_strength,
        )
        if weight_stats:
            adaptation_strength = float(weight_stats["adaptation_strength"])
            weight_adaptation_notes = [str(item) for item in weight_stats["weight_adaptation_notes"]]
    else:
        weight_adaptation_notes.append(
            f"Eligible track count {eligible_track_count} < {settings.thresholds.small_playlist_limit}; adapted weights skipped."
        )

    stats = PlaylistStatsPreview(
        playlist_id=playlist_id or "",
        track_count_total=track_count_total,
        track_count_analyzed=track_count_analyzed,
        eligible_track_count=eligible_track_count,
        avg_harmonic=None if avg_harmonic is None else round(avg_harmonic, 4),
        key_diversity=round(key_diversity, 4),
        bpm_range=None if bpm_range is None else round(float(bpm_range), 4),
        energy_spread=round(energy_spread, 4),
        bass_spread=round(bass_spread, 4),
        drums_spread=round(drums_spread, 4),
        vocals_spread=round(vocals_spread, 4),
        harmonic_spread=round(harmonic_spread, 4),
        groove_spread=round(groove_spread, 4),
        adapted_weights=None if adapted_weights is None else {key: round(value, 6) for key, value in adapted_weights.items()},
        adaptation_strength=None if adaptation_strength is None else round(adaptation_strength, 4),
        weight_adaptation_notes=weight_adaptation_notes,
        status="ok" if adapted_weights is not None else "relative_only",
        relative_signature=relative_signature,
    )
    return RelativePlaylistPreview(
        playlist=playlist_name,
        playlist_id=playlist_id,
        is_limited=is_limited,
        limited_track_count=track_count_total,
        playlist_stats=stats,
        tracks=sorted(previews, key=lambda track: track.position),
    )


def preview_to_json(preview: RelativePlaylistPreview) -> str:
    return json.dumps(preview.to_payload(), indent=2, sort_keys=True)
