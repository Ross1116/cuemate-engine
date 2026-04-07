from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImportedTrack:
    id: str
    file_path: Path
    file_hash: str
    title: str | None
    artist: str | None
    genre: str | None
    duration_seconds: float | None
    bpm_imported: float | None
    bpm_tag: float | None
    key_imported: str | None
    key_tag: str | None
    import_source: str = "local_files"

    def to_track_row(self, timestamp: str) -> dict[str, Any]:
        return {
            "id": self.id,
            "file_path": self.file_path.as_posix(),
            "file_hash": self.file_hash,
            "title": self.title,
            "artist": self.artist,
            "genre": self.genre,
            "duration_seconds": self.duration_seconds,
            "imported_bpm": self.bpm_imported,
            "imported_key": self.key_imported,
            "import_source": self.import_source,
            "imported_at": timestamp,
            "updated_at": timestamp,
        }


@dataclass(frozen=True, kw_only=True)
class AnalysisResult:
    track_id: str
    source_file_hash: str
    bpm: float
    bpm_confidence: float
    bpm_source: str
    time_signature: str
    time_signature_confidence: float
    key: str
    key_number: int
    key_letter: str
    key_confidence: float
    key_source: str
    key_imported: str | None
    key_tagged: str | None
    key_agreement: int | None
    energy_abs: float
    energy_heuristic_abs: float | None = None
    energy_sustained: float | None = None
    energy_peak: float | None = None
    danceability_abs: float | None = None
    arousal_abs: float | None = None
    valence_abs: float | None = None
    mood_aggressive_abs: float | None = None
    mood_party_abs: float | None = None
    mood_relaxed_abs: float | None = None
    energy_essentia_fused: float | None = None
    energy_essentia_bucket: str | None = None
    essentia_semantic_signature: str | None = None
    essentia_semantic_source: str | None = None
    essentia_semantic_inferred_at: str | None = None
    loudness_lufs: float
    loudness_norm: float
    bass_abs: float
    drums_abs: float | None
    harmonic_abs: float | None
    groove_abs: float | None
    vocals_abs: float | None
    vocals_confidence: float | None
    analysis_mode: str
    analyzed_at: str
    analysis_signature: str
    config_signature: str
    scoring_contract_id_at_analysis: str | None = None

    def to_db_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["user_id"] = "local"
        return payload


@dataclass(frozen=True, kw_only=True)
class FastAnalysisResult:
    track_id: str
    source_file_hash: str
    bpm: float
    bpm_confidence: float
    bpm_source: str
    key: str
    key_number: int
    key_letter: str
    key_confidence: float
    key_source: str
    key_imported: str | None
    key_tagged: str | None
    key_agreement: int | None
    analyzed_at: str
    analysis_signature: str
    config_signature: str

    def to_db_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["user_id"] = "local"
        return payload