from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

from cuemate_analysis.models import AnalysisResult, FastAnalysisResult, ImportedTrack


logger = logging.getLogger(__name__)


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def upsert_track(self, track: ImportedTrack, timestamp: str) -> None:
        payload = track.to_track_row(timestamp)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO tracks (
                  id, user_id, file_path, file_hash, title, artist, genre,
                  duration_seconds, imported_bpm, imported_key, import_source, imported_at, updated_at
                ) VALUES (
                  :id, 'local', :file_path, :file_hash, :title, :artist, :genre,
                  :duration_seconds, :imported_bpm, :imported_key, :import_source, :imported_at, :updated_at
                )
                ON CONFLICT(id) DO UPDATE SET
                  file_path = excluded.file_path,
                  file_hash = excluded.file_hash,
                  title = excluded.title,
                  artist = excluded.artist,
                  genre = excluded.genre,
                  duration_seconds = excluded.duration_seconds,
                  imported_bpm = excluded.imported_bpm,
                  imported_key = excluded.imported_key,
                  import_source = excluded.import_source,
                  updated_at = excluded.updated_at
                """,
                payload,
            )

    def upsert_playlist(self, playlist_id: str, name: str, track_count: int, timestamp: str) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO playlists (id, user_id, name, track_count, created_at, updated_at)
                VALUES (?, 'local', ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  name = excluded.name,
                  track_count = excluded.track_count,
                  updated_at = excluded.updated_at
                """,
                (playlist_id, name, track_count, timestamp, timestamp),
            )

    def replace_playlist_tracks(
        self, playlist_id: str, track_ids: Iterable[str], timestamp: str
    ) -> None:
        count = 0
        with self.connection:
            self.connection.execute("DELETE FROM playlist_tracks WHERE playlist_id = ?", (playlist_id,))
            for position, track_id in enumerate(track_ids, start=1):
                count = position
                self.connection.execute(
                    """
                    INSERT INTO playlist_tracks (playlist_id, track_id, position, added_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (playlist_id, track_id, position, timestamp),
                )
            self.connection.execute(
                "UPDATE playlists SET track_count = ?, updated_at = ? WHERE id = ?",
                (count, timestamp, playlist_id),
            )
            # Mark canonical relative state stale for this playlist (if a stats row exists).
            self.connection.execute(
                """
                UPDATE playlist_stats
                SET is_stale = 1, stale_reason = 'playlist_membership_changed', stale_marked_at = ?
                WHERE playlist_id = ?
                """,
                (timestamp, playlist_id),
            )

    def get_playlist(self, name: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM playlists WHERE name = ?",
            (name,),
        ).fetchone()

    def get_playlist_name_by_id(self, playlist_id: str) -> str | None:
        row = self.connection.execute(
            "SELECT name FROM playlists WHERE id = ?",
            (playlist_id,),
        ).fetchone()
        return str(row[0]) if row is not None else None

    def get_playlist_tracks(self, playlist_name: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT
              p.id AS playlist_id,
              p.name AS playlist_name,
              pt.position,
              t.id AS track_id,
              t.file_path,
              t.file_hash,
              t.title,
              t.artist,
              t.genre,
              t.duration_seconds,
              t.imported_bpm,
              t.imported_key,
              t.import_source,
              ff.analysis_signature AS fast_analysis_signature,
              ff.config_signature AS fast_config_signature,
              ff.source_file_hash AS fast_source_file_hash,
              ff.bpm AS fast_bpm,
              ff.bpm_confidence AS fast_bpm_confidence,
              ff.bpm_source AS fast_bpm_source,
              ff.key AS fast_key,
              ff.key_number AS fast_key_number,
              ff.key_letter AS fast_key_letter,
              ff.key_confidence AS fast_key_confidence,
              ff.key_source AS fast_key_source,
              ff.key_imported AS fast_key_imported,
              ff.key_tagged AS fast_key_tagged,
              ff.key_agreement AS fast_key_agreement,
              ff.analyzed_at AS fast_analyzed_at,
              f.analysis_mode,
              f.analysis_signature,
              f.config_signature,
              f.source_file_hash,
              f.bpm,
              f.key,
              f.energy_abs,
              f.danceability_abs,
              f.arousal_abs,
              f.valence_abs,
              f.mood_aggressive_abs,
              f.mood_party_abs,
              f.mood_relaxed_abs,
              f.energy_essentia_fused,
              f.energy_essentia_bucket,
              f.essentia_semantic_signature,
              f.essentia_semantic_source,
              f.essentia_semantic_inferred_at,
              f.loudness_norm,
              f.bass_abs,
              f.drums_abs,
              f.groove_abs,
              f.analyzed_at
            FROM playlists p
            JOIN playlist_tracks pt ON pt.playlist_id = p.id
            JOIN tracks t ON t.id = pt.track_id
            LEFT JOIN track_features_fast ff ON ff.track_id = t.id
            LEFT JOIN track_features_abs f ON f.track_id = t.id
            WHERE p.name = ?
            ORDER BY pt.position ASC
            """,
            (playlist_name,),
        ).fetchall()

    def get_table_columns(self, table_name: str) -> set[str]:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
            raise ValueError(f"Unsafe table name: {table_name}")
        exists = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        if exists is None:
            return set()
        rows = self.connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(row[1]) for row in rows}

    def get_playlist_relative_inputs(self, playlist_name: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT
              p.id AS playlist_id,
              p.name AS playlist_name,
              pt.position,
              t.id AS track_id,
              t.file_path,
              t.title,
              t.artist,
              f.track_id IS NOT NULL AS has_absolute_analysis,
              f.bpm,
              f.key,
              f.key_confidence,
              f.key_source,
              f.key_agreement,
              f.energy_abs,
              f.energy_heuristic_abs,
              f.energy_essentia_fused,
              f.energy_essentia_bucket,
              f.danceability_abs,
              f.arousal_abs,
              f.valence_abs,
              f.mood_aggressive_abs,
              f.mood_party_abs,
              f.mood_relaxed_abs,
              f.bass_abs,
              f.drums_abs,
              f.harmonic_abs,
              f.groove_abs,
              f.vocals_abs,
              f.vocals_confidence,
              f.essentia_semantic_signature,
              f.essentia_semantic_source,
              f.analyzed_at,
              f.analysis_signature,
              f.config_signature
            FROM playlists p
            JOIN playlist_tracks pt ON pt.playlist_id = p.id
            JOIN tracks t ON t.id = pt.track_id
            LEFT JOIN track_features_abs f ON f.track_id = t.id
            WHERE p.name = ?
            ORDER BY pt.position ASC
            """,
            (playlist_name,),
        ).fetchall()

    def get_existing_analysis(self, track_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM track_features_abs WHERE track_id = ?",
            (track_id,),
        ).fetchone()

    def get_existing_fast_analysis(self, track_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM track_features_fast WHERE track_id = ?",
            (track_id,),
        ).fetchone()

    def create_analysis_job(
        self,
        *,
        playlist_id: str,
        track_id: str,
        track_path: str,
        analysis_mode: str,
        analysis_signature: str,
        config_signature: str,
        source_file_hash: str,
        priority: int,
        created_at: str,
    ) -> int:
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO analysis_jobs (
                  playlist_id, track_id, track_path, job_kind, status, priority, analysis_mode,
                  analysis_signature, config_signature, source_file_hash, created_at
                ) VALUES (?, ?, ?, 'full', 'pending', ?, ?, ?, ?, ?, ?)
                """,
                (
                    playlist_id,
                    track_id,
                    track_path,
                    priority,
                    analysis_mode,
                    analysis_signature,
                    config_signature,
                    source_file_hash,
                    created_at,
                ),
            )
        return int(cursor.lastrowid)

    def create_analysis_job_with_kind(
        self,
        *,
        playlist_id: str | None,
        track_id: str,
        track_path: str,
        job_kind: str,
        analysis_mode: str,
        analysis_signature: str,
        config_signature: str,
        source_file_hash: str,
        priority: int,
        created_at: str,
    ) -> int:
        with self.connection:
            self.connection.execute(
                """
                DELETE FROM analysis_jobs
                WHERE track_id = ?
                AND job_kind = ?
                AND status = 'pending'
                AND analysis_signature = ?
                AND config_signature = ?
                """,
                (track_id, job_kind, analysis_signature, config_signature),
            )

            cursor = self.connection.execute(
                """
                INSERT INTO analysis_jobs (
                playlist_id, track_id, track_path, job_kind, status, priority, analysis_mode,
                analysis_signature, config_signature, source_file_hash, created_at
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
                """,
                (
                    playlist_id,
                    track_id,
                    track_path,
                    job_kind,
                    priority,
                    analysis_mode,
                    analysis_signature,
                    config_signature,
                    source_file_hash,
                    created_at,
                ),
            )
        return int(cursor.lastrowid)

    def mark_analysis_job_started(self, job_id: int, started_at: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE analysis_jobs SET status = 'running', started_at = ? WHERE id = ?",
                (started_at, job_id),
            )

    def mark_analysis_job_completed(
        self,
        job_id: int,
        duration_seconds: float,
        timing_breakdown: dict[str, Any],
        completed_at: str,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """
                UPDATE analysis_jobs
                SET status = 'completed',
                    duration_seconds = ?,
                    timing_breakdown = ?,
                    completed_at = ?
                WHERE id = ?
                """,
                (duration_seconds, json.dumps(timing_breakdown, sort_keys=True), completed_at, job_id),
            )

    def mark_analysis_job_failed(
        self,
        job_id: int,
        error_message: str,
        duration_seconds: float,
        completed_at: str,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """
                UPDATE analysis_jobs
                SET status = 'failed',
                    error_message = ?,
                    duration_seconds = ?,
                    completed_at = ?
                WHERE id = ?
                """,
                (error_message, duration_seconds, completed_at, job_id),
            )

    def upsert_track_features(self, result: AnalysisResult) -> None:
        payload = result.to_db_row()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO track_features_abs (
                  track_id, user_id, source_file_hash, bpm, bpm_confidence, bpm_source,
                  time_signature, time_signature_confidence, key, key_number, key_letter,
                  key_confidence, key_source, key_imported, key_tagged, key_agreement,
                  energy_abs, energy_heuristic_abs, energy_sustained, energy_peak, danceability_abs, arousal_abs, valence_abs,
                  mood_aggressive_abs, mood_party_abs, mood_relaxed_abs,
                  energy_essentia_fused, energy_essentia_bucket, essentia_semantic_signature,
                  essentia_semantic_source, essentia_semantic_inferred_at, loudness_lufs, loudness_norm,
                  bass_abs, drums_abs, harmonic_abs, groove_abs, vocals_abs,
                  vocals_confidence, analysis_mode, analyzed_at, analysis_signature,
                  config_signature, scoring_contract_id_at_analysis
                ) VALUES (
                  :track_id, :user_id, :source_file_hash, :bpm, :bpm_confidence, :bpm_source,
                  :time_signature, :time_signature_confidence, :key, :key_number, :key_letter,
                  :key_confidence, :key_source, :key_imported, :key_tagged, :key_agreement,
                  :energy_abs, :energy_heuristic_abs, :energy_sustained, :energy_peak, :danceability_abs, :arousal_abs, :valence_abs,
                  :mood_aggressive_abs, :mood_party_abs, :mood_relaxed_abs,
                  :energy_essentia_fused, :energy_essentia_bucket, :essentia_semantic_signature,
                  :essentia_semantic_source, :essentia_semantic_inferred_at, :loudness_lufs, :loudness_norm,
                  :bass_abs, :drums_abs, :harmonic_abs, :groove_abs, :vocals_abs,
                  :vocals_confidence, :analysis_mode, :analyzed_at, :analysis_signature,
                  :config_signature, :scoring_contract_id_at_analysis
                )
                ON CONFLICT(track_id) DO UPDATE SET
                  source_file_hash = excluded.source_file_hash,
                  bpm = excluded.bpm,
                  bpm_confidence = excluded.bpm_confidence,
                  bpm_source = excluded.bpm_source,
                  time_signature = excluded.time_signature,
                  time_signature_confidence = excluded.time_signature_confidence,
                  key = excluded.key,
                  key_number = excluded.key_number,
                  key_letter = excluded.key_letter,
                  key_confidence = excluded.key_confidence,
                  key_source = excluded.key_source,
                  key_imported = excluded.key_imported,
                  key_tagged = excluded.key_tagged,
                  key_agreement = excluded.key_agreement,
                  energy_abs = excluded.energy_abs,
                  energy_heuristic_abs = excluded.energy_heuristic_abs,
                  energy_sustained = excluded.energy_sustained,
                  energy_peak = excluded.energy_peak,
                  danceability_abs = excluded.danceability_abs,
                  arousal_abs = excluded.arousal_abs,
                  valence_abs = excluded.valence_abs,
                  mood_aggressive_abs = excluded.mood_aggressive_abs,
                  mood_party_abs = excluded.mood_party_abs,
                  mood_relaxed_abs = excluded.mood_relaxed_abs,
                  energy_essentia_fused = excluded.energy_essentia_fused,
                  energy_essentia_bucket = excluded.energy_essentia_bucket,
                  essentia_semantic_signature = excluded.essentia_semantic_signature,
                  essentia_semantic_source = excluded.essentia_semantic_source,
                  essentia_semantic_inferred_at = excluded.essentia_semantic_inferred_at,
                  loudness_lufs = excluded.loudness_lufs,
                  loudness_norm = excluded.loudness_norm,
                  bass_abs = excluded.bass_abs,
                  drums_abs = excluded.drums_abs,
                  harmonic_abs = excluded.harmonic_abs,
                  groove_abs = excluded.groove_abs,
                  vocals_abs = excluded.vocals_abs,
                  vocals_confidence = excluded.vocals_confidence,
                  analysis_mode = excluded.analysis_mode,
                  analyzed_at = excluded.analyzed_at,
                  analysis_signature = excluded.analysis_signature,
                  config_signature = excluded.config_signature,
                  scoring_contract_id_at_analysis = excluded.scoring_contract_id_at_analysis
                """,
                payload,
            )

    def upsert_track_fast_features(self, result: FastAnalysisResult) -> None:
        payload = result.to_db_row()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO track_features_fast (
                  track_id, user_id, source_file_hash, bpm, bpm_confidence, bpm_source,
                  key, key_number, key_letter, key_confidence, key_source,
                  key_imported, key_tagged, key_agreement,
                  analyzed_at, analysis_signature, config_signature
                ) VALUES (
                  :track_id, :user_id, :source_file_hash, :bpm, :bpm_confidence, :bpm_source,
                  :key, :key_number, :key_letter, :key_confidence, :key_source,
                  :key_imported, :key_tagged, :key_agreement,
                  :analyzed_at, :analysis_signature, :config_signature
                )
                ON CONFLICT(track_id) DO UPDATE SET
                  source_file_hash = excluded.source_file_hash,
                  bpm = excluded.bpm,
                  bpm_confidence = excluded.bpm_confidence,
                  bpm_source = excluded.bpm_source,
                  key = excluded.key,
                  key_number = excluded.key_number,
                  key_letter = excluded.key_letter,
                  key_confidence = excluded.key_confidence,
                  key_source = excluded.key_source,
                  key_imported = excluded.key_imported,
                  key_tagged = excluded.key_tagged,
                  key_agreement = excluded.key_agreement,
                  analyzed_at = excluded.analyzed_at,
                  analysis_signature = excluded.analysis_signature,
                  config_signature = excluded.config_signature
                """,
                payload,
            )

    def get_track_details(self, track_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT
              t.id,
              t.file_path,
              t.file_hash,
              t.title,
              t.artist,
              t.genre,
              t.duration_seconds,
              t.imported_bpm,
              t.imported_key,
              t.import_source,
              t.imported_at,
              t.updated_at,
              ff.source_file_hash AS fast_source_file_hash,
              ff.bpm AS fast_bpm,
              ff.bpm_confidence AS fast_bpm_confidence,
              ff.bpm_source AS fast_bpm_source,
              ff.key AS fast_key,
              ff.key_number AS fast_key_number,
              ff.key_letter AS fast_key_letter,
              ff.key_confidence AS fast_key_confidence,
              ff.key_source AS fast_key_source,
              ff.key_imported AS fast_key_imported,
              ff.key_tagged AS fast_key_tagged,
              ff.key_agreement AS fast_key_agreement,
              ff.analysis_signature AS fast_analysis_signature,
              ff.config_signature AS fast_config_signature,
              ff.analyzed_at AS fast_analyzed_at,
              f.source_file_hash,
              f.bpm,
              f.bpm_confidence,
              f.bpm_source,
              f.time_signature,
              f.time_signature_confidence,
              f.key,
              f.key_number,
              f.key_letter,
              f.key_confidence,
              f.key_source,
              f.key_imported,
              f.key_tagged,
              f.key_agreement,
              f.energy_abs,
              f.energy_heuristic_abs,
              f.energy_sustained,
              f.energy_peak,
              f.danceability_abs,
              f.arousal_abs,
              f.valence_abs,
              f.mood_aggressive_abs,
              f.mood_party_abs,
              f.mood_relaxed_abs,
              f.energy_essentia_fused,
              f.energy_essentia_bucket,
              f.essentia_semantic_signature,
              f.essentia_semantic_source,
              f.essentia_semantic_inferred_at,
              f.loudness_lufs,
              f.loudness_norm,
              f.bass_abs,
              f.drums_abs,
              f.harmonic_abs,
              f.groove_abs,
              f.vocals_abs,
              f.vocals_confidence,
              f.analysis_mode,
              f.analysis_signature,
              f.config_signature,
              f.scoring_contract_id_at_analysis,
              f.analyzed_at
            FROM tracks t
            LEFT JOIN track_features_fast ff ON ff.track_id = t.id
            LEFT JOIN track_features_abs f ON f.track_id = t.id
            WHERE t.id = ?
            """,
            (track_id,),
        ).fetchone()
        if row is None:
            return None
        return {key: row[key] for key in row.keys()}

    def get_track_row(self, track_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT
              t.id,
              t.file_path,
              t.file_hash,
              t.title,
              t.artist,
              t.genre,
              t.duration_seconds,
              t.imported_bpm,
              t.imported_key,
              t.import_source
            FROM tracks t
            WHERE t.id = ?
            """,
            (track_id,),
        ).fetchone()

    def get_pending_analysis_jobs(
        self,
        *,
        job_kind: str | None = None,
        limit: int = 100,
    ) -> list[sqlite3.Row]:
        if job_kind is not None:
            return self.connection.execute(
                """
                SELECT *
                FROM analysis_jobs
                WHERE status = 'pending' AND job_kind = ?
                ORDER BY priority DESC, created_at ASC
                LIMIT ?
                """,
                (job_kind, limit),
            ).fetchall()
        return self.connection.execute(
            """
            SELECT *
            FROM analysis_jobs
            WHERE status = 'pending'
            ORDER BY priority DESC, created_at ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    # ------------------------------------------------------------------
    # Canonical relative persistence (Phase 2)
    # ------------------------------------------------------------------

    def replace_canonical_relative_rows(
        self,
        playlist_id: str,
        track_rows: list[dict[str, Any]],
        stats_row: dict[str, Any],
        timestamp: str,
    ) -> None:
        """Atomically replace all track_features_rel rows for a playlist and upsert playlist_stats."""
        with self.connection:
            self.connection.execute(
                "DELETE FROM track_features_rel WHERE playlist_id = ?",
                (playlist_id,),
            )
            for row in track_rows:
                self.connection.execute(
                    """
                    INSERT INTO track_features_rel (
                      playlist_id, track_id, position,
                      energy_rel, bass_rel, drums_rel, vocals_rel, groove_rel,
                      energy_spread, bass_spread, drums_spread, vocals_spread, groove_spread,
                      intensity_band, intensity_membership, role_hints,
                      valid_as_of_track_count,
                      relative_signature, analysis_signature, config_signature,
                      refreshed_at
                    ) VALUES (
                      :playlist_id, :track_id, :position,
                      :energy_rel, :bass_rel, :drums_rel, :vocals_rel, :groove_rel,
                      :energy_spread, :bass_spread, :drums_spread, :vocals_spread, :groove_spread,
                      :intensity_band, :intensity_membership, :role_hints,
                      :valid_as_of_track_count,
                      :relative_signature, :analysis_signature, :config_signature,
                      :refreshed_at
                    )
                    """,
                    row,
                )
            self.connection.execute(
                """
                INSERT INTO playlist_stats (
                  playlist_id,
                  track_count_total, track_count_analyzed, eligible_track_count,
                  energy_spread, bass_spread, drums_spread, vocals_spread,
                  harmonic_spread, groove_spread,
                  avg_harmonic, key_diversity, bpm_range,
                  adapted_weights, adaptation_strength, weight_adaptation_notes,
                  status, energy_source_used, relative_signature,
                  refreshed_at, is_stale, stale_reason, stale_marked_at
                ) VALUES (
                  :playlist_id,
                  :track_count_total, :track_count_analyzed, :eligible_track_count,
                  :energy_spread, :bass_spread, :drums_spread, :vocals_spread,
                  :harmonic_spread, :groove_spread,
                  :avg_harmonic, :key_diversity, :bpm_range,
                  :adapted_weights, :adaptation_strength, :weight_adaptation_notes,
                  :status, :energy_source_used, :relative_signature,
                  :refreshed_at, 0, NULL, NULL
                )
                ON CONFLICT(playlist_id) DO UPDATE SET
                  track_count_total    = excluded.track_count_total,
                  track_count_analyzed = excluded.track_count_analyzed,
                  eligible_track_count = excluded.eligible_track_count,
                  energy_spread    = excluded.energy_spread,
                  bass_spread      = excluded.bass_spread,
                  drums_spread     = excluded.drums_spread,
                  vocals_spread    = excluded.vocals_spread,
                  harmonic_spread  = excluded.harmonic_spread,
                  groove_spread    = excluded.groove_spread,
                  avg_harmonic     = excluded.avg_harmonic,
                  key_diversity    = excluded.key_diversity,
                  bpm_range        = excluded.bpm_range,
                  adapted_weights         = excluded.adapted_weights,
                  adaptation_strength     = excluded.adaptation_strength,
                  weight_adaptation_notes = excluded.weight_adaptation_notes,
                  status             = excluded.status,
                  energy_source_used = excluded.energy_source_used,
                  relative_signature = excluded.relative_signature,
                  refreshed_at       = excluded.refreshed_at,
                  is_stale           = 0,
                  stale_reason       = NULL,
                  stale_marked_at    = NULL
                """,
                stats_row,
            )

    def get_playlist_stats(self, playlist_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM playlist_stats WHERE playlist_id = ?",
            (playlist_id,),
        ).fetchone()

    def get_persisted_relative_rows(self, playlist_id: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT r.*, t.title, t.artist, t.file_path
            FROM track_features_rel r
            JOIN tracks t ON t.id = r.track_id
            WHERE r.playlist_id = ?
            ORDER BY r.position ASC
            """,
            (playlist_id,),
        ).fetchall()

    def get_playlists_containing_tracks(self, track_ids: Iterable[str]) -> list[str]:
        """Return playlist_ids of all playlists that contain any of the given track_ids."""
        ids = list(track_ids)
        if not ids:
            return []
        placeholders = ", ".join("?" * len(ids))
        rows = self.connection.execute(
            f"SELECT DISTINCT playlist_id FROM playlist_tracks WHERE track_id IN ({placeholders})",
            ids,
        ).fetchall()
        return [str(row[0]) for row in rows]

    def mark_playlists_stale(
        self,
        playlist_ids: Iterable[str],
        reason: str,
        timestamp: str,
    ) -> None:
        """Mark playlist_stats rows stale. Playlists without a stats row are ignored (treated as needs-refresh)."""
        with self.connection:
            for playlist_id in playlist_ids:
                self.connection.execute(
                    """
                    UPDATE playlist_stats
                    SET is_stale = 1, stale_reason = ?, stale_marked_at = ?
                    WHERE playlist_id = ?
                    """,
                    (reason, timestamp, playlist_id),
                )
    
    def claim_pending_analysis_jobs(
        self,
        *,
        job_kind: str,
        limit: int = 100,
        started_at: str,
    ) -> list[sqlite3.Row]:
        with self.connection:
            rows = self.connection.execute(
                """
                WITH to_claim AS (
                    SELECT id
                    FROM analysis_jobs
                    WHERE status = 'pending' AND job_kind = ?
                    ORDER BY priority DESC, created_at ASC
                    LIMIT ?
                )
                UPDATE analysis_jobs
                SET status = 'running',
                    started_at = ?
                WHERE id IN (SELECT id FROM to_claim)
                RETURNING *
                """,
                (job_kind, limit, started_at),
            ).fetchall()
        return rows
    
    # ------------------------------------------------------------------
    # Scoring queries (Phase 2 / Milestone 3)
    # ------------------------------------------------------------------

    def get_scoring_candidates(
        self,
        playlist_id: str,
        exclude_track_ids: list[str] | None = None,
    ) -> list[sqlite3.Row]:
        """Return all playlist tracks with abs+rel features for scoring.

        Tracks that have no absolute analysis row are excluded — they cannot
        be scored without BPM/key data. Tracks listed in `exclude_track_ids`
        are also excluded (used to remove the current track and history).
        """
        exclusions = list(exclude_track_ids or [])
        exclusion_clause = ""
        params: list[Any] = [playlist_id]
        if exclusions:
            placeholders = ", ".join("?" * len(exclusions))
            exclusion_clause = f"\n              AND t.id NOT IN ({placeholders})"
            params.extend(exclusions)
        query = f"""
            SELECT
              t.id          AS track_id,
              t.title,
              t.artist,
              t.file_path,
              pt.position,
              f.bpm,
              f.key,
              f.key_confidence,
              f.key_source,
              f.key_agreement,
              f.energy_abs,
              f.bass_abs,
              f.drums_abs,
              f.vocals_abs,
              f.vocals_confidence,
              f.groove_abs,
              f.harmonic_abs,
              r.energy_rel,
              r.bass_rel,
              r.drums_rel,
              r.vocals_rel,
              r.groove_rel,
              r.intensity_band,
              r.role_hints
            FROM playlist_tracks pt
            JOIN tracks t ON t.id = pt.track_id
            JOIN track_features_abs f ON f.track_id = t.id
            LEFT JOIN track_features_rel r
              ON r.track_id = t.id AND r.playlist_id = pt.playlist_id
            WHERE pt.playlist_id = ?
              {exclusion_clause}
            ORDER BY pt.position ASC
        """
        return self.connection.execute(query, params).fetchall()

    def get_track_scoring_context(
        self,
        track_id: str,
        playlist_id: str,
    ) -> sqlite3.Row | None:
        """Return a single track's abs+rel feature row for scoring."""
        return self.connection.execute(
            """
            SELECT
              t.id          AS track_id,
              t.title,
              t.artist,
              t.file_path,
              pt.position,
              f.bpm,
              f.key,
              f.key_confidence,
              f.key_source,
              f.key_agreement,
              f.energy_abs,
              f.bass_abs,
              f.drums_abs,
              f.vocals_abs,
              f.vocals_confidence,
              f.groove_abs,
              f.harmonic_abs,
              r.energy_rel,
              r.bass_rel,
              r.drums_rel,
              r.vocals_rel,
              r.groove_rel,
              r.intensity_band,
              r.role_hints
            FROM playlist_tracks pt
            JOIN tracks t ON t.id = pt.track_id
            JOIN track_features_abs f ON f.track_id = t.id
            LEFT JOIN track_features_rel r
              ON r.track_id = t.id AND r.playlist_id = pt.playlist_id
            WHERE pt.playlist_id = ?
              AND t.id = ?
            """,
            (playlist_id, track_id),
        ).fetchone()

    def get_playlist_stats_for_scoring(self, playlist_id: str) -> dict[str, Any] | None:
        """Return the playlist_stats row as a plain dict, JSON-decoding adapted_weights.

        Returns None when no stats row exists (playlist not yet enriched).
        """
        row = self.connection.execute(
            "SELECT * FROM playlist_stats WHERE playlist_id = ?",
            (playlist_id,),
        ).fetchone()
        if row is None:
            return None
        result: dict[str, Any] = {key: row[key] for key in row.keys()}
        raw_weights = result.get("adapted_weights")
        if isinstance(raw_weights, str):
            try:
                result["adapted_weights"] = json.loads(raw_weights)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.exception(
                    "Invalid adapted_weights JSON in playlist_stats",
                    extra={
                        "playlist_id": result.get("playlist_id"),
                        "relative_signature": result.get("relative_signature"),
                        "raw_weights": raw_weights,
                    },
                )
                raise ValueError(
                    f"Corrupted adapted_weights for playlist '{result.get('playlist_id')}': {raw_weights!r}"
                ) from exc
        return result

    def get_analysis_jobs_by_ids(self, job_ids: Iterable[int]) -> list[sqlite3.Row]:
        ids = [int(job_id) for job_id in job_ids]
        if not ids:
            return []
        placeholders = ", ".join("?" * len(ids))
        return self.connection.execute(
            f"""
            SELECT *
            FROM analysis_jobs
            WHERE id IN ({placeholders})
            """,
            ids,
        ).fetchall()
