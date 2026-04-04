from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

from cuemate_analysis.models import AnalysisResult, ImportedTrack


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

    def get_playlist(self, name: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM playlists WHERE name = ?",
            (name,),
        ).fetchone()

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
              f.analysis_mode,
              f.analysis_signature,
              f.config_signature,
              f.source_file_hash,
              f.bpm,
              f.key,
              f.energy_abs,
              f.energy_hybrid,
              f.energy_learned,
              f.energy_learned_bucket,
              f.energy_model_signature,
              f.energy_model_source,
              f.energy_model_inferred_at,
              f.analyzed_at
            FROM playlists p
            JOIN playlist_tracks pt ON pt.playlist_id = p.id
            JOIN tracks t ON t.id = pt.track_id
            LEFT JOIN track_features_abs f ON f.track_id = t.id
            WHERE p.name = ?
            ORDER BY pt.position ASC
            """,
            (playlist_name,),
        ).fetchall()

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
              f.energy_abs,
              f.energy_hybrid,
              f.energy_learned,
              f.energy_learned_bucket,
              f.bass_abs,
              f.drums_abs,
              f.harmonic_abs,
              f.groove_abs,
              f.vocals_abs,
              f.vocals_confidence,
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
                  playlist_id, track_id, track_path, status, priority, analysis_mode,
                  analysis_signature, config_signature, source_file_hash, created_at
                ) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
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
                  energy_abs, energy_sustained, energy_peak, energy_hybrid, energy_learned,
                  energy_learned_bucket, energy_model_signature, energy_model_source,
                  energy_model_inferred_at, loudness_lufs, loudness_norm,
                  bass_abs, drums_abs, harmonic_abs, groove_abs, vocals_abs,
                  vocals_confidence, analysis_mode, analyzed_at, analysis_signature,
                  config_signature, scoring_contract_id_at_analysis
                ) VALUES (
                  :track_id, :user_id, :source_file_hash, :bpm, :bpm_confidence, :bpm_source,
                  :time_signature, :time_signature_confidence, :key, :key_number, :key_letter,
                  :key_confidence, :key_source, :key_imported, :key_tagged, :key_agreement,
                  :energy_abs, :energy_sustained, :energy_peak, :energy_hybrid, :energy_learned,
                  :energy_learned_bucket, :energy_model_signature, :energy_model_source,
                  :energy_model_inferred_at, :loudness_lufs, :loudness_norm,
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
                  energy_sustained = excluded.energy_sustained,
                  energy_peak = excluded.energy_peak,
                  energy_hybrid = excluded.energy_hybrid,
                  energy_learned = excluded.energy_learned,
                  energy_learned_bucket = excluded.energy_learned_bucket,
                  energy_model_signature = excluded.energy_model_signature,
                  energy_model_source = excluded.energy_model_source,
                  energy_model_inferred_at = excluded.energy_model_inferred_at,
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
              f.energy_sustained,
              f.energy_peak,
              f.energy_hybrid,
              f.energy_learned,
              f.energy_learned_bucket,
              f.energy_model_signature,
              f.energy_model_source,
              f.energy_model_inferred_at,
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
            LEFT JOIN track_features_abs f ON f.track_id = t.id
            WHERE t.id = ?
            """,
            (track_id,),
        ).fetchone()
        if row is None:
            return None
        return {key: row[key] for key in row.keys()}
