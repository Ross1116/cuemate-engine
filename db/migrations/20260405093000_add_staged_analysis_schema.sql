-- migrate:up
CREATE TABLE IF NOT EXISTS track_features_fast (
  track_id TEXT PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL DEFAULT 'local',
  source_file_hash TEXT NOT NULL,
  bpm REAL NOT NULL,
  bpm_confidence REAL NOT NULL,
  bpm_source TEXT NOT NULL,
  key TEXT NOT NULL,
  key_number INTEGER NOT NULL,
  key_letter TEXT NOT NULL,
  key_confidence REAL NOT NULL,
  key_source TEXT NOT NULL,
  key_imported TEXT,
  key_tagged TEXT,
  key_agreement INTEGER,
  analyzed_at TEXT NOT NULL,
  analysis_signature TEXT NOT NULL,
  config_signature TEXT NOT NULL
);

ALTER TABLE analysis_jobs ADD COLUMN job_kind TEXT NOT NULL DEFAULT 'full';

-- migrate:down
DROP TABLE IF EXISTS track_features_fast;
-- SQLite does not support dropping analysis_jobs.job_kind in-place.
-- This migration is intentionally irreversible for the analysis_jobs alteration.
