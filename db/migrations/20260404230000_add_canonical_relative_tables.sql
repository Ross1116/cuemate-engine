-- migrate:up
-- Canonical relative feature snapshot: one current row per (playlist_id, track_id).
-- Stores only the canonical energy lane (energy_abs as source).
-- Replaced atomically on each refresh; no history rows.
CREATE TABLE IF NOT EXISTS track_features_rel (
  playlist_id TEXT NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
  track_id    TEXT NOT NULL REFERENCES tracks(id)    ON DELETE CASCADE,
  position    INTEGER NOT NULL,
  -- canonical relative scores
  energy_rel  REAL NOT NULL,
  bass_rel    REAL NOT NULL,
  drums_rel   REAL NOT NULL,
  vocals_rel  REAL,
  groove_rel  REAL NOT NULL,
  -- per-track spread context (playlist-level, duplicated for fast per-track reads)
  energy_spread REAL NOT NULL,
  bass_spread   REAL NOT NULL,
  drums_spread  REAL NOT NULL,
  vocals_spread REAL NOT NULL,
  groove_spread REAL NOT NULL,
  -- classification outputs
  intensity_band       TEXT NOT NULL,
  intensity_membership TEXT NOT NULL,  -- JSON: {"low":…,"groove":…,"drive":…,"peak":…}
  role_hints           TEXT NOT NULL,  -- JSON array
  valid_as_of_track_count INTEGER NOT NULL,
  -- provenance
  relative_signature  TEXT NOT NULL,
  analysis_signature  TEXT NOT NULL,
  config_signature    TEXT NOT NULL,
  refreshed_at        TEXT NOT NULL,
  PRIMARY KEY (playlist_id, track_id)
);
CREATE INDEX IF NOT EXISTS idx_track_features_rel_playlist
  ON track_features_rel(playlist_id, position ASC);

-- Canonical playlist summary: one current row per playlist_id.
-- Also carries the stale-state flags used by the refresh orchestrator.
CREATE TABLE IF NOT EXISTS playlist_stats (
  playlist_id TEXT PRIMARY KEY REFERENCES playlists(id) ON DELETE CASCADE,
  -- track counts
  track_count_total    INTEGER NOT NULL,
  track_count_analyzed INTEGER NOT NULL,
  eligible_track_count INTEGER NOT NULL,
  -- spread stats (nullable when eligible_track_count < min_for_relative)
  energy_spread    REAL,
  bass_spread      REAL,
  drums_spread     REAL,
  vocals_spread    REAL,
  harmonic_spread  REAL,
  groove_spread    REAL,
  avg_harmonic     REAL,
  key_diversity    REAL,
  bpm_range        REAL,
  -- weight adaptation outputs
  adapted_weights         TEXT,   -- JSON object or NULL
  adaptation_strength     REAL,
  weight_adaptation_notes TEXT,   -- JSON array
  -- status / provenance
  status              TEXT NOT NULL,  -- "ok","relative_only","insufficient_tracks"
  energy_source_used  TEXT NOT NULL DEFAULT 'canonical',
  relative_signature  TEXT NOT NULL,
  refreshed_at        TEXT NOT NULL,
  -- stale-state fields
  is_stale         INTEGER NOT NULL DEFAULT 0,  -- 0=current, 1=stale
  stale_reason     TEXT,    -- "absolute_track_changed","playlist_membership_changed","relative_signature_changed"
  stale_marked_at  TEXT
);

-- migrate:down
DROP TABLE IF EXISTS track_features_rel;
DROP TABLE IF EXISTS playlist_stats;
