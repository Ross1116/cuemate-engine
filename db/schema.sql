CREATE TABLE IF NOT EXISTS "schema_migrations" (version varchar(128) primary key);
CREATE TABLE schema_metadata (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  "key" TEXT NOT NULL UNIQUE,
  value TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE tracks (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL DEFAULT 'local',
  file_path TEXT NOT NULL UNIQUE,
  file_hash TEXT,
  title TEXT,
  artist TEXT,
  genre TEXT,
  duration_seconds REAL,
  import_source TEXT,
  imported_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
, imported_bpm REAL, imported_key TEXT);
CREATE INDEX idx_tracks_file_hash ON tracks(file_hash);
CREATE TABLE playlists (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL DEFAULT 'local',
  name TEXT NOT NULL UNIQUE,
  track_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE track_features_abs (
  track_id TEXT PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL DEFAULT 'local',
  source_file_hash TEXT NOT NULL,
  bpm REAL NOT NULL,
  bpm_confidence REAL NOT NULL,
  bpm_source TEXT NOT NULL,
  time_signature TEXT DEFAULT '4/4',
  time_signature_confidence REAL DEFAULT 0.6,
  key TEXT NOT NULL,
  key_number INTEGER NOT NULL,
  key_letter TEXT NOT NULL,
  key_confidence REAL NOT NULL,
  key_source TEXT NOT NULL,
  key_imported TEXT,
  key_tagged TEXT,
  key_agreement INTEGER,
  energy_abs REAL NOT NULL,
  energy_sustained REAL,
  energy_peak REAL,
  loudness_lufs REAL NOT NULL,
  loudness_norm REAL NOT NULL,
  bass_abs REAL NOT NULL,
  drums_abs REAL,
  harmonic_abs REAL,
  groove_abs REAL,
  vocals_abs REAL,
  vocals_confidence REAL,
  analysis_mode TEXT NOT NULL DEFAULT 'full',
  analyzed_at TEXT NOT NULL,
  analysis_signature TEXT NOT NULL,
  config_signature TEXT NOT NULL,
  scoring_contract_id_at_analysis TEXT
, energy_hybrid REAL, energy_learned REAL, energy_learned_bucket TEXT, energy_model_signature TEXT, energy_model_source TEXT, energy_model_inferred_at TEXT, danceability_abs REAL, arousal_abs REAL, valence_abs REAL, mood_aggressive_abs REAL, mood_party_abs REAL, mood_relaxed_abs REAL, energy_essentia_fused REAL, energy_essentia_bucket TEXT, essentia_semantic_signature TEXT, essentia_semantic_source TEXT, essentia_semantic_inferred_at TEXT, energy_heuristic_abs REAL);
CREATE TABLE analysis_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  playlist_id TEXT REFERENCES playlists(id) ON DELETE SET NULL,
  track_id TEXT REFERENCES tracks(id) ON DELETE SET NULL,
  track_path TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  priority INTEGER NOT NULL DEFAULT 0,
  analysis_mode TEXT NOT NULL DEFAULT 'full',
  analysis_signature TEXT NOT NULL,
  config_signature TEXT NOT NULL,
  source_file_hash TEXT,
  error_message TEXT,
  duration_seconds REAL,
  timing_breakdown TEXT,
  created_at TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT
, job_kind TEXT NOT NULL DEFAULT 'full');
CREATE INDEX idx_analysis_jobs_status_priority
  ON analysis_jobs(status, priority DESC, created_at ASC);
CREATE TABLE IF NOT EXISTS "playlist_tracks" (
  playlist_id TEXT NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
  track_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
  position INTEGER NOT NULL,
  added_at TEXT NOT NULL,
  PRIMARY KEY (playlist_id, track_id),
  UNIQUE (playlist_id, position)
);
CREATE TABLE track_features_rel (
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
CREATE INDEX idx_track_features_rel_playlist
  ON track_features_rel(playlist_id, position ASC);
CREATE TABLE playlist_stats (
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
  feedback_tuned_weights  TEXT,
  feedback_tuning_notes   TEXT,
  feedback_event_count    INTEGER NOT NULL DEFAULT 0,
  feedback_last_tuned_at  TEXT,
  feedback_tuning_metrics TEXT,
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
CREATE TABLE track_features_fast (
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
CREATE TABLE manual_corrections (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL DEFAULT 'local',
  track_id TEXT NOT NULL REFERENCES tracks(id),
  field TEXT NOT NULL,
  old_value TEXT NOT NULL,
  new_value TEXT NOT NULL,
  corrected_at TEXT NOT NULL
);
CREATE TABLE recommendation_events (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL DEFAULT 'local',
  playlist_id TEXT NOT NULL REFERENCES playlists(id),
  current_track_id TEXT NOT NULL REFERENCES tracks(id),
  target TEXT NOT NULL,
  candidate_count INTEGER NOT NULL,
  recommendation_confidence REAL,
  recommendations_status TEXT NOT NULL DEFAULT 'available',
  lanes_returned TEXT NOT NULL,
  track_chosen TEXT REFERENCES tracks(id),
  chosen_was_recommended INTEGER,
  skipped_over TEXT,
  adapted_weights TEXT,
  scoring_contract_id TEXT NOT NULL,
  timestamp TEXT NOT NULL
);
CREATE TABLE recommendation_event_items (
  event_id TEXT NOT NULL REFERENCES recommendation_events(id) ON DELETE CASCADE,
  lane_id TEXT NOT NULL,
  lane_rank INTEGER NOT NULL,
  candidate_track_id TEXT NOT NULL REFERENCES tracks(id),
  final_score REAL NOT NULL,
  raw_score REAL NOT NULL,
  penalty_multiplier REAL NOT NULL,
  move TEXT NOT NULL,
  move_confidence REAL NOT NULL,
  risk TEXT NOT NULL,
  risk_score REAL NOT NULL,
  primary_lane TEXT,
  secondary_lane INTEGER NOT NULL DEFAULT 0,
  component_scores_json TEXT NOT NULL,
  confidences_json TEXT NOT NULL,
  weights_used_json TEXT NOT NULL,
  transition_features_json TEXT NOT NULL,
  PRIMARY KEY (event_id, lane_id, lane_rank, candidate_track_id)
);
CREATE TABLE feedback_tuning_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  playlist_id TEXT NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'pending',
  trigger_event_id TEXT REFERENCES recommendation_events(id) ON DELETE SET NULL,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  error_message TEXT
);
CREATE TABLE sync_outbox (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  entity_type TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  action TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  synced_at TEXT
);
CREATE TABLE playlist_sync_state (
  playlist_id TEXT PRIMARY KEY REFERENCES playlists(id) ON DELETE CASCADE,
  last_snapshot_id TEXT NOT NULL UNIQUE,
  last_snapshot_generated_at TEXT NOT NULL,
  last_snapshot_acked_at TEXT,
  updated_at TEXT NOT NULL
);
CREATE INDEX idx_manual_corrections_track_id ON manual_corrections (track_id);
CREATE INDEX idx_recommendation_events_playlist_id ON recommendation_events (playlist_id);
CREATE INDEX idx_recommendation_events_timestamp ON recommendation_events (timestamp);
CREATE INDEX idx_recommendation_events_status ON recommendation_events (recommendations_status);
CREATE INDEX idx_recommendation_event_items_event_id ON recommendation_event_items (event_id, lane_id, lane_rank);
CREATE INDEX idx_recommendation_event_items_candidate_track ON recommendation_event_items (candidate_track_id);
CREATE INDEX idx_feedback_tuning_jobs_status_created ON feedback_tuning_jobs (status, created_at);
CREATE UNIQUE INDEX idx_feedback_tuning_jobs_playlist_pending
  ON feedback_tuning_jobs (playlist_id)
  WHERE status IN ('pending', 'running');
CREATE INDEX idx_sync_outbox_unsynced ON sync_outbox (synced_at, id);
-- Dbmate schema migrations
INSERT INTO "schema_migrations" (version) VALUES
  ('20260403112734'),
  ('20260403154500'),
  ('20260404143000'),
  ('20260404173000'),
  ('20260404193000'),
  ('20260404210000'),
  ('20260404220000'),
  ('20260404230000'),
  ('20260405093000'),
  ('20260409183000'),
  ('20260410030000'),
  ('20260410060000');
