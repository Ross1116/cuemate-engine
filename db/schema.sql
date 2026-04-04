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
  updated_at TEXT NOT NULL,
  imported_bpm REAL,
  imported_key TEXT
);
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
);
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
);
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
-- Dbmate schema migrations
INSERT INTO "schema_migrations" (version) VALUES
  ('20260403112734'),
  ('20260403154500'),
  ('20260404143000'),
  ('20260404173000');
