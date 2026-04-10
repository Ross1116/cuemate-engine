-- migrate:up
CREATE TABLE IF NOT EXISTS manual_corrections (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL DEFAULT 'local',
  track_id TEXT NOT NULL REFERENCES tracks(id),
  field TEXT NOT NULL,
  old_value TEXT NOT NULL,
  new_value TEXT NOT NULL,
  corrected_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendation_events (
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

CREATE TABLE IF NOT EXISTS sync_outbox (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  entity_type TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  action TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  synced_at TEXT
);

CREATE TABLE IF NOT EXISTS playlist_sync_state (
  playlist_id TEXT PRIMARY KEY REFERENCES playlists(id) ON DELETE CASCADE,
  last_snapshot_id TEXT NOT NULL UNIQUE,
  last_snapshot_generated_at TEXT NOT NULL,
  last_snapshot_acked_at TEXT,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sync_outbox_unsynced ON sync_outbox (synced_at, id);

-- migrate:down
DROP INDEX IF EXISTS idx_sync_outbox_unsynced;
DROP TABLE IF EXISTS playlist_sync_state;
DROP TABLE IF EXISTS sync_outbox;
DROP TABLE IF EXISTS recommendation_events;
DROP TABLE IF EXISTS manual_corrections;
