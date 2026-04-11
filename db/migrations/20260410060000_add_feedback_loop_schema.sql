-- migrate:up
ALTER TABLE playlist_stats ADD COLUMN feedback_tuned_weights TEXT;
ALTER TABLE playlist_stats ADD COLUMN feedback_tuning_notes TEXT;
ALTER TABLE playlist_stats ADD COLUMN feedback_event_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE playlist_stats ADD COLUMN feedback_last_tuned_at TEXT;
ALTER TABLE playlist_stats ADD COLUMN feedback_tuning_metrics TEXT;

CREATE TABLE IF NOT EXISTS recommendation_event_items (
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

CREATE TABLE IF NOT EXISTS feedback_tuning_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  playlist_id TEXT NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'pending',
  trigger_event_id TEXT REFERENCES recommendation_events(id) ON DELETE SET NULL,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_recommendation_event_items_event_id
  ON recommendation_event_items (event_id, lane_id, lane_rank);
CREATE INDEX IF NOT EXISTS idx_recommendation_event_items_candidate_track
  ON recommendation_event_items (candidate_track_id);
CREATE INDEX IF NOT EXISTS idx_feedback_tuning_jobs_status_created
  ON feedback_tuning_jobs (status, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_tuning_jobs_playlist_pending
  ON feedback_tuning_jobs (playlist_id)
  WHERE status = 'pending';

-- migrate:down
DROP INDEX IF EXISTS idx_feedback_tuning_jobs_playlist_pending;
DROP INDEX IF EXISTS idx_feedback_tuning_jobs_status_created;
DROP INDEX IF EXISTS idx_recommendation_event_items_candidate_track;
DROP INDEX IF EXISTS idx_recommendation_event_items_event_id;
DROP TABLE IF EXISTS feedback_tuning_jobs;
DROP TABLE IF EXISTS recommendation_event_items;
-- Irreversible migration: cannot safely remove feedback columns from playlist_stats without rebuilding playlist_stats.
SELECT cueMate_irreversible_feedback_loop_migration_blocked();
