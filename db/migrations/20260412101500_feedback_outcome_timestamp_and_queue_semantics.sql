-- migrate:up
ALTER TABLE recommendation_events ADD COLUMN played_at TEXT;
CREATE INDEX IF NOT EXISTS idx_recommendation_events_played_at
  ON recommendation_events (played_at);

DROP INDEX IF EXISTS idx_feedback_tuning_jobs_playlist_pending;
CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_tuning_jobs_playlist_pending
  ON feedback_tuning_jobs (playlist_id)
  WHERE status = 'pending';

-- migrate:down
DROP INDEX IF EXISTS idx_recommendation_events_played_at;
DROP INDEX IF EXISTS idx_feedback_tuning_jobs_playlist_pending;
CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_tuning_jobs_playlist_pending
  ON feedback_tuning_jobs (playlist_id)
  WHERE status IN ('pending', 'running');

-- Irreversible migration: cannot safely remove recommendation_events.played_at without rebuilding the table.
SELECT cueMate_irreversible_feedback_outcome_migration_blocked();
