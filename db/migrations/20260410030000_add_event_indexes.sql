-- migrate:up
CREATE INDEX IF NOT EXISTS idx_manual_corrections_track_id ON manual_corrections (track_id);
CREATE INDEX IF NOT EXISTS idx_recommendation_events_playlist_id ON recommendation_events (playlist_id);
CREATE INDEX IF NOT EXISTS idx_recommendation_events_timestamp ON recommendation_events (timestamp);
CREATE INDEX IF NOT EXISTS idx_recommendation_events_status ON recommendation_events (recommendations_status);

-- migrate:down
DROP INDEX IF EXISTS idx_recommendation_events_status;
DROP INDEX IF EXISTS idx_recommendation_events_timestamp;
DROP INDEX IF EXISTS idx_recommendation_events_playlist_id;
DROP INDEX IF EXISTS idx_manual_corrections_track_id;
