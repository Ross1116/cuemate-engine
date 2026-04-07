-- migrate:up
-- Legacy scaffold from the deprecated learned-energy lane.
-- These columns are intentionally retained for local DB compatibility but are not used by the current app.
ALTER TABLE track_features_abs ADD COLUMN energy_hybrid REAL;
ALTER TABLE track_features_abs ADD COLUMN energy_learned REAL;
ALTER TABLE track_features_abs ADD COLUMN energy_learned_bucket TEXT;
ALTER TABLE track_features_abs ADD COLUMN energy_model_signature TEXT;
ALTER TABLE track_features_abs ADD COLUMN energy_model_source TEXT;
ALTER TABLE track_features_abs ADD COLUMN energy_model_inferred_at TEXT;

-- migrate:down
ALTER TABLE track_features_abs DROP COLUMN IF EXISTS energy_hybrid;
ALTER TABLE track_features_abs DROP COLUMN IF EXISTS energy_learned;
ALTER TABLE track_features_abs DROP COLUMN IF EXISTS energy_learned_bucket;
ALTER TABLE track_features_abs DROP COLUMN IF EXISTS energy_model_signature;
ALTER TABLE track_features_abs DROP COLUMN IF EXISTS energy_model_source;
ALTER TABLE track_features_abs DROP COLUMN IF EXISTS energy_model_inferred_at;
