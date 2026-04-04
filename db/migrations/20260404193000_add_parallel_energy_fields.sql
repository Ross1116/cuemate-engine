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
-- Irreversible compatibility migration; legacy columns are intentionally left in place.
SELECT 1;
