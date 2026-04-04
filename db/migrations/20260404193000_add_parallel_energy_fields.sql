-- migrate:up
ALTER TABLE track_features_abs ADD COLUMN energy_hybrid REAL;
ALTER TABLE track_features_abs ADD COLUMN energy_learned REAL;
ALTER TABLE track_features_abs ADD COLUMN energy_learned_bucket TEXT;
ALTER TABLE track_features_abs ADD COLUMN energy_model_signature TEXT;
ALTER TABLE track_features_abs ADD COLUMN energy_model_source TEXT;
ALTER TABLE track_features_abs ADD COLUMN energy_model_inferred_at TEXT;

-- migrate:down
SELECT 1;
