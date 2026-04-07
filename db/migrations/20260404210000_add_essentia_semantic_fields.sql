-- migrate:up
ALTER TABLE track_features_abs ADD COLUMN danceability_abs REAL;
ALTER TABLE track_features_abs ADD COLUMN arousal_abs REAL;
ALTER TABLE track_features_abs ADD COLUMN valence_abs REAL;
ALTER TABLE track_features_abs ADD COLUMN mood_aggressive_abs REAL;
ALTER TABLE track_features_abs ADD COLUMN mood_party_abs REAL;
ALTER TABLE track_features_abs ADD COLUMN mood_relaxed_abs REAL;
ALTER TABLE track_features_abs ADD COLUMN energy_essentia_fused REAL;
ALTER TABLE track_features_abs ADD COLUMN energy_essentia_bucket TEXT;
ALTER TABLE track_features_abs ADD COLUMN essentia_semantic_signature TEXT;
ALTER TABLE track_features_abs ADD COLUMN essentia_semantic_source TEXT;
ALTER TABLE track_features_abs ADD COLUMN essentia_semantic_inferred_at TEXT;

-- migrate:down
ALTER TABLE track_features_abs DROP COLUMN IF EXISTS danceability_abs;
ALTER TABLE track_features_abs DROP COLUMN IF EXISTS arousal_abs;
ALTER TABLE track_features_abs DROP COLUMN IF EXISTS valence_abs;
ALTER TABLE track_features_abs DROP COLUMN IF EXISTS mood_aggressive_abs;
ALTER TABLE track_features_abs DROP COLUMN IF EXISTS mood_party_abs;
ALTER TABLE track_features_abs DROP COLUMN IF EXISTS mood_relaxed_abs;
ALTER TABLE track_features_abs DROP COLUMN IF EXISTS energy_essentia_fused;
ALTER TABLE track_features_abs DROP COLUMN IF EXISTS energy_essentia_bucket;
ALTER TABLE track_features_abs DROP COLUMN IF EXISTS essentia_semantic_signature;
ALTER TABLE track_features_abs DROP COLUMN IF EXISTS essentia_semantic_source;
ALTER TABLE track_features_abs DROP COLUMN IF EXISTS essentia_semantic_inferred_at;
