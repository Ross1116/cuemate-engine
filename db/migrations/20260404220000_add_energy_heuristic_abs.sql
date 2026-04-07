-- migrate:up
-- Add energy_heuristic_abs column to preserve the old DSP-only heuristic energy score
-- as a legacy/support field now that energy_abs is redefined as the canonical fused intensity score.
ALTER TABLE track_features_abs ADD COLUMN energy_heuristic_abs REAL;

-- migrate:down
-- Irreversible on SQLite without rebuilding track_features_abs.
SELECT 1;
