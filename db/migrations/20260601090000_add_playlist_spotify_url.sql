-- migrate:up
ALTER TABLE playlists ADD COLUMN spotify_url TEXT;

-- migrate:down
ALTER TABLE playlists DROP COLUMN IF EXISTS spotify_url;
