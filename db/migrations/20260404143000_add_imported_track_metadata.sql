-- migrate:up
ALTER TABLE tracks ADD COLUMN imported_bpm REAL;
ALTER TABLE tracks ADD COLUMN imported_key TEXT;

-- migrate:down
CREATE TABLE tracks__rollback (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL DEFAULT 'local',
  file_path TEXT NOT NULL UNIQUE,
  file_hash TEXT,
  title TEXT,
  artist TEXT,
  genre TEXT,
  duration_seconds REAL,
  import_source TEXT,
  imported_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

INSERT INTO tracks__rollback (
  id, user_id, file_path, file_hash, title, artist, genre,
  duration_seconds, import_source, imported_at, updated_at
)
SELECT
  id, user_id, file_path, file_hash, title, artist, genre,
  duration_seconds, import_source, imported_at, updated_at
FROM tracks;

DROP TABLE tracks;
ALTER TABLE tracks__rollback RENAME TO tracks;
CREATE INDEX idx_tracks_file_hash ON tracks(file_hash);
