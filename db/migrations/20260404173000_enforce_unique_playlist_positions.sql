-- migrate:up
CREATE TABLE playlist_tracks__new (
  playlist_id TEXT NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
  track_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
  position INTEGER NOT NULL,
  added_at TEXT NOT NULL,
  PRIMARY KEY (playlist_id, track_id),
  UNIQUE (playlist_id, position)
);

CREATE TEMP TABLE playlist_tracks__deduped AS
WITH ranked_tracks AS (
  SELECT
    playlist_id,
    track_id,
    position,
    added_at,
    ROW_NUMBER() OVER (
      PARTITION BY playlist_id, position
      ORDER BY added_at ASC, track_id ASC
    ) AS row_rank
  FROM playlist_tracks
)
SELECT
  playlist_id,
  track_id,
  position,
  added_at
FROM ranked_tracks
WHERE row_rank = 1;

INSERT INTO playlist_tracks__new (playlist_id, track_id, position, added_at)
SELECT playlist_id, track_id, position, added_at
FROM playlist_tracks__deduped;

DROP TABLE playlist_tracks;
ALTER TABLE playlist_tracks__new RENAME TO playlist_tracks;
DROP TABLE playlist_tracks__deduped;

-- migrate:down
CREATE TABLE playlist_tracks__rollback (
  playlist_id TEXT NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
  track_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
  position INTEGER NOT NULL,
  added_at TEXT NOT NULL,
  PRIMARY KEY (playlist_id, track_id)
);

INSERT INTO playlist_tracks__rollback (playlist_id, track_id, position, added_at)
SELECT playlist_id, track_id, position, added_at
FROM playlist_tracks;

DROP TABLE playlist_tracks;
ALTER TABLE playlist_tracks__rollback RENAME TO playlist_tracks;

CREATE INDEX idx_playlist_tracks_playlist_position
  ON playlist_tracks(playlist_id, position);
