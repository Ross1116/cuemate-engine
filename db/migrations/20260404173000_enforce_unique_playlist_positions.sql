-- migrate:up
CREATE TABLE playlist_tracks__new (
  playlist_id TEXT NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
  track_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
  position INTEGER NOT NULL,
  added_at TEXT NOT NULL,
  PRIMARY KEY (playlist_id, track_id),
  UNIQUE (playlist_id, position)
);

INSERT INTO playlist_tracks__new (playlist_id, track_id, position, added_at)
SELECT playlist_id, track_id, position, added_at
FROM playlist_tracks;

DROP TABLE playlist_tracks;
ALTER TABLE playlist_tracks__new RENAME TO playlist_tracks;

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
