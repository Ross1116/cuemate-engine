-- migrate:up

CREATE TABLE remote_pairing_tokens (
  token_hash TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  used_at TEXT,
  device_label TEXT
);

CREATE TABLE remote_sessions (
  session_hash TEXT PRIMARY KEY,
  device_label TEXT,
  created_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  revoked_at TEXT
);

CREATE INDEX idx_remote_pairing_tokens_expires ON remote_pairing_tokens (expires_at);
CREATE INDEX idx_remote_sessions_expires ON remote_sessions (expires_at);

-- migrate:down

DROP INDEX IF EXISTS idx_remote_sessions_expires;
DROP INDEX IF EXISTS idx_remote_pairing_tokens_expires;
DROP TABLE IF EXISTS remote_sessions;
DROP TABLE IF EXISTS remote_pairing_tokens;
