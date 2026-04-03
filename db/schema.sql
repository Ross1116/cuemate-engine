CREATE TABLE IF NOT EXISTS "schema_migrations" (version varchar(128) primary key);
-- Dbmate schema migrations
INSERT INTO "schema_migrations" (version) VALUES
  ('20260403112734');

CREATE TABLE IF NOT EXISTS schema_metadata (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  "key" TEXT NOT NULL UNIQUE,
  value TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
