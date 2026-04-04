from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterable

from cuemate_analysis.config import find_repo_root


DEFAULT_INFERENCE_CACHE_PATH = "data/inference-cache.db"


CREATE_CACHE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS model_inference_cache (
  cache_key TEXT PRIMARY KEY,
  backend TEXT NOT NULL,
  file_path TEXT NOT NULL,
  file_mtime_ns INTEGER NOT NULL,
  file_size INTEGER NOT NULL,
  model_signature TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_model_inference_cache_backend_path
  ON model_inference_cache(backend, file_path);
"""


@dataclass(frozen=True)
class ModelInferenceCacheEntry:
    backend: str
    cache_key: str
    file_path: str
    file_mtime_ns: int
    file_size: int
    model_signature: str
    payload: dict[str, Any]


def resolve_inference_cache_path(repo_root: Path | None = None) -> Path:
    root = find_repo_root(repo_root)
    raw_value = os.getenv("CUEMATE_INFERENCE_CACHE_PATH", DEFAULT_INFERENCE_CACHE_PATH)
    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate.resolve()


class PersistentInferenceCache:
    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.executescript(CREATE_CACHE_TABLE_SQL)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "PersistentInferenceCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def fetch_payloads(self, backend: str, cache_keys: Iterable[str]) -> dict[str, dict[str, Any]]:
        resolved_keys = [key for key in cache_keys if key]
        if not resolved_keys:
            return {}
        placeholders = ", ".join("?" for _ in resolved_keys)
        rows = self.connection.execute(
            f"""
            SELECT cache_key, payload_json
            FROM model_inference_cache
            WHERE backend = ? AND cache_key IN ({placeholders})
            """,
            [backend, *resolved_keys],
        ).fetchall()
        return {
            str(row["cache_key"]): json.loads(str(row["payload_json"]))
            for row in rows
        }

    def upsert_entries(self, entries: Iterable[ModelInferenceCacheEntry]) -> None:
        payloads = list(entries)
        if not payloads:
            return
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO model_inference_cache (
                  cache_key, backend, file_path, file_mtime_ns, file_size,
                  model_signature, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                  backend = excluded.backend,
                  file_path = excluded.file_path,
                  file_mtime_ns = excluded.file_mtime_ns,
                  file_size = excluded.file_size,
                  model_signature = excluded.model_signature,
                  payload_json = excluded.payload_json,
                  updated_at = CURRENT_TIMESTAMP
                """,
                [
                    (
                        item.cache_key,
                        item.backend,
                        item.file_path,
                        item.file_mtime_ns,
                        item.file_size,
                        item.model_signature,
                        json.dumps(item.payload, sort_keys=True),
                    )
                    for item in payloads
                ],
            )

    def purge(
        self,
        *,
        backend: str | None = None,
        file_paths: Iterable[str] | None = None,
        purge_all: bool = False,
    ) -> int:
        clauses: list[str] = []
        parameters: list[Any] = []
        if backend:
            clauses.append("backend = ?")
            parameters.append(backend)

        resolved_paths = [str(Path(path).resolve().as_posix()) for path in (file_paths or [])]
        if resolved_paths:
            placeholders = ", ".join("?" for _ in resolved_paths)
            clauses.append(f"file_path IN ({placeholders})")
            parameters.extend(resolved_paths)

        if not clauses and not purge_all:
            raise ValueError("Refusing to purge the entire inference cache without purge_all=True.")

        sql = "DELETE FROM model_inference_cache"
        if clauses:
            sql = f"{sql} WHERE {' AND '.join(clauses)}"

        with self.connection:
            cursor = self.connection.execute(sql, parameters)
        return int(cursor.rowcount or 0)
