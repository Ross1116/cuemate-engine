from pathlib import Path

from cuemate_analysis.persistent_inference_cache import (
    ModelInferenceCacheEntry,
    PersistentInferenceCache,
)


def test_persistent_inference_cache_round_trip_and_purge(tmp_path: Path) -> None:
    cache_path = tmp_path / "inference-cache.db"

    with PersistentInferenceCache(cache_path) as cache:
        cache.upsert_entries(
            [
                ModelInferenceCacheEntry(
                    backend="tempocnn",
                    cache_key="tempo-key",
                    file_path="D:/music/tempo.flac",
                    file_mtime_ns=1,
                    file_size=100,
                    model_signature="tempo-v1",
                    payload={"backend": "tempocnn", "bpm": 128.0, "available": True},
                ),
                ModelInferenceCacheEntry(
                    backend="musicalkeycnn",
                    cache_key="key-key",
                    file_path="D:/music/key.flac",
                    file_mtime_ns=2,
                    file_size=200,
                    model_signature="key-v1",
                    payload={"backend": "musicalkeycnn", "key": "8A", "available": True},
                ),
            ]
        )

        tempo_payloads = cache.fetch_payloads("tempocnn", ["tempo-key", "missing"])
        assert tempo_payloads == {
            "tempo-key": {"backend": "tempocnn", "bpm": 128.0, "available": True}
        }

        deleted_scoped = cache.purge(backend="tempocnn", file_paths=["D:/music/tempo.flac"])
        assert deleted_scoped == 1
        assert cache.fetch_payloads("tempocnn", ["tempo-key"]) == {}

        deleted_remaining = cache.purge(purge_all=True)
        assert deleted_remaining == 1
        assert cache.fetch_payloads("musicalkeycnn", ["key-key"]) == {}
