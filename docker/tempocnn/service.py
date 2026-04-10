from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import essentia.standard as es
from cuemate_analysis.path_safety import resolve_allowed_roots, resolve_existing_file_path


logger = logging.getLogger(__name__)

MODEL_CACHE: dict[str, object] = {}
RESULT_CACHE: dict[tuple[str, str, int, int], dict[str, object]] = {}
TEMPOCNN_AUDIO_WORKERS = max(1, min(4, os.cpu_count() or 1))
ALLOWED_SERVICE_ROOTS_ENV = "CUEMATE_SERVICE_ALLOWED_ROOTS"


def detect_gpu_counts() -> tuple[int | None, int | None]:
    try:
        import tensorflow as tf
    except Exception:
        return None, None
    physical = len(tf.config.list_physical_devices("GPU"))
    logical = len(tf.config.list_logical_devices("GPU"))
    return physical, logical


def get_model(model_path: Path):
    model_key = model_path.as_posix()
    model = MODEL_CACHE.get(model_key)
    if model is None:
        model = es.TempoCNN(graphFilename=str(model_path))
        MODEL_CACHE[model_key] = model
    return model


def load_tempo_audio(track_path: Path):
    return es.MonoLoader(filename=str(track_path), sampleRate=11025, resampleQuality=4)()


def analyze_audio(model, track_path: str, tempo_audio) -> dict[str, object]:
    global_tempo, local_tempi, local_probs = model(tempo_audio)

    local_tempi_array = np.asarray(local_tempi, dtype=float)
    local_probs_array = np.asarray(local_probs, dtype=float)
    spread = (
        float(np.median(np.abs(local_tempi_array - float(global_tempo))))
        if local_tempi_array.size
        else None
    )
    agreement = (
        float(np.mean(np.abs(local_tempi_array - float(global_tempo)) <= 2.0))
        if local_tempi_array.size
        else 0.0
    )
    stability = (
        max(0.0, min(1.0, 1.0 - ((spread or 0.0) / max(float(global_tempo) * 0.05, 1.0))))
        if local_tempi_array.size
        else 0.0
    )
    confidence = (agreement + stability) / 2.0 if local_tempi_array.size else 0.0
    return {
        "track_path": track_path,
        "bpm": float(global_tempo),
        "confidence": float(confidence),
        "local_count": int(local_tempi_array.size),
        "tempo_spread": spread,
        "agreement_with_global": agreement,
        "probability_peak": float(np.max(local_probs_array)) if local_probs_array.size else None,
    }


def analyze_tracks(model, track_paths: list[Path]) -> list[dict[str, object]]:
    worker_count = max(1, min(TEMPOCNN_AUDIO_WORKERS, len(track_paths)))

    def load_one(track_path: Path):
        try:
            return track_path, load_tempo_audio(track_path), None
        except Exception as exc:
            return track_path, None, str(exc)

    if worker_count == 1:
        loaded = [load_one(track_path) for track_path in track_paths]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            loaded = list(executor.map(load_one, track_paths))

    results: list[dict[str, object]] = []
    for track_path, tempo_audio, load_error in loaded:
        if load_error is not None or tempo_audio is None:
            results.append({"track_path": track_path.as_posix(), "error": load_error or "audio_load_failed"})
            continue
        try:
            results.append(analyze_audio(model, track_path.as_posix(), tempo_audio))
        except Exception as exc:
            results.append({"track_path": track_path.as_posix(), "error": str(exc)})
    return results


def build_cache_key(model_path: Path, track_path: Path) -> tuple[str, str, int, int]:
    stat_result = track_path.stat()
    return (
        model_path.as_posix(),
        track_path.as_posix(),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_size),
    )


class TempoCNNHandler(BaseHTTPRequestHandler):
    server_version = "CueMateTempoCNN/0.1"

    def log_message(self, format, *args):
        return

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        tf_physical_gpu_count, tf_logical_gpu_count = detect_gpu_counts()
        self._send_json(
            HTTPStatus.OK,
            {
                "status": "ok",
                "loaded_models": sorted(MODEL_CACHE.keys()),
                "tf_physical_gpu_count": tf_physical_gpu_count,
                "tf_logical_gpu_count": tf_logical_gpu_count,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/analyze-bpm":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid_json: {exc}"})
            return

        model_path = str(payload.get("model_path") or "").strip()
        track_paths = payload.get("tracks") or []
        if not model_path or not isinstance(track_paths, list) or not track_paths:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "model_path and tracks[] are required."},
            )
            return

        allowed_roots = resolve_allowed_roots(os.getenv(ALLOWED_SERVICE_ROOTS_ENV))
        try:
            resolved_model_path = resolve_existing_file_path(
                model_path,
                "model_path",
                allowed_roots=allowed_roots,
            )
            resolved_track_paths = [
                resolve_existing_file_path(
                    str(track_path),
                    "track_path",
                    allowed_roots=allowed_roots,
                )
                for track_path in track_paths
            ]
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        tf_physical_gpu_count, tf_logical_gpu_count = detect_gpu_counts()
        try:
            model = get_model(resolved_model_path)
        except Exception as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": f"model_load_failed: {exc}",
                    "tf_physical_gpu_count": tf_physical_gpu_count,
                    "tf_logical_gpu_count": tf_logical_gpu_count,
                },
            )
            return

        cached_results: dict[str, dict[str, object]] = {}
        immediate_results: dict[str, dict[str, object]] = {}
        missing_track_paths: list[Path] = []
        for track_path in resolved_track_paths:
            track_key = track_path.as_posix()
            try:
                cache_key = build_cache_key(resolved_model_path, track_path)
            except Exception as exc:
                immediate_results[track_key] = {"track_path": track_key, "error": str(exc)}
                continue
            cached = RESULT_CACHE.get(cache_key)
            if cached is None:
                missing_track_paths.append(track_path)
                continue
            cached_results[track_key] = dict(cached)

        computed_results: dict[str, dict[str, object]] = {}
        if missing_track_paths:
            for item in analyze_tracks(model, missing_track_paths):
                track_path = str(item.get("track_path") or "")
                computed_results[track_path] = item
                if "error" not in item and track_path:
                    RESULT_CACHE[
                        build_cache_key(
                            resolved_model_path,
                            Path(track_path),
                        )
                    ] = dict(item)

        results = [
            dict(
                immediate_results.get(track_path.as_posix())
                or cached_results.get(track_path.as_posix())
                or computed_results.get(track_path.as_posix())
                or {"track_path": track_path.as_posix(), "error": "missing_result"}
            )
            for track_path in resolved_track_paths
        ]

        self._send_json(
            HTTPStatus.OK,
            {
                "tf_physical_gpu_count": tf_physical_gpu_count,
                "tf_logical_gpu_count": tf_logical_gpu_count,
                "results": results,
            },
        )


def main() -> int:
    port = int(os.getenv("TEMPOCNN_SERVICE_PORT", "47831"))
    default_model = os.getenv("CUEMATE_TEMPOCNN_DEFAULT_MODEL")
    if default_model:
        try:
            get_model(
                resolve_existing_file_path(
                    default_model,
                    "CUEMATE_TEMPOCNN_DEFAULT_MODEL",
                    allowed_roots=resolve_allowed_roots(os.getenv(ALLOWED_SERVICE_ROOTS_ENV)),
                )
            )
        except Exception as exc:
            logger.warning("Failed to preload default tempo model '%s': %s", default_model, exc, exc_info=exc)
    server = ThreadingHTTPServer(("0.0.0.0", port), TempoCNNHandler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
