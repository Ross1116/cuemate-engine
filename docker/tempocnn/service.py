from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import essentia.standard as es


MODEL_CACHE: dict[str, object] = {}


def detect_gpu_counts() -> tuple[int | None, int | None]:
    try:
        import tensorflow as tf
    except Exception:
        return None, None
    physical = len(tf.config.list_physical_devices("GPU"))
    logical = len(tf.config.list_logical_devices("GPU"))
    return physical, logical


def get_model(model_path: str):
    model = MODEL_CACHE.get(model_path)
    if model is None:
        model = es.TempoCNN(graphFilename=model_path)
        MODEL_CACHE[model_path] = model
    return model


def analyze_track(model, track_path: str) -> dict[str, object]:
    tempo_audio = es.MonoLoader(filename=track_path, sampleRate=11025, resampleQuality=4)()
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

        tf_physical_gpu_count, tf_logical_gpu_count = detect_gpu_counts()
        try:
            model = get_model(model_path)
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

        results: list[dict[str, object]] = []
        for track_path in track_paths:
            try:
                results.append(analyze_track(model, str(track_path)))
            except Exception as exc:
                results.append({"track_path": str(track_path), "error": str(exc)})

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
            get_model(default_model)
        except Exception:
            pass
    server = ThreadingHTTPServer(("0.0.0.0", port), TempoCNNHandler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
