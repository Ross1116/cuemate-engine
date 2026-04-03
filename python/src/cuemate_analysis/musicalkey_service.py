from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cuemate_analysis.musicalkey_runtime import (
    MUSICALKEYCNN_POLICY_BALANCED,
    detect_gpu_counts,
    load_model,
    normalize_policy_choice,
    predict_key,
    warm_pipeline,
)


class MusicalKeyHandler(BaseHTTPRequestHandler):
    server_version = "CueMateMusicalKeyCNN/0.1"

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
        physical_gpu_count, logical_gpu_count = detect_gpu_counts()
        self._send_json(
            HTTPStatus.OK,
            {
                "status": "ok",
                "torch_physical_gpu_count": physical_gpu_count,
                "torch_logical_gpu_count": logical_gpu_count,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/analyze-key":
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
        device = str(payload.get("device") or "auto").strip().lower()
        track_paths = payload.get("tracks") or []
        policy = str(payload.get("policy") or MUSICALKEYCNN_POLICY_BALANCED).strip().lower()
        if not model_path or not isinstance(track_paths, list) or not track_paths:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "model_path and tracks[] are required."},
            )
            return
        try:
            normalized_policy = normalize_policy_choice(policy)
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid_policy: {exc}"})
            return

        physical_gpu_count, logical_gpu_count = detect_gpu_counts()
        try:
            model, runner_device = load_model(model_path, device_choice=device)
        except Exception as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": f"model_load_failed: {exc}",
                    "torch_physical_gpu_count": physical_gpu_count,
                    "torch_logical_gpu_count": logical_gpu_count,
                },
            )
            return

        results: list[dict[str, object]] = []
        for track_path in track_paths:
            try:
                results.append(predict_key(model, runner_device, str(track_path), policy=normalized_policy))
            except Exception as exc:
                results.append({"track_path": str(track_path), "error": str(exc)})

        self._send_json(
            HTTPStatus.OK,
            {
                "torch_physical_gpu_count": physical_gpu_count,
                "torch_logical_gpu_count": logical_gpu_count,
                "runner_device": str(runner_device),
                "policy": normalized_policy,
                "results": results,
            },
        )


def main() -> int:
    port = int(os.getenv("MUSICALKEYCNN_SERVICE_PORT", "47832"))
    default_model = os.getenv("CUEMATE_MUSICALKEYCNN_DEFAULT_MODEL")
    default_device = os.getenv("CUEMATE_MUSICALKEYCNN_DEFAULT_DEVICE", "auto")
    if default_model:
        try:
            model, runner_device = load_model(default_model, device_choice=default_device)
            warm_pipeline(model, runner_device)
        except Exception:
            pass
    server = ThreadingHTTPServer(("0.0.0.0", port), MusicalKeyHandler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
