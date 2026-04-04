from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cuemate_analysis.musicalkey_runtime import (
    MUSICALKEYCNN_POLICY_BALANCED,
    detect_gpu_counts,
    load_model,
    normalize_policy_choice,
    predict_keys,
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
        try:
            resolved_track_paths = [str(track_path) for track_path in track_paths]
            cached_results: dict[str, dict[str, object]] = {}
            missing_track_paths: list[str] = []
            for track_path in resolved_track_paths:
                cache_key = build_cache_key(model_path, normalized_policy, track_path)
                cached = RESULT_CACHE.get(cache_key)
                if cached is None:
                    missing_track_paths.append(track_path)
                    continue
                cached_results[track_path] = dict(cached)

            computed_results: dict[str, dict[str, object]] = {}
            if missing_track_paths:
                for item in predict_keys(model, runner_device, missing_track_paths, policy=normalized_policy):
                    track_path = str(item.get("track_path") or "")
                    computed_results[track_path] = item
                    if "error" not in item and track_path:
                        RESULT_CACHE[build_cache_key(model_path, normalized_policy, track_path)] = dict(item)

            results = [
                dict(cached_results.get(track_path) or computed_results.get(track_path) or {"track_path": track_path, "error": "missing_result"})
                for track_path in resolved_track_paths
            ]
        except Exception:
            for track_path in track_paths:
                try:
                    results.extend(predict_keys(model, runner_device, [str(track_path)], policy=normalized_policy))
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


RESULT_CACHE: dict[tuple[str, str, str, int, int], dict[str, object]] = {}


def build_cache_key(model_path: str, policy: str, track_path: str) -> tuple[str, str, str, int, int]:
    stat_result = Path(track_path).stat()
    return (
        model_path,
        policy,
        str(Path(track_path).resolve()),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_size),
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
