from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from typing import Any

import numpy as np

import essentia.standard as es


BUNDLE_STATE_CACHE: dict[str, dict[str, Any]] = {}
RESULT_CACHE: dict[tuple[str, str, int, int, str], dict[str, object]] = {}
SEMANTIC_AUDIO_WORKERS = max(1, min(4, os.cpu_count() or 1))
SEMANTIC_INFERENCE_WORKERS = max(1, min(3, int(os.getenv("ESSENTIA_SEMANTIC_INFERENCE_WORKERS", "1"))))
SAMPLE_RATE = 16000
MUSICNN_BATCH_SIZE = int(os.getenv("ESSENTIA_SEMANTIC_MUSICNN_BATCH_SIZE", "256"))
SEMANTIC_MAX_DURATION_SECONDS = max(0.0, float(os.getenv("ESSENTIA_SEMANTIC_MAX_DURATION_SECONDS", "90")))
THREAD_LOCAL = threading.local()
AUDIO_LOAD_EXECUTOR = ThreadPoolExecutor(max_workers=SEMANTIC_AUDIO_WORKERS, thread_name_prefix="essentia-audio")
INFERENCE_EXECUTOR = ThreadPoolExecutor(
    max_workers=SEMANTIC_INFERENCE_WORKERS,
    thread_name_prefix="essentia-infer",
)
MODEL_FILENAMES = {
    "musicnn_embedding_pb": "musicnn/msd-musicnn-1.pb",
    "musicnn_embedding_json": "musicnn/msd-musicnn-1.json",
    "deam_head_pb": "musicnn/deam-msd-musicnn-2.pb",
    "deam_head_json": "musicnn/deam-msd-musicnn-2.json",
    "danceability_pb": "musicnn/danceability-musicnn-msd-2.pb",
    "danceability_json": "musicnn/danceability-musicnn-msd-2.json",
    "mood_aggressive_pb": "musicnn/mood_aggressive-musicnn-msd-1.pb",
    "mood_aggressive_json": "musicnn/mood_aggressive-musicnn-msd-1.json",
    "mood_party_pb": "musicnn/mood_party-musicnn-msd-1.pb",
    "mood_party_json": "musicnn/mood_party-musicnn-msd-1.json",
    "mood_relaxed_pb": "musicnn/mood_relaxed-musicnn-msd-1.pb",
    "mood_relaxed_json": "musicnn/mood_relaxed-musicnn-msd-1.json",
}


def detect_gpu_counts() -> tuple[int | None, int | None]:
    try:
        import tensorflow as tf
    except Exception:
        return None, None
    physical = len(tf.config.list_physical_devices("GPU"))
    logical = len(tf.config.list_logical_devices("GPU"))
    return physical, logical


def build_musicnn_predictor(graph_filename: Path, output: str):
    kwargs = {
        "graphFilename": str(graph_filename),
        "output": output,
    }
    if MUSICNN_BATCH_SIZE != 0:
        kwargs["batchSize"] = MUSICNN_BATCH_SIZE
    return es.TensorflowPredictMusiCNN(**kwargs)


def trim_semantic_audio(audio):
    if SEMANTIC_MAX_DURATION_SECONDS <= 0:
        return audio
    max_samples = int(SAMPLE_RATE * SEMANTIC_MAX_DURATION_SECONDS)
    if len(audio) <= max_samples:
        return audio
    start = max(0, (len(audio) - max_samples) // 2)
    end = start + max_samples
    return audio[start:end]


def resolve_model_paths(model_root: str) -> dict[str, Path]:
    root = Path(model_root).resolve()
    return {name: root / relative for name, relative in MODEL_FILENAMES.items()}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_positive_index(metadata: dict[str, Any], *, keyword: str, default_index: int = 0) -> int:
    classes = metadata.get("classes") or metadata.get("class_names") or []
    for index, label in enumerate(classes):
        if keyword in str(label).strip().lower():
            return index
    return default_index


def make_bundle_cache_key(model_root: str, family_policy: str) -> str:
    return f"{Path(model_root).resolve().as_posix()}::{family_policy}"


def validate_model_paths(model_root: str) -> dict[str, Path]:
    paths = resolve_model_paths(model_root)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Essentia semantic model artifact missing: {path}")
    return paths


def build_runtime_state(*, family_policy: str) -> dict[str, Any]:
    family_map = {
        "danceability": "musicnn",
        "arousal_valence": "musicnn",
        "mood_aggressive": "musicnn",
        "mood_party": "musicnn",
        "mood_relaxed": "musicnn",
    }
    physical_gpu_count, logical_gpu_count = detect_gpu_counts()
    return {
        "family_policy": family_policy,
        "family_map": family_map,
        "tf_physical_gpu_count": physical_gpu_count,
        "tf_logical_gpu_count": logical_gpu_count,
        "runner_device": "cuda" if (logical_gpu_count or 0) > 0 else "cpu",
    }


def validate_requested_device(state: dict[str, Any], requested_device: str) -> None:
    normalized = (requested_device or "auto").strip().lower()
    logical_gpu_count = int(state.get("tf_logical_gpu_count") or 0)
    if normalized == "cuda" and logical_gpu_count <= 0:
        raise RuntimeError("CUDA was explicitly requested but no TensorFlow logical GPU devices are available.")


def load_bundle_state(model_root: str, family_policy: str) -> dict[str, Any]:
    cache_key = make_bundle_cache_key(model_root, family_policy)
    state = BUNDLE_STATE_CACHE.get(cache_key)
    if state is not None:
        return state
    validate_model_paths(model_root)
    state = build_runtime_state(family_policy=family_policy)
    BUNDLE_STATE_CACHE[cache_key] = state
    return state


def create_bundle(model_root: str, family_policy: str) -> dict[str, Any]:
    paths = validate_model_paths(model_root)
    state = load_bundle_state(model_root, family_policy)
    bundle = {
        "embedding_model": build_musicnn_predictor(paths["musicnn_embedding_pb"], "model/dense/BiasAdd"),
        "deam_head": es.TensorflowPredict2D(
            graphFilename=str(paths["deam_head_pb"]),
            output="model/Identity",
        ),
        "deam_meta": load_json(paths["deam_head_json"]),
        "danceability_model": build_musicnn_predictor(paths["danceability_pb"], "model/Sigmoid"),
        "danceability_meta": load_json(paths["danceability_json"]),
        "mood_aggressive_model": build_musicnn_predictor(paths["mood_aggressive_pb"], "model/Sigmoid"),
        "mood_aggressive_meta": load_json(paths["mood_aggressive_json"]),
        "mood_party_model": build_musicnn_predictor(paths["mood_party_pb"], "model/Sigmoid"),
        "mood_party_meta": load_json(paths["mood_party_json"]),
        "mood_relaxed_model": build_musicnn_predictor(paths["mood_relaxed_pb"], "model/Sigmoid"),
        "mood_relaxed_meta": load_json(paths["mood_relaxed_json"]),
        "family_map": dict(state["family_map"]),
        "tf_physical_gpu_count": state["tf_physical_gpu_count"],
        "tf_logical_gpu_count": state["tf_logical_gpu_count"],
        "runner_device": state["runner_device"],
    }
    return bundle


def load_thread_bundle(model_root: str, family_policy: str):
    cache_key = make_bundle_cache_key(model_root, family_policy)
    thread_bundles = getattr(THREAD_LOCAL, "bundles", None)
    if thread_bundles is None:
        thread_bundles = {}
        THREAD_LOCAL.bundles = thread_bundles
    bundle = thread_bundles.get(cache_key)
    if bundle is not None:
        return bundle
    bundle = create_bundle(model_root, family_policy)
    thread_bundles[cache_key] = bundle
    return bundle


def load_semantic_audio(track_path: str):
    audio = es.MonoLoader(filename=track_path, sampleRate=SAMPLE_RATE, resampleQuality=4)()
    return trim_semantic_audio(audio)


def aggregate_prediction(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.array([float(array)], dtype=float)
    if array.ndim == 1:
        return array
    flattened = array.reshape(-1, array.shape[-1])
    return np.mean(flattened, axis=0)


def select_binary_score(value: Any, metadata: dict[str, Any], keyword: str) -> float:
    aggregated = aggregate_prediction(value)
    if aggregated.size == 1:
        return float(np.clip(aggregated[0], 0.0, 1.0))
    positive_index = infer_positive_index(metadata, keyword=keyword, default_index=0)
    return float(np.clip(aggregated[min(positive_index, aggregated.size - 1)], 0.0, 1.0))


def select_deam_scores(value: Any, metadata: dict[str, Any]) -> tuple[float, float]:
    aggregated = aggregate_prediction(value)
    labels = [str(label).strip().lower() for label in (metadata.get("classes") or metadata.get("class_names") or [])]
    arousal_index = labels.index("arousal") if "arousal" in labels else 0
    valence_index = labels.index("valence") if "valence" in labels else min(1, aggregated.size - 1)
    return (
        float(np.clip(aggregated[min(arousal_index, aggregated.size - 1)], 0.0, 1.0)),
        float(np.clip(aggregated[min(valence_index, aggregated.size - 1)], 0.0, 1.0)),
    )


def analyze_audio(bundle, track_path: str, semantic_audio) -> dict[str, object]:
    embeddings = bundle["embedding_model"](semantic_audio)
    deam_prediction = bundle["deam_head"](embeddings)
    arousal_abs, valence_abs = select_deam_scores(deam_prediction, bundle["deam_meta"])
    danceability_abs = select_binary_score(bundle["danceability_model"](semantic_audio), bundle["danceability_meta"], "dance")
    mood_aggressive_abs = select_binary_score(bundle["mood_aggressive_model"](semantic_audio), bundle["mood_aggressive_meta"], "aggress")
    mood_party_abs = select_binary_score(bundle["mood_party_model"](semantic_audio), bundle["mood_party_meta"], "party")
    mood_relaxed_abs = select_binary_score(bundle["mood_relaxed_model"](semantic_audio), bundle["mood_relaxed_meta"], "relax")
    return {
        "track_path": track_path,
        "danceability_abs": danceability_abs,
        "arousal_abs": arousal_abs,
        "valence_abs": valence_abs,
        "mood_aggressive_abs": mood_aggressive_abs,
        "mood_party_abs": mood_party_abs,
        "mood_relaxed_abs": mood_relaxed_abs,
        "semantic_source": "best_per_task:musicnn_stable",
        "family_map": dict(bundle["family_map"]),
        "runner_device": bundle["runner_device"],
        "tf_physical_gpu_count": bundle["tf_physical_gpu_count"],
        "tf_logical_gpu_count": bundle["tf_logical_gpu_count"],
    }


def build_cache_key(model_root: str, family_policy: str, track_path: str) -> tuple[str, str, int, int, str]:
    stat_result = Path(track_path).stat()
    return (
        str(Path(model_root).resolve()),
        str(Path(track_path).resolve()),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_size),
        family_policy,
    )


def warm_inference_worker(model_root: str, family_policy: str) -> bool:
    load_thread_bundle(model_root, family_policy)
    return True


def analyze_loaded_audio(model_root: str, family_policy: str, track_path: str, semantic_audio) -> dict[str, object]:
    bundle = load_thread_bundle(model_root, family_policy)
    return analyze_audio(bundle, track_path, semantic_audio)


def analyze_tracks(model_root: str, family_policy: str, track_paths: list[str]) -> list[dict[str, object]]:
    audio_worker_count = max(1, min(SEMANTIC_AUDIO_WORKERS, len(track_paths)))
    inference_worker_count = max(1, min(SEMANTIC_INFERENCE_WORKERS, len(track_paths)))

    def load_one(track_path: str):
        try:
            return track_path, load_semantic_audio(track_path), None
        except Exception as exc:
            return track_path, None, str(exc)

    if audio_worker_count == 1:
        loaded = [load_one(track_path) for track_path in track_paths]
    else:
        loaded = list(AUDIO_LOAD_EXECUTOR.map(load_one, track_paths))

    results: list[dict[str, object]] = []
    pending: list[tuple[str, Any]] = []
    for track_path, semantic_audio, load_error in loaded:
        if load_error is not None or semantic_audio is None:
            results.append({"track_path": str(track_path), "error": load_error or "audio_load_failed"})
            continue
        pending.append((track_path, semantic_audio))
    if not pending:
        return results

    if inference_worker_count == 1:
        for track_path, semantic_audio in pending:
            try:
                results.append(analyze_loaded_audio(model_root, family_policy, track_path, semantic_audio))
            except Exception as exc:
                results.append({"track_path": str(track_path), "error": str(exc)})
        return results

    futures: dict[Future[dict[str, object]], str] = {
        INFERENCE_EXECUTOR.submit(analyze_loaded_audio, model_root, family_policy, track_path, semantic_audio): track_path
        for track_path, semantic_audio in pending
    }
    for future, track_path in futures.items():
        try:
            results.append(future.result())
        except Exception as exc:
            results.append({"track_path": str(track_path), "error": str(exc)})
    return results


class EssentiaSemanticHandler(BaseHTTPRequestHandler):
    server_version = "CueMateEssentiaSemantics/0.1"

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
                "loaded_bundles": sorted(BUNDLE_STATE_CACHE.keys()),
                "tf_physical_gpu_count": tf_physical_gpu_count,
                "tf_logical_gpu_count": tf_logical_gpu_count,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/analyze-semantics":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid_json: {exc}"})
            return
        if not isinstance(payload, dict):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json_shape"})
            return
        model_root = str(payload.get("model_root") or "").strip()
        family_policy = str(payload.get("family_policy") or "best_per_task").strip().lower()
        track_paths = payload.get("tracks") or []
        if not model_root or not isinstance(track_paths, list) or not track_paths:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "model_root and tracks[] are required."})
            return
        try:
            bundle_state = load_bundle_state(model_root, family_policy)
        except Exception as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": f"model_load_failed: {exc}",
                    "tf_physical_gpu_count": None,
                    "tf_logical_gpu_count": None,
                },
            )
            return
        resolved_track_paths = [str(track_path) for track_path in track_paths]
        cached_results: dict[str, dict[str, object]] = {}
        immediate_results: dict[str, dict[str, object]] = {}
        missing_track_paths: list[str] = []
        for track_path in resolved_track_paths:
            try:
                cache_key = build_cache_key(model_root, family_policy, track_path)
            except Exception as exc:
                immediate_results[track_path] = {"track_path": track_path, "error": str(exc)}
                continue
            cached = RESULT_CACHE.get(cache_key)
            if cached is None:
                missing_track_paths.append(track_path)
                continue
            cached_results[track_path] = dict(cached)

        computed_results: dict[str, dict[str, object]] = {}
        if missing_track_paths:
            for item in analyze_tracks(model_root, family_policy, missing_track_paths):
                track_path = str(item.get("track_path") or "")
                computed_results[track_path] = item
                if "error" not in item and track_path:
                    RESULT_CACHE[build_cache_key(model_root, family_policy, track_path)] = dict(item)

        results = [
            dict(
                immediate_results.get(track_path)
                or cached_results.get(track_path)
                or computed_results.get(track_path)
                or {"track_path": track_path, "error": "missing_result"}
            )
            for track_path in resolved_track_paths
        ]
        self._send_json(
            HTTPStatus.OK,
            {
                "tf_physical_gpu_count": bundle_state["tf_physical_gpu_count"],
                "tf_logical_gpu_count": bundle_state["tf_logical_gpu_count"],
                "results": results,
            },
        )


def main() -> int:
    port = int(os.getenv("ESSENTIA_SEMANTIC_SERVICE_PORT", "47833"))
    model_root = os.getenv("CUEMATE_ESSENTIA_SEMANTIC_MODEL_ROOT", "/workspace/python/models/essentia_semantics")
    family_policy = os.getenv("CUEMATE_ESSENTIA_SEMANTIC_MODEL_FAMILY_POLICY", "best_per_task")
    requested_device = os.getenv("CUEMATE_ESSENTIA_SEMANTIC_DEVICE", "auto")
    runtime_state = load_bundle_state(model_root, family_policy)
    validate_requested_device(runtime_state, requested_device)
    if SEMANTIC_INFERENCE_WORKERS > 1:
        warm_futures = [
            INFERENCE_EXECUTOR.submit(warm_inference_worker, model_root, family_policy)
            for _ in range(SEMANTIC_INFERENCE_WORKERS)
        ]
        for future in warm_futures:
            future.result()
    server = ThreadingHTTPServer(("0.0.0.0", port), EssentiaSemanticHandler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
