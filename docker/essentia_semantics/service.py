from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from typing import Any

import essentia.standard as es
import numpy as np


TEMPO_MODEL_CACHE: dict[str, object] = {}
SEMANTIC_BUNDLE_STATE_CACHE: dict[str, dict[str, Any]] = {}
TEMPO_RESULT_CACHE: dict[tuple[str, str, int, int], dict[str, object]] = {}
SEMANTIC_RESULT_CACHE: dict[tuple[str, str, int, int, str], dict[str, object]] = {}
TEMPO_AUDIO_WORKERS = max(1, min(4, os.cpu_count() or 1))
SEMANTIC_AUDIO_WORKERS = max(1, min(4, os.cpu_count() or 1))
SEMANTIC_INFERENCE_WORKERS = max(1, min(3, int(os.getenv("ESSENTIA_SEMANTIC_INFERENCE_WORKERS", "1"))))
SAMPLE_RATE = 16000
TEMPO_SAMPLE_RATE = 11025
MUSICNN_BATCH_SIZE = int(os.getenv("ESSENTIA_SEMANTIC_MUSICNN_BATCH_SIZE", "256"))
THREAD_LOCAL = threading.local()
AUDIO_LOAD_EXECUTOR = ThreadPoolExecutor(max_workers=SEMANTIC_AUDIO_WORKERS, thread_name_prefix="tf-audio")
INFERENCE_EXECUTOR = ThreadPoolExecutor(max_workers=SEMANTIC_INFERENCE_WORKERS, thread_name_prefix="tf-infer")
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


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return float(max(minimum, min(maximum, value)))


def detect_gpu_counts() -> tuple[int | None, int | None]:
    try:
        import tensorflow as tf
    except Exception:
        return None, None
    physical = len(tf.config.list_physical_devices("GPU"))
    logical = len(tf.config.list_logical_devices("GPU"))
    return physical, logical


def get_tempo_model(model_path: str):
    model = TEMPO_MODEL_CACHE.get(model_path)
    if model is None:
        model = es.TempoCNN(graphFilename=model_path)
        TEMPO_MODEL_CACHE[model_path] = model
    return model


def build_musicnn_predictor(graph_filename: Path, output: str):
    kwargs = {"graphFilename": str(graph_filename), "output": output}
    if MUSICNN_BATCH_SIZE != 0:
        kwargs["batchSize"] = MUSICNN_BATCH_SIZE
    return es.TensorflowPredictMusiCNN(**kwargs)


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


def fingerprint_model_artifacts(model_root: str) -> str:
    paths = resolve_model_paths(model_root)
    identity: list[dict[str, object]] = []
    for name, path in sorted(paths.items()):
        stat_result = path.stat()
        identity.append(
            {
                "name": name,
                "path": path.resolve().as_posix(),
                "mtime_ns": int(stat_result.st_mtime_ns),
                "size": int(stat_result.st_size),
            }
        )
    digest = json.dumps(identity, sort_keys=True).encode("utf-8")
    return hashlib.sha1(digest).hexdigest()[:16]


def make_bundle_cache_key(model_root: str, family_policy: str) -> str:
    resolved_root = Path(model_root).resolve().as_posix()
    artifact_fingerprint = fingerprint_model_artifacts(model_root)
    return f"{resolved_root}::{artifact_fingerprint}::{family_policy}"


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
    state = SEMANTIC_BUNDLE_STATE_CACHE.get(cache_key)
    if state is not None:
        return state
    validate_model_paths(model_root)
    state = build_runtime_state(family_policy=family_policy)
    SEMANTIC_BUNDLE_STATE_CACHE[cache_key] = state
    return state


def create_bundle(model_root: str, family_policy: str) -> dict[str, Any]:
    paths = validate_model_paths(model_root)
    state = load_bundle_state(model_root, family_policy)
    return {
        "embedding_model": build_musicnn_predictor(paths["musicnn_embedding_pb"], "model/dense/BiasAdd"),
        "deam_head": es.TensorflowPredict2D(graphFilename=str(paths["deam_head_pb"]), output="model/Identity"),
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


def load_thread_bundle(model_root: str, family_policy: str):
    cache_key = make_bundle_cache_key(model_root, family_policy)
    thread_bundles = getattr(THREAD_LOCAL, "bundles", None)
    if thread_bundles is None:
        thread_bundles = {}
        THREAD_LOCAL.bundles = thread_bundles
    bundle = thread_bundles.get(cache_key)
    if bundle is None:
        bundle = create_bundle(model_root, family_policy)
        thread_bundles[cache_key] = bundle
    return bundle


def load_tempo_audio(track_path: str):
    return es.MonoLoader(filename=track_path, sampleRate=TEMPO_SAMPLE_RATE, resampleQuality=4)()


def load_semantic_audio(track_path: str):
    return es.MonoLoader(filename=track_path, sampleRate=SAMPLE_RATE, resampleQuality=4)()


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


def analyze_tempo_audio(model, track_path: str, tempo_audio) -> dict[str, object]:
    global_tempo, local_tempi, local_probs = model(tempo_audio)
    local_tempi_array = np.asarray(local_tempi, dtype=float)
    local_probs_array = np.asarray(local_probs, dtype=float)
    spread = float(np.median(np.abs(local_tempi_array - float(global_tempo)))) if local_tempi_array.size else None
    agreement = float(np.mean(np.abs(local_tempi_array - float(global_tempo)) <= 2.0)) if local_tempi_array.size else 0.0
    stability = max(0.0, min(1.0, 1.0 - ((spread or 0.0) / max(float(global_tempo) * 0.05, 1.0)))) if local_tempi_array.size else 0.0
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


def analyze_semantic_segment(bundle, track_path: str, semantic_audio) -> dict[str, float]:
    embeddings = bundle["embedding_model"](semantic_audio)
    deam_prediction = bundle["deam_head"](embeddings)
    arousal_abs, valence_abs = select_deam_scores(deam_prediction, bundle["deam_meta"])
    danceability_abs = select_binary_score(bundle["danceability_model"](semantic_audio), bundle["danceability_meta"], "dance")
    mood_aggressive_abs = select_binary_score(bundle["mood_aggressive_model"](semantic_audio), bundle["mood_aggressive_meta"], "aggress")
    mood_party_abs = select_binary_score(bundle["mood_party_model"](semantic_audio), bundle["mood_party_meta"], "party")
    mood_relaxed_abs = select_binary_score(bundle["mood_relaxed_model"](semantic_audio), bundle["mood_relaxed_meta"], "relax")
    return {
        "danceability_abs": danceability_abs,
        "arousal_abs": arousal_abs,
        "valence_abs": valence_abs,
        "mood_aggressive_abs": mood_aggressive_abs,
        "mood_party_abs": mood_party_abs,
        "mood_relaxed_abs": mood_relaxed_abs,
    }


def estimate_semantic_confidence(sample: dict[str, float]) -> float:
    confidence_parts = [
        abs(float(sample["arousal_abs"]) - 0.5) * 2.0,
        abs(float(sample["danceability_abs"]) - 0.5) * 2.0,
        abs(float(sample["mood_party_abs"]) - 0.5) * 2.0,
        abs(float(sample["mood_relaxed_abs"]) - 0.5) * 2.0,
        abs(float(sample["mood_aggressive_abs"]) - 0.5) * 2.0,
    ]
    return clamp(sum(confidence_parts) / len(confidence_parts))


def compute_fused_proxy(sample: dict[str, float], auxiliary: dict[str, float | None]) -> float:
    return clamp(
        (0.34 * float(sample["arousal_abs"]))
        + (0.24 * float(sample["danceability_abs"]))
        + (0.18 * float(sample["mood_party_abs"]))
        + (0.14 * (1.0 - float(sample["mood_relaxed_abs"])))
        + (0.05 * float(sample["mood_aggressive_abs"]))
        + (0.03 * float(auxiliary.get("loudness_norm") or 0.0))
        + (0.02 * float(auxiliary.get("bass_abs") or 0.0))
    )


def extract_center_excerpt(audio: np.ndarray, excerpt_seconds: float) -> np.ndarray:
    if excerpt_seconds <= 0:
        return audio
    excerpt_samples = min(len(audio), max(1, int(excerpt_seconds * SAMPLE_RATE)))
    if len(audio) <= excerpt_samples:
        return audio
    start = max(0, (len(audio) - excerpt_samples) // 2)
    return audio[start:start + excerpt_samples]


def extract_window(audio: np.ndarray, start_ratio: float, excerpt_seconds: float) -> np.ndarray:
    excerpt_samples = min(len(audio), max(1, int(excerpt_seconds * SAMPLE_RATE)))
    if len(audio) <= excerpt_samples:
        return audio
    start = int(max(0.0, min(1.0, start_ratio)) * max(len(audio) - excerpt_samples, 0))
    return audio[start:start + excerpt_samples]


def extract_peak_window(audio: np.ndarray, excerpt_seconds: float) -> np.ndarray:
    excerpt_samples = min(len(audio), max(1, int(excerpt_seconds * SAMPLE_RATE)))
    if len(audio) <= excerpt_samples:
        return audio
    frame_length = min(excerpt_samples, 4096)
    hop_length = max(512, frame_length // 4)
    rms = es.RMS()
    best_start = max(0, (len(audio) - excerpt_samples) // 2)
    best_energy = -1.0
    for start in range(0, max(len(audio) - excerpt_samples, 1), hop_length):
        window = audio[start:start + excerpt_samples]
        energy = float(rms(window))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return audio[best_start:best_start + excerpt_samples]


def weighted_average_samples(samples: list[tuple[float, dict[str, float]]]) -> dict[str, float]:
    total_weight = sum(weight for weight, _ in samples) or 1.0
    keys = samples[0][1].keys()
    return {
        key: sum(weight * float(sample[key]) for weight, sample in samples) / total_weight
        for key in keys
    }


def analyze_semantic_audio(bundle, track_path: str, semantic_audio: np.ndarray, auxiliary: dict[str, float | None], request_cfg: dict[str, float]) -> dict[str, object]:
    middle = analyze_semantic_segment(bundle, track_path, extract_center_excerpt(semantic_audio, request_cfg["default_excerpt_seconds"]))
    semantic_confidence = estimate_semantic_confidence(middle)
    semantic_fused = compute_fused_proxy(middle, auxiliary)
    triggers: list[str] = []
    dsp_proxy = auxiliary.get("dsp_proxy_energy")
    if dsp_proxy is not None and abs(float(dsp_proxy) - semantic_fused) >= request_cfg["mismatch_threshold"]:
        triggers.append("semantic_vs_dsp_mismatch")
    if semantic_confidence < request_cfg["confidence_threshold"]:
        triggers.append("low_semantic_confidence")
    structure_rms_cv = auxiliary.get("structure_rms_cv")
    if structure_rms_cv is not None and float(structure_rms_cv) >= request_cfg["structure_rms_cv_threshold"]:
        triggers.append("track_structure_suspicion")
    playlist_outlier_score = auxiliary.get("playlist_outlier_score")
    if playlist_outlier_score is not None and float(playlist_outlier_score) >= request_cfg["outlier_zscore_threshold"]:
        triggers.append("playlist_outlier")

    if triggers:
        multisample_seconds = request_cfg["multisample_excerpt_seconds"]
        samples = [
            (0.15, analyze_semantic_segment(bundle, track_path, extract_window(semantic_audio, 0.0, multisample_seconds))),
            (0.40, analyze_semantic_segment(bundle, track_path, extract_center_excerpt(semantic_audio, multisample_seconds))),
            (0.30, analyze_semantic_segment(bundle, track_path, extract_peak_window(semantic_audio, multisample_seconds))),
            (0.15, analyze_semantic_segment(bundle, track_path, extract_window(semantic_audio, 1.0, multisample_seconds))),
        ]
        averaged = weighted_average_samples(samples)
        return {
            "track_path": track_path,
            **averaged,
            "semantic_confidence": clamp(sum(estimate_semantic_confidence(sample) * weight for weight, sample in samples) / sum(weight for weight, _ in samples)),
            "sampling_mode": "multisample",
            "sampling_triggers": triggers,
            "semantic_source": "best_per_task:musicnn_stable",
        }

    return {
        "track_path": track_path,
        **middle,
        "semantic_confidence": semantic_confidence,
        "sampling_mode": "middle_excerpt",
        "sampling_triggers": [],
        "semantic_source": "best_per_task:musicnn_stable",
    }


def build_tempo_cache_key(model_path: str, track_path: str) -> tuple[str, str, int, int]:
    stat_result = Path(track_path).stat()
    return (model_path, str(Path(track_path).resolve()), int(stat_result.st_mtime_ns), int(stat_result.st_size))


def build_semantic_cache_key(model_root: str, family_policy: str, track_path: str, sampling_mode: str) -> tuple[str, str, int, int, str]:
    stat_result = Path(track_path).stat()
    artifact_fingerprint = fingerprint_model_artifacts(model_root)
    return (artifact_fingerprint, str(Path(track_path).resolve()), int(stat_result.st_mtime_ns), int(stat_result.st_size), f"{family_policy}:{sampling_mode}")


def analyze_tempo_tracks(model, track_paths: list[str]) -> list[dict[str, object]]:
    worker_count = max(1, min(TEMPO_AUDIO_WORKERS, len(track_paths)))

    def load_one(track_path: str):
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
            results.append({"track_path": track_path, "error": load_error or "audio_load_failed"})
            continue
        try:
            results.append(analyze_tempo_audio(model, track_path, tempo_audio))
        except Exception as exc:
            results.append({"track_path": track_path, "error": str(exc)})
    return results


def analyze_semantic_tracks(
    model_root: str,
    family_policy: str,
    track_paths: list[str],
    auxiliary_features_by_track: dict[str, dict[str, float | None]],
    request_cfg: dict[str, float],
) -> list[dict[str, object]]:
    worker_count = max(1, min(SEMANTIC_AUDIO_WORKERS, len(track_paths)))

    def load_one(track_path: str):
        try:
            return track_path, load_semantic_audio(track_path), None
        except Exception as exc:
            return track_path, None, str(exc)

    if worker_count == 1:
        loaded = [load_one(track_path) for track_path in track_paths]
    else:
        loaded = list(AUDIO_LOAD_EXECUTOR.map(load_one, track_paths))

    results: list[dict[str, object]] = []
    pending: list[tuple[str, np.ndarray]] = []
    for track_path, semantic_audio, load_error in loaded:
        if load_error is not None or semantic_audio is None:
            results.append({"track_path": track_path, "error": load_error or "audio_load_failed"})
            continue
        pending.append((track_path, semantic_audio))
    if not pending:
        return results

    bundle = load_thread_bundle(model_root, family_policy)

    for track_path, semantic_audio in pending:
        auxiliary = dict(auxiliary_features_by_track.get(track_path, {}) or {})
        cache_key = build_semantic_cache_key(model_root, family_policy, track_path, "adaptive")
        cached = SEMANTIC_RESULT_CACHE.get(cache_key)
        if cached is not None:
            results.append(dict(cached))
            continue
        try:
            payload = analyze_semantic_audio(bundle, track_path, semantic_audio, auxiliary, request_cfg)
            payload.update(
                {
                    "family_map": dict(bundle["family_map"]),
                    "runner_device": bundle["runner_device"],
                    "tf_physical_gpu_count": bundle["tf_physical_gpu_count"],
                    "tf_logical_gpu_count": bundle["tf_logical_gpu_count"],
                }
            )
            SEMANTIC_RESULT_CACHE[cache_key] = dict(payload)
            results.append(payload)
        except Exception as exc:
            results.append({"track_path": track_path, "error": str(exc)})
    return results


class SharedTensorflowHandler(BaseHTTPRequestHandler):
    server_version = "CueMateSharedTensorflow/0.1"

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
                "loaded_tempo_models": sorted(TEMPO_MODEL_CACHE.keys()),
                "loaded_semantic_bundles": sorted(SEMANTIC_BUNDLE_STATE_CACHE.keys()),
                "tf_physical_gpu_count": tf_physical_gpu_count,
                "tf_logical_gpu_count": tf_logical_gpu_count,
            },
        )

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid_json: {exc}"})
            return

        if self.path == "/analyze-bpm":
            self._handle_analyze_bpm(payload)
            return
        if self.path == "/analyze-semantics":
            self._handle_analyze_semantics(payload)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def _handle_analyze_bpm(self, payload: dict[str, Any]) -> None:
        model_path = str(payload.get("model_path") or "").strip()
        track_paths = payload.get("tracks") or []
        if not model_path or not isinstance(track_paths, list) or not track_paths:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "model_path and tracks[] are required."})
            return
        tf_physical_gpu_count, tf_logical_gpu_count = detect_gpu_counts()
        try:
            model = get_tempo_model(model_path)
        except Exception as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"model_load_failed: {exc}", "tf_physical_gpu_count": tf_physical_gpu_count, "tf_logical_gpu_count": tf_logical_gpu_count},
            )
            return

        resolved_track_paths = [str(track_path) for track_path in track_paths]
        results: list[dict[str, object]] = []
        missing: list[str] = []
        cached_results: dict[str, dict[str, object]] = {}
        for track_path in resolved_track_paths:
            try:
                cache_key = build_tempo_cache_key(model_path, track_path)
            except Exception as exc:
                results.append({"track_path": track_path, "error": str(exc)})
                continue
            cached = TEMPO_RESULT_CACHE.get(cache_key)
            if cached is None:
                missing.append(track_path)
            else:
                cached_results[track_path] = dict(cached)
        computed = {
            str(item.get("track_path")): item
            for item in analyze_tempo_tracks(model, missing)
        }
        for track_path in missing:
            item = computed.get(track_path)
            if item and "error" not in item:
                TEMPO_RESULT_CACHE[build_tempo_cache_key(model_path, track_path)] = dict(item)
        ordered_results = []
        for track_path in resolved_track_paths:
            ordered_results.append(
                cached_results.get(track_path)
                or computed.get(track_path)
                or {"track_path": track_path, "error": "missing_result"}
            )
        self._send_json(
            HTTPStatus.OK,
            {
                "tf_physical_gpu_count": tf_physical_gpu_count,
                "tf_logical_gpu_count": tf_logical_gpu_count,
                "results": ordered_results,
            },
        )

    def _handle_analyze_semantics(self, payload: dict[str, Any]) -> None:
        model_root = str(payload.get("model_root") or "").strip()
        track_paths = payload.get("tracks") or []
        device = str(payload.get("device") or "auto")
        family_policy = str(payload.get("family_policy") or "best_per_task")
        auxiliary_features_by_track = payload.get("auxiliary_features_by_track") or {}
        if not model_root or not isinstance(track_paths, list) or not track_paths:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "model_root and tracks[] are required."})
            return
        try:
            state = load_bundle_state(model_root, family_policy)
            validate_requested_device(state, device)
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        request_cfg = {
            "default_excerpt_seconds": float(payload.get("default_excerpt_seconds") or 60.0),
            "multisample_excerpt_seconds": float(payload.get("multisample_excerpt_seconds") or 30.0),
            "mismatch_threshold": float(payload.get("mismatch_threshold") or 0.22),
            "confidence_threshold": float(payload.get("confidence_threshold") or 0.58),
            "structure_rms_cv_threshold": float(payload.get("structure_rms_cv_threshold") or 0.45),
            "outlier_zscore_threshold": float(payload.get("outlier_zscore_threshold") or 1.35),
        }
        results = analyze_semantic_tracks(model_root, family_policy, [str(track_path) for track_path in track_paths], auxiliary_features_by_track, request_cfg)
        self._send_json(
            HTTPStatus.OK,
            {
                "results": results,
                "tf_physical_gpu_count": state["tf_physical_gpu_count"],
                "tf_logical_gpu_count": state["tf_logical_gpu_count"],
            },
        )


def main() -> int:
    port = int(os.getenv("ESSENTIA_SEMANTIC_SERVICE_PORT", "47833"))
    default_model_root = os.getenv("CUEMATE_ESSENTIA_SEMANTIC_MODEL_ROOT")
    default_family_policy = os.getenv("CUEMATE_ESSENTIA_SEMANTIC_MODEL_FAMILY_POLICY", "best_per_task")
    default_tempo_model = os.getenv("CUEMATE_TEMPOCNN_DEFAULT_MODEL")
    if default_model_root:
        try:
            load_thread_bundle(default_model_root, default_family_policy)
        except Exception:
            pass
    if default_tempo_model:
        try:
            get_tempo_model(default_tempo_model)
        except Exception:
            pass
    server = ThreadingHTTPServer(("0.0.0.0", port), SharedTensorflowHandler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
