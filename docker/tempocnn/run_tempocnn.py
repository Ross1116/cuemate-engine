from __future__ import annotations

import json
import sys

import numpy as np

import essentia.standard as es


def detect_gpu_counts() -> tuple[int | None, int | None]:
    try:
        import tensorflow as tf
    except Exception:
        return None, None
    physical = len(tf.config.list_physical_devices("GPU"))
    logical = len(tf.config.list_logical_devices("GPU"))
    return physical, logical


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "Usage: python /workspace/docker/tempocnn/run_tempocnn.py <track-path> <model-path>",
            file=sys.stderr,
        )
        return 2

    track_path, model_path = argv[1], argv[2]
    tf_physical_gpu_count, tf_logical_gpu_count = detect_gpu_counts()

    tempo_audio = es.MonoLoader(filename=track_path, sampleRate=11025, resampleQuality=4)()
    global_tempo, local_tempi, local_probs = es.TempoCNN(graphFilename=model_path)(tempo_audio)

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
    print(
        json.dumps(
            {
                "bpm": float(global_tempo),
                "confidence": float(confidence),
                "local_count": int(local_tempi_array.size),
                "tempo_spread": spread,
                "agreement_with_global": agreement,
                "probability_peak": (
                    float(np.max(local_probs_array)) if local_probs_array.size else None
                ),
                "tf_physical_gpu_count": tf_physical_gpu_count,
                "tf_logical_gpu_count": tf_logical_gpu_count,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
