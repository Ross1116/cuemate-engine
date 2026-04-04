from __future__ import annotations

from types import SimpleNamespace

from cuemate_analysis.cli import print_backend_diagnostics


def test_print_backend_diagnostics_summarizes_cache_and_gpu(capsys) -> None:
    estimates = [
        SimpleNamespace(
            available=True,
            elapsed_ms=120.0,
            details={"tf_physical_gpu_count": 1, "tf_logical_gpu_count": 1},
            notes=["Persistent inference cache hit.", "Warm Docker service path."],
        ),
        SimpleNamespace(
            available=False,
            elapsed_ms=240.0,
            details={"runner_device": "cpu"},
            notes=["service_failed"],
        ),
    ]

    print_backend_diagnostics("Essentia semantics", estimates, requested_device="auto")
    captured = capsys.readouterr().out

    assert "Essentia semantics diagnostics:" in captured
    assert "- requested_device: auto" in captured
    assert "- results: 1/2 available" in captured
    assert "- persistent_cache_hits: 1/2" in captured
    assert "- avg_elapsed_ms: 180.0" in captured
    assert "- runner_device(s): cpu, cuda" in captured
    assert "- tensorflow_gpus=1 physical / 1 logical" in captured
