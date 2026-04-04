from __future__ import annotations

from types import SimpleNamespace

from cuemate_analysis.cli import handle_purge_model_cache, print_backend_diagnostics


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


def test_handle_purge_model_cache_preserves_warm_services_by_default(monkeypatch, capsys) -> None:
    settings = SimpleNamespace(database_path="ignored.db")
    calls: list[tuple[str, bool]] = []

    monkeypatch.setattr("cuemate_analysis.cli.load_runtime_settings", lambda: settings)
    monkeypatch.setattr(
        "cuemate_analysis.cli.purge_tempocnn_cache",
        lambda file_paths=None, clear_warm_service=True: calls.append(("tempocnn", clear_warm_service)) or 1,
    )
    monkeypatch.setattr(
        "cuemate_analysis.cli.purge_musicalkeycnn_cache",
        lambda file_paths=None, clear_warm_service=True: calls.append(("musicalkeycnn", clear_warm_service)) or 2,
    )
    monkeypatch.setattr(
        "cuemate_analysis.cli.purge_essentia_semantic_cache",
        lambda file_paths=None, clear_warm_service=True: calls.append(("essentia_semantics", clear_warm_service)) or 3,
    )
    monkeypatch.setattr("cuemate_analysis.cli.resolve_inference_cache_path", lambda: "cache.db")

    exit_code = handle_purge_model_cache(
        SimpleNamespace(backend="all", playlist=None, path=[], clear_warm_services=False)
    )
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert calls == [
        ("tempocnn", False),
        ("musicalkeycnn", False),
        ("essentia_semantics", False),
    ]
    assert "warm service state preserved" in captured
