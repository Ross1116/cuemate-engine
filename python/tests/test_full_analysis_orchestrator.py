from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cuemate_analysis.cli import handle_analyze_essentia_playlist, handle_analyze_playlist
from cuemate_analysis.essentia_semantic_backend import EssentiaSemanticEstimate
from cuemate_analysis.key_backend import KeyEstimate
from cuemate_analysis.models import AnalysisResult, ImportedTrack
from cuemate_analysis.tempo_backend import TempoEstimate


def _row(position: int, *, track_id: str, file_path: str) -> dict[str, object]:
    return {
        "playlist_id": "pl_1",
        "track_id": track_id,
        "file_path": file_path,
        "title": f"Track {position}",
        "artist": "Artist",
        "genre": None,
        "import_source": "local_files",
        "imported_bpm": None,
        "imported_key": None,
        "position": position,
        "source_file_hash": f"hash-{position}",
        "analysis_signature": None,
        "analysis_mode": None,
        "config_signature": None,
        "essentia_semantic_signature": None,
        "energy_essentia_fused": None,
    }


def _settings() -> SimpleNamespace:
    analysis = SimpleNamespace(
        sample_rate=22050,
        mono=True,
        key_backend="musicalkeycnn",
        key_model_path="python/models/musicalkeycnn/keynet.pt",
        key_device="auto",
        key_policy="full_track",
        fast_pass_enabled=True,
        analysis_signature_seed="test-seed",
        full_chunk_size=2,
        tempo_chunk_size=2,
        key_chunk_size=2,
        essentia_chunk_size=2,
        dsp_workers=2,
        essentia_semantics_enabled=True,
        essentia_semantic_device="auto",
        essentia_semantic_image="cuemate-essentia-semantics:local",
        essentia_semantic_model_family_policy="best_per_task",
        essentia_semantic_model_root="python/models/essentia_semantics",
        essentia_semantic_default_excerpt_seconds=60.0,
        essentia_semantic_multisample_excerpt_seconds=30.0,
        essentia_semantic_trigger_mismatch_threshold=0.22,
        essentia_semantic_trigger_confidence_threshold=0.58,
        essentia_semantic_trigger_structure_rms_cv=0.45,
        essentia_semantic_trigger_outlier_zscore=1.35,
    )
    return SimpleNamespace(
        analysis=analysis,
        database_path=Path("ignored.db"),
        config_signature="cfg",
        analysis_signature="sig",
    )


def _track_for_row(row: dict[str, object]) -> ImportedTrack:
    return ImportedTrack(
        id=str(row["track_id"]),
        file_path=Path(str(row["file_path"])),
        file_hash=f"{row['track_id']}-hash",
        title=str(row["title"]),
        artist=str(row["artist"]),
        genre=None,
        duration_seconds=180.0,
        bpm_imported=None,
        bpm_tag=None,
        key_imported=None,
        key_tag=None,
    )


class _FakeDatabase:
    def __init__(self, rows):
        self.rows = rows
        self.completed: list[dict[str, object]] = []
        self.failed: list[dict[str, object]] = []
        self.saved_results: list[AnalysisResult] = []
        self.saved_fast_results: list[object] = []
        self._next_job_id = 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def get_playlist_tracks(self, playlist_name: str):
        return self.rows

    def create_analysis_job(self, **kwargs):
        job_id = self._next_job_id
        self._next_job_id += 1
        return job_id

    def create_analysis_job_with_kind(self, **kwargs):
        job_id = self._next_job_id
        self._next_job_id += 1
        return job_id

    def mark_analysis_job_started(self, job_id: int, started_at: str) -> None:
        return None

    def upsert_track(self, track: ImportedTrack, timestamp: str) -> None:
        return None

    def upsert_track_features(self, result: AnalysisResult) -> None:
        self.saved_results.append(result)

    def upsert_track_fast_features(self, result) -> None:
        self.saved_fast_results.append(result)

    def get_table_columns(self, table_name: str):
        if table_name == "track_features_abs":
            return {"energy_heuristic_abs"}
        if table_name == "track_features_fast":
            return {"track_id"}
        return set()

    def mark_analysis_job_completed(
        self,
        job_id: int,
        duration_seconds: float,
        timing_breakdown: dict[str, object],
        completed_at: str,
    ) -> None:
        self.completed.append(
            {
                "job_id": job_id,
                "duration_seconds": duration_seconds,
                "timing_breakdown": timing_breakdown,
            }
        )

    def mark_analysis_job_failed(
        self,
        job_id: int,
        error_message: str,
        duration_seconds: float,
        completed_at: str,
    ) -> None:
        self.failed.append(
            {
                "job_id": job_id,
                "error_message": error_message,
            }
        )


def test_handle_analyze_essentia_playlist_chunks_text_output(monkeypatch, capsys) -> None:
    rows = [
        _row(1, track_id="trk_1", file_path="D:/Music/one.wav"),
        _row(2, track_id="trk_2", file_path="D:/Music/two.wav"),
        _row(3, track_id="trk_3", file_path="D:/Music/three.wav"),
    ]
    fake_db = _FakeDatabase(rows)
    settings = _settings()
    chunk_sizes: list[int] = []

    def fake_prefetch(chunk_rows, runtime_settings):
        chunk_sizes.append(len(chunk_rows))
        estimates = {}
        for row in chunk_rows:
            path = Path(str(row["file_path"])).resolve()
            estimates[path] = EssentiaSemanticEstimate(
                backend="essentia_semantics",
                danceability_abs=0.8,
                arousal_abs=0.7,
                valence_abs=0.5,
                mood_aggressive_abs=0.4,
                mood_party_abs=0.6,
                mood_relaxed_abs=0.2,
                semantic_confidence=0.7,
                energy_essentia_fused=0.65,
                energy_essentia_bucket="drive",
                elapsed_ms=25.0,
                details={"runner_device": "cuda", "tf_physical_gpu_count": 1, "tf_logical_gpu_count": 1},
                notes=[],
                available=True,
            )
        return estimates

    monkeypatch.setattr("cuemate_analysis.cli.load_runtime_settings", lambda: settings)
    monkeypatch.setattr("cuemate_analysis.cli.Database", lambda _: fake_db)
    monkeypatch.setattr("cuemate_analysis.cli.prefetch_essentia_semantic_estimates", fake_prefetch)

    exit_code = handle_analyze_essentia_playlist(
        SimpleNamespace(playlist="EDM", limit=None, json=False, output=None)
    )
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert chunk_sizes == [2, 1]
    assert "[1/3] Track 1" in captured
    assert "[3/3] Track 3" in captured
    assert "Essentia semantics diagnostics:" in captured


def test_handle_analyze_playlist_marks_degraded_rows_with_lane_status(monkeypatch, capsys) -> None:
    rows = [
        _row(1, track_id="trk_1", file_path="D:/Music/one.wav"),
        _row(2, track_id="trk_2", file_path="D:/Music/two.wav"),
    ]
    fake_db = _FakeDatabase(rows)
    settings = _settings()

    tempo_results = {
        Path("D:/Music/one.wav").resolve(): TempoEstimate(
            backend="tempocnn",
            bpm=128.0,
            confidence=0.9,
            elapsed_ms=10.0,
            details={"runner_device": "cuda"},
            notes=[],
            available=True,
        ),
        Path("D:/Music/two.wav").resolve(): TempoEstimate(
            backend="tempocnn",
            bpm=None,
            confidence=None,
            elapsed_ms=10.0,
            details={"runner_device": "cuda"},
            notes=["service_failed"],
            available=False,
        ),
    }
    key_results = {
        Path("D:/Music/one.wav").resolve(): KeyEstimate(
            backend="musicalkeycnn",
            key="8A",
            key_number=8,
            key_letter="A",
            confidence=0.8,
            elapsed_ms=12.0,
            details={"runner_device": "cuda"},
            notes=[],
            available=True,
        ),
        Path("D:/Music/two.wav").resolve(): KeyEstimate(
            backend="musicalkeycnn",
            key="9A",
            key_number=9,
            key_letter="A",
            confidence=0.75,
            elapsed_ms=12.0,
            details={"runner_device": "cuda"},
            notes=["Persistent inference cache hit."],
            available=True,
        ),
    }
    essentia_results = {
        Path("D:/Music/one.wav").resolve(): EssentiaSemanticEstimate(
            backend="essentia_semantics",
            danceability_abs=0.8,
            arousal_abs=0.7,
            valence_abs=0.5,
            mood_aggressive_abs=0.4,
            mood_party_abs=0.6,
            mood_relaxed_abs=0.2,
            semantic_confidence=0.7,
            energy_essentia_fused=0.65,
            energy_essentia_bucket="drive",
            elapsed_ms=20.0,
            details={"runner_device": "cuda"},
            notes=[],
            available=True,
        ),
        Path("D:/Music/two.wav").resolve(): EssentiaSemanticEstimate(
            backend="essentia_semantics",
            danceability_abs=None,
            arousal_abs=None,
            valence_abs=None,
            mood_aggressive_abs=None,
            mood_party_abs=None,
            mood_relaxed_abs=None,
            semantic_confidence=None,
            energy_essentia_fused=None,
            energy_essentia_bucket=None,
            elapsed_ms=20.0,
            details={"runner_device": "cuda"},
            notes=["service_failed"],
            available=False,
        ),
    }
    dsp_results = {
        Path("D:/Music/one.wav").resolve(): SimpleNamespace(
            available=True,
            elapsed_ms=8.0,
            details={"runner_device": "cpu"},
            notes=[],
        ),
        Path("D:/Music/two.wav").resolve(): SimpleNamespace(
            available=True,
            elapsed_ms=8.0,
            details={"runner_device": "cpu"},
            notes=[],
        ),
    }

    def fake_build_analysis_result(track, settings_obj, analysis_mode, **kwargs):
        suffix = "fallback" if not kwargs["prefetched_tempocnn_estimate"].available else "tempocnn"
        return AnalysisResult(
            track_id=track.id,
            source_file_hash=track.file_hash,
            bpm=128.0,
            bpm_confidence=0.9,
            bpm_source=suffix,
            time_signature="4/4",
            time_signature_confidence=0.8,
            key="8A",
            key_number=8,
            key_letter="A",
            key_confidence=0.8,
            key_source="musicalkeycnn",
            key_imported=None,
            key_tagged=None,
            key_agreement=None,
            energy_abs=0.6,
            energy_heuristic_abs=0.6,
            energy_sustained=0.5,
            energy_peak=0.7,
            danceability_abs=None,
            arousal_abs=None,
            valence_abs=None,
            mood_aggressive_abs=None,
            mood_party_abs=None,
            mood_relaxed_abs=None,
            energy_essentia_fused=None,
            energy_essentia_bucket=None,
            essentia_semantic_signature=None,
            essentia_semantic_source=None,
            essentia_semantic_inferred_at=None,
            loudness_lufs=-8.0,
            loudness_norm=0.7,
            bass_abs=0.3,
            drums_abs=0.6,
            harmonic_abs=0.5,
            groove_abs=0.4,
            vocals_abs=None,
            vocals_confidence=None,
            analysis_mode=analysis_mode,
            analyzed_at="2026-04-04T00:00:00Z",
            analysis_signature="sig",
            config_signature="cfg",
        )

    monkeypatch.setattr("cuemate_analysis.cli.load_runtime_settings", lambda: settings)
    monkeypatch.setattr("cuemate_analysis.cli.Database", lambda _: fake_db)
    monkeypatch.setattr("cuemate_analysis.cli.track_from_playlist_row", _track_for_row)
    monkeypatch.setattr("cuemate_analysis.cli.run_dsp_lane", lambda prepared, *_: dsp_results)
    monkeypatch.setattr("cuemate_analysis.cli.run_tempo_lane", lambda prepared, *_: tempo_results)
    monkeypatch.setattr("cuemate_analysis.cli.run_key_lane", lambda prepared, *_: key_results)
    monkeypatch.setattr("cuemate_analysis.cli.run_essentia_lane", lambda prepared, *_, **__: essentia_results)
    monkeypatch.setattr("cuemate_analysis.cli.build_analysis_result", fake_build_analysis_result)

    exit_code = handle_analyze_playlist(
        SimpleNamespace(playlist="EDM", analysis_mode="full", force=True)
    )
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert len(fake_db.saved_results) == 2
    assert fake_db.failed
    assert "degraded: missing tempocnn, essentia_semantics" in captured
    enrichment_completions = [
        item for item in fake_db.completed if "degraded" in item["timing_breakdown"]
    ]
    assert enrichment_completions[0]["timing_breakdown"]["degraded"] is False
    assert enrichment_completions[1]["timing_breakdown"]["degraded"] is True
    assert enrichment_completions[1]["timing_breakdown"]["missing_lanes"] == ["tempocnn", "essentia_semantics"]
    assert enrichment_completions[1]["timing_breakdown"]["lane_status"]["musicalkeycnn"] == "cached"
