import json
import sqlite3
from pathlib import Path

from cuemate_analysis.cli import main
from cuemate_analysis.config import (
    AnalysisSettings,
    RuntimeSettings,
    ScoringSettings,
    SemanticCalibrationSettings,
    ThresholdSettings,
    WeightAdaptationSettings,
)
from cuemate_analysis.relative_context import (
    PlaylistStatsPreview,
    RelativeTrackInput,
    RelativePlaylistPreview,
    RelativeTrackPreview,
    _preview_to_db_rows,
    assign_intensity_band,
    assign_role_hints,
    compute_intensity_membership,
    compute_relative_playlist_preview,
    robust_scale,
    row_to_relative_track_input,
)


def _insert_absolute_track(
    connection: sqlite3.Connection,
    *,
    playlist_id: str,
    track_id: str,
    position: int,
    energy: float,
    bass: float,
    drums: float,
    harmonic: float,
    groove: float,
    bpm: float,
    key: str,
    vocals: float | None = None,
    vocals_confidence: float | None = None,
) -> None:
    now = "2026-04-04T00:00:00Z"
    connection.execute(
        """
        INSERT INTO tracks (
          id, file_path, file_hash, title, artist, genre, duration_seconds, import_source, imported_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            track_id,
            f"D:/Music/{track_id}.wav",
            f"hash-{track_id}",
            f"Track {position}",
            "Artist",
            None,
            180.0,
            "local_files",
            now,
            now,
        ),
    )
    connection.execute(
        "INSERT INTO playlist_tracks (playlist_id, track_id, position, added_at) VALUES (?, ?, ?, ?)",
        (playlist_id, track_id, position, now),
    )
    connection.execute(
        """
        INSERT INTO track_features_abs (
          track_id, source_file_hash, bpm, bpm_confidence, bpm_source, time_signature,
          time_signature_confidence, key, key_number, key_letter, key_confidence, key_source,
          key_imported, key_tagged, key_agreement, energy_abs, energy_heuristic_abs, energy_sustained, energy_peak,
          loudness_lufs, loudness_norm, bass_abs, drums_abs, harmonic_abs, groove_abs,
          vocals_abs, vocals_confidence, analysis_mode, analyzed_at, analysis_signature,
          config_signature, scoring_contract_id_at_analysis
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            track_id,
            f"hash-{track_id}",
            bpm,
            0.9,
            "tempocnn",
            "4/4",
            0.7,
            key,
            8,
            "A",
            0.8,
            "musicalkeycnn",
            None,
            key,
            1,
            energy,
            energy,
            energy,
            energy,
            -8.0,
            0.7,
            bass,
            drums,
            harmonic,
            groove,
            vocals,
            vocals_confidence,
            "full",
            now,
            "m1-test",
            "default",
            None,
        ),
    )


def _create_relative_test_db(tmp_path: Path, playlist_name: str, track_count: int) -> Path:
    database_path = tmp_path / "relative.db"
    schema_path = Path(__file__).resolve().parents[2] / "db" / "schema.sql"
    connection = sqlite3.connect(database_path)
    connection.executescript(schema_path.read_text(encoding="utf-8"))
    now = "2026-04-04T00:00:00Z"
    playlist_id = "plt_test"
    connection.execute(
        "INSERT INTO playlists (id, name, track_count, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (playlist_id, playlist_name, track_count, now, now),
    )
    for index in range(1, track_count + 1):
        vocals = 0.85 if index % 3 == 0 else None
        vocals_confidence = 0.9 if vocals is not None else None
        _insert_absolute_track(
            connection,
            playlist_id=playlist_id,
            track_id=f"trk_{index:02d}",
            position=index,
            energy=0.20 + (index * 0.03),
            bass=0.25 + (index * 0.02),
            drums=0.30 + (index * 0.02),
            harmonic=0.35 + (index * 0.015),
            groove=0.28 + (index * 0.02),
            bpm=120.0 + index,
            key=f"{((index - 1) % 12) + 1}A",
            vocals=vocals,
            vocals_confidence=vocals_confidence,
        )
    connection.commit()
    connection.close()
    return database_path


def _load_relative_rows(database_path: Path, playlist_name: str):
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT
          p.id AS playlist_id,
          p.name AS playlist_name,
          pt.position,
          t.id AS track_id,
          t.file_path,
          t.title,
          t.artist,
          f.track_id IS NOT NULL AS has_absolute_analysis,
          f.bpm,
          f.key,
          f.energy_abs,
          f.energy_heuristic_abs,
          f.energy_essentia_fused,
          f.energy_essentia_bucket,
          f.bass_abs,
          f.drums_abs,
          f.harmonic_abs,
          f.groove_abs,
          f.vocals_abs,
          f.vocals_confidence,
          f.analyzed_at,
          f.analysis_signature,
          f.config_signature
        FROM playlists p
        JOIN playlist_tracks pt ON pt.playlist_id = p.id
        JOIN tracks t ON t.id = pt.track_id
        LEFT JOIN track_features_abs f ON f.track_id = t.id
        WHERE p.name = ?
        ORDER BY pt.position ASC
        """,
        (playlist_name,),
    ).fetchall()
    connection.close()
    return rows


def _settings_for_database(database_path: Path) -> RuntimeSettings:
    repo_root = database_path.parent.resolve()
    env_path = repo_root / ".env.test"
    config_path = repo_root / "config.test.json"

    analysis = AnalysisSettings(
        sample_rate=22050,
        mono=True,
        key_backend="musicalkeycnn",
        key_model_path=None,
        key_device="auto",
        key_policy="full_track",
        parallel_workers=2,
        max_workers_auto=False,
        per_track_timeout_seconds=120,
        model_preload=False,
        fast_pass_enabled=True,
        analysis_signature_seed="test-analysis-seed",
        energy_source_default="canonical",
        essentia_semantics_enabled=True,
        essentia_semantic_image="cuemate-essentia-semantics:test",
        essentia_semantic_device="auto",
        essentia_semantic_model_family_policy="best_per_task",
        essentia_semantic_model_root="python/models/essentia_semantics",
        essentia_semantic_default_excerpt_seconds=60.0,
        essentia_semantic_multisample_excerpt_seconds=30.0,
        essentia_semantic_trigger_mismatch_threshold=0.22,
        essentia_semantic_trigger_confidence_threshold=0.58,
        essentia_semantic_trigger_structure_rms_cv=0.45,
        essentia_semantic_trigger_outlier_zscore=1.35,
        full_chunk_size=4,
        tempo_chunk_size=8,
        key_chunk_size=8,
        essentia_chunk_size=4,
        dsp_workers=2,
    )

    thresholds = ThresholdSettings(
        small_playlist_limit=12,
        min_playlist_for_relative=5,
    )

    scoring = ScoringSettings(
        static_weights={
            "target_energy": 0.22,
            "transition_support": 0.18,
            "bass_transition": 0.15,
            "vocal_transition": 0.13,
            "harmonic": 0.12,
            "tempo": 0.10,
            "history_fit": 0.06,
            "rhythmic_continuity": 0.04,
        },
        weight_floors={
            "target_energy": 0.08,
            "transition_support": 0.05,
            "bass_transition": 0.04,
            "vocal_transition": 0.03,
            "harmonic": 0.04,
            "tempo": 0.03,
            "history_fit": 0.03,
            "rhythmic_continuity": 0.02,
        },
    )

    weight_adaptation = WeightAdaptationSettings(
        mode="auto",
        adaptation_strength=0.7,
    )

    semantic_calibration = SemanticCalibrationSettings(
        calibration_version="identity",
        heads={},
    )

    return RuntimeSettings(
        repo_root=repo_root,
        env_path=env_path,
        config_path=config_path,
        database_path=database_path,
        database_url=f"sqlite:{database_path.as_posix()}",
        config_signature="test-config",
        analysis_signature="test-analysis",
        analysis=analysis,
        thresholds=thresholds,
        scoring=scoring,
        weight_adaptation=weight_adaptation,
        semantic_calibration=semantic_calibration,
    )


def test_robust_scale_handles_normal_and_zero_spread() -> None:
    assert robust_scale(0.5, 0.0, 1.0) == 0.5
    assert robust_scale(0.5, 0.5, 0.5) == 0.5


def test_intensity_membership_is_bounded() -> None:
    membership = compute_intensity_membership(0.6, None, 0.7)
    assert set(membership) == {"low", "groove", "drive", "peak"}
    assert all(0.0 <= value <= 1.0 for value in membership.values())


def test_assign_intensity_band_in_batch_mode_ignores_hysteresis_defaults() -> None:
    assert assign_intensity_band(0.82, None, 0.9, previous_band=None, previous_energy_rel=None) == "Peak"


def test_role_hints_do_not_treat_unknown_vocals_as_instrumental() -> None:
    roles = assign_role_hints(0.75, None, 0.95, 0.85, vocals_confidence=None)
    assert "bass_driver" not in roles
    assert "pressure_builder" in roles


def test_role_hints_include_vocal_feature_when_vocals_are_known() -> None:
    roles = assign_role_hints(0.55, 0.9, 0.4, 0.5, vocals_confidence=0.9)
    assert "vocal_feature" in roles


def test_role_hints_use_neutral_mid_energy_label() -> None:
    roles = assign_role_hints(0.50, None, 0.4, 0.5, vocals_confidence=None)
    assert "steady_energy" in roles
    assert "safe_continuation" not in roles


def test_relative_preview_marks_small_playlists_as_insufficient(tmp_path: Path) -> None:
    database_path = _create_relative_test_db(tmp_path, "Small", 4)
    settings = _settings_for_database(database_path)
    preview = compute_relative_playlist_preview(
        [row_to_relative_track_input(row) for row in _load_relative_rows(database_path, "Small")],
        settings,
        playlist_name="Small",
        is_limited=False,
    )
    assert preview.playlist_stats.status == "insufficient_tracks"
    assert preview.tracks == []


def test_relative_preview_computes_rows_but_skips_weights_for_small_eligible_playlist(tmp_path: Path) -> None:
    database_path = _create_relative_test_db(tmp_path, "Medium", 8)
    settings = _settings_for_database(database_path)
    preview = compute_relative_playlist_preview(
        [row_to_relative_track_input(row) for row in _load_relative_rows(database_path, "Medium")],
        settings,
        playlist_name="Medium",
        is_limited=False,
    )
    assert preview.playlist_stats.status == "relative_only"
    assert len(preview.tracks) == 8
    assert preview.playlist_stats.adapted_weights is None


def test_relative_preview_small_playlist_spreads_match_weight_profile_logic(tmp_path: Path) -> None:
    database_path = tmp_path / "small_spreads.db"
    settings = _settings_for_database(database_path)
    rows = [
        RelativeTrackInput(
            playlist_id="plt_test",
            playlist_name="SmallSpreads",
            track_id=f"trk_{index}",
            position=index,
            file_path=f"D:/Music/trk_{index}.wav",
            title=f"Track {index}",
            artist="Artist",
            has_absolute_analysis=True,
            bpm=120.0 + index,
            key="8A",
            key_confidence=0.85,
            key_source="musicalkeycnn",
            key_agreement=1,
            energy_abs=0.20 + (index * 0.05),
            energy_heuristic_abs=0.20 + (index * 0.05),
            energy_essentia_fused=None,
            energy_essentia_bucket=None,
            bass_abs=0.20 + (index * 0.03),
            drums_abs=0.40,
            harmonic_abs=0.35,
            groove_abs=0.30 + (index * 0.02),
            vocals_abs=None,
            vocals_confidence=None,
            analyzed_at="2026-04-04T00:00:00Z",
            analysis_signature="m1-test",
            config_signature="default",
        )
        for index in range(1, 6)
    ]
    preview = compute_relative_playlist_preview(
        rows,
        settings,
        playlist_name="SmallSpreads",
        is_limited=False,
    )
    assert preview.playlist_stats.drums_spread == 0.0
    assert preview.playlist_stats.harmonic_spread == 0.0


def test_relative_preview_computes_weights_for_large_playlist(tmp_path: Path) -> None:
    database_path = _create_relative_test_db(tmp_path, "Large", 12)
    settings = _settings_for_database(database_path)
    preview = compute_relative_playlist_preview(
        [row_to_relative_track_input(row) for row in _load_relative_rows(database_path, "Large")],
        settings,
        playlist_name="Large",
        is_limited=False,
    )
    assert preview.playlist_stats.status == "ok"
    assert preview.playlist_stats.adapted_weights is not None
    assert "adaptation_strength blend" in " ".join(preview.playlist_stats.weight_adaptation_notes)


def test_relative_preview_is_deterministic(tmp_path: Path) -> None:
    database_path = _create_relative_test_db(tmp_path, "Stable", 12)
    settings = _settings_for_database(database_path)
    first = compute_relative_playlist_preview(
        [row_to_relative_track_input(row) for row in _load_relative_rows(database_path, "Stable")],
        settings,
        playlist_name="Stable",
        is_limited=False,
    ).to_payload()
    second = compute_relative_playlist_preview(
        [row_to_relative_track_input(row) for row in _load_relative_rows(database_path, "Stable")],
        settings,
        playlist_name="Stable",
        is_limited=False,
    ).to_payload()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_preview_to_db_rows_coalesces_missing_drums_and_groove() -> None:
    preview = RelativePlaylistPreview(
        playlist="Persist",
        playlist_id="plt_test",
        is_limited=False,
        limited_track_count=1,
        playlist_stats=PlaylistStatsPreview(
            playlist_id="plt_test",
            track_count_total=1,
            track_count_analyzed=1,
            eligible_track_count=1,
            avg_harmonic=0.4,
            key_diversity=0.1,
            bpm_range=2.0,
            energy_spread=0.2,
            bass_spread=0.1,
            drums_spread=0.0,
            vocals_spread=0.3,
            harmonic_spread=0.0,
            groove_spread=0.2,
            adapted_weights=None,
            adaptation_strength=None,
            weight_adaptation_notes=[],
            status="relative_only",
            energy_source_used="canonical",
            relative_signature="sig",
        ),
        tracks=[
            RelativeTrackPreview(
                track_id="trk_1",
                playlist_id="plt_test",
                position=1,
                title="Track 1",
                artist="Artist",
                file_path="D:/Music/trk_1.wav",
                energy_source_used="canonical",
                energy_rel=0.5,
                bass_rel=0.5,
                drums_rel=None,
                vocals_rel=None,
                groove_rel=None,
                energy_spread=0.2,
                bass_spread=0.1,
                drums_spread=0.0,
                vocals_spread=0.3,
                groove_spread=0.2,
                intensity_band="Groove",
                intensity_membership={"low": 0.1, "groove": 0.8, "drive": 0.2, "peak": 0.0},
                role_hints=["steady_energy"],
                valid_as_of_track_count=1,
                analyzed_at=None,
                analysis_signature="m1-test",
                config_signature="default",
            )
        ],
    )
    track_rows, _stats_row = _preview_to_db_rows(preview, "plt_test", "2026-04-04T00:00:00Z")
    assert track_rows[0]["drums_rel"] == 0.5
    assert track_rows[0]["groove_rel"] == 0.5


def test_relative_preview_can_use_essentia_fused_with_heuristic_fallback(tmp_path: Path) -> None:
    database_path = tmp_path / "essentia.db"
    settings = _settings_for_database(database_path)
    rows = [
        RelativeTrackInput(
            playlist_id="plt_test",
            playlist_name="Essentia",
            track_id=f"trk_{index}",
            position=index,
            file_path=f"D:/Music/trk_{index}.wav",
            title=f"Track {index}",
            artist="Artist",
            has_absolute_analysis=True,
            bpm=120.0 + index,
            key="8A",
            key_confidence=0.85,
            key_source="musicalkeycnn",
            key_agreement=1,
            energy_abs=0.20 + (index * 0.05),
            energy_heuristic_abs=0.20 + (index * 0.05),
            energy_essentia_fused=None if index == 1 else 0.15 + (index * 0.07),
            energy_essentia_bucket=None,
            bass_abs=0.20 + (index * 0.05),
            drums_abs=0.25 + (index * 0.05),
            harmonic_abs=0.30 + (index * 0.04),
            groove_abs=0.28 + (index * 0.03),
            vocals_abs=None,
            vocals_confidence=None,
            analyzed_at="2026-04-04T00:00:00Z",
            analysis_signature="m1-test",
            config_signature="default",
        )
        for index in range(1, 6)
    ]
    preview = compute_relative_playlist_preview(
        rows,
        settings,
        playlist_name="Essentia",
        is_limited=False,
        energy_source="canonical",
    )
    assert preview.tracks[0].energy_source_used == "canonical"
    assert all(track.energy_source_used == "canonical" for track in preview.tracks)


def test_cli_analyze_relative_playlist_json_and_csv(tmp_path: Path, monkeypatch, capsys) -> None:
    database_path = _create_relative_test_db(tmp_path, "CLI", 12)
    monkeypatch.setattr("cuemate_analysis.cli.load_runtime_settings", lambda: _settings_for_database(database_path))
    output_path = tmp_path / "relative.csv"

    assert main(["analyze-relative-playlist", "--playlist", "CLI", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["playlist"] == "CLI"
    assert payload["playlist_stats"]["status"] == "ok"
    assert len(payload["tracks"]) == 12

    assert main(["analyze-relative-playlist", "--playlist", "CLI", "--output", str(output_path)]) == 0
    csv_text = output_path.read_text(encoding="utf-8")
    assert "intensity_membership" in csv_text
    assert "role_hints" in csv_text


def test_refresh_relative_playlist_persists_canonical_tables(tmp_path: Path, monkeypatch, capsys) -> None:
    database_path = _create_relative_test_db(tmp_path, "Persisted", 12)
    monkeypatch.setattr("cuemate_analysis.cli.load_runtime_settings", lambda: _settings_for_database(database_path))

    assert main(["refresh-relative-playlist", "--playlist", "Persisted", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["playlist"] == "Persisted"
    assert len(payload["tracks"]) == 12

    connection = sqlite3.connect(database_path)
    track_count = connection.execute("SELECT COUNT(*) FROM track_features_rel WHERE playlist_id = 'plt_test'").fetchone()[0]
    stats_row = connection.execute(
        "SELECT is_stale, energy_source_used, status FROM playlist_stats WHERE playlist_id = 'plt_test'"
    ).fetchone()
    connection.close()

    assert track_count == 12
    assert stats_row == (0, "canonical", "ok")


def test_canonical_relative_auto_refreshes_when_missing(tmp_path: Path, monkeypatch, capsys) -> None:
    database_path = _create_relative_test_db(tmp_path, "Auto", 12)
    monkeypatch.setattr("cuemate_analysis.cli.load_runtime_settings", lambda: _settings_for_database(database_path))

    assert main(["analyze-relative-playlist", "--playlist", "Auto", "--energy-source", "canonical", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["playlist_stats"]["status"] == "ok"

    connection = sqlite3.connect(database_path)
    persisted_count = connection.execute("SELECT COUNT(*) FROM track_features_rel WHERE playlist_id = 'plt_test'").fetchone()[0]
    stats_row = connection.execute(
        "SELECT is_stale FROM playlist_stats WHERE playlist_id = 'plt_test'"
    ).fetchone()
    connection.close()

    assert persisted_count == 12
    assert stats_row == (0,)


def test_canonical_relative_reads_persisted_rows_when_current(tmp_path: Path, monkeypatch, capsys) -> None:
    database_path = _create_relative_test_db(tmp_path, "Current", 12)
    monkeypatch.setattr("cuemate_analysis.cli.load_runtime_settings", lambda: _settings_for_database(database_path))

    assert main(["refresh-relative-playlist", "--playlist", "Current", "--json"]) == 0
    capsys.readouterr()

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError("canonical path should have read persisted relative rows")

    monkeypatch.setattr("cuemate_analysis.cli.compute_relative_playlist_preview", fail_if_recomputed)

    assert main(["analyze-relative-playlist", "--playlist", "Current", "--energy-source", "canonical", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["playlist"] == "Current"
    assert len(payload["tracks"]) == 12
