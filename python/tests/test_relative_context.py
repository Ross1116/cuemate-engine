import json
import sqlite3
from pathlib import Path

from cuemate_analysis.cli import main
from cuemate_analysis.config import load_runtime_settings
from cuemate_analysis.relative_context import (
    RelativeTrackInput,
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
          key_imported, key_tagged, key_agreement, energy_abs, energy_sustained, energy_peak,
          loudness_lufs, loudness_norm, bass_abs, drums_abs, harmonic_abs, groove_abs,
          vocals_abs, vocals_confidence, analysis_mode, analyzed_at, analysis_signature,
          config_signature, scoring_contract_id_at_analysis
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    settings = load_runtime_settings()
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
    settings = load_runtime_settings()
    preview = compute_relative_playlist_preview(
        [row_to_relative_track_input(row) for row in _load_relative_rows(database_path, "Medium")],
        settings,
        playlist_name="Medium",
        is_limited=False,
    )
    assert preview.playlist_stats.status == "relative_only"
    assert len(preview.tracks) == 8
    assert preview.playlist_stats.adapted_weights is None


def test_relative_preview_computes_weights_for_large_playlist(tmp_path: Path) -> None:
    database_path = _create_relative_test_db(tmp_path, "Large", 12)
    settings = load_runtime_settings()
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
    settings = load_runtime_settings()
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


def test_relative_preview_can_use_essentia_fused_with_heuristic_fallback() -> None:
    settings = load_runtime_settings()
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
            energy_abs=0.20 + (index * 0.05),
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
        energy_source="essentia_fused",
    )
    assert preview.tracks[0].energy_source_used == "heuristic"
    assert any(track.energy_source_used == "essentia_fused" for track in preview.tracks[1:])


def test_cli_analyze_relative_playlist_json_and_csv(tmp_path: Path, monkeypatch, capsys) -> None:
    database_path = _create_relative_test_db(tmp_path, "CLI", 12)
    settings = load_runtime_settings()
    monkeypatch.setattr(
        "cuemate_analysis.cli.load_runtime_settings",
        lambda: settings.__class__(
            repo_root=settings.repo_root,
            env_path=settings.env_path,
            config_path=settings.config_path,
            database_path=database_path,
            database_url=f"sqlite:{database_path.as_posix()}",
            config_signature=settings.config_signature,
            analysis_signature=settings.analysis_signature,
            analysis=settings.analysis,
            thresholds=settings.thresholds,
            scoring=settings.scoring,
            weight_adaptation=settings.weight_adaptation,
        ),
    )
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
