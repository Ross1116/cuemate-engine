"""
Phase 2 scoring tests: DB queries + config integration.

Verifies:
- get_scoring_candidates returns expected rows, respects exclusion list
- get_track_scoring_context returns a single track row
- get_playlist_stats_for_scoring decodes adapted_weights JSON
- row_to_scoring_track_context converts rows correctly (str/None/JSON fields)
- build_scoring_config builds the expected flat dict from RuntimeSettings
- ScoringSettings __post_init__ defaults
- Full query → score_candidate pipeline with fixture DB
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from cuemate_analysis.database import Database
from cuemate_analysis.scoring import (
    ScoringTrackContext,
    row_to_scoring_track_context,
    score_candidate,
)


# ---------------------------------------------------------------------------
# Fixture DB helpers
# ---------------------------------------------------------------------------

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "db" / "schema.sql"
_NOW = "2026-04-07T00:00:00Z"
_PLAYLIST_ID = "plt_scoring_test"
_PLAYLIST_NAME = "Scoring Test Playlist"


def _bootstrap_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute(
        "INSERT INTO playlists (id, name, track_count, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (_PLAYLIST_ID, _PLAYLIST_NAME, 0, _NOW, _NOW),
    )
    conn.commit()
    return conn


def _insert_track(
    conn: sqlite3.Connection,
    *,
    track_id: str,
    position: int,
    bpm: float = 128.0,
    key: str = "8A",
    key_confidence: float = 0.80,
    key_source: str = "musicalkeycnn",
    key_agreement: int = 1,
    energy: float = 0.65,
    bass: float = 0.55,
    drums: float = 0.60,
    harmonic: float = 0.40,
    groove: float = 0.50,
    vocals: float | None = 0.20,
    vocals_confidence: float | None = 0.75,
    # rel features (optional — only inserted when energy_rel is not None)
    energy_rel: float | None = None,
    bass_rel: float | None = None,
    drums_rel: float | None = None,
    vocals_rel: float | None = None,
    groove_rel: float | None = None,
    intensity_band: str | None = None,
    role_hints: list[str] | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO tracks (id, file_path, file_hash, title, artist, genre,
          duration_seconds, import_source, imported_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (track_id, f"D:/Music/{track_id}.flac", f"hash-{track_id}",
         f"Track {position}", "Test Artist", None, 210.0, "local_files", _NOW, _NOW),
    )
    conn.execute(
        "INSERT INTO playlist_tracks (playlist_id, track_id, position, added_at) VALUES (?, ?, ?, ?)",
        (_PLAYLIST_ID, track_id, position, _NOW),
    )
    conn.execute(
        """
        INSERT INTO track_features_abs (
          track_id, source_file_hash, bpm, bpm_confidence, bpm_source,
          time_signature, time_signature_confidence,
          key, key_number, key_letter, key_confidence, key_source,
          key_imported, key_tagged, key_agreement,
          energy_abs, energy_heuristic_abs, energy_sustained, energy_peak,
          loudness_lufs, loudness_norm,
          bass_abs, drums_abs, harmonic_abs, groove_abs,
          vocals_abs, vocals_confidence,
          analysis_mode, analyzed_at, analysis_signature,
          config_signature, scoring_contract_id_at_analysis
        ) VALUES (
          ?, ?, ?, ?, ?,
          ?, ?,
          ?, ?, ?, ?, ?,
          ?, ?, ?,
          ?, ?, ?, ?,
          ?, ?,
          ?, ?, ?, ?,
          ?, ?,
          ?, ?, ?,
          ?, ?
        )
        """,
        (
            track_id, f"hash-{track_id}", bpm, 0.92, "tempocnn",
            "4/4", 0.7,
            key, 8, "A", key_confidence, key_source,
            None, key, key_agreement,
            energy, energy, energy * 0.9, energy * 1.1,
            -8.5, 0.72,
            bass, drums, harmonic, groove,
            vocals, vocals_confidence,
            "full", _NOW, "m1-test",
            "default", None,
        ),
    )
    if energy_rel is not None:
        conn.execute(
            """
            INSERT INTO track_features_rel (
              playlist_id, track_id, position,
              energy_rel, bass_rel, drums_rel, vocals_rel, groove_rel,
              energy_spread, bass_spread, drums_spread, vocals_spread, groove_spread,
              intensity_band, intensity_membership, role_hints,
              valid_as_of_track_count,
              relative_signature, analysis_signature, config_signature, refreshed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _PLAYLIST_ID, track_id, position,
                energy_rel,
                bass_rel if bass_rel is not None else 0.5,
                drums_rel if drums_rel is not None else 0.5,
                vocals_rel if vocals_rel is not None else 0.2,
                groove_rel if groove_rel is not None else 0.5,
                0.30, 0.25, 0.25, 0.20, 0.20,
                intensity_band or "Drive",
                json.dumps({"low": 0.1, "groove": 0.4, "drive": 0.8, "peak": 0.2}),
                json.dumps(role_hints or ["safe_continuation"]),
                8,
                "m2-test", "m1-test", "default", _NOW,
            ),
        )


def _insert_playlist_stats(
    conn: sqlite3.Connection,
    *,
    adapted_weights: dict[str, float] | None = None,
) -> None:
    weights_json = json.dumps(adapted_weights) if adapted_weights else None
    conn.execute(
        """
        INSERT INTO playlist_stats (
          playlist_id, track_count_total, track_count_analyzed, eligible_track_count,
          energy_spread, bass_spread, drums_spread, vocals_spread,
          harmonic_spread, groove_spread, avg_harmonic, key_diversity, bpm_range,
          adapted_weights, adaptation_strength, weight_adaptation_notes,
          status, energy_source_used, relative_signature, refreshed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _PLAYLIST_ID, 8, 8, 8,
            0.30, 0.25, 0.25, 0.20,
            0.15, 0.20, 0.42, 0.67, 8.0,
            weights_json, 0.7, json.dumps(["adaptation_strength blend = 0.70"]),
            "ok", "canonical", "m2-test", _NOW,
        ),
    )


@pytest.fixture()
def fixture_db(tmp_path: Path):
    """Return a Database instance populated with 8 tracks (6 with rel features)."""
    db_path = tmp_path / "scoring_test.db"
    conn = _bootstrap_db(db_path)

    tracks = [
        dict(track_id="trk_01", position=1, bpm=128.0, key="8A",
             energy=0.40, bass=0.40, drums=0.50, harmonic=0.35, groove=0.45,
             energy_rel=0.20, bass_rel=0.25, drums_rel=0.40, vocals_rel=0.10, groove_rel=0.30,
             intensity_band="Low", role_hints=["opener"]),
        dict(track_id="trk_02", position=2, bpm=130.0, key="9A",
             energy=0.55, bass=0.55, drums=0.60, harmonic=0.42, groove=0.52,
             energy_rel=0.45, bass_rel=0.50, drums_rel=0.55, vocals_rel=0.25, groove_rel=0.50,
             intensity_band="Groove", role_hints=["safe_continuation"]),
        dict(track_id="trk_03", position=3, bpm=131.0, key="9B",
             energy=0.65, bass=0.62, drums=0.68, harmonic=0.48, groove=0.58,
             energy_rel=0.55, bass_rel=0.60, drums_rel=0.65, vocals_rel=0.30, groove_rel=0.60,
             intensity_band="Drive", role_hints=["safe_continuation"]),
        dict(track_id="trk_04", position=4, bpm=132.0, key="10A",
             energy=0.72, bass=0.68, drums=0.74, harmonic=0.52, groove=0.64,
             energy_rel=0.65, bass_rel=0.68, drums_rel=0.72, vocals_rel=0.35, groove_rel=0.68,
             intensity_band="Drive", role_hints=["pressure_builder"]),
        dict(track_id="trk_05", position=5, bpm=133.0, key="11A",
             energy=0.80, bass=0.75, drums=0.82, harmonic=0.55, groove=0.72,
             energy_rel=0.78, bass_rel=0.77, drums_rel=0.80, vocals_rel=0.15, groove_rel=0.75,
             intensity_band="Peak", role_hints=["peak_tool"]),
        dict(track_id="trk_06", position=6, bpm=134.0, key="11B",
             energy=0.85, bass=0.80, drums=0.88, harmonic=0.58, groove=0.78,
             energy_rel=0.85, bass_rel=0.83, drums_rel=0.87, vocals_rel=0.08, groove_rel=0.80,
             intensity_band="Peak", role_hints=["peak_tool", "bass_driver"]),
        # trk_07 and trk_08: abs features only, no rel row → NULL rel columns
        dict(track_id="trk_07", position=7, bpm=126.0, key="7A",
             energy=0.48, bass=0.45, drums=0.52, harmonic=0.38, groove=0.44),
        dict(track_id="trk_08", position=8, bpm=127.0, key="7B",
             energy=0.50, bass=0.47, drums=0.54, harmonic=0.40, groove=0.46),
    ]
    for t in tracks:
        _insert_track(conn, **t)

    _insert_playlist_stats(conn, adapted_weights={
        "target_energy": 0.25, "transition_support": 0.16,
        "bass_transition": 0.14, "vocal_transition": 0.11,
        "harmonic": 0.14, "tempo": 0.09, "history_fit": 0.06, "rhythmic_continuity": 0.05,
    })

    conn.execute(
        "UPDATE playlists SET track_count = 8 WHERE id = ?",
        (_PLAYLIST_ID,),
    )
    conn.commit()
    conn.close()

    return Database(db_path)


# ---------------------------------------------------------------------------
# Tests: get_scoring_candidates
# ---------------------------------------------------------------------------


class TestGetScoringCandidates:
    def test_returns_all_tracks_with_abs(self, fixture_db):
        rows = fixture_db.get_scoring_candidates(_PLAYLIST_ID)
        assert len(rows) == 8

    def test_excludes_specified_track_ids(self, fixture_db):
        rows = fixture_db.get_scoring_candidates(_PLAYLIST_ID, exclude_track_ids=["trk_01", "trk_02"])
        ids = [r["track_id"] for r in rows]
        assert "trk_01" not in ids
        assert "trk_02" not in ids
        assert len(rows) == 6

    def test_excludes_empty_list_returns_all(self, fixture_db):
        rows = fixture_db.get_scoring_candidates(_PLAYLIST_ID, exclude_track_ids=[])
        assert len(rows) == 8

    def test_rows_have_expected_columns(self, fixture_db):
        rows = fixture_db.get_scoring_candidates(_PLAYLIST_ID)
        row = rows[0]
        for col in ("track_id", "bpm", "key", "key_confidence", "key_source",
                    "key_agreement", "energy_rel", "bass_rel", "intensity_band", "role_hints"):
            assert col in row.keys()

    def test_rows_ordered_by_position(self, fixture_db):
        rows = fixture_db.get_scoring_candidates(_PLAYLIST_ID)
        positions = [r["position"] for r in rows]
        assert positions == sorted(positions)

    def test_tracks_without_rel_have_null_rel_fields(self, fixture_db):
        rows = fixture_db.get_scoring_candidates(_PLAYLIST_ID)
        no_rel = [r for r in rows if r["track_id"] in ("trk_07", "trk_08")]
        assert len(no_rel) == 2
        for row in no_rel:
            assert row["energy_rel"] is None

    def test_nonexistent_playlist_returns_empty(self, fixture_db):
        rows = fixture_db.get_scoring_candidates("plt_does_not_exist")
        assert rows == []


# ---------------------------------------------------------------------------
# Tests: get_track_scoring_context
# ---------------------------------------------------------------------------


class TestGetTrackScoringContext:
    def test_returns_row_for_existing_track(self, fixture_db):
        row = fixture_db.get_track_scoring_context("trk_03", _PLAYLIST_ID)
        assert row is not None
        assert row["track_id"] == "trk_03"

    def test_returns_none_for_missing_track(self, fixture_db):
        row = fixture_db.get_track_scoring_context("trk_99", _PLAYLIST_ID)
        assert row is None

    def test_returns_none_for_wrong_playlist(self, fixture_db):
        row = fixture_db.get_track_scoring_context("trk_01", "plt_other")
        assert row is None

    def test_bpm_and_key_present(self, fixture_db):
        row = fixture_db.get_track_scoring_context("trk_01", _PLAYLIST_ID)
        assert row["bpm"] == pytest.approx(128.0)
        assert row["key"] == "8A"

    def test_rel_features_present_for_enriched_track(self, fixture_db):
        row = fixture_db.get_track_scoring_context("trk_02", _PLAYLIST_ID)
        assert row["energy_rel"] is not None
        assert row["intensity_band"] == "Groove"

    def test_rel_features_null_for_non_enriched_track(self, fixture_db):
        row = fixture_db.get_track_scoring_context("trk_07", _PLAYLIST_ID)
        assert row["energy_rel"] is None


# ---------------------------------------------------------------------------
# Tests: get_playlist_stats_for_scoring
# ---------------------------------------------------------------------------


class TestGetPlaylistStatsForScoring:
    def test_returns_dict_for_existing_stats(self, fixture_db):
        stats = fixture_db.get_playlist_stats_for_scoring(_PLAYLIST_ID)
        assert stats is not None
        assert isinstance(stats, dict)

    def test_returns_none_for_missing_playlist(self, fixture_db):
        stats = fixture_db.get_playlist_stats_for_scoring("plt_no_stats")
        assert stats is None

    def test_adapted_weights_decoded_as_dict(self, fixture_db):
        stats = fixture_db.get_playlist_stats_for_scoring(_PLAYLIST_ID)
        assert stats is not None
        aw = stats["adapted_weights"]
        assert isinstance(aw, dict)
        assert "target_energy" in aw

    def test_spread_fields_present(self, fixture_db):
        stats = fixture_db.get_playlist_stats_for_scoring(_PLAYLIST_ID)
        for field in ("energy_spread", "bass_spread", "drums_spread", "vocals_spread"):
            assert field in stats

    def test_no_adapted_weights_row(self, tmp_path):
        db_path = tmp_path / "no_weights.db"
        conn = _bootstrap_db(db_path)
        _insert_track(conn, track_id="trk_x", position=1)
        _insert_playlist_stats(conn, adapted_weights=None)
        conn.commit()
        conn.close()
        db = Database(db_path)
        stats = db.get_playlist_stats_for_scoring(_PLAYLIST_ID)
        assert stats is not None
        assert stats["adapted_weights"] is None
        db.close()


# ---------------------------------------------------------------------------
# Tests: row_to_scoring_track_context
# ---------------------------------------------------------------------------


class TestRowToScoringTrackContext:
    def _make_row(self, **overrides) -> dict[str, Any]:
        base: dict[str, Any] = {
            "track_id": "trk_01",
            "bpm": 128.0,
            "key": "8A",
            "key_confidence": 0.80,
            "key_source": "musicalkeycnn",
            "key_agreement": 1,
            "energy_rel": 0.50,
            "bass_rel": 0.45,
            "drums_rel": 0.55,
            "vocals_rel": 0.20,
            "groove_rel": 0.48,
            "intensity_band": "Drive",
            "role_hints": json.dumps(["safe_continuation", "pressure_builder"]),
        }
        base.update(overrides)
        return base

    def test_basic_conversion(self):
        row = self._make_row()
        ctx = row_to_scoring_track_context(row)
        assert isinstance(ctx, ScoringTrackContext)
        assert ctx.track_id == "trk_01"
        assert ctx.bpm == pytest.approx(128.0)
        assert ctx.key == "8A"
        assert ctx.key_confidence == pytest.approx(0.80)
        assert ctx.key_agreement == 1
        assert ctx.energy_rel == pytest.approx(0.50)

    def test_role_hints_json_decoded(self):
        row = self._make_row(role_hints=json.dumps(["opener", "bass_driver"]))
        ctx = row_to_scoring_track_context(row)
        assert ctx.role_hints == ["opener", "bass_driver"]

    def test_role_hints_already_list(self):
        row = self._make_row(role_hints=["peak_tool"])
        ctx = row_to_scoring_track_context(row)
        assert ctx.role_hints == ["peak_tool"]

    def test_role_hints_none_becomes_empty(self):
        row = self._make_row(role_hints=None)
        ctx = row_to_scoring_track_context(row)
        assert ctx.role_hints == []

    def test_role_hints_invalid_json_becomes_empty(self):
        row = self._make_row(role_hints="not-json")
        ctx = row_to_scoring_track_context(row)
        assert ctx.role_hints == []

    def test_null_rel_fields_become_none(self):
        row = self._make_row(energy_rel=None, bass_rel=None, vocals_rel=None,
                             intensity_band=None, role_hints=None)
        ctx = row_to_scoring_track_context(row)
        assert ctx.energy_rel is None
        assert ctx.bass_rel is None
        assert ctx.vocals_rel is None
        assert ctx.intensity_band is None

    def test_null_key_fields_become_none(self):
        row = self._make_row(key=None, key_confidence=None, key_source=None, key_agreement=None)
        ctx = row_to_scoring_track_context(row)
        assert ctx.key is None
        assert ctx.key_confidence is None
        assert ctx.key_source is None
        assert ctx.key_agreement is None

    def test_id_alias(self):
        row = self._make_row()
        ctx = row_to_scoring_track_context(row)
        assert ctx.id == ctx.track_id

    def test_accepts_sqlite_row(self, fixture_db):
        row = fixture_db.get_track_scoring_context("trk_03", _PLAYLIST_ID)
        ctx = row_to_scoring_track_context(row)
        assert ctx.track_id == "trk_03"
        assert ctx.bpm > 0


# ---------------------------------------------------------------------------
# Tests: build_scoring_config
# ---------------------------------------------------------------------------


class TestBuildScoringConfig:
    def test_returns_expected_keys(self, tmp_path):
        from cuemate_analysis.config import load_runtime_settings
        settings = load_runtime_settings(
            Path(__file__).resolve().parents[2]
        )
        from cuemate_analysis.config import build_scoring_config
        cfg = build_scoring_config(settings)
        for key in ("target", "static_weights", "weight_floors", "harmonic_confidence_floor",
                    "thresholds", "move_types", "penalties",
                    "contrast_threshold", "secondary_contrast_threshold", "max_per_lane"):
            assert key in cfg

    def test_default_target_is_maintain(self):
        from cuemate_analysis.config import load_runtime_settings, build_scoring_config
        settings = load_runtime_settings(Path(__file__).resolve().parents[2])
        cfg = build_scoring_config(settings)
        assert cfg["target"] == "maintain"

    def test_target_override(self):
        from cuemate_analysis.config import load_runtime_settings, build_scoring_config
        settings = load_runtime_settings(Path(__file__).resolve().parents[2])
        cfg = build_scoring_config(settings, target="build")
        assert cfg["target"] == "build"

    def test_harmonic_confidence_floor_matches_constant(self):
        from cuemate_analysis.config import load_runtime_settings, build_scoring_config
        from cuemate_analysis.scoring import HARMONIC_CONFIDENCE_FLOOR
        settings = load_runtime_settings(Path(__file__).resolve().parents[2])
        cfg = build_scoring_config(settings)
        assert cfg["harmonic_confidence_floor"] == pytest.approx(HARMONIC_CONFIDENCE_FLOOR)

    def test_thresholds_present(self):
        from cuemate_analysis.config import load_runtime_settings, build_scoring_config
        settings = load_runtime_settings(Path(__file__).resolve().parents[2])
        cfg = build_scoring_config(settings)
        assert "bpm_hard" in cfg["thresholds"]
        assert "bpm_soft" in cfg["thresholds"]
        assert "cooldown_window" in cfg["thresholds"]


# ---------------------------------------------------------------------------
# Tests: ScoringSettings defaults
# ---------------------------------------------------------------------------


class TestScoringSettingsDefaults:
    def test_defaults_populated_by_post_init(self):
        from cuemate_analysis.config import ScoringSettings
        s = ScoringSettings(static_weights={}, weight_floors={})
        assert s.thresholds is not None
        assert "bpm_hard" in s.thresholds
        assert s.move_types is not None
        assert "jump_threshold" in s.move_types
        assert s.penalties is not None
        assert "max_total_penalty" in s.penalties

    def test_explicit_values_not_overwritten(self):
        from cuemate_analysis.config import ScoringSettings
        custom_t = {"bpm_hard": 10.0, "bpm_soft": 5.0, "cooldown_window": 8}
        s = ScoringSettings(static_weights={}, weight_floors={}, thresholds=custom_t)
        assert s.thresholds["bpm_hard"] == 10.0


# ---------------------------------------------------------------------------
# Integration: DB query → row_to_scoring_track_context → score_candidate
# ---------------------------------------------------------------------------


class TestScoringPipeline:
    _DEFAULT_CONFIG: dict = {
        "target": "maintain",
        "thresholds": {"bpm_hard": 8.0, "bpm_soft": 3.0, "cooldown_window": 5},
        "move_types": {
            "jump_threshold": 0.12, "build_threshold": 0.05, "maintain_range": 0.05,
            "reset_energy_threshold": -0.08, "reset_vocal_threshold": 0.50, "drop_threshold": -0.05,
        },
        "penalties": {"max_total_penalty": 0.80, "bpm_over_soft": 0.30,
                      "key_mismatch": 0.45, "vocal_clash": 0.35},
    }

    def test_full_pipeline_scores_candidates(self, fixture_db):
        current_row = fixture_db.get_track_scoring_context("trk_03", _PLAYLIST_ID)
        candidate_rows = fixture_db.get_scoring_candidates(
            _PLAYLIST_ID, exclude_track_ids=["trk_03"]
        )
        stats = fixture_db.get_playlist_stats_for_scoring(_PLAYLIST_ID)

        current = row_to_scoring_track_context(current_row)
        candidates = [row_to_scoring_track_context(r) for r in candidate_rows]

        results = [
            score_candidate(current, c, [], self._DEFAULT_CONFIG, stats)
            for c in candidates
        ]

        assert len(results) == 7
        for r in results:
            assert 0.0 <= r["score"] <= 1.0
            assert r["move"] in ("maintain", "build", "jump", "reset", "drop")
            assert r["risk"] in ("low", "medium", "high")

    def test_adapted_weights_used_from_stats(self, fixture_db):
        current_row = fixture_db.get_track_scoring_context("trk_02", _PLAYLIST_ID)
        candidate_row = fixture_db.get_track_scoring_context("trk_03", _PLAYLIST_ID)
        stats = fixture_db.get_playlist_stats_for_scoring(_PLAYLIST_ID)

        current = row_to_scoring_track_context(current_row)
        candidate = row_to_scoring_track_context(candidate_row)

        result = score_candidate(current, candidate, [], self._DEFAULT_CONFIG, stats)
        # Adapted weights should be used (not static)
        assert result["weights_used"] == stats["adapted_weights"]

    def test_pipeline_without_rel_features(self, fixture_db):
        # trk_07 has no rel row — energy_rel etc. are None
        current_row = fixture_db.get_track_scoring_context("trk_07", _PLAYLIST_ID)
        candidate_row = fixture_db.get_track_scoring_context("trk_08", _PLAYLIST_ID)
        stats = fixture_db.get_playlist_stats_for_scoring(_PLAYLIST_ID)

        current = row_to_scoring_track_context(current_row)
        candidate = row_to_scoring_track_context(candidate_row)

        # Should not raise even with None rel features
        result = score_candidate(current, candidate, [], self._DEFAULT_CONFIG, stats)
        assert 0.0 <= result["score"] <= 1.0

    def test_role_hints_roundtrip_through_db(self, fixture_db):
        row = fixture_db.get_track_scoring_context("trk_05", _PLAYLIST_ID)
        ctx = row_to_scoring_track_context(row)
        assert "peak_tool" in ctx.role_hints

    def test_key_agreement_roundtrip(self, fixture_db):
        row = fixture_db.get_track_scoring_context("trk_01", _PLAYLIST_ID)
        ctx = row_to_scoring_track_context(row)
        assert ctx.key_agreement == 1

    def test_harmonic_confidence_corroborated_via_db(self, fixture_db):
        from cuemate_analysis.scoring import build_confidence_map
        current_row = fixture_db.get_track_scoring_context("trk_01", _PLAYLIST_ID)
        candidate_row = fixture_db.get_track_scoring_context("trk_02", _PLAYLIST_ID)
        current = row_to_scoring_track_context(current_row)
        candidate = row_to_scoring_track_context(candidate_row)
        conf = build_confidence_map(current, candidate)
        # Both tracks have key_agreement = 1 → full confidence
        assert conf["harmonic"] == pytest.approx(1.0)
