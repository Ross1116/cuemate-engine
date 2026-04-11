from __future__ import annotations

import importlib
import sys
from importlib import resources
from pathlib import Path

import grpc
import pytest

from cuemate_analysis.config import build_scoring_config, load_runtime_settings
from cuemate_analysis.scoring import (
    ScoringTrackContext,
    get_recommendations,
    get_scoring_metadata,
    score_candidate,
)


def _track(
    *,
    track_id: str,
    bpm: float,
    key: str | None,
    energy_rel: float | None,
    bass_rel: float | None,
    vocals_rel: float | None,
    title: str,
) -> ScoringTrackContext:
    return ScoringTrackContext(
        track_id=track_id,
        bpm=bpm,
        key=key,
        key_confidence=0.9 if key else None,
        key_source="musicalkeycnn" if key else None,
        key_agreement=1 if key else None,
        energy_rel=energy_rel,
        bass_rel=bass_rel,
        drums_rel=0.5,
        vocals_rel=vocals_rel,
        groove_rel=0.5,
        intensity_band="Drive",
        role_hints=["anthem"] if energy_rel and energy_rel > 0.7 else ["groove"],
        title=title,
        artist="Test Artist",
    )


@pytest.fixture
def scoring_proto_runtime(tmp_path, monkeypatch):
    grpc_tools_protoc = pytest.importorskip("grpc_tools.protoc")
    repo_root = Path(__file__).resolve().parents[2]
    out_root = tmp_path / "generated"
    for package_path in [
        out_root / "djengine",
        out_root / "djengine" / "scoring",
        out_root / "djengine" / "scoring" / "v1",
    ]:
        package_path.mkdir(parents=True, exist_ok=True)
        (package_path / "__init__.py").write_text('"""generated test package"""', encoding="utf-8")

    proto_file = repo_root / "proto" / "djengine" / "scoring" / "v1" / "scoring.proto"
    result = grpc_tools_protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{repo_root / 'proto'}",
            f"-I{resources.files('grpc_tools').joinpath('_proto')}",
            f"--python_out={out_root}",
            f"--grpc_python_out={out_root}",
            str(proto_file),
        ]
    )
    assert result == 0

    for name in [module_name for module_name in list(sys.modules) if module_name == "djengine" or module_name.startswith("djengine.")]:
        sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(str(out_root))
    importlib.invalidate_caches()

    pb2 = importlib.import_module("djengine.scoring.v1.scoring_pb2")
    pb2_grpc = importlib.import_module("djengine.scoring.v1.scoring_pb2_grpc")
    return pb2, pb2_grpc


def _signature_message(pb2, payload: dict[str, str]):
    return pb2.SignatureMetadata(
        analysis_signature=payload["analysis_signature"],
        config_signature=payload["config_signature"],
        scoring_contract_id=payload["scoring_contract_id"],
    )


def _track_message(pb2, track: ScoringTrackContext, signatures: dict[str, str]):
    message = pb2.TrackContext(
        track_id=track.track_id,
        signatures=_signature_message(pb2, signatures),
        bpm=track.bpm,
        musical_key=track.key or "",
        key_source=track.key_source or "",
        intensity_band=track.intensity_band or "",
        role_hints=track.role_hints,
        title=track.title or "",
        artist=track.artist or "",
    )
    if track.key_confidence is not None:
        message.key_confidence = track.key_confidence
    if track.key_agreement is not None:
        message.key_agreement = track.key_agreement
    if track.energy_rel is not None:
        message.energy_rel = track.energy_rel
    if track.bass_rel is not None:
        message.bass_rel = track.bass_rel
    if track.drums_rel is not None:
        message.drums_rel = track.drums_rel
    if track.vocals_rel is not None:
        message.vocals_rel = track.vocals_rel
    if track.groove_rel is not None:
        message.groove_rel = track.groove_rel
    return message


def _history_messages(pb2):
    item = pb2.HistoryEntry(track_id="trk_history", musical_key="8A", relation="played", plays_ago=1)
    item.energy_rel = 0.48
    return [item]


def test_get_recommendations_matches_direct_scorer(scoring_proto_runtime):
    pb2, pb2_grpc = scoring_proto_runtime
    from cuemate_analysis.scoring_service import build_grpc_server

    settings = load_runtime_settings()
    signatures = get_scoring_metadata(settings)["active_signatures"]
    current = _track(
        track_id="trk_current",
        bpm=128.0,
        key="8A",
        energy_rel=0.52,
        bass_rel=0.50,
        vocals_rel=0.10,
        title="Current",
    )
    candidates = [
        _track(
            track_id="trk_build",
            bpm=129.0,
            key="8A",
            energy_rel=0.65,
            bass_rel=0.62,
            vocals_rel=0.20,
            title="Build",
        ),
        _track(
            track_id="trk_reset",
            bpm=126.0,
            key="7A",
            energy_rel=0.38,
            bass_rel=0.35,
            vocals_rel=0.35,
            title="Reset",
        ),
    ]
    history = [{"track_id": "trk_history", "key": "8A", "energy_rel": 0.48}]
    config = build_scoring_config(settings, target="build")
    playlist_stats = {"energy_spread": 0.18}
    direct = get_recommendations(
        current_track=current,
        candidates=candidates,
        history=history,
        config=config,
        playlist_stats=playlist_stats,
        target="build",
        max_per_lane=2,
    )

    server = build_grpc_server(settings)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        stub = pb2_grpc.ScoringServiceStub(channel)
        request = pb2.GetRecommendationsRequest(
            current_track=_track_message(pb2, current, signatures),
            candidates=[_track_message(pb2, candidate, signatures) for candidate in candidates],
            history=_history_messages(pb2),
            target_lane="build",
            max_per_lane=2,
        )
        request.playlist_stats.energy_spread = 0.18
        response = stub.GetRecommendations(request, timeout=5)
    finally:
        channel.close()
        server.stop(0.5)

    assert list(response.lane_order) == list(direct["lane_order"])
    assert response.meta.current_track_id == direct["meta"]["current_track_id"]
    assert response.meta.fallback_note == (direct["meta"]["fallback_note"] or "")
    assert response.recommendation_confidence == pytest.approx(direct["recommendation_confidence"], abs=1e-6)
    lane_map = {lane.lane_group.lane_id: lane for lane in response.lanes}
    assert lane_map["build"].availability == "empty"
    assert lane_map["build"].empty_reason
    direct_top = direct["lanes"][direct["lane_order"][0]][0]
    top_lane = lane_map[direct["lane_order"][0]]
    assert top_lane.items[0].candidate.track_id == direct_top["candidate"].track_id
    assert top_lane.items[0].move == direct_top["move"]
    assert top_lane.items[0].final_score == pytest.approx(direct_top["score"], abs=1e-6)
    assert top_lane.items[0].ranking_strength >= 0.0
    assert list(top_lane.items[0].reasons)


def test_score_candidate_matches_direct_scorer(scoring_proto_runtime):
    pb2, pb2_grpc = scoring_proto_runtime
    from cuemate_analysis.scoring_service import build_grpc_server

    settings = load_runtime_settings()
    signatures = get_scoring_metadata(settings)["active_signatures"]
    current = _track(
        track_id="trk_current",
        bpm=128.0,
        key="8A",
        energy_rel=0.55,
        bass_rel=0.55,
        vocals_rel=0.10,
        title="Current",
    )
    candidate = _track(
        track_id="trk_candidate",
        bpm=127.5,
        key="9A",
        energy_rel=0.41,
        bass_rel=0.36,
        vocals_rel=0.32,
        title="Candidate",
    )
    history = [{"track_id": "trk_history", "key": "8A", "energy_rel": 0.52}]
    config = build_scoring_config(settings, target="reset")
    playlist_stats = {"energy_spread": 0.18}
    direct = score_candidate(current, candidate, history, config, playlist_stats)

    server = build_grpc_server(settings)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        stub = pb2_grpc.ScoringServiceStub(channel)
        request = pb2.ScoreCandidateRequest(
            current_track=_track_message(pb2, current, signatures),
            candidate=_track_message(pb2, candidate, signatures),
            history=_history_messages(pb2),
            target_lane="reset",
        )
        request.playlist_stats.energy_spread = 0.18
        response = stub.ScoreCandidate(request, timeout=5)
    finally:
        channel.close()
        server.stop(0.5)

    scored = response.scored_candidate
    assert scored.candidate.track_id == direct["candidate"].track_id
    assert scored.final_score == pytest.approx(direct["score"], abs=1e-6)
    assert scored.raw_score == pytest.approx(direct["raw_score"], abs=1e-6)
    assert scored.move == direct["move"]
    assert scored.risk == direct["risk"]
    assert dict(scored.component_scores)["harmonic"] == pytest.approx(direct["component_scores"]["harmonic"], abs=1e-6)


def test_get_scoring_metadata_matches_direct_metadata(scoring_proto_runtime):
    pb2, pb2_grpc = scoring_proto_runtime
    from cuemate_analysis.scoring_service import build_grpc_server

    settings = load_runtime_settings()
    direct = get_scoring_metadata(settings)
    server = build_grpc_server(settings)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        stub = pb2_grpc.ScoringServiceStub(channel)
        response = stub.GetScoringMetadata(pb2.GetScoringMetadataRequest(), timeout=5)
    finally:
        channel.close()
        server.stop(0.5)

    assert response.active_signatures.analysis_signature == direct["active_signatures"]["analysis_signature"]
    assert response.active_signatures.config_signature == direct["active_signatures"]["config_signature"]
    assert response.active_signatures.scoring_contract_id == direct["active_signatures"]["scoring_contract_id"]
    assert response.engine_version == direct["engine_version"]
    assert response.status_note == direct["status_note"]
    assert response.expected_relative_signature == direct["expected_relative_signature"]
    component_states = {item.component_id: item.status for item in response.components}
    assert component_states["transition_support"] == "stubbed"


def test_service_rejects_missing_signatures(scoring_proto_runtime):
    pb2, pb2_grpc = scoring_proto_runtime
    from cuemate_analysis.scoring_service import build_grpc_server

    settings = load_runtime_settings()
    current = _track(
        track_id="trk_current",
        bpm=128.0,
        key="8A",
        energy_rel=0.52,
        bass_rel=0.50,
        vocals_rel=0.10,
        title="Current",
    )
    candidate = _track(
        track_id="trk_candidate",
        bpm=129.0,
        key="8A",
        energy_rel=0.65,
        bass_rel=0.62,
        vocals_rel=0.20,
        title="Candidate",
    )

    server = build_grpc_server(settings)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        stub = pb2_grpc.ScoringServiceStub(channel)
        request = pb2.ScoreCandidateRequest(
            current_track=_track_message(pb2, current, {"analysis_signature": "", "config_signature": "", "scoring_contract_id": ""}),
            candidate=_track_message(pb2, candidate, {"analysis_signature": "", "config_signature": "", "scoring_contract_id": ""}),
            target_lane="maintain",
        )
        with pytest.raises(grpc.RpcError) as exc_info:
            stub.ScoreCandidate(request, timeout=5)
    finally:
        channel.close()
        server.stop(0.5)

    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION


def test_playlist_stats_proto_uses_weight_source_enum(scoring_proto_runtime):
    pb2, _ = scoring_proto_runtime
    from cuemate_analysis.scoring_service import _playlist_stats_from_proto

    message = pb2.PlaylistStatsContext()
    message.weight_source_enum = pb2.WEIGHT_SOURCE_FEEDBACK_TUNED

    payload = _playlist_stats_from_proto(message)

    assert payload is not None
    assert payload["weight_source"] == "feedback_tuned_weights"


def test_playlist_stats_proto_omits_unknown_weight_source(scoring_proto_runtime):
    pb2, _ = scoring_proto_runtime
    from cuemate_analysis.scoring_service import _playlist_stats_from_proto

    message = pb2.PlaylistStatsContext()

    payload = _playlist_stats_from_proto(message)

    assert payload is None
