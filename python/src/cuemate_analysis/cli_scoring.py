"""CLI handlers for scoring/recommendation commands."""
from __future__ import annotations

import argparse
import json
import os

import cuemate_analysis.cli as _cli
from cuemate_analysis.feedback_shared import build_feedback_weight_layers


def _serialize(obj):
    from cuemate_analysis.scoring import ScoringTrackContext

    if isinstance(obj, ScoringTrackContext):
        return obj.track_id
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    return obj


def _emit_console(text: str) -> None:
    os.write(1, (text + "\n").encode("utf-8", errors="replace"))


def handle_recommend_next(args: argparse.Namespace) -> int:
    from cuemate_analysis.config import build_scoring_config
    from cuemate_analysis.scoring import get_recommendations, row_to_scoring_track_context

    settings = _cli.load_runtime_settings()
    config = build_scoring_config(settings, target=args.target)

    with _cli.Database(settings.database_path) as db:
        playlist_row = db.get_playlist(args.playlist)
        if playlist_row is None:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")
        playlist_id = str(playlist_row["id"])

        playlist_stats = db.get_playlist_stats_for_scoring(playlist_id)
        _cli._ensure_scoring_relative_freshness(args.playlist, playlist_stats, settings)

        candidate_rows = db.get_scoring_candidates(playlist_id)
        if not candidate_rows:
            raise SystemExit(f"No analyzed tracks found for playlist '{args.playlist}'.")

        current_id = args.current_track
        if current_id is None:
            current_id = str(candidate_rows[0]["track_id"])

        current_row = db.get_track_scoring_context(current_id, playlist_id)
        if current_row is None:
            raise SystemExit(f"Track '{current_id}' not found in playlist '{args.playlist}'.")
        current = row_to_scoring_track_context(current_row)
        candidates = [row_to_scoring_track_context(r) for r in candidate_rows]

    if args.current_track is None and not args.json:
        current_label = _cli.format_track_label(current.track_id, current.artist, current.title)
        print(f"No --current-track given; using first analyzed track: {current_label}")

    result = get_recommendations(
        current,
        candidates,
        history=[],
        config=config,
        playlist_stats=playlist_stats,
        target=args.target,
        max_per_lane=args.max_per_lane,
    )

    if args.json:
        _emit_console(json.dumps(_serialize(result), indent=2, sort_keys=True))
        return 0

    conf = result["recommendation_confidence"]
    meta = result["meta"]
    current_label = _cli.format_track_label(current.track_id, current.artist, current.title)
    print(f"\nRecommendations for '{args.playlist}' | current: {current_label}")
    print(f"Target: {args.target}  |  Confidence: {conf:.2f}  |  Scored: {meta['scored_candidates']} tracks\n")
    fallback_note = meta.get("fallback_note")
    if fallback_note:
        print(f"  Note: {fallback_note}\n")

    for lane in result["lane_order"]:
        items = result["lanes"].get(lane, [])
        if not items:
            continue
        print(f"  [{lane.upper()}]")
        for item in items:
            cand = item["candidate"]
            score = item["score"]
            move = item["move"]
            risk = item["risk"]
            bpm_rel = item["transition_features"].get("effective_bpm_distance", 0.0)
            key_label = _cli.normalize_display_text(item["transition_features"].get("key_compat_label", "-"))
            secondary = " (contrast)" if item.get("secondary_lane") else ""
            cand_label = _cli.format_track_label(cand.track_id, cand.artist, cand.title)
            print(
                f"    {cand_label}\n"
                f"      score={score:.3f}  move={move:<8}  risk={risk:<6}  bpm_dist={bpm_rel:.1f}  key={key_label}{secondary}"
            )
        print()

    return 0


def handle_score_pair(args: argparse.Namespace) -> int:
    from cuemate_analysis.config import build_scoring_config
    from cuemate_analysis.scoring import row_to_scoring_track_context, score_candidate

    settings = _cli.load_runtime_settings()
    config = build_scoring_config(settings, target=args.target)

    with _cli.Database(settings.database_path) as db:
        playlist_row = db.get_playlist(args.playlist)
        if playlist_row is None:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")
        playlist_id = str(playlist_row["id"])

        playlist_stats = db.get_playlist_stats_for_scoring(playlist_id)
        _cli._ensure_scoring_relative_freshness(args.playlist, playlist_stats, settings)

        current_row = db.get_track_scoring_context(args.current, playlist_id)
        if current_row is None:
            raise SystemExit(f"Track '{args.current}' not found in playlist '{args.playlist}'.")
        candidate_row = db.get_track_scoring_context(args.candidate, playlist_id)
        if candidate_row is None:
            raise SystemExit(f"Track '{args.candidate}' not found in playlist '{args.playlist}'.")

    current = row_to_scoring_track_context(current_row)
    candidate = row_to_scoring_track_context(candidate_row)

    result = score_candidate(
        current=current,
        candidate=candidate,
        history=[],
        config=config,
        playlist_stats=playlist_stats,
    )

    if args.json:
        _emit_console(json.dumps(_serialize(result), indent=2, sort_keys=True))
        return 0

    def _track_label(t) -> str:
        return _cli.format_track_label(t.track_id, t.artist, t.title)

    _emit_console(f"\nScore pair: {_track_label(current)}  ->  {_track_label(candidate)}")
    _emit_console(f"  Final score:      {result['score']:.4f}  (raw: {result['raw_score']:.4f})")
    _emit_console(f"  Penalty:          {result['penalty_multiplier']:.4f}")
    _emit_console(f"  Move:             {result['move']} (confidence={result['move_confidence']:.2f})")
    _emit_console(f"  Risk:             {result['risk']} (score={result['risk_score']:.3f})")
    _emit_console(f"  Contrast score:   {result['contrast_score']:.3f}")

    _emit_console("\n  Transition features:")
    for k, v in sorted(result["transition_features"].items()):
        _emit_console(f"    {k:<32}  {v}")

    _emit_console("\n  Component scores:")
    for k, v in sorted(result["component_scores"].items()):
        w = result["weights_used"].get(k, 0.0)
        c = result["confidences"].get(k, 1.0)
        if v is None:
            _emit_console(f"    {k:<24}  {'stub':<8}  weight={w:.4f}  (not implemented)")
        else:
            _emit_console(f"    {k:<24}  {v:.4f}  weight={w:.4f}  conf={c:.4f}")

    if result["penalty_factors"]:
        _emit_console("\n  Penalty factors:")
        for factor in result["penalty_factors"]:
            name = factor.get("factor", "unknown")
            penalty = factor.get("effective_penalty", factor.get("raw_penalty", ""))
            severity = factor.get("severity", "")
            _emit_console(f"    {name:<28}  penalty={penalty}  severity={severity}")

    if result["risk_factors"]:
        _emit_console("\n  Risk factors:")
        for note in result["risk_factors"]:
            _emit_console(f"    {note}")

    stub_components = sorted(
        name for name, value in result["component_scores"].items() if value is None
    )
    current_vocals_rel = result["transition_features"].get("current_vocals_rel")
    candidate_vocals_rel = result["transition_features"].get("candidate_vocals_rel")
    if stub_components or current_vocals_rel is None or candidate_vocals_rel is None:
        _emit_console("\n  Notes:")
        if stub_components:
            joined = ", ".join(stub_components)
            _emit_console(f"    Stubbed and excluded from weighted scoring: {joined}.")
        if current_vocals_rel is None or candidate_vocals_rel is None:
            _emit_console(
                "    vocals_abs / vocals_rel are not populated yet; missing vocal fields are unknown, not silence."
            )

    return 0


def handle_inspect_scoring_weights(args: argparse.Namespace) -> int:
    from cuemate_analysis.config import build_scoring_config

    settings = _cli.load_runtime_settings()
    config = build_scoring_config(settings, target="maintain")

    with _cli.Database(settings.database_path) as db:
        playlist_row = db.get_playlist(args.playlist)
        if playlist_row is None:
            raise SystemExit(f"Playlist '{args.playlist}' was not found.")
        playlist_id = str(playlist_row["id"])
        playlist_stats = db.get_playlist_stats_for_scoring(playlist_id)
        _cli._ensure_scoring_relative_freshness(args.playlist, playlist_stats, settings)

    weight_layers = build_feedback_weight_layers(playlist_stats, settings, config=config)
    static_weights = weight_layers["static"]
    base_weights = weight_layers["base"]
    tuned_weights = weight_layers["tuned"]
    effective_weights = weight_layers["effective"]
    weight_source = weight_layers["source"]

    if args.json:
        print(json.dumps({
            "playlist": args.playlist,
            "static_weights": static_weights,
            "base_weights": base_weights,
            "tuned_weights": tuned_weights,
            "effective_weights": effective_weights,
            "weight_source": weight_source,
            "weight_floors": config["weight_floors"],
        }, indent=2, sort_keys=True))
        return 0

    print(f"\nScoring weights for '{args.playlist}'\n")
    print(f"  Active weight source:    {weight_source}")
    print(f"  {'Component':<24}  {'Static':>8}  {'Base':>8}  {'Tuned':>8}  {'Effective':>9}")
    print(f"  {'-'*24}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}")
    for key in sorted(static_weights):
        s = static_weights.get(key, 0.0)
        a = (base_weights or {}).get(key) if base_weights else None
        t = (tuned_weights or {}).get(key) if tuned_weights else None
        e = effective_weights.get(key, s)
        a_str = f"{a:.4f}" if a is not None else "   n/a  "
        t_str = f"{t:.4f}" if t is not None else "   n/a  "
        print(f"  {key:<24}  {s:>8.4f}  {a_str:>8}  {t_str:>8}  {e:>9.4f}")

    if weight_source == "feedback_tuned_weights":
        print("\n  Feedback tuning is active and overrides the heuristic base weights.")
    elif weight_source == "adapted_weights":
        print("\n  Heuristic playlist adaptation is active (no tuned override yet).")
    else:
        print("\n  No playlist-specific weights are active; using static defaults.")

    return 0


def handle_inspect_scoring_metadata(args: argparse.Namespace) -> int:
    from cuemate_analysis.scoring import check_analysis_compatibility, get_scoring_metadata

    settings = _cli.load_runtime_settings()
    metadata = get_scoring_metadata(settings)

    compatibility = None
    if (
        args.analysis_signature is not None
        or args.config_signature is not None
        or args.scoring_contract_id_at_analysis is not None
    ):
        compatibility = check_analysis_compatibility(
            args.analysis_signature,
            args.config_signature,
            args.scoring_contract_id_at_analysis,
            scoring_metadata=metadata,
        )

    if args.json:
        payload: dict[str, object] = {"metadata": metadata}
        if compatibility is not None:
            payload["compatibility"] = compatibility
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    active = metadata["active_signatures"]
    print("\nScoring metadata\n")
    print(f"  analysis_signature:   {active['analysis_signature']}")
    print(f"  config_signature:     {active['config_signature']}")
    print(f"  scoring_contract_id:  {active['scoring_contract_id']}")
    print(f"  engine_version:       {metadata['engine_version']}")
    print(f"  healthy:              {metadata['healthy']}")
    print(f"  status_note:          {metadata['status_note']}")

    print("\n  Capability flags:")
    for key, value in sorted(metadata["capability_flags"].items()):
        print(f"    {key:<32} {value}")

    print("\n  Supported lanes:")
    for lane in metadata["supported_lane_groups"]:
        print(f"    {lane['lane_id']:<10} {lane['summary']}")

    print("\n  Components:")
    for component in metadata["components"]:
        available = component.get("available")
        component_active = component.get("active")
        if available is False:
            state = "stubbed"
        elif component_active is False:
            state = "inactive"
        else:
            state = "active"
        print(
            f"    {component['component_id']:<24} weight={component['weight']:.4f}  state={state:<8}  {component['description']}"
        )

    if compatibility is not None:
        print("\n  Compatibility:")
        print(f"    exact_match:         {compatibility['exact_match']}")
        print(f"    compatible:          {compatibility['compatible']}")
        print(f"    requires_reanalysis: {compatibility['requires_reanalysis']}")
        print(f"    reason:              {compatibility['reason']}")
        notes = compatibility.get("notes", [])
        if notes:
            print("    notes:")
            for note in notes:
                print(f"      {note}")

    return 0


def handle_serve_scoring(args: argparse.Namespace) -> int:
    from cuemate_analysis.scoring_service import serve_scoring_grpc

    settings = _cli.load_runtime_settings()
    return serve_scoring_grpc(
        host=args.host,
        port=args.port,
        settings=settings,
        max_workers=args.max_workers,
    )
