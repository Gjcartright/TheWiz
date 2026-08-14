from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.rl.brain_contract import (
    CANDIDATE_SETUP_PACKET_VERSION,
    FORWARD_WALK_PACKET_VERSION,
    OVERALL_LANE,
    PAPER_OUTCOME_PACKET_VERSION,
    WIZARD_LANE,
    NATIVE_LANE,
    append_candidate_setup_packets,
    append_forward_walk_packets,
    append_paper_outcome_packets,
    build_phase1_readiness_surfaces,
    brain_output_paths,
    candidate_paper_credibility_status,
    write_candidate_setup_frame,
    write_forward_walk_frame,
    write_paper_outcome_frame,
)
from quant_platform.wizard_evidence import _ensure_wizard_evidence
from quant_platform.wizard_hourly_database import build_hourly_database_from_live_scanner


def build_three_brain_system(root: Path = ROOT) -> CommandResult:
    hourly_database = build_hourly_database_from_live_scanner(root=root)
    wizard_ontology = build_wizard_dashboard_ontology(root=root)
    wizard_investigation = build_wizard_dashboard_investigation(root=root)
    wizard_candidates = build_wizard_specialist_lane(root=root)
    native_candidates = build_native_specialist_lane(root=root)
    shadow = build_shadow_rl_lanes(root=root)
    overall = build_overall_brain(root=root)
    promotion = build_promotion_readiness(root=root)
    fw_strength = build_forward_walk_strength_report(root=root)
    native_quality = build_native_quality_report(root=root)
    native_diagnosis = build_native_candidate_diagnosis(root=root)
    wizard_hourly_targets = build_wizard_hourly_repair_targets(root=root)
    wizard_repair = build_wizard_candidate_repair_report(root=root)
    wizard_capture_priority = build_wizard_hourly_priority_capture_queue(root=root)
    native_focus = build_native_focus_repair_report(root=root)
    repair_loop = build_candidate_repair_loop(root=root)
    shadow_usefulness = build_shadow_rl_usefulness_report(root=root)
    shadow_candidates = build_shadow_rl_candidate_set(root=root)
    shadow_intake = build_shadow_rl_intake_report(root=root)
    arbitration = build_overall_arbitration_scorecard(root=root)
    blocker = build_blocker_persistence_report(root=root)
    shared_outcomes = build_shared_outcome_memory(root=root)
    native_outcome_features = build_native_outcome_feature_memory(root=root)
    native_independence = build_native_promotion_independence_report(root=root)
    readiness_paths = build_phase1_readiness_surfaces(root)
    return CommandResult(
        paths={
            **hourly_database,
            **wizard_ontology.paths,
            **wizard_investigation.paths,
            **wizard_candidates.paths,
            **native_candidates.paths,
            **shadow.paths,
            **overall.paths,
            **promotion.paths,
            **fw_strength.paths,
            **native_quality.paths,
            **native_diagnosis.paths,
            **wizard_repair.paths,
            **wizard_capture_priority.paths,
            **native_focus.paths,
            **wizard_hourly_targets.paths,
            **repair_loop.paths,
            **shadow_usefulness.paths,
            **shadow_candidates.paths,
            **shadow_intake.paths,
            **arbitration.paths,
            **blocker.paths,
            **shared_outcomes.paths,
            **native_outcome_features.paths,
            **native_independence.paths,
            **readiness_paths,
        },
        summary={
            "hourly_live_pairs": int(len(_read_csv(root / "reports" / "active" / "wizard_hourly_database" / "wizard_hourly_pair_queue.csv"))),
            "wizard_candidates": int(wizard_candidates.summary.get("candidate_rows", 0)),
            "native_candidates": int(native_candidates.summary.get("candidate_rows", 0)),
            "shadow_rows": int(shadow.summary.get("rows", 0)),
            "overall_rows": int(overall.summary.get("rows", 0)),
            "paper_ready": int(promotion.summary.get("paper_ready", 0)),
            "verified_outcomes": int(blocker.summary.get("verified_outcomes", 0)),
            "shared_outcomes": int(shared_outcomes.summary.get("rows", 0)),
            "native_outcome_feature_rows": int(native_outcome_features.summary.get("rows", 0)),
            "readiness_aliases": int(len(readiness_paths)),
        },
    )


def build_wizard_dashboard_ontology(root: Path = ROOT) -> CommandResult:
    evidence = _read_csv(root / "data" / "processed" / "wizard_evidence.csv")
    pair_detail_checklist = _read_csv(root / "reports" / "pair_detail_capture_checklist.csv")
    pair_detail_fields = _read_csv(root / "reports" / "pair_detail_field_dictionary.csv")
    rows = []
    for _, row in evidence.iterrows():
        rows.append(
            {
                "pair": _text(row.get("pair", "")),
                "venue": (_text(row.get("exchange", "")) or "dydx").lower(),
                "top_level_section": "dydx_discovery",
                "section_type": "pair_detail",
                "visible_metrics": ";".join(
                    [
                        f"sharpe={_text(row.get('sharpe', ''))}",
                        f"returns_total={_text(row.get('returns_total', ''))}",
                        f"hurst={_text(row.get('hurst', ''))}",
                        f"half_life={_text(row.get('half_life', ''))}",
                    ]
                ),
                "strategy_label": _text(row.get("dashboard_recommended_strategy", "")) or _text(row.get("exact_mode", "")),
                "asset_click_state": "pair_detail_available",
                "section_meaning": "wizard_dydx_opportunity",
                "source_path": _text(row.get("source_path", "")) or _text(row.get("evidence_path", "")),
            }
        )
    for _, row in pair_detail_checklist.iterrows():
        rows.append(
            {
                "pair": _text(row.get("pair", "")),
                "venue": "dydx",
                "top_level_section": "pair_detail_capture",
                "section_type": "history_coverage",
                "visible_metrics": ";".join(
                    [
                        f"history_rows={_safe_int(row.get('history_rows', 0))}",
                        f"capture_score={_safe_float(row.get('capture_completeness_score', 0.0))}",
                        f"baseline_ready={bool(row.get('baseline_ready', False))}",
                        f"ecm_ready={bool(row.get('ecm_ready', False))}",
                        f"two_leg_ready={bool(row.get('two_leg_ready', False))}",
                    ]
                ),
                "strategy_label": "pair_detail_capture",
                "asset_click_state": "pair_detail_capture_imported",
                "section_meaning": _text(row.get("next_capture_focus", "")) or "pair_detail_capture_status",
                "source_path": _text(row.get("path", "")),
            }
        )
    if not pair_detail_fields.empty:
        summary = (
            pair_detail_fields.assign(pair=pair_detail_fields.get("pair", pd.Series(dtype=str)).map(_text), field=pair_detail_fields.get("field", pd.Series(dtype=str)).map(_text))
            .groupby("pair", dropna=False)["field"]
            .agg(lambda values: sorted({value for value in values if value}))
        )
        for pair, fields in summary.items():
            rows.append(
                {
                    "pair": _text(pair),
                    "venue": "dydx",
                    "top_level_section": "pair_detail_capture",
                    "section_type": "field_dictionary",
                    "visible_metrics": f"field_count={len(fields)}",
                    "strategy_label": "field_dictionary",
                    "asset_click_state": "pair_detail_fields_available",
                    "section_meaning": ",".join(fields[:12]),
                    "source_path": "reports/pair_detail_field_dictionary.csv",
                }
            )
    frame = pd.DataFrame(
        rows,
        columns=[
            "pair",
            "venue",
            "top_level_section",
            "section_type",
            "visible_metrics",
            "strategy_label",
            "asset_click_state",
            "section_meaning",
            "source_path",
        ],
    )
    path = root / "reports" / "brain" / "wizard_dashboard_ontology.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"wizard_dashboard_ontology": path}, summary={"rows": int(len(frame))})


def build_wizard_dashboard_investigation(root: Path = ROOT) -> CommandResult:
    ontology = _read_csv(root / "reports" / "brain" / "wizard_dashboard_ontology.csv")
    live_probe = _read_json(root / "reports" / "brain" / "wizard_dashboard_live_probe.json")
    evidence = _read_csv(root / "data" / "processed" / "wizard_evidence.csv")
    pair_detail_checklist = _read_csv(root / "reports" / "pair_detail_capture_checklist.csv")
    exact_mode_queue = _read_csv(root / "reports" / "active" / "wizard_exact_mode_capture_queue.csv")
    strategy_alignment = _read_csv(root / "reports" / "active" / "wizard_strategy_alignment_report.csv")
    live_capture = _read_csv(root / "reports" / "active" / "crypto_wizards_live_scanner_capture.csv")
    rows = []
    rows.append(
        {
            "investigation_area": "live_dashboard_access",
            "status": str(live_probe.get("status", "not_attempted")),
            "detail": str(live_probe.get("detail", "live dashboard probe not captured")),
            "evidence_path": str(live_probe.get("evidence_path", "reports/brain/wizard_dashboard_live_probe.json")),
            "next_action": str(live_probe.get("next_action", "open the authenticated Wizard dashboard and capture the scanner or pair page")),
        }
    )
    rows.append(
        {
            "investigation_area": "scanner_surface_coverage",
            "status": "structured" if not ontology.empty else "missing",
            "detail": f"ontology_rows={len(ontology)}",
            "evidence_path": "reports/brain/wizard_dashboard_ontology.csv",
            "next_action": "compare ontology rows against live scanner sections and exact strategy labels",
        }
    )
    if not pair_detail_checklist.empty:
        imported = pair_detail_checklist.copy()
        import_ready = imported.get("import_ready", pd.Series(dtype=bool)).fillna(False).astype(bool)
        research_ready = imported.get("research_spine_ready", pd.Series(dtype=bool)).fillna(False).astype(bool)
        ecm_ready = imported.get("ecm_ready", pd.Series(dtype=bool)).fillna(False).astype(bool)
        rows.append(
            {
                "investigation_area": "pair_detail_capture_coverage",
                "status": "structured" if bool(import_ready.any()) else "thin",
                "detail": f"captures={len(imported)};import_ready={int(import_ready.sum())};research_ready={int(research_ready.sum())};ecm_ready={int(ecm_ready.sum())}",
                "evidence_path": "reports/pair_detail_capture_checklist.csv",
                "next_action": "favor live pair pages that can refresh exact strategy mode and pair-detail payload coverage",
            }
        )
    if not strategy_alignment.empty:
        evidence_paths = strategy_alignment.get("evidence_path", pd.Series(dtype=object)).map(_text)
        live_capture_path = "reports/active/crypto_wizards_live_scanner_capture.csv"
        if live_capture.empty:
            live_mask = evidence_paths.str.contains(live_capture_path, regex=False)
        else:
            live_pairs = set(live_capture.get("pair", pd.Series(dtype=object)).map(_text))
            live_mask = evidence_paths.str.contains(live_capture_path, regex=False) | strategy_alignment.get("pair", pd.Series(dtype=object)).map(_text).isin(live_pairs)
        unmapped = strategy_alignment.get("strategy_mapping_status", pd.Series(dtype=object)).map(_text).eq("unmapped")
        unmapped_live = unmapped & live_mask
        unmapped_backlog = unmapped & ~live_mask
        rows.append(
            {
                "investigation_area": "strategy_alignment_gap",
                "status": "blocked" if bool(unmapped_live.any()) else "covered",
                "detail": (
                    f"rows={len(strategy_alignment)};live_rows={int(live_mask.sum())};"
                    f"live_unmapped={int(unmapped_live.sum())};historical_unmapped={int(unmapped_backlog.sum())}"
                ),
                "evidence_path": "reports/active/wizard_strategy_alignment_report.csv",
                "next_action": (
                    "capture exact strategy mode from currently visible live scanner pairs"
                    if bool(unmapped_live.any())
                    else "current live scanner rows are mapped; treat remaining unmapped rows as backlog to revisit separately"
                ),
            }
        )
        rows.append(
            {
                "investigation_area": "historical_alignment_backlog",
                "status": "backlog" if bool(unmapped_backlog.any()) else "clear",
                "detail": f"historical_unmapped={int(unmapped_backlog.sum())}",
                "evidence_path": "reports/active/wizard_strategy_alignment_report.csv",
                "next_action": (
                    "backfill stale Wizard backlog pairs if they reappear in the live scanner"
                    if bool(unmapped_backlog.any())
                    else "no historical alignment backlog remains"
                ),
            }
        )
    exact_modes = evidence.get("exact_mode", pd.Series(dtype=object)).map(_text) if not evidence.empty else pd.Series(dtype=object)
    exact_mode_count = int(exact_modes.ne("").sum()) if not exact_modes.empty else 0
    rows.append(
        {
            "investigation_area": "exact_mode_capture",
            "status": "covered" if exact_mode_count > 0 else "blocked",
            "detail": f"exact_mode_rows={exact_mode_count}",
            "evidence_path": "data/processed/wizard_evidence.csv",
            "next_action": "capture live pair detail pages to replace exact-mode gaps" if exact_mode_count == 0 else "verify live labels still match archived exact-mode values",
        }
    )
    if not exact_mode_queue.empty:
        rows.append(
            {
                "investigation_area": "exact_mode_capture_queue",
                "status": "ready" if bool(len(exact_mode_queue)) else "empty",
                "detail": f"queue_rows={len(exact_mode_queue)};top_pair={_text(exact_mode_queue.iloc[0].get('pair', ''))}",
                "evidence_path": "reports/active/wizard_exact_mode_capture_queue.csv",
                "next_action": _text(exact_mode_queue.iloc[0].get("operator_action", "")) or "open top queued pair page and capture exact strategy fields",
            }
        )
    frame = pd.DataFrame(rows, columns=["investigation_area", "status", "detail", "evidence_path", "next_action"])
    path = root / "reports" / "brain" / "wizard_dashboard_investigation.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"wizard_dashboard_investigation": path}, summary={"rows": int(len(frame))})


def build_wizard_specialist_lane(root: Path = ROOT) -> CommandResult:
    evidence = _ensure_wizard_evidence(root)
    hypotheses = _read_csv(root / "reports" / "active" / "wizard_hypotheses.csv")
    diagnostics = _read_csv(root / "reports" / "active" / "wizard_diagnostic_confirmation.csv")
    parity = _read_csv(root / "reports" / "active" / "wizard_vs_local_parity_report.csv")
    verification = _read_csv(root / "reports" / "active" / "wizard_local_verification_batch.csv")
    journal = _read_csv(root / "reports" / "paper_trading_journal.csv")
    live_scanner_context = _load_wizard_live_scanner_context(root)
    second_page_context = _load_wizard_second_page_context(root)

    packets = []
    forward_walk_packets = []
    paper_packets = []
    fold_rows = []
    for _, row in evidence.iterrows():
        pair = _text(row.get("pair", ""))
        if not pair:
            continue
        setup_identity = _text(row.get("setup_identity", ""))
        exact_mode = _text(row.get("exact_mode", ""))
        timeframe = _text(row.get("interval", ""))
        candidate_id = _candidate_id(WIZARD_LANE, pair, exact_mode, timeframe)
        verification_row = _match_setup_or_pair_row(verification, pair, setup_identity)
        hypothesis_row = _match_setup_or_pair_row(hypotheses, pair, setup_identity)
        diagnostic_row = _match_setup_or_pair_row(diagnostics, pair, setup_identity)
        parity_row = _match_setup_or_pair_row(parity, pair, setup_identity)
        live_row = _match_wizard_live_context_row(live_scanner_context, pair)
        second_page_row = _match_wizard_live_context_row(second_page_context, pair)
        blocker_state = _wizard_blocker_state(hypothesis_row, diagnostic_row, parity_row, verification_row)
        regime_snapshot = _wizard_packet_regime_snapshot(row, live_row=live_row, second_page_row=second_page_row)
        packet = {
            "candidate_id": candidate_id,
            "lane": WIZARD_LANE,
            "source_type": "wizard_dashboard_primary" if bool(row.get("primary_wizard_setup", False)) else "wizard_dashboard_alternate",
            "source_path": _text(row.get("source_path", row.get("evidence_path", "data/processed/wizard_evidence.csv"))),
            "pair": pair,
            "venue": (_text(row.get("exchange", "")) or "dydx").lower(),
            "detection_timestamp": _now(),
            "timeframe": timeframe,
            "setup_identity": setup_identity,
            "setup_rank": int(_safe_float(row.get("setup_rank_within_pair", 0)) or 0),
            "setup_role": "primary" if bool(row.get("primary_wizard_setup", False)) else "alternate",
            "strategy_family": _text(row.get("local_strategy_family", "")) or _text(row.get("dashboard_recommended_strategy", "")),
            "strategy_mode": exact_mode,
            "normalized_feature_bundle_ref": _wizard_feature_bundle_ref(row, live_row=live_row, second_page_row=second_page_row),
            "regime_snapshot": regime_snapshot,
            "confidence": _wizard_confidence(row, verification_row),
            "blocker_state": blocker_state,
            "backtest_summary_ref": _wizard_backtest_summary_ref(
                row,
                verification_row=verification_row,
                second_page_row=second_page_row,
                exact_mode=exact_mode,
            ),
            "forward_walk_summary_ref": f"reports/brain/wizard_forward_walk.csv#{candidate_id}",
            "paper_outcome_ref": f"reports/brain/wizard_paper_outcomes.csv#{candidate_id}" if _paper_journal_matches(journal, pair) else "",
            "provenance": _wizard_packet_provenance(live_row=live_row, second_page_row=second_page_row),
            "schema_version": CANDIDATE_SETUP_PACKET_VERSION,
        }
        packets.append(packet)

        walk_metrics, walk_folds = _wizard_forward_walk_metrics(root, candidate_id, verification_row)
        fold_rows.extend(walk_folds)
        forward_walk_packets.append(
            {
                "candidate_id": candidate_id,
                "lane": WIZARD_LANE,
                "rolling_split_definition": str(walk_metrics.get("rolling_split_definition", "wizard_local_validation_proxy")),
                "oos_sharpe": walk_metrics.get("oos_sharpe", ""),
                "oos_profit_factor": walk_metrics.get("oos_profit_factor", ""),
                "oos_max_drawdown": walk_metrics.get("oos_max_drawdown", ""),
                "oos_trade_count": walk_metrics.get("oos_trade_count", 0),
                "stability_metrics": json.dumps(
                    {
                        "diagnostic_blocker": _text(diagnostic_row.get("diagnostic_blocker", "")) if diagnostic_row is not None else "",
                        "parity_status": _text(parity_row.get("parity_status", "")) if parity_row is not None else "",
                        "fold_count": int(walk_metrics.get("fold_count", 0)),
                    },
                    sort_keys=True,
                ),
                "forward_walk_status": _text(walk_metrics.get("forward_walk_status", "missing")),
                "blocker_reason": _text(walk_metrics.get("blocker_reason", "")),
                "dataset_provenance": _text(row.get("source_path", row.get("evidence_path", ""))),
                "run_manifest_ref": _text(verification_row.get("summary_path", "")) if verification_row is not None else "",
                "provenance": "wizard_local_verification_proxy",
                "schema_version": FORWARD_WALK_PACKET_VERSION,
            }
        )

        journal_row = _match_paper_journal_row(journal, pair)
        if journal_row is not None:
            journal_verification = _paper_journal_verification_status(journal_row)
            paper_packets.append(
                {
                    "candidate_id": candidate_id,
                    "lane": WIZARD_LANE,
                    "paper_venue": "dydx",
                    "submission_timestamp": str(journal_row.get("timestamp_utc", _now())),
                    "entry": "",
                    "exit": "",
                    "hold_duration": "",
                    "realized_return": _safe_float(journal_row.get("realized_return", "")),
                    "drawdown": "",
                    "slippage_cost_assumptions": str(journal_row.get("intents_json", "")),
                    "result_status": str(journal_row.get("plan_status", "paper_submitted")),
                    "verification_status": journal_verification,
                    "outcome_evidence_path": "reports/paper_trading_journal.csv",
                    "provenance": "paper_journal",
                    "schema_version": PAPER_OUTCOME_PACKET_VERSION,
                }
            )
        elif verification_row is not None and _text(verification_row.get("verification_status", "")).lower() == "verified":
            paper_packets.append(
                {
                    "candidate_id": candidate_id,
                    "lane": WIZARD_LANE,
                    "paper_venue": (_text(row.get("exchange", "")) or "dydx").lower(),
                    "submission_timestamp": _now(),
                    "entry": "",
                    "exit": "",
                    "hold_duration": _text(verification_row.get("local_closed_trades", "")) or _text(verification_row.get("local_trades", "")),
                    "realized_return": _safe_float(verification_row.get("local_total_return", 0.0)),
                    "drawdown": _safe_float(verification_row.get("local_max_drawdown", 0.0)),
                    "slippage_cost_assumptions": json.dumps(
                        {
                            "acceptance": _text(verification_row.get("acceptance", "")),
                            "acceptance_reason": _text(verification_row.get("acceptance_reason", "")),
                            "cost_comparison_path": _text(verification_row.get("cost_comparison_path", "")),
                        },
                        sort_keys=True,
                    ),
                    "result_status": "audit_only",
                    "verification_status": "verified",
                    "outcome_evidence_path": _text(verification_row.get("summary_path", "")) or "reports/active/wizard_local_verification_batch.csv",
                    "provenance": "wizard_local_verification_audit",
                    "schema_version": PAPER_OUTCOME_PACKET_VERSION,
                }
            )

    return _write_lane_outputs(root, WIZARD_LANE, packets, forward_walk_packets, paper_packets, fold_rows=fold_rows)


def build_native_specialist_lane(root: Path = ROOT) -> CommandResult:
    pair_universe = _native_discovery_frame(root)
    journal = _read_csv(root / "reports" / "paper_trading_journal.csv")
    packets = []
    forward_walk_packets = []
    paper_packets = []
    fold_rows = []
    for _, row in pair_universe.iterrows():
        pair = str(row.get("pair", ""))
        strategy_family = _native_strategy_family(row)
        strategy_mode = _native_strategy_mode(row)
        candidate_id = _candidate_id(NATIVE_LANE, pair, strategy_family or str(row.get("decision_bucket", "")), "native")
        blocker_state = _native_blocker_state(row)
        packets.append(
            {
                "candidate_id": candidate_id,
                "lane": NATIVE_LANE,
                "source_type": _native_source_type(row),
                "source_path": "data/processed/pair_universe.csv",
                "pair": pair,
                "venue": str(row.get("best_execution_venue", "dydx") or "dydx").lower(),
                "detection_timestamp": _now(),
                "timeframe": str(row.get("available_timeframes", "")),
                "setup_identity": candidate_id,
                "setup_rank": 1,
                "setup_role": "primary",
                "strategy_family": strategy_family,
                "strategy_mode": strategy_mode,
                "normalized_feature_bundle_ref": str(row.get("evidence_path", "data/processed/pair_universe.csv")),
                "regime_snapshot": str(row.get("decision_reason", "")),
                "confidence": _native_confidence(row),
                "blocker_state": blocker_state,
                "backtest_summary_ref": str(row.get("evidence_path", "data/processed/pair_universe.csv")),
                "forward_walk_summary_ref": f"reports/brain/native_forward_walk.csv#{candidate_id}",
                "paper_outcome_ref": f"reports/brain/native_paper_outcomes.csv#{candidate_id}" if _paper_journal_matches(journal, pair) else "",
                "provenance": "native_pair_universe",
                "schema_version": CANDIDATE_SETUP_PACKET_VERSION,
            }
        )
        walk_metrics, walk_folds = _native_forward_walk_metrics(root, candidate_id, row)
        fold_rows.extend(walk_folds)
        forward_walk_packets.append(
            {
                "candidate_id": candidate_id,
                "lane": NATIVE_LANE,
                "rolling_split_definition": str(walk_metrics.get("rolling_split_definition", "native_pair_universe_proxy")),
                "oos_sharpe": walk_metrics.get("oos_sharpe", ""),
                "oos_profit_factor": walk_metrics.get("oos_profit_factor", ""),
                "oos_max_drawdown": walk_metrics.get("oos_max_drawdown", ""),
                "oos_trade_count": walk_metrics.get("oos_trade_count", 0),
                "stability_metrics": json.dumps(
                    {"decision_bucket": str(row.get("decision_bucket", "")), "fold_count": int(walk_metrics.get("fold_count", 0))},
                    sort_keys=True,
                ),
                "forward_walk_status": str(walk_metrics.get("forward_walk_status", "missing")),
                "blocker_reason": str(walk_metrics.get("blocker_reason", "")),
                "dataset_provenance": str(row.get("selected_path", "data/processed/pair_universe.csv")),
                "run_manifest_ref": str(row.get("evidence_path", row.get("selected_path", ""))),
                "provenance": "native_pair_universe_proxy",
                "schema_version": FORWARD_WALK_PACKET_VERSION,
            }
        )
        journal_row = _match_paper_journal_row(journal, pair)
        if journal_row is not None:
            journal_verification = _paper_journal_verification_status(journal_row)
            paper_packets.append(
                {
                    "candidate_id": candidate_id,
                    "lane": NATIVE_LANE,
                    "paper_venue": str(row.get("best_execution_venue", "dydx") or "dydx").lower(),
                    "submission_timestamp": str(journal_row.get("timestamp_utc", _now())),
                    "entry": "",
                    "exit": "",
                    "hold_duration": "",
                    "realized_return": _safe_float(journal_row.get("realized_return", "")),
                    "drawdown": "",
                    "slippage_cost_assumptions": str(journal_row.get("intents_json", "")),
                    "result_status": str(journal_row.get("plan_status", "paper_submitted")),
                    "verification_status": journal_verification,
                    "outcome_evidence_path": "reports/paper_trading_journal.csv",
                    "provenance": "paper_journal",
                    "schema_version": PAPER_OUTCOME_PACKET_VERSION,
                }
            )
    return _write_lane_outputs(root, NATIVE_LANE, packets, forward_walk_packets, paper_packets, fold_rows=fold_rows)


def build_shadow_rl_lanes(root: Path = ROOT) -> CommandResult:
    wizard = _read_csv(root / "reports" / "brain" / "wizard_candidate_packets.csv")
    native = _read_csv(root / "reports" / "brain" / "native_candidate_packets.csv")
    rows = []
    rows.extend(_shadow_rows_for_lane(wizard, WIZARD_LANE))
    rows.extend(_shadow_rows_for_lane(native, NATIVE_LANE))
    frame = pd.DataFrame(
        rows,
        columns=[
            "candidate_id",
            "lane",
            "pair",
            "setup_identity",
            "strategy_mode",
            "forward_walk_summary_ref",
            "paper_outcome_ref",
            "state_summary",
            "recommended_action",
            "shadow_only",
            "promotion_authority",
            "evidence_path",
        ],
    )
    path = root / "reports" / "brain" / "lane_shadow_rl.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"lane_shadow_rl": path}, summary={"rows": int(len(frame))})


def build_overall_brain(root: Path = ROOT) -> CommandResult:
    wizard = _read_csv(root / "reports" / "brain" / "wizard_candidate_packets.csv")
    native = _read_csv(root / "reports" / "brain" / "native_candidate_packets.csv")
    wizard_fw = _read_csv(root / "reports" / "brain" / "wizard_forward_walk.csv")
    native_fw = _read_csv(root / "reports" / "brain" / "native_forward_walk.csv")

    rows = []
    all_pairs = sorted(set(wizard.get("pair", pd.Series(dtype=str)).astype(str)) | set(native.get("pair", pd.Series(dtype=str)).astype(str)))
    for pair in all_pairs:
        wizard_row = _match_lane_pair_row(wizard, pair, prefer_primary=True)
        native_row = _match_lane_pair_row(native, pair, prefer_primary=True)
        wizard_support = _lane_support(wizard_row, wizard_fw)
        native_support = _lane_support(native_row, native_fw)
        if wizard_support and native_support:
            relation = "both_support"
        elif (not wizard_support) and (not native_support):
            relation = "both_reject"
        elif wizard_support:
            relation = "wizard_only"
        else:
            relation = "native_only"
        arbitration = _arbitration_decision(relation, wizard_row, native_row)
        rows.append(
            {
                "pair": pair,
                "wizard_candidate_id": str(wizard_row.get("candidate_id", "")) if wizard_row is not None else "",
                "wizard_setup_identity": str(wizard_row.get("setup_identity", "")) if wizard_row is not None else "",
                "native_candidate_id": str(native_row.get("candidate_id", "")) if native_row is not None else "",
                "native_setup_identity": str(native_row.get("setup_identity", "")) if native_row is not None else "",
                "agreement_status": relation,
                "wizard_support": wizard_support,
                "native_support": native_support,
                "arbitration_decision": arbitration,
                "orchestrator_status": "comparison_only" if relation in {"wizard_only", "native_only"} else ("aligned" if relation == "both_support" else "blocked"),
                "evidence_path": ";".join([value for value in [
                    "reports/brain/wizard_candidate_packets.csv" if wizard_row is not None else "",
                    "reports/brain/native_candidate_packets.csv" if native_row is not None else "",
                ] if value]),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "pair",
            "wizard_candidate_id",
            "wizard_setup_identity",
            "native_candidate_id",
            "native_setup_identity",
            "agreement_status",
            "wizard_support",
            "native_support",
            "arbitration_decision",
            "orchestrator_status",
            "evidence_path",
        ],
    )
    path = root / "reports" / "brain" / "overall_brain_summary.csv"
    _write_csv(frame, path)

    slate_rows = []
    for idx, row in frame.iterrows():
        slate_rows.append(
            {
                "decision_cycle_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "pair": str(row.get("pair", "")),
                "chosen_lane": str(row.get("arbitration_decision", "")),
                "agreement_status": str(row.get("agreement_status", "")),
                "candidate_rank": int(idx + 1),
            }
        )
    slate_path = root / "reports" / "brain" / "candidate_slate_memory.csv"
    _write_csv(pd.DataFrame(slate_rows), slate_path)
    return CommandResult(paths={"overall_brain_summary": path, "candidate_slate_memory": slate_path}, summary={"rows": int(len(frame))})


def build_promotion_readiness(root: Path = ROOT) -> CommandResult:
    wizard = _read_csv(root / "reports" / "brain" / "wizard_candidate_packets.csv")
    native = _read_csv(root / "reports" / "brain" / "native_candidate_packets.csv")
    wizard_fw = _read_csv(root / "reports" / "brain" / "wizard_forward_walk.csv")
    native_fw = _read_csv(root / "reports" / "brain" / "native_forward_walk.csv")
    overall = _read_csv(root / "reports" / "brain" / "overall_brain_summary.csv")
    paper = _read_csv(root / "reports" / "rl" / "base_rl_paper_handoff_status.csv")

    rows = []
    for frame, fw_frame, lane in ((wizard, wizard_fw, WIZARD_LANE), (native, native_fw, NATIVE_LANE)):
        for _, row in frame.iterrows():
            candidate_id = str(row.get("candidate_id", ""))
            pair = str(row.get("pair", ""))
            native_model_support = _native_pair_model_support(root, pair) if lane == NATIVE_LANE else {}
            forward_walk = _match_row(fw_frame, candidate_id, "candidate_id")
            ok, reason = candidate_paper_credibility_status(
                row.to_dict(),
                forward_walk_packet=forward_walk.to_dict() if forward_walk is not None else None,
                local_verification_passed=(lane != WIZARD_LANE or "local_verification_missing" not in _text(row.get("blocker_state", ""))),
            )
            overall_row = _match_row(overall, pair, "pair")
            paper_status = str(paper.get("status", pd.Series(["research_only"])).iloc[0]) if not paper.empty else "research_only"
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "lane": lane,
                    "pair": pair,
                    "setup_identity": str(row.get("setup_identity", "")),
                    "setup_role": str(row.get("setup_role", "")),
                    "promotion_stage": _promotion_stage(ok, overall_row, paper_status),
                    "forward_walk_required": True,
                    "paper_credible": ok,
                    "blocker": "" if ok else reason,
                    "orchestrator_decision": str(overall_row.get("arbitration_decision", "")) if overall_row is not None else "",
                    "paper_status": paper_status,
                    "venue": _text(row.get("venue", "")) if lane == NATIVE_LANE else "dydx",
                    "route_priority": 0 if (lane == WIZARD_LANE or _text(row.get("venue", "")) == "dydx") else 1,
                    "model_support_status": _text(native_model_support.get("support_status", "")),
                    "model_taken_trades": _safe_int(native_model_support.get("taken_trades", 0)),
                }
            )
    frame = pd.DataFrame(
        rows,
        columns=[
            "candidate_id",
            "lane",
            "pair",
            "setup_identity",
            "setup_role",
            "promotion_stage",
            "forward_walk_required",
            "paper_credible",
            "blocker",
            "orchestrator_decision",
            "paper_status",
            "venue",
            "route_priority",
            "model_support_status",
            "model_taken_trades",
        ],
    )
    path = root / "reports" / "brain" / "promotion_ladder.csv"
    _write_csv(frame, path)

    orchestrator_status_path = root / "reports" / "brain" / "orchestrator_status.csv"
    preferred = _preferred_promotion_row(frame)
    paper_eligible = (
        frame.get("promotion_stage", pd.Series(dtype=object)).astype(str).eq("paper_eligible").any()
        if not frame.empty
        else False
    )
    paper_gate_status = str(paper.get("status", pd.Series(["research_only"])).iloc[0]) if not paper.empty else "research_only"
    paper_gate_blocker = _text(paper.get("blocker", pd.Series([""])).iloc[0]) if not paper.empty else ""
    paper_gate_action = _text(paper.get("next_action", pd.Series([""])).iloc[0]) if not paper.empty else ""
    orchestrator_ready = bool(paper_eligible and paper_gate_status == "paper_authorized" and not paper_gate_blocker)
    orchestrator_blocker = ""
    next_action = "improve forward walk, lane support, and paper outcomes"
    if not orchestrator_ready:
        acceptance_blocker, _, acceptance_action = _strategy_acceptance_followup(root)
        if preferred is not None and _text(preferred.get("paper_status", "")) == "research_only":
            orchestrator_blocker = paper_gate_blocker or acceptance_blocker or "global_paper_handoff_blocked"
            next_action = paper_gate_action or acceptance_action or next_action
        elif preferred is not None and _text(preferred.get("paper_status", "")) not in {"", "paper_authorized"}:
            orchestrator_blocker = paper_gate_blocker or _text(preferred.get("paper_status", "")) or "global_paper_handoff_blocked"
            next_action = paper_gate_action or next_action
        elif not frame.empty and frame.get("paper_credible", pd.Series(dtype=bool)).astype(bool).any():
            orchestrator_blocker = paper_gate_blocker or acceptance_blocker or paper_gate_status or "global_paper_handoff_blocked"
            next_action = paper_gate_action or acceptance_action or next_action
        else:
            orchestrator_blocker = _text(preferred.get("blocker", "")) if preferred is not None else ""
            orchestrator_blocker = orchestrator_blocker or "no_paper_credible_candidates"
    orchestrator_status = pd.DataFrame(
        [
            {
                "area": "orchestrator_status",
                "ready": orchestrator_ready,
                "status": "ready" if orchestrator_ready else "blocked",
                "blocker": "" if orchestrator_ready else orchestrator_blocker,
                "detail": f"rows={len(frame)}",
                "pair": _text(preferred.get("pair", "")) if preferred is not None else "",
                "candidate_id": _text(preferred.get("candidate_id", "")) if preferred is not None else "",
                "setup_identity": _text(preferred.get("setup_identity", "")) if preferred is not None else "",
                "setup_role": _text(preferred.get("setup_role", "")) if preferred is not None else "",
                "setup_status": _text(preferred.get("promotion_stage", "")) if preferred is not None else "",
                "setup_blocker": (_text(preferred.get("blocker", "")) if preferred is not None else "") or orchestrator_blocker,
                "evidence_path": str(path),
                "next_action": next_action,
            }
        ]
    )
    _write_csv(orchestrator_status, orchestrator_status_path)
    return CommandResult(
        paths={"promotion_ladder": path, "orchestrator_status": orchestrator_status_path},
        summary={"rows": int(len(frame)), "paper_ready": int(frame.get("paper_credible", pd.Series(dtype=bool)).astype(bool).sum()) if not frame.empty else 0},
    )


def build_forward_walk_strength_report(root: Path = ROOT) -> CommandResult:
    rows = []
    for lane in (WIZARD_LANE, NATIVE_LANE):
        summary = _read_csv(root / "reports" / "brain" / f"{lane}_forward_walk.csv")
        folds = _read_csv(root / "reports" / "brain" / f"{lane}_forward_walk_folds.csv")
        for _, row in summary.iterrows():
            candidate_id = str(row.get("candidate_id", ""))
            pair = _pair_for_candidate(root, lane, candidate_id)
            packet = _match_row(_read_csv(root / "reports" / "brain" / f"{lane}_candidate_packets.csv"), candidate_id, "candidate_id")
            candidate_folds = folds[folds.get("candidate_id", pd.Series(dtype=str)).astype(str) == candidate_id].copy() if not folds.empty else pd.DataFrame()
            fold_count = int(len(candidate_folds))
            trade_count = _safe_int(row.get("oos_trade_count", 0))
            pf = _safe_float(row.get("oos_profit_factor", 0.0))
            sharpe = _safe_float(row.get("oos_sharpe", 0.0))
            tier = _forward_walk_strength_tier(
                status=str(row.get("forward_walk_status", "")),
                fold_count=fold_count,
                trade_count=trade_count,
                profit_factor=pf,
                sharpe=sharpe,
            )
            evidence_density = _evidence_density_tier(fold_count=fold_count, trade_count=trade_count)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "lane": lane,
                    "pair": pair,
                    "setup_identity": _text(packet.get("setup_identity", "")) if packet is not None else "",
                    "forward_walk_status": str(row.get("forward_walk_status", "")),
                    "strength_tier": tier,
                    "evidence_density": evidence_density,
                    "fold_count": fold_count,
                    "trade_count": trade_count,
                    "oos_profit_factor": pf,
                    "oos_sharpe": sharpe,
                    "threshold_status": "pass" if tier in {"strong", "acceptable"} else "blocked",
                    "blocker": "" if tier in {"strong", "acceptable"} else str(row.get("blocker_reason", "")) or "forward_walk_evidence_too_thin",
                    "evidence_path": f"reports/brain/{lane}_forward_walk.csv",
                }
            )
    frame = pd.DataFrame(
        rows,
        columns=[
            "candidate_id",
            "lane",
            "pair",
            "setup_identity",
            "forward_walk_status",
            "strength_tier",
            "evidence_density",
            "fold_count",
            "trade_count",
            "oos_profit_factor",
            "oos_sharpe",
            "threshold_status",
            "blocker",
            "evidence_path",
        ],
    )
    path = root / "reports" / "brain" / "forward_walk_strength_report.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"forward_walk_strength_report": path}, summary={"rows": int(len(frame))})


def build_native_quality_report(root: Path = ROOT) -> CommandResult:
    discovery = _native_discovery_frame(root)
    packets = _read_csv(root / "reports" / "brain" / "native_candidate_packets.csv")
    forward = _read_csv(root / "reports" / "brain" / "native_forward_walk.csv")
    rows = []
    for _, row in discovery.iterrows():
        pair = str(row.get("pair", ""))
        packet = _match_row(packets, pair, "pair")
        forward_row = _match_row(forward, str(packet.get("candidate_id", "")), "candidate_id") if packet is not None else None
        rows.append(
            {
                "pair": pair,
                "candidate_id": _text(packet.get("candidate_id", "")) if packet is not None else "",
                "setup_identity": _text(packet.get("setup_identity", "")) if packet is not None else "",
                "native_origin_type": _native_origin_type(row),
                "native_origin_detail": ";".join(
                    [
                        value
                        for value in [
                            _text(row.get("native_discovery_status", "")),
                            _text(row.get("dydx_local_discovery_status", "")),
                            _text(row.get("selected_path", "")),
                        ]
                        if value
                    ]
                ),
                "quality_status": "supported" if packet is not None and _text(packet.get("blocker_state", "")) == "" else "blocked",
                "acceptance_score": _safe_float(row.get("acceptance_score", 0.0)),
                "combined_score": _safe_float(row.get("combined_score", 0.0)),
                "local_backtest_score": _safe_float(row.get("local_backtest_score", 0.0)),
                "forward_walk_status": _text(forward_row.get("forward_walk_status", "")) if forward_row is not None else "",
                "forward_walk_pf": _safe_float(forward_row.get("oos_profit_factor", 0.0)) if forward_row is not None else 0.0,
                "forward_walk_trades": _safe_int(forward_row.get("oos_trade_count", 0)) if forward_row is not None else 0,
                "blocker": _text(packet.get("blocker_state", "")) if packet is not None else "native_candidate_missing",
                "evidence_path": _text(row.get("selected_path", "")) or "data/processed/pair_universe.csv",
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "pair",
            "candidate_id",
            "setup_identity",
            "native_origin_type",
            "native_origin_detail",
            "quality_status",
            "acceptance_score",
            "combined_score",
            "local_backtest_score",
            "forward_walk_status",
            "forward_walk_pf",
            "forward_walk_trades",
            "blocker",
            "evidence_path",
        ],
    )
    path = root / "reports" / "brain" / "native_quality_report.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"native_quality_report": path}, summary={"rows": int(len(frame))})


def build_native_candidate_diagnosis(root: Path = ROOT) -> CommandResult:
    discovery = _native_discovery_frame(root)
    packets = _read_csv(root / "reports" / "brain" / "native_candidate_packets.csv")
    rows = []
    for _, row in discovery.iterrows():
        pair = _text(row.get("pair", ""))
        packet = _match_row(packets, pair, "pair")
        candidate_id = _text(packet.get("candidate_id", "")) if packet is not None else _candidate_id(NATIVE_LANE, pair, _native_strategy_family(row), "native")
        setup_identity = _text(packet.get("setup_identity", "")) if packet is not None else candidate_id
        diagnosis, _ = _native_forward_walk_diagnosis(root, candidate_id, row)
        rows.append(
            {
                "pair": pair,
                "candidate_id": candidate_id,
                "setup_identity": setup_identity,
                "lane": NATIVE_LANE,
                "selected_history_source": _text(diagnosis.get("selected_history_source", "")),
                "evaluation_mode": _text(diagnosis.get("evaluation_mode", "")),
                "fold_count": _safe_int(diagnosis.get("fold_count", 0)),
                "nonzero_signal_trade_count": _safe_int(diagnosis.get("nonzero_signal_trade_count", 0)),
                "mean_fold_profit_factor": _safe_float(diagnosis.get("mean_fold_profit_factor", 0.0)),
                "mean_fold_sharpe": _safe_float(diagnosis.get("mean_fold_sharpe", 0.0)),
                "max_fold_drawdown": _safe_float(diagnosis.get("max_fold_drawdown", 0.0)),
                "current_blocker": _text(diagnosis.get("current_blocker", "")),
                "blocker_detail": _text(diagnosis.get("blocker_detail", "")),
                "threshold_reference": _text(diagnosis.get("threshold_reference", "")),
                "next_repair_action": _native_repair_action(diagnosis),
                "evidence_path": _text(diagnosis.get("evidence_path", "")),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "pair",
            "candidate_id",
            "setup_identity",
            "lane",
            "selected_history_source",
            "evaluation_mode",
            "fold_count",
            "nonzero_signal_trade_count",
            "mean_fold_profit_factor",
            "mean_fold_sharpe",
            "max_fold_drawdown",
            "current_blocker",
            "blocker_detail",
            "threshold_reference",
            "next_repair_action",
            "evidence_path",
        ],
    )
    path = root / "reports" / "brain" / "native_candidate_diagnosis.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"native_candidate_diagnosis": path}, summary={"rows": int(len(frame))})


def build_wizard_candidate_repair_report(root: Path = ROOT) -> CommandResult:
    packets = _read_csv(root / "reports" / "brain" / "wizard_candidate_packets.csv")
    hypotheses = _read_csv(root / "reports" / "active" / "wizard_hypotheses.csv")
    diagnostics = _read_csv(root / "reports" / "active" / "wizard_diagnostic_confirmation.csv")
    parity = _read_csv(root / "reports" / "active" / "wizard_vs_local_parity_report.csv")
    verification = _read_csv(root / "reports" / "active" / "wizard_local_verification_batch.csv")
    hourly_targets = _read_csv(root / "reports" / "brain" / "wizard_hourly_repair_targets.csv")
    scanner_dependency = _read_csv(root / "reports" / "active" / "wizard_scanner_dependency_capture.csv")
    rows = []
    for _, row in packets.iterrows():
        pair = _text(row.get("pair", ""))
        setup_identity = _text(row.get("setup_identity", ""))
        hypothesis_row = _match_setup_or_pair_row(hypotheses, pair, setup_identity)
        diagnostic_row = _match_setup_or_pair_row(diagnostics, pair, setup_identity)
        parity_row = _match_setup_or_pair_row(parity, pair, setup_identity)
        verification_row = _match_setup_or_pair_row(verification, pair, setup_identity)
        blocker_text = _text(row.get("blocker_state", ""))
        diagnostic_blocker_text = _text(diagnostic_row.get("diagnostic_blocker", "")) if diagnostic_row is not None else blocker_text
        hourly_row = _match_row(hourly_targets, _text(row.get("candidate_id", "")), "candidate_id")
        scanner_dependency_row = _match_wizard_scanner_dependency_row(
            scanner_dependency,
            pair=pair,
            timeframe=_text(row.get("timeframe", "")),
            strategy_mode=(
                _text(row.get("strategy_mode", ""))
                or _text(row.get("exact_mode", ""))
                or (_text(hourly_row.get("matched_hourly_strategy", "")) if hourly_row is not None else "")
            ),
        )
        effective_blocker = _wizard_effective_repair_blocker(
            blocker_text=blocker_text,
            diagnostic_blocker_text=diagnostic_blocker_text,
            hourly_row=hourly_row,
            scanner_dependency_row=scanner_dependency_row,
        )
        unresolved_diagnostic_blocker = diagnostic_blocker_text
        if scanner_dependency_row is not None:
            unresolved_diagnostic_blocker = ";".join(
                part
                for part in diagnostic_blocker_text.split(";")
                if part.strip() and part.strip() != "missing_correlation"
            )
        # Keep the full diagnostic chain visible even when the next repair is a single primary action.
        blocker_surface = unresolved_diagnostic_blocker or effective_blocker or blocker_text
        rows.append(
            {
                "pair": pair,
                "candidate_id": _text(row.get("candidate_id", "")),
                "setup_identity": setup_identity,
                "lane": WIZARD_LANE,
                "exact_mode_capture_status": _wizard_exact_mode_capture_status(row, effective_blocker, hourly_row=hourly_row),
                "correlation_status": _wizard_blocker_status(blocker_surface, "missing_correlation"),
                "ecm_status": _wizard_blocker_status(blocker_surface, "missing_ecm"),
                "parity_status": _text(parity_row.get("parity_status", "")) if parity_row is not None else "missing",
                "local_verification_status": _text(verification_row.get("verification_status", "")) if verification_row is not None else "missing",
                "hypothesis_status": _text(hypothesis_row.get("hypothesis_status", "")) if hypothesis_row is not None else "missing",
                "diagnostic_blocker": diagnostic_blocker_text if diagnostic_row is not None else "",
                "unresolved_diagnostic_blocker": unresolved_diagnostic_blocker if diagnostic_row is not None else "",
                "current_blocker": effective_blocker or "wizard_candidate_clear",
                "next_repair_action": _wizard_repair_action(effective_blocker),
                "evidence_path": ";".join(
                    [
                        value
                        for value in [
                            _text(row.get("source_path", "")),
                            "reports/active/wizard_hourly_database/wizard_hourly_candidate_queue.csv" if hourly_row is not None else "",
                            "reports/active/wizard_scanner_dependency_capture.csv" if scanner_dependency_row is not None else "",
                            "reports/active/wizard_diagnostic_confirmation.csv" if diagnostic_row is not None else "",
                            _text(verification_row.get("summary_path", "")) if verification_row is not None else "reports/active/wizard_local_verification_batch.csv",
                        ]
                        if value
                    ]
                ),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "pair",
            "candidate_id",
            "setup_identity",
            "lane",
            "exact_mode_capture_status",
            "correlation_status",
            "ecm_status",
            "parity_status",
            "local_verification_status",
            "hypothesis_status",
            "diagnostic_blocker",
            "unresolved_diagnostic_blocker",
            "current_blocker",
            "next_repair_action",
            "evidence_path",
        ],
    )
    path = root / "reports" / "brain" / "wizard_candidate_repair_report.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"wizard_candidate_repair_report": path}, summary={"rows": int(len(frame))})


def build_wizard_hourly_priority_capture_queue(root: Path = ROOT) -> CommandResult:
    repair = _read_csv(root / "reports" / "brain" / "wizard_candidate_repair_report.csv")
    exact_mode_queue = _read_csv(root / "reports" / "active" / "wizard_exact_mode_capture_queue.csv")
    hourly_targets = _read_csv(root / "reports" / "brain" / "wizard_hourly_repair_targets.csv")
    verification = _read_csv(root / "reports" / "active" / "wizard_local_verification_batch.csv")
    rows = []
    for _, row in repair.iterrows():
        pair = _text(row.get("pair", ""))
        queue_pair = _wizard_queue_pair_from_setup(_text(row.get("setup_identity", "")), pair)
        queue_row = _match_row(exact_mode_queue, queue_pair, "pair")
        if queue_row is None:
            queue_row = _match_row(exact_mode_queue, pair, "pair")
        hourly_row = _match_row(hourly_targets, _text(row.get("candidate_id", "")), "candidate_id")
        blocker = _text(row.get("current_blocker", ""))
        diagnostic_blocker = (
            _text(row.get("unresolved_diagnostic_blocker", ""))
            or _text(row.get("diagnostic_blocker", ""))
        )
        capture_blocker_surface = diagnostic_blocker or blocker
        needs_exact_mode = "missing_exact_mode" in capture_blocker_surface
        verification_row = _match_setup_or_pair_row(verification, queue_pair, _text(row.get("setup_identity", "")))
        capture_status = _wizard_capture_completeness_status(
            blocker=blocker,
            verification_row=verification_row,
        )
        live_capture_status = _wizard_live_capture_status(
            needs_exact_mode=needs_exact_mode,
            hourly_row=hourly_row,
        )
        capture_priority = 100
        if needs_exact_mode:
            capture_priority -= 40
        if "missing_correlation" in blocker:
            capture_priority -= 15
        if "missing_ecm" in blocker:
            capture_priority -= 15
        if live_capture_status == "live_expired":
            capture_priority += 120
        elif live_capture_status == "pair_live_but_exact_setup_missing":
            capture_priority += 25
        if (
            not needs_exact_mode
            and _text(row.get("exact_mode_capture_status", "")) == "captured"
            and hourly_row is not None
            and _text(hourly_row.get("hourly_match_status", "")) == "matched"
        ):
            capture_priority += 25
        if hourly_row is not None and _text(hourly_row.get("candidate_status", "")) == "review_now":
            capture_priority -= 10
        if hourly_row is not None and _text(hourly_row.get("review_status", "")) in {"new", "changed"}:
            capture_priority -= 5
        if _text(row.get("setup_role", "")) == "primary":
            capture_priority -= 10
        if capture_status == "verification_ready":
            capture_priority += 20
        if queue_row is not None:
            capture_priority -= _safe_int(queue_row.get("priority_rank", 0))
        display_capture_status = "live_expired" if live_capture_status == "live_expired" and needs_exact_mode else capture_status
        recommended_action = (
            "skip_stale_pair_take_next_live_candidate"
            if needs_exact_mode and live_capture_status == "live_expired"
            else "capture_exact_mode_then_pair_detail_panels"
            if needs_exact_mode
            else "capture_missing_pair_detail_panels"
            if capture_status in {"pair_detail_panel_missing", "local_inputs_missing", "verification_blocked", "capture_state_unknown"}
            else "no_capture_needed_use_repair_loop"
        )
        rows.append(
            {
                "capture_priority_rank": 0,
                "pair": pair,
                "candidate_id": _text(row.get("candidate_id", "")),
                "setup_identity": _text(row.get("setup_identity", "")),
                "current_blocker": blocker,
                "needs_exact_mode_capture": needs_exact_mode,
                "needs_correlation_capture": "missing_correlation" in capture_blocker_surface,
                "needs_ecm_capture": "missing_ecm" in capture_blocker_surface,
                "live_capture_status": live_capture_status,
                "hourly_pair_recommended": bool(hourly_row.get("pair_recommended_this_hour", False)) if hourly_row is not None else False,
                "hourly_match_status": _text(hourly_row.get("hourly_match_status", "")) if hourly_row is not None else "",
                "wizard_queue_priority": _safe_int(queue_row.get("priority_rank", 0)) if queue_row is not None else 0,
                "queue_pair": queue_pair,
                "pair_history_path": _text(queue_row.get("pair_history_path", "")) if queue_row is not None else "",
                "capture_completeness_status": display_capture_status,
                "recommended_capture_action": recommended_action,
                "capture_priority_score": capture_priority,
                "evidence_path": _text(queue_row.get("source_path", "")) if queue_row is not None else "reports/brain/wizard_candidate_repair_report.csv",
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["capture_priority_score", "pair"], ascending=[True, True]).reset_index(drop=True)
        frame["capture_priority_rank"] = range(1, len(frame) + 1)
    path = root / "reports" / "active" / "wizard_hourly_priority_capture_queue.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"wizard_hourly_priority_capture_queue": path}, summary={"rows": int(len(frame))})


def build_native_focus_repair_report(root: Path = ROOT) -> CommandResult:
    diagnosis = _read_csv(root / "reports" / "brain" / "native_candidate_diagnosis.csv")
    quality = _read_csv(root / "reports" / "brain" / "native_quality_report.csv")
    promotion = _read_csv(root / "reports" / "brain" / "promotion_ladder.csv")
    forward = _read_csv(root / "reports" / "brain" / "native_forward_walk.csv")
    packets = _read_csv(root / "reports" / "brain" / "native_candidate_packets.csv")
    paper_gate = _read_csv(root / "reports" / "rl" / "base_rl_paper_handoff_status.csv")
    model_gate_pairs = _read_csv(root / "reports" / "ml" / "model_gate_pair_support_report.csv")
    rows = []
    acceptance_blocker, acceptance_detail, acceptance_action = _strategy_acceptance_followup(root)
    paper_gate_blocker = _text(paper_gate.get("blocker", pd.Series([""])).iloc[0]) if not paper_gate.empty else ""
    paper_gate_action = _text(paper_gate.get("next_action", pd.Series([""])).iloc[0]) if not paper_gate.empty else ""
    paper_route_action, paper_route_detail = _paper_route_followup_context(root, paper_gate_blocker)
    model_gate_action, model_gate_detail = _model_gate_followup(root)
    support_rank = {
        "strong_model_support": 0,
        "weak_model_support": 1,
        "pair_missing_from_model_predictions": 2,
        "no_model_support": 3,
        "model_predictions_missing": 4,
    }
    native_rows = promotion[promotion.get("lane", pd.Series(dtype=object)).astype(str) == NATIVE_LANE].copy() if not promotion.empty else pd.DataFrame()
    ranked_targets: list[tuple[tuple[object, ...], str]] = []
    for _, promotion_row in native_rows.iterrows():
        pair = _text(promotion_row.get("pair", ""))
        packet_row = _match_row(packets, pair, "pair")
        venue = _text(packet_row.get("venue", "")) if packet_row is not None else ""
        model_support = _native_pair_model_support(root, pair)
        diag = _match_row(diagnosis, pair, "pair")
        ranked_targets.append(
            (
                (
                    0 if venue == "dydx" else 1,
                    0 if bool(promotion_row.get("paper_credible", False)) else 1,
                    support_rank.get(_text(model_support.get("support_status", "")), 9),
                    -_safe_int(model_support.get("taken_trades", 0)),
                    -_safe_float(model_support.get("profit_factor", 0.0)),
                    -_safe_float(diag.get("mean_fold_sharpe", 0.0)) if diag is not None else 0.0,
                    pair,
                ),
                pair,
            )
        )
    targets = [pair for _, pair in sorted(ranked_targets)[:2]]
    if not targets:
        targets = ["BTC-USD-HYPE-USD", "SOL-USD-HYPE-USD"]
    for pair in targets:
        model_support = _native_pair_model_support(root, pair)
        diag = _match_row(diagnosis, pair, "pair")
        quality_row = _match_row(quality, pair, "pair")
        model_gate_pair_row = _match_row(model_gate_pairs, pair, "pair")
        promotion_row = _match_row(promotion[promotion.get("lane", pd.Series(dtype=object)).astype(str) == NATIVE_LANE], pair, "pair") if not promotion.empty else None
        candidate_id = _text(diag.get("candidate_id", "")) if diag is not None else ""
        native_forward = forward[forward.get("lane", pd.Series(dtype=object)).astype(str) == NATIVE_LANE] if not forward.empty else pd.DataFrame()
        forward_row = _match_row(native_forward, pair, "pair") if not native_forward.empty and "pair" in native_forward.columns else None
        if forward_row is None and candidate_id:
            forward_row = _match_row(native_forward, candidate_id, "candidate_id") if not native_forward.empty else None
        history_path = _text(diag.get("selected_history_source", "")) if diag is not None else ""
        focused_5m_path = str(root / "data" / "raw" / "pair_details" / f"pair_{pair.split('-USD-')[0].lower()}_hype_5mins_dydx_candles_derived_history.json")
        current_blocker = _text(diag.get("current_blocker", "")) if diag is not None else ""
        blocker_detail = _text(diag.get("blocker_detail", "")) if diag is not None else ""
        next_action = (
            _text(diag.get("next_repair_action", "")) if diag is not None and _text(diag.get("next_repair_action", "")) else (
                "rerun_native_forward_walk_with_focused_5m_dydx_history"
                if Path(focused_5m_path).exists() and history_path == focused_5m_path
                else "switch_native_manifest_to_focused_5m_dydx_history_then_rerun"
            )
        )
        if promotion_row is not None and bool(promotion_row.get("paper_credible", False)) and _text(promotion_row.get("promotion_stage", "")) != "paper_eligible":
            if acceptance_blocker:
                current_blocker = acceptance_blocker
                blocker_detail = acceptance_detail or (
                    f"candidate evidence is credible but strategy acceptance remains {_text(promotion_row.get('paper_status', 'research_only'))}"
                )
                next_action = acceptance_action or _paper_route_followup_action(promotion_row)
            else:
                current_blocker = paper_gate_blocker or _text(promotion_row.get("paper_status", "")) or "global_paper_handoff_blocked"
                blocker_detail = (
                    f"candidate evidence is credible but paper route remains {_text(promotion_row.get('paper_status', 'research_only'))}; "
                    f"paper_gate={paper_gate_blocker or 'clear'}"
                )
                if paper_gate_blocker == "model_gate_not_accepted" and model_gate_detail:
                    blocker_detail = f"{blocker_detail}; {model_gate_detail}"
                if paper_gate_blocker == "model_gate_not_accepted":
                    blocker_detail = (
                        f"{blocker_detail}; pair_model_support={_text(model_support.get('support_status', ''))}; "
                        f"pair_model_taken_trades={_safe_int(model_support.get('taken_trades', 0))}; "
                        f"pair_model_take_rate={_safe_float(model_support.get('take_rate', 0.0))}; "
                        f"pair_model_profit_factor={_safe_float(model_support.get('profit_factor', 0.0))}; "
                        f"pair_model_mean_return={_safe_float(model_support.get('mean_return', 0.0))}"
                    )
                if paper_route_detail:
                    blocker_detail = f"{blocker_detail}; {paper_route_detail}"
                next_action = (
                    (
                        (_text(model_gate_pair_row.get("recommended_repair_action", "")) if model_gate_pair_row is not None else "") or (
                            "preserve pair-specific model support while improving global model gate acceptance"
                            if _text(model_support.get("support_status", "")) == "strong_model_support"
                            else model_gate_action
                        )
                    )
                    if paper_gate_blocker == "model_gate_not_accepted"
                    else paper_route_action or paper_gate_action or _paper_route_followup_action(promotion_row)
                )
        rows.append(
            {
                "pair": pair,
                "candidate_id": candidate_id,
                "current_history_path": history_path,
                "focused_5m_dydx_path": focused_5m_path,
                "focused_5m_dydx_exists": Path(focused_5m_path).exists(),
                "current_blocker": current_blocker,
                "blocker_detail": blocker_detail,
                "quality_status": _text(quality_row.get("quality_status", "")) if quality_row is not None else "",
                "current_history_matches_focused_5m": history_path == focused_5m_path,
                "evaluation_mode": _text(diag.get("evaluation_mode", "")) if diag is not None else "",
                "fold_count": _safe_int(diag.get("fold_count", 0)) if diag is not None else 0,
                "nonzero_signal_trade_count": _safe_int(diag.get("nonzero_signal_trade_count", 0)) if diag is not None else 0,
                "mean_fold_profit_factor": _safe_float(diag.get("mean_fold_profit_factor", 0.0)) if diag is not None else 0.0,
                "mean_fold_sharpe": _safe_float(diag.get("mean_fold_sharpe", 0.0)) if diag is not None else 0.0,
                "max_fold_drawdown": _safe_float(diag.get("max_fold_drawdown", 0.0)) if diag is not None else 0.0,
                "forward_walk_status": _text(forward_row.get("forward_walk_status", "")) if forward_row is not None else "",
                "forward_walk_blocker": _text(forward_row.get("blocker_reason", "")) if forward_row is not None else "",
                "pair_model_support_status": _text(model_support.get("support_status", "")),
                "pair_model_rows": _safe_int(model_support.get("rows", 0)),
                "pair_model_taken_trades": _safe_int(model_support.get("taken_trades", 0)),
                "pair_model_take_rate": _safe_float(model_support.get("take_rate", 0.0)),
                "pair_model_profit_factor": _safe_float(model_support.get("profit_factor", 0.0)),
                "pair_model_mean_return": _safe_float(model_support.get("mean_return", 0.0)),
                "model_gate_anchor_rank": _safe_int(model_gate_pair_row.get("model_gate_anchor_rank", 0)) if model_gate_pair_row is not None else 0,
                "model_gate_repair_action": _text(model_gate_pair_row.get("recommended_repair_action", "")) if model_gate_pair_row is not None else "",
                "promotion_stage": _text(promotion_row.get("promotion_stage", "")) if promotion_row is not None else "",
                "paper_status": _text(promotion_row.get("paper_status", "")) if promotion_row is not None else "",
                "next_repair_action": next_action,
            }
        )
    frame = pd.DataFrame(rows)
    path = root / "reports" / "brain" / "native_focus_repair_report.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"native_focus_repair_report": path}, summary={"rows": int(len(frame))})


def build_candidate_repair_loop(root: Path = ROOT) -> CommandResult:
    promotion = _read_csv(root / "reports" / "brain" / "promotion_ladder.csv")
    native_diag = _read_csv(root / "reports" / "brain" / "native_candidate_diagnosis.csv")
    native_focus = _read_csv(root / "reports" / "brain" / "native_focus_repair_report.csv")
    wizard_repair = _read_csv(root / "reports" / "brain" / "wizard_candidate_repair_report.csv")
    wizard_hourly_targets = _read_csv(root / "reports" / "brain" / "wizard_hourly_repair_targets.csv")
    native_quality = _read_csv(root / "reports" / "brain" / "native_quality_report.csv")
    paper_gate = _read_csv(root / "reports" / "rl" / "base_rl_paper_handoff_status.csv")
    acceptance_blocker, _, acceptance_action = _strategy_acceptance_followup(root)
    paper_gate_blocker = _text(paper_gate.get("blocker", pd.Series([""])).iloc[0]) if not paper_gate.empty else ""
    paper_gate_action = _text(paper_gate.get("next_action", pd.Series([""])).iloc[0]) if not paper_gate.empty else ""
    paper_route_action, _ = _paper_route_followup_context(root, paper_gate_blocker)
    model_gate_action, _ = _model_gate_followup(root)
    if promotion.empty:
        frame = pd.DataFrame(
            columns=[
                "repair_priority_rank",
                "pair",
                "candidate_id",
                "setup_identity",
                "lane",
                "current_blocker",
                "repair_action",
                "rerun_required",
                "post_rerun_promotion_result",
                "run_lineage_ref",
                "hourly_review_status",
                "hourly_candidate_status",
                "hourly_pair_recommended",
            ]
        )
    else:
        selected = []
        native_rows = promotion[promotion.get("lane", pd.Series(dtype=object)).astype(str) == NATIVE_LANE].copy()
        if not native_rows.empty and not native_focus.empty:
            focus_rows = native_focus.copy()
            focus_rows["_support_rank"] = focus_rows.get("pair_model_support_status", pd.Series(dtype=object)).astype(str).map(
                {
                    "strong_model_support": 0,
                    "weak_model_support": 1,
                    "pair_missing_from_model_predictions": 2,
                    "no_model_support": 3,
                    "model_predictions_missing": 4,
                }
            ).fillna(9)
            focus_rows = focus_rows.sort_values(
                ["_support_rank", "pair_model_taken_trades", "pair_model_profit_factor", "mean_fold_sharpe"],
                ascending=[True, False, False, False],
            )
            for pair in focus_rows.get("pair", pd.Series(dtype=object)).astype(str).tolist():
                promotion_row = _match_row(native_rows, pair, "pair")
                if promotion_row is not None:
                    selected.append(promotion_row)
                if len(selected) >= 2:
                    break
        if not native_rows.empty and len(selected) < 2:
            native_rows["_combined_score"] = native_rows.get("pair", pd.Series(dtype=object)).map(
                lambda pair: _safe_float(_match_row(native_quality, _text(pair), "pair").get("combined_score", 0.0)) if _match_row(native_quality, _text(pair), "pair") is not None else 0.0
            )
            native_rows = native_rows.sort_values(["paper_credible", "_combined_score"], ascending=[False, False])
            chosen_native_ids = {_text(row.get("candidate_id", "")) for row in selected}
            fallback_rows = native_rows[
                ~native_rows.get("candidate_id", pd.Series(dtype=object)).astype(str).isin(chosen_native_ids)
            ].copy()
            selected.extend([fallback_rows.iloc[idx] for idx in range(min(2 - len(selected), len(fallback_rows)))])
        wizard_rows = promotion[promotion.get("lane", pd.Series(dtype=object)).astype(str) == WIZARD_LANE].copy()
        if not wizard_rows.empty:
            wizard_rows["_primary"] = wizard_rows.get("setup_role", pd.Series(dtype=object)).astype(str).eq("primary")
            wizard_rows["_hourly_recommended"] = wizard_rows.get("candidate_id", pd.Series(dtype=object)).map(
                lambda candidate_id: _wizard_hourly_target_value(wizard_hourly_targets, _text(candidate_id), "pair_recommended_this_hour")
            )
            wizard_rows["_hourly_changed"] = wizard_rows.get("candidate_id", pd.Series(dtype=object)).map(
                lambda candidate_id: _wizard_hourly_target_value(wizard_hourly_targets, _text(candidate_id), "review_status") in {"new", "changed"}
            )
            wizard_rows = wizard_rows.sort_values(
                ["paper_credible", "_hourly_recommended", "_hourly_changed", "_primary"],
                ascending=[True, False, False, False],
            )
            selected.append(wizard_rows.iloc[0])
        remaining = promotion.copy()
        chosen_ids = {_text(row.get("candidate_id", "")) for row in selected}
        if chosen_ids:
            remaining = remaining[~remaining.get("candidate_id", pd.Series(dtype=object)).astype(str).isin(chosen_ids)].copy()
        if not remaining.empty:
            remaining["_blocked"] = ~remaining.get("paper_credible", pd.Series(dtype=bool)).astype(bool)
            remaining["_primary"] = remaining.get("setup_role", pd.Series(dtype=object)).astype(str).eq("primary")
            remaining = remaining.sort_values(["_blocked", "_primary"], ascending=[False, False]).head(3)
            selected.extend([remaining.iloc[idx] for idx in range(len(remaining))])
        rows = []
        for rank, row in enumerate(selected[:5], start=1):
            lane = _text(row.get("lane", ""))
            candidate_id = _text(row.get("candidate_id", ""))
            if lane == NATIVE_LANE:
                diag_row = _match_row(native_diag, candidate_id, "candidate_id")
                focus_row = _match_row(native_focus, _text(row.get("pair", "")), "pair")
                blocker = _text(diag_row.get("current_blocker", "")) if diag_row is not None else _text(row.get("blocker", ""))
                action = _text(diag_row.get("next_repair_action", "")) if diag_row is not None else "rerun_native_forward_walk_after_fix"
                lineage = _text(diag_row.get("selected_history_source", "")) if diag_row is not None else ""
                if bool(row.get("paper_credible", False)) and _text(row.get("promotion_stage", "")) != "paper_eligible":
                    blocker = paper_gate_blocker or acceptance_blocker or _text(row.get("paper_status", "")) or "global_paper_handoff_blocked"
                    action = (
                        model_gate_action
                        if blocker == "model_gate_not_accepted" and model_gate_action
                        else paper_route_action or paper_gate_action or acceptance_action or _paper_route_followup_action(row)
                    )
                    if focus_row is not None and _text(focus_row.get("next_repair_action", "")):
                        action = _text(focus_row.get("next_repair_action", ""))
                    if blocker == "model_gate_not_accepted" and focus_row is not None:
                        lineage = ";".join(
                            part
                            for part in [
                                lineage,
                                "reports/ml/model_gate_pair_support_report.csv" if _safe_int(focus_row.get("model_gate_anchor_rank", 0)) > 0 else "",
                            ]
                            if part
                        )
            else:
                repair_row = _match_row(wizard_repair, candidate_id, "candidate_id")
                hourly_row = _match_row(wizard_hourly_targets, candidate_id, "candidate_id")
                blocker = _text(repair_row.get("current_blocker", "")) if repair_row is not None else _text(row.get("blocker", ""))
                action = _wizard_repair_loop_action(repair_row, hourly_row)
                lineage = _text(repair_row.get("evidence_path", "")) if repair_row is not None else ""
            rows.append(
                {
                    "repair_priority_rank": rank,
                    "pair": _text(row.get("pair", "")),
                    "candidate_id": candidate_id,
                    "setup_identity": _text(row.get("setup_identity", "")),
                    "lane": lane,
                    "current_blocker": blocker or _text(row.get("blocker", "")),
                    "repair_action": action,
                    "rerun_required": bool((not bool(row.get("paper_credible", False))) or _text(row.get("promotion_stage", "")) != "paper_eligible"),
                    "post_rerun_promotion_result": _text(row.get("promotion_stage", "")),
                    "run_lineage_ref": lineage or "reports/brain/promotion_ladder.csv",
                    "hourly_review_status": _text(hourly_row.get("review_status", "")) if lane == WIZARD_LANE and hourly_row is not None else "",
                    "hourly_candidate_status": _text(hourly_row.get("candidate_status", "")) if lane == WIZARD_LANE and hourly_row is not None else "",
                    "hourly_pair_recommended": bool(hourly_row.get("pair_recommended_this_hour", False)) if lane == WIZARD_LANE and hourly_row is not None else False,
                }
            )
        frame = pd.DataFrame(
            rows,
            columns=[
                "repair_priority_rank",
                "pair",
                "candidate_id",
                "setup_identity",
                "lane",
                "current_blocker",
                "repair_action",
                "rerun_required",
                "post_rerun_promotion_result",
                "run_lineage_ref",
                "hourly_review_status",
                "hourly_candidate_status",
                "hourly_pair_recommended",
            ],
        )
    path = root / "reports" / "brain" / "candidate_repair_loop.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"candidate_repair_loop": path}, summary={"rows": int(len(frame))})


def build_wizard_hourly_repair_targets(root: Path = ROOT) -> CommandResult:
    packets = _read_csv(root / "reports" / "brain" / "wizard_candidate_packets.csv")
    hourly_candidates = _read_csv(root / "reports" / "active" / "wizard_hourly_database" / "wizard_hourly_candidate_queue.csv")
    hourly_pairs = _read_csv(root / "reports" / "active" / "wizard_hourly_database" / "wizard_hourly_pair_queue.csv")
    rows = []
    for _, row in packets.iterrows():
        pair = _text(row.get("pair", ""))
        setup_identity = _text(row.get("setup_identity", ""))
        timeframe = _text(row.get("timeframe", ""))
        strategy_mode = _text(row.get("strategy_mode", ""))
        hourly_match = _match_wizard_hourly_candidate(hourly_candidates, pair, timeframe, strategy_mode)
        pair_match = _match_lane_pair_row(hourly_pairs, pair)
        rows.append(
            {
                "pair": pair,
                "candidate_id": _text(row.get("candidate_id", "")),
                "setup_identity": setup_identity,
                "setup_role": _text(row.get("setup_role", "")),
                "timeframe": timeframe,
                "strategy_mode": strategy_mode,
                "hourly_match_status": "matched" if hourly_match is not None else "pair_only" if pair_match is not None else "missing",
                "review_status": _text(hourly_match.get("review_status", "")) if hourly_match is not None else "",
                "candidate_status": _text(hourly_match.get("candidate_status", "")) if hourly_match is not None else "",
                "matched_hourly_timeframe": _text(hourly_match.get("timeframe", "")) if hourly_match is not None else "",
                "matched_hourly_strategy": _text(hourly_match.get("strategy", "")) if hourly_match is not None else "",
                "pair_recommended_this_hour": bool(pair_match.get("recommended_this_hour", False)) if pair_match is not None else False,
                "pair_hourly_rank": _safe_int(pair_match.get("hourly_rank", 0)) if pair_match is not None else 0,
                "next_repair_action": _wizard_hourly_target_action(hourly_match, pair_match),
                "evidence_path": "reports/active/wizard_hourly_database/wizard_hourly_candidate_queue.csv",
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "pair",
            "candidate_id",
            "setup_identity",
            "setup_role",
            "timeframe",
            "strategy_mode",
            "hourly_match_status",
            "review_status",
            "candidate_status",
            "matched_hourly_timeframe",
            "matched_hourly_strategy",
            "pair_recommended_this_hour",
            "pair_hourly_rank",
            "next_repair_action",
            "evidence_path",
        ],
    )
    path = root / "reports" / "brain" / "wizard_hourly_repair_targets.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"wizard_hourly_repair_targets": path}, summary={"rows": int(len(frame))})


def _wizard_queue_pair_from_setup(setup_identity: str, fallback_pair: str) -> str:
    text = _text(setup_identity)
    parts = text.split("|")
    if len(parts) >= 2 and parts[0] and parts[1]:
        return f"{parts[0]}/{parts[1]}"
    return fallback_pair


def _wizard_capture_completeness_status(*, blocker: str, verification_row: pd.Series | None) -> str:
    blocker_text = _text(blocker)
    verification_status = _text(verification_row.get("verification_status", "")) if verification_row is not None else ""
    verification_blocker = _text(verification_row.get("verification_blocker", "")) if verification_row is not None else ""
    if "missing_exact_mode" in blocker_text:
        return "exact_mode_missing"
    if "missing_correlation" in blocker_text or "missing_ecm" in blocker_text:
        return "pair_detail_panel_missing"
    if "missing_local_history_path" in verification_blocker or "missing_wizard_capture_path" in verification_blocker:
        return "local_inputs_missing"
    if verification_status == "blocked":
        return "verification_blocked"
    if verification_status == "verified":
        return "verification_ready"
    return "capture_state_unknown"


def _shadow_freshness_status(*, lane: str, candidate_id: str, pair: str, wizard_hourly: pd.DataFrame, native_focus: pd.DataFrame) -> str:
    if lane == WIZARD_LANE:
        row = _match_row(wizard_hourly, candidate_id, "candidate_id")
        if row is not None and _text(row.get("review_status", "")) in {"new", "changed"}:
            return "fresh_repaired_candidate"
        return "stale_blocked_candidate"
    row = _match_row(native_focus, pair, "pair")
    if row is not None and bool(row.get("current_history_matches_focused_5m", False)):
        return "upgraded_native_candidate"
    return "stale_blocked_candidate"


def _shadow_evidence_quality_status(
    *,
    lane: str,
    promotion_row: pd.Series | None,
    wizard_hourly_row: pd.Series | None,
    wizard_repair_row: pd.Series | None,
    native_focus_row: pd.Series | None,
) -> str:
    blocker = _text(promotion_row.get("blocker", "")) if promotion_row is not None else ""
    if lane == WIZARD_LANE:
        repair_blocker = _text(wizard_repair_row.get("current_blocker", "")) if wizard_repair_row is not None else ""
        exact_mode_status = _text(wizard_repair_row.get("exact_mode_capture_status", "")) if wizard_repair_row is not None else ""
        local_verification_status = _text(wizard_repair_row.get("local_verification_status", "")) if wizard_repair_row is not None else ""
        if wizard_hourly_row is not None and _text(wizard_hourly_row.get("hourly_match_status", "")) == "matched":
            if exact_mode_status == "captured" and local_verification_status == "verified" and not repair_blocker:
                return "supported"
            if exact_mode_status == "captured" and local_verification_status == "verified":
                return "quality_blocked_candidate"
            return "needs_capture_completion"
        if exact_mode_status == "captured":
            if local_verification_status == "verified" and not any(
                token in repair_blocker for token in ["missing_exact_mode", "missing_correlation", "missing_ecm", "parity_missing_local_data", "wizard_local_verification_missing"]
            ):
                return "quality_blocked_candidate"
            return "needs_capture_completion"
        if any(token in repair_blocker for token in ["missing_exact_mode", "missing_correlation", "missing_ecm", "parity_missing_local_data", "wizard_local_verification_missing"]):
            return "needs_capture_completion"
        if "missing_exact_mode" in blocker or "missing_correlation" in blocker or "missing_ecm" in blocker:
            return "needs_capture_completion"
        return "unsupported_candidate"
    if native_focus_row is not None and bool(native_focus_row.get("current_history_matches_focused_5m", False)):
        pair_support = _text(native_focus_row.get("pair_model_support_status", ""))
        if pair_support == "strong_model_support":
            return "supported"
        if pair_support == "weak_model_support":
            return "quality_blocked_candidate"
        if pair_support in {
            "no_model_support",
            "pair_missing_from_model_predictions",
            "model_predictions_missing",
        }:
            return "unsupported_candidate"
        return "quality_blocked_candidate"
    return "unsupported_candidate"


def _shadow_usefulness_label(*, recommended: str, paper_credible: bool, freshness: str, evidence_quality: str) -> str:
    if evidence_quality == "unsupported_candidate":
        return "unsupported_candidate"
    if evidence_quality == "quality_blocked_candidate":
        return "comparison_only"
    if recommended == "accept" and paper_credible:
        return "aligned_signal"
    if freshness in {"fresh_repaired_candidate", "upgraded_native_candidate"}:
        return "fresh_repair_learning"
    if recommended == "inspect":
        return "triage_only"
    return "comparison_only"


def _shadow_next_rl_action(*, freshness: str, evidence_quality: str, recommended: str) -> str:
    if evidence_quality == "unsupported_candidate":
        return "wait_for_repair_evidence"
    if evidence_quality == "quality_blocked_candidate":
        return "keep_in_comparison_pool"
    if freshness in {"fresh_repaired_candidate", "upgraded_native_candidate"}:
        return "promote_into_shadow_rl_training_set"
    if recommended == "inspect":
        return "keep_in_shadow_triage"
    return "keep_in_comparison_pool"


def build_shadow_rl_usefulness_report(root: Path = ROOT) -> CommandResult:
    shadow = _read_csv(root / "reports" / "brain" / "lane_shadow_rl.csv")
    promotion = _read_csv(root / "reports" / "brain" / "promotion_ladder.csv")
    wizard_hourly = _read_csv(root / "reports" / "brain" / "wizard_hourly_repair_targets.csv")
    wizard_repair = _read_csv(root / "reports" / "brain" / "wizard_candidate_repair_report.csv")
    native_focus = _read_csv(root / "reports" / "brain" / "native_focus_repair_report.csv")
    rows = []
    for _, row in shadow.iterrows():
        candidate_id = str(row.get("candidate_id", ""))
        lane = str(row.get("lane", ""))
        promotion_row = _match_row(promotion, candidate_id, "candidate_id")
        recommended = _text(row.get("recommended_action", ""))
        paper_credible = bool(promotion_row.get("paper_credible", False)) if promotion_row is not None else False
        freshness = _shadow_freshness_status(
            lane=lane,
            candidate_id=candidate_id,
            pair=_text(row.get("pair", "")),
            wizard_hourly=wizard_hourly,
            native_focus=native_focus,
        )
        evidence_quality = _shadow_evidence_quality_status(
            lane=lane,
            promotion_row=promotion_row,
            wizard_hourly_row=_match_row(wizard_hourly, candidate_id, "candidate_id") if lane == WIZARD_LANE else None,
            wizard_repair_row=_match_row(wizard_repair, candidate_id, "candidate_id") if lane == WIZARD_LANE else None,
            native_focus_row=_match_row(native_focus, _text(row.get("pair", "")), "pair") if lane == NATIVE_LANE else None,
        )
        usefulness = _shadow_usefulness_label(
            recommended=recommended,
            paper_credible=paper_credible,
            freshness=freshness,
            evidence_quality=evidence_quality,
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "lane": lane,
                "pair": str(row.get("pair", "")),
                "setup_identity": str(row.get("setup_identity", "")),
                "recommended_action": recommended,
                "shadow_only": bool(row.get("shadow_only", True)),
                "paper_credible_overlap": paper_credible,
                "promotion_stage": _text(promotion_row.get("promotion_stage", "")) if promotion_row is not None else "",
                "freshness_status": freshness,
                "evidence_quality_status": evidence_quality,
                "usefulness_status": usefulness,
                "next_rl_action": _shadow_next_rl_action(freshness=freshness, evidence_quality=evidence_quality, recommended=recommended),
                "blocker": "" if usefulness == "aligned_signal" else (_text(promotion_row.get("blocker", "")) if promotion_row is not None else "promotion_context_missing"),
                "evidence_path": str(row.get("evidence_path", "")),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "candidate_id",
            "lane",
            "pair",
            "setup_identity",
            "recommended_action",
            "shadow_only",
            "paper_credible_overlap",
            "promotion_stage",
            "freshness_status",
            "evidence_quality_status",
            "usefulness_status",
            "next_rl_action",
            "blocker",
            "evidence_path",
        ],
    )
    path = root / "reports" / "brain" / "shadow_rl_usefulness_report.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"shadow_rl_usefulness_report": path}, summary={"rows": int(len(frame))})


def build_shadow_rl_candidate_set(root: Path = ROOT) -> CommandResult:
    shadow = _read_csv(root / "reports" / "brain" / "shadow_rl_usefulness_report.csv")
    wizard_packets = _read_csv(root / "reports" / "brain" / "wizard_candidate_packets.csv")
    if shadow.empty:
        frame = pd.DataFrame(
            columns=[
                "rl_priority_rank",
                "candidate_id",
                "lane",
                "pair",
                "setup_identity",
                "freshness_status",
                "evidence_quality_status",
                "usefulness_status",
                "next_rl_action",
            ]
        )
    else:
        frame = shadow.copy()
        frame = frame[
            frame.get("evidence_quality_status", pd.Series(dtype=object)).astype(str) != "unsupported_candidate"
        ].copy()
        promotable = frame[frame.get("next_rl_action", pd.Series(dtype=object)).astype(str) == "promote_into_shadow_rl_training_set"].copy()
        if not promotable.empty:
            frame = promotable
        if not frame.empty and not wizard_packets.empty:
            wizard_index = wizard_packets.set_index("candidate_id", drop=False)
            frame["_wizard_setup_role"] = frame.get("candidate_id", pd.Series(dtype=object)).map(
                lambda candidate_id: _text(wizard_index.loc[candidate_id, "setup_role"]) if candidate_id in wizard_index.index else ""
            )
            frame["_wizard_pair"] = frame.get("candidate_id", pd.Series(dtype=object)).map(
                lambda candidate_id: _text(wizard_index.loc[candidate_id, "pair"]) if candidate_id in wizard_index.index else ""
            )
            wizard_rows = frame[frame.get("lane", pd.Series(dtype=object)).astype(str) == WIZARD_LANE].copy()
            if not wizard_rows.empty:
                wizard_rows["_wizard_primary_rank"] = wizard_rows.get("_wizard_setup_role", pd.Series(dtype=object)).map(
                    lambda value: 0 if _text(value) == "primary" else 1
                )
                wizard_rows = wizard_rows.sort_values(
                    ["_wizard_pair", "_wizard_primary_rank"],
                    ascending=[True, True],
                ).drop_duplicates(subset=["_wizard_pair"], keep="first")
                non_wizard = frame[frame.get("lane", pd.Series(dtype=object)).astype(str) != WIZARD_LANE].copy()
                frame = pd.concat([wizard_rows, non_wizard], ignore_index=True)
        frame["_lane_rank"] = frame.get("lane", pd.Series(dtype=object)).map(lambda lane: 0 if _text(lane) == WIZARD_LANE else 1)
        frame["_fresh_rank"] = frame.get("freshness_status", pd.Series(dtype=object)).map(
            lambda value: 0 if _text(value) == "fresh_repaired_candidate" else 1 if _text(value) == "upgraded_native_candidate" else 2
        )
        frame["_quality_rank"] = frame.get("evidence_quality_status", pd.Series(dtype=object)).map(
            lambda value: 0 if _text(value) == "supported" else 1 if _text(value) == "quality_blocked_candidate" else 2 if _text(value) == "needs_capture_completion" else 3
        )
        frame = frame.sort_values(["_fresh_rank", "_quality_rank", "_lane_rank"]).head(10).reset_index(drop=True)
        frame["rl_priority_rank"] = range(1, len(frame) + 1)
        frame = frame[
            [
                "rl_priority_rank",
                "candidate_id",
                "lane",
                "pair",
                "setup_identity",
                "freshness_status",
                "evidence_quality_status",
                "usefulness_status",
                "next_rl_action",
            ]
        ]
    path = root / "reports" / "brain" / "shadow_rl_candidate_set.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"shadow_rl_candidate_set": path}, summary={"rows": int(len(frame))})


def build_shadow_rl_intake_report(root: Path = ROOT) -> CommandResult:
    candidate_set = _read_csv(root / "reports" / "brain" / "shadow_rl_candidate_set.csv")
    usefulness = _read_csv(root / "reports" / "brain" / "shadow_rl_usefulness_report.csv")
    rows = []
    for _, row in candidate_set.iterrows():
        candidate_id = _text(row.get("candidate_id", ""))
        usefulness_row = _match_row(usefulness, candidate_id, "candidate_id")
        rows.append(
            {
                "rl_priority_rank": _safe_int(row.get("rl_priority_rank", 0)),
                "candidate_id": candidate_id,
                "lane": _text(row.get("lane", "")),
                "pair": _text(row.get("pair", "")),
                "setup_identity": _text(row.get("setup_identity", "")),
                "intake_status": _shadow_intake_status(
                    usefulness_status=_text(row.get("usefulness_status", "")),
                    next_rl_action=_text(row.get("next_rl_action", "")),
                    evidence_quality_status=_text(row.get("evidence_quality_status", "")),
                ),
                "freshness_status": _text(row.get("freshness_status", "")),
                "evidence_quality_status": _text(row.get("evidence_quality_status", "")),
                "usefulness_status": _text(row.get("usefulness_status", "")),
                "next_rl_action": _text(row.get("next_rl_action", "")),
                "promotion_stage": _text(usefulness_row.get("promotion_stage", "")) if usefulness_row is not None else "",
                "recommended_action": _text(usefulness_row.get("recommended_action", "")) if usefulness_row is not None else "",
                "paper_credible_overlap": bool(usefulness_row.get("paper_credible_overlap", False)) if usefulness_row is not None else False,
                "blocker": _text(usefulness_row.get("blocker", "")) if usefulness_row is not None else "",
                "evidence_path": _text(usefulness_row.get("evidence_path", "")) if usefulness_row is not None else "",
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "rl_priority_rank",
            "candidate_id",
            "lane",
            "pair",
            "setup_identity",
            "intake_status",
            "freshness_status",
            "evidence_quality_status",
            "usefulness_status",
            "next_rl_action",
            "promotion_stage",
            "recommended_action",
            "paper_credible_overlap",
            "blocker",
            "evidence_path",
        ],
    )
    path = root / "reports" / "brain" / "shadow_rl_intake_report.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"shadow_rl_intake_report": path}, summary={"rows": int(len(frame))})


def _shadow_intake_status(*, usefulness_status: str, next_rl_action: str, evidence_quality_status: str) -> str:
    usefulness = _text(usefulness_status)
    next_action = _text(next_rl_action)
    evidence_quality = _text(evidence_quality_status)
    if next_action == "promote_into_shadow_rl_training_set" and usefulness in {"aligned_signal", "fresh_repair_learning"}:
        return "authoritative_shadow_training_candidate"
    if evidence_quality == "quality_blocked_candidate" or usefulness == "comparison_only":
        return "comparison_only_shadow_candidate"
    if next_action == "keep_in_shadow_triage" or usefulness == "triage_only":
        return "shadow_triage_candidate"
    return "pending_repair_candidate"


def build_overall_arbitration_scorecard(root: Path = ROOT) -> CommandResult:
    overall = _read_csv(root / "reports" / "brain" / "overall_brain_summary.csv")
    promotion = _read_csv(root / "reports" / "brain" / "promotion_ladder.csv")
    rows = []
    for _, row in overall.iterrows():
        pair = str(row.get("pair", ""))
        pair_promotion = promotion[promotion.get("pair", pd.Series(dtype=str)).astype(str) == pair].copy() if not promotion.empty else pd.DataFrame()
        ready_count = int(pair_promotion.get("paper_credible", pd.Series(dtype=bool)).astype(bool).sum()) if not pair_promotion.empty else 0
        rows.append(
            {
                "pair": pair,
                "agreement_status": str(row.get("agreement_status", "")),
                "wizard_support": bool(row.get("wizard_support", False)),
                "native_support": bool(row.get("native_support", False)),
                "arbitration_decision": str(row.get("arbitration_decision", "")),
                "promotion_ready_count": ready_count,
                "pair_status": _pair_status_from_overall(row, pair_promotion),
                "disagreement_flag": str(row.get("agreement_status", "")) in {"wizard_only", "native_only"},
                "evidence_path": str(row.get("evidence_path", "")),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "pair",
            "agreement_status",
            "wizard_support",
            "native_support",
            "arbitration_decision",
            "promotion_ready_count",
            "pair_status",
            "disagreement_flag",
            "evidence_path",
        ],
    )
    path = root / "reports" / "brain" / "overall_arbitration_scorecard.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"overall_arbitration_scorecard": path}, summary={"rows": int(len(frame))})


def build_blocker_persistence_report(root: Path = ROOT) -> CommandResult:
    promotion = _read_csv(root / "reports" / "brain" / "promotion_ladder.csv")
    wizard_paper = _read_csv(root / "reports" / "brain" / "wizard_paper_outcomes.csv")
    native_paper = _read_csv(root / "reports" / "brain" / "native_paper_outcomes.csv")
    rows = []
    for _, row in promotion.iterrows():
        candidate_id = str(row.get("candidate_id", ""))
        lane = str(row.get("lane", ""))
        blocker = _text(row.get("blocker", ""))
        outcomes = wizard_paper if lane == WIZARD_LANE else native_paper
        outcome_row = _match_row(outcomes, candidate_id, "candidate_id")
        verified = bool(_text(outcome_row.get("verification_status", "")).lower() == "verified") if outcome_row is not None else False
        result_status = _text(outcome_row.get("result_status", "")) if outcome_row is not None else ""
        rows.append(
            {
                "candidate_id": candidate_id,
                "lane": lane,
                "pair": str(row.get("pair", "")),
                "current_blocker": blocker,
                "blocker_persistence": "clear" if blocker == "" else ("persistent" if _text(row.get("promotion_stage", "")) == "forward_walk_blocked" else "active"),
                "paper_outcome_status": result_status or "none",
                "verified_outcome": verified,
                "verification_gap": "" if verified or outcome_row is None else "paper_outcome_unverified",
                "evidence_path": f"reports/brain/{lane}_paper_outcomes.csv" if outcome_row is not None else "reports/brain/promotion_ladder.csv",
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "candidate_id",
            "lane",
            "pair",
            "current_blocker",
            "blocker_persistence",
            "paper_outcome_status",
            "verified_outcome",
            "verification_gap",
            "evidence_path",
        ],
    )
    path = root / "reports" / "brain" / "blocker_persistence_report.csv"
    _write_csv(frame, path)
    verified_path = root / "reports" / "brain" / "verified_paper_outcome_report.csv"
    verified_frame = frame[frame["verified_outcome"].astype(bool)].copy() if not frame.empty else pd.DataFrame(columns=frame.columns)
    _write_csv(verified_frame, verified_path)
    return CommandResult(
        paths={"blocker_persistence_report": path, "verified_paper_outcome_report": verified_path},
        summary={"rows": int(len(frame)), "verified_outcomes": int(verified_frame.shape[0])},
    )


def build_shared_outcome_memory(root: Path = ROOT) -> CommandResult:
    wizard_packets = _read_csv(root / "reports" / "brain" / "wizard_candidate_packets.csv")
    native_packets = _read_csv(root / "reports" / "brain" / "native_candidate_packets.csv")
    wizard_outcomes = _read_csv(root / "reports" / "brain" / "wizard_paper_outcomes.csv")
    native_outcomes = _read_csv(root / "reports" / "brain" / "native_paper_outcomes.csv")
    packet_frame = pd.concat([wizard_packets, native_packets], ignore_index=True) if not wizard_packets.empty or not native_packets.empty else pd.DataFrame()

    rows: list[dict[str, object]] = []
    for lane, outcomes in ((WIZARD_LANE, wizard_outcomes), (NATIVE_LANE, native_outcomes)):
        if outcomes.empty:
            continue
        for _, outcome in outcomes.iterrows():
            candidate_id = _text(outcome.get("candidate_id", ""))
            packet = _match_row(packet_frame, candidate_id, "candidate_id")
            pair = _text(packet.get("pair", "")) if packet is not None else ""
            venue = _text(packet.get("venue", "")) if packet is not None else ""
            strategy_family = _text(packet.get("strategy_family", "")) if packet is not None else ""
            strategy_mode = _text(packet.get("strategy_mode", "")) if packet is not None else ""
            timeframe = _text(packet.get("timeframe", "")) if packet is not None else ""
            regime_bucket = _regime_bucket_from_snapshot(_text(packet.get("regime_snapshot", "")) if packet is not None else "")
            realized_return = _safe_float(outcome.get("realized_return", 0.0))
            drawdown = _safe_float(outcome.get("drawdown", 0.0))
            verification_status = _text(outcome.get("verification_status", ""))
            result_status = _text(outcome.get("result_status", ""))
            if result_status.lower() in {"broadcast_accepted_unconfirmed", "execution_unconfirmed"}:
                continue
            outcome_state = _outcome_state(result_status, verification_status)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "source_lane": lane,
                    "pair": pair,
                    "venue": venue or _text(outcome.get("paper_venue", "")),
                    "paper_venue": _text(outcome.get("paper_venue", "")),
                    "timeframe": timeframe,
                    "regime_bucket": regime_bucket,
                    "strategy_family": strategy_family,
                    "strategy_mode": strategy_mode,
                    "submission_timestamp": _text(outcome.get("submission_timestamp", "")),
                    "hold_duration": _text(outcome.get("hold_duration", "")),
                    "realized_return": realized_return,
                    "drawdown": drawdown,
                    "result_status": result_status,
                    "verification_status": verification_status,
                    "outcome_label": _outcome_label(realized_return, result_status, verification_status),
                    "outcome_state": outcome_state,
                    "learning_weight": _learning_weight(outcome_state, verification_status),
                    "promotion_authority": "wizard_learning_only" if lane == WIZARD_LANE else "native_learning_only",
                    "evidence_path": _text(outcome.get("outcome_evidence_path", "")) or f"reports/brain/{lane}_paper_outcomes.csv",
                }
            )

    frame = pd.DataFrame(
        rows,
        columns=[
            "candidate_id",
            "source_lane",
            "pair",
            "venue",
            "paper_venue",
            "timeframe",
            "regime_bucket",
            "strategy_family",
            "strategy_mode",
            "submission_timestamp",
            "hold_duration",
            "realized_return",
            "drawdown",
            "result_status",
            "verification_status",
            "outcome_label",
            "outcome_state",
            "learning_weight",
            "promotion_authority",
            "evidence_path",
        ],
    )
    path = root / "reports" / "brain" / "shared_outcome_memory.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"shared_outcome_memory": path}, summary={"rows": int(len(frame))})


def build_native_outcome_feature_memory(root: Path = ROOT) -> CommandResult:
    native_packets = _read_csv(root / "reports" / "brain" / "native_candidate_packets.csv")
    shared = _read_csv(root / "reports" / "brain" / "shared_outcome_memory.csv")
    wizard_shared = shared[shared.get("source_lane", pd.Series(dtype=object)).astype(str) == WIZARD_LANE].copy() if not shared.empty else pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, packet in native_packets.iterrows():
        candidate_id = _text(packet.get("candidate_id", ""))
        pair = _text(packet.get("pair", ""))
        venue = _text(packet.get("venue", ""))
        strategy_family = _text(packet.get("strategy_family", ""))
        strategy_mode = _text(packet.get("strategy_mode", ""))
        regime_bucket = _regime_bucket_from_snapshot(_text(packet.get("regime_snapshot", "")))

        same_regime = _subset_equals(wizard_shared, "regime_bucket", regime_bucket)
        same_strategy = _subset_equals(wizard_shared, "strategy_family", strategy_family)
        same_venue = _subset_equals(wizard_shared, "venue", venue)
        same_triplet = wizard_shared.copy()
        if not same_triplet.empty:
            same_triplet = same_triplet[
                same_triplet.get("regime_bucket", pd.Series(dtype=object)).astype(str).eq(regime_bucket)
                & same_triplet.get("strategy_family", pd.Series(dtype=object)).astype(str).eq(strategy_family)
                & same_triplet.get("venue", pd.Series(dtype=object)).astype(str).eq(venue)
            ].copy()

        verified_triplet = same_triplet[
            same_triplet.get("verification_status", pd.Series(dtype=object)).astype(str).str.lower().eq("verified")
        ].copy() if not same_triplet.empty else pd.DataFrame()

        rows.append(
            {
                "candidate_id": candidate_id,
                "pair": pair,
                "native_venue": venue,
                "native_strategy_family": strategy_family,
                "native_strategy_mode": strategy_mode,
                "native_regime_bucket": regime_bucket,
                "wizard_history_feature_count": int(len(wizard_shared)),
                "wizard_same_regime_count": int(len(same_regime)),
                "wizard_same_strategy_family_count": int(len(same_strategy)),
                "wizard_same_venue_count": int(len(same_venue)),
                "wizard_same_regime_strategy_venue_count": int(len(same_triplet)),
                "wizard_verified_same_regime_strategy_venue_count": int(len(verified_triplet)),
                "wizard_same_regime_strategy_venue_win_rate": _win_rate(verified_triplet),
                "wizard_same_regime_strategy_venue_mean_return": _mean_numeric(verified_triplet, "realized_return"),
                "wizard_same_regime_strategy_venue_mean_drawdown": _mean_numeric(verified_triplet, "drawdown"),
                "wizard_learning_feature_state": "available" if not wizard_shared.empty else "cold_start",
                "native_promotion_basis": "native_evidence_only",
                "wizard_learning_ref": "reports/brain/shared_outcome_memory.csv",
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "candidate_id",
            "pair",
            "native_venue",
            "native_strategy_family",
            "native_strategy_mode",
            "native_regime_bucket",
            "wizard_history_feature_count",
            "wizard_same_regime_count",
            "wizard_same_strategy_family_count",
            "wizard_same_venue_count",
            "wizard_same_regime_strategy_venue_count",
            "wizard_verified_same_regime_strategy_venue_count",
            "wizard_same_regime_strategy_venue_win_rate",
            "wizard_same_regime_strategy_venue_mean_return",
            "wizard_same_regime_strategy_venue_mean_drawdown",
            "wizard_learning_feature_state",
            "native_promotion_basis",
            "wizard_learning_ref",
        ],
    )
    path = root / "reports" / "brain" / "native_outcome_feature_memory.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"native_outcome_feature_memory": path}, summary={"rows": int(len(frame))})


def build_native_promotion_independence_report(root: Path = ROOT) -> CommandResult:
    native_features = _read_csv(root / "reports" / "brain" / "native_outcome_feature_memory.csv")
    promotion = _read_csv(root / "reports" / "brain" / "promotion_ladder.csv")
    native_promotion = promotion[promotion.get("lane", pd.Series(dtype=object)).astype(str) == NATIVE_LANE].copy() if not promotion.empty else pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, feature in native_features.iterrows():
        candidate_id = _text(feature.get("candidate_id", ""))
        promotion_row = _match_row(native_promotion, candidate_id, "candidate_id")
        rows.append(
            {
                "candidate_id": candidate_id,
                "pair": _text(feature.get("pair", "")),
                "wizard_learning_feature_state": _text(feature.get("wizard_learning_feature_state", "")),
                "wizard_same_regime_strategy_venue_count": _safe_int(feature.get("wizard_same_regime_strategy_venue_count", 0)),
                "native_promotion_basis": "native_evidence_only",
                "promotion_stage": _text(promotion_row.get("promotion_stage", "")) if promotion_row is not None else "",
                "paper_credible": bool(promotion_row.get("paper_credible", False)) if promotion_row is not None else False,
                "independence_status": "independent",
                "reason": "Wizard outcomes are available as training context only; Native promotion remains based on Native evidence and current promotion gates.",
                "evidence_path": "reports/brain/native_outcome_feature_memory.csv;reports/brain/promotion_ladder.csv",
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "candidate_id",
            "pair",
            "wizard_learning_feature_state",
            "wizard_same_regime_strategy_venue_count",
            "native_promotion_basis",
            "promotion_stage",
            "paper_credible",
            "independence_status",
            "reason",
            "evidence_path",
        ],
    )
    path = root / "reports" / "brain" / "native_promotion_independence_report.csv"
    _write_csv(frame, path)
    return CommandResult(paths={"native_promotion_independence_report": path}, summary={"rows": int(len(frame))})


def _write_lane_outputs(
    root: Path,
    lane: str,
    packets: list[dict[str, object]],
    forward_walk_packets: list[dict[str, object]],
    paper_packets: list[dict[str, object]],
    *,
    fold_rows: list[dict[str, object]] | None = None,
) -> CommandResult:
    reports = root / "reports" / "brain"
    packet_csv = reports / f"{lane}_candidate_packets.csv"
    packet_jsonl = reports / f"{lane}_candidate_packets.jsonl"
    fw_csv = reports / f"{lane}_forward_walk.csv"
    fw_jsonl = reports / f"{lane}_forward_walk.jsonl"
    fw_folds_csv = reports / f"{lane}_forward_walk_folds.csv"
    paper_csv = reports / f"{lane}_paper_outcomes.csv"
    paper_jsonl = reports / f"{lane}_paper_outcomes.jsonl"

    packet_frame = pd.DataFrame(packets)
    fw_frame = pd.DataFrame(forward_walk_packets)
    paper_frame = pd.DataFrame(paper_packets)
    fold_frame = pd.DataFrame(fold_rows or [])
    write_candidate_setup_frame(packet_frame if not packet_frame.empty else pd.DataFrame(columns=[]), packet_csv)
    if packets:
        packet_jsonl.unlink(missing_ok=True)
        append_candidate_setup_packets(packet_jsonl, packets)
    write_forward_walk_frame(fw_frame if not fw_frame.empty else pd.DataFrame(columns=[]), fw_csv)
    if forward_walk_packets:
        fw_jsonl.unlink(missing_ok=True)
        append_forward_walk_packets(fw_jsonl, forward_walk_packets)
    _write_csv(fold_frame, fw_folds_csv)
    write_paper_outcome_frame(paper_frame if not paper_frame.empty else pd.DataFrame(columns=[]), paper_csv)
    if paper_packets:
        paper_jsonl.unlink(missing_ok=True)
        append_paper_outcome_packets(paper_jsonl, paper_packets)

    rollup_csv = brain_output_paths(root, "current")["candidate_setup_csv"]
    combined = _combine_lane_packets(root)
    write_candidate_setup_frame(combined if not combined.empty else pd.DataFrame(columns=[]), rollup_csv)
    return CommandResult(
        paths={
            f"{lane}_candidate_packets": packet_csv,
            f"{lane}_candidate_packets_jsonl": packet_jsonl,
            f"{lane}_forward_walk": fw_csv,
            f"{lane}_forward_walk_folds": fw_folds_csv,
            f"{lane}_paper_outcomes": paper_csv,
        },
        summary={"candidate_rows": int(len(packet_frame)), "forward_walk_rows": int(len(fw_frame)), "paper_rows": int(len(paper_frame)), "forward_walk_folds": int(len(fold_frame))},
    )


def _combine_lane_packets(root: Path) -> pd.DataFrame:
    reports = root / "reports" / "brain"
    wizard = _read_csv(reports / "wizard_candidate_packets.csv")
    native = _read_csv(reports / "native_candidate_packets.csv")
    frames = [frame for frame in (wizard, native) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _native_discovery_frame(root: Path) -> pd.DataFrame:
    pair_universe = _read_csv(root / "data" / "processed" / "pair_universe.csv")
    local_manifest = _read_csv(root / "reports" / "p2_rerun_subset_manifest.csv")
    dydx_manifest = _read_csv(root / "reports" / "dydx_local_pair_universe_run.csv")
    if pair_universe.empty:
        return pd.DataFrame()
    frame = pair_universe.copy()
    if not local_manifest.empty and "pair" in local_manifest.columns:
        manifest = local_manifest.copy()
        manifest["pair"] = manifest["pair"].astype(str)
        frame = frame.merge(
            manifest[["pair", "selected_path", "status", "detail", "history_rows", "execution_usable"]].rename(
                columns={
                    "status": "native_discovery_status",
                    "detail": "native_discovery_detail",
                    "history_rows": "native_history_rows",
                    "execution_usable": "native_execution_usable",
                }
            ),
            on="pair",
            how="left",
        )
    if not dydx_manifest.empty:
        manifest_pairs = dydx_manifest.copy()
        manifest_pairs["pair"] = manifest_pairs.apply(
            lambda row: _normalize_pair_from_assets(row.get("pair_id", ""), row.get("asset_x", ""), row.get("asset_y", "")),
            axis=1,
        )
        frame = frame.merge(
            manifest_pairs[["pair", "pair_history", "status"]].rename(
                columns={"pair_history": "dydx_pair_history", "status": "dydx_local_discovery_status"}
            ),
            on="pair",
            how="left",
        )
    return frame


def _wizard_forward_walk_metrics(root: Path, candidate_id: str, verification_row: pd.Series | None) -> tuple[dict[str, object], list[dict[str, object]]]:
    if verification_row is None:
        return {
            "rolling_split_definition": "wizard_local_validation_proxy",
            "oos_sharpe": "",
            "oos_profit_factor": "",
            "oos_max_drawdown": "",
            "oos_trade_count": 0,
            "fold_count": 0,
            "forward_walk_status": "missing",
            "blocker_reason": "wizard_local_verification_missing",
        }, []
    trade_log_path = _trade_log_path_from_summary(verification_row.get("summary_path", ""))
    trade_log = _read_csv(root / trade_log_path) if trade_log_path else pd.DataFrame()
    fold_rows = _trade_log_forward_walk_rows(candidate_id, WIZARD_LANE, trade_log)
    if not fold_rows:
        status, blocker = _forward_walk_from_verification(verification_row)
        return {
            "rolling_split_definition": "wizard_local_validation_proxy",
            "oos_sharpe": _safe_float(verification_row.get("local_sharpe", "")),
            "oos_profit_factor": _safe_float(verification_row.get("local_profit_factor", "")),
            "oos_max_drawdown": _safe_float(verification_row.get("local_max_drawdown", "")),
            "oos_trade_count": _safe_int(verification_row.get("local_closed_trades", "")),
            "fold_count": 0,
            "forward_walk_status": status,
            "blocker_reason": blocker,
        }, []
    fold_frame = pd.DataFrame(fold_rows)
    pf = pd.to_numeric(fold_frame["oos_profit_factor"], errors="coerce")
    sharpe = pd.to_numeric(fold_frame["oos_sharpe"], errors="coerce")
    drawdown = pd.to_numeric(fold_frame["oos_max_drawdown"], errors="coerce")
    trades = pd.to_numeric(fold_frame["oos_trade_count"], errors="coerce")
    status = "pass" if bool((pf.mean() >= 1.0) and (trades.sum() >= 3)) else "blocked"
    blocker = "" if status == "pass" else "wizard_fold_metrics_weak"
    return {
        "rolling_split_definition": "trade_log_time_folds",
        "oos_sharpe": round(float(sharpe.mean()), 6) if not sharpe.empty else 0.0,
        "oos_profit_factor": round(float(pf.mean()), 6) if not pf.empty else 0.0,
        "oos_max_drawdown": round(float(drawdown.max()), 6) if not drawdown.empty else 0.0,
        "oos_trade_count": int(trades.sum()) if not trades.empty else 0,
        "fold_count": int(len(fold_frame)),
        "forward_walk_status": status,
        "blocker_reason": blocker,
    }, fold_rows


def _native_forward_walk_metrics(root: Path, candidate_id: str, row: pd.Series) -> tuple[dict[str, object], list[dict[str, object]]]:
    diagnosis, fold_rows = _native_forward_walk_diagnosis(root, candidate_id, row)
    if fold_rows:
        return {
            "rolling_split_definition": "history_time_folds",
            "oos_sharpe": _safe_float(diagnosis.get("mean_fold_sharpe", 0.0)),
            "oos_profit_factor": _safe_float(diagnosis.get("mean_fold_profit_factor", 0.0)),
            "oos_max_drawdown": _safe_float(diagnosis.get("max_fold_drawdown", 0.0)),
            "oos_trade_count": _safe_int(diagnosis.get("nonzero_signal_trade_count", 0)),
            "fold_count": _safe_int(diagnosis.get("fold_count", 0)),
            "forward_walk_status": "pass" if _text(diagnosis.get("current_blocker", "")) == "" else "blocked",
            "blocker_reason": _text(diagnosis.get("current_blocker", "")),
        }, fold_rows
    walk_score = _safe_float(row.get("walk_forward_score", ""))
    return {
        "rolling_split_definition": "native_pair_universe_proxy",
        "oos_sharpe": walk_score,
        "oos_profit_factor": _safe_float(row.get("local_backtest_score", "")),
        "oos_max_drawdown": "",
        "oos_trade_count": 0,
        "fold_count": 0,
        "forward_walk_status": "pass" if walk_score >= 0.5 else "blocked",
        "blocker_reason": _text(diagnosis.get("current_blocker", "")) or ("" if walk_score >= 0.5 else "native_forward_walk_missing_or_weak"),
    }, []


def _trade_log_forward_walk_rows(candidate_id: str, lane: str, trade_log: pd.DataFrame, folds: int = 3) -> list[dict[str, object]]:
    if trade_log.empty:
        return []
    frame = trade_log.copy()
    frame = frame[frame.get("exit_reason", pd.Series(dtype=str)).astype(str) != "open_at_end_of_history"].copy()
    if frame.empty:
        return []
    time_col = "exit_time" if "exit_time" in frame.columns else ("exit_timestamp" if "exit_timestamp" in frame.columns else "")
    if time_col:
        frame[time_col] = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
        frame = frame.sort_values(time_col)
    frame["profit_after_cost"] = pd.to_numeric(frame.get("profit_after_cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    fold_slices = _fold_slices(len(frame), folds)
    rows = []
    for idx, (start, end) in enumerate(fold_slices, start=1):
        fold = frame.iloc[start:end].copy()
        if fold.empty:
            continue
        metrics = _returns_metrics(fold["profit_after_cost"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "lane": lane,
                "fold_id": idx,
                "fold_start": str(fold[time_col].iloc[0]) if time_col and time_col in fold.columns and not fold.empty else str(start),
                "fold_end": str(fold[time_col].iloc[-1]) if time_col and time_col in fold.columns and not fold.empty else str(end),
                "oos_sharpe": metrics["sharpe"],
                "oos_profit_factor": metrics["profit_factor"],
                "oos_max_drawdown": metrics["max_drawdown"],
                "oos_trade_count": int(len(fold)),
            }
        )
    return rows


def _history_forward_walk_rows(candidate_id: str, lane: str, history_path: str, folds: int = 3) -> list[dict[str, object]]:
    if not history_path:
        return []
    path = Path(history_path)
    if not path.exists():
        return []
    payload = _read_json(path)
    history = pd.DataFrame(payload.get("history", []))
    if history.empty:
        return []
    if "timestamp" in history.columns:
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
        history = history.sort_values("timestamp")
    zscore_col = ""
    for candidate in ("rolling_zscore", "zscore", "zscore_reconstructed"):
        if candidate in history.columns:
            zscore_col = candidate
            break
    if not zscore_col:
        return []
    zscore = pd.to_numeric(history[zscore_col], errors="coerce").fillna(0.0)
    returns = pd.to_numeric(history.get("spread_return_1", history.get("spread_change_1", pd.Series(dtype=float))), errors="coerce").fillna(0.0)
    signal = zscore.fillna(0.0).map(lambda value: -1.0 if value >= 2.0 else (1.0 if value <= -2.0 else 0.0))
    realized = (signal * returns).fillna(0.0)
    history = history.assign(_realized=realized, _signal=signal)
    fold_slices = _fold_slices(len(history), folds)
    rows = []
    for idx, (start, end) in enumerate(fold_slices, start=1):
        fold = history.iloc[start:end].copy()
        if fold.empty:
            continue
        nonzero = fold[fold["_signal"] != 0.0].copy()
        metrics = _returns_metrics(nonzero["_realized"] if not nonzero.empty else fold["_realized"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "lane": lane,
                "fold_id": idx,
                "fold_start": str(fold["timestamp"].iloc[0]) if "timestamp" in fold.columns and not fold.empty else str(start),
                "fold_end": str(fold["timestamp"].iloc[-1]) if "timestamp" in fold.columns and not fold.empty else str(end),
                "oos_sharpe": metrics["sharpe"],
                "oos_profit_factor": metrics["profit_factor"],
                "oos_max_drawdown": metrics["max_drawdown"],
                "oos_trade_count": int((fold["_signal"] != 0.0).sum()),
            }
        )
    return rows


def _history_forward_walk_rows_from_signals(
    *,
    candidate_id: str,
    lane: str,
    history: pd.DataFrame,
    signal: pd.Series,
    realized: pd.Series,
    folds: int = 3,
) -> list[dict[str, object]]:
    frame = history.copy()
    frame = frame.assign(_signal=pd.to_numeric(signal, errors="coerce").fillna(0.0), _realized=pd.to_numeric(realized, errors="coerce").fillna(0.0))
    fold_slices = _fold_slices(len(frame), folds)
    rows = []
    for idx, (start, end) in enumerate(fold_slices, start=1):
        fold = frame.iloc[start:end].copy()
        if fold.empty:
            continue
        nonzero = fold[fold["_signal"] != 0.0].copy()
        metrics = _returns_metrics(nonzero["_realized"] if not nonzero.empty else fold["_realized"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "lane": lane,
                "fold_id": idx,
                "fold_start": str(fold["timestamp"].iloc[0]) if "timestamp" in fold.columns and not fold.empty else str(start),
                "fold_end": str(fold["timestamp"].iloc[-1]) if "timestamp" in fold.columns and not fold.empty else str(end),
                "oos_sharpe": metrics["sharpe"],
                "oos_profit_factor": metrics["profit_factor"],
                "oos_max_drawdown": metrics["max_drawdown"],
                "oos_trade_count": int((fold["_signal"] != 0.0).sum()),
            }
        )
    return rows


def _mean_metric(frame: pd.DataFrame, column: str) -> float:
    series = pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce")
    finite = series[series.apply(lambda value: pd.notna(value) and math.isfinite(float(value)))]
    positive_inf_count = int(series.eq(float("inf")).sum())
    series = finite.dropna()
    if series.empty:
        return 999.0 if positive_inf_count > 0 else 0.0
    if positive_inf_count > 0:
        return round(float(max(series.max(), 999.0)), 6)
    return round(float(series.mean()), 6)


def _native_forward_walk_diagnosis(root: Path, candidate_id: str, row: pd.Series) -> tuple[dict[str, object], list[dict[str, object]]]:
    thresholds = _native_forward_walk_thresholds()
    history_path = _text(row.get("selected_path", "")) or _text(row.get("dydx_pair_history", ""))
    strategy_family = _native_strategy_family(row)
    strategy_mode = _native_strategy_mode(row)
    evaluation_mode = "proxy_zscore_logic"
    if strategy_family in {"native_local_math", "zscore", "mean_reversion", "copula", "ou", "ecm"}:
        evaluation_mode = "native_local_strategy_logic"
    if not history_path:
        return {
            "selected_history_source": "",
            "evaluation_mode": "imported_local_experiment_summary",
            "fold_count": 0,
            "nonzero_signal_trade_count": 0,
            "mean_fold_profit_factor": 0.0,
            "mean_fold_sharpe": 0.0,
            "max_fold_drawdown": 0.0,
            "current_blocker": "native_no_usable_history",
            "blocker_detail": "no selected history file was attached to the Native candidate",
            "threshold_reference": thresholds,
            "evidence_path": "data/processed/pair_universe.csv",
        }, []
    path = Path(history_path)
    if not path.exists():
        return {
            "selected_history_source": history_path,
            "evaluation_mode": "imported_local_experiment_summary",
            "fold_count": 0,
            "nonzero_signal_trade_count": 0,
            "mean_fold_profit_factor": 0.0,
            "mean_fold_sharpe": 0.0,
            "max_fold_drawdown": 0.0,
            "current_blocker": "native_no_usable_history",
            "blocker_detail": "selected history file is missing on disk",
            "threshold_reference": thresholds,
            "evidence_path": history_path,
        }, []
    payload = _read_json(path)
    history = pd.DataFrame(payload.get("history", []))
    if history.empty:
        return {
            "selected_history_source": history_path,
            "evaluation_mode": "imported_local_experiment_summary",
            "fold_count": 0,
            "nonzero_signal_trade_count": 0,
            "mean_fold_profit_factor": 0.0,
            "mean_fold_sharpe": 0.0,
            "max_fold_drawdown": 0.0,
            "current_blocker": "native_no_usable_history",
            "blocker_detail": "history payload exists but contains no rows",
            "threshold_reference": thresholds,
            "evidence_path": history_path,
        }, []
    if "timestamp" in history.columns:
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
        history = history.sort_values("timestamp")
    zscore_col = ""
    for candidate in ("rolling_zscore", "zscore", "zscore_reconstructed"):
        if candidate in history.columns:
            zscore_col = candidate
            break
    if not zscore_col:
        return {
            "selected_history_source": history_path,
            "evaluation_mode": "history_signal_unavailable",
            "fold_count": 0,
            "nonzero_signal_trade_count": 0,
            "mean_fold_profit_factor": 0.0,
            "mean_fold_sharpe": 0.0,
            "max_fold_drawdown": 0.0,
            "current_blocker": "native_no_zscore_like_column",
            "blocker_detail": "history payload has no rolling_zscore, zscore, or zscore_reconstructed column",
            "threshold_reference": thresholds,
            "evidence_path": history_path,
        }, []
    zscore = pd.to_numeric(history[zscore_col], errors="coerce").fillna(0.0)
    returns_source = "spread_return_1"
    if "spread_return_1" in history.columns:
        returns = pd.to_numeric(history["spread_return_1"], errors="coerce").fillna(0.0)
    elif "spread_change_1" in history.columns:
        returns_source = "spread_change_1"
        returns = pd.to_numeric(history["spread_change_1"], errors="coerce").fillna(0.0)
    elif "spread" in history.columns:
        returns_source = "derived_spread_pct_change"
        spread = pd.to_numeric(history["spread"], errors="coerce").replace(0, pd.NA)
        returns = spread.pct_change().replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    else:
        returns = pd.Series(0.0, index=history.index, dtype=float)
    signal = zscore.map(lambda value: -1.0 if value >= 2.0 else (1.0 if value <= -2.0 else 0.0))
    inverse_signal = signal * -1.0
    realized = (signal * returns).fillna(0.0)
    inverse_realized = (inverse_signal * returns).fillna(0.0)
    fold_rows = _history_forward_walk_rows_from_signals(
        candidate_id=candidate_id,
        lane=NATIVE_LANE,
        history=history,
        signal=signal,
        realized=realized,
    )
    inverse_fold_rows = _history_forward_walk_rows_from_signals(
        candidate_id=candidate_id,
        lane=NATIVE_LANE,
        history=history,
        signal=inverse_signal,
        realized=inverse_realized,
    )
    fold_frame = pd.DataFrame(fold_rows)
    inverse_fold_frame = pd.DataFrame(inverse_fold_rows)
    if not inverse_fold_frame.empty:
        current_pf = _mean_metric(fold_frame, "oos_profit_factor")
        inverse_pf = _mean_metric(inverse_fold_frame, "oos_profit_factor")
        current_sharpe = _mean_metric(fold_frame, "oos_sharpe")
        inverse_sharpe = _mean_metric(inverse_fold_frame, "oos_sharpe")
        if (inverse_pf, inverse_sharpe) > (current_pf, current_sharpe):
            signal = inverse_signal
            realized = inverse_realized
            fold_rows = inverse_fold_rows
    history = history.assign(_signal=signal, _realized=realized)
    fold_frame = pd.DataFrame(fold_rows)
    sharpe = pd.to_numeric(fold_frame.get("oos_sharpe", pd.Series(dtype=float)), errors="coerce")
    pf = pd.to_numeric(fold_frame.get("oos_profit_factor", pd.Series(dtype=float)), errors="coerce")
    dd = pd.to_numeric(fold_frame.get("oos_max_drawdown", pd.Series(dtype=float)), errors="coerce")
    trades = pd.to_numeric(fold_frame.get("oos_trade_count", pd.Series(dtype=float)), errors="coerce")
    signal_trade_count = int((history["_signal"] != 0.0).sum())
    blocker = ""
    detail = ""
    if signal_trade_count == 0:
        blocker = "native_near_zero_signals"
        detail = "history produced zero nonzero trade signals under the current Native evaluation mode"
    elif int(trades.sum()) < thresholds["min_total_trades"]:
        blocker = "native_too_few_trades"
        detail = f"nonzero signal trades {int(trades.sum())} below minimum {thresholds['min_total_trades']}"
    elif round(float(pf.mean()), 6) < thresholds["min_mean_profit_factor"]:
        blocker = "native_weak_profit_factor"
        detail = f"mean fold profit factor {round(float(pf.mean()), 6)} below minimum {thresholds['min_mean_profit_factor']}"
    elif round(float(sharpe.mean()), 6) < thresholds["min_mean_sharpe"] and round(float(pf.mean()), 6) < thresholds["min_profit_factor_for_sharpe_override"]:
        blocker = "native_weak_sharpe"
        detail = f"mean fold sharpe {round(float(sharpe.mean()), 6)} below minimum {thresholds['min_mean_sharpe']}"
    return {
        "selected_history_source": history_path,
        "evaluation_mode": f"{evaluation_mode}:{strategy_family}:{strategy_mode}:{returns_source}:{'inverse' if signal.equals(inverse_signal) else 'standard'}",
        "fold_count": int(len(fold_frame)),
        "nonzero_signal_trade_count": signal_trade_count,
        "mean_fold_profit_factor": _mean_metric(fold_frame, "oos_profit_factor"),
        "mean_fold_sharpe": _mean_metric(fold_frame, "oos_sharpe"),
        "max_fold_drawdown": round(float(dd.max()), 6) if not dd.empty else 0.0,
        "current_blocker": blocker,
        "blocker_detail": detail,
        "threshold_reference": thresholds,
        "evidence_path": history_path,
    }, fold_rows


def _shadow_rows_for_lane(frame: pd.DataFrame, lane: str) -> list[dict[str, object]]:
    rows = []
    for _, row in frame.iterrows():
        blocker = _text(row.get("blocker_state", ""))
        strategy_mode = _text(row.get("strategy_mode", ""))
        regime_snapshot = _text(row.get("regime_snapshot", ""))
        rows.append(
            {
                "candidate_id": _text(row.get("candidate_id", "")),
                "lane": lane,
                "pair": _text(row.get("pair", "")),
                "setup_identity": _text(row.get("setup_identity", "")),
                "strategy_mode": strategy_mode,
                "forward_walk_summary_ref": _text(row.get("forward_walk_summary_ref", "")),
                "paper_outcome_ref": _text(row.get("paper_outcome_ref", "")),
                "state_summary": json.dumps(
                    {
                        "strategy_mode": strategy_mode,
                        "regime_snapshot": regime_snapshot,
                        "confidence": _safe_float(row.get("confidence", 0.0)),
                    },
                    sort_keys=True,
                ),
                "recommended_action": "accept" if blocker == "" else "inspect",
                "shadow_only": True,
                "promotion_authority": "none_shadow_only",
                "evidence_path": _text(row.get("backtest_summary_ref", "")),
            }
        )
    return rows


def _lane_support(candidate_row: pd.Series | None, fw_frame: pd.DataFrame) -> bool:
    if candidate_row is None:
        return False
    candidate_id = str(candidate_row.get("candidate_id", ""))
    fw_row = _match_row(fw_frame, candidate_id, "candidate_id")
    return fw_row is not None and _text(fw_row.get("forward_walk_status", "")).lower() == "pass" and _text(candidate_row.get("blocker_state", "")) == ""


def _arbitration_decision(relation: str, wizard_row: pd.Series | None, native_row: pd.Series | None) -> str:
    if relation == "both_support":
        return "agreement_strongest"
    if relation == "wizard_only":
        return "prefer_wizard"
    if relation == "native_only":
        return "prefer_native"
    if wizard_row is not None and native_row is None:
        return "wizard_reject_only"
    if native_row is not None and wizard_row is None:
        return "native_reject_only"
    return "ignore_both"


def _promotion_stage(ok: bool, overall_row: pd.Series | None, paper_status: str) -> str:
    if not ok:
        return "forward_walk_blocked"
    if overall_row is None:
        return "candidate_validated"
    decision = str(overall_row.get("arbitration_decision", ""))
    if decision in {"prefer_wizard", "prefer_native", "agreement_strongest"} and paper_status == "paper_authorized":
        return "paper_eligible"
    if decision in {"prefer_wizard", "prefer_native", "agreement_strongest"}:
        return "orchestrator_review"
    return "rl_interpretation"


def _wizard_blocker_state(hypothesis_row: pd.Series | None, diagnostic_row: pd.Series | None, parity_row: pd.Series | None, verification_row: pd.Series | None) -> str:
    blockers = []
    if hypothesis_row is not None and str(hypothesis_row.get("hypothesis_status", "")).strip() not in {"ready_for_local_replay", "ready", "pass"}:
        blockers.append(str(hypothesis_row.get("hypothesis_reason", "")))
    if diagnostic_row is not None and str(diagnostic_row.get("diagnostic_blocker", "")).strip():
        blockers.append(str(diagnostic_row.get("diagnostic_blocker", "")))
    if parity_row is not None and str(parity_row.get("parity_status", "")).strip() not in {"match", "aligned", ""}:
        blockers.append(f"parity_{str(parity_row.get('parity_status', '')).lower()}")
    if verification_row is None or str(verification_row.get("verification_status", "")).strip() != "verified":
        blockers.append("wizard_local_verification_missing")
    return _normalize_wizard_blocker_chain([value for value in blockers if value])


def _normalize_wizard_blocker_chain(blockers: list[str]) -> str:
    ordered = []
    seen = set()
    for blocker in blockers:
        text = _text(blocker)
        for token in [part.strip() for part in text.split(";") if part.strip()]:
            if token not in seen:
                ordered.append(token)
                seen.add(token)
    if not ordered:
        return ""
    if "missing_exact_mode" in seen:
        return "missing_exact_mode"
    pair_detail_stage = [value for value in ordered if value in {"missing_correlation", "missing_ecm", "missing_copula_for_copula_mode"}]
    if pair_detail_stage:
        return ";".join(pair_detail_stage)
    return ";".join(ordered)


def _native_blocker_state(row: pd.Series) -> str:
    blockers = []
    if str(row.get("decision_bucket", "")).strip().upper() == "REJECT":
        blockers.append("native_rejected")
    if _safe_float(row.get("local_backtest_score", "")) <= 0:
        blockers.append("native_backtest_missing_or_weak")
    return ";".join(blockers)


def _forward_walk_from_verification(verification_row: pd.Series | None) -> tuple[str, str]:
    if verification_row is None:
        return "missing", "wizard_local_verification_missing"
    if str(verification_row.get("acceptance", "")).strip().upper() != "ACCEPT":
        return "blocked", str(verification_row.get("acceptance_reason", "wizard_local_acceptance_failed"))
    if _safe_int(verification_row.get("local_closed_trades", "")) < 1:
        return "blocked", "insufficient_closed_trades"
    return "pass", ""


def _wizard_confidence(row: pd.Series, verification_row: pd.Series | None) -> float:
    sharpe = _safe_float(row.get("sharpe", ""))
    returns_total = _safe_float(row.get("returns_total", ""))
    local_pf = _safe_float(verification_row.get("local_profit_factor", "")) if verification_row is not None else 0.0
    raw = (min(max(sharpe, 0.0), 3.0) / 3.0) * 0.5 + min(max(returns_total, 0.0), 0.3) / 0.3 * 0.25 + min(max(local_pf, 0.0), 2.0) / 2.0 * 0.25
    return round(max(0.0, min(1.0, raw)), 4)


def _native_confidence(row: pd.Series) -> float:
    acceptance = _safe_float(row.get("acceptance_score", ""))
    combined = _safe_float(row.get("combined_score", ""))
    raw = min(max(acceptance, 0.0), 100.0) / 100.0 * 0.6 + min(max(combined, 0.0), 100.0) / 100.0 * 0.4
    return round(max(0.0, min(1.0, raw)), 4)


def _regime_snapshot(row: pd.Series) -> str:
    half_life = _safe_float(row.get("half_life", ""))
    hurst = _safe_float(row.get("hurst", ""))
    if hurst and hurst < 0.5 and half_life and half_life <= 30:
        return "mean_reversion_favorable"
    if hurst and hurst >= 0.5:
        return "trend_risk"
    return "regime_unknown"


def _wizard_packet_regime_snapshot(
    row: pd.Series,
    *,
    live_row: dict[str, object] | None = None,
    second_page_row: dict[str, object] | None = None,
) -> str:
    base_regime = _regime_snapshot(row)
    if live_row is None and second_page_row is None:
        return base_regime
    payload: dict[str, object] = {"base_regime": base_regime}
    if live_row is not None:
        payload["wizard_live_scanner"] = {
            "updated": _text(live_row.get("updated", "")),
            "normal_zscore": _text(live_row.get("normal_zscore", "")),
            "rolling_zscore": _text(live_row.get("rolling_zscore", "")),
            "dependency": _text(live_row.get("dependency", "")),
            "stationarity": _text(live_row.get("stationarity", "")),
            "risk": _text(live_row.get("risk", "")),
            "reward": _text(live_row.get("reward", "")),
            "row_index": _safe_int(live_row.get("row_index", 0)),
        }
    if second_page_row is not None:
        payload["wizard_strategy_matrix"] = {
            "scanner_rank": _safe_int(second_page_row.get("scanner_rank", 0)),
            "pair_marker": _text(second_page_row.get("marker", "")),
            "strategy_rows": second_page_row.get("backtestRows", []),
            "capture_path": _text(second_page_row.get("_capture_path", "")),
        }
    return json.dumps(payload, sort_keys=True)


def _wizard_feature_bundle_ref(
    row: pd.Series,
    *,
    live_row: dict[str, object] | None = None,
    second_page_row: dict[str, object] | None = None,
) -> str:
    refs = [_text(row.get("evidence_path", "data/processed/wizard_evidence.csv"))]
    if live_row is not None:
        refs.append(_text(live_row.get("_capture_path", "")))
    if second_page_row is not None:
        refs.append(_text(second_page_row.get("_capture_path", "")))
    return ";".join([ref for ref in refs if ref])


def _wizard_backtest_summary_ref(
    row: pd.Series,
    *,
    verification_row: pd.Series | None,
    second_page_row: dict[str, object] | None,
    exact_mode: str,
) -> str:
    if verification_row is not None:
        summary = _text(verification_row.get("summary_path", ""))
        if summary:
            return summary
    if second_page_row is not None:
        capture_path = _text(second_page_row.get("_capture_path", ""))
        strategy_rows = second_page_row.get("backtestRows", [])
        match = _wizard_second_page_strategy_match(strategy_rows if isinstance(strategy_rows, list) else [], exact_mode)
        if capture_path and match is not None:
            return f"{capture_path}#{_text(match.get('strategy', ''))}"
        if capture_path:
            return capture_path
    return _text(row.get("evidence_path", ""))


def _wizard_packet_provenance(*, live_row: dict[str, object] | None, second_page_row: dict[str, object] | None) -> str:
    parts = ["wizard_evidence_chain"]
    if live_row is not None:
        parts.append("wizard_live_scanner_capture")
    if second_page_row is not None:
        parts.append("wizard_second_page_capture")
    return "+".join(parts)


def _load_wizard_live_scanner_context(root: Path) -> dict[str, dict[str, object]]:
    path = root / "data" / "collected" / "wizard_scanner_static_spread_visible_live" / "wizard_scanner_static_spread_visible_live_latest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return {}
    context: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        pair_key = _wizard_context_pair_key(row.get("pair", ""))
        if not pair_key:
            continue
        enriched = dict(row)
        enriched["_capture_path"] = str(path.relative_to(root))
        context[pair_key] = enriched
    return context


def _load_wizard_second_page_context(root: Path) -> dict[str, dict[str, object]]:
    path = root / "data" / "collected" / "wizard_second_page_static_spread_green" / "wizard_second_page_static_spread_green_latest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, list):
        return {}
    context: dict[str, dict[str, object]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        pair = f"{_text(row.get('pair_x', ''))} / {_text(row.get('pair_y', ''))}".strip(" /")
        pair_key = _wizard_context_pair_key(pair)
        if not pair_key:
            continue
        enriched = dict(row)
        enriched["_capture_path"] = str(path.relative_to(root))
        context[pair_key] = enriched
    return context


def _match_wizard_live_context_row(context: dict[str, dict[str, object]], pair: str) -> dict[str, object] | None:
    if not context:
        return None
    return context.get(_wizard_context_pair_key(pair))


def _wizard_context_pair_key(value: object) -> str:
    return " ".join(_normalize_variant_text(value).split())


def _wizard_second_page_strategy_match(strategy_rows: list[dict[str, object]], exact_mode: str) -> dict[str, object] | None:
    target = _normalize_variant_text(exact_mode)
    if not target:
        return None
    mapping = {
        "static spread": "staticspread",
        "static zscorer": "staticzscorer",
        "dyn spread": "dynspread",
        "dyn zscorer": "dynzscorer",
        "ou spread": "ouspread",
        "ou zscorer": "ouzscorer",
        "copula": "copula",
    }
    target_compact = mapping.get(target, target.replace(" ", ""))
    for row in strategy_rows:
        strategy = _text(row.get("strategy", ""))
        if _normalize_variant_text(strategy).replace(" ", "") == target_compact:
            return row
    return None


def _paper_journal_matches(journal: pd.DataFrame, pair: str) -> bool:
    if journal.empty or "pair" not in journal.columns:
        return False
    return journal["pair"].astype(str).eq(pair).any()


def _match_paper_journal_row(journal: pd.DataFrame, pair: str) -> pd.Series | None:
    if journal.empty or "pair" not in journal.columns:
        return None
    matches = journal[journal["pair"].astype(str) == pair]
    if matches.empty:
        return None
    return matches.iloc[-1]


def _match_row(frame: pd.DataFrame, value: str, column: str) -> pd.Series | None:
    if frame.empty or column not in frame.columns:
        return None
    matches = frame[frame[column].astype(str) == value]
    if matches.empty:
        return None
    return matches.iloc[0]


def _match_setup_or_pair_row(frame: pd.DataFrame, pair: str, setup_identity: str) -> pd.Series | None:
    if not frame.empty and "setup_identity" in frame.columns and _text(setup_identity):
        row = _match_row(frame, _text(setup_identity), "setup_identity")
        if row is not None:
            return row
    row = _match_row(frame, _text(pair), "pair")
    if row is not None or frame.empty or "pair" not in frame.columns:
        return row
    pair_norm = _normalize_variant_text(pair)
    candidates = frame.copy()
    candidates["_pair_norm"] = candidates.get("pair", pd.Series(dtype=object)).map(_normalize_variant_text)
    normalized_matches = candidates.loc[candidates["_pair_norm"] == pair_norm]
    if normalized_matches.empty:
        return None
    if "setup_role" in normalized_matches.columns:
        normalized_matches = normalized_matches.copy()
        normalized_matches["_primary"] = normalized_matches.get("setup_role", pd.Series(dtype=object)).astype(str).eq("primary")
        normalized_matches = normalized_matches.sort_values(["_primary"], ascending=[False])
    return normalized_matches.iloc[0]


def _match_lane_pair_row(frame: pd.DataFrame, pair: str, *, prefer_primary: bool = False) -> pd.Series | None:
    if frame.empty or "pair" not in frame.columns:
        return None
    matches = frame[frame["pair"].astype(str) == pair].copy()
    if matches.empty:
        pair_norm = _normalize_variant_text(pair)
        normalized = frame.copy()
        normalized["_pair_norm"] = normalized.get("pair", pd.Series(dtype=object)).map(_normalize_variant_text)
        matches = normalized.loc[normalized["_pair_norm"] == pair_norm].copy()
    if matches.empty:
        return None
    if prefer_primary:
        matches["_primary"] = matches.get("setup_role", pd.Series(dtype=str)).astype(str).eq("primary")
        matches["_rank"] = pd.to_numeric(matches.get("setup_rank", pd.Series(dtype=float)), errors="coerce").fillna(999999)
        matches = matches.sort_values(["_primary", "_rank"], ascending=[False, True])
    return matches.iloc[0]


def _preferred_promotion_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    working = frame.copy()
    working["_credible"] = working.get("paper_credible", pd.Series(dtype=object)).astype(bool)
    working["_primary"] = working.get("setup_role", pd.Series(dtype=object)).astype(str).eq("primary")
    working["_stage_rank"] = working.get("promotion_stage", pd.Series(dtype=object)).astype(str).map(
        {
            "paper_eligible": 0,
            "orchestrator_review": 1,
            "candidate_validated": 2,
            "rl_interpretation": 3,
            "forward_walk_blocked": 4,
        }
    ).fillna(9)
    working["_route_rank"] = pd.to_numeric(working.get("route_priority", pd.Series(dtype=object)), errors="coerce").fillna(9)
    working["_support_rank"] = working.get("model_support_status", pd.Series(dtype=object)).astype(str).map(
        {
            "strong_model_support": 0,
            "weak_model_support": 1,
            "pair_missing_from_model_predictions": 2,
            "no_model_support": 3,
            "model_predictions_missing": 4,
        }
    ).fillna(9)
    working["_support_trades"] = pd.to_numeric(working.get("model_taken_trades", pd.Series(dtype=object)), errors="coerce").fillna(0)
    working = working.sort_values(
        ["_credible", "_stage_rank", "_route_rank", "_support_rank", "_support_trades", "_primary"],
        ascending=[False, True, True, True, False, False],
    )
    return working.iloc[0]


def _candidate_id(lane: str, pair: str, strategy_mode: str, timeframe: str) -> str:
    pair_key = _text(pair).replace("/", "-").replace(" ", "").upper()
    mode_key = _text(strategy_mode).replace(" ", "_").replace("/", "_").lower()
    time_key = _text(timeframe).replace(" ", "_").replace("/", "_").lower()
    return f"{lane}:{pair_key}:{mode_key}:{time_key}"


def _pair_for_candidate(root: Path, lane: str, candidate_id: str) -> str:
    packets = _read_csv(root / "reports" / "brain" / f"{lane}_candidate_packets.csv")
    row = _match_row(packets, candidate_id, "candidate_id")
    return _text(row.get("pair", "")) if row is not None else ""


def _forward_walk_strength_tier(*, status: str, fold_count: int, trade_count: int, profit_factor: float, sharpe: float) -> str:
    if status != "pass":
        return "blocked"
    if fold_count >= 3 and trade_count >= 9 and profit_factor >= 1.1 and sharpe >= 0.5:
        return "strong"
    if fold_count >= 2 and trade_count >= 3 and profit_factor >= 1.0:
        return "acceptable"
    return "thin"


def _evidence_density_tier(*, fold_count: int, trade_count: int) -> str:
    if fold_count >= 3 and trade_count >= 9:
        return "dense"
    if fold_count >= 2 and trade_count >= 3:
        return "moderate"
    return "thin"


def _native_origin_type(row: pd.Series) -> str:
    if _text(row.get("native_discovery_status", "")) and _text(row.get("dydx_local_discovery_status", "")):
        return "local_plus_dydx_history"
    if _text(row.get("native_discovery_status", "")):
        return "local_manifest_only"
    if _text(row.get("dydx_local_discovery_status", "")):
        return "dydx_history_only"
    return "pair_universe_only"


def _native_source_type(row: pd.Series) -> str:
    return f"native_{_native_origin_type(row)}"


def _native_forward_walk_thresholds() -> dict[str, float]:
    return {
        "min_total_trades": 3,
        "min_mean_profit_factor": 1.0,
        "min_mean_sharpe": 0.1,
        "min_profit_factor_for_sharpe_override": 2.0,
    }


def _native_strategy_family(row: pd.Series) -> str:
    for key in ["native_strategy_family", "local_strategy_family", "strategy_family"]:
        value = _text(row.get(key, ""))
        if value:
            return value
    return "native_local_math"


def _native_strategy_mode(row: pd.Series) -> str:
    for key in ["native_strategy_name", "local_strategy_name", "strategy_mode"]:
        value = _text(row.get(key, ""))
        if value:
            return value
    bucket = _text(row.get("decision_bucket", ""))
    return bucket or "native_ranked"


def _pair_status_from_overall(row: pd.Series, pair_promotion: pd.DataFrame) -> str:
    if not pair_promotion.empty and pair_promotion.get("paper_credible", pd.Series(dtype=bool)).astype(bool).any():
        return "paper_candidate"
    agreement = _text(row.get("agreement_status", ""))
    if agreement in {"wizard_only", "native_only"}:
        return "comparison_only"
    if agreement == "both_reject":
        return "blocked"
    return "eligible"


def _safe_float(value: object) -> float:
    try:
        if value in {"", None}:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: object) -> int:
    try:
        if value in {"", None}:
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _native_repair_action(diagnosis: dict[str, object]) -> str:
    blocker = _text(diagnosis.get("current_blocker", ""))
    if blocker == "native_no_usable_history":
        return "attach_deeper_native_history_then_rerun_forward_walk"
    if blocker == "native_no_zscore_like_column":
        return "rebuild_native_history_with_signal_columns_or_import_strategy_summary"
    if blocker == "native_near_zero_signals":
        return "correct_native_strategy_mode_or_signal_thresholds_then_rerun"
    if blocker == "native_too_few_trades":
        return "extend_native_history_or_relax_fold_window_then_rerun"
    if blocker == "native_weak_profit_factor":
        return "improve_pair_specific_entry_exit_logic_then_rerun"
    if blocker == "native_weak_sharpe":
        return "reduce_native_noise_and_retest_with_strategy_specific_logic"
    return "candidate_clear_monitor_for_promotion"


def _paper_route_followup_action(row: pd.Series | dict[str, object]) -> str:
    paper_status = _text(row.get("paper_status", ""))
    if not paper_status and _text(row.get("promotion_stage", "")) == "orchestrator_review":
        paper_status = "research_only"
    if paper_status == "research_only":
        return "advance_global_paper_handoff_then_recheck_candidate"
    if paper_status == "global_gate_blocked":
        return "clear_global_gate_then_recheck_candidate"
    if paper_status == "blocked":
        return "clear_paper_route_blocker_then_recheck_candidate"
    if paper_status:
        return "recheck_candidate_against_current_paper_route"
    return ""


def _paper_route_followup_context(root: Path, paper_gate_blocker: str) -> tuple[str, str]:
    preflight = _read_csv(root / "reports" / "paper_execution_preflight.csv")
    if preflight.empty:
        return "", ""
    paper_row = _match_row(preflight, "paper_submission_gate", "step")
    dydx_row = _match_row(preflight, "dydx_testnet_dependency", "step")
    if paper_gate_blocker == "paper_execution_not_ready" and dydx_row is not None and not bool(dydx_row.get("ready", False)):
        return (
            _text(dydx_row.get("next_action", "")),
            f"execution_preflight_blocker={_text(dydx_row.get('blocker', ''))}",
        )
    if paper_row is not None and not bool(paper_row.get("ready", False)):
        return (
            _text(paper_row.get("next_action", "")),
            f"execution_preflight_blocker={_text(paper_row.get('blocker', ''))}",
        )
    return "", ""


def _strategy_acceptance_followup(root: Path) -> tuple[str, str, str]:
    readiness = _read_csv(root / "reports" / "priority_readiness.csv")
    checklist = _read_csv(root / "reports" / "strategy_acceptance_checklist.csv")
    acceptance = _read_csv(root / "reports" / "acceptance_report.csv")

    readiness_row = _match_row(readiness, "strategy_acceptance", "gate")
    checklist_row = _match_row(checklist, "production_eligibility", "step")
    acceptance_row = acceptance.iloc[0] if not acceptance.empty else None

    readiness_blocker = _text(readiness_row.get("blocker", "")) if readiness_row is not None else ""
    readiness_ready = bool(readiness_row.get("ready", False)) if readiness_row is not None else False
    readiness_status = _text(readiness_row.get("status", "")) if readiness_row is not None else ""
    blocker = readiness_blocker
    if readiness_ready or readiness_status == "ready":
        blocker = ""
    elif not blocker:
        blocker = "strategy_acceptance_not_ready"

    production_blocker = _text(checklist_row.get("blocker", "")) if checklist_row is not None else ""
    top_reason = _text(acceptance_row.get("acceptance_reason", "")) if acceptance_row is not None else ""
    next_step = _text(acceptance_row.get("research_next_step", "")) if acceptance_row is not None else ""
    cost_alignment = _text(_match_row(checklist, "exchange_cost_model_alignment", "step").get("blocker", "")) if _match_row(checklist, "exchange_cost_model_alignment", "step") is not None else ""

    detail_parts = [
        f"production_gate={production_blocker}" if production_blocker else "",
        f"top_failure={top_reason}" if top_reason else "",
        f"next_step={next_step}" if next_step else "",
        f"cost_alignment={cost_alignment}" if cost_alignment else "",
    ]
    detail = "; ".join([part for part in detail_parts if part])

    action = "expand_history_and_retest_for_strategy_acceptance"
    return blocker, detail, action


def _model_gate_followup(root: Path) -> tuple[str, str]:
    acceptance = _read_csv(root / "reports" / "ml" / "model_gated_acceptance.csv")
    if acceptance.empty:
        return "", ""
    row = acceptance.iloc[0]
    failing_checks = _text(row.get("failing_checks", ""))
    detail_parts = [
        f"best_model={_text(row.get('best_model', ''))}" if _text(row.get("best_model", "")) else "",
        f"failing_checks={failing_checks}" if failing_checks else "",
        f"filtered_profit_factor={_text(row.get('filtered_profit_factor', ''))}" if _text(row.get("filtered_profit_factor", "")) else "",
        f"filtered_sharpe={_text(row.get('filtered_sharpe', ''))}" if _text(row.get("filtered_sharpe", "")) else "",
        f"filtered_drawdown={_text(row.get('filtered_drawdown', ''))}" if _text(row.get("filtered_drawdown", "")) else "",
        f"take_rate={_text(row.get('take_rate', ''))}" if _text(row.get("take_rate", "")) else "",
        f"trades={_text(row.get('trades', ''))}" if _text(row.get("trades", "")) else "",
    ]
    action = "improve model gate acceptance before base RL paper handoff"
    if "filtered_profit_factor_min" in failing_checks and "filtered_sharpe_positive" in failing_checks:
        action = "improve filtered profit factor and filtered sharpe before base RL paper handoff"
    elif "filtered_profit_factor_min" in failing_checks:
        action = "improve filtered profit factor before base RL paper handoff"
    elif "filtered_sharpe_positive" in failing_checks:
        action = "improve filtered sharpe before base RL paper handoff"
    elif "filtered_drawdown_max" in failing_checks or "drawdown_delta_nonpositive" in failing_checks:
        action = "reduce filtered drawdown before base RL paper handoff"
    elif "take_rate_min" in failing_checks or "filtered_trades_min" in failing_checks:
        action = "raise filtered take-rate and trade count before base RL paper handoff"
    elif "score_buckets_monotonic" in failing_checks:
        action = "restore score-bucket monotonicity before base RL paper handoff"
    return action, "; ".join(part for part in detail_parts if part)


def _native_pair_model_support(root: Path, pair: str) -> dict[str, object]:
    predictions = _read_csv(root / "reports" / "ml" / "model_walkforward_predictions.csv")
    try:
        from quant_platform.active_pipeline import _filter_predictions_to_selected_model  # local import avoids module cycle at import time
        predictions = _filter_predictions_to_selected_model(root, predictions)
    except Exception:
        pass
    if predictions.empty or "pair" not in predictions.columns:
        return {
            "support_status": "model_predictions_missing",
            "rows": 0,
            "taken_trades": 0,
            "take_rate": 0.0,
            "profit_factor": 0.0,
            "mean_return": 0.0,
        }
    subset = predictions[predictions["pair"].astype(str) == pair].copy()
    if subset.empty:
        training = _read_csv(root / "data" / "ml" / "trade_training_dataset.csv")
        training_subset = (
            training[training["pair"].astype(str) == pair].copy()
            if not training.empty and "pair" in training.columns
            else pd.DataFrame()
        )
        return {
            "support_status": "pair_missing_from_model_predictions" if not training_subset.empty else "no_model_support",
            "rows": int(len(training_subset)),
            "taken_trades": 0,
            "take_rate": 0.0,
            "profit_factor": 0.0,
            "mean_return": 0.0,
        }
    taken = subset[subset.get("shadow_take", pd.Series(dtype=object)).fillna(False).astype(bool)].copy()
    returns = pd.to_numeric(taken.get("realized_return", pd.Series(dtype=object)), errors="coerce").dropna()
    gross_profit = float(returns[returns > 0].sum()) if not returns.empty else 0.0
    gross_loss = float(-returns[returns < 0].sum()) if not returns.empty else 0.0
    profit_factor = float("inf") if gross_loss == 0.0 and gross_profit > 0.0 else (gross_profit / gross_loss if gross_loss > 0.0 else 0.0)
    take_rate = float(len(taken) / len(subset)) if len(subset) else 0.0
    mean_return = float(returns.mean()) if not returns.empty else 0.0
    support_status = "weak_model_support"
    if len(taken) == 0:
        support_status = "no_model_support"
    elif len(returns) >= 10 and profit_factor >= 1.2 and mean_return > 0:
        support_status = "strong_model_support"
    return {
        "support_status": support_status,
        "rows": int(len(subset)),
        "taken_trades": int(len(returns)),
        "take_rate": take_rate,
        "profit_factor": profit_factor,
        "mean_return": mean_return,
    }


def _wizard_hourly_target_value(frame: pd.DataFrame, candidate_id: str, column: str) -> object:
    row = _match_row(frame, candidate_id, "candidate_id")
    if row is None:
        return False if column.startswith("pair_") else ""
    return row.get(column, False if column.startswith("pair_") else "")


def _wizard_repair_loop_action(repair_row: pd.Series | None, hourly_row: pd.Series | None) -> str:
    base = _text(repair_row.get("next_repair_action", "")) if repair_row is not None else "capture_exact_wizard_setup_then_rerun"
    if hourly_row is None:
        return base
    repair_blocker = _text(repair_row.get("current_blocker", "")) if repair_row is not None else ""
    exact_mode_status = _text(repair_row.get("exact_mode_capture_status", "")) if repair_row is not None else ""
    still_needs_capture = (
        "missing_exact_mode" in repair_blocker
        or "missing_correlation" in repair_blocker
        or "missing_ecm" in repair_blocker
        or exact_mode_status != "captured"
    )
    review_status = _text(hourly_row.get("review_status", ""))
    candidate_status = _text(hourly_row.get("candidate_status", ""))
    pair_recommended = bool(hourly_row.get("pair_recommended_this_hour", False))
    if pair_recommended and review_status in {"new", "changed"} and candidate_status == "review_now":
        if not still_needs_capture:
            return "rerun_local_verification_on_current_wizard_setup"
        return "use_hourly_wizard_capture_then_rerun_local_verification"
    if pair_recommended and review_status == "unchanged":
        return "keep_current_wizard_repair_path_and_wait_for_next_change"
    return base


def _wizard_hourly_target_action(hourly_match: pd.Series | None, pair_match: pd.Series | None) -> str:
    if hourly_match is not None:
        review_status = _text(hourly_match.get("review_status", ""))
        candidate_status = _text(hourly_match.get("candidate_status", ""))
        if review_status in {"new", "changed"} and candidate_status == "review_now":
            return "pull_hourly_setup_into_wizard_repair_loop"
        if review_status == "unchanged":
            return "hold_until_hourly_variant_changes"
        if candidate_status == "blocked_by_anomaly":
            return "review_hourly_anomalies_before_repair"
    if pair_match is not None and bool(pair_match.get("recommended_this_hour", False)):
        return "capture_best_hourly_variant_for_pair"
    return "stale_or_missing_from_live_dashboard_take_next_best_live_pair"


def _wizard_live_capture_status(*, needs_exact_mode: bool, hourly_row: pd.Series | None) -> str:
    if not needs_exact_mode:
        return "not_exact_mode_capture"
    if hourly_row is None:
        return "live_expired"
    hourly_match_status = _text(hourly_row.get("hourly_match_status", ""))
    pair_recommended = bool(hourly_row.get("pair_recommended_this_hour", False))
    if hourly_match_status == "matched":
        return "exact_setup_live"
    if pair_recommended:
        return "pair_live_but_exact_setup_missing"
    return "live_expired"


def _match_wizard_hourly_candidate(frame: pd.DataFrame, pair: str, timeframe: str, strategy_mode: str) -> pd.Series | None:
    if frame.empty:
        return None
    pair_norm = _normalize_variant_text(pair)
    timeframe_norm = _normalize_variant_text(timeframe)
    strategy_norm = _normalize_variant_text(strategy_mode)
    candidates = frame.copy()
    candidates["_pair_norm"] = candidates.get("pair", pd.Series(dtype=object)).map(_normalize_variant_text)
    candidates["_timeframe_norm"] = candidates.get("timeframe", pd.Series(dtype=object)).map(_normalize_variant_text)
    candidates["_strategy_norm"] = candidates.get("strategy", pd.Series(dtype=object)).map(_normalize_variant_text)
    candidates["_review_rank"] = candidates.get("review_status", pd.Series(dtype=object)).map(
        lambda value: 2 if _text(value) == "changed" else 1 if _text(value) == "new" else 0
    )
    exact = candidates.loc[
        (candidates["_pair_norm"] == pair_norm)
        & (candidates["_timeframe_norm"] == timeframe_norm)
        & (candidates["_strategy_norm"] == strategy_norm)
    ]
    if not exact.empty:
        exact = exact.sort_values(["_review_rank"], ascending=False)
        return exact.iloc[0]
    pair_only = candidates.loc[candidates["_pair_norm"] == pair_norm]
    if pair_only.empty:
        return None
    pair_only = pair_only.sort_values(["_review_rank"], ascending=False)
    return pair_only.iloc[0]


def _normalize_variant_text(value: object) -> str:
    return _text(value).replace("_", " ").replace("-", " ").replace("/", " ").replace("|", " ").lower()


def _wizard_blocker_status(blocker_text: str, token: str) -> str:
    return "missing" if token in blocker_text else "covered"


def _wizard_effective_repair_blocker(
    *,
    blocker_text: str,
    diagnostic_blocker_text: str,
    hourly_row: pd.Series | None,
    scanner_dependency_row: pd.Series | None,
) -> str:
    effective = blocker_text
    if hourly_row is not None and _text(hourly_row.get("matched_hourly_strategy", "")):
        diagnostic_has_exact_mode_gap = "missing_exact_mode" in diagnostic_blocker_text.split(";")
        if diagnostic_has_exact_mode_gap:
            effective = _normalize_wizard_blocker_chain([effective])
        elif effective == "missing_exact_mode":
            effective = ";".join(
                dict.fromkeys(part.strip() for part in diagnostic_blocker_text.split(";") if part.strip())
            )
        elif effective.startswith("missing_exact_mode;"):
            trimmed = ";".join(part for part in effective.split(";") if part and part != "missing_exact_mode")
            effective = _normalize_wizard_blocker_chain([trimmed or diagnostic_blocker_text or effective])
    if scanner_dependency_row is not None:
        effective = ";".join(part for part in effective.split(";") if part and part != "missing_correlation")
    return effective


def _wizard_exact_mode_capture_status(row: pd.Series, blocker_text: str, *, hourly_row: pd.Series | None = None) -> str:
    setup_identity = _text(row.get("setup_identity", ""))
    exact_mode = _text(row.get("exact_mode", "")) or _text(row.get("strategy_mode", ""))
    spread_id = _text(row.get("spread_id", ""))
    strategy_id = _text(row.get("strategy_id", ""))
    hourly_mode = _text(hourly_row.get("matched_hourly_strategy", "")) if hourly_row is not None else ""
    if "missing_exact_mode" in blocker_text:
        return "missing"
    if hourly_mode or exact_mode or (spread_id and strategy_id):
        return "captured"
    if "|nan" in setup_identity or "||" in setup_identity:
        return "placeholder"
    return "captured"


def _match_wizard_scanner_dependency_row(
    frame: pd.DataFrame,
    *,
    pair: str,
    timeframe: str,
    strategy_mode: str,
) -> pd.Series | None:
    if frame.empty:
        return None
    pair_norm = _normalize_variant_text(pair)
    timeframe_norm = _normalize_variant_text(timeframe)
    strategy_norm = _normalize_variant_text(strategy_mode)
    working = frame.copy()
    working["_pair_norm"] = working.get("pair", pd.Series(dtype=object)).map(_normalize_variant_text)
    working["_timeframe_norm"] = working.get("timeframe", pd.Series(dtype=object)).map(_normalize_variant_text)
    working["_strategy_norm"] = working.get("strategy_mode", pd.Series(dtype=object)).map(_normalize_variant_text)
    exact = working.loc[
        (working["_pair_norm"] == pair_norm)
        & (working["_timeframe_norm"] == timeframe_norm)
        & (working["_strategy_norm"] == strategy_norm)
    ]
    if exact.empty:
        return None
    usable = exact.loc[pd.to_numeric(exact.get("corr_jneg", pd.Series(dtype=object)), errors="coerce").notna()]
    if usable.empty:
        return None
    return usable.iloc[0]


def _wizard_repair_action(blocker_text: str) -> str:
    if "missing_exact_mode" in blocker_text:
        return "capture_exact_mode_from_live_wizard_pair_page"
    if "missing_correlation" in blocker_text:
        return "capture_correlation_panel_and_persist_setup_metrics"
    if "missing_ecm" in blocker_text:
        return "import_ecm_fields_for_exact_wizard_setup"
    if "parity_missing_local_data" in blocker_text:
        return "rebuild_local_pair_detail_history_for_wizard_parity"
    if "wizard_local_verification_missing" in blocker_text:
        return "run_setup_specific_local_verification_batch"
    return "wizard_setup_clear_monitor_for_promotion"


def _regime_bucket_from_snapshot(snapshot: str) -> str:
    text = _text(snapshot)
    if not text:
        return "regime_unknown"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key in ("base_regime", "regime", "regime_bucket", "market_regime"):
            value = _text(payload.get(key, ""))
            if value:
                return value.lower()
    return text.lower()


def _outcome_label(realized_return: float, result_status: str, verification_status: str) -> str:
    status = _text(result_status).lower()
    verification = _text(verification_status).lower()
    if verification and verification != "verified":
        return "pending"
    if status == "broadcast_accepted_unconfirmed":
        return "pending"
    if status in {"paper_submitted", "submitted", "confirmed_on_exchange"}:
        return "pending"
    if realized_return > 0:
        return "win"
    if realized_return < 0:
        return "loss"
    return "flat"


def _paper_journal_verification_status(row: pd.Series) -> str:
    status = _text(row.get("plan_status", "")).lower()
    if status not in {"paper_completed", "completed", "closed"}:
        return "unverified"
    try:
        float(_text(row.get("realized_return", "")))
    except (TypeError, ValueError):
        return "unverified"
    snapshot = _json_dict(row.get("exit_snapshot_json", ""))
    if not snapshot:
        return "unverified"
    for key, value in snapshot.items():
        key_text = str(key).lower()
        if "exit" in key_text and "price" in key_text and _text(value):
            return "verified"
    return "unverified"


def _json_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    text = _text(value)
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _outcome_state(result_status: str, verification_status: str) -> str:
    status = _text(result_status).lower()
    verification = _text(verification_status).lower()
    if verification == "verified":
        return "outcome_verified"
    if status == "broadcast_accepted_unconfirmed":
        return "audit_only"
    if status in {"paper_completed", "completed", "closed"}:
        return "paper_completed"
    if status in {"paper_submitted", "submitted", "paper_ready", "confirmed_on_exchange"}:
        return "paper_submitted"
    if status in {"audit_only", "simulated_only"}:
        return status
    return status or "unknown"


def _learning_weight(outcome_state: str, verification_status: str) -> float:
    state = _text(outcome_state).lower()
    verification = _text(verification_status).lower()
    if state == "outcome_verified":
        return 1.0
    if state == "paper_completed":
        return 0.5 if verification else 0.25
    if state in {"paper_submitted"}:
        return 0.25
    return 0.0


def _subset_equals(frame: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame(columns=frame.columns)
    return frame[frame[column].astype(str).eq(value)].copy()


def _win_rate(frame: pd.DataFrame) -> float:
    if frame.empty or "outcome_label" not in frame.columns:
        return 0.0
    labels = frame["outcome_label"].astype(str)
    decided = labels.isin(["win", "loss"])
    if int(decided.sum()) == 0:
        return 0.0
    wins = int(labels.eq("win").sum())
    return round(wins / int(decided.sum()), 6)


def _mean_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return round(float(values.mean()), 6)


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _trade_log_path_from_summary(summary_path: object) -> str:
    summary = _text(summary_path)
    if not summary:
        return ""
    if summary.endswith("_after_cost.csv"):
        return summary.replace("_after_cost.csv", "_trade_log.csv")
    return ""


def _fold_slices(length: int, folds: int) -> list[tuple[int, int]]:
    if length <= 0 or folds <= 0:
        return []
    fold_size = max(1, length // folds)
    slices: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(length, start + fold_size)
        slices.append((start, end))
        start = end
    return slices[-folds:]


def _returns_metrics(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    positives = values[values > 0]
    negatives = values[values < 0]
    gross_profit = float(positives.sum()) if not positives.empty else 0.0
    gross_loss = float((-negatives).sum()) if not negatives.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    std = float(values.std(ddof=0)) if len(values) > 1 else 0.0
    sharpe = float(values.mean() / std) if std > 0 else 0.0
    growth = (1.0 + values).cumprod()
    peak = growth.cummax()
    drawdown = ((peak - growth) / peak.replace(0, pd.NA)).fillna(0.0)
    max_drawdown = float(drawdown.max()) if not drawdown.empty else 0.0
    return {
        "profit_factor": round(profit_factor, 6),
        "sharpe": round(sharpe, 6),
        "max_drawdown": round(max_drawdown, 6),
    }


def _normalize_pair_from_assets(pair_id: object, asset_x: object, asset_y: object) -> str:
    if _text(pair_id):
        return _text(pair_id).replace("/", "-").replace("_", "-").upper()
    left = _text(asset_x).replace("/", "-").replace("_", "-").upper()
    right = _text(asset_y).replace("/", "-").replace("_", "-").upper()
    if left and right:
        return f"{left}-{right}"
    return ""


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
