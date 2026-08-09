from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.rl.brain_contract import (
    BRAIN_MEMORY_AGENT,
    BRAIN_SCHEMA_VERSION,
    CANDIDATE_SETUP_PACKET_VERSION,
    FORWARD_WALK_PACKET_VERSION,
    PAPER_OUTCOME_PACKET_VERSION,
    PHASE1_SPINE_SCHEMA_VERSION,
    WIZARD_LANE,
    append_candidate_setup_packets,
    append_brain_memory,
    append_forward_walk_packets,
    build_overall_candidate_summary,
    build_phase1_readiness_surfaces,
    candidate_paper_credibility_status,
    brain_output_paths,
    normalize_pair,
    _clean_text,
    _coerce_blockers,
    _overall_readiness_row,
    _paper_status_row,
    valid_candidate_setup_packet,
    valid_brain_suggestion_row,
    valid_forward_walk_packet,
    valid_memory_event,
    valid_paper_outcome_packet,
    validate_candidate_setup_frame,
    validate_suggestion_frame,
    write_candidate_setup_frame,
    write_cycle_summary,
    write_suggestion_frame,
)
from quant_platform import active_pipeline
from quant_platform.rl.brain_cycle import build_brain_readiness_report, score_brain_cycle_readiness


def test_brain_pair_normalization():
    assert normalize_pair("eth-btc") == "ETH-BTC"
    assert normalize_pair("ETH_BTC") == "ETH-BTC"
    assert normalize_pair("ETH/BTC") == "ETH-BTC"


def test_brain_suggestion_schema_validation():
    frame = pd.DataFrame(
        [
            {
                "cycle_id": "2026",
                "pair": "ETH-BTC",
                "strategy_name": "static_spread",
                "variant": "candidate_001",
                "entry_logic": "entry_abs_zscore >= 0.65",
                "exit_logic": "fixed_exit",
                "entry_threshold": 0.65,
                "hold_bars_min": 4,
                "hold_bars_max": 12,
                "confidence": 0.77,
                "expected_return_delta": 0.012,
                "expected_drawdown_delta": -0.002,
                "evidence_path": "reports/rl/rl_learning_cycle_summary.csv",
                "blocker": "",
                "status": "ready",
                "created_at": "2026-06-28T00:00:00Z",
                "schema_version": BRAIN_SCHEMA_VERSION,
            }
        ]
    )

    ok, reason = validate_suggestion_frame(frame)
    assert ok is True
    assert reason == "valid"


def test_brain_suggestion_schema_blocks_bad_confidence():
    frame = pd.DataFrame(
        [
            {
                "cycle_id": "2026",
                "pair": "ETH-BTC",
                "strategy_name": "static_spread",
                "variant": "candidate_001",
                "entry_logic": "entry_abs_zscore >= 0.65",
                "exit_logic": "fixed_exit",
                "entry_threshold": 0.65,
                "hold_bars_min": 4,
                "hold_bars_max": 12,
                "confidence": 1.7,
                "expected_return_delta": 0.012,
                "expected_drawdown_delta": -0.002,
                "evidence_path": "reports/rl/rl_learning_cycle_summary.csv",
                "blocker": "",
                "status": "ready",
                "created_at": "2026-06-28T00:00:00Z",
                "schema_version": BRAIN_SCHEMA_VERSION,
            }
        ]
    )
    ok, reason = validate_suggestion_frame(frame)
    assert ok is False
    assert reason == "confidence_out_of_range"


def test_brain_writes_and_validates_memory_event(tmp_path):
    event = {
        "timestamp": "2026-06-28T00:00:00Z",
        "agent": BRAIN_MEMORY_AGENT,
        "task_id": "brain:001",
        "task_type": "run_brain_cycle",
        "cycle_id": "20260628",
        "pair": "ETH-BTC",
        "outcome_known": True,
        "outcome_label": "passed",
        "blocker": "",
        "next_step": "review",
    }
    assert valid_memory_event(event) == (True, "valid")


def test_brain_writes_suggestion_and_summary(tmp_path: Path):
    frame = pd.DataFrame(
        [
            {
                "cycle_id": "20260628",
                "pair": "eth/btc",
                "strategy_name": "static_spread",
                "variant": "candidate_001",
                "entry_logic": "entry_abs_zscore >= 0.65",
                "exit_logic": "time_exit",
                "entry_threshold": 0.65,
                "hold_bars_min": 4,
                "hold_bars_max": 12,
                "confidence": 0.7,
                "expected_return_delta": 0.01,
                "expected_drawdown_delta": -0.001,
                "evidence_path": "reports/rl/rl_learning_cycle_summary.csv",
                "blocker": "",
                "status": "ready",
                "created_at": "2026-06-28T00:00:00Z",
                "schema_version": BRAIN_SCHEMA_VERSION,
            }
        ]
    )
    paths = brain_output_paths(tmp_path, "20260628")
    csv_path = write_suggestion_frame(frame, paths["candidate_csv"])
    summary = pd.read_csv(csv_path)

    assert summary["pair"].iloc[0] == "ETH-BTC"
    ok, reason = valid_brain_suggestion_row(summary.iloc[0].to_dict())
    assert ok
    assert reason == "valid"

    summary_path = write_cycle_summary(paths["cycle_summary"], {"cycle_id": "20260628", "status": "ready"})
    assert summary_path.exists()
    memory_path = paths["memory_jsonl"]
    append_brain_memory(
        memory_path,
        {
            "agent": BRAIN_MEMORY_AGENT,
            "task_id": "brain:001",
            "task_type": "run_brain_cycle",
            "cycle_id": "20260628",
            "pair": "ETH-BTC",
            "outcome_known": True,
            "outcome_label": "passed",
            "blocker": "",
            "next_step": "review suggestions",
        },
    )
    lines = memory_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_brain_readiness_score_prefers_ready_candidates():
    frame = pd.DataFrame(
        [
            {"status": "ready", "confidence": 0.8, "pair": "ETH-BTC", "cycle_id": "20260628T0001"},
            {"status": "blocked", "confidence": 0.2, "pair": "ETH-BTC", "cycle_id": "20260628T0001"},
        ]
    )
    score = score_brain_cycle_readiness(frame)
    assert score["candidate_count"] == 2
    assert score["ready_count"] == 1
    assert score["blocked_count"] == 1
    assert score["readiness_score"] == score["ready_count"] / 2 * 0.55 + 0.5 * 0.45
    assert score["readiness_warnings"] == "low_provider_quality;low_mean_confidence"


def test_build_brain_readiness_report_uses_rollup(tmp_path):
    rollup = pd.DataFrame(
        [
            {
                "cycle_id": "20260628T1201Z",
                "pair": "BTC-USD/ETH-USD",
                "status": "ready",
                "confidence": 0.9,
            }
        ]
    )
    rollup_path = tmp_path / "reports" / "brain" / "brain_candidate_rollup.csv"
    rollup_path.parent.mkdir(parents=True, exist_ok=True)
    rollup.to_csv(rollup_path, index=False)

    result = build_brain_readiness_report(
        root=tmp_path,
        candidate_rollup_path=rollup_path,
        score_threshold=0.7,
    )

    assert result.summary["readiness_gate"] == "pass"
    assert int(result.summary["candidate_rows"]) == 1
    assert "brain_readiness_report" in result.paths
    assert result.paths["brain_readiness_report"].exists()
    latest_report = tmp_path / "reports" / "brain" / "brain_readiness_report.csv"
    assert latest_report.exists()
    latest = pd.read_csv(latest_report)
    assert int(latest.iloc[0]["candidate_rows"]) == 1


def _candidate_packet(lane: str, candidate_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "lane": lane,
        "source_type": "wizard_dashboard_primary" if lane == "wizard" else "local_research",
        "source_path": "reports/source.csv",
        "pair": "BTC-USD-ETH-USD",
        "venue": "dydx",
        "detection_timestamp": "2026-07-02T12:00:00Z",
        "timeframe": "1h",
        "setup_identity": f"{candidate_id}|1h|rolling_zscore",
        "setup_rank": 1,
        "setup_role": "primary",
        "strategy_family": "mean_reversion",
        "strategy_mode": "rolling_zscore",
        "normalized_feature_bundle_ref": "data/features/bundle.parquet",
        "regime_snapshot": "neutral",
        "confidence": 0.82,
        "blocker_state": "",
        "backtest_summary_ref": "reports/backtest.csv",
        "forward_walk_summary_ref": "reports/forward_walk.json",
        "paper_outcome_ref": "",
        "provenance": "unit_test",
        "schema_version": CANDIDATE_SETUP_PACKET_VERSION,
    }


def _forward_walk_packet(candidate_id: str, lane: str = "wizard", status: str = "pass", blocker: str = "") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "lane": lane,
        "rolling_split_definition": "train=180d,test=30d,step=30d",
        "oos_sharpe": 1.1,
        "oos_profit_factor": 1.3,
        "oos_max_drawdown": -0.08,
        "oos_trade_count": 23,
        "stability_metrics": "variance=0.12",
        "forward_walk_status": status,
        "blocker_reason": blocker,
        "dataset_provenance": "data/processed/pairs.csv",
        "run_manifest_ref": "reports/rl/base_rl_run_manifest.json",
        "provenance": "unit_test",
        "schema_version": FORWARD_WALK_PACKET_VERSION,
    }


def test_phase1_candidate_schema_validates_wizard_and_native_packets(tmp_path: Path):
    wizard = _candidate_packet("wizard", "cand_wizard")
    native = _candidate_packet("native", "cand_native")

    assert valid_candidate_setup_packet(wizard) == (True, "valid")
    assert valid_candidate_setup_packet(native) == (True, "valid")

    frame = pd.DataFrame([wizard, native])
    assert validate_candidate_setup_frame(frame) == (True, "valid")

    path = brain_output_paths(tmp_path, "20260702")["candidate_setup_csv"]
    written = write_candidate_setup_frame(frame, path)
    assert written.exists()
    saved = pd.read_csv(written)
    assert set(saved["lane"]) == {"wizard", "native"}


def test_phase1_forward_walk_linkage_is_required_for_paper_credibility():
    candidate = _candidate_packet("native", "cand_native")
    candidate["forward_walk_summary_ref"] = ""
    ok, reason = candidate_paper_credibility_status(candidate, local_verification_passed=True)
    assert ok is False
    assert reason == "forward_walk_link_missing"


def test_phase1_paper_outcome_requires_stable_candidate_id_and_lane():
    packet = {
        "candidate_id": "",
        "lane": "",
        "paper_venue": "dydx",
        "submission_timestamp": "2026-07-02T12:00:00Z",
        "entry": "100",
        "exit": "101",
        "hold_duration": "4h",
        "realized_return": 0.01,
        "drawdown": -0.01,
        "slippage_cost_assumptions": "bps=8",
        "result_status": "paper_completed",
        "verification_status": "verified",
        "outcome_evidence_path": "reports/paper.json",
        "provenance": "unit_test",
        "schema_version": PAPER_OUTCOME_PACKET_VERSION,
    }
    assert valid_paper_outcome_packet(packet) == (False, "paper_outcome_candidate_id_missing")


def test_phase1_wizard_evidence_cannot_promote_without_local_verification_and_forward_walk():
    candidate = _candidate_packet("wizard", "cand_wizard")
    forward_walk = _forward_walk_packet("cand_wizard", lane="wizard", status="pass")

    ok, reason = candidate_paper_credibility_status(
        candidate,
        forward_walk_packet=forward_walk,
        local_verification_passed=False,
    )
    assert ok is False
    assert reason == "wizard_local_verification_missing"


def test_phase1_candidate_with_setup_blockers_is_not_paper_credible_even_if_forward_walk_passes():
    candidate = _candidate_packet("wizard", "cand_wizard")
    candidate["blocker_state"] = "weak_correlation;parity_not_replicated"
    forward_walk = _forward_walk_packet("cand_wizard", lane="wizard", status="pass")

    ok, reason = candidate_paper_credibility_status(
        candidate,
        forward_walk_packet=forward_walk,
        local_verification_passed=True,
    )

    assert ok is False
    assert reason == "weak_correlation;parity_not_replicated"


def test_phase1_candidate_paper_credibility_ignores_nan_blocker_state():
    candidate = _candidate_packet("native", "cand_native")
    candidate["blocker_state"] = float("nan")
    forward_walk = _forward_walk_packet("cand_native", lane="native", status="pass")

    ok, reason = candidate_paper_credibility_status(
        candidate,
        forward_walk_packet=forward_walk,
        local_verification_passed=True,
    )

    assert ok is True
    assert reason == "paper_credible"


def test_phase1_overall_summary_consumes_normalized_packets_without_lane_specific_fields():
    frame = pd.DataFrame(
        [
            _candidate_packet("wizard", "cand_wizard"),
            _candidate_packet("native", "cand_native"),
        ]
    )
    summary = build_overall_candidate_summary(frame)
    assert set(summary["lane"]) == {"wizard", "native"}
    assert int(summary.loc[summary["lane"] == "wizard", "candidate_count"].iloc[0]) == 1


def test_phase1_readiness_aliases_ignore_historical_pass_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(active_pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(active_pipeline, "ACTIVE", tmp_path / "reports" / "active")
    monkeypatch.setattr(active_pipeline, "DASHBOARD", tmp_path / "reports" / "dashboard")
    monkeypatch.setattr(active_pipeline, "ML_REPORTS", tmp_path / "reports" / "ml")
    monkeypatch.setattr(active_pipeline, "DATA_ML", tmp_path / "data" / "ml")
    monkeypatch.setattr(active_pipeline, "MODELS", tmp_path / "models" / "trade_gate")

    (tmp_path / "reports" / "brain").mkdir(parents=True)
    (tmp_path / "reports" / "active").mkdir(parents=True)
    (tmp_path / "reports" / "dashboard").mkdir(parents=True)
    (tmp_path / "reports" / "rl").mkdir(parents=True)
    (tmp_path / "reports" / "ml").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "ml").mkdir(parents=True)
    (tmp_path / "models" / "trade_gate").mkdir(parents=True)

    pd.DataFrame([{"score_gate": "pass", "readiness_score": 0.92}]).to_csv(
        tmp_path / "reports" / "brain" / "brain_readiness_report_20260702.csv", index=False
    )
    pd.DataFrame([{"status": "research_only", "blocker": "pair_specific_rl_support_missing", "paper_authorized": False}]).to_csv(
        tmp_path / "reports" / "rl" / "base_rl_paper_handoff_status.csv", index=False
    )
    pd.DataFrame([{"accepted": False}]).to_csv(tmp_path / "reports" / "ml" / "model_gated_acceptance.csv", index=False)
    pd.DataFrame([{"pair": "BTC-USD-ETH-USD", "decision_bucket": "WATCH"}]).to_csv(
        tmp_path / "data" / "processed" / "pair_universe.csv", index=False
    )

    readiness_paths = build_phase1_readiness_surfaces(tmp_path)
    assert Path(readiness_paths["wizard_readiness"]).exists()

    current = active_pipeline.current_state(root=tmp_path)
    state = pd.read_csv(current.paths["current_state"])
    wizard_row = state.loc[state["area"] == "wizard_readiness"].iloc[0]
    assert wizard_row["status"] == "blocked"
    blocker = str(wizard_row["blocker"])
    assert (
        "wizard_pair_detail_capture_missing" in blocker
        or "wizard_current_pairs_missing" in blocker
        or "wizard_timeframe_capture_incomplete" in blocker
    )
    assert {"candidate_id", "setup_identity", "setup_role", "setup_status", "setup_blocker"}.issubset(state.columns)


def test_phase1_wizard_readiness_reflects_live_investigation_blockers(tmp_path):
    (tmp_path / "reports" / "brain").mkdir(parents=True)
    (tmp_path / "reports" / "rl").mkdir(parents=True)

    pd.DataFrame([{"score_gate": "pass", "readiness_score": 0.93}]).to_csv(
        tmp_path / "reports" / "brain" / "brain_readiness_report.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "investigation_area": "live_dashboard_access",
                "status": "signin_redirect",
                "detail": "signin",
                "evidence_path": "reports/brain/wizard_dashboard_live_probe.json",
                "next_action": "sign in",
            },
            {
                "investigation_area": "exact_mode_capture",
                "status": "blocked",
                "detail": "exact_mode_rows=0",
                "evidence_path": "data/processed/wizard_evidence.csv",
                "next_action": "capture pair page",
            },
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_dashboard_investigation.csv", index=False)
    pd.DataFrame(columns=["candidate_id"]).to_csv(tmp_path / "reports" / "brain" / "candidate_setup_packets.csv", index=False)
    pd.DataFrame([{"status": "research_only", "blocker": "pair_specific_rl_support_missing", "paper_authorized": False}]).to_csv(
        tmp_path / "reports" / "rl" / "base_rl_paper_handoff_status.csv", index=False
    )

    readiness_paths = build_phase1_readiness_surfaces(tmp_path)
    wizard = pd.read_csv(readiness_paths["wizard_readiness"]).iloc[0]

    assert wizard["status"] == "blocked"
    assert "wizard_investigation:live_dashboard_access;exact_mode_capture" in str(wizard["blocker"])
    assert {"candidate_id", "setup_identity", "setup_role", "setup_status", "setup_blocker"}.issubset(wizard.index)


def test_current_state_strategy_acceptance_prefers_priority_gate_truth(tmp_path):
    (tmp_path / "reports" / "brain").mkdir(parents=True)
    (tmp_path / "reports" / "rl").mkdir(parents=True)
    (tmp_path / "reports" / "ml").mkdir(parents=True)
    (tmp_path / "reports" / "dashboard").mkdir(parents=True)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "ml").mkdir(parents=True)
    (tmp_path / "models" / "trade_gate").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "gate": "strategy_acceptance",
                "ready": True,
                "status": "ready",
                "evidence": "strategies=19;production_eligible=1",
                "blocker": "",
                "next_action": "allow research-gated paper plans",
            }
        ]
    ).to_csv(tmp_path / "reports" / "priority_readiness.csv", index=False)
    pd.DataFrame(
        [
            {
                "step": "exchange_cost_model_alignment",
                "ready": False,
                "status": "blocked",
                "blocker": "missing_cost_execution_alignment:BTC-USD-BLUR-USD",
            }
        ]
    ).to_csv(tmp_path / "reports" / "strategy_acceptance_checklist.csv", index=False)
    pd.DataFrame([{"score_gate": "pass", "readiness_score": 0.92}]).to_csv(
        tmp_path / "reports" / "brain" / "brain_readiness_report.csv", index=False
    )
    pd.DataFrame([{"status": "research_only", "blocker": "paper_execution_not_ready", "paper_authorized": False}]).to_csv(
        tmp_path / "reports" / "rl" / "base_rl_paper_handoff_status.csv", index=False
    )
    pd.DataFrame([{"accepted": False}]).to_csv(tmp_path / "reports" / "ml" / "model_gated_acceptance.csv", index=False)
    pd.DataFrame([{"pair": "BTC-USD-HYPE-USD", "decision_bucket": "PROMOTE"}]).to_csv(
        tmp_path / "data" / "processed" / "pair_universe.csv", index=False
    )

    build_phase1_readiness_surfaces(tmp_path)
    current = active_pipeline.current_state(root=tmp_path)
    state = pd.read_csv(current.paths["current_state"])
    row = state.loc[state["area"] == "strategy_acceptance"].iloc[0]

    assert bool(row["ready"]) is True
    assert row["status"] == "ready"
    assert pd.isna(row["blocker"]) or str(row["blocker"]).strip() == ""
    assert "checklist_blockers=missing_cost_execution_alignment:BTC-USD-BLUR-USD" in str(row["detail"])
    assert row["next_action"] == "allow research-gated paper plans"


def test_phase1_wizard_readiness_prefers_setup_blocker_over_generic_report_warning(tmp_path):
    (tmp_path / "reports" / "brain").mkdir(parents=True)
    (tmp_path / "reports" / "rl").mkdir(parents=True)
    (tmp_path / "reports" / "active").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "score_gate": "pass",
                "readiness_score": 0.93,
                "readiness_warnings": "low_provider_quality;single_status_profile",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "brain_readiness_report.csv", index=False)
    pd.DataFrame(
        [
            {
                **_candidate_packet("wizard", "cand_wizard"),
                "pair": "EUR-USD-FIL-USD",
                "setup_identity": "EUR-USD|FIL-USD|daily",
                "setup_role": "primary",
                "blocker_state": "missing_exact_mode",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_candidate_packets.csv", index=False)
    pd.DataFrame(
        [
            {
                "investigation_area": "live_dashboard_access",
                "status": "covered",
                "detail": "ok",
                "evidence_path": "reports/brain/wizard_dashboard_live_probe.json",
                "next_action": "monitor",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_dashboard_investigation.csv", index=False)
    pd.DataFrame([{"pair": "EUR-USD/FIL-USD"}]).to_csv(
        tmp_path / "reports" / "active" / "crypto_wizards_live_scanner_capture.csv", index=False
    )
    pd.DataFrame(
        [
            {"pair": "EUR-USD/FIL-USD", "timeframe": timeframe, "capture_status": "captured"}
            for timeframe in ("Daily", "4 Hour", "1 Hour", "5 Min")
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_research_pair_detail_capture.csv", index=False)
    pd.DataFrame([{"status": "research_only", "blocker": "pair_specific_rl_support_missing", "paper_authorized": False}]).to_csv(
        tmp_path / "reports" / "rl" / "base_rl_paper_handoff_status.csv", index=False
    )

    readiness_paths = build_phase1_readiness_surfaces(tmp_path)
    wizard = pd.read_csv(readiness_paths["wizard_readiness"]).iloc[0]

    assert wizard["status"] == "blocked"
    assert wizard["blocker"] == "missing_exact_mode"
    assert wizard["setup_blocker"] == "missing_exact_mode"


def test_phase1_wizard_readiness_prefers_captured_exact_setup_over_stale_missing_exact_mode(tmp_path):
    (tmp_path / "reports" / "brain").mkdir(parents=True)
    (tmp_path / "reports" / "rl").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "score_gate": "pass",
                "readiness_score": 0.93,
                "readiness_warnings": "low_provider_quality;single_status_profile",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "brain_readiness_report.csv", index=False)
    pd.DataFrame(
        [
            {
                **_candidate_packet("wizard", "wizard:ETH-USD-MORPHO-USD::daily"),
                "pair": "ETH-USD-MORPHO-USD",
                "setup_identity": "ETH-USD|MORPHO-USD|daily",
                "setup_role": "primary",
                "setup_rank": 1,
                "confidence": 1.0,
                "blocker_state": "missing_exact_mode",
            },
            {
                **_candidate_packet("wizard", "wizard:DOGE-USD-MORPHO-USD:static_(spread):daily"),
                "pair": "DOGE-USD-MORPHO-USD",
                "setup_identity": "DOGE-USD|MORPHO-USD|daily||static_(spread)",
                "setup_role": "primary",
                "setup_rank": 1,
                "confidence": 1.0,
                "strategy_family": "zscore",
                "strategy_mode": "Static (Spread)",
                "blocker_state": "missing_correlation;missing_ecm",
            },
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_candidate_packets.csv", index=False)
    pd.DataFrame(
        [
            {
                "investigation_area": "live_dashboard_access",
                "status": "covered",
                "detail": "ok",
                "evidence_path": "reports/brain/wizard_dashboard_live_probe.json",
                "next_action": "monitor",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_dashboard_investigation.csv", index=False)
    pd.DataFrame([{"status": "research_only", "blocker": "pair_specific_rl_support_missing", "paper_authorized": False}]).to_csv(
        tmp_path / "reports" / "rl" / "base_rl_paper_handoff_status.csv", index=False
    )

    readiness_paths = build_phase1_readiness_surfaces(tmp_path)
    wizard = pd.read_csv(readiness_paths["wizard_readiness"]).iloc[0]

    assert wizard["candidate_id"] == "wizard:DOGE-USD-MORPHO-USD:static_(spread):daily"
    assert wizard["pair"] == "DOGE-USD-MORPHO-USD"
    assert wizard["setup_identity"] == "DOGE-USD|MORPHO-USD|daily||static_(spread)"
    assert wizard["setup_blocker"] == "missing_correlation;missing_ecm"


def test_phase1_wizard_readiness_prefers_high_quality_pair_detail_over_weak_timeframe_packet(tmp_path):
    (tmp_path / "reports" / "brain").mkdir(parents=True)
    (tmp_path / "reports" / "rl").mkdir(parents=True)
    (tmp_path / "reports" / "active").mkdir(parents=True)

    pd.DataFrame([{"score_gate": "pass", "readiness_score": 0.93}]).to_csv(
        tmp_path / "reports" / "brain" / "brain_readiness_report.csv", index=False
    )
    pd.DataFrame(
        [
            {
                **_candidate_packet("wizard", "wizard:BNB-USD-UNI-USD:ou_(spread):5min"),
                "pair": "BNB-USD-UNI-USD",
                "timeframe": "5min",
                "setup_identity": "BNB-USD|UNI-USD|5min|1000|ou_(spread)",
                "strategy_mode": "OU (Spread)",
                "blocker_state": "sharpe_below_1.75",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_candidate_packets.csv", index=False)
    pd.DataFrame([{"pair": "BNB-USD/UNI-USD"}]).to_csv(
        tmp_path / "reports" / "active" / "crypto_wizards_live_scanner_capture.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "pair": "BNB-USD/UNI-USD",
                "asset_x": "BNB-USD",
                "asset_y": "UNI-USD",
                "venue": "dydx",
                "timeframe": "5 Min",
                "strategy_label": "OU (Spread)",
                "periods_input": 1000,
                "capture_status": "captured",
                "sharpe_top": 1.2,
                "max_drawdown_top": 10.0,
            },
            {
                "pair": "BNB-USD/UNI-USD",
                "asset_x": "BNB-USD",
                "asset_y": "UNI-USD",
                "venue": "dydx",
                "timeframe": "Daily",
                "strategy_label": "OU (Spread)",
                "periods_input": 1000,
                "capture_status": "captured",
                "sharpe_top": 2.13,
                "max_drawdown_top": 9.0,
                "detail_capture_timestamp_utc": "2026-07-09T12:00:00Z",
            },
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_research_pair_detail_capture.csv", index=False)
    pd.DataFrame(
        [
            {
                "investigation_area": "live_dashboard_access",
                "status": "covered",
                "detail": "ok",
                "evidence_path": "reports/brain/wizard_dashboard_live_probe.json",
                "next_action": "monitor",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_dashboard_investigation.csv", index=False)
    pd.DataFrame([{"status": "research_only", "blocker": "pair_specific_rl_support_missing", "paper_authorized": False}]).to_csv(
        tmp_path / "reports" / "rl" / "base_rl_paper_handoff_status.csv", index=False
    )

    readiness_paths = build_phase1_readiness_surfaces(tmp_path)
    wizard = pd.read_csv(readiness_paths["wizard_readiness"]).iloc[0]

    assert wizard["candidate_id"] == "wizard:BNB-USD-UNI-USD:ou_spread:daily"
    assert wizard["pair"] == "BNB-USD-UNI-USD"
    assert "BNB-USD|UNI-USD|daily|1000|ou_spread" == wizard["setup_identity"]
    assert "sharpe_below_1.75" not in str(wizard["blocker"])


def test_phase1_native_readiness_blocks_when_candidate_quality_is_not_supported(tmp_path):
    (tmp_path / "reports" / "brain").mkdir(parents=True)
    (tmp_path / "reports" / "rl").mkdir(parents=True)

    pd.DataFrame([_candidate_packet("native", "cand_native")]).to_csv(
        tmp_path / "reports" / "brain" / "candidate_setup_packets.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "candidate_id": "cand_native",
                "quality_status": "blocked",
                "pair": "BTC-USD-HYPE-USD",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "native_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "cand_native",
                "lane": "native",
                "paper_credible": False,
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "promotion_ladder.csv", index=False)
    pd.DataFrame([{"status": "research_only", "blocker": "pair_specific_rl_support_missing", "paper_authorized": False}]).to_csv(
        tmp_path / "reports" / "rl" / "base_rl_paper_handoff_status.csv", index=False
    )

    readiness_paths = build_phase1_readiness_surfaces(tmp_path)
    native = pd.read_csv(readiness_paths["native_readiness"]).iloc[0]

    assert native["status"] == "candidate_quality_blocked"
    assert native["blocker"] == "native_candidate_quality_not_supported"


def test_phase1_paper_status_uses_sharper_strategy_acceptance_blocker(tmp_path):
    (tmp_path / "reports" / "brain").mkdir(parents=True)
    (tmp_path / "reports" / "rl").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                **_candidate_packet("native", "cand_native"),
                "pair": "BTC-USD-HYPE-USD",
                "setup_identity": "native:BTC-USD-HYPE-USD:native_local_math:native",
                "setup_role": "primary",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "candidate_setup_packets.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "cand_native",
                "lane": "native",
                "pair": "BTC-USD-HYPE-USD",
                "setup_identity": "native:BTC-USD-HYPE-USD:native_local_math:native",
                "setup_role": "primary",
                "paper_credible": True,
                "promotion_stage": "orchestrator_review",
                "blocker": "",
                "paper_status": "research_only",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "promotion_ladder.csv", index=False)
    pd.DataFrame(
        [
            {"status": "research_only", "blocker": "strategy_acceptance_not_ready", "paper_authorized": False}
        ]
    ).to_csv(tmp_path / "reports" / "rl" / "base_rl_paper_handoff_status.csv", index=False)
    pd.DataFrame(
        [
            {"gate": "strategy_acceptance", "blocker": "no_strategy_passes_production_gates"},
        ]
    ).to_csv(tmp_path / "reports" / "priority_readiness.csv", index=False)

    readiness_paths = build_phase1_readiness_surfaces(tmp_path)
    paper = pd.read_csv(readiness_paths["paper_status"]).iloc[0]

    assert paper["blocker"] == "no_strategy_passes_production_gates"
    assert paper["pair"] == "BTC-USD-HYPE-USD"
    assert paper["setup_status"] == "orchestrator_review"


def test_phase1_paper_status_row_ignores_placeholder_blocker(tmp_path):
    (tmp_path / "reports" / "brain").mkdir(parents=True)
    (tmp_path / "reports" / "rl").mkdir(parents=True)

    pd.DataFrame([{"status": "research_only", "blocker": float("nan"), "paper_authorized": False, "paper_execution_ready": False}]).to_csv(
        tmp_path / "reports" / "rl" / "base_rl_paper_handoff_status.csv", index=False
    )

    paper = _paper_status_row(
        tmp_path,
        tmp_path / "reports" / "rl" / "base_rl_paper_handoff_status.csv",
    )

    assert paper["status"] == "research_only"
    assert paper["blocker"] == "paper_execution_not_ready"


def test_phase1_overall_readiness_row_coalesces_placeholder_blockers(tmp_path):
    wizard_row = {
        "status": "ready",
        "ready": True,
        "blocker": "nan",
        "summary": "wizard=ready",
    }
    native_row = {
        "status": "candidate_quality_blocked",
        "ready": False,
        "blocker": "native_candidate_quality_not_supported",
        "summary": "native=blocked",
    }
    paper_row = {
        "status": "research_only",
        "ready": False,
        "blocker": "native_candidate_quality_not_supported;nan",
        "summary": "paper=blocked",
        "candidate_id": "cand_native",
    }

    row = _overall_readiness_row(tmp_path, wizard_row, native_row, paper_row)

    assert row["ready"] is False
    assert row["status"] == "blocked"
    assert row["blocker"] == "native_candidate_quality_not_supported"
    assert row["setup_blocker"] == "native_candidate_quality_not_supported"


def test_internal_blocker_text_helpers_clean_and_combine_blockers():
    assert _clean_text(float("nan")) == ""
    assert _coerce_blockers(["", "nan", "a", "nan", "A", "a", "None"]) == "a;A"


def test_phase1_paper_status_marks_paper_authorized_as_ready(tmp_path):
    (tmp_path / "reports" / "brain").mkdir(parents=True)
    (tmp_path / "reports" / "rl").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                **_candidate_packet("native", "cand_native"),
                "pair": "BTC-USD-LDO-USD",
                "setup_identity": "native:BTC-USD-LDO-USD:native_local_math:native",
                "setup_role": "primary",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "candidate_setup_packets.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "cand_native",
                "lane": "native",
                "pair": "BTC-USD-LDO-USD",
                "setup_identity": "native:BTC-USD-LDO-USD:native_local_math:native",
                "setup_role": "primary",
                "paper_credible": True,
                "promotion_stage": "orchestrator_review",
                "blocker": "",
                "paper_status": "paper_authorized",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "promotion_ladder.csv", index=False)
    pd.DataFrame(
        [
            {"status": "paper_authorized", "blocker": "", "paper_authorized": True}
        ]
    ).to_csv(tmp_path / "reports" / "rl" / "base_rl_paper_handoff_status.csv", index=False)

    readiness_paths = build_phase1_readiness_surfaces(tmp_path)
    paper = pd.read_csv(readiness_paths["paper_status"]).iloc[0]

    assert paper["status"] == "paper_authorized"
    assert bool(paper["ready"]) is True


def test_phase1_schema_version_mismatch_fails_candidate_packet():
    packet = _candidate_packet("native", "cand_native")
    packet["schema_version"] = PHASE1_SPINE_SCHEMA_VERSION + "_wrong"
    assert valid_candidate_setup_packet(packet) == (False, "candidate_schema_version_mismatch")


def test_phase1_append_forward_walk_packets_validates_schema(tmp_path: Path):
    path = brain_output_paths(tmp_path, "20260702")["forward_walk_jsonl"]
    append_forward_walk_packets(path, [_forward_walk_packet("cand_wizard")])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
