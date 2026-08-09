import json
from pathlib import Path

import pandas as pd

import quant_platform.active_pipeline as active_pipeline
from quant_platform.three_brain_system import (
    build_candidate_repair_loop,
    build_shared_outcome_memory,
    build_native_outcome_feature_memory,
    build_native_promotion_independence_report,
    build_native_focus_repair_report,
    build_overall_brain,
    build_promotion_readiness,
    build_shadow_rl_candidate_set,
    build_shadow_rl_intake_report,
    build_shadow_rl_lanes,
    build_shadow_rl_usefulness_report,
    build_three_brain_system,
    build_wizard_hourly_priority_capture_queue,
    build_wizard_candidate_repair_report,
    build_wizard_dashboard_investigation,
    _normalize_wizard_blocker_chain,
    _native_forward_walk_diagnosis,
    _wizard_repair_loop_action,
    _shadow_evidence_quality_status,
    _wizard_exact_mode_capture_status,
    build_wizard_dashboard_ontology,
    build_wizard_specialist_lane,
    build_native_specialist_lane,
)
from quant_platform.rl.brain_contract import build_phase1_readiness_surfaces
from quant_platform.wizard_evidence import build_wizard_evidence, build_wizard_strategy_alignment_report


def _write_minimal_inputs(tmp_path: Path) -> None:
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "reports" / "active").mkdir(parents=True)
    (tmp_path / "reports" / "brain").mkdir(parents=True)
    (tmp_path / "reports" / "rl").mkdir(parents=True)
    (tmp_path / "reports" / "dashboard").mkdir(parents=True)
    (tmp_path / "reports" / "ml").mkdir(parents=True)
    (tmp_path / "reports" / "active").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "active" / "wizard_hourly_database").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "asset_x": "BTC-USD",
                "asset_y": "ETH-USD",
                "exchange": "dydx",
                "interval": "1h",
                "dashboard_recommended_strategy": "z-score",
                "exact_mode": "Static (ZScoreR)",
                "local_strategy_family": "zscore",
                "sharpe": 2.1,
                "returns_total": 0.21,
                "hurst": 0.42,
                "half_life": 12,
                "source_path": "reports/source.csv",
                "evidence_path": "reports/wizard.csv",
            }
        ]
    ).to_csv(tmp_path / "data" / "processed" / "wizard_evidence.csv", index=False)

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "hypothesis_status": "ready_for_local_replay",
                "hypothesis_reason": "",
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_hypotheses.csv", index=False)
    pd.DataFrame(
        [
            {
                "priority_rank": 1,
                "pair": "BTC-USD-ETH-USD",
                "asset_x": "BTC-USD",
                "asset_y": "ETH-USD",
                "interval": "1h",
                "sharpe": 2.1,
                "returns_total": 0.21,
                "returns_total_pct": 21.0,
                "mode_blocker": "missing_exact_mode",
                "pair_page_url": "https://cryptowizards.net/wizards/zscore/pair/1?origin=scanner",
                "required_capture_fields": "selected_strategy_value;spread_id;strategy_id;exact_mode;period;interval;backtest_settings",
                "operator_action": "open_pair_page_and_capture_exact_mode",
                "source_path": "reports/source.csv",
                "evidence_path": "reports/source.csv",
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_exact_mode_capture_queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "interval": "1h",
                "dashboard_recommended_strategy": "",
                "exact_mode": "",
                "spread_id": "",
                "strategy_id": "",
                "local_strategy_id": "",
                "local_strategy_name": "",
                "local_strategy_family": "",
                "strategy_mapping_status": "unmapped",
                "source_first_action": "capture_dashboard_strategy_first",
                "evidence_path": "reports/source.csv",
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_strategy_alignment_report.csv", index=False)

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "diagnostic_blocker": "",
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_diagnostic_confirmation.csv", index=False)

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "parity_status": "match",
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_vs_local_parity_report.csv", index=False)

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "verification_status": "verified",
                "acceptance": "ACCEPT",
                "acceptance_reason": "",
                "local_sharpe": 1.4,
                "local_profit_factor": 1.3,
                "local_max_drawdown": -0.08,
                "local_closed_trades": 9,
                "summary_path": "reports/active/btc_eth_after_cost.csv",
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_local_verification_batch.csv", index=False)
    pd.DataFrame(
        [
            {"exit_time": "2026-07-01T00:00:00Z", "exit_reason": "close", "profit_after_cost": 0.02},
            {"exit_time": "2026-07-01T01:00:00Z", "exit_reason": "close", "profit_after_cost": 0.01},
            {"exit_time": "2026-07-01T02:00:00Z", "exit_reason": "close", "profit_after_cost": -0.005},
            {"exit_time": "2026-07-01T03:00:00Z", "exit_reason": "close", "profit_after_cost": 0.015},
            {"exit_time": "2026-07-01T04:00:00Z", "exit_reason": "close", "profit_after_cost": 0.01},
            {"exit_time": "2026-07-01T05:00:00Z", "exit_reason": "close", "profit_after_cost": -0.002},
        ]
    ).to_csv(tmp_path / "reports" / "active" / "btc_eth_trade_log.csv", index=False)

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "decision_bucket": "PROMOTE",
                "best_execution_venue": "dydx",
                "available_timeframes": "1h",
                "best_wizard_local_strategy_family": "zscore",
                "best_wizard_local_strategy_name": "Classic ZScore Mean Reversion",
                "acceptance_score": 82,
                "combined_score": 88,
                "local_backtest_score": 1.4,
                "walk_forward_score": 0.7,
                "decision_reason": "native_support",
                "evidence_path": "data/processed/pair_universe.csv",
            }
        ]
    ).to_csv(tmp_path / "data" / "processed" / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "selected_path": str(tmp_path / "data" / "raw" / "pair_details" / "pair_btc_eth_5mins_dydx_long_history_derived_history.json"),
                "status": "copied",
                "detail": "quality_report_source",
                "history_rows": 12,
                "execution_usable": True,
            }
        ]
    ).to_csv(tmp_path / "reports" / "p2_rerun_subset_manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair_id": "",
                "asset_x": "BTC-USD",
                "asset_y": "ETH-USD",
                "pair_history": str(tmp_path / "data" / "raw" / "pair_details" / "pair_btc_eth_5mins_dydx_long_history_derived_history.json"),
                "status": "built",
            }
        ]
    ).to_csv(tmp_path / "reports" / "dydx_local_pair_universe_run.csv", index=False)
    (tmp_path / "data" / "raw" / "pair_details").mkdir(parents=True, exist_ok=True)
    pair_history = {
        "history": [
            {"timestamp": "2026-07-01T00:00:00Z", "rolling_zscore": 2.2, "spread_return_1": 0.01},
            {"timestamp": "2026-07-01T00:05:00Z", "rolling_zscore": 0.5, "spread_return_1": 0.0},
            {"timestamp": "2026-07-01T00:10:00Z", "rolling_zscore": -2.1, "spread_return_1": 0.012},
            {"timestamp": "2026-07-01T00:15:00Z", "rolling_zscore": 0.1, "spread_return_1": 0.0},
            {"timestamp": "2026-07-01T00:20:00Z", "rolling_zscore": 2.4, "spread_return_1": 0.008},
            {"timestamp": "2026-07-01T00:25:00Z", "rolling_zscore": 0.0, "spread_return_1": 0.0},
            {"timestamp": "2026-07-01T00:30:00Z", "rolling_zscore": -2.5, "spread_return_1": 0.013},
            {"timestamp": "2026-07-01T00:35:00Z", "rolling_zscore": 0.3, "spread_return_1": 0.0},
            {"timestamp": "2026-07-01T00:40:00Z", "rolling_zscore": 2.3, "spread_return_1": 0.007},
        ]
    }
    (tmp_path / "data" / "raw" / "pair_details" / "pair_btc_eth_5mins_dydx_long_history_derived_history.json").write_text(
        __import__("json").dumps(pair_history),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-07-02T12:00:00Z",
                "pair": "BTC-USD-ETH-USD",
                "plan_status": "paper_ready",
                "intents_json": '{"venue":"dydx"}',
            }
        ]
    ).to_csv(tmp_path / "reports" / "paper_trading_journal.csv", index=False)

    pd.DataFrame(
        [
            {
                "status": "research_only",
                "blocker": "pair_specific_rl_support_missing",
                "paper_authorized": False,
            }
        ]
    ).to_csv(tmp_path / "reports" / "rl" / "base_rl_paper_handoff_status.csv", index=False)

    pd.DataFrame([{"accepted": False}]).to_csv(tmp_path / "reports" / "ml" / "model_gated_acceptance.csv", index=False)
    pd.DataFrame([{"check": "artifact:pair_universe.csv", "ready": True, "blocker": "", "evidence_path": "x", "next_action": "x"}]).to_csv(
        tmp_path / "reports" / "active" / "system_check.csv", index=False
    )
    pd.DataFrame().to_csv(tmp_path / "reports" / "brain" / "brain_readiness_report.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "reports" / "brain" / "paper_readiness_trend.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "reports" / "rl" / "base_rl_training_report.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "reports" / "rl" / "base_rl_evaluation_report.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "reports" / "rl" / "base_rl_pair_coverage.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "reports" / "rl" / "base_rl_blocked_actions.csv", index=False)
    pd.DataFrame(
        [
            {
                "path": str(tmp_path / "data" / "raw" / "pair_details" / "pair_btc_eth_capture.json"),
                "pair": "BTC-USD-ETH-USD",
                "history_rows": 100,
                "capture_completeness_score": 92.0,
                "baseline_ready": True,
                "ecm_ready": True,
                "two_leg_ready": True,
                "import_ready": True,
                "research_spine_ready": True,
                "next_capture_focus": "ready_for_research_spine",
            }
        ]
    ).to_csv(tmp_path / "reports" / "pair_detail_capture_checklist.csv", index=False)
    pd.DataFrame(
        [
            {"field": "spread", "type": "float", "example": "0.1", "source": "pair_detail", "pair": "BTC-USD-ETH-USD"},
            {"field": "zscore", "type": "float", "example": "2.1", "source": "pair_detail", "pair": "BTC-USD-ETH-USD"},
            {"field": "ecm_strength", "type": "float", "example": "0.8", "source": "pair_detail", "pair": "BTC-USD-ETH-USD"},
        ]
    ).to_csv(tmp_path / "reports" / "pair_detail_field_dictionary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scan_timestamp": "2026-07-03T23:00:00+00:00",
                "pair": "BTC-USD-ETH-USD",
                "periods": 320,
                "timeframe": "1h",
                "strategy": "Static (ZScoreR)",
                "annualized_return_pct": 12.0,
                "sharpe": 1.2,
                "closed_trades": 5,
                "review_status": "changed",
                "highest_anomaly_severity": "",
                "candidate_status": "review_now",
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_hourly_database" / "wizard_hourly_candidate_queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "actionable_variant_count": 1,
                "blocked_variant_count": 0,
                "top_strategy": "Static (ZScoreR)",
                "top_timeframe": "1h",
                "top_annualized_return_pct": 12.0,
                "top_sharpe": 1.2,
                "top_closed_trades": 5,
                "pair_status": "review_now",
                "hourly_rank": 1,
                "recommended_this_hour": True,
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_hourly_database" / "wizard_hourly_pair_queue.csv", index=False)
    (tmp_path / "reports" / "brain" / "wizard_dashboard_live_probe.json").write_text(
        __import__("json").dumps(
            {
                "status": "signin_redirect",
                "detail": "scanner route redirected to sign-in during live Chrome probe",
                "evidence_path": "reports/brain/wizard_dashboard_live_probe.json",
                "next_action": "sign in to Crypto Wizards in Chrome and rerun the scanner inspection",
            }
        ),
        encoding="utf-8",
    )


def test_three_brain_build_writes_evidence_chain_outputs(tmp_path):
    _write_minimal_inputs(tmp_path)

    wizard_ontology = build_wizard_dashboard_ontology(root=tmp_path)
    wizard = build_wizard_specialist_lane(root=tmp_path)
    native = build_native_specialist_lane(root=tmp_path)
    shadow = build_shadow_rl_lanes(root=tmp_path)
    overall = build_overall_brain(root=tmp_path)
    promotion = build_promotion_readiness(root=tmp_path)
    build_three_brain_system(root=tmp_path)

    assert wizard_ontology.paths["wizard_dashboard_ontology"].exists()
    assert wizard.paths["wizard_candidate_packets"].exists()
    assert wizard.paths["wizard_forward_walk_folds"].exists()
    assert native.paths["native_candidate_packets"].exists()
    assert native.paths["native_forward_walk_folds"].exists()
    assert shadow.paths["lane_shadow_rl"].exists()
    assert overall.paths["overall_brain_summary"].exists()
    assert (tmp_path / "reports" / "brain" / "shadow_rl_intake_report.csv").exists()
    assert promotion.paths["promotion_ladder"].exists()
    assert (tmp_path / "reports" / "brain" / "wizard_dashboard_investigation.csv").exists()
    assert (tmp_path / "reports" / "brain" / "forward_walk_strength_report.csv").exists()
    assert (tmp_path / "reports" / "brain" / "native_quality_report.csv").exists()
    assert (tmp_path / "reports" / "brain" / "native_candidate_diagnosis.csv").exists()
    assert (tmp_path / "reports" / "brain" / "wizard_candidate_repair_report.csv").exists()
    assert (tmp_path / "reports" / "active" / "wizard_hourly_priority_capture_queue.csv").exists()
    assert (tmp_path / "reports" / "brain" / "native_focus_repair_report.csv").exists()
    assert (tmp_path / "reports" / "brain" / "wizard_hourly_repair_targets.csv").exists()
    assert (tmp_path / "reports" / "brain" / "candidate_repair_loop.csv").exists()
    assert (tmp_path / "reports" / "brain" / "shadow_rl_usefulness_report.csv").exists()
    assert (tmp_path / "reports" / "brain" / "shadow_rl_candidate_set.csv").exists()
    assert (tmp_path / "reports" / "brain" / "overall_arbitration_scorecard.csv").exists()
    assert (tmp_path / "reports" / "brain" / "blocker_persistence_report.csv").exists()
    assert (tmp_path / "reports" / "brain" / "shared_outcome_memory.csv").exists()
    assert (tmp_path / "reports" / "brain" / "native_outcome_feature_memory.csv").exists()
    assert (tmp_path / "reports" / "brain" / "native_promotion_independence_report.csv").exists()

    wizard_packets = pd.read_csv(wizard.paths["wizard_candidate_packets"])
    native_packets = pd.read_csv(native.paths["native_candidate_packets"])
    promotion_ladder = pd.read_csv(promotion.paths["promotion_ladder"])
    fw_strength = pd.read_csv(tmp_path / "reports" / "brain" / "forward_walk_strength_report.csv")
    native_quality = pd.read_csv(tmp_path / "reports" / "brain" / "native_quality_report.csv")
    native_diagnosis = pd.read_csv(tmp_path / "reports" / "brain" / "native_candidate_diagnosis.csv")
    wizard_repair = pd.read_csv(tmp_path / "reports" / "brain" / "wizard_candidate_repair_report.csv")
    repair_loop = pd.read_csv(tmp_path / "reports" / "brain" / "candidate_repair_loop.csv")
    shadow_usefulness = pd.read_csv(tmp_path / "reports" / "brain" / "shadow_rl_usefulness_report.csv")
    arbitration = pd.read_csv(tmp_path / "reports" / "brain" / "overall_arbitration_scorecard.csv")
    blocker = pd.read_csv(tmp_path / "reports" / "brain" / "blocker_persistence_report.csv")

    assert set(wizard_packets["lane"]) == {"wizard"}
    assert set(native_packets["lane"]) == {"native"}
    assert "forward_walk_required" in promotion_ladder.columns
    assert len(pd.read_csv(wizard.paths["wizard_forward_walk_folds"])) >= 1
    assert len(pd.read_csv(native.paths["native_forward_walk_folds"])) >= 1
    assert {"strength_tier", "evidence_density", "threshold_status"}.issubset(fw_strength.columns)
    assert {"native_origin_type", "quality_status", "forward_walk_status"}.issubset(native_quality.columns)
    assert {"current_blocker", "next_repair_action", "evaluation_mode"}.issubset(native_diagnosis.columns)
    assert {"exact_mode_capture_status", "current_blocker", "next_repair_action"}.issubset(wizard_repair.columns)
    assert {"repair_priority_rank", "repair_action", "rerun_required", "hourly_review_status", "hourly_candidate_status", "hourly_pair_recommended"}.issubset(repair_loop.columns)
    assert {"usefulness_status", "paper_credible_overlap"}.issubset(shadow_usefulness.columns)
    assert {"pair_status", "disagreement_flag", "promotion_ready_count"}.issubset(arbitration.columns)
    assert {"blocker_persistence", "verified_outcome", "paper_outcome_status"}.issubset(blocker.columns)


def test_shared_outcome_memory_merges_lane_metadata(tmp_path):
    _write_minimal_inputs(tmp_path)
    wizard = build_wizard_specialist_lane(root=tmp_path)
    native = build_native_specialist_lane(root=tmp_path)

    wizard_outcomes = pd.read_csv(wizard.paths["wizard_paper_outcomes"])
    native_outcomes = pd.read_csv(native.paths["native_paper_outcomes"])
    wizard_outcomes.loc[0, ["realized_return", "drawdown", "result_status", "verification_status"]] = [0.08, -0.03, "paper_completed", "verified"]
    native_outcomes.loc[0, ["realized_return", "drawdown", "result_status", "verification_status"]] = [-0.02, -0.05, "paper_completed", "unverified"]
    wizard_outcomes.to_csv(wizard.paths["wizard_paper_outcomes"], index=False)
    native_outcomes.to_csv(native.paths["native_paper_outcomes"], index=False)

    result = build_shared_outcome_memory(root=tmp_path)
    frame = pd.read_csv(result.paths["shared_outcome_memory"])

    assert len(frame) == 2
    assert set(frame["source_lane"]) == {"wizard", "native"}
    assert set(frame["promotion_authority"]) == {"wizard_learning_only", "native_learning_only"}
    assert set(frame["outcome_label"]) == {"win", "pending"}
    assert "regime_bucket" in frame.columns


def test_shared_outcome_memory_skips_unconfirmed_broadcast_rows(tmp_path):
    _write_minimal_inputs(tmp_path)
    wizard = build_wizard_specialist_lane(root=tmp_path)
    native = build_native_specialist_lane(root=tmp_path)

    wizard_outcomes = pd.read_csv(wizard.paths["wizard_paper_outcomes"])
    native_outcomes = pd.read_csv(native.paths["native_paper_outcomes"])
    wizard_outcomes.loc[0, ["realized_return", "drawdown", "result_status", "verification_status"]] = [0.0, 0.0, "broadcast_accepted_unconfirmed", ""]
    native_outcomes.loc[0, ["realized_return", "drawdown", "result_status", "verification_status"]] = [0.05, -0.02, "paper_completed", "verified"]
    wizard_outcomes.to_csv(wizard.paths["wizard_paper_outcomes"], index=False)
    native_outcomes.to_csv(native.paths["native_paper_outcomes"], index=False)

    result = build_shared_outcome_memory(root=tmp_path)
    frame = pd.read_csv(result.paths["shared_outcome_memory"])

    assert len(frame) == 1
    assert set(frame["source_lane"]) == {"native"}


def test_wizard_lane_seeds_outcome_memory_from_verified_local_verification(tmp_path):
    _write_minimal_inputs(tmp_path)
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-07-02T12:00:00Z",
                "pair": "UNRELATED-PAIR",
                "plan_status": "paper_ready",
                "intents_json": '{"venue":"dydx"}',
            }
        ]
    ).to_csv(tmp_path / "reports" / "paper_trading_journal.csv", index=False)
    result = build_wizard_specialist_lane(root=tmp_path)
    outcomes = pd.read_csv(result.paths["wizard_paper_outcomes"])

    assert len(outcomes) == 1
    row = outcomes.iloc[0]
    assert row["result_status"] == "audit_only"
    assert row["verification_status"] == "verified"
    assert row["hold_duration"] != ""
    assert row["provenance"] == "wizard_local_verification_audit"


def test_wizard_lane_keeps_closed_journal_without_exit_price_unverified(tmp_path):
    _write_minimal_inputs(tmp_path)
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-07-02T12:00:00Z",
                "pair": "BTC-USD-ETH-USD",
                "plan_status": "paper_completed",
                "intents_json": "{}",
                "realized_return": "0.04",
                "exit_snapshot_json": "{}",
            }
        ]
    ).to_csv(tmp_path / "reports" / "paper_trading_journal.csv", index=False)

    result = build_wizard_specialist_lane(root=tmp_path)
    outcomes = pd.read_csv(result.paths["wizard_paper_outcomes"])

    assert len(outcomes) == 1
    row = outcomes.iloc[0]
    assert row["result_status"] == "paper_completed"
    assert row["verification_status"] == "unverified"


def test_wizard_lane_verifies_closed_journal_with_exit_price(tmp_path):
    _write_minimal_inputs(tmp_path)
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-07-02T12:00:00Z",
                "pair": "BTC-USD-ETH-USD",
                "plan_status": "paper_completed",
                "intents_json": "{}",
                "realized_return": "0.04",
                "exit_snapshot_json": json.dumps({"venue_exit_price_x": 101.0}),
            }
        ]
    ).to_csv(tmp_path / "reports" / "paper_trading_journal.csv", index=False)

    result = build_wizard_specialist_lane(root=tmp_path)
    outcomes = pd.read_csv(result.paths["wizard_paper_outcomes"])

    assert len(outcomes) == 1
    row = outcomes.iloc[0]
    assert row["result_status"] == "paper_completed"
    assert row["verification_status"] == "verified"


def test_native_outcome_features_and_independence_stay_learning_only(tmp_path):
    _write_minimal_inputs(tmp_path)
    wizard = build_wizard_specialist_lane(root=tmp_path)
    build_native_specialist_lane(root=tmp_path)
    build_overall_brain(root=tmp_path)
    build_promotion_readiness(root=tmp_path)

    wizard_outcomes = pd.read_csv(wizard.paths["wizard_paper_outcomes"])
    wizard_outcomes.loc[0, ["realized_return", "drawdown", "result_status", "verification_status"]] = [0.06, -0.02, "paper_completed", "verified"]
    wizard_outcomes.to_csv(wizard.paths["wizard_paper_outcomes"], index=False)

    build_shared_outcome_memory(root=tmp_path)
    feature_result = build_native_outcome_feature_memory(root=tmp_path)
    independence_result = build_native_promotion_independence_report(root=tmp_path)

    features = pd.read_csv(feature_result.paths["native_outcome_feature_memory"])
    independence = pd.read_csv(independence_result.paths["native_promotion_independence_report"])

    assert len(features) == 1
    assert features.iloc[0]["native_promotion_basis"] == "native_evidence_only"
    assert int(features.iloc[0]["wizard_history_feature_count"]) >= 1
    assert features.iloc[0]["wizard_learning_feature_state"] == "available"
    assert len(independence) == 1
    assert independence.iloc[0]["independence_status"] == "independent"
    assert independence.iloc[0]["native_promotion_basis"] == "native_evidence_only"


def test_three_brain_overall_summary_tracks_agreement_status(tmp_path):
    _write_minimal_inputs(tmp_path)
    build_wizard_specialist_lane(root=tmp_path)
    build_native_specialist_lane(root=tmp_path)

    result = build_overall_brain(root=tmp_path)
    frame = pd.read_csv(result.paths["overall_brain_summary"])

    assert len(frame) == 1
    assert frame.iloc[0]["agreement_status"] == "both_support"
    assert frame.iloc[0]["arbitration_decision"] == "agreement_strongest"


def test_overall_brain_prefers_primary_wizard_setup_for_same_pair(tmp_path):
    (tmp_path / "reports" / "brain").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "candidate_id": "wizard:btc-usd-eth-usd:copula:1h",
                "lane": "wizard",
                "pair": "BTC-USD-ETH-USD",
                "setup_identity": "BTC-USD|ETH-USD|1h|320|copula",
                "setup_rank": 2,
                "setup_role": "alternate",
                "blocker_state": "",
            },
            {
                "candidate_id": "wizard:btc-usd-eth-usd:static_(zscorer):1h",
                "lane": "wizard",
                "pair": "BTC-USD-ETH-USD",
                "setup_identity": "BTC-USD|ETH-USD|1h|320|static_(zscorer)",
                "setup_rank": 1,
                "setup_role": "primary",
                "blocker_state": "",
            },
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_candidate_packets.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "wizard:btc-usd-eth-usd:copula:1h",
                "forward_walk_status": "pass",
                "blocker_reason": "",
            },
            {
                "candidate_id": "wizard:btc-usd-eth-usd:static_(zscorer):1h",
                "forward_walk_status": "pass",
                "blocker_reason": "",
            },
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_forward_walk.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "native:btc-usd-eth-usd:zscore:native",
                "lane": "native",
                "pair": "BTC-USD-ETH-USD",
                "setup_identity": "native:btc-usd-eth-usd:zscore:native",
                "setup_rank": 1,
                "setup_role": "primary",
                "blocker_state": "",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "native_candidate_packets.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "native:btc-usd-eth-usd:zscore:native",
                "forward_walk_status": "pass",
                "blocker_reason": "",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "native_forward_walk.csv", index=False)

    result = build_overall_brain(root=tmp_path)
    frame = pd.read_csv(result.paths["overall_brain_summary"])

    assert len(frame) == 1
    assert frame.iloc[0]["wizard_candidate_id"] == "wizard:btc-usd-eth-usd:static_(zscorer):1h"
    assert frame.iloc[0]["wizard_setup_identity"] == "BTC-USD|ETH-USD|1h|320|static_(zscorer)"


def test_native_lane_prefers_local_discovery_manifest_fields(tmp_path):
    _write_minimal_inputs(tmp_path)
    result = build_native_specialist_lane(root=tmp_path)
    frame = pd.read_csv(result.paths["native_candidate_packets"])
    assert frame.iloc[0]["normalized_feature_bundle_ref"].endswith("data/processed/pair_universe.csv")
    assert frame.iloc[0]["setup_role"] == "primary"
    assert int(frame.iloc[0]["setup_rank"]) == 1
    forward = pd.read_csv(result.paths["native_forward_walk"])
    assert "history_time_folds" in str(forward.iloc[0]["rolling_split_definition"])


def test_wizard_lane_carries_primary_setup_identity(tmp_path):
    _write_minimal_inputs(tmp_path)
    result = build_wizard_specialist_lane(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_candidate_packets"])

    assert {"setup_identity", "setup_rank", "setup_role"}.issubset(frame.columns)
    assert frame.iloc[0]["setup_role"] in {"primary", "alternate"}
    assert int(frame.iloc[0]["setup_rank"]) >= 1
    assert str(frame.iloc[0]["setup_identity"]).strip() != ""


def test_wizard_lane_enriches_packet_with_live_scanner_and_second_page_context(tmp_path):
    _write_minimal_inputs(tmp_path)
    live_dir = tmp_path / "data" / "collected" / "wizard_scanner_static_spread_visible_live"
    second_dir = tmp_path / "data" / "collected" / "wizard_second_page_static_spread_green"
    live_dir.mkdir(parents=True, exist_ok=True)
    second_dir.mkdir(parents=True, exist_ok=True)

    (live_dir / "wizard_scanner_static_spread_visible_live_latest.json").write_text(
        __import__("json").dumps(
            {
                "captured_at": "2026-07-04T19:04:00Z",
                "rows": [
                    {
                        "row_index": 1,
                        "pair": "BTC-USD / ETH-USD",
                        "updated": "2026-Jul-04 10:17:38 UTC",
                        "normal_zscore": "-1.58",
                        "rolling_zscore": "0.51",
                        "dependency": "16.1%x|y56.6%y|x",
                        "stationarity": "89.3%corr | JnEG | 0.80hurst | 15.2half life | 280σ32σ",
                        "risk": "102.3%102.5% | -1.8%VaR",
                        "reward": "21.6%return | 2.03sharpe",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (second_dir / "wizard_second_page_static_spread_green_latest.json").write_text(
        __import__("json").dumps(
            [
                {
                    "scanner_rank": 1,
                    "pair_x": "BTC-USD",
                    "pair_y": "ETH-USD",
                    "marker": "BTC-USD (asset X)\nETH-USD (asset Y)",
                    "backtestRows": [
                        {"strategy": "StaticZScoreR", "returns": "31.0%", "sharpe": "2.1"},
                        {"strategy": "Copula", "returns": "12.8%", "sharpe": "1.5"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = build_wizard_specialist_lane(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_candidate_packets"])
    row = frame.iloc[0]
    regime = __import__("json").loads(row["regime_snapshot"])

    assert "wizard_scanner_static_spread_visible_live_latest.json" in row["normalized_feature_bundle_ref"]
    assert "wizard_second_page_static_spread_green_latest.json" in row["normalized_feature_bundle_ref"]
    assert regime["wizard_live_scanner"]["normal_zscore"] == "-1.58"
    assert any(item["strategy"] == "StaticZScoreR" for item in regime["wizard_strategy_matrix"]["strategy_rows"])
    assert "wizard_second_page_capture" in row["provenance"]


def test_promotion_ladder_preserves_distinct_wizard_setup_identity_rows(tmp_path):
    _write_minimal_inputs(tmp_path)
    pd.DataFrame(
        [
            {
                "candidate_id": "wizard:btc-usd-eth-usd:static_(zscorer):1h",
                "lane": "wizard",
                "source_type": "wizard_dashboard_primary",
                "source_path": "reports/source.csv",
                "pair": "BTC-USD-ETH-USD",
                "venue": "dydx",
                "detection_timestamp": "2026-07-02T12:00:00Z",
                "timeframe": "1h",
                "setup_identity": "BTC-USD|ETH-USD|1h|320|static_(zscorer)",
                "setup_rank": 1,
                "setup_role": "primary",
                "strategy_family": "zscore",
                "strategy_mode": "Static (ZScoreR)",
                "normalized_feature_bundle_ref": "reports/wizard.csv",
                "regime_snapshot": "mean_reversion_favorable",
                "confidence": 0.85,
                "blocker_state": "",
                "backtest_summary_ref": "reports/active/btc_eth_after_cost.csv",
                "forward_walk_summary_ref": "reports/brain/wizard_forward_walk.csv#wizard:btc-usd-eth-usd:static_(zscorer):1h",
                "paper_outcome_ref": "",
                "provenance": "unit_test",
                "schema_version": "phase1_spine_v1",
            },
            {
                "candidate_id": "wizard:btc-usd-eth-usd:copula:1h",
                "lane": "wizard",
                "source_type": "wizard_dashboard_alternate",
                "source_path": "reports/source.csv",
                "pair": "BTC-USD-ETH-USD",
                "venue": "dydx",
                "detection_timestamp": "2026-07-02T12:00:00Z",
                "timeframe": "1h",
                "setup_identity": "BTC-USD|ETH-USD|1h|320|copula",
                "setup_rank": 2,
                "setup_role": "alternate",
                "strategy_family": "copula",
                "strategy_mode": "Copula",
                "normalized_feature_bundle_ref": "reports/wizard.csv",
                "regime_snapshot": "mean_reversion_favorable",
                "confidence": 0.72,
                "blocker_state": "",
                "backtest_summary_ref": "reports/active/btc_eth_after_cost.csv",
                "forward_walk_summary_ref": "reports/brain/wizard_forward_walk.csv#wizard:btc-usd-eth-usd:copula:1h",
                "paper_outcome_ref": "",
                "provenance": "unit_test",
                "schema_version": "phase1_spine_v1",
            },
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_candidate_packets.csv", index=False)
    pd.DataFrame(
        [
            {"candidate_id": "wizard:btc-usd-eth-usd:static_(zscorer):1h", "lane": "wizard", "forward_walk_status": "pass", "blocker_reason": ""},
            {"candidate_id": "wizard:btc-usd-eth-usd:copula:1h", "lane": "wizard", "forward_walk_status": "pass", "blocker_reason": ""},
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_forward_walk.csv", index=False)
    pd.DataFrame(columns=["candidate_id", "lane", "pair"]).to_csv(tmp_path / "reports" / "brain" / "native_candidate_packets.csv", index=False)
    pd.DataFrame(columns=["candidate_id", "lane"]).to_csv(tmp_path / "reports" / "brain" / "native_forward_walk.csv", index=False)
    pd.DataFrame([{"pair": "BTC-USD-ETH-USD", "wizard_candidate_id": "wizard:btc-usd-eth-usd:static_(zscorer):1h", "wizard_setup_identity": "BTC-USD|ETH-USD|1h|320|static_(zscorer)", "native_candidate_id": "", "native_setup_identity": "", "agreement_status": "wizard_only", "wizard_support": True, "native_support": False, "arbitration_decision": "wizard_preferred", "orchestrator_status": "comparison_only", "evidence_path": "reports/brain/wizard_candidate_packets.csv"}]).to_csv(
        tmp_path / "reports" / "brain" / "overall_brain_summary.csv", index=False
    )

    result = build_promotion_readiness(root=tmp_path)
    frame = pd.read_csv(result.paths["promotion_ladder"])

    assert len(frame.loc[frame["lane"] == "wizard"]) == 2
    assert set(frame.loc[frame["lane"] == "wizard", "setup_identity"]) == {
        "BTC-USD|ETH-USD|1h|320|static_(zscorer)",
        "BTC-USD|ETH-USD|1h|320|copula",
    }


def test_three_brain_current_state_surfaces_orchestrator_status(tmp_path, monkeypatch):
    _write_minimal_inputs(tmp_path)
    build_three_brain_system(root=tmp_path)

    monkeypatch.setattr(active_pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(active_pipeline, "ACTIVE", tmp_path / "reports" / "active")
    monkeypatch.setattr(active_pipeline, "DASHBOARD", tmp_path / "reports" / "dashboard")
    monkeypatch.setattr(active_pipeline, "ML_REPORTS", tmp_path / "reports" / "ml")
    monkeypatch.setattr(active_pipeline, "DATA_ML", tmp_path / "data" / "ml")
    monkeypatch.setattr(active_pipeline, "MODELS", tmp_path / "models" / "trade_gate")
    (tmp_path / "data" / "ml").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "trade_gate").mkdir(parents=True, exist_ok=True)

    current = active_pipeline.current_state(root=tmp_path)
    dashboard = active_pipeline.build_command_dashboard(root=tmp_path)
    state = pd.read_csv(current.paths["current_state"])

    assert "orchestrator_status" in set(state["area"])
    assert {"candidate_id", "setup_identity", "setup_role", "setup_status", "setup_blocker"}.issubset(state.columns)
    assert dashboard.paths["overall_brain_summary"].exists()
    assert dashboard.paths["promotion_ladder"].exists()
    assert dashboard.paths["orchestrator_status"].exists()
    assert dashboard.paths["wizard_dashboard_investigation"].exists()
    assert dashboard.paths["forward_walk_strength_report"].exists()
    assert dashboard.paths["native_quality_report"].exists()
    assert dashboard.paths["native_candidate_diagnosis"].exists()
    assert dashboard.paths["wizard_candidate_repair_report"].exists()
    assert dashboard.paths["wizard_hourly_priority_capture_queue"].exists()
    assert dashboard.paths["native_focus_repair_report"].exists()
    assert dashboard.paths["wizard_hourly_repair_targets"].exists()
    assert dashboard.paths["candidate_repair_loop"].exists()
    assert dashboard.paths["shadow_rl_usefulness_report"].exists()
    assert dashboard.paths["shadow_rl_candidate_set"].exists()
    assert dashboard.paths["shadow_rl_intake_report"].exists()
    assert dashboard.paths["overall_arbitration_scorecard"].exists()
    assert dashboard.paths["blocker_persistence_report"].exists()
    wizard_row = state.loc[state["area"] == "wizard_readiness"].iloc[0]
    assert str(wizard_row["setup_identity"]).strip() != ""


def test_wizard_dashboard_investigation_carries_live_probe_status(tmp_path):
    _write_minimal_inputs(tmp_path)
    build_wizard_dashboard_ontology(root=tmp_path)
    build_three_brain_system(root=tmp_path)

    frame = pd.read_csv(tmp_path / "reports" / "brain" / "wizard_dashboard_investigation.csv")
    live = frame.loc[frame["investigation_area"] == "live_dashboard_access"].iloc[0]
    queue = frame.loc[frame["investigation_area"] == "exact_mode_capture_queue"].iloc[0]

    assert live["status"] == "signin_redirect"
    assert "sign-in" in live["detail"]
    assert queue["status"] == "ready"
    assert "top_pair=BTC-USD-ETH-USD" in queue["detail"]


def test_build_three_brain_system_refreshes_phase1_readiness_aliases(tmp_path):
    _write_minimal_inputs(tmp_path)
    pd.DataFrame([{"score_gate": "pass", "readiness_score": 0.91}]).to_csv(
        tmp_path / "reports" / "brain" / "brain_readiness_report.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "gate": "strategy_acceptance",
                "ready": True,
                "status": "ready",
                "blocker": "",
                "next_action": "go",
            }
        ]
    ).to_csv(tmp_path / "reports" / "priority_readiness.csv", index=False)

    result = build_three_brain_system(root=tmp_path)

    assert "wizard_readiness" in result.paths
    wizard = pd.read_csv(result.paths["wizard_readiness"])
    native = pd.read_csv(result.paths["native_readiness"])
    overall = pd.read_csv(result.paths["overall_readiness"])
    paper = pd.read_csv(result.paths["paper_status"])

    assert len(wizard) == 1
    assert len(native) == 1
    assert len(overall) == 1
    assert len(paper) == 1
    assert wizard.iloc[0]["area"] == "wizard_readiness"
    assert native.iloc[0]["area"] == "native_readiness"
    assert overall.iloc[0]["area"] == "overall_readiness"
    assert paper.iloc[0]["area"] == "paper_status"


def test_live_scanner_capture_reduces_exact_mode_blocker(tmp_path):
    _write_minimal_inputs(tmp_path)
    (tmp_path / "reports" / "active" / "crypto_wizards_live_scanner_capture.csv").write_text(
        "\n".join(
            [
                "pair,asset_x,asset_y,exchange,interval,dashboard_recommended_strategy,exact_mode,dashboard_pair_rank,returns_total,returns_total_pct,sharpe,source_path",
                "EUR-USD/NEAR-USD,EUR-USD,NEAR-USD,dydx,,OU (Spread),OU (Spread),1,0.52,52.0,1.6,reports/brain/wizard_dashboard_live_probe.json",
                "EUR-USD/RUNE-USD,EUR-USD,RUNE-USD,dydx,,Copula,Copula,2,0.238,23.8,1.45,reports/brain/wizard_dashboard_live_probe.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "brain" / "wizard_dashboard_live_probe.json").write_text(
        __import__("json").dumps(
            {
                "status": "dashboard_accessed",
                "detail": "scanner reachable after authenticated launch",
                "evidence_path": "reports/brain/wizard_dashboard_live_probe.json",
                "next_action": "capture more exact modes",
            }
        ),
        encoding="utf-8",
    )

    build_wizard_evidence(root=tmp_path)
    build_wizard_strategy_alignment_report(root=tmp_path)
    build_wizard_dashboard_ontology(root=tmp_path)
    build_three_brain_system(root=tmp_path)
    pd.DataFrame([{"score_gate": "pass", "readiness_score": 0.91}]).to_csv(
        tmp_path / "reports" / "brain" / "brain_readiness_report.csv", index=False
    )
    build_phase1_readiness_surfaces(tmp_path)

    investigation = pd.read_csv(tmp_path / "reports" / "brain" / "wizard_dashboard_investigation.csv")
    alignment = pd.read_csv(tmp_path / "reports" / "active" / "wizard_strategy_alignment_report.csv")
    live = investigation.loc[investigation["investigation_area"] == "live_dashboard_access"].iloc[0]
    exact_mode = investigation.loc[investigation["investigation_area"] == "exact_mode_capture"].iloc[0]
    strategy_gap = investigation.loc[investigation["investigation_area"] == "strategy_alignment_gap"].iloc[0]
    backlog = investigation.loc[investigation["investigation_area"] == "historical_alignment_backlog"].iloc[0]
    blocker = pd.read_csv(tmp_path / "reports" / "brain" / "wizard_readiness.csv").iloc[0]

    assert live["status"] == "dashboard_accessed"
    assert exact_mode["status"] == "covered"
    assert strategy_gap["status"] == "covered"
    assert backlog["status"] == "clear"
    assert "exact_mode_capture" not in str(blocker["blocker"])
    assert "strategy_alignment_gap" not in str(blocker["blocker"])
    assert "mapped_from_exact_mode" in set(alignment["strategy_mapping_status"])


def test_historical_alignment_backlog_is_visible_without_blocking_live_gap(tmp_path):
    _write_minimal_inputs(tmp_path)
    pd.DataFrame(
        [
            {
                "pair": "EUR-USD/FIL-USD",
                "interval": "daily",
                "dashboard_recommended_strategy": "",
                "exact_mode": "",
                "spread_id": "",
                "strategy_id": "",
                "local_strategy_id": "",
                "local_strategy_name": "",
                "local_strategy_family": "",
                "strategy_mapping_status": "unmapped",
                "source_first_action": "capture_dashboard_strategy_first",
                "evidence_path": "reports/active/crypto_wizards_next_best_sharpe_returns_queue.csv",
            },
            {
                "pair": "EUR-USD/NEAR-USD",
                "interval": "",
                "dashboard_recommended_strategy": "OU (Spread)",
                "exact_mode": "OU (Spread)",
                "spread_id": "2",
                "strategy_id": "1",
                "local_strategy_id": "14",
                "local_strategy_name": "OU Optimal",
                "local_strategy_family": "mean_reversion",
                "strategy_mapping_status": "mapped_from_exact_mode",
                "source_first_action": "run_dashboard_strategy_first:14",
                "evidence_path": "reports/active/crypto_wizards_live_scanner_capture.csv",
            },
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_strategy_alignment_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "EUR-USD/NEAR-USD",
                "asset_x": "EUR-USD",
                "asset_y": "NEAR-USD",
                "exchange": "dydx",
                "exact_mode": "OU (Spread)",
                "dashboard_recommended_strategy": "OU (Spread)",
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "crypto_wizards_live_scanner_capture.csv", index=False)
    (tmp_path / "reports" / "brain" / "wizard_dashboard_live_probe.json").write_text(
        __import__("json").dumps(
            {
                "status": "dashboard_accessed",
                "detail": "scanner reachable after authenticated launch",
                "evidence_path": "reports/brain/wizard_dashboard_live_probe.json",
                "next_action": "capture more exact modes",
            }
        ),
        encoding="utf-8",
    )
    build_wizard_dashboard_ontology(root=tmp_path)
    result = build_wizard_dashboard_investigation(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_dashboard_investigation"])

    strategy_gap = frame.loc[frame["investigation_area"] == "strategy_alignment_gap"].iloc[0]
    backlog = frame.loc[frame["investigation_area"] == "historical_alignment_backlog"].iloc[0]

    assert strategy_gap["status"] == "covered"
    assert "live_unmapped=0" in str(strategy_gap["detail"])
    assert backlog["status"] == "backlog"
    assert "historical_unmapped=1" in str(backlog["detail"])


def test_wizard_dashboard_ontology_includes_pair_detail_sections_when_available(tmp_path):
    _write_minimal_inputs(tmp_path)
    result = build_wizard_dashboard_ontology(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_dashboard_ontology"])

    assert "pair_detail_capture" in set(frame["top_level_section"])
    pair_detail_rows = frame[frame["top_level_section"] == "pair_detail_capture"]
    assert {"history_coverage", "field_dictionary"}.issubset(set(pair_detail_rows["section_type"]))


def test_forward_walk_strength_and_native_quality_encode_behavior(tmp_path):
    _write_minimal_inputs(tmp_path)
    build_three_brain_system(root=tmp_path)

    fw = pd.read_csv(tmp_path / "reports" / "brain" / "forward_walk_strength_report.csv")
    native_quality = pd.read_csv(tmp_path / "reports" / "brain" / "native_quality_report.csv")
    arbitration = pd.read_csv(tmp_path / "reports" / "brain" / "overall_arbitration_scorecard.csv")

    wizard_row = fw.loc[fw["lane"] == "wizard"].iloc[0]
    native_row = native_quality.loc[native_quality["pair"] == "BTC-USD-ETH-USD"].iloc[0]
    arb_row = arbitration.loc[arbitration["pair"] == "BTC-USD-ETH-USD"].iloc[0]

    assert wizard_row["strength_tier"] in {"acceptable", "strong"}
    assert wizard_row["evidence_density"] in {"moderate", "dense"}
    assert wizard_row["threshold_status"] == "pass"
    assert native_row["native_origin_type"] in {"local_manifest_only", "local_plus_dydx_history"}
    assert native_row["quality_status"] == "supported"
    assert arb_row["agreement_status"] == "both_support"
    assert arb_row["promotion_ready_count"] >= 1


def test_native_candidate_diagnosis_splits_concrete_blockers(tmp_path):
    _write_minimal_inputs(tmp_path)
    history_path = tmp_path / "data" / "raw" / "pair_details" / "pair_btc_eth_5mins_dydx_long_history_derived_history.json"
    history_path.write_text(
        __import__("json").dumps(
            {
                "history": [
                    {"timestamp": "2026-07-01T00:00:00Z", "rolling_zscore": 2.2, "spread_return_1": 0.01},
                    {"timestamp": "2026-07-01T00:05:00Z", "rolling_zscore": 0.1, "spread_return_1": 0.0},
                    {"timestamp": "2026-07-01T00:10:00Z", "rolling_zscore": 0.2, "spread_return_1": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    build_three_brain_system(root=tmp_path)

    diagnosis = pd.read_csv(tmp_path / "reports" / "brain" / "native_candidate_diagnosis.csv")
    row = diagnosis.loc[diagnosis["pair"] == "BTC-USD-ETH-USD"].iloc[0]

    assert row["current_blocker"] == "native_too_few_trades"
    assert "minimum 3" in str(row["blocker_detail"])
    assert row["evaluation_mode"].startswith("native_local_strategy_logic:")


def test_wizard_candidate_repair_report_surfaces_blocker_chain(tmp_path):
    _write_minimal_inputs(tmp_path)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "setup_identity": "BTC-USD|ETH-USD|1h||nan",
                "diagnostic_blocker": "missing_exact_mode;missing_correlation;missing_ecm;parity_missing_local_data",
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_diagnostic_confirmation.csv", index=False)
    pd.DataFrame(columns=["pair", "setup_identity", "verification_status", "summary_path"]).to_csv(
        tmp_path / "reports" / "active" / "wizard_local_verification_batch.csv", index=False
    )
    build_three_brain_system(root=tmp_path)

    repair = pd.read_csv(tmp_path / "reports" / "brain" / "wizard_candidate_repair_report.csv")
    row = repair.loc[repair["pair"] == "BTC-USD-ETH-USD"].iloc[0]

    assert row["exact_mode_capture_status"] in {"missing", "placeholder"}
    assert row["correlation_status"] == "missing"
    assert row["ecm_status"] == "missing"
    assert row["local_verification_status"] == "missing"
    assert "missing_exact_mode" in str(row["current_blocker"])
    assert str(row["current_blocker"]) == "missing_exact_mode"


def test_normalize_wizard_blocker_chain_prioritizes_capture_stage():
    assert _normalize_wizard_blocker_chain(
        ["missing_exact_mode", "missing_correlation", "parity_missing_local_data", "wizard_local_verification_missing"]
    ) == "missing_exact_mode"
    assert _normalize_wizard_blocker_chain(
        ["missing_correlation", "missing_ecm", "parity_missing_local_data", "wizard_local_verification_missing"]
    ) == "missing_correlation;missing_ecm"
    assert _normalize_wizard_blocker_chain(
        ["parity_missing_local_data", "wizard_local_verification_missing"]
    ) == "parity_missing_local_data;wizard_local_verification_missing"


def test_candidate_repair_loop_stays_setup_specific(tmp_path):
    _write_minimal_inputs(tmp_path)
    build_three_brain_system(root=tmp_path)

    repair_loop = pd.read_csv(tmp_path / "reports" / "brain" / "candidate_repair_loop.csv")

    assert {"candidate_id", "setup_identity", "lane", "current_blocker", "repair_action"}.issubset(repair_loop.columns)
    assert repair_loop["setup_identity"].astype(str).str.strip().ne("").all()


def test_wizard_exact_mode_capture_status_uses_captured_mode_even_with_sparse_setup_identity():
    row = pd.Series(
        {
            "setup_identity": "BLUR-USD|ETHFI-USD|daily||copula",
            "exact_mode": "Copula",
            "spread_id": 1,
            "strategy_id": 3,
        }
    )

    assert _wizard_exact_mode_capture_status(row, "missing_ecm;parity_missing_local_data") == "captured"


def test_wizard_candidate_repair_report_matches_verification_rows_across_pair_formats(tmp_path):
    (tmp_path / "reports" / "brain").mkdir(parents=True)
    (tmp_path / "reports" / "active").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BLUR-USD-ETHFI-USD",
                "candidate_id": "wizard:BLUR-USD-ETHFI-USD:copula:daily",
                "setup_identity": "BLUR-USD|ETHFI-USD|daily||copula",
                "source_path": "reports/active/crypto_wizards_pair_page_capture.csv",
                "blocker_state": "",
                "exact_mode": "Copula",
                "strategy_mode": "Copula",
                "spread_id": 1,
                "strategy_id": 3,
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_candidate_packets.csv", index=False)
    pd.DataFrame(
        [{"pair": "BLUR-USD/ETHFI-USD", "setup_identity": "BLUR-USD|ETHFI-USD|daily||copula", "hypothesis_status": "HYPOTHESIS_READY"}]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_hypotheses.csv", index=False)
    pd.DataFrame(
        [{"pair": "BLUR-USD/ETHFI-USD", "setup_identity": "BLUR-USD|ETHFI-USD|daily||copula", "diagnostic_blocker": ""}]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_diagnostic_confirmation.csv", index=False)
    pd.DataFrame(
        [{"pair": "BLUR-USD/ETHFI-USD", "setup_identity": "BLUR-USD|ETHFI-USD|daily||copula", "parity_status": "NOT_REPLICATED"}]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_vs_local_parity_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BLUR-USD/ETHFI-USD",
                "setup_identity": "BLUR-USD|ETHFI-USD|daily|1000|copula",
                "verification_status": "verified",
                "summary_path": "reports/active/blurusd_ethfiusd_copula_verification_after_cost.csv",
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_local_verification_batch.csv", index=False)

    result = build_wizard_candidate_repair_report(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_candidate_repair_report"])
    row = frame.iloc[0]

    assert row["local_verification_status"] == "verified"
    assert row["parity_status"] == "NOT_REPLICATED"


def test_shadow_evidence_quality_uses_wizard_repair_state_for_captured_exact_mode():
    promotion_row = pd.Series({"blocker": "local_verification_not_run"})
    wizard_repair_row = pd.Series(
        {
            "exact_mode_capture_status": "captured",
            "current_blocker": "missing_ecm;parity_missing_local_data;wizard_local_verification_missing",
        }
    )

    assert (
        _shadow_evidence_quality_status(
            lane="wizard",
            promotion_row=promotion_row,
            wizard_hourly_row=None,
            wizard_repair_row=wizard_repair_row,
            native_focus_row=None,
        )
        == "needs_capture_completion"
    )


def test_wizard_candidate_repair_report_uses_hourly_match_to_clear_missing_exact_mode(tmp_path):
    (tmp_path / "reports" / "brain").mkdir(parents=True)
    (tmp_path / "reports" / "active").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "DOGE-USD-MORPHO-USD",
                "candidate_id": "wizard:DOGE-USD-MORPHO-USD::daily",
                "setup_identity": "DOGE-USD|MORPHO-USD|daily",
                "source_path": "reports/active/crypto_wizards_next_best_sharpe_returns_queue.csv",
                "blocker_state": "missing_exact_mode",
                "exact_mode": "",
                "strategy_mode": "",
                "spread_id": "",
                "strategy_id": "",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_candidate_packets.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "DOGE-USD-MORPHO-USD",
                "setup_identity": "DOGE-USD|MORPHO-USD|daily",
                "diagnostic_blocker": "missing_correlation;missing_ecm;missing_mean_reversion",
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_diagnostic_confirmation.csv", index=False)
    pd.DataFrame(columns=["pair", "setup_identity", "verification_status", "summary_path"]).to_csv(
        tmp_path / "reports" / "active" / "wizard_local_verification_batch.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "candidate_id": "wizard:DOGE-USD-MORPHO-USD::daily",
                "matched_hourly_strategy": "Static (Spread)",
                "review_status": "new",
                "candidate_status": "review_now",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_hourly_repair_targets.csv", index=False)

    result = build_wizard_candidate_repair_report(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_candidate_repair_report"])
    row = frame.iloc[0]

    assert row["exact_mode_capture_status"] in {"captured", "placeholder"}
    assert row["current_blocker"] == "missing_correlation;missing_ecm;missing_mean_reversion"
    assert row["next_repair_action"] == "capture_correlation_panel_and_persist_setup_metrics"
    assert "wizard_hourly_candidate_queue.csv" in row["evidence_path"]


def test_build_three_brain_system_refreshes_hourly_targets_before_wizard_repair(tmp_path):
    _write_minimal_inputs(tmp_path)
    pd.DataFrame(
        [
            {
                "scan_timestamp": "2026-07-04T13:20:00+00:00",
                "pair": "BTC-USD/ETH-USD",
                "periods": "scanner",
                "timeframe": "1h",
                "strategy": "Static (ZScoreR)",
                "annualized_return_pct": "12.0",
                "sharpe": "1.2",
                "closed_trades": "",
                "review_status": "new",
                "highest_anomaly_severity": "",
                "candidate_status": "review_now",
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_hourly_database" / "wizard_hourly_candidate_queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD/ETH-USD",
                "actionable_variant_count": 1,
                "blocked_variant_count": 0,
                "top_strategy": "Static (ZScoreR)",
                "top_timeframe": "1h",
                "top_annualized_return_pct": "12.0",
                "top_sharpe": "1.2",
                "top_closed_trades": "",
                "pair_status": "review_now",
                "hourly_rank": 1,
                "recommended_this_hour": True,
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_hourly_database" / "wizard_hourly_pair_queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "candidate_id": "wizard:BTC-USD-ETH-USD:static_(zscorer):1h",
                "setup_identity": "BTC-USD|ETH-USD|1h|320|static_(zscorer)",
                "source_path": "reports/source.csv",
                "blocker_state": "missing_exact_mode;missing_correlation;missing_ecm;missing_mean_reversion",
                "strategy_mode": "Static (ZScoreR)",
                "timeframe": "1h",
                "spread_id": "3",
                "strategy_id": "1",
                "setup_role": "primary",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_candidate_packets.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "setup_identity": "BTC-USD|ETH-USD|1h|320|static_(zscorer)",
                "diagnostic_blocker": "missing_correlation;missing_ecm;missing_mean_reversion",
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_diagnostic_confirmation.csv", index=False)

    build_three_brain_system(root=tmp_path)

    repair = pd.read_csv(tmp_path / "reports" / "brain" / "wizard_candidate_repair_report.csv")
    row = repair.loc[repair["candidate_id"] == "wizard:BTC-USD-ETH-USD:static_(zscorer):1h"].iloc[0]

    assert row["exact_mode_capture_status"] in {"captured", "placeholder"}
    assert "missing_exact_mode" not in row["current_blocker"]
    assert row["current_blocker"] == "missing_correlation;missing_ecm"
    assert row["next_repair_action"] == "capture_correlation_panel_and_persist_setup_metrics"


def test_build_three_brain_system_refreshes_hourly_queue_from_live_scanner(tmp_path):
    _write_minimal_inputs(tmp_path)
    hourly = tmp_path / "reports" / "active" / "wizard_hourly_database"
    pd.DataFrame(
        [
            {
                "scan_timestamp": "2026-07-04T13:20:00+00:00",
                "source_name": "wizard_scanner_live_dom",
                "pair": "BTC-USD/MORPHO-USD",
                "periods": "scanner",
                "timeframe": "Live",
                "strategy": "OU (Spread)",
                "sharpe": "1.4",
                "annualized_return_pct": "67.5",
                "review_status": "new",
                "should_deep_capture": True,
                "anomaly_count": 0,
                "highest_anomaly_severity": "",
            }
        ]
    ).to_csv(hourly / "wizard_hourly_current_alias.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD/MORPHO-USD",
                "actionable_variant_count": 1,
                "blocked_variant_count": 0,
                "top_strategy": "OU (Spread)",
                "top_timeframe": "Live",
                "top_annualized_return_pct": "67.5",
                "top_sharpe": "1.4",
                "top_closed_trades": "",
                "pair_status": "review_now",
                "hourly_rank": 1,
                "recommended_this_hour": True,
            }
        ]
    ).to_csv(hourly / "wizard_hourly_pair_queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "scan_timestamp": "2026-07-04T13:20:00+00:00",
                "pair": "BTC-USD/MORPHO-USD",
                "periods": "scanner",
                "timeframe": "Live",
                "strategy": "OU (Spread)",
                "annualized_return_pct": "67.5",
                "sharpe": "1.4",
                "closed_trades": "",
                "review_status": "new",
                "highest_anomaly_severity": "",
                "candidate_status": "review_now",
            }
        ]
    ).to_csv(hourly / "wizard_hourly_candidate_queue.csv", index=False)

    live_dir = tmp_path / "data" / "collected" / "wizard_scanner_static_spread_visible_live"
    live_dir.mkdir(parents=True)
    (live_dir / "wizard_scanner_static_spread_visible_live_latest.json").write_text(
        json.dumps(
            {
                "captured_at": "2026-07-04T19:52:57.957Z",
                "scanner_filters": {"strategy": "Static (Spread)"},
                "rows": [
                    {"pair": "EIGEN-USD / XRP-USD", "pair_x": "EIGEN-USD", "pair_y": "XRP-USD", "reward_sharpe": "2.04"},
                    {"pair": "DOGE-USD / SYRUP-USD", "pair_x": "DOGE-USD", "pair_y": "SYRUP-USD", "reward_sharpe": "2.03"},
                ],
            }
        ),
        encoding="utf-8",
    )

    build_three_brain_system(root=tmp_path)

    pair_queue = pd.read_csv(hourly / "wizard_hourly_pair_queue.csv")

    assert "EIGEN-USD/XRP-USD" in set(pair_queue["pair"].astype(str))
    assert "DOGE-USD/SYRUP-USD" in set(pair_queue["pair"].astype(str))
    assert "BTC-USD/MORPHO-USD" not in set(pair_queue["pair"].astype(str))


def test_wizard_candidate_repair_report_uses_scanner_dependency_capture_to_clear_missing_correlation(tmp_path):
    (tmp_path / "reports" / "brain").mkdir(parents=True)
    (tmp_path / "reports" / "active").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "DOGE-USD-MORPHO-USD",
                "candidate_id": "wizard:DOGE-USD-MORPHO-USD:static_(spread):daily",
                "setup_identity": "DOGE-USD|MORPHO-USD|daily||static_(spread)",
                "source_path": "reports/brain/wizard_dashboard_live_probe.json",
                "blocker_state": "missing_correlation;missing_ecm",
                "exact_mode": "Static (Spread)",
                "strategy_mode": "Static (Spread)",
                "timeframe": "daily",
                "spread_id": "3",
                "strategy_id": "1",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_candidate_packets.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "DOGE-USD-MORPHO-USD",
                "setup_identity": "DOGE-USD|MORPHO-USD|daily||static_(spread)",
                "diagnostic_blocker": "missing_correlation;missing_ecm;missing_mean_reversion",
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_diagnostic_confirmation.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "wizard:DOGE-USD-MORPHO-USD:static_(spread):daily",
                "matched_hourly_strategy": "Static (Spread)",
                "review_status": "new",
                "candidate_status": "review_now",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_hourly_repair_targets.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "DOGE-USD-MORPHO-USD",
                "timeframe": "daily",
                "strategy_mode": "Static (Spread)",
                "corr_jneg": 0.89,
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_scanner_dependency_capture.csv", index=False)

    result = build_wizard_candidate_repair_report(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_candidate_repair_report"])
    row = frame.iloc[0]

    assert row["exact_mode_capture_status"] == "captured"
    assert row["correlation_status"] == "covered"
    assert row["ecm_status"] == "missing"
    assert row["current_blocker"] == "missing_ecm"
    assert row["next_repair_action"] == "import_ecm_fields_for_exact_wizard_setup"
    assert "wizard_scanner_dependency_capture.csv" in row["evidence_path"]


def test_wizard_candidate_repair_report_uses_hourly_strategy_to_match_scanner_dependency_capture(tmp_path):
    (tmp_path / "reports" / "brain").mkdir(parents=True)
    (tmp_path / "reports" / "active").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "DOGE-USD-EIGEN-USD",
                "candidate_id": "wizard:DOGE-USD-EIGEN-USD::daily",
                "setup_identity": "DOGE-USD|EIGEN-USD|daily",
                "source_path": "reports/brain/wizard_dashboard_live_probe.json",
                "blocker_state": "missing_correlation;missing_ecm;missing_mean_reversion",
                "timeframe": "daily",
                "strategy_mode": "",
                "exact_mode": "",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_candidate_packets.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "DOGE-USD-EIGEN-USD",
                "setup_identity": "DOGE-USD|EIGEN-USD|daily",
                "diagnostic_blocker": "missing_correlation;missing_ecm;missing_mean_reversion",
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_diagnostic_confirmation.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "wizard:DOGE-USD-EIGEN-USD::daily",
                "matched_hourly_strategy": "Static (Spread)",
                "review_status": "new",
                "candidate_status": "review_now",
            }
        ]
    ).to_csv(tmp_path / "reports" / "brain" / "wizard_hourly_repair_targets.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "DOGE-USD-EIGEN-USD",
                "timeframe": "daily",
                "strategy_mode": "Static (Spread)",
                "corr_jneg": 0.98,
            }
        ]
    ).to_csv(tmp_path / "reports" / "active" / "wizard_scanner_dependency_capture.csv", index=False)

    result = build_wizard_candidate_repair_report(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_candidate_repair_report"])
    row = frame.iloc[0]

    assert row["exact_mode_capture_status"] == "captured"
    assert row["correlation_status"] == "covered"
    assert row["current_blocker"] == "missing_ecm;missing_mean_reversion"


def test_shadow_evidence_quality_distinguishes_quality_blocked_wizard_candidate():
    promotion_row = pd.Series({"blocker": "wizard_fold_metrics_weak"})
    wizard_repair_row = pd.Series(
        {
            "exact_mode_capture_status": "captured",
            "local_verification_status": "verified",
            "current_blocker": "wizard_candidate_ready_for_local_after_cost_test;parity_not_replicated",
        }
    )

    assert (
        _shadow_evidence_quality_status(
            lane="wizard",
            promotion_row=promotion_row,
            wizard_hourly_row=None,
            wizard_repair_row=wizard_repair_row,
            native_focus_row=None,
        )
        == "quality_blocked_candidate"
    )


def test_wizard_hourly_targets_feed_candidate_repair_loop(tmp_path):
    _write_minimal_inputs(tmp_path)
    build_three_brain_system(root=tmp_path)

    hourly_targets = pd.read_csv(tmp_path / "reports" / "brain" / "wizard_hourly_repair_targets.csv")
    repair_loop = pd.read_csv(tmp_path / "reports" / "brain" / "candidate_repair_loop.csv")

    hourly_row = hourly_targets.loc[hourly_targets["pair"] == "BTC-USD-ETH-USD"].iloc[0]
    repair_row = repair_loop.loc[(repair_loop["pair"] == "BTC-USD-ETH-USD") & (repair_loop["lane"] == "wizard")].iloc[0]

    assert hourly_row["hourly_match_status"] == "matched"
    assert hourly_row["review_status"] == "changed"
    assert bool(hourly_row["pair_recommended_this_hour"]) is True
    assert repair_row["hourly_review_status"] == "changed"
    assert bool(repair_row["hourly_pair_recommended"]) is True
    assert repair_row["repair_action"] == "rerun_local_verification_on_current_wizard_setup"


def test_wizard_repair_loop_action_switches_to_rerun_when_capture_is_already_complete():
    repair_row = pd.Series(
        {
            "next_repair_action": "wizard_setup_clear_monitor_for_promotion",
            "current_blocker": "sharpe_below_1.75;weak_correlation;parity_not_replicated",
            "exact_mode_capture_status": "captured",
        }
    )
    hourly_row = pd.Series(
        {
            "review_status": "changed",
            "candidate_status": "review_now",
            "pair_recommended_this_hour": True,
        }
    )

    assert _wizard_repair_loop_action(repair_row, hourly_row) == "rerun_local_verification_on_current_wizard_setup"


def test_wizard_hourly_targets_match_slash_pair_queue_entries(tmp_path):
    _write_minimal_inputs(tmp_path)
    pair_queue_path = tmp_path / "reports" / "active" / "wizard_hourly_database" / "wizard_hourly_pair_queue.csv"
    pair_queue = pd.read_csv(pair_queue_path)
    pair_queue.loc[:, "pair"] = "BTC-USD/ETH-USD"
    pair_queue.to_csv(pair_queue_path, index=False)

    build_three_brain_system(root=tmp_path)

    hourly_targets = pd.read_csv(tmp_path / "reports" / "brain" / "wizard_hourly_repair_targets.csv")
    repair_loop = pd.read_csv(tmp_path / "reports" / "brain" / "candidate_repair_loop.csv")

    hourly_row = hourly_targets.loc[hourly_targets["candidate_id"] == "wizard:BTC-USD-ETH-USD:static_(zscorer):1h"].iloc[0]
    repair_row = repair_loop.loc[repair_loop["candidate_id"] == "wizard:BTC-USD-ETH-USD:static_(zscorer):1h"].iloc[0]

    assert bool(hourly_row["pair_recommended_this_hour"]) is True
    assert int(hourly_row["pair_hourly_rank"]) == 1
    assert bool(repair_row["hourly_pair_recommended"]) is True


def test_native_focus_repair_report_surfaces_hype_upgrade_paths(tmp_path):
    _write_minimal_inputs(tmp_path)
    build_three_brain_system(root=tmp_path)

    focus = pd.read_csv(tmp_path / "reports" / "brain" / "native_focus_repair_report.csv")
    assert not focus.empty
    lead_row = focus.iloc[0]

    assert "focused_5m_dydx_path" in focus.columns
    assert "mean_fold_profit_factor" in focus.columns
    assert "forward_walk_status" in focus.columns
    assert lead_row["next_repair_action"] in {
        "switch_native_manifest_to_focused_5m_dydx_history_then_rerun",
        "rerun_native_forward_walk_with_focused_5m_dydx_history",
        "candidate_clear_monitor_for_promotion",
        "preserve strong pair-specific model support while improving global model gate acceptance",
        "expand_history_and_retest_for_strategy_acceptance",
    }


def test_candidate_repair_loop_prioritizes_supported_native_candidates(tmp_path):
    reports = tmp_path / "reports" / "brain"
    reports.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "candidate_id": "native:AAA",
                "lane": "native",
                "pair": "AAA-BBB",
                "setup_identity": "native:AAA",
                "setup_role": "primary",
                "promotion_stage": "forward_walk_blocked",
                "paper_credible": False,
                "blocker": "native_weak_profit_factor",
            },
            {
                "candidate_id": "native:HYPE",
                "lane": "native",
                "pair": "BTC-USD-HYPE-USD",
                "setup_identity": "native:HYPE",
                "setup_role": "primary",
                "promotion_stage": "orchestrator_review",
                "paper_credible": True,
                "blocker": "",
                "paper_status": "research_only",
            },
        ]
    ).to_csv(reports / "promotion_ladder.csv", index=False)
    pd.DataFrame(
        [
            {"candidate_id": "native:AAA", "pair": "AAA-BBB", "current_blocker": "native_weak_profit_factor", "next_repair_action": "improve_pair_specific_entry_exit_logic_then_rerun", "selected_history_source": "aaa.json"},
            {"candidate_id": "native:HYPE", "pair": "BTC-USD-HYPE-USD", "current_blocker": "", "next_repair_action": "candidate_clear_monitor_for_promotion", "selected_history_source": "hype.json"},
        ]
    ).to_csv(reports / "native_candidate_diagnosis.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "AAA-BBB", "combined_score": 0.2},
            {"pair": "BTC-USD-HYPE-USD", "combined_score": 0.9},
        ]
    ).to_csv(reports / "native_quality_report.csv", index=False)
    pd.DataFrame(columns=["candidate_id", "current_blocker", "next_repair_action", "evidence_path"]).to_csv(
        reports / "wizard_candidate_repair_report.csv", index=False
    )
    pd.DataFrame(columns=["candidate_id", "review_status", "candidate_status", "pair_recommended_this_hour"]).to_csv(
        reports / "wizard_hourly_repair_targets.csv", index=False
    )

    build_candidate_repair_loop(root=tmp_path)

    repair_loop = pd.read_csv(reports / "candidate_repair_loop.csv")
    native_rows = repair_loop.loc[repair_loop["lane"] == "native"]

    assert not native_rows.empty
    assert native_rows.iloc[0]["pair"] == "BTC-USD-HYPE-USD"
    assert native_rows.iloc[0]["repair_action"] == "expand_history_and_retest_for_strategy_acceptance"
    assert native_rows.iloc[0]["current_blocker"] == "strategy_acceptance_not_ready"
    assert bool(native_rows.iloc[0]["rerun_required"])


def test_candidate_repair_loop_prefers_global_paper_gate_for_supported_native_candidates(tmp_path):
    reports = tmp_path / "reports" / "brain"
    rl_reports = tmp_path / "reports" / "rl"
    reports.mkdir(parents=True, exist_ok=True)
    rl_reports.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "candidate_id": "native:HYPE",
                "lane": "native",
                "pair": "BTC-USD-HYPE-USD",
                "setup_identity": "native:HYPE",
                "setup_role": "primary",
                "promotion_stage": "orchestrator_review",
                "paper_credible": True,
                "blocker": "",
                "paper_status": "research_only",
            },
        ]
    ).to_csv(reports / "promotion_ladder.csv", index=False)
    pd.DataFrame(
        [
            {"candidate_id": "native:HYPE", "pair": "BTC-USD-HYPE-USD", "current_blocker": "", "next_repair_action": "candidate_clear_monitor_for_promotion", "selected_history_source": "hype.json"},
        ]
    ).to_csv(reports / "native_candidate_diagnosis.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "combined_score": 0.9},
        ]
    ).to_csv(reports / "native_quality_report.csv", index=False)
    pd.DataFrame(columns=["candidate_id", "current_blocker", "next_repair_action", "evidence_path"]).to_csv(
        reports / "wizard_candidate_repair_report.csv", index=False
    )
    pd.DataFrame(columns=["candidate_id", "review_status", "candidate_status", "pair_recommended_this_hour"]).to_csv(
        reports / "wizard_hourly_repair_targets.csv", index=False
    )
    pd.DataFrame(
        [
            {"blocker": "paper_execution_not_ready", "next_action": "wire_dydx_submission_path"},
        ]
    ).to_csv(rl_reports / "base_rl_paper_handoff_status.csv", index=False)
    pd.DataFrame(
        [
            {"gate": "strategy_acceptance", "ready": True, "status": "ready", "blocker": ""},
        ]
    ).to_csv(tmp_path / "reports" / "priority_readiness.csv", index=False)
    pd.DataFrame(
        [
            {"step": "production_eligibility", "ready": True, "status": "ready", "blocker": ""},
        ]
    ).to_csv(tmp_path / "reports" / "strategy_acceptance_checklist.csv", index=False)
    pd.DataFrame(
        [
            {"acceptance_reason": "", "research_next_step": ""},
        ]
    ).to_csv(tmp_path / "reports" / "acceptance_report.csv", index=False)

    build_candidate_repair_loop(root=tmp_path)

    repair_loop = pd.read_csv(reports / "candidate_repair_loop.csv")
    native_row = repair_loop.loc[repair_loop["lane"] == "native"].iloc[0]

    assert native_row["current_blocker"] == "paper_execution_not_ready"
    assert native_row["repair_action"] == "wire_dydx_submission_path"


def test_native_focus_repair_report_surfaces_global_paper_gate_for_strong_hype_candidates(tmp_path):
    reports = tmp_path / "reports" / "brain"
    reports.mkdir(parents=True, exist_ok=True)
    top_reports = tmp_path / "reports"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "pair_btc_hype_5mins_dydx_candles_derived_history.json").write_text("{}", encoding="utf-8")
    (pair_dir / "pair_sol_hype_5mins_dydx_candles_derived_history.json").write_text("{}", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "candidate_id": "native:BTC-HYPE",
                "selected_history_source": str(pair_dir / "pair_btc_hype_5mins_dydx_candles_derived_history.json"),
                "current_blocker": "",
                "blocker_detail": "",
                "next_repair_action": "candidate_clear_monitor_for_promotion",
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "candidate_id": "native:SOL-HYPE",
                "selected_history_source": str(pair_dir / "pair_sol_hype_5mins_dydx_candles_derived_history.json"),
                "current_blocker": "",
                "blocker_detail": "",
                "next_repair_action": "candidate_clear_monitor_for_promotion",
            },
        ]
    ).to_csv(reports / "native_candidate_diagnosis.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "quality_status": "supported"},
            {"pair": "SOL-USD-HYPE-USD", "quality_status": "supported"},
        ]
    ).to_csv(reports / "native_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "lane": "native",
                "paper_credible": True,
                "promotion_stage": "orchestrator_review",
                "paper_status": "research_only",
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "lane": "native",
                "paper_credible": True,
                "promotion_stage": "orchestrator_review",
                "paper_status": "research_only",
            },
        ]
    ).to_csv(reports / "promotion_ladder.csv", index=False)
    pd.DataFrame(
        [
            {"gate": "strategy_acceptance", "blocker": "strategy_acceptance_not_ready"},
        ]
    ).to_csv(top_reports / "priority_readiness.csv", index=False)
    pd.DataFrame(
        [
            {
                "step": "production_eligibility",
                "blocker": "no_production_eligible_strategy",
            },
            {
                "step": "exchange_cost_model_alignment",
                "blocker": "missing_cost_execution_alignment:BTC-USD-WIF-USD",
            },
        ]
    ).to_csv(top_reports / "strategy_acceptance_checklist.csv", index=False)
    pd.DataFrame(
        [
            {
                "acceptance_reason": "passing_pairs<2",
                "research_next_step": "expand_history_and_retest",
            }
        ]
    ).to_csv(top_reports / "acceptance_report.csv", index=False)

    result = build_native_focus_repair_report(root=tmp_path)
    focus = pd.read_csv(result.paths["native_focus_repair_report"])
    btc_row = focus.loc[focus["pair"] == "BTC-USD-HYPE-USD"].iloc[0]

    assert btc_row["current_blocker"] == "strategy_acceptance_not_ready"
    assert "production_gate=no_production_eligible_strategy" in str(btc_row["blocker_detail"])
    assert "top_failure=passing_pairs<2" in str(btc_row["blocker_detail"])
    assert btc_row["next_repair_action"] == "expand_history_and_retest_for_strategy_acceptance"


def test_native_focus_repair_report_uses_paper_gate_after_strategy_acceptance_clears(tmp_path):
    reports = tmp_path / "reports" / "brain"
    rl_reports = tmp_path / "reports" / "rl"
    reports.mkdir(parents=True, exist_ok=True)
    rl_reports.mkdir(parents=True, exist_ok=True)
    top_reports = tmp_path / "reports"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "pair_btc_hype_5mins_dydx_candles_derived_history.json").write_text("{}", encoding="utf-8")
    (pair_dir / "pair_sol_hype_5mins_dydx_candles_derived_history.json").write_text("{}", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "candidate_id": "native:BTC-HYPE",
                "selected_history_source": str(pair_dir / "pair_btc_hype_5mins_dydx_candles_derived_history.json"),
                "current_blocker": "",
                "blocker_detail": "",
                "next_repair_action": "candidate_clear_monitor_for_promotion",
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "candidate_id": "native:SOL-HYPE",
                "selected_history_source": str(pair_dir / "pair_sol_hype_5mins_dydx_candles_derived_history.json"),
                "current_blocker": "",
                "blocker_detail": "",
                "next_repair_action": "candidate_clear_monitor_for_promotion",
            },
        ]
    ).to_csv(reports / "native_candidate_diagnosis.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "quality_status": "supported"},
            {"pair": "SOL-USD-HYPE-USD", "quality_status": "supported"},
        ]
    ).to_csv(reports / "native_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "lane": "native",
                "paper_credible": True,
                "promotion_stage": "orchestrator_review",
                "paper_status": "research_only",
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "lane": "native",
                "paper_credible": True,
                "promotion_stage": "orchestrator_review",
                "paper_status": "research_only",
            },
        ]
    ).to_csv(reports / "promotion_ladder.csv", index=False)
    pd.DataFrame(
        [
            {"gate": "strategy_acceptance", "ready": True, "status": "ready", "blocker": ""},
        ]
    ).to_csv(top_reports / "priority_readiness.csv", index=False)
    pd.DataFrame(
        [
            {"status": "research_only", "blocker": "paper_execution_not_ready", "next_action": "wire_dydx_submission_path"},
        ]
    ).to_csv(rl_reports / "base_rl_paper_handoff_status.csv", index=False)

    result = build_native_focus_repair_report(root=tmp_path)
    focus = pd.read_csv(result.paths["native_focus_repair_report"])
    btc_row = focus.loc[focus["pair"] == "BTC-USD-HYPE-USD"].iloc[0]

    assert btc_row["current_blocker"] == "paper_execution_not_ready"
    assert "paper_gate=paper_execution_not_ready" in str(btc_row["blocker_detail"])
    assert btc_row["next_repair_action"] == "wire_dydx_submission_path"


def test_native_focus_repair_report_prefers_execution_preflight_action_for_paper_gate(tmp_path):
    reports = tmp_path / "reports" / "brain"
    rl_reports = tmp_path / "reports" / "rl"
    top_reports = tmp_path / "reports"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    reports.mkdir(parents=True, exist_ok=True)
    rl_reports.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "pair_btc_hype_5mins_dydx_candles_derived_history.json").write_text("{}", encoding="utf-8")
    (pair_dir / "pair_sol_hype_5mins_dydx_candles_derived_history.json").write_text("{}", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "candidate_id": "native:BTC-HYPE",
                "selected_history_source": str(pair_dir / "pair_btc_hype_5mins_dydx_candles_derived_history.json"),
                "current_blocker": "",
                "blocker_detail": "",
                "next_repair_action": "candidate_clear_monitor_for_promotion",
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "candidate_id": "native:SOL-HYPE",
                "selected_history_source": str(pair_dir / "pair_sol_hype_5mins_dydx_candles_derived_history.json"),
                "current_blocker": "",
                "blocker_detail": "",
                "next_repair_action": "candidate_clear_monitor_for_promotion",
            },
        ]
    ).to_csv(reports / "native_candidate_diagnosis.csv", index=False)
    pd.DataFrame(
        [{"pair": "BTC-USD-HYPE-USD", "quality_status": "supported"}, {"pair": "SOL-USD-HYPE-USD", "quality_status": "supported"}]
    ).to_csv(reports / "native_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "lane": "native", "paper_credible": True, "promotion_stage": "orchestrator_review", "paper_status": "research_only"},
            {"pair": "SOL-USD-HYPE-USD", "lane": "native", "paper_credible": True, "promotion_stage": "orchestrator_review", "paper_status": "research_only"},
        ]
    ).to_csv(reports / "promotion_ladder.csv", index=False)
    pd.DataFrame([{"gate": "strategy_acceptance", "ready": True, "status": "ready", "blocker": ""}]).to_csv(
        top_reports / "priority_readiness.csv", index=False
    )
    pd.DataFrame([{"status": "research_only", "blocker": "paper_execution_not_ready", "next_action": "generic_paper_gate_action"}]).to_csv(
        rl_reports / "base_rl_paper_handoff_status.csv", index=False
    )
    pd.DataFrame(
        [
            {"step": "strategy_acceptance_dependency", "ready": True, "status": "ready", "blocker": "", "evidence": "", "next_action": "allow research-gated paper plans"},
            {"step": "dydx_testnet_dependency", "ready": False, "status": "blocked", "blocker": "missing_wallet_address;missing_private_key", "evidence": "", "next_action": "set DYDX_TESTNET_WALLET_ADDRESS and DYDX_TESTNET_PRIVATE_KEY in .env.local"},
            {"step": "paper_submission_gate", "ready": False, "status": "blocked", "blocker": "strategy_or_dydx_gate_not_ready", "evidence": "", "next_action": "do not submit paper orders until strategy and dYdX dependencies are ready"},
        ]
    ).to_csv(top_reports / "paper_execution_preflight.csv", index=False)

    result = build_native_focus_repair_report(root=tmp_path)
    focus = pd.read_csv(result.paths["native_focus_repair_report"])
    btc_row = focus.loc[focus["pair"] == "BTC-USD-HYPE-USD"].iloc[0]

    assert btc_row["next_repair_action"] == "set DYDX_TESTNET_WALLET_ADDRESS and DYDX_TESTNET_PRIVATE_KEY in .env.local"
    assert "execution_preflight_blocker=missing_wallet_address;missing_private_key" in str(btc_row["blocker_detail"])


def test_native_focus_repair_report_surfaces_model_gate_failure_detail(tmp_path):
    reports = tmp_path / "reports" / "brain"
    rl_reports = tmp_path / "reports" / "rl"
    ml_reports = tmp_path / "reports" / "ml"
    top_reports = tmp_path / "reports"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    reports.mkdir(parents=True, exist_ok=True)
    rl_reports.mkdir(parents=True, exist_ok=True)
    ml_reports.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "pair_btc_hype_5mins_dydx_candles_derived_history.json").write_text("{}", encoding="utf-8")
    (pair_dir / "pair_sol_hype_5mins_dydx_candles_derived_history.json").write_text("{}", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "candidate_id": "native:BTC-HYPE",
                "selected_history_source": str(pair_dir / "pair_btc_hype_5mins_dydx_candles_derived_history.json"),
                "current_blocker": "",
                "blocker_detail": "",
                "next_repair_action": "candidate_clear_monitor_for_promotion",
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "candidate_id": "native:SOL-HYPE",
                "selected_history_source": str(pair_dir / "pair_sol_hype_5mins_dydx_candles_derived_history.json"),
                "current_blocker": "",
                "blocker_detail": "",
                "next_repair_action": "candidate_clear_monitor_for_promotion",
            },
        ]
    ).to_csv(reports / "native_candidate_diagnosis.csv", index=False)
    pd.DataFrame(
        [{"pair": "BTC-USD-HYPE-USD", "quality_status": "supported"}, {"pair": "SOL-USD-HYPE-USD", "quality_status": "supported"}]
    ).to_csv(reports / "native_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "lane": "native", "paper_credible": True, "promotion_stage": "orchestrator_review", "paper_status": "research_only"},
            {"pair": "SOL-USD-HYPE-USD", "lane": "native", "paper_credible": True, "promotion_stage": "orchestrator_review", "paper_status": "research_only"},
        ]
    ).to_csv(reports / "promotion_ladder.csv", index=False)
    pd.DataFrame([{"gate": "strategy_acceptance", "ready": True, "status": "ready", "blocker": ""}]).to_csv(
        top_reports / "priority_readiness.csv", index=False
    )
    pd.DataFrame([{"status": "research_only", "blocker": "model_gate_not_accepted", "next_action": "improve model gate acceptance before base RL paper handoff"}]).to_csv(
        rl_reports / "base_rl_paper_handoff_status.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "accepted": False,
                "blocker": "model_acceptance_gates_not_met",
                "best_model": "logistic_regression",
                "filtered_profit_factor": 0.14,
                "filtered_sharpe": -0.03,
                "filtered_drawdown": 0.2,
                "take_rate": 0.21,
                "trades": 236,
                "failing_checks": "filtered_profit_factor_min;filtered_sharpe_positive",
                "acceptance_reason": "filtered_profit_factor_min;filtered_sharpe_positive",
            }
        ]
    ).to_csv(ml_reports / "model_gated_acceptance.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "shadow_take": True, "realized_return": 0.05},
            {"pair": "BTC-USD-HYPE-USD", "shadow_take": True, "realized_return": 0.04},
            {"pair": "BTC-USD-HYPE-USD", "shadow_take": True, "realized_return": 0.03},
            {"pair": "BTC-USD-HYPE-USD", "shadow_take": True, "realized_return": 0.02},
            {"pair": "BTC-USD-HYPE-USD", "shadow_take": True, "realized_return": 0.01},
            {"pair": "BTC-USD-HYPE-USD", "shadow_take": True, "realized_return": 0.02},
            {"pair": "BTC-USD-HYPE-USD", "shadow_take": True, "realized_return": 0.03},
            {"pair": "BTC-USD-HYPE-USD", "shadow_take": True, "realized_return": 0.01},
            {"pair": "BTC-USD-HYPE-USD", "shadow_take": True, "realized_return": 0.02},
            {"pair": "BTC-USD-HYPE-USD", "shadow_take": True, "realized_return": -0.01},
            {"pair": "BTC-USD-HYPE-USD", "shadow_take": False, "realized_return": -0.02},
            {"pair": "SOL-USD-HYPE-USD", "shadow_take": False, "realized_return": -0.04},
            {"pair": "SOL-USD-HYPE-USD", "shadow_take": False, "realized_return": 0.02},
        ]
    ).to_csv(ml_reports / "model_walkforward_predictions.csv", index=False)

    result = build_native_focus_repair_report(root=tmp_path)
    focus = pd.read_csv(result.paths["native_focus_repair_report"])
    btc_row = focus.loc[focus["pair"] == "BTC-USD-HYPE-USD"].iloc[0]
    sol_row = focus.loc[focus["pair"] == "SOL-USD-HYPE-USD"].iloc[0]

    assert btc_row["current_blocker"] == "model_gate_not_accepted"
    assert "best_model=logistic_regression" in str(btc_row["blocker_detail"])
    assert "failing_checks=filtered_profit_factor_min;filtered_sharpe_positive" in str(btc_row["blocker_detail"])
    assert btc_row["pair_model_support_status"] == "strong_model_support"
    assert btc_row["next_repair_action"] == "preserve pair-specific model support while improving global model gate acceptance"
    assert sol_row["pair_model_support_status"] == "no_model_support"


def test_native_focus_repair_report_uses_pair_specific_model_support_action_when_support_is_missing(tmp_path):
    reports = tmp_path / "reports" / "brain"
    rl_reports = tmp_path / "reports" / "rl"
    ml_reports = tmp_path / "reports" / "ml"
    top_reports = tmp_path / "reports"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    reports.mkdir(parents=True, exist_ok=True)
    rl_reports.mkdir(parents=True, exist_ok=True)
    ml_reports.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "pair_btc_hype_5mins_dydx_candles_derived_history.json").write_text("{}", encoding="utf-8")
    (pair_dir / "pair_sol_hype_5mins_dydx_candles_derived_history.json").write_text("{}", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "candidate_id": "native:BTC-HYPE",
                "selected_history_source": str(pair_dir / "pair_btc_hype_5mins_dydx_candles_derived_history.json"),
                "current_blocker": "",
                "blocker_detail": "",
                "next_repair_action": "candidate_clear_monitor_for_promotion",
                "evaluation_mode": "native_local_strategy_logic:native_local_math:PROMOTE:spread_return_1:standard",
                "fold_count": 3,
                "nonzero_signal_trade_count": 111,
                "mean_fold_profit_factor": 999.0,
                "mean_fold_sharpe": 0.117806,
                "max_fold_drawdown": 0.010088,
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "candidate_id": "native:SOL-HYPE",
                "selected_history_source": str(pair_dir / "pair_sol_hype_5mins_dydx_candles_derived_history.json"),
                "current_blocker": "",
                "blocker_detail": "",
                "next_repair_action": "candidate_clear_monitor_for_promotion",
                "evaluation_mode": "native_local_strategy_logic:native_local_math:PROMOTE:spread_return_1:standard",
                "fold_count": 3,
                "nonzero_signal_trade_count": 177,
                "mean_fold_profit_factor": 1.54846,
                "mean_fold_sharpe": 0.170547,
                "max_fold_drawdown": 0.06313,
            },
        ]
    ).to_csv(reports / "native_candidate_diagnosis.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "quality_status": "supported"},
            {"pair": "SOL-USD-HYPE-USD", "quality_status": "supported"},
        ]
    ).to_csv(reports / "native_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "candidate_id": "native:BTC-HYPE", "lane": "native", "promotion_stage": "orchestrator_review", "paper_credible": True, "paper_status": "research_only"},
            {"pair": "SOL-USD-HYPE-USD", "candidate_id": "native:SOL-HYPE", "lane": "native", "promotion_stage": "orchestrator_review", "paper_credible": True, "paper_status": "research_only"},
        ]
    ).to_csv(reports / "promotion_ladder.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "candidate_id": "native:BTC-HYPE", "lane": "native", "forward_walk_status": "pass", "blocker_reason": ""},
            {"pair": "SOL-USD-HYPE-USD", "candidate_id": "native:SOL-HYPE", "lane": "native", "forward_walk_status": "pass", "blocker_reason": ""},
        ]
    ).to_csv(reports / "native_forward_walk.csv", index=False)
    pd.DataFrame([{"gate": "strategy_acceptance", "ready": True, "status": "ready", "blocker": ""}]).to_csv(
        top_reports / "priority_readiness.csv", index=False
    )
    pd.DataFrame([{"status": "research_only", "blocker": "model_gate_not_accepted", "next_action": "improve model gate acceptance before base RL paper handoff"}]).to_csv(
        rl_reports / "base_rl_paper_handoff_status.csv", index=False
    )
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "rows": 24, "taken_trades": 0, "take_rate": 0.0, "profit_factor": 0.0, "mean_return": 0.0, "support_status": "no_model_support", "model_gate_anchor_rank": 52, "recommended_repair_action": "wait_for_pair_specific_model_support"},
            {"pair": "SOL-USD-HYPE-USD", "rows": 28, "taken_trades": 0, "take_rate": 0.0, "profit_factor": 0.0, "mean_return": 0.0, "support_status": "no_model_support", "model_gate_anchor_rank": 1, "recommended_repair_action": "wait_for_pair_specific_model_support"},
        ]
    ).to_csv(ml_reports / "model_gate_pair_support_report.csv", index=False)
    pd.DataFrame([{"accepted": False, "best_model": "logistic_regression", "failing_checks": "filtered_profit_factor_min;filtered_sharpe_positive", "blocker": "model_acceptance_gates_not_met"}]).to_csv(
        ml_reports / "model_gated_acceptance.csv", index=False
    )
    pd.DataFrame().to_csv(reports / "pair_detail_quality_report.csv", index=False)

    result = build_native_focus_repair_report(root=tmp_path)
    focus = pd.read_csv(result.paths["native_focus_repair_report"])

    assert set(focus["next_repair_action"]) == {"wait_for_pair_specific_model_support"}


def test_native_focus_repair_report_surfaces_forward_walk_metrics_for_hype_candidates(tmp_path):
    reports = tmp_path / "reports" / "brain"
    rl_reports = tmp_path / "reports" / "rl"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    reports.mkdir(parents=True, exist_ok=True)
    rl_reports.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "pair_btc_hype_5mins_dydx_candles_derived_history.json").write_text("{}", encoding="utf-8")
    (pair_dir / "pair_sol_hype_5mins_dydx_candles_derived_history.json").write_text("{}", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "candidate_id": "native:BTC-HYPE",
                "selected_history_source": str(pair_dir / "pair_btc_hype_5mins_dydx_candles_derived_history.json"),
                "current_blocker": "",
                "blocker_detail": "",
                "next_repair_action": "candidate_clear_monitor_for_promotion",
                "evaluation_mode": "native_local_strategy_logic:native_local_math:PROMOTE:spread_return_1:standard",
                "fold_count": 3,
                "nonzero_signal_trade_count": 111,
                "mean_fold_profit_factor": 999.0,
                "mean_fold_sharpe": 0.117806,
                "max_fold_drawdown": 0.010088,
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "candidate_id": "native:SOL-HYPE",
                "selected_history_source": str(pair_dir / "pair_sol_hype_5mins_dydx_candles_derived_history.json"),
                "current_blocker": "",
                "blocker_detail": "",
                "next_repair_action": "candidate_clear_monitor_for_promotion",
                "evaluation_mode": "native_local_strategy_logic:native_local_math:PROMOTE:spread_return_1:standard",
                "fold_count": 3,
                "nonzero_signal_trade_count": 177,
                "mean_fold_profit_factor": 1.54846,
                "mean_fold_sharpe": 0.170547,
                "max_fold_drawdown": 0.06313,
            },
        ]
    ).to_csv(reports / "native_candidate_diagnosis.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "quality_status": "supported"},
            {"pair": "SOL-USD-HYPE-USD", "quality_status": "supported"},
        ]
    ).to_csv(reports / "native_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "lane": "native", "forward_walk_status": "pass", "blocker_reason": ""},
            {"pair": "SOL-USD-HYPE-USD", "lane": "native", "forward_walk_status": "pass", "blocker_reason": ""},
        ]
    ).to_csv(reports / "native_forward_walk.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "lane": "native", "paper_credible": True, "promotion_stage": "orchestrator_review", "paper_status": "research_only"},
            {"pair": "SOL-USD-HYPE-USD", "lane": "native", "paper_credible": True, "promotion_stage": "orchestrator_review", "paper_status": "research_only"},
        ]
    ).to_csv(reports / "promotion_ladder.csv", index=False)
    pd.DataFrame(
        [{"status": "research_only", "blocker": "paper_execution_not_ready", "next_action": "wire_dydx_submission_path"}]
    ).to_csv(rl_reports / "base_rl_paper_handoff_status.csv", index=False)
    pd.DataFrame([{"gate": "strategy_acceptance", "ready": True, "status": "ready", "blocker": ""}]).to_csv(
        tmp_path / "reports" / "priority_readiness.csv", index=False
    )

    result = build_native_focus_repair_report(root=tmp_path)
    focus = pd.read_csv(result.paths["native_focus_repair_report"])
    btc_row = focus.loc[focus["pair"] == "BTC-USD-HYPE-USD"].iloc[0]
    sol_row = focus.loc[focus["pair"] == "SOL-USD-HYPE-USD"].iloc[0]

    assert str(btc_row["evaluation_mode"]).startswith("native_local_strategy_logic:")
    assert int(btc_row["fold_count"]) == 3
    assert float(btc_row["mean_fold_profit_factor"]) >= 1.0
    assert btc_row["forward_walk_status"] == "pass"
    assert btc_row["promotion_stage"] == "orchestrator_review"
    assert sol_row["forward_walk_status"] == "pass"
    assert float(sol_row["mean_fold_sharpe"]) >= 0.1


def test_native_focus_repair_report_matches_forward_walk_by_candidate_id_when_pair_missing(tmp_path):
    reports = tmp_path / "reports" / "brain"
    rl_reports = tmp_path / "reports" / "rl"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    reports.mkdir(parents=True, exist_ok=True)
    rl_reports.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    history_path = pair_dir / "pair_btc_hype_5mins_dydx_candles_derived_history.json"
    history_path.write_text("{}", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "candidate_id": "native:BTC-HYPE",
                "selected_history_source": str(history_path),
                "current_blocker": "",
                "blocker_detail": "",
                "next_repair_action": "candidate_clear_monitor_for_promotion",
                "evaluation_mode": "native_local_strategy_logic:native_local_math:PROMOTE:spread_return_1:standard",
                "fold_count": 3,
                "nonzero_signal_trade_count": 111,
                "mean_fold_profit_factor": 999.0,
                "mean_fold_sharpe": 0.117806,
                "max_fold_drawdown": 0.010088,
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "candidate_id": "native:SOL-HYPE",
                "selected_history_source": str(pair_dir / "pair_sol_hype_5mins_dydx_candles_derived_history.json"),
                "current_blocker": "",
                "blocker_detail": "",
                "next_repair_action": "candidate_clear_monitor_for_promotion",
            },
        ]
    ).to_csv(reports / "native_candidate_diagnosis.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "quality_status": "supported"},
            {"pair": "SOL-USD-HYPE-USD", "quality_status": "supported"},
        ]
    ).to_csv(reports / "native_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {"candidate_id": "native:BTC-HYPE", "lane": "native", "forward_walk_status": "pass", "blocker_reason": ""},
        ]
    ).to_csv(reports / "native_forward_walk.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "lane": "native", "paper_credible": True, "promotion_stage": "orchestrator_review", "paper_status": "research_only"},
            {"pair": "SOL-USD-HYPE-USD", "lane": "native", "paper_credible": True, "promotion_stage": "orchestrator_review", "paper_status": "research_only"},
        ]
    ).to_csv(reports / "promotion_ladder.csv", index=False)
    pd.DataFrame(
        [{"status": "research_only", "blocker": "paper_execution_not_ready", "next_action": "wire_dydx_submission_path"}]
    ).to_csv(rl_reports / "base_rl_paper_handoff_status.csv", index=False)
    pd.DataFrame([{"gate": "strategy_acceptance", "ready": True, "status": "ready", "blocker": ""}]).to_csv(
        tmp_path / "reports" / "priority_readiness.csv", index=False
    )

    result = build_native_focus_repair_report(root=tmp_path)
    focus = pd.read_csv(result.paths["native_focus_repair_report"])
    btc_row = focus.loc[focus["pair"] == "BTC-USD-HYPE-USD"].iloc[0]

    assert btc_row["forward_walk_status"] == "pass"


def test_candidate_repair_loop_prefers_execution_preflight_action_for_paper_gate(tmp_path):
    reports = tmp_path / "reports" / "brain"
    rl_reports = tmp_path / "reports" / "rl"
    top_reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rl_reports.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "candidate_id": "native:HYPE",
                "lane": "native",
                "pair": "BTC-USD-HYPE-USD",
                "setup_identity": "native:HYPE",
                "setup_role": "primary",
                "promotion_stage": "orchestrator_review",
                "paper_credible": True,
                "blocker": "",
                "paper_status": "research_only",
            }
        ]
    ).to_csv(reports / "promotion_ladder.csv", index=False)
    pd.DataFrame(
        [
            {"candidate_id": "native:HYPE", "pair": "BTC-USD-HYPE-USD", "current_blocker": "", "next_repair_action": "candidate_clear_monitor_for_promotion", "selected_history_source": "hype.json"},
        ]
    ).to_csv(reports / "native_candidate_diagnosis.csv", index=False)
    pd.DataFrame([{"pair": "BTC-USD-HYPE-USD", "combined_score": 0.9}]).to_csv(reports / "native_quality_report.csv", index=False)
    pd.DataFrame(columns=["candidate_id", "current_blocker", "evidence_path"]).to_csv(reports / "wizard_candidate_repair_report.csv", index=False)
    pd.DataFrame(columns=["candidate_id", "review_status", "candidate_status", "pair_recommended_this_hour"]).to_csv(
        reports / "wizard_hourly_repair_targets.csv", index=False
    )
    pd.DataFrame([{"status": "research_only", "blocker": "paper_execution_not_ready", "next_action": "generic_paper_gate_action"}]).to_csv(
        rl_reports / "base_rl_paper_handoff_status.csv", index=False
    )
    pd.DataFrame([{"gate": "strategy_acceptance", "ready": True, "status": "ready", "blocker": ""}]).to_csv(
        top_reports / "priority_readiness.csv", index=False
    )
    pd.DataFrame(
        [
            {"step": "dydx_testnet_dependency", "ready": False, "status": "blocked", "blocker": "missing_wallet_address;missing_private_key", "evidence": "", "next_action": "set DYDX_TESTNET_WALLET_ADDRESS and DYDX_TESTNET_PRIVATE_KEY in .env.local"},
            {"step": "paper_submission_gate", "ready": False, "status": "blocked", "blocker": "strategy_or_dydx_gate_not_ready", "evidence": "", "next_action": "do not submit paper orders until strategy and dYdX dependencies are ready"},
        ]
    ).to_csv(top_reports / "paper_execution_preflight.csv", index=False)

    result = build_candidate_repair_loop(root=tmp_path)
    repair = pd.read_csv(result.paths["candidate_repair_loop"])
    row = repair.loc[repair["candidate_id"] == "native:HYPE"].iloc[0]

    assert row["repair_action"] == "set DYDX_TESTNET_WALLET_ADDRESS and DYDX_TESTNET_PRIVATE_KEY in .env.local"


def test_promotion_readiness_keeps_orchestrator_blocked_until_paper_authorized(tmp_path):
    reports = tmp_path / "reports" / "brain"
    rl_reports = tmp_path / "reports" / "rl"
    reports.mkdir(parents=True, exist_ok=True)
    rl_reports.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "candidate_id": "native:BTC-HYPE",
                "lane": "native",
                "source_type": "local_research",
                "source_path": "reports/source.csv",
                "pair": "BTC-USD-HYPE-USD",
                "venue": "dydx",
                "detection_timestamp": "2026-07-02T12:00:00Z",
                "timeframe": "5MINS",
                "setup_identity": "native:BTC-HYPE",
                "setup_rank": 1,
                "setup_role": "primary",
                "strategy_family": "native_local_math",
                "strategy_mode": "PROMOTE",
                "normalized_feature_bundle_ref": "data/features/bundle.parquet",
                "regime_snapshot": "neutral",
                "confidence": 0.9,
                "backtest_summary_ref": "reports/backtest.csv",
                "forward_walk_summary_ref": "reports/brain/native_forward_walk.csv#native:BTC-HYPE",
                "paper_outcome_ref": "",
                "provenance": "unit_test",
                "schema_version": "phase1_spine_v1",
                "blocker_state": "",
            }
        ]
    ).to_csv(reports / "native_candidate_packets.csv", index=False)
    pd.DataFrame(columns=["candidate_id"]).to_csv(reports / "wizard_candidate_packets.csv", index=False)
    pd.DataFrame(
        [
                {
                    "candidate_id": "native:BTC-HYPE",
                    "lane": "native",
                    "rolling_split_definition": "train=180d,test=30d,step=30d",
                    "oos_sharpe": 1.1,
                    "oos_profit_factor": 1.4,
                    "oos_max_drawdown": -0.05,
                    "oos_trade_count": 25,
                    "stability_metrics": "variance=0.1",
                    "forward_walk_status": "pass",
                    "blocker_reason": "",
                    "dataset_provenance": "data/processed/pairs.csv",
                    "run_manifest_ref": "reports/rl/base_rl_run_manifest.json",
                    "provenance": "unit_test",
                    "schema_version": "phase1_spine_v1",
                }
            ]
        ).to_csv(reports / "native_forward_walk.csv", index=False)
    pd.DataFrame(columns=["candidate_id"]).to_csv(reports / "wizard_forward_walk.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "arbitration_decision": "prefer_native",
            }
        ]
    ).to_csv(reports / "overall_brain_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "status": "research_only",
                "blocker": "strategy_acceptance_not_ready",
                "paper_authorized": False,
            }
        ]
    ).to_csv(rl_reports / "base_rl_paper_handoff_status.csv", index=False)

    result = build_promotion_readiness(root=tmp_path)

    promotion = pd.read_csv(result.paths["promotion_ladder"])
    orchestrator = pd.read_csv(result.paths["orchestrator_status"]).iloc[0]

    assert promotion.iloc[0]["paper_credible"]
    assert promotion.iloc[0]["promotion_stage"] == "orchestrator_review"
    assert not bool(orchestrator["ready"])
    assert orchestrator["status"] == "blocked"
    assert orchestrator["blocker"] == "strategy_acceptance_not_ready"


def test_promotion_readiness_uses_global_paper_gate_blocker_when_strategy_acceptance_is_already_ready(tmp_path):
    reports = tmp_path / "reports" / "brain"
    rl_reports = tmp_path / "reports" / "rl"
    reports.mkdir(parents=True, exist_ok=True)
    rl_reports.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "candidate_id": "native:BTC-HYPE",
                "lane": "native",
                "source_type": "local_research",
                "source_path": "reports/source.csv",
                "pair": "BTC-USD-HYPE-USD",
                "venue": "dydx",
                "detection_timestamp": "2026-07-02T12:00:00Z",
                "timeframe": "5MINS",
                "setup_identity": "native:BTC-HYPE",
                "setup_rank": 1,
                "setup_role": "primary",
                "strategy_family": "native_local_math",
                "strategy_mode": "PROMOTE",
                "normalized_feature_bundle_ref": "data/features/bundle.parquet",
                "regime_snapshot": "neutral",
                "confidence": 0.9,
                "backtest_summary_ref": "reports/backtest.csv",
                "forward_walk_summary_ref": "reports/brain/native_forward_walk.csv#native:BTC-HYPE",
                "paper_outcome_ref": "",
                "provenance": "unit_test",
                "schema_version": "phase1_spine_v1",
                "blocker_state": "",
            }
        ]
    ).to_csv(reports / "native_candidate_packets.csv", index=False)
    pd.DataFrame(columns=["candidate_id"]).to_csv(reports / "wizard_candidate_packets.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "native:BTC-HYPE",
                "lane": "native",
                "rolling_split_definition": "train=180d,test=30d,step=30d",
                "oos_sharpe": 1.1,
                "oos_profit_factor": 1.4,
                "oos_max_drawdown": -0.05,
                "oos_trade_count": 25,
                "stability_metrics": "variance=0.1",
                "forward_walk_status": "pass",
                "blocker_reason": "",
                "dataset_provenance": "data/processed/pairs.csv",
                "run_manifest_ref": "reports/rl/base_rl_run_manifest.json",
                "provenance": "unit_test",
                "schema_version": "phase1_spine_v1",
            }
        ]
    ).to_csv(reports / "native_forward_walk.csv", index=False)
    pd.DataFrame(columns=["candidate_id"]).to_csv(reports / "wizard_forward_walk.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "arbitration_decision": "prefer_native",
            }
        ]
    ).to_csv(reports / "overall_brain_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "status": "research_only",
                "blocker": "paper_execution_not_ready",
                "next_action": "wire_dydx_submission_path",
                "paper_authorized": False,
            }
        ]
    ).to_csv(rl_reports / "base_rl_paper_handoff_status.csv", index=False)
    pd.DataFrame(
        [
            {"gate": "strategy_acceptance", "ready": True, "status": "ready", "blocker": ""},
        ]
    ).to_csv(tmp_path / "reports" / "priority_readiness.csv", index=False)
    pd.DataFrame(
        [
            {"step": "production_eligibility", "ready": True, "status": "ready", "blocker": ""},
        ]
    ).to_csv(tmp_path / "reports" / "strategy_acceptance_checklist.csv", index=False)
    pd.DataFrame(
        [
            {"acceptance_reason": "", "research_next_step": ""},
        ]
    ).to_csv(tmp_path / "reports" / "acceptance_report.csv", index=False)

    result = build_promotion_readiness(root=tmp_path)

    orchestrator = pd.read_csv(result.paths["orchestrator_status"]).iloc[0]

    assert orchestrator["blocker"] == "paper_execution_not_ready"
    assert orchestrator["next_action"] == "wire_dydx_submission_path"


def test_native_forward_walk_derives_spread_returns_and_can_flip_orientation(tmp_path):
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True, exist_ok=True)
    history_path = pair_dir / "pair_test_hype_like.json"
    history = []
    spread = 100.0
    for idx in range(12):
        if idx % 4 == 0:
            z = 2.5
            spread += 2.0
        elif idx % 4 == 2:
            z = -2.5
            spread -= 2.0
        else:
            z = 0.2
            spread += 0.1
        history.append({"timestamp": f"2026-07-01T00:{idx:02d}:00Z", "zscore": z, "spread": spread})
    history_path.write_text(__import__("json").dumps({"history": history}), encoding="utf-8")

    row = pd.Series(
        {
            "selected_path": str(history_path),
            "decision_bucket": "PROMOTE",
            "native_strategy_family": "native_local_math",
        }
    )
    diagnosis, fold_rows = _native_forward_walk_diagnosis(tmp_path, "native:test", row)

    assert fold_rows
    assert "derived_spread_pct_change" in str(diagnosis["evaluation_mode"])
    assert "inverse" in str(diagnosis["evaluation_mode"])
    assert float(diagnosis["mean_fold_profit_factor"]) > 1.0


def test_blocker_persistence_writes_verified_outcome_subset(tmp_path):
    _write_minimal_inputs(tmp_path)
    build_three_brain_system(root=tmp_path)

    wizard_outcomes = pd.read_csv(tmp_path / "reports" / "brain" / "wizard_paper_outcomes.csv")
    wizard_outcomes.loc[:, "verification_status"] = "verified"
    wizard_outcomes.loc[:, "result_status"] = "paper_completed"
    wizard_outcomes.to_csv(tmp_path / "reports" / "brain" / "wizard_paper_outcomes.csv", index=False)

    from quant_platform.three_brain_system import build_blocker_persistence_report

    build_blocker_persistence_report(root=tmp_path)
    blocker = pd.read_csv(tmp_path / "reports" / "brain" / "blocker_persistence_report.csv")
    verified = pd.read_csv(tmp_path / "reports" / "brain" / "verified_paper_outcome_report.csv")

    wizard_rows = blocker.loc[blocker["lane"] == "wizard"]
    assert wizard_rows["verified_outcome"].astype(bool).any()
    assert not verified.empty
    assert set(verified["lane"]) == {"wizard"}
    assert set(verified["paper_outcome_status"]) == {"paper_completed"}


def test_shadow_rl_usefulness_distinguishes_triage_and_comparison(tmp_path):
    _write_minimal_inputs(tmp_path)
    build_three_brain_system(root=tmp_path)

    shadow = pd.read_csv(tmp_path / "reports" / "brain" / "shadow_rl_usefulness_report.csv")
    candidate_set = pd.read_csv(tmp_path / "reports" / "brain" / "shadow_rl_candidate_set.csv")

    assert "setup_identity" in shadow.columns
    assert {"freshness_status", "evidence_quality_status", "next_rl_action"}.issubset(shadow.columns)
    assert not candidate_set.empty
    assert {"rl_priority_rank", "candidate_id", "next_rl_action"}.issubset(candidate_set.columns)
    assert set(shadow["usefulness_status"]).issubset(
        {"aligned_signal", "fresh_repair_learning", "unsupported_candidate", "triage_only", "comparison_only"}
    )


def test_shadow_rl_candidate_set_prefers_promotable_training_candidates(tmp_path):
    reports = tmp_path / "reports" / "brain"
    reports.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "candidate_id": "wizard:DOGE",
                "lane": "wizard",
                "pair": "DOGE-USD-MORPHO-USD",
                "setup_identity": "DOGE",
                "recommended_action": "inspect",
                "shadow_only": True,
                "paper_credible_overlap": False,
                "promotion_stage": "forward_walk_blocked",
                "freshness_status": "fresh_repaired_candidate",
                "evidence_quality_status": "supported",
                "usefulness_status": "fresh_repair_learning",
                "next_rl_action": "promote_into_shadow_rl_training_set",
                "blocker": "local_verification_not_run",
                "evidence_path": "wizard.csv",
            },
            {
                "candidate_id": "native:BTC",
                "lane": "native",
                "pair": "BTC-USD-HYPE-USD",
                "setup_identity": "BTC",
                "recommended_action": "accept",
                "shadow_only": True,
                "paper_credible_overlap": False,
                "promotion_stage": "orchestrator_review",
                "freshness_status": "upgraded_native_candidate",
                "evidence_quality_status": "supported",
                "usefulness_status": "aligned_signal",
                "next_rl_action": "promote_into_shadow_rl_training_set",
                "blocker": "",
                "evidence_path": "native.csv",
            },
            {
                "candidate_id": "wizard:STALE",
                "lane": "wizard",
                "pair": "ETH-USD-MORPHO-USD",
                "setup_identity": "ETH",
                "recommended_action": "inspect",
                "shadow_only": True,
                "paper_credible_overlap": False,
                "promotion_stage": "forward_walk_blocked",
                "freshness_status": "stale_blocked_candidate",
                "evidence_quality_status": "needs_capture_completion",
                "usefulness_status": "triage_only",
                "next_rl_action": "keep_in_shadow_triage",
                "blocker": "missing_exact_mode",
                "evidence_path": "stale.csv",
            },
        ]
    ).to_csv(reports / "shadow_rl_usefulness_report.csv", index=False)

    build_shadow_rl_candidate_set(root=tmp_path)
    candidate_set = pd.read_csv(reports / "shadow_rl_candidate_set.csv")

    assert set(candidate_set["candidate_id"]) == {"wizard:DOGE", "native:BTC"}
    assert set(candidate_set["next_rl_action"]) == {"promote_into_shadow_rl_training_set"}


def test_shadow_rl_usefulness_does_not_treat_hourly_matched_but_blocked_wizard_setup_as_supported(tmp_path):
    reports = tmp_path / "reports" / "brain"
    reports.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "candidate_id": "wizard:DOGE",
                "lane": "wizard",
                "pair": "DOGE-USD-MORPHO-USD",
                "setup_identity": "DOGE|MORPHO",
                "recommended_action": "inspect",
                "shadow_only": True,
                "evidence_path": "wizard.csv",
            }
        ]
    ).to_csv(reports / "lane_shadow_rl.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "wizard:DOGE",
                "paper_credible": False,
                "promotion_stage": "forward_walk_blocked",
                "blocker": "weak_correlation;parity_not_replicated",
            }
        ]
    ).to_csv(reports / "promotion_ladder.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "wizard:DOGE",
                "hourly_match_status": "matched",
                "review_status": "new",
            }
        ]
    ).to_csv(reports / "wizard_hourly_repair_targets.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "wizard:DOGE",
                "current_blocker": "weak_correlation;parity_not_replicated",
                "exact_mode_capture_status": "captured",
                "local_verification_status": "verified",
            }
        ]
    ).to_csv(reports / "wizard_candidate_repair_report.csv", index=False)
    pd.DataFrame(columns=["pair", "current_history_matches_focused_5m"]).to_csv(reports / "native_focus_repair_report.csv", index=False)

    build_shadow_rl_usefulness_report(root=tmp_path)
    shadow = pd.read_csv(reports / "shadow_rl_usefulness_report.csv")

    assert list(shadow["evidence_quality_status"]) == ["quality_blocked_candidate"]
    assert list(shadow["usefulness_status"]) == ["comparison_only"]
    assert list(shadow["next_rl_action"]) == ["keep_in_comparison_pool"]


def test_shadow_rl_usefulness_requires_pair_specific_native_model_support_for_supported_status(tmp_path):
    reports = tmp_path / "reports" / "brain"
    reports.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "candidate_id": "native:BTC",
                "lane": "native",
                "pair": "BTC-USD-HYPE-USD",
                "setup_identity": "native:BTC",
                "recommended_action": "accept",
                "shadow_only": True,
                "evidence_path": "btc.csv",
            },
            {
                "candidate_id": "native:SOL",
                "lane": "native",
                "pair": "SOL-USD-HYPE-USD",
                "setup_identity": "native:SOL",
                "recommended_action": "accept",
                "shadow_only": True,
                "evidence_path": "sol.csv",
            },
        ]
    ).to_csv(reports / "lane_shadow_rl.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "native:BTC",
                "paper_credible": True,
                "promotion_stage": "orchestrator_review",
                "blocker": "model_gate_not_accepted",
            },
            {
                "candidate_id": "native:SOL",
                "paper_credible": True,
                "promotion_stage": "orchestrator_review",
                "blocker": "model_gate_not_accepted",
            },
        ]
    ).to_csv(reports / "promotion_ladder.csv", index=False)
    pd.DataFrame(columns=["candidate_id"]).to_csv(reports / "wizard_hourly_repair_targets.csv", index=False)
    pd.DataFrame(columns=["candidate_id"]).to_csv(reports / "wizard_candidate_repair_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "current_history_matches_focused_5m": True,
                "pair_model_support_status": "strong_model_support",
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "current_history_matches_focused_5m": True,
                "pair_model_support_status": "no_model_support",
            },
        ]
    ).to_csv(reports / "native_focus_repair_report.csv", index=False)

    build_shadow_rl_usefulness_report(root=tmp_path)
    shadow = pd.read_csv(reports / "shadow_rl_usefulness_report.csv")

    btc = shadow.loc[shadow["candidate_id"] == "native:BTC"].iloc[0]
    sol = shadow.loc[shadow["candidate_id"] == "native:SOL"].iloc[0]

    assert btc["evidence_quality_status"] == "supported"
    assert btc["next_rl_action"] == "promote_into_shadow_rl_training_set"
    assert sol["evidence_quality_status"] == "unsupported_candidate"
    assert sol["usefulness_status"] == "unsupported_candidate"
    assert sol["next_rl_action"] == "wait_for_repair_evidence"


def test_shadow_rl_candidate_set_prefers_primary_wizard_setup_per_pair(tmp_path):
    reports = tmp_path / "reports" / "brain"
    reports.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "candidate_id": "wizard:DOGE:primary",
                "lane": "wizard",
                "pair": "DOGE-USD-EIGEN-USD",
                "setup_identity": "DOGE|primary",
                "recommended_action": "inspect",
                "shadow_only": True,
                "paper_credible_overlap": False,
                "promotion_stage": "forward_walk_blocked",
                "freshness_status": "fresh_repaired_candidate",
                "evidence_quality_status": "supported",
                "usefulness_status": "fresh_repair_learning",
                "next_rl_action": "promote_into_shadow_rl_training_set",
                "blocker": "wizard_fold_metrics_weak",
                "evidence_path": "wizard_primary.csv",
            },
            {
                "candidate_id": "wizard:DOGE:alternate",
                "lane": "wizard",
                "pair": "DOGE-USD-EIGEN-USD",
                "setup_identity": "DOGE|alternate",
                "recommended_action": "inspect",
                "shadow_only": True,
                "paper_credible_overlap": False,
                "promotion_stage": "forward_walk_blocked",
                "freshness_status": "fresh_repaired_candidate",
                "evidence_quality_status": "supported",
                "usefulness_status": "fresh_repair_learning",
                "next_rl_action": "promote_into_shadow_rl_training_set",
                "blocker": "wizard_fold_metrics_weak",
                "evidence_path": "wizard_alternate.csv",
            },
        ]
    ).to_csv(reports / "shadow_rl_usefulness_report.csv", index=False)
    pd.DataFrame(
        [
            {"candidate_id": "wizard:DOGE:primary", "pair": "DOGE-USD-EIGEN-USD", "setup_role": "primary"},
            {"candidate_id": "wizard:DOGE:alternate", "pair": "DOGE-USD-EIGEN-USD", "setup_role": "alternate"},
        ]
    ).to_csv(reports / "wizard_candidate_packets.csv", index=False)

    build_shadow_rl_candidate_set(root=tmp_path)
    candidate_set = pd.read_csv(reports / "shadow_rl_candidate_set.csv")

    assert list(candidate_set["candidate_id"]) == ["wizard:DOGE:primary"]


def test_shadow_rl_intake_report_mirrors_authoritative_candidate_set(tmp_path):
    reports = tmp_path / "reports" / "brain"
    reports.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "rl_priority_rank": 1,
                "candidate_id": "wizard:DOGE",
                "lane": "wizard",
                "pair": "DOGE-USD-MORPHO-USD",
                "setup_identity": "DOGE",
                "freshness_status": "fresh_repaired_candidate",
                "evidence_quality_status": "supported",
                "usefulness_status": "fresh_repair_learning",
                "next_rl_action": "promote_into_shadow_rl_training_set",
            }
        ]
    ).to_csv(reports / "shadow_rl_candidate_set.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "wizard:DOGE",
                "lane": "wizard",
                "pair": "DOGE-USD-MORPHO-USD",
                "setup_identity": "DOGE",
                "recommended_action": "inspect",
                "shadow_only": True,
                "paper_credible_overlap": False,
                "promotion_stage": "forward_walk_blocked",
                "freshness_status": "fresh_repaired_candidate",
                "evidence_quality_status": "supported",
                "usefulness_status": "fresh_repair_learning",
                "next_rl_action": "promote_into_shadow_rl_training_set",
                "blocker": "local_verification_not_run",
                "evidence_path": "wizard.csv",
            }
        ]
    ).to_csv(reports / "shadow_rl_usefulness_report.csv", index=False)

    build_shadow_rl_intake_report(root=tmp_path)
    frame = pd.read_csv(reports / "shadow_rl_intake_report.csv")

    assert list(frame["candidate_id"]) == ["wizard:DOGE"]
    assert list(frame["intake_status"]) == ["authoritative_shadow_training_candidate"]
    assert list(frame["blocker"]) == ["local_verification_not_run"]


def test_shadow_rl_candidate_set_excludes_unsupported_candidates(tmp_path):
    reports = tmp_path / "reports" / "brain"
    reports.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "candidate_id": "wizard:DOGE",
                "lane": "wizard",
                "pair": "DOGE-USD-MORPHO-USD",
                "setup_identity": "DOGE",
                "recommended_action": "inspect",
                "shadow_only": True,
                "paper_credible_overlap": False,
                "promotion_stage": "forward_walk_blocked",
                "freshness_status": "fresh_repaired_candidate",
                "evidence_quality_status": "supported",
                "usefulness_status": "fresh_repair_learning",
                "next_rl_action": "promote_into_shadow_rl_training_set",
                "blocker": "local_verification_not_run",
                "evidence_path": "wizard.csv",
            },
            {
                "candidate_id": "native:SOL",
                "lane": "native",
                "pair": "SOL-USD-HYPE-USD",
                "setup_identity": "SOL",
                "recommended_action": "accept",
                "shadow_only": True,
                "paper_credible_overlap": False,
                "promotion_stage": "orchestrator_review",
                "freshness_status": "upgraded_native_candidate",
                "evidence_quality_status": "unsupported_candidate",
                "usefulness_status": "unsupported_candidate",
                "next_rl_action": "wait_for_repair_evidence",
                "blocker": "model_gate_not_accepted",
                "evidence_path": "native.csv",
            },
        ]
    ).to_csv(reports / "shadow_rl_usefulness_report.csv", index=False)
    pd.DataFrame(columns=["candidate_id", "pair", "setup_role"]).to_csv(reports / "wizard_candidate_packets.csv", index=False)

    build_shadow_rl_candidate_set(root=tmp_path)
    candidate_set = pd.read_csv(reports / "shadow_rl_candidate_set.csv")

    assert list(candidate_set["candidate_id"]) == ["wizard:DOGE"]


def test_shadow_rl_intake_report_labels_comparison_candidates_honestly(tmp_path):
    reports = tmp_path / "reports" / "brain"
    reports.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "rl_priority_rank": 1,
                "candidate_id": "wizard:DOGE",
                "lane": "wizard",
                "pair": "DOGE-USD-MORPHO-USD",
                "setup_identity": "DOGE",
                "freshness_status": "fresh_repaired_candidate",
                "evidence_quality_status": "quality_blocked_candidate",
                "usefulness_status": "comparison_only",
                "next_rl_action": "keep_in_comparison_pool",
            }
        ]
    ).to_csv(reports / "shadow_rl_candidate_set.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "wizard:DOGE",
                "lane": "wizard",
                "pair": "DOGE-USD-MORPHO-USD",
                "setup_identity": "DOGE",
                "recommended_action": "inspect",
                "shadow_only": True,
                "paper_credible_overlap": False,
                "promotion_stage": "forward_walk_blocked",
                "freshness_status": "fresh_repaired_candidate",
                "evidence_quality_status": "quality_blocked_candidate",
                "usefulness_status": "comparison_only",
                "next_rl_action": "keep_in_comparison_pool",
                "blocker": "wizard_fold_metrics_weak",
                "evidence_path": "wizard.csv",
            }
        ]
    ).to_csv(reports / "shadow_rl_usefulness_report.csv", index=False)

    build_shadow_rl_intake_report(root=tmp_path)
    frame = pd.read_csv(reports / "shadow_rl_intake_report.csv")

    assert frame.iloc[0]["intake_status"] == "comparison_only_shadow_candidate"


def test_wizard_hourly_priority_queue_prefers_true_capture_gaps_over_verification_ready_rows(tmp_path):
    reports = tmp_path / "reports" / "brain"
    active = tmp_path / "reports" / "active"
    reports.mkdir(parents=True, exist_ok=True)
    active.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
                {
                    "pair": "DOGE-USD-MORPHO-USD",
                    "candidate_id": "wizard:DOGE",
                    "setup_identity": "DOGE",
                    "exact_mode_capture_status": "captured",
                    "current_blocker": "sharpe_below_1.75;weak_correlation;parity_not_replicated",
                    "setup_role": "primary",
                },
            {
                "pair": "DOGE-USD-EIGEN-USD",
                "candidate_id": "wizard:EIGEN",
                "setup_identity": "EIGEN",
                "exact_mode_capture_status": "missing",
                "current_blocker": "missing_exact_mode",
                "setup_role": "primary",
            },
        ]
    ).to_csv(reports / "wizard_candidate_repair_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "wizard:DOGE",
                "hourly_match_status": "matched",
                "review_status": "new",
                "candidate_status": "review_now",
                "pair_recommended_this_hour": True,
            },
            {
                "candidate_id": "wizard:EIGEN",
                "hourly_match_status": "missing",
                "review_status": "",
                "candidate_status": "",
                "pair_recommended_this_hour": False,
            },
        ]
    ).to_csv(reports / "wizard_hourly_repair_targets.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "DOGE-USD/MORPHO-USD", "priority_rank": 0},
            {"pair": "DOGE-USD/EIGEN-USD", "priority_rank": 0},
        ]
    ).to_csv(active / "wizard_exact_mode_capture_queue.csv", index=False)
    pd.DataFrame([{"pair": "DOGE-USD/MORPHO-USD", "setup_identity": "DOGE", "verification_status": "verified"}]).to_csv(
        active / "wizard_local_verification_batch.csv", index=False
    )

    result = build_wizard_hourly_priority_capture_queue(root=tmp_path)
    queue = pd.read_csv(result.paths["wizard_hourly_priority_capture_queue"])

    stale_row = queue.loc[queue["candidate_id"] == "wizard:EIGEN"].iloc[0]
    assert stale_row["live_capture_status"] == "live_expired"
    assert stale_row["capture_completeness_status"] == "live_expired"
    assert stale_row["recommended_capture_action"] == "skip_stale_pair_take_next_live_candidate"
    doge_row = queue.loc[queue["candidate_id"] == "wizard:DOGE"].iloc[0]
    assert doge_row["capture_completeness_status"] == "verification_ready"
    assert doge_row["recommended_capture_action"] == "no_capture_needed_use_repair_loop"
    assert stale_row["capture_priority_rank"] > doge_row["capture_priority_rank"]
