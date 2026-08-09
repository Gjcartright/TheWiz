from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import pytest

from quant_platform import active_pipeline as active_pipeline_module

from quant_platform.active_pipeline import (
    _canonical_hyperliquid_state_rows,
    _command_center_markdown,
    _configured_env_key_present,
    _data_health_rows,
    _venue_route_state_row,
    _venue_lane_test_rows,
    _wizard_pair_metrics,
    archive_from_index,
    build_artifact_index,
    build_command_dashboard,
    build_market_venue_context,
    build_multi_venue_history_readiness,
    build_pair_universe,
    build_trade_dataset,
    build_venue_lane_test_plan,
    build_venue_route_scorecard,
    export_trade_gate_model,
    focused_paper_validation_rows,
    model_gate_pair_support_report,
    paper_candidate_shortlist_rows,
    _filter_shortlist_to_visible_dydx_markets,
    system_check,
)


def test_canonical_current_state_uses_decision_and_audit_truth(tmp_path):
    council = tmp_path / "reports" / "orchestration" / "teacher_council"
    council.mkdir(parents=True)
    pd.DataFrame([{"selection_status": "BLOCKED"}]).to_csv(council / "statistical_selection_controls.csv", index=False)
    pd.DataFrame([{"status": "BLOCKED", "action": "abstain"}]).to_csv(council / "council_decisions.csv", index=False)
    pd.DataFrame([{"status": "PASS"}, {"status": "BLOCKED"}]).to_csv(council / "student_training_readiness.csv", index=False)
    pd.DataFrame([{"verdict": "veto"}]).to_csv(council / "portfolio_critic.csv", index=False)
    authority = pd.Series(
        {
            "run_id": "run-1",
            "candidate_set_id": "set-1",
            "status": "RESEARCH_ONLY",
            "research_ready": True,
            "paper_ready": False,
            "execution_allowed": False,
            "blocker": "evidence_not_ready",
        }
    )

    rows = {row["area"]: row for row in _canonical_hyperliquid_state_rows(tmp_path, authority)}

    assert rows["teacher_council"]["ready"] is False
    assert rows["student_learning"]["ready"] is False
    assert rows["portfolio_critic"]["ready"] is False
    assert rows["hyperliquid_walkforward"]["ready"] is False


def test_canonical_current_state_scopes_bandit_blockers_away_from_supervised_student(tmp_path):
    council = tmp_path / "reports" / "orchestration" / "teacher_council"
    council.mkdir(parents=True)
    pd.DataFrame([{"selection_status": "BLOCKED"}]).to_csv(council / "statistical_selection_controls.csv", index=False)
    pd.DataFrame([{"status": "BLOCKED", "action": "abstain"}]).to_csv(council / "council_decisions.csv", index=False)
    pd.DataFrame(
        [
            {"scope": "all_learning", "status": "PASS"},
            {"scope": "supervised_student", "status": "PASS"},
            {"scope": "contextual_bandit", "status": "BLOCKED"},
        ]
    ).to_csv(council / "student_training_readiness.csv", index=False)
    pd.DataFrame([{"verdict": "veto"}]).to_csv(council / "portfolio_critic.csv", index=False)
    authority = pd.Series(
        {
            "run_id": "run-1",
            "candidate_set_id": "set-1",
            "status": "RESEARCH_ONLY",
            "execution_allowed": False,
        }
    )

    rows = {row["area"]: row for row in _canonical_hyperliquid_state_rows(tmp_path, authority)}

    assert rows["student_learning"]["ready"] is True


def test_artifact_index_classifies_without_moving_files():
    result = build_artifact_index()

    frame = pd.read_csv(result.paths["artifact_index"])

    assert not frame.empty
    assert {"active", "historical_evidence", "do_not_move"}.issubset(set(frame["status"]))
    assert Path("src/quant_platform/cli.py").as_posix() in set(frame["path"])
    assert Path("README.md").as_posix() in set(frame["path"])


def test_pair_universe_separates_discovery_from_acceptance_and_blocks_vendor_only_promote():
    result = build_pair_universe()

    frame = pd.read_csv(result.paths["pair_universe"])

    assert not frame.empty
    assert {"discovery_score", "acceptance_score", "decision_bucket", "decision_reason", "evidence_path"}.issubset(frame.columns)
    promoted = frame[frame["decision_bucket"] == "PROMOTE"]
    if not promoted.empty:
        assert (promoted["acceptance_score"] >= 70).all()
    assert not frame["decision_reason"].astype(str).str.contains("dashboard_only_promote", case=False).any()


def test_wizard_pair_metrics_matches_normalized_legs_and_excludes_stale_score_inputs():
    tables = {
        "evidence": pd.DataFrame(
            [
                {
                    "pair": "ETHUSDT/FIDAUSDT",
                    "asset_x": "ETHUSDT",
                    "asset_y": "FIDAUSDT",
                    "exchange": "binance",
                    "exact_mode": "Copula",
                    "mode_valid": True,
                    "sharpe": 9.0,
                    "returns_total": 0.9,
                    "source_authority": "discovery_only",
                    "source_timestamp": "2026-07-01T00:00:00Z",
                    "source_fresh": False,
                    "source_health": "healthy",
                },
                {
                    "pair": "ETHUSDT/FIDAUSDT",
                    "asset_x": "ETHUSDT",
                    "asset_y": "FIDAUSDT",
                    "exchange": "binance",
                    "exact_mode": "Copula",
                    "mode_valid": True,
                    "sharpe": 2.4,
                    "returns_total": 0.3,
                    "source_authority": "discovery_only",
                    "source_timestamp": "2026-08-05T16:00:00Z",
                    "source_fresh": True,
                    "source_health": "healthy",
                    "stationarity_status": "engle_granger",
                    "engle_granger_cointegrated": True,
                    "engle_granger_trend": False,
                    "johansen_cointegrated": False,
                    "zscore_last": -2.1,
                    "zscore_roll_last": -1.2,
                    "volume_min": 185_000,
                },
            ]
        ),
        "diagnostics": pd.DataFrame(
            [
                {
                    "pair": "ETHUSDT/FIDAUSDT",
                    "asset_x": "ETHUSDT",
                    "asset_y": "FIDAUSDT",
                    "exact_mode": "Copula",
                    "wizard_diagnostic_score": 42.0,
                    "source_fresh": False,
                    "source_health": "healthy",
                },
                {
                    "pair": "ETHUSDT/FIDAUSDT",
                    "asset_x": "ETHUSDT",
                    "asset_y": "FIDAUSDT",
                    "exact_mode": "Copula",
                    "wizard_diagnostic_score": 12.0,
                    "source_fresh": True,
                    "source_health": "healthy",
                },
            ]
        ),
        "hypotheses": pd.DataFrame(
            [
                {
                    "pair": "ETHUSDT/FIDAUSDT",
                    "asset_x": "ETHUSDT",
                    "asset_y": "FIDAUSDT",
                    "exact_mode": "Copula",
                    "hypothesis_status": "DISCOVERY_ONLY",
                    "source_fresh": True,
                    "source_health": "healthy",
                }
            ]
        ),
        "parity": pd.DataFrame(),
    }

    metrics = _wizard_pair_metrics("ETH-USD-FIDA-USD", tables)

    assert metrics["best_wizard_sharpe"] == 2.4
    assert metrics["best_wizard_exchange"] == "binance"
    assert metrics["wizard_evidence_state"] == "DISCOVERY_ONLY"
    assert metrics["zscore_score"] == 7.2
    assert metrics["wizard_diagnostic_score"] == 12.0
    assert metrics["cointegration_score"] == 4.0
    assert metrics["wizard_hypothesis_status"] == "DISCOVERY_ONLY"
    assert metrics["wizard_zscore_last"] == -2.1


def test_pair_universe_routes_fresh_wizard_candidate_to_its_research_venue(tmp_path):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT/FIDAUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "FIDAUSDT",
                "exchange": "binance",
                "interval": "daily",
                "exact_mode": "Copula",
                "mode_valid": True,
                "sharpe": 2.4,
                "returns_total": 0.3,
                "source_authority": "discovery_only",
                "source_timestamp": "2026-08-05T16:00:00Z",
                "source_fresh": True,
                "source_health": "healthy",
                "stationarity_status": "engle_granger",
            }
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)

    result = build_pair_universe(root=tmp_path)
    frame = pd.read_csv(result.paths["pair_universe"])
    row = frame.iloc[0]

    assert row["pair"] == "ETHUSDT-FIDAUSDT"
    assert row["exchange"] == "binance"
    assert row["best_wizard_exchange"] == "binance"
    assert row["field_freshness"] == "fresh_wizard_discovery"
    assert row["wizard_evidence_state"] == "DISCOVERY_ONLY"
    assert row["decision_bucket"] == "FETCH_MORE_DATA"
    assert "wizard_discovery_only_needs_pair_detail_and_local_replay" in row["missing_data_reason"]


def test_market_venue_context_keeps_sources_authority_aware():
    result = build_market_venue_context()

    frame = pd.read_csv(result.paths["market_venue_context"])
    lanes = pd.read_csv(result.paths["venue_lanes"])

    assert not frame.empty
    assert {
        "asset",
        "venue",
        "source_system",
        "execution_authority",
        "promotion_allowed",
        "venue_lane",
        "funding_pulse_status",
        "blocker",
        "evidence_path",
    }.issubset(frame.columns)
    assert "funding_pulse_needs_api_key" in set(frame["blocker"].astype(str))
    context_only = frame[frame["source_system"].isin(["coinglass", "gmx", "dexscreener", "funding_pulse"])]
    assert not context_only["promotion_allowed"].astype(bool).any()
    hyperliquid = frame[frame["venue"].astype(str).str.lower() == "hyperliquid"]
    if not hyperliquid.empty:
        historical_hyperliquid = hyperliquid[hyperliquid["source_system"].astype(str) != "hyperliquid_public_api"]
        if not historical_hyperliquid.empty:
            assert historical_hyperliquid["blocker"].astype(str).str.contains("missing_hyperliquid_local_replay").all()
        public_hyperliquid = hyperliquid[hyperliquid["source_system"].astype(str) == "hyperliquid_public_api"]
        if not public_hyperliquid.empty:
            assert public_hyperliquid["blocker"].astype(str).str.contains("requires_pair_history_cost_slippage_and_preflight").all()
            assert not public_hyperliquid["promotion_allowed"].astype(bool).any()
    assert not lanes.empty
    assert {"asset", "best_lane", "next_action"}.issubset(lanes.columns)


def test_venue_lane_test_plan_routes_hyperliquid_without_promoting(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {"asset": "AAA", "best_lane": "hyperliquid_research_candidate", "blockers": "", "next_action": "collect_history"},
            {"asset": "BBB", "best_lane": "hyperliquid_research_candidate", "blockers": "", "next_action": "collect_history"},
        ]
    ).to_csv(active / "venue_lane_classification.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "AAA-USD/BBB-USD",
                "asset_x": "AAA",
                "asset_y": "BBB",
                "exchange": "hyperliquid",
                "exact_mode": "OU (Spread)",
                "mode_valid": True,
                "source_fresh": True,
                "passes_sharpe_gate": True,
                "sharpe": 2.1,
                "returns_total": 0.12,
                "discovery_min_returns_total": 0.10,
                "source_system": "crypto_wizards_live_scanner_capture",
                "source_authority": "discovery_only",
                "evidence_path": "reports/active/wizard.csv",
            }
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)
    result = build_venue_lane_test_plan(root=tmp_path)

    frame = pd.read_csv(result.paths["venue_lane_test_plan"])

    assert not frame.empty
    assert {"pair_lane", "test_status", "funding_pulse_status", "next_step", "evidence_path"}.issubset(frame.columns)
    assert set(frame["funding_pulse_status"].dropna().unique()) == {"needs_api_key"}
    hyperliquid_rows = frame[frame["pair_lane"].astype(str).str.contains("hyperliquid", na=False)]
    if not hyperliquid_rows.empty:
        assert hyperliquid_rows["test_status"].astype(str).str.contains("hyperliquid").all()
        assert not hyperliquid_rows["next_step"].astype(str).str.contains("promote", case=False, na=False).any()


def test_venue_route_scorecard_requires_current_complete_two_leg_evidence(tmp_path):
    processed = tmp_path / "data" / "processed"
    active = tmp_path / "reports" / "active"
    processed.mkdir(parents=True)
    active.mkdir(parents=True)
    timestamp = "2026-08-05T12:00:00+00:00"
    pair = "BTC-USD-ETH-USD"

    pd.DataFrame(
        [
            {
                "pair": pair,
                "asset_x": "BTC-USD",
                "asset_y": "ETH-USD",
                "combined_score": 91.0,
                "available_venues": "dydx;hyperliquid",
                "funding_drag_bps": 1.1,
                "field_freshness": "current_snapshot",
            }
        ]
    ).to_csv(processed / "pair_universe.csv", index=False)
    context_rows = []
    for venue in ("dydx", "hyperliquid"):
        for asset in ("BTC", "ETH"):
            context_rows.append(
                {
                    "asset": asset,
                    "venue": venue,
                    "tradable": True,
                    "execution_authority": True,
                    "source_timestamp": timestamp,
                    "volume_24h": 2_000_000.0,
                    "open_interest_usd": 10_000_000.0,
                    "funding_rate": "0.0001",
                    "evidence_path": f"reports/active/{venue}_snapshot.csv",
                }
            )
    pd.DataFrame(context_rows).to_csv(processed / "market_venue_context.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": pair,
                "venue": "dydx",
                "execution_ready": True,
                "ready_for_submission": False,
                "cost_model_aligned": True,
                "cost_model_profile": "dydx_perpetual_cost_model",
                "evidence": "reports/paper_venue_preflight.csv",
            },
            {
                "pair": pair,
                "venue": "hyperliquid",
                "execution_ready": True,
                "ready_for_submission": False,
                "cost_model_aligned": False,
                "cost_model_profile": "missing_pair_cost_model",
                "evidence": "reports/paper_venue_preflight.csv",
            },
        ]
    ).to_csv(tmp_path / "reports" / "paper_venue_preflight.csv", index=False)
    pd.DataFrame(
        [
            {"pair": pair, "asset_x": "BTC-USD", "asset_y": "ETH-USD", "wizard_exchange": "dydx", "test_status": "dydx_replayed"},
            {"pair": pair, "asset_x": "BTC-USD", "asset_y": "ETH-USD", "wizard_exchange": "hyperliquid", "test_status": "hyperliquid_history_needed"},
        ]
    ).to_csv(active / "venue_lane_test_plan.csv", index=False)
    pd.DataFrame().to_csv(active / "multi_venue_history_readiness_2026-06-25.csv", index=False)

    result = build_venue_route_scorecard(root=tmp_path, as_of=pd.Timestamp("2026-08-05T12:30:00+00:00"))
    scorecard = pd.read_csv(result.paths["venue_route_scorecard"])
    recommendations = pd.read_csv(result.paths["venue_route_recommendations"])
    by_venue = scorecard.set_index("venue")

    assert bool(by_venue.loc["dydx", "validation_route_ready"]) is True
    assert bool(by_venue.loc["hyperliquid", "research_route_ready"]) is True
    assert bool(by_venue.loc["hyperliquid", "validation_route_ready"]) is False
    assert "missing_pair_cost_model" in by_venue.loc["hyperliquid", "blockers"]
    assert recommendations.loc[0, "recommended_execution_venue"] == "dydx"
    assert pd.isna(recommendations.loc[0, "recommended_paper_venue"])

    stale = build_venue_route_scorecard(root=tmp_path, as_of=pd.Timestamp("2026-08-07T12:30:00+00:00"))
    stale_recommendations = pd.read_csv(stale.paths["venue_route_recommendations"])
    assert pd.isna(stale_recommendations.loc[0, "recommended_execution_venue"])
    assert "stale_or_missing_venue_context" in stale_recommendations.loc[0, "blockers"]


def test_venue_route_scorecard_credits_fresh_hyperliquid_bundle_but_not_missing_costs(tmp_path):
    processed = tmp_path / "data" / "processed"
    active = tmp_path / "reports" / "active"
    processed.mkdir(parents=True)
    active.mkdir(parents=True)
    pair = "BTC-USD-ETH-USD"
    timestamp = "2026-08-05T12:00:00+00:00"
    pd.DataFrame(
        [
            {
                "pair": pair,
                "asset_x": "BTC-USD",
                "asset_y": "ETH-USD",
                "combined_score": 75.0,
                "available_venues": "hyperliquid",
            }
        ]
    ).to_csv(processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "asset": asset,
                "venue": "hyperliquid",
                "tradable": True,
                "execution_authority": True,
                "source_timestamp": timestamp,
                "volume_24h": 2_000_000.0,
                "open_interest_usd": 10_000_000.0,
                "funding_rate": "0.0001",
                "evidence_path": "data/raw/hyperliquid_market_snapshots/snapshot.json",
            }
            for asset in ("BTC", "ETH")
        ]
    ).to_csv(processed / "market_venue_context.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": pair,
                "venue": "hyperliquid",
                "execution_ready": True,
                "ready_for_submission": False,
                "cost_model_aligned": False,
                "cost_model_profile": "missing_pair_cost_model",
                "evidence": "reports/paper_venue_preflight.csv",
            }
        ]
    ).to_csv(tmp_path / "reports" / "paper_venue_preflight.csv", index=False)
    pd.DataFrame().to_csv(active / "venue_lane_test_plan.csv", index=False)
    pd.DataFrame().to_csv(active / "multi_venue_history_readiness_2026-06-25.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": pair,
                "asset_x": "BTC",
                "asset_y": "ETH",
                "venue": "hyperliquid",
                "history_ready": True,
                "history_status": "history_ready_for_local_replay",
                "evidence_path": "reports/active/hyperliquid_research_bundle.csv",
            }
        ]
    ).to_csv(active / "hyperliquid_research_bundle.csv", index=False)

    result = build_venue_route_scorecard(root=tmp_path, as_of=pd.Timestamp("2026-08-05T12:30:00+00:00"))
    row = pd.read_csv(result.paths["venue_route_scorecard"]).iloc[0]

    assert bool(row["history_ready"]) is True
    assert row["history_status"] == "history_ready_for_local_replay"
    assert bool(row["validation_route_ready"]) is False
    assert "missing_pair_cost_model" in row["blockers"]
    assert "missing_venue_specific_slippage_calibration" in row["blockers"]
    assert "hyperliquid_research_bundle.csv" in row["evidence_path"]


def test_venue_route_scorecard_uses_fee_profile_without_confusing_it_for_slippage_calibration(tmp_path):
    processed = tmp_path / "data" / "processed"
    active = tmp_path / "reports" / "active"
    processed.mkdir(parents=True)
    active.mkdir(parents=True)
    pair = "BTC-USD-ETH-USD"
    timestamp = "2026-08-05T12:00:00+00:00"
    pd.DataFrame(
        [{"pair": pair, "asset_x": "BTC-USD", "asset_y": "ETH-USD", "combined_score": 75.0, "available_venues": "hyperliquid"}]
    ).to_csv(processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "asset": asset,
                "venue": "hyperliquid",
                "tradable": True,
                "execution_authority": True,
                "source_timestamp": timestamp,
                "volume_24h": 2_000_000.0,
                "open_interest_usd": 10_000_000.0,
                "funding_rate": "0.0001",
                "evidence_path": "data/raw/hyperliquid_market_snapshots/snapshot.json",
            }
            for asset in ("BTC", "ETH")
        ]
    ).to_csv(processed / "market_venue_context.csv", index=False)
    pd.DataFrame(
        [{"pair": pair, "venue": "hyperliquid", "execution_ready": True, "ready_for_submission": False, "cost_model_aligned": False, "cost_model_profile": "missing_pair_cost_model"}]
    ).to_csv(tmp_path / "reports" / "paper_venue_preflight.csv", index=False)
    pd.DataFrame().to_csv(active / "venue_lane_test_plan.csv", index=False)
    pd.DataFrame().to_csv(active / "multi_venue_history_readiness_2026-06-25.csv", index=False)
    pd.DataFrame(
        [{"pair": pair, "asset_x": "BTC", "asset_y": "ETH", "venue": "hyperliquid", "history_ready": True, "history_status": "history_ready_for_local_replay"}]
    ).to_csv(active / "hyperliquid_research_bundle.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": pair,
                "asset_x": "BTC",
                "asset_y": "ETH",
                "venue": "hyperliquid",
                "cost_model_ready": True,
                "cost_model_status": "official_base_tier_conservative_fee_profile",
                "slippage_model_ready": False,
                "slippage_model_status": "insufficient_l2_depth_samples",
                "evidence_path": "reports/active/hyperliquid_pair_cost_model.csv",
            }
        ]
    ).to_csv(active / "hyperliquid_pair_cost_model.csv", index=False)

    result = build_venue_route_scorecard(root=tmp_path, as_of=pd.Timestamp("2026-08-05T12:30:00+00:00"))
    row = pd.read_csv(result.paths["venue_route_scorecard"]).iloc[0]

    assert bool(row["cost_model_ready"]) is True
    assert row["cost_model_status"] == "official_base_tier_conservative_fee_profile"
    assert bool(row["slippage_model_ready"]) is False
    assert row["slippage_model_status"] == "insufficient_l2_depth_samples"
    assert bool(row["validation_route_ready"]) is False
    assert "missing_pair_cost_model" not in row["blockers"]
    assert "insufficient_l2_depth_samples" in row["blockers"]


def test_venue_route_dashboard_treats_csv_nan_as_no_route():
    recommendations = pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "recommended_execution_venue": float("nan"),
                "recommended_paper_venue": float("nan"),
                "blockers": "missing_matching_venue_history",
            }
        ]
    )
    health = _data_health_rows(pd.DataFrame(), pd.DataFrame(), recommendations)
    route_health = health[health["area"] == "venue_routes_without_execution_evidence"].iloc[0]
    markdown = _command_center_markdown(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        health,
        route_recommendations=recommendations,
    )

    assert bool(route_health["ready"]) is False
    assert "pairs_without_execution_route=1" in route_health["blocker"]
    assert "- evidence-complete execution routes: 0" in markdown
    assert "- paper-submission-ready routes: 0" in markdown


def test_venue_route_current_state_is_research_only_until_execution_evidence(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "recommended_research_venue": "hyperliquid",
                "recommended_execution_venue": float("nan"),
                "recommended_paper_venue": float("nan"),
                "blockers": "missing_matching_venue_history",
            }
        ]
    ).to_csv(active / "venue_route_recommendations.csv", index=False)

    row = _venue_route_state_row(tmp_path)

    assert row["ready"] is False
    assert row["status"] == "research_only"
    assert row["blocker"] == "missing_matching_venue_history"
    assert "execution_routes=0" in row["detail"]


def test_venue_lane_test_plan_falls_back_to_gated_wizard_evidence(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT-TRUMPUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "TRUMPUSDT",
                "exchange": "binance",
                "interval": "daily",
                "exact_mode": "OU (Spread)",
                "mode_valid": True,
                "sharpe": 2.25,
                "returns_total": 0.24,
                "passes_sharpe_gate": True,
                "passes_returns_total_gt_20pct": True,
                "evidence_path": "reports/active/wizard_evidence.csv",
            }
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)

    result = build_venue_lane_test_plan(root=tmp_path)
    frame = pd.read_csv(result.paths["venue_lane_test_plan"])

    assert frame["pair"].tolist() == ["ETHUSDT-TRUMPUSDT"]
    assert frame.iloc[0]["test_status"] == "binance_research_only"
    assert "wizard_evidence.csv" in str(frame.iloc[0]["evidence_path"])


def test_venue_lane_test_plan_keeps_fresh_wizard_modes_when_legacy_queue_exists(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "OLDUSDT/LEGACYUSDT",
                "asset_x": "OLDUSDT",
                "asset_y": "LEGACYUSDT",
                "wizard_exchange": "binance",
                "wizard_sharpe": 2.0,
                "wizard_returns_total": 0.2,
            }
        ]
    ).to_csv(active / "crypto_wizards_next_best_sharpe_returns_queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT/FIDAUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "FIDAUSDT",
                "exchange": "binance",
                "interval": "daily",
                "exact_mode": "Copula",
                "mode_valid": True,
                "sharpe": 2.4,
                "returns_total": 0.3,
                "passes_sharpe_gate": True,
                "passes_returns_total_gt_20pct": True,
                "source_fresh": True,
                "source_authority": "discovery_only",
                "source_timestamp": "2026-08-05T12:00:00Z",
                "hypothesis_status": "DISCOVERY_ONLY",
                "evidence_path": "reports/active/wizard_evidence.csv",
            },
            {
                "pair": "ETHUSDT/FIDAUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "FIDAUSDT",
                "exchange": "binance",
                "interval": "daily",
                "exact_mode": "OU (Spread)",
                "mode_valid": True,
                "sharpe": 2.1,
                "returns_total": 0.25,
                "passes_sharpe_gate": True,
                "passes_returns_total_gt_20pct": True,
                "source_fresh": True,
                "source_authority": "discovery_only",
                "source_timestamp": "2026-08-05T12:00:00Z",
                "hypothesis_status": "DISCOVERY_ONLY",
                "evidence_path": "reports/active/wizard_evidence.csv",
            },
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)

    result = build_venue_lane_test_plan(root=tmp_path)
    frame = pd.read_csv(result.paths["venue_lane_test_plan"])
    fresh = frame[frame["pair"].eq("ETHUSDT/FIDAUSDT")]

    assert set(fresh["exact_mode"]) == {"Copula", "OU (Spread)"}
    assert fresh["source_fresh"].map(bool).all()
    assert set(fresh["source_authority"]) == {"discovery_only"}


def test_venue_lane_test_plan_routes_wizard_non_dydx_as_research_only():
    queue = pd.DataFrame(
        [
            {
                "pair": "ETHUSDT-TRUMPUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "TRUMPUSDT",
                "wizard_exchange": "binance",
                "asset_x_normalized": "ETH-USDT",
                "asset_y_normalized": "TRUMP-USDT",
                "normalized_pair": "ETH-USDT-TRUMP-USDT",
                "wizard_sharpe": 3.37,
                "wizard_returns_total": 0.205,
                "source_path": "reports/active/crypto_wizards_exchange_sample_2026-06-25.csv",
            },
            {
                "pair": "PEPE-USD-ETH-BTC",
                "asset_x": "PEPE-USD",
                "asset_y": "ETH-BTC",
                "scanner_exchange": "coinbase",
                "asset_x_normalized": "PEPE-USD",
                "asset_y_normalized": "ETH-BTC",
                "normalized_pair": "PEPE-USD-ETH-BTC",
                "wizard_sharpe": 2.95,
                "wizard_returns_total": 0.22,
                "source_path": "reports/active/crypto_wizards_exchange_sample_2026-06-25.csv",
            },
            {
                "pair": "ETHUSDT-TRUMPUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "TRUMPUSDT",
                "wizard_exchange": "bybit",
                "asset_x_normalized": "ETH-USDT",
                "asset_y_normalized": "TRUMP-USDT",
                "normalized_pair": "ETH-USDT-TRUMP-USDT",
                "wizard_sharpe": 2.9,
                "wizard_returns_total": 0.18,
                "source_path": "reports/active/crypto_wizards_exchange_sample_2026-06-25.csv",
            },
        ]
    )

    frame = pd.DataFrame(_venue_lane_test_rows(pd.DataFrame(), pd.DataFrame(), queue))

    assert set(frame["wizard_exchange"]) == {"binance", "coinbase", "bybit"}
    assert set(frame["pair_lane"]) == {"binance_research_lane", "coinbase_research_lane", "bybit_research_lane"}
    assert set(frame["test_status"]) == {"binance_research_only", "coinbase_research_only", "bybit_research_only"}
    assert len(frame[frame["normalized_pair"] == "ETH-USDT-TRUMP-USDT"]) == 2
    assert frame["next_step"].astype(str).str.contains("do_not_promote_until").all()


def test_multi_venue_history_readiness_ranks_fetchable_research_candidates(tmp_path, monkeypatch):
    from quant_platform import active_pipeline

    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    monkeypatch.setattr(active_pipeline, "ACTIVE", active)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT-TRUMPUSDT",
                "wizard_exchange": "binance",
                "asset_x": "ETHUSDT",
                "asset_y": "TRUMPUSDT",
                "asset_x_normalized": "ETH-USDT",
                "asset_y_normalized": "TRUMP-USDT",
                "normalized_pair": "ETH-USDT-TRUMP-USDT",
                "sharpe": 3.37,
                "zscore_norm": -0.64,
                "zscore_roll": 0.98,
            },
            {
                "pair": "BNB-USD-ETH-USD",
                "wizard_exchange": "dydx",
                "asset_x": "BNB-USD",
                "asset_y": "ETH-USD",
                "asset_x_normalized": "BNB-USD",
                "asset_y_normalized": "ETH-USD",
                "normalized_pair": "BNB-USD-ETH-USD",
                "sharpe": 1.98,
                "zscore_norm": -0.89,
                "zscore_roll": -0.41,
            },
        ]
    ).to_csv(active / "crypto_wizards_multi_venue_sharpe_rows_2026-06-25.csv", index=False)

    result = build_multi_venue_history_readiness(root=tmp_path, top_n=10)
    frame = pd.read_csv(result.paths["multi_venue_history_readiness"])

    assert {"readiness_status", "history_source_status", "cost_model_status", "next_step"}.issubset(frame.columns)
    binance = frame[frame["wizard_exchange"] == "binance"].iloc[0]
    dydx = frame[frame["wizard_exchange"] == "dydx"].iloc[0]
    assert binance["readiness_status"] == "ready_to_fetch"
    assert "needs_spot_fee_model" in binance["blockers"]
    assert dydx["readiness_status"] == "ready_for_replay"


def test_multi_venue_history_readiness_falls_back_to_queue_when_primary_rows_missing(tmp_path, monkeypatch):
    from quant_platform import active_pipeline

    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    monkeypatch.setattr(active_pipeline, "ACTIVE", active)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT-TRUMPUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "TRUMPUSDT",
                "wizard_exchange": "binance",
                "asset_x_normalized": "ETH-USDT",
                "asset_y_normalized": "TRUMP-USDT",
                "normalized_pair": "ETH-USDT-TRUMP-USDT",
                "sharpe": 3.37,
                "returns_total": 0.205,
            }
        ]
    ).to_csv(active / "crypto_wizards_next_best_sharpe_returns_queue.csv", index=False)

    result = build_multi_venue_history_readiness(root=tmp_path, top_n=10)
    frame = pd.read_csv(result.paths["multi_venue_history_readiness"])

    assert len(frame) == 1
    assert frame.iloc[0]["wizard_exchange"] == "binance"
    assert frame.iloc[0]["readiness_status"] == "ready_to_fetch"
    assert "crypto_wizards_next_best_sharpe_returns_queue.csv" in str(frame.iloc[0]["evidence_path"])


def test_multi_venue_history_readiness_prefers_current_wizard_shortlist(tmp_path, monkeypatch):
    from quant_platform import active_pipeline

    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    monkeypatch.setattr(active_pipeline, "ACTIVE", active)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT/FIDAUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "FIDAUSDT",
                "exchange": "binance",
                "interval": "daily",
                "period": 365,
                "setup_identity": "ETHUSDT|FIDAUSDT|daily|365|copula",
                "exact_mode": "Copula",
                "sharpe": 2.46,
                "returns_total": 0.41,
                "source_authority": "discovery_only",
                "source_timestamp": "2026-08-05T16:13:59Z",
                "source_fresh": True,
                "discovery_screen_status": "SCREEN_PASS",
                "research_quality_status": "RESEARCH_READY",
                "research_blockers": "",
                "zscore_last": 0.0,
                "zscore_roll_last": 0.0,
            }
        ]
    ).to_csv(active / "wizard_discovery_shortlist.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "STALEUSDT/OLDUSDT",
                "asset_x": "STALEUSDT",
                "asset_y": "OLDUSDT",
                "wizard_exchange": "binance",
                "sharpe": 99.0,
            }
        ]
    ).to_csv(active / "crypto_wizards_multi_venue_sharpe_rows_2026-06-25.csv", index=False)

    result = build_multi_venue_history_readiness(root=tmp_path, top_n=10)
    frame = pd.read_csv(result.paths["multi_venue_history_readiness"])

    assert result.paths["multi_venue_history_readiness"].name == "multi_venue_history_readiness.csv"
    assert list(frame["pair"]) == ["ETHUSDT/FIDAUSDT"]
    assert frame.iloc[0]["wizard_exchange"] == "binance"
    assert frame.iloc[0]["exact_mode"] == "Copula"
    assert bool(frame.iloc[0]["source_fresh"])
    assert frame.iloc[0]["candidate_source_kind"] == "current_wizard_shortlist"


def test_trade_dataset_writes_leakage_audit_and_required_labels():
    result = build_trade_dataset()

    dataset = pd.read_csv(result.paths["dataset_csv"])
    audit = pd.read_csv(result.paths["leakage_audit"])

    assert {"good_trade", "profit_after_cost", "max_adverse_excursion", "max_favorable_excursion", "hold_bars", "exit_reason"}.issubset(dataset.columns)
    assert {"uses_future_data", "uses_dashboard_hindsight", "feature_completeness_score", "leakage_blocker", "evidence_path"}.issubset(audit.columns)
    assert not audit["uses_future_data"].astype(bool).any()
    assert dataset["profit_after_cost"].ge(-1.0).all()
    assert dataset["return_aggregation"].eq("compounded_bar_returns_zero_floor").all()
    assert dataset["return_unit"].eq("fraction_of_equity").all()
    assert audit["leakage_blocker"].fillna("").eq("").all()


def test_system_check_reports_active_artifacts():
    result = system_check()

    frame = pd.read_csv(result.paths["system_check"])

    assert not frame.empty
    assert {"check", "ready", "blocker", "evidence_path", "next_action"}.issubset(frame.columns)
    assert "artifact:pair_universe.csv" in set(frame["check"])
    assert "artifact:wizard_mode_matrix_capture_queue.csv" in set(frame["check"])
    assert "artifact:wizard_mode_comparison.csv" in set(frame["check"])
    assert "artifact:current_wizard_ou_optimal_overlay_ledger.csv" in set(
        frame["check"]
    )
    assert "storage:deep_dashboard_refresh" in set(frame["check"])
    assert "storage:off_volume_archive_destination" in set(frame["check"])
    assert "storage:verified_archive_copy" in set(frame["check"])
    assert "storage:archive_release_dry_run" in set(frame["check"])
    assert "hyperliquid_testnet:deterministic_lifecycle_protocol" in set(
        frame["check"]
    )


def test_configured_env_key_presence_is_consistent_without_loading_secret(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("HYPERLIQUID_MASTER_ADDRESS", raising=False)
    (tmp_path / ".env.local").write_text(
        "HYPERLIQUID_MASTER_ADDRESS=0xconfigured\nEMPTY_KEY=\n",
        encoding="utf-8",
    )

    assert _configured_env_key_present(tmp_path, "HYPERLIQUID_MASTER_ADDRESS")
    assert not _configured_env_key_present(tmp_path, "EMPTY_KEY")
    assert not _configured_env_key_present(tmp_path, "MISSING_KEY")


def test_atomic_csv_write_preserves_prior_evidence_on_storage_failure(
    tmp_path, monkeypatch
):
    target = tmp_path / "canonical.csv"
    prior = b"pair,status\nBTC-ETH,accepted\n"
    target.write_bytes(prior)

    def fail_after_partial_write(self, destination, *args, **kwargs):
        Path(destination).write_bytes(b"pair,status\nBTC-ETH")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_after_partial_write)

    with pytest.raises(OSError, match="No space left on device"):
        active_pipeline_module._write_csv(pd.DataFrame([{"pair": "SOL-WLD"}]), target)

    assert target.read_bytes() == prior
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_optional_parquet_write_preserves_prior_evidence_on_storage_failure(
    tmp_path, monkeypatch
):
    target = tmp_path / "canonical.parquet"
    prior = b"prior-parquet-evidence"
    target.write_bytes(prior)

    def fail_after_partial_write(self, destination, *args, **kwargs):
        Path(destination).write_bytes(b"partial")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_after_partial_write)

    result = active_pipeline_module._write_parquet_if_available(
        pd.DataFrame([{"pair": "SOL-WLD"}]), target
    )

    assert result == "not_written:OSError"
    assert target.read_bytes() == prior
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_export_trade_gate_blocks_without_model_acceptance(tmp_path, monkeypatch):
    from quant_platform import active_pipeline

    monkeypatch.setattr(active_pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(active_pipeline, "ML_REPORTS", tmp_path / "reports" / "ml")
    monkeypatch.setattr(active_pipeline, "MODELS", tmp_path / "models" / "trade_gate")

    result = export_trade_gate_model(root=tmp_path)

    report = result.paths["export_report"].read_text(encoding="utf-8")
    assert "model_gated_backtest_not_accepted" in report


def test_dashboard_rows_include_reason_blocker_freshness_and_evidence():
    build_pair_universe()
    result = build_command_dashboard()

    live = pd.read_csv(result.paths["live_signals"])
    wizard_discovery = pd.read_csv(result.paths["wizard_discovery"])
    wizard_shortlist = pd.read_csv(result.paths["wizard_discovery_shortlist"])
    wizard_copula = pd.read_csv(result.paths["wizard_copula_discovery"])
    capture_queue = pd.read_csv(result.paths["wizard_pair_detail_capture_queue"])
    replay_handoff = pd.read_csv(result.paths["wizard_replay_handoff"])
    mode_comparison = pd.read_csv(result.paths["wizard_mode_comparison"])
    exploratory_cost_sensitivity = pd.read_csv(result.paths["wizard_exploratory_cost_sensitivity"])
    multi_venue = pd.read_csv(result.paths["multi_venue_history_readiness"])
    binance_readiness = pd.read_csv(result.paths["binance_spot_pair_readiness"])
    wizard_control = pd.read_csv(result.paths["wizard_control_plane"])
    candidate_ranking = pd.read_csv(result.paths["candidate_ranking"])
    data_health = pd.read_csv(result.paths["data_health"])
    command_center = result.paths["command_center"].read_text(encoding="utf-8")

    assert result.paths["hyperliquid_auxiliary_4h_walkforward"].exists()
    assert result.paths["student_mode_training_manifest"].exists()
    assert result.paths["hyperliquid_testnet_margin"].exists()
    assert result.paths["hyperliquid_testnet_preflight"].exists()
    assert result.paths["hyperliquid_evidence_cadence"].exists()
    assert result.paths["exhaustive_hyperliquid_concentration"].exists()
    assert result.paths["exhaustive_hyperliquid_leverage_status"].exists()
    assert result.paths["exhaustive_hyperliquid_leverage_candidates"].exists()
    assert result.paths["exhaustive_hyperliquid_leverage_scenarios"].exists()
    assert result.paths["exhaustive_hyperliquid_learning"].exists()
    assert result.paths["wizard_sweep_settings_capture_queue"].exists()
    assert result.paths["wizard_api_contract"].exists()
    assert result.paths["wizard_config_lineage"].exists()
    assert result.paths["current_wizard_ou_optimal_overlay_ledger"].exists()
    assert result.paths["current_wizard_ou_optimal_overlay_coverage"].exists()
    assert result.paths["current_wizard_ou_optimal_overlay_validation"].exists()
    assert "Wizard pair-page modes / scanner overlays: 7 / ou_optimal" in command_center
    ou_accounting = re.search(
        r"OU Optimal source rows accounted / true / false: (\d+) / (\d+) / (\d+)",
        command_center,
    )
    assert ou_accounting is not None
    total_rows, true_rows, false_rows = map(int, ou_accounting.groups())
    assert total_rows > 0
    assert total_rows == true_rows + false_rows
    assert "projected L2 captures remaining after rolling expiry" in command_center
    assert "family-wide statistical selection passes" in command_center
    assert "latest frozen validation" in command_center
    assert "dated research learning records" in command_center
    assert "training-eligible experiment summaries" in command_center

    assert {"reason", "blocker", "feature_timestamp", "evidence_path"}.issubset(live.columns)
    assert {"check", "status", "blocker", "evidence_path"}.issubset(wizard_control.columns)
    assert {"wizard_ranking_ready", "wizard_ranking_blocker", "wizard_rank"}.issubset(
        candidate_ranking.columns
    )
    assert "wizard_ranking_authority" in set(data_health["area"])
    assert {"source_fresh", "discovery_screen_status", "research_quality_status", "next_step"}.issubset(wizard_discovery.columns)
    assert {"exact_mode", "sharpe", "returns_total_pct", "research_quality_status"}.issubset(wizard_shortlist.columns)
    assert {"copula", "u1_given_u2", "u2_given_u1", "entry_signal_status"}.issubset(wizard_copula.columns)
    assert {"priority", "required_settings", "capture_status", "next_step"}.issubset(capture_queue.columns)
    assert {
        "settings_status",
        "venue_history_status",
        "exploratory_replay_status",
        "replay_status",
        "acceptance_authority",
    }.issubset(replay_handoff.columns)
    assert {
        "cost_case",
        "execution_risk_bps",
        "scenario_status",
        "acceptance_eligible",
        "paper_or_execution_eligible",
        "evidence_path",
    }.issubset(exploratory_cost_sensitivity.columns)
    assert {"exact_mode", "source_fresh", "candidate_source_kind", "readiness_status"}.issubset(multi_venue.columns)
    assert {"history_status", "wizard_settings_status", "research_execution_status", "next_action"}.issubset(binance_readiness.columns)
    assert {"exact_mode", "history_match_status", "acceptance_eligible", "paper_or_execution_eligible", "evidence_path"}.issubset(mode_comparison.columns)
    assert "## Wizard Discovery" in command_center
    assert "## Exhaustive Wizard To Hyperliquid" in command_center
    assert "rolling sample floor" in command_center
    assert " of 12" in command_center
    assert "## Copula Research" in command_center
    assert "## Exact-Mode Replay Handoff" in command_center
    assert "## Exploratory Cost Sensitivity" in command_center
    assert "## Cross-Mode Research Comparison" in command_center
    if "deep refresh blocked by storage" in command_center:
        assert "deep profile rebuilds evidence" not in command_center
    assert "## All-Venue Research Lanes" in command_center
    assert "Copula conditional values are a hypothesis signal, not a trade direction" in command_center
    assert "public spot history is research evidence only" in command_center


def test_paper_shortlist_and_validation_include_rl_winner_context(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    reports_rl = tmp_path / "reports" / "rl"
    reports_rl.mkdir(parents=True)
    reports_ml = tmp_path / "reports" / "ml"
    reports_ml.mkdir(parents=True)
    reports_root = tmp_path / "reports"
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "best_execution_venue": "dydx",
                "combined_score": 92.0,
                "acceptance_score": 81.0,
                "funding_drag_bps": 1.5,
                "available_timeframes": "1d",
                "decision_bucket": "PROMOTE",
                "decision_reason": "ready",
                "evidence_path": "reports/example.csv",
            },
            {
                "pair": "SOL-USD-TRX-USD",
                "best_execution_venue": "dydx",
                "combined_score": 88.0,
                "acceptance_score": 79.0,
                "funding_drag_bps": 2.0,
                "available_timeframes": "1d",
                "decision_bucket": "PROMOTE",
                "decision_reason": "ready",
                "evidence_path": "reports/example2.csv",
            },
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy_name": "learning_policy_005",
                "winner": 1,
                "profit_factor": 6.0,
                "stop_loss_pct": 0.07,
                "take_profit_pct": 0.14,
                "session_loss_cap_pct": 0.10,
                "top_pairs_entered": "BTC-USD-ETH-USD:15;SOL-USD-TRX-USD:7",
            }
        ]
    ).to_csv(reports_rl / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame([{"accepted": True, "blocker": ""}]).to_csv(reports_ml / "model_gated_acceptance.csv", index=False)
    pd.DataFrame([{"ready": True}, {"ready": True}]).to_csv(reports_root / "paper_execution_preflight.csv", index=False)

    shortlist = paper_candidate_shortlist_rows(root=tmp_path)
    focused = focused_paper_validation_rows(root=tmp_path)

    assert {"rl_policy_name", "rl_pair_focus_count", "rl_stop_loss_pct", "rl_take_profit_pct", "rl_session_loss_cap_pct"}.issubset(shortlist.columns)
    assert shortlist.iloc[0]["pair"] == "BTC-USD-ETH-USD"
    assert int(shortlist.iloc[0]["rl_pair_focus_count"]) == 15
    assert "rl_focus" in str(shortlist.iloc[0]["shortlist_reason"])
    assert {"rl_policy_winner", "rl_pair_focus_count", "rl_stop_loss_pct"}.issubset(focused.columns)
    assert focused.iloc[0]["validation_status"] == "hold_for_pair_specific_model_support"
    assert "pair-specific model support" in str(focused.iloc[0]["next_action"])


def test_paper_shortlist_prefers_quality_eligible_pairs(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True)
    (reports_root / "rl").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-TRX-USD",
                "best_execution_venue": "dydx",
                "combined_score": 99.0,
                "acceptance_score": 80.0,
                "funding_drag_bps": 1.0,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_only",
                "evidence_path": "reports/high_score.csv",
            },
            {
                "pair": "BTC-USD-ETH-USD",
                "best_execution_venue": "dydx",
                "combined_score": 95.0,
                "acceptance_score": 79.0,
                "funding_drag_bps": 1.2,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_and_quality",
                "evidence_path": "reports/quality.csv",
            },
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-TRX-USD", "research_execution_usable": False, "execution_usable": False},
            {"pair": "BTC-USD-ETH-USD", "research_execution_usable": True, "execution_usable": True},
        ]
    ).to_csv(reports_root / "pair_detail_quality_report.csv", index=False)

    shortlist = paper_candidate_shortlist_rows(root=tmp_path)

    assert shortlist["pair"].tolist() == ["BTC-USD-ETH-USD"]


def test_paper_shortlist_execution_filter_can_be_disabled(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "best_execution_venue": "injective",
                "combined_score": 99.0,
                "acceptance_score": 80.0,
                "funding_drag_bps": 1.0,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_and_quality",
                "evidence_path": "reports/candidate.csv",
            }
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)

    # Without a compatibility table row, this pair should be marked as incompatible and filtered out
    shortlist_filtered = paper_candidate_shortlist_rows(root=tmp_path)
    # With the optional gate disabled, we should still get the raw candidate row for diagnostics.
    shortlist_unfiltered = paper_candidate_shortlist_rows(root=tmp_path, require_execution_compatible=False)

    assert shortlist_filtered.empty is True
    assert not shortlist_unfiltered.empty
    assert shortlist_unfiltered.iloc[0]["pair"] == "BTC-USD-ETH-USD"
    assert bool(shortlist_unfiltered.iloc[0]["execution_compatible"]) is False


def test_paper_shortlist_accepts_confirmed_hyperliquid_pair_route(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    active = tmp_path / "reports" / "active"
    data_processed.mkdir(parents=True)
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "best_execution_venue": "hyperliquid",
                "execution_venue_ready": True,
                "combined_score": 99.0,
                "acceptance_score": 80.0,
                "funding_drag_bps": 1.0,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "route_valid",
                "evidence_path": "reports/hyperliquid_candidate.csv",
            }
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "testnet_perp_x": "BTC#3",
                "testnet_perp_y": "ETH#4",
                "mirrorable_for_paper": True,
                "mirror_blocker": "",
            }
        ]
    ).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)

    shortlist = paper_candidate_shortlist_rows(root=tmp_path)

    assert shortlist["pair"].tolist() == ["BTC-USD-ETH-USD"]
    assert bool(shortlist.iloc[0]["execution_compatible"]) is True
    assert str(shortlist.iloc[0]["execution_blocker"]) == ""


def test_filter_shortlist_to_visible_dydx_markets_keeps_pairs_when_market_status_unknown(tmp_path, monkeypatch):
    class _Adapter:
        def __init__(self, payloads):
            self.payloads = payloads

        def market_data(self, market: str):
            if market == "ETH-USD":
                raise RuntimeError("indexer timeout")
            payload = self.payloads.get(market)
            if payload is None:
                return {"payload": {"markets": {}}}
            return payload

    def _adapter_factory(_config):
        return _Adapter(
            {
                "BNB-USD": {"payload": {"markets": {"BNB-USD": {"status": "ACTIVE"}}}},
            }
        )

    monkeypatch.setattr("quant_platform.active_pipeline.build_dydx_indexer_adapter", _adapter_factory)

    shortlist = pd.DataFrame(
        [
            {"pair": "BNB-USD-ETH-USD", "best_execution_venue": "dydx"},
            {"pair": "ETH-USD-TRX-USD", "best_execution_venue": "binance"},
        ]
    )

    filtered = _filter_shortlist_to_visible_dydx_markets(shortlist)

    assert len(filtered) == 2
    assert set(filtered["pair"]) == {"BNB-USD-ETH-USD", "ETH-USD-TRX-USD"}


def test_filter_shortlist_to_visible_dydx_markets_rejects_inactive_markets(tmp_path, monkeypatch):
    class _Adapter:
        def market_data(self, market: str):
            rows = {
                "BNB-USD": {"payload": {"markets": {"BNB-USD": {"status": "ACTIVE"}}}},
                "ETH-USD": {"payload": {"markets": {"ETH-USD": {"status": "INACTIVE"}}}},
            }
            return rows.get(market, {"payload": {"markets": {}}})

    monkeypatch.setattr("quant_platform.active_pipeline.build_dydx_indexer_adapter", lambda _config: _Adapter())

    shortlist = pd.DataFrame([{"pair": "BNB-USD-ETH-USD", "best_execution_venue": "dydx"}])

    filtered = _filter_shortlist_to_visible_dydx_markets(shortlist)

    assert filtered.empty


def test_paper_shortlist_prefers_strong_pair_model_support_over_higher_rl_focus(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True)
    (reports_root / "rl").mkdir(parents=True)
    (reports_root / "brain").mkdir(parents=True)
    (reports_root / "ml").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "best_execution_venue": "dydx",
                "combined_score": 99.0,
                "acceptance_score": 80.0,
                "funding_drag_bps": 1.0,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_and_quality",
                "evidence_path": "reports/btc.csv",
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "best_execution_venue": "dydx",
                "combined_score": 103.0,
                "acceptance_score": 75.0,
                "funding_drag_bps": 1.2,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_and_quality",
                "evidence_path": "reports/sol.csv",
            },
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy_name": "learning_policy_005",
                "winner": 1,
                "profit_factor": 6.0,
                "stop_loss_pct": 0.07,
                "take_profit_pct": 0.14,
                "session_loss_cap_pct": 0.10,
                "top_pairs_entered": "BTC-USD-HYPE-USD:8;SOL-USD-HYPE-USD:14",
            }
        ]
    ).to_csv(reports_root / "rl" / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "research_execution_usable": True, "execution_usable": True},
            {"pair": "SOL-USD-HYPE-USD", "research_execution_usable": True, "execution_usable": True},
        ]
    ).to_csv(reports_root / "pair_detail_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "pair_model_support_status": "strong_model_support",
                "pair_model_taken_trades": 22,
                "pair_model_profit_factor": 4.39,
                "pair_model_mean_return": 0.0145,
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "pair_model_support_status": "no_model_support",
                "pair_model_taken_trades": 0,
                "pair_model_profit_factor": 0.0,
                "pair_model_mean_return": 0.0,
            },
        ]
    ).to_csv(reports_root / "brain" / "native_focus_repair_report.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "candidate_id": "native:BTC", "setup_identity": "native:BTC", "setup_role": "primary", "paper_credible": True},
            {"pair": "SOL-USD-HYPE-USD", "candidate_id": "native:SOL", "setup_identity": "native:SOL", "setup_role": "primary", "paper_credible": True},
        ]
    ).to_csv(reports_root / "brain" / "promotion_ladder.csv", index=False)

    shortlist = paper_candidate_shortlist_rows(root=tmp_path)

    assert shortlist["pair"].tolist() == ["BTC-USD-HYPE-USD"]
    assert shortlist.iloc[0]["pair_model_support_status"] == "strong_model_support"
    assert "strong_pair_model_support" in str(shortlist.iloc[0]["shortlist_reason"])


def test_paper_shortlist_can_promote_watch_candidate_when_route_valid_and_strong_model_supported(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True)
    (reports_root / "rl").mkdir(parents=True)
    (reports_root / "brain").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-LDO-USD",
                "best_execution_venue": "dydx",
                "combined_score": 44.6,
                "acceptance_score": 70.0,
                "funding_drag_bps": 1.0,
                "available_timeframes": "5m",
                "decision_bucket": "WATCH",
                "decision_reason": "route_valid_watch",
                "evidence_path": "reports/ldo.csv",
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "best_execution_venue": "dydx",
                "combined_score": 102.9,
                "acceptance_score": 74.9,
                "funding_drag_bps": 1.2,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_and_quality",
                "evidence_path": "reports/sol.csv",
            },
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-LDO-USD", "research_execution_usable": True, "execution_usable": True},
            {"pair": "SOL-USD-HYPE-USD", "research_execution_usable": True, "execution_usable": True},
        ]
    ).to_csv(reports_root / "pair_detail_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-LDO-USD",
                "pair_model_support_status": "strong_model_support",
                "pair_model_taken_trades": 21,
                "pair_model_profit_factor": 2.56,
                "pair_model_mean_return": 0.036,
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "pair_model_support_status": "no_model_support",
                "pair_model_taken_trades": 0,
                "pair_model_profit_factor": 0.0,
                "pair_model_mean_return": 0.0,
            },
        ]
    ).to_csv(reports_root / "brain" / "native_focus_repair_report.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-LDO-USD", "candidate_id": "native:LDO", "setup_identity": "native:LDO", "setup_role": "primary", "paper_credible": True},
            {"pair": "SOL-USD-HYPE-USD", "candidate_id": "native:SOL", "setup_identity": "native:SOL", "setup_role": "primary", "paper_credible": True},
        ]
    ).to_csv(reports_root / "brain" / "promotion_ladder.csv", index=False)

    shortlist = paper_candidate_shortlist_rows(root=tmp_path)

    assert shortlist.iloc[0]["pair"] == "BTC-USD-LDO-USD"
    assert shortlist.iloc[0]["pair_model_support_status"] == "strong_model_support"


def test_paper_shortlist_does_not_let_rl_focus_override_explicitly_unsupported_pairs(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True)
    (reports_root / "rl").mkdir(parents=True)
    (reports_root / "brain").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "best_execution_venue": "dydx",
                "combined_score": 113.7,
                "acceptance_score": 75.0,
                "funding_drag_bps": 1.0,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_and_quality",
                "evidence_path": "reports/btc.csv",
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "best_execution_venue": "dydx",
                "combined_score": 102.9,
                "acceptance_score": 74.9,
                "funding_drag_bps": 1.2,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_and_quality",
                "evidence_path": "reports/sol.csv",
            },
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy_name": "learning_policy_005",
                "winner": 1,
                "profit_factor": 6.0,
                "stop_loss_pct": 0.07,
                "take_profit_pct": 0.14,
                "session_loss_cap_pct": 0.10,
                "top_pairs_entered": "SOL-USD-HYPE-USD:14",
            }
        ]
    ).to_csv(reports_root / "rl" / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "research_execution_usable": True, "execution_usable": True},
            {"pair": "SOL-USD-HYPE-USD", "research_execution_usable": True, "execution_usable": True},
        ]
    ).to_csv(reports_root / "pair_detail_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "pair_model_support_status": "no_model_support",
                "pair_model_taken_trades": 0,
                "pair_model_profit_factor": 0.0,
                "pair_model_mean_return": 0.0,
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "pair_model_support_status": "no_model_support",
                "pair_model_taken_trades": 0,
                "pair_model_profit_factor": 0.0,
                "pair_model_mean_return": 0.0,
            },
        ]
    ).to_csv(reports_root / "brain" / "native_focus_repair_report.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "candidate_id": "native:BTC", "setup_identity": "native:BTC", "setup_role": "primary", "paper_credible": True},
            {"pair": "SOL-USD-HYPE-USD", "candidate_id": "native:SOL", "setup_identity": "native:SOL", "setup_role": "primary", "paper_credible": True},
        ]
    ).to_csv(reports_root / "brain" / "promotion_ladder.csv", index=False)

    shortlist = paper_candidate_shortlist_rows(root=tmp_path)

    assert shortlist.iloc[0]["pair"] == "BTC-USD-HYPE-USD"
    assert shortlist.iloc[0]["pair_model_support_status"] == "no_model_support"


def test_focused_paper_validation_blocks_when_pair_model_support_is_not_strong(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True)
    (reports_root / "rl").mkdir(parents=True)
    (reports_root / "brain").mkdir(parents=True)
    (reports_root / "ml").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "SOL-USD-HYPE-USD",
                "best_execution_venue": "dydx",
                "combined_score": 103.0,
                "acceptance_score": 75.0,
                "funding_drag_bps": 1.2,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_and_quality",
                "evidence_path": "reports/sol.csv",
            },
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy_name": "learning_policy_005",
                "winner": 1,
                "profit_factor": 6.0,
                "stop_loss_pct": 0.07,
                "take_profit_pct": 0.14,
                "session_loss_cap_pct": 0.10,
                "top_pairs_entered": "SOL-USD-HYPE-USD:14",
            }
        ]
    ).to_csv(reports_root / "rl" / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame(
        [{"pair": "SOL-USD-HYPE-USD", "research_execution_usable": True, "execution_usable": True}]
    ).to_csv(reports_root / "pair_detail_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "SOL-USD-HYPE-USD",
                "pair_model_support_status": "no_model_support",
                "pair_model_taken_trades": 0,
                "pair_model_profit_factor": 0.0,
                "pair_model_mean_return": 0.0,
            },
        ]
    ).to_csv(reports_root / "brain" / "native_focus_repair_report.csv", index=False)
    pd.DataFrame(
        [{"pair": "SOL-USD-HYPE-USD", "candidate_id": "native:SOL", "setup_identity": "native:SOL", "setup_role": "primary", "paper_credible": True}]
    ).to_csv(reports_root / "brain" / "promotion_ladder.csv", index=False)
    pd.DataFrame([{"accepted": True, "blocker": ""}]).to_csv(reports_root / "ml" / "model_gated_acceptance.csv", index=False)
    pd.DataFrame([{"ready": True}, {"ready": True}]).to_csv(reports_root / "paper_execution_preflight.csv", index=False)

    focused = focused_paper_validation_rows(root=tmp_path)

    assert focused.iloc[0]["pair_model_support_status"] == "no_model_support"
    assert focused.iloc[0]["validation_status"] == "hold_for_pair_specific_model_support"
    assert "pair-specific model support" in str(focused.iloc[0]["next_action"])


def test_model_gate_pair_support_report_ranks_strong_anchor_candidate(tmp_path):
    reports_ml = tmp_path / "reports" / "ml"
    reports_ml.mkdir(parents=True)
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "combined_score": 113.7},
            {"pair": "SOL-USD-HYPE-USD", "combined_score": 102.9},
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {"accepted": False, "failing_checks": "filtered_profit_factor_min;filtered_sharpe_positive"},
        ]
    ).to_csv(reports_ml / "model_gated_acceptance.csv", index=False)
    pd.DataFrame(
        [
            *[
                {"pair": "BTC-USD-HYPE-USD", "shadow_take": True, "realized_return": value}
                for value in [0.05, 0.04, 0.03, 0.02, 0.01, 0.02, 0.03, 0.01, 0.02, -0.01]
            ],
            *[
                {"pair": "BTC-USD-HYPE-USD", "shadow_take": False, "realized_return": value}
                for value in [-0.02, 0.01]
            ],
            *[
                {"pair": "SOL-USD-HYPE-USD", "shadow_take": False, "realized_return": value}
                for value in [-0.04, 0.02, -0.01]
            ],
        ]
    ).to_csv(reports_ml / "model_walkforward_predictions.csv", index=False)

    result = model_gate_pair_support_report(root=tmp_path)
    report = pd.read_csv(result.paths["model_gate_pair_support_report"])

    assert report.iloc[0]["pair"] == "BTC-USD-HYPE-USD"
    assert report.iloc[0]["support_status"] == "strong_model_support"
    assert "preserve strong pair-specific model support" in str(report.iloc[0]["recommended_repair_action"])


def test_model_gate_pair_support_report_prefers_current_shortlist_anchor_over_global_stronger_pair(tmp_path):
    reports_ml = tmp_path / "reports" / "ml"
    reports_ml.mkdir(parents=True)
    reports_rl = tmp_path / "reports" / "rl"
    reports_rl.mkdir(parents=True)
    reports_brain = tmp_path / "reports" / "brain"
    reports_brain.mkdir(parents=True)
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    reports_root = tmp_path / "reports"
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "best_execution_venue": "dydx",
                "combined_score": 113.7,
                "acceptance_score": 74.9,
                "funding_drag_bps": 0.01,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "ready",
                "evidence_path": "reports/btc.csv",
            },
            {
                "pair": "BTCUSDT-SOLUSDT",
                "best_execution_venue": "dydx",
                "combined_score": 70.0,
                "acceptance_score": 60.0,
                "funding_drag_bps": 0.01,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "ready",
                "evidence_path": "reports/other.csv",
            },
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy_name": "learning_policy_005",
                "winner": 1,
                "profit_factor": 6.0,
                "stop_loss_pct": 0.07,
                "take_profit_pct": 0.14,
                "session_loss_cap_pct": 0.10,
                "top_pairs_entered": "",
            }
        ]
    ).to_csv(reports_rl / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "research_execution_usable": True, "execution_usable": True},
            {"pair": "BTCUSDT-SOLUSDT", "research_execution_usable": True, "execution_usable": True},
        ]
    ).to_csv(reports_root / "pair_detail_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "pair_model_support_status": "strong_model_support",
                "pair_model_taken_trades": 22,
                "pair_model_profit_factor": 4.39,
                "pair_model_mean_return": 0.0145,
            },
        ]
    ).to_csv(reports_brain / "native_focus_repair_report.csv", index=False)
    pd.DataFrame(
        [{"pair": "BTC-USD-HYPE-USD", "candidate_id": "native:BTC", "setup_identity": "native:BTC", "setup_role": "primary", "paper_credible": True}]
    ).to_csv(reports_brain / "promotion_ladder.csv", index=False)
    pd.DataFrame(
        [{"accepted": False, "failing_checks": "filtered_profit_factor_min", "blocker": "model_gated_backtest_not_accepted"}]
    ).to_csv(reports_ml / "model_gated_acceptance.csv", index=False)
    pd.DataFrame(
        [
            *[
                {"pair": "BTCUSDT-SOLUSDT", "shadow_take": True, "realized_return": value}
                for value in [0.05] * 25
            ],
            *[
                {"pair": "BTC-USD-HYPE-USD", "shadow_take": True, "realized_return": value}
                for value in [0.03] * 10
            ],
        ]
    ).to_csv(reports_ml / "model_walkforward_predictions.csv", index=False)

    report = pd.read_csv(model_gate_pair_support_report(root=tmp_path).paths["model_gate_pair_support_report"])

    assert report.iloc[0]["pair"] == "BTC-USD-HYPE-USD"


def test_focused_paper_validation_uses_model_gate_anchor_action_for_strong_support_candidate(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True)
    (reports_root / "rl").mkdir(parents=True)
    (reports_root / "brain").mkdir(parents=True)
    (reports_root / "ml").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "best_execution_venue": "dydx",
                "combined_score": 113.7,
                "acceptance_score": 74.9,
                "funding_drag_bps": 0.01,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "ready",
                "evidence_path": "reports/btc.csv",
            },
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy_name": "learning_policy_005",
                "winner": 1,
                "profit_factor": 6.0,
                "stop_loss_pct": 0.07,
                "take_profit_pct": 0.14,
                "session_loss_cap_pct": 0.10,
                "top_pairs_entered": "",
            }
        ]
    ).to_csv(reports_root / "rl" / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame(
        [{"pair": "BTC-USD-HYPE-USD", "research_execution_usable": True, "execution_usable": True}]
    ).to_csv(reports_root / "pair_detail_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "pair_model_support_status": "strong_model_support",
                "pair_model_taken_trades": 22,
                "pair_model_profit_factor": 4.39,
                "pair_model_mean_return": 0.0145,
            },
        ]
    ).to_csv(reports_root / "brain" / "native_focus_repair_report.csv", index=False)
    pd.DataFrame(
        [{"pair": "BTC-USD-HYPE-USD", "candidate_id": "native:BTC", "setup_identity": "native:BTC", "setup_role": "primary", "paper_credible": True}]
    ).to_csv(reports_root / "brain" / "promotion_ladder.csv", index=False)
    pd.DataFrame(
        [{"accepted": False, "failing_checks": "filtered_profit_factor_min;filtered_sharpe_positive", "blocker": "model_acceptance_gates_not_met"}]
    ).to_csv(reports_root / "ml" / "model_gated_acceptance.csv", index=False)
    pd.DataFrame(
        [{"pair": "BTC-USD-HYPE-USD", "shadow_take": True, "realized_return": value} for value in [0.03] * 10]
    ).to_csv(reports_root / "ml" / "model_walkforward_predictions.csv", index=False)
    pd.DataFrame([{"ready": True}, {"ready": True}]).to_csv(reports_root / "paper_execution_preflight.csv", index=False)
    model_gate_pair_support_report(root=tmp_path)

    focused = focused_paper_validation_rows(root=tmp_path)

    assert focused.iloc[0]["pair_model_support_status"] == "strong_model_support"
    assert focused.iloc[0]["validation_status"] == "hold_for_model_or_gate"
    assert "preserve strong pair-specific model support" in str(focused.iloc[0]["next_action"])


def test_archive_dry_run_writes_manifest_and_moves_nothing():
    build_artifact_index()
    result = archive_from_index(dry_run=True)

    frame = pd.read_csv(result.paths["archive_manifest"])

    assert "planned_action" in frame.columns
    assert set(frame["planned_action"].dropna().unique()).issubset({"would_archive"})
