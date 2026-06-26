from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import (
    _venue_lane_test_rows,
    archive_from_index,
    build_artifact_index,
    build_command_dashboard,
    build_market_venue_context,
    build_multi_venue_history_readiness,
    build_pair_universe,
    build_trade_dataset,
    build_venue_lane_test_plan,
    export_trade_gate_model,
    system_check,
)


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
        assert hyperliquid["blocker"].astype(str).str.contains("missing_hyperliquid_local_replay").all()
    assert not lanes.empty
    assert {"asset", "best_lane", "next_action"}.issubset(lanes.columns)


def test_venue_lane_test_plan_routes_hyperliquid_without_promoting():
    build_market_venue_context()
    result = build_venue_lane_test_plan()

    frame = pd.read_csv(result.paths["venue_lane_test_plan"])

    assert not frame.empty
    assert {"pair_lane", "test_status", "funding_pulse_status", "next_step", "evidence_path"}.issubset(frame.columns)
    assert set(frame["funding_pulse_status"].dropna().unique()) == {"needs_api_key"}
    hyperliquid_rows = frame[frame["pair_lane"].astype(str).str.contains("hyperliquid", na=False)]
    if not hyperliquid_rows.empty:
        assert hyperliquid_rows["test_status"].astype(str).str.contains("hyperliquid").all()
        assert not hyperliquid_rows["next_step"].astype(str).str.contains("promote", case=False, na=False).any()


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


def test_trade_dataset_writes_leakage_audit_and_required_labels():
    result = build_trade_dataset()

    dataset = pd.read_csv(result.paths["dataset_csv"])
    audit = pd.read_csv(result.paths["leakage_audit"])

    assert {"good_trade", "profit_after_cost", "max_adverse_excursion", "max_favorable_excursion", "hold_bars", "exit_reason"}.issubset(dataset.columns)
    assert {"uses_future_data", "uses_dashboard_hindsight", "feature_completeness_score", "leakage_blocker", "evidence_path"}.issubset(audit.columns)
    assert not audit["uses_future_data"].astype(bool).any()


def test_system_check_reports_active_artifacts():
    result = system_check()

    frame = pd.read_csv(result.paths["system_check"])

    assert not frame.empty
    assert {"check", "ready", "blocker", "evidence_path", "next_action"}.issubset(frame.columns)
    assert "artifact:pair_universe.csv" in set(frame["check"])


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

    assert {"reason", "blocker", "feature_timestamp", "evidence_path"}.issubset(live.columns)


def test_archive_dry_run_writes_manifest_and_moves_nothing():
    build_artifact_index()
    result = archive_from_index(dry_run=True)

    frame = pd.read_csv(result.paths["archive_manifest"])

    assert "planned_action" in frame.columns
    assert set(frame["planned_action"].dropna().unique()).issubset({"would_archive"})
