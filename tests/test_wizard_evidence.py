from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from quant_platform.wizard_evidence import (
    DISCOVERY_MIN_RETURNS_TOTAL,
    DISCOVERY_MIN_SHARPE,
    WIZARD_PAIR_SETTINGS_CAPTURE_TEMPLATE_COLUMNS,
    build_wizard_diagnostic_confirmation,
    build_wizard_discovery_triage,
    build_wizard_evidence,
    build_wizard_exact_mode_capture_queue,
    build_wizard_exploratory_cost_sensitivity,
    build_wizard_hypotheses,
    build_wizard_local_parity,
    build_wizard_mode_matrix_capture_queue,
    build_wizard_mode_replay_capability,
    build_wizard_pair_detail_capture_queue,
    build_wizard_pair_settings_capture_template,
    build_wizard_replay_handoff,
    build_wizard_strategy_alignment_report,
    ids_from_exact_mode,
    import_wizard_pair_settings_capture,
    mode_from_ids,
)


def test_exact_mode_id_mapping_round_trips():
    assert mode_from_ids(3, 1) == "Static (Spread)"
    assert mode_from_ids(1, 3) == "Copula"
    assert ids_from_exact_mode("OU (ZScoreR)") == (2, 2)


def test_scanner_evidence_preserves_stationarity_signal_liquidity_and_risk_fields(tmp_path):
    active = tmp_path / "reports" / "active"
    raw = tmp_path / "data" / "raw" / "crypto_wizards_scanner"
    active.mkdir(parents=True)
    raw.mkdir(parents=True)
    source = raw / "daily_merged.json"
    _write_json(
        source,
        [
            {
                "_row_index": 0,
                "symbol_1": "ETHUSDT",
                "symbol_2": "FIDAUSDT",
                "exchange": "binance",
                "interval": "daily",
                "period": 365,
                "pair_id": 101,
                "id": "101-1-3",
                "spread_id": 1,
                "strategy_id": 3,
                "sharpe": 2.46,
                "returns_total": 0.4147,
                "backtest_ts": 1785536148,
                "coint_eg": True,
                "coint_eg_inc_trend": True,
                "coint_eg_p": 0.0039,
                "johansen_coint": False,
                "zscore_last": -2.15,
                "zscore_roll_last": -1.8,
                "zscore_window": 52,
                "mini_zscore": [1.0, -2.15],
                "stddev_cross": 2,
                "zero_cross": 27,
                "sym_1_volume": 16_243_231.0,
                "sym_2_volume": 185_032.27,
                "sym_1_volatility_lt": 0.78,
                "sym_2_volatility_lt": 0.87,
                "x_weighting": 0.5,
                "y_weighting": 0.5,
                "profile_match": True,
                "ou_optimal": False,
                "closed": 1,
                "win_rate": 0.75,
                "mdd": -0.0174,
                "var": -0.0106,
                "cvar": -0.0136,
                "ml_confidence": 0.4,
            }
        ],
    )
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT/FIDAUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "FIDAUSDT",
                "exchange": "binance",
                "interval": "daily",
                "source_timestamp": "2026-08-05T16:00:00Z",
                "api_rank": 1,
                "source_path": str(source.relative_to(tmp_path)),
            }
        ]
    ).to_csv(active / "crypto_wizards_live_scanner_capture.csv", index=False)

    result = build_wizard_evidence(root=tmp_path, now=datetime(2026, 8, 5, 16, 10, tzinfo=timezone.utc))
    row = pd.read_csv(result.paths["wizard_evidence"]).iloc[0]

    assert row["exact_mode"] == "Copula"
    assert row["stationarity_status"] == "engle_granger_with_trend"
    assert bool(row["engle_granger_cointegrated"])
    assert bool(row["engle_granger_trend"])
    assert not bool(row["johansen_cointegrated"])
    assert row["zscore_last"] == -2.15
    assert row["zscore_roll_last"] == -1.8
    assert row["volume_min"] == 185_032.27
    assert row["drawdown"] == -0.0174
    assert row["cvar"] == -0.0136
    assert row["mini_zscore"] == "[1.0,-2.15]"
    assert str(row["backtest_timestamp_utc"]).startswith("2026-")


def test_discovery_triage_separates_current_screen_from_research_and_stale_evidence(tmp_path):
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
                "source_authority": "discovery_only",
                "source_timestamp": "2026-08-05T16:00:00Z",
                "source_fresh": True,
                "source_health": "healthy",
                "sharpe": 2.4,
                "returns_total": 0.3,
                "returns_total_pct": 30.0,
                "passes_sharpe_gate": True,
                "passes_returns_total_gt_20pct": True,
                "stationarity_status": "engle_granger",
                "engle_granger_cointegrated": True,
                "engle_granger_trend": False,
                "johansen_cointegrated": False,
                "hurst": 0.35,
                "half_life": 12.0,
                "volume_min": 200_000.0,
                "u1_given_u2": 0.04,
                "u2_given_u1": 0.96,
                "evidence_path": "data/raw/wizard.json",
            },
            {
                "pair": "OLDUSDT/STALEUSDT",
                "asset_x": "OLDUSDT",
                "asset_y": "STALEUSDT",
                "exchange": "bybit",
                "interval": "daily",
                "exact_mode": "OU (Spread)",
                "mode_valid": True,
                "source_authority": "discovery_only",
                "source_timestamp": "2026-07-01T00:00:00Z",
                "source_fresh": False,
                "source_health": "healthy",
                "sharpe": 9.0,
                "returns_total": 0.9,
                "passes_sharpe_gate": True,
                "passes_returns_total_gt_20pct": True,
            },
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)

    result = build_wizard_discovery_triage(root=tmp_path)
    triage = pd.read_csv(result.paths["wizard_discovery_triage"])
    shortlist = pd.read_csv(result.paths["wizard_discovery_shortlist"])
    copula = pd.read_csv(result.paths["wizard_copula_discovery_triage"])

    current = triage[triage["pair"].eq("ETHUSDT/FIDAUSDT")].iloc[0]
    stale = triage[triage["pair"].eq("OLDUSDT/STALEUSDT")].iloc[0]
    assert current["discovery_screen_status"] == "SCREEN_PASS"
    assert current["research_quality_status"] == "RESEARCH_READY"
    assert current["entry_signal_status"] == "COPULA_THRESHOLD_CAPTURE_REQUIRED"
    assert current["copula_data_status"] == "CONDITIONAL_VALUES_CAPTURED"
    assert stale["discovery_screen_status"] == "STALE_SOURCE"
    assert shortlist["pair"].tolist() == ["ETHUSDT/FIDAUSDT"]
    assert copula["pair"].tolist() == ["ETHUSDT/FIDAUSDT"]


def test_pair_detail_capture_queue_prioritizes_fresh_research_ready_copula_setup(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
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
                "spread_id": 1,
                "strategy_id": 3,
                "sharpe": 2.4,
                "returns_total_pct": 30.0,
                "research_quality_status": "RESEARCH_READY",
                "research_blockers": "",
                "source_timestamp": "2026-08-05T16:00:00Z",
                "source_fresh": True,
                "source_authority": "discovery_only",
                "discovery_screen_status": "SCREEN_PASS",
                "backtest_settings_complete": False,
                "backtest_settings_missing": "entry_thresholds;exit_thresholds;wizard_cost_assumptions;copula_thresholds",
                "evidence_path": "data/raw/wizard.json",
            },
            {
                "pair": "OLDUSDT/STALEUSDT",
                "source_fresh": False,
                "discovery_screen_status": "SCREEN_PASS",
                "backtest_settings_complete": False,
            },
        ]
    ).to_csv(active / "wizard_discovery_triage.csv", index=False)

    result = build_wizard_pair_detail_capture_queue(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_pair_detail_capture_queue"])

    assert frame["pair"].tolist() == ["ETHUSDT/FIDAUSDT"]
    row = frame.iloc[0]
    assert row["priority"] == "P1"
    assert "copula_thresholds" in row["required_settings"]
    assert "copula_directional_entry_rule" in row["required_settings"]
    assert row["capture_status"] == "NEEDS_DASHBOARD_PAIR_DETAIL_CAPTURE"


def test_replay_handoff_joins_mode_settings_history_and_economics_without_execution_authority(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
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
                "spread_id": 1,
                "strategy_id": 3,
                "source_timestamp": "2026-08-05T16:00:00Z",
                "source_fresh": True,
                "source_authority": "discovery_only",
                "sharpe": 2.4,
                "returns_total_pct": 30.0,
                "research_quality_status": "RESEARCH_READY",
                "research_blockers": "",
                "entry_signal_status": "COPULA_THRESHOLD_CAPTURE_REQUIRED",
                "backtest_settings_complete": False,
                "backtest_settings_missing": "entry_thresholds;exit_thresholds;wizard_cost_assumptions;copula_thresholds",
                "evidence_path": "data/raw/wizard.json",
            },
            {
                "pair": "BTC-USD/ETH-USD",
                "asset_x": "BTC-USD",
                "asset_y": "ETH-USD",
                "exchange": "dydx",
                "interval": "daily",
                "period": 365,
                "setup_identity": "BTC-USD|ETH-USD|daily|365|static_(spread)",
                "exact_mode": "Static (Spread)",
                "spread_id": 3,
                "strategy_id": 1,
                "source_timestamp": "2026-08-05T16:00:00Z",
                "source_fresh": True,
                "source_authority": "pair_detail_capture",
                "sharpe": 2.1,
                "returns_total_pct": 22.0,
                "research_quality_status": "RESEARCH_READY",
                "research_blockers": "",
                "entry_signal_status": "AT_OR_BEYOND_Z2",
                "backtest_settings_complete": True,
                "backtest_settings_missing": "",
                "evidence_path": "data/raw/captured_settings.json",
            },
            {
                "pair": "SOLUSDT/AVAXUSDT",
                "asset_x": "SOLUSDT",
                "asset_y": "AVAXUSDT",
                "exchange": "binance",
                "interval": "daily",
                "period": 365,
                "setup_identity": "SOLUSDT|AVAXUSDT|daily|365|ou_(spread)",
                "exact_mode": "OU (Spread)",
                "spread_id": 2,
                "strategy_id": 1,
                "source_timestamp": "2026-08-05T16:00:00Z",
                "source_fresh": True,
                "source_authority": "pair_detail_capture",
                "sharpe": 2.0,
                "returns_total_pct": 21.0,
                "research_quality_status": "RESEARCH_READY",
                "research_blockers": "",
                "entry_signal_status": "AT_OR_BEYOND_Z2",
                "backtest_settings_complete": True,
                "backtest_settings_missing": "",
                "evidence_path": "data/raw/captured_ou_settings.json",
            },
        ]
    ).to_csv(active / "wizard_discovery_shortlist.csv", index=False)
    pd.DataFrame(
        [
            {
                "setup_identity": "ETHUSDT|FIDAUSDT|daily|365|copula",
                "pair": "ETHUSDT/FIDAUSDT",
                "exact_mode": "Copula",
                "capture_status": "NEEDS_DASHBOARD_PAIR_DETAIL_CAPTURE",
                "required_settings": "entry_thresholds;exit_thresholds;copula_thresholds;copula_directional_entry_rule;copula_exit_rule",
                "next_step": "capture_settings_then_replay_copula_on_binance",
                "evidence_path": "reports/active/capture_queue.csv",
            }
        ]
    ).to_csv(active / "wizard_pair_detail_capture_queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "setup_identity": "ETHUSDT|FIDAUSDT|daily|365|copula",
                "pair": "ETHUSDT/FIDAUSDT",
                "exact_mode": "Copula",
                "readiness_status": "ready_to_fetch",
                "cost_model_status": "needs_spot_fee_model",
                "slippage_model_status": "needs_orderbook_or_volume_slippage_model",
                "funding_or_borrow_status": "borrow_short_cost_tracked_not_research_gate",
                "evidence_path": "reports/active/multi.csv",
            },
            {
                "setup_identity": "BTC-USD|ETH-USD|daily|365|static_(spread)",
                "pair": "BTC-USD/ETH-USD",
                "exact_mode": "Static (Spread)",
                "readiness_status": "ready_for_replay",
                "cost_model_status": "available_dydx_cost_model",
                "slippage_model_status": "available_dydx_slippage_model",
                "funding_or_borrow_status": "available_dydx_funding",
                "evidence_path": "reports/active/multi.csv",
            },
            {
                "setup_identity": "SOLUSDT|AVAXUSDT|daily|365|ou_(spread)",
                "pair": "SOLUSDT/AVAXUSDT",
                "exact_mode": "OU (Spread)",
                "readiness_status": "ready_to_fetch",
                "cost_model_status": "needs_spot_fee_model",
                "slippage_model_status": "needs_orderbook_or_volume_slippage_model",
                "funding_or_borrow_status": "borrow_short_cost_tracked_not_research_gate",
                "evidence_path": "reports/active/multi.csv",
            },
        ]
    ).to_csv(active / "multi_venue_history_readiness.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT/FIDAUSDT",
                "exact_mode": "Copula",
                "history_status": "history_ready_needs_settings_and_cost_model",
                "cost_model_status": "placeholder_only_not_validated",
                "slippage_status": "venue_specific_slippage_not_calibrated",
                "funding_borrow_status": "spot_borrow_short_cost_not_modelled",
                "research_execution_status": "research_only_execution_blocked",
                "evidence_path": "reports/active/binance.csv",
            },
            {
                "pair": "SOLUSDT/AVAXUSDT",
                "exact_mode": "OU (Spread)",
                "history_status": "history_ready_needs_settings_and_cost_model",
                "cost_model_status": "placeholder_only_not_validated",
                "slippage_status": "venue_specific_slippage_not_calibrated",
                "funding_borrow_status": "spot_borrow_short_cost_not_modelled",
                "research_execution_status": "research_only_execution_blocked",
                "evidence_path": "reports/active/binance.csv",
            }
        ]
    ).to_csv(active / "binance_spot_pair_readiness.csv", index=False)

    result = build_wizard_replay_handoff(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_replay_handoff"])
    binance = frame[frame["pair"].eq("ETHUSDT/FIDAUSDT")].iloc[0]
    dydx = frame[frame["pair"].eq("BTC-USD/ETH-USD")].iloc[0]
    provisional = frame[frame["pair"].eq("SOLUSDT/AVAXUSDT")].iloc[0]

    assert binance["replay_status"] == "BLOCKED_SETTINGS_CAPTURE"
    assert "exact_wizard_settings_missing" in binance["replay_blockers"]
    assert binance["next_step"] == "capture_settings_then_replay_copula_on_binance"
    assert binance["exploratory_replay_status"] == "BLOCKED_SETTINGS_CAPTURE"
    assert provisional["exploratory_replay_status"] == "READY_FOR_EXPLORATORY_LOCAL_REPLAY"
    assert provisional["replay_status"] == "BLOCKED_VENUE_ECONOMICS"
    assert provisional["exploratory_cost_policy"] == "provisional_cost_sensitivity_required"
    assert dydx["replay_status"] == "READY_FOR_COSTED_LOCAL_REPLAY"
    assert dydx["exploratory_replay_status"] == "READY_FOR_EXPLORATORY_LOCAL_REPLAY"
    assert dydx["acceptance_authority"] == "local_point_in_time_costed_replay_only"


def test_exploratory_cost_sensitivity_is_reproducible_and_never_acceptance_authority(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "setup_identity": "SOLUSDT|AVAXUSDT|daily|365|ou_(spread)",
                "pair": "SOLUSDT/AVAXUSDT",
                "exchange": "binance",
                "interval": "daily",
                "exact_mode": "OU (Spread)",
                "research_quality_status": "RESEARCH_READY",
                "settings_status": "CURRENT_PAIR_DETAIL_SETTINGS_CAPTURED",
                "venue_history_status": "history_ready_needs_settings_and_cost_model",
                "exploratory_replay_status": "READY_FOR_EXPLORATORY_LOCAL_REPLAY",
                "exploratory_next_step": "run_provisional_exact_mode_local_replay_with_cost_sensitivity",
                "evidence_path": "reports/active/handoff_evidence.csv",
            },
            {
                "setup_identity": "ETHUSDT|FIDAUSDT|daily|365|copula",
                "pair": "ETHUSDT/FIDAUSDT",
                "exchange": "binance",
                "interval": "daily",
                "exact_mode": "Copula",
                "research_quality_status": "RESEARCH_READY",
                "settings_status": "NEEDS_DASHBOARD_PAIR_DETAIL_CAPTURE",
                "venue_history_status": "history_ready_needs_settings_and_cost_model",
                "exploratory_replay_status": "BLOCKED_SETTINGS_CAPTURE",
                "exploratory_next_step": "capture_exact_wizard_settings_before_local_replay",
                "evidence_path": "reports/active/handoff_evidence.csv",
            },
        ]
    ).to_csv(active / "wizard_replay_handoff.csv", index=False)
    profile_path = tmp_path / "cost_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "profile_id": "test_research_sensitivity",
                "profile_version": "v1",
                "cost_assumption_authority": "provisional_research_sensitivity_only",
                "acceptance_eligible": False,
                "paper_or_execution_eligible": False,
                "scenarios": [
                    {
                        "cost_case": "gross_only_upper_bound",
                        "taker_fee_bps": 0.0,
                        "slippage_bps": 0.0,
                        "execution_risk_bps": 0.0,
                        "funding_or_borrow_bps_per_day": 0.0,
                        "partial_fill_probability": 0.0,
                        "partial_fill_fraction": 0.0,
                        "partial_fill_penalty_bps": 0.0,
                    },
                    {
                        "cost_case": "provisional_base_sensitivity",
                        "taker_fee_bps": 5.0,
                        "slippage_bps": 4.0,
                        "execution_risk_bps": 2.0,
                        "funding_or_borrow_bps_per_day": 1.0,
                        "partial_fill_probability": 0.1,
                        "partial_fill_fraction": 0.5,
                        "partial_fill_penalty_bps": 2.0,
                    },
                    {
                        "cost_case": "provisional_stress_sensitivity",
                        "taker_fee_bps": 7.5,
                        "slippage_bps": 8.0,
                        "execution_risk_bps": 4.0,
                        "funding_or_borrow_bps_per_day": 3.0,
                        "partial_fill_probability": 0.1,
                        "partial_fill_fraction": 0.5,
                        "partial_fill_penalty_bps": 2.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = build_wizard_exploratory_cost_sensitivity(root=tmp_path, profile_path=profile_path)
    frame = pd.read_csv(result.paths["wizard_exploratory_cost_sensitivity"])
    ready = frame[frame["pair"].eq("SOLUSDT/AVAXUSDT")]
    blocked = frame[frame["pair"].eq("ETHUSDT/FIDAUSDT")]
    base = ready[ready["cost_case"].eq("provisional_base_sensitivity")].iloc[0]

    assert len(frame) == 6
    assert set(ready["scenario_status"]) == {"READY_FOR_PROVISIONAL_COST_SENSITIVITY"}
    assert set(blocked["scenario_status"]) == {"BLOCKED_BY_EXPLORATORY_REPLAY_HANDOFF"}
    assert base["execution_risk_bps"] == 2.0
    assert base["two_leg_round_trip_execution_cost_bps"] == 44.4
    assert not frame["acceptance_eligible"].astype(bool).any()
    assert not frame["paper_or_execution_eligible"].astype(bool).any()
    assert result.summary["acceptance_eligible_rows"] == 0
    assert "never an economic result" in result.paths["wizard_exploratory_cost_sensitivity_summary"].read_text(encoding="utf-8")


def test_pair_settings_capture_template_and_import_require_observed_exact_mode_settings(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "priority": "P1",
                "setup_identity": "ETHUSDT|FIDAUSDT|daily|365|copula",
                "pair": "ETHUSDT/FIDAUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "FIDAUSDT",
                "exchange": "binance",
                "interval": "daily",
                "period": 365,
                "exact_mode": "Copula",
                "spread_id": 1,
                "strategy_id": 3,
                "required_settings": "entry_thresholds;exit_thresholds;copula_thresholds;copula_directional_entry_rule;copula_exit_rule",
            }
        ]
    ).to_csv(active / "wizard_pair_detail_capture_queue.csv", index=False)

    template = build_wizard_pair_settings_capture_template(root=tmp_path)
    template_frame = pd.read_csv(template.paths["wizard_pair_settings_capture_template"])

    assert template_frame.loc[0, "exact_mode"] == "Copula"
    assert {"capture_evidence_path", "capture_confirmed", "entry_long_operator", "copula_direction_view"}.issubset(
        template_frame.columns
    )

    captured = {column: "" for column in WIZARD_PAIR_SETTINGS_CAPTURE_TEMPLATE_COLUMNS}
    captured.update(
        {
            "pair": "ETHUSDT/FIDAUSDT",
            "asset_x": "ETHUSDT",
            "asset_y": "FIDAUSDT",
            "exchange": "binance",
            "interval": "daily",
            "period": 365,
            "exact_mode": "Copula",
            "spread_id": 1,
            "strategy_id": 3,
            "capture_timestamp_utc": "2026-08-05T12:00:00Z",
            "capture_evidence_path": "data/raw/wizard/eth_fida_copula_capture.png",
            "capture_confirmed": True,
            "pair_page_url": "https://cryptowizards.net/wizards/zscore/pair/101",
            "entry_long_operator": "<=",
            "entry_long_value": 0.05,
            "entry_long_position": "long_x_short_y",
            "entry_short_operator": ">=",
            "entry_short_value": 0.95,
            "entry_short_position": "short_x_long_y",
            "exit_long_operator": ">=",
            "exit_long_value": 0.45,
            "exit_short_operator": "<=",
            "exit_short_value": 0.55,
            "capital_weighting_slider_value": 0.5,
            "capital_weighting_asset": "ETHUSDT",
            "capital_weighting": "Risk Reduction",
            "metric_mode": "Standard",
            "x_weighting": 0.5,
            "y_weighting": 0.5,
            "hedge_ratio": 1.0,
            "commission_rate": 0.0002,
            "slippage_rate": 0.0004,
            "stop_loss_rate": 0.0,
            "exit_n_periods": 0,
            "spread_type": "Copula",
            "copula_family": "gaussian",
            "copula_signal_type": "arbitrage",
            "copula_direction_view": "u1_given_u2",
            "copula_entry_lower": 0.05,
            "copula_entry_upper": 0.95,
            "copula_exit_lower": 0.45,
            "copula_exit_upper": 0.55,
        }
    )
    invalid = dict(captured)
    invalid["pair"] = "SOLUSDT/AVAXUSDT"
    invalid["asset_x"] = "SOLUSDT"
    invalid["asset_y"] = "AVAXUSDT"
    invalid["entry_long_operator"] = ""
    capture_path = tmp_path / "completed_settings_capture.csv"
    pd.DataFrame([captured, invalid], columns=WIZARD_PAIR_SETTINGS_CAPTURE_TEMPLATE_COLUMNS).to_csv(capture_path, index=False)

    imported = import_wizard_pair_settings_capture(capture_path, root=tmp_path)
    validation = pd.read_csv(imported.paths["wizard_pair_settings_capture_validation"])
    active_capture = pd.read_csv(imported.paths["wizard_pair_page_settings_capture"])
    evidence = build_wizard_evidence(root=tmp_path, now=datetime(2026, 8, 5, 12, 10, tzinfo=timezone.utc))
    evidence_frame = pd.read_csv(evidence.paths["wizard_evidence"])

    assert imported.summary == {"input_rows": 2, "accepted_rows": 1, "blocked_rows": 1}
    assert validation["accepted_into_active_capture"].tolist() == [True, False]
    assert "missing_entry_thresholds" in validation.loc[1, "validation_blockers"]
    assert active_capture["pair"].tolist() == ["ETHUSDT/FIDAUSDT"]
    assert active_capture["config_schema_version"].eq("wizard_run_config.v1").all()
    assert active_capture["settings_config_hash"].str.len().eq(64).all()
    assert validation.loc[0, "settings_config_hash"] == active_capture.loc[0, "settings_config_hash"]
    assert active_capture.loc[0, "capital_weighting"] == "Risk Reduction"
    assert active_capture.loc[0, "metric_mode"] == "Standard"
    assert active_capture.loc[0, "spread_type"] == "Copula"
    assert bool(evidence_frame.loc[0, "backtest_settings_complete"])
    assert evidence_frame.loc[0, "source_system"] == "crypto_wizards_pair_page_capture"


def test_mode_replay_capability_separates_captured_research_replay_from_acceptance(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "priority": "P1",
                "setup_identity": "AAAUSDT|BBBUSDT|daily|365|static_(spread)",
                "pair": "AAAUSDT/BBBUSDT",
                "asset_x": "AAAUSDT",
                "asset_y": "BBBUSDT",
                "exchange": "binance",
                "interval": "daily",
                "period": 365,
                "exact_mode": "Static (Spread)",
                "research_quality_status": "RESEARCH_READY",
                "venue_history_status": "history_ready_needs_settings_and_cost_model",
                "evidence_path": "reports/active/wizard_replay_handoff.csv",
            },
            {
                "priority": "P1",
                "setup_identity": "CCCUSDT|DDDUSDT|daily|365|copula",
                "pair": "CCCUSDT/DDDUSDT",
                "asset_x": "CCCUSDT",
                "asset_y": "DDDUSDT",
                "exchange": "binance",
                "interval": "daily",
                "period": 365,
                "exact_mode": "Copula",
                "research_quality_status": "RESEARCH_READY",
                "venue_history_status": "history_ready_needs_settings_and_cost_model",
                "evidence_path": "reports/active/wizard_replay_handoff.csv",
            },
        ]
    ).to_csv(active / "wizard_replay_handoff.csv", index=False)
    pd.DataFrame(
        [
            {
                "setup_identity": "AAAUSDT|BBBUSDT|daily|365|static_(spread)",
                "pair": "AAAUSDT/BBBUSDT",
                "asset_x": "AAAUSDT",
                "asset_y": "BBBUSDT",
                "interval": "daily",
                "period": 365,
                "exact_mode": "Static (Spread)",
                "capture_confirmed": True,
                "backtest_settings_complete": True,
                "capture_evidence_path": "data/raw/wizard/aaa_bbb_static.png",
                "entry_long_operator": "<=",
                "entry_long_value": -2.0,
                "entry_long_position": "long_x_short_y",
                "entry_short_operator": ">=",
                "entry_short_value": 2.0,
                "entry_short_position": "short_x_long_y",
                "exit_long_operator": ">=",
                "exit_long_value": 0.0,
                "exit_short_operator": "<=",
                "exit_short_value": 0.0,
                "hedge_ratio": 1.2,
            }
        ]
    ).to_csv(active / "crypto_wizards_pair_page_capture_settings.csv", index=False)

    result = build_wizard_mode_replay_capability(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_mode_replay_capability"])
    static = frame.loc[frame["pair"].eq("AAAUSDT/BBBUSDT")].iloc[0]
    copula = frame.loc[frame["pair"].eq("CCCUSDT/DDDUSDT")].iloc[0]

    assert static["mode_capability_status"] == "READY_FOR_RESEARCH_MODE_REPLAY"
    assert bool(static["research_replay_eligible"])
    assert static["mode_fidelity_status"] == "local_formula_approximation"
    assert not bool(static["acceptance_eligible"])
    assert not bool(static["paper_or_execution_eligible"])
    assert copula["mode_capability_status"] == "BLOCKED_SETTINGS_CAPTURE"
    assert "copula_family" in copula["missing_mode_inputs"]


def test_pair_settings_capture_rejects_invalid_leg_direction_mapping(tmp_path):
    captured = {column: "" for column in WIZARD_PAIR_SETTINGS_CAPTURE_TEMPLATE_COLUMNS}
    captured.update(
        {
            "pair": "AAAUSDT/BBBUSDT",
            "asset_x": "AAAUSDT",
            "asset_y": "BBBUSDT",
            "exchange": "binance",
            "interval": "daily",
            "period": 365,
            "exact_mode": "Static (Spread)",
            "spread_id": 3,
            "strategy_id": 1,
            "capture_timestamp_utc": "2026-08-05T12:00:00Z",
            "capture_evidence_path": "data/raw/wizard/aaa_bbb_static_capture.png",
            "capture_confirmed": True,
            "entry_long_operator": "<=",
            "entry_long_value": -2.0,
            "entry_long_position": "made_up_direction",
            "entry_short_operator": ">=",
            "entry_short_value": 2.0,
            "entry_short_position": "short_x_long_y",
            "exit_long_operator": ">=",
            "exit_long_value": 0.0,
            "exit_short_operator": "<=",
            "exit_short_value": 0.0,
            "capital_weighting_slider_value": 0.5,
            "commission_rate": 0.0002,
            "slippage_rate": 0.0004,
            "hedge_ratio": 1.0,
        }
    )
    input_path = tmp_path / "invalid_direction_capture.csv"
    pd.DataFrame([captured], columns=WIZARD_PAIR_SETTINGS_CAPTURE_TEMPLATE_COLUMNS).to_csv(input_path, index=False)

    result = import_wizard_pair_settings_capture(input_path, root=tmp_path)
    validation = pd.read_csv(result.paths["wizard_pair_settings_capture_validation"])

    assert result.summary == {"input_rows": 1, "accepted_rows": 0, "blocked_rows": 1}
    assert "missing_invalid_entry_direction_mapping" in validation.loc[0, "validation_blockers"]


def test_mode_matrix_capture_queue_expands_a_fresh_pair_into_all_seven_modes(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "setup_identity": "AAAUSDT|BBBUSDT|daily|365|static_(spread)",
                "pair": "AAAUSDT/BBBUSDT",
                "asset_x": "AAAUSDT",
                "asset_y": "BBBUSDT",
                "exchange": "binance",
                "interval": "daily",
                "period": 365,
                "exact_mode": "Static (Spread)",
                "sharpe": 2.4,
                "returns_total": 0.31,
                "returns_total_pct": 31.0,
                "research_quality_status": "RESEARCH_READY",
                "source_fresh": True,
                "source_timestamp": "2026-08-05T12:00:00Z",
                "evidence_path": "reports/active/wizard_discovery_shortlist.csv",
            }
        ]
    ).to_csv(active / "wizard_discovery_shortlist.csv", index=False)

    result = build_wizard_mode_matrix_capture_queue(root=tmp_path)
    queue = pd.read_csv(result.paths["wizard_mode_matrix_capture_queue"])
    template = build_wizard_pair_settings_capture_template(root=tmp_path)
    template_frame = pd.read_csv(template.paths["wizard_pair_settings_capture_template"])

    assert result.summary == {"rows": 7, "pairs": 1, "capture_required": 7}
    assert set(queue["exact_mode"]) == {
        "Static (Spread)",
        "Static (ZScoreR)",
        "Dyn (Spread)",
        "Dyn (ZScoreR)",
        "OU (Spread)",
        "OU (ZScoreR)",
        "Copula",
    }
    assert queue["mode_switch_required"].sum() == 6
    assert not queue["acceptance_eligible"].astype(bool).any()
    assert "copula_family" in queue.loc[queue["exact_mode"].eq("Copula"), "required_settings"].iloc[0]
    assert len(template_frame) == 7


def test_missing_exact_mode_is_blocked(tmp_path):
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True)
    _write_json(
        pair_dir / "pair_SOL-USD_WLD-USD_wizard.json",
        {
            "pair": "SOL-USD/WLD-USD",
            "asset_x": "SOL-USD",
            "asset_y": "WLD-USD",
            "interval": "daily",
            "sharpe": 3.1,
            "returns_total": 0.42,
        },
    )

    result = build_wizard_evidence(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_evidence"])

    assert len(frame) == 1
    assert not bool(frame["mode_valid"].iloc[0])
    assert frame["mode_blocker"].iloc[0] == "missing_exact_mode"


def test_dashboard_recommended_strategy_maps_to_exact_mode_and_local_strategy(tmp_path):
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True)
    _write_json(
        pair_dir / "pair_SOL-USD_WLD-USD_wizard.json",
        {
            "pair": "SOL-USD/WLD-USD",
            "asset_x": "SOL-USD",
            "asset_y": "WLD-USD",
            "interval": "daily",
            "dashboard_recommended_strategy": "Kalman Dynamic Spread",
            "sharpe": 3.1,
            "returns_total": 0.42,
        },
    )

    result = build_wizard_evidence(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_evidence"])

    assert bool(frame["mode_valid"].iloc[0])
    assert frame["exact_mode"].iloc[0] == "Dyn (Spread)"
    assert int(frame["local_strategy_id"].iloc[0]) == 1
    assert frame["local_strategy_name"].iloc[0] == "Classic ZScore Mean Reversion"


def test_pair_page_capture_source_outranks_generic_active_rows(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BLUR-USD/ETHFI-USD",
                "asset_x": "BLUR-USD",
                "asset_y": "ETHFI-USD",
                "interval": "daily",
                "dashboard_recommended_strategy": "Copula",
                "exact_mode": "Copula",
                "spread_id": 1,
                "strategy_id": 3,
                "sharpe": 1.9,
                "returns_total": 0.54,
                "source_path": "reports/active/crypto_wizards_next_best_sharpe_returns_queue.csv",
            }
        ]
    ).to_csv(active / "crypto_wizards_next_best_sharpe_returns_queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BLUR-USD/ETHFI-USD",
                "asset_x": "BLUR-USD",
                "asset_y": "ETHFI-USD",
                "interval": "daily",
                "dashboard_recommended_strategy": "Copula",
                "exact_mode": "Copula",
                "spread_id": 1,
                "strategy_id": 3,
                "pearson": 0.638,
                "spearman": 0.352,
                "kendall": 0.234,
                "copula": "gaussian",
                "corr_copula": 0.871,
                "u1_given_u2": 0.162,
                "u2_given_u1": 0.541,
                "sharpe": 1.9,
                "returns_total": 0.54,
                "source_path": "reports/active/crypto_wizards_pair_page_capture.csv",
            }
        ]
    ).to_csv(active / "crypto_wizards_pair_page_capture.csv", index=False)

    result = build_wizard_evidence(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_evidence"])

    assert len(frame) == 1
    assert frame["source_system"].iloc[0] == "crypto_wizards_pair_page_capture"
    assert float(frame["pearson"].iloc[0]) == 0.638
    assert frame["exact_mode"].iloc[0] == "Copula"


def test_pair_page_capture_can_carry_ecm_availability_into_diagnostics(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BLUR-USD/ETHFI-USD",
                "asset_x": "BLUR-USD",
                "asset_y": "ETHFI-USD",
                "interval": "daily",
                "dashboard_recommended_strategy": "Copula",
                "exact_mode": "Copula",
                "spread_id": 1,
                "strategy_id": 3,
                "pearson": 0.638,
                "spearman": 0.352,
                "kendall": 0.234,
                "copula": "gaussian",
                "corr_copula": 0.871,
                "u1_given_u2": 0.162,
                "u2_given_u1": 0.541,
                "ecm_x_available": True,
                "ecm_y_available": True,
                "ecm_strength_available": True,
                "source_path": "reports/active/crypto_wizards_pair_page_capture.csv",
            }
        ]
    ).to_csv(active / "crypto_wizards_pair_page_capture.csv", index=False)

    evidence = build_wizard_evidence(root=tmp_path)
    diagnostics = build_wizard_diagnostic_confirmation(root=tmp_path)
    evidence_frame = pd.read_csv(evidence.paths["wizard_evidence"])
    diagnostics_frame = pd.read_csv(diagnostics.paths["diagnostics"])

    assert bool(evidence_frame["ecm_x_available"].iloc[0])
    assert bool(evidence_frame["ecm_y_available"].iloc[0])
    assert bool(evidence_frame["ecm_strength_available"].iloc[0])
    assert diagnostics_frame["ecm_status"].iloc[0] == "confirmed_ecm"


def test_hourly_pair_panel_capture_becomes_canonical_wizard_evidence(tmp_path):
    hourly = tmp_path / "reports" / "active" / "wizard_hourly_database"
    hourly.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "scan_timestamp": "2026-07-04T05:18:57+00:00",
                "source_name": "wizard_scanner_live_dom",
                "pair": "DOGE-USD/EIGEN-USD",
                "periods": 320,
                "timeframe": "Daily",
                "strategy": "Static (Spread)",
                "top_hurst": 0.55,
                "top_half_life": 10.7,
                "top_corr": 82.0,
                "annualized_return_pct": 2.0,
                "sharpe": 0.1,
            }
        ]
    ).to_csv(hourly / "wizard_hourly_current_alias.csv", index=False)

    result = build_wizard_evidence(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_evidence"])

    assert len(frame) == 1
    assert frame["source_system"].iloc[0] == "crypto_wizards_hourly_pair_panel"
    assert frame["exact_mode"].iloc[0] == "Static (Spread)"
    assert frame["dashboard_recommended_strategy"].iloc[0] == "Static (Spread)"
    assert float(frame["hurst"].iloc[0]) == 0.55
    assert float(frame["half_life"].iloc[0]) == 10.7
    assert float(frame["pearson"].iloc[0]) == 0.82


def test_hourly_pair_panel_capture_outranks_missing_exact_mode_active_row(tmp_path):
    active = tmp_path / "reports" / "active"
    hourly = active / "wizard_hourly_database"
    hourly.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "DOGE-USD/EIGEN-USD",
                "asset_x": "DOGE-USD",
                "asset_y": "EIGEN-USD",
                "interval": "daily",
                "sharpe": 0.17,
                "returns_total": 0.23,
                "source_path": "reports/active/crypto_wizards_next_best_sharpe_returns_queue.csv",
            }
        ]
    ).to_csv(active / "crypto_wizards_next_best_sharpe_returns_queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "scan_timestamp": "2026-07-04T05:18:57+00:00",
                "source_name": "wizard_scanner_live_dom",
                "pair": "DOGE-USD/EIGEN-USD",
                "periods": 320,
                "timeframe": "Daily",
                "strategy": "Static (Spread)",
                "top_hurst": 0.55,
                "top_half_life": 10.7,
                "top_corr": 82.0,
                "annualized_return_pct": 2.0,
                "sharpe": 0.1,
            }
        ]
    ).to_csv(hourly / "wizard_hourly_current_alias.csv", index=False)

    result = build_wizard_evidence(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_evidence"])

    assert len(frame) == 1
    assert bool(frame["mode_valid"].iloc[0])
    assert frame["exact_mode"].iloc[0] == "Static (Spread)"
    assert frame["source_system"].iloc[0] == "crypto_wizards_hourly_pair_panel"


def test_copula_mode_does_not_require_mean_reversion_fields(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BLUR-USD/ETHFI-USD",
                "asset_x": "BLUR-USD",
                "asset_y": "ETHFI-USD",
                "interval": "daily",
                "dashboard_recommended_strategy": "Copula",
                "exact_mode": "Copula",
                "spread_id": 1,
                "strategy_id": 3,
                "pearson": 0.638,
                "spearman": 0.352,
                "kendall": 0.234,
                "copula": "gaussian",
                "corr_copula": 0.871,
                "u1_given_u2": 0.162,
                "u2_given_u1": 0.541,
            }
        ]
    ).to_csv(active / "crypto_wizards_pair_page_capture.csv", index=False)

    diagnostics = build_wizard_diagnostic_confirmation(root=tmp_path)
    frame = pd.read_csv(diagnostics.paths["diagnostics"])

    assert frame["mean_reversion_status"].iloc[0] == "copula_mode_no_mean_reversion_required"
    assert "missing_mean_reversion" not in frame["diagnostic_blocker"].iloc[0]


def test_hourly_pair_panel_proxy_metrics_flow_into_diagnostics(tmp_path):
    hourly = tmp_path / "reports" / "active" / "wizard_hourly_database"
    hourly.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "scan_timestamp": "2026-07-04T05:18:57+00:00",
                "source_name": "wizard_scanner_live_dom",
                "pair": "DOGE-USD/EIGEN-USD",
                "periods": 320,
                "timeframe": "Daily",
                "strategy": "Static (Spread)",
                "top_hurst": 0.55,
                "top_half_life": 10.7,
                "top_corr": 82.0,
                "annualized_return_pct": 2.0,
                "sharpe": 0.1,
            }
        ]
    ).to_csv(hourly / "wizard_hourly_current_alias.csv", index=False)

    diagnostics = build_wizard_diagnostic_confirmation(root=tmp_path)
    frame = pd.read_csv(diagnostics.paths["diagnostics"])

    assert frame["correlation_status"].iloc[0] == "partial_correlation"
    assert frame["mean_reversion_status"].iloc[0] == "confirmed_mean_reversion"
    assert "missing_mean_reversion" not in frame["diagnostic_blocker"].iloc[0]


def test_hourly_pair_queue_top_strategy_becomes_primary_setup(tmp_path):
    hourly = tmp_path / "reports" / "active" / "wizard_hourly_database"
    hourly.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "scan_timestamp": "2026-07-04T05:18:57+00:00",
                "source_name": "wizard_scanner_live_dom",
                "pair": "DOGE-USD/EIGEN-USD",
                "periods": 320,
                "timeframe": "Daily",
                "strategy": "Copula",
                "top_hurst": 0.55,
                "top_half_life": 10.7,
                "top_corr": 82.0,
            },
            {
                "scan_timestamp": "2026-07-04T05:18:57+00:00",
                "source_name": "wizard_scanner_live_dom",
                "pair": "DOGE-USD/EIGEN-USD",
                "periods": 320,
                "timeframe": "Daily",
                "strategy": "Static (Spread)",
                "top_hurst": 0.55,
                "top_half_life": 10.7,
                "top_corr": 82.0,
            },
        ]
    ).to_csv(hourly / "wizard_hourly_current_alias.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "DOGE-USD/EIGEN-USD",
                "top_strategy": "Static (Spread)",
                "top_timeframe": "Daily",
                "top_annualized_return_pct": 2.0,
                "top_sharpe": 0.1,
                "pair_status": "review_now",
                "hourly_rank": 1,
                "recommended_this_hour": True,
            }
        ]
    ).to_csv(hourly / "wizard_hourly_pair_queue.csv", index=False)

    result = build_wizard_evidence(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_evidence"])
    pair_rows = frame[frame["pair"].astype(str).str.contains("DOGE-USD", na=False)].sort_values("setup_rank_within_pair")

    assert pair_rows.iloc[0]["exact_mode"] == "Static (Spread)"
    assert bool(pair_rows.iloc[0]["primary_wizard_setup"])


def test_hourly_pair_panel_rows_backfill_ecm_and_mean_reversion_from_local_history(tmp_path):
    hourly = tmp_path / "reports" / "active" / "wizard_hourly_database"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    hourly.mkdir(parents=True)
    pair_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "scan_timestamp": "2026-07-04T05:18:57+00:00",
                "source_name": "wizard_scanner_live_dom",
                "pair": "DOGE-USD/MORPHO-USD",
                "periods": 320,
                "timeframe": "Daily",
                "strategy": "Static (Spread)",
                "annualized_return_pct": 22.8,
                "sharpe": 1.4,
            }
        ]
    ).to_csv(hourly / "wizard_hourly_current_alias.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "DOGE-USD/MORPHO-USD",
                "top_strategy": "Static (Spread)",
                "top_timeframe": "Daily",
                "top_annualized_return_pct": 22.8,
                "top_sharpe": 1.4,
                "pair_status": "review_now",
                "hourly_rank": 1,
                "recommended_this_hour": True,
            }
        ]
    ).to_csv(hourly / "wizard_hourly_pair_queue.csv", index=False)
    _write_json(
        pair_dir / "pair_doge_morpho_5mins_dydx_candles_derived_history.json",
        {
            "pair": "DOGE-USD-MORPHO-USD",
            "asset_x": "DOGE-USD",
            "asset_y": "MORPHO-USD",
            "exchange": "dydx",
            "interval": "5min",
            "ecm_x_available": True,
            "ecm_y_available": True,
            "ecm_strength_available": True,
            "history": [
                {"price_x": 1.0, "price_y": 2.0, "ecm_x": 0.1, "ecm_y": 0.2, "ecm_strength": 0.3, "hurst": 0.47, "half_life": 6.4, "hedge_ratio": 0.2},
                {"price_x": 2.0, "price_y": 4.0, "ecm_x": 0.1, "ecm_y": 0.2, "ecm_strength": 0.3, "hurst": 0.47, "half_life": 6.4, "hedge_ratio": 0.2},
                {"price_x": 3.0, "price_y": 6.0, "ecm_x": 0.1, "ecm_y": 0.2, "ecm_strength": 0.3, "hurst": 0.47, "half_life": 6.4, "hedge_ratio": 0.2},
            ],
        },
    )

    diagnostics = build_wizard_diagnostic_confirmation(root=tmp_path)
    frame = pd.read_csv(diagnostics.paths["diagnostics"])
    pair_rows = frame[frame["pair"].astype(str).str.contains("DOGE-USD", na=False)]
    primary = pair_rows.iloc[0]

    assert primary["correlation_status"] == "confirmed_correlation"
    assert primary["ecm_status"] == "confirmed_ecm"
    assert primary["mean_reversion_status"] == "confirmed_mean_reversion"


def test_collected_live_scanner_capture_becomes_wizard_evidence(tmp_path):
    collected = tmp_path / "data" / "collected" / "wizard_scanner_static_spread_visible_live"
    collected.mkdir(parents=True)
    payload = {
        "captured_at": "2026-07-04T19:04:00Z",
        "source": "in_app_browser_live_scanner_visible_top",
        "rows": [
            {
                "row_index": 1,
                "pair": "DOGE-USD / SYRUP-USD",
                "pair_x": "DOGE-USD",
                "pair_y": "SYRUP-USD",
                "updated": "2026-Jul-04 10:17:38 UTC",
                "strategy": "staticspread",
                "stationarity": "89.3%corr | JnEG | 0.80hurst | 15.2half life | 280σ32σ",
                "reward": "21.6%return | 2.03sharpe",
            }
        ],
    }
    (collected / "wizard_scanner_static_spread_visible_live_latest.json").write_text(
        __import__("json").dumps(payload),
        encoding="utf-8",
    )

    result = build_wizard_evidence(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_evidence"])

    assert len(frame) == 1
    assert frame["source_system"].iloc[0] == "crypto_wizards_live_scanner_capture"
    assert frame["exact_mode"].iloc[0] == "Static (Spread)"
    assert frame["dashboard_recommended_strategy"].iloc[0] == "Static (Spread)"
    assert float(frame["sharpe"].iloc[0]) == 2.03
    assert float(frame["returns_total_pct"].iloc[0]) == 21.6


def test_strategy_alignment_report_emits_source_first_action(tmp_path):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "SOL-USD/WLD-USD",
                "interval": "daily",
                "dashboard_recommended_strategy": "Copula",
                "exact_mode": "Copula",
                "spread_id": 1,
                "strategy_id": 3,
                "local_strategy_id": 5,
                "local_strategy_name": "Pure Copula",
                "local_strategy_family": "copula",
                "strategy_mapping_status": "mapped_from_exact_mode",
                "evidence_path": "wizard",
            }
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)

    result = build_wizard_strategy_alignment_report(root=tmp_path)
    frame = pd.read_csv(result.paths["alignment"])

    assert frame["source_first_action"].iloc[0] == "run_wizard_primary_setup_first:5"


def test_primary_wizard_setup_rank_is_first_class(tmp_path):
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True)
    _write_json(
        pair_dir / "pair_BLUR-USD_WLD-USD_static.json",
        {
            "pair": "BLUR-USD/WLD-USD",
            "asset_x": "BLUR-USD",
            "asset_y": "WLD-USD",
            "interval": "daily",
            "period": 320,
            "dashboard_pair_rank": 1,
            "dashboard_selector_score": 1.9,
            "spread_id": 3,
            "strategy_id": 1,
            "sharpe": 1.9,
            "returns_total": 0.535,
        },
    )
    _write_json(
        pair_dir / "pair_BLUR-USD_WLD-USD_copula.json",
        {
            "pair": "BLUR-USD/WLD-USD",
            "asset_x": "BLUR-USD",
            "asset_y": "WLD-USD",
            "interval": "daily",
            "period": 320,
            "dashboard_pair_rank": 2,
            "dashboard_selector_score": 1.4,
            "spread_id": 1,
            "strategy_id": 3,
            "sharpe": 1.4,
            "returns_total": 0.424,
        },
    )

    result = build_wizard_evidence(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_evidence"])

    primary = frame.loc[frame["primary_wizard_setup"].astype(bool)].iloc[0]
    alternate = frame.loc[~frame["primary_wizard_setup"].astype(bool)].iloc[0]

    assert primary["exact_mode"] == "Static (Spread)"
    assert int(primary["setup_rank_within_pair"]) == 1
    assert primary["setup_role"] == "primary"
    assert "BLUR-USD|WLD-USD|daily|320|static_(spread)" == primary["setup_identity"]
    assert alternate["setup_role"] == "alternate"
    assert int(alternate["setup_rank_within_pair"]) == 2


def test_exact_mode_pair_detail_outranks_placeholder_active_report_for_same_pair(tmp_path):
    active = tmp_path / "reports" / "active"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    active.mkdir(parents=True)
    pair_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "pair": "EUR-USD/FIL-USD",
                "asset_x": "EUR-USD",
                "asset_y": "FIL-USD",
                "interval": "daily",
                "dashboard_pair_rank": 1,
                "sharpe": 2.77,
                "returns_total": 0.297,
                "source_path": "reports/active/crypto_wizards_next_best_sharpe_returns_queue.csv",
            }
        ]
    ).to_csv(active / "crypto_wizards_next_best_sharpe_returns_queue.csv", index=False)

    _write_json(
        pair_dir / "pair_EUR-USD_FIL-USD_Dydx_Daily_365_daily_cw_backtest_history.json",
        {
            "pair": "EUR-USD/FIL-USD",
            "asset_x": "EUR-USD",
            "asset_y": "FIL-USD",
            "exchange": "dydx",
            "interval": "daily",
            "period": 365,
            "strategy_mode": "static",
            "sharpe": 2.7784205720334625,
            "returns_total": 0.29746466038575536,
        },
    )
    _write_json(
        pair_dir / "pair_eur_fil_cw_daily_sharpe_1day_dydx_long_history_derived_history.json",
        {
            "pair": "EUR-USD-FIL-USD",
            "asset_x": "EUR-USD",
            "asset_y": "FIL-USD",
            "exchange": "dydx",
            "interval": "1day",
            "period": 581,
            "strategy_mode": "static",
            "sharpe": 2.7784205720334625,
            "returns_total": 0.29746466038575536,
        },
    )

    result = build_wizard_evidence(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_evidence"])

    eur_fil = frame.loc[frame["asset_x"].astype(str).eq("EUR-USD") & frame["asset_y"].astype(str).eq("FIL-USD")]
    primary = eur_fil.loc[eur_fil["primary_wizard_setup"].astype(bool)].iloc[0]

    assert len(eur_fil) == 1
    assert primary["pair"] == "EUR-USD/FIL-USD"
    assert primary["exact_mode"] == "Static (Spread)"
    assert bool(primary["mode_valid"])
    assert primary["setup_role"] == "primary"
    assert "1day" not in set(eur_fil["interval"].astype(str))


def test_derived_history_backfills_correlation_and_ecm_on_exact_setup_row(tmp_path):
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True)

    _write_json(
        pair_dir / "pair_EUR-USD_FIL-USD_Dydx_Daily_365_daily_cw_backtest_history.json",
        {
            "pair": "EUR-USD/FIL-USD",
            "asset_x": "EUR-USD",
            "asset_y": "FIL-USD",
            "exchange": "dydx",
            "interval": "daily",
            "period": 365,
            "strategy_mode": "static",
            "sharpe": 2.77,
            "returns_total": 0.297,
        },
    )
    _write_json(
        pair_dir / "pair_eur_fil_cw_daily_sharpe_1day_dydx_long_history_derived_history.json",
        {
            "pair": "EUR-USD-FIL-USD",
            "asset_x": "EUR-USD",
            "asset_y": "FIL-USD",
            "exchange": "dydx",
            "interval": "1day",
            "period": 581,
            "strategy_mode": "static",
            "ecm_x_available": True,
            "ecm_y_available": True,
            "ecm_strength_available": True,
            "history": [
                {"price_x": 1.0, "price_y": 2.0, "ecm_x": 0.1, "ecm_y": 0.2, "ecm_strength": 0.3},
                {"price_x": 2.0, "price_y": 4.0, "ecm_x": 0.1, "ecm_y": 0.2, "ecm_strength": 0.3},
                {"price_x": 3.0, "price_y": 6.0, "ecm_x": 0.1, "ecm_y": 0.2, "ecm_strength": 0.3},
            ],
        },
    )

    result = build_wizard_evidence(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_evidence"])
    eur_fil = frame.loc[frame["asset_x"].astype(str).eq("EUR-USD") & frame["asset_y"].astype(str).eq("FIL-USD")].iloc[0]

    assert eur_fil["exact_mode"] == "Static (Spread)"
    assert bool(eur_fil["mode_valid"])
    assert float(eur_fil["pearson"]) == 1.0
    assert float(eur_fil["spearman"]) == 1.0
    assert float(eur_fil["kendall"]) == 1.0
    assert bool(eur_fil["ecm_x_available"])
    assert bool(eur_fil["ecm_y_available"])
    assert bool(eur_fil["ecm_strength_available"])


def test_timestamp_missing_wizard_evidence_is_stale_before_local_data_is_considered(tmp_path):
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True)
    _write_json(
        pair_dir / "pair_SOL-USD_WLD-USD_wizard.json",
        {
            "pair": "SOL-USD/WLD-USD",
            "asset_x": "SOL-USD",
            "asset_y": "WLD-USD",
            "interval": "daily",
            "spread_id": 2,
            "strategy_id": 1,
            "sharpe": 3.1,
            "returns_total": 0.42,
        },
    )

    result = build_wizard_hypotheses(root=tmp_path)
    frame = pd.read_csv(result.paths["hypotheses"])

    assert frame["hypothesis_status"].iloc[0] == "STALE_EVIDENCE"
    assert frame["hypothesis_reason"].iloc[0] == "source_timestamp_missing"


def test_pair_page_percent_return_is_normalized_without_losing_raw_value(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BNB-USD/WLD-USD",
                "asset_x": "BNB-USD",
                "asset_y": "WLD-USD",
                "interval": "daily",
                "exact_mode": "Static (Spread)",
                "spread_id": 3,
                "strategy_id": 1,
                "sharpe": 2.82,
                "returns_total": "70.8%",
                "returns_total_pct": "84.2%",
                "capture_timestamp_utc": "2026-08-05T12:00:00Z",
            }
        ]
    ).to_csv(active / "crypto_wizards_pair_page_capture.csv", index=False)

    result = build_wizard_evidence(root=tmp_path, now=datetime(2026, 8, 5, 12, tzinfo=timezone.utc))
    frame = pd.read_csv(result.paths["wizard_evidence"])

    assert frame["returns_total_raw"].iloc[0] == "70.8%"
    assert frame["returns_total_unit"].iloc[0] == "percent_text"
    assert float(frame["returns_total"].iloc[0]) == 0.708
    assert float(frame["returns_total_pct"].iloc[0]) == 70.8


def test_fresh_scanner_evidence_is_discovery_only_even_when_gates_pass(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    as_of = datetime(2026, 8, 5, 12, tzinfo=timezone.utc)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT/FIDAUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "FIDAUSDT",
                "exchange": "binance",
                "interval": "Daily",
                "exact_mode": "OU (Spread)",
                "sharpe": 2.4,
                "returns_total": 0.32,
                "capture_timestamp_utc": as_of.isoformat(),
            }
        ]
    ).to_csv(active / "crypto_wizards_live_scanner_capture.csv", index=False)

    build_wizard_evidence(root=tmp_path, now=as_of)
    result = build_wizard_hypotheses(root=tmp_path, now=as_of)
    frame = pd.read_csv(result.paths["hypotheses"])

    assert frame["source_authority"].iloc[0] == "discovery_only"
    assert bool(frame["source_fresh"].iloc[0])
    assert frame["hypothesis_status"].iloc[0] == "DISCOVERY_ONLY"


def test_stale_pair_page_capture_never_reaches_hypothesis_ready(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BNB-USD/WLD-USD",
                "asset_x": "BNB-USD",
                "asset_y": "WLD-USD",
                "interval": "daily",
                "exact_mode": "Static (Spread)",
                "spread_id": 3,
                "strategy_id": 1,
                "sharpe": 2.8,
                "returns_total": "70.8%",
                "capture_timestamp_utc": "2026-08-01T12:00:00Z",
            }
        ]
    ).to_csv(active / "crypto_wizards_pair_page_capture.csv", index=False)

    build_wizard_evidence(root=tmp_path, now=datetime(2026, 8, 5, 12, tzinfo=timezone.utc))
    result = build_wizard_hypotheses(root=tmp_path, now=datetime(2026, 8, 5, 12, tzinfo=timezone.utc))
    frame = pd.read_csv(result.paths["hypotheses"])

    assert frame["hypothesis_status"].iloc[0] == "STALE_EVIDENCE"
    assert frame["hypothesis_reason"].iloc[0] == "source_stale_over_24h"


def test_fresh_healthy_detail_requires_settings_and_matching_local_history_before_ready(tmp_path):
    active = tmp_path / "reports" / "active"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    active.mkdir(parents=True)
    pair_dir.mkdir(parents=True)
    as_of = datetime(2026, 8, 5, 12, tzinfo=timezone.utc)
    pd.DataFrame(
        [
            {
                "pair": "BNB-USD/WLD-USD",
                "asset_x": "BNB-USD",
                "asset_y": "WLD-USD",
                "interval": "daily",
                "exact_mode": "Static (Spread)",
                "spread_id": 3,
                "strategy_id": 1,
                "sharpe": 2.8,
                "returns_total": "28%",
                "capture_timestamp_utc": as_of.isoformat(),
                    "entry_long_operator": ">=",
                    "entry_long_value": 2.0,
                    "entry_long_position": "long_x_short_y",
                    "entry_short_operator": "<=",
                    "entry_short_value": -2.0,
                    "entry_short_position": "short_x_long_y",
                "exit_long_operator": "<=",
                "exit_long_value": 0.0,
                "exit_short_operator": ">=",
                "exit_short_value": 0.0,
                    "capital_weighting_slider_value": 0.5,
                    "hedge_ratio": 1.0,
                    "slippage_rate": 0.0004,
                "commission_rate": 0.0002,
            }
        ]
    ).to_csv(active / "crypto_wizards_pair_page_capture.csv", index=False)
    _write_json(
        pair_dir / "pair_BNB-USD_WLD-USD_daily_dydx_derived_history.json",
        {
            "pair": "BNB-USD/WLD-USD",
            "asset_x": "BNB-USD",
            "asset_y": "WLD-USD",
            "interval": "daily",
            "history": [{"price_x": 1.0, "price_y": 2.0}, {"price_x": 2.0, "price_y": 4.0}],
        },
    )

    build_wizard_evidence(root=tmp_path, now=as_of)
    result = build_wizard_hypotheses(root=tmp_path, now=as_of)
    frame = pd.read_csv(result.paths["hypotheses"])

    assert bool(frame["backtest_settings_complete"].iloc[0])
    assert bool(frame["local_data_available"].iloc[0])
    assert frame["hypothesis_status"].iloc[0] == "HYPOTHESIS_READY"


def test_merged_scanner_capture_matches_each_active_row_before_mode_enrichment(tmp_path):
    active = tmp_path / "reports" / "active"
    raw = tmp_path / "data" / "raw" / "crypto_wizards_scanner"
    active.mkdir(parents=True)
    raw.mkdir(parents=True)
    as_of = datetime(2026, 8, 5, 12, tzinfo=timezone.utc)
    capture = raw / "merged.json"
    _write_json(
        capture,
        [
            {
                "_row_index": 0,
                "_strategy_request": "Spread",
                "symbol_1": "ETHUSDT",
                "symbol_2": "TIAUSDC",
                "exchange": "binance",
                "interval": "daily",
                "spread_id": 3,
                "strategy_id": 1,
                "spread_type": "static",
                "strategy": "zscore",
            },
            {
                "_row_index": 1,
                "_strategy_request": "Copula",
                "symbol_1": "AVAXUSDT",
                "symbol_2": "NXPCUSDT",
                "exchange": "bybit",
                "interval": "daily",
                "spread_id": 1,
                "strategy_id": 3,
                "spread_type": "dynamic",
                "strategy": "copula",
            },
        ],
    )
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT/TIAUSDC",
                "asset_x": "ETHUSDT",
                "asset_y": "TIAUSDC",
                "exchange": "binance",
                "interval": "Daily",
                "api_strategy": "Spread",
                "api_rank": 1,
                "exact_mode": "Static (Spread)",
                "sharpe": 2.1,
                "returns_total": 0.25,
                "capture_timestamp_utc": as_of.isoformat(),
                "source_path": str(capture.relative_to(tmp_path)),
            },
            {
                "pair": "AVAXUSDT/NXPCUSDT",
                "asset_x": "AVAXUSDT",
                "asset_y": "NXPCUSDT",
                "exchange": "bybit",
                "interval": "Daily",
                "api_strategy": "Copula",
                "api_rank": 2,
                "exact_mode": "Dyn (Copula)",
                "sharpe": 2.2,
                "returns_total": 0.26,
                "capture_timestamp_utc": as_of.isoformat(),
                "source_path": str(capture.relative_to(tmp_path)),
            },
        ]
    ).to_csv(active / "crypto_wizards_live_scanner_capture.csv", index=False)

    result = build_wizard_evidence(root=tmp_path, now=as_of)
    frame = pd.read_csv(result.paths["wizard_evidence"]).set_index("pair")

    assert frame.loc["ETHUSDT/TIAUSDC", "exact_mode"] == "Static (Spread)"
    assert frame.loc["AVAXUSDT/NXPCUSDT", "exact_mode"] == "Copula"
    assert int(frame.loc["ETHUSDT/TIAUSDC", "spread_id"]) == 3
    assert int(frame.loc["AVAXUSDT/NXPCUSDT", "strategy_id"]) == 3


def test_discovery_sharpe_gate_is_one_point_seventy_five_or_above(tmp_path):
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True)
    _write_json(
        pair_dir / "pair_BLUR-USD_ETHFI-USD_wizard.json",
        {
            "pair": "BLUR-USD/ETHFI-USD",
            "asset_x": "BLUR-USD",
            "asset_y": "ETHFI-USD",
            "interval": "daily",
            "spread_id": 3,
            "strategy_id": 1,
            "sharpe": DISCOVERY_MIN_SHARPE,
            "returns_total": 0.21,
        },
    )

    result = build_wizard_evidence(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_evidence"])

    assert bool(frame["passes_sharpe_gate"].iloc[0])
    assert bool(frame["passes_sharpe_gt_2"].iloc[0])
    assert float(frame["discovery_min_sharpe"].iloc[0]) == DISCOVERY_MIN_SHARPE


def test_discovery_returns_gate_is_above_ten_percent(tmp_path):
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True)
    _write_json(
        pair_dir / "pair_ETH-USD_FIL-USD_wizard.json",
        {
            "pair": "ETH-USD/FIL-USD",
            "asset_x": "ETH-USD",
            "asset_y": "FIL-USD",
            "interval": "daily",
            "spread_id": 3,
            "strategy_id": 1,
            "sharpe": 2.0,
            "returns_total": DISCOVERY_MIN_RETURNS_TOTAL + 0.001,
        },
    )

    result = build_wizard_evidence(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_evidence"])

    assert bool(frame["passes_returns_total_gt_20pct"].iloc[0])
    assert float(frame["discovery_min_returns_total"].iloc[0]) == DISCOVERY_MIN_RETURNS_TOTAL


def test_missing_ecm_creates_diagnostic_blocker(tmp_path):
    evidence_dir = tmp_path / "data" / "processed"
    evidence_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "SOL-USD/WLD-USD",
                "interval": "daily",
                "exact_mode": "OU (Spread)",
                "pearson": 0.8,
                "spearman": 0.76,
                "kendall": 0.6,
                "hurst": 0.42,
                "half_life": 12,
                "ecm_x_available": False,
                "ecm_y_available": False,
                "ecm_strength_available": False,
                "evidence_path": "wizard",
            }
        ]
    ).to_csv(evidence_dir / "wizard_evidence.csv", index=False)

    result = build_wizard_diagnostic_confirmation(root=tmp_path)
    frame = pd.read_csv(result.paths["diagnostics"])

    assert frame["ecm_status"].iloc[0] == "missing_ecm"
    assert "missing_ecm" in frame["diagnostic_blocker"].iloc[0]


def test_local_parity_flags_zscore_mismatch(tmp_path):
    processed = tmp_path / "data" / "processed"
    raw = tmp_path / "data" / "raw" / "pair_details"
    processed.mkdir(parents=True)
    raw.mkdir(parents=True)
    wizard_path = raw / "pair_SOL-USD_WLD-USD_wizard.json"
    local_path = raw / "pair_SOL-USD_WLD-USD_daily_dydx_derived_history.json"
    _write_json(
        wizard_path,
        {
            "pair": "SOL-USD/WLD-USD",
            "asset_x": "SOL-USD",
            "asset_y": "WLD-USD",
            "interval": "daily",
            "history": [{"zscore": 2.4, "rolling_zscore": 2.2}],
        },
    )
    _write_json(
        local_path,
        {
            "pair": "SOL-USD/WLD-USD",
            "asset_x": "SOL-USD",
            "asset_y": "WLD-USD",
            "interval": "daily",
            "history": [{"zscore": 0.1, "rolling_zscore": 0.0}],
        },
    )
    pd.DataFrame(
        [
            {
                "pair": "SOL-USD/WLD-USD",
                "asset_x": "SOL-USD",
                "asset_y": "WLD-USD",
                "interval": "daily",
                "exact_mode": "OU (Spread)",
                "source_path": str(wizard_path),
            }
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)

    result = build_wizard_local_parity(root=tmp_path)
    frame = pd.read_csv(result.paths["parity"])

    assert frame["parity_status"].iloc[0] == "MISMATCH"


def test_exact_mode_capture_queue_keeps_only_high_sharpe_return_blockers(tmp_path):
    processed = tmp_path / "data" / "processed"
    raw = tmp_path / "data" / "raw" / "pair_details"
    processed.mkdir(parents=True)
    raw.mkdir(parents=True)
    source = raw / "pair_one.json"
    _write_json(source, {"url": "https://cryptowizards.net/wizards/zscore/pair/1?origin=scanner"})
    pd.DataFrame(
        [
            {
                "pair": "BNB-USD/STX-USD",
                "asset_x": "BNB-USD",
                "asset_y": "STX-USD",
                "interval": "daily",
                "sharpe": 3.0,
                "returns_total": 0.54,
                "returns_total_pct": 54.0,
                "mode_valid": False,
                "passes_sharpe_gt_2": True,
                "passes_returns_total_gt_20pct": True,
                "mode_blocker": "missing_exact_mode",
                "source_path": str(source),
                "evidence_path": str(source),
            },
            {
                "pair": "LOW-USD/RET-USD",
                "asset_x": "LOW-USD",
                "asset_y": "RET-USD",
                "interval": "daily",
                "sharpe": 1.5,
                "returns_total": 0.54,
                "returns_total_pct": 54.0,
                "mode_valid": False,
                "passes_sharpe_gt_2": False,
                "passes_returns_total_gt_20pct": True,
                "mode_blocker": "missing_exact_mode",
                "source_path": "",
                "evidence_path": "wizard",
            },
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)

    result = build_wizard_exact_mode_capture_queue(root=tmp_path)
    frame = pd.read_csv(result.paths["exact_mode_capture_queue"])

    assert frame["pair"].tolist() == ["BNB-USD/STX-USD"]
    assert frame["pair_page_url"].iloc[0].endswith("/pair/1?origin=scanner")
    assert "selected_strategy_value" in frame["required_capture_fields"].iloc[0]


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
