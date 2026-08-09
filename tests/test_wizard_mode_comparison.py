from __future__ import annotations

import json

import pandas as pd

from quant_platform.wizard_mode_comparison import build_wizard_mode_comparison


def _captured_settings(mode: str) -> dict[str, object]:
    return {
        "pair": "AAAUSDT/BBBUSDT",
        "asset_x": "AAAUSDT",
        "asset_y": "BBBUSDT",
        "exchange": "binance",
        "interval": "daily",
        "period": 12,
        "exact_mode": mode,
        "capture_confirmed": True,
        "backtest_settings_complete": True,
        "capture_evidence_path": f"data/raw/wizard/{mode.replace(' ', '_')}.png",
        "entry_long_operator": "<=",
        "entry_long_value": -2.85,
        "entry_long_position": "long_x_short_y",
        "entry_short_operator": ">=",
        "entry_short_value": -2.65,
        "entry_short_position": "short_x_long_y",
        "exit_long_operator": ">=",
        "exit_long_value": -2.75,
        "exit_short_operator": "<=",
        "exit_short_value": -2.75,
        "hedge_ratio": 1.9,
        "x_weighting": 0.4,
        "y_weighting": 0.6,
        "zscore_window": 3,
        "dynamic_hedge_ratio_method": "rolling_ols_log_prices",
        "dynamic_hedge_ratio_window": 3,
        "ou_mu": -2.8,
        "ou_sigma": 0.1,
    }


def _write_history(
    path,
    *,
    asset_x: str = "AAAUSDT",
    asset_y: str = "BBBUSDT",
    exchange: str = "binance",
    interval: str = "daily",
):
    rows = []
    for index, (price_x, price_y) in enumerate(
        [(100, 50), (103, 50), (106, 51), (104, 51), (101, 50), (98, 49), (96, 49), (98, 50), (101, 50), (104, 51), (102, 50), (100, 50)]
    ):
        rows.append(
            {
                "timestamp": f"2026-08-{index + 1:02d}T00:00:00Z",
                "price_x": price_x,
                "price_y": price_y,
                "hedge_ratio": 1.9,
                "beta": 1.0,
                "funding_bps_per_day": 0.0,
                "funding_x_bps": 1.0,
                "funding_y_bps": -2.0,
            }
        )
    path.write_text(
        json.dumps(
            {
                "asset_x": asset_x,
                "asset_y": asset_y,
                "exchange": exchange,
                "interval": interval,
                "history": rows,
            }
        ),
        encoding="utf-8",
    )


def test_mode_comparison_runs_captured_modes_with_provisional_cost_cases(tmp_path):
    active = tmp_path / "reports" / "active"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    active.mkdir(parents=True)
    pair_dir.mkdir(parents=True)
    _write_history(pair_dir / "pair_AAAUSDT_BBBUSDT_daily_history.json")
    pd.DataFrame([_captured_settings("Static (Spread)"), _captured_settings("Dyn (Spread)")]).to_csv(
        active / "crypto_wizards_pair_page_capture_settings.csv",
        index=False,
    )

    result = build_wizard_mode_comparison(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_mode_comparison"])

    assert result.summary["comparison_groups"] == 1
    assert len(frame) == 6
    assert set(frame["cost_case"]) == {"zero_cost_upper_bound", "provisional_base", "provisional_stress"}
    assert set(frame["exact_mode"]) == {"Static (Spread)", "Dyn (Spread)"}
    assert set(frame["mode_replay_status"]) == {"READY_FOR_RESEARCH_REPLAY"}
    assert set(frame["mode_fidelity_status"]) == {"local_formula_approximation"}
    assert set(frame["history_match_status"]) == {"LOCAL_HISTORY_MATCHED_BY_LEGS_INTERVAL_AND_EXCHANGE_RESEARCH_ONLY"}
    assert set(frame["source_history_rows"]) == {12}
    assert set(frame["history_rows"]) == {12}
    assert set(frame["sample_parity_status"]) == {"MATCHED_OBSERVATION_COUNT"}
    assert set(frame["pnl_weighting_source"]) == {"captured_wizard_x_y_weighting"}
    assert (frame["intended_weight_x"] == 0.4).all()
    assert (frame["intended_weight_y"] == 0.6).all()
    assert (frame["realized_weight_x"] == 0.4).all()
    assert (frame["realized_weight_y"] == 0.6).all()
    assert set(frame["comparison_validity"]) == {"RESEARCH_DIAGNOSTIC_ONLY"}
    assert not frame["training_eligible"].astype(bool).any()
    assert (frame.loc[frame["cost_case"] == "zero_cost_upper_bound", "total_funding"] == 0.0).all()
    assert frame.loc[frame["cost_case"] == "provisional_base", "total_funding"].sum() > 0.0
    assert frame["open_trade_policy"].eq(
        "report_mark_to_market_and_forced_close_use_conservative_result"
    ).all()
    assert (frame["conservative_total_return"] <= frame["total_return"]).all()
    assert (frame["conservative_total_return"] <= frame["forced_close_total_return"]).all()
    assert (frame["conservative_max_drawdown"] >= frame["max_drawdown"]).all()
    assert not frame["acceptance_eligible"].astype(bool).any()
    assert not frame["paper_or_execution_eligible"].astype(bool).any()
    assert not frame["promotion_allowed"].astype(bool).any()


def test_mode_comparison_blocks_without_matching_history(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame([_captured_settings("Static (Spread)")]).to_csv(
        active / "crypto_wizards_pair_page_capture_settings.csv",
        index=False,
    )

    result = build_wizard_mode_comparison(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_mode_comparison"])

    assert len(frame) == 1
    assert frame.loc[0, "mode_replay_status"] == "BLOCKED_HISTORY"
    assert "matching_local_two_leg_history_missing" in frame.loc[0, "mode_missing_inputs"]
    assert not bool(frame.loc[0, "acceptance_eligible"])


def test_mode_comparison_keeps_cross_venue_history_research_only(tmp_path):
    active = tmp_path / "reports" / "active"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    active.mkdir(parents=True)
    pair_dir.mkdir(parents=True)
    _write_history(pair_dir / "pair_AAAUSDT_BBBUSDT_daily_history.json")
    pd.DataFrame([_captured_settings("Static (Spread)")]).to_csv(
        active / "crypto_wizards_pair_page_capture_settings.csv",
        index=False,
    )
    payload_path = pair_dir / "pair_AAAUSDT_BBBUSDT_daily_history.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["exchange"] = "hyperliquid"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    result = build_wizard_mode_comparison(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_mode_comparison"])

    assert set(frame["history_match_status"]) == {"LOCAL_HISTORY_LEGS_INTERVAL_VENUE_MISMATCH_RESEARCH_ONLY"}
    assert frame["next_step"].str.contains("same_venue_history").all()
    assert not frame["acceptance_eligible"].astype(bool).any()
    assert frame["validity_blocker"].str.contains("wizard_source_venue_differs_from_local_venue").all()


def test_mode_comparison_matches_compact_wizard_symbols_to_hourly_hyperliquid_history(tmp_path):
    active = tmp_path / "reports" / "active"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    active.mkdir(parents=True)
    pair_dir.mkdir(parents=True)
    capture = _captured_settings("Static (Spread)")
    capture.update(
        {
            "pair": "BTCUSDT / EIGENPERP",
            "asset_x": "BTCUSDT",
            "asset_y": "EIGENPERP",
            "exchange": "ByBit",
            "interval": "hourly",
        }
    )
    pd.DataFrame([capture]).to_csv(active / "crypto_wizards_pair_page_capture_settings.csv", index=False)
    history_path = pair_dir / "pair_btc_eigen_hyperliquid_1h_derived_history.json"
    _write_history(
        history_path,
        asset_x="BTC-USD",
        asset_y="EIGEN-USD",
        exchange="hyperliquid",
        interval="1h",
    )

    result = build_wizard_mode_comparison(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_mode_comparison"])

    assert len(frame) == 3
    assert set(frame["mode_replay_status"]) == {"READY_FOR_RESEARCH_REPLAY"}
    assert set(frame["interval"]) == {"hourly"}
    assert frame["history_path"].str.contains(history_path.name, regex=False).all()
    assert set(frame["history_match_status"]) == {"LOCAL_HISTORY_LEGS_INTERVAL_VENUE_MISMATCH_RESEARCH_ONLY"}


def test_mode_comparison_uses_captured_weights_and_scanner_period_not_history_hedge(tmp_path):
    active = tmp_path / "reports" / "active"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    active.mkdir(parents=True)
    pair_dir.mkdir(parents=True)
    history_path = pair_dir / "pair_AAAUSDT_BBBUSDT_daily_history.json"
    _write_history(history_path)
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    older = [dict(row, timestamp=f"2026-07-{index + 1:02d}T00:00:00Z", hedge_ratio=999999.0) for index, row in enumerate(payload["history"])]
    for row in payload["history"]:
        row["hedge_ratio"] = 999999.0
    payload["history"] = older + payload["history"]
    history_path.write_text(json.dumps(payload), encoding="utf-8")
    capture = _captured_settings("Static (Spread)")
    capture["period"] = 12
    pd.DataFrame([capture]).to_csv(active / "crypto_wizards_pair_page_capture_settings.csv", index=False)

    result = build_wizard_mode_comparison(root=tmp_path)
    frame = pd.read_csv(result.paths["wizard_mode_comparison"])

    assert set(frame["source_history_rows"]) == {24}
    assert set(frame["history_rows"]) == {12}
    assert (frame["realized_weight_x"] == 0.4).all()
    assert (frame["realized_weight_y"] == 0.6).all()
