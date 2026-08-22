from __future__ import annotations

import json

import pandas as pd

from quant_platform.wizard_local_verification import build_wizard_local_verification_batch, static_spread_signal, verify_wizard_local_mode


def test_static_spread_signal_uses_wizard_entry_exit_thresholds():
    zscore = pd.Series([0.0, 2.2, 1.0, -0.1, 0.0, -2.3, -1.0, 0.2])

    signal, trades = static_spread_signal(zscore)

    assert signal.tolist() == [0.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    assert [trade["direction"] for trade in trades] == ["long_x_short_y", "short_x_long_y"]
    assert all(trade["exit_reason"] == "static_spread_zero_cross" for trade in trades)


def test_verify_wizard_local_mode_writes_fresh_after_cost_reports(tmp_path):
    history_path = tmp_path / "history.json"
    wizard_path = tmp_path / "wizard.json"
    rows = []
    zscores = [0.0, 2.2, 1.4, -0.1, 0.0, -2.4, -1.0, 0.3, 0.0, 0.1]
    for idx, zscore in enumerate(zscores):
        rows.append(
            {
                "timestamp": f"2026-06-{idx + 1:02d}T00:00:00Z",
                "price_x": 100 + idx,
                "price_y": 50 + idx * 0.1,
                "spread": float(zscore),
                "zscore": float(zscore),
                "hedge_ratio": 1.0,
                "beta": 1.0,
                "funding_bps_per_day": 0.0,
            }
        )
    history_path.write_text(
        json.dumps(
            {
                "asset_x": "AAA-USD",
                "asset_y": "BBB-USD",
                "interval": "1day",
                "period": 320,
                "history": rows,
            }
        ),
        encoding="utf-8",
    )
    wizard_path.write_text(json.dumps({"period": 320}), encoding="utf-8")

    result = verify_wizard_local_mode(
        root=tmp_path,
        history_path=history_path,
        wizard_capture_path=wizard_path,
        output_name="synthetic_static_spread",
        current_date="2026-06-10",
    )

    summary = pd.read_csv(result.paths["summary"])
    costs = pd.read_csv(result.paths["cost_comparison"])
    trades = pd.read_csv(result.paths["trade_log"])

    assert summary.loc[0, "pair"] == "AAA-USD/BBB-USD"
    assert summary.loc[0, "local_observations"] == len(rows)
    assert "local_history_rows<320" in summary.loc[0, "acceptance_reason"]
    assert summary.loc[0, "mode_fidelity_status"] == "generic_zscore_proxy"
    assert summary.loc[0, "acceptance"] == "BLOCKED"
    assert {"zero_cost", "base_cost_used", "stress_cost"}.issubset(set(costs["cost_case"]))
    assert len(trades) == 2


def test_verify_wizard_local_mode_uses_captured_mode_formula_without_zscore_and_blocks_acceptance(tmp_path):
    history_path = tmp_path / "history.json"
    wizard_path = tmp_path / "wizard.json"
    rows = []
    for idx in range(12):
        rows.append(
            {
                "timestamp": f"2026-06-{idx + 1:02d}T00:00:00Z",
                "price_x": 100 + idx,
                "price_y": 50 + idx * 0.2,
                "hedge_ratio": 1.9,
                "beta": 1.0,
                "funding_bps_per_day": 0.0,
            }
        )
    history_path.write_text(
        json.dumps(
            {
                "asset_x": "AAA-USD",
                "asset_y": "BBB-USD",
                "interval": "1day",
                "period": len(rows),
                "history": rows,
            }
        ),
        encoding="utf-8",
    )
    wizard_path.write_text(json.dumps({"period": len(rows)}), encoding="utf-8")
    settings = {
        "capture_confirmed": True,
        "capture_evidence_path": "data/raw/wizard/aaa_bbb_static_capture.png",
        "entry_long_operator": "<=",
        "entry_long_value": -2.82,
        "entry_long_position": "short_x_long_y",
        "entry_short_operator": ">=",
        "entry_short_value": -2.70,
        "entry_short_position": "long_x_short_y",
        "exit_long_operator": ">=",
        "exit_long_value": -2.78,
        "exit_short_operator": "<=",
        "exit_short_value": -2.76,
        "hedge_ratio": 1.9,
    }

    result = verify_wizard_local_mode(
        root=tmp_path,
        history_path=history_path,
        wizard_capture_path=wizard_path,
        output_name="captured_static_spread",
        current_date="2026-06-12",
        exact_mode="Static (Spread)",
        mode_settings=settings,
    )

    summary = pd.read_csv(result.paths["summary"]).iloc[0]

    assert summary["mode_replay_status"] == "READY_FOR_RESEARCH_REPLAY"
    assert summary["mode_metric_name"] == "static_spread_expanding_zscore"
    assert summary["mode_fidelity_status"] == "local_formula_approximation"
    assert "captured-settings local Static (Spread)" in summary["signal_source"]
    assert summary["acceptance"] == "BLOCKED"
    assert not bool(summary["promotion_allowed"])


def test_build_wizard_local_verification_batch_reports_verified_and_blocked_candidates(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    history_path = tmp_path / "data" / "raw" / "pair_details" / "pair_AAA-USD_BBB-USD_Dydx_Daily_320_history.json"
    history_path.parent.mkdir(parents=True)
    rows = []
    zscores = [0.0, 2.2, 1.4, -0.1, 0.0, -2.4, -1.0, 0.3, 0.0, 0.1]
    for idx, zscore in enumerate(zscores):
        rows.append(
            {
                "timestamp": f"2026-06-{idx + 1:02d}T00:00:00Z",
                "price_x": 100 + idx,
                "price_y": 50 + idx * 0.1,
                "spread": float(zscore),
                "zscore": float(zscore),
                "hedge_ratio": 1.0,
                "beta": 1.0,
                "funding_bps_per_day": 0.0,
            }
        )
    history_path.write_text(
        json.dumps(
            {
                "asset_x": "AAA-USD",
                "asset_y": "BBB-USD",
                "interval": "daily",
                "period": 320,
                "strategy_mode": "static",
                "history": rows,
            }
        ),
        encoding="utf-8",
    )
    incomplete_path = tmp_path / "data" / "raw" / "pair_details" / "pair_CCC-USD_DDD-USD_Dydx_Daily_320_history.json"
    incomplete_path.write_text(
        json.dumps(
            {
                "asset_x": "CCC-USD",
                "asset_y": "DDD-USD",
                "interval": "daily",
                "strategy_mode": "static",
                "history": [{"timestamp": "2026-06-01T00:00:00Z", "zscore": 0.0}],
            }
        ),
        encoding="utf-8",
    )
    queue_path = active / "queue.csv"
    pd.DataFrame(
        [
            {
                "pair": "AAA-USD/BBB-USD",
                "asset_x": "AAA-USD",
                "asset_y": "BBB-USD",
                "interval": "daily",
                "sharpe": 2.5,
                "returns_total": 0.3,
                "returns_total_pct": 30.0,
                "source_group": "test",
                "pair_history_path": str(history_path.relative_to(tmp_path)),
            },
            {
                "pair": "CCC-USD/DDD-USD",
                "asset_x": "CCC-USD",
                "asset_y": "DDD-USD",
                "interval": "daily",
                "sharpe": 2.1,
                "returns_total": 0.25,
                "returns_total_pct": 25.0,
                "source_group": "test",
                "pair_history_path": str(incomplete_path.relative_to(tmp_path)),
            },
        ]
    ).to_csv(queue_path, index=False)

    result = build_wizard_local_verification_batch(root=tmp_path, queue_path=queue_path, current_date="2026-06-10")
    frame = pd.read_csv(result.paths["batch"])

    assert result.summary["candidates"] == 2
    assert set(frame["verification_status"]) == {"proxy_only", "blocked"}
    blocked = frame[frame["verification_status"] == "blocked"].iloc[0]
    assert "local_history_missing_columns" in blocked["verification_blocker"]


def test_build_wizard_local_verification_batch_uses_queue_exact_mode_and_csv_capture_for_copula(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True)

    history_path = pair_dir / "pair_blur_ethfi_5mins_dydx_candles_derived_history.json"
    rows = []
    zscores = [0.0, 2.1, 1.2, 0.0, -2.2, -1.1, 0.0, 2.0, 0.0]
    for idx, zscore in enumerate(zscores):
        rows.append(
            {
                "timestamp": f"2026-06-{idx + 1:02d}T00:00:00Z",
                "price_x": 10 + idx * 0.1,
                "price_y": 20 + idx * 0.2,
                "spread": float(zscore),
                "zscore": float(zscore),
                "hedge_ratio": 1.0,
                "beta": 1.0,
            }
        )
    history_path.write_text(
        json.dumps(
            {
                "pair": "BLUR-USD-ETHFI-USD",
                "asset_x": "BLUR-USD",
                "asset_y": "ETHFI-USD",
                "interval": "5mins",
                "period": len(rows),
                "history": rows,
            }
        ),
        encoding="utf-8",
    )
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
            }
        ]
    ).to_csv(active / "crypto_wizards_pair_page_capture.csv", index=False)
    queue_path = active / "queue.csv"
    pd.DataFrame(
        [
            {
                "pair": "BLUR-USD/ETHFI-USD",
                "asset_x": "BLUR-USD",
                "asset_y": "ETHFI-USD",
                "interval": "daily",
                "exact_mode": "Copula",
                "dashboard_recommended_strategy": "Copula",
                "spread_id": 1,
                "strategy_id": 3,
                "sharpe": 1.9,
                "returns_total": 0.54,
                "returns_total_pct": 54.0,
                "source_group": "test",
                "pair_history_path": "data/raw/pair_details/pair_BLUR-USD_ETHFI-USD_missing.json",
            }
        ]
    ).to_csv(queue_path, index=False)

    result = build_wizard_local_verification_batch(root=tmp_path, queue_path=queue_path, current_date="2026-06-10")
    frame = pd.read_csv(result.paths["batch"])
    row = frame.iloc[0]

    assert row["exact_mode"] == "Copula"
    assert "missing_exact_mode_capture" not in str(row["verification_blocker"])
    assert "missing_local_history_path" not in str(row["verification_blocker"])
    assert "missing_wizard_capture_path" not in str(row["verification_blocker"])


def test_build_wizard_local_verification_batch_allows_zscorer_exact_modes(tmp_path):
    active = tmp_path / "reports" / "active"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    active.mkdir(parents=True)
    pair_dir.mkdir(parents=True)

    history_path = pair_dir / "pair_morpho_pendle_5mins_dydx_candles_derived_history.json"
    rows = []
    zscores = [0.0, -2.1, -1.2, 0.2, 0.0, 2.3, 1.0, -0.1, 0.0]
    for idx, zscore in enumerate(zscores):
        rows.append(
            {
                "timestamp": f"2026-06-{idx + 1:02d}T00:00:00Z",
                "price_x": 2.0 + idx * 0.01,
                "price_y": 1.0 + idx * 0.01,
                "spread": float(zscore),
                "zscore": float(zscore),
                "hedge_ratio": 1.0,
                "beta": 1.0,
            }
        )
    history_path.write_text(
        json.dumps(
            {
                "pair": "MORPHO-USD-PENDLE-USD",
                "asset_x": "MORPHO-USD",
                "asset_y": "PENDLE-USD",
                "interval": "5mins",
                "period": len(rows),
                "history": rows,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "pair": "MORPHO-USD/PENDLE-USD",
                "asset_x": "MORPHO-USD",
                "asset_y": "PENDLE-USD",
                "interval": "Live",
                "exact_mode": "Dyn (ZScoreR)",
                "dashboard_recommended_strategy": "Dyn (ZScoreR)",
                "source_group": "test",
                "pair_history_path": str(history_path.relative_to(tmp_path)),
                "sharpe": 0.27,
                "returns_total_pct": 23.1,
            }
        ]
    ).to_csv(active / "queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "MORPHO-USD/PENDLE-USD",
                "asset_x": "MORPHO-USD",
                "asset_y": "PENDLE-USD",
                "interval": "Live",
                "exact_mode": "Dyn (ZScoreR)",
                "dashboard_recommended_strategy": "Dyn (ZScoreR)",
            }
        ]
    ).to_csv(active / "crypto_wizards_pair_page_capture.csv", index=False)

    result = build_wizard_local_verification_batch(
        root=tmp_path,
        queue_path=active / "queue.csv",
        current_date="2026-06-10",
    )
    frame = pd.read_csv(result.paths["batch"])
    row = frame.iloc[0]

    assert row["exact_mode"] == "Dyn (ZScoreR)"
    assert row["verification_status"] == "proxy_only"
    assert "unsupported_exact_mode" not in str(row["verification_blocker"])


def test_build_wizard_local_verification_batch_enriches_sparse_queue_rows_from_primary_wizard_evidence(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    pair_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "pair": "EUR-USD/FIL-USD",
                "asset_x": "EUR-USD",
                "asset_y": "FIL-USD",
                "interval": "daily",
                "period": 365,
                "setup_identity": "EUR-USD|FIL-USD|daily|365|static_(spread)",
                "setup_role": "primary",
                "primary_wizard_setup": True,
                "exact_mode": "Static (Spread)",
                "spread_id": 3,
                "strategy_id": 1,
                "sharpe": 2.77,
                "returns_total": 0.297,
                "returns_total_pct": 29.7,
                "source_path": "data/raw/pair_details/pair_EUR-USD_FIL-USD_Dydx_Daily_365_daily_cw_backtest_history.json",
            }
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)

    wizard_capture = pair_dir / "pair_EUR-USD_FIL-USD_Dydx_Daily_365_daily_cw_backtest_history.json"
    wizard_capture.write_text(
        json.dumps(
            {
                "pair": "EUR-USD/FIL-USD",
                "asset_x": "EUR-USD",
                "asset_y": "FIL-USD",
                "interval": "daily",
                "period": 365,
                "strategy_mode": "static",
                "history": [{"timestamp": 0, "zscore": -1.0, "rolling_zscore": -0.8}],
            }
        ),
        encoding="utf-8",
    )

    local_history = pair_dir / "pair_eur_fil_cw_daily_sharpe_1day_dydx_long_history_derived_history.json"
    local_rows = []
    for idx, zscore in enumerate([0.0, 2.2, 1.5, 0.0, -2.1, -1.0, 0.0, 2.0, 0.0, -2.0]):
        local_rows.append(
            {
                "timestamp": f"2026-06-{idx + 1:02d}T00:00:00Z",
                "price_x": 1.0 + idx * 0.01,
                "price_y": 2.0 + idx * 0.02,
                "spread": float(zscore),
                "zscore": float(zscore),
                "rolling_zscore": float(zscore),
                "hedge_ratio": 1.0,
                "beta": 1.0,
            }
        )
    local_history.write_text(
        json.dumps(
            {
                "pair": "EUR-USD-FIL-USD",
                "asset_x": "EUR-USD",
                "asset_y": "FIL-USD",
                "interval": "1day",
                "period": len(local_rows),
                "history": local_rows,
            }
        ),
        encoding="utf-8",
    )

    queue_path = active / "queue.csv"
    pd.DataFrame([{"pair": "EUR-USD/FIL-USD", "sharpe": 2.77, "returns_total": 0.297}]).to_csv(queue_path, index=False)

    result = build_wizard_local_verification_batch(root=tmp_path, queue_path=queue_path, current_date="2026-06-10")
    frame = pd.read_csv(result.paths["batch"])
    row = frame.iloc[0]

    assert row["setup_identity"] == "EUR-USD|FIL-USD|daily|365|static_(spread)"
    assert row["exact_mode"] == "Static (Spread)"
    assert str(row["history_path"]).endswith("pair_eur_fil_cw_daily_sharpe_1day_dydx_long_history_derived_history.json")
    assert str(row["wizard_capture_path"]).endswith("pair_EUR-USD_FIL-USD_Dydx_Daily_365_daily_cw_backtest_history.json")
    assert "missing_exact_mode_capture" not in str(row["verification_blocker"])
    assert "missing_local_history_path" not in str(row["verification_blocker"])
    assert "missing_wizard_capture_path" not in str(row["verification_blocker"])


def test_build_wizard_local_verification_batch_uses_canonical_wizard_packet_and_scanner_capture(tmp_path):
    active = tmp_path / "reports" / "active"
    brain = tmp_path / "reports" / "brain"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    brain.mkdir(parents=True)
    pair_dir.mkdir(parents=True)
    processed.mkdir(parents=True)

    local_history = pair_dir / "pair_doge_eigen_5mins_dydx_candles_derived_history.json"
    rows = []
    for idx, zscore in enumerate([0.0, 2.1, 1.4, 0.0, -2.3, -1.2, 0.0, 2.0, 0.0, -2.0]):
        rows.append(
            {
                "timestamp": f"2026-06-{idx + 1:02d}T00:00:00Z",
                "price_x": 1.0 + idx * 0.01,
                "price_y": 2.0 + idx * 0.02,
                "spread": float(zscore),
                "zscore": float(zscore),
                "rolling_zscore": float(zscore),
                "hedge_ratio": 1.0,
                "beta": 1.0,
            }
        )
    local_history.write_text(
        json.dumps(
            {
                "pair": "DOGE-USD-EIGEN-USD",
                "asset_x": "DOGE-USD",
                "asset_y": "EIGEN-USD",
                "interval": "5mins",
                "period": len(rows),
                "history": rows,
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "candidate_id": "wizard:DOGE-USD-EIGEN-USD::daily",
                "pair": "DOGE-USD-EIGEN-USD",
                "timeframe": "daily",
                "strategy_mode": "Static (Spread)",
                "setup_identity": "DOGE-USD|EIGEN-USD|daily",
                "setup_role": "primary",
                "source_path": "reports/active/crypto_wizards_next_best_sharpe_returns_queue.csv",
            }
        ]
    ).to_csv(brain / "wizard_candidate_packets.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "DOGE-USD/EIGEN-USD",
                "asset_x": "DOGE-USD",
                "asset_y": "EIGEN-USD",
                "interval": "daily",
                "sharpe": 0.17,
                "returns_total": 0.23,
                "returns_total_pct": 23.0,
            }
        ]
    ).to_csv(active / "crypto_wizards_next_best_sharpe_returns_queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "DOGE-USD-EIGEN-USD",
                "candidate_id": "wizard:DOGE-USD-EIGEN-USD::daily",
                "timeframe": "daily",
                "matched_hourly_strategy": "Static (Spread)",
            }
        ]
    ).to_csv(brain / "wizard_hourly_repair_targets.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "DOGE-USD-EIGEN-USD",
                "timeframe": "daily",
                "strategy_mode": "Static (Spread)",
                "corr_jneg": 0.98,
            }
        ]
    ).to_csv(active / "wizard_scanner_dependency_capture.csv", index=False)

    queue_path = active / "queue.csv"
    pd.DataFrame(
        [
            {
                "pair": "DOGE-USD/EIGEN-USD",
                "asset_x": "DOGE-USD",
                "asset_y": "EIGEN-USD",
                "interval": "daily",
                "sharpe": 0.17,
                "returns_total": 0.23,
                "returns_total_pct": 23.0,
                "source_group": "test",
            }
        ]
    ).to_csv(queue_path, index=False)

    result = build_wizard_local_verification_batch(root=tmp_path, queue_path=queue_path, current_date="2026-06-10")
    frame = pd.read_csv(result.paths["batch"])
    row = frame.iloc[0]

    assert row["exact_mode"] == "Static (Spread)"
    assert str(row["history_path"]).endswith("pair_doge_eigen_5mins_dydx_candles_derived_history.json")
    assert "missing_local_history_path" not in str(row["verification_blocker"])
