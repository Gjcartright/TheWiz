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
    assert {"zero_cost", "base_cost_used", "stress_cost"}.issubset(set(costs["cost_case"]))
    assert len(trades) == 2


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
    assert set(frame["verification_status"]) == {"verified", "blocked"}
    blocked = frame[frame["verification_status"] == "blocked"].iloc[0]
    assert "local_history_missing_columns" in blocked["verification_blocker"]
