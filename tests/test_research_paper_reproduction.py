from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.research_paper_reproduction import (
    build_causal_pair_frame,
    build_one_sided_stability_diagnostics,
    build_panel_residual_dataset,
    estimate_gph_fractional_d,
    load_hyperliquid_daily_panel,
    run_dynamic_no_trade_band_tests,
    run_panel_residual_walkforward,
)


def _synthetic_panel(*, assets: int = 10, rows: int = 380) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    index = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    market = rng.normal(0.0004, 0.018, rows)
    closes: dict[str, np.ndarray] = {}
    volumes: dict[str, np.ndarray] = {}
    for number in range(assets):
        residual = rng.normal(0.0, 0.008 + number / 100_000.0, rows)
        returns = (0.7 + number / 50.0) * market + residual
        closes[f"A{number}"] = 100.0 * np.exp(np.cumsum(returns))
        volumes[f"A{number}"] = np.exp(12.0 + rng.normal(0.0, 0.2, rows))
    return pd.DataFrame(closes, index=index), pd.DataFrame(volumes, index=index)


def test_panel_loader_records_collection_vintage_blocker(tmp_path: Path):
    candle_dir = tmp_path / "data" / "raw" / "hyperliquid_candles"
    candle_dir.mkdir(parents=True)
    timestamps = pd.date_range("2024-01-01", periods=310, freq="D", tz="UTC")
    for asset_number in range(8):
        candles = [
            {
                "startedAt": timestamp.isoformat(),
                "close": str(100 + asset_number + row_number / 10),
                "usdVolume": str(1_000_000 + row_number),
            }
            for row_number, timestamp in enumerate(timestamps)
        ]
        path = candle_dir / f"A{asset_number}_1d_candles.json"
        path.write_text(
            json.dumps({"candles": candles, "fetched_at": "2026-01-01T00:00:00Z"}),
            encoding="utf-8",
        )

    close, volume, audit = load_hyperliquid_daily_panel(root=tmp_path)

    assert close.shape == (310, 8)
    assert volume.shape == close.shape
    assert audit["collection_after_history"].all()
    assert not audit["point_in_time_vintage"].any()
    assert audit["blocker"].eq("historical_payload_not_as_traded_vintage").all()


def test_panel_walkforward_is_causal_and_includes_whole_asset_holdout():
    close, volume = _synthetic_panel()
    dataset = build_panel_residual_dataset(
        close=close,
        volume=volume,
        factor_window=60,
        residual_window=20,
    )

    assert (dataset["feature_timestamp"] < dataset["label_timestamp"]).all()
    assert not dataset["uses_future_data"].any()
    assert not dataset["promotion_authority"].any()

    predictions, metrics = run_panel_residual_walkforward(
        dataset,
        minimum_train_dates=100,
        test_dates=40,
    )

    assert set(predictions["population"]) == {"seen_asset", "whole_asset_holdout"}
    assert {"pooled_logistic", "mean_reversion_sign", "persistence_sign"} == set(metrics["model"])
    assert (pd.to_datetime(predictions["train_end_at"]) < predictions["feature_timestamp"]).all()
    assert not metrics["promotion_authority"].any()


def test_fractional_estimator_separates_white_noise_from_integrated_series():
    rng = np.random.default_rng(7)
    white_noise = pd.Series(rng.normal(size=512))
    integrated = pd.Series(np.cumsum(rng.normal(size=512)))

    white_d = estimate_gph_fractional_d(white_noise)
    integrated_d = estimate_gph_fractional_d(integrated)

    assert -0.5 < white_d < 0.5
    assert integrated_d > white_d + 0.5


def test_causal_pair_frame_uses_canonical_y_on_x_orientation():
    rng = np.random.default_rng(19)
    index = pd.date_range("2024-01-01", periods=260, freq="D", tz="UTC")
    log_x = 4.0 + np.cumsum(rng.normal(0.0005, 0.01, len(index)))
    residual = rng.normal(0.0, 0.002, len(index))
    log_y = 0.35 + 1.7 * log_x + residual
    close = pd.DataFrame(
        {
            "X": np.exp(log_x),
            "Y": np.exp(log_y),
        },
        index=index,
    )

    frame = build_causal_pair_frame(
        close,
        "X",
        "Y",
        hedge_window=90,
        zscore_window=40,
    )

    assert np.isclose(frame["beta"].median(), 1.7, atol=0.08)
    expected_spread = log_y[-1] - 0.35 - 1.7 * log_x[-1]
    assert np.isclose(frame["spread"].iloc[-1], expected_spread, atol=0.02)
    assert frame["spread"].std() < 0.01
    beta = frame["beta"].iloc[-1]
    target_position = close.index.get_loc(frame.index[-1])
    prior_x = pd.Series(log_x, index=index).iloc[target_position - 90:target_position]
    prior_y = pd.Series(log_y, index=index).iloc[target_position - 90:target_position]
    oracle_beta = prior_y.cov(prior_x) / prior_x.var()
    assert np.isclose(beta, oracle_beta)

    contemporaneous_shock = close.copy()
    contemporaneous_shock.loc[frame.index[-1], "X"] *= np.exp(5.0)
    contemporaneous_shock.loc[frame.index[-1], "Y"] *= np.exp(-4.0)
    shocked_frame = build_causal_pair_frame(
        contemporaneous_shock,
        "X",
        "Y",
        hedge_window=90,
        zscore_window=40,
    )
    assert np.isclose(shocked_frame.loc[frame.index[-1], "beta"], beta)
    assert not np.isclose(shocked_frame.loc[frame.index[-1], "spread"], frame["spread"].iloc[-1])

    expected_pair_return = ((log_y[-1] - log_y[-2]) - beta * (log_x[-1] - log_x[-2])) / (
        1.0 + abs(beta)
    )
    assert np.isclose(frame["pair_return"].iloc[-1], expected_pair_return)

    prefix = build_causal_pair_frame(
        close.iloc[:-1],
        "X",
        "Y",
        hedge_window=90,
        zscore_window=40,
    )
    pd.testing.assert_series_equal(
        frame.loc[prefix.index[-1]],
        prefix.iloc[-1],
        check_names=False,
    )


def test_stability_and_no_trade_bands_remain_research_only():
    index = pd.date_range("2024-01-01", periods=360, freq="D", tz="UTC")
    phase = np.linspace(0.0, 16.0 * np.pi, len(index))
    frame = pd.DataFrame(
        {
            "beta": 1.1,
            "spread": np.sin(phase) + np.linspace(0.0, 0.3, len(index)),
            "zscore": 2.0 * np.sin(phase),
            "pair_return": pd.Series(0.01 * np.cos(phase), index=index),
        },
        index=index,
    )
    pairs = {
        "AAA-USD-BBB-USD": {
            "frame": frame,
            "asset_x": "AAA",
            "asset_y": "BBB",
            "round_trip_cost_bps": 20.0,
            "cost_evidence_path": "evidence.csv",
            "cost_model_id": "cost-1",
        }
    }

    stability = build_one_sided_stability_diagnostics(pairs)
    bands = run_dynamic_no_trade_band_tests(pairs)

    assert not stability["uses_future_data"].any()
    assert not stability["exact_fcvar"].any()
    assert not stability["exact_bai_perron"].any()
    assert not stability["promotion_authority"].any()
    continuous = bands[bands["policy"].eq("continuous_rebalance")].iloc[0]
    dynamic = bands[bands["policy"].eq("dynamic_cost_vol_band")].iloc[0]
    assert dynamic["position_changes"] <= continuous["position_changes"]
    assert dynamic["total_turnover"] <= continuous["total_turnover"]
    assert not bands["funding_modeled"].any()
    assert not bands["promotion_authority"].any()
