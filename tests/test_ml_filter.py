from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from quant_platform.experiments import PairDataset
from quant_platform.ml_filter import (
    available_model_specs,
    build_trade_filter_dataset,
    shadow_trade_filter_predictions,
    shadow_model_branch_comparison,
    train_trade_filter_walkforward,
)
from quant_platform.strategies import STRATEGIES


def _pair_history_frame(n: int = 240) -> pd.DataFrame:
    x = np.linspace(0, 18 * np.pi, n)
    spread = np.sin(x) + 0.15 * np.sin(x / 3.0)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
            "spread": spread,
            "zscore": spread * 2.7,
            "price_x": 100 + np.linspace(0, 4, n) + spread,
            "price_y": 50 + np.linspace(0, 2, n) - spread,
            "hedge_ratio": 1.2,
            "hedge_ratio_stability": 0.9,
            "beta": 0.85,
            "funding_x_bps": np.where(np.arange(n) % 4 == 0, 1.5, 0.5),
            "funding_y_bps": np.where(np.arange(n) % 5 == 0, 0.25, -0.25),
            "funding_bps_per_day": 1.0,
            "realized_volatility_percentile": np.clip(np.abs(np.sin(x / 4.0)), 0, 1),
            "cvar": np.abs(spread) * 0.05 + 0.01,
            "var": np.abs(spread) * 0.03 + 0.005,
            "tail_dependence": 0.2 + np.abs(np.sin(x / 5.0)) * 0.3,
            "crisis_probability": np.clip(np.abs(np.cos(x / 6.0)) * 0.4, 0, 1),
            "liquidity_score": 0.7 + np.abs(np.sin(x / 7.0)) * 0.2,
            "bid_ask_spread_bps": 4.0 + np.abs(np.sin(x)) * 2.0,
            "slippage_bps": 3.0 + np.abs(np.cos(x)) * 2.0,
            "volume_x_usd": 10000 + np.arange(n) * 10,
            "volume_y_usd": 8000 + np.arange(n) * 8,
            "regime": np.where(np.arange(n) % 3 == 0, "bull", "range"),
            "regime_strategy_match": 0.55,
            "cointegration_pvalue": 0.04 + np.abs(np.sin(x / 8.0)) * 0.05,
            "ecm_strength": 0.55 + np.sin(x / 9.0) * 0.1,
            "ecm_x": np.sin(x / 4.0) * 0.03,
            "ecm_y": np.cos(x / 4.0) * 0.03,
            "half_life": 18 + np.abs(np.sin(x / 2.0)) * 8,
            "hurst": 0.35 + np.abs(np.cos(x / 3.0)) * 0.1,
            "conditional_probability_distortion": np.tanh(spread),
            "copula_calibration_score": 0.6 + np.abs(np.sin(x / 5.0)) * 0.2,
            "u1_given_u2": 0.55 + np.sin(x / 5.0) * 0.1,
            "u2_given_u1": 0.45 - np.sin(x / 5.0) * 0.1,
            "composite_score": 45 + np.sin(x / 2.0) * 10,
            "ml_confidence": 0.5 + np.sin(x / 3.0) * 0.1,
            "profile_match": 0.55 + np.cos(x / 3.0) * 0.1,
            "ou_optimal": 0.6 + np.sin(x / 4.0) * 0.1,
        }
    )
    return frame


def _candidate_dataset(n: int = 180) -> pd.DataFrame:
    idx = np.arange(n)
    probability_driver = np.sin(idx / 6.0)
    realized_return = np.where(probability_driver > 0, 0.02, -0.015) + np.where(idx % 7 == 0, -0.005, 0.0)
    frame = pd.DataFrame(
        {
            "trade_id": [f"trade-{i:03d}" for i in idx],
            "pair": np.where(idx % 2 == 0, "BTC-USD-SOL-USD", "DOGE-USD-ETH-USD"),
            "strategy_id": 143022,
            "strategy_name": "Canonical Branch",
            "family": "canonical",
            "backtest_mode": "two_leg",
            "entry_timestamp": pd.date_range("2026-02-01", periods=n, freq="h", tz="UTC"),
            "exit_timestamp": pd.date_range("2026-02-01 01:00:00", periods=n, freq="h", tz="UTC"),
            "entry_bar_index": idx,
            "exit_bar_index": idx + 1,
            "trade_bars": 2,
            "signal_side": np.where(idx % 2 == 0, "long_spread", "short_spread"),
            "label_profitable": (realized_return > 0).astype(int),
            "realized_return": realized_return,
            "gross_trade_return": realized_return + 0.002,
            "trade_cost_drag": 0.002,
            "entry_zscore": probability_driver * 2.0,
            "entry_abs_zscore": np.abs(probability_driver) * 2.0,
            "zscore_change_1": np.gradient(probability_driver),
            "zscore_change_3": np.gradient(probability_driver, edge_order=1),
            "spread_level": probability_driver * 10,
            "spread_change_1": np.gradient(probability_driver * 10),
            "spread_change_3": np.gradient(probability_driver * 10, edge_order=1),
            "spread_vol_12": 0.1 + np.abs(np.sin(idx / 10.0)) * 0.05,
            "spread_vol_48": 0.15 + np.abs(np.cos(idx / 12.0)) * 0.05,
            "hedge_ratio": 1.2,
            "hedge_ratio_stability": 0.95,
            "beta": 0.9,
            "realized_volatility_percentile": np.abs(np.sin(idx / 13.0)),
            "cvar": 0.02 + np.abs(np.cos(idx / 9.0)) * 0.01,
            "var": 0.01 + np.abs(np.sin(idx / 9.0)) * 0.005,
            "tail_dependence": 0.25 + np.abs(np.sin(idx / 8.0)) * 0.1,
            "crisis_probability": np.abs(np.cos(idx / 11.0)) * 0.2,
            "liquidity_score": 0.75 + np.abs(np.sin(idx / 14.0)) * 0.1,
            "bid_ask_spread_bps": 4.0 + np.abs(np.sin(idx / 5.0)),
            "slippage_bps": 3.0 + np.abs(np.cos(idx / 5.0)),
            "volume_x_usd": 10000 + idx * 20,
            "volume_y_usd": 9000 + idx * 15,
            "funding_x_bps": 0.5,
            "funding_y_bps": -0.25,
            "funding_diff_bps": -0.75,
            "funding_abs_total_bps": 0.75,
            "funding_bps_per_day": 1.0,
            "regime": np.where(idx % 3 == 0, "bull", "range"),
            "regime_strategy_match": 0.55,
            "cointegration_pvalue": 0.05 + np.abs(np.sin(idx / 20.0)) * 0.02,
            "ecm_strength": 0.5 + np.sin(idx / 18.0) * 0.1,
            "ecm_x": np.sin(idx / 12.0) * 0.02,
            "ecm_y": np.cos(idx / 12.0) * 0.02,
            "half_life": 18 + np.abs(np.sin(idx / 7.0)) * 6,
            "hurst": 0.38 + np.abs(np.cos(idx / 6.0)) * 0.05,
            "conditional_probability_distortion": probability_driver * 0.4,
            "copula_calibration_score": 0.65 + np.abs(np.sin(idx / 16.0)) * 0.1,
            "u1_given_u2": 0.55 + np.sin(idx / 17.0) * 0.05,
            "u2_given_u1": 0.45 - np.sin(idx / 17.0) * 0.05,
            "composite_score": 50 + probability_driver * 10,
            "ml_confidence": 0.5 + probability_driver * 0.1,
            "profile_match": 0.55 + np.cos(idx / 10.0) * 0.05,
            "ou_optimal": 0.6 + np.sin(idx / 9.0) * 0.05,
        }
    )
    return frame


def test_build_trade_filter_dataset_creates_candidate_entry_rows():
    datasets = [PairDataset("BTC-USD-SOL-USD", _pair_history_frame())]

    frame = build_trade_filter_dataset(datasets, strategies=(STRATEGIES[0],))

    assert not frame.empty
    assert {"trade_id", "entry_timestamp", "exit_timestamp", "label_profitable", "realized_return"}.issubset(frame.columns)
    assert frame["entry_timestamp"].lt(frame["exit_timestamp"]).all()
    assert set(frame["signal_side"]).issubset({"long_spread", "short_spread"})


def test_train_trade_filter_walkforward_writes_outputs(tmp_path):
    dataset = _candidate_dataset()

    paths = train_trade_filter_walkforward(dataset, output_dir=tmp_path, n_splits=3, min_train_rows=60)

    assert set(paths) == {"dataset", "folds", "predictions", "summary", "best_model", "manifest"}
    summary = pd.read_csv(paths["summary"])
    folds = pd.read_csv(paths["folds"])
    assert not summary.empty
    assert not folds.empty
    assert {"model_name", "median_filtered_profit_factor", "profit_factor_delta", "promising"}.issubset(summary.columns)
    with open(paths["best_model"], "rb") as handle:
        artifact = pickle.load(handle)
    assert "estimator" in artifact
    assert "threshold" in artifact


def test_shadow_trade_filter_predictions_scores_existing_dataset(tmp_path):
    dataset = _candidate_dataset()
    paths = train_trade_filter_walkforward(dataset, output_dir=tmp_path / "study", n_splits=3, min_train_rows=60)

    output = shadow_trade_filter_predictions(dataset, model_artifact_path=paths["best_model"], output_path=tmp_path / "shadow.csv")

    frame = pd.read_csv(output)
    assert not frame.empty
    assert {"probability_profitable", "shadow_take", "model_name", "threshold"}.issubset(frame.columns)
    assert frame["shadow_take"].isin([True, False]).all()


def test_shadow_model_branch_comparison_summarizes_model_and_pair_slices():
    predictions = pd.DataFrame(
        {
            "model_name": [
                "fallback",
                "fallback",
                "fallback",
                "logistic",
                "logistic",
                "logistic",
                "fallback",
                "logistic",
            ],
            "pair": [
                "BTC-USD-SOL-USD",
                "BTC-USD-SOL-USD",
                "DOGE-USD-ETH-USD",
                "BTC-USD-SOL-USD",
                "BTC-USD-SOL-USD",
                "DOGE-USD-ETH-USD",
                "ETH-USD-SOL-USD",
                "ETH-USD-SOL-USD",
            ],
            "realized_return": [0.03, -0.02, 0.01, 0.03, -0.02, 0.01, -0.30, -0.30],
            "shadow_take": [True, False, True, "true", "true", "false", True, True],
        }
    )

    model_report, pair_report = shadow_model_branch_comparison(
        predictions,
        pairs=("BTC-USD-SOL-USD", "DOGE-USD-ETH-USD"),
    )

    assert list(model_report["model_name"]) == ["fallback", "logistic"]
    assert int(model_report.loc[model_report["model_name"] == "fallback", "take_rows"].iloc[0]) == 2
    assert int(model_report.loc[model_report["model_name"] == "logistic", "take_rows"].iloc[0]) == 2
    assert set(pair_report["pair"]) == {"BTC-USD-SOL-USD", "DOGE-USD-ETH-USD"}
    assert "ETH-USD-SOL-USD" not in set(pair_report["pair"])
    btc_fallback = pair_report[
        (pair_report["model_name"] == "fallback") & (pair_report["pair"] == "BTC-USD-SOL-USD")
    ].iloc[0]
    assert btc_fallback["take_rows"] == 1
    assert btc_fallback["taken_mean_return"] == 0.03


def test_available_model_specs_contains_linear_and_boosted_family():
    specs = available_model_specs()
    names = {spec.name for spec in specs}
    assert "logistic_regression" in names
    assert any(spec.family.startswith("boosted_tree") for spec in specs)
