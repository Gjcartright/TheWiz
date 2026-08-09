from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from quant_platform.experiments import PairDataset
from quant_platform.ml_filter import (
    _compounded_return_path,
    _model_selection_score,
    _purged_pair_aware_splits,
    _select_probability_threshold,
    _threshold_quality_score,
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
    assert frame["realized_return"].ge(-1.0).all()
    assert frame["max_adverse_excursion"].ge(-1.0).all()
    assert frame["return_aggregation"].eq("compounded_bar_returns_zero_floor").all()
    assert frame["return_unit"].eq("fraction_of_equity").all()


def test_trade_dataset_records_multiple_completed_zscore_lifecycles():
    history = _pair_history_frame().iloc[:24].copy()
    lifecycle = [0.0, 2.2, 1.0, 0.1, 0.0, -2.3, -1.0, -0.1, 0.0, 2.4, 1.0, 0.0]
    history["zscore"] = lifecycle * 2
    history["zscore_reconstructed"] = lifecycle * 2
    history["spread"] = history["zscore"]

    frame = build_trade_filter_dataset(
        [PairDataset("BTC-USD-SOL-USD", history)], strategies=(STRATEGIES[0],)
    )

    assert len(frame) == 6
    assert frame["entry_timestamp"].lt(frame["exit_timestamp"]).all()
    assert frame["trade_bars"].max() < len(history)


def test_trade_dataset_excludes_right_censored_open_positions():
    history = _pair_history_frame().iloc[:24].copy()
    history["zscore"] = 0.0
    history["zscore_reconstructed"] = 0.0
    history.loc[history.index[-2]:, "zscore_reconstructed"] = 2.5
    history["spread"] = history["zscore_reconstructed"]

    frame = build_trade_filter_dataset(
        [PairDataset("BTC-USD-SOL-USD", history)], strategies=(STRATEGIES[0],)
    )

    assert frame.empty


def test_trade_return_path_compounds_fractional_bar_returns_and_floors_bankruptcy():
    path = _compounded_return_path(pd.Series([0.10, -0.10]))
    bankrupt = _compounded_return_path(pd.Series([0.05, -1.20, 0.50]))

    assert path.iloc[-1] == pytest.approx(-0.01)
    assert bankrupt.iloc[-1] == pytest.approx(-1.0)


def test_build_trade_filter_dataset_rejects_integer_bar_indexes_as_timestamps():
    history = _pair_history_frame()
    history["timestamp"] = np.arange(len(history))

    frame = build_trade_filter_dataset([PairDataset("BTC-USD-SOL-USD", history)], strategies=(STRATEGIES[0],))

    assert frame.empty


def test_build_trade_filter_dataset_normalizes_timeframe_aliases():
    history = _pair_history_frame()
    history["interval"] = "daily"

    frame = build_trade_filter_dataset([PairDataset("BTC-USD-SOL-USD", history)], strategies=(STRATEGIES[0],))

    assert set(frame["timeframe"]) == {"1d"}


def test_train_trade_filter_walkforward_writes_outputs(tmp_path):
    dataset = _candidate_dataset()
    dataset["wizard_learning_feature_state"] = "cold_start"

    paths = train_trade_filter_walkforward(dataset, output_dir=tmp_path, n_splits=3, min_train_rows=60)

    assert set(paths) == {"dataset", "folds", "predictions", "summary", "best_model", "manifest"}
    summary = pd.read_csv(paths["summary"])
    folds = pd.read_csv(paths["folds"])
    assert not summary.empty
    assert not folds.empty
    assert {"model_name", "median_filtered_profit_factor", "profit_factor_delta", "promising"}.issubset(summary.columns)
    assert folds["split_scheme"].eq("purged_embargoed_pair_aware_timestamp_groups").all()
    assert folds["embargo_periods"].eq(1).all()
    with open(paths["best_model"], "rb") as handle:
        artifact = pickle.load(handle)
    assert artifact["model_name"] == summary.iloc[0]["model_name"]
    assert "estimator" in artifact
    assert "threshold" in artifact
    assert artifact["evaluation_scheme"] == "purged_embargoed_pair_aware_timestamp_groups"


def test_pair_aware_splits_group_timestamps_and_purge_overlapping_labels():
    timestamps = pd.date_range("2026-01-01", periods=18, freq="h", tz="UTC")
    frame = pd.DataFrame(
        [
            {
                "pair": pair,
                "entry_timestamp": timestamp,
                "exit_timestamp": timestamp + pd.Timedelta(hours=2),
            }
            for timestamp in timestamps
            for pair in ("BTC-USD/ETH-USD", "SOL-USD/HYPE-USD")
        ]
    )

    splits = _purged_pair_aware_splits(frame, n_splits=3, embargo_periods=1)

    assert len(splits) == 3
    for train_idx, test_idx, audit in splits:
        train = frame.iloc[train_idx]
        test = frame.iloc[test_idx]
        assert set(train["entry_timestamp"]).isdisjoint(set(test["entry_timestamp"]))
        for pair in set(test["pair"]):
            pair_train = train.loc[train["pair"] == pair]
            pair_test = test.loc[test["pair"] == pair]
            assert pair_train["exit_timestamp"].max() < pair_test["entry_timestamp"].min()
        assert audit["purged_or_embargoed_rows"] > 0


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


def test_threshold_quality_score_penalizes_sparse_weak_thresholds():
    weak_sparse = {
        "profit_factor": 1.1,
        "sharpe": 0.8,
        "drawdown": 0.7,
        "expectancy": 0.003,
        "total_return": 0.08,
        "trade_count": 4,
    }
    stronger_balanced = {
        "profit_factor": 1.05,
        "sharpe": 0.75,
        "drawdown": 0.15,
        "expectancy": 0.0025,
        "total_return": 0.06,
        "trade_count": 20,
    }

    assert _threshold_quality_score(stronger_balanced) > _threshold_quality_score(weak_sparse)


def test_select_probability_threshold_prefers_better_risk_adjusted_cutoff():
    probability = np.array([0.52, 0.58, 0.63, 0.68, 0.73, 0.78, 0.83, 0.88])
    realized_returns = np.array([-0.08, -0.07, -0.05, -0.03, 0.03, 0.04, 0.05, 0.06])

    threshold = _select_probability_threshold(probability, realized_returns)

    assert threshold >= 0.75


def test_model_selection_score_prefers_promising_lower_drawdown_model():
    stronger = {
        "promising": True,
        "profit_factor_delta": 0.30,
        "sharpe_delta": 2.0,
        "expectancy_delta": 0.02,
        "worst_filtered_drawdown": 0.25,
    }
    weaker = {
        "promising": False,
        "profit_factor_delta": 0.10,
        "sharpe_delta": 2.5,
        "expectancy_delta": 0.01,
        "worst_filtered_drawdown": 0.95,
    }

    assert _model_selection_score(stronger) > _model_selection_score(weaker)


def test_available_model_specs_contains_linear_and_boosted_family():
    specs = available_model_specs()
    names = {spec.name for spec in specs}
    assert "logistic_regression" in names
    assert any(spec.family.startswith("boosted_tree") for spec in specs)
