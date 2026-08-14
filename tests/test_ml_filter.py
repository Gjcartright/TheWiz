from __future__ import annotations

import json
import pickle
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from quant_platform.experiments import PairDataset
from quant_platform.ml_filter import (
    CATEGORICAL_FEATURES,
    GLOBAL_PURGED_SPLIT_SCHEME,
    NON_FEATURE_COLUMNS,
    _build_model_pipeline,
    _chronology_isolated_fold_partitions,
    _compounded_return_path,
    _model_selection_score,
    _purged_pair_aware_splits,
    _select_probability_threshold,
    _threshold_quality_score,
    available_model_specs,
    build_trade_filter_dataset,
    model_selection_leaderboard,
    shadow_model_branch_comparison,
    shadow_trade_filter_predictions,
    train_trade_filter_walkforward,
)
from quant_platform.strategies import STRATEGIES


def test_registered_learning_lineage_and_outcomes_cannot_be_model_features():
    forbidden = {
        "registered_contract_id",
        "registered_execution_id",
        "registered_semantic_hypothesis_id",
        "registered_hypothesis_outcome",
        "registered_candidate",
        "accepted_stage4_survivor",
        "experiment_id",
        "pair_group_key",
        "label_timestamp",
        "profit_after_cost",
        "total_fees",
        "total_slippage",
        "total_funding",
        "hold_bars",
        "testnet_order_authority",
        "live_trading_authorized",
    }

    assert forbidden.issubset(NON_FEATURE_COLUMNS)
    assert "orientation" not in NON_FEATURE_COLUMNS
    assert "orientation" in CATEGORICAL_FEATURES


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
    history = _pair_history_frame()
    history["exchange"] = "hyperliquid"
    history["source_path"] = "history.json"
    datasets = [PairDataset("BTC-USD-SOL-USD", history)]

    frame = build_trade_filter_dataset(datasets, strategies=(STRATEGIES[0],))

    assert not frame.empty
    assert {"trade_id", "entry_timestamp", "exit_timestamp", "label_profitable", "realized_return"}.issubset(frame.columns)
    assert frame["source_venue"].eq("hyperliquid").all()
    assert frame["source_path"].eq("history.json").all()
    assert frame["trade_id"].str.contains("hyperliquid").all()
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
    dataset["exit_timestamp"] = (
        pd.to_datetime(dataset["entry_timestamp"], utc=True)
        + pd.Timedelta(minutes=30)
    ).map(pd.Timestamp.isoformat)
    dataset["wizard_learning_feature_state"] = "cold_start"
    dataset["experiment_id"] = [f"experiment-{index}" for index in range(len(dataset))]
    dataset["registered_contract_id"] = "contract-1"
    dataset["registered_execution_id"] = "execution-1"
    dataset["registered_semantic_hypothesis_id"] = "hypothesis-1"
    dataset["registered_hypothesis_outcome"] = "ACCEPTED_SURVIVOR"
    dataset["registered_candidate"] = True
    dataset["accepted_stage4_survivor"] = True
    dataset["exact_mode"] = "Copula"
    dataset["orientation"] = "reverse"

    paths = train_trade_filter_walkforward(dataset, output_dir=tmp_path, n_splits=3, min_train_rows=60)

    assert set(paths) == {
        "dataset",
        "folds",
        "predictions",
        "selection_leaderboard",
        "summary",
        "best_model",
        "manifest",
    }
    summary = pd.read_csv(paths["summary"])
    folds = pd.read_csv(paths["folds"])
    predictions = pd.read_csv(paths["predictions"])
    leaderboard = pd.read_csv(paths["selection_leaderboard"])
    assert not summary.empty
    assert not folds.empty
    assert not leaderboard.empty
    assert leaderboard.iloc[0]["chosen_model"]
    assert leaderboard.iloc[0]["selection_rank"] == 1
    assert {"model_name", "median_filtered_profit_factor", "profit_factor_delta", "promising"}.issubset(summary.columns)
    assert folds["split_scheme"].eq(GLOBAL_PURGED_SPLIT_SCHEME).all()
    assert folds["global_label_purge"].map(bool).all()
    assert folds["global_label_overlap_rows_after_purge"].eq(0).all()
    assert (
        pd.to_datetime(folds["train_label_end_max"], utc=True)
        < pd.to_datetime(folds["test_start"], utc=True)
    ).all()
    assert folds["embargo_periods"].eq(1).all()
    assert folds["threshold_calibration_scheme"].eq(
        "training_only_minimum_participation_v1"
    ).all()
    assert folds["minimum_training_take_rate"].eq(0.10).all()
    assert folds["training_take_rate_floor_pass"].eq(
        folds["training_take_rate_at_threshold"].ge(0.10)
    ).all()
    assert set(folds["selection_phase"]) == {
        "model_selection",
        "untouched_evaluation",
    }
    assert {
        "experiment_id",
        "registered_contract_id",
        "registered_execution_id",
        "registered_semantic_hypothesis_id",
        "registered_hypothesis_outcome",
        "registered_candidate",
        "accepted_stage4_survivor",
        "exact_mode",
        "orientation",
        "global_label_purge",
        "global_label_overlap_rows_after_purge",
        "test_start",
        "train_label_end_max",
        "selection_phase",
        "selection_isolation_scheme",
    }.issubset(predictions.columns)
    with open(paths["best_model"], "rb") as handle:
        artifact = pickle.load(handle)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert artifact["model_name"] == manifest["chosen_model"]
    assert manifest["best_model_sha256"] == sha256(paths["best_model"].read_bytes()).hexdigest()
    assert artifact["model_name"] in set(summary["model_name"].astype(str))
    assert "estimator" in artifact
    assert "threshold" in artifact
    assert artifact["threshold_calibration_scheme"] == (
        "training_only_minimum_participation_v1"
    )
    assert artifact["minimum_training_take_rate"] == 0.10
    assert artifact["evaluation_scheme"] == GLOBAL_PURGED_SPLIT_SCHEME
    assert manifest["selection_isolation_scheme"] == (
        "chronological_model_selection_then_untouched_evaluation_v1"
    )
    assert manifest["selection_evaluation_boundary_scheme"] == (
        "label_complete_chronology_gap_v1"
    )
    assert pd.Timestamp(manifest["selection_label_end_boundary"]) < pd.Timestamp(
        manifest["untouched_evaluation_start_boundary"]
    )
    assert manifest["threshold_calibration_scheme"] == (
        "training_only_minimum_participation_v1"
    )
    assert manifest["minimum_training_take_rate"] == 0.10
    assert manifest["selection_folds"] == 1
    assert manifest["untouched_evaluation_folds"] == 1
    assert manifest["chosen_model_selection_eligible"] == bool(
        leaderboard.iloc[0]["promising"]
    )
    assert manifest["selection_eligible_models"] == int(
        leaderboard["promising"].astype(bool).sum()
    )
    assert manifest["selection_outcome"] in {
        "ELIGIBLE_MODEL_SELECTED",
        "NO_ELIGIBLE_MODEL_DIAGNOSTIC_ONLY",
    }


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
        assert audit["global_label_purge"] is True
        assert audit["global_label_overlap_rows_after_purge"] == 0
        assert train["exit_timestamp"].max() < test["entry_timestamp"].min()


def test_model_selection_boundary_inserts_gap_until_labels_are_complete():
    entries = pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC")
    exits = pd.Series(entries + pd.Timedelta(days=1))
    exits.iloc[5] = pd.Timestamp("2026-01-09", tz="UTC")
    ordered = pd.DataFrame(
        {
            "entry_timestamp": entries,
            "exit_timestamp": exits,
        }
    )
    splits = [
        (np.arange(0, start), np.arange(start, stop), {})
        for start, stop in ((2, 4), (4, 6), (6, 8), (9, 11))
    ]

    selection, evaluation, gap, selection_end, evaluation_start = (
        _chronology_isolated_fold_partitions(
            ordered,
            splits,
            [1, 2, 3, 4],
        )
    )

    assert selection == {1, 2}
    assert gap == {3}
    assert evaluation == {4}
    assert selection_end == pd.Timestamp("2026-01-09", tz="UTC")
    assert evaluation_start == pd.Timestamp("2026-01-10", tz="UTC")


def test_global_label_purge_removes_overlap_from_pair_absent_in_test_window():
    timestamps = pd.date_range("2026-01-01", periods=18, freq="h", tz="UTC")
    rows = [
        {
            "pair": "BTC-USD/ETH-USD",
            "entry_timestamp": timestamp,
            "exit_timestamp": timestamp + pd.Timedelta(hours=1),
        }
        for timestamp in timestamps
    ]
    rows.append(
        {
            "pair": "ETH-USD/SOL-USD",
            "entry_timestamp": timestamps[3],
            "exit_timestamp": timestamps[-1] + pd.Timedelta(hours=2),
        }
    )
    frame = pd.DataFrame(rows).sort_values("entry_timestamp").reset_index(drop=True)

    splits = _purged_pair_aware_splits(frame, n_splits=3, embargo_periods=1)

    assert splits
    for train_idx, test_idx, audit in splits:
        train = frame.iloc[train_idx]
        test = frame.iloc[test_idx]
        assert train["exit_timestamp"].max() < test["entry_timestamp"].min()
        assert audit["global_label_overlap_rows_after_purge"] == 0


def test_shadow_trade_filter_predictions_scores_existing_dataset(tmp_path):
    dataset = _candidate_dataset()
    dataset["exit_timestamp"] = (
        pd.to_datetime(dataset["entry_timestamp"], utc=True)
        + pd.Timedelta(minutes=30)
    ).map(pd.Timestamp.isoformat)
    paths = train_trade_filter_walkforward(dataset, output_dir=tmp_path / "study", n_splits=3, min_train_rows=60)

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    output = shadow_trade_filter_predictions(
        dataset,
        model_artifact_path=paths["best_model"],
        output_path=tmp_path / "shadow.csv",
        expected_model_sha256=manifest["best_model_sha256"],
    )

    frame = pd.read_csv(output)
    assert not frame.empty
    assert {"probability_profitable", "shadow_take", "model_name", "threshold"}.issubset(frame.columns)
    assert frame["shadow_take"].isin([True, False]).all()


def test_shadow_trade_filter_predictions_rejects_tampered_model_before_deserialization(tmp_path):
    model_path = tmp_path / "ml_trade_filter_best_model.pkl"
    model_path.write_bytes(b"not a pickle")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        shadow_trade_filter_predictions(
            pd.DataFrame(),
            model_artifact_path=model_path,
            output_path=tmp_path / "shadow.csv",
            expected_model_sha256="0" * 64,
        )


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


def test_select_probability_threshold_rejects_sparse_training_cutoff():
    probability = np.array([0.40] * 85 + [0.65] * 10 + [0.85] * 5)
    realized_returns = np.array([-0.01] * 85 + [0.01] * 10 + [0.20] * 5)

    threshold = _select_probability_threshold(probability, realized_returns)

    assert float((probability >= threshold).mean()) >= 0.10
    assert threshold <= 0.65


def test_select_probability_threshold_does_not_fake_unavailable_participation():
    probability = np.array([0.40] * 95 + [0.60] * 5)
    realized_returns = np.array([-0.01] * 95 + [0.20] * 5)

    threshold = _select_probability_threshold(probability, realized_returns)

    assert threshold == 0.50
    assert float((probability >= threshold).mean()) < 0.10


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


def test_model_selection_prefers_eligible_model_over_sparse_high_score():
    selection_returns = [0.50, *([0.01] * 9), *([-0.02] * 10)]
    rows = []
    for model_name, taken_rows in (
        ("eligible_model", 10),
        ("sparse_high_score_model", 1),
    ):
        rows.extend(
            {
                "model_name": model_name,
                "fold": 1,
                "selection_phase": "model_selection",
                "selection_isolation_scheme": (
                    "chronological_selection_folds_then_untouched_evaluation_folds"
                ),
                "shadow_take": index < taken_rows,
                "realized_return": realized_return,
            }
            for index, realized_return in enumerate(selection_returns)
        )
        rows.extend(
            {
                "model_name": model_name,
                "fold": 2,
                "selection_phase": "untouched_evaluation",
                "selection_isolation_scheme": (
                    "chronological_selection_folds_then_untouched_evaluation_folds"
                ),
                "shadow_take": True,
                "realized_return": realized_return,
            }
            for realized_return in (0.01, -0.01)
        )

    leaderboard = model_selection_leaderboard(pd.DataFrame(rows))
    eligible = leaderboard.loc[
        leaderboard["model_name"].eq("eligible_model")
    ].iloc[0]
    sparse = leaderboard.loc[
        leaderboard["model_name"].eq("sparse_high_score_model")
    ].iloc[0]

    assert sparse["selection_score"] > eligible["selection_score"]
    assert bool(eligible["promising"])
    assert not bool(sparse["promising"])
    assert leaderboard.iloc[0]["model_name"] == "eligible_model"
    assert bool(leaderboard.iloc[0]["chosen_model"])


def test_available_model_specs_contains_all_preregistered_model_families():
    specs = available_model_specs()
    names = {spec.name for spec in specs}
    assert "logistic_regression" in names
    assert "random_forest_regularized" in names
    assert any(spec.family == "bagged_tree_regularized" for spec in specs)
    assert any(spec.family.startswith("boosted_tree") for spec in specs)


def test_preregistered_random_forest_uses_locked_regularization_parameters():
    pipeline = _build_model_pipeline(
        "random_forest_regularized",
        numeric_features=["entry_zscore"],
        categorical_features=["pair"],
    )
    parameters = pipeline.named_steps["model"].get_params()

    assert parameters["n_estimators"] == 160
    assert parameters["max_depth"] == 8
    assert parameters["min_samples_leaf"] == 50
    assert parameters["max_features"] == "sqrt"
    assert parameters["class_weight"] == "balanced_subsample"
    assert parameters["random_state"] == 7
    assert parameters["n_jobs"] == -1
