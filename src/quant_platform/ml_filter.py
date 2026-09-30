from __future__ import annotations

import json
import pickle
import re
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from math import ceil, isfinite
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from quant_platform.backtest import (
    CostModel,
    backtest_two_leg_spread_with_ledger,
    max_drawdown,
)
from quant_platform.experiments import PairDataset
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_text,
)
from quant_platform.strategies import STRATEGIES, STRATEGY_REQUIRED_COLUMNS, StrategySpec

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None


ML_DATASET_COLUMNS = [
    "trade_id",
    "pair",
    "timeframe",
    "source_venue",
    "source_path",
    "source_registry_path",
    "registered_history_sha256",
    "feature_provenance",
    "exact_mode",
    "orientation",
    "strategy_id",
    "strategy_name",
    "family",
    "backtest_mode",
    "entry_timestamp",
    "exit_timestamp",
    "entry_bar_index",
    "exit_bar_index",
    "trade_bars",
    "signal_side",
    "label_profitable",
    "realized_return",
    "gross_trade_return",
    "trade_cost_drag",
    "max_adverse_excursion",
    "max_favorable_excursion",
    "return_aggregation",
    "return_unit",
    "entry_zscore",
    "entry_abs_zscore",
    "zscore_change_1",
    "zscore_change_3",
    "spread_level",
    "spread_change_1",
    "spread_change_3",
    "spread_vol_12",
    "spread_vol_48",
    "hedge_ratio",
    "hedge_ratio_stability",
    "beta",
    "realized_volatility_percentile",
    "cvar",
    "var",
    "tail_dependence",
    "crisis_probability",
    "liquidity_score",
    "bid_ask_spread_bps",
    "slippage_bps",
    "volume_x_usd",
    "volume_y_usd",
    "funding_x_bps",
    "funding_y_bps",
    "funding_diff_bps",
    "funding_abs_total_bps",
    "funding_bps_per_day",
    "regime",
    "regime_strategy_match",
    "cointegration_pvalue",
    "ecm_strength",
    "ecm_x",
    "ecm_y",
    "half_life",
    "hurst",
    "conditional_probability_distortion",
    "copula_calibration_score",
    "u1_given_u2",
    "u2_given_u1",
    "composite_score",
    "ml_confidence",
    "profile_match",
    "ou_optimal",
]

TARGET_COLUMN = "label_profitable"
GLOBAL_PURGED_SPLIT_SCHEME = "globally_purged_embargoed_pair_aware_timestamp_groups_v2"
MODEL_SELECTION_ISOLATION_SCHEME = "chronological_model_selection_then_untouched_evaluation_v1"
MODEL_SELECTION_PHASE = "model_selection"
UNTOUCHED_EVALUATION_PHASE = "untouched_evaluation"
MODEL_SELECTION_BOUNDARY_SCHEME = "label_complete_chronology_gap_v1"
MINIMUM_TRAINING_TAKE_RATE = 0.10
THRESHOLD_CALIBRATION_SCHEME = "training_only_minimum_participation_v1"
RETURN_COLUMN = "realized_return"
TIMESTAMP_COLUMN = "entry_timestamp"
CATEGORICAL_FEATURES = [
    "pair",
    "timeframe",
    "source_venue",
    "strategy_name",
    "family",
    "regime",
    "backtest_mode",
    "signal_side",
    "orientation",
]
NON_FEATURE_COLUMNS = {
    "trade_id",
    "source_trade_id",
    "schema_version",
    "registered_contract_id",
    "registered_execution_id",
    "registered_semantic_hypothesis_id",
    "registered_hypothesis_outcome",
    "registered_candidate",
    "accepted_stage4_survivor",
    "experiment_id",
    "native_promotion_basis",
    "shared_outcome_count",
    "shared_outcome_mean_drawdown",
    "shared_outcome_mean_return",
    "shared_outcome_win_rate",
    "shared_verified_outcome_count",
    "wizard_history_feature_count",
    "wizard_learning_feature_state",
    "wizard_same_regime_count",
    "wizard_same_regime_strategy_venue_count",
    "wizard_same_regime_strategy_venue_mean_drawdown",
    "wizard_same_regime_strategy_venue_mean_return",
    "wizard_same_regime_strategy_venue_win_rate",
    "wizard_same_strategy_family_count",
    "wizard_same_venue_count",
    "wizard_verified_same_regime_strategy_venue_count",
    "pair_group_key",
    "pair_group_id",
    "walkforward_id",
    "observed_cost_replay_id",
    "regime_attribution_id",
    "source_path",
    "source_registry_path",
    "registered_history_sha256",
    "feature_provenance",
    "exact_mode",
    "strategy_id",
    TARGET_COLUMN,
    RETURN_COLUMN,
    "gross_trade_return",
    "trade_cost_drag",
    "max_adverse_excursion",
    "max_favorable_excursion",
    "return_aggregation",
    "return_unit",
    "entry_timestamp",
    "exit_timestamp",
    "entry_bar_index",
    "exit_bar_index",
    "trade_bars",
    "label_timestamp",
    "profit_after_cost",
    "good_trade",
    "hold_bars",
    "exit_reason",
    "entry_index",
    "exit_index",
    "bars",
    "gross_factor",
    "net_factor",
    "gross_return",
    "total_fees",
    "total_slippage",
    "total_funding",
    "total_execution_risk",
    "total_partial_fill",
    "is_closed",
    "side",
    "backtest_label",
    "paper_label",
    "live_label",
    "ledger_type",
    "feature_source",
    "feature_known_at_or_before_entry",
    "feature_uses_future_data",
    "uses_dashboard_hindsight",
    "feature_completeness_score",
    "evidence_path",
    "promotion_authority",
    "testnet_candidate_authority",
    "testnet_order_authority",
    "live_trading_authorized",
    "wizard_exchange",
    "wizard_timeframe",
    "hyperliquid_interval",
    "asset_x",
    "asset_y",
}

MODEL_FEATURE_COLUMNS = tuple(
    column for column in ML_DATASET_COLUMNS if column not in NON_FEATURE_COLUMNS
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    available: bool
    unavailable_reason: str = ""


def _purged_pair_aware_splits(
    ordered: pd.DataFrame,
    *,
    n_splits: int,
    embargo_periods: int,
) -> list[tuple[np.ndarray, np.ndarray, dict[str, object]]]:
    """Build chronological panel splits with per-pair label purging and embargo."""

    if TIMESTAMP_COLUMN not in ordered.columns or "exit_timestamp" not in ordered.columns:
        raise ValueError("purged evaluation requires entry_timestamp and exit_timestamp")
    entries = pd.to_datetime(ordered[TIMESTAMP_COLUMN], utc=True, errors="coerce", format="mixed")
    exits = pd.to_datetime(ordered["exit_timestamp"], utc=True, errors="coerce", format="mixed")
    if entries.isna().any() or exits.isna().any():
        raise ValueError("purged evaluation requires valid entry and exit timestamps")
    unique_times = pd.Index(entries.drop_duplicates().sort_values())
    if len(unique_times) <= n_splits:
        raise ValueError("not enough unique entry timestamps for requested purged splits")
    pairs = (
        ordered.get("pair", pd.Series("__all__", index=ordered.index))
        .fillna("__missing_pair__")
        .astype(str)
    )
    splitter = TimeSeriesSplit(n_splits=n_splits)
    results: list[tuple[np.ndarray, np.ndarray, dict[str, object]]] = []
    for train_time_idx, test_time_idx in splitter.split(unique_times):
        train_times = unique_times.take(train_time_idx)
        test_times = unique_times.take(test_time_idx)
        train_mask = entries.isin(train_times)
        test_mask = entries.isin(test_times)
        initial_train_rows = int(train_mask.sum())
        global_test_start = entries.loc[test_mask].min()
        global_overlap_before = int((train_mask & exits.ge(global_test_start)).sum())
        train_mask.loc[train_mask & exits.ge(global_test_start)] = False
        test_pairs = sorted(set(pairs.loc[test_mask]))
        for pair in test_pairs:
            pair_test = test_mask & pairs.eq(pair)
            pair_train = train_mask & pairs.eq(pair)
            pair_test_start = entries.loc[pair_test].min()
            train_mask.loc[pair_train & exits.ge(pair_test_start)] = False
            if embargo_periods > 0:
                remaining_pair_times = (
                    entries.loc[train_mask & pairs.eq(pair)].drop_duplicates().sort_values()
                )
                embargo_times = set(remaining_pair_times.tail(embargo_periods))
                if embargo_times:
                    train_mask.loc[train_mask & pairs.eq(pair) & entries.isin(embargo_times)] = (
                        False
                    )
        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if not len(train_idx) or not len(test_idx):
            continue
        train_pairs = sorted(set(pairs.iloc[train_idx]))
        pair_overlap = sorted(set(train_pairs).intersection(test_pairs))
        global_overlap_after = int((train_mask & exits.ge(global_test_start)).sum())
        audit = {
            "split_scheme": GLOBAL_PURGED_SPLIT_SCHEME,
            "global_label_purge": True,
            "global_label_overlap_rows_before_purge": global_overlap_before,
            "global_label_overlap_rows_after_purge": global_overlap_after,
            "embargo_periods": embargo_periods,
            "purged_or_embargoed_rows": initial_train_rows - len(train_idx),
            "train_unique_timestamps": int(entries.iloc[train_idx].nunique()),
            "test_unique_timestamps": int(entries.iloc[test_idx].nunique()),
            "train_pairs": ";".join(train_pairs),
            "test_pairs": ";".join(test_pairs),
            "pair_overlap": ";".join(pair_overlap),
            "test_start": entries.iloc[test_idx].min().isoformat(),
            "test_end": entries.iloc[test_idx].max().isoformat(),
            "train_label_end_max": exits.iloc[train_idx].max().isoformat(),
        }
        results.append((train_idx, test_idx, audit))
    if not results:
        raise ValueError("no valid purged pair-aware folds were produced")
    return results


def _chronology_isolated_fold_partitions(
    ordered: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray, dict[str, object]]],
    eligible_fold_numbers: list[int],
) -> tuple[set[int], set[int], set[int], pd.Timestamp, pd.Timestamp]:
    """Reserve evaluation folds after every selection label is observable."""

    selection_fold_count = max(1, len(eligible_fold_numbers) // 2)
    if selection_fold_count >= len(eligible_fold_numbers):
        raise ValueError(
            "model selection isolation requires separate selection and untouched evaluation folds"
        )
    selection_fold_numbers = set(eligible_fold_numbers[:selection_fold_count])
    selection_label_ends = [
        pd.to_datetime(
            ordered.iloc[splits[fold_number - 1][1]]["exit_timestamp"],
            utc=True,
            errors="coerce",
        ).max()
        for fold_number in sorted(selection_fold_numbers)
    ]
    selection_label_end = max(selection_label_ends)
    remaining_fold_numbers = eligible_fold_numbers[selection_fold_count:]
    evaluation_fold_numbers = {
        fold_number
        for fold_number in remaining_fold_numbers
        if pd.to_datetime(
            ordered.iloc[splits[fold_number - 1][1]][TIMESTAMP_COLUMN],
            utc=True,
            errors="coerce",
        ).min()
        > selection_label_end
    }
    if not evaluation_fold_numbers:
        raise ValueError(
            "model selection isolation has no untouched evaluation fold "
            "after selection labels are complete"
        )
    first_evaluation_fold = min(evaluation_fold_numbers)
    evaluation_fold_numbers = {
        fold_number
        for fold_number in evaluation_fold_numbers
        if fold_number >= first_evaluation_fold
    }
    chronology_gap_fold_numbers = set(remaining_fold_numbers).difference(evaluation_fold_numbers)
    evaluation_start = min(
        pd.to_datetime(
            ordered.iloc[splits[fold_number - 1][1]][TIMESTAMP_COLUMN],
            utc=True,
            errors="coerce",
        ).min()
        for fold_number in evaluation_fold_numbers
    )
    if selection_label_end >= evaluation_start:
        raise ValueError("model selection outcomes overlap untouched evaluation")
    return (
        selection_fold_numbers,
        evaluation_fold_numbers,
        chronology_gap_fold_numbers,
        selection_label_end,
        evaluation_start,
    )


def expected_walkforward_prediction_membership(
    dataset: pd.DataFrame,
    *,
    n_splits: int,
    min_train_rows: int,
    embargo_periods: int,
) -> pd.DataFrame:
    """Rebuild the exact rows assigned to registered model evidence folds."""

    required = {"trade_id", TARGET_COLUMN, TIMESTAMP_COLUMN, "exit_timestamp"}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError("walk-forward membership dataset missing columns: " + ",".join(missing))
    ordered = dataset.copy()
    ordered[TIMESTAMP_COLUMN] = pd.to_datetime(
        ordered[TIMESTAMP_COLUMN], utc=True, errors="coerce", format="mixed"
    )
    ordered["exit_timestamp"] = pd.to_datetime(
        ordered["exit_timestamp"], utc=True, errors="coerce", format="mixed"
    )
    ordered = (
        ordered.dropna(subset=[TIMESTAMP_COLUMN, "exit_timestamp"])
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )
    trade_ids = ordered["trade_id"].fillna("").astype(str).str.strip()
    if (
        ordered.empty
        or trade_ids.eq("").any()
        or trade_ids.duplicated().any()
        or not ordered["exit_timestamp"].gt(ordered[TIMESTAMP_COLUMN]).all()
    ):
        raise ValueError("walk-forward membership requires unique causal trade rows")

    requested_splits = max(2, int(n_splits))
    requested_minimum = int(min_train_rows)
    requested_embargo = max(0, int(embargo_periods))
    if requested_minimum <= 0:
        raise ValueError("minimum train rows must be positive")
    splits = _purged_pair_aware_splits(
        ordered,
        n_splits=requested_splits,
        embargo_periods=requested_embargo,
    )
    eligible_fold_numbers = [
        fold_number
        for fold_number, (train_idx, test_idx, _) in enumerate(splits, start=1)
        if len(train_idx) >= requested_minimum
        and len(test_idx) > 0
        and ordered.iloc[train_idx][TARGET_COLUMN].nunique(dropna=True) >= 2
    ]
    (
        selection_fold_numbers,
        evaluation_fold_numbers,
        _,
        _,
        _,
    ) = _chronology_isolated_fold_partitions(
        ordered,
        splits,
        eligible_fold_numbers,
    )
    rows: list[pd.DataFrame] = []
    for fold_number, (_, test_idx, _) in enumerate(splits, start=1):
        if fold_number in selection_fold_numbers:
            selection_phase = MODEL_SELECTION_PHASE
        elif fold_number in evaluation_fold_numbers:
            selection_phase = UNTOUCHED_EVALUATION_PHASE
        else:
            continue
        membership = ordered.iloc[test_idx][["trade_id"]].copy()
        membership["fold"] = fold_number
        membership["selection_phase"] = selection_phase
        membership["walkforward_splits_requested"] = requested_splits
        membership["minimum_train_rows_requested"] = requested_minimum
        membership["embargo_periods"] = requested_embargo
        rows.append(membership)
    if not rows:
        raise ValueError("walk-forward membership has no eligible evidence rows")
    return (
        pd.concat(rows, ignore_index=True).sort_values(["fold", "trade_id"]).reset_index(drop=True)
    )


def walkforward_prediction_membership_matches(
    predictions: pd.DataFrame,
    dataset: pd.DataFrame,
    *,
    n_splits: int,
    min_train_rows: int,
    embargo_periods: int,
) -> bool:
    """Require exact row, fold, phase, and protocol membership for one model."""

    fields = (
        "trade_id",
        "fold",
        "selection_phase",
        "walkforward_splits_requested",
        "minimum_train_rows_requested",
        "embargo_periods",
    )
    if predictions.empty or not set(fields).issubset(predictions.columns):
        return False
    if "model_name" in predictions and predictions["model_name"].astype(str).nunique() != 1:
        return False
    observed = predictions[list(fields)].copy()
    observed["trade_id"] = observed["trade_id"].fillna("").astype(str).str.strip()
    if observed["trade_id"].eq("").any() or observed["trade_id"].duplicated().any():
        return False
    try:
        expected = expected_walkforward_prediction_membership(
            dataset,
            n_splits=n_splits,
            min_train_rows=min_train_rows,
            embargo_periods=embargo_periods,
        )
    except (TypeError, ValueError):
        return False
    if set(observed["trade_id"]) != set(expected["trade_id"]):
        return False
    merged = observed.merge(
        expected,
        on="trade_id",
        suffixes=("_observed", "_expected"),
        validate="one_to_one",
    )
    for field in fields[1:]:
        if field == "selection_phase":
            matches = (
                merged[f"{field}_observed"]
                .fillna("")
                .astype(str)
                .eq(merged[f"{field}_expected"].fillna("").astype(str))
            )
        else:
            observed_values = pd.to_numeric(merged[f"{field}_observed"], errors="coerce")
            expected_values = pd.to_numeric(merged[f"{field}_expected"], errors="coerce")
            matches = observed_values.notna() & observed_values.eq(expected_values)
        if not matches.all():
            return False
    return True


def build_trade_filter_dataset(
    datasets: Iterable[PairDataset],
    *,
    strategies: Iterable[StrategySpec] = STRATEGIES,
    cost_model: CostModel | None = None,
    min_rows: int = 20,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    costs = cost_model or CostModel()
    for dataset in datasets:
        frame = dataset.frame.copy()
        if len(frame) < min_rows:
            continue
        if "timestamp" not in frame.columns:
            continue
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce", format="mixed")
        valid_timestamp = timestamps.notna() & timestamps.dt.year.between(
            2009, pd.Timestamp.now(tz="UTC").year + 1
        )
        frame = frame.loc[valid_timestamp].copy()
        frame["timestamp"] = timestamps.loc[valid_timestamp]
        if len(frame) < min_rows:
            continue
        # A spread-point proxy has no trade notional with which to label a
        # supervised model. Keep it in research diagnostics only.
        if not {"price_x", "price_y", "hedge_ratio"}.issubset(frame.columns):
            continue
        for strategy in strategies:
            if strategy.signal_function is None:
                continue
            required = STRATEGY_REQUIRED_COLUMNS.get(strategy.id, {"spread"})
            if not required.issubset(frame.columns):
                continue
            signal = strategy.signal_function(frame).reindex(frame.index).fillna(0.0).astype(float)
            if not signal.ne(0.0).any():
                continue
            rows.extend(
                _candidate_rows_for_signal(
                    pair=dataset.pair,
                    frame=frame,
                    strategy=strategy,
                    signal=signal,
                    cost_model=costs,
                )
            )
    dataset_frame = pd.DataFrame(rows)
    if dataset_frame.empty:
        return pd.DataFrame(columns=ML_DATASET_COLUMNS)
    dataset_frame = dataset_frame.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    return dataset_frame.loc[
        :, [column for column in ML_DATASET_COLUMNS if column in dataset_frame.columns]
    ]


def available_model_specs() -> list[ModelSpec]:
    specs = [
        ModelSpec(name="logistic_regression", family="linear", available=True),
        ModelSpec(
            name="random_forest_regularized",
            family="bagged_tree_regularized",
            available=True,
        ),
    ]
    if XGBClassifier is not None:
        specs.append(ModelSpec(name="xgboost", family="boosted_tree", available=True))
    elif LGBMClassifier is not None:
        specs.append(ModelSpec(name="lightgbm", family="boosted_tree", available=True))
    else:
        specs.append(
            ModelSpec(
                name="gradient_boosting_fallback",
                family="boosted_tree_fallback",
                available=True,
                unavailable_reason="xgboost_and_lightgbm_not_installed",
            )
        )
    return specs


def train_trade_filter_walkforward(
    dataset: pd.DataFrame,
    *,
    output_dir: str | Path,
    n_splits: int = 5,
    min_train_rows: int = 100,
    embargo_periods: int = 1,
) -> dict[str, Path]:
    if dataset.empty:
        raise ValueError("dataset is empty")
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"dataset missing target column: {TARGET_COLUMN}")
    if dataset[TARGET_COLUMN].nunique(dropna=True) < 2:
        raise ValueError("dataset requires both profitable and unprofitable labels")

    ordered = dataset.copy()
    ordered[TIMESTAMP_COLUMN] = pd.to_datetime(
        ordered[TIMESTAMP_COLUMN], utc=True, errors="coerce", format="mixed"
    )
    if "exit_timestamp" not in ordered.columns:
        raise ValueError("dataset missing exit_timestamp required for purged evaluation")
    ordered["exit_timestamp"] = pd.to_datetime(
        ordered["exit_timestamp"], utc=True, errors="coerce", format="mixed"
    )
    ordered = (
        ordered.dropna(subset=[TIMESTAMP_COLUMN, "exit_timestamp"])
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )
    if ordered.empty or not ordered["exit_timestamp"].gt(ordered[TIMESTAMP_COLUMN]).all():
        raise ValueError("dataset requires exit_timestamp after entry_timestamp for every row")
    feature_columns = [column for column in MODEL_FEATURE_COLUMNS if column in ordered.columns]
    if not feature_columns:
        raise ValueError("dataset has no approved entry-time model features")
    categorical_features = [
        column
        for column in feature_columns
        if column in CATEGORICAL_FEATURES or not pd.api.types.is_numeric_dtype(ordered[column])
    ]
    numeric_features = [column for column in feature_columns if column not in categorical_features]

    splits = _purged_pair_aware_splits(
        ordered,
        n_splits=max(2, n_splits),
        embargo_periods=max(0, int(embargo_periods)),
    )
    eligible_fold_numbers = [
        fold_number
        for fold_number, (train_idx, test_idx, _) in enumerate(splits, start=1)
        if len(train_idx) >= min_train_rows
        and len(test_idx) > 0
        and ordered.iloc[train_idx][TARGET_COLUMN].nunique(dropna=True) >= 2
    ]
    (
        selection_fold_numbers,
        evaluation_fold_numbers,
        chronology_gap_fold_numbers,
        selection_label_end_boundary,
        evaluation_start_boundary,
    ) = _chronology_isolated_fold_partitions(
        ordered,
        splits,
        eligible_fold_numbers,
    )
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    model_artifacts: list[dict[str, Any]] = []

    for spec in available_model_specs():
        fold_results: list[dict[str, Any]] = []
        for fold_number, (train_idx, test_idx, split_audit) in enumerate(splits, start=1):
            if fold_number in selection_fold_numbers:
                selection_phase = MODEL_SELECTION_PHASE
            elif fold_number in evaluation_fold_numbers:
                selection_phase = UNTOUCHED_EVALUATION_PHASE
            else:
                continue
            train = ordered.iloc[train_idx].copy()
            test = ordered.iloc[test_idx].copy()
            if len(train) < min_train_rows or test.empty:
                continue
            if train[TARGET_COLUMN].nunique(dropna=True) < 2:
                continue
            estimator = _build_model_pipeline(spec.name, numeric_features, categorical_features)
            estimator.fit(train[feature_columns], train[TARGET_COLUMN].astype(int))

            train_probability = estimator.predict_proba(train[feature_columns])[:, 1]
            test_probability = estimator.predict_proba(test[feature_columns])[:, 1]
            threshold = _select_probability_threshold(
                train_probability, train[RETURN_COLUMN].astype(float).to_numpy()
            )
            training_take_rate = float((train_probability >= threshold).mean())
            baseline_metrics = _trade_metric_summary(test[RETURN_COLUMN].astype(float))
            filtered_metrics = _trade_metric_summary(
                test.loc[test_probability >= threshold, RETURN_COLUMN].astype(float)
            )
            precision = _safe_precision(
                test[TARGET_COLUMN].astype(int), test_probability >= threshold
            )
            recall = _safe_recall(test[TARGET_COLUMN].astype(int), test_probability >= threshold)
            auc = _safe_auc(test[TARGET_COLUMN].astype(int), test_probability)

            row = {
                "model_name": spec.name,
                "model_family": spec.family,
                "fold": fold_number,
                "walkforward_splits_requested": max(2, int(n_splits)),
                "minimum_train_rows_requested": int(min_train_rows),
                "selection_phase": selection_phase,
                "selection_isolation_scheme": MODEL_SELECTION_ISOLATION_SCHEME,
                "selection_evaluation_boundary_scheme": (MODEL_SELECTION_BOUNDARY_SCHEME),
                "chronology_gap_folds": ";".join(
                    str(value) for value in sorted(chronology_gap_fold_numbers)
                ),
                "selection_label_end_boundary": (selection_label_end_boundary.isoformat()),
                "untouched_evaluation_start_boundary": (evaluation_start_boundary.isoformat()),
                **split_audit,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "test_label_end_max": test["exit_timestamp"].max().isoformat(),
                "threshold": float(threshold),
                "threshold_calibration_scheme": THRESHOLD_CALIBRATION_SCHEME,
                "minimum_training_take_rate": MINIMUM_TRAINING_TAKE_RATE,
                "training_take_rate_at_threshold": training_take_rate,
                "training_take_rate_floor_pass": (training_take_rate >= MINIMUM_TRAINING_TAKE_RATE),
                "precision": precision,
                "recall": recall,
                "auc": auc,
                "baseline_profit_factor": baseline_metrics["profit_factor"],
                "baseline_sharpe": baseline_metrics["sharpe"],
                "baseline_drawdown": baseline_metrics["drawdown"],
                "baseline_expectancy": baseline_metrics["expectancy"],
                "baseline_trade_count": baseline_metrics["trade_count"],
                "baseline_total_return": baseline_metrics["total_return"],
                "filtered_profit_factor": filtered_metrics["profit_factor"],
                "filtered_sharpe": filtered_metrics["sharpe"],
                "filtered_drawdown": filtered_metrics["drawdown"],
                "filtered_expectancy": filtered_metrics["expectancy"],
                "filtered_trade_count": filtered_metrics["trade_count"],
                "filtered_total_return": filtered_metrics["total_return"],
                "filtered_take_rate": float((test_probability >= threshold).mean())
                if len(test_probability)
                else 0.0,
                "profit_factor_delta": filtered_metrics["profit_factor"]
                - baseline_metrics["profit_factor"],
                "sharpe_delta": filtered_metrics["sharpe"] - baseline_metrics["sharpe"],
                "drawdown_delta": filtered_metrics["drawdown"] - baseline_metrics["drawdown"],
                "expectancy_delta": filtered_metrics["expectancy"] - baseline_metrics["expectancy"],
                "trade_count_delta": filtered_metrics["trade_count"]
                - baseline_metrics["trade_count"],
            }
            fold_rows.append(row)
            fold_results.append(row)

            prediction_columns = [
                "trade_id",
                "pair",
                "strategy_id",
                "strategy_name",
                "family",
                "backtest_mode",
                "entry_timestamp",
                "exit_timestamp",
                RETURN_COLUMN,
                TARGET_COLUMN,
            ]
            prediction_columns.extend(
                column
                for column in (
                    "feature_timestamp",
                    "label_timestamp",
                    "timeframe",
                    "source_venue",
                    "regime",
                    "exact_mode",
                    "orientation",
                    "experiment_id",
                    "registered_contract_id",
                    "registered_execution_id",
                    "registered_semantic_hypothesis_id",
                    "registered_hypothesis_outcome",
                    "registered_candidate",
                    "accepted_stage4_survivor",
                )
                if column in test.columns
            )
            prediction_frame = test[prediction_columns].copy()
            prediction_frame["model_name"] = spec.name
            prediction_frame["fold"] = fold_number
            prediction_frame["walkforward_splits_requested"] = max(2, int(n_splits))
            prediction_frame["minimum_train_rows_requested"] = int(min_train_rows)
            prediction_frame["selection_phase"] = selection_phase
            prediction_frame["selection_isolation_scheme"] = MODEL_SELECTION_ISOLATION_SCHEME
            prediction_frame["selection_evaluation_boundary_scheme"] = (
                MODEL_SELECTION_BOUNDARY_SCHEME
            )
            prediction_frame["chronology_gap_folds"] = ";".join(
                str(value) for value in sorted(chronology_gap_fold_numbers)
            )
            prediction_frame["selection_label_end_boundary"] = (
                selection_label_end_boundary.isoformat()
            )
            prediction_frame["untouched_evaluation_start_boundary"] = (
                evaluation_start_boundary.isoformat()
            )
            prediction_frame["probability_profitable"] = test_probability
            prediction_frame["shadow_take"] = test_probability >= threshold
            prediction_frame["threshold"] = threshold
            prediction_frame["threshold_calibration_scheme"] = THRESHOLD_CALIBRATION_SCHEME
            prediction_frame["minimum_training_take_rate"] = MINIMUM_TRAINING_TAKE_RATE
            prediction_frame["training_take_rate_at_threshold"] = training_take_rate
            prediction_frame["training_take_rate_floor_pass"] = (
                training_take_rate >= MINIMUM_TRAINING_TAKE_RATE
            )
            prediction_frame["split_scheme"] = split_audit["split_scheme"]
            prediction_frame["embargo_periods"] = split_audit["embargo_periods"]
            prediction_frame["global_label_purge"] = split_audit["global_label_purge"]
            prediction_frame["global_label_overlap_rows_after_purge"] = split_audit[
                "global_label_overlap_rows_after_purge"
            ]
            prediction_frame["test_start"] = split_audit["test_start"]
            prediction_frame["train_label_end_max"] = split_audit["train_label_end_max"]
            prediction_rows.extend(prediction_frame.to_dict(orient="records"))

        if not fold_results:
            continue

        fold_frame = pd.DataFrame(fold_results)
        selection_frame = fold_frame.loc[
            fold_frame["selection_phase"].eq(MODEL_SELECTION_PHASE)
        ].copy()
        evaluation_frame = fold_frame.loc[
            fold_frame["selection_phase"].eq(UNTOUCHED_EVALUATION_PHASE)
        ].copy()
        if selection_frame.empty or evaluation_frame.empty:
            continue
        selection_label_end = pd.to_datetime(
            selection_frame["test_label_end_max"], utc=True, errors="coerce"
        ).max()
        evaluation_start = pd.to_datetime(
            evaluation_frame["test_start"], utc=True, errors="coerce"
        ).min()
        if (
            pd.isna(selection_label_end)
            or pd.isna(evaluation_start)
            or selection_label_end >= evaluation_start
        ):
            raise ValueError("model selection outcomes overlap untouched evaluation")
        aggregate = {
            "model_name": spec.name,
            "model_family": spec.family,
            "folds": int(len(fold_frame)),
            "selection_folds": int(len(selection_frame)),
            "untouched_evaluation_folds": int(len(evaluation_frame)),
            "selection_isolation_scheme": MODEL_SELECTION_ISOLATION_SCHEME,
            "selection_evaluation_boundary_scheme": (MODEL_SELECTION_BOUNDARY_SCHEME),
            "chronology_gap_folds": ";".join(
                str(value) for value in sorted(chronology_gap_fold_numbers)
            ),
            "chronology_gap_fold_count": len(chronology_gap_fold_numbers),
            "threshold_calibration_scheme": THRESHOLD_CALIBRATION_SCHEME,
            "minimum_training_take_rate": MINIMUM_TRAINING_TAKE_RATE,
            "selection_label_end_max": selection_label_end.isoformat(),
            "untouched_evaluation_start": evaluation_start.isoformat(),
            "median_baseline_profit_factor": float(
                selection_frame["baseline_profit_factor"].median()
            ),
            "median_baseline_sharpe": float(selection_frame["baseline_sharpe"].median()),
            "worst_baseline_drawdown": float(selection_frame["baseline_drawdown"].max()),
            "median_baseline_expectancy": float(selection_frame["baseline_expectancy"].median()),
            "total_baseline_trades": int(selection_frame["baseline_trade_count"].sum()),
            "median_filtered_profit_factor": float(
                selection_frame["filtered_profit_factor"].median()
            ),
            "median_filtered_sharpe": float(selection_frame["filtered_sharpe"].median()),
            "worst_filtered_drawdown": float(selection_frame["filtered_drawdown"].max()),
            "median_filtered_expectancy": float(selection_frame["filtered_expectancy"].median()),
            "total_filtered_trades": int(selection_frame["filtered_trade_count"].sum()),
            "median_precision": float(selection_frame["precision"].median()),
            "median_recall": float(selection_frame["recall"].median()),
            "median_auc": float(selection_frame["auc"].median()),
            "median_take_rate": float(selection_frame["filtered_take_rate"].median()),
            "minimum_selection_training_take_rate": float(
                selection_frame["training_take_rate_at_threshold"].min()
            ),
            "selection_training_take_rate_floor_pass": bool(
                selection_frame["training_take_rate_floor_pass"].all()
            ),
            "profit_factor_delta": float(selection_frame["profit_factor_delta"].median()),
            "sharpe_delta": float(selection_frame["sharpe_delta"].median()),
            "drawdown_delta": float(selection_frame["drawdown_delta"].median()),
            "expectancy_delta": float(selection_frame["expectancy_delta"].median()),
            "trade_count_delta": int(round(float(selection_frame["trade_count_delta"].median()))),
            "promising": _promising_fold_frame(selection_frame),
            "evaluation_median_baseline_profit_factor": float(
                evaluation_frame["baseline_profit_factor"].median()
            ),
            "evaluation_median_baseline_sharpe": float(
                evaluation_frame["baseline_sharpe"].median()
            ),
            "evaluation_worst_baseline_drawdown": float(
                evaluation_frame["baseline_drawdown"].max()
            ),
            "evaluation_total_baseline_trades": int(evaluation_frame["baseline_trade_count"].sum()),
            "evaluation_median_filtered_profit_factor": float(
                evaluation_frame["filtered_profit_factor"].median()
            ),
            "evaluation_median_filtered_sharpe": float(
                evaluation_frame["filtered_sharpe"].median()
            ),
            "evaluation_worst_filtered_drawdown": float(
                evaluation_frame["filtered_drawdown"].max()
            ),
            "evaluation_total_filtered_trades": int(evaluation_frame["filtered_trade_count"].sum()),
            "evaluation_median_take_rate": float(evaluation_frame["filtered_take_rate"].median()),
            "minimum_evaluation_training_take_rate": float(
                evaluation_frame["training_take_rate_at_threshold"].min()
            ),
            "evaluation_training_take_rate_floor_pass": bool(
                evaluation_frame["training_take_rate_floor_pass"].all()
            ),
            "evaluation_profit_factor_delta": float(
                evaluation_frame["profit_factor_delta"].median()
            ),
            "evaluation_sharpe_delta": float(evaluation_frame["sharpe_delta"].median()),
            "evaluation_drawdown_delta": float(evaluation_frame["drawdown_delta"].median()),
            "dependency_note": spec.unavailable_reason,
            "evaluation_scheme": GLOBAL_PURGED_SPLIT_SCHEME,
            "embargo_periods": max(0, int(embargo_periods)),
        }
        aggregate["selection_score"] = _model_selection_score(aggregate)
        aggregate_rows.append(aggregate)

        full_estimator = _build_model_pipeline(spec.name, numeric_features, categorical_features)
        full_estimator.fit(ordered[feature_columns], ordered[TARGET_COLUMN].astype(int))
        full_probability = full_estimator.predict_proba(ordered[feature_columns])[:, 1]
        full_threshold = _select_probability_threshold(
            full_probability, ordered[RETURN_COLUMN].astype(float).to_numpy()
        )
        artifact = {
            "model_name": spec.name,
            "model_family": spec.family,
            "threshold": float(full_threshold),
            "threshold_calibration_scheme": THRESHOLD_CALIBRATION_SCHEME,
            "minimum_training_take_rate": MINIMUM_TRAINING_TAKE_RATE,
            "training_take_rate_at_threshold": float((full_probability >= full_threshold).mean()),
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "trained_until": ordered[TIMESTAMP_COLUMN].max().isoformat(),
            "estimator": full_estimator,
            "dependency_note": spec.unavailable_reason,
            "evaluation_scheme": GLOBAL_PURGED_SPLIT_SCHEME,
            "selection_isolation_scheme": MODEL_SELECTION_ISOLATION_SCHEME,
            "selection_evaluation_boundary_scheme": (MODEL_SELECTION_BOUNDARY_SCHEME),
            "chronology_gap_folds": tuple(sorted(chronology_gap_fold_numbers)),
            "selection_label_end_boundary": (selection_label_end_boundary.isoformat()),
            "untouched_evaluation_start_boundary": (evaluation_start_boundary.isoformat()),
            "embargo_periods": max(0, int(embargo_periods)),
        }
        model_artifacts.append(artifact)
    if not fold_rows or not model_artifacts:
        raise ValueError("no valid walk-forward folds were produced")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "ml_trade_filter_dataset.csv"
    fold_path = output / "ml_trade_filter_walkforward_folds.csv"
    prediction_path = output / "ml_trade_filter_walkforward_predictions.csv"
    leaderboard_path = output / "ml_trade_filter_selection_leaderboard.csv"
    summary_path = output / "ml_trade_filter_comparison_summary.csv"
    best_model_path = output / "ml_trade_filter_best_model.pkl"
    manifest_path = output / "ml_trade_filter_manifest.json"

    atomic_write_csv(ordered, dataset_path, index=False)
    atomic_write_csv(pd.DataFrame(fold_rows).sort_values(["model_name", "fold"]), fold_path, index=False)
    predictions_frame = pd.DataFrame(prediction_rows).sort_values(["model_name", "entry_timestamp"])
    atomic_write_csv(predictions_frame, prediction_path, index=False)
    leaderboard = model_selection_leaderboard(predictions_frame)
    if leaderboard.empty:
        raise ValueError("model selection leaderboard could not be replayed")
    atomic_write_csv(leaderboard, leaderboard_path, index=False)
    summary_frame = pd.DataFrame(aggregate_rows).sort_values(
        ["promising", "selection_score", "expectancy_delta", "profit_factor_delta", "sharpe_delta"],
        ascending=[False, False, False, False, False],
    )
    chosen_model_name = str(leaderboard.iloc[0]["model_name"])
    preferred_summary = summary_frame.loc[
        summary_frame["model_name"].astype(str).eq(chosen_model_name)
    ].iloc[0]
    artifact_index = {str(artifact["model_name"]): artifact for artifact in model_artifacts}
    chosen_artifact = artifact_index.get(chosen_model_name)
    if chosen_artifact is None:
        raise ValueError(f"chosen model artifact missing for {chosen_model_name}")
    atomic_write_csv(summary_frame, summary_path, index=False)

    atomic_write_bytes(best_model_path, pickle.dumps(chosen_artifact))
    best_model_sha256 = sha256(best_model_path.read_bytes()).hexdigest()
    manifest = {
        "chosen_model": chosen_artifact["model_name"],
        "chosen_model_selection_eligible": bool(leaderboard.iloc[0]["promising"]),
        "selection_eligible_models": int(leaderboard["promising"].astype(bool).sum()),
        "selection_outcome": (
            "ELIGIBLE_MODEL_SELECTED"
            if bool(leaderboard.iloc[0]["promising"])
            else "NO_ELIGIBLE_MODEL_DIAGNOSTIC_ONLY"
        ),
        "trained_until": chosen_artifact["trained_until"],
        "best_model_sha256": best_model_sha256,
        "threshold": chosen_artifact["threshold"],
        "threshold_calibration_scheme": chosen_artifact["threshold_calibration_scheme"],
        "minimum_training_take_rate": chosen_artifact["minimum_training_take_rate"],
        "training_take_rate_at_threshold": chosen_artifact["training_take_rate_at_threshold"],
        "artifacts_written": [
            str(path)
            for path in (
                dataset_path,
                fold_path,
                prediction_path,
                leaderboard_path,
                summary_path,
                best_model_path,
            )
        ],
        "models_evaluated": [artifact["model_name"] for artifact in model_artifacts],
        "evaluation_scheme": GLOBAL_PURGED_SPLIT_SCHEME,
        "selection_isolation_scheme": MODEL_SELECTION_ISOLATION_SCHEME,
        "selection_evaluation_boundary_scheme": (MODEL_SELECTION_BOUNDARY_SCHEME),
        "chronology_gap_folds": sorted(chronology_gap_fold_numbers),
        "selection_label_end_boundary": (selection_label_end_boundary.isoformat()),
        "untouched_evaluation_start_boundary": (evaluation_start_boundary.isoformat()),
        "selection_folds": int(preferred_summary["selection_folds"]),
        "untouched_evaluation_folds": int(preferred_summary["untouched_evaluation_folds"]),
        "embargo_periods": max(0, int(embargo_periods)),
        "walkforward_splits_requested": max(2, int(n_splits)),
        "minimum_train_rows_requested": int(min_train_rows),
    }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "dataset": dataset_path,
        "folds": fold_path,
        "predictions": prediction_path,
        "selection_leaderboard": leaderboard_path,
        "summary": summary_path,
        "best_model": best_model_path,
        "manifest": manifest_path,
    }


def shadow_trade_filter_predictions(
    dataset: pd.DataFrame,
    *,
    model_artifact_path: str | Path,
    output_path: str | Path,
    expected_model_sha256: str,
) -> Path:
    artifact_path = Path(model_artifact_path)
    expected_hash = str(expected_model_sha256).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("model artifact SHA-256 is required before deserialization")
    actual_hash = sha256(artifact_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("model artifact SHA-256 mismatch; refusing deserialization")
    with artifact_path.open("rb") as handle:
        artifact = pickle.load(handle)
    estimator = artifact["estimator"]
    feature_columns = list(artifact["feature_columns"])
    threshold = float(artifact["threshold"])

    available = dataset.copy()
    available[TIMESTAMP_COLUMN] = pd.to_datetime(
        available[TIMESTAMP_COLUMN], utc=True, errors="coerce", format="mixed"
    )
    available = (
        available.dropna(subset=[TIMESTAMP_COLUMN])
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )
    for column in feature_columns:
        if column not in available.columns:
            available[column] = np.nan

    probabilities = estimator.predict_proba(available[feature_columns])[:, 1]
    shadow = available[
        [
            "trade_id",
            "pair",
            "strategy_id",
            "strategy_name",
            "family",
            "backtest_mode",
            "entry_timestamp",
            "exit_timestamp",
            RETURN_COLUMN,
            TARGET_COLUMN,
        ]
    ].copy()
    shadow["model_name"] = artifact["model_name"]
    shadow["threshold"] = threshold
    shadow["probability_profitable"] = probabilities
    shadow["shadow_take"] = probabilities >= threshold
    shadow["trained_until"] = artifact.get("trained_until", "")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(shadow, output, index=False)
    return output


def shadow_model_branch_comparison(
    predictions: pd.DataFrame,
    *,
    pairs: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"model_name", "pair", RETURN_COLUMN, "shadow_take"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {','.join(sorted(missing))}")

    frame = predictions.copy()
    if pairs:
        requested_pairs = tuple(pair for pair in pairs if pair)
        frame = frame[frame["pair"].isin(requested_pairs)].copy()
    else:
        requested_pairs = tuple(sorted(frame["pair"].dropna().astype(str).unique()))
    if frame.empty:
        raise ValueError("no prediction rows remain after pair filtering")

    frame[RETURN_COLUMN] = pd.to_numeric(frame[RETURN_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[RETURN_COLUMN, "model_name", "pair"]).copy()
    frame["shadow_take"] = _bool_series(frame["shadow_take"])
    if frame.empty:
        raise ValueError("no prediction rows with realized returns are available")

    model_rows = [
        _shadow_comparison_row(
            scope="aggregate",
            model_name=str(model_name),
            pairs=requested_pairs,
            frame=model_frame,
        )
        for model_name, model_frame in frame.groupby("model_name", sort=True)
    ]
    pair_rows = [
        _shadow_pair_comparison_row(str(model_name), str(pair), pair_frame)
        for (model_name, pair), pair_frame in frame.groupby(["model_name", "pair"], sort=True)
    ]
    model_report = pd.DataFrame(model_rows).sort_values(
        ["taken_mean_return", "taken_profit_factor", "taken_sharpe"],
        ascending=[False, False, False],
    )
    pair_report = pd.DataFrame(pair_rows).sort_values(["pair", "model_name"]).reset_index(drop=True)
    return model_report.reset_index(drop=True), pair_report


def _candidate_rows_for_signal(
    *,
    pair: str,
    frame: pd.DataFrame,
    strategy: StrategySpec,
    signal: pd.Series,
    cost_model: CostModel,
) -> list[dict[str, Any]]:
    detailed, backtest_mode = _detailed_backtest_frame(frame, signal, cost_model)
    if backtest_mode != "two_leg":
        return []
    entry_mask = detailed["signal_target"].ne(0.0) & detailed["signal_target"].shift(1).fillna(
        0.0
    ).eq(0.0)
    exit_mask = detailed["signal_target"].eq(0.0) & detailed["signal_target"].shift(1).fillna(
        0.0
    ).ne(0.0)
    entry_positions = np.flatnonzero(entry_mask.to_numpy())
    exit_positions = np.flatnonzero(exit_mask.to_numpy())
    rows: list[dict[str, Any]] = []
    timeframe = ""
    for column in ("timeframe", "interval"):
        if column in frame.columns:
            values = frame[column].dropna().astype(str)
            if not values.empty:
                timeframe = _normalize_timeframe(values.iloc[0])
                break

    for ordinal, entry_pos in enumerate(entry_positions, start=1):
        future_exits = exit_positions[exit_positions >= entry_pos]
        # Open positions at the end of a history are right-censored. They have
        # no valid realized label and must not enter supervised training.
        if not len(future_exits):
            continue
        exit_pos = int(future_exits[0])
        if exit_pos <= entry_pos:
            continue
        segment = detailed.iloc[entry_pos : exit_pos + 1]
        entry_row = detailed.iloc[entry_pos]
        source_venue = str(entry_row.get("exchange", "unknown") or "unknown").strip().lower()
        source_path = str(entry_row.get("source_path", "") or "").strip()
        source_registry_path = str(entry_row.get("source_registry_path", "") or "").strip()
        registered_history_sha256 = str(
            entry_row.get("registered_history_sha256", "") or ""
        ).strip()
        feature_provenance = ";".join(
            f"{column.removesuffix('_feature_source')}={entry_row.get(column)}"
            for column in sorted(frame.columns)
            if column.endswith("_feature_source") and str(entry_row.get(column, "") or "").strip()
        )
        trade_id = (
            f"{pair}|{source_venue}|{timeframe}|{strategy.id}|"
            f"{pd.Timestamp(entry_row['timestamp']).isoformat()}|{ordinal}"
        )
        net_path = _compounded_return_path(segment["net_return"])
        gross_path = _compounded_return_path(segment["gross_return"])
        realized_return = float(net_path.iloc[-1]) if not net_path.empty else 0.0
        gross_return = float(gross_path.iloc[-1]) if not gross_path.empty else 0.0
        row = {
            "trade_id": trade_id,
            "pair": pair,
            "timeframe": timeframe,
            "source_venue": source_venue,
            "source_path": source_path,
            "source_registry_path": source_registry_path,
            "registered_history_sha256": registered_history_sha256,
            "feature_provenance": feature_provenance,
            "exact_mode": str(entry_row.get("exact_mode", "") or ""),
            "orientation": str(entry_row.get("orientation", "") or ""),
            "strategy_id": strategy.id,
            "strategy_name": strategy.name,
            "family": strategy.family,
            "backtest_mode": backtest_mode,
            "entry_timestamp": pd.Timestamp(entry_row["timestamp"]).isoformat(),
            "exit_timestamp": pd.Timestamp(detailed.iloc[exit_pos]["timestamp"]).isoformat(),
            "entry_bar_index": int(entry_pos),
            "exit_bar_index": int(exit_pos),
            "trade_bars": int(exit_pos - entry_pos + 1),
            "signal_side": "long_spread"
            if float(entry_row["signal_target"]) > 0
            else "short_spread",
            "label_profitable": int(realized_return > 0.0),
            "realized_return": realized_return,
            "gross_trade_return": gross_return,
            "trade_cost_drag": max(gross_return - realized_return, 0.0),
            "max_adverse_excursion": min(float(net_path.min()), 0.0) if not net_path.empty else 0.0,
            "max_favorable_excursion": max(float(net_path.max()), 0.0)
            if not net_path.empty
            else 0.0,
            "return_aggregation": "compounded_bar_returns_zero_floor",
            "return_unit": "fraction_of_equity",
        }
        row.update(_entry_feature_row(detailed, entry_pos))
        rows.append(row)
    return rows


def _compounded_return_path(returns: pd.Series) -> pd.Series:
    values = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # A fractional equity return cannot continue below zero. Any single-bar loss
    # at or below -100% therefore produces a terminal -100% path.
    factors = (1.0 + values).clip(lower=0.0)
    return factors.cumprod() - 1.0


def _normalize_timeframe(value: object) -> str:
    text = str(value or "").strip().lower().replace("_", "").replace(" ", "")
    aliases = {
        "1day": "1d",
        "daily": "1d",
        "day": "1d",
        "1hour": "1h",
        "60min": "1h",
        "60mins": "1h",
        "15min": "15m",
        "15mins": "15m",
        "5min": "5m",
        "5mins": "5m",
    }
    return aliases.get(text, text)


def _entry_feature_row(frame: pd.DataFrame, entry_pos: int) -> dict[str, Any]:
    row = frame.iloc[entry_pos]
    source_by_feature = {
        "entry_zscore": "zscore",
        "zscore_change_1": "zscore_change_1",
        "zscore_change_3": "zscore_change_3",
        "spread_level": "spread",
        "spread_change_1": "spread_change_1",
        "spread_change_3": "spread_change_3",
        "spread_vol_12": "spread_vol_12",
        "spread_vol_48": "spread_vol_48",
        "hedge_ratio": "hedge_ratio",
        "hedge_ratio_stability": "hedge_ratio_stability",
        "beta": "beta",
        "realized_volatility_percentile": "realized_volatility_percentile",
        "cvar": "cvar",
        "var": "var",
        "tail_dependence": "tail_dependence",
        "crisis_probability": "crisis_probability",
        "liquidity_score": "liquidity_score",
        "bid_ask_spread_bps": "bid_ask_spread_bps",
        "slippage_bps": "slippage_bps",
        "volume_x_usd": "volume_x_usd",
        "volume_y_usd": "volume_y_usd",
        "funding_x_bps": "funding_x_bps",
        "funding_y_bps": "funding_y_bps",
        "funding_diff_bps": "funding_diff_bps",
        "funding_abs_total_bps": "funding_abs_total_bps",
        "funding_bps_per_day": "funding_bps_per_day",
        "regime_strategy_match": "regime_strategy_match",
        "cointegration_pvalue": "cointegration_pvalue",
        "ecm_strength": "ecm_strength",
        "ecm_x": "ecm_x",
        "ecm_y": "ecm_y",
        "half_life": "half_life",
        "hurst": "hurst",
        "conditional_probability_distortion": "conditional_probability_distortion",
        "copula_calibration_score": "copula_calibration_score",
        "u1_given_u2": "u1_given_u2",
        "u2_given_u1": "u2_given_u1",
        "composite_score": "composite_score",
        "ml_confidence": "ml_confidence",
        "profile_match": "profile_match",
        "ou_optimal": "ou_optimal",
    }
    features = {
        feature: _entry_numeric_value(row, source)
        for feature, source in source_by_feature.items()
    }
    features["entry_abs_zscore"] = abs(features["entry_zscore"])
    features["regime"] = str(row.get("regime", "UNKNOWN") or "UNKNOWN")
    return features


def _entry_numeric_value(row: pd.Series, column: str) -> float:
    if isinstance(row.get(column), (bool, np.bool_)):
        return float("nan")
    value = pd.to_numeric(
        pd.Series([row.get(column, np.nan)]), errors="coerce"
    ).iloc[0]
    return float(value) if pd.notna(value) and np.isfinite(value) else float("nan")


def _shadow_comparison_row(
    *,
    scope: str,
    model_name: str,
    pairs: Iterable[str],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    baseline = _trade_metric_summary(frame[RETURN_COLUMN])
    taken = frame[frame["shadow_take"]]
    skipped = frame[~frame["shadow_take"]]
    taken_metrics = _trade_metric_summary(taken[RETURN_COLUMN])
    return {
        "scope": scope,
        "model_name": model_name,
        "pairs": ";".join(pairs),
        "rows": int(len(frame)),
        "take_rows": int(len(taken)),
        "take_rate": float(len(taken) / len(frame)) if len(frame) else 0.0,
        "baseline_trade_count": int(baseline["trade_count"]),
        "baseline_mean_return": float(baseline["expectancy"]),
        "baseline_total_return": float(baseline["total_return"]),
        "baseline_profit_factor": float(baseline["profit_factor"]),
        "baseline_sharpe": float(baseline["sharpe"]),
        "baseline_drawdown": float(baseline["drawdown"]),
        "taken_trade_count": int(taken_metrics["trade_count"]),
        "taken_mean_return": float(taken_metrics["expectancy"]),
        "taken_total_return": float(taken_metrics["total_return"]),
        "taken_profit_factor": float(taken_metrics["profit_factor"]),
        "taken_sharpe": float(taken_metrics["sharpe"]),
        "taken_drawdown": float(taken_metrics["drawdown"]),
        "taken_win_rate": float((taken[RETURN_COLUMN] > 0).mean()) if len(taken) else 0.0,
        "skipped_mean_return": float(skipped[RETURN_COLUMN].mean()) if len(skipped) else 0.0,
    }


def _shadow_pair_comparison_row(model_name: str, pair: str, frame: pd.DataFrame) -> dict[str, Any]:
    row = _shadow_comparison_row(scope="pair", model_name=model_name, pairs=(pair,), frame=frame)
    row.pop("scope")
    row.pop("pairs")
    row.pop("baseline_trade_count")
    row.pop("baseline_total_return")
    row.pop("taken_trade_count")
    row.pop("taken_total_return")
    row["pair"] = pair
    ordered_columns = [
        "model_name",
        "pair",
        "rows",
        "take_rows",
        "take_rate",
        "baseline_mean_return",
        "baseline_profit_factor",
        "baseline_sharpe",
        "baseline_drawdown",
        "taken_mean_return",
        "taken_profit_factor",
        "taken_sharpe",
        "taken_drawdown",
        "taken_win_rate",
        "skipped_mean_return",
    ]
    return {column: row[column] for column in ordered_columns}


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def _detailed_backtest_frame(
    frame: pd.DataFrame, signal: pd.Series, cost_model: CostModel
) -> tuple[pd.DataFrame, str]:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data = data.dropna(subset=["timestamp"]).reset_index(drop=True)
    data["signal_target"] = (
        signal.reindex(frame.index)
        .fillna(0.0)
        .astype(float)
        .iloc[: len(data)]
        .reset_index(drop=True)
    )
    data["spread"] = pd.to_numeric(
        data.get("spread", pd.Series(0.0, index=data.index)), errors="coerce"
    ).fillna(0.0)
    data["zscore"] = pd.to_numeric(
        data.get("zscore", pd.Series(0.0, index=data.index)), errors="coerce"
    ).fillna(0.0)
    data["zscore_change_1"] = data["zscore"].diff().fillna(0.0)
    data["zscore_change_3"] = data["zscore"].diff(3).fillna(0.0)
    data["spread_change_1"] = data["spread"].diff().fillna(0.0)
    data["spread_change_3"] = data["spread"].diff(3).fillna(0.0)
    data["spread_vol_12"] = data["spread"].diff().rolling(12, min_periods=2).std().fillna(0.0)
    data["spread_vol_48"] = data["spread"].diff().rolling(48, min_periods=2).std().fillna(0.0)
    if {"funding_x_bps", "funding_y_bps"}.issubset(data.columns):
        funding_x = pd.to_numeric(data["funding_x_bps"], errors="coerce")
        funding_y = pd.to_numeric(data["funding_y_bps"], errors="coerce")
        data["funding_diff_bps"] = funding_y - funding_x
        data["funding_abs_total_bps"] = funding_x.abs() + funding_y.abs()
    else:
        # The cost model may supply an explicit simulation assumption, but it
        # must not become an observed entry feature for the ML model.
        data["funding_diff_bps"] = np.nan
        data["funding_abs_total_bps"] = np.nan

    if {"price_x", "price_y", "hedge_ratio"}.issubset(data.columns):
        backtest_mode = "two_leg"
        _, ledger = backtest_two_leg_spread_with_ledger(
            data,
            data["signal_target"],
            cost_model,
            interval=(
                _normalize_timeframe(data["interval"].dropna().iloc[0])
                if "interval" in data.columns and not data["interval"].dropna().empty
                else None
            ),
        )
        gross_return = ledger.bar_ledger["gross_return"].reset_index(drop=True)
        net_return = ledger.bar_ledger["net_return"].reset_index(drop=True)
        cost_drag = (
            ledger.bar_ledger[["fees", "slippage", "funding", "execution_risk", "partial_fill"]]
            .sum(axis=1)
            .reset_index(drop=True)
        )
    elif "spread" in data.columns:
        backtest_mode = "spread"
        spread_return = data["spread"].diff().fillna(0.0)
        signal_position = data["signal_target"].shift(1).fillna(0.0)
        gross_return = signal_position * spread_return
        turnover = data["signal_target"].diff().abs().fillna(data["signal_target"].abs())
        trading_cost = turnover * cost_model.round_trip_cost() / 2.0
        funding_cost = signal_position.abs() * cost_model.funding_per_bar()
        net_return = gross_return - trading_cost - funding_cost
        cost_drag = trading_cost + funding_cost
    else:
        raise ValueError(
            "candidate labels require spread or complete price_x/price_y/hedge_ratio inputs"
        )

    data["gross_return"] = gross_return.astype(float)
    data["net_return"] = net_return.astype(float)
    data["cost_drag"] = pd.to_numeric(cost_drag, errors="coerce").fillna(0.0).astype(float)
    return data, backtest_mode


def _build_model_pipeline(
    name: str, numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    if name == "logistic_regression":
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif name == "random_forest_regularized":
        estimator = RandomForestClassifier(
            n_estimators=160,
            max_depth=8,
            min_samples_leaf=50,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=7,
            n_jobs=-1,
        )
    elif name == "xgboost" and XGBClassifier is not None:
        estimator = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=7,
        )
    elif name == "lightgbm" and LGBMClassifier is not None:
        estimator = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=7,
        )
    else:
        estimator = GradientBoostingClassifier(random_state=7)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def _trade_metric_summary(returns: pd.Series) -> dict[str, float]:
    series = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if series.empty:
        return {
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "drawdown": 0.0,
            "expectancy": 0.0,
            "trade_count": 0,
            "total_return": 0.0,
        }
    wins = series[series > 0]
    losses = series[series < 0]
    if losses.empty:
        profit_factor = float("inf") if not wins.empty else 0.0
    else:
        profit_factor = float(wins.sum() / abs(losses.sum()))
    std = float(series.std(ddof=1)) if len(series) >= 2 else 0.0
    sharpe = 0.0 if std == 0.0 else float(np.sqrt(len(series)) * series.mean() / std)
    equity = (1.0 + series).cumprod()
    return {
        "profit_factor": float(5.0 if not isfinite(profit_factor) else profit_factor),
        "sharpe": sharpe,
        "drawdown": float(max_drawdown(equity)),
        "expectancy": float(series.mean()),
        "trade_count": int(len(series)),
        "total_return": float(equity.iloc[-1] - 1.0),
    }


def _select_probability_threshold(
    probability: np.ndarray,
    realized_returns: np.ndarray,
    *,
    minimum_take_rate: float = MINIMUM_TRAINING_TAKE_RATE,
) -> float:
    probability = np.asarray(probability, dtype=float)
    realized_returns = np.asarray(realized_returns, dtype=float)
    if probability.shape != realized_returns.shape:
        raise ValueError("probability and realized_returns must have identical shape")
    if probability.size == 0:
        return 0.50
    if not 0.0 < minimum_take_rate <= 1.0:
        raise ValueError("minimum_take_rate must be in (0, 1]")
    minimum_accepted = max(1, ceil(probability.size * minimum_take_rate))
    best_threshold = 0.50
    best_score = float("-inf")
    for threshold in np.arange(0.50, 0.81, 0.05):
        accepted_mask = probability >= threshold
        if int(accepted_mask.sum()) < minimum_accepted:
            continue
        accepted = realized_returns[accepted_mask]
        metrics = _trade_metric_summary(pd.Series(accepted, dtype="float64"))
        score = _threshold_quality_score(metrics)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _threshold_quality_score(metrics: dict[str, float]) -> float:
    trade_count = int(metrics.get("trade_count", 0) or 0)
    if trade_count <= 0:
        return float("-inf")

    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    drawdown = float(metrics.get("drawdown", 0.0) or 0.0)
    expectancy = float(metrics.get("expectancy", 0.0) or 0.0)
    total_return = float(metrics.get("total_return", 0.0) or 0.0)

    score = profit_factor + sharpe * 0.5 - drawdown + expectancy * 10.0 + total_return * 0.2
    if trade_count < 10:
        score -= (10 - trade_count) * 0.05
    return float(score)


def _safe_precision(y_true: pd.Series, y_pred: np.ndarray) -> float:
    try:
        return float(precision_score(y_true, y_pred, zero_division=0))
    except Exception:
        return 0.0


def _safe_recall(y_true: pd.Series, y_pred: np.ndarray) -> float:
    try:
        return float(recall_score(y_true, y_pred, zero_division=0))
    except Exception:
        return 0.0


def _safe_auc(y_true: pd.Series, probability: np.ndarray) -> float:
    try:
        if len(pd.Series(y_true).unique()) < 2:
            return 0.0
        return float(roc_auc_score(y_true, probability))
    except Exception:
        return 0.0


def _promising_fold_frame(frame: pd.DataFrame) -> bool:
    median_pf_gain = float(frame["profit_factor_delta"].median())
    median_sharpe_gain = float(frame["sharpe_delta"].median())
    median_dd_gain = float(frame["drawdown_delta"].median())
    median_expectancy_gain = float(frame["expectancy_delta"].median())
    median_take_rate = float(frame["filtered_take_rate"].median())
    training_floor_pass = bool(
        "training_take_rate_floor_pass" not in frame.columns
        or frame["training_take_rate_floor_pass"].map(bool).all()
    )
    return (
        median_pf_gain > 0.0
        and median_sharpe_gain > 0.0
        and median_dd_gain <= 0.0
        and median_expectancy_gain > 0.0
        and median_take_rate >= 0.10
        and training_floor_pass
    )


def _model_selection_score(aggregate: dict[str, Any]) -> float:
    promising_bonus = 2.0 if bool(aggregate.get("promising", False)) else 0.0
    profit_factor_delta = float(aggregate.get("profit_factor_delta", 0.0) or 0.0)
    sharpe_delta = float(aggregate.get("sharpe_delta", 0.0) or 0.0)
    expectancy_delta = float(aggregate.get("expectancy_delta", 0.0) or 0.0)
    filtered_drawdown = float(aggregate.get("worst_filtered_drawdown", 1.0) or 1.0)
    drawdown_penalty = max(filtered_drawdown - 0.30, 0.0) * 2.0
    return float(
        promising_bonus
        + profit_factor_delta
        + sharpe_delta * 0.5
        + expectancy_delta * 20.0
        - drawdown_penalty
    )


def model_selection_leaderboard(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "selection_rank",
        "chosen_model",
        "model_name",
        "selection_isolation_scheme",
        "selection_folds",
        "untouched_evaluation_folds",
        "selection_rows",
        "selection_taken_rows",
        "selection_take_rate",
        "profit_factor_delta",
        "sharpe_delta",
        "drawdown_delta",
        "expectancy_delta",
        "worst_filtered_drawdown",
        "promising",
        "selection_score",
        "promotion_authority",
        "testnet_candidate_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    ]
    required = {
        "model_name",
        "fold",
        "selection_phase",
        "selection_isolation_scheme",
        "shadow_take",
        RETURN_COLUMN,
    }
    if predictions.empty or not required.issubset(predictions.columns):
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for model_name, model_rows in predictions.groupby("model_name", sort=True, dropna=False):
        selection = model_rows.loc[
            model_rows["selection_phase"].astype(str).eq(MODEL_SELECTION_PHASE)
        ].copy()
        evaluation = model_rows.loc[
            model_rows["selection_phase"].astype(str).eq(UNTOUCHED_EVALUATION_PHASE)
        ].copy()
        if selection.empty or evaluation.empty:
            continue
        fold_rows: list[dict[str, Any]] = []
        for _, fold in selection.groupby("fold", sort=True, dropna=False):
            baseline_returns = pd.to_numeric(fold[RETURN_COLUMN], errors="coerce").dropna()
            taken_mask = fold["shadow_take"].map(
                lambda value: str(value).strip().lower() in {"1", "true", "yes", "pass", "ready"}
            )
            filtered_returns = pd.to_numeric(
                fold.loc[taken_mask, RETURN_COLUMN], errors="coerce"
            ).dropna()
            baseline = _trade_metric_summary(baseline_returns)
            filtered = _trade_metric_summary(filtered_returns)
            training_floor = fold.get(
                "training_take_rate_floor_pass",
                pd.Series(True, index=fold.index),
            ).map(lambda value: str(value).strip().lower() in {"1", "true", "yes", "pass", "ready"})
            fold_rows.append(
                {
                    "profit_factor_delta": filtered["profit_factor"] - baseline["profit_factor"],
                    "sharpe_delta": filtered["sharpe"] - baseline["sharpe"],
                    "drawdown_delta": filtered["drawdown"] - baseline["drawdown"],
                    "expectancy_delta": filtered["expectancy"] - baseline["expectancy"],
                    "filtered_drawdown": filtered["drawdown"],
                    "filtered_take_rate": (float(taken_mask.mean()) if len(fold) else 0.0),
                    "training_take_rate_floor_pass": bool(
                        not training_floor.empty and training_floor.all()
                    ),
                }
            )
        fold_frame = pd.DataFrame(fold_rows)
        if fold_frame.empty:
            continue
        aggregate = {
            "profit_factor_delta": float(fold_frame["profit_factor_delta"].median()),
            "sharpe_delta": float(fold_frame["sharpe_delta"].median()),
            "drawdown_delta": float(fold_frame["drawdown_delta"].median()),
            "expectancy_delta": float(fold_frame["expectancy_delta"].median()),
            "worst_filtered_drawdown": float(fold_frame["filtered_drawdown"].max()),
            "promising": _promising_fold_frame(fold_frame),
        }
        taken = selection["shadow_take"].map(
            lambda value: str(value).strip().lower() in {"1", "true", "yes", "pass", "ready"}
        )
        rows.append(
            {
                "model_name": str(model_name),
                "selection_isolation_scheme": (MODEL_SELECTION_ISOLATION_SCHEME),
                "selection_folds": int(selection["fold"].nunique()),
                "untouched_evaluation_folds": int(evaluation["fold"].nunique()),
                "selection_rows": int(len(selection)),
                "selection_taken_rows": int(taken.sum()),
                "selection_take_rate": float(taken.mean()),
                **aggregate,
                "selection_score": _model_selection_score(aggregate),
                "promotion_authority": False,
                "testnet_candidate_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    result = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "promising",
                "selection_score",
                "expectancy_delta",
                "profit_factor_delta",
                "sharpe_delta",
                "model_name",
            ],
            ascending=[False, False, False, False, False, True],
        )
        .reset_index(drop=True)
    )
    result.insert(0, "chosen_model", False)
    result.insert(0, "selection_rank", result.index + 1)
    result.loc[result.index[0], "chosen_model"] = True
    return result[columns]
