from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import json
import pickle
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from quant_platform.backtest import CostModel, max_drawdown
from quant_platform.experiments import PairDataset
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
RETURN_COLUMN = "realized_return"
TIMESTAMP_COLUMN = "entry_timestamp"
CATEGORICAL_FEATURES = ["pair", "strategy_name", "family", "regime", "backtest_mode", "signal_side"]
NON_FEATURE_COLUMNS = {
    "trade_id",
    "strategy_id",
    TARGET_COLUMN,
    RETURN_COLUMN,
    "gross_trade_return",
    "trade_cost_drag",
    "entry_timestamp",
    "exit_timestamp",
    "entry_bar_index",
    "exit_bar_index",
    "trade_bars",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    available: bool
    unavailable_reason: str = ""


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
    return dataset_frame.loc[:, [column for column in ML_DATASET_COLUMNS if column in dataset_frame.columns]]


def available_model_specs() -> list[ModelSpec]:
    specs = [ModelSpec(name="logistic_regression", family="linear", available=True)]
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
) -> dict[str, Path]:
    if dataset.empty:
        raise ValueError("dataset is empty")
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"dataset missing target column: {TARGET_COLUMN}")
    if dataset[TARGET_COLUMN].nunique(dropna=True) < 2:
        raise ValueError("dataset requires both profitable and unprofitable labels")

    ordered = dataset.copy()
    ordered[TIMESTAMP_COLUMN] = pd.to_datetime(ordered[TIMESTAMP_COLUMN], utc=True, errors="coerce")
    ordered = ordered.dropna(subset=[TIMESTAMP_COLUMN]).sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    feature_columns = [column for column in ordered.columns if column not in NON_FEATURE_COLUMNS]
    feature_columns = [column for column in feature_columns if column != TARGET_COLUMN]
    numeric_features = [column for column in feature_columns if column not in CATEGORICAL_FEATURES]
    categorical_features = [column for column in CATEGORICAL_FEATURES if column in feature_columns]

    splitter = TimeSeriesSplit(n_splits=max(2, n_splits))
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    chosen_artifact: dict[str, Any] | None = None
    best_gain = float("-inf")
    model_artifacts: list[dict[str, Any]] = []

    for spec in available_model_specs():
        fold_results: list[dict[str, Any]] = []
        for fold_number, (train_idx, test_idx) in enumerate(splitter.split(ordered), start=1):
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
            threshold = _select_probability_threshold(train_probability, train[RETURN_COLUMN].astype(float).to_numpy())
            baseline_metrics = _trade_metric_summary(test[RETURN_COLUMN].astype(float))
            filtered_metrics = _trade_metric_summary(test.loc[test_probability >= threshold, RETURN_COLUMN].astype(float))
            precision = _safe_precision(test[TARGET_COLUMN].astype(int), test_probability >= threshold)
            recall = _safe_recall(test[TARGET_COLUMN].astype(int), test_probability >= threshold)
            auc = _safe_auc(test[TARGET_COLUMN].astype(int), test_probability)

            row = {
                "model_name": spec.name,
                "model_family": spec.family,
                "fold": fold_number,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "threshold": float(threshold),
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
                "filtered_take_rate": float((test_probability >= threshold).mean()) if len(test_probability) else 0.0,
                "profit_factor_delta": filtered_metrics["profit_factor"] - baseline_metrics["profit_factor"],
                "sharpe_delta": filtered_metrics["sharpe"] - baseline_metrics["sharpe"],
                "drawdown_delta": filtered_metrics["drawdown"] - baseline_metrics["drawdown"],
                "expectancy_delta": filtered_metrics["expectancy"] - baseline_metrics["expectancy"],
                "trade_count_delta": filtered_metrics["trade_count"] - baseline_metrics["trade_count"],
            }
            fold_rows.append(row)
            fold_results.append(row)

            prediction_frame = test[
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
            prediction_frame["model_name"] = spec.name
            prediction_frame["fold"] = fold_number
            prediction_frame["probability_profitable"] = test_probability
            prediction_frame["shadow_take"] = test_probability >= threshold
            prediction_frame["threshold"] = threshold
            prediction_rows.extend(prediction_frame.to_dict(orient="records"))

        if not fold_results:
            continue

        fold_frame = pd.DataFrame(fold_results)
        aggregate = {
            "model_name": spec.name,
            "model_family": spec.family,
            "folds": int(len(fold_frame)),
            "median_baseline_profit_factor": float(fold_frame["baseline_profit_factor"].median()),
            "median_baseline_sharpe": float(fold_frame["baseline_sharpe"].median()),
            "worst_baseline_drawdown": float(fold_frame["baseline_drawdown"].max()),
            "median_baseline_expectancy": float(fold_frame["baseline_expectancy"].median()),
            "total_baseline_trades": int(fold_frame["baseline_trade_count"].sum()),
            "median_filtered_profit_factor": float(fold_frame["filtered_profit_factor"].median()),
            "median_filtered_sharpe": float(fold_frame["filtered_sharpe"].median()),
            "worst_filtered_drawdown": float(fold_frame["filtered_drawdown"].max()),
            "median_filtered_expectancy": float(fold_frame["filtered_expectancy"].median()),
            "total_filtered_trades": int(fold_frame["filtered_trade_count"].sum()),
            "median_precision": float(fold_frame["precision"].median()),
            "median_recall": float(fold_frame["recall"].median()),
            "median_auc": float(fold_frame["auc"].median()),
            "median_take_rate": float(fold_frame["filtered_take_rate"].median()),
            "profit_factor_delta": float(fold_frame["profit_factor_delta"].median()),
            "sharpe_delta": float(fold_frame["sharpe_delta"].median()),
            "drawdown_delta": float(fold_frame["drawdown_delta"].median()),
            "expectancy_delta": float(fold_frame["expectancy_delta"].median()),
            "trade_count_delta": int(round(float(fold_frame["trade_count_delta"].median()))),
            "promising": _promising_fold_frame(fold_frame),
            "dependency_note": spec.unavailable_reason,
        }
        aggregate_rows.append(aggregate)

        full_estimator = _build_model_pipeline(spec.name, numeric_features, categorical_features)
        full_estimator.fit(ordered[feature_columns], ordered[TARGET_COLUMN].astype(int))
        full_probability = full_estimator.predict_proba(ordered[feature_columns])[:, 1]
        full_threshold = _select_probability_threshold(full_probability, ordered[RETURN_COLUMN].astype(float).to_numpy())
        artifact = {
            "model_name": spec.name,
            "model_family": spec.family,
            "threshold": float(full_threshold),
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "trained_until": ordered[TIMESTAMP_COLUMN].max().isoformat(),
            "estimator": full_estimator,
            "dependency_note": spec.unavailable_reason,
        }
        model_artifacts.append(artifact)
        gain = aggregate["expectancy_delta"] + aggregate["profit_factor_delta"] * 0.01 + aggregate["sharpe_delta"] * 0.01
        if gain > best_gain:
            best_gain = gain
            chosen_artifact = artifact

    if not fold_rows or chosen_artifact is None:
        raise ValueError("no valid walk-forward folds were produced")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "ml_trade_filter_dataset.csv"
    fold_path = output / "ml_trade_filter_walkforward_folds.csv"
    prediction_path = output / "ml_trade_filter_walkforward_predictions.csv"
    summary_path = output / "ml_trade_filter_comparison_summary.csv"
    best_model_path = output / "ml_trade_filter_best_model.pkl"
    manifest_path = output / "ml_trade_filter_manifest.json"

    ordered.to_csv(dataset_path, index=False)
    pd.DataFrame(fold_rows).sort_values(["model_name", "fold"]).to_csv(fold_path, index=False)
    pd.DataFrame(prediction_rows).sort_values(["model_name", "entry_timestamp"]).to_csv(prediction_path, index=False)
    summary_frame = pd.DataFrame(aggregate_rows).sort_values(
        ["promising", "expectancy_delta", "profit_factor_delta", "sharpe_delta"],
        ascending=[False, False, False, False],
    )
    summary_frame.to_csv(summary_path, index=False)

    with best_model_path.open("wb") as handle:
        pickle.dump(chosen_artifact, handle)
    manifest = {
        "chosen_model": chosen_artifact["model_name"],
        "trained_until": chosen_artifact["trained_until"],
        "threshold": chosen_artifact["threshold"],
        "artifacts_written": [str(path) for path in (dataset_path, fold_path, prediction_path, summary_path, best_model_path)],
        "models_evaluated": [artifact["model_name"] for artifact in model_artifacts],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "dataset": dataset_path,
        "folds": fold_path,
        "predictions": prediction_path,
        "summary": summary_path,
        "best_model": best_model_path,
        "manifest": manifest_path,
    }


def shadow_trade_filter_predictions(
    dataset: pd.DataFrame,
    *,
    model_artifact_path: str | Path,
    output_path: str | Path,
) -> Path:
    artifact_path = Path(model_artifact_path)
    with artifact_path.open("rb") as handle:
        artifact = pickle.load(handle)
    estimator = artifact["estimator"]
    feature_columns = list(artifact["feature_columns"])
    threshold = float(artifact["threshold"])

    available = dataset.copy()
    available[TIMESTAMP_COLUMN] = pd.to_datetime(available[TIMESTAMP_COLUMN], utc=True, errors="coerce")
    available = available.dropna(subset=[TIMESTAMP_COLUMN]).sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
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
    shadow.to_csv(output, index=False)
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
    entry_mask = detailed["signal_target"].ne(0.0) & detailed["signal_target"].shift(1).fillna(0.0).eq(0.0)
    exit_mask = detailed["signal_target"].eq(0.0) & detailed["signal_target"].shift(1).fillna(0.0).ne(0.0)
    entry_positions = np.flatnonzero(entry_mask.to_numpy())
    exit_positions = np.flatnonzero(exit_mask.to_numpy())
    rows: list[dict[str, Any]] = []

    for ordinal, entry_pos in enumerate(entry_positions, start=1):
        future_exits = exit_positions[exit_positions >= entry_pos]
        exit_pos = int(future_exits[0]) if len(future_exits) else int(len(detailed) - 1)
        segment = detailed.iloc[entry_pos : exit_pos + 1]
        entry_row = detailed.iloc[entry_pos]
        trade_id = f"{pair}|{strategy.id}|{pd.Timestamp(entry_row['timestamp']).isoformat()}|{ordinal}"
        realized_return = float(segment["net_return"].sum())
        gross_return = float(segment["gross_return"].sum())
        row = {
            "trade_id": trade_id,
            "pair": pair,
            "strategy_id": strategy.id,
            "strategy_name": strategy.name,
            "family": strategy.family,
            "backtest_mode": backtest_mode,
            "entry_timestamp": pd.Timestamp(entry_row["timestamp"]).isoformat(),
            "exit_timestamp": pd.Timestamp(detailed.iloc[exit_pos]["timestamp"]).isoformat(),
            "entry_bar_index": int(entry_pos),
            "exit_bar_index": int(exit_pos),
            "trade_bars": int(exit_pos - entry_pos + 1),
            "signal_side": "long_spread" if float(entry_row["signal_target"]) > 0 else "short_spread",
            "label_profitable": int(realized_return > 0.0),
            "realized_return": realized_return,
            "gross_trade_return": gross_return,
            "trade_cost_drag": float(segment["cost_drag"].sum()),
        }
        row.update(_entry_feature_row(detailed, entry_pos))
        rows.append(row)
    return rows


def _entry_feature_row(frame: pd.DataFrame, entry_pos: int) -> dict[str, Any]:
    row = frame.iloc[entry_pos]
    return {
        "entry_zscore": float(row.get("zscore", 0.0) or 0.0),
        "entry_abs_zscore": abs(float(row.get("zscore", 0.0) or 0.0)),
        "zscore_change_1": float(row.get("zscore_change_1", 0.0) or 0.0),
        "zscore_change_3": float(row.get("zscore_change_3", 0.0) or 0.0),
        "spread_level": float(row.get("spread", 0.0) or 0.0),
        "spread_change_1": float(row.get("spread_change_1", 0.0) or 0.0),
        "spread_change_3": float(row.get("spread_change_3", 0.0) or 0.0),
        "spread_vol_12": float(row.get("spread_vol_12", 0.0) or 0.0),
        "spread_vol_48": float(row.get("spread_vol_48", 0.0) or 0.0),
        "hedge_ratio": float(row.get("hedge_ratio", 1.0) or 1.0),
        "hedge_ratio_stability": float(row.get("hedge_ratio_stability", 0.0) or 0.0),
        "beta": float(row.get("beta", 1.0) or 1.0),
        "realized_volatility_percentile": float(row.get("realized_volatility_percentile", 0.0) or 0.0),
        "cvar": float(row.get("cvar", 0.0) or 0.0),
        "var": float(row.get("var", 0.0) or 0.0),
        "tail_dependence": float(row.get("tail_dependence", 0.0) or 0.0),
        "crisis_probability": float(row.get("crisis_probability", 0.0) or 0.0),
        "liquidity_score": float(row.get("liquidity_score", 0.0) or 0.0),
        "bid_ask_spread_bps": float(row.get("bid_ask_spread_bps", 0.0) or 0.0),
        "slippage_bps": float(row.get("slippage_bps", 0.0) or 0.0),
        "volume_x_usd": float(row.get("volume_x_usd", 0.0) or 0.0),
        "volume_y_usd": float(row.get("volume_y_usd", 0.0) or 0.0),
        "funding_x_bps": float(row.get("funding_x_bps", 0.0) or 0.0),
        "funding_y_bps": float(row.get("funding_y_bps", 0.0) or 0.0),
        "funding_diff_bps": float(row.get("funding_diff_bps", 0.0) or 0.0),
        "funding_abs_total_bps": float(row.get("funding_abs_total_bps", 0.0) or 0.0),
        "funding_bps_per_day": float(row.get("funding_bps_per_day", 0.0) or 0.0),
        "regime": str(row.get("regime", "UNKNOWN") or "UNKNOWN"),
        "regime_strategy_match": float(row.get("regime_strategy_match", 0.0) or 0.0),
        "cointegration_pvalue": float(row.get("cointegration_pvalue", 1.0) or 1.0),
        "ecm_strength": float(row.get("ecm_strength", 0.0) or 0.0),
        "ecm_x": float(row.get("ecm_x", 0.0) or 0.0),
        "ecm_y": float(row.get("ecm_y", 0.0) or 0.0),
        "half_life": float(row.get("half_life", 0.0) or 0.0),
        "hurst": float(row.get("hurst", 0.0) or 0.0),
        "conditional_probability_distortion": float(row.get("conditional_probability_distortion", 0.0) or 0.0),
        "copula_calibration_score": float(row.get("copula_calibration_score", 0.0) or 0.0),
        "u1_given_u2": float(row.get("u1_given_u2", 0.0) or 0.0),
        "u2_given_u1": float(row.get("u2_given_u1", 0.0) or 0.0),
        "composite_score": float(row.get("composite_score", 0.0) or 0.0),
        "ml_confidence": float(row.get("ml_confidence", 0.0) or 0.0),
        "profile_match": float(row.get("profile_match", 0.0) or 0.0),
        "ou_optimal": float(row.get("ou_optimal", 0.0) or 0.0),
    }


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


def _detailed_backtest_frame(frame: pd.DataFrame, signal: pd.Series, cost_model: CostModel) -> tuple[pd.DataFrame, str]:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data = data.dropna(subset=["timestamp"]).reset_index(drop=True)
    data["signal_target"] = signal.reindex(frame.index).fillna(0.0).astype(float).iloc[: len(data)].reset_index(drop=True)
    data["spread"] = pd.to_numeric(data.get("spread", pd.Series(0.0, index=data.index)), errors="coerce").fillna(0.0)
    data["zscore"] = pd.to_numeric(data.get("zscore", pd.Series(0.0, index=data.index)), errors="coerce").fillna(0.0)
    data["zscore_change_1"] = data["zscore"].diff().fillna(0.0)
    data["zscore_change_3"] = data["zscore"].diff(3).fillna(0.0)
    data["spread_change_1"] = data["spread"].diff().fillna(0.0)
    data["spread_change_3"] = data["spread"].diff(3).fillna(0.0)
    data["spread_vol_12"] = data["spread"].diff().rolling(12, min_periods=2).std().fillna(0.0)
    data["spread_vol_48"] = data["spread"].diff().rolling(48, min_periods=2).std().fillna(0.0)
    data["funding_diff_bps"] = (
        pd.to_numeric(data.get("funding_y_bps", pd.Series(0.0, index=data.index)), errors="coerce").fillna(0.0)
        - pd.to_numeric(data.get("funding_x_bps", pd.Series(0.0, index=data.index)), errors="coerce").fillna(0.0)
    )
    data["funding_abs_total_bps"] = (
        pd.to_numeric(data.get("funding_x_bps", pd.Series(0.0, index=data.index)), errors="coerce").fillna(0.0).abs()
        + pd.to_numeric(data.get("funding_y_bps", pd.Series(0.0, index=data.index)), errors="coerce").fillna(0.0).abs()
    )

    if {"price_x", "price_y"}.issubset(data.columns):
        backtest_mode = "two_leg"
        price_x = pd.to_numeric(data["price_x"], errors="coerce").ffill().bfill()
        price_y = pd.to_numeric(data["price_y"], errors="coerce").ffill().bfill()
        returns_x = price_x.pct_change().fillna(0.0)
        returns_y = price_y.pct_change().fillna(0.0)
        hedge_ratio = pd.to_numeric(data.get("hedge_ratio", 1.0), errors="coerce").fillna(1.0)
        beta = pd.to_numeric(data.get("beta", 1.0), errors="coerce").replace(0, 1.0).fillna(1.0).abs()
        signal_position = data["signal_target"].shift(1).fillna(0.0)
        gross_scale = 1.0 + hedge_ratio.abs() * beta
        target_weight_y = data["signal_target"] / gross_scale
        target_weight_x = -data["signal_target"] * hedge_ratio * beta / gross_scale
        weight_y = signal_position / gross_scale
        weight_x = -signal_position * hedge_ratio * beta / gross_scale
        turnover_x = target_weight_x.diff().abs().fillna(target_weight_x.abs())
        turnover_y = target_weight_y.diff().abs().fillna(target_weight_y.abs())
        turnover = turnover_x + turnover_y
        gross_return = weight_x * returns_x + weight_y * returns_y
        fee_cost = turnover * cost_model.taker_fee_bps / 10_000.0
        slippage_cost = turnover * cost_model.slippage_bps / 10_000.0
        execution_risk_cost = turnover * cost_model.execution_risk_bps / 10_000.0
        partial_fill_cost = (
            turnover
            * cost_model.partial_fill_probability
            * (1.0 - cost_model.partial_fill_fraction)
            * cost_model.partial_fill_penalty_bps
            / 10_000.0
        )
        funding_x = pd.to_numeric(data.get("funding_x_bps", cost_model.funding_bps_per_day), errors="coerce").fillna(cost_model.funding_bps_per_day)
        funding_y = pd.to_numeric(data.get("funding_y_bps", cost_model.funding_bps_per_day), errors="coerce").fillna(cost_model.funding_bps_per_day)
        funding_cost = (
            weight_x.abs() * funding_x.abs() / 10_000.0 / cost_model.bars_per_day
            + weight_y.abs() * funding_y.abs() / 10_000.0 / cost_model.bars_per_day
        )
        net_return = gross_return - fee_cost - slippage_cost - execution_risk_cost - partial_fill_cost - funding_cost
        cost_drag = fee_cost + slippage_cost + execution_risk_cost + partial_fill_cost + funding_cost
    else:
        backtest_mode = "spread"
        spread_return = data["spread"].diff().fillna(0.0)
        signal_position = data["signal_target"].shift(1).fillna(0.0)
        gross_return = signal_position * spread_return
        turnover = data["signal_target"].diff().abs().fillna(data["signal_target"].abs())
        trading_cost = turnover * cost_model.round_trip_cost() / 2.0
        funding_cost = signal_position.abs() * cost_model.funding_per_bar()
        net_return = gross_return - trading_cost - funding_cost
        cost_drag = trading_cost + funding_cost

    data["gross_return"] = gross_return.astype(float)
    data["net_return"] = net_return.astype(float)
    data["cost_drag"] = pd.to_numeric(cost_drag, errors="coerce").fillna(0.0).astype(float)
    return data, backtest_mode


def _build_model_pipeline(name: str, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
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
    std = float(series.std(ddof=0))
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


def _select_probability_threshold(probability: np.ndarray, realized_returns: np.ndarray) -> float:
    best_threshold = 0.50
    best_score = float("-inf")
    for threshold in np.arange(0.50, 0.81, 0.05):
        accepted = realized_returns[probability >= threshold]
        metrics = _trade_metric_summary(pd.Series(accepted, dtype="float64"))
        score = metrics["expectancy"] + metrics["total_return"] * 0.1 + metrics["profit_factor"] * 0.001
        if metrics["trade_count"] == 0:
            score -= 1.0
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


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
    return (
        median_pf_gain > 0.0
        and median_sharpe_gain > 0.0
        and median_dd_gain <= 0.0
        and median_expectancy_gain > 0.0
        and median_take_rate >= 0.10
    )
