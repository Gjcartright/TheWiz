from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

from quant_platform.active_pipeline import ROOT, CommandResult
from quant_platform.dydx_candles import load_loose_candle_payload

PANEL_PAPER_ID = "paper_bd860d3475116e96"
FRACTIONAL_PAPER_ID = "paper_83bedd65a3817a87"
CHARACTERISTIC_PAPER_ID = "paper_a2e2cdcb8d0f2fac"
REBALANCING_PAPER_ID = "paper_9d33d77526db17f3"
FEATURE_COLUMNS = [
    "residual",
    "residual_z",
    "residual_lag_1",
    "asset_return",
    "market_return",
    "asset_volatility",
    "beta",
    "volume_z",
]


def run_paper_reproduction_suite(*, root: Path = ROOT) -> CommandResult:
    """Run bounded, causal paper experiments without granting trading authority."""

    report_dir = root / "reports" / "research" / "papers" / "reproductions"
    report_dir.mkdir(parents=True, exist_ok=True)
    close, volume, audit = load_hyperliquid_daily_panel(root=root)
    if close.shape[0] < 300 or close.shape[1] < 8:
        raise ValueError("paper reproduction requires at least 300 common daily rows and 8 assets")

    dataset = build_panel_residual_dataset(close=close, volume=volume)
    predictions, panel_metrics = run_panel_residual_walkforward(dataset)
    pairs = cost_ready_pair_frames(root=root, close=close)
    stability = build_one_sided_stability_diagnostics(pairs)
    bands = run_dynamic_no_trade_band_tests(pairs)
    characteristics = build_crypto_characteristic_candidates(
        root=root,
        close=close,
        volume=volume,
    )
    gates = _reproduction_gates(
        close=close,
        panel_metrics=panel_metrics,
        pairs=pairs,
        stability=stability,
        bands=bands,
        characteristics=characteristics,
    )
    red_team = build_reproduction_red_team(
        audit=audit,
        panel_metrics=panel_metrics,
        stability=stability,
        bands=bands,
        characteristics=characteristics,
        gates=gates,
    )

    paths = {
        "data_audit": report_dir / "paper_reproduction_data_audit.csv",
        "panel_predictions": report_dir / "panel_residual_walkforward_predictions.csv",
        "panel_metrics": report_dir / "panel_residual_walkforward_metrics.csv",
        "stability": report_dir / "one_sided_stability_diagnostics.csv",
        "no_trade_bands": report_dir / "dynamic_no_trade_band_results.csv",
        "characteristics": report_dir / "crypto_characteristic_candidates.csv",
        "gates": report_dir / "paper_reproduction_gate.csv",
        "red_team": report_dir / "paper_reproduction_red_team.csv",
        "red_team_md": report_dir / "paper_reproduction_red_team.md",
        "summary": report_dir / "paper_reproduction_summary.md",
    }
    frames = {
        "data_audit": audit,
        "panel_predictions": predictions,
        "panel_metrics": panel_metrics,
        "stability": stability,
        "no_trade_bands": bands,
        "characteristics": characteristics,
        "gates": gates,
        "red_team": red_team,
    }
    for name, frame in frames.items():
        atomic_write_csv(frame, paths[name], index=False)
    atomic_write_text(paths["red_team_md"], _red_team_markdown(red_team), encoding="utf-8")
    atomic_write_text(paths["summary"], _summary_markdown(
            close=close,
            audit=audit,
            panel_metrics=panel_metrics,
            stability=stability,
            bands=bands,
            characteristics=characteristics,
            gates=gates,
        ), encoding="utf-8")
    return CommandResult(
        paths=paths,
        summary={
            "assets": int(close.shape[1]),
            "common_daily_rows": int(close.shape[0]),
            "panel_prediction_rows": len(predictions),
            "walkforward_folds": int(panel_metrics["fold"].nunique()),
            "cost_ready_pairs": len(pairs),
            "stability_rows": len(stability),
            "characteristic_candidates": len(characteristics),
            "red_team_failures": int(red_team["status"].eq("FAIL").sum()),
            "red_team_blockers": int(red_team["status"].eq("BLOCKED").sum()),
            "signal_eligible": 0,
            "execution_eligible": 0,
        },
    )


def load_hyperliquid_daily_panel(
    *,
    root: Path,
    minimum_rows: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candle_dir = root / "data" / "raw" / "hyperliquid_candles"
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    audit_rows: list[dict[str, Any]] = []
    for path in sorted(candle_dir.glob("*_1d_candles.json")):
        asset = path.name.removesuffix("_1d_candles.json").upper()
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = load_loose_candle_payload(path)
        frame = pd.DataFrame(rows)
        frame["timestamp"] = pd.to_datetime(frame.get("startedAt"), utc=True, errors="coerce")
        frame["close_value"] = pd.to_numeric(frame.get("close"), errors="coerce")
        frame["volume_value"] = pd.to_numeric(frame.get("usdVolume"), errors="coerce")
        frame = (
            frame.dropna(subset=["timestamp", "close_value"])
            .loc[lambda value: value["close_value"] > 0]
            .sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
        )
        eligible = len(frame) >= minimum_rows
        if eligible:
            closes[asset] = frame.set_index("timestamp")["close_value"].rename(asset)
            volumes[asset] = frame.set_index("timestamp")["volume_value"].rename(asset)
        fetched_at = str(payload.get("fetched_at", "")) if isinstance(payload, dict) else ""
        latest = frame["timestamp"].max() if not frame.empty else pd.NaT
        fetched = pd.to_datetime(fetched_at, utc=True, errors="coerce")
        audit_rows.append(
            {
                "asset": asset,
                "interval": "1d",
                "rows": len(frame),
                "earliest_candle_at": _iso(frame["timestamp"].min()),
                "latest_candle_at": _iso(latest),
                "collected_at": _iso(fetched),
                "collection_after_history": bool(
                    pd.notna(fetched) and pd.notna(latest) and fetched > latest
                ),
                "minimum_rows": minimum_rows,
                "panel_eligible": eligible,
                "point_in_time_vintage": False,
                "blocker": "historical_payload_not_as_traded_vintage",
                "evidence_path": str(path.relative_to(root)),
                "promotion_authority": False,
            }
        )
    if not closes:
        raise ValueError("no eligible Hyperliquid daily candles found")
    close = pd.concat(closes.values(), axis=1, join="inner").dropna().sort_index()
    volume = pd.concat(volumes.values(), axis=1).reindex(close.index).reindex(columns=close.columns)
    return close, volume, pd.DataFrame(audit_rows)


def build_panel_residual_dataset(
    *,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    factor_window: int = 90,
    residual_window: int = 30,
) -> pd.DataFrame:
    returns = np.log(close).diff()
    market = returns.mean(axis=1)
    market_mean = market.rolling(factor_window, min_periods=factor_window).mean().shift(1)
    market_variance = market.rolling(factor_window, min_periods=factor_window).var().shift(1)
    records: list[pd.DataFrame] = []
    for asset in close.columns:
        asset_return = returns[asset]
        beta = (
            asset_return.rolling(factor_window, min_periods=factor_window)
            .cov(market)
            .shift(1)
            .div(market_variance.replace(0.0, np.nan))
        )
        alpha = (
            asset_return.rolling(factor_window, min_periods=factor_window).mean().shift(1)
            - beta * market_mean
        )
        residual = asset_return - alpha - beta * market
        residual_mean = (
            residual.rolling(residual_window, min_periods=residual_window).mean().shift(1)
        )
        residual_std = residual.rolling(residual_window, min_periods=residual_window).std().shift(1)
        log_volume = np.log1p(pd.to_numeric(volume[asset], errors="coerce"))
        volume_mean = (
            log_volume.rolling(residual_window, min_periods=residual_window).mean().shift(1)
        )
        volume_std = log_volume.rolling(residual_window, min_periods=residual_window).std().shift(1)
        feature_timestamp = pd.Series(returns.index, index=returns.index)
        frame = pd.DataFrame(
            {
                "asset": asset,
                "feature_timestamp": feature_timestamp,
                "label_timestamp": feature_timestamp.shift(-1),
                "residual": residual,
                "residual_z": (residual - residual_mean) / residual_std.replace(0.0, np.nan),
                "residual_lag_1": residual.shift(1),
                "asset_return": asset_return,
                "market_return": market,
                "asset_volatility": asset_return.rolling(
                    residual_window, min_periods=residual_window
                )
                .std()
                .shift(1),
                "beta": beta,
                "volume_z": (log_volume - volume_mean) / volume_std.replace(0.0, np.nan),
                "next_residual": residual.shift(-1),
            },
            index=returns.index,
        )
        frame["label"] = (frame["next_residual"] > 0.0).astype(int)
        records.append(frame)
    dataset = pd.concat(records, ignore_index=True)
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["label_timestamp", "next_residual"])
    dataset["uses_future_data"] = False
    dataset["feature_known_at_entry"] = True
    dataset["point_in_time_vintage"] = False
    dataset["promotion_authority"] = False
    dataset["blocker"] = "historical_payload_not_as_traded_vintage"
    if not bool((dataset["feature_timestamp"] < dataset["label_timestamp"]).all()):
        raise ValueError("panel residual dataset violates feature/label chronology")
    return dataset.sort_values(["feature_timestamp", "asset"]).reset_index(drop=True)


def run_panel_residual_walkforward(
    dataset: pd.DataFrame,
    *,
    minimum_train_dates: int = 180,
    test_dates: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets = sorted(dataset["asset"].unique())
    holdout_assets = set(assets[::5])
    dates = pd.Index(sorted(dataset["feature_timestamp"].unique()))
    predictions: list[pd.DataFrame] = []
    fold = 0
    train_end = minimum_train_dates
    while train_end + test_dates <= len(dates):
        fold += 1
        train_dates = dates[:train_end]
        evaluation_dates = dates[train_end : train_end + test_dates]
        train = dataset[
            dataset["feature_timestamp"].isin(train_dates) & ~dataset["asset"].isin(holdout_assets)
        ]
        test = dataset[dataset["feature_timestamp"].isin(evaluation_dates)].copy()
        if train["label"].nunique() < 2 or test.empty:
            train_end += test_dates
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
        )
        model.fit(train[FEATURE_COLUMNS], train["label"])
        model_probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        model_rows = [
            ("pooled_logistic", model_probabilities),
            ("mean_reversion_sign", (test["residual"] < 0.0).astype(float).to_numpy()),
            ("persistence_sign", (test["residual"] > 0.0).astype(float).to_numpy()),
        ]
        for model_name, probabilities in model_rows:
            result = test[
                ["asset", "feature_timestamp", "label_timestamp", "label", "residual_z"]
            ].copy()
            result["fold"] = fold
            result["model"] = model_name
            result["probability_positive"] = probabilities
            result["prediction"] = (result["probability_positive"] >= 0.5).astype(int)
            result["population"] = np.where(
                result["asset"].isin(holdout_assets),
                "whole_asset_holdout",
                "seen_asset",
            )
            result["train_start_at"] = _iso(train_dates.min())
            result["train_end_at"] = _iso(train_dates.max())
            result["test_start_at"] = _iso(evaluation_dates.min())
            result["test_end_at"] = _iso(evaluation_dates.max())
            result["uses_future_data"] = False
            result["promotion_authority"] = False
            result["blocker"] = "research_baseline_not_strategy_acceptance"
            predictions.append(result)
        train_end += test_dates
    if not predictions:
        raise ValueError("insufficient dates for panel residual walk-forward")
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metric_rows: list[dict[str, Any]] = []
    for (fold_id, model_name, population), group in prediction_frame.groupby(
        ["fold", "model", "population"], sort=True
    ):
        labels = group["label"].astype(int)
        probabilities = group["probability_positive"].astype(float)
        predicted = group["prediction"].astype(int)
        metric_rows.append(
            {
                "paper_id": PANEL_PAPER_ID,
                "fold": fold_id,
                "model": model_name,
                "population": population,
                "observations": len(group),
                "assets": group["asset"].nunique(),
                "accuracy": accuracy_score(labels, predicted),
                "balanced_accuracy": balanced_accuracy_score(labels, predicted),
                "brier_score": brier_score_loss(labels, probabilities),
                "roc_auc": (
                    roc_auc_score(labels, probabilities) if labels.nunique() == 2 else np.nan
                ),
                "test_start_at": _iso(group["feature_timestamp"].min()),
                "test_end_at": _iso(group["feature_timestamp"].max()),
                "exact_paper_reproduction": False,
                "method_status": "bounded_low_capacity_baseline",
                "uses_future_data": False,
                "promotion_authority": False,
                "blocker": "dataset_architecture_and_execution_parity_not_reproduced",
            }
        )
    return prediction_frame, pd.DataFrame(metric_rows)


def cost_ready_pair_frames(*, root: Path, close: pd.DataFrame) -> dict[str, dict[str, Any]]:
    path = root / "data" / "processed" / "hyperliquid_pair_cost_models.csv"
    if not path.exists():
        return {}
    costs = pd.read_csv(path).fillna("")
    costs = costs[costs["strict_observed_cost_ready"].astype(str).str.lower().eq("true")]
    result: dict[str, dict[str, Any]] = {}
    for row in costs.itertuples(index=False):
        asset_x = str(row.asset_x).upper()
        asset_y = str(row.asset_y).upper()
        if asset_x not in close.columns or asset_y not in close.columns:
            continue
        frame = build_causal_pair_frame(close[[asset_x, asset_y]], asset_x, asset_y)
        if len(frame) < 220:
            continue
        result[str(row.pair)] = {
            "frame": frame,
            "asset_x": asset_x,
            "asset_y": asset_y,
            "round_trip_cost_bps": float(row.estimated_pair_round_trip_cost_bps),
            "cost_evidence_path": str(row.evidence_path),
            "cost_model_id": str(row.cost_model_id),
        }
    return result


def build_causal_pair_frame(
    close: pd.DataFrame,
    asset_x: str,
    asset_y: str,
    *,
    hedge_window: int = 90,
    zscore_window: int = 60,
) -> pd.DataFrame:
    prices = close[[asset_x, asset_y]].dropna()
    logs = np.log(prices)
    x = logs[asset_x]
    y = logs[asset_y]
    beta = (
        x.rolling(hedge_window, min_periods=hedge_window)
        .cov(y)
        .shift(1)
        .div(y.rolling(hedge_window, min_periods=hedge_window).var().shift(1))
    )
    alpha = x.rolling(hedge_window, min_periods=hedge_window).mean().shift(1) - beta * y.rolling(
        hedge_window, min_periods=hedge_window
    ).mean().shift(1)
    spread = x - alpha - beta * y
    spread_mean = spread.rolling(zscore_window, min_periods=zscore_window).mean().shift(1)
    spread_std = spread.rolling(zscore_window, min_periods=zscore_window).std().shift(1)
    returns = logs.diff()
    pair_return = (returns[asset_x] - beta * returns[asset_y]) / (1.0 + beta.abs())
    frame = pd.DataFrame(
        {
            "beta": beta,
            "spread": spread,
            "zscore": (spread - spread_mean) / spread_std.replace(0.0, np.nan),
            "pair_return": pair_return,
        }
    )
    return frame.dropna().sort_index()


def build_one_sided_stability_diagnostics(
    pairs: dict[str, dict[str, Any]],
    *,
    diagnostic_window: int = 180,
    update_every: int = 7,
    cusum_drift: float = 0.25,
    cusum_threshold: float = 5.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair, payload in pairs.items():
        frame = payload["frame"]
        positive = 0.0
        negative = 0.0
        latest_d = np.nan
        latest_adf = np.nan
        for index in range(diagnostic_window, len(frame)):
            window = frame["spread"].iloc[index - diagnostic_window : index]
            current = float(frame["spread"].iloc[index])
            scale = float(window.std(ddof=1))
            zscore = (current - float(window.mean())) / scale if scale > 0 else 0.0
            positive = max(0.0, positive + zscore - cusum_drift)
            negative = min(0.0, negative + zscore + cusum_drift)
            alarm = positive > cusum_threshold or negative < -cusum_threshold
            if index % update_every == 0:
                latest_d = estimate_gph_fractional_d(window)
                latest_adf = _adf_pvalue(window)
            rows.append(
                {
                    "paper_id": FRACTIONAL_PAPER_ID,
                    "pair": pair,
                    "feature_timestamp": _iso(frame.index[index]),
                    "estimation_start_at": _iso(window.index.min()),
                    "estimation_end_at": _iso(window.index.max()),
                    "hedge_ratio": float(frame["beta"].iloc[index]),
                    "spread_zscore": zscore,
                    "gph_fractional_d_proxy": latest_d,
                    "adf_pvalue": latest_adf,
                    "cusum_positive": positive,
                    "cusum_negative": negative,
                    "sequential_break_alarm": alarm,
                    "exact_fcvar": False,
                    "exact_bai_perron": False,
                    "method_status": "one_sided_gph_and_cusum_proxy",
                    "uses_future_data": False,
                    "detection_delay_modeled": True,
                    "promotion_authority": False,
                    "blocker": "proxy_not_exact_fcvar_and_short_crypto_history",
                    "evidence_path": payload["cost_evidence_path"],
                }
            )
            if alarm:
                positive = 0.0
                negative = 0.0
    return pd.DataFrame(rows)


def estimate_gph_fractional_d(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 40:
        return float("nan")
    values = values - values.mean()
    frequencies = 2.0 * np.pi * np.fft.rfftfreq(len(values))[1:]
    periodogram = (np.abs(np.fft.rfft(values)[1:]) ** 2) / (2.0 * np.pi * len(values))
    bandwidth = min(len(frequencies), max(8, int(math.sqrt(len(values)))))
    x = np.log(4.0 * np.sin(frequencies[:bandwidth] / 2.0) ** 2)
    y = np.log(np.maximum(periodogram[:bandwidth], np.finfo(float).tiny))
    slope = np.polyfit(x, y, 1)[0]
    return float(-slope)


def run_dynamic_no_trade_band_tests(pairs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair, payload in pairs.items():
        frame = payload["frame"].copy()
        desired = (-frame["zscore"] / 3.0).clip(-1.0, 1.0)
        pair_volatility = frame["pair_return"].rolling(30, min_periods=20).std().shift(1)
        one_way_cost = payload["round_trip_cost_bps"] / 20_000.0
        dynamic_threshold = (one_way_cost / pair_volatility.replace(0.0, np.nan)).clip(0.05, 0.50)
        thresholds = {
            "continuous_rebalance": pd.Series(0.0, index=frame.index),
            "static_no_trade_band": pd.Series(0.10, index=frame.index),
            "dynamic_cost_vol_band": dynamic_threshold.fillna(0.50),
        }
        for policy, threshold in thresholds.items():
            result = _simulate_rebalancing(
                pair_return=frame["pair_return"],
                desired_position=desired,
                threshold=threshold,
                one_way_cost=one_way_cost,
            )
            rows.append(
                {
                    "paper_id": REBALANCING_PAPER_ID,
                    "pair": pair,
                    "policy": policy,
                    "observations": len(result),
                    "position_changes": int((result["turnover"] > 0.0).sum()),
                    "total_turnover": result["turnover"].sum(),
                    "average_band": float(threshold.reindex(result.index).mean()),
                    "gross_return": _compound(result["gross_return"]),
                    "net_return": _compound(result["net_return"]),
                    "annualized_sharpe": _annualized_sharpe(result["net_return"]),
                    "max_drawdown": _max_drawdown(result["net_return"]),
                    "daily_profit_factor": _profit_factor(result["net_return"]),
                    "total_cost": float(result["cost"].sum()),
                    "round_trip_cost_bps": payload["round_trip_cost_bps"],
                    "cost_model_id": payload["cost_model_id"],
                    "uses_future_data": False,
                    "funding_modeled": False,
                    "promotion_authority": False,
                    "blocker": "component_test_requires_funding_and_full_strategy_replay",
                    "evidence_path": payload["cost_evidence_path"],
                }
            )
    return pd.DataFrame(rows)


def _simulate_rebalancing(
    *,
    pair_return: pd.Series,
    desired_position: pd.Series,
    threshold: pd.Series,
    one_way_cost: float,
) -> pd.DataFrame:
    aligned = pd.concat(
        [
            pair_return.rename("pair_return"),
            desired_position.rename("desired"),
            threshold.rename("band"),
        ],
        axis=1,
    ).dropna()
    current = 0.0
    rows: list[dict[str, float]] = []
    for timestamp, row in aligned.iterrows():
        previous = current
        desired = float(row["desired"])
        if abs(desired - previous) >= float(row["band"]):
            current = desired
        turnover = abs(current - previous)
        gross = previous * float(row["pair_return"])
        cost = turnover * one_way_cost
        rows.append(
            {
                "timestamp": timestamp,
                "position": current,
                "turnover": turnover,
                "gross_return": gross,
                "cost": cost,
                "net_return": gross - cost,
            }
        )
    return pd.DataFrame(rows).set_index("timestamp")


def build_crypto_characteristic_candidates(
    *,
    root: Path,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    minimum_history: int = 120,
) -> pd.DataFrame:
    as_of = close.index.max()
    returns = np.log(close).diff()
    funding_by_asset = _load_raw_funding_history(root=root, as_of=as_of)
    feature_rows: list[dict[str, Any]] = []
    for asset in close.columns:
        asset_returns = returns[asset].dropna()
        if len(asset_returns) < minimum_history:
            continue
        asset_funding = funding_by_asset.get(asset, pd.DataFrame())
        if asset_funding.empty:
            continue
        feature_rows.append(
            {
                "asset": asset,
                "volatility_30d": asset_returns.tail(30).std(),
                "volatility_90d": asset_returns.tail(90).std(),
                "momentum_30d": np.log(close[asset].iloc[-1] / close[asset].iloc[-31]),
                "momentum_90d": np.log(close[asset].iloc[-1] / close[asset].iloc[-91]),
                "median_usd_volume_30d": pd.to_numeric(
                    volume[asset].tail(30), errors="coerce"
                ).median(),
                "mean_funding_7d": asset_funding.tail(24 * 7)["funding_rate"].mean(),
                "mean_funding_30d": asset_funding.tail(24 * 30)["funding_rate"].mean(),
                "funding_observations": len(asset_funding),
                "price_as_of": as_of,
                "funding_as_of": asset_funding["timestamp"].max(),
            }
        )
    features = pd.DataFrame(feature_rows).dropna()
    if len(features) < 6:
        return pd.DataFrame()
    columns = [
        "volatility_30d",
        "volatility_90d",
        "momentum_30d",
        "momentum_90d",
        "median_usd_volume_30d",
        "mean_funding_7d",
        "mean_funding_30d",
    ]
    matrix = features[columns].copy()
    matrix["median_usd_volume_30d"] = np.log1p(matrix["median_usd_volume_30d"])
    scaled = StandardScaler().fit_transform(matrix)
    cluster_count = min(5, max(2, round(math.sqrt(len(features)))))
    features["cluster"] = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=20,
    ).fit_predict(scaled)
    coordinates = dict(zip(features["asset"], scaled, strict=True))
    rows: list[dict[str, Any]] = []
    for cluster, group in features.groupby("cluster"):
        for asset_x, asset_y in combinations(sorted(group["asset"]), 2):
            correlation = returns[[asset_x, asset_y]].tail(90).corr().iloc[0, 1]
            distance = float(np.linalg.norm(coordinates[asset_x] - coordinates[asset_y]))
            similarity = 0.5 * (1.0 / (1.0 + distance)) + 0.5 * ((correlation + 1.0) / 2.0)
            rows.append(
                {
                    "paper_id": CHARACTERISTIC_PAPER_ID,
                    "pair": f"{asset_x}-USD-{asset_y}-USD",
                    "asset_x": asset_x,
                    "asset_y": asset_y,
                    "cluster": int(cluster),
                    "characteristic_distance": distance,
                    "return_correlation_90d": correlation,
                    "discovery_score": similarity,
                    "price_as_of": _iso(as_of),
                    "funding_as_of": _iso(
                        min(
                            group.loc[group["asset"].eq(asset_x), "funding_as_of"].iloc[0],
                            group.loc[group["asset"].eq(asset_y), "funding_as_of"].iloc[0],
                        )
                    ),
                    "uses_future_data": False,
                    "point_in_time_vintage": False,
                    "decision_bucket": "RESEARCH_ONLY",
                    "promotion_authority": False,
                    "blocker": "discovery_only_requires_pair_specific_walkforward_and_costs",
                    "evidence_path": ("data/raw/hyperliquid_candles;data/raw/hyperliquid_funding"),
                }
            )
    return pd.DataFrame(rows).sort_values("discovery_score", ascending=False).reset_index(drop=True)


def _load_raw_funding_history(*, root: Path, as_of: pd.Timestamp) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    funding_dir = root / "data" / "raw" / "hyperliquid_funding"
    for path in sorted(funding_dir.glob("*_funding.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("funding", []) if isinstance(payload, dict) else []
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        frame["timestamp"] = pd.to_datetime(
            pd.to_numeric(frame.get("time"), errors="coerce"),
            unit="ms",
            utc=True,
            errors="coerce",
        )
        frame["funding_rate"] = pd.to_numeric(frame.get("fundingRate"), errors="coerce")
        frame = frame.dropna(subset=["timestamp", "funding_rate"])
        frame = frame[frame["timestamp"] <= as_of].sort_values("timestamp")
        if not frame.empty:
            asset = path.name.removesuffix("_funding.json").upper()
            result[asset] = frame
    return result


def _reproduction_gates(
    *,
    close: pd.DataFrame,
    panel_metrics: pd.DataFrame,
    pairs: dict[str, dict[str, Any]],
    stability: pd.DataFrame,
    bands: pd.DataFrame,
    characteristics: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment": "panel_residual_direction",
                "status": "COMPLETE_RESEARCH_ONLY" if not panel_metrics.empty else "BLOCKED",
                "evidence_rows": len(panel_metrics),
                "exact_paper_reproduction": False,
                "signal_eligible": False,
                "execution_eligible": False,
                "blocker": "historical_vintage_model_architecture_and_trading_mapping_not_reproduced",
            },
            {
                "experiment": "one_sided_fractional_breaks",
                "status": "COMPLETE_PROXY_ONLY" if not stability.empty else "BLOCKED",
                "evidence_rows": len(stability),
                "exact_paper_reproduction": False,
                "signal_eligible": False,
                "execution_eligible": False,
                "blocker": "gph_cusum_proxy_is_not_fcvar_bai_perron",
            },
            {
                "experiment": "crypto_characteristic_discovery",
                "status": "COMPLETE_DISCOVERY_ONLY" if not characteristics.empty else "BLOCKED",
                "evidence_rows": len(characteristics),
                "exact_paper_reproduction": False,
                "signal_eligible": False,
                "execution_eligible": False,
                "blocker": "candidate_generation_requires_pair_walkforward_acceptance",
            },
            {
                "experiment": "dynamic_no_trade_band",
                "status": "COMPLETE_COMPONENT_ONLY" if not bands.empty else "BLOCKED",
                "evidence_rows": len(bands),
                "exact_paper_reproduction": False,
                "signal_eligible": False,
                "execution_eligible": False,
                "blocker": "funding_full_strategy_and_asynchronous_fill_replay_required",
            },
            {
                "experiment": "suite_authority",
                "status": "RESEARCH_ONLY",
                "evidence_rows": int(close.shape[0] * close.shape[1] + len(pairs)),
                "exact_paper_reproduction": False,
                "signal_eligible": False,
                "execution_eligible": False,
                "blocker": "no_reproduction_artifact_can_authorize_a_trade",
            },
        ]
    )


def build_reproduction_red_team(
    *,
    audit: pd.DataFrame,
    panel_metrics: pd.DataFrame,
    stability: pd.DataFrame,
    bands: pd.DataFrame,
    characteristics: pd.DataFrame,
    gates: pd.DataFrame,
) -> pd.DataFrame:
    holdout = panel_metrics[panel_metrics["population"].eq("whole_asset_holdout")]
    logistic = holdout[holdout["model"].eq("pooled_logistic")]
    baselines = holdout[~holdout["model"].eq("pooled_logistic")]
    logistic_accuracy = float(logistic["balanced_accuracy"].mean())
    best_baseline = float(baselines.groupby("model")["balanced_accuracy"].mean().max())
    dynamic = bands[bands["policy"].eq("dynamic_cost_vol_band")]
    static = bands[bands["policy"].eq("static_no_trade_band")]
    dynamic_net = dynamic.set_index("pair")["net_return"]
    static_net = static.set_index("pair")["net_return"]
    dynamic_wins = int((dynamic_net.reindex(static_net.index) > static_net).sum())
    profitable_policies = int((bands["net_return"] > 0.0).sum())
    alarm_rate = (
        float(stability["sequential_break_alarm"].mean()) if not stability.empty else float("nan")
    )
    no_authority = bool(
        not gates["signal_eligible"].any() and not gates["execution_eligible"].any()
    )
    rows = [
        {
            "check": "whole_asset_holdout_incremental_edge",
            "status": "PASS" if logistic_accuracy >= best_baseline + 0.02 else "FAIL",
            "observed": (
                f"logistic_balanced_accuracy={logistic_accuracy:.4f};"
                f"best_simple_baseline={best_baseline:.4f}"
            ),
            "required": "at_least_0.02_balanced_accuracy_improvement",
            "risk": "complexity_without_incremental_predictive_value",
            "corrective_action": "retain simple baselines and collect more causal history",
        },
        {
            "check": "minimum_chronological_folds",
            "status": "PASS" if panel_metrics["fold"].nunique() >= 3 else "FAIL",
            "observed": f"folds={panel_metrics['fold'].nunique()}",
            "required": "at_least_3_non_overlapping_test_folds",
            "risk": "single_regime_or_fold_luck",
            "corrective_action": "extend daily history before model escalation",
        },
        {
            "check": "as_traded_data_vintage",
            "status": "PASS" if audit["point_in_time_vintage"].all() else "FAIL",
            "observed": f"point_in_time_assets={int(audit['point_in_time_vintage'].sum())}/{len(audit)}",
            "required": "all_historical_features_have_as_traded_vintages",
            "risk": "silent_vendor_revision_or_collection_hindsight",
            "corrective_action": "start immutable daily snapshots and forbid retroactive promotion",
        },
        {
            "check": "dynamic_band_dominates_static_band",
            "status": "PASS" if len(static_net) > 0 and dynamic_wins == len(static_net) else "FAIL",
            "observed": f"dynamic_net_wins={dynamic_wins}/{len(static_net)}",
            "required": "dynamic_band_net_return_exceeds_static_for_every_pair",
            "risk": "adaptive_complexity_adds_turnover_without_edge",
            "corrective_action": "treat dynamic threshold as rejected until recalibrated out of sample",
        },
        {
            "check": "positive_after_cost_component_results",
            "status": "PASS" if profitable_policies == len(bands) and len(bands) > 0 else "FAIL",
            "observed": f"profitable_pair_policy_cells={profitable_policies}/{len(bands)}",
            "required": "positive_net_return_for_every_component_cell",
            "risk": "cost_reduction_mistaken_for_trading_edge",
            "corrective_action": "do not attach the component to live strategy logic",
        },
        {
            "check": "sequential_break_alarm_calibration",
            "status": "PASS" if pd.notna(alarm_rate) and alarm_rate <= 0.05 else "BLOCKED",
            "observed": f"alarm_rate={alarm_rate:.4f}",
            "required": "prospectively_calibrated_false_alarm_rate_at_or_below_0.05",
            "risk": "over_sensitive_regime_filter_suppresses_or_churns_trades",
            "corrective_action": "calibrate threshold on separate history and report detection delay",
        },
        {
            "check": "characteristic_candidates_have_acceptance_evidence",
            "status": "BLOCKED",
            "observed": f"discovery_candidates={len(characteristics)};accepted=0",
            "required": "pair_specific_walkforward_cost_funding_and_regime_acceptance",
            "risk": "similarity_score_is_mistaken_for_tradeability",
            "corrective_action": "queue candidates for local replay without promoting them",
        },
        {
            "check": "authority_firewall",
            "status": "PASS" if no_authority else "FAIL",
            "observed": f"signal_or_execution_authority_present={not no_authority}",
            "required": "zero_authority_until_full_acceptance",
            "risk": "research_artifact_reaches_order_path",
            "corrective_action": "preserve hard research-only blockers",
        },
    ]
    return pd.DataFrame(rows)


def _red_team_markdown(red_team: pd.DataFrame) -> str:
    lines = [
        "# Paper Reproduction Red Team",
        "",
        "The red team attempts to falsify each apparent improvement before it can influence trading.",
        "",
    ]
    for row in red_team.itertuples(index=False):
        lines.extend(
            [
                f"## {row.check}",
                "",
                f"- status: `{row.status}`",
                f"- observed: {row.observed}",
                f"- required: {row.required}",
                f"- risk: {row.risk}",
                f"- corrective action: {row.corrective_action}",
                "",
            ]
        )
    lines.append("No failed or blocked experiment has signal or execution authority.")
    return "\n".join(lines) + "\n"


def _summary_markdown(
    *,
    close: pd.DataFrame,
    audit: pd.DataFrame,
    panel_metrics: pd.DataFrame,
    stability: pd.DataFrame,
    bands: pd.DataFrame,
    characteristics: pd.DataFrame,
    gates: pd.DataFrame,
) -> str:
    logistic = panel_metrics[panel_metrics["model"].eq("pooled_logistic")]
    best = logistic.sort_values("balanced_accuracy", ascending=False).head(1)
    best_text = "not available"
    if not best.empty:
        row = best.iloc[0]
        best_text = (
            f"fold {int(row['fold'])}, {row['population']}, "
            f"balanced accuracy {float(row['balanced_accuracy']):.3f}, "
            f"AUC {float(row['roc_auc']):.3f}"
        )
    return "\n".join(
        [
            "# Paper Reproduction Suite",
            "",
            "This suite tests bounded causal implications from the paper library. It does not reproduce every paper and has no signal or execution authority.",
            "",
            f"- panel: {close.shape[1]} assets x {close.shape[0]} common daily observations",
            f"- panel collection-vintage blockers: {int((~audit['point_in_time_vintage']).sum())}",
            f"- strongest pooled-logistic diagnostic: {best_text}",
            f"- one-sided stability rows: {len(stability)}",
            f"- observed-cost no-trade-band rows: {len(bands)}",
            f"- characteristic candidates: {len(characteristics)}",
            "",
            "## Authority",
            "",
            "Every experiment is research-only. Historical payloads were collected after much of the represented market history, exact FCVAR/Bai-Perron and paper model architectures were not reproduced, and component tests omit full trade lifecycle parity.",
            "",
            "## Gates",
            "",
            *[
                f"- `{row.experiment}`: `{row.status}`; blocker: {row.blocker}"
                for row in gates.itertuples(index=False)
            ],
            "",
        ]
    )


def _adf_pvalue(series: pd.Series) -> float:
    try:
        return float(adfuller(pd.to_numeric(series, errors="coerce").dropna(), autolag="AIC")[1])
    except (ValueError, np.linalg.LinAlgError):
        return float("nan")


def _iso(value: Any) -> str:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    return timestamp.isoformat() if pd.notna(timestamp) else ""


def _compound(returns: pd.Series) -> float:
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0)


def _annualized_sharpe(returns: pd.Series) -> float:
    values = returns.dropna()
    standard_deviation = float(values.std(ddof=1))
    if len(values) < 2 or standard_deviation <= 0:
        return float("nan")
    return float(math.sqrt(252.0) * values.mean() / standard_deviation)


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(abs(drawdown.min())) if len(drawdown) else 0.0


def _profit_factor(returns: pd.Series) -> float:
    gains = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    return gains / losses if losses > 0 else float("inf")
