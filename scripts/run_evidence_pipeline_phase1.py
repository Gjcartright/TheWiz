from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_LONG = ROOT / "data" / "raw" / "dydx_long_history"
RAW_SINGLE = ROOT / "data" / "raw" / "dydx_candles"
OUT = ROOT / "reports" / "evidence_pipeline"

PAIRS = {
    "ETH-SOL": ("eth_sol", "ETH-USD", "SOL-USD"),
    "BTC-DOGE": ("btc_doge", "BTC-USD", "DOGE-USD"),
    "BTC-SOL": ("btc_sol", "BTC-USD", "SOL-USD"),
    "ETH-LINK": ("eth_link", "ETH-USD", "LINK-USD"),
    "DOGE-XRP": ("doge_xrp", "DOGE-USD", "XRP-USD"),
}

TIMEFRAMES = {
    "5m": ("5MINS", 288 * 365),
    "15m": ("15MINS", 96 * 365),
    "1h": ("1HOUR", 24 * 365),
    "4h": ("4HOURS", 6 * 365),
    "1d": ("1DAY", 365),
}

COST_BUCKETS = {
    "base": {
        "fee_bps": 5.0,
        "slippage_bps": 4.0,
        "execution_risk_bps": 2.0,
        "funding_bps_per_day": 1.0,
    },
    "stress": {
        "fee_bps": 7.5,
        "slippage_bps": 8.0,
        "execution_risk_bps": 4.0,
        "funding_bps_per_day": 3.0,
    },
}

STRATEGIES = {
    "zscore_2_0_exit_0": {
        "kind": "zscore",
        "lookback": 96,
        "entry": 2.0,
        "exit": 0.0,
        "stop": 4.0,
        "max_hold": 96,
    },
    "zscore_1_5_exit_0_25": {
        "kind": "zscore",
        "lookback": 96,
        "entry": 1.5,
        "exit": 0.25,
        "stop": 3.5,
        "max_hold": 96,
    },
    "zscore_2_5_exit_0": {
        "kind": "zscore",
        "lookback": 96,
        "entry": 2.5,
        "exit": 0.0,
        "stop": 4.5,
        "max_hold": 120,
    },
    "ou_mean_reversion": {
        "kind": "ou",
        "lookback": 192,
        "entry": 1.75,
        "exit": 0.25,
        "stop": 3.5,
        "max_hold": 144,
    },
    "copula_dislocation": {
        "kind": "copula",
        "lookback": 192,
        "entry": 0.62,
        "exit": 0.18,
        "stop": 0.92,
        "max_hold": 144,
    },
    "beta_dislocation": {
        "kind": "beta",
        "lookback": 144,
        "entry": 1.9,
        "exit": 0.30,
        "stop": 3.6,
        "max_hold": 120,
    },
}


def _read_json_candles(path: Path) -> list[dict[str, object]]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    return data.get("candles", []) if isinstance(data, dict) else []


def _candle_frame(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        for candle in _read_json_candles(path):
            rows.append(
                {
                    "timestamp": candle.get("startedAt"),
                    "close": candle.get("close"),
                    "volume": candle.get("usdVolume"),
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    return (
        frame.dropna(subset=["timestamp", "close"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .set_index("timestamp")
    )


def _long_paths(pair_dir: str, ticker: str, timeframe_api: str) -> list[Path]:
    return sorted((RAW_LONG / pair_dir).glob(f"window_*/{ticker}_{timeframe_api}_candles.json"))


def _single_paths(ticker: str, timeframe_api: str) -> list[Path]:
    path = RAW_SINGLE / f"{ticker}_{timeframe_api}_candles.json"
    return [path] if path.exists() else []


def load_pair(pair_dir: str, asset_x: str, asset_y: str, timeframe_api: str) -> tuple[pd.DataFrame, str, int, int]:
    x_paths = _long_paths(pair_dir, asset_x, timeframe_api)
    y_paths = _long_paths(pair_dir, asset_y, timeframe_api)
    source_type = "long_history"
    if not x_paths or not y_paths:
        x_paths = _single_paths(asset_x, timeframe_api)
        y_paths = _single_paths(asset_y, timeframe_api)
        source_type = "single_window_fallback"

    x_frame = _candle_frame(x_paths)
    y_frame = _candle_frame(y_paths)
    if x_frame.empty or y_frame.empty:
        return pd.DataFrame(), "missing", len(x_frame), len(y_frame)

    pair = (
        x_frame[["close", "volume"]]
        .rename(columns={"close": "price_x", "volume": "volume_x"})
        .join(
            y_frame[["close", "volume"]].rename(columns={"close": "price_y", "volume": "volume_y"}),
            how="inner",
        )
        .dropna(subset=["price_x", "price_y"])
    )
    pair = pair[(pair["price_x"] > 0) & (pair["price_y"] > 0)]
    if pair.empty:
        return pair, source_type, len(x_frame), len(y_frame)

    log_x = np.log(pair["price_x"])
    log_y = np.log(pair["price_y"])
    pair["spread"] = log_x - log_y
    ret_x = log_x.diff()
    ret_y = log_y.diff()
    beta = ret_x.rolling(96, min_periods=20).cov(ret_y) / ret_y.rolling(96, min_periods=20).var().replace(0, np.nan)
    pair["beta"] = beta.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(-5.0, 5.0).abs()
    pair["volume"] = pair[["volume_x", "volume_y"]].min(axis=1)
    return pair, source_type, len(x_frame), len(y_frame)


def feature_frame(frame: pd.DataFrame, lookback: int) -> pd.DataFrame:
    min_periods = max(20, lookback // 4)
    spread = frame["spread"]
    zscore = (spread - spread.rolling(lookback, min_periods=min_periods).mean()) / spread.rolling(
        lookback, min_periods=min_periods
    ).std().replace(0, np.nan)
    log_x = np.log(frame["price_x"])
    log_y = np.log(frame["price_y"])
    ret_x = log_x.diff()
    ret_y = log_y.diff()
    corr = ret_x.rolling(lookback, min_periods=min_periods).corr(ret_y)
    spread_vol = spread.diff().rolling(lookback, min_periods=min_periods).std()
    vol_rank = spread_vol.rolling(lookback * 3, min_periods=max(30, lookback)).rank(pct=True)
    beta_dislocation = (frame["beta"] - frame["beta"].rolling(lookback, min_periods=min_periods).mean()) / frame[
        "beta"
    ].rolling(lookback, min_periods=min_periods).std().replace(0, np.nan)
    copula_distortion = np.tanh(zscore.fillna(0.0) / 2.25)
    ou_score = (1.0 - zscore.abs().rolling(lookback, min_periods=min_periods).mean().rank(pct=True)).clip(0.0, 1.0)
    return pd.DataFrame(
        {
            "zscore": zscore,
            "corr": corr,
            "vol_rank": vol_rank,
            "beta_dislocation": beta_dislocation,
            "copula_distortion": copula_distortion,
            "ou_score": ou_score,
        },
        index=frame.index,
    )


def strategy_signal(features: pd.DataFrame, config: dict[str, float]) -> pd.Series:
    kind = str(config["kind"])
    entry = float(config["entry"])
    exit_level = float(config["exit"])
    signal = pd.Series(0.0, index=features.index)
    zscore = features["zscore"]

    if kind == "copula":
        source = features["copula_distortion"]
        signal[source > entry] = -1.0
        signal[source < -entry] = 1.0
        signal[source.abs() <= exit_level] = 0.0
    elif kind == "ou":
        source = zscore
        ok = features["ou_score"].fillna(0.0) >= 0.35
        signal[(source > entry) & ok] = -1.0
        signal[(source < -entry) & ok] = 1.0
        signal[source.abs() <= exit_level] = 0.0
    elif kind == "beta":
        source = zscore + features["beta_dislocation"].fillna(0.0) * 0.35
        signal[source > entry] = -1.0
        signal[source < -entry] = 1.0
        signal[source.abs() <= exit_level] = 0.0
    else:
        source = zscore
        signal[source > entry] = -1.0
        signal[source < -entry] = 1.0
        signal[source.abs() <= exit_level] = 0.0

    return signal.replace(0.0, np.nan).ffill().fillna(0.0)


def apply_stops(signal: pd.Series, features: pd.DataFrame, config: dict[str, float]) -> pd.Series:
    stop = float(config["stop"])
    max_hold = int(config["max_hold"])
    z_abs = features["zscore"].abs().fillna(0.0).to_numpy()
    raw = signal.fillna(0.0).to_numpy()
    out = np.zeros(len(raw), dtype=float)
    position = 0.0
    age = 0
    for idx, target in enumerate(raw):
        if position == 0.0 and target != 0.0:
            position = target
            age = 0
        elif position != 0.0:
            age += 1
            if target == 0.0 or np.sign(target) != np.sign(position) or z_abs[idx] >= stop or age >= max_hold:
                position = 0.0
                age = 0
            else:
                position = target
        out[idx] = position
    return pd.Series(out, index=signal.index)


def backtest(frame: pd.DataFrame, signal: pd.Series, periods_per_year: int, costs: dict[str, float]) -> dict[str, float]:
    data = frame.join(signal.rename("signal"), how="inner")
    if data.empty:
        return empty_metrics()

    price_x = data["price_x"].to_numpy(dtype=float)
    price_y = data["price_y"].to_numpy(dtype=float)
    beta = data["beta"].replace(0.0, 1.0).abs().fillna(1.0).to_numpy(dtype=float)
    target = data["signal"].fillna(0.0).to_numpy(dtype=float)
    ret_x = np.r_[0.0, np.diff(price_x) / price_x[:-1]]
    ret_y = np.r_[0.0, np.diff(price_y) / price_y[:-1]]
    unit_cost = (costs["fee_bps"] + costs["slippage_bps"] + costs["execution_risk_bps"]) / 10_000.0
    funding = (costs["funding_bps_per_day"] / 10_000.0) / max(periods_per_year / 365.0, 1.0)

    position = 0.0
    last_target = 0.0
    current_trade = 0.0
    current_holding = 0
    trades: list[float] = []
    holdings: list[int] = []
    returns: list[float] = []
    equity = 1.0
    equity_curve: list[float] = []

    for idx in range(len(data)):
        scale = 1.0 + abs(beta[idx])
        weight_y = position / scale
        weight_x = -position * abs(beta[idx]) / scale
        gross = weight_x * ret_x[idx] + weight_y * ret_y[idx]
        turnover = abs(target[idx] - last_target)
        net = gross - turnover * unit_cost - abs(position) * funding
        returns.append(net)
        equity *= 1.0 + net
        equity_curve.append(equity)

        if position != 0.0:
            current_trade += net
            current_holding += 1

        if position != 0.0 and target[idx] == 0.0:
            trades.append(current_trade)
            holdings.append(current_holding)
            current_trade = 0.0
            current_holding = 0

        if position == 0.0 and target[idx] != 0.0:
            current_trade = 0.0
            current_holding = 0

        position = target[idx]
        last_target = target[idx]

    if position != 0.0:
        trades.append(current_trade)
        holdings.append(current_holding)

    return metric_summary(returns, equity_curve, trades, holdings, periods_per_year)


def empty_metrics() -> dict[str, float]:
    return {
        "trades": 0,
        "profit_factor": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "avg_holding_bars": 0.0,
        "expectancy": 0.0,
        "total_return": 0.0,
    }


def metric_summary(
    returns: list[float], equity_curve: list[float], trades: list[float], holdings: list[int], periods_per_year: int
) -> dict[str, float]:
    if not returns:
        return empty_metrics()
    returns_s = pd.Series(returns, dtype="float64")
    equity = pd.Series(equity_curve, dtype="float64")
    trade_s = pd.Series(trades, dtype="float64")
    wins = trade_s[trade_s > 0.0]
    losses = trade_s[trade_s < 0.0]
    loss_sum = abs(float(losses.sum()))
    profit_factor = float(wins.sum() / loss_sum) if loss_sum > 0 else (float("inf") if float(wins.sum()) > 0 else 0.0)
    std = float(returns_s.std(ddof=0))
    sharpe = float(math.sqrt(periods_per_year) * returns_s.mean() / std) if std > 0 else 0.0
    peak = equity.cummax()
    max_dd = float(((peak - equity) / peak.replace(0, np.nan)).fillna(0.0).max())
    return {
        "trades": int(len(trade_s)),
        "profit_factor": profit_factor if math.isfinite(profit_factor) else 999.0,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": float((trade_s > 0.0).mean()) if len(trade_s) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "avg_holding_bars": float(pd.Series(holdings).mean()) if holdings else 0.0,
        "expectancy": float(trade_s.mean()) if len(trade_s) else 0.0,
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
    }


def acceptance_reason(metrics: dict[str, float], stress: bool = False) -> str:
    failures: list[str] = []
    if metrics["trades"] < 100:
        failures.append("trades<100")
    pf_target = 1.3 if stress else 1.8
    if metrics["profit_factor"] < pf_target:
        failures.append(f"profit_factor<{pf_target}")
    if metrics["sharpe"] <= 0:
        failures.append("sharpe<=0")
    if metrics["max_drawdown"] > 0.15:
        failures.append("max_drawdown>0.15")
    if metrics["expectancy"] <= 0:
        failures.append("expectancy<=0")
    return "ACCEPT" if not failures else "reject: " + "; ".join(failures)


def format_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    data = frame[columns].head(limit) if limit else frame[columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in data.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    coverage_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []

    for pair, (pair_dir, asset_x, asset_y) in PAIRS.items():
        for timeframe, (timeframe_api, periods_per_year) in TIMEFRAMES.items():
            frame, source_type, x_rows, y_rows = load_pair(pair_dir, asset_x, asset_y, timeframe_api)
            coverage_rows.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "target_required": True,
                    "available": len(frame) >= 120,
                    "source_type": source_type,
                    "paired_rows": len(frame),
                    "asset_x_rows": x_rows,
                    "asset_y_rows": y_rows,
                    "blocker": "" if len(frame) >= 120 else "missing_or_insufficient_local_history",
                }
            )
            if len(frame) < 120:
                continue

            for strategy, config in STRATEGIES.items():
                features = feature_frame(frame, int(config["lookback"]))
                base_signal = strategy_signal(features, config)
                signal = apply_stops(base_signal, features, config)
                for cost_bucket, costs in COST_BUCKETS.items():
                    metrics = backtest(frame, signal, periods_per_year, costs)
                    result_rows.append(
                        {
                            "pair": pair,
                            "timeframe": timeframe,
                            "strategy": strategy,
                            "cost_bucket": cost_bucket,
                            **metrics,
                            "acceptance_reason": acceptance_reason(metrics, stress=cost_bucket != "base"),
                            "source_type": source_type,
                            "observations": len(frame),
                        }
                    )

    coverage = pd.DataFrame(coverage_rows)
    results = pd.DataFrame(result_rows)
    results["accepted"] = results["acceptance_reason"].eq("ACCEPT")
    results["rank_score"] = (
        results["accepted"].astype(int) * 100_000
        + results["profit_factor"].fillna(0.0).clip(upper=20.0) * 1000.0
        + results["sharpe"].fillna(0.0).clip(lower=-10.0, upper=20.0) * 100.0
        - results["max_drawdown"].fillna(1.0).clip(upper=5.0) * 100.0
        + np.minimum(results["trades"].fillna(0.0), 300.0)
    )
    results = results.sort_values(["accepted", "rank_score", "trades"], ascending=[False, False, False]).reset_index(
        drop=True
    )
    results.insert(0, "rank", np.arange(1, len(results) + 1))

    pair_summary = (
        results.groupby("pair", dropna=False)
        .agg(
            runs=("rank", "count"),
            accepted_runs=("accepted", "sum"),
            best_rank=("rank", "min"),
            best_profit_factor=("profit_factor", "max"),
            best_sharpe=("sharpe", "max"),
            min_max_drawdown=("max_drawdown", "min"),
            max_trades=("trades", "max"),
        )
        .reset_index()
        .sort_values(["accepted_runs", "best_profit_factor"], ascending=[False, False])
    )

    coverage.to_csv(OUT / "phase1_coverage_inventory.csv", index=False)
    results.to_csv(OUT / "phase1_baseline_inventory_ranked.csv", index=False)
    pair_summary.to_csv(OUT / "phase1_pair_summary.csv", index=False)

    base_100 = results[(results["cost_bucket"] == "base") & (results["trades"] >= 100)].copy()
    rank_columns = [
        "rank",
        "pair",
        "timeframe",
        "strategy",
        "cost_bucket",
        "trades",
        "profit_factor",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "avg_win",
        "avg_loss",
        "avg_holding_bars",
        "acceptance_reason",
    ]
    coverage_columns = [
        "pair",
        "timeframe",
        "available",
        "source_type",
        "paired_rows",
        "asset_x_rows",
        "asset_y_rows",
        "blocker",
    ]
    report = [
        "# Evidence Pipeline Phase 1 Baseline Inventory",
        "",
        "Clean baseline inventory for ETH-SOL, BTC-DOGE, BTC-SOL, ETH-LINK, and DOGE-XRP across available local timeframes.",
        "",
        "## Scope",
        "",
        f"- Baseline rows: {len(results):,}",
        f"- Accepted rows: {int(results['accepted'].sum()):,}",
        "- Strategies: z-score 2/0, z-score 1.5/0.25, z-score 2.5/0, OU mean reversion, copula dislocation, beta dislocation",
        "- Cost buckets: base, stress",
        "- Acceptance: base PF >= 1.8, stress PF >= 1.3, trades >= 100, max DD <= 15%, positive Sharpe, positive expectancy",
        "",
        "## Coverage Inventory",
        "",
        format_table(coverage, coverage_columns),
        "",
        "## Top Ranked Baselines",
        "",
        format_table(results, rank_columns, 30),
        "",
        "## Best 100+ Trade Base-Cost Baselines",
        "",
        format_table(base_100, rank_columns, 30) if len(base_100) else "No base-cost baseline reached 100 trades.",
        "",
        "## Pair Summary",
        "",
        format_table(pair_summary, list(pair_summary.columns)),
        "",
        "## Blockers",
        "",
    ]
    missing = coverage[~coverage["available"]]
    if missing.empty:
        report.append("- No local timeframe coverage blockers for the required matrix.")
    else:
        report.append(format_table(missing, coverage_columns))
    report.extend(
        [
            "",
            "## Next Phase",
            "",
            "- Phase 2 should fetch or build missing long-history windows where this report shows fallback or missing coverage.",
            "- Phase 3 should use these baselines as the control group for soft trade quality scoring.",
        ]
    )
    (OUT / "phase1_baseline_inventory_report.md").write_text("\n".join(report) + "\n")

    print(
        json.dumps(
            {
                "baseline_rows": int(len(results)),
                "accepted_rows": int(results["accepted"].sum()),
                "coverage_rows": int(len(coverage)),
                "report": str(OUT / "phase1_baseline_inventory_report.md"),
                "ranked_csv": str(OUT / "phase1_baseline_inventory_ranked.csv"),
                "coverage_csv": str(OUT / "phase1_coverage_inventory.csv"),
                "best": results.iloc[0][rank_columns[1:]].to_dict() if len(results) else None,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
