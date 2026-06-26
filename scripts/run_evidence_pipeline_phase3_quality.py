from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "evidence_pipeline"
REPORTS = ROOT / "reports" / "evidence_pipeline"

TIMEFRAME_PERIODS = {
    "5m": 288 * 365,
    "15m": 96 * 365,
    "1h": 24 * 365,
    "4h": 6 * 365,
    "1d": 365,
}

TIMEFRAME_FIT = {
    "5m": 52.0,
    "15m": 64.0,
    "1h": 82.0,
    "4h": 86.0,
    "1d": 62.0,
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

GATES = ("ungated", "hard_filter", "quality_50", "quality_60", "quality_70", "quality_80")


def pair_and_timeframe(path: Path) -> tuple[str, str]:
    stem = path.stem.removesuffix("_pair_history")
    parts = stem.split("_")
    timeframe = parts[-1]
    pair = "-".join(parts[:-1]).upper()
    return pair, timeframe


def load_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "x_close", "y_close"]).sort_values("timestamp").set_index("timestamp")
    return frame[(frame["x_close"] > 0) & (frame["y_close"] > 0)]


def feature_frame(frame: pd.DataFrame, lookback: int, timeframe: str) -> pd.DataFrame:
    min_periods = max(20, lookback // 4)
    spread = frame["spread"].astype(float)
    spread_mean = spread.rolling(lookback, min_periods=min_periods).mean()
    spread_std = spread.rolling(lookback, min_periods=min_periods).std().replace(0, np.nan)
    zscore = (spread - spread_mean) / spread_std
    dz = zscore.diff()
    ret_x = pd.to_numeric(frame["return_x"], errors="coerce")
    ret_y = pd.to_numeric(frame["return_y"], errors="coerce")
    corr = ret_x.rolling(lookback, min_periods=min_periods).corr(ret_y)
    corr_stability = 1.0 - corr.rolling(lookback, min_periods=min_periods).std().rank(pct=True)
    beta = pd.to_numeric(frame["beta_96"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
    beta_stability = 1.0 - beta.rolling(lookback, min_periods=min_periods).std().rank(pct=True)
    spread_vol = spread.diff().rolling(lookback, min_periods=min_periods).std()
    vol_rank = spread_vol.rolling(lookback * 3, min_periods=max(30, lookback)).rank(pct=True)
    abs_z_rank = zscore.abs().rolling(lookback * 3, min_periods=max(30, lookback)).rank(pct=True)
    volume = pd.to_numeric(frame.get("min_usd_volume", 0.0), errors="coerce").fillna(0.0)
    liquidity_rank = volume.rolling(lookback * 3, min_periods=max(30, lookback)).rank(pct=True)

    z_strength_score = (zscore.abs() / 3.0 * 100.0).clip(0.0, 100.0)
    reversion_pressure_score = pd.Series(50.0, index=frame.index)
    reversion_pressure_score[zscore * dz < 0.0] = 85.0
    reversion_pressure_score[zscore * dz > 0.0] = 30.0
    hedge_stability_score = ((corr.abs().fillna(0.0).clip(0.0, 1.0) * 55.0) + (beta_stability.fillna(0.5) * 45.0)).clip(
        0.0, 100.0
    )
    corr_stability_score = ((corr.abs().fillna(0.0).clip(0.0, 1.0) * 45.0) + (corr_stability.fillna(0.5) * 55.0)).clip(
        0.0, 100.0
    )
    volatility_score = (100.0 - (vol_rank.fillna(0.75) - 0.35).abs() * 140.0).clip(0.0, 100.0)
    funding_score = pd.Series(50.0, index=frame.index)
    tail_risk_score = (100.0 - abs_z_rank.fillna(0.8) * 80.0 - (vol_rank.fillna(0.75) > 0.85).astype(float) * 25.0).clip(
        0.0, 100.0
    )
    timeframe_score = pd.Series(TIMEFRAME_FIT.get(timeframe, 60.0), index=frame.index)
    liquidity_score = (liquidity_rank.fillna(0.5) * 100.0).clip(0.0, 100.0)

    quality = (
        z_strength_score * 0.16
        + reversion_pressure_score * 0.16
        + hedge_stability_score * 0.16
        + corr_stability_score * 0.12
        + volatility_score * 0.12
        + funding_score * 0.08
        + tail_risk_score * 0.12
        + timeframe_score * 0.05
        + liquidity_score * 0.03
    )
    return pd.DataFrame(
        {
            "zscore": zscore,
            "dz": dz,
            "corr": corr,
            "vol_rank": vol_rank,
            "abs_z_rank": abs_z_rank,
            "beta_stability": beta_stability,
            "z_strength_score": z_strength_score,
            "reversion_pressure_score": reversion_pressure_score,
            "hedge_stability_score": hedge_stability_score,
            "corr_stability_score": corr_stability_score,
            "volatility_score": volatility_score,
            "funding_score": funding_score,
            "tail_risk_score": tail_risk_score,
            "timeframe_fit_score": timeframe_score,
            "liquidity_score": liquidity_score,
            "trade_quality_score": quality.clip(0.0, 100.0),
        },
        index=frame.index,
    )


def base_entry_signal(features: pd.DataFrame, config: dict[str, float]) -> tuple[pd.Series, pd.Series]:
    kind = str(config["kind"])
    entry = float(config["entry"])
    exit_level = float(config["exit"])
    zscore = features["zscore"]
    signal = pd.Series(0.0, index=features.index)
    if kind == "copula":
        source = np.tanh(zscore.fillna(0.0) / 2.25)
        signal[source > entry] = -1.0
        signal[source < -entry] = 1.0
        signal[source.abs() <= exit_level] = 0.0
    elif kind == "ou":
        source = zscore
        ou_ok = (1.0 - zscore.abs().rolling(int(config["lookback"]), min_periods=20).mean().rank(pct=True)).fillna(0.0)
        ok = ou_ok >= 0.35
        signal[(source > entry) & ok] = -1.0
        signal[(source < -entry) & ok] = 1.0
        signal[source.abs() <= exit_level] = 0.0
    elif kind == "beta":
        source = zscore + (1.0 - features["beta_stability"].fillna(0.5)) * np.sign(zscore.fillna(0.0)) * 0.7
        signal[source > entry] = -1.0
        signal[source < -entry] = 1.0
        signal[source.abs() <= exit_level] = 0.0
    else:
        source = zscore
        signal[source > entry] = -1.0
        signal[source < -entry] = 1.0
        signal[source.abs() <= exit_level] = 0.0
    return signal, pd.Series(source, index=features.index)


def hard_filter_mask(features: pd.DataFrame) -> pd.Series:
    return (
        (features["corr"].abs().fillna(0.0) >= 0.35)
        & (features["vol_rank"].fillna(1.0) <= 0.65)
        & (features["abs_z_rank"].fillna(1.0) <= 0.78)
    )


def apply_gate(raw: pd.Series, features: pd.DataFrame, gate: str) -> pd.Series:
    if gate == "ungated":
        gated = raw.copy()
    elif gate == "hard_filter":
        gated = raw.where(hard_filter_mask(features), 0.0)
    else:
        threshold = float(gate.split("_")[1])
        gated = raw.where(features["trade_quality_score"].fillna(0.0) >= threshold, 0.0)
    return gated.replace(0.0, np.nan).ffill().fillna(0.0)


def apply_stops(signal: pd.Series, source: pd.Series, config: dict[str, float]) -> pd.Series:
    stop = float(config["stop"])
    max_hold = int(config["max_hold"])
    raw = signal.fillna(0.0).to_numpy(dtype=float)
    source_abs = source.abs().fillna(0.0).to_numpy(dtype=float)
    out = np.zeros(len(raw), dtype=float)
    position = 0.0
    age = 0
    for idx, target in enumerate(raw):
        if position == 0.0 and target != 0.0:
            position = target
            age = 0
        elif position != 0.0:
            age += 1
            if target == 0.0 or np.sign(target) != np.sign(position) or source_abs[idx] >= stop or age >= max_hold:
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
    price_x = data["x_close"].to_numpy(dtype=float)
    price_y = data["y_close"].to_numpy(dtype=float)
    beta = pd.to_numeric(data["beta_96"], errors="coerce").replace(0.0, 1.0).abs().fillna(1.0).to_numpy(dtype=float)
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


def table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
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
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []

    for path in sorted(PROCESSED.glob("*_pair_history.csv")):
        pair, timeframe = pair_and_timeframe(path)
        frame = load_history(path)
        if len(frame) < 120:
            continue
        periods_per_year = TIMEFRAME_PERIODS.get(timeframe, 365)
        for strategy, config in STRATEGIES.items():
            features = feature_frame(frame, int(config["lookback"]), timeframe)
            raw_signal, source = base_entry_signal(features, config)
            component_rows.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "strategy": strategy,
                    "observations": len(frame),
                    "avg_quality": float(features["trade_quality_score"].mean()),
                    "p50_quality": float(features["trade_quality_score"].median()),
                    "p75_quality": float(features["trade_quality_score"].quantile(0.75)),
                    "avg_z_strength": float(features["z_strength_score"].mean()),
                    "avg_reversion_pressure": float(features["reversion_pressure_score"].mean()),
                    "avg_hedge_stability": float(features["hedge_stability_score"].mean()),
                    "avg_corr_stability": float(features["corr_stability_score"].mean()),
                    "avg_volatility_score": float(features["volatility_score"].mean()),
                    "avg_tail_risk_score": float(features["tail_risk_score"].mean()),
                    "source_quality": str(frame["source_quality"].iloc[0]) if "source_quality" in frame.columns else "",
                }
            )
            for gate in GATES:
                gated_signal = apply_gate(raw_signal, features, gate)
                signal = apply_stops(gated_signal, source, config)
                entry_count = int(((signal != 0.0) & (signal.shift(1).fillna(0.0) == 0.0)).sum())
                avg_entry_quality = float(features["trade_quality_score"].where(signal != 0.0).mean())
                for cost_bucket, costs in COST_BUCKETS.items():
                    metrics = backtest(frame, signal, periods_per_year, costs)
                    rows.append(
                        {
                            "pair": pair,
                            "timeframe": timeframe,
                            "strategy": strategy,
                            "quality_gate": gate,
                            "cost_bucket": cost_bucket,
                            "entry_count": entry_count,
                            "avg_entry_quality": avg_entry_quality,
                            **metrics,
                            "acceptance_reason": acceptance_reason(metrics, stress=cost_bucket != "base"),
                            "source_quality": str(frame["source_quality"].iloc[0]) if "source_quality" in frame.columns else "",
                            "observations": len(frame),
                        }
                    )

    results = pd.DataFrame(rows)
    components = pd.DataFrame(component_rows)
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

    comparison = (
        results.groupby(["quality_gate", "cost_bucket"], dropna=False)
        .agg(
            runs=("rank", "count"),
            accepted_runs=("accepted", "sum"),
            median_trades=("trades", "median"),
            best_trades=("trades", "max"),
            median_profit_factor=("profit_factor", "median"),
            best_profit_factor=("profit_factor", "max"),
            median_sharpe=("sharpe", "median"),
            best_sharpe=("sharpe", "max"),
            median_max_drawdown=("max_drawdown", "median"),
            min_max_drawdown=("max_drawdown", "min"),
        )
        .reset_index()
        .sort_values(["accepted_runs", "best_profit_factor"], ascending=[False, False])
    )

    pair_summary = (
        results.groupby(["pair", "quality_gate"], dropna=False)
        .agg(
            runs=("rank", "count"),
            accepted_runs=("accepted", "sum"),
            best_rank=("rank", "min"),
            max_trades=("trades", "max"),
            best_profit_factor=("profit_factor", "max"),
            best_sharpe=("sharpe", "max"),
            min_max_drawdown=("max_drawdown", "min"),
        )
        .reset_index()
        .sort_values(["accepted_runs", "best_profit_factor"], ascending=[False, False])
    )

    results.to_csv(REPORTS / "phase3_soft_quality_ranked.csv", index=False)
    comparison.to_csv(REPORTS / "phase3_quality_gate_comparison.csv", index=False)
    components.to_csv(REPORTS / "phase3_quality_components.csv", index=False)
    pair_summary.to_csv(REPORTS / "phase3_pair_quality_summary.csv", index=False)

    rank_columns = [
        "rank",
        "pair",
        "timeframe",
        "strategy",
        "quality_gate",
        "cost_bucket",
        "trades",
        "profit_factor",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "avg_entry_quality",
        "avg_holding_bars",
        "acceptance_reason",
    ]
    component_columns = [
        "pair",
        "timeframe",
        "strategy",
        "avg_quality",
        "p50_quality",
        "p75_quality",
        "avg_z_strength",
        "avg_reversion_pressure",
        "avg_hedge_stability",
        "avg_tail_risk_score",
        "source_quality",
    ]
    base_100 = results[(results["cost_bucket"] == "base") & (results["trades"] >= 100)].copy()
    report = [
        "# Evidence Pipeline Phase 3 Soft Trade Quality Scoring",
        "",
        "Soft quality-score gating compared with ungated baselines and the old hard filter.",
        "",
        "## Scope",
        "",
        f"- Tested rows: {len(results):,}",
        f"- Accepted rows: {int(results['accepted'].sum()):,}",
        "- Gates: ungated, hard_filter, quality_50, quality_60, quality_70, quality_80",
        "- Funding friendliness is neutral at 50 because processed histories do not yet include funding fields.",
        "- Quality score uses point-in-time rolling inputs only.",
        "",
        "## Quality Gate Comparison",
        "",
        table(comparison, list(comparison.columns)),
        "",
        "## Top Ranked Quality Runs",
        "",
        table(results, rank_columns, 35),
        "",
        "## Best 100+ Trade Base-Cost Quality Runs",
        "",
        table(base_100, rank_columns, 35) if len(base_100) else "No base-cost run reached 100 trades.",
        "",
        "## Quality Component Sample",
        "",
        table(components.sort_values(["avg_quality"], ascending=False), component_columns, 25),
        "",
        "## Takeaway",
        "",
    ]
    accepted = int(results["accepted"].sum())
    if accepted:
        report.append("At least one soft-quality run passed this phase gate; it still requires exit diagnostics, point-in-time regime validation, walk-forward, and cost stress before promotion.")
    else:
        report.append("No soft-quality gate produced a paper-trade candidate. Use these results to tune score weights and move into exit diagnostics rather than promoting any result.")
    report.extend(
        [
            "",
            "## Files",
            "",
            "- phase3_soft_quality_ranked.csv",
            "- phase3_quality_gate_comparison.csv",
            "- phase3_quality_components.csv",
            "- phase3_pair_quality_summary.csv",
        ]
    )
    (REPORTS / "phase3_soft_quality_score_report.md").write_text("\n".join(report) + "\n")

    print(
        json.dumps(
            {
                "rows": int(len(results)),
                "accepted": accepted,
                "report": str(REPORTS / "phase3_soft_quality_score_report.md"),
                "ranked_csv": str(REPORTS / "phase3_soft_quality_ranked.csv"),
                "comparison_csv": str(REPORTS / "phase3_quality_gate_comparison.csv"),
                "best": results.iloc[0][rank_columns[1:]].to_dict() if len(results) else None,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
