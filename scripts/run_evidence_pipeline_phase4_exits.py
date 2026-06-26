from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import run_evidence_pipeline_phase3_quality as phase3


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "evidence_pipeline"
REPORTS = ROOT / "reports" / "evidence_pipeline"
PHASE3_RANKED = REPORTS / "phase3_soft_quality_ranked.csv"

EXIT_STYLES = (
    "zscore_normalization_exit",
    "reversion_velocity_stall_exit",
    "mfe_giveback_exit",
    "half_life_timeout_exit",
    "volatility_expansion_exit",
    "hostile_regime_exit",
    "trailing_spread_profit_exit",
)


def history_path(pair: str, timeframe: str) -> Path:
    return PROCESSED / f"{pair.lower().replace('-', '_')}_{timeframe}_pair_history.csv"


def candidate_rows(limit: int = 80) -> pd.DataFrame:
    ranked = pd.read_csv(PHASE3_RANKED)
    serious = ranked[(ranked["cost_bucket"] == "base") & (ranked["trades"] >= 100)].copy()
    serious = serious.sort_values(["rank", "profit_factor", "sharpe"], ascending=[True, False, False])
    return serious.head(limit).reset_index(drop=True)


def source_for(frame: pd.DataFrame, features: pd.DataFrame, strategy: str) -> pd.Series:
    raw, source = phase3.base_entry_signal(features, phase3.STRATEGIES[strategy])
    return source


def candidate_signal(frame: pd.DataFrame, features: pd.DataFrame, row: pd.Series) -> tuple[pd.Series, pd.Series]:
    config = phase3.STRATEGIES[str(row["strategy"])]
    raw, source = phase3.base_entry_signal(features, config)
    gated = phase3.apply_gate(raw, features, str(row["quality_gate"]))
    signal = phase3.apply_stops(gated, source, config)
    return signal, source


def two_leg_returns(frame: pd.DataFrame, position: np.ndarray, costs: dict[str, float], periods_per_year: int) -> np.ndarray:
    price_x = frame["x_close"].to_numpy(dtype=float)
    price_y = frame["y_close"].to_numpy(dtype=float)
    beta = pd.to_numeric(frame["beta_96"], errors="coerce").replace(0.0, 1.0).abs().fillna(1.0).to_numpy(dtype=float)
    ret_x = np.r_[0.0, np.diff(price_x) / price_x[:-1]]
    ret_y = np.r_[0.0, np.diff(price_y) / price_y[:-1]]
    unit_cost = (costs["fee_bps"] + costs["slippage_bps"] + costs["execution_risk_bps"]) / 10_000.0
    funding = (costs["funding_bps_per_day"] / 10_000.0) / max(periods_per_year / 365.0, 1.0)
    returns = np.zeros(len(frame), dtype=float)
    last_target = 0.0
    for idx in range(len(frame)):
        prior = position[idx - 1] if idx else 0.0
        target = position[idx]
        scale = 1.0 + abs(beta[idx])
        weight_y = prior / scale
        weight_x = -prior * abs(beta[idx]) / scale
        gross = weight_x * ret_x[idx] + weight_y * ret_y[idx]
        turnover = abs(target - last_target)
        returns[idx] = gross - turnover * unit_cost - abs(prior) * funding
        last_target = target
    return returns


def regime_label(features: pd.DataFrame, idx: int) -> str:
    vol_rank = float(features["vol_rank"].iloc[idx]) if not pd.isna(features["vol_rank"].iloc[idx]) else 1.0
    corr = abs(float(features["corr"].iloc[idx])) if not pd.isna(features["corr"].iloc[idx]) else 0.0
    z_abs = abs(float(features["zscore"].iloc[idx])) if not pd.isna(features["zscore"].iloc[idx]) else 0.0
    if vol_rank > 0.85 or z_abs > 3.0:
        return "crisis"
    if corr >= 0.45 and vol_rank <= 0.65:
        return "stable_hedge"
    if vol_rank <= 0.55:
        return "calm_vol"
    return "mixed"


def holding_bucket(holding: int) -> str:
    if holding <= 12:
        return "short"
    if holding <= 48:
        return "medium"
    return "long"


def extract_trade_diagnostics(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    signal: pd.Series,
    source: pd.Series,
    row: pd.Series,
) -> list[dict[str, object]]:
    periods_per_year = phase3.TIMEFRAME_PERIODS.get(str(row["timeframe"]), 365)
    returns = two_leg_returns(frame, signal.to_numpy(dtype=float), phase3.COST_BUCKETS["base"], periods_per_year)
    positions = signal.to_numpy(dtype=float)
    source_abs = source.abs().fillna(np.nan).to_numpy(dtype=float)
    trades: list[dict[str, object]] = []
    start: int | None = None
    side = 0.0
    for idx, pos in enumerate(positions):
        prior = positions[idx - 1] if idx else 0.0
        if prior == 0.0 and pos != 0.0:
            start = idx
            side = pos
        if start is not None and prior != 0.0 and pos == 0.0:
            end = idx
            trade_returns = returns[start : end + 1]
            cumulative = np.cumsum(trade_returns)
            if len(cumulative) == 0:
                start = None
                continue
            mfe = float(np.max(cumulative))
            mae = float(np.min(cumulative))
            final = float(cumulative[-1])
            peak_idx = int(np.argmax(cumulative))
            mean_candidates = np.where(source_abs[start : end + 1] <= float(phase3.STRATEGIES[str(row["strategy"])]["exit"]))[0]
            time_to_mean = int(mean_candidates[0]) if len(mean_candidates) else -1
            trades.append(
                {
                    "pair": row["pair"],
                    "timeframe": row["timeframe"],
                    "strategy": row["strategy"],
                    "quality_gate": row["quality_gate"],
                    "entry_time": frame.index[start],
                    "exit_time": frame.index[end],
                    "side": side,
                    "holding_bars": end - start + 1,
                    "holding_bucket": holding_bucket(end - start + 1),
                    "final_pnl": final,
                    "mae": mae,
                    "mfe": mfe,
                    "time_to_peak_profit": peak_idx,
                    "time_to_mean_reversion": time_to_mean,
                    "profit_giveback_after_peak": max(0.0, mfe - final),
                    "entry_zscore": float(features["zscore"].iloc[start]) if not pd.isna(features["zscore"].iloc[start]) else np.nan,
                    "exit_zscore": float(features["zscore"].iloc[end]) if not pd.isna(features["zscore"].iloc[end]) else np.nan,
                    "entry_regime": regime_label(features, start),
                    "exit_regime": regime_label(features, end),
                    "spread_change": float(frame["spread"].iloc[end] - frame["spread"].iloc[start]),
                }
            )
            start = None
            side = 0.0
    return trades


def smart_exit_signal(
    raw_gated: pd.Series,
    source: pd.Series,
    features: pd.DataFrame,
    config: dict[str, float],
    exit_style: str,
) -> pd.Series:
    raw = raw_gated.fillna(0.0).to_numpy(dtype=float)
    src = source.fillna(0.0).to_numpy(dtype=float)
    src_abs = np.abs(src)
    vol_rank = features["vol_rank"].fillna(1.0).to_numpy(dtype=float)
    corr_abs = features["corr"].abs().fillna(0.0).to_numpy(dtype=float)
    zscore = features["zscore"].fillna(0.0).to_numpy(dtype=float)
    out = np.zeros(len(raw), dtype=float)
    position = 0.0
    age = 0
    trade_pnl = 0.0
    peak_pnl = 0.0
    prev_abs: list[float] = []
    prices_dummy = np.zeros(len(raw), dtype=float)
    returns = prices_dummy
    exit_level = float(config["exit"])
    stop = float(config["stop"])
    max_hold = int(config["max_hold"])

    # PnL-aware exits use spread movement as a lightweight proxy while building the target path.
    for idx, target in enumerate(raw):
        if idx > 0 and position != 0.0:
            spread_delta = zscore[idx - 1] - zscore[idx]
            trade_pnl += float(np.sign(position) * spread_delta * 0.002)
            peak_pnl = max(peak_pnl, trade_pnl)
        hard_exit = False
        if position == 0.0 and target != 0.0:
            position = target
            age = 0
            trade_pnl = 0.0
            peak_pnl = 0.0
            prev_abs = []
        elif position != 0.0:
            age += 1
            prev_abs.append(src_abs[idx])
            if src_abs[idx] <= exit_level:
                hard_exit = True
            if src_abs[idx] >= stop:
                hard_exit = True
            if exit_style == "zscore_normalization_exit" and src_abs[idx] <= max(exit_level, 0.20):
                hard_exit = True
            elif exit_style == "reversion_velocity_stall_exit" and age >= 6 and len(prev_abs) >= 4:
                recent = prev_abs[-4:]
                if recent[-1] >= min(recent[:3]):
                    hard_exit = True
            elif exit_style == "mfe_giveback_exit" and peak_pnl >= 0.012 and trade_pnl < peak_pnl * 0.55:
                hard_exit = True
            elif exit_style == "half_life_timeout_exit" and age >= max(12, min(max_hold, int(config["lookback"]) // 2)):
                hard_exit = True
            elif exit_style == "volatility_expansion_exit" and vol_rank[idx] >= 0.82:
                hard_exit = True
            elif exit_style == "hostile_regime_exit" and (vol_rank[idx] >= 0.85 or corr_abs[idx] < 0.20):
                hard_exit = True
            elif exit_style == "trailing_spread_profit_exit" and peak_pnl >= 0.010 and trade_pnl < peak_pnl * 0.70:
                hard_exit = True
            if age >= max_hold:
                hard_exit = True
            if target == 0.0 or (target != 0.0 and np.sign(target) != np.sign(position)):
                hard_exit = True
            if hard_exit:
                position = 0.0
                age = 0
                trade_pnl = 0.0
                peak_pnl = 0.0
                prev_abs = []
            else:
                position = target if target != 0.0 else position
        out[idx] = position
    return pd.Series(out, index=raw_gated.index)


def metrics_for_signal(frame: pd.DataFrame, signal: pd.Series, timeframe: str, cost_bucket: str) -> dict[str, float]:
    periods_per_year = phase3.TIMEFRAME_PERIODS.get(timeframe, 365)
    return phase3.backtest(frame, signal, periods_per_year, phase3.COST_BUCKETS[cost_bucket])


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
    candidates = candidate_rows(limit=80)
    diagnostics: list[dict[str, object]] = []
    exit_rows: list[dict[str, object]] = []

    for _, row in candidates.iterrows():
        path = history_path(str(row["pair"]), str(row["timeframe"]))
        if not path.exists():
            continue
        frame = phase3.load_history(path)
        config = phase3.STRATEGIES[str(row["strategy"])]
        features = phase3.feature_frame(frame, int(config["lookback"]), str(row["timeframe"]))
        raw, source = phase3.base_entry_signal(features, config)
        raw_gated = phase3.apply_gate(raw, features, str(row["quality_gate"]))
        baseline_signal = phase3.apply_stops(raw_gated, source, config)
        diagnostics.extend(extract_trade_diagnostics(frame, features, baseline_signal, source, row))

        for exit_style in EXIT_STYLES:
            signal = smart_exit_signal(raw_gated, source, features, config, exit_style)
            for cost_bucket in phase3.COST_BUCKETS:
                metrics = metrics_for_signal(frame, signal, str(row["timeframe"]), cost_bucket)
                exit_rows.append(
                    {
                        "pair": row["pair"],
                        "timeframe": row["timeframe"],
                        "strategy": row["strategy"],
                        "quality_gate": row["quality_gate"],
                        "exit_style": exit_style,
                        "cost_bucket": cost_bucket,
                        **metrics,
                        "acceptance_reason": phase3.acceptance_reason(metrics, stress=cost_bucket != "base"),
                        "source_phase3_rank": int(row["rank"]),
                        "source_phase3_pf": float(row["profit_factor"]),
                        "source_phase3_dd": float(row["max_drawdown"]),
                    }
                )

    diag = pd.DataFrame(diagnostics)
    exits = pd.DataFrame(exit_rows)
    exits["accepted"] = exits["acceptance_reason"].eq("ACCEPT")
    exits["rank_score"] = (
        exits["accepted"].astype(int) * 100_000
        + exits["profit_factor"].fillna(0.0).clip(upper=20.0) * 1000.0
        + exits["sharpe"].fillna(0.0).clip(lower=-10.0, upper=20.0) * 100.0
        - exits["max_drawdown"].fillna(1.0).clip(upper=5.0) * 100.0
        + np.minimum(exits["trades"].fillna(0.0), 300.0)
    )
    exits = exits.sort_values(["accepted", "rank_score", "trades"], ascending=[False, False, False]).reset_index(
        drop=True
    )
    exits.insert(0, "rank", np.arange(1, len(exits) + 1))

    if diag.empty:
        diag_summary = pd.DataFrame()
        regime_summary = pd.DataFrame()
        holding_summary = pd.DataFrame()
    else:
        diag_summary = (
            diag.groupby(["pair", "timeframe", "strategy", "quality_gate"], dropna=False)
            .agg(
                trades=("final_pnl", "count"),
                median_mae=("mae", "median"),
                median_mfe=("mfe", "median"),
                median_giveback=("profit_giveback_after_peak", "median"),
                median_time_to_peak=("time_to_peak_profit", "median"),
                median_holding=("holding_bars", "median"),
                win_rate=("final_pnl", lambda s: float((s > 0).mean())),
                avg_final_pnl=("final_pnl", "mean"),
            )
            .reset_index()
            .sort_values(["avg_final_pnl"], ascending=False)
        )
        regime_summary = (
            diag.groupby(["entry_regime"], dropna=False)
            .agg(
                trades=("final_pnl", "count"),
                win_rate=("final_pnl", lambda s: float((s > 0).mean())),
                avg_final_pnl=("final_pnl", "mean"),
                median_mae=("mae", "median"),
                median_mfe=("mfe", "median"),
                median_giveback=("profit_giveback_after_peak", "median"),
            )
            .reset_index()
            .sort_values("avg_final_pnl", ascending=False)
        )
        holding_summary = (
            diag.groupby(["holding_bucket"], dropna=False)
            .agg(
                trades=("final_pnl", "count"),
                win_rate=("final_pnl", lambda s: float((s > 0).mean())),
                avg_final_pnl=("final_pnl", "mean"),
                median_mae=("mae", "median"),
                median_mfe=("mfe", "median"),
                median_giveback=("profit_giveback_after_peak", "median"),
            )
            .reset_index()
            .sort_values("avg_final_pnl", ascending=False)
        )

    exit_summary = (
        exits.groupby(["exit_style", "cost_bucket"], dropna=False)
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

    diag.to_csv(REPORTS / "phase4_exit_trade_diagnostics.csv", index=False)
    diag_summary.to_csv(REPORTS / "phase4_exit_diagnostics_summary.csv", index=False)
    regime_summary.to_csv(REPORTS / "phase4_exit_regime_summary.csv", index=False)
    holding_summary.to_csv(REPORTS / "phase4_exit_holding_summary.csv", index=False)
    exits.to_csv(REPORTS / "phase4_smart_exit_ranked.csv", index=False)
    exit_summary.to_csv(REPORTS / "phase4_smart_exit_summary.csv", index=False)

    rank_columns = [
        "rank",
        "pair",
        "timeframe",
        "strategy",
        "quality_gate",
        "exit_style",
        "cost_bucket",
        "trades",
        "profit_factor",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "avg_holding_bars",
        "acceptance_reason",
    ]
    base_100 = exits[(exits["cost_bucket"] == "base") & (exits["trades"] >= 100)].copy()
    report = [
        "# Evidence Pipeline Phase 4 Exit Research",
        "",
        "Exit diagnostics and smarter-exit tests for serious Phase 3 candidates with at least 100 base-cost trades.",
        "",
        "## Scope",
        "",
        f"- Phase 3 candidate rows used: {len(candidates):,}",
        f"- Diagnostic trades extracted: {len(diag):,}",
        f"- Smart-exit rows: {len(exits):,}",
        f"- Accepted smart-exit rows: {int(exits['accepted'].sum()):,}",
        "",
        "## Smart Exit Summary",
        "",
        table(exit_summary, list(exit_summary.columns)),
        "",
        "## Top Ranked Smart Exits",
        "",
        table(exits, rank_columns, 35),
        "",
        "## Best 100+ Trade Base-Cost Smart Exits",
        "",
        table(base_100, rank_columns, 35) if len(base_100) else "No base-cost smart-exit run reached 100 trades.",
        "",
        "## Exit Diagnostic Summary",
        "",
        table(
            diag_summary,
            [
                "pair",
                "timeframe",
                "strategy",
                "quality_gate",
                "trades",
                "median_mae",
                "median_mfe",
                "median_giveback",
                "median_time_to_peak",
                "median_holding",
                "win_rate",
                "avg_final_pnl",
            ],
            30,
        ),
        "",
        "## Regime Diagnostic Summary",
        "",
        table(regime_summary, list(regime_summary.columns)),
        "",
        "## Holding-Time Diagnostic Summary",
        "",
        table(holding_summary, list(holding_summary.columns)),
        "",
        "## Takeaway",
        "",
    ]
    if int(exits["accepted"].sum()):
        report.append("At least one smart exit passed the Phase 4 gate, but it still requires point-in-time regime validation, walk-forward, and cost stress before promotion.")
    else:
        report.append("No smart exit produced a paper-trade candidate. The exit diagnostics should be used to tune Phase 5 regime/size-down rules and later walk-forward tests.")
    report.extend(
        [
            "",
            "## Files",
            "",
            "- phase4_exit_trade_diagnostics.csv",
            "- phase4_exit_diagnostics_summary.csv",
            "- phase4_exit_regime_summary.csv",
            "- phase4_exit_holding_summary.csv",
            "- phase4_smart_exit_ranked.csv",
            "- phase4_smart_exit_summary.csv",
        ]
    )
    (REPORTS / "phase4_exit_research_report.md").write_text("\n".join(report) + "\n")

    print(
        json.dumps(
            {
                "candidates": int(len(candidates)),
                "diagnostic_trades": int(len(diag)),
                "smart_exit_rows": int(len(exits)),
                "accepted": int(exits["accepted"].sum()),
                "report": str(REPORTS / "phase4_exit_research_report.md"),
                "ranked_csv": str(REPORTS / "phase4_smart_exit_ranked.csv"),
                "diagnostics_csv": str(REPORTS / "phase4_exit_trade_diagnostics.csv"),
                "best": exits.iloc[0][rank_columns[1:]].to_dict() if len(exits) else None,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
