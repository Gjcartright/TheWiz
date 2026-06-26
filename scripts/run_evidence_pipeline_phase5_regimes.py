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

REGIMES = (
    "all",
    "range_only",
    "exclude_crisis",
    "calm_vol_only",
    "stable_correlation",
    "stable_hedge",
    "range_and_stable_hedge",
)

BEHAVIORS = ("entry_only", "hard_exit", "size_down")


def history_path(pair: str, timeframe: str) -> Path:
    return PROCESSED / f"{pair.lower().replace('-', '_')}_{timeframe}_pair_history.csv"


def candidate_rows(limit: int = 100) -> pd.DataFrame:
    ranked = pd.read_csv(PHASE3_RANKED)
    serious = ranked[(ranked["cost_bucket"] == "base") & (ranked["trades"] >= 100)].copy()
    serious = serious.sort_values(["rank", "profit_factor", "sharpe"], ascending=[True, False, False])
    return serious.head(limit).reset_index(drop=True)


def point_in_time_regime_features(frame: pd.DataFrame, lookback: int) -> pd.DataFrame:
    min_periods = max(20, lookback // 4)
    spread = pd.to_numeric(frame["spread"], errors="coerce")
    ret_x = pd.to_numeric(frame["return_x"], errors="coerce")
    ret_y = pd.to_numeric(frame["return_y"], errors="coerce")
    spread_return = spread.diff()
    rolling_spread_vol = spread_return.rolling(lookback, min_periods=min_periods).std()
    spread_vol_rank = rolling_spread_vol.rolling(lookback * 3, min_periods=max(30, lookback)).rank(pct=True)
    trend_strength = spread.diff(max(2, lookback // 8)).abs() / rolling_spread_vol.replace(0.0, np.nan)
    trend_rank = trend_strength.rolling(lookback * 3, min_periods=max(30, lookback)).rank(pct=True)
    corr = ret_x.rolling(lookback, min_periods=min_periods).corr(ret_y)
    corr_stability = 1.0 - corr.rolling(lookback, min_periods=min_periods).std().rank(pct=True)
    liquidity = pd.to_numeric(frame.get("min_usd_volume", 0.0), errors="coerce").fillna(0.0)
    liquidity_rank = liquidity.rolling(lookback * 3, min_periods=max(30, lookback)).rank(pct=True)
    # Funding is not present in the current processed histories; keep a neutral point-in-time placeholder.
    funding_spread_proxy = pd.Series(0.0, index=frame.index)
    return pd.DataFrame(
        {
            "rolling_spread_vol": rolling_spread_vol,
            "spread_vol_rank": spread_vol_rank,
            "trend_rank": trend_rank,
            "corr": corr,
            "corr_stability": corr_stability,
            "liquidity_rank": liquidity_rank,
            "funding_spread_proxy": funding_spread_proxy,
        },
        index=frame.index,
    )


def regime_mask(regime_features: pd.DataFrame, name: str) -> pd.Series:
    if name == "all":
        return pd.Series(True, index=regime_features.index)
    range_only = (
        (regime_features["spread_vol_rank"].fillna(1.0) <= 0.70)
        & (regime_features["trend_rank"].fillna(1.0) <= 0.70)
        & (regime_features["corr"].abs().fillna(0.0) >= 0.25)
    )
    exclude_crisis = ~(
        (regime_features["spread_vol_rank"].fillna(1.0) >= 0.88)
        | (regime_features["trend_rank"].fillna(1.0) >= 0.90)
        | (regime_features["liquidity_rank"].fillna(0.5) <= 0.05)
    )
    calm_vol = regime_features["spread_vol_rank"].fillna(1.0) <= 0.55
    stable_corr = (
        (regime_features["corr"].abs().fillna(0.0) >= 0.35)
        & (regime_features["corr_stability"].fillna(0.0) >= 0.45)
    )
    stable_hedge = (
        (regime_features["corr"].abs().fillna(0.0) >= 0.45)
        & (regime_features["corr_stability"].fillna(0.0) >= 0.50)
        & (regime_features["spread_vol_rank"].fillna(1.0) <= 0.70)
    )
    if name == "range_only":
        return range_only
    if name == "exclude_crisis":
        return exclude_crisis
    if name == "calm_vol_only":
        return calm_vol
    if name == "stable_correlation":
        return stable_corr
    if name == "stable_hedge":
        return stable_hedge
    if name == "range_and_stable_hedge":
        return range_only & stable_hedge
    raise ValueError(f"unknown regime: {name}")


def apply_regime_behavior(signal: pd.Series, mask: pd.Series, behavior: str) -> pd.Series:
    raw = signal.fillna(0.0).to_numpy(dtype=float)
    allowed = mask.reindex(signal.index).fillna(False).to_numpy(dtype=bool)
    out = np.zeros(len(raw), dtype=float)
    position = 0.0
    for idx, target in enumerate(raw):
        if behavior == "hard_exit":
            position = target if allowed[idx] else 0.0
        elif behavior == "entry_only":
            if position == 0.0:
                position = target if allowed[idx] else 0.0
            else:
                if target == 0.0 or np.sign(target) != np.sign(position):
                    position = 0.0 if not allowed[idx] else target
                else:
                    position = target
        elif behavior == "size_down":
            position = target if allowed[idx] else target * 0.5
        else:
            raise ValueError(f"unknown behavior: {behavior}")
        out[idx] = position
    return pd.Series(out, index=signal.index)


def regime_coverage_summary(mask: pd.Series) -> dict[str, float]:
    values = mask.fillna(False).astype(bool)
    return {
        "regime_allowed_share": float(values.mean()) if len(values) else 0.0,
        "regime_allowed_bars": int(values.sum()),
    }


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
    candidates = candidate_rows()
    rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for _, candidate in candidates.iterrows():
        pair = str(candidate["pair"])
        timeframe = str(candidate["timeframe"])
        strategy = str(candidate["strategy"])
        quality_gate = str(candidate["quality_gate"])
        path = history_path(pair, timeframe)
        if not path.exists():
            continue
        frame = phase3.load_history(path)
        config = phase3.STRATEGIES[strategy]
        features = phase3.feature_frame(frame, int(config["lookback"]), timeframe)
        raw, source = phase3.base_entry_signal(features, config)
        gated = phase3.apply_gate(raw, features, quality_gate)
        base_signal = phase3.apply_stops(gated, source, config)
        regime_features = point_in_time_regime_features(frame, int(config["lookback"]))
        for regime in REGIMES:
            mask = regime_mask(regime_features, regime)
            cov = regime_coverage_summary(mask)
            coverage_rows.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "strategy": strategy,
                    "quality_gate": quality_gate,
                    "regime": regime,
                    **cov,
                    "source_phase3_rank": int(candidate["rank"]),
                }
            )
            for behavior in BEHAVIORS:
                regime_signal = apply_regime_behavior(base_signal, mask, behavior)
                for cost_bucket, costs in phase3.COST_BUCKETS.items():
                    metrics = phase3.backtest(
                        frame,
                        regime_signal,
                        phase3.TIMEFRAME_PERIODS.get(timeframe, 365),
                        costs,
                    )
                    rows.append(
                        {
                            "pair": pair,
                            "timeframe": timeframe,
                            "strategy": strategy,
                            "quality_gate": quality_gate,
                            "regime": regime,
                            "regime_behavior": behavior,
                            "cost_bucket": cost_bucket,
                            **metrics,
                            "acceptance_reason": phase3.acceptance_reason(metrics, stress=cost_bucket != "base"),
                            "source_phase3_rank": int(candidate["rank"]),
                            "source_phase3_pf": float(candidate["profit_factor"]),
                            "source_phase3_dd": float(candidate["max_drawdown"]),
                            **cov,
                        }
                    )

    results = pd.DataFrame(rows)
    coverage = pd.DataFrame(coverage_rows).drop_duplicates()
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

    summary = (
        results.groupby(["regime", "regime_behavior", "cost_bucket"], dropna=False)
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
        results.groupby(["pair", "regime", "regime_behavior"], dropna=False)
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

    results.to_csv(REPORTS / "phase5_point_in_time_regime_ranked.csv", index=False)
    summary.to_csv(REPORTS / "phase5_regime_behavior_summary.csv", index=False)
    pair_summary.to_csv(REPORTS / "phase5_pair_regime_summary.csv", index=False)
    coverage.to_csv(REPORTS / "phase5_regime_coverage.csv", index=False)

    rank_columns = [
        "rank",
        "pair",
        "timeframe",
        "strategy",
        "quality_gate",
        "regime",
        "regime_behavior",
        "cost_bucket",
        "trades",
        "profit_factor",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "regime_allowed_share",
        "acceptance_reason",
    ]
    base_100 = results[(results["cost_bucket"] == "base") & (results["trades"] >= 100)].copy()
    report = [
        "# Evidence Pipeline Phase 5 Point-In-Time Regime Filters",
        "",
        "Point-in-time regime overlays using rolling volatility, trend strength, correlation stability, spread volatility, liquidity, and a neutral funding placeholder.",
        "",
        "## Scope",
        "",
        f"- Phase 3 serious candidates used: {len(candidates):,}",
        f"- Regime test rows: {len(results):,}",
        f"- Accepted rows: {int(results['accepted'].sum()):,}",
        "- Regimes: all, range_only, exclude_crisis, calm_vol_only, stable_correlation, stable_hedge, range_and_stable_hedge",
        "- Behaviors: entry_only, hard_exit, size_down",
        "- No future returns, future drawdown, or hindsight labels are used.",
        "- Funding spread is neutral because processed histories do not yet include funding.",
        "",
        "## Regime Behavior Summary",
        "",
        table(summary, list(summary.columns)),
        "",
        "## Top Ranked Regime Runs",
        "",
        table(results, rank_columns, 35),
        "",
        "## Best 100+ Trade Base-Cost Regime Runs",
        "",
        table(base_100, rank_columns, 35) if len(base_100) else "No base-cost regime run reached 100 trades.",
        "",
        "## Pair Regime Summary",
        "",
        table(pair_summary, list(pair_summary.columns), 35),
        "",
        "## Takeaway",
        "",
    ]
    if int(results["accepted"].sum()):
        report.append("At least one regime overlay passed this phase gate. It still requires walk-forward validation and cost stress before any promotion.")
    else:
        report.append("No point-in-time regime overlay produced a paper-trade candidate. The best results should be treated as research inputs for walk-forward rejection/promotion checks, not as deployable strategies.")
    report.extend(
        [
            "",
            "## Files",
            "",
            "- phase5_point_in_time_regime_ranked.csv",
            "- phase5_regime_behavior_summary.csv",
            "- phase5_pair_regime_summary.csv",
            "- phase5_regime_coverage.csv",
        ]
    )
    (REPORTS / "phase5_point_in_time_regime_report.md").write_text("\n".join(report) + "\n")

    print(
        json.dumps(
            {
                "candidates": int(len(candidates)),
                "rows": int(len(results)),
                "accepted": int(results["accepted"].sum()),
                "report": str(REPORTS / "phase5_point_in_time_regime_report.md"),
                "ranked_csv": str(REPORTS / "phase5_point_in_time_regime_ranked.csv"),
                "best": results.iloc[0][rank_columns[1:]].to_dict() if len(results) else None,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
