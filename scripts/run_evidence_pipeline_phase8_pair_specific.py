from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_evidence_pipeline_phase3_quality as phase3
import run_evidence_pipeline_phase5_regimes as phase5
import run_evidence_pipeline_phase6_walk_forward as phase6


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "evidence_pipeline"


PAIR_TREATMENTS = {
    "ETH-SOL": {
        "role": "primary research candidate",
        "timeframes": ("1h", "4h"),
        "strategies": ("zscore_1_5_exit_0_25", "zscore_2_0_exit_0", "ou_mean_reversion"),
        "quality_gates": ("hard_filter", "quality_60"),
        "regimes": ("range_only", "exclude_crisis", "stable_hedge"),
        "behaviors": ("entry_only", "hard_exit"),
        "entry_styles": ("reversion_confirmed", "beta_anchor"),
        "exit_styles": ("fast_mean_exit", "profit_giveback_exit"),
        "hypothesis": "Mean reversion can work only when 1h/4h hedge stability and drawdown stay controlled.",
    },
    "BTC-DOGE": {
        "role": "dangerous research candidate",
        "timeframes": ("4h", "1d"),
        "strategies": ("beta_dislocation", "copula_dislocation"),
        "quality_gates": ("hard_filter", "quality_60"),
        "regimes": ("exclude_crisis", "calm_vol_only", "stable_hedge"),
        "behaviors": ("hard_exit",),
        "entry_styles": ("drawdown_first", "extreme_only"),
        "exit_styles": ("vol_cut_exit", "profit_giveback_exit"),
        "hypothesis": "BTC-DOGE must reduce drawdown before PF matters; 4h/1d only for this repair pass.",
    },
    "BTC-SOL": {
        "role": "coverage and anchor candidate",
        "timeframes": ("1h", "4h"),
        "strategies": ("beta_dislocation", "zscore_1_5_exit_0_25"),
        "quality_gates": ("hard_filter", "quality_60"),
        "regimes": ("range_only", "exclude_crisis", "stable_hedge"),
        "behaviors": ("entry_only", "hard_exit"),
        "entry_styles": ("beta_anchor", "reversion_confirmed"),
        "exit_styles": ("fast_mean_exit", "profit_giveback_exit"),
        "hypothesis": "BTC-SOL needs beta anchoring and relative-value filters to avoid validation collapse.",
    },
    "ETH-LINK": {
        "role": "stationarity clue, weak-correlation candidate",
        "timeframes": ("1h", "4h"),
        "strategies": ("ou_mean_reversion", "beta_dislocation"),
        "quality_gates": ("hard_filter", "quality_70"),
        "regimes": ("stable_correlation", "stable_hedge"),
        "behaviors": ("entry_only", "hard_exit"),
        "entry_styles": ("stable_corr",),
        "exit_styles": ("fast_mean_exit", "stall_exit"),
        "hypothesis": "ETH-LINK must require rolling correlation stability; otherwise stationarity clues are not enough.",
    },
    "DOGE-XRP": {
        "role": "speculative high-beta retail candidate",
        "timeframes": ("5m",),
        "strategies": ("zscore_1_5_exit_0_25", "copula_dislocation", "beta_dislocation"),
        "quality_gates": ("hard_filter", "quality_60"),
        "regimes": ("exclude_crisis", "calm_vol_only"),
        "behaviors": ("entry_only", "hard_exit"),
        "entry_styles": ("calm_tail", "drawdown_first"),
        "exit_styles": ("vol_cut_exit", "fast_mean_exit"),
        "hypothesis": "DOGE-XRP is coverage-limited and must pass volatility/crisis filters before any promotion.",
    },
}

CONTEXT_CACHE: dict[tuple[str, str, str], dict[str, object]] = {}
REQUIRED_TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")


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


def available_history(pair: str, timeframe: str) -> bool:
    return phase5.history_path(pair, timeframe).exists()


def entry_mask(
    entry_style: str,
    features: pd.DataFrame,
    regime_features: pd.DataFrame,
) -> pd.Series:
    zscore = features["zscore"].fillna(0.0)
    dz = features["dz"].fillna(0.0)
    if entry_style == "standard":
        mask = pd.Series(True, index=features.index)
    elif entry_style == "reversion_confirmed":
        mask = zscore * dz < 0.0
    elif entry_style == "beta_anchor":
        mask = (features["beta_stability"].fillna(0.0) >= 0.55) & (features["corr"].abs().fillna(0.0) >= 0.35)
    elif entry_style == "stable_corr":
        mask = (features["corr"].abs().fillna(0.0) >= 0.45) & (
            regime_features["corr_stability"].fillna(0.0) >= 0.50
        )
    elif entry_style == "drawdown_first":
        mask = (regime_features["spread_vol_rank"].fillna(1.0) <= 0.50) & (
            regime_features["trend_rank"].fillna(1.0) <= 0.65
        )
    elif entry_style == "extreme_only":
        mask = (features["abs_z_rank"].fillna(0.0) >= 0.55) & (features["vol_rank"].fillna(1.0) <= 0.75)
    elif entry_style == "calm_tail":
        mask = (regime_features["spread_vol_rank"].fillna(1.0) <= 0.60) & (
            features["abs_z_rank"].fillna(1.0) <= 0.80
        )
    else:
        raise ValueError(f"unknown entry style: {entry_style}")
    return mask.reindex(features.index).fillna(False)


def apply_exit_style(
    signal: pd.Series,
    features: pd.DataFrame,
    regime_features: pd.DataFrame,
    exit_style: str,
    config: dict[str, float],
) -> pd.Series:
    if exit_style == "base_stops":
        return signal.copy()

    raw = signal.fillna(0.0).to_numpy(dtype=float)
    zscore = features["zscore"].fillna(0.0).to_numpy(dtype=float)
    dz = features["dz"].fillna(0.0).to_numpy(dtype=float)
    spread_vol_rank = regime_features["spread_vol_rank"].fillna(1.0).to_numpy(dtype=float)
    trend_rank = regime_features["trend_rank"].fillna(1.0).to_numpy(dtype=float)
    max_hold = int(config["max_hold"])
    out = np.zeros(len(raw), dtype=float)
    position = 0.0
    age = 0
    entry_z = 0.0
    best_favorable = 0.0

    for idx, target in enumerate(raw):
        if position == 0.0 and target != 0.0:
            position = target
            age = 0
            entry_z = zscore[idx]
            best_favorable = 0.0
        elif position != 0.0:
            age += 1
            favorable = position * (zscore[idx] - entry_z)
            best_favorable = max(best_favorable, favorable)
            should_exit = target == 0.0 or np.sign(target) != np.sign(position)
            if exit_style == "fast_mean_exit":
                should_exit = should_exit or abs(zscore[idx]) <= max(float(config["exit"]), 0.50)
            elif exit_style == "stall_exit":
                should_exit = should_exit or (age >= 6 and position * dz[idx] <= 0.0)
            elif exit_style == "profit_giveback_exit":
                giveback = best_favorable - favorable
                should_exit = should_exit or (best_favorable >= 0.80 and giveback >= max(0.30, best_favorable * 0.40))
            elif exit_style == "vol_cut_exit":
                should_exit = should_exit or spread_vol_rank[idx] >= 0.75 or trend_rank[idx] >= 0.88
            elif exit_style == "time_decay_exit":
                should_exit = should_exit or age >= max(6, int(max_hold * 0.55))
            else:
                raise ValueError(f"unknown exit style: {exit_style}")
            if should_exit:
                position = 0.0
                age = 0
                entry_z = 0.0
                best_favorable = 0.0
            else:
                position = target
        out[idx] = position
    return pd.Series(out, index=signal.index)


def build_pair_specific_signal(
    pair: str,
    timeframe: str,
    strategy: str,
    quality_gate: str,
    regime: str,
    behavior: str,
    entry_style: str,
    exit_style: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    context_key = (pair, timeframe, strategy)
    if context_key not in CONTEXT_CACHE:
        path = phase5.history_path(pair, timeframe)
        frame = phase3.load_history(path)
        config = phase3.STRATEGIES[strategy]
        lookback = int(config["lookback"])
        features = phase3.feature_frame(frame, lookback, timeframe)
        regime_features = phase5.point_in_time_regime_features(frame, lookback)
        raw_signal, source = phase3.base_entry_signal(features, config)
        CONTEXT_CACHE[context_key] = {
            "frame": frame,
            "config": config,
            "features": features,
            "regime_features": regime_features,
            "raw_signal": raw_signal,
            "source": source,
        }
    context = CONTEXT_CACHE[context_key]
    frame = context["frame"]
    config = context["config"]
    features = context["features"]
    regime_features = context["regime_features"]
    raw_signal = context["raw_signal"]
    source = context["source"]
    raw_signal = raw_signal.where(entry_mask(entry_style, features, regime_features), 0.0)
    gated_signal = phase3.apply_gate(raw_signal, features, quality_gate)
    stopped_signal = phase3.apply_stops(gated_signal, source, config)
    mask = phase5.regime_mask(regime_features, regime)
    regime_signal = phase5.apply_regime_behavior(stopped_signal, mask, behavior)
    final_signal = apply_exit_style(regime_signal, features, regime_features, exit_style, config)
    return frame, final_signal, mask


def walk_forward_metrics(frame: pd.DataFrame, signal: pd.Series, timeframe: str) -> dict[str, object]:
    folds: dict[str, dict[str, object]] = {}
    for fold_name, start_share, end_share in phase6.FOLDS:
        folds[fold_name] = phase6.fold_metrics(
            frame,
            signal,
            timeframe,
            fold_name,
            phase6.fold_slice(len(frame), start_share, end_share),
        )
    train = folds["train"]
    validation = folds["validation"]
    test = folds["test"]
    stability = phase6.stability_label(train, validation, test)
    sample_shaped = phase6.is_sample_shaped(train, validation, test)
    reason = phase6.walk_forward_reason(train, validation, test, stability)
    return {
        "train_trades": int(train["trades"]),
        "validation_trades": int(validation["trades"]),
        "test_trades": int(test["trades"]),
        "train_profit_factor": float(train["profit_factor"]),
        "validation_profit_factor": float(validation["profit_factor"]),
        "test_profit_factor": float(test["profit_factor"]),
        "train_max_drawdown": float(train["max_drawdown"]),
        "validation_max_drawdown": float(validation["max_drawdown"]),
        "test_max_drawdown": float(test["max_drawdown"]),
        "parameter_stability": stability,
        "sample_shaped": bool(sample_shaped),
        "walk_forward_reason": reason,
        "walk_forward_pass": reason == "PASS",
    }


def pair_label(row: pd.Series) -> str:
    if bool(row["walk_forward_pass"]):
        return "cost_stress_candidate"
    if row["trades"] >= 100 and row["profit_factor"] >= 1.2 and row["max_drawdown"] <= 0.20:
        return "watchlist"
    if row["trades"] < 30:
        return "reject_tiny_sample"
    return "research_only"


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []

    for pair, treatment in PAIR_TREATMENTS.items():
        for required_timeframe in REQUIRED_TIMEFRAMES:
            if not available_history(pair, required_timeframe):
                missing_rows.append(
                    {
                        "pair": pair,
                        "timeframe": required_timeframe,
                        "blocker": "missing required processed pair history",
                        "role": treatment["role"],
                    }
                )
        for timeframe in treatment["timeframes"]:
            if not available_history(pair, timeframe):
                missing_rows.append(
                    {
                        "pair": pair,
                        "timeframe": timeframe,
                        "blocker": "missing processed pair history",
                        "role": treatment["role"],
                    }
                )
                continue
            for strategy in treatment["strategies"]:
                for quality_gate in treatment["quality_gates"]:
                    for regime in treatment["regimes"]:
                        for behavior in treatment["behaviors"]:
                            for entry_style in treatment["entry_styles"]:
                                for exit_style in treatment["exit_styles"]:
                                    frame, signal, mask = build_pair_specific_signal(
                                        pair,
                                        timeframe,
                                        strategy,
                                        quality_gate,
                                        regime,
                                        behavior,
                                        entry_style,
                                        exit_style,
                                    )
                                    metrics = phase3.backtest(
                                        frame,
                                        signal,
                                        phase3.TIMEFRAME_PERIODS.get(timeframe, 365),
                                        phase3.COST_BUCKETS["base"],
                                    )
                                    wf = walk_forward_metrics(frame, signal, timeframe)
                                    rows.append(
                                        {
                                            "pair": pair,
                                            "role": treatment["role"],
                                            "timeframe": timeframe,
                                            "strategy": strategy,
                                            "quality_gate": quality_gate,
                                            "regime": regime,
                                            "regime_behavior": behavior,
                                            "entry_style": entry_style,
                                            "exit_style": exit_style,
                                            "regime_allowed_share": float(mask.fillna(False).mean())
                                            if len(mask)
                                            else 0.0,
                                            **metrics,
                                            **wf,
                                        }
                                    )

    results = pd.DataFrame(rows)
    missing = pd.DataFrame(missing_rows)
    if results.empty:
        return results, pd.DataFrame(), missing

    results["base_acceptance_reason"] = results.apply(
        lambda row: phase3.acceptance_reason(
            {
                "trades": int(row["trades"]),
                "profit_factor": float(row["profit_factor"]),
                "sharpe": float(row["sharpe"]),
                "max_drawdown": float(row["max_drawdown"]),
                "expectancy": float(row["expectancy"]),
            }
        ),
        axis=1,
    )
    results["pair_label"] = results.apply(pair_label, axis=1)
    results["rank_score"] = (
        results["walk_forward_pass"].astype(int) * 100_000
        + results["base_acceptance_reason"].eq("ACCEPT").astype(int) * 20_000
        + results["profit_factor"].fillna(0.0).clip(upper=20.0) * 400.0
        + results["test_profit_factor"].where(results["test_trades"] >= 30, 0.0).fillna(0.0).clip(upper=20.0)
        * 800.0
        + results["sharpe"].fillna(0.0).clip(lower=-10.0, upper=20.0) * 50.0
        - results["max_drawdown"].fillna(1.0).clip(upper=5.0) * 250.0
        - results["sample_shaped"].astype(int) * 1_500.0
        + np.minimum(results["trades"].fillna(0.0), 200.0)
    )
    results = results.sort_values(
        ["walk_forward_pass", "sample_shaped", "rank_score", "trades"], ascending=[False, True, False, False]
    ).reset_index(drop=True)
    results.insert(0, "phase8_rank", np.arange(1, len(results) + 1))
    pair_summary = (
        results.groupby("pair", dropna=False)
        .agg(
            tested_runs=("phase8_rank", "count"),
            base_accepts=("base_acceptance_reason", lambda s: int((s == "ACCEPT").sum())),
            walk_forward_passes=("walk_forward_pass", "sum"),
            best_rank=("phase8_rank", "min"),
            best_trades=("trades", "max"),
            best_profit_factor=("profit_factor", "max"),
            best_test_profit_factor=("test_profit_factor", "max"),
            lowest_max_drawdown=("max_drawdown", "min"),
            lowest_test_drawdown=("test_max_drawdown", "min"),
            sample_shaped_runs=("sample_shaped", "sum"),
        )
        .reset_index()
        .sort_values(["walk_forward_passes", "best_profit_factor"], ascending=[False, False])
    )
    return results, pair_summary, missing


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    results, pair_summary, missing = run()
    results.to_csv(REPORTS / "phase8_pair_specific_ranked.csv", index=False)
    pair_summary.to_csv(REPORTS / "phase8_pair_specific_summary.csv", index=False)
    missing.to_csv(REPORTS / "phase8_pair_specific_missing_coverage.csv", index=False)

    rank_columns = [
        "phase8_rank",
        "pair",
        "timeframe",
        "strategy",
        "quality_gate",
        "regime",
        "regime_behavior",
        "entry_style",
        "exit_style",
        "trades",
        "profit_factor",
        "sharpe",
        "max_drawdown",
        "test_trades",
        "test_profit_factor",
        "test_max_drawdown",
        "parameter_stability",
        "sample_shaped",
        "pair_label",
        "base_acceptance_reason",
        "walk_forward_reason",
    ]
    report = [
        "# Evidence Pipeline Phase 8 Pair-Specific Treatment",
        "",
        "Pair-specific repair pass using the Phase 6 failures as constraints. This does not promote any strategy unless it clears base metrics and walk-forward.",
        "",
        "## Scope",
        "",
        f"- Pair-specific runs tested: {len(results):,}",
        f"- Base metric accepts: {int((results['base_acceptance_reason'] == 'ACCEPT').sum()) if len(results) else 0:,}",
        f"- Walk-forward passes: {int(results['walk_forward_pass'].sum()) if len(results) else 0:,}",
        f"- Sample-shaped runs: {int(results['sample_shaped'].sum()) if len(results) else 0:,}",
        "- Phase 7 cost stress is still gated; only walk-forward passers should advance.",
        "",
        "## Pair Hypotheses",
        "",
    ]
    for pair, treatment in PAIR_TREATMENTS.items():
        report.append(f"- {pair}: {treatment['hypothesis']}")
    report.extend(
        [
            "",
            "## Ranked Pair-Specific Results",
            "",
            table(results, rank_columns, 50),
            "",
            "## Pair Summary",
            "",
            table(pair_summary, list(pair_summary.columns), None) if len(pair_summary) else "_No pair summary._",
            "",
            "## Known Required Coverage Gaps",
            "",
            table(missing, list(missing.columns), None) if len(missing) else "_No missing required histories._",
            "",
            "## Gap Analysis",
            "",
            "- Phase 8 still depends on neutral funding inputs, so funding-aware selection remains incomplete.",
            "- DOGE-XRP is still represented only by 5m local history in this pass.",
            "- BTC-DOGE remains drawdown-first; high PF without DD control is not useful for promotion.",
            "- ETH-LINK's strongest rows must be discounted when correlation-stable filters collapse trade count.",
            "",
            "## Premortem",
            "",
            "- If a pair-specific repair looks good in the full sample but fails walk-forward, assume parameter fit rather than edge.",
            "- If a repair improves drawdown by killing trades, it will fail promotion through sample-size fragility.",
            "- If 4h/1d BTC-DOGE cannot produce enough trades, it should remain research-only even if calmer.",
            "",
            "## Red Team",
            "",
            "- Do not let pair-specific tailoring become hindsight optimization; every row is checked chronologically.",
            "- Any row with a strong train PF and weak validation/test PF is evidence against the repair, not a near miss.",
            "- No row may move to paper trading without both walk-forward and future cost stress evidence.",
            "",
            "## Takeaway",
            "",
        ]
    )
    if len(results) and int(results["walk_forward_pass"].sum()):
        report.append("At least one pair-specific row passed walk-forward and can proceed to Phase 7 cost stress. It is still not deployable.")
    else:
        report.append("No pair-specific repair passed walk-forward. The current strategy set remains research-only.")
    report.extend(
        [
            "",
            "## Files",
            "",
            "- phase8_pair_specific_ranked.csv",
            "- phase8_pair_specific_summary.csv",
            "- phase8_pair_specific_missing_coverage.csv",
        ]
    )
    (REPORTS / "phase8_pair_specific_report.md").write_text("\n".join(report) + "\n")

    print(
        json.dumps(
            {
                "rows": int(len(results)),
                "base_accepts": int((results["base_acceptance_reason"] == "ACCEPT").sum()) if len(results) else 0,
                "walk_forward_passes": int(results["walk_forward_pass"].sum()) if len(results) else 0,
                "sample_shaped": int(results["sample_shaped"].sum()) if len(results) else 0,
                "report": str(REPORTS / "phase8_pair_specific_report.md"),
                "best": results.iloc[0][rank_columns[1:]].to_dict() if len(results) else None,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
