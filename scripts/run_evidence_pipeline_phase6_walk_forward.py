from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import run_evidence_pipeline_phase3_quality as phase3
import run_evidence_pipeline_phase5_regimes as phase5


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "evidence_pipeline"
PHASE5_RANKED = REPORTS / "phase5_point_in_time_regime_ranked.csv"

PAIRS = ("ETH-SOL", "BTC-DOGE", "BTC-SOL", "ETH-LINK", "DOGE-XRP")
FOLDS = (
    ("train", 0.0, 0.6),
    ("validation", 0.6, 0.8),
    ("test", 0.8, 1.0),
)


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


def candidate_rows(limit: int = 120) -> pd.DataFrame:
    ranked = pd.read_csv(PHASE5_RANKED)
    base = ranked[ranked["cost_bucket"].eq("base")].copy()
    serious = base[base["trades"] >= 100].sort_values("rank").head(limit)
    best_by_pair = base.sort_values("rank").groupby("pair", as_index=False).head(1)
    candidates = pd.concat([serious, best_by_pair], ignore_index=True)
    key = ["pair", "timeframe", "strategy", "quality_gate", "regime", "regime_behavior"]
    candidates = candidates.drop_duplicates(subset=key).sort_values("rank").reset_index(drop=True)
    candidates.insert(0, "phase6_candidate_id", np.arange(1, len(candidates) + 1))
    return candidates


def build_signal(candidate: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    pair = str(candidate["pair"])
    timeframe = str(candidate["timeframe"])
    strategy = str(candidate["strategy"])
    quality_gate = str(candidate["quality_gate"])
    regime = str(candidate["regime"])
    behavior = str(candidate["regime_behavior"])
    path = phase5.history_path(pair, timeframe)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = phase3.load_history(path)
    config = phase3.STRATEGIES[strategy]
    lookback = int(config["lookback"])
    features = phase3.feature_frame(frame, lookback, timeframe)
    raw_signal, source = phase3.base_entry_signal(features, config)
    gated_signal = phase3.apply_gate(raw_signal, features, quality_gate)
    stopped_signal = phase3.apply_stops(gated_signal, source, config)
    regime_features = phase5.point_in_time_regime_features(frame, lookback)
    mask = phase5.regime_mask(regime_features, regime)
    signal = phase5.apply_regime_behavior(stopped_signal, mask, behavior)
    return frame, signal, mask


def fold_slice(length: int, start_share: float, end_share: float) -> slice:
    start = int(math.floor(length * start_share))
    end = int(math.floor(length * end_share)) if end_share < 1.0 else length
    return slice(start, end)


def fold_metrics(frame: pd.DataFrame, signal: pd.Series, timeframe: str, fold_name: str, slc: slice) -> dict[str, object]:
    fold_frame = frame.iloc[slc].copy()
    fold_signal = signal.iloc[slc].copy()
    metrics = phase3.backtest(
        fold_frame,
        fold_signal,
        phase3.TIMEFRAME_PERIODS.get(timeframe, 365),
        phase3.COST_BUCKETS["base"],
    )
    return {
        "fold": fold_name,
        "fold_start": str(fold_frame.index.min()) if len(fold_frame) else "",
        "fold_end": str(fold_frame.index.max()) if len(fold_frame) else "",
        "fold_bars": int(len(fold_frame)),
        **metrics,
    }


def stability_label(train: dict[str, object], validation: dict[str, object], test: dict[str, object]) -> str:
    train_pf = float(train["profit_factor"])
    val_pf = float(validation["profit_factor"])
    test_pf = float(test["profit_factor"])
    if min(int(train["trades"]), int(validation["trades"]), int(test["trades"])) < 20:
        return "insufficient_fold_trades"
    if train_pf <= 0.0:
        return "unprofitable_train"
    if val_pf >= 0.9 * train_pf and test_pf >= 0.9 * train_pf:
        return "stable"
    if val_pf >= 0.75 * train_pf and test_pf >= 0.75 * train_pf:
        return "moderately_stable"
    return "unstable"


def is_sample_shaped(train: dict[str, object], validation: dict[str, object], test: dict[str, object]) -> bool:
    train_pf = float(train["profit_factor"])
    val_pf = float(validation["profit_factor"])
    test_pf = float(test["profit_factor"])
    return (
        train_pf >= 1.25
        and (val_pf < 1.0 or test_pf < 1.0)
        or min(int(validation["trades"]), int(test["trades"])) < 20
        or (train_pf - min(val_pf, test_pf)) >= 0.75
    )


def walk_forward_reason(train: dict[str, object], validation: dict[str, object], test: dict[str, object], stability: str) -> str:
    failures: list[str] = []
    if min(int(train["trades"]), int(validation["trades"]), int(test["trades"])) < 30:
        failures.append("fold_trades<30")
    for name, metrics in (("train", train), ("validation", validation), ("test", test)):
        if float(metrics["profit_factor"]) < 1.3:
            failures.append(f"{name}_profit_factor<1.3")
        if float(metrics["max_drawdown"]) > 0.15:
            failures.append(f"{name}_drawdown>0.15")
        if float(metrics["sharpe"]) <= 0.0:
            failures.append(f"{name}_sharpe<=0")
        if float(metrics["expectancy"]) <= 0.0:
            failures.append(f"{name}_expectancy<=0")
    if stability not in {"stable", "moderately_stable"}:
        failures.append(f"parameter_stability={stability}")
    return "PASS" if not failures else "reject: " + "; ".join(failures)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    candidates = candidate_rows()
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    blocked_rows: list[dict[str, object]] = []

    for _, candidate in candidates.iterrows():
        candidate_id = int(candidate["phase6_candidate_id"])
        pair = str(candidate["pair"])
        timeframe = str(candidate["timeframe"])
        try:
            frame, signal, mask = build_signal(candidate)
        except FileNotFoundError as exc:
            blocked_rows.append(
                {
                    "phase6_candidate_id": candidate_id,
                    "pair": pair,
                    "timeframe": timeframe,
                    "blocker": f"missing history: {exc}",
                }
            )
            continue

        fold_result_by_name: dict[str, dict[str, object]] = {}
        for fold_name, start_share, end_share in FOLDS:
            metrics = fold_metrics(frame, signal, timeframe, fold_name, fold_slice(len(frame), start_share, end_share))
            fold_result_by_name[fold_name] = metrics
            fold_rows.append(
                {
                    "phase6_candidate_id": candidate_id,
                    "pair": pair,
                    "timeframe": timeframe,
                    "strategy": candidate["strategy"],
                    "quality_gate": candidate["quality_gate"],
                    "regime": candidate["regime"],
                    "regime_behavior": candidate["regime_behavior"],
                    **metrics,
                }
            )

        train = fold_result_by_name["train"]
        validation = fold_result_by_name["validation"]
        test = fold_result_by_name["test"]
        stability = stability_label(train, validation, test)
        sample_shaped = is_sample_shaped(train, validation, test)
        reason = walk_forward_reason(train, validation, test, stability)
        summary_rows.append(
            {
                "phase6_candidate_id": candidate_id,
                "source_phase5_rank": int(candidate["rank"]),
                "pair": pair,
                "timeframe": timeframe,
                "strategy": candidate["strategy"],
                "quality_gate": candidate["quality_gate"],
                "regime": candidate["regime"],
                "regime_behavior": candidate["regime_behavior"],
                "source_phase5_trades": int(candidate["trades"]),
                "source_phase5_profit_factor": float(candidate["profit_factor"]),
                "source_phase5_sharpe": float(candidate["sharpe"]),
                "source_phase5_max_drawdown": float(candidate["max_drawdown"]),
                "regime_allowed_share": float(mask.fillna(False).mean()) if len(mask) else 0.0,
                "train_trades": int(train["trades"]),
                "validation_trades": int(validation["trades"]),
                "test_trades": int(test["trades"]),
                "train_profit_factor": float(train["profit_factor"]),
                "validation_profit_factor": float(validation["profit_factor"]),
                "test_profit_factor": float(test["profit_factor"]),
                "train_sharpe": float(train["sharpe"]),
                "validation_sharpe": float(validation["sharpe"]),
                "test_sharpe": float(test["sharpe"]),
                "train_max_drawdown": float(train["max_drawdown"]),
                "validation_max_drawdown": float(validation["max_drawdown"]),
                "test_max_drawdown": float(test["max_drawdown"]),
                "parameter_stability": stability,
                "sample_shaped": bool(sample_shaped),
                "walk_forward_reason": reason,
                "walk_forward_pass": reason == "PASS",
            }
        )

    results = pd.DataFrame(summary_rows)
    folds = pd.DataFrame(fold_rows)
    blocked = pd.DataFrame(blocked_rows)

    if not results.empty:
        effective_validation_pf = results["validation_profit_factor"].where(results["validation_trades"] >= 30, 0.0)
        effective_test_pf = results["test_profit_factor"].where(results["test_trades"] >= 30, 0.0)
        effective_test_sharpe = results["test_sharpe"].where(results["test_trades"] >= 30, -10.0)
        results["rank_score"] = (
            results["walk_forward_pass"].astype(int) * 100_000
            + effective_test_pf.fillna(0.0).clip(upper=20.0) * 1000.0
            + effective_validation_pf.fillna(0.0).clip(upper=20.0) * 300.0
            + effective_test_sharpe.fillna(-10.0).clip(lower=-10.0, upper=20.0) * 100.0
            - results["test_max_drawdown"].fillna(1.0).clip(upper=5.0) * 100.0
            - results["sample_shaped"].astype(int) * 2_500.0
            + np.minimum(results["test_trades"].fillna(0.0), 100.0)
        )
        results = results.sort_values(
            ["walk_forward_pass", "sample_shaped", "rank_score", "test_trades"], ascending=[False, True, False, False]
        ).reset_index(drop=True)
        results.insert(0, "phase6_rank", np.arange(1, len(results) + 1))

    pair_summary = (
        results.groupby("pair", dropna=False)
        .agg(
            tested_candidates=("phase6_candidate_id", "count"),
            walk_forward_passes=("walk_forward_pass", "sum"),
            best_rank=("phase6_rank", "min"),
            best_test_pf=("test_profit_factor", "max"),
            best_validation_pf=("validation_profit_factor", "max"),
            lowest_test_drawdown=("test_max_drawdown", "min"),
            max_test_trades=("test_trades", "max"),
            sample_shaped_runs=("sample_shaped", "sum"),
        )
        .reset_index()
        .sort_values(["walk_forward_passes", "best_test_pf"], ascending=[False, False])
        if not results.empty
        else pd.DataFrame()
    )

    results.to_csv(REPORTS / "phase6_walk_forward_results.csv", index=False)
    folds.to_csv(REPORTS / "phase6_walk_forward_fold_metrics.csv", index=False)
    candidates.to_csv(REPORTS / "phase6_walk_forward_candidates.csv", index=False)
    blocked.to_csv(REPORTS / "phase6_walk_forward_blockers.csv", index=False)
    pair_summary.to_csv(REPORTS / "phase6_walk_forward_pair_summary.csv", index=False)

    rank_columns = [
        "phase6_rank",
        "pair",
        "timeframe",
        "strategy",
        "quality_gate",
        "regime",
        "regime_behavior",
        "source_phase5_trades",
        "train_trades",
        "validation_trades",
        "test_trades",
        "train_profit_factor",
        "validation_profit_factor",
        "test_profit_factor",
        "train_max_drawdown",
        "validation_max_drawdown",
        "test_max_drawdown",
        "parameter_stability",
        "sample_shaped",
        "walk_forward_reason",
    ]
    report = [
        "# Evidence Pipeline Phase 6 Walk-Forward Validation",
        "",
        "Chronological walk-forward validation using train 60%, validation 20%, and test 20%. Inputs reuse Phase 5 point-in-time regime overlays and base costs only.",
        "",
        "## Scope",
        "",
        f"- Candidate rows tested: {len(results):,}",
        f"- Candidate selection: top 100+ trade Phase 5 base-cost rows, plus the best base-cost row for any missing pair.",
        f"- Walk-forward passes: {int(results['walk_forward_pass'].sum()) if len(results) else 0:,}",
        f"- Sample-shaped runs: {int(results['sample_shaped'].sum()) if len(results) else 0:,}",
        "- Passing this phase requires each fold to have at least 30 trades, PF >= 1.3, DD <= 15%, positive Sharpe, positive expectancy, and stable or moderately stable parameters.",
        "",
        "## Ranked Walk-Forward Results",
        "",
        table(results, rank_columns, 40),
        "",
        "## Pair Summary",
        "",
        table(pair_summary, list(pair_summary.columns), None) if len(pair_summary) else "_No pair summary._",
        "",
        "## Gap Analysis",
        "",
        "- No Phase 5 row had already met the PF/DD gate, so Phase 6 is validating research candidates rather than promotion candidates.",
        "- Validation and test folds expose whether the apparent edge survives time ordering; failures here block cost stress promotion.",
        "- DOGE-XRP remains coverage-limited versus the other pairs because several XRP local histories are still missing from Phase 2.",
        "- Funding is still neutral in the local histories, so funding-aware acceptance must wait for enriched data.",
        "",
        "## Premortem",
        "",
        "- The most likely failure mode is a signal that works in the train slice but loses PF in validation or test.",
        "- The second likely failure mode is drawdown expanding beyond 15% once the trade path is split chronologically.",
        "- Low validation or test trade counts can make a high PF look attractive while still being too fragile for paper trading.",
        "",
        "## Red Team",
        "",
        "- Treat any high test PF with fewer than 30 fold trades as a false positive until proven otherwise.",
        "- Treat stable-looking PF with high drawdown as an execution and sizing failure, not as a near miss.",
        "- Do not advance any row to cost stress unless it passes this report without sample-shaped warnings.",
        "",
        "## Takeaway",
        "",
    ]
    if len(results) and int(results["walk_forward_pass"].sum()):
        report.append("At least one candidate passed walk-forward and can proceed to Phase 7 cost stress. It is still not deployable until stress tests pass.")
    else:
        report.append("No candidate passed walk-forward. These strategies remain research-only and should not be paper traded from the current evidence.")
    report.extend(
        [
            "",
            "## Files",
            "",
            "- phase6_walk_forward_results.csv",
            "- phase6_walk_forward_fold_metrics.csv",
            "- phase6_walk_forward_candidates.csv",
            "- phase6_walk_forward_pair_summary.csv",
            "- phase6_walk_forward_blockers.csv",
        ]
    )
    (REPORTS / "phase6_walk_forward_report.md").write_text("\n".join(report) + "\n")

    print(
        json.dumps(
            {
                "candidates": int(len(candidates)),
                "tested": int(len(results)),
                "walk_forward_passes": int(results["walk_forward_pass"].sum()) if len(results) else 0,
                "sample_shaped": int(results["sample_shaped"].sum()) if len(results) else 0,
                "report": str(REPORTS / "phase6_walk_forward_report.md"),
                "best": results.iloc[0][rank_columns[1:]].to_dict() if len(results) else None,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
