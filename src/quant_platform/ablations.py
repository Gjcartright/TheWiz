from __future__ import annotations

from dataclasses import dataclass
from math import isinf
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AblationSpec:
    name: str
    tested_strategy_id: int
    baseline_strategy_id: int
    tested_component: str
    hypothesis: str


DEFAULT_ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec("zscore_plus_ecm_vs_zscore", 2, 1, "ecm", "ECM confirmation improves classic z-score mean reversion."),
    AblationSpec("zscore_plus_copula_vs_zscore", 3, 1, "copula", "Copula confirmation improves classic z-score mean reversion."),
    AblationSpec("ecm_copula_zscore_vs_zscore", 4, 1, "ecm+copula", "ECM and copula together improve classic z-score."),
    AblationSpec("ecm_copula_zscore_vs_zscore_copula", 4, 3, "ecm", "ECM adds incremental value beyond z-score plus copula."),
    AblationSpec("pure_copula_vs_zscore", 5, 1, "copula_only", "Copula-only dislocation can outperform z-score."),
    AblationSpec("dual_conditional_copula_vs_pure_copula", 6, 5, "dual_conditional_copula", "Dual conditional copula improves pure copula."),
    AblationSpec("tail_event_reversion_vs_pure_copula", 7, 5, "tail_event_filter", "Tail-event filtering improves copula entries."),
    AblationSpec("pure_ecm_vs_zscore", 8, 1, "ecm_only", "ECM-only signals can outperform z-score."),
    AblationSpec("ecm_leadership_vs_pure_ecm", 9, 8, "ecm_leadership", "ECM leadership improves pure ECM."),
    AblationSpec("half_life_optimized_vs_zscore", 11, 1, "half_life", "Half-life thresholds improve z-score."),
    AblationSpec("hurst_filter_vs_zscore", 12, 1, "hurst", "Hurst filtering improves z-score."),
    AblationSpec("hurst_half_life_vs_zscore", 13, 1, "hurst+half_life", "Hurst plus half-life improves z-score."),
    AblationSpec("ou_optimal_vs_zscore", 14, 1, "ou_optimal", "OU optimal filtering improves z-score."),
    AblationSpec("proprietary_stack_vs_zscore", 17, 1, "proprietary_stack", "Proprietary confidence stack improves z-score."),
    AblationSpec("dynamic_threshold_vs_zscore", 20, 1, "dynamic_threshold", "Dynamic thresholds improve static z-score."),
    AblationSpec("regime_filtered_vs_zscore", 24, 1, "regime_filter", "Regime filtering improves z-score."),
    AblationSpec("copula_risk_filter_vs_pure_copula", 29, 5, "copula_tail_risk_filter", "Copula tail-risk filtering improves pure copula."),
    AblationSpec("copula_dislocation_ranking_vs_pure_copula", 33, 5, "copula_dislocation_ranking", "Lower copula threshold improves pure copula."),
    AblationSpec("copula_regime_vs_pure_copula", 34, 5, "copula_regime_filter", "Regime filtering improves pure copula."),
    AblationSpec("copula_persistence_vs_pure_copula", 35, 5, "copula_persistence", "Copula persistence improves pure copula."),
    AblationSpec("copula_ecm_vs_pure_copula", 36, 5, "ecm", "ECM filtering improves pure copula."),
)


def _finite_metric(value: float, cap: float = 10.0) -> float:
    if pd.isna(value):
        return 0.0
    if isinf(float(value)):
        return cap
    return float(value)


def _delta(tested: float, baseline: float, higher_is_better: bool = True) -> float:
    raw = _finite_metric(tested) - _finite_metric(baseline)
    return raw if higher_is_better else -raw


def ablation_report(
    results: pd.DataFrame,
    specs: tuple[AblationSpec, ...] = DEFAULT_ABLATIONS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if results.empty:
        return pd.DataFrame()

    evaluated = results[results["status"] == "evaluated"].copy()
    for spec in specs:
        tested = evaluated[evaluated["strategy_id"] == spec.tested_strategy_id]
        baseline = evaluated[evaluated["strategy_id"] == spec.baseline_strategy_id]
        tested_name = _strategy_name(results, spec.tested_strategy_id)
        baseline_name = _strategy_name(results, spec.baseline_strategy_id)
        if tested.empty or baseline.empty:
            rows.append(
                _missing_row(
                    spec,
                    tested_name,
                    baseline_name,
                    "missing_tested_strategy" if tested.empty else "missing_baseline_strategy",
                )
            )
            continue

        merged = tested.merge(
            baseline,
            on=["pair", "regime", "cost_bucket"],
            suffixes=("_tested", "_baseline"),
        )
        if merged.empty:
            rows.append(_missing_row(spec, tested_name, baseline_name, "no_matched_pair_regime_cost_rows"))
            continue

        comparison_rows = []
        for _, row in merged.iterrows():
            pf_delta = _delta(row["profit_factor_tested"], row["profit_factor_baseline"])
            sharpe_delta = _delta(row["sharpe_tested"], row["sharpe_baseline"])
            expectancy_delta = _delta(row["expectancy_tested"], row["expectancy_baseline"])
            drawdown_delta = _delta(row["max_drawdown_tested"], row["max_drawdown_baseline"], higher_is_better=False)
            win_rate_delta = _delta(row["win_rate_tested"], row["win_rate_baseline"])
            trade_delta = _finite_metric(row["trades_tested"]) - _finite_metric(row["trades_baseline"])
            comparison_rows.append(
                {
                    "pf_delta": pf_delta,
                    "sharpe_delta": sharpe_delta,
                    "expectancy_delta": expectancy_delta,
                    "drawdown_delta": drawdown_delta,
                    "win_rate_delta": win_rate_delta,
                    "trade_delta": trade_delta,
                    "tested_eligible": bool(row["eligible_tested"]),
                    "baseline_eligible": bool(row["eligible_baseline"]),
                }
            )
        comparison = pd.DataFrame(comparison_rows)
        score = (
            comparison["pf_delta"].median() * 0.35
            + comparison["sharpe_delta"].median() * 0.25
            + comparison["expectancy_delta"].median() * 10.0
            + comparison["drawdown_delta"].median() * 2.0
            + comparison["win_rate_delta"].median() * 0.10
        )
        rows.append(
            {
                "ablation": spec.name,
                "tested_component": spec.tested_component,
                "tested_strategy_id": spec.tested_strategy_id,
                "tested_strategy_name": tested_name,
                "baseline_strategy_id": spec.baseline_strategy_id,
                "baseline_strategy_name": baseline_name,
                "hypothesis": spec.hypothesis,
                "status": "evaluated",
                "matched_runs": int(len(merged)),
                "tested_eligible_runs": int(comparison["tested_eligible"].sum()),
                "baseline_eligible_runs": int(comparison["baseline_eligible"].sum()),
                "median_pf_delta": float(comparison["pf_delta"].median()),
                "median_sharpe_delta": float(comparison["sharpe_delta"].median()),
                "median_expectancy_delta": float(comparison["expectancy_delta"].median()),
                "median_drawdown_improvement": float(comparison["drawdown_delta"].median()),
                "median_win_rate_delta": float(comparison["win_rate_delta"].median()),
                "median_trade_delta": float(comparison["trade_delta"].median()),
                "incremental_score": float(score),
                "conclusion": _conclusion(float(score), comparison["pf_delta"], comparison["drawdown_delta"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "incremental_score", "matched_runs"], ascending=[True, False, False])


def _strategy_name(results: pd.DataFrame, strategy_id: int) -> str:
    names = results.loc[results["strategy_id"] == strategy_id, "strategy_name"].dropna().unique()
    return str(names[0]) if len(names) else ""


def _missing_row(spec: AblationSpec, tested_name: str, baseline_name: str, reason: str) -> dict[str, object]:
    return {
        "ablation": spec.name,
        "tested_component": spec.tested_component,
        "tested_strategy_id": spec.tested_strategy_id,
        "tested_strategy_name": tested_name,
        "baseline_strategy_id": spec.baseline_strategy_id,
        "baseline_strategy_name": baseline_name,
        "hypothesis": spec.hypothesis,
        "status": reason,
        "matched_runs": 0,
        "tested_eligible_runs": 0,
        "baseline_eligible_runs": 0,
        "median_pf_delta": 0.0,
        "median_sharpe_delta": 0.0,
        "median_expectancy_delta": 0.0,
        "median_drawdown_improvement": 0.0,
        "median_win_rate_delta": 0.0,
        "median_trade_delta": 0.0,
        "incremental_score": -1_000_000.0,
        "conclusion": reason,
    }


def _conclusion(score: float, pf_delta: pd.Series, drawdown_delta: pd.Series) -> str:
    if score > 0 and pf_delta.median() >= 0 and drawdown_delta.median() >= 0:
        return "adds_value"
    if score > 0:
        return "mixed_positive"
    if score < 0:
        return "hurts"
    return "neutral"


def write_ablation_report(
    results: pd.DataFrame,
    output_path: str | Path,
    specs: tuple[AblationSpec, ...] = DEFAULT_ABLATIONS,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ablation_report(results, specs).to_csv(output, index=False)
    return output

