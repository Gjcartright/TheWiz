from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_evidence_pipeline_phase3_quality as phase3
import run_evidence_pipeline_phase8_pair_specific as phase8


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "evidence_pipeline"
PHASE8_RANKED = REPORTS / "phase8_pair_specific_ranked.csv"

COST_STRESS_BUCKETS = {
    "base": phase3.COST_BUCKETS["base"],
    "stress": phase3.COST_BUCKETS["stress"],
    "ugly_fill": {
        "fee_bps": 10.0,
        "slippage_bps": 20.0,
        "execution_risk_bps": 8.0,
        "funding_bps_per_day": 5.0,
    },
    "funding_penalty": {
        "fee_bps": 5.0,
        "slippage_bps": 4.0,
        "execution_risk_bps": 2.0,
        "funding_bps_per_day": 12.0,
    },
    "slippage_shock": {
        "fee_bps": 5.0,
        "slippage_bps": 25.0,
        "execution_risk_bps": 8.0,
        "funding_bps_per_day": 3.0,
    },
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


def stress_reason(metrics: dict[str, float], cost_bucket: str) -> str:
    failures: list[str] = []
    if cost_bucket == "base" and metrics["profit_factor"] < 1.8:
        failures.append("base_profit_factor<1.8")
    if cost_bucket != "base" and metrics["profit_factor"] < 1.3:
        failures.append(f"{cost_bucket}_profit_factor<1.3")
    if metrics["trades"] < 100:
        failures.append("trades<100")
    if metrics["max_drawdown"] > 0.15:
        failures.append("max_drawdown>0.15")
    if metrics["sharpe"] <= 0:
        failures.append("sharpe<=0")
    if metrics["expectancy"] <= 0:
        failures.append("expectancy<=0")
    return "PASS" if not failures else "reject: " + "; ".join(failures)


def eligible_candidates() -> pd.DataFrame:
    if not PHASE8_RANKED.exists():
        return pd.DataFrame()
    ranked = pd.read_csv(PHASE8_RANKED)
    return ranked[ranked["walk_forward_pass"].astype(bool)].copy()


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    candidates = eligible_candidates()
    rows: list[dict[str, object]] = []

    for _, candidate in candidates.iterrows():
        frame, signal, _ = phase8.build_pair_specific_signal(
            str(candidate["pair"]),
            str(candidate["timeframe"]),
            str(candidate["strategy"]),
            str(candidate["quality_gate"]),
            str(candidate["regime"]),
            str(candidate["regime_behavior"]),
            str(candidate["entry_style"]),
            str(candidate["exit_style"]),
        )
        for bucket, costs in COST_STRESS_BUCKETS.items():
            metrics = phase3.backtest(
                frame,
                signal,
                phase3.TIMEFRAME_PERIODS.get(str(candidate["timeframe"]), 365),
                costs,
            )
            rows.append(
                {
                    "source_phase8_rank": int(candidate["phase8_rank"]),
                    "pair": candidate["pair"],
                    "timeframe": candidate["timeframe"],
                    "strategy": candidate["strategy"],
                    "quality_gate": candidate["quality_gate"],
                    "regime": candidate["regime"],
                    "regime_behavior": candidate["regime_behavior"],
                    "entry_style": candidate["entry_style"],
                    "exit_style": candidate["exit_style"],
                    "cost_bucket": bucket,
                    **metrics,
                    "stress_reason": stress_reason(metrics, bucket),
                    "stress_pass": stress_reason(metrics, bucket) == "PASS",
                }
            )

    results = pd.DataFrame(rows)
    if not results.empty:
        results["rank_score"] = (
            results["stress_pass"].astype(int) * 100_000
            + results["profit_factor"].fillna(0.0).clip(upper=20.0) * 1000.0
            + results["sharpe"].fillna(0.0).clip(lower=-10.0, upper=20.0) * 100.0
            - results["max_drawdown"].fillna(1.0).clip(upper=5.0) * 100.0
        )
        results = results.sort_values(["stress_pass", "rank_score"], ascending=[False, False]).reset_index(drop=True)
        results.insert(0, "phase7_rank", np.arange(1, len(results) + 1))
    else:
        results = pd.DataFrame(
            columns=[
                "phase7_rank",
                "source_phase8_rank",
                "pair",
                "timeframe",
                "strategy",
                "quality_gate",
                "regime",
                "regime_behavior",
                "entry_style",
                "exit_style",
                "cost_bucket",
                "trades",
                "profit_factor",
                "sharpe",
                "max_drawdown",
                "stress_reason",
                "stress_pass",
            ]
        )

    results.to_csv(REPORTS / "phase7_cost_stress_results.csv", index=False)
    candidates.to_csv(REPORTS / "phase7_cost_stress_eligible_candidates.csv", index=False)

    columns = [
        "phase7_rank",
        "pair",
        "timeframe",
        "strategy",
        "regime",
        "entry_style",
        "exit_style",
        "cost_bucket",
        "trades",
        "profit_factor",
        "sharpe",
        "max_drawdown",
        "stress_reason",
    ]
    report = [
        "# Evidence Pipeline Phase 7 Cost Stress",
        "",
        "Cost stress is gated by walk-forward. Only candidates that passed Phase 6 or Phase 8 walk-forward are eligible.",
        "",
        "## Scope",
        "",
        f"- Eligible walk-forward candidates: {len(candidates):,}",
        f"- Cost-stress rows run: {len(results):,}",
        f"- Cost-stress passes: {int(results['stress_pass'].sum()) if len(results) else 0:,}",
        "- Stress buckets: base, stress, ugly_fill, funding_penalty, slippage_shock.",
        "",
        "## Results",
        "",
        table(results, columns, 50),
        "",
        "## Gate Decision",
        "",
    ]
    if len(candidates):
        report.append("At least one walk-forward candidate was stress-tested. Paper-trade eligibility still depends on stress pass evidence.")
    else:
        report.append("No candidates were eligible for cost stress because no Phase 6 or Phase 8 row passed walk-forward. This blocks paper-trade promotion.")
    report.extend(
        [
            "",
            "## Gap Analysis",
            "",
            "- Cost stress cannot rescue a strategy that has already failed walk-forward.",
            "- Funding-specific stress remains approximate until real funding fields are added to local histories.",
            "",
            "## Premortem",
            "",
            "- If a future row barely passes walk-forward, ugly fills and funding penalty are the most likely next failure points.",
            "- Any strategy that only survives base costs should stay out of paper trading.",
            "",
            "## Red Team",
            "",
            "- Do not run stress on rejected walk-forward rows and reinterpret a good stress bucket as promotion evidence.",
            "- Require related-regime or related-timeframe survival before upgrading any future candidate.",
            "",
            "## Files",
            "",
            "- phase7_cost_stress_results.csv",
            "- phase7_cost_stress_eligible_candidates.csv",
        ]
    )
    (REPORTS / "phase7_cost_stress_report.md").write_text("\n".join(report) + "\n")

    print(
        json.dumps(
            {
                "eligible_candidates": int(len(candidates)),
                "rows": int(len(results)),
                "passes": int(results["stress_pass"].sum()) if len(results) else 0,
                "report": str(REPORTS / "phase7_cost_stress_report.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
