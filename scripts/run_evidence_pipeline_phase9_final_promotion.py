from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "evidence_pipeline"

PAIRS = ("ETH-SOL", "BTC-DOGE", "BTC-SOL", "ETH-LINK", "DOGE-XRP")


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


def load_csv(name: str) -> pd.DataFrame:
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def final_label(pair: str, best: pd.Series | None, stress_pf: float | None, stress_pass: bool) -> tuple[str, str, str]:
    if best is None:
        return "reject", "No tested row available.", "Restore coverage and rerun baseline."
    if bool(best.get("walk_forward_pass", False)) and stress_pass and stress_pf is not None and stress_pf >= 1.3:
        return "paper_trade_candidate", "Passed walk-forward and cost stress; still requires operational review.", "Prepare paper-trade runbook."
    if bool(best.get("walk_forward_pass", False)):
        return "watchlist", "Passed walk-forward but has not passed cost stress.", "Run and pass Phase 7 stress."
    if pair == "BTC-SOL" and float(best["profit_factor"]) >= 1.3 and float(best["test_profit_factor"]) >= 1.2:
        return "watchlist", "Closest repair clue, but DD/PF gates still fail.", "Focus on drawdown repair and validation PF."
    if pair == "DOGE-XRP":
        return "reject", "Only 5m coverage is available and tested logic is weak.", "Fetch missing XRP histories before more strategy work."
    return "research_only", "No row passed walk-forward and stress was not eligible.", "Redesign entries/exits or enrich data before retesting."


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    phase8 = load_csv("phase8_pair_specific_ranked.csv")
    phase7 = load_csv("phase7_cost_stress_results.csv")
    missing = load_csv("phase8_pair_specific_missing_coverage.csv")

    rows: list[dict[str, object]] = []
    for pair in PAIRS:
        pair_rows = phase8[phase8["pair"].eq(pair)].copy() if not phase8.empty else pd.DataFrame()
        best = pair_rows.sort_values("phase8_rank").iloc[0] if not pair_rows.empty else None
        stress_rows = phase7[phase7["pair"].eq(pair)].copy() if not phase7.empty and "pair" in phase7.columns else pd.DataFrame()
        stress_pf = None
        stress_pass = False
        if not stress_rows.empty:
            stress_pf = float(stress_rows.loc[stress_rows["cost_bucket"].ne("base"), "profit_factor"].min())
            stress_pass = bool(stress_rows["stress_pass"].all())
        label, reason, next_action = final_label(pair, best, stress_pf, stress_pass)
        pair_missing = missing[missing["pair"].eq(pair)] if not missing.empty else pd.DataFrame()
        rows.append(
            {
                "pair": pair,
                "final_label": label,
                "best_strategy": "" if best is None else best["strategy"],
                "best_timeframe": "" if best is None else best["timeframe"],
                "best_regime_overlay": "" if best is None else best["regime"],
                "best_entry_style": "" if best is None else best["entry_style"],
                "best_exit_style": "" if best is None else best["exit_style"],
                "trades": 0 if best is None else int(best["trades"]),
                "profit_factor": 0.0 if best is None else float(best["profit_factor"]),
                "sharpe": 0.0 if best is None else float(best["sharpe"]),
                "max_drawdown": 0.0 if best is None else float(best["max_drawdown"]),
                "stress_pf": "" if stress_pf is None else stress_pf,
                "walk_forward_test_pf": 0.0 if best is None else float(best["test_profit_factor"]),
                "walk_forward_pass": False if best is None else bool(best["walk_forward_pass"]),
                "cost_stress_pass": stress_pass,
                "missing_required_timeframes": ", ".join(pair_missing["timeframe"].astype(str).tolist())
                if not pair_missing.empty
                else "",
                "reason_for_label": reason,
                "next_action": next_action,
            }
        )

    results = pd.DataFrame(rows)
    label_order = {
        "production_candidate": 0,
        "paper_trade_candidate": 1,
        "watchlist": 2,
        "research_only": 3,
        "reject": 4,
    }
    results["label_order"] = results["final_label"].map(label_order).fillna(9)
    results = results.sort_values(["label_order", "pair"]).drop(columns=["label_order"]).reset_index(drop=True)
    results.to_csv(REPORTS / "phase9_final_promotion_labels.csv", index=False)

    columns = [
        "pair",
        "final_label",
        "best_strategy",
        "best_timeframe",
        "best_regime_overlay",
        "best_entry_style",
        "best_exit_style",
        "trades",
        "profit_factor",
        "sharpe",
        "max_drawdown",
        "stress_pf",
        "walk_forward_test_pf",
        "missing_required_timeframes",
        "reason_for_label",
        "next_action",
    ]
    report = [
        "# Evidence Pipeline Phase 9 Final Promotion Report",
        "",
        "Final labels are evidence-gated. No strategy is marked deployable unless it passes walk-forward and cost stress.",
        "",
        "## Final Labels",
        "",
        table(results, columns, None),
        "",
        "## Gate Summary",
        "",
        f"- Production candidates: {int(results['final_label'].eq('production_candidate').sum())}",
        f"- Paper-trade candidates: {int(results['final_label'].eq('paper_trade_candidate').sum())}",
        f"- Watchlist: {int(results['final_label'].eq('watchlist').sum())}",
        f"- Research-only: {int(results['final_label'].eq('research_only').sum())}",
        f"- Reject: {int(results['final_label'].eq('reject').sum())}",
        "",
        "## Gap Analysis",
        "",
        "- The strongest current outputs are research clues, not tradable candidates.",
        "- No row passed walk-forward, so Phase 7 had no eligible candidates to stress-test.",
        "- DOGE-XRP still lacks required 15m, 1h, 4h, and 1d processed histories.",
        "- Funding remains a placeholder in the local data, so funding-aware acceptance is still incomplete.",
        "",
        "## Premortem",
        "",
        "- Promoting now would likely fail because validation/test PF decays and drawdown remains too high.",
        "- The nearest watchlist idea, BTC-SOL 4h, can still fail if drawdown repair reduces trade count too much.",
        "- BTC-DOGE can look attractive in late test slices while remaining dangerous across the full path.",
        "",
        "## Red Team",
        "",
        "- Do not promote ETH-SOL's high PF rows because the broad evidence says those are sample-shaped or unstable.",
        "- Do not promote BTC-DOGE until full-period drawdown is controlled before cost stress.",
        "- Do not continue DOGE-XRP strategy tuning until missing coverage is fixed.",
        "",
        "## Files",
        "",
        "- phase9_final_promotion_labels.csv",
    ]
    (REPORTS / "phase9_final_promotion_report.md").write_text("\n".join(report) + "\n")

    print(
        json.dumps(
            {
                "labels": results["final_label"].value_counts().to_dict(),
                "report": str(REPORTS / "phase9_final_promotion_report.md"),
                "csv": str(REPORTS / "phase9_final_promotion_labels.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
