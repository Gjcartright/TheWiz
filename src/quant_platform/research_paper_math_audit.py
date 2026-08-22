from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

from datetime import date
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import ROOT, CommandResult


def run_paper_critical_checks(*, root: Path = ROOT) -> CommandResult:
    report_base = root / "reports" / "research" / "papers"
    inventory_path = report_base / "paper_inventory.csv"
    if not inventory_path.exists():
        raise FileNotFoundError("paper inventory missing; run ingest-paper-library first")
    inventory = pd.read_csv(inventory_path).fillna("")
    known = set(inventory["paper_id"].astype(str))
    required = {
        "paper_f0b8fced125866b5",
        "paper_8ee8ed6d01c3aa01",
        "paper_bd860d3475116e96",
        "paper_83bedd65a3817a87",
    }
    missing = sorted(required - known)
    if missing:
        raise ValueError(f"critical paper checks missing inventory sources: {missing}")

    discovery_start = date(2020, 9, 1)
    discovery_end = date(2023, 1, 31)
    test_start = date(2017, 4, 24)
    test_end = date(2022, 12, 22)
    overlap_start = max(discovery_start, test_start)
    overlap_end = min(discovery_end, test_end)
    overlap_days = max(0, (overlap_end - overlap_start).days + 1)

    rows = [
        {
            "check_id": "credit_sharpe_denominator",
            "paper_id": "paper_f0b8fced125866b5",
            "check_type": "formula_reconciliation",
            "paper_evidence": "Page 6 defines Sharpe as (E(R)-Rf)/V(R); page 5 identifies V as variance.",
            "expected": "Sharpe=(E(R)-Rf)/sqrt(V(R)), followed by consistent annualization.",
            "observed": "Variance denominator rather than standard-deviation denominator.",
            "status": "fail",
            "severity": "critical",
            "promotion_blocker": True,
            "required_action": "Recompute all reported Sharpe values from underlying returns before comparison.",
        },
        {
            "check_id": "clustering_rl_discovery_test_overlap",
            "paper_id": "paper_8ee8ed6d01c3aa01",
            "check_type": "temporal_lineage",
            "paper_evidence": "Page 5 trains the CAE from September 2020 to January 2023 and tests the RL policy from April 2017 to December 2022.",
            "expected": "Every discovery representation must end before its associated test period begins.",
            "observed": f"Discovery and test intervals overlap by {overlap_days} calendar days.",
            "status": "fail",
            "severity": "critical",
            "promotion_blocker": True,
            "required_action": "Quarantine all reported performance and rebuild pair discovery inside each causal fold.",
        },
        {
            "check_id": "deep_pairs_execution_cost_parity",
            "paper_id": "paper_bd860d3475116e96",
            "check_type": "execution_assumption",
            "paper_evidence": "Pages 19-22 test proportional costs but explicitly state that bid-ask spread is omitted.",
            "expected": "Two-leg fees, spread, slippage, funding, asynchronous fills, and margin must be represented.",
            "observed": "Useful cost sensitivity without venue-level execution parity.",
            "status": "blocked",
            "severity": "critical",
            "promotion_blocker": True,
            "required_action": "Reproduce with observed Hyperliquid L2 and funding evidence.",
        },
        {
            "check_id": "fcvar_break_point_in_time",
            "paper_id": "paper_83bedd65a3817a87",
            "check_type": "point_in_time_reproducibility",
            "paper_evidence": "The paper combines full-sample FCVAR estimation and Bai-Perron break dating over 2008-2024.",
            "expected": "Live features must use one-sided estimation and include detection delay.",
            "observed": "Published regime dates are descriptive and cannot be treated as decision-time labels.",
            "status": "blocked",
            "severity": "critical",
            "promotion_blocker": True,
            "required_action": "Implement sequential break detection before any strategy test.",
        },
    ]
    report = pd.DataFrame(rows)
    csv_path = report_base / "paper_math_reconciliation.csv"
    md_path = report_base / "paper_math_reconciliation.md"
    atomic_write_csv(report, csv_path, index=False)
    atomic_write_text(md_path, _markdown(report), encoding="utf-8")
    return CommandResult(
        paths={"paper_math_reconciliation": csv_path, "paper_math_reconciliation_md": md_path},
        summary={
            "checks": len(report),
            "failed": int(report["status"].eq("fail").sum()),
            "blocked": int(report["status"].eq("blocked").sum()),
            "critical_blockers": int(
                (report["severity"].eq("critical") & report["promotion_blocker"]).sum()
            ),
            "discovery_test_overlap_days": overlap_days,
            "execution_eligible": 0,
        },
    )


def _markdown(report: pd.DataFrame) -> str:
    lines = [
        "# Paper Math Reconciliation",
        "",
        "These are deterministic critical checks, not a complete reproduction of every equation.",
        "",
    ]
    for _, row in report.iterrows():
        lines.extend(
            [
                f"## {row['check_id']}",
                "",
                f"- paper: `{row['paper_id']}`",
                f"- status: `{row['status']}`",
                f"- severity: `{row['severity']}`",
                f"- evidence: {row['paper_evidence']}",
                f"- expected: {row['expected']}",
                f"- observed: {row['observed']}",
                f"- required action: {row['required_action']}",
                "",
            ]
        )
    lines.append("No checked paper has signal or execution authority.")
    return "\n".join(lines) + "\n"
