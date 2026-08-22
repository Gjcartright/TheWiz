"""Supreme Team checkpoint for the dynamic-agent control plane.

The report is a concise, evidence-linked gap analysis, pre-mortem, post-mortem,
and red-team review. It does not create strategy evidence or alter routing.
"""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_platform.orchestration.copula_shadow_comparison import EVENTS_FILENAME
from quant_platform.orchestration.dynamic_rollout import build_dynamic_rollout_gate


ROOT = Path(__file__).resolve().parents[3]


def build_dynamic_supreme_team_checkpoint(*, root: Path = ROOT, now: datetime | None = None) -> dict[str, Path | int]:
    """Write an evidence-linked Supreme Team checkpoint for dynamic rollout."""

    now = now or datetime.now(timezone.utc)
    directory = root / "reports" / "orchestration" / "dynamic_agents"
    rollout_path = directory / "strategy_cell_rollout.csv"
    comparison_path = directory / EVENTS_FILENAME
    rollout_result = build_dynamic_rollout_gate(root=root, now=now)
    rollout = _read_csv(rollout_path)
    qualifying = int(rollout_result["qualifying_agreements"])
    rollout_blocked = rollout.empty or (rollout["rollout_status"].astype(str).str.contains("BLOCKED|CONTINUE").all())
    timestamp = now.strftime("%Y-%m-%d_%H%M%SZ")
    output = directory / f"supreme_team_dynamic_control_{timestamp}.csv"

    rows = [
        _row(
            "gap_analysis",
            "Copula rollout evidence is incomplete" if rollout_blocked else "Copula shadow threshold is met",
            "critical" if rollout_blocked else "low",
            "Collect distinct, fresh, complete Copula comparisons or start one independent shadow cell only.",
            comparison_path,
            root,
        ),
        _row(
            "pre_mortem",
            "Repeated agreement could be driven by one pair, stale snapshot, or identical replay configuration.",
            "high",
            "Require distinct candidates, fresh point-in-time captures, and diversify configuration before broader rollout.",
            comparison_path,
            root,
        ),
        _row(
            "post_mortem",
            f"Observed {qualifying} qualifying Copula shadow comparisons; rollout state is {'blocked' if rollout_blocked else 'eligible for shadow only'}.",
            "medium" if rollout_blocked else "low",
            "Keep decisions non-authoritative and retain evidence, veto, and comparison records for every rerun.",
            rollout_path,
            root,
        ),
        _row(
            "red_team",
            "A passing cell could be mistaken for execution approval or hide a safety veto in a separate artifact.",
            "critical",
            "Enforce false promotion/paper flags in the rollout report and require independent venue, cost, and safety gates later.",
            rollout_path,
            root,
        ),
    ]
    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(frame, output, index=False)
    markdown = output.with_suffix(".md")
    atomic_write_text(markdown, _markdown(frame, qualifying, rollout_blocked), encoding="utf-8")
    return {"csv": output, "markdown": markdown, "qualifying_agreements": qualifying}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _row(lens: str, finding: str, severity: str, next_step: str, evidence_path: Path, root: Path) -> dict[str, str]:
    return {
        "lens": lens,
        "finding": finding,
        "severity": severity,
        "required_action": next_step,
        "evidence_path": str(evidence_path.relative_to(root)) if evidence_path.exists() else "",
    }


def _markdown(frame: pd.DataFrame, qualifying: int, rollout_blocked: bool) -> str:
    state = "BLOCKED" if rollout_blocked else "SHADOW-ONLY ELIGIBLE"
    return "\n".join(
        [
            "# Dynamic Control Supreme Team Checkpoint",
            "",
            f"Rollout state: **{state}**. Qualifying Copula agreements: **{qualifying}**.",
            "No output in this checkpoint grants paper or live execution.",
            "",
            frame.to_markdown(index=False),
            "",
        ]
    )
