"""Evidence-gated expansion policy for dynamic strategy cells.

This module is deliberately conservative: qualifying Copula shadow comparisons
may make another strategy cell eligible for its own shadow trial. They never
authorize a paper or live route.
"""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_platform.orchestration.copula_shadow_comparison import EVENTS_FILENAME, load_copula_comparison_events
from quant_platform.orchestration.dynamic_arbiter import validate_copula_evidence
from quant_platform.orchestration.dynamic_ledger import DynamicAgentLedger
from quant_platform.orchestration.dynamic_router import COPULA_FRESHNESS_SECONDS

ROOT = Path(__file__).resolve().parents[3]
COPULA_SHADOW_REQUIRED_AGREEMENTS = 5
STRATEGY_FAMILIES: tuple[str, ...] = (
    "Static Spread",
    "Static ZScoreR",
    "Dyn Spread",
    "Dyn ZScoreR",
    "OU Spread",
    "OU ZScoreR",
    "Copula",
)


def build_dynamic_rollout_gate(
    *,
    root: Path = ROOT,
    required_agreements: int = COPULA_SHADOW_REQUIRED_AGREEMENTS,
    now: datetime | None = None,
) -> dict[str, Path | int | str]:
    """Report which strategy cells may enter shadow mode, if any."""

    if required_agreements < 1:
        raise ValueError("required_agreements must be at least one")

    now = now or datetime.now(timezone.utc)
    directory = root / "reports" / "orchestration" / "dynamic_agents"
    comparison_path = directory / EVENTS_FILENAME
    comparisons = load_copula_comparison_events(root)
    qualifying = _qualifying_agreements(comparisons, root=root, now=now)
    copula_ready = len(qualifying) >= required_agreements
    blockers = _copula_blockers(comparisons, qualifying, required_agreements)

    rows: list[dict[str, object]] = []
    for strategy_family in STRATEGY_FAMILIES:
        if strategy_family == "Copula":
            status = "ELIGIBLE_FOR_NEXT_SHADOW_CELL" if copula_ready else "COPULA_SHADOW_CONTINUE"
            reason = "copula_shadow_agreement_gate_passed" if copula_ready else ";".join(blockers)
        else:
            status = "ELIGIBLE_FOR_SHADOW_ONLY" if copula_ready else "BLOCKED_PENDING_COPULA_SHADOW_ACCEPTANCE"
            reason = (
                "copula_shadow_gate_passed; cell still requires independent shadow validation"
                if copula_ready
                else ";".join(blockers)
            )
        rows.append(
            {
                "strategy_family": strategy_family,
                "phase": "shadow",
                "rollout_status": status,
                "promotion_allowed": False,
                "paper_execution_allowed": False,
                "required_copula_agreements": required_agreements,
                "qualifying_copula_agreements": len(qualifying),
                "reason": reason,
                "evidence_path": _relative_or_empty(comparison_path, root),
                "next_step": _next_step(strategy_family, copula_ready),
            }
        )

    output = directory / "strategy_cell_rollout.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    atomic_write_csv(frame, output, index=False)
    markdown = output.with_suffix(".md")
    atomic_write_text(markdown, _markdown(frame), encoding="utf-8")
    return {
        "csv": output,
        "markdown": markdown,
        "copula_rollout_status": "ready" if copula_ready else "blocked",
        "qualifying_agreements": len(qualifying),
    }


def _qualifying_agreements(frame: pd.DataFrame, *, root: Path, now: datetime) -> pd.DataFrame:
    required = {
        "comparison",
        "shadow_only",
        "dynamic_veto_count",
        "dynamic_evidence_count",
        "sequential_decision",
        "dynamic_decision",
        "candidate_id",
        "configuration_id",
        "source_snapshot_id",
        "source_timestamp",
    }
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    working = frame.copy()
    working["source_timestamp"] = pd.to_datetime(working["source_timestamp"], utc=True, errors="coerce")
    fresh_after = pd.Timestamp(now - pd.Timedelta(seconds=COPULA_FRESHNESS_SECONDS))
    qualified = working.loc[
        (frame["comparison"] == "exact_agreement")
        & (frame["shadow_only"].astype(str).str.lower() == "true")
        & (pd.to_numeric(frame["dynamic_veto_count"], errors="coerce").fillna(-1) == 0)
        & (pd.to_numeric(frame["dynamic_evidence_count"], errors="coerce").fillna(0) >= 5)
        & (frame["sequential_decision"] == "TEST")
        & (frame["dynamic_decision"] == "TEST")
        & working["source_timestamp"].notna()
        & (working["source_timestamp"] >= fresh_after)
        & (working["source_timestamp"] <= pd.Timestamp(now))
    ].copy()
    evidence_by_id = {packet.packet_id: packet for packet in DynamicAgentLedger(root).read_evidence()}
    verified_rows: list[int] = []
    for index, row in qualified.iterrows():
        packet_ids = tuple(row["evidence_packet_ids"])
        packets = tuple(evidence_by_id[packet_id] for packet_id in packet_ids if packet_id in evidence_by_id)
        if len(packets) != 5:
            continue
        if validate_copula_evidence(str(row["candidate_id"]), packets, (), root=root, now=now):
            continue
        verified_rows.append(index)
    # One canonical configuration and one source snapshot can contribute once.
    return qualified.loc[verified_rows].drop_duplicates(subset=["configuration_id", "source_snapshot_id"]).drop_duplicates(subset=["configuration_id"])


def _copula_blockers(frame: pd.DataFrame, qualifying: pd.DataFrame, required_agreements: int) -> list[str]:
    blockers: list[str] = []
    if frame.empty:
        blockers.append("missing_immutable_copula_comparison_event")
    if len(qualifying) < required_agreements:
        blockers.append(f"need_{required_agreements - len(qualifying)}_more_complete_exact_agreements")
    if not frame.empty and "comparison" in frame and (frame["comparison"] == "requires_review").any():
        blockers.append("copula_shadow_requires_review")
    if not frame.empty and "dynamic_veto_count" in frame and (pd.to_numeric(frame["dynamic_veto_count"], errors="coerce").fillna(0) > 0).any():
        blockers.append("copula_shadow_has_safety_veto")
    if not frame.empty and "source_timestamp" in frame:
        timestamps = pd.to_datetime(frame["source_timestamp"], utc=True, errors="coerce")
        if timestamps.isna().any():
            blockers.append("copula_shadow_invalid_source_timestamp")
    return blockers or ["copula_shadow_gate_not_satisfied"]


def _relative_or_empty(path: Path, root: Path) -> str:
    return str(path.relative_to(root)) if path.exists() else ""


def _next_step(strategy_family: str, copula_ready: bool) -> str:
    if strategy_family == "Copula":
        return "open the next strategy cell in shadow mode" if copula_ready else "collect another complete Copula shadow comparison"
    if copula_ready:
        return f"start {strategy_family} in shadow mode with independent evidence gates"
    return "do not route this cell until the Copula shadow gate passes"


def _markdown(frame: pd.DataFrame) -> str:
    lines = ["# Dynamic Strategy Cell Rollout", "", "All rows are non-authoritative shadow controls. No row permits paper or live execution.", ""]
    lines.append(frame.to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)
