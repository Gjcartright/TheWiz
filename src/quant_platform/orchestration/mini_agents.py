from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT


AGENT_DIR = "reports/orchestration"


@dataclass(frozen=True)
class MiniAgentSpec:
    agent: str
    purpose: str
    input_reports: str
    output_reports: str
    promotion_authority: str
    next_action_type: str


MINI_AGENTS: tuple[MiniAgentSpec, ...] = (
    MiniAgentSpec(
        agent="discovery_agent",
        purpose="Find venue-aware candidate pairs from Wizard, Apify, dYdX, Binance, Coinbase, ByBit, and other source lanes.",
        input_reports="data/processed/wizard_evidence.csv;reports/active/wizard_evidence_summary.md;reports/active/multi_venue_history_readiness_2026-06-25.csv",
        output_reports="reports/agents/discovery_candidates.csv",
        promotion_authority="none_discovery_only",
        next_action_type="capture_exact_mode_or_fetch_history",
    ),
    MiniAgentSpec(
        agent="venue_evidence_agent",
        purpose="Check symbol mapping, venue lane, candle source, liquidity context, cost model, slippage model, and funding/borrow assumptions.",
        input_reports="reports/active/multi_venue_history_readiness_2026-06-25.csv;reports/active/venue_lane_test_plan.csv",
        output_reports="reports/agents/venue_evidence.csv",
        promotion_authority="none_evidence_only",
        next_action_type="repair_mapping_or_fetch_venue_history",
    ),
    MiniAgentSpec(
        agent="data_quality_agent",
        purpose="Audit local histories for depth, freshness, missing candles, stale fields, duplicated rows, and bad symbols.",
        input_reports="data/raw/pair_details;reports/active/binance_spot_history_readiness.csv;reports/active/hyperliquid_lane_readiness.csv",
        output_reports="reports/agents/data_quality_audit.csv",
        promotion_authority="none_quality_only",
        next_action_type="refresh_or_extend_history",
    ),
    MiniAgentSpec(
        agent="strategy_test_agent",
        purpose="Run exact-mode local replay and strategy-family tests across static, dynamic, OU, copula, ECM, regime, entry, and exit styles.",
        input_reports="reports/active/wizard_exact_mode_capture_queue.csv;reports/active/*strategy*;data/raw/pair_details",
        output_reports="reports/agents/strategy_test_results.csv",
        promotion_authority="local_replay_evidence_only",
        next_action_type="run_exact_mode_or_strategy_family_sweep",
    ),
    MiniAgentSpec(
        agent="rl_idea_agent",
        purpose="Run RL idea scout, extract policy ideas, trade fingerprints, exit ideas, sizing ideas, and similar-pair candidates.",
        input_reports="reports/rl/rl_training_report.csv;reports/rl/rl_execution_backtest.csv;data/raw/pair_details",
        output_reports="reports/agents/rl_ideas.csv;reports/agents/rl_pair_similarity.csv",
        promotion_authority="none_rl_hint_only",
        next_action_type="run_rl_idea_scout_or_similarity_search",
    ),
    MiniAgentSpec(
        agent="cost_risk_agent",
        purpose="Evaluate fees, slippage, funding, execution-risk cost, drawdown, trade count, net return, and break-even cost headroom.",
        input_reports="reports/active/*after_cost.csv;reports/active/*cost_comparison.csv;reports/active/binance_exact_mode_strategy_sweep_2026-06-25.csv",
        output_reports="reports/agents/cost_risk_review.csv",
        promotion_authority="risk_evidence_only",
        next_action_type="review_cost_risk_or_size_limits",
    ),
    MiniAgentSpec(
        agent="red_team_agent",
        purpose="Search for hindsight, overfit, thin trades, fake precision, stale data, venue mismatch, and single-pair concentration.",
        input_reports="reports/active;reports/ml;reports/rl",
        output_reports="reports/agents/red_team_review.csv",
        promotion_authority="none_can_block_only",
        next_action_type="red_team_review",
    ),
    MiniAgentSpec(
        agent="decision_agent",
        purpose="Combine agent evidence into PROMOTE, WATCH, FETCH_MORE_DATA, or REJECT without allowing any single agent to promote alone.",
        input_reports="reports/agents/*.csv;data/processed/pair_universe.csv",
        output_reports="reports/agents/final_decision_board.csv",
        promotion_authority="orchestrator_only_after_acceptance_gates",
        next_action_type="update_decision_board",
    ),
)


def build_mini_agent_orchestration(root: Path = ROOT) -> CommandResult:
    output_dir = root / AGENT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = _agent_registry_frame(root)
    queue = _next_action_queue(root)
    registry_path = output_dir / "mini_agent_registry.csv"
    queue_path = output_dir / "next_action_queue.csv"
    summary_path = output_dir / "mini_agent_orchestration.md"
    registry.to_csv(registry_path, index=False)
    queue.to_csv(queue_path, index=False)
    summary_path.write_text(_summary_markdown(registry, queue), encoding="utf-8")
    return CommandResult(
        paths={
            "mini_agent_registry": registry_path,
            "next_action_queue": queue_path,
            "mini_agent_summary": summary_path,
        },
        summary={
            "rows": len(queue),
            "agents": len(registry),
            "blocked_promotions": int(queue["promotion_allowed"].astype(str).str.lower().eq("false").sum()) if not queue.empty else 0,
        },
    )


def _agent_registry_frame(root: Path) -> pd.DataFrame:
    rows = []
    for priority, spec in enumerate(MINI_AGENTS, start=1):
        input_status = _input_status(root, spec.input_reports)
        rows.append(
            {
                "priority": priority,
                "agent": spec.agent,
                "purpose": spec.purpose,
                "input_reports": spec.input_reports,
                "input_status": input_status,
                "output_reports": spec.output_reports,
                "promotion_authority": spec.promotion_authority,
                "next_action_type": spec.next_action_type,
                "enabled": True,
            }
        )
    return pd.DataFrame(rows)


def _next_action_queue(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(_wizard_capture_tasks(root))
    rows.extend(_venue_history_tasks(root))
    rows.extend(_rl_tasks(root))
    rows.extend(_red_team_tasks(root))
    if not rows:
        rows.append(
            {
                "rank": 1,
                "assigned_agent": "discovery_agent",
                "task_type": "refresh_discovery",
                "pair": "",
                "reason": "no_agent_tasks_found",
                "priority": "medium",
                "promotion_allowed": False,
                "evidence_path": "reports/orchestration/mini_agent_registry.csv",
                "next_step": "run discovery and venue source sweep",
            }
        )
    frame = pd.DataFrame(rows)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    frame["_priority_rank"] = frame["priority"].map(priority_order).fillna(9)
    frame = frame.sort_values(["_priority_rank", "assigned_agent", "pair"]).drop(columns=["_priority_rank"]).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame


def _wizard_capture_tasks(root: Path) -> list[dict[str, object]]:
    path = root / "reports" / "active" / "wizard_exact_mode_capture_queue.csv"
    frame = _read_csv(path)
    rows = []
    for _, row in frame.head(20).iterrows():
        rows.append(
            _task(
                assigned_agent="discovery_agent",
                task_type="capture_exact_mode",
                pair=row.get("pair", ""),
                reason="passes_discovery_gate_missing_exact_mode",
                priority="high",
                evidence_path=path,
                next_step="open Wizard pair page and capture exact mode, spread id, strategy id, period, and settings",
            )
        )
    return rows


def _venue_history_tasks(root: Path) -> list[dict[str, object]]:
    path = root / "reports" / "active" / "multi_venue_history_readiness_2026-06-25.csv"
    frame = _read_csv(path)
    rows = []
    for _, row in frame.head(20).iterrows():
        if str(row.get("readiness_status", "")) not in {"ready_to_fetch", "ready_for_replay"}:
            continue
        rows.append(
            _task(
                assigned_agent="venue_evidence_agent",
                task_type="fetch_or_replay_venue_history",
                pair=row.get("pair", ""),
                reason=f"{row.get('wizard_exchange', '')}:{row.get('readiness_status', '')}",
                priority="high" if str(row.get("readiness_status", "")) == "ready_to_fetch" else "medium",
                evidence_path=path,
                next_step=row.get("next_step", "fetch venue candles and build local history"),
            )
        )
    return rows


def _rl_tasks(root: Path) -> list[dict[str, object]]:
    training_path = root / "reports" / "rl" / "rl_training_report.csv"
    acceptance_path = root / "reports" / "rl" / "rl_acceptance_report.csv"
    training = _read_csv(training_path)
    acceptance = _read_csv(acceptance_path)
    blocker = ""
    if acceptance.empty:
        blocker = "missing_rl_acceptance_report"
    elif "accepted" in acceptance and not acceptance["accepted"].astype(bool).any():
        blocker = str(acceptance.get("blocker", pd.Series(["rl_acceptance_not_passed"])).iloc[0])
    if training.empty or blocker:
        return [
            _task(
                assigned_agent="rl_idea_agent",
                task_type="run_rl_idea_scout",
                pair="",
                reason=blocker or "missing_rl_training_report",
                priority="medium",
                evidence_path=training_path if training_path.exists() else acceptance_path,
                next_step="run RL research, extract policy ideas, fingerprints, exits, sizing, and similar-pair candidates",
            )
        ]
    return [
        _task(
            assigned_agent="rl_idea_agent",
            task_type="extract_rl_similarity_candidates",
            pair="",
            reason="rl_research_reports_available",
            priority="medium",
            evidence_path=training_path,
            next_step="extract RL fingerprints and search for similar pairs",
        )
    ]


def _red_team_tasks(root: Path) -> list[dict[str, object]]:
    evidence = [
        root / "reports" / "active" / "binance_exact_mode_strategy_sweep_2026-06-25.csv",
        root / "data" / "processed" / "pair_universe.csv",
    ]
    existing = [path for path in evidence if path.exists()]
    if not existing:
        return []
    return [
        _task(
            assigned_agent="red_team_agent",
            task_type="red_team_current_candidates",
            pair="",
            reason="review_high_sharpe_or_rl_generated_candidates_before_any_promotion",
            priority="medium",
            evidence_path=existing[0],
            next_step="look for hindsight, thin trades, drawdown, stale data, and venue mismatch",
        )
    ]


def _task(
    *,
    assigned_agent: str,
    task_type: str,
    pair: object,
    reason: object,
    priority: str,
    evidence_path: Path,
    next_step: object,
) -> dict[str, object]:
    return {
        "assigned_agent": assigned_agent,
        "task_type": task_type,
        "pair": pair,
        "reason": reason,
        "priority": priority,
        "promotion_allowed": False,
        "evidence_path": _rel(evidence_path),
        "next_step": next_step,
    }


def _input_status(root: Path, input_reports: str) -> str:
    statuses = []
    for raw in input_reports.split(";"):
        item = raw.strip()
        if not item:
            continue
        if "*" in item:
            matches = list(root.glob(item))
            statuses.append(f"{item}:matches={len(matches)}")
        else:
            statuses.append(f"{item}:{'present' if (root / item).exists() else 'missing'}")
    return ";".join(statuses)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _summary_markdown(registry: pd.DataFrame, queue: pd.DataFrame) -> str:
    lines = [
        "# Mini-Agent Orchestration",
        "",
        "- Mini agents create evidence and tasks; they do not promote pairs by themselves.",
        "- RL is an idea scout and similarity-search worker, not acceptance authority.",
        "- The orchestrator owns task order, blockers, dashboard reporting, and final decision buckets.",
        "",
        f"- Agents: `{len(registry)}`",
        f"- Next-action tasks: `{len(queue)}`",
        "",
        "## Agents",
        "",
        registry[["priority", "agent", "next_action_type", "promotion_authority", "input_status"]].to_markdown(index=False),
        "",
        "## Next Action Queue",
        "",
        queue.head(30).to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)
