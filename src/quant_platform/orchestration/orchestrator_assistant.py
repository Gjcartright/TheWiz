from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.orchestration.mini_agents import build_mini_agent_orchestration


ORCHESTRATION_DIR = "reports/orchestration"
AGENT_MEMORY_DIR = "data/agent_memory"
AGENT_REPORT_DIR = "reports/agents"


def build_orchestrator_assistant(root: Path = ROOT) -> CommandResult:
    """Build dynamic task cards and per-agent memory from the mini-agent queue."""

    orchestration_dir = root / ORCHESTRATION_DIR
    orchestration_dir.mkdir(parents=True, exist_ok=True)
    agent_report_dir = root / AGENT_REPORT_DIR
    agent_report_dir.mkdir(parents=True, exist_ok=True)

    mini_agent_result = build_mini_agent_orchestration(root)
    registry = _read_csv(mini_agent_result.paths["mini_agent_registry"])
    queue = _read_csv(mini_agent_result.paths["next_action_queue"])
    tasks = _assistant_tasks(queue, registry)
    task_cards = _task_cards(tasks, registry)
    memory_events = _memory_events(tasks)

    tasks_path = orchestration_dir / "orchestrator_assistant_tasks.csv"
    cards_path = orchestration_dir / "task_cards.jsonl"
    reasoning_path = orchestration_dir / "orchestrator_assistant_reasoning.md"
    learning_path = agent_report_dir / "agent_learning_summary.csv"
    effectiveness_path = agent_report_dir / "agent_effectiveness.csv"

    tasks.to_csv(tasks_path, index=False)
    _write_jsonl(cards_path, task_cards)
    _append_agent_memory(root, memory_events)
    learning = _learning_summary(root, registry)
    effectiveness = _effectiveness_summary(learning)
    learning.to_csv(learning_path, index=False)
    effectiveness.to_csv(effectiveness_path, index=False)
    reasoning_path.write_text(_reasoning_markdown(tasks, registry, learning), encoding="utf-8")

    return CommandResult(
        paths={
            "orchestrator_assistant_tasks": tasks_path,
            "task_cards": cards_path,
            "orchestrator_assistant_reasoning": reasoning_path,
            "agent_learning_summary": learning_path,
            "agent_effectiveness": effectiveness_path,
        },
        summary={
            "rows": len(tasks),
            "agents_with_memory": int(learning["agent"].nunique()) if not learning.empty else 0,
            "task_cards": len(task_cards),
            "promotion_allowed": int(tasks["promotion_allowed"].astype(bool).sum()) if not tasks.empty else 0,
        },
    )


def _assistant_tasks(queue: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "task_id",
        "rank",
        "assigned_agent",
        "task_type",
        "pair",
        "why_now",
        "priority",
        "inputs_needed",
        "expected_output",
        "blocking_condition",
        "promotion_allowed",
        "evidence_path",
        "next_step",
        "assistant_decision",
    ]
    if queue.empty:
        return pd.DataFrame(columns=columns)
    registry_by_agent = registry.set_index("agent") if not registry.empty and "agent" in registry.columns else pd.DataFrame()
    rows = []
    for _, row in queue.iterrows():
        agent = str(row.get("assigned_agent", ""))
        spec = registry_by_agent.loc[agent] if not registry_by_agent.empty and agent in registry_by_agent.index else pd.Series(dtype=object)
        task_type = str(row.get("task_type", ""))
        pair = str(row.get("pair", "") or "")
        evidence = str(row.get("evidence_path", "") or "")
        why_now = str(row.get("reason", "") or "")
        task_id = _task_id(agent, task_type, pair, why_now, evidence)
        rows.append(
            {
                "task_id": task_id,
                "rank": row.get("rank", ""),
                "assigned_agent": agent,
                "task_type": task_type,
                "pair": pair,
                "why_now": why_now,
                "priority": row.get("priority", "medium"),
                "inputs_needed": spec.get("input_reports", evidence),
                "expected_output": spec.get("output_reports", ""),
                "blocking_condition": _blocking_condition(task_type, why_now),
                "promotion_allowed": False,
                "evidence_path": evidence,
                "next_step": row.get("next_step", ""),
                "assistant_decision": _assistant_decision(task_type, why_now),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _task_cards(tasks: pd.DataFrame, registry: pd.DataFrame) -> list[dict[str, object]]:
    registry_by_agent = registry.set_index("agent") if not registry.empty and "agent" in registry.columns else pd.DataFrame()
    cards: list[dict[str, object]] = []
    for _, task in tasks.iterrows():
        agent = str(task.get("assigned_agent", ""))
        spec = registry_by_agent.loc[agent] if not registry_by_agent.empty and agent in registry_by_agent.index else pd.Series(dtype=object)
        cards.append(
            {
                "task_id": task.get("task_id", ""),
                "assigned_agent": agent,
                "task_type": task.get("task_type", ""),
                "pair": task.get("pair", ""),
                "why_now": task.get("why_now", ""),
                "inputs_needed": _split_reports(task.get("inputs_needed", "")),
                "expected_output": _split_reports(task.get("expected_output", "")),
                "blocking_condition": task.get("blocking_condition", ""),
                "priority": task.get("priority", "medium"),
                "promotion_allowed": False,
                "promotion_authority": spec.get("promotion_authority", "none"),
                "evidence_path": task.get("evidence_path", ""),
                "next_step": task.get("next_step", ""),
            }
        )
    return cards


def _memory_events(tasks: pd.DataFrame) -> list[dict[str, object]]:
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, task in tasks.iterrows():
        rows.append(
            {
                "timestamp": timestamp,
                "agent": task.get("assigned_agent", ""),
                "task_id": task.get("task_id", ""),
                "task_type": task.get("task_type", ""),
                "pair": task.get("pair", ""),
                "input_evidence": task.get("evidence_path", ""),
                "action_taken": "task_assigned",
                "finding": task.get("why_now", ""),
                "confidence": _confidence(task.get("priority", "")),
                "blocker": task.get("blocking_condition", ""),
                "next_step": task.get("next_step", ""),
                "outcome_known": False,
                "outcome_label": "",
                "learning_note": "assigned_by_orchestrator_assistant",
                "promotion_allowed": False,
            }
        )
    return rows


def _append_agent_memory(root: Path, events: list[dict[str, object]]) -> None:
    memory_root = root / AGENT_MEMORY_DIR
    memory_root.mkdir(parents=True, exist_ok=True)
    for event in events:
        agent = str(event.get("agent", "") or "unknown_agent")
        path = memory_root / f"{agent}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def _learning_summary(root: Path, registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    known_agents = list(registry["agent"]) if not registry.empty and "agent" in registry.columns else []
    for agent in known_agents:
        path = root / AGENT_MEMORY_DIR / f"{agent}.jsonl"
        events = _read_jsonl(path)
        outcome_known = sum(1 for event in events if bool(event.get("outcome_known", False)))
        positive = sum(1 for event in events if str(event.get("outcome_label", "")).lower() in {"accepted", "validated", "passed"})
        rows.append(
            {
                "agent": agent,
                "memory_path": _rel(path, root),
                "events": len(events),
                "outcomes_known": outcome_known,
                "positive_outcomes": positive,
                "learning_status": "learning_ready" if events else "no_memory_events_yet",
                "next_learning_step": "attach outcomes to prior task_ids" if events and outcome_known == 0 else "continue_collecting_agent_events",
            }
        )
    return pd.DataFrame(rows)


def _effectiveness_summary(learning: pd.DataFrame) -> pd.DataFrame:
    if learning.empty:
        return pd.DataFrame(columns=["agent", "effectiveness_score", "reason"])
    rows = []
    for _, row in learning.iterrows():
        events = int(row.get("events", 0) or 0)
        known = int(row.get("outcomes_known", 0) or 0)
        positive = int(row.get("positive_outcomes", 0) or 0)
        if known:
            score = round(positive / known, 4)
            reason = "outcome_based"
        elif events:
            score = 0.0
            reason = "memory_exists_outcomes_pending"
        else:
            score = 0.0
            reason = "no_memory_events"
        rows.append({"agent": row.get("agent", ""), "effectiveness_score": score, "reason": reason})
    return pd.DataFrame(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _split_reports(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def _task_id(agent: str, task_type: str, pair: str, why_now: str, evidence: str) -> str:
    raw = "|".join([agent, task_type, pair, why_now, evidence])
    return f"task_{uuid5(NAMESPACE_URL, raw).hex[:12]}"


def _blocking_condition(task_type: str, why_now: str) -> str:
    if "capture_exact_mode" in task_type:
        return "missing_exact_mode"
    if "fetch" in task_type or "history" in task_type:
        return "missing_or_unverified_local_history"
    if "rl" in task_type:
        return "rl_idea_not_validated_by_local_replay"
    if "red_team" in task_type:
        return "red_team_review_pending"
    return str(why_now or "task_not_completed")


def _assistant_decision(task_type: str, why_now: str) -> str:
    if "capture_exact_mode" in task_type:
        return "assign_capture_before_replay"
    if "fetch" in task_type or "history" in task_type:
        return "assign_history_build_before_strategy_test"
    if "rl" in task_type:
        return "assign_rl_as_idea_scout_not_promotion_authority"
    if "red_team" in task_type:
        return "assign_red_team_before_any_promotion"
    return f"assign_task:{why_now}"


def _confidence(priority: object) -> float:
    return {"high": 0.8, "medium": 0.55, "low": 0.35}.get(str(priority).lower(), 0.5)


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _reasoning_markdown(tasks: pd.DataFrame, registry: pd.DataFrame, learning: pd.DataFrame) -> str:
    lines = [
        "# Orchestrator Assistant",
        "",
        "The orchestrator assistant converts the mini-agent queue into specific task cards and persistent agent-memory events.",
        "",
        "Rules:",
        "",
        "- Tasks can prioritize work, but cannot promote pairs.",
        "- RL tasks are idea-scout tasks until local replay validates them.",
        "- Red-team review is required before any later promotion decision.",
        "- Agent memory records assignments now; outcomes are attached later by `task_id`.",
        "",
        f"- Agents: `{len(registry)}`",
        f"- Tasks: `{len(tasks)}`",
        f"- Agents with memory events: `{int((learning['events'] > 0).sum()) if not learning.empty else 0}`",
        "",
    ]
    if not tasks.empty:
        lines.extend(["## Task Cards", "", tasks[["task_id", "assigned_agent", "task_type", "pair", "priority", "assistant_decision"]].head(30).to_markdown(index=False), ""])
    if not learning.empty:
        lines.extend(["## Agent Memory", "", learning.to_markdown(index=False), ""])
    return "\n".join(lines)
