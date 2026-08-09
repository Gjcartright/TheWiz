from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT, current_multi_venue_history_readiness_path


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
        agent="youtube_research_brain",
        purpose="Collect reviewed video research, map claims to exact Wizard modes, generate pair hypotheses, and learn from local replication outcomes.",
        input_reports="data/external/youtube/brain/video_registry.csv;data/processed/youtube_brain/claims.csv;reports/active/crypto_wizards_live_scanner_capture.csv;reports/active/youtube_hypothesis_validation_queue.csv;reports/active/workflow_walk_forward_ranked.csv",
        output_reports="reports/agents/youtube_brain_hypotheses.csv;reports/agents/youtube_brain_replication_scorecard.csv;reports/dashboard/youtube_brain_status.csv;reports/dashboard/youtube_hypothesis_validation.csv",
        promotion_authority="none_research_only",
        next_action_type="refresh_youtube_brain_or_test_hypothesis",
    ),
    MiniAgentSpec(
        agent="discovery_agent",
        purpose="Find venue-aware candidate pairs from Wizard, Apify, dYdX, Binance, Coinbase, ByBit, and other source lanes.",
        input_reports="data/processed/wizard_evidence.csv;reports/active/wizard_evidence_summary.md;reports/active/multi_venue_history_readiness.csv",
        output_reports="reports/agents/discovery_candidates.csv",
        promotion_authority="none_discovery_only",
        next_action_type="capture_exact_mode_or_fetch_history",
    ),
    MiniAgentSpec(
        agent="venue_evidence_agent",
        purpose="Check symbol mapping, venue lane, candle source, liquidity context, cost model, slippage model, and funding/borrow assumptions.",
        input_reports="reports/active/multi_venue_history_readiness.csv;reports/active/venue_lane_test_plan.csv",
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
        input_reports="reports/active/wizard_replay_handoff.csv;reports/active/wizard_exact_mode_capture_queue.csv;reports/active/*strategy*;data/raw/pair_details",
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
        agent="magicka_agent",
        purpose="Run the parallel magicka shadow learner over synthetic policy variants and return best-performing policy proposals.",
        input_reports="data/ml/trade_training_dataset.csv;reports/rl/rl_training_report.csv;reports/rl/rl_evaluation_report.csv",
        output_reports="reports/rl/rl_learning_cycle_summary.csv;reports/rl/rl_learning_cycle_backtests.csv;reports/agents/rl_learning_agent_experiments.csv",
        promotion_authority="none_research_only",
        next_action_type="run_magicka_learning_cycle",
    ),
    MiniAgentSpec(
        agent="sequential_thinking_magicka_agent",
        purpose="Run sequential-thinking reviews of magicka and model gates to propose the next learning experiments.",
        input_reports="reports/rl/rl_acceptance_report.csv;reports/ml/model_gated_acceptance.csv;reports/dashboard/blocked_trades_dashboard.csv;reports/rl/rl_learning_cycle_summary.csv",
        output_reports="reports/rl/sequential_thinking_magicka_recommendations.csv;reports/agents/sequential_thinking_magicka_experiments.csv",
        promotion_authority="none_research_only",
        next_action_type="run_sequential_thinking_magicka",
    ),
    MiniAgentSpec(
        agent="rl_similarity_agent",
        purpose="Refine RL candidate space by generating and validating similar-pair candidate sets for review.",
        input_reports="reports/agents/rl_ideas.csv;reports/agents/rl_idea_summary.csv;data/ml/trade_training_dataset.csv",
        output_reports="reports/agents/rl_pair_similarity.csv",
        promotion_authority="none_rl_hint_only",
        next_action_type="extract_rl_similarity_candidates",
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
    effectiveness = _agent_effectiveness(root)
    queue = _next_action_queue(root, effectiveness=effectiveness)
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


def _next_action_queue(root: Path, *, effectiveness: dict[str, float] | None = None) -> pd.DataFrame:
    effectiveness = effectiveness or {}
    rows: list[dict[str, object]] = []
    rows.extend(_youtube_tasks(root))
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
    priority_order = {"high": 0.0, "medium": 1.0, "low": 2.0}
    frame["_base_priority"] = frame["priority"].map(priority_order).fillna(9.0)
    frame["_effectiveness"] = frame["assigned_agent"].map(effectiveness).fillna(0.0)
    frame["_effective_rank"] = _effective_rank(frame["_base_priority"], frame["_effectiveness"]).round(4)
    frame = (
        frame.sort_values(["_effective_rank", "assigned_agent", "pair"])
        .drop(columns=["_base_priority", "_effectiveness"])
        .reset_index(drop=True)
    )
    frame.insert(0, "rank", range(1, len(frame) + 1))
    frame["agent_effectiveness"] = frame["assigned_agent"].map(effectiveness).fillna(0.0)
    frame = frame.drop(columns=["_effective_rank"])
    return frame


def _youtube_tasks(root: Path) -> list[dict[str, object]]:
    collection_path = root / "reports" / "agents" / "youtube_brain_collection_status.csv"
    hypothesis_path = root / "reports" / "agents" / "youtube_brain_hypotheses.csv"
    collection = _read_csv(collection_path)
    hypotheses = _read_csv(hypothesis_path)
    validation_path = root / "reports" / "active" / "youtube_hypothesis_validation_queue.csv"
    validation = _read_csv(validation_path)
    rows: list[dict[str, object]] = []
    if collection.empty:
        rows.append(
            _task(
                assigned_agent="youtube_research_brain",
                task_type="refresh_youtube_brain",
                pair="",
                reason="youtube_collection_status_missing",
                priority="medium",
                evidence_path=collection_path,
                next_step="run run-youtube-brain to refresh research memory and pair hypotheses",
            )
        )
    for _, row in hypotheses.head(20).iterrows():
        if not _boolish(row.get("wizard_discovery_pass")):
            continue
        pair = str(row.get("pair", ""))
        mode = str(row.get("exact_mode", ""))
        match = _youtube_validation_match(validation, pair=pair, exact_mode=mode)
        validation_status = str(match.get("validation_status", ""))
        if not match.empty and not _boolish(match.get("slippage_model_ready", False)):
            rows.append(
                _task(
                    assigned_agent="venue_evidence_agent",
                    task_type="calibrate_hyperliquid_l2_slippage",
                    pair=pair,
                    reason=f"{mode}:hyperliquid_l2_slippage_not_calibrated",
                    priority="high",
                    evidence_path=validation_path,
                    next_step="continue scheduled Hyperliquid L2 snapshots until both legs have 12 independent samples",
                )
            )
        if validation_status == "BLOCKED_MISSING_EXACT_SETTINGS":
            rows.append(
                _task(
                    assigned_agent="discovery_agent",
                    task_type="capture_youtube_hypothesis_exact_mode",
                    pair=pair,
                    reason=f"{mode}:wizard_exact_settings_not_live_confirmed",
                    priority="high",
                    evidence_path=validation_path,
                    next_step="complete and import the prefilled Wizard settings capture for this exact mode",
                )
            )
            continue
        if validation_status in {"BLOCKED_VENUE_ECONOMICS", "BLOCKED_POINT_IN_TIME_EVIDENCE"}:
            rows.append(
                _task(
                    assigned_agent="venue_evidence_agent",
                    task_type="complete_youtube_hypothesis_venue_evidence",
                    pair=pair,
                    reason=f"{mode}:{match.get('blocker', validation_status)}",
                    priority="high",
                    evidence_path=validation_path,
                    next_step=str(match.get("next_step", "continue Hyperliquid evidence cadence")),
                )
            )
            continue
        rows.append(
            _task(
                assigned_agent="strategy_test_agent",
                task_type="test_youtube_brain_hypothesis",
                pair=pair,
                reason=f"{mode}:youtube_research_hypothesis_requires_replication",
                priority="high",
                evidence_path=hypothesis_path,
                next_step=row.get("next_step", "run exact-mode costed walk-forward and regime tests"),
            )
        )
    return rows


def _youtube_validation_match(frame: pd.DataFrame, *, pair: str, exact_mode: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    pair_key = pair.upper().replace("-USD", "").replace("/", "|")
    mode_key = "".join(character for character in exact_mode.lower() if character.isalnum())
    pair_match = frame.get("pair", pd.Series("", index=frame.index)).astype(str).map(
        lambda value: value.upper().replace("-USD", "").replace("/", "|")
    )
    mode_match = frame.get("exact_mode", pd.Series("", index=frame.index)).astype(str).map(
        lambda value: "".join(character for character in value.lower() if character.isalnum())
    )
    matches = frame.loc[pair_match.eq(pair_key) & mode_match.eq(mode_key)]
    return matches.iloc[0] if not matches.empty else pd.Series(dtype=object)


def _wizard_capture_tasks(root: Path) -> list[dict[str, object]]:
    active = root / "reports" / "active"
    detail_path = active / "wizard_pair_detail_capture_queue.csv"
    detail = _read_csv(detail_path)
    path = active / "wizard_exact_mode_capture_queue.csv"
    frame = _read_csv(path)
    rows = []
    for _, row in detail.head(20).iterrows():
        priority = "high" if str(row.get("priority", "")) == "P1" else "medium"
        rows.append(
            _task(
                assigned_agent="discovery_agent",
                task_type="capture_pair_detail_settings",
                pair=row.get("pair", ""),
                reason=f"{row.get('exact_mode', '')}:screen_pass_missing_dashboard_settings",
                priority=priority,
                evidence_path=detail_path,
                next_step="capture current Wizard exact-mode settings, then route the pair to Hyperliquid validation",
            )
        )
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
    # The current production research lane is Wizard discovery to Hyperliquid.
    # Once its validation queue exists, legacy dYdX replay tasks are superseded.
    if (root / "reports" / "active" / "youtube_hypothesis_validation_queue.csv").exists():
        return []
    path = current_multi_venue_history_readiness_path(root)
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
    learning_summary_path = root / "reports" / "rl" / "rl_learning_cycle_summary.csv"
    training = _read_csv(training_path)
    acceptance = _read_csv(acceptance_path)
    blocker = ""
    if acceptance.empty:
        blocker = "missing_rl_acceptance_report"
    elif "accepted" in acceptance and not acceptance["accepted"].astype(bool).any():
        blocker = str(acceptance.get("blocker", pd.Series(["rl_acceptance_not_passed"])).iloc[0])
    magicka_ready = learning_summary_path.exists()
    if training.empty or blocker:
        return [
            _task(
                assigned_agent="magicka_agent",
                task_type="run_magicka_learning_cycle",
                pair="",
                reason=blocker or "missing_rl_training_report",
                priority="medium",
                evidence_path=training_path if training_path.exists() else learning_summary_path,
                next_step="run magicka learner in shadow mode to discover better policy variants",
            ),
            _task(
                assigned_agent="sequential_thinking_magicka_agent",
                task_type="run_sequential_thinking_magicka",
                pair="",
                reason=blocker or "missing_rl_training_report",
                priority="medium",
                evidence_path=learning_summary_path if learning_summary_path.exists() else training_path,
                next_step="run sequential-thinking magicka to propose the next candidate policy adjustments",
            ),
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
            assigned_agent="magicka_agent",
            task_type="run_magicka_learning_cycle",
            pair="",
            reason="refresh_rl_learning_candidates",
            priority="medium",
            evidence_path=learning_summary_path if magicka_ready else training_path,
            next_step="run magicka learner and refresh shadow policy candidates and holding-time experiments",
        ),
        _task(
            assigned_agent="sequential_thinking_magicka_agent",
            task_type="run_sequential_thinking_magicka",
            pair="",
            reason="refresh_magicka_recommendations",
            priority="medium",
            evidence_path=learning_summary_path,
            next_step="run sequential-thinking magicka and convert best recommendations into coaching tasks",
        ),
        _task(
            assigned_agent="rl_idea_agent",
            task_type="run_rl_idea_scout",
            pair="",
            reason="refresh_rl_ideas_from_recent_research",
            priority="medium",
            evidence_path=training_path,
            next_step="run RL idea scout to update candidate hypotheses and evidence",
        ),
        _task(
            assigned_agent="rl_similarity_agent",
            task_type="extract_rl_similarity_candidates",
            pair="",
            reason="rl_research_reports_available",
            priority="medium",
            evidence_path=training_path,
            next_step="extract RL fingerprints and search for similar pairs",
        ),
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


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "pass", "accepted"}


def _agent_effectiveness(root: Path) -> dict[str, float]:
    path = root / "reports" / "agents" / "agent_effectiveness.csv"
    if not path.exists():
        return {}
    frame = _read_csv(path)
    if frame.empty or "agent" not in frame.columns or "effectiveness_score" not in frame.columns:
        return {}
    scores = pd.to_numeric(frame["effectiveness_score"], errors="coerce").fillna(0.0)
    return {
        str(agent): float(max(0.0, min(1.0, score)))
        for agent, score in zip(frame["agent"].astype(str), scores, strict=False)
        if agent
    }


def _effective_rank(base_priority: pd.Series, effectiveness: pd.Series) -> pd.Series:
    bounded = effectiveness.clip(lower=0.0, upper=1.0)
    return base_priority - 0.65 * bounded


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
