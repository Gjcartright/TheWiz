from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT


AGENT_REPORT_DIR = "reports/agents"


@dataclass(frozen=True)
class StrategySpecialist:
    strategy_family: str
    wizard_mode: str
    specialist_agent: str
    focus: str
    required_reference: str


STRATEGY_SPECIALISTS: tuple[StrategySpecialist, ...] = (
    StrategySpecialist("Static Spread", "Static (Spread)", "static_spread_agent", "fixed hedge spread dislocation entries", "spread_formula"),
    StrategySpecialist("Static ZScoreR", "Static (ZScoreR)", "static_zscorer_agent", "fixed hedge normalized z-score reversions", "zscore_formula"),
    StrategySpecialist("Dyn Spread", "Dyn (Spread)", "dynamic_spread_agent", "rolling hedge spread dislocation entries", "dynamic_hedge_formula"),
    StrategySpecialist("Dyn ZScoreR", "Dyn (ZScoreR)", "dynamic_zscorer_agent", "rolling hedge normalized z-score reversions", "dynamic_zscore_formula"),
    StrategySpecialist("OU Spread", "OU (Spread)", "ou_spread_agent", "OU spread pullback and half-life behavior", "ou_formula"),
    StrategySpecialist("OU ZScoreR", "OU (ZScoreR)", "ou_zscorer_agent", "OU normalized dislocation behavior", "ou_zscore_formula"),
    StrategySpecialist("Copula", "Copula", "copula_agent", "tail dependency and conditional dislocation arbitrage", "copula_formula"),
)


def build_specialist_scoreboard(root: Path = ROOT) -> CommandResult:
    output_dir = root / AGENT_REPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    wizard = _read_csv(root / "data" / "processed" / "wizard_evidence.csv")
    local = _read_csv(root / "reports" / "active" / "binance_exact_mode_strategy_sweep_2026-06-25.csv")
    learning = _read_csv(root / "reports" / "agents" / "agent_learning_summary.csv")
    rl_acceptance = _read_csv(root / "reports" / "rl" / "rl_acceptance_report.csv")
    formula_dictionary = root / "docs" / "formula_dictionary.md"

    scoreboard = pd.DataFrame(
        [
            _score_specialist(
                specialist,
                wizard=wizard,
                local=local,
                learning=learning,
                rl_acceptance=rl_acceptance,
                formula_dictionary=formula_dictionary,
                root=root,
            )
            for specialist in STRATEGY_SPECIALISTS
        ]
    )
    horizontal = _horizontal_scores(scoreboard, learning)

    scoreboard_path = output_dir / "specialist_strategy_scoreboard.csv"
    scoreboard_md_path = output_dir / "specialist_strategy_scoreboard.md"
    horizontal_path = output_dir / "horizontal_agent_scores.csv"
    scoreboard.to_csv(scoreboard_path, index=False)
    horizontal.to_csv(horizontal_path, index=False)
    scoreboard_md_path.write_text(_scoreboard_markdown(scoreboard, horizontal), encoding="utf-8")

    return CommandResult(
        paths={
            "specialist_strategy_scoreboard": scoreboard_path,
            "specialist_strategy_scoreboard_md": scoreboard_md_path,
            "horizontal_agent_scores": horizontal_path,
        },
        summary={
            "rows": len(scoreboard),
            "specialists": len(scoreboard),
            "promotion_allowed": int(scoreboard["promotion_allowed"].astype(bool).sum()),
            "promote_testing": int(scoreboard["decision"].eq("PROMOTE_TESTING").sum()),
        },
    )


def _score_specialist(
    specialist: StrategySpecialist,
    *,
    wizard: pd.DataFrame,
    local: pd.DataFrame,
    learning: pd.DataFrame,
    rl_acceptance: pd.DataFrame,
    formula_dictionary: Path,
    root: Path,
) -> dict[str, object]:
    wizard_rows = _rows_for_wizard_mode(wizard, specialist.wizard_mode)
    local_rows = _rows_for_strategy_family(local, specialist.strategy_family)

    research_score = _research_score(wizard_rows)
    test_score = _test_score(local_rows)
    reference_score = 1.0 if formula_dictionary.exists() else 0.0
    memory_score = _memory_score(learning, specialist.specialist_agent)
    rl_score = _rl_score(rl_acceptance)
    combined_score = round(
        0.25 * test_score + 0.20 * research_score + 0.20 * reference_score + 0.20 * memory_score + 0.15 * rl_score,
        4,
    )
    blocker, decision = _decision(
        research_score=research_score,
        test_score=test_score,
        reference_score=reference_score,
        wizard_rows=wizard_rows,
        local_rows=local_rows,
    )
    reason = _reason(decision, blocker, wizard_rows, local_rows)
    evidence_path = _evidence_path(root, wizard_rows, local_rows, formula_dictionary)

    return {
        "strategy_family": specialist.strategy_family,
        "wizard_exact_mode": specialist.wizard_mode,
        "specialist_agent": specialist.specialist_agent,
        "focus": specialist.focus,
        "research_score": research_score,
        "memory_score": memory_score,
        "test_score": test_score,
        "reference_score": reference_score,
        "rl_score": rl_score,
        "combined_score": combined_score,
        "decision": decision,
        "reason": reason,
        "blocker": blocker,
        "wizard_evidence_rows": len(wizard_rows),
        "local_test_rows": len(local_rows),
        "best_wizard_sharpe": _max_numeric(wizard_rows, "sharpe"),
        "best_wizard_returns_total": _max_numeric(wizard_rows, "returns_total"),
        "best_local_sharpe": _max_numeric(local_rows, "sharpe"),
        "best_local_profit_factor": _max_numeric(local_rows, "profit_factor"),
        "worst_local_max_drawdown": _max_numeric(local_rows, "max_drawdown"),
        "promotion_allowed": False,
        "promotion_authority": "none_scoreboard_only_local_replay_acceptance_required",
        "next_step": _next_step(decision, specialist.strategy_family),
        "evidence_path": evidence_path,
    }


def _rows_for_wizard_mode(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    for column in ("exact_mode", "wizard_exact_mode", "mode", "resolved_mode"):
        if column in frame:
            return frame[frame[column].astype(str).eq(mode)].copy()
    return frame.iloc[0:0].copy()


def _rows_for_strategy_family(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    candidates = [
        family,
        family.replace(" ZScoreR", " (ZScoreR)").replace(" Spread", " (Spread)"),
        family.replace("Dyn", "Dynamic"),
    ]
    for column in ("strategy_family", "wizard_exact_mode", "exact_mode", "exact_mode_label", "strategy", "strategy_name", "mode", "family"):
        if column in frame:
            values = frame[column].astype(str)
            mask = values.isin(candidates)
            for candidate in candidates:
                mask = mask | values.str.contains(candidate, case=False, regex=False)
            return frame[mask].copy()
    return frame.iloc[0:0].copy()


def _research_score(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    score = 0.2
    if "passes_sharpe_gate" in frame and frame["passes_sharpe_gate"].astype(bool).any():
        score += 0.3
    elif _max_numeric(frame, "sharpe") >= 1.75:
        score += 0.3
    if _max_numeric(frame, "returns_total") > 0.10:
        score += 0.3
    if any(column in frame for column in ("pair", "wizard_pair_id", "pair_id")):
        score += 0.2
    return round(min(score, 1.0), 4)


def _test_score(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    accepted = _truthy_column_any(frame, ("accepted", "passes_acceptance", "passed", "local_acceptance_passed"))
    if accepted:
        return 1.0
    score = 0.15
    if _max_numeric(frame, "profit_factor") >= 1.8:
        score += 0.25
    if _max_numeric(frame, "sharpe") >= 1.2:
        score += 0.25
    drawdown = _max_numeric(frame, "max_drawdown")
    if 0 < drawdown <= 0.15:
        score += 0.20
    if _max_numeric(frame, "closed_trades") >= 3 or _max_numeric(frame, "trades") >= 3:
        score += 0.15
    return round(min(score, 0.9), 4)


def _memory_score(learning: pd.DataFrame, specialist_agent: str) -> float:
    if learning.empty:
        return 0.0
    if "agent" in learning and learning["agent"].astype(str).eq(specialist_agent).any():
        return 0.6
    events = _max_numeric(learning, "events")
    return 0.2 if events > 0 else 0.0


def _rl_score(rl_acceptance: pd.DataFrame) -> float:
    if rl_acceptance.empty:
        return 0.0
    if "accepted" in rl_acceptance and rl_acceptance["accepted"].astype(bool).any():
        return 0.8
    return 0.2


def _decision(
    *,
    research_score: float,
    test_score: float,
    reference_score: float,
    wizard_rows: pd.DataFrame,
    local_rows: pd.DataFrame,
) -> tuple[str, str]:
    if reference_score <= 0:
        return "missing_reference_formula", "BLOCKED_REFERENCE"
    if wizard_rows.empty and local_rows.empty:
        return "missing_strategy_family_evidence", "FETCH_MORE_DATA"
    if local_rows.empty:
        return "missing_local_replay", "FETCH_MORE_DATA"
    if test_score >= 0.75 and research_score >= 0.5:
        return "", "PROMOTE_TESTING"
    if test_score >= 0.35 or research_score >= 0.5:
        return "needs_more_or_cleaner_evidence", "WATCH"
    return "weak_evidence", "REJECT"


def _reason(decision: str, blocker: str, wizard_rows: pd.DataFrame, local_rows: pd.DataFrame) -> str:
    if decision == "PROMOTE_TESTING":
        return "specialist_has_research_and_local_replay_evidence_but_no_acceptance_authority"
    if blocker:
        return blocker
    if wizard_rows.empty:
        return "no_wizard_discovery_rows_for_exact_mode"
    if local_rows.empty:
        return "wizard_discovery_exists_but_local_replay_missing"
    return "insufficient_after_cost_local_evidence"


def _next_step(decision: str, family: str) -> str:
    if decision == "PROMOTE_TESTING":
        return f"run broader local after-cost replay for {family} across candidate pairs and regimes"
    if decision == "FETCH_MORE_DATA":
        return f"capture Wizard exact mode and local history for {family}"
    if decision == "WATCH":
        return f"add more trades and red-team {family} for drawdown, thin trades, and venue mismatch"
    if decision == "BLOCKED_REFERENCE":
        return f"add formula reference for {family}"
    return f"keep {family} as research-only until new evidence improves"


def _horizontal_scores(scoreboard: pd.DataFrame, learning: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for agent, score_column in (
        ("research_agent", "research_score"),
        ("memory_agent", "memory_score"),
        ("test_agent", "test_score"),
        ("reference_agent", "reference_score"),
        ("rl_agent", "rl_score"),
    ):
        rows.append(
            {
                "agent": agent,
                "average_score": round(float(scoreboard[score_column].mean()), 4) if not scoreboard.empty else 0.0,
                "covered_strategy_families": int(scoreboard[score_column].gt(0).sum()) if not scoreboard.empty else 0,
                "learning_events": _learning_events(learning, agent),
                "promotion_allowed": False,
                "role": _horizontal_role(agent),
            }
        )
    return pd.DataFrame(rows)


def _horizontal_role(agent: str) -> str:
    return {
        "research_agent": "find and rank hypotheses",
        "memory_agent": "record what each specialist learned",
        "test_agent": "verify ideas with local replay",
        "reference_agent": "keep formulas and field definitions explicit",
        "rl_agent": "suggest ideas and similar-pair fingerprints",
    }.get(agent, "support")


def _learning_events(learning: pd.DataFrame, agent: str) -> int:
    if learning.empty or "agent" not in learning:
        return 0
    matching = learning[learning["agent"].astype(str).str.contains(agent.replace("_agent", ""), case=False, regex=False)]
    if "events" in matching:
        return int(pd.to_numeric(matching["events"], errors="coerce").fillna(0).sum())
    return len(matching)


def _evidence_path(root: Path, wizard_rows: pd.DataFrame, local_rows: pd.DataFrame, formula_dictionary: Path) -> str:
    paths = []
    if not wizard_rows.empty:
        paths.append(root / "data" / "processed" / "wizard_evidence.csv")
    if not local_rows.empty:
        paths.append(root / "reports" / "active" / "binance_exact_mode_strategy_sweep_2026-06-25.csv")
    if formula_dictionary.exists():
        paths.append(formula_dictionary)
    return ";".join(str(path) for path in paths)


def _max_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return round(float(values.max()), 6)


def _truthy_column_any(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    for column in columns:
        if column in frame and frame[column].astype(str).str.lower().isin({"true", "1", "yes", "pass", "passed", "accepted"}).any():
            return True
    return False


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _scoreboard_markdown(scoreboard: pd.DataFrame, horizontal: pd.DataFrame) -> str:
    lines = [
        "# Specialist Strategy Scoreboard",
        "",
        "This report gives each strategy family its own specialist lane while keeping promotion authority outside the specialist score.",
        "A high score means run more local testing, not accept a strategy for trading.",
        "",
        "## Strategy Specialists",
        "",
    ]
    if scoreboard.empty:
        lines.append("No specialist rows were generated.")
    else:
        columns = ["strategy_family", "combined_score", "decision", "blocker", "next_step"]
        lines.append(scoreboard[columns].to_markdown(index=False))
    lines.extend(["", "## Horizontal Agents", ""])
    if horizontal.empty:
        lines.append("No horizontal agent rows were generated.")
    else:
        lines.append(horizontal[["agent", "average_score", "covered_strategy_families", "role"]].to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Specialist rows cannot promote a pair or strategy by themselves.",
            "- Crypto Wizards evidence stays discovery-only.",
            "- RL evidence stays idea-only until local after-cost replay proves it.",
            "- Missing evidence creates a blocker or next step instead of a hidden score.",
        ]
    )
    return "\n".join(lines) + "\n"
