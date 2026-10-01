from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import ROOT, CommandResult
from quant_platform.orchestration.corrective_runtime import (
    atomic_append_text,
    atomic_write_csv,
    atomic_write_text,
)
from quant_platform.rl.features import attach_copula_dashboard_features
from quant_platform.rl.rl_acceptance import (
    MAXIMUM_CONCENTRATION,
    MINIMUM_TAKE_RATE,
    minimum_trade_count,
    return_summary,
)
from quant_platform.rl.rl_backtest import simulate_strategy_returns
from quant_platform.runtime_types import strict_bool


def run_rl_learning_cycle(
    root: Path = ROOT,
    *,
    pair_id: str = "",
    policy_candidates: int = 12,
) -> CommandResult:
    """Run a separate RL learning loop (magicka) over synthetic policy variants in research mode."""

    reports = root / "reports" / "rl"
    agents = root / "reports" / "agents"
    models = root / "models" / "rl" / "learning"
    reports.mkdir(parents=True, exist_ok=True)
    agents.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    dataset_path = root / "data" / "ml" / "trade_training_dataset.csv"
    dataset = _read_csv(dataset_path)
    copula_journal_path = root / "reports" / "active" / "wizard_research_journal.csv"
    dataset, copula_join_audit = attach_copula_dashboard_features(dataset, _read_csv(copula_journal_path))
    if pair_id and not dataset.empty and "pair" in dataset.columns:
        dataset = dataset[dataset["pair"].astype(str).str.replace("/", "-").str.contains(pair_id.replace("/", "-"), case=False, regex=False)]
        if not copula_join_audit.empty:
            copula_join_audit = copula_join_audit[
                copula_join_audit["pair"].astype(str).str.replace("/", "-").str.contains(pair_id.replace("/", "-"), case=False, regex=False)
            ]

    cycle_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path_summary = reports / "rl_learning_cycle_summary.csv"
    path_backtest = reports / "rl_learning_cycle_backtests.csv"
    path_best = reports / "rl_learning_cycle_best_policy.json"
    path_learning_ideas = reports / "rl_learning_ideas.csv"
    path_agent = agents / "rl_learning_agent_experiments.csv"
    path_log = root / "data" / "agent_memory" / "rl_learning_agent.jsonl"
    path_copula_audit = reports / "rl_learning_copula_dashboard_join_audit.csv"
    path_split_audit = reports / "rl_learning_split_audit.csv"
    atomic_write_csv(copula_join_audit, path_copula_audit, index=False)

    if dataset.empty:
        summary = pd.DataFrame(
            [
                {
                    "cycle_id": cycle_id,
                    "pair_filter": pair_id,
                    "status": "blocked",
                    "blocker": "missing_trade_dataset",
                    "candidate_count": 0,
                    "best_policy_name": "",
                    "best_policy_row": "",
                    "best_take_rate": 0.0,
                    "best_profit_factor": 0.0,
                    "best_total_return": 0.0,
                    "copula_dashboard_attached": 0,
                    "created_at": _now(),
                }
            ]
        )
        atomic_write_csv(summary, path_summary, index=False)
        atomic_write_csv(pd.DataFrame(columns=_learning_backtest_columns()), path_backtest, index=False)
        atomic_write_csv(pd.DataFrame(columns=_learning_idea_columns()), path_learning_ideas, index=False)
        atomic_write_csv(pd.DataFrame(columns=_learning_agent_columns()), path_agent, index=False)
        atomic_write_csv(pd.DataFrame(columns=_split_audit_columns()), path_split_audit, index=False)
        atomic_write_text(path_best, "{}", encoding="utf-8")
        _append_cycle_memory(path_log, cycle_id, "blocked", "missing_trade_dataset", pair_id)
        return CommandResult(
            paths={
                "summary": path_summary,
                "backtests": path_backtest,
                "best_policy": path_best,
                "learning_ideas": path_learning_ideas,
                "agent_experiments": path_agent,
                "copula_dashboard_join_audit": path_copula_audit,
                "split_audit": path_split_audit,
            },
            summary={"cycle_id": cycle_id, "status": "blocked", "policy_count": 0},
        )

    ordered_dataset, partitions, split_audit = _chronological_rl_partitions(dataset)
    atomic_write_csv(split_audit, path_split_audit, index=False)
    split_ready = bool(not split_audit.empty and split_audit["status"].eq("ready").all())
    split_blocker = "" if split_ready else _first_nonempty(split_audit.get("blocker", pd.Series(dtype=str)))
    validation = partitions.get("validation", pd.DataFrame()) if split_ready else ordered_dataset
    test = partitions.get("test", pd.DataFrame()) if split_ready else pd.DataFrame()
    evaluation_split = "validation" if split_ready else "diagnostic_full_sample"

    policy_grid = _policy_grid(min(policy_candidates, 20))
    policies_by_name = {str(policy["policy_name"]): policy for policy in policy_grid}
    evaluation_rows: list[dict[str, object]] = []
    per_policy_logs: list[pd.DataFrame] = []
    baseline_validation = return_summary(
        "validation_baseline",
        validation,
        _return_series(validation),
        len(validation),
        source_frame=validation,
    )

    for policy in policy_grid:
        simulated = simulate_strategy_returns(validation, policy)
        per_policy_log = simulated["frame"]
        per_policy_log = _add_policy_columns(per_policy_log, policy, cycle_id)
        per_policy_log["evaluation_split"] = evaluation_split
        per_policy_logs.append(per_policy_log)
        entered_mask = per_policy_log.get("simulation_reason", pd.Series([""] * len(per_policy_log))).eq("entered")
        summary = return_summary(
            policy["policy_name"],
            per_policy_log.loc[entered_mask],
            simulated["returns"],
            len(validation),
            source_frame=validation,
        )
        summary["pair_id"] = pair_id
        summary["cycle_id"] = cycle_id
        summary["policy_name"] = policy["policy_name"]
        summary["evaluation_split"] = evaluation_split
        summary["split_ready"] = split_ready
        summary["split_blocker"] = split_blocker
        summary["training_rows"] = int(len(partitions.get("train", pd.DataFrame())))
        summary["validation_rows"] = int(len(validation))
        summary["test_rows"] = int(len(test))
        summary["entry_threshold"] = policy.get("entry_threshold", "")
        summary["max_position_fraction"] = policy.get("max_position_fraction", "")
        summary["volatility_penalty_weight"] = policy.get("volatility_penalty_weight", "")
        summary["hold_cap_pct"] = policy.get("hold_cap_pct", "")
        summary["stop_loss_pct"] = policy.get("stop_loss_pct", "")
        summary["take_profit_pct"] = policy.get("take_profit_pct", "")
        summary["max_trade_drawdown_pct"] = policy.get("max_trade_drawdown_pct", "")
        summary["session_loss_cap_pct"] = policy.get("session_loss_cap_pct", "")
        summary["top_pairs_entered"] = _top_pairs_entered(per_policy_log, entered_mask)
        summary["winner"] = 0
        summary.update(_policy_gate_outcomes(summary, baseline_validation, len(validation), prefix="validation"))
        summary["status"] = "blocked"
        summary["live_enabled"] = False
        evaluation_rows.append(summary)

    backtests = pd.DataFrame(evaluation_rows)
    if backtests.empty:
        backtests = pd.DataFrame(columns=_learning_backtest_columns())

    best_row = _select_policy_candidate(backtests)
    if best_row.empty:
        best_payload: dict[str, object] = {"status": "blocked", "blocker": "no_candidate_generated", "cycle_id": cycle_id}
    else:
        top = best_row.iloc[0]
        backtests.loc[backtests.index == top.name, "winner"] = 1
        best_payload = top.to_dict()
        selected_policy = policies_by_name[str(top.get("policy_name", ""))]
        test_gate_passed = False
        test_log = pd.DataFrame()
        if split_ready and not test.empty:
            simulated_test = simulate_strategy_returns(test, selected_policy)
            test_log = _add_policy_columns(simulated_test["frame"], selected_policy, cycle_id)
            test_log["evaluation_split"] = "held_out_test"
            entered_test = test_log.get("simulation_reason", pd.Series([""] * len(test_log))).eq("entered")
            test_summary = return_summary(
                str(selected_policy["policy_name"]),
                test_log.loc[entered_test],
                simulated_test["returns"],
                len(test),
                source_frame=test,
            )
            baseline_test = return_summary(
                "test_baseline",
                test,
                _return_series(test),
                len(test),
                source_frame=test,
            )
            test_gates = _policy_gate_outcomes(test_summary, baseline_test, len(test), prefix="test")
            test_gate_passed = strict_bool(test_gates["test_eligible"])
            for key, value in test_summary.items():
                if key != "variant":
                    best_payload[f"test_{key}"] = value
            best_payload.update(test_gates)
            best_payload.update(
                {
                    "test_baseline_profit_factor": baseline_test["profit_factor"],
                    "test_baseline_sharpe": baseline_test["sharpe"],
                    "test_baseline_max_drawdown": baseline_test["max_drawdown"],
                }
            )
            per_policy_logs.append(test_log)

        validation_passed = strict_bool(top.get("validation_eligible", False)) and split_ready
        oos_validated = validation_passed and test_gate_passed
        best_payload["policy_selection_status"] = "OOS_VALIDATED" if oos_validated else "REJECTED"
        best_payload["status"] = "research_only" if oos_validated else "blocked"
        best_payload["blocker"] = "rl_live_use_blocked" if oos_validated else (
            split_blocker or "validation_or_held_out_test_gates_not_met"
        )
        best_payload["live_enabled"] = False
        best_payload["created_at"] = _now()
        backtests.loc[backtests.index == top.name, "status"] = best_payload["status"]
        backtests.loc[backtests.index == top.name, "policy_selection_status"] = best_payload["policy_selection_status"]
        backtests.loc[backtests.index == top.name, "blocker"] = best_payload["blocker"]
        for key, value in best_payload.items():
            if str(key).startswith("test_"):
                backtests.loc[backtests.index == top.name, key] = value
    best_payload.setdefault("created_at", _now())

    backtest_log = pd.concat(per_policy_logs, ignore_index=True) if per_policy_logs else pd.DataFrame(columns=_learning_backtest_columns())
    backtest_log = _coerce_numeric_cols(backtest_log)
    backtests["created_at"] = backtests.get("created_at", pd.Series(dtype=object)).fillna(_now())
    _write_json(path_best, best_payload)

    ideas = _build_learning_ideas(backtest_log, best_payload)
    atomic_write_csv(backtest_log, path_backtest, index=False)
    atomic_write_csv(backtests, path_summary, index=False)
    atomic_write_csv(ideas, path_learning_ideas, index=False)
    # keep an agent-facing artifact for queue tracing
    agent_frame = pd.DataFrame(
        [
            {
                "cycle_id": cycle_id,
                "pair_filter": pair_id,
                "candidate_policies": len(backtests),
                "winner_policy": best_payload.get("policy_name", ""),
                "winner_profit_factor": float(best_payload.get("profit_factor", 0.0) or 0.0),
                "status": best_payload.get("status", "blocked"),
                "blocker": best_payload.get("blocker", ""),
                "policy_selection_status": best_payload.get("policy_selection_status", "REJECTED"),
                "oos_test_passed": strict_bool(best_payload.get("test_eligible", False)),
                "split_ready": split_ready,
                "live_enabled": False,
                "generated_at": _now(),
                "copula_dashboard_attached": int(copula_join_audit.get("join_status", pd.Series(dtype=str)).eq("attached").sum()),
                "copula_dashboard_stale": int(copula_join_audit.get("join_status", pd.Series(dtype=str)).eq("stale_snapshot").sum()),
            }
        ]
    )
    atomic_write_csv(agent_frame, path_agent, index=False)
    _append_cycle_memory(
        path_log,
        cycle_id,
        "passed" if str(best_payload.get("policy_selection_status", "")) == "OOS_VALIDATED" else "failed",
        str(best_payload.get("blocker", "")) if str(best_payload.get("status", "")) != "ready" else "",
        pair_id,
        extra={
            "winner_policy": best_payload.get("policy_name", ""),
            "best_profit_factor": best_payload.get("profit_factor", 0.0),
            "policy_selection_status": best_payload.get("policy_selection_status", "REJECTED"),
            "split_ready": split_ready,
            "live_enabled": False,
        },
    )

    return CommandResult(
        paths={
            "summary": path_summary,
            "backtests": path_backtest,
            "best_policy": path_best,
            "learning_ideas": path_learning_ideas,
            "agent_experiments": path_agent,
            "copula_dashboard_join_audit": path_copula_audit,
            "split_audit": path_split_audit,
        },
        summary={
            "cycle_id": cycle_id,
            "status": str(best_payload.get("status", "blocked")),
            "policy_count": int(len(backtests)),
            "winner_policy": best_payload.get("policy_name", ""),
            "winner_profit_factor": float(best_payload.get("profit_factor", 0.0) or 0.0),
        },
    )


def _chronological_rl_partitions(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    minimum_rows: tuple[int, int, int] = (50, 30, 30),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    ordered = frame.copy()
    entry_column = next((name for name in ("feature_timestamp", "entry_timestamp") if name in ordered.columns), "")
    label_column = next((name for name in ("label_timestamp", "exit_timestamp") if name in ordered.columns), "")
    if not entry_column or not label_column:
        audit = _blocked_split_audit("missing_entry_or_label_timestamp")
        return ordered, {}, audit

    ordered["_rl_entry_time"] = pd.to_datetime(ordered[entry_column], utc=True, errors="coerce", format="mixed")
    ordered["_rl_label_time"] = pd.to_datetime(ordered[label_column], utc=True, errors="coerce", format="mixed")
    valid = ordered["_rl_entry_time"].notna() & ordered["_rl_label_time"].notna()
    valid &= ordered["_rl_entry_time"].lt(ordered["_rl_label_time"])
    if not bool(valid.all()):
        audit = _blocked_split_audit(
            "invalid_or_noncausal_entry_label_timestamps",
            source_rows=len(ordered),
            invalid_rows=int((~valid).sum()),
        )
        return ordered, {}, audit

    ordered = ordered.sort_values(["_rl_entry_time", "_rl_label_time"]).reset_index(drop=True)
    unique_entries = pd.Index(ordered["_rl_entry_time"].drop_duplicates().sort_values())
    if len(unique_entries) < 5:
        audit = _blocked_split_audit(
            "insufficient_unique_entry_timestamps_for_three_way_split",
            source_rows=len(ordered),
            unique_entry_timestamps=len(unique_entries),
        )
        return ordered, {}, audit

    train_boundary_index = min(max(1, int(len(unique_entries) * train_fraction)), len(unique_entries) - 2)
    test_boundary_index = min(
        max(train_boundary_index + 1, int(len(unique_entries) * (train_fraction + validation_fraction))),
        len(unique_entries) - 1,
    )
    validation_start = unique_entries[train_boundary_index]
    test_start = unique_entries[test_boundary_index]

    raw_masks = {
        "train": ordered["_rl_entry_time"].lt(validation_start),
        "validation": ordered["_rl_entry_time"].ge(validation_start) & ordered["_rl_entry_time"].lt(test_start),
        "test": ordered["_rl_entry_time"].ge(test_start),
    }
    retained_masks = {
        "train": raw_masks["train"] & ordered["_rl_label_time"].lt(validation_start),
        "validation": raw_masks["validation"] & ordered["_rl_label_time"].lt(test_start),
        "test": raw_masks["test"],
    }
    partitions = {
        name: ordered.loc[mask].drop(columns=["_rl_entry_time", "_rl_label_time"]).reset_index(drop=True)
        for name, mask in retained_masks.items()
    }

    counts = {name: len(partitions[name]) for name in ("train", "validation", "test")}
    blockers = [
        f"insufficient_{name}_rows_after_global_label_purge:{counts[name]}<{minimum}"
        for name, minimum in zip(("train", "validation", "test"), minimum_rows)
        if counts[name] < minimum
    ]
    status = "ready" if not blockers else "blocked"
    blocker = ";".join(blockers)
    rows: list[dict[str, object]] = []
    for name in ("train", "validation", "test"):
        retained = retained_masks[name]
        split_entries = ordered.loc[retained, "_rl_entry_time"]
        split_labels = ordered.loc[retained, "_rl_label_time"]
        rows.append(
            {
                "split": name,
                "status": status,
                "blocker": blocker,
                "source_rows": int(len(ordered)),
                "raw_split_rows": int(raw_masks[name].sum()),
                "retained_rows": int(retained.sum()),
                "purged_overlap_rows": int(raw_masks[name].sum() - retained.sum()),
                "unique_entry_timestamps": int(split_entries.nunique()),
                "entry_start": split_entries.min().isoformat() if not split_entries.empty else "",
                "entry_end": split_entries.max().isoformat() if not split_entries.empty else "",
                "label_end_max": split_labels.max().isoformat() if not split_labels.empty else "",
                "validation_start": validation_start.isoformat(),
                "test_start": test_start.isoformat(),
                "pair_count": int(partitions[name].get("pair", pd.Series(dtype=str)).astype(str).nunique()),
                "timeframe_count": int(partitions[name].get("timeframe", pd.Series(dtype=str)).astype(str).nunique()),
                "global_label_purge": True,
                "train_fraction": float(train_fraction),
                "validation_fraction": float(validation_fraction),
                "minimum_rows_contract": ";".join(
                    str(int(value)) for value in minimum_rows
                ),
                "selection_use": "policy_selection" if name == "validation" else ("untouched_evaluation" if name == "test" else "diagnostic_only"),
            }
        )
    return ordered.drop(columns=["_rl_entry_time", "_rl_label_time"]), partitions, pd.DataFrame(rows)


def _blocked_split_audit(blocker: str, **values: object) -> pd.DataFrame:
    rows = []
    for name in ("train", "validation", "test"):
        row = {
            "split": name,
            "status": "blocked",
            "blocker": blocker,
            "source_rows": int(values.get("source_rows", 0) or 0),
            "raw_split_rows": 0,
            "retained_rows": 0,
            "purged_overlap_rows": 0,
            "unique_entry_timestamps": int(values.get("unique_entry_timestamps", 0) or 0),
            "entry_start": "",
            "entry_end": "",
            "label_end_max": "",
            "validation_start": "",
            "test_start": "",
            "pair_count": 0,
            "timeframe_count": 0,
            "global_label_purge": True,
            "selection_use": "blocked",
            "invalid_rows": int(values.get("invalid_rows", 0) or 0),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _return_series(frame: pd.DataFrame) -> pd.Series:
    for column in ("profit_after_cost", "realized_return", "trade_return", "return", "returns"):
        if column in frame.columns:
            if not frame.columns.is_unique:
                return pd.Series(np.nan, index=frame.index, dtype=float)
            raw = frame[column]
            invalid_type = raw.map(lambda value: isinstance(value, (bool, np.bool_, complex, np.complexfloating)))
            return pd.to_numeric(raw.where(~invalid_type, np.nan), errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _policy_gate_outcomes(
    summary: dict[str, object],
    baseline: dict[str, object],
    total_rows: int,
    *,
    prefix: str,
) -> dict[str, object]:
    minimum_trades = minimum_trade_count(total_rows)
    def metric(row: dict[str, object], key: str) -> float:
        value = row.get(key)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            return float("nan")
        return float(value) if np.isfinite(value) else float("nan")

    checks = {
        "qualified_calendar_metrics": summary.get("metrics_status") == "qualified" and baseline.get("metrics_status") == "qualified",
        "profit_factor_improves": metric(summary, "profit_factor") > metric(baseline, "profit_factor"),
        "drawdown_not_worse": metric(summary, "max_drawdown") <= metric(baseline, "max_drawdown"),
        "sharpe_not_materially_worse": metric(summary, "sharpe") >= metric(baseline, "sharpe") - 0.25,
        "minimum_trades": metric(summary, "trades") >= minimum_trades,
        "minimum_take_rate": metric(summary, "take_rate") >= MINIMUM_TAKE_RATE,
        "pair_concentration": metric(summary, "pair_concentration") <= MAXIMUM_CONCENTRATION,
        "pair_pnl_concentration": metric(summary, "pair_pnl_concentration") <= MAXIMUM_CONCENTRATION,
        "timeframe_selection_concentration": metric(summary, "timeframe_concentration") <= MAXIMUM_CONCENTRATION,
        "timeframe_pnl_concentration": metric(summary, "timeframe_pnl_concentration") <= MAXIMUM_CONCENTRATION,
        "regime_concentration": metric(summary, "regime_concentration") <= MAXIMUM_CONCENTRATION,
        "regime_pnl_concentration": metric(summary, "regime_pnl_concentration") <= MAXIMUM_CONCENTRATION,
    }
    failed = [name for name, passed in checks.items() if not passed]
    output: dict[str, object] = {
        f"{prefix}_eligible": not failed,
        f"{prefix}_gate_count": int(sum(checks.values())),
        f"{prefix}_gate_failures": ";".join(failed),
        f"{prefix}_minimum_trades": minimum_trades,
        f"{prefix}_baseline_profit_factor": baseline.get("profit_factor", 0.0),
        f"{prefix}_baseline_sharpe": baseline.get("sharpe", 0.0),
        f"{prefix}_baseline_max_drawdown": baseline.get("max_drawdown", 0.0),
    }
    output.update({f"{prefix}_gate_{name}": bool(passed) for name, passed in checks.items()})
    return output


def _select_policy_candidate(backtests: pd.DataFrame) -> pd.DataFrame:
    if backtests.empty:
        return backtests.head(0)
    ranked = backtests.copy()
    ranked["_eligible_rank"] = ranked.get("validation_eligible", False).map(strict_bool).astype(int)
    ranked["_gate_rank"] = pd.to_numeric(ranked.get("validation_gate_count", 0), errors="coerce").fillna(0)
    ranked["_pf_rank"] = pd.to_numeric(ranked.get("profit_factor", 0.0), errors="coerce").replace([np.inf, -np.inf], 10.0).fillna(0.0).clip(upper=10.0)
    ranked["_sharpe_rank"] = pd.to_numeric(ranked.get("sharpe", 0.0), errors="coerce").fillna(0.0)
    ranked["_drawdown_rank"] = pd.to_numeric(ranked.get("max_drawdown", 1.0), errors="coerce").fillna(1.0)
    ranked["_trades_rank"] = pd.to_numeric(ranked.get("trades", 0), errors="coerce").fillna(0)
    ranked = ranked.sort_values(
        ["_eligible_rank", "_gate_rank", "_pf_rank", "_sharpe_rank", "_drawdown_rank", "_trades_rank", "policy_name"],
        ascending=[False, False, False, False, True, False, True],
    )
    return backtests.loc[[ranked.index[0]]]


def _first_nonempty(series: pd.Series) -> str:
    values = series.fillna("").astype(str)
    values = values[values.str.strip().ne("")]
    return values.iloc[0] if not values.empty else "rl_chronological_split_not_ready"


def _policy_grid(max_policies: int) -> list[dict[str, object]]:
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    hold_caps = [0.15, 0.30, 0.45]
    volatility_weights = [0.15, 0.25, 0.35]
    stop_losses = [0.03, 0.05, 0.07]
    session_caps = [0.10, 0.15]
    dimensions = [thresholds, hold_caps, volatility_weights, stop_losses, session_caps]
    index_combinations = list(product(*(range(len(values)) for values in dimensions)))
    target_count = min(max_policies, len(index_combinations))
    threshold_seeds = [
        (
            index,
            index % len(hold_caps),
            (index // 2) % len(volatility_weights),
            (index * 2) % len(stop_losses),
            index % len(session_caps),
        )
        for index in range(len(thresholds))
    ]
    selected = threshold_seeds[:target_count]
    while len(selected) < target_count:
        remaining = [combo for combo in index_combinations if combo not in selected]
        best = max(
            remaining,
            key=lambda combo: (
                min(_normalized_grid_distance(combo, chosen, dimensions) for chosen in selected),
                combo,
            ),
        )
        selected.append(best)

    rows = []
    for policy_id, indexes in enumerate(selected, start=1):
        threshold, hold_cap, vol, stop_loss, session_cap = (
            dimensions[position][index] for position, index in enumerate(indexes)
        )
        rows.append(
            {
                "policy_name": f"learning_policy_{policy_id:03d}",
                "entry_threshold": float(threshold),
                "max_position_fraction": 1.0,
                "volatility_penalty_weight": float(vol),
                "hold_cap_pct": float(hold_cap),
                "stop_loss_pct": float(stop_loss),
                "take_profit_pct": float(max(stop_loss * 2.0, 0.08)),
                "max_trade_drawdown_pct": float(max(stop_loss * 1.25, stop_loss)),
                "session_loss_cap_pct": float(session_cap),
            }
        )
    return rows


def _normalized_grid_distance(
    left: tuple[int, ...],
    right: tuple[int, ...],
    dimensions: list[list[float]],
) -> float:
    return float(
        sum(
            ((left[index] - right[index]) / max(len(dimensions[index]) - 1, 1)) ** 2
            for index in range(len(dimensions))
        )
    )


def _add_policy_columns(frame: pd.DataFrame, policy: dict[str, object], cycle_id: str) -> pd.DataFrame:
    result = frame.copy()
    result["policy_name"] = str(policy.get("policy_name", "learning_policy"))
    result["policy_cycle_id"] = cycle_id
    result["learning_entry_threshold"] = policy.get("entry_threshold", 0.0)
    result["learning_hold_cap_pct"] = policy.get("hold_cap_pct", 0.0)
    result["learning_volatility_weight"] = policy.get("volatility_penalty_weight", 0.0)
    result["learning_stop_loss_pct"] = policy.get("stop_loss_pct", 0.0)
    result["learning_take_profit_pct"] = policy.get("take_profit_pct", 0.0)
    result["learning_session_loss_cap_pct"] = policy.get("session_loss_cap_pct", 0.0)
    return result


def _top_pairs_entered(frame: pd.DataFrame, entered_mask: pd.Series) -> str:
    if not isinstance(entered_mask, pd.Series):
        return ""
    entered = frame.loc[entered_mask]
    if entered.empty or "pair" not in entered.columns:
        return ""
    top = entered["pair"].astype(str).value_counts().head(5)
    return ";".join(f"{name}:{int(count)}" for name, count in top.items())


def _build_learning_ideas(log: pd.DataFrame, best_payload: dict[str, object]) -> pd.DataFrame:
    if log.empty:
        return pd.DataFrame(columns=_learning_idea_columns())
    cols = _learning_idea_columns()
    winner = str(best_payload.get("policy_name", ""))
    candidate = log[log.get("policy_name") == winner] if "policy_name" in log.columns else log.iloc[0:0]
    if candidate.empty:
        candidate = log
    if "evaluation_split" in candidate.columns:
        validation_only = candidate[candidate["evaluation_split"].astype(str).isin({"validation", "diagnostic_full_sample"})]
        if not validation_only.empty:
            candidate = validation_only
    candidate = candidate[candidate.get("simulation_reason", "") == "entered"].copy() if "simulation_reason" in candidate.columns else candidate
    if candidate.empty:
        return pd.DataFrame(columns=cols)
    candidate = candidate.sort_values("simulated_return", ascending=False)
    candidate = candidate.head(25)
    out = pd.DataFrame(
        {
            "policy_name": candidate.get("policy_name", pd.Series(dtype=object)),
            "pair": candidate.get("pair", pd.Series(dtype=object)),
            "strategy_name": candidate.get("strategy_name", pd.Series(dtype=object)),
            "simulated_return": candidate.get("simulated_return", pd.Series(dtype=object)),
            "base_return": candidate.get("base_return", pd.Series(dtype=object)),
            "entry_side": candidate.get("entry_side", pd.Series(dtype=object)),
            "entry_signal_strength": candidate.get("entry_signal_strength", pd.Series(dtype=object)),
            "proposed_hold_bars": candidate.get("proposed_hold_bars", pd.Series(dtype=object)),
            "actual_exit_bars": candidate.get("actual_exit_bars", pd.Series(dtype=object)),
            "exit_reason": candidate.get("exit_reason", pd.Series(dtype=object)),
            "stop_triggered": candidate.get("stop_triggered", pd.Series(dtype=object)),
            "idea_confidence": pd.Series([0.5] * len(candidate), index=candidate.index),
            "idea_type": "rl_learning_cycle_entry",
            "cycle_id": candidate.get("policy_cycle_id", pd.Series([_now()] * len(candidate), index=candidate.index)),
            "policy": winner,
            "authority": "research_only" if best_payload.get("policy_selection_status") == "OOS_VALIDATED" else "blocked_research_hypothesis",
            "evidence_split": candidate.get("evaluation_split", pd.Series(["diagnostic_full_sample"] * len(candidate), index=candidate.index)),
            "blocker": str(best_payload.get("blocker", "")),
            "generated_at": _now(),
        }
    )
    return out[cols]


def _coerce_numeric_cols(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    for column in [
        "simulated_return",
        "base_return",
        "entry_signal_strength",
        "proposed_hold_bars",
        "actual_exit_bars",
        "learning_entry_threshold",
        "learning_hold_cap_pct",
        "learning_volatility_weight",
        "learning_stop_loss_pct",
        "learning_take_profit_pct",
        "learning_session_loss_cap_pct",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _learning_backtest_columns() -> list[str]:
    return [
        "variant",
        "pair",
        "pair_id",
        "cycle_id",
        "policy_name",
        "policy_cycle_id",
        "trades",
        "take_rate",
        "profit_factor",
        "sharpe",
        "max_drawdown",
        "total_return",
        "entry_threshold",
        "max_position_fraction",
        "volatility_penalty_weight",
        "hold_cap_pct",
        "stop_loss_pct",
        "take_profit_pct",
        "max_trade_drawdown_pct",
        "session_loss_cap_pct",
        "pair_concentration",
        "timeframe_concentration",
        "created_at",
        "winner",
        "top_pairs_entered",
    ]


def _learning_idea_columns() -> list[str]:
    return [
        "policy_name",
        "pair",
        "strategy_name",
        "simulated_return",
        "base_return",
        "entry_side",
        "entry_signal_strength",
        "proposed_hold_bars",
        "actual_exit_bars",
        "exit_reason",
        "stop_triggered",
        "idea_confidence",
        "idea_type",
        "cycle_id",
        "policy",
        "authority",
        "evidence_split",
        "blocker",
        "generated_at",
    ]


def _learning_agent_columns() -> list[str]:
    return [
        "cycle_id",
        "pair_filter",
        "candidate_policies",
        "winner_policy",
        "winner_profit_factor",
        "status",
        "blocker",
        "policy_selection_status",
        "oos_test_passed",
        "split_ready",
        "live_enabled",
        "generated_at",
    ]


def _split_audit_columns() -> list[str]:
    return [
        "split",
        "status",
        "blocker",
        "source_rows",
        "raw_split_rows",
        "retained_rows",
        "purged_overlap_rows",
        "unique_entry_timestamps",
        "entry_start",
        "entry_end",
        "label_end_max",
        "validation_start",
        "test_start",
        "pair_count",
        "timeframe_count",
        "global_label_purge",
        "selection_use",
        "invalid_rows",
    ]


def _append_cycle_memory(
    memory_path: Path,
    cycle_id: str,
    outcome_label: str,
    blocker: str,
    pair_id: str,
    agent: str = "rl_learning_agent",
    task_id: str | None = None,
    *,
    extra: dict[str, object] | None = None,
) -> None:
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    task = task_id or f"{agent}:{cycle_id}"
    event: dict[str, object] = {
        "timestamp": _now(),
        "agent": agent,
        "task_id": task,
        "task_type": "run_rl_learning_cycle" if agent == "rl_learning_agent" else f"{agent}:run",
        "pair": pair_id,
        "action_taken": "learning_cycle_evaluated",
        "outcome_known": True,
        "outcome_label": outcome_label,
        "blocker": blocker,
        "learning_note": "rl_learning_cycle_completed",
        "confidence": 0.75,
        "next_step": "apply_best_policy_to_rl_idea_generation",
    }
    if extra:
        event.update(extra)
    atomic_append_text(memory_path, _json_dumps(event) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    atomic_write_text(path, _json_dumps(payload), encoding="utf-8")


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def run_magicka_learning_cycle(
    root: Path = ROOT,
    *,
    pair_id: str = "",
    policy_candidates: int = 12,
) -> CommandResult:
    """Shadow learner alias with its own product name: magicka."""
    return run_rl_learning_cycle(root=root, pair_id=pair_id, policy_candidates=policy_candidates)


def run_sequential_thinking_magicka(
    root: Path = ROOT,
    *,
    pair_id: str = "",
    policy_candidates: int = 12,
    max_recommendations: int = 12,
) -> CommandResult:
    """Sequential-thinking coach pass that proposes next RL experiments from previous gate outcomes."""

    reports = root / "reports" / "rl"
    agents = root / "reports" / "agents"
    reports.mkdir(parents=True, exist_ok=True)
    agents.mkdir(parents=True, exist_ok=True)

    dataset_path = root / "data" / "ml" / "trade_training_dataset.csv"
    dataset = _read_csv(dataset_path)
    cycle_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    acceptance_path = reports / "rl_acceptance_report.csv"
    model_acceptance_path = root / "reports" / "ml" / "model_gated_acceptance.csv"
    learning_summary_path = reports / "rl_learning_cycle_summary.csv"
    blocked_trades_path = root / "reports" / "dashboard" / "blocked_trades_dashboard.csv"
    recommendations_path = reports / "sequential_thinking_magicka_recommendations.csv"
    memory_path = agents / "sequential_thinking_magicka_experiments.csv"

    acceptance = _read_csv(acceptance_path)
    model_acceptance = _read_csv(model_acceptance_path)
    learning_summary = _read_csv(learning_summary_path)
    blocked_trades = _read_csv(blocked_trades_path)

    memory_events_path = root / "data" / "agent_memory" / "sequential_thinking_magicka.jsonl"
    if dataset.empty:
        summary = pd.DataFrame(
            [
                {
                    "cycle_id": cycle_id,
                    "pair_filter": pair_id,
                    "status": "blocked",
                    "blocker": "missing_trade_dataset",
                    "recommendation_count": 0,
                    "winner_policy": "",
                    "created_at": _now(),
                }
            ]
        )
        atomic_write_csv(summary, memory_path, index=False)
        atomic_write_csv(pd.DataFrame(columns=_seq_recommendation_columns()), recommendations_path, index=False)
        _append_cycle_memory(
            memory_events_path,
            cycle_id,
            "failed",
            "missing_trade_dataset",
            pair_id,
            agent="sequential_thinking_magicka",
        )
        return CommandResult(
            paths={
                "recommendations": recommendations_path,
                "sequential_magicka_memory": memory_path,
            },
            summary={"cycle_id": cycle_id, "status": "blocked", "recommendation_count": 0, "blocker": "missing_trade_dataset"},
        )

    rl_metrics = acceptance.iloc[0].to_dict() if not acceptance.empty else {}
    model_metrics = model_acceptance.iloc[0].to_dict() if not model_acceptance.empty else {}
    recommendations = _build_sequential_recommendations(
        rl_metrics=rl_metrics,
        model_metrics=model_metrics,
        blocked_trades=blocked_trades,
        learning_summary=learning_summary,
        pair_id=pair_id,
        max_recommendations=max_recommendations,
    )
    recommendations_df = pd.DataFrame(recommendations, columns=_seq_recommendation_columns())
    recommendations_df["cycle_id"] = cycle_id
    recommendations_df["created_at"] = _now()
    atomic_write_csv(recommendations_df, recommendations_path, index=False)

    best_policy = str(learning_summary.iloc[0].get("winner_policy", "")) if not learning_summary.empty else ""
    summary = pd.DataFrame(
        [
            {
                "cycle_id": cycle_id,
                "pair_filter": pair_id,
                "status": "ready",
                "blocker": "",
                "recommendation_count": int(len(recommendations_df)),
                "winner_policy": best_policy,
                "created_at": _now(),
            }
        ]
    )
    atomic_write_csv(summary, memory_path, index=False)

    _append_cycle_memory(
        memory_events_path,
        cycle_id,
        "passed" if recommendations else "passed_without_new_hypotheses",
        "",
        pair_id,
        agent="sequential_thinking_magicka",
        extra={
            "recommendation_count": int(len(recommendations_df)),
            "winner_policy": best_policy,
            "supporting_evidence": "reports/rl/rl_acceptance_report.csv;reports/ml/model_gated_acceptance.csv;reports/dashboard/blocked_trades_dashboard.csv",
        },
    )

    return CommandResult(
        paths={
            "recommendations": recommendations_path,
            "sequential_magicka_memory": memory_path,
        },
        summary={
            "cycle_id": cycle_id,
            "status": "ready",
            "recommendation_count": int(len(recommendations_df)),
            "winner_policy": best_policy,
        },
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seq_recommendation_columns() -> list[str]:
    return [
        "cycle_id",
        "pair_id",
        "focus_area",
        "observation",
        "proposed_change",
        "priority",
        "expected_effect",
        "supporting_metric",
        "task_id",
        "created_at",
    ]


def _build_sequential_recommendations(
    *,
    rl_metrics: dict[str, object],
    model_metrics: dict[str, object],
    blocked_trades: pd.DataFrame,
    learning_summary: pd.DataFrame,
    pair_id: str,
    max_recommendations: int,
) -> list[dict[str, object]]:
    recommendations: list[dict[str, object]] = []
    task_id = f"sequential_thinking_magicka:{pair_id or 'all'}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    def _as_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _as_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"true", "1", "yes"}

    rl_accepted = _as_bool(rl_metrics.get("accepted", False))
    model_accepted = _as_bool(model_metrics.get("accepted", False))
    if rl_metrics:
        pair_concentration = _as_float(rl_metrics.get("pair_concentration", 1.0))
        timeframe_concentration = _as_float(rl_metrics.get("timeframe_concentration", 1.0))
        take_rate = _as_float(rl_metrics.get("rl_take_rate", 0.0))
        blocker = str(rl_metrics.get("blocker", ""))
        if not rl_accepted:
            if pair_concentration > 0.65:
                recommendations.append(
                    {
                        "pair_id": pair_id,
                        "focus_area": "pair_diversity",
                        "observation": f"RL pair concentration above threshold ({pair_concentration:.3f}).",
                        "proposed_change": "Add a pair-cap regularizer in next run and expand top_pairs_entered target cap by 0.05.",
                        "priority": "high",
                        "expected_effect": "reduce regime overfit and improve generalization across pairs",
                        "supporting_metric": f"pair_concentration={pair_concentration:.3f};blocker={blocker or 'n/a'}",
                        "task_id": task_id,
                    }
                )
            if timeframe_concentration > 0.65:
                recommendations.append(
                    {
                        "pair_id": pair_id,
                        "focus_area": "timeframe_diversity",
                        "observation": f"RL timeframe concentration above threshold ({timeframe_concentration:.3f}).",
                        "proposed_change": "Diversify policy test pool across adjacent timeframes before filtering on concentration.",
                        "priority": "high",
                        "expected_effect": "reduce single-timeframe dependence and fragility",
                        "supporting_metric": f"timeframe_concentration={timeframe_concentration:.3f}",
                        "task_id": task_id,
                    }
                )
            if take_rate < 0.2:
                recommendations.append(
                    {
                        "pair_id": pair_id,
                        "focus_area": "entry_threshold",
                        "observation": f"RL take-rate below productive range ({take_rate:.3f}).",
                        "proposed_change": "Lower entry_threshold by 0.05 and raise hold_cap_pct in one candidate band.",
                        "priority": "medium",
                        "expected_effect": "increase signal throughput while preserving quality filters",
                        "supporting_metric": f"rl_take_rate={take_rate:.3f}",
                        "task_id": task_id,
                    }
                )

    if model_metrics:
        model_pf = _as_float(model_metrics.get("gated_profit_factor", 0.0))
        model_trades = _as_float(model_metrics.get("gated_trades", 0.0))
        model_drawdown = _as_float(model_metrics.get("gated_drawdown", 0.0))
        blocker = str(model_metrics.get("blocker", ""))
        if not model_accepted:
            if model_pf < 1.2:
                recommendations.append(
                    {
                        "pair_id": pair_id,
                        "focus_area": "profit_margin",
                        "observation": f"Model gated PF below gate ({model_pf:.3f}, blocker={blocker or 'n/a'}).",
                        "proposed_change": "Re-weight cost-aware features and disable low-confidence trade bins in train set.",
                        "priority": "high",
                        "expected_effect": "improve gated PF and incremental edge",
                        "supporting_metric": f"gated_profit_factor={model_pf:.3f};gated_drawdown={model_drawdown:.3f}",
                        "task_id": task_id,
                    }
                )
            if model_trades < 20 or model_drawdown > 0.3:
                recommendations.append(
                    {
                        "pair_id": pair_id,
                        "focus_area": "gating_pressure",
                        "observation": f"Model gating is too tight (gated_trades={int(model_trades)}, gated_drawdown={model_drawdown:.3f}).",
                        "proposed_change": "Use a staged threshold policy in next run (probability at 0.70 then 0.65 fallback).",
                        "priority": "medium",
                        "expected_effect": "raise tested trade count while controlling max drawdown",
                        "supporting_metric": f"gated_trades={int(model_trades)};gated_drawdown={model_drawdown:.3f}",
                        "task_id": task_id,
                    }
                )

    if not learning_summary.empty:
        blocked = str(learning_summary.iloc[0].get("blocker", ""))
        best_stop = _as_float(learning_summary.iloc[0].get("stop_loss_pct", 0.0))
        best_session_cap = _as_float(learning_summary.iloc[0].get("session_loss_cap_pct", 0.0))
        if blocked and blocked != "nan":
            recommendations.append(
                {
                    "pair_id": pair_id,
                    "focus_area": "learning_bootstrap",
                    "observation": f"Magicka cycle still blocked in latest run ({blocked}).",
                    "proposed_change": "Increase max policy candidates and test additional hold-cap buckets in next learning pass.",
                    "priority": "medium",
                    "expected_effect": "discover a cleaner candidate that passes gates first time",
                    "supporting_metric": f"magicka_blocker={blocked}",
                    "task_id": task_id,
                }
            )
        if best_stop > 0:
            recommendations.append(
                {
                    "pair_id": pair_id,
                    "focus_area": "risk_controls",
                    "observation": f"Latest learning sweep includes stop-loss={best_stop:.3f} and session-cap={best_session_cap:.3f}.",
                    "proposed_change": "Carry the best stop-loss and session-loss-cap settings into the next paper-validation batch and compare against a tighter stop band.",
                    "priority": "high" if best_stop >= 0.05 else "medium",
                    "expected_effect": "turn post-hoc drawdown control into explicit learned exit discipline",
                    "supporting_metric": f"stop_loss_pct={best_stop:.3f};session_loss_cap_pct={best_session_cap:.3f}",
                    "task_id": task_id,
                }
            )
    else:
        recommendations.append(
            {
                "pair_id": pair_id,
                "focus_area": "learning_bootstrap",
                "observation": "No recent magicka cycle summary is available.",
                "proposed_change": "Run magicka learning cycle first to create a fresh policy baseline.",
                "priority": "high",
                "expected_effect": "bootstrap policy-space exploration for coaching loop",
                "supporting_metric": "magicka_summary_rows=0",
                "task_id": task_id,
            }
        )

    if not blocked_trades.empty and "blocker" in blocked_trades.columns:
        top_blockers = (
            blocked_trades["blocker"].astype(str).value_counts().head(3).to_dict()
            if not blocked_trades.empty
            else {}
        )
        if "model_gate_not_accepted" in top_blockers:
            recommendations.append(
                {
                    "pair_id": pair_id,
                    "focus_area": "gate_alignment",
                    "observation": "Blocked trade set still dominated by model_gate_not_accepted.",
                    "proposed_change": "Prioritize policies that increase model coverage on evidence-rich pairs in next policy sweep.",
                    "priority": "high",
                    "expected_effect": "convert more candidates from PROMOTE candidate set to model-compatible candidates",
                    "supporting_metric": ";".join(f"{k}:{v}" for k, v in top_blockers.items()),
                    "task_id": task_id,
                }
            )

    unique_rows = []
    seen = set()
    for row in recommendations:
        key = (row["focus_area"], row["proposed_change"])
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
        if len(unique_rows) >= max_recommendations:
            break
    return unique_rows


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
