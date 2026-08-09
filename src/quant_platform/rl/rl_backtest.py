from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.rl.features import (
    attach_copula_dashboard_features,
    build_rl_feature_frame,
    leakage_columns,
    write_feature_schema,
)
from quant_platform.rl.rl_acceptance import return_summary, rl_acceptance_report


def run_rl_research(root: Path = ROOT, pair_id: str = "") -> CommandResult:
    reports = root / "reports" / "rl"
    dashboard = root / "reports" / "dashboard"
    models = root / "models" / "rl"
    reports.mkdir(parents=True, exist_ok=True)
    dashboard.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    dataset_path = root / "data" / "ml" / "trade_training_dataset.csv"
    dataset = _read_csv(dataset_path)
    copula_journal_path = root / "reports" / "active" / "wizard_research_journal.csv"
    copula_journal = _read_csv(copula_journal_path)
    dataset, copula_join_audit = attach_copula_dashboard_features(dataset, copula_journal)
    if pair_id and not dataset.empty and "pair" in dataset.columns:
        dataset = dataset[dataset["pair"].astype(str).str.replace("/", "-").str.contains(pair_id.replace("/", "-"), case=False, regex=False)]
        if not copula_join_audit.empty:
            copula_join_audit = copula_join_audit[copula_join_audit["pair"].astype(str).str.replace("/", "-").str.contains(pair_id.replace("/", "-"), case=False, regex=False)]
    blocker = ""
    if dataset.empty:
        blocker = "missing_trade_dataset"
    leaked = leakage_columns(dataset.columns) if not dataset.empty else []

    paths = {
        "training_report": reports / "rl_training_report.csv",
        "evaluation_report": reports / "rl_evaluation_report.csv",
        "execution_backtest": reports / "rl_execution_backtest.csv",
        "acceptance_report": reports / "rl_acceptance_report.csv",
        "blocked_actions": reports / "rl_blocked_actions.csv",
        "leakage_audit": reports / "rl_leakage_audit.csv",
        "split_audit": reports / "rl_split_audit.csv",
        "copula_dashboard_join_audit": reports / "rl_copula_dashboard_join_audit.csv",
        "feature_schema": models / "feature_schema.json",
        "dashboard_research_status": dashboard / "rl_research_status.csv",
        "dashboard_acceptance": dashboard / "rl_acceptance_report.csv",
        "dashboard_blocked_actions": dashboard / "rl_blocked_actions.csv",
        "training_report_json": models / "training_report.json",
        "acceptance_report_json": models / "acceptance_report.json",
    }
    write_feature_schema(paths["feature_schema"])
    join_statuses = copula_join_audit.get("join_status", pd.Series(dtype=str)).value_counts().to_dict()
    leakage_audit = pd.DataFrame(
        [
            {
                "feature_source_rows": len(dataset),
                "excluded_label_columns": ";".join(sorted(leaked)),
                "uses_future_data": False,
                "leakage_blocker": "",
                "copula_dashboard_join_statuses": json.dumps(join_statuses, sort_keys=True),
                "copula_dashboard_requires_point_in_time_snapshot": True,
                "copula_dashboard_return_or_sharpe_used": False,
                "evidence_path": f"{dataset_path};{copula_journal_path}",
            }
        ]
    )
    if blocker:
        blocked = _blocked_frame(blocker, pair_id)
        training = pd.DataFrame([{"status": "blocked", "blocker": blocker, "live_enabled": False, "rows": len(dataset)}])
        evaluation = pd.DataFrame(columns=["variant", "evaluation_split", "trades", "take_rate", "profit_factor", "sharpe", "max_drawdown", "total_return"])
        acceptance = rl_acceptance_report(evaluation)
        simulated = simulate_strategy_returns(dataset, {})
        per_trade_log = simulated["frame"]
        split_audit = pd.DataFrame(
            [{"split": "all", "status": "blocked", "blocker": blocker, "retained_rows": 0, "global_label_purge": True}]
        )
    else:
        # Imported lazily to avoid a module cycle: the learning agent uses this simulator.
        from quant_platform.rl.rl_learning_agent import _chronological_rl_partitions

        feature_source = dataset.drop(columns=leaked, errors="ignore")
        features = build_rl_feature_frame(feature_source)
        ordered, partitions, split_audit = _chronological_rl_partitions(dataset)
        split_ready = bool(not split_audit.empty and split_audit["status"].eq("ready").all())
        calibration = partitions.get("train", pd.DataFrame()) if split_ready else ordered
        policy_plan = _build_policy(calibration)
        evaluation_rows: list[dict[str, object]] = []
        per_trade_frames: list[pd.DataFrame] = []
        splits = (("validation", "validation"), ("test", "held_out_test")) if split_ready else (("diagnostic", "diagnostic_full_sample"),)
        for source_split, evaluation_split in splits:
            source = partitions.get(source_split, pd.DataFrame()) if split_ready else ordered
            split_simulated = simulate_strategy_returns(source, policy_plan)
            rl_rows = split_simulated["frame"] if isinstance(split_simulated["frame"], pd.DataFrame) else pd.DataFrame()
            rl_rows["evaluation_split"] = evaluation_split
            per_trade_frames.append(rl_rows)
            rl_mask = split_simulated.get("active_mask", pd.Series(dtype=bool))
            if not isinstance(rl_mask, pd.Series) or rl_mask.empty:
                rl_mask = pd.Series([False] * len(rl_rows), index=rl_rows.index) if not rl_rows.empty else pd.Series(dtype=bool)
            baseline = return_summary(
                "non_rl_baseline",
                source,
                _return_column(source),
                len(source),
                source_frame=source,
            )
            policy_summary = return_summary(
                "safe_rl_policy",
                rl_rows.loc[rl_mask],
                split_simulated["returns"],
                len(source),
                source_frame=source,
            )
            baseline["evaluation_split"] = evaluation_split
            policy_summary["evaluation_split"] = evaluation_split
            evaluation_rows.extend([baseline, policy_summary])
        evaluation = pd.DataFrame(evaluation_rows)
        acceptance = rl_acceptance_report(evaluation)
        simulated = simulate_strategy_returns(calibration, policy_plan)
        per_trade_log = pd.concat(per_trade_frames, ignore_index=True) if per_trade_frames else simulated["frame"]
        split_blocker = "" if split_ready else str(split_audit.get("blocker", pd.Series(["rl_chronological_split_not_ready"])).iloc[0])
        training = pd.DataFrame(
            [
                {
                    "status": "research_only",
                    "blocker": "rl_live_use_blocked",
                    "live_enabled": False,
                    "rows": len(dataset),
                    "features": features.shape[1],
                    "policy": policy_plan["policy_name"],
                    "policy_calibration_split": "globally_purged_train" if split_ready else "diagnostic_full_sample",
                    "split_ready": split_ready,
                    "split_blocker": split_blocker,
                    "train_rows": int(len(partitions.get("train", pd.DataFrame()))) if split_ready else 0,
                    "validation_rows": int(len(partitions.get("validation", pd.DataFrame()))) if split_ready else 0,
                    "held_out_test_rows": int(len(partitions.get("test", pd.DataFrame()))) if split_ready else 0,
                    "copula_dashboard_attached": int(join_statuses.get("attached", 0)),
                    "copula_dashboard_stale": int(join_statuses.get("stale_snapshot", 0)),
                }
            ]
        )
        blocked = _blocked_frame(str(acceptance.get("blocker", pd.Series(["rl_live_use_blocked"])).iloc[0]) or "rl_live_use_blocked", pair_id)

    leakage_audit["global_label_purge"] = bool(
        not split_audit.empty and split_audit.get("global_label_purge", pd.Series([False])).fillna(False).astype(bool).all()
    )
    leakage_audit["split_status"] = (
        "ready" if not split_audit.empty and split_audit.get("status", pd.Series(dtype=str)).eq("ready").all() else "blocked"
    )
    leakage_audit["split_evidence_path"] = str(paths["split_audit"])
    training.to_csv(paths["training_report"], index=False)
    evaluation.to_csv(paths["evaluation_report"], index=False)
    per_trade_log.to_csv(paths["execution_backtest"], index=False)
    acceptance.to_csv(paths["acceptance_report"], index=False)
    blocked.to_csv(paths["blocked_actions"], index=False)
    leakage_audit.to_csv(paths["leakage_audit"], index=False)
    split_audit.to_csv(paths["split_audit"], index=False)
    copula_join_audit.to_csv(paths["copula_dashboard_join_audit"], index=False)
    training.to_csv(paths["dashboard_research_status"], index=False)
    acceptance.to_csv(paths["dashboard_acceptance"], index=False)
    blocked.to_csv(paths["dashboard_blocked_actions"], index=False)
    paths["training_report_json"].write_text(json.dumps(training.iloc[0].to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    paths["acceptance_report_json"].write_text(json.dumps(acceptance.iloc[0].to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8")
    return CommandResult(paths=paths, summary={"rows": int(len(dataset)), "accepted": bool(acceptance.get("accepted", pd.Series([False])).iloc[0]), "blocker": str(acceptance.get("blocker", pd.Series([""])).iloc[0])})


def _return_column(frame: pd.DataFrame) -> pd.Series:
    for column in ["profit_after_cost", "realized_return", "trade_return", "return", "returns"]:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=frame.index)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def simulate_strategy_returns(frame: pd.DataFrame, policy: dict[str, object]) -> dict[str, object]:
    """Public policy simulator for research-grade RL experiments."""
    return _simulate_strategy_returns(frame, policy)


def _build_policy(frame: pd.DataFrame) -> dict[str, object]:
    data = frame.copy()
    zscores = _to_numeric(data.get("entry_abs_zscore", pd.Series(0.0, index=data.index)).fillna(0.0))
    hold_bars = _to_numeric(data.get("hold_bars", data.get("trade_bars", pd.Series(1000, index=data.index)))).fillna(1000)
    min_bars = float(hold_bars.replace([np.inf, -np.inf], np.nan).min() if not hold_bars.empty else 12.0)
    min_hold = max(1.0, min_bars) if math.isfinite(min_bars) else 12.0
    return {
        "policy_name": "simulated_quantile_hold_policy",
        "entry_threshold": float(zscores.quantile(0.70)) if not zscores.empty else 0.0,
        "max_position_fraction": 1.0,
        "min_hold_bars": int(min_hold),
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.12,
        "max_trade_drawdown_pct": 0.08,
        "session_loss_cap_pct": 0.15,
    }


def _simulate_strategy_returns(frame: pd.DataFrame, policy: dict[str, object]) -> dict[str, object]:
    if frame.empty:
        empty = pd.DataFrame(columns=["variant", "trade_id", "pair", "strategy_name", "timeframe", "strategy_id", "entry_bar_index", "exit_bar_index", "entry_signal_strength", "entry_side", "proposed_hold_bars", "actual_exit_bars", "base_return", "simulated_return", "policy_name", "simulation_reason", "exit_reason", "stop_triggered", "risk_cap_triggered", "entry_timestamp", "exit_timestamp"])
        return {"returns": pd.Series(dtype=float), "frame": empty, "active_mask": pd.Series(dtype=bool)}

    data = frame.copy()
    data["base_return"] = _return_column(data)
    data["entry_abs_zscore"] = _to_numeric(data.get("entry_abs_zscore", pd.Series(0.0, index=data.index)).fillna(0.0))
    data["trade_bars"] = _to_numeric(data.get("trade_bars", pd.Series(1, index=data.index)).fillna(1)).replace(0, 1)
    data["hold_bars"] = _to_numeric(data.get("hold_bars", data["trade_bars"]).fillna(data["trade_bars"]))
    data["entry_side"] = data.get("signal_side", pd.Series("long_spread", index=data.index)).astype(str).str.lower()
    data["entry_side"] = data["entry_side"].where(data["entry_side"].isin({"long_spread", "short_spread", "long", "short"}), "long")

    zscores = data["entry_abs_zscore"].clip(lower=0.0)
    entry_threshold = float(policy.get("entry_threshold", 0.0))
    stop_loss_pct = max(float(policy.get("stop_loss_pct", 0.05) or 0.05), 0.0)
    take_profit_pct = max(float(policy.get("take_profit_pct", 0.12) or 0.12), 0.0)
    max_trade_drawdown_pct = max(float(policy.get("max_trade_drawdown_pct", stop_loss_pct) or stop_loss_pct), 0.0)
    session_loss_cap_pct = max(float(policy.get("session_loss_cap_pct", 0.15) or 0.15), 0.0)
    z_cap = float(zscores.quantile(0.95) or 1.0) or 1.0
    position_fraction = min(max(float(policy.get("max_position_fraction", 1.0) or 1.0), 0.0), 1.0)
    hold_cap_pct = min(max(float(policy.get("hold_cap_pct", 1.0) or 1.0), 0.01), 1.0)
    volatility_penalty_weight = min(
        max(float(policy.get("volatility_penalty_weight", 0.35) or 0.35), 0.0),
        1.0,
    )

    strength_pct = (zscores / z_cap).clip(0.0, 1.0)
    proposed_hold = (data["hold_bars"] * (0.25 + 0.65 * strength_pct)).round().astype(float)
    proposed_hold = proposed_hold.where(proposed_hold > 0, 1)
    per_trade_hold_cap = (data["trade_bars"].clip(lower=1) * hold_cap_pct).apply(math.ceil).clip(lower=1)
    min_hold = max(float(policy.get("min_hold_bars", 1) or 1), 1.0)
    per_trade_min_hold = np.minimum(per_trade_hold_cap, min_hold)
    proposed_hold = np.minimum(np.maximum(proposed_hold, per_trade_min_hold), per_trade_hold_cap)
    threshold_active = zscores >= entry_threshold
    volatility_gate = _to_numeric(data.get("realized_volatility_percentile", pd.Series(0.0, index=data.index)).fillna(0.0)).fillna(0.0)
    vol_penalty = 1.0 - (volatility_gate.clip(0.0, 1.0) * volatility_penalty_weight)
    proposed_hold = (proposed_hold * vol_penalty).round().astype(int).clip(lower=1)
    # only hold for part of the trade horizon; "held_fraction" simulates early exits
    held_fraction = (proposed_hold / data["trade_bars"].replace(0, 1)).clip(0.0, 1.0)
    # profit_after_cost/realized_return already reflects strategy direction and costs.
    # Prorating that net outcome is conservative research proxying, not bar-path replay.
    simulated = data["base_return"] * held_fraction * position_fraction
    simulated[~threshold_active] = 0.0
    max_adverse = _to_numeric(data.get("max_adverse_excursion", pd.Series(0.0, index=data.index))).fillna(0.0).abs()
    max_favorable = _to_numeric(data.get("max_favorable_excursion", pd.Series(0.0, index=data.index))).fillna(0.0).abs()
    stop_triggered = threshold_active & (max_adverse >= stop_loss_pct)
    profit_triggered = threshold_active & ~stop_triggered & (max_favorable >= take_profit_pct)
    risk_cap_triggered = threshold_active & (max_adverse >= max_trade_drawdown_pct)
    simulated = simulated.clip(
        lower=-stop_loss_pct * position_fraction,
        upper=take_profit_pct * position_fraction,
    )
    simulated[stop_triggered] = -stop_loss_pct * position_fraction
    simulated[profit_triggered] = take_profit_pct * position_fraction
    session_time = pd.to_datetime(
        data.get(
            "entry_timestamp",
            data.get("feature_timestamp", pd.Series(pd.NaT, index=data.index)),
        ),
        utc=True,
        errors="coerce",
    )
    session_date = session_time.dt.strftime("%Y-%m-%d").fillna("unknown_date")
    session_pair = data.get("pair", pd.Series("unknown_pair", index=data.index)).astype(str)
    session_key = session_pair + "|" + session_date
    candidate_factor = 1.0 + simulated.where(threshold_active, 0.0)
    session_equity = candidate_factor.groupby(session_key, sort=False).cumprod()
    session_peak = session_equity.groupby(session_key, sort=False).cummax().clip(lower=1.0)
    session_drawdown = ((session_peak - session_equity) / session_peak.replace(0, np.nan)).fillna(0.0)
    session_breach = session_drawdown > session_loss_cap_pct
    # The trade that breaches a loss cap is realized; only later trades in the
    # same pair/day proxy session are blocked.
    session_blocked = session_breach.groupby(session_key, sort=False).shift(fill_value=False)
    session_blocked = session_blocked.groupby(session_key, sort=False).cummax().astype(bool)
    active = threshold_active & ~session_blocked
    simulated[session_blocked] = 0.0
    active_mask = pd.Series(active, index=data.index)
    exit_reason = pd.Series("threshold_exit", index=data.index, dtype="object")
    exit_reason = exit_reason.where(~profit_triggered, "take_profit")
    exit_reason = exit_reason.where(~risk_cap_triggered | stop_triggered, "drawdown_guard")
    exit_reason = exit_reason.where(~stop_triggered, "stop_loss")
    exit_reason = exit_reason.where(active, "not_entered")
    exit_reason = exit_reason.where(~session_blocked, "session_loss_cap")

    per_trade = pd.DataFrame(
        {
            "variant": "safe_rl_policy",
            "trade_id": data.get("trade_id", pd.Series(range(len(data)), index=data.index)),
            "pair": data.get("pair", ""),
            "strategy_name": data.get("strategy_name", ""),
            "timeframe": data.get("timeframe", ""),
            "strategy_id": data.get("strategy_id", ""),
            "entry_bar_index": _to_numeric(data.get("entry_bar_index", pd.Series(0, index=data.index))).fillna(0).astype(int),
            "exit_bar_index": data.get("exit_bar_index", pd.Series(0, index=data.index)).astype(str),
            "entry_signal_strength": zscores,
            "entry_side": data["entry_side"],
            "proposed_hold_bars": proposed_hold,
            "actual_exit_bars": (held_fraction * data["trade_bars"]).astype(int).clip(lower=1),
            "base_return": data["base_return"],
            "simulated_return": simulated,
            "policy_name": policy.get("policy_name", "simulated_quantile_hold_policy"),
            "simulation_reason": np.select(
                [session_blocked, ~threshold_active],
                ["session_loss_cap", "below_entry_threshold"],
                default="entered",
            ),
            "exit_reason": exit_reason,
            "stop_triggered": stop_triggered,
            "risk_cap_triggered": risk_cap_triggered | session_blocked,
            "return_basis": "net_after_cost_strategy_return",
            "cost_treatment": "no_second_cost_charge;proportional_net_return_proxy",
            "position_fraction": position_fraction,
            "hold_cap_pct": hold_cap_pct,
            "volatility_penalty_weight": volatility_penalty_weight,
            "session_scope": "pair_utc_day_proxy",
        }
    )
    return {
        "returns": simulated[active],
        "frame": per_trade,
        "active_mask": active_mask,
    }


def _blocked_frame(blocker: str, pair_id: str = "") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair": pair_id,
                "timeframe": "",
                "strategy": "rl_research",
                "regime": "",
                "rl_action": "blocked",
                "rl_reason": blocker,
                "blocker": blocker,
                "evidence_path": "reports/rl/rl_acceptance_report.csv",
                "live_enabled": False,
            }
        ]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
