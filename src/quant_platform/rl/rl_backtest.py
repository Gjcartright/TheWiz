from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import ROOT, CommandResult
from quant_platform.orchestration.corrective_runtime import (
    atomic_copy_file,
    atomic_write_csv,
    atomic_write_text,
)
from quant_platform.rl.features import (
    attach_copula_dashboard_features,
    build_rl_feature_frame,
    leakage_columns,
    write_feature_schema,
)
from quant_platform.rl.rl_acceptance import return_summary, rl_acceptance_report
from quant_platform.runtime_types import strict_bool


def run_rl_research(
    root: Path = ROOT,
    pair_id: str = "",
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    minimum_rows: tuple[int, int, int] = (50, 30, 30),
    entry_threshold_quantile: float = 0.70,
) -> CommandResult:
    if (
        not 0.0 < float(train_fraction) < 1.0
        or not 0.0 < float(validation_fraction) < 1.0
        or float(train_fraction) + float(validation_fraction) >= 1.0
        or len(minimum_rows) != 3
        or any(int(value) <= 0 for value in minimum_rows)
        or not 0.0 <= float(entry_threshold_quantile) <= 1.0
    ):
        raise ValueError("invalid registered RL partition or calibration parameters")
    reports = root / "reports" / "rl"
    dashboard = root / "reports" / "dashboard"
    models = root / "models" / "rl"
    reports.mkdir(parents=True, exist_ok=True)
    dashboard.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    active_pointer_path = root / "data" / "ml" / "active_trade_dataset.json"
    active_pointer = _read_json(active_pointer_path)
    dataset_path = root / "data" / "ml" / "trade_training_dataset.csv"
    if str(active_pointer.get("status", "")) == "ACTIVE_RESEARCH_DATASET":
        pointed_path = Path(str(active_pointer.get("active_dataset_path", "")))
        dataset_path = pointed_path if pointed_path.is_absolute() else root / pointed_path
    expected_dataset_hash = str(active_pointer.get("active_dataset_sha256", ""))
    actual_dataset_hash = _sha256_file(dataset_path) if dataset_path.is_file() else ""
    dataset_id = str(active_pointer.get("dataset_id", ""))
    dataset_lineage_ready = bool(
        dataset_id
        and expected_dataset_hash
        and actual_dataset_hash == expected_dataset_hash
    )
    dataset_lineage = {
        "training_dataset_id": dataset_id,
        "training_dataset_sha256": actual_dataset_hash,
        "training_dataset_pointer": str(active_pointer_path.relative_to(root)),
        "dataset_lineage_ready": dataset_lineage_ready,
    }
    dataset = _read_csv(dataset_path)
    copula_journal_path = root / "reports" / "active" / "wizard_research_journal.csv"
    copula_journal = _read_csv(copula_journal_path)
    dataset, copula_join_audit = attach_copula_dashboard_features(dataset, copula_journal)
    if pair_id and not dataset.empty and "pair" in dataset.columns:
        dataset = dataset[dataset["pair"].astype(str).str.replace("/", "-").str.contains(pair_id.replace("/", "-"), case=False, regex=False)]
        if not copula_join_audit.empty:
            copula_join_audit = copula_join_audit[copula_join_audit["pair"].astype(str).str.replace("/", "-").str.contains(pair_id.replace("/", "-"), case=False, regex=False)]
    blocker = "" if dataset_lineage_ready else "rl_active_dataset_lineage_not_ready"
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
        "lineage_report_json": models / "rl_lineage.json",
    }
    superseded_receipt = _snapshot_rl_research_outputs(root, paths)
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
        ordered, partitions, split_audit = _chronological_rl_partitions(
            dataset,
            train_fraction=float(train_fraction),
            validation_fraction=float(validation_fraction),
            minimum_rows=tuple(int(value) for value in minimum_rows),
        )
        split_ready = bool(not split_audit.empty and split_audit["status"].eq("ready").all())
        calibration = partitions.get("train", pd.DataFrame()) if split_ready else ordered
        policy_plan = _build_policy(
            calibration,
            entry_threshold_quantile=float(entry_threshold_quantile),
            strength_calibration_source=(
                "globally_purged_training_partition" if split_ready else "diagnostic_full_sample"
            ),
        )
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
        not split_audit.empty and split_audit.get("global_label_purge", pd.Series([False])).map(strict_bool).all()
    )
    leakage_audit["split_status"] = (
        "ready" if not split_audit.empty and split_audit.get("status", pd.Series(dtype=str)).eq("ready").all() else "blocked"
    )
    leakage_audit["split_evidence_path"] = str(paths["split_audit"])
    for frame in (
        training,
        evaluation,
        per_trade_log,
        acceptance,
        blocked,
        leakage_audit,
        split_audit,
        copula_join_audit,
    ):
        for key, value in dataset_lineage.items():
            frame[key] = value
        frame["testnet_order_authority"] = False
        frame["live_trading_authorized"] = False
    atomic_write_csv(training, paths["training_report"], index=False)
    atomic_write_csv(evaluation, paths["evaluation_report"], index=False)
    atomic_write_csv(per_trade_log, paths["execution_backtest"], index=False)
    atomic_write_csv(acceptance, paths["acceptance_report"], index=False)
    blocked.to_csv(paths["blocked_actions"], index=False)
    atomic_write_csv(leakage_audit, paths["leakage_audit"], index=False)
    atomic_write_csv(split_audit, paths["split_audit"], index=False)
    atomic_write_csv(copula_join_audit, paths["copula_dashboard_join_audit"], index=False)
    atomic_write_csv(training, paths["dashboard_research_status"], index=False)
    atomic_write_csv(acceptance, paths["dashboard_acceptance"], index=False)
    blocked.to_csv(paths["dashboard_blocked_actions"], index=False)
    feature_schema = _read_json(paths["feature_schema"])
    feature_schema.update(
        {
            **dataset_lineage,
            "policy_action_inputs": [
                "entry_abs_zscore",
                "realized_volatility_percentile",
                "timeframe",
            ],
            "forbidden_policy_action_inputs": [
                "hold_bars",
                "trade_bars",
                "max_adverse_excursion",
                "max_favorable_excursion",
                "exit_timestamp",
            ],
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
    )
    atomic_write_text(paths["feature_schema"], json.dumps(feature_schema, indent=2, sort_keys=True), encoding="utf-8")
    lineage_report = {
        "schema_version": "thewiz.rl_research_lineage.v1",
        **dataset_lineage,
        "copula_journal_sha256": (
            _sha256_file(copula_journal_path)
            if copula_journal_path.is_file()
            else ""
        ),
        "superseded_rl_receipt": (
            str(superseded_receipt.relative_to(root))
            if superseded_receipt
            else ""
        ),
        "accepted": bool(
            acceptance.get("accepted", pd.Series([False])).iloc[0]
        ),
        "policy_action_uses_realized_hold_duration": False,
        "mae_mfe_exit_simulation": "conservative_retrospective_research_proxy",
        "registered_policy_parameters": {
            "train_fraction": float(train_fraction),
            "validation_fraction": float(validation_fraction),
            "minimum_rows": [int(value) for value in minimum_rows],
            "entry_threshold_quantile": float(entry_threshold_quantile),
        },
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    atomic_write_text(paths["lineage_report_json"], json.dumps(lineage_report, indent=2, sort_keys=True), encoding="utf-8")
    atomic_write_text(paths["training_report_json"], json.dumps(training.iloc[0].to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    atomic_write_text(paths["acceptance_report_json"], json.dumps(acceptance.iloc[0].to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8")
    return CommandResult(
        paths=paths,
        summary={
            "rows": int(len(dataset)),
            "training_dataset_id": dataset_id,
            "dataset_lineage_ready": dataset_lineage_ready,
            "accepted": bool(
                acceptance.get("accepted", pd.Series([False])).iloc[0]
            ),
            "blocker": str(
                acceptance.get("blocker", pd.Series([""])).iloc[0]
            ),
        },
    )


def _return_column(frame: pd.DataFrame) -> pd.Series:
    """Extract trade outcomes without manufacturing missing or invalid returns."""
    if not frame.columns.is_unique:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    for column in ["profit_after_cost", "realized_return", "trade_return", "return", "returns"]:
        if column in frame.columns:
            raw = frame[column]
            invalid_type = raw.map(
                lambda value: isinstance(value, (bool, np.bool_, complex, np.complexfloating))
            )
            return pd.to_numeric(raw.where(~invalid_type, np.nan), errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def simulate_strategy_returns(frame: pd.DataFrame, policy: dict[str, object]) -> dict[str, object]:
    """Public policy simulator for research-grade RL experiments."""
    return _simulate_strategy_returns(frame, policy)


def _build_policy(
    frame: pd.DataFrame,
    *,
    entry_threshold_quantile: float = 0.70,
    strength_calibration_source: str = "policy_calibration_frame",
) -> dict[str, object]:
    data = frame.copy()
    zscores = _to_numeric(data.get("entry_abs_zscore", pd.Series(0.0, index=data.index)).fillna(0.0))
    finite_zscores = zscores.replace([np.inf, -np.inf], np.nan).dropna()
    entry_threshold = (
        float(finite_zscores.quantile(entry_threshold_quantile))
        if not finite_zscores.empty
        else 0.0
    )
    strength_cap = float(finite_zscores.quantile(0.95)) if not finite_zscores.empty else 0.0
    return {
        "policy_name": "simulated_quantile_hold_policy",
        "entry_threshold": entry_threshold,
        "entry_threshold_calibration_quantile": float(entry_threshold_quantile),
        "zscore_strength_cap": max(strength_cap, entry_threshold, 1.0),
        "strength_calibration_source": strength_calibration_source,
        "max_position_fraction": 1.0,
        "min_hold_bars": 1,
        "target_hold_bars_by_timeframe": {
            "5m": 24,
            "15m": 16,
            "1h": 12,
            "4h": 6,
            "1d": 3,
        },
        "default_target_hold_bars": 3,
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
    default_cap = max(entry_threshold, 1.0)
    try:
        configured_cap = float(policy.get("zscore_strength_cap", default_cap))
    except (TypeError, ValueError):
        configured_cap = default_cap
    z_cap = max(configured_cap, default_cap) if math.isfinite(configured_cap) else default_cap
    position_fraction = min(max(float(policy.get("max_position_fraction", 1.0) or 1.0), 0.0), 1.0)
    hold_cap_pct = min(max(float(policy.get("hold_cap_pct", 1.0) or 1.0), 0.01), 1.0)
    volatility_penalty_weight = min(
        max(float(policy.get("volatility_penalty_weight", 0.35) or 0.35), 0.0),
        1.0,
    )

    strength_pct = (zscores / z_cap).clip(0.0, 1.0)
    hold_targets = policy.get("target_hold_bars_by_timeframe", {})
    if not isinstance(hold_targets, dict):
        hold_targets = {}
    timeframe = data.get("timeframe", pd.Series("", index=data.index)).astype(str)
    declared_hold = timeframe.map(
        lambda value: float(
            hold_targets.get(
                value,
                policy.get("default_target_hold_bars", 3),
            )
        )
    )
    proposed_hold = (
        declared_hold * (0.35 + 0.65 * strength_pct)
    ).round().clip(lower=1)
    per_trade_hold_cap = (data["trade_bars"].clip(lower=1) * hold_cap_pct).apply(math.ceil).clip(lower=1)
    min_hold = max(float(policy.get("min_hold_bars", 1) or 1), 1.0)
    proposed_hold = np.maximum(proposed_hold, min_hold)
    threshold_active = zscores >= entry_threshold
    volatility_gate = _to_numeric(data.get("realized_volatility_percentile", pd.Series(0.0, index=data.index)).fillna(0.0)).fillna(0.0)
    vol_penalty = 1.0 - (volatility_gate.clip(0.0, 1.0) * volatility_penalty_weight)
    proposed_hold = (proposed_hold * vol_penalty).round().astype(int).clip(lower=1)
    realized_hold_proxy = np.minimum(proposed_hold, per_trade_hold_cap)
    # only hold for part of the trade horizon; "held_fraction" simulates early exits
    held_fraction = (
        realized_hold_proxy / data["trade_bars"].replace(0, 1)
    ).clip(0.0, 1.0)
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
            "regime": data.get("regime", ""),
            "strategy_id": data.get("strategy_id", ""),
            "exact_mode": data.get("exact_mode", ""),
            "orientation": data.get("orientation", ""),
            "experiment_id": data.get("experiment_id", ""),
            "registered_contract_id": data.get(
                "registered_contract_id", ""
            ),
            "registered_execution_id": data.get(
                "registered_execution_id", ""
            ),
            "registered_semantic_hypothesis_id": data.get(
                "registered_semantic_hypothesis_id", ""
            ),
            "registered_hypothesis_outcome": data.get(
                "registered_hypothesis_outcome", ""
            ),
            "registered_candidate": data.get("registered_candidate", False),
            "accepted_stage4_survivor": data.get(
                "accepted_stage4_survivor", False
            ),
            "feature_timestamp": data.get("feature_timestamp", ""),
            "entry_timestamp": data.get("entry_timestamp", ""),
            "exit_timestamp": data.get("exit_timestamp", ""),
            "entry_bar_index": _to_numeric(data.get("entry_bar_index", pd.Series(0, index=data.index))).fillna(0).astype(int),
            "exit_bar_index": data.get("exit_bar_index", pd.Series(0, index=data.index)).astype(str),
            "entry_signal_strength": zscores,
            "zscore_strength_cap": z_cap,
            "strength_calibration_source": policy.get("strength_calibration_source", "fixed_policy_default"),
            "entry_side": data["entry_side"],
            "proposed_hold_bars": proposed_hold,
            "actual_exit_bars": realized_hold_proxy.astype(int).clip(lower=1),
            "base_return": data["base_return"],
            "simulated_return": simulated,
            "policy_name": policy.get("policy_name", "simulated_quantile_hold_policy"),
            "entry_threshold_calibration_quantile": policy.get(
                "entry_threshold_calibration_quantile", 0.70
            ),
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


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_rl_research_outputs(
    root: Path, paths: dict[str, Path]
) -> Path | None:
    acceptance = paths["acceptance_report"]
    if not acceptance.is_file():
        return None
    evidence_hash = _sha256_file(acceptance)
    snapshot_dir = root / "reports" / "rl" / "runs" / f"superseded_{evidence_hash[:20]}"
    receipt = snapshot_dir / "rl_run_receipt.json"
    if receipt.is_file():
        return receipt
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    preserved = []
    for name, source in paths.items():
        if not source.is_file():
            continue
        destination = snapshot_dir / f"{name}{source.suffix}"
        atomic_copy_file(source, destination, immutable=True)
        preserved.append(name)
    atomic_write_text(receipt, json.dumps(
            {
                "schema_version": "thewiz.rl_research_lineage.v1",
                "status": "SUPERSEDED_PRESERVED",
                "acceptance_sha256": evidence_hash,
                "artifacts": preserved,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
            indent=2,
            sort_keys=True,
        ), encoding="utf-8")
    return receipt
