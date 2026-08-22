"""Materialize seven-mode teacher and six-critic evidence from Hyperliquid."""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import promote_staged_file

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.backtest import BacktestResult, CostModel, FundingPolicy, backtest_two_leg_spread
from quant_platform.economic_contract import (
    ECONOMIC_CONTRACT_VERSION,
    action_for_signal,
    tail_actions,
    y_on_x_log_spread,
)
from quant_platform.orchestration.teacher_contracts import EXACT_MODES, MATH_V2, REQUIRED_CRITICS, ExactMode
from quant_platform.orchestration.hyperliquid_run_manifest import build_hyperliquid_run_manifest
from quant_platform.statistics.math_v2 import (
    estimate_hurst_dfa,
    fit_engle_granger,
    fit_ou,
    rolling_gaussian_copula_conditionals,
)
from quant_platform.wizard_mode_replay import build_local_mode_signal


ROOT = Path(__file__).resolve().parents[3]
FORMULA_VERSION = "local-seven-mode-math-v2.3-y-on-x"
SETTINGS_VERSION = "anchored-local-policy-v3-y-on-x"

MODE_REPLAY_NAMES: dict[ExactMode, str] = {
    ExactMode.STATIC_SPREAD: "Static (Spread)",
    ExactMode.STATIC_ZSCORER: "Static (ZScoreR)",
    ExactMode.DYN_SPREAD: "Dyn (Spread)",
    ExactMode.DYN_ZSCORER: "Dyn (ZScoreR)",
    ExactMode.OU_SPREAD: "OU (Spread)",
    ExactMode.OU_ZSCORER: "OU (ZScoreR)",
    ExactMode.COPULA: "Copula",
}


@dataclass(frozen=True)
class TeacherMaterializationPolicy:
    min_history_rows: int = 320
    test_rows: int = 100
    zscore_window: int = 60
    dynamic_window: int = 90
    copula_window: int = 120
    min_closed_trades: int = 3
    max_history_age_hours: float = 30.0
    max_drawdown: float = 0.35


def materialize_teacher_evidence(
    *,
    root: Path = ROOT,
    policy: TeacherMaterializationPolicy | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Create canonical adapter inputs from one common out-of-sample split."""

    policy = policy or TeacherMaterializationPolicy()
    now = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    output = root / "reports" / "orchestration" / "teacher_council"
    active.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    bundle_path = active / "hyperliquid_research_bundle.csv"
    cost_path = active / "workflow_hyperliquid_pair_cost_model.csv"
    inventory_path = active / "hyperliquid_testnet_market_inventory.csv"
    queue_path = active / "hyperliquid_wizard_hypothesis_queue.csv"
    marker_path = active / "math_v2_acceptance.json"
    teacher_path = active / "math_v2_teacher_inputs.csv"
    critic_path = active / "math_v2_critic_inputs.csv"
    teacher_coverage_path = output / "teacher_evidence_coverage.csv"
    critic_coverage_path = output / "critic_evidence_coverage.csv"
    next_actions_path = output / "teacher_evidence_next_actions.csv"
    summary_path = output / "teacher_evidence_materialization.md"

    manifest_result = build_hyperliquid_run_manifest(root=root, now=now)
    manifest_path = Path(manifest_result["manifest"])
    manifest_candidates_path = Path(manifest_result["candidates"])
    manifest_validation_path = Path(manifest_result["validation"])
    bundle = _read_csv(manifest_candidates_path)
    if not bundle.empty:
        bundle["interval"] = bundle.get("timeframe", "1d")
        bundle["history_rows"] = bundle.get("lookback", 0)
    costs = _read_csv(cost_path)
    inventory = _read_csv(inventory_path)
    queue = _read_csv(queue_path)
    family_controls_path = output / "research_family_selection_controls.csv"
    walkforward_path = family_controls_path if family_controls_path.exists() else output / "walkforward_mode_summary.csv"
    walkforward = _read_csv(walkforward_path)
    math_ready = _math_marker_passes(marker_path)

    teacher_rows: list[dict[str, object]] = []
    critic_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    if math_ready:
        for _, candidate in bundle.iterrows():
            teachers, critics, context = _materialize_context(
                candidate,
                costs=costs,
                inventory=inventory,
                queue=queue,
                walkforward=walkforward,
                root=root,
                policy=policy,
                now=now,
                source_paths=(
                    bundle_path,
                    cost_path,
                    inventory_path,
                    queue_path,
                    marker_path,
                    manifest_path,
                    manifest_candidates_path,
                    manifest_validation_path,
                    walkforward_path,
                ),
            )
            teacher_rows.extend(teachers)
            critic_rows.extend(critics)
            context_rows.append(context)

    teacher_frame = pd.DataFrame(teacher_rows)
    critic_frame = pd.DataFrame(critic_rows)
    _atomic_csv(teacher_frame, teacher_path)
    _atomic_csv(critic_frame, critic_path)
    _atomic_csv(teacher_frame, teacher_coverage_path)
    _atomic_csv(critic_frame, critic_coverage_path)
    next_actions = _next_actions(teacher_frame, critic_frame)
    _atomic_csv(next_actions, next_actions_path)

    complete_contexts = int(sum(row.get("teacher_rows") == 7 and row.get("critic_rows") == 6 for row in context_rows))
    eligible_teachers = int(
        teacher_frame.get("blockers", pd.Series(dtype=str)).fillna("").astype(str).str.strip().eq("").sum()
    ) if not teacher_frame.empty else 0
    critic_vetoes = int(
        critic_frame.get("verdict", pd.Series(dtype=str)).fillna("").astype(str).str.lower().eq("veto").sum()
    ) if not critic_frame.empty else 0
    status = "MATERIALIZED" if math_ready and complete_contexts > 0 else "BLOCKED"
    atomic_write_text(summary_path, _markdown(
            status=status,
            math_ready=math_ready,
            bundle_rows=len(bundle),
            context_rows=context_rows,
            teacher_rows=teacher_frame,
            critic_rows=critic_frame,
            eligible_teachers=eligible_teachers,
            critic_vetoes=critic_vetoes,
            next_actions=next_actions,
        ), encoding="utf-8")
    return {
        "status": status,
        "run_id": str(manifest_result.get("run_id", "")),
        "candidate_set_id": str(manifest_result.get("candidate_set_id", "")),
        "contexts": complete_contexts,
        "teacher_rows": len(teacher_frame),
        "critic_rows": len(critic_frame),
        "eligible_teachers": eligible_teachers,
        "critic_vetoes": critic_vetoes,
        "teacher_inputs": teacher_path,
        "critic_inputs": critic_path,
        "teacher_coverage": teacher_coverage_path,
        "critic_coverage": critic_coverage_path,
        "next_actions": next_actions_path,
        "summary": summary_path,
    }


def _materialize_context(
    candidate: pd.Series,
    *,
    costs: pd.DataFrame,
    inventory: pd.DataFrame,
    queue: pd.DataFrame,
    walkforward: pd.DataFrame,
    root: Path,
    policy: TeacherMaterializationPolicy,
    now: datetime,
    source_paths: tuple[Path, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    pair = _text(candidate.get("pair", ""))
    run_id = _text(candidate.get("run_id", ""))
    candidate_set_id = _text(candidate.get("candidate_set_id", ""))
    base_setup_identity = _text(candidate.get("setup_identity", ""))
    history_path = _resolve_path(_text(candidate.get("history_path", "")), root=root)
    if history_path is None or not history_path.exists():
        return [], [], {"pair": pair, "teacher_rows": 0, "critic_rows": 0, "blocker": "history_missing"}
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], [], {"pair": pair, "teacher_rows": 0, "critic_rows": 0, "blocker": "history_invalid_json"}
    frame = _clean_history(payload)
    if len(frame) < policy.min_history_rows:
        return [], [], {
            "pair": pair,
            "teacher_rows": 0,
            "critic_rows": 0,
            "blocker": f"history_rows<{policy.min_history_rows}",
        }

    test_rows = min(policy.test_rows, len(frame) - policy.min_history_rows + 1)
    split = len(frame) - test_rows
    train = frame.iloc[:split].copy()
    test = frame.iloc[split:].copy()
    history_hash = sha256(history_path.read_bytes()).hexdigest()
    cost_row = _pair_row(costs, pair)
    cost_model, cost_model_version = _cost_model(cost_row)
    latest = frame.index[-1].to_pydatetime()
    age_hours = (now - latest).total_seconds() / 3600.0
    history_fresh = 0.0 <= age_hours <= policy.max_history_age_hours
    snapshot_id = "teacher_snapshot_" + sha256(
        "|".join(
            [
                pair,
                history_hash,
                SETTINGS_VERSION,
                ECONOMIC_CONTRACT_VERSION,
                cost_model_version,
                latest.isoformat(),
            ]
        ).encode("utf-8")
    ).hexdigest()[:20]
    nominated_mode = _nominated_mode(queue, pair)

    engle_granger = fit_engle_granger(train["price_x"], train["price_y"])
    hedge_ratio = _fitted_hedge_ratio(engle_granger)
    if not math.isfinite(hedge_ratio) or hedge_ratio <= 0.0:
        return [], [], {
            "run_id": run_id,
            "candidate_set_id": candidate_set_id,
            "pair": pair,
            "teacher_rows": 0,
            "critic_rows": 0,
            "blocker": "positive_y_on_x_hedge_ratio_unavailable",
        }
    frame["hedge_ratio"] = hedge_ratio
    train["hedge_ratio"] = hedge_ratio
    test["hedge_ratio"] = hedge_ratio
    static_spread = y_on_x_log_spread(
        frame["price_x"], frame["price_y"], hedge_ratio
    )
    ou = fit_ou(static_spread.iloc[:split])
    hurst = estimate_hurst_dfa(static_spread.iloc[:split])
    copula = rolling_gaussian_copula_conditionals(
        frame["price_x"].pct_change(),
        frame["price_y"].pct_change(),
        window=policy.copula_window,
        min_rows=min(60, policy.copula_window),
    )
    frame["u1_given_u2"] = copula["u1_given_u2"]
    frame["u2_given_u1"] = copula["u2_given_u1"]

    shared_evidence = ";".join(
        _relative(path, root=root)
        for path in (*source_paths, history_path)
        if path.exists()
    )
    teacher_rows: list[dict[str, object]] = []
    mode_results: dict[ExactMode, BacktestResult] = {}
    for mode in EXACT_MODES:
        replay_name = MODE_REPLAY_NAMES[mode]
        settings, settings_blockers = _settings_for_mode(
            mode,
            frame=frame,
            split=split,
            hedge_ratio=hedge_ratio,
            ou=ou,
            policy=policy,
        )
        replay = build_local_mode_signal(frame, settings, exact_mode=replay_name)
        signal = replay.signal.iloc[split:].copy()
        result = backtest_two_leg_spread(test, signal, cost_model, interval=_text(candidate.get("interval", "1d")) or "1d")
        mode_results[mode] = result
        blockers = list(settings_blockers)
        walk_row = _mode_row(
            walkforward,
            pair=pair,
            mode=mode.value,
            timeframe=_text(candidate.get("interval", "1d")) or "1d",
        )
        walk_status = _text(walk_row.get("selection_status", ""))
        if walkforward.empty:
            blockers.append("walkforward_evidence_missing")
        elif walk_row.empty:
            blockers.append("walkforward_mode_missing")
        elif walk_status != "PASS":
            blockers.extend(_split_blockers(walk_row.get("selection_blocker", "walkforward_selection_not_passed")))
        if replay.mode_replay_status != "READY_FOR_RESEARCH_REPLAY":
            blockers.append("mode_replay_inputs_invalid")
            blockers.extend(replay.missing_inputs)
        if result.sharpe_status != "valid":
            blockers.append("sharpe_invalid")
        if result.trades < policy.min_closed_trades:
            blockers.append(f"closed_trades<{policy.min_closed_trades}")
        if result.expectancy_lower_95 is None:
            blockers.append("expectancy_lower_bound_unavailable")
        if not history_fresh:
            blockers.append("history_stale")
        if not _truthy(cost_row.get("cost_model_ready", False)):
            blockers.append("cost_model_not_ready")
        if not _truthy(cost_row.get("slippage_model_ready", False)):
            blockers.append("slippage_model_not_ready")
        blockers = list(dict.fromkeys(blockers))
        action = _teacher_action(float(signal.iloc[-1]) if not signal.empty else 0.0)
        holding_bars = _test_holding_bars(replay.trades, split=split)
        metric_last = _last_finite(replay.metric)
        teacher_rows.append(
            {
                "run_id": run_id,
                "candidate_set_id": candidate_set_id,
                "setup_id": _setup_id(snapshot_id, mode.value),
                "setup_identity": base_setup_identity,
                "mode_setup_identity": f"{base_setup_identity}|{_mode_slug(mode)}",
                "pair": pair,
                "venue": "hyperliquid",
                "timeframe": _text(candidate.get("interval", "1d")) or "1d",
                "lookback": len(frame),
                "source_snapshot_id": snapshot_id,
                "source_timestamp": latest.isoformat(),
                "exact_mode": mode.value,
                "proposed_action": action,
                "confidence": _confidence(result),
                "uncertainty": _uncertainty(result, blockers),
                "expected_net_return": result.expectancy if result.trades else np.nan,
                "lower_bound_net_return": result.expectancy_lower_95,
                "expected_holding_bars": holding_bars,
                "entry_style": _entry_style(settings, mode),
                "exit_style": _exit_style(settings, mode),
                "invalidation_condition": "dependency_break_or_parameter_instability",
                "required_regime": "non_crisis",
                "source_system": "hyperliquid_anchored_local_replay",
                "formula_version": FORMULA_VERSION,
                "math_version": MATH_V2,
                "mode_fidelity_status": "local_validated_estimator",
                "point_in_time_status": "confirmed",
                "history_hash": history_hash,
                "settings_version": SETTINGS_VERSION,
                "economic_contract_version": ECONOMIC_CONTRACT_VERSION,
                "cost_model_version": cost_model_version,
                "train_start": train.index[0].isoformat(),
                "train_end": train.index[-1].isoformat(),
                "test_start": test.index[0].isoformat(),
                "test_end": test.index[-1].isoformat(),
                "blockers": ";".join(blockers),
                "evidence_path": shared_evidence,
                "wizard_nominated": mode == nominated_mode,
                "mode_metric_name": replay.metric_name,
                "mode_metric_last": metric_last,
                "train_rows": len(train),
                "test_rows": len(test),
                "closed_trades": result.trades,
                "open_trades": result.open_trades,
                "profit_factor": result.profit_factor,
                "expectancy": result.expectancy,
                "expectancy_lower_95": result.expectancy_lower_95,
                "sharpe": result.sharpe,
                "max_drawdown": result.max_drawdown,
                "total_return": result.total_return,
                "reconciliation_error": result.reconciliation_error,
                "history_age_hours": age_hours,
                "cost_model_ready": _truthy(cost_row.get("cost_model_ready", False)),
                "slippage_model_ready": _truthy(cost_row.get("slippage_model_ready", False)),
                "walkforward_folds": _integer(walk_row.get("folds", 0)),
                "walkforward_total_trades": _integer(walk_row.get("total_trades", 0)),
                "walkforward_positive_folds": _integer(walk_row.get("positive_return_folds", 0)),
                "walkforward_bh_qvalue": _optional_number(walk_row.get("bh_qvalue")),
                "walkforward_selection_status": walk_status or "MISSING",
            }
        )

    critic_rows = _critic_rows(
        pair=pair,
        candidate=candidate,
        frame=frame,
        train=train,
        test=test,
        latest=latest,
        snapshot_id=snapshot_id,
        engle_granger=engle_granger,
        hurst=hurst,
        mode_results=mode_results,
        cost_row=cost_row,
        inventory=inventory,
        history_fresh=history_fresh,
        shared_evidence=shared_evidence,
        policy=policy,
    )
    walk_selection_passes = int(
        walkforward.loc[
            walkforward.get("pair", pd.Series("", index=walkforward.index)).astype(str).eq(pair),
            "selection_status",
        ].astype(str).eq("PASS").sum()
    ) if not walkforward.empty and "selection_status" in walkforward.columns else 0
    for critic in critic_rows:
        critic["run_id"] = run_id
        critic["candidate_set_id"] = candidate_set_id
        critic["setup_identity"] = base_setup_identity
        if critic.get("critic_type") == "outcome" and walk_selection_passes == 0:
            critic["verdict"] = "veto"
            critic["score"] = 0.0
            critic["reason"] = f"{critic.get('reason', '')};walkforward_selection_passes=0"
            existing = _split_blockers(critic.get("blocker_codes", ""))
            critic["blocker_codes"] = ";".join(dict.fromkeys([*existing, "no_walkforward_mode_passes_selection_controls"]))
    return teacher_rows, critic_rows, {
        "run_id": run_id,
        "candidate_set_id": candidate_set_id,
        "pair": pair,
        "teacher_rows": len(teacher_rows),
        "critic_rows": len(critic_rows),
        "blocker": "",
    }


def _settings_for_mode(
    mode: ExactMode,
    *,
    frame: pd.DataFrame,
    split: int,
    hedge_ratio: float,
    ou: Any,
    policy: TeacherMaterializationPolicy,
) -> tuple[dict[str, object], tuple[str, ...]]:
    spread = y_on_x_log_spread(frame["price_x"], frame["price_y"], hedge_ratio)
    ou_valid = ou.validity_status == "valid"
    ou_mu = float(ou.values.get("mu", spread.iloc[:split].mean()))
    innovation_sigma = float(
        ou.values.get("innovation_sigma", spread.iloc[:split].std(ddof=1))
    )
    phi = float(ou.values.get("phi", 0.0))
    denominator = math.sqrt(max(1.0 - phi**2, np.finfo(float).eps))
    ou_sigma = innovation_sigma / denominator
    if not math.isfinite(ou_sigma) or ou_sigma <= 0.0:
        ou_sigma = 1.0
    lower_action, upper_action = tail_actions(
        mode,
        copula_direction_view="u1_given_u2",
    )
    settings: dict[str, object] = {
        "capture_confirmed": True,
        "entry_long_operator": "<=",
        "entry_long_value": -2.0,
        "entry_long_position": lower_action.value,
        "entry_short_operator": ">=",
        "entry_short_value": 2.0,
        "entry_short_position": upper_action.value,
        "exit_long_operator": ">=",
        "exit_long_value": 0.0,
        "exit_short_operator": "<=",
        "exit_short_value": 0.0,
        "hedge_ratio": hedge_ratio,
        "zscore_window": policy.zscore_window,
        "dynamic_hedge_ratio_method": "rolling_ols_log_prices",
        "dynamic_hedge_ratio_window": policy.dynamic_window,
        "ou_mu": ou_mu,
        "ou_sigma": ou_sigma,
        "copula_family": "gaussian",
        "copula_signal_type": "conditional_cdf_tail",
        "copula_direction_view": "u1_given_u2",
        "copula_entry_lower": 0.10,
        "copula_entry_upper": 0.90,
        "copula_exit_lower": 0.45,
        "copula_exit_upper": 0.55,
        "economic_contract_version": ECONOMIC_CONTRACT_VERSION,
    }
    blockers: list[str] = []
    if mode in {ExactMode.STATIC_SPREAD, ExactMode.DYN_SPREAD}:
        preview = build_local_mode_signal(frame, settings, exact_mode=MODE_REPLAY_NAMES[mode])
        training_metric = pd.to_numeric(preview.metric.iloc[:split], errors="coerce").dropna()
        if len(training_metric) < 60:
            blockers.append("spread_threshold_training_rows<60")
        else:
            settings["entry_long_value"] = float(training_metric.quantile(0.10))
            settings["entry_short_value"] = float(training_metric.quantile(0.90))
            median = float(training_metric.median())
            settings["exit_long_value"] = median
            settings["exit_short_value"] = median
    if mode == ExactMode.OU_SPREAD:
        settings["entry_long_value"] = -2.0 * ou_sigma
        settings["entry_short_value"] = 2.0 * ou_sigma
    if mode in {ExactMode.OU_SPREAD, ExactMode.OU_ZSCORER} and not ou_valid:
        blockers.append(f"ou_fit_invalid:{ou.validity_reason}")
    if mode == ExactMode.COPULA:
        conditional = pd.to_numeric(
            frame.get("u1_given_u2", pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )
        if conditional.iloc[:split].dropna().empty:
            blockers.append("causal_copula_training_values_missing")
    return settings, tuple(blockers)


def _critic_rows(
    *,
    pair: str,
    candidate: pd.Series,
    frame: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    latest: datetime,
    snapshot_id: str,
    engle_granger: Any,
    hurst: Any,
    mode_results: dict[ExactMode, BacktestResult],
    cost_row: pd.Series,
    inventory: pd.DataFrame,
    history_fresh: bool,
    shared_evidence: str,
    policy: TeacherMaterializationPolicy,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    context = {
        "pair": pair,
        "venue": "hyperliquid",
        "timeframe": _text(candidate.get("interval", "1d")) or "1d",
        "lookback": len(frame),
        "source_snapshot_id": snapshot_id,
        "source_timestamp": latest.isoformat(),
        "authority": "local_point_in_time",
        "point_in_time_status": "confirmed",
        "evidence_path": shared_evidence,
    }

    if engle_granger.validity_status != "valid":
        rows.append(_critic(context, "dependency", "unknown", None, engle_granger.validity_reason, "dependency_fit_invalid"))
    else:
        pvalue = float(engle_granger.values["cointegration_pvalue"])
        verdict = "pass" if pvalue < 0.05 else "warn"
        rows.append(_critic(context, "dependency", verdict, 1.0 - min(pvalue, 1.0), f"Engle-Granger p={pvalue:.6g}", "" if verdict == "pass" else "cointegration_not_confirmed_5pct"))

    if hurst.validity_status != "valid":
        rows.append(_critic(context, "regime", "unknown", None, hurst.validity_reason, "hurst_fit_invalid"))
    else:
        hurst_value = float(hurst.values["hurst"])
        verdict = "pass" if hurst_value < 0.55 else "warn"
        rows.append(_critic(context, "regime", verdict, float(np.clip(1.0 - hurst_value, 0.0, 1.0)), f"training DFA H={hurst_value:.6g}", "" if verdict == "pass" else "weak_mean_reversion_regime"))

    finite_drawdowns = [result.max_drawdown for result in mode_results.values() if math.isfinite(result.max_drawdown)]
    best_drawdown = min(finite_drawdowns) if finite_drawdowns else float("inf")
    if best_drawdown <= policy.max_drawdown:
        risk_verdict, risk_blocker = "pass", ""
    elif math.isfinite(best_drawdown):
        risk_verdict, risk_blocker = "veto", "all_modes_drawdown_above_limit"
    else:
        risk_verdict, risk_blocker = "unknown", "risk_metrics_missing"
    rows.append(_critic(context, "risk", risk_verdict, float(np.clip(1.0 - best_drawdown, 0.0, 1.0)) if math.isfinite(best_drawdown) else None, f"best mode max drawdown={best_drawdown}", risk_blocker))

    cost_ready = _truthy(cost_row.get("cost_model_ready", False))
    slippage_ready = _truthy(cost_row.get("slippage_model_ready", False))
    funding_columns = {"funding_x_bps", "funding_y_bps"}
    funding_ready = funding_columns.issubset(frame.columns) and bool(frame[list(funding_columns)].notna().all(axis=1).mean() >= 0.95)
    cost_pass = cost_ready and slippage_ready and funding_ready
    cost_blockers = []
    if not cost_ready:
        cost_blockers.append("cost_model_not_ready")
    if not slippage_ready:
        cost_blockers.append("slippage_model_not_ready")
    if not funding_ready:
        cost_blockers.append("funding_coverage_below_95pct")
    rows.append(_critic(context, "cost", "pass" if cost_pass else "veto", 1.0 if cost_pass else 0.0, f"cost={cost_ready};slippage={slippage_ready};funding={funding_ready}", ";".join(cost_blockers)))

    asset_x = _text(candidate.get("asset_x", "")).upper()
    asset_y = _text(candidate.get("asset_y", "")).upper()
    tradable_assets = set(
        inventory.loc[inventory.get("tradable_perp", pd.Series(False, index=inventory.index)).map(_truthy), "asset"]
        .astype(str)
        .str.upper()
    ) if not inventory.empty and "asset" in inventory.columns else set()
    execution_pass = asset_x in tradable_assets and asset_y in tradable_assets and history_fresh
    execution_blockers = []
    if asset_x not in tradable_assets:
        execution_blockers.append(f"testnet_perp_missing:{asset_x}")
    if asset_y not in tradable_assets:
        execution_blockers.append(f"testnet_perp_missing:{asset_y}")
    if not history_fresh:
        execution_blockers.append("history_stale")
    rows.append(_critic(context, "execution", "pass" if execution_pass else "veto", 1.0 if execution_pass else 0.0, f"testnet legs={asset_x in tradable_assets}/{asset_y in tradable_assets};history_fresh={history_fresh}", ";".join(execution_blockers)))

    supported = [
        result
        for result in mode_results.values()
        if result.trades >= policy.min_closed_trades and result.expectancy_lower_95 is not None
    ]
    positive_lower = [result for result in supported if float(result.expectancy_lower_95 or 0.0) > 0.0]
    positive_mean = [result for result in supported if result.expectancy > 0.0]
    if positive_lower:
        outcome_verdict, outcome_blocker = "pass", ""
    elif positive_mean:
        outcome_verdict, outcome_blocker = "warn", "positive_mean_without_positive_lower_bound"
    else:
        outcome_verdict, outcome_blocker = "veto", "no_supported_positive_oos_mode"
    best_lower = max((float(result.expectancy_lower_95) for result in supported), default=float("nan"))
    rows.append(_critic(context, "outcome", outcome_verdict, float(np.clip(0.5 + best_lower * 5.0, 0.0, 1.0)) if math.isfinite(best_lower) else None, f"supported_modes={len(supported)};positive_lower={len(positive_lower)};best_lower={best_lower}", outcome_blocker))
    return rows


def _critic(
    context: dict[str, object],
    critic_type: str,
    verdict: str,
    score: float | None,
    reason: str,
    blockers: str,
) -> dict[str, object]:
    return {
        **context,
        "critic_type": critic_type,
        "verdict": verdict,
        "score": score,
        "reason": reason,
        "blocker_codes": blockers,
    }


def _clean_history(payload: dict[str, Any]) -> pd.DataFrame:
    history = pd.DataFrame(payload.get("history", []))
    required = {"timestamp", "price_x", "price_y"}
    if history.empty or not required.issubset(history.columns):
        return pd.DataFrame(columns=sorted(required))
    columns = [
        column
        for column in ("timestamp", "price_x", "price_y", "funding_x_bps", "funding_y_bps", "volume_x_usd", "volume_y_usd")
        if column in history.columns
    ]
    frame = history[columns].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["price_x"] = pd.to_numeric(frame["price_x"], errors="coerce")
    frame["price_y"] = pd.to_numeric(frame["price_y"], errors="coerce")
    for column in ("funding_x_bps", "funding_y_bps", "volume_x_usd", "volume_y_usd"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "price_x", "price_y"])
    frame = frame[(frame["price_x"] > 0.0) & (frame["price_y"] > 0.0)]
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    return frame.set_index("timestamp", drop=False)


def _cost_model(row: pd.Series) -> tuple[CostModel, str]:
    fee = _number(row.get("taker_fee_bps", 5.0), 5.0)
    slippage = _number(row.get("pair_one_way_slippage_bps", 5.0), 5.0)
    execution_risk = _number(row.get("execution_risk_bps", 2.0), 2.0)
    version = _text(row.get("fee_profile_id", "hyperliquid_cost_fallback")) or "hyperliquid_cost_fallback"
    version += f"|notional={_text(row.get('leg_notional_usd', 'unknown'))}|slippage={slippage:.8g}"
    return CostModel(
        taker_fee_bps=fee,
        slippage_bps=slippage,
        execution_risk_bps=execution_risk,
        funding_bps_per_day=0.0,
        bars_per_day=1,
        funding_policy=FundingPolicy.CONSERVATIVE_ABSOLUTE_DRAG.value,
    ), version


def _fitted_hedge_ratio(result: Any) -> float:
    value = result.values.get("hedge_ratio") if result.validity_status == "valid" else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) and number > 0.0 else float("nan")


def _teacher_action(signal: float) -> str:
    return action_for_signal(signal).value


def _confidence(result: BacktestResult) -> float:
    trade_support = min(result.trades / 10.0, 1.0)
    profit_factor = 2.0 if math.isinf(result.profit_factor) else max(result.profit_factor, 0.0)
    pf_support = min(profit_factor / 2.0, 1.0)
    sharpe_support = float(np.clip(0.5 + result.sharpe / 6.0, 0.0, 1.0)) if math.isfinite(result.sharpe) else 0.0
    return float(np.clip(0.15 + 0.30 * trade_support + 0.30 * pf_support + 0.25 * sharpe_support, 0.05, 0.95))


def _uncertainty(result: BacktestResult, blockers: list[str]) -> float:
    sampling = 1.0 / math.sqrt(max(result.trades, 1))
    blocker_penalty = min(len(blockers) * 0.05, 0.30)
    return float(np.clip(sampling + blocker_penalty, 0.05, 0.95))


def _entry_style(settings: dict[str, object], mode: ExactMode) -> str:
    if mode == ExactMode.COPULA:
        return f"causal_gaussian_conditional_cdf:{settings['copula_entry_lower']}/{settings['copula_entry_upper']}"
    return f"anchored_threshold:{settings['entry_long_value']}/{settings['entry_short_value']}"


def _exit_style(settings: dict[str, object], mode: ExactMode) -> str:
    if mode == ExactMode.COPULA:
        return f"conditional_normalization:{settings['copula_exit_lower']}/{settings['copula_exit_upper']}"
    return f"mean_cross:{settings['exit_long_value']}/{settings['exit_short_value']}"


def _test_holding_bars(trades: list[dict[str, object]], *, split: int) -> float | None:
    values = [
        float(trade["bars_held"])
        for trade in trades
        if int(trade.get("start_i", -1)) >= split
        and trade.get("exit_reason") != "open_at_end_of_history"
        and trade.get("bars_held") not in (None, "")
    ]
    return float(np.mean(values)) if values else None


def _last_finite(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.iloc[-1]) if not values.empty else None


def _nominated_mode(queue: pd.DataFrame, pair: str) -> ExactMode | None:
    if queue.empty or not {"pair", "exact_mode"}.issubset(queue.columns):
        return None
    rows = queue.loc[queue["pair"].astype(str) == pair]
    if rows.empty:
        return None
    try:
        from quant_platform.orchestration.teacher_contracts import normalize_exact_mode

        return normalize_exact_mode(str(rows.iloc[0]["exact_mode"]))
    except ValueError:
        return None


def _pair_row(frame: pd.DataFrame, pair: str) -> pd.Series:
    if frame.empty or "pair" not in frame.columns:
        return pd.Series(dtype=object)
    rows = frame.loc[frame["pair"].astype(str) == pair]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def _mode_row(frame: pd.DataFrame, *, pair: str, mode: str, timeframe: str = "") -> pd.Series:
    if frame.empty or not {"pair", "exact_mode"}.issubset(frame.columns):
        return pd.Series(dtype=object)
    rows = frame.loc[frame["pair"].astype(str).eq(pair) & frame["exact_mode"].astype(str).eq(mode)]
    if timeframe and "timeframe" in rows.columns:
        rows = rows.loc[rows["timeframe"].astype(str).eq(timeframe)]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def _mode_slug(mode: ExactMode) -> str:
    return "_".join(mode.value.lower().replace("zscorer", "zscore_r").split())


def _split_blockers(value: object) -> list[str]:
    return [part.strip() for part in _text(value).split(";") if part.strip()]


def _integer(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _optional_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _setup_id(snapshot_id: str, mode: str) -> str:
    return "setup_" + sha256(f"{snapshot_id}|{mode}|{FORMULA_VERSION}".encode("utf-8")).hexdigest()[:20]


def _math_marker_passes(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "passed"
        and payload.get("math_version") == MATH_V2
        and payload.get("all_checks_passed") is True
        and payload.get("generated_by") == "quant_platform.math_v2_acceptance"
    )


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    promote_staged_file(temporary, path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _resolve_path(value: str, *, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _relative(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _number(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _text(value: object) -> str:
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _markdown(
    *,
    status: str,
    math_ready: bool,
    bundle_rows: int,
    context_rows: list[dict[str, object]],
    teacher_rows: pd.DataFrame,
    critic_rows: pd.DataFrame,
    eligible_teachers: int,
    critic_vetoes: int,
    next_actions: pd.DataFrame,
) -> str:
    blockers = (
        teacher_rows.get("blockers", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.split(";")
        .explode()
        .str.strip()
    ) if not teacher_rows.empty else pd.Series(dtype=str)
    blocker_counts = blockers[blockers.ne("")].value_counts().rename_axis("blocker").reset_index(name="rows")
    verdicts = critic_rows.get("verdict", pd.Series(dtype=str)).value_counts().rename_axis("verdict").reset_index(name="rows")
    lines = [
        "# Teacher Evidence Materialization",
        "",
        f"- Status: **{status}**",
        f"- Core Math V2 marker: `{math_ready}`",
        f"- Hyperliquid contexts discovered: `{bundle_rows}`",
        f"- Complete 7-teacher/6-critic contexts: `{sum(row.get('teacher_rows') == 7 and row.get('critic_rows') == 6 for row in context_rows)}`",
        f"- Teacher rows: `{len(teacher_rows)}`",
        f"- Teacher rows currently free of blockers: `{eligible_teachers}`",
        f"- Critic vetoes: `{critic_vetoes}`",
        "- Formula authority: `local_validated_estimator`, never `vendor_exact`.",
        "- Evaluation: one anchored out-of-sample split; broader rolling folds are still required for promotion.",
        "",
        "## Teacher Blockers",
        "",
        blocker_counts.to_markdown(index=False) if not blocker_counts.empty else "No teacher blockers.",
        "",
        "## Critic Verdicts",
        "",
        verdicts.to_markdown(index=False) if not verdicts.empty else "No critic evidence.",
        "",
        "## Ranked Next Actions",
        "",
        next_actions.to_markdown(index=False) if not next_actions.empty else "No next actions generated.",
        "",
    ]
    return "\n".join(lines)


def _next_actions(teachers: pd.DataFrame, critics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    blocker_series = teachers.get("blockers", pd.Series(dtype=str)).fillna("").astype(str)
    slippage_rows = int(blocker_series.str.contains("slippage_model_not_ready", regex=False).sum())
    thin_rows = int(blocker_series.str.contains("closed_trades<", regex=False).sum())
    lower_bound_rows = int(blocker_series.str.contains("expectancy_lower_bound_unavailable", regex=False).sum())
    dependency_rows = int(
        critics.loc[critics.get("critic_type", pd.Series(dtype=str)).eq("dependency"), "verdict"]
        .astype(str)
        .isin(["warn", "veto", "unknown"])
        .sum()
    ) if not critics.empty and "verdict" in critics.columns else 0
    regime_rows = int(
        critics.loc[critics.get("critic_type", pd.Series(dtype=str)).eq("regime"), "verdict"]
        .astype(str)
        .isin(["warn", "veto", "unknown"])
        .sum()
    ) if not critics.empty and "verdict" in critics.columns else 0
    outcome_vetoes = int(
        (
            critics.get("critic_type", pd.Series(dtype=str)).eq("outcome")
            & critics.get("verdict", pd.Series(dtype=str)).eq("veto")
        ).sum()
    ) if not critics.empty else 0
    if slippage_rows:
        rows.append(
            {
                "priority": 1,
                "action": "collect_repeated_l2_samples_for_every_active_pair_leg",
                "affected_rows": slippage_rows,
                "gate_unlocked": "cost_critic_and_teacher_cost_lineage",
                "reason": "one snapshot per leg cannot calibrate a stable p95 slippage model",
            }
        )
    if thin_rows or lower_bound_rows:
        rows.append(
            {
                "priority": 2,
                "action": "run_multi_fold_walk_forward_and_add_supported_timeframes",
                "affected_rows": max(thin_rows, lower_bound_rows),
                "gate_unlocked": "trade_count_and_expectancy_lower_bound",
                "reason": "do not lower thresholds merely to manufacture closed trades",
            }
        )
    if dependency_rows or regime_rows:
        rows.append(
            {
                "priority": 3,
                "action": "refresh_wizard_candidates_and_pre_rank_local_dependency_stability",
                "affected_rows": max(dependency_rows, regime_rows),
                "gate_unlocked": "dependency_and_regime_critics",
                "reason": "current candidates show weak cointegration or non-mean-reverting training spreads",
            }
        )
    if outcome_vetoes:
        rows.append(
            {
                "priority": 4,
                "action": "retain_negative_modes_as_labels_and_reject_current_setups",
                "affected_rows": outcome_vetoes,
                "gate_unlocked": "student_training_dataset_without_survivorship_bias",
                "reason": "failed out-of-sample modes are valuable negative examples, not promotion candidates",
            }
        )
    rows.append(
        {
            "priority": 5,
            "action": "materialize_candidate_trade_rows_with_action_propensities",
            "affected_rows": len(teachers),
            "gate_unlocked": "supervised_student_and_contextual_bandit_readiness",
            "reason": "learning remains blocked until every proposal and abstention has an auditable outcome row",
        }
    )
    return pd.DataFrame(rows).sort_values("priority").reset_index(drop=True)
