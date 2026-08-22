"""Purged walk-forward and search-selection controls for seven-mode research."""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import promote_staged_file

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import math
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.backtest import backtest_two_leg_spread_with_ledger
from quant_platform.economic_contract import y_on_x_log_spread
from quant_platform.hyperliquid import build_hyperliquid_research_bundle
from quant_platform.orchestration.hyperliquid_run_manifest import load_hyperliquid_run_manifest
from quant_platform.orchestration.teacher_contracts import EXACT_MODES, ExactMode
from quant_platform.orchestration.teacher_evidence_materializer import (
    MODE_REPLAY_NAMES,
    TeacherMaterializationPolicy,
    _clean_history,
    _cost_model,
    _fitted_hedge_ratio,
    _read_csv,
    _resolve_path,
    _settings_for_mode,
)
from quant_platform.performance_math import MATH_VERSION
from quant_platform.statistics.math_v2 import (
    estimate_hurst_dfa,
    fit_engle_granger,
    fit_ou,
    rolling_gaussian_copula_conditionals,
)
from quant_platform.statistical_validation import (
    benjamini_hochberg,
    circular_block_bootstrap_mean,
)
from quant_platform.trade_ledger import TradeLedgerResult
from quant_platform.wizard_mode_replay import build_local_mode_signal


ROOT = Path(__file__).resolve().parents[3]
VALIDATION_VERSION = "purged-expanding-walkforward-v3-y-on-x"


@dataclass(frozen=True)
class WalkForwardPolicy:
    folds: int = 5
    min_train_rows: int = 320
    min_test_rows: int = 50
    embargo_rows: int = 1
    false_discovery_rate: float = 0.10
    min_total_trades: int = 10
    min_positive_folds: int = 3
    max_drawdown: float = 0.35
    max_hedge_ratio_cv: float = 0.35


def build_hyperliquid_walkforward_validation(
    *,
    root: Path = ROOT,
    policy: WalkForwardPolicy | None = None,
    candidate_path: Path | None = None,
    output_prefix: str = "walkforward",
) -> dict[str, object]:
    """Evaluate every canonical pair/mode on expanding, embargoed OOS folds."""

    policy = policy or WalkForwardPolicy()
    manifest = load_hyperliquid_run_manifest(root)
    if not output_prefix or Path(output_prefix).name != output_prefix:
        raise ValueError("output_prefix must be a plain filename prefix")
    candidates_path = candidate_path or root / "reports" / "active" / "hyperliquid_run_candidates.csv"
    costs_path = root / "reports" / "active" / "workflow_hyperliquid_pair_cost_model.csv"
    candidates = _prepare_validation_candidates(_read_csv(candidates_path), manifest=manifest, source_path=candidates_path)
    costs = _read_csv(costs_path)
    output = root / "reports" / "orchestration" / "teacher_council"
    output.mkdir(parents=True, exist_ok=True)
    folds_path = output / f"{output_prefix}_fold_evidence.csv"
    trades_path = output / f"{output_prefix}_trade_events.csv"
    summary_path = output / f"{output_prefix}_mode_summary.csv"
    selection_path = output / ("statistical_selection_controls.csv" if output_prefix == "walkforward" else f"{output_prefix}_selection_controls.csv")
    markdown_path = output / f"{output_prefix}_validation.md"

    fold_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    if manifest and not candidates.empty:
        for _, candidate in candidates.iterrows():
            candidate_folds, candidate_trades = _candidate_folds(
                candidate,
                costs=costs,
                root=root,
                policy=policy,
                trade_evidence_path=_relative(trades_path, root),
            )
            fold_rows.extend(candidate_folds)
            trade_rows.extend(candidate_trades)
    folds = pd.DataFrame(fold_rows, columns=_fold_columns())
    trades = pd.DataFrame(trade_rows, columns=_trade_event_columns())
    summary = _summarize_folds(folds, policy=policy, evidence_path=_relative(folds_path, root))
    selection = _selection_controls(summary, policy=policy, evidence_path=_relative(summary_path, root))
    if not summary.empty and not selection.empty:
        summary = summary.merge(
            selection[["run_id", "candidate_set_id", "pair", "exact_mode", "raw_pvalue", "bh_qvalue", "selection_status", "selection_blocker"]],
            on=["run_id", "candidate_set_id", "pair", "exact_mode"],
            how="left",
            validate="one_to_one",
        )
    _atomic_csv(folds, folds_path)
    _atomic_csv(trades, trades_path)
    _atomic_csv(summary, summary_path)
    _atomic_csv(selection, selection_path)
    atomic_write_text(markdown_path, _markdown(folds, summary, selection, policy), encoding="utf-8")
    status = "MATERIALIZED" if not folds.empty else "BLOCKED"
    accepted = int(selection.get("selection_status", pd.Series(dtype=str)).eq("PASS").sum()) if not selection.empty else 0
    return {
        "status": status,
        "fold_rows": len(folds),
        "trade_rows": len(trades),
        "mode_rows": len(summary),
        "selection_passes": accepted,
        "folds": folds_path,
        "trades": trades_path,
        "summary": summary_path,
        "selection": selection_path,
        "markdown": markdown_path,
    }


def build_hyperliquid_auxiliary_timeframe_validation(
    *,
    root: Path = ROOT,
    interval: str = "4h",
    intraday_days: int = 800,
    refresh: bool = True,
    policy: WalkForwardPolicy | None = None,
) -> dict[str, object]:
    """Build a local-only robustness lane without changing the daily manifest."""

    primary_path = root / "reports" / "active" / "hyperliquid_run_candidates.csv"
    primary = _read_csv(primary_path)
    pair_count = int(primary.get("pair", pd.Series(dtype=str)).dropna().astype(str).nunique())
    if pair_count <= 0:
        return {"status": "BLOCKED", "blocker": "canonical_hyperliquid_candidates_missing"}
    stem = f"hyperliquid_research_bundle_auxiliary_{interval}"
    bundle = build_hyperliquid_research_bundle(
        root=root,
        max_pairs=pair_count,
        intervals=(interval,),
        intraday_days=intraday_days,
        refresh=refresh,
        candidate_path=primary_path,
        output_stem=stem,
    )
    bundle_path = Path(bundle.paths["hyperliquid_research_bundle"])
    validation = build_hyperliquid_walkforward_validation(
        root=root,
        policy=policy,
        candidate_path=bundle_path,
        output_prefix=f"auxiliary_{interval}_walkforward",
    )
    return {**validation, "bundle": bundle_path, "bundle_summary": bundle.summary}


def build_hyperliquid_research_family_controls(
    *,
    root: Path = ROOT,
    policy: WalkForwardPolicy | None = None,
) -> dict[str, object]:
    """Control selection across timeframes and repeated, non-identical research looks."""

    policy = policy or WalkForwardPolicy()
    output = root / "reports" / "orchestration" / "teacher_council"
    daily_path = output / "statistical_selection_controls.csv"
    auxiliary_path = output / "auxiliary_4h_walkforward_selection_controls.csv"
    selection_path = output / "research_family_selection_controls.csv"
    registry_path = output / "research_family_registry.csv"
    markdown_path = output / "research_family_selection_controls.md"
    daily = _read_csv(daily_path)
    auxiliary = _read_csv(auxiliary_path)
    family_complete = not daily.empty and not auxiliary.empty
    frames: list[pd.DataFrame] = []
    for source, lane, path in (
        (daily, "primary_daily", daily_path),
        (auxiliary, "auxiliary_4h", auxiliary_path),
    ):
        if source.empty:
            continue
        frame = source.copy()
        frame["research_lane"] = lane
        frame["lane_evidence_path"] = _relative(path, root)
        frame = frame.rename(
            columns={
                "selection_status": "lane_selection_status",
                "selection_blocker": "lane_selection_blocker",
                "bh_qvalue": "lane_bh_qvalue",
                "false_discovery_rate": "lane_false_discovery_rate",
            }
        )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    source_fingerprint = _selection_source_fingerprint(daily_path, auxiliary_path)
    parent_candidate_set_id = _research_parent_candidate_set_id(daily, auxiliary)
    family_id = sha256(f"{parent_candidate_set_id}|{source_fingerprint}".encode("utf-8")).hexdigest()
    registry, attempt_number, alpha_spend = _register_research_family_attempt(
        registry_path=registry_path,
        family_id=family_id,
        parent_candidate_set_id=parent_candidate_set_id,
        source_fingerprint=source_fingerprint,
        base_alpha=policy.false_discovery_rate,
    )
    if combined.empty:
        controls = pd.DataFrame(columns=_research_family_columns())
    else:
        controls = combined.copy()
        controls["research_family_bh_qvalue"] = _benjamini_hochberg(
            pd.to_numeric(controls["raw_pvalue"], errors="coerce")
        )
        controls["research_family_id"] = family_id
        controls["parent_candidate_set_id"] = parent_candidate_set_id
        controls["research_attempt_number"] = attempt_number
        controls["research_family_tests"] = len(controls)
        controls["base_false_discovery_rate"] = policy.false_discovery_rate
        controls["alpha_spend_limit"] = alpha_spend
        controls["source_fingerprint"] = source_fingerprint
        controls["research_family_complete"] = family_complete
        final_status: list[str] = []
        final_blockers: list[str] = []
        for _, row in controls.iterrows():
            blockers = [
                blocker
                for blocker in _split_semicolon(row.get("lane_selection_blocker", ""))
                if blocker != "false_discovery_gate_failed"
            ]
            qvalue = _finite_or_nan(row.get("research_family_bh_qvalue"))
            if not math.isfinite(qvalue) or qvalue > alpha_spend:
                blockers.append("research_family_false_discovery_gate_failed")
            if not family_complete:
                blockers.append("research_family_incomplete")
            blockers = list(dict.fromkeys(blockers))
            final_status.append("PASS" if not blockers else "BLOCKED")
            final_blockers.append(";".join(blockers))
        controls["selection_status"] = final_status
        controls["selection_blocker"] = final_blockers
        controls["evidence_path"] = controls["lane_evidence_path"].astype(str) + ";" + _relative(registry_path, root)
        controls = controls.reindex(columns=_research_family_columns())
    _atomic_csv(controls, selection_path)
    _atomic_csv(registry, registry_path)
    primary_passes = int(
        controls.loc[controls.get("research_lane", pd.Series(dtype=str)).astype(str).eq("primary_daily"), "selection_status"]
        .astype(str)
        .eq("PASS")
        .sum()
    ) if not controls.empty else 0
    total_passes = int(controls.get("selection_status", pd.Series(dtype=str)).astype(str).eq("PASS").sum())
    atomic_write_text(markdown_path, "\n".join(
            [
                "# Hyperliquid Research-Family Selection Controls",
                "",
                f"- Research family: `{family_id}`",
                f"- Parent candidate set: `{parent_candidate_set_id}`",
                f"- Attempt number: `{attempt_number}`",
                f"- Tests controlled together: `{len(controls)}`",
                f"- Base FDR budget: `{policy.false_discovery_rate:.6f}`",
                f"- This-attempt alpha spend: `{alpha_spend:.6f}`",
                f"- Family complete: `{family_complete}`",
                f"- Primary selection passes: `{primary_passes}`",
                f"- Total selection passes: `{total_passes}`",
                "",
                "Daily and auxiliary tests share one Benjamini-Hochberg family. Repeated non-identical looks consume an idempotent alpha-spending budget.",
                "",
            ]
        ), encoding="utf-8")
    return {
        "status": "MATERIALIZED" if not controls.empty else "BLOCKED",
        "family_complete": family_complete,
        "research_family_id": family_id,
        "attempt_number": attempt_number,
        "alpha_spend_limit": alpha_spend,
        "tests": len(controls),
        "primary_selection_passes": primary_passes,
        "selection_passes": total_passes,
        "selection": selection_path,
        "registry": registry_path,
        "markdown": markdown_path,
    }


def _prepare_validation_candidates(
    candidates: pd.DataFrame,
    *,
    manifest: dict[str, object],
    source_path: Path,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    frame = candidates.copy()
    auxiliary = "auxiliary" in source_path.name
    source_hash = sha256(source_path.read_bytes()).hexdigest() if source_path.exists() else ""
    parent_run_id = str(manifest.get("run_id", ""))
    parent_candidate_set_id = str(manifest.get("candidate_set_id", ""))
    auxiliary_set_id = f"{parent_candidate_set_id}:aux:{source_hash[:12]}" if auxiliary else parent_candidate_set_id
    if "timeframe" not in frame.columns:
        frame["timeframe"] = frame.get("interval", "1d")
    if "lookback" not in frame.columns:
        frame["lookback"] = frame.get("history_rows", 0)
    if "run_id" not in frame.columns:
        frame["run_id"] = parent_run_id
    if "candidate_set_id" not in frame.columns:
        frame["candidate_set_id"] = auxiliary_set_id
    if "source_snapshot_id" not in frame.columns:
        frame["source_snapshot_id"] = f"snapshot_{source_hash[:20]}"
    if "setup_identity" not in frame.columns:
        frame["setup_identity"] = frame.apply(
            lambda row: "|".join(
                [
                    str(row.get("pair", "")).replace("/", "|"),
                    str(row.get("timeframe", "1d")).lower(),
                    str(row.get("lookback", 0)),
                    "auxiliary_all_modes" if auxiliary else "all_modes",
                ]
            ),
            axis=1,
        )
    frame["research_lane"] = "auxiliary_timeframe_robustness" if auxiliary else "canonical_daily"
    frame["wizard_timeframe_authority"] = "daily_pair_discovery_only" if auxiliary else "matching_daily_discovery"
    return frame


def _candidate_folds(
    candidate: pd.Series,
    *,
    costs: pd.DataFrame,
    root: Path,
    policy: WalkForwardPolicy,
    trade_evidence_path: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    history_path = _resolve_path(str(candidate.get("history_path", "")), root=root)
    if history_path is None or not history_path.exists():
        return [], []
    try:
        import json

        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], []
    frame = _clean_history(payload)
    boundaries = _fold_boundaries(len(frame), policy)
    if not boundaries:
        return [], []
    pair = str(candidate.get("pair", ""))
    cost_row = _pair_row(costs, pair)
    cost_model, cost_version = _cost_model(cost_row)
    rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    replay_policy = TeacherMaterializationPolicy(min_history_rows=policy.min_train_rows)
    for fold_number, train_end, test_start, test_end in boundaries:
        causal = frame.iloc[:test_end].copy()
        train = causal.iloc[:train_end].copy()
        test = causal.iloc[test_start:test_end].copy()
        dependency = fit_engle_granger(train["price_x"], train["price_y"])
        hedge_ratio = _fitted_hedge_ratio(dependency)
        if not math.isfinite(hedge_ratio) or hedge_ratio <= 0.0:
            continue
        causal["hedge_ratio"] = hedge_ratio
        train["hedge_ratio"] = hedge_ratio
        spread = y_on_x_log_spread(
            causal["price_x"], causal["price_y"], hedge_ratio
        )
        ou = fit_ou(spread.iloc[:train_end])
        hurst = estimate_hurst_dfa(spread.iloc[:train_end])
        copula = rolling_gaussian_copula_conditionals(
            causal["price_x"].pct_change(),
            causal["price_y"].pct_change(),
            window=replay_policy.copula_window,
            min_rows=min(60, replay_policy.copula_window),
        )
        causal["u1_given_u2"] = copula["u1_given_u2"]
        causal["u2_given_u1"] = copula["u2_given_u1"]
        test["hedge_ratio"] = hedge_ratio
        for mode in EXACT_MODES:
            settings, setting_blockers = _settings_for_mode(
                mode,
                frame=causal,
                split=train_end,
                hedge_ratio=hedge_ratio,
                ou=ou,
                policy=replay_policy,
            )
            replay = build_local_mode_signal(causal, settings, exact_mode=MODE_REPLAY_NAMES[mode])
            signal = replay.signal.iloc[test_start:test_end].copy()
            result, ledger = backtest_two_leg_spread_with_ledger(
                test,
                signal,
                cost_model,
                interval=str(candidate.get("timeframe", "1d")),
            )
            blockers = list(setting_blockers)
            if replay.mode_replay_status != "READY_FOR_RESEARCH_REPLAY":
                blockers.append("mode_replay_inputs_invalid")
            if dependency.validity_status != "valid":
                blockers.append("dependency_fit_invalid")
            setup_identity = f"{candidate.get('setup_identity', '')}|{_mode_slug(mode)}"
            rows.append(
                {
                    "run_id": str(candidate.get("run_id", "")),
                    "candidate_set_id": str(candidate.get("candidate_set_id", "")),
                    "setup_identity": setup_identity,
                    "pair": pair,
                    "venue": "hyperliquid",
                    "timeframe": str(candidate.get("timeframe", "1d")),
                    "exact_mode": mode.value,
                    "fold": fold_number,
                    "train_start": train.index[0].isoformat(),
                    "train_end": train.index[-1].isoformat(),
                    "test_start": test.index[0].isoformat(),
                    "test_end": test.index[-1].isoformat(),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "embargo_rows": policy.embargo_rows,
                    "hedge_ratio": hedge_ratio,
                    "cointegration_pvalue": _result_number(dependency, "cointegration_pvalue"),
                    "hurst": _result_number(hurst, "hurst"),
                    "cost_model_version": cost_version,
                    "trades": result.trades,
                    "profit_factor": result.profit_factor,
                    "sharpe": result.sharpe,
                    "expectancy": result.expectancy,
                    "expectancy_lower_95": result.expectancy_lower_95,
                    "max_drawdown": result.max_drawdown,
                    "total_return": result.total_return,
                    "reconciliation_error": result.reconciliation_error,
                    "blockers": ";".join(dict.fromkeys(blockers)),
                    "validation_version": VALIDATION_VERSION,
                    "evidence_path": ";".join(
                        [
                            str(candidate.get("history_path", "")),
                            "reports/active/hyperliquid_run_manifest.json",
                            "reports/active/workflow_hyperliquid_pair_cost_model.csv",
                        ]
                    ),
                }
            )
            trade_rows.extend(
                _trade_event_rows(
                    candidate=candidate,
                    mode=mode,
                    fold_number=fold_number,
                    setup_identity=setup_identity,
                    test=test,
                    replay_metric=replay.metric.iloc[test_start:test_end],
                    metric_name=replay.metric_name,
                    ledger=ledger,
                    history_path=history_path,
                    cost_version=cost_version,
                    blockers=blockers,
                    trade_evidence_path=trade_evidence_path,
                )
            )
    return rows, trade_rows


def _trade_event_rows(
    *,
    candidate: pd.Series,
    mode: ExactMode,
    fold_number: int,
    setup_identity: str,
    test: pd.DataFrame,
    replay_metric: pd.Series,
    metric_name: str,
    ledger: TradeLedgerResult,
    history_path: Path,
    cost_version: str,
    blockers: list[str],
    trade_evidence_path: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if ledger.closed_trades.empty:
        return rows
    for _, trade in ledger.closed_trades.iterrows():
        entry_index = int(trade.get("entry_index", -1))
        exit_index = int(trade.get("exit_index", -1))
        if entry_index < 0 or exit_index < entry_index or exit_index >= len(test):
            continue
        feature_timestamp = pd.Timestamp(trade.get("entry_timestamp"))
        label_timestamp = pd.Timestamp(trade.get("exit_timestamp"))
        if feature_timestamp.tzinfo is None:
            feature_timestamp = feature_timestamp.tz_localize("UTC")
        else:
            feature_timestamp = feature_timestamp.tz_convert("UTC")
        if label_timestamp.tzinfo is None:
            label_timestamp = label_timestamp.tz_localize("UTC")
        else:
            label_timestamp = label_timestamp.tz_convert("UTC")
        interval_returns = pd.to_numeric(
            ledger.bar_ledger["net_return"].iloc[entry_index : exit_index + 1],
            errors="coerce",
        ).fillna(0.0)
        path_return = (1.0 + interval_returns).cumprod() - 1.0
        adverse = max(0.0, -float(path_return.min())) if not path_return.empty else 0.0
        favorable = max(0.0, float(path_return.max())) if not path_return.empty else 0.0
        profit = float(trade.get("profit_after_cost", 0.0))
        side = int(np.sign(float(trade.get("side", 0.0))))
        proposed_action = "short_x_long_y" if side > 0 else "long_x_short_y"
        timestamp_valid = feature_timestamp < label_timestamp
        row_blockers = list(blockers)
        if not timestamp_valid:
            row_blockers.append("feature_timestamp_not_before_label_timestamp")
        event_material = "|".join(
            [
                str(candidate.get("run_id", "")),
                setup_identity,
                str(fold_number),
                str(trade.get("trade_id", "")),
                feature_timestamp.isoformat(),
            ]
        )
        entry_row = test.iloc[entry_index]
        rows.append(
            {
                "training_event_id": "training_" + sha256(event_material.encode("utf-8")).hexdigest()[:20],
                "context_id": str(candidate.get("source_snapshot_id", "")) or f"{setup_identity}|fold-{fold_number}",
                "candidate_id": setup_identity,
                "pair": str(candidate.get("pair", "")),
                "timeframe": str(candidate.get("timeframe", "1d")),
                "exact_mode": mode.value,
                "proposed_action": proposed_action,
                "feature_timestamp": feature_timestamp.isoformat(),
                "label_timestamp": label_timestamp.isoformat(),
                "point_in_time_status": "confirmed",
                "math_version": MATH_VERSION,
                "source_system": "hyperliquid_walkforward_trade_ledger",
                "label_source": "backtest_label",
                "uses_wizard_as_label": False,
                "uses_dashboard_hindsight": False,
                "profit_after_cost": profit,
                "good_trade": int(profit > 0.0),
                "max_adverse_excursion": adverse,
                "max_favorable_excursion": favorable,
                "hold_bars": int(trade.get("bars", 0)),
                "exit_reason": str(trade.get("exit_reason", "")),
                "action_propensity": 1.0,
                "evidence_path": ";".join(
                    [
                        str(history_path),
                        trade_evidence_path,
                    ]
                ),
                "run_id": str(candidate.get("run_id", "")),
                "candidate_set_id": str(candidate.get("candidate_set_id", "")),
                "setup_identity": setup_identity,
                "fold": fold_number,
                "trade_id": int(trade.get("trade_id", 0)),
                "entry_metric": _finite_or_nan(replay_metric.iloc[entry_index]),
                "metric_name": metric_name,
                "entry_price_x": _finite_or_nan(entry_row.get("price_x")),
                "entry_price_y": _finite_or_nan(entry_row.get("price_y")),
                "hedge_ratio": _finite_or_nan(entry_row.get("hedge_ratio")),
                "gross_return": _finite_or_nan(trade.get("gross_return")),
                "total_fees": _finite_or_nan(trade.get("total_fees")),
                "total_slippage": _finite_or_nan(trade.get("total_slippage")),
                "total_funding": _finite_or_nan(trade.get("total_funding")),
                "total_execution_risk": _finite_or_nan(trade.get("total_execution_risk")),
                "total_partial_fill": _finite_or_nan(trade.get("total_partial_fill")),
                "cost_model_version": cost_version,
                "validation_version": VALIDATION_VERSION,
                "record_granularity": "trade_entry",
                "training_eligible": not row_blockers,
                "propensity_source": "logged_deterministic_behavior_policy",
                "behavior_policy_id": f"{VALIDATION_VERSION}:{_mode_slug(mode)}",
                "behavior_policy_exploratory": False,
                "training_blocker": ";".join(dict.fromkeys(row_blockers)),
                "research_lane": str(candidate.get("research_lane", "canonical_daily")),
                "wizard_timeframe_authority": str(candidate.get("wizard_timeframe_authority", "matching_daily_discovery")),
            }
        )
    return rows


def _fold_boundaries(rows: int, policy: WalkForwardPolicy) -> list[tuple[int, int, int, int]]:
    available = rows - policy.min_train_rows - policy.embargo_rows * policy.folds
    if policy.folds <= 0 or available < policy.min_test_rows * policy.folds:
        return []
    test_rows = available // policy.folds
    boundaries: list[tuple[int, int, int, int]] = []
    train_end = policy.min_train_rows
    for fold in range(1, policy.folds + 1):
        test_start = train_end + policy.embargo_rows
        test_end = rows if fold == policy.folds else test_start + test_rows
        if test_end - test_start < policy.min_test_rows:
            return []
        boundaries.append((fold, train_end, test_start, test_end))
        train_end = test_end
    return boundaries


def _summarize_folds(
    folds: pd.DataFrame,
    *,
    policy: WalkForwardPolicy,
    evidence_path: str,
) -> pd.DataFrame:
    columns = _summary_columns()
    if folds.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    keys = ["run_id", "candidate_set_id", "setup_identity", "pair", "venue", "timeframe", "exact_mode"]
    for values, group in folds.groupby(keys, dropna=False, sort=True):
        returns = pd.to_numeric(group["total_return"], errors="coerce").dropna()
        trades = pd.to_numeric(group["trades"], errors="coerce").fillna(0)
        hedge = pd.to_numeric(group["hedge_ratio"], errors="coerce").dropna()
        hedge_cv = float(hedge.std(ddof=1) / abs(hedge.mean())) if len(hedge) >= 2 and abs(hedge.mean()) > 1e-12 else 0.0
        pvalue = _one_sided_mean_pvalue(returns)
        fold_expectancy = _weighted_mean(group["expectancy"], trades)
        rows.append(
            dict(
                zip(keys, values)
            )
            | {
                "folds": int(group["fold"].nunique()),
                "total_trades": int(trades.sum()),
                "positive_return_folds": int((returns > 0.0).sum()),
                "positive_expectancy_folds": int((pd.to_numeric(group["expectancy"], errors="coerce") > 0.0).sum()),
                "median_profit_factor": float(pd.to_numeric(group["profit_factor"], errors="coerce").median()),
                "median_sharpe": float(pd.to_numeric(group["sharpe"], errors="coerce").median()),
                "worst_drawdown": float(pd.to_numeric(group["max_drawdown"], errors="coerce").max()),
                "mean_fold_return": float(returns.mean()) if not returns.empty else float("nan"),
                "fold_return_std": float(returns.std(ddof=1)) if len(returns) >= 2 else float("nan"),
                "weighted_expectancy": fold_expectancy,
                "hedge_ratio_cv": hedge_cv,
                "raw_pvalue": pvalue,
                "fold_coverage_status": "PASS" if int(group["fold"].nunique()) >= policy.folds else "BLOCKED",
                "parameter_stability_status": "PASS" if hedge_cv <= policy.max_hedge_ratio_cv else "BLOCKED",
                "validation_version": VALIDATION_VERSION,
                "evidence_path": evidence_path,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _selection_controls(
    summary: pd.DataFrame,
    *,
    policy: WalkForwardPolicy,
    evidence_path: str,
) -> pd.DataFrame:
    columns = _selection_columns()
    if summary.empty:
        return pd.DataFrame(columns=columns)
    working = summary.copy()
    working["bh_qvalue"] = _benjamini_hochberg(pd.to_numeric(working["raw_pvalue"], errors="coerce"))
    rows: list[dict[str, object]] = []
    for _, row in working.iterrows():
        blockers: list[str] = []
        if int(row.get("folds", 0)) < policy.folds:
            blockers.append("insufficient_walkforward_folds")
        if int(row.get("total_trades", 0)) < policy.min_total_trades:
            blockers.append("insufficient_total_oos_trades")
        if int(row.get("positive_return_folds", 0)) < policy.min_positive_folds:
            blockers.append("insufficient_positive_folds")
        if not math.isfinite(float(row.get("bh_qvalue", float("nan")))) or float(row.get("bh_qvalue")) > policy.false_discovery_rate:
            blockers.append("false_discovery_gate_failed")
        if float(row.get("worst_drawdown", float("inf"))) > policy.max_drawdown:
            blockers.append("walkforward_drawdown_gate_failed")
        if str(row.get("parameter_stability_status")) != "PASS":
            blockers.append("hedge_ratio_instability")
        blockers.append("deflated_sharpe_blocked_return_series_required")
        rows.append(
            {
                "run_id": row.get("run_id", ""),
                "candidate_set_id": row.get("candidate_set_id", ""),
                "setup_identity": row.get("setup_identity", ""),
                "pair": row.get("pair", ""),
                "timeframe": row.get("timeframe", ""),
                "exact_mode": row.get("exact_mode", ""),
                "family_tests": len(working),
                "raw_pvalue": row.get("raw_pvalue", float("nan")),
                "bh_qvalue": row.get("bh_qvalue", float("nan")),
                "false_discovery_rate": policy.false_discovery_rate,
                "deflated_sharpe_status": "BLOCKED_RETURN_SERIES_REQUIRED",
                "parameter_stability_status": row.get("parameter_stability_status", "BLOCKED"),
                "selection_status": "PASS" if not blockers else "BLOCKED",
                "selection_blocker": ";".join(blockers),
                "evidence_path": evidence_path,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    return benjamini_hochberg(pvalues)


def _one_sided_mean_pvalue(values: pd.Series) -> float:
    return circular_block_bootstrap_mean(values).pvalue


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    value = pd.to_numeric(values, errors="coerce")
    weight = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    valid = value.notna() & weight.gt(0.0)
    if not valid.any():
        return float("nan")
    return float(np.average(value.loc[valid], weights=weight.loc[valid]))


def _result_number(result: object, key: str) -> float:
    values = getattr(result, "values", {})
    try:
        number = float(values.get(key))
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _finite_or_nan(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _pair_row(frame: pd.DataFrame, pair: str) -> pd.Series:
    if frame.empty or "pair" not in frame.columns:
        return pd.Series(dtype=object)
    rows = frame.loc[frame["pair"].astype(str) == pair]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def _mode_slug(mode: ExactMode) -> str:
    return "_".join(mode.value.lower().replace("zscorer", "zscore_r").split())


def _fold_columns() -> list[str]:
    return [
        "run_id", "candidate_set_id", "setup_identity", "pair", "venue", "timeframe", "exact_mode",
        "fold", "train_start", "train_end", "test_start", "test_end", "train_rows", "test_rows",
        "embargo_rows", "hedge_ratio", "cointegration_pvalue", "hurst", "cost_model_version", "trades",
        "profit_factor", "sharpe", "expectancy", "expectancy_lower_95", "max_drawdown", "total_return",
        "reconciliation_error", "blockers", "validation_version", "evidence_path",
    ]


def _trade_event_columns() -> list[str]:
    return [
        "training_event_id", "context_id", "candidate_id", "pair", "timeframe", "exact_mode",
        "proposed_action", "feature_timestamp", "label_timestamp", "point_in_time_status", "math_version",
        "source_system", "label_source", "uses_wizard_as_label", "uses_dashboard_hindsight",
        "profit_after_cost", "good_trade", "max_adverse_excursion", "max_favorable_excursion", "hold_bars",
        "exit_reason", "action_propensity", "evidence_path", "run_id", "candidate_set_id", "setup_identity",
        "fold", "trade_id", "entry_metric", "metric_name", "entry_price_x", "entry_price_y", "hedge_ratio",
        "gross_return", "total_fees", "total_slippage", "total_funding", "total_execution_risk",
        "total_partial_fill", "cost_model_version", "validation_version", "record_granularity",
        "training_eligible", "propensity_source", "behavior_policy_id", "behavior_policy_exploratory",
        "training_blocker", "research_lane", "wizard_timeframe_authority",
    ]


def _summary_columns() -> list[str]:
    return [
        "run_id", "candidate_set_id", "setup_identity", "pair", "venue", "timeframe", "exact_mode", "folds",
        "total_trades", "positive_return_folds", "positive_expectancy_folds", "median_profit_factor", "median_sharpe",
        "worst_drawdown", "mean_fold_return", "fold_return_std", "weighted_expectancy", "hedge_ratio_cv",
        "raw_pvalue", "fold_coverage_status", "parameter_stability_status", "validation_version", "evidence_path",
    ]


def _selection_columns() -> list[str]:
    return [
        "run_id", "candidate_set_id", "setup_identity", "pair", "timeframe", "exact_mode", "family_tests",
        "raw_pvalue", "bh_qvalue", "false_discovery_rate", "deflated_sharpe_status", "parameter_stability_status",
        "selection_status", "selection_blocker", "evidence_path",
    ]


def _research_family_columns() -> list[str]:
    return [
        "run_id", "candidate_set_id", "parent_candidate_set_id", "setup_identity", "pair", "timeframe",
        "exact_mode", "research_lane", "family_tests", "raw_pvalue", "lane_bh_qvalue",
        "lane_false_discovery_rate", "lane_selection_status", "lane_selection_blocker",
        "research_family_id", "research_attempt_number", "research_family_tests",
        "research_family_bh_qvalue", "base_false_discovery_rate", "alpha_spend_limit",
        "deflated_sharpe_status", "parameter_stability_status", "research_family_complete",
        "selection_status", "selection_blocker", "source_fingerprint", "lane_evidence_path", "evidence_path",
    ]


def _selection_source_fingerprint(*paths: Path) -> str:
    digest = sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes() if path.exists() else b"missing")
    return digest.hexdigest()


def _research_parent_candidate_set_id(daily: pd.DataFrame, auxiliary: pd.DataFrame) -> str:
    for frame in (daily, auxiliary):
        if frame.empty or "candidate_set_id" not in frame.columns:
            continue
        values = frame["candidate_set_id"].dropna().astype(str)
        if not values.empty:
            return values.iloc[0].split(":aux:", 1)[0]
    return ""


def _register_research_family_attempt(
    *,
    registry_path: Path,
    family_id: str,
    parent_candidate_set_id: str,
    source_fingerprint: str,
    base_alpha: float,
) -> tuple[pd.DataFrame, int, float]:
    columns = [
        "research_family_id", "registered_at_utc", "parent_candidate_set_id", "source_fingerprint",
        "research_attempt_number", "base_alpha_budget", "alpha_spend_limit", "cumulative_alpha_spent",
    ]
    registry = _read_csv(registry_path).reindex(columns=columns)
    existing = registry.loc[
        registry.get("research_family_id", pd.Series(dtype=str)).astype(str).eq(family_id)
    ] if not registry.empty else pd.DataFrame()
    if not existing.empty:
        row = existing.iloc[0]
        return registry, int(row["research_attempt_number"]), float(row["alpha_spend_limit"])
    attempts = pd.to_numeric(registry.get("research_attempt_number", pd.Series(dtype=float)), errors="coerce").dropna()
    attempt_number = int(attempts.max()) + 1 if not attempts.empty else 1
    alpha_spend = float(base_alpha / (attempt_number * (attempt_number + 1)))
    previous_spend = pd.to_numeric(
        registry.get("alpha_spend_limit", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0.0).sum()
    row = pd.DataFrame(
        [
            {
                "research_family_id": family_id,
                "registered_at_utc": datetime.now(timezone.utc).isoformat(),
                "parent_candidate_set_id": parent_candidate_set_id,
                "source_fingerprint": source_fingerprint,
                "research_attempt_number": attempt_number,
                "base_alpha_budget": base_alpha,
                "alpha_spend_limit": alpha_spend,
                "cumulative_alpha_spent": min(base_alpha, float(previous_spend + alpha_spend)),
            }
        ]
    )
    registry = pd.concat([registry, row], ignore_index=True)
    return registry.reindex(columns=columns), attempt_number, alpha_spend


def _split_semicolon(value: object) -> list[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip() and item.strip().lower() != "nan"]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    promote_staged_file(temporary, path)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _markdown(folds: pd.DataFrame, summary: pd.DataFrame, selection: pd.DataFrame, policy: WalkForwardPolicy) -> str:
    return "\n".join(
        [
            "# Hyperliquid Walk-Forward Validation",
            "",
            f"- Validation version: `{VALIDATION_VERSION}`",
            f"- Requested folds: `{policy.folds}`",
            f"- Embargo rows: `{policy.embargo_rows}`",
            f"- Fold evidence rows: `{len(folds)}`",
            f"- Mode summaries: `{len(summary)}`",
            f"- Selection passes: `{int(selection.get('selection_status', pd.Series(dtype=str)).eq('PASS').sum()) if not selection.empty else 0}`",
            "- Deflated Sharpe is evaluated only when at least five genuinely out-of-sample folds exist.",
            "",
            "## Selection Controls",
            "",
            selection.to_markdown(index=False) if not selection.empty else "No selection evidence.",
            "",
        ]
    )
