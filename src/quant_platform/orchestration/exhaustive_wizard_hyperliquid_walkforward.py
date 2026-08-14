"""Purged expanding walk-forward research for exhaustive Wizard hypotheses."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.backtest import (
    CostModel,
    FundingPolicy,
    backtest_two_leg_spread_with_ledger,
    max_drawdown,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_canonical_replay import (
    _load_history,
    _mode_settings,
    _orient_history,
    _unique_mode_rows,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_observed_cost_replay import (
    _longest_observed_funding_segment,
)
from quant_platform.performance_math import calculate_annualized_sharpe
from quant_platform.wizard_mode_replay import build_local_mode_signal


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "exhaustive_wizard_hyperliquid_walkforward.v1"
FOLD_COUNT = 5
INITIAL_TRAIN_FRACTION = 0.50
EMBARGO_BARS = 20
MIN_TRAIN_ROWS = 180
MIN_TEST_ROWS = 30
MIN_AGGREGATE_TRADES = 10
MIN_PROFIT_FACTOR = 1.10
MAX_DRAWDOWN = 0.50
MIN_POSITIVE_FOLDS = 3
MAX_POSITIVE_FOLD_CONCENTRATION = 0.80
FALSE_DISCOVERY_RATE = 0.10
MAX_HEDGE_RATIO_CV = 0.35
RESEARCH_ONLY_REASON = (
    "walk_forward_is_research_only;local_formula_approximation;"
    "current_l2_depth_is_point_in_time_not_historical_execution_evidence;"
    "mode_fidelity_parity_not_proven"
)
EMPTY_TRADE_COLUMNS = [
    "schema_version",
    "walkforward_id",
    "observed_cost_replay_id",
    "experiment_id",
    "pair_group_id",
    "pair",
    "wizard_exchange",
    "wizard_timeframe",
    "hyperliquid_interval",
    "exact_mode",
    "orientation",
    "ledger_type",
    "fold_number",
    "trade_id",
    "side",
    "entry_timestamp",
    "exit_timestamp",
    "exit_reason",
    "bars",
    "gross_return",
    "profit_after_cost",
    "total_fees",
    "total_slippage",
    "total_funding",
    "total_execution_risk",
    "total_partial_fill",
    "backtest_label",
    "paper_label",
    "live_label",
    "live_trading_authorized",
]


def run_exhaustive_wizard_hyperliquid_walkforward(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Run purged expanding folds for every completed observed-cost replay.

    Full-sample winners are tagged as a primary cohort, but selection for fold
    execution depends only on replay completion and sufficient history. This
    keeps weak cells in the evaluation set instead of using future performance
    to decide which hypotheses receive out-of-sample tests.
    """

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    observed_manifest_path = (
        active / "exhaustive_wizard_hyperliquid_observed_cost_replay_manifest.json"
    )
    if not observed_manifest_path.exists():
        raise FileNotFoundError("Observed-cost replay manifest is required")
    observed_manifest = json.loads(observed_manifest_path.read_text(encoding="utf-8"))

    run_id = _text(observed_manifest.get("run_id"))
    preflight_id = _text(observed_manifest.get("replay_preflight_id"))
    history_run_id = _text(observed_manifest.get("history_run_id"))
    funding_evidence_id = _text(observed_manifest.get("funding_evidence_id"))
    cost_evidence_id = _text(observed_manifest.get("cost_evidence_id"))
    observed_replay_id = _text(observed_manifest.get("observed_cost_replay_id"))

    artifacts = observed_manifest.get("artifacts", {})
    snapshots = observed_manifest.get("input_snapshots", {})
    input_paths = {
        "observed_results": root / _text(artifacts.get("snapshot_results")),
        "mode_ledger": root / _text(snapshots.get("mode_ledger")),
        "pair_cost_evidence": root / _text(snapshots.get("pair_cost_evidence")),
        "experiment_preflight": root / _text(snapshots.get("experiment_preflight")),
        "observed_manifest": observed_manifest_path,
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Walk-forward inputs missing: {missing}")

    observed = _read_csv(input_paths["observed_results"])
    modes = _read_csv(input_paths["mode_ledger"])
    pair_costs = _read_csv(input_paths["pair_cost_evidence"])
    experiments = _read_csv(input_paths["experiment_preflight"])
    _require_unique(observed, "experiment_id")
    _require_unique(experiments, "experiment_id")
    _require_unique(pair_costs, "pair_group_id")
    if len(observed) != len(experiments) or set(observed["experiment_id"].astype(str)) != set(
        experiments["experiment_id"].astype(str)
    ):
        raise ValueError("Observed replay does not account for the preflight experiment universe")

    policy = {
        "fold_count": FOLD_COUNT,
        "initial_train_fraction": INITIAL_TRAIN_FRACTION,
        "embargo_bars": EMBARGO_BARS,
        "minimum_train_rows": MIN_TRAIN_ROWS,
        "minimum_test_rows": MIN_TEST_ROWS,
        "minimum_aggregate_trades": MIN_AGGREGATE_TRADES,
        "minimum_profit_factor": MIN_PROFIT_FACTOR,
        "maximum_drawdown": MAX_DRAWDOWN,
        "minimum_positive_folds": MIN_POSITIVE_FOLDS,
        "maximum_positive_fold_concentration": MAX_POSITIVE_FOLD_CONCENTRATION,
        "false_discovery_rate": FALSE_DISCOVERY_RATE,
        "maximum_hedge_ratio_cv": MAX_HEDGE_RATIO_CV,
        "candidate_policy": "all_completed_observed_cost_replays_no_performance_prefilter",
        "primary_cohort_policy": "full_sample_research_rank_eligible_label_only",
        "fold_position_policy": "start_flat_wait_for_source_flat_force_flat_at_end",
        "fit_policy": "training_only_all_mode_hedge_ratio_and_ou_parameters",
    }
    material = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "observed_cost_replay_id": observed_replay_id,
        "as_of": as_of.isoformat(),
        "policy": policy,
        "input_hashes": {name: _file_hash(path) for name, path in input_paths.items()},
    }
    run_hash = sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    walkforward_id = f"hlwalk_{as_of.strftime('%Y%m%dT%H%M%S%fZ')}_{run_hash[:8]}"
    observed_snapshot_manifest = root / _text(artifacts.get("snapshot_manifest"))
    snapshot_dir = observed_snapshot_manifest.parent / "walkforwards" / walkforward_id
    snapshot_input_dir = snapshot_dir / "inputs"
    snapshot_input_dir.mkdir(parents=True, exist_ok=True)
    snapshot_inputs: dict[str, Path] = {}
    for name, source in input_paths.items():
        suffix = source.suffix or ".dat"
        target = snapshot_input_dir / f"{name}{suffix}"
        shutil.copy2(source, target)
        snapshot_inputs[name] = target

    paths = {
        "status": active / "exhaustive_wizard_hyperliquid_walkforward_status.csv",
        "candidates": active / "exhaustive_wizard_hyperliquid_walkforward_candidates.csv",
        "ranked": active / "exhaustive_wizard_hyperliquid_walkforward_ranked.csv",
        "folds": active / "exhaustive_wizard_hyperliquid_walkforward_folds.csv",
        "trades": active / "exhaustive_wizard_hyperliquid_walkforward_trades.csv",
        "bars": active / "exhaustive_wizard_hyperliquid_walkforward_bars.csv",
        "manifest": active / "exhaustive_wizard_hyperliquid_walkforward_manifest.json",
        "summary_md": active / "exhaustive_wizard_hyperliquid_walkforward_summary.md",
        "snapshot_status": snapshot_dir / "walkforward_status.csv",
        "snapshot_candidates": snapshot_dir / "walkforward_candidates.csv",
        "snapshot_ranked": snapshot_dir / "walkforward_ranked.csv",
        "snapshot_folds": snapshot_dir / "walkforward_folds.csv",
        "snapshot_trades": snapshot_dir / "walkforward_trades.csv",
        "snapshot_bars": snapshot_dir / "walkforward_bars.csv",
        "snapshot_manifest": snapshot_dir / "manifest.json",
        "snapshot_summary_md": snapshot_dir / "summary.md",
    }

    active_pair_group_ids = {_text(value) for value in observed["pair_group_id"]}
    if "" in active_pair_group_ids:
        raise ValueError("Walk-forward replay identity missing pair_group_id")
    mode_lookup = _unique_mode_rows(
        modes,
        allowed_pair_group_ids=active_pair_group_ids,
    )
    pair_lookup = _row_lookup(pair_costs, "pair_group_id")
    history_cache: dict[str, pd.DataFrame] = {}
    status_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    bar_rows: list[dict[str, object]] = []

    for observed_row in observed.itertuples():
        base = _status_base(
            observed_row,
            walkforward_id=walkforward_id,
            evidence_paths=snapshot_inputs.values(),
            root=root,
        )
        if _text(observed_row.replay_status) != "OBSERVED_COST_RESEARCH_REPLAY_COMPLETE":
            status_rows.append(
                {
                    **base,
                    "walkforward_status": "NOT_RUN_PRIOR_REPLAY_BLOCKED",
                    "walkforward_blocker": (
                        _text(getattr(observed_row, "replay_blocker", ""))
                        or _text(observed_row.replay_status)
                    ),
                }
            )
            continue

        pair_group_id = _text(observed_row.pair_group_id)
        exact_mode = _text(observed_row.exact_mode)
        orientation = _text(observed_row.orientation)
        pair_cost = pair_lookup.get(pair_group_id)
        mode_row = mode_lookup.get((pair_group_id, exact_mode, orientation))
        if pair_cost is None or mode_row is None:
            blocker = (
                "pair_cost_evidence_missing" if pair_cost is None else "captured_mode_row_missing"
            )
            status_rows.append(
                {
                    **base,
                    "walkforward_status": "BLOCKED_WALK_FORWARD_INPUTS",
                    "walkforward_blocker": blocker,
                }
            )
            continue

        try:
            history_path = root / _text(pair_cost.enriched_history_path)
            cache_key = f"{history_path}:{_text(observed_row.hyperliquid_interval)}"
            if cache_key not in history_cache:
                history_cache[cache_key] = _longest_observed_funding_segment(
                    _load_history(history_path),
                    interval=_text(observed_row.hyperliquid_interval),
                )
            raw_history = history_cache[cache_key].copy()
            raw_history["slippage_x_model_bps"] = _required_number(
                pair_cost.slippage_x_p95_bps,
                "slippage_x_p95_bps",
            )
            raw_history["slippage_y_model_bps"] = _required_number(
                pair_cost.slippage_y_p95_bps,
                "slippage_y_p95_bps",
            )
            history = _orient_history(raw_history, orientation=orientation)
            folds = _build_folds(len(history))
        except Exception as exc:
            status_rows.append(
                {
                    **base,
                    "walkforward_status": "BLOCKED_WALK_FORWARD_INPUTS",
                    "walkforward_blocker": f"{type(exc).__name__}:{exc}",
                }
            )
            continue

        costs = CostModel(
            taker_fee_bps=_required_number(pair_cost.taker_fee_bps, "taker_fee_bps"),
            slippage_bps=_required_number(
                pair_cost.pair_one_way_slippage_bps,
                "pair_one_way_slippage_bps",
            ),
            execution_risk_bps=_required_number(
                pair_cost.execution_risk_bps,
                "execution_risk_bps",
            ),
            funding_bps_per_day=0.0,
            funding_policy=FundingPolicy.SIGNED_REALIZED.value,
        )
        paired_ou = mode_lookup.get((pair_group_id, "OU (Spread)", orientation))
        captured_settings, setting_source = _mode_settings(mode_row, paired_ou=paired_ou)
        candidate_fold_rows: list[dict[str, object]] = []
        candidate_trades: list[pd.DataFrame] = []
        candidate_bars: list[pd.DataFrame] = []
        candidate_blocker = ""

        for fold in folds:
            try:
                train = history.iloc[: fold["train_end"]].copy()
                settings, fit_diagnostics = _fit_training_parameters(
                    train,
                    captured_settings,
                    exact_mode=exact_mode,
                )
                signal_history = history.iloc[: fold["test_end"]].copy()
                hedge_ratio = _number(settings.get("hedge_ratio"))
                if hedge_ratio is not None:
                    signal_history["hedge_ratio"] = hedge_ratio
                mode_result = build_local_mode_signal(
                    signal_history,
                    settings,
                    exact_mode=exact_mode,
                )
                if mode_result.mode_replay_status != "READY_FOR_RESEARCH_REPLAY":
                    raise ValueError(
                        "mode_signal_blocked:" + ";".join(mode_result.missing_inputs)
                    )
                eval_start = fold["test_start"] - 1
                evaluation = signal_history.iloc[eval_start : fold["test_end"]].copy()
                source_signal = mode_result.signal.reindex(signal_history.index)
                eval_signal = _flat_fold_signal(
                    source_signal,
                    evaluation.index,
                    prior_position=source_signal.iloc[eval_start],
                )
                if hedge_ratio is not None:
                    evaluation["hedge_ratio"] = hedge_ratio
                result, ledger = backtest_two_leg_spread_with_ledger(
                    evaluation,
                    eval_signal,
                    costs,
                    interval=_text(observed_row.hyperliquid_interval),
                )
                closed, bars = _attach_causal_entry_features(
                    closed=ledger.closed_trades,
                    bars=ledger.bar_ledger,
                    signal_history=signal_history,
                    mode_metric=mode_result.metric,
                    metric_name=mode_result.metric_name,
                    settings=settings,
                    exact_mode=exact_mode,
                )
            except Exception as exc:
                candidate_blocker = f"fold_{fold['fold_number']}:{type(exc).__name__}:{exc}"
                break

            fold_metrics = {
                **_identity_fields(observed_row),
                "schema_version": SCHEMA_VERSION,
                "walkforward_id": walkforward_id,
                "observed_cost_replay_id": observed_replay_id,
                "fold_number": fold["fold_number"],
                "fold_status": "COMPLETE",
                "fold_blocker": "",
                "train_start_at": _timestamp(history.iloc[0]["timestamp"]),
                "train_end_at": _timestamp(history.iloc[fold["train_end"] - 1]["timestamp"]),
                "embargo_start_at": _timestamp(history.iloc[fold["train_end"]]["timestamp"]),
                "test_start_at": _timestamp(history.iloc[fold["test_start"]]["timestamp"]),
                "test_end_at": _timestamp(history.iloc[fold["test_end"] - 1]["timestamp"]),
                "train_rows": fold["train_end"],
                "embargo_rows": fold["test_start"] - fold["train_end"],
                "test_rows": fold["test_end"] - fold["test_start"],
                "mode_setting_source": setting_source,
                "fitted_hedge_ratio": fit_diagnostics.get("fitted_hedge_ratio"),
                "fitted_ou_mu": fit_diagnostics.get("fitted_ou_mu"),
                "fitted_ou_sigma": fit_diagnostics.get("fitted_ou_sigma"),
                "fit_uses_test_data": False,
                **asdict(result),
                "fold_positive_after_cost": bool(result.total_return > 0.0),
                "acceptance_status": "BLOCKED",
                "acceptance_reason": RESEARCH_ONLY_REASON,
                "live_trading_authorized": False,
            }
            candidate_fold_rows.append(fold_metrics)

            if not closed.empty:
                closed["fold_number"] = fold["fold_number"]
                candidate_trades.append(closed)
            bars["fold_number"] = fold["fold_number"]
            candidate_bars.append(bars)

        if candidate_blocker or len(candidate_fold_rows) != FOLD_COUNT:
            status_rows.append(
                {
                    **base,
                    "walkforward_status": "BLOCKED_WALK_FORWARD_REPLAY",
                    "walkforward_blocker": candidate_blocker or "incomplete_fold_accounting",
                }
            )
            continue

        aggregate = _aggregate_candidate(
            candidate_fold_rows,
            candidate_trades,
            candidate_bars,
            interval=_text(observed_row.hyperliquid_interval),
        )
        gate_blockers = _walkforward_gate_blockers(aggregate)
        passed = not gate_blockers
        candidate_row = {
            **base,
            **aggregate,
            "primary_candidate": _truthy(
                getattr(observed_row, "research_rank_eligible", False)
            ),
            "walkforward_status": (
                "PASS_RESEARCH_WALK_FORWARD" if passed else "FAIL_RESEARCH_WALK_FORWARD"
            ),
            "walkforward_blocker": ";".join(gate_blockers),
            "acceptance_status": "BLOCKED",
            "acceptance_reason": (
                f"{RESEARCH_ONLY_REASON};{_text(pair_cost.cost_blocker)}"
            ),
            "acceptance_eligible": False,
            "live_trading_authorized": False,
        }
        candidate_rows.append(candidate_row)
        status_rows.append(candidate_row.copy())
        fold_rows.extend(candidate_fold_rows)

        for lifecycle, frames in (("closed", candidate_trades), ("bar", candidate_bars)):
            for frame in frames:
                for record in frame.reset_index(drop=True).to_dict("records"):
                    payload = {
                        **_identity_fields(observed_row),
                        "schema_version": SCHEMA_VERSION,
                        "walkforward_id": walkforward_id,
                        "observed_cost_replay_id": observed_replay_id,
                        "ledger_type": lifecycle,
                        **record,
                        "backtest_label": True,
                        "paper_label": False,
                        "live_label": False,
                        "live_trading_authorized": False,
                    }
                    if lifecycle == "closed":
                        trade_rows.append(payload)
                    else:
                        bar_rows.append(payload)

    status = pd.DataFrame(status_rows)
    if len(status) != len(observed) or status["experiment_id"].nunique() != len(observed):
        raise ValueError("Walk-forward failed complete experiment accounting")
    walkforward_status_defaults = {
        "folds_complete": 0,
        "positive_folds": 0,
        "aggregate_trades": 0,
        "aggregate_profit_factor": "",
        "aggregate_expectancy": "",
        "aggregate_sharpe": "",
        "aggregate_max_drawdown": "",
        "aggregate_total_return": "",
        "fold_return_raw_pvalue": "",
        "bh_qvalue": "",
        "parameter_stability_status": "NOT_EVALUATED",
        "deflated_sharpe_status": "NOT_EVALUATED",
    }
    for column, default in walkforward_status_defaults.items():
        status[column] = status.get(
            column,
            pd.Series(index=status.index, dtype=object),
        ).fillna(default)
    candidates = (
        pd.DataFrame(candidate_rows)
        if candidate_rows
        else status.iloc[0:0].copy()
    )
    candidates = _add_statistical_selection_controls(candidates)
    statistical_columns = [
        "family_tests",
        "fold_return_raw_pvalue",
        "bh_qvalue",
        "false_discovery_rate",
        "hedge_ratio_cv",
        "parameter_stability_status",
        "deflated_sharpe_status",
        "statistical_selection_status",
        "statistical_selection_blocker",
    ]
    if not candidates.empty:
        candidate_statistics = candidates.set_index("experiment_id")[statistical_columns]
        candidate_mask = status["experiment_id"].isin(candidate_statistics.index)
        for column in statistical_columns:
            status.loc[candidate_mask, column] = status.loc[
                candidate_mask, "experiment_id"
            ].map(candidate_statistics[column])
    status["statistical_selection_status"] = status.get(
        "statistical_selection_status", pd.Series(index=status.index, dtype=object)
    ).fillna("NOT_EVALUATED")
    status["statistical_selection_blocker"] = status.get(
        "statistical_selection_blocker", pd.Series(index=status.index, dtype=object)
    ).fillna("walk_forward_not_completed")
    folds_frame = (
        pd.DataFrame(fold_rows)
        if fold_rows
        else pd.DataFrame(columns=["experiment_id", "fold_number", "fold_status"])
    )
    trades = (
        pd.DataFrame(trade_rows)
        if trade_rows
        else pd.DataFrame(columns=EMPTY_TRADE_COLUMNS)
    )
    bars = (
        pd.DataFrame(bar_rows)
        if bar_rows
        else pd.DataFrame(columns=["experiment_id", "timestamp", "fold_number"])
    )
    ranked = _rank_candidates(candidates)

    for frame, active_path, snapshot_path in (
        (status, paths["status"], paths["snapshot_status"]),
        (candidates, paths["candidates"], paths["snapshot_candidates"]),
        (ranked, paths["ranked"], paths["snapshot_ranked"]),
        (folds_frame, paths["folds"], paths["snapshot_folds"]),
        (trades, paths["trades"], paths["snapshot_trades"]),
        (bars, paths["bars"], paths["snapshot_bars"]),
    ):
        frame.to_csv(active_path, index=False)
        frame.to_csv(snapshot_path, index=False)

    status_counts = _status_counts(status, "walkforward_status")
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "replay_preflight_id": preflight_id,
        "history_run_id": history_run_id,
        "funding_evidence_id": funding_evidence_id,
        "cost_evidence_id": cost_evidence_id,
        "observed_cost_replay_id": observed_replay_id,
        "walkforward_id": walkforward_id,
        "created_at": as_of.isoformat(),
        "experiments": int(len(status)),
        "unique_experiment_ids": int(status["experiment_id"].nunique()),
        "experiment_status_counts": status_counts,
        "experiment_status_accounted": bool(sum(status_counts.values()) == len(status)),
        "completed_observed_replays": int(
            observed["replay_status"].eq("OBSERVED_COST_RESEARCH_REPLAY_COMPLETE").sum()
        ),
        "walkforward_candidates_completed": int(len(candidates)),
        "primary_candidates_completed": int(
            candidates.get("primary_candidate", pd.Series(dtype=bool)).map(_truthy).sum()
        ),
        "walkforward_passes": int(
            candidates.get("walkforward_status", pd.Series(dtype=str))
            .eq("PASS_RESEARCH_WALK_FORWARD")
            .sum()
        ),
        "statistical_selection_passes": int(
            candidates.get("statistical_selection_status", pd.Series(dtype=str))
            .eq("PASS")
            .sum()
        ),
        "folds_expected": int(len(candidates) * FOLD_COUNT),
        "folds_complete": int(len(folds_frame)),
        "closed_trade_rows": int(len(trades)),
        "bar_ledger_rows": int(len(bars)),
        "policy": policy,
        "selection_hindsight_used_for_fold_execution": False,
        "canonical_replay_leverage": 1.0,
        "acceptance_eligible_replays": 0,
        "live_trading_authorized": False,
        "input_hashes": material["input_hashes"],
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
        "input_snapshots": {
            name: _relative(path, root) for name, path in snapshot_inputs.items()
        },
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    paths["manifest"].write_text(manifest_text, encoding="utf-8")
    paths["snapshot_manifest"].write_text(manifest_text, encoding="utf-8")
    paths["summary_md"].write_text(summary_text, encoding="utf-8")
    paths["snapshot_summary_md"].write_text(summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _build_folds(rows: int) -> list[dict[str, int]]:
    initial_train = max(MIN_TRAIN_ROWS, int(math.floor(rows * INITIAL_TRAIN_FRACTION)))
    remaining = rows - initial_train
    test_block = remaining // FOLD_COUNT
    if test_block < EMBARGO_BARS + MIN_TEST_ROWS:
        raise ValueError(
            f"insufficient_rows_for_purged_folds:{rows};required_test_block>="
            f"{EMBARGO_BARS + MIN_TEST_ROWS}"
        )
    folds: list[dict[str, int]] = []
    for offset in range(FOLD_COUNT):
        train_end = initial_train + offset * test_block
        test_start = train_end + EMBARGO_BARS
        test_end = rows if offset == FOLD_COUNT - 1 else initial_train + (offset + 1) * test_block
        if test_end - test_start < MIN_TEST_ROWS:
            raise ValueError(f"fold_{offset + 1}_test_rows_below_{MIN_TEST_ROWS}")
        folds.append(
            {
                "fold_number": offset + 1,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
    return folds


def _fit_training_parameters(
    train: pd.DataFrame,
    captured_settings: dict[str, object],
    *,
    exact_mode: str,
) -> tuple[dict[str, object], dict[str, float | None]]:
    settings = dict(captured_settings)
    diagnostics: dict[str, float | None] = {
        "fitted_hedge_ratio": None,
        "fitted_ou_mu": None,
        "fitted_ou_sigma": None,
    }
    prices = train[["price_x", "price_y"]].apply(pd.to_numeric, errors="coerce")
    prices = prices.loc[(prices["price_x"] > 0.0) & (prices["price_y"] > 0.0)].dropna()
    if len(prices) < MIN_TRAIN_ROWS:
        raise ValueError(f"finite_training_prices_{len(prices)}_below_{MIN_TRAIN_ROWS}")
    log_x = np.log(prices["price_x"])
    log_y = np.log(prices["price_y"])
    variance_x = float(log_x.var(ddof=0))
    if not math.isfinite(variance_x) or variance_x <= 1e-12:
        raise ValueError("training_log_x_variance_too_small")
    beta = float(log_x.cov(log_y, ddof=0) / variance_x)
    if not math.isfinite(beta) or abs(beta) <= 1e-8:
        raise ValueError("training_hedge_ratio_invalid")
    settings["hedge_ratio"] = beta
    diagnostics["fitted_hedge_ratio"] = beta
    if exact_mode.startswith("OU"):
        spread = log_y - beta * log_x
        mu = float(spread.mean())
        sigma = float(spread.std(ddof=0))
        if not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 1e-12:
            raise ValueError("training_ou_parameters_invalid")
        settings["ou_mu"] = mu
        settings["ou_sigma"] = sigma
        diagnostics["fitted_ou_mu"] = mu
        diagnostics["fitted_ou_sigma"] = sigma
    return settings, diagnostics


def _attach_causal_entry_features(
    *,
    closed: pd.DataFrame,
    bars: pd.DataFrame,
    signal_history: pd.DataFrame,
    mode_metric: pd.Series,
    metric_name: str,
    settings: dict[str, object],
    exact_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach only candle-close information known when each position changed."""

    features = _causal_feature_frame(
        signal_history=signal_history,
        mode_metric=mode_metric,
        metric_name=metric_name,
        settings=settings,
        exact_mode=exact_mode,
    )
    keyed = features.drop_duplicates("_feature_key", keep="last")

    bar_ledger = bars.copy()
    bar_ledger["_feature_key"] = _utc_key(bar_ledger.get("timestamp"))
    bar_ledger = bar_ledger.merge(
        keyed,
        on="_feature_key",
        how="left",
        validate="many_to_one",
    ).drop(columns="_feature_key")

    trade_ledger = closed.copy()
    if not trade_ledger.empty:
        trade_ledger["_feature_key"] = _utc_key(
            trade_ledger.get("entry_timestamp")
        )
        trade_ledger = trade_ledger.merge(
            keyed,
            on="_feature_key",
            how="left",
            validate="many_to_one",
        ).drop(columns="_feature_key")
        missing = pd.to_numeric(
            trade_ledger.get("mode_metric", pd.Series(dtype=float)),
            errors="coerce",
        ).isna()
        if missing.any():
            raise ValueError("closed_trade_entry_feature_join_incomplete")
        trade_ledger["label_timestamp"] = trade_ledger["exit_timestamp"]
    return trade_ledger, bar_ledger


def _causal_feature_frame(
    *,
    signal_history: pd.DataFrame,
    mode_metric: pd.Series,
    metric_name: str,
    settings: dict[str, object],
    exact_mode: str,
) -> pd.DataFrame:
    timestamps = pd.to_datetime(
        signal_history.get("timestamp"), utc=True, errors="coerce", format="mixed"
    )
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("causal_feature_timestamps_invalid_or_duplicate")
    price_x = pd.to_numeric(signal_history.get("price_x"), errors="coerce")
    price_y = pd.to_numeric(signal_history.get("price_y"), errors="coerce")
    hedge_ratio = pd.to_numeric(
        signal_history.get(
            "hedge_ratio",
            pd.Series(settings.get("hedge_ratio"), index=signal_history.index),
        ),
        errors="coerce",
    )
    spread = np.log(price_y.where(price_y > 0.0)) - hedge_ratio * np.log(
        price_x.where(price_x > 0.0)
    )
    returns_x = price_x.pct_change(fill_method=None)
    returns_y = price_y.pct_change(fill_method=None)
    pair_return = returns_y - hedge_ratio * returns_x
    realized_volatility = pair_return.rolling(20, min_periods=10).std(ddof=0)
    volatility_percentile = realized_volatility.rolling(
        252, min_periods=20
    ).rank(pct=True)
    correlation = returns_x.rolling(60, min_periods=20).corr(returns_y)
    hedge_mean = hedge_ratio.rolling(60, min_periods=20).mean().abs()
    hedge_cv = hedge_ratio.rolling(60, min_periods=20).std(ddof=0).div(
        hedge_mean.where(hedge_mean > 1e-12)
    )
    hedge_stability = 1.0 / (1.0 + hedge_cv.clip(lower=0.0))
    metric = pd.to_numeric(mode_metric.reindex(signal_history.index), errors="coerce")
    is_copula = exact_mode == "Copula"
    is_rolling = exact_mode.endswith("ZScoreR)")
    funding = _first_numeric_series(
        signal_history,
        ("funding_bps_per_day", "pair_funding_bps_per_day"),
    )
    slippage_x = _first_numeric_series(
        signal_history,
        ("slippage_x_model_bps", "slippage_x_p95_bps"),
    ).abs()
    slippage_y = _first_numeric_series(
        signal_history,
        ("slippage_y_model_bps", "slippage_y_p95_bps"),
    ).abs()
    pair_slippage = slippage_x.add(slippage_y, fill_value=0.0)
    liquidity_score = 1.0 / (1.0 + pair_slippage.clip(lower=0.0))
    frame = pd.DataFrame(
        {
            "_feature_key": timestamps,
            "feature_timestamp": timestamps.map(
                lambda value: value.isoformat() if pd.notna(value) else ""
            ),
            "mode_metric": metric,
            "mode_metric_lag_1": metric.shift(1),
            "mode_metric_slope_5": metric.diff(5).div(5.0),
            "mode_metric_name": metric_name,
            "zscore": np.nan if is_copula else metric,
            "rolling_zscore": metric if is_rolling else np.nan,
            "spread": spread,
            "spread_slope": spread.diff(5).div(5.0),
            "realized_volatility_20": realized_volatility,
            "realized_volatility_percentile": volatility_percentile,
            "correlation": correlation,
            "hedge_ratio": hedge_ratio,
            "hedge_ratio_stability": hedge_stability,
            "beta": hedge_ratio,
            "beta_stability": hedge_stability,
            "funding_bps_per_day": funding,
            "liquidity_score": liquidity_score,
            "conditional_probability_distortion": (
                metric.sub(0.5).abs() if is_copula else np.nan
            ),
            "strategy": exact_mode,
            "strategy_name": exact_mode,
            "family": exact_mode,
            "entry_style": "wizard_exact_mode_captured_thresholds",
            "exit_style": "wizard_exact_mode_captured_thresholds",
            "mode_fidelity_status": "local_formula_approximation",
            "feature_source": "causal_walkforward_entry_bar",
            "feature_known_at_or_before_entry": True,
            "feature_uses_future_data": False,
            "uses_dashboard_hindsight": False,
        },
        index=signal_history.index,
    )
    if is_copula and metric_name in {"u1_given_u2", "u2_given_u1"}:
        frame[metric_name] = metric
    return frame.reset_index(drop=True)


def _first_numeric_series(
    frame: pd.DataFrame, names: tuple[str, ...]
) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(0.0, index=frame.index, dtype="float64")


def _utc_key(values: object) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce", format="mixed")


def _flat_fold_signal(
    source: pd.Series,
    evaluation_index: pd.Index,
    *,
    prior_position: object,
) -> pd.Series:
    raw = pd.to_numeric(source.reindex(evaluation_index), errors="coerce").fillna(0.0)
    output = pd.Series(0.0, index=evaluation_index, dtype="float64")
    waiting_for_flat = abs(_number(prior_position) or 0.0) > 1e-12
    for position in range(1, len(raw)):
        value = float(raw.iloc[position])
        if waiting_for_flat:
            if abs(value) <= 1e-12:
                waiting_for_flat = False
            continue
        output.iloc[position] = value
    if len(output):
        output.iloc[-1] = 0.0
    return output


def _aggregate_candidate(
    fold_rows: list[dict[str, object]],
    trade_frames: list[pd.DataFrame],
    bar_frames: list[pd.DataFrame],
    *,
    interval: str,
) -> dict[str, object]:
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    bars = pd.concat(bar_frames, ignore_index=True) if bar_frames else pd.DataFrame()
    closed_returns = (
        pd.to_numeric(trades["profit_after_cost"], errors="coerce").dropna()
        if not trades.empty
        else pd.Series(dtype="float64")
    )
    wins = closed_returns.loc[closed_returns > 0.0]
    losses = closed_returns.loc[closed_returns < 0.0]
    if not losses.empty:
        profit_factor = float(wins.sum() / abs(losses.sum()))
    elif not wins.empty:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0
    net_returns = (
        pd.to_numeric(bars["net_return"], errors="coerce").fillna(0.0)
        if not bars.empty
        else pd.Series(dtype="float64")
    )
    equity = (1.0 + net_returns).cumprod()
    sharpe = calculate_annualized_sharpe(net_returns, interval=interval)
    fold_returns = [float(_number(row.get("total_return")) or 0.0) for row in fold_rows]
    fitted_hedges = pd.to_numeric(
        pd.Series([row.get("fitted_hedge_ratio") for row in fold_rows]),
        errors="coerce",
    ).dropna()
    hedge_ratio_cv = (
        float(fitted_hedges.std(ddof=1) / abs(fitted_hedges.mean()))
        if len(fitted_hedges) >= 2 and abs(float(fitted_hedges.mean())) > 1e-12
        else float("inf")
    )
    positive_returns = [value for value in fold_returns if value > 0.0]
    concentration = (
        max(positive_returns) / sum(positive_returns) if positive_returns else 1.0
    )
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    return {
        "folds_complete": len(fold_rows),
        "positive_folds": sum(value > 0.0 for value in fold_returns),
        "positive_fold_profit_concentration": concentration,
        "fold_return_raw_pvalue": _one_sided_mean_pvalue(pd.Series(fold_returns)),
        "hedge_ratio_cv": hedge_ratio_cv,
        "aggregate_trades": int(len(closed_returns)),
        "aggregate_profit_factor": profit_factor,
        "aggregate_expectancy": (
            float(closed_returns.mean()) if not closed_returns.empty else 0.0
        ),
        "aggregate_sharpe": sharpe.value,
        "aggregate_sharpe_status": (
            sharpe.status if sharpe.status == "valid" else f"blocked:{sharpe.reason}"
        ),
        "aggregate_max_drawdown": max_drawdown(equity),
        "aggregate_total_return": total_return,
        "aggregate_total_fees": _sum_column(bars, "fees"),
        "aggregate_total_slippage": _sum_column(bars, "slippage"),
        "aggregate_total_funding": _sum_column(bars, "funding"),
        "aggregate_total_execution_risk": _sum_column(bars, "execution_risk"),
        "aggregate_total_partial_fill_cost": _sum_column(bars, "partial_fill"),
        "aggregate_bar_rows": int(len(bars)),
        "aggregate_open_trades": int(
            sum(int(_number(row.get("open_trades")) or 0) for row in fold_rows)
        ),
    }


def _walkforward_gate_blockers(metrics: dict[str, object]) -> list[str]:
    blockers: list[str] = []
    if int(_number(metrics.get("folds_complete")) or 0) != FOLD_COUNT:
        blockers.append(f"folds_complete!={FOLD_COUNT}")
    trades = int(_number(metrics.get("aggregate_trades")) or 0)
    if trades < MIN_AGGREGATE_TRADES:
        blockers.append(f"aggregate_trades<{MIN_AGGREGATE_TRADES}")
    profit_factor = _number(metrics.get("aggregate_profit_factor"))
    if profit_factor is None or not math.isfinite(profit_factor):
        blockers.append("aggregate_profit_factor_not_finite")
    elif profit_factor <= MIN_PROFIT_FACTOR:
        blockers.append(f"aggregate_profit_factor<={MIN_PROFIT_FACTOR:g}")
    total_return = _number(metrics.get("aggregate_total_return"))
    if total_return is None or total_return <= 0.0:
        blockers.append("aggregate_after_cost_total_return_not_positive")
    if _text(metrics.get("aggregate_sharpe_status")) != "valid":
        blockers.append("aggregate_sharpe_not_valid")
    elif (_number(metrics.get("aggregate_sharpe")) or 0.0) <= 0.0:
        blockers.append("aggregate_sharpe_not_positive")
    drawdown = _number(metrics.get("aggregate_max_drawdown"))
    if drawdown is None or drawdown > MAX_DRAWDOWN:
        blockers.append(f"aggregate_max_drawdown>{MAX_DRAWDOWN:g}")
    positive_folds = int(_number(metrics.get("positive_folds")) or 0)
    if positive_folds < MIN_POSITIVE_FOLDS:
        blockers.append(f"positive_folds<{MIN_POSITIVE_FOLDS}")
    concentration = _number(metrics.get("positive_fold_profit_concentration"))
    if concentration is None or concentration > MAX_POSITIVE_FOLD_CONCENTRATION:
        blockers.append(
            f"positive_fold_profit_concentration>{MAX_POSITIVE_FOLD_CONCENTRATION:g}"
        )
    if int(_number(metrics.get("aggregate_open_trades")) or 0) != 0:
        blockers.append("aggregate_open_trades_not_zero")
    return blockers


def _add_statistical_selection_controls(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    controlled = candidates.copy()
    controlled["family_tests"] = len(controlled)
    controlled["bh_qvalue"] = _benjamini_hochberg(
        pd.to_numeric(controlled["fold_return_raw_pvalue"], errors="coerce")
    )
    controlled["false_discovery_rate"] = FALSE_DISCOVERY_RATE
    controlled["parameter_stability_status"] = np.where(
        pd.to_numeric(controlled["hedge_ratio_cv"], errors="coerce").le(MAX_HEDGE_RATIO_CV),
        "PASS",
        "BLOCKED",
    )
    controlled["deflated_sharpe_status"] = np.where(
        pd.to_numeric(controlled["folds_complete"], errors="coerce").ge(5),
        "PASS_FOLD_COUNT_PREREQUISITE",
        "BLOCKED_FEWER_THAN_5_FOLDS",
    )
    statuses: list[str] = []
    blockers: list[str] = []
    for row in controlled.itertuples():
        row_blockers: list[str] = []
        qvalue = _number(row.bh_qvalue)
        if qvalue is None or not math.isfinite(qvalue) or qvalue > FALSE_DISCOVERY_RATE:
            row_blockers.append("false_discovery_gate_failed")
        if _text(row.parameter_stability_status) != "PASS":
            row_blockers.append("hedge_ratio_instability")
        if _text(row.deflated_sharpe_status) != "PASS_FOLD_COUNT_PREREQUISITE":
            row_blockers.append("deflated_sharpe_fold_count_prerequisite_failed")
        if _text(row.walkforward_status) != "PASS_RESEARCH_WALK_FORWARD":
            row_blockers.append("research_walk_forward_gate_failed")
        statuses.append("PASS" if not row_blockers else "BLOCKED")
        blockers.append(";".join(row_blockers))
    controlled["statistical_selection_status"] = statuses
    controlled["statistical_selection_blocker"] = blockers
    return controlled


def _benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    values = pd.to_numeric(pvalues, errors="coerce")
    result = pd.Series(float("nan"), index=values.index, dtype="float64")
    valid = values.dropna().clip(0.0, 1.0).sort_values()
    if valid.empty:
        return result
    count = len(valid)
    adjusted = valid * count / np.arange(1, count + 1)
    adjusted = pd.Series(
        np.minimum.accumulate(adjusted.iloc[::-1])[::-1],
        index=valid.index,
    ).clip(upper=1.0)
    result.loc[adjusted.index] = adjusted
    return result


def _one_sided_mean_pvalue(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 3:
        return float("nan")
    standard_error = float(clean.std(ddof=1) / math.sqrt(len(clean)))
    if not math.isfinite(standard_error) or standard_error <= 0.0:
        return 0.0 if float(clean.mean()) > 0.0 else 1.0
    zscore = float(clean.mean() / standard_error)
    return float(0.5 * math.erfc(zscore / math.sqrt(2.0)))


def _status_base(
    row: object,
    *,
    walkforward_id: str,
    evidence_paths: object,
    root: Path,
) -> dict[str, object]:
    return {
        **_identity_fields(row),
        "schema_version": SCHEMA_VERSION,
        "walkforward_id": walkforward_id,
        "observed_cost_replay_id": _text(getattr(row, "observed_cost_replay_id", "")),
        "prior_replay_status": _text(row.replay_status),
        "prior_replay_blocker": _text(getattr(row, "replay_blocker", "")),
        "primary_candidate": _truthy(getattr(row, "research_rank_eligible", False)),
        "canonical_replay_leverage": 1.0,
        "walkforward_status": "",
        "walkforward_blocker": "",
        "acceptance_status": "BLOCKED",
        "acceptance_reason": "walk_forward_not_complete",
        "acceptance_eligible": False,
        "evidence_path": ";".join(
            _relative(Path(path), root) for path in evidence_paths
        ),
        "live_trading_authorized": False,
    }


def _identity_fields(row: object) -> dict[str, object]:
    return {
        "exhaustive_run_id": _text(getattr(row, "exhaustive_run_id", "")),
        "experiment_id": _text(row.experiment_id),
        "pair_group_id": _text(row.pair_group_id),
        "pair": _text(row.pair),
        "wizard_exchange": _text(getattr(row, "wizard_exchange", "")),
        "wizard_timeframe": _text(getattr(row, "wizard_timeframe", "")),
        "hyperliquid_interval": _text(getattr(row, "hyperliquid_interval", "")),
        "exact_mode": _text(row.exact_mode),
        "orientation": _text(row.orientation),
        "asset_x": _text(getattr(row, "asset_x", "")),
        "asset_y": _text(getattr(row, "asset_y", "")),
    }


def _rank_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    ranked = candidates.copy()
    ranked["_passed"] = ranked["walkforward_status"].eq("PASS_RESEARCH_WALK_FORWARD")
    ranked = ranked.sort_values(
        [
            "_passed",
            "aggregate_profit_factor",
            "aggregate_sharpe",
            "aggregate_max_drawdown",
            "aggregate_trades",
        ],
        ascending=[False, False, False, True, False],
        na_position="last",
    ).drop(columns="_passed").reset_index(drop=True)
    ranked.insert(0, "walkforward_rank", range(1, len(ranked) + 1))
    return ranked


def _row_lookup(frame: pd.DataFrame, key: str) -> dict[str, object]:
    return {_text(getattr(row, key)): row for row in frame.itertuples()}


def _require_unique(frame: pd.DataFrame, key: str) -> None:
    if frame.empty or key not in frame.columns or frame[key].astype(str).duplicated().any():
        raise ValueError(f"Expected non-empty unique {key} rows")


def _status_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        _text(status) or "MISSING_STATUS": int(count)
        for status, count in frame[column].value_counts(dropna=False).items()
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Exhaustive Hyperliquid Purged Walk-Forward Research",
            "",
            f"- Run: `{summary['run_id']}`",
            f"- Observed-cost replay: `{summary['observed_cost_replay_id']}`",
            f"- Walk-forward: `{summary['walkforward_id']}`",
            f"- Experiments accounted: {summary['unique_experiment_ids']} / {summary['experiments']}",
            f"- Completed observed replays considered: {summary['completed_observed_replays']}",
            f"- Walk-forward candidates completed: {summary['walkforward_candidates_completed']}",
            f"- Primary full-sample cohort completed: {summary['primary_candidates_completed']}",
            f"- Research walk-forward passes: {summary['walkforward_passes']}",
            f"- Multiple-testing selection passes: {summary['statistical_selection_passes']}",
            f"- Folds complete: {summary['folds_complete']} / {summary['folds_expected']}",
            f"- Closed-trade rows: {summary['closed_trade_rows']}",
            f"- Acceptance-eligible replays: {summary['acceptance_eligible_replays']}",
            f"- Live trading authorized: `{str(summary['live_trading_authorized']).lower()}`",
            "",
            "Every completed observed-cost replay is evaluated when its history supports three purged expanding folds. Full-sample rank is a label only and does not select fold execution. Each fold starts flat, waits out inherited positions, refits hedge and OU parameters on training data only, and force-closes at the fold boundary.",
            "",
            "Passing this research gate does not grant acceptance. L2 depth calibration, exact Wizard parity, leverage stress, and Hyperliquid Testnet lifecycle evidence remain separate required stages.",
            "",
        ]
    )


def _required_number(value: object, name: str) -> float:
    parsed = _number(value)
    if parsed is None or not math.isfinite(parsed):
        raise ValueError(f"{name}_missing_or_nonfinite")
    return parsed


def _number(value: object) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else None


def _sum_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _timestamp(value: object) -> str:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return parsed.isoformat() if pd.notna(parsed) else ""


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
