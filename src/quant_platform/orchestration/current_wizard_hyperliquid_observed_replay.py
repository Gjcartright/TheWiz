"""Observed-cost replay for current Wizard to Hyperliquid hypotheses."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.backtest import (
    CostModel,
    FundingPolicy,
    backtest_two_leg_spread_with_ledger,
)
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_csv,
    atomic_write_text,
    immutable_snapshot_copy,
)
from quant_platform.orchestration.current_wizard_hyperliquid_replay import (
    MINIMUM_RESEARCH_RANK_TRADES,
    _entry_style,
    _exit_style,
    _exposure_hedge_ratio,
    _load_history,
    _mode_equivalence_warning,
    _orient_history,
    _rank_blocker,
)
from quant_platform.wizard_mode_replay import build_local_mode_signal

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_hyperliquid_observed_cost_replay.v1"
ACCEPTANCE_BLOCKERS = (
    "walk_forward_not_run;regime_robustness_not_run;parameter_sensitivity_not_run;"
    "failure_attribution_not_run;local_formula_approximation;mode_fidelity_parity_not_proven"
)


def run_current_wizard_hyperliquid_observed_cost_replay(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Rerun every completed current cell with observed funding and L2 costs."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    cost_manifest_path = active / "current_wizard_hyperliquid_cost_manifest.json"
    if not cost_manifest_path.exists():
        raise FileNotFoundError("Current cost evidence manifest is required")
    cost_manifest = _read_json(cost_manifest_path)
    artifacts = cost_manifest.get("artifacts", {})
    input_paths = {
        "canonical": root / _text(cost_manifest.get("input_snapshots", {}).get("canonical_replay")),
        "pair_costs": root / _text(artifacts.get("snapshot_pairs")),
        "experiment_costs": root / _text(artifacts.get("snapshot_experiments")),
        "cost_manifest": root / _text(artifacts.get("snapshot_manifest")),
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Current observed-cost replay inputs missing: {missing}")
    canonical = pd.read_csv(input_paths["canonical"])
    pair_costs = pd.read_csv(input_paths["pair_costs"])
    experiment_costs = pd.read_csv(input_paths["experiment_costs"])
    if (
        canonical["experiment_id"].duplicated().any()
        or experiment_costs["experiment_id"].duplicated().any()
    ):
        raise ValueError("Current observed-cost inputs contain duplicate experiment ids")
    if set(canonical["experiment_id"].astype(str)) != set(
        experiment_costs["experiment_id"].astype(str)
    ):
        raise ValueError("Current cost readiness does not account for canonical experiments")
    if pair_costs["pair_group_key"].duplicated().any():
        raise ValueError("Current pair cost evidence contains duplicate pair keys")

    material = {
        "schema_version": SCHEMA_VERSION,
        "cost_evidence_id": _text(cost_manifest.get("cost_evidence_id")),
        "canonical_replay_id": _text(cost_manifest.get("canonical_replay_id")),
        "as_of": as_of.isoformat(),
        "input_hashes": {name: _file_hash(path) for name, path in input_paths.items()},
    }
    observed_replay_id = "cwobserved_" + sha256(_canonical_json(material).encode()).hexdigest()[:20]
    cost_snapshot_manifest = input_paths["cost_manifest"]
    snapshot_dir = cost_snapshot_manifest.parent / "observed_replays" / observed_replay_id
    input_dir = snapshot_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    snapshot_inputs: dict[str, Path] = {}
    for name, source in input_paths.items():
        target = immutable_snapshot_copy(source, input_dir, artifact_name=name)
        snapshot_inputs[name] = target

    pair_lookup = {_text(row.pair_group_key): row for row in pair_costs.itertuples()}
    readiness_lookup = {_text(row.experiment_id): row for row in experiment_costs.itertuples()}
    history_cache: dict[str, pd.DataFrame] = {}
    result_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    for row in canonical.itertuples():
        experiment_id = _text(row.experiment_id)
        pair_key = _text(row.pair_group_key)
        exact_mode = _text(row.exact_mode)
        orientation = _text(row.orientation)
        pair_cost = pair_lookup.get(pair_key)
        readiness = readiness_lookup.get(experiment_id)
        base = _base_row(
            row,
            observed_replay_id=observed_replay_id,
            cost_evidence_id=material["cost_evidence_id"],
            evidence_paths=snapshot_inputs.values(),
            root=root,
        )
        if _text(row.replay_status) != "RESEARCH_REPLAY_COMPLETE":
            result_rows.append(
                {
                    **base,
                    "replay_status": "NOT_RUN_CANONICAL_REPLAY_BLOCKED",
                    "replay_blocker": _text(row.replay_blocker) or _text(row.replay_status),
                }
            )
            continue
        if pair_cost is None or readiness is None:
            result_rows.append(
                {
                    **base,
                    "replay_status": "BLOCKED_COST_EVIDENCE_MISSING",
                    "replay_blocker": "pair_or_experiment_cost_evidence_missing",
                }
            )
            continue
        if _text(readiness.cost_replay_status) != "READY_FOR_OBSERVED_COST_RESEARCH":
            result_rows.append(
                {
                    **base,
                    "replay_status": "BLOCKED_COST_EVIDENCE",
                    "replay_blocker": _text(readiness.cost_replay_blocker)
                    or _text(readiness.cost_replay_status),
                }
            )
            continue
        try:
            history_path = root / _text(pair_cost.enriched_history_path)
            if str(history_path) not in history_cache:
                history_cache[str(history_path)] = _load_history(history_path)
            oriented = _orient_history(history_cache[str(history_path)], orientation=orientation)
            test_rows = int(row.test_rows)
            if test_rows <= 0 or test_rows > len(oriented):
                raise ValueError("canonical_test_window_invalid")
            test = oriented.iloc[-test_rows:].copy()
            prior_history = oriented.iloc[:-test_rows].copy()
            settings = json.loads(_text(row.settings_json))
            test["hedge_ratio"] = _exposure_hedge_ratio(
                test,
                exact_mode=exact_mode,
                settings=settings,
                prior_history=prior_history,
            )
            mode_result = build_local_mode_signal(test, settings, exact_mode=exact_mode)
            if mode_result.mode_replay_status != "READY_FOR_RESEARCH_REPLAY":
                raise ValueError("mode_inputs_unavailable:" + ";".join(mode_result.missing_inputs))
            costs = _cost_model(pair_cost, timeframe=_text(row.timeframe))
            result, ledger = backtest_two_leg_spread_with_ledger(
                test,
                mode_result.signal,
                costs,
                interval=_interval(_text(row.timeframe)),
            )
        except Exception as exc:
            result_rows.append(
                {
                    **base,
                    "replay_status": "BLOCKED_REPLAY_ERROR",
                    "replay_blocker": f"{safe_exception_code(exc)}",
                }
            )
            continue
        strict_ready = bool(pair_cost.cost_acceptance_ready)
        acceptance_reason = ACCEPTANCE_BLOCKERS
        if not strict_ready:
            acceptance_reason = "strict_l2_calibration_incomplete;" + acceptance_reason
        result_rows.append(
            {
                **base,
                **asdict(result),
                "replay_status": "OBSERVED_COST_RESEARCH_REPLAY_COMPLETE",
                "replay_blocker": "",
                "metric_name": mode_result.metric_name,
                "mode_fidelity_status": mode_result.mode_fidelity_status,
                "mode_fidelity_reason": mode_result.mode_fidelity_reason,
                "settings_json": _canonical_json(settings),
                "entry_style": _entry_style(settings, exact_mode),
                "exit_style": _exit_style(settings, exact_mode),
                "mode_equivalence_warning": _mode_equivalence_warning(exact_mode),
                "strict_cost_calibration_ready": strict_ready,
                "funding_evidence_attached": True,
                "slippage_evidence_attached": True,
                "pair_one_way_slippage_bps": float(pair_cost.pair_one_way_slippage_bps),
                "estimated_pair_round_trip_cost_bps": float(
                    pair_cost.estimated_pair_round_trip_cost_bps
                ),
                "acceptance_status": "BLOCKED",
                "acceptance_reason": acceptance_reason,
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )
        for lifecycle, frame in (("closed", ledger.closed_trades), ("open", ledger.open_trades)):
            for trade in frame.to_dict("records"):
                trade_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "observed_cost_replay_id": observed_replay_id,
                        "cost_evidence_id": material["cost_evidence_id"],
                        "experiment_id": experiment_id,
                        "pair_group_key": pair_key,
                        "pair": _text(row.pair),
                        "exact_mode": exact_mode,
                        "orientation": orientation,
                        "trade_lifecycle": lifecycle,
                        **trade,
                        "backtest_label": True,
                        "paper_label": False,
                        "live_label": False,
                        "promotion_authority": False,
                        "live_trading_authorized": False,
                    }
                )

    results = pd.DataFrame(result_rows)
    if len(results) != len(canonical) or results["experiment_id"].nunique() != len(canonical):
        raise ValueError("Observed-cost replay failed complete experiment accounting")
    complete = results["replay_status"].eq("OBSERVED_COST_RESEARCH_REPLAY_COMPLETE")
    results["research_rank_eligible"] = False
    results["research_rank_blocker"] = "replay_not_complete"
    blockers = results.loc[complete].apply(_rank_blocker, axis=1)
    results.loc[complete, "research_rank_blocker"] = blockers
    results.loc[complete, "research_rank_eligible"] = blockers.eq("")
    ranked = _ranked(results.loc[complete].copy())
    trades = pd.DataFrame(trade_rows)
    validation = _validation(canonical, results)
    if not validation["status"].eq("PASS").all():
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise ValueError("Observed-cost replay validation failed: " + ",".join(failed))
    paths = {
        "results": active / "current_wizard_hyperliquid_observed_cost_replay.csv",
        "ranked": active / "current_wizard_hyperliquid_observed_cost_replay_ranked.csv",
        "trades": active / "current_wizard_hyperliquid_observed_cost_replay_trades.csv",
        "validation": active / "current_wizard_hyperliquid_observed_cost_replay_validation.csv",
        "manifest": active / "current_wizard_hyperliquid_observed_cost_replay_manifest.json",
        "summary_md": active / "current_wizard_hyperliquid_observed_cost_replay_summary.md",
        "snapshot_results": snapshot_dir / "observed_cost_replay.csv",
        "snapshot_ranked": snapshot_dir / "observed_cost_replay_ranked.csv",
        "snapshot_trades": snapshot_dir / "observed_cost_replay_trades.csv",
        "snapshot_validation": snapshot_dir / "validation.csv",
        "snapshot_manifest": snapshot_dir / "manifest.json",
        "snapshot_summary_md": snapshot_dir / "summary.md",
    }
    for frame, active_key, snapshot_key in (
        (results, "results", "snapshot_results"),
        (ranked, "ranked", "snapshot_ranked"),
        (trades, "trades", "snapshot_trades"),
        (validation, "validation", "snapshot_validation"),
    ):
        atomic_write_csv(frame, paths[active_key], index=False)
        atomic_write_csv(frame, paths[snapshot_key], index=False)
    status_counts = results["replay_status"].value_counts().to_dict()
    summary: dict[str, object] = {
        **material,
        "observed_cost_replay_id": observed_replay_id,
        "experiments_accounted": int(len(results)),
        "replays_complete": int(status_counts.get("OBSERVED_COST_RESEARCH_REPLAY_COMPLETE", 0)),
        "research_rank_eligible_replays": int(results["research_rank_eligible"].astype(bool).sum()),
        "strict_cost_calibrated_replays": int(
            results.get("strict_cost_calibration_ready", pd.Series(False, index=results.index))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "trade_ledger_rows": int(len(trades)),
        "minimum_research_rank_trades": MINIMUM_RESEARCH_RANK_TRADES,
        "acceptance_eligible_replays": 0,
        "promotion_authority": False,
        "live_trading_authorized": False,
        "status_counts": status_counts,
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
        "input_snapshots": {name: _relative(path, root) for name, path in snapshot_inputs.items()},
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    atomic_write_text(paths["manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["summary_md"], summary_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_summary_md"], summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _cost_model(pair_cost: object, *, timeframe: str) -> CostModel:
    return CostModel(
        taker_fee_bps=float(pair_cost.taker_fee_bps),
        slippage_bps=float(pair_cost.pair_one_way_slippage_bps),
        execution_risk_bps=float(pair_cost.execution_risk_bps),
        funding_bps_per_day=0.0,
        bars_per_day=24 if _interval(timeframe) == "1h" else 1,
        funding_policy=FundingPolicy.SIGNED_REALIZED.value,
    )


def _base_row(
    row: object,
    *,
    observed_replay_id: str,
    cost_evidence_id: str,
    evidence_paths: Any,
    root: Path,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "observed_cost_replay_id": observed_replay_id,
        "cost_evidence_id": cost_evidence_id,
        "canonical_replay_id": _text(row.canonical_replay_id),
        "experiment_id": _text(row.experiment_id),
        "pair_group_key": _text(row.pair_group_key),
        "pair": _text(row.pair),
        "wizard_exchange": _text(row.wizard_exchange),
        "timeframe": _text(row.timeframe),
        "exact_mode": _text(row.exact_mode),
        "orientation": _text(row.orientation),
        "canonical_replay_leverage": 1.0,
        "evidence_path": ";".join(_relative(path, root) for path in evidence_paths),
        "promotion_authority": False,
        "live_trading_authorized": False,
    }


def _ranked(completed: pd.DataFrame) -> pd.DataFrame:
    if completed.empty:
        return completed
    eligible = completed.loc[completed["research_rank_eligible"].astype(bool)].sort_values(
        ["profit_factor", "sharpe", "max_drawdown", "trades"],
        ascending=[False, False, True, False],
        na_position="last",
    )
    eligible.insert(0, "research_rank", range(1, len(eligible) + 1))
    ineligible = completed.loc[~completed["research_rank_eligible"].astype(bool)].sort_values(
        ["trades", "sharpe", "max_drawdown"], ascending=[False, False, True]
    )
    ineligible.insert(0, "research_rank", "")
    return pd.concat([eligible, ineligible], ignore_index=True)


def _validation(canonical: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "experiment_count_preserved": len(canonical) == len(results),
        "experiment_ids_unique": results["experiment_id"].nunique() == len(results),
        "experiment_ids_preserved": set(canonical["experiment_id"].astype(str))
        == set(results["experiment_id"].astype(str)),
        "statuses_accounted": results["replay_status"].astype(str).ne("").all(),
        "one_x_only": results["canonical_replay_leverage"].eq(1.0).all(),
        "acceptance_disabled": results.get(
            "acceptance_status", pd.Series("BLOCKED", index=results.index)
        )
        .fillna("BLOCKED")
        .eq("BLOCKED")
        .all(),
        "live_trading_disabled": not results["live_trading_authorized"].astype(bool).any(),
    }
    return pd.DataFrame(
        [
            {"check": check, "status": "PASS" if passed else "FAIL"}
            for check, passed in checks.items()
        ]
    )


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Current Wizard Hyperliquid Observed-Cost Replay",
            "",
            f"- Replay: `{summary['observed_cost_replay_id']}`",
            f"- Experiments accounted: {summary['experiments_accounted']}",
            f"- Replays complete: {summary['replays_complete']}",
            f"- Research-rank eligible: {summary['research_rank_eligible_replays']}",
            f"- Strictly calibrated: {summary['strict_cost_calibrated_replays']}",
            "- Leverage: 1x canonical only",
            "- Funding: signed realized Hyperliquid history",
            "- Promotion authority: no",
            "- Live trading authorized: no",
            "",
        ]
    )


def _interval(timeframe: str) -> str:
    return timeframe.lower().replace("daily", "1d").replace("hourly", "1h")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
