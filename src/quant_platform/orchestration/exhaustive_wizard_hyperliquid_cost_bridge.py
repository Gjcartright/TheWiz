"""Join point-in-time Hyperliquid costs to every exhaustive Wizard experiment."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pandas as pd

from quant_platform.active_pipeline import CommandResult


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "exhaustive_wizard_hyperliquid_cost_bridge.v1"
MIN_PROVISIONAL_FUNDED_ROWS = 250
MAX_FEE_EVIDENCE_AGE_DAYS = 30


def build_exhaustive_wizard_hyperliquid_cost_evidence(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Materialize pair and experiment cost readiness without authorizing execution."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    paths_in = {
        "preflight": active / "exhaustive_wizard_hyperliquid_replay_preflight.csv",
        "pair_history": active / "exhaustive_wizard_hyperliquid_pair_history_results.csv",
        "funding_pairs": active / "exhaustive_wizard_hyperliquid_funding_pair_coverage.csv",
        "funding_assets": active / "exhaustive_wizard_hyperliquid_funding_asset_results.csv",
        "funding_manifest": active / "exhaustive_wizard_hyperliquid_funding_manifest.json",
        "cost_model": active / "hyperliquid_pair_cost_model.csv",
        "cadence": active / "hyperliquid_evidence_cadence.csv",
    }
    required = [
        paths_in["preflight"],
        paths_in["pair_history"],
        paths_in["funding_pairs"],
        paths_in["funding_assets"],
        paths_in["funding_manifest"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Exhaustive cost evidence inputs missing: {missing}")

    experiments = _read_csv(paths_in["preflight"])
    histories = _read_csv(paths_in["pair_history"])
    funding_pairs = _read_csv(paths_in["funding_pairs"])
    funding_assets = _read_csv(paths_in["funding_assets"])
    cost_model = _read_csv(paths_in["cost_model"])
    cadence = _read_csv(paths_in["cadence"])
    funding_manifest = json.loads(paths_in["funding_manifest"].read_text(encoding="utf-8"))

    run_id = _single_identity(experiments, "exhaustive_run_id")
    preflight_id = _single_identity(experiments, "replay_preflight_id")
    history_run_id = _single_identity(histories, "history_run_id")
    funding_evidence_id = _text(funding_manifest.get("funding_evidence_id"))
    _require_identity(histories, "exhaustive_run_id", run_id)
    _require_identity(histories, "replay_preflight_id", preflight_id)
    _require_identity(funding_pairs, "exhaustive_run_id", run_id)
    _require_identity(funding_pairs, "history_run_id", history_run_id)
    if funding_evidence_id != _single_identity(funding_pairs, "funding_evidence_id"):
        raise ValueError("Funding manifest and pair coverage identities do not match")
    if histories["pair_group_id"].duplicated().any():
        raise ValueError("Pair history contains duplicate pair_group_id values")
    if funding_pairs["pair_group_id"].duplicated().any():
        raise ValueError("Funding coverage contains duplicate pair_group_id values")

    material = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "preflight_id": preflight_id,
        "history_run_id": history_run_id,
        "funding_evidence_id": funding_evidence_id,
        "as_of": as_of.isoformat(),
        "input_hashes": {name: _file_hash(path) for name, path in paths_in.items()},
        "policies": {
            "minimum_provisional_funded_rows": MIN_PROVISIONAL_FUNDED_ROWS,
            "maximum_fee_evidence_age_days": MAX_FEE_EVIDENCE_AGE_DAYS,
            "slippage_acceptance_uses_source_required_samples": True,
            "provisional_replay_uses_longest_contiguous_observed_funding_segment": True,
        },
    }
    evidence_hash = sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    cost_evidence_id = f"hlcost_{evidence_hash[:20]}"
    snapshot_dir = (
        root
        / "reports"
        / "snapshots"
        / "exhaustive_wizard_hyperliquid"
        / run_id
        / "cost_evidence"
        / cost_evidence_id
    )
    input_dir = snapshot_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    snapshot_inputs: dict[str, Path] = {}
    for name, source in paths_in.items():
        if not source.exists():
            continue
        target = input_dir / source.name
        shutil.copy2(source, target)
        snapshot_inputs[name] = target

    funding_cutoff_safe = bool(
        not funding_assets.empty
        and pd.to_numeric(funding_assets["post_cutoff_rows"], errors="coerce").fillna(1).eq(0).all()
        and funding_assets["timestamp_parse_valid"].map(_truthy).all()
        and funding_assets["fetch_complete_flag"].map(_truthy).all()
    )
    funding_lookup = _index_rows(funding_pairs, "pair_group_id")
    cost_lookup = _pair_cost_lookup(cost_model)
    cadence_lookup = _index_rows(cadence, "pair")
    pair_rows = [
        _pair_cost_row(
            history,
            funding=funding_lookup.get(_text(history.pair_group_id)),
            cost=cost_lookup.get(_pair_key(history.asset_x, history.asset_y)),
            cadence_lookup=cadence_lookup,
            funding_cutoff_safe=funding_cutoff_safe,
            as_of=as_of,
            root=root,
            cost_evidence_id=cost_evidence_id,
            run_id=run_id,
            preflight_id=preflight_id,
            history_run_id=history_run_id,
            funding_evidence_id=funding_evidence_id,
            input_paths=snapshot_inputs.values(),
        )
        for history in histories.itertuples()
    ]
    pair_frame = pd.DataFrame(pair_rows)
    pair_lookup = _index_rows(pair_frame, "pair_group_id")
    experiment_rows = [
        _experiment_cost_row(
            experiment,
            pair_cost=pair_lookup.get(_text(experiment.pair_group_id)),
            cost_evidence_id=cost_evidence_id,
            history_run_id=history_run_id,
            funding_evidence_id=funding_evidence_id,
        )
        for experiment in experiments.itertuples()
    ]
    experiment_frame = pd.DataFrame(experiment_rows)

    paths = {
        "pair_cost_evidence": active / "exhaustive_wizard_hyperliquid_pair_cost_evidence.csv",
        "experiment_cost_readiness": active
        / "exhaustive_wizard_hyperliquid_experiment_cost_readiness.csv",
        "manifest": active / "exhaustive_wizard_hyperliquid_cost_evidence_manifest.json",
        "summary_md": active / "exhaustive_wizard_hyperliquid_cost_evidence_summary.md",
        "snapshot_pair_cost_evidence": snapshot_dir / "pair_cost_evidence.csv",
        "snapshot_experiment_cost_readiness": snapshot_dir / "experiment_cost_readiness.csv",
        "snapshot_manifest": snapshot_dir / "manifest.json",
        "snapshot_summary_md": snapshot_dir / "summary.md",
    }
    pair_frame.to_csv(paths["pair_cost_evidence"], index=False)
    pair_frame.to_csv(paths["snapshot_pair_cost_evidence"], index=False)
    experiment_frame.to_csv(paths["experiment_cost_readiness"], index=False)
    experiment_frame.to_csv(paths["snapshot_experiment_cost_readiness"], index=False)

    pair_status_counts = _status_counts(pair_frame, "cost_evidence_status")
    experiment_status_counts = _status_counts(experiment_frame, "cost_replay_status")
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "replay_preflight_id": preflight_id,
        "history_run_id": history_run_id,
        "funding_evidence_id": funding_evidence_id,
        "cost_evidence_id": cost_evidence_id,
        "built_at": as_of.isoformat(),
        "pair_work_items": int(len(pair_frame)),
        "pair_groups_unique": int(pair_frame["pair_group_id"].nunique()),
        "pair_status_counts": pair_status_counts,
        "pair_status_accounted": bool(sum(pair_status_counts.values()) == len(pair_frame)),
        "pairs_ready_for_provisional_cost_research": int(
            pair_frame["provisional_cost_research_ready"].map(_truthy).sum()
        ),
        "pairs_ready_for_cost_calibrated_replay": int(
            pair_frame["cost_acceptance_ready"].map(_truthy).sum()
        ),
        "experiments": int(len(experiment_frame)),
        "experiments_unique": int(experiment_frame["experiment_id"].nunique()),
        "experiment_status_counts": experiment_status_counts,
        "experiment_status_accounted": bool(
            sum(experiment_status_counts.values()) == len(experiment_frame)
        ),
        "funding_cutoff_safe": funding_cutoff_safe,
        "live_trading_authorized": False,
        "input_hashes": material["input_hashes"],
        "policies": material["policies"],
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
        "input_snapshots": {name: _relative(path, root) for name, path in snapshot_inputs.items()},
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    paths["manifest"].write_text(manifest_text, encoding="utf-8")
    paths["snapshot_manifest"].write_text(manifest_text, encoding="utf-8")
    paths["summary_md"].write_text(summary_text, encoding="utf-8")
    paths["snapshot_summary_md"].write_text(summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _pair_cost_row(
    history: object,
    *,
    funding: dict[str, object] | None,
    cost: dict[str, object] | None,
    cadence_lookup: dict[str, dict[str, object]],
    funding_cutoff_safe: bool,
    as_of: datetime,
    root: Path,
    cost_evidence_id: str,
    run_id: str,
    preflight_id: str,
    history_run_id: str,
    funding_evidence_id: str,
    input_paths: object,
) -> dict[str, object]:
    funding = funding or {}
    cost = cost or {}
    cadence = cadence_lookup.get(_text(cost.get("pair")), {})
    history_ready = _text(history.history_status) == "READY_FOR_CANONICAL_REPLAY"
    enriched_path = _text(funding.get("enriched_history_path"))
    funded_rows = _integer(funding.get("funding_both_aligned_rows"))
    contiguous_funded_rows = _integer(funding.get("funding_longest_contiguous_rows"))
    funding_ready = _truthy(funding.get("funding_acceptance_ready"))
    fee_ready = _truthy(cost.get("cost_model_ready"))
    fee_age_days = _age_days(cost.get("fee_source_checked_at"), as_of)
    fee_fresh = fee_ready and fee_age_days is not None and fee_age_days <= MAX_FEE_EVIDENCE_AGE_DAYS
    sample_x = _integer(cost.get("slippage_samples_x"))
    sample_y = _integer(cost.get("slippage_samples_y"))
    required_samples = _integer(cost.get("required_slippage_samples"))
    slippage_ready = _truthy(cost.get("slippage_model_ready"))
    l2_age_hours = _age_hours(cost.get("freshest_sample_at"), as_of)
    l2_window_hours = _number(cost.get("slippage_window_hours"))
    l2_fresh = bool(
        l2_age_hours is not None and l2_window_hours is not None and l2_age_hours <= l2_window_hours
    )
    pair_slippage = _number(cost.get("pair_one_way_slippage_bps"))
    provisional_ready = bool(
        history_ready
        and funding_cutoff_safe
        and enriched_path
        and contiguous_funded_rows >= MIN_PROVISIONAL_FUNDED_ROWS
        and fee_fresh
        and pair_slippage is not None
        and sample_x >= 1
        and sample_y >= 1
        and l2_fresh
    )
    acceptance_ready = bool(
        provisional_ready and funding_ready and slippage_ready and required_samples > 0
    )
    blockers: list[str] = []
    if not history_ready:
        blockers.append(_text(history.history_blocker) or "pair_history_not_ready")
    if not funding_cutoff_safe:
        blockers.append("funding_evidence_cutoff_or_timestamp_invalid")
    if not enriched_path or contiguous_funded_rows < MIN_PROVISIONAL_FUNDED_ROWS:
        blockers.append(
            f"observed_funding_contiguous_rows_{contiguous_funded_rows}_below_"
            f"{MIN_PROVISIONAL_FUNDED_ROWS}"
        )
    if not funding_ready:
        blockers.append(_text(funding.get("funding_blocker")) or "funding_acceptance_not_ready")
    if not cost:
        blockers.append("hyperliquid_pair_cost_model_missing")
    elif not fee_ready:
        blockers.append(_text(cost.get("cost_model_status")) or "fee_model_not_ready")
    elif not fee_fresh:
        blockers.append("fee_evidence_stale_or_unparseable")
    if pair_slippage is None or sample_x < 1 or sample_y < 1:
        blockers.append("observed_l2_slippage_unavailable")
    if not slippage_ready:
        blockers.append(
            f"slippage_calibration_incomplete:x={sample_x};y={sample_y};required={required_samples}"
        )
    if not l2_fresh:
        blockers.append("l2_cost_evidence_stale_or_unparseable")
    blockers = _deduplicate(blockers)
    if acceptance_ready:
        status = "READY_FOR_COST_CALIBRATED_REPLAY"
        next_step = "run_cost_calibrated_point_in_time_replay"
    elif provisional_ready:
        status = "READY_FOR_PROVISIONAL_COST_RESEARCH"
        next_step = "run_observed_intersection_replay_and_continue_l2_cadence"
    else:
        status = "BLOCKED_COST_EVIDENCE"
        next_step = "resolve_cost_evidence_blockers"
    evidence = [
        _text(history.evidence_path),
        _text(funding.get("evidence_path")),
        _text(cost.get("evidence_path")),
        _text(cadence.get("evidence_path")),
        *(_relative(Path(path), root) for path in input_paths),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "exhaustive_run_id": run_id,
        "replay_preflight_id": preflight_id,
        "history_run_id": history_run_id,
        "funding_evidence_id": funding_evidence_id,
        "cost_evidence_id": cost_evidence_id,
        "pair_group_id": _text(history.pair_group_id),
        "pair": _text(history.pair),
        "wizard_exchange": _text(history.wizard_exchange),
        "wizard_timeframe": _text(history.wizard_timeframe),
        "hyperliquid_interval": _text(history.hyperliquid_interval),
        "asset_x": _text(history.asset_x),
        "asset_y": _text(history.asset_y),
        "history_status": _text(history.history_status),
        "history_blocker": _text(history.history_blocker),
        "history_rows": _integer(history.history_rows),
        "funding_status": _text(funding.get("funding_status")),
        "funding_blocker": _text(funding.get("funding_blocker")),
        "funding_both_coverage": _number(funding.get("funding_both_coverage")) or 0.0,
        "funding_intersection_rows": funded_rows,
        "funding_longest_contiguous_rows": contiguous_funded_rows,
        "funding_cutoff_safe": funding_cutoff_safe,
        "funding_acceptance_ready": funding_ready,
        "enriched_history_path": enriched_path,
        "cost_model_present": bool(cost),
        "fee_profile_id": _text(cost.get("fee_profile_id")),
        "taker_fee_bps": _number(cost.get("taker_fee_bps")),
        "execution_risk_bps": _number(cost.get("execution_risk_bps")),
        "fee_source_checked_at": _text(cost.get("fee_source_checked_at")),
        "fee_evidence_age_days": fee_age_days,
        "fee_evidence_fresh": fee_fresh,
        "slippage_model_status": _text(cost.get("slippage_model_status")),
        "slippage_model_ready": slippage_ready,
        "slippage_samples_x": sample_x,
        "slippage_samples_y": sample_y,
        "required_slippage_samples": required_samples,
        "slippage_x_p95_bps": _number(cost.get("slippage_x_p95_bps")),
        "slippage_y_p95_bps": _number(cost.get("slippage_y_p95_bps")),
        "pair_one_way_slippage_bps": pair_slippage,
        "estimated_pair_round_trip_cost_bps": _number(
            cost.get("estimated_pair_round_trip_cost_bps")
        ),
        "freshest_l2_sample_at": _text(cost.get("freshest_sample_at")),
        "l2_evidence_age_hours": l2_age_hours,
        "l2_evidence_fresh": l2_fresh,
        "cadence_status": _text(cadence.get("status")),
        "next_l2_sample_due_at": _text(cadence.get("next_sample_due_at")),
        "provisional_cost_research_ready": provisional_ready,
        "provisional_cost_policy": (
            "longest_contiguous_observed_funding_segment_plus_current_l2_p95_research_only"
        ),
        "cost_acceptance_ready": acceptance_ready,
        "cost_evidence_status": status,
        "cost_blocker": ";".join(blockers),
        "next_step": next_step,
        "evidence_path": ";".join(_deduplicate([value for value in evidence if value])),
        "live_trading_authorized": False,
    }


def _experiment_cost_row(
    experiment: object,
    *,
    pair_cost: dict[str, object] | None,
    cost_evidence_id: str,
    history_run_id: str,
    funding_evidence_id: str,
) -> dict[str, object]:
    pair_cost = pair_cost or {}
    preflight_status = _text(experiment.preflight_status)
    if preflight_status == "NOT_APPLICABLE_WIZARD_MODE":
        status = "NOT_APPLICABLE_WIZARD_MODE"
        blocker = _text(experiment.preflight_blocker)
    elif preflight_status != "READY_FOR_HISTORY":
        status = "BLOCKED_BEFORE_COST_EVIDENCE"
        blocker = _text(experiment.preflight_blocker) or preflight_status
    elif _truthy(pair_cost.get("cost_acceptance_ready")):
        status = "READY_FOR_COST_CALIBRATED_REPLAY"
        blocker = ""
    elif _truthy(pair_cost.get("provisional_cost_research_ready")):
        status = "READY_FOR_PROVISIONAL_COST_RESEARCH"
        blocker = _text(pair_cost.get("cost_blocker"))
    else:
        status = "BLOCKED_COST_EVIDENCE"
        blocker = _text(pair_cost.get("cost_blocker")) or "pair_cost_evidence_missing"
    evidence = [
        _text(experiment.evidence_path),
        _text(pair_cost.get("evidence_path")),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "exhaustive_run_id": _text(experiment.exhaustive_run_id),
        "replay_preflight_id": _text(experiment.replay_preflight_id),
        "history_run_id": history_run_id,
        "funding_evidence_id": funding_evidence_id,
        "cost_evidence_id": cost_evidence_id,
        "experiment_id": _text(experiment.experiment_id),
        "pair_group_id": _text(experiment.pair_group_id),
        "pair": _text(experiment.pair),
        "wizard_exchange": _text(experiment.wizard_exchange),
        "wizard_timeframe": _text(experiment.wizard_timeframe),
        "hyperliquid_interval": _text(experiment.hyperliquid_interval),
        "exact_mode": _text(experiment.exact_mode),
        "orientation": _text(experiment.orientation),
        "asset_x": _text(experiment.asset_x),
        "asset_y": _text(experiment.asset_y),
        "preflight_status": preflight_status,
        "cost_evidence_status": _text(pair_cost.get("cost_evidence_status")),
        "provisional_cost_research_ready": _truthy(
            pair_cost.get("provisional_cost_research_ready")
        ),
        "cost_acceptance_ready": _truthy(pair_cost.get("cost_acceptance_ready")),
        "cost_replay_status": status,
        "cost_replay_blocker": blocker,
        "evidence_path": ";".join(_deduplicate([value for value in evidence if value])),
        "live_trading_authorized": False,
    }


def _pair_cost_lookup(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    for record in frame.to_dict("records"):
        key = _pair_key(record.get("asset_x"), record.get("asset_y"))
        if not key:
            continue
        if key in lookup:
            raise ValueError(f"Pair cost model contains duplicate asset key: {key}")
        lookup[key] = record
    return lookup


def _pair_key(asset_x: object, asset_y: object) -> str:
    assets = sorted({_text(asset_x).upper(), _text(asset_y).upper()} - {""})
    return "|".join(assets) if len(assets) == 2 else ""


def _index_rows(frame: pd.DataFrame, column: str) -> dict[str, dict[str, object]]:
    if frame.empty or column not in frame.columns:
        return {}
    return {
        _text(record.get(column)): record
        for record in frame.to_dict("records")
        if _text(record.get(column))
    }


def _single_identity(frame: pd.DataFrame, column: str) -> str:
    values = {_text(value) for value in frame.get(column, pd.Series(dtype=str)) if _text(value)}
    if len(values) != 1:
        raise ValueError(f"Expected one {column}, found {sorted(values)}")
    return next(iter(values))


def _require_identity(frame: pd.DataFrame, column: str, expected: str) -> None:
    if _single_identity(frame, column) != expected:
        raise ValueError(f"{column} does not match the active exhaustive run")


def _status_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        _text(status) or "MISSING_STATUS": int(count)
        for status, count in frame[column].value_counts(dropna=False).items()
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Exhaustive Wizard to Hyperliquid Cost Evidence",
            "",
            f"- Run: `{summary['run_id']}`",
            f"- Cost evidence: `{summary['cost_evidence_id']}`",
            f"- Pair work items: {summary['pair_work_items']}",
            f"- Pair statuses accounted: `{str(summary['pair_status_accounted']).lower()}`",
            "- Pairs ready for provisional cost research: "
            f"{summary['pairs_ready_for_provisional_cost_research']}",
            "- Pairs ready for cost-calibrated replay: "
            f"{summary['pairs_ready_for_cost_calibrated_replay']}",
            f"- Experiments: {summary['experiments']}",
            "- Experiment statuses accounted: "
            f"`{str(summary['experiment_status_accounted']).lower()}`",
            f"- Funding cutoff safe: `{str(summary['funding_cutoff_safe']).lower()}`",
            f"- Live trading authorized: `{str(summary['live_trading_authorized']).lower()}`",
            "",
            "Provisional research uses only the observed funding intersection and current L2 p95 estimate. It is not acceptance evidence. Cost-calibrated replay remains blocked until every funding, fee, slippage-sample, and freshness gate passes.",
            "",
        ]
    )


def _age_days(value: object, as_of: datetime) -> float | None:
    timestamp = pd.to_datetime(_text(value), utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return max(0.0, (pd.Timestamp(as_of) - timestamp).total_seconds() / 86_400.0)


def _age_hours(value: object, as_of: datetime) -> float | None:
    timestamp = pd.to_datetime(_text(value), utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return max(0.0, (pd.Timestamp(as_of) - timestamp).total_seconds() / 3_600.0)


def _number(value: object) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else None


def _integer(value: object) -> int:
    parsed = _number(value)
    return int(parsed) if parsed is not None else 0


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
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
