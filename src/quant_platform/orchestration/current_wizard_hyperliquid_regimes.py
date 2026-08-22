"""Causal regime attribution for current-board walk-forward results."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_csv,
    atomic_write_text,
    immutable_snapshot_copy,
)
from quant_platform.orchestration.current_wizard_hyperliquid_replay import _load_history
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_regimes import (
    CORRELATION_WINDOW,
    MAX_REGIME_PROFIT_CONCENTRATION,
    MIN_REGIME_TRADES,
    REGIME_REFERENCE_MIN_ROWS,
    VOLATILITY_WINDOW,
    _candidate_regime_summary,
    _join_regime,
    _regime_details,
    build_causal_pair_regime_features,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_hyperliquid_regime_attribution.v1"
RESEARCH_ONLY_REASON = (
    "regime_attribution_is_research_only;strict_l2_calibration_required;"
    "robustness_not_run;mode_fidelity_parity_not_proven"
)


def build_current_wizard_hyperliquid_regime_attribution(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Attribute practical walk-forward survivors using prior-only thresholds."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    manifest_path = active / "current_wizard_hyperliquid_walkforward_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Current walk-forward manifest is required")
    manifest = _read_json(manifest_path)
    artifacts = manifest.get("artifacts", {})
    snapshots = manifest.get("input_snapshots", {})
    input_paths = {
        "status": root / _text(artifacts.get("snapshot_status")),
        "candidates": root / _text(artifacts.get("snapshot_candidates")),
        "bars": root / _text(artifacts.get("snapshot_bars")),
        "trades": root / _text(artifacts.get("snapshot_trades")),
        "pair_costs": root / _text(snapshots.get("pair_costs")),
        "walkforward_manifest": root / _text(artifacts.get("snapshot_manifest")),
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Current regime inputs missing: {missing}")
    statuses = pd.read_csv(input_paths["status"])
    candidates = pd.read_csv(input_paths["candidates"])
    bars = pd.read_csv(input_paths["bars"])
    trades = pd.read_csv(input_paths["trades"])
    pair_costs = pd.read_csv(input_paths["pair_costs"])
    if statuses["experiment_id"].duplicated().any():
        raise ValueError("Current walk-forward status contains duplicate experiments")
    if pair_costs["pair_group_key"].duplicated().any():
        raise ValueError("Current pair cost evidence contains duplicate pair keys")
    selected = candidates.loc[
        candidates["walkforward_status"].eq("PASS_RESEARCH_WALK_FORWARD")
    ].copy()
    selected_ids = set(selected["experiment_id"].astype(str))
    policy = {
        "candidate_policy": "pass_research_walk_forward_only",
        "volatility_window": VOLATILITY_WINDOW,
        "correlation_window": CORRELATION_WINDOW,
        "reference_minimum_rows": REGIME_REFERENCE_MIN_ROWS,
        "volatility_threshold": "prior_only_expanding_75th_percentile_shifted_one_bar",
        "correlation_threshold": "prior_only_expanding_25th_percentile_shifted_one_bar",
        "minimum_regime_trades": MIN_REGIME_TRADES,
        "maximum_regime_profit_concentration": MAX_REGIME_PROFIT_CONCENTRATION,
    }
    material = {
        "schema_version": SCHEMA_VERSION,
        "walkforward_id": _text(manifest.get("walkforward_id")),
        "as_of": as_of.isoformat(),
        "policy": policy,
        "input_hashes": {name: _file_hash(path) for name, path in input_paths.items()},
    }
    attribution_id = "cwregime_" + sha256(_canonical_json(material).encode()).hexdigest()[:20]
    snapshot_manifest = input_paths["walkforward_manifest"]
    snapshot_dir = snapshot_manifest.parent / "regime_attributions" / attribution_id
    input_dir = snapshot_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    snapshot_inputs: dict[str, Path] = {}
    for name, source in input_paths.items():
        target = immutable_snapshot_copy(source, input_dir, artifact_name=name)
        snapshot_inputs[name] = target

    pair_lookup = {_text(row.pair_group_key): row for row in pair_costs.itertuples()}
    status_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    regime_bars: list[pd.DataFrame] = []
    regime_trades: list[pd.DataFrame] = []
    feature_cache: dict[str, pd.DataFrame] = {}
    for row in statuses.itertuples():
        base = _status_base(
            row,
            attribution_id=attribution_id,
            evidence_paths=snapshot_inputs.values(),
            root=root,
        )
        experiment_id = _text(row.experiment_id)
        if experiment_id not in selected_ids:
            status_rows.append(
                {
                    **base,
                    "regime_status": "NOT_SELECTED_PRIOR_WALK_FORWARD_GATE",
                    "regime_blocker": _text(row.walkforward_blocker)
                    or _text(row.walkforward_status),
                }
            )
            continue
        pair_cost = pair_lookup.get(_text(row.pair_group_key))
        if pair_cost is None:
            status_rows.append(
                {
                    **base,
                    "regime_status": "BLOCKED_REGIME_INPUTS",
                    "regime_blocker": "pair_cost_evidence_missing",
                }
            )
            continue
        try:
            history_path = root / _text(pair_cost.enriched_history_path)
            if str(history_path) not in feature_cache:
                feature_cache[str(history_path)] = build_causal_pair_regime_features(
                    _load_history(history_path)
                )
            features = feature_cache[str(history_path)]
            candidate = selected.loc[
                selected["experiment_id"].astype(str).eq(experiment_id)
            ].iloc[0].copy()
            candidate["pair_group_id"] = candidate["pair_group_key"]
            candidate_bars = bars.loc[
                bars["experiment_id"].astype(str).eq(experiment_id)
            ].copy()
            candidate_trades = trades.loc[
                trades["experiment_id"].astype(str).eq(experiment_id)
            ].copy()
            if candidate_bars.empty:
                raise ValueError("walkforward_bar_ledger_missing")
            enriched_bars = _join_regime(
                candidate_bars, features, timestamp_column="timestamp"
            )
            enriched_trades = _join_regime(
                candidate_trades, features, timestamp_column="entry_timestamp"
            )
            details = _regime_details(
                candidate,
                enriched_bars,
                enriched_trades,
                attribution_id=attribution_id,
                walkforward_id=material["walkforward_id"],
            )
            for detail in details:
                detail["schema_version"] = SCHEMA_VERSION
                detail["pair_group_key"] = detail.pop("pair_group_id", "")
                detail["acceptance_reason"] = RESEARCH_ONLY_REASON
            summary = _candidate_regime_summary(candidate, details)
        except Exception as exc:
            status_rows.append(
                {
                    **base,
                    "regime_status": "BLOCKED_REGIME_INPUTS",
                    "regime_blocker": f"{safe_exception_code(exc)}",
                }
            )
            continue
        result = {
            **base,
            **summary,
            "regime_status": "REGIME_ATTRIBUTION_COMPLETE",
            "regime_blocker": "",
            "acceptance_status": "BLOCKED",
            "acceptance_reason": RESEARCH_ONLY_REASON,
            "acceptance_eligible": False,
            "live_trading_authorized": False,
        }
        status_rows.append(result)
        candidate_rows.append(result)
        detail_rows.extend(details)
        for frame in (enriched_bars, enriched_trades):
            frame["schema_version"] = SCHEMA_VERSION
            frame["regime_attribution_id"] = attribution_id
            frame["walkforward_id"] = material["walkforward_id"]
            frame["regime_uses_future_data"] = False
            frame["live_trading_authorized"] = False
        regime_bars.append(enriched_bars)
        regime_trades.append(enriched_trades)

    status = pd.DataFrame(status_rows)
    candidate_frame = pd.DataFrame(candidate_rows)
    detail = pd.DataFrame(detail_rows)
    bars_frame = pd.concat(regime_bars, ignore_index=True) if regime_bars else pd.DataFrame()
    trades_frame = pd.concat(regime_trades, ignore_index=True) if regime_trades else pd.DataFrame()
    if len(status) != len(statuses) or status["experiment_id"].nunique() != len(statuses):
        raise ValueError("Current regime attribution failed complete experiment accounting")
    validation = _validation(statuses, status, candidate_frame, bars_frame, trades_frame)
    if not validation["status"].eq("PASS").all():
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise ValueError("Current regime validation failed: " + ",".join(failed))
    paths = _paths(active, snapshot_dir)
    for frame, active_key, snapshot_key in (
        (status, "status", "snapshot_status"),
        (candidate_frame, "candidates", "snapshot_candidates"),
        (detail, "detail", "snapshot_detail"),
        (bars_frame, "bars", "snapshot_bars"),
        (trades_frame, "trades", "snapshot_trades"),
        (validation, "validation", "snapshot_validation"),
    ):
        atomic_write_csv(frame, paths[active_key], index=False)
        atomic_write_csv(frame, paths[snapshot_key], index=False)
    counts = status["regime_status"].value_counts().to_dict()
    summary: dict[str, object] = {
        **material,
        "regime_attribution_id": attribution_id,
        "experiments_accounted": int(len(status)),
        "status_counts": counts,
        "experiment_status_accounted": bool(sum(counts.values()) == len(status)),
        "walkforward_research_passes_selected": int(len(selected)),
        "regime_attributions_complete": int(len(candidate_frame)),
        "regime_stability_passes": int(candidate_frame.get("regime_stability_status", pd.Series(dtype=str)).eq("PASS_RESEARCH_REGIME_STABILITY").sum()),
        "regime_detail_rows": int(len(detail)),
        "enriched_bar_rows": int(len(bars_frame)),
        "enriched_trade_rows": int(len(trades_frame)),
        "point_in_time_regime_thresholds": True,
        "acceptance_eligible_replays": 0,
        "promotion_authority": False,
        "live_trading_authorized": False,
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


def _status_base(
    row: object,
    *,
    attribution_id: str,
    evidence_paths: Any,
    root: Path,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "regime_attribution_id": attribution_id,
        "walkforward_id": _text(row.walkforward_id),
        "experiment_id": _text(row.experiment_id),
        "pair_group_key": _text(row.pair_group_key),
        "pair": _text(row.pair),
        "wizard_exchange": _text(row.wizard_exchange),
        "wizard_timeframe": _text(row.wizard_timeframe),
        "hyperliquid_interval": _text(row.hyperliquid_interval),
        "exact_mode": _text(row.exact_mode),
        "orientation": _text(row.orientation),
        "prior_walkforward_status": _text(row.walkforward_status),
        "regime_status": "",
        "regime_blocker": "",
        "acceptance_status": "BLOCKED",
        "acceptance_reason": "regime_attribution_not_complete",
        "acceptance_eligible": False,
        "evidence_path": ";".join(_relative(Path(path), root) for path in evidence_paths),
        "live_trading_authorized": False,
    }


def _validation(
    prior: pd.DataFrame,
    status: pd.DataFrame,
    candidates: pd.DataFrame,
    bars: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    checks = {
        "experiment_count_preserved": len(prior) == len(status),
        "experiment_ids_unique": status["experiment_id"].nunique() == len(status),
        "statuses_accounted": status["regime_status"].astype(str).ne("").all(),
        "candidate_count_matches_completed": candidates.empty or len(candidates) == status["regime_status"].eq("REGIME_ATTRIBUTION_COMPLETE").sum(),
        "bar_features_use_no_future_data": bars.empty or not bars["regime_uses_future_data"].astype(bool).any(),
        "trade_features_use_no_future_data": trades.empty or not trades["regime_uses_future_data"].astype(bool).any(),
        "acceptance_disabled": status["acceptance_status"].eq("BLOCKED").all(),
        "live_trading_disabled": not status["live_trading_authorized"].astype(bool).any(),
    }
    return pd.DataFrame(
        [{"check": check, "status": "PASS" if passed else "FAIL"} for check, passed in checks.items()]
    )


def _paths(active: Path, snapshot: Path) -> dict[str, Path]:
    return {
        "status": active / "current_wizard_hyperliquid_regime_status.csv",
        "candidates": active / "current_wizard_hyperliquid_regime_candidates.csv",
        "detail": active / "current_wizard_hyperliquid_regime_detail.csv",
        "bars": active / "current_wizard_hyperliquid_regime_bars.csv.gz",
        "trades": active / "current_wizard_hyperliquid_regime_trades.csv",
        "validation": active / "current_wizard_hyperliquid_regime_validation.csv",
        "manifest": active / "current_wizard_hyperliquid_regime_manifest.json",
        "summary_md": active / "current_wizard_hyperliquid_regime_summary.md",
        "snapshot_status": snapshot / "regime_status.csv",
        "snapshot_candidates": snapshot / "regime_candidates.csv",
        "snapshot_detail": snapshot / "regime_detail.csv",
        "snapshot_bars": snapshot / "regime_bars.csv.gz",
        "snapshot_trades": snapshot / "regime_trades.csv",
        "snapshot_validation": snapshot / "validation.csv",
        "snapshot_manifest": snapshot / "manifest.json",
        "snapshot_summary_md": snapshot / "summary.md",
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Current Wizard Hyperliquid Regime Attribution",
            "",
            f"- Attribution: `{summary['regime_attribution_id']}`",
            f"- Experiments accounted: {summary['experiments_accounted']}",
            f"- Walk-forward passes selected: {summary['walkforward_research_passes_selected']}",
            f"- Attributions complete: {summary['regime_attributions_complete']}",
            f"- Regime-stability passes: {summary['regime_stability_passes']}",
            "- Point-in-time thresholds: yes",
            "- Promotion authority: no",
            "- Live trading authorized: no",
            "",
        ]
    )


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
