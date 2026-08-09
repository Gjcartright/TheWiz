from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import math

import pandas as pd

from quant_platform.wizard_run_config import WizardRunConfiguration, canonical_exact_mode
from quant_platform.wizard_policy import DEFAULT_WIZARD_DISCOVERY_POLICY, load_wizard_discovery_policy


ROOT = Path(__file__).resolve().parents[2]
WIZARD_CONTROL_SCHEMA_VERSION = "wizard_control_plane.v1"
MIN_DISCOVERY_SHARPE = DEFAULT_WIZARD_DISCOVERY_POLICY.min_sharpe
MIN_DISCOVERY_RETURN = DEFAULT_WIZARD_DISCOVERY_POLICY.min_returns_total_pct / 100.0

MANIFEST_REQUIRED_COLUMNS = {
    "schema_version",
    "sweep_id",
    "request_id",
    "strategy",
    "exchange",
    "interval",
    "config_hash",
    "status",
    "response_hash",
    "evidence_path",
    "sweep_complete",
    "discovery_authority",
}
CANDIDATE_METADATA_COLUMNS = {
    "sweep_id",
    "request_id",
    "sweep_config_hash",
    "sweep_strategy",
    "sweep_exchange",
    "sweep_interval",
    "sweep_captured_at",
    "sweep_source_timestamp",
    "sweep_response_hash",
    "sweep_evidence_path",
    "sweep_complete",
    "discovery_authority",
}
CANDIDATE_DISCOVERY_COLUMNS = {"symbol_1", "symbol_2", "sharpe", "returns_total"}
SUMMARY_REQUIRED_FIELDS = {
    "schema_version",
    "sweep_id",
    "started_at",
    "planned_cells",
    "completed_cells",
    "candidate_rows",
    "sweep_complete",
    "discovery_authority",
    "blocker",
}

HEALTH_COLUMNS = [
    "schema_version",
    "check",
    "status",
    "blocking",
    "blocker",
    "detail",
    "evidence_path",
    "checked_at",
]
CONTRACT_COLUMNS = [
    "contract",
    "check",
    "status",
    "blocking",
    "rows_checked",
    "failure_count",
    "missing_fields",
    "detail",
    "evidence_path",
]
FRESHNESS_COLUMNS = [
    "pair",
    "wizard_exchange",
    "timeframe",
    "period",
    "strategy",
    "source_timestamp",
    "captured_at",
    "source_age_hours",
    "capture_age_hours",
    "source_fresh",
    "capture_fresh",
    "freshness_blocker",
    "sweep_config_hash",
    "evidence_path",
]
SETTINGS_QUEUE_COLUMNS = [
    "priority_rank",
    "pair",
    "asset_x",
    "asset_y",
    "wizard_exchange",
    "timeframe",
    "period",
    "strategy",
    "spread_type",
    "exact_mode",
    "sharpe",
    "returns_total",
    "closed_trades",
    "passes_discovery_gate",
    "discovery_policy_schema_version",
    "discovery_policy_hash",
    "min_closed_trades_for_proof",
    "passes_research_spend_gate",
    "source_fresh",
    "control_plane_ready",
    "actionable",
    "required_settings",
    "captured_settings",
    "missing_settings",
    "capture_status",
    "discovery_config_hash",
    "candidate_config_hash",
    "settings_config_hash",
    "settings_capture_evidence_path",
    "source_timestamp",
    "captured_at",
    "evidence_path",
    "next_step",
]
LINEAGE_COLUMNS = [
    "artifact",
    "path",
    "required_for_ranking",
    "exists",
    "rows",
    "config_hash_column",
    "config_hash_coverage_pct",
    "status",
    "blocker",
]


@dataclass(frozen=True)
class CommandResult:
    paths: dict[str, Path]
    summary: dict[str, object]


def build_wizard_control_plane(
    root: Path = ROOT,
    *,
    max_age_hours: float = 24.0,
    min_sharpe: float | None = None,
    min_return: float | None = None,
    min_closed_trades_for_proof: int | None = None,
    max_settings_candidates: int = 50,
    now: datetime | None = None,
) -> CommandResult:
    """Validate a Wizard sweep before it can drive ranking or downstream work."""

    if max_age_hours <= 0:
        raise ValueError("max_age_hours must be positive")
    if max_settings_candidates <= 0:
        raise ValueError("max_settings_candidates must be positive")

    checked_at = _as_utc(now or datetime.now(timezone.utc))
    policy = load_wizard_discovery_policy(root)
    min_sharpe = policy.min_sharpe if min_sharpe is None else float(min_sharpe)
    min_return = policy.min_returns_total_pct / 100.0 if min_return is None else float(min_return)
    min_closed_trades_for_proof = (
        policy.min_closed_trades_for_proof
        if min_closed_trades_for_proof is None
        else int(min_closed_trades_for_proof)
    )
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    manifest_path = active / "wizard_sweep_manifest.csv"
    candidates_path = active / "wizard_sweep_candidates.csv"
    sweep_summary_path = active / "wizard_sweep_summary.json"

    manifest = _read_csv(manifest_path)
    candidates = _read_csv(candidates_path)
    sweep_summary = _read_json(sweep_summary_path)

    contract = _build_contract_report(
        root=root,
        manifest=manifest,
        candidates=candidates,
        summary=sweep_summary,
        manifest_path=manifest_path,
        candidates_path=candidates_path,
        summary_path=sweep_summary_path,
    )
    contract_ready = not bool(
        ((contract["blocking"] == True) & (contract["status"] == "fail")).any()  # noqa: E712
    )

    sweep_complete = _sweep_is_complete(manifest, sweep_summary)
    freshness = _build_freshness_report(
        candidates,
        summary=sweep_summary,
        checked_at=checked_at,
        max_age_hours=max_age_hours,
    )
    fresh_candidates = int(freshness["source_fresh"].sum()) if not freshness.empty else 0
    candidate_rows = len(candidates)

    provisional_ready = bool(sweep_complete and contract_ready and fresh_candidates > 0)
    settings_queue = _build_settings_queue(
        candidates,
        freshness=freshness,
        control_plane_ready=provisional_ready,
        min_sharpe=min_sharpe,
        min_return=min_return,
        min_closed_trades_for_proof=min_closed_trades_for_proof,
        policy_schema_version=policy.schema_version,
        policy_hash=policy.policy_hash,
        max_candidates=max_settings_candidates,
    )
    settings_queue = _apply_validated_settings_captures(
        settings_queue,
        _read_csv(active / "crypto_wizards_pair_page_capture_settings.csv"),
    )

    lineage = _build_lineage_audit(root, manifest, candidates, settings_queue)
    required_lineage = lineage[lineage["required_for_ranking"] == True]  # noqa: E712
    lineage_ready = bool(not required_lineage.empty and required_lineage["status"].eq("pass").all())
    ready = bool(provisional_ready and lineage_ready)
    if not settings_queue.empty:
        settings_queue["control_plane_ready"] = ready
        settings_queue["actionable"] = (
            settings_queue["passes_discovery_gate"].astype(bool)
            & settings_queue["source_fresh"].astype(bool)
            & ready
        )
        settings_queue["capture_status"] = settings_queue.apply(_capture_status, axis=1)
        settings_queue["next_step"] = settings_queue.apply(_capture_next_step, axis=1)

    blockers = _control_blockers(
        artifacts_present=manifest_path.exists() and candidates_path.exists() and sweep_summary_path.exists(),
        sweep_complete=sweep_complete,
        contract_ready=contract_ready,
        candidate_rows=candidate_rows,
        fresh_candidates=fresh_candidates,
        lineage_ready=lineage_ready,
    )
    health = _build_health_report(
        checked_at=checked_at,
        root=root,
        artifacts_present=manifest_path.exists() and candidates_path.exists() and sweep_summary_path.exists(),
        sweep_complete=sweep_complete,
        contract_ready=contract_ready,
        candidate_rows=candidate_rows,
        fresh_candidates=fresh_candidates,
        lineage_ready=lineage_ready,
        ready=ready,
        blockers=blockers,
        manifest_path=manifest_path,
        candidates_path=candidates_path,
        summary_path=sweep_summary_path,
    )

    paths = {
        "wizard_control_plane_health": active / "wizard_control_plane_health.csv",
        "wizard_sweep_settings_capture_queue": active / "wizard_sweep_settings_capture_queue.csv",
        "wizard_api_contract_report": active / "wizard_api_contract_report.csv",
        "wizard_config_lineage_audit": active / "wizard_config_lineage_audit.csv",
        "wizard_freshness_blockers": active / "wizard_freshness_blockers.csv",
        "wizard_control_plane_summary": active / "wizard_control_plane_summary.json",
        "wizard_control_plane_summary_md": active / "wizard_control_plane_summary.md",
    }
    _write_csv(health, paths["wizard_control_plane_health"])
    _write_csv(settings_queue, paths["wizard_sweep_settings_capture_queue"])
    _write_csv(contract, paths["wizard_api_contract_report"])
    _write_csv(lineage, paths["wizard_config_lineage_audit"])
    _write_csv(freshness, paths["wizard_freshness_blockers"])

    summary: dict[str, object] = {
        "schema_version": WIZARD_CONTROL_SCHEMA_VERSION,
        "checked_at": checked_at.isoformat(),
        "ready": ready,
        "status": "ready" if ready else "blocked",
        "blocker": ";".join(blockers),
        "blockers": blockers,
        "sweep_id": sweep_summary.get("sweep_id", ""),
        "sweep_complete": sweep_complete,
        "discovery_authority": sweep_summary.get("discovery_authority", "unknown"),
        "api_contract_ready": contract_ready,
        "config_lineage_ready": lineage_ready,
        "candidate_rows": candidate_rows,
        "fresh_candidate_rows": fresh_candidates,
        "discovery_gate_rows": int(settings_queue["passes_discovery_gate"].sum())
        if not settings_queue.empty
        else 0,
        "research_spend_gate_rows": int(settings_queue["passes_research_spend_gate"].sum())
        if not settings_queue.empty
        else 0,
        "actionable_settings_rows": int(settings_queue["actionable"].sum())
        if not settings_queue.empty
        else 0,
        "min_sharpe": min_sharpe,
        "min_return": min_return,
        "min_closed_trades_for_proof": min_closed_trades_for_proof,
        "discovery_policy_schema_version": policy.schema_version,
        "discovery_policy_hash": policy.policy_hash,
        "max_age_hours": max_age_hours,
        "rows": len(health),
    }
    paths["wizard_control_plane_summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["wizard_control_plane_summary_md"].write_text(
        _summary_markdown(summary, health), encoding="utf-8"
    )
    return CommandResult(paths=paths, summary=summary)


def _build_contract_report(
    *,
    root: Path,
    manifest: pd.DataFrame,
    candidates: pd.DataFrame,
    summary: dict[str, Any],
    manifest_path: Path,
    candidates_path: Path,
    summary_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(
        contract: str,
        check: str,
        passed: bool | None,
        *,
        blocking: bool,
        rows_checked: int = 0,
        failures: int = 0,
        missing: set[str] | None = None,
        detail: str = "",
        path: Path,
    ) -> None:
        rows.append(
            {
                "contract": contract,
                "check": check,
                "status": "not_applicable" if passed is None else ("pass" if passed else "fail"),
                "blocking": blocking,
                "rows_checked": rows_checked,
                "failure_count": failures,
                "missing_fields": ";".join(sorted(missing or set())),
                "detail": detail,
                "evidence_path": str(path.relative_to(root)) if path.exists() else str(path),
            }
        )

    summary_missing = SUMMARY_REQUIRED_FIELDS - set(summary)
    add(
        "sweep_summary",
        "required_fields",
        summary_path.exists() and not summary_missing,
        blocking=True,
        failures=len(summary_missing),
        missing=summary_missing,
        path=summary_path,
    )
    manifest_missing = MANIFEST_REQUIRED_COLUMNS - set(manifest.columns)
    add(
        "sweep_manifest",
        "required_columns",
        manifest_path.exists() and not manifest_missing,
        blocking=True,
        rows_checked=len(manifest),
        failures=len(manifest_missing),
        missing=manifest_missing,
        path=manifest_path,
    )
    candidate_metadata_missing = CANDIDATE_METADATA_COLUMNS - set(candidates.columns)
    add(
        "sweep_candidates",
        "metadata_columns",
        candidates_path.exists() and not candidate_metadata_missing,
        blocking=True,
        rows_checked=len(candidates),
        failures=len(candidate_metadata_missing),
        missing=candidate_metadata_missing,
        path=candidates_path,
    )
    discovery_missing = CANDIDATE_DISCOVERY_COLUMNS - set(candidates.columns)
    add(
        "sweep_candidates",
        "discovery_columns",
        None if candidates.empty else not discovery_missing,
        blocking=True,
        rows_checked=len(candidates),
        failures=0 if candidates.empty else len(discovery_missing),
        missing=set() if candidates.empty else discovery_missing,
        detail="validated only when the API returned candidates",
        path=candidates_path,
    )

    if manifest.empty or "status" not in manifest:
        add(
            "sweep_manifest",
            "completed_response_lineage",
            None,
            blocking=True,
            detail="no completed rows",
            path=manifest_path,
        )
    else:
        completed = manifest[manifest["status"].astype(str).eq("completed")]
        if completed.empty:
            add(
                "sweep_manifest",
                "completed_response_lineage",
                None,
                blocking=True,
                detail="no completed rows",
                path=manifest_path,
            )
        else:
            valid_hash = completed.get("response_hash", pd.Series("", index=completed.index)).map(_valid_hash)
            valid_config = completed.get("config_hash", pd.Series("", index=completed.index)).map(_valid_hash)
            evidence_exists = completed.get("evidence_path", pd.Series("", index=completed.index)).map(
                lambda value: bool(str(value).strip()) and (root / str(value)).is_file()
            )
            failures = int((~(valid_hash & valid_config & evidence_exists)).sum())
            add(
                "sweep_manifest",
                "completed_response_lineage",
                failures == 0,
                blocking=True,
                rows_checked=len(completed),
                failures=failures,
                detail="completed rows require response/config hashes and an immutable raw snapshot",
                path=manifest_path,
            )

    if candidates.empty or discovery_missing:
        add(
            "sweep_candidates",
            "numeric_discovery_values",
            None,
            blocking=True,
            detail="no candidate rows available",
            path=candidates_path,
        )
    else:
        sharpe = pd.to_numeric(candidates["sharpe"], errors="coerce")
        returns = pd.to_numeric(candidates["returns_total"], errors="coerce")
        symbols = candidates["symbol_1"].astype(str).str.strip().ne("") & candidates[
            "symbol_2"
        ].astype(str).str.strip().ne("")
        valid = sharpe.map(math.isfinite) & returns.map(math.isfinite) & symbols
        failures = int((~valid).sum())
        add(
            "sweep_candidates",
            "numeric_discovery_values",
            failures == 0,
            blocking=True,
            rows_checked=len(candidates),
            failures=failures,
            detail="symbol_1, symbol_2, sharpe, and returns_total must be usable",
            path=candidates_path,
        )

    return pd.DataFrame(rows, columns=CONTRACT_COLUMNS)


def _build_freshness_report(
    candidates: pd.DataFrame,
    *,
    summary: dict[str, Any],
    checked_at: datetime,
    max_age_hours: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in candidates.iterrows():
        source_value = row.get("sweep_source_timestamp", row.get("backtest_ts", ""))
        captured_value = row.get("sweep_captured_at", summary.get("started_at", ""))
        source_at = _parse_timestamp(source_value)
        captured_at = _parse_timestamp(captured_value)
        source_age = _age_hours(source_at, checked_at)
        capture_age = _age_hours(captured_at, checked_at)
        source_fresh = source_age is not None and 0 <= source_age <= max_age_hours
        capture_fresh = capture_age is not None and 0 <= capture_age <= max_age_hours
        blockers: list[str] = []
        if source_at is None:
            blockers.append("source_timestamp_missing")
        elif not source_fresh:
            blockers.append("source_data_stale")
        if captured_at is None:
            blockers.append("capture_timestamp_missing")
        elif not capture_fresh:
            blockers.append("sweep_capture_stale")
        if not _truthy(row.get("sweep_complete", False)):
            blockers.append("partial_sweep")
        rows.append(
            {
                "pair": _candidate_pair(row),
                "wizard_exchange": row.get("sweep_exchange", row.get("exchange", "")),
                "timeframe": row.get("sweep_interval", row.get("interval", "")),
                "strategy": row.get("sweep_strategy", row.get("strategy", "")),
                "source_timestamp": source_at.isoformat() if source_at else "",
                "captured_at": captured_at.isoformat() if captured_at else "",
                "source_age_hours": round(source_age, 4) if source_age is not None else "",
                "capture_age_hours": round(capture_age, 4) if capture_age is not None else "",
                "source_fresh": source_fresh,
                "capture_fresh": capture_fresh,
                "freshness_blocker": ";".join(blockers),
                "sweep_config_hash": row.get("sweep_config_hash", ""),
                "evidence_path": row.get("sweep_evidence_path", ""),
            }
        )
    return pd.DataFrame(rows, columns=FRESHNESS_COLUMNS)


def _build_settings_queue(
    candidates: pd.DataFrame,
    *,
    freshness: pd.DataFrame,
    control_plane_ready: bool,
    min_sharpe: float,
    min_return: float,
    min_closed_trades_for_proof: int,
    policy_schema_version: str,
    policy_hash: str,
    max_candidates: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for position, (_, row) in enumerate(candidates.iterrows()):
        sharpe = _as_float(row.get("sharpe"))
        returns_total = _as_float(row.get("returns_total"))
        passes = bool(
            sharpe is not None
            and returns_total is not None
            and sharpe >= min_sharpe
            and returns_total >= min_return
        )
        if not passes:
            continue
        freshness_row = freshness.iloc[position] if position < len(freshness) else pd.Series(dtype=object)
        strategy = row.get("sweep_strategy", row.get("strategy", ""))
        spread_type = row.get("spread_type", row.get("spread", ""))
        exact_mode = canonical_exact_mode(strategy=strategy, spread_type=spread_type)
        required = _required_settings(strategy, spread_type)
        captured = [field for field in required if _has_value(row.get(field))]
        missing = [field for field in required if field not in captured]
        config = WizardRunConfiguration(
            source="prescanned_candidate",
            symbol_1=row.get("symbol_1"),
            symbol_2=row.get("symbol_2"),
            wizard_exchange=row.get("sweep_exchange", row.get("exchange")),
            interval=row.get("sweep_interval", row.get("interval")),
            period=_as_int(row.get("period")),
            strategy=strategy,
            spread_type=spread_type,
            exact_mode=exact_mode,
            roll_window=_as_int(row.get("roll_window", row.get("zscore_window"))),
            entry_level=_as_float(row.get("entry_level")),
            exit_level=_as_float(row.get("exit_level")),
            copula_entry_lower=_as_float(row.get("copula_entry_lower")),
            copula_entry_upper=_as_float(row.get("copula_entry_upper")),
            copula_exit_lower=_as_float(row.get("copula_exit_lower")),
            copula_exit_upper=_as_float(row.get("copula_exit_upper")),
            exit_n_periods=_as_int(row.get("exit_n_periods")),
            stop_loss_rate=_as_float(row.get("stop_loss_rate")),
            commission_rate=_as_float(row.get("commission_rate")),
            slippage_rate=_as_float(row.get("slippage_rate")),
            x_weighting=_as_float(row.get("x_weighting")),
            y_weighting=_as_float(row.get("y_weighting")),
            capital_weighting=_as_text(row.get("capital_weighting")),
            metric_mode=_as_text(row.get("metric_mode")),
            input_data_hash=_as_text(row.get("sweep_response_hash")),
        )
        source_fresh = bool(freshness_row.get("source_fresh", False))
        actionable = bool(control_plane_ready and source_fresh)
        closed_trades = _as_float(row.get("closed", row.get("backtest_closed", "")))
        passes_research_spend = bool(
            closed_trades is not None and closed_trades >= min_closed_trades_for_proof
        )
        rows.append(
            {
                "priority_rank": 0,
                "pair": _candidate_pair(row),
                "asset_x": row.get("symbol_1", ""),
                "asset_y": row.get("symbol_2", ""),
                "wizard_exchange": row.get("sweep_exchange", row.get("exchange", "")),
                "timeframe": row.get("sweep_interval", row.get("interval", "")),
                "period": row.get("period", ""),
                "strategy": strategy,
                "spread_type": spread_type,
                "exact_mode": exact_mode,
                "sharpe": sharpe,
                "returns_total": returns_total,
                "closed_trades": closed_trades if closed_trades is not None else "",
                "passes_discovery_gate": passes,
                "discovery_policy_schema_version": policy_schema_version,
                "discovery_policy_hash": policy_hash,
                "min_closed_trades_for_proof": min_closed_trades_for_proof,
                "passes_research_spend_gate": passes_research_spend,
                "source_fresh": source_fresh,
                "control_plane_ready": control_plane_ready,
                "actionable": actionable,
                "required_settings": ";".join(required),
                "captured_settings": ";".join(captured),
                "missing_settings": ";".join(missing),
                "capture_status": "",
                "discovery_config_hash": row.get("sweep_config_hash", ""),
                "candidate_config_hash": config.config_hash,
                "settings_config_hash": "",
                "settings_capture_evidence_path": "",
                "source_timestamp": freshness_row.get("source_timestamp", ""),
                "captured_at": freshness_row.get("captured_at", ""),
                "evidence_path": row.get("sweep_evidence_path", ""),
                "next_step": "",
            }
        )
    frame = pd.DataFrame(rows, columns=SETTINGS_QUEUE_COLUMNS)
    if frame.empty:
        return frame
    frame = frame.sort_values(["sharpe", "returns_total"], ascending=[False, False])
    frame = frame.drop_duplicates(
        subset=["pair", "wizard_exchange", "timeframe", "exact_mode"], keep="first"
    ).head(max_candidates)
    frame = frame.reset_index(drop=True)
    frame["priority_rank"] = frame.index + 1
    frame["capture_status"] = frame.apply(_capture_status, axis=1)
    frame["next_step"] = frame.apply(_capture_next_step, axis=1)
    return frame[SETTINGS_QUEUE_COLUMNS]


def _apply_validated_settings_captures(
    queue: pd.DataFrame, captures: pd.DataFrame
) -> pd.DataFrame:
    if queue.empty or captures.empty or "candidate_config_hash" not in captures.columns:
        return queue
    valid = captures.copy()
    if "capture_confirmed" in valid.columns:
        valid = valid[valid["capture_confirmed"].map(_truthy)]
    if "backtest_settings_complete" in valid.columns:
        valid = valid[valid["backtest_settings_complete"].map(_truthy)]
    valid = valid[valid["candidate_config_hash"].map(_as_text).ne("")]
    if valid.empty:
        return queue
    if "capture_timestamp_utc" in valid.columns:
        valid = valid.assign(
            _capture_sort=pd.to_datetime(valid["capture_timestamp_utc"], utc=True, errors="coerce")
        ).sort_values("_capture_sort")
    capture_index = {
        _as_text(row.get("candidate_config_hash")): row
        for _, row in valid.iterrows()
    }
    updated = queue.copy()
    for index, row in updated.iterrows():
        capture = capture_index.get(_as_text(row.get("candidate_config_hash")))
        if capture is None:
            continue
        updated.at[index, "captured_settings"] = row.get("required_settings", "")
        updated.at[index, "missing_settings"] = ""
        updated.at[index, "capture_status"] = "SETTINGS_COMPLETE"
        updated.at[index, "settings_config_hash"] = capture.get("settings_config_hash", "")
        updated.at[index, "settings_capture_evidence_path"] = capture.get(
            "capture_evidence_path", capture.get("evidence_path", "")
        )
    return updated


def _build_lineage_audit(
    root: Path,
    manifest: pd.DataFrame,
    candidates: pd.DataFrame,
    settings_queue: pd.DataFrame,
) -> pd.DataFrame:
    artifacts = [
        ("wizard_sweep_manifest", root / "reports" / "active" / "wizard_sweep_manifest.csv", True, manifest, ("config_hash",)),
        ("wizard_sweep_candidates", root / "reports" / "active" / "wizard_sweep_candidates.csv", True, candidates, ("sweep_config_hash",)),
        ("wizard_sweep_settings_capture_queue", root / "reports" / "active" / "wizard_sweep_settings_capture_queue.csv", True, settings_queue, ("candidate_config_hash",)),
        ("wizard_research_journal", root / "reports" / "active" / "wizard_research_journal.csv", False, None, ("config_hash", "candidate_config_hash", "sweep_config_hash")),
        ("hyperliquid_wizard_hypothesis_queue", root / "reports" / "active" / "hyperliquid_wizard_hypothesis_queue.csv", False, None, ("config_hash", "candidate_config_hash", "sweep_config_hash")),
        ("trade_training_dataset", root / "data" / "ml" / "trade_training_dataset.csv", False, None, ("config_hash", "candidate_config_hash", "sweep_config_hash")),
        ("rl_training_report", root / "reports" / "rl" / "rl_training_report.csv", False, None, ("config_hash", "candidate_config_hash", "sweep_config_hash")),
    ]
    rows: list[dict[str, object]] = []
    for artifact, path, required, supplied, columns in artifacts:
        frame = supplied if supplied is not None else _read_csv(path)
        available = path.exists() or supplied is not None
        column = next((candidate for candidate in columns if candidate in frame.columns), "")
        if frame.empty:
            coverage = 100.0 if required and available else 0.0
        elif column:
            coverage = round(100.0 * frame[column].map(_valid_hash).mean(), 2)
        else:
            coverage = 0.0
        if required:
            passed = available and (frame.empty or bool(column)) and coverage == 100.0
            status = "pass" if passed else "fail"
            blocker = "" if passed else "required_config_identity_missing"
        elif not path.exists():
            status = "not_present"
            blocker = ""
        elif frame.empty:
            status = "no_rows"
            blocker = ""
        elif coverage == 100.0:
            status = "pass"
            blocker = ""
        else:
            status = "warning"
            blocker = "downstream_config_identity_incomplete"
        rows.append(
            {
                "artifact": artifact,
                "path": str(path.relative_to(root)),
                "required_for_ranking": required,
                "exists": available,
                "rows": len(frame),
                "config_hash_column": column,
                "config_hash_coverage_pct": coverage,
                "status": status,
                "blocker": blocker,
            }
        )
    return pd.DataFrame(rows, columns=LINEAGE_COLUMNS)


def _build_health_report(
    *,
    checked_at: datetime,
    root: Path,
    artifacts_present: bool,
    sweep_complete: bool,
    contract_ready: bool,
    candidate_rows: int,
    fresh_candidates: int,
    lineage_ready: bool,
    ready: bool,
    blockers: list[str],
    manifest_path: Path,
    candidates_path: Path,
    summary_path: Path,
) -> pd.DataFrame:
    timestamp = checked_at.isoformat()
    rows = [
        ("sweep_artifacts", artifacts_present, "wizard_sweep_artifacts_missing", "manifest, candidates, and summary must exist", summary_path),
        ("complete_sweep", sweep_complete, "wizard_sweep_not_complete", "only complete_discovery has ranking authority", manifest_path),
        ("api_contract", contract_ready, "wizard_api_contract_failed", "required API fields, types, and raw lineage must validate", candidates_path),
        ("candidate_freshness", fresh_candidates > 0, "no_fresh_wizard_candidates", f"fresh={fresh_candidates};total={candidate_rows}", candidates_path),
        ("config_lineage", lineage_ready, "wizard_config_lineage_incomplete", "required ranking artifacts must carry 64-character configuration hashes", manifest_path),
        ("ranking_authority", ready, "wizard_ranking_blocked", ";".join(blockers) or "complete, fresh, contract-valid discovery", summary_path),
    ]
    return pd.DataFrame(
        [
            {
                "schema_version": WIZARD_CONTROL_SCHEMA_VERSION,
                "check": name,
                "status": "pass" if passed else "block",
                "blocking": True,
                "blocker": "" if passed else blocker,
                "detail": detail,
                "evidence_path": str(path.relative_to(root)) if path.exists() else str(path),
                "checked_at": timestamp,
            }
            for name, passed, blocker, detail, path in rows
        ],
        columns=HEALTH_COLUMNS,
    )


def _sweep_is_complete(manifest: pd.DataFrame, summary: dict[str, Any]) -> bool:
    if manifest.empty:
        return False
    planned = _as_int(summary.get("planned_cells"))
    completed = _as_int(summary.get("completed_cells"))
    return bool(
        _truthy(summary.get("sweep_complete"))
        and str(summary.get("discovery_authority", "")) == "complete_discovery"
        and planned is not None
        and completed == planned
        and planned == len(manifest)
        and manifest.get("status", pd.Series("", index=manifest.index)).astype(str).eq("completed").all()
        and manifest.get("sweep_complete", pd.Series(False, index=manifest.index)).map(_truthy).all()
        and manifest.get("discovery_authority", pd.Series("", index=manifest.index)).astype(str).eq("complete_discovery").all()
    )


def _control_blockers(
    *,
    artifacts_present: bool,
    sweep_complete: bool,
    contract_ready: bool,
    candidate_rows: int,
    fresh_candidates: int,
    lineage_ready: bool,
) -> list[str]:
    blockers: list[str] = []
    if not artifacts_present:
        blockers.append("wizard_sweep_artifacts_missing")
    if not sweep_complete:
        blockers.append("wizard_sweep_not_complete")
    if not contract_ready:
        blockers.append("wizard_api_contract_failed")
    if candidate_rows == 0:
        blockers.append("wizard_candidates_missing")
    elif fresh_candidates == 0:
        blockers.append("no_fresh_wizard_candidates")
    if not lineage_ready:
        blockers.append("wizard_config_lineage_incomplete")
    return blockers


def _required_settings(strategy: object, spread_type: object) -> list[str]:
    base = [
        "period",
        "commission_rate",
        "slippage_rate",
        "x_weighting",
        "y_weighting",
        "capital_weighting",
        "metric_mode",
        "stop_loss_rate",
        "exit_n_periods",
    ]
    mode = canonical_exact_mode(strategy=strategy, spread_type=spread_type)
    if mode == "copula":
        return [
            *base,
            "copula_entry_lower",
            "copula_entry_upper",
            "copula_exit_lower",
            "copula_exit_upper",
        ]
    if "zscore" in mode:
        return [*base, "spread_type", "roll_window", "entry_level", "exit_level"]
    return [*base, "spread_type", "entry_level", "exit_level"]


def _capture_status(row: pd.Series) -> str:
    if not bool(row.get("control_plane_ready", False)):
        return "BLOCKED_CONTROL_PLANE"
    if not bool(row.get("source_fresh", False)):
        return "BLOCKED_STALE_SOURCE"
    if str(row.get("missing_settings", "")):
        return "CAPTURE_REQUIRED"
    return "SETTINGS_COMPLETE"


def _capture_next_step(row: pd.Series) -> str:
    status = _capture_status(row)
    if status == "BLOCKED_CONTROL_PLANE":
        return "complete a fresh full Wizard sweep and rerun wizard-control-plane"
    if status == "BLOCKED_STALE_SOURCE":
        return "refresh this Wizard candidate before opening pair settings"
    if status == "CAPTURE_REQUIRED":
        return f"capture exact Wizard pair-page settings: {row.get('missing_settings', '')}"
    return "replay the exact captured configuration locally with Hyperliquid costs"


def _summary_markdown(summary: dict[str, object], health: pd.DataFrame) -> str:
    lines = [
        "# Wizard Control Plane",
        "",
        f"- Status: `{summary['status']}`",
        f"- Ranking authority: `{str(summary['ready']).lower()}`",
        f"- Sweep complete: `{str(summary['sweep_complete']).lower()}`",
        f"- API contract ready: `{str(summary['api_contract_ready']).lower()}`",
        f"- Configuration lineage ready: `{str(summary['config_lineage_ready']).lower()}`",
        f"- Candidates: {summary['candidate_rows']} total, {summary['fresh_candidate_rows']} fresh",
        f"- Discovery gate: Sharpe >= {summary['min_sharpe']}, returns_total >= {summary['min_return']}",
        f"- Blocker: `{summary['blocker'] or 'none'}`",
        "",
        health[["check", "status", "blocker", "detail"]].to_markdown(index=False),
        "",
        "A complete Wizard sweep can nominate research candidates. It cannot accept a trade; exact settings and local Hyperliquid replay remain required.",
        "",
    ]
    return "\n".join(lines)


def _candidate_pair(row: pd.Series) -> str:
    pair = _as_text(row.get("pair"))
    if pair:
        return pair
    left = _as_text(row.get("symbol_1"))
    right = _as_text(row.get("symbol_2"))
    return f"{left} / {right}" if left and right else left or right or ""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_timestamp(value: object) -> datetime | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
        if math.isfinite(numeric):
            if numeric > 10_000_000_000:
                numeric /= 1000.0
            return datetime.fromtimestamp(numeric, tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        pass
    parsed = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _age_hours(observed_at: datetime | None, checked_at: datetime) -> float | None:
    if observed_at is None:
        return None
    return (checked_at - observed_at).total_seconds() / 3600.0


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _valid_hash(value: object) -> bool:
    text = _as_text(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _has_value(value: object) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return bool(str(value).strip())


def _as_text(value: object) -> str:
    return str(value).strip() if _has_value(value) else ""


def _as_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_int(value: object) -> int | None:
    parsed = _as_float(value)
    return int(parsed) if parsed is not None else None
