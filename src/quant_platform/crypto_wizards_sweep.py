from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.api_extraction import CryptoWizardsFetchError
from quant_platform.crypto_wizards_catalog import BASE_URL, ENDPOINTS
from quant_platform.crypto_wizards_history import (
    fetch_credits_used,
    fetch_prescanned_payload,
    prescanned_pairs_from_payload,
)
from quant_platform.orchestration.corrective_external_effects import (
    current_external_effect_issuer,
    read_authorized_credential,
    reserved_external_effect_session,
    run_authorized_credit_call,
)
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_csv,
    atomic_write_text,
    write_immutable_bytes,
)
from quant_platform.orchestration.effect_authority import EffectAuthorityError
from quant_platform.wizard_credit_ledger import (
    DISCOVERY_LANE,
    reconcile_wizard_credit_lane,
    reserve_wizard_credit_lane,
)
from quant_platform.wizard_run_config import (
    WizardRunConfiguration,
    api_wizard_discovery_interval,
    api_wizard_exchange,
    api_wizard_strategy,
    canonical_wizard_strategy,
    normalized_key,
)

WIZARD_SWEEP_SCHEMA_VERSION = "wizard_discovery_sweep.v1"
WIZARD_CRYPTO_EXCHANGES = ("Binance", "BinanceUs", "ByBit", "Coinbase", "Dydx")
WIZARD_DISCOVERY_INTERVALS = ("Daily", "Hourly")
WIZARD_DISCOVERY_STRATEGIES = ("Spread", "ZScoreRoll", "Copula")
WIZARD_DISCOVERY_PRIORITIES = ("Sharpe",)
PRESCANNED_CREDIT_COST = next(
    endpoint.credits for endpoint in ENDPOINTS if endpoint.name == "prescanned_get"
)


@dataclass(frozen=True)
class WizardSweepCell:
    sweep_id: str
    request_id: str
    priority: str
    strategy: str
    exchange: str
    interval: str
    endpoint: str
    credit_cost: int
    config_hash: str

    def params(self) -> dict[str, str]:
        return {
            "priority": self.priority,
            "strategy": self.strategy,
            "exchange": self.exchange,
            "interval": self.interval,
        }


@dataclass(frozen=True)
class WizardCreditUsage:
    used: int | None
    limit: int
    remaining: int | None
    known: bool
    source_fields: str


@dataclass(frozen=True)
class WizardSweepResult:
    paths: dict[str, Path]
    summary: dict[str, object]


def restore_complete_wizard_sweep_from_raw(
    *,
    root: str | Path,
    sweep_id: str | None = None,
) -> WizardSweepResult:
    """Restore the canonical active sweep from an immutable complete raw snapshot."""

    root_path = Path(root)
    raw_base = root_path / "data" / "raw" / "crypto_wizards" / "prescanned"
    groups: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(raw_base.glob("**/*.json")):
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(envelope, dict):
            continue
        metadata = envelope.get("capture_metadata")
        request = envelope.get("request")
        response = envelope.get("response")
        if (
            not isinstance(metadata, dict)
            or not isinstance(request, dict)
            or not isinstance(response, list)
        ):
            continue
        candidate_sweep_id = str(metadata.get("sweep_id", "")).strip()
        if not candidate_sweep_id:
            continue
        groups.setdefault(candidate_sweep_id, []).append((path, envelope))

    expected_request_ids = {
        cell.request_id for cell in build_wizard_sweep_cells(sweep_id="recovery")
    }
    requested_ids = [sweep_id] if sweep_id else sorted(groups, reverse=True)
    selected_id = ""
    selected_files: list[tuple[Path, dict[str, Any]]] = []
    for candidate_id in requested_ids:
        files = groups.get(str(candidate_id), [])
        actual_request_ids = {
            str(envelope["capture_metadata"].get("request_id", "")).strip() for _, envelope in files
        }
        if actual_request_ids == expected_request_ids and len(files) == len(expected_request_ids):
            selected_id = str(candidate_id)
            selected_files = files
            break
    if not selected_files:
        target = sweep_id or "latest"
        raise ValueError(f"no complete 30-cell raw Wizard sweep found for {target}")

    manifest_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    started_values: list[str] = []
    for raw_path, envelope in selected_files:
        metadata = envelope["capture_metadata"]
        request = envelope["request"]
        response = envelope["response"]
        request_id = str(metadata.get("request_id", "")).strip()
        captured_at = str(metadata.get("captured_at", "")).strip()
        started_values.append(captured_at)
        response_hash = str(metadata.get("response_hash", "")).strip() or _json_hash(response)
        evidence_path = str(raw_path.relative_to(root_path))
        pairs = prescanned_pairs_from_payload(response)
        cell = next(
            cell
            for cell in build_wizard_sweep_cells(sweep_id=selected_id)
            if cell.request_id == request_id
        )
        manifest_rows.append(
            {
                "schema_version": WIZARD_SWEEP_SCHEMA_VERSION,
                "sweep_id": selected_id,
                "request_id": request_id,
                "priority": request.get("priority", cell.priority),
                "strategy": request.get("strategy", cell.strategy),
                "exchange": request.get("exchange", cell.exchange),
                "interval": request.get("interval", cell.interval),
                "endpoint": metadata.get("endpoint", cell.endpoint),
                "config_hash": metadata.get("config_hash", cell.config_hash),
                "credit_cost": metadata.get("credit_cost", cell.credit_cost),
                "status": "completed",
                "row_count": len(pairs),
                "requested_at": captured_at,
                "completed_at": captured_at,
                "response_hash": response_hash,
                "evidence_path": evidence_path,
                "error": "",
            }
        )
        for rank, pair in enumerate(pairs, start=1):
            candidate_rows.append(
                {
                    **pair,
                    "sweep_id": selected_id,
                    "request_id": request_id,
                    "sweep_config_hash": metadata.get("config_hash", cell.config_hash),
                    "sweep_rank": rank,
                    "sweep_priority": request.get("priority", cell.priority),
                    "sweep_strategy": request.get("strategy", cell.strategy),
                    "sweep_exchange": request.get("exchange", cell.exchange),
                    "sweep_interval": request.get("interval", cell.interval),
                    "sweep_captured_at": captured_at,
                    "sweep_source_timestamp": pair.get("backtest_ts")
                    or pair.get("updated_at")
                    or "",
                    "point_in_time_status": "historical_snapshot_discovery",
                    "sweep_response_hash": response_hash,
                    "sweep_evidence_path": evidence_path,
                    "sweep_complete": True,
                    "discovery_authority": "complete_discovery",
                    "sweep_blocker": "",
                }
            )

    total_cells = len(manifest_rows)
    completed_credits = int(sum(int(row["credit_cost"]) for row in manifest_rows))
    for row in manifest_rows:
        row.update(
            {
                "planned_cells": total_cells,
                "completed_cells": total_cells,
                "completion_percentage": 100.0,
                "planned_credits": completed_credits,
                "completed_credits": completed_credits,
                "attempted_credits": completed_credits,
                "credits_used_before": None,
                "credit_limit": None,
                "credits_remaining_before": None,
                "reserved_credits": None,
                "sweep_complete": True,
                "discovery_authority": "complete_discovery",
                "sweep_blocker": "",
            }
        )
    summary: dict[str, object] = {
        "schema_version": WIZARD_SWEEP_SCHEMA_VERSION,
        "sweep_id": selected_id,
        "started_at": min(value for value in started_values if value)
        if any(started_values)
        else "",
        "execute": True,
        "restored_from_raw": True,
        "planned_cells": total_cells,
        "completed_cells": total_cells,
        "failed_cells": 0,
        "completion_percentage": 100.0,
        "planned_credits": completed_credits,
        "completed_credits": completed_credits,
        "attempted_credits": completed_credits,
        "credits_used_before": None,
        "credit_limit": None,
        "credits_remaining_before": None,
        "reserved_credits": None,
        "credit_usage_known": False,
        "candidate_rows": len(candidate_rows),
        "active_candidate_rows": len(candidate_rows),
        "active_snapshot_preserved": False,
        "sweep_complete": True,
        "discovery_authority": "complete_discovery",
        "blocker": "",
    }
    reports_dir = root_path / "reports" / "active"
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": reports_dir / "wizard_sweep_manifest.csv",
        "candidates": reports_dir / "wizard_sweep_candidates.csv",
        "summary": reports_dir / "wizard_sweep_summary.json",
        "summary_md": reports_dir / "wizard_sweep_summary.md",
        "raw_snapshot_dir": raw_base,
    }
    atomic_write_csv(pd.DataFrame(manifest_rows, columns=_manifest_columns()), paths["manifest"], index=False)
    atomic_write_csv(_candidate_frame(candidate_rows), paths["candidates"], index=False)
    atomic_write_text(paths["summary"], json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    atomic_write_text(paths["summary_md"], _summary_markdown(summary), encoding="utf-8")
    return WizardSweepResult(paths=paths, summary=summary)


PrescannedFetcher = Callable[..., Any]
CreditsFetcher = Callable[..., Any]


def build_wizard_sweep_cells(
    *,
    sweep_id: str,
    exchanges: Iterable[str] = WIZARD_CRYPTO_EXCHANGES,
    intervals: Iterable[str] = WIZARD_DISCOVERY_INTERVALS,
    strategies: Iterable[str] = WIZARD_DISCOVERY_STRATEGIES,
    priorities: Iterable[str] = WIZARD_DISCOVERY_PRIORITIES,
) -> list[WizardSweepCell]:
    exchange_values = _validated_values(
        exchanges, WIZARD_CRYPTO_EXCHANGES, "exchange", api_wizard_exchange
    )
    interval_values = _validated_values(
        intervals,
        WIZARD_DISCOVERY_INTERVALS,
        "interval",
        api_wizard_discovery_interval,
    )
    strategy_values = _validated_values(
        strategies, WIZARD_DISCOVERY_STRATEGIES, "strategy", api_wizard_strategy
    )
    priority_values = _dedupe(str(value).strip() for value in priorities if str(value).strip())
    if not priority_values:
        raise ValueError("at least one Crypto Wizards priority is required")

    cells: list[WizardSweepCell] = []
    for priority, strategy, exchange, interval in product(
        priority_values,
        strategy_values,
        exchange_values,
        interval_values,
    ):
        configuration = WizardRunConfiguration(
            source="prescanned_api",
            wizard_exchange=exchange,
            interval=interval,
            strategy=strategy,
            priority=priority,
            exact_mode=f"{canonical_wizard_strategy(strategy)}_family_discovery",
        )
        request_identity = {
            "priority": priority,
            "strategy": strategy,
            "exchange": exchange,
            "interval": interval,
        }
        request_id = sha256(
            json.dumps(request_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        cells.append(
            WizardSweepCell(
                sweep_id=sweep_id,
                request_id=request_id,
                priority=priority,
                strategy=strategy,
                exchange=exchange,
                interval=interval,
                endpoint="/v1beta/prescanned",
                credit_cost=PRESCANNED_CREDIT_COST,
                config_hash=configuration.config_hash,
            )
        )
    return cells


def parse_wizard_credit_usage(payload: Any, *, configured_limit: int = 1000) -> WizardCreditUsage:
    if configured_limit <= 0:
        raise ValueError("configured_limit must be positive")
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        used = float(payload)
        if used >= 0 and used.is_integer():
            resolved_used = int(used)
            return WizardCreditUsage(
                used=resolved_used,
                limit=int(configured_limit),
                remaining=max(int(configured_limit) - resolved_used, 0),
                known=True,
                source_fields="scalar_response",
            )
    flattened = _flatten_numeric_fields(payload)
    used = _first_numeric(
        flattened,
        "credits_used",
        "creditsused",
        "used_credits",
        "usedcredits",
        "daily_credits_used",
        "dailycreditsused",
        "usage",
    )
    limit = _first_numeric(
        flattened,
        "credit_limit",
        "creditlimit",
        "daily_limit",
        "dailylimit",
        "daily_allowance",
        "dailyallowance",
        "credits_total",
        "creditstotal",
        "limit",
    )
    remaining = _first_numeric(
        flattened,
        "credits_remaining",
        "creditsremaining",
        "remaining_credits",
        "remainingcredits",
        "remaining",
    )
    resolved_limit = int(limit) if limit is not None and limit > 0 else int(configured_limit)
    resolved_used = int(used) if used is not None and used >= 0 else None
    resolved_remaining = int(remaining) if remaining is not None and remaining >= 0 else None
    if resolved_used is None and resolved_remaining is not None:
        resolved_used = max(resolved_limit - resolved_remaining, 0)
    if resolved_remaining is None and resolved_used is not None:
        resolved_remaining = max(resolved_limit - resolved_used, 0)
    used_source = next(
        (
            key
            for key in flattened
            if normalized_key(key.rsplit(".", 1)[-1])
            in {"creditsused", "usedcredits", "dailycreditsused", "usage"}
        ),
        "",
    )
    remaining_source = next(
        (
            key
            for key in flattened
            if normalized_key(key.rsplit(".", 1)[-1])
            in {"creditsremaining", "remainingcredits", "remaining"}
        ),
        "",
    )
    return WizardCreditUsage(
        used=resolved_used,
        limit=resolved_limit,
        remaining=resolved_remaining,
        known=resolved_used is not None and resolved_remaining is not None,
        source_fields=";".join(value for value in (used_source, remaining_source) if value),
    )


def run_wizard_discovery_sweep(
    *,
    root: str | Path,
    execute: bool = False,
    api_key: str | None = None,
    base_url: str = BASE_URL,
    exchanges: Iterable[str] = WIZARD_CRYPTO_EXCHANGES,
    intervals: Iterable[str] = WIZARD_DISCOVERY_INTERVALS,
    strategies: Iterable[str] = WIZARD_DISCOVERY_STRATEGIES,
    priorities: Iterable[str] = WIZARD_DISCOVERY_PRIORITIES,
    daily_credit_limit: int = 1000,
    reserved_credits: int = 100,
    timeout: float = 30.0,
    now: datetime | None = None,
    credits_fetcher: CreditsFetcher | None = None,
    prescanned_fetcher: PrescannedFetcher | None = None,
    credit_reserver: Callable[..., Any] | None = None,
    credit_reconciler: Callable[..., Any] | None = None,
    credit_lane: str = DISCOVERY_LANE,
    publish_active: bool = True,
) -> WizardSweepResult:
    if daily_credit_limit <= 0:
        raise ValueError("daily_credit_limit must be positive")
    if reserved_credits < 0:
        raise ValueError("reserved_credits must be non-negative")
    started_at = _as_utc(now or datetime.now(UTC))
    sweep_id = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    cells = build_wizard_sweep_cells(
        sweep_id=sweep_id,
        exchanges=exchanges,
        intervals=intervals,
        strategies=strategies,
        priorities=priorities,
    )
    planned_credits = sum(cell.credit_cost for cell in cells)
    reports_dir = Path(root) / "reports" / "active"
    raw_dir = (
        Path(root)
        / "data"
        / "raw"
        / "crypto_wizards"
        / "prescanned"
        / started_at.date().isoformat()
    )
    credit_raw_dir = (
        Path(root)
        / "data"
        / "raw"
        / "crypto_wizards"
        / "credit_usage"
        / started_at.date().isoformat()
    )
    credit_usage_before_evidence = credit_raw_dir / f"{sweep_id}_before.json"
    credit_usage_after_evidence = credit_raw_dir / f"{sweep_id}_after.json"
    reports_dir.mkdir(parents=True, exist_ok=True)

    credit_usage = WizardCreditUsage(None, daily_credit_limit, None, False, "")
    credit_reservation_status = "NOT_REQUESTED"
    credit_reservation_blocker = ""
    credit_reservation_id = ""
    credit_reservation_path = ""
    credit_reconciliation_status = "NOT_REQUESTED"
    credit_reconciliation_blocker = ""
    credit_reconciliation_id = ""
    credit_reconciliation_path = ""
    preflight_blocker = "preflight_only"
    if execute:
        if not api_key:
            preflight_blocker = "missing_crypto_wizards_api_key"
        else:
            try:
                reservation = (credit_reserver or reserve_wizard_credit_lane)(
                    root=Path(root),
                    lane=credit_lane,
                    planned_credits=planned_credits,
                    now=started_at,
                    daily_credit_limit=daily_credit_limit,
                    protected_reserve=reserved_credits,
                    max_external_requests=len(cells) + 2,
                )
                credit_reservation_status = str(reservation.summary.get("status", "BLOCKED"))
                credit_reservation_blocker = str(reservation.summary.get("blocker", ""))
                credit_reservation_id = str(reservation.summary.get("reservation_id", ""))
                credit_reservation_path = _relative_path(
                    reservation.paths.get("reservation"), root=Path(root)
                )
                remaining_lane_credits = int(
                    reservation.summary.get("lane_remaining_reserved_credits", 0) or 0
                )
                external_spend_authorized = (
                    reservation.summary.get("external_spend_authorized") is True
                )
                if credit_reservation_status not in {"PASS", "REUSED"}:
                    preflight_blocker = "shared_credit_reservation_blocked:" + (
                        credit_reservation_blocker or "unknown_reservation_failure"
                    )
                elif not external_spend_authorized:
                    preflight_blocker = (
                        "reused_credit_reservation_does_not_authorize_external_replay"
                    )
                elif planned_credits > remaining_lane_credits:
                    preflight_blocker = "insufficient_lane_reservation_remaining_for_complete_sweep"
                else:
                    credit_payload = _fetch_credit_usage(
                        fetcher=credits_fetcher,
                        api_key=api_key,
                        base_url=base_url,
                        timeout=timeout,
                        result_recorder=lambda payload: _write_credit_usage_snapshot(
                            credit_usage_before_evidence,
                            sweep_id=sweep_id,
                            phase="before",
                            captured_at=datetime.now(UTC).isoformat(),
                            payload=payload,
                        ),
                    )
                    credit_usage = parse_wizard_credit_usage(
                        credit_payload,
                        configured_limit=daily_credit_limit,
                    )
                    if not credit_usage.known:
                        preflight_blocker = "credit_usage_unknown"
                    elif planned_credits > max((credit_usage.remaining or 0) - reserved_credits, 0):
                        preflight_blocker = "insufficient_credits_for_complete_sweep"
                    else:
                        preflight_blocker = ""
            except (CryptoWizardsFetchError, OSError, ValueError, TypeError) as exc:
                preflight_blocker = f"credit_preflight_failed:{_safe_error(exc)}"

    manifest_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    completed_credits = 0
    attempted_credits = 0
    for cell in cells:
        status = "planned" if not execute else f"blocked_{preflight_blocker.split(':', 1)[0]}"
        error = "" if not execute else preflight_blocker
        row_count = 0
        raw_response_path = ""
        response_hash = ""
        requested_at = ""
        completed_at = ""
        if execute and not preflight_blocker:
            requested_at = datetime.now(UTC).isoformat()
            attempted_credits += cell.credit_cost
            try:
                raw_path = raw_dir / f"{sweep_id}_{cell.request_id}.json"
                payload = _fetch_prescanned(
                    fetcher=prescanned_fetcher,
                    api_key=api_key,
                    base_url=base_url,
                    priority=cell.priority,
                    strategy=cell.strategy,
                    exchange=cell.exchange,
                    interval=cell.interval,
                    timeout=timeout,
                    result_recorder=lambda value,
                    evidence_path=raw_path,
                    sweep_cell=cell,
                    captured_at=requested_at: _write_raw_snapshot(
                        evidence_path,
                        cell=sweep_cell,
                        captured_at=captured_at,
                        payload=value,
                        response_hash=_json_hash(value),
                    ),
                )
                response_hash = _json_hash(payload)
                raw_response_path = str(raw_path.relative_to(Path(root)))
                pairs = prescanned_pairs_from_payload(payload)
                row_count = len(pairs)
                completed_at = datetime.now(UTC).isoformat()
                completed_credits += cell.credit_cost
                status = "completed"
                error = ""
                for rank, pair in enumerate(pairs, start=1):
                    candidate_rows.append(
                        {
                            **pair,
                            "sweep_id": sweep_id,
                            "request_id": cell.request_id,
                            "sweep_config_hash": cell.config_hash,
                            "sweep_rank": rank,
                            "sweep_priority": cell.priority,
                            "sweep_strategy": cell.strategy,
                            "sweep_exchange": cell.exchange,
                            "sweep_interval": cell.interval,
                            "sweep_captured_at": completed_at,
                            "sweep_source_timestamp": pair.get("backtest_ts")
                            or pair.get("updated_at")
                            or "",
                            "point_in_time_status": "historical_snapshot_discovery",
                            "sweep_response_hash": response_hash,
                            "sweep_evidence_path": raw_response_path,
                        }
                    )
            except (CryptoWizardsFetchError, ValueError, TypeError) as exc:
                completed_at = datetime.now(UTC).isoformat()
                status = "failed"
                error = _safe_error(exc)
        manifest_rows.append(
            {
                "schema_version": WIZARD_SWEEP_SCHEMA_VERSION,
                "sweep_id": sweep_id,
                "request_id": cell.request_id,
                "priority": cell.priority,
                "strategy": cell.strategy,
                "exchange": cell.exchange,
                "interval": cell.interval,
                "endpoint": cell.endpoint,
                "config_hash": cell.config_hash,
                "credit_cost": cell.credit_cost,
                "status": status,
                "row_count": row_count,
                "requested_at": requested_at,
                "completed_at": completed_at,
                "response_hash": response_hash,
                "evidence_path": raw_response_path,
                "error": error,
            }
        )

    credits_used_after: int | None = credit_usage.used
    if execute and attempted_credits > 0:
        try:
            after_payload = _fetch_credit_usage(
                fetcher=credits_fetcher,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                result_recorder=lambda payload: _write_credit_usage_snapshot(
                    credit_usage_after_evidence,
                    sweep_id=sweep_id,
                    phase="after",
                    captured_at=datetime.now(UTC).isoformat(),
                    payload=payload,
                ),
            )
            credits_used_after = parse_wizard_credit_usage(
                after_payload,
                configured_limit=daily_credit_limit,
            ).used
        except (CryptoWizardsFetchError, OSError, ValueError, TypeError) as exc:
            credits_used_after = None
            credit_reconciliation_blocker = (
                f"vendor_credit_after_fetch_failed:{_safe_error(exc)}"
            )

    if execute and credit_reservation_id:
        try:
            activity_rows = [
                {
                    "lane": str(row["request_id"]),
                    "external_requests": 1,
                    "credit_cost": int(row["credit_cost"]),
                    "attempted_credits": int(row["credit_cost"]),
                    "completed_credits": (
                        int(row["credit_cost"])
                        if row["status"] == "completed"
                        else 0
                    ),
                }
                for row in manifest_rows
                if bool(row["requested_at"])
            ]
            reconciliation = (credit_reconciler or reconcile_wizard_credit_lane)(
                root=Path(root),
                lane=credit_lane,
                reservation_id=credit_reservation_id,
                reconciliation_key=sweep_id,
                attempted_credits=attempted_credits,
                completed_credits=completed_credits,
                external_requests=sum(bool(row["requested_at"]) for row in manifest_rows),
                observed_used_before=credit_usage.used,
                observed_used_after=credits_used_after,
                activity_rows=activity_rows,
                now=started_at,
            )
            credit_reconciliation_status = str(reconciliation.summary.get("status", "BLOCKED"))
            credit_reconciliation_blocker = str(reconciliation.summary.get("blocker", ""))
            credit_reconciliation_id = str(reconciliation.summary.get("reconciliation_id", ""))
            credit_reconciliation_path = _relative_path(
                reconciliation.paths.get("reconciliation"), root=Path(root)
            )
        except (OSError, TypeError, ValueError) as exc:
            credit_reconciliation_status = "BLOCKED"
            credit_reconciliation_blocker = credit_reconciliation_blocker or (
                f"credit_reconciliation_failed:{_safe_error(exc)}"
            )

    completed_cells = sum(row["status"] == "completed" for row in manifest_rows)
    failed_cells = sum(row["status"] == "failed" for row in manifest_rows)
    total_cells = len(manifest_rows)
    completion_percentage = round(100.0 * completed_cells / total_cells, 2) if total_cells else 0.0
    reconciliation_passes = credit_reconciliation_status in {
        "PASS_RECONCILED",
        "REUSED_RECONCILIATION",
    }
    sweep_complete = bool(
        execute and total_cells and completed_cells == total_cells and reconciliation_passes
    )
    if not execute:
        authority = "preflight_only"
        blocker = "execution_not_requested"
    elif preflight_blocker:
        authority = "blocked_partial_discovery"
        blocker = preflight_blocker
    elif failed_cells:
        authority = "blocked_partial_discovery"
        blocker = f"failed_requests:{failed_cells}"
    elif not reconciliation_passes:
        authority = "blocked_partial_discovery"
        blocker = "credit_reconciliation_blocked:" + (
            credit_reconciliation_blocker or credit_reconciliation_status
        )
    else:
        authority = "complete_discovery"
        blocker = ""
    for row in manifest_rows:
        row.update(
            {
                "planned_cells": total_cells,
                "completed_cells": completed_cells,
                "completion_percentage": completion_percentage,
                "planned_credits": planned_credits,
                "completed_credits": completed_credits,
                "attempted_credits": attempted_credits,
                "credits_used_before": credit_usage.used,
                "credits_used_after": credits_used_after,
                "observed_credit_delta": (
                    credits_used_after - credit_usage.used
                    if credits_used_after is not None and credit_usage.used is not None
                    else None
                ),
                "credit_limit": credit_usage.limit,
                "credits_remaining_before": credit_usage.remaining,
                "reserved_credits": reserved_credits,
                "credit_reservation_status": credit_reservation_status,
                "credit_reservation_id": credit_reservation_id,
                "credit_reconciliation_status": credit_reconciliation_status,
                "credit_reconciliation_id": credit_reconciliation_id,
                "sweep_complete": sweep_complete,
                "discovery_authority": authority,
                "sweep_blocker": blocker,
            }
        )
    for row in candidate_rows:
        row["sweep_complete"] = sweep_complete
        row["discovery_authority"] = authority
        row["sweep_blocker"] = blocker

    manifest_path = reports_dir / "wizard_sweep_manifest.csv"
    candidates_path = reports_dir / "wizard_sweep_candidates.csv"
    summary_path = reports_dir / "wizard_sweep_summary.json"
    summary_md_path = reports_dir / "wizard_sweep_summary.md"
    attempt_manifest_path = reports_dir / "wizard_sweep_latest_attempt_manifest.csv"
    attempt_candidates_path = reports_dir / "wizard_sweep_latest_attempt_candidates.csv"
    attempt_summary_path = reports_dir / "wizard_sweep_latest_attempt_summary.json"
    attempt_summary_md_path = reports_dir / "wizard_sweep_latest_attempt_summary.md"
    manifest_frame = pd.DataFrame(manifest_rows, columns=_manifest_columns())
    candidate_frame = _candidate_frame(candidate_rows)
    atomic_write_csv(manifest_frame, attempt_manifest_path, index=False)
    atomic_write_csv(candidate_frame, attempt_candidates_path, index=False)

    prior_active_candidates = _read_csv(candidates_path)
    publish_current_snapshot = bool(sweep_complete and publish_active)
    active_snapshot_preserved = bool(
        not publish_current_snapshot
        and not prior_active_candidates.empty
        and manifest_path.exists()
    )
    if publish_current_snapshot or not active_snapshot_preserved:
        atomic_write_csv(manifest_frame, manifest_path, index=False)
        atomic_write_csv(candidate_frame, candidates_path, index=False)
    summary: dict[str, object] = {
        "schema_version": WIZARD_SWEEP_SCHEMA_VERSION,
        "sweep_id": sweep_id,
        "started_at": started_at.isoformat(),
        "execute": execute,
        "planned_cells": total_cells,
        "completed_cells": completed_cells,
        "failed_cells": failed_cells,
        "completion_percentage": completion_percentage,
        "planned_credits": planned_credits,
        "completed_credits": completed_credits,
        "attempted_credits": attempted_credits,
        "credits_used_before": credit_usage.used,
        "credits_used_after": credits_used_after,
        "observed_credit_delta": (
            credits_used_after - credit_usage.used
            if credits_used_after is not None and credit_usage.used is not None
            else None
        ),
        "credit_limit": credit_usage.limit,
        "credits_remaining_before": credit_usage.remaining,
        "reserved_credits": reserved_credits,
        "credit_reservation_status": credit_reservation_status,
        "credit_reservation_blocker": credit_reservation_blocker,
        "credit_reservation_id": credit_reservation_id,
        "credit_reservation_path": credit_reservation_path,
        "credit_reconciliation_status": credit_reconciliation_status,
        "credit_reconciliation_blocker": credit_reconciliation_blocker,
        "credit_reconciliation_id": credit_reconciliation_id,
        "credit_reconciliation_path": credit_reconciliation_path,
        "credit_usage_known": credit_usage.known,
        "credit_usage_before_evidence_path": _relative_path(
            credit_usage_before_evidence if credit_usage_before_evidence.is_file() else None,
            root=Path(root),
        ),
        "credit_usage_before_evidence_sha256": _path_sha256(
            credit_usage_before_evidence
        ),
        "credit_usage_after_evidence_path": _relative_path(
            credit_usage_after_evidence if credit_usage_after_evidence.is_file() else None,
            root=Path(root),
        ),
        "credit_usage_after_evidence_sha256": _path_sha256(
            credit_usage_after_evidence
        ),
        "candidate_rows": len(candidate_rows),
        "publish_active_requested": publish_active,
        "published_active_snapshot": publish_current_snapshot,
        "active_candidate_rows": (
            len(prior_active_candidates) if active_snapshot_preserved else len(candidate_rows)
        ),
        "active_snapshot_preserved": active_snapshot_preserved,
        "sweep_complete": sweep_complete,
        "discovery_authority": authority,
        "blocker": blocker,
    }
    atomic_write_text(attempt_summary_path, json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    atomic_write_text(attempt_summary_md_path, _summary_markdown(summary), encoding="utf-8")
    if publish_current_snapshot or not active_snapshot_preserved:
        atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        atomic_write_text(summary_md_path, _summary_markdown(summary), encoding="utf-8")
    result_paths = {
        "manifest": attempt_manifest_path,
        "candidates": attempt_candidates_path,
        "summary": attempt_summary_path,
        "summary_md": attempt_summary_md_path,
        "active_manifest": manifest_path,
        "active_candidates": candidates_path,
        "active_summary": summary_path,
        "active_summary_md": summary_md_path,
        "raw_snapshot_dir": raw_dir,
    }
    if credit_usage_before_evidence.is_file():
        result_paths["credit_usage_before_evidence"] = credit_usage_before_evidence
    if credit_usage_after_evidence.is_file():
        result_paths["credit_usage_after_evidence"] = credit_usage_after_evidence
    return WizardSweepResult(
        paths=result_paths,
        summary=summary,
    )


def run_authorized_wizard_discovery_sweep(
    *,
    root: str | Path,
    api_key: str | None = None,
    credential_reader: Callable[[str], str | None] | None = None,
    now: datetime | None = None,
    exchanges: Iterable[str] = WIZARD_CRYPTO_EXCHANGES,
    intervals: Iterable[str] = WIZARD_DISCOVERY_INTERVALS,
    strategies: Iterable[str] = WIZARD_DISCOVERY_STRATEGIES,
    priorities: Iterable[str] = WIZARD_DISCOVERY_PRIORITIES,
    daily_credit_limit: int = 1000,
    reserved_credits: int = 100,
    timeout: float = 30.0,
    credits_fetcher: CreditsFetcher | None = None,
    prescanned_fetcher: PrescannedFetcher | None = None,
    credit_reconciler: Callable[..., Any] | None = None,
    credit_lane: str = DISCOVERY_LANE,
    publish_active: bool = True,
) -> WizardSweepResult:
    """Execute one sweep under the supervisor's exact ledger reservation."""

    issuer = current_external_effect_issuer()
    if issuer is None:
        raise EffectAuthorityError("wizard_discovery_effect_issuer_missing")
    root_path = Path(root)
    observed = _as_utc(now or datetime.now(UTC))
    selected_exchanges = tuple(exchanges)
    selected_intervals = tuple(intervals)
    selected_strategies = tuple(strategies)
    selected_priorities = tuple(priorities)
    cells = build_wizard_sweep_cells(
        sweep_id=observed.strftime("%Y%m%dT%H%M%S%fZ"),
        exchanges=selected_exchanges,
        intervals=selected_intervals,
        strategies=selected_strategies,
        priorities=selected_priorities,
    )
    planned_credits = sum(cell.credit_cost for cell in cells)
    request_ceiling = len(cells) + 2
    reservation = reserve_wizard_credit_lane(
        root=root_path,
        lane=credit_lane,
        planned_credits=planned_credits,
        now=observed,
        daily_credit_limit=daily_credit_limit,
        protected_reserve=reserved_credits,
        max_external_requests=request_ceiling,
    )

    def execute_with_reservation(authorized_key: str) -> WizardSweepResult:
        return run_wizard_discovery_sweep(
            root=root_path,
            execute=True,
            api_key=authorized_key,
            exchanges=selected_exchanges,
            intervals=selected_intervals,
            strategies=selected_strategies,
            priorities=selected_priorities,
            daily_credit_limit=daily_credit_limit,
            reserved_credits=reserved_credits,
            timeout=timeout,
            now=observed,
            credits_fetcher=credits_fetcher,
            prescanned_fetcher=prescanned_fetcher,
            credit_reserver=lambda **_kwargs: reservation,
            credit_reconciler=credit_reconciler,
            credit_lane=credit_lane,
            publish_active=publish_active,
        )

    if reservation.summary.get("external_spend_authorized") is not True:
        return execute_with_reservation(api_key or "")
    reservation_path = Path(reservation.paths["reservation"])
    if not _reservation_path_is_valid(root_path, reservation_path):
        raise EffectAuthorityError("wizard_discovery_reservation_path_invalid")
    reservation_id = str(reservation.summary.get("reservation_id", ""))
    binding_id = str(
        reservation.summary.get("effect_reservation_binding_id", "")
    )
    if not reservation_id or not binding_id:
        raise EffectAuthorityError("wizard_discovery_reservation_binding_missing")
    reservation_sha256 = sha256(reservation_path.read_bytes()).hexdigest()
    with reserved_external_effect_session(
        reservation_id=reservation_id,
        reservation_sha256=reservation_sha256,
        max_total_requests=request_ceiling,
        max_total_credits=planned_credits,
        reservation_binding_id=binding_id,
    ):
        authorized_key = read_authorized_credential(
            "CRYPTO_WIZARDS_API_KEY",
            reader=credential_reader
            or (lambda _key: api_key or os.getenv("CRYPTO_WIZARDS_API_KEY")),
        )
        return execute_with_reservation(authorized_key)


def _fetch_credit_usage(
    *,
    fetcher: CreditsFetcher | None,
    api_key: str,
    base_url: str,
    timeout: float,
    result_recorder: Callable[[Any], str] | None = None,
) -> Any:
    if fetcher is None:
        return fetch_credits_used(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            result_recorder=result_recorder,
        )
    target = f"{base_url.rstrip('/')}/v1beta/credits-used"
    request_payload = json.dumps(
        {"method": "GET", "target": target, "timeout_seconds": timeout},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return run_authorized_credit_call(
        target=target,
        operation="credits_used_get",
        method="GET",
        request_payload=request_payload,
        request_count=1,
        credit_cost=0,
        callback=lambda: fetcher(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        ),
        result_recorder=result_recorder,
    )


def _fetch_prescanned(
    *,
    fetcher: PrescannedFetcher | None,
    api_key: str,
    base_url: str,
    priority: str,
    strategy: str,
    exchange: str,
    interval: str,
    timeout: float,
    result_recorder: Callable[[Any], str] | None = None,
) -> Any:
    if fetcher is None:
        return fetch_prescanned_payload(
            api_key=api_key,
            base_url=base_url,
            priority=priority,
            strategy=strategy,
            exchange=exchange,
            interval=interval,
            timeout=timeout,
            result_recorder=result_recorder,
        )
    target = f"{base_url.rstrip('/')}/v1beta/prescanned"
    request_payload = json.dumps(
        {
            "exchange": exchange,
            "interval": interval,
            "method": "GET",
            "priority": priority,
            "strategy": strategy,
            "target": target,
            "timeout_seconds": timeout,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return run_authorized_credit_call(
        target=target,
        operation="prescanned_get",
        method="GET",
        request_payload=request_payload,
        request_count=1,
        credit_cost=PRESCANNED_CREDIT_COST,
        callback=lambda: fetcher(
            api_key=api_key,
            base_url=base_url,
            priority=priority,
            strategy=strategy,
            exchange=exchange,
            interval=interval,
            timeout=timeout,
        ),
        result_recorder=result_recorder,
    )


def _reservation_path_is_valid(root: Path, path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        path.resolve().relative_to(
            root.resolve() / "data" / "research" / "wizard_credit_ledger"
        )
    except (OSError, ValueError):
        return False
    return True


def _write_raw_snapshot(
    path: Path,
    *,
    cell: WizardSweepCell,
    captured_at: str,
    payload: Any,
    response_hash: str,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "capture_metadata": {
            "schema_version": WIZARD_SWEEP_SCHEMA_VERSION,
            "sweep_id": cell.sweep_id,
            "request_id": cell.request_id,
            "config_hash": cell.config_hash,
            "endpoint": cell.endpoint,
            "credit_cost": cell.credit_cost,
            "captured_at": captured_at,
            "response_hash": response_hash,
        },
        "request": cell.params(),
        "response": payload,
    }
    encoded = json.dumps(envelope, indent=2, sort_keys=True).encode("utf-8")
    write_immutable_bytes(path, encoded)
    return sha256(encoded).hexdigest()


def _write_credit_usage_snapshot(
    path: Path,
    *,
    sweep_id: str,
    phase: str,
    captured_at: str,
    payload: Any,
) -> str:
    envelope = {
        "capture_metadata": {
            "schema_version": WIZARD_SWEEP_SCHEMA_VERSION,
            "sweep_id": sweep_id,
            "capture_type": "credit_usage",
            "phase": phase,
            "captured_at": captured_at,
            "response_hash": _json_hash(payload),
        },
        "response": payload,
    }
    encoded = json.dumps(envelope, indent=2, sort_keys=True).encode("utf-8")
    write_immutable_bytes(path, encoded)
    return sha256(encoded).hexdigest()


def _path_sha256(path: Path) -> str:
    try:
        return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
    except OSError:
        return ""


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Crypto Wizards Discovery Sweep",
            "",
            f"- Sweep: `{summary['sweep_id']}`",
            f"- Authority: `{summary['discovery_authority']}`",
            f"- Complete: `{str(summary['sweep_complete']).lower()}`",
            f"- Requests: {summary['completed_cells']} / {summary['planned_cells']}",
            f"- Completion: {summary['completion_percentage']}%",
            f"- Planned credits: {summary['planned_credits']}",
            f"- Completed credits: {summary['completed_credits']}",
            f"- Attempted credits: {summary['attempted_credits']}",
            f"- Reserved credits: {summary['reserved_credits']}",
            f"- Candidate rows: {summary['candidate_rows']}",
            f"- Active candidate rows: {summary.get('active_candidate_rows', summary['candidate_rows'])}",
            f"- Prior active snapshot preserved: `{str(summary.get('active_snapshot_preserved', False)).lower()}`",
            f"- Blocker: `{summary['blocker'] or 'none'}`",
            "",
            "Only `complete_discovery` may be treated as a complete Wizard market sweep. It remains discovery evidence, not Hyperliquid acceptance authority.",
            "",
        ]
    )


def _candidate_frame(candidate_rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(candidate_rows)
    metadata_columns = _candidate_metadata_columns()
    if frame.empty:
        return pd.DataFrame(columns=metadata_columns)
    return frame[
        metadata_columns + [column for column in frame.columns if column not in metadata_columns]
    ]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()


def _manifest_columns() -> list[str]:
    return [
        "schema_version",
        "sweep_id",
        "request_id",
        "priority",
        "strategy",
        "exchange",
        "interval",
        "endpoint",
        "config_hash",
        "credit_cost",
        "status",
        "row_count",
        "requested_at",
        "completed_at",
        "response_hash",
        "evidence_path",
        "error",
        "planned_cells",
        "completed_cells",
        "completion_percentage",
        "planned_credits",
        "completed_credits",
        "attempted_credits",
        "credits_used_before",
        "credit_limit",
        "credits_remaining_before",
        "reserved_credits",
        "credit_reservation_status",
        "credit_reservation_id",
        "credit_reconciliation_status",
        "credit_reconciliation_id",
        "sweep_complete",
        "discovery_authority",
        "sweep_blocker",
    ]


def _candidate_metadata_columns() -> list[str]:
    return [
        "sweep_id",
        "request_id",
        "sweep_config_hash",
        "sweep_rank",
        "sweep_priority",
        "sweep_strategy",
        "sweep_exchange",
        "sweep_interval",
        "sweep_captured_at",
        "sweep_source_timestamp",
        "point_in_time_status",
        "sweep_response_hash",
        "sweep_evidence_path",
        "sweep_complete",
        "discovery_authority",
        "sweep_blocker",
    ]


def _validated_values(
    values: Iterable[str],
    allowed: tuple[str, ...],
    label: str,
    normalizer: Callable[[object], str],
) -> list[str]:
    normalized = _dedupe(normalizer(value) for value in values)
    if not normalized:
        raise ValueError(f"at least one Crypto Wizards {label} is required")
    allowed_set = set(allowed)
    unexpected = [value for value in normalized if value not in allowed_set]
    if unexpected:
        raise ValueError(f"unsupported Crypto Wizards {label}: {', '.join(unexpected)}")
    return normalized


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _flatten_numeric_fields(payload: Any, prefix: str = "") -> dict[str, float]:
    rows: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                rows.update(_flatten_numeric_fields(value, path))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                rows[path] = float(value)
            elif isinstance(value, str):
                try:
                    rows[path] = float(value.strip())
                except ValueError:
                    continue
    return rows


def _first_numeric(fields: dict[str, float], *names: str) -> float | None:
    wanted = {normalized_key(name) for name in names}
    for key, value in fields.items():
        if normalized_key(key.rsplit(".", 1)[-1]) in wanted:
            return value
    return None


def _json_hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _safe_error(exc: BaseException) -> str:
    return re.sub(r"\s+", " ", str(exc)).strip()[:300]


def _relative_path(path: object, *, root: Path) -> str:
    if path is None:
        return ""
    resolved = Path(path)
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
