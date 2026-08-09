from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable
import json
import re

import pandas as pd

from quant_platform.api_extraction import CryptoWizardsFetchError
from quant_platform.crypto_wizards_catalog import BASE_URL, ENDPOINTS
from quant_platform.crypto_wizards_history import (
    fetch_credits_used,
    fetch_prescanned_payload,
    prescanned_pairs_from_payload,
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
        if not isinstance(metadata, dict) or not isinstance(request, dict) or not isinstance(response, list):
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
            str(envelope["capture_metadata"].get("request_id", "")).strip()
            for _, envelope in files
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
                    "sweep_source_timestamp": pair.get("backtest_ts") or pair.get("updated_at") or "",
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
        "started_at": min(value for value in started_values if value) if any(started_values) else "",
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
    pd.DataFrame(manifest_rows, columns=_manifest_columns()).to_csv(paths["manifest"], index=False)
    _candidate_frame(candidate_rows).to_csv(paths["candidates"], index=False)
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    paths["summary_md"].write_text(_summary_markdown(summary), encoding="utf-8")
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
    publish_active: bool = True,
) -> WizardSweepResult:
    if daily_credit_limit <= 0:
        raise ValueError("daily_credit_limit must be positive")
    if reserved_credits < 0:
        raise ValueError("reserved_credits must be non-negative")
    started_at = _as_utc(now or datetime.now(timezone.utc))
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
    reports_dir.mkdir(parents=True, exist_ok=True)

    credit_usage = WizardCreditUsage(None, daily_credit_limit, None, False, "")
    preflight_blocker = "preflight_only"
    if execute:
        if not api_key:
            preflight_blocker = "missing_crypto_wizards_api_key"
        else:
            try:
                credit_payload = (credits_fetcher or fetch_credits_used)(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
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
            except (CryptoWizardsFetchError, ValueError, TypeError) as exc:
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
            requested_at = datetime.now(timezone.utc).isoformat()
            attempted_credits += cell.credit_cost
            try:
                payload = (prescanned_fetcher or fetch_prescanned_payload)(
                    api_key=api_key,
                    base_url=base_url,
                    priority=cell.priority,
                    strategy=cell.strategy,
                    exchange=cell.exchange,
                    interval=cell.interval,
                    timeout=timeout,
                )
                pairs = prescanned_pairs_from_payload(payload)
                response_hash = _json_hash(payload)
                raw_path = raw_dir / f"{sweep_id}_{cell.request_id}.json"
                _write_raw_snapshot(
                    raw_path,
                    cell=cell,
                    captured_at=requested_at,
                    payload=payload,
                    response_hash=response_hash,
                )
                raw_response_path = str(raw_path.relative_to(Path(root)))
                row_count = len(pairs)
                completed_at = datetime.now(timezone.utc).isoformat()
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
                completed_at = datetime.now(timezone.utc).isoformat()
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

    completed_cells = sum(row["status"] == "completed" for row in manifest_rows)
    failed_cells = sum(row["status"] == "failed" for row in manifest_rows)
    total_cells = len(manifest_rows)
    completion_percentage = round(100.0 * completed_cells / total_cells, 2) if total_cells else 0.0
    sweep_complete = bool(execute and total_cells and completed_cells == total_cells)
    if not execute:
        authority = "preflight_only"
        blocker = "execution_not_requested"
    elif preflight_blocker:
        authority = "blocked_partial_discovery"
        blocker = preflight_blocker
    elif failed_cells:
        authority = "blocked_partial_discovery"
        blocker = f"failed_requests:{failed_cells}"
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
                "credit_limit": credit_usage.limit,
                "credits_remaining_before": credit_usage.remaining,
                "reserved_credits": reserved_credits,
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
    manifest_frame.to_csv(attempt_manifest_path, index=False)
    candidate_frame.to_csv(attempt_candidates_path, index=False)

    prior_active_candidates = _read_csv(candidates_path)
    publish_current_snapshot = bool(sweep_complete and publish_active)
    active_snapshot_preserved = bool(
        not publish_current_snapshot
        and not prior_active_candidates.empty
        and manifest_path.exists()
    )
    if publish_current_snapshot or not active_snapshot_preserved:
        manifest_frame.to_csv(manifest_path, index=False)
        candidate_frame.to_csv(candidates_path, index=False)
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
        "credit_limit": credit_usage.limit,
        "credits_remaining_before": credit_usage.remaining,
        "reserved_credits": reserved_credits,
        "credit_usage_known": credit_usage.known,
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
    attempt_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    attempt_summary_md_path.write_text(_summary_markdown(summary), encoding="utf-8")
    if publish_current_snapshot or not active_snapshot_preserved:
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        summary_md_path.write_text(_summary_markdown(summary), encoding="utf-8")
    return WizardSweepResult(
        paths={
            "manifest": attempt_manifest_path,
            "candidates": attempt_candidates_path,
            "summary": attempt_summary_path,
            "summary_md": attempt_summary_md_path,
            "active_manifest": manifest_path,
            "active_candidates": candidates_path,
            "active_summary": summary_path,
            "active_summary_md": summary_md_path,
            "raw_snapshot_dir": raw_dir,
        },
        summary=summary,
    )


def _write_raw_snapshot(
    path: Path,
    *,
    cell: WizardSweepCell,
    captured_at: str,
    payload: Any,
    response_hash: str,
) -> None:
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
    path.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")


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


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
