from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

from quant_platform.orchestration.corrective_external_effects import (
    ExternalEffectSession,
    current_external_effect_session,
    read_authorized_credential,
    run_authorized_credit_call,
)
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_csv,
    atomic_write_text,
    write_immutable_json,
)
from quant_platform.orchestration.effect_authority import EffectAuthorityError

APIFY_API_TARGET = "https://api.apify.com"
APIFY_TOKEN_ENV = "APIFY_API_TOKEN"
APIFY_ATTEMPT_SCHEMA_VERSION = "thewiz.apify_actor_attempt.v1"

APIFY_DATASET_COLUMNS = [
    "source_id",
    "category",
    "status",
    "sample_status",
    "primary_fields",
    "helps_pair_selection",
    "helps_liquidity",
    "helps_funding",
    "helps_history",
    "execution_authority",
    "integration_priority",
    "limitations",
    "next_pipeline_action",
    "evidence",
]


APIFY_MANIFEST_COLUMNS = [
    "timestamp_utc",
    "source_id",
    "run_status",
    "sample_status",
    "sample_rows",
    "output_path",
    "evidence",
    "run_id",
    "started_at_utc",
    "finished_at_utc",
    "duration_ms",
    "usage_currency",
    "usage_credits",
    "usage_amount",
    "usage_meta",
    "authorized_credit_ceiling",
    "effect_network_requests",
    "effect_credit_units",
    "effect_reservation_id",
    "credit_reconciliation_status",
    "actor_attempt_id",
    "actor_intent_path",
    "actor_terminal_path",
    "actor_reconciliation_required",
    "provider_response_count",
    "provider_response_binding_set_sha256",
]


@dataclass(frozen=True)
class RefreshResult:
    coverage_path: Path
    manifest_path: Path
    source_count: int
    sampled_count: int
    needs_key_count: int
    failed_count: int


def _to_slug(source_id: str) -> str:
    return source_id.strip().replace("/", "_").replace(" ", "_").replace("-", "_").lower()


def _normalize_source_name(source_id: str) -> str:
    return source_id.strip()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _apify_default_coverage_rows() -> dict[str, dict[str, object]]:
    return {
        "apify/actors": {
            "category": "utility",
            "status": "available",
            "sample_status": "not_market_data",
            "primary_fields": "Actor discovery and execution metadata",
            "helps_pair_selection": "no",
            "helps_liquidity": "no",
            "helps_funding": "no",
            "helps_history": "no",
            "execution_authority": "no",
            "integration_priority": "useful",
            "limitations": "Utility only, not a market feed",
            "next_pipeline_action": "Use for discovering and launching Apify actors",
            "evidence": "Configured in Apify MCP URL",
        },
        "apify/docs": {
            "category": "utility",
            "status": "available",
            "sample_status": "not_market_data",
            "primary_fields": "Apify documentation lookup",
            "helps_pair_selection": "no",
            "helps_liquidity": "no",
            "helps_funding": "no",
            "helps_history": "no",
            "execution_authority": "no",
            "integration_priority": "useful",
            "limitations": "Utility only, not a market feed",
            "next_pipeline_action": "Use for actor docs and integration details",
            "evidence": "Configured in Apify MCP URL",
        },
    }


def parse_apify_sources_from_mcp_url(url: str) -> list[str]:
    parsed = urlparse(url or "")
    tools_value = parse_qs(parsed.query).get("tools", [""])[0]
    if not tools_value:
        return []
    ids: list[str] = []
    for raw in tools_value.split(","):
        item = raw.strip()
        if not item:
            continue
        if item in {"actors", "docs"}:
            ids.append(f"apify/{item}")
            continue
        if "/" in item or "~" in item:
            ids.append(item)
    deduped: list[str] = []
    for source in ids:
        if source not in deduped:
            deduped.append(source)
    return deduped


def infer_apify_venue(source_id: str) -> str:
    lower = source_id.lower()
    if "binance" in lower:
        return "binance"
    if "coinbase" in lower:
        return "coinbase"
    if "bybit" in lower:
        return "bybit"
    if "hyperliquid" in lower:
        return "hyperliquid"
    if "dydx" in lower:
        return "dydx"
    if "funding-pulse" in lower or "fundingpulse" in lower:
        return "cross_exchange"
    if "coinglass" in lower:
        return "coinglass"
    if "dexscreener" in lower:
        return "dexscreener"
    if "gmx" in lower:
        return "gmx"
    if "coingecko" in lower:
        return "coingecko"
    if "coinmarketcap" in lower:
        return "coinmarketcap"
    return source_id.split("/")[0] if "/" in source_id else source_id


def _coverage_path(root: Path) -> Path:
    return root / "reports" / "active" / "apify_mcp_source_coverage_2026-06-25.csv"


def _manifest_path(root: Path) -> Path:
    return root / "reports" / "active" / "apify_source_capture_manifest.csv"


def _build_default_capture_rows(
    sources: Iterable[str], prior: pd.DataFrame
) -> list[dict[str, object]]:
    prior_rows = {
        str(row.get("source_id", "")).strip(): row.to_dict() for _, row in prior.iterrows()
    }
    defaults = _apify_default_coverage_rows()
    rows: list[dict[str, object]] = []
    for source in sources:
        source = _normalize_source_name(source)
        if not source:
            continue
        if source in defaults:
            template = dict(defaults[source])
            row = {
                "source_id": source,
                **template,
                "evidence": prior_rows.get(source, {}).get(
                    "evidence", template.get("evidence", "APIFY MCP URL")
                ),
            }
        else:
            lower = source.lower()
            category = "supplemental"
            if "funding" in lower and "pulse" in lower:
                category = "cross_exchange_perp_feed"
            elif (
                "hyperliquid" in lower
                and "funding" in lower
                or "markets" in lower
                or "market" in lower
            ):
                category = "perp_market_feed"
            elif "gmx" in lower:
                category = "defi_perp_feed" if "arbitrum" in lower else "defi_context"
            elif "coinglass" in lower:
                category = "cross_exchange_market_feed"
            elif "dex" in lower:
                category = "dex_liquidity_feed"
            elif "coinmarketcap" in lower or "coingecko" in lower:
                category = "broad_market_reference"

            row = {
                "source_id": source,
                "category": category,
                "status": "checked",
                "sample_status": "not_sampled",
                "primary_fields": "(to confirm)",
                "helps_pair_selection": "yes",
                "helps_liquidity": "limited",
                "helps_funding": "limited",
                "helps_history": "limited",
                "execution_authority": "partial",
                "integration_priority": "useful",
                "limitations": "Requires live actor execution path; sample not yet refreshed.",
                "next_pipeline_action": "refresh through refresh-apify-sources",
                "evidence": "Source configured in APIFY_MCP_SERVER_URL",
            }
            prior_row = prior_rows.get(source)
            if prior_row:
                for field, value in prior_row.items():
                    if field in row and value != "" and value is not None:
                        row[field] = value
        rows.append(row)
    return rows


def _classify_output_file(source_id: str) -> str:
    lower = source_id.lower()
    if "dydx" in lower:
        if "funding" in lower:
            return "apify_dydx_funding_snapshot"
        return "apify_dydx_markets_snapshot"
    return f"apify_{_to_slug(source_id)}_snapshot"


def _raw_root(root: Path) -> Path:
    return root / "data" / "raw"


def _dydx_inbox(root: Path) -> Path:
    return _raw_root(root) / "dydx_inbox"


def _enrichment_root(root: Path) -> Path:
    return _raw_root(root) / "enrichment"


def _snapshot_target_path(root: Path, source_id: str) -> Path:
    base_name = f"{_classify_output_file(source_id)}_latest.json"
    if "dydx" in source_id.lower():
        return _dydx_inbox(root) / base_name
    output_dir = _enrichment_root(root) / _to_slug(source_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / base_name


def _normalize_usage_value(value: object) -> str | int | float | None:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v if v else None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    return str(value)


def _extract_usage_details(
    run_payload: dict[str, object] | None, run_id: str | None
) -> dict[str, object]:
    payload = run_payload or {}
    details: dict[str, object] = {
        "run_id": run_id,
        "started_at_utc": payload.get("startedAt")
        or payload.get("started_at")
        or payload.get("createdAt")
        or payload.get("started_at_utc"),
        "finished_at_utc": payload.get("finishedAt")
        or payload.get("finished_at")
        or payload.get("endedAt")
        or payload.get("ended_at"),
        "duration_ms": payload.get("stats", {}).get("runTimeMillis")
        if isinstance(payload.get("stats"), dict)
        else None,
    }

    candidate_usage_fields = (
        payload.get("billing"),
        payload.get("usage"),
        payload.get("stats"),
        payload.get("meta", {}).get("stats") if isinstance(payload.get("meta"), dict) else None,
        payload.get("meta") if isinstance(payload.get("meta"), dict) else None,
    )
    usage_currency = None
    usage_amount = None
    usage_credits = None
    usage_meta = None

    def pick(*keys):
        for field in candidate_usage_fields:
            if not isinstance(field, dict):
                continue
            for key in keys:
                if key in field:
                    return field[key]
        return None

    usage_currency = pick("currency", "cost_currency")
    usage_amount = pick(
        "cost", "amount", "amountUsd", "billingUsd", "costUsd", "credits", "computeUnits"
    )
    usage_credits = pick(
        "credits", "computeUnits", "platformCredits", "creditsUsed", "credits_used"
    )

    if isinstance(payload.get("meta"), dict):
        usage_meta = payload["meta"]
    elif isinstance(payload.get("billing"), dict):
        usage_meta = payload["billing"]
    elif isinstance(payload.get("usage"), dict):
        usage_meta = payload["usage"]
    elif isinstance(payload.get("stats"), dict):
        usage_meta = payload["stats"]

    return {
        "run_id": run_id,
        "started_at_utc": _normalize_usage_value(details["started_at_utc"]),
        "finished_at_utc": _normalize_usage_value(details["finished_at_utc"]),
        "duration_ms": _normalize_usage_value(details["duration_ms"]),
        "usage_currency": _normalize_usage_value(usage_currency),
        "usage_amount": _normalize_usage_value(usage_amount),
        "usage_credits": _normalize_usage_value(usage_credits),
        "usage_meta": json.dumps(usage_meta, ensure_ascii=False, sort_keys=True)
        if isinstance(usage_meta, dict)
        else None,
    }


def _authorized_apify_http_call(
    *,
    operation: str,
    payload: dict[str, object],
    credit_cost: int,
    callback: Callable[[], object],
    response_recorder: Callable[[str, object], str] | None = None,
) -> object:
    methods = {
        "APIFY_ACTOR_START": "POST",
        "APIFY_ACTOR_POLL": "GET",
        "APIFY_DATASET_FETCH": "GET",
    }
    try:
        method = methods[operation]
    except KeyError as exc:
        raise ValueError("unsupported governed Apify operation") from exc
    request_payload = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return run_authorized_credit_call(
        target=APIFY_API_TARGET,
        operation=operation,
        method=method,
        request_payload=request_payload,
        request_count=1,
        credit_cost=credit_cost,
        callback=callback,
        result_recorder=(
            None
            if response_recorder is None
            else lambda response: response_recorder(operation, response)
        ),
    )


def _run_actor_fetch(
    api_token: str,
    source_id: str,
    source_input: dict[str, object] | None = None,
    timeout: int = 90,
    *,
    actor_credit_ceiling: int,
    request_client: object = requests,
    sleep_fn: Callable[[float], None] = time.sleep,
    jitter_fn: Callable[[float, float], float] = random.uniform,
    clock_fn: Callable[[], float] = time.time,
    response_recorder: Callable[[str, object], str] | None = None,
) -> tuple[bool, str, int, list[object], dict[str, object]]:
    """Run one actor through reservation-bound credential, network, and credit permits."""

    if not api_token.strip():
        raise ValueError("Apify API token is required")
    if not source_id.strip():
        raise ValueError("Apify source_id is required")
    if timeout <= 0:
        raise ValueError("Apify timeout must be positive")
    if actor_credit_ceiling <= 0:
        raise ValueError("Apify actor_credit_ceiling must be positive")

    base = APIFY_API_TARGET
    actor = source_id.replace("/", "~")
    run_payload = {
        "memory": 2048,
        "timeout": timeout * 1000,
        "input": source_input or {},
    }
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

    run_url = f"{base}/v2/acts/{requests.utils.quote(actor, safe='~@')}/runs"
    created_at = clock_fn()
    try:
        started = _authorized_apify_http_call(
            operation="APIFY_ACTOR_START",
            payload={
                "actor": source_id,
                "input": source_input or {},
                "memory_mb": 2048,
                "timeout_seconds": timeout,
            },
            credit_cost=actor_credit_ceiling,
            callback=lambda: request_client.post(
                run_url,
                headers=headers,
                json=run_payload,
                timeout=30,
            ),
            response_recorder=response_recorder,
        )
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        return False, f"network_error:{exc.__class__.__name__}", 0, [], {}

    if started.status_code >= 400:
        return False, f"run_request_failed:{started.status_code}", 0, [], {}

    try:
        started_json = started.json()
    except (TypeError, ValueError):
        return False, "run_response_invalid_json", 0, [], {}
    if not isinstance(started_json, dict):
        return False, "run_response_invalid_shape", 0, [], {}
    run_data = started_json.get("data", started_json)
    if not isinstance(run_data, dict):
        return False, "run_data_invalid_shape", 0, [], {}
    run_id = (
        run_data.get("id")
        or run_data.get("runId")
        or run_data.get("actRunId")
        or run_data.get("actorRunId")
    )
    if not run_id:
        return False, "run_id_missing", 0, [], {}

    run_endpoint = f"{base}/v2/actor-runs/{run_id}"
    deadline = created_at + timeout
    payload: dict[str, object] = {}
    succeeded = False
    while clock_fn() < deadline:
        sleep_fn(1 + jitter_fn(0, 1))
        try:
            status = _authorized_apify_http_call(
                operation="APIFY_ACTOR_POLL",
                payload={"run_id": str(run_id)},
                credit_cost=0,
                callback=lambda: request_client.get(
                    run_endpoint,
                    headers=headers,
                    timeout=30,
                ),
                response_recorder=response_recorder,
            )
        except requests.RequestException as exc:  # pragma: no cover - network dependent
            return False, f"run_poll_error:{exc.__class__.__name__}", 0, [], {}
        if status.status_code >= 400:
            return False, f"run_poll_failed:{status.status_code}", 0, [], {}
        try:
            status_json = status.json()
        except (TypeError, ValueError):
            return False, "run_poll_invalid_json", 0, [], {}
        if not isinstance(status_json, dict):
            return False, "run_poll_invalid_shape", 0, [], {}
        payload_value = status_json.get("data", status_json)
        if not isinstance(payload_value, dict):
            return False, "run_poll_data_invalid_shape", 0, [], {}
        payload = payload_value
        state = (payload.get("status") or "").upper()
        if state in {"SUCCEEDED", "FAILED", "ABORTED", "TIMED_OUT", "CRASHED"}:
            if state != "SUCCEEDED":
                return False, f"run_{state.lower()}", 0, [], _extract_usage_details(payload, run_id)
            succeeded = True
            break
    if not succeeded:
        return (
            False,
            f"run_incomplete:{(payload.get('status') or 'unknown').lower()}",
            0,
            [],
            _extract_usage_details(payload, run_id),
        )

    dataset_id = payload.get("defaultDatasetId")
    if not dataset_id:
        return True, "succeeded_no_dataset", 0, [], _extract_usage_details(payload, run_id)

    item_url = f"{base}/v2/datasets/{dataset_id}/items"
    items_response = _authorized_apify_http_call(
        operation="APIFY_DATASET_FETCH",
        payload={"dataset_id": str(dataset_id), "format": "json", "clean": True, "limit": 500},
        credit_cost=0,
        callback=lambda: request_client.get(
            item_url,
            headers=headers,
            params={"format": "json", "clean": "1", "limit": 500},
            timeout=30,
        ),
        response_recorder=response_recorder,
    )
    if items_response.status_code >= 400:
        return (
            False,
            f"dataset_fetch_failed:{items_response.status_code}",
            0,
            [],
            _extract_usage_details(payload, run_id),
        )

    try:
        items = items_response.json()
    except (TypeError, ValueError):
        return False, "dataset_invalid_json", 0, [], _extract_usage_details(payload, run_id)
    if not isinstance(items, list):
        return False, "dataset_invalid_shape", 0, [], _extract_usage_details(payload, run_id)
    usage = _extract_usage_details(payload, run_id)
    usage["dataset_id"] = dataset_id
    usage["item_count_reported"] = len(items)
    return True, "sampled", len(items), items, usage


def _credit_reconciliation_status(value: object, ceiling: int) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return "UNKNOWN_REPORTED_USAGE"
    try:
        reported = float(value)
    except (TypeError, ValueError):
        return "BLOCKED_INVALID_REPORTED_USAGE"
    if not math.isfinite(reported) or reported < 0:
        return "BLOCKED_INVALID_REPORTED_USAGE"
    if reported > ceiling:
        return "BLOCKED_REPORTED_USAGE_EXCEEDS_AUTHORIZED_CEILING"
    return "PASS_REPORTED_USAGE_WITHIN_AUTHORIZED_CEILING"


def _update_row_after_fetch(
    row: dict[str, object],
    fetched_ok: bool,
    sample_status: str,
    rows: int,
    output_path: Path,
    now: str,
) -> dict[str, object]:
    out = dict(row)
    out["status"] = "available" if fetched_ok or sample_status == "sampled" else "needs_attention"
    out["sample_status"] = sample_status
    out["evidence"] = str(output_path)
    out["next_pipeline_action"] = (
        "normalize into market_venue_context and pair pipeline"
        if fetched_ok and sample_status == "sampled"
        else out.get("next_pipeline_action", "refresh required")
    )
    if fetched_ok and rows == 0:
        out["sample_status"] = "sampled_sparse"
    out["last_fetched"] = now
    out["sample_rows"] = rows
    return out


def _apify_attempt_id(*, session: ExternalEffectSession, source_id: str) -> str:
    material = (
        f"{session.run_id}|{session.intended_slot_id}|"
        f"{session.account_scope_id}|{session.reservation_id}|{source_id}"
    )
    return f"apifyattempt_{sha256(material.encode('utf-8')).hexdigest()[:24]}"


def _apify_attempt_paths(root: Path, attempt_id: str) -> tuple[Path, Path]:
    base = root / "data" / "research" / "apify_actor_attempts"
    return (
        base / "intents" / f"{attempt_id}.json",
        base / "terminals" / f"{attempt_id}.json",
    )


def _seal_apify_attempt(body: dict[str, object]) -> dict[str, object]:
    payload_sha256 = sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {**body, "payload_sha256": payload_sha256}


def _load_apify_attempt(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Apify actor attempt must be an object")
    observed = str(payload.get("payload_sha256", ""))
    body = dict(payload)
    body.pop("payload_sha256", None)
    expected = str(_seal_apify_attempt(body)["payload_sha256"])
    if observed != expected:
        raise ValueError("Apify actor attempt hash mismatch")
    return payload


def _assert_no_unresolved_apify_attempt(
    *,
    root: Path,
    source_id: str,
    account_scope_id: str,
) -> None:
    intent_root = root / "data" / "research" / "apify_actor_attempts" / "intents"
    if not intent_root.exists():
        return
    for intent_path in sorted(intent_root.glob("*.json")):
        intent = _load_apify_attempt(intent_path)
        if (
            str(intent.get("source_id", "")) != source_id
            or str(intent.get("account_scope_id", "")) != account_scope_id
        ):
            continue
        attempt_id = str(intent.get("attempt_id", ""))
        _, terminal_path = _apify_attempt_paths(root, attempt_id)
        if not terminal_path.exists():
            raise EffectAuthorityError("apify_actor_start_reconciliation_required")
        terminal = _load_apify_attempt(terminal_path)
        if terminal.get("reconciliation_required") is True:
            raise EffectAuthorityError("apify_actor_start_reconciliation_required")


def _publish_apify_actor_intent(
    *,
    root: Path,
    session: ExternalEffectSession,
    source_id: str,
    actor_credit_ceiling: int,
    timestamp_utc: str,
) -> tuple[dict[str, object], Path, Path]:
    _assert_no_unresolved_apify_attempt(
        root=root,
        source_id=source_id,
        account_scope_id=session.account_scope_id,
    )
    attempt_id = _apify_attempt_id(session=session, source_id=source_id)
    intent_path, terminal_path = _apify_attempt_paths(root, attempt_id)
    if intent_path.exists():
        raise EffectAuthorityError("apify_actor_attempt_already_exists_for_run")
    body: dict[str, object] = {
        "schema_version": APIFY_ATTEMPT_SCHEMA_VERSION,
        "record_type": "ACTOR_START_INTENT",
        "attempt_id": attempt_id,
        "created_at_utc": timestamp_utc,
        "scheduler_run_id": session.run_id,
        "intended_slot_id": session.intended_slot_id,
        "provider_id": "apify",
        "account_scope_id": session.account_scope_id,
        "source_id": source_id,
        "source_input_sha256": sha256(b"{}").hexdigest(),
        "effect_reservation_id": session.reservation_id,
        "effect_reservation_binding_id": session.reservation_binding_id,
        "actor_credit_ceiling": actor_credit_ceiling,
        "order_submission_included": False,
        "promotion_authority": False,
        "live_trading_authorized": False,
    }
    intent = _seal_apify_attempt(body)
    write_immutable_json(intent_path, intent)
    return intent, intent_path, terminal_path


def _record_apify_provider_response(
    *,
    root: Path,
    session: ExternalEffectSession,
    source_id: str,
    attempt_id: str,
    bindings: list[dict[str, str]],
    operation: str,
    response: object,
) -> str:
    call_index = session.consumed_requests
    if call_index <= 0:
        raise EffectAuthorityError("apify_response_call_index_invalid")
    try:
        body = response.json()
    except (AttributeError, TypeError, ValueError) as exc:
        body = {
            "response_json_error": type(exc).__name__,
            "response_text_sha256": sha256(
                str(getattr(response, "text", "") or "").encode("utf-8")
            ).hexdigest(),
        }
    normalized_body = json.loads(
        json.dumps(body, ensure_ascii=True, sort_keys=True, default=str)
    )
    body_sha256 = sha256(
        json.dumps(
            normalized_body,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    path = (
        root
        / "data"
        / "raw"
        / "apify"
        / "provider_responses"
        / datetime.now(UTC).date().isoformat()
        / session.run_id
        / f"{attempt_id}_{call_index:04d}_{operation.lower()}.json"
    )
    envelope = {
        "schema_version": "thewiz.apify_provider_response.v1",
        "captured_at_utc": _now(),
        "attempt_id": attempt_id,
        "scheduler_run_id": session.run_id,
        "intended_slot_id": session.intended_slot_id,
        "provider_id": "apify",
        "account_scope_id": session.account_scope_id,
        "source_id": source_id,
        "effect_reservation_id": session.reservation_id,
        "call_index": call_index,
        "operation": operation,
        "status_code": int(getattr(response, "status_code", 0) or 0),
        "response_body_sha256": body_sha256,
        "response": normalized_body,
        "order_submission_included": False,
        "promotion_authority": False,
        "live_trading_authorized": False,
    }
    write_immutable_json(path, envelope)
    file_sha256 = sha256(path.read_bytes()).hexdigest()
    bindings.append(
        {
            "operation": operation,
            "path": str(path.relative_to(root)),
            "sha256": file_sha256,
        }
    )
    return file_sha256


def _apify_outcome_requires_reconciliation(status: str) -> bool:
    normalized = status.strip().lower()
    if normalized.startswith(
        (
            "network_error:",
            "run_response_invalid_",
            "run_data_invalid_",
            "run_id_missing",
            "run_poll_",
            "run_incomplete:",
        )
    ):
        return True
    if normalized.startswith("run_request_failed:"):
        try:
            return int(normalized.rsplit(":", 1)[1]) >= 500
        except ValueError:
            return True
    return False


def _publish_apify_actor_terminal(
    *,
    intent: dict[str, object],
    terminal_path: Path,
    timestamp_utc: str,
    provider_status: str,
    final_status: str,
    usage_record: dict[str, object],
    request_delta: int,
    credit_delta: int,
    manifest_path: Path,
    provider_response_bindings: list[dict[str, str]],
) -> bool:
    reconciliation_required = _apify_outcome_requires_reconciliation(
        provider_status
    )
    body: dict[str, object] = {
        "schema_version": APIFY_ATTEMPT_SCHEMA_VERSION,
        "record_type": "ACTOR_ATTEMPT_TERMINAL",
        "attempt_id": str(intent["attempt_id"]),
        "completed_at_utc": timestamp_utc,
        "scheduler_run_id": str(intent["scheduler_run_id"]),
        "intended_slot_id": str(intent["intended_slot_id"]),
        "provider_id": "apify",
        "account_scope_id": str(intent["account_scope_id"]),
        "source_id": str(intent["source_id"]),
        "effect_reservation_id": str(intent["effect_reservation_id"]),
        "effect_reservation_binding_id": str(
            intent["effect_reservation_binding_id"]
        ),
        "provider_run_id": usage_record.get("run_id"),
        "provider_status": provider_status,
        "final_status": final_status,
        "effect_network_requests": request_delta,
        "effect_credit_units": credit_delta,
        "reconciliation_required": reconciliation_required,
        "active_manifest_path": str(manifest_path),
        "provider_response_bindings": provider_response_bindings,
        "provider_response_binding_set_sha256": sha256(
            json.dumps(
                provider_response_bindings,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "order_submission_included": False,
        "promotion_authority": False,
        "live_trading_authorized": False,
    }
    write_immutable_json(terminal_path, _seal_apify_attempt(body))
    return reconciliation_required


def refresh_apify_sources(
    *,
    root: Path,
    mcp_url: str,
    source_filter: str | None = None,
    do_fetch: bool = True,
    api_token: str | None = None,
    wait_seconds: int = 90,
    actor_credit_ceiling: int = 0,
    actor_fetcher: Callable[..., tuple[bool, str, int, list[object], dict[str, object]]]
    | None = None,
) -> RefreshResult:
    root.mkdir(parents=True, exist_ok=True)
    root_reports = root / "reports" / "active"
    root_reports.mkdir(parents=True, exist_ok=True)
    _dydx_inbox(root).mkdir(parents=True, exist_ok=True)
    _enrichment_root(root).mkdir(parents=True, exist_ok=True)

    coverage = _coverage_path(root)
    manifest = _manifest_path(root)

    sources = parse_apify_sources_from_mcp_url(mcp_url)
    source_set = sorted({_normalize_source_name(s) for s in sources})
    if source_filter:
        source_set = [s for s in source_set if s.lower() == source_filter.lower()]

    prior = (
        pd.read_csv(coverage) if coverage.exists() else pd.DataFrame(columns=APIFY_DATASET_COLUMNS)
    )
    rows = _build_default_capture_rows(source_set, prior)
    now = _now()
    fetchable_sources = [row for row in rows if row.get("category") not in {"utility"}]
    authorized_token: str | None = None
    if do_fetch and fetchable_sources:
        if actor_credit_ceiling <= 0:
            raise ValueError("Apify fetch requires a positive actor_credit_ceiling")
        if current_external_effect_session() is None:
            raise EffectAuthorityError("apify_external_effect_session_missing")
        authorized_token = read_authorized_credential(
            APIFY_TOKEN_ENV,
            reader=lambda _key: api_token or os.getenv(APIFY_TOKEN_ENV),
        )
    actor_fetcher = actor_fetcher or _run_actor_fetch

    sampled = 0
    needs_key = 0
    failed = 0
    manifest_rows = []

    for row_index, row in enumerate(rows):
        source_id = str(row.get("source_id", "")).strip()
        row["status"] = row.get("status", "checked")
        run_status = "not_run"
        provider_status = "not_run"
        sample_status = str(row.get("sample_status", "not_sampled"))
        sample_rows = 0
        output_path = _snapshot_target_path(root, source_id)
        actor_intent: dict[str, object] | None = None
        actor_intent_path: Path | None = None
        actor_terminal_path: Path | None = None
        actor_reconciliation_required = False
        provider_response_bindings: list[dict[str, str]] = []
        usage_record: dict[str, object] = {
            "run_id": None,
            "started_at_utc": None,
            "finished_at_utc": None,
            "duration_ms": None,
            "usage_currency": None,
            "usage_credits": None,
            "usage_amount": None,
            "usage_meta": None,
            "authorized_credit_ceiling": 0,
            "effect_network_requests": 0,
            "effect_credit_units": 0,
            "effect_reservation_id": "",
            "credit_reconciliation_status": "NOT_REQUIRED",
        }

        if do_fetch:
            if row.get("category") in {"utility"}:
                sample_status = str(row.get("sample_status", "not_market_data"))
            else:
                session = current_external_effect_session()
                if session is None or authorized_token is None:
                    raise EffectAuthorityError("apify_authority_lost_before_actor_call")
                (
                    actor_intent,
                    actor_intent_path,
                    actor_terminal_path,
                ) = _publish_apify_actor_intent(
                    root=root,
                    session=session,
                    source_id=source_id,
                    actor_credit_ceiling=actor_credit_ceiling,
                    timestamp_utc=now,
                )
                attempt_id = str(actor_intent["attempt_id"])

                def record_provider_response(
                    operation: str,
                    response: object,
                    *,
                    _session: ExternalEffectSession = session,
                    _source_id: str = source_id,
                    _attempt_id: str = attempt_id,
                    _bindings: list[dict[str, str]] = provider_response_bindings,
                ) -> str:
                    return _record_apify_provider_response(
                        root=root,
                        session=_session,
                        source_id=_source_id,
                        attempt_id=_attempt_id,
                        bindings=_bindings,
                        operation=operation,
                        response=response,
                    )

                requests_before = session.consumed_requests
                credits_before = session.consumed_credits
                ok, status, fetched_rows, items, usage = actor_fetcher(
                    api_token=authorized_token,
                    source_id=source_id,
                    timeout=wait_seconds,
                    actor_credit_ceiling=actor_credit_ceiling,
                    response_recorder=record_provider_response,
                )
                request_delta = session.consumed_requests - requests_before
                credit_delta = session.consumed_credits - credits_before
                provider_status = status
                run_status = status
                sample_status = status
                sample_rows = int(fetched_rows)
                usage_record = dict(usage or {})
                credit_reconciliation = _credit_reconciliation_status(
                    usage_record.get("usage_credits"),
                    actor_credit_ceiling,
                )
                effect_accounting_valid = request_delta > 0 and credit_delta == actor_credit_ceiling
                usage_record.update(
                    {
                        "authorized_credit_ceiling": actor_credit_ceiling,
                        "effect_network_requests": request_delta,
                        "effect_credit_units": credit_delta,
                        "effect_reservation_id": session.reservation_id,
                        "credit_reconciliation_status": credit_reconciliation,
                    }
                )
                if not effect_accounting_valid:
                    ok = False
                    status = "effect_accounting_mismatch"
                elif credit_reconciliation != ("PASS_REPORTED_USAGE_WITHIN_AUTHORIZED_CEILING"):
                    ok = False
                    status = credit_reconciliation.lower()
                run_status = status
                sample_status = status
                if ok and status in {"sampled", "sampled_sparse"}:
                    atomic_write_text(output_path, json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
                    sampled += 1
                    row = _update_row_after_fetch(row, True, status, sample_rows, output_path, now)
                else:
                    failed += 1
                    row = _update_row_after_fetch(row, False, status, sample_rows, output_path, now)

        if sample_status == "needs_api_key":
            needs_key += 1
        elif row.get("sample_status") in {"sampled", "sampled_sparse"}:
            pass

        manifest_rows.append(
            {
                "timestamp_utc": now,
                "source_id": source_id,
                "run_status": run_status,
                "sample_status": sample_status,
                "sample_rows": sample_rows,
                "output_path": str(output_path),
                "evidence": str(row.get("evidence", "")),
                "run_id": usage_record.get("run_id", None),
                "started_at_utc": usage_record.get("started_at_utc", None),
                "finished_at_utc": usage_record.get("finished_at_utc", None),
                "duration_ms": usage_record.get("duration_ms", None),
                "usage_currency": usage_record.get("usage_currency", None),
                "usage_credits": usage_record.get("usage_credits", None),
                "usage_amount": usage_record.get("usage_amount", None),
                "usage_meta": usage_record.get("usage_meta", None),
                "authorized_credit_ceiling": usage_record.get("authorized_credit_ceiling", 0),
                "effect_network_requests": usage_record.get("effect_network_requests", 0),
                "effect_credit_units": usage_record.get("effect_credit_units", 0),
                "effect_reservation_id": usage_record.get("effect_reservation_id", ""),
                "credit_reconciliation_status": usage_record.get(
                    "credit_reconciliation_status", "NOT_REQUIRED"
                ),
                "actor_attempt_id": str((actor_intent or {}).get("attempt_id", "")),
                "actor_intent_path": str(actor_intent_path or ""),
                "actor_terminal_path": str(actor_terminal_path or ""),
                "actor_reconciliation_required": actor_reconciliation_required,
                "provider_response_count": len(provider_response_bindings),
                "provider_response_binding_set_sha256": sha256(
                    json.dumps(
                        provider_response_bindings,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
        rows[row_index] = row
        coverage_frame = pd.DataFrame(rows)
        coverage_frame = coverage_frame.sort_values("source_id").reset_index(drop=True)
        atomic_write_csv(coverage_frame, coverage, index=False)
        manifest_frame = pd.DataFrame(manifest_rows)
        manifest_frame = manifest_frame.reindex(
            columns=APIFY_MANIFEST_COLUMNS,
            fill_value="",
        )
        atomic_write_csv(manifest_frame, manifest, index=False)
        if actor_intent is not None and actor_terminal_path is not None:
            actor_reconciliation_required = _publish_apify_actor_terminal(
                intent=actor_intent,
                terminal_path=actor_terminal_path,
                timestamp_utc=_now(),
                provider_status=provider_status,
                final_status=run_status,
                usage_record=usage_record,
                request_delta=int(
                    usage_record.get("effect_network_requests", 0) or 0
                ),
                credit_delta=int(usage_record.get("effect_credit_units", 0) or 0),
                manifest_path=manifest,
                provider_response_bindings=provider_response_bindings,
            )
            manifest_rows[-1]["actor_reconciliation_required"] = (
                actor_reconciliation_required
            )
            manifest_frame = pd.DataFrame(manifest_rows).reindex(
                columns=APIFY_MANIFEST_COLUMNS,
                fill_value="",
            )
            atomic_write_csv(manifest_frame, manifest, index=False)
            if actor_reconciliation_required:
                raise EffectAuthorityError(
                    "apify_actor_start_reconciliation_required"
                )

    coverage_frame = pd.DataFrame(rows)
    coverage_frame = coverage_frame.sort_values("source_id").reset_index(drop=True)
    atomic_write_csv(coverage_frame, coverage, index=False)

    manifest_frame = pd.DataFrame(manifest_rows)
    manifest_frame = manifest_frame.reindex(columns=APIFY_MANIFEST_COLUMNS, fill_value="")
    atomic_write_csv(manifest_frame, manifest, index=False)

    return RefreshResult(
        coverage_path=coverage,
        manifest_path=manifest,
        source_count=len(source_set),
        sampled_count=sampled,
        needs_key_count=needs_key,
        failed_count=failed,
    )
