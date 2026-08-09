from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import json
import random
import time
from typing import Iterable

import pandas as pd
import requests


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
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _build_default_capture_rows(sources: Iterable[str], prior: pd.DataFrame) -> list[dict[str, object]]:
    prior_rows = {str(row.get("source_id", "")).strip(): row.to_dict() for _, row in prior.iterrows()}
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
                "evidence": prior_rows.get(source, {}).get("evidence", template.get("evidence", "APIFY MCP URL")),
            }
        else:
            lower = source.lower()
            category = "supplemental"
            if "funding" in lower and "pulse" in lower:
                category = "cross_exchange_perp_feed"
            elif "hyperliquid" in lower and "funding" in lower:
                category = "perp_market_feed"
            elif "markets" in lower or "market" in lower:
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


def _extract_usage_details(run_payload: dict[str, object] | None, run_id: str | None) -> dict[str, object]:
    payload = run_payload or {}
    details: dict[str, object] = {
        "run_id": run_id,
        "started_at_utc": payload.get("startedAt") or payload.get("started_at") or payload.get("createdAt") or payload.get("started_at_utc"),
        "finished_at_utc": payload.get("finishedAt") or payload.get("finished_at") or payload.get("endedAt") or payload.get("ended_at"),
        "duration_ms": payload.get("stats", {}).get("runTimeMillis") if isinstance(payload.get("stats"), dict) else None,
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
    usage_amount = pick("cost", "amount", "amountUsd", "billingUsd", "costUsd", "credits", "computeUnits")
    usage_credits = pick("credits", "computeUnits", "platformCredits", "creditsUsed", "credits_used")

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
        "usage_meta": json.dumps(usage_meta, ensure_ascii=False, sort_keys=True) if isinstance(usage_meta, dict) else None,
    }


def _run_actor_fetch(api_token: str, source_id: str, source_input: dict[str, object] | None = None, timeout: int = 90) -> tuple[bool, str, int, list[object], dict[str, object]]:
    """Run an Apify actor and return (ok, status, rows, payload, usage)."""
    base = "https://api.apify.com"
    actor = source_id.replace("/", "~")
    run_payload = {
        "memory": 2048,
        "timeout": timeout * 1000,
        "input": source_input or {},
    }
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    params = {"token": api_token}

    run_url = f"{base}/v2/acts/{requests.utils.quote(actor, safe='~@')}/runs"
    created_at = datetime.now(timezone.utc)
    try:
        started = requests.post(run_url, headers=headers, params=params, json=run_payload, timeout=30)
    except Exception as exc:  # pragma: no cover - network dependent
        return False, f"network_error:{exc.__class__.__name__}", 0, [], {}

    if started.status_code >= 400:
        return False, f"run_request_failed:{started.status_code}", 0, [], {}

    started_json = started.json()
    run_data = started_json.get("data", started_json)
    run_id = run_data.get("id") or run_data.get("runId") or run_data.get("actRunId") or run_data.get("actorRunId")
    if not run_id:
        return False, "run_id_missing", 0, [], {}

    run_endpoint = f"{base}/v2/actor-runs/{run_id}"
    deadline = created_at.timestamp() + timeout
    payload = {}
    succeeded = False
    while time.time() < deadline:
        time.sleep(1 + random.uniform(0, 1))
        try:
            status = requests.get(run_endpoint, headers=headers, params=params, timeout=30)
        except Exception as exc:  # pragma: no cover - network dependent
            return False, f"run_poll_error:{exc.__class__.__name__}", 0, [], {}
        if status.status_code >= 400:
            return False, f"run_poll_failed:{status.status_code}", 0, [], {}
        payload = status.json().get("data", status.json())
        state = (payload.get("status") or "").upper()
        if state in {"SUCCEEDED", "FAILED", "ABORTED", "TIMED_OUT", "CRASHED"}:
            if state != "SUCCEEDED":
                return False, f"run_{state.lower()}", 0, [], _extract_usage_details(payload, run_id)
            succeeded = True
            break
    if not succeeded:
        return False, f"run_incomplete:{(payload.get('status') or 'unknown').lower()}", 0, [], _extract_usage_details(payload, run_id)

    dataset_id = payload.get("defaultDatasetId")
    if not dataset_id:
        return True, "succeeded_no_dataset", 0, [], _extract_usage_details(payload, run_id)

    item_url = f"{base}/v2/datasets/{dataset_id}/items"
    items_response = requests.get(
        item_url,
        headers=headers,
        params={"token": api_token, "format": "json", "clean": "1", "limit": 500},
        timeout=30,
    )
    if items_response.status_code >= 400:
        return False, f"dataset_fetch_failed:{items_response.status_code}", 0, [], _extract_usage_details(payload, run_id)

    items = items_response.json()
    if not isinstance(items, list):
        return True, "sampled", 0, [], _extract_usage_details(payload, run_id)
    usage = _extract_usage_details(payload, run_id)
    usage["dataset_id"] = dataset_id
    usage["item_count_reported"] = len(items)
    return True, "sampled", len(items), items, usage


def _update_row_after_fetch(row: dict[str, object], fetched_ok: bool, sample_status: str, rows: int, output_path: Path, now: str) -> dict[str, object]:
    out = dict(row)
    out["status"] = "available" if fetched_ok or sample_status == "sampled" else "needs_attention"
    out["sample_status"] = sample_status
    out["evidence"] = str(output_path)
    out["next_pipeline_action"] = (
        "normalize into market_venue_context and pair pipeline" if fetched_ok and sample_status == "sampled" else out.get("next_pipeline_action", "refresh required")
    )
    if fetched_ok and rows == 0:
        out["sample_status"] = "sampled_sparse"
    out["last_fetched"] = now
    out["sample_rows"] = rows
    return out


def refresh_apify_sources(
    *,
    root: Path,
    mcp_url: str,
    source_filter: str | None = None,
    do_fetch: bool = True,
    api_token: str | None = None,
    wait_seconds: int = 90,
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

    prior = pd.read_csv(coverage) if coverage.exists() else pd.DataFrame(columns=APIFY_DATASET_COLUMNS)
    rows = _build_default_capture_rows(source_set, prior)
    now = _now()

    sampled = 0
    needs_key = 0
    failed = 0
    manifest_rows = []

    for row in rows:
        source_id = str(row.get("source_id", "")).strip()
        row["status"] = row.get("status", "checked")
        run_status = "not_run"
        sample_status = str(row.get("sample_status", "not_sampled"))
        sample_rows = 0
        output_path = _snapshot_target_path(root, source_id)

        if do_fetch:
            if (not api_token) or sample_status.startswith("needs_api_key"):
                sample_status = "needs_api_key"
                usage_record = {
                    "run_id": None,
                    "started_at_utc": None,
                    "finished_at_utc": None,
                    "duration_ms": None,
                    "usage_currency": None,
                    "usage_credits": None,
                    "usage_amount": None,
                    "usage_meta": None,
                }
            elif row.get("category") in {"utility"}:
                sample_status = str(row.get("sample_status", "not_market_data"))
                usage_record = {
                    "run_id": None,
                    "started_at_utc": None,
                    "finished_at_utc": None,
                    "duration_ms": None,
                    "usage_currency": None,
                    "usage_credits": None,
                    "usage_amount": None,
                    "usage_meta": None,
                }
            else:
                ok, status, fetched_rows, items, usage = _run_actor_fetch(api_token=api_token.strip(), source_id=source_id, timeout=wait_seconds)
                run_status = status
                sample_status = status
                sample_rows = int(fetched_rows)
                usage_record = dict(usage or {})
                if ok and status in {"sampled", "sampled_sparse"}:
                    output_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
                    sampled += 1
                    row = _update_row_after_fetch(row, True, status, sample_rows, output_path, now)
                else:
                    failed += 1
                    row = _update_row_after_fetch(row, False, status, sample_rows, output_path, now)
        else:
            usage_record = {
                "run_id": None,
                "started_at_utc": None,
                "finished_at_utc": None,
                "duration_ms": None,
                "usage_currency": None,
                "usage_credits": None,
                "usage_amount": None,
                "usage_meta": None,
            }

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
            }
        )

    coverage_frame = pd.DataFrame(rows)
    coverage_frame = coverage_frame.sort_values("source_id").reset_index(drop=True)
    coverage_frame.to_csv(coverage, index=False)

    manifest_frame = pd.DataFrame(manifest_rows)
    manifest_frame = manifest_frame.reindex(columns=APIFY_MANIFEST_COLUMNS, fill_value="")
    manifest_frame.to_csv(manifest, index=False)

    return RefreshResult(
        coverage_path=coverage,
        manifest_path=manifest,
        source_count=len(source_set),
        sampled_count=sampled,
        needs_key_count=needs_key,
        failed_count=failed,
    )
