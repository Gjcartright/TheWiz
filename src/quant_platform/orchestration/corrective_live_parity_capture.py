"""Read-only Hyperliquid Testnet/live input-parity evidence capture."""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_platform.orchestration.corrective_hyperliquid_network import (
    run_authorized_hyperliquid_info_call,
)
from quant_platform.orchestration.corrective_live_canary import (
    PARITY_ARTIFACT_SCHEMA_VERSION,
    PARITY_INPUT_CONTRACTS,
    PARITY_SCHEMA_VERSION,
    REQUIRED_PARITY_INPUTS,
    _parity_rows,
    _payload_hash,
    _policy_id,
    _validate_policy_core,
)
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import promote_staged_file
from quant_platform.orchestration.effect_authority import EffectAuthorityError

ROOT = Path(__file__).resolve().parents[3]
HYPERLIQUID_TESTNET_INFO_URL = "https://api.hyperliquid-testnet.xyz/info"
HYPERLIQUID_LIVE_INFO_URL = "https://api.hyperliquid.xyz/info"
ALLOWED_INFO_REQUEST_TYPES = {
    "metaAndAssetCtxs",
    "allMids",
    "l2Book",
    "clearinghouseState",
}
MINIMUM_ORDER_NOTIONAL_USD = 10.0


def capture_live_input_parity_evidence(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    account_address: str | None = None,
    testnet_session: requests.Session | None = None,
    live_session: requests.Session | None = None,
    timeout: int = 20,
    calculation_artifact_path: Path | None = None,
    refresh_upstream: bool = True,
    upstream_prevalidated: bool = False,
) -> dict[str, Any]:
    """Capture six Testnet/live parity inputs without signing or submitting."""

    as_of = _as_utc(now)
    active = root / "reports" / "active"
    status_path = active / "live_input_parity_capture_status.json"
    evidence_path = active / "testnet_live_input_parity_evidence.json"
    parity_path = active / "testnet_live_input_parity.csv"
    candidate_path = active / "testnet_candidate_receipt.json"
    sample_path = active / "realized_testnet_sample_sufficiency.csv"
    supreme_path = root / "reports" / "supreme_team" / "testnet_evidence_checkpoint.json"
    policy_path = root / "config" / "live_canary_policy.json"

    from quant_platform.orchestration.corrective_release_gates import (
        _validated_testnet_candidate_receipt,
        build_testnet_supreme_team_checkpoint,
    )

    upstream_revalidated = bool(upstream_prevalidated)
    if refresh_upstream:
        build_testnet_supreme_team_checkpoint(root=root, now=as_of)
        upstream_revalidated = True
    candidate = _read_json(candidate_path)
    sample = _read_csv(sample_path)
    supreme = _read_json(supreme_path)
    policy = _read_json(policy_path)
    address = str(account_address or os.getenv("HYPERLIQUID_MASTER_ADDRESS", "")).strip()

    candidate_valid, candidate_blockers = _validated_testnet_candidate_receipt(
        root=root,
        candidate=candidate,
    )
    sample_pass = bool(not sample.empty and sample.get("status", pd.Series(dtype=str)).eq("PASS").all())
    supreme_pass = supreme.get("checkpoint_status") == "PASS"
    _policy_valid, policy_blockers = _validate_policy_core(policy)
    blockers = [
        *([] if upstream_revalidated else ["stage6_upstream_evidence_not_revalidated"]),
        *([] if candidate_valid else candidate_blockers or ["validated_live_candidate_identity_missing"]),
        *([] if sample_pass else ["realized_testnet_sample_sufficiency_not_passed"]),
        *([] if supreme_pass else ["testnet_supreme_team_checkpoint_not_passed"]),
        *policy_blockers,
        *([] if _valid_address(address) else ["hyperliquid_public_account_address_missing_or_invalid"]),
        *([] if timeout > 0 else ["live_input_parity_timeout_invalid"]),
    ]
    if blockers:
        status = _capture_status(
            generated_at=as_of,
            status="BLOCKED_UPSTREAM",
            blockers=blockers,
            evidence_path=evidence_path,
            parity_path=parity_path,
            upstream_revalidated=upstream_revalidated,
        )
        _atomic_json(status, status_path)
        return {"status": status["status"], "blockers": blockers, "status_path": status_path}

    assets = (
        str(candidate.get("asset_x", "")).strip().upper(),
        str(candidate.get("asset_y", "")).strip().upper(),
    )
    if not all(assets) or assets[0] == assets[1]:
        blockers = ["live_input_parity_candidate_assets_invalid"]
        status = _capture_status(
            generated_at=as_of,
            status="BLOCKED_UPSTREAM",
            blockers=blockers,
            evidence_path=evidence_path,
            parity_path=parity_path,
            upstream_revalidated=upstream_revalidated,
        )
        _atomic_json(status, status_path)
        return {"status": status["status"], "blockers": blockers, "status_path": status_path}

    capture_id = (
        as_of.strftime("%Y%m%dT%H%M%S%fZ")
        + "_"
        + str(candidate.get("candidate_receipt_id", ""))[-12:]
    )
    raw_root = root / "data" / "live" / "parity_raw" / capture_id
    snapshot_root = root / "data" / "live" / "parity_snapshots" / capture_id
    test_client = testnet_session or requests.Session()
    live_client = live_session or requests.Session()
    environments = {
        "testnet": (test_client, HYPERLIQUID_TESTNET_INFO_URL),
        "live": (live_client, HYPERLIQUID_LIVE_INFO_URL),
    }
    raw: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in environments}
    try:
        for request_name, request_payload in (
            ("meta", {"type": "metaAndAssetCtxs"}),
            ("mids", {"type": "allMids"}),
        ):
            for environment, (session, info_url) in environments.items():
                raw[environment][request_name] = _capture_info_request(
                    root=root,
                    raw_root=raw_root,
                    environment=environment,
                    request_name=request_name,
                    info_url=info_url,
                    payload=request_payload,
                    session=session,
                    captured_at=as_of,
                    timeout=timeout,
                )
        for asset in assets:
            request_name = f"l2_{asset}"
            for environment, (session, info_url) in environments.items():
                raw[environment][request_name] = _capture_info_request(
                    root=root,
                    raw_root=raw_root,
                    environment=environment,
                    request_name=request_name,
                    info_url=info_url,
                    payload={"type": "l2Book", "coin": asset},
                    session=session,
                    captured_at=as_of,
                    timeout=timeout,
                )
        for environment, (session, info_url) in environments.items():
            raw[environment]["margin"] = _capture_info_request(
                root=root,
                raw_root=raw_root,
                environment=environment,
                request_name="margin",
                info_url=info_url,
                payload={"type": "clearinghouseState", "user": address},
                session=session,
                captured_at=as_of,
                timeout=timeout,
            )
        normalized = {
            environment: _normalize_environment(
                payloads=payloads,
                assets=assets,
                captured_at=as_of,
                target_notional_usd=float(policy["canary"]["maximum_total_notional_usd"]) / 2.0,
            )
            for environment, payloads in raw.items()
        }
    except (
        EffectAuthorityError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        requests.RequestException,
    ) as exc:
        blockers = [f"live_input_parity_capture_failed:{safe_exception_code(exc)}"]
        status = _capture_status(
            generated_at=as_of,
            status="BLOCKED_CAPTURE_FAILED",
            blockers=blockers,
            evidence_path=evidence_path,
            parity_path=parity_path,
            upstream_revalidated=upstream_revalidated,
        )
        _atomic_json(status, status_path)
        return {"status": status["status"], "blockers": blockers, "status_path": status_path}

    calculation_path = calculation_artifact_path or Path(__file__)
    try:
        _relative(calculation_path, root)
    except (OSError, ValueError) as exc:
        blockers = [f"live_input_parity_calculation_artifact_invalid:{safe_exception_code(exc)}"]
        status = _capture_status(
            generated_at=as_of,
            status="BLOCKED_CAPTURE_FAILED",
            blockers=blockers,
            evidence_path=evidence_path,
            parity_path=parity_path,
            upstream_revalidated=upstream_revalidated,
        )
        _atomic_json(status, status_path)
        return {"status": status["status"], "blockers": blockers, "status_path": status_path}
    calculation_hash = _file_hash(calculation_path)
    inputs: list[dict[str, Any]] = []
    for input_name in REQUIRED_PARITY_INPUTS:
        row: dict[str, Any] = {"input": input_name}
        for environment in ("testnet", "live"):
            snapshot = {
                "schema_version": PARITY_ARTIFACT_SCHEMA_VERSION,
                "input": input_name,
                "environment": environment,
                "candidate_receipt_id": str(candidate["candidate_receipt_id"]),
                "captured_at_utc": as_of.isoformat(),
                "values": normalized[environment][input_name],
                "units": PARITY_INPUT_CONTRACTS[input_name],
                "calculation_artifact_path": _relative(calculation_path, root),
                "calculation_artifact_sha256": calculation_hash,
                "source_artifacts": _source_artifacts_for_input(
                    raw[environment], input_name=input_name, assets=assets
                ),
                "capture_complete": True,
                "private_key_accessed": False,
                "order_submission_performed": False,
                "live_trading_authorized": False,
            }
            snapshot["receipt_sha256"] = _payload_hash(snapshot)
            snapshot_path = snapshot_root / f"{input_name}-{environment}.json"
            _atomic_json(snapshot, snapshot_path)
            row[f"{environment}_timestamp_utc"] = as_of.isoformat()
            row[f"{environment}_artifact_path"] = _relative(snapshot_path, root)
            row[f"{environment}_artifact_sha256"] = _file_hash(snapshot_path)
        inputs.append(row)

    evidence_core = {
        "schema_version": PARITY_SCHEMA_VERSION,
        "captured_at_utc": as_of.isoformat(),
        "candidate_receipt_id": str(candidate["candidate_receipt_id"]),
        "live_canary_policy_id": _policy_id(policy),
        "capture_id": capture_id,
        "account_address_masked": _mask_address(address),
        "inputs": inputs,
        "read_only_info_requests": True,
        "private_key_accessed": False,
        "order_submission_performed": False,
        "live_trading_authorized": False,
        "stage6_upstream_revalidated": upstream_revalidated,
    }
    evidence = {
        **evidence_core,
        "parity_receipt_id": "liveinputparity_" + _payload_hash(evidence_core)[:20],
    }
    evidence["receipt_sha256"] = _payload_hash(evidence)
    _atomic_json(evidence, evidence_path)
    parity = _parity_rows(
        root=root,
        evidence=evidence,
        candidate=candidate,
        policy=policy,
        policy_id=_policy_id(policy),
        as_of=as_of,
        upstream_ready=True,
    )
    _atomic_csv(parity, parity_path)
    parity_pass = bool(not parity.empty and parity["status"].eq("PASS").all())
    status = _capture_status(
        generated_at=as_of,
        status="PASS" if parity_pass else "BLOCKED_SEMANTIC_VALIDATION",
        blockers=(
            []
            if parity_pass
            else list(parity.loc[parity["status"].ne("PASS"), "blocker"].astype(str))
        ),
        evidence_path=evidence_path,
        parity_path=parity_path,
        upstream_revalidated=upstream_revalidated,
    )
    status.update(
        {
            "capture_id": capture_id,
            "parity_receipt_id": evidence["parity_receipt_id"],
            "raw_artifact_count": sum(len(items) for items in raw.values()),
            "snapshot_artifact_count": len(inputs) * 2,
        }
    )
    status["receipt_sha256"] = _payload_hash(status)
    _atomic_json(status, status_path)
    return {
        "status": status["status"],
        "blockers": status["blockers"],
        "status_path": status_path,
        "evidence_path": evidence_path,
        "parity_path": parity_path,
        "raw_root": raw_root,
        "snapshot_root": snapshot_root,
        "order_submission_performed": False,
        "live_trading_authorized": False,
    }


def _capture_info_request(
    *,
    root: Path,
    raw_root: Path,
    environment: str,
    request_name: str,
    info_url: str,
    payload: dict[str, Any],
    session: requests.Session,
    captured_at: datetime,
    timeout: int,
) -> dict[str, Any]:
    expected_urls = {
        "testnet": HYPERLIQUID_TESTNET_INFO_URL,
        "live": HYPERLIQUID_LIVE_INFO_URL,
    }
    if environment not in expected_urls or info_url != expected_urls[environment]:
        raise ValueError("unsafe_hyperliquid_non_info_endpoint")
    if payload.get("type") not in ALLOWED_INFO_REQUEST_TYPES:
        raise ValueError("unsafe_hyperliquid_info_request_type")
    raw_path = raw_root / f"{environment}-{request_name}.json"
    raw_receipt: dict[str, Any] = {}

    def record_response(parsed: Any) -> str:
        raw_receipt.update(
            {
                "schema_version": "thewiz.hyperliquid_readonly_parity_raw.v1",
                "environment": environment,
                "info_url": info_url,
                "request": payload,
                "captured_at_utc": captured_at.isoformat(),
                "response": parsed,
                "read_only_info_request": True,
                "private_key_accessed": False,
                "order_submission_performed": False,
            }
        )
        raw_receipt["receipt_sha256"] = _payload_hash(raw_receipt)
        _atomic_json(raw_receipt, raw_path)
        return _file_hash(raw_path)

    parsed = run_authorized_hyperliquid_info_call(
        target=info_url,
        payload=payload,
        operation_prefix=(
            "HYPERLIQUID_TESTNET"
            if environment == "testnet"
            else "HYPERLIQUID_MAINNET"
        ),
        transport=lambda: _raw_hyperliquid_parity_info_call(
            session=session,
            info_url=info_url,
            payload=payload,
            timeout=timeout,
        ),
        result_recorder=record_response,
    )
    return {
        "payload": parsed,
        "path": _relative(raw_path, root),
        "sha256": _file_hash(raw_path),
        "request_type": str(payload["type"]),
    }


def _raw_hyperliquid_parity_info_call(
    *,
    session: requests.Session,
    info_url: str,
    payload: dict[str, Any],
    timeout: int,
) -> Any:
    response = session.post(info_url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _normalize_environment(
    *,
    payloads: dict[str, dict[str, Any]],
    assets: tuple[str, str],
    captured_at: datetime,
    target_notional_usd: float,
) -> dict[str, dict[str, float]]:
    markets = _market_context(payloads["meta"]["payload"], assets=assets)
    mids_payload = payloads["mids"]["payload"]
    if not isinstance(mids_payload, dict):
        raise TypeError("hyperliquid_all_mids_response_invalid")
    mids = {asset: _positive_number(mids_payload.get(asset), f"missing_mid:{asset}") for asset in assets}
    books = {
        asset: _book_metrics(
            payloads[f"l2_{asset}"]["payload"],
            target_notional_usd=target_notional_usd,
        )
        for asset in assets
    }
    margin = _margin_metrics(payloads["margin"]["payload"], assets=assets)
    captured_ms = float(int(captured_at.timestamp() * 1000))
    x, y = assets
    return {
        "market_metadata": {
            "asset_x_size_decimals": float(markets[x]["size_decimals"]),
            "asset_y_size_decimals": float(markets[y]["size_decimals"]),
            "asset_x_max_leverage": float(markets[x]["max_leverage"]),
            "asset_y_max_leverage": float(markets[y]["max_leverage"]),
            "asset_x_minimum_notional": MINIMUM_ORDER_NOTIONAL_USD,
            "asset_y_minimum_notional": MINIMUM_ORDER_NOTIONAL_USD,
        },
        "mark_and_mid_prices": {
            "asset_x_mark": float(markets[x]["mark"]),
            "asset_y_mark": float(markets[y]["mark"]),
            "asset_x_mid": mids[x],
            "asset_y_mid": mids[y],
        },
        "funding_rate_and_timestamp": {
            "asset_x_funding_rate": float(markets[x]["funding"]),
            "asset_y_funding_rate": float(markets[y]["funding"]),
            "asset_x_funding_timestamp": captured_ms,
            "asset_y_funding_timestamp": captured_ms,
        },
        "l2_depth_and_slippage": {
            "asset_x_bid_depth": books[x]["bid_depth"],
            "asset_x_ask_depth": books[x]["ask_depth"],
            "asset_y_bid_depth": books[y]["bid_depth"],
            "asset_y_ask_depth": books[y]["ask_depth"],
            "asset_x_slippage": books[x]["worst_slippage_bps"],
            "asset_y_slippage": books[y]["worst_slippage_bps"],
        },
        "size_precision_and_minimum_notional": {
            "asset_x_size_decimals": float(markets[x]["size_decimals"]),
            "asset_y_size_decimals": float(markets[y]["size_decimals"]),
            "asset_x_minimum_notional": MINIMUM_ORDER_NOTIONAL_USD,
            "asset_y_minimum_notional": MINIMUM_ORDER_NOTIONAL_USD,
        },
        "margin_and_liquidation_inputs": margin,
    }


def _market_context(payload: Any, *, assets: tuple[str, str]) -> dict[str, dict[str, float]]:
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("hyperliquid_meta_context_response_invalid")
    metadata, contexts = payload[0], payload[1]
    universe = metadata.get("universe") if isinstance(metadata, dict) else None
    if not isinstance(universe, list) or not isinstance(contexts, list):
        raise TypeError("hyperliquid_meta_context_universe_invalid")
    rows: dict[str, dict[str, float]] = {}
    for market, context in zip(universe, contexts, strict=False):
        if not isinstance(market, dict) or not isinstance(context, dict):
            continue
        asset = str(market.get("name", "")).strip().upper()
        if asset not in assets or market.get("isDelisted") is True:
            continue
        rows[asset] = {
            "size_decimals": _nonnegative_number(
                market.get("szDecimals"), f"missing_size_decimals:{asset}"
            ),
            "max_leverage": _positive_number(
                market.get("maxLeverage"), f"missing_max_leverage:{asset}"
            ),
            "mark": _positive_number(context.get("markPx"), f"missing_mark:{asset}"),
            "funding": _finite_number(context.get("funding"), f"missing_funding:{asset}"),
        }
    if set(rows) != set(assets):
        raise ValueError("hyperliquid_candidate_markets_missing_or_delisted")
    return rows


def _book_metrics(payload: Any, *, target_notional_usd: float) -> dict[str, float]:
    if not isinstance(payload, dict) or not isinstance(payload.get("levels"), list):
        raise TypeError("hyperliquid_l2_book_response_invalid")
    levels = payload["levels"]
    if len(levels) < 2 or not isinstance(levels[0], list) or not isinstance(levels[1], list):
        raise ValueError("hyperliquid_l2_book_levels_invalid")
    bids = _normalized_levels(levels[0])
    asks = _normalized_levels(levels[1])
    if not bids or not asks:
        raise ValueError("hyperliquid_l2_book_empty_side")
    best_bid = max(price for price, _ in bids)
    best_ask = min(price for price, _ in asks)
    if best_bid <= 0.0 or best_ask <= best_bid:
        raise ValueError("hyperliquid_l2_book_crossed_or_invalid")
    mid = (best_bid + best_ask) / 2.0
    buy_average = _notional_weighted_price(sorted(asks), target_notional_usd)
    sell_average = _notional_weighted_price(
        sorted(bids, reverse=True), target_notional_usd
    )
    return {
        "bid_depth": sum(price * size for price, size in bids),
        "ask_depth": sum(price * size for price, size in asks),
        "worst_slippage_bps": max(
            0.0,
            (buy_average / mid - 1.0) * 10_000.0,
            (1.0 - sell_average / mid) * 10_000.0,
        ),
    }


def _margin_metrics(payload: Any, *, assets: tuple[str, str]) -> dict[str, float]:
    if not isinstance(payload, dict):
        raise TypeError("hyperliquid_clearinghouse_response_invalid")
    summary = payload.get("marginSummary")
    if not isinstance(summary, dict):
        raise TypeError("hyperliquid_margin_summary_missing")
    liquidation = {asset: 0.0 for asset in assets}
    positions = payload.get("assetPositions")
    if not isinstance(positions, list):
        raise TypeError("hyperliquid_asset_positions_missing")
    for row in positions:
        position = row.get("position") if isinstance(row, dict) else None
        if not isinstance(position, dict):
            continue
        asset = str(position.get("coin", "")).strip().upper()
        if asset in liquidation and position.get("liquidationPx") not in {None, ""}:
            liquidation[asset] = _nonnegative_number(
                position.get("liquidationPx"), f"invalid_liquidation_price:{asset}"
            )
    maintenance = payload.get("crossMaintenanceMarginUsed", 0.0)
    x, y = assets
    return {
        "account_value": _nonnegative_number(
            summary.get("accountValue"), "missing_account_value"
        ),
        "withdrawable": _nonnegative_number(payload.get("withdrawable"), "missing_withdrawable"),
        "initial_margin_requirement": _nonnegative_number(
            summary.get("totalMarginUsed"), "missing_initial_margin_requirement"
        ),
        "maintenance_margin_requirement": _nonnegative_number(
            maintenance, "missing_maintenance_margin_requirement"
        ),
        "asset_x_liquidation_price": liquidation[x],
        "asset_y_liquidation_price": liquidation[y],
    }


def _normalized_levels(rows: list[Any]) -> list[tuple[float, float]]:
    normalized: list[tuple[float, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        price = _positive_number(row.get("px"), "invalid_l2_price")
        size = _positive_number(row.get("sz"), "invalid_l2_size")
        normalized.append((price, size))
    return normalized


def _notional_weighted_price(levels: list[tuple[float, float]], target: float) -> float:
    if target <= 0.0:
        raise ValueError("target_notional_must_be_positive")
    remaining = target
    quantity = 0.0
    spent = 0.0
    for price, size in levels:
        available = price * size
        used_notional = min(remaining, available)
        used_quantity = used_notional / price
        quantity += used_quantity
        spent += used_notional
        remaining -= used_notional
        if remaining <= 1e-9:
            break
    if remaining > 1e-9 or quantity <= 0.0:
        raise ValueError("l2_depth_insufficient_for_live_canary_notional")
    return spent / quantity


def _source_artifacts_for_input(
    payloads: dict[str, dict[str, Any]],
    *,
    input_name: str,
    assets: tuple[str, str],
) -> dict[str, dict[str, str]]:
    names = {
        "market_metadata": ("meta",),
        "mark_and_mid_prices": ("meta", "mids"),
        "funding_rate_and_timestamp": ("meta",),
        "l2_depth_and_slippage": tuple(f"l2_{asset}" for asset in assets),
        "size_precision_and_minimum_notional": ("meta",),
        "margin_and_liquidation_inputs": ("margin",),
    }[input_name]
    return {
        name: {
            "path": str(payloads[name]["path"]),
            "sha256": str(payloads[name]["sha256"]),
            "request_type": str(payloads[name]["request_type"]),
        }
        for name in names
    }


def _capture_status(
    *,
    generated_at: datetime,
    status: str,
    blockers: list[str],
    evidence_path: Path,
    parity_path: Path,
    upstream_revalidated: bool,
) -> dict[str, Any]:
    payload = {
        "schema_version": "thewiz.live_input_parity_capture_status.v1",
        "generated_at_utc": generated_at.isoformat(),
        "status": status,
        "blockers": list(dict.fromkeys(blockers)),
        "evidence_path": str(evidence_path),
        "parity_path": str(parity_path),
        "stage6_upstream_revalidated": upstream_revalidated,
        "read_only_info_requests": True,
        "private_key_accessed": False,
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    payload["receipt_sha256"] = _payload_hash(payload)
    return payload


def _finite_number(value: Any, blocker: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(blocker) from exc
    if not math.isfinite(number):
        raise ValueError(blocker)
    return number


def _positive_number(value: Any, blocker: str) -> float:
    number = _finite_number(value, blocker)
    if number <= 0.0:
        raise ValueError(blocker)
    return number


def _nonnegative_number(value: Any, blocker: str) -> float:
    number = _finite_number(value, blocker)
    if number < 0.0:
        raise ValueError(blocker)
    return number


def _valid_address(value: str) -> bool:
    text = value.strip().lower()
    return bool(
        len(text) == 42
        and text.startswith("0x")
        and all(character in "0123456789abcdef" for character in text[2:])
    )


def _mask_address(value: str) -> str:
    return f"{value[:6]}...{value[-4:]}" if len(value) >= 12 else ""


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    promote_staged_file(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    promote_staged_file(temporary, path)


def _as_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
