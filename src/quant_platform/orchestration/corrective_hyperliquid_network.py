"""Single permit-bound gateway for Hyperliquid public `/info` reads."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from hashlib import sha256
from typing import Any, TypeVar

from quant_platform.orchestration.corrective_external_effects import (
    run_authorized_public_call,
)
from quant_platform.orchestration.effect_authority import EffectAuthorityError

T = TypeVar("T")

HYPERLIQUID_INFO_OPERATION_SUFFIXES = {
    "allMids": "ALL_MIDS",
    "candleSnapshot": "CANDLE_SNAPSHOT",
    "clearinghouseState": "CLEARINGHOUSE_STATE",
    "fundingHistory": "FUNDING_HISTORY",
    "l2Book": "L2_BOOK",
    "meta": "META",
    "metaAndAssetCtxs": "META_AND_ASSET_CONTEXTS",
    "openOrders": "OPEN_ORDERS",
    "spotClearinghouseState": "SPOT_CLEARINGHOUSE_STATE",
    "userFillsByTime": "USER_FILLS_BY_TIME",
    "userFunding": "USER_FUNDING",
    "userRole": "USER_ROLE",
}
HYPERLIQUID_INFO_OPERATION_PREFIXES = frozenset(
    {"HYPERLIQUID_MAINNET", "HYPERLIQUID_TESTNET"}
)


def hyperliquid_info_operation(
    payload: dict[str, object],
    *,
    operation_prefix: str,
) -> str:
    """Return the exact allowlisted operation name before any transport call."""

    normalized_prefix = operation_prefix.strip().upper()
    if normalized_prefix not in HYPERLIQUID_INFO_OPERATION_PREFIXES:
        raise EffectAuthorityError("hyperliquid_info_operation_prefix_denied")
    request_type = str(payload.get("type", "")).strip()
    suffix = HYPERLIQUID_INFO_OPERATION_SUFFIXES.get(request_type)
    if suffix is None:
        raise EffectAuthorityError("hyperliquid_info_request_type_denied")
    return f"{normalized_prefix}_{suffix}"


def run_authorized_hyperliquid_info_call(
    *,
    target: str,
    payload: dict[str, object],
    operation_prefix: str,
    transport: Callable[[], T],
    result_recorder: Callable[[T], str] | None = None,
) -> T:
    """Consume one public-network permit around an injected raw transport."""

    operation = hyperliquid_info_operation(
        payload,
        operation_prefix=operation_prefix,
    )
    request_payload = _canonical_json(payload).encode("utf-8")
    return run_authorized_public_call(
        target=target,
        operation=operation,
        method="POST",
        request_payload=request_payload,
        request_count=1,
        callback=transport,
        result_recorder=result_recorder or hyperliquid_info_response_sha256,
    )


def hyperliquid_info_response_sha256(value: Any) -> str:
    """Hash one JSON-compatible response; non-finite data fails the effect."""

    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def hyperliquid_info_contract_operations(
    *,
    operation_prefix: str,
) -> frozenset[str]:
    """Expose the complete exact operation set for policy and test fixtures."""

    normalized_prefix = operation_prefix.strip().upper()
    if normalized_prefix not in HYPERLIQUID_INFO_OPERATION_PREFIXES:
        raise EffectAuthorityError("hyperliquid_info_operation_prefix_denied")
    return frozenset(
        f"{normalized_prefix}_{suffix}"
        for suffix in HYPERLIQUID_INFO_OPERATION_SUFFIXES.values()
    )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EffectAuthorityError("hyperliquid_info_payload_not_canonical_json") from exc


def is_private_hyperliquid_info_transport_name(name: str) -> bool:
    """Static-audit helper for the only acceptable raw transport naming form."""

    return bool(re.fullmatch(r"_raw_hyperliquid(?:_[a-z0-9]+)*_info(?:_call|_client)", name))
