"""Read-only readiness contract for a future Hyperliquid mainnet canary executor.

This module deliberately has no order-submission method.  It verifies the
mainnet account, agent, markets, and local signing path, then writes a sealed
receipt that the live-canary control plane can inspect.  Building the receipt
cannot grant trading authority.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import requests
from eth_account import Account
from eth_account.messages import encode_defunct

ROOT = Path(__file__).resolve().parents[3]
HYPERLIQUID_MAINNET_URL = "https://api.hyperliquid.xyz"
HYPERLIQUID_MAINNET_INFO_URL = f"{HYPERLIQUID_MAINNET_URL}/info"
PREFLIGHT_SCHEMA_VERSION = "thewiz.hyperliquid_live_canary_executor_preflight.v4"
PREFLIGHT_PATH = ROOT / "reports" / "active" / "hyperliquid_live_canary_executor_preflight.json"
ADDRESS_PATTERN = re.compile(r"^0x[a-fA-F0-9]{40}$")
FORBIDDEN_RAW_KEY_ENVS = (
    "HYPERLIQUID_LIVE_AGENT_PRIVATE_KEY",
    "HYPERLIQUID_AGENT_PRIVATE_KEY",
    "HYPERLIQUID_TESTNET_AGENT_PRIVATE_KEY",
)


@dataclass(frozen=True)
class HyperliquidLiveCanaryConfig:
    """Mainnet read-only preflight configuration with no private-key field."""

    network: str = "mainnet"
    base_url: str = HYPERLIQUID_MAINNET_URL
    master_address: str | None = None
    agent_address: str | None = None
    keychain_service: str | None = None

    @classmethod
    def from_env(cls) -> HyperliquidLiveCanaryConfig:
        return cls(
            network=os.getenv("HYPERLIQUID_LIVE_NETWORK", "mainnet").strip().lower() or "mainnet",
            base_url=os.getenv("HYPERLIQUID_LIVE_BASE_URL", HYPERLIQUID_MAINNET_URL)
            .strip()
            .rstrip("/"),
            master_address=_env_value("HYPERLIQUID_LIVE_MASTER_ADDRESS"),
            agent_address=_env_value("HYPERLIQUID_LIVE_AGENT_ADDRESS"),
            keychain_service=_env_value("HYPERLIQUID_LIVE_AGENT_KEYCHAIN_SERVICE"),
        )

    def configuration_blockers(self) -> list[str]:
        blockers: list[str] = []
        if self.network != "mainnet":
            blockers.append("live_canary_executor_network_not_mainnet")
        if self.base_url.rstrip("/") != HYPERLIQUID_MAINNET_URL:
            blockers.append("live_canary_executor_mainnet_endpoint_invalid")
        if not _valid_address(self.master_address):
            blockers.append("live_canary_executor_master_address_invalid")
        if not _valid_address(self.agent_address):
            blockers.append("live_canary_executor_agent_address_invalid")
        if not str(self.keychain_service or "").strip():
            blockers.append("live_canary_executor_keychain_service_missing")
        if any(os.getenv(name, "").strip() for name in FORBIDDEN_RAW_KEY_ENVS):
            blockers.append("live_canary_executor_raw_private_key_env_forbidden")
        return blockers


def read_live_agent_key_from_keychain(service: str, account: str) -> str:
    """Load the dedicated mainnet agent key without printing or persisting it."""

    result = subprocess.run(
        ["security", "find-generic-password", "-w", "-s", service, "-a", account],
        capture_output=True,
        check=False,
        text=True,
    )
    secret = result.stdout.strip()
    if result.returncode != 0 or not secret:
        raise ValueError("live_canary_executor_keychain_entry_missing")
    return secret


def build_live_canary_executor_preflight(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    candidate: dict[str, Any],
    policy_id: str,
    config: HyperliquidLiveCanaryConfig | None = None,
    info_client: Callable[[dict[str, Any]], Any] | None = None,
    keychain_reader: Callable[[str, str], str] | None = None,
    permit_credential_and_network_checks: bool = True,
) -> Path:
    """Write a sealed, read-only mainnet executor readiness receipt."""

    as_of = _as_utc(now or datetime.now(UTC))
    resolved = config or HyperliquidLiveCanaryConfig.from_env()
    from quant_platform.orchestration.corrective_live_canary_execution import (
        live_canary_executor_contract,
    )

    executor_contract = live_canary_executor_contract()
    blockers = list(resolved.configuration_blockers())
    if not permit_credential_and_network_checks:
        blockers.append("live_canary_executor_upstream_not_ready")
    sdk_available = importlib.util.find_spec("hyperliquid") is not None
    if not sdk_available:
        blockers.append("live_canary_executor_sdk_missing")

    payload: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "generated_at_utc": as_of.isoformat(),
        "candidate_receipt_id": str(candidate.get("candidate_receipt_id", "")),
        "candidate_receipt_sha256": str(candidate.get("receipt_sha256", "")),
        "live_canary_policy_id": policy_id,
        "pair": str(candidate.get("pair", "")),
        "asset_x": str(candidate.get("asset_x", "")).strip().upper(),
        "asset_y": str(candidate.get("asset_y", "")).strip().upper(),
        "network": resolved.network,
        "base_url": resolved.base_url.rstrip("/"),
        "exact_mainnet_endpoint": resolved.base_url.rstrip("/") == HYPERLIQUID_MAINNET_URL,
        "master_address_configured": _valid_address(resolved.master_address),
        "agent_address_configured": _valid_address(resolved.agent_address),
        "keychain_service_configured": bool(str(resolved.keychain_service or "").strip()),
        "credential_and_network_checks_permitted": bool(permit_credential_and_network_checks),
        "raw_private_key_env_absent": not any(
            os.getenv(name, "").strip() for name in FORBIDDEN_RAW_KEY_ENVS
        ),
        "sdk_available": sdk_available,
        "agent_key_present": False,
        "agent_key_matches_address": False,
        "local_signature_created": False,
        "agent_role_checked": False,
        "agent_authorized_for_master": False,
        "market_metadata_checked": False,
        "both_candidate_markets_tradable": False,
        "account_state_checked": False,
        "account_has_positive_value": False,
        "account_has_withdrawable_margin": False,
        "account_flat": False,
        "open_orders_checked": False,
        "no_open_orders": False,
        "read_only_info_requests": True,
        "manual_executor_control_contract_present": bool(
            executor_contract.get("submission_capable") is True
            and executor_contract.get("automatic_execution") is False
            and executor_contract.get("one_run_only") is True
            and executor_contract.get("atomic_approval_reservation") is True
            and executor_contract.get("entry_retry_allowed") is False
            and executor_contract.get("persistent_live_authority") is False
        ),
        "executor_contract_id": executor_contract["executor_contract_id"],
        "executor_source_sha256": executor_contract["executor_source_sha256"],
        "executor_implementation_bundle_sha256": executor_contract[
            "executor_implementation_bundle_sha256"
        ],
        "submission_capable_executor_contract_present": True,
        "order_submission_method_present": False,
        "private_key_persisted": False,
        "order_submission_performed": False,
        "canary_execution_authority": False,
        "live_trading_authorized": False,
    }

    wallet = None
    if (
        permit_credential_and_network_checks
        and not blockers
        and resolved.keychain_service
        and resolved.agent_address
    ):
        try:
            reader = keychain_reader or read_live_agent_key_from_keychain
            secret = reader(resolved.keychain_service, resolved.agent_address)
            payload["agent_key_present"] = bool(secret)
            wallet = Account.from_key(secret)
            payload["agent_key_matches_address"] = (
                wallet.address.lower() == resolved.agent_address.lower()
            )
            if not payload["agent_key_matches_address"]:
                blockers.append("live_canary_executor_agent_key_address_mismatch")
        except Exception:  # noqa: BLE001 - secret-store adapters must fail closed
            blockers.append("live_canary_executor_keychain_entry_missing")

    if wallet is not None and not blockers:
        try:
            Account.sign_message(
                encode_defunct(
                    text=(
                        "TheWiz Hyperliquid mainnet no-order preflight:"
                        f"{payload['candidate_receipt_id']}:{as_of.isoformat()}"
                    )
                ),
                wallet.key,
            )
            payload["local_signature_created"] = True
        except Exception:  # noqa: BLE001 - signing adapters must fail closed
            blockers.append("live_canary_executor_local_signature_failed")

    if not blockers and resolved.master_address and resolved.agent_address:
        fetch = info_client or _default_info_client(resolved.base_url)
        try:
            role = fetch({"type": "userRole", "user": resolved.agent_address})
            payload["agent_role_checked"] = True
            role_name = str((role or {}).get("role", "")) if isinstance(role, dict) else ""
            linked_master = (
                str(((role or {}).get("data") or {}).get("user", ""))
                if isinstance(role, dict)
                else ""
            )
            payload["agent_authorized_for_master"] = bool(
                role_name == "agent" and linked_master.lower() == resolved.master_address.lower()
            )
            if not payload["agent_authorized_for_master"]:
                blockers.append("live_canary_executor_agent_authorization_unverified")
        except Exception:  # noqa: BLE001 - external read-only clients must fail closed
            blockers.append("live_canary_executor_agent_role_check_failed")

        try:
            meta = fetch({"type": "meta"})
            universe = meta.get("universe", []) if isinstance(meta, dict) else []
            markets = {
                str(row.get("name", "")).strip().upper()
                for row in universe
                if isinstance(row, dict) and not bool(row.get("isDelisted", False))
            }
            payload["market_metadata_checked"] = True
            assets = {payload["asset_x"], payload["asset_y"]} - {""}
            payload["both_candidate_markets_tradable"] = len(assets) == 2 and assets <= markets
            if not payload["both_candidate_markets_tradable"]:
                blockers.append("live_canary_executor_candidate_markets_unavailable")
        except Exception:  # noqa: BLE001 - external read-only clients must fail closed
            blockers.append("live_canary_executor_market_metadata_check_failed")

        try:
            state = fetch({"type": "clearinghouseState", "user": resolved.master_address})
            payload["account_state_checked"] = True
            margin = state.get("marginSummary", {}) if isinstance(state, dict) else {}
            account_value = _number(margin.get("accountValue")) or 0.0
            withdrawable = _number(state.get("withdrawable")) or 0.0
            positions = state.get("assetPositions", []) if isinstance(state, dict) else []
            payload["account_has_positive_value"] = account_value > 0.0
            payload["account_has_withdrawable_margin"] = withdrawable > 0.0
            payload["account_flat"] = not any(abs(_position_size(row)) > 1e-12 for row in positions)
            if not payload["account_has_positive_value"]:
                blockers.append("live_canary_executor_account_value_not_positive")
            if not payload["account_has_withdrawable_margin"]:
                blockers.append("live_canary_executor_withdrawable_margin_not_positive")
            if not payload["account_flat"]:
                blockers.append("live_canary_executor_account_not_flat")
        except Exception:  # noqa: BLE001 - external read-only clients must fail closed
            blockers.append("live_canary_executor_account_state_check_failed")

        try:
            orders = fetch({"type": "openOrders", "user": resolved.master_address})
            payload["open_orders_checked"] = True
            payload["no_open_orders"] = isinstance(orders, list) and not orders
            if not payload["no_open_orders"]:
                blockers.append("live_canary_executor_open_orders_present")
        except Exception:  # noqa: BLE001 - external read-only clients must fail closed
            blockers.append("live_canary_executor_open_orders_check_failed")

    blockers = sorted(set(blockers))
    readiness_checks = (
        "exact_mainnet_endpoint",
        "master_address_configured",
        "agent_address_configured",
        "keychain_service_configured",
        "credential_and_network_checks_permitted",
        "raw_private_key_env_absent",
        "sdk_available",
        "agent_key_present",
        "agent_key_matches_address",
        "local_signature_created",
        "agent_role_checked",
        "agent_authorized_for_master",
        "market_metadata_checked",
        "both_candidate_markets_tradable",
        "account_state_checked",
        "account_has_positive_value",
        "account_has_withdrawable_margin",
        "account_flat",
        "open_orders_checked",
        "no_open_orders",
        "manual_executor_control_contract_present",
        "submission_capable_executor_contract_present",
    )
    ready = not blockers and all(payload.get(field) is True for field in readiness_checks)
    payload["preflight_status"] = (
        "PASS_READ_ONLY"
        if ready
        else ("BLOCKED_UPSTREAM" if not permit_credential_and_network_checks else "BLOCKED")
    )
    payload["blockers"] = blockers
    payload["next_action"] = (
        "obtain_exact_signed_one_run_authorization"
        if ready
        else (
            "complete_stage6_evidence_before_live_preflight"
            if not permit_credential_and_network_checks
            else "resolve_live_canary_executor_preflight_blockers"
        )
    )
    payload["receipt_sha256"] = _payload_hash(payload)
    path = root / "reports" / "active" / PREFLIGHT_PATH.name
    _atomic_json(payload, path)
    return path


def validate_live_canary_executor_preflight(
    *,
    receipt: dict[str, Any],
    candidate: dict[str, Any],
    policy_id: str,
    as_of: datetime,
    maximum_age_seconds: float,
) -> tuple[bool, list[str]]:
    """Validate the sealed readiness receipt without treating it as authority."""

    blockers: list[str] = []
    from quant_platform.orchestration.corrective_live_canary_execution import (
        live_canary_executor_contract,
    )

    current_contract = live_canary_executor_contract()
    if receipt.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        blockers.append("live_canary_executor_preflight_schema_invalid")
    if receipt.get("receipt_sha256") != _payload_hash(receipt):
        blockers.append("live_canary_executor_preflight_hash_invalid")
    expected = {
        "candidate_receipt_id": candidate.get("candidate_receipt_id", ""),
        "candidate_receipt_sha256": candidate.get("receipt_sha256", ""),
        "live_canary_policy_id": policy_id,
        "pair": candidate.get("pair", ""),
        "asset_x": str(candidate.get("asset_x", "")).strip().upper(),
        "asset_y": str(candidate.get("asset_y", "")).strip().upper(),
    }
    if any(receipt.get(field) != value for field, value in expected.items()):
        blockers.append("live_canary_executor_preflight_binding_mismatch")
    if any(
        receipt.get(field) != current_contract.get(field)
        for field in (
            "executor_contract_id",
            "executor_source_sha256",
            "executor_implementation_bundle_sha256",
        )
    ):
        blockers.append("live_canary_executor_preflight_contract_binding_mismatch")
    generated = _timestamp(receipt.get("generated_at_utc"))
    age = (_as_utc(as_of) - generated).total_seconds() if generated else math.inf
    if generated is None or age < 0.0 or age > maximum_age_seconds:
        blockers.append("live_canary_executor_preflight_stale_or_future")
    required_true = (
        "exact_mainnet_endpoint",
        "master_address_configured",
        "agent_address_configured",
        "keychain_service_configured",
        "credential_and_network_checks_permitted",
        "raw_private_key_env_absent",
        "sdk_available",
        "agent_key_present",
        "agent_key_matches_address",
        "local_signature_created",
        "agent_role_checked",
        "agent_authorized_for_master",
        "market_metadata_checked",
        "both_candidate_markets_tradable",
        "account_state_checked",
        "account_has_positive_value",
        "account_has_withdrawable_margin",
        "account_flat",
        "open_orders_checked",
        "no_open_orders",
        "read_only_info_requests",
        "manual_executor_control_contract_present",
        "submission_capable_executor_contract_present",
    )
    if receipt.get("preflight_status") != "PASS_READ_ONLY" or not all(
        receipt.get(field) is True for field in required_true
    ):
        blockers.append("live_canary_executor_read_only_preflight_not_passed")
    if (
        receipt.get("order_submission_method_present") is not False
        or receipt.get("private_key_persisted") is not False
        or receipt.get("order_submission_performed") is not False
        or receipt.get("canary_execution_authority") is not False
        or receipt.get("live_trading_authorized") is not False
    ):
        blockers.append("live_canary_executor_preflight_claims_execution_or_authority")
    return not blockers, sorted(set(blockers))


def _default_info_client(base_url: str) -> Callable[[dict[str, Any]], Any]:
    session = requests.Session()

    def fetch(payload: dict[str, Any]) -> Any:
        response = session.post(f"{base_url.rstrip('/')}/info", json=payload, timeout=20)
        response.raise_for_status()
        return response.json()

    return fetch


def _position_size(row: Any) -> float:
    if not isinstance(row, dict):
        return 0.0
    position = row.get("position") if isinstance(row.get("position"), dict) else row
    return _number(position.get("szi")) or 0.0


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _as_utc(parsed)


def _valid_address(value: str | None) -> bool:
    return bool(value and ADDRESS_PATTERN.fullmatch(value.strip()))


def _env_value(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _payload_hash(payload: dict[str, Any]) -> str:
    core = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return sha256(_canonical_json(core).encode("utf-8")).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
