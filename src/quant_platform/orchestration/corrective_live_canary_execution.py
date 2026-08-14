"""Fail-closed, one-use Hyperliquid mainnet canary executor.

The default path is status-only.  A submission path exists only for one exact
wallet-approved canary and remains bounded by immutable evidence, a fresh
read-only preflight, atomic approval reservation, IOC orders, and recovery to
flat.  It never grants persistent live-trading authority.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from typing import Any

import requests
from eth_account import Account
from eth_account.messages import encode_defunct

from quant_platform.orchestration.corrective_live_canary_executor import (
    HyperliquidLiveCanaryConfig,
    read_live_agent_key_from_keychain,
    validate_live_canary_executor_preflight,
)
from quant_platform.orchestration.corrective_live_canary_outcome import (
    EXECUTION_SCHEMA_VERSION,
    _identity,
)

ROOT = Path(__file__).resolve().parents[3]
AUTHORIZATION_SCHEMA_VERSION = "thewiz.live_canary_authorization.v5"
EXECUTOR_STATUS_SCHEMA_VERSION = "thewiz.live_canary_executor_status.v2"
EXECUTOR_RESERVATION_SCHEMA_VERSION = "thewiz.live_canary_reservation.v2"
EXECUTOR_INCIDENT_SCHEMA_VERSION = "thewiz.live_canary_executor_incident.v1"
EXECUTOR_CONTRACT_VERSION = "hyperliquid-mainnet-one-canary-v2"
APPROVAL_SCHEMA_VERSION = "thewiz.live_canary_user_approval.v3"
EXECUTOR_IMPLEMENTATION_RELATIVE_PATHS = (
    Path("src/quant_platform/orchestration/corrective_live_canary.py"),
    Path("src/quant_platform/orchestration/corrective_live_canary_execution.py"),
    Path("src/quant_platform/orchestration/corrective_live_canary_executor.py"),
    Path("src/quant_platform/orchestration/corrective_live_canary_outcome.py"),
)
LIVE_ENABLE_ENV = "QPA_ENABLE_HYPERLIQUID_LIVE_CANARY"
LIVE_ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_ONE_LIVE_HYPERLIQUID_CANARY"
STATUS_PATH = ROOT / "reports" / "active" / "live_canary_executor_status.json"
STATE_PATH = ROOT / "reports" / "active" / "live_canary_executor_state.json"
INCIDENT_PATH = ROOT / "reports" / "active" / "live_canary_executor_incident.json"
EXECUTION_PATH = ROOT / "reports" / "active" / "live_canary_execution_receipt.json"
LEDGER_PATH = ROOT / "data" / "live" / "canary_authorization_use_ledger.jsonl"
LOCK_PATH = ROOT / "data" / "live" / ".live_canary_executor.lock"


@dataclass(frozen=True)
class LiveCanaryExecutorResult:
    status: str
    blockers: tuple[str, ...]
    order_submission_performed: bool = False
    reconciled_flat: bool = False
    execution_id: str = ""
    status_path: str = ""
    incident_path: str = ""
    live_trading_authorized: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "blockers": list(self.blockers),
            "order_submission_performed": self.order_submission_performed,
            "reconciled_flat": self.reconciled_flat,
            "execution_id": self.execution_id,
            "status_path": self.status_path,
            "incident_path": self.incident_path,
            "live_trading_authorized": False,
        }


def live_canary_executor_contract() -> dict[str, Any]:
    """Return the exact implementation-bound contract used by authorization."""

    implementation_bundle = {
        path.as_posix(): sha256((ROOT / path).read_bytes()).hexdigest()
        for path in sorted(EXECUTOR_IMPLEMENTATION_RELATIVE_PATHS, key=lambda item: item.as_posix())
    }
    implementation_bundle_sha256 = sha256(
        _canonical_json(implementation_bundle).encode("utf-8")
    ).hexdigest()
    source_path = Path(
        "src/quant_platform/orchestration/corrective_live_canary_execution.py"
    )
    contract = {
        "executor_contract_version": EXECUTOR_CONTRACT_VERSION,
        "executor_source_path": source_path.as_posix(),
        "executor_source_sha256": implementation_bundle[source_path.as_posix()],
        "executor_implementation_bundle": implementation_bundle,
        "executor_implementation_bundle_sha256": implementation_bundle_sha256,
        "submission_capable": True,
        "automatic_execution": False,
        "one_run_only": True,
        "atomic_approval_reservation": True,
        "entry_time_in_force": "IOC",
        "exit_reduce_only": True,
        "entry_retry_allowed": False,
        "persistent_live_authority": False,
    }
    contract["executor_contract_id"] = (
        "livecanaryexecutor_" + _payload_hash(contract)[:20]
    )
    return contract


def run_live_canary_executor(
    *,
    root: Path = ROOT,
    execute: bool = False,
    recover_only: bool = False,
    authorization_sha256: str = "",
    approval_id: str = "",
    acknowledgement: str = "",
    now: datetime | None = None,
    config: HyperliquidLiveCanaryConfig | None = None,
    info_client: Callable[[dict[str, Any]], Any] | None = None,
    keychain_reader: Callable[[str, str], str] | None = None,
    exchange_factory: Callable[[Any, HyperliquidLiveCanaryConfig], Any] | None = None,
    waiter: Callable[[float], None] | None = None,
) -> LiveCanaryExecutorResult:
    """Validate or execute one exact canary; default invocation cannot submit."""

    as_of = _as_utc(now or _utc_now())
    execution_time_override = bool(execute and not recover_only and now is not None)
    artifacts = (
        _load_recovery_artifacts(root, approval_id)
        if recover_only
        else _load_artifacts(root)
    )
    resolved = config or HyperliquidLiveCanaryConfig.from_env()
    if recover_only:
        local_blockers = _recovery_local_blockers(
            root=root,
            artifacts=artifacts,
            authorization_sha256=authorization_sha256,
            approval_id=approval_id,
            acknowledgement=acknowledgement,
            require_execution=execute,
            config=resolved,
        )
    else:
        local_blockers = _local_execution_blockers(
            root=root,
            artifacts=artifacts,
            as_of=as_of,
            authorization_sha256=authorization_sha256,
            approval_id=approval_id,
            acknowledgement=acknowledgement,
            require_execution=execute,
            config=resolved,
        )
        if execution_time_override:
            local_blockers.append("live_canary_execution_time_override_forbidden")
    if not execute:
        blockers = _unique(["live_canary_execute_flag_not_set", *local_blockers])
        return _status_result(
            root=root,
            as_of=as_of,
            status="NO_SUBMISSION_STATUS_ONLY",
            blockers=blockers,
        )
    if recover_only and not _reservation_path(root, approval_id).is_file():
        local_blockers.append("live_canary_recovery_reservation_missing")
    if local_blockers:
        return _status_result(
            root=root,
            as_of=as_of,
            status="BLOCKED_BEFORE_KEY_ACCESS",
            blockers=_unique(local_blockers),
        )

    fetch = info_client or _default_info_client(resolved.base_url)
    runtime, runtime_blockers = _runtime_pre_reservation_checks(
        artifacts=artifacts,
        config=resolved,
        fetch=fetch,
    )
    if recover_only:
        runtime_blockers = _recovery_runtime_blockers(runtime_blockers)
    if runtime_blockers:
        return _status_result(
            root=root,
            as_of=as_of,
            status="BLOCKED_BEFORE_APPROVAL_RESERVATION",
            blockers=runtime_blockers,
        )

    with _exclusive_executor_lock(root) as lock_acquired:
        if not lock_acquired:
            return _status_result(
                root=root,
                as_of=as_of,
                status="BLOCKED_EXECUTOR_LOCK_PRESENT",
                blockers=["live_canary_executor_lock_present"],
            )
        artifacts = (
            _load_recovery_artifacts(root, approval_id)
            if recover_only
            else _load_artifacts(root)
        )
        if not recover_only:
            as_of = _as_utc(_utc_now())
        if recover_only:
            recheck = _recovery_local_blockers(
                root=root,
                artifacts=artifacts,
                authorization_sha256=authorization_sha256,
                approval_id=approval_id,
                acknowledgement=acknowledgement,
                require_execution=True,
                config=resolved,
            )
        else:
            recheck = _local_execution_blockers(
                root=root,
                artifacts=artifacts,
                as_of=as_of,
                authorization_sha256=authorization_sha256,
                approval_id=approval_id,
                acknowledgement=acknowledgement,
                require_execution=True,
                config=resolved,
            )
        if recheck:
            return _status_result(
                root=root,
                as_of=as_of,
                status="BLOCKED_DURING_ATOMIC_RECHECK",
                blockers=recheck,
            )
        runtime, runtime_blockers = _runtime_pre_reservation_checks(
            artifacts=artifacts,
            config=resolved,
            fetch=fetch,
        )
        if runtime_blockers and not recover_only:
            return _status_result(
                root=root,
                as_of=as_of,
                status="BLOCKED_DURING_RUNTIME_RECHECK",
                blockers=runtime_blockers,
            )
        if recover_only:
            return _recover_reserved_canary(
                root=root,
                as_of=as_of,
                artifacts=artifacts,
                config=resolved,
                fetch=fetch,
                keychain_reader=keychain_reader,
                exchange_factory=exchange_factory,
            )

        reservation = _reserve_approval(
            root=root,
            as_of=as_of,
            authorization_sha256=authorization_sha256,
            approval_id=approval_id,
            artifacts=artifacts,
            config=resolved,
        )
        started = datetime.now(UTC)
        state = {
            "schema_version": "thewiz.live_canary_executor_state.v1",
            "phase": "APPROVAL_RESERVED",
            "started_at_utc": started.isoformat(),
            "approval_id": approval_id,
            "reservation_id": reservation["reservation_id"],
            "authorization_receipt_sha256": authorization_sha256,
            "master_address": resolved.master_address,
            "agent_address": resolved.agent_address,
            "entry_submit_attempted": False,
            "entry_retry_allowed": False,
            "exit_submit_attempted": False,
            "recovery_attempted": False,
            "live_trading_authorized": False,
        }
        _write_state(root, state)
        try:
            wallet = _load_wallet(resolved, keychain_reader)
            exchange = _build_exchange(wallet, resolved, exchange_factory)
            for leg in artifacts["approval"]["legs"]:
                exchange.update_leverage(1, _market(leg), is_cross=True)
            state["phase"] = "ENTRY_SUBMITTING_UNCONFIRMED"
            state["entry_submit_attempted"] = True
            _write_state(root, state)
            entry_response = exchange.bulk_orders(
                [_entry_order_request(leg) for leg in artifacts["approval"]["legs"]]
            )
            state["entry_response_summary"] = _response_summary(entry_response)
            _write_state(root, state)
        except Exception as exc:  # noqa: BLE001 - ambiguous submission must recover
            return _recover_after_anomaly(
                root=root,
                as_of=as_of,
                artifacts=artifacts,
                config=resolved,
                fetch=fetch,
                exchange=locals().get("exchange"),
                state=state,
                incident=f"live_canary_entry_submission_ambiguous:{type(exc).__name__}",
            )

        sleep = waiter or time.sleep
        expected = _expected_entry_positions(artifacts["approval"]["legs"])
        positions = _wait_for_positions(
            fetch=fetch,
            master_address=str(resolved.master_address),
            markets=set(expected),
            expected=expected,
            waiter=sleep,
        )
        if not _positions_match(positions, expected):
            return _recover_after_anomaly(
                root=root,
                as_of=as_of,
                artifacts=artifacts,
                config=resolved,
                fetch=fetch,
                exchange=exchange,
                state=state,
                incident="live_canary_entry_not_exactly_filled_no_retry",
            )

        try:
            fresh_mids = _positive_mids(fetch, set(expected))
            max_slippage = float(artifacts["approval"]["maximum_slippage_bps"])
            meta = _market_meta(fetch)
            exit_requests = _exit_order_requests(
                legs=artifacts["approval"]["legs"],
                mids=fresh_mids,
                market_meta=meta,
                maximum_slippage_bps=max_slippage,
            )
            state["phase"] = "EXIT_SUBMITTING_UNCONFIRMED"
            state["exit_submit_attempted"] = True
            _write_state(root, state)
            exit_response = exchange.bulk_orders(exit_requests)
            state["exit_response_summary"] = _response_summary(exit_response)
            _write_state(root, state)
        except Exception as exc:  # noqa: BLE001 - risk reduction must continue
            return _recover_after_anomaly(
                root=root,
                as_of=as_of,
                artifacts=artifacts,
                config=resolved,
                fetch=fetch,
                exchange=exchange,
                state=state,
                incident=f"live_canary_exit_submission_ambiguous:{type(exc).__name__}",
            )

        final_positions = _wait_for_positions(
            fetch=fetch,
            master_address=str(resolved.master_address),
            markets=set(expected),
            expected={market: 0.0 for market in expected},
            waiter=sleep,
        )
        open_orders = _pair_open_orders(
            fetch({"type": "openOrders", "user": resolved.master_address}),
            set(expected),
        )
        if not _positions_match(final_positions, {market: 0.0 for market in expected}) or open_orders:
            return _recover_after_anomaly(
                root=root,
                as_of=as_of,
                artifacts=artifacts,
                config=resolved,
                fetch=fetch,
                exchange=exchange,
                state=state,
                incident="live_canary_exit_not_flat_or_order_free",
            )
        return _finalize_success(
            root=root,
            started=started,
            artifacts=artifacts,
            config=resolved,
            fetch=fetch,
            state=state,
            authorization_sha256=authorization_sha256,
            final_positions=final_positions,
        )


def _load_artifacts(root: Path) -> dict[str, Any]:
    active = root / "reports" / "active"
    return {
        "authorization_path": active / "live_canary_authorization.json",
        "authorization": _read_json(active / "live_canary_authorization.json"),
        "approval_path": active / "live_canary_user_approval.json",
        "approval": _read_json(active / "live_canary_user_approval.json"),
        "candidate_path": active / "testnet_candidate_receipt.json",
        "candidate": _read_json(active / "testnet_candidate_receipt.json"),
        "preflight_path": active / "hyperliquid_live_canary_executor_preflight.json",
        "preflight": _read_json(active / "hyperliquid_live_canary_executor_preflight.json"),
        "policy_path": root / "config" / "live_canary_policy.json",
        "policy": _read_json(root / "config" / "live_canary_policy.json"),
    }


def _load_recovery_artifacts(root: Path, approval_id: str) -> dict[str, Any]:
    artifacts = _load_artifacts(root)
    reservation = _read_json(_reservation_path(root, approval_id))
    frozen_path = _safe_root_path(
        root, str(reservation.get("authorization_receipt_path", ""))
    )
    if frozen_path is not None and frozen_path.is_file():
        artifacts["authorization_path"] = frozen_path
        artifacts["authorization"] = _read_json(frozen_path)
    artifacts["reservation"] = reservation
    artifacts["reservation_path"] = _reservation_path(root, approval_id)
    return artifacts


def _recovery_local_blockers(
    *,
    root: Path,
    artifacts: dict[str, Any],
    authorization_sha256: str,
    approval_id: str,
    acknowledgement: str,
    require_execution: bool,
    config: HyperliquidLiveCanaryConfig,
) -> list[str]:
    blockers = list(config.configuration_blockers())
    reservation = artifacts.get("reservation") or {}
    reservation_path = artifacts.get("reservation_path")
    if not (
        isinstance(reservation_path, Path)
        and reservation_path.is_file()
        and reservation.get("schema_version") == EXECUTOR_RESERVATION_SCHEMA_VERSION
        and reservation.get("receipt_sha256") == _payload_hash(reservation)
        and reservation.get("approval_id") == approval_id
        and reservation.get("authorization_receipt_sha256")
        == authorization_sha256
        and reservation.get("authorization_receipt_sha256")
        == _file_hash(artifacts["authorization_path"])
        and reservation.get("authorization_id")
        == artifacts["authorization"].get("authorization_id")
        and reservation.get("executor_contract_id")
        == live_canary_executor_contract()["executor_contract_id"]
        and reservation.get("executor_implementation_bundle_sha256")
        == live_canary_executor_contract()[
            "executor_implementation_bundle_sha256"
        ]
        and str(reservation.get("master_address", "")).lower()
        == str(config.master_address).lower()
        and str(reservation.get("agent_address", "")).lower()
        == str(config.agent_address).lower()
        and reservation.get("entry_retry_allowed") is False
        and reservation.get("authorization_reusable") is False
        and reservation.get("live_trading_authorized") is False
    ):
        blockers.append("live_canary_recovery_reservation_invalid_or_mismatched")
    state = _read_json(root / "reports" / "active" / STATE_PATH.name)
    if state and (
        state.get("receipt_sha256") != _payload_hash(state)
        or state.get("approval_id") != approval_id
        or state.get("authorization_receipt_sha256") != authorization_sha256
        or state.get("entry_retry_allowed") is not False
    ):
        blockers.append("live_canary_recovery_state_invalid_or_mismatched")
    if (root / "reports" / "active" / EXECUTION_PATH.name).is_file():
        blockers.append("live_canary_recovery_blocked_after_complete_execution_receipt")
    if require_execution:
        if os.getenv(LIVE_ENABLE_ENV, "").strip().lower() != "true":
            blockers.append("live_canary_environment_enable_missing")
        if acknowledgement != LIVE_ACKNOWLEDGEMENT:
            blockers.append("live_canary_exact_acknowledgement_missing")
    return _unique(blockers)


def _recovery_runtime_blockers(blockers: list[str]) -> list[str]:
    required = (
        "live_canary_runtime_agent_authorization_failed",
        "live_canary_runtime_market_unavailable",
        "live_canary_runtime_read_failed:",
    )
    return [blocker for blocker in blockers if blocker.startswith(required)]


def _local_execution_blockers(
    *,
    root: Path,
    artifacts: dict[str, Any],
    as_of: datetime,
    authorization_sha256: str,
    approval_id: str,
    acknowledgement: str,
    require_execution: bool,
    config: HyperliquidLiveCanaryConfig,
    permit_existing_reservation: bool = False,
) -> list[str]:
    blockers: list[str] = []
    auth = artifacts["authorization"]
    approval = artifacts["approval"]
    candidate = artifacts["candidate"]
    policy = artifacts["policy"]
    preflight = artifacts["preflight"]
    contract = live_canary_executor_contract()
    auth_path = artifacts["authorization_path"]
    approval_path = artifacts["approval_path"]
    if not all(path.is_file() for path in (auth_path, approval_path, artifacts["candidate_path"], artifacts["policy_path"], artifacts["preflight_path"])):
        blockers.append("live_canary_required_artifact_missing")
    actual_auth_hash = _file_hash(auth_path)
    if not authorization_sha256 or authorization_sha256 != actual_auth_hash:
        blockers.append("live_canary_authorization_hash_acknowledgement_mismatch")
    if auth.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION or auth.get(
        "receipt_sha256"
    ) != _payload_hash(auth):
        blockers.append("live_canary_authorization_receipt_invalid")
    immutable_auth_path = _safe_root_path(
        root, str(auth.get("immutable_authorization_path", ""))
    )
    if not (
        str(auth.get("authorization_id", "")).startswith("livecanaryauth_")
        and immutable_auth_path is not None
        and immutable_auth_path.is_file()
        and _read_json(immutable_auth_path) == auth
        and _file_hash(immutable_auth_path) == actual_auth_hash
    ):
        blockers.append("live_canary_immutable_authorization_missing_or_mismatched")
    required_auth = {
        "authorization_status": "AUTHORIZED_FOR_ONE_LIVE_CANARY",
        "authorization_reusable": False,
        "manual_executor_available": True,
        "canary_execution_authority": True,
        "live_trading_authorized": False,
        "order_submission_performed": False,
        "prerequisite_sample_pass": True,
        "prerequisite_supreme_team_pass": True,
        "prerequisite_stage6_release_pass": True,
        "prerequisite_input_parity_pass": True,
        "prerequisite_executor_preflight_pass": True,
    }
    if any(auth.get(field) != expected for field, expected in required_auth.items()) or auth.get(
        "blockers"
    ) != []:
        blockers.append("live_canary_authorization_not_executable")
    if any(
        auth.get(field) != contract.get(field)
        for field in (
            "executor_contract_id",
            "executor_source_sha256",
            "executor_implementation_bundle_sha256",
        )
    ):
        blockers.append("live_canary_executor_contract_binding_mismatch")
    if not approval_id or approval.get("approval_id") != approval_id or auth.get(
        "user_approval_id"
    ) != approval_id:
        blockers.append("live_canary_approval_id_acknowledgement_mismatch")
    if not _approval_valid(approval=approval, auth=auth, policy=policy, candidate=candidate, as_of=as_of, contract=contract):
        blockers.append("live_canary_signed_approval_invalid")
    max_age = _number((policy.get("canary") or {}).get("maximum_input_age_seconds")) or 0.0
    auth_time = _timestamp(auth.get("generated_at_utc"))
    if auth_time is None or not 0.0 <= (as_of - auth_time).total_seconds() <= max_age:
        blockers.append("live_canary_authorization_stale_or_future")
    preflight_ready, preflight_blockers = validate_live_canary_executor_preflight(
        receipt=preflight,
        candidate=candidate,
        policy_id=_policy_id(policy),
        as_of=as_of,
        maximum_age_seconds=max_age,
    )
    if not preflight_ready:
        blockers.extend(preflight_blockers)
    if auth.get("executor_preflight_sha256") != _file_hash(artifacts["preflight_path"]):
        blockers.append("live_canary_executor_preflight_file_binding_mismatch")
    if auth.get("candidate_receipt_id") != candidate.get("candidate_receipt_id") or auth.get(
        "candidate_receipt_sha256"
    ) != candidate.get("receipt_sha256"):
        blockers.append("live_canary_candidate_binding_mismatch")
    blockers.extend(
        _current_authorization_evidence_blockers(
            root=root,
            artifacts=artifacts,
            as_of=as_of,
            maximum_age_seconds=max_age,
        )
    )
    blockers.extend(config.configuration_blockers())
    if require_execution:
        if os.getenv(LIVE_ENABLE_ENV, "").strip().lower() != "true":
            blockers.append("live_canary_environment_enable_missing")
        if acknowledgement != LIVE_ACKNOWLEDGEMENT:
            blockers.append("live_canary_exact_acknowledgement_missing")
    used = _approval_used(root, approval_id)
    if used and not (permit_existing_reservation and _reservation_path(root, approval_id).is_file()):
        blockers.append("live_canary_approval_already_reserved_or_used")
    return _unique(blockers)


def _current_authorization_evidence_blockers(
    *,
    root: Path,
    artifacts: dict[str, Any],
    as_of: datetime,
    maximum_age_seconds: float,
) -> list[str]:
    """Revalidate every current authorization input before key access."""

    from quant_platform.orchestration.corrective_live_canary import (
        _parity_rows,
        _policy_id,
        _validate_policy_core,
    )
    from quant_platform.orchestration.corrective_release_gates import (
        _validated_testnet_candidate_receipt,
        validate_stage6_release_evidence,
    )

    blockers: list[str] = []
    auth = artifacts["authorization"]
    approval = artifacts["approval"]
    candidate = artifacts["candidate"]
    policy = artifacts["policy"]

    candidate_valid, candidate_blockers = _validated_testnet_candidate_receipt(
        root=root,
        candidate=candidate,
    )
    if not candidate_valid:
        blockers.append("live_canary_current_candidate_evidence_invalid")
        blockers.extend(candidate_blockers)
    candidate_time = _timestamp(candidate.get("generated_at_utc"))
    candidate_age = (
        (as_of - candidate_time).total_seconds()
        if candidate_time is not None
        else math.inf
    )
    if (
        candidate_time is None
        or candidate_age < 0.0
        or candidate_age > maximum_age_seconds
    ):
        blockers.append("live_canary_candidate_receipt_stale_or_future")

    policy_valid, _policy_blockers = _validate_policy_core(policy)
    policy_id = _policy_id(policy) if policy_valid else ""
    if not policy_valid:
        blockers.append("live_canary_current_policy_invalid")
    policy_receipt_path = (
        root / "data" / "live" / "canary_policies" / f"{policy_id}.json"
    )
    policy_pointer_path = root / "data" / "live" / "active_canary_policy.json"
    policy_receipt = _read_json(policy_receipt_path)
    policy_pointer = _read_json(policy_pointer_path)
    expected_policy_receipt = {**policy, "live_canary_policy_id": policy_id}
    expected_policy_receipt["receipt_sha256"] = _payload_hash(
        expected_policy_receipt
    )
    policy_receipt_hash = (
        _file_hash(policy_receipt_path) if policy_receipt_path.is_file() else ""
    )
    expected_policy_relative = _relative(policy_receipt_path, root)
    policy_pointer_valid = bool(
        policy_pointer_path.is_file()
        and policy_pointer.get("schema_version")
        == "thewiz.live_canary_policy_pointer.v1"
        and policy_pointer.get("live_canary_policy_id") == policy_id
        and policy_pointer.get("policy_path") == expected_policy_relative
        and policy_pointer.get("policy_sha256") == policy_receipt_hash
        and policy_pointer.get("live_trading_authorized") is False
        and policy_pointer.get("receipt_sha256") == _payload_hash(policy_pointer)
    )
    if not (
        policy_valid
        and policy_receipt_path.is_file()
        and policy_receipt == expected_policy_receipt
        and auth.get("live_canary_policy_id") == policy_id
        and approval.get("live_canary_policy_id") == policy_id
        and auth.get("live_canary_policy_receipt_sha256") == policy_receipt_hash
        and policy_pointer_valid
    ):
        blockers.append("live_canary_registered_policy_evidence_invalid")

    sample_path = (
        root / "reports" / "active" / "realized_testnet_sample_sufficiency.csv"
    )
    supreme_path = (
        root / "reports" / "supreme_team" / "testnet_evidence_checkpoint.json"
    )
    sample_hash = _file_hash(sample_path) if sample_path.is_file() else ""
    supreme_hash = _file_hash(supreme_path) if supreme_path.is_file() else ""
    sample_hash_bound = bool(
        sample_hash
        and auth.get("testnet_sample_evidence_sha256") == sample_hash
        and approval.get("testnet_sample_evidence_sha256") == sample_hash
    )
    supreme_hash_bound = bool(
        supreme_hash
        and auth.get("testnet_supreme_team_evidence_sha256") == supreme_hash
        and approval.get("testnet_supreme_team_evidence_sha256") == supreme_hash
    )
    if not sample_hash_bound or not _sample_evidence_passes(sample_path):
        blockers.append("live_canary_testnet_sample_evidence_invalid_or_changed")
    supreme = _read_json(supreme_path)
    supreme_pass = bool(
        supreme.get("checkpoint_status") == "PASS"
        and not supreme.get("blockers")
        and supreme.get("testnet_order_authority") is not True
        and supreme.get("live_trading_authorized") is not True
    )
    if not supreme_hash_bound or not supreme_pass:
        blockers.append("live_canary_supreme_team_evidence_invalid_or_changed")

    (
        stage6_release_valid,
        stage6_release,
        stage6_release_path,
        stage6_release_blockers,
    ) = validate_stage6_release_evidence(
        root=root,
        candidate=candidate,
        sample_evidence_path=sample_path,
        supreme_evidence_path=supreme_path,
    )
    expected_stage6_bindings = {
        "stage6_release_receipt_id": stage6_release.get(
            "stage6_release_receipt_id", ""
        ),
        "stage6_release_receipt_path": (
            _relative(stage6_release_path, root)
            if stage6_release_path is not None
            else ""
        ),
        "stage6_release_receipt_sha256": (
            _file_hash(stage6_release_path)
            if stage6_release_path is not None and stage6_release_path.is_file()
            else ""
        ),
    }
    if (
        not stage6_release_valid
        or any(
            auth.get(field) != value or approval.get(field) != value
            for field, value in expected_stage6_bindings.items()
        )
    ):
        blockers.append("live_canary_stage6_release_evidence_invalid_or_changed")
        blockers.extend(stage6_release_blockers)

    parity_path = (
        root / "reports" / "active" / "testnet_live_input_parity_evidence.json"
    )
    parity = _read_json(parity_path)
    parity_binding_valid = bool(
        parity_path.is_file()
        and parity.get("parity_receipt_id") == auth.get("parity_receipt_id")
        and parity.get("parity_receipt_id") == approval.get("parity_receipt_id")
    )
    parity_rows = _parity_rows(
        root=root,
        evidence=parity,
        candidate=candidate,
        policy=policy,
        policy_id=policy_id,
        as_of=as_of,
        upstream_ready=bool(
            candidate_valid
            and policy_valid
            and sample_hash_bound
            and supreme_hash_bound
            and _sample_evidence_passes(sample_path)
            and supreme_pass
        ),
    )
    parity_pass = bool(
        parity_binding_valid
        and not parity_rows.empty
        and parity_rows["status"].eq("PASS").all()
    )
    if not parity_pass:
        blockers.append("live_canary_current_input_parity_evidence_invalid")
    return _unique(blockers)


def _sample_evidence_passes(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return False
    return bool(
        rows
        and all(str(row.get("status", "")).strip().upper() == "PASS" for row in rows)
        and all(
            str(row.get(field, "")).strip().lower() not in {"true", "1", "yes"}
            for row in rows
            for field in ("testnet_order_authority", "live_trading_authorized")
        )
    )


def _approval_valid(
    *,
    approval: dict[str, Any],
    auth: dict[str, Any],
    policy: dict[str, Any],
    candidate: dict[str, Any],
    as_of: datetime,
    contract: dict[str, Any],
) -> bool:
    if not (
        approval.get("schema_version") == APPROVAL_SCHEMA_VERSION
        and approval.get("approved") is True
        and approval.get("one_run_only") is True
        and approval.get("order_submission_performed") is False
        and approval.get("live_trading_authorized") is False
    ):
        return False
    issued = _timestamp(approval.get("issued_at_utc"))
    expires = _timestamp(approval.get("expires_at_utc"))
    if issued is None or expires is None or not (issued <= as_of < expires) or expires - issued > _minutes(15):
        return False
    bindings = {
        "candidate_receipt_id": candidate.get("candidate_receipt_id", ""),
        "candidate_receipt_sha256": candidate.get("receipt_sha256", ""),
        "live_canary_policy_id": _policy_id(policy),
        "pair": candidate.get("pair", ""),
        "executor_contract_id": contract["executor_contract_id"],
        "executor_source_sha256": contract["executor_source_sha256"],
        "executor_implementation_bundle_sha256": contract[
            "executor_implementation_bundle_sha256"
        ],
        "stage6_release_receipt_id": auth.get("stage6_release_receipt_id", ""),
        "stage6_release_receipt_path": auth.get("stage6_release_receipt_path", ""),
        "stage6_release_receipt_sha256": auth.get(
            "stage6_release_receipt_sha256", ""
        ),
    }
    if any(approval.get(field) != value for field, value in bindings.items()):
        return False
    if auth.get("exact_pair") != approval.get("pair") or auth.get("exact_side_and_sizes") != approval.get("legs"):
        return False
    core = {key: value for key, value in approval.items() if key not in {"approval_id", "wallet_signature"}}
    if approval.get("approval_id") != "livecanaryapproval_" + _payload_hash(core)[:20]:
        return False
    signer = str(policy.get("authorized_signer_address", "")).strip()
    if signer.lower() != str(approval.get("authorized_signer_address", "")).strip().lower():
        return False
    signed = {key: value for key, value in approval.items() if key != "wallet_signature"}
    try:
        recovered = Account.recover_message(
            encode_defunct(text=_canonical_json(signed)),
            signature=str(approval.get("wallet_signature", "")),
        )
    except (TypeError, ValueError):
        return False
    return recovered.lower() == signer.lower()


def _runtime_pre_reservation_checks(
    *, artifacts: dict[str, Any], config: HyperliquidLiveCanaryConfig, fetch: Callable[[dict[str, Any]], Any]
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    approval = artifacts["approval"]
    markets = {_market(leg) for leg in approval.get("legs", []) if isinstance(leg, dict)}
    runtime: dict[str, Any] = {}
    try:
        role = fetch({"type": "userRole", "user": config.agent_address})
        linked = str(((role or {}).get("data") or {}).get("user", "")) if isinstance(role, dict) else ""
        if not (isinstance(role, dict) and role.get("role") == "agent" and linked.lower() == str(config.master_address).lower()):
            blockers.append("live_canary_runtime_agent_authorization_failed")
        meta = _market_meta(fetch)
        mids = _positive_mids(fetch, markets)
        state = fetch({"type": "clearinghouseState", "user": config.master_address})
        orders = fetch({"type": "openOrders", "user": config.master_address})
        runtime.update({"meta": meta, "mids": mids, "state": state, "open_orders": orders})
        if not markets or any(market not in meta for market in markets):
            blockers.append("live_canary_runtime_market_unavailable")
        positions = _positions(state, markets)
        if not _positions_match(positions, {market: 0.0 for market in markets}):
            blockers.append("live_canary_runtime_pair_not_flat")
        if _pair_open_orders(orders, markets):
            blockers.append("live_canary_runtime_pair_open_orders_present")
        withdrawable = _number((state or {}).get("withdrawable")) if isinstance(state, dict) else None
        cap = _number(approval.get("maximum_total_notional_usd")) or math.inf
        if withdrawable is None or withdrawable < cap:
            blockers.append("live_canary_runtime_withdrawable_below_canary_cap")
        blockers.extend(_leg_runtime_blockers(approval=approval, meta=meta, mids=mids))
    except Exception as exc:  # noqa: BLE001 - all public-data failures block reservation
        blockers.append(f"live_canary_runtime_read_failed:{type(exc).__name__}")
    return runtime, _unique(blockers)


def _leg_runtime_blockers(*, approval: dict[str, Any], meta: dict[str, Any], mids: dict[str, float]) -> list[str]:
    blockers: list[str] = []
    total_mid_notional = 0.0
    slippage_bps = _number(approval.get("maximum_slippage_bps")) or 0.0
    for leg in approval.get("legs", []):
        market = _market(leg)
        side = str(leg.get("side", "")).upper()
        size = _number(leg.get("size"))
        limit_price = _number(leg.get("limit_price"))
        mid = mids.get(market)
        row = meta.get(market, {})
        sz_decimals = _integer(row.get("szDecimals"))
        if size is None or limit_price is None or mid is None or sz_decimals is None:
            blockers.append(f"live_canary_runtime_leg_data_invalid:{market}")
            continue
        total_mid_notional += size * mid
        if not _decimal_places_valid(size, sz_decimals):
            blockers.append(f"live_canary_runtime_size_precision_invalid:{market}")
        if size * mid < 10.0:
            blockers.append(f"live_canary_runtime_leg_below_minimum_notional:{market}")
        if side == "BUY":
            observed = (limit_price - mid) / mid * 10_000.0
            marketable = limit_price >= mid
        elif side == "SELL":
            observed = (mid - limit_price) / mid * 10_000.0
            marketable = limit_price <= mid
        else:
            observed, marketable = math.inf, False
        if not marketable or observed < 0.0 or observed > slippage_bps + 1e-9:
            blockers.append(f"live_canary_runtime_limit_outside_slippage:{market}")
    cap = _number(approval.get("maximum_total_notional_usd")) or 0.0
    if not 0.0 < total_mid_notional <= cap:
        blockers.append("live_canary_runtime_total_notional_outside_cap")
    return blockers


def _reserve_approval(
    *, root: Path, as_of: datetime, authorization_sha256: str, approval_id: str,
    artifacts: dict[str, Any], config: HyperliquidLiveCanaryConfig
) -> dict[str, Any]:
    path = _reservation_path(root, approval_id)
    payload = {
        "schema_version": EXECUTOR_RESERVATION_SCHEMA_VERSION,
        "reserved_at_utc": as_of.isoformat(),
        "approval_id": approval_id,
        "authorization_receipt_sha256": authorization_sha256,
        "authorization_id": artifacts["authorization"].get("authorization_id", ""),
        "authorization_receipt_path": artifacts["authorization"].get(
            "immutable_authorization_path", ""
        ),
        "candidate_receipt_id": artifacts["candidate"].get("candidate_receipt_id", ""),
        "executor_contract_id": live_canary_executor_contract()["executor_contract_id"],
        "executor_implementation_bundle_sha256": live_canary_executor_contract()[
            "executor_implementation_bundle_sha256"
        ],
        "master_address": config.master_address,
        "agent_address": config.agent_address,
        "entry_retry_allowed": False,
        "authorization_reusable": False,
        "live_trading_authorized": False,
    }
    payload["reservation_id"] = "livecanaryreservation_" + _payload_hash(payload)[:20]
    payload["receipt_sha256"] = _payload_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return payload


def _finalize_success(
    *, root: Path, started: datetime, artifacts: dict[str, Any], config: HyperliquidLiveCanaryConfig,
    fetch: Callable[[dict[str, Any]], Any], state: dict[str, Any], authorization_sha256: str,
    final_positions: dict[str, float]
) -> LiveCanaryExecutorResult:
    completed = datetime.now(UTC)
    start_ms = int(started.timestamp() * 1000) - 1
    end_ms = int(completed.timestamp() * 1000) + 1
    fills_raw = fetch({
        "type": "userFillsByTime", "user": config.master_address,
        "startTime": start_ms, "endTime": end_ms, "aggregateByTime": True,
    })
    funding_raw = fetch({
        "type": "userFunding", "user": config.master_address,
        "startTime": start_ms, "endTime": end_ms,
    })
    entry_fills, exit_fills, fill_blockers = _reconciled_fill_rows(
        fills_raw=fills_raw, legs=artifacts["approval"]["legs"], started=started, completed=completed
    )
    if fill_blockers:
        return _incident_result(
            root=root, as_of=completed, artifacts=artifacts, state=state,
            incident=";".join(fill_blockers), order_submission_performed=True,
            reconciled_flat=True,
        )
    funding = _funding_total(funding_raw)
    if funding is None:
        return _incident_result(
            root=root, as_of=completed, artifacts=artifacts, state=state,
            incident="live_canary_funding_reconciliation_invalid",
            order_submission_performed=True, reconciled_flat=True,
        )
    for fill in entry_fills + exit_fills:
        fill["funding_pnl_usd"] = 0.0
    exit_fills[-1]["funding_pnl_usd"] = funding
    gross = sum(_leg_gross(entry, exit_fill) for entry, exit_fill in zip(entry_fills, exit_fills, strict=True))
    fees = sum(float(fill["fee_usd"]) for fill in entry_fills + exit_fills)
    execution = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "authorization_receipt_sha256": authorization_sha256,
        "authorization_id": artifacts["authorization"]["authorization_id"],
        "authorization_receipt_path": artifacts["authorization"][
            "immutable_authorization_path"
        ],
        "approval_id": artifacts["approval"]["approval_id"],
        "candidate_receipt_id": artifacts["candidate"]["candidate_receipt_id"],
        "live_canary_policy_id": artifacts["authorization"]["live_canary_policy_id"],
        "executor_contract_id": live_canary_executor_contract()["executor_contract_id"],
        "executor_source_sha256": live_canary_executor_contract()["executor_source_sha256"],
        "executor_implementation_bundle_sha256": live_canary_executor_contract()[
            "executor_implementation_bundle_sha256"
        ],
        "pair": artifacts["candidate"]["pair"],
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "entry_fills": entry_fills,
        "exit_fills": exit_fills,
        "gross_pnl_usd": gross,
        "fees_usd": fees,
        "funding_pnl_usd": funding,
        "net_pnl_usd": gross - fees + funding,
        "final_positions": {market: float(value) for market, value in final_positions.items()},
        "open_order_ids": [],
        "unresolved_incidents": [],
        "reconciled_flat": True,
        "entry_retry_attempted": False,
        "repeat_authorized": False,
        "live_trading_authorized": False,
    }
    identity = _identity(execution, "execution")
    execution_id = "livecanaryexec_" + identity[:20]
    immutable = root / "data" / "live" / "canary_executions" / f"{execution_id}.json"
    execution.update({
        "execution_id": execution_id,
        "receipt_identity_sha256": identity,
        "immutable_execution_path": _relative(immutable, root),
    })
    _write_immutable_json(execution, immutable)
    _atomic_json(execution, root / "reports" / "active" / EXECUTION_PATH.name)
    _append_ledger(root, {
        "approval_id": artifacts["approval"]["approval_id"],
        "execution_id": execution_id,
        "authorization_receipt_sha256": authorization_sha256,
        "reservation_id": _read_json(_reservation_path(root, artifacts["approval"]["approval_id"])).get("reservation_id", ""),
    })
    state.update({"phase": "FLAT_RECONCILED", "execution_id": execution_id, "reconciled_flat": True})
    _write_state(root, state)
    return _status_result(
        root=root, as_of=completed, status="EXECUTED_ONE_CANARY_PENDING_SUPREME_REVIEW",
        blockers=[], order_submission_performed=True, reconciled_flat=True, execution_id=execution_id,
    )


def _recover_after_anomaly(
    *, root: Path, as_of: datetime, artifacts: dict[str, Any], config: HyperliquidLiveCanaryConfig,
    fetch: Callable[[dict[str, Any]], Any], exchange: Any, state: dict[str, Any], incident: str
) -> LiveCanaryExecutorResult:
    state["recovery_attempted"] = True
    state["phase"] = "RECOVERY_REQUIRED"
    _write_state(root, state)
    recovery_blockers = _recover_to_flat(
        config=config, fetch=fetch, exchange=exchange,
        markets={_market(leg) for leg in artifacts["approval"].get("legs", [])},
    )
    return _incident_result(
        root=root, as_of=datetime.now(UTC), artifacts=artifacts, state=state,
        incident=";".join([incident, *recovery_blockers]),
        order_submission_performed=bool(state.get("entry_submit_attempted")),
        reconciled_flat=not recovery_blockers,
    )


def _recover_reserved_canary(
    *, root: Path, as_of: datetime, artifacts: dict[str, Any], config: HyperliquidLiveCanaryConfig,
    fetch: Callable[[dict[str, Any]], Any], keychain_reader: Callable[[str, str], str] | None,
    exchange_factory: Callable[[Any, HyperliquidLiveCanaryConfig], Any] | None
) -> LiveCanaryExecutorResult:
    try:
        wallet = _load_wallet(config, keychain_reader)
        exchange = _build_exchange(wallet, config, exchange_factory)
    except Exception as exc:  # noqa: BLE001
        return _incident_result(
            root=root, as_of=as_of, artifacts=artifacts, state=_read_json(root / "reports" / "active" / STATE_PATH.name),
            incident=f"live_canary_recovery_setup_failed:{type(exc).__name__}",
            order_submission_performed=False, reconciled_flat=False,
        )
    markets = {_market(leg) for leg in artifacts["approval"].get("legs", [])}
    recovery_blockers = _recover_to_flat(config=config, fetch=fetch, exchange=exchange, markets=markets)
    return _incident_result(
        root=root, as_of=datetime.now(UTC), artifacts=artifacts,
        state=_read_json(root / "reports" / "active" / STATE_PATH.name),
        incident="live_canary_recovery_only_run" + (";" + ";".join(recovery_blockers) if recovery_blockers else ""),
        order_submission_performed=bool(markets), reconciled_flat=not recovery_blockers,
    )


def _recover_to_flat(
    *, config: HyperliquidLiveCanaryConfig, fetch: Callable[[dict[str, Any]], Any], exchange: Any,
    markets: set[str]
) -> list[str]:
    if exchange is None:
        return ["live_canary_recovery_exchange_unavailable"]
    blockers: list[str] = []
    try:
        orders = _pair_open_orders(fetch({"type": "openOrders", "user": config.master_address}), markets)
        if orders:
            exchange.bulk_cancel([{"coin": _market(row), "oid": int(row["oid"])} for row in orders])
        positions = _positions(fetch({"type": "clearinghouseState", "user": config.master_address}), markets)
        for market, size in positions.items():
            if abs(size) > 1e-12:
                exchange.market_close(market, sz=abs(size), slippage=0.01)
        final_positions = _positions(fetch({"type": "clearinghouseState", "user": config.master_address}), markets)
        final_orders = _pair_open_orders(fetch({"type": "openOrders", "user": config.master_address}), markets)
        if not _positions_match(final_positions, {market: 0.0 for market in markets}):
            blockers.append("live_canary_recovery_position_remains")
        if final_orders:
            blockers.append("live_canary_recovery_open_order_remains")
    except Exception as exc:  # noqa: BLE001
        blockers.append(f"live_canary_recovery_failed:{type(exc).__name__}")
    return _unique(blockers)


def _incident_result(
    *, root: Path, as_of: datetime, artifacts: dict[str, Any], state: dict[str, Any], incident: str,
    order_submission_performed: bool, reconciled_flat: bool
) -> LiveCanaryExecutorResult:
    incident_payload = {
        "schema_version": EXECUTOR_INCIDENT_SCHEMA_VERSION,
        "recorded_at_utc": as_of.isoformat(),
        "approval_id": artifacts["approval"].get("approval_id", ""),
        "authorization_receipt_sha256": _file_hash(artifacts["authorization_path"]),
        "authorization_id": artifacts["authorization"].get("authorization_id", ""),
        "authorization_receipt_path": artifacts["authorization"].get(
            "immutable_authorization_path", ""
        ),
        "candidate_receipt_id": artifacts["candidate"].get("candidate_receipt_id", ""),
        "incident": incident,
        "entry_retry_attempted": False,
        "recovery_attempted": bool(state.get("recovery_attempted", True)),
        "reconciled_flat": reconciled_flat,
        "repeat_authorized": False,
        "live_trading_authorized": False,
    }
    incident_payload["incident_id"] = "livecanaryincident_" + _payload_hash(incident_payload)[:20]
    incident_payload["receipt_sha256"] = _payload_hash(incident_payload)
    immutable = root / "data" / "live" / "canary_incidents" / f"{incident_payload['incident_id']}.json"
    _write_immutable_json(incident_payload, immutable)
    path = root / "reports" / "active" / INCIDENT_PATH.name
    _atomic_json(incident_payload, path)
    if not _ledger_has_approval(root, str(incident_payload["approval_id"])):
        _append_ledger(root, {
            "approval_id": incident_payload["approval_id"],
            "execution_id": incident_payload["incident_id"],
            "authorization_receipt_sha256": incident_payload["authorization_receipt_sha256"],
            "incident": True,
        })
    state.update({"phase": "INCIDENT_FLAT" if reconciled_flat else "INCIDENT_UNRESOLVED", "incident_id": incident_payload["incident_id"], "reconciled_flat": reconciled_flat})
    _write_state(root, state)
    return _status_result(
        root=root, as_of=as_of,
        status="INCIDENT_RECOVERED_FLAT" if reconciled_flat else "INCIDENT_RECOVERY_REQUIRED",
        blockers=[incident], order_submission_performed=order_submission_performed,
        reconciled_flat=reconciled_flat, incident_path=_relative(path, root),
    )


def _reconciled_fill_rows(
    *, fills_raw: Any, legs: list[dict[str, Any]], started: datetime, completed: datetime
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if not isinstance(fills_raw, list):
        return [], [], ["live_canary_user_fills_response_invalid"]
    rows = [row for row in fills_raw if isinstance(row, dict) and _market(row) in {_market(leg) for leg in legs}]
    entry_rows: list[dict[str, Any]] = []
    exit_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for leg in legs:
        market = _market(leg)
        entry_side = str(leg.get("side", "")).upper()
        exit_side = "SELL" if entry_side == "BUY" else "BUY"
        entry = _aggregate_fills(rows, market=market, side=entry_side)
        exit_fill = _aggregate_fills(rows, market=market, side=exit_side)
        approved_size = _number(leg.get("size")) or 0.0
        if entry is None or exit_fill is None:
            blockers.append(f"live_canary_fill_missing:{market}")
            continue
        if not math.isclose(float(entry["size"]), approved_size, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(float(exit_fill["size"]), approved_size, rel_tol=0.0, abs_tol=1e-12):
            blockers.append(f"live_canary_fill_size_mismatch:{market}")
        entry_time = _timestamp(entry["timestamp_utc"])
        exit_time = _timestamp(exit_fill["timestamp_utc"])
        if entry_time is None or exit_time is None or not (started <= entry_time < exit_time <= completed):
            blockers.append(f"live_canary_fill_timeline_invalid:{market}")
        entry_rows.append(entry)
        exit_rows.append(exit_fill)
    if len(entry_rows) != 2 or len(exit_rows) != 2:
        blockers.append("live_canary_requires_exact_two_entry_and_exit_fills")
    identities = [str(row.get("exchange_order_id", "")) for row in entry_rows + exit_rows]
    if len(set(identities)) != 4 or any(not value for value in identities):
        blockers.append("live_canary_fill_order_identity_invalid")
    return entry_rows, exit_rows, _unique(blockers)


def _aggregate_fills(rows: list[dict[str, Any]], *, market: str, side: str) -> dict[str, Any] | None:
    selected = [row for row in rows if _market(row) == market and _fill_side(row) == side]
    if not selected:
        return None
    sizes = [_number(row.get("sz")) for row in selected]
    prices = [_number(row.get("px")) for row in selected]
    fees = [_number(row.get("fee", 0.0)) for row in selected]
    times = [_integer(row.get("time")) for row in selected]
    order_ids = {str(row.get("oid", "")) for row in selected}
    fee_tokens = {str(row.get("feeToken", "USDC")).upper() for row in selected}
    if any(value is None for value in sizes + prices + fees + times) or len(order_ids) != 1 or fee_tokens - {"USDC"}:
        return None
    total_size = sum(float(value) for value in sizes if value is not None)
    if total_size <= 0.0:
        return None
    average = sum(float(size) * float(price) for size, price in zip(sizes, prices, strict=True)) / total_size
    identity = sha256(_canonical_json(selected).encode("utf-8")).hexdigest()[:24]
    return {
        "fill_id": "hlfill_" + identity,
        "exchange_order_id": next(iter(order_ids)),
        "market": market,
        "side": side,
        "size": total_size,
        "price": average,
        "fee_usd": sum(float(value) for value in fees if value is not None),
        "funding_pnl_usd": 0.0,
        "timestamp_utc": datetime.fromtimestamp(max(int(value) for value in times if value is not None) / 1000.0, tz=UTC).isoformat(),
    }


def _funding_total(rows: Any) -> float | None:
    if not isinstance(rows, list):
        return None
    total = 0.0
    for row in rows:
        if not isinstance(row, dict):
            return None
        delta = row.get("delta") if isinstance(row.get("delta"), dict) else row
        value = _number(delta.get("usdc") if isinstance(delta, dict) else None)
        if value is None:
            return None
        total += value
    return total


def _entry_order_request(leg: dict[str, Any]) -> dict[str, Any]:
    return {
        "coin": _market(leg),
        "is_buy": str(leg.get("side", "")).upper() == "BUY",
        "sz": float(leg["size"]),
        "limit_px": float(leg["limit_price"]),
        "order_type": {"limit": {"tif": "Ioc"}},
        "reduce_only": False,
    }


def _exit_order_requests(
    *, legs: list[dict[str, Any]], mids: dict[str, float], market_meta: dict[str, Any],
    maximum_slippage_bps: float
) -> list[dict[str, Any]]:
    requests_out = []
    slippage = maximum_slippage_bps / 10_000.0
    for leg in legs:
        market = _market(leg)
        exit_buy = str(leg.get("side", "")).upper() == "SELL"
        raw = mids[market] * (1.0 + slippage if exit_buy else 1.0 - slippage)
        decimals = int(market_meta[market]["szDecimals"])
        limit_px = round(float(f"{raw:.5g}"), 6 - decimals)
        requests_out.append({
            "coin": market, "is_buy": exit_buy, "sz": float(leg["size"]),
            "limit_px": limit_px, "order_type": {"limit": {"tif": "Ioc"}},
            "reduce_only": True,
        })
    return requests_out


def _wait_for_positions(
    *, fetch: Callable[[dict[str, Any]], Any], master_address: str, markets: set[str],
    expected: dict[str, float], waiter: Callable[[float], None]
) -> dict[str, float]:
    observed = {market: 0.0 for market in markets}
    for attempt in range(4):
        observed = _positions(fetch({"type": "clearinghouseState", "user": master_address}), markets)
        if _positions_match(observed, expected):
            return observed
        if attempt < 3:
            waiter(0.75)
    return observed


def _positions(state: Any, markets: set[str]) -> dict[str, float]:
    result = {market: 0.0 for market in markets}
    rows = state.get("assetPositions", []) if isinstance(state, dict) else []
    if not isinstance(rows, list):
        raise TypeError("live_canary_positions_missing")
    for row in rows:
        position = row.get("position", {}) if isinstance(row, dict) else {}
        market = _market(position)
        size = _number(position.get("szi")) if isinstance(position, dict) else None
        if market in result and size is not None:
            result[market] = size
    return result


def _positions_match(actual: dict[str, float], expected: dict[str, float]) -> bool:
    return set(actual) == set(expected) and all(
        math.isclose(float(actual[key]), float(expected[key]), rel_tol=0.0, abs_tol=1e-12)
        for key in expected
    )


def _expected_entry_positions(legs: list[dict[str, Any]]) -> dict[str, float]:
    return {
        _market(leg): float(leg["size"]) * (1.0 if str(leg["side"]).upper() == "BUY" else -1.0)
        for leg in legs
    }


def _pair_open_orders(rows: Any, markets: set[str]) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise TypeError("live_canary_open_orders_missing")
    return [row for row in rows if isinstance(row, dict) and _market(row) in markets]


def _market_meta(fetch: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    payload = fetch({"type": "meta"})
    rows = payload.get("universe", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        raise TypeError("live_canary_market_meta_invalid")
    return {
        _market(row): row for row in rows
        if isinstance(row, dict) and _market(row) and not bool(row.get("isDelisted", False))
    }


def _positive_mids(fetch: Callable[[dict[str, Any]], Any], markets: set[str]) -> dict[str, float]:
    payload = fetch({"type": "allMids"})
    if not isinstance(payload, dict):
        raise TypeError("live_canary_all_mids_invalid")
    result = {_market({"market": key}): _number(value) for key, value in payload.items()}
    if any(result.get(market) is None or float(result[market]) <= 0.0 for market in markets):
        raise ValueError("live_canary_candidate_mid_missing")
    return {market: float(result[market]) for market in markets}


def _load_wallet(
    config: HyperliquidLiveCanaryConfig,
    keychain_reader: Callable[[str, str], str] | None,
) -> Any:
    reader = keychain_reader or read_live_agent_key_from_keychain
    secret = reader(str(config.keychain_service), str(config.agent_address))
    wallet = Account.from_key(secret)
    if wallet.address.lower() != str(config.agent_address).lower():
        raise ValueError("live_canary_agent_key_address_mismatch")
    return wallet


def _build_exchange(
    wallet: Any,
    config: HyperliquidLiveCanaryConfig,
    factory: Callable[[Any, HyperliquidLiveCanaryConfig], Any] | None,
) -> Any:
    if factory is not None:
        return factory(wallet, config)
    from hyperliquid.exchange import Exchange

    return Exchange(
        wallet,
        base_url=config.base_url,
        account_address=config.master_address,
        timeout=20.0,
    )


def _default_info_client(base_url: str) -> Callable[[dict[str, Any]], Any]:
    session = requests.Session()

    def fetch(payload: dict[str, Any]) -> Any:
        response = session.post(f"{base_url.rstrip('/')}/info", json=payload, timeout=20)
        response.raise_for_status()
        return response.json()

    return fetch


def _status_result(
    *, root: Path, as_of: datetime, status: str, blockers: list[str],
    order_submission_performed: bool = False, reconciled_flat: bool = False,
    execution_id: str = "", incident_path: str = ""
) -> LiveCanaryExecutorResult:
    payload = {
        "schema_version": EXECUTOR_STATUS_SCHEMA_VERSION,
        "generated_at_utc": as_of.isoformat(),
        "status": status,
        "blockers": _unique(blockers),
        "executor_contract_id": live_canary_executor_contract()["executor_contract_id"],
        "executor_implementation_bundle_sha256": live_canary_executor_contract()[
            "executor_implementation_bundle_sha256"
        ],
        "order_submission_performed": order_submission_performed,
        "reconciled_flat": reconciled_flat,
        "execution_id": execution_id,
        "incident_path": incident_path,
        "repeat_authorized": False,
        "live_trading_authorized": False,
    }
    payload["receipt_sha256"] = _payload_hash(payload)
    path = root / "reports" / "active" / STATUS_PATH.name
    _atomic_json(payload, path)
    return LiveCanaryExecutorResult(
        status=status, blockers=tuple(payload["blockers"]),
        order_submission_performed=order_submission_performed,
        reconciled_flat=reconciled_flat, execution_id=execution_id,
        status_path=_relative(path, root), incident_path=incident_path,
    )


@contextmanager
def _exclusive_executor_lock(root: Path):
    path = root / "data" / "live" / LOCK_PATH.name
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        yield False
        return
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield True
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _approval_used(root: Path, approval_id: str) -> bool:
    return _reservation_path(root, approval_id).is_file() or _ledger_has_approval(root, approval_id)


def _ledger_has_approval(root: Path, approval_id: str) -> bool:
    path = root / "data" / "live" / LEDGER_PATH.name
    if not approval_id or not path.is_file():
        return False
    try:
        return any(
            isinstance(row, dict) and row.get("approval_id") == approval_id
            for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        )
    except (OSError, json.JSONDecodeError):
        return True


def _append_ledger(root: Path, payload: dict[str, Any]) -> None:
    path = root / "data" / "live" / LEDGER_PATH.name
    path.parent.mkdir(parents=True, exist_ok=True)
    if _ledger_has_approval(root, str(payload.get("approval_id", ""))):
        raise ValueError("live_canary_ledger_duplicate_approval")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, (_canonical_json(payload) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reservation_path(root: Path, approval_id: str) -> Path:
    safe = approval_id if approval_id and all(char.isalnum() or char in "_-" for char in approval_id) else "invalid"
    return root / "data" / "live" / "canary_authorization_reservations" / f"{safe}.json"


def _write_state(root: Path, state: dict[str, Any]) -> None:
    payload = dict(state)
    payload["updated_at_utc"] = datetime.now(UTC).isoformat()
    payload["receipt_sha256"] = _payload_hash(payload)
    _atomic_json(payload, root / "reports" / "active" / STATE_PATH.name)


def _response_summary(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {"status": "invalid_response"}
    statuses = (((response.get("response") or {}).get("data") or {}).get("statuses"))
    return {
        "status": str(response.get("status", "")),
        "leg_status_count": len(statuses) if isinstance(statuses, list) else 0,
        "all_legs_returned": isinstance(statuses, list) and len(statuses) == 2,
    }


def _fill_side(row: dict[str, Any]) -> str:
    raw = str(row.get("side", "")).upper()
    return "BUY" if raw in {"B", "BUY"} else "SELL" if raw in {"A", "SELL"} else ""


def _leg_gross(entry: dict[str, Any], exit_fill: dict[str, Any]) -> float:
    size = float(entry["size"])
    if entry["side"] == "BUY":
        return (float(exit_fill["price"]) - float(entry["price"])) * size
    return (float(entry["price"]) - float(exit_fill["price"])) * size


def _market(row: dict[str, Any]) -> str:
    raw = row.get("market", row.get("coin", row.get("name", "")))
    return str(raw).strip().upper().removesuffix("-USD")


def _decimal_places_valid(value: float, decimals: int) -> bool:
    try:
        scaled = Decimal(str(value)).scaleb(decimals)
    except (InvalidOperation, ValueError):
        return False
    return scaled == scaled.to_integral_value()


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _as_utc(parsed)


def _minutes(value: int):
    from datetime import timedelta

    return timedelta(minutes=value)


def _policy_id(policy: dict[str, Any]) -> str:
    return "livecanarypolicy_" + _payload_hash(policy)[:20]


def _payload_hash(payload: dict[str, Any]) -> str:
    core = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return sha256(_canonical_json(core).encode("utf-8")).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _safe_root_path(root: Path, raw: str) -> Path | None:
    if not raw.strip():
        return None
    candidate = Path(raw)
    path = candidate if candidate.is_absolute() else root / candidate
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return path


def _write_immutable_json(payload: dict[str, Any], path: Path) -> None:
    existing = _read_json(path)
    if existing and existing != payload:
        raise ValueError(f"immutable live-canary artifact conflict: {path}")
    _atomic_json(payload, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
