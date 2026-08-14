"""Fail-closed Hyperliquid Testnet account, signing, and pair-order support.

The single-order adapter is deliberately record-only for pair-trading workflows.
Two-leg submissions must use ``HyperliquidTestnetPairExecutor`` so both legs are
sent in one signed bulk-order action after an explicit, runtime-only approval.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import math
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from eth_account import Account

from quant_platform.execution import ExecutionMode, FillReport, OrderIntent

ROOT = Path(__file__).resolve().parents[2]
HYPERLIQUID_TESTNET_URL = "https://api.hyperliquid-testnet.xyz"
HYPERLIQUID_TESTNET_INFO_URL = f"{HYPERLIQUID_TESTNET_URL}/info"
HYPERLIQUID_TESTNET_PRECHECK_CSV = ROOT / "reports" / "active" / "hyperliquid_testnet_preflight.csv"
HYPERLIQUID_TESTNET_PRECHECK_MD = ROOT / "reports" / "active" / "hyperliquid_testnet_preflight.md"
HYPERLIQUID_TESTNET_MARGIN_CSV = ROOT / "reports" / "active" / "hyperliquid_testnet_margin_snapshot.csv"
HYPERLIQUID_TESTNET_MARGIN_MD = ROOT / "reports" / "active" / "hyperliquid_testnet_margin_snapshot.md"
HYPERLIQUID_TESTNET_EXECUTION_STATE_JSON = (
    ROOT / "reports" / "active" / "hyperliquid_testnet_pair_execution_state.json"
)
ADDRESS_PATTERN = re.compile(r"^0x[a-fA-F0-9]{40}$")
RAW_AGENT_KEY_ENV = "HYPERLIQUID_TESTNET_AGENT_PRIVATE_KEY"


@dataclass(frozen=True)
class HyperliquidTestnetConfig:
    """Non-production account configuration with no raw private-key field."""

    mode: ExecutionMode = ExecutionMode.PAPER
    network: str = "testnet"
    base_url: str = HYPERLIQUID_TESTNET_URL
    master_address: str | None = None
    agent_address: str | None = None
    keychain_service: str | None = None
    submit_orders: bool = False
    max_pair_notional_usd: float = 25.0
    order_approval_id: str | None = None
    requested_leverage: int = 1
    margin_mode: str = "cross"
    one_x_testnet_proof_id: str | None = None
    leverage_scenario_id: str | None = None

    @classmethod
    def paper_testnet_from_env(cls) -> "HyperliquidTestnetConfig":
        return cls(
            network=os.getenv("HYPERLIQUID_NETWORK", "testnet").strip().lower() or "testnet",
            base_url=os.getenv("HYPERLIQUID_TESTNET_BASE_URL", HYPERLIQUID_TESTNET_URL).strip().rstrip("/"),
            master_address=_env_value("HYPERLIQUID_MASTER_ADDRESS"),
            agent_address=_env_value("HYPERLIQUID_AGENT_ADDRESS"),
            keychain_service=_env_value("HYPERLIQUID_TESTNET_AGENT_KEYCHAIN_SERVICE"),
            submit_orders=_env_truthy("HYPERLIQUID_TESTNET_SUBMIT_ORDERS"),
            max_pair_notional_usd=_env_positive_float(
                "HYPERLIQUID_TESTNET_MAX_PAIR_NOTIONAL_USD", 25.0
            ),
            requested_leverage=_env_positive_int(
                "HYPERLIQUID_TESTNET_REQUESTED_LEVERAGE", 1
            ),
            margin_mode=os.getenv(
                "HYPERLIQUID_TESTNET_MARGIN_MODE", "cross"
            ).strip().lower()
            or "cross",
            one_x_testnet_proof_id=_env_value(
                "HYPERLIQUID_TESTNET_ONE_X_PROOF_ID"
            ),
            leverage_scenario_id=_env_value(
                "HYPERLIQUID_TESTNET_LEVERAGE_SCENARIO_ID"
            ),
        )

    def configuration_blockers(self) -> list[str]:
        blockers: list[str] = []
        if self.mode != ExecutionMode.PAPER:
            blockers.append("mode_not_paper")
        if self.network != "testnet":
            blockers.append("unsafe_non_testnet_network")
        if self.base_url.rstrip("/") != HYPERLIQUID_TESTNET_URL:
            blockers.append("unsafe_non_testnet_base_url")
        if not _valid_address(self.master_address):
            blockers.append("missing_or_invalid_hyperliquid_master_address")
        if not _valid_address(self.agent_address):
            blockers.append("missing_or_invalid_hyperliquid_agent_address")
        if not str(self.keychain_service or "").strip():
            blockers.append("missing_hyperliquid_agent_keychain_service")
        if os.getenv(RAW_AGENT_KEY_ENV, "").strip():
            blockers.append("raw_hyperliquid_agent_key_env_forbidden")
        if self.max_pair_notional_usd <= 0:
            blockers.append("invalid_hyperliquid_max_pair_notional")
        if not isinstance(self.requested_leverage, int) or self.requested_leverage < 1:
            blockers.append("invalid_hyperliquid_requested_leverage")
        if self.margin_mode not in {"cross", "isolated"}:
            blockers.append("invalid_hyperliquid_margin_mode")
        return blockers

    def submission_blockers(self) -> list[str]:
        blockers = self.configuration_blockers()
        if not self.submit_orders:
            blockers.append("hyperliquid_testnet_submit_orders_false")
        if not str(self.order_approval_id or "").strip():
            blockers.append("missing_explicit_hyperliquid_order_approval")
        if self.requested_leverage > 1:
            if not str(self.one_x_testnet_proof_id or "").strip():
                blockers.append("hyperliquid_one_x_testnet_proof_missing")
            if not str(self.leverage_scenario_id or "").strip():
                blockers.append("hyperliquid_leverage_scenario_approval_missing")
        return _unique(blockers)


@dataclass(frozen=True)
class HyperliquidPairExecutionResult:
    status: str
    reason: str
    fills: tuple[FillReport, ...] = ()
    recovery_actions: tuple[str, ...] = ()
    reconciled: bool = False
    order_submission_performed: bool = False
    live_trading_authorized: bool = False
    state_path: str = ""


def hyperliquid_sdk_installed() -> bool:
    return importlib.util.find_spec("hyperliquid") is not None


def read_hyperliquid_agent_key_from_keychain(service: str, account: str) -> str:
    """Read the dedicated Testnet agent key without logging or persisting it."""

    result = subprocess.run(
        ["security", "find-generic-password", "-w", "-s", service, "-a", account],
        capture_output=True,
        check=False,
        text=True,
    )
    secret = result.stdout.strip()
    if result.returncode != 0 or not secret:
        raise ValueError("hyperliquid_agent_keychain_entry_missing")
    return secret


class HyperliquidTestnetOrderAdapter:
    """Record-only single-leg adapter; pair routing must use the batch executor."""

    exchange_submission_capable = False
    record_only = True

    def place_order(self, intent: OrderIntent, config: object | None = None) -> FillReport:
        return FillReport(
            order_id="hyperliquid-testnet-pair-executor-required",
            market=str(intent.market),
            side=str(intent.side),
            size=float(intent.size),
            avg_price=float(intent.limit_price or 0.0),
            fee=0.0,
            slippage_bps=0.0,
            status="paper_blocked_hyperliquid_pair_executor_required",
        )


class HyperliquidTestnetPairAdapter:
    """Pair-capable adapter used by the common paper-execution router."""

    exchange_submission_capable = True
    pair_submission_capable = True
    record_only = False

    def __init__(self, executor: HyperliquidTestnetPairExecutor | None = None) -> None:
        self.executor = executor or HyperliquidTestnetPairExecutor(
            state_path=HYPERLIQUID_TESTNET_EXECUTION_STATE_JSON
        )

    def place_order(self, intent: OrderIntent, config: object | None = None) -> FillReport:
        return HyperliquidTestnetOrderAdapter().place_order(intent, config)

    def submit_pair(
        self,
        intents: Sequence[OrderIntent],
        config: object | None = None,
    ) -> HyperliquidPairExecutionResult:
        resolved = config if isinstance(config, HyperliquidTestnetConfig) else None
        return self.executor.submit_pair(intents, resolved)


class HyperliquidTestnetPairExecutor:
    """Testnet-only two-leg executor using Hyperliquid's official SDK bulk action."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        keychain_reader: Callable[[str, str], str] | None = None,
        exchange_factory: Callable[..., Any] | None = None,
        approval_validator: (
            Callable[[str, HyperliquidTestnetConfig], Sequence[str]] | None
        ) = None,
        state_path: Path | None = None,
    ) -> None:
        self._session = session or requests.Session()
        self._keychain_reader = keychain_reader or read_hyperliquid_agent_key_from_keychain
        self._exchange_factory = exchange_factory
        self._approval_validator = approval_validator or _signed_approval_blockers
        self._uses_default_approval_validator = approval_validator is None
        self._state_path = state_path

    def no_order_preflight(self, config: HyperliquidTestnetConfig | None = None) -> dict[str, object]:
        resolved = config or HyperliquidTestnetConfig.paper_testnet_from_env()
        blockers = list(resolved.configuration_blockers())
        result: dict[str, object] = {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "network": resolved.network,
            "testnet_base_url": resolved.base_url.rstrip("/") == HYPERLIQUID_TESTNET_URL,
            "master_address_configured": _valid_address(resolved.master_address),
            "agent_address_configured": _valid_address(resolved.agent_address),
            "keychain_service_configured": bool(str(resolved.keychain_service or "").strip()),
            "sdk_available": hyperliquid_sdk_installed(),
            "agent_key_present": False,
            "agent_key_matches_address": False,
            "agent_role": "",
            "agent_authorized_for_master": False,
            "local_signature_created": False,
            "submit_orders_enabled": bool(resolved.submit_orders),
            "order_submission_performed": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        if not bool(result["sdk_available"]):
            blockers.append("missing_hyperliquid_python_sdk")

        wallet = None
        if not blockers and resolved.keychain_service and resolved.agent_address:
            try:
                secret = self._keychain_reader(resolved.keychain_service, resolved.agent_address)
                result["agent_key_present"] = bool(secret)
                wallet = Account.from_key(secret)
                result["agent_key_matches_address"] = wallet.address.lower() == resolved.agent_address.lower()
                if not bool(result["agent_key_matches_address"]):
                    blockers.append("hyperliquid_agent_key_address_mismatch")
            except Exception:
                blockers.append("hyperliquid_agent_keychain_entry_missing")

        if resolved.agent_address and resolved.master_address and not any(
            blocker.startswith("missing_or_invalid") for blocker in blockers
        ):
            try:
                role_payload = self._info({"type": "userRole", "user": resolved.agent_address}, resolved)
                role = str(role_payload.get("role") or "")
                linked_master = str((role_payload.get("data") or {}).get("user") or "")
                result["agent_role"] = role
                result["agent_authorized_for_master"] = (
                    role == "agent" and linked_master.lower() == resolved.master_address.lower()
                )
                if not bool(result["agent_authorized_for_master"]):
                    blockers.append("hyperliquid_agent_authorization_unverified")
            except Exception:
                blockers.append("hyperliquid_agent_role_check_failed")

        if wallet is not None and not blockers and bool(result["sdk_available"]):
            try:
                self._create_local_noop_signature(wallet)
                result["local_signature_created"] = True
            except Exception:
                blockers.append("hyperliquid_local_signature_check_failed")

        blockers = _unique(blockers)
        result["blockers"] = ";".join(blockers)
        result["ready_for_no_order_preflight"] = not blockers
        result["ready_for_testnet_submit"] = False
        result["next_action"] = _next_action(result, resolved)
        return result

    def submit_pair(
        self,
        intents: Sequence[OrderIntent],
        config: HyperliquidTestnetConfig | None = None,
    ) -> HyperliquidPairExecutionResult:
        resolved = config or HyperliquidTestnetConfig.paper_testnet_from_env()
        constructor_blockers: list[str] = []
        if self._exchange_factory is None and self._state_path is None:
            constructor_blockers.append(
                "hyperliquid_real_exchange_requires_durable_execution_state"
            )
        if self._exchange_factory is None and not self._uses_default_approval_validator:
            constructor_blockers.append(
                "hyperliquid_real_exchange_custom_approval_validator_forbidden"
            )
        if constructor_blockers:
            return HyperliquidPairExecutionResult(
                status="pair_blocked",
                reason=";".join(constructor_blockers),
            )
        with self._exclusive_submission_lock() as lock_acquired:
            if not lock_acquired:
                return HyperliquidPairExecutionResult(
                    status="pair_blocked",
                    reason="hyperliquid_testnet_pair_executor_lock_present",
                    state_path=self._state_path_text(),
                )
            return self._submit_pair_locked(intents, resolved)

    def _submit_pair_locked(
        self,
        intents: Sequence[OrderIntent],
        resolved: HyperliquidTestnetConfig,
    ) -> HyperliquidPairExecutionResult:
        blockers = resolved.submission_blockers()
        blockers.extend(self._approval_blockers(intents, resolved))
        blockers.extend(_pair_intent_blockers(intents, resolved))
        prior_state = self._read_execution_state()
        state_blockers = self._submission_state_blockers(
            prior_state, intents, resolved
        )
        blockers.extend(state_blockers)
        if (
            "hyperliquid_duplicate_entry_approval_consumed" in state_blockers
            and prior_state.get("state_integrity_valid") is True
        ):
            prior_state["duplicate_entry_blocked"] = True
            prior_state["duplicate_entry_blocked_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            self._persist_execution_state(prior_state)
        blockers = _unique(blockers)
        if blockers:
            return HyperliquidPairExecutionResult(
                status="pair_blocked",
                reason=";".join(blockers),
            )

        preflight = self.no_order_preflight(resolved)
        if not bool(preflight.get("ready_for_no_order_preflight")):
            return HyperliquidPairExecutionResult(
                status="pair_blocked",
                reason=str(preflight.get("blockers") or "hyperliquid_no_order_preflight_failed"),
            )

        account_blockers = self._pair_account_blockers(intents, resolved)
        if account_blockers:
            return HyperliquidPairExecutionResult(
                status="pair_blocked",
                reason=";".join(account_blockers),
            )

        market_rule_blockers = self._pair_market_rule_blockers(intents, resolved)
        if market_rule_blockers:
            return HyperliquidPairExecutionResult(
                status="pair_blocked",
                reason=";".join(market_rule_blockers),
            )

        exit_price_blockers = self._pair_exit_price_blockers(intents, resolved)
        if exit_price_blockers:
            return HyperliquidPairExecutionResult(
                status="pair_blocked",
                reason=";".join(exit_price_blockers),
            )

        final_blockers = self._approval_blockers(intents, resolved)
        final_blockers.extend(self._pair_exit_price_blockers(intents, resolved))
        latest_state = self._read_execution_state()
        final_blockers.extend(
            self._submission_state_blockers(latest_state, intents, resolved)
        )
        final_blockers = _unique(final_blockers)
        if final_blockers:
            return HyperliquidPairExecutionResult(
                status="pair_blocked",
                reason=";".join(final_blockers),
                state_path=self._state_path_text(),
            )
        prior_state = latest_state

        state = self._new_execution_state(intents, resolved)
        if (
            all(intent.reduce_only for intent in intents)
            and prior_state.get("state_integrity_valid") is True
            and prior_state.get("order_approval_id") == resolved.order_approval_id
        ):
            state["entry_submit_attempted"] = bool(
                prior_state.get("entry_submit_attempted", False)
            )
            state["entry_execution_state_id"] = str(
                prior_state.get("execution_state_id", "")
            )
            state["duplicate_entry_blocked"] = bool(
                prior_state.get("duplicate_entry_blocked", False)
            )
            state["duplicate_entry_blocked_at_utc"] = str(
                prior_state.get("duplicate_entry_blocked_at_utc", "")
            )
        self._persist_execution_state(state)
        try:
            wallet = self._load_wallet(resolved)
            exchange = self._build_exchange(wallet, resolved)
        except Exception as exc:
            state["phase"] = "PRE_SUBMISSION_FAILED"
            state["blocker"] = f"hyperliquid_pair_setup_error:{type(exc).__name__}"
            self._persist_execution_state(state)
            return HyperliquidPairExecutionResult(
                status="pair_submission_error",
                reason=str(state["blocker"]),
                state_path=self._state_path_text(),
            )
        try:
            if not all(intent.reduce_only for intent in intents):
                for intent in intents:
                    exchange.update_leverage(
                        resolved.requested_leverage,
                        _normalize_perp_coin(intent.market),
                        is_cross=resolved.margin_mode == "cross",
                    )
        except Exception as exc:
            state["phase"] = "PRE_SUBMISSION_FAILED"
            state["blocker"] = (
                f"hyperliquid_pair_leverage_configuration_error:{type(exc).__name__}"
            )
            self._persist_execution_state(state)
            return HyperliquidPairExecutionResult(
                status="pair_configuration_error",
                reason=str(state["blocker"]),
                state_path=self._state_path_text(),
            )

        submission_blockers = self._approval_blockers(intents, resolved)
        submission_blockers.extend(
            self._pair_exit_price_blockers(intents, resolved)
        )
        latest_state = self._read_execution_state()
        state_unchanged = bool(
            self._state_path is None
            or (
                latest_state.get("state_integrity_valid") is True
                and latest_state.get("execution_state_id")
                == state.get("execution_state_id")
            )
        )
        if not state_unchanged:
            submission_blockers.append(
                "hyperliquid_pair_execution_state_changed_before_submission"
            )
        submission_blockers = _unique(submission_blockers)
        if submission_blockers:
            if state_unchanged:
                state["phase"] = "PRE_SUBMISSION_FAILED"
                state["blocker"] = ";".join(submission_blockers)
                self._persist_execution_state(state)
            return HyperliquidPairExecutionResult(
                status="pair_blocked",
                reason=";".join(submission_blockers),
                state_path=self._state_path_text(),
            )

        requests = [self._order_request(intent) for intent in intents]
        state["phase"] = "SUBMITTING_UNCONFIRMED"
        if all(intent.reduce_only for intent in intents):
            state["exit_submit_attempted"] = True
        else:
            state["entry_submit_attempted"] = True
        state["submit_attempted_at_utc"] = datetime.now(timezone.utc).isoformat()
        self._persist_execution_state(state)
        try:
            response = exchange.bulk_orders(requests)
        except Exception as exc:
            state["phase"] = "RECONCILE_REQUIRED"
            state["blocker"] = (
                f"hyperliquid_bulk_response_unknown:{type(exc).__name__}"
            )
            self._persist_execution_state(state)
            recovery = self._recover_pair_to_flat(
                exchange=exchange,
                intents=intents,
                fills=(),
                config=resolved,
            )
            return self._recovery_result(
                recovery=recovery,
                state=state,
                fills=(),
                recovered_reason="hyperliquid_unknown_submission_recovered_flat",
            )

        fills = _fills_from_response(response, intents)
        state["exchange_reference_ids"] = [
            fill.order_id
            for fill in fills
            if fill.order_id and fill.order_id != "hyperliquid-testnet-bulk-order"
        ]
        state["exchange_references"] = [
            {"market": fill.market, "order_id": fill.order_id}
            for fill in fills
            if fill.order_id and fill.order_id != "hyperliquid-testnet-bulk-order"
        ]
        confirmed = sum(fill.status == "paper_submitted" for fill in fills)
        if confirmed == len(intents):
            status = "pair_submitted"
            reason = "bulk_order_submitted_pending_exchange_confirmation"
        elif confirmed:
            status = "pair_partial"
            reason = "hyperliquid_pair_partial_manual_recovery_required"
        elif any(fill.status == "broadcast_accepted_unconfirmed" for fill in fills):
            status = "pair_unconfirmed"
            reason = "hyperliquid_bulk_response_unconfirmed"
        else:
            status = "pair_rejected"
            reason = "exchange_rejected_bulk_order"
        recovery_actions: tuple[str, ...] = ()
        reconciled = False
        if status in {"pair_partial", "pair_unconfirmed"}:
            recovery = self._recover_pair_to_flat(
                exchange=exchange,
                intents=intents,
                fills=fills,
                config=resolved,
            )
            recovery_actions = tuple(recovery["actions"])
            reconciled = bool(recovery["reconciled"])
            if reconciled:
                status = "pair_recovered_flat"
                reason = "hyperliquid_pair_submission_anomaly_recovered_flat"
            else:
                status = "pair_recovery_failed"
                reason = str(recovery["blocker"])
        state["phase"] = _execution_phase(status, reconciled)
        state["blocker"] = "" if reconciled or status == "pair_submitted" else reason
        state["recovery_actions"] = list(recovery_actions)
        state["reconciled"] = reconciled
        self._persist_execution_state(state)
        return HyperliquidPairExecutionResult(
            status=status,
            reason=reason,
            fills=fills,
            recovery_actions=recovery_actions,
            reconciled=reconciled,
            order_submission_performed=True,
            live_trading_authorized=False,
            state_path=self._state_path_text(),
        )

    def recover_incomplete_pair(
        self,
        config: HyperliquidTestnetConfig | None = None,
    ) -> HyperliquidPairExecutionResult:
        """Recover a journaled uncertain pair without ever retrying its entry."""

        with self._exclusive_submission_lock() as lock_acquired:
            if not lock_acquired:
                return HyperliquidPairExecutionResult(
                    status="pair_recovery_blocked",
                    reason="hyperliquid_testnet_pair_executor_lock_present",
                    state_path=self._state_path_text(),
                )
            return self._recover_incomplete_pair_locked(config)

    def _recover_incomplete_pair_locked(
        self,
        config: HyperliquidTestnetConfig | None = None,
    ) -> HyperliquidPairExecutionResult:

        state = self._read_execution_state()
        if not state:
            return HyperliquidPairExecutionResult(
                status="pair_recovery_blocked",
                reason="hyperliquid_pair_execution_state_missing",
                state_path=self._state_path_text(),
            )
        if str(state.get("phase", "")) in {
            "FLAT_RECONCILED",
            "PRE_SUBMISSION_FAILED",
            "REJECTED",
        }:
            return HyperliquidPairExecutionResult(
                status="pair_recovery_not_required",
                reason="hyperliquid_pair_execution_state_terminal",
                reconciled=bool(state.get("reconciled", False)),
                state_path=self._state_path_text(),
            )
        resolved = config or HyperliquidTestnetConfig.paper_testnet_from_env()
        blockers = resolved.configuration_blockers()
        if not resolved.submit_orders:
            blockers.append("hyperliquid_testnet_submit_orders_false")
        if not str(resolved.order_approval_id or "").strip():
            blockers.append("missing_explicit_hyperliquid_order_approval")
        if state.get("state_integrity_valid") is not True:
            blockers.append("hyperliquid_pair_execution_state_hash_invalid")
        if state.get("approval_validated_at_entry") is not True:
            blockers.append("hyperliquid_recovery_entry_approval_not_recorded")
        if state.get("entry_submit_attempted") is not True:
            blockers.append("hyperliquid_recovery_entry_attempt_not_recorded")
        if state.get("order_approval_id") != resolved.order_approval_id:
            blockers.append("hyperliquid_recovery_approval_id_mismatch")
        if state.get("master_address") != resolved.master_address:
            blockers.append("hyperliquid_recovery_master_address_mismatch")
        if state.get("agent_address") != resolved.agent_address:
            blockers.append("hyperliquid_recovery_agent_address_mismatch")
        intents = _intents_from_execution_state(state)
        if len(intents) != 2:
            blockers.append("hyperliquid_recovery_intent_state_invalid")
        blockers = _unique(blockers)
        if blockers:
            return HyperliquidPairExecutionResult(
                status="pair_recovery_blocked",
                reason=";".join(blockers),
                state_path=self._state_path_text(),
            )
        preflight = self.no_order_preflight(resolved)
        if not bool(preflight.get("ready_for_no_order_preflight")):
            return HyperliquidPairExecutionResult(
                status="pair_recovery_blocked",
                reason=str(
                    preflight.get("blockers")
                    or "hyperliquid_no_order_preflight_failed"
                ),
                state_path=self._state_path_text(),
            )
        try:
            wallet = self._load_wallet(resolved)
            exchange = self._build_exchange(wallet, resolved)
        except Exception as exc:
            return HyperliquidPairExecutionResult(
                status="pair_recovery_failed",
                reason=f"hyperliquid_pair_recovery_setup_error:{type(exc).__name__}",
                state_path=self._state_path_text(),
            )
        reference_rows = (
            state.get("exchange_references")
            if isinstance(state.get("exchange_references"), list)
            else []
        )
        intents_by_market = {
            _normalize_perp_coin(intent.market): intent for intent in intents
        }
        synthetic_fills = tuple(
            FillReport(
                order_id=str(reference.get("order_id", "")),
                market=str(reference.get("market", "")),
                side=intents_by_market[
                    _normalize_perp_coin(str(reference.get("market", "")))
                ].side,
                size=intents_by_market[
                    _normalize_perp_coin(str(reference.get("market", "")))
                ].size,
                avg_price=float(
                    intents_by_market[
                        _normalize_perp_coin(str(reference.get("market", "")))
                    ].limit_price
                    or 0.0
                ),
                fee=0.0,
                slippage_bps=0.0,
                status="broadcast_accepted_unconfirmed",
            )
            for reference in reference_rows
            if isinstance(reference, dict)
            and _normalize_perp_coin(str(reference.get("market", "")))
            in intents_by_market
        )
        recovery = self._recover_pair_to_flat(
            exchange=exchange,
            intents=intents,
            fills=synthetic_fills,
            config=resolved,
        )
        return self._recovery_result(
            recovery=recovery,
            state=state,
            fills=synthetic_fills,
            recovered_reason="hyperliquid_restart_recovered_flat",
        )

    def _new_execution_state(
        self,
        intents: Sequence[OrderIntent],
        config: HyperliquidTestnetConfig,
    ) -> dict[str, object]:
        created_at = datetime.now(timezone.utc).isoformat()
        intent_rows = [
            {
                "market": _normalize_perp_coin(intent.market),
                "side": str(intent.side).upper(),
                "size": float(intent.size),
                "limit_price": float(intent.limit_price or 0.0),
                "reduce_only": bool(intent.reduce_only),
            }
            for intent in intents
        ]
        identity = {
            "created_at_utc": created_at,
            "order_approval_id": str(config.order_approval_id or ""),
            "approval_validated_at_entry": True,
            "intents": intent_rows,
        }
        return {
            "schema_version": "hyperliquid_testnet_pair_execution_state.v1",
            "execution_state_id": "hltestnetstate_"
            + sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:20],
            "created_at_utc": created_at,
            "updated_at_utc": created_at,
            "network": config.network,
            "master_address": config.master_address,
            "agent_address": config.agent_address,
            "order_approval_id": str(config.order_approval_id or ""),
            "approval_validated_at_entry": True,
            "requested_leverage": int(config.requested_leverage),
            "margin_mode": config.margin_mode,
            "one_x_testnet_proof_id": str(config.one_x_testnet_proof_id or ""),
            "leverage_scenario_id": str(config.leverage_scenario_id or ""),
            "intents": intent_rows,
            "submit_kind": "exit" if all(intent.reduce_only for intent in intents) else "entry",
            "phase": "INTENT_PREPARED",
            "entry_submit_attempted": False,
            "exit_submit_attempted": False,
            "entry_execution_state_id": "",
            "duplicate_entry_blocked": False,
            "duplicate_entry_blocked_at_utc": "",
            "submit_attempted_at_utc": "",
            "exchange_reference_ids": [],
            "exchange_references": [],
            "recovery_actions": [],
            "reconciled": False,
            "blocker": "",
            "live_trading_authorized": False,
        }

    def _approval_blockers(
        self,
        intents: Sequence[OrderIntent],
        config: HyperliquidTestnetConfig,
    ) -> list[str]:
        if not config.order_approval_id:
            return []
        is_exit = all(intent.reduce_only for intent in intents)
        blockers = (
            _signed_approval_blockers(
                config.order_approval_id,
                config,
                validation_scope="exit" if is_exit else "entry",
            )
            if self._uses_default_approval_validator
            else list(self._approval_validator(config.order_approval_id, config))
        )
        if self._uses_default_approval_validator and not blockers:
            blockers.extend(
                _signed_approval_intent_blockers(
                    config.order_approval_id,
                    intents,
                )
            )
        return _unique(blockers)

    def _submission_state_blockers(
        self,
        state: dict[str, object],
        intents: Sequence[OrderIntent],
        config: HyperliquidTestnetConfig,
    ) -> list[str]:
        is_entry = not all(intent.reduce_only for intent in intents)
        if not state:
            return [] if is_entry else ["hyperliquid_exit_entry_state_missing"]
        if state.get("state_integrity_valid") is not True:
            return ["hyperliquid_pair_execution_state_hash_invalid"]
        prior_approval = str(state.get("order_approval_id", ""))
        current_approval = str(config.order_approval_id or "")
        if (
            is_entry
            and prior_approval == current_approval
            and state.get("entry_submit_attempted") is True
        ):
            return ["hyperliquid_duplicate_entry_approval_consumed"]
        if not is_entry:
            if prior_approval != current_approval:
                return ["hyperliquid_exit_approval_does_not_match_entry_state"]
            if state.get("entry_submit_attempted") is not True:
                return ["hyperliquid_exit_requires_recorded_entry_attempt"]
            if state.get("exit_submit_attempted") is True:
                return ["hyperliquid_duplicate_exit_submission_blocked"]
            if str(state.get("phase", "")) != "AWAITING_EXCHANGE_CONFIRMATION":
                return ["hyperliquid_exit_requires_open_lifecycle_state"]
        terminal_phases = {"FLAT_RECONCILED", "PRE_SUBMISSION_FAILED", "REJECTED"}
        if (
            prior_approval != current_approval
            and str(state.get("phase", "")) not in terminal_phases
        ):
            return ["hyperliquid_previous_pair_lifecycle_not_terminal"]
        return []

    def _recovery_result(
        self,
        *,
        recovery: dict[str, object],
        state: dict[str, object],
        fills: Sequence[FillReport],
        recovered_reason: str,
    ) -> HyperliquidPairExecutionResult:
        actions = tuple(str(value) for value in recovery.get("actions", []))
        reconciled = bool(recovery.get("reconciled", False))
        state["phase"] = "FLAT_RECONCILED" if reconciled else "RECOVERY_FAILED"
        state["recovery_actions"] = list(actions)
        state["reconciled"] = reconciled
        state["blocker"] = "" if reconciled else str(recovery.get("blocker", ""))
        self._persist_execution_state(state)
        return HyperliquidPairExecutionResult(
            status="pair_recovered_flat" if reconciled else "pair_recovery_failed",
            reason=recovered_reason if reconciled else str(recovery.get("blocker", "")),
            fills=tuple(fills),
            recovery_actions=actions,
            reconciled=reconciled,
            order_submission_performed=True,
            live_trading_authorized=False,
            state_path=self._state_path_text(),
        )

    def _persist_execution_state(self, state: dict[str, object]) -> None:
        if self._state_path is None:
            return
        payload = dict(state)
        payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        payload.pop("state_integrity_valid", None)
        payload.pop("state_hash", None)
        payload["state_hash"] = sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(self._state_path)

    @contextmanager
    def _exclusive_submission_lock(self):
        if self._state_path is None:
            yield True
            return
        lock_path = self._state_path.with_suffix(self._state_path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_execution_state(self) -> dict[str, object]:
        if self._state_path is None or not self._state_path.is_file():
            return {}
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"state_integrity_valid": False}
        if not isinstance(payload, dict):
            return {"state_integrity_valid": False}
        supplied_hash = str(payload.pop("state_hash", ""))
        expected_hash = sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
        payload["state_hash"] = supplied_hash
        payload["state_integrity_valid"] = bool(
            supplied_hash and supplied_hash == expected_hash
        )
        return payload

    def _state_path_text(self) -> str:
        return str(self._state_path) if self._state_path is not None else ""

    def _pair_account_blockers(
        self,
        intents: Sequence[OrderIntent],
        config: HyperliquidTestnetConfig,
    ) -> list[str]:
        coins = {_normalize_perp_coin(intent.market) for intent in intents}
        try:
            state = self._info(
                {"type": "clearinghouseState", "user": config.master_address},
                config,
            )
            open_orders = self._info_value(
                {"type": "openOrders", "user": config.master_address},
                config,
            )
        except Exception:
            return ["hyperliquid_pair_account_state_precheck_failed"]
        if not isinstance(state.get("assetPositions"), list):
            return ["hyperliquid_pair_account_positions_missing"]
        if not isinstance(open_orders, list):
            return ["hyperliquid_pair_open_orders_missing"]
        blockers: list[str] = []
        positions = _pair_position_sizes(state, coins)
        is_exit = all(intent.reduce_only for intent in intents)
        if not is_exit and any(
            abs(size) > 0.0 for size in positions.values()
        ):
            blockers.append("hyperliquid_pair_markets_not_flat_before_entry")
        if is_exit:
            for intent in intents:
                coin = _normalize_perp_coin(intent.market)
                position = float(positions.get(coin, 0.0))
                side = str(intent.side).strip().upper()
                if abs(position) <= 0.0:
                    blockers.append(f"hyperliquid_exit_position_missing:{coin}")
                    continue
                expected_side = "SELL" if position > 0.0 else "BUY"
                if side != expected_side:
                    blockers.append(f"hyperliquid_exit_side_not_risk_reducing:{coin}")
                if float(intent.size) > abs(position) + 1e-12:
                    blockers.append(f"hyperliquid_exit_size_exceeds_open_position:{coin}")
        if _pair_open_orders(open_orders, coins):
            blockers.append("hyperliquid_pair_open_orders_exist_before_submit")
        return blockers

    def _pair_market_rule_blockers(
        self,
        intents: Sequence[OrderIntent],
        config: HyperliquidTestnetConfig,
    ) -> list[str]:
        try:
            meta = self._info({"type": "meta"}, config)
        except Exception:
            return ["hyperliquid_testnet_meta_query_failed"]
        universe = meta.get("universe")
        if not isinstance(universe, list):
            return ["hyperliquid_testnet_meta_universe_missing"]
        markets = {
            _normalize_perp_coin(str(row.get("name", ""))): row
            for row in universe
            if isinstance(row, dict) and str(row.get("name", "")).strip()
        }
        blockers: list[str] = []
        for intent in intents:
            coin = _normalize_perp_coin(intent.market)
            market = markets.get(coin)
            if market is None:
                blockers.append(f"hyperliquid_testnet_market_missing:{coin}")
                continue
            if _env_like_truthy(market.get("isDelisted", False)):
                blockers.append(f"hyperliquid_testnet_market_delisted:{coin}")
            sz_decimals = _safe_nonnegative_int(market.get("szDecimals"))
            max_leverage = _safe_positive_int(market.get("maxLeverage"))
            if sz_decimals is None:
                blockers.append(f"hyperliquid_sz_decimals_missing:{coin}")
                continue
            if max_leverage is None:
                blockers.append(f"hyperliquid_max_leverage_missing:{coin}")
            elif config.requested_leverage > max_leverage:
                blockers.append(f"hyperliquid_requested_leverage_above_market_max:{coin}")
            if _env_like_truthy(market.get("onlyIsolated", False)) and config.margin_mode != "isolated":
                blockers.append(f"hyperliquid_market_requires_isolated_margin:{coin}")
            if not _size_precision_valid(intent.size, sz_decimals):
                blockers.append(f"hyperliquid_size_precision_invalid:{coin}")
            if not _price_precision_valid(intent.limit_price, sz_decimals):
                blockers.append(f"hyperliquid_price_precision_invalid:{coin}")
            notional = float(intent.size) * float(intent.limit_price or 0.0)
            if notional < 10.0:
                blockers.append(f"hyperliquid_leg_notional_below_10_usd:{coin}")
        return _unique(blockers)

    def _pair_exit_price_blockers(
        self,
        intents: Sequence[OrderIntent],
        config: HyperliquidTestnetConfig,
    ) -> list[str]:
        if not all(intent.reduce_only for intent in intents):
            return []
        if not self._uses_default_approval_validator:
            return []
        slippage_bps = _signed_exit_slippage_bps(config.order_approval_id)
        if slippage_bps is None:
            return ["hyperliquid_runtime_exit_slippage_policy_invalid"]
        try:
            mids = self._info({"type": "allMids"}, config)
        except Exception:
            return ["hyperliquid_runtime_exit_mids_unavailable"]
        return _exit_price_band_blockers(intents, mids, slippage_bps)

    def _recover_pair_to_flat(
        self,
        *,
        exchange: Any,
        intents: Sequence[OrderIntent],
        fills: Sequence[FillReport],
        config: HyperliquidTestnetConfig,
    ) -> dict[str, object]:
        coins = {_normalize_perp_coin(intent.market) for intent in intents}
        actions: list[str] = ["block_duplicate_entry_retry"]
        try:
            open_orders = self._info_value(
                {"type": "openOrders", "user": config.master_address},
                config,
            )
            if not isinstance(open_orders, list):
                raise ValueError("hyperliquid_pair_open_orders_missing")
            cancel_requests = _recovery_cancel_requests(fills, open_orders, coins)
            if cancel_requests:
                exchange.bulk_cancel(cancel_requests)
                actions.append("cancel_pair_open_orders")
            state = self._info(
                {"type": "clearinghouseState", "user": config.master_address},
                config,
            )
            positions = _pair_position_sizes(state, coins)
            for coin, size in positions.items():
                if abs(size) <= 0.0:
                    continue
                exchange.market_close(coin, sz=abs(size))
                actions.append(f"reduce_only_flatten:{coin}")
            final_state = self._info(
                {"type": "clearinghouseState", "user": config.master_address},
                config,
            )
            final_orders = self._info_value(
                {"type": "openOrders", "user": config.master_address},
                config,
            )
            if not isinstance(final_orders, list):
                raise ValueError("hyperliquid_pair_open_orders_missing")
            final_positions = _pair_position_sizes(final_state, coins)
            flat = all(abs(size) <= 0.0 for size in final_positions.values())
            no_orders = not _pair_open_orders(final_orders, coins)
            if flat and no_orders:
                actions.append("confirm_pair_flat_and_order_free")
                return {"reconciled": True, "actions": actions, "blocker": ""}
            blockers = []
            if not flat:
                blockers.append("hyperliquid_pair_orphan_position_remains")
            if not no_orders:
                blockers.append("hyperliquid_pair_open_order_remains")
            return {
                "reconciled": False,
                "actions": actions,
                "blocker": ";".join(blockers),
            }
        except Exception as exc:
            return {
                "reconciled": False,
                "actions": actions,
                "blocker": f"hyperliquid_pair_recovery_error:{type(exc).__name__}",
            }

    def _info(self, payload: dict[str, object], config: HyperliquidTestnetConfig) -> dict[str, Any]:
        parsed = self._info_value(payload, config)
        if not isinstance(parsed, dict):
            raise ValueError("unexpected_hyperliquid_info_response")
        return parsed

    def _info_value(
        self,
        payload: dict[str, object],
        config: HyperliquidTestnetConfig,
    ) -> Any:
        response = self._session.post(f"{config.base_url.rstrip('/')}/info", json=payload, timeout=20)
        response.raise_for_status()
        parsed = response.json()
        return parsed

    @staticmethod
    def _create_local_noop_signature(wallet: Any) -> None:
        from hyperliquid.utils.signing import sign_l1_action

        signature = sign_l1_action(wallet, {"type": "noop"}, None, 1, None, False)
        if not isinstance(signature, dict) or not signature:
            raise ValueError("invalid_hyperliquid_local_signature")

    def _load_wallet(self, config: HyperliquidTestnetConfig) -> Any:
        if not config.keychain_service or not config.agent_address:
            raise ValueError("hyperliquid_agent_configuration_missing")
        return Account.from_key(self._keychain_reader(config.keychain_service, config.agent_address))

    def _build_exchange(self, wallet: Any, config: HyperliquidTestnetConfig) -> Any:
        if self._exchange_factory is not None:
            return self._exchange_factory(wallet, config)
        from hyperliquid.exchange import Exchange

        return Exchange(wallet, base_url=config.base_url, account_address=config.master_address)

    @staticmethod
    def _order_request(intent: OrderIntent) -> dict[str, object]:
        return {
            "coin": _normalize_perp_coin(intent.market),
            "is_buy": str(intent.side).upper() == "BUY",
            "sz": float(intent.size),
            "limit_px": float(intent.limit_price or 0.0),
            "order_type": {"limit": {"tif": "Ioc"}},
            "reduce_only": bool(intent.reduce_only),
        }


def hyperliquid_testnet_no_order_preflight(
    config: HyperliquidTestnetConfig | None = None,
    *,
    session: requests.Session | None = None,
    keychain_reader: Callable[[str, str], str] | None = None,
) -> dict[str, object]:
    executor = HyperliquidTestnetPairExecutor(session=session, keychain_reader=keychain_reader)
    return executor.no_order_preflight(config)


def write_hyperliquid_testnet_preflight_report(
    root: Path = ROOT,
    config: HyperliquidTestnetConfig | None = None,
    *,
    session: requests.Session | None = None,
    keychain_reader: Callable[[str, str], str] | None = None,
) -> pd.DataFrame:
    result = hyperliquid_testnet_no_order_preflight(config, session=session, keychain_reader=keychain_reader)
    frame = pd.DataFrame([result])
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    csv_path = active / HYPERLIQUID_TESTNET_PRECHECK_CSV.name
    md_path = active / HYPERLIQUID_TESTNET_PRECHECK_MD.name
    frame.to_csv(csv_path, index=False)
    lines = [
        "# Hyperliquid Testnet Preflight",
        "",
        "This report performs account, authorization, and local-signing checks only. It never submits an order.",
        "",
        frame.to_markdown(index=False),
        "",
        "The agent private key is never written to this report.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return frame


def hyperliquid_testnet_margin_snapshot(
    config: HyperliquidTestnetConfig | None = None,
    *,
    session: requests.Session | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Read Testnet clearinghouse state without loading a key or submitting."""

    resolved = config or HyperliquidTestnetConfig.paper_testnet_from_env()
    checked_at = now or datetime.now(timezone.utc)
    blockers: list[str] = []
    if resolved.network != "testnet":
        blockers.append("unsafe_non_testnet_network")
    if resolved.base_url.rstrip("/") != HYPERLIQUID_TESTNET_URL:
        blockers.append("unsafe_non_testnet_base_url")
    if not _valid_address(resolved.master_address):
        blockers.append("missing_or_invalid_hyperliquid_master_address")
    payload: dict[str, Any] = {}
    spot_payload: dict[str, Any] = {}
    if not blockers:
        client = session or requests.Session()
        try:
            response = client.post(
                f"{resolved.base_url.rstrip('/')}/info",
                json={"type": "clearinghouseState", "user": resolved.master_address},
                timeout=20,
            )
            response.raise_for_status()
            parsed = response.json()
            if not isinstance(parsed, dict):
                raise ValueError("unexpected_hyperliquid_clearinghouse_response")
            payload = parsed
        except Exception:
            blockers.append("hyperliquid_testnet_margin_query_failed")
        try:
            response = client.post(
                f"{resolved.base_url.rstrip('/')}/info",
                json={"type": "spotClearinghouseState", "user": resolved.master_address},
                timeout=20,
            )
            response.raise_for_status()
            parsed = response.json()
            if not isinstance(parsed, dict):
                raise ValueError("unexpected_hyperliquid_spot_clearinghouse_response")
            spot_payload = parsed
        except Exception:
            blockers.append("hyperliquid_testnet_spot_balance_query_failed")

    summary = payload.get("marginSummary") if isinstance(payload.get("marginSummary"), dict) else {}
    account_value = _safe_nonnegative_float(summary.get("accountValue"))
    margin_used = _safe_nonnegative_float(summary.get("totalMarginUsed"))
    total_notional = _safe_nonnegative_float(summary.get("totalNtlPos"))
    withdrawable = _safe_nonnegative_float(payload.get("withdrawable"))
    spot_usdc = _spot_balance(spot_payload, "USDC")
    if payload and account_value is None:
        blockers.append("hyperliquid_testnet_account_value_missing")
    if payload and margin_used is None:
        blockers.append("hyperliquid_testnet_margin_used_missing")
    margin_buffer = None
    margin_utilization = None
    if account_value is not None and margin_used is not None:
        if account_value <= 0.0:
            if spot_usdc is not None and spot_usdc > 0.0:
                blockers.append("testnet_usdc_requires_spot_to_perp_transfer")
            else:
                blockers.append("hyperliquid_testnet_account_value_not_positive")
        else:
            margin_buffer = max(account_value - margin_used, 0.0) / account_value
            margin_utilization = margin_used / account_value
    positions = payload.get("assetPositions") if isinstance(payload.get("assetPositions"), list) else []
    open_positions = sum(1 for position in positions if _position_is_open(position))
    blockers = _unique(blockers)
    return {
        "checked_at_utc": checked_at.astimezone(timezone.utc).isoformat(),
        "network": resolved.network,
        "master_address": _masked_address(resolved.master_address),
        "account_value_usd": account_value,
        "margin_used_usd": margin_used,
        "total_notional_usd": total_notional,
        "withdrawable_usd": withdrawable,
        "spot_usdc_usd": spot_usdc,
        "margin_buffer": margin_buffer,
        "margin_utilization": margin_utilization,
        "open_positions": open_positions,
        "source_system": "hyperliquid_testnet_clearinghouse_state",
        "point_in_time_status": "confirmed" if not blockers else "blocked",
        "status": "READY" if not blockers else "BLOCKED",
        "blockers": ";".join(blockers),
    }


def write_hyperliquid_testnet_margin_snapshot(
    root: Path = ROOT,
    config: HyperliquidTestnetConfig | None = None,
    *,
    session: requests.Session | None = None,
    now: datetime | None = None,
) -> pd.DataFrame:
    result = hyperliquid_testnet_margin_snapshot(config, session=session, now=now)
    frame = pd.DataFrame([result])
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    csv_path = active / HYPERLIQUID_TESTNET_MARGIN_CSV.name
    md_path = active / HYPERLIQUID_TESTNET_MARGIN_MD.name
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(csv_path)
    md_path.write_text(
        "\n".join(
            [
                "# Hyperliquid Testnet Margin Snapshot",
                "",
                "This is a read-only clearinghouse-state query. It never loads a signing key or submits an order.",
                "",
                frame.to_markdown(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return frame


def _pair_intent_blockers(intents: Sequence[OrderIntent], config: HyperliquidTestnetConfig) -> list[str]:
    blockers: list[str] = []
    if len(intents) != 2:
        return ["hyperliquid_pair_executor_requires_exactly_two_legs"]
    coins = [_normalize_perp_coin(intent.market) for intent in intents]
    if len(set(coins)) != 2:
        blockers.append("hyperliquid_pair_legs_must_use_distinct_markets")
    sides = [str(intent.side).upper() for intent in intents]
    if any(side not in {"BUY", "SELL"} for side in sides):
        blockers.append("hyperliquid_pair_side_invalid")
    if not all(intent.reduce_only for intent in intents) and sides[0] == sides[1]:
        blockers.append("hyperliquid_pair_entry_legs_must_be_opposite")
    total_notional = 0.0
    for intent in intents:
        if float(intent.size) <= 0:
            blockers.append("hyperliquid_pair_size_must_be_positive")
        if intent.limit_price is None or float(intent.limit_price) <= 0:
            blockers.append("hyperliquid_pair_limit_price_required")
        else:
            total_notional += float(intent.size) * float(intent.limit_price)
    if total_notional > config.max_pair_notional_usd:
        blockers.append("hyperliquid_pair_notional_exceeds_testnet_cap")
    return _unique(blockers)


def _fills_from_response(response: object, intents: Sequence[OrderIntent]) -> tuple[FillReport, ...]:
    leg_statuses = _response_leg_statuses(response)
    return tuple(
        _fill_from_response(
            response,
            intent,
            leg_statuses[index] if index < len(leg_statuses) else None,
        )
        for index, intent in enumerate(intents)
    )


def _fill_from_response(response: object, intent: OrderIntent, leg_status: object | None) -> FillReport:
    order_id = _response_order_id(leg_status) or "hyperliquid-testnet-bulk-order"
    overall_ok = isinstance(response, dict) and response.get("status") == "ok"
    if not overall_ok:
        status = "paper_submission_rejected"
    elif _response_error(leg_status):
        status = f"paper_submission_rejected:{_response_error(leg_status)}"
    elif _response_order_id(leg_status) or _response_fill_confirmation(leg_status):
        status = "paper_submitted"
    else:
        status = "broadcast_accepted_unconfirmed"
    return FillReport(
        order_id=order_id,
        market=_normalize_perp_coin(intent.market),
        side=str(intent.side).upper(),
        size=float(intent.size),
        avg_price=float(intent.limit_price or 0.0),
        fee=0.0,
        slippage_bps=0.0,
        status=status,
    )


def _response_leg_statuses(response: object) -> list[object]:
    if isinstance(response, dict):
        statuses = response.get("statuses")
        if isinstance(statuses, list):
            return statuses
        for value in response.values():
            nested = _response_leg_statuses(value)
            if nested:
                return nested
    return []


def _response_error(value: object) -> str:
    if isinstance(value, dict):
        error = value.get("error")
        if error not in (None, ""):
            return str(error)
        for nested in value.values():
            found = _response_error(nested)
            if found:
                return found
    return ""


def _response_fill_confirmation(value: object) -> bool:
    if isinstance(value, dict):
        if any(key in value for key in ("filled", "resting")):
            return True
        return any(_response_fill_confirmation(nested) for nested in value.values())
    return False


def _response_order_id(response: object) -> str:
    if isinstance(response, dict):
        for key in ("oid", "orderId", "cloid"):
            value = response.get(key)
            if value not in (None, ""):
                return str(value)
        for value in response.values():
            nested = _response_order_id(value)
            if nested:
                return nested
    if isinstance(response, list):
        for value in response:
            nested = _response_order_id(value)
            if nested:
                return nested
    return ""


def _pair_position_sizes(
    payload: dict[str, Any],
    coins: set[str],
) -> dict[str, float]:
    positions = {coin: 0.0 for coin in coins}
    rows = payload.get("assetPositions")
    if not isinstance(rows, list):
        raise ValueError("hyperliquid_pair_account_positions_missing")
    for row in rows:
        if not isinstance(row, dict):
            continue
        position = row.get("position") if isinstance(row.get("position"), dict) else row
        coin = _normalize_perp_coin(str(position.get("coin", "")))
        if coin not in coins:
            continue
        try:
            positions[coin] = float(position.get("szi", 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("hyperliquid_pair_position_size_invalid") from exc
    return positions


def _pair_open_orders(rows: Sequence[object], coins: set[str]) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            coin = _normalize_perp_coin(str(row.get("coin", "")))
        except ValueError:
            continue
        if coin in coins:
            matches.append(row)
    return matches


def _recovery_cancel_requests(
    fills: Sequence[FillReport],
    open_orders: Sequence[object],
    coins: set[str],
) -> list[dict[str, object]]:
    requests_by_key: dict[tuple[str, int], dict[str, object]] = {}
    for fill in fills:
        try:
            oid = int(str(fill.order_id))
        except (TypeError, ValueError):
            continue
        coin = _normalize_perp_coin(fill.market)
        if coin in coins:
            requests_by_key[(coin, oid)] = {"coin": coin, "oid": oid}
    for order in _pair_open_orders(open_orders, coins):
        try:
            coin = _normalize_perp_coin(str(order.get("coin", "")))
            oid = int(order.get("oid"))
        except (TypeError, ValueError):
            continue
        requests_by_key[(coin, oid)] = {"coin": coin, "oid": oid}
    return list(requests_by_key.values())


def _execution_phase(status: str, reconciled: bool) -> str:
    if reconciled:
        return "FLAT_RECONCILED"
    return {
        "pair_submitted": "AWAITING_EXCHANGE_CONFIRMATION",
        "pair_rejected": "REJECTED",
        "pair_recovery_failed": "RECOVERY_FAILED",
    }.get(status, "RECONCILE_REQUIRED")


def _intents_from_execution_state(state: dict[str, object]) -> tuple[OrderIntent, ...]:
    rows = state.get("intents") if isinstance(state.get("intents"), list) else []
    intents: list[OrderIntent] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            intents.append(
                OrderIntent(
                    market=str(row.get("market", "")),
                    side=str(row.get("side", "")),
                    size=float(row.get("size", 0.0) or 0.0),
                    limit_price=float(row.get("limit_price", 0.0) or 0.0),
                    reduce_only=bool(row.get("reduce_only", False)),
                )
            )
        except (TypeError, ValueError):
            return ()
    return tuple(intents)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _normalize_perp_coin(value: str) -> str:
    text = str(value or "").strip().upper().replace("_", "-").replace("/", "-")
    for suffix in ("-USD", "-USDC", "-USDT"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    if not text or not re.fullmatch(r"[A-Z0-9:]+", text):
        raise ValueError("invalid_hyperliquid_perp_market")
    return text


def _next_action(result: dict[str, object], config: HyperliquidTestnetConfig) -> str:
    blockers = str(result.get("blockers") or "")
    if blockers:
        return "repair_hyperliquid_testnet_preflight_blockers"
    if not config.submit_orders:
        return "keep_submission_disabled_until_a_specific_two_leg_test_is_approved"
    return "create_a_runtime_only_order_approval_before_any_two_leg_submission"


def _safe_nonnegative_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0.0 else None


def _position_is_open(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    position = value.get("position") if isinstance(value.get("position"), dict) else value
    try:
        return abs(float(position.get("szi", 0.0))) > 0.0
    except (TypeError, ValueError):
        return False


def _spot_balance(payload: dict[str, Any], coin: str) -> float | None:
    balances = payload.get("balances") if isinstance(payload.get("balances"), list) else []
    for balance in balances:
        if isinstance(balance, dict) and str(balance.get("coin", "")).upper() == coin.upper():
            return _safe_nonnegative_float(balance.get("total"))
    return None


def _masked_address(value: str | None) -> str:
    text = str(value or "")
    return f"{text[:6]}...{text[-4:]}" if _valid_address(text) else ""


def _valid_address(value: str | None) -> bool:
    return bool(ADDRESS_PATTERN.fullmatch(str(value or "").strip()))


def _env_value(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


def _env_positive_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _safe_nonnegative_int(value: object) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _safe_positive_int(value: object) -> int | None:
    number = _safe_nonnegative_int(value)
    return number if number is not None and number > 0 else None


def _size_precision_valid(value: object, sz_decimals: int) -> bool:
    try:
        size = Decimal(str(value))
    except InvalidOperation:
        return False
    if not size.is_finite() or size <= 0:
        return False
    decimal_places = max(-size.normalize().as_tuple().exponent, 0)
    return decimal_places <= sz_decimals


def _price_precision_valid(value: object, sz_decimals: int) -> bool:
    try:
        price = Decimal(str(value))
    except InvalidOperation:
        return False
    if not price.is_finite() or price <= 0:
        return False
    normalized = price.normalize()
    if normalized == normalized.to_integral_value():
        return True
    decimal_places = max(-normalized.as_tuple().exponent, 0)
    significant_digits = len(normalized.as_tuple().digits)
    return decimal_places <= 6 - sz_decimals and significant_digits <= 5


def _env_like_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _signed_approval_blockers(
    approval_id: str,
    config: HyperliquidTestnetConfig,
    *,
    validation_scope: str = "entry",
) -> list[str]:
    try:
        from quant_platform.orchestration.hyperliquid_learning_and_risk import (
            validate_testnet_smoke_approval,
        )

        result = validate_testnet_smoke_approval(
            approval_id=approval_id,
            root=ROOT,
            config=config,
            validation_scope=validation_scope,
        )
    except Exception:
        return ["hyperliquid_signed_approval_validation_failed"]
    blockers = result.get("blockers", [])
    resolved = (
        [str(blocker) for blocker in blockers]
        if isinstance(blockers, list)
        else ["hyperliquid_signed_approval_validation_failed"]
    )
    if not (
        result.get("status") == "PASS"
        and result.get("execution_allowed") is True
        and result.get("testnet_order_authority") is True
        and result.get("live_trading_authorized") is False
        and result.get("authority_scope") == validation_scope
    ):
        resolved.append("hyperliquid_testnet_explicit_order_authority_not_granted")
    return _unique(resolved)


def _signed_approval_intent_blockers(
    approval_id: str,
    intents: Sequence[OrderIntent],
) -> list[str]:
    path = ROOT / "reports" / "active" / "hyperliquid_testnet_smoke_approval.json"
    try:
        approval = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["hyperliquid_signed_approval_payload_unavailable"]
    if not isinstance(approval, dict) or str(approval.get("approval_id", "")) != str(
        approval_id
    ):
        return ["hyperliquid_signed_approval_payload_mismatch"]
    return _approval_intent_blockers(approval, intents)


def _signed_exit_slippage_bps(approval_id: str | None) -> float | None:
    path = ROOT / "reports" / "active" / "hyperliquid_testnet_smoke_approval.json"
    try:
        approval = json.loads(path.read_text(encoding="utf-8"))
        if str(approval.get("approval_id", "")) != str(approval_id or ""):
            return None
        value = float(approval.get("maximum_exit_slippage_bps"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if 0.0 < value <= 50.0 else None


def _exit_price_band_blockers(
    intents: Sequence[OrderIntent],
    mids: dict[str, object],
    maximum_slippage_bps: float,
) -> list[str]:
    blockers: list[str] = []
    if not 0.0 < maximum_slippage_bps <= 50.0:
        return ["hyperliquid_runtime_exit_slippage_policy_invalid"]
    band = maximum_slippage_bps / 10_000.0
    for index, intent in enumerate(intents):
        try:
            coin = _normalize_perp_coin(intent.market)
            mid = float(mids.get(coin, 0.0))
            price = float(intent.limit_price or 0.0)
        except (TypeError, ValueError):
            blockers.append(f"hyperliquid_runtime_exit_mid_or_price_invalid:{index}")
            continue
        if not all(math.isfinite(value) and value > 0.0 for value in (mid, price)):
            blockers.append(f"hyperliquid_runtime_exit_mid_or_price_invalid:{index}")
            continue
        side = str(intent.side).strip().upper()
        if side == "SELL":
            if price > mid:
                blockers.append(f"hyperliquid_runtime_exit_limit_not_marketable:{index}")
            if price < mid * (1.0 - band) - 1e-12:
                blockers.append(f"hyperliquid_runtime_exit_price_below_slippage_band:{index}")
        elif side == "BUY":
            if price < mid:
                blockers.append(f"hyperliquid_runtime_exit_limit_not_marketable:{index}")
            if price > mid * (1.0 + band) + 1e-12:
                blockers.append(f"hyperliquid_runtime_exit_price_above_slippage_band:{index}")
        else:
            blockers.append(f"hyperliquid_runtime_exit_side_invalid:{index}")
    return _unique(blockers)


def _approval_intent_blockers(
    approval: dict[str, object],
    intents: Sequence[OrderIntent],
) -> list[str]:
    legs = approval.get("legs") if isinstance(approval.get("legs"), list) else []
    if len(legs) != 2 or len(intents) != 2 or not all(
        isinstance(leg, dict) for leg in legs
    ):
        return ["hyperliquid_runtime_approval_leg_shape_mismatch"]
    reduce_only_values = {bool(intent.reduce_only) for intent in intents}
    if len(reduce_only_values) != 1:
        return ["hyperliquid_pair_reduce_only_must_match_across_legs"]

    is_exit = all(intent.reduce_only for intent in intents)
    blockers: list[str] = []
    for index, (leg, intent) in enumerate(zip(legs, intents, strict=True)):
        assert isinstance(leg, dict)
        try:
            approved_market = _normalize_perp_coin(str(leg.get("market", "")))
            runtime_market = _normalize_perp_coin(intent.market)
            approved_size = Decimal(str(leg.get("size", "")))
            runtime_size = Decimal(str(intent.size))
            approved_price = Decimal(str(leg.get("limit_price", "")))
            runtime_price = Decimal(str(intent.limit_price))
        except (InvalidOperation, TypeError, ValueError):
            blockers.append(f"hyperliquid_runtime_approval_leg_invalid:{index}")
            continue
        if not all(
            value.is_finite()
            for value in (approved_size, runtime_size, approved_price, runtime_price)
        ):
            blockers.append(f"hyperliquid_runtime_approval_leg_invalid:{index}")
            continue
        approved_side = str(leg.get("side", "")).strip().upper()
        runtime_side = str(intent.side).strip().upper()
        if runtime_market != approved_market:
            blockers.append(f"hyperliquid_runtime_approval_market_mismatch:{index}")
        if is_exit:
            expected_side = "SELL" if approved_side == "BUY" else "BUY"
            if runtime_side != expected_side:
                blockers.append(f"hyperliquid_runtime_exit_side_mismatch:{index}")
            if runtime_size <= 0 or runtime_size > approved_size:
                blockers.append(f"hyperliquid_runtime_exit_size_exceeds_approval:{index}")
        else:
            if runtime_side != approved_side:
                blockers.append(f"hyperliquid_runtime_entry_side_mismatch:{index}")
            if runtime_size != approved_size:
                blockers.append(f"hyperliquid_runtime_entry_size_mismatch:{index}")
            if runtime_price != approved_price:
                blockers.append(f"hyperliquid_runtime_entry_price_mismatch:{index}")
    if is_exit:
        expected_policy = {
            "reduce_only": True,
            "opposite_side": True,
            "market_scope": "approved_entry_markets",
            "maximum_size": "approved_entry_size_per_leg",
            "entry_notional_cap_not_reapplied_to_exit": True,
        }
        if approval.get("exit_policy") != expected_policy:
            blockers.append("hyperliquid_runtime_exit_policy_missing_or_invalid")
        try:
            maximum_exit_slippage_bps = Decimal(
                str(approval.get("maximum_exit_slippage_bps", ""))
            )
        except (InvalidOperation, TypeError, ValueError):
            maximum_exit_slippage_bps = Decimal("NaN")
        if (
            not maximum_exit_slippage_bps.is_finite()
            or maximum_exit_slippage_bps <= 0
            or maximum_exit_slippage_bps > Decimal(50)
        ):
            blockers.append("hyperliquid_runtime_exit_slippage_policy_invalid")
    return _unique(blockers)
