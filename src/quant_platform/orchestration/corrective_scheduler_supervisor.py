"""Deterministic supervisor for every governed Phase 00 scheduler invocation."""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from quant_platform.orchestration.corrective_effect_guard import (
    phase00_effect_guard,
)
from quant_platform.orchestration.corrective_external_effect_policy import (
    ExternalEffectPolicyError,
    Phase00ExternalEffectPolicy,
    ProviderExternalEffectPolicy,
    load_phase00_external_effect_policy,
)
from quant_platform.orchestration.corrective_external_effects import (
    ExternalEffectCallContract,
    ExternalEffectIssuer,
    external_effect_issuer_bundle_session,
    external_effect_issuer_session,
)
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import (
    RUNTIME_TIMEZONE,
    SchedulerContract,
    scheduler_contract,
    scheduler_run_identity,
)
from quant_platform.orchestration.corrective_scheduler_lock import (
    GovernedEvidenceLockBusy,
    GovernedEvidenceMaintenanceActive,
    governed_evidence_write_lock,
    scheduler_process_owner_matches,
)
from quant_platform.orchestration.corrective_scheduler_terminal import (
    SchedulerSlotAlreadyClaimed,
    build_scheduler_run_intent,
    build_scheduler_terminal_receipt,
    claim_scheduler_slot,
    load_validated_scheduler_run_intent,
    load_validated_scheduler_slot_claim,
    load_validated_scheduler_terminal_receipt_by_run,
    new_scheduler_run_id,
    publish_scheduler_run_intent,
    publish_scheduler_terminal_receipt,
    scheduler_terminal_exit_code,
    terminal_receipt_exists,
)
from quant_platform.orchestration.effect_authority import (
    PHASE00_REPAIR_PROFILE,
    PHASE00_WIZARD_RESEARCH_PROFILE,
    EffectAuthority,
    publication_authority_session,
)

SCHEDULER_SUPERVISOR_POLICY_VERSION = "thewiz.scheduler_supervisor.v1"
MAX_SUPERVISED_PUBLICATION_BYTES = 1024**3
ABANDONED_RUN_GRACE_SECONDS = 60
SCHEDULER_RECOVERY_POLICY_VERSION = "thewiz.scheduler_crash_recovery.v1"
WIZARD_PROVIDER_ID = "crypto_wizards"
HYPERLIQUID_PUBLIC_PROVIDER_ID = "hyperliquid_public"


@dataclass(frozen=True)
class SupervisedSchedulerResult:
    """Complete supervised outcome with its immutable terminal evidence."""

    run_id: str
    intended_slot: str
    result_summary: dict[str, Any]
    result_paths: dict[str, str]
    terminal_receipt: dict[str, Any]
    terminal_paths: dict[str, Path]
    exit_code: int


@dataclass(frozen=True)
class SchedulerRecoveryResult:
    """Crash receipts published before a new scheduler invocation begins."""

    recovered_receipts: tuple[dict[str, Any], ...]
    blockers: tuple[str, ...]


def intended_scheduler_slot(
    contract: SchedulerContract,
    *,
    now: datetime,
) -> str:
    """Return the deterministic slot credited by one invocation."""

    observed = _as_utc(now)
    if contract.interval_seconds is not None:
        interval = contract.interval_seconds
        if interval <= 0:
            raise ValueError("scheduler interval must be positive")
        epoch = int(observed.timestamp())
        slot_epoch = epoch - (epoch % interval)
        slot = datetime.fromtimestamp(slot_epoch, tz=UTC)
        return f"{slot.isoformat()}/PT{interval}S"
    if contract.calendar_hour is None or contract.calendar_minute is None:
        raise ValueError("scheduler contract has no deterministic schedule")
    local_zone = ZoneInfo(RUNTIME_TIMEZONE)
    local_now = observed.astimezone(local_zone)
    local_slot = local_now.replace(
        hour=contract.calendar_hour,
        minute=contract.calendar_minute,
        second=0,
        microsecond=0,
    )
    if local_now < local_slot:
        local_slot -= timedelta(days=1)
    return f"{local_slot.astimezone(UTC).isoformat()}/P1D"


def recover_abandoned_scheduler_runs(
    *,
    root: Path,
    scheduler_key: str,
    authority: EffectAuthority,
    recovered_at: datetime,
    minimum_age_seconds: int = ABANDONED_RUN_GRACE_SECONDS,
    owner_matcher: Callable[[dict[str, Any]], bool | None] = (
        scheduler_process_owner_matches
    ),
) -> SchedulerRecoveryResult:
    """Terminalize conclusively abandoned intents without retrying their effects."""

    if minimum_age_seconds < 0:
        raise ValueError("minimum abandoned-run age must be non-negative")
    observed_at = _as_utc(recovered_at)
    intent_root = (
        root / "data" / "research" / "scheduler_run_intents" / scheduler_key
    )
    if not intent_root.exists():
        return SchedulerRecoveryResult((), ())
    recovered: list[dict[str, Any]] = []
    blockers: list[str] = []
    for path in sorted(intent_root.glob("*.json")):
        run_id = path.stem
        try:
            intent = load_validated_scheduler_run_intent(
                root,
                scheduler_key=scheduler_key,
                run_id=run_id,
            )
        except (OSError, TypeError, ValueError) as exc:
            token = sha256(path.name.encode("utf-8")).hexdigest()[:16]
            blockers.append(
                f"scheduler_run_intent_invalid:{token}:{safe_exception_code(exc)}"
            )
            continue
        if terminal_receipt_exists(root, scheduler_key=scheduler_key, run_id=run_id):
            try:
                load_validated_scheduler_terminal_receipt_by_run(
                    root,
                    scheduler_key=scheduler_key,
                    run_id=run_id,
                    expected_runtime_identity=intent,
                    require_launchd=False,
                )
            except (OSError, TypeError, ValueError) as exc:
                blockers.append(
                    f"scheduler_existing_terminal_invalid:{run_id}:"
                    f"{safe_exception_code(exc)}"
                )
            continue
        started_at = datetime.fromisoformat(str(intent["started_at_utc"])).astimezone(UTC)
        if observed_at < started_at + timedelta(seconds=minimum_age_seconds):
            continue
        owner_status = owner_matcher(intent)
        if owner_status is True:
            continue
        if owner_status is None:
            blockers.append(f"scheduler_abandoned_owner_ambiguous:{run_id}")
            continue

        slot_claimed = False
        try:
            load_validated_scheduler_slot_claim(
                root,
                scheduler_key=scheduler_key,
                intended_slot=str(intent["intended_slot"]),
                expected_run_id=run_id,
                expected_runtime_identity=intent,
            )
            slot_claimed = True
        except FileNotFoundError:
            slot_claimed = False
        except (OSError, TypeError, ValueError) as exc:
            blockers.append(
                f"scheduler_abandoned_slot_claim_invalid:{run_id}:"
                f"{safe_exception_code(exc)}"
            )
            continue

        accounting = authority.run_accounting(
            run_id=run_id,
            intended_slot_id=str(intent["intended_slot"]),
        )
        abandoned_reservations_failed = 0
        if (
            int(accounting["open_reservations"]) > 0
            and accounting.get("provider_effect_absence_proven") is True
        ):
            abandoned_reservations_failed = (
                authority.fail_open_external_reservations_without_provider_effects(
                    run_id=run_id,
                    intended_slot_id=str(intent["intended_slot"]),
                )
            )
            accounting = authority.run_accounting(
                run_id=run_id,
                intended_slot_id=str(intent["intended_slot"]),
            )
        recovery_blockers = [
            "scheduler_process_abandoned",
            "scheduler_crash_retry_requires_manual_reauthorization",
        ]
        if slot_claimed:
            recovery_blockers.append("scheduler_slot_claim_consumed")
        recovery_blockers.extend(str(value) for value in accounting["blockers"])
        if abandoned_reservations_failed:
            recovery_blockers.append(
                "external_reservation_abandoned_before_provider_effect"
            )
        if int(accounting["external_credits_consumed"]) > 0:
            recovery_blockers.append("external_credit_reconciliation_required")
        if int(accounting["order_attempts"]) > 0:
            recovery_blockers.append("abandoned_run_order_activity_detected")
        retryable = False
        runtime_identity = {
            "scheduler_key": scheduler_key,
            **{
                field: intent[field]
                for field in (
                    "trigger_provenance",
                    "capability_profile",
                    "capability_profile_sha256",
                    "runtime_contract_sha256",
                    "source_fingerprint_sha256",
                    "configuration_fingerprint_sha256",
                    "dependency_fingerprint_sha256",
                    "interpreter_fingerprint_sha256",
                    "schedule_fingerprint_sha256",
                )
            },
        }
        receipt = build_scheduler_terminal_receipt(
            run_id=run_id,
            intended_slot=str(intent["intended_slot"]),
            runtime_identity=runtime_identity,
            started_at=started_at,
            completed_at=max(observed_at, started_at),
            terminal_status="CRASH_RECOVERED",
            process_health="DEGRADED",
            business_state="INCOMPLETE",
            retryable=retryable,
            intended_slot_credit=False,
            blockers=sorted(set(recovery_blockers)),
            result_summary={"effect_accounting": accounting},
            external_calls=int(accounting["external_calls"]),
            external_credits_reserved=int(accounting["external_credits_reserved"]),
            external_credits_consumed=int(accounting["external_credits_consumed"]),
            external_credits_reconciled=0,
            order_attempts=0,
            order_submissions=0,
            child_exit_code=None,
            authority_advanced=False,
            promotion_authority=False,
            live_trading_authorized=False,
        )
        with publication_authority_session(
            authority=authority,
            run_id=run_id,
            intended_slot_id=str(intent["intended_slot"]),
            policy_version=SCHEDULER_RECOVERY_POLICY_VERSION,
            source_fingerprint_sha256=str(intent["source_fingerprint_sha256"]),
            runtime_fingerprint_sha256=str(intent["runtime_contract_sha256"]),
            configuration_fingerprint_sha256=str(
                intent["configuration_fingerprint_sha256"]
            ),
            allowed_scopes=frozenset({"scheduler_terminal"}),
            allowed_target_prefixes=(root.resolve(),),
            max_total_bytes=8 * 1024**2,
        ):
            publish_scheduler_terminal_receipt(root, receipt)
        recovered.append(receipt)
    return SchedulerRecoveryResult(tuple(recovered), tuple(sorted(set(blockers))))


def supervise_scheduler_run(
    *,
    root: Path,
    contract_key: str,
    publication_scope: str,
    callback: Callable[[], Any],
    now: datetime | None = None,
    require_launchd_provenance: bool = True,
) -> SupervisedSchedulerResult:
    """Run one scheduler callback under provenance, authority, and terminal fencing."""

    started = _as_utc(now or datetime.now(UTC))
    contract = scheduler_contract(contract_key)
    runtime_identity = scheduler_run_identity(
        root,
        contract=contract,
        require_launchd=require_launchd_provenance,
    )
    intended_slot = intended_scheduler_slot(contract, now=started)
    run_id = new_scheduler_run_id(scheduler_key=contract.key, now=started)
    wizard_external_research = (
        runtime_identity.get("capability_profile") == "WIZARD_EXTERNAL_RESEARCH"
    )
    effect_policy: Phase00ExternalEffectPolicy | None = None
    effect_provider_policies: tuple[ProviderExternalEffectPolicy, ...] = ()
    effect_policy_blockers: list[str] = []
    if wizard_external_research:
        try:
            effect_policy = load_phase00_external_effect_policy(
                root,
                as_of=started,
            )
            provider_ids = (
                (WIZARD_PROVIDER_ID, HYPERLIQUID_PUBLIC_PROVIDER_ID)
                if contract.key == "daily_research"
                else (WIZARD_PROVIDER_ID,)
            )
            effect_provider_policies = tuple(
                effect_policy.provider(provider_id)
                for provider_id in provider_ids
            )
        except ExternalEffectPolicyError as exc:
            effect_policy_blockers.append(exc.code)
    authority = EffectAuthority(
        root=root,
        secret=secrets.token_bytes(32),
        issuer_id=f"scheduler_supervisor:{contract.key}",
        profile=(
            PHASE00_WIZARD_RESEARCH_PROFILE
            if wizard_external_research
            else PHASE00_REPAIR_PROFILE
        ),
    )
    recovery = recover_abandoned_scheduler_runs(
        root=root,
        scheduler_key=contract.key,
        authority=authority,
        recovered_at=started,
    )
    result_summary: dict[str, Any] = {
        "recovered_abandoned_run_ids": [
            str(receipt["run_id"]) for receipt in recovery.recovered_receipts
        ],
        "external_effect_policy_status": (
            "BLOCKED"
            if effect_policy_blockers
            else "PASS"
            if wizard_external_research
            else "NOT_REQUIRED"
        ),
    }
    if effect_policy is not None:
        result_summary.update(effect_policy.receipt_fields(root))
        result_summary["external_effect_provider_ids"] = [
            provider.provider_id for provider in effect_provider_policies
        ]
    result_paths: dict[str, str] = {}
    terminal_status = "BLOCKED"
    process_health = "HEALTHY"
    business_state = "BLOCKED"
    retryable = False
    blockers: list[str] = [*recovery.blockers, *effect_policy_blockers]
    pending_exception: BaseException | None = None

    with publication_authority_session(
        authority=authority,
        run_id=run_id,
        intended_slot_id=intended_slot,
        policy_version=SCHEDULER_SUPERVISOR_POLICY_VERSION,
        source_fingerprint_sha256=runtime_identity["source_fingerprint_sha256"],
        runtime_fingerprint_sha256=runtime_identity["runtime_contract_sha256"],
        configuration_fingerprint_sha256=runtime_identity["configuration_fingerprint_sha256"],
        allowed_scopes=frozenset({publication_scope, "scheduler_terminal"}),
        allowed_target_prefixes=(root.resolve(),),
        max_total_bytes=MAX_SUPERVISED_PUBLICATION_BYTES,
    ):
        intent = build_scheduler_run_intent(
            run_id=run_id,
            intended_slot=intended_slot,
            runtime_identity=runtime_identity,
            started_at=started,
        )
        publish_scheduler_run_intent(root, intent)
        if blockers:
            terminal_status = "BLOCKED"
            business_state = "BLOCKED"
        elif not runtime_identity.get("runtime_environment_valid", False):
            blockers.extend(runtime_identity.get("runtime_environment_blockers", []))
            terminal_status = "BLOCKED"
            business_state = "BLOCKED"
        else:
            try:
                claim_scheduler_slot(
                    root,
                    run_id=run_id,
                    intended_slot=intended_slot,
                    runtime_identity=runtime_identity,
                    claimed_at=started,
                )
                if wizard_external_research:
                    if effect_policy is None or not effect_provider_policies:
                        raise RuntimeError(
                            "external_effect_policy_missing_after_validation"
                        )
                    issuer_kwargs = {
                        "authority": authority,
                        "run_id": run_id,
                        "intended_slot_id": intended_slot,
                        "source_fingerprint_sha256": runtime_identity[
                            "source_fingerprint_sha256"
                        ],
                        "runtime_fingerprint_sha256": runtime_identity[
                            "runtime_contract_sha256"
                        ],
                        "configuration_fingerprint_sha256": runtime_identity[
                            "configuration_fingerprint_sha256"
                        ],
                        "policy_version": effect_policy.permit_policy_version,
                    }
                    if len(effect_provider_policies) == 1:
                        provider = effect_provider_policies[0]
                        external_issuer_context = external_effect_issuer_session(
                            **issuer_kwargs,
                            provider_id=provider.provider_id,
                            account_scope_id=provider.account_scope_id,
                            allowed_targets=provider.allowed_targets,
                            allowed_credential_keys=provider.allowed_credential_keys,
                            max_total_requests=provider.max_total_requests_per_run,
                            max_total_credits=provider.max_total_credits_per_run,
                            allowed_call_contracts=frozenset(
                                ExternalEffectCallContract(
                                    operation=row.operation,
                                    method=row.method,
                                    target=row.target,
                                    credit_units_per_request=row.credit_units,
                                )
                                for row in provider.endpoint_pricing
                            ),
                        )
                    else:
                        issuers = tuple(
                            ExternalEffectIssuer(
                                **issuer_kwargs,
                                provider_id=provider.provider_id,
                                account_scope_id=provider.account_scope_id,
                                allowed_targets=provider.allowed_targets,
                                allowed_credential_keys=(
                                    provider.allowed_credential_keys
                                ),
                                max_total_requests=(
                                    provider.max_total_requests_per_run
                                ),
                                max_total_credits=(
                                    provider.max_total_credits_per_run
                                ),
                                allowed_call_contracts=frozenset(
                                    ExternalEffectCallContract(
                                        operation=row.operation,
                                        method=row.method,
                                        target=row.target,
                                        credit_units_per_request=row.credit_units,
                                    )
                                    for row in provider.endpoint_pricing
                                ),
                            )
                            for provider in effect_provider_policies
                        )
                        external_issuer_context = (
                            external_effect_issuer_bundle_session(issuers)
                        )
                else:
                    external_issuer_context = nullcontext()
                with (
                    governed_evidence_write_lock(
                        root,
                        blocking=False,
                        scope=publication_scope,
                        run_id=run_id,
                    ),
                    phase00_effect_guard(),
                    external_issuer_context,
                ):
                    result = callback()
                callback_summary, result_paths = _normalize_result(result)
                result_summary = {**callback_summary, **result_summary}
                (
                    terminal_status,
                    process_health,
                    business_state,
                    retryable,
                    blockers,
                ) = _terminal_semantics(result_summary)
            except SchedulerSlotAlreadyClaimed as exc:
                terminal_status = "DEFERRED"
                business_state = "DEFERRED"
                retryable = False
                blockers = [safe_exception_code(exc)]
            except GovernedEvidenceMaintenanceActive as exc:
                terminal_status = "DEFERRED"
                business_state = "DEFERRED"
                retryable = False
                blockers = [safe_exception_code(exc)]
            except GovernedEvidenceLockBusy as exc:
                terminal_status = "DEFERRED"
                business_state = "DEFERRED"
                retryable = False
                blockers = [safe_exception_code(exc)]
            except BaseException as exc:  # noqa: BLE001 - terminalize every child crash
                terminal_status = "FAILED"
                process_health = "FAILED"
                business_state = "FAILED"
                blockers = [f"scheduler_callback_crash:{type(exc).__name__}"]
                pending_exception = exc

        accounting = authority.run_accounting(
            run_id=run_id,
            intended_slot_id=intended_slot,
        )
        result_summary["effect_accounting"] = accounting
        accounting_blockers = [
            str(value) for value in accounting.get("blockers", [])
        ]
        for field in (
            "external_calls",
            "external_credits_reserved",
            "external_credits_consumed",
        ):
            reported = _safe_counter(result_summary, field)
            durable = int(accounting[field])
            result_summary[f"reported_{field}"] = reported
            result_summary[field] = durable
            if reported != durable and (reported or durable):
                accounting_blockers.append(
                    f"scheduler_effect_accounting_mismatch:{field}"
                )
        if accounting_blockers:
            blockers = sorted({*blockers, *accounting_blockers})
            terminal_status = "FAILED"
            process_health = "FAILED"
            business_state = "FAILED"
            retryable = False

        completed = datetime.now(UTC)
        receipt = build_scheduler_terminal_receipt(
            run_id=run_id,
            intended_slot=intended_slot,
            runtime_identity=runtime_identity,
            started_at=started,
            completed_at=max(completed, started),
            terminal_status=terminal_status,
            process_health=process_health,
            business_state=business_state,
            retryable=retryable,
            intended_slot_credit=terminal_status == "PASS",
            blockers=blockers,
            result_summary=result_summary,
            external_calls=_safe_counter(result_summary, "external_calls"),
            external_credits_reserved=_safe_counter(
                result_summary,
                "external_credits_reserved",
            ),
            external_credits_consumed=_safe_counter(
                result_summary,
                "external_credits_consumed",
            ),
            external_credits_reconciled=_safe_counter(
                result_summary,
                "external_credits_reconciled",
            ),
            order_attempts=0,
            order_submissions=0,
            authority_advanced=False,
            promotion_authority=False,
            live_trading_authorized=False,
        )
        terminal_paths = publish_scheduler_terminal_receipt(root, receipt)

    if pending_exception is not None:
        raise pending_exception
    return SupervisedSchedulerResult(
        run_id=run_id,
        intended_slot=intended_slot,
        result_summary=result_summary,
        result_paths=result_paths,
        terminal_receipt=receipt,
        terminal_paths=terminal_paths,
        exit_code=scheduler_terminal_exit_code(receipt),
    )


def _normalize_result(result: Any) -> tuple[dict[str, Any], dict[str, str]]:
    if hasattr(result, "summary") and hasattr(result, "paths"):
        summary = result.summary
        paths = result.paths
    elif isinstance(result, dict) and "summary" in result:
        summary = result.get("summary", {})
        paths = result.get("paths", {})
    elif isinstance(result, dict):
        summary = result
        paths = {}
    else:
        raise TypeError("scheduler callback returned an unsupported result")
    if not isinstance(summary, dict) or not isinstance(paths, dict):
        raise TypeError("scheduler result summary and paths must be mappings")
    sanitized_summary = json.loads(json.dumps(summary, default=str))
    sanitized_paths = {str(key): str(value) for key, value in paths.items()}
    return sanitized_summary, sanitized_paths


def _terminal_semantics(
    summary: dict[str, Any],
) -> tuple[str, str, str, bool, list[str]]:
    status = str(summary.get("status", "BLOCKED")).strip().upper()
    blockers = sorted(
        {str(value).strip() for value in summary.get("blockers", []) if str(value).strip()}
    )
    authority_fields = (
        "authority_advanced",
        "candidate_promotion_authority",
        "promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if any(_truthy(summary.get(field)) for field in authority_fields):
        blockers.append("scheduler_result_claimed_prohibited_authority")
        return "FAILED", "FAILED", "FAILED", False, sorted(set(blockers))
    if _safe_counter(summary, "order_attempts") or _safe_counter(
        summary,
        "order_submissions",
    ):
        blockers.append("scheduler_result_reported_order_activity")
        return "FAILED", "FAILED", "FAILED", False, sorted(set(blockers))
    if status == "PASS" and not blockers:
        return "PASS", "HEALTHY", "PASS", False, []
    if not blockers:
        blockers = [f"scheduler_business_status:{status or 'MISSING'}"]
    if status.startswith("DEFERRED"):
        return "DEFERRED", "HEALTHY", "DEFERRED", True, blockers
    if status.startswith("INCOMPLETE"):
        return "INCOMPLETE", "DEGRADED", "INCOMPLETE", True, blockers
    if status in {"FAILED", "ERROR", "CRASH"}:
        return "FAILED", "FAILED", "FAILED", False, blockers
    return "BLOCKED", "HEALTHY", "BLOCKED", False, blockers


def _safe_counter(summary: dict[str, Any], field: str) -> int:
    value = summary.get(field, 0)
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "pass",
        "ready",
    }


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
