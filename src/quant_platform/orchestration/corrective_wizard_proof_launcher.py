"""Lightweight launch guard for the heavy Crypto Wizards proof scheduler."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

from quant_platform.orchestration.corrective_external_effects import (
    current_external_effect_issuer,
    read_authorized_credential,
    reserved_external_effect_session,
)
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_text,
    scheduler_python_path,
    write_immutable_json,
)
from quant_platform.orchestration.corrective_scheduler_lock import (
    publication_lease_for_path,
)
from quant_platform.orchestration.corrective_scheduler_supervisor import (
    supervise_scheduler_run,
)
from quant_platform.orchestration.corrective_wizard_capture_manifest import (
    validate_capture_manifest_source_receipt,
)
from quant_platform.orchestration.corrective_wizard_capture_reconciliation import (
    validate_capture_reconciliation_evidence,
)
from quant_platform.orchestration.corrective_wizard_ou_v5_holdout import (
    validate_ou_v5_stage3_evidence,
)
from quant_platform.orchestration.corrective_wizard_ou_v6_holdout import (
    validate_ou_v6_stage3_evidence,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_wizard_proof_launcher.v1"
LAUNCHER_RECEIPT_DIR = "data/research/wizard_proof_launcher_receipts"
HEAVY_MODULE = "quant_platform.orchestration.corrective_wizard_proof_scheduler"
STAGE3_LOCAL_EVIDENCE_PATHS = (
    "reports/active/hyperliquid_wizard_vendor_mode_proofs.csv",
    "reports/active/wizard_copula_behavioral_status.json",
    "reports/active/wizard_copula_behavioral_v2_status.json",
    "reports/active/wizard_dynamic_v2_activation_status.json",
    "reports/active/wizard_dynamic_v2_proof_refresh_status.json",
    "reports/supreme_team/wizard_dynamic_v2_review.json",
    "reports/active/wizard_ou_v2_holdout_status.json",
    "reports/active/wizard_ou_trend_selector_v1_receipt.json",
    "reports/active/wizard_ou_trend_selector_v1_derivation.csv",
    "reports/active/wizard_ou_trend_selector_v1_predictions.csv",
    "reports/active/wizard_ou_v3_capture_status.json",
    "reports/active/wizard_ou_v3_holdout_status.json",
    "reports/active/wizard_ou_v3_supersession_gate.json",
    "reports/active/wizard_ou_v3_review_packet.json",
    "reports/active/wizard_ou_v3_activation_status.json",
    "reports/active/wizard_ou_v3_proof_refresh_status.json",
    "reports/active/wizard_ou_v4_holdout_receipt.json",
    "reports/active/wizard_ou_v4_capture_status.json",
    "reports/active/wizard_ou_v4_holdout_status.json",
    "reports/active/wizard_ou_v4_derivation.csv",
    "reports/active/wizard_ou_v4_predictions.csv",
    "reports/active/wizard_ou_v4_supersession_gate.json",
    "reports/active/wizard_ou_v4_review_packet.json",
    "reports/active/wizard_ou_v4_activation_status.json",
    "reports/active/wizard_ou_v4_proof_refresh_status.json",
    "reports/supreme_team/wizard_ou_v4_review.json",
    "reports/active/wizard_ou_v5_holdout_receipt.json",
    "reports/active/wizard_ou_v5_capture_status.json",
    "reports/active/wizard_ou_v5_holdout_status.json",
    "reports/active/wizard_ou_v5_failure_attribution.csv",
    "reports/active/wizard_ou_v5_failure_attribution.json",
    "reports/active/wizard_ou_v5_derivation.csv",
    "reports/active/wizard_ou_v5_predictions.csv",
    "reports/active/wizard_ou_v5_supersession_gate.json",
    "reports/active/wizard_ou_v5_review_packet.json",
    "reports/active/wizard_ou_v5_activation_status.json",
    "reports/active/wizard_ou_v5_proof_refresh_status.json",
    "reports/supreme_team/wizard_ou_v5_review.json",
    "reports/supreme_team/wizard_ou_v5_failure_checkpoint.md",
    "reports/active/wizard_ou_v6_holdout_receipt.json",
    "reports/active/wizard_ou_v6_capture_status.json",
    "reports/active/wizard_ou_v6_holdout_status.json",
    "reports/active/wizard_ou_v6_derivation.csv",
    "reports/active/wizard_ou_v6_predictions.csv",
    "reports/active/wizard_ou_v6_supersession_gate.json",
    "reports/active/wizard_ou_v6_review_packet.json",
    "reports/active/wizard_ou_v6_activation_status.json",
    "reports/active/wizard_ou_v6_proof_refresh_status.json",
    "reports/supreme_team/wizard_ou_v6_review.json",
    "reports/active/corrective_wizard_next_capture_manifest.json",
    "reports/active/corrective_wizard_capture_reconciliation.json",
    "reports/active/wizard_mode_comparator_contract.csv",
    "config/wizard_copula_behavioral_parity.json",
    "config/wizard_copula_behavioral_parity_v2.json",
    "config/wizard_ou_comparator_v2_holdout.json",
    "config/wizard_ou_comparator_v3_holdout.json",
    "config/wizard_ou_comparator_v4_holdout.json",
    "config/wizard_ou_comparator_v5_holdout.json",
    "config/wizard_ou_comparator_v6_holdout.json",
    "config/wizard_ou_trend_selector_v1_holdout.json",
    "src/quant_platform/wizard_hyperliquid_mode_proof.py",
    "src/quant_platform/orchestration/corrective_wizard_copula_behavioral.py",
    "src/quant_platform/orchestration/corrective_wizard_dynamic_supreme_review.py",
    "src/quant_platform/orchestration/corrective_wizard_ou_holdout.py",
    "src/quant_platform/orchestration/corrective_wizard_ou_v4_holdout.py",
    "src/quant_platform/orchestration/corrective_wizard_ou_v5_holdout.py",
    "src/quant_platform/orchestration/corrective_wizard_ou_v5_failure_attribution.py",
    "src/quant_platform/orchestration/corrective_wizard_ou_v6_holdout.py",
    "src/quant_platform/orchestration/corrective_wizard_ou_v6_supreme_review.py",
    "src/quant_platform/wizard_ou_v6_comparator_activation.py",
    "src/quant_platform/orchestration/corrective_wizard_ou_v4_supreme_review.py",
    "src/quant_platform/orchestration/corrective_wizard_ou_v5_supreme_review.py",
    "src/quant_platform/wizard_dynamic_comparator_activation.py",
    "src/quant_platform/wizard_ou_v4_comparator_activation.py",
    "src/quant_platform/wizard_ou_v5_comparator_activation.py",
)


def run_wizard_proof_launcher(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    execute: bool = True,
    force: bool = False,
    python: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    unattended_preflight_builder: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Skip heavy imports after a proven daily attempt; otherwise run the scheduler."""

    checked_at = _as_utc(now)
    active = root / "reports" / "active"
    latest_path = active / "corrective_wizard_proof_scheduler_status.json"
    receipts = active / "wizard_proof_scheduler_receipts"
    receipt_paths_before = _dated_receipt_paths(
        receipts=receipts,
        attempt_date=checked_at.date().isoformat(),
    )
    state, evidence_paths = _daily_attempt_state(
        latest_path=latest_path,
        receipts=receipts,
        attempt_date=checked_at.date().isoformat(),
    )
    latest = _read_json(latest_path)
    latest_execution = _latest_immutable_execution_receipt(
        receipts=receipts,
        attempt_date=checked_at.date().isoformat(),
    )
    latest_status_capture_manifest_binding_complete = _capture_manifest_binding_complete(
        latest, root=root
    )
    latest_status_stage3_evidence_complete = _stage3_evidence_complete(latest, root=root)
    latest_immutable_execution_capture_manifest_binding_complete = (
        _capture_manifest_binding_complete(latest_execution, root=root)
    )
    latest_immutable_execution_stage3_evidence_complete = _stage3_evidence_complete(
        latest_execution, root=root
    )
    internal_continuation_reason = _internal_registered_continuation_reason(
        root=root,
        latest=latest_execution or latest,
        same_day_attempt_state=state,
    )
    prior_stage4_contract_id = (
        _active_stage4_contract_id(root)
        if internal_continuation_reason in {"stage4_generation", "stage4_active_execution"}
        else ""
    )
    internal_continuation_needed = bool(internal_continuation_reason)
    invoke_internal_continuation = bool(execute and internal_continuation_needed and not force)
    should_run = bool(force or not execute or state == "CLEAR" or internal_continuation_needed)
    if state == "UNVERIFIABLE" and not force:
        should_run = False
        launcher_status = "BLOCKED_UNVERIFIABLE_SAME_DAY_EVIDENCE"
        blocker = "same_day_proof_receipt_evidence_malformed"
    elif internal_continuation_needed and execute and not force:
        launcher_status = "INTERNAL_REGISTERED_CONTINUATION_REQUIRED"
        blocker = ""
    elif state == "ATTEMPTED" and execute and not force:
        launcher_status = "DEFERRED_SAME_UTC_DAY_LIGHTWEIGHT"
        blocker = "bounded_external_proof_cycle_already_attempted_this_utc_day"
    else:
        launcher_status = "HEAVY_SCHEDULER_REQUIRED"
        blocker = ""

    command: list[str] = []
    return_code: int | None = None
    scheduler_status = ""
    scheduler_receipt_id = ""
    scheduler_receipt_path = ""
    scheduler_receipt_valid = False
    scheduler_receipt_blocker = ""
    scheduler_started_at = ""
    scheduler_external_attempt_made = False
    scheduler_stage3_evidence_complete = False
    scheduler_capture_manifest_binding_complete = False
    scheduler_registered_rerun_status = ""
    scheduler_learning_handoff_status = ""
    scheduler_stage5_research_gate_pass = False
    scheduler_dynamic_v2_proof_refresh_status = ""
    scheduler_dynamic_v2_comparator_generation = 0
    scheduler_dynamic_v2_proofs_refreshed = 0
    scheduler_ou_v3_proof_refresh_status = ""
    scheduler_ou_v3_comparator_generation = 0
    scheduler_ou_v3_proofs_refreshed = 0
    scheduler_ou_v4_proof_refresh_status = ""
    scheduler_ou_v4_comparator_generation = 0
    scheduler_ou_v4_proofs_refreshed = 0
    scheduler_ou_v5_proof_refresh_status = ""
    scheduler_ou_v5_comparator_generation = 0
    scheduler_ou_v5_proofs_refreshed = 0
    unattended_preflight_required = bool(
        unattended_preflight_builder is not None and execute and not invoke_internal_continuation
    )
    unattended_preflight_status = "NOT_REQUIRED"
    unattended_preflight_receipt_id = ""
    unattended_preflight_receipt_path = ""
    unattended_preflight_blockers: list[str] = []
    next_eligible = str(latest.get("next_external_attempt_eligible_at", ""))
    if should_run and unattended_preflight_required:
        try:
            preflight = unattended_preflight_builder(root=root, now=checked_at)
            preflight_summary = dict(preflight.summary)
            unattended_preflight_status = str(
                preflight_summary.get("status", "BLOCKED_UNATTENDED_EXTERNAL_PREFLIGHT")
            )
            unattended_preflight_receipt_id = str(preflight_summary.get("receipt_id", ""))
            unattended_preflight_receipt_path = _relative(
                Path(preflight.paths.get("immutable_receipt", "")), root
            )
            unattended_preflight_blockers = [
                str(value) for value in preflight_summary.get("blockers", [])
            ]
        except Exception as exc:  # noqa: BLE001 - launcher must fail closed
            unattended_preflight_status = "BLOCKED_UNATTENDED_EXTERNAL_PREFLIGHT"
            unattended_preflight_blockers = [
                f"unattended_preflight_failed:{safe_exception_code(exc)}"
            ]
        if unattended_preflight_status != "PASS_UNATTENDED_EXTERNAL_PREFLIGHT":
            should_run = False
            launcher_status = "BLOCKED_UNATTENDED_EXTERNAL_PREFLIGHT"
            blocker = (
                unattended_preflight_blockers[0]
                if unattended_preflight_blockers
                else "unattended_external_preflight_not_pass"
            )
    if should_run:
        executable = python or scheduler_python_path(root)
        if not executable.is_file():
            launcher_status = "BLOCKED_SCHEDULER_PYTHON_MISSING"
            blocker = f"scheduler_python_missing:{executable}"
            should_run = False
        else:
            command = [str(executable), "-m", HEAVY_MODULE]
            if execute:
                command.append("--execute")
            if invoke_internal_continuation:
                command.append("--internal-continuation-only")
            if force:
                command.append("--force")
            env = os.environ.copy()
            source = str(root / "src")
            env["PYTHONPATH"] = (
                source if not env.get("PYTHONPATH") else source + os.pathsep + env["PYTHONPATH"]
            )
            completed = runner(
                command,
                cwd=root,
                env=env,
                capture_output=False,
                text=True,
                check=False,
            )
            return_code = int(completed.returncode)
            if return_code != 0:
                launcher_status = "HEAVY_SCHEDULER_FAILED"
                blocker = f"heavy_scheduler_exit_code:{return_code}"
            else:
                (
                    scheduler_receipt,
                    scheduler_receipt_file,
                    scheduler_receipt_blocker,
                ) = _fresh_scheduler_receipt(
                    receipts=receipts,
                    attempt_date=checked_at.date().isoformat(),
                    checked_at=checked_at,
                    prior_paths=receipt_paths_before,
                    execute=execute,
                    force=force,
                    internal_continuation_only=invoke_internal_continuation,
                )
                scheduler_receipt_valid = bool(scheduler_receipt)
                if not scheduler_receipt_valid:
                    launcher_status = "BLOCKED_SCHEDULER_RECEIPT_UNVERIFIED"
                    blocker = scheduler_receipt_blocker
                else:
                    scheduler_status = str(scheduler_receipt.get("status", ""))
                    scheduler_receipt_id = str(scheduler_receipt.get("receipt_id", ""))
                    scheduler_receipt_path = _relative(scheduler_receipt_file, root)
                    scheduler_started_at = str(scheduler_receipt.get("started_at_utc", ""))
                    scheduler_external_attempt_made = bool(
                        scheduler_receipt.get("external_attempt_made_this_cycle", False)
                    )
                    scheduler_registered_rerun_status = str(
                        scheduler_receipt.get("registered_rerun_status", "")
                    )
                    scheduler_learning_handoff_status = str(
                        scheduler_receipt.get("registered_learning_handoff_status", "")
                    )
                    scheduler_stage5_research_gate_pass = bool(
                        scheduler_receipt.get("registered_stage5_research_gate_pass", False)
                    )
                    scheduler_dynamic_v2_proof_refresh_status = str(
                        scheduler_receipt.get("dynamic_v2_proof_refresh_status", "")
                    )
                    scheduler_dynamic_v2_comparator_generation = _safe_int(
                        scheduler_receipt.get("dynamic_v2_comparator_generation", 0)
                    )
                    scheduler_dynamic_v2_proofs_refreshed = _safe_int(
                        scheduler_receipt.get("dynamic_v2_proofs_refreshed", 0)
                    )
                    scheduler_ou_v3_proof_refresh_status = str(
                        scheduler_receipt.get("ou_v3_proof_refresh_status", "")
                    )
                    scheduler_ou_v3_comparator_generation = _safe_int(
                        scheduler_receipt.get("ou_v3_comparator_generation", 0)
                    )
                    scheduler_ou_v3_proofs_refreshed = _safe_int(
                        scheduler_receipt.get("ou_v3_proofs_refreshed", 0)
                    )
                    scheduler_ou_v4_proof_refresh_status = str(
                        scheduler_receipt.get("ou_v4_proof_refresh_status", "")
                    )
                    scheduler_ou_v4_comparator_generation = _safe_int(
                        scheduler_receipt.get("ou_v4_comparator_generation", 0)
                    )
                    scheduler_ou_v4_proofs_refreshed = _safe_int(
                        scheduler_receipt.get("ou_v4_proofs_refreshed", 0)
                    )
                    scheduler_ou_v5_proof_refresh_status = str(
                        scheduler_receipt.get("ou_v5_proof_refresh_status", "")
                    )
                    scheduler_ou_v5_comparator_generation = _safe_int(
                        scheduler_receipt.get("ou_v5_comparator_generation", 0)
                    )
                    scheduler_ou_v5_proofs_refreshed = _safe_int(
                        scheduler_receipt.get("ou_v5_proofs_refreshed", 0)
                    )
                    scheduler_stage3_evidence_complete = _stage3_evidence_complete(
                        scheduler_receipt, root=root
                    )
                    scheduler_capture_manifest_binding_complete = (
                        _capture_manifest_binding_complete(scheduler_receipt, root=root)
                    )
                    next_eligible = str(
                        scheduler_receipt.get("next_external_attempt_eligible_at", next_eligible)
                    )
                    launcher_status, blocker = _reconciled_launcher_outcome(
                        root=root,
                        scheduler_receipt=scheduler_receipt,
                        execute=execute,
                        internal_continuation_reason=(
                            internal_continuation_reason if not force else ""
                        ),
                        prior_stage4_contract_id=prior_stage4_contract_id,
                    )

    if state == "ATTEMPTED" and not next_eligible:
        next_eligible = datetime.combine(
            checked_at.date() + timedelta(days=1),
            datetime.min.time(),
            tzinfo=UTC,
        ).isoformat()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "checked_at_utc": checked_at.isoformat(),
        "attempt_date_utc": checked_at.date().isoformat(),
        "launcher_status": launcher_status,
        "same_day_attempt_state": state,
        "same_day_evidence_count": len(evidence_paths),
        "same_day_evidence_paths": evidence_paths[-10:],
        "heavy_scheduler_invoked": bool(command),
        "heavy_scheduler_return_code": return_code,
        "heavy_scheduler_receipt_valid": scheduler_receipt_valid,
        "heavy_scheduler_receipt_blocker": scheduler_receipt_blocker,
        "scheduler_status": scheduler_status,
        "scheduler_started_at_utc": scheduler_started_at,
        "scheduler_receipt_id": scheduler_receipt_id,
        "scheduler_receipt_path": scheduler_receipt_path,
        "scheduler_stage3_evidence_complete": scheduler_stage3_evidence_complete,
        "scheduler_capture_manifest_binding_complete": (
            scheduler_capture_manifest_binding_complete
        ),
        "latest_status_capture_manifest_binding_complete": (
            latest_status_capture_manifest_binding_complete
        ),
        "latest_status_stage3_evidence_complete": (latest_status_stage3_evidence_complete),
        "latest_immutable_execution_capture_manifest_binding_complete": (
            latest_immutable_execution_capture_manifest_binding_complete
        ),
        "latest_immutable_execution_stage3_evidence_complete": (
            latest_immutable_execution_stage3_evidence_complete
        ),
        "scheduler_registered_rerun_status": scheduler_registered_rerun_status,
        "scheduler_learning_handoff_status": scheduler_learning_handoff_status,
        "scheduler_stage5_research_gate_pass": scheduler_stage5_research_gate_pass,
        "scheduler_dynamic_v2_proof_refresh_status": (scheduler_dynamic_v2_proof_refresh_status),
        "scheduler_dynamic_v2_comparator_generation": (scheduler_dynamic_v2_comparator_generation),
        "scheduler_dynamic_v2_proofs_refreshed": scheduler_dynamic_v2_proofs_refreshed,
        "scheduler_ou_v3_proof_refresh_status": (scheduler_ou_v3_proof_refresh_status),
        "scheduler_ou_v3_comparator_generation": (scheduler_ou_v3_comparator_generation),
        "scheduler_ou_v3_proofs_refreshed": scheduler_ou_v3_proofs_refreshed,
        "scheduler_ou_v4_proof_refresh_status": (scheduler_ou_v4_proof_refresh_status),
        "scheduler_ou_v4_comparator_generation": (scheduler_ou_v4_comparator_generation),
        "scheduler_ou_v4_proofs_refreshed": scheduler_ou_v4_proofs_refreshed,
        "scheduler_ou_v5_proof_refresh_status": scheduler_ou_v5_proof_refresh_status,
        "scheduler_ou_v5_comparator_generation": scheduler_ou_v5_comparator_generation,
        "scheduler_ou_v5_proofs_refreshed": scheduler_ou_v5_proofs_refreshed,
        "unattended_external_preflight_required": unattended_preflight_required,
        "unattended_external_preflight_status": unattended_preflight_status,
        "unattended_external_preflight_receipt_id": unattended_preflight_receipt_id,
        "unattended_external_preflight_receipt_path": unattended_preflight_receipt_path,
        "unattended_external_preflight_blockers": unattended_preflight_blockers,
        "internal_registered_continuation_needed": internal_continuation_needed,
        "internal_continuation_reason": internal_continuation_reason,
        "internal_continuation_prior_stage4_contract_id": (prior_stage4_contract_id),
        "internal_continuation_only_invoked": bool(command and invoke_internal_continuation),
        "external_calls_forbidden_for_invocation": bool(command and invoke_internal_continuation),
        "execute_requested": execute,
        "force_requested": force,
        "blocker": blocker,
        "next_external_attempt_eligible_at": next_eligible,
        "external_attempt_made_by_launcher": scheduler_external_attempt_made,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    payload["completed_at_utc"] = datetime.now(UTC).isoformat()
    payload["launcher_receipt_id"] = (
        "wizardlauncher_" + sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:20]
    )
    immutable_receipt_path = (
        root
        / LAUNCHER_RECEIPT_DIR
        / payload["attempt_date_utc"]
        / f"{payload['launcher_receipt_id']}.json"
    )
    _write_or_validate_immutable_json(payload, immutable_receipt_path)
    payload["immutable_launcher_receipt_path"] = _relative(
        immutable_receipt_path,
        root,
    )
    payload["immutable_launcher_receipt_sha256"] = sha256(
        immutable_receipt_path.read_bytes()
    ).hexdigest()
    status_path = active / "corrective_wizard_proof_launcher_status.json"
    _publish_latest_launcher_status(payload, status_path)
    from quant_platform.orchestration.corrective_canonical_status import (
        build_canonical_program_status,
    )

    canonical = build_canonical_program_status(root=root, now=checked_at)
    payload["launcher_status_path"] = _relative(status_path, root)
    payload["canonical_status_path"] = _relative(canonical.paths["status"], root)
    return payload


def _dated_receipt_paths(*, receipts: Path, attempt_date: str) -> set[Path]:
    if not receipts.is_dir():
        return set()
    return {path.resolve() for path in receipts.glob(f"{attempt_date}_*.json") if path.is_file()}


def latest_verified_immutable_scheduler_execution(*, root: Path = ROOT) -> dict[str, Any]:
    """Audit the newest scheduler execution without trusting mutable status files."""

    receipts = root / "reports" / "active" / "wizard_proof_scheduler_receipts"
    candidates: list[tuple[datetime, Path, dict[str, Any]]] = []
    invalid_started_at: list[str] = []
    if receipts.is_dir():
        for path in receipts.glob("*.json"):
            payload, valid_json = _try_read_json(path)
            if not valid_json or payload.get("execution_requested") is not True:
                continue
            if payload.get("schema_version") != ("thewiz.corrective_wizard_proof_scheduler.v1"):
                continue
            try:
                started_at = _parse_utc(str(payload.get("started_at_utc", "")))
            except ValueError:
                invalid_started_at.append(f"scheduler_receipt_started_at_invalid:{path.name}")
                continue
            candidates.append((started_at, path, payload))

    if not candidates:
        blockers = invalid_started_at or ["final_immutable_scheduler_execution_receipt_missing"]
        return {
            "status": "BLOCKED_FINAL_IMMUTABLE_EXECUTION",
            "valid": False,
            "receipt_id": "",
            "cycle_receipt_path": "",
            "immutable_receipt_path": "",
            "started_at_utc": "",
            "attempt_date_utc": "",
            "capture_manifest_binding_complete": False,
            "stage3_evidence_complete": False,
            "blockers": blockers,
            "research_only": True,
            "candidate_promotion_authority": False,
            "order_submission_included": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }

    started_at, path, payload = max(candidates, key=lambda item: item[0])
    blockers: list[str] = []
    attempt_date = str(payload.get("attempt_date_utc", ""))
    if attempt_date != started_at.date().isoformat():
        blockers.append(f"scheduler_receipt_attempt_date_mismatch:{path.name}")
    if payload.get("final_immutable_receipt_required") is not True:
        blockers.append(f"scheduler_receipt_not_final_immutable:{path.name}")
    if payload.get("research_only") is not True:
        blockers.append(f"scheduler_receipt_not_research_only:{path.name}")
    if any(
        payload.get(field) is not False
        for field in (
            "candidate_promotion_authority",
            "order_submission_included",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    ):
        blockers.append(f"scheduler_receipt_authority_violation:{path.name}")
    if not _scheduler_receipt_id_valid(payload):
        blockers.append(f"scheduler_receipt_id_mismatch:{path.name}")
    if payload.get("final_immutable_receipt_required") is True:
        immutable_blocker = _immutable_scheduler_receipt_blocker(
            payload=payload,
            path=path,
        )
        if immutable_blocker:
            blockers.append(immutable_blocker)

    receipt_id = str(payload.get("receipt_id", ""))
    immutable = (
        root / "data" / "research" / "wizard_proof_scheduler_receipts" / f"{receipt_id}.json"
    )
    valid = not blockers
    return {
        "status": (
            "PASS_FINAL_IMMUTABLE_EXECUTION" if valid else "BLOCKED_FINAL_IMMUTABLE_EXECUTION"
        ),
        "valid": valid,
        "receipt_id": receipt_id,
        "cycle_receipt_path": _relative(path, root),
        "immutable_receipt_path": _relative(immutable, root) if valid else "",
        "started_at_utc": started_at.isoformat(),
        "attempt_date_utc": attempt_date,
        "capture_manifest_binding_complete": bool(
            valid and _capture_manifest_binding_complete(payload, root=root)
        ),
        "stage3_evidence_complete": bool(valid and _stage3_evidence_complete(payload, root=root)),
        "blockers": blockers,
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _latest_immutable_execution_receipt(*, receipts: Path, attempt_date: str) -> dict[str, Any]:
    valid: list[tuple[datetime, dict[str, Any]]] = []
    if not receipts.is_dir():
        return {}
    for path in receipts.glob(f"{attempt_date}_*.json"):
        payload, valid_json = _try_read_json(path)
        if not valid_json:
            continue
        if payload.get("schema_version") != "thewiz.corrective_wizard_proof_scheduler.v1":
            continue
        if payload.get("attempt_date_utc") != attempt_date:
            continue
        if payload.get("execution_requested") is not True:
            continue
        if payload.get("research_only") is not True:
            continue
        if any(
            payload.get(field) is not False
            for field in (
                "candidate_promotion_authority",
                "order_submission_included",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        ):
            continue
        if not _scheduler_receipt_id_valid(payload):
            continue
        if payload.get(
            "final_immutable_receipt_required"
        ) is True and _immutable_scheduler_receipt_blocker(payload=payload, path=path):
            continue
        try:
            started_at = _parse_utc(str(payload.get("started_at_utc", "")))
        except ValueError:
            continue
        valid.append((started_at, payload))
    return max(valid, key=lambda item: item[0])[1] if valid else {}


def _fresh_scheduler_receipt(
    *,
    receipts: Path,
    attempt_date: str,
    checked_at: datetime,
    prior_paths: set[Path],
    execute: bool,
    force: bool,
    internal_continuation_only: bool,
) -> tuple[dict[str, Any], Path, str]:
    candidates = sorted(
        (
            path
            for path in receipts.glob(f"{attempt_date}_*.json")
            if path.is_file() and path.resolve() not in prior_paths
        ),
        reverse=True,
    )
    if not candidates:
        return {}, Path(), "heavy_scheduler_returned_without_fresh_receipt"
    errors: list[str] = []
    valid: list[tuple[datetime, Path, dict[str, Any]]] = []
    for path in candidates:
        payload, valid_json = _try_read_json(path)
        if not valid_json:
            errors.append(f"malformed_scheduler_receipt:{path.name}")
            continue
        blocker = _scheduler_receipt_blocker(
            payload=payload,
            path=path,
            attempt_date=attempt_date,
            checked_at=checked_at,
            execute=execute,
            force=force,
            internal_continuation_only=internal_continuation_only,
        )
        if blocker:
            errors.append(blocker)
            continue
        started_at = _parse_utc(str(payload["started_at_utc"]))
        valid.append((started_at, path, payload))
    if not valid:
        return {}, Path(), ";".join(errors) or "fresh_scheduler_receipt_invalid"
    _, path, payload = max(valid, key=lambda item: item[0])
    return payload, path, ""


def _scheduler_receipt_blocker(
    *,
    payload: dict[str, Any],
    path: Path,
    attempt_date: str,
    checked_at: datetime,
    execute: bool,
    force: bool,
    internal_continuation_only: bool,
) -> str:
    if payload.get("schema_version") != "thewiz.corrective_wizard_proof_scheduler.v1":
        return f"scheduler_receipt_schema_mismatch:{path.name}"
    if payload.get("attempt_date_utc") != attempt_date:
        return f"scheduler_receipt_attempt_date_mismatch:{path.name}"
    try:
        started_at = _parse_utc(str(payload.get("started_at_utc", "")))
    except ValueError:
        return f"scheduler_receipt_started_at_invalid:{path.name}"
    if started_at < checked_at - timedelta(seconds=1):
        return f"scheduler_receipt_predates_launcher:{path.name}"
    if payload.get("execution_requested") is not execute:
        return f"scheduler_receipt_execution_mode_mismatch:{path.name}"
    if payload.get("force_requested") is not force:
        return f"scheduler_receipt_force_mode_mismatch:{path.name}"
    if payload.get("internal_continuation_only") is not internal_continuation_only:
        return f"scheduler_receipt_continuation_mode_mismatch:{path.name}"
    if any(
        payload.get(field) is not False
        for field in (
            "candidate_promotion_authority",
            "order_submission_included",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    ):
        return f"scheduler_receipt_authority_violation:{path.name}"
    if payload.get("research_only") is not True:
        return f"scheduler_receipt_not_research_only:{path.name}"
    if not _scheduler_receipt_id_valid(payload):
        return f"scheduler_receipt_id_mismatch:{path.name}"
    if payload.get("final_immutable_receipt_required") is True:
        immutable_blocker = _immutable_scheduler_receipt_blocker(
            payload=payload,
            path=path,
        )
        if immutable_blocker:
            return immutable_blocker
    return ""


def _scheduler_receipt_id_valid(payload: dict[str, Any]) -> bool:
    receipt_id = str(payload.get("receipt_id", ""))
    unsigned = dict(payload)
    unsigned.pop("receipt_id", None)
    expected = "wizardproof_" + sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()[:20]
    return receipt_id == expected


def _immutable_scheduler_receipt_blocker(
    *,
    payload: dict[str, Any],
    path: Path,
) -> str:
    receipt_id = str(payload.get("receipt_id", ""))
    try:
        root = path.resolve().parents[3]
    except IndexError:
        return f"scheduler_immutable_receipt_root_invalid:{path.name}"
    immutable = (
        root / "data" / "research" / "wizard_proof_scheduler_receipts" / f"{receipt_id}.json"
    )
    if not immutable.is_file():
        return f"scheduler_immutable_receipt_missing:{path.name}"
    try:
        if immutable.read_bytes() != path.read_bytes():
            return f"scheduler_immutable_receipt_mismatch:{path.name}"
    except OSError:
        return f"scheduler_immutable_receipt_unreadable:{path.name}"
    return ""


def _stage3_evidence_complete(receipt: dict[str, Any], *, root: Path | None = None) -> bool:
    try:
        eligible = int(receipt.get("queue_eligible", 0) or 0)
        captured = int(receipt.get("responses_captured_after", 0) or 0)
        accepted = int(receipt.get("accepted_mode_evidence_cells", 0) or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        eligible > 0
        and captured >= eligible
        and accepted == eligible
        and receipt.get("parity_refresh_status") == "PASS"
        and receipt.get("credit_reconciliation_status")
        in {
            "PASS_RECONCILED",
            "REUSED_RECONCILIATION",
            "PRIOR_RECONCILIATION_VERIFIED",
        }
        and _ou_v4_scheduler_evidence_complete(receipt)
        and _ou_v5_scheduler_evidence_complete(receipt, root=root)
        and _ou_v6_scheduler_evidence_complete(receipt, root=root)
        and _capture_manifest_reconciliation_complete(receipt, root=root)
    )


def _ou_v4_scheduler_evidence_complete(receipt: dict[str, Any]) -> bool:
    if (
        receipt.get("ou_v5_prospectively_registered") is True
        or receipt.get("ou_v6_prospectively_registered") is True
    ):
        return True
    if receipt.get("ou_v4_prospectively_registered") is not True:
        return True
    try:
        required = int(receipt.get("ou_v4_required_responses", 0) or 0)
        available = int(receipt.get("ou_v4_responses_available", 0) or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        required == 8
        and available == required
        and receipt.get("ou_v4_holdout_status") == "COMPLETE"
        and receipt.get("ou_v4_evaluation_status") == "PASS"
        and receipt.get("ou_v4_response_accounting_valid") is True
        and receipt.get("ou_v4_automatic_activation") is False
        and receipt.get("ou_v4_research_only") is True
    )


def _ou_v5_scheduler_evidence_complete(
    receipt: dict[str, Any], *, root: Path | None = None
) -> bool:
    v6_registered = bool(
        receipt.get("ou_v6_prospectively_registered") is True
        or (
            root is not None
            and (
                (root / "config/wizard_ou_comparator_v6_holdout.json").is_file()
                or (root / "reports/active/wizard_ou_v6_holdout_receipt.json").is_file()
            )
        )
    )
    if v6_registered:
        return True
    if root is not None:
        return bool(
            validate_ou_v5_stage3_evidence(root=root, evidence=receipt).get("status") == "PASS"
        )
    if receipt.get("ou_v5_prospectively_registered") is not True:
        return True
    try:
        required = int(receipt.get("ou_v5_required_responses", 0) or 0)
        available = int(receipt.get("ou_v5_responses_available", 0) or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        required == 8
        and available == required
        and receipt.get("ou_v5_holdout_status") == "COMPLETE"
        and receipt.get("ou_v5_evaluation_status") == "PASS"
        and receipt.get("ou_v5_response_accounting_valid") is True
        and receipt.get("ou_v5_activation_status") == "APPLIED_RESEARCH_COMPARATOR_ONLY"
        and _safe_int(receipt.get("ou_v5_comparator_generation", 0)) == 5
        and receipt.get("ou_v5_proof_refresh_status") == "PASS"
        and _safe_int(receipt.get("ou_v5_proofs_refreshed", 0)) >= 8
        and receipt.get("ou_v5_activation_automatic") is False
        and receipt.get("ou_v5_research_only") is True
    )


def _ou_v6_scheduler_evidence_complete(
    receipt: dict[str, Any], *, root: Path | None = None
) -> bool:
    if root is not None:
        return bool(
            validate_ou_v6_stage3_evidence(root=root, evidence=receipt).get("status") == "PASS"
        )
    if receipt.get("ou_v6_prospectively_registered") is not True:
        return True
    try:
        required = int(receipt.get("ou_v6_required_responses", 0) or 0)
        available = int(receipt.get("ou_v6_responses_available", 0) or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        required == 8
        and available == required
        and receipt.get("ou_v6_holdout_status") == "COMPLETE"
        and receipt.get("ou_v6_evaluation_status") == "PASS"
        and receipt.get("ou_v6_response_accounting_valid") is True
        and receipt.get("ou_v6_activation_status") == "APPLIED_RESEARCH_COMPARATOR_ONLY"
        and _safe_int(receipt.get("ou_v6_comparator_generation", 0)) == 6
        and receipt.get("ou_v6_proof_refresh_status") == "PASS"
        and _safe_int(receipt.get("ou_v6_proofs_refreshed", 0)) >= 8
        and receipt.get("ou_v6_activation_automatic") is False
        and receipt.get("ou_v6_research_only") is True
        and receipt.get("ou_v6_final_successor_iteration") is True
        and receipt.get("ou_v6_successor_after_failure_allowed") is False
        and receipt.get("ou_v6_terminal_failure") is False
    )


def _capture_manifest_reconciliation_complete(
    receipt: dict[str, Any], *, root: Path | None = None
) -> bool:
    try:
        required = int(receipt.get("capture_reconciliation_required_calls", 0) or 0)
        completed = int(receipt.get("capture_reconciliation_completed_calls", 0) or 0)
        pending = int(receipt.get("capture_reconciliation_pending_calls", 0) or 0)
        blocked = int(receipt.get("capture_reconciliation_blocked_calls", 0) or 0)
    except (TypeError, ValueError):
        return False
    status = str(receipt.get("capture_reconciliation_status", ""))
    counts_complete = bool(completed == required and pending == 0 and blocked == 0)
    source_valid = bool(
        root is None
        or validate_capture_reconciliation_evidence(root=root, evidence=receipt).get("status")
        == "PASS"
    )
    return bool(
        _capture_manifest_binding_complete(receipt, root=root)
        and receipt.get("capture_manifest_accounting_valid") is True
        and receipt.get("capture_manifest_continuity_valid") is True
        and receipt.get("capture_manifest_drift_detected") is False
        and receipt.get("capture_reconciliation_valid") is True
        and receipt.get("capture_reconciliation_complete") is True
        and counts_complete
        and status == "PASS"
        and required > 0
        and source_valid
    )


def _capture_manifest_binding_complete(
    receipt: dict[str, Any], *, root: Path | None = None
) -> bool:
    manifest_id = str(receipt.get("capture_manifest_id", "")).strip()
    manifest_path = str(receipt.get("capture_manifest_immutable_path", "")).strip()
    manifest_sha256 = str(receipt.get("capture_manifest_immutable_sha256", "")).strip().lower()
    candidate_id = str(receipt.get("capture_manifest_candidate_id", "")).strip()
    candidate_path = str(receipt.get("capture_manifest_candidate_immutable_path", "")).strip()
    candidate_sha256 = (
        str(receipt.get("capture_manifest_candidate_immutable_sha256", "")).strip().lower()
    )
    reconciliation_id = str(receipt.get("capture_reconciliation_manifest_id", "")).strip()
    reconciliation_path = str(receipt.get("capture_reconciliation_manifest_path", "")).strip()
    reconciliation_sha256 = (
        str(receipt.get("capture_reconciliation_manifest_sha256", "")).strip().lower()
    )
    source_receipt_id = str(receipt.get("capture_manifest_source_receipt_id", "")).strip()
    source_receipt_path = str(receipt.get("capture_manifest_source_receipt_path", "")).strip()
    source_receipt_sha256 = (
        str(receipt.get("capture_manifest_source_receipt_sha256", "")).strip().lower()
    )
    source_artifacts_sha256 = (
        str(receipt.get("capture_manifest_source_artifacts_sha256", "")).strip().lower()
    )
    digest_valid = bool(
        len(manifest_sha256) == 64
        and all(character in "0123456789abcdef" for character in manifest_sha256)
    )
    source_digest_valid = bool(
        len(source_receipt_sha256) == 64
        and all(character in "0123456789abcdef" for character in source_receipt_sha256)
        and len(source_artifacts_sha256) == 64
        and all(character in "0123456789abcdef" for character in source_artifacts_sha256)
    )
    fields_complete = bool(
        receipt.get("capture_manifest_enforced") is True
        and receipt.get("capture_manifest_candidate_binding_valid") is True
        and receipt.get("capture_manifest_candidate_source_binding_valid") is True
        and manifest_id.startswith("wizardcapture_")
        and manifest_path.startswith("data/research/wizard_capture_manifests/")
        and manifest_path.endswith(".json")
        and digest_valid
        and candidate_id == manifest_id
        and candidate_path == manifest_path
        and candidate_sha256 == manifest_sha256
        and reconciliation_id == manifest_id
        and reconciliation_path == manifest_path
        and reconciliation_sha256 == manifest_sha256
        and source_receipt_id.startswith("wizardcapturesources_")
        and source_receipt_path
        == f"data/research/wizard_capture_manifest_sources/{manifest_id}.json"
        and source_digest_valid
    )
    if not fields_complete or root is None:
        return fields_complete
    source_binding = validate_capture_manifest_source_receipt(
        root=root,
        manifest_id=manifest_id,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    return bool(
        source_binding.get("status") == "PASS"
        and str(source_binding.get("receipt_id", "")) == source_receipt_id
        and str(source_binding.get("receipt_path", "")) == source_receipt_path
        and str(source_binding.get("receipt_sha256", "")).lower() == source_receipt_sha256
        and str(source_binding.get("source_artifacts_sha256", "")).lower()
        == source_artifacts_sha256
    )


def _reconciled_launcher_outcome(
    *,
    root: Path,
    scheduler_receipt: dict[str, Any],
    execute: bool,
    internal_continuation_reason: str,
    prior_stage4_contract_id: str = "",
) -> tuple[str, str]:
    internal_continuation_only = bool(internal_continuation_reason)
    scheduler_status = str(scheduler_receipt.get("status", ""))
    scheduler_blockers = [
        str(value) for value in scheduler_receipt.get("blockers", []) if str(value)
    ]
    if scheduler_status == "FAILED" or scheduler_status.startswith("BLOCKED"):
        detail = scheduler_blockers[0] if scheduler_blockers else scheduler_status
        return "HEAVY_SCHEDULER_BLOCKED", f"scheduler_outcome:{detail}"
    if not execute and scheduler_status == "PLANNED":
        return "HEAVY_SCHEDULER_PLANNED", ""
    if internal_continuation_reason == "stage3_dynamic_v2_refresh":
        activation = _validated_dynamic_v2_activation(root)
        if (
            scheduler_receipt.get("dynamic_v2_proof_refresh_status") == "PASS"
            and _safe_int(scheduler_receipt.get("dynamic_v2_comparator_generation", 0)) == 2
            and _safe_int(scheduler_receipt.get("dynamic_v2_proofs_refreshed", 0)) >= 4
            and activation
            and _dynamic_v2_refresh_complete(
                root=root,
                activation_id=str(activation.get("activation_id", "")),
            )
        ):
            return "INTERNAL_DYNAMIC_V2_REFRESH_COMPLETED", ""
        return (
            "INTERNAL_DYNAMIC_V2_REFRESH_INCOMPLETE",
            "reviewed_dynamic_v2_refresh_not_complete",
        )
    if internal_continuation_reason == "stage3_ou_v3_refresh":
        activation = _validated_ou_v3_activation(root)
        if (
            scheduler_receipt.get("ou_v3_proof_refresh_status") == "PASS"
            and _safe_int(scheduler_receipt.get("ou_v3_comparator_generation", 0)) == 3
            and _safe_int(scheduler_receipt.get("ou_v3_proofs_refreshed", 0)) >= 4
            and activation
            and _ou_v3_refresh_complete(
                root=root,
                activation_id=str(activation.get("activation_id", "")),
            )
        ):
            return "INTERNAL_OU_V3_REFRESH_COMPLETED", ""
        return (
            "INTERNAL_OU_V3_REFRESH_INCOMPLETE",
            "reviewed_ou_v3_refresh_not_complete",
        )
    if internal_continuation_reason == "stage3_ou_v4_refresh":
        activation = _validated_ou_v4_activation(root)
        if (
            scheduler_receipt.get("ou_v4_proof_refresh_status") == "PASS"
            and _safe_int(scheduler_receipt.get("ou_v4_comparator_generation", 0)) == 4
            and _safe_int(scheduler_receipt.get("ou_v4_proofs_refreshed", 0)) >= 8
            and activation
            and _ou_v4_refresh_complete(
                root=root,
                activation_id=str(activation.get("activation_id", "")),
            )
        ):
            return "INTERNAL_OU_V4_REFRESH_COMPLETED", ""
        return (
            "INTERNAL_OU_V4_REFRESH_INCOMPLETE",
            "reviewed_ou_v4_refresh_not_complete",
        )
    if internal_continuation_reason == "stage3_ou_v5_refresh":
        activation = _validated_ou_v5_activation(root)
        if (
            scheduler_receipt.get("ou_v5_proof_refresh_status") == "PASS"
            and _safe_int(scheduler_receipt.get("ou_v5_comparator_generation", 0)) == 5
            and _safe_int(scheduler_receipt.get("ou_v5_proofs_refreshed", 0)) >= 8
            and activation
            and _ou_v5_refresh_complete(
                root=root,
                activation_id=str(activation.get("activation_id", "")),
            )
        ):
            return "INTERNAL_OU_V5_REFRESH_COMPLETED", ""
        return (
            "INTERNAL_OU_V5_REFRESH_INCOMPLETE",
            "reviewed_ou_v5_refresh_not_complete",
        )
    if internal_continuation_reason == "stage3_local_reconciliation":
        if _stage3_evidence_complete(scheduler_receipt, root=root):
            return "INTERNAL_STAGE3_RECONCILIATION_COMPLETED", ""
        return (
            "INTERNAL_STAGE3_RECONCILIATION_INCOMPLETE",
            "stage3_local_evidence_still_incomplete",
        )
    if _stage3_evidence_complete(scheduler_receipt, root=root):
        if internal_continuation_only:
            if internal_continuation_reason == "stage5_handoff_recovery":
                rerun_status = str(scheduler_receipt.get("registered_rerun_status", ""))
                learning_status = str(
                    scheduler_receipt.get("registered_learning_handoff_status", "")
                )
                if (
                    rerun_status == "ALREADY_COMPLETE"
                    and learning_status
                    in {
                        "PASS_RESEARCH_LEARNING_GATES",
                        "REJECTED_RESEARCH_LEARNING_GATES",
                    }
                    and _terminal_learning_status_valid(
                        root=root,
                        learning=_read_json(
                            root / "reports" / "active" / "registered_learning_research_status.json"
                        ),
                        execution_id=str(_current_execution_receipt(root).get("execution_id", "")),
                    )
                ):
                    return "INTERNAL_STAGE5_RECOVERY_COMPLETED", ""
                return (
                    "INTERNAL_STAGE5_RECOVERY_INCOMPLETE",
                    "registered_learning_handoff_not_terminal",
                )
            if (
                internal_continuation_reason == "stage4_generation"
                and scheduler_receipt.get("registered_rerun_status")
                == "PASS_REGISTERED_RERUN_ACCOUNTED"
                and _stage4_generation_continuation_complete(
                    root=root,
                    scheduler_receipt=scheduler_receipt,
                    prior_contract_id=prior_stage4_contract_id,
                )
            ):
                return "INTERNAL_REGISTERED_CONTINUATION_COMPLETED", ""
            if (
                internal_continuation_reason == "stage4_active_execution"
                and scheduler_receipt.get("registered_rerun_status")
                == "PASS_REGISTERED_RERUN_ACCOUNTED"
                and _stage4_active_execution_continuation_complete(
                    root=root,
                    scheduler_receipt=scheduler_receipt,
                    prior_contract_id=prior_stage4_contract_id,
                )
            ):
                return "INTERNAL_REGISTERED_CONTINUATION_COMPLETED", ""
            return (
                "INTERNAL_REGISTERED_CONTINUATION_INCOMPLETE",
                "registered_rerun_generation_not_terminal_or_not_rotated",
            )
        return "HEAVY_SCHEDULER_EVIDENCE_COMPLETE", ""
    if not _capture_manifest_binding_complete(scheduler_receipt, root=root):
        return (
            "HEAVY_SCHEDULER_RECONCILED_INCOMPLETE",
            "scheduler_research_incomplete:capture_manifest_binding_incomplete",
        )
    detail = scheduler_blockers[0] if scheduler_blockers else scheduler_status
    if not scheduler_blockers and _capture_manifest_reconciliation_complete(
        scheduler_receipt, root=root
    ):
        return (
            "HEAVY_SCHEDULER_RECONCILED_RESEARCH_INCOMPLETE",
            f"scheduler_research_incomplete:{detail or 'unknown_status'}",
        )
    return (
        "HEAVY_SCHEDULER_RECONCILED_INCOMPLETE",
        f"scheduler_research_incomplete:{detail or 'unknown_status'}",
    )


def _internal_registered_continuation_reason(
    *, root: Path, latest: dict[str, Any], same_day_attempt_state: str
) -> str:
    if same_day_attempt_state != "ATTEMPTED":
        return ""
    if _dynamic_v2_refresh_needed(root=root, latest=latest):
        return "stage3_dynamic_v2_refresh"
    if not _ou_v6_final_successor_active(latest):
        if _ou_v5_refresh_needed(root=root, latest=latest):
            return "stage3_ou_v5_refresh"
        if _ou_v4_refresh_needed(root=root, latest=latest):
            return "stage3_ou_v4_refresh"
        if _ou_v3_refresh_needed(root=root, latest=latest):
            return "stage3_ou_v3_refresh"
    if _stage3_local_reconciliation_needed(root=root, latest=latest):
        return "stage3_local_reconciliation"
    proof_complete = bool(
        _stage3_evidence_complete(latest, root=root)
        and latest.get("registered_rerun_status") == "PASS_REGISTERED_RERUN_ACCOUNTED"
    )
    if not proof_complete:
        return ""
    checkpoint = root / "reports" / "active" / "seven_stage_goal_checkpoint.csv"
    if not checkpoint.is_file():
        return ""
    try:
        with checkpoint.open(encoding="utf-8", newline="") as handle:
            stage_four = next(
                (row for row in csv.DictReader(handle) if str(row.get("stage", "")).strip() == "4"),
                None,
            )
    except (OSError, csv.Error):
        return ""
    if not stage_four:
        return ""
    if (
        str(stage_four.get("status", "")) != "PASS"
        and str(stage_four.get("next_action", ""))
        == "advance_registered_rerun_to_unaccounted_current_family_cohort"
    ):
        return "stage4_generation"
    if (
        str(stage_four.get("status", "")) != "PASS"
        and str(stage_four.get("next_action", ""))
        == "execute_active_registered_rerun_after_ready_chain"
    ):
        return "stage4_active_execution"
    if str(stage_four.get("status", "")) == "PASS" and _stage5_recovery_needed(
        root=root,
        stage_four=stage_four,
    ):
        return "stage5_handoff_recovery"
    return ""


def _ou_v6_final_successor_active(latest: dict[str, Any]) -> bool:
    """Avoid loading superseded OU activation stacks once v6 owns the lane."""

    return bool(
        latest.get("ou_v6_prospectively_registered") is True
        and latest.get("ou_v6_final_successor_iteration") is True
        and latest.get("ou_v6_successor_after_failure_allowed") is False
        and latest.get("ou_v6_research_only") is True
        and latest.get("ou_v6_activation_automatic") is False
        and latest.get("candidate_promotion_authority") is not True
        and latest.get("testnet_order_authority") is not True
        and latest.get("live_trading_authorized") is not True
    )


def _stage3_local_reconciliation_needed(*, root: Path, latest: dict[str, Any]) -> bool:
    pending_capture_calls = max(
        _safe_int(latest.get("capture_manifest_pending_calls")),
        _safe_int(latest.get("capture_reconciliation_pending_calls")),
    )
    blocked_capture_calls = _safe_int(latest.get("capture_reconciliation_blocked_calls"))
    if pending_capture_calls > 0 or blocked_capture_calls > 0:
        return False
    try:
        eligible = int(latest.get("queue_eligible", 0) or 0)
        responses = int(latest.get("responses_captured_after", 0) or 0)
    except (TypeError, ValueError):
        return False
    accounting_valid = latest.get("credit_reconciliation_status") in {
        "PASS_RECONCILED",
        "REUSED_RECONCILIATION",
        "PRIOR_RECONCILIATION_VERIFIED",
    }
    stage3_incomplete = not _stage3_evidence_complete(latest, root=root)
    if not (eligible > 0 and responses >= eligible and accounting_valid and stage3_incomplete):
        return False
    prior = str(latest.get("stage3_local_evidence_fingerprint", "")).strip()
    return not prior or prior != _stage3_local_evidence_fingerprint(root)


def _stage3_local_evidence_fingerprint(root: Path) -> str:
    material: list[dict[str, str]] = []
    for relative in STAGE3_LOCAL_EVIDENCE_PATHS:
        path = root / relative
        material.append(
            {
                "path": relative,
                "sha256": (sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"),
            }
        )
    return sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _dynamic_v2_refresh_needed(*, root: Path, latest: dict[str, Any]) -> bool:
    activation = _validated_dynamic_v2_activation(root)
    if not activation:
        return False
    if latest.get("credit_reconciliation_status") not in {
        "PASS_RECONCILED",
        "REUSED_RECONCILIATION",
        "PRIOR_RECONCILIATION_VERIFIED",
    }:
        return False
    activation_id = str(activation.get("activation_id", "")).strip()
    return bool(
        activation_id
        and not _dynamic_v2_refresh_complete(
            root=root,
            activation_id=activation_id,
        )
    )


def _validated_dynamic_v2_activation(root: Path) -> dict[str, Any]:
    active = _read_json(root / "reports" / "active" / "wizard_dynamic_v2_activation_status.json")
    raw_path = str(active.get("immutable_activation_path", "")).strip()
    expected_hash = str(active.get("immutable_activation_sha256", "")).strip()
    path = Path(raw_path)
    immutable_path = path if path.is_absolute() else root / path
    evidence_root = root / "data" / "research" / "wizard_dynamic_comparator_activations"
    try:
        immutable_path.resolve().relative_to(evidence_root.resolve())
    except (OSError, ValueError):
        return {}
    if (
        active.get("schema_version") != "thewiz.wizard_dynamic_v2_activation.v2"
        or active.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY"
        or active.get("apply_requested") is not True
        or _safe_int(active.get("comparator_generation", 0)) != 2
        or not str(active.get("reviewer", "")).strip()
        or len(str(active.get("review_note", "")).strip()) < 20
        or not str(active.get("review_packet_id_submitted", "")).strip()
        or not raw_path
        or not immutable_path.is_file()
        or len(expected_hash) != 64
        or sha256(immutable_path.read_bytes()).hexdigest() != expected_hash
        or any(
            active.get(field) is not False
            for field in (
                "candidate_promotion_authority",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
    ):
        return {}
    receipt = _read_json(immutable_path)
    activation_id = str(receipt.get("activation_id", "")).strip()
    stable = {
        key: value
        for key, value in receipt.items()
        if key not in {"evaluated_at_utc", "activation_id"}
    }
    expected_id = (
        "dynamicv2activation_" + sha256(_canonical_json(stable).encode("utf-8")).hexdigest()[:20]
    )
    snapshot_path = _resolve_relative_path(
        root,
        str(receipt.get("source_proof_snapshot_path", "")),
    )
    snapshot_hash = str(receipt.get("source_proof_snapshot_sha256", "")).strip()
    if (
        receipt.get("schema_version") != "thewiz.wizard_dynamic_v2_activation.v2"
        or receipt.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY"
        or receipt.get("apply_requested") is not True
        or _safe_int(receipt.get("comparator_generation", 0)) != 2
        or activation_id != expected_id
        or immutable_path.stem != activation_id
        or active.get("activation_id") != activation_id
        or active.get("reviewer") != receipt.get("reviewer")
        or active.get("review_note") != receipt.get("review_note")
        or active.get("review_packet_id_submitted") != receipt.get("review_packet_id_submitted")
        or any(active.get(key) != value for key, value in receipt.items())
        or snapshot_path is None
        or not snapshot_path.is_file()
        or len(snapshot_hash) != 64
        or sha256(snapshot_path.read_bytes()).hexdigest() != snapshot_hash
        or any(
            receipt.get(field) is not False
            for field in (
                "candidate_promotion_authority",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
        or not _dynamic_v2_supreme_binding_valid(root=root, activation=receipt)
    ):
        return {}
    return active


def _dynamic_v2_supreme_binding_valid(*, root: Path, activation: dict[str, Any]) -> bool:
    raw_path = str(activation.get("supreme_review_path", "")).strip()
    expected_hash = str(activation.get("supreme_review_sha256", "")).strip().lower()
    review_path = _resolve_relative_path(root, raw_path)
    review_root = root / "data" / "research" / "wizard_dynamic_supreme_reviews"
    try:
        path_safe = bool(
            review_path is not None
            and review_path.is_file()
            and review_path.resolve().is_relative_to(review_root.resolve())
        )
    except (OSError, ValueError):
        path_safe = False
    if (
        not path_safe
        or len(expected_hash) != 64
        or sha256(review_path.read_bytes()).hexdigest() != expected_hash
    ):
        return False
    review = _read_json(review_path)
    review_id = str(review.get("review_id", "")).strip()
    stable = dict(review)
    stable.pop("review_id", None)
    expected_review_id = (
        "dynamicv2supreme_" + sha256(_canonical_json(stable).encode("utf-8")).hexdigest()[:20]
    )
    packet_path = _resolve_relative_path(root, str(activation.get("review_packet_path", "")))
    packet_hash = str(activation.get("review_packet_sha256", "")).strip().lower()
    return bool(
        review.get("schema_version") == "thewiz.wizard_dynamic_v2_supreme_review.v1"
        and review.get("status") == "PASS_ADVISORY_ONLY"
        and review.get("recommendation") == "RECOMMEND_EXPLICIT_RESEARCH_COMPARATOR_ACTIVATION"
        and review.get("activation_preflight_status") == "READY_REQUIRES_EXPLICIT_APPLY"
        and not review.get("blockers")
        and review_id == expected_review_id
        and review_path.name == f"{review_id}.json"
        and activation.get("supreme_review_id") == review_id
        and review.get("review_packet_id") == activation.get("review_packet_id")
        and review.get("immutable_review_packet_path") == activation.get("review_packet_path")
        and review.get("immutable_review_packet_sha256") == activation.get("review_packet_sha256")
        and packet_path is not None
        and packet_path.is_file()
        and len(packet_hash) == 64
        and sha256(packet_path.read_bytes()).hexdigest() == packet_hash
        and _safe_int(review.get("required_cells")) == 4
        and _safe_int(review.get("passed_cells")) == 4
        and review.get("cohorts_disjoint") is True
        and review.get("raw_bindings_valid") is True
        and _safe_float(review.get("max_abs_reconstruction_error")) <= 1e-9
        and review.get("human_approval_required") is True
        and review.get("human_approval_recorded") is False
        and review.get("activation_applied_by_supreme_team") is False
        and review.get("research_only") is True
        and all(
            review.get(field) is False
            for field in (
                "candidate_promotion_authority",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
    )


def _dynamic_v2_refresh_complete(*, root: Path, activation_id: str) -> bool:
    refresh = _read_json(
        root / "reports" / "active" / "wizard_dynamic_v2_proof_refresh_status.json"
    )
    try:
        captured = int(refresh.get("captured_dynamic_rows", 0) or 0)
        exact = int(refresh.get("exact_dynamic_rows", 0) or 0)
        refreshed = int(refresh.get("refreshed_dynamic_rows", 0) or 0)
        generation = int(refresh.get("comparator_generation", 0) or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        activation_id
        and refresh.get("schema_version") == "thewiz.wizard_dynamic_v2_proof_refresh.v1"
        and refresh.get("status") == "PASS"
        and refresh.get("activation_id") == activation_id
        and generation == 2
        and captured >= 8
        and exact == captured
        and refreshed >= 8
        and refresh.get("research_only") is True
        and refresh.get("original_comparator_mutated") is False
        and refresh.get("raw_vendor_evidence_mutated") is False
        and all(
            refresh.get(field) is False
            for field in (
                "candidate_promotion_authority",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
    )


def _ou_v4_refresh_needed(*, root: Path, latest: dict[str, Any]) -> bool:
    activation = _validated_ou_v4_activation(root)
    if not activation:
        return False
    if latest.get("credit_reconciliation_status") not in {
        "PASS_RECONCILED",
        "REUSED_RECONCILIATION",
        "PRIOR_RECONCILIATION_VERIFIED",
    }:
        return False
    activation_id = str(activation.get("activation_id", "")).strip()
    return bool(
        activation_id
        and not _ou_v4_refresh_complete(
            root=root,
            activation_id=activation_id,
        )
    )


def _validated_ou_v4_activation(root: Path) -> dict[str, Any]:
    from quant_platform.wizard_ou_v4_comparator_activation import (
        load_validated_ou_v4_activation,
    )

    try:
        return load_validated_ou_v4_activation(root=root) or {}
    except ValueError:
        return {}


def _ou_v4_refresh_complete(*, root: Path, activation_id: str) -> bool:
    refresh = _read_json(root / "reports" / "active" / "wizard_ou_v4_proof_refresh_status.json")
    try:
        captured = int(refresh.get("captured_ou_rows", 0) or 0)
        exact = int(refresh.get("exact_ou_rows", 0) or 0)
        refreshed = int(refresh.get("refreshed_ou_rows", 0) or 0)
        generation = int(refresh.get("comparator_generation", 0) or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        activation_id
        and refresh.get("schema_version") == "thewiz.wizard_ou_v4_proof_refresh.v1"
        and refresh.get("status") == "PASS"
        and refresh.get("activation_id") == activation_id
        and generation == 4
        and captured >= 4
        and exact == captured
        and refreshed >= 4
        and refresh.get("research_only") is True
        and refresh.get("raw_vendor_evidence_mutated") is False
        and all(
            refresh.get(field) is False
            for field in (
                "candidate_promotion_authority",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
    )


def _ou_v5_refresh_needed(*, root: Path, latest: dict[str, Any]) -> bool:
    activation = _validated_ou_v5_activation(root)
    if not activation:
        return False
    if latest.get("credit_reconciliation_status") not in {
        "PASS_RECONCILED",
        "REUSED_RECONCILIATION",
        "PRIOR_RECONCILIATION_VERIFIED",
    }:
        return False
    activation_id = str(activation.get("activation_id", "")).strip()
    return bool(
        activation_id
        and not _ou_v5_refresh_complete(
            root=root,
            activation_id=activation_id,
        )
    )


def _validated_ou_v5_activation(root: Path) -> dict[str, Any]:
    from quant_platform.wizard_ou_v5_comparator_activation import (
        load_validated_ou_v5_activation,
    )

    try:
        return load_validated_ou_v5_activation(root=root) or {}
    except ValueError:
        return {}


def _ou_v5_refresh_complete(*, root: Path, activation_id: str) -> bool:
    refresh = _read_json(root / "reports" / "active" / "wizard_ou_v5_proof_refresh_status.json")
    try:
        captured = int(refresh.get("captured_ou_rows", 0) or 0)
        exact = int(refresh.get("exact_ou_rows", 0) or 0)
        refreshed = int(refresh.get("refreshed_ou_rows", 0) or 0)
        generation = int(refresh.get("comparator_generation", 0) or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        activation_id
        and refresh.get("schema_version") == "thewiz.wizard_ou_v5_proof_refresh.v1"
        and refresh.get("status") == "PASS"
        and refresh.get("activation_id") == activation_id
        and generation == 5
        and captured >= 8
        and exact == captured
        and refreshed >= 8
        and refresh.get("research_only") is True
        and refresh.get("raw_vendor_evidence_mutated") is False
        and all(
            refresh.get(field) is False
            for field in (
                "candidate_promotion_authority",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
    )


def _ou_v3_refresh_needed(*, root: Path, latest: dict[str, Any]) -> bool:
    activation = _validated_ou_v3_activation(root)
    if not activation:
        return False
    if latest.get("credit_reconciliation_status") not in {
        "PASS_RECONCILED",
        "REUSED_RECONCILIATION",
        "PRIOR_RECONCILIATION_VERIFIED",
    }:
        return False
    activation_id = str(activation.get("activation_id", "")).strip()
    return bool(
        activation_id
        and not _ou_v3_refresh_complete(
            root=root,
            activation_id=activation_id,
        )
    )


def _validated_ou_v3_activation(root: Path) -> dict[str, Any]:
    active = _read_json(root / "reports" / "active" / "wizard_ou_v3_activation_status.json")
    raw_path = str(active.get("immutable_activation_path", "")).strip()
    expected_hash = str(active.get("immutable_activation_sha256", "")).strip()
    path = Path(raw_path)
    immutable_path = path if path.is_absolute() else root / path
    evidence_root = root / "data" / "research" / "wizard_ou_v3_comparator_activations"
    try:
        immutable_path.resolve().relative_to(evidence_root.resolve())
    except (OSError, ValueError):
        return {}
    if (
        active.get("schema_version") != "thewiz.wizard_ou_v3_activation.v1"
        or active.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY"
        or active.get("apply_requested") is not True
        or _safe_int(active.get("comparator_generation", 0)) != 3
        or not str(active.get("reviewer", "")).strip()
        or len(str(active.get("review_note", "")).strip()) < 20
        or not str(active.get("review_packet_id_submitted", "")).strip()
        or active.get("review_packet_id_submitted") != active.get("review_packet_id")
        or not raw_path
        or not immutable_path.is_file()
        or len(expected_hash) != 64
        or sha256(immutable_path.read_bytes()).hexdigest() != expected_hash
        or any(
            active.get(field) is not False
            for field in (
                "candidate_promotion_authority",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
    ):
        return {}
    receipt = _read_json(immutable_path)
    activation_id = str(receipt.get("activation_id", "")).strip()
    stable = {
        key: value
        for key, value in receipt.items()
        if key not in {"evaluated_at_utc", "activation_id"}
    }
    expected_id = (
        "ouv3activation_" + sha256(_canonical_json(stable).encode("utf-8")).hexdigest()[:20]
    )
    snapshot_path = _resolve_relative_path(
        root,
        str(receipt.get("source_proof_snapshot_path", "")),
    )
    snapshot_hash = str(receipt.get("source_proof_snapshot_sha256", "")).strip()
    if (
        receipt.get("schema_version") != "thewiz.wizard_ou_v3_activation.v1"
        or receipt.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY"
        or receipt.get("apply_requested") is not True
        or _safe_int(receipt.get("comparator_generation", 0)) != 3
        or activation_id != expected_id
        or immutable_path.stem != activation_id
        or active.get("activation_id") != activation_id
        or active.get("reviewer") != receipt.get("reviewer")
        or active.get("review_note") != receipt.get("review_note")
        or active.get("review_packet_id_submitted") != receipt.get("review_packet_id_submitted")
        or snapshot_path is None
        or not snapshot_path.is_file()
        or len(snapshot_hash) != 64
        or sha256(snapshot_path.read_bytes()).hexdigest() != snapshot_hash
        or any(
            receipt.get(field) is not False
            for field in (
                "candidate_promotion_authority",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
    ):
        return {}
    return active


def _ou_v3_refresh_complete(*, root: Path, activation_id: str) -> bool:
    refresh = _read_json(root / "reports" / "active" / "wizard_ou_v3_proof_refresh_status.json")
    try:
        captured = int(refresh.get("captured_ou_rows", 0) or 0)
        exact = int(refresh.get("exact_ou_rows", 0) or 0)
        refreshed = int(refresh.get("refreshed_ou_rows", 0) or 0)
        generation = int(refresh.get("comparator_generation", 0) or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        activation_id
        and refresh.get("schema_version") == "thewiz.wizard_ou_v3_proof_refresh.v1"
        and refresh.get("status") == "PASS"
        and refresh.get("activation_id") == activation_id
        and generation == 3
        and captured >= 4
        and exact == captured
        and refreshed >= 4
        and refresh.get("research_only") is True
        and refresh.get("raw_vendor_evidence_mutated") is False
        and all(
            refresh.get(field) is False
            for field in (
                "candidate_promotion_authority",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
    )


def _resolve_relative_path(root: Path, raw_path: str) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return None
    try:
        resolved = (root / path).resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return resolved


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _stage5_recovery_needed(*, root: Path, stage_four: dict[str, Any]) -> bool:
    receipt = _current_execution_receipt(root)
    if not receipt:
        return False
    contract_id = str(receipt.get("contract_id", "")).strip()
    evidence = str(stage_four.get("evidence_progress", ""))
    try:
        accepted_breadth = bool(
            int(receipt.get("stage4_final_one_x_survivors", 0) or 0) >= 3
            and int(receipt.get("stage4_independent_supporting_clusters", 0) or 0) >= 3
            and int(receipt.get("stage4_independent_full_survivor_clusters", 0) or 0) >= 3
            and int(receipt.get("stage4_independent_supporting_pairs", 0) or 0) >= 3
            and int(receipt.get("stage4_independent_full_survivor_pairs", 0) or 0) >= 3
        )
    except (TypeError, ValueError):
        accepted_breadth = False
    if (
        not contract_id
        or f"active_contract_id={contract_id}" not in evidence
        or receipt.get("conclusion_status") != "ACCEPTED_REGISTERED_SURVIVORS"
        or not accepted_breadth
        or any(
            receipt.get(field) is not False
            for field in (
                "order_submission_performed",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
    ):
        return False
    learning = _read_json(root / "reports" / "active" / "registered_learning_research_status.json")
    execution_id = str(receipt.get("execution_id", "")).strip()
    learning_execution_id = str(learning.get("registered_execution_id", "")).strip()
    learning_status = str(learning.get("status", "")).strip()
    terminal_valid = _terminal_learning_status_valid(
        root=root,
        learning=learning,
        execution_id=execution_id,
    )
    if terminal_valid:
        return False
    return bool(
        execution_id
        and (
            not learning
            or learning_execution_id != execution_id
            or learning_status
            in {
                "HANDOFF_FAILED",
                "FAILED",
                "BLOCKED_LOCK",
                "BLOCKED_STAGE4",
                "BLOCKED_STAGE4_NO_ACCEPTED_SURVIVOR",
                "BLOCKED_STAGE4_CURRENT_FAMILY_INCOMPLETE",
                "PASS_RESEARCH_LEARNING_GATES",
                "REJECTED_RESEARCH_LEARNING_GATES",
            }
        )
    )


def _active_stage4_contract_id(root: Path) -> str:
    contract = _read_json(root / "reports" / "active" / "registered_research_rerun_contract.json")
    return str(contract.get("contract_id", "")).strip()


def _stage4_generation_continuation_complete(
    *,
    root: Path,
    scheduler_receipt: dict[str, Any],
    prior_contract_id: str,
) -> bool:
    if not prior_contract_id:
        return False
    active_contract = _read_json(
        root / "reports" / "active" / "registered_research_rerun_contract.json"
    )
    contract_id = str(active_contract.get("contract_id", "")).strip()
    try:
        generation = int(active_contract.get("generation", 0) or 0)
    except (TypeError, ValueError):
        return False
    if (
        active_contract.get("schema_version") != "thewiz.corrective_registered_rerun.v1"
        or not contract_id
        or contract_id == prior_contract_id
        or active_contract.get("prior_contract_id") != prior_contract_id
        or generation < 2
        or any(
            active_contract.get(field) is not False
            for field in (
                "promotion_authority",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
    ):
        return False

    execution_status = _read_json(
        root / "reports" / "active" / "registered_research_rerun_execution.json"
    )
    status_path = _resolve_relative_path(
        root, str(execution_status.get("execution_receipt_path", "")).strip()
    )
    scheduler_path = _resolve_relative_path(
        root,
        str(scheduler_receipt.get("registered_rerun_receipt_path", "")).strip(),
    )
    expected_root = root / "data" / "research" / "registered_rerun_executions"
    try:
        path_safe = bool(
            status_path is not None
            and scheduler_path is not None
            and status_path == scheduler_path
            and status_path.resolve().is_relative_to(expected_root.resolve())
        )
    except (OSError, ValueError):
        path_safe = False
    expected_hash = str(execution_status.get("execution_receipt_sha256", "")).strip()
    if (
        not path_safe
        or status_path is None
        or not status_path.is_file()
        or len(expected_hash) != 64
        or sha256(status_path.read_bytes()).hexdigest() != expected_hash
    ):
        return False
    execution = _read_json(status_path)
    embedded_hash = str(execution.get("receipt_sha256", "")).strip()
    unsigned = dict(execution)
    unsigned.pop("receipt_sha256", None)
    if (
        execution.get("schema_version") != "thewiz.corrective_registered_rerun_execution.v1"
        or execution.get("status") != "PASS_REGISTERED_RERUN_ACCOUNTED"
        or execution.get("contract_id") != contract_id
        or execution.get("conclusion_status")
        not in {
            "ACCEPTED_REGISTERED_SURVIVORS",
            "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS",
        }
        or embedded_hash != sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
        or any(
            execution.get(field) is not False
            for field in (
                "order_submission_performed",
                "promotion_authority",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
    ):
        return False

    checkpoint = root / "reports" / "active" / "seven_stage_goal_checkpoint.csv"
    try:
        with checkpoint.open(encoding="utf-8", newline="") as handle:
            stage_rows = [
                row for row in csv.DictReader(handle) if str(row.get("stage", "")).strip() == "4"
            ]
    except (OSError, csv.Error):
        return False
    if len(stage_rows) != 1:
        return False
    stage_four = stage_rows[0]
    evidence = str(stage_four.get("evidence_progress", ""))
    return bool(
        str(stage_four.get("status", "")) == "PASS"
        and f"active_contract_id={contract_id}" in evidence
        and "final_receipt_conclusion_bound=True" in evidence
        and (
            "stage4_terminal_outcome=ACCEPTED_VALID_INDEPENDENT_SURVIVORS" in evidence
            or "stage4_terminal_outcome=CONCLUSIVE_REJECTION_CURRENT_FAMILY" in evidence
        )
        and not _truthy(stage_four.get("testnet_order_authority"))
        and not _truthy(stage_four.get("live_trading_authorized"))
    )


def _stage4_active_execution_continuation_complete(
    *,
    root: Path,
    scheduler_receipt: dict[str, Any],
    prior_contract_id: str,
) -> bool:
    """Verify one local active-contract execution without requiring another vendor call."""

    if not prior_contract_id:
        return False
    active_contract = _read_json(
        root / "reports" / "active" / "registered_research_rerun_contract.json"
    )
    active_contract_id = str(active_contract.get("contract_id", "")).strip()
    if (
        active_contract.get("schema_version") != "thewiz.corrective_registered_rerun.v1"
        or not active_contract_id
        or any(
            active_contract.get(field) is not False
            for field in (
                "promotion_authority",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
    ):
        return False

    execution_status = _read_json(
        root / "reports" / "active" / "registered_research_rerun_execution.json"
    )
    status_path = _resolve_relative_path(
        root, str(execution_status.get("execution_receipt_path", "")).strip()
    )
    scheduler_path = _resolve_relative_path(
        root,
        str(scheduler_receipt.get("registered_rerun_receipt_path", "")).strip(),
    )
    expected_root = root / "data" / "research" / "registered_rerun_executions"
    try:
        path_safe = bool(
            status_path is not None
            and scheduler_path is not None
            and status_path == scheduler_path
            and status_path.resolve().is_relative_to(expected_root.resolve())
        )
    except (OSError, ValueError):
        path_safe = False
    expected_hash = str(execution_status.get("execution_receipt_sha256", "")).strip()
    if (
        not path_safe
        or status_path is None
        or not status_path.is_file()
        or len(expected_hash) != 64
        or sha256(status_path.read_bytes()).hexdigest() != expected_hash
    ):
        return False
    execution = _read_json(status_path)
    embedded_hash = str(execution.get("receipt_sha256", "")).strip()
    unsigned = dict(execution)
    unsigned.pop("receipt_sha256", None)
    if (
        execution.get("schema_version") != "thewiz.corrective_registered_rerun_execution.v1"
        or execution.get("status") != "PASS_REGISTERED_RERUN_ACCOUNTED"
        or execution.get("contract_id") != prior_contract_id
        or execution.get("conclusion_status")
        not in {
            "ACCEPTED_REGISTERED_SURVIVORS",
            "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS",
        }
        or embedded_hash != sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
        or any(
            execution.get(field) is not False
            for field in (
                "order_submission_performed",
                "promotion_authority",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
    ):
        return False

    checkpoint = root / "reports" / "active" / "seven_stage_goal_checkpoint.csv"
    try:
        with checkpoint.open(encoding="utf-8", newline="") as handle:
            stage_rows = [
                row for row in csv.DictReader(handle) if str(row.get("stage", "")).strip() == "4"
            ]
    except (OSError, csv.Error):
        return False
    if len(stage_rows) != 1:
        return False
    stage_four = stage_rows[0]
    if _truthy(stage_four.get("testnet_order_authority")) or _truthy(
        stage_four.get("live_trading_authorized")
    ):
        return False

    stage_status = str(stage_four.get("status", ""))
    next_action = str(stage_four.get("next_action", ""))
    evidence = str(stage_four.get("evidence_progress", ""))
    if stage_status == "PASS":
        return bool(
            active_contract_id == prior_contract_id
            and f"active_contract_id={prior_contract_id}" in evidence
            and "final_receipt_conclusion_bound=True" in evidence
            and (
                "stage4_terminal_outcome=ACCEPTED_VALID_INDEPENDENT_SURVIVORS" in evidence
                or "stage4_terminal_outcome=CONCLUSIVE_REJECTION_CURRENT_FAMILY" in evidence
            )
        )

    if stage_status not in {"IN_PROGRESS", "BLOCKED"}:
        return False
    if next_action == "advance_registered_rerun_to_unaccounted_current_family_cohort":
        return active_contract_id == prior_contract_id
    if next_action == "execute_active_registered_rerun_after_ready_chain":
        return bool(
            active_contract_id != prior_contract_id
            and active_contract.get("prior_contract_id") == prior_contract_id
        )
    return False


def _current_execution_receipt(root: Path) -> dict[str, Any]:
    execution_status = _read_json(
        root / "reports" / "active" / "registered_research_rerun_execution.json"
    )
    raw_path = str(execution_status.get("execution_receipt_path", "")).strip()
    expected_hash = str(execution_status.get("execution_receipt_sha256", "")).strip()
    path = Path(raw_path)
    receipt_path = path if path.is_absolute() else root / path
    if (
        not raw_path
        or not receipt_path.is_file()
        or len(expected_hash) != 64
        or sha256(receipt_path.read_bytes()).hexdigest() != expected_hash
    ):
        return {}
    receipt = _read_json(receipt_path)
    return receipt if receipt else {}


def _terminal_learning_status_valid(
    *, root: Path, learning: dict[str, Any], execution_id: str
) -> bool:
    status = str(learning.get("status", "")).strip()
    if status not in {
        "PASS_RESEARCH_LEARNING_GATES",
        "REJECTED_RESEARCH_LEARNING_GATES",
        "ALREADY_COMPLETE",
    }:
        return False
    raw_path = str(learning.get("learning_receipt_path", "")).strip()
    expected_hash = str(learning.get("learning_receipt_sha256", "")).strip()
    path = Path(raw_path)
    receipt_path = path if path.is_absolute() else root / path
    expected_parent = (root / "data" / "research" / "registered_learning").resolve()
    try:
        receipt_path.resolve().relative_to(expected_parent)
    except (OSError, ValueError):
        return False
    if (
        not execution_id
        or learning.get("registered_execution_id") != execution_id
        or not raw_path
        or not receipt_path.is_file()
        or len(expected_hash) != 64
        or sha256(receipt_path.read_bytes()).hexdigest() != expected_hash
    ):
        return False
    receipt = _read_json(receipt_path)
    receipt_status = str(receipt.get("status", "")).strip()
    if receipt_status not in {
        "PASS_RESEARCH_LEARNING_GATES",
        "REJECTED_RESEARCH_LEARNING_GATES",
    }:
        return False
    gate_pass = receipt_status == "PASS_RESEARCH_LEARNING_GATES"
    authority_fields = (
        "order_submission_performed",
        "promotion_authority",
        "testnet_candidate_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if not (
        receipt.get("schema_version") == "thewiz.corrective_registered_learning.v1"
        and receipt.get("registered_execution_id") == execution_id
        and (status == "ALREADY_COMPLETE" or receipt_status == status)
        and bool(receipt.get("stage5_research_gate_pass", False)) == gate_pass
        and bool(learning.get("stage5_research_gate_pass", False)) == gate_pass
        and all(receipt.get(field) is False for field in authority_fields)
    ):
        return False

    # Import only after the cheap checks because normal daily launcher runs should
    # not pay the registered-learning dependency cost.
    from quant_platform.orchestration.corrective_registered_learning import (
        latest_verified_registered_learning,
    )

    audit = latest_verified_registered_learning(root=root)
    expected_audit_status = (
        "PASS_VERIFIED_REGISTERED_LEARNING_ACCEPTANCE"
        if gate_pass
        else "PASS_VERIFIED_REGISTERED_LEARNING_REJECTION"
    )
    audited_raw_path = str(audit.get("learning_receipt_path", "")).strip()
    audited_path = Path(audited_raw_path)
    audited_receipt_path = audited_path if audited_path.is_absolute() else root / audited_path
    return bool(
        audit.get("status") == expected_audit_status
        and audit.get("evidence_valid") is True
        and audit.get("registered_execution_id") == execution_id
        and bool(audit.get("stage5_research_gate_pass", False)) == gate_pass
        and audited_raw_path
        and audited_receipt_path.resolve() == receipt_path.resolve()
        and audit.get("learning_receipt_sha256") == expected_hash
        and all(audit.get(field) is False for field in authority_fields)
    )


def _daily_attempt_state(
    *, latest_path: Path, receipts: Path, attempt_date: str
) -> tuple[str, list[str]]:
    evidence: list[str] = []
    malformed = False
    attempted = False
    candidates = [latest_path]
    if receipts.is_dir():
        candidates.extend(sorted(receipts.glob(f"{attempt_date}_*.json")))
    for path in candidates:
        if not path.is_file():
            continue
        payload, valid_json = _try_read_json(path)
        if not valid_json:
            malformed = True
            evidence.append(str(path))
            continue
        if str(payload.get("attempt_date_utc", "")) != attempt_date:
            continue
        evidence.append(str(path))
        if payload.get("external_attempt_made") is True:
            attempted = True
    if attempted:
        return "ATTEMPTED", evidence
    if malformed:
        return "UNVERIFIABLE", evidence
    return "CLEAR", evidence


def _try_read_json(path: Path) -> tuple[dict[str, Any], bool]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, False
    return (payload, True) if isinstance(payload, dict) else ({}, False)


def _read_json(path: Path) -> dict[str, Any]:
    payload, valid = _try_read_json(path)
    return payload if valid else {}


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        publication_scope="wizard_proof_launcher",
    )


def _write_or_validate_immutable_json(payload: dict[str, Any], path: Path) -> None:
    try:
        write_immutable_json(
            path,
            payload,
            publication_scope="wizard_proof_launcher",
        )
    except ValueError as exc:
        if not str(exc).startswith("immutable artifact collision:"):
            raise
        raise ValueError(f"immutable launcher receipt collision: {path}") from exc


def _publish_latest_launcher_status(payload: dict[str, Any], path: Path) -> bool:
    """Publish the newest operational heartbeat without letting dry runs replace it."""

    with publication_lease_for_path(path, scope="wizard_proof_launcher"):
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with lock_path.open("a", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            current = _read_json(path)
            candidate_execute = payload.get("execute_requested") is True
            current_execute = current.get("execute_requested") is True
            if current_execute and not candidate_execute:
                return False
            if candidate_execute and not current_execute:
                _atomic_json(payload, path)
                return True
            candidate_completed_at = _launcher_completion_time(payload)
            current_completed_at = _launcher_completion_time(current)
            if (
                candidate_completed_at is not None
                and current_completed_at is not None
                and candidate_completed_at < current_completed_at
            ):
                return False
            _atomic_json(payload, path)
            return True


def _launcher_completion_time(payload: dict[str, Any]) -> datetime | None:
    return _try_parse_utc(
        str(payload.get("completed_at_utc") or payload.get("checked_at_utc") or "")
    )


def validate_launcher_receipt_binding(*, root: Path, status: dict[str, Any]) -> dict[str, Any]:
    receipt_id = str(status.get("launcher_receipt_id", "")).strip()
    attempt_date = str(status.get("attempt_date_utc", "")).strip()
    relative = str(status.get("immutable_launcher_receipt_path", "")).strip()
    expected_sha256 = str(status.get("immutable_launcher_receipt_sha256", "")).strip().lower()
    expected_relative = (
        f"{LAUNCHER_RECEIPT_DIR}/{attempt_date}/{receipt_id}.json"
        if attempt_date and receipt_id
        else ""
    )
    path = root / relative
    receipt_root = (root / LAUNCHER_RECEIPT_DIR).resolve()
    blockers: list[str] = []
    try:
        path_safe = bool(
            relative == expected_relative
            and not Path(relative).is_absolute()
            and path.is_file()
            and not path.is_symlink()
            and path.resolve().is_relative_to(receipt_root)
        )
    except (OSError, ValueError):
        path_safe = False
    if not path_safe:
        blockers.append("launcher_immutable_receipt_path_invalid")
    actual_sha256 = sha256(path.read_bytes()).hexdigest() if path_safe else ""
    if len(expected_sha256) != 64 or actual_sha256 != expected_sha256:
        blockers.append("launcher_immutable_receipt_hash_invalid")
    receipt = _read_json(path) if path_safe else {}
    stable = dict(receipt)
    observed_id = str(stable.pop("launcher_receipt_id", "")).strip()
    calculated_id = (
        "wizardlauncher_" + sha256(_canonical_json(stable).encode("utf-8")).hexdigest()[:20]
        if stable
        else ""
    )
    if (
        not receipt
        or receipt.get("schema_version") != SCHEMA_VERSION
        or observed_id != receipt_id
        or calculated_id != receipt_id
        or receipt.get("attempt_date_utc") != attempt_date
        or any(status.get(key) != value for key, value in receipt.items())
        or any(
            receipt.get(field) is not False
            for field in (
                "candidate_promotion_authority",
                "order_submission_included",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
    ):
        blockers.append("launcher_immutable_receipt_content_invalid")
    blockers = sorted(set(blockers))
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "receipt_id": receipt_id,
        "receipt_path": relative,
        "receipt_sha256": actual_sha256,
    }


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("scheduler receipt timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _try_parse_utc(value: str) -> datetime | None:
    try:
        return _parse_utc(value)
    except (TypeError, ValueError):
        return None


def _launcher_exit_code(result: dict[str, Any]) -> int:
    status = str(result.get("launcher_status", ""))
    child_code = result.get("heavy_scheduler_return_code")
    if status == "HEAVY_SCHEDULER_FAILED":
        try:
            return max(int(child_code or 1), 1)
        except (TypeError, ValueError):
            return 1
    if status in {
        "BLOCKED_UNVERIFIABLE_SAME_DAY_EVIDENCE",
        "BLOCKED_SCHEDULER_PYTHON_MISSING",
        "BLOCKED_SCHEDULER_RECEIPT_UNVERIFIED",
        "BLOCKED_UNATTENDED_EXTERNAL_PREFLIGHT",
        "HEAVY_SCHEDULER_BLOCKED",
        "HEAVY_SCHEDULER_RECONCILED_INCOMPLETE",
        "INTERNAL_REGISTERED_CONTINUATION_INCOMPLETE",
        "INTERNAL_STAGE5_RECOVERY_INCOMPLETE",
        "INTERNAL_DYNAMIC_V2_REFRESH_INCOMPLETE",
    }:
        return 2
    return 0


def _launcher_supervisor_result(
    *,
    execute: bool,
    force: bool,
    unattended_preflight_builder: Callable[..., Any] | None,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    launcher = run_wizard_proof_launcher(
        execute=execute,
        force=force,
        runner=runner,
        unattended_preflight_builder=unattended_preflight_builder,
    )
    exit_code = _launcher_exit_code(launcher)
    blockers = [str(value) for value in launcher.get("blockers", []) if str(value)]
    if exit_code and not blockers:
        blockers = [
            f"wizard_proof_launcher_status:{launcher.get('launcher_status', 'UNKNOWN')}"
        ]
    return {
        "status": "PASS" if exit_code == 0 else "BLOCKED",
        "blockers": blockers,
        "launcher_result": launcher,
        "external_calls": 0,
        "external_credits_reserved": 0,
        "external_credits_consumed": 0,
        "external_credits_reconciled": 0,
        "order_attempts": 0,
        "order_submissions": 0,
        "authority_advanced": False,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _run_heavy_scheduler_in_process(
    command: list[str],
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Preserve supervisor authority across the heavy scheduler boundary."""

    root = Path(kwargs.get("cwd", ROOT)).resolve()
    expected_prefix = [str(scheduler_python_path(root)), "-m", HEAVY_MODULE]
    flags = command[len(expected_prefix) :]
    allowed_flags = {"--execute", "--force", "--internal-continuation-only"}
    if command[: len(expected_prefix)] != expected_prefix:
        raise ValueError("wizard_heavy_in_process_command_identity_mismatch")
    if len(flags) != len(set(flags)) or any(flag not in allowed_flags for flag in flags):
        raise ValueError("wizard_heavy_in_process_flags_invalid")
    if current_external_effect_issuer() is None:
        raise RuntimeError("wizard_heavy_in_process_effect_issuer_missing")
    from quant_platform.orchestration.corrective_wizard_proof_scheduler import (
        run_corrective_wizard_proof_cycle,
    )

    run_corrective_wizard_proof_cycle(
        root=root,
        execute="--execute" in flags,
        force="--force" in flags,
        internal_continuation_only="--internal-continuation-only" in flags,
    )
    return subprocess.CompletedProcess(command, 0)


def _authorized_unattended_preflight(*, root: Path, now: datetime) -> Any:
    """Authorize one free credit-status request before the paid lane opens."""

    issuer = current_external_effect_issuer()
    if issuer is None:
        raise RuntimeError("wizard_preflight_effect_issuer_missing")
    material = {
        "run_id": issuer.run_id,
        "intended_slot_id": issuer.intended_slot_id,
        "purpose": "wizard_unattended_credit_preflight",
        "max_total_requests": 1,
        "max_total_credits": 0,
    }
    reservation_sha256 = sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with reserved_external_effect_session(
        reservation_id=f"wizard-preflight:{issuer.run_id}",
        reservation_sha256=reservation_sha256,
        max_total_requests=1,
        max_total_credits=0,
    ):
        api_key = read_authorized_credential("CRYPTO_WIZARDS_API_KEY")
        from quant_platform.orchestration.corrective_wizard_unattended_preflight import (
            build_wizard_unattended_external_preflight,
        )

        return build_wizard_unattended_external_preflight(
            root=root,
            now=now,
            api_key=api_key,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    supervised = supervise_scheduler_run(
        root=ROOT,
        contract_key="wizard_proof",
        publication_scope="wizard_external_research",
        callback=lambda: _launcher_supervisor_result(
            execute=args.execute,
            force=args.force,
            runner=_run_heavy_scheduler_in_process,
            unattended_preflight_builder=(
                _authorized_unattended_preflight if args.execute else None
            ),
        ),
        require_launchd_provenance=True,
    )
    print(
        json.dumps(
            {
                "summary": supervised.result_summary,
                "terminal_receipt": supervised.terminal_receipt,
                "terminal_paths": {
                    key: str(value) for key, value in supervised.terminal_paths.items()
                },
            },
            indent=2,
        )
    )
    if supervised.exit_code:
        raise SystemExit(supervised.exit_code)


if __name__ == "__main__":
    main()
