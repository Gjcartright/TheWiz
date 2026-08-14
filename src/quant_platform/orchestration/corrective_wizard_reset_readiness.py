"""Audit the zero-authority Crypto Wizards post-reset automation contract."""

from __future__ import annotations

import csv
import json
import os
import plistlib
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_runtime import (
    RUNTIME_TEMP_DIRNAME,
    RUNTIME_TEMP_ENV_NAMES,
    workspace_launch_agent_path,
)
from quant_platform.orchestration.corrective_wizard_browser_auth import (
    validate_wizard_browser_auth_readiness,
)
from quant_platform.orchestration.corrective_wizard_capture_manifest import (
    classify_capture_manifest_source_drift,
    validate_capture_manifest_source_receipt,
)
from quant_platform.orchestration.corrective_wizard_ou_v5_holdout import (
    audit_ou_v5_capture_readiness,
)
from quant_platform.orchestration.corrective_wizard_ou_v6_holdout import (
    audit_ou_v6_capture_readiness,
)
from quant_platform.orchestration.corrective_wizard_proof_launcher import (
    validate_launcher_receipt_binding,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_wizard_reset_readiness.v5"
RESET_STATE_VOLATILE_FIELDS = frozenset(
    {
        "checked_at_utc",
        "launcher_heartbeat_age_seconds",
        "launcher_receipt_id",
        "launcher_receipt_path",
        "launcher_receipt_sha256",
        "receipt_id",
    }
)
LAUNCH_AGENT_LABEL = "com.thewiz.corrective-wizard-proof"
LAUNCHER_MODULE = "quant_platform.orchestration.corrective_wizard_proof_launcher"
MAX_INTERVAL_SECONDS = 600
HEARTBEAT_GRACE_MULTIPLIER = 2
API_KEY_ENV = "CRYPTO_WIZARDS_API_KEY"
MIN_RUNTIME_TEMP_FREE_BYTES = 1024**3


def build_corrective_wizard_reset_readiness(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    launch_agent_path: Path | None = None,
    launch_agent_loaded: bool | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    minimum_runtime_temp_free_bytes: int = MIN_RUNTIME_TEMP_FREE_BYTES,
) -> CommandResult:
    """Prove that the frozen Stage 3 capture can start safely after reset.

    This audit is deliberately read-only with respect to external systems. It
    does not call Crypto Wizards, reserve credits, promote candidates, or submit
    orders.
    """

    checked_at = _as_utc(now)
    active = root / "reports" / "active"
    manifest_path = active / "corrective_wizard_next_capture_manifest.json"
    scheduler_observation_path = (
        active / "corrective_wizard_proof_scheduler_observation_status.json"
    )
    scheduler_execution_path = active / "corrective_wizard_proof_scheduler_execution_status.json"
    scheduler_legacy_path = active / "corrective_wizard_proof_scheduler_status.json"
    scheduler_path = (
        scheduler_observation_path
        if scheduler_observation_path.is_file()
        else scheduler_legacy_path
    )
    launcher_path = active / "corrective_wizard_proof_launcher_status.json"
    budget_path = active / "wizard_credit_budget_contract.json"
    launch_agent_path = launch_agent_path or workspace_launch_agent_path(root, LAUNCH_AGENT_LABEL)

    manifest = _read_json(manifest_path)
    scheduler = _read_json(scheduler_path)
    execution_scheduler = _read_json(scheduler_execution_path)
    role_split_enforced = bool(
        scheduler_observation_path.is_file() and scheduler_execution_path.is_file()
    )
    launcher = _read_json(launcher_path)
    budget = _read_json(budget_path)
    checks: list[dict[str, str]] = []

    plist_payload: dict[str, Any] = {}
    plist_blocker = ""
    if not launch_agent_path.is_file():
        plist_blocker = "wizard_proof_launch_agent_missing"
    else:
        try:
            with launch_agent_path.open("rb") as handle:
                candidate = plistlib.load(handle)
            if isinstance(candidate, dict):
                plist_payload = candidate
            else:
                plist_blocker = "wizard_proof_launch_agent_not_dictionary"
        except (OSError, plistlib.InvalidFileException) as exc:
            plist_blocker = f"wizard_proof_launch_agent_invalid:{type(exc).__name__}"
    _add_check(
        checks,
        name="launch_agent_plist_parse",
        passed=bool(plist_payload) and not plist_blocker,
        observed="valid" if plist_payload else "invalid_or_missing",
        evidence_path=str(launch_agent_path),
        blocker=plist_blocker or "wizard_proof_launch_agent_invalid",
    )

    expected_python = root / ".venv312" / "bin" / "python"
    expected_arguments = [
        str(expected_python),
        "-m",
        LAUNCHER_MODULE,
        "--execute",
    ]
    arguments = [str(value) for value in plist_payload.get("ProgramArguments", [])]
    _add_check(
        checks,
        name="launch_agent_command_contract",
        passed=arguments == expected_arguments and expected_python.is_file(),
        observed="exact_execute_command" if arguments == expected_arguments else "mismatch",
        evidence_path=str(launch_agent_path),
        blocker="wizard_proof_launch_agent_command_mismatch",
    )
    interval = _safe_int(plist_payload.get("StartInterval"), default=-1)
    _add_check(
        checks,
        name="launch_agent_interval",
        passed=60 <= interval <= MAX_INTERVAL_SECONDS,
        observed=str(interval),
        evidence_path=str(launch_agent_path),
        blocker="wizard_proof_launch_agent_interval_out_of_bounds",
    )
    environment = plist_payload.get("EnvironmentVariables", {})
    expected_source = str(root / "src")
    _add_check(
        checks,
        name="launch_agent_repository_binding",
        passed=bool(
            plist_payload.get("Label") == LAUNCH_AGENT_LABEL
            and plist_payload.get("WorkingDirectory") == str(root)
            and isinstance(environment, dict)
            and environment.get("PYTHONPATH") == expected_source
        ),
        observed="bound_to_current_repository",
        evidence_path=str(launch_agent_path),
        blocker="wizard_proof_launch_agent_repository_binding_mismatch",
    )
    expected_runtime_temp = root / RUNTIME_TEMP_DIRNAME
    temp_binding_valid = bool(
        isinstance(environment, dict)
        and all(
            environment.get(name) == str(expected_runtime_temp) for name in RUNTIME_TEMP_ENV_NAMES
        )
    )
    _add_check(
        checks,
        name="launch_agent_workspace_temp_binding",
        passed=temp_binding_valid,
        observed=(
            "bound_to_workspace_temp"
            if temp_binding_valid
            else ";".join(f"{name}={environment.get(name, '')}" for name in RUNTIME_TEMP_ENV_NAMES)
        ),
        evidence_path=str(launch_agent_path),
        blocker="wizard_proof_launch_agent_temp_binding_mismatch",
    )
    temp_viable, temp_observed = _runtime_temp_viability(
        expected_runtime_temp,
        minimum_free_bytes=minimum_runtime_temp_free_bytes,
    )
    _add_check(
        checks,
        name="workspace_runtime_temp_viability",
        passed=temp_viable,
        observed=temp_observed,
        evidence_path=str(expected_runtime_temp),
        blocker="wizard_proof_workspace_temp_not_viable",
    )

    if launch_agent_loaded is None:
        launch_agent_loaded = _launch_agent_is_loaded(runner)
    _add_check(
        checks,
        name="launch_agent_loaded",
        passed=bool(launch_agent_loaded),
        observed="loaded" if launch_agent_loaded else "not_loaded",
        evidence_path=f"launchctl:gui/{os.getuid()}/{LAUNCH_AGENT_LABEL}",
        blocker="wizard_proof_launch_agent_not_loaded",
    )

    manifest_id = str(manifest.get("manifest_id", ""))
    immutable_relative = str(manifest.get("immutable_manifest_path", ""))
    immutable_path = root / immutable_relative
    expected_digest = str(manifest.get("immutable_manifest_sha256", "")).lower()
    actual_digest = _file_sha256(immutable_path) if immutable_path.is_file() else ""
    manifest_binding_valid = bool(
        manifest.get("schema_version") == "thewiz.corrective_wizard_next_capture_manifest.v1"
        and manifest.get("status") == "PASS"
        and manifest.get("manifest_enforced") is True
        and manifest_id.startswith("wizardcapture_")
        and immutable_relative.startswith("data/research/wizard_capture_manifests/")
        and immutable_relative.endswith(f"/{manifest_id}.json")
        and len(expected_digest) == 64
        and actual_digest == expected_digest
    )
    _add_check(
        checks,
        name="immutable_capture_manifest_binding",
        passed=manifest_binding_valid,
        observed=manifest_id or "missing",
        evidence_path=immutable_relative or str(manifest_path),
        blocker="frozen_capture_manifest_binding_invalid",
    )

    (
        source_lineage_valid,
        source_artifact_count,
        source_artifacts_sha256,
        source_lineage_observed,
    ) = _manifest_source_lineage(root=root, manifest=manifest)
    _add_check(
        checks,
        name="capture_manifest_source_lineage",
        passed=source_lineage_valid,
        observed=source_lineage_observed,
        evidence_path=_relative(manifest_path, root),
        blocker="frozen_capture_manifest_source_lineage_invalid",
    )
    source_drift_valid, source_drift = _manifest_source_drift(
        root=root,
        manifest=manifest,
        source_lineage_valid=source_lineage_valid,
    )
    _add_check(
        checks,
        name="capture_source_current_drift_classification",
        passed=source_drift_valid,
        observed=(
            f"{source_drift['classification']};"
            f"metadata={len(source_drift['metadata_only_paths'])};"
            f"scientific={len(source_drift['scientific_or_structural_paths'])}"
        ),
        evidence_path=_relative(manifest_path, root),
        blocker="wizard_capture_source_scientific_or_structural_drift",
    )

    pending_calls = _safe_int(manifest.get("pending_calls"))
    planned_credits = _safe_int(manifest.get("planned_credits"))
    proof_ceiling = _safe_int(manifest.get("proof_lane_credit_ceiling"))
    lane_totals = manifest.get("lane_totals", {})
    lane_calls, lane_credits = _lane_accounting(lane_totals)
    accounting_valid = bool(
        pending_calls > 0
        and planned_credits > 0
        and planned_credits <= proof_ceiling
        and lane_calls == pending_calls
        and lane_credits == planned_credits
        and manifest.get("runtime_credit_preflight_required") is True
    )
    _add_check(
        checks,
        name="capture_manifest_accounting",
        passed=accounting_valid,
        observed=f"calls={pending_calls};credits={planned_credits};ceiling={proof_ceiling}",
        evidence_path=_relative(manifest_path, root),
        blocker="frozen_capture_manifest_accounting_invalid",
    )

    ou_v5_lane = lane_totals.get("ou_v5_holdout", {})
    ou_v5_calls = _safe_int(ou_v5_lane.get("calls")) if isinstance(ou_v5_lane, dict) else 0
    ou_v5_capture_audit = (
        audit_ou_v5_capture_readiness(root=root, now=checked_at)
        if ou_v5_calls > 0
        else {
            "status": "NOT_APPLICABLE",
            "blockers": [],
            "required_cells": 0,
            "responses_available": 0,
            "missing_cells": 0,
            "unresolved_call_intents": 0,
        }
    )
    ou_v5_capture_ready = bool(
        ou_v5_calls == 0
        or (
            ou_v5_capture_audit.get("status") == "PASS"
            and _safe_int(ou_v5_capture_audit.get("required_cells")) == ou_v5_calls
        )
    )
    _add_check(
        checks,
        name="ou_v5_cross_day_call_replay_safety",
        passed=ou_v5_capture_ready,
        observed=(
            f"lane_calls={ou_v5_calls};"
            f"missing={_safe_int(ou_v5_capture_audit.get('missing_cells'))};"
            "unresolved_intents="
            f"{_safe_int(ou_v5_capture_audit.get('unresolved_call_intents'))}"
        ),
        evidence_path="reports/active/wizard_ou_v5_holdout_receipt.json",
        blocker=(
            str(ou_v5_capture_audit.get("blockers", [""])[0])
            if ou_v5_capture_audit.get("blockers")
            else "wizard_ou_v5_capture_readiness_invalid"
        ),
    )

    ou_v6_lane = lane_totals.get("ou_v6_holdout", {})
    ou_v6_calls = _safe_int(ou_v6_lane.get("calls")) if isinstance(ou_v6_lane, dict) else 0
    ou_v6_capture_audit = (
        audit_ou_v6_capture_readiness(root=root, now=checked_at)
        if ou_v6_calls > 0
        else {
            "status": "NOT_APPLICABLE",
            "blockers": [],
            "required_cells": 0,
            "responses_available": 0,
            "missing_cells": 0,
            "unresolved_call_intents": 0,
        }
    )
    ou_v6_capture_ready = bool(
        ou_v6_calls == 0
        or (
            ou_v6_capture_audit.get("status") == "PASS"
            and _safe_int(ou_v6_capture_audit.get("required_cells")) == ou_v6_calls
        )
    )
    _add_check(
        checks,
        name="ou_v6_cross_day_call_replay_safety",
        passed=ou_v6_capture_ready,
        observed=(
            f"lane_calls={ou_v6_calls};"
            f"missing={_safe_int(ou_v6_capture_audit.get('missing_cells'))};"
            "unresolved_intents="
            f"{_safe_int(ou_v6_capture_audit.get('unresolved_call_intents'))}"
        ),
        evidence_path="reports/active/wizard_ou_v6_holdout_receipt.json",
        blocker=(
            str(ou_v6_capture_audit.get("blockers", [""])[0])
            if ou_v6_capture_audit.get("blockers")
            else "wizard_ou_v6_capture_readiness_invalid"
        ),
    )

    scheduler_continuity_valid = bool(
        scheduler.get("schema_version") == "thewiz.corrective_wizard_proof_scheduler.v1"
        and (
            not role_split_enforced
            or (
                scheduler.get("execution_requested") is False
                and scheduler.get("pointer_role") == "observation"
            )
        )
        and scheduler.get("capture_manifest_enforced") is True
        and scheduler.get("capture_manifest_id") == manifest_id
        and scheduler.get("capture_manifest_immutable_path") == immutable_relative
        and scheduler.get("capture_manifest_immutable_sha256") == expected_digest
        and scheduler.get("capture_manifest_candidate_id") == manifest_id
        and scheduler.get("capture_manifest_candidate_immutable_path") == immutable_relative
        and scheduler.get("capture_manifest_candidate_immutable_sha256") == expected_digest
        and scheduler.get("capture_manifest_candidate_binding_valid") is True
        and scheduler.get("capture_manifest_candidate_source_binding_valid") is True
        and scheduler.get("capture_manifest_continuity_status")
        in {"PASS_NEW_FROZEN_COHORT", "PASS_PRIOR_UNRESOLVED_COHORT_MATCH"}
        and scheduler.get("capture_manifest_continuity_valid") is True
        and scheduler.get("capture_manifest_drift_detected") is False
    )
    _add_check(
        checks,
        name="scheduler_capture_manifest_continuity",
        passed=scheduler_continuity_valid,
        observed=(str(scheduler.get("capture_manifest_continuity_status", "missing"))),
        evidence_path=_relative(scheduler_path, root),
        blocker="wizard_scheduler_capture_manifest_continuity_invalid",
    )

    reset_at = _try_parse_utc(str(manifest.get("next_external_attempt_eligible_at", "")))
    reset_contract_valid = bool(
        reset_at
        and reset_at.minute == 0
        and reset_at.second == 0
        and reset_at.microsecond == 0
        and reset_at.hour == 0
        and checked_at - timedelta(seconds=max(interval, 0)) <= reset_at
        and reset_at <= checked_at + timedelta(hours=24)
    )
    capture_window_status = (
        "WAITING_FOR_RESET" if reset_at and checked_at < reset_at else "DUE_AFTER_RESET"
    )
    _add_check(
        checks,
        name="utc_reset_contract",
        passed=reset_contract_valid,
        observed=reset_at.isoformat() if reset_at else "invalid",
        evidence_path=_relative(manifest_path, root),
        blocker="wizard_capture_reset_timestamp_invalid",
    )

    execution_scheduler_binding_valid = bool(
        not role_split_enforced
        or _scheduler_execution_binding_valid(
            root=root,
            pointer=execution_scheduler,
            reset_at=reset_at,
        )
    )
    _add_check(
        checks,
        name="scheduler_execution_reset_binding",
        passed=execution_scheduler_binding_valid,
        observed=(
            str(execution_scheduler.get("receipt_id", "missing"))
            if role_split_enforced
            else "legacy_single_pointer_compatibility"
        ),
        evidence_path=(
            _relative(scheduler_execution_path, root)
            if role_split_enforced
            else _relative(scheduler_path, root)
        ),
        blocker="wizard_scheduler_execution_reset_binding_invalid",
    )

    browser_auth_path = active / "wizard_browser_auth_readiness.json"
    browser_auth = _read_json(browser_auth_path)
    browser_auth_required_at = (
        reset_at if reset_at is not None and checked_at < reset_at else checked_at
    )
    browser_auth_current = validate_wizard_browser_auth_readiness(
        root=root,
        now=checked_at,
        status_path=browser_auth_path,
    )
    browser_auth_at_window = validate_wizard_browser_auth_readiness(
        root=root,
        now=browser_auth_required_at,
        status_path=browser_auth_path,
    )
    browser_auth_valid_until = _try_parse_utc(str(browser_auth.get("valid_until_utc", "")))
    browser_auth_window_ready = bool(
        browser_auth_current.get("status") == "PASS"
        and browser_auth_at_window.get("status") == "PASS"
        and browser_auth_current.get("receipt_id") == browser_auth_at_window.get("receipt_id")
        and browser_auth_valid_until is not None
        and browser_auth_valid_until >= browser_auth_required_at
    )
    _add_check(
        checks,
        name="browser_auth_capture_window_coverage",
        passed=browser_auth_window_ready,
        observed=(
            f"required_at={browser_auth_required_at.isoformat()};"
            f"valid_until={browser_auth_valid_until.isoformat() if browser_auth_valid_until else 'invalid'};"
            f"current={browser_auth_current.get('status', 'BLOCKED')};"
            f"window={browser_auth_at_window.get('status', 'BLOCKED')}"
        ),
        evidence_path=str(
            browser_auth_current.get("receipt_path", _relative(browser_auth_path, root))
        ),
        blocker="wizard_browser_auth_not_valid_at_capture_window",
    )

    launcher_checked_at = _try_parse_utc(str(launcher.get("checked_at_utc", "")))
    heartbeat_age_seconds = (
        max(0.0, (checked_at - launcher_checked_at).total_seconds())
        if launcher_checked_at
        else float("inf")
    )
    heartbeat_limit = max(interval, MAX_INTERVAL_SECONDS) * HEARTBEAT_GRACE_MULTIPLIER
    launcher_authority_safe = _authority_is_zero(launcher)
    launcher_reset_matches = bool(
        reset_at
        and str(launcher.get("next_external_attempt_eligible_at", "")) == reset_at.isoformat()
    )
    heartbeat_valid = bool(
        launcher_checked_at
        and launcher_checked_at <= checked_at + timedelta(seconds=5)
        and heartbeat_age_seconds <= heartbeat_limit
        and launcher.get("execute_requested") is True
        and launcher_authority_safe
        and launcher_reset_matches
    )
    _add_check(
        checks,
        name="launcher_heartbeat_and_reset_binding",
        passed=heartbeat_valid,
        observed=f"age_seconds={heartbeat_age_seconds:.1f}",
        evidence_path=_relative(launcher_path, root),
        blocker="wizard_proof_launcher_heartbeat_or_reset_binding_invalid",
    )
    launcher_receipt_binding = validate_launcher_receipt_binding(
        root=root,
        status=launcher,
    )
    _add_check(
        checks,
        name="launcher_immutable_receipt_binding",
        passed=launcher_receipt_binding["status"] == "PASS",
        observed=str(launcher_receipt_binding.get("receipt_id", "missing")),
        evidence_path=str(launcher_receipt_binding.get("receipt_path", launcher_path)),
        blocker="wizard_proof_launcher_immutable_receipt_invalid",
    )

    credential = _credential_configuration(root)
    _add_check(
        checks,
        name="wizard_api_credential_configuration",
        passed=bool(credential["configured"] and credential["secure"]),
        observed=str(credential["source"]),
        evidence_path=str(credential["evidence_path"]),
        blocker=str(credential["blocker"]),
    )

    budget_ceiling = sum(
        _safe_int(budget.get(field))
        for field in (
            "exact_mode_proof_credit_ceiling",
            "copula_behavioral_credit_ceiling",
            "ou_v3_prospective_credit_ceiling",
            "ou_v4_prospective_credit_ceiling",
            "ou_v5_prospective_credit_ceiling",
            "ou_v6_prospective_credit_ceiling",
        )
    )
    budget_valid = bool(
        budget.get("status") == "PASS"
        and budget.get("runtime_credit_preflight_still_required") is True
        and _authority_is_zero(budget)
        and planned_credits <= budget_ceiling
        and _safe_int(budget.get("headroom_after_reserve")) >= planned_credits
        and str(budget.get("wizard_daily_credit_reset_utc", "")) == "00:00"
    )
    _add_check(
        checks,
        name="shared_credit_budget_contract",
        passed=budget_valid,
        observed=f"planned={planned_credits};lane_ceiling={budget_ceiling}",
        evidence_path=_relative(budget_path, root),
        blocker="wizard_shared_credit_budget_not_ready",
    )

    authority_safe = (
        _authority_is_zero(manifest)
        and _authority_is_zero(budget)
        and _authority_is_zero(browser_auth)
    )
    _add_check(
        checks,
        name="zero_trading_authority",
        passed=authority_safe,
        observed="research_capture_only" if authority_safe else "authority_violation",
        evidence_path=_relative(manifest_path, root),
        blocker="wizard_reset_readiness_authority_violation",
    )

    blockers = [row["blocker"] for row in checks if row["status"] == "BLOCKED"]
    status = "PASS_RESET_AUTOMATION_READY" if not blockers else "BLOCKED_RESET_AUTOMATION"
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checked_at_utc": checked_at.isoformat(),
        "status": status,
        "capture_window_status": capture_window_status,
        "manifest_id": manifest_id,
        "immutable_manifest_path": immutable_relative,
        "immutable_manifest_sha256": expected_digest,
        "manifest_source_lineage_valid": source_lineage_valid,
        "manifest_source_artifact_count": source_artifact_count,
        "manifest_source_artifacts_sha256": source_artifacts_sha256,
        "manifest_source_current_drift_valid": source_drift_valid,
        "manifest_source_current_drift_classification": source_drift["classification"],
        "manifest_source_current_metadata_only_paths": source_drift["metadata_only_paths"],
        "manifest_source_current_scientific_or_structural_paths": source_drift[
            "scientific_or_structural_paths"
        ],
        "scheduler_manifest_continuity_valid": scheduler_continuity_valid,
        "scheduler_manifest_continuity_status": str(
            scheduler.get("capture_manifest_continuity_status", "")
        ),
        "scheduler_role_split_enforced": role_split_enforced,
        "scheduler_observation_pointer_path": _relative(scheduler_path, root),
        "scheduler_execution_pointer_path": (
            _relative(scheduler_execution_path, root) if role_split_enforced else ""
        ),
        "scheduler_execution_reset_binding_valid": execution_scheduler_binding_valid,
        "scheduler_execution_receipt_id": str(execution_scheduler.get("receipt_id", "")),
        "pending_calls": pending_calls,
        "planned_credits": planned_credits,
        "ou_v5_capture_readiness_status": str(ou_v5_capture_audit.get("status", "")),
        "ou_v5_capture_readiness_blockers": list(ou_v5_capture_audit.get("blockers", [])),
        "ou_v5_unresolved_call_intents": _safe_int(
            ou_v5_capture_audit.get("unresolved_call_intents")
        ),
        "ou_v6_capture_readiness_status": str(ou_v6_capture_audit.get("status", "")),
        "ou_v6_capture_readiness_blockers": list(ou_v6_capture_audit.get("blockers", [])),
        "ou_v6_unresolved_call_intents": _safe_int(
            ou_v6_capture_audit.get("unresolved_call_intents")
        ),
        "next_external_attempt_eligible_at": reset_at.isoformat() if reset_at else "",
        "browser_auth_required_at_utc": browser_auth_required_at.isoformat(),
        "browser_auth_valid_at_capture_window": browser_auth_window_ready,
        "browser_auth_valid_until_utc": (
            browser_auth_valid_until.isoformat() if browser_auth_valid_until else ""
        ),
        "browser_auth_receipt_id": str(browser_auth_current.get("receipt_id", "")),
        "browser_auth_receipt_path": str(browser_auth_current.get("receipt_path", "")),
        "browser_auth_receipt_sha256": str(browser_auth_current.get("receipt_sha256", "")),
        "launch_agent_label": LAUNCH_AGENT_LABEL,
        "launch_agent_path": str(launch_agent_path),
        "launch_agent_loaded": bool(launch_agent_loaded),
        "launch_agent_interval_seconds": interval,
        "runtime_temp_path": str(expected_runtime_temp),
        "runtime_temp_minimum_free_bytes": minimum_runtime_temp_free_bytes,
        "runtime_temp_ready": temp_binding_valid and temp_viable,
        "launcher_heartbeat_age_seconds": (
            round(heartbeat_age_seconds, 3) if heartbeat_age_seconds != float("inf") else None
        ),
        "launcher_receipt_binding_valid": (launcher_receipt_binding["status"] == "PASS"),
        "launcher_receipt_id": str(launcher_receipt_binding.get("receipt_id", "")),
        "launcher_receipt_path": str(launcher_receipt_binding.get("receipt_path", "")),
        "launcher_receipt_sha256": str(launcher_receipt_binding.get("receipt_sha256", "")),
        "credential_configured": bool(credential["configured"]),
        "credential_source": str(credential["source"]),
        "runtime_credit_preflight_required": True,
        "checks_total": len(checks),
        "checks_passed": sum(row["status"] == "PASS" for row in checks),
        "blockers": blockers,
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    receipt_material = dict(payload)
    receipt_id = (
        "wizardresetreadiness_"
        + sha256(_canonical_json(receipt_material).encode("utf-8")).hexdigest()[:20]
    )
    payload["receipt_id"] = receipt_id

    active.mkdir(parents=True, exist_ok=True)
    immutable_dir = root / "data" / "research" / "wizard_reset_readiness"
    immutable_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "checks": active / "wizard_reset_readiness_checks.csv",
        "status": active / "wizard_reset_readiness.json",
        "summary": active / "wizard_reset_readiness.md",
        "immutable_receipt": immutable_dir / f"{receipt_id}.json",
    }
    _write_csv(checks, paths["checks"])
    _write_json(payload, paths["status"])
    _write_text(paths["summary"], _markdown(payload, checks))
    _write_immutable_json(payload, paths["immutable_receipt"])
    payload["evidence_paths"] = {key: _relative(path, root) for key, path in paths.items()}
    return CommandResult(paths=paths, summary=payload)


def wizard_reset_readiness_state_sha256(receipt: dict[str, Any]) -> str:
    """Hash safety-relevant reset state without observation-time churn."""

    material = {
        key: value
        for key, value in receipt.items()
        if key not in RESET_STATE_VOLATILE_FIELDS and key != "evidence_paths"
    }
    return sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def validate_wizard_reset_readiness_receipt(
    *, root: Path = ROOT, receipt: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate a reset receipt, its immutable twin, and zero authority."""

    payload = (
        receipt
        if receipt is not None
        else _read_json(root / "reports" / "active" / "wizard_reset_readiness.json")
    )
    blockers: list[str] = []
    receipt_id = str(payload.get("receipt_id", "")).strip()
    material = {key: value for key, value in payload.items() if key != "receipt_id"}
    expected_id = (
        "wizardresetreadiness_" + sha256(_canonical_json(material).encode("utf-8")).hexdigest()[:20]
    )
    immutable_relative = f"data/research/wizard_reset_readiness/{receipt_id}.json"
    immutable_path = root / immutable_relative
    immutable = _read_json(immutable_path)
    reported_blockers = payload.get("blockers")
    checks_total = _safe_int(payload.get("checks_total"), default=-1)
    checks_passed = _safe_int(payload.get("checks_passed"), default=-1)

    if payload.get("schema_version") != SCHEMA_VERSION:
        blockers.append("wizard_reset_readiness_schema_version_invalid")
    if receipt_id != expected_id:
        blockers.append("wizard_reset_readiness_receipt_id_invalid")
    if not immutable_path.is_file():
        blockers.append("wizard_reset_readiness_immutable_receipt_missing")
    elif immutable != payload:
        blockers.append("wizard_reset_readiness_immutable_receipt_mismatch")
    if payload.get("status") not in {
        "PASS_RESET_AUTOMATION_READY",
        "BLOCKED_RESET_AUTOMATION",
    }:
        blockers.append("wizard_reset_readiness_status_invalid")
    if not isinstance(reported_blockers, list):
        blockers.append("wizard_reset_readiness_blockers_invalid")
        reported_blockers = []
    if checks_total <= 0 or checks_passed < 0 or checks_passed > checks_total:
        blockers.append("wizard_reset_readiness_check_accounting_invalid")
    if payload.get("status") == "PASS_RESET_AUTOMATION_READY" and (
        checks_passed != checks_total or reported_blockers
    ):
        blockers.append("wizard_reset_readiness_pass_claim_invalid")
    if payload.get("status") == "BLOCKED_RESET_AUTOMATION" and not reported_blockers:
        blockers.append("wizard_reset_readiness_blocked_claim_missing_reason")
    if not _authority_is_zero(payload):
        blockers.append("wizard_reset_readiness_authority_violation")

    browser_receipt_id = str(payload.get("browser_auth_receipt_id", "")).strip()
    browser_receipt_relative = str(payload.get("browser_auth_receipt_path", "")).strip()
    browser_receipt_expected = (
        f"data/research/wizard_browser_auth_readiness/{browser_receipt_id}.json"
    )
    browser_receipt_path = _safe_root_file(root, browser_receipt_relative)
    browser_receipt = _read_json(browser_receipt_path) if browser_receipt_path else {}
    browser_receipt_material = {
        key: value for key, value in browser_receipt.items() if key != "receipt_id"
    }
    browser_receipt_expected_id = (
        "wizardbrowserreadiness_"
        + sha256(_canonical_json(browser_receipt_material).encode("utf-8")).hexdigest()[:20]
    )
    browser_receipt_sha = (
        _file_sha256(browser_receipt_path)
        if browser_receipt_path is not None and browser_receipt_path.is_file()
        else ""
    )
    browser_required_at = _try_parse_utc(str(payload.get("browser_auth_required_at_utc", "")))
    browser_valid_until = _try_parse_utc(str(payload.get("browser_auth_valid_until_utc", "")))
    browser_sources_valid = True
    observed_route_kinds: set[str] = set()
    observed_source_paths: set[str] = set()
    for item in browser_receipt.get("selected_evidence", []):
        if not isinstance(item, dict):
            browser_sources_valid = False
            continue
        source_relative = str(item.get("source_path", ""))
        source_path = _safe_root_file(root, source_relative)
        if (
            source_path is None
            or not source_path.is_file()
            or _file_sha256(source_path) != str(item.get("source_sha256", ""))
        ):
            browser_sources_valid = False
        observed_route_kinds.add(str(item.get("route_kind", "")))
        observed_source_paths.add(source_relative)
    required_route_kinds = {str(value) for value in browser_receipt.get("required_route_kinds", [])}
    browser_binding_valid = bool(
        payload.get("browser_auth_valid_at_capture_window") is True
        and browser_receipt_id.startswith("wizardbrowserreadiness_")
        and browser_receipt_relative == browser_receipt_expected
        and browser_receipt_path is not None
        and browser_receipt_path.is_file()
        and browser_receipt_sha == str(payload.get("browser_auth_receipt_sha256", ""))
        and browser_receipt.get("receipt_id") == browser_receipt_id
        and browser_receipt_id == browser_receipt_expected_id
        and browser_receipt.get("status") == "PASS_AUTHENTICATED_BROWSER_ROUTES"
        and browser_receipt.get("authenticated_routes_ready") is True
        and browser_receipt.get("valid_until_utc") == payload.get("browser_auth_valid_until_utc")
        and browser_required_at is not None
        and browser_valid_until is not None
        and browser_valid_until >= browser_required_at
        and required_route_kinds == {"scanner", "pair_detail"}
        and observed_route_kinds == required_route_kinds
        and len(observed_source_paths) == len(required_route_kinds)
        and browser_sources_valid
        and _authority_is_zero(browser_receipt)
    )
    if not browser_binding_valid:
        blockers.append("wizard_reset_readiness_browser_auth_binding_invalid")

    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "receipt_id": receipt_id,
        "receipt_status": str(payload.get("status", "")),
        "state_sha256": wizard_reset_readiness_state_sha256(payload),
        "immutable_receipt_path": immutable_relative,
        "immutable_receipt_sha256": (
            _file_sha256(immutable_path) if immutable_path.is_file() else ""
        ),
    }


def _launch_agent_is_loaded(
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> bool:
    try:
        completed = runner(
            ["launchctl", "print", f"gui/{os.getuid()}/{LAUNCH_AGENT_LABEL}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return completed.returncode == 0


def _runtime_temp_viability(path: Path, *, minimum_free_bytes: int) -> tuple[bool, str]:
    if minimum_free_bytes < 0:
        return False, "minimum_free_bytes_invalid"
    if path.is_symlink():
        return False, "symbolic_link_not_allowed"
    if not path.is_dir():
        return False, "directory_missing"
    try:
        mode = path.stat().st_mode
        free_bytes = shutil.disk_usage(path).free
        if not mode & 0o200:
            return False, f"owner_write_permission_missing;free_bytes={free_bytes}"
        if free_bytes < minimum_free_bytes:
            return False, f"insufficient_free_bytes:{free_bytes}<{minimum_free_bytes}"
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".thewiz-runtime-probe-",
            dir=path,
        ) as probe:
            probe.write(b"ready")
            probe.flush()
    except OSError as exc:
        return False, f"write_probe_failed:{type(exc).__name__}"
    return True, f"writable;free_bytes={free_bytes}"


def _credential_configuration(root: Path) -> dict[str, object]:
    for name in (".env.local", ".env"):
        path = root / name
        value = _declared_value(path, API_KEY_ENV)
        if value is None:
            continue
        secure = not bool(path.stat().st_mode & 0o077)
        configured = bool(value and not _looks_placeholder(value))
        blocker = ""
        if not configured:
            blocker = "crypto_wizards_api_key_empty_or_placeholder"
        elif not secure:
            blocker = f"crypto_wizards_secret_file_permissions_insecure:{name}"
        return {
            "configured": configured,
            "secure": secure,
            "source": name,
            "evidence_path": name,
            "blocker": blocker,
        }
    if os.getenv(API_KEY_ENV, "").strip():
        return {
            "configured": False,
            "secure": True,
            "source": "process_environment_only",
            "evidence_path": "process_environment",
            "blocker": "wizard_api_credential_not_persisted_for_launch_agent",
        }
    return {
        "configured": False,
        "secure": False,
        "source": "missing",
        "evidence_path": ".env.local;.env;process_environment",
        "blocker": "CRYPTO_WIZARDS_API_KEY_missing",
    }


def _declared_value(path: Path, key: str) -> str | None:
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip("'\"")
    return None


def _looks_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {
        "",
        "changeme",
        "replace_me",
        "replace-me",
        "your_api_key",
        "your-api-key",
    } or normalized.startswith("<")


def _authority_is_zero(payload: dict[str, Any]) -> bool:
    return all(
        payload.get(field) is False
        for field in (
            "candidate_promotion_authority",
            "order_submission_included",
            "testnet_order_authority",
            "live_trading_authorized",
        )
        if field in payload
    )


def _strict_authority_is_zero(payload: dict[str, Any]) -> bool:
    return all(
        payload.get(field) is False
        for field in (
            "candidate_promotion_authority",
            "order_submission_included",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    )


def _scheduler_execution_binding_valid(
    *,
    root: Path,
    pointer: dict[str, Any],
    reset_at: datetime | None,
) -> bool:
    if reset_at is None:
        return False
    source_relative = str(pointer.get("pointer_source_path", ""))
    source_path = _safe_root_file(root, source_relative)
    source = _read_json(source_path) if source_path is not None else {}
    started_at = _try_parse_utc(str(pointer.get("started_at_utc", "")))
    prior_attempt_date = (reset_at - timedelta(days=1)).date().isoformat()
    source_matches_pointer = bool(
        source
        and all(pointer.get(key) == value for key, value in source.items())
        and source.get("receipt_id") == pointer.get("receipt_id")
    )
    return bool(
        pointer.get("schema_version") == "thewiz.corrective_wizard_proof_scheduler.v1"
        and pointer.get("execution_requested") is True
        and pointer.get("pointer_role") == "execution"
        and pointer.get("external_attempt_made") is True
        and pointer.get("attempt_date_utc") == prior_attempt_date
        and pointer.get("next_external_attempt_eligible_at") == reset_at.isoformat()
        and str(pointer.get("receipt_id", "")).startswith("wizardproof_")
        and started_at is not None
        and started_at < reset_at
        and source_path is not None
        and source_path.is_file()
        and source_matches_pointer
        and _strict_authority_is_zero(pointer)
        and _strict_authority_is_zero(source)
    )


def _manifest_source_lineage(*, root: Path, manifest: dict[str, Any]) -> tuple[bool, int, str, str]:
    binding = validate_capture_manifest_source_receipt(
        root=root,
        manifest_id=manifest.get("manifest_id"),
        manifest_path=manifest.get("immutable_manifest_path"),
        manifest_sha256=manifest.get("immutable_manifest_sha256"),
    )
    receipt_artifacts = binding.get("source_artifacts", [])
    frozen_artifacts = [
        {"path": _text(item.get("path")), "sha256": _text(item.get("sha256")).lower()}
        for item in receipt_artifacts
        if isinstance(item, dict)
    ]
    frozen_artifacts.sort(key=lambda item: item["path"])
    active_artifacts = manifest.get("source_artifacts")
    normalized_active = (
        sorted(
            [
                {
                    "path": _text(item.get("path")),
                    "sha256": _text(item.get("sha256")).lower(),
                }
                for item in active_artifacts
                if isinstance(item, dict)
            ],
            key=lambda item: item["path"],
        )
        if isinstance(active_artifacts, list)
        else []
    )
    fingerprint = _text(binding.get("source_artifacts_sha256"))
    valid = bool(
        binding.get("status") == "PASS"
        and frozen_artifacts
        and normalized_active == frozen_artifacts
        and manifest.get("source_artifacts_sha256") == fingerprint
        and manifest.get("source_receipt_id") == binding.get("receipt_id")
        and manifest.get("source_receipt_path") == binding.get("receipt_path")
        and manifest.get("source_receipt_sha256") == binding.get("receipt_sha256")
    )
    observed = (
        f"sources={len(frozen_artifacts)};sha256={fingerprint}"
        if valid
        else ";".join(binding.get("blockers", [])) or "active_source_receipt_binding_mismatch"
    )
    return valid, len(frozen_artifacts), fingerprint, observed


def _manifest_source_drift(
    *, root: Path, manifest: dict[str, Any], source_lineage_valid: bool
) -> tuple[bool, dict[str, Any]]:
    binding = validate_capture_manifest_source_receipt(
        root=root,
        manifest_id=manifest.get("manifest_id"),
        manifest_path=manifest.get("immutable_manifest_path"),
        manifest_sha256=manifest.get("immutable_manifest_sha256"),
    )
    source_receipt = {
        "source_artifacts": binding.get("source_artifacts", []),
        "source_artifacts_sha256": binding.get("source_artifacts_sha256", ""),
    }
    drift = classify_capture_manifest_source_drift(
        root=root,
        source_receipt=source_receipt,
    )
    metadata_only_paths = manifest.get("source_artifacts_current_metadata_only_paths")
    scientific_paths = manifest.get("source_artifacts_current_scientific_or_structural_paths")
    declarations_match = bool(
        manifest.get("source_artifacts_current_sha256") == drift["current_source_artifacts_sha256"]
        and manifest.get("source_artifacts_current_match") is drift["current_match"]
        and manifest.get("source_artifacts_current_drift_classification") == drift["classification"]
        and metadata_only_paths == drift["metadata_only_paths"]
        and scientific_paths == drift["scientific_or_structural_paths"]
    )
    safe_classification = bool(
        drift["classification"] in {"EXACT_MATCH", "GENERATED_METADATA_ONLY"}
        and not drift["scientific_or_structural_paths"]
    )
    return bool(
        source_lineage_valid
        and binding.get("status") == "PASS"
        and declarations_match
        and safe_classification
    ), drift


def _lane_accounting(payload: Any) -> tuple[int, int]:
    if not isinstance(payload, dict):
        return 0, 0
    calls = 0
    credits = 0
    for lane in payload.values():
        if not isinstance(lane, dict):
            return 0, 0
        calls += _safe_int(lane.get("calls"))
        credits += _safe_int(lane.get("credits"))
    return calls, credits


def _add_check(
    checks: list[dict[str, str]],
    *,
    name: str,
    passed: bool,
    observed: str,
    evidence_path: str,
    blocker: str,
) -> None:
    checks.append(
        {
            "check": name,
            "status": "PASS" if passed else "BLOCKED",
            "observed": observed,
            "evidence_path": evidence_path,
            "blocker": "" if passed else blocker,
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_root_file(root: Path, relative: str) -> Path | None:
    if not relative or Path(relative).is_absolute():
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["check", "status", "observed", "evidence_path", "blocker"],
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_immutable_json(payload: dict[str, Any], path: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"immutable readiness receipt collision: {path}")
        return
    _write_text(path, encoded)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _markdown(payload: dict[str, Any], checks: list[dict[str, str]]) -> str:
    lines = [
        "# Wizard Reset Readiness",
        "",
        f"- Status: `{payload['status']}`",
        f"- Capture window: `{payload['capture_window_status']}`",
        f"- Manifest: `{payload['manifest_id']}`",
        f"- Pending calls / planned credits: `{payload['pending_calls']} / {payload['planned_credits']}`",
        f"- Next eligible UTC: `{payload['next_external_attempt_eligible_at']}`",
        f"- Browser auth valid through capture window: `{payload['browser_auth_valid_at_capture_window']}`",
        f"- Browser auth valid until: `{payload['browser_auth_valid_until_utc']}`",
        f"- LaunchAgent loaded: `{payload['launch_agent_loaded']}`",
        f"- Workspace runtime temp ready: `{payload['runtime_temp_ready']}`",
        f"- Current source drift: `{payload['manifest_source_current_drift_classification']}`",
        "- Runtime credit preflight remains required.",
        "- No candidate, Testnet-order, or live-trading authority is granted.",
        "",
        "| Check | Status | Observed | Blocker |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['check']} | {row['status']} | {row['observed']} | {row['blocker']} |"
        for row in checks
    )
    return "\n".join(lines) + "\n"


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _try_parse_utc(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)
