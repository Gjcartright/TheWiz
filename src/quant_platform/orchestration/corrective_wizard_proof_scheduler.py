"""Bounded, research-only scheduling for Crypto Wizards exact-mode proofs."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

from quant_platform.active_pipeline import CommandResult
from quant_platform.crypto_wizards_history import (
    CryptoWizardsFetchError,
    fetch_credits_used,
)
from quant_platform.crypto_wizards_sweep import parse_wizard_credit_usage
from quant_platform.env import load_selected_env_keys
from quant_platform.orchestration.corrective_canonical_status import (
    SCHEDULER_EXECUTION_POINTER,
    publish_scheduler_status_pointers,
)
from quant_platform.orchestration.corrective_daily_scheduler import _acquire_lock
from quant_platform.orchestration.corrective_external_effects import (
    ExternalEffectSession,
    current_external_effect_issuer,
    current_external_effect_session,
    read_authorized_credential,
    reserved_external_effect_session,
)
from quant_platform.orchestration.corrective_program import complete_corrective_plan
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_registered_rerun_executor import (
    run_registered_research_rerun,
)
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_text,
    ensure_runtime_temp_directory,
    ensure_scheduler_log_directory,
    scheduler_contract,
    scheduler_launch_agent_plist,
    scheduler_log_directory,
    scheduler_python_path,
    scheduler_run_identity,
    write_immutable_json,
    write_launch_agent_plist,
)
from quant_platform.orchestration.corrective_scheduler_lock import (
    GovernedEvidenceLockBusy,
    GovernedEvidenceMaintenanceActive,
    governed_evidence_write_lock,
)
from quant_platform.orchestration.corrective_scheduler_supervisor import (
    supervise_scheduler_run,
)
from quant_platform.orchestration.corrective_wizard_capture_manifest import (
    build_corrective_wizard_capture_manifest,
    validate_capture_manifest_source_receipt,
)
from quant_platform.orchestration.corrective_wizard_capture_reconciliation import (
    reconcile_corrective_wizard_capture_manifest,
    validate_capture_manifest_binding,
    validate_capture_reconciliation_evidence,
)
from quant_platform.orchestration.corrective_wizard_copula_behavioral import (
    run_current_copula_behavioral_proofs,
)
from quant_platform.orchestration.corrective_wizard_dynamic_holdout import (
    build_dynamic_v2_review_handoff,
    build_dynamic_v2_reviewed_activation,
    build_dynamic_v2_supersession_gate,
    evaluate_dynamic_v2_holdout,
)
from quant_platform.orchestration.corrective_wizard_ou_holdout import (
    build_ou_v3_review_handoff,
    build_ou_v3_reviewed_activation,
    build_ou_v3_supersession_gate,
    run_ou_v3_prospective_holdout,
)
from quant_platform.orchestration.corrective_wizard_ou_v4_holdout import (
    run_ou_v4_prospective_holdout,
)
from quant_platform.orchestration.corrective_wizard_ou_v4_supreme_review import (
    build_ou_v4_supreme_review,
)
from quant_platform.orchestration.corrective_wizard_ou_v5_failure_attribution import (
    build_ou_v5_failure_attribution,
)
from quant_platform.orchestration.corrective_wizard_ou_v5_holdout import (
    run_ou_v5_prospective_holdout,
)
from quant_platform.orchestration.corrective_wizard_ou_v5_supreme_review import (
    build_ou_v5_supreme_review,
)
from quant_platform.orchestration.corrective_wizard_ou_v6_holdout import (
    run_ou_v6_prospective_holdout,
    validate_ou_v6_stage3_evidence,
)
from quant_platform.orchestration.corrective_wizard_ou_v6_supreme_review import (
    build_ou_v6_supreme_review,
)
from quant_platform.orchestration.corrective_wizard_parity import (
    build_corrective_wizard_parity,
)
from quant_platform.orchestration.effect_authority import EffectAuthorityError
from quant_platform.wizard_credit_budget import (
    COPULA_POST_CREDIT_COST,
    DEFAULT_PROOF_MAX_BATCHES,
    OU_V3_CUSTOM_SERIES_CREDIT_COST,
    OU_V4_CUSTOM_SERIES_CREDIT_COST,
    OU_V5_CUSTOM_SERIES_CREDIT_COST,
    OU_V6_CUSTOM_SERIES_CREDIT_COST,
    build_wizard_credit_budget_contract,
)
from quant_platform.wizard_credit_ledger import (
    PROOF_LANE,
    reconcile_wizard_credit_lane,
    reserve_wizard_credit_lane,
    validate_wizard_credit_lane_evidence,
)
from quant_platform.wizard_hyperliquid_mode_proof import (
    CUSTOM_SERIES_CREDIT_COST,
    DEFAULT_RESERVED_CREDITS,
    REQUEST_CAP,
    audit_exhaustive_wizard_mode_proof_inputs,
    build_exhaustive_wizard_mode_proof_queue,
    count_captured_exact_mode_responses,
    count_completed_exact_mode_proofs,
    refresh_activated_dynamic_v2_proofs,
    refresh_activated_ou_v3_proofs,
    refresh_activated_ou_v4_proofs,
    refresh_activated_ou_v5_proofs,
    refresh_activated_ou_v6_proofs,
    run_hyperliquid_wizard_mode_proofs,
)
from quant_platform.wizard_ou_v4_comparator_activation import (
    build_ou_v4_review_packet,
    build_ou_v4_supersession_gate,
    build_reviewed_ou_v4_activation,
)
from quant_platform.wizard_ou_v5_comparator_activation import (
    build_ou_v5_review_packet,
    build_ou_v5_supersession_gate,
    build_reviewed_ou_v5_activation,
)
from quant_platform.wizard_ou_v6_comparator_activation import (
    build_ou_v6_review_packet,
    build_ou_v6_supersession_gate,
    build_reviewed_ou_v6_activation,
)

ROOT = Path(__file__).resolve().parents[3]
WIZARD_API_BASE = "https://api.cryptowizards.net"
WIZARD_BACKTEST_ENDPOINT = f"{WIZARD_API_BASE}/v1beta/backtest"
WIZARD_COPULA_ENDPOINT = f"{WIZARD_API_BASE}/v1beta/copula"
WIZARD_CREDITS_ENDPOINT = f"{WIZARD_API_BASE}/v1beta/credits-used"
SCHEMA_VERSION = "thewiz.corrective_wizard_proof_scheduler.v1"
LAUNCH_AGENT_LABEL = "com.thewiz.corrective-wizard-proof"
DEFAULT_INTERVAL_SECONDS = 10 * 60
PROOF_LOCK_TIMEOUT_SECONDS = 4 * 60 * 60
DEFAULT_MAX_BATCHES = DEFAULT_PROOF_MAX_BATCHES
WIZARD_DAILY_CREDIT_RESET_UTC = "00:00"
PROOF_ENABLE_ENV = "QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF"
API_KEY_ENV = "CRYPTO_WIZARDS_API_KEY"
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
    "reports/active/corrective_wizard_next_capture_manifest.csv",
    "reports/active/corrective_wizard_next_capture_manifest.json",
    "reports/active/corrective_wizard_capture_reconciliation.csv",
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
    "src/quant_platform/orchestration/corrective_wizard_ou_v5_supreme_review.py",
    "src/quant_platform/wizard_ou_v5_comparator_activation.py",
    "src/quant_platform/orchestration/corrective_wizard_capture_manifest.py",
    "src/quant_platform/orchestration/corrective_wizard_capture_reconciliation.py",
    "src/quant_platform/orchestration/corrective_wizard_ou_holdout.py",
    "src/quant_platform/orchestration/corrective_wizard_ou_v4_holdout.py",
    "src/quant_platform/orchestration/corrective_wizard_ou_v5_holdout.py",
    "src/quant_platform/orchestration/corrective_wizard_ou_v5_failure_attribution.py",
    "src/quant_platform/orchestration/corrective_wizard_ou_v6_holdout.py",
    "src/quant_platform/orchestration/corrective_wizard_ou_v6_supreme_review.py",
    "src/quant_platform/orchestration/corrective_wizard_ou_v4_supreme_review.py",
    "src/quant_platform/wizard_dynamic_comparator_activation.py",
    "src/quant_platform/wizard_ou_v4_comparator_activation.py",
    "src/quant_platform/wizard_ou_v6_comparator_activation.py",
)


def run_corrective_wizard_proof_cycle(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    execute: bool = False,
    force: bool = False,
    internal_continuation_only: bool = False,
    max_batches: int = DEFAULT_MAX_BATCHES,
    max_proofs_per_batch: int = REQUEST_CAP,
    queue_builder: Callable[..., CommandResult] = build_exhaustive_wizard_mode_proof_queue,
    proof_runner: Callable[..., CommandResult] = run_hyperliquid_wizard_mode_proofs,
    parity_refresher: Callable[..., CommandResult] = build_corrective_wizard_parity,
    checkpoint_refresher: Callable[..., CommandResult] | None = complete_corrective_plan,
    registered_rerun_runner: Callable[..., CommandResult] | None = (run_registered_research_rerun),
    budget_builder: Callable[..., CommandResult] = build_wizard_credit_budget_contract,
    input_auditor: Callable[..., CommandResult] = audit_exhaustive_wizard_mode_proof_inputs,
    capture_manifest_builder: Callable[..., CommandResult] = (
        build_corrective_wizard_capture_manifest
    ),
    capture_manifest_reconciler: Callable[..., CommandResult] = (
        reconcile_corrective_wizard_capture_manifest
    ),
    dynamic_holdout_evaluator: Callable[..., CommandResult] | None = (evaluate_dynamic_v2_holdout),
    dynamic_supersession_gate_builder: Callable[..., CommandResult] | None = (
        build_dynamic_v2_supersession_gate
    ),
    dynamic_activation_planner: Callable[..., CommandResult] | None = (
        build_dynamic_v2_reviewed_activation
    ),
    dynamic_review_packet_builder: Callable[..., CommandResult] | None = (
        build_dynamic_v2_review_handoff
    ),
    dynamic_proof_refresher: Callable[..., CommandResult] | None = (
        refresh_activated_dynamic_v2_proofs
    ),
    copula_behavioral_runner: Callable[..., CommandResult] | None = (
        run_current_copula_behavioral_proofs
    ),
    ou_v3_holdout_runner: Callable[..., CommandResult] | None = (run_ou_v3_prospective_holdout),
    ou_v3_supersession_gate_builder: Callable[..., CommandResult] | None = (
        build_ou_v3_supersession_gate
    ),
    ou_v3_review_packet_builder: Callable[..., CommandResult] | None = (build_ou_v3_review_handoff),
    ou_v3_activation_planner: Callable[..., CommandResult] | None = (
        build_ou_v3_reviewed_activation
    ),
    ou_v3_proof_refresher: Callable[..., CommandResult] | None = (refresh_activated_ou_v3_proofs),
    ou_v4_holdout_runner: Callable[..., CommandResult] | None = (run_ou_v4_prospective_holdout),
    ou_v4_supersession_gate_builder: Callable[..., CommandResult] | None = (
        build_ou_v4_supersession_gate
    ),
    ou_v4_review_packet_builder: Callable[..., CommandResult] | None = (build_ou_v4_review_packet),
    ou_v4_supreme_reviewer: Callable[..., CommandResult] | None = (build_ou_v4_supreme_review),
    ou_v4_activation_planner: Callable[..., CommandResult] | None = (
        build_reviewed_ou_v4_activation
    ),
    ou_v4_proof_refresher: Callable[..., CommandResult] | None = (refresh_activated_ou_v4_proofs),
    ou_v5_holdout_runner: Callable[..., CommandResult] | None = (run_ou_v5_prospective_holdout),
    ou_v5_failure_attributor: Callable[..., CommandResult] | None = (
        build_ou_v5_failure_attribution
    ),
    ou_v5_supersession_gate_builder: Callable[..., CommandResult] | None = (
        build_ou_v5_supersession_gate
    ),
    ou_v5_review_packet_builder: Callable[..., CommandResult] | None = (build_ou_v5_review_packet),
    ou_v5_supreme_reviewer: Callable[..., CommandResult] | None = (build_ou_v5_supreme_review),
    ou_v5_activation_planner: Callable[..., CommandResult] | None = (
        build_reviewed_ou_v5_activation
    ),
    ou_v5_proof_refresher: Callable[..., CommandResult] | None = (refresh_activated_ou_v5_proofs),
    ou_v6_holdout_runner: Callable[..., CommandResult] | None = (run_ou_v6_prospective_holdout),
    ou_v6_supersession_gate_builder: Callable[..., CommandResult] | None = (
        build_ou_v6_supersession_gate
    ),
    ou_v6_review_packet_builder: Callable[..., CommandResult] | None = (build_ou_v6_review_packet),
    ou_v6_supreme_reviewer: Callable[..., CommandResult] | None = (build_ou_v6_supreme_review),
    ou_v6_activation_planner: Callable[..., CommandResult] | None = (
        build_reviewed_ou_v6_activation
    ),
    ou_v6_proof_refresher: Callable[..., CommandResult] | None = (refresh_activated_ou_v6_proofs),
    credit_reserver: Callable[..., CommandResult] = reserve_wizard_credit_lane,
    credit_reconciler: Callable[..., CommandResult] = reconcile_wizard_credit_lane,
    credits_fetcher: Callable[..., Any] | None = None,
    credential_reader: Callable[[str], str | None] | None = None,
) -> CommandResult:
    """Run one daily, credit-reserved proof cycle and persist its full decision receipt."""

    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if not 1 <= max_proofs_per_batch <= REQUEST_CAP:
        raise ValueError(f"max_proofs_per_batch must be between 1 and {REQUEST_CAP}")
    if internal_continuation_only and (not execute or force):
        raise ValueError("internal_continuation_only requires execute=True and force=False")
    credits_fetcher = credits_fetcher or fetch_credits_used

    started_at = _as_utc(now)
    active = root / "reports" / "active"
    receipts = active / "wizard_proof_scheduler_receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    lock_path = active / ".corrective_wizard_proof.lock"
    latest_path = active / "corrective_wizard_proof_scheduler_status.json"
    execution_status_path = active / SCHEDULER_EXECUTION_POINTER
    receipt_path = receipts / started_at.strftime("%Y-%m-%d_%H%M%S_%f.json")
    previous = _read_json(execution_status_path) or _read_json(latest_path)
    prior_external_attempt_today = _already_attempted_today(previous, started_at)
    blockers: list[str] = []
    batch_summaries: list[dict[str, Any]] = []
    queue_path = active / "exhaustive_wizard_exact_mode_proof_queue.csv"
    proof_path = active / "hyperliquid_wizard_vendor_mode_proofs.csv"
    queue_eligible = 0
    completed_before = 0
    completed_after = completed_before
    responses_captured_before = 0
    responses_captured_after = responses_captured_before
    external_attempt_made = False
    external_attempt_made_this_cycle = False
    proof_request_slots_selected = 0
    exact_mode_requests_attempted = 0
    exact_mode_responses_captured_this_cycle = 0
    exact_mode_attempted_credits = 0
    exact_mode_completed_credits = 0
    copula_behavioral_attempted_credits = 0
    copula_behavioral_responses_captured_this_cycle = 0
    copula_behavioral_completed_credits = 0
    copula_behavioral_response_accounting_valid = True
    ou_v3_status = "NOT_CONFIGURED"
    ou_v3_capture_status_path = active / "wizard_ou_v3_capture_status.json"
    ou_v3_evaluation_status = "NOT_EVALUATED"
    ou_v3_missing_cells_before = 0
    ou_v3_required_responses = 0
    ou_v3_responses_available = 0
    ou_v3_calls_made = 0
    ou_v3_responses_captured_this_cycle = 0
    ou_v3_credits_used_before: int | None = None
    ou_v3_attempted_credits = 0
    ou_v3_completed_credits = 0
    ou_v3_response_accounting_valid = True
    ou_v3_supersession_gate_status = "NOT_CONFIGURED"
    ou_v3_supersession_gate_path = active / "wizard_ou_v3_supersession_gate.json"
    ou_v3_review_packet_status = "NOT_CONFIGURED"
    ou_v3_review_packet_id = ""
    ou_v3_review_packet_path = active / "wizard_ou_v3_review_packet.json"
    ou_v3_activation_status = "NOT_CONFIGURED"
    ou_v3_activation_id = ""
    ou_v3_activation_path = active / "wizard_ou_v3_activation_status.json"
    ou_v3_comparator_generation = 1
    ou_v3_proof_refresh_status = "NOT_CONFIGURED"
    ou_v3_proofs_refreshed = 0
    ou_v3_proof_refresh_path = active / "wizard_ou_v3_proof_refresh_status.json"
    ou_v4_status = "NOT_CONFIGURED"
    ou_v4_capture_status_path = active / "wizard_ou_v4_capture_status.json"
    ou_v4_evaluation_status = "NOT_EVALUATED"
    ou_v4_missing_cells_before = 0
    ou_v4_required_responses = 0
    ou_v4_responses_available = 0
    ou_v4_calls_made = 0
    ou_v4_responses_captured_this_cycle = 0
    ou_v4_credits_used_before: int | None = None
    ou_v4_attempted_credits = 0
    ou_v4_completed_credits = 0
    ou_v4_response_accounting_valid = True
    ou_v4_contract = root / "config" / "wizard_ou_comparator_v4_holdout.json"
    ou_v4_supersession_gate_status = "NOT_CONFIGURED"
    ou_v4_supersession_gate_path = active / "wizard_ou_v4_supersession_gate.json"
    ou_v4_review_packet_status = "NOT_CONFIGURED"
    ou_v4_review_packet_id = ""
    ou_v4_review_packet_path = active / "wizard_ou_v4_review_packet.json"
    ou_v4_supreme_review_status = "NOT_CONFIGURED"
    ou_v4_supreme_review_recommendation = ""
    ou_v4_supreme_review_path = root / "reports/supreme_team/wizard_ou_v4_review.json"
    ou_v4_activation_status = "NOT_CONFIGURED"
    ou_v4_activation_id = ""
    ou_v4_activation_path = active / "wizard_ou_v4_activation_status.json"
    ou_v4_comparator_generation = 1
    ou_v4_proof_refresh_status = "NOT_CONFIGURED"
    ou_v4_proofs_refreshed = 0
    ou_v4_proof_refresh_path = active / "wizard_ou_v4_proof_refresh_status.json"
    ou_v5_status = "NOT_CONFIGURED"
    ou_v5_capture_status_path = active / "wizard_ou_v5_capture_status.json"
    ou_v5_evaluation_status = "NOT_EVALUATED"
    ou_v5_missing_cells_before = 0
    ou_v5_required_responses = 0
    ou_v5_responses_available = 0
    ou_v5_calls_made = 0
    ou_v5_responses_captured_this_cycle = 0
    ou_v5_credits_used_before: int | None = None
    ou_v5_attempted_credits = 0
    ou_v5_completed_credits = 0
    ou_v5_response_accounting_valid = True
    ou_v5_failure_attribution_status = "WAITING_HOLDOUT"
    ou_v5_failure_attribution_id = ""
    ou_v5_failure_attribution_path = active / "wizard_ou_v5_failure_attribution.json"
    ou_v5_contract = root / "config" / "wizard_ou_comparator_v5_holdout.json"
    ou_v5_supersession_gate_status = "NOT_CONFIGURED"
    ou_v5_supersession_gate_path = active / "wizard_ou_v5_supersession_gate.json"
    ou_v5_review_packet_status = "NOT_CONFIGURED"
    ou_v5_review_packet_id = ""
    ou_v5_review_packet_path = active / "wizard_ou_v5_review_packet.json"
    ou_v5_supreme_review_status = "NOT_CONFIGURED"
    ou_v5_supreme_review_recommendation = ""
    ou_v5_supreme_review_path = root / "reports/supreme_team/wizard_ou_v5_review.json"
    ou_v5_activation_status = "NOT_CONFIGURED"
    ou_v5_activation_id = ""
    ou_v5_activation_path = active / "wizard_ou_v5_activation_status.json"
    ou_v5_comparator_generation = 1
    ou_v5_proof_refresh_status = "NOT_CONFIGURED"
    ou_v5_proofs_refreshed = 0
    ou_v5_proof_refresh_path = active / "wizard_ou_v5_proof_refresh_status.json"
    ou_v6_status = "NOT_CONFIGURED"
    ou_v6_capture_status_path = active / "wizard_ou_v6_capture_status.json"
    ou_v6_evaluation_status = "NOT_EVALUATED"
    ou_v6_missing_cells_before = 0
    ou_v6_required_responses = 0
    ou_v6_responses_available = 0
    ou_v6_calls_made = 0
    ou_v6_responses_captured_this_cycle = 0
    ou_v6_credits_used_before: int | None = None
    ou_v6_attempted_credits = 0
    ou_v6_completed_credits = 0
    ou_v6_response_accounting_valid = True
    ou_v6_contract = root / "config" / "wizard_ou_comparator_v6_holdout.json"
    ou_v6_terminal_failure = False
    ou_v6_supersession_gate_status = "NOT_CONFIGURED"
    ou_v6_supersession_gate_path = active / "wizard_ou_v6_supersession_gate.json"
    ou_v6_review_packet_status = "NOT_CONFIGURED"
    ou_v6_review_packet_id = ""
    ou_v6_review_packet_path = active / "wizard_ou_v6_review_packet.json"
    ou_v6_supreme_review_status = "NOT_CONFIGURED"
    ou_v6_supreme_review_recommendation = ""
    ou_v6_supreme_review_path = root / "reports/supreme_team/wizard_ou_v6_review.json"
    ou_v6_activation_status = "NOT_CONFIGURED"
    ou_v6_activation_id = ""
    ou_v6_activation_path = active / "wizard_ou_v6_activation_status.json"
    ou_v6_comparator_generation = 1
    ou_v6_proof_refresh_status = "NOT_CONFIGURED"
    ou_v6_proofs_refreshed = 0
    ou_v6_proof_refresh_path = active / "wizard_ou_v6_proof_refresh_status.json"
    external_lanes_authorized_this_cycle = False
    exact_mode_external_authorized = False
    ou_v3_external_authorized = False
    ou_v4_external_authorized = False
    ou_v5_external_authorized = False
    ou_v6_external_authorized = False
    copula_external_authorized = False
    exact_mode_manifest_call_limit = 0
    ou_v3_manifest_call_limit = 0
    ou_v4_manifest_call_limit = 0
    ou_v5_manifest_call_limit = 0
    ou_v6_manifest_call_limit = 0
    copula_manifest_call_limit = 0
    proof_lane_attempted_credits = 0
    proof_lane_completed_credits = 0
    proof_lane_uncompleted_attempted_credits = 0
    observed_used_before: int | None = None
    observed_used_after: int | None = None
    status = "BLOCKED"
    parity_status = "NOT_REFRESHED"
    checkpoint_status = "NOT_REFRESHED"
    checkpoint_pre_rerun_status = "NOT_REFRESHED"
    checkpoint_post_rerun_status = "NOT_REFRESHED"
    checkpoint_refresh_phase = "NONE"
    credit_budget_status = "NOT_CHECKED"
    scheduled_credit_ceiling = 0
    credit_headroom_after_reserve = 0
    credit_budget_path = active / "wizard_credit_budget_contract.json"
    capture_manifest_status = "NOT_CHECKED"
    capture_manifest_state = "NOT_CHECKED"
    capture_manifest_enforced = False
    capture_manifest_id = ""
    capture_manifest_path = active / "corrective_wizard_next_capture_manifest.json"
    capture_manifest_immutable_path = ""
    capture_manifest_immutable_sha256 = ""
    capture_manifest_pending_calls = 0
    capture_manifest_planned_credits = 0
    capture_manifest_capture_eligible_now = False
    capture_manifest_next_eligible_at = ""
    capture_manifest_blockers: list[str] = []
    capture_manifest_lane_totals: dict[str, dict[str, int]] = {}
    capture_manifest_accounting_valid = True
    capture_manifest_candidate_id = ""
    capture_manifest_candidate_immutable_path = ""
    capture_manifest_candidate_immutable_sha256 = ""
    capture_manifest_candidate_binding_valid = True
    capture_manifest_candidate_source_binding_valid = True
    capture_manifest_source_receipt_id = ""
    capture_manifest_source_receipt_path = ""
    capture_manifest_source_receipt_sha256 = ""
    capture_manifest_source_artifacts_sha256 = ""
    capture_manifest_previous_id = str(previous.get("capture_manifest_id", ""))
    capture_manifest_previous_immutable_path = str(
        previous.get("capture_manifest_immutable_path", "")
    )
    capture_manifest_previous_immutable_sha256 = str(
        previous.get("capture_manifest_immutable_sha256", "")
    )
    capture_manifest_carried_forward = False
    capture_manifest_drift_detected = False
    capture_manifest_continuity_status = "NOT_CHECKED"
    capture_manifest_continuity_valid = True
    capture_manifest_continuity_blockers: list[str] = []
    capture_reconciliation_status = "NOT_REQUESTED"
    capture_reconciliation_path = active / "corrective_wizard_capture_reconciliation.json"
    capture_reconciliation_id = ""
    capture_reconciliation_immutable_path = ""
    capture_reconciliation_immutable_sha256 = ""
    capture_reconciliation_manifest_id = ""
    capture_reconciliation_manifest_path = ""
    capture_reconciliation_manifest_sha256 = ""
    capture_reconciliation_current_status = "NOT_REQUESTED"
    capture_reconciliation_carried_forward = False
    capture_reconciliation_required_calls = 0
    capture_reconciliation_completed_calls = 0
    capture_reconciliation_pending_calls = 0
    capture_reconciliation_blocked_calls = 0
    capture_reconciliation_blockers: list[str] = []
    capture_reconciliation_local_state_valid = True
    capture_reconciliation_valid = False
    capture_reconciliation_complete = False
    credit_reservation_status = "NOT_REQUESTED"
    credit_reservation_blocker = ""
    credit_reservation_id = ""
    credit_reservation_path = ""
    external_effect_reservation_binding_id = ""
    credit_reconciliation_status = "NOT_REQUESTED"
    credit_reconciliation_blocker = ""
    credit_reconciliation_id = ""
    credit_reconciliation_path = ""
    skip_credit_reconciliation_this_cycle = False
    proof_lane_credit_ceiling = 0
    credit_reservation_planned_credits = 0
    if prior_external_attempt_today:
        credit_reservation_status = str(previous.get("credit_reservation_status", "NOT_REQUESTED"))
        credit_reservation_blocker = str(previous.get("credit_reservation_blocker", ""))
        credit_reservation_id = str(previous.get("credit_reservation_id", ""))
        credit_reservation_path = str(previous.get("credit_reservation_path", ""))
        credit_reconciliation_status = str(
            previous.get("credit_reconciliation_status", "NOT_REQUESTED")
        )
        credit_reconciliation_blocker = str(previous.get("credit_reconciliation_blocker", ""))
        credit_reconciliation_id = str(previous.get("credit_reconciliation_id", ""))
        credit_reconciliation_path = str(previous.get("credit_reconciliation_path", ""))
    input_audit_status = "NOT_CHECKED"
    input_audit_ready = 0
    input_audit_blocked = 0
    input_audit_retry_safe = 0
    input_audit_changed_after_vendor_4xx = 0
    input_audit_unchanged_vendor_4xx = 0
    input_audit_path = active / "exhaustive_wizard_mode_proof_input_audit.json"
    input_audit_detail_path = active / "exhaustive_wizard_mode_proof_input_audit.csv"
    registered_rerun_status = "NOT_EVALUATED"
    registered_rerun_receipt_path = ""
    registered_learning_handoff_status = "NOT_EVALUATED"
    registered_stage5_research_gate_pass = False
    registered_learning_handoff_blocker = ""
    dynamic_holdout_status = "NOT_CONFIGURED"
    dynamic_holdout_status_path = active / "wizard_dynamic_v2_holdout_status.json"
    dynamic_holdout_supersession_eligible = False
    dynamic_supersession_gate_status = "NOT_CONFIGURED"
    dynamic_supersession_gate_path = active / "wizard_dynamic_v2_supersession_gate.json"
    dynamic_activation_status = "NOT_CONFIGURED"
    dynamic_activation_id = ""
    dynamic_activation_path = active / "wizard_dynamic_v2_activation_status.json"
    dynamic_comparator_generation = 1
    dynamic_proof_refresh_status = "NOT_CONFIGURED"
    dynamic_proof_refresh_path = active / "wizard_dynamic_v2_proof_refresh_status.json"
    dynamic_proofs_refreshed = 0
    dynamic_review_packet_status = "NOT_CONFIGURED"
    dynamic_review_packet_id = ""
    dynamic_review_packet_path = active / "wizard_dynamic_v2_review_packet.json"
    copula_behavioral_status = "NOT_CONFIGURED"
    copula_behavioral_status_path = active / "wizard_copula_behavioral_status.json"
    copula_behavioral_expected_cells = 0
    copula_behavioral_cells_passed = 0
    copula_behavioral_provenance_cells = 0
    copula_behavioral_endpoint_calls = 0
    copula_behavioral_endpoint_calls_required = 0
    copula_behavioral_new_capture_cells = 0
    copula_behavioral_reused_capture_cells = 0
    copula_behavioral_parity_proven = False
    copula_formula_parity_proven = False
    copula_cohort_receipt_id = ""
    copula_cohort_receipt_path = ""
    copula_cohort_receipt_sha256 = ""
    copula_cohort_receipt_valid = False
    formula_proofs_expected = 0
    accepted_mode_evidence_cells = 0
    lock_acquired = False
    external_stack = ExitStack()
    external_effect_session: ExternalEffectSession | None = None
    external_effect_authority_status = "NOT_REQUIRED"
    external_effect_reservation_sha256 = ""
    authorized_api_key: str | None = None
    env_security: dict[str, Any] = {
        "api_key_present": False,
        "key_source": (
            "not_checked_before_reservation" if execute else "not_checked_non_execute"
        ),
        "check_performed": False,
        "insecure_secret_files": _insecure_secret_files(root),
    }
    try:
        _acquire_lock(
            lock_path,
            now=started_at,
            timeout_seconds=PROOF_LOCK_TIMEOUT_SECONDS,
        )
        lock_acquired = True
        queue_result = queue_builder(root=root)
        queue_path = Path(queue_result.paths.get("queue", queue_path))
        queue_eligible = int(queue_result.summary.get("eligible_rows", 0) or 0)
        completed_before = count_completed_exact_mode_proofs(
            queue_path=queue_path,
            proof_path=proof_path,
        )
        completed_after = completed_before
        responses_captured_before = count_captured_exact_mode_responses(
            queue_path=queue_path,
            proof_path=proof_path,
        )
        responses_captured_after = responses_captured_before
        input_audit = input_auditor(root=root, queue_path=queue_path)
        input_audit_status = str(input_audit.summary.get("status", "BLOCKED"))
        input_audit_ready = int(input_audit.summary.get("ready_rows", 0) or 0)
        input_audit_blocked = int(input_audit.summary.get("blocked_rows", 0) or 0)
        input_audit_retry_safe = int(
            input_audit.summary.get("retry_safe_rows", input_audit_ready) or 0
        )
        input_audit_changed_after_vendor_4xx = int(
            input_audit.summary.get("changed_after_vendor_4xx_rows", 0) or 0
        )
        input_audit_unchanged_vendor_4xx = int(
            input_audit.summary.get("unchanged_vendor_4xx_rows", 0) or 0
        )
        input_audit_path = Path(input_audit.paths.get("input_audit_summary", input_audit_path))
        input_audit_detail_path = Path(
            input_audit.paths.get("input_audit", input_audit_detail_path)
        )
        budget = budget_builder(
            root=root,
            now=started_at,
            proof_max_batches=max_batches,
            proof_batch_size=max_proofs_per_batch,
        )
        credit_budget_status = str(budget.summary.get("status", "BLOCKED"))
        scheduled_credit_ceiling = int(budget.summary.get("scheduled_credit_ceiling", 0) or 0)
        credit_headroom_after_reserve = int(budget.summary.get("headroom_after_reserve", 0) or 0)
        proof_lane_credit_ceiling = (
            int(budget.summary.get("exact_mode_proof_credit_ceiling", 0) or 0)
            + int(budget.summary.get("copula_behavioral_credit_ceiling", 0) or 0)
            + int(budget.summary.get("ou_v3_prospective_credit_ceiling", 0) or 0)
            + int(budget.summary.get("ou_v4_prospective_credit_ceiling", 0) or 0)
            + int(budget.summary.get("ou_v5_prospective_credit_ceiling", 0) or 0)
            + int(budget.summary.get("ou_v6_prospective_credit_ceiling", 0) or 0)
        )
        credit_budget_path = Path(budget.paths.get("budget_summary", credit_budget_path))
        capture_manifest = capture_manifest_builder(
            root=root,
            now=started_at,
            queue_path=queue_path,
            proof_path=proof_path,
            input_audit_path=input_audit_detail_path,
            credit_budget_path=credit_budget_path,
        )
        capture_manifest_status = str(capture_manifest.summary.get("status", "BLOCKED"))
        capture_manifest_state = str(capture_manifest.summary.get("capture_state", "BLOCKED"))
        capture_manifest_enforced = bool(capture_manifest.summary.get("manifest_enforced", False))
        capture_manifest_id = str(capture_manifest.summary.get("manifest_id", ""))
        capture_manifest_path = Path(capture_manifest.paths.get("status", capture_manifest_path))
        capture_manifest_immutable_path = str(
            capture_manifest.summary.get("immutable_manifest_path", "")
        )
        capture_manifest_immutable_sha256 = str(
            capture_manifest.summary.get("immutable_manifest_sha256", "")
        )
        capture_manifest_pending_calls = int(capture_manifest.summary.get("pending_calls", 0) or 0)
        capture_manifest_planned_credits = int(
            capture_manifest.summary.get("planned_credits", 0) or 0
        )
        capture_manifest_capture_eligible_now = bool(
            capture_manifest.summary.get("capture_eligible_now", False)
        )
        capture_manifest_next_eligible_at = str(
            capture_manifest.summary.get("next_external_attempt_eligible_at", "")
        )
        capture_manifest_blockers = [
            str(value) for value in capture_manifest.summary.get("blockers", [])
        ]
        capture_manifest_lane_totals = dict(capture_manifest.summary.get("lane_totals", {}))
        capture_manifest_candidate_id = capture_manifest_id
        capture_manifest_candidate_immutable_path = capture_manifest_immutable_path
        capture_manifest_candidate_immutable_sha256 = capture_manifest_immutable_sha256
        capture_manifest_source_receipt_id = str(
            capture_manifest.summary.get("source_receipt_id", "")
        )
        capture_manifest_source_receipt_path = str(
            capture_manifest.summary.get("source_receipt_path", "")
        )
        capture_manifest_source_receipt_sha256 = str(
            capture_manifest.summary.get("source_receipt_sha256", "")
        )
        capture_manifest_source_artifacts_sha256 = str(
            capture_manifest.summary.get("source_artifacts_sha256", "")
        )
        if capture_manifest_enforced:
            candidate_binding = validate_capture_manifest_binding(
                root=root,
                manifest_id=capture_manifest_candidate_id,
                manifest_path=capture_manifest_candidate_immutable_path,
                manifest_sha256=capture_manifest_candidate_immutable_sha256,
                expected_pending_calls=capture_manifest_pending_calls,
                expected_planned_credits=capture_manifest_planned_credits,
                expected_lane_totals=capture_manifest_lane_totals,
            )
            candidate_source_binding = validate_capture_manifest_source_receipt(
                root=root,
                manifest_id=capture_manifest_candidate_id,
                manifest_path=capture_manifest_candidate_immutable_path,
                manifest_sha256=capture_manifest_candidate_immutable_sha256,
            )
            capture_manifest_candidate_source_binding_valid = bool(
                candidate_source_binding.get("status") == "PASS"
            )
            capture_manifest_candidate_binding_valid = bool(
                candidate_binding.get("status") == "PASS"
                and capture_manifest_candidate_source_binding_valid
            )
            if not capture_manifest_candidate_binding_valid:
                capture_manifest_continuity_status = "BLOCKED_CANDIDATE_BINDING"
                capture_manifest_continuity_valid = False
                capture_manifest_continuity_blockers.extend(
                    str(value) for value in candidate_binding.get("blockers", [])
                )
                capture_manifest_continuity_blockers.extend(
                    str(value) for value in candidate_source_binding.get("blockers", [])
                )
            else:
                capture_manifest_continuity_status = "PASS_NEW_FROZEN_COHORT"

        if _capture_manifest_unresolved(previous):
            capture_manifest_carried_forward = True
            previous_binding = validate_capture_manifest_binding(
                root=root,
                manifest_id=capture_manifest_previous_id,
                manifest_path=capture_manifest_previous_immutable_path,
                manifest_sha256=capture_manifest_previous_immutable_sha256,
                expected_pending_calls=int(previous.get("capture_manifest_pending_calls", 0) or 0),
                expected_planned_credits=int(
                    previous.get("capture_manifest_planned_credits", 0) or 0
                ),
                expected_lane_totals=dict(previous.get("capture_manifest_lane_totals", {})),
            )
            previous_source_binding = validate_capture_manifest_source_receipt(
                root=root,
                manifest_id=capture_manifest_previous_id,
                manifest_path=capture_manifest_previous_immutable_path,
                manifest_sha256=capture_manifest_previous_immutable_sha256,
            )
            previous_binding_valid = bool(
                previous_binding.get("status") == "PASS"
                and previous_source_binding.get("status") == "PASS"
            )
            if not previous_binding_valid:
                capture_manifest_continuity_status = "BLOCKED_PRIOR_BINDING"
                capture_manifest_continuity_valid = False
                capture_manifest_continuity_blockers.extend(
                    f"prior_{value}" for value in previous_binding.get("blockers", [])
                )
                capture_manifest_continuity_blockers.extend(
                    f"prior_{value}" for value in previous_source_binding.get("blockers", [])
                )
            else:
                capture_manifest_id = capture_manifest_previous_id
                capture_manifest_immutable_path = capture_manifest_previous_immutable_path
                capture_manifest_immutable_sha256 = capture_manifest_previous_immutable_sha256
                capture_manifest_pending_calls = int(previous_binding.get("pending_calls", 0) or 0)
                capture_manifest_planned_credits = int(
                    previous_binding.get("planned_credits", 0) or 0
                )
                capture_manifest_lane_totals = dict(previous_binding.get("lane_totals", {}))
                capture_manifest_drift_detected = bool(
                    not capture_manifest_candidate_binding_valid
                    or capture_manifest_candidate_id != capture_manifest_id
                    or capture_manifest_candidate_immutable_path != capture_manifest_immutable_path
                    or capture_manifest_candidate_immutable_sha256
                    != capture_manifest_immutable_sha256
                )
                if capture_manifest_drift_detected:
                    capture_manifest_continuity_status = "BLOCKED_COHORT_DRIFT"
                    capture_manifest_continuity_valid = False
                    capture_manifest_continuity_blockers.append(
                        "unresolved_capture_manifest_candidate_drift:"
                        f"{capture_manifest_id}:"
                        f"{capture_manifest_candidate_id}"
                    )
                elif capture_manifest_candidate_binding_valid:
                    capture_manifest_continuity_status = "PASS_PRIOR_UNRESOLVED_COHORT_MATCH"
                    capture_manifest_continuity_valid = True

        capture_manifest_continuity_blockers = sorted(
            set(filter(None, capture_manifest_continuity_blockers))
        )
        if capture_manifest_enforced:
            exact_mode_manifest_call_limit = _manifest_lane_calls(
                capture_manifest_lane_totals, "exact_mode_backtest"
            )
            ou_v3_manifest_call_limit = _manifest_lane_calls(
                capture_manifest_lane_totals, "ou_v3_holdout"
            )
            ou_v4_manifest_call_limit = _manifest_lane_calls(
                capture_manifest_lane_totals, "ou_v4_holdout"
            )
            ou_v5_manifest_call_limit = _manifest_lane_calls(
                capture_manifest_lane_totals, "ou_v5_holdout"
            )
            ou_v6_manifest_call_limit = _manifest_lane_calls(
                capture_manifest_lane_totals, "ou_v6_holdout"
            )
            copula_manifest_call_limit = _manifest_lane_calls(
                capture_manifest_lane_totals, "copula_behavioral"
            )
            credit_reservation_planned_credits = capture_manifest_planned_credits
        else:
            exact_mode_manifest_call_limit = max_batches * max_proofs_per_batch
            ou_v3_manifest_call_limit = 4
            ou_v4_manifest_call_limit = 8
            ou_v5_manifest_call_limit = 8
            ou_v6_manifest_call_limit = 8
            copula_manifest_call_limit = REQUEST_CAP
            credit_reservation_planned_credits = proof_lane_credit_ceiling
        prior_credit_evidence = (
            _validate_prior_daily_credit_evidence(
                root=root,
                timestamp=started_at,
                previous=previous,
            )
            if execute and not force and prior_external_attempt_today
            else {"status": "NOT_REQUIRED", "blocker": ""}
        )

        if internal_continuation_only and not prior_external_attempt_today:
            status = "BLOCKED_INTERNAL_CONTINUATION_WITHOUT_PRIOR_DAILY_ATTEMPT"
            blockers.append(
                "internal_registered_continuation_requires_verified_prior_daily_attempt"
            )
        elif (
            input_audit_status != "PASS"
            or input_audit_ready != queue_eligible
            or input_audit_retry_safe != queue_eligible
            or input_audit_unchanged_vendor_4xx != 0
        ):
            status = "BLOCKED_INPUT_AUDIT"
            blockers.append(
                f"exact_mode_input_audit_failed:{input_audit_ready}_ready:"
                f"{input_audit_blocked}_blocked:{input_audit_retry_safe}_retry_safe:"
                f"{input_audit_unchanged_vendor_4xx}_unchanged_vendor_4xx:"
                f"{queue_eligible}_eligible"
            )
        elif credit_budget_status != "PASS":
            status = "BLOCKED_CREDIT_BUDGET_CONTRACT"
            blockers.append("scheduled_wizard_lanes_exceed_daily_budget")
        elif capture_manifest_enforced and capture_manifest_status != "PASS":
            status = "BLOCKED_CAPTURE_MANIFEST"
            blockers.extend(capture_manifest_blockers or ["capture_manifest_not_pass"])
        elif capture_manifest_enforced and not capture_manifest_continuity_valid:
            status = (
                "BLOCKED_CAPTURE_MANIFEST_DRIFT"
                if capture_manifest_drift_detected
                else "BLOCKED_CAPTURE_MANIFEST_CONTINUITY"
            )
            blockers.extend(
                capture_manifest_continuity_blockers or ["capture_manifest_continuity_not_proven"]
            )
        elif execute and env_security["insecure_secret_files"]:
            status = "BLOCKED_ENVIRONMENT"
            blockers.append("secret_file_permissions_too_open")
        elif (
            execute
            and not force
            and prior_external_attempt_today
            and capture_manifest_enforced
            and not capture_manifest_capture_eligible_now
            and prior_credit_evidence.get("status") == "PASS"
        ):
            status = "DEFERRED_CAPTURE_MANIFEST_WINDOW"
            external_attempt_made = True
            skip_credit_reconciliation_this_cycle = True
            credit_reservation_status = "PRIOR_RESERVATION_VERIFIED"
            credit_reservation_blocker = ""
            credit_reservation_id = str(prior_credit_evidence.get("reservation_id", ""))
            credit_reservation_path = str(prior_credit_evidence.get("reservation_path", ""))
            credit_reconciliation_status = "PRIOR_RECONCILIATION_VERIFIED"
            credit_reconciliation_blocker = ""
            credit_reconciliation_id = str(prior_credit_evidence.get("reconciliation_id", ""))
            credit_reconciliation_path = str(prior_credit_evidence.get("reconciliation_path", ""))
            blockers.append("capture_manifest_not_yet_eligible")
        elif execute and not force and prior_external_attempt_today:
            try:
                reservation = credit_reserver(
                    root=root,
                    lane=PROOF_LANE,
                    planned_credits=credit_reservation_planned_credits,
                    now=started_at,
                    daily_credit_limit=int(budget.summary.get("daily_credit_limit", 0) or 0),
                    protected_reserve=int(budget.summary.get("reserved_credits", 0) or 0),
                )
                credit_reservation_status = str(reservation.summary.get("status", "BLOCKED"))
                credit_reservation_blocker = str(reservation.summary.get("blocker", ""))
                credit_reservation_id = str(reservation.summary.get("reservation_id", ""))
                credit_reservation_path = (
                    _relative(Path(reservation.paths["reservation"]), root)
                    if "reservation" in reservation.paths
                    else ""
                )
                previous_reconciliation_id = str(previous.get("credit_reconciliation_id", ""))
                reconciliations = {
                    str(value) for value in reservation.summary.get("lane_reconciliation_ids", [])
                }
                prior_accounting_verified = bool(
                    credit_reservation_status == "REUSED"
                    and previous_reconciliation_id
                    and previous_reconciliation_id in reconciliations
                )
            except (OSError, TypeError, ValueError) as exc:
                prior_accounting_verified = False
                credit_reservation_status = "BLOCKED"
                credit_reservation_blocker = (
                    f"shared_credit_reservation_failed:{safe_exception_code(exc)}"
                )
            external_attempt_made = True
            if prior_accounting_verified:
                status = "DEFERRED_SAME_UTC_DAY"
                credit_reconciliation_status = "PRIOR_RECONCILIATION_VERIFIED"
                credit_reconciliation_id = str(previous.get("credit_reconciliation_id", ""))
                credit_reconciliation_path = str(previous.get("credit_reconciliation_path", ""))
                blockers.append("bounded_external_proof_cycle_already_attempted_this_utc_day")
            else:
                status = "BLOCKED_PRIOR_CREDIT_ACCOUNTING"
                blockers.append(
                    credit_reservation_blocker
                    or "prior_external_attempt_missing_immutable_credit_reconciliation"
                )
        else:
            if execute and capture_manifest_enforced and not capture_manifest_capture_eligible_now:
                status = "DEFERRED_CAPTURE_MANIFEST_WINDOW"
                blockers.append("capture_manifest_not_yet_eligible")
            else:
                reservation_ready = True
                if execute:
                    try:
                        reservation = credit_reserver(
                            root=root,
                            lane=PROOF_LANE,
                            planned_credits=credit_reservation_planned_credits,
                            now=started_at,
                            daily_credit_limit=int(
                                budget.summary.get("daily_credit_limit", 0) or 0
                            ),
                            protected_reserve=int(budget.summary.get("reserved_credits", 0) or 0),
                            max_external_requests=_external_request_ceiling(
                                max_batches=max_batches,
                                exact_mode_calls=exact_mode_manifest_call_limit,
                                ou_v3_calls=ou_v3_manifest_call_limit,
                                ou_v4_calls=ou_v4_manifest_call_limit,
                                ou_v5_calls=ou_v5_manifest_call_limit,
                                ou_v6_calls=ou_v6_manifest_call_limit,
                                copula_calls=copula_manifest_call_limit,
                            ),
                        )
                        credit_reservation_status = str(
                            reservation.summary.get("status", "BLOCKED")
                        )
                        credit_reservation_blocker = str(reservation.summary.get("blocker", ""))
                        credit_reservation_id = str(reservation.summary.get("reservation_id", ""))
                        external_effect_reservation_binding_id = str(
                            reservation.summary.get(
                                "effect_reservation_binding_id",
                                "",
                            )
                        )
                        credit_reservation_path = (
                            _relative(Path(reservation.paths["reservation"]), root)
                            if "reservation" in reservation.paths
                            else ""
                        )
                        remaining_lane_credits = int(
                            reservation.summary.get("lane_remaining_reserved_credits", 0) or 0
                        )
                        external_spend_authorized = (
                            reservation.summary.get("external_spend_authorized") is True
                        )
                        reservation_ready = bool(
                            credit_reservation_status in {"PASS", "REUSED"}
                            and external_spend_authorized
                            and credit_reservation_planned_credits <= remaining_lane_credits
                        )
                    except (OSError, TypeError, ValueError) as exc:
                        reservation_ready = False
                        credit_reservation_status = "BLOCKED"
                        credit_reservation_blocker = (
                            f"shared_credit_reservation_failed:{safe_exception_code(exc)}"
                        )
                    if not reservation_ready:
                        status = "BLOCKED_SHARED_CREDIT_RESERVATION"
                        blockers.append(
                            credit_reservation_blocker or "insufficient_lane_reservation_remaining"
                        )
                if reservation_ready:
                    external_effect_required = bool(
                        execute
                        and not internal_continuation_only
                        and any(
                            value > 0
                            for value in (
                                exact_mode_manifest_call_limit,
                                ou_v3_manifest_call_limit,
                                ou_v4_manifest_call_limit,
                                ou_v5_manifest_call_limit,
                                ou_v6_manifest_call_limit,
                                copula_manifest_call_limit,
                            )
                        )
                    )
                    if external_effect_required:
                        try:
                            if current_external_effect_issuer() is None:
                                raise EffectAuthorityError(
                                    "wizard_external_effect_issuer_missing"
                                )
                            (
                                external_effect_reservation_sha256,
                                reservation_artifact,
                            ) = _validate_external_reservation(
                                root=root,
                                reservation_path=credit_reservation_path,
                                reservation_id=credit_reservation_id,
                            )
                            external_effect_session = external_stack.enter_context(
                                reserved_external_effect_session(
                                    reservation_id=credit_reservation_id,
                                    reservation_sha256=(
                                        external_effect_reservation_sha256
                                    ),
                                    max_total_requests=_external_request_ceiling(
                                        max_batches=max_batches,
                                        exact_mode_calls=(
                                            exact_mode_manifest_call_limit
                                        ),
                                        ou_v3_calls=ou_v3_manifest_call_limit,
                                        ou_v4_calls=ou_v4_manifest_call_limit,
                                        ou_v5_calls=ou_v5_manifest_call_limit,
                                        ou_v6_calls=ou_v6_manifest_call_limit,
                                        copula_calls=copula_manifest_call_limit,
                                    ),
                                    max_total_credits=(
                                        credit_reservation_planned_credits
                                    ),
                                    reservation_binding_id=(
                                        external_effect_reservation_binding_id
                                        or None
                                    ),
                                )
                            )
                            authorized_api_key = read_authorized_credential(
                                API_KEY_ENV,
                                reader=(
                                    credential_reader
                                    or (
                                        lambda key: _deferred_credential_reader(
                                            root, key
                                        )
                                    )
                                ),
                            )
                            env_security.update(
                                {
                                    "api_key_present": True,
                                    "key_source": (
                                        "injected_authorized_reader"
                                        if credential_reader is not None
                                        else "authorized_selected_environment"
                                    ),
                                    "check_performed": True,
                                    "reservation_artifact": _relative(
                                        reservation_artifact, root
                                    ),
                                }
                            )
                            external_effect_authority_status = "PASS"
                        except (
                            EffectAuthorityError,
                            OSError,
                            TypeError,
                            ValueError,
                            json.JSONDecodeError,
                        ) as exc:
                            reservation_ready = False
                            external_effect_authority_status = "BLOCKED"
                            status = "BLOCKED_EXTERNAL_EFFECT_AUTHORITY"
                            blockers.append(
                                "wizard_external_effect_authority_failed:"
                                f"{safe_exception_code(exc)}"
                            )
                    else:
                        external_effect_authority_status = (
                            "NOT_REQUIRED_INTERNAL_CONTINUATION"
                            if internal_continuation_only
                            else "NOT_REQUIRED_NO_EXTERNAL_CALLS"
                        )
                if reservation_ready:
                    external_lanes_authorized_this_cycle = bool(
                        external_effect_session is not None
                    )
                    exact_mode_external_authorized = bool(
                        external_lanes_authorized_this_cycle and exact_mode_manifest_call_limit > 0
                    )
                    ou_v3_external_authorized = bool(
                        external_lanes_authorized_this_cycle and ou_v3_manifest_call_limit > 0
                    )
                    ou_v4_external_authorized = bool(
                        external_lanes_authorized_this_cycle and ou_v4_manifest_call_limit > 0
                    )
                    ou_v5_external_authorized = bool(
                        external_lanes_authorized_this_cycle and ou_v5_manifest_call_limit > 0
                    )
                    ou_v6_external_authorized = bool(
                        external_lanes_authorized_this_cycle and ou_v6_manifest_call_limit > 0
                    )
                    copula_external_authorized = bool(
                        external_lanes_authorized_this_cycle and copula_manifest_call_limit > 0
                    )
                    if exact_mode_external_authorized or (
                        not execute and exact_mode_manifest_call_limit > 0
                    ):
                        (
                            status,
                            completed_after,
                            responses_captured_after,
                            external_attempt_made_this_cycle,
                            proof_request_slots_selected,
                        ) = _run_batches(
                            root=root,
                            started_at=started_at,
                            execute=exact_mode_external_authorized,
                            max_batches=max_batches,
                            max_proofs_per_batch=max_proofs_per_batch,
                            max_external_calls=exact_mode_manifest_call_limit,
                            queue_path=queue_path,
                            queue_eligible=queue_eligible,
                            completed_before=completed_before,
                            responses_captured_before=responses_captured_before,
                            proof_runner=proof_runner,
                            api_key=authorized_api_key,
                            batch_summaries=batch_summaries,
                            blockers=blockers,
                        )
                    elif external_lanes_authorized_this_cycle:
                        status = "MANIFESTED_NON_EXACT_LANES_READY"
                    external_attempt_made = external_attempt_made_this_cycle

        dynamic_contract = root / "config" / "wizard_dynamic_comparator_v2_holdout.json"
        if dynamic_holdout_evaluator is not None and dynamic_contract.is_file():
            try:
                dynamic_holdout = dynamic_holdout_evaluator(root=root, now=started_at)
                dynamic_holdout_status = str(dynamic_holdout.summary.get("status", "BLOCKED"))
                dynamic_holdout_supersession_eligible = bool(
                    dynamic_holdout.summary.get("comparator_supersession_eligible", False)
                )
                dynamic_holdout_status_path = Path(
                    dynamic_holdout.paths.get("status", dynamic_holdout_status_path)
                )
                if dynamic_supersession_gate_builder is not None:
                    supersession_gate = dynamic_supersession_gate_builder(root=root)
                    dynamic_supersession_gate_status = str(
                        supersession_gate.summary.get("status", "BLOCKED")
                    )
                    dynamic_supersession_gate_path = Path(
                        supersession_gate.paths.get(
                            "supersession_gate", dynamic_supersession_gate_path
                        )
                    )
                if dynamic_activation_planner is not None:
                    if dynamic_review_packet_builder is not None:
                        try:
                            review_packet = dynamic_review_packet_builder(
                                root=root,
                                now=started_at,
                            )
                            dynamic_review_packet_status = str(
                                review_packet.summary.get("status", "BLOCKED")
                            )
                            dynamic_review_packet_id = str(
                                review_packet.summary.get("review_packet_id", "")
                            )
                            dynamic_review_packet_path = Path(
                                review_packet.paths.get("review_packet", dynamic_review_packet_path)
                            )
                        except Exception as exc:  # noqa: BLE001 - fail closed
                            dynamic_review_packet_status = "EVALUATION_FAILED"
                            blockers.append(
                                f"dynamic_v2_review_packet_failed:{safe_exception_code(exc)}"
                            )
                    try:
                        activation = dynamic_activation_planner(
                            root=root,
                            apply=False,
                            now=started_at,
                        )
                        dynamic_activation_status = str(activation.summary.get("status", "BLOCKED"))
                        dynamic_activation_id = str(activation.summary.get("activation_id", ""))
                        dynamic_comparator_generation = (
                            int(activation.summary.get("comparator_generation", 2) or 2)
                            if dynamic_activation_status == "APPLIED_RESEARCH_COMPARATOR_ONLY"
                            else 1
                        )
                        dynamic_activation_path = Path(
                            activation.paths.get("activation_status", dynamic_activation_path)
                        )
                    except Exception as exc:  # noqa: BLE001 - fail closed on lineage
                        dynamic_activation_status = "EVALUATION_FAILED"
                        blockers.append(f"dynamic_v2_activation_failed:{safe_exception_code(exc)}")
                if dynamic_proof_refresher is not None:
                    try:
                        refresh = dynamic_proof_refresher(
                            root=root,
                            proof_path=proof_path,
                            now=started_at,
                        )
                        dynamic_proof_refresh_status = str(refresh.summary.get("status", "BLOCKED"))
                        dynamic_proofs_refreshed = int(
                            refresh.summary.get("refreshed_dynamic_rows", 0) or 0
                        )
                        dynamic_proof_refresh_path = Path(
                            refresh.paths.get("refresh_status", dynamic_proof_refresh_path)
                        )
                        if dynamic_proof_refresh_status == "PASS":
                            completed_after = count_completed_exact_mode_proofs(
                                queue_path=queue_path,
                                proof_path=proof_path,
                            )
                            responses_captured_after = count_captured_exact_mode_responses(
                                queue_path=queue_path,
                                proof_path=proof_path,
                            )
                    except Exception as exc:  # noqa: BLE001 - retain raw evidence
                        dynamic_proof_refresh_status = "EVALUATION_FAILED"
                        blockers.append(
                            f"dynamic_v2_proof_refresh_failed:{safe_exception_code(exc)}"
                        )
            except Exception as exc:  # noqa: BLE001 - retain captured proof evidence
                dynamic_holdout_status = "EVALUATION_FAILED"
                blockers.append(f"dynamic_v2_holdout_failed:{safe_exception_code(exc)}")

        ou_v3_contract = root / "config" / "wizard_ou_comparator_v3_holdout.json"
        if ou_v3_holdout_runner is not None and ou_v3_contract.is_file():
            try:
                ou_v3 = _run_external_lane(
                    authorized=ou_v3_external_authorized,
                    target=WIZARD_BACKTEST_ENDPOINT,
                    operation="POST_OU_V3_HOLDOUT_WITH_CREDIT_PREFLIGHT",
                    lane="ou_v3_holdout",
                    call_limit=ou_v3_manifest_call_limit,
                    credit_cost_per_call=OU_V3_CUSTOM_SERIES_CREDIT_COST,
                    callback=lambda: ou_v3_holdout_runner(
                        root=root,
                        now=started_at,
                        execute=ou_v3_external_authorized,
                        api_key=authorized_api_key,
                    ),
                )
                ou_v3_status = str(ou_v3.summary.get("status", "BLOCKED"))
                ou_v3_evaluation_status = str(
                    ou_v3.summary.get("evaluation_status", "NOT_EVALUATED")
                )
                ou_v3_missing_cells_before = int(ou_v3.summary.get("missing_cells_before", 0) or 0)
                ou_v3_required_responses = int(ou_v3.summary.get("required_responses", 0) or 0)
                ou_v3_responses_available = int(ou_v3.summary.get("responses_available", 0) or 0)
                ou_v3_calls_made = int(ou_v3.summary.get("calls_made", 0) or 0)
                ou_v3_responses_captured_this_cycle = int(
                    ou_v3.summary.get("responses_captured", 0) or 0
                )
                raw_credits_used_before = ou_v3.summary.get("credits_used_before")
                ou_v3_credits_used_before = (
                    int(raw_credits_used_before) if raw_credits_used_before is not None else None
                )
                ou_v3_attempted_credits = int(ou_v3.summary.get("credits_attempted", 0) or 0)
                ou_v3_completed_credits = int(ou_v3.summary.get("credits_completed", 0) or 0)
                ou_v3_capture_status_path = Path(
                    ou_v3.paths.get("capture_status", ou_v3_capture_status_path)
                )
                if (
                    ou_v3_responses_captured_this_cycle > ou_v3_calls_made
                    or ou_v3_attempted_credits != ou_v3_calls_made * OU_V3_CUSTOM_SERIES_CREDIT_COST
                    or ou_v3_completed_credits
                    != ou_v3_responses_captured_this_cycle * OU_V3_CUSTOM_SERIES_CREDIT_COST
                ):
                    ou_v3_response_accounting_valid = False
                    blockers.append("ou_v3_response_or_credit_accounting_invalid")
                if ou_v3_external_authorized and ou_v3_missing_cells_before > 0:
                    external_attempt_made_this_cycle = True
                    external_attempt_made = True
            except Exception as exc:  # noqa: BLE001 - preserve preregistered evidence
                ou_v3_status = "EVALUATION_FAILED"
                blockers.append(f"ou_v3_holdout_failed:{safe_exception_code(exc)}")

            if ou_v3_supersession_gate_builder is not None:
                try:
                    ou_gate = ou_v3_supersession_gate_builder(root=root)
                    ou_v3_supersession_gate_status = str(ou_gate.summary.get("status", "BLOCKED"))
                    ou_v3_supersession_gate_path = Path(
                        ou_gate.paths.get("supersession_gate", ou_v3_supersession_gate_path)
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed
                    ou_v3_supersession_gate_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v3_supersession_gate_failed:{safe_exception_code(exc)}")
            if ou_v3_review_packet_builder is not None:
                try:
                    ou_packet = ou_v3_review_packet_builder(
                        root=root,
                        now=started_at,
                    )
                    ou_v3_review_packet_status = str(ou_packet.summary.get("status", "BLOCKED"))
                    ou_v3_review_packet_id = str(ou_packet.summary.get("review_packet_id", ""))
                    ou_v3_review_packet_path = Path(
                        ou_packet.paths.get("review_packet", ou_v3_review_packet_path)
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed
                    ou_v3_review_packet_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v3_review_packet_failed:{safe_exception_code(exc)}")
            if ou_v3_activation_planner is not None:
                try:
                    ou_activation = ou_v3_activation_planner(
                        root=root,
                        apply=False,
                        now=started_at,
                    )
                    ou_v3_activation_status = str(ou_activation.summary.get("status", "BLOCKED"))
                    ou_v3_activation_id = str(ou_activation.summary.get("activation_id", ""))
                    ou_v3_comparator_generation = (
                        int(ou_activation.summary.get("comparator_generation", 3) or 3)
                        if ou_v3_activation_status == "APPLIED_RESEARCH_COMPARATOR_ONLY"
                        else 1
                    )
                    ou_v3_activation_path = Path(
                        ou_activation.paths.get("activation_status", ou_v3_activation_path)
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed
                    ou_v3_activation_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v3_activation_failed:{safe_exception_code(exc)}")
            if ou_v3_proof_refresher is not None:
                try:
                    ou_refresh = ou_v3_proof_refresher(
                        root=root,
                        proof_path=proof_path,
                        now=started_at,
                    )
                    ou_v3_proof_refresh_status = str(ou_refresh.summary.get("status", "BLOCKED"))
                    ou_v3_proofs_refreshed = int(
                        ou_refresh.summary.get("refreshed_ou_rows", 0) or 0
                    )
                    ou_v3_proof_refresh_path = Path(
                        ou_refresh.paths.get("refresh_status", ou_v3_proof_refresh_path)
                    )
                    if ou_v3_proof_refresh_status == "PASS":
                        completed_after = count_completed_exact_mode_proofs(
                            queue_path=queue_path,
                            proof_path=proof_path,
                        )
                        responses_captured_after = count_captured_exact_mode_responses(
                            queue_path=queue_path,
                            proof_path=proof_path,
                        )
                except Exception as exc:  # noqa: BLE001 - retain raw evidence
                    ou_v3_proof_refresh_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v3_proof_refresh_failed:{safe_exception_code(exc)}")

        if ou_v4_holdout_runner is not None and ou_v4_contract.is_file():
            try:
                ou_v4 = _run_external_lane(
                    authorized=ou_v4_external_authorized,
                    target=WIZARD_BACKTEST_ENDPOINT,
                    operation="POST_OU_V4_HOLDOUT_WITH_CREDIT_PREFLIGHT",
                    lane="ou_v4_holdout",
                    call_limit=ou_v4_manifest_call_limit,
                    credit_cost_per_call=OU_V4_CUSTOM_SERIES_CREDIT_COST,
                    callback=lambda: ou_v4_holdout_runner(
                        root=root,
                        now=started_at,
                        execute=ou_v4_external_authorized,
                        api_key=authorized_api_key,
                    ),
                )
                ou_v4_status = str(ou_v4.summary.get("status", "BLOCKED"))
                ou_v4_evaluation_status = str(
                    ou_v4.summary.get("evaluation_status", "NOT_EVALUATED")
                )
                ou_v4_missing_cells_before = int(ou_v4.summary.get("missing_cells_before", 0) or 0)
                ou_v4_required_responses = int(ou_v4.summary.get("required_responses", 0) or 0)
                ou_v4_responses_available = int(ou_v4.summary.get("responses_available", 0) or 0)
                ou_v4_calls_made = int(ou_v4.summary.get("calls_made", 0) or 0)
                ou_v4_responses_captured_this_cycle = int(
                    ou_v4.summary.get("responses_captured", 0) or 0
                )
                raw_ou_v4_credits_before = ou_v4.summary.get("credits_used_before")
                ou_v4_credits_used_before = (
                    int(raw_ou_v4_credits_before) if raw_ou_v4_credits_before is not None else None
                )
                ou_v4_attempted_credits = int(ou_v4.summary.get("credits_attempted", 0) or 0)
                ou_v4_completed_credits = int(ou_v4.summary.get("credits_completed", 0) or 0)
                ou_v4_capture_status_path = Path(
                    ou_v4.paths.get("capture_status", ou_v4_capture_status_path)
                )
                if (
                    ou_v4_responses_captured_this_cycle > ou_v4_calls_made
                    or ou_v4_attempted_credits != ou_v4_calls_made * OU_V4_CUSTOM_SERIES_CREDIT_COST
                    or ou_v4_completed_credits
                    != ou_v4_responses_captured_this_cycle * OU_V4_CUSTOM_SERIES_CREDIT_COST
                ):
                    ou_v4_response_accounting_valid = False
                    blockers.append("ou_v4_response_or_credit_accounting_invalid")
                if ou_v4_external_authorized and ou_v4_missing_cells_before > 0:
                    external_attempt_made_this_cycle = True
                    external_attempt_made = True
                if ou_v4_external_authorized and ou_v4_status in {
                    "BLOCKED",
                    "FAILED",
                }:
                    blocker = str(ou_v4.summary.get("blocker", ""))
                    errors = [str(value) for value in ou_v4.summary.get("errors", [])]
                    blockers.extend(
                        [
                            blocker or "ou_v4_capture_not_complete",
                            *errors,
                        ]
                    )
            except Exception as exc:  # noqa: BLE001 - preserve frozen v4 evidence
                ou_v4_status = "EVALUATION_FAILED"
                blockers.append(f"ou_v4_holdout_failed:{safe_exception_code(exc)}")

            if ou_v4_supersession_gate_builder is not None:
                try:
                    v4_gate = ou_v4_supersession_gate_builder(root=root)
                    ou_v4_supersession_gate_status = str(v4_gate.summary.get("status", "BLOCKED"))
                    ou_v4_supersession_gate_path = Path(
                        v4_gate.paths.get("supersession_gate", ou_v4_supersession_gate_path)
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed
                    ou_v4_supersession_gate_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v4_supersession_gate_failed:{safe_exception_code(exc)}")
            if ou_v4_review_packet_builder is not None:
                try:
                    v4_packet = ou_v4_review_packet_builder(
                        root=root,
                        now=started_at,
                    )
                    ou_v4_review_packet_status = str(v4_packet.summary.get("status", "BLOCKED"))
                    ou_v4_review_packet_id = str(v4_packet.summary.get("review_packet_id", ""))
                    ou_v4_review_packet_path = Path(
                        v4_packet.paths.get("review_packet", ou_v4_review_packet_path)
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed
                    ou_v4_review_packet_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v4_review_packet_failed:{safe_exception_code(exc)}")
            if ou_v4_supreme_reviewer is not None:
                try:
                    v4_supreme = ou_v4_supreme_reviewer(
                        root=root,
                        now=started_at,
                    )
                    ou_v4_supreme_review_status = str(v4_supreme.summary.get("status", "BLOCKED"))
                    ou_v4_supreme_review_recommendation = str(
                        v4_supreme.summary.get("recommendation", "")
                    )
                    ou_v4_supreme_review_path = Path(
                        v4_supreme.paths.get("status", ou_v4_supreme_review_path)
                    )
                except Exception as exc:  # noqa: BLE001 - advisory fails closed
                    ou_v4_supreme_review_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v4_supreme_review_failed:{safe_exception_code(exc)}")
            if ou_v4_activation_planner is not None:
                try:
                    v4_activation = ou_v4_activation_planner(
                        root=root,
                        apply=False,
                        now=started_at,
                    )
                    ou_v4_activation_status = str(v4_activation.summary.get("status", "BLOCKED"))
                    ou_v4_activation_id = str(v4_activation.summary.get("activation_id", ""))
                    ou_v4_comparator_generation = (
                        int(v4_activation.summary.get("comparator_generation", 4) or 4)
                        if ou_v4_activation_status == "APPLIED_RESEARCH_COMPARATOR_ONLY"
                        else 1
                    )
                    ou_v4_activation_path = Path(
                        v4_activation.paths.get("activation_status", ou_v4_activation_path)
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed on lineage
                    ou_v4_activation_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v4_activation_failed:{safe_exception_code(exc)}")
            if ou_v4_proof_refresher is not None:
                try:
                    v4_refresh = ou_v4_proof_refresher(
                        root=root,
                        proof_path=proof_path,
                        now=started_at,
                    )
                    ou_v4_proof_refresh_status = str(v4_refresh.summary.get("status", "BLOCKED"))
                    ou_v4_proofs_refreshed = int(
                        v4_refresh.summary.get("refreshed_ou_rows", 0) or 0
                    )
                    ou_v4_proof_refresh_path = Path(
                        v4_refresh.paths.get("refresh_status", ou_v4_proof_refresh_path)
                    )
                    if ou_v4_proof_refresh_status == "PASS":
                        completed_after = count_completed_exact_mode_proofs(
                            queue_path=queue_path,
                            proof_path=proof_path,
                        )
                        responses_captured_after = count_captured_exact_mode_responses(
                            queue_path=queue_path,
                            proof_path=proof_path,
                        )
                except Exception as exc:  # noqa: BLE001 - retain raw evidence
                    ou_v4_proof_refresh_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v4_proof_refresh_failed:{safe_exception_code(exc)}")

        if ou_v5_holdout_runner is not None and ou_v5_contract.is_file():
            try:
                ou_v5 = _run_external_lane(
                    authorized=ou_v5_external_authorized,
                    target=WIZARD_BACKTEST_ENDPOINT,
                    operation="POST_OU_V5_HOLDOUT_WITH_CREDIT_PREFLIGHT",
                    lane="ou_v5_holdout",
                    call_limit=ou_v5_manifest_call_limit,
                    credit_cost_per_call=OU_V5_CUSTOM_SERIES_CREDIT_COST,
                    callback=lambda: ou_v5_holdout_runner(
                        root=root,
                        now=started_at,
                        execute=ou_v5_external_authorized,
                        api_key=authorized_api_key,
                    ),
                )
                ou_v5_status = str(ou_v5.summary.get("status", "BLOCKED"))
                ou_v5_evaluation_status = str(
                    ou_v5.summary.get("evaluation_status", "NOT_EVALUATED")
                )
                ou_v5_missing_cells_before = int(ou_v5.summary.get("missing_cells_before", 0) or 0)
                ou_v5_required_responses = int(ou_v5.summary.get("required_responses", 0) or 0)
                ou_v5_responses_available = int(ou_v5.summary.get("responses_available", 0) or 0)
                ou_v5_calls_made = int(ou_v5.summary.get("calls_made", 0) or 0)
                ou_v5_responses_captured_this_cycle = int(
                    ou_v5.summary.get("responses_captured", 0) or 0
                )
                raw_ou_v5_credits_before = ou_v5.summary.get("credits_used_before")
                ou_v5_credits_used_before = (
                    int(raw_ou_v5_credits_before) if raw_ou_v5_credits_before is not None else None
                )
                ou_v5_attempted_credits = int(ou_v5.summary.get("credits_attempted", 0) or 0)
                ou_v5_completed_credits = int(ou_v5.summary.get("credits_completed", 0) or 0)
                ou_v5_capture_status_path = Path(
                    ou_v5.paths.get("capture_status", ou_v5_capture_status_path)
                )
                if (
                    ou_v5_responses_captured_this_cycle > ou_v5_calls_made
                    or ou_v5_attempted_credits != ou_v5_calls_made * OU_V5_CUSTOM_SERIES_CREDIT_COST
                    or ou_v5_completed_credits
                    != ou_v5_responses_captured_this_cycle * OU_V5_CUSTOM_SERIES_CREDIT_COST
                ):
                    ou_v5_response_accounting_valid = False
                    blockers.append("ou_v5_response_or_credit_accounting_invalid")
                if ou_v5_external_authorized and ou_v5_missing_cells_before > 0:
                    external_attempt_made_this_cycle = True
                    external_attempt_made = True
                if ou_v5_external_authorized and ou_v5_status in {
                    "BLOCKED",
                    "FAILED",
                }:
                    blocker = str(ou_v5.summary.get("blocker", ""))
                    errors = [str(value) for value in ou_v5.summary.get("errors", [])]
                    blockers.extend([blocker or "ou_v5_capture_not_complete", *errors])
            except Exception as exc:  # noqa: BLE001 - preserve frozen v5 evidence
                ou_v5_status = "EVALUATION_FAILED"
                blockers.append(f"ou_v5_holdout_failed:{safe_exception_code(exc)}")

            if ou_v5_evaluation_status == "PASS":
                ou_v5_failure_attribution_status = "NOT_REQUIRED_HOLDOUT_PASS"
            elif ou_v5_evaluation_status == "FAIL" and ou_v5_failure_attributor is not None:
                try:
                    v5_attribution = ou_v5_failure_attributor(
                        root=root,
                        now=started_at,
                    )
                    ou_v5_failure_attribution_status = str(
                        v5_attribution.summary.get("status", "BLOCKED")
                    )
                    ou_v5_failure_attribution_id = str(
                        v5_attribution.summary.get("attribution_id", "")
                    )
                    ou_v5_failure_attribution_path = Path(
                        v5_attribution.paths.get("status", ou_v5_failure_attribution_path)
                    )
                    if ou_v5_failure_attribution_status != "PASS_FAILURE_ATTRIBUTION_COMPLETE":
                        blockers.append("ou_v5_failure_attribution_incomplete")
                except Exception as exc:  # noqa: BLE001 - preserve negative evidence
                    ou_v5_failure_attribution_status = "ATTRIBUTION_FAILED"
                    blockers.append(f"ou_v5_failure_attribution_failed:{safe_exception_code(exc)}")

            if ou_v5_supersession_gate_builder is not None:
                try:
                    v5_gate = ou_v5_supersession_gate_builder(root=root)
                    ou_v5_supersession_gate_status = str(v5_gate.summary.get("status", "BLOCKED"))
                    ou_v5_supersession_gate_path = Path(
                        v5_gate.paths.get("supersession_gate", ou_v5_supersession_gate_path)
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed
                    ou_v5_supersession_gate_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v5_supersession_gate_failed:{safe_exception_code(exc)}")
            if ou_v5_review_packet_builder is not None:
                try:
                    v5_packet = ou_v5_review_packet_builder(root=root, now=started_at)
                    ou_v5_review_packet_status = str(v5_packet.summary.get("status", "BLOCKED"))
                    ou_v5_review_packet_id = str(v5_packet.summary.get("review_packet_id", ""))
                    ou_v5_review_packet_path = Path(
                        v5_packet.paths.get("review_packet", ou_v5_review_packet_path)
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed
                    ou_v5_review_packet_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v5_review_packet_failed:{safe_exception_code(exc)}")
            if ou_v5_supreme_reviewer is not None:
                try:
                    v5_supreme = ou_v5_supreme_reviewer(root=root, now=started_at)
                    ou_v5_supreme_review_status = str(v5_supreme.summary.get("status", "BLOCKED"))
                    ou_v5_supreme_review_recommendation = str(
                        v5_supreme.summary.get("recommendation", "")
                    )
                    ou_v5_supreme_review_path = Path(
                        v5_supreme.paths.get("status", ou_v5_supreme_review_path)
                    )
                except Exception as exc:  # noqa: BLE001 - advisory fails closed
                    ou_v5_supreme_review_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v5_supreme_review_failed:{safe_exception_code(exc)}")
            if ou_v5_activation_planner is not None:
                try:
                    v5_activation = ou_v5_activation_planner(
                        root=root,
                        apply=False,
                        now=started_at,
                    )
                    ou_v5_activation_status = str(v5_activation.summary.get("status", "BLOCKED"))
                    ou_v5_activation_id = str(v5_activation.summary.get("activation_id", ""))
                    ou_v5_comparator_generation = (
                        int(v5_activation.summary.get("comparator_generation", 5) or 5)
                        if ou_v5_activation_status == "APPLIED_RESEARCH_COMPARATOR_ONLY"
                        else 1
                    )
                    ou_v5_activation_path = Path(
                        v5_activation.paths.get("activation_status", ou_v5_activation_path)
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed on lineage
                    ou_v5_activation_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v5_activation_failed:{safe_exception_code(exc)}")
            if ou_v5_proof_refresher is not None:
                try:
                    v5_refresh = ou_v5_proof_refresher(
                        root=root,
                        proof_path=proof_path,
                        now=started_at,
                    )
                    ou_v5_proof_refresh_status = str(v5_refresh.summary.get("status", "BLOCKED"))
                    ou_v5_proofs_refreshed = int(
                        v5_refresh.summary.get("refreshed_ou_rows", 0) or 0
                    )
                    ou_v5_proof_refresh_path = Path(
                        v5_refresh.paths.get("refresh_status", ou_v5_proof_refresh_path)
                    )
                    if ou_v5_proof_refresh_status == "PASS":
                        completed_after = count_completed_exact_mode_proofs(
                            queue_path=queue_path,
                            proof_path=proof_path,
                        )
                        responses_captured_after = count_captured_exact_mode_responses(
                            queue_path=queue_path,
                            proof_path=proof_path,
                        )
                except Exception as exc:  # noqa: BLE001 - retain raw evidence
                    ou_v5_proof_refresh_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v5_proof_refresh_failed:{safe_exception_code(exc)}")

        if ou_v6_holdout_runner is not None and ou_v6_contract.is_file():
            try:
                ou_v6 = _run_external_lane(
                    authorized=ou_v6_external_authorized,
                    target=WIZARD_BACKTEST_ENDPOINT,
                    operation="POST_OU_V6_HOLDOUT_WITH_CREDIT_PREFLIGHT",
                    lane="ou_v6_holdout",
                    call_limit=ou_v6_manifest_call_limit,
                    credit_cost_per_call=OU_V6_CUSTOM_SERIES_CREDIT_COST,
                    callback=lambda: ou_v6_holdout_runner(
                        root=root,
                        now=started_at,
                        execute=ou_v6_external_authorized,
                        api_key=authorized_api_key,
                    ),
                )
                ou_v6_status = str(ou_v6.summary.get("status", "BLOCKED"))
                ou_v6_evaluation_status = str(
                    ou_v6.summary.get("evaluation_status", "NOT_EVALUATED")
                )
                ou_v6_missing_cells_before = int(ou_v6.summary.get("missing_cells_before", 0) or 0)
                ou_v6_required_responses = int(ou_v6.summary.get("required_responses", 0) or 0)
                ou_v6_responses_available = int(ou_v6.summary.get("responses_available", 0) or 0)
                ou_v6_calls_made = int(ou_v6.summary.get("calls_made", 0) or 0)
                ou_v6_responses_captured_this_cycle = int(
                    ou_v6.summary.get("responses_captured", 0) or 0
                )
                raw_ou_v6_credits_before = ou_v6.summary.get("credits_used_before")
                ou_v6_credits_used_before = (
                    int(raw_ou_v6_credits_before) if raw_ou_v6_credits_before is not None else None
                )
                ou_v6_attempted_credits = int(ou_v6.summary.get("credits_attempted", 0) or 0)
                ou_v6_completed_credits = int(ou_v6.summary.get("credits_completed", 0) or 0)
                ou_v6_capture_status_path = Path(
                    ou_v6.paths.get("capture_status", ou_v6_capture_status_path)
                )
                if (
                    ou_v6_responses_captured_this_cycle > ou_v6_calls_made
                    or ou_v6_attempted_credits != ou_v6_calls_made * OU_V6_CUSTOM_SERIES_CREDIT_COST
                    or ou_v6_completed_credits
                    != ou_v6_responses_captured_this_cycle * OU_V6_CUSTOM_SERIES_CREDIT_COST
                ):
                    ou_v6_response_accounting_valid = False
                    blockers.append("ou_v6_response_or_credit_accounting_invalid")
                if ou_v6_external_authorized and ou_v6_missing_cells_before > 0:
                    external_attempt_made_this_cycle = True
                    external_attempt_made = True
                if ou_v6_external_authorized and ou_v6_status in {"BLOCKED", "FAILED"}:
                    blocker = str(ou_v6.summary.get("blocker", ""))
                    errors = [str(value) for value in ou_v6.summary.get("errors", [])]
                    blockers.extend([blocker or "ou_v6_capture_not_complete", *errors])
                if ou_v6_evaluation_status == "FAIL":
                    ou_v6_terminal_failure = True
                    blockers.append("ou_v6_terminal_failure_exact_local_ou_parity_rejected")
            except Exception as exc:  # noqa: BLE001 - preserve frozen v6 evidence
                ou_v6_status = "EVALUATION_FAILED"
                blockers.append(f"ou_v6_holdout_failed:{safe_exception_code(exc)}")

            if ou_v6_supersession_gate_builder is not None:
                try:
                    v6_gate = ou_v6_supersession_gate_builder(root=root)
                    ou_v6_supersession_gate_status = str(v6_gate.summary.get("status", "BLOCKED"))
                    ou_v6_supersession_gate_path = Path(
                        v6_gate.paths.get("supersession_gate", ou_v6_supersession_gate_path)
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed
                    ou_v6_supersession_gate_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v6_supersession_gate_failed:{safe_exception_code(exc)}")
            if ou_v6_review_packet_builder is not None:
                try:
                    v6_packet = ou_v6_review_packet_builder(root=root, now=started_at)
                    ou_v6_review_packet_status = str(v6_packet.summary.get("status", "BLOCKED"))
                    ou_v6_review_packet_id = str(v6_packet.summary.get("review_packet_id", ""))
                    ou_v6_review_packet_path = Path(
                        v6_packet.paths.get("review_packet", ou_v6_review_packet_path)
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed
                    ou_v6_review_packet_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v6_review_packet_failed:{safe_exception_code(exc)}")
            if ou_v6_supreme_reviewer is not None:
                try:
                    v6_supreme = ou_v6_supreme_reviewer(root=root, now=started_at)
                    ou_v6_supreme_review_status = str(v6_supreme.summary.get("status", "BLOCKED"))
                    ou_v6_supreme_review_recommendation = str(
                        v6_supreme.summary.get("recommendation", "")
                    )
                    ou_v6_supreme_review_path = Path(
                        v6_supreme.paths.get("status", ou_v6_supreme_review_path)
                    )
                except Exception as exc:  # noqa: BLE001 - advisory fails closed
                    ou_v6_supreme_review_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v6_supreme_review_failed:{safe_exception_code(exc)}")
            if ou_v6_activation_planner is not None:
                try:
                    v6_activation = ou_v6_activation_planner(
                        root=root,
                        apply=False,
                        now=started_at,
                    )
                    ou_v6_activation_status = str(v6_activation.summary.get("status", "BLOCKED"))
                    ou_v6_activation_id = str(v6_activation.summary.get("activation_id", ""))
                    ou_v6_comparator_generation = (
                        int(v6_activation.summary.get("comparator_generation", 6) or 6)
                        if ou_v6_activation_status == "APPLIED_RESEARCH_COMPARATOR_ONLY"
                        else 1
                    )
                    ou_v6_activation_path = Path(
                        v6_activation.paths.get("activation_status", ou_v6_activation_path)
                    )
                except Exception as exc:  # noqa: BLE001 - fail closed on lineage
                    ou_v6_activation_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v6_activation_failed:{safe_exception_code(exc)}")
            if ou_v6_proof_refresher is not None:
                try:
                    v6_refresh = ou_v6_proof_refresher(
                        root=root,
                        proof_path=proof_path,
                        now=started_at,
                    )
                    ou_v6_proof_refresh_status = str(v6_refresh.summary.get("status", "BLOCKED"))
                    ou_v6_proofs_refreshed = int(
                        v6_refresh.summary.get("refreshed_ou_rows", 0) or 0
                    )
                    ou_v6_proof_refresh_path = Path(
                        v6_refresh.paths.get("refresh_status", ou_v6_proof_refresh_path)
                    )
                    if ou_v6_proof_refresh_status == "PASS":
                        completed_after = count_completed_exact_mode_proofs(
                            queue_path=queue_path,
                            proof_path=proof_path,
                        )
                        responses_captured_after = count_captured_exact_mode_responses(
                            queue_path=queue_path,
                            proof_path=proof_path,
                        )
                except Exception as exc:  # noqa: BLE001 - retain raw evidence
                    ou_v6_proof_refresh_status = "EVALUATION_FAILED"
                    blockers.append(f"ou_v6_proof_refresh_failed:{safe_exception_code(exc)}")

        copula_v2_contract = root / "config" / "wizard_copula_behavioral_parity_v2.json"
        copula_contract = (
            copula_v2_contract
            if copula_v2_contract.is_file()
            else root / "config" / "wizard_copula_behavioral_parity.json"
        )
        if copula_behavioral_runner is not None and copula_contract.is_file():
            try:
                copula_behavioral = _run_external_lane(
                    authorized=copula_external_authorized,
                    target=WIZARD_COPULA_ENDPOINT,
                    operation="POST_COPULA_BEHAVIORAL_WITH_CREDIT_PREFLIGHT",
                    lane="copula_behavioral",
                    call_limit=copula_manifest_call_limit,
                    credit_cost_per_call=COPULA_POST_CREDIT_COST,
                    callback=lambda: copula_behavioral_runner(
                        root=root,
                        now=started_at,
                        execute=copula_external_authorized,
                        api_key=authorized_api_key,
                    ),
                )
                copula_behavioral_status = str(copula_behavioral.summary.get("status", "BLOCKED"))
                copula_behavioral_expected_cells = int(
                    copula_behavioral.summary.get("expected_cells", 0) or 0
                )
                copula_behavioral_cells_passed = int(
                    copula_behavioral.summary.get("behavioral_cells_passed", 0) or 0
                )
                copula_behavioral_provenance_cells = int(
                    copula_behavioral.summary.get("provenance_cells_complete", 0) or 0
                )
                copula_behavioral_endpoint_calls = int(
                    copula_behavioral.summary.get("endpoint_calls_made", 0) or 0
                )
                if copula_behavioral_endpoint_calls > 0:
                    external_attempt_made_this_cycle = True
                    external_attempt_made = True
                copula_behavioral_responses_captured_this_cycle = int(
                    copula_behavioral.summary.get("endpoint_responses_captured_this_cycle", 0) or 0
                )
                if (
                    copula_behavioral_responses_captured_this_cycle
                    > copula_behavioral_endpoint_calls
                ):
                    blockers.append("copula_endpoint_response_accounting_exceeds_calls")
                    copula_behavioral_response_accounting_valid = False
                copula_behavioral_endpoint_calls_required = int(
                    copula_behavioral.summary.get("endpoint_calls_required", 0) or 0
                )
                copula_behavioral_new_capture_cells = int(
                    copula_behavioral.summary.get("newly_captured_cells", 0) or 0
                )
                copula_behavioral_reused_capture_cells = int(
                    copula_behavioral.summary.get("reused_immutable_capture_cells", 0) or 0
                )
                reported_copula_behavioral_parity = bool(
                    copula_behavioral.summary.get("behavioral_parity_proven", False)
                )
                if (
                    reported_copula_behavioral_parity
                    and copula_behavioral_endpoint_calls > 0
                    and copula_behavioral_responses_captured_this_cycle
                    != copula_behavioral_endpoint_calls
                ):
                    blockers.append("copula_behavioral_pass_has_incomplete_response_accounting")
                    copula_behavioral_response_accounting_valid = False
                copula_behavioral_parity_proven = bool(
                    reported_copula_behavioral_parity
                    and copula_behavioral_response_accounting_valid
                )
                copula_formula_parity_proven = bool(
                    copula_behavioral.summary.get("formula_parity_proven", False)
                )
                copula_behavioral_status_path = Path(
                    copula_behavioral.paths.get("status", copula_behavioral_status_path)
                )
                copula_cohort_receipt_id = str(
                    copula_behavioral.summary.get("cohort_receipt_id", "")
                )
                copula_cohort_receipt_path = str(
                    copula_behavioral.summary.get("cohort_receipt_path", "")
                )
                copula_cohort_receipt_sha256 = str(
                    copula_behavioral.summary.get("cohort_receipt_sha256", "")
                )
                copula_cohort_receipt_valid = _valid_copula_cohort_binding(
                    root=root,
                    cohort_id=copula_cohort_receipt_id,
                    cohort_path=copula_cohort_receipt_path,
                    cohort_sha256=copula_cohort_receipt_sha256,
                )
                if copula_behavioral_parity_proven and not copula_cohort_receipt_valid:
                    blockers.append("copula_behavioral_pass_missing_valid_immutable_cohort")
            except Exception as exc:  # noqa: BLE001 - retain proof evidence and fail closed
                copula_behavioral_status = "EVALUATION_FAILED"
                blockers.append(f"copula_behavioral_evaluation_failed:{safe_exception_code(exc)}")

        exact_mode_requests_attempted = sum(
            int(batch.get("external_proof_requests", 0) or 0) for batch in batch_summaries
        )
        exact_mode_responses_captured_this_cycle = sum(
            int(batch.get("selected_responses_captured", 0) or 0) for batch in batch_summaries
        )
        exact_mode_attempted_credits = exact_mode_requests_attempted * CUSTOM_SERIES_CREDIT_COST
        exact_mode_completed_credits = (
            exact_mode_responses_captured_this_cycle * CUSTOM_SERIES_CREDIT_COST
        )
        copula_behavioral_attempted_credits = (
            copula_behavioral_endpoint_calls * COPULA_POST_CREDIT_COST
        )
        copula_behavioral_completed_credits = (
            copula_behavioral_responses_captured_this_cycle * COPULA_POST_CREDIT_COST
        )
        proof_lane_attempted_credits = (
            exact_mode_attempted_credits
            + copula_behavioral_attempted_credits
            + ou_v3_attempted_credits
            + ou_v4_attempted_credits
            + ou_v5_attempted_credits
            + ou_v6_attempted_credits
        )
        proof_lane_completed_credits = (
            exact_mode_completed_credits
            + copula_behavioral_completed_credits
            + ou_v3_completed_credits
            + ou_v4_completed_credits
            + ou_v5_completed_credits
            + ou_v6_completed_credits
        )
        if capture_manifest_enforced:
            realized_by_lane = {
                "exact_mode_backtest": {
                    "calls": exact_mode_requests_attempted,
                    "credits": exact_mode_attempted_credits,
                },
                "ou_v3_holdout": {
                    "calls": ou_v3_calls_made,
                    "credits": ou_v3_attempted_credits,
                },
                "ou_v4_holdout": {
                    "calls": ou_v4_calls_made,
                    "credits": ou_v4_attempted_credits,
                },
                "ou_v5_holdout": {
                    "calls": ou_v5_calls_made,
                    "credits": ou_v5_attempted_credits,
                },
                "ou_v6_holdout": {
                    "calls": ou_v6_calls_made,
                    "credits": ou_v6_attempted_credits,
                },
                "copula_behavioral": {
                    "calls": copula_behavioral_endpoint_calls,
                    "credits": copula_behavioral_attempted_credits,
                },
            }
            for lane, realized in realized_by_lane.items():
                planned = capture_manifest_lane_totals.get(lane, {})
                if int(realized["calls"]) > int(planned.get("calls", 0) or 0):
                    capture_manifest_accounting_valid = False
                    blockers.append(f"capture_manifest_call_ceiling_exceeded:{lane}")
                if int(realized["credits"]) > int(planned.get("credits", 0) or 0):
                    capture_manifest_accounting_valid = False
                    blockers.append(f"capture_manifest_credit_ceiling_exceeded:{lane}")
            if not capture_manifest_accounting_valid:
                status = "BLOCKED_CAPTURE_MANIFEST_ACCOUNTING"
        if capture_manifest_enforced:
            if not capture_manifest_immutable_path:
                capture_reconciliation_status = "BLOCKED_EVIDENCE"
                capture_reconciliation_blockers = ["capture_manifest_immutable_path_missing"]
                capture_reconciliation_valid = False
            else:
                try:
                    reconciliation = capture_manifest_reconciler(
                        root=root,
                        now=started_at,
                        manifest_path=root / capture_manifest_immutable_path,
                        proof_path=proof_path,
                    )
                    capture_reconciliation_status = str(
                        reconciliation.summary.get("status", "BLOCKED_EVIDENCE")
                    )
                    capture_reconciliation_path = Path(
                        reconciliation.paths.get("status", capture_reconciliation_path)
                    )
                    capture_reconciliation_id = str(
                        reconciliation.summary.get("reconciliation_id", "")
                    )
                    capture_reconciliation_immutable_path = str(
                        reconciliation.summary.get("immutable_reconciliation_path", "")
                    )
                    capture_reconciliation_immutable_sha256 = str(
                        reconciliation.summary.get("immutable_reconciliation_sha256", "")
                    )
                    capture_reconciliation_manifest_id = str(
                        reconciliation.summary.get("manifest_id", "")
                    )
                    capture_reconciliation_manifest_path = str(
                        reconciliation.summary.get("manifest_path", "")
                    )
                    capture_reconciliation_manifest_sha256 = str(
                        reconciliation.summary.get("manifest_sha256", "")
                    )
                    capture_reconciliation_required_calls = int(
                        reconciliation.summary.get("required_calls", 0) or 0
                    )
                    capture_reconciliation_completed_calls = int(
                        reconciliation.summary.get("completed_calls", 0) or 0
                    )
                    capture_reconciliation_pending_calls = int(
                        reconciliation.summary.get("pending_calls", 0) or 0
                    )
                    capture_reconciliation_blocked_calls = int(
                        reconciliation.summary.get("blocked_calls", 0) or 0
                    )
                    capture_reconciliation_blockers = [
                        str(value) for value in reconciliation.summary.get("blockers", [])
                    ]
                except (OSError, TypeError, ValueError) as exc:
                    capture_reconciliation_status = "BLOCKED_EVIDENCE"
                    capture_reconciliation_blockers = [
                        f"capture_reconciliation_failed:{safe_exception_code(exc)}"
                    ]
            capture_reconciliation_current_status = capture_reconciliation_status
            current_reconciliation_complete = bool(
                capture_reconciliation_status == "PASS"
                and capture_reconciliation_required_calls > 0
                and capture_reconciliation_completed_calls == capture_reconciliation_required_calls
                and capture_reconciliation_pending_calls == 0
                and capture_reconciliation_blocked_calls == 0
            )
            current_reconciliation_not_required = bool(
                capture_manifest_pending_calls == 0
                and capture_reconciliation_status == "NOT_REQUIRED"
                and capture_reconciliation_required_calls == 0
                and capture_reconciliation_completed_calls == 0
                and capture_reconciliation_pending_calls == 0
                and capture_reconciliation_blocked_calls == 0
            )
            plan_only_reconciliation_pending = bool(
                not external_lanes_authorized_this_cycle
                and capture_reconciliation_status == "PENDING"
                and capture_reconciliation_blocked_calls == 0
            )
            capture_reconciliation_local_state_valid = bool(
                current_reconciliation_complete
                or current_reconciliation_not_required
                or plan_only_reconciliation_pending
            )
            capture_reconciliation_valid = current_reconciliation_complete
            capture_reconciliation_complete = current_reconciliation_complete
            if (
                not capture_reconciliation_complete
                and capture_reconciliation_status == "NOT_REQUIRED"
                and capture_manifest_pending_calls == 0
                and validate_capture_reconciliation_evidence(root=root, evidence=previous).get(
                    "status"
                )
                == "PASS"
            ):
                capture_reconciliation_status = "PASS"
                capture_reconciliation_valid = True
                capture_reconciliation_complete = True
                capture_reconciliation_carried_forward = True
                capture_reconciliation_id = str(previous.get("capture_reconciliation_id", ""))
                capture_reconciliation_immutable_path = str(
                    previous.get("capture_reconciliation_immutable_path", "")
                )
                capture_reconciliation_immutable_sha256 = str(
                    previous.get("capture_reconciliation_immutable_sha256", "")
                )
                capture_reconciliation_manifest_id = str(
                    previous.get("capture_reconciliation_manifest_id", "")
                )
                capture_reconciliation_manifest_path = str(
                    previous.get("capture_reconciliation_manifest_path", "")
                )
                capture_reconciliation_manifest_sha256 = str(
                    previous.get("capture_reconciliation_manifest_sha256", "")
                )
                capture_reconciliation_required_calls = int(
                    previous.get("capture_reconciliation_required_calls", 0) or 0
                )
                capture_reconciliation_completed_calls = int(
                    previous.get("capture_reconciliation_completed_calls", 0) or 0
                )
                capture_reconciliation_pending_calls = int(
                    previous.get("capture_reconciliation_pending_calls", 0) or 0
                )
                capture_reconciliation_blocked_calls = int(
                    previous.get("capture_reconciliation_blocked_calls", 0) or 0
                )
            if not capture_reconciliation_local_state_valid:
                status = "BLOCKED_CAPTURE_MANIFEST_RECONCILIATION"
                blockers.extend(
                    capture_reconciliation_blockers
                    or [
                        (
                            "capture_manifest_responses_not_fully_reconciled:"
                            f"{capture_reconciliation_completed_calls}_of_"
                            f"{capture_reconciliation_required_calls}"
                        )
                    ]
                )
        proof_lane_uncompleted_attempted_credits = max(
            proof_lane_attempted_credits - proof_lane_completed_credits,
            0,
        )
        observed_used_before = (
            _first_observed_credit_usage(batch_summaries)
            if _first_observed_credit_usage(batch_summaries) is not None
            else ou_v3_credits_used_before
            if ou_v3_credits_used_before is not None
            else ou_v4_credits_used_before
            if ou_v4_credits_used_before is not None
            else ou_v5_credits_used_before
            if ou_v5_credits_used_before is not None
            else ou_v6_credits_used_before
        )
        observed_used_after = observed_used_before
        if proof_lane_attempted_credits > 0:
            try:
                if not authorized_api_key:
                    raise ValueError("post_run_credit_usage_api_key_missing")
                after_payload = credits_fetcher(api_key=authorized_api_key)
                after_usage = parse_wizard_credit_usage(after_payload)
                if not after_usage.known or after_usage.used is None:
                    raise ValueError("post_run_credit_usage_unknown")
                observed_used_after = after_usage.used
            except (CryptoWizardsFetchError, OSError, TypeError, ValueError) as exc:
                credit_reconciliation_status = "BLOCKED"
                credit_reconciliation_blocker = (
                    f"vendor_credit_after_fetch_failed:{safe_exception_code(exc)}"
                )
        if execute and credit_reservation_id and not skip_credit_reconciliation_this_cycle:
            try:
                if credit_reconciliation_blocker:
                    raise ValueError(credit_reconciliation_blocker)
                activity_rows = [
                    {
                        "lane": "exact_mode",
                        "external_requests": exact_mode_requests_attempted,
                        "credit_cost": CUSTOM_SERIES_CREDIT_COST,
                        "attempted_credits": exact_mode_attempted_credits,
                        "completed_credits": exact_mode_completed_credits,
                    },
                    {
                        "lane": "copula_behavioral",
                        "external_requests": copula_behavioral_endpoint_calls,
                        "credit_cost": COPULA_POST_CREDIT_COST,
                        "attempted_credits": copula_behavioral_attempted_credits,
                        "completed_credits": copula_behavioral_completed_credits,
                    },
                    {
                        "lane": "ou_v3",
                        "external_requests": ou_v3_calls_made,
                        "credit_cost": OU_V3_CUSTOM_SERIES_CREDIT_COST,
                        "attempted_credits": ou_v3_attempted_credits,
                        "completed_credits": ou_v3_completed_credits,
                    },
                    {
                        "lane": "ou_v4",
                        "external_requests": ou_v4_calls_made,
                        "credit_cost": OU_V4_CUSTOM_SERIES_CREDIT_COST,
                        "attempted_credits": ou_v4_attempted_credits,
                        "completed_credits": ou_v4_completed_credits,
                    },
                    {
                        "lane": "ou_v5",
                        "external_requests": ou_v5_calls_made,
                        "credit_cost": OU_V5_CUSTOM_SERIES_CREDIT_COST,
                        "attempted_credits": ou_v5_attempted_credits,
                        "completed_credits": ou_v5_completed_credits,
                    },
                    {
                        "lane": "ou_v6",
                        "external_requests": ou_v6_calls_made,
                        "credit_cost": OU_V6_CUSTOM_SERIES_CREDIT_COST,
                        "attempted_credits": ou_v6_attempted_credits,
                        "completed_credits": ou_v6_completed_credits,
                    },
                ]
                reconciliation = credit_reconciler(
                    root=root,
                    lane=PROOF_LANE,
                    reservation_id=credit_reservation_id,
                    reconciliation_key=started_at.isoformat(),
                    attempted_credits=proof_lane_attempted_credits,
                    completed_credits=proof_lane_completed_credits,
                    external_requests=(
                        exact_mode_requests_attempted
                        + copula_behavioral_endpoint_calls
                        + ou_v3_calls_made
                        + ou_v4_calls_made
                        + ou_v5_calls_made
                        + ou_v6_calls_made
                    ),
                    observed_used_before=observed_used_before,
                    observed_used_after=observed_used_after,
                    activity_rows=activity_rows,
                    now=started_at,
                )
                credit_reconciliation_status = str(reconciliation.summary.get("status", "BLOCKED"))
                credit_reconciliation_blocker = str(reconciliation.summary.get("blocker", ""))
                credit_reconciliation_id = str(reconciliation.summary.get("reconciliation_id", ""))
                credit_reconciliation_path = (
                    _relative(Path(reconciliation.paths["reconciliation"]), root)
                    if "reconciliation" in reconciliation.paths
                    else ""
                )
            except (OSError, TypeError, ValueError) as exc:
                credit_reconciliation_status = "BLOCKED"
                credit_reconciliation_blocker = credit_reconciliation_blocker or (
                    f"credit_reconciliation_failed:{safe_exception_code(exc)}"
                )
            if credit_reconciliation_status not in {
                "PASS_RECONCILED",
                "REUSED_RECONCILIATION",
            }:
                status = "BLOCKED_CREDIT_RECONCILIATION"
                blockers.append(credit_reconciliation_blocker or "credit_reconciliation_not_proven")

        formula_proofs_expected = max(queue_eligible - copula_behavioral_expected_cells, 0)
        accepted_mode_evidence_cells = completed_after + (
            copula_behavioral_cells_passed
            if copula_behavioral_parity_proven
            and copula_cohort_receipt_valid
            and copula_behavioral_provenance_cells == copula_behavioral_expected_cells
            else 0
        )
        credit_accounting_proven = bool(
            not execute
            or credit_reconciliation_status
            in {
                "PASS_RECONCILED",
                "REUSED_RECONCILIATION",
                "PRIOR_RECONCILIATION_VERIFIED",
            }
        )

        if (
            completed_after > completed_before
            or responses_captured_after > responses_captured_before
        ):
            try:
                parity = parity_refresher(root=root)
                parity_status = str(parity.summary.get("status", "BLOCKED"))
            except Exception as exc:  # noqa: BLE001 - receipt must retain proof progress
                parity_status = "REFRESH_FAILED"
                blockers.append(f"parity_refresh_failed:{safe_exception_code(exc)}")
        elif queue_eligible > 0 and responses_captured_after >= queue_eligible:
            try:
                parity = parity_refresher(root=root)
                parity_status = str(parity.summary.get("status", "BLOCKED"))
            except Exception as exc:  # noqa: BLE001 - recover a completed proof queue
                parity_status = "REFRESH_FAILED"
                blockers.append(f"parity_refresh_failed:{safe_exception_code(exc)}")
        if ou_v6_terminal_failure:
            status = "BLOCKED_OU_V6_TERMINAL_FAILURE"
        elif (
            parity_status == "PASS"
            and queue_eligible > 0
            and responses_captured_after >= queue_eligible
            and accepted_mode_evidence_cells == queue_eligible
            and completed_after < queue_eligible
            and credit_accounting_proven
            and capture_manifest_accounting_valid
            and capture_manifest_continuity_valid
            and capture_reconciliation_complete
        ):
            status = "COMPLETE_ACCEPTED_MODE_EVIDENCE"
        elif (
            parity_status == "PASS"
            and queue_eligible > 0
            and responses_captured_after >= queue_eligible
            and accepted_mode_evidence_cells != queue_eligible
            and credit_accounting_proven
            and capture_manifest_accounting_valid
            and capture_manifest_continuity_valid
            and capture_reconciliation_complete
        ):
            status = "BLOCKED_ACCEPTED_MODE_EVIDENCE_MISMATCH"
            blockers.append(
                "parity_pass_but_accepted_mode_evidence_count_mismatch:"
                f"{accepted_mode_evidence_cells}_of_{queue_eligible}"
            )
    except FileExistsError as exc:
        blockers.append(safe_exception_code(exc))
        status = "BLOCKED_LOCK"
    except Exception as exc:  # noqa: BLE001 - persist a fail-closed scheduler receipt
        blockers.append(f"wizard_proof_scheduler_error:{safe_exception_code(exc)}")
        status = "FAILED"
    finally:
        external_stack.close()
        if lock_acquired:
            lock_path.unlink(missing_ok=True)

    external_attempt_made = bool(prior_external_attempt_today or external_attempt_made_this_cycle)
    configured_proof_request_capacity = max_batches * max_proofs_per_batch
    remaining_vendor_responses = max(queue_eligible - responses_captured_after, 0)
    remaining_custom_series_credits = remaining_vendor_responses * CUSTOM_SERIES_CREDIT_COST
    next_utc_reset_at = started_at.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        days=1
    )
    next_external_attempt_eligible_at = next_utc_reset_at if external_attempt_made else started_at
    next_cohort_capacity_ready = bool(
        input_audit_status == "PASS"
        and credit_budget_status == "PASS"
        and remaining_vendor_responses <= configured_proof_request_capacity
    )
    next_cohort_readiness = (
        "CAPACITY_READY_RUNTIME_PREFLIGHT_REQUIRED"
        if next_cohort_capacity_ready
        else "BLOCKED_CAPACITY_OR_INPUT_CONTRACT"
    )

    receipt = {
        "schema_version": SCHEMA_VERSION,
        **scheduler_run_identity(
            root,
            contract=scheduler_contract("wizard_proof"),
        ),
        "attempt_date_utc": started_at.date().isoformat(),
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "execution_requested": execute,
        "force_requested": force,
        "internal_continuation_only": internal_continuation_only,
        "external_attempt_made": external_attempt_made,
        "external_attempt_made_this_cycle": external_attempt_made_this_cycle,
        "proof_request_slots_selected": proof_request_slots_selected,
        "exact_mode_requests_attempted": exact_mode_requests_attempted,
        "exact_mode_responses_captured_this_cycle": (exact_mode_responses_captured_this_cycle),
        "exact_mode_attempted_credits": exact_mode_attempted_credits,
        "exact_mode_completed_credits": exact_mode_completed_credits,
        "copula_behavioral_attempted_credits": (copula_behavioral_attempted_credits),
        "copula_behavioral_responses_captured_this_cycle": (
            copula_behavioral_responses_captured_this_cycle
        ),
        "copula_behavioral_completed_credits": (copula_behavioral_completed_credits),
        "copula_behavioral_response_accounting_valid": (
            copula_behavioral_response_accounting_valid
        ),
        "ou_v3_holdout_status": ou_v3_status,
        "ou_v3_capture_status_path": _relative(ou_v3_capture_status_path, root),
        "ou_v3_evaluation_status": ou_v3_evaluation_status,
        "ou_v3_missing_cells_before": ou_v3_missing_cells_before,
        "ou_v3_required_responses": ou_v3_required_responses,
        "ou_v3_responses_available": ou_v3_responses_available,
        "ou_v3_calls_made": ou_v3_calls_made,
        "ou_v3_responses_captured_this_cycle": (ou_v3_responses_captured_this_cycle),
        "ou_v3_attempted_credits": ou_v3_attempted_credits,
        "ou_v3_completed_credits": ou_v3_completed_credits,
        "ou_v3_response_accounting_valid": ou_v3_response_accounting_valid,
        "ou_v3_supersession_gate_status": ou_v3_supersession_gate_status,
        "ou_v3_supersession_gate_path": _relative(ou_v3_supersession_gate_path, root),
        "ou_v3_review_packet_status": ou_v3_review_packet_status,
        "ou_v3_review_packet_id": ou_v3_review_packet_id,
        "ou_v3_review_packet_path": _relative(ou_v3_review_packet_path, root),
        "ou_v3_review_required": True,
        "ou_v3_activation_status": ou_v3_activation_status,
        "ou_v3_activation_id": ou_v3_activation_id,
        "ou_v3_activation_path": _relative(ou_v3_activation_path, root),
        "ou_v3_comparator_generation": ou_v3_comparator_generation,
        "ou_v3_activation_automatic": False,
        "ou_v3_proof_refresh_status": ou_v3_proof_refresh_status,
        "ou_v3_proof_refresh_path": _relative(ou_v3_proof_refresh_path, root),
        "ou_v3_proofs_refreshed": ou_v3_proofs_refreshed,
        "ou_v4_holdout_status": ou_v4_status,
        "ou_v4_capture_status_path": _relative(ou_v4_capture_status_path, root),
        "ou_v4_evaluation_status": ou_v4_evaluation_status,
        "ou_v4_missing_cells_before": ou_v4_missing_cells_before,
        "ou_v4_required_responses": ou_v4_required_responses,
        "ou_v4_responses_available": ou_v4_responses_available,
        "ou_v4_calls_made": ou_v4_calls_made,
        "ou_v4_responses_captured_this_cycle": (ou_v4_responses_captured_this_cycle),
        "ou_v4_attempted_credits": ou_v4_attempted_credits,
        "ou_v4_completed_credits": ou_v4_completed_credits,
        "ou_v4_response_accounting_valid": ou_v4_response_accounting_valid,
        "ou_v4_prospectively_registered": ou_v4_contract.is_file(),
        "ou_v4_supersession_gate_status": ou_v4_supersession_gate_status,
        "ou_v4_supersession_gate_path": _relative(ou_v4_supersession_gate_path, root),
        "ou_v4_review_packet_status": ou_v4_review_packet_status,
        "ou_v4_review_packet_id": ou_v4_review_packet_id,
        "ou_v4_review_packet_path": _relative(ou_v4_review_packet_path, root),
        "ou_v4_review_required": True,
        "ou_v4_supreme_review_status": ou_v4_supreme_review_status,
        "ou_v4_supreme_review_recommendation": (ou_v4_supreme_review_recommendation),
        "ou_v4_supreme_review_path": _relative(ou_v4_supreme_review_path, root),
        "ou_v4_activation_status": ou_v4_activation_status,
        "ou_v4_activation_id": ou_v4_activation_id,
        "ou_v4_activation_path": _relative(ou_v4_activation_path, root),
        "ou_v4_comparator_generation": ou_v4_comparator_generation,
        "ou_v4_activation_automatic": False,
        "ou_v4_proof_refresh_status": ou_v4_proof_refresh_status,
        "ou_v4_proof_refresh_path": _relative(ou_v4_proof_refresh_path, root),
        "ou_v4_proofs_refreshed": ou_v4_proofs_refreshed,
        "ou_v4_automatic_activation": False,
        "ou_v4_research_only": True,
        "ou_v5_holdout_status": ou_v5_status,
        "ou_v5_capture_status_path": _relative(ou_v5_capture_status_path, root),
        "ou_v5_evaluation_status": ou_v5_evaluation_status,
        "ou_v5_missing_cells_before": ou_v5_missing_cells_before,
        "ou_v5_required_responses": ou_v5_required_responses,
        "ou_v5_responses_available": ou_v5_responses_available,
        "ou_v5_calls_made": ou_v5_calls_made,
        "ou_v5_responses_captured_this_cycle": ou_v5_responses_captured_this_cycle,
        "ou_v5_attempted_credits": ou_v5_attempted_credits,
        "ou_v5_completed_credits": ou_v5_completed_credits,
        "ou_v5_response_accounting_valid": ou_v5_response_accounting_valid,
        "ou_v5_failure_attribution_status": ou_v5_failure_attribution_status,
        "ou_v5_failure_attribution_id": ou_v5_failure_attribution_id,
        "ou_v5_failure_attribution_path": _relative(ou_v5_failure_attribution_path, root),
        "ou_v5_prospectively_registered": ou_v5_contract.is_file(),
        "ou_v5_supersession_gate_status": ou_v5_supersession_gate_status,
        "ou_v5_supersession_gate_path": _relative(ou_v5_supersession_gate_path, root),
        "ou_v5_review_required": True,
        "ou_v5_review_packet_status": ou_v5_review_packet_status,
        "ou_v5_review_packet_id": ou_v5_review_packet_id,
        "ou_v5_review_packet_path": _relative(ou_v5_review_packet_path, root),
        "ou_v5_supreme_review_status": ou_v5_supreme_review_status,
        "ou_v5_supreme_review_recommendation": ou_v5_supreme_review_recommendation,
        "ou_v5_supreme_review_path": _relative(ou_v5_supreme_review_path, root),
        "ou_v5_activation_status": ou_v5_activation_status,
        "ou_v5_activation_id": ou_v5_activation_id,
        "ou_v5_activation_path": _relative(ou_v5_activation_path, root),
        "ou_v5_comparator_generation": ou_v5_comparator_generation,
        "ou_v5_proof_refresh_status": ou_v5_proof_refresh_status,
        "ou_v5_proof_refresh_path": _relative(ou_v5_proof_refresh_path, root),
        "ou_v5_proofs_refreshed": ou_v5_proofs_refreshed,
        "ou_v5_activation_automatic": False,
        "ou_v5_research_only": True,
        "ou_v6_holdout_status": ou_v6_status,
        "ou_v6_capture_status_path": _relative(ou_v6_capture_status_path, root),
        "ou_v6_evaluation_status": ou_v6_evaluation_status,
        "ou_v6_missing_cells_before": ou_v6_missing_cells_before,
        "ou_v6_required_responses": ou_v6_required_responses,
        "ou_v6_responses_available": ou_v6_responses_available,
        "ou_v6_calls_made": ou_v6_calls_made,
        "ou_v6_responses_captured_this_cycle": ou_v6_responses_captured_this_cycle,
        "ou_v6_credits_used_before": ou_v6_credits_used_before,
        "ou_v6_attempted_credits": ou_v6_attempted_credits,
        "ou_v6_completed_credits": ou_v6_completed_credits,
        "ou_v6_response_accounting_valid": ou_v6_response_accounting_valid,
        "ou_v6_prospectively_registered": ou_v6_contract.is_file(),
        "ou_v6_final_successor_iteration": True,
        "ou_v6_successor_after_failure_allowed": False,
        "ou_v6_terminal_failure": ou_v6_terminal_failure,
        "ou_v6_supersession_gate_status": ou_v6_supersession_gate_status,
        "ou_v6_supersession_gate_path": _relative(ou_v6_supersession_gate_path, root),
        "ou_v6_review_required": True,
        "ou_v6_review_packet_status": ou_v6_review_packet_status,
        "ou_v6_review_packet_id": ou_v6_review_packet_id,
        "ou_v6_review_packet_path": _relative(ou_v6_review_packet_path, root),
        "ou_v6_supreme_review_status": ou_v6_supreme_review_status,
        "ou_v6_supreme_review_recommendation": ou_v6_supreme_review_recommendation,
        "ou_v6_supreme_review_path": _relative(ou_v6_supreme_review_path, root),
        "ou_v6_activation_status": ou_v6_activation_status,
        "ou_v6_activation_id": ou_v6_activation_id,
        "ou_v6_activation_path": _relative(ou_v6_activation_path, root),
        "ou_v6_comparator_generation": ou_v6_comparator_generation,
        "ou_v6_proof_refresh_status": ou_v6_proof_refresh_status,
        "ou_v6_proof_refresh_path": _relative(ou_v6_proof_refresh_path, root),
        "ou_v6_proofs_refreshed": ou_v6_proofs_refreshed,
        "ou_v6_activation_automatic": False,
        "ou_v6_research_only": True,
        "proof_lane_attempted_credits": proof_lane_attempted_credits,
        "proof_lane_completed_credits": proof_lane_completed_credits,
        "proof_lane_uncompleted_attempted_credits": (proof_lane_uncompleted_attempted_credits),
        "vendor_credits_used_before": observed_used_before,
        "vendor_credits_used_after": observed_used_after,
        "vendor_credits_used_delta": (
            observed_used_after - observed_used_before
            if observed_used_after is not None and observed_used_before is not None
            else None
        ),
        "external_effect_authority_status": external_effect_authority_status,
        "external_effect_session_established": external_effect_session is not None,
        "external_effect_reservation_sha256": (
            external_effect_reservation_sha256
        ),
        "external_effect_authorized_request_units": (
            external_effect_session.consumed_requests
            if external_effect_session is not None
            else 0
        ),
        "external_effect_authorized_credit_units": (
            external_effect_session.consumed_credits
            if external_effect_session is not None
            else 0
        ),
        "external_effect_credential_permits": (
            len(external_effect_session.credential_receipts)
            if external_effect_session is not None
            else 0
        ),
        "external_effect_network_permits": (
            len(external_effect_session.network_receipts)
            if external_effect_session is not None
            else 0
        ),
        "external_effect_credit_permits": (
            len(external_effect_session.credit_receipts)
            if external_effect_session is not None
            else 0
        ),
        "external_lanes_authorized_this_cycle": (external_lanes_authorized_this_cycle),
        "capture_manifest_runtime_lane_enforcement": capture_manifest_enforced,
        "exact_mode_external_authorized": exact_mode_external_authorized,
        "ou_v3_external_authorized": ou_v3_external_authorized,
        "ou_v4_external_authorized": ou_v4_external_authorized,
        "ou_v5_external_authorized": ou_v5_external_authorized,
        "ou_v6_external_authorized": ou_v6_external_authorized,
        "copula_external_authorized": copula_external_authorized,
        "exact_mode_manifest_call_limit": exact_mode_manifest_call_limit,
        "ou_v3_manifest_call_limit": ou_v3_manifest_call_limit,
        "ou_v4_manifest_call_limit": ou_v4_manifest_call_limit,
        "ou_v5_manifest_call_limit": ou_v5_manifest_call_limit,
        "ou_v6_manifest_call_limit": ou_v6_manifest_call_limit,
        "copula_manifest_call_limit": copula_manifest_call_limit,
        "max_batches": max_batches,
        "max_proofs_per_batch": max_proofs_per_batch,
        "reserved_credits": DEFAULT_RESERVED_CREDITS,
        "credit_budget_status": credit_budget_status,
        "scheduled_credit_ceiling": scheduled_credit_ceiling,
        "credit_headroom_after_reserve": credit_headroom_after_reserve,
        "proof_lane_credit_ceiling": proof_lane_credit_ceiling,
        "credit_reservation_planned_credits": credit_reservation_planned_credits,
        "capture_manifest_status": capture_manifest_status,
        "capture_manifest_state": capture_manifest_state,
        "capture_manifest_enforced": capture_manifest_enforced,
        "capture_manifest_id": capture_manifest_id,
        "capture_manifest_path": _relative(capture_manifest_path, root),
        "capture_manifest_immutable_path": capture_manifest_immutable_path,
        "capture_manifest_immutable_sha256": capture_manifest_immutable_sha256,
        "capture_manifest_pending_calls": capture_manifest_pending_calls,
        "capture_manifest_planned_credits": capture_manifest_planned_credits,
        "capture_manifest_capture_eligible_now": (capture_manifest_capture_eligible_now),
        "capture_manifest_next_eligible_at": capture_manifest_next_eligible_at,
        "capture_manifest_blockers": capture_manifest_blockers,
        "capture_manifest_lane_totals": capture_manifest_lane_totals,
        "capture_manifest_accounting_valid": capture_manifest_accounting_valid,
        "capture_manifest_candidate_id": capture_manifest_candidate_id,
        "capture_manifest_candidate_immutable_path": (capture_manifest_candidate_immutable_path),
        "capture_manifest_candidate_immutable_sha256": (
            capture_manifest_candidate_immutable_sha256
        ),
        "capture_manifest_candidate_binding_valid": (capture_manifest_candidate_binding_valid),
        "capture_manifest_candidate_source_binding_valid": (
            capture_manifest_candidate_source_binding_valid
        ),
        "capture_manifest_source_receipt_id": capture_manifest_source_receipt_id,
        "capture_manifest_source_receipt_path": capture_manifest_source_receipt_path,
        "capture_manifest_source_receipt_sha256": capture_manifest_source_receipt_sha256,
        "capture_manifest_source_artifacts_sha256": (capture_manifest_source_artifacts_sha256),
        "capture_manifest_previous_id": capture_manifest_previous_id,
        "capture_manifest_previous_immutable_path": (capture_manifest_previous_immutable_path),
        "capture_manifest_previous_immutable_sha256": (capture_manifest_previous_immutable_sha256),
        "capture_manifest_carried_forward": capture_manifest_carried_forward,
        "capture_manifest_drift_detected": capture_manifest_drift_detected,
        "capture_manifest_continuity_status": capture_manifest_continuity_status,
        "capture_manifest_continuity_valid": capture_manifest_continuity_valid,
        "capture_manifest_continuity_blockers": (capture_manifest_continuity_blockers),
        "capture_reconciliation_status": capture_reconciliation_status,
        "capture_reconciliation_path": _relative(capture_reconciliation_path, root),
        "capture_reconciliation_id": capture_reconciliation_id,
        "capture_reconciliation_immutable_path": (capture_reconciliation_immutable_path),
        "capture_reconciliation_immutable_sha256": (capture_reconciliation_immutable_sha256),
        "capture_reconciliation_manifest_id": capture_reconciliation_manifest_id,
        "capture_reconciliation_manifest_path": capture_reconciliation_manifest_path,
        "capture_reconciliation_manifest_sha256": (capture_reconciliation_manifest_sha256),
        "capture_reconciliation_current_status": (capture_reconciliation_current_status),
        "capture_reconciliation_carried_forward": (capture_reconciliation_carried_forward),
        "capture_reconciliation_required_calls": (capture_reconciliation_required_calls),
        "capture_reconciliation_completed_calls": (capture_reconciliation_completed_calls),
        "capture_reconciliation_pending_calls": (capture_reconciliation_pending_calls),
        "capture_reconciliation_blocked_calls": (capture_reconciliation_blocked_calls),
        "capture_reconciliation_blockers": capture_reconciliation_blockers,
        "capture_reconciliation_valid": capture_reconciliation_valid,
        "capture_reconciliation_local_state_valid": (capture_reconciliation_local_state_valid),
        "capture_reconciliation_complete": capture_reconciliation_complete,
        "credit_reservation_status": credit_reservation_status,
        "credit_reservation_blocker": credit_reservation_blocker,
        "credit_reservation_id": credit_reservation_id,
        "credit_reservation_path": credit_reservation_path,
        "effect_reservation_binding_id": (
            external_effect_reservation_binding_id
        ),
        "credit_reconciliation_status": credit_reconciliation_status,
        "credit_reconciliation_blocker": credit_reconciliation_blocker,
        "credit_reconciliation_id": credit_reconciliation_id,
        "credit_reconciliation_path": credit_reconciliation_path,
        "input_audit_status": input_audit_status,
        "input_audit_ready": input_audit_ready,
        "input_audit_blocked": input_audit_blocked,
        "input_audit_retry_safe": input_audit_retry_safe,
        "input_audit_changed_after_vendor_4xx": (input_audit_changed_after_vendor_4xx),
        "input_audit_unchanged_vendor_4xx": input_audit_unchanged_vendor_4xx,
        "wizard_daily_credit_reset_utc": WIZARD_DAILY_CREDIT_RESET_UTC,
        "queue_eligible": queue_eligible,
        "completed_before": completed_before,
        "completed_after": completed_after,
        "new_completed_proofs": max(completed_after - completed_before, 0),
        "responses_captured_before": responses_captured_before,
        "responses_captured_after": responses_captured_after,
        "new_responses_captured": max(responses_captured_after - responses_captured_before, 0),
        "remaining_vendor_responses": remaining_vendor_responses,
        "configured_proof_request_capacity": configured_proof_request_capacity,
        "remaining_custom_series_credits": remaining_custom_series_credits,
        "next_utc_reset_at": next_utc_reset_at.isoformat(),
        "next_external_attempt_eligible_at": (next_external_attempt_eligible_at.isoformat()),
        "next_cohort_capacity_ready": next_cohort_capacity_ready,
        "next_cohort_readiness": next_cohort_readiness,
        "next_cohort_runtime_credit_preflight_required": True,
        "next_cohort_queue_reordering_applied": False,
        "quarantined_failed_batch_continuation_supported": True,
        "batch_summaries": batch_summaries,
        "parity_refresh_status": parity_status,
        "checkpoint_refresh_status": checkpoint_status,
        "checkpoint_pre_rerun_status": checkpoint_pre_rerun_status,
        "checkpoint_post_rerun_status": checkpoint_post_rerun_status,
        "checkpoint_refresh_phase": checkpoint_refresh_phase,
        "registered_rerun_status": registered_rerun_status,
        "registered_rerun_receipt_path": registered_rerun_receipt_path,
        "registered_learning_handoff_status": registered_learning_handoff_status,
        "registered_stage5_research_gate_pass": registered_stage5_research_gate_pass,
        "registered_learning_handoff_blocker": registered_learning_handoff_blocker,
        "dynamic_v2_holdout_status": dynamic_holdout_status,
        "dynamic_v2_holdout_status_path": _relative(dynamic_holdout_status_path, root),
        "dynamic_v2_supersession_eligible": dynamic_holdout_supersession_eligible,
        "dynamic_v2_supersession_automatic": False,
        "dynamic_v2_supersession_gate_status": dynamic_supersession_gate_status,
        "dynamic_v2_supersession_gate_path": _relative(dynamic_supersession_gate_path, root),
        "dynamic_v2_activation_status": dynamic_activation_status,
        "dynamic_v2_activation_id": dynamic_activation_id,
        "dynamic_v2_activation_path": _relative(dynamic_activation_path, root),
        "dynamic_v2_comparator_generation": dynamic_comparator_generation,
        "dynamic_v2_activation_automatic": False,
        "dynamic_v2_proof_refresh_status": dynamic_proof_refresh_status,
        "dynamic_v2_proof_refresh_path": _relative(dynamic_proof_refresh_path, root),
        "dynamic_v2_proofs_refreshed": dynamic_proofs_refreshed,
        "dynamic_v2_review_packet_status": dynamic_review_packet_status,
        "dynamic_v2_review_packet_id": dynamic_review_packet_id,
        "dynamic_v2_review_packet_path": _relative(dynamic_review_packet_path, root),
        "dynamic_v2_review_required": True,
        "copula_behavioral_status": copula_behavioral_status,
        "copula_behavioral_status_path": _relative(copula_behavioral_status_path, root),
        "copula_behavioral_expected_cells": copula_behavioral_expected_cells,
        "copula_behavioral_cells_passed": copula_behavioral_cells_passed,
        "copula_behavioral_provenance_cells": copula_behavioral_provenance_cells,
        "copula_behavioral_endpoint_calls": copula_behavioral_endpoint_calls,
        "copula_behavioral_endpoint_calls_required": (copula_behavioral_endpoint_calls_required),
        "copula_behavioral_new_capture_cells": (copula_behavioral_new_capture_cells),
        "copula_behavioral_reused_capture_cells": (copula_behavioral_reused_capture_cells),
        "copula_behavioral_parity_proven": copula_behavioral_parity_proven,
        "copula_formula_parity_proven": copula_formula_parity_proven,
        "copula_cohort_receipt_id": copula_cohort_receipt_id,
        "copula_cohort_receipt_path": copula_cohort_receipt_path,
        "copula_cohort_receipt_sha256": copula_cohort_receipt_sha256,
        "copula_cohort_receipt_valid": copula_cohort_receipt_valid,
        "formula_proofs_expected": formula_proofs_expected,
        "accepted_mode_evidence_cells": accepted_mode_evidence_cells,
        "stage3_local_evidence_fingerprint": _stage3_local_evidence_fingerprint(root),
        "api_key_check_performed": bool(env_security["check_performed"]),
        "api_key_present": bool(env_security["api_key_present"]),
        "api_key_source": str(env_security["key_source"]),
        "insecure_secret_files": list(env_security["insecure_secret_files"]),
        "blockers": blockers,
        "lock_released": not lock_path.exists(),
        "research_only": True,
        "final_immutable_receipt_required": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "queue_path": _relative(queue_path, root),
        "proof_path": _relative(proof_path, root),
        "credit_budget_path": _relative(credit_budget_path, root),
        "credit_ledger_status_path": "reports/active/wizard_credit_ledger_status.json",
        "input_audit_path": _relative(input_audit_path, root),
    }
    ou_v6_stage3_validation = validate_ou_v6_stage3_evidence(
        root=root,
        evidence=receipt,
    )
    receipt["ou_v6_stage3_validation_status"] = str(
        ou_v6_stage3_validation.get("status", "BLOCKED")
    )
    receipt["ou_v6_stage3_validation_blockers"] = [
        str(value) for value in ou_v6_stage3_validation.get("blockers", [])
    ]
    receipt["receipt_id"] = (
        "wizardproof_" + sha256(_canonical_json(receipt).encode("utf-8")).hexdigest()[:20]
    )
    _atomic_json(receipt, receipt_path)
    pointer_paths = publish_scheduler_status_pointers(
        root=root,
        payload=receipt,
        receipt_path=receipt_path,
    )
    checkpoint_refresh_needed = bool(
        completed_after > completed_before
        or responses_captured_after > responses_captured_before
        or external_attempt_made_this_cycle
        or (
            queue_eligible > 0
            and responses_captured_after >= queue_eligible
            and parity_status == "PASS"
        )
    )
    if checkpoint_refresher is not None and checkpoint_refresh_needed:
        try:
            checkpoint = checkpoint_refresher(root=root, now=started_at)
            checkpoint_status = str(
                checkpoint.summary.get(
                    "operational_acceptance_status",
                    checkpoint.summary.get("status", "BLOCKED"),
                )
            )
            checkpoint_pre_rerun_status = checkpoint_status
            checkpoint_refresh_phase = "PRE_REGISTERED_RERUN"
        except Exception as exc:  # noqa: BLE001 - receipt must retain proof progress
            checkpoint_status = "REFRESH_FAILED"
            checkpoint_pre_rerun_status = checkpoint_status
            checkpoint_refresh_phase = "PRE_REGISTERED_RERUN"
            blockers.append(f"checkpoint_refresh_failed:{safe_exception_code(exc)}")
        receipt["checkpoint_refresh_status"] = checkpoint_status
        receipt["checkpoint_pre_rerun_status"] = checkpoint_pre_rerun_status
        receipt["checkpoint_post_rerun_status"] = checkpoint_post_rerun_status
        receipt["checkpoint_refresh_phase"] = checkpoint_refresh_phase
        receipt["blockers"] = blockers
        receipt.pop("receipt_id", None)
        receipt["receipt_id"] = (
            "wizardproof_" + sha256(_canonical_json(receipt).encode("utf-8")).hexdigest()[:20]
        )
        _atomic_json(receipt, receipt_path)
        pointer_paths = publish_scheduler_status_pointers(
            root=root,
            payload=receipt,
            receipt_path=receipt_path,
        )
    registered_rerun_handoff_ready = bool(
        execute
        and registered_rerun_runner is not None
        and queue_eligible > 0
        and responses_captured_after >= queue_eligible
        and accepted_mode_evidence_cells == queue_eligible
        and parity_status == "PASS"
        and credit_reconciliation_status
        in {
            "PASS_RECONCILED",
            "REUSED_RECONCILIATION",
            "PRIOR_RECONCILIATION_VERIFIED",
        }
        and capture_manifest_accounting_valid
        and capture_manifest_continuity_valid
        and capture_reconciliation_complete
        and receipt["ou_v6_stage3_validation_status"] == "PASS"
        and not ou_v6_terminal_failure
        and checkpoint_status != "REFRESH_FAILED"
    )
    if registered_rerun_handoff_ready:
        try:
            _write_final_immutable_scheduler_receipt(
                root=root,
                active_receipt_path=receipt_path,
                receipt=receipt,
            )
        except (OSError, TypeError, ValueError) as exc:
            registered_rerun_handoff_ready = False
            registered_rerun_status = "BLOCKED_IMMUTABLE_STAGE3_RECEIPT"
            blockers.append(
                f"pre_registered_rerun_immutable_receipt_failed:{safe_exception_code(exc)}"
            )
            receipt["registered_rerun_status"] = registered_rerun_status
            receipt["blockers"] = blockers
            receipt.pop("receipt_id", None)
            receipt["receipt_id"] = (
                "wizardproof_" + sha256(_canonical_json(receipt).encode("utf-8")).hexdigest()[:20]
            )
            _atomic_json(receipt, receipt_path)
            pointer_paths = publish_scheduler_status_pointers(
                root=root,
                payload=receipt,
                receipt_path=receipt_path,
            )
    if registered_rerun_handoff_ready:
        try:
            registered = registered_rerun_runner(
                root=root,
                now=started_at,
                execute=True,
            )
            registered_rerun_status = str(registered.summary.get("status", "BLOCKED"))
            registered_learning_handoff_status = str(
                registered.summary.get("learning_handoff_status", "NOT_EVALUATED")
            )
            registered_stage5_research_gate_pass = bool(
                registered.summary.get("stage5_research_gate_pass", False)
            )
            registered_learning_handoff_blocker = str(
                registered.summary.get("learning_handoff_blocker", "")
            )
            registered_rerun_receipt_path = (
                _relative(Path(registered.paths["execution_receipt"]), root)
                if "execution_receipt" in registered.paths
                else ""
            )
        except Exception as exc:  # noqa: BLE001 - proof evidence must remain durable
            registered_rerun_status = "FAILED"
            blockers.append(f"registered_rerun_failed:{safe_exception_code(exc)}")
        if checkpoint_refresher is not None:
            try:
                checkpoint = checkpoint_refresher(root=root, now=started_at)
                checkpoint_post_rerun_status = str(
                    checkpoint.summary.get(
                        "operational_acceptance_status",
                        checkpoint.summary.get("status", "BLOCKED"),
                    )
                )
                checkpoint_status = checkpoint_post_rerun_status
                checkpoint_refresh_phase = "POST_REGISTERED_RERUN"
            except Exception as exc:  # noqa: BLE001 - retain immutable Stage 4 evidence
                checkpoint_post_rerun_status = "REFRESH_FAILED"
                checkpoint_status = checkpoint_post_rerun_status
                checkpoint_refresh_phase = "POST_REGISTERED_RERUN"
                blockers.append(
                    f"post_registered_rerun_checkpoint_refresh_failed:{safe_exception_code(exc)}"
                )
        receipt["registered_rerun_status"] = registered_rerun_status
        receipt["registered_rerun_receipt_path"] = registered_rerun_receipt_path
        receipt["registered_learning_handoff_status"] = registered_learning_handoff_status
        receipt["registered_stage5_research_gate_pass"] = registered_stage5_research_gate_pass
        receipt["registered_learning_handoff_blocker"] = registered_learning_handoff_blocker
        receipt["checkpoint_refresh_status"] = checkpoint_status
        receipt["checkpoint_pre_rerun_status"] = checkpoint_pre_rerun_status
        receipt["checkpoint_post_rerun_status"] = checkpoint_post_rerun_status
        receipt["checkpoint_refresh_phase"] = checkpoint_refresh_phase
        receipt["blockers"] = blockers
        receipt.pop("receipt_id", None)
        receipt["receipt_id"] = (
            "wizardproof_" + sha256(_canonical_json(receipt).encode("utf-8")).hexdigest()[:20]
        )
        _atomic_json(receipt, receipt_path)
        pointer_paths = publish_scheduler_status_pointers(
            root=root,
            payload=receipt,
            receipt_path=receipt_path,
        )
    immutable_receipt_path = _write_final_immutable_scheduler_receipt(
        root=root,
        active_receipt_path=receipt_path,
        receipt=receipt,
    )
    return CommandResult(
        paths={
            "latest_status": latest_path,
            "execution_status": pointer_paths["execution"],
            "observation_status": pointer_paths["observation"],
            "cycle_receipt": receipt_path,
            "immutable_cycle_receipt": immutable_receipt_path,
            "proof_queue": queue_path,
            "proofs": proof_path,
            "credit_budget": credit_budget_path,
            "capture_manifest": capture_manifest_path,
            "capture_reconciliation": capture_reconciliation_path,
            "input_audit": input_audit_path,
            "copula_behavioral_status": copula_behavioral_status_path,
            "dynamic_v2_activation_status": dynamic_activation_path,
            "dynamic_v2_proof_refresh_status": dynamic_proof_refresh_path,
            "dynamic_v2_review_packet": dynamic_review_packet_path,
            "ou_v3_capture_status": ou_v3_capture_status_path,
            "ou_v3_supersession_gate": ou_v3_supersession_gate_path,
            "ou_v3_review_packet": ou_v3_review_packet_path,
            "ou_v3_activation_status": ou_v3_activation_path,
            "ou_v3_proof_refresh_status": ou_v3_proof_refresh_path,
            "ou_v4_capture_status": ou_v4_capture_status_path,
            "ou_v4_supersession_gate": ou_v4_supersession_gate_path,
            "ou_v4_review_packet": ou_v4_review_packet_path,
            "ou_v4_supreme_review": ou_v4_supreme_review_path,
            "ou_v4_activation_status": ou_v4_activation_path,
            "ou_v4_proof_refresh_status": ou_v4_proof_refresh_path,
            "ou_v5_capture_status": ou_v5_capture_status_path,
            "ou_v5_failure_attribution": ou_v5_failure_attribution_path,
            "ou_v5_supersession_gate": ou_v5_supersession_gate_path,
            "ou_v5_review_packet": ou_v5_review_packet_path,
            "ou_v5_supreme_review": ou_v5_supreme_review_path,
            "ou_v5_activation_status": ou_v5_activation_path,
            "ou_v5_proof_refresh_status": ou_v5_proof_refresh_path,
            "ou_v6_capture_status": ou_v6_capture_status_path,
            "ou_v6_supersession_gate": ou_v6_supersession_gate_path,
            "ou_v6_review_packet": ou_v6_review_packet_path,
            "ou_v6_supreme_review": ou_v6_supreme_review_path,
            "ou_v6_activation_status": ou_v6_activation_path,
            "ou_v6_proof_refresh_status": ou_v6_proof_refresh_path,
        },
        summary=receipt,
    )


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


def _capture_manifest_unresolved(receipt: dict[str, Any]) -> bool:
    """Keep a positive frozen cohort authoritative until reconciliation completes."""

    if not _truthy(receipt.get("capture_manifest_enforced")):
        return False
    if _truthy(receipt.get("capture_reconciliation_complete")):
        return False
    try:
        pending_calls = int(receipt.get("capture_manifest_pending_calls", 0) or 0)
        required_calls = int(receipt.get("capture_reconciliation_required_calls", 0) or 0)
        completed_calls = int(receipt.get("capture_reconciliation_completed_calls", 0) or 0)
    except (TypeError, ValueError):
        return True
    reconciliation_status = str(receipt.get("capture_reconciliation_status", ""))
    return bool(
        pending_calls > 0
        or required_calls > completed_calls
        or reconciliation_status in {"PENDING", "BLOCKED_EVIDENCE"}
    )


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _manifest_lane_calls(lane_totals: dict[str, dict[str, int]], lane: str) -> int:
    try:
        calls = int(lane_totals.get(lane, {}).get("calls", 0) or 0)
    except (AttributeError, TypeError, ValueError):
        return 0
    return max(calls, 0)


def _validate_external_reservation(
    *,
    root: Path,
    reservation_path: str,
    reservation_id: str,
) -> tuple[str, Path]:
    if not reservation_path.strip() or not reservation_id.strip():
        raise ValueError("external_reservation_identity_or_path_missing")
    candidate = Path(reservation_path)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("external_reservation_path_outside_repository") from exc
    if not resolved.is_file():
        raise ValueError("external_reservation_artifact_missing")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("external_reservation_payload_not_object")
    if str(payload.get("reservation_id", "")) != reservation_id:
        raise ValueError("external_reservation_identity_mismatch")
    return sha256(resolved.read_bytes()).hexdigest(), resolved


def _external_request_ceiling(
    *,
    max_batches: int,
    exact_mode_calls: int,
    ou_v3_calls: int,
    ou_v4_calls: int,
    ou_v5_calls: int,
    ou_v6_calls: int,
    copula_calls: int,
) -> int:
    calls = (
        exact_mode_calls,
        ou_v3_calls,
        ou_v4_calls,
        ou_v5_calls,
        ou_v6_calls,
        copula_calls,
    )
    if max_batches <= 0 or any(value < 0 for value in calls):
        raise ValueError("external request ceiling inputs are invalid")
    request_calls = sum(calls)
    exact_credit_preflights = min(max_batches, exact_mode_calls)
    other_credit_preflights = sum(value > 0 for value in calls[1:])
    post_run_credit_reconciliation = 1
    return (
        request_calls
        + exact_credit_preflights
        + other_credit_preflights
        + post_run_credit_reconciliation
    )


def _run_external_lane(
    *,
    authorized: bool,
    target: str,
    operation: str,
    lane: str,
    call_limit: int,
    credit_cost_per_call: int,
    callback: Callable[[], CommandResult],
) -> CommandResult:
    if not authorized:
        return callback()
    if call_limit <= 0 or credit_cost_per_call <= 0:
        raise ValueError("authorized external lane requires positive call and credit limits")
    if current_external_effect_session() is None:
        raise EffectAuthorityError("wizard_external_effect_session_missing")
    _ = (target, operation, lane)
    return callback()


def _run_batches(
    *,
    root: Path,
    started_at: datetime,
    execute: bool,
    max_batches: int,
    max_proofs_per_batch: int,
    max_external_calls: int,
    queue_path: Path,
    queue_eligible: int,
    completed_before: int,
    responses_captured_before: int,
    proof_runner: Callable[..., CommandResult],
    api_key: str | None,
    batch_summaries: list[dict[str, Any]],
    blockers: list[str],
) -> tuple[str, int, int, bool, int]:
    if max_external_calls < 0:
        raise ValueError("max_external_calls must not be negative")
    completed = completed_before
    responses_captured = responses_captured_before
    external_attempt_made = False
    selected_slots = 0
    previous_enable = os.environ.get(PROOF_ENABLE_ENV)
    if execute:
        os.environ[PROOF_ENABLE_ENV] = "true"
    try:
        for batch_number in range(1, max_batches + 1):
            remaining_manifest_calls = max_external_calls - selected_slots
            if remaining_manifest_calls <= 0:
                return (
                    "MANIFEST_CALL_LIMIT_REACHED",
                    completed,
                    responses_captured,
                    external_attempt_made,
                    selected_slots,
                )
            batch_call_limit = min(max_proofs_per_batch, remaining_manifest_calls)

            def run_batch(
                *, batch_call_limit: int = batch_call_limit
            ) -> CommandResult:
                return proof_runner(
                    root=root,
                    max_pairs=batch_call_limit,
                    execute=execute,
                    api_key=api_key,
                    now=started_at,
                    queue_path=queue_path,
                    reserved_credits=DEFAULT_RESERVED_CREDITS,
                )

            if execute and current_external_effect_session() is None:
                raise EffectAuthorityError("wizard_external_effect_session_missing")
            result = run_batch()
            summary = _sanitized_batch_summary(result.summary, batch_number=batch_number)
            batch_summaries.append(summary)
            selected = int(summary["selected"])
            selected_responses_captured = int(summary["selected_responses_captured"])
            external_proof_requests = int(summary["external_proof_requests"])
            selected_request_failed = int(summary["selected_request_failed"])
            selected_failures_quarantined = bool(summary["selected_failures_quarantined"])
            credit_status = str(summary["credit_preflight_status"])
            bounded_counts = (
                selected,
                selected_responses_captured,
                external_proof_requests,
                selected_request_failed,
            )
            if any(value < 0 for value in bounded_counts):
                raise ValueError("proof_batch_summary_contains_negative_count")
            if selected > batch_call_limit:
                raise ValueError("proof_batch_selected_exceeds_requested_cap")
            if selected_responses_captured > selected:
                raise ValueError("proof_batch_responses_exceed_selected")
            if external_proof_requests > selected:
                raise ValueError("proof_batch_external_requests_exceed_selected")
            if selected_request_failed > selected:
                raise ValueError("proof_batch_failures_exceed_selected")
            selected_slots += selected
            if execute and external_proof_requests > 0:
                external_attempt_made = True
            completed = max(completed, int(summary["queue_completed"]))
            responses_captured = max(
                responses_captured,
                int(summary["queue_responses_captured"]),
            )

            if not execute:
                return (
                    "PLANNED",
                    completed,
                    responses_captured,
                    False,
                    selected_slots,
                )
            if selected <= 0:
                if queue_eligible <= completed:
                    status = "COMPLETE_QUEUE"
                elif queue_eligible <= responses_captured:
                    status = "COMPLETE_RESPONSE_CAPTURE_FORMULA_PROOF_INCOMPLETE"
                elif any(
                    batch.get("batch_outcome") == "QUARANTINED_REQUEST_FAILURES_CONTINUE"
                    for batch in batch_summaries
                ):
                    blockers.append("one_or_more_proof_requests_failed_without_response")
                    status = "BLOCKED_REMAINING_REQUESTS_FAILED"
                else:
                    status = "NO_ELIGIBLE_PROOFS"
                return (
                    status,
                    completed,
                    responses_captured,
                    external_attempt_made,
                    selected_slots,
                )
            if credit_status != "PASS":
                blockers.append(str(summary["credit_blocker"]) or "credit_preflight_blocked")
                status = (
                    "PARTIAL_CREDIT_DEFERRED"
                    if completed > completed_before
                    else "DEFERRED_CREDIT_RESET"
                )
                return (
                    status,
                    completed,
                    responses_captured,
                    external_attempt_made,
                    selected_slots,
                )
            if selected_responses_captured <= 0 or responses_captured <= responses_captured_before:
                if (
                    external_proof_requests == selected
                    and selected_request_failed == selected
                    and selected_failures_quarantined
                ):
                    summary["batch_outcome"] = "QUARANTINED_REQUEST_FAILURES_CONTINUE"
                    continue
                blockers.append("proof_batch_made_no_response_capture_progress")
                return (
                    "BLOCKED_NO_PROGRESS",
                    completed,
                    responses_captured,
                    external_attempt_made,
                    selected_slots,
                )
            if queue_eligible > 0 and completed >= queue_eligible:
                return (
                    "COMPLETE_QUEUE",
                    completed,
                    responses_captured,
                    external_attempt_made,
                    selected_slots,
                )
            if queue_eligible > 0 and responses_captured >= queue_eligible:
                return (
                    "COMPLETE_RESPONSE_CAPTURE_FORMULA_PROOF_INCOMPLETE",
                    completed,
                    responses_captured,
                    external_attempt_made,
                    selected_slots,
                )
            completed_before = completed
            responses_captured_before = responses_captured
        blockers.append("bounded_daily_batch_limit_reached")
        return (
            "BOUNDED_BATCH_LIMIT",
            completed,
            responses_captured,
            external_attempt_made,
            selected_slots,
        )
    finally:
        if execute:
            if previous_enable is None:
                os.environ.pop(PROOF_ENABLE_ENV, None)
            else:
                os.environ[PROOF_ENABLE_ENV] = previous_enable


def install_corrective_wizard_proof_launch_agent(
    *,
    root: Path = ROOT,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    system_path: Path | None = None,
) -> dict[str, Any]:
    """Install, but do not bootstrap, the bounded proof scheduler."""

    if interval_seconds < 60:
        raise ValueError("proof scheduler interval must be at least 60 seconds")
    python = scheduler_python_path(root)
    if not python.is_file():
        raise FileNotFoundError(f"scheduler Python missing: {python}")
    logs = ensure_scheduler_log_directory(root)
    ensure_runtime_temp_directory(root)
    payload = _launch_agent_plist(
        root=root,
        python=python,
        logs=logs,
        interval_seconds=interval_seconds,
    )
    publication = write_launch_agent_plist(
        root=root,
        label=LAUNCH_AGENT_LABEL,
        payload=payload,
        system_path=system_path,
    )
    return {
        "label": LAUNCH_AGENT_LABEL,
        "plist": publication["workspace_plist"],
        **publication,
        "interval_seconds": interval_seconds,
        "reserved_credits": DEFAULT_RESERVED_CREDITS,
        "research_only": True,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _insecure_secret_files(root: Path) -> list[str]:
    """Check private-file modes without reading credential values."""

    insecure: list[str] = []
    for name in (".env.local", ".env"):
        path = root / name
        if path.is_file() and path.stat().st_mode & 0o077:
            insecure.append(_relative(path, root))
    return insecure


def _deferred_credential_reader(root: Path, key: str) -> str | None:
    value = os.getenv(key, "").strip()
    if value:
        return value
    for name in (".env.local", ".env"):
        path = root / name
        loaded = load_selected_env_keys(
            path,
            allowed_keys={key},
            override=False,
        )
        if key in loaded:
            return os.getenv(key, "").strip() or None
    return None


def _file_declares_key(path: Path, key: str) -> bool:
    if not path.is_file():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.split("=", 1)[0].strip() == key:
            return True
    return False


def _already_attempted_today(previous: dict[str, Any], now: datetime) -> bool:
    return bool(
        previous.get("attempt_date_utc") == now.date().isoformat()
        and previous.get("external_attempt_made") is True
    )


def _validate_prior_daily_credit_evidence(
    *, root: Path, timestamp: datetime, previous: dict[str, Any]
) -> dict[str, Any]:
    day = timestamp.date().isoformat()
    reservation_path = (
        root
        / "data"
        / "research"
        / "wizard_credit_ledger"
        / day
        / "reservations"
        / f"{PROOF_LANE}.json"
    )
    reservation = _read_json(reservation_path)
    reservation_id = str(reservation.get("reservation_id", ""))
    reconciliation_id = str(previous.get("credit_reconciliation_id", ""))
    reconciliation_path = str(previous.get("credit_reconciliation_path", ""))
    if not reservation_id or not reconciliation_id or not reconciliation_path:
        return {
            "status": "BLOCKED",
            "blocker": "prior_daily_credit_evidence_identity_missing",
        }
    return validate_wizard_credit_lane_evidence(
        root=root,
        lane=PROOF_LANE,
        credit_date_utc=day,
        reservation_id=reservation_id,
        reconciliation_id=reconciliation_id,
        reservation_path=_relative(reservation_path, root),
        reconciliation_path=reconciliation_path,
    )


def _valid_copula_cohort_binding(
    *, root: Path, cohort_id: str, cohort_path: str, cohort_sha256: str
) -> bool:
    if not cohort_id or not cohort_path or not cohort_sha256:
        return False
    relative = Path(cohort_path)
    if relative.is_absolute():
        return False
    path = root / relative
    evidence_root = root / "data" / "research" / "wizard_copula_behavioral_cohorts"
    try:
        path.resolve().relative_to(evidence_root.resolve())
    except ValueError:
        return False
    return bool(
        path.is_file()
        and path.stem == cohort_id
        and sha256(path.read_bytes()).hexdigest() == cohort_sha256
    )


def _sanitized_batch_summary(summary: dict[str, Any], *, batch_number: int) -> dict[str, Any]:
    return {
        "batch_number": batch_number,
        "queue_eligible": int(summary.get("queue_eligible", 0) or 0),
        "selected": int(summary.get("selected", 0) or 0),
        "completed": int(summary.get("completed", 0) or 0),
        "queue_completed": int(summary.get("queue_completed", summary.get("completed", 0)) or 0),
        "queue_responses_captured": int(
            summary.get(
                "queue_responses_captured",
                summary.get("queue_completed", summary.get("completed", 0)),
            )
            or 0
        ),
        "selected_completed": int(summary.get("selected_completed", 0) or 0),
        "selected_responses_captured": int(
            summary.get(
                "selected_responses_captured",
                summary.get("selected_completed", 0),
            )
            or 0
        ),
        "external_proof_requests": int(summary.get("external_proof_requests", 0) or 0),
        "selected_credit_blocked": int(summary.get("selected_credit_blocked", 0) or 0),
        "selected_request_failed": int(summary.get("selected_request_failed", 0) or 0),
        "selected_failures_quarantined": bool(summary.get("selected_failures_quarantined", False)),
        "execution_enabled": bool(summary.get("execution_enabled", False)),
        "credit_preflight_status": str(summary.get("credit_preflight_status", "")),
        "credits_used_before": summary.get("credits_used_before", ""),
        "credits_remaining_before": summary.get("credits_remaining_before", ""),
        "reserved_credits": int(summary.get("reserved_credits", DEFAULT_RESERVED_CREDITS)),
        "credit_blocker": str(
            summary.get("credit_preflight_blocker", summary.get("credit_blocker", ""))
        ),
    }


def _first_observed_credit_usage(batch_summaries: list[dict[str, Any]]) -> int | None:
    for batch in batch_summaries:
        value = batch.get("credits_used_before")
        if isinstance(value, bool) or value in {None, ""}:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            return parsed
    return None


def _launch_agent_plist(*, root: Path, python: Path, logs: Path, interval_seconds: int) -> str:
    if python != scheduler_python_path(root):
        raise ValueError("proof scheduler interpreter must use canonical runtime")
    if logs != scheduler_log_directory(root):
        raise ValueError("proof scheduler logs must use canonical runtime")
    return scheduler_launch_agent_plist(
        root,
        contract=scheduler_contract("wizard_proof"),
        interval_seconds=interval_seconds,
    )


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        publication_scope="wizard_external_research",
    )


def _write_final_immutable_scheduler_receipt(
    *,
    root: Path,
    active_receipt_path: Path,
    receipt: dict[str, Any],
) -> Path:
    receipt_id = str(receipt.get("receipt_id", ""))
    unsigned = dict(receipt)
    unsigned.pop("receipt_id", None)
    expected_id = (
        "wizardproof_" + sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()[:20]
    )
    if receipt_id != expected_id:
        raise ValueError("final scheduler receipt id does not match its content")
    if receipt.get("final_immutable_receipt_required") is not True:
        raise ValueError("final scheduler receipt immutable requirement is missing")
    encoded = active_receipt_path.read_bytes()
    expected_encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if encoded != expected_encoded:
        raise ValueError("active scheduler receipt is not the finalized payload")
    immutable = (
        root / "data" / "research" / "wizard_proof_scheduler_receipts" / f"{receipt_id}.json"
    )
    write_immutable_json(
        immutable,
        receipt,
        publication_scope="wizard_external_research",
    )
    if immutable.read_bytes() != encoded:
        raise ValueError("immutable scheduler receipt verification failed")
    return immutable


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--internal-continuation-only", action="store_true")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.install:
        try:
            with governed_evidence_write_lock(
                ROOT, blocking=False, scope="scheduler_config"
            ):
                result: Any = install_corrective_wizard_proof_launch_agent()
        except (GovernedEvidenceMaintenanceActive, GovernedEvidenceLockBusy) as exc:
            result = {
                "summary": {
                    "status": "DEFERRED_PHASE00_OR_GOVERNED_LOCK",
                    "blockers": [safe_exception_code(exc)],
                    "live_trading_authorized": False,
                },
                "paths": {},
            }
    else:
        supervised = supervise_scheduler_run(
            root=ROOT,
            contract_key="wizard_proof",
            publication_scope="wizard_external_research",
            callback=lambda: run_corrective_wizard_proof_cycle(
                execute=args.execute,
                force=args.force,
                internal_continuation_only=args.internal_continuation_only,
            ),
            require_launchd_provenance=True,
        )
        result = {
            "summary": supervised.result_summary,
            "paths": supervised.result_paths,
            "terminal_receipt": supervised.terminal_receipt,
            "terminal_paths": {
                key: str(value) for key, value in supervised.terminal_paths.items()
            },
        }
        print(json.dumps(result, indent=2, default=str))
        if supervised.exit_code:
            raise SystemExit(supervised.exit_code)
        return
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
