"""Fail-closed handoff from registered evidence gaps to a scheduled family rerun."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_data_evidence import (
    validate_pair_cost_bundle_artifacts,
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
from quant_platform.wizard_credit_ledger import (
    PROOF_LANE,
    validate_wizard_credit_lane_evidence,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_registered_rerun.v1"
READY_SCHEMA_VERSION = "thewiz.corrective_registered_rerun_ready.v2"
CONCLUSION_SCHEMA_VERSION = "thewiz.corrective_registered_rerun_conclusion.v2"
SOURCE_FAMILY_SCHEMA_VERSION = "thewiz.registered_rerun_source_family.v1"
FAMILY_PREFLIGHT_SCHEMA_VERSION = "thewiz.registered_rerun_family_preflight.v1"
LEGACY_REGISTERED_CONTRACT_MATERIAL_FIELDS = (
    "schema_version",
    "registered_candidates",
    "acceptance_policy_id",
    "holdout_policy_id",
    "discovery_policy_sha256",
    "source_family_sha256",
    "source_family_rows",
)
GENERATION_REGISTERED_CONTRACT_MATERIAL_FIELDS = (
    *LEGACY_REGISTERED_CONTRACT_MATERIAL_FIELDS,
    "generation",
    "prior_contract_id",
)
REGISTERED_CONTRACT_MATERIAL_FIELDS = (
    *GENERATION_REGISTERED_CONTRACT_MATERIAL_FIELDS,
    "candidate_selection_path",
    "candidate_selection_sha256",
    "candidate_selection_status",
)


def validate_registered_rerun_contract_identity(contract: dict[str, Any]) -> None:
    """Recompute production contract IDs so rehashed edits still fail closed."""

    contract_id = _text(contract.get("contract_id"))
    if not contract_id.startswith("registeredrerun_"):
        return
    selection_fields = {
        "candidate_selection_path",
        "candidate_selection_sha256",
        "candidate_selection_status",
    }
    generation_fields = {"generation", "prior_contract_id"}
    if selection_fields.intersection(contract):
        fields = REGISTERED_CONTRACT_MATERIAL_FIELDS
    elif generation_fields.intersection(contract):
        fields = GENERATION_REGISTERED_CONTRACT_MATERIAL_FIELDS
    else:
        fields = LEGACY_REGISTERED_CONTRACT_MATERIAL_FIELDS
    material = {field: contract.get(field) for field in fields}
    expected_id = "registeredrerun_" + sha256(
        _canonical_json(material).encode("utf-8")
    ).hexdigest()[:20]
    if contract_id != expected_id:
        raise ValueError("registered rerun immutable contract identity mismatch")


def build_registered_rerun_gate(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    """Freeze the registered hypotheses and evaluate research-rerun readiness."""

    as_of = _as_utc(now)
    active = root / "reports" / "active"
    batch_path = active / "current_hypothesis_batch.csv"
    queue_path = active / "walkforward_near_miss_queue.csv"
    ledger_path = root / "data" / "research" / "hypothesis_ledger.jsonl"
    ledger_audit_path = active / "hypothesis_ledger_audit.csv"
    acceptance_path = active / "acceptance_policy_receipt.json"
    holdout_path = active / "holdout_policy_receipt.json"
    discovery_policy_path = root / "config" / "wizard_discovery_policy.json"
    matrix_path = active / "current_wizard_hyperliquid_experiment_matrix.csv"
    chain_path = active / "current_wizard_hyperliquid_chain_validation_manifest.json"
    cost_path = root / "data" / "processed" / "hyperliquid_pair_cost_models.csv"
    parity_path = active / "wizard_mode_parity.csv"
    proof_scheduler_path = active / "corrective_wizard_proof_scheduler_status.json"
    l2_status_path = active / "corrective_l2_capture_status.json"
    final_survivor_path = active / "final_1x_survivor_receipt.json"
    failure_path = active / "current_wizard_hyperliquid_failure_attribution.csv"
    failure_manifest_path = (
        active / "current_wizard_hyperliquid_failure_attribution_manifest.json"
    )

    batch = _read_csv(batch_path)
    queue = _read_csv(queue_path)
    ledger = _read_jsonl(ledger_path)
    ledger_audit = _read_csv(ledger_audit_path)
    acceptance = _read_json(acceptance_path)
    holdout = _read_json(holdout_path)
    matrix = _read_csv(matrix_path)
    chain = _read_json(chain_path)
    costs = _read_csv(cost_path)
    parity = _read_csv(parity_path)
    proof_scheduler = _read_json(proof_scheduler_path)
    l2_status = _read_json(l2_status_path)
    final_survivor = _read_json(final_survivor_path)
    failure = _read_csv(failure_path)
    failure_manifest = _read_json(failure_manifest_path)
    family_preflight = build_registered_family_handoff_preflight(root=root, now=as_of)
    candidate_selection_path = Path(family_preflight.paths["preflight"])
    candidate_selection = _read_csv(candidate_selection_path)
    try:
        expected_hypotheses = int(family_preflight.summary.get("hypotheses", -1))
    except (TypeError, ValueError):
        expected_hypotheses = -1
    selected_semantic_ids = candidate_selection.get(
        "semantic_hypothesis_id", pd.Series(dtype=str)
    ).map(_text)
    candidate_selection_valid = bool(
        expected_hypotheses >= 0
        and len(candidate_selection) == expected_hypotheses
        and selected_semantic_ids.ne("").all()
        and not selected_semantic_ids.duplicated().any()
        and not candidate_selection.get(
            "testnet_order_authority", pd.Series(False, index=candidate_selection.index)
        )
        .map(_truthy)
        .any()
        and not candidate_selection.get(
            "live_trading_authorized", pd.Series(False, index=candidate_selection.index)
        )
        .map(_truthy)
        .any()
    )
    if not candidate_selection_valid:
        raise ValueError("registered rerun candidate selection artifact invalid")

    contract_path = active / "registered_research_rerun_contract.json"
    contract = _load_or_create_contract(
        root=root,
        as_of=as_of,
        active_contract_path=contract_path,
        batch=candidate_selection,
        queue=queue,
        ledger=ledger,
        acceptance=acceptance,
        holdout=holdout,
        discovery_policy_path=discovery_policy_path,
        matrix_path=matrix_path,
        chain=chain,
        generation_rollover_ready=family_preflight.summary["status"] == "PASS",
        candidate_selection_path=candidate_selection_path,
        candidate_selection_status=str(family_preflight.summary["status"]),
    )
    frozen_matrix_path, frozen_source_receipt_path = resolve_registered_source_family(
        root=root,
        contract=contract,
        current_matrix_path=matrix_path,
    )
    matrix = _read_csv(frozen_matrix_path)
    candidates = contract.get("registered_candidates", [])
    if not isinstance(candidates, list):
        candidates = []

    accepted_mode_evidence = _accepted_mode_evidence_mask(parity)
    parity_complete = bool(
        len(parity) == 14 and accepted_mode_evidence.all()
    )
    proof_queue_complete = _proof_scheduler_mixed_evidence_complete(
        root=root,
        proof_scheduler=proof_scheduler,
    )
    capture_reconciliation = validate_capture_reconciliation_evidence(
        root=root, evidence=proof_scheduler
    )
    stage_two_eligible = int(
        l2_status.get(
            "stage_two_pair_cost_eligible",
            l2_status.get("strict_pair_cost_eligible", 0),
        )
        or 0
    )
    stage_two_ready = int(
        l2_status.get(
            "stage_two_pair_cost_ready",
            l2_status.get("strict_pair_cost_ready", 0),
        )
        or 0
    )
    stage_two_acceptance = str(
        l2_status.get(
            "stage_two_pair_cost_acceptance_status",
            l2_status.get("strict_pair_cost_acceptance_status", ""),
        )
    )
    stage_two_complete = bool(
        stage_two_eligible > 0
        and stage_two_ready == stage_two_eligible
        and stage_two_acceptance == "PASS"
    )
    policy_ready = bool(
        acceptance.get("status") == "PASS"
        and holdout.get("status") == "PASS"
        and acceptance.get("policy_id") == contract.get("acceptance_policy_id")
        and holdout.get("policy_id") == contract.get("holdout_policy_id")
        and _file_hash(discovery_policy_path)
        == contract.get("discovery_policy_sha256")
    )
    chain_integrity = bool(
        _file_hash(frozen_matrix_path) == contract.get("source_family_sha256")
        and len(matrix) == int(contract.get("source_family_rows", -1))
        and "experiment_id" in matrix
        and not matrix["experiment_id"].astype(str).duplicated().any()
    )

    rows = [
        _candidate_gate_row(
            candidate=candidate,
            as_of=as_of,
            queue=queue,
            ledger=ledger,
            ledger_audit=ledger_audit,
            matrix=matrix,
            costs=costs,
            parity=parity,
            parity_complete=parity_complete,
            proof_queue_complete=proof_queue_complete,
            stage_two_complete=stage_two_complete,
            policy_ready=policy_ready,
            chain_integrity=chain_integrity,
        )
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    gate = pd.DataFrame(rows)
    all_candidates_ready = bool(
        not gate.empty
        and gate.get("pre_rerun_gate_ready", pd.Series(False, index=gate.index))
        .map(_truthy)
        .all()
    )
    ready_receipt_path = (
        root
        / "data"
        / "research"
        / "registered_rerun_ready"
        / f"{contract.get('contract_id', 'missing')}.json"
    )
    ready_receipt = _read_json(ready_receipt_path)
    scheduler_cycle_path, _ = _proof_scheduler_cycle_receipt(
        root=root,
        proof_scheduler=proof_scheduler,
    )
    if all_candidates_ready and not ready_receipt:
        if scheduler_cycle_path is None:
            raise ValueError(
                "registered rerun ready receipt requires a bound proof cycle"
            )
        ready_core = {
            "schema_version": READY_SCHEMA_VERSION,
            "contract_id": contract.get("contract_id", ""),
            "evidence_ready_at_utc": as_of.isoformat(),
            "registered_hypotheses": len(gate),
            "parity_cells_proven": int(
                accepted_mode_evidence.sum()
            ),
            "proof_cells_completed": int(
                proof_scheduler.get("accepted_mode_evidence_cells", 0) or 0
            ),
            "formula_proof_cells_completed": int(
                proof_scheduler.get("completed_after", 0) or 0
            ),
            "formula_proof_cells_expected": int(
                proof_scheduler.get("formula_proofs_expected", 0) or 0
            ),
            "copula_behavioral_cells_completed": int(
                proof_scheduler.get("copula_behavioral_cells_passed", 0) or 0
            ),
            "copula_behavioral_cells_expected": int(
                proof_scheduler.get("copula_behavioral_expected_cells", 0) or 0
            ),
            "accepted_mode_evidence_cells": int(
                proof_scheduler.get("accepted_mode_evidence_cells", 0) or 0
            ),
            "pair_cost_model_ids": sorted(
                set(gate.get("cost_model_id", pd.Series(dtype=str)).astype(str)) - {""}
            ),
            "parity_sha256": _file_hash(parity_path),
            "proof_scheduler_receipt_id": proof_scheduler.get("receipt_id", ""),
            "proof_scheduler_cycle_receipt_path": _relative(
                scheduler_cycle_path, root
            ),
            "proof_scheduler_cycle_receipt_sha256": _file_hash(
                scheduler_cycle_path
            ),
            "copula_cohort_receipt_id": _text(
                proof_scheduler.get("copula_cohort_receipt_id")
            ),
            "copula_cohort_receipt_path": _text(
                proof_scheduler.get("copula_cohort_receipt_path")
            ),
            "copula_cohort_receipt_sha256": _text(
                proof_scheduler.get("copula_cohort_receipt_sha256")
            ),
            "capture_reconciliation_id": _text(
                proof_scheduler.get("capture_reconciliation_id")
            ),
            "capture_reconciliation_immutable_path": _text(
                proof_scheduler.get("capture_reconciliation_immutable_path")
            ),
            "capture_reconciliation_immutable_sha256": _text(
                proof_scheduler.get("capture_reconciliation_immutable_sha256")
            ),
            "capture_reconciliation_manifest_id": _text(
                proof_scheduler.get("capture_reconciliation_manifest_id")
            ),
            "capture_reconciliation_manifest_path": _text(
                proof_scheduler.get("capture_reconciliation_manifest_path")
            ),
            "capture_reconciliation_manifest_sha256": _text(
                proof_scheduler.get("capture_reconciliation_manifest_sha256")
            ),
            "capture_reconciliation_required_calls": int(
                proof_scheduler.get("capture_reconciliation_required_calls", 0) or 0
            ),
            "capture_reconciliation_completed_calls": int(
                proof_scheduler.get("capture_reconciliation_completed_calls", 0) or 0
            ),
            "scheduled_research_rerun_authorized": True,
            "promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        ready_receipt = {
            **ready_core,
            "ready_receipt_id": "registeredready_"
            + sha256(_canonical_json(ready_core).encode("utf-8")).hexdigest()[:20],
        }
        _write_or_validate_immutable_json(ready_receipt, ready_receipt_path)
    if ready_receipt:
        _validate_ready_receipt_lineage(
            root=root,
            ready_receipt=ready_receipt,
            contract=contract,
            parity_path=parity_path,
        )

    ready_at = pd.to_datetime(
        ready_receipt.get("evidence_ready_at_utc"), utc=True, errors="coerce"
    )
    chain_at = pd.to_datetime(chain.get("as_of"), utc=True, errors="coerce")
    chain_after_ready = bool(
        pd.notna(ready_at) and pd.notna(chain_at) and chain_at > ready_at
    )
    failure_at = pd.to_datetime(
        failure_manifest.get("as_of"), utc=True, errors="coerce"
    )
    failure_identity = _text(failure_manifest.get("failure_attribution_id"))
    failure_integrity = bool(
        pd.notna(ready_at)
        and pd.notna(failure_at)
        and failure_at > ready_at
        and len(failure) == len(matrix)
        and failure.get("experiment_id", pd.Series(dtype=str)).astype(str).nunique()
        == len(matrix)
        and int(failure_manifest.get("experiments_accounted", 0) or 0)
        == len(matrix)
        and int(failure_manifest.get("unique_experiment_ids", 0) or 0)
        == len(matrix)
        and _truthy(failure_manifest.get("experiment_status_accounted"))
        and _text(chain.get("stage_identities", {}).get("failure_attribution"))
        == failure_identity
        and _text(chain.get("input_hashes", {}).get("failure_attribution_manifest"))
        == _file_hash(failure_manifest_path)
        and _text(failure_manifest.get("input_hashes", {}).get("matrix"))
        == _file_hash(matrix_path)
    )
    registered_ids = {
        str(candidate.get("semantic_hypothesis_id", "")) for candidate in candidates
    } - {""}
    registered_outcomes = _resolve_registered_outcomes(
        candidates=candidates,
        queue=queue,
        final_survivor=final_survivor,
        failure_attribution=(failure if failure_integrity else pd.DataFrame()),
    )
    rerun_results_accounted = bool(
        ready_receipt
        and chain_after_ready
        and chain_integrity
        and failure_integrity
        and registered_ids
        and registered_ids == set(registered_outcomes)
        and all(outcome != "UNACCOUNTED" for outcome in registered_outcomes.values())
    )
    accepted_registered = sum(
        outcome == "ACCEPTED_SURVIVOR" for outcome in registered_outcomes.values()
    )
    rejected_registered = sum(
        outcome == "REJECTED_BY_FROZEN_GATES"
        for outcome in registered_outcomes.values()
    )
    conclusion_status = "INCOMPLETE"
    if rerun_results_accounted:
        conclusion_status = (
            "ACCEPTED_REGISTERED_SURVIVORS"
            if accepted_registered > 0
            else "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS"
        )
    conclusion_path = (
        root
        / "data"
        / "research"
        / "registered_rerun_conclusions"
        / f"{contract.get('contract_id', 'missing')}.json"
    )
    conclusion_receipt = _read_json(conclusion_path)
    if rerun_results_accounted and not conclusion_receipt:
        conclusion_core = {
            "schema_version": CONCLUSION_SCHEMA_VERSION,
            "contract_id": contract.get("contract_id", ""),
            "concluded_at_utc": as_of.isoformat(),
            "evidence_ready_at_utc": ready_receipt.get(
                "evidence_ready_at_utc", ""
            ),
            "result_chain_as_of_utc": chain.get("as_of", ""),
            "result_chain_sha256": _file_hash(chain_path),
            "final_survivor_receipt_sha256": _file_hash(final_survivor_path),
            "failure_attribution_sha256": _file_hash(failure_path),
            "failure_attribution_manifest_sha256": _file_hash(
                failure_manifest_path
            ),
            "registered_hypotheses": len(registered_ids),
            "accepted_registered_hypotheses": accepted_registered,
            "rejected_registered_hypotheses": rejected_registered,
            "outcomes": registered_outcomes,
            "conclusion_status": conclusion_status,
            "thresholds_changed_after_contract": False,
            "promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        conclusion_receipt = {
            **conclusion_core,
            "conclusion_id": "registeredconclusion_"
            + sha256(
                _canonical_json(conclusion_core).encode("utf-8")
            ).hexdigest()[:20],
        }
        _write_or_validate_immutable_json(conclusion_receipt, conclusion_path)
    elif conclusion_receipt:
        if (
            not _contract_conclusively_accounted(contract, conclusion_receipt)
            or conclusion_receipt.get("contract_id") != contract.get("contract_id")
            or conclusion_receipt.get("outcomes") != registered_outcomes
            or conclusion_receipt.get("conclusion_status") != conclusion_status
        ):
            raise ValueError("registered rerun conclusion immutable receipt mismatch")
    if not gate.empty:
        gate["registered_rerun_outcome"] = gate["semantic_hypothesis_id"].map(
            registered_outcomes
        ).fillna("UNACCOUNTED")
        gate["evidence_ready_receipt_present"] = bool(ready_receipt)
        gate["current_chain_after_evidence_ready"] = chain_after_ready
        gate["registered_rerun_result_accounted"] = rerun_results_accounted
        gate["promotion_authority"] = False
        gate["testnet_order_authority"] = False
        gate["live_trading_authorized"] = False

    contract_candidate_ids = {
        _text(candidate.get("semantic_hypothesis_id"))
        for candidate in contract.get("registered_candidates", [])
        if isinstance(candidate, dict)
    } - {""}
    pending_candidate_ids = set(
        batch.get("semantic_hypothesis_id", pd.Series(dtype=str)).astype(str)
    ) - {""}
    pending_family_rollover_required = bool(
        pending_candidate_ids and pending_candidate_ids != contract_candidate_ids
    )

    if (
        rerun_results_accounted
        and pending_family_rollover_required
        and family_preflight.summary["status"] != "PASS"
    ):
        status = "BLOCKED_PENDING_FAMILY_PREFLIGHT"
        next_action = "repair_pending_family_non_vendor_preflight"
    elif rerun_results_accounted and pending_family_rollover_required:
        status = "BLOCKED_PENDING_FAMILY_ROLLOVER"
        next_action = "repair_pending_family_contract_rollover"
    elif rerun_results_accounted:
        status = "PASS_REGISTERED_RERUN_ACCOUNTED"
        next_action = "evaluate_registered_results_without_threshold_changes"
    elif ready_receipt:
        status = "READY_FOR_NEXT_SCHEDULED_RESEARCH_RERUN"
        next_action = "registered_rerun_executor_runs_frozen_downstream_family"
    elif not parity_complete or not proof_queue_complete:
        status = "BLOCKED_VENDOR_PARITY"
        next_action = _next_vendor_parity_action(proof_scheduler)
    else:
        status = "BLOCKED_PREREQUISITES"
        next_action = "repair_first_failed_registered_rerun_gate"

    gate_path = active / "registered_research_rerun_gate.csv"
    summary_path = active / "registered_research_rerun_gate.json"
    _atomic_csv(gate, gate_path)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": as_of.isoformat(),
        "status": status,
        "contract_id": contract.get("contract_id", ""),
        "registered_candidates": len(gate),
        "candidate_gates_ready": int(
            gate.get("pre_rerun_gate_ready", pd.Series(dtype=bool)).map(_truthy).sum()
        ),
        "stage_two_complete": stage_two_complete,
        "vendor_parity_cells_proven": int(
            accepted_mode_evidence.sum()
        ),
        "vendor_formula_parity_cells_proven": int(
            parity.get(
                "vendor_exact_mode_parity_proven",
                pd.Series(False, index=parity.index),
            )
            .map(_truthy)
            .sum()
        ),
        "copula_behavioral_parity_cells_proven": int(
            (
                parity.get("exact_mode", pd.Series(dtype=str))
                .astype(str)
                .eq("Copula")
                & parity.get(
                    "vendor_behavioral_parity_proven",
                    pd.Series(False, index=parity.index),
                ).map(_truthy)
            ).sum()
        ),
        "vendor_parity_cells_required": 14,
        "proof_queue_complete": proof_queue_complete,
        "capture_reconciliation_status": capture_reconciliation["status"],
        "capture_reconciliation_id": capture_reconciliation["reconciliation_id"],
        "capture_reconciliation_required_calls": capture_reconciliation[
            "required_calls"
        ],
        "capture_reconciliation_completed_calls": capture_reconciliation[
            "completed_calls"
        ],
        "capture_reconciliation_blockers": capture_reconciliation["blockers"],
        "policy_ready": policy_ready,
        "chain_integrity": chain_integrity,
        "frozen_source_family_path": _relative(frozen_matrix_path, root),
        "frozen_source_family_receipt_path": _relative(
            frozen_source_receipt_path, root
        ),
        "post_ready_failure_attribution_integrity": failure_integrity,
        "frozen_discovery_policy": True,
        "family_rerun_scope": "full_policy_defined_family",
        "promotion_evaluation_scope": "registered_semantic_hypotheses_only",
        "evidence_ready_receipt_present": bool(ready_receipt),
        "registered_rerun_results_accounted": rerun_results_accounted,
        "registered_rerun_conclusion_status": conclusion_status,
        "accepted_registered_hypotheses": accepted_registered,
        "rejected_registered_hypotheses": rejected_registered,
        "pending_family_preflight_status": family_preflight.summary["status"],
        "pending_family_hypotheses": family_preflight.summary["hypotheses"],
        "pending_family_ready": family_preflight.summary["ready_hypotheses"],
        "pending_family_pairs": family_preflight.summary["pairs"],
        "pending_family_independent_clusters": family_preflight.summary[
            "independent_clusters"
        ],
        "pending_family_rollover_required": pending_family_rollover_required,
        "next_action": next_action,
        "rerun_execution_included": False,
        "registered_rerun_executor_available": True,
        "automatic_post_parity_handoff": True,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": (
            "reports/active/registered_research_rerun_contract.json;"
            f"{_relative(frozen_source_receipt_path, root)};"
            f"{_relative(frozen_matrix_path, root)};"
            "reports/active/registered_research_rerun_gate.csv;"
            "data/research/hypothesis_ledger.jsonl;"
            "data/processed/hyperliquid_pair_cost_models.csv;"
            "reports/active/wizard_mode_parity.csv;"
            "reports/active/corrective_wizard_capture_reconciliation.json;"
            "reports/active/registered_rerun_family_preflight.csv;"
            "reports/active/registered_rerun_family_preflight.json"
        ),
    }
    _atomic_json(summary, summary_path)
    paths = {
        "contract": contract_path,
        "gate": gate_path,
        "summary": summary_path,
        "frozen_source_family": frozen_matrix_path,
        "frozen_source_family_receipt": frozen_source_receipt_path,
        "family_preflight": Path(family_preflight.paths["preflight"]),
        "family_preflight_summary": Path(family_preflight.paths["summary"]),
    }
    if ready_receipt:
        paths["ready_receipt"] = ready_receipt_path
    if conclusion_receipt:
        paths["conclusion_receipt"] = conclusion_path
    return CommandResult(paths=paths, summary=summary)


def build_registered_family_handoff_preflight(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    """Prove pending Stage 4 candidates are runnable except for vendor evidence."""

    as_of = _as_utc(now)
    active = root / "reports" / "active"
    batch_path = active / "current_hypothesis_batch.csv"
    queue_path = active / "walkforward_near_miss_queue.csv"
    matrix_path = active / "current_wizard_hyperliquid_experiment_matrix.csv"
    ledger_audit_path = active / "hypothesis_ledger_audit.csv"
    active_costs_path = root / "data" / "processed" / "hyperliquid_pair_cost_models.csv"
    chain_path = active / "current_wizard_hyperliquid_chain_validation_manifest.json"
    acceptance_path = active / "acceptance_policy_receipt.json"
    holdout_path = active / "holdout_policy_receipt.json"
    history_manifest_path = active / "current_wizard_hyperliquid_history_manifest.json"
    l2_status_path = active / "corrective_l2_capture_status.json"

    batch = _read_csv(batch_path)
    queue = _read_csv(queue_path)
    matrix = _read_csv(matrix_path)
    ledger_audit = _read_csv(ledger_audit_path)
    chain = _read_json(chain_path)
    acceptance = _read_json(acceptance_path)
    holdout = _read_json(holdout_path)
    history_manifest = _read_json(history_manifest_path)
    l2_status = _read_json(l2_status_path)

    snapshot_manifest_path = _root_artifact(
        root,
        _text(history_manifest.get("artifacts", {}).get("snapshot_manifest")),
    )
    snapshot_pair_results_path = _root_artifact(
        root,
        _text(history_manifest.get("artifacts", {}).get("snapshot_pair_results")),
    )
    history_results = (
        _read_csv(snapshot_pair_results_path)
        if snapshot_pair_results_path is not None
        else pd.DataFrame()
    )
    history_manifest_bound = bool(
        snapshot_manifest_path is not None
        and snapshot_manifest_path.is_file()
        and _file_hash(snapshot_manifest_path) == _file_hash(history_manifest_path)
    )

    (
        costs,
        cost_model_source_path,
        cost_bundle_valid,
        active_cost_model_matches_bundle,
    ) = _resolve_bound_cost_models(
        root=root,
        active_costs_path=active_costs_path,
        l2_status=l2_status,
    )
    policies_valid = bool(
        acceptance.get("status") == "PASS" and holdout.get("status") == "PASS"
    )
    source_family_valid = bool(
        chain.get("chain_status") == "PASS"
        and int(chain.get("experiment_authority_count", 0) or 0) == len(matrix)
        and not matrix.empty
        and "experiment_id" in matrix
        and not matrix["experiment_id"].astype(str).duplicated().any()
        and not _truthy(chain.get("promotion_authority"))
        and not _truthy(chain.get("testnet_order_authority"))
        and not _truthy(chain.get("live_trading_authorized"))
    )

    rows = [
        _pending_family_preflight_row(
            candidate=row,
            as_of=as_of,
            queue=queue,
            matrix=matrix,
            ledger_audit=ledger_audit,
            costs=costs,
            history_results=history_results,
            acceptance=acceptance,
            holdout=holdout,
            history_manifest_bound=history_manifest_bound,
            cost_bundle_valid=cost_bundle_valid,
            source_family_valid=source_family_valid,
            root=root,
        )
        for row in batch.to_dict("records")
    ]
    frame = _collapse_pending_family_preflight_rows(pd.DataFrame(rows))
    ready = int(
        frame.get("non_vendor_preflight_ready", pd.Series(dtype=bool))
        .map(_truthy)
        .sum()
    )
    status = "PASS" if len(frame) > 0 and ready == len(frame) else "BLOCKED"
    preflight_path = active / "registered_rerun_family_preflight.csv"
    summary_path = active / "registered_rerun_family_preflight.json"
    _atomic_csv(frame, preflight_path)
    summary = {
        "schema_version": FAMILY_PREFLIGHT_SCHEMA_VERSION,
        "generated_at_utc": as_of.isoformat(),
        "status": status,
        "hypotheses": len(frame),
        "ready_hypotheses": ready,
        "blocked_hypotheses": len(frame) - ready,
        "pairs": int(frame.get("pair", pd.Series(dtype=str)).astype(str).nunique()),
        "independent_clusters": int(
            batch.get("equivalence_cluster_id", pd.Series(dtype=str))
            .astype(str)
            .nunique()
        ),
        "history_ready": int(
            frame.get("history_ready", pd.Series(dtype=bool)).map(_truthy).sum()
        ),
        "strict_cost_ready": int(
            frame.get("strict_cost_ready", pd.Series(dtype=bool)).map(_truthy).sum()
        ),
        "registration_ready": int(
            frame.get("registration_ready", pd.Series(dtype=bool)).map(_truthy).sum()
        ),
        "identity_ready": int(
            frame.get("identity_ready", pd.Series(dtype=bool)).map(_truthy).sum()
        ),
        "history_manifest_bound": history_manifest_bound,
        "cost_bundle_valid": cost_bundle_valid,
        "cost_model_source_path": (
            _relative(cost_model_source_path, root)
            if cost_model_source_path is not None
            else ""
        ),
        "cost_model_source_sha256": (
            _file_hash(cost_model_source_path)
            if cost_model_source_path is not None
            else ""
        ),
        "active_cost_model_matches_bundle": active_cost_model_matches_bundle,
        "policies_valid": policies_valid,
        "source_family_valid": source_family_valid,
        "vendor_evidence_included": False,
        "rerun_execution_included": False,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": (
            "reports/active/current_hypothesis_batch.csv;"
            "reports/active/walkforward_near_miss_queue.csv;"
            "reports/active/current_wizard_hyperliquid_experiment_matrix.csv;"
            "reports/active/hypothesis_ledger_audit.csv;"
            "data/processed/hyperliquid_pair_cost_models.csv;"
            f"{_relative(cost_model_source_path, root) if cost_model_source_path is not None else ''};"
            "reports/active/current_wizard_hyperliquid_history_manifest.json;"
            "reports/active/registered_rerun_family_preflight.csv"
        ),
    }
    _atomic_json(summary, summary_path)
    return CommandResult(
        paths={"preflight": preflight_path, "summary": summary_path},
        summary=summary,
    )


def _proof_scheduler_mixed_evidence_complete(
    *, root: Path, proof_scheduler: dict[str, Any]
) -> bool:
    queue_eligible = int(proof_scheduler.get("queue_eligible", 0) or 0)
    formula_expected = int(
        proof_scheduler.get("formula_proofs_expected", 0) or 0
    )
    formula_completed = int(proof_scheduler.get("completed_after", 0) or 0)
    copula_expected = int(
        proof_scheduler.get("copula_behavioral_expected_cells", 0) or 0
    )
    copula_passed = int(
        proof_scheduler.get("copula_behavioral_cells_passed", 0) or 0
    )
    copula_provenance = int(
        proof_scheduler.get("copula_behavioral_provenance_cells", 0) or 0
    )
    accepted = int(
        proof_scheduler.get("accepted_mode_evidence_cells", 0) or 0
    )
    responses = int(
        proof_scheduler.get("responses_captured_after", 0) or 0
    )
    cycle_path, cycle = _proof_scheduler_cycle_receipt(
        root=root,
        proof_scheduler=proof_scheduler,
    )
    copula_complete = bool(
        copula_expected == 0
        or (
            copula_passed == copula_expected
            and copula_provenance == copula_expected
            and _truthy(
                proof_scheduler.get("copula_behavioral_parity_proven")
            )
            and not _truthy(
                proof_scheduler.get("copula_formula_parity_proven")
            )
            and _truthy(proof_scheduler.get("copula_cohort_receipt_valid"))
            and _valid_bound_artifact(
                root=root,
                relative_path=_text(
                    proof_scheduler.get("copula_cohort_receipt_path")
                ),
                expected_sha256=_text(
                    proof_scheduler.get("copula_cohort_receipt_sha256")
                ),
                expected_root=(
                    root
                    / "data"
                    / "research"
                    / "wizard_copula_behavioral_cohorts"
                ),
                expected_stem=_text(
                    proof_scheduler.get("copula_cohort_receipt_id")
                ),
            )
        )
    )
    authority_safe = not any(
        _truthy(proof_scheduler.get(key))
        for key in (
            "candidate_promotion_authority",
            "order_submission_included",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    )
    credit_evidence_complete = _proof_scheduler_credit_evidence_complete(
        root=root,
        proof_scheduler=proof_scheduler,
    )
    capture_evidence_complete = (
        validate_capture_reconciliation_evidence(
            root=root, evidence=proof_scheduler
        ).get("status")
        == "PASS"
    )
    ou_v5_evidence_complete = bool(
        validate_ou_v5_stage3_evidence(root=root, evidence=proof_scheduler).get(
            "status"
        )
        == "PASS"
    )
    source_binding = validate_capture_manifest_source_receipt(
        root=root,
        manifest_id=proof_scheduler.get("capture_manifest_id"),
        manifest_path=proof_scheduler.get("capture_manifest_immutable_path"),
        manifest_sha256=proof_scheduler.get("capture_manifest_immutable_sha256"),
    )
    source_evidence_complete = bool(
        _truthy(proof_scheduler.get("capture_manifest_candidate_source_binding_valid"))
        and source_binding.get("status") == "PASS"
        and _text(source_binding.get("receipt_id"))
        == _text(proof_scheduler.get("capture_manifest_source_receipt_id"))
        and _text(source_binding.get("receipt_path"))
        == _text(proof_scheduler.get("capture_manifest_source_receipt_path"))
        and _text(source_binding.get("receipt_sha256")).lower()
        == _text(proof_scheduler.get("capture_manifest_source_receipt_sha256")).lower()
        and _text(source_binding.get("source_artifacts_sha256")).lower()
        == _text(proof_scheduler.get("capture_manifest_source_artifacts_sha256")).lower()
    )
    return bool(
        queue_eligible > 0
        and formula_expected + copula_expected == queue_eligible
        and formula_completed == formula_expected
        and accepted == queue_eligible
        and responses >= queue_eligible
        and copula_complete
        and _text(proof_scheduler.get("parity_refresh_status")) == "PASS"
        and _text(proof_scheduler.get("status"))
        in {"COMPLETE_QUEUE", "COMPLETE_ACCEPTED_MODE_EVIDENCE"}
        and cycle_path is not None
        and bool(cycle)
        and credit_evidence_complete
        and capture_evidence_complete
        and ou_v5_evidence_complete
        and source_evidence_complete
        and _truthy(proof_scheduler.get("capture_manifest_continuity_valid"))
        and authority_safe
    )


def _proof_scheduler_credit_evidence_complete(
    *, root: Path, proof_scheduler: dict[str, Any]
) -> bool:
    """Re-verify the immutable shared-credit ledger before Stage 4 handoff."""

    if (
        _text(proof_scheduler.get("credit_reconciliation_status"))
        not in {"PASS_RECONCILED", "REUSED_RECONCILIATION"}
        or not _truthy(
            proof_scheduler.get("copula_behavioral_response_accounting_valid")
        )
    ):
        return False
    try:
        attempted = int(proof_scheduler.get("proof_lane_attempted_credits", -1))
        completed = int(proof_scheduler.get("proof_lane_completed_credits", -1))
        unresolved = int(
            proof_scheduler.get("proof_lane_uncompleted_attempted_credits", -1)
        )
        external_requests = int(
            proof_scheduler.get("exact_mode_requests_attempted", -1)
        ) + int(proof_scheduler.get("copula_behavioral_endpoint_calls", -1))
    except (TypeError, ValueError):
        return False
    if (
        attempted < 0
        or completed < 0
        or external_requests < 0
        or attempted != completed
        or unresolved != 0
    ):
        return False
    evidence = validate_wizard_credit_lane_evidence(
        root=root,
        lane=PROOF_LANE,
        credit_date_utc=_text(proof_scheduler.get("attempt_date_utc")),
        reservation_id=_text(proof_scheduler.get("credit_reservation_id")),
        reconciliation_id=_text(proof_scheduler.get("credit_reconciliation_id")),
        reservation_path=_text(proof_scheduler.get("credit_reservation_path")),
        reconciliation_path=_text(
            proof_scheduler.get("credit_reconciliation_path")
        ),
    )
    return bool(
        evidence.get("status") == "PASS"
        and int(evidence.get("attempted_credits", -1)) == attempted
        and int(evidence.get("completed_credits", -1)) == completed
        and int(evidence.get("external_requests", -1)) == external_requests
        and not any(
            _truthy(evidence.get(key))
            for key in (
                "order_submission_included",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
    )


def _proof_scheduler_cycle_receipt(
    *, root: Path, proof_scheduler: dict[str, Any]
) -> tuple[Path | None, dict[str, Any]]:
    relative = _text(proof_scheduler.get("receipt_path"))
    if not relative:
        return None, {}
    expected_root = root / "reports" / "active" / "wizard_proof_scheduler_receipts"
    path = root / relative
    if not _path_within(path, expected_root) or not path.is_file():
        return None, {}
    cycle = _read_json(path)
    if not cycle or _text(cycle.get("receipt_id")) != _text(
        proof_scheduler.get("receipt_id")
    ):
        return None, {}
    material = {key: value for key, value in cycle.items() if key != "receipt_id"}
    expected_receipt_id = "wizardproof_" + sha256(
        _canonical_json(material).encode("utf-8")
    ).hexdigest()[:20]
    if _text(cycle.get("receipt_id")) != expected_receipt_id:
        return None, {}
    immutable_root = root / "data" / "research" / "wizard_proof_scheduler_receipts"
    immutable_path = immutable_root / f"{expected_receipt_id}.json"
    if (
        not _path_within(immutable_path, immutable_root)
        or not immutable_path.is_file()
        or immutable_path.read_bytes() != path.read_bytes()
    ):
        return None, {}
    bound_fields = (
        "attempt_date_utc",
        "status",
        "queue_eligible",
        "completed_after",
        "responses_captured_after",
        "formula_proofs_expected",
        "accepted_mode_evidence_cells",
        "copula_behavioral_expected_cells",
        "copula_behavioral_cells_passed",
        "copula_behavioral_provenance_cells",
        "copula_behavioral_endpoint_calls",
        "copula_behavioral_responses_captured_this_cycle",
        "copula_behavioral_response_accounting_valid",
        "copula_behavioral_parity_proven",
        "copula_formula_parity_proven",
        "copula_cohort_receipt_id",
        "copula_cohort_receipt_path",
        "copula_cohort_receipt_sha256",
        "copula_cohort_receipt_valid",
        "parity_refresh_status",
        "exact_mode_requests_attempted",
        "exact_mode_responses_captured_this_cycle",
        "proof_lane_attempted_credits",
        "proof_lane_completed_credits",
        "proof_lane_uncompleted_attempted_credits",
        "credit_reconciliation_status",
        "credit_reservation_id",
        "credit_reservation_path",
        "credit_reconciliation_id",
        "credit_reconciliation_path",
        "capture_manifest_accounting_valid",
        "capture_manifest_id",
        "capture_manifest_immutable_path",
        "capture_manifest_immutable_sha256",
        "capture_manifest_candidate_id",
        "capture_manifest_candidate_immutable_path",
        "capture_manifest_candidate_immutable_sha256",
        "capture_manifest_candidate_binding_valid",
        "capture_manifest_candidate_source_binding_valid",
        "capture_manifest_source_receipt_id",
        "capture_manifest_source_receipt_path",
        "capture_manifest_source_receipt_sha256",
        "capture_manifest_source_artifacts_sha256",
        "capture_manifest_carried_forward",
        "capture_manifest_drift_detected",
        "capture_manifest_continuity_status",
        "capture_manifest_continuity_valid",
        "capture_manifest_continuity_blockers",
        "capture_reconciliation_status",
        "capture_reconciliation_valid",
        "capture_reconciliation_local_state_valid",
        "capture_reconciliation_complete",
        "capture_reconciliation_id",
        "capture_reconciliation_immutable_path",
        "capture_reconciliation_immutable_sha256",
        "capture_reconciliation_manifest_id",
        "capture_reconciliation_manifest_path",
        "capture_reconciliation_manifest_sha256",
        "capture_reconciliation_required_calls",
        "capture_reconciliation_completed_calls",
        "capture_reconciliation_pending_calls",
        "capture_reconciliation_blocked_calls",
        "ou_v5_prospectively_registered",
        "ou_v5_holdout_status",
        "ou_v5_evaluation_status",
        "ou_v5_required_responses",
        "ou_v5_responses_available",
        "ou_v5_response_accounting_valid",
        "ou_v5_activation_status",
        "ou_v5_comparator_generation",
        "ou_v5_proof_refresh_status",
        "ou_v5_proofs_refreshed",
        "ou_v5_activation_automatic",
        "ou_v5_research_only",
    )
    if any(cycle.get(field) != proof_scheduler.get(field) for field in bound_fields):
        return None, {}
    if any(
        _truthy(cycle.get(key))
        for key in (
            "candidate_promotion_authority",
            "order_submission_included",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    ):
        return None, {}
    return immutable_path, cycle


def _validate_ready_receipt_lineage(
    *,
    root: Path,
    ready_receipt: dict[str, Any],
    contract: dict[str, Any],
    parity_path: Path,
) -> None:
    material = {
        key: value
        for key, value in ready_receipt.items()
        if key != "ready_receipt_id"
    }
    expected_id = "registeredready_" + sha256(
        _canonical_json(material).encode("utf-8")
    ).hexdigest()[:20]
    if _text(ready_receipt.get("ready_receipt_id")) != expected_id:
        raise ValueError("registered rerun ready receipt content identity mismatch")
    if _text(ready_receipt.get("schema_version")) != READY_SCHEMA_VERSION:
        raise ValueError("registered rerun ready receipt schema mismatch")
    if _text(ready_receipt.get("contract_id")) != _text(
        contract.get("contract_id")
    ):
        raise ValueError("registered rerun ready receipt contract mismatch")
    if _text(ready_receipt.get("parity_sha256")) != _file_hash(parity_path):
        raise ValueError("registered rerun ready receipt parity binding mismatch")
    cycle_path = _text(
        ready_receipt.get("proof_scheduler_cycle_receipt_path")
    )
    if not _valid_bound_artifact(
        root=root,
        relative_path=cycle_path,
        expected_sha256=_text(
            ready_receipt.get("proof_scheduler_cycle_receipt_sha256")
        ),
        expected_root=(
            root / "reports" / "active" / "wizard_proof_scheduler_receipts"
        ),
    ):
        raise ValueError("registered rerun ready receipt scheduler binding mismatch")
    cycle = _read_json(root / cycle_path)
    if _text(cycle.get("receipt_id")) != _text(
        ready_receipt.get("proof_scheduler_receipt_id")
    ):
        raise ValueError("registered rerun ready receipt scheduler ID mismatch")
    cycle_with_path = {**cycle, "receipt_path": cycle_path}
    if not _proof_scheduler_mixed_evidence_complete(
        root=root,
        proof_scheduler=cycle_with_path,
    ):
        raise ValueError(
            "registered rerun ready receipt scheduler evidence is incomplete"
        )
    count_bindings = {
        "proof_cells_completed": "accepted_mode_evidence_cells",
        "formula_proof_cells_completed": "completed_after",
        "formula_proof_cells_expected": "formula_proofs_expected",
        "copula_behavioral_cells_completed": (
            "copula_behavioral_cells_passed"
        ),
        "copula_behavioral_cells_expected": (
            "copula_behavioral_expected_cells"
        ),
        "accepted_mode_evidence_cells": "accepted_mode_evidence_cells",
        "capture_reconciliation_required_calls": (
            "capture_reconciliation_required_calls"
        ),
        "capture_reconciliation_completed_calls": (
            "capture_reconciliation_completed_calls"
        ),
    }
    for ready_field, cycle_field in count_bindings.items():
        ready_value = ready_receipt.get(ready_field)
        cycle_value = cycle.get(cycle_field)
        if (
            ready_value is None
            or cycle_value is None
            or int(ready_value) != int(cycle_value)
        ):
            raise ValueError(
                f"registered rerun ready receipt count mismatch: {ready_field}"
            )
    for field in (
        "copula_cohort_receipt_id",
        "copula_cohort_receipt_path",
        "copula_cohort_receipt_sha256",
        "capture_reconciliation_id",
        "capture_reconciliation_immutable_path",
        "capture_reconciliation_immutable_sha256",
        "capture_reconciliation_manifest_id",
        "capture_reconciliation_manifest_path",
        "capture_reconciliation_manifest_sha256",
    ):
        if _text(ready_receipt.get(field)) != _text(cycle.get(field)):
            raise ValueError(
                f"registered rerun ready receipt lineage mismatch: {field}"
            )
    copula_expected = int(
        ready_receipt.get("copula_behavioral_cells_expected", 0) or 0
    )
    if copula_expected > 0 and not _valid_bound_artifact(
        root=root,
        relative_path=_text(ready_receipt.get("copula_cohort_receipt_path")),
        expected_sha256=_text(
            ready_receipt.get("copula_cohort_receipt_sha256")
        ),
        expected_root=(
            root / "data" / "research" / "wizard_copula_behavioral_cohorts"
        ),
        expected_stem=_text(ready_receipt.get("copula_cohort_receipt_id")),
    ):
        raise ValueError("registered rerun ready receipt Copula binding mismatch")
    if not _truthy(ready_receipt.get("scheduled_research_rerun_authorized")):
        raise ValueError("registered rerun ready receipt lacks research authority")
    if any(
        _truthy(ready_receipt.get(key))
        for key in (
            "promotion_authority",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    ):
        raise ValueError("registered rerun ready receipt grants order authority")


def _valid_bound_artifact(
    *,
    root: Path,
    relative_path: str,
    expected_sha256: str,
    expected_root: Path,
    expected_stem: str = "",
) -> bool:
    if not relative_path or not expected_sha256:
        return False
    relative = Path(relative_path)
    if relative.is_absolute():
        return False
    path = root / relative
    return bool(
        _path_within(path, expected_root)
        and path.is_file()
        and (not expected_stem or path.stem == expected_stem)
        and _file_hash(path) == expected_sha256
    )


def _path_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _next_vendor_parity_action(proof_scheduler: dict[str, Any]) -> str:
    eligible = int(proof_scheduler.get("queue_eligible", 0) or 0)
    proven = int(proof_scheduler.get("completed_after", 0) or 0)
    captured = int(proof_scheduler.get("responses_captured_after", 0) or 0)
    if not _truthy(proof_scheduler.get("capture_reconciliation_complete")):
        return "reconcile_frozen_wizard_capture_manifest"
    if eligible > 0 and captured >= eligible and proven < eligible:
        return "implement_mode_specific_comparators_from_preserved_vendor_series"
    return "proof_scheduler_completes_bounded_exact_mode_queue"


def _resolve_registered_outcomes(
    *,
    candidates: list[dict[str, Any]],
    queue: pd.DataFrame,
    final_survivor: dict[str, Any],
    failure_attribution: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Account for accepted and rejected hypotheses without assuming queue survival."""

    rejected_ids = set(
        queue.get("semantic_hypothesis_id", pd.Series(dtype=str)).astype(str)
    ) - {""}
    accepted_ids = {
        _text(value) for value in final_survivor.get("final_experiment_ids", [])
    } - {""}
    accounted_experiment_ids = set(
        (failure_attribution if failure_attribution is not None else pd.DataFrame())
        .get("experiment_id", pd.Series(dtype=str))
        .astype(str)
    ) - {""}
    outcomes: dict[str, str] = {}
    for candidate in candidates:
        semantic_id = _text(candidate.get("semantic_hypothesis_id"))
        experiment_id = _text(candidate.get("source_experiment_id"))
        if not semantic_id:
            continue
        if experiment_id in accepted_ids:
            outcomes[semantic_id] = "ACCEPTED_SURVIVOR"
        elif experiment_id in accounted_experiment_ids or semantic_id in rejected_ids:
            outcomes[semantic_id] = "REJECTED_BY_FROZEN_GATES"
        else:
            outcomes[semantic_id] = "UNACCOUNTED"
    return outcomes


def _load_or_create_contract(
    *,
    root: Path,
    as_of: datetime,
    active_contract_path: Path,
    batch: pd.DataFrame,
    queue: pd.DataFrame,
    ledger: list[dict[str, Any]],
    acceptance: dict[str, Any],
    holdout: dict[str, Any],
    discovery_policy_path: Path,
    matrix_path: Path,
    chain: dict[str, Any],
    generation_rollover_ready: bool = True,
    candidate_selection_path: Path | None = None,
    candidate_selection_status: str = "",
) -> dict[str, Any]:
    existing = _read_json(active_contract_path)
    if existing:
        immutable = root / str(existing.get("immutable_contract_path", ""))
        immutable_payload = _read_json(immutable)
        if not immutable_payload or immutable_payload != existing:
            raise ValueError("registered rerun contract immutable copy mismatch")
    if batch.empty:
        if existing:
            return existing
        raise ValueError("registered rerun has no candidate selection")
    queue_by_identity = {
        (
            _text(row.get("semantic_hypothesis_id")),
            _text(row.get("experiment_id")),
        ): row
        for row in queue.to_dict("records")
        if _text(row.get("semantic_hypothesis_id"))
        and _text(row.get("experiment_id"))
    }
    queue_by_semantic: dict[str, list[dict[str, Any]]] = {}
    for row in queue.to_dict("records"):
        queue_by_semantic.setdefault(
            _text(row.get("semantic_hypothesis_id")), []
        ).append(row)
    ledger_by_id: dict[str, list[dict[str, Any]]] = {}
    for record in ledger:
        ledger_by_id.setdefault(
            _text(record.get("semantic_hypothesis_id")), []
        ).append(record)
    candidates = []
    for row in batch.to_dict("records"):
        semantic_id = _text(row.get("semantic_hypothesis_id"))
        experiment_id = _text(row.get("experiment_id"))
        queue_row = queue_by_identity.get((semantic_id, experiment_id), {})
        if not queue_row and len(queue_by_semantic.get(semantic_id, [])) == 1:
            queue_row = queue_by_semantic[semantic_id][0]
        records = ledger_by_id.get(semantic_id, [])
        first = min(
            records,
            key=lambda record: _text(record.get("registered_at_utc")),
            default={},
        )
        candidates.append(
            {
                "semantic_hypothesis_id": semantic_id,
                "source_experiment_id": experiment_id,
                "pair_group_key": _text(row.get("pair_group_key"))
                or _text(queue_row.get("pair_group_key")),
                "pair": _text(row.get("pair")),
                "exact_mode": _text(row.get("exact_mode")),
                "orientation": _text(row.get("orientation")),
                "preflight_selection_reason": _text(
                    row.get("representative_selection_reason")
                ),
                "preflight_alternative_experiment_ids": _text(
                    row.get("alternative_experiment_ids")
                ),
                "cost_model_id": _text(row.get("cost_model_id")),
                "cost_model_as_of_utc": _text(row.get("cost_model_as_of_utc")),
                "history_run_id": _text(row.get("history_run_id")),
                "history_path": _text(row.get("history_path")),
                "history_sha256": _text(row.get("history_sha256")),
                "registered_at_utc": _text(first.get("registered_at_utc")),
                "ledger_record_hash": _text(first.get("record_hash")),
                "hypothesis_registered_before_next_test": _truthy(
                    row.get("hypothesis_registered_before_next_test")
                ),
                "next_test_executed_at_contract": _truthy(
                    row.get("next_test_executed")
                ),
                "execution_authority_at_contract": _truthy(
                    row.get("execution_authority")
                ),
                "testnet_order_authority_at_contract": _truthy(
                    row.get("testnet_order_authority")
                ),
                "live_trading_authorized_at_contract": _truthy(
                    row.get("live_trading_authorized")
                ),
            }
        )
    candidate_semantic_ids = [
        _text(candidate.get("semantic_hypothesis_id")) for candidate in candidates
    ]
    if (
        not candidate_semantic_ids
        or "" in candidate_semantic_ids
        or len(candidate_semantic_ids) != len(set(candidate_semantic_ids))
    ):
        raise ValueError("registered rerun contract semantic candidate identities invalid")
    material = {
        "schema_version": SCHEMA_VERSION,
        "registered_candidates": candidates,
        "acceptance_policy_id": acceptance.get("policy_id", ""),
        "holdout_policy_id": holdout.get("policy_id", ""),
        "discovery_policy_sha256": _file_hash(discovery_policy_path),
        "source_family_sha256": _file_hash(matrix_path),
        "source_family_rows": int(chain.get("experiment_authority_count", 0) or 0),
        "candidate_selection_path": (
            _relative(candidate_selection_path, root)
            if candidate_selection_path is not None
            else ""
        ),
        "candidate_selection_sha256": (
            _file_hash(candidate_selection_path)
            if candidate_selection_path is not None
            else ""
        ),
        "candidate_selection_status": candidate_selection_status,
    }
    material_matches_existing = bool(
        existing
        and all(existing.get(key) == value for key, value in material.items())
    )
    if material_matches_existing:
        return existing
    prior_contract_id = ""
    generation = 1
    if existing:
        prior_contract_id = _text(existing.get("contract_id"))
        conclusion_path = (
            root
            / "data"
            / "research"
            / "registered_rerun_conclusions"
            / f"{prior_contract_id}.json"
        )
        conclusion = _read_json(conclusion_path)
        matrix = _read_csv(matrix_path)
        candidate_experiment_ids = [
            _text(candidate.get("source_experiment_id")) for candidate in candidates
        ]
        matrix_experiment_counts = (
            matrix.get("experiment_id", pd.Series(dtype=str)).astype(str).value_counts()
        )
        candidate_identities_valid = bool(
            candidate_experiment_ids
            and "" not in candidate_experiment_ids
            and len(set(candidate_experiment_ids)) == len(candidate_experiment_ids)
            and all(
                int(matrix_experiment_counts.get(experiment_id, 0)) == 1
                for experiment_id in candidate_experiment_ids
            )
        )
        source_family_valid = bool(
            candidates
            and int(material["source_family_rows"]) > 0
            and len(_text(material["source_family_sha256"])) == 64
            and chain.get("chain_status") == "PASS"
            and int(chain.get("experiment_authority_count", 0) or 0) == len(matrix)
            and candidate_identities_valid
        )
        if not _contract_conclusively_accounted(existing, conclusion):
            return existing
        if not source_family_valid:
            return existing
        if not generation_rollover_ready:
            return existing
        generation = int(existing.get("generation", 1) or 1) + 1
    material["generation"] = generation
    material["prior_contract_id"] = prior_contract_id
    contract_id = "registeredrerun_" + sha256(
        _canonical_json(material).encode("utf-8")
    ).hexdigest()[:20]
    immutable_path = (
        root / "data" / "research" / "registered_rerun_contracts" / f"{contract_id}.json"
    )
    contract = {
        **material,
        "contract_id": contract_id,
        "created_at_utc": as_of.isoformat(),
        "immutable_contract_path": _relative(immutable_path, root),
        "full_family_multiplicity_required": True,
        "promotion_evaluation_registered_only": True,
        "threshold_changes_after_contract_permitted": False,
        "rerun_execution_included": False,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    _atomic_json(contract, immutable_path)
    _atomic_json(contract, active_contract_path)
    return contract


def _contract_conclusively_accounted(
    contract: dict[str, Any], conclusion: dict[str, Any]
) -> bool:
    candidates = contract.get("registered_candidates", [])
    if not isinstance(candidates, list):
        return False
    candidate_ids = [
        _text(candidate.get("semantic_hypothesis_id"))
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    if (
        not candidate_ids
        or "" in candidate_ids
        or len(candidate_ids) != len(candidates)
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        return False
    expected_ids = set(candidate_ids)
    outcomes = conclusion.get("outcomes", {})
    if not isinstance(outcomes, dict) or set(outcomes) != expected_ids:
        return False
    allowed = {"ACCEPTED_SURVIVOR", "REJECTED_BY_FROZEN_GATES"}
    if not outcomes or any(outcome not in allowed for outcome in outcomes.values()):
        return False
    accepted = sum(
        outcome == "ACCEPTED_SURVIVOR" for outcome in outcomes.values()
    )
    rejected = sum(
        outcome == "REJECTED_BY_FROZEN_GATES" for outcome in outcomes.values()
    )
    try:
        registered_count = int(conclusion.get("registered_hypotheses", -1))
        accepted_count = int(
            conclusion.get("accepted_registered_hypotheses", -1)
        )
        rejected_count = int(
            conclusion.get("rejected_registered_hypotheses", -1)
        )
    except (TypeError, ValueError):
        return False
    status = _text(conclusion.get("conclusion_status"))
    conclusion_core = {
        key: value for key, value in conclusion.items() if key != "conclusion_id"
    }
    expected_conclusion_id = "registeredconclusion_" + sha256(
        _canonical_json(conclusion_core).encode("utf-8")
    ).hexdigest()[:20]
    return bool(
        conclusion.get("schema_version") == CONCLUSION_SCHEMA_VERSION
        and conclusion.get("conclusion_id") == expected_conclusion_id
        and conclusion.get("contract_id") == contract.get("contract_id")
        and registered_count == len(expected_ids)
        and accepted_count == accepted
        and rejected_count == rejected
        and accepted + rejected == len(expected_ids)
        and (
            (status == "ACCEPTED_REGISTERED_SURVIVORS" and accepted > 0)
            or (
                status == "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS"
                and accepted == 0
            )
        )
        and not _truthy(conclusion.get("promotion_authority"))
        and not _truthy(conclusion.get("testnet_order_authority"))
        and not _truthy(conclusion.get("live_trading_authorized"))
    )


def contract_conclusively_accounted(
    contract: dict[str, Any], conclusion: dict[str, Any]
) -> bool:
    """Expose the fail-closed registered conclusion validator to checkpoint gates."""

    return _contract_conclusively_accounted(contract, conclusion)


def resolve_registered_source_family(
    *,
    root: Path,
    contract: dict[str, Any],
    current_matrix_path: Path | None = None,
) -> tuple[Path, Path]:
    """Bind a rerun contract to an immutable source-family matrix across daily refreshes."""

    contract_id = _text(contract.get("contract_id"))
    expected_hash = _text(contract.get("source_family_sha256"))
    expected_rows = int(contract.get("source_family_rows", -1) or -1)
    if not contract_id or len(expected_hash) != 64 or expected_rows <= 0:
        raise ValueError("registered rerun source family contract is invalid")
    bundle = root / "data" / "research" / "registered_rerun_source_families" / contract_id
    frozen_matrix = bundle / "experiment_matrix.csv"
    receipt_path = bundle / "receipt.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path)
        if (
            receipt.get("schema_version") != SOURCE_FAMILY_SCHEMA_VERSION
            or receipt.get("contract_id") != contract_id
            or receipt.get("source_family_sha256") != expected_hash
            or int(receipt.get("source_family_rows", -1) or -1) != expected_rows
            or _file_hash(frozen_matrix) != expected_hash
            or len(_read_csv(frozen_matrix)) != expected_rows
            or _truthy(receipt.get("promotion_authority"))
            or _truthy(receipt.get("testnet_order_authority"))
            or _truthy(receipt.get("live_trading_authorized"))
        ):
            raise ValueError("registered rerun source family receipt mismatch")
        return frozen_matrix, receipt_path

    search_paths: list[Path] = []
    if current_matrix_path is not None:
        search_paths.append(current_matrix_path)
    snapshot_root = root / "reports" / "snapshots" / "current_wizard_hyperliquid"
    if snapshot_root.exists():
        search_paths.extend(sorted(snapshot_root.glob("**/experiment_matrix.csv")))
    run_root = root / "reports" / "runs" / "current_wizard_hyperliquid_daily"
    if run_root.exists():
        search_paths.extend(sorted(run_root.glob("**/snapshot_experiments.csv")))
    source = next(
        (
            path
            for path in dict.fromkeys(search_paths)
            if _file_hash(path) == expected_hash and len(_read_csv(path)) == expected_rows
        ),
        None,
    )
    if source is None:
        raise ValueError("registered rerun frozen source family is unavailable")
    frozen_matrix.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = source.read_bytes()
    if frozen_matrix.is_file():
        if frozen_matrix.read_bytes() != expected_bytes:
            raise ValueError("registered rerun frozen source family changed")
    else:
        temporary = frozen_matrix.with_suffix(frozen_matrix.suffix + ".tmp")
        temporary.write_bytes(expected_bytes)
        temporary.replace(frozen_matrix)
    receipt_core = {
        "schema_version": SOURCE_FAMILY_SCHEMA_VERSION,
        "contract_id": contract_id,
        "source_family_path": _relative(frozen_matrix, root),
        "source_family_sha256": expected_hash,
        "source_family_rows": expected_rows,
        "recovered_from_path": _relative(source, root),
        "daily_active_layer_independent": True,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    receipt = {
        **receipt_core,
        "receipt_id": "registeredsource_"
        + sha256(_canonical_json(receipt_core).encode("utf-8")).hexdigest()[:20],
    }
    _write_or_validate_immutable_json(receipt, receipt_path)
    return frozen_matrix, receipt_path


def _accepted_mode_evidence_mask(parity: pd.DataFrame) -> pd.Series:
    """Apply the predeclared mode-specific Wizard evidence contract."""

    if parity.empty:
        return pd.Series(False, index=parity.index, dtype=bool)
    exact_mode = parity.get(
        "exact_mode", pd.Series("", index=parity.index)
    ).astype(str)
    formula = parity.get(
        "vendor_exact_mode_parity_proven",
        pd.Series(False, index=parity.index),
    ).map(_truthy)
    behavioral = parity.get(
        "vendor_behavioral_parity_proven",
        pd.Series(False, index=parity.index),
    ).map(_truthy)
    mixed_gate = parity.get(
        "vendor_mode_evidence_gate_passed",
        formula | behavioral,
    ).map(_truthy)
    return formula | (exact_mode.eq("Copula") & behavioral & mixed_gate)


def _candidate_gate_row(
    *,
    candidate: dict[str, Any],
    as_of: datetime,
    queue: pd.DataFrame,
    ledger: list[dict[str, Any]],
    ledger_audit: pd.DataFrame,
    matrix: pd.DataFrame,
    costs: pd.DataFrame,
    parity: pd.DataFrame,
    parity_complete: bool,
    proof_queue_complete: bool,
    stage_two_complete: bool,
    policy_ready: bool,
    chain_integrity: bool,
) -> dict[str, Any]:
    semantic_id = _text(candidate.get("semantic_hypothesis_id"))
    experiment_id = _text(candidate.get("source_experiment_id"))
    pair_key = _text(candidate.get("pair_group_key"))
    registered_at = pd.to_datetime(
        candidate.get("registered_at_utc"), utc=True, errors="coerce"
    )
    queue_match = queue.loc[
        queue.get("semantic_hypothesis_id", pd.Series(dtype=str))
        .astype(str)
        .eq(semantic_id)
    ]
    matrix_match = matrix.loc[
        matrix.get("experiment_id", pd.Series(dtype=str)).astype(str).eq(experiment_id)
    ]
    cost_match = costs.loc[
        costs.get("pair_group_key", pd.Series(dtype=str)).astype(str).eq(pair_key)
    ]
    parity_match = parity.loc[
        parity.get("exact_mode", pd.Series(dtype=str))
        .astype(str)
        .eq(_text(candidate.get("exact_mode")))
        & parity.get("orientation", pd.Series(dtype=str))
        .astype(str)
        .eq(_text(candidate.get("orientation")))
    ]
    audit_match = ledger_audit.loc[
        ledger_audit.get("semantic_hypothesis_id", pd.Series(dtype=str))
        .astype(str)
        .eq(semantic_id)
    ]
    ledger_match = [
        record
        for record in ledger
        if _text(record.get("semantic_hypothesis_id")) == semantic_id
    ]
    authority_safe = bool(
        not _truthy(candidate.get("execution_authority_at_contract"))
        and not _truthy(candidate.get("testnet_order_authority_at_contract"))
        and not _truthy(candidate.get("live_trading_authorized_at_contract"))
        and all(
            not _truthy(record.get("promotion_authority"))
            and not _truthy(record.get("testnet_order_authority"))
            and not _truthy(record.get("live_trading_authorized"))
            for record in ledger_match
        )
        and (
            audit_match.empty
            or (
                not _truthy(audit_match.iloc[0].get("promotion_authority"))
                and not _truthy(
                    audit_match.iloc[0].get("live_trading_authorized")
                )
            )
        )
        and (
            queue_match.empty
            or (
                not _truthy(queue_match.iloc[0].get("execution_authority"))
                and not _truthy(queue_match.iloc[0].get("promotion_authority"))
                and not _truthy(queue_match.iloc[0].get("testnet_order_authority"))
                and not _truthy(queue_match.iloc[0].get("live_trading_authorized"))
            )
        )
    )
    registration_valid = bool(
        semantic_id
        and pd.notna(registered_at)
        and _truthy(candidate.get("hypothesis_registered_before_next_test"))
        and not _truthy(candidate.get("next_test_executed_at_contract"))
        and len(ledger_match) >= 1
        and len(audit_match) == 1
        and _truthy(audit_match.iloc[0].get("ledger_chain_valid"))
        and authority_safe
    )
    identity_match = bool(
        len(matrix_match) == 1
        and _text(matrix_match.iloc[0].get("pair")) == _text(candidate.get("pair"))
        and _text(matrix_match.iloc[0].get("pair_group_key")) == pair_key
        and _text(matrix_match.iloc[0].get("exact_mode"))
        == _text(candidate.get("exact_mode"))
        and _text(matrix_match.iloc[0].get("orientation"))
        == _text(candidate.get("orientation"))
    )
    cost_row = cost_match.iloc[0] if len(cost_match) == 1 else pd.Series(dtype=object)
    cost_at = pd.to_datetime(
        cost_row.get("model_as_of_utc"), utc=True, errors="coerce"
    )
    cost_ready = bool(
        len(cost_match) == 1
        and _truthy(cost_row.get("strict_observed_cost_ready"))
        and _truthy(cost_row.get("cost_acceptance_ready"))
        and _text(cost_row.get("cost_model_status")) == "STRICT_OBSERVED"
        and pd.notna(cost_at)
        and pd.notna(registered_at)
        and cost_at > registered_at
        and cost_at <= pd.Timestamp(as_of)
        and pd.Timestamp(as_of) - cost_at <= pd.Timedelta(hours=2)
        and all(
            len(_text(cost_row.get(field))) == 64
            for field in (
                "candidate_set_sha256",
                "cost_status_sha256",
                "l2_samples_sha256",
                "fee_profile_sha256",
            )
        )
    )
    candidate_parity_ready = bool(
        len(parity_match) == 1
        and _accepted_mode_evidence_mask(parity_match).all()
    )
    blockers = []
    for passed, blocker in (
        (registration_valid, "registration_or_ledger_invalid"),
        (identity_match, "registered_identity_missing_from_frozen_source_family"),
        (stage_two_complete and cost_ready, "strict_cost_evidence_missing_stale_or_pre_registration"),
        (parity_complete, "all_14_vendor_mode_evidence_cells_not_proven"),
        (candidate_parity_ready, "candidate_mode_orientation_parity_not_proven"),
        (proof_queue_complete, "bounded_proof_queue_not_complete"),
        (policy_ready, "frozen_policy_receipts_mismatch"),
        (chain_integrity, "frozen_source_family_integrity_failed"),
    ):
        if not passed:
            blockers.append(blocker)
    return {
        "schema_version": SCHEMA_VERSION,
        "semantic_hypothesis_id": semantic_id,
        "source_experiment_id": experiment_id,
        "pair_group_key": pair_key,
        "pair": _text(candidate.get("pair")),
        "exact_mode": _text(candidate.get("exact_mode")),
        "orientation": _text(candidate.get("orientation")),
        "registered_at_utc": _text(candidate.get("registered_at_utc")),
        "registration_valid": registration_valid,
        "authority_boundary_safe": authority_safe,
        "current_family_identity_match": identity_match,
        "frozen_source_family_identity_match": identity_match,
        "stage_two_complete": stage_two_complete,
        "cost_model_id": _text(cost_row.get("cost_model_id")),
        "strict_cost_ready_after_registration": cost_ready,
        "all_14_vendor_parity_cells_proven": parity_complete,
        "all_14_vendor_mode_evidence_cells_proven": parity_complete,
        "candidate_mode_orientation_parity_proven": candidate_parity_ready,
        "bounded_proof_queue_complete": proof_queue_complete,
        "frozen_policy_ready": policy_ready,
        "current_family_chain_integrity": chain_integrity,
        "frozen_source_family_integrity": chain_integrity,
        "pre_rerun_gate_ready": not blockers,
        "blocker": ";".join(blockers),
        "rerun_scope": "full_policy_defined_family",
        "promotion_evaluation_scope": "registered_semantic_hypothesis_only",
        "rerun_execution_included": False,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _collapse_pending_family_preflight_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select one evidence-ready venue variant per semantic hypothesis."""

    if frame.empty or "semantic_hypothesis_id" not in frame:
        return frame
    rows: list[dict[str, Any]] = []
    semantic_ids = frame["semantic_hypothesis_id"].map(_text)
    working = frame.assign(_semantic_id=semantic_ids)
    identity_columns = (
        "equivalence_cluster_id",
        "pair",
        "wizard_timeframe",
        "exact_mode",
        "orientation",
    )
    component_columns = (
        "registration_ready",
        "identity_ready",
        "strict_cost_ready",
        "history_ready",
    )
    for semantic_id, group in working.groupby("_semantic_id", sort=False):
        ranked = group.copy()
        ranked["_ready_rank"] = ranked["non_vendor_preflight_ready"].map(_truthy)
        ranked["_component_rank"] = sum(
            ranked[column].map(_truthy).astype(int) for column in component_columns
        )
        ranked["_experiment_rank"] = ranked["experiment_id"].map(_text)
        ranked = ranked.sort_values(
            ["_ready_rank", "_component_rank", "_experiment_rank"],
            ascending=[False, False, True],
            kind="stable",
        )
        selected = ranked.iloc[0].to_dict()
        ready_variants = int(group["non_vendor_preflight_ready"].map(_truthy).sum())
        identity_conflict = any(
            column not in group
            or group[column].map(_text).eq("").any()
            or group[column].map(_text).nunique() != 1
            for column in identity_columns
        )
        blocker = _text(selected.get("blocker"))
        if identity_conflict:
            conflict = "semantic_identity_conflict_across_experiment_variants"
            blocker = ";".join(value for value in (blocker, conflict) if value)
            selected["non_vendor_preflight_ready"] = False
        selected["source_experiment_count"] = len(group)
        selected["ready_experiment_count"] = ready_variants
        selected["alternative_experiment_ids"] = ";".join(
            sorted(
                experiment_id
                for experiment_id in group["experiment_id"].map(_text)
                if experiment_id and experiment_id != _text(selected["experiment_id"])
            )
        )
        selected["representative_selection_reason"] = (
            "IDENTITY_CONFLICT_BLOCKED"
            if identity_conflict
            else "EVIDENCE_READY_VARIANT_SELECTED"
            if ready_variants > 0
            else "BEST_AVAILABLE_VARIANT_BLOCKED"
        )
        selected["blocker"] = blocker
        for column in ("_semantic_id", "_ready_rank", "_component_rank", "_experiment_rank"):
            selected.pop(column, None)
        rows.append(selected)
    return pd.DataFrame(rows)


def _pending_family_preflight_row(
    *,
    candidate: dict[str, Any],
    as_of: datetime,
    queue: pd.DataFrame,
    matrix: pd.DataFrame,
    ledger_audit: pd.DataFrame,
    costs: pd.DataFrame,
    history_results: pd.DataFrame,
    acceptance: dict[str, Any],
    holdout: dict[str, Any],
    history_manifest_bound: bool,
    cost_bundle_valid: bool,
    source_family_valid: bool,
    root: Path,
) -> dict[str, Any]:
    semantic_id = _text(candidate.get("semantic_hypothesis_id"))
    experiment_id = _text(candidate.get("experiment_id"))
    queue_semantic_match = queue.loc[
        queue.get("semantic_hypothesis_id", pd.Series(dtype=str))
        .astype(str)
        .eq(semantic_id)
    ]
    queue_match = queue_semantic_match
    if "experiment_id" in queue and experiment_id:
        queue_match = queue_semantic_match.loc[
            queue_semantic_match["experiment_id"].astype(str).eq(experiment_id)
        ]
    pair_key = (
        _text(queue_match.iloc[0].get("pair_group_key"))
        if len(queue_match) == 1
        else ""
    )
    matrix_match = matrix.loc[
        matrix.get("experiment_id", pd.Series(dtype=str)).astype(str).eq(experiment_id)
    ]
    audit_match = ledger_audit.loc[
        ledger_audit.get("semantic_hypothesis_id", pd.Series(dtype=str))
        .astype(str)
        .eq(semantic_id)
    ]
    cost_match = costs.loc[
        costs.get("pair_group_key", pd.Series(dtype=str)).astype(str).eq(pair_key)
    ]
    history_match = history_results.loc[
        history_results.get("pair_group_key", pd.Series(dtype=str))
        .astype(str)
        .eq(pair_key)
    ]
    audit_row = audit_match.iloc[0] if len(audit_match) == 1 else pd.Series(dtype=object)
    cost_row = cost_match.iloc[0] if len(cost_match) == 1 else pd.Series(dtype=object)
    history_row = (
        history_match.iloc[0] if len(history_match) == 1 else pd.Series(dtype=object)
    )
    history_path = _root_artifact(root, _text(history_row.get("history_path")))
    cost_at = pd.to_datetime(
        cost_row.get("model_as_of_utc"), utc=True, errors="coerce"
    )
    authority_safe = bool(
        not _truthy(candidate.get("execution_authority"))
        and not _truthy(candidate.get("testnet_order_authority"))
        and not _truthy(candidate.get("live_trading_authorized"))
        and not _truthy(audit_row.get("promotion_authority"))
        and not _truthy(audit_row.get("live_trading_authorized"))
    )
    policy_lineage_ready = bool(
        acceptance.get("status") == "PASS"
        and holdout.get("status") == "PASS"
        and _text(audit_row.get("acceptance_policy_id"))
        == _text(acceptance.get("policy_id"))
        and _text(audit_row.get("holdout_policy_id"))
        == _text(holdout.get("policy_id"))
    )
    registration_ready = bool(
        semantic_id
        and len(queue_match) == 1
        and len(audit_match) == 1
        and _truthy(candidate.get("hypothesis_registered_before_next_test"))
        and not _truthy(candidate.get("next_test_executed"))
        and _truthy(audit_row.get("ledger_chain_valid"))
        and authority_safe
        and policy_lineage_ready
    )
    identity_ready = bool(
        len(matrix_match) == 1
        and pair_key
        and _text(matrix_match.iloc[0].get("pair_group_key")) == pair_key
        and _text(matrix_match.iloc[0].get("pair")) == _text(candidate.get("pair"))
        and _text(matrix_match.iloc[0].get("exact_mode"))
        == _text(candidate.get("exact_mode"))
        and _text(matrix_match.iloc[0].get("orientation"))
        == _text(candidate.get("orientation"))
    )
    strict_cost_ready = bool(
        cost_bundle_valid
        and len(cost_match) == 1
        and _truthy(cost_row.get("strict_observed_cost_ready"))
        and _truthy(cost_row.get("cost_acceptance_ready"))
        and _text(cost_row.get("cost_model_status")) == "STRICT_OBSERVED"
        and pd.notna(cost_at)
        and cost_at <= pd.Timestamp(as_of)
        and pd.Timestamp(as_of) - cost_at <= pd.Timedelta(hours=2)
        and all(
            len(_text(cost_row.get(field))) == 64
            for field in (
                "candidate_set_sha256",
                "cost_status_sha256",
                "l2_samples_sha256",
                "fee_profile_sha256",
            )
        )
    )
    history_ready = bool(
        history_manifest_bound
        and len(history_match) == 1
        and _text(history_row.get("history_status"))
        == "READY_FOR_CANONICAL_1X_REPLAY"
        and int(history_row.get("history_rows", 0) or 0) >= 50
        and history_path is not None
        and history_path.is_file()
        and len(_file_hash(history_path)) == 64
    )
    blockers = []
    for passed, blocker in (
        (registration_ready, "registration_policy_or_authority_invalid"),
        (identity_ready, "pending_identity_missing_from_current_source_family"),
        (source_family_valid, "current_source_family_integrity_failed"),
        (strict_cost_ready, "strict_cost_identity_missing_stale_or_unbound"),
        (history_ready, "canonical_history_missing_or_unbound"),
    ):
        if not passed:
            blockers.append(blocker)
    return {
        "schema_version": FAMILY_PREFLIGHT_SCHEMA_VERSION,
        "semantic_hypothesis_id": semantic_id,
        "experiment_id": experiment_id,
        "equivalence_cluster_id": _text(candidate.get("equivalence_cluster_id")),
        "pair_group_key": pair_key,
        "pair": _text(candidate.get("pair")),
        "wizard_timeframe": _text(candidate.get("wizard_timeframe")),
        "exact_mode": _text(candidate.get("exact_mode")),
        "orientation": _text(candidate.get("orientation")),
        "registration_cohort_role": _text(candidate.get("registration_cohort_role")),
        "hypothesis_registered_before_next_test": _truthy(
            candidate.get("hypothesis_registered_before_next_test")
        ),
        "next_test_executed": _truthy(candidate.get("next_test_executed")),
        "execution_authority": _truthy(candidate.get("execution_authority")),
        "registration_ready": registration_ready,
        "authority_boundary_safe": authority_safe,
        "policy_lineage_ready": policy_lineage_ready,
        "identity_ready": identity_ready,
        "source_family_valid": source_family_valid,
        "cost_model_id": _text(cost_row.get("cost_model_id")),
        "cost_model_as_of_utc": _text(cost_row.get("model_as_of_utc")),
        "strict_cost_ready": strict_cost_ready,
        "history_run_id": _text(history_row.get("history_run_id")),
        "history_rows": int(history_row.get("history_rows", 0) or 0),
        "history_path": _text(history_row.get("history_path")),
        "history_sha256": _file_hash(history_path) if history_path is not None else "",
        "history_ready": history_ready,
        "non_vendor_preflight_ready": not blockers,
        "blocker": ";".join(blockers),
        "vendor_evidence_included": False,
        "rerun_execution_included": False,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _root_artifact(root: Path, value: str) -> Path | None:
    if not value:
        return None
    relative = Path(value)
    if relative.is_absolute():
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _resolve_bound_cost_models(
    *, root: Path, active_costs_path: Path, l2_status: dict[str, Any]
) -> tuple[pd.DataFrame, Path | None, bool, bool]:
    """Resolve the immutable cost model used by the registered executor."""

    pointer_path = (
        root / "reports" / "active" / "hyperliquid_pair_cost_bundle_pointer.json"
    )
    if pointer_path.is_file():
        pointer = _read_json(pointer_path)
        pointer_core = {
            key: value for key, value in pointer.items() if key != "receipt_sha256"
        }
        pointer_material = {
            key: value for key, value in pointer_core.items() if key != "pointer_id"
        }
        bundle_id = _text(pointer.get("bundle_id"))
        expected_directory = (
            root / "data" / "research" / "l2_cost_model_receipts" / bundle_id
        )
        bundle_path = _root_artifact(
            root, _text(pointer.get("bundle_manifest_path"))
        )
        model_path = _root_artifact(root, _text(pointer.get("pair_cost_models_path")))
        active_path = _root_artifact(
            root, _text(pointer.get("active_pair_cost_models_path"))
        )
        bundle = _read_json(bundle_path) if bundle_path is not None else {}
        bundle_core = {
            key: value for key, value in bundle.items() if key != "receipt_id"
        }
        input_hashes = bundle.get("input_hashes")
        expected_bundle_id = ""
        if isinstance(input_hashes, dict) and _text(bundle.get("model_as_of_utc")):
            bundle_material = {
                "model_as_of_utc": _text(bundle.get("model_as_of_utc")),
                **input_hashes,
            }
            expected_bundle_id = "l2costbundle_" + sha256(
                json.dumps(bundle_material, sort_keys=True).encode("utf-8")
            ).hexdigest()[:20]
        expected_receipt_id = "l2costreceipt_" + sha256(
            json.dumps(bundle_core, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        pointer_valid = bool(
            pointer.get("schema_version") == "thewiz.l2_cost_model_pointer.v1"
            and bundle_id.startswith("l2costbundle_")
            and bundle_id == expected_bundle_id
            and pointer.get("pointer_id")
            == "l2costpointer_"
            + sha256(_canonical_json(pointer_material).encode("utf-8")).hexdigest()[:20]
            and pointer.get("receipt_sha256")
            == sha256(_canonical_json(pointer_core).encode("utf-8")).hexdigest()
            and bundle_path == expected_directory / "receipt.json"
            and model_path == expected_directory / "pair_cost_models.csv"
            and bundle_path is not None
            and bundle_path.is_file()
            and _file_hash(bundle_path) == _text(pointer.get("bundle_manifest_sha256"))
            and bundle.get("schema_version") == "thewiz.l2_cost_model_receipt.v1"
            and _text(bundle.get("bundle_id")) == bundle_id
            and _text(bundle.get("receipt_id")) == expected_receipt_id
            and _text(bundle.get("pair_cost_models_path"))
            == _text(pointer.get("pair_cost_models_path"))
            and model_path is not None
            and model_path.is_file()
            and _file_hash(model_path) == _text(pointer.get("pair_cost_models_sha256"))
            and _file_hash(model_path) == _text(bundle.get("pair_cost_models_sha256"))
            and active_path == active_costs_path.resolve()
            and active_path is not None
            and active_path.is_file()
            and _file_hash(active_path)
            == _text(pointer.get("active_pair_cost_models_sha256"))
            and _file_hash(active_path) == _file_hash(model_path)
            and pointer.get("promotion_authority") is False
            and pointer.get("testnet_order_authority") is False
            and pointer.get("live_trading_authorized") is False
            and bundle.get("promotion_authority") is False
            and bundle.get("testnet_order_authority") is False
            and bundle.get("live_trading_authorized") is False
        )
        if not pointer_valid:
            return pd.DataFrame(), None, False, False
        costs = _read_csv(model_path)
        bundle_blockers = validate_pair_cost_bundle_artifacts(
            root=root,
            bundle_manifest_path=bundle_path,
            pair_cost_models_path=model_path,
        )
        authority_columns_valid = all(
            column in costs.columns and not costs[column].map(_truthy).any()
            for column in (
                "promotion_authority",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        )
        if bundle_blockers or not authority_columns_valid:
            return pd.DataFrame(), None, False, False
        return costs, model_path, True, True

    bundle_path = _root_artifact(
        root, _text(l2_status.get("pair_cost_bundle_manifest_path"))
    )
    bundle = _read_json(bundle_path) if bundle_path is not None else {}
    model_path = _root_artifact(root, _text(bundle.get("pair_cost_models_path")))
    bundle_valid = bool(
        bundle_path is not None
        and bundle_path.is_file()
        and _file_hash(bundle_path)
        == _text(l2_status.get("pair_cost_bundle_manifest_sha256"))
        and model_path is not None
        and model_path.is_file()
        and _file_hash(model_path) == _text(bundle.get("pair_cost_models_sha256"))
    )
    costs = _read_csv(model_path) if bundle_valid and model_path is not None else pd.DataFrame()
    if bundle_valid and bundle_path is not None and model_path is not None:
        bundle_valid = not validate_pair_cost_bundle_artifacts(
            root=root,
            bundle_manifest_path=bundle_path,
            pair_cost_models_path=model_path,
        )
        bundle_valid = bool(
            bundle_valid
            and not costs.empty
            and all(
                column in costs.columns and not costs[column].map(_truthy).any()
                for column in (
                    "promotion_authority",
                    "testnet_order_authority",
                    "live_trading_authorized",
                )
            )
        )
    if not bundle_valid:
        costs = pd.DataFrame()
        model_path = None
    active_matches = bool(
        model_path is not None
        and active_costs_path.is_file()
        and _file_hash(active_costs_path) == _file_hash(model_path)
    )
    return costs, model_path, bundle_valid, active_matches


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []
        rows.append(payload)
    return rows


def _file_hash(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_or_validate_immutable_json(
    payload: dict[str, Any], path: Path
) -> str:
    expected = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    expected_hash = sha256(expected).hexdigest()
    if path.is_file():
        if _file_hash(path) != expected_hash:
            raise ValueError(f"immutable registered rerun artifact changed: {path}")
        return expected_hash
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(expected)
    temporary.replace(path)
    return expected_hash


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "pass",
        "ready",
    }


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


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


if __name__ == "__main__":
    result = build_registered_rerun_gate()
    print(json.dumps(result.summary, indent=2))
