from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from quant_platform.orchestration.corrective_stage4_handoff_readiness import (
    build_corrective_stage4_handoff_readiness,
    validate_stage4_handoff_readiness_receipt,
)
from quant_platform.orchestration.corrective_wizard_reset_readiness import (
    wizard_reset_readiness_state_sha256,
)
from tests.browser_auth_support import write_browser_auth_binding

NOW = datetime(2026, 8, 11, 16, 30, tzinfo=UTC)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _prepare_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    active = root / "reports" / "active"
    active.mkdir(parents=True)

    source_rows = [
        {"experiment_id": "experiment-active", "pair": "ETH-PYTH"},
        {"experiment_id": "experiment-other", "pair": "ADA-DOGE"},
    ]
    source_probe = root / "source-probe.csv"
    _write_csv(source_probe, source_rows)
    source_sha = sha256(source_probe.read_bytes()).hexdigest()
    source_probe.unlink()
    registered_candidates = [
        {
            "semantic_hypothesis_id": "hypothesis-active",
            "source_experiment_id": "experiment-active",
            "pair_group_key": "dydx|daily|ETH|PYTH",
            "execution_authority_at_contract": False,
            "testnet_order_authority_at_contract": False,
            "live_trading_authorized_at_contract": False,
        }
    ]
    contract_material = {
        "schema_version": "thewiz.corrective_registered_rerun.v1",
        "registered_candidates": registered_candidates,
        "acceptance_policy_id": None,
        "holdout_policy_id": None,
        "discovery_policy_sha256": None,
        "source_family_sha256": source_sha,
        "source_family_rows": 2,
    }
    contract_id = (
        "registeredrerun_"
        + sha256(
            json.dumps(
                contract_material,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:20]
    )
    source_relative = (
        f"data/research/registered_rerun_source_families/{contract_id}/experiment_matrix.csv"
    )
    source_path = root / source_relative
    _write_csv(source_path, source_rows)
    source_receipt = {
        "schema_version": "thewiz.registered_rerun_source_family.v1",
        "contract_id": contract_id,
        "source_family_path": source_relative,
        "source_family_rows": 2,
        "source_family_sha256": source_sha,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    _write_json(source_path.parent / "receipt.json", source_receipt)

    contract_relative = f"data/research/registered_rerun_contracts/{contract_id}.json"
    contract = {
        "schema_version": "thewiz.corrective_registered_rerun.v1",
        "contract_id": contract_id,
        "immutable_contract_path": contract_relative,
        "source_family_sha256": source_sha,
        "source_family_rows": 2,
        "registered_candidates": registered_candidates,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    _write_json(root / contract_relative, contract)
    _write_json(active / "registered_research_rerun_contract.json", contract)

    _write_csv(
        active / "registered_research_rerun_gate.csv",
        [
            {
                "semantic_hypothesis_id": "hypothesis-active",
                "registration_valid": True,
                "authority_boundary_safe": True,
                "strict_cost_ready_after_registration": True,
                "frozen_source_family_identity_match": True,
                "frozen_source_family_integrity": True,
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ],
    )
    _write_json(
        active / "registered_research_rerun_gate.json",
        {
            "schema_version": "thewiz.corrective_registered_rerun.v1",
            "contract_id": contract_id,
            "status": "BLOCKED_VENDOR_PARITY",
            "registered_rerun_executor_available": True,
            "automatic_post_parity_handoff": True,
            "family_rerun_scope": "full_policy_defined_family",
            "promotion_evaluation_scope": "registered_semantic_hypotheses_only",
            "pending_family_rollover_required": True,
            "registered_rerun_results_accounted": False,
            "promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )

    pending_rows = []
    batch_rows = []
    for index, pair in enumerate(("ETH-NEAR", "ADA-DOGE", "ADA-NEAR"), start=1):
        identity = {
            "semantic_hypothesis_id": f"hypothesis-pending-{index}",
            "experiment_id": f"experiment-pending-{index}",
            "equivalence_cluster_id": f"cluster-{index}",
            "pair": pair,
            "wizard_timeframe": "daily",
            "exact_mode": "Static (ZScoreR)",
            "orientation": "original",
            "registration_cohort_role": "walkforward_near_miss",
        }
        pending_rows.append(
            {
                **identity,
                "registration_ready": True,
                "identity_ready": True,
                "strict_cost_ready": True,
                "history_ready": True,
                "non_vendor_preflight_ready": True,
                "blocker": "",
                "vendor_evidence_included": False,
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
                "source_experiment_count": 1,
                "alternative_experiment_ids": "",
            }
        )
        batch_rows.append(identity)
    _write_csv(active / "registered_rerun_family_preflight.csv", pending_rows)
    _write_csv(active / "current_hypothesis_batch.csv", batch_rows)
    _write_json(
        active / "registered_rerun_family_preflight.json",
        {
            "schema_version": "thewiz.registered_rerun_family_preflight.v1",
            "status": "PASS",
            "hypotheses": 3,
            "ready_hypotheses": 3,
            "pairs": 3,
            "independent_clusters": 3,
            "source_family_valid": True,
            "cost_bundle_valid": True,
            "history_manifest_bound": True,
            "vendor_evidence_included": False,
            "rerun_execution_included": False,
            "promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )

    manifest_id = "wizardcapture_0123456789abcdef0123"
    manifest_relative = f"data/research/wizard_capture_manifests/{manifest_id}.json"
    _write_json(root / manifest_relative, {"manifest_id": manifest_id, "calls": 13})
    manifest_sha = sha256((root / manifest_relative).read_bytes()).hexdigest()
    _write_json(
        active / "corrective_wizard_next_capture_manifest.json",
        {
            "schema_version": "thewiz.corrective_wizard_next_capture_manifest.v1",
            "status": "PASS",
            "manifest_enforced": True,
            "manifest_id": manifest_id,
            "immutable_manifest_path": manifest_relative,
            "immutable_manifest_sha256": manifest_sha,
            "pending_calls": 13,
            "candidate_promotion_authority": False,
            "order_submission_included": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    browser_auth_binding = write_browser_auth_binding(
        root,
        captured_at=datetime(2026, 8, 11, 16, 20, tzinfo=UTC),
        required_at=NOW,
    )
    reset_material = {
        "schema_version": "thewiz.corrective_wizard_reset_readiness.v5",
        "status": "PASS_RESET_AUTOMATION_READY",
        "manifest_id": manifest_id,
        "immutable_manifest_path": manifest_relative,
        "immutable_manifest_sha256": manifest_sha,
        "launch_agent_loaded": True,
        "checks_passed": 16,
        "checks_total": 16,
        "scheduler_manifest_continuity_valid": True,
        "scheduler_manifest_continuity_status": ("PASS_PRIOR_UNRESOLVED_COHORT_MATCH"),
        "blockers": [],
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        **browser_auth_binding,
    }
    reset_receipt_id = (
        "wizardresetreadiness_"
        + sha256(
            json.dumps(reset_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
    )
    reset = {**reset_material, "receipt_id": reset_receipt_id}
    _write_json(active / "wizard_reset_readiness.json", reset)
    _write_json(
        root / "data" / "research" / "wizard_reset_readiness" / f"{reset_receipt_id}.json",
        reset,
    )

    protocol_id = "stage5protocol_0123456789abcdef0123"
    protocol_relative = f"data/research/registered_stage5_protocols/{protocol_id}.json"
    _write_json(root / protocol_relative, {"protocol_id": protocol_id, "frozen": True})
    protocol_sha = sha256((root / protocol_relative).read_bytes()).hexdigest()
    _write_json(
        active / "registered_stage5_protocol.json",
        {
            "schema_version": "thewiz.registered_stage5_protocol_pointer.v1",
            "status": "PASS_PROSPECTIVE_PROTOCOL_REGISTERED",
            "protocol_id": protocol_id,
            "protocol_receipt_path": protocol_relative,
            "protocol_receipt_sha256": protocol_sha,
            "research_only": True,
            "thresholds_changed_after_results": False,
            "promotion_authority": False,
            "testnet_candidate_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    _write_json(
        active / "corrective_wizard_proof_scheduler_status.json",
        {
            "schema_version": "thewiz.corrective_wizard_proof_scheduler.v1",
            "capture_manifest_enforced": True,
            "capture_manifest_id": manifest_id,
            "capture_manifest_immutable_path": manifest_relative,
            "capture_manifest_immutable_sha256": manifest_sha,
            "capture_manifest_candidate_id": manifest_id,
            "capture_manifest_candidate_immutable_path": manifest_relative,
            "capture_manifest_candidate_immutable_sha256": manifest_sha,
            "capture_manifest_candidate_binding_valid": True,
            "capture_manifest_candidate_source_binding_valid": True,
            "capture_manifest_continuity_status": ("PASS_PRIOR_UNRESOLVED_COHORT_MATCH"),
            "capture_manifest_continuity_valid": True,
            "capture_manifest_drift_detected": False,
            "research_only": True,
            "candidate_promotion_authority": False,
            "order_submission_included": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    return root


def _validate_protocol_fixture(*, root: Path):
    pointer = _read_json(root / "reports" / "active" / "registered_stage5_protocol.json")
    receipt_path = root / str(pointer["protocol_receipt_path"])
    receipt = _read_json(receipt_path)
    if receipt.get("protocol_id") != pointer.get("protocol_id"):
        raise ValueError("fixture Stage 5 protocol identity mismatch")
    return receipt_path, receipt


def _run(root: Path, *, protocol_validator=_validate_protocol_fixture):
    return build_corrective_stage4_handoff_readiness(
        root=root,
        now=NOW,
        stage5_protocol_validator=protocol_validator,
    )


def _refresh_reset(root: Path, **updates: object) -> dict[str, object]:
    active = root / "reports" / "active" / "wizard_reset_readiness.json"
    reset = _read_json(active)
    reset.pop("receipt_id", None)
    reset.update(updates)
    receipt_id = (
        "wizardresetreadiness_"
        + sha256(
            json.dumps(reset, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
    )
    refreshed = {**reset, "receipt_id": receipt_id}
    _write_json(active, refreshed)
    _write_json(
        root / "data" / "research" / "wizard_reset_readiness" / f"{receipt_id}.json",
        refreshed,
    )
    return refreshed


def test_stage4_handoff_readiness_passes_without_authority(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)

    result = _run(root)

    assert result.summary["status"] == "PASS_STAGE4_HANDOFF_READY"
    assert result.summary["handoff_state"] == ("READY_AWAITING_STAGE3_THEN_ACTIVE_CONTRACT")
    assert result.summary["checks_passed"] == result.summary["checks_total"] == 15
    assert result.summary["pending_family_independent_clusters"] == 3
    assert result.summary["pending_family_semantic_batch_coverage"] == 3
    assert result.summary["active_pending_semantic_overlap"] == 0
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    assert result.paths["immutable_receipt"].is_file()
    validation = validate_stage4_handoff_readiness_receipt(root=root)
    assert validation["status"] == "PASS"
    assert validation["receipt_id"] == result.summary["receipt_id"]
    returned_validation = validate_stage4_handoff_readiness_receipt(
        root=root,
        receipt=result.summary,
    )
    assert returned_validation["status"] == "PASS"
    assert returned_validation["receipt_id"] == result.summary["receipt_id"]


def test_stage4_handoff_readiness_is_idempotent_at_same_time(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)

    first = _run(root)
    second = _run(root)

    assert first.summary["receipt_id"] == second.summary["receipt_id"]
    assert first.paths["immutable_receipt"] == second.paths["immutable_receipt"]


def test_stage4_handoff_validator_accepts_equivalent_reset_observation_refresh(
    tmp_path: Path,
) -> None:
    root = _prepare_root(tmp_path)
    result = _run(root)
    original_state = result.summary["stage3_reset_state_sha256"]

    refreshed = _refresh_reset(
        root,
        checked_at_utc="2026-08-11T16:31:00+00:00",
        launcher_heartbeat_age_seconds=42.0,
        launcher_receipt_id="wizardlauncher_refreshed",
        launcher_receipt_path=(
            "data/research/wizard_proof_launcher_receipts/2026-08-11/"
            "wizardlauncher_refreshed.json"
        ),
        launcher_receipt_sha256="b" * 64,
    )
    validation = validate_stage4_handoff_readiness_receipt(root=root)

    assert wizard_reset_readiness_state_sha256(refreshed) == original_state
    assert validation["status"] == "PASS"
    assert validation["current_stage3_reset_receipt_id"] == refreshed["receipt_id"]


def test_stage4_handoff_validator_blocks_current_reset_safety_regression(
    tmp_path: Path,
) -> None:
    root = _prepare_root(tmp_path)
    _run(root)
    refreshed = _refresh_reset(
        root,
        status="BLOCKED_RESET_AUTOMATION",
        checks_passed=15,
        blockers=["wizard_proof_launch_agent_not_loaded"],
        launch_agent_loaded=False,
    )

    validation = validate_stage4_handoff_readiness_receipt(root=root)

    assert validation["status"] == "BLOCKED"
    assert validation["current_stage3_reset_receipt_id"] == refreshed["receipt_id"]
    assert "stage4_handoff_current_stage3_reset_not_ready" in validation["blockers"]
    assert "stage4_handoff_current_stage3_reset_state_mismatch" in validation["blockers"]


def test_stage4_handoff_builder_blocks_reset_regression_during_audit(
    tmp_path: Path,
) -> None:
    root = _prepare_root(tmp_path)

    def regress_reset_during_protocol_validation(*, root: Path):
        validated = _validate_protocol_fixture(root=root)
        _refresh_reset(
            root,
            status="BLOCKED_RESET_AUTOMATION",
            checks_passed=15,
            blockers=["wizard_proof_launch_agent_not_loaded"],
            launch_agent_loaded=False,
        )
        return validated

    result = _run(root, protocol_validator=regress_reset_during_protocol_validation)

    assert result.summary["status"] == "BLOCKED_STAGE4_HANDOFF"
    assert "stage3_reset_current_state_changed_during_handoff_audit" in result.summary["blockers"]


def test_stage4_handoff_blocks_tampered_contract(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    active_contract = root / "reports" / "active" / "registered_research_rerun_contract.json"
    contract = _read_json(active_contract)
    contract["source_family_rows"] = 999
    _write_json(active_contract, contract)

    result = _run(root)

    assert "active_registered_contract_binding_invalid" in result.summary["blockers"]


def test_stage4_handoff_blocks_tampered_source_family(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    source = next(
        (root / "data" / "research" / "registered_rerun_source_families").glob(
            "*/experiment_matrix.csv"
        )
    )
    source.write_text("experiment_id,pair\ntampered,X-Y\n", encoding="utf-8")

    result = _run(root)

    assert "active_registered_source_family_binding_invalid" in result.summary["blockers"]


def test_stage4_handoff_blocks_active_candidate_without_strict_cost(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    gate_rows = root / "reports" / "active" / "registered_research_rerun_gate.csv"
    rows = _read_csv_rows(gate_rows)
    rows[0]["strict_cost_ready_after_registration"] = "False"
    _write_csv(gate_rows, rows)

    result = _run(root)

    assert "active_registered_candidate_non_vendor_inputs_invalid" in result.summary["blockers"]


def test_stage4_handoff_blocks_insufficient_independent_breadth(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    path = root / "reports" / "active" / "registered_rerun_family_preflight.json"
    preflight = _read_json(path)
    preflight["independent_clusters"] = 2
    _write_json(path, preflight)

    result = _run(root)

    assert "pending_registered_family_preflight_invalid" in result.summary["blockers"]


def test_stage4_handoff_blocks_tampered_stage3_manifest(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    manifest = _read_json(
        root / "reports" / "active" / "corrective_wizard_next_capture_manifest.json"
    )
    (root / str(manifest["immutable_manifest_path"])).write_text("tampered\n", encoding="utf-8")

    result = _run(root)

    assert "stage3_frozen_capture_manifest_binding_invalid" in result.summary["blockers"]


def test_stage4_handoff_blocks_reset_binding_drift(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    path = root / "reports" / "active" / "wizard_reset_readiness.json"
    reset = _read_json(path)
    reset["manifest_id"] = "wizardcapture_different"
    _write_json(path, reset)

    result = _run(root)

    assert "stage3_reset_automation_not_bound_or_ready" in result.summary["blockers"]


def test_stage4_handoff_blocks_rehashed_mutable_reset_without_immutable_twin(
    tmp_path: Path,
) -> None:
    root = _prepare_root(tmp_path)
    path = root / "reports" / "active" / "wizard_reset_readiness.json"
    reset = _read_json(path)
    reset["checks_total"] = 17
    reset["checks_passed"] = 17
    material = {key: value for key, value in reset.items() if key != "receipt_id"}
    reset["receipt_id"] = (
        "wizardresetreadiness_"
        + sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
    )
    _write_json(path, reset)

    result = _run(root)

    assert "stage3_reset_automation_not_bound_or_ready" in result.summary["blockers"]


def test_stage4_handoff_blocks_generation_sequence_mismatch(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    path = root / "reports" / "active" / "registered_research_rerun_gate.json"
    gate = _read_json(path)
    gate["pending_family_rollover_required"] = False
    _write_json(path, gate)

    result = _run(root)

    assert "registered_stage4_generation_sequence_invalid" in result.summary["blockers"]


def test_stage4_handoff_blocks_active_pending_semantic_overlap(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    active = root / "reports" / "active"
    for name in (
        "registered_rerun_family_preflight.csv",
        "current_hypothesis_batch.csv",
    ):
        path = active / name
        rows = _read_csv_rows(path)
        rows[0]["semantic_hypothesis_id"] = "hypothesis-active"
        _write_csv(path, rows)

    result = _run(root)

    assert (
        "registered_stage4_active_pending_identity_partition_invalid" in result.summary["blockers"]
    )
    assert result.summary["active_pending_semantic_overlap"] == 1


def test_stage4_handoff_blocks_pending_batch_omission(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    path = root / "reports" / "active" / "current_hypothesis_batch.csv"
    rows = _read_csv_rows(path)
    rows.append(
        {
            **rows[0],
            "semantic_hypothesis_id": "hypothesis-omitted",
            "experiment_id": "experiment-omitted",
            "equivalence_cluster_id": "cluster-omitted",
            "pair": "SOL-NEAR",
        }
    )
    _write_csv(path, rows)

    result = _run(root)

    assert (
        "registered_stage4_active_pending_identity_partition_invalid" in result.summary["blockers"]
    )


def test_stage4_handoff_blocks_unrepresented_batch_experiment_variant(
    tmp_path: Path,
) -> None:
    root = _prepare_root(tmp_path)
    path = root / "reports" / "active" / "current_hypothesis_batch.csv"
    rows = _read_csv_rows(path)
    rows.append({**rows[0], "experiment_id": "experiment-pending-1-alternative"})
    _write_csv(path, rows)

    result = _run(root)

    assert (
        "registered_stage4_active_pending_identity_partition_invalid" in result.summary["blockers"]
    )


def test_stage4_handoff_blocks_pending_representative_drift(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    path = root / "reports" / "active" / "registered_rerun_family_preflight.csv"
    rows = _read_csv_rows(path)
    rows[0]["experiment_id"] = "experiment-not-in-batch"
    _write_csv(path, rows)

    result = _run(root)

    assert (
        "registered_stage4_active_pending_identity_partition_invalid" in result.summary["blockers"]
    )


def test_stage4_handoff_blocks_forged_active_and_immutable_contract(
    tmp_path: Path,
) -> None:
    root = _prepare_root(tmp_path)
    active_path = root / "reports" / "active" / "registered_research_rerun_contract.json"
    contract = _read_json(active_path)
    contract["acceptance_policy_id"] = "forged-policy"
    _write_json(active_path, contract)
    _write_json(root / str(contract["immutable_contract_path"]), contract)

    result = _run(root)

    assert "active_registered_contract_binding_invalid" in result.summary["blockers"]


def test_stage4_handoff_blocks_tampered_stage5_protocol(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    protocol = _read_json(root / "reports" / "active" / "registered_stage5_protocol.json")
    (root / str(protocol["protocol_receipt_path"])).write_text("tampered\n", encoding="utf-8")

    result = _run(root)

    assert "prospective_stage5_protocol_binding_invalid" in result.summary["blockers"]


def test_stage4_handoff_blocks_stage5_current_binding_drift(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)

    def drifted_protocol(**_):
        raise ValueError("registered Stage 5 protocol current binding mismatch")

    result = _run(root, protocol_validator=drifted_protocol)

    assert "prospective_stage5_protocol_binding_invalid" in result.summary["blockers"]


def test_stage4_handoff_blocks_scheduler_authority_leak(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    path = root / "reports" / "active" / "corrective_wizard_proof_scheduler_status.json"
    scheduler = _read_json(path)
    scheduler["testnet_order_authority"] = True
    _write_json(path, scheduler)

    result = _run(root)

    assert "stage4_handoff_authority_violation" in result.summary["blockers"]


def test_stage4_handoff_rejects_immutable_receipt_collision(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    first = _run(root)
    first.paths["immutable_receipt"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="immutable Stage 4 readiness receipt collision"):
        _run(root)


def test_stage4_handoff_validator_rejects_rehashed_mutable_pass(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    _run(root)
    path = root / "reports" / "active" / "stage4_handoff_readiness.json"
    receipt = _read_json(path)
    receipt["checks_total"] = 16
    receipt["checks_passed"] = 16
    material = {key: value for key, value in receipt.items() if key != "receipt_id"}
    receipt["receipt_id"] = (
        "stage4handoff_"
        + sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
    )
    _write_json(path, receipt)

    validation = validate_stage4_handoff_readiness_receipt(root=root)

    assert validation["status"] == "BLOCKED"
    assert "stage4_handoff_immutable_receipt_missing" in validation["blockers"]


def test_stage4_handoff_validator_blocks_stale_registered_gate(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    _run(root)
    path = root / "reports" / "active" / "registered_research_rerun_gate.json"
    gate = _read_json(path)
    gate["status"] = "READY_FOR_NEXT_SCHEDULED_RESEARCH_RERUN"
    _write_json(path, gate)

    validation = validate_stage4_handoff_readiness_receipt(root=root)

    assert validation["status"] == "BLOCKED"
    assert "stage4_handoff_source_binding_invalid:active_gate" in validation["blockers"]


def test_stage4_handoff_validator_blocks_stale_pending_orientation(
    tmp_path: Path,
) -> None:
    root = _prepare_root(tmp_path)
    _run(root)
    path = root / "reports" / "active" / "current_hypothesis_batch.csv"
    rows = _read_csv_rows(path)
    rows[0]["orientation"] = "reverse"
    _write_csv(path, rows)

    validation = validate_stage4_handoff_readiness_receipt(root=root)

    assert validation["status"] == "BLOCKED"
    assert (
        "stage4_handoff_source_binding_invalid:current_hypothesis_batch" in validation["blockers"]
    )


def test_stage4_handoff_validator_blocks_original_reset_receipt_tamper(
    tmp_path: Path,
) -> None:
    root = _prepare_root(tmp_path)
    result = _run(root)
    path = (
        root
        / "data"
        / "research"
        / "wizard_reset_readiness"
        / f"{result.summary['stage3_reset_receipt_id']}.json"
    )
    path.write_text("{}\n", encoding="utf-8")

    validation = validate_stage4_handoff_readiness_receipt(root=root)

    assert validation["status"] == "BLOCKED"
    assert "stage4_handoff_source_binding_invalid:immutable_stage3_reset" in validation["blockers"]
