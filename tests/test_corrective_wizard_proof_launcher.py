from __future__ import annotations

import inspect
import json
import subprocess
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest

import quant_platform.orchestration.corrective_wizard_proof_launcher as launcher
from quant_platform.orchestration.corrective_external_effects import (
    current_external_effect_issuer,
    external_effect_issuer_session,
)
from quant_platform.orchestration.corrective_wizard_capture_manifest import (
    _load_or_create_source_receipt,
)
from quant_platform.orchestration.corrective_wizard_proof_launcher import (
    HEAVY_MODULE,
    _capture_manifest_binding_complete,
    _launcher_exit_code,
    _ou_v4_scheduler_evidence_complete,
    _ou_v5_scheduler_evidence_complete,
    _ou_v6_scheduler_evidence_complete,
    _run_heavy_scheduler_in_process,
    _stage3_evidence_complete,
    _stage3_local_evidence_fingerprint,
    _stage4_active_execution_continuation_complete,
    _terminal_learning_status_valid,
    latest_verified_immutable_scheduler_execution,
    run_wizard_proof_launcher,
)
from quant_platform.orchestration.effect_authority import (
    PHASE00_WIZARD_RESEARCH_PROFILE,
    EffectAuthority,
)
from tests.capture_reconciliation_support import (
    write_valid_capture_reconciliation_evidence,
)

NOW = datetime(2026, 8, 10, 12, tzinfo=UTC)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _status(root, *, attempted):
    path = root / "reports" / "active" / "corrective_wizard_proof_scheduler_status.json"
    _write_json(
        path,
        {
            "attempt_date_utc": "2026-08-10",
            "external_attempt_made": attempted,
            "next_external_attempt_eligible_at": "2026-08-11T00:00:00+00:00",
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    return path


def _capture_reconciliation_complete_fields(root=None):
    if root is not None:
        return write_valid_capture_reconciliation_evidence(root)

    manifest_id = "wizardcapture_test"
    manifest_path = "data/research/wizard_capture_manifests/wizardcapture_test.json"
    manifest_sha256 = "a" * 64
    source_receipt_id = "wizardcapturesources_test"
    source_receipt_path = "data/research/wizard_capture_manifest_sources/wizardcapture_test.json"
    source_receipt_sha256 = "b" * 64
    source_artifacts_sha256 = "c" * 64
    if root is not None:
        immutable_manifest = root / manifest_path
        _write_json(immutable_manifest, {"manifest_id": manifest_id, "calls": 13})
        manifest_sha256 = sha256(immutable_manifest.read_bytes()).hexdigest()
        source_path = root / "reports" / "active" / "launcher_fixture_source.csv"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text("call_id\ncall-1\n", encoding="utf-8")
        source_receipt, immutable_source_receipt, _ = _load_or_create_source_receipt(
            root=root,
            manifest_id=manifest_id,
            immutable_manifest_path=immutable_manifest,
            source_artifacts=[
                {
                    "path": str(source_path.relative_to(root)),
                    "sha256": sha256(source_path.read_bytes()).hexdigest(),
                }
            ],
            retrofit=False,
        )
        source_receipt_id = source_receipt["receipt_id"]
        source_receipt_path = str(immutable_source_receipt.relative_to(root))
        source_receipt_sha256 = sha256(immutable_source_receipt.read_bytes()).hexdigest()
        source_artifacts_sha256 = source_receipt["source_artifacts_sha256"]
    return {
        "capture_manifest_enforced": True,
        "capture_manifest_id": manifest_id,
        "capture_manifest_immutable_path": manifest_path,
        "capture_manifest_immutable_sha256": manifest_sha256,
        "capture_manifest_candidate_binding_valid": True,
        "capture_manifest_candidate_source_binding_valid": True,
        "capture_manifest_source_receipt_id": source_receipt_id,
        "capture_manifest_source_receipt_path": source_receipt_path,
        "capture_manifest_source_receipt_sha256": source_receipt_sha256,
        "capture_manifest_source_artifacts_sha256": source_artifacts_sha256,
        "capture_manifest_candidate_id": manifest_id,
        "capture_manifest_candidate_immutable_path": manifest_path,
        "capture_manifest_candidate_immutable_sha256": manifest_sha256,
        "capture_manifest_accounting_valid": True,
        "capture_manifest_continuity_valid": True,
        "capture_manifest_drift_detected": False,
        "capture_reconciliation_status": "PASS",
        "capture_reconciliation_valid": True,
        "capture_reconciliation_complete": True,
        "capture_reconciliation_required_calls": 13,
        "capture_reconciliation_completed_calls": 13,
        "capture_reconciliation_pending_calls": 0,
        "capture_reconciliation_blocked_calls": 0,
        "capture_reconciliation_manifest_id": manifest_id,
        "capture_reconciliation_manifest_path": manifest_path,
        "capture_reconciliation_manifest_sha256": manifest_sha256,
    }


def _python(root):
    path = root / ".venv" / "bin" / "python3"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_production_heavy_runner_requires_supervisor_effect_issuer(tmp_path):
    python = _python(tmp_path)
    command = [str(python), "-m", HEAVY_MODULE, "--execute"]

    with pytest.raises(RuntimeError, match="effect_issuer_missing"):
        _run_heavy_scheduler_in_process(command, cwd=tmp_path)


def test_production_heavy_runner_preserves_supervisor_effect_issuer(
    tmp_path,
    monkeypatch,
):
    python = _python(tmp_path)
    observed = {}

    def fake_cycle(**kwargs):
        observed["issuer"] = current_external_effect_issuer()
        observed["kwargs"] = kwargs

    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_wizard_proof_scheduler."
        "run_corrective_wizard_proof_cycle",
        fake_cycle,
    )
    authority = EffectAuthority(
        root=tmp_path,
        secret=b"wizard-launcher-test-authority-32b",
        issuer_id="wizard-launcher-test-supervisor",
        profile=PHASE00_WIZARD_RESEARCH_PROFILE,
    )
    with external_effect_issuer_session(
        authority=authority,
        run_id="wizard-launcher-test-run",
        intended_slot_id="wizard-launcher-test-slot",
        source_fingerprint_sha256="a" * 64,
        runtime_fingerprint_sha256="b" * 64,
        configuration_fingerprint_sha256="c" * 64,
        provider_id="crypto_wizards",
        account_scope_id="wizard-research-test-account",
        allowed_targets=frozenset(
            {"https://api.cryptowizards.net/v1beta/backtest"}
        ),
        allowed_credential_keys=frozenset({"CRYPTO_WIZARDS_API_KEY"}),
        max_total_requests=10,
        max_total_credits=20,
    ) as issuer:
        command = [
            str(python),
            "-m",
            HEAVY_MODULE,
            "--execute",
            "--force",
            "--internal-continuation-only",
        ]
        completed = _run_heavy_scheduler_in_process(command, cwd=tmp_path)

    assert completed.returncode == 0
    assert completed.args == command
    assert observed["issuer"] is issuer
    assert observed["issuer"].provider_id == "crypto_wizards"
    assert observed["issuer"].authority.profile.name == (
        "PHASE00_WIZARD_RESEARCH_NO_ORDER"
    )
    assert observed["kwargs"] == {
        "root": tmp_path.resolve(),
        "execute": True,
        "force": True,
        "internal_continuation_only": True,
    }


def test_stage3_completion_requires_exact_immutable_manifest_binding():
    receipt = {
        "queue_eligible": 28,
        "responses_captured_after": 28,
        "accepted_mode_evidence_cells": 28,
        "parity_refresh_status": "PASS",
        "credit_reconciliation_status": "PASS_RECONCILED",
        **_capture_reconciliation_complete_fields(),
    }

    assert _capture_manifest_binding_complete(receipt) is True
    assert _stage3_evidence_complete(receipt) is True
    for field in (
        "capture_manifest_id",
        "capture_manifest_immutable_path",
        "capture_manifest_immutable_sha256",
        "capture_manifest_candidate_binding_valid",
        "capture_manifest_candidate_source_binding_valid",
        "capture_manifest_source_receipt_id",
        "capture_manifest_source_receipt_path",
        "capture_manifest_source_receipt_sha256",
        "capture_manifest_source_artifacts_sha256",
        "capture_reconciliation_manifest_id",
        "capture_reconciliation_manifest_path",
        "capture_reconciliation_manifest_sha256",
    ):
        tampered = dict(receipt)
        tampered.pop(field)
        assert _stage3_evidence_complete(tampered) is False

    mismatched = dict(receipt)
    mismatched["capture_reconciliation_manifest_sha256"] = "b" * 64
    assert _stage3_evidence_complete(mismatched) is False


def test_v6_scheduler_gate_supersedes_v5_and_enforces_terminal_policy():
    waiting = {
        "ou_v6_prospectively_registered": True,
        "ou_v6_required_responses": 8,
        "ou_v6_responses_available": 0,
        "ou_v6_holdout_status": "PLANNED",
        "ou_v6_evaluation_status": "WAITING_VENDOR_RESPONSES",
        "ou_v6_research_only": True,
        "ou_v6_final_successor_iteration": True,
        "ou_v6_successor_after_failure_allowed": False,
        "ou_v6_terminal_failure": False,
    }
    assert _ou_v5_scheduler_evidence_complete(waiting)
    assert not _ou_v6_scheduler_evidence_complete(waiting)

    complete = {
        **waiting,
        "ou_v6_responses_available": 8,
        "ou_v6_holdout_status": "COMPLETE",
        "ou_v6_evaluation_status": "PASS",
        "ou_v6_response_accounting_valid": True,
        "ou_v6_activation_status": "APPLIED_RESEARCH_COMPARATOR_ONLY",
        "ou_v6_comparator_generation": 6,
        "ou_v6_proof_refresh_status": "PASS",
        "ou_v6_proofs_refreshed": 8,
        "ou_v6_activation_automatic": False,
    }
    assert _ou_v6_scheduler_evidence_complete(complete)

    terminal_failure = {**complete, "ou_v6_terminal_failure": True}
    assert not _ou_v6_scheduler_evidence_complete(terminal_failure)


def test_v6_final_successor_skips_superseded_ou_refresh_imports(
    tmp_path, monkeypatch
):
    latest = {
        "ou_v6_prospectively_registered": True,
        "ou_v6_final_successor_iteration": True,
        "ou_v6_successor_after_failure_allowed": False,
        "ou_v6_research_only": True,
        "ou_v6_activation_automatic": False,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    monkeypatch.setattr(launcher, "_dynamic_v2_refresh_needed", lambda **_: False)
    monkeypatch.setattr(launcher, "_stage3_local_reconciliation_needed", lambda **_: False)
    monkeypatch.setattr(launcher, "_stage3_evidence_complete", lambda *_, **__: False)

    def superseded_refresh(**_):
        raise AssertionError("superseded OU activation stack was loaded")

    monkeypatch.setattr(launcher, "_ou_v5_refresh_needed", superseded_refresh)
    monkeypatch.setattr(launcher, "_ou_v4_refresh_needed", superseded_refresh)
    monkeypatch.setattr(launcher, "_ou_v3_refresh_needed", superseded_refresh)

    reason = launcher._internal_registered_continuation_reason(
        root=tmp_path,
        latest=latest,
        same_day_attempt_state="ATTEMPTED",
    )

    assert reason == ""


def test_stage3_completion_reverifies_immutable_source_snapshots(tmp_path):
    receipt = {
        "queue_eligible": 28,
        "responses_captured_after": 28,
        "accepted_mode_evidence_cells": 28,
        "parity_refresh_status": "PASS",
        "credit_reconciliation_status": "PASS_RECONCILED",
        **_capture_reconciliation_complete_fields(tmp_path),
    }
    assert _stage3_evidence_complete(receipt, root=tmp_path)

    source_receipt = json.loads(
        (tmp_path / receipt["capture_manifest_source_receipt_path"]).read_text()
    )
    snapshot = tmp_path / source_receipt["source_artifacts"][0]["snapshot_path"]
    snapshot.chmod(0o600)
    snapshot.write_text("tampered\n", encoding="utf-8")

    assert not _capture_manifest_binding_complete(receipt, root=tmp_path)
    assert not _stage3_evidence_complete(receipt, root=tmp_path)


def test_stage3_completion_refuses_omitted_current_ou_v5_registration(tmp_path):
    receipt = {
        "queue_eligible": 28,
        "responses_captured_after": 28,
        "accepted_mode_evidence_cells": 28,
        "parity_refresh_status": "PASS",
        "credit_reconciliation_status": "PASS_RECONCILED",
        **_capture_reconciliation_complete_fields(tmp_path),
    }
    contract = tmp_path / "config" / "wizard_ou_comparator_v5_holdout.json"
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text("{}\n", encoding="utf-8")

    assert not _stage3_evidence_complete(receipt, root=tmp_path)


def test_stage3_completion_requires_registered_ou_v4_to_pass():
    complete = {
        "ou_v4_prospectively_registered": True,
        "ou_v4_required_responses": 8,
        "ou_v4_responses_available": 8,
        "ou_v4_holdout_status": "COMPLETE",
        "ou_v4_evaluation_status": "PASS",
        "ou_v4_response_accounting_valid": True,
        "ou_v4_automatic_activation": False,
        "ou_v4_research_only": True,
    }
    assert _ou_v4_scheduler_evidence_complete(complete)
    assert not _ou_v4_scheduler_evidence_complete({**complete, "ou_v4_responses_available": 7})
    assert not _ou_v4_scheduler_evidence_complete({**complete, "ou_v4_evaluation_status": "FAIL"})

    receipt = {
        "queue_eligible": 28,
        "responses_captured_after": 28,
        "accepted_mode_evidence_cells": 28,
        "parity_refresh_status": "PASS",
        "credit_reconciliation_status": "PASS_RECONCILED",
        **_capture_reconciliation_complete_fields(),
        **complete,
    }
    assert _stage3_evidence_complete(receipt)
    assert not _stage3_evidence_complete({**receipt, "ou_v4_evaluation_status": "FAIL"})


def test_stage3_completion_requires_registered_ou_v5_to_pass():
    complete = {
        "ou_v5_prospectively_registered": True,
        "ou_v5_required_responses": 8,
        "ou_v5_responses_available": 8,
        "ou_v5_holdout_status": "COMPLETE",
        "ou_v5_evaluation_status": "PASS",
        "ou_v5_response_accounting_valid": True,
        "ou_v5_activation_status": "APPLIED_RESEARCH_COMPARATOR_ONLY",
        "ou_v5_comparator_generation": 5,
        "ou_v5_proof_refresh_status": "PASS",
        "ou_v5_proofs_refreshed": 8,
        "ou_v5_activation_automatic": False,
        "ou_v5_research_only": True,
    }
    assert _ou_v5_scheduler_evidence_complete(complete)
    assert not _ou_v5_scheduler_evidence_complete({**complete, "ou_v5_responses_available": 7})
    assert not _ou_v5_scheduler_evidence_complete({**complete, "ou_v5_evaluation_status": "FAIL"})
    assert not _ou_v5_scheduler_evidence_complete(
        {**complete, "ou_v5_activation_status": "READY_REQUIRES_EXPLICIT_APPLY"}
    )
    assert not _ou_v5_scheduler_evidence_complete(
        {**complete, "ou_v5_proof_refresh_status": "NOT_ACTIVE"}
    )
    assert _ou_v4_scheduler_evidence_complete(
        {
            "ou_v5_prospectively_registered": True,
            "ou_v4_prospectively_registered": True,
            "ou_v4_evaluation_status": "FAIL",
        }
    )

    receipt = {
        "queue_eligible": 28,
        "responses_captured_after": 28,
        "accepted_mode_evidence_cells": 28,
        "parity_refresh_status": "PASS",
        "credit_reconciliation_status": "PASS_RECONCILED",
        **_capture_reconciliation_complete_fields(),
        **complete,
    }
    assert _stage3_evidence_complete(receipt)
    assert not _stage3_evidence_complete({**receipt, "ou_v5_evaluation_status": "FAIL"})


def _scheduler_receipt(
    root,
    *,
    now=NOW,
    status="COMPLETE_ACCEPTED_MODE_EVIDENCE",
    execute=True,
    force=False,
    internal_continuation_only=False,
    external_attempt=True,
    stage3_complete=True,
    registered_rerun_status="PASS_REGISTERED_RERUN_ACCOUNTED",
    learning_handoff_status="NOT_EVALUATED",
    stage5_research_gate_pass=False,
    dynamic_refresh_status="NOT_ACTIVE",
    dynamic_generation=1,
    dynamic_refreshed=0,
    ou_refresh_status="NOT_ACTIVE",
    ou_generation=1,
    ou_refreshed=0,
    ou_v4_refresh_status="NOT_ACTIVE",
    ou_v4_generation=1,
    ou_v4_refreshed=0,
    ou_v5_refresh_status="NOT_ACTIVE",
    ou_v5_generation=1,
    ou_v5_refreshed=0,
    registered_rerun_receipt_path="",
    responses_captured=None,
    accepted_mode_evidence=None,
    capture_reconciliation_status="PASS",
    capture_reconciliation_valid=True,
    capture_reconciliation_complete=True,
    capture_reconciliation_required_calls=None,
    capture_reconciliation_completed_calls=None,
    capture_reconciliation_pending_calls=0,
    capture_reconciliation_blocked_calls=0,
    local_evidence_fingerprint="",
    blockers=None,
):
    capture_evidence = _capture_reconciliation_complete_fields(root)
    payload = {
        "schema_version": "thewiz.corrective_wizard_proof_scheduler.v1",
        "attempt_date_utc": now.date().isoformat(),
        "started_at_utc": now.isoformat(),
        "status": status,
        "execution_requested": execute,
        "force_requested": force,
        "internal_continuation_only": internal_continuation_only,
        "external_attempt_made": external_attempt,
        "external_attempt_made_this_cycle": external_attempt,
        "queue_eligible": 28,
        "responses_captured_after": (
            responses_captured if responses_captured is not None else (28 if stage3_complete else 8)
        ),
        "accepted_mode_evidence_cells": (
            accepted_mode_evidence
            if accepted_mode_evidence is not None
            else (28 if stage3_complete else 4)
        ),
        "parity_refresh_status": "PASS" if stage3_complete else "BLOCKED",
        "credit_reconciliation_status": "PASS_RECONCILED",
        **capture_evidence,
        "capture_manifest_accounting_valid": True,
        "capture_manifest_continuity_valid": True,
        "capture_manifest_drift_detected": False,
        "capture_reconciliation_status": capture_reconciliation_status,
        "capture_reconciliation_valid": capture_reconciliation_valid,
        "capture_reconciliation_complete": capture_reconciliation_complete,
        "capture_reconciliation_required_calls": (
            capture_evidence["capture_reconciliation_required_calls"]
            if capture_reconciliation_required_calls is None
            else capture_reconciliation_required_calls
        ),
        "capture_reconciliation_completed_calls": (
            capture_evidence["capture_reconciliation_completed_calls"]
            if capture_reconciliation_completed_calls is None
            else capture_reconciliation_completed_calls
        ),
        "capture_reconciliation_pending_calls": capture_reconciliation_pending_calls,
        "capture_reconciliation_blocked_calls": capture_reconciliation_blocked_calls,
        "registered_rerun_status": registered_rerun_status,
        "registered_rerun_receipt_path": registered_rerun_receipt_path,
        "registered_learning_handoff_status": learning_handoff_status,
        "registered_stage5_research_gate_pass": stage5_research_gate_pass,
        "registered_learning_handoff_blocker": "",
        "dynamic_v2_proof_refresh_status": dynamic_refresh_status,
        "dynamic_v2_comparator_generation": dynamic_generation,
        "dynamic_v2_proofs_refreshed": dynamic_refreshed,
        "ou_v3_proof_refresh_status": ou_refresh_status,
        "ou_v3_comparator_generation": ou_generation,
        "ou_v3_proofs_refreshed": ou_refreshed,
        "ou_v4_proof_refresh_status": ou_v4_refresh_status,
        "ou_v4_comparator_generation": ou_v4_generation,
        "ou_v4_proofs_refreshed": ou_v4_refreshed,
        "ou_v5_proof_refresh_status": ou_v5_refresh_status,
        "ou_v5_comparator_generation": ou_v5_generation,
        "ou_v5_proofs_refreshed": ou_v5_refreshed,
        "next_external_attempt_eligible_at": "2026-08-11T00:00:00+00:00",
        "blockers": list(blockers or []),
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    if local_evidence_fingerprint:
        payload["stage3_local_evidence_fingerprint"] = local_evidence_fingerprint
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    payload["receipt_id"] = "wizardproof_" + sha256(canonical.encode("utf-8")).hexdigest()[:20]
    path = (
        root
        / "reports"
        / "active"
        / "wizard_proof_scheduler_receipts"
        / now.strftime("%Y-%m-%d_%H%M%S_%f.json")
    )
    _write_json(path, payload)
    _write_json(
        root / "reports" / "active" / "corrective_wizard_proof_scheduler_status.json",
        {**payload, "receipt_path": str(path.relative_to(root))},
    )
    return path, payload


def _require_immutable_scheduler_receipt(
    root,
    path,
    payload,
    *,
    write_twin=True,
):
    payload["final_immutable_receipt_required"] = True
    payload.pop("receipt_id", None)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    payload["receipt_id"] = "wizardproof_" + sha256(canonical.encode("utf-8")).hexdigest()[:20]
    _write_json(path, payload)
    _write_json(
        root / "reports" / "active" / "corrective_wizard_proof_scheduler_status.json",
        {**payload, "receipt_path": str(path.relative_to(root))},
    )
    immutable = (
        root
        / "data"
        / "research"
        / "wizard_proof_scheduler_receipts"
        / f"{payload['receipt_id']}.json"
    )
    if write_twin:
        immutable.parent.mkdir(parents=True, exist_ok=True)
        immutable.write_bytes(path.read_bytes())
    return immutable


def _complete_stage4_generation(
    root,
    *,
    prior_contract_id="registeredrerun-prior",
    contract_id="registeredrerun-current",
):
    active = root / "reports" / "active"
    contract = {
        "schema_version": "thewiz.corrective_registered_rerun.v1",
        "contract_id": contract_id,
        "generation": 2,
        "prior_contract_id": prior_contract_id,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    _write_json(active / "registered_research_rerun_contract.json", contract)
    execution = {
        "schema_version": "thewiz.corrective_registered_rerun_execution.v1",
        "execution_id": "registeredexecution-current",
        "contract_id": contract_id,
        "status": "PASS_REGISTERED_RERUN_ACCOUNTED",
        "conclusion_status": "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS",
        "order_submission_performed": False,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    execution["receipt_sha256"] = sha256(
        json.dumps(execution, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    execution_path = (
        root / "data" / "research" / "registered_rerun_executions" / f"{contract_id}.json"
    )
    _write_json(execution_path, execution)
    relative = str(execution_path.relative_to(root))
    _write_json(
        active / "registered_research_rerun_execution.json",
        {
            "status": "PASS_REGISTERED_RERUN_ACCOUNTED",
            "execution_receipt_path": relative,
            "execution_receipt_sha256": sha256(execution_path.read_bytes()).hexdigest(),
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    (active / "seven_stage_goal_checkpoint.csv").write_text(
        "stage,status,evidence_progress,testnet_order_authority,live_trading_authorized\n"
        f"4,PASS,active_contract_id={contract_id};"
        "stage4_terminal_outcome=CONCLUSIVE_REJECTION_CURRENT_FAMILY;"
        "final_receipt_conclusion_bound=True,False,False\n",
        encoding="utf-8",
    )
    return relative


def _accepted_stage4_state(root, *, learning_status):
    execution = {
        "schema_version": "thewiz.corrective_registered_rerun_execution.v1",
        "execution_id": "execution-accepted",
        "contract_id": "contract-accepted",
        "conclusion_status": "ACCEPTED_REGISTERED_SURVIVORS",
        "stage4_final_one_x_survivors": 3,
        "stage4_independent_supporting_clusters": 3,
        "stage4_independent_full_survivor_clusters": 3,
        "stage4_independent_supporting_pairs": 3,
        "stage4_independent_full_survivor_pairs": 3,
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    execution_path = root / "data" / "research" / "registered_rerun_executions" / "accepted.json"
    _write_json(execution_path, execution)
    _write_json(
        root / "reports" / "active" / "registered_research_rerun_execution.json",
        {
            "status": "ALREADY_COMPLETE",
            "execution_receipt_path": str(execution_path.relative_to(root)),
            "execution_receipt_sha256": sha256(execution_path.read_bytes()).hexdigest(),
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    _write_learning_status(root, status=learning_status)
    checkpoint = root / "reports" / "active" / "seven_stage_goal_checkpoint.csv"
    checkpoint.write_text(
        "stage,status,next_action,evidence_progress\n"
        "4,PASS,,active_contract_id=contract-accepted;"
        "stage4_terminal_outcome=ACCEPTED_VALID_INDEPENDENT_SURVIVORS\n",
        encoding="utf-8",
    )


def _write_learning_status(root, *, status, immutable=True):
    payload = {
        "schema_version": "thewiz.corrective_registered_learning.v1",
        "registered_execution_id": "execution-accepted",
        "status": status,
        "stage5_research_gate_pass": status == "PASS_RESEARCH_LEARNING_GATES",
        "order_submission_performed": False,
        "promotion_authority": False,
        "testnet_candidate_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    active = {
        "status": status,
        "registered_execution_id": "execution-accepted",
        "stage5_research_gate_pass": payload["stage5_research_gate_pass"],
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    if immutable and status in {
        "PASS_RESEARCH_LEARNING_GATES",
        "REJECTED_RESEARCH_LEARNING_GATES",
    }:
        receipt_path = root / "data" / "research" / "registered_learning" / "accepted.json"
        _write_json(receipt_path, payload)
        active["learning_receipt_path"] = str(receipt_path.relative_to(root))
        active["learning_receipt_sha256"] = sha256(receipt_path.read_bytes()).hexdigest()
    _write_json(
        root / "reports" / "active" / "registered_learning_research_status.json",
        active,
    )


def _patch_verified_learning_audit(monkeypatch):
    def verified_audit(*, root):
        active = json.loads(
            (root / "reports" / "active" / "registered_learning_research_status.json").read_text()
        )
        gate_pass = bool(active["stage5_research_gate_pass"])
        return {
            "status": (
                "PASS_VERIFIED_REGISTERED_LEARNING_ACCEPTANCE"
                if gate_pass
                else "PASS_VERIFIED_REGISTERED_LEARNING_REJECTION"
            ),
            "evidence_valid": True,
            "registered_execution_id": active["registered_execution_id"],
            "stage5_research_gate_pass": gate_pass,
            "learning_receipt_path": active["learning_receipt_path"],
            "learning_receipt_sha256": active["learning_receipt_sha256"],
            "order_submission_performed": False,
            "promotion_authority": False,
            "testnet_candidate_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }

    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_registered_learning."
        "latest_verified_registered_learning",
        verified_audit,
    )


def _applied_dynamic_v2_state(root):
    snapshot = (
        root
        / "data"
        / "research"
        / "wizard_dynamic_comparator_activations"
        / "proof_snapshots"
        / "proof.csv"
    )
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text("exact_mode,formula_proof_complete\n", encoding="utf-8")
    packet_id = "dynamicv2review_test"
    packet = root / "data" / "research" / "wizard_dynamic_review_packets" / f"{packet_id}.json"
    _write_json(packet, {"review_packet_id": packet_id})
    packet_path = str(packet.relative_to(root))
    packet_hash = sha256(packet.read_bytes()).hexdigest()
    supreme_stable = {
        "schema_version": "thewiz.wizard_dynamic_v2_supreme_review.v1",
        "status": "PASS_ADVISORY_ONLY",
        "recommendation": "RECOMMEND_EXPLICIT_RESEARCH_COMPARATOR_ACTIVATION",
        "blockers": [],
        "review_packet_id": packet_id,
        "review_packet_path": "reports/active/wizard_dynamic_v2_review_packet.json",
        "immutable_review_packet_path": packet_path,
        "immutable_review_packet_sha256": packet_hash,
        "activation_preflight_status": "READY_REQUIRES_EXPLICIT_APPLY",
        "required_cells": 4,
        "passed_cells": 4,
        "cohorts_disjoint": True,
        "raw_bindings_valid": True,
        "max_abs_reconstruction_error": 0.0,
        "findings": [],
        "human_approval_required": True,
        "human_approval_recorded": False,
        "activation_applied_by_supreme_team": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    supreme_id = (
        "dynamicv2supreme_"
        + sha256(
            json.dumps(
                supreme_stable,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()[:20]
    )
    supreme = root / "data" / "research" / "wizard_dynamic_supreme_reviews" / f"{supreme_id}.json"
    _write_json(supreme, {**supreme_stable, "review_id": supreme_id})
    receipt = {
        "schema_version": "thewiz.wizard_dynamic_v2_activation.v2",
        "evaluated_at_utc": NOW.isoformat(),
        "status": "APPLIED_RESEARCH_COMPARATOR_ONLY",
        "apply_requested": True,
        "reviewer": "codex-research-reviewer",
        "review_note": "Reviewed all four disjoint holdout cells and immutable bindings.",
        "review_packet_id_submitted": packet_id,
        "review_packet_id": packet_id,
        "review_packet_path": packet_path,
        "review_packet_sha256": packet_hash,
        "supreme_review_id": supreme_id,
        "supreme_review_path": str(supreme.relative_to(root)),
        "supreme_review_sha256": sha256(supreme.read_bytes()).hexdigest(),
        "comparator_generation": 2,
        "source_proof_snapshot_path": str(snapshot.relative_to(root)),
        "source_proof_snapshot_sha256": sha256(snapshot.read_bytes()).hexdigest(),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    stable = {
        key: value
        for key, value in receipt.items()
        if key not in {"evaluated_at_utc", "activation_id"}
    }
    activation_id = (
        "dynamicv2activation_"
        + sha256(
            json.dumps(
                stable,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()[:20]
    )
    receipt["activation_id"] = activation_id
    immutable = (
        root
        / "data"
        / "research"
        / "wizard_dynamic_comparator_activations"
        / f"{activation_id}.json"
    )
    _write_json(immutable, receipt)
    _write_json(
        root / "reports" / "active" / "wizard_dynamic_v2_activation_status.json",
        {
            **receipt,
            "immutable_activation_path": str(immutable.relative_to(root)),
            "immutable_activation_sha256": sha256(immutable.read_bytes()).hexdigest(),
        },
    )
    _write_dynamic_refresh_status(
        root,
        activation_id=activation_id,
        status="NOT_ACTIVE",
        generation=1,
        captured=0,
        exact=0,
        refreshed=0,
    )
    return activation_id


def _write_dynamic_refresh_status(
    root,
    *,
    activation_id,
    status,
    generation,
    captured,
    exact,
    refreshed,
):
    _write_json(
        root / "reports" / "active" / "wizard_dynamic_v2_proof_refresh_status.json",
        {
            "schema_version": "thewiz.wizard_dynamic_v2_proof_refresh.v1",
            "status": status,
            "activation_id": activation_id,
            "comparator_generation": generation,
            "captured_dynamic_rows": captured,
            "exact_dynamic_rows": exact,
            "refreshed_dynamic_rows": refreshed,
            "research_only": True,
            "original_comparator_mutated": False,
            "raw_vendor_evidence_mutated": False,
            "candidate_promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )


def _applied_ou_v3_state(root):
    snapshot = (
        root
        / "data"
        / "research"
        / "wizard_ou_v3_comparator_activations"
        / "proof_snapshots"
        / "proof.csv"
    )
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text("exact_mode,formula_proof_complete\n", encoding="utf-8")
    receipt = {
        "schema_version": "thewiz.wizard_ou_v3_activation.v1",
        "evaluated_at_utc": NOW.isoformat(),
        "status": "APPLIED_RESEARCH_COMPARATOR_ONLY",
        "apply_requested": True,
        "reviewer": "codex-research-reviewer",
        "review_note": "Reviewed OU formula and selector holdouts independently.",
        "review_packet_id": "ouv3review_test",
        "review_packet_id_submitted": "ouv3review_test",
        "comparator_generation": 3,
        "source_proof_snapshot_path": str(snapshot.relative_to(root)),
        "source_proof_snapshot_sha256": sha256(snapshot.read_bytes()).hexdigest(),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    stable = {
        key: value
        for key, value in receipt.items()
        if key not in {"evaluated_at_utc", "activation_id"}
    }
    activation_id = (
        "ouv3activation_"
        + sha256(
            json.dumps(
                stable,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()[:20]
    )
    receipt["activation_id"] = activation_id
    immutable = (
        root / "data" / "research" / "wizard_ou_v3_comparator_activations" / f"{activation_id}.json"
    )
    _write_json(immutable, receipt)
    _write_json(
        root / "reports" / "active" / "wizard_ou_v3_activation_status.json",
        {
            **receipt,
            "immutable_activation_path": str(immutable.relative_to(root)),
            "immutable_activation_sha256": sha256(immutable.read_bytes()).hexdigest(),
        },
    )
    _write_ou_refresh_status(
        root,
        activation_id=activation_id,
        status="NOT_ACTIVE",
        generation=1,
        captured=0,
        exact=0,
        refreshed=0,
    )
    return activation_id


def _write_ou_refresh_status(
    root,
    *,
    activation_id,
    status,
    generation,
    captured,
    exact,
    refreshed,
):
    _write_json(
        root / "reports" / "active" / "wizard_ou_v3_proof_refresh_status.json",
        {
            "schema_version": "thewiz.wizard_ou_v3_proof_refresh.v1",
            "status": status,
            "activation_id": activation_id,
            "comparator_generation": generation,
            "captured_ou_rows": captured,
            "exact_ou_rows": exact,
            "refreshed_ou_rows": refreshed,
            "research_only": True,
            "raw_vendor_evidence_mutated": False,
            "candidate_promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )


def test_same_day_attempt_skips_heavy_scheduler_and_preserves_zero_authority(tmp_path):
    _status(tmp_path, attempted=True)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        runner=lambda *_, **__: (_ for _ in ()).throw(AssertionError("not called")),
    )

    assert result["launcher_status"] == "DEFERRED_SAME_UTC_DAY_LIGHTWEIGHT"
    assert result["heavy_scheduler_invoked"] is False
    assert result["external_attempt_made_by_launcher"] is False
    assert result["testnet_order_authority"] is False
    assert result["live_trading_authorized"] is False
    persisted = json.loads(
        (
            tmp_path / "reports" / "active" / "corrective_wizard_proof_launcher_status.json"
        ).read_text()
    )
    assert persisted["launcher_receipt_id"].startswith("wizardlauncher_")
    binding = launcher.validate_launcher_receipt_binding(
        root=tmp_path,
        status=persisted,
    )
    assert binding["status"] == "PASS"
    assert (tmp_path / binding["receipt_path"]).is_file()


def test_launcher_has_no_child_process_default_and_main_binds_in_process() -> None:
    runner = inspect.signature(run_wizard_proof_launcher).parameters["runner"]

    assert runner.default is inspect.Signature.empty
    main_source = inspect.getsource(launcher.main)
    assert "runner=_run_heavy_scheduler_in_process" in main_source
    assert "subprocess.run" not in main_source


def test_launcher_receipt_tampering_invalidates_active_binding(tmp_path):
    _status(tmp_path, attempted=True)
    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        runner=lambda *_, **__: (_ for _ in ()).throw(AssertionError("not called")),
    )
    receipt_path = tmp_path / result["immutable_launcher_receipt_path"]
    receipt_path.chmod(0o600)
    receipt_path.write_text("{}\n", encoding="utf-8")

    binding = launcher.validate_launcher_receipt_binding(
        root=tmp_path,
        status=result,
    )

    assert binding["status"] == "BLOCKED"
    assert "launcher_immutable_receipt_hash_invalid" in binding["blockers"]
    assert "launcher_immutable_receipt_content_invalid" in binding["blockers"]


def test_launcher_immutable_receipt_rejects_collision(tmp_path):
    path = tmp_path / "data" / "research" / "launcher" / "receipt.json"
    payload = {"launcher_receipt_id": "wizardlauncher_test", "status": "PASS"}
    launcher._write_or_validate_immutable_json(payload, path)
    launcher._write_or_validate_immutable_json(payload, path)
    path.chmod(0o600)
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="immutable launcher receipt collision"):
        launcher._write_or_validate_immutable_json(payload, path)


def test_earlier_completed_launcher_cannot_overwrite_active_status(tmp_path):
    status_path = tmp_path / "reports" / "active" / "corrective_wizard_proof_launcher_status.json"
    newer = {
        "checked_at_utc": "2026-08-10T12:01:00+00:00",
        "completed_at_utc": "2026-08-10T12:03:00+00:00",
        "launcher_status": "DEFERRED_SAME_UTC_DAY_LIGHTWEIGHT",
    }
    older = {
        "checked_at_utc": "2026-08-10T12:00:00+00:00",
        "completed_at_utc": "2026-08-10T12:02:00+00:00",
        "launcher_status": "INTERNAL_STAGE3_RECONCILIATION_INCOMPLETE",
    }

    assert launcher._publish_latest_launcher_status(newer, status_path) is True
    assert launcher._publish_latest_launcher_status(older, status_path) is False

    assert json.loads(status_path.read_text(encoding="utf-8")) == newer


def test_older_started_launcher_can_publish_later_completed_evidence(tmp_path):
    status_path = tmp_path / "reports" / "active" / "corrective_wizard_proof_launcher_status.json"
    lightweight = {
        "checked_at_utc": "2026-08-10T12:01:00+00:00",
        "completed_at_utc": "2026-08-10T12:02:00+00:00",
        "launcher_status": "DEFERRED_SAME_UTC_DAY_LIGHTWEIGHT",
    }
    late_heavy_evidence = {
        "checked_at_utc": "2026-08-10T12:00:00+00:00",
        "completed_at_utc": "2026-08-10T12:03:00+00:00",
        "launcher_status": "HEAVY_SCHEDULER_EVIDENCE_COMPLETE",
    }

    assert launcher._publish_latest_launcher_status(lightweight, status_path) is True
    assert launcher._publish_latest_launcher_status(late_heavy_evidence, status_path) is True

    assert json.loads(status_path.read_text(encoding="utf-8")) == late_heavy_evidence


def test_launcher_dry_run_cannot_replace_execute_mode_operational_heartbeat(tmp_path):
    status_path = tmp_path / "reports" / "active" / "corrective_wizard_proof_launcher_status.json"
    operational = {
        "checked_at_utc": "2026-08-13T03:00:00+00:00",
        "completed_at_utc": "2026-08-13T03:00:01+00:00",
        "execute_requested": True,
        "launcher_status": "DEFERRED_SAME_UTC_DAY_LIGHTWEIGHT",
    }
    inspection = {
        "checked_at_utc": "2026-08-13T03:01:00+00:00",
        "completed_at_utc": "2026-08-13T03:01:01+00:00",
        "execute_requested": False,
        "launcher_status": "HEAVY_SCHEDULER_PLANNED",
    }

    assert launcher._publish_latest_launcher_status(operational, status_path) is True
    assert launcher._publish_latest_launcher_status(inspection, status_path) is False

    persisted = json.loads(status_path.read_text(encoding="utf-8"))
    assert persisted == operational


def test_launcher_execute_mode_heartbeat_replaces_dry_run_even_if_completed_earlier(
    tmp_path,
):
    status_path = tmp_path / "reports" / "active" / "corrective_wizard_proof_launcher_status.json"
    inspection = {
        "checked_at_utc": "2026-08-13T03:01:00+00:00",
        "completed_at_utc": "2026-08-13T03:01:01+00:00",
        "execute_requested": False,
        "launcher_status": "HEAVY_SCHEDULER_PLANNED",
    }
    operational = {
        "checked_at_utc": "2026-08-13T03:00:00+00:00",
        "completed_at_utc": "2026-08-13T03:00:01+00:00",
        "execute_requested": True,
        "launcher_status": "DEFERRED_SAME_UTC_DAY_LIGHTWEIGHT",
    }

    assert launcher._publish_latest_launcher_status(inspection, status_path) is True
    assert launcher._publish_latest_launcher_status(operational, status_path) is True

    persisted = json.loads(status_path.read_text(encoding="utf-8"))
    assert persisted == operational


def test_deferred_check_reports_latest_manifest_binding_separately(tmp_path):
    _scheduler_receipt(
        tmp_path,
        stage3_complete=False,
        registered_rerun_status="NOT_EVALUATED",
    )

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        runner=lambda *_, **__: (_ for _ in ()).throw(AssertionError("not called")),
    )

    assert result["launcher_status"] == "DEFERRED_SAME_UTC_DAY_LIGHTWEIGHT"
    assert result["heavy_scheduler_invoked"] is False
    assert result["scheduler_capture_manifest_binding_complete"] is False
    assert result["latest_status_capture_manifest_binding_complete"] is True
    assert result["latest_status_stage3_evidence_complete"] is False
    assert result["latest_immutable_execution_capture_manifest_binding_complete"] is True
    assert result["latest_immutable_execution_stage3_evidence_complete"] is False


def test_same_day_changed_stage3_evidence_runs_local_reconciliation_only(tmp_path):
    _scheduler_receipt(
        tmp_path,
        status="COMPLETE_RESPONSE_CAPTURE_FORMULA_PROOF_INCOMPLETE",
        stage3_complete=False,
        responses_captured=28,
        accepted_mode_evidence=8,
        local_evidence_fingerprint="0" * 64,
        registered_rerun_status="NOT_EVALUATED",
    )
    python = _python(tmp_path)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        _scheduler_receipt(
            tmp_path,
            now=datetime(2026, 8, 10, 12, 0, 1, tzinfo=UTC),
            status="DEFERRED_SAME_UTC_DAY",
            internal_continuation_only=True,
            external_attempt=False,
            stage3_complete=False,
            responses_captured=28,
            accepted_mode_evidence=8,
            local_evidence_fingerprint=_stage3_local_evidence_fingerprint(tmp_path),
            registered_rerun_status="NOT_EVALUATED",
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=runner,
    )

    assert calls == [
        [
            str(python),
            "-m",
            HEAVY_MODULE,
            "--execute",
            "--internal-continuation-only",
        ]
    ]
    assert result["internal_continuation_reason"] == ("stage3_local_reconciliation")
    assert result["launcher_status"] == ("INTERNAL_STAGE3_RECONCILIATION_INCOMPLETE")
    assert result["external_calls_forbidden_for_invocation"] is True
    assert result["external_attempt_made_by_launcher"] is False


def test_pending_frozen_capture_suppresses_futile_local_reconciliation(tmp_path):
    _scheduler_receipt(
        tmp_path,
        status="DEFERRED_CAPTURE_MANIFEST_WINDOW",
        stage3_complete=False,
        responses_captured=28,
        accepted_mode_evidence=12,
        capture_reconciliation_status="PENDING",
        capture_reconciliation_valid=False,
        capture_reconciliation_complete=False,
        capture_reconciliation_required_calls=8,
        capture_reconciliation_completed_calls=0,
        capture_reconciliation_pending_calls=8,
        capture_reconciliation_blocked_calls=0,
        local_evidence_fingerprint="0" * 64,
        registered_rerun_status="NOT_EVALUATED",
    )

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        runner=lambda *_, **__: (_ for _ in ()).throw(AssertionError("not called")),
    )

    assert result["internal_continuation_reason"] == ""
    assert result["launcher_status"] == "DEFERRED_SAME_UTC_DAY_LIGHTWEIGHT"
    assert result["heavy_scheduler_invoked"] is False
    assert result["external_attempt_made_by_launcher"] is False


def test_dry_run_does_not_request_execute_only_internal_continuation(tmp_path):
    _scheduler_receipt(
        tmp_path,
        status="COMPLETE_RESPONSE_CAPTURE_FORMULA_PROOF_INCOMPLETE",
        stage3_complete=False,
        responses_captured=28,
        accepted_mode_evidence=8,
        local_evidence_fingerprint="0" * 64,
        registered_rerun_status="NOT_EVALUATED",
    )
    python = _python(tmp_path)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        _scheduler_receipt(
            tmp_path,
            now=datetime(2026, 8, 10, 12, 0, 1, tzinfo=UTC),
            status="PLANNED",
            execute=False,
            internal_continuation_only=False,
            external_attempt=False,
            stage3_complete=False,
            responses_captured=28,
            accepted_mode_evidence=8,
            registered_rerun_status="NOT_EVALUATED",
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=False,
        python=python,
        runner=runner,
    )

    assert calls == [[str(python), "-m", HEAVY_MODULE]]
    assert result["launcher_status"] == "HEAVY_SCHEDULER_PLANNED"
    assert result["internal_registered_continuation_needed"] is True
    assert result["internal_continuation_only_invoked"] is False
    assert result["external_calls_forbidden_for_invocation"] is False
    assert result["external_attempt_made_by_launcher"] is False


def test_same_stage3_fingerprint_does_not_repeat_local_reconciliation(tmp_path):
    fingerprint = _stage3_local_evidence_fingerprint(tmp_path)
    _scheduler_receipt(
        tmp_path,
        status="COMPLETE_RESPONSE_CAPTURE_FORMULA_PROOF_INCOMPLETE",
        stage3_complete=False,
        responses_captured=28,
        accepted_mode_evidence=8,
        local_evidence_fingerprint=fingerprint,
        registered_rerun_status="NOT_EVALUATED",
    )

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        runner=lambda *_, **__: (_ for _ in ()).throw(AssertionError("not called")),
    )

    assert result["launcher_status"] == "DEFERRED_SAME_UTC_DAY_LIGHTWEIGHT"
    assert result["heavy_scheduler_invoked"] is False


def test_ou_selector_registration_changes_stage3_local_fingerprint(tmp_path):
    before = _stage3_local_evidence_fingerprint(tmp_path)
    _write_json(
        tmp_path / "reports" / "active" / "wizard_ou_trend_selector_v1_receipt.json",
        {
            "status": "REGISTERED_WAITING_VENDOR_RESPONSES",
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    after = _stage3_local_evidence_fingerprint(tmp_path)

    assert before != after


def test_dated_receipt_prevents_duplicate_when_latest_status_loses_attempt(tmp_path):
    _status(tmp_path, attempted=False)
    receipt = (
        tmp_path
        / "reports"
        / "active"
        / "wizard_proof_scheduler_receipts"
        / "2026-08-10_000001_000000.json"
    )
    _write_json(
        receipt,
        {
            "attempt_date_utc": "2026-08-10",
            "external_attempt_made": True,
            "research_only": True,
        },
    )

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        runner=lambda *_, **__: (_ for _ in ()).throw(AssertionError("not called")),
    )

    assert result["same_day_attempt_state"] == "ATTEMPTED"
    assert result["heavy_scheduler_invoked"] is False
    assert str(receipt) in result["same_day_evidence_paths"]
    assert result["same_day_evidence_count"] == 2


def test_malformed_same_day_receipt_blocks_instead_of_risking_duplicate(tmp_path):
    receipt = (
        tmp_path
        / "reports"
        / "active"
        / "wizard_proof_scheduler_receipts"
        / "2026-08-10_000001_000000.json"
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text("{broken", encoding="utf-8")

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        runner=lambda *_, **__: (_ for _ in ()).throw(AssertionError("not called")),
    )

    assert result["launcher_status"] == "BLOCKED_UNVERIFIABLE_SAME_DAY_EVIDENCE"
    assert result["same_day_attempt_state"] == "UNVERIFIABLE"
    assert result["heavy_scheduler_invoked"] is False


def test_clear_day_invokes_exact_heavy_scheduler_without_shell(tmp_path):
    calls = []
    python = _python(tmp_path)

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        _scheduler_receipt(tmp_path)
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=runner,
    )

    assert result["launcher_status"] == "HEAVY_SCHEDULER_EVIDENCE_COMPLETE"
    assert result["heavy_scheduler_invoked"] is True
    assert calls[0][0] == [str(python), "-m", HEAVY_MODULE, "--execute"]
    assert calls[0][1]["cwd"] == tmp_path
    assert calls[0][1]["check"] is False
    assert "shell" not in calls[0][1]
    assert result["heavy_scheduler_receipt_valid"] is True
    assert result["scheduler_status"] == "COMPLETE_ACCEPTED_MODE_EVIDENCE"
    assert result["scheduler_stage3_evidence_complete"] is True
    assert result["scheduler_capture_manifest_binding_complete"] is True
    assert result["external_attempt_made_by_launcher"] is True
    assert _launcher_exit_code(result) == 0


def test_new_utc_day_ignores_prior_day_attempt_and_runs_frozen_capture(tmp_path):
    python = _python(tmp_path)
    prior_day = NOW - timedelta(days=1)
    _scheduler_receipt(
        tmp_path,
        now=prior_day,
        stage3_complete=False,
        registered_rerun_status="NOT_EVALUATED",
    )
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        _scheduler_receipt(tmp_path, now=NOW)
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=runner,
    )

    assert result["same_day_attempt_state"] == "CLEAR"
    assert result["heavy_scheduler_invoked"] is True
    assert result["scheduler_stage3_evidence_complete"] is True
    assert result["scheduler_capture_manifest_binding_complete"] is True
    assert calls[0][0] == [str(python), "-m", HEAVY_MODULE, "--execute"]


def test_manifest_binding_mismatch_is_reported_explicitly(tmp_path):
    python = _python(tmp_path)

    def runner(command, **_kwargs):
        path, payload = _scheduler_receipt(tmp_path)
        payload["capture_reconciliation_manifest_sha256"] = "b" * 64
        payload.pop("receipt_id")
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        payload["receipt_id"] = "wizardproof_" + sha256(canonical.encode("utf-8")).hexdigest()[:20]
        _write_json(path, payload)
        _write_json(
            tmp_path / "reports" / "active" / "corrective_wizard_proof_scheduler_status.json",
            {**payload, "receipt_path": str(path.relative_to(tmp_path))},
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=runner,
    )

    assert result["launcher_status"] == ("HEAVY_SCHEDULER_RECONCILED_INCOMPLETE")
    assert result["scheduler_capture_manifest_binding_complete"] is False
    assert result["scheduler_stage3_evidence_complete"] is False
    assert result["blocker"] == (
        "scheduler_research_incomplete:capture_manifest_binding_incomplete"
    )


def test_launcher_refuses_forged_pass_when_immutable_reconciliation_disappears(
    tmp_path,
):
    python = _python(tmp_path)

    def runner(command, **_kwargs):
        _, payload = _scheduler_receipt(tmp_path)
        reconciliation = tmp_path / payload["capture_reconciliation_immutable_path"]
        reconciliation.unlink()
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=runner,
    )

    assert result["heavy_scheduler_receipt_valid"] is True
    assert result["scheduler_capture_manifest_binding_complete"] is True
    assert result["scheduler_stage3_evidence_complete"] is False
    assert result["launcher_status"] == "HEAVY_SCHEDULER_RECONCILED_INCOMPLETE"
    assert result["external_attempt_made_by_launcher"] is True
    assert result["testnet_order_authority"] is False
    assert result["live_trading_authorized"] is False
    assert _launcher_exit_code(result) == 2


def test_launcher_refuses_stage3_completion_when_capture_reconciliation_is_pending(
    tmp_path,
):
    python = _python(tmp_path)

    def runner(command, **kwargs):
        _scheduler_receipt(
            tmp_path,
            status="COMPLETE_ACCEPTED_MODE_EVIDENCE",
            capture_reconciliation_status="PENDING",
            capture_reconciliation_valid=False,
            capture_reconciliation_completed_calls=0,
            capture_reconciliation_pending_calls=13,
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=runner,
    )

    assert result["launcher_status"] == "HEAVY_SCHEDULER_RECONCILED_INCOMPLETE"
    assert result["scheduler_stage3_evidence_complete"] is False
    assert result["testnet_order_authority"] is False
    assert result["live_trading_authorized"] is False
    assert _launcher_exit_code(result) == 2


def test_launcher_refuses_not_required_as_stage3_capture_evidence(tmp_path):
    python = _python(tmp_path)

    def runner(command, **kwargs):
        _scheduler_receipt(
            tmp_path,
            status="COMPLETE_ACCEPTED_MODE_EVIDENCE",
            capture_reconciliation_status="NOT_REQUIRED",
            capture_reconciliation_valid=True,
            capture_reconciliation_complete=True,
            capture_reconciliation_required_calls=0,
            capture_reconciliation_completed_calls=0,
            capture_reconciliation_pending_calls=0,
            capture_reconciliation_blocked_calls=0,
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=runner,
    )

    assert result["launcher_status"] == "HEAVY_SCHEDULER_RECONCILED_INCOMPLETE"
    assert result["scheduler_stage3_evidence_complete"] is False
    assert result["testnet_order_authority"] is False
    assert result["live_trading_authorized"] is False
    assert _launcher_exit_code(result) == 2


def test_reconciled_negative_research_outcome_exits_operationally_clean(tmp_path):
    python = _python(tmp_path)

    def runner(command, **kwargs):
        _scheduler_receipt(
            tmp_path,
            status="MANIFESTED_NON_EXACT_LANES_READY",
            stage3_complete=False,
            registered_rerun_status="NOT_EVALUATED",
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=runner,
    )

    assert result["launcher_status"] == ("HEAVY_SCHEDULER_RECONCILED_RESEARCH_INCOMPLETE")
    assert result["scheduler_stage3_evidence_complete"] is False
    assert result["scheduler_capture_manifest_binding_complete"] is True
    assert result["blocker"] == ("scheduler_research_incomplete:MANIFESTED_NON_EXACT_LANES_READY")
    assert result["testnet_order_authority"] is False
    assert result["live_trading_authorized"] is False
    assert _launcher_exit_code(result) == 0


def test_zero_exit_without_fresh_scheduler_receipt_fails_closed(tmp_path):
    python = _python(tmp_path)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    assert result["launcher_status"] == "BLOCKED_SCHEDULER_RECEIPT_UNVERIFIED"
    assert result["heavy_scheduler_receipt_valid"] is False
    assert result["blocker"] == "heavy_scheduler_returned_without_fresh_receipt"
    assert result["external_attempt_made_by_launcher"] is False
    assert result["testnet_order_authority"] is False
    assert _launcher_exit_code(result) == 2


def test_marked_scheduler_receipt_requires_immutable_twin(tmp_path):
    python = _python(tmp_path)

    def runner(command, **kwargs):
        path, payload = _scheduler_receipt(tmp_path)
        _require_immutable_scheduler_receipt(
            tmp_path,
            path,
            payload,
            write_twin=False,
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=runner,
    )

    assert result["launcher_status"] == "BLOCKED_SCHEDULER_RECEIPT_UNVERIFIED"
    assert result["heavy_scheduler_receipt_valid"] is False
    assert "scheduler_immutable_receipt_missing" in result["blocker"]
    assert result["testnet_order_authority"] is False
    assert result["live_trading_authorized"] is False
    audit = latest_verified_immutable_scheduler_execution(root=tmp_path)
    assert audit["status"] == "BLOCKED_FINAL_IMMUTABLE_EXECUTION"
    assert audit["valid"] is False
    assert "scheduler_immutable_receipt_missing" in audit["blockers"][0]


def test_marked_scheduler_receipt_requires_byte_identical_twin(tmp_path):
    python = _python(tmp_path)

    def runner(command, **kwargs):
        path, payload = _scheduler_receipt(tmp_path)
        _require_immutable_scheduler_receipt(tmp_path, path, payload)
        path.write_bytes(path.read_bytes() + b"\n")
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=runner,
    )

    assert result["launcher_status"] == "BLOCKED_SCHEDULER_RECEIPT_UNVERIFIED"
    assert result["heavy_scheduler_receipt_valid"] is False
    assert "scheduler_immutable_receipt_mismatch" in result["blocker"]
    audit = latest_verified_immutable_scheduler_execution(root=tmp_path)
    assert audit["valid"] is False
    assert "scheduler_immutable_receipt_mismatch" in audit["blockers"][0]


def test_marked_scheduler_receipt_with_matching_twin_is_accepted(tmp_path):
    python = _python(tmp_path)

    def runner(command, **kwargs):
        path, payload = _scheduler_receipt(tmp_path)
        _require_immutable_scheduler_receipt(tmp_path, path, payload)
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=runner,
    )

    assert result["heavy_scheduler_receipt_valid"] is True
    assert result["scheduler_stage3_evidence_complete"] is True
    assert "scheduler_immutable_receipt" not in result["blocker"]
    audit = latest_verified_immutable_scheduler_execution(root=tmp_path)
    assert audit["status"] == "PASS_FINAL_IMMUTABLE_EXECUTION"
    assert audit["valid"] is True
    assert audit["receipt_id"] == result["scheduler_receipt_id"]
    assert audit["stage3_evidence_complete"] is True
    assert audit["testnet_order_authority"] is False
    assert audit["live_trading_authorized"] is False


def test_legacy_execution_guards_same_day_but_cannot_prove_final_execution(tmp_path):
    python = _python(tmp_path)
    _scheduler_receipt(tmp_path)

    audit = latest_verified_immutable_scheduler_execution(root=tmp_path)
    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 99),
    )

    assert audit["status"] == "BLOCKED_FINAL_IMMUTABLE_EXECUTION"
    assert audit["valid"] is False
    assert "scheduler_receipt_not_final_immutable" in audit["blockers"][0]
    assert result["same_day_attempt_state"] == "ATTEMPTED"
    assert result["heavy_scheduler_invoked"] is False


def test_newer_invalid_execution_cannot_fall_back_to_older_valid_twin(tmp_path):
    old_path, old_payload = _scheduler_receipt(
        tmp_path,
        now=NOW - timedelta(hours=1),
    )
    _require_immutable_scheduler_receipt(tmp_path, old_path, old_payload)
    new_path, new_payload = _scheduler_receipt(tmp_path, now=NOW)
    _require_immutable_scheduler_receipt(
        tmp_path,
        new_path,
        new_payload,
        write_twin=False,
    )

    audit = latest_verified_immutable_scheduler_execution(root=tmp_path)

    assert audit["valid"] is False
    assert audit["receipt_id"] == new_payload["receipt_id"]
    assert audit["receipt_id"] != old_payload["receipt_id"]
    assert "scheduler_immutable_receipt_missing" in audit["blockers"][0]


def test_blocked_scheduler_receipt_cannot_masquerade_as_completed(tmp_path):
    python = _python(tmp_path)

    def runner(command, **kwargs):
        _scheduler_receipt(
            tmp_path,
            status="BLOCKED_NO_PROGRESS",
            stage3_complete=False,
            blockers=["proof_batch_made_no_response_capture_progress"],
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=runner,
    )

    assert result["launcher_status"] == "HEAVY_SCHEDULER_BLOCKED"
    assert result["heavy_scheduler_receipt_valid"] is True
    assert result["scheduler_status"] == "BLOCKED_NO_PROGRESS"
    assert result["scheduler_stage3_evidence_complete"] is False
    assert result["blocker"] == ("scheduler_outcome:proof_batch_made_no_response_capture_progress")
    assert result["external_attempt_made_by_launcher"] is True
    assert _launcher_exit_code(result) == 2


def test_forged_scheduler_receipt_id_fails_closed(tmp_path):
    python = _python(tmp_path)

    def runner(command, **kwargs):
        path, payload = _scheduler_receipt(tmp_path)
        payload["receipt_id"] = "wizardproof_forged"
        _write_json(path, payload)
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=runner,
    )

    assert result["launcher_status"] == "BLOCKED_SCHEDULER_RECEIPT_UNVERIFIED"
    assert result["heavy_scheduler_receipt_valid"] is False
    assert "scheduler_receipt_id_mismatch" in result["blocker"]
    assert result["external_attempt_made_by_launcher"] is False
    assert _launcher_exit_code(result) == 2


def test_force_is_explicitly_forwarded_to_heavy_scheduler(tmp_path):
    _status(tmp_path, attempted=True)
    python = _python(tmp_path)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        _scheduler_receipt(tmp_path, force=True)
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        force=True,
        python=python,
        runner=runner,
    )

    assert calls == [[str(python), "-m", HEAVY_MODULE, "--execute", "--force"]]
    assert result["force_requested"] is True


def test_same_day_internal_generation_continuation_runs_without_external_permission(
    tmp_path,
):
    status = _status(tmp_path, attempted=True)
    _write_json(
        status,
        {
            "attempt_date_utc": "2026-08-10",
            "external_attempt_made": True,
            "next_external_attempt_eligible_at": "2026-08-11T00:00:00+00:00",
            "queue_eligible": 28,
            "responses_captured_after": 28,
            "accepted_mode_evidence_cells": 28,
            "parity_refresh_status": "PASS",
            "credit_reconciliation_status": "PASS_RECONCILED",
            **_capture_reconciliation_complete_fields(tmp_path),
            "registered_rerun_status": "PASS_REGISTERED_RERUN_ACCOUNTED",
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    checkpoint = tmp_path / "reports" / "active" / "seven_stage_goal_checkpoint.csv"
    checkpoint.write_text(
        "stage,status,next_action\n"
        "4,IN_PROGRESS,advance_registered_rerun_to_unaccounted_current_family_cohort\n",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "reports" / "active" / "registered_research_rerun_contract.json",
        {"contract_id": "registeredrerun-prior"},
    )
    python = _python(tmp_path)
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        execution_path = _complete_stage4_generation(tmp_path)
        _scheduler_receipt(
            tmp_path,
            internal_continuation_only=True,
            external_attempt=False,
            registered_rerun_receipt_path=execution_path,
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=runner,
    )

    assert calls[0][0] == [
        str(python),
        "-m",
        HEAVY_MODULE,
        "--execute",
        "--internal-continuation-only",
    ]
    assert result["launcher_status"] == ("INTERNAL_REGISTERED_CONTINUATION_COMPLETED")
    assert result["internal_registered_continuation_needed"] is True
    assert result["internal_continuation_only_invoked"] is True
    assert result["external_calls_forbidden_for_invocation"] is True
    assert result["external_attempt_made_by_launcher"] is False


def test_same_day_active_generation_executes_without_external_permission(tmp_path):
    status = _status(tmp_path, attempted=True)
    _write_json(
        status,
        {
            "attempt_date_utc": "2026-08-10",
            "external_attempt_made": True,
            "next_external_attempt_eligible_at": "2026-08-11T00:00:00+00:00",
            "queue_eligible": 28,
            "responses_captured_after": 28,
            "accepted_mode_evidence_cells": 28,
            "parity_refresh_status": "PASS",
            "credit_reconciliation_status": "PASS_RECONCILED",
            **_capture_reconciliation_complete_fields(tmp_path),
            "registered_rerun_status": "PASS_REGISTERED_RERUN_ACCOUNTED",
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    checkpoint = tmp_path / "reports" / "active" / "seven_stage_goal_checkpoint.csv"
    checkpoint.write_text(
        "stage,status,next_action\n"
        "4,IN_PROGRESS,execute_active_registered_rerun_after_ready_chain\n",
        encoding="utf-8",
    )
    active_contract = {
        "schema_version": "thewiz.corrective_registered_rerun.v1",
        "contract_id": "registeredrerun-current",
        "generation": 2,
        "prior_contract_id": "registeredrerun-prior",
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    _write_json(
        tmp_path / "reports" / "active" / "registered_research_rerun_contract.json",
        active_contract,
    )
    python = _python(tmp_path)
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        execution_path = _complete_stage4_generation(
            tmp_path,
            prior_contract_id="registeredrerun-prior",
            contract_id="registeredrerun-current",
        )
        _scheduler_receipt(
            tmp_path,
            internal_continuation_only=True,
            external_attempt=False,
            registered_rerun_receipt_path=execution_path,
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=runner,
    )

    assert calls[0][0] == [
        str(python),
        "-m",
        HEAVY_MODULE,
        "--execute",
        "--internal-continuation-only",
    ]
    assert result["internal_continuation_reason"] == "stage4_active_execution"
    assert result["internal_continuation_prior_stage4_contract_id"] == ("registeredrerun-current")
    assert result["launcher_status"] == ("INTERNAL_REGISTERED_CONTINUATION_COMPLETED")
    assert result["external_calls_forbidden_for_invocation"] is True
    assert result["external_attempt_made_by_launcher"] is False


def test_active_generation_execution_can_roll_into_another_local_generation(tmp_path):
    execution_path = _complete_stage4_generation(
        tmp_path,
        prior_contract_id="registeredrerun-parent",
        contract_id="registeredrerun-executed",
    )
    _write_json(
        tmp_path / "reports" / "active" / "registered_research_rerun_contract.json",
        {
            "schema_version": "thewiz.corrective_registered_rerun.v1",
            "contract_id": "registeredrerun-next",
            "generation": 3,
            "prior_contract_id": "registeredrerun-executed",
            "promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    (tmp_path / "reports" / "active" / "seven_stage_goal_checkpoint.csv").write_text(
        "stage,status,next_action,testnet_order_authority,live_trading_authorized\n"
        "4,IN_PROGRESS,execute_active_registered_rerun_after_ready_chain,False,False\n",
        encoding="utf-8",
    )

    assert _stage4_active_execution_continuation_complete(
        root=tmp_path,
        scheduler_receipt={"registered_rerun_receipt_path": execution_path},
        prior_contract_id="registeredrerun-executed",
    )


def test_planning_latest_cannot_hide_immutable_execution_continuation(tmp_path):
    _scheduler_receipt(tmp_path)
    status = tmp_path / "reports" / "active" / "corrective_wizard_proof_scheduler_status.json"
    _write_json(
        status,
        {
            "attempt_date_utc": "2026-08-10",
            "external_attempt_made": True,
            "execution_requested": False,
            "status": "PLANNED",
            "registered_rerun_status": "NOT_EVALUATED",
            "next_external_attempt_eligible_at": "2026-08-11T00:00:00+00:00",
        },
    )
    checkpoint = tmp_path / "reports" / "active" / "seven_stage_goal_checkpoint.csv"
    checkpoint.write_text(
        "stage,status,next_action\n"
        "4,IN_PROGRESS,advance_registered_rerun_to_unaccounted_current_family_cohort\n",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "reports" / "active" / "registered_research_rerun_contract.json",
        {"contract_id": "registeredrerun-prior"},
    )
    python = _python(tmp_path)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        execution_path = _complete_stage4_generation(tmp_path)
        _scheduler_receipt(
            tmp_path,
            now=datetime(2026, 8, 10, 12, 0, 1, tzinfo=UTC),
            internal_continuation_only=True,
            external_attempt=False,
            registered_rerun_receipt_path=execution_path,
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=runner,
    )

    assert calls == [
        [
            str(python),
            "-m",
            HEAVY_MODULE,
            "--execute",
            "--internal-continuation-only",
        ]
    ]
    assert result["internal_registered_continuation_needed"] is True
    assert result["launcher_status"] == "INTERNAL_REGISTERED_CONTINUATION_COMPLETED"
    assert result["heavy_scheduler_receipt_valid"] is True
    assert result["external_attempt_made_by_launcher"] is False


def test_stage4_generation_pass_claim_without_rotation_fails_closed(tmp_path):
    status = _status(tmp_path, attempted=True)
    _write_json(
        status,
        {
            "attempt_date_utc": "2026-08-10",
            "external_attempt_made": True,
            "next_external_attempt_eligible_at": "2026-08-11T00:00:00+00:00",
            "queue_eligible": 28,
            "responses_captured_after": 28,
            "accepted_mode_evidence_cells": 28,
            "parity_refresh_status": "PASS",
            "credit_reconciliation_status": "PASS_RECONCILED",
            **_capture_reconciliation_complete_fields(tmp_path),
            "registered_rerun_status": "PASS_REGISTERED_RERUN_ACCOUNTED",
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    active = tmp_path / "reports" / "active"
    _write_json(
        active / "registered_research_rerun_contract.json",
        {"contract_id": "registeredrerun-prior"},
    )
    (active / "seven_stage_goal_checkpoint.csv").write_text(
        "stage,status,next_action\n"
        "4,IN_PROGRESS,advance_registered_rerun_to_unaccounted_current_family_cohort\n",
        encoding="utf-8",
    )

    def runner(command, **kwargs):
        _scheduler_receipt(
            tmp_path,
            internal_continuation_only=True,
            external_attempt=False,
            registered_rerun_status="PASS_REGISTERED_RERUN_ACCOUNTED",
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=_python(tmp_path),
        runner=runner,
    )

    assert result["launcher_status"] == ("INTERNAL_REGISTERED_CONTINUATION_INCOMPLETE")
    assert result["blocker"] == ("registered_rerun_generation_not_terminal_or_not_rotated")
    assert result["internal_continuation_prior_stage4_contract_id"] == ("registeredrerun-prior")
    assert result["external_attempt_made_by_launcher"] is False
    assert _launcher_exit_code(result) == 2


def test_reviewed_dynamic_v2_activation_refreshes_same_day_without_external_calls(
    tmp_path,
):
    _scheduler_receipt(
        tmp_path,
        stage3_complete=False,
        registered_rerun_status="NOT_EVALUATED",
    )
    activation_id = _applied_dynamic_v2_state(tmp_path)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        _write_dynamic_refresh_status(
            tmp_path,
            activation_id=activation_id,
            status="PASS",
            generation=2,
            captured=8,
            exact=8,
            refreshed=8,
        )
        _scheduler_receipt(
            tmp_path,
            now=datetime(2026, 8, 10, 12, 0, 1, tzinfo=UTC),
            status="DEFERRED_SAME_UTC_DAY",
            internal_continuation_only=True,
            external_attempt=False,
            stage3_complete=False,
            registered_rerun_status="NOT_EVALUATED",
            dynamic_refresh_status="PASS",
            dynamic_generation=2,
            dynamic_refreshed=8,
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=_python(tmp_path),
        runner=runner,
    )

    assert calls == [
        [
            str(_python(tmp_path)),
            "-m",
            HEAVY_MODULE,
            "--execute",
            "--internal-continuation-only",
        ]
    ]
    assert result["internal_continuation_reason"] == "stage3_dynamic_v2_refresh"
    assert result["launcher_status"] == "INTERNAL_DYNAMIC_V2_REFRESH_COMPLETED"
    assert result["scheduler_dynamic_v2_proof_refresh_status"] == "PASS"
    assert result["scheduler_dynamic_v2_comparator_generation"] == 2
    assert result["scheduler_dynamic_v2_proofs_refreshed"] == 8
    assert result["external_attempt_made_by_launcher"] is False
    assert _launcher_exit_code(result) == 0


def test_reviewed_ou_v3_activation_refreshes_same_day_without_external_calls(
    tmp_path,
):
    _scheduler_receipt(
        tmp_path,
        stage3_complete=False,
        registered_rerun_status="NOT_EVALUATED",
    )
    activation_id = _applied_ou_v3_state(tmp_path)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        _write_ou_refresh_status(
            tmp_path,
            activation_id=activation_id,
            status="PASS",
            generation=3,
            captured=8,
            exact=8,
            refreshed=8,
        )
        _scheduler_receipt(
            tmp_path,
            now=datetime(2026, 8, 10, 12, 0, 1, tzinfo=UTC),
            status="DEFERRED_SAME_UTC_DAY",
            internal_continuation_only=True,
            external_attempt=False,
            stage3_complete=False,
            registered_rerun_status="NOT_EVALUATED",
            ou_refresh_status="PASS",
            ou_generation=3,
            ou_refreshed=8,
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=_python(tmp_path),
        runner=runner,
    )

    assert calls == [
        [
            str(_python(tmp_path)),
            "-m",
            HEAVY_MODULE,
            "--execute",
            "--internal-continuation-only",
        ]
    ]
    assert result["internal_continuation_reason"] == "stage3_ou_v3_refresh"
    assert result["launcher_status"] == "INTERNAL_OU_V3_REFRESH_COMPLETED"
    assert result["scheduler_ou_v3_proof_refresh_status"] == "PASS"
    assert result["scheduler_ou_v3_comparator_generation"] == 3
    assert result["scheduler_ou_v3_proofs_refreshed"] == 8
    assert result["external_attempt_made_by_launcher"] is False
    assert _launcher_exit_code(result) == 0


def test_reviewed_ou_v4_activation_refreshes_same_day_without_external_calls(
    tmp_path,
    monkeypatch,
):
    _scheduler_receipt(
        tmp_path,
        stage3_complete=False,
        registered_rerun_status="NOT_EVALUATED",
    )
    activation_id = "ouv4activation_test"
    refresh_state = {"complete": False}
    monkeypatch.setattr(
        launcher,
        "_validated_ou_v4_activation",
        lambda root: {"activation_id": activation_id},
    )
    monkeypatch.setattr(
        launcher,
        "_ou_v4_refresh_complete",
        lambda **_: refresh_state["complete"],
    )
    calls = []

    def runner(command, **kwargs):
        del kwargs
        calls.append(command)
        refresh_state["complete"] = True
        _scheduler_receipt(
            tmp_path,
            now=datetime(2026, 8, 10, 12, 0, 1, tzinfo=UTC),
            status="DEFERRED_SAME_UTC_DAY",
            internal_continuation_only=True,
            external_attempt=False,
            stage3_complete=False,
            registered_rerun_status="NOT_EVALUATED",
            ou_v4_refresh_status="PASS",
            ou_v4_generation=4,
            ou_v4_refreshed=8,
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=_python(tmp_path),
        runner=runner,
    )

    assert calls == [
        [
            str(_python(tmp_path)),
            "-m",
            HEAVY_MODULE,
            "--execute",
            "--internal-continuation-only",
        ]
    ]
    assert result["internal_continuation_reason"] == "stage3_ou_v4_refresh"
    assert result["launcher_status"] == "INTERNAL_OU_V4_REFRESH_COMPLETED"
    assert result["scheduler_ou_v4_proof_refresh_status"] == "PASS"
    assert result["scheduler_ou_v4_comparator_generation"] == 4
    assert result["scheduler_ou_v4_proofs_refreshed"] == 8
    assert result["external_attempt_made_by_launcher"] is False
    assert _launcher_exit_code(result) == 0


def test_reviewed_ou_v5_activation_refreshes_same_day_without_external_calls(
    tmp_path,
    monkeypatch,
):
    _scheduler_receipt(
        tmp_path,
        stage3_complete=False,
        registered_rerun_status="NOT_EVALUATED",
    )
    activation_id = "ouv5activation_test"
    refresh_state = {"complete": False}
    monkeypatch.setattr(
        launcher,
        "_validated_ou_v5_activation",
        lambda root: {"activation_id": activation_id},
    )
    monkeypatch.setattr(
        launcher,
        "_ou_v5_refresh_complete",
        lambda **_: refresh_state["complete"],
    )
    calls = []

    def runner(command, **kwargs):
        del kwargs
        calls.append(command)
        refresh_state["complete"] = True
        _scheduler_receipt(
            tmp_path,
            now=datetime(2026, 8, 10, 12, 0, 1, tzinfo=UTC),
            status="DEFERRED_SAME_UTC_DAY",
            internal_continuation_only=True,
            external_attempt=False,
            stage3_complete=False,
            registered_rerun_status="NOT_EVALUATED",
            ou_v5_refresh_status="PASS",
            ou_v5_generation=5,
            ou_v5_refreshed=8,
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=_python(tmp_path),
        runner=runner,
    )

    assert calls == [
        [
            str(_python(tmp_path)),
            "-m",
            HEAVY_MODULE,
            "--execute",
            "--internal-continuation-only",
        ]
    ]
    assert result["internal_continuation_reason"] == "stage3_ou_v5_refresh"
    assert result["launcher_status"] == "INTERNAL_OU_V5_REFRESH_COMPLETED"
    assert result["scheduler_ou_v5_proof_refresh_status"] == "PASS"
    assert result["scheduler_ou_v5_comparator_generation"] == 5
    assert result["scheduler_ou_v5_proofs_refreshed"] == 8
    assert result["external_attempt_made_by_launcher"] is False
    assert _launcher_exit_code(result) == 0


def test_forged_dynamic_v2_activation_cannot_trigger_internal_refresh(tmp_path):
    _scheduler_receipt(
        tmp_path,
        stage3_complete=False,
        registered_rerun_status="NOT_EVALUATED",
    )
    _applied_dynamic_v2_state(tmp_path)
    status_path = tmp_path / "reports" / "active" / "wizard_dynamic_v2_activation_status.json"
    status = json.loads(status_path.read_text())
    status["immutable_activation_sha256"] = "0" * 64
    _write_json(status_path, status)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        runner=lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("forged activation must not launch an internal refresh")
        ),
    )

    assert result["internal_registered_continuation_needed"] is False
    assert result["launcher_status"] == "DEFERRED_SAME_UTC_DAY_LIGHTWEIGHT"


def test_dynamic_v2_scheduler_claim_without_bound_refresh_fails_closed(tmp_path):
    _scheduler_receipt(
        tmp_path,
        stage3_complete=False,
        registered_rerun_status="NOT_EVALUATED",
    )
    _applied_dynamic_v2_state(tmp_path)

    def runner(command, **kwargs):
        _scheduler_receipt(
            tmp_path,
            now=datetime(2026, 8, 10, 12, 0, 1, tzinfo=UTC),
            status="DEFERRED_SAME_UTC_DAY",
            internal_continuation_only=True,
            external_attempt=False,
            stage3_complete=False,
            registered_rerun_status="NOT_EVALUATED",
            dynamic_refresh_status="PASS",
            dynamic_generation=2,
            dynamic_refreshed=8,
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=_python(tmp_path),
        runner=runner,
    )

    assert result["launcher_status"] == "INTERNAL_DYNAMIC_V2_REFRESH_INCOMPLETE"
    assert result["blocker"] == "reviewed_dynamic_v2_refresh_not_complete"
    assert result["external_attempt_made_by_launcher"] is False
    assert _launcher_exit_code(result) == 2


def test_same_day_stage5_handoff_failure_recovers_without_external_requests(
    tmp_path,
    monkeypatch,
):
    _patch_verified_learning_audit(monkeypatch)
    _scheduler_receipt(tmp_path)
    _accepted_stage4_state(tmp_path, learning_status="HANDOFF_FAILED")
    python = _python(tmp_path)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        _write_learning_status(
            tmp_path,
            status="PASS_RESEARCH_LEARNING_GATES",
        )
        _scheduler_receipt(
            tmp_path,
            now=datetime(2026, 8, 10, 12, 0, 1, tzinfo=UTC),
            internal_continuation_only=True,
            external_attempt=False,
            registered_rerun_status="ALREADY_COMPLETE",
            learning_handoff_status="PASS_RESEARCH_LEARNING_GATES",
            stage5_research_gate_pass=True,
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=python,
        runner=runner,
    )

    assert calls == [
        [
            str(python),
            "-m",
            HEAVY_MODULE,
            "--execute",
            "--internal-continuation-only",
        ]
    ]
    assert result["internal_continuation_reason"] == "stage5_handoff_recovery"
    assert result["launcher_status"] == "INTERNAL_STAGE5_RECOVERY_COMPLETED"
    assert result["scheduler_registered_rerun_status"] == "ALREADY_COMPLETE"
    assert result["scheduler_learning_handoff_status"] == ("PASS_RESEARCH_LEARNING_GATES")
    assert result["scheduler_stage5_research_gate_pass"] is True
    assert result["external_attempt_made_by_launcher"] is False
    assert _launcher_exit_code(result) == 0


def test_terminal_stage5_rejection_is_not_retried_same_day(tmp_path, monkeypatch):
    _patch_verified_learning_audit(monkeypatch)
    _scheduler_receipt(tmp_path)
    _accepted_stage4_state(
        tmp_path,
        learning_status="REJECTED_RESEARCH_LEARNING_GATES",
    )

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        runner=lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("terminal Stage 5 research result must not be retried")
        ),
    )

    assert result["internal_registered_continuation_needed"] is False
    assert result["internal_continuation_reason"] == ""
    assert result["launcher_status"] == "DEFERRED_SAME_UTC_DAY_LIGHTWEIGHT"


def test_structural_terminal_stage5_receipt_is_rejected_when_full_audit_blocks(
    tmp_path,
    monkeypatch,
):
    _accepted_stage4_state(
        tmp_path,
        learning_status="PASS_RESEARCH_LEARNING_GATES",
    )
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_registered_learning."
        "latest_verified_registered_learning",
        lambda **_: {
            "status": "BLOCKED_REGISTERED_LEARNING_EVIDENCE",
            "evidence_valid": False,
            "registered_execution_id": "execution-accepted",
            "stage5_research_gate_pass": False,
            "order_submission_performed": False,
            "promotion_authority": False,
            "testnet_candidate_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    learning = json.loads(
        (tmp_path / "reports" / "active" / "registered_learning_research_status.json").read_text()
    )

    assert not _terminal_learning_status_valid(
        root=tmp_path,
        learning=learning,
        execution_id="execution-accepted",
    )


def test_verified_already_complete_stage5_status_is_terminal(tmp_path, monkeypatch):
    _accepted_stage4_state(
        tmp_path,
        learning_status="PASS_RESEARCH_LEARNING_GATES",
    )
    active_path = tmp_path / "reports" / "active" / "registered_learning_research_status.json"
    active = json.loads(active_path.read_text())
    active["status"] = "ALREADY_COMPLETE"
    _write_json(active_path, active)
    _patch_verified_learning_audit(monkeypatch)

    assert _terminal_learning_status_valid(
        root=tmp_path,
        learning=active,
        execution_id="execution-accepted",
    )


def test_forged_terminal_stage5_status_without_immutable_receipt_is_retried(
    tmp_path,
    monkeypatch,
):
    _patch_verified_learning_audit(monkeypatch)
    _scheduler_receipt(tmp_path)
    _accepted_stage4_state(tmp_path, learning_status="HANDOFF_FAILED")
    _write_learning_status(
        tmp_path,
        status="PASS_RESEARCH_LEARNING_GATES",
        immutable=False,
    )
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        _write_learning_status(
            tmp_path,
            status="REJECTED_RESEARCH_LEARNING_GATES",
        )
        _scheduler_receipt(
            tmp_path,
            now=datetime(2026, 8, 10, 12, 0, 1, tzinfo=UTC),
            internal_continuation_only=True,
            external_attempt=False,
            registered_rerun_status="ALREADY_COMPLETE",
            learning_handoff_status="REJECTED_RESEARCH_LEARNING_GATES",
            stage5_research_gate_pass=False,
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=_python(tmp_path),
        runner=runner,
    )

    assert calls
    assert result["internal_continuation_reason"] == "stage5_handoff_recovery"
    assert result["launcher_status"] == "INTERNAL_STAGE5_RECOVERY_COMPLETED"
    assert result["scheduler_stage5_research_gate_pass"] is False
    assert result["external_attempt_made_by_launcher"] is False


def test_scheduler_terminal_claim_without_learning_receipt_fails_closed(tmp_path):
    _scheduler_receipt(tmp_path)
    _accepted_stage4_state(tmp_path, learning_status="HANDOFF_FAILED")

    def runner(command, **kwargs):
        _scheduler_receipt(
            tmp_path,
            now=datetime(2026, 8, 10, 12, 0, 1, tzinfo=UTC),
            internal_continuation_only=True,
            external_attempt=False,
            registered_rerun_status="ALREADY_COMPLETE",
            learning_handoff_status="PASS_RESEARCH_LEARNING_GATES",
            stage5_research_gate_pass=True,
        )
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=_python(tmp_path),
        runner=runner,
    )

    assert result["launcher_status"] == "INTERNAL_STAGE5_RECOVERY_INCOMPLETE"
    assert result["blocker"] == "registered_learning_handoff_not_terminal"
    assert result["external_attempt_made_by_launcher"] is False
    assert _launcher_exit_code(result) == 2


def test_malformed_stage4_breadth_fails_closed_without_recovery(tmp_path):
    _scheduler_receipt(tmp_path)
    _accepted_stage4_state(tmp_path, learning_status="HANDOFF_FAILED")
    execution_status_path = (
        tmp_path / "reports" / "active" / "registered_research_rerun_execution.json"
    )
    execution_status = json.loads(execution_status_path.read_text())
    execution_path = tmp_path / execution_status["execution_receipt_path"]
    execution = json.loads(execution_path.read_text())
    execution["stage4_independent_supporting_pairs"] = "not-an-integer"
    _write_json(execution_path, execution)
    execution_status["execution_receipt_sha256"] = sha256(execution_path.read_bytes()).hexdigest()
    _write_json(execution_status_path, execution_status)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        runner=lambda *_, **__: (_ for _ in ()).throw(
            AssertionError("malformed breadth must not authorize recovery")
        ),
    )

    assert result["internal_registered_continuation_needed"] is False
    assert result["launcher_status"] == "DEFERRED_SAME_UTC_DAY_LIGHTWEIGHT"
