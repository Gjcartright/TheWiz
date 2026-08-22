from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.orchestration import corrective_wizard_capture_reconciliation
from quant_platform.orchestration.corrective_wizard_capture_manifest import (
    build_corrective_wizard_capture_manifest,
)
from quant_platform.orchestration.corrective_wizard_capture_reconciliation import (
    frozen_capture_manifest_mutation_blockers,
    reconcile_corrective_wizard_capture_manifest,
    validate_capture_manifest_binding,
    validate_capture_reconciliation_evidence,
)
from quant_platform.orchestration.corrective_wizard_ou_v4_holdout import (
    _call_intent_path,
    _call_intent_payload,
)
from quant_platform.orchestration.corrective_wizard_ou_v4_holdout import (
    _write_or_validate_immutable_json as _write_ou_v4_immutable_json,
)
from quant_platform.orchestration.corrective_wizard_ou_v5_holdout import (
    _call_completion_path as _call_completion_path_v5,
)
from quant_platform.orchestration.corrective_wizard_ou_v5_holdout import (
    _call_completion_payload as _call_completion_payload_v5,
)
from quant_platform.orchestration.corrective_wizard_ou_v5_holdout import (
    _call_intent_path as _call_intent_path_v5,
)
from quant_platform.orchestration.corrective_wizard_ou_v5_holdout import (
    _call_intent_payload as _call_intent_payload_v5,
)
from quant_platform.orchestration.corrective_wizard_ou_v5_holdout import (
    _write_or_validate_immutable_json as _write_ou_v5_immutable_json,
)
from quant_platform.orchestration.corrective_wizard_ou_v6_holdout import (
    _call_completion_path as _call_completion_path_v6,
)
from quant_platform.orchestration.corrective_wizard_ou_v6_holdout import (
    _call_completion_payload as _call_completion_payload_v6,
)
from quant_platform.orchestration.corrective_wizard_ou_v6_holdout import (
    _call_intent_path as _call_intent_path_v6,
)
from quant_platform.orchestration.corrective_wizard_ou_v6_holdout import (
    _call_intent_payload as _call_intent_payload_v6,
)
from quant_platform.orchestration.corrective_wizard_ou_v6_holdout import (
    _write_or_validate_immutable_json as _write_ou_v6_immutable_json,
)
from tests.test_corrective_wizard_capture_manifest import (
    _copula_plans,
    _write_inputs,
    _write_v4_contract,
    _write_v5_contract,
    _write_v6_contract,
)

NOW = datetime(2026, 8, 11, 3, 30, tzinfo=UTC)


def test_immutable_reconciliation_publication_never_leaves_partial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "immutable" / "reconciliation.json"

    def fail_publish(
        _source: Path,
        _destination: Path,
        *,
        follow_symlinks: bool,
    ) -> None:
        assert follow_symlinks is False
        raise OSError("simulated publication interruption")

    monkeypatch.setattr(corrective_wizard_capture_reconciliation.os, "link", fail_publish)

    with pytest.raises(OSError, match="simulated publication interruption"):
        corrective_wizard_capture_reconciliation._write_or_validate_immutable_json(
            {"reconciliation_id": "test"},
            target,
        )

    assert not target.exists()
    assert list(target.parent.glob(".*.tmp")) == []


def test_reconciliation_reports_pending_without_claiming_authority(tmp_path):
    paths, manifest_path = _build_manifest(tmp_path)

    result = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )

    assert result.summary["status"] == "PENDING"
    assert result.summary["completed_calls"] == 0
    assert result.summary["pending_calls"] == 13
    assert result.summary["blocked_calls"] == 0
    assert result.summary["reconciliation_id"] == ""
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_reconciliation_binds_all_thirteen_responses_and_is_immutable(tmp_path):
    paths, manifest_path = _build_manifest(tmp_path)
    _capture_all_calls(tmp_path, manifest_path, paths["proof_path"])

    first = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )
    second = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=datetime(2026, 8, 11, 4, 0, tzinfo=UTC),
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )

    assert first.summary["status"] == "PASS"
    assert first.summary["completed_calls"] == 13
    assert first.summary["pending_calls"] == 0
    assert first.summary["blocked_calls"] == 0
    assert first.summary["all_response_hashes_bound"] is True
    assert first.summary["all_request_bindings_valid"] is True
    assert first.summary["reconciliation_id"] == second.summary["reconciliation_id"]
    assert Path(first.paths["immutable_reconciliation"]).is_file()

    evidence = _scheduler_evidence(first.summary)
    validation = validate_capture_reconciliation_evidence(root=tmp_path, evidence=evidence)
    assert validation["status"] == "PASS"
    assert validation["required_calls"] == 13
    assert validation["completed_calls"] == 13


def test_frozen_manifest_blocks_mutation_until_immutable_reconciliation(tmp_path):
    paths, manifest_path = _build_manifest(tmp_path)
    pending = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )

    blockers = frozen_capture_manifest_mutation_blockers(root=tmp_path)
    assert pending.summary["status"] == "PENDING"
    assert blockers == [f"frozen_capture_manifest_unresolved:{pending.summary['manifest_id']}"]

    _capture_all_calls(tmp_path, manifest_path, paths["proof_path"])
    complete = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )

    assert complete.summary["status"] == "PASS"
    assert frozen_capture_manifest_mutation_blockers(root=tmp_path) == []


def test_immutable_reconciliation_validator_rejects_pending_and_tampering(tmp_path):
    paths, manifest_path = _build_manifest(tmp_path)
    pending = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )
    pending_validation = validate_capture_reconciliation_evidence(
        root=tmp_path,
        evidence=_scheduler_evidence(pending.summary),
    )
    assert pending_validation["status"] == "BLOCKED"
    assert "capture_reconciliation_status_not_pass" in pending_validation["blockers"]

    _capture_all_calls(tmp_path, manifest_path, paths["proof_path"])
    complete = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )
    evidence = _scheduler_evidence(complete.summary)
    immutable = tmp_path / evidence["capture_reconciliation_immutable_path"]
    immutable.chmod(0o600)
    immutable.write_text('{"tampered":true}\n', encoding="utf-8")

    tampered = validate_capture_reconciliation_evidence(root=tmp_path, evidence=evidence)
    assert tampered["status"] == "BLOCKED"
    assert "capture_reconciliation_immutable_binding_invalid" in tampered["blockers"]


def test_reconciliation_blocks_a_request_changed_after_registration(tmp_path):
    paths, manifest_path = _build_manifest(tmp_path)
    _capture_all_calls(tmp_path, manifest_path, paths["proof_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ou_call = next(call for call in manifest["calls"] if call["lane"] == "ou_v3_holdout")
    request_path = tmp_path / ou_call["request_path"]
    request_path.write_text('{"tampered":true}\n', encoding="utf-8")

    result = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )

    assert result.summary["status"] == "BLOCKED_EVIDENCE"
    assert result.summary["blocked_calls"] == 1
    assert any("request_hash_mismatch" in item for item in result.summary["blockers"])
    assert result.summary["reconciliation_id"] == ""


def test_reconciliation_binds_all_eight_ou_v4_responses(tmp_path):
    paths = _write_inputs(tmp_path)
    _write_v4_contract(tmp_path)
    manifest = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: _copula_plans(),
    )
    manifest_path = Path(manifest.paths["immutable_manifest"])
    _capture_all_calls(tmp_path, manifest_path, paths["proof_path"])

    result = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )

    assert result.summary["status"] == "PASS"
    assert result.summary["required_calls"] == 21
    assert result.summary["completed_calls"] == 21
    assert result.summary["lane_totals"]["ou_v4_holdout"] == {
        "required_calls": 8,
        "completed_calls": 8,
        "pending_calls": 0,
        "blocked_calls": 0,
        "completed_credits": 16,
    }
    assert result.summary["all_ou_v4_call_intents_bound"] is True
    assert result.summary["ou_v4_call_intents_observed"] == 8


def test_reconciliation_blocks_ou_v4_response_without_precall_intent(tmp_path):
    paths = _write_inputs(tmp_path)
    _write_v4_contract(tmp_path)
    manifest = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: [],
    )
    manifest_path = Path(manifest.paths["immutable_manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = next(call for call in payload["calls"] if call["lane"] == "ou_v4_holdout")
    _write_json(tmp_path / first["expected_output_path"], {"status": "ok"})

    result = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )

    assert result.summary["status"] == "BLOCKED_EVIDENCE"
    assert result.summary["blocked_calls"] == 1
    assert any(
        "ou_v4_response_has_no_call_attempt_intent" in blocker
        for blocker in result.summary["blockers"]
    )


def test_reconciliation_binds_all_eight_ou_v5_responses_and_intents(tmp_path):
    paths = _write_inputs(tmp_path)
    _write_v5_contract(tmp_path)
    manifest = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: [],
    )
    manifest_path = Path(manifest.paths["immutable_manifest"])
    _capture_all_calls(tmp_path, manifest_path, paths["proof_path"])

    result = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )

    assert result.summary["status"] == "PASS"
    assert result.summary["required_calls"] == 13
    assert result.summary["lane_totals"]["ou_v5_holdout"] == {
        "required_calls": 8,
        "completed_calls": 8,
        "pending_calls": 0,
        "blocked_calls": 0,
        "completed_credits": 16,
    }
    assert result.summary["all_ou_v5_call_intents_bound"] is True
    assert result.summary["ou_v5_call_intents_observed"] == 8
    assert result.summary["all_ou_v5_call_completions_bound"] is True
    assert result.summary["ou_v5_completion_receipts_observed"] == 8
    validation = validate_capture_reconciliation_evidence(
        root=tmp_path,
        evidence=_scheduler_evidence(result.summary),
    )
    assert validation["status"] == "PASS"


def test_reconciliation_binds_all_eight_ou_v6_responses_intents_and_completions(
    tmp_path,
):
    paths = _write_inputs(tmp_path)
    _write_v6_contract(tmp_path)
    manifest = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: [],
    )
    manifest_path = Path(manifest.paths["immutable_manifest"])
    _capture_all_calls(tmp_path, manifest_path, paths["proof_path"])

    result = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )

    assert result.summary["status"] == "PASS"
    assert result.summary["lane_totals"]["ou_v6_holdout"] == {
        "required_calls": 8,
        "completed_calls": 8,
        "pending_calls": 0,
        "blocked_calls": 0,
        "completed_credits": 16,
    }
    assert result.summary["all_ou_v6_call_intents_bound"] is True
    assert result.summary["ou_v6_call_intents_observed"] == 8
    assert result.summary["all_ou_v6_call_completions_bound"] is True
    assert result.summary["ou_v6_completion_receipts_observed"] == 8
    validation = validate_capture_reconciliation_evidence(
        root=tmp_path,
        evidence=_scheduler_evidence(result.summary),
    )
    assert validation["status"] == "PASS"


def test_completion_mutation_revokes_previously_passing_reconciliation(tmp_path):
    paths = _write_inputs(tmp_path)
    _write_v5_contract(tmp_path)
    manifest = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: [],
    )
    manifest_path = Path(manifest.paths["immutable_manifest"])
    _capture_all_calls(tmp_path, manifest_path, paths["proof_path"])
    result = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )
    assert result.summary["status"] == "PASS"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = next(call for call in payload["calls"] if call["lane"] == "ou_v5_holdout")
    completion_path = _call_completion_path_v5(
        root=tmp_path,
        call_id=first["call_id"],
    )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["pair"] = "TAMPERED-PAIR"
    completion_path.chmod(0o600)
    completion_path.write_text(
        json.dumps(completion, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation = validate_capture_reconciliation_evidence(
        root=tmp_path,
        evidence=_scheduler_evidence(result.summary),
    )

    assert validation["status"] == "BLOCKED"
    assert "capture_reconciliation_receipt_contract_invalid" in validation["blockers"]


def test_reconciliation_blocks_ou_v5_response_without_precall_intent(tmp_path):
    paths = _write_inputs(tmp_path)
    _write_v5_contract(tmp_path)
    manifest = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: [],
    )
    manifest_path = Path(manifest.paths["immutable_manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = next(call for call in payload["calls"] if call["lane"] == "ou_v5_holdout")
    _write_json(tmp_path / first["expected_output_path"], {"status": "ok"})

    result = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )

    assert result.summary["status"] == "BLOCKED_EVIDENCE"
    assert any(
        "ou_v5_response_has_no_call_attempt_intent" in blocker
        for blocker in result.summary["blockers"]
    )


def test_reconciliation_blocks_ou_v5_response_and_intent_without_completion(tmp_path):
    paths = _write_inputs(tmp_path)
    _write_v5_contract(tmp_path)
    manifest = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: [],
    )
    manifest_path = Path(manifest.paths["immutable_manifest"])
    _capture_all_calls(tmp_path, manifest_path, paths["proof_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = next(call for call in payload["calls"] if call["lane"] == "ou_v5_holdout")
    _call_completion_path_v5(root=tmp_path, call_id=first["call_id"]).unlink()

    result = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )

    assert result.summary["status"] == "BLOCKED_EVIDENCE"
    assert result.summary["blocked_calls"] == 1
    assert result.summary["all_ou_v5_call_completions_bound"] is False
    assert result.summary["ou_v5_completion_receipts_observed"] == 7
    assert any("ou_v5_call_completion_missing" in blocker for blocker in result.summary["blockers"])


def test_reconciliation_blocks_tampered_ou_v4_precall_intent(tmp_path):
    paths = _write_inputs(tmp_path)
    _write_v4_contract(tmp_path)
    manifest = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: [],
    )
    manifest_path = Path(manifest.paths["immutable_manifest"])
    _capture_all_calls(tmp_path, manifest_path, paths["proof_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    call = next(item for item in payload["calls"] if item["lane"] == "ou_v4_holdout")
    intent_path = _call_intent_path(root=tmp_path, timestamp=NOW, call_id=call["call_id"])
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    intent["pair"] = "TAMPERED-PAIR"
    intent_path.chmod(0o600)
    intent_path.write_text(json.dumps(intent, sort_keys=True) + "\n", encoding="utf-8")

    result = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )

    assert result.summary["status"] == "BLOCKED_EVIDENCE"
    assert result.summary["blocked_calls"] == 1
    assert any("call_intent_binding_mismatch" in blocker for blocker in result.summary["blockers"])


def test_reconciliation_preserves_multi_day_ou_v4_attempt_lineage(tmp_path):
    paths = _write_inputs(tmp_path)
    _write_v4_contract(tmp_path)
    manifest = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: [],
    )
    manifest_path = Path(manifest.paths["immutable_manifest"])
    _capture_all_calls(tmp_path, manifest_path, paths["proof_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    call = next(item for item in payload["calls"] if item["lane"] == "ou_v4_holdout")
    contract_path = tmp_path / call["evidence_path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    binding = next(
        row
        for row in contract["holdout_bindings"]
        if row["pair"] == call["pair"]
        and row["exact_mode"] == call["exact_mode"]
        and row["orientation"] == call["orientation"]
    )
    retry_at = NOW + timedelta(days=1)
    retry_path = _call_intent_path(
        root=tmp_path,
        timestamp=retry_at,
        call_id=call["call_id"],
    )
    retry_intent = _call_intent_payload(
        root=tmp_path,
        timestamp=retry_at,
        contract_path=contract_path,
        binding=binding,
        call_id=call["call_id"],
    )
    _write_ou_v4_immutable_json(retry_intent, retry_path)

    result = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=retry_at,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )
    detail = pd.read_csv(result.paths["detail"])
    row = detail.loc[detail["call_id"].eq(call["call_id"])].iloc[0]

    assert result.summary["status"] == "PASS"
    assert result.summary["ou_v4_call_intents_observed"] == 9
    assert int(row["call_intent_count"]) == 2
    assert bool(row["multi_day_recovery"]) is True
    assert int(row["ambiguous_prior_attempts"]) == 1


def test_reconciliation_validator_requires_retained_ou_v4_intent_files(tmp_path):
    paths = _write_inputs(tmp_path)
    _write_v4_contract(tmp_path)
    manifest = build_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: [],
    )
    manifest_path = Path(manifest.paths["immutable_manifest"])
    _capture_all_calls(tmp_path, manifest_path, paths["proof_path"])
    complete = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    call = next(item for item in payload["calls"] if item["lane"] == "ou_v4_holdout")
    _call_intent_path(root=tmp_path, timestamp=NOW, call_id=call["call_id"]).unlink()

    validation = validate_capture_reconciliation_evidence(
        root=tmp_path,
        evidence=_scheduler_evidence(complete.summary),
    )

    assert validation["status"] == "BLOCKED"
    assert "capture_reconciliation_receipt_contract_invalid" in validation["blockers"]


def test_reconciliation_blocks_a_copula_request_byte_change_after_registration(
    tmp_path,
):
    paths, manifest_path = _build_manifest(tmp_path)
    _capture_all_calls(tmp_path, manifest_path, paths["proof_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    copula_call = next(call for call in manifest["calls"] if call["lane"] == "copula_behavioral")
    request_path = tmp_path / copula_call["request_path"]
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    request_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    result = reconcile_corrective_wizard_capture_manifest(
        root=tmp_path,
        now=NOW,
        manifest_path=manifest_path,
        proof_path=paths["proof_path"],
    )

    assert result.summary["status"] == "BLOCKED_EVIDENCE"
    assert result.summary["blocked_calls"] == 2
    assert all(
        "request_hash_mismatch" in item
        for item in result.summary["blockers"]
        if item.startswith("wizcall_")
    )


def test_capture_manifest_binding_validator_rejects_tampering(tmp_path):
    _, manifest_path = _build_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence = {
        "manifest_id": manifest["manifest_id"],
        "manifest_path": str(manifest_path.relative_to(tmp_path)),
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "pending_calls": manifest["pending_calls"],
        "planned_credits": manifest["planned_credits"],
        "lane_totals": manifest["lane_totals"],
    }

    valid = validate_capture_manifest_binding(
        root=tmp_path,
        manifest_id=evidence["manifest_id"],
        manifest_path=evidence["manifest_path"],
        manifest_sha256=evidence["manifest_sha256"],
        expected_pending_calls=evidence["pending_calls"],
        expected_planned_credits=evidence["planned_credits"],
        expected_lane_totals=evidence["lane_totals"],
    )
    assert valid["status"] == "PASS"

    manifest_path.chmod(0o600)
    manifest_path.write_text('{"tampered":true}\n', encoding="utf-8")
    tampered = validate_capture_manifest_binding(
        root=tmp_path,
        manifest_id=evidence["manifest_id"],
        manifest_path=evidence["manifest_path"],
        manifest_sha256=evidence["manifest_sha256"],
    )
    assert tampered["status"] == "BLOCKED"
    assert "capture_manifest_immutable_binding_invalid" in tampered["blockers"]


def _build_manifest(root: Path) -> tuple[dict[str, Path], Path]:
    paths = _write_inputs(root)
    manifest = build_corrective_wizard_capture_manifest(
        root=root,
        now=NOW,
        **paths,
        copula_plan_provider=lambda _: _copula_plans(),
    )
    return paths, Path(manifest.paths["immutable_manifest"])


def _capture_all_calls(root: Path, manifest_path: Path, proof_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exact_rows = []
    for call in manifest["calls"]:
        v5_completion = None
        v6_completion = None
        if call["lane"] == "exact_mode_backtest":
            request_path = root / "data" / "raw" / "exact" / "request.json"
            response_path = root / "data" / "raw" / "exact" / "response.json"
            _write_json(request_path, {"exact": "request"})
            _write_json(response_path, {"status": "ok"})
            exact_rows.append(
                {
                    "pair": call["pair"],
                    "pair_group_id": call["pair_group_id"],
                    "local_interval": "1d",
                    "exact_mode": call["exact_mode"],
                    "orientation": call["orientation"],
                    "proof_observations": call["observations"],
                    "vendor_response_captured": True,
                    "mode_proof_status": "completed",
                    "request_path": str(request_path.relative_to(root)),
                    "response_path": str(response_path.relative_to(root)),
                    "evidence_path": str(response_path.relative_to(root)),
                }
            )
            continue
        request_path = root / call["request_path"]
        response_path = root / call["expected_output_path"]
        if call["lane"] == "copula_behavioral":
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(
                json.dumps({"pair": call["pair"]}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if call["lane"] == "ou_v4_holdout":
            contract_path = root / call["evidence_path"]
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            binding = next(
                row
                for row in contract["holdout_bindings"]
                if row["pair"] == call["pair"]
                and row["exact_mode"] == call["exact_mode"]
                and row["orientation"] == call["orientation"]
            )
            intent_path = _call_intent_path(
                root=root,
                timestamp=NOW,
                call_id=call["call_id"],
            )
            intent = _call_intent_payload(
                root=root,
                timestamp=NOW,
                contract_path=contract_path,
                binding=binding,
                call_id=call["call_id"],
            )
            _write_ou_v4_immutable_json(intent, intent_path)
        if call["lane"] == "ou_v5_holdout":
            contract_path = root / call["evidence_path"]
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            binding = next(
                row
                for row in contract["holdout_bindings"]
                if row["pair"] == call["pair"]
                and row["exact_mode"] == call["exact_mode"]
                and row["orientation"] == call["orientation"]
            )
            intent_path = _call_intent_path_v5(
                root=root,
                timestamp=NOW,
                call_id=call["call_id"],
            )
            intent = _call_intent_payload_v5(
                root=root,
                timestamp=NOW,
                contract_path=contract_path,
                binding=binding,
                call_id=call["call_id"],
            )
            _write_ou_v5_immutable_json(intent, intent_path)
            v5_completion = (contract_path, binding, intent_path)
        if call["lane"] == "ou_v6_holdout":
            contract_path = root / call["evidence_path"]
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            binding = next(
                row
                for row in contract["holdout_bindings"]
                if row["pair"] == call["pair"]
                and row["exact_mode"] == call["exact_mode"]
                and row["orientation"] == call["orientation"]
            )
            intent_path = _call_intent_path_v6(
                root=root,
                timestamp=NOW,
                call_id=call["call_id"],
            )
            intent = _call_intent_payload_v6(
                root=root,
                timestamp=NOW,
                contract_path=contract_path,
                binding=binding,
                call_id=call["call_id"],
            )
            _write_ou_v6_immutable_json(intent, intent_path)
            v6_completion = (contract_path, binding, intent_path)
        _write_json(response_path, {"status": "ok", "call_id": call["call_id"]})
        if v5_completion is not None:
            contract_path, binding, intent_path = v5_completion
            completion_path = _call_completion_path_v5(
                root=root,
                call_id=call["call_id"],
            )
            completion = _call_completion_payload_v5(
                root=root,
                timestamp=NOW,
                contract_path=contract_path,
                binding=binding,
                call_id=call["call_id"],
                intent_path=intent_path,
                response_path=response_path,
            )
            _write_ou_v5_immutable_json(completion, completion_path)
        if v6_completion is not None:
            contract_path, binding, intent_path = v6_completion
            completion_path = _call_completion_path_v6(
                root=root,
                call_id=call["call_id"],
            )
            completion = _call_completion_payload_v6(
                root=root,
                timestamp=NOW,
                contract_path=contract_path,
                binding=binding,
                call_id=call["call_id"],
                intent_path=intent_path,
                response_path=response_path,
            )
            _write_ou_v6_immutable_json(completion, completion_path)
    pd.DataFrame(exact_rows).to_csv(proof_path, index=False)


def _scheduler_evidence(summary: dict[str, object]) -> dict[str, object]:
    complete = summary.get("status") == "PASS"
    return {
        "capture_manifest_accounting_valid": True,
        "capture_reconciliation_status": summary.get("status", ""),
        "capture_reconciliation_valid": complete,
        "capture_reconciliation_complete": complete,
        "capture_reconciliation_id": summary.get("reconciliation_id", ""),
        "capture_reconciliation_immutable_path": summary.get("immutable_reconciliation_path", ""),
        "capture_reconciliation_immutable_sha256": summary.get(
            "immutable_reconciliation_sha256", ""
        ),
        "capture_reconciliation_manifest_id": summary.get("manifest_id", ""),
        "capture_reconciliation_manifest_path": summary.get("manifest_path", ""),
        "capture_reconciliation_manifest_sha256": summary.get("manifest_sha256", ""),
        "capture_reconciliation_required_calls": summary.get("required_calls", 0),
        "capture_reconciliation_completed_calls": summary.get("completed_calls", 0),
        "capture_reconciliation_pending_calls": summary.get("pending_calls", 0),
        "capture_reconciliation_blocked_calls": summary.get("blocked_calls", 0),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
