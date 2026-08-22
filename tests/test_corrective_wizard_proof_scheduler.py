from __future__ import annotations

import json
import plistlib
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration import corrective_wizard_proof_scheduler
from quant_platform.orchestration.corrective_external_effects import (
    RESEARCH_EXTERNAL_EFFECT_PROFILE,
    external_effect_issuer_session,
    read_authorized_credential,
    reserved_external_effect_session,
)
from quant_platform.orchestration.corrective_wizard_capture_manifest import (
    _load_or_create_source_receipt,
)
from quant_platform.orchestration.corrective_wizard_proof_scheduler import (
    PROOF_LOCK_TIMEOUT_SECONDS,
    WIZARD_BACKTEST_ENDPOINT,
    WIZARD_COPULA_ENDPOINT,
    WIZARD_CREDITS_ENDPOINT,
    _launch_agent_plist,
    run_corrective_wizard_proof_cycle,
)
from quant_platform.orchestration.effect_authority import EffectAuthority
from quant_platform.orchestration.corrective_runtime import (
    SCHEDULER_BOOTSTRAP_MODULE,
    SCHEDULER_CAPABILITY_PROFILES,
    scheduler_contract,
)
from quant_platform.wizard_credit_ledger import (
    PROOF_LANE,
    reconcile_wizard_credit_lane,
    reserve_wizard_credit_lane,
)
from quant_platform.wizard_hyperliquid_mode_proof import (
    count_completed_exact_mode_proofs,
)
from tests.capture_reconciliation_support import (
    write_valid_capture_reconciliation_evidence,
)

NOW = datetime(2026, 8, 10, 0, 5, tzinfo=UTC)
SECRET = "wizard-secret-value"
HASH = "a" * 64


@pytest.fixture(autouse=True)
def _authorized_external_effect_issuer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def unexpected_live_credit_fetch(**_kwargs):
        raise AssertionError("scheduler_test_unmocked_live_credit_fetch")

    monkeypatch.setattr(
        corrective_wizard_proof_scheduler,
        "fetch_credits_used",
        unexpected_live_credit_fetch,
    )
    authority = EffectAuthority(
        root=tmp_path,
        secret=b"scheduler-test-effect-authority-secret",
        issuer_id="scheduler-test-supervisor",
        profile=RESEARCH_EXTERNAL_EFFECT_PROFILE,
    )
    with external_effect_issuer_session(
        authority=authority,
        run_id="scheduler-test-run",
        intended_slot_id="scheduler-test-slot",
        source_fingerprint_sha256=HASH,
        runtime_fingerprint_sha256=HASH,
        configuration_fingerprint_sha256=HASH,
        provider_id="crypto_wizards",
        account_scope_id="wizard-research-test-account",
        allowed_targets=frozenset(
            {
                WIZARD_BACKTEST_ENDPOINT,
                WIZARD_COPULA_ENDPOINT,
                WIZARD_CREDITS_ENDPOINT,
            }
        ),
        allowed_credential_keys=frozenset({"CRYPTO_WIZARDS_API_KEY"}),
        max_total_requests=10_000,
        max_total_credits=10_000,
    ):
        yield


def _queue_builder(root: Path, *, eligible: int = 5) -> CommandResult:
    path = root / "reports" / "active" / "exhaustive_wizard_exact_mode_proof_queue.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("pair\n", encoding="utf-8")
    return CommandResult(paths={"queue": path}, summary={"eligible_rows": eligible})


def _result(
    *,
    eligible: int,
    selected: int,
    completed: int,
    selected_completed: int,
    responses_captured: int | None = None,
    selected_responses_captured: int | None = None,
    external_proof_requests: int | None = None,
    selected_request_failed: int = 0,
    selected_failures_quarantined: bool = False,
    credit_status: str = "PASS",
    credit_blocker: str = "",
) -> CommandResult:
    responses_captured = completed if responses_captured is None else responses_captured
    selected_responses_captured = (
        selected_completed if selected_responses_captured is None else selected_responses_captured
    )
    external_proof_requests = (
        selected_responses_captured
        if external_proof_requests is None and credit_status == "PASS"
        else 0
        if external_proof_requests is None
        else external_proof_requests
    )
    return CommandResult(
        paths={},
        summary={
            "queue_eligible": eligible,
            "selected": selected,
            "completed": completed,
            "queue_completed": completed,
            "queue_responses_captured": responses_captured,
            "selected_completed": selected_completed,
            "selected_responses_captured": selected_responses_captured,
            "external_proof_requests": external_proof_requests,
            "selected_request_failed": selected_request_failed,
            "selected_failures_quarantined": selected_failures_quarantined,
            "selected_credit_blocked": selected if credit_status != "PASS" else 0,
            "execution_enabled": True,
            "credit_preflight_status": credit_status,
            "credit_preflight_blocker": credit_blocker,
            "credits_used_before": 20,
            "credits_remaining_before": 980,
            "reserved_credits": 100,
        },
    )


def _write_secure_env(root: Path) -> None:
    path = root / ".env.local"
    path.write_text(f"CRYPTO_WIZARDS_API_KEY={SECRET}\n", encoding="utf-8")
    path.chmod(0o600)


def _capture_manifest_builder(
    *,
    exact_calls: int = 0,
    ou_calls: int = 0,
    ou_v4_calls: int = 0,
    ou_v5_calls: int = 0,
    ou_v6_calls: int = 0,
    copula_calls: int = 0,
):
    lane_totals = {
        "exact_mode_backtest": {"calls": exact_calls, "credits": exact_calls * 2},
        "ou_v3_holdout": {"calls": ou_calls, "credits": ou_calls * 2},
        "ou_v4_holdout": {"calls": ou_v4_calls, "credits": ou_v4_calls * 2},
        "ou_v5_holdout": {"calls": ou_v5_calls, "credits": ou_v5_calls * 2},
        "ou_v6_holdout": {"calls": ou_v6_calls, "credits": ou_v6_calls * 2},
        "copula_behavioral": {"calls": copula_calls, "credits": copula_calls},
    }
    total_calls = exact_calls + ou_calls + ou_v4_calls + ou_v5_calls + ou_v6_calls + copula_calls
    total_credits = (
        exact_calls * 2
        + ou_calls * 2
        + ou_v4_calls * 2
        + ou_v5_calls * 2
        + ou_v6_calls * 2
        + copula_calls
    )

    def builder(*, root, now, **_):
        calls = [
            {
                "call_id": f"test_{lane}_{index}",
                "lane": lane,
                "credit_cost": credit_cost,
            }
            for lane, count, credit_cost in (
                ("exact_mode_backtest", exact_calls, 2),
                ("ou_v3_holdout", ou_calls, 2),
                ("ou_v4_holdout", ou_v4_calls, 2),
                ("ou_v5_holdout", ou_v5_calls, 2),
                ("ou_v6_holdout", ou_v6_calls, 2),
                ("copula_behavioral", copula_calls, 1),
            )
            for index in range(count)
        ]
        manifest_core = {
            "schema_version": "thewiz.wizard_capture_cohort.v1",
            "calls": calls,
            "lane_totals": lane_totals,
            "pending_calls": total_calls,
            "planned_credits": total_credits,
            "proof_lane_credit_ceiling": total_credits,
            "intentional_cross_lane_overlap_policy": "test_fixture",
            "runtime_credit_preflight_required": True,
            "order_submission_included": False,
            "research_only": True,
            "candidate_promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        manifest_id = (
            "wizardcapture_"
            + sha256(
                json.dumps(
                    manifest_core,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()[:20]
        )
        immutable = root / "data" / "research" / "wizard_capture_manifests" / f"{manifest_id}.json"
        immutable.parent.mkdir(parents=True, exist_ok=True)
        immutable.write_text(
            json.dumps(
                {**manifest_core, "manifest_id": manifest_id},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        source = root / "reports" / "active" / "capture_manifest_test_source.json"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('{"fixture":true}\n', encoding="utf-8")
        source_receipt, source_receipt_path, _ = _load_or_create_source_receipt(
            root=root,
            manifest_id=manifest_id,
            immutable_manifest_path=immutable,
            source_artifacts=[
                {
                    "path": str(source.relative_to(root)),
                    "sha256": sha256(source.read_bytes()).hexdigest(),
                }
            ],
            retrofit=False,
        )
        status = root / "reports" / "active" / "capture_manifest_test.json"
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"status": status},
            summary={
                "status": "PASS",
                "capture_state": "READY_RUNTIME_CREDIT_PREFLIGHT_REQUIRED",
                "manifest_enforced": True,
                "manifest_id": manifest_id,
                "immutable_manifest_path": str(immutable.relative_to(root)),
                "immutable_manifest_sha256": sha256(immutable.read_bytes()).hexdigest(),
                "source_receipt_id": source_receipt["receipt_id"],
                "source_receipt_path": str(source_receipt_path.relative_to(root)),
                "source_receipt_sha256": sha256(source_receipt_path.read_bytes()).hexdigest(),
                "source_artifacts_sha256": source_receipt["source_artifacts_sha256"],
                "pending_calls": total_calls,
                "planned_credits": total_credits,
                "capture_eligible_now": True,
                "next_external_attempt_eligible_at": now.isoformat(),
                "blockers": [],
                "lane_totals": lane_totals,
            },
        )

    return builder


def _write_pending_manifest_scheduler_status(
    root: Path, manifest: CommandResult
) -> dict[str, object]:
    summary = manifest.summary
    previous: dict[str, object] = {
        "attempt_date_utc": (NOW - timedelta(days=1)).date().isoformat(),
        "external_attempt_made": False,
        "capture_manifest_enforced": True,
        "capture_manifest_id": summary["manifest_id"],
        "capture_manifest_immutable_path": summary["immutable_manifest_path"],
        "capture_manifest_immutable_sha256": summary["immutable_manifest_sha256"],
        "capture_manifest_pending_calls": summary["pending_calls"],
        "capture_manifest_planned_credits": summary["planned_credits"],
        "capture_manifest_lane_totals": summary["lane_totals"],
        "capture_reconciliation_status": "PENDING",
        "capture_reconciliation_valid": False,
        "capture_reconciliation_complete": False,
        "capture_reconciliation_required_calls": summary["pending_calls"],
        "capture_reconciliation_completed_calls": 0,
        "capture_reconciliation_pending_calls": summary["pending_calls"],
        "capture_reconciliation_blocked_calls": 0,
    }
    path = root / "reports" / "active" / "corrective_wizard_proof_scheduler_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(previous, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return previous


def _passing_capture_reconciler(*, required_calls: int):
    def reconciler(*, root, **_):
        status = root / "reports" / "active" / "capture_reconciliation_test.json"
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"status": status},
            summary={
                "status": "PASS" if required_calls else "NOT_REQUIRED",
                "required_calls": required_calls,
                "completed_calls": required_calls,
                "pending_calls": 0,
                "blocked_calls": 0,
                "blockers": [],
                "reconciliation_id": ("wizardcapturerecon_test" if required_calls else ""),
                "immutable_reconciliation_path": (
                    "data/research/wizard_capture_reconciliations/test.json"
                    if required_calls
                    else ""
                ),
            },
        )

    return reconciler


def _write_copula_cohort(root: Path, *, cohort_id: str = "copulacohort_test") -> dict[str, str]:
    path = root / "data" / "research" / "wizard_copula_behavioral_cohorts" / f"{cohort_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    return {
        "cohort_receipt_id": cohort_id,
        "cohort_receipt_path": str(path.relative_to(root)),
        "cohort_receipt_sha256": sha256(path.read_bytes()).hexdigest(),
    }


def _input_auditor(root: Path, queue_path: Path) -> CommandResult:
    path = root / "reports" / "active" / "input_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return CommandResult(
        paths={"input_audit_summary": path},
        summary={
            "status": "PASS",
            "ready_rows": 5,
            "blocked_rows": 0,
            "retry_safe_rows": 5,
            "changed_after_vendor_4xx_rows": 0,
            "unchanged_vendor_4xx_rows": 0,
        },
    )


def test_preflight_makes_no_external_attempt_and_preserves_no_order_authority(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    calls = []

    def proof_runner(**kwargs):
        calls.append(kwargs)
        return _result(eligible=5, selected=3, completed=0, selected_completed=0)

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=False,
        queue_builder=_queue_builder,
        proof_runner=proof_runner,
        input_auditor=_input_auditor,
        checkpoint_refresher=None,
    )

    assert result.summary["status"] == "PLANNED"
    assert result.summary["api_key_check_performed"] is False
    assert result.summary["api_key_present"] is False
    assert result.summary["api_key_source"] == "not_checked_non_execute"
    assert result.summary["external_attempt_made"] is False
    assert result.summary["order_submission_included"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    assert result.summary["remaining_vendor_responses"] == 5
    assert result.summary["configured_proof_request_capacity"] == 30
    assert result.summary["remaining_custom_series_credits"] == 10
    assert result.summary["next_cohort_capacity_ready"] is True
    assert result.summary["next_cohort_readiness"] == ("CAPACITY_READY_RUNTIME_PREFLIGHT_REQUIRED")
    assert result.summary["next_utc_reset_at"] == ("2026-08-11T00:00:00+00:00")
    assert result.summary["next_external_attempt_eligible_at"] == ("2026-08-10T00:05:00+00:00")
    assert result.summary["next_cohort_queue_reordering_applied"] is False
    assert calls[0]["execute"] is False


def test_ou_v3_holdout_is_plan_only_during_scheduler_preflight(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    contract = tmp_path / "config" / "wizard_ou_comparator_v3_holdout.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    calls = []

    def ou_runner(**kwargs):
        calls.append(kwargs)
        path = tmp_path / "reports" / "active" / "ou_v3_status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"capture_status": path},
            summary={
                "status": "PLANNED",
                "evaluation_status": "WAITING_VENDOR_RESPONSES",
                "missing_cells_before": 4,
                "required_responses": 4,
                "responses_available": 0,
                "calls_made": 0,
                "responses_captured": 0,
                "credits_attempted": 0,
                "credits_completed": 0,
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=False,
        queue_builder=_queue_builder,
        proof_runner=lambda **_: _result(eligible=5, selected=3, completed=0, selected_completed=0),
        input_auditor=_input_auditor,
        ou_v3_holdout_runner=ou_runner,
        checkpoint_refresher=None,
    )

    assert len(calls) == 1
    assert calls[0]["execute"] is False
    assert result.summary["ou_v3_holdout_status"] == "PLANNED"
    assert result.summary["ou_v3_attempted_credits"] == 0
    assert result.summary["external_attempt_made_this_cycle"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_ou_v3_governance_lane_never_auto_applies_activation(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    contract = tmp_path / "config" / "wizard_ou_comparator_v3_holdout.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    activation_calls = []

    def artifact(name: str, summary: dict[str, object]) -> CommandResult:
        path = tmp_path / "reports" / "active" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(paths={name: path}, summary=summary)

    def activation_planner(**kwargs):
        activation_calls.append(kwargs)
        return artifact(
            "activation_status",
            {
                "status": "READY_REQUIRES_EXPLICIT_APPLY",
                "comparator_generation": 3,
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=False,
        queue_builder=_queue_builder,
        proof_runner=lambda **_: _result(eligible=5, selected=3, completed=0, selected_completed=0),
        input_auditor=_input_auditor,
        ou_v3_holdout_runner=lambda **_: artifact(
            "capture_status",
            {
                "status": "COMPLETE",
                "evaluation_status": "PASS",
                "missing_cells_before": 0,
                "required_responses": 4,
                "responses_available": 4,
                "calls_made": 0,
                "responses_captured": 0,
                "credits_attempted": 0,
                "credits_completed": 0,
            },
        ),
        ou_v3_supersession_gate_builder=lambda **_: artifact(
            "supersession_gate",
            {"status": "READY_FOR_REVIEWED_SUPERSESSION"},
        ),
        ou_v3_review_packet_builder=lambda **_: artifact(
            "review_packet",
            {
                "status": "READY_FOR_EXPLICIT_REVIEW",
                "review_packet_id": "ouv3review_test",
            },
        ),
        ou_v3_activation_planner=activation_planner,
        ou_v3_proof_refresher=lambda **_: artifact(
            "refresh_status",
            {"status": "NOT_ACTIVE", "refreshed_ou_rows": 0},
        ),
        checkpoint_refresher=None,
    )

    assert len(activation_calls) == 1
    assert activation_calls[0]["apply"] is False
    assert result.summary["ou_v3_supersession_gate_status"] == ("READY_FOR_REVIEWED_SUPERSESSION")
    assert result.summary["ou_v3_review_packet_status"] == ("READY_FOR_EXPLICIT_REVIEW")
    assert result.summary["ou_v3_activation_status"] == ("READY_REQUIRES_EXPLICIT_APPLY")
    assert result.summary["ou_v3_comparator_generation"] == 1
    assert result.summary["ou_v3_proof_refresh_status"] == "NOT_ACTIVE"
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_ou_v4_governance_lane_never_auto_applies_activation(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    contract = tmp_path / "config" / "wizard_ou_comparator_v4_holdout.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    activation_calls = []

    def artifact(name: str, summary: dict[str, object]) -> CommandResult:
        path = tmp_path / "reports" / "active" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(paths={name: path}, summary=summary)

    def activation_planner(**kwargs):
        activation_calls.append(kwargs)
        return artifact(
            "activation_status",
            {
                "status": "READY_REQUIRES_EXPLICIT_APPLY",
                "comparator_generation": 4,
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=False,
        queue_builder=_queue_builder,
        proof_runner=lambda **_: _result(
            eligible=5,
            selected=3,
            completed=0,
            selected_completed=0,
        ),
        input_auditor=_input_auditor,
        ou_v4_holdout_runner=lambda **_: artifact(
            "capture_status",
            {
                "status": "COMPLETE",
                "evaluation_status": "PASS",
                "missing_cells_before": 0,
                "required_responses": 8,
                "responses_available": 8,
                "calls_made": 0,
                "responses_captured": 0,
                "credits_attempted": 0,
                "credits_completed": 0,
            },
        ),
        ou_v4_supersession_gate_builder=lambda **_: artifact(
            "supersession_gate",
            {"status": "READY_FOR_REVIEWED_SUPERSESSION"},
        ),
        ou_v4_review_packet_builder=lambda **_: artifact(
            "review_packet",
            {
                "status": "READY_FOR_EXPLICIT_REVIEW",
                "review_packet_id": "ouv4review_test",
            },
        ),
        ou_v4_supreme_reviewer=lambda **_: artifact(
            "status",
            {
                "status": "PASS_ADVISORY_ONLY",
                "recommendation": ("RECOMMEND_EXPLICIT_RESEARCH_COMPARATOR_ACTIVATION"),
                "activation_applied_by_supreme_team": False,
            },
        ),
        ou_v4_activation_planner=activation_planner,
        ou_v4_proof_refresher=lambda **_: artifact(
            "refresh_status",
            {"status": "NOT_ACTIVE", "refreshed_ou_rows": 0},
        ),
        checkpoint_refresher=None,
    )

    assert len(activation_calls) == 1
    assert activation_calls[0]["apply"] is False
    assert result.summary["ou_v4_supersession_gate_status"] == ("READY_FOR_REVIEWED_SUPERSESSION")
    assert result.summary["ou_v4_review_packet_status"] == ("READY_FOR_EXPLICIT_REVIEW")
    assert result.summary["ou_v4_supreme_review_status"] == "PASS_ADVISORY_ONLY"
    assert result.summary["ou_v4_activation_status"] == ("READY_REQUIRES_EXPLICIT_APPLY")
    assert result.summary["ou_v4_comparator_generation"] == 1
    assert result.summary["ou_v4_proof_refresh_status"] == "NOT_ACTIVE"
    assert result.summary["ou_v4_activation_automatic"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_ou_v5_governance_lane_never_auto_applies_activation(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    contract = tmp_path / "config" / "wizard_ou_comparator_v5_holdout.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    activation_calls = []

    def artifact(name: str, summary: dict[str, object]) -> CommandResult:
        path = tmp_path / "reports" / "active" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(paths={name: path}, summary=summary)

    def activation_planner(**kwargs):
        activation_calls.append(kwargs)
        return artifact(
            "activation_status",
            {
                "status": "READY_REQUIRES_EXPLICIT_APPLY",
                "comparator_generation": 5,
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=False,
        queue_builder=_queue_builder,
        proof_runner=lambda **_: _result(
            eligible=5,
            selected=3,
            completed=0,
            selected_completed=0,
        ),
        input_auditor=_input_auditor,
        ou_v5_holdout_runner=lambda **_: artifact(
            "capture_status",
            {
                "status": "COMPLETE",
                "evaluation_status": "PASS",
                "missing_cells_before": 0,
                "required_responses": 8,
                "responses_available": 8,
                "calls_made": 0,
                "responses_captured": 0,
                "credits_attempted": 0,
                "credits_completed": 0,
            },
        ),
        ou_v5_supersession_gate_builder=lambda **_: artifact(
            "supersession_gate",
            {"status": "READY_FOR_REVIEWED_SUPERSESSION"},
        ),
        ou_v5_review_packet_builder=lambda **_: artifact(
            "review_packet",
            {
                "status": "READY_FOR_EXPLICIT_REVIEW",
                "review_packet_id": "ouv5review_test",
            },
        ),
        ou_v5_supreme_reviewer=lambda **_: artifact(
            "status",
            {
                "status": "PASS_ADVISORY_ONLY",
                "recommendation": "RECOMMEND_EXPLICIT_RESEARCH_COMPARATOR_ACTIVATION",
                "activation_applied_by_supreme_team": False,
            },
        ),
        ou_v5_activation_planner=activation_planner,
        ou_v5_proof_refresher=lambda **_: artifact(
            "refresh_status",
            {"status": "NOT_ACTIVE", "refreshed_ou_rows": 0},
        ),
        checkpoint_refresher=None,
    )

    assert len(activation_calls) == 1
    assert activation_calls[0]["apply"] is False
    assert result.summary["ou_v5_supersession_gate_status"] == ("READY_FOR_REVIEWED_SUPERSESSION")
    assert result.summary["ou_v5_review_packet_status"] == "READY_FOR_EXPLICIT_REVIEW"
    assert result.summary["ou_v5_supreme_review_status"] == "PASS_ADVISORY_ONLY"
    assert result.summary["ou_v5_activation_status"] == "READY_REQUIRES_EXPLICIT_APPLY"
    assert result.summary["ou_v5_comparator_generation"] == 1
    assert result.summary["ou_v5_proof_refresh_status"] == "NOT_ACTIVE"
    assert result.summary["ou_v5_failure_attribution_status"] == "NOT_REQUIRED_HOLDOUT_PASS"
    assert result.summary["ou_v5_activation_automatic"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_ou_v5_failed_holdout_builds_attribution_without_successor_registration(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    contract = tmp_path / "config" / "wizard_ou_comparator_v5_holdout.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    attribution_calls = []

    def artifact(name: str, summary: dict[str, object]) -> CommandResult:
        path = tmp_path / "reports" / "active" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(paths={name: path}, summary=summary)

    def attributor(**kwargs):
        attribution_calls.append(kwargs)
        return artifact(
            "status",
            {
                "status": "PASS_FAILURE_ATTRIBUTION_COMPLETE",
                "attribution_id": "ouv5failure_test",
                "successor_design_automatic": False,
                "successor_registration_automatic": False,
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=False,
        queue_builder=_queue_builder,
        proof_runner=lambda **_: _result(
            eligible=5,
            selected=3,
            completed=0,
            selected_completed=0,
        ),
        input_auditor=_input_auditor,
        ou_v5_holdout_runner=lambda **_: artifact(
            "capture_status",
            {
                "status": "COMPLETE",
                "evaluation_status": "FAIL",
                "missing_cells_before": 0,
                "required_responses": 8,
                "responses_available": 8,
                "calls_made": 0,
                "responses_captured": 0,
                "credits_attempted": 0,
                "credits_completed": 0,
            },
        ),
        ou_v5_failure_attributor=attributor,
        ou_v5_supersession_gate_builder=None,
        ou_v5_review_packet_builder=None,
        ou_v5_supreme_reviewer=None,
        ou_v5_activation_planner=None,
        ou_v5_proof_refresher=None,
        checkpoint_refresher=None,
    )

    assert len(attribution_calls) == 1
    assert result.summary["ou_v5_failure_attribution_status"] == (
        "PASS_FAILURE_ATTRIBUTION_COMPLETE"
    )
    assert result.summary["ou_v5_failure_attribution_id"] == "ouv5failure_test"
    assert result.summary["ou_v5_activation_automatic"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_fresh_scheduler_window_reconciles_ou_v3_capture_credits(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    contract = tmp_path / "config" / "wizard_ou_comparator_v3_holdout.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    ou_calls = []
    reconciliations = []

    def ou_runner(**kwargs):
        ou_calls.append(kwargs)
        path = tmp_path / "reports" / "active" / "ou_v3_status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"capture_status": path},
            summary={
                "status": "COMPLETE",
                "evaluation_status": "PASS",
                "missing_cells_before": 4,
                "required_responses": 4,
                "responses_available": 4,
                "calls_made": 4,
                "responses_captured": 4,
                "credits_used_before": 20,
                "credits_attempted": 8,
                "credits_completed": 8,
            },
        )

    def credit_reconciler(**kwargs):
        reconciliations.append(kwargs)
        path = tmp_path / "data" / "research" / "reconciliation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"reconciliation": path},
            summary={
                "status": "PASS_RECONCILED",
                "blocker": "",
                "reconciliation_id": "ou-v3-reconciliation",
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=1),
        proof_runner=lambda **_: _result(
            eligible=1,
            selected=1,
            completed=1,
            selected_completed=1,
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={
                "status": "PASS",
                "ready_rows": 1,
                "blocked_rows": 0,
                "retry_safe_rows": 1,
                "unchanged_vendor_4xx_rows": 0,
            },
        ),
        ou_v3_holdout_runner=ou_runner,
        credit_reconciler=credit_reconciler,
        parity_refresher=lambda **_: CommandResult(paths={}, summary={"status": "PASS"}),
        checkpoint_refresher=None,
        registered_rerun_runner=None,
        credits_fetcher=lambda **_: {"credits_used": 30},
    )

    assert len(ou_calls) == 1
    assert ou_calls[0]["execute"] is True
    assert result.summary["ou_v3_calls_made"] == 4
    assert result.summary["ou_v3_responses_captured_this_cycle"] == 4
    assert result.summary["ou_v3_attempted_credits"] == 8
    assert result.summary["ou_v3_completed_credits"] == 8
    assert result.summary["proof_lane_attempted_credits"] == 10
    assert result.summary["proof_lane_completed_credits"] == 10
    assert result.summary["proof_lane_credit_ceiling"] == 124
    assert reconciliations[0]["external_requests"] == 5
    assert reconciliations[0]["attempted_credits"] == 10
    assert reconciliations[0]["completed_credits"] == 10
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_fresh_scheduler_window_governs_ou_v4_capture_and_credits(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    contract = tmp_path / "config" / "wizard_ou_comparator_v4_holdout.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    calls = []
    credit_reconciliations = []

    def ou_v4_runner(**kwargs):
        calls.append(kwargs)
        path = tmp_path / "reports" / "active" / "ou_v4_status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"capture_status": path},
            summary={
                "status": "COMPLETE",
                "evaluation_status": "PASS",
                "missing_cells_before": 8,
                "required_responses": 8,
                "responses_available": 8,
                "calls_made": 8,
                "responses_captured": 8,
                "credits_used_before": 20,
                "credits_attempted": 16,
                "credits_completed": 16,
            },
        )

    def credit_reconciler(**kwargs):
        credit_reconciliations.append(kwargs)
        path = tmp_path / "data" / "research" / "v4_reconciliation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"reconciliation": path},
            summary={
                "status": "PASS_RECONCILED",
                "blocker": "",
                "reconciliation_id": "ou-v4-reconciliation",
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=1),
        proof_runner=lambda **_: _result(
            eligible=1,
            selected=1,
            completed=1,
            selected_completed=1,
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={
                "status": "PASS",
                "ready_rows": 1,
                "blocked_rows": 0,
                "retry_safe_rows": 1,
                "unchanged_vendor_4xx_rows": 0,
            },
        ),
        capture_manifest_builder=_capture_manifest_builder(
            exact_calls=1,
            ou_v4_calls=8,
        ),
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=9),
        ou_v3_holdout_runner=None,
        ou_v4_holdout_runner=ou_v4_runner,
        credit_reconciler=credit_reconciler,
        parity_refresher=lambda **_: CommandResult(paths={}, summary={"status": "PASS"}),
        checkpoint_refresher=None,
        registered_rerun_runner=None,
        credits_fetcher=lambda **_: {"credits_used": 38},
    )

    assert len(calls) == 1
    assert calls[0]["execute"] is True
    assert result.summary["ou_v4_calls_made"] == 8
    assert result.summary["ou_v4_responses_captured_this_cycle"] == 8
    assert result.summary["ou_v4_attempted_credits"] == 16
    assert result.summary["ou_v4_completed_credits"] == 16
    assert result.summary["proof_lane_attempted_credits"] == 18
    assert result.summary["proof_lane_completed_credits"] == 18
    assert result.summary["credit_reservation_planned_credits"] == 18
    assert result.summary["capture_manifest_accounting_valid"] is True
    assert credit_reconciliations[0]["external_requests"] == 9
    assert credit_reconciliations[0]["attempted_credits"] == 18
    assert credit_reconciliations[0]["completed_credits"] == 18
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_fresh_scheduler_window_governs_ou_v5_capture_and_credits(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    contract = tmp_path / "config" / "wizard_ou_comparator_v5_holdout.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    calls = []
    credit_reconciliations = []

    def ou_v5_runner(**kwargs):
        calls.append(kwargs)
        path = tmp_path / "reports" / "active" / "ou_v5_status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"capture_status": path},
            summary={
                "status": "COMPLETE",
                "evaluation_status": "PASS",
                "missing_cells_before": 8,
                "required_responses": 8,
                "responses_available": 8,
                "calls_made": 8,
                "responses_captured": 8,
                "credits_used_before": 20,
                "credits_attempted": 16,
                "credits_completed": 16,
            },
        )

    def credit_reconciler(**kwargs):
        credit_reconciliations.append(kwargs)
        path = tmp_path / "data" / "research" / "v5_reconciliation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"reconciliation": path},
            summary={
                "status": "PASS_RECONCILED",
                "blocker": "",
                "reconciliation_id": "ou-v5-reconciliation",
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=1),
        proof_runner=lambda **_: _result(
            eligible=1,
            selected=1,
            completed=1,
            selected_completed=1,
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={
                "status": "PASS",
                "ready_rows": 1,
                "blocked_rows": 0,
                "retry_safe_rows": 1,
                "unchanged_vendor_4xx_rows": 0,
            },
        ),
        capture_manifest_builder=_capture_manifest_builder(
            exact_calls=1,
            ou_v5_calls=8,
        ),
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=9),
        ou_v3_holdout_runner=None,
        ou_v4_holdout_runner=None,
        ou_v5_holdout_runner=ou_v5_runner,
        credit_reconciler=credit_reconciler,
        parity_refresher=lambda **_: CommandResult(paths={}, summary={"status": "PASS"}),
        checkpoint_refresher=None,
        registered_rerun_runner=None,
        credits_fetcher=lambda **_: {"credits_used": 38},
    )

    assert len(calls) == 1
    assert calls[0]["execute"] is True
    assert result.summary["ou_v5_external_authorized"] is True
    assert result.summary["ou_v5_calls_made"] == 8
    assert result.summary["ou_v5_responses_captured_this_cycle"] == 8
    assert result.summary["ou_v5_attempted_credits"] == 16
    assert result.summary["ou_v5_completed_credits"] == 16
    assert result.summary["proof_lane_attempted_credits"] == 18
    assert result.summary["proof_lane_completed_credits"] == 18
    assert result.summary["credit_reservation_planned_credits"] == 18
    assert result.summary["capture_manifest_accounting_valid"] is True
    assert credit_reconciliations[0]["external_requests"] == 9
    assert credit_reconciliations[0]["attempted_credits"] == 18
    assert credit_reconciliations[0]["completed_credits"] == 18
    assert result.summary["ou_v5_activation_automatic"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_fresh_scheduler_window_governs_final_ou_v6_capture_without_activation(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    contract = tmp_path / "config" / "wizard_ou_comparator_v6_holdout.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    calls = []

    def ou_v6_runner(**kwargs):
        calls.append(kwargs)
        path = tmp_path / "reports" / "active" / "ou_v6_status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"capture_status": path},
            summary={
                "status": "COMPLETE",
                "evaluation_status": "PASS",
                "missing_cells_before": 8,
                "required_responses": 8,
                "responses_available": 8,
                "calls_made": 8,
                "responses_captured": 8,
                "credits_used_before": 20,
                "credits_attempted": 16,
                "credits_completed": 16,
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=1),
        proof_runner=lambda **_: _result(
            eligible=1,
            selected=1,
            completed=1,
            selected_completed=1,
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={
                "status": "PASS",
                "ready_rows": 1,
                "blocked_rows": 0,
                "retry_safe_rows": 1,
                "unchanged_vendor_4xx_rows": 0,
            },
        ),
        capture_manifest_builder=_capture_manifest_builder(
            exact_calls=1,
            ou_v6_calls=8,
        ),
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=9),
        ou_v3_holdout_runner=None,
        ou_v4_holdout_runner=None,
        ou_v5_holdout_runner=None,
        ou_v6_holdout_runner=ou_v6_runner,
        parity_refresher=lambda **_: CommandResult(paths={}, summary={"status": "PASS"}),
        checkpoint_refresher=None,
        registered_rerun_runner=None,
        credits_fetcher=lambda **_: {"credits_used": 38},
    )

    assert len(calls) == 1
    assert calls[0]["execute"] is True
    assert result.summary["ou_v6_external_authorized"] is True
    assert result.summary["ou_v6_calls_made"] == 8
    assert result.summary["ou_v6_responses_captured_this_cycle"] == 8
    assert result.summary["ou_v6_attempted_credits"] == 16
    assert result.summary["ou_v6_completed_credits"] == 16
    assert result.summary["proof_lane_attempted_credits"] == 18
    assert result.summary["proof_lane_completed_credits"] == 18
    assert result.summary["ou_v6_terminal_failure"] is False
    assert result.summary["ou_v6_supersession_gate_status"] == "BLOCKED"
    assert result.summary["ou_v6_review_packet_status"] == "WAITING_FOR_HOLDOUT"
    assert result.summary["ou_v6_supreme_review_status"] == "BLOCKED"
    assert result.summary["ou_v6_activation_status"] == "BLOCKED"
    assert result.summary["ou_v6_proof_refresh_status"] == "NOT_ACTIVE"
    assert result.summary["ou_v6_activation_automatic"] is False
    assert result.summary["ou_v6_comparator_generation"] == 1
    assert result.paths["ou_v6_capture_status"].is_file()
    assert result.paths["ou_v6_supersession_gate"].is_file()
    assert result.paths["ou_v6_review_packet"].is_file()
    assert result.paths["ou_v6_supreme_review"].is_file()
    assert result.paths["ou_v6_activation_status"].is_file()
    assert result.paths["ou_v6_proof_refresh_status"].is_file()
    assert result.summary["candidate_promotion_authority"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_terminal_ou_v6_failure_cannot_invoke_registered_rerun(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    monkeypatch.setattr(
        corrective_wizard_proof_scheduler,
        "validate_ou_v6_stage3_evidence",
        lambda **_: {"status": "PASS", "blockers": []},
    )
    _write_secure_env(tmp_path)
    contract = tmp_path / "config" / "wizard_ou_comparator_v6_holdout.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    registered_calls = []

    def ou_v6_runner(**_):
        path = tmp_path / "reports" / "active" / "ou_v6_status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"capture_status": path},
            summary={
                "status": "COMPLETE",
                "evaluation_status": "FAIL",
                "missing_cells_before": 8,
                "required_responses": 8,
                "responses_available": 8,
                "calls_made": 8,
                "responses_captured": 8,
                "credits_used_before": 20,
                "credits_attempted": 16,
                "credits_completed": 16,
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=1),
        proof_runner=lambda **_: _result(
            eligible=1,
            selected=1,
            completed=1,
            selected_completed=1,
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={
                "status": "PASS",
                "ready_rows": 1,
                "blocked_rows": 0,
                "retry_safe_rows": 1,
                "unchanged_vendor_4xx_rows": 0,
            },
        ),
        capture_manifest_builder=_capture_manifest_builder(
            exact_calls=1,
            ou_v6_calls=8,
        ),
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=9),
        ou_v3_holdout_runner=None,
        ou_v4_holdout_runner=None,
        ou_v5_holdout_runner=None,
        ou_v6_holdout_runner=ou_v6_runner,
        parity_refresher=lambda **_: CommandResult(paths={}, summary={"status": "PASS"}),
        checkpoint_refresher=None,
        registered_rerun_runner=lambda **kwargs: registered_calls.append(kwargs),
        credits_fetcher=lambda **_: {"credits_used": 38},
    )

    assert result.summary["status"] == "BLOCKED_OU_V6_TERMINAL_FAILURE"
    assert result.summary["ou_v6_terminal_failure"] is True
    assert result.summary["ou_v6_stage3_validation_status"] == "PASS"
    assert "ou_v6_terminal_failure_exact_local_ou_parity_rejected" in result.summary["blockers"]
    assert registered_calls == []
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_frozen_manifest_blocks_unlisted_external_lanes_before_runner_call(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    contract = tmp_path / "config" / "wizard_ou_comparator_v4_holdout.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    exact_calls = []
    ou_v4_calls = []

    def exact_runner(**kwargs):
        exact_calls.append(kwargs)
        raise AssertionError("unlisted exact-mode lane must not run")

    def ou_v4_runner(**kwargs):
        ou_v4_calls.append(kwargs)
        path = tmp_path / "reports" / "active" / "ou_v4_only_status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"capture_status": path},
            summary={
                "status": "COMPLETE",
                "evaluation_status": "PASS",
                "missing_cells_before": 8,
                "required_responses": 8,
                "responses_available": 8,
                "calls_made": 8,
                "responses_captured": 8,
                "credits_used_before": 0,
                "credits_attempted": 16,
                "credits_completed": 16,
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=10),
        proof_runner=exact_runner,
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={
                "status": "PASS",
                "ready_rows": 10,
                "blocked_rows": 0,
                "retry_safe_rows": 10,
                "unchanged_vendor_4xx_rows": 0,
            },
        ),
        capture_manifest_builder=_capture_manifest_builder(ou_v4_calls=8),
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=8),
        ou_v3_holdout_runner=None,
        ou_v4_holdout_runner=ou_v4_runner,
        copula_behavioral_runner=None,
        credit_reconciler=lambda **_: CommandResult(
            paths={},
            summary={
                "status": "PASS_RECONCILED",
                "blocker": "",
                "reconciliation_id": "v4-only-reconciliation",
            },
        ),
        parity_refresher=lambda **_: CommandResult(paths={}, summary={"status": "PASS"}),
        checkpoint_refresher=None,
        registered_rerun_runner=None,
        credits_fetcher=lambda **_: {"credits_used": 16},
    )

    assert exact_calls == []
    assert len(ou_v4_calls) == 1
    assert ou_v4_calls[0]["execute"] is True
    assert result.summary["exact_mode_external_authorized"] is False
    assert result.summary["ou_v4_external_authorized"] is True
    assert result.summary["exact_mode_requests_attempted"] == 0
    assert result.summary["ou_v4_calls_made"] == 8
    assert result.summary["credit_reservation_planned_credits"] == 16
    assert result.summary["proof_lane_attempted_credits"] == 16
    assert result.summary["capture_manifest_accounting_valid"] is True


def test_frozen_manifest_caps_exact_mode_batches_before_additional_calls(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    requested_batch_sizes = []

    def exact_runner(**kwargs):
        requested_batch_sizes.append(kwargs["max_pairs"])
        selected = int(kwargs["max_pairs"])
        return _result(
            eligible=10,
            selected=selected,
            completed=selected,
            selected_completed=selected,
            responses_captured=selected,
            selected_responses_captured=selected,
            external_proof_requests=selected,
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=10),
        proof_runner=exact_runner,
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={
                "status": "PASS",
                "ready_rows": 10,
                "blocked_rows": 0,
                "retry_safe_rows": 10,
                "unchanged_vendor_4xx_rows": 0,
            },
        ),
        capture_manifest_builder=_capture_manifest_builder(exact_calls=2),
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=2),
        ou_v3_holdout_runner=None,
        ou_v4_holdout_runner=None,
        copula_behavioral_runner=None,
        parity_refresher=lambda **_: CommandResult(paths={}, summary={"status": "PASS"}),
        checkpoint_refresher=None,
        registered_rerun_runner=None,
    )

    assert requested_batch_sizes == [2]
    assert result.summary["exact_mode_manifest_call_limit"] == 2
    assert result.summary["exact_mode_requests_attempted"] == 2
    assert result.summary["exact_mode_attempted_credits"] == 4
    assert result.summary["credit_reservation_planned_credits"] == 4
    assert result.summary["capture_manifest_accounting_valid"] is True


def test_same_day_budget_expansion_defers_with_verified_prior_credit_evidence(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    reservation = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        planned_credits=76,
        now=NOW,
    )
    reconciliation = reconcile_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        reservation_id=reservation.summary["reservation_id"],
        reconciliation_key=NOW.isoformat(),
        attempted_credits=0,
        completed_credits=0,
        external_requests=0,
        observed_used_before=None,
        now=NOW,
    )
    status_path = tmp_path / "reports" / "active" / "corrective_wizard_proof_scheduler_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "attempt_date_utc": NOW.date().isoformat(),
                "external_attempt_made": True,
                "credit_reconciliation_id": reconciliation.summary["reconciliation_id"],
                "credit_reconciliation_path": str(
                    reconciliation.paths["reconciliation"].relative_to(tmp_path)
                ),
            }
        ),
        encoding="utf-8",
    )
    base_manifest_builder = _capture_manifest_builder(ou_v4_calls=8)

    def deferred_manifest_builder(**kwargs):
        result = base_manifest_builder(**kwargs)
        result.summary.update(
            {
                "capture_state": "DEFERRED_UNTIL_UTC_RESET",
                "capture_eligible_now": False,
                "next_external_attempt_eligible_at": (NOW + timedelta(days=1))
                .replace(hour=0, minute=0)
                .isoformat(),
            }
        )
        return result

    def pending_reconciler(*, root, **_):
        path = root / "reports" / "active" / "pending_reconciliation.json"
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"status": path},
            summary={
                "status": "PENDING",
                "manifest_id": "",
                "required_calls": 8,
                "completed_calls": 0,
                "pending_calls": 8,
                "blocked_calls": 0,
                "blockers": [],
            },
        )

    reservation_calls = []
    proof_calls = []
    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW + timedelta(minutes=10),
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=0),
        proof_runner=lambda **kwargs: proof_calls.append(kwargs),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={
                "status": "PASS",
                "ready_rows": 0,
                "blocked_rows": 0,
                "retry_safe_rows": 0,
                "unchanged_vendor_4xx_rows": 0,
            },
        ),
        capture_manifest_builder=deferred_manifest_builder,
        capture_manifest_reconciler=pending_reconciler,
        credit_reserver=lambda **kwargs: reservation_calls.append(kwargs),
        checkpoint_refresher=None,
        registered_rerun_runner=None,
        copula_behavioral_runner=None,
        ou_v3_holdout_runner=None,
        ou_v4_holdout_runner=None,
    )

    assert result.summary["status"] == "DEFERRED_CAPTURE_MANIFEST_WINDOW"
    assert result.summary["api_key_check_performed"] is False
    assert result.summary["api_key_present"] is False
    assert result.summary["api_key_source"] == "not_checked_before_reservation"
    assert result.summary["credit_reservation_status"] == ("PRIOR_RESERVATION_VERIFIED")
    assert result.summary["credit_reconciliation_status"] == ("PRIOR_RECONCILIATION_VERIFIED")
    assert result.summary["credit_reservation_blocker"] == ""
    assert result.summary["credit_reconciliation_blocker"] == ""
    assert result.summary["credit_reservation_planned_credits"] == 16
    assert result.summary["external_attempt_made_this_cycle"] is False
    assert result.summary["proof_lane_attempted_credits"] == 0
    assert reservation_calls == []
    assert proof_calls == []
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_next_cohort_capacity_fails_closed_when_remaining_exceeds_cap(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=False,
        max_batches=1,
        max_proofs_per_batch=3,
        queue_builder=_queue_builder,
        proof_runner=lambda **_: _result(eligible=5, selected=3, completed=0, selected_completed=0),
        input_auditor=_input_auditor,
        checkpoint_refresher=None,
    )

    assert result.summary["remaining_vendor_responses"] == 5
    assert result.summary["configured_proof_request_capacity"] == 3
    assert result.summary["next_cohort_capacity_ready"] is False
    assert result.summary["next_cohort_readiness"] == ("BLOCKED_CAPACITY_OR_INPUT_CONTRACT")


def test_execute_batches_to_queue_completion_then_refreshes_evidence(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    results = iter(
        [
            _result(eligible=5, selected=3, completed=3, selected_completed=3),
            _result(eligible=5, selected=2, completed=5, selected_completed=2),
        ]
    )
    proof_calls = []
    parity_calls = []
    checkpoint_calls = []

    def proof_runner(**kwargs):
        proof_calls.append(kwargs)
        return next(results)

    def parity_refresher(**kwargs):
        parity_calls.append(kwargs)
        return CommandResult(paths={}, summary={"status": "BLOCKED"})

    def checkpoint_refresher(**kwargs):
        checkpoint_calls.append(kwargs)
        published = json.loads(
            (
                tmp_path / "reports" / "active" / "corrective_wizard_proof_scheduler_status.json"
            ).read_text(encoding="utf-8")
        )
        assert published["queue_eligible"] == 5
        assert published["completed_after"] == 5
        assert published["responses_captured_after"] == 5
        return CommandResult(paths={}, summary={"status": "BLOCKED"})

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=_queue_builder,
        proof_runner=proof_runner,
        input_auditor=_input_auditor,
        parity_refresher=parity_refresher,
        checkpoint_refresher=checkpoint_refresher,
        credits_fetcher=lambda **_: {"credits_used": 30},
    )

    assert result.summary["status"] == "COMPLETE_QUEUE"
    assert result.summary["new_completed_proofs"] == 5
    assert result.summary["proof_request_slots_selected"] == 5
    assert result.summary["reserved_credits"] == 100
    assert len(proof_calls) == 2
    assert len(parity_calls) == 1
    assert len(checkpoint_calls) == 1
    assert all(call["max_pairs"] == 3 for call in proof_calls)
    receipt = result.paths["cycle_receipt"].read_text(encoding="utf-8")
    assert SECRET not in receipt
    receipt_payload = json.loads(receipt)
    assert receipt_payload["api_key_source"] == "authorized_selected_environment"
    assert receipt_payload["final_immutable_receipt_required"] is True
    assert (
        result.paths["immutable_cycle_receipt"].read_bytes()
        == result.paths["cycle_receipt"].read_bytes()
    )
    assert result.paths["immutable_cycle_receipt"].stem == receipt_payload["receipt_id"]


def test_scheduler_records_dynamic_holdout_without_granting_authority(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    contract = tmp_path / "config" / "wizard_dynamic_comparator_v2_holdout.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    status_path = tmp_path / "reports" / "active" / "dynamic_status.json"
    calls = []
    activation_calls = []
    review_packet_calls = []

    def evaluator(**kwargs):
        calls.append(kwargs)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"status": status_path},
            summary={
                "status": "PASS",
                "comparator_supersession_eligible": True,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        )

    def supersession_gate(**kwargs):
        path = tmp_path / "reports" / "active" / "supersession_gate.json"
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"supersession_gate": path},
            summary={
                "status": "READY_FOR_REVIEWED_SUPERSESSION",
                "supersession_applied": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        )

    def activation_planner(**kwargs):
        activation_calls.append(kwargs)
        path = tmp_path / "reports" / "active" / "activation_status.json"
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"activation_status": path},
            summary={
                "status": "READY_REQUIRES_EXPLICIT_APPLY",
                "comparator_generation": 2,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
            },
        )

    def review_packet_builder(**kwargs):
        review_packet_calls.append(kwargs)
        path = tmp_path / "reports" / "active" / "review_packet.json"
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"review_packet": path},
            summary={
                "status": "READY_FOR_EXPLICIT_REVIEW",
                "review_packet_id": "dynamicv2review_test",
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
            },
        )

    def proof_refresher(**kwargs):
        path = tmp_path / "reports" / "active" / "refresh_status.json"
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"refresh_status": path},
            summary={
                "status": "NOT_ACTIVE",
                "refreshed_dynamic_rows": 0,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=False,
        queue_builder=_queue_builder,
        proof_runner=lambda **_: _result(eligible=5, selected=0, completed=0, selected_completed=0),
        input_auditor=_input_auditor,
        dynamic_holdout_evaluator=evaluator,
        dynamic_supersession_gate_builder=supersession_gate,
        dynamic_activation_planner=activation_planner,
        dynamic_review_packet_builder=review_packet_builder,
        dynamic_proof_refresher=proof_refresher,
        checkpoint_refresher=None,
    )

    assert len(calls) == 1
    assert len(activation_calls) == 1
    assert len(review_packet_calls) == 1
    assert activation_calls[0]["apply"] is False
    assert result.summary["dynamic_v2_holdout_status"] == "PASS"
    assert result.summary["dynamic_v2_supersession_eligible"] is True
    assert result.summary["dynamic_v2_supersession_automatic"] is False
    assert result.summary["dynamic_v2_activation_automatic"] is False
    assert result.summary["dynamic_v2_comparator_generation"] == 1
    assert result.summary["dynamic_v2_activation_status"] == "READY_REQUIRES_EXPLICIT_APPLY"
    assert result.summary["dynamic_v2_proof_refresh_status"] == "NOT_ACTIVE"
    assert result.summary["dynamic_v2_review_packet_status"] == "READY_FOR_EXPLICIT_REVIEW"
    assert result.summary["dynamic_v2_review_packet_id"] == "dynamicv2review_test"
    assert result.summary["dynamic_v2_review_required"] is True
    assert (
        result.summary["dynamic_v2_supersession_gate_status"] == "READY_FOR_REVIEWED_SUPERSESSION"
    )
    assert result.summary["candidate_promotion_authority"] is False
    assert result.summary["testnet_order_authority"] is False


def test_completed_parity_queue_hands_off_to_registered_research_rerun(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    rerun_calls = []
    checkpoint_calls = []
    state = {"rerun_complete": False}
    execution_receipt = tmp_path / "data" / "research" / "registered.json"
    execution_receipt.parent.mkdir(parents=True)
    execution_receipt.write_text("{}", encoding="utf-8")

    def registered_rerun_runner(**kwargs):
        latest = json.loads(
            (
                tmp_path / "reports" / "active" / "corrective_wizard_proof_scheduler_status.json"
            ).read_text(encoding="utf-8")
        )
        active_cycle = tmp_path / latest["receipt_path"]
        immutable_cycle = (
            tmp_path
            / "data"
            / "research"
            / "wizard_proof_scheduler_receipts"
            / f"{latest['receipt_id']}.json"
        )
        assert immutable_cycle.is_file()
        assert immutable_cycle.read_bytes() == active_cycle.read_bytes()
        rerun_calls.append(kwargs)
        state["rerun_complete"] = True
        return CommandResult(
            paths={"execution_receipt": execution_receipt},
            summary={
                "status": "PASS_REGISTERED_RERUN_ACCOUNTED",
                "learning_handoff_status": "REJECTED_RESEARCH_LEARNING_GATES",
                "stage5_research_gate_pass": False,
                "learning_handoff_blocker": "model_or_rl_oos_gate_failed",
            },
        )

    def checkpoint_refresher(**kwargs):
        checkpoint_calls.append(kwargs)
        return CommandResult(
            paths={},
            summary={
                "operational_acceptance_status": (
                    "POST_RERUN_REFRESHED" if state["rerun_complete"] else "PRE_RERUN_REFRESHED"
                )
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=1),
        proof_runner=lambda **_: _result(
            eligible=1,
            selected=1,
            completed=1,
            selected_completed=1,
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={"status": "PASS", "ready_rows": 1, "blocked_rows": 0},
        ),
        capture_manifest_builder=_capture_manifest_builder(exact_calls=1),
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=1),
        parity_refresher=lambda **_: CommandResult(paths={}, summary={"status": "PASS"}),
        checkpoint_refresher=checkpoint_refresher,
        registered_rerun_runner=registered_rerun_runner,
        credits_fetcher=lambda **_: {"credits_used": 22},
    )

    assert result.summary["status"] == "COMPLETE_QUEUE"
    assert result.summary["registered_rerun_status"] == ("PASS_REGISTERED_RERUN_ACCOUNTED")
    assert result.summary["registered_rerun_receipt_path"] == ("data/research/registered.json")
    assert result.summary["registered_learning_handoff_status"] == (
        "REJECTED_RESEARCH_LEARNING_GATES"
    )
    assert result.summary["registered_stage5_research_gate_pass"] is False
    assert result.summary["registered_learning_handoff_blocker"] == ("model_or_rl_oos_gate_failed")
    assert len(rerun_calls) == 1
    assert rerun_calls[0]["execute"] is True
    assert len(checkpoint_calls) == 2
    assert result.summary["checkpoint_pre_rerun_status"] == "PRE_RERUN_REFRESHED"
    assert result.summary["checkpoint_post_rerun_status"] == "POST_RERUN_REFRESHED"
    assert result.summary["checkpoint_refresh_status"] == "POST_RERUN_REFRESHED"
    assert result.summary["checkpoint_refresh_phase"] == "POST_REGISTERED_RERUN"


def test_immutable_publication_failure_is_durable_and_blocks_stage4(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    rerun_calls = []

    def publication_failure(**_):
        raise OSError("injected immutable publication failure")

    monkeypatch.setattr(
        corrective_wizard_proof_scheduler,
        "_write_final_immutable_scheduler_receipt",
        publication_failure,
    )

    with pytest.raises(OSError, match="injected immutable publication failure"):
        run_corrective_wizard_proof_cycle(
            root=tmp_path,
            now=NOW,
            execute=True,
            queue_builder=lambda root: _queue_builder(root, eligible=1),
            proof_runner=lambda **_: _result(
                eligible=1,
                selected=1,
                completed=1,
                selected_completed=1,
            ),
            input_auditor=lambda root, queue_path: CommandResult(
                paths={"input_audit_summary": queue_path},
                summary={"status": "PASS", "ready_rows": 1, "blocked_rows": 0},
            ),
            capture_manifest_builder=_capture_manifest_builder(exact_calls=1),
            capture_manifest_reconciler=_passing_capture_reconciler(required_calls=1),
            parity_refresher=lambda **_: CommandResult(
                paths={},
                summary={"status": "PASS"},
            ),
            checkpoint_refresher=None,
            registered_rerun_runner=lambda **kwargs: rerun_calls.append(kwargs),
            credits_fetcher=lambda **_: {"credits_used": 22},
        )

    latest_path = tmp_path / "reports" / "active" / "corrective_wizard_proof_scheduler_status.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    cycle = json.loads((tmp_path / latest["receipt_path"]).read_text(encoding="utf-8"))

    assert rerun_calls == []
    assert latest["registered_rerun_status"] == ("BLOCKED_IMMUTABLE_STAGE3_RECEIPT")
    assert cycle["registered_rerun_status"] == ("BLOCKED_IMMUTABLE_STAGE3_RECEIPT")
    assert any(
        blocker.startswith("pre_registered_rerun_immutable_receipt_failed:OSError:")
        for blocker in latest["blockers"]
    )
    assert latest["candidate_promotion_authority"] is False
    assert latest["testnet_order_authority"] is False
    assert latest["live_trading_authorized"] is False


def test_transient_immutable_publication_failure_stays_blocked_for_cycle(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    rerun_calls = []
    attempts = 0
    original = corrective_wizard_proof_scheduler._write_final_immutable_scheduler_receipt

    def fail_once(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected transient publication failure")
        return original(**kwargs)

    monkeypatch.setattr(
        corrective_wizard_proof_scheduler,
        "_write_final_immutable_scheduler_receipt",
        fail_once,
    )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=1),
        proof_runner=lambda **_: _result(
            eligible=1,
            selected=1,
            completed=1,
            selected_completed=1,
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={"status": "PASS", "ready_rows": 1, "blocked_rows": 0},
        ),
        capture_manifest_builder=_capture_manifest_builder(exact_calls=1),
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=1),
        parity_refresher=lambda **_: CommandResult(
            paths={},
            summary={"status": "PASS"},
        ),
        checkpoint_refresher=None,
        registered_rerun_runner=lambda **kwargs: rerun_calls.append(kwargs),
        credits_fetcher=lambda **_: {"credits_used": 22},
    )

    assert attempts == 2
    assert rerun_calls == []
    assert result.summary["registered_rerun_status"] == ("BLOCKED_IMMUTABLE_STAGE3_RECEIPT")
    assert any(
        blocker.startswith("pre_registered_rerun_immutable_receipt_failed:OSError:")
        for blocker in result.summary["blockers"]
    )
    assert result.paths["immutable_cycle_receipt"].is_file()
    assert result.summary["candidate_promotion_authority"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_internal_continuation_runs_registered_generation_without_proof_requests(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    (active / "corrective_wizard_proof_scheduler_status.json").write_text(
        json.dumps(
            {
                "attempt_date_utc": "2026-08-10",
                "external_attempt_made": True,
                "credit_reconciliation_status": "PASS_RECONCILED",
                "credit_reconciliation_id": "reconcile-1",
                "credit_reconciliation_path": "data/research/reconcile-1.json",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_wizard_proof_scheduler."
        "count_completed_exact_mode_proofs",
        lambda **_: 1,
    )
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_wizard_proof_scheduler."
        "count_captured_exact_mode_responses",
        lambda **_: 1,
    )
    rerun_calls = []
    execution_receipt = tmp_path / "data" / "research" / "registered-generation-2.json"
    execution_receipt.parent.mkdir(parents=True)
    execution_receipt.write_text("{}", encoding="utf-8")

    def registered_runner(**kwargs):
        rerun_calls.append(kwargs)
        return CommandResult(
            paths={"execution_receipt": execution_receipt},
            summary={"status": "PASS_REGISTERED_RERUN_ACCOUNTED"},
        )

    def budget_builder(**_):
        path = active / "budget.json"
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"budget_summary": path},
            summary={
                "status": "PASS",
                "scheduled_credit_ceiling": 100,
                "headroom_after_reserve": 800,
                "exact_mode_proof_credit_ceiling": 2,
                "copula_behavioral_credit_ceiling": 0,
                "daily_credit_limit": 1000,
                "reserved_credits": 100,
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        internal_continuation_only=True,
        queue_builder=lambda root: _queue_builder(root, eligible=1),
        proof_runner=lambda **_: (_ for _ in ()).throw(
            AssertionError("internal continuation must not call the Wizard proof runner")
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={"status": "PASS", "ready_rows": 1, "blocked_rows": 0},
        ),
        budget_builder=budget_builder,
        capture_manifest_builder=_capture_manifest_builder(),
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=1),
        credit_reserver=lambda **_: CommandResult(
            paths={},
            summary={
                "status": "REUSED",
                "blocker": "",
                "reservation_id": "reservation-1",
                "lane_reconciliation_ids": ["reconcile-1"],
                "lane_remaining_reserved_credits": 2,
            },
        ),
        credit_reconciler=lambda **_: CommandResult(
            paths={},
            summary={
                "status": "REUSED_RECONCILIATION",
                "blocker": "",
                "reconciliation_id": "reconcile-1",
            },
        ),
        parity_refresher=lambda **_: CommandResult(paths={}, summary={"status": "PASS"}),
        checkpoint_refresher=lambda **_: CommandResult(
            paths={}, summary={"operational_acceptance_status": "BLOCKED"}
        ),
        registered_rerun_runner=registered_runner,
    )

    assert result.summary["internal_continuation_only"] is True
    assert result.summary["external_attempt_made_this_cycle"] is False
    assert result.summary["proof_request_slots_selected"] == 0
    assert result.summary["registered_rerun_status"] == ("PASS_REGISTERED_RERUN_ACCOUNTED")
    assert len(rerun_calls) == 1
    assert rerun_calls[0]["execute"] is True
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_post_registered_rerun_checkpoint_failure_is_persisted_fail_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    execution_receipt = tmp_path / "data" / "research" / "registered.json"
    execution_receipt.parent.mkdir(parents=True)
    execution_receipt.write_text("{}", encoding="utf-8")
    checkpoint_calls = []

    def checkpoint_refresher(**kwargs):
        checkpoint_calls.append(kwargs)
        if len(checkpoint_calls) == 2:
            raise RuntimeError("post-rerun refresh unavailable")
        return CommandResult(
            paths={}, summary={"operational_acceptance_status": "PRE_RERUN_REFRESHED"}
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=1),
        proof_runner=lambda **_: _result(
            eligible=1,
            selected=1,
            completed=1,
            selected_completed=1,
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={"status": "PASS", "ready_rows": 1, "blocked_rows": 0},
        ),
        capture_manifest_builder=_capture_manifest_builder(exact_calls=1),
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=1),
        parity_refresher=lambda **_: CommandResult(paths={}, summary={"status": "PASS"}),
        checkpoint_refresher=checkpoint_refresher,
        registered_rerun_runner=lambda **_: CommandResult(
            paths={"execution_receipt": execution_receipt},
            summary={"status": "PASS_REGISTERED_RERUN_ACCOUNTED"},
        ),
        credits_fetcher=lambda **_: {"credits_used": 22},
    )

    assert result.summary["registered_rerun_status"] == "PASS_REGISTERED_RERUN_ACCOUNTED"
    assert result.summary["checkpoint_pre_rerun_status"] == "PRE_RERUN_REFRESHED"
    assert result.summary["checkpoint_post_rerun_status"] == "REFRESH_FAILED"
    assert result.summary["checkpoint_refresh_status"] == "REFRESH_FAILED"
    assert result.summary["checkpoint_refresh_phase"] == "POST_REGISTERED_RERUN"
    assert any(
        blocker.startswith("post_registered_rerun_checkpoint_refresh_failed:RuntimeError")
        for blocker in result.summary["blockers"]
    )


def test_mixed_formula_and_copula_completion_hands_off_without_formula_relabel(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    contract = tmp_path / "config" / "wizard_copula_behavioral_parity.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    rerun_calls = []
    execution_receipt = tmp_path / "data" / "research" / "mixed.json"
    execution_receipt.parent.mkdir(parents=True)
    execution_receipt.write_text("{}", encoding="utf-8")

    def copula_runner(**kwargs):
        path = tmp_path / "reports" / "active" / "copula_mixed.json"
        path.write_text("{}", encoding="utf-8")
        assert kwargs["execute"] is True
        cohort = _write_copula_cohort(tmp_path)
        return CommandResult(
            paths={"status": path},
            summary={
                "status": "PASS",
                "expected_cells": 1,
                "behavioral_cells_passed": 1,
                "provenance_cells_complete": 1,
                "endpoint_calls_made": 2,
                "endpoint_responses_captured_this_cycle": 2,
                "behavioral_parity_proven": True,
                "formula_parity_proven": False,
                **cohort,
            },
        )

    def registered_runner(**kwargs):
        rerun_calls.append(kwargs)
        return CommandResult(
            paths={"execution_receipt": execution_receipt},
            summary={"status": "PASS_REGISTERED_RERUN_ACCOUNTED"},
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=2),
        proof_runner=lambda **_: _result(
            eligible=2,
            selected=2,
            completed=1,
            selected_completed=1,
            responses_captured=2,
            selected_responses_captured=2,
            external_proof_requests=2,
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={"status": "PASS", "ready_rows": 2, "blocked_rows": 0},
        ),
        capture_manifest_builder=_capture_manifest_builder(exact_calls=2, copula_calls=2),
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=4),
        copula_behavioral_runner=copula_runner,
        parity_refresher=lambda **_: CommandResult(paths={}, summary={"status": "PASS"}),
        checkpoint_refresher=lambda **_: CommandResult(
            paths={}, summary={"operational_acceptance_status": "BLOCKED"}
        ),
        registered_rerun_runner=registered_runner,
        credits_fetcher=lambda **_: {"credits_used": 26},
    )

    assert result.summary["status"] == "COMPLETE_ACCEPTED_MODE_EVIDENCE"
    assert result.summary["completed_after"] == 1
    assert result.summary["responses_captured_after"] == 2
    assert result.summary["formula_proofs_expected"] == 1
    assert result.summary["accepted_mode_evidence_cells"] == 2
    assert result.summary["copula_formula_parity_proven"] is False
    assert result.summary["exact_mode_requests_attempted"] == 2
    assert result.summary["exact_mode_attempted_credits"] == 4
    assert result.summary["copula_behavioral_attempted_credits"] == 2
    assert result.summary["proof_lane_attempted_credits"] == 6
    assert result.summary["proof_lane_completed_credits"] == 6
    assert result.summary["proof_lane_uncompleted_attempted_credits"] == 0
    assert len(rerun_calls) == 1
    assert result.summary["candidate_promotion_authority"] is False
    assert result.summary["testnet_order_authority"] is False


def test_copula_pass_without_immutable_cohort_cannot_handoff(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    contract = tmp_path / "config" / "wizard_copula_behavioral_parity.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    rerun_calls = []

    def copula_runner(**kwargs):
        path = tmp_path / "reports" / "active" / "copula_unbound.json"
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"status": path},
            summary={
                "status": "PASS",
                "expected_cells": 1,
                "behavioral_cells_passed": 1,
                "provenance_cells_complete": 1,
                "endpoint_calls_made": 2,
                "endpoint_responses_captured_this_cycle": 2,
                "behavioral_parity_proven": True,
                "formula_parity_proven": False,
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=2),
        proof_runner=lambda **_: _result(
            eligible=2,
            selected=2,
            completed=1,
            selected_completed=1,
            responses_captured=2,
            selected_responses_captured=2,
            external_proof_requests=2,
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={"status": "PASS", "ready_rows": 2, "blocked_rows": 0},
        ),
        capture_manifest_builder=_capture_manifest_builder(exact_calls=2, copula_calls=2),
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=4),
        copula_behavioral_runner=copula_runner,
        parity_refresher=lambda **_: CommandResult(paths={}, summary={"status": "PASS"}),
        checkpoint_refresher=lambda **_: CommandResult(
            paths={}, summary={"operational_acceptance_status": "BLOCKED"}
        ),
        registered_rerun_runner=lambda **kwargs: rerun_calls.append(kwargs),
        credits_fetcher=lambda **_: {"credits_used": 26},
    )

    assert result.summary["status"] == ("BLOCKED_ACCEPTED_MODE_EVIDENCE_MISMATCH")
    assert result.summary["accepted_mode_evidence_cells"] == 1
    assert result.summary["copula_cohort_receipt_valid"] is False
    assert "copula_behavioral_pass_missing_valid_immutable_cohort" in (result.summary["blockers"])
    assert rerun_calls == []
    assert result.summary["candidate_promotion_authority"] is False
    assert result.summary["testnet_order_authority"] is False


def test_copula_pass_with_incomplete_response_accounting_cannot_handoff(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    contract = tmp_path / "config" / "wizard_copula_behavioral_parity.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    rerun_calls = []

    def copula_runner(**kwargs):
        path = tmp_path / "reports" / "active" / "copula_partial.json"
        path.write_text("{}", encoding="utf-8")
        cohort = _write_copula_cohort(tmp_path, cohort_id="copulacohort_partial")
        return CommandResult(
            paths={"status": path},
            summary={
                "status": "PASS",
                "expected_cells": 1,
                "behavioral_cells_passed": 1,
                "provenance_cells_complete": 1,
                "endpoint_calls_made": 2,
                "endpoint_responses_captured_this_cycle": 1,
                "behavioral_parity_proven": True,
                "formula_parity_proven": False,
                **cohort,
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=2),
        proof_runner=lambda **_: _result(
            eligible=2,
            selected=2,
            completed=1,
            selected_completed=1,
            responses_captured=2,
            selected_responses_captured=2,
            external_proof_requests=2,
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={"status": "PASS", "ready_rows": 2, "blocked_rows": 0},
        ),
        capture_manifest_builder=_capture_manifest_builder(exact_calls=2, copula_calls=2),
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=4),
        copula_behavioral_runner=copula_runner,
        parity_refresher=lambda **_: CommandResult(paths={}, summary={"status": "PASS"}),
        checkpoint_refresher=lambda **_: CommandResult(
            paths={}, summary={"operational_acceptance_status": "BLOCKED"}
        ),
        registered_rerun_runner=lambda **kwargs: rerun_calls.append(kwargs),
        credits_fetcher=lambda **_: {"credits_used": 26},
    )

    assert result.summary["status"] == "BLOCKED_ACCEPTED_MODE_EVIDENCE_MISMATCH"
    assert result.summary["copula_behavioral_response_accounting_valid"] is False
    assert result.summary["copula_behavioral_parity_proven"] is False
    assert result.summary["copula_behavioral_attempted_credits"] == 2
    assert result.summary["copula_behavioral_completed_credits"] == 1
    assert result.summary["proof_lane_attempted_credits"] == 6
    assert result.summary["proof_lane_completed_credits"] == 5
    assert result.summary["proof_lane_uncompleted_attempted_credits"] == 1
    assert (
        "copula_behavioral_pass_has_incomplete_response_accounting" in (result.summary["blockers"])
    )
    assert result.summary["accepted_mode_evidence_cells"] == 1
    assert rerun_calls == []
    assert result.summary["candidate_promotion_authority"] is False
    assert result.summary["testnet_order_authority"] is False


def test_reviewed_dynamic_and_ou_v4_activations_complete_full_mixed_handoff_same_day(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    config = tmp_path / "config"
    config.mkdir(parents=True)
    (config / "wizard_copula_behavioral_parity.json").write_text("{}", encoding="utf-8")
    (config / "wizard_dynamic_comparator_v2_holdout.json").write_text("{}", encoding="utf-8")
    (config / "wizard_ou_comparator_v4_holdout.json").write_text("{}", encoding="utf-8")
    state = {
        "completed": 8,
        "responses": 8,
        "reviewed": False,
        "external_exact_calls": 0,
    }
    proof_calls = []
    copula_calls = []
    registered_calls = []

    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_wizard_proof_scheduler."
        "count_completed_exact_mode_proofs",
        lambda **_: state["completed"],
    )
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_wizard_proof_scheduler."
        "count_captured_exact_mode_responses",
        lambda **_: state["responses"],
    )

    def proof_runner(**kwargs):
        proof_calls.append(kwargs)
        remaining = 20 - state["external_exact_calls"]
        selected = min(int(kwargs["max_pairs"]), remaining)
        state["external_exact_calls"] += selected
        state["responses"] = 8 + state["external_exact_calls"]
        return _result(
            eligible=28,
            selected=selected,
            completed=8,
            selected_completed=0,
            responses_captured=state["responses"],
            selected_responses_captured=selected,
            external_proof_requests=selected,
        )

    def dynamic_result(name, summary):
        path = tmp_path / "reports" / "active" / f"{name}.json"
        path.write_text("{}", encoding="utf-8")
        return CommandResult(paths={name: path}, summary=summary)

    def dynamic_refresher(**kwargs):
        if state["reviewed"]:
            state["completed"] = 16
            status = "PASS"
            refreshed = 8
        else:
            status = "NOT_ACTIVE"
            refreshed = 0
        return dynamic_result(
            "refresh_status",
            {
                "status": status,
                "refreshed_dynamic_rows": refreshed,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
            },
        )

    def ou_v4_refresher(**kwargs):
        if state["reviewed"]:
            state["completed"] = 24
            status = "PASS"
            refreshed = 8
        else:
            status = "NOT_ACTIVE"
            refreshed = 0
        return dynamic_result(
            "ou_v4_refresh_status",
            {
                "status": status,
                "refreshed_ou_rows": refreshed,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
            },
        )

    def copula_runner(**kwargs):
        copula_calls.append(kwargs)
        path = tmp_path / "reports" / "active" / "copula_full.json"
        path.write_text("{}", encoding="utf-8")
        cohort = _write_copula_cohort(tmp_path, cohort_id="copulacohort_full")
        executing = bool(kwargs["execute"])
        return CommandResult(
            paths={"status": path},
            summary={
                "status": "PASS",
                "expected_cells": 4,
                "behavioral_cells_passed": 4,
                "provenance_cells_complete": 4,
                "endpoint_calls_made": 8 if executing else 0,
                "endpoint_responses_captured_this_cycle": 8 if executing else 0,
                "endpoint_calls_required": 8 if executing else 0,
                "newly_captured_cells": 4 if executing else 0,
                "reused_immutable_capture_cells": 0 if executing else 4,
                "behavioral_parity_proven": True,
                "formula_parity_proven": False,
                **cohort,
            },
        )

    def parity_refresher(**kwargs):
        return CommandResult(
            paths={},
            summary={"status": "PASS" if state["completed"] == 24 else "BLOCKED"},
        )

    execution_receipt = tmp_path / "data" / "research" / "full-rerun.json"
    execution_receipt.parent.mkdir(parents=True, exist_ok=True)
    execution_receipt.write_text("{}", encoding="utf-8")

    def registered_runner(**kwargs):
        registered_calls.append(kwargs)
        return CommandResult(
            paths={"execution_receipt": execution_receipt},
            summary={"status": "PASS_REGISTERED_RERUN_ACCOUNTED"},
        )

    common = {
        "root": tmp_path,
        "execute": True,
        "queue_builder": lambda root: _queue_builder(root, eligible=28),
        "proof_runner": proof_runner,
        "input_auditor": lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={"status": "PASS", "ready_rows": 28, "blocked_rows": 0},
        ),
        "capture_manifest_builder": _capture_manifest_builder(exact_calls=20, copula_calls=8),
        "capture_manifest_reconciler": _passing_capture_reconciler(required_calls=28),
        "dynamic_holdout_evaluator": lambda **_: dynamic_result(
            "status",
            {
                "status": "PASS",
                "comparator_supersession_eligible": True,
            },
        ),
        "dynamic_supersession_gate_builder": lambda **_: dynamic_result(
            "supersession_gate",
            {"status": "READY_FOR_REVIEWED_SUPERSESSION"},
        ),
        "dynamic_activation_planner": lambda **_: dynamic_result(
            "activation_status",
            {
                "status": (
                    "APPLIED_RESEARCH_COMPARATOR_ONLY"
                    if state["reviewed"]
                    else "READY_REQUIRES_EXPLICIT_APPLY"
                ),
                "comparator_generation": 2,
            },
        ),
        "dynamic_review_packet_builder": lambda **_: dynamic_result(
            "review_packet",
            {
                "status": "READY_FOR_EXPLICIT_REVIEW",
                "review_packet_id": "dynamicv2review_full",
            },
        ),
        "dynamic_proof_refresher": dynamic_refresher,
        "ou_v4_holdout_runner": lambda **_: dynamic_result(
            "capture_status",
            {
                "status": "COMPLETE",
                "evaluation_status": "PASS",
                "missing_cells_before": 0,
                "required_responses": 8,
                "responses_available": 8,
                "calls_made": 0,
                "responses_captured": 0,
                "credits_attempted": 0,
                "credits_completed": 0,
            },
        ),
        "ou_v4_supersession_gate_builder": lambda **_: dynamic_result(
            "ou_v4_supersession_gate",
            {"status": "READY_FOR_REVIEWED_SUPERSESSION"},
        ),
        "ou_v4_review_packet_builder": lambda **_: dynamic_result(
            "ou_v4_review_packet",
            {
                "status": "READY_FOR_EXPLICIT_REVIEW",
                "review_packet_id": "ouv4review_full",
            },
        ),
        "ou_v4_supreme_reviewer": lambda **_: dynamic_result(
            "ou_v4_supreme_review",
            {
                "status": "PASS_ADVISORY_ONLY",
                "recommendation": "RECOMMEND_EXPLICIT_RESEARCH_COMPARATOR_ACTIVATION",
                "activation_applied_by_supreme_team": False,
            },
        ),
        "ou_v4_activation_planner": lambda **_: dynamic_result(
            "ou_v4_activation_status",
            {
                "status": (
                    "APPLIED_RESEARCH_COMPARATOR_ONLY"
                    if state["reviewed"]
                    else "READY_REQUIRES_EXPLICIT_APPLY"
                ),
                "comparator_generation": 4,
            },
        ),
        "ou_v4_proof_refresher": ou_v4_refresher,
        "copula_behavioral_runner": copula_runner,
        "parity_refresher": parity_refresher,
        "checkpoint_refresher": lambda **_: CommandResult(
            paths={}, summary={"operational_acceptance_status": "BLOCKED"}
        ),
        "registered_rerun_runner": registered_runner,
        "credits_fetcher": lambda **_: {"credits_used": 68},
    }

    first = run_corrective_wizard_proof_cycle(now=NOW, **common)
    state["reviewed"] = True
    second = run_corrective_wizard_proof_cycle(now=NOW + timedelta(minutes=10), **common)

    assert len(proof_calls) == 7
    assert [call["execute"] for call in copula_calls] == [True, False]
    assert first.summary["accepted_mode_evidence_cells"] == 12
    assert first.summary["registered_rerun_status"] == "NOT_EVALUATED"
    assert second.summary["status"] == "COMPLETE_ACCEPTED_MODE_EVIDENCE"
    assert second.summary["completed_after"] == 24
    assert second.summary["formula_proofs_expected"] == 24
    assert second.summary["accepted_mode_evidence_cells"] == 28
    assert second.summary["dynamic_v2_comparator_generation"] == 2
    assert second.summary["dynamic_v2_proofs_refreshed"] == 8
    assert second.summary["ou_v4_comparator_generation"] == 4
    assert second.summary["ou_v4_proofs_refreshed"] == 8
    assert second.summary["copula_behavioral_reused_capture_cells"] == 4
    assert second.summary["copula_cohort_receipt_valid"] is True
    assert second.summary["registered_rerun_status"] == ("PASS_REGISTERED_RERUN_ACCOUNTED")
    assert len(registered_calls) == 1
    assert registered_calls[0]["execute"] is True
    assert second.summary["candidate_promotion_authority"] is False
    assert second.summary["testnet_order_authority"] is False
    assert second.summary["live_trading_authorized"] is False


def test_credit_block_retries_without_claiming_a_proof_attempt(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    calls = []

    def proof_runner(**kwargs):
        calls.append(kwargs)
        return _result(
            eligible=5,
            selected=3,
            completed=0,
            selected_completed=0,
            credit_status="BLOCKED",
            credit_blocker="insufficient_credits_after_reserve",
        )

    first = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=_queue_builder,
        proof_runner=proof_runner,
        input_auditor=_input_auditor,
        checkpoint_refresher=None,
    )
    second = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW + timedelta(minutes=10),
        execute=True,
        queue_builder=_queue_builder,
        proof_runner=proof_runner,
        input_auditor=_input_auditor,
        checkpoint_refresher=None,
    )
    third = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW + timedelta(minutes=20),
        execute=True,
        queue_builder=_queue_builder,
        proof_runner=proof_runner,
        input_auditor=_input_auditor,
        checkpoint_refresher=None,
    )

    assert first.summary["status"] == "DEFERRED_CREDIT_RESET"
    assert "insufficient_credits_after_reserve" in first.summary["blockers"]
    assert first.summary["external_attempt_made_this_cycle"] is False
    assert second.summary["status"] == "DEFERRED_CREDIT_RESET"
    assert second.summary["external_attempt_made"] is False
    assert second.summary["external_attempt_made_this_cycle"] is False
    assert third.summary["status"] == "DEFERRED_CREDIT_RESET"
    assert third.summary["external_attempt_made"] is False
    assert third.summary["external_attempt_made_this_cycle"] is False
    assert len(calls) == 3


def test_real_proof_request_enforces_same_day_attempt_cap(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    calls = []

    def proof_runner(**kwargs):
        calls.append(kwargs)
        return _result(
            eligible=3,
            selected=3,
            completed=3,
            selected_completed=3,
            responses_captured=3,
            selected_responses_captured=3,
            external_proof_requests=3,
        )

    first = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=3),
        proof_runner=proof_runner,
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={"status": "PASS", "ready_rows": 3, "blocked_rows": 0},
        ),
        checkpoint_refresher=None,
        credits_fetcher=lambda **_: {"credits_used": 26},
    )
    planning = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW + timedelta(minutes=10),
        execute=False,
        queue_builder=lambda root: _queue_builder(root, eligible=3),
        proof_runner=proof_runner,
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={"status": "PASS", "ready_rows": 3, "blocked_rows": 0},
        ),
        checkpoint_refresher=None,
        credits_fetcher=lambda **_: {"credits_used": 26},
    )
    second = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW + timedelta(minutes=20),
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=3),
        proof_runner=proof_runner,
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={"status": "PASS", "ready_rows": 3, "blocked_rows": 0},
        ),
        checkpoint_refresher=None,
        credits_fetcher=lambda **_: {"credits_used": 26},
    )

    assert first.summary["status"] == "COMPLETE_QUEUE"
    assert first.summary["external_attempt_made_this_cycle"] is True
    assert planning.summary["status"] == "PLANNED"
    assert planning.summary["external_attempt_made"] is True
    assert planning.summary["external_attempt_made_this_cycle"] is False
    assert second.summary["status"] == "DEFERRED_SAME_UTC_DAY"
    assert [call["execute"] for call in calls] == [True, False]


def test_copula_behavioral_endpoint_executes_only_with_fresh_parent_attempt(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    contract = tmp_path / "config" / "wizard_copula_behavioral_parity.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    copula_calls = []

    def copula_runner(**kwargs):
        copula_calls.append(kwargs)
        path = tmp_path / "reports" / "active" / "copula_status.json"
        path.write_text("{}", encoding="utf-8")
        executing = bool(kwargs["execute"])
        cohort = _write_copula_cohort(tmp_path)
        return CommandResult(
            paths={"status": path},
            summary={
                "status": "PASS",
                "expected_cells": 4,
                "behavioral_cells_passed": 4,
                "provenance_cells_complete": 4,
                "endpoint_calls_made": 8 if executing else 0,
                "endpoint_responses_captured_this_cycle": 8 if executing else 0,
                "endpoint_calls_required": 8 if executing else 0,
                "newly_captured_cells": 4 if executing else 0,
                "reused_immutable_capture_cells": 0 if executing else 4,
                "behavioral_parity_proven": True,
                "formula_parity_proven": False,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
                **cohort,
            },
        )

    first = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=3),
        proof_runner=lambda **_: _result(
            eligible=3,
            selected=3,
            completed=3,
            selected_completed=3,
            external_proof_requests=3,
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={"status": "PASS", "ready_rows": 3, "blocked_rows": 0},
        ),
        copula_behavioral_runner=copula_runner,
        checkpoint_refresher=None,
        credits_fetcher=lambda **_: {"credits_used": 34},
    )
    second = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW + timedelta(minutes=10),
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=3),
        proof_runner=lambda **_: (_ for _ in ()).throw(
            AssertionError("same-day parent proof call must remain deferred")
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={"status": "PASS", "ready_rows": 3, "blocked_rows": 0},
        ),
        copula_behavioral_runner=copula_runner,
        checkpoint_refresher=None,
        credits_fetcher=lambda **_: {"credits_used": 34},
    )

    assert [call["execute"] for call in copula_calls] == [True, False]
    assert first.summary["copula_behavioral_status"] == "PASS"
    assert first.summary["copula_behavioral_endpoint_calls"] == 8
    assert first.summary["copula_behavioral_provenance_cells"] == 4
    assert first.summary["copula_behavioral_parity_proven"] is True
    assert first.summary["copula_formula_parity_proven"] is False
    assert second.summary["status"] == "DEFERRED_SAME_UTC_DAY"
    assert second.summary["copula_behavioral_status"] == "PASS"
    assert second.summary["copula_behavioral_endpoint_calls"] == 0
    assert second.summary["copula_behavioral_endpoint_calls_required"] == 0
    assert second.summary["copula_behavioral_provenance_cells"] == 4
    assert second.summary["copula_behavioral_reused_capture_cells"] == 4
    assert second.summary["copula_cohort_receipt_valid"] is True
    assert second.summary["candidate_promotion_authority"] is False
    assert second.summary["testnet_order_authority"] is False


def test_copula_capture_can_execute_when_exact_lane_has_no_pending_call(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    contract = tmp_path / "config" / "wizard_copula_behavioral_parity.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    copula_calls = []

    def capture_manifest_builder(**kwargs):
        return _capture_manifest_builder(copula_calls=8)(**kwargs)

    def copula_runner(**kwargs):
        copula_calls.append(kwargs)
        path = tmp_path / "reports" / "active" / "copula_status.json"
        path.write_text("{}", encoding="utf-8")
        cohort = _write_copula_cohort(tmp_path)
        return CommandResult(
            paths={"status": path},
            summary={
                "status": "PASS",
                "expected_cells": 4,
                "behavioral_cells_passed": 4,
                "provenance_cells_complete": 4,
                "endpoint_calls_made": 8,
                "endpoint_responses_captured_this_cycle": 8,
                "endpoint_calls_required": 8,
                "newly_captured_cells": 4,
                "reused_immutable_capture_cells": 0,
                "behavioral_parity_proven": True,
                "formula_parity_proven": False,
                **cohort,
            },
        )

    def capture_manifest_reconciler(**kwargs):
        path = tmp_path / "reports" / "active" / "capture_reconciliation.json"
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"status": path},
            summary={
                "status": "PASS",
                "required_calls": 8,
                "completed_calls": 8,
                "pending_calls": 0,
                "blocked_calls": 0,
                "blockers": [],
                "reconciliation_id": "wizardcapturerecon_test",
                "immutable_reconciliation_path": (
                    "data/research/wizard_capture_reconciliations/test.json"
                ),
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=0),
        proof_runner=lambda **_: _result(
            eligible=0,
            selected=0,
            completed=0,
            selected_completed=0,
            external_proof_requests=0,
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={
                "status": "PASS",
                "ready_rows": 0,
                "blocked_rows": 0,
                "retry_safe_rows": 0,
                "unchanged_vendor_4xx_rows": 0,
            },
        ),
        capture_manifest_builder=capture_manifest_builder,
        capture_manifest_reconciler=capture_manifest_reconciler,
        copula_behavioral_runner=copula_runner,
        checkpoint_refresher=None,
    )

    assert [call["execute"] for call in copula_calls] == [True]
    assert result.summary["exact_mode_requests_attempted"] == 0
    assert result.summary["copula_behavioral_endpoint_calls"] == 8
    assert result.summary["external_attempt_made_this_cycle"] is True
    assert result.summary["capture_manifest_accounting_valid"] is True
    assert result.summary["capture_reconciliation_status"] == "PASS"
    assert result.summary["capture_reconciliation_valid"] is True
    assert result.summary["testnet_order_authority"] is False


def test_executing_manifest_cannot_complete_with_pending_reconciliation(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)

    def capture_manifest_builder(**kwargs):
        return _capture_manifest_builder(exact_calls=1)(**kwargs)

    def capture_manifest_reconciler(**kwargs):
        path = tmp_path / "reports" / "active" / "capture_reconciliation.json"
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"status": path},
            summary={
                "status": "PENDING",
                "required_calls": 1,
                "completed_calls": 0,
                "pending_calls": 1,
                "blocked_calls": 0,
                "blockers": [],
                "reconciliation_id": "",
                "immutable_reconciliation_path": "",
            },
        )

    registered_calls = []
    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=1),
        proof_runner=lambda **_: _result(
            eligible=1,
            selected=1,
            completed=1,
            selected_completed=1,
            external_proof_requests=1,
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={
                "status": "PASS",
                "ready_rows": 1,
                "blocked_rows": 0,
                "retry_safe_rows": 1,
                "unchanged_vendor_4xx_rows": 0,
            },
        ),
        capture_manifest_builder=capture_manifest_builder,
        capture_manifest_reconciler=capture_manifest_reconciler,
        parity_refresher=lambda **_: CommandResult(paths={}, summary={"status": "PASS"}),
        checkpoint_refresher=None,
        registered_rerun_runner=lambda **kwargs: registered_calls.append(kwargs),
        credits_fetcher=lambda **_: {"credits_used": 22},
    )

    assert result.summary["status"] == "BLOCKED_CAPTURE_MANIFEST_RECONCILIATION"
    assert result.summary["capture_reconciliation_status"] == "PENDING"
    assert result.summary["capture_reconciliation_valid"] is False
    assert result.summary["capture_reconciliation_complete"] is False
    assert result.summary["capture_reconciliation_completed_calls"] == 0
    assert result.summary["capture_reconciliation_required_calls"] == 1
    assert any(
        "capture_manifest_responses_not_fully_reconciled:0_of_1" in blocker
        for blocker in result.summary["blockers"]
    )
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    assert registered_calls == []


def test_plan_only_pending_reconciliation_is_safe_but_not_valid_evidence(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)

    def pending_reconciler(*, root, **_):
        path = root / "reports" / "active" / "capture_reconciliation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"status": path},
            summary={
                "status": "PENDING",
                "required_calls": 1,
                "completed_calls": 0,
                "pending_calls": 1,
                "blocked_calls": 0,
                "blockers": [],
                "reconciliation_id": "",
                "immutable_reconciliation_path": "",
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=False,
        queue_builder=lambda root: _queue_builder(root, eligible=1),
        proof_runner=lambda **_: _result(
            eligible=1,
            selected=1,
            completed=0,
            selected_completed=0,
            external_proof_requests=0,
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={
                "status": "PASS",
                "ready_rows": 1,
                "blocked_rows": 0,
                "retry_safe_rows": 1,
                "unchanged_vendor_4xx_rows": 0,
            },
        ),
        capture_manifest_builder=_capture_manifest_builder(exact_calls=1),
        capture_manifest_reconciler=pending_reconciler,
        checkpoint_refresher=None,
    )

    assert result.summary["status"] == "PLANNED"
    assert result.summary["capture_reconciliation_status"] == "PENDING"
    assert result.summary["capture_reconciliation_local_state_valid"] is True
    assert result.summary["capture_reconciliation_valid"] is False
    assert result.summary["capture_reconciliation_complete"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_zero_call_continuation_carries_only_verified_immutable_reconciliation(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    capture_evidence = write_valid_capture_reconciliation_evidence(tmp_path)
    previous = {
        "attempt_date_utc": NOW.date().isoformat(),
        "external_attempt_made": True,
        **capture_evidence,
    }
    (active / "corrective_wizard_proof_scheduler_status.json").write_text(
        json.dumps(previous, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW + timedelta(minutes=5),
        execute=False,
        queue_builder=lambda root: _queue_builder(root, eligible=0),
        proof_runner=lambda **_: _result(
            eligible=0,
            selected=0,
            completed=0,
            selected_completed=0,
            external_proof_requests=0,
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={
                "status": "PASS",
                "ready_rows": 0,
                "blocked_rows": 0,
                "retry_safe_rows": 0,
                "unchanged_vendor_4xx_rows": 0,
            },
        ),
        capture_manifest_builder=_capture_manifest_builder(),
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=0),
        checkpoint_refresher=None,
    )

    assert result.summary["capture_reconciliation_current_status"] == "NOT_REQUIRED"
    assert result.summary["capture_reconciliation_carried_forward"] is True
    assert result.summary["capture_reconciliation_status"] == "PASS"
    assert result.summary["capture_reconciliation_complete"] is True
    assert (
        result.summary["capture_reconciliation_id"] == capture_evidence["capture_reconciliation_id"]
    )
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_unresolved_frozen_manifest_match_allows_external_runner(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    manifest_builder = _capture_manifest_builder(exact_calls=1)
    prior_manifest = manifest_builder(root=tmp_path, now=NOW)
    prior_time = NOW - timedelta(days=1)
    prior_reservation = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        planned_credits=2,
        now=prior_time,
    )
    prior_reconciliation = reconcile_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        reservation_id=prior_reservation.summary["reservation_id"],
        reconciliation_key=prior_time.isoformat(),
        attempted_credits=0,
        completed_credits=0,
        external_requests=0,
        now=prior_time,
    )
    previous = _write_pending_manifest_scheduler_status(tmp_path, prior_manifest)
    previous.update(
        {
            "attempt_date_utc": prior_time.date().isoformat(),
            "external_attempt_made": True,
            "credit_reservation_id": prior_reservation.summary["reservation_id"],
            "credit_reservation_path": str(
                prior_reservation.paths["reservation"].relative_to(tmp_path)
            ),
            "credit_reconciliation_id": prior_reconciliation.summary["reconciliation_id"],
            "credit_reconciliation_path": str(
                prior_reconciliation.paths["reconciliation"].relative_to(tmp_path)
            ),
        }
    )
    status_path = tmp_path / "reports" / "active" / "corrective_wizard_proof_scheduler_status.json"
    status_path.write_text(
        json.dumps(previous, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    proof_calls = []

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=1),
        proof_runner=lambda **kwargs: (
            proof_calls.append(kwargs)
            or _result(
                eligible=1,
                selected=1,
                completed=1,
                selected_completed=1,
            )
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={
                "status": "PASS",
                "ready_rows": 1,
                "blocked_rows": 0,
                "retry_safe_rows": 1,
                "unchanged_vendor_4xx_rows": 0,
            },
        ),
        capture_manifest_builder=manifest_builder,
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=1),
        parity_refresher=lambda **_: CommandResult(paths={}, summary={"status": "PASS"}),
        registered_rerun_runner=None,
        checkpoint_refresher=None,
        copula_behavioral_runner=None,
        ou_v3_holdout_runner=None,
    )

    assert len(proof_calls) == 1
    assert result.summary["capture_manifest_carried_forward"] is True
    assert result.summary["capture_manifest_drift_detected"] is False
    assert result.summary["capture_manifest_continuity_valid"] is True
    assert result.summary["capture_manifest_continuity_status"] == (
        "PASS_PRIOR_UNRESOLVED_COHORT_MATCH"
    )
    assert result.summary["credit_reservation_status"] == "PASS"
    assert result.summary["credit_reservation_id"] != prior_reservation.summary["reservation_id"]
    assert result.summary["credit_reservation_path"] == (
        "data/research/wizard_credit_ledger/2026-08-10/"
        "reservations/exact_mode_and_copula_proofs.json"
    )
    assert result.summary["credit_reservation_planned_credits"] == 2
    assert result.summary["proof_lane_attempted_credits"] == 2
    assert result.summary["proof_lane_completed_credits"] == 2


def test_unresolved_frozen_manifest_drift_blocks_before_credit_or_vendor_calls(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    prior_manifest = _capture_manifest_builder(exact_calls=1)(root=tmp_path, now=NOW)
    previous = _write_pending_manifest_scheduler_status(tmp_path, prior_manifest)
    vendor_calls = []
    reservation_calls = []

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=2),
        proof_runner=lambda **kwargs: vendor_calls.append(kwargs),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={
                "status": "PASS",
                "ready_rows": 2,
                "blocked_rows": 0,
                "retry_safe_rows": 2,
                "unchanged_vendor_4xx_rows": 0,
            },
        ),
        capture_manifest_builder=_capture_manifest_builder(exact_calls=2),
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=1),
        credit_reserver=lambda **kwargs: reservation_calls.append(kwargs),
        registered_rerun_runner=None,
        checkpoint_refresher=None,
        copula_behavioral_runner=None,
        ou_v3_holdout_runner=None,
    )

    assert vendor_calls == []
    assert reservation_calls == []
    assert result.summary["status"] == "BLOCKED_CAPTURE_MANIFEST_DRIFT"
    assert result.summary["capture_manifest_id"] == previous["capture_manifest_id"]
    assert result.summary["capture_manifest_candidate_id"] != previous["capture_manifest_id"]
    assert result.summary["capture_manifest_carried_forward"] is True
    assert result.summary["capture_manifest_drift_detected"] is True
    assert result.summary["capture_manifest_continuity_valid"] is False
    assert result.summary["external_lanes_authorized_this_cycle"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_tampered_unresolved_manifest_blocks_before_credit_or_vendor_calls(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    prior_manifest = _capture_manifest_builder(exact_calls=1)(root=tmp_path, now=NOW)
    previous = _write_pending_manifest_scheduler_status(tmp_path, prior_manifest)
    immutable = tmp_path / str(previous["capture_manifest_immutable_path"])
    immutable.write_text('{"tampered":true}\n', encoding="utf-8")
    vendor_calls = []
    reservation_calls = []

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=1),
        proof_runner=lambda **kwargs: vendor_calls.append(kwargs),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={
                "status": "PASS",
                "ready_rows": 1,
                "blocked_rows": 0,
                "retry_safe_rows": 1,
                "unchanged_vendor_4xx_rows": 0,
            },
        ),
        capture_manifest_builder=lambda **_: prior_manifest,
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=1),
        credit_reserver=lambda **kwargs: reservation_calls.append(kwargs),
        registered_rerun_runner=None,
        checkpoint_refresher=None,
        copula_behavioral_runner=None,
        ou_v3_holdout_runner=None,
    )

    assert vendor_calls == []
    assert reservation_calls == []
    assert result.summary["status"] == "BLOCKED_CAPTURE_MANIFEST_CONTINUITY"
    assert result.summary["capture_manifest_candidate_binding_valid"] is False
    assert result.summary["capture_manifest_continuity_valid"] is False
    assert any(
        blocker.startswith("prior_capture_manifest_immutable_binding_invalid")
        for blocker in result.summary["capture_manifest_continuity_blockers"]
    )
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_tampered_unresolved_source_snapshot_blocks_before_external_calls(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    prior_manifest = _capture_manifest_builder(exact_calls=1)(root=tmp_path, now=NOW)
    _write_pending_manifest_scheduler_status(tmp_path, prior_manifest)
    source_receipt = json.loads(
        (tmp_path / prior_manifest.summary["source_receipt_path"]).read_text(encoding="utf-8")
    )
    snapshot = tmp_path / source_receipt["source_artifacts"][0]["snapshot_path"]
    snapshot.chmod(0o600)
    snapshot.write_text("tampered\n", encoding="utf-8")
    vendor_calls = []
    reservation_calls = []

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: _queue_builder(root, eligible=1),
        proof_runner=lambda **kwargs: vendor_calls.append(kwargs),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={
                "status": "PASS",
                "ready_rows": 1,
                "blocked_rows": 0,
                "retry_safe_rows": 1,
                "unchanged_vendor_4xx_rows": 0,
            },
        ),
        capture_manifest_builder=lambda **_: prior_manifest,
        capture_manifest_reconciler=_passing_capture_reconciler(required_calls=1),
        credit_reserver=lambda **kwargs: reservation_calls.append(kwargs),
        registered_rerun_runner=None,
        checkpoint_refresher=None,
        copula_behavioral_runner=None,
        ou_v3_holdout_runner=None,
    )

    assert vendor_calls == []
    assert reservation_calls == []
    assert result.summary["status"] == "BLOCKED_CAPTURE_MANIFEST_CONTINUITY"
    assert result.summary["capture_manifest_candidate_source_binding_valid"] is False
    assert any(
        "capture_manifest_source_snapshot_invalid" in blocker
        for blocker in result.summary["capture_manifest_continuity_blockers"]
    )
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_zero_progress_stops_after_one_batch(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    calls = []

    def proof_runner(**kwargs):
        calls.append(kwargs)
        return _result(eligible=5, selected=3, completed=0, selected_completed=0)

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=_queue_builder,
        proof_runner=proof_runner,
        input_auditor=_input_auditor,
        checkpoint_refresher=None,
    )

    assert result.summary["status"] == "BLOCKED_NO_PROGRESS"
    assert result.summary["blockers"] == ["proof_batch_made_no_response_capture_progress"]
    assert len(calls) == 1


def test_external_attempt_without_progress_still_refreshes_checkpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    checkpoint_calls = []

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=_queue_builder,
        proof_runner=lambda **_: _result(
            eligible=5,
            selected=3,
            completed=0,
            selected_completed=0,
            external_proof_requests=3,
        ),
        input_auditor=_input_auditor,
        checkpoint_refresher=lambda **kwargs: (
            checkpoint_calls.append(kwargs)
            or CommandResult(
                paths={},
                summary={"operational_acceptance_status": "BLOCKED"},
            )
        ),
        credits_fetcher=lambda **_: {"credits_used": 26},
    )

    assert result.summary["status"] == "BLOCKED_NO_PROGRESS"
    assert result.summary["external_attempt_made_this_cycle"] is True
    assert result.summary["checkpoint_refresh_status"] == "BLOCKED"
    assert result.summary["checkpoint_refresh_phase"] == "PRE_REGISTERED_RERUN"
    assert len(checkpoint_calls) == 1
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_quarantined_failed_batch_continues_to_unrelated_queue_cells(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    results = iter(
        [
            _result(
                eligible=5,
                selected=3,
                completed=0,
                selected_completed=0,
                responses_captured=0,
                selected_responses_captured=0,
                external_proof_requests=3,
                selected_request_failed=3,
                selected_failures_quarantined=True,
            ),
            _result(
                eligible=5,
                selected=2,
                completed=2,
                selected_completed=2,
                responses_captured=2,
                selected_responses_captured=2,
                external_proof_requests=2,
            ),
            _result(
                eligible=5,
                selected=0,
                completed=2,
                selected_completed=0,
                responses_captured=2,
                selected_responses_captured=0,
                external_proof_requests=0,
            ),
        ]
    )
    calls = []

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=_queue_builder,
        proof_runner=lambda **kwargs: calls.append(kwargs) or next(results),
        input_auditor=_input_auditor,
        checkpoint_refresher=None,
        credits_fetcher=lambda **_: {"credits_used": 30},
    )

    assert result.summary["status"] == "BLOCKED_REMAINING_REQUESTS_FAILED"
    assert result.summary["quarantined_failed_batch_continuation_supported"] is True
    assert result.summary["new_responses_captured"] == 2
    assert result.summary["exact_mode_requests_attempted"] == 5
    assert result.summary["exact_mode_responses_captured_this_cycle"] == 2
    assert result.summary["exact_mode_attempted_credits"] == 10
    assert result.summary["exact_mode_completed_credits"] == 4
    assert result.summary["proof_lane_uncompleted_attempted_credits"] == 6
    assert len(calls) == 3
    assert result.summary["batch_summaries"][0]["batch_outcome"] == (
        "QUARANTINED_REQUEST_FAILURES_CONTINUE"
    )
    assert result.summary["blockers"] == ["one_or_more_proof_requests_failed_without_response"]
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_polling_interval_cannot_expire_an_active_proof_lock(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    lock = active / ".corrective_wizard_proof.lock"
    lock.write_text(
        json.dumps(
            {
                "pid": 999999,
                "started_at_utc": (NOW - timedelta(minutes=20)).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda **_: (_ for _ in ()).throw(
            AssertionError("active proof lock must block a second cycle")
        ),
        checkpoint_refresher=None,
    )

    assert PROOF_LOCK_TIMEOUT_SECONDS == 4 * 60 * 60
    assert result.summary["status"] == "BLOCKED_LOCK"
    assert result.summary["blockers"] == ["active_scheduler_lock_present"]
    assert result.summary["external_attempt_made_this_cycle"] is False
    assert lock.is_file()


def test_response_capture_progress_does_not_masquerade_as_formula_completion(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    results = iter(
        [
            _result(
                eligible=5,
                selected=3,
                completed=0,
                selected_completed=0,
                responses_captured=3,
                selected_responses_captured=3,
            ),
            _result(
                eligible=5,
                selected=2,
                completed=0,
                selected_completed=0,
                responses_captured=5,
                selected_responses_captured=2,
            ),
        ]
    )
    parity_calls = []
    checkpoint_calls = []

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=_queue_builder,
        proof_runner=lambda **_: next(results),
        input_auditor=_input_auditor,
        parity_refresher=lambda **kwargs: (
            parity_calls.append(kwargs) or CommandResult(paths={}, summary={"status": "BLOCKED"})
        ),
        checkpoint_refresher=lambda **kwargs: (
            checkpoint_calls.append(kwargs)
            or CommandResult(
                paths={},
                summary={"operational_acceptance_status": "BLOCKED"},
            )
        ),
        credits_fetcher=lambda **_: {"credits_used": 30},
    )

    assert result.summary["status"] == ("COMPLETE_RESPONSE_CAPTURE_FORMULA_PROOF_INCOMPLETE")
    assert result.summary["new_completed_proofs"] == 0
    assert result.summary["new_responses_captured"] == 5
    assert len(parity_calls) == 1
    assert len(checkpoint_calls) == 1


def test_insecure_secret_file_blocks_before_api_call(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    path = tmp_path / ".env.local"
    path.write_text(f"CRYPTO_WIZARDS_API_KEY={SECRET}\n", encoding="utf-8")
    path.chmod(0o644)
    calls = []

    def proof_runner(**kwargs):
        calls.append(kwargs)
        raise AssertionError("proof runner must not be called")

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=_queue_builder,
        proof_runner=proof_runner,
        input_auditor=_input_auditor,
        checkpoint_refresher=None,
    )

    assert result.summary["status"] == "BLOCKED_ENVIRONMENT"
    assert result.summary["api_key_check_performed"] is False
    assert result.summary["api_key_present"] is False
    assert result.summary["api_key_source"] == "not_checked_before_reservation"
    assert result.summary["insecure_secret_files"] == [".env.local"]
    assert result.summary["blockers"] == ["secret_file_permissions_too_open"]
    assert "CRYPTO_WIZARDS_API_KEY" not in corrective_wizard_proof_scheduler.os.environ
    assert calls == []


def test_authorized_scheduler_environment_prefers_env_local_over_env(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("CRYPTO_WIZARDS_API_KEY=base-key\n", encoding="utf-8")
    env.chmod(0o600)
    local = tmp_path / ".env.local"
    local.write_text("CRYPTO_WIZARDS_API_KEY=local-key\n", encoding="utf-8")
    local.chmod(0o600)

    with reserved_external_effect_session(
        reservation_id="environment-precedence-local",
        reservation_sha256=HASH,
        max_total_requests=1,
        max_total_credits=0,
    ):
        value = read_authorized_credential(
            "CRYPTO_WIZARDS_API_KEY",
            reader=lambda key: corrective_wizard_proof_scheduler._deferred_credential_reader(
                tmp_path, key
            ),
        )

    assert value == "local-key"
    assert corrective_wizard_proof_scheduler.os.environ["CRYPTO_WIZARDS_API_KEY"] == ("local-key")


def test_authorized_scheduler_process_environment_precedes_secret_files(tmp_path, monkeypatch):
    monkeypatch.setenv("CRYPTO_WIZARDS_API_KEY", "process-key")
    _write_secure_env(tmp_path)

    with reserved_external_effect_session(
        reservation_id="environment-precedence-process",
        reservation_sha256=HASH,
        max_total_requests=1,
        max_total_credits=0,
    ):
        value = read_authorized_credential(
            "CRYPTO_WIZARDS_API_KEY",
            reader=lambda key: corrective_wizard_proof_scheduler._deferred_credential_reader(
                tmp_path, key
            ),
        )

    assert value == "process-key"
    assert corrective_wizard_proof_scheduler.os.environ["CRYPTO_WIZARDS_API_KEY"] == ("process-key")


def test_failed_input_audit_blocks_before_environment_or_api(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    calls = []

    def proof_runner(**kwargs):
        calls.append(kwargs)
        raise AssertionError("proof runner must not be called")

    def input_auditor(root: Path, queue_path: Path) -> CommandResult:
        path = root / "reports" / "active" / "input_audit.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"input_audit_summary": path},
            summary={"status": "BLOCKED", "ready_rows": 4, "blocked_rows": 1},
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=_queue_builder,
        proof_runner=proof_runner,
        input_auditor=input_auditor,
        checkpoint_refresher=None,
    )

    assert result.summary["status"] == "BLOCKED_INPUT_AUDIT"
    assert result.summary["external_attempt_made"] is False
    assert result.summary["input_audit_blocked"] == 1
    assert calls == []


def test_inconsistent_retry_safety_audit_blocks_before_api(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    calls = []

    def proof_runner(**kwargs):
        calls.append(kwargs)
        raise AssertionError("proof runner must not be called")

    def input_auditor(root: Path, queue_path: Path) -> CommandResult:
        path = root / "reports" / "active" / "input_audit.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"input_audit_summary": path},
            summary={
                "status": "PASS",
                "ready_rows": 5,
                "blocked_rows": 0,
                "retry_safe_rows": 4,
                "changed_after_vendor_4xx_rows": 0,
                "unchanged_vendor_4xx_rows": 1,
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=_queue_builder,
        proof_runner=proof_runner,
        input_auditor=input_auditor,
        checkpoint_refresher=None,
    )

    assert result.summary["status"] == "BLOCKED_INPUT_AUDIT"
    assert result.summary["external_attempt_made"] is False
    assert result.summary["input_audit_retry_safe"] == 4
    assert result.summary["input_audit_unchanged_vendor_4xx"] == 1
    assert calls == []


def test_failed_shared_credit_budget_blocks_before_api_call(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    calls = []

    def proof_runner(**kwargs):
        calls.append(kwargs)
        raise AssertionError("proof runner must not be called")

    def budget_builder(**kwargs):
        path = tmp_path / "reports" / "active" / "budget.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return CommandResult(
            paths={"budget_summary": path},
            summary={
                "status": "BLOCKED",
                "scheduled_credit_ceiling": 1100,
                "headroom_after_reserve": -200,
            },
        )

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=_queue_builder,
        proof_runner=proof_runner,
        input_auditor=_input_auditor,
        budget_builder=budget_builder,
        checkpoint_refresher=None,
    )

    assert result.summary["status"] == "BLOCKED_CREDIT_BUDGET_CONTRACT"
    assert result.summary["external_attempt_made"] is False
    assert calls == []


def test_orphaned_proof_reservation_blocks_crash_retry_before_api_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    _write_secure_env(tmp_path)
    orphaned = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        planned_credits=2,
        now=NOW,
    )
    assert orphaned.summary["external_spend_authorized"] is False
    proof_calls: list[dict[str, object]] = []

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW + timedelta(minutes=1),
        execute=True,
        queue_builder=_queue_builder,
        proof_runner=lambda **kwargs: proof_calls.append(kwargs),
        input_auditor=_input_auditor,
        capture_manifest_builder=_capture_manifest_builder(exact_calls=1),
        checkpoint_refresher=None,
        registered_rerun_runner=None,
        dynamic_holdout_evaluator=None,
        dynamic_supersession_gate_builder=None,
        dynamic_activation_planner=None,
        dynamic_review_packet_builder=None,
        dynamic_proof_refresher=None,
        copula_behavioral_runner=None,
    )

    assert result.summary["status"] == "BLOCKED_SHARED_CREDIT_RESERVATION"
    assert result.summary["credit_reservation_status"] == "REUSED"
    assert "insufficient_lane_reservation_remaining" in result.summary["blockers"]
    assert proof_calls == []
    assert result.summary["external_attempt_made_this_cycle"] is False


def test_launch_agent_contains_no_secret_or_order_capability(tmp_path):
    python = tmp_path / ".venv" / "bin" / "python3"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    logs = tmp_path / "reports" / "active" / "schedule_logs"
    logs.mkdir(parents=True)

    plist = _launch_agent_plist(
        root=tmp_path,
        python=python,
        logs=logs,
        interval_seconds=600,
    )

    payload = plistlib.loads(plist.encode("utf-8"))
    contract = scheduler_contract("wizard_proof")
    capability = SCHEDULER_CAPABILITY_PROFILES[contract.capability_profile]
    assert payload["ProgramArguments"] == [
        str(python),
        "-m",
        SCHEDULER_BOOTSTRAP_MODULE,
        str(tmp_path.resolve()),
        contract.key,
    ]
    assert contract.module.endswith("corrective_wizard_proof_launcher")
    assert not contract.module.endswith("corrective_wizard_proof_scheduler")
    assert contract.action == "--execute"
    assert capability.wizard_api_allowed is True
    assert capability.wizard_credit_spend_allowed is True
    assert capability.keychain_allowed is False
    assert capability.order_adapter_allowed is False
    assert capability.order_submission_allowed is False
    assert payload["StartInterval"] == 600
    assert "CRYPTO_WIZARDS_API_KEY" not in plist
    assert SECRET not in plist
    assert "testnet" not in plist.lower()
    assert "live" not in plist.lower()
    assert "order" not in plist.lower()
    environment = payload["EnvironmentVariables"]
    assert environment["TMPDIR"] == f"{tmp_path}/.runtime_tmp"
    assert environment["TMP"] == f"{tmp_path}/.runtime_tmp"
    assert environment["TEMP"] == f"{tmp_path}/.runtime_tmp"
    assert environment["TZ"] == "America/New_York"


def test_queue_completion_count_excludes_unrelated_legacy_proofs(tmp_path):
    queue_path = tmp_path / "queue.csv"
    proof_path = tmp_path / "proofs.csv"
    pd.DataFrame(
        [
            {
                "pair_group_id": "group-a",
                "pair": "ADA-ALGO",
                "local_interval": "1d",
                "exact_mode": "Copula",
                "orientation": "original",
                "proof_observations": 360,
                "vendor_custom_series_eligible": True,
            },
            {
                "pair_group_id": "group-a",
                "pair": "ALGO-ADA",
                "local_interval": "1d",
                "exact_mode": "Copula",
                "orientation": "reverse",
                "proof_observations": 360,
                "vendor_custom_series_eligible": True,
            },
            {
                "pair_group_id": "group-a",
                "pair": "ADA-ALGO",
                "local_interval": "1d",
                "exact_mode": "Static (Spread)",
                "orientation": "original",
                "proof_observations": 360,
                "vendor_custom_series_eligible": True,
            },
        ]
    ).to_csv(queue_path, index=False)
    pd.DataFrame(
        [
            {
                "pair_group_id": "legacy-group",
                "pair": "BTC-EIGEN",
                "local_interval": "1h",
                "exact_mode": "Static (Spread)",
                "orientation": "original",
                "proof_observations": 360,
                "mode_proof_status": "completed",
                "vendor_formula_parity_status": "exact_reconstruction",
                "vendor_history_available": True,
                "proof_window_kind": "scanner_horizon_parity",
            },
            {
                "pair_group_id": "group-a",
                "pair": "ADA-ALGO",
                "local_interval": "1d",
                "exact_mode": "Copula",
                "orientation": "original",
                "proof_observations": 360,
                "mode_proof_status": "completed",
                "vendor_formula_parity_status": "exact_reconstruction",
                "vendor_history_available": True,
                "proof_window_kind": "scanner_horizon_parity",
            },
            {
                "pair_group_id": "group-a",
                "pair": "ADA-ALGO",
                "local_interval": "1d",
                "exact_mode": "Static (Spread)",
                "orientation": "original",
                "proof_observations": 360,
                "mode_proof_status": "completed",
                "vendor_formula_parity_status": "exact_reconstruction",
                "vendor_history_available": True,
                "proof_window_kind": "scanner_horizon_parity",
            },
        ]
    ).to_csv(proof_path, index=False)

    assert (
        count_completed_exact_mode_proofs(
            queue_path=queue_path,
            proof_path=proof_path,
        )
        == 1
    )
