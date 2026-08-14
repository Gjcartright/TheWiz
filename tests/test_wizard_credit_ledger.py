from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

import pytest

from quant_platform import wizard_credit_ledger
from quant_platform.active_pipeline import CommandResult
from quant_platform.crypto_wizards_sweep import run_wizard_discovery_sweep
from quant_platform.orchestration.corrective_wizard_proof_scheduler import (
    run_corrective_wizard_proof_cycle,
)
from quant_platform.wizard_credit_ledger import (
    DISCOVERY_LANE,
    PROOF_LANE,
    reconcile_wizard_credit_lane,
    reserve_wizard_credit_lane,
    validate_wizard_credit_lane_evidence,
)

NOW = datetime(2026, 8, 11, 0, 5, tzinfo=UTC)


def test_credit_receipt_publication_never_leaves_partial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "ledger" / "reservation.json"

    def fail_publish(_source: Path, _destination: Path) -> None:
        raise OSError("simulated publication interruption")

    monkeypatch.setattr(wizard_credit_ledger.os, "link", fail_publish)

    with pytest.raises(OSError, match="simulated publication interruption"):
        wizard_credit_ledger._write_exclusive_json(
            {"reservation_id": "test"},
            target,
        )

    assert not target.exists()
    assert list(target.parent.glob(".*.tmp")) == []


def test_shared_lanes_reserve_under_one_daily_ceiling_and_reuse_idempotently(tmp_path):
    discovery = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=DISCOVERY_LANE,
        planned_credits=300,
        now=NOW,
    )
    proof = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        planned_credits=68,
        now=NOW,
    )
    reused = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=DISCOVERY_LANE,
        planned_credits=300,
        now=NOW + timedelta(hours=1),
    )

    assert discovery.summary["status"] == "PASS"
    assert proof.summary["status"] == "PASS"
    assert proof.summary["total_reserved_credits"] == 368
    assert proof.summary["headroom_after_reservations"] == 532
    assert reused.summary["status"] == "REUSED"
    assert reused.summary["reservation_id"] == discovery.summary["reservation_id"]
    assert discovery.summary["external_spend_authorized"] is True
    assert reused.summary["external_spend_authorized"] is False
    assert reused.summary["lane_external_authority_remaining_credits"] == 0


def test_concurrent_lane_reservations_are_serialized_under_shared_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_publication_entered = Event()
    release_first_publication = Event()
    original_write = wizard_credit_ledger._write_exclusive_json

    def delayed_first_publication(payload: dict[str, object], path: Path) -> None:
        if payload.get("lane") == DISCOVERY_LANE:
            first_publication_entered.set()
            assert release_first_publication.wait(timeout=5)
        original_write(payload, path)

    monkeypatch.setattr(
        wizard_credit_ledger,
        "_write_exclusive_json",
        delayed_first_publication,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        discovery_future = pool.submit(
            reserve_wizard_credit_lane,
            root=tmp_path,
            lane=DISCOVERY_LANE,
            planned_credits=850,
            now=NOW,
        )
        assert first_publication_entered.wait(timeout=5)
        proof_future = pool.submit(
            reserve_wizard_credit_lane,
            root=tmp_path,
            lane=PROOF_LANE,
            planned_credits=68,
            now=NOW,
        )
        assert not proof_future.done()
        release_first_publication.set()
        discovery = discovery_future.result(timeout=5)
        proof = proof_future.result(timeout=5)

    assert discovery.summary["status"] == "PASS"
    assert proof.summary["status"] == "BLOCKED"
    assert proof.summary["blocker"] == ("combined_lane_reservations_exceed_usable_daily_credits")
    reservation_files = list(
        (
            tmp_path
            / "data"
            / "research"
            / "wizard_credit_ledger"
            / NOW.date().isoformat()
            / "reservations"
        ).glob("*.json")
    )
    assert [path.name for path in reservation_files] == [f"{DISCOVERY_LANE}.json"]


def test_unreconciled_reservation_cannot_authorize_crash_retry_vendor_calls(tmp_path):
    first = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=DISCOVERY_LANE,
        planned_credits=300,
        now=NOW,
    )
    assert first.summary["external_spend_authorized"] is True
    credit_calls: list[object] = []
    sweep_calls: list[object] = []

    retry = run_wizard_discovery_sweep(
        root=tmp_path,
        execute=True,
        api_key="test-key",
        now=NOW + timedelta(minutes=1),
        credits_fetcher=lambda **kwargs: credit_calls.append(kwargs) or {"credits_used": 0},
        prescanned_fetcher=lambda **kwargs: sweep_calls.append(kwargs) or [],
    )

    assert retry.summary["credit_reservation_status"] == "REUSED"
    assert retry.summary["blocker"] == (
        "reused_credit_reservation_does_not_authorize_external_replay"
    )
    assert retry.summary["attempted_credits"] == 0
    assert credit_calls == []
    assert sweep_calls == []


def test_reconciled_zero_attempt_reservation_can_retry_without_false_lockout(tmp_path):
    reservation = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        planned_credits=68,
        now=NOW,
    )
    reconcile_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        reservation_id=reservation.summary["reservation_id"],
        reconciliation_key="credit-preflight-blocked",
        attempted_credits=0,
        completed_credits=0,
        external_requests=0,
        now=NOW,
    )

    retry = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        planned_credits=68,
        now=NOW + timedelta(minutes=1),
    )

    assert retry.summary["status"] == "REUSED"
    assert retry.summary["zero_attempt_reconciliation_retry"] is True
    assert retry.summary["external_spend_authorized"] is True
    assert retry.summary["lane_external_authority_remaining_credits"] == 68


def test_conflicting_or_combined_over_budget_reservations_fail_closed(tmp_path):
    first = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=DISCOVERY_LANE,
        planned_credits=850,
        now=NOW,
    )
    conflict = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=DISCOVERY_LANE,
        planned_credits=300,
        now=NOW,
    )
    overflow = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        planned_credits=68,
        now=NOW,
    )

    assert first.summary["status"] == "PASS"
    assert conflict.summary["status"] == "BLOCKED"
    assert conflict.summary["blocker"] == "existing_lane_reservation_contract_mismatch"
    assert overflow.summary["status"] == "BLOCKED"
    assert overflow.summary["blocker"] == ("combined_lane_reservations_exceed_usable_daily_credits")


def test_utc_reset_creates_independent_daily_reservations(tmp_path):
    first = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=DISCOVERY_LANE,
        planned_credits=300,
        now=NOW,
    )
    next_day = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=DISCOVERY_LANE,
        planned_credits=300,
        now=NOW + timedelta(days=1),
    )

    assert first.summary["status"] == "PASS"
    assert next_day.summary["status"] == "PASS"
    assert first.summary["reservation_id"] != next_day.summary["reservation_id"]
    assert first.paths["reservation"].parent.parent != next_day.paths["reservation"].parent.parent


def test_reconciliation_is_immutable_retry_safe_and_bounded(tmp_path):
    reservation = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        planned_credits=68,
        now=NOW,
    )
    kwargs = {
        "root": tmp_path,
        "lane": PROOF_LANE,
        "reservation_id": reservation.summary["reservation_id"],
        "reconciliation_key": "proof-cycle-1",
        "attempted_credits": 48,
        "completed_credits": 46,
        "external_requests": 28,
        "observed_used_before": 300,
        "now": NOW,
    }
    reconciled = reconcile_wizard_credit_lane(**kwargs)
    reused = reconcile_wizard_credit_lane(**kwargs)
    mismatch = reconcile_wizard_credit_lane(**{**kwargs, "completed_credits": 44})
    overrun = reconcile_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        reservation_id=reservation.summary["reservation_id"],
        reconciliation_key="proof-cycle-2",
        attempted_credits=21,
        completed_credits=21,
        external_requests=11,
        observed_used_before=348,
        now=NOW,
    )

    assert reconciled.summary["status"] == "PASS_RECONCILED"
    assert reused.summary["status"] == "REUSED_RECONCILIATION"
    assert reused.summary["reconciliation_id"] == reconciled.summary["reconciliation_id"]
    assert mismatch.summary["status"] == "BLOCKED"
    assert mismatch.summary["blocker"] == "existing_reconciliation_contract_mismatch"
    assert overrun.summary["status"] == "BLOCKED"
    assert overrun.summary["blocker"] == "reconciled_lane_attempts_exceed_reservation"


def test_reconciliation_rejects_impossible_zero_credit_external_requests(tmp_path):
    reservation = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        planned_credits=68,
        now=NOW,
    )

    with pytest.raises(ValueError, match="external_requests cannot exceed attempted_credits"):
        reconcile_wizard_credit_lane(
            root=tmp_path,
            lane=PROOF_LANE,
            reservation_id=reservation.summary["reservation_id"],
            reconciliation_key="impossible-zero-credit-request",
            attempted_credits=0,
            completed_credits=0,
            external_requests=1,
            now=NOW,
        )


def test_sealed_but_identity_forged_reservation_blocks_ledger(tmp_path):
    reservation = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=DISCOVERY_LANE,
        planned_credits=300,
        now=NOW,
    )
    path = reservation.paths["reservation"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("receipt_sha256")
    payload["planned_credits"] = 299
    path.write_text(
        json.dumps(wizard_credit_ledger._seal(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    blocked = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        planned_credits=68,
        now=NOW,
    )

    assert blocked.summary["status"] == "BLOCKED"
    assert blocked.summary["blocker"] == (f"malformed_credit_reservation:{DISCOVERY_LANE}.json")


def test_credit_evidence_rejects_noncanonical_or_traversal_date(tmp_path):
    for invalid in ("2026-8-11", "../../reports", "not-a-date"):
        result = validate_wizard_credit_lane_evidence(
            root=tmp_path,
            lane=DISCOVERY_LANE,
            credit_date_utc=invalid,
            reservation_id="missing",
            reconciliation_id="missing",
            reservation_path="missing",
            reconciliation_path="missing",
        )
        assert result == {"status": "BLOCKED", "blocker": "invalid_credit_date_utc"}


def test_malformed_ledger_blocks_discovery_before_any_vendor_call(tmp_path):
    malformed = (
        tmp_path
        / "data"
        / "research"
        / "wizard_credit_ledger"
        / NOW.date().isoformat()
        / "reservations"
        / f"{DISCOVERY_LANE}.json"
    )
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{}\n", encoding="utf-8")
    credit_calls: list[object] = []
    sweep_calls: list[object] = []

    result = run_wizard_discovery_sweep(
        root=tmp_path,
        execute=True,
        api_key="test-key",
        now=NOW,
        credits_fetcher=lambda **kwargs: credit_calls.append(kwargs) or {"credits_used": 0},
        prescanned_fetcher=lambda **kwargs: sweep_calls.append(kwargs) or [],
    )

    assert result.summary["credit_reservation_status"] == "BLOCKED"
    assert str(result.summary["blocker"]).startswith("shared_credit_reservation_blocked:")
    assert credit_calls == []
    assert sweep_calls == []
    assert result.summary["sweep_complete"] is False


def test_malformed_ledger_blocks_proof_scheduler_before_proof_runner(tmp_path, monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    env = tmp_path / ".env.local"
    env.write_text("CRYPTO_WIZARDS_API_KEY=test-key\n", encoding="utf-8")
    env.chmod(0o600)
    malformed = (
        tmp_path
        / "data"
        / "research"
        / "wizard_credit_ledger"
        / NOW.date().isoformat()
        / "reservations"
        / f"{PROOF_LANE}.json"
    )
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{}\n", encoding="utf-8")
    queue = tmp_path / "reports" / "active" / "queue.csv"
    queue.parent.mkdir(parents=True)
    queue.write_text("pair\n", encoding="utf-8")
    proof_calls: list[object] = []

    result = run_corrective_wizard_proof_cycle(
        root=tmp_path,
        now=NOW,
        execute=True,
        queue_builder=lambda root: CommandResult(
            paths={"queue": queue}, summary={"eligible_rows": 1}
        ),
        input_auditor=lambda root, queue_path: CommandResult(
            paths={"input_audit_summary": queue_path},
            summary={"status": "PASS", "ready_rows": 1, "blocked_rows": 0},
        ),
        proof_runner=lambda **kwargs: proof_calls.append(kwargs),
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
    assert result.summary["credit_reservation_status"] == "BLOCKED"
    assert proof_calls == []
    assert result.summary["order_submission_included"] is False
