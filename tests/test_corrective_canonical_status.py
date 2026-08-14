from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from quant_platform.orchestration.corrective_canonical_status import (
    build_canonical_program_status,
    materialize_scheduler_status_roles,
    publish_scheduler_status_pointers,
    rebind_checkpoint_stage4_evidence,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _scheduler_receipt(*, execute: bool, started: str, receipt_id: str) -> dict[str, object]:
    return {
        "schema_version": "thewiz.corrective_wizard_proof_scheduler.v1",
        "execution_requested": execute,
        "started_at_utc": started,
        "receipt_id": receipt_id,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _root(tmp_path: Path) -> Path:
    active = tmp_path / "reports" / "active"
    receipts = active / "wizard_proof_scheduler_receipts"
    receipts.mkdir(parents=True)
    _write_json(
        active / "corrective_plan_completion.json",
        {
            "schema_version": "thewiz.corrective_program_completion.v1",
            "generated_at_utc": "2026-08-14T14:00:00+00:00",
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    _write_json(
        active / "stage4_handoff_readiness.json",
        {
            "receipt_id": "stage4_current",
            "checked_at_utc": "2026-08-14T14:00:00+00:00",
            "source_closure_sha256": "a" * 64,
            "candidate_promotion_authority": False,
            "order_submission_included": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    _write_json(
        active / "corrective_wizard_proof_launcher_status.json",
        {
            "launcher_receipt_id": "launcher_current",
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    rows = []
    for stage in range(1, 8):
        rows.append(
            {
                "stage": stage,
                "status": "PASS" if stage == 2 else "BLOCKED",
                "evidence_progress": (
                    "stage4_handoff_receipt_id=stage4_current;"
                    f"stage4_handoff_source_closure_sha256={'a' * 64};"
                    if stage == 4
                    else ""
                ),
                "evidence_path": "reports/active/stage4_handoff_readiness.json",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    pd.DataFrame(rows).to_csv(active / "seven_stage_goal_checkpoint.csv", index=False)
    _write_json(
        receipts / "execution.json",
        _scheduler_receipt(
            execute=True,
            started="2026-08-14T13:00:00+00:00",
            receipt_id="execution_current",
        ),
    )
    _write_json(
        receipts / "observation.json",
        _scheduler_receipt(
            execute=False,
            started="2026-08-14T13:30:00+00:00",
            receipt_id="observation_current",
        ),
    )
    return tmp_path


def test_canonical_status_passes_with_current_child_bindings(tmp_path: Path) -> None:
    root = _root(tmp_path)
    result = build_canonical_program_status(root=root, now=datetime(2026, 8, 14, 14, 1, tzinfo=UTC))

    assert result.summary["status"] == "PASS_CANONICAL_CURRENT"
    assert result.summary["stage4_receipt_binding_match"] is True
    assert result.summary["scheduler_execution_pointer_valid"] is True
    assert result.summary["scheduler_observation_pointer_valid"] is True
    assert result.summary["testnet_order_authority"] is False
    assert result.paths["immutable_receipt"].is_file()


def test_canonical_status_blocks_stale_stage4_binding(tmp_path: Path) -> None:
    root = _root(tmp_path)
    stage4 = root / "reports" / "active" / "stage4_handoff_readiness.json"
    _write_json(
        stage4,
        {
            "receipt_id": "stage4_rotated",
            "checked_at_utc": "2026-08-14T14:05:00+00:00",
            "source_closure_sha256": "b" * 64,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )

    result = build_canonical_program_status(root=root)

    assert result.summary["status"] == "BLOCKED_CANONICAL_STALE"
    assert "canonical_checkpoint_stage4_receipt_stale_or_missing" in result.summary["blockers"]
    assert "canonical_completion_precedes_current_stage4_handoff" in result.summary["blockers"]


def test_canonical_status_accepts_new_heartbeat_with_same_stage4_closure(tmp_path: Path) -> None:
    root = _root(tmp_path)
    stage4 = root / "reports" / "active" / "stage4_handoff_readiness.json"
    _write_json(
        stage4,
        {
            "receipt_id": "stage4_heartbeat_rotated",
            "checked_at_utc": "2026-08-14T14:05:00+00:00",
            "source_closure_sha256": "a" * 64,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )

    result = build_canonical_program_status(root=root)

    assert result.summary["status"] == "PASS_CANONICAL_CURRENT"
    assert result.summary["stage4_exact_receipt_binding_match"] is False
    assert result.summary["stage4_closure_binding_match"] is True
    assert result.summary["stage4_receipt_binding_match"] is True


def test_stage4_rebind_updates_only_stage4_evidence_and_repairs_canonical_status(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    active = root / "reports" / "active"
    stage4 = active / "stage4_handoff_readiness.json"
    _write_json(
        stage4,
        {
            "receipt_id": "stage4_rotated",
            "checked_at_utc": "2026-08-14T14:05:00+00:00",
            "source_closure_sha256": "b" * 64,
            "candidate_promotion_authority": False,
            "order_submission_included": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    checkpoint_path = active / "seven_stage_goal_checkpoint.csv"
    before = pd.read_csv(checkpoint_path, keep_default_na=False)

    result = rebind_checkpoint_stage4_evidence(
        root=root,
        now=datetime(2026, 8, 14, 14, 6, tzinfo=UTC),
        stage4_validator=lambda **_: {"status": "PASS", "blockers": []},
    )

    after = pd.read_csv(checkpoint_path, keep_default_na=False)
    assert result.summary["status"] == "PASS_STAGE4_CHECKPOINT_REBOUND"
    assert result.summary["only_stage4_evidence_progress_changed"] is True
    assert result.paths["immutable_receipt"].is_file()
    assert before.drop(columns=["evidence_progress"]).equals(
        after.drop(columns=["evidence_progress"])
    )
    assert before.loc[before["stage"].ne(4), "evidence_progress"].equals(
        after.loc[after["stage"].ne(4), "evidence_progress"]
    )
    evidence = after.loc[after["stage"].eq(4), "evidence_progress"].iloc[0]
    assert "stage4_handoff_receipt_id=stage4_rotated" in evidence
    assert f"stage4_handoff_source_closure_sha256={'b' * 64}" in evidence
    assert "stage4_handoff_checked_at=2026-08-14T14:05:00+00:00" in evidence

    canonical = build_canonical_program_status(
        root=root,
        now=datetime(2026, 8, 14, 14, 6, tzinfo=UTC),
    )
    assert canonical.summary["status"] == "PASS_CANONICAL_CURRENT"


def test_stage4_rebind_fails_closed_without_validated_zero_authority_source(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    checkpoint_path = root / "reports" / "active" / "seven_stage_goal_checkpoint.csv"
    before = checkpoint_path.read_bytes()

    result = rebind_checkpoint_stage4_evidence(
        root=root,
        stage4_validator=lambda **_: {
            "status": "BLOCKED",
            "blockers": ["immutable_mismatch"],
        },
    )

    assert result.summary["status"] == "BLOCKED_STAGE4_CHECKPOINT_REBIND"
    assert "canonical_stage4_source_receipt_invalid" in result.summary["blockers"]
    assert checkpoint_path.read_bytes() == before


def test_stage4_rebind_requires_explicit_zero_authority_fields(tmp_path: Path) -> None:
    root = _root(tmp_path)
    stage4_path = root / "reports" / "active" / "stage4_handoff_readiness.json"
    stage4 = json.loads(stage4_path.read_text(encoding="utf-8"))
    stage4.pop("order_submission_included")
    _write_json(stage4_path, stage4)

    result = rebind_checkpoint_stage4_evidence(
        root=root,
        stage4_validator=lambda **_: {"status": "PASS", "blockers": []},
    )

    assert result.summary["status"] == "BLOCKED_STAGE4_CHECKPOINT_REBIND"
    assert "canonical_stage4_source_authority_not_strictly_zero" in result.summary["blockers"]


def test_scheduler_role_pointers_cannot_shadow_each_other(tmp_path: Path) -> None:
    root = _root(tmp_path)
    active = root / "reports" / "active"
    dry_run = _scheduler_receipt(
        execute=False,
        started="2026-08-14T14:30:00+00:00",
        receipt_id="observation_new",
    )
    source = active / "wizard_proof_scheduler_receipts" / "observation_new.json"
    _write_json(source, dry_run)
    publish_scheduler_status_pointers(root=root, payload=dry_run, receipt_path=source)
    paths = materialize_scheduler_status_roles(root=root)

    execution = json.loads(paths["execution"].read_text(encoding="utf-8"))
    observation = json.loads(paths["observation"].read_text(encoding="utf-8"))
    assert execution["receipt_id"] == "execution_current"
    assert execution["execution_requested"] is True
    assert observation["receipt_id"] == "observation_new"
    assert observation["execution_requested"] is False
