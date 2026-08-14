from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from threading import Barrier, Thread

import pandas as pd
import pytest

from quant_platform.active_pipeline import CommandResult
from quant_platform.crypto_wizards_sweep import run_wizard_discovery_sweep
from quant_platform.orchestration import corrective_daily_scheduler as daily_scheduler
from quant_platform.orchestration.corrective_daily_scheduler import (
    SCHEMA_VERSION,
    _acquire_lock,
    _daily_scheduler_evidence_status,
    _file_sha256,
    _launch_agent_plist,
    _publish_daily_receipt,
    _scheduled_research_exit_code,
    build_daily_cadence_acceptance,
    refresh_corrective_checkpoint_after_scheduled_run,
    run_daily_cadence_fault_tests,
    run_scheduled_research,
)
from quant_platform.orchestration.current_wizard_hyperliquid_cadence import STAGES
from quant_platform.orchestration.current_wizard_hyperliquid_daily_runner import (
    _build_stage_semantic_evidence,
)

NOW = datetime(2026, 8, 9, 23, 0, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[1]


def test_daily_launch_agent_uses_workspace_runtime_temp(tmp_path: Path) -> None:
    python = tmp_path / ".venv312" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    logs = tmp_path / "reports" / "active" / "schedule_logs"
    logs.mkdir(parents=True)

    plist = _launch_agent_plist(
        root=tmp_path,
        python=python,
        hour=6,
        minute=15,
        logs=logs,
    )

    assert f"<key>TMPDIR</key><string>{tmp_path}/.runtime_tmp</string>" in plist
    assert f"<key>TMP</key><string>{tmp_path}/.runtime_tmp</string>" in plist
    assert f"<key>TEMP</key><string>{tmp_path}/.runtime_tmp</string>" in plist


def _valid_daily_receipt(tmp_path, *, day: int, run_status: str = "PASS"):
    run_date = f"2026-08-{day:02d}"
    run_id = f"cwdaily_test_{day:02d}_{run_status.lower()}"
    run_dir = tmp_path / "reports" / "runs" / "current_wizard_hyperliquid_daily" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    status_path = run_dir / "daily_run_status.csv"
    manifest = {
        "schema_version": "current_wizard_hyperliquid_daily_runner.v1",
        "daily_run_id": run_id,
        "as_of": f"{run_date}T06:15:00+00:00",
        "run_status": run_status,
        "execution_requested": True,
        "stages": 19,
        "stages_passed": 19 if run_status == "PASS" else 0,
        "order_submission_authority": False,
        "live_trading_authorized": False,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    frame = pd.DataFrame(
        [
            {
                "schema_version": "current_wizard_hyperliquid_daily_runner.v1",
                "daily_run_id": run_id,
                "sequence": sequence,
                "stage": stage,
                "status": "PASS" if run_status == "PASS" else "FAILED",
                "output_sha256": sha256(f"{run_id}:{sequence}".encode()).hexdigest(),
                "started_at_utc": f"{run_date}T06:{14 + sequence:02d}:00+00:00",
                "completed_at_utc": f"{run_date}T06:{15 + sequence:02d}:00+00:00",
                "execution_requested": True,
                "order_submission_authority": False,
                "live_trading_authorized": False,
            }
            for sequence, stage, *_ in STAGES
        ]
    )
    frame.to_csv(status_path, index=False)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "started_at_utc": f"{run_date}T06:15:00+00:00",
        "completed_at_utc": f"{run_date}T06:34:00+00:00",
        "execution_requested": True,
        "run_status": run_status,
        "blockers": [] if run_status == "PASS" else ["test_failure"],
        "research_board_current": run_status == "PASS",
        "stale_output_actionable": False,
        "scheduler_lock_released": True,
        "testnet_execution_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "daily_run_id": run_id,
        "daily_run_manifest": str(manifest_path.relative_to(tmp_path)),
        "daily_run_manifest_sha256": _file_sha256(manifest_path),
        "daily_run_status": str(status_path.relative_to(tmp_path)),
        "daily_run_status_sha256": _file_sha256(status_path),
        "source_evidence_immutable": True,
    }
    return _publish_daily_receipt(receipt, root=tmp_path)


def _valid_semantic_daily_receipt(
    tmp_path,
    *,
    day: int = 11,
    isolation_status: str = "PASS",
    forbidden_commands: tuple[str, ...] = (),
    external_execution_included: bool = False,
    include_isolation_manifest: bool = True,
    resolved_command: str = "python -m quant_platform.cli system-check",
):
    run_date = f"2026-08-{day:02d}"
    now = datetime(2026, 8, day, 6, 15, tzinfo=UTC)
    run_id = f"cwdaily_semantic_{day:02d}"
    run_dir = tmp_path / "reports" / "runs" / "current_wizard_hyperliquid_daily" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    sweep = run_wizard_discovery_sweep(
        root=tmp_path,
        execute=True,
        api_key="test-key",
        now=now,
        credits_fetcher=lambda **_: {"credits_used": 0, "credit_limit": 1000},
        prescanned_fetcher=lambda **_: {"pairs": []},
    )
    output_sha256 = "d" * 64
    semantic = _build_stage_semantic_evidence(
        root=tmp_path,
        run_dir=run_dir,
        run_id=run_id,
        stage="wizard_exhaustive_discovery",
        stdout=json.dumps(
            {
                "summary": sweep.summary,
                "paths": {key: str(value) for key, value in sweep.paths.items()},
            }
        ),
        output_sha256=output_sha256,
        command_started_at=now,
        command_completed_at=now.replace(minute=16),
    )
    assert semantic["status"] == "PASS"
    manifest_path = run_dir / "manifest.json"
    status_path = run_dir / "daily_run_status.csv"
    manifest = {
        "schema_version": "current_wizard_hyperliquid_daily_runner.v1",
        "daily_run_id": run_id,
        "as_of": now.isoformat(),
        "run_status": "PASS",
        "execution_requested": True,
        "stages": 19,
        "stages_passed": 19,
        "order_submission_authority": False,
        "live_trading_authorized": False,
    }
    if include_isolation_manifest:
        manifest.update(
            {
                "stage3_command_isolation_status": isolation_status,
                "stage3_forbidden_commands": list(forbidden_commands),
                "stage3_external_execution_included": external_execution_included,
            }
        )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rows = []
    for sequence, stage, *_ in STAGES:
        is_wizard = stage == "wizard_exhaustive_discovery"
        stage_output_sha256 = (
            output_sha256 if is_wizard else sha256(f"{run_id}:{sequence}".encode()).hexdigest()
        )
        if is_wizard:
            stage_semantic_path = tmp_path / semantic["evidence_path"]
        else:
            stage_dir = run_dir / "semantic_evidence" / stage
            artifact_path = stage_dir / "artifacts" / "proof.txt"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(f"{run_id}:{stage}\n", encoding="utf-8")
            binding = {
                "key": "proof",
                "source_path": str(artifact_path.relative_to(tmp_path)),
                "source_sha256": _file_sha256(artifact_path),
                "snapshot_path": str(artifact_path.relative_to(tmp_path)),
                "snapshot_sha256": _file_sha256(artifact_path),
                "size_bytes": artifact_path.stat().st_size,
            }
            stage_body = {
                "schema_version": "thewiz.daily_stage_semantic_evidence.v1",
                "daily_run_id": run_id,
                "stage": stage,
                "validated_at_utc": now.isoformat(),
                "status": "PASS",
                "blockers": [],
                "command_output_sha256": stage_output_sha256,
                "stage_summary": {"research_only": True},
                "validation_rows": 1,
                "artifact_bindings": [binding],
                "artifact_binding_set_sha256": sha256(
                    json.dumps(
                        [binding], sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode()
                ).hexdigest(),
                "research_only": True,
                "candidate_promotion_authority": False,
                "order_submission_included": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
            stage_body["semantic_receipt_sha256"] = sha256(
                json.dumps(
                    stage_body, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
            stage_semantic_path = stage_dir / "receipt.json"
            stage_semantic_path.write_text(
                json.dumps(stage_body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        rows.append(
            {
                "schema_version": "current_wizard_hyperliquid_daily_runner.v1",
                "daily_run_id": run_id,
                "sequence": sequence,
                "stage": stage,
                "status": "PASS",
                "output_sha256": stage_output_sha256,
                "semantic_validation_status": "PASS",
                "semantic_validation_blocker": "",
                "semantic_evidence_path": str(stage_semantic_path.relative_to(tmp_path)),
                "semantic_evidence_sha256": _file_sha256(stage_semantic_path),
                "resolved_command": resolved_command,
                "started_at_utc": f"{run_date}T06:{14 + sequence:02d}:00+00:00",
                "completed_at_utc": f"{run_date}T06:{15 + sequence:02d}:00+00:00",
                "execution_requested": True,
                "order_submission_authority": False,
                "live_trading_authorized": False,
            }
        )
    pd.DataFrame(rows).to_csv(status_path, index=False)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "started_at_utc": now.isoformat(),
        "completed_at_utc": f"{run_date}T06:34:00+00:00",
        "execution_requested": True,
        "run_status": "PASS",
        "blockers": [],
        "research_board_current": True,
        "stale_output_actionable": False,
        "scheduler_lock_released": True,
        "testnet_execution_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "daily_run_id": run_id,
        "daily_run_manifest": str(manifest_path.relative_to(tmp_path)),
        "daily_run_manifest_sha256": _file_sha256(manifest_path),
        "daily_run_status": str(status_path.relative_to(tmp_path)),
        "daily_run_status_sha256": _file_sha256(status_path),
        "source_evidence_immutable": True,
    }
    return _publish_daily_receipt(receipt, root=tmp_path)


def test_overlap_lock_fails_closed_and_stale_lock_recovers(tmp_path):
    lock = tmp_path / "lock"
    _acquire_lock(lock, now=NOW, timeout_seconds=60)


def test_stale_timestamp_cannot_evict_a_live_lock_owner(tmp_path):
    lock = tmp_path / "live-owner.lock"
    lock.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at_utc": "2020-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    try:
        _acquire_lock(lock, now=NOW, timeout_seconds=60)
    except FileExistsError as exc:
        assert str(exc) == "active_scheduler_lock_present"
    else:
        raise AssertionError("a live lock owner must never be evicted by age alone")

    assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()
    try:
        _acquire_lock(lock, now=NOW, timeout_seconds=60)
    except FileExistsError:
        pass
    else:
        raise AssertionError("overlapping run must be blocked")
    lock.write_text(json.dumps({"started_at_utc": "2020-01-01T00:00:00+00:00"}))
    _acquire_lock(lock, now=NOW, timeout_seconds=60)


def test_daily_scheduler_module_cli_starts_cleanly():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "quant_platform.orchestration.corrective_daily_scheduler",
            "--help",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--execute" in completed.stdout


def test_daily_scheduler_exit_policy_fails_closed():
    passing = CommandResult(
        paths={},
        summary={
            "status": "PASS",
            "research_board_current": True,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    planned = CommandResult(paths={}, summary={"status": "PLANNED"})
    failed = CommandResult(
        paths={},
        summary={"status": "FAILED", "research_board_current": False},
    )
    unsafe = CommandResult(
        paths={},
        summary={
            "status": "PASS",
            "research_board_current": True,
            "testnet_order_authority": True,
        },
    )

    assert _scheduled_research_exit_code(passing, execute=True) == 0
    assert _scheduled_research_exit_code(planned, execute=False) == 0
    assert _scheduled_research_exit_code(planned, execute=True) == 2
    assert _scheduled_research_exit_code(failed, execute=True) == 2
    assert _scheduled_research_exit_code(unsafe, execute=True) == 3


def test_planning_scheduler_cannot_revoke_active_execution_authority(tmp_path, monkeypatch):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    execution_status = active / "daily_schedule_status.csv"
    execution_authority = active / "daily_research_authority.json"
    execution_status.write_text("frozen execution status\n", encoding="utf-8")
    execution_authority.write_text('{"research_board_current": true}\n', encoding="utf-8")

    def fake_pipeline(*, root, now, **kwargs):
        run_id = "cwdaily_plan_isolated"
        run_dir = root / "reports" / "runs" / "current_wizard_hyperliquid_daily" / run_id
        run_dir.mkdir(parents=True)
        manifest = run_dir / "manifest.json"
        status = run_dir / "daily_run_status.csv"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "current_wizard_hyperliquid_daily_runner.v1",
                    "daily_run_id": run_id,
                    "as_of": now.isoformat(),
                    "run_status": "PLANNED",
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "daily_run_id": run_id,
                    "sequence": sequence,
                    "stage": stage,
                    "status": "PLANNED",
                    "execution_requested": False,
                    "order_submission_authority": False,
                    "live_trading_authorized": False,
                }
                for sequence, stage, *_ in STAGES
            ]
        ).to_csv(status, index=False)
        return CommandResult(
            paths={
                "dated_daily_run_manifest": manifest,
                "dated_daily_run_status": status,
            },
            summary={"run_status": "PLANNED", "daily_run_id": run_id},
        )

    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_daily_scheduler."
        "run_current_wizard_hyperliquid_daily_pipeline",
        fake_pipeline,
    )

    result = run_scheduled_research(
        root=tmp_path,
        execute=False,
        now=datetime(2026, 8, 10, 12, 0, tzinfo=UTC),
        minimum_free_bytes=1,
    )

    assert result.summary["status"] == "PLANNED"
    assert result.paths["daily_schedule_status"].name == "daily_schedule_plan_status.csv"
    assert result.paths["daily_authority"].name == "daily_research_plan_authority.json"
    assert execution_status.read_text(encoding="utf-8") == "frozen execution status\n"
    assert execution_authority.read_text(encoding="utf-8") == ('{"research_board_current": true}\n')


def test_long_running_cycle_validates_acceptance_at_completion_time(tmp_path, monkeypatch):
    started_at = datetime(2026, 8, 11, 10, 15, tzinfo=UTC)
    completed_at = datetime(2026, 8, 11, 10, 47, tzinfo=UTC)
    observed_validation_times = []

    def fake_pipeline(**_):
        run_id = "cwdaily_long_running_test"
        run_dir = tmp_path / "reports" / "runs" / "current_wizard_hyperliquid_daily" / run_id
        run_dir.mkdir(parents=True)
        manifest = run_dir / "manifest.json"
        status = run_dir / "daily_run_status.csv"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "current_wizard_hyperliquid_daily_runner.v1",
                    "daily_run_id": run_id,
                    "as_of": started_at.isoformat(),
                    "run_status": "PASS",
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "daily_run_id": run_id,
                    "sequence": sequence,
                    "stage": stage,
                    "status": "PASS",
                }
                for sequence, stage, *_ in STAGES
            ]
        ).to_csv(status, index=False)
        return CommandResult(
            paths={
                "dated_daily_run_manifest": manifest,
                "dated_daily_run_status": status,
            },
            summary={"run_status": "PASS", "daily_run_id": run_id},
        )

    def fake_acceptance(*, root, now):
        observed_validation_times.append(now)
        path = root / "reports" / "active" / "daily_cadence_acceptance.csv"
        pd.DataFrame([{"consecutive_complete_cycles": 3}]).to_csv(path, index=False)
        return path

    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_daily_scheduler."
        "run_current_wizard_hyperliquid_daily_pipeline",
        fake_pipeline,
    )
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_daily_scheduler.build_daily_cadence_acceptance",
        fake_acceptance,
    )

    result = run_scheduled_research(
        root=tmp_path,
        execute=True,
        now=started_at,
        minimum_free_bytes=1,
        clock=lambda: completed_at,
    )
    receipt = json.loads(result.paths["daily_receipt"].read_text(encoding="utf-8"))

    assert observed_validation_times == [completed_at]
    assert receipt["completed_at_utc"] == completed_at.isoformat()
    assert result.summary["qualifying_consecutive_daily_cycles"] == 3


def test_fault_suite_revokes_current_status(tmp_path):
    frame = run_daily_cadence_fault_tests(root=tmp_path, now=NOW)
    assert frame["status"].eq("PASS").all()
    assert not frame.loc[frame["failure_injected"], "research_board_current"].any()
    assert not frame["stale_output_actionable"].any()


def test_seven_distinct_consecutive_receipts_are_required(tmp_path):
    for day in range(11, 18):
        _valid_semantic_daily_receipt(tmp_path, day=day)
    path = build_daily_cadence_acceptance(
        root=tmp_path,
        now=datetime(2026, 8, 17, 23, 0, tzinfo=UTC),
    )
    frame = pd.read_csv(path)
    assert frame["consecutive_complete_cycles"].max() == 7
    assert frame.iloc[-1]["cadence_acceptance_status"] == "PASS"
    assert frame["receipt_valid"].all()
    assert frame["semantic_contract_status"].eq("PASS").all()


def test_legacy_receipt_remains_valid_but_cannot_qualify_without_semantics(tmp_path):
    _valid_daily_receipt(tmp_path, day=9)

    frame = pd.read_csv(build_daily_cadence_acceptance(root=tmp_path, now=NOW))

    assert bool(frame.iloc[0]["receipt_valid"])
    assert frame.iloc[0]["semantic_contract_status"] == "BLOCKED"
    assert not bool(frame.iloc[0]["qualifying_cycle"])
    assert "daily_receipt_semantic_columns_missing" in frame.iloc[0]["semantic_contract_blocker"]


def test_semantic_legacy_manifest_uses_hash_bound_command_isolation_audit(tmp_path):
    _valid_semantic_daily_receipt(
        tmp_path,
        day=10,
        include_isolation_manifest=False,
    )

    frame = pd.read_csv(
        build_daily_cadence_acceptance(
            root=tmp_path,
            now=datetime(2026, 8, 10, 23, 0, tzinfo=UTC),
        )
    )

    assert bool(frame.iloc[0]["receipt_valid"])
    assert frame.iloc[0]["semantic_contract_status"] == "PASS"
    assert frame.iloc[0]["semantic_contract_source"] == ("immutable_resolved_command_audit")
    assert bool(frame.iloc[0]["qualifying_cycle"])


def test_semantic_legacy_command_audit_rejects_stage3_proof_execution(tmp_path):
    _valid_semantic_daily_receipt(
        tmp_path,
        day=10,
        include_isolation_manifest=False,
        resolved_command=("python -m quant_platform.cli run-hyperliquid-wizard-mode-proofs"),
    )

    frame = pd.read_csv(
        build_daily_cadence_acceptance(
            root=tmp_path,
            now=datetime(2026, 8, 10, 23, 0, tzinfo=UTC),
        )
    )

    assert bool(frame.iloc[0]["receipt_valid"])
    assert frame.iloc[0]["semantic_contract_status"] == "BLOCKED"
    assert (
        "daily_receipt_stage3_forbidden_commands_present"
        in frame.iloc[0]["semantic_contract_blocker"]
    )
    assert not bool(frame.iloc[0]["qualifying_cycle"])


def test_semantic_legacy_command_audit_rejects_direct_stage3_module(tmp_path):
    _valid_semantic_daily_receipt(
        tmp_path,
        day=10,
        include_isolation_manifest=False,
        resolved_command=(
            "python -m "
            "quant_platform.orchestration.corrective_wizard_proof_scheduler --execute"
        ),
    )

    frame = pd.read_csv(
        build_daily_cadence_acceptance(
            root=tmp_path,
            now=datetime(2026, 8, 10, 23, 0, tzinfo=UTC),
        )
    )

    assert frame.iloc[0]["semantic_contract_status"] == "BLOCKED"
    assert "daily_receipt_stage3_forbidden_commands_present" in frame.iloc[0][
        "semantic_contract_blocker"
    ]
    assert not bool(frame.iloc[0]["qualifying_cycle"])


def test_daily_scheduler_status_uses_validated_receipt_evidence():
    assert (
        _daily_scheduler_evidence_status(
            install_action_status="INSTALLED_NOT_STARTED",
            valid_receipts=2,
            qualifying_receipts=2,
            consecutive_complete_cycles=2,
        )
        == "INSTALLED_WITH_QUALIFYING_RECEIPTS"
    )
    assert (
        _daily_scheduler_evidence_status(
            install_action_status="INSTALLED_NOT_STARTED",
            valid_receipts=7,
            qualifying_receipts=7,
            consecutive_complete_cycles=7,
        )
        == "INSTALLED_CADENCE_ACCEPTED"
    )
    assert (
        _daily_scheduler_evidence_status(
            install_action_status="NOT_INSTALLED_BY_THIS_RUN",
            valid_receipts=2,
            qualifying_receipts=2,
            consecutive_complete_cycles=2,
        )
        == "NOT_INSTALLED_BY_THIS_RUN"
    )


def test_august_11_receipt_requires_and_accepts_full_wizard_semantic_evidence(tmp_path):
    receipt_path = _valid_semantic_daily_receipt(tmp_path)
    frame = pd.read_csv(
        build_daily_cadence_acceptance(
            root=tmp_path,
            now=datetime(2026, 8, 11, 23, 0, tzinfo=UTC),
        )
    )

    assert receipt_path.is_file()
    assert bool(frame.iloc[0]["receipt_valid"])
    assert bool(frame.iloc[0]["qualifying_cycle"])


def test_new_daily_receipt_requires_clean_stage3_command_isolation(tmp_path):
    cases = (
        (
            {"isolation_status": "BLOCKED"},
            "daily_receipt_stage3_command_isolation_not_passed",
        ),
        (
            {"forbidden_commands": ("run-wizard-copula-proof",)},
            "daily_receipt_stage3_forbidden_commands_present",
        ),
        (
            {"external_execution_included": True},
            "daily_receipt_stage3_external_execution_included",
        ),
    )
    for index, (overrides, expected_blocker) in enumerate(cases):
        case_root = tmp_path / f"case_{index}"
        _valid_semantic_daily_receipt(case_root, **overrides)
        frame = pd.read_csv(
            build_daily_cadence_acceptance(
                root=case_root,
                now=datetime(2026, 8, 11, 23, 0, tzinfo=UTC),
            )
        )

        assert not bool(frame.iloc[0]["receipt_valid"])
        assert not bool(frame.iloc[0]["qualifying_cycle"])
        assert expected_blocker in frame.iloc[0]["validation_blocker"]


def test_mutated_wizard_raw_capture_revokes_new_daily_receipt(tmp_path):
    _valid_semantic_daily_receipt(tmp_path)
    raw_path = next((tmp_path / "data" / "raw" / "crypto_wizards" / "prescanned").glob("**/*.json"))
    raw_path.write_text(raw_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    frame = pd.read_csv(
        build_daily_cadence_acceptance(
            root=tmp_path,
            now=datetime(2026, 8, 11, 23, 0, tzinfo=UTC),
        )
    )

    assert not bool(frame.iloc[0]["receipt_valid"])
    assert not bool(frame.iloc[0]["qualifying_cycle"])
    assert "daily_receipt_wizard_raw_binding_hash_mismatch" in frame.iloc[0]["validation_blocker"]


def test_new_daily_receipt_without_semantic_columns_fails_closed(tmp_path):
    _valid_daily_receipt(tmp_path, day=11)
    frame = pd.read_csv(
        build_daily_cadence_acceptance(
            root=tmp_path,
            now=datetime(2026, 8, 11, 23, 0, tzinfo=UTC),
        )
    )

    assert not bool(frame.iloc[0]["receipt_valid"])
    assert "daily_receipt_semantic_columns_missing" in frame.iloc[0]["validation_blocker"]


def test_successful_scheduled_run_refreshes_non_order_checkpoint(tmp_path):
    calls = []
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    daily_receipt = _valid_daily_receipt(tmp_path, day=9)
    checkpoint_path = active / "seven_stage_goal_checkpoint.csv"
    checkpoint_path.write_text("stage,status\n4,BLOCKED\n", encoding="utf-8")
    (active / "registered_research_rerun_gate.json").write_text(
        json.dumps(
            {
                "status": "PASS_REGISTERED_RERUN_ACCOUNTED",
                "registered_rerun_results_accounted": True,
                "registered_rerun_conclusion_status": ("CONCLUSIVE_ZERO_REGISTERED_SURVIVORS"),
            }
        ),
        encoding="utf-8",
    )

    def refresher(**kwargs):
        calls.append(kwargs)
        return CommandResult(
            paths={"seven_stage_checkpoint": checkpoint_path},
            summary={
                "operational_acceptance_status": "BLOCKED",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        )

    result = CommandResult(
        paths={"daily_receipt": daily_receipt},
        summary={
            "status": "PASS",
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    checkpoint = refresh_corrective_checkpoint_after_scheduled_run(
        result=result,
        root=tmp_path,
        now=NOW,
        refresher=refresher,
    )

    assert checkpoint is not None
    assert len(calls) == 1
    assert calls[0]["root"] == tmp_path
    handoff = json.loads(result.paths["post_run_handoff"].read_text(encoding="utf-8"))
    assert handoff["daily_receipt_id"].startswith("dailyreceipt_")
    assert handoff["registered_rerun_results_accounted"] is True
    assert handoff["registered_rerun_conclusion_status"] == ("CONCLUSIVE_ZERO_REGISTERED_SURVIVORS")
    assert len(handoff["daily_receipt_sha256"]) == 64
    assert len(handoff["checkpoint_sha256"]) == 64
    assert handoff["testnet_order_authority"] is False
    assert handoff["live_trading_authorized"] is False


def test_mutated_receipt_or_dated_source_cannot_qualify(tmp_path):
    receipt_path = _valid_daily_receipt(tmp_path, day=9)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["research_board_current"] = False
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    frame = pd.read_csv(build_daily_cadence_acceptance(root=tmp_path, now=NOW))
    assert not bool(frame.iloc[0]["receipt_valid"])
    assert not bool(frame.iloc[0]["qualifying_cycle"])
    assert "daily_receipt_identity_hash_mismatch" in frame.iloc[0]["validation_blocker"]

    receipt_path = _valid_daily_receipt(tmp_path, day=8)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    status_path = tmp_path / receipt["daily_run_status"]
    status_path.write_text(status_path.read_text() + "\n", encoding="utf-8")
    frame = pd.read_csv(build_daily_cadence_acceptance(root=tmp_path, now=NOW))
    day_eight = frame.loc[frame["run_date"].eq("2026-08-08")].iloc[0]
    assert not bool(day_eight["receipt_valid"])
    assert "daily_receipt_status_hash_mismatch" in day_eight["validation_blocker"]


def test_copied_receipt_cannot_count_as_a_second_calendar_day(tmp_path):
    source = _valid_semantic_daily_receipt(tmp_path, day=11)
    copied = source.parent / "2026-08-12.json"
    copied.write_bytes(source.read_bytes())

    frame = pd.read_csv(
        build_daily_cadence_acceptance(
            root=tmp_path,
            now=datetime(2026, 8, 12, 23, 0, tzinfo=UTC),
        )
    )

    assert len(frame) == 1
    assert frame.iloc[0]["run_date"] == "2026-08-11"
    assert not bool(frame.iloc[0]["receipt_valid"])
    assert not bool(frame.iloc[0]["qualifying_cycle"])
    assert "daily_receipt_calendar_filename_mismatch" in frame.iloc[0]["validation_blocker"]
    assert frame["consecutive_complete_cycles"].max() == 0


def test_self_consistent_pass_receipt_with_partial_stage_run_cannot_qualify(tmp_path):
    selected = _valid_semantic_daily_receipt(tmp_path, day=11)
    receipt = json.loads(selected.read_text(encoding="utf-8"))
    status_path = tmp_path / receipt["daily_run_status"]
    status = pd.read_csv(status_path).iloc[:-1]
    status.to_csv(status_path, index=False)
    receipt["daily_run_status_sha256"] = _file_sha256(status_path)
    for field in ("receipt_id", "receipt_identity_sha256", "immutable_receipt_path"):
        receipt.pop(field, None)
    selected.unlink()
    republished = _publish_daily_receipt(receipt, root=tmp_path)

    frame = pd.read_csv(
        build_daily_cadence_acceptance(
            root=tmp_path,
            now=datetime(2026, 8, 11, 23, 0, tzinfo=UTC),
        )
    )

    assert republished.is_file()
    assert not bool(frame.iloc[0]["receipt_valid"])
    assert not bool(frame.iloc[0]["qualifying_cycle"])
    assert "daily_receipt_stage_count_mismatch" in frame.iloc[0]["validation_blocker"]


def test_self_consistent_pass_receipt_with_substituted_stage_cannot_qualify(tmp_path):
    selected = _valid_semantic_daily_receipt(tmp_path, day=11)
    receipt = json.loads(selected.read_text(encoding="utf-8"))
    status_path = tmp_path / receipt["daily_run_status"]
    status = pd.read_csv(status_path, keep_default_na=False)
    target = status.index[-1]
    semantic_path = tmp_path / status.loc[target, "semantic_evidence_path"]
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic["stage"] = "substituted_research_stage"
    semantic.pop("semantic_receipt_sha256")
    semantic["semantic_receipt_sha256"] = sha256(
        json.dumps(
            semantic, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
    status.loc[target, "stage"] = "substituted_research_stage"
    status.loc[target, "semantic_evidence_sha256"] = _file_sha256(semantic_path)
    status.to_csv(status_path, index=False)
    receipt["daily_run_status_sha256"] = _file_sha256(status_path)
    for field in ("receipt_id", "receipt_identity_sha256", "immutable_receipt_path"):
        receipt.pop(field, None)
    selected.unlink()
    _publish_daily_receipt(receipt, root=tmp_path)

    frame = pd.read_csv(
        build_daily_cadence_acceptance(
            root=tmp_path,
            now=datetime(2026, 8, 11, 23, 0, tzinfo=UTC),
        )
    )

    assert not bool(frame.iloc[0]["receipt_valid"])
    assert not bool(frame.iloc[0]["qualifying_cycle"])
    assert "daily_receipt_stage_identity_mismatch" in frame.iloc[0]["validation_blocker"]


def test_self_consistent_authority_bearing_stage_cannot_qualify(tmp_path):
    selected = _valid_semantic_daily_receipt(tmp_path, day=11)
    receipt = json.loads(selected.read_text(encoding="utf-8"))
    status_path = tmp_path / receipt["daily_run_status"]
    status = pd.read_csv(status_path, keep_default_na=False)
    status["testnet_order_authority"] = False
    status.loc[status.index[-1], "testnet_order_authority"] = True
    status.to_csv(status_path, index=False)
    receipt["daily_run_status_sha256"] = _file_sha256(status_path)
    for field in ("receipt_id", "receipt_identity_sha256", "immutable_receipt_path"):
        receipt.pop(field, None)
    selected.unlink()
    _publish_daily_receipt(receipt, root=tmp_path)

    frame = pd.read_csv(
        build_daily_cadence_acceptance(
            root=tmp_path,
            now=datetime(2026, 8, 11, 23, 0, tzinfo=UTC),
        )
    )

    assert not bool(frame.iloc[0]["receipt_valid"])
    assert not bool(frame.iloc[0]["qualifying_cycle"])
    assert "daily_receipt_stage_authority_present" in frame.iloc[0]["validation_blocker"]


def test_first_valid_pass_for_calendar_day_is_frozen(tmp_path):
    first = _valid_daily_receipt(tmp_path, day=9)
    first_payload = first.read_text(encoding="utf-8")
    second = _valid_daily_receipt(tmp_path, day=9, run_status="FAILED")
    assert second == first
    assert first.read_text(encoding="utf-8") == first_payload
    archive = tmp_path / "reports" / "active" / "daily_schedule_receipt_archive"
    assert len(list(archive.glob("*.json"))) == 2


def test_same_day_valid_pass_is_reused_without_external_pipeline(tmp_path, monkeypatch):
    receipt_path = _valid_semantic_daily_receipt(tmp_path, day=11)
    before = receipt_path.read_bytes()
    archive = tmp_path / "reports" / "active" / "daily_schedule_receipt_archive"
    archive_count = len(list(archive.glob("*.json")))

    def external_pipeline_forbidden(**_):
        raise AssertionError("a validated same-day PASS must not rerun external research")

    monkeypatch.setattr(
        daily_scheduler,
        "run_current_wizard_hyperliquid_daily_pipeline",
        external_pipeline_forbidden,
    )

    result = run_scheduled_research(
        root=tmp_path,
        execute=True,
        now=datetime(2026, 8, 11, 7, 0, tzinfo=UTC),
        minimum_free_bytes=1,
    )

    assert result.summary["status"] == "PASS"
    assert result.summary["same_day_receipt_reused"] is True
    assert result.summary["external_pipeline_invoked"] is False
    assert result.paths["daily_receipt"] == receipt_path
    assert receipt_path.read_bytes() == before
    assert len(list(archive.glob("*.json"))) == archive_count
    assert not (tmp_path / "reports" / "active" / ".corrective_daily.lock").exists()
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_daily_receipt_publish_rejects_non_iso_or_traversal_date(tmp_path):
    archive = tmp_path / "reports" / "active" / "daily_schedule_receipt_archive"

    with pytest.raises(ValueError, match="daily_receipt_run_date_invalid"):
        _publish_daily_receipt(
            {
                "schema_version": SCHEMA_VERSION,
                "run_date": "../../outside",
                "run_status": "BLOCKED",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
            root=tmp_path,
        )

    assert not archive.exists()
    assert not (tmp_path / "reports" / "outside.json").exists()


def test_concurrent_daily_publishers_freeze_exactly_one_first_valid_pass(tmp_path, monkeypatch):
    barrier = Barrier(3)
    errors = []
    results = []

    def validate_selected(path, **_):
        payload = json.loads(path.read_text()) if path.is_file() else {}
        return bool(payload.get("run_status") == "PASS"), []

    monkeypatch.setattr(daily_scheduler, "_validate_daily_receipt", validate_selected)

    def publish(marker):
        try:
            barrier.wait()
            results.append(
                daily_scheduler._publish_daily_receipt(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_date": "2026-08-13",
                        "run_status": "PASS",
                        "marker": marker,
                        "testnet_order_authority": False,
                        "live_trading_authorized": False,
                    },
                    root=tmp_path,
                )
            )
        except Exception as exc:  # noqa: BLE001 - surfaced through the parent thread
            errors.append(exc)

    threads = [Thread(target=publish, args=(marker,)) for marker in ("first", "second")]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert len(results) == 2
    assert results[0] == results[1]
    selected = json.loads(results[0].read_text())
    assert selected["marker"] in {"first", "second"}
    archive = tmp_path / "reports" / "active" / "daily_schedule_receipt_archive"
    assert len(list(archive.glob("*.json"))) == 2


def test_daily_publish_recovers_after_archive_lands_before_date_selection(tmp_path, monkeypatch):
    original_atomic_json = daily_scheduler._atomic_json
    selection_failed = False

    def fail_first_selection(payload, path):
        nonlocal selection_failed
        if path.parent.name == "daily_schedule_receipts" and not selection_failed:
            selection_failed = True
            raise OSError("simulated selection publication interruption")
        original_atomic_json(payload, path)

    monkeypatch.setattr(
        daily_scheduler,
        "_validate_daily_receipt",
        lambda *_, **__: (False, ["not_selected_yet"]),
    )
    monkeypatch.setattr(daily_scheduler, "_atomic_json", fail_first_selection)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "run_date": "2026-08-13",
        "run_status": "BLOCKED",
        "marker": "recoverable-attempt",
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }

    with pytest.raises(OSError, match="simulated selection publication interruption"):
        daily_scheduler._publish_daily_receipt(receipt, root=tmp_path)

    archive = tmp_path / "reports" / "active" / "daily_schedule_receipt_archive"
    selection = tmp_path / "reports" / "active" / "daily_schedule_receipts" / "2026-08-13.json"
    assert len(list(archive.glob("*.json"))) == 1
    assert not selection.exists()

    monkeypatch.setattr(daily_scheduler, "_atomic_json", original_atomic_json)
    recovered = daily_scheduler._publish_daily_receipt(receipt, root=tmp_path)

    assert recovered == selection
    assert json.loads(recovered.read_text())["marker"] == "recoverable-attempt"
    assert len(list(archive.glob("*.json"))) == 1


def test_failed_or_unsafe_scheduled_run_cannot_refresh_checkpoint(tmp_path):
    blocked = refresh_corrective_checkpoint_after_scheduled_run(
        result=CommandResult(paths={}, summary={"status": "BLOCKED"}),
        root=tmp_path,
        now=NOW,
        refresher=lambda **_: (_ for _ in ()).throw(AssertionError("not called")),
    )
    assert blocked is None

    try:
        refresh_corrective_checkpoint_after_scheduled_run(
            result=CommandResult(
                paths={},
                summary={"status": "PASS", "testnet_order_authority": True},
            ),
            root=tmp_path,
            now=NOW,
            refresher=lambda **_: CommandResult(paths={}, summary={}),
        )
    except ValueError as exc:
        assert "acquired authority" in str(exc)
    else:
        raise AssertionError("unsafe scheduled result must fail closed")
