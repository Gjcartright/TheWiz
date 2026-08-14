from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from quant_platform.crypto_wizards_sweep import run_wizard_discovery_sweep
from quant_platform.orchestration import current_wizard_hyperliquid_daily_runner
from quant_platform.orchestration.current_wizard_hyperliquid_cadence import (
    MINIMUM_FREE_BYTES,
)
from quant_platform.orchestration.current_wizard_hyperliquid_daily_runner import (
    STAGE3_PROOF_COMMANDS,
    _build_stage_semantic_evidence,
    _secure_daily_child_environment,
    _validate_stage3_command_isolation,
    run_current_wizard_hyperliquid_daily_pipeline,
)


@pytest.fixture(autouse=True)
def _default_wizard_api_key(monkeypatch):
    monkeypatch.setenv("CRYPTO_WIZARDS_API_KEY", "test-wizard-key")


def _write_pair_queue(root):
    active = root / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair_group_key": "binance|daily|BTC|ETH",
                "history_request_status": "READY_TO_FETCH",
            },
            {
                "pair_group_key": "binance|daily|SOL|WLD",
                "history_request_status": "READY_TO_FETCH",
            },
            {
                "pair_group_key": "blocked|pair",
                "history_request_status": "BLOCKED",
            },
        ]
    ).to_csv(active / "current_wizard_hyperliquid_pair_history_queue.csv", index=False)


def test_daily_runner_blocks_before_any_command_when_storage_is_low(tmp_path):
    _write_pair_queue(tmp_path)
    calls = []

    def forbidden_runner(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("no command may run below the storage floor")

    result = run_current_wizard_hyperliquid_daily_pipeline(
        root=tmp_path,
        execute=True,
        now=datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
        available_disk_bytes=MINIMUM_FREE_BYTES - 1,
        command_runner=forbidden_runner,
    )
    frame = pd.read_csv(result.paths["daily_run_status"], keep_default_na=False)

    assert result.summary["run_status"] == "BLOCKED_STORAGE"
    assert len(frame) == 19
    assert frame.iloc[0]["status"] == "BLOCKED_STORAGE"
    assert frame.iloc[1:]["status"].eq("NOT_STARTED_DEPENDENCY").all()
    assert calls == []
    assert frame["order_submission_authority"].astype(str).str.lower().eq("false").all()
    assert frame["live_trading_authorized"].astype(str).str.lower().eq("false").all()


def test_planning_run_cannot_overwrite_active_execution_artifacts(tmp_path):
    _write_pair_queue(tmp_path)
    active = tmp_path / "reports" / "active"
    execution_status = active / "current_wizard_hyperliquid_daily_run_status.csv"
    execution_manifest = active / "current_wizard_hyperliquid_daily_run_manifest.json"
    execution_summary = active / "current_wizard_hyperliquid_daily_run_summary.md"
    execution_status.write_text("frozen execution status\n", encoding="utf-8")
    execution_manifest.write_text('{"frozen": true}\n', encoding="utf-8")
    execution_summary.write_text("frozen execution summary\n", encoding="utf-8")

    result = run_current_wizard_hyperliquid_daily_pipeline(
        root=tmp_path,
        execute=False,
        now=datetime(2026, 8, 10, 12, 0, tzinfo=UTC),
        available_disk_bytes=MINIMUM_FREE_BYTES + 1,
    )

    assert result.paths["daily_run_status"].name == (
        "current_wizard_hyperliquid_daily_plan_status.csv"
    )
    assert result.paths["daily_run_manifest"].name == (
        "current_wizard_hyperliquid_daily_plan_manifest.json"
    )
    assert execution_status.read_text(encoding="utf-8") == "frozen execution status\n"
    assert execution_manifest.read_text(encoding="utf-8") == '{"frozen": true}\n'
    assert execution_summary.read_text(encoding="utf-8") == "frozen execution summary\n"
    assert result.paths["dated_daily_run_status"].is_file()
    assert result.paths["dated_daily_run_manifest"].is_file()


def test_daily_runner_executes_all_research_stages_and_resolves_pair_keys(tmp_path):
    _write_pair_queue(tmp_path)
    calls = []

    def successful_runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    result = run_current_wizard_hyperliquid_daily_pipeline(
        root=tmp_path,
        execute=True,
        now=datetime(2026, 8, 8, 13, 0, tzinfo=UTC),
        available_disk_bytes=MINIMUM_FREE_BYTES + 1,
        command_runner=successful_runner,
        semantic_evidence_builder=lambda **_: {
            "status": "PASS",
            "blocker": "",
            "evidence_path": "reports/runs/test/semantic.json",
            "evidence_sha256": "a" * 64,
        },
    )
    frame = pd.read_csv(result.paths["daily_run_status"], keep_default_na=False)

    assert result.summary["run_status"] == "PASS"
    assert len(calls) == 19
    assert frame["status"].eq("PASS").all()
    history = next(command for command in calls if "materialize-current-wizard-hyperliquid-history" in command)
    keys = history[history.index("--current-pair-group-keys") + 1]
    assert keys == "binance|daily|BTC|ETH,binance|daily|SOL|WLD"
    assert "blocked|pair" not in keys
    flattened = " ".join(" ".join(command) for command in calls).lower()
    assert "submit-pair" not in flattened
    assert "sign-smoke-approval" not in flattened
    assert "live" not in flattened
    assert result.summary["stage3_command_isolation_status"] == "PASS"
    assert result.summary["stage3_forbidden_commands"] == []
    assert result.summary["stage3_external_execution_included"] is False


def test_daily_runner_loads_owner_only_wizard_key_for_launchd_child_without_leaking_it(
    tmp_path, monkeypatch
):
    _write_pair_queue(tmp_path)
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    secret = "launchd-only-wizard-secret"
    env_file = tmp_path / ".env.local"
    env_file.write_text(f"CRYPTO_WIZARDS_API_KEY={secret}\n", encoding="utf-8")
    env_file.chmod(0o600)
    child_environments = []

    def successful_runner(command, **kwargs):
        child_environments.append(dict(kwargs["env"]))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    result = run_current_wizard_hyperliquid_daily_pipeline(
        root=tmp_path,
        execute=True,
        now=datetime(2026, 8, 13, 10, 15, tzinfo=UTC),
        available_disk_bytes=MINIMUM_FREE_BYTES + 1,
        command_runner=successful_runner,
        semantic_evidence_builder=lambda **_: {
            "status": "PASS",
            "blocker": "",
            "evidence_path": "reports/runs/test/semantic.json",
            "evidence_sha256": "a" * 64,
        },
    )

    assert result.summary["run_status"] == "PASS"
    assert result.summary["wizard_api_credential_status"] == "PASS"
    assert result.summary["wizard_api_credential_source"] == ".env.local"
    assert child_environments
    assert all(env["CRYPTO_WIZARDS_API_KEY"] == secret for env in child_environments)
    published = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result.paths.values()
        if path.suffix in {".csv", ".json", ".md"}
    )
    assert secret not in published


def test_daily_runner_blocks_insecure_wizard_key_file_before_wizard_network_call(
    tmp_path, monkeypatch
):
    _write_pair_queue(tmp_path)
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text("CRYPTO_WIZARDS_API_KEY=unsafe-secret\n", encoding="utf-8")
    env_file.chmod(0o644)
    calls = []

    def successful_runner(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    result = run_current_wizard_hyperliquid_daily_pipeline(
        root=tmp_path,
        execute=True,
        now=datetime(2026, 8, 13, 10, 15, tzinfo=UTC),
        available_disk_bytes=MINIMUM_FREE_BYTES + 1,
        command_runner=successful_runner,
        semantic_evidence_builder=lambda **_: {
            "status": "PASS",
            "blocker": "",
            "evidence_path": "reports/runs/test/semantic.json",
            "evidence_sha256": "a" * 64,
        },
    )
    frame = pd.read_csv(result.paths["daily_run_status"], keep_default_na=False)

    assert len(calls) == 1
    assert "system-check" in calls[0][0]
    assert "CRYPTO_WIZARDS_API_KEY" not in calls[0][1]
    assert frame.iloc[1]["stage"] == "wizard_exhaustive_discovery"
    assert frame.iloc[1]["status"] == "BLOCKED_INPUT"
    assert frame.iloc[1]["execution_mode"] == "BLOCKED_CREDENTIAL"
    assert "insecure_wizard_api_key_file:.env.local" in frame.iloc[1]["blocker"]
    assert result.summary["wizard_api_credential_status"] == "BLOCKED"
    assert result.summary["wizard_api_credential_insecure_files"] == [".env.local"]


def test_daily_child_environment_prefers_process_key_over_secure_local_file(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CRYPTO_WIZARDS_API_KEY", "process-key")
    env_file = tmp_path / ".env.local"
    env_file.write_text("CRYPTO_WIZARDS_API_KEY=file-key\n", encoding="utf-8")
    env_file.chmod(0o600)

    env, credential = _secure_daily_child_environment(tmp_path)

    assert env["CRYPTO_WIZARDS_API_KEY"] == "process-key"
    assert credential == {
        "status": "PASS",
        "source": "process_environment",
        "blocker": "",
        "insecure_files": [],
    }


def test_daily_runner_stage3_command_isolation_fails_closed(tmp_path):
    def hostile_resolver(root, stage):
        del root
        command = ["python", "-m", "quant_platform.cli", "system-check"]
        if stage == "wizard_exhaustive_discovery":
            command[-1] = "run-hyperliquid-wizard-mode-proofs"
        return command, ""

    result = _validate_stage3_command_isolation(
        tmp_path,
        stage_command_resolver=hostile_resolver,
    )

    assert result == {
        "status": "BLOCKED",
        "forbidden_commands": ["run-hyperliquid-wizard-mode-proofs"],
        "stage3_external_execution_included": True,
    }


@pytest.mark.parametrize(
    "command",
    sorted(
        {
            "register-wizard-ou-v4-holdout",
            "run-wizard-ou-v4-holdout",
            "review-wizard-ou-v4",
            "register-wizard-ou-v5-holdout",
            "run-wizard-ou-v5-holdout",
            "review-wizard-ou-v5",
            "build-wizard-reset-readiness",
            "build-wizard-comparator-review-control",
        }
    ),
)
def test_daily_runner_isolates_all_current_ou_proof_commands(tmp_path, command):
    assert command in STAGE3_PROOF_COMMANDS

    def hostile_resolver(root, stage):
        del root
        selected = command if stage == "wizard_exhaustive_discovery" else "system-check"
        return ["python", "-m", "quant_platform.cli", selected], ""

    result = _validate_stage3_command_isolation(
        tmp_path,
        stage_command_resolver=hostile_resolver,
    )

    assert result["status"] == "BLOCKED"
    assert result["forbidden_commands"] == [command]


@pytest.mark.parametrize(
    "direct_target",
    (
        "quant_platform.orchestration.corrective_wizard_proof_scheduler",
        "quant_platform.orchestration.corrective_wizard_ou_v5_holdout",
        "/workspace/src/quant_platform/orchestration/corrective_wizard_proof_launcher.py",
    ),
)
def test_daily_runner_blocks_direct_stage3_module_or_script(tmp_path, direct_target):
    def hostile_resolver(root, stage):
        del root
        target = direct_target if stage == "wizard_exhaustive_discovery" else "system-check"
        return ["python", "-m", target], ""

    result = _validate_stage3_command_isolation(
        tmp_path,
        stage_command_resolver=hostile_resolver,
    )

    assert result["status"] == "BLOCKED"
    assert result["forbidden_commands"] == [direct_target]


def test_daily_pipeline_rejects_stage3_command_before_writes_or_execution(
    tmp_path, monkeypatch
):
    calls = []

    def hostile_resolver(root, stage):
        del root
        command = ["python", "-m", "quant_platform.cli", "system-check"]
        if stage == "wizard_exhaustive_discovery":
            command[-1] = "run-wizard-copula-proof"
        return command, ""

    monkeypatch.setattr(
        current_wizard_hyperliquid_daily_runner,
        "_stage_command",
        hostile_resolver,
    )

    with pytest.raises(
        RuntimeError,
        match="daily_stage3_command_isolation_failed:run-wizard-copula-proof",
    ):
        run_current_wizard_hyperliquid_daily_pipeline(
            root=tmp_path,
            execute=True,
            now=datetime(2026, 8, 11, 6, 15, tzinfo=UTC),
            available_disk_bytes=MINIMUM_FREE_BYTES + 1,
            command_runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        )

    assert calls == []
    assert not (tmp_path / "reports").exists()


def test_zero_exit_not_required_semantics_fail_closed_at_first_stage(tmp_path):
    _write_pair_queue(tmp_path)
    calls = []

    def successful_runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    result = run_current_wizard_hyperliquid_daily_pipeline(
        root=tmp_path,
        execute=True,
        now=datetime(2026, 8, 11, 6, 15, tzinfo=UTC),
        available_disk_bytes=MINIMUM_FREE_BYTES + 1,
        command_runner=successful_runner,
        semantic_evidence_builder=lambda **_: {
            "status": "NOT_REQUIRED",
            "blocker": "",
            "evidence_path": "",
            "evidence_sha256": "",
        },
    )
    frame = pd.read_csv(result.paths["daily_run_status"], keep_default_na=False)

    assert result.summary["run_status"] == "FAILED"
    assert result.summary["stages_blocked_or_not_started"] == 19
    assert len(calls) == 1
    assert frame.iloc[0]["stage"] == "storage_preflight"
    assert frame.iloc[0]["status"] == "FAILED_SEMANTIC"
    assert frame.iloc[0]["semantic_validation_status"] == "NOT_REQUIRED"
    assert frame.iloc[1:]["status"].eq("NOT_STARTED_DEPENDENCY").all()


def test_daily_runner_default_is_plan_only(tmp_path):
    _write_pair_queue(tmp_path)

    result = run_current_wizard_hyperliquid_daily_pipeline(
        root=tmp_path,
        now=datetime(2026, 8, 8, 14, 0, tzinfo=UTC),
        available_disk_bytes=MINIMUM_FREE_BYTES + 1,
    )
    frame = pd.read_csv(result.paths["daily_run_status"], keep_default_na=False)

    assert result.summary["run_status"] == "PLANNED"
    assert result.summary["execution_requested"] is False
    assert frame["status"].eq("PLANNED").all()


def test_zero_exit_with_blocked_wizard_semantics_halts_daily_run(tmp_path):
    _write_pair_queue(tmp_path)
    calls = []

    def successful_runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    def semantics(*, stage, **kwargs):
        if stage == "wizard_exhaustive_discovery":
            return {
                "status": "BLOCKED",
                "blocker": "wizard_sweep_not_complete",
                "evidence_path": "reports/runs/test/semantic.json",
                "evidence_sha256": "b" * 64,
            }
        return {
            "status": "PASS",
            "blocker": "",
            "evidence_path": "reports/runs/test/semantic.json",
            "evidence_sha256": "a" * 64,
        }

    result = run_current_wizard_hyperliquid_daily_pipeline(
        root=tmp_path,
        execute=True,
        now=datetime(2026, 8, 11, 6, 15, tzinfo=UTC),
        available_disk_bytes=MINIMUM_FREE_BYTES + 1,
        command_runner=successful_runner,
        semantic_evidence_builder=semantics,
    )
    frame = pd.read_csv(result.paths["daily_run_status"], keep_default_na=False)

    assert result.summary["run_status"] == "FAILED"
    assert len(calls) == 2
    wizard = frame.loc[frame["stage"].eq("wizard_exhaustive_discovery")].iloc[0]
    assert wizard["status"] == "FAILED_SEMANTIC"
    assert wizard["semantic_validation_status"] == "BLOCKED"
    assert "wizard_sweep_not_complete" in wizard["blocker"]
    assert frame.iloc[2:]["status"].eq("NOT_STARTED_DEPENDENCY").all()


def test_wizard_semantic_receipt_binds_full_sweep_raw_and_credit_evidence(tmp_path):
    now = datetime(2026, 8, 11, 6, 15, tzinfo=UTC)
    sweep = run_wizard_discovery_sweep(
        root=tmp_path,
        execute=True,
        api_key="test-key",
        now=now,
        credits_fetcher=lambda **_: {"credits_used": 0, "credit_limit": 1000},
        prescanned_fetcher=lambda **_: {"pairs": []},
    )
    stdout = json.dumps(
        {
            "summary": sweep.summary,
            "paths": {key: str(value) for key, value in sweep.paths.items()},
        }
    )
    run_dir = tmp_path / "reports" / "runs" / "current_wizard_hyperliquid_daily" / "run"
    run_dir.mkdir(parents=True)
    result = _build_stage_semantic_evidence(
        root=tmp_path,
        run_dir=run_dir,
        run_id="run",
        stage="wizard_exhaustive_discovery",
        stdout=stdout,
        output_sha256="c" * 64,
        command_started_at=now,
        command_completed_at=now.replace(minute=16),
    )

    assert result["status"] == "PASS"
    receipt = json.loads((tmp_path / result["evidence_path"]).read_text(encoding="utf-8"))
    assert receipt["raw_snapshot_count"] == 30
    assert receipt["credit_evidence"]["status"] == "PASS"
    assert receipt["credit_evidence"]["attempted_credits"] == 300
    assert receipt["testnet_order_authority"] is False
    assert receipt["live_trading_authorized"] is False


def test_daily_runner_reuses_complete_same_day_wizard_sweep_without_api_call(
    tmp_path,
):
    now = datetime(2026, 8, 11, 6, 15, tzinfo=UTC)
    _write_pair_queue(tmp_path)
    run_wizard_discovery_sweep(
        root=tmp_path,
        execute=True,
        api_key="test-key",
        now=now.replace(hour=1),
        credits_fetcher=lambda **_: {"credits_used": 0, "credit_limit": 1000},
        prescanned_fetcher=lambda **_: {"pairs": []},
    )
    calls = []

    def successful_runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    def semantics(**kwargs):
        if kwargs["stage"] == "wizard_exhaustive_discovery":
            return _build_stage_semantic_evidence(**kwargs)
        return {
            "status": "PASS",
            "blocker": "",
            "evidence_path": "reports/runs/test/semantic.json",
            "evidence_sha256": "a" * 64,
        }

    result = run_current_wizard_hyperliquid_daily_pipeline(
        root=tmp_path,
        execute=True,
        now=now,
        available_disk_bytes=MINIMUM_FREE_BYTES + 1,
        command_runner=successful_runner,
        semantic_evidence_builder=semantics,
    )
    frame = pd.read_csv(result.paths["daily_run_status"], keep_default_na=False)
    wizard = frame.loc[frame["stage"].eq("wizard_exhaustive_discovery")].iloc[0]

    assert result.summary["run_status"] == "PASS"
    assert wizard["status"] == "PASS"
    assert wizard["execution_mode"] == "REUSED_SAME_DAY"
    assert not any("crypto-wizards-full-sweep" in command for command in calls)
    receipt = json.loads((tmp_path / wizard["semantic_evidence_path"]).read_text())
    assert receipt["same_day_sweep_reused"] is True
    assert receipt["raw_snapshot_count"] == 30


def test_daily_runner_does_not_reuse_prior_day_wizard_sweep(tmp_path):
    now = datetime(2026, 8, 12, 6, 15, tzinfo=UTC)
    _write_pair_queue(tmp_path)
    run_wizard_discovery_sweep(
        root=tmp_path,
        execute=True,
        api_key="test-key",
        now=now.replace(day=11, hour=1),
        credits_fetcher=lambda **_: {"credits_used": 0, "credit_limit": 1000},
        prescanned_fetcher=lambda **_: {"pairs": []},
    )
    calls = []

    def successful_runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    run_current_wizard_hyperliquid_daily_pipeline(
        root=tmp_path,
        execute=True,
        now=now,
        available_disk_bytes=MINIMUM_FREE_BYTES + 1,
        command_runner=successful_runner,
        semantic_evidence_builder=lambda **_: {
            "status": "PASS",
            "blocker": "",
            "evidence_path": "reports/runs/test/semantic.json",
            "evidence_sha256": "a" * 64,
        },
    )

    assert any("crypto-wizards-full-sweep" in command for command in calls)


def _build_generic_semantics(
    tmp_path,
    *,
    stage: str,
    summary: dict[str, object],
    paths: dict[str, object],
    started: datetime,
    completed: datetime,
):
    run_dir = tmp_path / "reports" / "runs" / "current_wizard_hyperliquid_daily" / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return _build_stage_semantic_evidence(
        root=tmp_path,
        run_dir=run_dir,
        run_id="run",
        stage=stage,
        stdout=json.dumps({"summary": summary, "paths": paths}),
        output_sha256="d" * 64,
        command_started_at=started,
        command_completed_at=completed,
    )


def _write_snapshot_contract(tmp_path, *, validation_status: str = "PASS"):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    validation = active / "snapshot_validation.csv"
    manifest = active / "snapshot_manifest.json"
    output = active / "snapshot_output.csv"
    pd.DataFrame([{"check": "complete", "status": validation_status}]).to_csv(
        validation, index=False
    )
    manifest.write_text(json.dumps({"schema_version": "test.v1"}), encoding="utf-8")
    pd.DataFrame([{"pair": "BTC-ETH"}]).to_csv(output, index=False)
    return {
        "snapshot_validation": str(validation),
        "snapshot_manifest": str(manifest),
        "snapshot_output": str(output),
    }


def test_snapshot_stage_semantics_seal_fresh_passing_artifacts(tmp_path):
    started = datetime.now(UTC) - timedelta(seconds=1)
    paths = _write_snapshot_contract(tmp_path)
    completed = datetime.now(UTC) + timedelta(seconds=1)

    result = _build_generic_semantics(
        tmp_path,
        stage="pair_mode_orientation_handoff",
        summary={"generated_at_utc": datetime.now(UTC).isoformat(), "rows": 1},
        paths=paths,
        started=started,
        completed=completed,
    )

    assert result["status"] == "PASS"
    receipt = json.loads((tmp_path / result["evidence_path"]).read_text(encoding="utf-8"))
    assert receipt["validation_rows"] == 1
    assert len(receipt["artifact_bindings"]) == 3
    assert all(
        (tmp_path / binding["snapshot_path"]).is_file()
        for binding in receipt["artifact_bindings"]
    )


def test_snapshot_stage_semantics_block_zero_exit_with_nonpassing_validation(tmp_path):
    started = datetime.now(UTC) - timedelta(seconds=1)
    paths = _write_snapshot_contract(tmp_path, validation_status="BLOCKED")
    completed = datetime.now(UTC) + timedelta(seconds=1)

    result = _build_generic_semantics(
        tmp_path,
        stage="pair_mode_orientation_handoff",
        summary={"generated_at_utc": datetime.now(UTC).isoformat()},
        paths=paths,
        started=started,
        completed=completed,
    )

    assert result["status"] == "BLOCKED"
    assert "snapshot_validation_has_nonpassing_rows" in result["blocker"]


def test_snapshot_stage_semantics_block_stale_reused_contract(tmp_path):
    paths = _write_snapshot_contract(tmp_path)
    stale_epoch = 1_600_000_000
    for path in paths.values():
        os.utime(path, (stale_epoch, stale_epoch))
    started = datetime.now(UTC) - timedelta(seconds=1)
    completed = datetime.now(UTC) + timedelta(seconds=1)

    result = _build_generic_semantics(
        tmp_path,
        stage="pair_mode_orientation_handoff",
        summary={"rows": 1},
        paths=paths,
        started=started,
        completed=completed,
    )

    assert result["status"] == "BLOCKED"
    assert "snapshot_contract_artifacts_not_fresh" in result["blocker"]


def test_snapshot_stage_semantics_block_embedded_execution_authority(tmp_path):
    started = datetime.now(UTC) - timedelta(seconds=1)
    paths = _write_snapshot_contract(tmp_path)
    completed = datetime.now(UTC) + timedelta(seconds=1)

    result = _build_generic_semantics(
        tmp_path,
        stage="pair_mode_orientation_handoff",
        summary={
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "testnet_order_authority": True,
        },
        paths=paths,
        started=started,
        completed=completed,
    )

    assert result["status"] == "BLOCKED"
    assert "stage_summary_contains_execution_authority" in result["blocker"]


def test_inventory_semantics_require_fresh_complete_tradable_inventory(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    inventory = active / "hyperliquid_testnet_market_inventory.csv"
    now = datetime.now(UTC)
    pd.DataFrame(
        [{"symbol": "BTC", "tradable_perp": True, "checked_at_utc": now.isoformat()}]
    ).to_csv(inventory, index=False)

    result = _build_generic_semantics(
        tmp_path,
        stage="hyperliquid_market_inventory",
        summary={
            "rows": 1,
            "tradable_perps": 1,
            "fetch_blocked_rows": 0,
            "checked_at_max_utc": now.isoformat(),
        },
        paths={"inventory": str(inventory)},
        started=now - timedelta(seconds=1),
        completed=now + timedelta(seconds=1),
    )

    assert result["status"] == "PASS"


def test_storage_semantics_allow_accounted_research_blockers(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    system_check = active / "system_check.csv"
    pd.DataFrame(
        [
            {"check": "env", "status": "READY"},
            {"check": "data", "status": "READY"},
            {"check": "live", "status": "BLOCKED"},
        ]
    ).to_csv(system_check, index=False)
    now = datetime.now(UTC)

    result = _build_generic_semantics(
        tmp_path,
        stage="storage_preflight",
        summary={"checks": 3, "ready": 2, "blocked": 1},
        paths={"system_check": str(system_check)},
        started=now - timedelta(seconds=1),
        completed=now + timedelta(seconds=1),
    )

    assert result["status"] == "PASS"


def test_unregistered_daily_stage_semantics_fail_closed(tmp_path):
    now = datetime.now(UTC)
    result = _build_generic_semantics(
        tmp_path,
        stage="unregistered_stage",
        summary={},
        paths={},
        started=now - timedelta(seconds=1),
        completed=now + timedelta(seconds=1),
    )

    assert result["status"] == "BLOCKED"
    assert "stage_semantic_contract_not_registered" in result["blocker"]
