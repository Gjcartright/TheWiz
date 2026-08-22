from __future__ import annotations

from pathlib import Path

import pytest

import quant_platform.orchestration.corrective_runtime as runtime
from quant_platform.orchestration.corrective_runtime import (
    ensure_runtime_temp_directory,
    launch_agent_runtime_environment,
    runtime_temp_directory,
    scheduler_contract,
    scheduler_launch_agent_payload,
    scheduler_run_identity,
    scheduler_runtime_contract,
    workspace_launch_agent_path,
    write_launch_agent_plist,
)


def test_runtime_temp_is_private_and_bound_to_workspace(tmp_path: Path) -> None:
    path = ensure_runtime_temp_directory(tmp_path)

    assert path == tmp_path / ".runtime_tmp"
    assert path.is_dir()
    assert path.stat().st_mode & 0o777 == 0o700
    assert launch_agent_runtime_environment(tmp_path) == {
        "PYTHONPATH": str(tmp_path / "src"),
        "TMPDIR": str(path),
        "TMP": str(path),
        "TEMP": str(path),
        "TZ": "America/New_York",
    }


def test_runtime_temp_rejects_symbolic_link(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".runtime_tmp").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        runtime_temp_directory(tmp_path)


def test_launch_agent_publication_survives_system_mirror_failure(tmp_path: Path) -> None:
    impossible_parent = tmp_path / "not-a-directory"
    impossible_parent.write_text("occupied", encoding="utf-8")

    result = write_launch_agent_plist(
        root=tmp_path,
        label="com.thewiz.test",
        payload="valid-plist-payload",
        system_path=impossible_parent / "agent.plist",
    )

    workspace_path = workspace_launch_agent_path(tmp_path, "com.thewiz.test")
    assert workspace_path.read_text(encoding="utf-8") == "valid-plist-payload"
    assert result["system_persistence_installed"] is False
    assert result["status"] == "WORKSPACE_RENDERED_SYSTEM_PERSISTENCE_BLOCKED"
    assert result["workspace_generation_status"] == "PASS"
    assert result["system_persistence_status"] == "BLOCKED"
    assert result["launchd_load_status"] == "NOT_ATTEMPTED"
    assert result["first_execution_status"] == "NOT_OBSERVED"
    assert str(result["system_persistence_blocker"]).startswith(
        "system_launch_agent_install_failed:"
    )


def test_launch_agent_enospc_preserves_existing_system_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_path = tmp_path / "LaunchAgents" / "com.thewiz.test.plist"
    system_path.parent.mkdir()
    system_path.write_text("prior-valid-plist", encoding="utf-8")
    original_write = runtime.atomic_write_text

    def faulted_write(path: Path, payload: str, **kwargs) -> None:
        if path == system_path:
            raise OSError(28, "fixture ENOSPC", str(path))
        original_write(path, payload, **kwargs)

    monkeypatch.setattr(runtime, "atomic_write_text", faulted_write)

    result = write_launch_agent_plist(
        root=tmp_path,
        label="com.thewiz.test",
        payload="replacement-plist",
        system_path=system_path,
    )

    assert system_path.read_text(encoding="utf-8") == "prior-valid-plist"
    assert result["system_persistence_installed"] is False
    assert result["system_install_mode"] == "blocked"
    assert result["system_persistence_blocker"].endswith(":OSError:28")


def test_runtime_contract_binds_source_dependency_config_and_schedule(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "quant_platform" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config = tmp_path / "config" / "policy.json"
    config.parent.mkdir()
    config.write_text("{}\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.0.0'\nrequires-python='>=3.11'\n",
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    before = scheduler_runtime_contract(
        tmp_path,
        contract=scheduler_contract("hyperliquid_l2"),
    )
    payload = scheduler_launch_agent_payload(
        tmp_path,
        contract=scheduler_contract("hyperliquid_l2"),
    )
    source.write_text("VALUE = 2\n", encoding="utf-8")
    after = scheduler_runtime_contract(
        tmp_path,
        contract=scheduler_contract("hyperliquid_l2"),
    )

    assert payload["ProgramArguments"][0] == str(
        tmp_path / ".venv" / "bin" / "python3"
    )
    assert payload["StandardOutPath"] == str(
        tmp_path / "reports" / "active" / "schedule_logs" / "l2.stdout.log"
    )
    assert payload["EnvironmentVariables"]["TZ"] == "America/New_York"
    assert payload["EnvironmentVariables"]["THEWIZ_SCHEDULER_PROFILE"] == (
        "PUBLIC_L2_ONLY"
    )
    assert payload["EnvironmentVariables"]["THEWIZ_SCHEDULER_PROFILE_SHA256"] == (
        before["capability_profile_sha256"]
    )
    assert before["capability_profile"]["order_submission_allowed"] is False
    assert before["source_fingerprint_sha256"] != after["source_fingerprint_sha256"]
    assert before["runtime_contract_sha256"] != after["runtime_contract_sha256"]


def test_runtime_contract_binds_git_head_without_invoking_git(tmp_path: Path) -> None:
    source = tmp_path / "src" / "quant_platform" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    git_directory = tmp_path / ".git"
    branch_ref = git_directory / "refs" / "heads" / "main"
    branch_ref.parent.mkdir(parents=True)
    (git_directory / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    branch_ref.write_text("1" * 40 + "\n", encoding="utf-8")

    before = scheduler_runtime_contract(
        tmp_path,
        contract=scheduler_contract("hyperliquid_l2"),
    )
    branch_ref.write_text("2" * 40 + "\n", encoding="utf-8")
    after = scheduler_runtime_contract(
        tmp_path,
        contract=scheduler_contract("hyperliquid_l2"),
    )

    assert before["repository_head"] == "1" * 40
    assert after["repository_head"] == "2" * 40
    assert before["source_tree_sha256"] == after["source_tree_sha256"]
    assert before["source_fingerprint_sha256"] != after["source_fingerprint_sha256"]
    assert before["runtime_contract_sha256"] != after["runtime_contract_sha256"]


def test_runtime_contract_binds_installed_distribution_metadata(tmp_path: Path) -> None:
    metadata = (
        tmp_path
        / ".venv"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "example-1.0.dist-info"
        / "METADATA"
    )
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Name: example\nVersion: 1.0\n", encoding="utf-8")

    before = scheduler_runtime_contract(
        tmp_path,
        contract=scheduler_contract("hyperliquid_l2"),
    )
    metadata.write_text("Name: example\nVersion: 1.1\n", encoding="utf-8")
    after = scheduler_runtime_contract(
        tmp_path,
        contract=scheduler_contract("hyperliquid_l2"),
    )

    assert (
        before["interpreter_fingerprint_sha256"]
        != after["interpreter_fingerprint_sha256"]
    )
    assert before["runtime_contract_sha256"] != after["runtime_contract_sha256"]


def test_scheduler_run_identity_fails_closed_when_launchd_is_required(
    tmp_path: Path,
) -> None:
    identity = scheduler_run_identity(
        tmp_path,
        contract=scheduler_contract("hyperliquid_l2"),
        environment={},
        require_launchd=True,
    )

    assert identity["runtime_environment_valid"] is False
    assert identity["trigger_provenance"] == "manual_or_test"
    assert "scheduler_launchd_provenance_missing" in identity[
        "runtime_environment_blockers"
    ]
    assert identity["order_submission_allowed"] is False


def test_scheduler_run_identity_rejects_profile_substitution(tmp_path: Path) -> None:
    contract = scheduler_contract("hyperliquid_l2")
    environment = launch_agent_runtime_environment(tmp_path, contract=contract)
    environment["THEWIZ_SCHEDULER_PROFILE"] = "WIZARD_EXTERNAL_RESEARCH"

    identity = scheduler_run_identity(
        tmp_path,
        contract=contract,
        environment=environment,
        require_launchd=True,
    )

    assert identity["runtime_environment_valid"] is False
    assert (
        "scheduler_runtime_environment_mismatch:THEWIZ_SCHEDULER_PROFILE"
        in identity["runtime_environment_blockers"]
    )
