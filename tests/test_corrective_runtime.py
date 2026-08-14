from __future__ import annotations

from pathlib import Path

import pytest

from quant_platform.orchestration.corrective_runtime import (
    _replace_existing_file_in_place,
    ensure_runtime_temp_directory,
    launch_agent_runtime_environment,
    runtime_temp_directory,
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
    assert str(result["system_persistence_blocker"]).startswith(
        "system_launch_agent_install_failed:"
    )


def test_existing_allocation_replacement_verifies_bytes(tmp_path: Path) -> None:
    path = tmp_path / "agent.plist"
    path.write_bytes(b"old" + b" " * 4093)

    _replace_existing_file_in_place(path, "new-payload")

    assert path.read_text(encoding="utf-8") == "new-payload"
