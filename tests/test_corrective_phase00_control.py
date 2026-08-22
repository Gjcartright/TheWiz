from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.orchestration.corrective_evidence_store import EvidenceStore
from quant_platform.orchestration.corrective_lineage import (
    ACTIVE_ARTIFACT_REGISTRY,
    active_artifact_rows,
)
from quant_platform.orchestration.corrective_phase00_control import (
    _bounded_process_observation,
    _classify_path,
    resume_phase00_maintenance,
    seal_phase00_active_artifact_lineage,
)
from quant_platform.orchestration.corrective_phase00_control import (
    build_phase00_quiesced_checkpoint as _build_phase00_quiesced_checkpoint,
)
from quant_platform.orchestration.corrective_phase00_control import (
    start_phase00_maintenance as _start_phase00_maintenance,
)
from quant_platform.orchestration.corrective_runtime import (
    atomic_append_text,
    atomic_copy_file,
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_parquet,
    atomic_write_text,
    create_exclusive_bytes,
    create_exclusive_json,
    create_exclusive_text,
    immutable_snapshot_copy,
    promote_staged_directory,
    promote_staged_file,
    write_immutable_bytes,
    write_immutable_json,
)
from quant_platform.orchestration.corrective_scheduler_lock import (
    GovernedEvidenceLockBusy,
    GovernedEvidenceMaintenanceActive,
    current_publication_lease,
    governed_evidence_read_lock,
    governed_evidence_write_lock,
)

NOW = datetime(2026, 8, 21, 16, 0, tzinfo=UTC)


def _isolated_runtime_runner(command, **kwargs):
    if command[0] == "pgrep":
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
    if command[0] == "launchctl":
        return subprocess.CompletedProcess(command, 113, stdout="", stderr="")
    check = bool(kwargs.pop("check", False))
    return subprocess.run(command, check=check, **kwargs)


def start_phase00_maintenance(**kwargs):
    kwargs.setdefault("runner", _isolated_runtime_runner)
    return _start_phase00_maintenance(**kwargs)


def build_phase00_quiesced_checkpoint(**kwargs):
    kwargs.setdefault("runner", _isolated_runtime_runner)
    return _build_phase00_quiesced_checkpoint(**kwargs)


def test_process_observation_blocks_when_a_producer_is_present() -> None:
    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "123 python -m "
                "quant_platform.orchestration.corrective_daily_scheduler\n"
            ),
            stderr="",
        )

    observation = _bounded_process_observation(runner)

    assert observation["status"] == "BLOCKED_PRODUCERS_PRESENT"
    assert observation["producer_count"] == 1


def test_process_observation_passes_when_no_producer_is_present() -> None:
    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    observation = _bounded_process_observation(runner)

    assert observation["status"] == "PASS_NO_PRODUCERS"
    assert observation["producer_count"] == 0


def test_gitignore_is_runtime_configuration_not_unknown() -> None:
    assert _classify_path(".gitignore") == "runtime_configuration"


def test_controller_self_issues_publication_authority_without_pytest_context(
    tmp_path: Path,
) -> None:
    root = _git_fixture(tmp_path)
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = {**os.environ, "PYTHONPATH": str(source_root)}
    script = """
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from quant_platform.orchestration.corrective_phase00_control import (
    build_phase00_quiesced_checkpoint,
    start_phase00_maintenance,
)


def runner(command, **kwargs):
    if command[0] == "pgrep":
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
    if command[0] == "launchctl":
        return subprocess.CompletedProcess(command, 113, stdout="", stderr="")
    return subprocess.run(command, **kwargs)


root = Path(sys.argv[1])
now = datetime(2026, 8, 21, 16, 0, tzinfo=UTC)
started = start_phase00_maintenance(
    root=root,
    reason="subprocess authority regression",
    now=now,
    runner=runner,
)
checkpoint = build_phase00_quiesced_checkpoint(
    root=root,
    now=now,
    runner=runner,
)
print(json.dumps({
    "maintenance_id": started.summary["maintenance_id"],
    "checkpoint_id": checkpoint.summary["checkpoint_id"],
    "snapshot_stable": checkpoint.summary["snapshot_stable"],
    "journal_exists": (
        root / ".runtime_control/effect_authority.sqlite3"
    ).is_file(),
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["maintenance_id"].startswith("phase00maint_")
    assert payload["checkpoint_id"].startswith("phase00checkpoint_")
    assert payload["snapshot_stable"] is True
    assert payload["journal_exists"] is True


def test_maintenance_blocks_new_governed_writers_and_resumes_after_checkpoint(
    tmp_path: Path,
) -> None:
    root = _git_fixture(tmp_path)
    started = start_phase00_maintenance(
        root=root,
        reason="test quiesced baseline",
        now=NOW,
    )

    with (
        pytest.raises(GovernedEvidenceMaintenanceActive),
        governed_evidence_write_lock(root, blocking=False),
    ):
        pass

    checkpoint = build_phase00_quiesced_checkpoint(root=root, now=NOW)
    resumed = resume_phase00_maintenance(
        root=root,
        maintenance_id=started.summary["maintenance_id"],
        now=NOW,
    )

    assert checkpoint.summary["snapshot_stable"] is True
    assert checkpoint.summary["checkpoint_capture_complete"] is True
    assert checkpoint.summary["status"] == "PASS_QUIESCED_BASELINE"
    assert checkpoint.summary["producer_publication_lease_active"] is False
    assert checkpoint.summary["implementation_status"] == "IMPLEMENTED"
    assert checkpoint.summary["descendant_regeneration_status"] == "NOT_STARTED"
    assert checkpoint.summary["live_trading_authorized"] is False
    assert resumed.summary["status"] == "RESUMED"
    assert not started.paths["maintenance_marker"].exists()
    with governed_evidence_write_lock(root, blocking=False):
        pass


def test_checkpoint_requires_active_maintenance(tmp_path: Path) -> None:
    root = _git_fixture(tmp_path)

    with pytest.raises(RuntimeError, match="requires_active_maintenance"):
        build_phase00_quiesced_checkpoint(root=root, now=NOW)


def test_direct_atomic_publisher_is_denied_during_maintenance(
    tmp_path: Path,
) -> None:
    root = _git_fixture(tmp_path)
    start_phase00_maintenance(root=root, reason="fixture", now=NOW)
    target = root / "reports" / "active" / "direct_helper_status.json"

    with pytest.raises(GovernedEvidenceMaintenanceActive, match="maintenance_active"):
        atomic_write_text(target, "{}\n")

    assert not target.exists()


def test_every_canonical_publication_boundary_blocks_during_maintenance(
    tmp_path: Path,
) -> None:
    root = _git_fixture(tmp_path)
    fixture_dir = root / "reports" / "fixture_inputs"
    fixture_dir.mkdir(parents=True)
    target_dir = root / "reports" / "active" / "publication_boundary_test"
    target_dir.mkdir(parents=True)
    source = fixture_dir / "source.txt"
    source.write_text("source", encoding="utf-8")
    staged_file = target_dir / "staged.txt"
    staged_file.write_text("staged", encoding="utf-8")
    staged_directory = target_dir / "staged_directory"
    staged_directory.mkdir()
    (staged_directory / "payload.txt").write_text("payload", encoding="utf-8")
    snapshot_dir = target_dir / "snapshots"
    store = EvidenceStore(root=root, scope="research")
    frame = pd.DataFrame([{"value": 1}])
    writers = [
        (target_dir / "text.txt", lambda path: atomic_write_text(path, "text")),
        (target_dir / "bytes.bin", lambda path: atomic_write_bytes(path, b"bytes")),
        (target_dir / "copy.txt", lambda path: atomic_copy_file(source, path)),
        (
            snapshot_dir / "snapshot.txt",
            lambda _path: immutable_snapshot_copy(
                source,
                snapshot_dir,
                artifact_name="snapshot.txt",
            ),
        ),
        (
            target_dir / "immutable.json",
            lambda path: write_immutable_json(path, {"value": 1}),
        ),
        (
            target_dir / "immutable.bin",
            lambda path: write_immutable_bytes(path, b"immutable"),
        ),
        (
            target_dir / "exclusive.bin",
            lambda path: create_exclusive_bytes(path, b"exclusive"),
        ),
        (
            target_dir / "exclusive.txt",
            lambda path: create_exclusive_text(path, "exclusive"),
        ),
        (
            target_dir / "exclusive.json",
            lambda path: create_exclusive_json(path, {"value": 1}),
        ),
        (
            target_dir / "append.txt",
            lambda path: atomic_append_text(path, "append"),
        ),
        (
            target_dir / "promoted.txt",
            lambda path: promote_staged_file(staged_file, path),
        ),
        (
            target_dir / "promoted_directory",
            lambda path: promote_staged_directory(staged_directory, path),
        ),
        (target_dir / "frame.csv", lambda path: atomic_write_csv(frame, path)),
        (
            target_dir / "frame.parquet",
            lambda path: atomic_write_parquet(frame, path),
        ),
        (
            target_dir / "store.txt",
            lambda path: store.publish_text(path, "store"),
        ),
        (
            target_dir / "store.json",
            lambda path: store.publish_json(path, {"value": 1}),
        ),
        (
            target_dir / "store_immutable.json",
            lambda path: store.publish_immutable_json(path, {"value": 1}),
        ),
    ]
    start_phase00_maintenance(root=root, reason="fixture", now=NOW)

    for target, writer in writers:
        with pytest.raises(GovernedEvidenceMaintenanceActive):
            writer(target)
        assert not target.exists()

    assert staged_file.is_file()
    assert staged_directory.is_dir()


def test_resume_requires_matching_id_and_complete_checkpoint(tmp_path: Path) -> None:
    root = _git_fixture(tmp_path)
    started = start_phase00_maintenance(root=root, reason="fixture", now=NOW)

    with pytest.raises(RuntimeError, match="id_mismatch"):
        resume_phase00_maintenance(root=root, maintenance_id="wrong", now=NOW)
    with pytest.raises(RuntimeError, match="requires_matching_checkpoint"):
        resume_phase00_maintenance(
            root=root,
            maintenance_id=started.summary["maintenance_id"],
            now=NOW,
        )


def test_mutation_between_manifests_blocks_snapshot(tmp_path: Path) -> None:
    root = _git_fixture(tmp_path)
    started = start_phase00_maintenance(root=root, reason="fixture", now=NOW)
    source = root / "src" / "quant_platform" / "fixture.py"

    result = build_phase00_quiesced_checkpoint(
        root=root,
        now=NOW,
        between_manifests=lambda: source.write_text("VALUE = 2\n", encoding="utf-8"),
    )

    assert result.summary["snapshot_stable"] is False
    assert result.summary["checkpoint_capture_complete"] is False
    assert "quiesced_manifests_not_byte_identical" in result.summary["blockers"]
    with pytest.raises(RuntimeError, match="requires_complete_checkpoint"):
        resume_phase00_maintenance(
            root=root,
            maintenance_id=started.summary["maintenance_id"],
            now=NOW,
        )
    stable = build_phase00_quiesced_checkpoint(root=root, now=NOW)
    assert stable.summary["checkpoint_capture_complete"] is True
    resume_phase00_maintenance(
        root=root,
        maintenance_id=started.summary["maintenance_id"],
        now=NOW,
    )


def test_stable_checkpoint_with_blockers_cannot_resume(tmp_path: Path) -> None:
    root = _git_fixture(tmp_path)
    started = start_phase00_maintenance(root=root, reason="fixture", now=NOW)
    (root / "unknown.bin").write_bytes(b"unclassified")

    checkpoint = build_phase00_quiesced_checkpoint(root=root, now=NOW)

    assert checkpoint.summary["snapshot_stable"] is True
    assert checkpoint.summary["checkpoint_capture_complete"] is False
    assert checkpoint.summary["status"] == "CAPTURED_BLOCKED_PHASE00"
    with pytest.raises(RuntimeError, match="requires_complete_checkpoint"):
        resume_phase00_maintenance(
            root=root,
            maintenance_id=started.summary["maintenance_id"],
            now=NOW,
        )


def test_checkpoint_blocks_unfenced_later_scope_publication(
    tmp_path: Path,
) -> None:
    root = _git_fixture(tmp_path)
    source = root / "src" / "quant_platform" / "later_publisher.py"
    source.write_text(
        "def publish(path):\n    path.write_text('unfenced')\n",
        encoding="utf-8",
    )
    _run(["git", "-C", str(root), "add", str(source)])
    _run(["git", "-C", str(root), "commit", "-m", "add later publisher"])
    start_phase00_maintenance(root=root, reason="fixture", now=NOW)

    checkpoint = build_phase00_quiesced_checkpoint(root=root, now=NOW)

    assert checkpoint.summary["checkpoint_capture_complete"] is False
    assert any(
        blocker.startswith("ungoverned_publication_surface:")
        and "later_publisher.py" in blocker
        for blocker in checkpoint.summary["blockers"]
    )


def test_checkpoint_blocks_unreviewed_financial_effect_surface(
    tmp_path: Path,
) -> None:
    root = _git_fixture(tmp_path)
    source = root / "src" / "quant_platform" / "new_venue.py"
    source.write_text(
        "def submit(exchange, order):\n    return exchange.create_order(order)\n",
        encoding="utf-8",
    )
    _run(["git", "-C", str(root), "add", str(source)])
    _run(["git", "-C", str(root), "commit", "-m", "add unreviewed venue"])
    started = start_phase00_maintenance(root=root, reason="fixture", now=NOW)

    checkpoint = build_phase00_quiesced_checkpoint(root=root, now=NOW)
    stable_manifest = json.loads(
        checkpoint.paths["stable_manifest"].read_text(encoding="utf-8")
    )

    assert started.summary["known_unfenced_financial_effect_surfaces"]
    assert checkpoint.summary["checkpoint_capture_complete"] is False
    assert stable_manifest["financial_effect_surface_index"]
    assert any(
        blocker.startswith("unfenced_financial_effect_surface:")
        and "new_venue.py" in blocker
        and blocker.endswith(":create_order")
        for blocker in checkpoint.summary["blockers"]
    )


def test_checkpoint_blocks_staging_without_governed_promotion(
    tmp_path: Path,
) -> None:
    root = _git_fixture(tmp_path)
    source = root / "src" / "quant_platform" / "orphan_staging.py"
    source.write_text(
        "def stage(temp):\n    temp.write_text('not promoted')\n",
        encoding="utf-8",
    )
    _run(["git", "-C", str(root), "add", str(source)])
    _run(["git", "-C", str(root), "commit", "-m", "add orphan staging"])
    start_phase00_maintenance(root=root, reason="fixture", now=NOW)

    checkpoint = build_phase00_quiesced_checkpoint(root=root, now=NOW)

    assert checkpoint.summary["checkpoint_capture_complete"] is False
    assert any(
        blocker.startswith("unpaired_staging_surface:")
        and "orphan_staging.py" in blocker
        for blocker in checkpoint.summary["blockers"]
    )


def test_read_guard_does_not_mint_writer_during_maintenance(tmp_path: Path) -> None:
    root = _git_fixture(tmp_path)
    start_phase00_maintenance(root=root, reason="fixture", now=NOW)

    with governed_evidence_read_lock(root, create_if_missing=True):
        assert current_publication_lease() is None
        with (
            pytest.raises(GovernedEvidenceLockBusy),
            governed_evidence_write_lock(root, blocking=False),
        ):
            pass
    with (
        pytest.raises(GovernedEvidenceMaintenanceActive),
        governed_evidence_write_lock(root, blocking=False),
    ):
        pass


def test_checkpoint_hashes_but_never_captures_secret_content(tmp_path: Path) -> None:
    root = _git_fixture(tmp_path)
    secret = "PRIVATE_KEY=do-not-copy-this-secret"
    (root / ".env").write_text(secret + "\n", encoding="utf-8")
    started = start_phase00_maintenance(root=root, reason="fixture", now=NOW)

    checkpoint = build_phase00_quiesced_checkpoint(root=root, now=NOW)
    artifacts = [
        checkpoint.paths["checkpoint_receipt"],
        checkpoint.paths["stable_manifest"],
        checkpoint.paths["active_checkpoint"],
        checkpoint.paths["source_evidence_index"],
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in artifacts)
    index_rows = checkpoint.paths["source_evidence_index"].read_text(encoding="utf-8")

    assert secret not in combined
    assert "do-not-copy-this-secret" not in combined
    assert "sensitive_secret_metadata_only" in index_rows
    assert "content_captured" in index_rows
    resume_phase00_maintenance(
        root=root,
        maintenance_id=started.summary["maintenance_id"],
        now=NOW,
    )


def test_lineage_seal_resolves_all_74_candidates_and_revokes_on_mutation(
    tmp_path: Path,
) -> None:
    root = _git_fixture(tmp_path)
    _write_registered_active_artifacts(root)
    start_phase00_maintenance(root=root, reason="lineage seal", now=NOW)
    blocked = build_phase00_quiesced_checkpoint(root=root, now=NOW)

    assert blocked.summary["snapshot_stable"] is True
    assert blocked.summary["checkpoint_capture_complete"] is False
    assert any(
        blocker.startswith("unresolved_mutable_pointer:")
        for blocker in blocked.summary["blockers"]
    )

    sealed = seal_phase00_active_artifact_lineage(root=root, now=NOW)
    accepted = build_phase00_quiesced_checkpoint(root=root, now=NOW)

    assert sealed.summary["status"] == "PASS_FAIL_CLOSED_LINEAGE_SEAL"
    assert sealed.summary["registry_rows"] == 74
    assert sealed.summary["immutable_envelopes"] == 74
    assert sealed.summary["unresolved_rows"] == 0
    assert sealed.summary["authority_eligible_rows"] == 0
    assert accepted.summary["checkpoint_capture_complete"] is True
    assert accepted.summary["blockers"] == []

    mutable = root / "reports/active/dashboard_refresh_status.csv"
    mutable.write_text("status\nPASS\n", encoding="utf-8")
    mutated_row = next(
        row
        for row in active_artifact_rows(root)
        if row["path"] == "reports/active/dashboard_refresh_status.csv"
    )
    stale = build_phase00_quiesced_checkpoint(root=root, now=NOW)

    assert (
        mutated_row["resolution_status"]
        == "UNRESOLVED_STALE_ARTIFACT_ENVELOPE"
    )
    assert stale.summary["checkpoint_capture_complete"] is False
    assert (
        "unresolved_mutable_pointer:"
        "reports/active/dashboard_refresh_status.csv"
        in stale.summary["blockers"]
    )


def test_maintenance_drain_timeout_creates_no_false_active_marker(
    tmp_path: Path,
) -> None:
    root = _git_fixture(tmp_path)
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = {
        **os.environ,
        "PYTHONPATH": str(source_root),
    }
    script = """
import sys
import time
from pathlib import Path

from quant_platform.orchestration.corrective_scheduler_lock import (
    governed_evidence_write_lock,
)
from quant_platform.orchestration.effect_authority import (
    PHASE00_REPAIR_PROFILE,
    EffectAuthority,
    publication_authority_session,
)

root = Path(sys.argv[1])
authority = EffectAuthority(
    root=root,
    secret=b"phase00-subprocess-test-secret-32-bytes-minimum",
    issuer_id="phase00_subprocess_test_supervisor",
    profile=PHASE00_REPAIR_PROFILE,
)
with publication_authority_session(
    authority=authority,
    run_id="phase00_subprocess_lock_test",
    intended_slot_id="phase00_subprocess_test_slot",
    policy_version="phase00.pytest.v1",
    source_fingerprint_sha256="f" * 64,
    runtime_fingerprint_sha256="f" * 64,
    configuration_fingerprint_sha256="f" * 64,
    allowed_scopes=frozenset({"*"}),
    allowed_target_prefixes=(root,),
    max_total_bytes=1024 * 1024,
):
    with governed_evidence_write_lock(root, blocking=True):
        print("READY", flush=True)
        time.sleep(0.5)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(root)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "READY"
    try:
        with pytest.raises(GovernedEvidenceLockBusy, match="timeout"):
            start_phase00_maintenance(
                root=root,
                reason="must time out",
                now=NOW,
                wait_timeout_seconds=0.05,
            )
        assert not (root / ".runtime_control" / "phase00_maintenance.json").exists()
    finally:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=2)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def _git_fixture(root: Path) -> Path:
    _run(["git", "init", str(root)])
    _run(["git", "-C", str(root), "config", "user.email", "phase00@example.invalid"])
    _run(["git", "-C", str(root), "config", "user.name", "Phase 00 Test"])
    source = root / "src" / "quant_platform" / "fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config = root / "config" / "policy.json"
    config.parent.mkdir()
    config.write_text("{}\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.0.0'\nrequires-python='>=3.11'\n",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    python = root / ".venv" / "bin" / "python3"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    (root / ".venv" / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    immutable = root / "data" / "research" / "fixture" / "receipt.json"
    immutable.parent.mkdir(parents=True)
    immutable.write_text('{"status":"PASS"}\n', encoding="utf-8")
    active = root / "reports" / "active" / "fixture_record.json"
    active.parent.mkdir(parents=True)
    active.write_text(
        json.dumps(
            {
                "status": "PASS",
                "receipt_path": "data/research/fixture/receipt.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _run(["git", "-C", str(root), "add", "."])
    _run(["git", "-C", str(root), "commit", "-m", "fixture"])
    return root.resolve()


def _write_registered_active_artifacts(root: Path) -> None:
    for registration in ACTIVE_ARTIFACT_REGISTRY:
        path = root / registration.path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            path.write_text("status\nBLOCKED\n", encoding="utf-8")
        elif path.suffix == ".md":
            path.write_text("# Research-only status\n", encoding="utf-8")
        elif registration.validator == "blocked_decision_validator":
            path.write_text(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "blockers": ["phase00_fixture"],
                        "testnet_order_authority": False,
                        "live_trading_authority": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text("{}\n", encoding="utf-8")


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
