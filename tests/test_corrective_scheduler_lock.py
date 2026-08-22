from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.orchestration import corrective_runtime as runtime
from quant_platform.orchestration.corrective_runtime import (
    atomic_append_text,
    atomic_copy_file,
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_parquet,
    atomic_write_text,
    create_exclusive_bytes,
    immutable_snapshot_copy,
    promote_staged_directory,
    promote_staged_file,
    write_immutable_bytes,
)
from quant_platform.orchestration.corrective_scheduler_lock import (
    PublicationLeaseError,
    _target_allowed_for_scope,
    current_publication_lease,
    governed_evidence_write_lock,
    validate_current_publication_lease,
)


def test_governed_evidence_lock_is_reentrant_in_one_process(tmp_path: Path) -> None:
    _repository_fixture(tmp_path)
    with governed_evidence_write_lock(tmp_path, blocking=True):
        first = current_publication_lease()
        with governed_evidence_write_lock(tmp_path, blocking=False):
            second = current_publication_lease()
            assert first is not None
            assert second == first


def test_governed_evidence_lock_releases_for_next_writer(tmp_path: Path) -> None:
    with governed_evidence_write_lock(tmp_path, blocking=True):
        pass
    with governed_evidence_write_lock(tmp_path, blocking=False) as path:
        assert path == tmp_path / ".runtime_locks" / "governed_evidence.lock"


def test_atomic_writer_reuses_current_lease(tmp_path: Path) -> None:
    _repository_fixture(tmp_path)
    target = tmp_path / "reports" / "active" / "status.json"

    with governed_evidence_write_lock(tmp_path, blocking=True, scope="research"):
        lease = validate_current_publication_lease(tmp_path, target=target)
        atomic_write_text(target, "{}\n", publication_scope="research")
        assert validate_current_publication_lease(tmp_path, target=target) == lease

    assert target.read_text(encoding="utf-8") == "{}\n"


def test_binary_writer_and_immutable_collision_use_same_fence(tmp_path: Path) -> None:
    _repository_fixture(tmp_path)
    target = tmp_path / "reports" / "active" / "artifact.bin"

    with governed_evidence_write_lock(tmp_path, blocking=True, scope="research"):
        atomic_write_bytes(target, b"\x00\xffpayload")
        with pytest.raises(ValueError, match="immutable artifact is writable"):
            write_immutable_bytes(target, b"\x00\xffpayload")
        target.unlink()
        write_immutable_bytes(target, b"\x00\xffpayload")
        write_immutable_bytes(target, b"\x00\xffpayload")
        with pytest.raises(ValueError, match="immutable artifact collision"):
            write_immutable_bytes(target, b"different")

    assert target.read_bytes() == b"\x00\xffpayload"


def test_staged_promotion_is_atomic_and_removes_source(tmp_path: Path) -> None:
    _repository_fixture(tmp_path)
    target = tmp_path / "reports" / "active" / "artifact.csv"
    target.parent.mkdir(parents=True)
    staged = target.parent / ".artifact.csv.test.tmp"
    staged.write_bytes(b"pair,score\nBTC-ETH,1\n")

    with governed_evidence_write_lock(tmp_path, blocking=True, scope="research"):
        promote_staged_file(staged, target)

    assert target.read_bytes() == b"pair,score\nBTC-ETH,1\n"
    assert not staged.exists()


def test_atomic_csv_append_preserves_existing_rows(tmp_path: Path) -> None:
    _repository_fixture(tmp_path)
    target = tmp_path / "reports" / "active" / "journal.csv"

    atomic_write_csv(pd.DataFrame([{"value": 1}]), target, index=False)
    atomic_write_csv(
        pd.DataFrame([{"value": 2}]),
        target,
        mode="a",
        header=False,
        index=False,
    )

    assert pd.read_csv(target)["value"].tolist() == [1, 2]


def test_atomic_text_append_preserves_order(tmp_path: Path) -> None:
    _repository_fixture(tmp_path)
    target = tmp_path / "data" / "audit.jsonl"

    atomic_append_text(target, '{"sequence":1}\n')
    atomic_append_text(target, '{"sequence":2}\n')

    assert target.read_text(encoding="utf-8").splitlines() == [
        '{"sequence":1}',
        '{"sequence":2}',
    ]


def test_write_all_completes_repeated_short_writes() -> None:
    class PartialSink:
        def __init__(self) -> None:
            self.payload = bytearray()

        def write(self, value) -> int:
            chunk = bytes(value[:3])
            self.payload.extend(chunk)
            return len(chunk)

    sink = PartialSink()
    runtime._write_all(sink, b"complete-payload")

    assert bytes(sink.payload) == b"complete-payload"


def test_write_all_rejects_zero_progress() -> None:
    class StalledSink:
        @staticmethod
        def write(_value) -> int:
            return 0

    with pytest.raises(OSError, match="invalid write progress"):
        runtime._write_all(StalledSink(), b"payload")


@pytest.mark.parametrize("fault", ["file_fsync", "replace", "open"])
def test_atomic_writer_io_fault_preserves_prior_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    target = tmp_path / "status.json"
    atomic_write_text(target, '{"generation":"old"}\n')

    if fault == "file_fsync":
        monkeypatch.setattr(
            runtime.os,
            "fsync",
            lambda _descriptor: (_ for _ in ()).throw(
                OSError(errno.ENOSPC, "injected disk full")
            ),
        )
    elif fault == "replace":
        monkeypatch.setattr(
            runtime.os,
            "replace",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError(errno.EIO, "injected rename failure")
            ),
        )
    else:
        monkeypatch.setattr(
            runtime.os,
            "open",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                PermissionError(errno.EACCES, "injected permission failure")
            ),
        )

    with pytest.raises(OSError):
        atomic_write_text(target, '{"generation":"new"}\n')

    assert target.read_text(encoding="utf-8") == '{"generation":"old"}\n'


def test_directory_fsync_fault_exposes_only_complete_new_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "status.json"
    atomic_write_text(target, '{"generation":"old"}\n')
    real_fsync = runtime.os.fsync
    calls = 0

    def fail_second_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.EIO, "injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(runtime.os, "fsync", fail_second_fsync)

    with pytest.raises(OSError, match="directory fsync failure"):
        atomic_write_text(target, '{"generation":"new"}\n')

    assert target.read_text(encoding="utf-8") == '{"generation":"new"}\n'


@pytest.mark.parametrize(
    ("killpoint", "expected_exit"),
    [
        ("before_write", 70),
        ("before_file_fsync", 71),
        ("before_replace", 72),
        ("before_directory_fsync", 73),
    ],
)
def test_atomic_writer_killpoints_expose_only_complete_generations(
    tmp_path: Path,
    killpoint: str,
    expected_exit: int,
) -> None:
    target = tmp_path / "status.json"
    old_payload = {"generation": "old", "values": list(range(200))}
    new_payload = {"generation": "new", "values": list(range(200, 400))}
    atomic_write_text(target, json.dumps(old_payload) + "\n")
    script = r"""
import json
import os
import sys
from pathlib import Path

from quant_platform.orchestration import corrective_runtime as runtime
from quant_platform.orchestration.effect_authority import (
    PHASE00_REPAIR_PROFILE,
    EffectAuthority,
    publication_authority_session,
)

target = Path(sys.argv[1])
killpoint = sys.argv[2]
payload = json.loads(sys.argv[3])
real_fsync = runtime.os.fsync

if killpoint == "before_write":
    os._exit(70)
if killpoint == "before_file_fsync":
    runtime.os.fsync = lambda _descriptor: os._exit(71)
elif killpoint == "before_replace":
    runtime.os.replace = lambda *_args, **_kwargs: os._exit(72)
elif killpoint == "before_directory_fsync":
    calls = {"count": 0}
    def injected_fsync(descriptor):
        calls["count"] += 1
        if calls["count"] == 2:
            os._exit(73)
        real_fsync(descriptor)
    runtime.os.fsync = injected_fsync

authority = EffectAuthority(
    root=target.parent,
    secret=b"killpoint-publication-authority-secret",
    issuer_id="killpoint-test",
    profile=PHASE00_REPAIR_PROFILE,
)
with publication_authority_session(
    authority=authority,
    run_id="killpoint-run",
    intended_slot_id="killpoint-slot",
    policy_version="phase00.killpoint.v1",
    source_fingerprint_sha256="a" * 64,
    runtime_fingerprint_sha256="b" * 64,
    configuration_fingerprint_sha256="c" * 64,
    allowed_scopes=frozenset({"research"}),
    allowed_target_prefixes=(target.parent,),
    max_total_bytes=1024 * 1024,
):
    runtime.atomic_write_text(target, json.dumps(payload) + "\n")
"""
    environment = {**os.environ, "PYTHONPATH": str(Path.cwd() / "src")}

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(target),
            killpoint,
            json.dumps(new_payload),
        ],
        cwd=Path.cwd(),
        env=environment,
        check=False,
    )

    assert completed.returncode == expected_exit
    observed = json.loads(target.read_text(encoding="utf-8"))
    assert observed in (old_payload, new_payload)
    atomic_write_text(target, json.dumps(new_payload) + "\n")
    assert json.loads(target.read_text(encoding="utf-8")) == new_payload


def test_atomic_immutable_copy_is_idempotent_and_rejects_changed_source(
    tmp_path: Path,
) -> None:
    _repository_fixture(tmp_path)
    source = tmp_path / "source.csv"
    target = tmp_path / "reports" / "active" / "snapshot.csv"
    source.write_bytes(b"pair,score\nBTC-ETH,1\n")

    atomic_copy_file(source, target, immutable=True)
    atomic_copy_file(source, target, immutable=True)
    source.write_bytes(b"pair,score\nBTC-ETH,2\n")

    with pytest.raises(ValueError, match="immutable artifact collision"):
        atomic_copy_file(source, target, immutable=True)
    assert target.read_bytes() == b"pair,score\nBTC-ETH,1\n"


def test_atomic_copy_rejects_symlink_source(tmp_path: Path) -> None:
    _repository_fixture(tmp_path)
    actual = tmp_path / "actual.csv"
    source = tmp_path / "source.csv"
    target = tmp_path / "reports" / "active" / "snapshot.csv"
    actual.write_bytes(b"pair,score\nBTC-ETH,1\n")
    source.symlink_to(actual)

    with pytest.raises(ValueError, match="symbolic link"):
        atomic_copy_file(source, target, immutable=True)
    assert not target.exists()


def test_semantic_snapshot_names_prevent_generic_basename_collision(
    tmp_path: Path,
) -> None:
    _repository_fixture(tmp_path)
    first = tmp_path / "first" / "manifest.json"
    second = tmp_path / "second" / "manifest.json"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b'{"stage":"first"}\n')
    second.write_bytes(b'{"stage":"second"}\n')
    snapshots = tmp_path / "reports" / "active" / "inputs"

    first_target = immutable_snapshot_copy(
        first,
        snapshots,
        artifact_name="first_manifest",
    )
    second_target = immutable_snapshot_copy(
        second,
        snapshots,
        artifact_name="second_manifest",
    )

    assert first_target.name == "first_manifest.json"
    assert second_target.name == "second_manifest.json"
    assert first_target.read_bytes() == first.read_bytes()
    assert second_target.read_bytes() == second.read_bytes()


def test_atomic_parquet_writer_round_trips(tmp_path: Path) -> None:
    _repository_fixture(tmp_path)
    target = tmp_path / "reports" / "active" / "dataset.parquet"
    frame = pd.DataFrame([{"pair": "BTC-ETH", "score": 1.25}])

    atomic_write_parquet(frame, target, index=False)

    pd.testing.assert_frame_equal(pd.read_parquet(target), frame)


def test_exclusive_create_rejects_identical_and_different_replays(
    tmp_path: Path,
) -> None:
    _repository_fixture(tmp_path)
    target = tmp_path / "reports" / "active" / "approval.json"

    create_exclusive_bytes(target, b"first")
    with pytest.raises(FileExistsError):
        create_exclusive_bytes(target, b"first")
    with pytest.raises(FileExistsError):
        create_exclusive_bytes(target, b"different")

    assert target.read_bytes() == b"first"


def test_directory_promotion_commits_complete_tree(tmp_path: Path) -> None:
    _repository_fixture(tmp_path)
    staged = tmp_path / "data" / "candidate.tmp"
    target = tmp_path / "data" / "candidate"
    (staged / "nested").mkdir(parents=True)
    (staged / "nested" / "receipt.json").write_text("{}\n", encoding="utf-8")

    promote_staged_directory(staged, target)

    assert (target / "nested" / "receipt.json").read_text(encoding="utf-8") == "{}\n"
    assert not staged.exists()


def test_nested_different_root_gets_distinct_lease_and_restores_parent(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _repository_fixture(first_root)
    _repository_fixture(second_root)

    with governed_evidence_write_lock(first_root, blocking=True):
        first = current_publication_lease()
        with governed_evidence_write_lock(second_root, blocking=True):
            second = current_publication_lease()
            assert first is not None
            assert second is not None
            assert second.root != first.root
        assert current_publication_lease() == first


def test_generation_mutation_blocks_final_publication(tmp_path: Path) -> None:
    _repository_fixture(tmp_path)
    lock_path = tmp_path / ".runtime_locks" / "governed_evidence.lock"

    with governed_evidence_write_lock(tmp_path, blocking=True):
        state = json.loads(lock_path.read_text(encoding="utf-8"))
        state["generation"] += 1
        lock_path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(PublicationLeaseError, match="generation_mismatch"):
            validate_current_publication_lease(tmp_path)


def test_publication_target_outside_root_is_denied(tmp_path: Path) -> None:
    _repository_fixture(tmp_path)
    with (
        governed_evidence_write_lock(tmp_path, blocking=True),
        pytest.raises(PublicationLeaseError, match="outside_root"),
    ):
        validate_current_publication_lease(
            tmp_path, target=tmp_path.parent / "outside.json"
        )


def test_cross_scope_reentry_is_denied(tmp_path: Path) -> None:
    _repository_fixture(tmp_path)

    with (
        governed_evidence_write_lock(tmp_path, blocking=True, scope="research"),
        pytest.raises(PublicationLeaseError, match="scope_reentry_denied"),
        governed_evidence_write_lock(
            tmp_path,
            blocking=False,
            scope="daily_research",
        ),
    ):
        pass


def test_unregistered_publication_scope_fails_closed(tmp_path: Path) -> None:
    _repository_fixture(tmp_path)
    target = tmp_path / "reports" / "active" / "unknown.json"

    with (
        pytest.raises(PublicationLeaseError, match="publication_authority_scope_denied"),
        governed_evidence_write_lock(
            tmp_path,
            blocking=True,
            scope="unregistered_scope",
        ),
    ):
        atomic_write_text(
            target,
            "{}\n",
            publication_scope="unregistered_scope",
        )

    assert not target.exists()
    assert _target_allowed_for_scope(
        tmp_path,
        target,
        "unregistered_scope",
    ) is False


def test_symlinked_lock_path_is_denied_without_touching_target(
    tmp_path: Path,
) -> None:
    _repository_fixture(tmp_path)
    lock_dir = tmp_path / ".runtime_locks"
    lock_dir.mkdir()
    redirected = tmp_path / "redirected.lock"
    redirected.write_text("unchanged", encoding="utf-8")
    (lock_dir / "governed_evidence.lock").symlink_to(redirected)

    with (
        pytest.raises(PublicationLeaseError, match="lock_open_failed"),
        governed_evidence_write_lock(tmp_path, blocking=False),
    ):
        pass

    assert redirected.read_text(encoding="utf-8") == "unchanged"


def test_replaced_lock_inode_invalidates_lease(tmp_path: Path) -> None:
    _repository_fixture(tmp_path)
    lock_path = tmp_path / ".runtime_locks" / "governed_evidence.lock"

    with governed_evidence_write_lock(tmp_path, blocking=True):
        replacement = lock_path.with_suffix(".replacement")
        replacement.write_text(lock_path.read_text(encoding="utf-8"), encoding="utf-8")
        os.replace(replacement, lock_path)
        with pytest.raises(PublicationLeaseError, match="lock_identity_mismatch"):
            validate_current_publication_lease(tmp_path)


def test_atomic_writer_denies_symlinked_parent_escape(tmp_path: Path) -> None:
    _repository_fixture(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "reports").symlink_to(outside, target_is_directory=True)
    target = tmp_path / "reports" / "active" / "escaped.json"

    with pytest.raises(PublicationLeaseError, match="outside_root"):
        atomic_write_text(target, "{}\n", publication_scope="research")

    assert not (outside / "active" / "escaped.json").exists()


def test_atomic_writer_denies_final_target_symlink(tmp_path: Path) -> None:
    _repository_fixture(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-target"
    outside.write_text("unchanged", encoding="utf-8")
    target = tmp_path / "reports" / "active" / "status.json"
    target.parent.mkdir(parents=True)
    target.symlink_to(outside)

    with pytest.raises(PublicationLeaseError, match="target_symlink_denied"):
        atomic_write_text(target, "{}\n", publication_scope="research")

    assert outside.read_text(encoding="utf-8") == "unchanged"


def _repository_fixture(root: Path) -> None:
    (root / "src" / "quant_platform").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
