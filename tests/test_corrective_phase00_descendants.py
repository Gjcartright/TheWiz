from __future__ import annotations

import json
import shutil
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

import quant_platform.orchestration.corrective_phase00_descendants as descendants
from quant_platform.orchestration.corrective_phase00_control import (
    _build_stable_manifest,
    _stable_manifest_freshness_projection,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
CHECKPOINT_ID = "phase00checkpoint_1234567890abcdef1234"
LINEAGE_ID = "phase00artifactmanifest_1234567890abcdef12345678"


def test_historical_audit_reconstructs_exact_zero_authority_decisions() -> None:
    audit = descendants._load_math_audit_evidence(REPOSITORY_ROOT)
    decisions = descendants._build_decision_rows(audit)

    assert len(decisions) == 15
    assert descendants._action_counts(decisions) == {
        "INVALIDATED_PENDING_CONTROLLED_REBUILD": 8,
        "PRESERVED_INPUT_ONLY": 1,
        "PRESERVED_INPUT_ONLY_REQUIRES_QUALITY_CHECK": 1,
        "PRESERVED_RESEARCH_ONLY": 3,
        "SUPERSEDED": 2,
    }
    assert {
        row["subject_id"]
        for row in decisions
        if row["subject_type"] == "math_incident"
    } == {"MATH-023", "MATH-024"}
    assert all(
        row[flag] is False
        for row in decisions
        for flag in descendants.ZERO_AUTHORITY
    )
    assert audit["historical_input_hash_mismatch_count"] > 0


def test_historical_audit_rejects_a_tampered_declared_artifact(
    tmp_path: Path,
) -> None:
    _copy_math_audit_bundle(tmp_path)
    impact = tmp_path / descendants.HISTORICAL_IMPACT
    impact.write_text(impact.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sha256_mismatch"):
        descendants._load_math_audit_evidence(tmp_path)


def test_current_checkpoint_preflight_rejects_stale_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_manifest = {"blockers": []}
    manifest_path = (
        tmp_path
        / "data/research/phase00_control/manifests"
        / f"{CHECKPOINT_ID}.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint = {
        "status": "PASS_QUIESCED_BASELINE",
        "checkpoint_capture_complete": True,
        "snapshot_stable": True,
        "blockers": [],
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_receipt_path": (
            f"data/research/phase00_control/checkpoints/{CHECKPOINT_ID}.json"
        ),
        "stable_manifest_path": (
            f"data/research/phase00_control/manifests/{CHECKPOINT_ID}.json"
        ),
        "stable_manifest_file_sha256": _file_sha256(manifest_path),
        "stable_manifest_sha256": "0" * 64,
        "freshness_manifest_sha256": "0" * 64,
    }
    receipt_path = (
        tmp_path
        / "data/research/phase00_control/checkpoints"
        / f"{CHECKPOINT_ID}.json"
    )
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    active = tmp_path / descendants.ACTIVE_CHECKPOINT
    active.parent.mkdir(parents=True)
    active.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        descendants,
        "_build_stable_manifest",
        lambda *_args, **_kwargs: {"blockers": []},
    )

    with pytest.raises(RuntimeError, match="checkpoint_is_stale"):
        descendants._require_current_checkpoint(tmp_path, runner=lambda *_a, **_k: None)


def test_descendant_bundle_publishes_and_observes_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _publish_fixture_bundle(tmp_path, monkeypatch)
    observation = descendants.observe_phase00_descendant_control(tmp_path)

    assert result.summary["status"] == "PASS_STALE_DESCENDANTS_INVALIDATED"
    assert result.summary["regenerated_descendant_count"] == 0
    assert result.summary["descendant_regeneration_status"] == (
        "INVALIDATED_PENDING_CONTROLLED_REBUILD"
    )
    assert result.summary["authority_flags"] == descendants.ZERO_AUTHORITY
    assert observation["validation_status"] == "PASS"
    assert observation["decision_count"] == 15


def test_observer_rejects_hash_valid_path_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _publish_fixture_bundle(tmp_path, monkeypatch)
    pointer_path = result.paths["active_descendant_control"]
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    original = result.paths["descendant_decisions"]
    alternate = tmp_path / "data/research/phase00_control/alternate.csv"
    alternate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, alternate)
    pointer["decisions_path"] = alternate.relative_to(tmp_path).as_posix()
    pointer["decisions_sha256"] = _file_sha256(alternate)
    pointer_path.write_text(
        json.dumps(pointer, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    observation = descendants.observe_phase00_descendant_control(tmp_path)

    assert observation["validation_status"] == "BLOCKED"
    assert any("path_substitution" in blocker for blocker in observation["blockers"])


def test_observer_rejects_tampered_immutable_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _publish_fixture_bundle(tmp_path, monkeypatch)
    decisions_path = result.paths["descendant_decisions"]
    decisions_path.chmod(0o600)
    decisions_path.write_bytes(decisions_path.read_bytes() + b"tamper\n")

    observation = descendants.observe_phase00_descendant_control(tmp_path)

    assert observation["validation_status"] == "BLOCKED"
    assert any("sha256_mismatch" in blocker for blocker in observation["blockers"])


def test_checkpoint_observer_accepts_only_valid_superseded_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _publish_fixture_bundle(tmp_path, monkeypatch)
    lineage_path = tmp_path / "reports/active/phase00_active_artifact_lineage.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["manifest_id"] = "phase00artifactmanifest_" + "2" * 24
    lineage_path.write_text(
        json.dumps(lineage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    valid_index = {f"artifact-{index}": {} for index in range(74)}
    monkeypatch.setattr(
        descendants,
        "load_active_artifact_envelope_index",
        lambda _root: (valid_index, ()),
    )
    monkeypatch.setattr(
        descendants,
        "load_artifact_envelope_index_for_manifest",
        lambda _root, manifest_id: (
            valid_index,
            (),
        )
        if manifest_id == LINEAGE_ID
        else ({}, ("unexpected_manifest",)),
    )

    default = descendants.observe_phase00_descendant_control(tmp_path)
    historical = descendants.observe_phase00_descendant_control(
        tmp_path,
        allow_superseded_lineage=True,
    )

    assert default["validation_status"] == "BLOCKED"
    assert "descendant_active_lineage_binding_mismatch" in default["blockers"][0]
    assert historical["validation_status"] == "PASS_HISTORICAL_ONLY"
    assert historical["status"] == "SUPERSEDED_DESCENDANT_CONTROL_LINEAGE"
    assert historical["historical_lineage_manifest_id"] == LINEAGE_ID
    assert historical["active_lineage_manifest_id"] == lineage["manifest_id"]
    assert historical["authority_flags"] == descendants.ZERO_AUTHORITY

    decisions = result.paths["descendant_decisions"]
    decisions.chmod(0o600)
    decisions.write_bytes(decisions.read_bytes() + b"tamper\n")
    tampered = descendants.observe_phase00_descendant_control(
        tmp_path,
        allow_superseded_lineage=True,
    )
    assert tampered["validation_status"] == "BLOCKED"
    assert "sha256_mismatch" in tampered["blockers"][0]


def test_stable_manifest_blocks_invalid_descendant_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        descendants,
        "observe_phase00_descendant_control",
        lambda _root, **_kwargs: {
            "validation_status": "BLOCKED",
            "status": "BLOCKED_INVALID_DESCENDANT_CONTROL",
            "descendant_regeneration_status": "BLOCKED_INVALID_CONTROL",
            "blockers": ["ValueError:tampered"],
            "authority_flags": dict(descendants.ZERO_AUTHORITY),
        },
    )
    root = _minimal_git_root(tmp_path)

    manifest = _build_stable_manifest(root, runner=_isolated_runner)

    assert "descendant_control_invalid:ValueError:tampered" in manifest["blockers"]


def test_freshness_projection_ignores_only_cross_process_control_telemetry() -> None:
    baseline = {
        "source_evidence_index": [
            {
                "path": ".runtime_control/effect_authority.sqlite3",
                "artifact_type": "file",
                "classification": "runtime_control",
                "sha256": "a" * 64,
                "metadata_sha256": "b" * 64,
                "size_bytes": 100,
                "hash_status": "HASHED",
                "content_captured": False,
                "implementation_status": "IMPLEMENTED",
                "descendant_regeneration_status": "NOT_EVALUATED",
            },
            {
                "path": "src/fixture.py",
                "sha256": "c" * 64,
                "size_bytes": 10,
            },
        ],
        "lease_index": [
            {
                "path": ".runtime_locks/governed_evidence.lock",
                "sha256": "d" * 64,
                "size_bytes": 1,
                "content_captured": False,
            }
        ],
        "runtime_observation": {
            "processes": {
                "status": "PASS_NO_PRODUCERS",
                "producer_count": 0,
                "returncode": 0,
                "output_sha256": "e" * 64,
            },
            "launchd": [
                {
                    "service": "daily_research",
                    "status": "PASS_UNLOADED",
                    "loaded": False,
                    "returncode": 113,
                    "output_sha256": "f" * 64,
                }
            ],
        },
    }
    telemetry_changed = json.loads(json.dumps(baseline))
    telemetry_changed["source_evidence_index"][0]["sha256"] = "1" * 64
    telemetry_changed["source_evidence_index"][0]["metadata_sha256"] = "2" * 64
    telemetry_changed["source_evidence_index"][0]["size_bytes"] = 200
    telemetry_changed["lease_index"][0]["sha256"] = "3" * 64
    telemetry_changed["runtime_observation"]["processes"]["returncode"] = 1

    assert _stable_manifest_freshness_projection(
        baseline
    ) == _stable_manifest_freshness_projection(telemetry_changed)

    telemetry_changed["source_evidence_index"][1]["sha256"] = "4" * 64
    assert _stable_manifest_freshness_projection(
        baseline
    ) != _stable_manifest_freshness_projection(telemetry_changed)


def _publish_fixture_bundle(
    root: Path, monkeypatch: pytest.MonkeyPatch
):
    (root / ".git").mkdir()
    _copy_math_audit_bundle(root)
    checkpoint_manifest_sha = "a" * 64
    checkpoint_receipt = {
        "checkpoint_id": CHECKPOINT_ID,
        "status": "PASS_QUIESCED_BASELINE",
        "checkpoint_capture_complete": True,
        "stable_manifest_file_sha256": checkpoint_manifest_sha,
    }
    checkpoint_path = (
        root
        / "data/research/phase00_control/checkpoints"
        / f"{CHECKPOINT_ID}.json"
    )
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text(
        json.dumps(checkpoint_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lineage_pointer = {
        "status": "SEALED_FAIL_CLOSED",
        "manifest_id": LINEAGE_ID,
        "registry_size": 74,
    }
    lineage_path = root / "reports/active/phase00_active_artifact_lineage.json"
    lineage_path.parent.mkdir(parents=True)
    lineage_path.write_text(
        json.dumps(lineage_pointer, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    @contextmanager
    def no_op_context(*_args, **_kwargs):
        yield

    monkeypatch.setattr(descendants, "_phase00_maintenance_controller_lock", no_op_context)
    monkeypatch.setattr(descendants, "_phase00_control_publication_authority", no_op_context)
    monkeypatch.setattr(
        descendants,
        "read_phase00_maintenance_state",
        lambda _root: {"blocking": True, "maintenance_id": "phase00maint_fixture"},
    )
    monkeypatch.setattr(
        descendants,
        "_require_current_checkpoint",
        lambda *_args, **_kwargs: (
            {
                "checkpoint_id": CHECKPOINT_ID,
                "stable_manifest_file_sha256": checkpoint_manifest_sha,
            },
            {"runtime_contracts": []},
        ),
    )
    monkeypatch.setattr(
        descendants,
        "_require_complete_lineage",
        lambda _root: {"manifest_id": LINEAGE_ID, "registry_size": 74},
    )
    monkeypatch.setattr(descendants, "_configuration_fingerprint", lambda _root: "c" * 64)
    return descendants.build_phase00_descendant_invalidation(root=root, now=NOW)


def _copy_math_audit_bundle(root: Path) -> None:
    authority_source = REPOSITORY_ROOT / descendants.MATH_AUTHORITY
    authority = json.loads(authority_source.read_text(encoding="utf-8"))
    paths = [descendants.MATH_AUTHORITY]
    paths.extend(Path(binding["path"]) for binding in authority["artifacts"].values())
    for relative in paths:
        source = REPOSITORY_ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _minimal_git_root(root: Path) -> Path:
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True, text=True)
    source = root / "src/fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(root), "add", "."],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.email=phase00@example.invalid",
            "-c",
            "user.name=Phase 00 Test",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return root.resolve()


def _isolated_runner(command, **kwargs):
    if command[0] == "pgrep":
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
    if command[0] == "launchctl":
        return subprocess.CompletedProcess(command, 113, stdout="", stderr="")
    check = bool(kwargs.pop("check", False))
    return subprocess.run(command, check=check, **kwargs)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
