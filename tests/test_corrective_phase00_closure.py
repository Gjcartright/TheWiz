from __future__ import annotations

import csv
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from io import StringIO
from pathlib import Path

import pytest

import quant_platform.orchestration.corrective_phase00_closure as closure
from quant_platform import cli
from quant_platform.active_pipeline import CommandResult

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 22, 14, 0, tzinfo=UTC)


def test_closure_observer_reports_not_started(tmp_path: Path) -> None:
    assert closure.observe_phase00_closure(tmp_path) == {
        "validation_status": "NOT_STARTED",
        "status": "NOT_STARTED",
        "blockers": [],
        "runtime_activation_authorized": False,
        "authority_flags": dict(closure.ZERO_AUTHORITY),
    }


def test_closure_observer_accepts_exact_v1_bundle_as_historical_only(
    tmp_path: Path,
) -> None:
    paths = _write_legacy_v1_closure(tmp_path)

    observation = closure.observe_phase00_closure(
        tmp_path,
        verify_current_tree=False,
    )

    assert observation["validation_status"] == "PASS_HISTORICAL_ONLY"
    assert observation["status"] == "SUPERSEDED_LEGACY_PHASE00_CLOSURE"
    assert observation["legacy_status"] == "PASS_PHASE00_IMPLEMENTATION_RESEARCH_ONLY"
    assert observation["blockers"] == []
    assert observation["authority_flags"] == closure.ZERO_AUTHORITY
    assert observation["runtime_activation_authorized"] is False

    paths["manifest"].write_bytes(paths["manifest"].read_bytes() + b"tamper\n")
    blocked = closure.observe_phase00_closure(
        tmp_path,
        verify_current_tree=False,
    )
    assert blocked["validation_status"] == "BLOCKED"
    assert "manifest_sha256_mismatch" in blocked["blockers"][0]


def test_closure_observer_rejects_unknown_legacy_schema(tmp_path: Path) -> None:
    paths = _write_legacy_v1_closure(tmp_path)
    pointer = json.loads(paths["pointer"].read_text(encoding="utf-8"))
    pointer["schema_version"] = "thewiz.phase00.closure_pointer.v0"
    paths["pointer"].write_text(json.dumps(pointer), encoding="utf-8")

    observation = closure.observe_phase00_closure(
        tmp_path,
        verify_current_tree=False,
    )

    assert observation["validation_status"] == "BLOCKED"
    assert observation["blockers"] == [
        "ValueError:closure_pointer_schema_invalid"
    ]


def test_closure_bundle_is_immutable_complete_and_research_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _publish_fixture_closure(tmp_path, monkeypatch)
    observation = closure.observe_phase00_closure(tmp_path)

    assert result.summary["status"] == "PASS_PHASE00_IMPLEMENTATION_RESEARCH_ONLY"
    assert result.summary["phase00_implementation_accepted"] is True
    assert result.summary["phase00_runtime_accepted"] is False
    assert result.summary["runtime_activation_authorized"] is False
    assert result.summary["external_calls_performed"] == 0
    assert result.summary["credentials_accessed"] is False
    assert result.summary["provider_credits_used"] == 0
    assert result.summary["orders_submitted"] == 0
    assert result.summary["immutability_contract"] == (
        "TAMPER_EVIDENT_WRITE_ONCE_READ_ONLY_MODE"
    )
    assert result.summary["authority_flags"] == closure.ZERO_AUTHORITY
    assert observation["validation_status"] == "PASS"
    assert all(
        path.stat().st_mode & 0o222 == 0
        for path in result.paths["bundle_root"].iterdir()
        if path.is_file()
    )

    acceptance = _read_csv(result.paths["active_acceptance"])
    assert len(acceptance) == 22
    assert sum(row["status"] == "PASS" for row in acceptance) == 20
    assert {
        row["control_id"]
        for row in acceptance
        if row["status"] == "BLOCKED_SEPARATE_OPERATOR_APPROVAL"
    } == {"P00-SL-020", "P00-SL-021"}

    fault_catalog = _read_csv(result.paths["active_fault_catalog"])
    assert len(fault_catalog) == 36
    assert {
        row["test_id"]
        for row in fault_catalog
        if row["closure_status"] == "DEFERRED_OPERATIONAL_NOT_EXECUTED"
    } == {"T00-030", "T00-035"}


def test_closure_observer_rejects_manifest_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _publish_fixture_closure(tmp_path, monkeypatch)
    manifest = result.paths["closure_manifest"]
    manifest.chmod(0o600)
    manifest.write_bytes(manifest.read_bytes() + b"tamper\n")

    observation = closure.observe_phase00_closure(tmp_path)

    assert observation["validation_status"] == "BLOCKED"
    assert any("manifest_sha256_mismatch" in item for item in observation["blockers"])


def test_closure_observer_rejects_current_tree_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _publish_fixture_closure(tmp_path, monkeypatch)
    source_receipt = json.loads(
        result.paths["source_receipt"].read_text(encoding="utf-8")
    )
    git_receipt = json.loads(result.paths["git_receipt"].read_text(encoding="utf-8"))
    drifted_checkpoint = {
        "checkpoint_id": source_receipt["post_checkpoint_id"],
        "freshness_manifest_sha256": "f" * 64,
        "stable_manifest_file_sha256": source_receipt[
            "post_stable_manifest_file_sha256"
        ],
    }
    current_manifest = {
        key: git_receipt[key]
        for key in ("git_branch", "git_head", "git_dirty", "dirty_path_count")
    }
    monkeypatch.setattr(
        closure,
        "_require_current_checkpoint",
        lambda *_args, **_kwargs: (drifted_checkpoint, current_manifest),
    )

    observation = closure.observe_phase00_closure(tmp_path)

    assert observation["validation_status"] == "BLOCKED"
    assert any("current_freshness_mismatch" in item for item in observation["blockers"])


def test_sanitized_verification_environment_excludes_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRYPTO_WIZARDS_API_KEY", "canary-secret")
    monkeypatch.setenv("HYPERLIQUID_PRIVATE_KEY", "canary-private")
    monkeypatch.setenv("APIFY_API_TOKEN", "canary-token")
    monkeypatch.setenv("PATH", "/usr/bin")

    environment = closure._sanitized_environment(tmp_path)

    assert environment["PATH"] == "/usr/bin"
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert environment["QPA_PHASE00_EXTERNAL_EFFECT_MODE"] == "NO_EXTERNAL_NO_ORDER"
    assert "CRYPTO_WIZARDS_API_KEY" not in environment
    assert "HYPERLIQUID_PRIVATE_KEY" not in environment
    assert "APIFY_API_TOKEN" not in environment


def test_phase00_cli_commands_do_not_load_env_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden_env_load(*_args, **_kwargs):
        raise AssertionError("Phase 00 command attempted to read an env file")

    monkeypatch.setattr(cli, "load_env_file", forbidden_env_load)
    monkeypatch.setattr(
        cli,
        "build_phase00_quiesced_checkpoint",
        lambda **_kwargs: CommandResult(paths={}, summary={"status": "PASS"}),
    )

    cli.main(["phase00-checkpoint"], load_environment=True)

    assert json.loads(capsys.readouterr().out)["summary"]["status"] == "PASS"


def test_phase00_no_external_mode_suppresses_all_cli_env_loading(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden_env_load(*_args, **_kwargs):
        raise AssertionError("closure verification attempted to read an env file")

    monkeypatch.setenv(
        "QPA_PHASE00_EXTERNAL_EFFECT_MODE",
        "NO_EXTERNAL_NO_ORDER",
    )
    monkeypatch.setattr(cli, "load_env_file", forbidden_env_load)
    monkeypatch.setattr(
        cli,
        "system_check",
        lambda: CommandResult(paths={}, summary={"status": "PASS"}),
    )

    cli.main(["system-check"], load_environment=True)

    assert json.loads(capsys.readouterr().out)["summary"]["status"] == "PASS"


def test_dependency_check_prefers_uv_and_targets_canonical_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / ".venv/bin/python"
    monkeypatch.setattr(closure.shutil, "which", lambda *_args, **_kwargs: "/usr/bin/uv")

    command = closure._dependency_check_command(
        python=python,
        environment={"PATH": "/usr/bin"},
    )

    assert command == [
        "/usr/bin/uv",
        "pip",
        "check",
        "--python",
        str(python),
    ]


def test_pytest_command_is_kernel_contained() -> None:
    command, receipt = closure._contained_pytest_command(
        ["/runtime/python", "-m", "pytest", "tests"]
    )

    assert command[:2] == ["/usr/bin/sandbox-exec", "-p"]
    assert command[-4:] == ["/runtime/python", "-m", "pytest", "tests"]
    assert "(deny network*)" in command[2]
    assert '"/usr/bin/security"' in command[2]
    assert receipt["status"] == "ENFORCED"
    assert receipt["network_denied"] is True
    assert receipt["keychain_security_exec_denied"] is True


def test_junit_parser_requires_real_tests() -> None:
    parsed = closure._parse_junit(
        b'<testsuites tests="12" failures="0" errors="0" skipped="1" time="1.5" />'
    )
    assert parsed == {
        "tests": 12,
        "failures": 0,
        "errors": 0,
        "skipped": 1,
        "time": 1.5,
    }
    assert closure._parse_junit(b"")["errors"] == 1


def test_control_evidence_defaults_to_not_proven(tmp_path: Path) -> None:
    config = REPOSITORY_ROOT / closure.CONTROL_EVIDENCE_MAP
    target = tmp_path / closure.CONTROL_EVIDENCE_MAP
    target.parent.mkdir(parents=True)
    shutil.copy2(config, target)

    evidence = closure._build_control_evidence(
        tmp_path,
        closure_id="phase00closure_" + "a" * 24,
        junit_xml=(
            b'<testsuites tests="2273" failures="0" errors="0" skipped="0" '
            b'time="1.0" />'
        ),
    )

    assert evidence["implementation_controls_proven"] is False
    assert evidence["implementation_faults_proven"] is False
    assert {
        row["status"]
        for row in evidence["acceptance_controls"]
        if row["requirement_id"] not in closure.RELOAD_CONTROLS
    } == {"NOT_PROVEN"}


def test_control_evidence_rejects_skipped_required_case(tmp_path: Path) -> None:
    config = REPOSITORY_ROOT / closure.CONTROL_EVIDENCE_MAP
    target = tmp_path / closure.CONTROL_EVIDENCE_MAP
    target.parent.mkdir(parents=True)
    shutil.copy2(config, target)
    mapping = json.loads(target.read_text(encoding="utf-8"))
    selector = mapping["acceptance_controls"]["P00-SL-001"][0]
    class_name, name = selector.rsplit("::", 1)
    junit = (
        f'<testsuite tests="1" failures="0" errors="0" skipped="1">'
        f'<testcase classname="{class_name}" name="{name}"><skipped /></testcase>'
        "</testsuite>"
    ).encode()

    evidence = closure._build_control_evidence(
        tmp_path,
        closure_id="phase00closure_" + "b" * 24,
        junit_xml=junit,
    )
    row = next(
        item
        for item in evidence["acceptance_controls"]
        if item["requirement_id"] == "P00-SL-001"
    )

    assert row["status"] == "NOT_PROVEN"
    assert row["matched_outcomes"] == ["skipped"]


def _publish_fixture_closure(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> CommandResult:
    (root / ".git").mkdir()
    source = root / "src/fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    python = root / ".venv/bin/python"
    ruff = root / ".venv/bin/ruff"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    ruff.write_text("", encoding="utf-8")
    for relative in (
        closure.HISTORICAL_ACCEPTANCE,
        closure.HISTORICAL_FAULT_CATALOG,
        closure.CONTROL_EVIDENCE_MAP,
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / relative, target)

    runtime_observation = {
        "processes": {
            "status": "PASS_NO_PRODUCERS",
            "producer_count": 0,
            "returncode": 1,
        },
        "launchd": [
            {
                "service": service,
                "status": "PASS_UNLOADED",
                "loaded": False,
                "returncode": 113,
            }
            for service in sorted(closure.EXPECTED_LAUNCHD_SERVICES)
        ],
    }
    preflight = {
        "maintenance_id": "phase00maint_fixture",
        "checkpoint": {
            "checkpoint_id": "phase00checkpoint_current000000000",
            "freshness_manifest_sha256": "a" * 64,
            "stable_manifest_file_sha256": "b" * 64,
            "runtime_observation": runtime_observation,
        },
        "manifest": {
            "git_branch": "fixture",
            "git_head": "c" * 40,
            "git_dirty": True,
            "dirty_path_count": 1,
            "source_evidence_index": [
                {
                    "path": "src/fixture.py",
                    "sha256": "d" * 64,
                }
            ],
            "publication_surface_index": [{"surface_id": "fixture"}],
            "financial_effect_surface_index": [{"surface_id": "fixture"}],
            "blockers": [],
        },
        "checkpoint_ids": [
            "phase00checkpoint_first0000000000",
            "phase00checkpoint_second000000000",
        ],
        "descendant": {
            "validation_status": "PASS",
            "control_id": "phase00descendants_fixture000000",
            "descendant_regeneration_status": (
                "INVALIDATED_PENDING_CONTROLLED_REBUILD"
            ),
        },
    }

    @contextmanager
    def no_op_context(*_args, **_kwargs):
        yield

    def fake_runner(command, **_kwargs):
        if "pytest" in command:
            junit_argument = next(
                value for value in command if str(value).startswith("--junitxml=")
            )
            junit_path = Path(str(junit_argument).split("=", 1)[1])
            junit_path.write_bytes(_fully_mapped_junit(root))
            return subprocess.CompletedProcess(
                command, 0, stdout=b"25 passed\n", stderr=b""
            )
        return subprocess.CompletedProcess(command, 0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr(closure, "_phase00_maintenance_controller_lock", no_op_context)
    monkeypatch.setattr(closure, "_phase00_control_publication_authority", no_op_context)
    monkeypatch.setattr(
        closure,
        "_closure_preflight",
        lambda *_args, **_kwargs: preflight,
    )
    monkeypatch.setattr(
        closure,
        "_require_current_checkpoint",
        lambda *_args, **_kwargs: (preflight["checkpoint"], preflight["manifest"]),
    )
    monkeypatch.setattr(
        closure,
        "observe_phase00_descendant_control",
        lambda *_args, **_kwargs: preflight["descendant"],
    )
    monkeypatch.setattr(closure, "_phase00_ruff_scope", lambda _root: ["src/fixture.py"])
    monkeypatch.setattr(closure, "_configuration_fingerprint", lambda _root: "e" * 64)
    return closure.build_phase00_closure_verification(
        root=root,
        now=NOW,
        runner=fake_runner,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(path.read_text(encoding="utf-8"))))


def _write_legacy_v1_closure(root: Path) -> dict[str, Path]:
    closure_id = "phase00closure_" + "1" * 24
    output_root = root / closure.CONTROL_ROOT / closure_id
    output_root.mkdir(parents=True)
    authority = dict(closure.ZERO_AUTHORITY)
    json_leaves = {
        name: {
            "schema_version": schema,
            "closure_id": closure_id,
            "status": "PASS_RUNTIME_QUIESCED" if name == "runtime_receipt.json" else "HISTORICAL",
            "runtime_activation_authorized": False,
            "authority_flags": authority,
        }
        for name, schema in closure.LEGACY_V1_JSON_SCHEMAS.items()
    }
    leaf_payloads = {
        name: json.dumps(payload, sort_keys=True).encode("utf-8")
        for name, payload in json_leaves.items()
    }
    for name in closure.LEGACY_V1_REQUIRED_BUNDLE_FILES:
        leaf_payloads.setdefault(name, f"legacy:{name}\n".encode())
    entries = []
    for name in sorted(leaf_payloads):
        path = output_root / name
        path.write_bytes(leaf_payloads[name])
        entries.append(
            {
                "name": name,
                "path": str(path.relative_to(root)),
                "sha256": sha256(leaf_payloads[name]).hexdigest(),
                "size_bytes": len(leaf_payloads[name]),
            }
        )
    manifest = {
        "schema_version": closure.LEGACY_V1_MANIFEST_SCHEMA_VERSION,
        "closure_id": closure_id,
        "entry_count": len(entries),
        "entries": entries,
        "authority_flags": authority,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    manifest_sha = sha256(manifest_path.read_bytes()).hexdigest()
    closure_receipt = {
        "schema_version": closure.LEGACY_V1_CLOSURE_SCHEMA_VERSION,
        "closure_id": closure_id,
        "status": "PASS_PHASE00_IMPLEMENTATION_RESEARCH_ONLY",
        "blockers": [],
        "manifest_path": str(manifest_path.relative_to(root)),
        "manifest_sha256": manifest_sha,
        "runtime_activation_authorized": False,
        "authority_flags": authority,
    }
    closure_path = output_root / "closure_receipt.json"
    closure_path.write_text(
        json.dumps(closure_receipt, sort_keys=True),
        encoding="utf-8",
    )
    pointer = {
        "schema_version": closure.LEGACY_V1_POINTER_SCHEMA_VERSION,
        "closure_id": closure_id,
        "status": closure_receipt["status"],
        "closure_receipt_path": str(closure_path.relative_to(root)),
        "closure_receipt_sha256": sha256(closure_path.read_bytes()).hexdigest(),
        "manifest_path": closure_receipt["manifest_path"],
        "manifest_sha256": manifest_sha,
        "phase00_runtime_accepted": False,
        "runtime_activation_authorized": False,
        "authority_flags": authority,
    }
    entries_by_name = {entry["name"]: entry for entry in entries}
    for name, (path_key, hash_key) in {
        "test_receipt.json": ("test_receipt_path", "test_receipt_sha256"),
        "acceptance_matrix.csv": (
            "acceptance_matrix_path",
            "acceptance_matrix_sha256",
        ),
        "fault_catalog.csv": ("fault_catalog_path", "fault_catalog_sha256"),
    }.items():
        pointer[path_key] = entries_by_name[name]["path"]
        pointer[hash_key] = entries_by_name[name]["sha256"]
    pointer_path = root / closure.ACTIVE_POINTER
    pointer_path.parent.mkdir(parents=True)
    pointer_path.write_text(json.dumps(pointer, sort_keys=True), encoding="utf-8")
    (root / closure.ACTIVE_ACCEPTANCE).write_bytes(
        leaf_payloads["acceptance_matrix.csv"]
    )
    (root / closure.ACTIVE_FAULT_CATALOG).write_bytes(
        leaf_payloads["fault_catalog.csv"]
    )
    return {
        "pointer": pointer_path,
        "manifest": manifest_path,
        "closure": closure_path,
    }


def _fully_mapped_junit(root: Path) -> bytes:
    mapping = json.loads((root / closure.CONTROL_EVIDENCE_MAP).read_text(encoding="utf-8"))
    selectors = sorted(
        {
            selector
            for group_name in ("acceptance_controls", "fault_contracts")
            for values in mapping[group_name].values()
            for selector in values
        }
    )
    suite = ET.Element(
        "testsuite",
        tests=str(len(selectors)),
        failures="0",
        errors="0",
        skipped="0",
        time="0.2",
    )
    for selector in selectors:
        class_name, name = selector.rsplit("::", 1)
        ET.SubElement(suite, "testcase", classname=class_name, name=name)
    return ET.tostring(suite, encoding="utf-8", xml_declaration=True)
