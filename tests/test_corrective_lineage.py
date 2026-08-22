from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from quant_platform.orchestration.corrective_lineage import (
    ACTIVE_ARTIFACT_REGISTRY,
    active_artifact_rows,
)


def test_registry_accounts_for_all_74_audited_candidates() -> None:
    paths = [entry.path for entry in ACTIVE_ARTIFACT_REGISTRY]

    assert len(paths) == 74
    assert len(set(paths)) == 74


def test_typed_resolver_ignores_receipt_ids_and_validates_nested_paths(
    tmp_path: Path,
) -> None:
    root = _repository_fixture(tmp_path)
    target = root / "data" / "research" / "receipt.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"status":"PASS"}\n', encoding="utf-8")
    source = root / "reports" / "active" / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    active = root / "reports" / "active" / "canonical_program_status.json"
    active.write_text(
        json.dumps(
            {
                "receipt_id": "canonicalstatus_not_a_path",
                "manifest_sha256": "a" * 64,
                "immutable_receipt_path": "data/research/receipt.json",
                "immutable_receipt_sha256": _sha256(target),
                "source_artifacts": {
                    "source": {"path": "reports/active/source.json", "sha256": _sha256(source)}
                },
            }
        ),
        encoding="utf-8",
    )

    row = active_artifact_rows(root)[0]

    assert row["reference_count"] == 2
    assert row["resolved_reference_count"] == 2
    assert row["hash_binding_count"] == 2
    assert row["hash_match_count"] == 2
    assert row["resolution_status"] == "RESOLVED_STRUCTURAL"
    assert row["authority_eligible"] is False


def test_declared_multi_path_field_is_split_without_treating_ids_as_paths(
    tmp_path: Path,
) -> None:
    root = _repository_fixture(tmp_path)
    first = root / "models" / "one.json"
    second = root / "reports" / "ml" / "two.csv"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("{}\n", encoding="utf-8")
    second.write_text("value\n1\n", encoding="utf-8")
    active = root / "reports" / "active" / "model_authority_status.json"
    active.write_text(
        json.dumps(
            {
                "active_dataset_id": "tradedataset_not_a_path",
                "evidence_path": "models/one.json;reports/ml/two.csv",
            }
        ),
        encoding="utf-8",
    )

    row = active_artifact_rows(root)[0]

    assert row["reference_count"] == 2
    assert row["resolved_reference_count"] == 2
    assert row["resolution_status"] == "UNRESOLVED_OWNERSHIP_UNRESOLVED"


def test_unregistered_pointer_candidate_fails_closed(tmp_path: Path) -> None:
    root = _repository_fixture(tmp_path)
    unknown = root / "reports" / "active" / "unknown_status.json"
    unknown.write_text("{}\n", encoding="utf-8")

    row = active_artifact_rows(root)[0]

    assert row["artifact_role"] == "unregistered"
    assert row["resolution_status"] == "UNRESOLVED_UNREGISTERED"
    assert row["reference_blockers"] == ["artifact_not_in_registry"]


def test_phase00_control_outputs_are_not_self_registered_as_candidates(
    tmp_path: Path,
) -> None:
    root = _repository_fixture(tmp_path)
    checkpoint = root / "reports" / "active" / "phase00_control_checkpoint.json"
    checkpoint.write_text("{}\n", encoding="utf-8")

    assert active_artifact_rows(root) == []


def test_presentation_and_lock_candidates_are_explicitly_excluded(
    tmp_path: Path,
) -> None:
    root = _repository_fixture(tmp_path)
    presentation = root / "reports" / "active" / "canonical_program_status.md"
    lock = root / "reports" / "active" / "corrective_wizard_proof_launcher_status.json.lock"
    presentation.write_text("# Status\n", encoding="utf-8")
    lock.write_text("{}\n", encoding="utf-8")

    rows = active_artifact_rows(root)

    assert {row["artifact_role"] for row in rows} == {"presentation", "lock"}
    assert {row["resolution_status"] for row in rows} == {"EXCLUDED_NON_POINTER"}


def _repository_fixture(root: Path) -> Path:
    (root / "src" / "quant_platform").mkdir(parents=True)
    (root / "reports" / "active").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return root.resolve()


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
