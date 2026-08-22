from __future__ import annotations

import copy
import json
from hashlib import sha256
from pathlib import Path

import pytest

from quant_platform.orchestration.corrective_lineage import (
    CANONICAL_RECEIPT_TYPES,
    LineageValidationError,
    append_lineage_event,
    build_canonical_receipt,
    build_descendant_invalidation_event,
    build_descendant_regeneration_event,
    canonical_event_sha256,
    canonical_receipt_sha256,
    lineage_parent,
    load_lineage_event_log,
    require_gate00f_authority,
    validate_gate00f_lineage,
    validate_lineage_files,
)

CREATED_AT = "2026-08-21T12:00:00Z"


def test_complete_canonical_chain_reaches_exactly_one_root() -> None:
    receipts = _chain("valid")

    result = validate_gate00f_lineage(receipts)

    assert result.valid is True
    assert result.authority_eligible is True
    assert result.blocker_codes == ()
    assert result.root_artifact_ids == ("source.valid",)
    assert result.authority_artifact_ids == ("authority.valid",)
    assert result.topological_order == tuple(
        f"{name}.{'valid'}"
        for name in ("source", "input", "run", "artifact", "quality", "authority")
    )
    require_gate00f_authority(result)


def test_tampered_content_and_resealed_parent_both_fail_closed() -> None:
    direct_tamper = copy.deepcopy(_chain("direct"))
    direct_tamper[3]["payload"]["artifact_sha256"] = "f" * 64

    direct_result = validate_gate00f_lineage(direct_tamper)

    assert "CONTENT_HASH_MISMATCH" in direct_result.blocker_codes
    assert direct_result.authority_eligible is False

    resealed_parent = copy.deepcopy(_chain("resealed"))
    resealed_parent[0]["payload"]["mapping_sha256"] = "e" * 64
    resealed_parent[0]["content_sha256"] = canonical_receipt_sha256(resealed_parent[0])

    parent_result = validate_gate00f_lineage(resealed_parent)

    assert "PARENT_HASH_MISMATCH" in parent_result.blocker_codes
    assert parent_result.authority_eligible is False


def test_missing_and_duplicate_identity_are_rejected() -> None:
    missing = _chain("missing")
    del missing[2]

    missing_result = validate_gate00f_lineage(missing)

    assert "MISSING_PARENT" in missing_result.blocker_codes

    duplicated = _chain("duplicate")
    duplicated.append(copy.deepcopy(duplicated[3]))

    duplicate_result = validate_gate00f_lineage(duplicated)

    assert "DUPLICATE_ARTIFACT_ID" in duplicate_result.blocker_codes
    assert duplicate_result.authority_eligible is False


def test_cycle_and_wrong_schema_are_rejected() -> None:
    cycle = copy.deepcopy(_chain("cycle"))
    cycle[0]["parents"] = [lineage_parent(cycle[-1])]
    cycle[0]["content_sha256"] = canonical_receipt_sha256(cycle[0])

    cycle_result = validate_gate00f_lineage(cycle)

    assert "LINEAGE_CYCLE" in cycle_result.blocker_codes

    wrong_schema = copy.deepcopy(_chain("schema"))
    wrong_schema[2]["schema_version"] = "thewiz.gate00f.run_manifest.v0"
    wrong_schema[2]["unexpected"] = True
    wrong_schema[2]["content_sha256"] = canonical_receipt_sha256(wrong_schema[2])

    schema_result = validate_gate00f_lineage(wrong_schema)

    assert "SCHEMA_VERSION_MISMATCH" in schema_result.blocker_codes
    assert "RECEIPT_SCHEMA_FIELDS_INVALID" in schema_result.blocker_codes


def test_stale_parent_and_version_drift_are_rejected_after_full_reseal() -> None:
    stale = copy.deepcopy(_chain("stale"))
    stale[0]["lineage_state"] = "RESEARCH_ONLY"
    _reseal_chain(stale)

    stale_result = validate_gate00f_lineage(stale)

    assert "STALE_PARENT" in stale_result.blocker_codes
    assert "STALE_AUTHORITY_LINEAGE" in stale_result.blocker_codes

    drift = copy.deepcopy(_chain("drift"))
    drift[2]["bindings"]["formula"] = {
        "version": "formula.v2",
        "sha256": _digest("formula.v2"),
    }
    _reseal_chain(drift)

    drift_result = validate_gate00f_lineage(drift)

    assert "VERSION_BINDING_MISMATCH" in drift_result.blocker_codes
    assert drift_result.authority_eligible is False


def test_invalidation_event_contains_exact_transitive_descendant_closure() -> None:
    receipts = _chain("old")
    event = build_descendant_invalidation_event(
        receipts,
        trigger_artifact_id="run.old",
        reason_code="FORMULA_DEFECT",
        recorded_at_utc="2026-08-21T13:00:00Z",
    )

    assert {row["artifact_id"]: row["depth"] for row in event["invalidated"]} == {
        "run.old": 0,
        "artifact.old": 1,
        "quality.old": 2,
        "authority.old": 3,
    }
    blocked = validate_gate00f_lineage(receipts, events=[event])
    assert "INVALIDATED_AUTHORITY_LINEAGE" in blocked.blocker_codes
    assert blocked.invalidated_artifact_ids == (
        "artifact.old",
        "authority.old",
        "quality.old",
        "run.old",
    )

    incomplete = copy.deepcopy(event)
    incomplete["invalidated"] = incomplete["invalidated"][:-1]
    incomplete["event_sha256"] = canonical_event_sha256(incomplete)

    incomplete_result = validate_gate00f_lineage(receipts, events=[incomplete])

    assert "INVALIDATION_CLOSURE_MISMATCH" in incomplete_result.blocker_codes

    duplicated = copy.deepcopy(event)
    duplicated["invalidated"].append(copy.deepcopy(duplicated["invalidated"][0]))
    duplicated["event_sha256"] = canonical_event_sha256(duplicated)

    duplicate_result = validate_gate00f_lineage(receipts, events=[duplicated])

    assert "DUPLICATE_EVENT_ROW" in duplicate_result.blocker_codes


def test_regeneration_uses_new_ids_and_never_revalidates_old_descendants() -> None:
    old = _chain("oldregen")
    new = _chain("newregen", created_at="2026-08-21T14:30:00Z")
    receipts = [*old, *new]
    invalidation = build_descendant_invalidation_event(
        receipts,
        trigger_artifact_id="run.oldregen",
        reason_code="ECONOMIC_CONTRACT_DEFECT",
        recorded_at_utc="2026-08-21T14:00:00Z",
    )
    regeneration = build_descendant_regeneration_event(
        receipts,
        invalidation_event=invalidation,
        replacements={
            "run.oldregen": "run.newregen",
            "artifact.oldregen": "artifact.newregen",
            "quality.oldregen": "quality.newregen",
            "authority.oldregen": "authority.newregen",
        },
        recorded_at_utc="2026-08-21T15:00:00Z",
    )

    new_result = validate_gate00f_lineage(
        receipts,
        authority_artifact_id="authority.newregen",
        events=[invalidation, regeneration],
    )
    old_result = validate_gate00f_lineage(
        receipts,
        authority_artifact_id="authority.oldregen",
        events=[invalidation, regeneration],
    )

    assert new_result.authority_eligible is True
    assert new_result.regenerated_artifact_ids == (
        "artifact.newregen",
        "authority.newregen",
        "quality.newregen",
        "run.newregen",
    )
    assert old_result.authority_eligible is False
    assert "INVALIDATED_AUTHORITY_LINEAGE" in old_result.blocker_codes


def test_append_only_event_log_rejects_chain_forks_and_tampering(
    tmp_path: Path,
) -> None:
    receipts = [
        *_chain("logold"),
        *_chain("lognew", created_at="2026-08-21T16:30:00Z"),
    ]
    invalidation = build_descendant_invalidation_event(
        receipts,
        trigger_artifact_id="run.logold",
        reason_code="SOURCE_DEFECT",
        recorded_at_utc="2026-08-21T16:00:00Z",
    )
    regeneration = build_descendant_regeneration_event(
        receipts,
        invalidation_event=invalidation,
        replacements={"run.logold": "run.lognew"},
        recorded_at_utc="2026-08-21T17:00:00Z",
    )
    log_path = tmp_path / "lineage_events.jsonl"

    append_lineage_event(log_path, invalidation)

    retrograde = copy.deepcopy(regeneration)
    retrograde["event_id"] = "lineage_regeneration:retrograde"
    retrograde["recorded_at_utc"] = "2026-08-21T15:00:00Z"
    retrograde["event_sha256"] = canonical_event_sha256(retrograde)
    with pytest.raises(LineageValidationError, match="EVENT_TIMESTAMP_OUT_OF_ORDER"):
        append_lineage_event(log_path, retrograde)

    append_lineage_event(log_path, regeneration)

    assert load_lineage_event_log(log_path) == [invalidation, regeneration]

    fork = copy.deepcopy(regeneration)
    fork["event_id"] = "lineage_regeneration:forked"
    fork["previous_event_sha256"] = invalidation["event_sha256"]
    fork["event_sha256"] = canonical_event_sha256(fork)
    with pytest.raises(LineageValidationError, match="EVENT_CHAIN_HASH_MISMATCH"):
        append_lineage_event(log_path, fork)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["reason_code"] = "TAMPERED_REASON"
    lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(LineageValidationError, match="EVENT_HASH_MISMATCH"):
        load_lineage_event_log(log_path)


def test_file_loader_rejects_duplicate_json_keys_and_symlinks(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"artifact_id":"one","artifact_id":"two"}\n', encoding="utf-8")

    duplicate_result = validate_lineage_files(tmp_path, [duplicate])

    assert "RECEIPT_FILE_INVALID" in duplicate_result.blocker_codes

    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    symlink = tmp_path / "linked.json"
    symlink.symlink_to(target)

    symlink_result = validate_lineage_files(tmp_path, [symlink])

    assert "RECEIPT_FILE_INVALID" in symlink_result.blocker_codes
    assert symlink_result.authority_eligible is False

    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    nested_target = real_parent / "nested.json"
    nested_target.write_text("{}\n", encoding="utf-8")
    linked_parent = tmp_path / "linked_parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    parent_result = validate_lineage_files(tmp_path, [linked_parent / "nested.json"])

    assert "RECEIPT_FILE_INVALID" in parent_result.blocker_codes


def _chain(name: str, *, created_at: str = CREATED_AT) -> list[dict[str, object]]:
    bindings = _bindings(name)
    payloads: dict[str, dict[str, str]] = {
        "source_mapping_receipt": {
            "source_snapshot_sha256": _digest(f"snapshot:{name}"),
            "mapping_sha256": _digest(f"mapping:{name}"),
        },
        "input_manifest": {"input_set_sha256": _digest(f"input:{name}")},
        "run_manifest": {"run_config_sha256": _digest(f"run:{name}")},
        "artifact_manifest": {"artifact_sha256": _digest(f"artifact:{name}")},
        "data_quality_receipt": {
            "checks_sha256": _digest(f"checks:{name}"),
            "status": "PASS",
        },
        "authority_receipt": {
            "accepted_artifact_id": f"artifact.{name}",
            "decision": "ACCEPT",
        },
    }
    id_names = ("source", "input", "run", "artifact", "quality", "authority")
    receipts: list[dict[str, object]] = []
    for artifact_type, id_name in zip(CANONICAL_RECEIPT_TYPES, id_names, strict=True):
        parents = () if not receipts else (lineage_parent(receipts[-1]),)
        receipts.append(
            build_canonical_receipt(
                artifact_type=artifact_type,
                artifact_id=f"{id_name}.{name}",
                created_at_utc=created_at,
                bindings=bindings,
                payload=payloads[artifact_type],
                parents=parents,
            )
        )
    return receipts


def _reseal_chain(receipts: list[dict[str, object]]) -> None:
    for index, receipt in enumerate(receipts):
        if index:
            receipt["parents"] = [lineage_parent(receipts[index - 1])]
        receipt["content_sha256"] = canonical_receipt_sha256(receipt)


def _bindings(name: str) -> dict[str, dict[str, str]]:
    return {
        binding: {
            "version": f"{binding}.{name}.v1",
            "sha256": _digest(f"{binding}:{name}"),
        }
        for binding in ("formula", "cost", "dataset", "runtime", "source", "policy")
    }


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()
