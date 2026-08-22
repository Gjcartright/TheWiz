from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from quant_platform.orchestration.corrective_active_artifact_envelopes import (
    EXPECTED_VALIDATORS,
    build_active_artifact_lineage_bundle,
    load_active_artifact_envelope_index,
)
from quant_platform.orchestration.corrective_lineage import (
    ACTIVE_ARTIFACT_REGISTRY,
    active_artifact_rows,
)

NOW = "2026-08-21T20:00:00+00:00"
FINGERPRINT = "a" * 64


def test_bundle_covers_all_74_registered_artifacts_and_every_validator() -> None:
    observed = {registration.validator for registration in ACTIVE_ARTIFACT_REGISTRY}

    assert len(ACTIVE_ARTIFACT_REGISTRY) == 74
    assert observed == EXPECTED_VALIDATORS


def test_unproven_authority_is_sealed_as_null_target_blocked(tmp_path: Path) -> None:
    root = _root(tmp_path)
    receipt = root / "data/research/example/receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"status":"PASS"}\n', encoding="utf-8")
    active = root / "reports/active/canonical_program_status.json"
    active.write_text(
        json.dumps(
            {
                "immutable_receipt_path": "data/research/example/receipt.json",
                "immutable_receipt_sha256": _sha256(receipt),
            }
        ),
        encoding="utf-8",
    )

    bundle = _bundle(root)
    envelope = _envelope(bundle, "reports/active/canonical_program_status.json")

    assert len(bundle["manifest"]["entries"]) == 74
    assert envelope["state"] == "BLOCKED"
    assert envelope["target"] is None
    assert envelope["domain_validation_status"] == "PASS"
    assert envelope["blockers"] == ["REGISTERED_DOMAIN_REPLAY_REQUIRED"]
    assert all(value is False for value in envelope["authority_flags"].values())


def test_blocked_decision_requires_null_target_and_zero_authority(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    active = root / "reports/active/live_canary_executor_status.json"
    active.write_text(
        json.dumps(
            {
                "status": "BLOCKED",
                "blockers": ["phase00_not_released"],
                "paper_authorized": False,
                "testnet_order_authority": False,
                "live_trading_authority": False,
            }
        ),
        encoding="utf-8",
    )

    envelope = _envelope(
        _bundle(root), "reports/active/live_canary_executor_status.json"
    )

    assert envelope["state"] == "BLOCKED"
    assert envelope["target"] is None
    assert envelope["blockers"] == ["phase00_not_released"]
    assert envelope["resolution_status"] == "RESOLVED_IMMUTABLE_BLOCKED"


def test_blocked_decision_with_nonzero_authority_invalidates_bundle(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    active = root / "reports/active/live_canary_executor_status.json"
    active.write_text(
        json.dumps(
            {
                "status": "BLOCKED",
                "blockers": ["contradictory_fixture"],
                "testnet_order_authority": True,
            }
        ),
        encoding="utf-8",
    )
    bundle = _bundle(root)
    _publish_bundle(root, bundle)

    index, issues = load_active_artifact_envelope_index(root)

    assert index == {}
    assert "ACTIVE_ENVELOPE_DOMAIN_VALIDATION_FAILED" in {
        issue.code for issue in issues
    }


def test_integrity_defect_is_superseded_without_rehashing_original(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    active = root / "reports/active/wizard_copula_behavioral_status.json"
    active.write_text(
        json.dumps(
            {
                "proof_queue_path": "reports/active/missing.csv",
                "proof_queue_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    original_sha256 = _sha256(active)

    envelope = _envelope(
        _bundle(root), "reports/active/wizard_copula_behavioral_status.json"
    )

    assert envelope["state"] == "SUPERSEDED_BLOCKED"
    assert envelope["target"] is None
    assert envelope["preserved_original_sha256"] == original_sha256
    assert envelope["supersedes_original_without_rehash"] is True
    assert envelope["observed_artifact"]["sha256"] == original_sha256


def test_valid_manifest_resolves_research_view_and_mutation_revokes_it(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    active = (
        root
        / "reports/active/current_wizard_hyperliquid_walkforward_status.csv"
    )
    active.write_text("pair,status\nBTC-ETH,BLOCKED\n", encoding="utf-8")
    bundle = _bundle(root)
    _publish_bundle(root, bundle)

    index, issues = load_active_artifact_envelope_index(root)
    sealed = active_artifact_rows(root)

    assert issues == ()
    assert len(index) == 74
    assert sealed[0]["resolution_status"] == "RESOLVED_IMMUTABLE_RESEARCH_ONLY"
    assert sealed[0]["domain_validation_status"] == "PASS"
    assert sealed[0]["authority_eligible"] is False
    assert sealed[0]["lineage_envelope_status"] == "VALID"

    active.write_text("pair,status\nBTC-ETH,PASS\n", encoding="utf-8")
    stale = active_artifact_rows(root)

    assert stale[0]["resolution_status"] == "UNRESOLVED_STALE_ARTIFACT_ENVELOPE"
    assert stale[0]["domain_validation_status"] == "BLOCKED_STALE_ENVELOPE"
    assert stale[0]["lineage_envelope_status"] == "STALE_ARTIFACT_HASH"


def test_tampered_envelope_invalidates_the_entire_manifest(tmp_path: Path) -> None:
    root = _root(tmp_path)
    active = root / "reports/active/dashboard_refresh_status.csv"
    active.write_text("status\nBLOCKED\n", encoding="utf-8")
    bundle = _bundle(root)
    _publish_bundle(root, bundle)
    first_path = root / bundle["envelopes"][0]["path"]
    first_path.write_text("{}\n", encoding="utf-8")

    index, issues = load_active_artifact_envelope_index(root)
    rows = active_artifact_rows(root)

    assert index == {}
    assert issues
    assert rows[0]["lineage_envelope_status"] == "INVALID_MANIFEST"
    assert rows[0]["authority_eligible"] is False


def _root(path: Path) -> Path:
    (path / "reports/active").mkdir(parents=True)
    return path.resolve()


def _bundle(root: Path) -> dict:
    return build_active_artifact_lineage_bundle(
        root=root,
        recorded_at_utc=NOW,
        source_fingerprint_sha256=FINGERPRINT,
        runtime_fingerprint_sha256="b" * 64,
        configuration_fingerprint_sha256="c" * 64,
    )


def _envelope(bundle: dict, artifact_path: str) -> dict:
    entry = next(
        value
        for value in bundle["manifest"]["entries"]
        if value["artifact_path"] == artifact_path
    )
    return next(
        value["payload"]
        for value in bundle["envelopes"]
        if value["payload"]["envelope_id"] == entry["envelope_id"]
    )


def _publish_bundle(root: Path, bundle: dict) -> None:
    for envelope in bundle["envelopes"]:
        _write_json(root / envelope["path"], envelope["payload"])
    _write_json(root / bundle["manifest_path"], bundle["manifest"])
    _write_json(root / bundle["pointer_path"], bundle["pointer"])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
