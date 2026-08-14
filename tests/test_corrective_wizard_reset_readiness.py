from __future__ import annotations

import csv
import json
import plistlib
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from quant_platform.orchestration import corrective_wizard_reset_readiness
from quant_platform.orchestration.corrective_wizard_browser_auth import (
    build_wizard_browser_auth_readiness,
)
from quant_platform.orchestration.corrective_wizard_capture_manifest import (
    _load_or_create_source_receipt,
)
from quant_platform.orchestration.corrective_wizard_reset_readiness import (
    LAUNCH_AGENT_LABEL,
    LAUNCHER_MODULE,
    build_corrective_wizard_reset_readiness,
    validate_wizard_reset_readiness_receipt,
    wizard_reset_readiness_state_sha256,
)

NOW = datetime(2026, 8, 11, 15, 30, tzinfo=UTC)
RESET = "2026-08-12T00:00:00+00:00"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _browser_auth_observation(route_kind: str) -> dict[str, object]:
    pair = route_kind == "pair_detail"
    url = (
        "https://cryptowizards.net/wizards/zscore/pair/6"
        if pair
        else "https://cryptowizards.net/wizards/zscore/scanner"
    )
    markers = (
        [
            "pair_mode_selector",
            "timeframe_selector",
            "ordered_asset_inputs",
            "rendered_asset_labels",
        ]
        if pair
        else [
            "scanner_filter_controls",
            "scanner_strategy_control",
            "scanner_exchange_control",
            "scanner_results_surface",
        ]
    )
    return {
        "schema_version": "wizard_browser_auth_observation.v1",
        "captured_at": "2026-08-11T15:25:00+00:00",
        "requested_url": url,
        "requested_url_source": "capture_argument",
        "final_url": url,
        "route_kind": route_kind,
        "member_navigation_targets": [
            "https://cryptowizards.net/wizards/account",
            "https://cryptowizards.net/wizards/zscore/scanner",
            "https://cryptowizards.net/wizards/zscore/trades",
        ],
        "protected_content_markers": markers,
        "sign_in_form_present": False,
        "verification_form_present": False,
        "public_marketing_shell_present": False,
        "browser_storage_accessed": False,
        "no_credentials_or_browser_storage_captured": True,
    }


def _build_browser_auth(root: Path, *, max_age_hours: float = 24.0) -> None:
    _write_json(
        root / "config" / "wizard_browser_auth_contract.json",
        {
            "schema_version": "thewiz.wizard_browser_auth_contract.v1",
            "max_age_hours": max_age_hours,
            "required_route_kinds": ["scanner", "pair_detail"],
            "promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    raw = root / "data" / "raw" / "crypto_wizards" / "browser_auth"
    for route_kind in ("scanner", "pair_detail"):
        _write_json(raw / f"{route_kind}.json", _browser_auth_observation(route_kind))
    build_wizard_browser_auth_readiness(root=root, now=NOW)


def _prepare_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    active = root / "reports" / "active"
    active.mkdir(parents=True)
    python = root / ".venv312" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    runtime_temp = root / ".runtime_tmp"
    runtime_temp.mkdir()
    runtime_temp.chmod(0o700)

    manifest_id = "wizardcapture_0123456789abcdef0123"
    immutable_relative = f"data/research/wizard_capture_manifests/{manifest_id}.json"
    immutable = root / immutable_relative
    _write_json(immutable, {"manifest_id": manifest_id, "calls": 13})
    immutable_digest = sha256(immutable.read_bytes()).hexdigest()
    source_path = active / "frozen_capture_source.csv"
    source_path.write_text("call_id\ncall-1\n", encoding="utf-8")
    source_digest = sha256(source_path.read_bytes()).hexdigest()
    source_receipt, source_receipt_path, _ = _load_or_create_source_receipt(
        root=root,
        manifest_id=manifest_id,
        immutable_manifest_path=immutable,
        source_artifacts=[
            {
                "path": str(source_path.relative_to(root)),
                "sha256": source_digest,
            }
        ],
        retrofit=True,
    )
    _write_json(
        active / "corrective_wizard_next_capture_manifest.json",
        {
            "schema_version": "thewiz.corrective_wizard_next_capture_manifest.v1",
            "status": "PASS",
            "capture_state": "DEFERRED_UNTIL_UTC_RESET",
            "manifest_enforced": True,
            "manifest_id": manifest_id,
            "immutable_manifest_path": immutable_relative,
            "immutable_manifest_sha256": immutable_digest,
            "pending_calls": 13,
            "planned_credits": 18,
            "proof_lane_credit_ceiling": 76,
            "lane_totals": {
                "exact_mode_backtest": {"calls": 1, "credits": 2},
                "ou_v3_holdout": {"calls": 4, "credits": 8},
                "copula_behavioral": {"calls": 8, "credits": 8},
            },
            "next_external_attempt_eligible_at": RESET,
            "runtime_credit_preflight_required": True,
            "source_artifacts": [
                {
                    "path": str(source_path.relative_to(root)),
                    "sha256": source_digest,
                }
            ],
            "source_artifacts_sha256": source_receipt["source_artifacts_sha256"],
            "source_artifacts_current_sha256": source_receipt["source_artifacts_sha256"],
            "source_artifacts_current_match": True,
            "source_artifacts_current_drift_classification": "EXACT_MATCH",
            "source_artifacts_current_metadata_only_paths": [],
            "source_artifacts_current_scientific_or_structural_paths": [],
            "source_receipt_id": source_receipt["receipt_id"],
            "source_receipt_path": str(source_receipt_path.relative_to(root)),
            "source_receipt_sha256": sha256(source_receipt_path.read_bytes()).hexdigest(),
            "research_only": True,
            "candidate_promotion_authority": False,
            "order_submission_included": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    launcher = {
        "schema_version": "thewiz.corrective_wizard_proof_launcher.v1",
        "checked_at_utc": "2026-08-11T15:25:00+00:00",
        "completed_at_utc": "2026-08-11T15:25:01+00:00",
        "attempt_date_utc": "2026-08-11",
        "execute_requested": True,
        "next_external_attempt_eligible_at": RESET,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    launcher_id = (
        "wizardlauncher_"
        + sha256(
            json.dumps(
                launcher,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()[:20]
    )
    launcher["launcher_receipt_id"] = launcher_id
    launcher_receipt_relative = (
        f"data/research/wizard_proof_launcher_receipts/2026-08-11/{launcher_id}.json"
    )
    launcher_receipt_path = root / launcher_receipt_relative
    _write_json(launcher_receipt_path, launcher)
    launcher["immutable_launcher_receipt_path"] = launcher_receipt_relative
    launcher["immutable_launcher_receipt_sha256"] = sha256(
        launcher_receipt_path.read_bytes()
    ).hexdigest()
    _write_json(active / "corrective_wizard_proof_launcher_status.json", launcher)
    _write_json(
        active / "corrective_wizard_proof_scheduler_status.json",
        {
            "schema_version": "thewiz.corrective_wizard_proof_scheduler.v1",
            "capture_manifest_enforced": True,
            "capture_manifest_id": manifest_id,
            "capture_manifest_immutable_path": immutable_relative,
            "capture_manifest_immutable_sha256": immutable_digest,
            "capture_manifest_candidate_id": manifest_id,
            "capture_manifest_candidate_immutable_path": immutable_relative,
            "capture_manifest_candidate_immutable_sha256": immutable_digest,
            "capture_manifest_candidate_binding_valid": True,
            "capture_manifest_candidate_source_binding_valid": True,
            "capture_manifest_continuity_status": "PASS_PRIOR_UNRESOLVED_COHORT_MATCH",
            "capture_manifest_continuity_valid": True,
            "capture_manifest_drift_detected": False,
        },
    )
    _write_json(
        active / "wizard_credit_budget_contract.json",
        {
            "status": "PASS",
            "exact_mode_proof_credit_ceiling": 60,
            "copula_behavioral_credit_ceiling": 8,
            "ou_v3_prospective_credit_ceiling": 8,
            "headroom_after_reserve": 524,
            "wizard_daily_credit_reset_utc": "00:00",
            "runtime_credit_preflight_still_required": True,
            "order_submission_included": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    env_path = root / ".env.local"
    env_path.write_text("CRYPTO_WIZARDS_API_KEY=test-key\n", encoding="utf-8")
    env_path.chmod(0o600)
    _build_browser_auth(root)

    plist_path = tmp_path / f"{LAUNCH_AGENT_LABEL}.plist"
    payload = {
        "Label": LAUNCH_AGENT_LABEL,
        "ProgramArguments": [
            str(python),
            "-m",
            LAUNCHER_MODULE,
            "--execute",
        ],
        "WorkingDirectory": str(root),
        "StartInterval": 600,
        "EnvironmentVariables": {
            "PYTHONPATH": str(root / "src"),
            "TMPDIR": str(runtime_temp),
            "TMP": str(runtime_temp),
            "TEMP": str(runtime_temp),
        },
    }
    with plist_path.open("wb") as handle:
        plistlib.dump(payload, handle)
    return root, plist_path


def _run(root: Path, plist_path: Path):
    return build_corrective_wizard_reset_readiness(
        root=root,
        now=NOW,
        launch_agent_path=plist_path,
        launch_agent_loaded=True,
        minimum_runtime_temp_free_bytes=1,
    )


def test_reset_readiness_passes_without_granting_authority(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)

    result = _run(root, plist_path)

    assert result.summary["status"] == "PASS_RESET_AUTOMATION_READY"
    assert result.summary["checks_passed"] == result.summary["checks_total"]
    assert result.summary["capture_window_status"] == "WAITING_FOR_RESET"
    assert result.summary["scheduler_manifest_continuity_valid"] is True
    assert result.summary["manifest_source_current_drift_valid"] is True
    assert result.summary["manifest_source_current_drift_classification"] == "EXACT_MATCH"
    assert result.summary["credential_source"] == ".env.local"
    assert result.summary["launcher_receipt_binding_valid"] is True
    assert result.summary["launcher_receipt_id"].startswith("wizardlauncher_")
    assert result.summary["browser_auth_valid_at_capture_window"] is True
    assert result.summary["browser_auth_receipt_id"].startswith("wizardbrowserreadiness_")
    assert result.summary["candidate_promotion_authority"] is False
    assert result.summary["order_submission_included"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    assert result.paths["immutable_receipt"].is_file()
    validation = validate_wizard_reset_readiness_receipt(root=root)
    assert validation["status"] == "PASS"
    assert validation["receipt_id"] == result.summary["receipt_id"]
    assert validation["state_sha256"] == wizard_reset_readiness_state_sha256(
        json.loads(result.paths["status"].read_text(encoding="utf-8"))
    )


def test_reset_contract_allows_three_scheduler_intervals_for_lock_retry() -> None:
    reset_at = datetime(2026, 8, 15, 0, 0, tzinfo=UTC)

    assert corrective_wizard_reset_readiness._reset_contract_is_valid(
        reset_at=reset_at,
        checked_at=datetime(2026, 8, 15, 0, 25, tzinfo=UTC),
        interval_seconds=600,
    )
    assert not corrective_wizard_reset_readiness._reset_contract_is_valid(
        reset_at=reset_at,
        checked_at=datetime(2026, 8, 15, 0, 31, tzinfo=UTC),
        interval_seconds=600,
    )


def test_reset_readiness_requires_observation_manifest_and_execution_reset_roles(
    tmp_path: Path,
) -> None:
    root, plist_path = _prepare_root(tmp_path)
    active = root / "reports" / "active"
    legacy_path = active / "corrective_wizard_proof_scheduler_status.json"
    observation_path = active / "corrective_wizard_proof_scheduler_observation_status.json"
    execution_path = active / "corrective_wizard_proof_scheduler_execution_status.json"
    observation = json.loads(legacy_path.read_text(encoding="utf-8"))
    observation.update({"execution_requested": False, "pointer_role": "observation"})
    _write_json(observation_path, observation)

    execution_receipt = {
        "schema_version": "thewiz.corrective_wizard_proof_scheduler.v1",
        "attempt_date_utc": "2026-08-11",
        "started_at_utc": "2026-08-11T15:00:00+00:00",
        "execution_requested": True,
        "external_attempt_made": True,
        "next_external_attempt_eligible_at": RESET,
        "receipt_id": "wizardproof_0123456789abcdef0123",
        **{
            key: False
            for key in (
                "candidate_promotion_authority",
                "order_submission_included",
                "testnet_order_authority",
                "live_trading_authorized",
            )
        },
    }
    execution_receipt_path = active / "wizard_proof_scheduler_receipts" / "execution.json"
    _write_json(execution_receipt_path, execution_receipt)
    execution = {
        **execution_receipt,
        "pointer_role": "execution",
        "pointer_source_path": str(execution_receipt_path.relative_to(root)),
    }
    _write_json(execution_path, execution)
    _write_json(
        legacy_path,
        {
            "schema_version": "thewiz.corrective_wizard_proof_scheduler.v1",
            "execution_requested": False,
            "capture_manifest_continuity_valid": False,
            "capture_manifest_drift_detected": True,
        },
    )

    result = _run(root, plist_path)

    assert result.summary["status"] == "PASS_RESET_AUTOMATION_READY"
    assert result.summary["scheduler_role_split_enforced"] is True
    assert result.summary["scheduler_execution_reset_binding_valid"] is True
    checks = list(csv.DictReader(result.paths["checks"].open(encoding="utf-8")))
    continuity = next(
        row for row in checks if row["check"] == "scheduler_capture_manifest_continuity"
    )
    assert continuity["evidence_path"].endswith(
        "corrective_wizard_proof_scheduler_observation_status.json"
    )


def test_reset_readiness_blocks_when_observation_pointer_replaces_execution_role(
    tmp_path: Path,
) -> None:
    root, plist_path = _prepare_root(tmp_path)
    active = root / "reports" / "active"
    legacy = json.loads(
        (active / "corrective_wizard_proof_scheduler_status.json").read_text(encoding="utf-8")
    )
    observation = {**legacy, "execution_requested": False, "pointer_role": "observation"}
    _write_json(active / "corrective_wizard_proof_scheduler_observation_status.json", observation)
    _write_json(active / "corrective_wizard_proof_scheduler_execution_status.json", observation)

    result = _run(root, plist_path)

    assert result.summary["status"] == "BLOCKED_RESET_AUTOMATION"
    assert result.summary["scheduler_execution_reset_binding_valid"] is False
    assert "wizard_scheduler_execution_reset_binding_invalid" in result.summary["blockers"]


def test_reset_readiness_blocks_execution_receipt_with_missing_authority_field(
    tmp_path: Path,
) -> None:
    root, plist_path = _prepare_root(tmp_path)
    active = root / "reports" / "active"
    legacy_path = active / "corrective_wizard_proof_scheduler_status.json"
    observation = json.loads(legacy_path.read_text(encoding="utf-8"))
    observation.update({"execution_requested": False, "pointer_role": "observation"})
    _write_json(active / "corrective_wizard_proof_scheduler_observation_status.json", observation)
    execution_receipt = {
        "schema_version": "thewiz.corrective_wizard_proof_scheduler.v1",
        "attempt_date_utc": "2026-08-11",
        "started_at_utc": "2026-08-11T15:00:00+00:00",
        "execution_requested": True,
        "external_attempt_made": True,
        "next_external_attempt_eligible_at": RESET,
        "receipt_id": "wizardproof_0123456789abcdef0123",
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
    }
    source = active / "wizard_proof_scheduler_receipts" / "missing_authority.json"
    _write_json(source, execution_receipt)
    _write_json(
        active / "corrective_wizard_proof_scheduler_execution_status.json",
        {
            **execution_receipt,
            "pointer_role": "execution",
            "pointer_source_path": str(source.relative_to(root)),
        },
    )

    result = _run(root, plist_path)

    assert result.summary["scheduler_execution_reset_binding_valid"] is False
    assert "wizard_scheduler_execution_reset_binding_invalid" in result.summary["blockers"]


def test_reset_readiness_blocks_ou_v5_unresolved_cross_day_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plist_path = _prepare_root(tmp_path)
    manifest_path = root / "reports/active/corrective_wizard_next_capture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pending_calls"] = 14
    manifest["planned_credits"] = 20
    manifest["lane_totals"]["ou_v5_holdout"] = {"calls": 1, "credits": 2}
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(
        corrective_wizard_reset_readiness,
        "audit_ou_v5_capture_readiness",
        lambda **_: {
            "status": "BLOCKED",
            "blockers": [
                (
                    "ou_v5_call_attempt_already_registered_without_response:"
                    "wizcall_example:2026-08-10"
                )
            ],
            "required_cells": 1,
            "responses_available": 0,
            "missing_cells": 1,
            "unresolved_call_intents": 1,
        },
    )

    result = _run(root, plist_path)

    assert result.summary["status"] == "BLOCKED_RESET_AUTOMATION"
    assert result.summary["ou_v5_unresolved_call_intents"] == 1
    assert (
        "ou_v5_call_attempt_already_registered_without_response:wizcall_example:2026-08-10"
    ) in result.summary["blockers"]


def test_reset_readiness_validator_rejects_active_receipt_without_immutable_twin(
    tmp_path: Path,
) -> None:
    root, plist_path = _prepare_root(tmp_path)
    result = _run(root, plist_path)
    active = json.loads(result.paths["status"].read_text(encoding="utf-8"))
    active["launch_agent_loaded"] = False
    material = {key: value for key, value in active.items() if key != "receipt_id"}
    active["receipt_id"] = (
        "wizardresetreadiness_"
        + sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
    )
    _write_json(result.paths["status"], active)

    validation = validate_wizard_reset_readiness_receipt(root=root)

    assert validation["status"] == "BLOCKED"
    assert "wizard_reset_readiness_immutable_receipt_missing" in validation["blockers"]


def test_reset_validator_rejects_tampered_bound_browser_receipt(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)
    result = _run(root, plist_path)
    status = json.loads(result.paths["status"].read_text(encoding="utf-8"))
    receipt_path = root / status["browser_auth_receipt_path"]
    receipt_path.chmod(0o600)
    receipt_path.write_text("{}\n", encoding="utf-8")

    validation = validate_wizard_reset_readiness_receipt(root=root)

    assert validation["status"] == "BLOCKED"
    assert "wizard_reset_readiness_browser_auth_binding_invalid" in validation["blockers"]


def test_reset_validator_rejects_tampered_bound_browser_source(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)
    result = _run(root, plist_path)
    status = json.loads(result.paths["status"].read_text(encoding="utf-8"))
    browser_receipt = json.loads(
        (root / status["browser_auth_receipt_path"]).read_text(encoding="utf-8")
    )
    source = root / browser_receipt["selected_evidence"][0]["source_path"]
    source.write_text("{}\n", encoding="utf-8")

    validation = validate_wizard_reset_readiness_receipt(root=root)

    assert validation["status"] == "BLOCKED"
    assert "wizard_reset_readiness_browser_auth_binding_invalid" in validation["blockers"]


def test_reset_validator_rejects_self_hashed_receipt_without_browser_binding(
    tmp_path: Path,
) -> None:
    root, plist_path = _prepare_root(tmp_path)
    result = _run(root, plist_path)
    payload = json.loads(result.paths["status"].read_text(encoding="utf-8"))
    payload.pop("receipt_id")
    for field in (
        "browser_auth_required_at_utc",
        "browser_auth_valid_at_capture_window",
        "browser_auth_valid_until_utc",
        "browser_auth_receipt_id",
        "browser_auth_receipt_path",
        "browser_auth_receipt_sha256",
    ):
        payload.pop(field)
    receipt_id = (
        "wizardresetreadiness_"
        + sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
    )
    payload["receipt_id"] = receipt_id
    _write_json(result.paths["status"], payload)
    _write_json(
        root / "data" / "research" / "wizard_reset_readiness" / f"{receipt_id}.json",
        payload,
    )

    validation = validate_wizard_reset_readiness_receipt(root=root)

    assert validation["status"] == "BLOCKED"
    assert "wizard_reset_readiness_browser_auth_binding_invalid" in validation["blockers"]


def test_reset_readiness_is_idempotent_for_same_point_in_time(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)

    first = _run(root, plist_path)
    second = _run(root, plist_path)

    assert first.summary["receipt_id"] == second.summary["receipt_id"]
    assert first.paths["immutable_receipt"] == second.paths["immutable_receipt"]


def test_reset_readiness_blocks_unloaded_launch_agent(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)

    result = build_corrective_wizard_reset_readiness(
        root=root,
        now=NOW,
        launch_agent_path=plist_path,
        launch_agent_loaded=False,
    )

    assert result.summary["status"] == "BLOCKED_RESET_AUTOMATION"
    assert "wizard_proof_launch_agent_not_loaded" in result.summary["blockers"]


def test_reset_readiness_blocks_browser_auth_expiring_before_window(
    tmp_path: Path,
) -> None:
    root, plist_path = _prepare_root(tmp_path)
    _build_browser_auth(root, max_age_hours=8.0)

    result = _run(root, plist_path)

    assert result.summary["browser_auth_valid_at_capture_window"] is False
    assert "wizard_browser_auth_not_valid_at_capture_window" in result.summary["blockers"]


def test_reset_readiness_blocks_missing_browser_auth_pointer(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)
    (root / "reports" / "active" / "wizard_browser_auth_readiness.json").unlink()

    result = _run(root, plist_path)

    assert result.summary["browser_auth_valid_at_capture_window"] is False
    assert "wizard_browser_auth_not_valid_at_capture_window" in result.summary["blockers"]


def test_reset_readiness_blocks_tampered_browser_auth_source(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)
    status = json.loads(
        (root / "reports" / "active" / "wizard_browser_auth_readiness.json").read_text(
            encoding="utf-8"
        )
    )
    source = root / status["selected_evidence"][0]["source_path"]
    source.write_text("{}\n", encoding="utf-8")

    result = _run(root, plist_path)

    assert result.summary["browser_auth_valid_at_capture_window"] is False
    assert "wizard_browser_auth_not_valid_at_capture_window" in result.summary["blockers"]


def test_reset_readiness_blocks_wrong_command(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)
    with plist_path.open("rb") as handle:
        payload = plistlib.load(handle)
    payload["ProgramArguments"][-1] = "--plan"
    with plist_path.open("wb") as handle:
        plistlib.dump(payload, handle)

    result = _run(root, plist_path)

    assert "wizard_proof_launch_agent_command_mismatch" in result.summary["blockers"]


def test_reset_readiness_blocks_system_temp_binding(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)
    with plist_path.open("rb") as handle:
        payload = plistlib.load(handle)
    payload["EnvironmentVariables"]["TMPDIR"] = "/tmp"
    with plist_path.open("wb") as handle:
        plistlib.dump(payload, handle)

    result = _run(root, plist_path)

    assert "wizard_proof_launch_agent_temp_binding_mismatch" in result.summary["blockers"]


def test_reset_readiness_blocks_missing_workspace_temp_directory(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)
    (root / ".runtime_tmp").rmdir()

    result = _run(root, plist_path)

    assert "wizard_proof_workspace_temp_not_viable" in result.summary["blockers"]


def test_reset_readiness_blocks_insufficient_workspace_temp_capacity(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)

    result = build_corrective_wizard_reset_readiness(
        root=root,
        now=NOW,
        launch_agent_path=plist_path,
        launch_agent_loaded=True,
        minimum_runtime_temp_free_bytes=10**30,
    )

    assert "wizard_proof_workspace_temp_not_viable" in result.summary["blockers"]


def test_reset_readiness_blocks_stale_launcher_heartbeat(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)
    launcher_path = root / "reports" / "active" / "corrective_wizard_proof_launcher_status.json"
    launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    launcher["checked_at_utc"] = "2026-08-11T14:00:00+00:00"
    _write_json(launcher_path, launcher)

    result = _run(root, plist_path)

    assert "wizard_proof_launcher_heartbeat_or_reset_binding_invalid" in result.summary["blockers"]


def test_reset_readiness_blocks_tampered_launcher_receipt(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)
    launcher_path = root / "reports" / "active" / "corrective_wizard_proof_launcher_status.json"
    launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
    (root / launcher["immutable_launcher_receipt_path"]).write_text(
        "{}\n",
        encoding="utf-8",
    )

    result = _run(root, plist_path)

    assert result.summary["launcher_receipt_binding_valid"] is False
    assert "wizard_proof_launcher_immutable_receipt_invalid" in result.summary["blockers"]


def test_reset_readiness_blocks_scheduler_manifest_drift(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)
    scheduler_path = root / "reports" / "active" / "corrective_wizard_proof_scheduler_status.json"
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    scheduler["capture_manifest_candidate_id"] = "wizardcapture_drifted"
    scheduler["capture_manifest_continuity_status"] = "BLOCKED_COHORT_DRIFT"
    scheduler["capture_manifest_continuity_valid"] = False
    scheduler["capture_manifest_drift_detected"] = True
    _write_json(scheduler_path, scheduler)

    result = _run(root, plist_path)

    assert result.summary["scheduler_manifest_continuity_valid"] is False
    assert "wizard_scheduler_capture_manifest_continuity_invalid" in result.summary["blockers"]


def test_reset_readiness_blocks_tampered_immutable_manifest(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)
    manifest = json.loads(
        (root / "reports" / "active" / "corrective_wizard_next_capture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    (root / manifest["immutable_manifest_path"]).write_text("tampered", encoding="utf-8")

    result = _run(root, plist_path)

    assert "frozen_capture_manifest_binding_invalid" in result.summary["blockers"]


def test_reset_readiness_blocks_tampered_manifest_source_snapshot(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)
    manifest = json.loads(
        (root / "reports" / "active" / "corrective_wizard_next_capture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source_receipt = json.loads((root / manifest["source_receipt_path"]).read_text())
    snapshot = root / source_receipt["source_artifacts"][0]["snapshot_path"]
    snapshot.write_text("call_id\nchanged\n", encoding="utf-8")

    result = _run(root, plist_path)

    assert "frozen_capture_manifest_source_lineage_invalid" in result.summary["blockers"]
    assert result.summary["manifest_source_lineage_valid"] is False


def test_reset_readiness_blocks_mutable_source_list_rewrite(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)
    manifest_path = root / "reports" / "active" / "corrective_wizard_next_capture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = root / manifest["source_artifacts"][0]["path"]
    source.write_text("call_id\nrewritten\n", encoding="utf-8")
    manifest["source_artifacts"][0]["sha256"] = sha256(source.read_bytes()).hexdigest()
    _write_json(manifest_path, manifest)

    result = _run(root, plist_path)

    assert "frozen_capture_manifest_source_lineage_invalid" in result.summary["blockers"]
    assert result.summary["manifest_source_lineage_valid"] is False


def test_reset_readiness_blocks_current_scientific_source_drift(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)
    manifest_path = root / "reports" / "active" / "corrective_wizard_next_capture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = root / manifest["source_artifacts"][0]["path"]
    source.write_text("call_id\nscientific-change\n", encoding="utf-8")

    result = _run(root, plist_path)

    assert result.summary["manifest_source_lineage_valid"] is True
    assert result.summary["manifest_source_current_drift_valid"] is False
    assert (
        result.summary["manifest_source_current_drift_classification"]
        == "SCIENTIFIC_OR_STRUCTURAL_DRIFT"
    )
    assert "wizard_capture_source_scientific_or_structural_drift" in result.summary["blockers"]


def test_reset_readiness_blocks_source_path_traversal(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)
    manifest_path = root / "reports" / "active" / "corrective_wizard_next_capture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outside = root.parent / "outside-source.csv"
    outside.write_text("call_id\noutside\n", encoding="utf-8")
    manifest["source_artifacts"] = [
        {
            "path": "../outside-source.csv",
            "sha256": sha256(outside.read_bytes()).hexdigest(),
        }
    ]
    _write_json(manifest_path, manifest)

    result = _run(root, plist_path)

    assert "frozen_capture_manifest_source_lineage_invalid" in result.summary["blockers"]
    assert result.summary["manifest_source_lineage_valid"] is False


def test_reset_readiness_blocks_missing_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plist_path = _prepare_root(tmp_path)
    (root / ".env.local").unlink()
    monkeypatch.delenv("CRYPTO_WIZARDS_API_KEY", raising=False)

    result = _run(root, plist_path)

    assert "CRYPTO_WIZARDS_API_KEY_missing" in result.summary["blockers"]


def test_reset_readiness_blocks_process_only_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, plist_path = _prepare_root(tmp_path)
    (root / ".env.local").unlink()
    monkeypatch.setenv("CRYPTO_WIZARDS_API_KEY", "interactive-only-key")

    result = _run(root, plist_path)

    assert result.summary["status"] == "BLOCKED_RESET_AUTOMATION"
    assert result.summary["credential_source"] == "process_environment_only"
    assert "wizard_api_credential_not_persisted_for_launch_agent" in result.summary["blockers"]


def test_reset_readiness_blocks_insufficient_credit_budget(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)
    budget_path = root / "reports" / "active" / "wizard_credit_budget_contract.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    budget["headroom_after_reserve"] = 10
    _write_json(budget_path, budget)

    result = _run(root, plist_path)

    assert "wizard_shared_credit_budget_not_ready" in result.summary["blockers"]


def test_reset_readiness_rejects_immutable_receipt_collision(tmp_path: Path) -> None:
    root, plist_path = _prepare_root(tmp_path)
    first = _run(root, plist_path)
    first.paths["immutable_receipt"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="immutable readiness receipt collision"):
        _run(root, plist_path)
