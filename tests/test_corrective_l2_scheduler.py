from __future__ import annotations

import json
import os
import plistlib
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration import corrective_l2_scheduler, effect_authority
from quant_platform.orchestration.corrective_l2_scheduler import (
    _inventory_refresh_blocker,
    _launch_agent_plist,
    _source_fingerprint,
    materialize_corrective_candidate_funding_assets,
    run_corrective_l2_capture,
    run_post_window_readiness_refresh,
    validate_l2_capture_receipt,
    validate_post_window_readiness_receipt,
)
from quant_platform.orchestration.corrective_runtime import (
    SCHEDULER_BOOTSTRAP_MODULE,
    launch_agent_runtime_environment,
    scheduler_contract,
)
from quant_platform.orchestration.corrective_scheduler_supervisor import (
    supervise_scheduler_run,
)
from tests.pair_cost_bundle_support import publish_valid_pair_cost_bundle

NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)


def _prepare_supervisor_root(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = root / "src" / "quant_platform"
    package.mkdir(parents=True)
    (package / "fixture.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\nname='l2-lattice-fixture'\n",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    contract = scheduler_contract("hyperliquid_l2")
    for name, value in launch_agent_runtime_environment(
        root,
        contract=contract,
    ).items():
        monkeypatch.setenv(name, value)


def _run_isolated_supervisor(**kwargs):
    test_authority = effect_authority._CURRENT_PUBLICATION_AUTHORITY.set(None)
    try:
        return supervise_scheduler_run(**kwargs)
    finally:
        effect_authority._CURRENT_PUBLICATION_AUTHORITY.reset(test_authority)


def _complete_capture_summary() -> dict[str, object]:
    return {
        "receipt_id": "l2receipt_fixture",
        "capture_blockers": [],
        "eligible_pairs": 1,
        "collector_summary": {"pairs": 1},
        "strict_pair_cost_eligible": 1,
        "strict_pair_cost_ready": 1,
        "strict_pair_cost_acceptance_status": "PASS",
        "registered_contract_candidates": 1,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _complete_post_window_summary() -> dict[str, object]:
    return {
        "status": "PASS_LOCAL_READINESS_REFRESH",
        "source_l2_receipt_id": "l2receipt_fixture",
        "eligible_pairs": 1,
        "ready_pairs": 1,
        "collecting_pairs": 0,
        "registered_contract_candidates": 1,
        "refresh_executed": True,
        "registered_gate_refresh_executed": True,
        "stage4_handoff_refresh_executed": True,
        "registered_gate_status": "PASS_REGISTERED_RERUN_ACCOUNTED",
        "stage4_handoff_status": "PASS_STAGE4_HANDOFF_READY",
        "stage4_handoff_validation_status": "PASS",
        "blockers": [],
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _complete_post_window_validation() -> dict[str, object]:
    return {
        "status": "PASS",
        "blockers": [],
        "source_l2_receipt_id": "l2receipt_fixture",
    }


def test_source_fingerprint_uses_change_token_for_large_append_only_ledger(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ledger = tmp_path / "data" / "research" / "hypothesis_ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"row":1}\n', encoding="utf-8")
    hashed_paths: list[Path] = []
    original_hash = corrective_l2_scheduler._file_sha256

    def tracked_hash(path: Path) -> str:
        hashed_paths.append(path)
        assert path != ledger
        return original_hash(path)

    monkeypatch.setattr(corrective_l2_scheduler, "_file_sha256", tracked_hash)
    before = _source_fingerprint(tmp_path)
    ledger.write_text('{"row":1}\n{"row":2}\n', encoding="utf-8")
    after = _source_fingerprint(tmp_path)

    key = "data/research/hypothesis_ledger.jsonl"
    assert before[key].startswith("stat:")
    assert after[key].startswith("stat:")
    assert before[key] != after[key]
    assert ledger not in hashed_paths


def _write_candidate_inputs(root: Path) -> None:
    active = root / "reports" / "active"
    processed = root / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    pd.DataFrame([{"experiment_id": "exp-1", "confirmation_role": "near_miss_remediation"}]).to_csv(
        active / "current_hypothesis_batch.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "experiment_id": "exp-1",
                "pair_group_key": "binance|daily|ETH|WIF",
                "pair": "ETH-USD-WIF-USD",
                "asset_x": "ETH",
                "asset_y": "WIF",
                "overall_research_rank": 1,
            }
        ]
    ).to_csv(active / "current_wizard_hyperliquid_failure_attribution.csv", index=False)
    pd.DataFrame([{"asset": "ETH", "tradable": True}, {"asset": "WIF", "tradable": True}]).to_csv(
        processed / "hyperliquid_market_context.csv", index=False
    )
    pd.DataFrame(
        [
            {"asset": "ETH", "funding_status": "COMPLETE", "funding_rows": 500},
            {"asset": "WIF", "funding_status": "COMPLETE", "funding_rows": 500},
        ]
    ).to_csv(active / "current_wizard_hyperliquid_funding_asset_results.csv", index=False)


def _write_mapping_contract(root: Path, *, checked_at: datetime) -> None:
    active = root / "reports" / "active"
    (active / "exhaustive_wizard_hyperliquid_run_manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "pair_group_key": "binance|daily|ETH|WIF",
                "mapping_refresh_id": "hlmap_existing",
                "current_inventory_checked_at": checked_at.isoformat(),
                "current_pair_ready": True,
            }
        ]
    ).to_csv(active / "exhaustive_wizard_hyperliquid_mapping_refresh.csv", index=False)


def _inventory_frame(now: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asset": "ETH",
                "tradable_perp": True,
                "checked_at_utc": now.isoformat(),
                "fetch_blocker": "",
            }
        ]
    )


def _capture_ready_registered_pair(tmp_path, monkeypatch) -> CommandResult:
    _write_candidate_inputs(tmp_path)
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    candidate_path = active / "corrective_l2_capture_candidates.csv"
    pd.DataFrame(
        [
            {
                "experiment_id": "registered-exp",
                "pair_group_key": "binance|daily|ETH|WIF",
                "pair": "ETH-USD-WIF-USD",
                "asset_x": "ETH",
                "asset_y": "WIF",
                "overall_research_rank": 1,
                "confirmation_role": "registered_research_rerun",
                "source_family": "registered_rerun_contract",
                "semantic_hypothesis_id": "semantic-1",
                "registered_contract_id": "registeredrerun_test",
                "registered_contract_candidate": True,
                "stage_two_candidate": True,
                "collection_eligible": True,
                "blocker": "",
                "evidence_path": "fixture",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ]
    ).to_csv(candidate_path, index=False)
    rows = []
    for offset in range(12):
        timestamp = NOW - pd.Timedelta(minutes=10 * offset)
        for asset in ("ETH", "WIF"):
            rows.append(
                {
                    "pair": "ETH-USD-WIF-USD",
                    "asset": asset,
                    "source_timestamp": timestamp.isoformat(),
                    "buy_complete": True,
                    "sell_complete": True,
                }
            )
    pd.DataFrame(rows).to_csv(
        processed / "hyperliquid_l2_slippage_samples.csv", index=False
    )
    monkeypatch.setattr(
        corrective_l2_scheduler,
        "build_l2_capture_candidate_set",
        lambda **_: {
            "path": candidate_path,
            "registered_hypotheses": 1,
            "registered_contract_candidates": 1,
            "candidate_pairs": 1,
            "eligible_pairs": 1,
            "stage_two_candidate_pairs": 1,
        },
    )
    capture = run_corrective_l2_capture(
        root=tmp_path,
        now=NOW,
        collector=lambda **_: CommandResult(paths={}, summary={"pairs": 1}),
    )
    bundle = publish_valid_pair_cost_bundle(
        root=tmp_path,
        now=NOW,
        specs=[
            {
                "experiment_id": "registered-exp",
                "pair_group_key": "binance|daily|ETH|WIF",
                "pair": "ETH-USD-WIF-USD",
                "asset_x": "ETH",
                "asset_y": "WIF",
                "confirmation_role": "registered_research_rerun",
                "source_family": "registered_rerun_contract",
                "semantic_hypothesis_id": "semantic-1",
                "registered_contract_id": "registeredrerun_test",
                "registered_contract_candidate": True,
            }
        ],
    )
    candidate_path.write_bytes(bundle["candidate_set"].read_bytes())
    active_status_path = active / "corrective_l2_capture_status.json"
    active_status = json.loads(active_status_path.read_text(encoding="utf-8"))
    receipt_path = tmp_path / active_status["receipt_path"]
    manifest = json.loads(bundle["bundle"].read_text(encoding="utf-8"))
    active_status.update(
        {
            "registered_contract_candidates": 1,
            "strict_pair_cost_eligible": 1,
            "strict_pair_cost_ready": 1,
            "strict_pair_cost_acceptance_status": "PASS",
            "post_window_ready_pairs": 1,
            "post_window_collecting_pairs": 0,
            "candidate_evidence_sha256": sha256(candidate_path.read_bytes()).hexdigest(),
            "pair_cost_bundle_id": manifest["bundle_id"],
            "pair_cost_bundle_manifest_path": str(bundle["bundle"].relative_to(tmp_path)),
            "pair_cost_bundle_manifest_sha256": sha256(bundle["bundle"].read_bytes()).hexdigest(),
            "pair_cost_bundle_pointer_path": str(
                bundle["pointer_snapshot"].relative_to(tmp_path)
            ),
            "pair_cost_bundle_pointer_sha256": sha256(
                bundle["pointer_snapshot"].read_bytes()
            ).hexdigest(),
            "active_pair_cost_bundle_pointer_path": str(
                bundle["pointer"].relative_to(tmp_path)
            ),
            "active_pair_cost_bundle_pointer_sha256": sha256(
                bundle["pointer"].read_bytes()
            ).hexdigest(),
            "pair_cost_model_snapshot_path": str(bundle["models"].relative_to(tmp_path)),
            "pair_cost_model_snapshot_sha256": sha256(bundle["models"].read_bytes()).hexdigest(),
        }
    )
    receipt_relative = active_status.pop("receipt_path")
    active_status.pop("receipt_id")
    active_status = corrective_l2_scheduler._build_l2_acceptance_summary(
        active_status
    )
    active_status["receipt_id"] = "l2receipt_" + sha256(
        json.dumps(active_status, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    receipt_path.write_text(
        json.dumps(active_status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    active_status_path.write_text(
        json.dumps(
            {**active_status, "receipt_path": receipt_relative},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    capture.summary.clear()
    capture.summary.update(active_status)
    capture.paths.update(
        {
            "pair_cost_bundle_manifest": bundle["bundle"],
            "pair_cost_bundle_pointer": bundle["pointer"],
            "pair_cost_bundle_pointer_snapshot": bundle["pointer_snapshot"],
            "pair_cost_model_snapshot": bundle["models"],
        }
    )
    return capture


def test_capture_is_public_read_only_and_writes_receipt(tmp_path, monkeypatch):
    _write_candidate_inputs(tmp_path)
    calls = []
    original_cost_builder = corrective_l2_scheduler.build_pair_cost_stress_surfaces

    def cost_builder(**kwargs):
        assert (tmp_path / "reports" / "active" / ".corrective_l2_capture.lock").exists()
        return original_cost_builder(**kwargs)

    monkeypatch.setattr(corrective_l2_scheduler, "build_pair_cost_stress_surfaces", cost_builder)

    def collector(**kwargs):
        calls.append(kwargs)
        return CommandResult(paths={}, summary={"pairs": 1, "slippage_models_ready": 0})

    result = run_corrective_l2_capture(root=tmp_path, now=NOW, collector=collector)

    assert result.summary["status"] == "BLOCKED"
    assert result.summary["capture_status"] == "PASS"
    assert result.summary["l2_collection_state"] == "COLLECTED"
    assert result.summary["l2_validation_state"] == "PENDING"
    assert result.summary["l2_acceptance_state"] == "BLOCKED"
    assert result.summary["l2_authorization_state"] == "PENDING"
    assert result.summary["l2_acceptance_lattice_complete"] is False
    assert result.summary["l2_terminal_slot_credit_eligible"] is False
    assert "l2_strict_cost_acceptance_not_pass" in result.summary["blockers"]
    assert result.summary["eligible_pairs"] == 1
    assert result.summary["evidence_evaluated_at_utc"] == NOW.isoformat()
    assert result.summary["order_submission_included"] is False
    assert result.summary["testnet_order_authority"] is False
    assert calls[0]["notionals"] == (1000.0,)
    assert calls[0]["min_samples"] == 12
    assert result.paths["capture_receipt"].exists()
    assert result.summary["post_window_ready_pairs"] == 0
    assert result.summary["post_window_candidate_refresh_executed"] is False
    assert result.paths["post_window_transition"].exists()
    assert result.summary["registered_pair_cost_models"] == 1
    assert result.summary["strict_pair_cost_ready"] == 0
    assert result.summary["stage_two_pair_cost_eligible"] == 1
    assert result.summary["stage_two_pair_cost_ready"] == 0
    assert result.summary["stage_two_pair_cost_acceptance_status"] == "BLOCKED"
    assert result.summary["pair_cost_bundle_id"].startswith("l2costbundle_")
    assert len(result.summary["pair_cost_bundle_manifest_sha256"]) == 64
    assert len(result.summary["pair_cost_bundle_pointer_sha256"]) == 64
    assert result.paths["pair_cost_bundle_manifest"].is_file()
    assert result.paths["pair_cost_bundle_pointer"].is_file()
    assert result.paths["pair_cost_model_snapshot"].is_file()
    assert result.paths["pair_cost_models"].exists()
    assert result.paths["pair_cost_stress"].exists()
    assert not (tmp_path / "reports" / "active" / ".corrective_l2_capture.lock").exists()


def test_complete_supervised_l2_result_contains_full_monotone_lattice() -> None:
    result = corrective_l2_scheduler._build_supervised_l2_result(
        capture=CommandResult(paths={}, summary=_complete_capture_summary()),
        readiness=CommandResult(
            paths={},
            summary=_complete_post_window_summary(),
        ),
        readiness_validation=_complete_post_window_validation(),
    )
    summary = result["summary"]

    assert summary["status"] == "PASS"
    assert summary["l2_acceptance_lattice"] == [
        "COLLECTED",
        "VALIDATED",
        "ACCEPTED",
        "AUTHORIZED",
    ]
    assert summary["l2_collection_state"] == "COLLECTED"
    assert summary["l2_validation_state"] == "VALIDATED"
    assert summary["l2_acceptance_state"] == "ACCEPTED"
    assert summary["l2_authorization_state"] == "AUTHORIZED"
    assert summary["l2_highest_state"] == "AUTHORIZED"
    assert summary["l2_acceptance_lattice_complete"] is True
    assert summary["l2_terminal_slot_credit_eligible"] is True
    assert summary["post_window_readiness_validation"]["status"] == "PASS"
    assert "post_window_readiness_validation" not in result
    assert summary["testnet_order_authority"] is False
    assert summary["live_trading_authorized"] is False


def test_l2_lattice_cannot_validate_before_collection() -> None:
    capture = _complete_capture_summary()
    capture["capture_blockers"] = ["collector_failed"]

    summary = corrective_l2_scheduler._build_l2_acceptance_summary(
        capture,
        post_window_summary=_complete_post_window_summary(),
        post_window_validation=_complete_post_window_validation(),
    )

    assert summary["status"] == "BLOCKED"
    assert summary["l2_collection_state"] == "BLOCKED"
    assert summary["l2_validation_state"] == "BLOCKED"
    assert summary["l2_acceptance_state"] == "BLOCKED"
    assert summary["l2_authorization_state"] == "BLOCKED"
    assert summary["l2_highest_state"] == "BLOCKED"
    assert summary["l2_terminal_slot_credit_eligible"] is False


def test_complete_l2_lattice_is_credited_by_scheduler_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_supervisor_root(tmp_path, monkeypatch)

    result = _run_isolated_supervisor(
        root=tmp_path,
        contract_key="hyperliquid_l2",
        publication_scope="public_l2",
        callback=lambda: corrective_l2_scheduler._build_supervised_l2_result(
            capture=CommandResult(paths={}, summary=_complete_capture_summary()),
            readiness=CommandResult(
                paths={},
                summary=_complete_post_window_summary(),
            ),
            readiness_validation=_complete_post_window_validation(),
        ),
        now=NOW,
    )

    assert result.exit_code == 0
    assert result.result_summary["status"] == "PASS"
    assert result.terminal_receipt["terminal_status"] == "PASS"
    assert result.terminal_receipt["intended_slot_credit"] is True


@pytest.mark.parametrize(
    ("failure", "expected_blocker"),
    [
        ("strict_cost", "l2_strict_cost_acceptance_not_pass"),
        (
            "post_window_cohort",
            "l2_post_window:post_window_source_cohort_changed_during_refresh",
        ),
        (
            "post_window_validation",
            "l2_post_window_validation:post_window_source_cohort_changed",
        ),
    ],
)
def test_incomplete_l2_lattice_denies_terminal_slot_credit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected_blocker: str,
) -> None:
    _prepare_supervisor_root(tmp_path, monkeypatch)
    capture = _complete_capture_summary()
    readiness = _complete_post_window_summary()
    validation = _complete_post_window_validation()
    if failure == "strict_cost":
        capture["strict_pair_cost_ready"] = 0
        capture["strict_pair_cost_acceptance_status"] = "BLOCKED"
    elif failure == "post_window_cohort":
        readiness["status"] = "BLOCKED_SOURCE_DRIFT"
        readiness["blockers"] = [
            "post_window_source_cohort_changed_during_refresh"
        ]
    else:
        validation["status"] = "BLOCKED"
        validation["blockers"] = ["post_window_source_cohort_changed"]

    result = _run_isolated_supervisor(
        root=tmp_path,
        contract_key="hyperliquid_l2",
        publication_scope="public_l2",
        callback=lambda: corrective_l2_scheduler._build_supervised_l2_result(
            capture=CommandResult(paths={}, summary=capture),
            readiness=CommandResult(paths={}, summary=readiness),
            readiness_validation=validation,
        ),
        now=NOW,
    )

    assert result.exit_code == 2
    assert result.result_summary["status"] == "BLOCKED"
    assert expected_blocker in result.result_summary["blockers"]
    assert result.result_summary["l2_acceptance_lattice_complete"] is False
    assert result.result_summary["l2_terminal_slot_credit_eligible"] is False
    assert result.terminal_receipt["terminal_status"] == "BLOCKED"
    assert result.terminal_receipt["intended_slot_credit"] is False


def test_incomplete_pair_cost_result_writes_blocked_receipt_instead_of_crashing(
    tmp_path, monkeypatch
):
    _write_candidate_inputs(tmp_path)
    original_cost_builder = corrective_l2_scheduler.build_pair_cost_stress_surfaces

    def incomplete_cost_builder(**kwargs):
        result = original_cost_builder(**kwargs)
        result.pop("bundle_pointer")
        return result

    monkeypatch.setattr(
        corrective_l2_scheduler,
        "build_pair_cost_stress_surfaces",
        incomplete_cost_builder,
    )

    result = run_corrective_l2_capture(
        root=tmp_path,
        now=NOW,
        collector=lambda **_: CommandResult(paths={}, summary={"pairs": 1}),
    )

    assert result.summary["status"] == "BLOCKED"
    assert "pair_cost_evidence_contract_missing:bundle_pointer" in result.summary["blockers"]
    assert result.paths["capture_receipt"].is_file()
    assert result.paths["latest_status"].is_file()
    assert result.paths["pair_cost_bundle_pointer"].name == (
        "missing_pair_cost_bundle_pointer.json"
    )
    assert result.summary["lock_released"] is True
    assert result.summary["order_submission_included"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_active_capture_lock_writes_blocked_receipt_without_crashing(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    lock = active / ".corrective_l2_capture.lock"
    lock.write_text(
        json.dumps({"pid": os.getpid(), "started_at_utc": NOW.isoformat()}),
        encoding="utf-8",
    )

    result = run_corrective_l2_capture(
        root=tmp_path,
        now=NOW,
        collector=lambda **_: (_ for _ in ()).throw(
            AssertionError("overlapping capture must not call the collector")
        ),
    )

    assert result.summary["status"] == "BLOCKED"
    assert "active_scheduler_lock_present" in result.summary["blockers"]
    assert result.summary["lock_released"] is False
    assert result.summary["order_submission_included"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    assert result.paths["capture_receipt"].is_file()
    assert result.paths["latest_status"].is_file()
    assert result.paths["pair_cost_bundle_pointer"].name == (
        "missing_pair_cost_bundle_pointer.json"
    )
    assert lock.is_file()


def test_fresh_mapping_is_not_refreshed_by_l2_cycle(tmp_path):
    _write_candidate_inputs(tmp_path)
    _write_mapping_contract(tmp_path, checked_at=NOW)

    result = run_corrective_l2_capture(
        root=tmp_path,
        now=NOW,
        collector=lambda **_: CommandResult(paths={}, summary={"pairs": 1}),
        testnet_inventory_refresher=lambda **_: (_ for _ in ()).throw(
            AssertionError("fresh mapping must not call inventory")
        ),
        mapping_refresher=lambda **_: (_ for _ in ()).throw(
            AssertionError("fresh mapping must not rebuild")
        ),
    )

    assert result.summary["status"] == "BLOCKED"
    assert result.summary["capture_status"] == "PASS"
    assert result.summary["mapping_maintenance_configured"] is True
    assert result.summary["mapping_refresh_due"] is False
    assert result.summary["mapping_refresh_status"] == "NOT_DUE"
    assert result.summary["mapping_inventory_refreshed"] is False
    assert result.summary["mapping_refresh_id"] == "hlmap_existing"
    assert result.summary["mapping_ready_pair_groups"] == 1
    assert result.summary["mapping_blocked_pair_groups"] == 0


def test_due_mapping_refreshes_inventory_then_mapping_inside_lock(tmp_path):
    _write_candidate_inputs(tmp_path)
    _write_mapping_contract(tmp_path, checked_at=NOW - pd.Timedelta(hours=7))
    calls = []
    active = tmp_path / "reports" / "active"

    def inventory_refresher(**_):
        assert (active / ".corrective_l2_capture.lock").exists()
        calls.append("inventory")
        return _inventory_frame(NOW)

    def mapping_refresher(**_):
        assert calls == ["inventory"]
        calls.append("mapping")
        path = active / "exhaustive_wizard_hyperliquid_mapping_refresh.csv"
        pd.DataFrame(
            [
                {
                    "pair_group_key": "binance|daily|ETH|WIF",
                    "current_inventory_checked_at": NOW.isoformat(),
                    "current_pair_ready": True,
                }
            ]
        ).to_csv(path, index=False)
        return CommandResult(
            paths={"mapping": path},
            summary={
                "mapping_refresh_id": "hlmap_test",
                "pair_groups": 1,
                "current_ready_pair_groups": 1,
                "current_blocked_pair_groups": 0,
                "live_trading_authorized": False,
            },
        )

    result = run_corrective_l2_capture(
        root=tmp_path,
        now=NOW,
        collector=lambda **_: CommandResult(paths={}, summary={"pairs": 1}),
        testnet_inventory_refresher=inventory_refresher,
        mapping_refresher=mapping_refresher,
    )

    assert calls == ["inventory", "mapping"]
    assert result.summary["status"] == "BLOCKED"
    assert result.summary["capture_status"] == "PASS"
    assert result.summary["mapping_refresh_status"] == "PASS"
    assert result.summary["mapping_inventory_refreshed"] is True
    assert result.summary["mapping_refresh_id"] == "hlmap_test"
    assert result.summary["mapping_ready_pair_groups"] == 1
    assert result.summary["mapping_refresh_public_read_only"] is True
    assert result.summary["order_submission_included"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    assert result.paths["mapping_mapping"] == (
        active / "exhaustive_wizard_hyperliquid_mapping_refresh.csv"
    )


def test_mapping_refresh_failure_before_hard_sla_is_warning(tmp_path):
    _write_candidate_inputs(tmp_path)
    _write_mapping_contract(tmp_path, checked_at=NOW - pd.Timedelta(hours=7))

    result = run_corrective_l2_capture(
        root=tmp_path,
        now=NOW,
        collector=lambda **_: CommandResult(paths={}, summary={"pairs": 1}),
        testnet_inventory_refresher=lambda **_: (_ for _ in ()).throw(
            RuntimeError("temporary testnet outage")
        ),
    )

    assert result.summary["status"] == "BLOCKED"
    assert result.summary["capture_status"] == "PASS"
    assert result.summary["mapping_refresh_status"] == "BLOCKED_WARNING"
    assert result.summary["capture_blockers"] == []
    assert any(
        warning.startswith("exhaustive_mapping_refresh_warning:RuntimeError")
        for warning in result.summary["operational_warnings"]
    )


def test_mapping_refresh_failure_after_hard_sla_blocks_receipt(tmp_path):
    _write_candidate_inputs(tmp_path)
    _write_mapping_contract(tmp_path, checked_at=NOW - pd.Timedelta(hours=25))

    result = run_corrective_l2_capture(
        root=tmp_path,
        now=NOW,
        collector=lambda **_: CommandResult(paths={}, summary={"pairs": 1}),
        testnet_inventory_refresher=lambda **_: (_ for _ in ()).throw(
            RuntimeError("persistent testnet outage")
        ),
    )

    assert result.summary["status"] == "BLOCKED"
    assert result.summary["mapping_refresh_status"] == "BLOCKED_HARD_STALE"
    assert result.summary["mapping_hard_stale_before"] is True
    assert any(
        blocker.startswith("exhaustive_mapping_refresh_hard_stale:RuntimeError")
        for blocker in result.summary["blockers"]
    )
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_inventory_timestamp_allows_bounded_fetch_runtime_but_rejects_far_future():
    assert _inventory_refresh_blocker(
        _inventory_frame(NOW + pd.Timedelta(minutes=14)), NOW
    ) == ""
    assert _inventory_refresh_blocker(
        _inventory_frame(NOW + pd.Timedelta(minutes=16)), NOW
    ) == "testnet_inventory_refresh_timestamp_in_future"


def test_ready_l2_pair_is_deferred_to_daily_wizard_refresh_without_action(tmp_path):
    _write_candidate_inputs(tmp_path)
    processed = tmp_path / "data" / "processed"
    rows = []
    for offset in range(12):
        timestamp = NOW - pd.Timedelta(minutes=10 * offset)
        for asset in ("ETH", "WIF"):
            rows.append(
                {
                    "pair": "ETH-USD-WIF-USD",
                    "asset": asset,
                    "source_timestamp": timestamp.isoformat(),
                    "buy_complete": True,
                    "sell_complete": True,
                }
            )
    pd.DataFrame(rows).to_csv(processed / "hyperliquid_l2_slippage_samples.csv", index=False)

    def collector(**kwargs):
        return CommandResult(paths={}, summary={"pairs": 1})

    result = run_corrective_l2_capture(root=tmp_path, now=NOW, collector=collector)
    transition = pd.read_csv(result.paths["post_window_transition"]).iloc[0]

    assert result.summary["post_window_ready_pairs"] == 1
    assert result.summary["post_window_collecting_pairs"] == 0
    assert transition["post_window_transition_status"] == ("READY_FOR_POST_WINDOW_WIZARD_REFRESH")
    assert transition["wizard_daily_credit_reset_utc"] == "00:00"
    assert not bool(transition["candidate_refresh_execution_included"])
    assert not bool(transition["order_submission_included"])
    assert not bool(transition["live_trading_authorized"])


def test_capture_materializes_missing_candidate_funding_before_cost_status(tmp_path):
    _write_candidate_inputs(tmp_path)
    active = tmp_path / "reports" / "active"
    pd.DataFrame(
        [{"asset": "ETH", "funding_status": "COMPLETE", "funding_rows": 500}]
    ).to_csv(
        active / "current_wizard_hyperliquid_funding_asset_results.csv",
        index=False,
    )
    calls = []

    def collector(**kwargs):
        return CommandResult(paths={}, summary={"pairs": 1})

    def funding_materializer(**kwargs):
        calls.append(kwargs)
        path = active / "current_wizard_hyperliquid_funding_asset_results.csv"
        pd.DataFrame(
            [
                {
                    "asset": "ETH",
                    "funding_status": "COMPLETE",
                    "funding_rows": 500,
                },
                {
                    "asset": "WIF",
                    "funding_status": "COMPLETE",
                    "funding_rows": 500,
                },
            ]
        ).to_csv(path, index=False)
        return CommandResult(
            paths={"assets": path},
            summary={"funding_assets_complete": 2},
        )

    result = run_corrective_l2_capture(
        root=tmp_path,
        now=NOW,
        collector=collector,
        funding_materializer=funding_materializer,
    )

    assert result.summary["status"] == "BLOCKED"
    assert result.summary["capture_status"] == "PASS"
    assert result.summary["funding_refresh_status"] == "PASS"
    assert result.summary["funding_refresh_missing_assets_before"] == ["WIF"]
    assert calls[0]["pair_group_keys"] == ["binance|daily|ETH|WIF"]
    assert calls[0]["fetch_funding"] is True
    assert result.summary["order_submission_included"] is False
    assert result.paths["funding_assets"] == (
        active / "current_wizard_hyperliquid_funding_asset_results.csv"
    )


def test_missing_candidate_funding_fails_closed_without_order_authority(tmp_path):
    _write_candidate_inputs(tmp_path)
    active = tmp_path / "reports" / "active"
    pd.DataFrame(
        [{"asset": "ETH", "funding_status": "COMPLETE", "funding_rows": 500}]
    ).to_csv(
        active / "current_wizard_hyperliquid_funding_asset_results.csv",
        index=False,
    )

    result = run_corrective_l2_capture(
        root=tmp_path,
        now=NOW,
        collector=lambda **_: CommandResult(paths={}, summary={"pairs": 1}),
        funding_materializer=lambda **_: (_ for _ in ()).throw(
            RuntimeError("funding unavailable")
        ),
    )

    assert result.summary["status"] == "BLOCKED"
    assert result.summary["funding_refresh_status"] == "BLOCKED"
    assert result.summary["funding_refresh_missing_assets_before"] == ["WIF"]
    assert any(
        blocker.startswith("funding_materialization_error:RuntimeError")
        for blocker in result.summary["blockers"]
    )
    assert result.summary["order_submission_included"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    assert not (active / ".corrective_l2_capture.lock").exists()


def test_registered_candidate_uses_supplemental_funding_lane(tmp_path, monkeypatch):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    candidate_path = active / "corrective_l2_capture_candidates.csv"
    pd.DataFrame(
        [
            {
                "experiment_id": "registered-exp",
                "pair_group_key": "dydx|daily|ETH|PYTH",
                "pair": "ETH-USD-PYTH-USD",
                "asset_x": "ETH",
                "asset_y": "PYTH",
                "source_family": "registered_rerun_contract",
                "registered_contract_candidate": True,
                "stage_two_candidate": True,
                "collection_eligible": True,
            }
        ]
    ).to_csv(candidate_path, index=False)
    pd.DataFrame(
        [{"asset": "ETH", "funding_status": "COMPLETE", "funding_rows": 500}]
    ).to_csv(
        active / "current_wizard_hyperliquid_funding_asset_results.csv",
        index=False,
    )
    rows = []
    for offset in range(12):
        timestamp = NOW - pd.Timedelta(minutes=10 * offset)
        for asset in ("ETH", "PYTH"):
            rows.append(
                {
                    "pair": "ETH-USD-PYTH-USD",
                    "asset": asset,
                    "source_timestamp": timestamp.isoformat(),
                    "buy_complete": True,
                    "sell_complete": True,
                }
            )
    pd.DataFrame(rows).to_csv(
        processed / "hyperliquid_l2_slippage_samples.csv", index=False
    )
    supplemental_calls = []

    monkeypatch.setattr(
        corrective_l2_scheduler,
        "build_l2_capture_candidate_set",
        lambda **_: {
            "path": candidate_path,
            "registered_hypotheses": 1,
            "registered_contract_candidates": 1,
            "candidate_pairs": 1,
            "eligible_pairs": 1,
            "stage_two_candidate_pairs": 1,
        },
    )

    def supplemental(**kwargs):
        supplemental_calls.append(kwargs)
        path = active / "corrective_l2_funding_asset_results.csv"
        pd.DataFrame(
            [{"asset": "PYTH", "funding_status": "COMPLETE", "funding_rows": 500}]
        ).to_csv(path, index=False)
        return CommandResult(
            paths={"assets": path},
            summary={"status": "PASS", "completed_assets": 1},
        )

    result = run_corrective_l2_capture(
        root=tmp_path,
        now=NOW,
        collector=lambda **_: CommandResult(paths={}, summary={"pairs": 1}),
        funding_materializer=lambda **_: (_ for _ in ()).throw(
            AssertionError("current-board materializer must not receive registered pair")
        ),
        supplemental_funding_materializer=supplemental,
    )

    assert supplemental_calls[0]["assets"] == ["PYTH"]
    assert result.summary["funding_refresh_status"] == "PASS"
    assert result.summary["funding_refresh_missing_assets_after"] == []
    assert result.summary["post_window_ready_pairs"] == 1
    assert result.paths["supplemental_funding_assets"] == (
        active / "corrective_l2_funding_asset_results.csv"
    )
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_supplemental_funding_receipt_is_immutable_and_point_in_time(tmp_path):
    def fetcher(*, coin, output_dir, end_time, **_):
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{coin}_funding.json"
        path.write_text(
            json.dumps(
                {
                    "coin": coin,
                    "fetch_complete": True,
                    "funding": [
                        {
                            "coin": coin,
                            "fundingRate": "0.00001",
                            "time": int(end_time.timestamp() * 1000) - 3_600_000,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    result = materialize_corrective_candidate_funding_assets(
        root=tmp_path,
        assets=["PYTH"],
        now=NOW,
        funding_fetcher=fetcher,
    )
    frame = pd.read_csv(result.paths["assets"])

    assert result.summary["status"] == "PASS"
    assert result.summary["completed_assets"] == 1
    assert len(result.summary["snapshot_sha256"]) == 64
    assert result.paths["snapshot_assets"].is_file()
    assert result.paths["receipt"].is_file()
    assert frame.iloc[0]["asset"] == "PYTH"
    assert frame.iloc[0]["funding_status"] == "COMPLETE"
    assert not bool(frame.iloc[0]["testnet_order_authority"])
    assert not bool(frame.iloc[0]["live_trading_authorized"])


def test_launch_agent_uses_five_minute_read_only_cadence(tmp_path):
    python = tmp_path / ".venv" / "bin" / "python3"
    python.parent.mkdir(parents=True)
    python.write_text("")
    logs = tmp_path / "reports" / "active" / "schedule_logs"
    logs.mkdir(parents=True)

    plist = _launch_agent_plist(
        root=tmp_path,
        python=python,
        logs=logs,
        interval_seconds=corrective_l2_scheduler.DEFAULT_INTERVAL_SECONDS,
    )

    payload = plistlib.loads(plist.encode("utf-8"))
    contract = scheduler_contract("hyperliquid_l2")
    assert corrective_l2_scheduler.DEFAULT_INTERVAL_SECONDS == 300
    assert payload["StartInterval"] == 300
    assert payload["ProgramArguments"] == [
        str(python),
        "-m",
        SCHEDULER_BOOTSTRAP_MODULE,
        str(tmp_path.resolve()),
        contract.key,
    ]
    assert contract.module.endswith("corrective_l2_scheduler")
    assert contract.action == "--capture"
    assert "--execute" not in plist
    assert "order" not in plist.lower()
    assert "ProcessType" not in plist
    assert "LowPriorityIO" not in plist
    environment = payload["EnvironmentVariables"]
    assert environment["TMPDIR"] == f"{tmp_path}/.runtime_tmp"
    assert environment["TMP"] == f"{tmp_path}/.runtime_tmp"
    assert environment["TEMP"] == f"{tmp_path}/.runtime_tmp"
    assert environment["TZ"] == "America/New_York"


def test_l2_receipt_binds_candidate_bundle_pointer_and_model(tmp_path, monkeypatch):
    result = _capture_ready_registered_pair(tmp_path, monkeypatch)

    validation = validate_l2_capture_receipt(root=tmp_path)

    assert validation["status"] == "PASS"
    assert validation["receipt_id"] == result.summary["receipt_id"]
    assert len(validation["candidate_evidence_sha256"]) == 64

    result.paths["candidate_set"].write_text("tampered\n", encoding="utf-8")
    validation = validate_l2_capture_receipt(root=tmp_path)

    assert validation["status"] == "BLOCKED"
    assert "l2_capture_candidate_binding_invalid" in validation["blockers"]


@pytest.mark.parametrize(
    ("artifact_key", "blocker"),
    [
        (
            "pair_cost_bundle_manifest",
            "l2_capture_pair_cost_bundle_manifest_binding_invalid",
        ),
        (
            "pair_cost_bundle_pointer_snapshot",
            "l2_capture_pair_cost_bundle_pointer_binding_invalid",
        ),
        (
            "pair_cost_model_snapshot",
            "l2_capture_pair_cost_model_snapshot_binding_invalid",
        ),
    ],
)
def test_l2_receipt_rejects_cost_artifact_tampering(
    tmp_path, monkeypatch, artifact_key, blocker
):
    capture = _capture_ready_registered_pair(tmp_path, monkeypatch)
    with capture.paths[artifact_key].open("ab") as handle:
        handle.write(b"tampered")

    validation = validate_l2_capture_receipt(root=tmp_path)

    assert validation["status"] == "BLOCKED"
    assert blocker in validation["blockers"]


def test_l2_receipt_survives_later_active_pointer_rotation_but_refresh_stops(
    tmp_path, monkeypatch
):
    capture = _capture_ready_registered_pair(tmp_path, monkeypatch)
    capture.paths["pair_cost_bundle_pointer"].write_text(
        json.dumps({"bundle_id": "later-valid-build"}) + "\n", encoding="utf-8"
    )

    historical_validation = validate_l2_capture_receipt(root=tmp_path)
    refresh = run_post_window_readiness_refresh(
        root=tmp_path,
        now=NOW,
        expected_l2_receipt_id=capture.summary["receipt_id"],
        gate_builder=lambda **_: (_ for _ in ()).throw(
            AssertionError("rotated routing pointer must stop before the gate")
        ),
        handoff_builder=lambda **_: (_ for _ in ()).throw(
            AssertionError("rotated routing pointer must stop before Stage 4")
        ),
        handoff_validator=lambda **_: (_ for _ in ()).throw(
            AssertionError("rotated routing pointer must stop before validation")
        ),
    )

    assert historical_validation["status"] == "PASS"
    assert refresh.summary["status"] == "BLOCKED_ACTIVE_ROUTING_POINTER"
    assert "post_window_active_pair_cost_pointer_changed" in refresh.summary[
        "blockers"
    ]
    assert refresh.summary["refresh_executed"] is False


def test_post_window_refresh_waits_without_calling_builders(tmp_path):
    _write_candidate_inputs(tmp_path)
    capture = run_corrective_l2_capture(
        root=tmp_path,
        now=NOW,
        collector=lambda **_: CommandResult(paths={}, summary={"pairs": 1}),
    )

    def forbidden(**_):
        raise AssertionError("readiness builders must not run before strict L2 is ready")

    refresh = run_post_window_readiness_refresh(
        root=tmp_path,
        now=NOW,
        expected_l2_receipt_id=capture.summary["receipt_id"],
        gate_builder=forbidden,
        handoff_builder=forbidden,
        handoff_validator=forbidden,
    )

    assert refresh.summary["status"] == "WAITING_STRICT_L2"
    assert refresh.summary["refresh_executed"] is False
    assert refresh.summary["registered_gate_refresh_executed"] is False
    assert refresh.summary["stage4_handoff_refresh_executed"] is False
    assert refresh.summary["external_wizard_call_included"] is False
    assert refresh.summary["order_submission_included"] is False
    assert refresh.paths["immutable_receipt"].is_file()
    assert validate_post_window_readiness_receipt(
        root=tmp_path, receipt=refresh.summary
    )["status"] == "PASS"


def test_ready_post_window_refresh_runs_local_gate_and_handoff_under_locks(
    tmp_path, monkeypatch
):
    capture = _capture_ready_registered_pair(tmp_path, monkeypatch)
    calls = []

    def gate_builder(**_):
        calls.append("gate")
        for name in corrective_l2_scheduler.POST_WINDOW_LOCK_NAMES:
            assert (tmp_path / "reports" / "active" / name).is_file()
        return CommandResult(
            paths={},
            summary={
                "status": "PASS_REGISTERED_RERUN_ACCOUNTED",
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        )

    def handoff_builder(**_):
        calls.append("handoff")
        return CommandResult(
            paths={},
            summary={
                "status": "PASS_STAGE4_HANDOFF_READY",
                "receipt_id": "stage4handoff_fixture",
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        )

    refresh = run_post_window_readiness_refresh(
        root=tmp_path,
        now=NOW,
        expected_l2_receipt_id=capture.summary["receipt_id"],
        gate_builder=gate_builder,
        handoff_builder=handoff_builder,
        handoff_validator=lambda **_: {"status": "PASS", "blockers": []},
    )

    assert calls == ["gate", "handoff"]
    assert refresh.summary["status"] == "PASS_LOCAL_READINESS_REFRESH"
    assert refresh.summary["refresh_executed"] is True
    assert refresh.summary["registered_gate_refresh_executed"] is True
    assert refresh.summary["stage4_handoff_refresh_executed"] is True
    assert refresh.summary["stage4_handoff_validation_status"] == "PASS"
    assert refresh.summary["testnet_order_authority"] is False
    assert refresh.summary["live_trading_authorized"] is False
    assert validate_post_window_readiness_receipt(
        root=tmp_path, receipt=refresh.summary
    )["status"] == "PASS"
    for name in corrective_l2_scheduler.POST_WINDOW_LOCK_NAMES:
        assert not (tmp_path / "reports" / "active" / name).exists()

    refresh.paths["immutable_receipt"].chmod(0o600)
    refresh.paths["immutable_receipt"].write_text("{}\n", encoding="utf-8")
    validation = validate_post_window_readiness_receipt(
        root=tmp_path, receipt=refresh.summary
    )
    assert validation["status"] == "BLOCKED"
    assert "l2_readiness_refresh_immutable_receipt_mismatch" in validation["blockers"]


@pytest.mark.parametrize(
    ("blocked_stage", "expected_blocker", "expected_calls"),
    [
        (
            "gate",
            "post_window_registered_gate_status:BLOCKED_VENDOR_PARITY",
            ["gate"],
        ),
        (
            "handoff",
            "post_window_stage4_handoff_status:BLOCKED_STAGE4_HANDOFF",
            ["gate", "handoff"],
        ),
    ],
)
def test_nonpassing_post_window_stage_blocks_supervised_l2_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked_stage: str,
    expected_blocker: str,
    expected_calls: list[str],
) -> None:
    capture = _capture_ready_registered_pair(tmp_path, monkeypatch)
    calls: list[str] = []

    def gate_builder(**_) -> CommandResult:
        calls.append("gate")
        status = (
            "BLOCKED_VENDOR_PARITY"
            if blocked_stage == "gate"
            else "PASS_REGISTERED_RERUN_ACCOUNTED"
        )
        return CommandResult(
            paths={},
            summary={
                "status": status,
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        )

    def handoff_builder(**_) -> CommandResult:
        calls.append("handoff")
        return CommandResult(
            paths={},
            summary={
                "status": "BLOCKED_STAGE4_HANDOFF",
                "receipt_id": "stage4handoff_blocked_fixture",
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        )

    refresh = run_post_window_readiness_refresh(
        root=tmp_path,
        now=NOW,
        expected_l2_receipt_id=capture.summary["receipt_id"],
        gate_builder=gate_builder,
        handoff_builder=handoff_builder,
        handoff_validator=lambda **_: {"status": "PASS", "blockers": []},
    )
    validation = validate_post_window_readiness_receipt(
        root=tmp_path,
        receipt=refresh.summary,
    )
    supervised = corrective_l2_scheduler._build_supervised_l2_result(
        capture=capture,
        readiness=refresh,
        readiness_validation=validation,
    )

    assert calls == expected_calls
    assert refresh.summary["status"] == "BLOCKED_LOCAL_READINESS_REFRESH"
    assert expected_blocker in refresh.summary["blockers"]
    assert validation["status"] == "PASS"
    assert supervised["summary"]["status"] == "BLOCKED"
    assert supervised["summary"]["l2_authorization_state"] == "BLOCKED"
    assert supervised["summary"]["l2_terminal_slot_credit_eligible"] is False
    assert supervised["summary"]["post_window_readiness_validation"] == validation


def test_post_window_refresh_blocks_candidate_cohort_tampering(tmp_path, monkeypatch):
    capture = _capture_ready_registered_pair(tmp_path, monkeypatch)
    capture.paths["candidate_set"].write_text("tampered\n", encoding="utf-8")

    def forbidden(**_):
        raise AssertionError("tampered evidence must never reach readiness builders")

    refresh = run_post_window_readiness_refresh(
        root=tmp_path,
        now=NOW,
        expected_l2_receipt_id=capture.summary["receipt_id"],
        gate_builder=forbidden,
        handoff_builder=forbidden,
        handoff_validator=forbidden,
    )

    assert refresh.summary["status"] == "BLOCKED_L2_RECEIPT"
    assert "l2_capture_candidate_binding_invalid" in refresh.summary["blockers"]
    assert refresh.summary["refresh_executed"] is False


def test_post_window_refresh_defers_when_daily_producer_is_active(tmp_path, monkeypatch):
    capture = _capture_ready_registered_pair(tmp_path, monkeypatch)
    daily_lock = tmp_path / "reports" / "active" / ".corrective_daily.lock"
    daily_lock.write_text(
        json.dumps({"pid": os.getpid(), "started_at_utc": NOW.isoformat()}),
        encoding="utf-8",
    )

    refresh = run_post_window_readiness_refresh(
        root=tmp_path,
        now=NOW,
        expected_l2_receipt_id=capture.summary["receipt_id"],
        gate_builder=lambda **_: (_ for _ in ()).throw(AssertionError("must defer")),
        handoff_builder=lambda **_: (_ for _ in ()).throw(AssertionError("must defer")),
        handoff_validator=lambda **_: (_ for _ in ()).throw(AssertionError("must defer")),
    )

    assert refresh.summary["status"] == "DEFERRED_CONCURRENT_PRODUCER"
    assert "post_window_concurrent_producer_active" in refresh.summary["blockers"]
    assert refresh.summary["refresh_executed"] is False
    assert daily_lock.is_file()


def test_post_window_refresh_rejects_downstream_authority(tmp_path, monkeypatch):
    capture = _capture_ready_registered_pair(tmp_path, monkeypatch)

    refresh = run_post_window_readiness_refresh(
        root=tmp_path,
        now=NOW,
        expected_l2_receipt_id=capture.summary["receipt_id"],
        gate_builder=lambda **_: CommandResult(
            paths={},
            summary={"status": "PASS", "promotion_authority": True},
        ),
        handoff_builder=lambda **_: (_ for _ in ()).throw(
            AssertionError("unsafe gate must stop before Stage 4")
        ),
        handoff_validator=lambda **_: {"status": "PASS", "blockers": []},
    )

    assert refresh.summary["status"] == "BLOCKED_LOCAL_READINESS_REFRESH"
    assert "post_window_registered_gate_authority_violation" in refresh.summary["blockers"]
    assert refresh.summary["registered_gate_refresh_executed"] is True
    assert refresh.summary["stage4_handoff_refresh_executed"] is False
    assert refresh.summary["promotion_authority"] is False
    assert refresh.summary["testnet_order_authority"] is False
    assert refresh.summary["live_trading_authorized"] is False


def test_post_window_refresh_rejects_source_drift_during_build(tmp_path, monkeypatch):
    capture = _capture_ready_registered_pair(tmp_path, monkeypatch)
    batch_path = tmp_path / "reports" / "active" / "current_hypothesis_batch.csv"

    def drifting_gate(**_):
        batch_path.write_text("experiment_id\nchanged\n", encoding="utf-8")
        return CommandResult(
            paths={},
            summary={
                "status": "BLOCKED_VENDOR_PARITY",
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        )

    refresh = run_post_window_readiness_refresh(
        root=tmp_path,
        now=NOW,
        expected_l2_receipt_id=capture.summary["receipt_id"],
        gate_builder=drifting_gate,
        handoff_builder=lambda **_: (_ for _ in ()).throw(
            AssertionError("source drift must stop before Stage 4")
        ),
        handoff_validator=lambda **_: {"status": "PASS", "blockers": []},
    )

    assert refresh.summary["status"] == "BLOCKED_LOCAL_READINESS_REFRESH"
    assert "post_window_source_cohort_changed_during_refresh" in refresh.summary[
        "blockers"
    ]
    assert refresh.summary["registered_gate_refresh_executed"] is True
    assert refresh.summary["stage4_handoff_refresh_executed"] is False
    assert refresh.summary["testnet_order_authority"] is False
