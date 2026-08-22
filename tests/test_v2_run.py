from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from quant_platform.v2_run import (
    build_v2_preflight_run,
    publish_v2_run_status,
    validate_v2_run,
)
from quant_platform.wizard_policy import DEFAULT_WIZARD_DISCOVERY_POLICY

NOW = datetime(2026, 8, 7, 18, 0, tzinfo=timezone.utc)
CANDIDATE_SET_ID = "whlset_fixture_current"


def _write_ready_sources(root) -> None:
    active = root / "reports" / "active"
    council = root / "reports" / "orchestration" / "teacher_council"
    active.mkdir(parents=True)
    council.mkdir(parents=True)
    policy_hash = DEFAULT_WIZARD_DISCOVERY_POLICY.policy_hash
    pd.DataFrame([{"pair": "BTC/EIGEN", "sharpe": 2.1, "returns_total": 0.20}]).to_csv(
        active / "wizard_sweep_candidates.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "pair": "BTC/EIGEN",
                "capture_confirmed": True,
                "backtest_settings_complete": True,
            }
        ]
    ).to_csv(active / "crypto_wizards_pair_page_capture_settings.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC/EIGEN",
                "candidate_set_id": CANDIDATE_SET_ID,
                "discovery_policy_hash": policy_hash,
                "passes_wizard_discovery_gate": True,
                "passes_research_spend_gate": True,
                "wizard_source_fresh": True,
            }
        ]
    ).to_csv(active / "hyperliquid_wizard_hypothesis_queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC/EIGEN",
                "candidate_set_id": CANDIDATE_SET_ID,
                "discovery_policy_hash": policy_hash,
                "mode_proof_status": "completed",
                "proof_window_kind": "scanner_horizon_parity",
            }
        ]
    ).to_csv(active / "hyperliquid_wizard_vendor_mode_proofs.csv", index=False)
    pd.DataFrame([{"pair": "BTC/EIGEN", "history_ready": True}]).to_csv(
        active / "hyperliquid_research_bundle.csv", index=False
    )
    pd.DataFrame(
        [{"pair": "BTC/EIGEN", "funding_ready": True, "funding_coverage_pct": 100.0}]
    ).to_csv(active / "hyperliquid_funding_coverage.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC/EIGEN",
                "cost_model_ready": True,
                "slippage_model_ready": True,
            }
        ]
    ).to_csv(active / "hyperliquid_pair_cost_model.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC/EIGEN",
                "candidate_set_id": CANDIDATE_SET_ID,
                "sample_parity_status": "MATCHED_OBSERVATION_COUNT",
                "comparison_validity": "VALID",
            }
        ]
    ).to_csv(active / "wizard_mode_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC/EIGEN",
                "candidate_set_id": CANDIDATE_SET_ID,
                "selection_status": "PASS",
            }
        ]
    ).to_csv(council / "walkforward_mode_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC/EIGEN",
                "candidate_set_id": CANDIDATE_SET_ID,
                "status": "SHADOW_TEST",
                "action": "enter",
            }
        ]
    ).to_csv(council / "council_decisions.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC/EIGEN",
                "candidate_set_id": CANDIDATE_SET_ID,
                "verdict": "pass",
            }
        ]
    ).to_csv(council / "portfolio_critic.csv", index=False)


def test_ready_run_is_sealed_and_ignores_later_global_changes(tmp_path):
    _write_ready_sources(tmp_path)

    result = build_v2_preflight_run(root=tmp_path, now=NOW)
    run_id = result.summary["run_id"]
    validation = validate_v2_run(root=tmp_path, run_id=run_id)

    assert result.summary["status"] == "READY_FOR_RESEARCH"
    assert result.summary["research_ready"] is True
    assert result.summary["paper_ready"] is False
    assert result.summary["execution_allowed"] is False
    assert validation["sealed"] is True
    completed = json.loads(
        (tmp_path / "reports" / "active" / "v2_latest_completed_run.json").read_text(
            encoding="utf-8"
        )
    )
    assert completed["run_id"] == run_id

    active_queue = tmp_path / "reports" / "active" / "hyperliquid_wizard_hypothesis_queue.csv"
    active_queue.write_text("corrupted outside the sealed run", encoding="utf-8")
    assert validate_v2_run(root=tmp_path, run_id=run_id)["sealed"] is True


def test_blocked_attempt_does_not_replace_latest_completed_pointer(tmp_path):
    _write_ready_sources(tmp_path)
    ready = build_v2_preflight_run(root=tmp_path, now=NOW)
    completed_path = tmp_path / "reports" / "active" / "v2_latest_completed_run.json"
    original_completed = json.loads(completed_path.read_text(encoding="utf-8"))
    queue_path = tmp_path / "reports" / "active" / "hyperliquid_wizard_hypothesis_queue.csv"
    queue = pd.read_csv(queue_path)
    queue["passes_research_spend_gate"] = False
    queue.to_csv(queue_path, index=False)

    blocked = build_v2_preflight_run(
        root=tmp_path,
        now=NOW.replace(microsecond=1),
    )
    attempt = json.loads(
        (tmp_path / "reports" / "active" / "v2_latest_attempt.json").read_text(
            encoding="utf-8"
        )
    )
    still_completed = json.loads(completed_path.read_text(encoding="utf-8"))

    assert ready.summary["status"] == "READY_FOR_RESEARCH"
    assert blocked.summary["status"] == "BLOCKED"
    assert "wizard_paid_proof_gate_failed" in blocked.summary["blockers"]
    assert attempt["run_id"] == blocked.summary["run_id"]
    assert still_completed == original_completed


def test_cross_run_council_evidence_blocks_authority(tmp_path):
    _write_ready_sources(tmp_path)
    council_path = tmp_path / "reports" / "orchestration" / "teacher_council" / "council_decisions.csv"
    council = pd.read_csv(council_path)
    council["candidate_set_id"] = "whlset_old_run"
    council.to_csv(council_path, index=False)

    result = build_v2_preflight_run(root=tmp_path, now=NOW)

    assert result.summary["status"] == "BLOCKED"
    assert "candidate_set_identity_mismatch:council_decisions" in result.summary["blockers"]
    assert not (tmp_path / "reports" / "active" / "v2_latest_completed_run.json").exists()


def test_tampering_invalidates_seal_and_cannot_publish_completion(tmp_path):
    _write_ready_sources(tmp_path)
    result = build_v2_preflight_run(root=tmp_path, now=NOW)
    run_id = result.summary["run_id"]
    snapshot = result.paths["run_dir"] / "wizard_capture" / "hyperliquid_wizard_hypothesis_queue.csv"
    snapshot.chmod(0o600)
    snapshot.write_text("tampered", encoding="utf-8")

    validation = validate_v2_run(root=tmp_path, run_id=run_id)
    published = publish_v2_run_status(root=tmp_path, run_id=run_id)

    assert validation["sealed"] is False
    assert any("v2_sealed_file_hash_mismatch" in value for value in validation["blockers"])
    assert published.summary["completed_published"] is False
    assert published.summary["execution_allowed"] is False


def test_invalid_run_identifier_cannot_escape_runs_directory(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_v2_run(root=tmp_path, run_id="../reports")


def test_missing_funding_coverage_blocks_without_crashing(tmp_path):
    _write_ready_sources(tmp_path)
    funding_path = (
        tmp_path
        / "reports"
        / "active"
        / "hyperliquid_funding_coverage.csv"
    )
    funding = pd.read_csv(funding_path).drop(columns=["funding_coverage_pct"])
    funding.to_csv(funding_path, index=False)

    result = build_v2_preflight_run(root=tmp_path, now=NOW)

    assert result.summary["status"] == "BLOCKED"
    assert "funding_coverage_below_95pct" in result.summary["blockers"]


def test_candidate_set_id_cannot_mask_wrong_pair_coverage(tmp_path):
    _write_ready_sources(tmp_path)
    council_path = (
        tmp_path
        / "reports"
        / "orchestration"
        / "teacher_council"
        / "council_decisions.csv"
    )
    council = pd.read_csv(council_path)
    council["pair"] = "DOGE/WLD"
    council.to_csv(council_path, index=False)

    result = build_v2_preflight_run(root=tmp_path, now=NOW)

    assert result.summary["status"] == "BLOCKED"
    assert "candidate_pair_coverage_mismatch:council_decisions" in result.summary["blockers"]


def test_walkforward_selection_status_is_required(tmp_path):
    _write_ready_sources(tmp_path)
    walkforward_path = (
        tmp_path
        / "reports"
        / "orchestration"
        / "teacher_council"
        / "walkforward_mode_summary.csv"
    )
    walkforward = pd.read_csv(walkforward_path).drop(columns=["selection_status"])
    walkforward.to_csv(walkforward_path, index=False)

    result = build_v2_preflight_run(root=tmp_path, now=NOW)

    assert result.summary["status"] == "BLOCKED"
    assert "walkforward_selection_not_passed" in result.summary["blockers"]
