from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from quant_platform.orchestration.current_wizard_hyperliquid_completion_audit import (
    EXACT_MODES,
    build_current_wizard_hyperliquid_completion_audit,
)


def test_completion_audit_allows_claim_only_when_every_requirement_is_satisfied(
    tmp_path,
):
    _seed_audit_root(tmp_path, daily_complete=True, discovery_prefilter=False)

    result = build_current_wizard_hyperliquid_completion_audit(
        root=tmp_path,
        now=datetime(2026, 8, 8, tzinfo=timezone.utc),
    )
    frame = pd.read_csv(result.paths["completion_audit"])

    assert result.summary["completion_status"] == "PASS"
    assert result.summary["completion_claim_allowed"] is True
    assert result.summary["blocked"] == 0
    assert result.summary["unproven"] == 0
    assert set(frame["status"]).issubset({"PROVEN", "CONDITIONALLY_PROVEN"})
    assert not frame["order_submission_performed"].astype(bool).any()
    assert not frame["live_trading_authorized"].astype(bool).any()


def test_completion_audit_refuses_prefiltered_or_unexecuted_daily_chain(tmp_path):
    _seed_audit_root(tmp_path, daily_complete=False, discovery_prefilter=True)

    result = build_current_wizard_hyperliquid_completion_audit(
        root=tmp_path,
        now=datetime(2026, 8, 8, tzinfo=timezone.utc),
    )
    frame = pd.read_csv(result.paths["completion_audit"]).set_index("requirement_id")

    assert result.summary["completion_status"] == "BLOCKED"
    assert result.summary["completion_claim_allowed"] is False
    assert frame.loc["exhaustive_discovery_no_prefilter", "status"] == "UNPROVEN"
    assert frame.loc["exact_modes_and_asymmetric_orientations", "status"] == "UNPROVEN"
    assert frame.loc["repeatable_fresh_daily_cycle", "status"] == "BLOCKED"
    assert frame.loc["no_live_trading_authorization", "status"] == "PROVEN"


def _seed_audit_root(
    root: Path, *, daily_complete: bool, discovery_prefilter: bool
) -> None:
    active = root / "reports" / "active"
    dashboard = root / "reports" / "dashboard"
    active.mkdir(parents=True)
    dashboard.mkdir(parents=True)

    _json(
        active / "exhaustive_wizard_api_refresh_manifest.json",
        {
            "sweep_complete": True,
            "api_source_rows": 2,
            "api_source_rows_accounted": 2,
            "api_pair_groups": 1,
            "discovery_prefilters": {
                "sharpe": 1.75 if discovery_prefilter else None,
                "returns_total": None,
            },
            "exact_modes_required": list(EXACT_MODES),
            "orientations_required": ["original", "reverse"],
            "no_silent_drops": True,
            "order_submission_performed": False,
            "live_trading_authorized": False,
        },
    )
    matrix_rows = []
    for mode in EXACT_MODES:
        for orientation in ("original", "reverse"):
            matrix_rows.append(
                {
                    "pair_group_key": "dydx|daily|BTC|ETH",
                    "exact_mode": mode,
                    "orientation": orientation,
                    "experiment_status": "READY_FOR_REPLAY",
                    "experiment_blocker": "",
                    "discovery_prefilter_applied": discovery_prefilter,
                }
            )
    pd.DataFrame(matrix_rows).to_csv(
        active / "current_wizard_hyperliquid_experiment_matrix.csv", index=False
    )
    _validation(active / "current_wizard_hyperliquid_handoff_validation.csv")
    _json(
        active / "current_wizard_hyperliquid_canonical_replay_manifest.json",
        {
            "experiments_accounted": 16,
            "unique_experiment_ids": 16,
            "mapping_blocked_experiments": 0,
            "canonical_replay_leverage": 1.0,
            "train_only_parameter_fit": True,
            "test_only_performance_measurement": True,
            "replay_status_count_total": 16,
            "research_replays_complete": 16,
            "research_rank_eligible_replays": 4,
            "blocked_point_in_time_history": 0,
            "order_submission_performed": False,
            "live_trading_authorized": False,
        },
    )
    _json(
        active / "current_wizard_hyperliquid_leverage_manifest.json",
        {
            "experiments_accounted": 16,
            "one_x_research_survivors_selected": 0,
            "leverage_candidates_complete": 0,
            "scenario_rows": 0,
            "expected_scenario_rows": 0,
            "order_submission_performed": False,
            "live_trading_authorized": False,
        },
    )
    _validation(active / "current_wizard_hyperliquid_leverage_validation.csv")
    _json(
        active / "current_wizard_hyperliquid_learning_manifest.json",
        {
            "records": 16,
            "unique_experiment_ids": 16,
            "training_eligible_records": 0,
            "paper_label_records": 0,
            "live_label_records": 0,
            "wizard_is_label_authority": False,
            "order_submission_performed": False,
            "live_trading_authorized": False,
        },
    )
    _validation(active / "current_wizard_hyperliquid_learning_validation.csv")
    _json(
        active / "current_wizard_hyperliquid_chain_validation_manifest.json",
        {
            "chain_status": "PASS",
            "checks": 1,
            "checks_passed": 1,
            "experiment_authority_count": 16,
            "order_submission_performed": False,
            "live_trading_authorized": False,
        },
    )
    _validation(active / "current_wizard_hyperliquid_chain_validation.csv")
    _json(
        active / "current_wizard_hyperliquid_testnet_protocol_manifest.json",
        {
            "protocol_status": "PASS",
            "required_scenarios_complete": True,
            "scenarios": 1,
            "scenarios_passed": 1,
            "simulation_is_testnet_proof": False,
            "actual_testnet_candidates": 0,
            "actual_testnet_lifecycle_proven": 0,
            "order_submission_performed": False,
            "live_trading_authorized": False,
        },
    )
    _json(
        active / "current_wizard_hyperliquid_daily_run_manifest.json",
        {
            "daily_run_id": "daily-test",
            "execution_requested": daily_complete,
            "run_status": "PASS" if daily_complete else "BLOCKED_STORAGE",
            "stages": 19,
            "stages_passed": 19 if daily_complete else 0,
            "final_free_disk_bytes": 4 * 1024**3 if daily_complete else 512 * 1024**2,
            "minimum_free_disk_bytes": 3 * 1024**3,
            "order_submission_performed": False,
            "live_trading_authorized": False,
        },
    )
    pd.DataFrame(
        [
            {
                "lock_state": "PERMANENT_RESEARCH_ONLY",
                "live_order_authority": False,
                "order_submission_performed": False,
            }
        ]
    ).to_csv(active / "current_wizard_hyperliquid_live_lock.csv", index=False)

    for path in _required_reports(root):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence\n", encoding="utf-8")


def _required_reports(root: Path) -> tuple[Path, ...]:
    active = root / "reports" / "active"
    return (
        active / "current_wizard_hyperliquid_canonical_replay_ranked.csv",
        active / "current_wizard_hyperliquid_walkforward_ranked.csv",
        active / "current_wizard_hyperliquid_concentration_status.csv",
        active / "current_wizard_hyperliquid_failure_attribution.csv",
        active / "current_wizard_hyperliquid_leverage_status.csv",
        active / "current_wizard_hyperliquid_learning_ledger.csv.gz",
        active / "current_wizard_hyperliquid_daily_run_status.csv",
        root / "reports" / "dashboard" / "command_center.md",
    )


def _validation(path: Path) -> None:
    pd.DataFrame([{"check": "fixture", "status": "PASS"}]).to_csv(
        path, index=False
    )


def _json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
