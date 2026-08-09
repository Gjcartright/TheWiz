from __future__ import annotations

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_program import (
    TASK_STATUS,
    _write_seven_stage_checkpoint,
)


def test_all_41_tasks_have_explicit_corrective_status():
    assert set(TASK_STATUS) == {f"T{index:02d}" for index in range(1, 42)}


def test_conditional_execution_tasks_never_marked_unconditionally_complete():
    for task in ("T33", "T34", "T35", "T36", "T39", "T40", "T41"):
        assert TASK_STATUS[task][0] != "completed"


def test_ou_optimal_is_accounted_as_scanner_overlay_not_missing_pair_page_mode():
    assert TASK_STATUS["T17"][0] == "completed"
    assert "scanner_overlay" in TASK_STATUS["T17"][1]


def test_seven_stage_checkpoint_never_grants_order_authority(tmp_path):
    def result(**summary):
        return CommandResult(paths={}, summary=summary)

    phases = {
        "venue_data_costs": result(
            strict_cost_ready_pairs=0,
            history_queued_pairs=1,
            history_ready_pairs=2,
        ),
        "wizard_parity": result(status="BLOCKED", blocker="parity_missing"),
        "statistical_remediation": result(
            final_one_x_survivors=0,
            near_miss_candidates=1,
            independent_supporting_clusters=0,
            blockers=["zero_survivors"],
        ),
        "daily_cadence": result(
            consecutive_complete_cycles=1,
            blocker="need_seven_days",
        ),
        "agent_learning_governance": result(
            model_authority="RESEARCH_ONLY",
            blockers=["incremental_edge_missing"],
        ),
        "conditional_release_gates": result(testnet_sample_sufficient=False),
    }
    csv_path, md_path = _write_seven_stage_checkpoint(
        root=tmp_path,
        phases=phases,
        completion={"live_trading_authorized": False},
    )
    frame = pd.read_csv(csv_path)

    assert len(frame) == 7
    assert not frame["testnet_order_authority"].astype(bool).any()
    assert not frame["live_trading_authorized"].astype(bool).any()
    assert frame.loc[frame["stage"].eq(1), "evidence_progress"].iloc[0] == "1/7"
    assert md_path.exists()
