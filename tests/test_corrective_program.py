from __future__ import annotations

from quant_platform.orchestration.corrective_program import TASK_STATUS


def test_all_41_tasks_have_explicit_corrective_status():
    assert set(TASK_STATUS) == {f"T{index:02d}" for index in range(1, 42)}


def test_conditional_execution_tasks_never_marked_unconditionally_complete():
    for task in ("T33", "T34", "T35", "T36", "T39", "T40", "T41"):
        assert TASK_STATUS[task][0] != "completed"
