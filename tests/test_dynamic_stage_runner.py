from __future__ import annotations

import pytest

from quant_platform.orchestration.dynamic_stage_runner import run_dynamic_stage
from quant_platform.orchestration.state import OrchestratorState, StageResult, StageStatus


def test_copula_stage_returns_a_real_stage_result_without_ml_imports(tmp_path):
    result = run_dynamic_stage("copula_shadow_comparison", OrchestratorState(pair_id="BNB-USD/WLD-USD", report_only=True, root=tmp_path), tmp_path)
    assert isinstance(result, StageResult)
    assert result.stage == "copula_shadow_comparison"
    assert "copula_shadow_" in result.reason


def test_dynamic_stage_rejects_unknown_stage_without_execution_fallback(tmp_path):
    result = run_dynamic_stage("not-a-stage", OrchestratorState(report_only=True, root=tmp_path), tmp_path)
    assert result.blocker == "unknown_dynamic_stage"


@pytest.mark.parametrize(
    ("execution_allowed", "expected_status"),
    [("False", StageStatus.BLOCKED), ("true", StageStatus.PASSED)],
)
def test_hyperliquid_stage_strictly_parses_execution_allowed(
    monkeypatch, tmp_path, execution_allowed, expected_status
):
    monkeypatch.setattr(
        "quant_platform.orchestration.dynamic_stage_runner.run_hyperliquid_research_cycle",
        lambda *, root, collect_l2: {
            "execution_allowed": execution_allowed,
            "blockers": "research_blocked",
            "receipt": "receipt.csv",
            "receipt_json": "receipt.json",
            "summary": "summary.md",
            "step_count": 0,
        },
    )
    result = run_dynamic_stage(
        "hyperliquid_research_cycle", OrchestratorState(root=tmp_path), tmp_path
    )
    assert result.status is expected_status
    if expected_status is StageStatus.BLOCKED:
        assert result.blocker == "research_blocked"
