from __future__ import annotations

from quant_platform.orchestration.dynamic_stage_runner import run_dynamic_stage
from quant_platform.orchestration.state import OrchestratorState, StageResult


def test_copula_stage_returns_a_real_stage_result_without_ml_imports(tmp_path):
    result = run_dynamic_stage("copula_shadow_comparison", OrchestratorState(pair_id="BNB-USD/WLD-USD", report_only=True, root=tmp_path), tmp_path)
    assert isinstance(result, StageResult)
    assert result.stage == "copula_shadow_comparison"
    assert "copula_shadow_" in result.reason


def test_dynamic_stage_rejects_unknown_stage_without_execution_fallback(tmp_path):
    result = run_dynamic_stage("not-a-stage", OrchestratorState(report_only=True, root=tmp_path), tmp_path)
    assert result.blocker == "unknown_dynamic_stage"
