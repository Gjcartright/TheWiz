from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "script_name",
    ["run_evidence_pipeline_phase1", "run_evidence_pipeline_phase3_quality"],
)
def test_legacy_evidence_pipeline_uses_sample_sharpe_and_initial_equity(script_name: str):
    module = _load_script(script_name)
    returns = [0.10, -0.05]
    equity = [1.10, 0.80]

    result = module.metric_summary(returns, equity, returns, [1, 1], 365)

    expected_sharpe = math.sqrt(365.0) * pd.Series(returns).mean() / pd.Series(returns).std(ddof=1)
    assert result["sharpe"] == pytest.approx(expected_sharpe)
    assert result["max_drawdown"] == pytest.approx(1.0 - 0.80 / 1.10)


@pytest.mark.parametrize(
    "script_name",
    ["run_evidence_pipeline_phase1", "run_evidence_pipeline_phase3_quality"],
)
def test_legacy_evidence_pipeline_counts_first_bar_loss_in_drawdown(script_name: str):
    module = _load_script(script_name)

    result = module.metric_summary([-0.10, -0.10], [0.90, 0.81], [-0.19], [2], 365)

    assert result["max_drawdown"] == pytest.approx(0.19)
