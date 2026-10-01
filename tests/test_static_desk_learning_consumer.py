"""Execute the actual pure consumer bodies without importing its effects layer.

This does not run training, publication, export or the orchestration package.
"""
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import quant_platform


ROOT = Path(quant_platform.__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("learning_consumer_acceptance", ROOT / "rl/rl_acceptance.py")
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)
SOURCE = ROOT / "rl/rl_learning_agent.py"
TREE = ast.parse(SOURCE.read_text())
NODES = [node for node in TREE.body if isinstance(node, ast.FunctionDef)
         and node.name in {"_return_series", "_policy_gate_outcomes"}]
assert len(NODES) == 2
NAMESPACE = {"np": np, "pd": pd, "minimum_trade_count": metrics.minimum_trade_count,
             "MAXIMUM_CONCENTRATION": metrics.MAXIMUM_CONCENTRATION,
             "MINIMUM_TAKE_RATE": metrics.MINIMUM_TAKE_RATE}
exec(compile(ast.Module(body=NODES, type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)


@pytest.mark.parametrize("value", [np.nan, None, "broken", np.inf, -np.inf, True, False, 0j, 1j])
def test_actual_learning_return_extractor_does_not_zero_fill(value):
    frame = pd.DataFrame({"profit_after_cost": [.02, value, -.01]})
    values = NAMESPACE["_return_series"](frame)
    assert not np.isfinite(values.iloc[1])
    summary = metrics.return_summary("diagnostic", frame, values, len(frame))
    assert summary["metrics_status"] == "unqualified"
    assert np.isnan(summary["sharpe"])


def test_absent_return_column_is_not_a_flat_return_stream():
    result = NAMESPACE["_return_series"](pd.DataFrame({"pair": ["a", "b"]}))
    assert result.isna().all()


@pytest.mark.parametrize("status", [None, "unqualified", "legacy"])
def test_attractive_but_unqualified_metrics_cannot_pass_learning_consumer(status):
    baseline = dict(profit_factor=1., max_drawdown=.2, sharpe=.5, trades=40,
                    take_rate=1., pair_concentration=.25, pair_pnl_concentration=.25,
                    timeframe_concentration=.25, timeframe_pnl_concentration=.25,
                    regime_concentration=.25, regime_pnl_concentration=.25,
                    metrics_status="qualified")
    challenger = {**baseline, "profit_factor": 2., "metrics_status": status}
    output = NAMESPACE["_policy_gate_outcomes"](challenger, baseline, 40, prefix="validation")
    assert not output["validation_eligible"]
    assert "qualified_calendar_metrics" in output["validation_gate_failures"]


def test_declared_qualified_control_preserves_existing_numeric_thresholds():
    baseline = dict(profit_factor=1., max_drawdown=.2, sharpe=.5, trades=40,
                    take_rate=1., pair_concentration=.25, pair_pnl_concentration=.25,
                    timeframe_concentration=.25, timeframe_pnl_concentration=.25,
                    regime_concentration=.25, regime_pnl_concentration=.25,
                    metrics_status="qualified")
    challenger = {**baseline, "profit_factor": 2.}
    assert NAMESPACE["_policy_gate_outcomes"](challenger, baseline, 40, prefix="validation")["validation_eligible"]
    challenger["max_drawdown"] = 0.
    assert NAMESPACE["_policy_gate_outcomes"](challenger, baseline, 40, prefix="validation")["validation_eligible"]
    challenger["sharpe"] = None
    assert not NAMESPACE["_policy_gate_outcomes"](challenger, baseline, 40, prefix="validation")["validation_eligible"]
