"""AST-isolated actual RL consumer helper; never import orchestration or train.

Pure acceptance helpers are loaded separately. No simulator implementation is
executed by this suite, and this is not a complete RL integration certificate.
"""
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import quant_platform


SOURCE_ROOT = Path(quant_platform.__file__).resolve().parent
BACKTEST_PATH = SOURCE_ROOT / "rl/rl_backtest.py"
BACKTEST_TREE = ast.parse(BACKTEST_PATH.read_text(encoding="utf-8"), filename=str(BACKTEST_PATH))
SELECTED = [node for node in BACKTEST_TREE.body
            if isinstance(node, ast.FunctionDef) and node.name == "_return_column"]
assert len(SELECTED) == 1
NAMESPACE = {"np": np, "pd": pd}
exec(compile(ast.Module(body=SELECTED, type_ignores=[]), str(BACKTEST_PATH), "exec"), NAMESPACE)
extract_returns = NAMESPACE["_return_column"]

ACCEPTANCE_SPEC = importlib.util.spec_from_file_location(
    "static_desk_consumer_acceptance_probe", SOURCE_ROOT / "rl/rl_acceptance.py"
)
acceptance = importlib.util.module_from_spec(ACCEPTANCE_SPEC)
ACCEPTANCE_SPEC.loader.exec_module(acceptance)


def _frame(values):
    count = len(values)
    return pd.DataFrame({
        "profit_after_cost": values,
        "pair": ["X-Y"] * count,
        "timeframe": ["1h"] * count,
        "regime": ["range"] * count,
    }, index=pd.date_range("2026-01-01", periods=count, freq="h", tz="UTC"))


@pytest.mark.parametrize("column", ["profit_after_cost", "realized_return", "trade_return", "return", "returns"])
def test_actual_return_helper_preserves_supported_valid_outcomes(column):
    frame = _frame(["0.02", "-0.01", "0.0"]).rename(columns={"profit_after_cost": column})
    result = extract_returns(frame)
    assert result.index.identical(frame.index)
    np.testing.assert_array_equal(result, [0.02, -0.01, 0.0])
    assert result.name == column


@pytest.mark.parametrize("bad", [np.nan, None, pd.NA, "missing", np.inf, -np.inf, True, False, 0j, 1j])
def test_invalid_return_is_not_silently_zeroed_before_summary(bad):
    frame = _frame([0.02, bad, -0.01])
    result = extract_returns(frame)
    assert result.index.identical(frame.index)
    assert len(result) == 3
    assert not np.isfinite(result.iloc[1])
    summary = acceptance.return_summary(
        "non_rl_baseline", frame, result, len(frame), source_frame=frame
    )
    assert summary["trades"] == 3
    assert summary["metrics_status"] == "unqualified"
    assert "empty_missing_nonfinite_or_invalid_trade_returns" in summary["metrics_reason"]
    assert np.isnan(summary["profit_factor"])
    assert np.isnan(summary["sharpe"])


def test_missing_return_column_keeps_population_but_no_fabricated_zero():
    frame = _frame([0.02, -0.01, 0.005]).drop(columns="profit_after_cost")
    result = extract_returns(frame)
    assert result.index.identical(frame.index)
    assert len(result) == len(frame)
    assert result.isna().all()
    summary = acceptance.return_summary("non_rl_baseline", frame, result, len(frame), source_frame=frame)
    assert summary["metrics_status"] == "unqualified"
    assert summary["trades"] == len(frame)
    assert np.isnan(summary["total_return"])


def test_primary_return_field_is_not_replaced_by_favorable_fallback():
    frame = _frame([0.02, np.nan, -0.01]).assign(realized_return=[1.0, 1.0, 1.0])
    result = extract_returns(frame)
    assert result.name == "profit_after_cost"
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == -0.01


def test_duplicate_columns_fail_closed_at_extraction():
    frame = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], columns=["return", "return"])
    result = extract_returns(frame)
    assert len(result) == len(frame)
    assert result.isna().all()


def test_empty_population_is_not_a_qualified_equity_stream():
    frame = pd.DataFrame(index=pd.Index([], name="empty"))
    result = extract_returns(frame)
    assert result.empty and result.index.identical(frame.index)
    summary = acceptance.return_summary("non_rl_baseline", frame, result, 0, source_frame=frame)
    assert summary["metrics_status"] == "unqualified"


def test_existing_legacy_baseline_call_cannot_invent_equity_qualification():
    frame = _frame([0.02, -0.01, 0.005, -0.002])
    # Mirrors the real existing caller's arguments, without running research.
    summary = acceptance.return_summary(
        "non_rl_baseline", frame, extract_returns(frame), len(frame), source_frame=frame
    )
    assert summary["metrics_status"] == "unqualified"
    assert summary["return_basis"] == "legacy_trade_sequence_diagnostic_only"
    assert "explicit_calendar_equity_contract_required" in summary["metrics_reason"]
    assert np.isnan(summary["sharpe"])
    assert np.isfinite(summary["legacy_trade_sequence_total_return"])
    report = acceptance.rl_acceptance_report(pd.DataFrame([
        {**summary, "evaluation_split": split, "variant": variant}
        for split in ("validation", "held_out_test")
        for variant in ("non_rl_baseline", "safe_rl_policy")
    ]))
    assert not report.iloc[0]["accepted"]
    assert not report.iloc[0]["capital_authority"]
    assert not report.iloc[0]["order_submission"]


def test_inspected_consumer_calls_still_require_explicit_equity_integration():
    # Source inspection, not end-to-end execution: these remain legacy callers.
    research = next(node for node in BACKTEST_TREE.body
                    if isinstance(node, ast.FunctionDef) and node.name == "run_rl_research")
    calls = [node for node in ast.walk(research) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "return_summary"]
    assert len(calls) == 2
    for call in calls:
        assert "source_frame" in {keyword.arg for keyword in call.keywords}
        assert not {"equity_returns", "equity_contract"} & {keyword.arg for keyword in call.keywords}
