"""Offline acceptance controls; load only the pure module, never the RL package."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import quant_platform


MODULE_PATH = Path(quant_platform.__file__).resolve().parent / "rl/rl_acceptance.py"
SPEC = importlib.util.spec_from_file_location("static_desk_acceptance_probe", MODULE_PATH)
acceptance = importlib.util.module_from_spec(SPEC)
# Deliberately do not register in sys.modules: historical direct probes do not.
SPEC.loader.exec_module(acceptance)


def _frame(n=40):
    groups = np.arange(n) % 4
    return pd.DataFrame({"pair": groups, "timeframe": groups, "regime": groups})


def _contract(stream="equity-stream", interval="1d"):
    return acceptance.EquityReturnContract(stream, interval, "common-initial-capital", "common_capital_net_pnl_fraction")


def _equity(start="2026-01-01", values=(.02, -.01, .015, -.005)):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="1D", tz="UTC"))


def _summary(variant="safe_rl_policy", start="2026-01-01", baseline=False):
    frame = _frame()
    # Every group has positive and negative outcomes; no concentrated winner.
    trades = pd.Series(np.tile([.01] * 4 + [-.01] * 4, 5) if baseline else np.tile([.02] * 4 + [-.01] * 4, 5))
    equity = _equity(start, (.01, -.02, .005, -.002) if baseline else (.02, -.01, .015, -.005))
    return acceptance.return_summary(variant, frame, trades, 40, source_frame=frame,
                                     equity_returns=equity, equity_contract=_contract(variant + start))


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _comparison():
    rows = []
    for split, start in (("validation", "2026-01-01"), ("held_out_test", "2026-02-01")):
        for variant in ("non_rl_baseline", "safe_rl_policy"):
            row = _summary(variant, start, variant == "non_rl_baseline")
            assert row["metrics_status"] == "qualified", row["metrics_reason"]
            for field in acceptance.GLOBAL_IDENTITIES:
                row[field] = _hash(field) if field.endswith("_sha256") else "experiment-1"
            row.update({"evaluation_split": split, "split_id": split, "policy_id": variant,
                        "training_label_cutoff_utc": "2025-12-01T00:00:00Z",
                        "model_fitted_at_utc": "2025-12-02T00:00:00Z"})
            for field in ("source_partition_sha256", "opportunity_set_sha256", "fold_receipt_sha256"):
                row[field] = _hash(split + field)
            for field in ("policy_configuration_sha256", "model_artifact_sha256"):
                row[field] = _hash(variant + field)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, None, "not-a-return", True, 1j])
def test_bad_trade_observation_never_becomes_zero(bad):
    values = pd.Series([.02, bad, -.01, .01], dtype=object)
    result = acceptance.return_summary("x", _frame(4), values, 4)
    assert result["trades"] == 4
    assert result["metrics_status"] == "unqualified"
    assert np.isnan(result["profit_factor"])
    assert np.isnan(result["total_return"])
    assert np.isnan(result["sharpe"])


def test_legacy_trade_timeframe_does_not_annualize_equity():
    frame = _frame(4).assign(timeframe="1d")
    result = acceptance.return_summary("x", frame, pd.Series([.02, -.01, .015, -.005]), 4)
    assert np.isnan(result["sharpe"])
    assert result["metrics_status"] == "unqualified"
    assert "explicit_calendar_equity_contract_required" in result["metrics_reason"]
    assert result["total_return"] == pytest.approx(np.prod([1.02, .99, 1.015, .995]) - 1)


def test_explicit_calendar_equity_is_independent_of_trade_event_count():
    result = _summary()
    values = np.array([.02, -.01, .015, -.005])
    assert result["metrics_status"] == "qualified"
    assert result["sharpe"] == pytest.approx(values.mean() / values.std(ddof=1) * np.sqrt(365))
    assert result["trades"] == 40 and result["equity_period_count"] == 4
    assert result["total_return"] == pytest.approx(np.prod(1 + values) - 1)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reverse", "naive", "constant", "wrong_interval", "terminal", "no_basis"])
def test_bad_calendar_or_contract_fails_qualification(mutation):
    frame = _frame(4)
    eq = _equity()
    contract = _contract()
    if mutation == "missing":
        eq = eq.drop(eq.index[1])
    elif mutation == "duplicate":
        eq.index = pd.DatetimeIndex([eq.index[0], eq.index[0], eq.index[2], eq.index[3]])
    elif mutation == "reverse":
        eq = eq.iloc[::-1]
    elif mutation == "naive":
        eq.index = eq.index.tz_localize(None)
    elif mutation == "constant":
        eq[:] = .01
    elif mutation == "wrong_interval":
        contract = _contract(interval="1h")
    elif mutation == "terminal":
        eq.iloc[1] = -1.1
    else:
        contract = acceptance.EquityReturnContract("s", "1d", "c", "per_trade_return")
    result = acceptance.return_summary("x", frame, pd.Series([.02, -.01, .015, -.005]), 4,
                                        equity_returns=eq, equity_contract=contract)
    assert result["metrics_status"] == "unqualified"
    assert np.isnan(result["sharpe"])
    if mutation == "terminal":
        assert result["equity_terminal_loss_observed"]


def test_undefined_profit_factor_cannot_qualify():
    result = acceptance.return_summary("x", _frame(4), pd.Series([.01] * 4), 4,
                                       equity_returns=_equity(), equity_contract=_contract())
    assert np.isinf(result["profit_factor"])
    assert result["metrics_status"] == "unqualified"


@pytest.mark.parametrize("mutation", ["duplicate", "foreign", "return_mismatch", "source_duplicate"])
def test_trade_and_source_observation_identity_is_required(mutation):
    source, frame = _frame(4), _frame(4)
    returns = pd.Series([.02, -.01, .015, -.005])
    if mutation == "duplicate":
        frame.index = returns.index = pd.Index([0, 0, 2, 3])
    elif mutation == "foreign":
        frame.index = returns.index = pd.Index([4, 5, 6, 7])
    elif mutation == "source_duplicate":
        source.index = pd.Index([0, 0, 2, 3])
    else:
        returns.index = pd.Index([3, 2, 1, 0])
    result = acceptance.return_summary("x", frame, returns, 4, source_frame=source,
                                       equity_returns=_equity(), equity_contract=_contract())
    assert result["metrics_status"] == "unqualified"


def test_legacy_terminal_trade_is_not_clipped_into_a_valid_return():
    result = acceptance.return_summary("x", _frame(4), pd.Series([.2, -1.1, .2, -.1]), 4)
    assert result["trade_loss_exceeds_reference_capital"]
    assert np.isnan(result["total_return"])
    assert result["metrics_status"] == "unqualified"


@pytest.mark.parametrize("column", ["pair", "timeframe", "regime"])
@pytest.mark.parametrize("target", ["frame", "source"])
def test_duplicate_frame_or_source_columns_fail_closed(column, target):
    frame, source = _frame(4), _frame(4)
    if target == "frame":
        frame = pd.concat([frame, frame[[column]]], axis=1)
    else:
        source = pd.concat([source, source[[column]]], axis=1)
    result = acceptance.return_summary("x", frame, pd.Series([.02, -.01, .015, -.005]), 4,
                                        source_frame=source, equity_returns=_equity(), equity_contract=_contract())
    assert result["metrics_status"] == "unqualified"
    assert np.isnan(result["sharpe"])


def test_qualified_comparison_control_and_order_invariance():
    frame = _comparison()
    for ordering in (frame, frame.iloc[::-1]):
        row = acceptance.rl_acceptance_report(ordering).iloc[0]
        assert bool(row["accepted"]), row["blocker"]
        assert bool(row["out_of_sample_evidence"])
        assert not bool(row["capital_authority"])
        assert "not_source_authentication_or_gate" in row["evidence_scope"]


@pytest.mark.parametrize("field", acceptance.GLOBAL_IDENTITIES + acceptance.SPLIT_IDENTITIES)
def test_mismatched_comparison_identity_is_blocked(field):
    frame = _comparison()
    frame.loc[0, field] = _hash("other") if field.endswith("_sha256") else "other"
    row = acceptance.rl_acceptance_report(frame).iloc[0]
    assert not bool(row["accepted"])
    assert not bool(row["out_of_sample_evidence"])


@pytest.mark.parametrize("mutation", ["duplicate", "extra", "missing", "invalid_metric", "unqualified", "overlap", "refit", "population", "reused_split", "malformed_split", "invalid_count", "invalid_clock", "missing_identity", "duplicate_column"])
def test_ambiguous_or_unqualified_comparison_is_blocked(mutation):
    frame = _comparison()
    if mutation == "duplicate":
        frame.iloc[1] = frame.iloc[0]
    elif mutation == "extra":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif mutation == "missing":
        frame = frame.iloc[:3]
    elif mutation == "invalid_metric":
        frame.loc[0, "sharpe"] = np.inf
    elif mutation == "unqualified":
        frame.loc[0, "metrics_status"] = "unqualified"
    elif mutation == "overlap":
        frame.loc[2:, "equity_start_utc"] = frame.loc[0, "equity_start_utc"]
        frame.loc[2:, "equity_end_utc"] = frame.loc[0, "equity_end_utc"]
    elif mutation == "refit":
        frame.loc[3, "model_artifact_sha256"] = _hash("refitted")
    elif mutation == "population":
        frame.loc[0, "opportunity_count"] = 39
    elif mutation == "reused_split":
        frame["opportunity_set_sha256"] = _hash("same")
    elif mutation == "malformed_split":
        frame["evaluation_split"] = frame["evaluation_split"].astype(object)
        frame.at[0, "evaluation_split"] = {"split": "validation"}
    elif mutation == "invalid_count":
        frame.loc[0, "trades"] = np.nan
    elif mutation == "invalid_clock":
        frame.loc[0, "model_fitted_at_utc"] = "2026-01-02T00:00:00Z"
    elif mutation == "missing_identity":
        frame = frame.drop(columns="cost_model_sha256")
    else:
        frame = pd.concat([frame, frame[["sharpe"]]], axis=1)
    row = acceptance.rl_acceptance_report(frame).iloc[0]
    assert not bool(row["accepted"])
    assert not bool(row["out_of_sample_evidence"])


def test_economic_thresholds_not_weakened_by_qualification():
    frame = _comparison()
    frame.loc[frame.variant.eq("safe_rl_policy"), "total_return"] = -.01
    row = acceptance.rl_acceptance_report(frame).iloc[0]
    assert not bool(row["accepted"])
    assert "positive_after_cost_return" in row["held_out_test_gate_failures"]


def test_legacy_summary_rows_are_reportable_but_never_accepted():
    frame = _comparison().drop(columns=list(acceptance.GLOBAL_IDENTITIES))
    row = acceptance.rl_acceptance_report(frame).iloc[0]
    assert not bool(row["accepted"])
    assert "unqualified_rl_comparison:missing_fields" in row["blocker"]


def test_fit_inside_first_equity_period_is_not_causal():
    frame = _comparison()
    frame["model_fitted_at_utc"] = "2025-12-31T12:00:00Z"
    row = acceptance.rl_acceptance_report(frame).iloc[0]
    assert not bool(row["accepted"])
    assert "training_or_fit_overlaps_evaluation" in row["blocker"]


def test_period_end_order_alone_does_not_prove_disjoint_splits():
    frame = _comparison()
    # Validation ends Jan 4; first test return covers Jan 3 noon--Jan 4 noon.
    frame.loc[2:, "equity_start_utc"] = "2026-01-04T12:00:00Z"
    frame.loc[2:, "equity_end_utc"] = "2026-01-07T12:00:00Z"
    row = acceptance.rl_acceptance_report(frame).iloc[0]
    assert not bool(row["accepted"])
    assert "overlapping_evaluation_splits" in row["blocker"]
