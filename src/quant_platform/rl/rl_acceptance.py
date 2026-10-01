from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
import re

import numpy as np
import pandas as pd

from quant_platform.performance_math import calculate_annualized_sharpe, normalize_interval, PERIODS_PER_YEAR

MINIMUM_TAKE_RATE = 0.10
MAXIMUM_CONCENTRATION = 0.65
EVIDENCE_SCOPE = "supplied_metrics_and_identity_validation_not_source_authentication_or_gate"
GLOBAL_IDENTITIES = (
    "experiment_id", "data_snapshot_sha256", "evaluation_configuration_sha256",
    "cost_model_sha256", "risk_policy_sha256", "universe_sha256", "comparison_manifest_sha256",
)
SPLIT_IDENTITIES = (
    "split_id", "source_partition_sha256", "opportunity_set_sha256", "fold_receipt_sha256",
    "equity_calendar_sha256", "equity_interval", "capital_basis_id", "profit_factor_basis",
)
POLICY_IDENTITIES = (
    "policy_id", "policy_configuration_sha256", "model_artifact_sha256",
    "equity_stream_id", "equity_stream_sha256",
)
METRIC_FIELDS = (
    "profit_factor", "max_drawdown", "sharpe", "total_return", "trades", "take_rate",
    "pair_concentration", "pair_pnl_concentration", "timeframe_concentration",
    "timeframe_pnl_concentration", "regime_concentration", "regime_pnl_concentration",
)


@dataclass(frozen=True)
class EquityReturnContract:
    """Declaration for a separate net simple account-equity calendar stream.

    Timestamps are period ends. The caller supplies every regular period,
    including flat periods, and net costs. Trade P&L fractions use one common
    capital denominator; this enables
    profit-factor aggregation. This declaration is not source authentication.
    """

    stream_id: str
    interval: str
    capital_basis_id: str
    trade_return_basis: str


def _finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and bool(np.isfinite(value))


def _valid_identity(field: str, value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    return not field.endswith("_sha256") or re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _utc_time(value: object) -> pd.Timestamp | None:
    if not isinstance(value, (str, pd.Timestamp)):
        return None
    try:
        stamp = pd.Timestamp(value)
        return stamp.tz_convert("UTC") if not pd.isna(stamp) and stamp.tzinfo is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _comparison_failure(evaluation: pd.DataFrame) -> str:
    """Validate the exact four-row comparison, never silently select row zero."""
    required = set(GLOBAL_IDENTITIES + SPLIT_IDENTITIES + POLICY_IDENTITIES + METRIC_FIELDS) | {
        "evaluation_split", "variant", "metrics_status", "opportunity_count",
        "equity_start_utc", "equity_end_utc", "equity_period_count",
        "training_label_cutoff_utc", "model_fitted_at_utc",
    }
    if not evaluation.columns.is_unique:
        return "duplicate_columns"
    if missing := sorted(required - set(evaluation.columns)):
        return "missing_fields:" + ",".join(missing)
    expected = {(split, variant) for split in ("validation", "held_out_test")
                for variant in ("non_rl_baseline", "safe_rl_policy")}
    pairs = list(zip(evaluation["evaluation_split"], evaluation["variant"]))
    if len(pairs) != 4 or any(not isinstance(a, str) or not isinstance(b, str) for a, b in pairs):
        return "exact_four_unique_rows_required"
    if set(pairs) != expected:
        return "exact_four_unique_rows_required"
    for field in GLOBAL_IDENTITIES + SPLIT_IDENTITIES + POLICY_IDENTITIES:
        if not all(_valid_identity(field, value) for value in evaluation[field]):
            return "invalid_identity:" + field
    for field in GLOBAL_IDENTITIES:
        if evaluation[field].nunique() != 1:
            return "mismatched_identity:" + field
    split_times = {}
    for split in ("validation", "held_out_test"):
        rows = evaluation[evaluation["evaluation_split"].eq(split)]
        for field in SPLIT_IDENTITIES:
            if rows[field].nunique() != 1:
                return "mismatched_split_identity:" + field
        for _, row in rows.iterrows():
            if not isinstance(row["metrics_status"], str) or row["metrics_status"] != "qualified":
                return "unqualified_metrics"
            if not all(_finite_number(row[field]) for field in METRIC_FIELDS):
                return "nonfinite_or_invalid_metrics"
            for field in ("trades", "opportunity_count", "equity_period_count"):
                value = row[field]
                if not _finite_number(value) or value < 0 or int(value) != value:
                    return "invalid_count:" + field
            if row["opportunity_count"] <= 0 or row["equity_period_count"] < 2 or row["trades"] > row["opportunity_count"]:
                return "invalid_opportunity_or_equity_count"
            if not np.isclose(row["take_rate"], row["trades"] / row["opportunity_count"], rtol=1e-12, atol=1e-12):
                return "inconsistent_take_rate"
            if row["profit_factor"] < 0 or not 0 <= row["max_drawdown"] < 1 or row["total_return"] <= -1:
                return "invalid_metric_domain"
            if any(not 0 <= row[field] <= 1 for field in METRIC_FIELDS if "concentration" in field or field == "take_rate"):
                return "invalid_concentration_or_take_rate"
            if row["profit_factor_basis"] != "common_capital_net_pnl_fraction":
                return "unqualified_profit_factor_basis"
            interval = normalize_interval(row["equity_interval"])
            if interval is None:
                return "unsupported_equity_interval"
            start, end, cutoff, fitted = (_utc_time(row[field]) for field in (
                "equity_start_utc", "equity_end_utc", "training_label_cutoff_utc", "model_fitted_at_utc"))
            if any(stamp is None for stamp in (start, end, cutoff, fitted)):
                return "invalid_causality_clock"
            seconds = 365 * 86400 / PERIODS_PER_YEAR[interval]
            window_start = start - pd.Timedelta(seconds=seconds)
            if not cutoff <= fitted <= window_start < start < end:
                return "training_or_fit_overlaps_evaluation"
            if not np.isclose((end - start).total_seconds(), (row["equity_period_count"] - 1) * seconds, rtol=0, atol=1e-6):
                return "equity_calendar_count_mismatch"
            split_times[split] = (window_start, end)
        for field in ("opportunity_count", "equity_period_count", "equity_start_utc", "equity_end_utc"):
            if rows[field].nunique() != 1:
                return "mismatched_comparison_population:" + field
    if split_times["validation"][1] > split_times["held_out_test"][0]:
        return "overlapping_evaluation_splits"
    for field in ("split_id", "source_partition_sha256", "opportunity_set_sha256", "fold_receipt_sha256"):
        if evaluation[field].nunique() != 2:
            return "reused_split_identity:" + field
    for variant in ("non_rl_baseline", "safe_rl_policy"):
        rows = evaluation[evaluation["variant"].eq(variant)]
        for field in ("policy_id", "policy_configuration_sha256", "model_artifact_sha256",
                      "training_label_cutoff_utc", "model_fitted_at_utc"):
            if rows[field].nunique() != 1:
                return "changed_evaluated_policy_identity:" + field
    if evaluation["policy_id"].nunique() != 2:
        return "distinct_policy_identities_required"
    return ""


def minimum_trade_count(total_rows: object) -> int:
    return max(20, int(np.ceil(float(total_rows or 0) * MINIMUM_TAKE_RATE)))


def rl_acceptance_report(evaluation: pd.DataFrame) -> pd.DataFrame:
    if evaluation.empty:
        return pd.DataFrame([_row(False, "missing_rl_evaluation")])
    if "evaluation_split" not in evaluation.columns:
        return pd.DataFrame([_row(False, "missing_rl_out_of_sample_evidence")])
    if failure := _comparison_failure(evaluation):
        return pd.DataFrame([_row(False, "unqualified_rl_comparison:" + failure)])
    validation = _split_gate(evaluation, "validation")
    test = _split_gate(evaluation, "held_out_test")
    if validation.get("missing") or test.get("missing"):
        missing = ";".join(
            split for split, result in (("validation", validation), ("held_out_test", test)) if result.get("missing")
        )
        return pd.DataFrame([_row(False, f"missing_rl_oos_split:{missing}")])
    accepted = bool(validation["accepted"] and test["accepted"])
    blocker = "" if accepted else "rl_validation_or_held_out_test_gates_not_met"
    raw_row = test["raw_row"]
    rl_row = test["rl_row"]
    validation_rl_row = validation["rl_row"]
    return pd.DataFrame(
        [
            {
                "accepted": accepted,
                "blocker": blocker,
                "raw_profit_factor": raw_row["profit_factor"],
                "rl_profit_factor": rl_row["profit_factor"],
                "raw_total_return": raw_row.get("total_return", np.nan),
                "rl_total_return": rl_row.get("total_return", np.nan),
                "validation_rl_total_return": validation_rl_row.get(
                    "total_return", np.nan
                ),
                "held_out_test_rl_total_return": rl_row.get(
                    "total_return", np.nan
                ),
                "raw_drawdown": raw_row["max_drawdown"],
                "rl_drawdown": rl_row["max_drawdown"],
                "rl_trades": rl_row["trades"],
                "rl_take_rate": rl_row["take_rate"],
                "pair_concentration": rl_row["pair_concentration"],
                "timeframe_concentration": rl_row["timeframe_concentration"],
                "raw_timeframe_concentration": rl_row.get("raw_timeframe_concentration", 1.0),
                "timeframe_pnl_concentration": rl_row.get("timeframe_pnl_concentration", 1.0),
                "pair_pnl_concentration": rl_row.get("pair_pnl_concentration", 1.0),
                "regime_concentration": rl_row.get("regime_concentration", 1.0),
                "regime_pnl_concentration": rl_row.get("regime_pnl_concentration", 1.0),
                "validation_passed": bool(validation["accepted"]),
                "held_out_test_passed": bool(test["accepted"]),
                "validation_gate_failures": validation["gate_failures"],
                "held_out_test_gate_failures": test["gate_failures"],
                "out_of_sample_evidence": True,
                "evidence_scope": EVIDENCE_SCOPE,
                "capital_authority": False,
                "order_submission": False,
                "acceptance_reason": "passed" if accepted else blocker,
            }
        ]
    )


def _split_gate(evaluation: pd.DataFrame, split: str) -> dict[str, object]:
    rows = evaluation[evaluation["evaluation_split"].astype(str).eq(split)]
    raw = rows[rows["variant"] == "non_rl_baseline"]
    rl = rows[rows["variant"] == "safe_rl_policy"]
    if raw.empty or rl.empty:
        return {"missing": True, "accepted": False, "gate_failures": "missing_baseline_or_rl_variant"}
    raw_row = raw.iloc[0]
    rl_row = rl.iloc[0]
    rl_total_return = pd.to_numeric(
        pd.Series([rl_row.get("total_return", np.nan)]), errors="coerce"
    ).iloc[0]
    checks = {
        "profit_factor_improves": rl_row["profit_factor"] > raw_row["profit_factor"],
        "positive_after_cost_return": bool(
            np.isfinite(rl_total_return) and rl_total_return > 0.0
        ),
        "drawdown_not_worse": rl_row["max_drawdown"] <= raw_row["max_drawdown"],
        "sharpe_not_materially_worse": rl_row["sharpe"] >= raw_row["sharpe"] - 0.25,
        "minimum_trades": rl_row["trades"] >= minimum_trade_count(raw_row["trades"]),
        "minimum_take_rate": rl_row["take_rate"] >= MINIMUM_TAKE_RATE,
        "pair_concentration": rl_row["pair_concentration"] <= MAXIMUM_CONCENTRATION,
        "pair_pnl_concentration": rl_row.get("pair_pnl_concentration", 1.0) <= MAXIMUM_CONCENTRATION,
        "timeframe_selection_concentration": rl_row["timeframe_concentration"] <= MAXIMUM_CONCENTRATION,
        "timeframe_pnl_concentration": rl_row.get("timeframe_pnl_concentration", 1.0) <= MAXIMUM_CONCENTRATION,
        "regime_concentration": rl_row.get("regime_concentration", 1.0) <= MAXIMUM_CONCENTRATION,
        "regime_pnl_concentration": rl_row.get("regime_pnl_concentration", 1.0) <= MAXIMUM_CONCENTRATION,
    }
    failures = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "missing": False,
        "accepted": not failures,
        "gate_failures": ";".join(failures),
        "raw_row": raw_row,
        "rl_row": rl_row,
    }


def return_summary(
    variant: str,
    frame: pd.DataFrame,
    returns: pd.Series,
    total_rows: int,
    *,
    source_frame: pd.DataFrame | None = None,
    equity_returns: pd.Series | None = None,
    equity_contract: EquityReturnContract | None = None,
) -> dict[str, object]:
    """Retain trade diagnostics; qualify only explicit net calendar-equity input.

    Legacy callers keep their keys but receive NaN Sharpe and an unqualified
    status. Per-trade timestamps/timeframes never establish an equity clock.
    Invalid observations are retained in the count, never replaced with zero.
    """
    reasons: list[str] = []
    values, valid = _finite_returns(returns)
    source = source_frame if source_frame is not None else frame
    count = len(returns) if isinstance(returns, pd.Series) else 0
    aligned = (isinstance(frame, pd.DataFrame) and isinstance(returns, pd.Series)
               and len(frame) == count and frame.index.equals(returns.index)
               and frame.index.is_unique and returns.index.is_unique and frame.columns.is_unique)
    if not valid:
        reasons.append("empty_missing_nonfinite_or_invalid_trade_returns")
    if not aligned:
        reasons.append("trade_frame_return_alignment_required")
    population_valid = (isinstance(total_rows, Integral) and not isinstance(total_rows, (bool, np.bool_))
                        and isinstance(source, pd.DataFrame) and total_rows == len(source)
                        and source.index.is_unique and source.columns.is_unique
                        and 0 < total_rows and count <= total_rows
                        and isinstance(frame, pd.DataFrame) and frame.index.isin(source.index).all())
    if not population_valid:
        reasons.append("invalid_source_population")
    financial_valid = valid and aligned
    gains = float(values[values > 0].sum()) if financial_valid else np.nan
    losses = float(-values[values < 0].sum()) if financial_valid else np.nan
    profit_factor = gains / losses if losses > 0 else np.inf if gains > 0 else np.nan
    trade_terminal = bool(financial_valid and (values <= -1).any())
    legacy_total, legacy_drawdown = _compound_metrics(values) if financial_valid else (np.nan, np.nan)
    row = {
        "variant": variant,
        "trades": count,
        "opportunity_count": total_rows,
        "take_rate": float(count / total_rows) if population_valid else np.nan,
        "profit_factor": profit_factor,
        "sharpe": np.nan,
        "max_drawdown": legacy_drawdown,
        "total_return": legacy_total,
        "legacy_trade_sequence_total_return": legacy_total,
        "legacy_trade_sequence_max_drawdown": legacy_drawdown,
        "trade_loss_exceeds_reference_capital": trade_terminal,
        "equity_terminal_loss_observed": False,
        "return_basis": "legacy_trade_sequence_diagnostic_only",
        "profit_factor_basis": "undeclared_trade_return_basis",
        "metrics_status": "unqualified",
        "pair_concentration": _concentration(frame, "pair"),
        "timeframe_concentration": _source_adjusted_selection_concentration(frame, source, "timeframe"),
        "raw_timeframe_concentration": _concentration(frame, "timeframe"),
        "pair_pnl_concentration": _positive_pnl_concentration(frame, values, "pair") if financial_valid else np.nan,
        "timeframe_pnl_concentration": _positive_pnl_concentration(frame, values, "timeframe") if financial_valid else np.nan,
        "regime_concentration": _concentration(frame, "regime"),
        "regime_pnl_concentration": _positive_pnl_concentration(frame, values, "regime") if financial_valid else np.nan,
    }
    if equity_returns is not None or equity_contract is not None:
        # A rejected explicit equity stream must not fall back to trade metrics
        # under the same aggregate keys. Legacy diagnostics remain named above.
        row.update({"total_return": np.nan, "max_drawdown": np.nan,
                    "return_basis": "unqualified_calendar_account_equity_returns"})
    if not isinstance(equity_contract, EquityReturnContract) or equity_returns is None:
        reasons.append("explicit_calendar_equity_contract_required")
    else:
        contract = equity_contract
        interval = normalize_interval(contract.interval)
        if not _valid_identity("stream_id", contract.stream_id) or not _valid_identity("capital_basis_id", contract.capital_basis_id) or interval is None:
            reasons.append("invalid_equity_contract")
        if contract.trade_return_basis != "common_capital_net_pnl_fraction":
            reasons.append("common_capital_net_pnl_fraction_required")
        eq, equity_valid = _finite_returns(equity_returns)
        calendar_valid = (isinstance(equity_returns, pd.Series)
                          and isinstance(equity_returns.index, pd.DatetimeIndex)
                          and equity_returns.index.tz is not None and len(equity_returns) >= 2)
        if not equity_valid or not calendar_valid:
            reasons.append("finite_timezone_aware_calendar_equity_required")
        else:
            sharpe = calculate_annualized_sharpe(eq, interval=contract.interval)
            if sharpe.status != "valid" or not _finite_number(sharpe.value):
                reasons.append("unqualified_equity_sharpe:" + sharpe.reason)
            eq_total, eq_drawdown = _compound_metrics(eq)
            row["equity_terminal_loss_observed"] = bool((eq <= -1).any())
            if not _finite_number(eq_total) or not _finite_number(eq_drawdown):
                reasons.append("invalid_or_terminal_equity_compounding")
            if not reasons:
                stamps = [stamp.isoformat() for stamp in eq.index.tz_convert("UTC")]
                row.update({
                    "sharpe": sharpe.value, "total_return": eq_total, "max_drawdown": eq_drawdown,
                    "return_basis": "net_simple_calendar_account_equity_returns",
                    "profit_factor_basis": contract.trade_return_basis,
                    "equity_stream_id": contract.stream_id, "capital_basis_id": contract.capital_basis_id,
                    "equity_interval": interval, "equity_start_utc": stamps[0], "equity_end_utc": stamps[-1],
                    "equity_period_count": len(eq), "equity_calendar_sha256": _digest(stamps),
                    "equity_stream_sha256": _digest([stamps, eq.tolist()]),
                })
    if not all(_finite_number(row[field]) for field in METRIC_FIELDS):
        reasons.append("nonfinite_or_undefined_metrics")
    row["metrics_status"] = "unqualified" if reasons else "qualified"
    row["metrics_reason"] = ";".join(sorted(set(reasons)))
    return row


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _finite_returns(returns: object) -> tuple[pd.Series, bool]:
    if not isinstance(returns, pd.Series):
        return pd.Series(dtype=float), False
    if returns.empty or any(isinstance(value, (bool, np.bool_, complex, np.complexfloating)) for value in returns):
        return pd.Series(np.nan, index=returns.index, dtype=float), False
    try:
        values = pd.to_numeric(returns, errors="coerce").astype(float)
        return values, bool(np.isfinite(values.to_numpy()).all())
    except (TypeError, ValueError, OverflowError):
        return pd.Series(np.nan, index=returns.index, dtype=float), False


def _compound_metrics(values: pd.Series) -> tuple[float, float]:
    if values.empty or (values <= -1).any():
        return np.nan, np.nan
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        equity = (1 + values).cumprod()
    if not np.isfinite(equity.to_numpy()).all() or (equity <= 0).any():
        return np.nan, np.nan
    peaks = equity.cummax().clip(lower=1.0)
    return float(equity.iloc[-1] - 1), float(((peaks - equity) / peaks).max())


def _concentration(frame: pd.DataFrame, column: str) -> float:
    if isinstance(frame, pd.DataFrame) and not frame.columns.is_unique:
        return np.nan
    if not isinstance(frame, pd.DataFrame) or frame.empty or column not in frame.columns:
        return 1.0
    counts = frame[column].astype(str).value_counts()
    return float(counts.iloc[0] / max(counts.sum(), 1)) if not counts.empty else 1.0


def _source_adjusted_selection_concentration(
    selected: pd.DataFrame,
    source: pd.DataFrame,
    column: str,
) -> float:
    if any(isinstance(frame, pd.DataFrame) and not frame.columns.is_unique for frame in (selected, source)):
        return np.nan
    if not isinstance(source, pd.DataFrame) or not isinstance(selected, pd.DataFrame) or source.empty or column not in source.columns or selected.empty or column not in selected.columns:
        return 1.0
    source_counts = source[column].fillna("UNKNOWN").astype(str).value_counts()
    selected_counts = selected[column].fillna("UNKNOWN").astype(str).value_counts()
    if not selected_counts.index.isin(source_counts.index).all() or (selected_counts > source_counts.reindex(selected_counts.index)).any():
        return np.nan
    selection_rates = selected_counts.reindex(source_counts.index, fill_value=0).div(source_counts)
    total_rate = float(selection_rates.sum())
    return float(selection_rates.max() / total_rate) if total_rate > 0 else 1.0


def _positive_pnl_concentration(
    frame: pd.DataFrame,
    returns: pd.Series,
    column: str,
) -> float:
    if frame.empty or column not in frame.columns or len(frame) != len(returns):
        return 1.0
    contribution = pd.DataFrame(
        {
            "group": frame[column].fillna("UNKNOWN").astype(str).to_numpy(),
            "positive_pnl": returns.clip(lower=0.0).to_numpy(),
        }
    ).groupby("group", dropna=False)["positive_pnl"].sum()
    total = float(contribution.sum())
    return float(contribution.max() / total) if total > 0 and not contribution.empty else 1.0


def _row(accepted: bool, blocker: str) -> dict[str, object]:
    return {"accepted": accepted, "blocker": blocker, "acceptance_reason": "passed" if accepted else blocker,
            "validation_passed": False, "held_out_test_passed": False,
            "out_of_sample_evidence": False, "evidence_scope": EVIDENCE_SCOPE,
            "capital_authority": False, "order_submission": False}
