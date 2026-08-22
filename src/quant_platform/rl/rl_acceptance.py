from __future__ import annotations

import numpy as np
import pandas as pd


MINIMUM_TAKE_RATE = 0.10
MAXIMUM_CONCENTRATION = 0.65


def minimum_trade_count(total_rows: object) -> int:
    return max(20, int(np.ceil(float(total_rows or 0) * MINIMUM_TAKE_RATE)))


def rl_acceptance_report(evaluation: pd.DataFrame) -> pd.DataFrame:
    if evaluation.empty:
        return pd.DataFrame([_row(False, "missing_rl_evaluation")])
    if "evaluation_split" not in evaluation.columns:
        return pd.DataFrame([_row(False, "missing_rl_out_of_sample_evidence")])
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
) -> dict[str, object]:
    returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = float(gains / losses) if losses else float("inf") if gains > 0 else 0.0
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    drawdown = ((peak - equity) / peak.replace(0, np.nan)).fillna(0.0)
    sample_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    pair_conc = _concentration(frame, "pair")
    raw_timeframe_conc = _concentration(frame, "timeframe")
    source = source_frame if source_frame is not None else frame
    timeframe_conc = _source_adjusted_selection_concentration(frame, source, "timeframe")
    return {
        "variant": variant,
        "trades": int(len(returns)),
        "take_rate": float(len(returns) / max(total_rows, 1)),
        "profit_factor": profit_factor,
        "sharpe": float(returns.mean() / sample_std * np.sqrt(len(returns))) if sample_std else 0.0,
        "max_drawdown": float(drawdown.max() if not drawdown.empty else 0.0),
        "total_return": float((1.0 + returns).clip(lower=0.0).prod() - 1.0),
        "pair_concentration": pair_conc,
        "timeframe_concentration": timeframe_conc,
        "raw_timeframe_concentration": raw_timeframe_conc,
        "pair_pnl_concentration": _positive_pnl_concentration(frame, returns, "pair"),
        "timeframe_pnl_concentration": _positive_pnl_concentration(frame, returns, "timeframe"),
        "regime_concentration": _concentration(frame, "regime"),
        "regime_pnl_concentration": _positive_pnl_concentration(frame, returns, "regime"),
    }


def _concentration(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 1.0
    counts = frame[column].astype(str).value_counts()
    return float(counts.iloc[0] / max(counts.sum(), 1)) if not counts.empty else 1.0


def _source_adjusted_selection_concentration(
    selected: pd.DataFrame,
    source: pd.DataFrame,
    column: str,
) -> float:
    if source.empty or column not in source.columns or selected.empty or column not in selected.columns:
        return 1.0
    source_counts = source[column].fillna("UNKNOWN").astype(str).value_counts()
    selected_counts = selected[column].fillna("UNKNOWN").astype(str).value_counts()
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
            "positive_pnl": pd.to_numeric(returns, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(),
        }
    ).groupby("group", dropna=False)["positive_pnl"].sum()
    total = float(contribution.sum())
    return float(contribution.max() / total) if total > 0 and not contribution.empty else 1.0


def _row(accepted: bool, blocker: str) -> dict[str, object]:
    return {"accepted": accepted, "blocker": blocker, "acceptance_reason": "passed" if accepted else blocker}
