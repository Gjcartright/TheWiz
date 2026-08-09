from __future__ import annotations

import numpy as np
import pandas as pd


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
    return pd.DataFrame(
        [
            {
                "accepted": accepted,
                "blocker": blocker,
                "raw_profit_factor": raw_row["profit_factor"],
                "rl_profit_factor": rl_row["profit_factor"],
                "raw_drawdown": raw_row["max_drawdown"],
                "rl_drawdown": rl_row["max_drawdown"],
                "rl_trades": rl_row["trades"],
                "rl_take_rate": rl_row["take_rate"],
                "pair_concentration": rl_row["pair_concentration"],
                "timeframe_concentration": rl_row["timeframe_concentration"],
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
    checks = {
        "profit_factor_improves": rl_row["profit_factor"] > raw_row["profit_factor"],
        "drawdown_not_worse": rl_row["max_drawdown"] <= raw_row["max_drawdown"],
        "sharpe_not_materially_worse": rl_row["sharpe"] >= raw_row["sharpe"] - 0.25,
        "minimum_trades": rl_row["trades"] >= max(20, raw_row["trades"] * 0.25),
        "minimum_take_rate": rl_row["take_rate"] >= 0.05,
        "pair_concentration": rl_row["pair_concentration"] <= 0.65,
        "timeframe_concentration": rl_row["timeframe_concentration"] <= 0.65,
    }
    failures = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "missing": False,
        "accepted": not failures,
        "gate_failures": ";".join(failures),
        "raw_row": raw_row,
        "rl_row": rl_row,
    }


def return_summary(variant: str, frame: pd.DataFrame, returns: pd.Series, total_rows: int) -> dict[str, object]:
    returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = float(gains / losses) if losses else float("inf") if gains > 0 else 0.0
    equity = (1.0 + returns).cumprod()
    drawdown = ((equity.cummax() - equity) / equity.cummax().replace(0, np.nan)).fillna(0.0)
    pair_conc = _concentration(frame, "pair")
    timeframe_conc = _concentration(frame, "timeframe")
    return {
        "variant": variant,
        "trades": int(len(returns)),
        "take_rate": float(len(returns) / max(total_rows, 1)),
        "profit_factor": profit_factor,
        "sharpe": float(returns.mean() / returns.std(ddof=0) * np.sqrt(len(returns))) if len(returns) > 1 and returns.std(ddof=0) else 0.0,
        "max_drawdown": float(drawdown.max() if not drawdown.empty else 0.0),
        "total_return": float(returns.sum()),
        "pair_concentration": pair_conc,
        "timeframe_concentration": timeframe_conc,
    }


def _concentration(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 1.0
    counts = frame[column].astype(str).value_counts()
    return float(counts.iloc[0] / max(counts.sum(), 1)) if not counts.empty else 1.0


def _row(accepted: bool, blocker: str) -> dict[str, object]:
    return {"accepted": accepted, "blocker": blocker, "acceptance_reason": "passed" if accepted else blocker}
