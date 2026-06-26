from __future__ import annotations

import numpy as np
import pandas as pd


def rl_acceptance_report(evaluation: pd.DataFrame) -> pd.DataFrame:
    if evaluation.empty:
        return pd.DataFrame([_row(False, "missing_rl_evaluation")])
    raw = evaluation[evaluation["variant"] == "non_rl_baseline"]
    rl = evaluation[evaluation["variant"] == "safe_rl_policy"]
    if raw.empty or rl.empty:
        return pd.DataFrame([_row(False, "missing_baseline_or_rl_variant")])
    raw_row = raw.iloc[0]
    rl_row = rl.iloc[0]
    accepted = bool(
        rl_row["profit_factor"] > raw_row["profit_factor"]
        and rl_row["max_drawdown"] <= raw_row["max_drawdown"]
        and rl_row["trades"] >= max(20, raw_row["trades"] * 0.25)
        and rl_row["take_rate"] >= 0.05
        and rl_row["pair_concentration"] <= 0.65
        and rl_row["timeframe_concentration"] <= 0.65
    )
    blocker = "" if accepted else "rl_acceptance_gates_not_met"
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
                "acceptance_reason": "passed" if accepted else blocker,
            }
        ]
    )


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
