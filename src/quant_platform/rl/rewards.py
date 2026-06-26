from __future__ import annotations


def rl_reward(
    pnl: float,
    *,
    fees: float = 0.0,
    slippage: float = 0.0,
    drawdown: float = 0.0,
    overtraded: bool = False,
    stale_data: bool = False,
    invalid_action: bool = False,
) -> float:
    reward = float(pnl) - float(fees) - float(slippage) - abs(float(drawdown)) * 0.25
    if overtraded:
        reward -= 0.001
    if stale_data:
        reward -= 0.01
    if invalid_action:
        reward -= 0.05
    return float(reward)
