from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.rl.actions import action_name
from quant_platform.rl.features import build_rl_feature_frame
from quant_platform.rl.rewards import rl_reward


@dataclass(frozen=True)
class SimpleDiscrete:
    n: int

    def contains(self, value: object) -> bool:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return False
        return 0 <= numeric < self.n


@dataclass(frozen=True)
class SimpleBox:
    shape: tuple[int, ...]
    dtype: str = "float32"


class PairTradingEnv:
    """Dependency-light Gymnasium-compatible pair-trading environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        max_position: float = 1.0,
        fee_bps: float = 5.0,
        slippage_bps: float = 4.0,
        stale: bool = False,
    ) -> None:
        self.source = frame.reset_index(drop=True).copy()
        self.features = build_rl_feature_frame(self.source)
        self.max_position = float(max_position)
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.stale = bool(stale)
        self.action_space = SimpleDiscrete(6)
        self.observation_space = SimpleBox((self.features.shape[1] + 3,))
        self.trade_log: list[dict[str, object]] = []
        self.blocked_actions: list[dict[str, object]] = []
        self.reset()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.index = 0
        self.position = 0.0
        self.position_direction = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.time_in_trade = 0
        self.trade_log = []
        self.blocked_actions = []
        return self._observation(), {"blocked": False}

    def step(self, action: int):
        invalid, reason = self._invalid_action(action)
        previous_equity = self.equity
        pnl = self._step_pnl()
        cost = self._action_cost(action, invalid)
        if not invalid:
            self._apply_action(action)
        self.equity *= 1.0 + pnl - cost
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-12)
        reward = rl_reward(
            self.equity - previous_equity,
            fees=cost * 0.5,
            slippage=cost * 0.5,
            drawdown=drawdown,
            overtraded=action in {1, 2, 3, 4, 5},
            stale_data=self.stale,
            invalid_action=invalid,
        )
        if invalid:
            self.blocked_actions.append(
                {"step": self.index, "action": action_name(action), "reason": reason, "position": self.position}
            )
        else:
            self.trade_log.append(
                {"step": self.index, "action": action_name(action), "position": self.position, "equity": self.equity}
            )
        if self.position:
            self.time_in_trade += 1
        else:
            self.time_in_trade = 0
        self.index += 1
        terminated = self.index >= len(self.features) - 1
        truncated = False
        return self._observation(), reward, terminated, truncated, {"blocked": invalid, "blocker": reason, "equity": self.equity}

    def _observation(self) -> np.ndarray:
        row_index = min(self.index, max(len(self.features) - 1, 0))
        values = self.features.iloc[row_index].to_numpy(dtype="float32") if len(self.features) else np.zeros(self.features.shape[1])
        state = np.array([self.position, self.position_direction, float(self.time_in_trade)], dtype="float32")
        return np.concatenate([values, state]).astype("float32")

    def _step_pnl(self) -> float:
        if self.index == 0 or not self.position:
            return 0.0
        spread = pd.to_numeric(self.source.get("spread", pd.Series(0.0, index=self.source.index)), errors="coerce").fillna(0.0)
        return float(self.position * spread.diff().fillna(0.0).iloc[self.index] * 0.01)

    def _action_cost(self, action: int, invalid: bool) -> float:
        if invalid or action == 0:
            return 0.0
        return (self.fee_bps + self.slippage_bps) / 10_000.0

    def _invalid_action(self, action: int) -> tuple[bool, str]:
        if not self.action_space.contains(action):
            return True, "unknown_action"
        if self.stale and action in {1, 2, 4, 5}:
            return True, "stale_data_blocks_position_action"
        if action in {1, 2} and self.position != 0:
            return True, "entry_while_position_open"
        if action in {3, 4, 5} and self.position == 0:
            return True, "position_action_without_open_position"
        if action == 5 and abs(self.position) >= self.max_position:
            return True, "max_position_reached"
        return False, ""

    def _apply_action(self, action: int) -> None:
        if action == 1:
            self.position = self.max_position
            self.position_direction = 1.0
        elif action == 2:
            self.position = -self.max_position
            self.position_direction = -1.0
        elif action == 3:
            self.position = 0.0
            self.position_direction = 0.0
        elif action == 4:
            self.position *= 0.5
            if abs(self.position) < 1e-12:
                self.position_direction = 0.0
        elif action == 5:
            direction = self.position_direction or 1.0
            self.position = direction * min(self.max_position, abs(self.position) + 0.25)
