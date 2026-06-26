from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RlAction:
    id: int
    name: str
    description: str


ACTIONS: dict[int, RlAction] = {
    0: RlAction(0, "hold", "keep current position"),
    1: RlAction(1, "enter_long_x_short_y", "enter long X / short Y"),
    2: RlAction(2, "enter_short_x_long_y", "enter short X / long Y"),
    3: RlAction(3, "close", "close current position"),
    4: RlAction(4, "reduce_size", "reduce current position size"),
    5: RlAction(5, "add_size", "add to current position size"),
}


def action_name(action: int) -> str:
    return ACTIONS.get(int(action), RlAction(int(action), "unknown", "unknown action")).name
