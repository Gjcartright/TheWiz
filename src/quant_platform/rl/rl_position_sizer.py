from __future__ import annotations


def capped_position_size(score: float, *, max_size: float = 1.0) -> float:
    if score < 0.55:
        return 0.0
    if score < 0.70:
        return 0.25 * max_size
    if score < 0.80:
        return 0.50 * max_size
    if score < 0.90:
        return 0.75 * max_size
    return 1.0 * max_size
