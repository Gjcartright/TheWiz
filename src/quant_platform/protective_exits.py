"""Local, zero-authority protective exits for deterministic OHLC simulations.

This module is deliberately not connected to any execution, strategy, or Wizard
path.  It models only what can be established from an OHLC bar: the open occurs
before the intrabar high/low, while the relative order of the high and low is
unknown.  Live stop-order slippage and venue-specific behavior remain outside
this contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum


class PositionSide(str, Enum):
    """Direction of the protected position."""

    LONG = "long"
    SHORT = "short"


class ExitReason(str, Enum):
    """Typed reason for a completed protective exit."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MAX_HOLD = "max_hold"


class ExitTriggerBasis(str, Enum):
    """Observable bar component used to determine an exit fill."""

    OPEN_GAP = "open_gap"
    INTRABAR = "intrabar"
    BAR_CLOSE = "bar_close"


class SameBarCollisionPolicy(str, Enum):
    """Policy for a bar whose range touches both protective levels."""

    STOP_FIRST = "stop_first"
    TAKE_PROFIT_FIRST = "take_profit_first"
    FAIL_CLOSED = "fail_closed"


class AmbiguousIntrabarPathError(RuntimeError):
    """Raised when fail-closed policy cannot order two intrabar triggers."""


class EntryNotAllowedError(RuntimeError):
    """Raised when an entry would violate engine sequencing or cooldown."""


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _bar_index(value: object, *, name: str = "bar_index") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class OHLCBar:
    """A validated positive-price OHLC bar."""

    open: float
    high: float
    low: float
    close: float

    def __post_init__(self) -> None:
        for name in ("open", "high", "low", "close"):
            object.__setattr__(self, name, _positive_finite(getattr(self, name), name=name))
        if self.low > self.high:
            raise ValueError("low must be less than or equal to high")
        if not self.low <= self.open <= self.high:
            raise ValueError("open must lie within the low/high range")
        if not self.low <= self.close <= self.high:
            raise ValueError("close must lie within the low/high range")


@dataclass(frozen=True)
class ProtectiveExitPolicy:
    """Fractional protective distances and deterministic timing controls.

    Both distances are fractions of entry price, so ``0.05`` means five
    percent.  Distances must be below one to keep every supported side's
    protective price strictly positive.

    ``max_hold_bars`` counts processed bars, including the entry bar when that
    bar is submitted to :meth:`ProtectiveExitEngine.process_bar`. Protective
    price triggers take precedence over a max-hold exit on the same bar.

    ``cooldown_bars`` is the number of complete bars after an exit during which
    a new entry is blocked.  Re-entry on the exit bar is always prohibited.
    """

    stop_loss_fraction: float
    take_profit_fraction: float
    max_hold_bars: int | None = None
    cooldown_bars: int = 0
    collision_policy: SameBarCollisionPolicy = SameBarCollisionPolicy.STOP_FIRST

    def __post_init__(self) -> None:
        for name in ("stop_loss_fraction", "take_profit_fraction"):
            value = _positive_finite(getattr(self, name), name=name)
            if value >= 1.0:
                raise ValueError(f"{name} must be less than one")
            object.__setattr__(self, name, value)

        if self.max_hold_bars is not None and (
            isinstance(self.max_hold_bars, bool)
            or not isinstance(self.max_hold_bars, int)
            or self.max_hold_bars <= 0
        ):
            raise ValueError("max_hold_bars must be a positive integer or None")
        if (
            isinstance(self.cooldown_bars, bool)
            or not isinstance(self.cooldown_bars, int)
            or self.cooldown_bars < 0
        ):
            raise ValueError("cooldown_bars must be a non-negative integer")
        try:
            normalized_policy = SameBarCollisionPolicy(self.collision_policy)
        except ValueError as exc:
            raise ValueError("collision_policy is unsupported") from exc
        object.__setattr__(self, "collision_policy", normalized_policy)


@dataclass(frozen=True)
class ProtectiveLevels:
    """Absolute stop-loss and take-profit prices for a position."""

    stop_loss: float
    take_profit: float

    def __post_init__(self) -> None:
        for name in ("stop_loss", "take_profit"):
            object.__setattr__(self, name, _positive_finite(getattr(self, name), name=name))


@dataclass(frozen=True)
class OpenPosition:
    """Immutable engine state for one open position."""

    side: PositionSide
    entry_price: float
    entry_bar_index: int
    levels: ProtectiveLevels
    bars_held: int = 0


@dataclass(frozen=True)
class ProtectiveExit:
    """A completed exit determined by the local OHLC policy."""

    reason: ExitReason
    trigger_basis: ExitTriggerBasis
    side: PositionSide
    bar_index: int
    entry_bar_index: int
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    fill_price: float
    bars_held: int

    @property
    def return_fraction(self) -> float:
        """Unsigned-notional return before costs for this directional trade."""

        direction = 1.0 if self.side is PositionSide.LONG else -1.0
        return direction * (self.fill_price / self.entry_price - 1.0)


@dataclass(frozen=True)
class ProtectiveExitState:
    """Serializable immutable snapshot used for deterministic replay checks."""

    position: OpenPosition | None
    last_processed_bar_index: int | None
    last_exit_bar_index: int | None


def protective_levels(
    *,
    side: PositionSide | str,
    entry_price: float,
    policy: ProtectiveExitPolicy,
) -> ProtectiveLevels:
    """Convert entry-relative distances to finite positive absolute prices.

    Reject overflow, underflow, or rounded-away distances. Rejecting an
    unsupported floating-point domain is preferable to changing the policy.
    """

    if not isinstance(policy, ProtectiveExitPolicy):
        raise TypeError("policy must be a ProtectiveExitPolicy")

    try:
        normalized_side = PositionSide(side)
    except ValueError as exc:
        raise ValueError("side must be 'long' or 'short'") from exc
    normalized_entry = _positive_finite(entry_price, name="entry_price")
    if normalized_side is PositionSide.LONG:
        stop_loss = normalized_entry * (1.0 - policy.stop_loss_fraction)
        take_profit = normalized_entry * (1.0 + policy.take_profit_fraction)
    else:
        stop_loss = normalized_entry * (1.0 + policy.stop_loss_fraction)
        take_profit = normalized_entry * (1.0 - policy.take_profit_fraction)
    levels = ProtectiveLevels(stop_loss=stop_loss, take_profit=take_profit)
    if normalized_side is PositionSide.LONG:
        ordered = levels.stop_loss < normalized_entry < levels.take_profit
    else:
        ordered = levels.take_profit < normalized_entry < levels.stop_loss
    if not ordered:
        raise ValueError("protective distances are not representable at the entry price")
    return levels


class ProtectiveExitEngine:
    """Single-position, deterministic OHLC protective-exit state machine."""

    def __init__(self, policy: ProtectiveExitPolicy) -> None:
        if not isinstance(policy, ProtectiveExitPolicy):
            raise TypeError("policy must be a ProtectiveExitPolicy")
        self.policy = policy
        self._position: OpenPosition | None = None
        self._last_processed_bar_index: int | None = None
        self._last_exit_bar_index: int | None = None

    @property
    def state(self) -> ProtectiveExitState:
        """Return an immutable snapshot of all path-dependent state."""

        return ProtectiveExitState(
            position=self._position,
            last_processed_bar_index=self._last_processed_bar_index,
            last_exit_bar_index=self._last_exit_bar_index,
        )

    def can_enter(self, *, bar_index: int) -> bool:
        """Return whether an entry may be placed before ``bar_index`` is processed."""

        index = _bar_index(bar_index)
        if self._position is not None:
            return False
        if self._last_processed_bar_index is not None and index <= self._last_processed_bar_index:
            return False
        if self._last_exit_bar_index is None:
            return True
        return index > self._last_exit_bar_index + self.policy.cooldown_bars

    def enter(
        self,
        *,
        side: PositionSide | str,
        entry_price: float,
        bar_index: int,
    ) -> OpenPosition:
        """Open one position before its first bar is processed.

        An entry is rejected while another position is open, on an already
        processed bar, or during the configured post-exit cooldown.
        """

        index = _bar_index(bar_index)
        if self._position is not None:
            raise EntryNotAllowedError("a position is already open")
        if self._last_processed_bar_index is not None and index <= self._last_processed_bar_index:
            raise EntryNotAllowedError("entry bar must be later than the last processed bar")
        if (
            self._last_exit_bar_index is not None
            and index <= self._last_exit_bar_index + self.policy.cooldown_bars
        ):
            raise EntryNotAllowedError("entry is blocked by same-bar protection or cooldown")

        try:
            normalized_side = PositionSide(side)
        except ValueError as exc:
            raise ValueError("side must be 'long' or 'short'") from exc
        normalized_entry = _positive_finite(entry_price, name="entry_price")
        position = OpenPosition(
            side=normalized_side,
            entry_price=normalized_entry,
            entry_bar_index=index,
            levels=protective_levels(
                side=normalized_side,
                entry_price=normalized_entry,
                policy=self.policy,
            ),
        )
        self._position = position
        return position

    def process_bar(self, *, bar_index: int, bar: OHLCBar) -> ProtectiveExit | None:
        """Advance the engine by one bar and return an exit when triggered.

        State commits only after the bar has an unambiguous outcome. Under
        ``FAIL_CLOSED``, an ambiguous bar raises and leaves the snapshot exactly
        as it was before this call.
        """

        index = _bar_index(bar_index)
        if not isinstance(bar, OHLCBar):
            raise TypeError("bar must be an OHLCBar")
        if self._last_processed_bar_index is not None and index <= self._last_processed_bar_index:
            raise ValueError("bar_index must be strictly increasing")
        if self._position is not None and index < self._position.entry_bar_index:
            raise ValueError("cannot process a bar before the entry bar")

        exit_event = None
        if self._position is not None:
            bars_held = self._position.bars_held + 1
            exit_event = self._evaluate_exit(
                position=self._position,
                bar=bar,
                bar_index=index,
                bars_held=bars_held,
            )

        self._last_processed_bar_index = index
        if self._position is None:
            return None
        if exit_event is None:
            self._position = replace(self._position, bars_held=self._position.bars_held + 1)
            return None

        self._position = None
        self._last_exit_bar_index = index
        return exit_event

    def _evaluate_exit(
        self,
        *,
        position: OpenPosition,
        bar: OHLCBar,
        bar_index: int,
        bars_held: int,
    ) -> ProtectiveExit | None:
        gap_reason = _gap_reason(position, bar.open)
        if gap_reason is not None:
            return _exit_event(
                position,
                reason=gap_reason,
                basis=ExitTriggerBasis.OPEN_GAP,
                fill_price=bar.open,
                bar_index=bar_index,
                bars_held=bars_held,
            )

        stop_hit, profit_hit = _intrabar_hits(position, bar)
        if stop_hit and profit_hit:
            if self.policy.collision_policy is SameBarCollisionPolicy.FAIL_CLOSED:
                raise AmbiguousIntrabarPathError(
                    f"bar {bar_index} touches both stop-loss and take-profit levels"
                )
            if self.policy.collision_policy is SameBarCollisionPolicy.STOP_FIRST:
                profit_hit = False
            else:
                stop_hit = False

        if stop_hit:
            return _exit_event(
                position,
                reason=ExitReason.STOP_LOSS,
                basis=ExitTriggerBasis.INTRABAR,
                fill_price=position.levels.stop_loss,
                bar_index=bar_index,
                bars_held=bars_held,
            )
        if profit_hit:
            return _exit_event(
                position,
                reason=ExitReason.TAKE_PROFIT,
                basis=ExitTriggerBasis.INTRABAR,
                fill_price=position.levels.take_profit,
                bar_index=bar_index,
                bars_held=bars_held,
            )
        if self.policy.max_hold_bars is not None and bars_held >= self.policy.max_hold_bars:
            return _exit_event(
                position,
                reason=ExitReason.MAX_HOLD,
                basis=ExitTriggerBasis.BAR_CLOSE,
                fill_price=bar.close,
                bar_index=bar_index,
                bars_held=bars_held,
            )
        return None


def _gap_reason(position: OpenPosition, open_price: float) -> ExitReason | None:
    if position.side is PositionSide.LONG:
        if open_price <= position.levels.stop_loss:
            return ExitReason.STOP_LOSS
        if open_price >= position.levels.take_profit:
            return ExitReason.TAKE_PROFIT
    else:
        if open_price >= position.levels.stop_loss:
            return ExitReason.STOP_LOSS
        if open_price <= position.levels.take_profit:
            return ExitReason.TAKE_PROFIT
    return None


def _intrabar_hits(position: OpenPosition, bar: OHLCBar) -> tuple[bool, bool]:
    if position.side is PositionSide.LONG:
        return bar.low <= position.levels.stop_loss, bar.high >= position.levels.take_profit
    return bar.high >= position.levels.stop_loss, bar.low <= position.levels.take_profit


def _exit_event(
    position: OpenPosition,
    *,
    reason: ExitReason,
    basis: ExitTriggerBasis,
    fill_price: float,
    bar_index: int,
    bars_held: int,
) -> ProtectiveExit:
    return ProtectiveExit(
        reason=reason,
        trigger_basis=basis,
        side=position.side,
        bar_index=bar_index,
        entry_bar_index=position.entry_bar_index,
        entry_price=position.entry_price,
        stop_loss_price=position.levels.stop_loss,
        take_profit_price=position.levels.take_profit,
        fill_price=fill_price,
        bars_held=bars_held,
    )
