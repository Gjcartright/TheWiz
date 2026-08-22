"""Deterministic diagnostic oracles for Phase 00 run control."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class RunControlState:
    """Minimal state needed to verify run-control safety invariants."""

    progress: int = 0
    lease_held: bool = False
    terminal: bool = False
    crashed: bool = False
    recovered: bool = False
    authority_commits: int = 0


RUN_CONTROL_EVENTS = (
    "launch",
    "collect",
    "validate",
    "publish",
    "handoff",
    "finish",
    "crash",
    "recover",
)


def evaluate_cadence_slo(
    *,
    expected_slots: Mapping[str, datetime],
    completions: Sequence[Mapping[str, Any]],
    maximum_delay_seconds: float,
    minimum_expected_slots: int,
) -> dict[str, Any]:
    """Replay retained terminal receipts against an explicit slot schedule."""

    if maximum_delay_seconds < 0:
        raise ValueError("maximum_delay_seconds must be non-negative")
    if minimum_expected_slots <= 0:
        raise ValueError("minimum_expected_slots must be positive")
    normalized_expected = {
        str(slot): _as_utc(timestamp)
        for slot, timestamp in expected_slots.items()
    }
    eligible: dict[str, list[datetime]] = {}
    ineligible_records = 0
    unexplained_slots: set[str] = set()
    for row in completions:
        slot = str(row.get("intended_slot", ""))
        is_eligible = bool(
            row.get("trigger_provenance") == "launchd"
            and row.get("terminal_status") == "PASS"
            and row.get("intended_slot_credit") is True
        )
        if not is_eligible:
            ineligible_records += 1
            continue
        if slot not in normalized_expected:
            unexplained_slots.add(slot or "<blank>")
            continue
        completed_at = datetime.fromisoformat(str(row.get("completed_at_utc", "")))
        eligible.setdefault(slot, []).append(_as_utc(completed_at))

    missing_slots: list[str] = []
    duplicate_slots: list[str] = []
    delayed_slots: list[str] = []
    future_authorized_slots: list[str] = []
    on_time = 0
    delay_seconds: dict[str, float] = {}
    for slot, expected_at in sorted(normalized_expected.items()):
        observed = sorted(eligible.get(slot, []))
        if not observed:
            missing_slots.append(slot)
            continue
        if len(observed) != 1:
            duplicate_slots.append(slot)
        delay = (observed[0] - expected_at).total_seconds()
        delay_seconds[slot] = delay
        if delay < 0:
            future_authorized_slots.append(slot)
        elif delay > maximum_delay_seconds:
            delayed_slots.append(slot)
        else:
            on_time += 1

    expected_count = len(normalized_expected)
    blockers: list[str] = []
    if expected_count < minimum_expected_slots:
        blockers.append("cadence_window_too_short")
    if missing_slots:
        blockers.append("cadence_slots_missing")
    if duplicate_slots:
        blockers.append("cadence_slots_duplicated")
    if delayed_slots:
        blockers.append("cadence_slots_delayed")
    if future_authorized_slots:
        blockers.append("cadence_future_authorization")
    if unexplained_slots:
        blockers.append("cadence_unexplained_slots")
    return {
        "status": "PASS_CADENCE_SLO" if not blockers else "BLOCKED_CADENCE_SLO",
        "expected_slots": expected_count,
        "eligible_completions": sum(len(values) for values in eligible.values()),
        "on_time_slots": on_time,
        "availability": (on_time / expected_count) if expected_count else 0.0,
        "missing_slots": missing_slots,
        "duplicate_slots": duplicate_slots,
        "delayed_slots": delayed_slots,
        "future_authorized_slots": future_authorized_slots,
        "unexplained_slots": sorted(unexplained_slots),
        "ineligible_records": ineligible_records,
        "delay_seconds": delay_seconds,
        "blockers": blockers,
        "promotion_authority": False,
        "order_authority": False,
    }


def transition_run_control(state: RunControlState, event: str) -> RunControlState:
    """Apply one allowed state transition or reject the interleaving."""

    if event not in RUN_CONTROL_EVENTS:
        raise ValueError("unknown_run_control_event")
    if event == "recover":
        if not state.terminal or not state.crashed or state.recovered:
            raise ValueError("recovery_transition_invalid")
        return RunControlState(**{**asdict(state), "recovered": True})
    if state.terminal:
        raise ValueError("terminal_state_is_closed")
    if event == "launch":
        if state.progress != 0 or state.lease_held:
            raise ValueError("launch_transition_invalid")
        return RunControlState(progress=1, lease_held=True)
    if event == "collect":
        return _advance(state, expected=1, target=2)
    if event == "validate":
        return _advance(state, expected=2, target=3)
    if event == "publish":
        advanced = _advance(state, expected=3, target=4)
        return RunControlState(**{**asdict(advanced), "authority_commits": 1})
    if event == "handoff":
        return _advance(state, expected=4, target=5)
    if event in {"finish", "crash"}:
        if not state.lease_held:
            raise ValueError("terminal_transition_requires_lease")
        return RunControlState(
            **{
                **asdict(state),
                "lease_held": False,
                "terminal": True,
                "crashed": event == "crash",
            }
        )
    raise AssertionError("unreachable run-control event")


def exhaustive_run_control_model(*, maximum_depth: int = 8) -> dict[str, Any]:
    """Explore all reachable bounded interleavings and verify safety properties."""

    if maximum_depth <= 0:
        raise ValueError("maximum_depth must be positive")
    initial = RunControlState()
    queue = deque([(initial, 0)])
    visited = {initial}
    transitions = 0
    rejected = 0
    violations: set[str] = set()
    while queue:
        state, depth = queue.popleft()
        violations.update(_state_violations(state))
        if not state.terminal and state.lease_held:
            try:
                recovered = transition_run_control(
                    transition_run_control(state, "crash"),
                    "recover",
                )
            except ValueError:
                violations.add("nonterminal_state_not_crash_recoverable")
            else:
                violations.update(_state_violations(recovered))
        elif not state.terminal and state.progress != 0:
            violations.add("active_nonterminal_state_lost_lease")
        if depth >= maximum_depth:
            continue
        for event in RUN_CONTROL_EVENTS:
            try:
                candidate = transition_run_control(state, event)
            except ValueError:
                rejected += 1
                continue
            transitions += 1
            if candidate not in visited:
                visited.add(candidate)
                queue.append((candidate, depth + 1))
    return {
        "status": "PASS_BOUNDED_MODEL" if not violations else "BLOCKED_BOUNDED_MODEL",
        "maximum_depth": maximum_depth,
        "reachable_states": len(visited),
        "valid_transitions": transitions,
        "rejected_interleavings": rejected,
        "violations": sorted(violations),
        "promotion_authority": False,
        "order_authority": False,
    }


def _advance(
    state: RunControlState,
    *,
    expected: int,
    target: int,
) -> RunControlState:
    if state.progress != expected or not state.lease_held:
        raise ValueError("run_control_transition_out_of_order")
    return RunControlState(**{**asdict(state), "progress": target})


def _state_violations(state: RunControlState) -> set[str]:
    violations: set[str] = set()
    if state.authority_commits not in {0, 1}:
        violations.add("multiple_authority_commits")
    if (state.progress >= 4) != (state.authority_commits == 1):
        violations.add("publication_authority_progress_mismatch")
    if state.progress >= 5 and state.authority_commits != 1:
        violations.add("handoff_without_publication_authority")
    if state.terminal and state.lease_held:
        violations.add("terminal_state_retains_lease")
    if state.recovered and (not state.terminal or not state.crashed):
        violations.add("recovery_without_terminal_crash")
    return violations


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("cadence timestamps must be timezone-aware")
    return value.astimezone(UTC)
