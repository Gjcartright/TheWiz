"""A real timer must escape ordinary producer error/retry handling."""
import subprocess
import time

import pytest

from quant_platform.orchestration.current_wizard_hyperliquid_daily_runner import (
    _in_process_stage_timeout,
)


def test_deadline_is_not_swallowed_as_an_ordinary_pair_failure():
    actions = []
    with pytest.raises(subprocess.TimeoutExpired), _in_process_stage_timeout(["fixture_stage"], 0.02):
        try:
            time.sleep(0.2)
        except Exception:  # noqa: BLE001 - reproduce the producer's ordinary retry handler
            actions.append("retry_or_next_pair")
        actions.append("late_work")
    assert actions == []


def test_caught_base_exception_still_cannot_return_stage_success():
    suppressed = []
    with pytest.raises(subprocess.TimeoutExpired), _in_process_stage_timeout(["fixture_stage"], 0.02):
        try:
            time.sleep(0.2)
        except BaseException:  # noqa: BLE001 - deliberately simulate an inner cancellation suppressor
            suppressed.append(True)
    assert suppressed == [True]


def test_deadline_restores_handler_and_timer_after_unwind():
    import signal
    before = signal.getsignal(signal.SIGALRM)
    with pytest.raises(subprocess.TimeoutExpired), _in_process_stage_timeout(["fixture_stage"], 0.02):
        time.sleep(0.2)
    assert signal.getsignal(signal.SIGALRM) is before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    with _in_process_stage_timeout(["next_fixture_stage"], 0.2):
        pass


def test_ordinary_producer_failure_is_not_reclassified_as_timeout():
    with pytest.raises(ValueError, match="ordinary_failure"), _in_process_stage_timeout(["fixture_stage"], 0.2):
        raise ValueError("ordinary_failure")
