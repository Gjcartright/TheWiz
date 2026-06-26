from __future__ import annotations


def execution_action_allowed(action: str, *, evidence_accepted: bool, model_accepted: bool, rl_accepted: bool) -> tuple[bool, str]:
    if not evidence_accepted:
        return False, "missing_local_acceptance_evidence"
    if not model_accepted:
        return False, "model_gate_not_accepted"
    if not rl_accepted:
        return False, "rl_acceptance_not_passed"
    if action == "live_submit":
        return False, "rl_live_submission_blocked_in_v1"
    return True, ""
