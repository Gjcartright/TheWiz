from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_wizard_proof_launcher import (
    _launcher_exit_code,
    run_wizard_proof_launcher,
)
from quant_platform.orchestration.corrective_wizard_unattended_preflight import (
    build_wizard_unattended_external_preflight,
)

NOW = datetime(2026, 8, 15, 0, 5, tzinfo=UTC)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _zero_authority() -> dict[str, bool]:
    return {
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _manifest(root: Path, *, reset_at: datetime = NOW - timedelta(minutes=5)) -> None:
    _write_json(
        root / "reports" / "active" / "corrective_wizard_next_capture_manifest.json",
        {
            "schema_version": "thewiz.corrective_wizard_next_capture_manifest.v1",
            "status": "PASS",
            "manifest_enforced": True,
            "runtime_credit_preflight_required": True,
            "manifest_id": "wizardcapture_0123456789abcdef0123",
            "pending_calls": 8,
            "planned_credits": 16,
            "next_external_attempt_eligible_at": reset_at.isoformat(),
            **_zero_authority(),
        },
    )


def _bound_result(
    root: Path,
    *,
    directory: str,
    filename: str,
    summary: dict[str, object],
) -> CommandResult:
    immutable = root / "data" / "research" / directory / filename
    _write_json(immutable, summary)
    active_summary = {
        **summary,
        "immutable_receipt_path": str(immutable.relative_to(root)),
        "immutable_receipt_sha256": sha256(immutable.read_bytes()).hexdigest(),
    }
    return CommandResult(paths={"immutable_receipt": immutable}, summary=active_summary)


def _reset_result(*, root: Path, now: datetime, status: str = "PASS_RESET_AUTOMATION_READY"):
    return CommandResult(
        paths={},
        summary={
            "status": status,
            "capture_window_status": "DUE_AFTER_RESET",
            "manifest_id": "wizardcapture_0123456789abcdef0123",
            "pending_calls": 8,
            "planned_credits": 16,
            "receipt_id": "wizardresetreadiness_test",
            **_zero_authority(),
        },
    )


def _canonical_result(*, root: Path, now: datetime) -> CommandResult:
    return _bound_result(
        root,
        directory="canonical_program_status",
        filename="canonicalstatus_test.json",
        summary={
            "status": "PASS_CANONICAL_CURRENT",
            "generated_at_utc": now.isoformat(),
            "receipt_id": "canonicalstatus_test",
            **_zero_authority(),
        },
    )


def _credit_result(
    *,
    root: Path,
    now: datetime,
    daily_limit: int,
    protected_reserve: int,
    required_credits: int,
) -> CommandResult:
    return _bound_result(
        root,
        directory="wizard_api_credit_receipts",
        filename="wizardcredit_test.json",
        summary={
            "status": "PASS_AUTHENTICATED_CREDIT_PREFLIGHT",
            "checked_at_utc": now.isoformat(),
            "receipt_id": "wizardcredit_test",
            "credit_limit": daily_limit,
            "protected_reserve": protected_reserve,
            "required_credits": required_credits,
            "credits_used": 0,
            "credits_remaining": daily_limit,
            "sufficient_after_reserve": True,
            "response_body_stored": False,
            "secret_value_stored": False,
            **_zero_authority(),
        },
    )


def test_unattended_preflight_binds_reset_canonical_and_1000_credit_policy(tmp_path: Path):
    _manifest(tmp_path)

    result = build_wizard_unattended_external_preflight(
        root=tmp_path,
        now=NOW,
        reset_readiness_builder=_reset_result,
        reset_readiness_validator=lambda **_: {"status": "PASS", "blockers": []},
        canonical_status_builder=_canonical_result,
        credit_receipt_capturer=_credit_result,
    )

    assert result.summary["status"] == "PASS_UNATTENDED_EXTERNAL_PREFLIGHT"
    assert result.summary["planned_credits"] == 16
    assert result.summary["daily_credit_limit"] == 1000
    assert result.summary["protected_credit_reserve"] == 100
    assert result.summary["credit_preflight_status"] == ("PASS_AUTHENTICATED_CREDIT_PREFLIGHT")
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    assert Path(result.paths["immutable_receipt"]).is_file()


def test_unattended_preflight_makes_no_credit_call_before_reset(tmp_path: Path):
    _manifest(tmp_path, reset_at=NOW + timedelta(hours=1))
    credit_calls: list[dict[str, object]] = []

    result = build_wizard_unattended_external_preflight(
        root=tmp_path,
        now=NOW,
        reset_readiness_builder=lambda **_: (_ for _ in ()).throw(
            AssertionError("reset readiness must not run before the window")
        ),
        reset_readiness_validator=lambda **_: {"status": "PASS"},
        canonical_status_builder=lambda **_: (_ for _ in ()).throw(
            AssertionError("canonical refresh must not run before the window")
        ),
        credit_receipt_capturer=lambda **kwargs: credit_calls.append(kwargs),
    )

    assert result.summary["status"] == "BLOCKED_UNATTENDED_EXTERNAL_PREFLIGHT"
    assert result.summary["blockers"] == ["unattended_frozen_manifest_not_due_or_invalid"]
    assert credit_calls == []


def test_unattended_preflight_does_not_call_credit_endpoint_when_reset_is_blocked(
    tmp_path: Path,
):
    _manifest(tmp_path)
    credit_calls: list[dict[str, object]] = []

    result = build_wizard_unattended_external_preflight(
        root=tmp_path,
        now=NOW,
        reset_readiness_builder=lambda **kwargs: _reset_result(
            **kwargs, status="BLOCKED_RESET_AUTOMATION"
        ),
        reset_readiness_validator=lambda **_: {"status": "BLOCKED"},
        canonical_status_builder=_canonical_result,
        credit_receipt_capturer=lambda **kwargs: credit_calls.append(kwargs),
    )

    assert result.summary["status"] == "BLOCKED_UNATTENDED_EXTERNAL_PREFLIGHT"
    assert "unattended_reset_readiness_not_bound" in result.summary["blockers"]
    assert credit_calls == []


def _launcher_preflight(root: Path, *, passed: bool) -> CommandResult:
    immutable = root / "data" / "research" / "preflight.json"
    immutable.parent.mkdir(parents=True, exist_ok=True)
    immutable.write_text("{}\n", encoding="utf-8")
    return CommandResult(
        paths={"immutable_receipt": immutable},
        summary={
            "status": (
                "PASS_UNATTENDED_EXTERNAL_PREFLIGHT"
                if passed
                else "BLOCKED_UNATTENDED_EXTERNAL_PREFLIGHT"
            ),
            "receipt_id": "wizardpreflight_test",
            "blockers": [] if passed else ["credit_not_ready"],
        },
    )


def _python(root: Path) -> Path:
    path = root / ".venv312" / "bin" / "python"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_launcher_blocks_unattended_scheduler_when_preflight_fails(tmp_path: Path):
    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=_python(tmp_path),
        unattended_preflight_builder=lambda **_: _launcher_preflight(tmp_path, passed=False),
        runner=lambda *_, **__: (_ for _ in ()).throw(AssertionError("scheduler called")),
    )

    assert result["launcher_status"] == "BLOCKED_UNATTENDED_EXTERNAL_PREFLIGHT"
    assert result["heavy_scheduler_invoked"] is False
    assert result["unattended_external_preflight_required"] is True
    assert result["unattended_external_preflight_status"] == (
        "BLOCKED_UNATTENDED_EXTERNAL_PREFLIGHT"
    )
    assert _launcher_exit_code(result) == 2


def test_launcher_invokes_scheduler_only_after_unattended_preflight_passes(tmp_path: Path):
    calls: list[list[str]] = []

    def runner(command, **_):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    result = run_wizard_proof_launcher(
        root=tmp_path,
        now=NOW,
        execute=True,
        python=_python(tmp_path),
        unattended_preflight_builder=lambda **_: _launcher_preflight(tmp_path, passed=True),
        runner=runner,
    )

    assert calls
    assert result["heavy_scheduler_invoked"] is True
    assert result["unattended_external_preflight_status"] == ("PASS_UNATTENDED_EXTERNAL_PREFLIGHT")
