from __future__ import annotations

import json
from datetime import UTC, datetime

from quant_platform.orchestration.corrective_wizard_api_credit_receipt import (
    capture_wizard_api_credit_receipt,
    publish_wizard_api_credit_receipt,
)


def test_credit_receipt_is_redacted_immutable_and_sufficient(tmp_path) -> None:
    result = capture_wizard_api_credit_receipt(
        root=tmp_path,
        now=datetime(2026, 8, 14, 15, 0, tzinfo=UTC),
        api_key="never-store-this-key",
        fetcher=lambda **_: {"credits_used": 700, "credit_limit": 1000},
        required_credits=16,
        protected_reserve=100,
    )

    encoded = result.paths["immutable_receipt"].read_text(encoding="utf-8")
    assert "never-store-this-key" not in encoded
    assert result.summary["status"] == "PASS_AUTHENTICATED_CREDIT_PREFLIGHT"
    assert result.summary["credits_remaining"] == 300
    assert result.summary["response_body_stored"] is False
    assert len(result.summary["response_sha256"]) == 64
    assert result.summary["testnet_order_authority"] is False


def test_credit_receipt_blocks_unknown_usage_without_storing_response(tmp_path) -> None:
    raw = {"unexpected": {"message": "ok"}}
    result = publish_wizard_api_credit_receipt(
        root=tmp_path,
        now=datetime(2026, 8, 14, 15, 1, tzinfo=UTC),
        response=raw,
        api_key_present=True,
        api_key_source="environment",
    )

    receipt = json.loads(result.paths["immutable_receipt"].read_text(encoding="utf-8"))
    assert receipt["status"] == "BLOCKED_CREDIT_PREFLIGHT"
    assert "credit_usage_unknown" in receipt["blockers"]
    assert "unexpected" not in receipt
    assert receipt["response_body_stored"] is False
