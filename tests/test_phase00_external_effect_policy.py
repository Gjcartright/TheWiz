from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from quant_platform.crypto_wizards_catalog import BASE_URL, ENDPOINTS
from quant_platform.orchestration.corrective_external_effect_policy import (
    EXTERNAL_EFFECT_POLICY_PATH,
    ExternalEffectPolicyError,
    load_phase00_external_effect_policy,
)
from quant_platform.orchestration.corrective_hyperliquid_network import (
    hyperliquid_info_contract_operations,
)

ROOT = Path(__file__).resolve().parents[1]
AS_OF = datetime(2026, 8, 22, 12, tzinfo=UTC)


def _copy_policy(root: Path) -> Path:
    target = root / EXTERNAL_EFFECT_POLICY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((ROOT / EXTERNAL_EFFECT_POLICY_PATH).read_bytes())
    return target


def test_current_policy_is_hash_bound_and_matches_wizard_endpoint_catalog() -> None:
    policy = load_phase00_external_effect_policy(ROOT, as_of=AS_OF)
    wizard = policy.provider("crypto_wizards")
    observed = {
        (row.method, row.target, row.credit_units)
        for row in wizard.endpoint_pricing
    }
    expected = {
        (endpoint.method, f"{BASE_URL}{endpoint.path}", endpoint.credits)
        for endpoint in ENDPOINTS
        if endpoint.method in {"GET", "POST"}
    }

    assert observed == expected
    assert policy.policy_sha256 == sha256(policy.path.read_bytes()).hexdigest()
    assert policy.permit_policy_version == (
        f"{policy.schema_version}:sha256:{policy.policy_sha256}"
    )
    assert wizard.max_total_requests_per_run == 2048
    assert wizard.max_total_credits_per_run == 1000
    assert wizard.pricing_provenance.evidence_status == (
        "LOCAL_CONTRACT_REQUIRES_PERIODIC_PROVIDER_REVIEW"
    )
    apify = policy.provider("apify")
    assert apify.endpoint_pricing[0].credit_units is None
    assert "RUNTIME_CEILING_REQUIRED" in (
        apify.pricing_provenance.evidence_status
    )
    hyperliquid = policy.provider("hyperliquid_public")
    assert hyperliquid.account_scope_id == "hyperliquid:public_research"
    assert hyperliquid.allowed_targets == frozenset(
        {
            "https://api.hyperliquid-testnet.xyz/info",
            "https://api.hyperliquid.xyz/info",
        }
    )
    assert hyperliquid.allowed_credential_keys == frozenset()
    assert hyperliquid.max_total_requests_per_run == 4096
    assert hyperliquid.max_total_credits_per_run == 0
    public_operations = {row.operation for row in hyperliquid.endpoint_pricing}
    assert public_operations == {
        "HYPERLIQUID_TESTNET_META",
        "HYPERLIQUID_TESTNET_ALL_MIDS",
        "HYPERLIQUID_TESTNET_META_AND_ASSET_CONTEXTS",
        "HYPERLIQUID_TESTNET_L2_BOOK",
        "HYPERLIQUID_TESTNET_CLEARINGHOUSE_STATE",
        "HYPERLIQUID_MAINNET_CANDLE_SNAPSHOT",
        "HYPERLIQUID_MAINNET_FUNDING_HISTORY",
        "HYPERLIQUID_MAINNET_META_AND_ASSET_CONTEXTS",
        "HYPERLIQUID_MAINNET_ALL_MIDS",
        "HYPERLIQUID_MAINNET_L2_BOOK",
        "HYPERLIQUID_MAINNET_CLEARINGHOUSE_STATE",
    }
    assert all(
        row.method == "POST" and row.credit_units == 0
        for row in hyperliquid.endpoint_pricing
    )
    preflight = policy.provider("hyperliquid_testnet_preflight")
    assert preflight.account_scope_id == "hyperliquid:testnet:agent_preflight"
    assert preflight.allowed_credential_keys == frozenset(
        {"HYPERLIQUID_TESTNET_AGENT_KEYCHAIN"}
    )
    assert preflight.max_total_requests_per_run == 1
    assert preflight.max_total_credits_per_run == 0
    assert {
        (row.operation, row.method, row.target, row.credit_units)
        for row in preflight.endpoint_pricing
    } == {
        (
            "HYPERLIQUID_TESTNET_USER_ROLE",
            "POST",
            "https://api.hyperliquid-testnet.xyz/info",
            0,
        )
    }
    live_preflight = policy.provider("hyperliquid_live_preflight")
    testnet_execution = policy.provider("hyperliquid_testnet_execution")
    live_execution = policy.provider("hyperliquid_live_execution")
    assert live_preflight.allowed_credential_keys == frozenset(
        {"HYPERLIQUID_LIVE_AGENT_KEYCHAIN"}
    )
    assert testnet_execution.allowed_credential_keys == frozenset(
        {"HYPERLIQUID_TESTNET_AGENT_KEYCHAIN"}
    )
    assert live_execution.allowed_credential_keys == frozenset(
        {"HYPERLIQUID_LIVE_AGENT_KEYCHAIN"}
    )
    assert {
        row.operation for row in live_preflight.endpoint_pricing
    } == {
        "HYPERLIQUID_MAINNET_USER_ROLE",
        "HYPERLIQUID_MAINNET_META",
        "HYPERLIQUID_MAINNET_CLEARINGHOUSE_STATE",
        "HYPERLIQUID_MAINNET_OPEN_ORDERS",
    }
    assert all(
        row.credit_units == 0
        for provider in (live_preflight, testnet_execution, live_execution)
        for row in provider.endpoint_pricing
    )


def test_missing_policy_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(
        ExternalEffectPolicyError,
        match="phase00_external_effect_policy_missing",
    ):
        load_phase00_external_effect_policy(tmp_path, as_of=AS_OF)


def test_every_hyperliquid_policy_operation_matches_gateway_and_environment() -> None:
    policy = load_phase00_external_effect_policy(ROOT, as_of=AS_OF)
    endpoint_by_prefix = {
        "HYPERLIQUID_MAINNET": "https://api.hyperliquid.xyz/info",
        "HYPERLIQUID_TESTNET": "https://api.hyperliquid-testnet.xyz/info",
    }
    for provider in policy.providers:
        for row in provider.endpoint_pricing:
            if not row.operation.startswith("HYPERLIQUID_"):
                continue
            prefix = next(
                candidate
                for candidate in endpoint_by_prefix
                if row.operation.startswith(f"{candidate}_")
            )
            assert row.operation in hyperliquid_info_contract_operations(
                operation_prefix=prefix
            )
            assert row.target == endpoint_by_prefix[prefix]
            assert row.method == "POST"
            assert row.credit_units == 0


def test_pricing_target_outside_allowlist_fails_closed(tmp_path: Path) -> None:
    path = _copy_policy(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["providers"]["crypto_wizards"]["endpoint_pricing"][0]["target"] = (
        "https://unreviewed.example.test/v1/backtest"
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ExternalEffectPolicyError,
        match="phase00_external_effect_policy_pricing_target_not_allowed",
    ):
        load_phase00_external_effect_policy(tmp_path, as_of=AS_OF)


def test_expired_policy_fails_closed(tmp_path: Path) -> None:
    _copy_policy(tmp_path)
    with pytest.raises(
        ExternalEffectPolicyError,
        match="phase00_external_effect_policy_review_expired",
    ):
        load_phase00_external_effect_policy(
            tmp_path,
            as_of=datetime(2027, 1, 1, tzinfo=UTC),
        )


def test_any_policy_change_changes_permit_policy_identity(tmp_path: Path) -> None:
    path = _copy_policy(tmp_path)
    before = load_phase00_external_effect_policy(tmp_path, as_of=AS_OF)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["providers"]["crypto_wizards"]["max_total_requests_per_run"] = 1024
    path.write_text(json.dumps(payload), encoding="utf-8")
    after = load_phase00_external_effect_policy(tmp_path, as_of=AS_OF)

    assert before.policy_sha256 != after.policy_sha256
    assert before.permit_policy_version != after.permit_policy_version
