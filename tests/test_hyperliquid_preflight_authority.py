from __future__ import annotations

import pytest
from eth_account import Account

from quant_platform.hyperliquid_testnet import (
    HyperliquidTestnetConfig,
    HyperliquidTestnetPairExecutor,
)
from quant_platform.orchestration.corrective_live_canary_executor import (
    read_live_agent_key_from_keychain,
)
from quant_platform.orchestration.effect_authority import EffectAuthorityError


class _ForbiddenSession:
    def __init__(self) -> None:
        self.requests = 0

    def post(self, *_args, **_kwargs):
        self.requests += 1
        raise AssertionError("network called without external-effect authority")


def test_no_order_preflight_blocks_before_keychain_and_network_without_authority() -> None:
    wallet = Account.create()
    keychain_reads = 0

    def reader(_service: str, _account: str) -> str:
        nonlocal keychain_reads
        keychain_reads += 1
        return wallet.key.hex()

    session = _ForbiddenSession()
    executor = HyperliquidTestnetPairExecutor(
        session=session,
        keychain_reader=reader,
    )
    config = HyperliquidTestnetConfig(
        master_address=Account.create().address,
        agent_address=wallet.address,
        keychain_service="thewiz-hyperliquid-testnet-agent",
    )

    result = executor.no_order_preflight(config)

    assert result["ready_for_no_order_preflight"] is False
    assert "external_effect_authority_session_missing" in str(result["blockers"])
    assert keychain_reads == 0
    assert session.requests == 0


def test_live_keychain_reader_blocks_before_injected_reader_without_authority() -> None:
    keychain_reads = 0

    def reader(_service: str, _account: str) -> str:
        nonlocal keychain_reads
        keychain_reads += 1
        return "forbidden"

    with pytest.raises(
        EffectAuthorityError,
        match="external_effect_authority_session_missing",
    ):
        read_live_agent_key_from_keychain(
            "thewiz-live-agent",
            Account.create().address,
            reader=reader,
        )

    assert keychain_reads == 0
