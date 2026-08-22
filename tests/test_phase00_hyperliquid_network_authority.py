from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path

import pytest

from quant_platform.orchestration.corrective_external_effects import (
    RESEARCH_EXTERNAL_EFFECT_PROFILE,
    ExternalEffectCallContract,
    external_effect_authority_session,
)
from quant_platform.orchestration.corrective_hyperliquid_network import (
    HYPERLIQUID_INFO_OPERATION_SUFFIXES,
    hyperliquid_info_contract_operations,
    hyperliquid_info_operation,
    run_authorized_hyperliquid_info_call,
)
from quant_platform.orchestration.effect_authority import (
    EffectAuthority,
    EffectAuthorityError,
    EffectKind,
)

ROOT = Path(__file__).resolve().parents[1]
TARGET = "https://api.hyperliquid-testnet.xyz/info"
HASH = "a" * 64


def _session(root: Path, *, target: str = TARGET):
    authority = EffectAuthority(
        root=root,
        secret=b"hyperliquid-network-authority-test",
        issuer_id="hyperliquid-network-test-supervisor",
        profile=RESEARCH_EXTERNAL_EFFECT_PROFILE,
    )
    return external_effect_authority_session(
        authority=authority,
        run_id="hyperliquid-network-run",
        intended_slot_id="hyperliquid-network-slot",
        source_fingerprint_sha256=HASH,
        runtime_fingerprint_sha256=HASH,
        configuration_fingerprint_sha256=HASH,
        provider_id="hyperliquid_public",
        account_scope_id="hyperliquid:public_research",
        reservation_id="hyperliquid-network-reservation",
        reservation_sha256=sha256(b"hyperliquid-network-reservation").hexdigest(),
        allowed_targets=frozenset({target}),
        allowed_credential_keys=frozenset(),
        max_total_requests=4,
        max_total_credits=0,
        allowed_call_contracts=frozenset(
            {
                ExternalEffectCallContract(
                    operation="HYPERLIQUID_TESTNET_META",
                    method="POST",
                    target=target,
                    credit_units_per_request=0,
                )
            }
        ),
    )


def test_missing_public_authority_stops_transport_before_call() -> None:
    calls: list[str] = []

    with pytest.raises(EffectAuthorityError, match="session_missing"):
        run_authorized_hyperliquid_info_call(
            target=TARGET,
            payload={"type": "meta"},
            operation_prefix="HYPERLIQUID_TESTNET",
            transport=lambda: calls.append("called"),
        )

    assert calls == []


def test_exact_public_call_consumes_public_permit_and_zero_credits(
    tmp_path: Path,
) -> None:
    with _session(tmp_path) as session:
        result = run_authorized_hyperliquid_info_call(
            target=TARGET,
            payload={"type": "meta"},
            operation_prefix="HYPERLIQUID_TESTNET",
            transport=lambda: {"universe": []},
        )

    assert result == {"universe": []}
    assert session.consumed_requests == 1
    assert session.consumed_credits == 0
    assert session.credential_receipts == []
    assert session.credit_receipts == []
    assert session.network_receipts[0].effect_kind == EffectKind.PUBLIC_NETWORK
    assert session.authority.outcome(
        session.network_receipts[0].permit_id
    ) == "SUCCEEDED"


@pytest.mark.parametrize(
    ("payload", "prefix", "reason"),
    [
        ({"type": "unknown"}, "HYPERLIQUID_TESTNET", "request_type_denied"),
        ({"type": "meta"}, "HYPERLIQUID_DEVNET", "prefix_denied"),
    ],
)
def test_unknown_request_or_environment_fails_before_transport(
    tmp_path: Path,
    payload: dict[str, object],
    prefix: str,
    reason: str,
) -> None:
    calls: list[str] = []
    with _session(tmp_path) as session, pytest.raises(
        EffectAuthorityError,
        match=reason,
    ):
        run_authorized_hyperliquid_info_call(
            target=TARGET,
            payload=payload,
            operation_prefix=prefix,
            transport=lambda: calls.append("called"),
        )

    assert calls == []
    assert session.consumed_requests == 0


def test_target_drift_fails_before_transport(tmp_path: Path) -> None:
    calls: list[str] = []
    with _session(tmp_path) as session, pytest.raises(
        EffectAuthorityError,
        match="external_network_target_denied",
    ):
        run_authorized_hyperliquid_info_call(
            target="https://example.invalid/info",
            payload={"type": "meta"},
            operation_prefix="HYPERLIQUID_TESTNET",
            transport=lambda: calls.append("called"),
        )

    assert calls == []
    assert session.consumed_requests == 0


def test_durable_response_failure_leaves_effect_unknown(tmp_path: Path) -> None:
    with _session(tmp_path) as session, pytest.raises(
        OSError,
        match="durable response write failed",
    ):
        run_authorized_hyperliquid_info_call(
            target=TARGET,
            payload={"type": "meta"},
            operation_prefix="HYPERLIQUID_TESTNET",
            transport=lambda: {"universe": []},
            result_recorder=lambda _value: (_ for _ in ()).throw(
                OSError("durable response write failed")
            ),
        )

    assert session.authority.outcome(
        session.network_receipts[0].permit_id
    ) == "UNKNOWN"


def test_operation_registry_is_complete_deterministic_and_environment_scoped() -> None:
    for prefix in ("HYPERLIQUID_TESTNET", "HYPERLIQUID_MAINNET"):
        expected = {
            f"{prefix}_{suffix}"
            for suffix in HYPERLIQUID_INFO_OPERATION_SUFFIXES.values()
        }
        assert hyperliquid_info_contract_operations(
            operation_prefix=prefix
        ) == expected
        assert {
            hyperliquid_info_operation(
                {"type": request_type},
                operation_prefix=prefix,
            )
            for request_type in HYPERLIQUID_INFO_OPERATION_SUFFIXES
        } == expected


def test_all_hyperliquid_raw_network_sinks_are_private_and_frozen() -> None:
    files = (
        "src/quant_platform/execution.py",
        "src/quant_platform/hyperliquid.py",
        "src/quant_platform/hyperliquid_testnet.py",
        "src/quant_platform/orchestration/hyperliquid_testnet_lifecycle_evidence.py",
        "src/quant_platform/orchestration/corrective_live_parity_capture.py",
        "src/quant_platform/orchestration/corrective_testnet_pair_execution.py",
        "src/quant_platform/orchestration/corrective_live_canary_executor.py",
        "src/quant_platform/orchestration/corrective_live_canary_execution.py",
    )
    expected = {
        ("src/quant_platform/execution.py", "_hyperliquid_post_json"),
        ("src/quant_platform/hyperliquid.py", "_raw_hyperliquid_mainnet_info_http_result"),
        ("src/quant_platform/hyperliquid_testnet.py", "_raw_hyperliquid_info_call"),
        (
            "src/quant_platform/orchestration/hyperliquid_testnet_lifecycle_evidence.py",
            "_raw_hyperliquid_lifecycle_info_call",
        ),
        (
            "src/quant_platform/orchestration/corrective_live_parity_capture.py",
            "_raw_hyperliquid_parity_info_call",
        ),
        (
            "src/quant_platform/orchestration/corrective_testnet_pair_execution.py",
            "_raw_hyperliquid_testnet_info_client",
        ),
        (
            "src/quant_platform/orchestration/corrective_live_canary_executor.py",
            "_raw_hyperliquid_mainnet_info_client",
        ),
        (
            "src/quant_platform/orchestration/corrective_live_canary_execution.py",
            "_raw_hyperliquid_mainnet_info_client",
        ),
    }
    observed: set[tuple[str, str]] = set()
    for relative in files:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        observed.update(
            (relative, function)
            for function in _raw_post_owners(tree)
        )

    assert observed == expected


def _raw_post_owners(tree: ast.AST) -> set[str]:
    owners: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.functions: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node: ast.Call) -> None:
            is_post = isinstance(node.func, ast.Attribute) and node.func.attr == "post"
            is_hyperliquid_urlopen = (
                isinstance(node.func, ast.Name)
                and node.func.id == "urlopen"
                and any("hyperliquid" in name for name in self.functions)
            )
            if is_post or is_hyperliquid_urlopen:
                raw_owner = next(
                    (
                        name
                        for name in reversed(self.functions)
                        if name.startswith("_raw_hyperliquid")
                        or name == "_hyperliquid_post_json"
                    ),
                    "",
                )
                assert raw_owner, f"unowned Hyperliquid POST at line {node.lineno}"
                owners.add(raw_owner)
            self.generic_visit(node)

    Visitor().visit(tree)
    return owners
