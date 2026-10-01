from __future__ import annotations

import ast
import os
import socket
from hashlib import sha256
from pathlib import Path

import pytest

from quant_platform.orchestration.corrective_effect_guard import (
    Phase00EffectDenied,
    phase00_effect_guard,
)
from quant_platform.orchestration.corrective_external_effects import (
    RESEARCH_EXTERNAL_EFFECT_PROFILE,
    ExternalEffectCallContract,
    ExternalEffectIssuer,
    current_external_effect_issuer,
    external_effect_authority_session,
    external_effect_issuer_bundle_session,
    external_effect_issuer_session,
    external_effect_provider_session,
    read_authorized_credential,
    read_authorized_keychain_credential,
    reserved_external_effect_session,
    run_authorized_credit_call,
    run_authorized_public_call,
)
from quant_platform.orchestration.effect_authority import (
    PHASE00_REPAIR_PROFILE,
    EffectAuthority,
    EffectAuthorityError,
    EffectKind,
)

HASH = "a" * 64
TARGET = "https://api.cryptowizards.net/v1/custom-series"
ROOT = Path(__file__).resolve().parents[1]


def _authority(root: Path, *, repair_only: bool = False) -> EffectAuthority:
    return EffectAuthority(
        root=root,
        secret=b"x" * 32,
        issuer_id="test-supervisor",
        profile=PHASE00_REPAIR_PROFILE if repair_only else RESEARCH_EXTERNAL_EFFECT_PROFILE,
    )


def _session(root: Path, *, repair_only: bool = False):
    return external_effect_authority_session(
        authority=_authority(root, repair_only=repair_only),
        run_id="run-1",
        intended_slot_id="slot-1",
        source_fingerprint_sha256=HASH,
        runtime_fingerprint_sha256=HASH,
        configuration_fingerprint_sha256=HASH,
        provider_id="crypto_wizards",
        account_scope_id="wizard-research-account",
        reservation_id="reservation-1",
        reservation_sha256=sha256(b"reservation").hexdigest(),
        allowed_targets=frozenset({TARGET}),
        allowed_credential_keys=frozenset({"CRYPTO_WIZARDS_API_KEY"}),
        max_total_requests=2,
        max_total_credits=20,
    )


def test_external_effect_requires_supervisor_session() -> None:
    with pytest.raises(EffectAuthorityError, match="session_missing"):
        read_authorized_credential("CRYPTO_WIZARDS_API_KEY", reader=lambda _key: "secret")


def test_public_network_requires_supervisor_session_before_callback() -> None:
    called = False

    def callback() -> None:
        nonlocal called
        called = True

    with pytest.raises(EffectAuthorityError, match="session_missing"):
        run_authorized_public_call(
            target=TARGET,
            operation="PUBLIC_READ",
            method="GET",
            request_payload=b"{}",
            request_count=1,
            callback=callback,
        )

    assert called is False


def test_phase00_file_only_profile_cannot_issue_external_effect(tmp_path: Path) -> None:
    with _session(tmp_path, repair_only=True), pytest.raises(
        EffectAuthorityError,
        match="effect_denied_by_profile",
    ):
        read_authorized_credential(
            "CRYPTO_WIZARDS_API_KEY",
            reader=lambda _key: "secret",
        )


def test_permit_bound_credential_and_call_work_inside_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRYPTO_WIZARDS_API_KEY", "secret-value")
    observed: list[tuple[str, int]] = []

    def fake_create_connection(address, *args, **kwargs):
        del args, kwargs
        observed.append(address)
        return object()

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)
    with _session(tmp_path) as session, phase00_effect_guard():
        secret = read_authorized_credential("CRYPTO_WIZARDS_API_KEY")
        result = run_authorized_credit_call(
            target=TARGET,
            operation="POST",
            request_payload=b'{"pair":"BTC-ETH"}',
            request_count=1,
            credit_cost=10,
            callback=lambda: (secret, socket.create_connection(("api.cryptowizards.net", 443))),
        )

    assert result[0] == "secret-value"
    assert observed == [("api.cryptowizards.net", 443)]
    assert session.consumed_requests == 1
    assert session.consumed_credits == 10
    assert len(session.credential_receipts) == 1
    assert len(session.network_receipts) == 1
    assert len(session.credit_receipts) == 1
    assert session.network_receipts[0].effect_kind == EffectKind.AUTHENTICATED_NETWORK
    assert (
        session.authority.outcome(session.network_receipts[0].permit_id)
        == "SUCCEEDED"
    )
    assert (
        session.authority.outcome(session.credit_receipts[0].permit_id)
        == "SUCCEEDED"
    )
    accounting = session.authority.run_accounting(
        run_id="run-1",
        intended_slot_id="slot-1",
    )
    assert accounting["external_reservations"] == 1
    assert accounting["external_credits_reserved"] == 20
    assert accounting["external_credits_consumed"] == 10
    assert accounting["external_calls"] == 1
    assert accounting["accounting_complete"] is True


def test_keychain_credential_requires_permit_and_exact_item(tmp_path: Path) -> None:
    observed: list[tuple[str, str]] = []

    def reader(service: str, account: str) -> str:
        observed.append((service, account))
        return "secret-value"

    with _session(tmp_path) as session, phase00_effect_guard():
        secret = read_authorized_keychain_credential(
            "CRYPTO_WIZARDS_API_KEY",
            service="thewiz-service",
            account="research-account",
            reader=reader,
        )

    assert secret == "secret-value"
    assert observed == [("thewiz-service", "research-account")]
    assert len(session.credential_receipts) == 1
    assert session.authority.outcome(
        session.credential_receipts[0].permit_id
    ) == "SUCCEEDED"


def test_keychain_reader_is_not_called_without_external_session() -> None:
    called = False

    def reader(_service: str, _account: str) -> str:
        nonlocal called
        called = True
        return "secret"

    with pytest.raises(EffectAuthorityError, match="session_missing"):
        read_authorized_keychain_credential(
            "CRYPTO_WIZARDS_API_KEY",
            service="service",
            account="account",
            reader=reader,
        )

    assert called is False


def test_all_raw_keychain_subprocess_sinks_are_private_and_frozen() -> None:
    expected = {
        (
            "src/quant_platform/hyperliquid_testnet.py",
            "_read_hyperliquid_agent_key_from_keychain",
        ),
        (
            "src/quant_platform/orchestration/corrective_live_canary_executor.py",
            "_read_live_agent_key_from_keychain",
        ),
    }
    observed: set[tuple[str, str]] = set()
    for path in sorted((ROOT / "src/quant_platform").rglob("*.py")):
        if path.name.startswith("._"):
            continue
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in (
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ):
            for call in (
                node for node in ast.walk(function) if isinstance(node, ast.Call)
            ):
                if not call.args or not isinstance(call.args[0], (ast.List, ast.Tuple)):
                    continue
                command = {
                    item.value
                    for item in call.args[0].elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                }
                if {"security", "find-generic-password"}.issubset(command):
                    observed.add((relative, function.name))

    assert observed == expected


def test_external_callback_crash_preserves_exact_failed_reservation(
    tmp_path: Path,
) -> None:
    with (
        pytest.raises(RuntimeError, match="simulated response loss"),
        _session(tmp_path) as session,
        phase00_effect_guard(),
    ):
            read_authorized_credential(
                "CRYPTO_WIZARDS_API_KEY",
                reader=lambda _key: "secret",
            )
            run_authorized_credit_call(
                target=TARGET,
                operation="POST",
                request_payload=b"{}",
                request_count=1,
                credit_cost=10,
                callback=lambda: (_ for _ in ()).throw(
                    RuntimeError("simulated response loss")
                ),
            )

    accounting = session.authority.run_accounting(
        run_id="run-1",
        intended_slot_id="slot-1",
    )
    assert accounting["failed_reservations"] == 1
    assert accounting["external_calls"] == 1
    assert accounting["external_credits_consumed"] == 10
    assert accounting["unknown_effects"] == 2
    assert "external_reservation_failed" in accounting["blockers"]
    assert "effect_outcome_unknown_or_in_flight" in accounting["blockers"]


def test_authorized_call_cannot_retarget_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRYPTO_WIZARDS_API_KEY", "secret-value")
    monkeypatch.setattr(socket, "create_connection", lambda *_args, **_kwargs: object())
    with _session(tmp_path), phase00_effect_guard():
        read_authorized_credential("CRYPTO_WIZARDS_API_KEY")
        with pytest.raises(Phase00EffectDenied, match="network_target_denied"):
            run_authorized_credit_call(
                target=TARGET,
                operation="POST",
                request_payload=b"{}",
                request_count=1,
                credit_cost=10,
                callback=lambda: socket.create_connection(("example.com", 443)),
            )


def test_budget_and_credential_reuse_fail_closed(tmp_path: Path) -> None:
    with _session(tmp_path) as session:
        assert read_authorized_credential(
            "CRYPTO_WIZARDS_API_KEY",
            reader=lambda _key: "secret",
        ) == "secret"
        with pytest.raises(EffectAuthorityError, match="credential_reuse_denied"):
            read_authorized_credential(
                "CRYPTO_WIZARDS_API_KEY",
                reader=lambda _key: "secret",
            )
        with pytest.raises(EffectAuthorityError, match="credit_budget_exhausted"):
            run_authorized_credit_call(
                target=TARGET,
                operation="POST",
                request_payload=b"{}",
                request_count=1,
                credit_cost=21,
                callback=lambda: None,
            )
        assert session.consumed_credits == 0


def test_missing_credential_consumes_no_network_or_credit_budget(tmp_path: Path) -> None:
    with _session(tmp_path) as session, pytest.raises(
        EffectAuthorityError,
        match="credential_missing",
    ):
        read_authorized_credential(
            "CRYPTO_WIZARDS_API_KEY",
            reader=lambda _key: None,
        )
    assert session.consumed_requests == 0
    assert session.consumed_credits == 0
    assert len(session.credential_receipts) == 1
    assert (
        session.authority.outcome(session.credential_receipts[0].permit_id)
        == "FAILED"
    )


def test_issuer_derives_only_reservation_bounded_session(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    with external_effect_issuer_session(
        authority=authority,
        run_id="run-1",
        intended_slot_id="slot-1",
        source_fingerprint_sha256=HASH,
        runtime_fingerprint_sha256=HASH,
        configuration_fingerprint_sha256=HASH,
        provider_id="crypto_wizards",
        account_scope_id="wizard-research-account",
        allowed_targets=frozenset({TARGET}),
        allowed_credential_keys=frozenset({"CRYPTO_WIZARDS_API_KEY"}),
        max_total_requests=3,
        max_total_credits=20,
    ), reserved_external_effect_session(
        reservation_id="reservation-1",
        reservation_sha256=sha256(b"reservation").hexdigest(),
        max_total_requests=2,
        max_total_credits=10,
    ) as session:
        assert session.reservation_id == "reservation-1"
        assert session.max_total_credits == 10


def test_zero_credit_network_call_records_no_credit_permit(tmp_path: Path) -> None:
    with _session(tmp_path) as session:
        read_authorized_credential(
            "CRYPTO_WIZARDS_API_KEY",
            reader=lambda _key: "secret",
        )
        assert run_authorized_credit_call(
            target=TARGET,
            operation="GET_CREDITS_USED",
            request_payload=b"{}",
            request_count=1,
            credit_cost=0,
            callback=lambda: {"used": 10},
        ) == {"used": 10}
    assert session.consumed_credits == 0
    assert session.credit_receipts == []
    assert session.authority.outcome(session.network_receipts[0].permit_id) == "SUCCEEDED"


def test_public_call_needs_no_credential_and_records_public_effect(
    tmp_path: Path,
) -> None:
    with _session(tmp_path) as session:
        assert run_authorized_public_call(
            target=TARGET,
            operation="PUBLIC_READ",
            method="GET",
            request_payload=b"{}",
            request_count=1,
            callback=lambda: {"rows": 1},
        ) == {"rows": 1}

    assert session.consumed_requests == 1
    assert session.consumed_credits == 0
    assert session.credential_receipts == []
    assert session.credit_receipts == []
    assert session.network_receipts[0].effect_kind == EffectKind.PUBLIC_NETWORK
    assert session.authority.outcome(
        session.network_receipts[0].permit_id
    ) == "SUCCEEDED"


def test_callback_exception_closes_effect_as_unknown(tmp_path: Path) -> None:
    with _session(tmp_path) as session:
        read_authorized_credential(
            "CRYPTO_WIZARDS_API_KEY",
            reader=lambda _key: "secret",
        )

        def fail() -> None:
            raise RuntimeError("provider response ambiguous")

        with pytest.raises(RuntimeError, match="provider response ambiguous"):
            run_authorized_credit_call(
                target=TARGET,
                operation="POST",
                request_payload=b"{}",
                request_count=1,
                credit_cost=2,
                callback=fail,
            )
    assert session.authority.outcome(session.network_receipts[0].permit_id) == "UNKNOWN"
    assert session.authority.outcome(session.credit_receipts[0].permit_id) == "UNKNOWN"


def test_external_success_is_recorded_only_after_result_evidence(tmp_path: Path) -> None:
    evidence_hash = "b" * 64
    observed: list[dict[str, int]] = []
    with _session(tmp_path) as session:
        read_authorized_credential(
            "CRYPTO_WIZARDS_API_KEY",
            reader=lambda _key: "secret",
        )
        result = run_authorized_credit_call(
            target=TARGET,
            operation="POST",
            request_payload=b"{}",
            request_count=1,
            credit_cost=2,
            callback=lambda: {"rows": 1},
            result_recorder=lambda value: observed.append(value) or evidence_hash,
        )

    assert result == {"rows": 1}
    assert observed == [{"rows": 1}]
    assert session.authority.outcome(session.network_receipts[0].permit_id) == "SUCCEEDED"
    assert session.authority.outcome(session.credit_receipts[0].permit_id) == "SUCCEEDED"


def test_result_evidence_failure_leaves_provider_effect_unknown(tmp_path: Path) -> None:
    with _session(tmp_path) as session:
        read_authorized_credential(
            "CRYPTO_WIZARDS_API_KEY",
            reader=lambda _key: "secret",
        )

        def fail_recorder(_result: object) -> str:
            raise OSError("simulated durable evidence failure")

        with pytest.raises(OSError, match="simulated durable evidence failure"):
            run_authorized_credit_call(
                target=TARGET,
                operation="POST",
                request_payload=b"{}",
                request_count=1,
                credit_cost=2,
                callback=lambda: {"rows": 1},
                result_recorder=fail_recorder,
            )

    assert session.authority.outcome(session.network_receipts[0].permit_id) == "UNKNOWN"
    assert session.authority.outcome(session.credit_receipts[0].permit_id) == "UNKNOWN"
    accounting = session.authority.run_accounting(
        run_id="run-1",
        intended_slot_id="slot-1",
    )
    assert accounting["unknown_effects"] == 2
    assert accounting["accounting_complete"] is False


def test_sensitive_environment_is_still_denied_without_exact_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRYPTO_WIZARDS_API_KEY", "secret")
    with phase00_effect_guard(), pytest.raises(
        Phase00EffectDenied,
        match="sensitive_env_denied",
    ):
        os.getenv("CRYPTO_WIZARDS_API_KEY")


def test_policy_bound_call_contract_denies_operation_method_and_credit_drift(
    tmp_path: Path,
) -> None:
    callback_calls: list[str] = []
    contract = ExternalEffectCallContract(
        operation="PRESCANNED_GET",
        method="GET",
        target=TARGET,
        credit_units_per_request=10,
    )
    with external_effect_authority_session(
        authority=_authority(tmp_path),
        run_id="contract-run",
        intended_slot_id="contract-slot",
        source_fingerprint_sha256=HASH,
        runtime_fingerprint_sha256=HASH,
        configuration_fingerprint_sha256=HASH,
        provider_id="crypto_wizards",
        account_scope_id="wizard-research-account",
        reservation_id="contract-reservation",
        reservation_sha256=sha256(b"contract-reservation").hexdigest(),
        allowed_targets=frozenset({TARGET}),
        allowed_credential_keys=frozenset(),
        max_total_requests=2,
        max_total_credits=20,
        allowed_call_contracts=frozenset({contract}),
    ) as session:
        attempts = (
            ("WRONG_OPERATION", "GET", 10),
            ("PRESCANNED_GET", "POST", 10),
            ("PRESCANNED_GET", "GET", 9),
        )
        for operation, method, credits in attempts:
            expected = (
                "external_call_credit_contract_mismatch"
                if credits == 9
                else "external_call_contract_denied"
            )
            with pytest.raises(EffectAuthorityError, match=expected):
                run_authorized_credit_call(
                    target=TARGET,
                    operation=operation,
                    method=method,
                    request_payload=b"{}",
                    request_count=1,
                    credit_cost=credits,
                    callback=lambda operation=operation: callback_calls.append(
                        operation
                    ),
                )

    assert callback_calls == []
    assert session.consumed_requests == 0
    assert session.consumed_credits == 0


def test_provider_bundle_requires_exact_selection_and_isolates_public_calls(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path)
    common = {
        "authority": authority,
        "run_id": "bundle-run",
        "intended_slot_id": "bundle-slot",
        "policy_version": "bundle-policy",
        "source_fingerprint_sha256": HASH,
        "runtime_fingerprint_sha256": HASH,
        "configuration_fingerprint_sha256": HASH,
    }
    wizard = ExternalEffectIssuer(
        **common,
        provider_id="crypto_wizards",
        account_scope_id="wizard-research-account",
        allowed_targets=frozenset({TARGET}),
        allowed_credential_keys=frozenset({"CRYPTO_WIZARDS_API_KEY"}),
        max_total_requests=1,
        max_total_credits=10,
        allowed_call_contracts=frozenset(
            {
                ExternalEffectCallContract(
                    operation="PRESCANNED_GET",
                    method="GET",
                    target=TARGET,
                    credit_units_per_request=10,
                )
            }
        ),
    )
    public_target = "https://api.hyperliquid-testnet.xyz/info"
    hyperliquid = ExternalEffectIssuer(
        **common,
        provider_id="hyperliquid_public",
        account_scope_id="hyperliquid:testnet:public_research",
        allowed_targets=frozenset({public_target}),
        allowed_credential_keys=frozenset(),
        max_total_requests=2,
        max_total_credits=0,
        allowed_call_contracts=frozenset(
            {
                ExternalEffectCallContract(
                    operation="HYPERLIQUID_TESTNET_META",
                    method="POST",
                    target=public_target,
                    credit_units_per_request=0,
                )
            }
        ),
    )
    observed: list[str] = []

    with external_effect_issuer_bundle_session((wizard, hyperliquid)):
        assert current_external_effect_issuer() is None
        with (
            pytest.raises(
                EffectAuthorityError,
                match="external_effect_issuer_missing",
            ),
            reserved_external_effect_session(
                reservation_id="unselected",
                reservation_sha256=sha256(b"unselected").hexdigest(),
                max_total_requests=1,
                max_total_credits=0,
            ),
        ):
            pass
        with external_effect_provider_session("hyperliquid_public"):
            assert current_external_effect_issuer() == hyperliquid
            with pytest.raises(
                EffectAuthorityError,
                match="external_effect_provider_mismatch",
            ), external_effect_provider_session("crypto_wizards"):
                pass
            with reserved_external_effect_session(
                reservation_id="public-reservation",
                reservation_sha256=sha256(b"public-reservation").hexdigest(),
                max_total_requests=1,
                max_total_credits=0,
            ) as session:
                assert run_authorized_public_call(
                    target=public_target,
                    operation="HYPERLIQUID_TESTNET_META",
                    method="POST",
                    request_payload=b'{"type":"meta"}',
                    request_count=1,
                    callback=lambda: observed.append("meta") or {"universe": []},
                ) == {"universe": []}
                with pytest.raises(
                    EffectAuthorityError,
                    match="external_network_target_denied",
                ):
                    run_authorized_public_call(
                        target=TARGET,
                        operation="HYPERLIQUID_TESTNET_META",
                        method="POST",
                        request_payload=b"{}",
                        request_count=1,
                        callback=lambda: observed.append("cross-provider"),
                    )

    accounting = authority.run_accounting(
        run_id="bundle-run",
        intended_slot_id="bundle-slot",
    )
    assert observed == ["meta"]
    assert session.consumed_requests == 1
    assert session.consumed_credits == 0
    assert session.network_receipts[0].effect_kind == EffectKind.PUBLIC_NETWORK
    assert accounting["external_calls"] == 1
    assert accounting["external_credits_consumed"] == 0
    assert accounting["accounting_complete"] is True
