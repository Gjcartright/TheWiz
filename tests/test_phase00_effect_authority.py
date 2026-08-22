from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from quant_platform.orchestration.corrective_runtime import atomic_write_text
from quant_platform.orchestration.effect_authority import (
    PHASE00_REPAIR_PROFILE,
    EffectAuthority,
    EffectAuthorityError,
    EffectKind,
    EffectPermit,
    EffectRequest,
    authorize_file_publication,
    publication_authority_session,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _authority(tmp_path: Path, clock: Clock) -> EffectAuthority:
    return EffectAuthority(
        root=tmp_path,
        secret=b"phase00-test-secret-material-32-bytes-minimum",
        issuer_id="phase00_supervisor",
        profile=PHASE00_REPAIR_PROFILE,
        clock=clock,
    )


def _permit(authority: EffectAuthority, tmp_path: Path) -> EffectPermit:
    return authority.issue(
        run_id="run_001",
        intended_slot_id="slot_20260821T120000Z",
        effect_kind=EffectKind.FILE_PUBLICATION,
        target=str((tmp_path / "reports" / "receipt.json").resolve()),
        operation="immutable_create",
        effect_scope="test_publication",
        payload_sha256="e" * 64,
        policy_version="phase00.v1",
        source_fingerprint_sha256=HASH_A,
        runtime_fingerprint_sha256=HASH_B,
        configuration_fingerprint_sha256=HASH_C,
        max_units=2048,
    )


def _request(permit: EffectPermit, *, units: int = 512) -> EffectRequest:
    return EffectRequest.from_permit(permit, actual_units=units)


def test_phase00_profile_denies_every_external_or_trading_effect(tmp_path: Path) -> None:
    clock = Clock()
    authority = _authority(tmp_path, clock)
    denied = set(EffectKind) - {EffectKind.FILE_PUBLICATION}
    for effect_kind in denied:
        with pytest.raises(EffectAuthorityError, match="effect_denied_by_profile"):
            authority.issue(
                run_id="run_001",
                intended_slot_id="slot_001",
                effect_kind=effect_kind,
                target="https://example.invalid",
                operation="call",
                effect_scope="external",
                payload_sha256="e" * 64,
                policy_version="phase00.v1",
                source_fingerprint_sha256=HASH_A,
                runtime_fingerprint_sha256=HASH_B,
                configuration_fingerprint_sha256=HASH_C,
                max_units=1,
                account_scope_id="account_test",
            )


def test_valid_permit_consumes_once_and_journal_is_private(tmp_path: Path) -> None:
    clock = Clock()
    authority = _authority(tmp_path, clock)
    permit = _permit(authority, tmp_path)
    receipt = authority.consume(permit, _request(permit))
    assert receipt.status == "CONSUMED"
    assert authority.state(permit.permit_id) == "CONSUMED"
    assert authority.journal_path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(EffectAuthorityError, match="not_usable:CONSUMED"):
        authority.consume(permit, _request(permit))


def test_tampered_permit_fails_signature_before_consumption(tmp_path: Path) -> None:
    clock = Clock()
    authority = _authority(tmp_path, clock)
    permit = _permit(authority, tmp_path)
    tampered = permit.model_copy(update={"max_units": permit.max_units + 1})
    with pytest.raises(EffectAuthorityError, match="signature_invalid"):
        authority.consume(tampered, _request(tampered))
    assert authority.state(permit.permit_id) == "ISSUED"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target", "/tmp/wrong.json"),
        ("operation", "replace"),
        ("run_id", "wrong_run"),
        ("intended_slot_id", "wrong_slot"),
        ("source_fingerprint_sha256", "d" * 64),
    ],
)
def test_wrong_scope_request_fails_closed(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    clock = Clock()
    authority = _authority(tmp_path, clock)
    permit = _permit(authority, tmp_path)
    request = _request(permit).model_copy(update={field: value})
    with pytest.raises(EffectAuthorityError, match=f"scope_mismatch:{field}"):
        authority.consume(permit, request)
    assert authority.state(permit.permit_id) == "ISSUED"


def test_oversized_request_does_not_consume_permit(tmp_path: Path) -> None:
    clock = Clock()
    authority = _authority(tmp_path, clock)
    permit = _permit(authority, tmp_path)
    with pytest.raises(EffectAuthorityError, match="units_exceed_permit"):
        authority.consume(permit, _request(permit, units=permit.max_units + 1))
    assert authority.state(permit.permit_id) == "ISSUED"


def test_expired_and_revoked_permits_fail_closed(tmp_path: Path) -> None:
    clock = Clock()
    authority = _authority(tmp_path, clock)
    expired = _permit(authority, tmp_path)
    clock.now += timedelta(minutes=6)
    with pytest.raises(EffectAuthorityError, match="permit_expired"):
        authority.consume(expired, _request(expired))

    clock.now = datetime(2026, 8, 21, 13, 0, tzinfo=UTC)
    revoked = _permit(authority, tmp_path)
    authority.revoke(revoked.permit_id, reason="operator_stop")
    assert authority.state(revoked.permit_id) == "REVOKED"
    with pytest.raises(EffectAuthorityError, match="not_usable:REVOKED"):
        authority.consume(revoked, _request(revoked))


def test_concurrent_replay_consumes_at_most_once(tmp_path: Path) -> None:
    clock = Clock()
    authority = _authority(tmp_path, clock)
    permit = _permit(authority, tmp_path)
    request = _request(permit)

    def consume() -> str:
        try:
            return authority.consume(permit, request).status
        except EffectAuthorityError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(lambda _: consume(), range(16)))
    assert outcomes.count("CONSUMED") == 1
    assert authority.state(permit.permit_id) == "CONSUMED"


def test_consumed_effect_requires_append_once_outcome_closure(tmp_path: Path) -> None:
    authority = _authority(tmp_path, Clock())
    permit = _permit(authority, tmp_path)
    authority.consume(permit, _request(permit))

    assert authority.outcome(permit.permit_id) == "IN_FLIGHT_OR_UNKNOWN"
    authority.record_outcome(
        permit.permit_id,
        status="SUCCEEDED",
        detail_sha256="d" * 64,
    )
    assert authority.outcome(permit.permit_id) == "SUCCEEDED"
    with pytest.raises(EffectAuthorityError, match="outcome_already_recorded"):
        authority.record_outcome(
            permit.permit_id,
            status="FAILED",
            detail_sha256="e" * 64,
        )


def test_unconsumed_effect_cannot_receive_outcome(tmp_path: Path) -> None:
    authority = _authority(tmp_path, Clock())
    permit = _permit(authority, tmp_path)
    with pytest.raises(EffectAuthorityError, match="permit_not_consumed:ISSUED"):
        authority.record_outcome(
            permit.permit_id,
            status="SUCCEEDED",
            detail_sha256="d" * 64,
        )


def test_symlinked_authority_journal_is_denied(tmp_path: Path) -> None:
    control = tmp_path / ".runtime_control"
    control.mkdir()
    target = tmp_path / "attacker.sqlite3"
    target.write_bytes(b"")
    (control / "effect_authority.sqlite3").symlink_to(target)
    with pytest.raises(EffectAuthorityError, match="journal_not_regular"):
        _authority(tmp_path, Clock())


def test_authority_construction_has_no_filesystem_side_effect(tmp_path: Path) -> None:
    authority = _authority(tmp_path, Clock())

    assert authority.journal_path.exists() is False
    assert (tmp_path / ".runtime_control").exists() is False


def test_v1_journal_migrates_additively_to_external_reservations(
    tmp_path: Path,
) -> None:
    control = tmp_path / ".runtime_control"
    control.mkdir(mode=0o700)
    journal = control / "effect_authority.sqlite3"
    with sqlite3.connect(journal) as connection:
        connection.execute(
            "CREATE TABLE journal_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO journal_metadata (key, value) VALUES ('schema_version', '1')"
        )

    authority = _authority(tmp_path, Clock())
    authority.prepare_journal()

    with sqlite3.connect(authority.journal_path) as connection:
        schema = connection.execute(
            "SELECT value FROM journal_metadata WHERE key = 'schema_version'"
        ).fetchone()
        reservation_table = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'external_reservations'
            """
        ).fetchone()
    assert schema == ("2",)
    assert reservation_table == ("external_reservations",)
    assert authority.journal_path.stat().st_mode & 0o777 == 0o600


def test_external_reservation_reregistration_is_time_independent(
    tmp_path: Path,
) -> None:
    clock = Clock()
    authority = _authority(tmp_path, clock)
    kwargs = {
        "run_id": "run_001",
        "intended_slot_id": "slot_001",
        "provider_id": "crypto_wizards",
        "account_scope_id": "wizard-research",
        "reservation_id": "wizard-reservation-001",
        "reservation_sha256": HASH_A,
        "max_total_requests": 7,
        "max_total_credits": 11,
    }

    first = authority.register_external_reservation(**kwargs)
    clock.now += timedelta(minutes=1)
    second = authority.register_external_reservation(**kwargs)

    assert second == first
    authority.fail_open_external_reservations_without_provider_effects(
        run_id="run_001",
        intended_slot_id="slot_001",
    )
    assert authority.external_reservation_retry_safe(
        reservation_id="wizard-reservation-001",
        reservation_sha256=HASH_A,
    ) is True


def test_publication_session_binds_scope_target_payload_and_budget(
    tmp_path: Path,
) -> None:
    clock = Clock()
    authority = _authority(tmp_path, clock)
    target = tmp_path / "reports" / "receipt.json"
    with publication_authority_session(
        authority=authority,
        run_id="run_001",
        intended_slot_id="slot_001",
        policy_version="phase00.v1",
        source_fingerprint_sha256=HASH_A,
        runtime_fingerprint_sha256=HASH_B,
        configuration_fingerprint_sha256=HASH_C,
        allowed_scopes=frozenset({"receipt"}),
        allowed_target_prefixes=(tmp_path / "reports",),
        max_total_bytes=4,
    ) as session:
        receipt = authorize_file_publication(
            root=tmp_path,
            target=target,
            publication_scope="receipt",
            operation="create",
            payload=b"data",
        )
        assert receipt.status == "CONSUMED"
        assert session.consumed_bytes == 4
        with pytest.raises(EffectAuthorityError, match="budget_exhausted"):
            authorize_file_publication(
                root=tmp_path,
                target=target,
                publication_scope="receipt",
                operation="replace",
                payload=b"x",
            )


def test_publication_session_denies_wrong_scope_and_target(tmp_path: Path) -> None:
    authority = _authority(tmp_path, Clock())
    with publication_authority_session(
        authority=authority,
        run_id="run_001",
        intended_slot_id="slot_001",
        policy_version="phase00.v1",
        source_fingerprint_sha256=HASH_A,
        runtime_fingerprint_sha256=HASH_B,
        configuration_fingerprint_sha256=HASH_C,
        allowed_scopes=frozenset({"receipt"}),
        allowed_target_prefixes=(tmp_path / "reports",),
        max_total_bytes=100,
    ):
        with pytest.raises(EffectAuthorityError, match="scope_denied"):
            authorize_file_publication(
                root=tmp_path,
                target=tmp_path / "reports" / "x.json",
                publication_scope="wrong",
                operation="create",
                payload=b"x",
            )
        with pytest.raises(EffectAuthorityError, match="target_denied"):
            authorize_file_publication(
                root=tmp_path,
                target=tmp_path / "data" / "x.json",
                publication_scope="receipt",
                operation="create",
                payload=b"x",
            )


def test_publication_session_denies_high_confidence_secret_before_permit(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path, Clock())
    target = tmp_path / "reports" / "secret.json"
    with publication_authority_session(
        authority=authority,
        run_id="run_001",
        intended_slot_id="slot_001",
        policy_version="phase00.v1",
        source_fingerprint_sha256=HASH_A,
        runtime_fingerprint_sha256=HASH_B,
        configuration_fingerprint_sha256=HASH_C,
        allowed_scopes=frozenset({"receipt"}),
        allowed_target_prefixes=(tmp_path / "reports",),
        max_total_bytes=1_024,
    ) as session:
        with pytest.raises(
            EffectAuthorityError,
            match="publication_payload_secret_detected:openai_key",
        ):
            authorize_file_publication(
                root=tmp_path,
                target=target,
                publication_scope="receipt",
                operation="create",
                payload=b'{"error":"sk-proj-phase00Canary123456789"}',
            )

        assert session.consumed_writes == 0
        assert session.consumed_bytes == 0


def test_unmanaged_write_requires_exact_external_target_authority(
    tmp_path: Path,
) -> None:
    authority_root = tmp_path / "authority-root"
    target = tmp_path / "external" / "agent.plist"
    wrong = tmp_path / "external" / "wrong.plist"
    authority = _authority(authority_root, Clock())

    with pytest.raises(
        EffectAuthorityError,
        match="unmanaged_publication_authority_session_missing",
    ):
        atomic_write_text(target, "payload", publication_scope="scheduler_config")

    with publication_authority_session(
        authority=authority,
        run_id="run_external",
        intended_slot_id="slot_external",
        policy_version="phase00.v1",
        source_fingerprint_sha256=HASH_A,
        runtime_fingerprint_sha256=HASH_B,
        configuration_fingerprint_sha256=HASH_C,
        allowed_scopes=frozenset({"scheduler_config"}),
        allowed_target_prefixes=(authority_root,),
        allowed_external_targets=(target,),
        max_total_bytes=100,
    ):
        atomic_write_text(
            target,
            "payload",
            publication_scope="scheduler_config",
        )
        with pytest.raises(EffectAuthorityError, match="target_denied"):
            atomic_write_text(
                wrong,
                "wrong",
                publication_scope="scheduler_config",
            )

    assert target.read_text(encoding="utf-8") == "payload"
    assert not wrong.exists()
