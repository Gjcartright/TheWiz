from __future__ import annotations

import socket
import subprocess
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path

import pytest

from quant_platform.orchestration.corrective_external_effects import (
    RESEARCH_EXTERNAL_EFFECT_PROFILE,
    external_effect_authority_session,
)
from quant_platform.orchestration.corrective_scheduler_lock import (
    SCOPED_TARGET_PREFIXES,
    governed_evidence_read_lock,
)
from quant_platform.orchestration.effect_authority import (
    PHASE00_REPAIR_PROFILE,
    EffectAuthority,
    publication_authority_session,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GOVERNED_EVIDENCE_PATHS = (
    "reports/active",
    "reports/dashboard",
    "data/external/youtube",
    "data/processed/youtube_brain",
)
NETWORK_SUBPROCESSES = {
    "curl",
    "ftp",
    "nc",
    "ncat",
    "netcat",
    "scp",
    "security",
    "sftp",
    "ssh",
    "telnet",
    "wget",
    "yt-dlp",
    "yt_dlp",
}
TEST_FINGERPRINT = "f" * 64
KEYCHAIN_AUTHORITY_TEST_FILES = {
    "test_corrective_live_canary.py",
    "test_corrective_live_canary_executor.py",
    "test_corrective_live_parity_capture.py",
    "test_corrective_testnet_pair_execution.py",
    "test_corrective_testnet_collateral_transfer.py",
    "test_current_wizard_hyperliquid_handoff.py",
    "test_gate00g_order_authority.py",
    "test_hyperliquid_testnet_lifecycle_evidence.py",
}
KEYCHAIN_AUTHORITY_TEST_NODES = {
    (
        "test_corrective_release_gates.py",
        "test_failed_submitted_attempt_is_in_execution_failure_denominator",
    ),
    (
        "test_corrective_release_gates.py",
        "test_unresolved_submitted_attempt_blocks_sample_release",
    ),
}


def _governed_evidence_hash() -> str:
    digest = sha256()
    result = subprocess.run(
        [
            "git",
            "diff",
            "--no-ext-diff",
            "--binary",
            "--",
            *GOVERNED_EVIDENCE_PATHS,
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    digest.update(result.stdout)
    for relative_root in GOVERNED_EVIDENCE_PATHS:
        evidence_root = REPOSITORY_ROOT / relative_root
        digest.update(f"root:{relative_root}\n".encode())
        if not evidence_root.exists():
            digest.update(b"missing\n")
            continue
        for path in sorted(evidence_root.rglob("*")):
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            stat = path.lstat()
            digest.update(
                f"{relative}\0{stat.st_mode}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
            )
            if path.is_symlink():
                digest.update(f"target:{path.readlink()}\n".encode())
    return digest.hexdigest()


@pytest.fixture(scope="session", autouse=True)
def preserve_governed_evidence_during_tests():
    with governed_evidence_read_lock(
        REPOSITORY_ROOT,
        blocking=True,
        create_if_missing=True,
    ):
        before = _governed_evidence_hash()
        yield
        after = _governed_evidence_hash()
        assert after == before, (
            "pytest mutated governed project evidence; tests must use temporary roots "
            "and must not refresh live sources"
        )


@pytest.fixture(autouse=True)
def deny_unmarked_external_network(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    if request.node.get_closest_marker("external_network") is not None:
        return

    def blocked_network(*args, **kwargs):
        del args, kwargs
        raise OSError("external network access is disabled during unit tests")

    original_run = subprocess.run
    original_popen = subprocess.Popen

    def command_tokens(popenargs, kwargs):
        command = kwargs.get("args", popenargs[0] if popenargs else ())
        return (
            [str(value) for value in command]
            if not isinstance(command, str)
            else command.split()
        )

    def deny_external_subprocess(tokens):
        normalized = {Path(token).name.lower() for token in tokens}
        if normalized & NETWORK_SUBPROCESSES:
            raise RuntimeError(
                "external or credential subprocess is disabled during unit tests"
            )

    def guarded_run(*popenargs, **kwargs):
        deny_external_subprocess(command_tokens(popenargs, kwargs))
        return original_run(*popenargs, **kwargs)

    def guarded_popen(*popenargs, **kwargs):
        deny_external_subprocess(command_tokens(popenargs, kwargs))
        return original_popen(*popenargs, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", blocked_network)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked_network)
    monkeypatch.setattr(socket.socket, "sendto", blocked_network)
    monkeypatch.setattr(socket, "create_connection", blocked_network)
    monkeypatch.setattr(socket, "getaddrinfo", blocked_network)
    monkeypatch.setattr(socket, "gethostbyaddr", blocked_network)
    monkeypatch.setattr(socket, "gethostbyname", blocked_network)
    monkeypatch.setattr(socket, "gethostbyname_ex", blocked_network)
    monkeypatch.setattr(subprocess, "Popen", guarded_popen)
    monkeypatch.setattr(subprocess, "run", guarded_run)


@pytest.fixture(autouse=True)
def explicit_test_publication_authority(
    request: pytest.FixtureRequest,
    tmp_path: Path,
):
    """Exercise production publication checks without granting live authority."""

    if request.node.path.name in {
        "test_phase00_effect_authority.py",
        "test_phase00_scheduler_supervisor.py",
    }:
        with nullcontext():
            yield
        return
    authority = EffectAuthority(
        root=tmp_path.parent,
        secret=b"pytest-publication-authority-secret-32-bytes-minimum",
        issuer_id="pytest_supervisor",
        profile=PHASE00_REPAIR_PROFILE,
    )
    with publication_authority_session(
        authority=authority,
        run_id=f"pytest_{sha256(request.node.nodeid.encode()).hexdigest()[:20]}",
        intended_slot_id="pytest_isolated_slot",
        policy_version="phase00.pytest.v1",
        source_fingerprint_sha256=TEST_FINGERPRINT,
        runtime_fingerprint_sha256=TEST_FINGERPRINT,
        configuration_fingerprint_sha256=TEST_FINGERPRINT,
        allowed_scopes=frozenset(SCOPED_TARGET_PREFIXES),
        allowed_target_prefixes=(tmp_path,),
        max_total_bytes=256 * 1024**2,
    ):
        yield


@pytest.fixture(autouse=True)
def explicit_hyperliquid_keychain_authority(
    request: pytest.FixtureRequest,
    tmp_path: Path,
):
    """Give selected adapter tests explicit credential/public-network authority."""

    test_identity = (request.node.path.name, request.node.name)
    if (
        request.node.path.name not in KEYCHAIN_AUTHORITY_TEST_FILES
        and test_identity not in KEYCHAIN_AUTHORITY_TEST_NODES
    ):
        yield
        return
    authority = EffectAuthority(
        root=tmp_path,
        secret=b"pytest-keychain-authority-secret-32-bytes-minimum",
        issuer_id="pytest_keychain_supervisor",
        profile=RESEARCH_EXTERNAL_EFFECT_PROFILE,
    )
    material = f"{request.node.nodeid}:{tmp_path}".encode("utf-8")
    with external_effect_authority_session(
        authority=authority,
        run_id=f"pytest-keychain-{sha256(request.node.nodeid.encode()).hexdigest()[:20]}",
        intended_slot_id="pytest-keychain-isolated-slot",
        source_fingerprint_sha256=TEST_FINGERPRINT,
        runtime_fingerprint_sha256=TEST_FINGERPRINT,
        configuration_fingerprint_sha256=TEST_FINGERPRINT,
        provider_id="hyperliquid_adapter_test",
        account_scope_id="hyperliquid:test:injected_reader",
        reservation_id=f"pytest-keychain-{sha256(material).hexdigest()[:20]}",
        reservation_sha256=sha256(material).hexdigest(),
        allowed_targets=frozenset(
            {
                "https://api.hyperliquid.xyz/info",
                "https://api.hyperliquid-testnet.xyz/info",
            }
        ),
        allowed_credential_keys=frozenset(
            {
                "HYPERLIQUID_LIVE_AGENT_KEYCHAIN",
                "HYPERLIQUID_TESTNET_AGENT_KEYCHAIN",
            }
        ),
        max_total_requests=128,
        max_total_credits=0,
    ):
        yield
