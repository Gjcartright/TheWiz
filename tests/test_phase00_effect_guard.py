from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest

from quant_platform.env import load_selected_env_keys
from quant_platform.orchestration.corrective_effect_guard import (
    Phase00EffectDenied,
    _authorized_keychain_effect,
    phase00_effect_guard,
)


def test_guard_denies_socket_before_operating_system_connection() -> None:
    with phase00_effect_guard(), pytest.raises(
        Phase00EffectDenied,
        match="network_target_denied:127.0.0.1",
    ):
        socket.create_connection(("127.0.0.1", 1))


def test_guard_denies_network_and_keychain_subprocesses() -> None:
    with phase00_effect_guard():
        with pytest.raises(Phase00EffectDenied, match="network_subprocess_denied"):
            subprocess.Popen(["curl", "https://example.invalid"])
        with pytest.raises(Phase00EffectDenied, match="keychain_subprocess_denied"):
            subprocess.Popen(
                ["security", "find-generic-password", "-w", "-s", "x"]
            )


def test_guard_allows_only_exact_authorized_keychain_item(monkeypatch) -> None:
    observed = []

    def fake_popen(args, *popen_args, **popen_kwargs):
        observed.append((args, popen_args, popen_kwargs))
        return object()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    command = [
        "security",
        "find-generic-password",
        "-w",
        "-s",
        "thewiz-agent",
        "-a",
        "0xabc",
    ]
    with phase00_effect_guard(), _authorized_keychain_effect(
        service="thewiz-agent",
        account="0xabc",
    ):
        subprocess.Popen(command)
        with pytest.raises(Phase00EffectDenied, match="keychain_subprocess_denied"):
            subprocess.Popen([*command[:-1], "0xdef"])
        with pytest.raises(Phase00EffectDenied, match="keychain_subprocess_denied"):
            subprocess.Popen([*command, "--extra"])

    assert [entry[0] for entry in observed] == [command]


def test_guard_denies_python_subprocess_escape() -> None:
    with phase00_effect_guard(), pytest.raises(
        Phase00EffectDenied,
        match="subprocess_denied:python",
    ):
        subprocess.run(
            [sys.executable, "-c", "print('local-only')"],
            check=True,
            capture_output=True,
            text=True,
        )


def test_guard_denies_sensitive_environment_and_selected_env_loader(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CRYPTO_WIZARDS_API_KEY", "never-return-this")
    env_file = tmp_path / ".env"
    env_file.write_text("CRYPTO_WIZARDS_API_KEY=never-load-this\n", encoding="utf-8")
    with phase00_effect_guard():
        with pytest.raises(Phase00EffectDenied, match="sensitive_env_denied"):
            os.getenv("CRYPTO_WIZARDS_API_KEY")
        with pytest.raises(Phase00EffectDenied, match="credential_access_denied"):
            load_selected_env_keys(
                env_file,
                allowed_keys={"CRYPTO_WIZARDS_API_KEY"},
            )


def test_guard_denies_sensitive_direct_environment_mapping_reads(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CRYPTO_WIZARDS_API_KEY", "never-return-this")

    with phase00_effect_guard():
        with pytest.raises(Phase00EffectDenied, match="sensitive_env_denied"):
            _ = os.environ["CRYPTO_WIZARDS_API_KEY"]
        with pytest.raises(Phase00EffectDenied, match="sensitive_env_denied"):
            os.environ.get("CRYPTO_WIZARDS_API_KEY")


def test_guard_denies_direct_os_process_escape_functions() -> None:
    with phase00_effect_guard():
        with pytest.raises(Phase00EffectDenied, match="os_process_escape_denied"):
            os.system("exit 0")
        with pytest.raises(Phase00EffectDenied, match="os_process_escape_denied"):
            os.execv("/usr/bin/true", ["true"])
