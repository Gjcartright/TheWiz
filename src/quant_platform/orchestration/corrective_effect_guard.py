"""Dynamic deny boundary for legacy effects inside Phase 00 schedulers."""

from __future__ import annotations

import os
import socket
import subprocess
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Phase00EffectDenied(RuntimeError):
    """Raised before a prohibited Phase 00 effect reaches the operating system."""


_GUARD_ACTIVE: ContextVar[bool] = ContextVar("thewiz_phase00_effect_guard", default=False)


@dataclass(frozen=True)
class _EffectAllowance:
    credential_keys: frozenset[str] = frozenset()
    network_hosts: frozenset[str] = frozenset()
    keychain_items: frozenset[tuple[str, str]] = frozenset()


_EFFECT_ALLOWANCE: ContextVar[_EffectAllowance | None] = ContextVar(
    "thewiz_phase00_effect_allowance",
    default=None,
)
_NETWORK_TRANSPORT_ACTIVE: ContextVar[bool] = ContextVar(
    "thewiz_phase00_network_transport_active",
    default=False,
)
_SENSITIVE_ENV_TOKENS = (
    "API_KEY",
    "API_SECRET",
    "PRIVATE_KEY",
    "PASSWORD",
    "PASSPHRASE",
    "ACCESS_TOKEN",
    "AUTH_TOKEN",
    "CREDENTIAL",
)
_NETWORK_EXECUTABLES = {
    "curl",
    "ftp",
    "nc",
    "ncat",
    "netcat",
    "scp",
    "sftp",
    "ssh",
    "telnet",
    "wget",
}
_OS_PROCESS_ESCAPE_FUNCTIONS = (
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "fork",
    "forkpty",
    "popen",
    "posix_spawn",
    "posix_spawnp",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "system",
)


def phase00_effect_guard_active() -> bool:
    return _GUARD_ACTIVE.get()


def _current_allowance() -> _EffectAllowance:
    return _EFFECT_ALLOWANCE.get() or _EffectAllowance()


def assert_credential_access_allowed(*, source: str) -> None:
    """Explicit credential boundary used by project-owned loaders/readers."""

    allowance = _current_allowance()
    if phase00_effect_guard_active() and source.upper() not in allowance.credential_keys:
        raise Phase00EffectDenied(f"phase00_credential_access_denied:{source}")


@contextmanager
def _authorized_credential_effect(*, key: str) -> Iterator[None]:
    """Open one narrow credential window after a permit has been consumed."""

    normalized = key.strip().upper()
    if not normalized:
        raise ValueError("credential key cannot be blank")
    current = _current_allowance()
    token = _EFFECT_ALLOWANCE.set(
        _EffectAllowance(
            credential_keys=current.credential_keys | {normalized},
            network_hosts=current.network_hosts,
            keychain_items=current.keychain_items,
        )
    )
    try:
        yield
    finally:
        _EFFECT_ALLOWANCE.reset(token)


@contextmanager
def _authorized_network_effect(*, hosts: frozenset[str]) -> Iterator[None]:
    """Open one narrow socket window after network and credit permits are consumed."""

    normalized = frozenset(host.strip().lower().rstrip(".") for host in hosts if host.strip())
    if not normalized:
        raise ValueError("at least one network host is required")
    current = _current_allowance()
    token = _EFFECT_ALLOWANCE.set(
        _EffectAllowance(
            credential_keys=current.credential_keys,
            network_hosts=current.network_hosts | normalized,
            keychain_items=current.keychain_items,
        )
    )
    try:
        yield
    finally:
        _EFFECT_ALLOWANCE.reset(token)


@contextmanager
def _authorized_keychain_effect(*, service: str, account: str) -> Iterator[None]:
    """Allow one exact macOS Keychain item after credential permit consumption."""

    normalized_service = service.strip()
    normalized_account = account.strip()
    if not normalized_service or not normalized_account:
        raise ValueError("keychain service and account are required")
    current = _current_allowance()
    token = _EFFECT_ALLOWANCE.set(
        _EffectAllowance(
            credential_keys=current.credential_keys,
            network_hosts=current.network_hosts,
            keychain_items=current.keychain_items
            | {(normalized_service, normalized_account)},
        )
    )
    try:
        yield
    finally:
        _EFFECT_ALLOWANCE.reset(token)


@contextmanager
def phase00_effect_guard() -> Iterator[None]:
    """Deny external and credential effects while allowing local computation."""

    if phase00_effect_guard_active():
        raise Phase00EffectDenied("nested_phase00_effect_guard_denied")
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection
    original_getenv = os.getenv
    original_popen = subprocess.Popen
    environment_type = type(os.environ)
    original_environ_getitem = environment_type.__getitem__
    original_os_process_functions = {
        name: getattr(os, name)
        for name in _OS_PROCESS_ESCAPE_FUNCTIONS
        if hasattr(os, name)
    }

    def guarded_connect(sock: socket.socket, address: Any) -> Any:
        if not _NETWORK_TRANSPORT_ACTIVE.get():
            _assert_network_address_allowed(address)
        return original_connect(sock, address)

    def guarded_connect_ex(sock: socket.socket, address: Any) -> Any:
        if not _NETWORK_TRANSPORT_ACTIVE.get():
            _assert_network_address_allowed(address)
        return original_connect_ex(sock, address)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        _assert_network_address_allowed(address)
        transport_token = _NETWORK_TRANSPORT_ACTIVE.set(True)
        try:
            return original_create_connection(address, *args, **kwargs)
        finally:
            _NETWORK_TRANSPORT_ACTIVE.reset(transport_token)

    def guarded_getenv(key: str, default: str | None = None) -> str | None:
        _assert_environment_key_allowed(key)
        return original_getenv(key, default)

    def guarded_environ_getitem(environ: Any, key: Any) -> Any:
        _assert_environment_key_allowed(key)
        return original_environ_getitem(environ, key)

    def guarded_popen(
        args: str | Sequence[str | os.PathLike[str]],
        *popen_args: Any,
        **popen_kwargs: Any,
    ) -> subprocess.Popen[Any]:
        _validate_subprocess(args)
        return original_popen(args, *popen_args, **popen_kwargs)

    def denied_os_process(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise Phase00EffectDenied("phase00_os_process_escape_denied")

    token = _GUARD_ACTIVE.set(True)
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.create_connection = guarded_create_connection
    os.getenv = guarded_getenv
    environment_type.__getitem__ = guarded_environ_getitem
    subprocess.Popen = guarded_popen
    for name in original_os_process_functions:
        setattr(os, name, denied_os_process)
    try:
        yield
    finally:
        for name, function in original_os_process_functions.items():
            setattr(os, name, function)
        subprocess.Popen = original_popen
        environment_type.__getitem__ = original_environ_getitem
        os.getenv = original_getenv
        socket.create_connection = original_create_connection
        socket.socket.connect_ex = original_connect_ex
        socket.socket.connect = original_connect
        _GUARD_ACTIVE.reset(token)


def _assert_environment_key_allowed(key: Any) -> None:
    if isinstance(key, bytes):
        normalized = key.decode(errors="replace").upper()
    else:
        normalized = str(key).upper()
    if (
        any(token in normalized for token in _SENSITIVE_ENV_TOKENS)
        and normalized not in _current_allowance().credential_keys
    ):
        raise Phase00EffectDenied(f"phase00_sensitive_env_denied:{normalized}")


def _assert_network_address_allowed(address: Any) -> None:
    if isinstance(address, tuple) and address:
        host = str(address[0]).strip().lower().rstrip(".")
    else:
        host = str(address).strip().lower().rstrip(".")
    if not host or host not in _current_allowance().network_hosts:
        raise Phase00EffectDenied(f"phase00_network_target_denied:{host or 'unknown'}")


def _validate_subprocess(
    command: str | Sequence[str | os.PathLike[str]],
) -> None:
    if isinstance(command, str):
        tokens = command.split()
    else:
        tokens = [str(value) for value in command]
    if not tokens:
        raise Phase00EffectDenied("phase00_empty_subprocess_denied")
    executable = Path(tokens[0]).name.lower()
    if executable in _NETWORK_EXECUTABLES:
        raise Phase00EffectDenied(
            f"phase00_network_subprocess_denied:{executable}"
        )
    lowered = [token.lower() for token in tokens[1:]]
    if executable == "security" and any(
        token in {"find-generic-password", "find-internet-password"}
        for token in lowered
    ):
        if isinstance(command, str):
            raise Phase00EffectDenied("phase00_keychain_shell_command_denied")
        if len(tokens) == 7 and tokens[1:5] == [
            "find-generic-password",
            "-w",
            "-s",
            tokens[4],
        ] and tokens[5] == "-a":
            item = (tokens[4], tokens[6])
            if item in _current_allowance().keychain_items:
                return
        raise Phase00EffectDenied("phase00_keychain_subprocess_denied")
    if executable == "osascript":
        raise Phase00EffectDenied("phase00_osascript_subprocess_denied")
    raise Phase00EffectDenied(f"phase00_subprocess_denied:{executable}")
