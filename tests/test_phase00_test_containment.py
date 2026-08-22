from __future__ import annotations

import socket
import subprocess

import pytest


def test_unit_test_containment_denies_dns_and_socket_variants() -> None:
    with pytest.raises(OSError, match="external network access is disabled"):
        socket.getaddrinfo("example.com", 443)
    with socket.socket() as stream_socket, pytest.raises(
        OSError,
        match="external network access is disabled",
    ):
        stream_socket.connect_ex(("127.0.0.1", 1))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as datagram_socket, pytest.raises(
        OSError,
        match="external network access is disabled",
    ):
        datagram_socket.sendto(b"probe", ("127.0.0.1", 9))


def test_unit_test_containment_denies_keychain_and_network_popen() -> None:
    with pytest.raises(RuntimeError, match="credential subprocess is disabled"):
        subprocess.Popen(
            [
                "security",
                "find-generic-password",
                "-w",
                "-s",
                "forbidden",
                "-a",
                "forbidden",
            ]
        )
    with pytest.raises(RuntimeError, match="credential subprocess is disabled"):
        subprocess.Popen(["curl", "https://example.invalid"])
