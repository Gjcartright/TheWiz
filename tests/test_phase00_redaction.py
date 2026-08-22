from __future__ import annotations

import ast
import json
import socket
import subprocess
from pathlib import Path

import pytest

from quant_platform.orchestration.corrective_redaction import (
    EXCEPTION_TOKEN_SCHEMA_VERSION,
    evidence_payload_secret_codes,
    redact_for_evidence,
    safe_exception_code,
    safe_exception_token,
    safe_validation_exception_code,
)

API_CANARY = "sk-proj-phase00ApiCanary123456789"
BEARER_CANARY = "eyJphase00Header.eyJphase00Payload.phase00Signature"
PASSWORD_CANARY = "phase00-url-password-canary"
QUERY_CANARY = "phase00-query-token-canary"
ENV_CANARY = "phase00-env-secret-canary"
PRIVATE_KEY_CANARY = "phase00PrivateKeyMaterialCanary"
RAW_PRIVATE_KEY_CANARY = "0x" + "a1" * 32
GITHUB_CANARY = "ghp_phase00GithubCanary1234567890"


def _encoded(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _assert_no_canaries(value: object) -> None:
    encoded = _encoded(value)
    for canary in (
        API_CANARY,
        BEARER_CANARY,
        PASSWORD_CANARY,
        QUERY_CANARY,
        ENV_CANARY,
        PRIVATE_KEY_CANARY,
        RAW_PRIVATE_KEY_CANARY,
        GITHUB_CANARY,
    ):
        assert canary not in encoded


def test_safe_exception_token_is_stable_bounded_and_never_stores_message() -> None:
    first = safe_exception_token(ValueError(f"provider failed with {API_CANARY}"))
    second = safe_exception_token(ValueError(f"provider failed with {API_CANARY}"))
    different = safe_exception_token(ValueError("different failure"))

    assert first == second
    assert first["schema_version"] == EXCEPTION_TOKEN_SCHEMA_VERSION
    assert first["exception_type"] == "builtins.ValueError"
    assert len(first["fingerprint_sha256"]) == 64
    assert first["fingerprint_sha256"] != different["fingerprint_sha256"]
    assert set(first) == {
        "schema_version",
        "exception_type",
        "fingerprint_sha256",
        "metadata",
    }
    assert len(first["metadata"]) == 8
    assert API_CANARY not in _encoded(first)
    assert "provider failed" not in _encoded(first)

    code = safe_exception_code(ValueError(f"provider failed with {API_CANARY}"))
    assert code.startswith("ValueError:")
    assert API_CANARY not in code
    assert "provider failed" not in code


def test_local_validation_code_retains_context_but_redacts_secrets() -> None:
    code = safe_validation_exception_code(
        ValueError(f"source_bindings invalid; api_key={API_CANARY}")
    )

    assert code.startswith("ValueError:source_bindings invalid")
    assert API_CANARY not in code
    assert "<redacted:secret>" in code


def test_high_confidence_secret_scan_is_streaming_and_avoids_ambiguous_hashes() -> None:
    split_payload = "x" * (64 * 1024 - 6) + " " + API_CANARY

    assert evidence_payload_secret_codes(split_payload) == ("openai_key",)
    assert evidence_payload_secret_codes(
        f"https://alice:{PASSWORD_CANARY}@example.test"
    ) == ("url_credentials",)
    assert evidence_payload_secret_codes(
        f"https://example.test?access_token={QUERY_CANARY}"
    ) == ("query_secret",)
    assert evidence_payload_secret_codes(
        '{"api_key_present": false, "credential_status": "MISSING"}'
    ) == ()
    assert evidence_payload_secret_codes(RAW_PRIVATE_KEY_CANARY) == ()


def test_recursive_redaction_removes_all_common_secret_forms() -> None:
    payload = {
        "api_key": API_CANARY,
        "headers": {"Authorization": f"Bearer {BEARER_CANARY}"},
        "connection": (
            f"https://alice:{PASSWORD_CANARY}@example.test/path"
            f"?safe=visible&access_token={QUERY_CANARY}"
        ),
        "message": (
            f"CRYPTO_WIZARDS_API_KEY={ENV_CANARY}\n"
            f"token={GITHUB_CANARY}\n"
            f"wallet={RAW_PRIVATE_KEY_CANARY}"
        ),
        "private_material": (
            "-----BEGIN PRIVATE KEY-----\n"
            f"{PRIVATE_KEY_CANARY}\n"
            "-----END PRIVATE KEY-----"
        ),
        "safe": ["ordinary text", 7, True, None],
    }

    redacted = redact_for_evidence(payload)

    _assert_no_canaries(redacted)
    assert redacted["api_key"] == "<redacted:secret_value>"
    assert redacted["safe"] == ["ordinary text", 7, True, None]
    assert "safe=visible" in redacted["connection"]
    assert "<redacted:url_credentials>" in redacted["connection"]
    assert "<redacted:query_secret>" in redacted["connection"]
    assert "<redacted:private_key>" in redacted["private_material"]


def test_redaction_is_deterministic_and_json_compatible() -> None:
    payload = {
        "nested": [
            {"password": PASSWORD_CANARY},
            f"X-API-Key: {API_CANARY}",
        ],
        "error": RuntimeError(f"runtime {ENV_CANARY}"),
        "binary": b"not-published-directly",
    }

    first = redact_for_evidence(payload)
    second = redact_for_evidence(payload)

    assert first == second
    assert _encoded(first) == _encoded(second)
    _assert_no_canaries(first)
    assert first["error"]["exception_type"] == "builtins.RuntimeError"
    assert first["binary"]["__redaction__"] == "binary_value"


def test_depth_string_item_and_cycle_diagnostics_are_bounded() -> None:
    cycle: list[object] = []
    cycle.append(cycle)
    payload = {
        "deep": {"one": {"two": {"three": API_CANARY}}},
        "long": f"prefix API_KEY={API_CANARY} " + "x" * 1_000,
        "many": list(range(20)),
        "cycle": cycle,
    }

    redacted = redact_for_evidence(
        payload,
        max_depth=3,
        max_string_chars=96,
        max_items=4,
    )

    _assert_no_canaries(redacted)
    assert len(redacted) <= 4
    assert len(redacted["long"]) <= 96
    assert "<truncated:" in redacted["long"]
    assert len(redacted["many"]) <= 4
    assert redacted["many"][-1]["__redaction__"] == "items_truncated"
    assert redacted["cycle"][0]["__redaction__"] == "cycle_detected"
    assert redacted["deep"]["one"]["two"]["__redaction__"] == (
        "max_depth_exceeded"
    )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_depth": -1}, ValueError),
        ({"max_depth": True}, TypeError),
        ({"max_string_chars": 31}, ValueError),
        ({"max_items": 0}, ValueError),
        ({"max_items": 4_097}, ValueError),
    ],
)
def test_limits_fail_closed(kwargs: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        redact_for_evidence({}, **kwargs)


def test_redaction_has_no_network_or_subprocess_effects(monkeypatch) -> None:
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("external")
        raise AssertionError("redaction attempted an external effect")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)

    result = redact_for_evidence(
        {
            "url": f"https://user:{PASSWORD_CANARY}@example.test?token={QUERY_CANARY}",
            "error": OSError(f"network failed with {API_CANARY}"),
        }
    )

    assert calls == []
    _assert_no_canaries(result)


def test_orchestration_has_no_unreviewed_raw_exception_string_surfaces() -> None:
    source_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "quant_platform"
        / "orchestration"
    )
    allowed_str_calls = {
        ("corrective_daily_scheduler.py", "_validate_daily_scheduler_lineage"),
        ("corrective_lineage.py", "_strict_json_loads"),
        ("corrective_redaction.py", "safe_exception_token"),
        ("corrective_scheduler_lock.py", "_require_publication_authority"),
        (
            "corrective_scheduler_runtime_readiness.py",
            "_terminal_receipt_observation",
        ),
        ("corrective_wizard_proof_launcher.py", "_write_or_validate_immutable_json"),
    }
    violations: list[str] = []

    for path in sorted(source_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        class Visitor(ast.NodeVisitor):
            def __init__(self, path_name: str) -> None:
                self.path_name = path_name
                self.function_stack: list[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.function_stack.append(node.name)
                self.generic_visit(node)
                self.function_stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
                if any(
                    isinstance(value, ast.FormattedValue)
                    and isinstance(value.value, ast.Name)
                    and value.value.id == "exc"
                    for value in node.values
                ):
                    violations.append(
                        f"{self.path_name}:{node.lineno}:raw_fstring_exc"
                    )
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                raw_exc = (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "str"
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "exc"
                )
                function_name = self.function_stack[-1] if self.function_stack else ""
                if raw_exc and (self.path_name, function_name) not in allowed_str_calls:
                    violations.append(
                        f"{self.path_name}:{node.lineno}:raw_str_exc"
                    )
                self.generic_visit(node)

        Visitor(path.name).visit(tree)

    assert violations == []
