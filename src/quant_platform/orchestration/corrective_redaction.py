"""Bounded, deterministic redaction for Phase 00 evidence surfaces.

The helpers in this module are deliberately pure. They do not inspect the
environment, open files, use credentials, or perform network operations.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from hashlib import sha256
from itertools import islice
from typing import Any, Final

EXCEPTION_TOKEN_SCHEMA_VERSION: Final = "thewiz.safe_exception_token.v1"

DEFAULT_MAX_DEPTH: Final = 8
DEFAULT_MAX_STRING_CHARS: Final = 2_048
DEFAULT_MAX_ITEMS: Final = 128

HARD_MAX_DEPTH: Final = 32
HARD_MAX_STRING_CHARS: Final = 65_536
HARD_MAX_ITEMS: Final = 4_096

_MAX_TYPE_CHARS: Final = 160
_MAX_EXCEPTION_ARGS: Final = 32
_MAX_MEASURED_MESSAGE_CHARS: Final = 1_000_000
_MAX_FINGERPRINT_CHARS: Final = 1_000_000
_STRING_SCAN_OVERLAP: Final = 4_096
_PAYLOAD_SCAN_CHARS: Final = 64 * 1024
_MAPPING_KEY_CHARS: Final = 256
_EVIDENCE_REASON_CODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:,=-]{0,255}")
_SAFE_STANDARD_EXCEPTION_CODES: Final = frozenset(
    {
        "active_scheduler_lock_present",
    }
)

_TYPE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_PEM_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]{0,48}PRIVATE KEY(?: BLOCK)?-----"
    r".*?(?:-----END [A-Z0-9 ]{0,48}PRIVATE KEY(?: BLOCK)?-----|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)
_URL_CREDENTIAL_RE = re.compile(
    r"(?P<scheme>https?://)[^/\s:@]+:[^/\s@]+@",
    flags=re.IGNORECASE,
)
_QUERY_SECRET_NAME = (
    r"api[_-]?key|api[_-]?secret|access[_-]?token|auth(?:entication|orization)?|"
    r"bearer|client[_-]?secret|private[_-]?key|password|passwd|passphrase|"
    r"secret|signature|sig|token"
)
_QUERY_SECRET_RE = re.compile(
    rf"(?P<prefix>[?&](?:{_QUERY_SECRET_NAME})=)[^&#\s]*",
    flags=re.IGNORECASE,
)
_BEARER_TOKEN_RE = re.compile(
    r"\bBearer[ \t]+[A-Za-z0-9._~+/=-]{8,}",
    flags=re.IGNORECASE,
)
_SECRET_FIELD_NAME = (
    r"[A-Z0-9_-]*(?:API[_-]?(?:KEY|SECRET)|ACCESS[_-]?TOKEN|"
    r"AUTH(?:ENTICATION|ORIZATION)?|BEARER[_-]?TOKEN|CLIENT[_-]?SECRET|"
    r"PRIVATE[_-]?KEY|PASSWORD|PASSWD|PASSPHRASE|SECRET|SIGNATURE|TOKEN)"
    r"[A-Z0-9_-]*"
)
_QUOTED_SECRET_ASSIGNMENT_RE = re.compile(
    rf"(?P<prefix>(?<![A-Za-z0-9_-])[\"']?{_SECRET_FIELD_NAME}[\"']?"
    r"\s*[:=]\s*)(?P<quote>[\"'])(?:[^\r\n]*?)(?P=quote)",
    flags=re.IGNORECASE,
)
_UNQUOTED_SECRET_ASSIGNMENT_RE = re.compile(
    rf"(?P<prefix>(?<![A-Za-z0-9_-]){_SECRET_FIELD_NAME}\s*[:=]\s*)"
    r"(?!<redacted:)[^\s,;}\]]+",
    flags=re.IGNORECASE,
)
_KNOWN_SECRET_PATTERNS = (
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{12,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", flags=re.IGNORECASE),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", flags=re.IGNORECASE),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{12,}\b", flags=re.IGNORECASE),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b0x[0-9a-fA-F]{64}\b"),
)
_HIGH_CONFIDENCE_SECRET_PATTERNS = (
    ("pem_private_key", _PEM_PRIVATE_KEY_RE),
    ("url_credentials", _URL_CREDENTIAL_RE),
    ("bearer_token", _BEARER_TOKEN_RE),
    ("openai_key", _KNOWN_SECRET_PATTERNS[0]),
    ("stripe_key", _KNOWN_SECRET_PATTERNS[1]),
    ("github_pat", _KNOWN_SECRET_PATTERNS[2]),
    ("github_token", _KNOWN_SECRET_PATTERNS[3]),
    ("aws_access_key", _KNOWN_SECRET_PATTERNS[4]),
    ("google_api_key", _KNOWN_SECRET_PATTERNS[5]),
    ("slack_token", _KNOWN_SECRET_PATTERNS[6]),
    ("jwt", _KNOWN_SECRET_PATTERNS[7]),
)
_SECRET_FIELD_SUFFIXES = (
    "apikey",
    "apisecret",
    "accesstoken",
    "authtoken",
    "authentication",
    "authorization",
    "bearertoken",
    "clientsecret",
    "privatekey",
    "password",
    "passwd",
    "passphrase",
    "secret",
    "signature",
    "token",
)


def safe_exception_token(exc: BaseException) -> dict[str, Any]:
    """Return stable exception evidence without retaining its raw message.

    The fingerprint is derived from the exception type and a bounded prefix of
    the rendered message plus its total length. No message or argument value is
    returned. Metadata has a fixed field count and bounded strings and counts.
    """

    if not isinstance(exc, BaseException):
        raise TypeError("safe_exception_token requires a BaseException")

    exception_type = _safe_type_name(type(exc))
    try:
        message = str(exc)
        render_error_type = ""
        message_rendered = True
    except BaseException as render_error:  # noqa: BLE001 - hostile __str__ boundary
        message = ""
        render_error_type = _safe_type_name(type(render_error))
        message_rendered = False

    digest = sha256()
    _update_digest(digest, EXCEPTION_TOKEN_SCHEMA_VERSION)
    _update_digest(digest, exception_type)
    _update_digest(digest, render_error_type)
    _update_bounded_text_digest(digest, message)

    argument_count = len(exc.args) if isinstance(exc.args, tuple) else 0
    cause = exc.__cause__
    context = exc.__context__
    message_chars = len(message)
    return {
        "schema_version": EXCEPTION_TOKEN_SCHEMA_VERSION,
        "exception_type": exception_type,
        "fingerprint_sha256": digest.hexdigest(),
        "metadata": {
            "message_rendered": message_rendered,
            "message_chars": min(message_chars, _MAX_MEASURED_MESSAGE_CHARS),
            "message_chars_capped": message_chars > _MAX_MEASURED_MESSAGE_CHARS,
            "argument_count": min(argument_count, _MAX_EXCEPTION_ARGS),
            "argument_count_capped": argument_count > _MAX_EXCEPTION_ARGS,
            "cause_type": _safe_type_name(type(cause)) if cause is not None else "",
            "context_type": (_safe_type_name(type(context)) if context is not None else ""),
            "render_error_type": render_error_type,
        },
    }


def safe_exception_code(exc: BaseException) -> str:
    """Return a compact diagnostic code without copying arbitrary messages.

    Project-owned exceptions may expose a validated ``evidence_reason_code``.
    That opt-in is trusted only for classes in the ``quant_platform`` package.
    Standard exceptions retain only a finite allowlist of exact internal codes;
    every other exception is represented by its short type and fingerprint.
    """

    token = safe_exception_token(exc)
    exception_type = str(token["exception_type"])
    short_type = exception_type.rsplit(".", 1)[-1]
    exception_module = str(getattr(type(exc), "__module__", ""))
    reason_code = getattr(exc, "evidence_reason_code", None)
    if (
        exception_module == "quant_platform"
        or exception_module.startswith("quant_platform.")
    ) and isinstance(reason_code, str) and _EVIDENCE_REASON_CODE_RE.fullmatch(reason_code):
        if getattr(exc, "evidence_reason_only", False) is True:
            return reason_code
        return f"{short_type}:{reason_code}"

    rendered = (
        exc.args[0]
        if isinstance(exc.args, tuple)
        and len(exc.args) == 1
        and isinstance(exc.args[0], str)
        else ""
    )
    if rendered in _SAFE_STANDARD_EXCEPTION_CODES:
        return rendered
    return f"{short_type}:{token['fingerprint_sha256'][:16]}"


def safe_validation_exception_code(exc: BaseException) -> str:
    """Retain bounded, redacted context for an audited local validator error.

    Callers must use this only around pure project-owned validation code. The
    helper accepts the narrow validation exception family, reads only one
    string argument, redacts it, collapses control whitespace, and otherwise
    falls back to the opaque exception code.
    """

    if not isinstance(exc, (FileNotFoundError, KeyError, TypeError, ValueError)):
        return safe_exception_code(exc)
    rendered = (
        exc.args[0]
        if isinstance(exc.args, tuple)
        and len(exc.args) == 1
        and isinstance(exc.args[0], str)
        else ""
    )
    if not rendered:
        return safe_exception_code(exc)
    redacted = _redact_string(rendered, max_chars=512)
    normalized = re.sub(r"\s+", " ", redacted).strip()
    if not normalized:
        return safe_exception_code(exc)
    return f"{type(exc).__name__}:{normalized}"


def evidence_payload_secret_codes(payload: bytes | str) -> tuple[str, ...]:
    """Return high-confidence secret classes found in a bounded streaming scan."""

    if isinstance(payload, bytes):
        text = payload.decode("utf-8", errors="ignore")
    elif isinstance(payload, str):
        text = payload
    else:
        raise TypeError("evidence payload must be bytes or str")

    matches: set[str] = set()
    carry = ""
    for offset in range(0, len(text), _PAYLOAD_SCAN_CHARS):
        chunk = carry + text[offset : offset + _PAYLOAD_SCAN_CHARS]
        for code, pattern in _HIGH_CONFIDENCE_SECRET_PATTERNS:
            if pattern.search(chunk):
                matches.add(code)
        for match in _QUERY_SECRET_RE.finditer(chunk):
            if not match.group(0).endswith("="):
                matches.add("query_secret")
        carry = chunk[-_STRING_SCAN_OVERLAP:]
    return tuple(sorted(matches))


def redact_for_evidence(
    value: Any,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_string_chars: int = DEFAULT_MAX_STRING_CHARS,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> Any:
    """Return a bounded, JSON-compatible, recursively redacted value.

    Mappings and sequences are capped at ``max_items`` output members. Nested
    containers beyond ``max_depth`` become diagnostic markers. Strings are
    redacted before they are truncated so a secret crossing the output boundary
    cannot be copied into evidence.
    """

    _validate_limit("max_depth", max_depth, minimum=0, maximum=HARD_MAX_DEPTH)
    _validate_limit(
        "max_string_chars",
        max_string_chars,
        minimum=32,
        maximum=HARD_MAX_STRING_CHARS,
    )
    _validate_limit("max_items", max_items, minimum=1, maximum=HARD_MAX_ITEMS)
    redactor = _EvidenceRedactor(
        max_depth=max_depth,
        max_string_chars=max_string_chars,
        max_items=max_items,
    )
    return redactor.redact(value, depth=0)


class _EvidenceRedactor:
    def __init__(self, *, max_depth: int, max_string_chars: int, max_items: int):
        self.max_depth = max_depth
        self.max_string_chars = max_string_chars
        self.max_items = max_items
        self._ancestors: set[int] = set()

    def redact(self, value: Any, *, depth: int) -> Any:
        if isinstance(value, str):
            return _redact_string(value, max_chars=self.max_string_chars)
        if isinstance(value, BaseException):
            return safe_exception_token(value)
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            if math.isfinite(value):
                return value
            return _diagnostic("non_finite_number", numeric_type=type(value).__name__)
        if isinstance(value, bytes):
            return {
                "__redaction__": "binary_value",
                "length": len(value),
                "sha256": sha256(value).hexdigest(),
            }
        if isinstance(value, Mapping):
            return self._redact_mapping(value, depth=depth)
        if isinstance(value, Sequence):
            return self._redact_sequence(value, depth=depth)
        return _diagnostic("unsupported_type", value_type=_safe_type_name(type(value)))

    def _redact_mapping(self, value: Mapping[Any, Any], *, depth: int) -> Any:
        if depth >= self.max_depth:
            return _diagnostic("max_depth_exceeded", depth=depth)
        identity = id(value)
        if identity in self._ancestors:
            return _diagnostic("cycle_detected", depth=depth)

        self._ancestors.add(identity)
        try:
            sampled = list(islice(value.items(), self.max_items + 1))
            truncated = len(sampled) > self.max_items
            process_count = self.max_items - 1 if truncated else len(sampled)
            output: dict[str, Any] = {}
            for key, child in sampled[:process_count]:
                safe_key = _safe_mapping_key(key, max_chars=self.max_string_chars)
                safe_key = _unique_mapping_key(output, safe_key)
                if isinstance(key, str) and _is_secret_field_name(key):
                    output[safe_key] = "<redacted:secret_value>"
                else:
                    output[safe_key] = self.redact(child, depth=depth + 1)
            if truncated:
                marker_key = _unique_mapping_key(output, "__redaction_diagnostic__")
                output[marker_key] = _diagnostic(
                    "items_truncated",
                    observed_at_least=len(sampled),
                    retained=process_count,
                )
            return output
        finally:
            self._ancestors.remove(identity)

    def _redact_sequence(self, value: Sequence[Any], *, depth: int) -> Any:
        if depth >= self.max_depth:
            return _diagnostic("max_depth_exceeded", depth=depth)
        identity = id(value)
        if identity in self._ancestors:
            return _diagnostic("cycle_detected", depth=depth)

        self._ancestors.add(identity)
        try:
            sampled = list(islice(iter(value), self.max_items + 1))
            truncated = len(sampled) > self.max_items
            process_count = self.max_items - 1 if truncated else len(sampled)
            output = [self.redact(child, depth=depth + 1) for child in sampled[:process_count]]
            if truncated:
                output.append(
                    _diagnostic(
                        "items_truncated",
                        observed_at_least=len(sampled),
                        retained=process_count,
                    )
                )
            return output
        finally:
            self._ancestors.remove(identity)


def _redact_string(value: str, *, max_chars: int) -> str:
    scan_limit = min(len(value), max_chars + _STRING_SCAN_OVERLAP)
    candidate = value[:scan_limit]
    candidate = _PEM_PRIVATE_KEY_RE.sub("<redacted:private_key>", candidate)
    candidate = _URL_CREDENTIAL_RE.sub(
        lambda match: f"{match.group('scheme')}<redacted:url_credentials>@",
        candidate,
    )
    candidate = _QUERY_SECRET_RE.sub(
        lambda match: f"{match.group('prefix')}<redacted:query_secret>",
        candidate,
    )
    candidate = _BEARER_TOKEN_RE.sub("Bearer <redacted:bearer_token>", candidate)
    candidate = _QUOTED_SECRET_ASSIGNMENT_RE.sub(
        lambda match: (
            f"{match.group('prefix')}{match.group('quote')}<redacted:secret>{match.group('quote')}"
        ),
        candidate,
    )
    candidate = _UNQUOTED_SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group('prefix')}<redacted:secret>",
        candidate,
    )
    for pattern in _KNOWN_SECRET_PATTERNS:
        candidate = pattern.sub("<redacted:credential>", candidate)

    if len(value) <= max_chars and len(candidate) <= max_chars:
        return candidate
    marker = _truncation_marker(value, max_chars=max_chars)
    retained_chars = max(0, max_chars - len(marker))
    return candidate[:retained_chars] + marker


def _truncation_marker(value: str, *, max_chars: int) -> str:
    digest = sha256()
    _update_bounded_text_digest(digest, value)
    marker = f"<truncated:{len(value)}:{digest.hexdigest()[:16]}>"
    if len(marker) <= max_chars:
        return marker
    return "<truncated>"[:max_chars]


def _safe_mapping_key(value: Any, *, max_chars: int) -> str:
    key_limit = min(max_chars, _MAPPING_KEY_CHARS)
    if isinstance(value, str):
        return _redact_string(value, max_chars=key_limit)
    if value is None or isinstance(value, (bool, int)):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return str(value)
    return f"<unsupported-key:{_safe_type_name(type(value))}>"


def _unique_mapping_key(output: Mapping[str, Any], candidate: str) -> str:
    if candidate not in output:
        return candidate
    suffix = 2
    while f"{candidate}#{suffix}" in output:
        suffix += 1
    return f"{candidate}#{suffix}"


def _is_secret_field_name(value: str) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", value.lower())
    return any(compact.endswith(suffix) for suffix in _SECRET_FIELD_SUFFIXES)


def _safe_type_name(value: type[Any]) -> str:
    raw = f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', '')}"
    normalized = _TYPE_COMPONENT_RE.sub("_", raw).strip("._") or "unknown"
    if len(normalized) <= _MAX_TYPE_CHARS:
        return normalized
    digest = sha256(normalized.encode("utf-8", errors="surrogatepass")).hexdigest()[:12]
    retained = _MAX_TYPE_CHARS - len(digest) - 1
    return f"{normalized[:retained]}_{digest}"


def _diagnostic(code: str, **metadata: Any) -> dict[str, Any]:
    return {"__redaction__": code, **metadata}


def _validate_limit(name: str, value: int, *, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _update_digest(digest: Any, value: str) -> None:
    digest.update(value.encode("utf-8", errors="surrogatepass"))
    digest.update(b"\x00")


def _update_bounded_text_digest(digest: Any, value: str) -> None:
    retained = value[:_MAX_FINGERPRINT_CHARS]
    for offset in range(0, len(retained), 8_192):
        digest.update(
            retained[offset : offset + 8_192].encode(
                "utf-8",
                errors="surrogatepass",
            )
        )
    digest.update(b"\x00length=")
    digest.update(str(len(value)).encode("ascii"))
    digest.update(b"\x00")


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_ITEMS",
    "DEFAULT_MAX_STRING_CHARS",
    "EXCEPTION_TOKEN_SCHEMA_VERSION",
    "evidence_payload_secret_codes",
    "redact_for_evidence",
    "safe_exception_code",
    "safe_exception_token",
]
