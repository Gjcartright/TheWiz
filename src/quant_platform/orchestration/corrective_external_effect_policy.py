"""Validated, hash-bound policy for Phase 00 provider effects."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

EXTERNAL_EFFECT_POLICY_PATH = Path("config/phase00_external_effect_policy.json")
EXTERNAL_EFFECT_POLICY_SCHEMA_VERSION = "thewiz.phase00_external_effect_policy.v1"
MAX_EXTERNAL_EFFECT_POLICY_BYTES = 1024**2
REQUIRED_PROVIDER_IDS = frozenset(
    {
        "crypto_wizards",
        "apify",
        "hyperliquid_live_execution",
        "hyperliquid_live_preflight",
        "hyperliquid_public",
        "hyperliquid_testnet_execution",
        "hyperliquid_testnet_preflight",
    }
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "policy_id",
        "effective_from_utc",
        "review_required_at_utc",
        "order_submission_included",
        "providers",
    }
)
_PROVIDER_FIELDS = frozenset(
    {
        "account_scope_id",
        "allowed_targets",
        "allowed_credential_keys",
        "max_total_requests_per_run",
        "max_total_credits_per_run",
        "order_submission_included",
        "pricing_provenance",
        "endpoint_pricing",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "source_type",
        "source_locator",
        "captured_at_utc",
        "effective_from_utc",
        "evidence_status",
        "unit_definition",
    }
)
_PRICING_FIELDS = frozenset(
    {
        "operation",
        "method",
        "target",
        "credit_units",
        "pricing_basis",
    }
)


class ExternalEffectPolicyError(ValueError):
    """Stable fail-closed policy error that is safe to place in receipts."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class EndpointPricing:
    operation: str
    method: str
    target: str
    credit_units: int | None
    pricing_basis: str


@dataclass(frozen=True)
class PricingProvenance:
    source_type: str
    source_locator: str
    captured_at_utc: str
    effective_from_utc: str
    evidence_status: str
    unit_definition: str


@dataclass(frozen=True)
class ProviderExternalEffectPolicy:
    provider_id: str
    account_scope_id: str
    allowed_targets: frozenset[str]
    allowed_credential_keys: frozenset[str]
    max_total_requests_per_run: int
    max_total_credits_per_run: int
    pricing_provenance: PricingProvenance
    endpoint_pricing: tuple[EndpointPricing, ...]


@dataclass(frozen=True)
class Phase00ExternalEffectPolicy:
    path: Path
    policy_sha256: str
    schema_version: str
    policy_id: str
    effective_from_utc: str
    review_required_at_utc: str
    providers: tuple[ProviderExternalEffectPolicy, ...]

    @property
    def permit_policy_version(self) -> str:
        return f"{self.schema_version}:sha256:{self.policy_sha256}"

    def provider(self, provider_id: str) -> ProviderExternalEffectPolicy:
        normalized = _required_text(provider_id, "provider_id")
        for provider in self.providers:
            if provider.provider_id == normalized:
                return provider
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_provider_missing"
        )

    def receipt_fields(self, root: Path) -> dict[str, object]:
        return {
            "external_effect_policy_id": self.policy_id,
            "external_effect_policy_schema_version": self.schema_version,
            "external_effect_policy_path": self.path.relative_to(root).as_posix(),
            "external_effect_policy_sha256": self.policy_sha256,
            "external_effect_policy_effective_from_utc": self.effective_from_utc,
            "external_effect_policy_review_required_at_utc": (
                self.review_required_at_utc
            ),
        }


def load_phase00_external_effect_policy(
    root: Path,
    *,
    as_of: datetime | None = None,
) -> Phase00ExternalEffectPolicy:
    """Load and validate the exact provider policy owned by this repository."""

    path = root / EXTERNAL_EFFECT_POLICY_PATH
    raw = _read_policy_bytes(root, path)
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_invalid_json"
        ) from exc
    if not isinstance(payload, dict):
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_invalid_shape"
        )
    _require_exact_fields(
        payload,
        _TOP_LEVEL_FIELDS,
        "phase00_external_effect_policy_top_level_fields_invalid",
    )
    if payload["schema_version"] != EXTERNAL_EFFECT_POLICY_SCHEMA_VERSION:
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_schema_invalid"
        )
    policy_id = _required_text(payload["policy_id"], "policy_id")
    if policy_id != "phase00_external_effect_policy":
        raise ExternalEffectPolicyError("phase00_external_effect_policy_id_invalid")
    if payload["order_submission_included"] is not False:
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_order_capability_forbidden"
        )

    effective = _utc_timestamp(payload["effective_from_utc"], "effective_from_utc")
    review_required = _utc_timestamp(
        payload["review_required_at_utc"],
        "review_required_at_utc",
    )
    if review_required <= effective:
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_review_window_invalid"
        )
    observed = _as_utc(as_of or datetime.now(UTC))
    if observed < effective:
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_not_effective"
        )
    if observed >= review_required:
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_review_expired"
        )

    providers_payload = payload["providers"]
    if not isinstance(providers_payload, dict):
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_providers_invalid"
        )
    if frozenset(providers_payload) != REQUIRED_PROVIDER_IDS:
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_provider_set_invalid"
        )
    providers = tuple(
        _provider_policy(
            provider_id=provider_id,
            payload=providers_payload[provider_id],
            observed=observed,
        )
        for provider_id in sorted(providers_payload)
    )
    return Phase00ExternalEffectPolicy(
        path=path,
        policy_sha256=sha256(raw).hexdigest(),
        schema_version=EXTERNAL_EFFECT_POLICY_SCHEMA_VERSION,
        policy_id=policy_id,
        effective_from_utc=effective.isoformat(),
        review_required_at_utc=review_required.isoformat(),
        providers=providers,
    )


def _provider_policy(
    *,
    provider_id: str,
    payload: object,
    observed: datetime,
) -> ProviderExternalEffectPolicy:
    if not isinstance(payload, dict):
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_provider_invalid"
        )
    _require_exact_fields(
        payload,
        _PROVIDER_FIELDS,
        "phase00_external_effect_policy_provider_fields_invalid",
    )
    if payload["order_submission_included"] is not False:
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_provider_order_capability_forbidden"
        )
    account_scope_id = _required_text(payload["account_scope_id"], "account_scope_id")
    targets = _unique_text_values(
        payload["allowed_targets"],
        "phase00_external_effect_policy_targets_invalid",
    )
    normalized_targets = frozenset(_https_target(value) for value in targets)
    credential_values = _unique_text_values(
        payload["allowed_credential_keys"],
        "phase00_external_effect_policy_credentials_invalid",
        allow_empty=True,
    )
    if any(value != value.upper() for value in credential_values):
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_credentials_invalid"
        )
    credential_keys = frozenset(credential_values)
    max_requests = _nonnegative_integer(
        payload["max_total_requests_per_run"],
        "phase00_external_effect_policy_request_budget_invalid",
    )
    max_credits = _nonnegative_integer(
        payload["max_total_credits_per_run"],
        "phase00_external_effect_policy_credit_budget_invalid",
    )
    if max_requests <= 0:
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_request_budget_invalid"
        )

    provenance_payload = payload["pricing_provenance"]
    if not isinstance(provenance_payload, dict):
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_pricing_provenance_invalid"
        )
    _require_exact_fields(
        provenance_payload,
        _PROVENANCE_FIELDS,
        "phase00_external_effect_policy_pricing_provenance_fields_invalid",
    )
    captured = _utc_timestamp(provenance_payload["captured_at_utc"], "captured_at_utc")
    pricing_effective = _utc_timestamp(
        provenance_payload["effective_from_utc"],
        "pricing_effective_from_utc",
    )
    if pricing_effective > captured or captured > observed:
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_pricing_dates_invalid"
        )
    provenance = PricingProvenance(
        source_type=_required_text(
            provenance_payload["source_type"],
            "source_type",
        ),
        source_locator=_required_text(
            provenance_payload["source_locator"],
            "source_locator",
        ),
        captured_at_utc=captured.isoformat(),
        effective_from_utc=pricing_effective.isoformat(),
        evidence_status=_required_text(
            provenance_payload["evidence_status"],
            "evidence_status",
        ),
        unit_definition=_required_text(
            provenance_payload["unit_definition"],
            "unit_definition",
        ),
    )

    pricing_payload = payload["endpoint_pricing"]
    if not isinstance(pricing_payload, list) or not pricing_payload:
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_endpoint_pricing_invalid"
        )
    pricing: list[EndpointPricing] = []
    identities: set[tuple[str, str]] = set()
    priced_targets: set[str] = set()
    for row in pricing_payload:
        if not isinstance(row, dict):
            raise ExternalEffectPolicyError(
                "phase00_external_effect_policy_endpoint_pricing_invalid"
            )
        _require_exact_fields(
            row,
            _PRICING_FIELDS,
            "phase00_external_effect_policy_endpoint_pricing_fields_invalid",
        )
        method = _required_text(row["method"], "method").upper()
        if method not in {"GET", "POST"}:
            raise ExternalEffectPolicyError(
                "phase00_external_effect_policy_endpoint_method_invalid"
            )
        target = _https_target(_required_text(row["target"], "target"))
        if target not in normalized_targets:
            raise ExternalEffectPolicyError(
                "phase00_external_effect_policy_pricing_target_not_allowed"
            )
        credit_value = row["credit_units"]
        credits = (
            None
            if credit_value is None
            else _nonnegative_integer(
                credit_value,
                "phase00_external_effect_policy_endpoint_credits_invalid",
            )
        )
        operation = _required_text(row["operation"], "operation")
        identity = (method, operation)
        if identity in identities:
            raise ExternalEffectPolicyError(
                "phase00_external_effect_policy_endpoint_pricing_duplicate"
            )
        identities.add(identity)
        priced_targets.add(target)
        pricing.append(
            EndpointPricing(
                operation=operation,
                method=method,
                target=target,
                credit_units=credits,
                pricing_basis=_required_text(
                    row["pricing_basis"],
                    "pricing_basis",
                ),
            )
        )
    if priced_targets != set(normalized_targets):
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_unpriced_target"
        )
    return ProviderExternalEffectPolicy(
        provider_id=provider_id,
        account_scope_id=account_scope_id,
        allowed_targets=normalized_targets,
        allowed_credential_keys=credential_keys,
        max_total_requests_per_run=max_requests,
        max_total_credits_per_run=max_credits,
        pricing_provenance=provenance,
        endpoint_pricing=tuple(pricing),
    )


def _read_policy_bytes(root: Path, path: Path) -> bytes:
    resolved_root = root.resolve()
    if path.is_symlink():
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_symlink_forbidden"
        )
    try:
        metadata = path.stat()
    except FileNotFoundError as exc:
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_missing"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_not_regular"
        )
    if metadata.st_size <= 0 or metadata.st_size > MAX_EXTERNAL_EFFECT_POLICY_BYTES:
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_size_invalid"
        )
    resolved = path.resolve()
    if resolved.parent != (resolved_root / "config"):
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_path_invalid"
        )
    return path.read_bytes()


def _require_exact_fields(
    payload: dict[str, Any],
    required: frozenset[str],
    code: str,
) -> None:
    if frozenset(payload) != required:
        raise ExternalEffectPolicyError(code)


def _unique_text_values(
    value: object,
    code: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ExternalEffectPolicyError(code)
    result = tuple(_required_text(item, code) for item in value)
    if len(result) != len(set(result)):
        raise ExternalEffectPolicyError(code)
    return result


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExternalEffectPolicyError(
            f"phase00_external_effect_policy_text_invalid:{field}"
        )
    return value.strip()


def _nonnegative_integer(value: object, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExternalEffectPolicyError(code)
    return value


def _https_target(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_target_invalid"
        )
    return value.rstrip("/")


def _utc_timestamp(value: object, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ExternalEffectPolicyError(
            f"phase00_external_effect_policy_timestamp_invalid:{field}"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ExternalEffectPolicyError(
            f"phase00_external_effect_policy_timestamp_not_utc:{field}"
        )
    return parsed.astimezone(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ExternalEffectPolicyError(
            "phase00_external_effect_policy_as_of_not_timezone_aware"
        )
    return value.astimezone(UTC)
