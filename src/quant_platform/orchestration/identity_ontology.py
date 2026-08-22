"""Collision-resistant identities for the Phase 00 control plane."""

from __future__ import annotations

import json
import re
from datetime import datetime
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

IDENTITY_ONTOLOGY_VERSION = "thewiz.identity_ontology.v2"
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


def _identity(prefix: str, payload: dict[str, object]) -> str:
    material = {
        "ontology_version": IDENTITY_ONTOLOGY_VERSION,
        "identity_type": prefix,
        **payload,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}_{sha256(encoded.encode('utf-8')).hexdigest()[:24]}"


def _text(value: object, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} cannot be blank")
    return normalized


def _hash(value: object, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HASH_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field} must be a 64-character lowercase sha256")
    return normalized


class _IdentityModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    def _bind(self, field: str, prefix: str, payload: dict[str, object]) -> None:
        expected = _identity(prefix, payload)
        observed = str(getattr(self, field, "") or "")
        if observed and observed != expected:
            raise ValueError(f"{field} does not match canonical identity")
        object.__setattr__(self, field, expected)


class CanonicalAssetIdentity(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    source_system: str
    source_asset_id: str
    canonical_symbol: str
    asset_id: str = ""

    @field_validator("source_system", "source_asset_id", "canonical_symbol")
    @classmethod
    def _normalize_text(cls, value: str, info) -> str:
        normalized = _text(value, field=info.field_name)
        return normalized.upper() if info.field_name == "canonical_symbol" else normalized.lower()

    @model_validator(mode="after")
    def _identity(self) -> CanonicalAssetIdentity:
        self._bind(
            "asset_id",
            "asset",
            {
                "source_system": self.source_system,
                "source_asset_id": self.source_asset_id,
                "canonical_symbol": self.canonical_symbol,
            },
        )
        return self


class VenueIdentity(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    venue_code: str
    venue_id: str = ""

    @field_validator("venue_code")
    @classmethod
    def _normalize_venue(cls, value: str) -> str:
        return _text(value, field="venue_code").lower()

    @model_validator(mode="after")
    def _identity(self) -> VenueIdentity:
        self._bind("venue_id", "venue", {"venue_code": self.venue_code})
        return self


class VenueProductLaneIdentity(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    venue_id: str
    product_lane: str
    settlement_asset_id: str = ""
    product_lane_id: str = ""

    @field_validator("venue_id", "product_lane")
    @classmethod
    def _normalize_required(cls, value: str, info) -> str:
        return _text(value, field=info.field_name).lower()

    @model_validator(mode="after")
    def _identity(self) -> VenueProductLaneIdentity:
        self._bind(
            "product_lane_id",
            "lane",
            {
                "venue_id": self.venue_id,
                "product_lane": self.product_lane,
                "settlement_asset_id": self.settlement_asset_id,
            },
        )
        return self


class VenueInstrumentIdentity(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    venue_id: str
    product_lane_id: str
    venue_market_id: str
    unified_symbol: str
    base_asset_id: str
    quote_asset_id: str
    settle_asset_id: str = ""
    instrument_id: str = ""

    @field_validator(
        "venue_id",
        "product_lane_id",
        "venue_market_id",
        "unified_symbol",
        "base_asset_id",
        "quote_asset_id",
    )
    @classmethod
    def _normalize_required(cls, value: str, info) -> str:
        normalized = _text(value, field=info.field_name)
        return normalized.upper() if info.field_name == "unified_symbol" else normalized

    @model_validator(mode="after")
    def _identity(self) -> VenueInstrumentIdentity:
        self._bind(
            "instrument_id",
            "instrument",
            {
                "venue_id": self.venue_id,
                "product_lane_id": self.product_lane_id,
                "venue_market_id": self.venue_market_id,
                "unified_symbol": self.unified_symbol,
                "base_asset_id": self.base_asset_id,
                "quote_asset_id": self.quote_asset_id,
                "settle_asset_id": self.settle_asset_id,
            },
        )
        return self


class AccountScopeIdentity(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    venue_id: str
    environment: Literal["research", "testnet", "live"]
    account_id_hash: str
    subaccount_id_hash: str = ""
    account_scope_id: str = ""

    @field_validator("account_id_hash")
    @classmethod
    def _validate_account_hash(cls, value: str) -> str:
        return _hash(value, field="account_id_hash")

    @field_validator("subaccount_id_hash")
    @classmethod
    def _validate_subaccount_hash(cls, value: str) -> str:
        return "" if not value else _hash(value, field="subaccount_id_hash")

    @model_validator(mode="after")
    def _identity(self) -> AccountScopeIdentity:
        self._bind(
            "account_scope_id",
            "account",
            {
                "venue_id": self.venue_id,
                "environment": self.environment,
                "account_id_hash": self.account_id_hash,
                "subaccount_id_hash": self.subaccount_id_hash,
            },
        )
        return self


class UnorderedEconomicPairIdentity(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    asset_a_id: str
    asset_b_id: str
    economic_pair_id: str = ""

    @model_validator(mode="after")
    def _identity(self) -> UnorderedEconomicPairIdentity:
        first, second = sorted((_text(self.asset_a_id, field="asset_a_id"), _text(self.asset_b_id, field="asset_b_id")))
        if first == second:
            raise ValueError("economic pair assets must differ")
        object.__setattr__(self, "asset_a_id", first)
        object.__setattr__(self, "asset_b_id", second)
        self._bind(
            "economic_pair_id",
            "economic_pair",
            {"asset_a_id": first, "asset_b_id": second},
        )
        return self


class OrderedStrategyPairIdentity(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    ordered_asset_x_id: str
    ordered_asset_y_id: str
    economic_pair_id: str = ""
    ordered_pair_id: str = ""

    @model_validator(mode="after")
    def _identity(self) -> OrderedStrategyPairIdentity:
        economic = UnorderedEconomicPairIdentity(
            asset_a_id=self.ordered_asset_x_id,
            asset_b_id=self.ordered_asset_y_id,
        )
        if self.economic_pair_id and self.economic_pair_id != economic.economic_pair_id:
            raise ValueError("economic_pair_id does not match ordered assets")
        object.__setattr__(self, "economic_pair_id", economic.economic_pair_id)
        self._bind(
            "ordered_pair_id",
            "ordered_pair",
            {
                "asset_x_id": self.ordered_asset_x_id,
                "asset_y_id": self.ordered_asset_y_id,
                "economic_pair_id": economic.economic_pair_id,
            },
        )
        return self


class StrategyModeIdentity(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    strategy_mode: str
    formula_version: str
    strategy_mode_id: str = ""

    @model_validator(mode="after")
    def _identity(self) -> StrategyModeIdentity:
        mode = _text(self.strategy_mode, field="strategy_mode").lower()
        version = _text(self.formula_version, field="formula_version")
        object.__setattr__(self, "strategy_mode", mode)
        object.__setattr__(self, "formula_version", version)
        self._bind(
            "strategy_mode_id",
            "strategy_mode",
            {"strategy_mode": mode, "formula_version": version},
        )
        return self


class Phase00CandidateIdentity(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    source_system: str
    wizard_pair_id_or_source_candidate_id: str
    ordered_asset_x: str
    ordered_asset_y: str
    scanner_venue: str
    scanner_instrument_x: str
    scanner_instrument_y: str
    target_venue: str
    target_product_lane: str
    target_instrument_x: str
    target_instrument_y: str
    account_scope: str
    strategy_mode: str
    formula_version: str
    timeframe: str
    lookback: int = Field(gt=0)
    scanner_cutoff: datetime
    source_snapshot_hash: str
    economic_pair_id: str = ""
    ordered_pair_id: str = ""
    candidate_id: str = ""

    @field_validator("scanner_cutoff")
    @classmethod
    def _timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("scanner_cutoff must be timezone-aware")
        return value

    @field_validator("source_snapshot_hash")
    @classmethod
    def _snapshot_hash(cls, value: str) -> str:
        return _hash(value, field="source_snapshot_hash")

    @model_validator(mode="after")
    def _identity(self) -> Phase00CandidateIdentity:
        ordered = OrderedStrategyPairIdentity(
            ordered_asset_x_id=self.ordered_asset_x,
            ordered_asset_y_id=self.ordered_asset_y,
        )
        if self.economic_pair_id and self.economic_pair_id != ordered.economic_pair_id:
            raise ValueError("economic_pair_id mismatch")
        if self.ordered_pair_id and self.ordered_pair_id != ordered.ordered_pair_id:
            raise ValueError("ordered_pair_id mismatch")
        object.__setattr__(self, "economic_pair_id", ordered.economic_pair_id)
        object.__setattr__(self, "ordered_pair_id", ordered.ordered_pair_id)
        payload = {
            name: getattr(self, name)
            for name in (
                "source_system",
                "wizard_pair_id_or_source_candidate_id",
                "ordered_pair_id",
                "scanner_venue",
                "scanner_instrument_x",
                "scanner_instrument_y",
                "target_venue",
                "target_product_lane",
                "target_instrument_x",
                "target_instrument_y",
                "account_scope",
                "strategy_mode",
                "formula_version",
                "timeframe",
                "lookback",
                "scanner_cutoff",
                "source_snapshot_hash",
            )
        }
        self._bind("candidate_id", "candidate", payload)
        return self


class DatasetIdentity(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    dataset_name: str
    schema_hash: str
    source_manifest_hash: str
    dataset_id: str = ""

    @model_validator(mode="after")
    def _identity(self) -> DatasetIdentity:
        self._bind(
            "dataset_id",
            "dataset",
            {
                "dataset_name": _text(self.dataset_name, field="dataset_name"),
                "schema_hash": _hash(self.schema_hash, field="schema_hash"),
                "source_manifest_hash": _hash(self.source_manifest_hash, field="source_manifest_hash"),
            },
        )
        return self


class FeatureSchemaIdentity(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    schema_name: str
    fields_hash: str
    formula_version: str
    feature_schema_id: str = ""

    @model_validator(mode="after")
    def _identity(self) -> FeatureSchemaIdentity:
        self._bind(
            "feature_schema_id",
            "feature_schema",
            {
                "schema_name": _text(self.schema_name, field="schema_name"),
                "fields_hash": _hash(self.fields_hash, field="fields_hash"),
                "formula_version": _text(self.formula_version, field="formula_version"),
            },
        )
        return self


class ModelIdentity(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    model_name: str
    model_version: str
    feature_schema_id: str
    training_dataset_id: str
    artifact_hash: str
    model_id: str = ""

    @model_validator(mode="after")
    def _identity(self) -> ModelIdentity:
        self._bind(
            "model_id",
            "model",
            {
                "model_name": _text(self.model_name, field="model_name"),
                "model_version": _text(self.model_version, field="model_version"),
                "feature_schema_id": _text(self.feature_schema_id, field="feature_schema_id"),
                "training_dataset_id": _text(self.training_dataset_id, field="training_dataset_id"),
                "artifact_hash": _hash(self.artifact_hash, field="artifact_hash"),
            },
        )
        return self


class RunSlotIdentity(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    scheduler_key: str
    intended_slot: str
    source_hash: str
    runtime_hash: str
    configuration_hash: str
    run_id: str = ""

    @model_validator(mode="after")
    def _identity(self) -> RunSlotIdentity:
        self._bind(
            "run_id",
            "run",
            {
                "scheduler_key": _text(self.scheduler_key, field="scheduler_key"),
                "intended_slot": _text(self.intended_slot, field="intended_slot"),
                "source_hash": _hash(self.source_hash, field="source_hash"),
                "runtime_hash": _hash(self.runtime_hash, field="runtime_hash"),
                "configuration_hash": _hash(self.configuration_hash, field="configuration_hash"),
            },
        )
        return self


class SignalProposalIdentity(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    candidate_id: str
    run_id: str
    feature_snapshot_hash: str
    strategy_mode_id: str
    proposal_id: str = ""

    @model_validator(mode="after")
    def _identity(self) -> SignalProposalIdentity:
        self._bind(
            "proposal_id",
            "proposal",
            {
                "candidate_id": _text(self.candidate_id, field="candidate_id"),
                "run_id": _text(self.run_id, field="run_id"),
                "feature_snapshot_hash": _hash(self.feature_snapshot_hash, field="feature_snapshot_hash"),
                "strategy_mode_id": _text(self.strategy_mode_id, field="strategy_mode_id"),
            },
        )
        return self


class OrderIdentity(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    proposal_id: str
    venue_instrument_id: str
    account_scope_id: str
    client_order_id: str
    order_id: str = ""

    @model_validator(mode="after")
    def _identity(self) -> OrderIdentity:
        self._bind(
            "order_id",
            "order",
            {
                "proposal_id": self.proposal_id,
                "venue_instrument_id": self.venue_instrument_id,
                "account_scope_id": self.account_scope_id,
                "client_order_id": self.client_order_id,
            },
        )
        return self


class FillIdentity(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    order_id: str
    venue_fill_id: str
    fill_id: str = ""

    @model_validator(mode="after")
    def _identity(self) -> FillIdentity:
        self._bind(
            "fill_id",
            "fill",
            {"order_id": self.order_id, "venue_fill_id": self.venue_fill_id},
        )
        return self


class AuthorityReceiptIdentity(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    run_id: str
    effect_kind: str
    permit_id: str
    receipt_hash: str
    authority_receipt_id: str = ""

    @model_validator(mode="after")
    def _identity(self) -> AuthorityReceiptIdentity:
        self._bind(
            "authority_receipt_id",
            "authority_receipt",
            {
                "run_id": self.run_id,
                "effect_kind": self.effect_kind,
                "permit_id": self.permit_id,
                "receipt_hash": _hash(self.receipt_hash, field="receipt_hash"),
            },
        )
        return self


class IdentityMappingReceipt(_IdentityModel):
    schema_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    source_system: str
    source_id: str
    canonical_type: str
    canonical_id: str
    source_snapshot_hash: str
    mapped_at: datetime
    mapping_receipt_id: str = ""

    @field_validator("mapped_at")
    @classmethod
    def _timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("mapped_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _identity(self) -> IdentityMappingReceipt:
        self._bind(
            "mapping_receipt_id",
            "identity_mapping",
            {
                "source_system": self.source_system,
                "source_id": self.source_id,
                "canonical_type": self.canonical_type,
                "canonical_id": self.canonical_id,
                "source_snapshot_hash": _hash(self.source_snapshot_hash, field="source_snapshot_hash"),
                "mapped_at": self.mapped_at,
            },
        )
        return self
