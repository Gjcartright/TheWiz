"""Offline-verifiable CCXT public-market evidence contract for Phase 00."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import write_immutable_bytes

CCXT_EVIDENCE_SCHEMA_VERSION = "thewiz.phase00_ccxt_evidence_contract.v1"
CCXT_CAPTURE_SCHEMA_VERSION = "thewiz.phase00_ccxt_capture.v1"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "ccxt_evidence_contract.json"
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TIMEFRAME_PATTERN = re.compile(r"([1-9][0-9]*)([mhdw])")
_TIMEFRAME_MULTIPLIERS = {
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
}
_RETRYABLE_EXCEPTIONS = {
    "DDoSProtection",
    "ExchangeNotAvailable",
    "NetworkError",
    "RateLimitExceeded",
    "RequestTimeout",
}
_SENSITIVE_OPTION_KEYS = {
    "apikey",
    "api_key",
    "password",
    "privatekey",
    "private_key",
    "secret",
    "token",
    "uid",
    "wallet",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_json(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_json_value(value: object, *, location: str = "value") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location}:non_finite_number")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _strict_json_value(item, location=f"{location}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            _strict_json_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{location}:unsupported_type:{type(value).__name__}")


def _timeframe_ms(timeframe: str) -> int:
    match = _TIMEFRAME_PATTERN.fullmatch(str(timeframe or "").strip())
    if match is None:
        raise ValueError("unsupported_timeframe_format")
    return int(match.group(1)) * _TIMEFRAME_MULTIPLIERS[match.group(2)]


def _find_sensitive_keys(value: object, *, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            token = str(key).replace("-", "_").lower()
            location = f"{prefix}.{key}" if prefix else str(key)
            if token in _SENSITIVE_OPTION_KEYS:
                found.append(location)
            found.extend(_find_sensitive_keys(item, prefix=location))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_find_sensitive_keys(item, prefix=f"{prefix}[{index}]"))
    return found


@dataclass(frozen=True)
class CcxtEvidenceContract:
    payload: dict[str, Any]
    contract_hash: str

    @classmethod
    def from_path(cls, path: Path = DEFAULT_CONTRACT_PATH) -> CcxtEvidenceContract:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("ccxt_contract_unreadable") from exc
        if not isinstance(raw, dict):
            raise TypeError("ccxt_contract_not_an_object")
        payload = _strict_json_value(raw, location="contract")
        assert isinstance(payload, dict)
        cls._validate(payload)
        return cls(payload=payload, contract_hash=_hash_json(payload))

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != CCXT_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("ccxt_contract_schema_mismatch")
        package = payload.get("package")
        if not isinstance(package, dict):
            raise TypeError("ccxt_contract_package_missing")
        if package.get("name") != "ccxt" or package.get("version_policy") != "exact":
            raise ValueError("ccxt_contract_package_policy_invalid")
        pin = str(package.get("pinned_version", ""))
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", pin):
            raise ValueError("ccxt_contract_version_pin_invalid")

        phase00 = payload.get("phase00")
        expected_phase00 = {
            "public_data_only": True,
            "credentials_allowed": False,
            "sandbox_allowed": False,
            "account_eligibility_authority": False,
            "order_authority": False,
        }
        if not isinstance(phase00, dict) or any(
            phase00.get(key) is not expected for key, expected in expected_phase00.items()
        ):
            raise ValueError("ccxt_contract_phase00_authority_invalid")

        methods = payload.get("methods")
        if not isinstance(methods, dict):
            raise TypeError("ccxt_contract_methods_missing")
        allowed = methods.get("allowed_public")
        forbidden = methods.get("forbidden")
        if allowed != ["load_markets", "fetch_ohlcv"]:
            raise ValueError("ccxt_contract_public_methods_invalid")
        if not isinstance(forbidden, list) or not {"create_order", "withdraw"}.issubset(forbidden):
            raise ValueError("ccxt_contract_forbidden_methods_incomplete")
        if set(allowed).intersection(forbidden):
            raise ValueError("ccxt_contract_method_overlap")

        runtime = payload.get("runtime")
        ohlcv = payload.get("ohlcv")
        if not isinstance(runtime, dict) or runtime.get("enable_rate_limit") is not True:
            raise ValueError("ccxt_contract_rate_limit_not_required")
        if not isinstance(runtime.get("timeout_ms"), int) or runtime["timeout_ms"] <= 0:
            raise ValueError("ccxt_contract_timeout_invalid")
        if (
            not isinstance(runtime.get("market_cache_max_age_ms"), int)
            or runtime["market_cache_max_age_ms"] <= 0
        ):
            raise ValueError("ccxt_contract_market_cache_invalid")
        if not isinstance(ohlcv, dict):
            raise TypeError("ccxt_contract_ohlcv_missing")
        numeric_limits = (
            "minimum_page_limit",
            "default_page_limit",
            "maximum_page_limit",
            "maximum_pages",
            "maximum_history_bars",
        )
        if any(not isinstance(ohlcv.get(key), int) or ohlcv[key] <= 0 for key in numeric_limits):
            raise ValueError("ccxt_contract_ohlcv_limits_invalid")
        if not (
            ohlcv["minimum_page_limit"]
            <= ohlcv["default_page_limit"]
            <= ohlcv["maximum_page_limit"]
        ):
            raise ValueError("ccxt_contract_page_limit_order_invalid")
        if (
            ohlcv.get("end_boundary") != "exclusive"
            or ohlcv.get("gap_policy") != "fail"
            or ohlcv.get("conflicting_duplicate_policy") != "fail"
            or ohlcv.get("incomplete_current_bar_policy") != "exclude"
        ):
            raise ValueError("ccxt_contract_data_quality_policy_invalid")

        lanes = payload.get("lanes")
        if not isinstance(lanes, dict) or not lanes:
            raise ValueError("ccxt_contract_lanes_missing")
        registry_ids: list[str] = []
        for lane_name, lane in lanes.items():
            if not str(lane_name).strip() or not isinstance(lane, dict):
                raise ValueError("ccxt_contract_lane_invalid")
            if not str(lane.get("exchange_id", "")).strip():
                raise ValueError(f"ccxt_contract_lane_exchange_missing:{lane_name}")
            registry_id = str(lane.get("parameter_registry_id", "")).strip()
            if not registry_id:
                raise ValueError(f"ccxt_contract_parameter_registry_missing:{lane_name}")
            registry_ids.append(registry_id)
            options = lane.get("exchange_options")
            params = lane.get("fetch_ohlcv_params")
            if not isinstance(options, dict) or not isinstance(params, dict):
                raise TypeError(f"ccxt_contract_lane_parameters_invalid:{lane_name}")
            sensitive = _find_sensitive_keys({"options": options, "params": params})
            if sensitive:
                raise ValueError(f"ccxt_contract_sensitive_parameters:{lane_name}")
            boundary = lane.get("end_boundary_parameter")
            if boundary is not None and not isinstance(boundary, str):
                raise ValueError(f"ccxt_contract_end_parameter_invalid:{lane_name}")
        if len(registry_ids) != len(set(registry_ids)):
            raise ValueError("ccxt_contract_parameter_registry_not_unique")

    @property
    def pinned_version(self) -> str:
        return str(self.payload["package"]["pinned_version"])

    def lane(self, lane_name: str) -> dict[str, Any]:
        lane = self.payload["lanes"].get(lane_name)
        if not isinstance(lane, dict):
            raise KeyError(lane_name)
        return lane


@dataclass(frozen=True)
class CcxtCaptureRequest:
    source_id: str
    lane: str
    exchange_id: str
    symbol: str
    timeframe: str
    since_ms: int
    end_ms: int
    captured_at_ms: int
    page_limit: int | None = None
    force_market_reload: bool = False


@dataclass(frozen=True)
class CcxtCaptureResult:
    schema_version: str
    status: str
    capture_id: str
    blockers: tuple[str, ...]
    request: CcxtCaptureRequest
    contract: dict[str, Any]
    runtime: dict[str, Any]
    market_provenance: dict[str, Any]
    market: dict[str, Any]
    request_receipts: tuple[dict[str, Any], ...]
    coverage_receipt: dict[str, Any]
    freshness_receipt: dict[str, Any]
    raw_ohlcv: tuple[dict[str, Any], ...]
    normalized_ohlcv: tuple[dict[str, Any], ...]

    @property
    def accepted(self) -> bool:
        return self.status == "accepted" and not self.blockers and bool(self.normalized_ohlcv)

    def receipt_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "capture_id": self.capture_id,
            "blockers": list(self.blockers),
            "request": asdict(self.request),
            "contract": self.contract,
            "runtime": self.runtime,
            "market_provenance": self.market_provenance,
            "market": self.market,
            "request_receipts": list(self.request_receipts),
            "coverage_receipt": self.coverage_receipt,
            "freshness_receipt": self.freshness_receipt,
            "raw_ohlcv_sha256": _hash_json(list(self.raw_ohlcv)),
            "normalized_ohlcv_sha256": _hash_json(list(self.normalized_ohlcv)),
            "raw_row_count": len(self.raw_ohlcv),
            "normalized_row_count": len(self.normalized_ohlcv),
        }


@dataclass
class _MarketCacheEntry:
    loaded_at_ms: int
    snapshot_hash: str
    load_request_hash: str
    load_reload: bool
    markets: dict[str, dict[str, Any]]


ExchangeFactory = Callable[[str, dict[str, Any]], object]


class CcxtEvidenceRuntime:
    """Captures public CCXT evidence through an injected, version-bound client factory."""

    def __init__(
        self,
        *,
        contract: CcxtEvidenceContract,
        exchange_factory: ExchangeFactory,
        package_version: str | None,
        source_revision: str,
        runtime_fingerprint: Mapping[str, object] | None = None,
    ) -> None:
        self.contract = contract
        self._exchange_factory = exchange_factory
        self.package_version = str(package_version or "")
        self.source_revision = str(source_revision or "").strip().lower()
        self.runtime_fingerprint = _strict_json_value(
            dict(runtime_fingerprint or {}), location="runtime_fingerprint"
        )
        if _find_sensitive_keys(self.runtime_fingerprint):
            raise ValueError("runtime_fingerprint_contains_credentials")
        self._clients: dict[str, object] = {}
        self._market_cache: dict[str, _MarketCacheEntry] = {}
        self.runtime_hash = _hash_json(
            {
                "contract_hash": contract.contract_hash,
                "package_version": self.package_version,
                "source_revision": self.source_revision,
                "runtime_fingerprint": self.runtime_fingerprint,
            }
        )

    def capture(self, request: CcxtCaptureRequest) -> CcxtCaptureResult:
        preflight, timeframe_ms, lane = self._preflight(request)
        if preflight:
            return self._result(
                request=request,
                status="terminal_failure",
                blockers=preflight,
                coverage={"exact_reconciliation": False},
            )
        assert timeframe_ms is not None and lane is not None

        client, client_blocker = self._client(request.lane, lane)
        if client_blocker:
            return self._result(
                request=request,
                status="terminal_failure",
                blockers=[client_blocker],
                coverage={"exact_reconciliation": False},
            )
        assert client is not None

        capabilities = getattr(client, "has", None)
        timeframes = getattr(client, "timeframes", None)
        capability_receipt = {
            "fetch_ohlcv": capabilities.get("fetchOHLCV")
            if isinstance(capabilities, Mapping)
            else None,
            "timeframes": sorted(str(item) for item in timeframes)
            if isinstance(timeframes, Mapping)
            else [],
        }
        if capability_receipt["fetch_ohlcv"] is not True:
            return self._result(
                request=request,
                status="terminal_failure",
                blockers=["fetch_ohlcv_capability_not_explicitly_true"],
                market_provenance={"capability_receipt": capability_receipt},
                coverage={"exact_reconciliation": False},
            )
        if not isinstance(timeframes, Mapping) or request.timeframe not in timeframes:
            return self._result(
                request=request,
                status="terminal_failure",
                blockers=["timeframe_not_supported"],
                market_provenance={"capability_receipt": capability_receipt},
                coverage={"exact_reconciliation": False},
            )

        market_entry, provenance, market_blocker = self._load_market(
            client=client,
            request=request,
            capability_receipt=capability_receipt,
        )
        if market_blocker:
            failure_status = (
                "retryable_failure"
                if market_blocker.startswith("load_markets:retryable_failure")
                else "terminal_failure"
            )
            return self._result(
                request=request,
                status=failure_status,
                blockers=[market_blocker],
                market_provenance=provenance,
                coverage={"exact_reconciliation": False},
            )
        assert market_entry is not None

        return self._capture_ohlcv(
            client=client,
            request=request,
            lane=lane,
            timeframe_ms=timeframe_ms,
            market=market_entry,
            market_provenance=provenance,
        )

    def _preflight(
        self, request: CcxtCaptureRequest
    ) -> tuple[list[str], int | None, dict[str, Any] | None]:
        blockers: list[str] = []
        if not self.package_version:
            blockers.append("ccxt_package_absent")
        elif self.package_version != self.contract.pinned_version:
            blockers.append("ccxt_package_version_mismatch")
        if not _SHA256_PATTERN.fullmatch(self.source_revision):
            blockers.append("ccxt_source_revision_invalid")
        for field_name in ("source_id", "lane", "exchange_id", "symbol", "timeframe"):
            if not str(getattr(request, field_name, "") or "").strip():
                blockers.append(f"request_{field_name}_missing")
        try:
            lane = self.contract.lane(request.lane)
        except KeyError:
            lane = None
            blockers.append("lane_not_registered")
        if lane is not None and request.exchange_id != lane["exchange_id"]:
            blockers.append("lane_exchange_mismatch")
        try:
            timeframe_ms = _timeframe_ms(request.timeframe)
        except ValueError:
            timeframe_ms = None
            blockers.append("timeframe_format_invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (request.since_ms, request.end_ms, request.captured_at_ms)
        ):
            blockers.append("utc_millisecond_boundary_invalid")
        elif timeframe_ms is not None:
            if request.since_ms >= request.end_ms:
                blockers.append("history_interval_empty")
            if request.since_ms % timeframe_ms or request.end_ms % timeframe_ms:
                blockers.append("history_boundary_not_timeframe_aligned")
            if request.captured_at_ms <= request.since_ms:
                blockers.append("capture_time_not_after_start")
            if request.end_ms > request.captured_at_ms + timeframe_ms:
                blockers.append("history_end_exceeds_current_bar")
            requested_bars = (request.end_ms - request.since_ms) // timeframe_ms
            if requested_bars > int(self.contract.payload["ohlcv"]["maximum_history_bars"]):
                blockers.append("history_bar_limit_exceeded")
        page_limit = request.page_limit
        if page_limit is not None:
            minimum = int(self.contract.payload["ohlcv"]["minimum_page_limit"])
            maximum = int(self.contract.payload["ohlcv"]["maximum_page_limit"])
            if (
                isinstance(page_limit, bool)
                or not isinstance(page_limit, int)
                or not minimum <= page_limit <= maximum
            ):
                blockers.append("page_limit_out_of_bounds")
        return sorted(set(blockers)), timeframe_ms, lane

    def _client(self, lane_name: str, lane: dict[str, Any]) -> tuple[object | None, str | None]:
        if lane_name in self._clients:
            return self._clients[lane_name], None
        options = {
            **lane["exchange_options"],
            "enableRateLimit": bool(self.contract.payload["runtime"]["enable_rate_limit"]),
            "timeout": int(self.contract.payload["runtime"]["timeout_ms"]),
        }
        if _find_sensitive_keys(options):
            return None, "exchange_options_contain_credentials"
        try:
            client = self._exchange_factory(str(lane["exchange_id"]), options)
        except Exception as exc:  # noqa: BLE001 - adapter exception types are injected
            classification = self._exception_status(exc)
            return None, f"exchange_factory:{classification}:{type(exc).__name__}"
        if str(getattr(client, "id", "")) != lane["exchange_id"]:
            return None, "exchange_client_id_mismatch"
        self._clients[lane_name] = client
        return client, None

    def _load_market(
        self,
        *,
        client: object,
        request: CcxtCaptureRequest,
        capability_receipt: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any], str | None]:
        cache = self._market_cache.get(request.lane)
        cache_max_age = int(self.contract.payload["runtime"]["market_cache_max_age_ms"])
        age_ms = request.captured_at_ms - cache.loaded_at_ms if cache is not None else None
        if age_ms is not None and age_ms < 0:
            return None, {"capability_receipt": capability_receipt}, "market_cache_time_reversal"
        reload_required = (
            request.force_market_reload or cache is None or age_ms is None or age_ms > cache_max_age
        )
        cache_status = "miss" if cache is None else "forced_reload"
        if cache is not None and age_ms is not None and age_ms > cache_max_age:
            cache_status = "stale_reloaded"
        elif cache is not None and not reload_required:
            cache_status = "fresh_reused"
        if reload_required:
            load_reload = request.force_market_reload or cache is not None
            load_request_hash = _hash_json(
                {
                    "method": "load_markets",
                    "exchange_id": request.exchange_id,
                    "lane": request.lane,
                    "reload": load_reload,
                }
            )
            try:
                loaded = client.load_markets(reload=load_reload)
                markets = self._normalize_markets(loaded)
            except Exception as exc:  # noqa: BLE001 - adapter exception types are injected
                classification = self._exception_status(exc)
                provenance = {
                    "capability_receipt": capability_receipt,
                    "cache_status": cache_status,
                    "cache_age_ms": age_ms,
                    "load_markets_status": classification,
                    "load_markets_request_hash": load_request_hash,
                    "load_markets_response_hash": _hash_json(
                        {
                            "status": classification,
                            "exception_class": type(exc).__name__,
                        }
                    ),
                    "load_markets_reload": load_reload,
                    "exception_class": type(exc).__name__,
                }
                return None, provenance, f"load_markets:{classification}:{type(exc).__name__}"
            cache = _MarketCacheEntry(
                loaded_at_ms=request.captured_at_ms,
                snapshot_hash=_hash_json(markets),
                load_request_hash=load_request_hash,
                load_reload=load_reload,
                markets=markets,
            )
            self._market_cache[request.lane] = cache
        assert cache is not None
        market = cache.markets.get(request.symbol)
        provenance = {
            "capability_receipt": capability_receipt,
            "cache_status": cache_status,
            "cache_age_ms": age_ms,
            "cache_max_age_ms": cache_max_age,
            "loaded_at_ms": cache.loaded_at_ms,
            "market_snapshot_hash": cache.snapshot_hash,
            "load_markets_status": "success",
            "load_markets_request_hash": cache.load_request_hash,
            "load_markets_response_hash": cache.snapshot_hash,
            "load_markets_reload": cache.load_reload,
            "force_reload": request.force_market_reload,
        }
        if market is None:
            return None, provenance, "market_symbol_not_loaded"
        return market, provenance, None

    @staticmethod
    def _normalize_markets(value: object) -> dict[str, dict[str, Any]]:
        if not isinstance(value, Mapping) or not value:
            raise ValueError("market_snapshot_empty_or_malformed")
        normalized: dict[str, dict[str, Any]] = {}
        for key, raw_market in sorted(value.items(), key=lambda pair: str(pair[0])):
            if not isinstance(raw_market, Mapping):
                raise TypeError("market_entry_malformed")
            symbol = str(raw_market.get("symbol") or key).strip()
            if not symbol:
                raise ValueError("market_symbol_missing")
            margin_modes = raw_market.get("marginModes", [])
            if isinstance(margin_modes, Mapping):
                margin_modes = sorted(
                    str(item) for item, enabled in margin_modes.items() if enabled
                )
            elif isinstance(margin_modes, (list, tuple, set)):
                margin_modes = sorted(str(item) for item in margin_modes)
            else:
                margin_modes = []
            normalized[symbol] = {
                "venue_market_id": str(raw_market.get("id", "")),
                "unified_symbol": symbol,
                "base": str(raw_market.get("base", "")),
                "quote": str(raw_market.get("quote", "")),
                "type": str(raw_market.get("type", "")),
                "spot": bool(raw_market.get("spot", False)),
                "swap": bool(raw_market.get("swap", False)),
                "future": bool(raw_market.get("future", False)),
                "contract": bool(raw_market.get("contract", False)),
                "settle": str(raw_market.get("settle", "") or ""),
                "contract_size": raw_market.get("contractSize"),
                "linear": raw_market.get("linear"),
                "inverse": raw_market.get("inverse"),
                "margin_modes": margin_modes,
                "active": raw_market.get("active"),
                "limits": _strict_json_value(
                    raw_market.get("limits", {}), location="market.limits"
                ),
                "precision": _strict_json_value(
                    raw_market.get("precision", {}), location="market.precision"
                ),
            }
        return normalized

    def _capture_ohlcv(
        self,
        *,
        client: object,
        request: CcxtCaptureRequest,
        lane: dict[str, Any],
        timeframe_ms: int,
        market: dict[str, Any],
        market_provenance: dict[str, Any],
    ) -> CcxtCaptureResult:
        ohlcv = self.contract.payload["ohlcv"]
        page_limit = request.page_limit or int(ohlcv["default_page_limit"])
        maximum_pages = int(ohlcv["maximum_pages"])
        complete_end_ms = min(
            request.end_ms,
            (request.captured_at_ms // timeframe_ms) * timeframe_ms,
        )
        cursor = request.since_ms
        raw_rows: list[dict[str, Any]] = []
        parsed_rows: list[tuple[int, float, float, float, float, float]] = []
        receipts: list[dict[str, Any]] = []
        collection_blocker: str | None = None
        collection_status: str | None = None

        for page_index in range(maximum_pages):
            if cursor >= request.end_ms:
                break
            remaining = max(1, math.ceil((request.end_ms - cursor) / timeframe_ms))
            limit = min(page_limit, remaining)
            params = dict(lane["fetch_ohlcv_params"])
            boundary_name = lane.get("end_boundary_parameter")
            if boundary_name:
                params[str(boundary_name)] = request.end_ms - 1
            request_payload = {
                "page_index": page_index,
                "exchange_id": request.exchange_id,
                "parameter_registry_id": lane["parameter_registry_id"],
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "since_ms": cursor,
                "end_ms_exclusive": request.end_ms,
                "limit": limit,
                "params": params,
            }
            request_hash = _hash_json(request_payload)
            try:
                response = client.fetch_ohlcv(
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    since=cursor,
                    limit=limit,
                    params=params,
                )
            except Exception as exc:  # noqa: BLE001 - adapter exception types are injected
                status = self._exception_status(exc)
                receipts.append(
                    {
                        **request_payload,
                        "request_hash": request_hash,
                        "response_hash": _hash_json(
                            {"status": status, "exception_class": type(exc).__name__}
                        ),
                        "status": status,
                        "exception_class": type(exc).__name__,
                        "row_count": 0,
                    }
                )
                collection_blocker = f"fetch_ohlcv:{status}:{type(exc).__name__}"
                collection_status = status
                break
            try:
                canonical_response = _strict_json_value(response, location="fetch_ohlcv_response")
            except (TypeError, ValueError):
                canonical_response = {"malformed_type": type(response).__name__}
                collection_blocker = "fetch_ohlcv:malformed_page"
                collection_status = "data_quality_failure"
            response_hash = _hash_json(canonical_response)
            if collection_blocker:
                receipts.append(
                    {
                        **request_payload,
                        "request_hash": request_hash,
                        "response_hash": response_hash,
                        "status": "malformed",
                        "row_count": 0,
                    }
                )
                break
            if not isinstance(response, list):
                receipts.append(
                    {
                        **request_payload,
                        "request_hash": request_hash,
                        "response_hash": response_hash,
                        "status": "malformed",
                        "row_count": 0,
                    }
                )
                collection_blocker = "fetch_ohlcv:malformed_page"
                collection_status = "data_quality_failure"
                break
            if len(response) > limit:
                receipts.append(
                    {
                        **request_payload,
                        "request_hash": request_hash,
                        "response_hash": response_hash,
                        "status": "malformed",
                        "row_count": len(response),
                    }
                )
                collection_blocker = "fetch_ohlcv:response_exceeds_requested_limit"
                collection_status = "data_quality_failure"
                break
            receipts.append(
                {
                    **request_payload,
                    "request_hash": request_hash,
                    "response_hash": response_hash,
                    "status": (
                        "empty"
                        if not response
                        else "partial"
                        if len(response) < limit
                        else "success"
                    ),
                    "row_count": len(response),
                }
            )
            if not response:
                break

            page_timestamps: list[int] = []
            for row_index, row in enumerate(response):
                raw_record = {
                    "page_index": page_index,
                    "row_index": row_index,
                    "row": _strict_json_value(row, location="ohlcv_row"),
                }
                raw_record["row_hash"] = _hash_json(raw_record["row"])
                raw_rows.append(raw_record)
                try:
                    parsed = self._parse_ohlcv_row(row, timeframe_ms=timeframe_ms)
                except (TypeError, ValueError) as exc:
                    collection_blocker = f"ohlcv_row_invalid:{safe_exception_code(exc)}"
                    collection_status = "data_quality_failure"
                    break
                parsed_rows.append(parsed)
                if cursor <= parsed[0] < request.end_ms:
                    page_timestamps.append(parsed[0])
            if collection_blocker:
                break
            if not page_timestamps:
                collection_blocker = "pagination_no_progress"
                collection_status = "partial" if parsed_rows else "data_quality_failure"
                break
            next_cursor = max(page_timestamps) + timeframe_ms
            if next_cursor <= cursor:
                collection_blocker = "pagination_no_progress"
                collection_status = "partial"
                break
            cursor = next_cursor
            if cursor >= request.end_ms:
                break
        else:
            collection_blocker = "pagination_page_limit_exceeded"
            collection_status = "partial"

        normalized, quality = self._normalize_ohlcv_rows(
            rows=parsed_rows,
            request=request,
            timeframe_ms=timeframe_ms,
            complete_end_ms=complete_end_ms,
        )
        blockers: list[str] = []
        if collection_blocker:
            blockers.append(collection_blocker)
        if quality["conflicting_duplicate_timestamps_ms"]:
            blockers.append("conflicting_duplicate_bar")
        if quality["missing_timestamps_ms"]:
            blockers.append("partial_history_gap")
        if quality["unaligned_timestamps_ms"]:
            blockers.append("ohlcv_timestamp_not_aligned")

        expected_count = int(quality["expected_bar_count"])
        observed_count = len(normalized)
        exact = (
            expected_count > 0
            and observed_count == expected_count
            and not quality["missing_timestamps_ms"]
            and not quality["conflicting_duplicate_timestamps_ms"]
            and not quality["unaligned_timestamps_ms"]
        )
        coverage = {
            "requested_since_ms": request.since_ms,
            "requested_end_ms_exclusive": request.end_ms,
            "complete_end_ms_exclusive": complete_end_ms,
            "timeframe_ms": timeframe_ms,
            "expected_bar_count": expected_count,
            "returned_complete_bar_count": observed_count,
            "returned_first_ms": normalized[0]["timestamp_ms"] if normalized else None,
            "returned_last_ms": normalized[-1]["timestamp_ms"] if normalized else None,
            "missing_timestamps_ms": quality["missing_timestamps_ms"],
            "outside_interval_count": quality["outside_interval_count"],
            "incomplete_bar_count": quality["incomplete_bar_count"],
            "identical_duplicate_count": quality["identical_duplicate_count"],
            "conflicting_duplicate_timestamps_ms": quality["conflicting_duplicate_timestamps_ms"],
            "unaligned_timestamps_ms": quality["unaligned_timestamps_ms"],
            "exact_reconciliation": exact,
        }
        if not exact and not blockers:
            blockers.append("coverage_not_exactly_reconciled")

        if collection_status is not None:
            status = collection_status
        elif not normalized:
            status = "empty"
            blockers.append("zero_complete_rows")
        elif quality["conflicting_duplicate_timestamps_ms"] or quality["unaligned_timestamps_ms"]:
            status = "data_quality_failure"
        elif quality["missing_timestamps_ms"] or not exact:
            status = "partial"
        else:
            status = "accepted"
        if status == "accepted" and blockers:
            status = "data_quality_failure"

        last_close_ms = normalized[-1]["timestamp_ms"] + timeframe_ms if normalized else None
        freshness = {
            "market_metadata_order": "first_order",
            "ohlcv_order": "second_order",
            "captured_at_ms": request.captured_at_ms,
            "market_age_ms": request.captured_at_ms - int(market_provenance["loaded_at_ms"]),
            "last_complete_bar_close_ms": last_close_ms,
            "ohlcv_age_ms": request.captured_at_ms - last_close_ms
            if last_close_ms is not None
            else None,
            "incomplete_current_bar_policy": "exclude",
        }
        return self._result(
            request=request,
            status=status,
            blockers=blockers,
            market_provenance=market_provenance,
            market=market,
            request_receipts=receipts,
            coverage=coverage,
            freshness=freshness,
            raw_ohlcv=raw_rows,
            normalized_ohlcv=normalized,
        )

    @staticmethod
    def _parse_ohlcv_row(
        row: object, *, timeframe_ms: int
    ) -> tuple[int, float, float, float, float, float]:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            raise ValueError("shape")
        timestamp_raw = row[0]
        if isinstance(timestamp_raw, bool) or not isinstance(timestamp_raw, (int, float)):
            raise TypeError("timestamp_type")
        timestamp = int(timestamp_raw)
        if timestamp != timestamp_raw or timestamp < 0:
            raise ValueError("timestamp_value")
        if timestamp % timeframe_ms:
            raise ValueError("timestamp_alignment")
        values: list[float] = []
        for item in row[1:6]:
            if isinstance(item, bool):
                raise TypeError("numeric_type")
            try:
                number = float(item)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("numeric_type") from exc
            if not math.isfinite(number):
                raise ValueError("non_finite")
            values.append(number)
        open_price, high, low, close, volume = values
        if min(open_price, high, low, close) <= 0 or volume < 0:
            raise ValueError("price_or_volume_domain")
        if high < max(open_price, close, low) or low > min(open_price, close, high):
            raise ValueError("ohlc_envelope")
        return timestamp, open_price, high, low, close, volume

    @staticmethod
    def _normalize_ohlcv_rows(
        *,
        rows: Sequence[tuple[int, float, float, float, float, float]],
        request: CcxtCaptureRequest,
        timeframe_ms: int,
        complete_end_ms: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        by_timestamp: dict[int, tuple[int, float, float, float, float, float]] = {}
        identical_duplicates = 0
        conflicting: set[int] = set()
        unaligned: set[int] = set()
        outside = 0
        incomplete = 0
        for row in rows:
            timestamp = row[0]
            if timestamp % timeframe_ms:
                unaligned.add(timestamp)
                continue
            if timestamp < request.since_ms or timestamp >= request.end_ms:
                outside += 1
                continue
            if timestamp >= complete_end_ms:
                incomplete += 1
                continue
            prior = by_timestamp.get(timestamp)
            if prior is None:
                by_timestamp[timestamp] = row
            elif prior == row:
                identical_duplicates += 1
            else:
                conflicting.add(timestamp)

        expected = list(range(request.since_ms, complete_end_ms, timeframe_ms))
        missing = sorted(set(expected).difference(by_timestamp))
        normalized = [
            {
                "timestamp_ms": timestamp,
                "open": by_timestamp[timestamp][1],
                "high": by_timestamp[timestamp][2],
                "low": by_timestamp[timestamp][3],
                "close": by_timestamp[timestamp][4],
                "raw_base_volume": by_timestamp[timestamp][5],
                "raw_base_volume_unit": "base_asset_units",
                "raw_quote_volume": None,
                "raw_quote_volume_unit": "unavailable",
                "usd_notional_volume": None,
                "usd_notional_volume_unit": "unavailable",
                "usd_notional_volume_source": "unavailable",
            }
            for timestamp in sorted(by_timestamp)
            if timestamp not in conflicting
        ]
        return normalized, {
            "expected_bar_count": len(expected),
            "missing_timestamps_ms": missing,
            "outside_interval_count": outside,
            "incomplete_bar_count": incomplete,
            "identical_duplicate_count": identical_duplicates,
            "conflicting_duplicate_timestamps_ms": sorted(conflicting),
            "unaligned_timestamps_ms": sorted(unaligned),
        }

    @staticmethod
    def _exception_status(exc: Exception) -> str:
        return (
            "retryable_failure"
            if type(exc).__name__ in _RETRYABLE_EXCEPTIONS
            else "terminal_failure"
        )

    def _result(
        self,
        *,
        request: CcxtCaptureRequest,
        status: str,
        blockers: Sequence[str],
        market_provenance: Mapping[str, Any] | None = None,
        market: Mapping[str, Any] | None = None,
        request_receipts: Sequence[Mapping[str, Any]] = (),
        coverage: Mapping[str, Any] | None = None,
        freshness: Mapping[str, Any] | None = None,
        raw_ohlcv: Sequence[Mapping[str, Any]] = (),
        normalized_ohlcv: Sequence[Mapping[str, Any]] = (),
    ) -> CcxtCaptureResult:
        contract_receipt = {
            "contract_version": self.contract.payload["contract_version"],
            "contract_hash": self.contract.contract_hash,
            "governing_reference": self.contract.payload["governing_reference"],
            "parameter_registry_id": self.contract.payload["lanes"]
            .get(request.lane, {})
            .get("parameter_registry_id"),
        }
        runtime_receipt = {
            "package_name": "ccxt",
            "package_version": self.package_version or None,
            "pinned_package_version": self.contract.pinned_version,
            "source_revision": self.source_revision or None,
            "runtime_hash": self.runtime_hash,
            "runtime_fingerprint": self.runtime_fingerprint,
            "public_data_only": True,
            "credentials_used": False,
            "network_authority": "injected_client_only",
        }
        material = {
            "schema_version": CCXT_CAPTURE_SCHEMA_VERSION,
            "status": status,
            "blockers": sorted({str(item) for item in blockers}),
            "request": asdict(request),
            "contract": contract_receipt,
            "runtime": runtime_receipt,
            "market_provenance": dict(market_provenance or {}),
            "market": dict(market or {}),
            "request_receipts": [dict(item) for item in request_receipts],
            "coverage_receipt": dict(coverage or {}),
            "freshness_receipt": dict(freshness or {}),
            "raw_ohlcv": [dict(item) for item in raw_ohlcv],
            "normalized_ohlcv": [dict(item) for item in normalized_ohlcv],
        }
        canonical = _strict_json_value(material, location="capture")
        assert isinstance(canonical, dict)
        capture_id = _hash_json(canonical)
        return CcxtCaptureResult(
            schema_version=CCXT_CAPTURE_SCHEMA_VERSION,
            status=status,
            capture_id=capture_id,
            blockers=tuple(canonical["blockers"]),
            request=request,
            contract=canonical["contract"],
            runtime=canonical["runtime"],
            market_provenance=canonical["market_provenance"],
            market=canonical["market"],
            request_receipts=tuple(canonical["request_receipts"]),
            coverage_receipt=canonical["coverage_receipt"],
            freshness_receipt=canonical["freshness_receipt"],
            raw_ohlcv=tuple(canonical["raw_ohlcv"]),
            normalized_ohlcv=tuple(canonical["normalized_ohlcv"]),
        )


def write_ccxt_capture_evidence(result: CcxtCaptureResult, *, root: Path) -> dict[str, Path]:
    directory = root / "data" / "external" / "ccxt" / "evidence" / result.capture_id
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw_ohlcv": directory / "raw_ohlcv.json",
        "normalized_ohlcv": directory / "normalized_ohlcv.json",
        "receipt": directory / "receipt.json",
    }
    payloads = {
        "raw_ohlcv": list(result.raw_ohlcv),
        "normalized_ohlcv": list(result.normalized_ohlcv),
        "receipt": result.receipt_dict(),
    }
    for key, path in paths.items():
        encoded = json.dumps(payloads[key], indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        try:
            write_immutable_bytes(path, encoded.encode("utf-8"))
        except ValueError as exc:
            raise FileExistsError(
                f"immutable_ccxt_evidence_conflict:{path}"
            ) from exc
    return paths
