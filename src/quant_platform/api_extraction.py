from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd
import requests

from quant_platform.crypto_wizards_catalog import BASE_URL
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_csv,
    atomic_write_text,
)


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    url: str
    method: str = "GET"
    params: dict[str, Any] | None = None


class CryptoWizardsFetchError(RuntimeError):
    """Raised when a live Crypto Wizards endpoint cannot be fetched."""


@dataclass(frozen=True)
class CryptoWizardsEndpointDiagnostic:
    name: str
    url: str
    dns_ok: bool
    dns_error: str
    http_ok: bool
    status_code: int | None
    error: str


@dataclass(frozen=True)
class CryptoWizardsLiveConfig:
    base_url: str | None
    api_key: str | None
    endpoints: tuple[EndpointSpec, ...] = ()

    @classmethod
    def from_env(
        cls,
        endpoints: tuple[EndpointSpec, ...] | None = None,
        base_url_env: str = "CRYPTO_WIZARDS_BASE_URL",
        api_key_env: str = "CRYPTO_WIZARDS_API_KEY",
        endpoints_env: str = "CRYPTO_WIZARDS_ENDPOINTS",
    ) -> CryptoWizardsLiveConfig:
        configured_endpoints = endpoints if endpoints is not None else parse_endpoint_specs(os.getenv(endpoints_env, ""))
        return cls(
            base_url=os.getenv(base_url_env),
            api_key=os.getenv(api_key_env),
            endpoints=configured_endpoints,
        )

    def missing_requirements(self) -> list[str]:
        missing: list[str] = []
        if not self.base_url:
            missing.append("CRYPTO_WIZARDS_BASE_URL")
        if not self.endpoints:
            missing.append("CRYPTO_WIZARDS_ENDPOINTS")
        return missing


def parse_endpoint_specs(value: str) -> tuple[EndpointSpec, ...]:
    specs: list[EndpointSpec] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" in item:
            name, url = item.split("=", 1)
        elif ":" in item:
            name, url = item.split(":", 1)
        else:
            raise ValueError(f"endpoint spec must be name=url or name:path: {item}")
        specs.append(EndpointSpec(name=name.strip(), url=url.strip()))
    return tuple(specs)


class CryptoWizardsExtractor:
    """Discovers and archives Crypto Wizards responses for research reproducibility."""

    def __init__(self, base_url: str, api_key: str | None = None, archive_dir: str | Path = "data/raw") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, endpoint: EndpointSpec) -> dict[str, Any]:
        headers = {"X-api-key": self.api_key} if self.api_key else {}
        headers["Content-Type"] = "application/json"
        url = self.endpoint_url(endpoint)
        try:
            if _is_official_wizard_api_url(url):
                from quant_platform.crypto_wizards_history import (
                    fetch_crypto_wizards_json,
                )

                method = endpoint.method.strip().upper()
                payload = fetch_crypto_wizards_json(
                    method=method,
                    url=url,
                    params=endpoint.params if method == "GET" else None,
                    payload=endpoint.params if method == "POST" else None,
                    api_key=self.api_key,
                    timeout=30,
                )
            else:
                response = requests.request(
                    endpoint.method,
                    url,
                    params=endpoint.params,
                    headers=headers,
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
        except requests.exceptions.RequestException as exc:
            raise CryptoWizardsFetchError(f"failed to fetch endpoint {endpoint.name} at {url}: {exc}") from exc
        except ValueError as exc:
            raise CryptoWizardsFetchError(
                f"failed to fetch endpoint {endpoint.name} at {url}: {exc}"
            ) from exc
        self.archive(endpoint.name, payload)
        return payload

    def endpoint_url(self, endpoint: EndpointSpec) -> str:
        if endpoint.url.startswith(("http://", "https://")):
            return endpoint.url
        return f"{self.base_url}/{endpoint.url.lstrip('/')}"

    @classmethod
    def from_live_config(
        cls,
        config: CryptoWizardsLiveConfig,
        archive_dir: str | Path = "data/raw",
    ) -> CryptoWizardsExtractor:
        missing = config.missing_requirements()
        if missing:
            raise ValueError(f"Crypto Wizards live config missing: {', '.join(missing)}")
        return cls(base_url=str(config.base_url), api_key=config.api_key, archive_dir=archive_dir)

    def fetch_all(self, endpoints: tuple[EndpointSpec, ...]) -> dict[str, Any]:
        return {endpoint.name: self.fetch(endpoint) for endpoint in endpoints}

    def diagnose_endpoint(self, endpoint: EndpointSpec, timeout: float = 10.0) -> CryptoWizardsEndpointDiagnostic:
        url = self.endpoint_url(endpoint)
        if _is_official_wizard_api_url(url):
            try:
                from quant_platform.crypto_wizards_history import (
                    fetch_crypto_wizards_json,
                )

                method = endpoint.method.strip().upper()
                fetch_crypto_wizards_json(
                    method=method,
                    url=url,
                    params=endpoint.params if method == "GET" else None,
                    payload=endpoint.params if method == "POST" else None,
                    api_key=self.api_key,
                    timeout=timeout,
                )
                return CryptoWizardsEndpointDiagnostic(
                    name=endpoint.name,
                    url=url,
                    dns_ok=True,
                    dns_error="",
                    http_ok=True,
                    status_code=200,
                    error="",
                )
            except (CryptoWizardsFetchError, ValueError) as exc:
                return CryptoWizardsEndpointDiagnostic(
                    name=endpoint.name,
                    url=url,
                    dns_ok=False,
                    dns_error="not_probed_outside_effect_authority",
                    http_ok=False,
                    status_code=None,
                    error=str(exc),
                )
        host = requests.utils.urlparse(url).hostname or ""
        dns_error = ""
        dns_ok = False
        try:
            socket.getaddrinfo(host, 443)
            dns_ok = True
        except OSError as exc:
            dns_error = str(exc)

        headers = {"X-api-key": self.api_key} if self.api_key else {}
        headers["Content-Type"] = "application/json"
        try:
            response = requests.request(endpoint.method, url, headers=headers, timeout=timeout)
            return CryptoWizardsEndpointDiagnostic(
                name=endpoint.name,
                url=url,
                dns_ok=dns_ok,
                dns_error=dns_error,
                http_ok=response.ok,
                status_code=response.status_code,
                error="" if response.ok else response.text[:240],
            )
        except requests.exceptions.RequestException as exc:
            return CryptoWizardsEndpointDiagnostic(
                name=endpoint.name,
                url=url,
                dns_ok=dns_ok,
                dns_error=dns_error,
                http_ok=False,
                status_code=None,
                error=str(exc),
            )

    def diagnose_all(self, endpoints: tuple[EndpointSpec, ...], timeout: float = 10.0) -> list[CryptoWizardsEndpointDiagnostic]:
        return [self.diagnose_endpoint(endpoint, timeout=timeout) for endpoint in endpoints]

    def archive(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.archive_dir / f"{name}.json"
        atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @staticmethod
    def discover_fields(payload: Any, prefix: str = "") -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        if isinstance(payload, dict):
            for key, value in payload.items():
                full = f"{prefix}.{key}" if prefix else key
                rows.append({"field": full, "type": type(value).__name__, "example": str(value)[:160]})
                rows.extend(CryptoWizardsExtractor.discover_fields(value, full))
        elif isinstance(payload, list) and payload:
            rows.extend(CryptoWizardsExtractor.discover_fields(payload[0], f"{prefix}[]"))
        return rows

    @staticmethod
    def write_discovered_fields(payloads: dict[str, Any], output_path: str | Path) -> None:
        all_rows: list[dict[str, str]] = []
        for endpoint, payload in payloads.items():
            for row in CryptoWizardsExtractor.discover_fields(payload):
                row["endpoint"] = endpoint
                all_rows.append(row)
        atomic_write_csv(pd.DataFrame(all_rows).drop_duplicates(), output_path, index=False)


def _is_official_wizard_api_url(url: str) -> bool:
    parsed = urlsplit(url)
    official = urlsplit(BASE_URL)
    return (
        parsed.scheme == official.scheme
        and parsed.hostname == official.hostname
        and parsed.port == official.port
    )
