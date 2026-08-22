"""Fail-closed provenance for authenticated Crypto Wizards browser captures."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_runtime import promote_staged_file

ROOT = Path(__file__).resolve().parents[3]
OBSERVATION_SCHEMA_VERSION = "wizard_browser_auth_observation.v1"
READINESS_SCHEMA_VERSION = "thewiz.wizard_browser_auth_readiness.v1"
CONTRACT_SCHEMA_VERSION = "thewiz.wizard_browser_auth_contract.v1"
DEFAULT_MAX_AGE_HOURS = 24.0
REQUIRED_ROUTE_KINDS = ("scanner", "pair_detail")
ALLOWED_HOSTS = {"cryptowizards.net", "www.cryptowizards.net"}
ROUTE_MARKERS = {
    "scanner": {
        "scanner_filter_controls",
        "scanner_strategy_control",
        "scanner_exchange_control",
        "scanner_results_surface",
    },
    "pair_detail": {
        "pair_mode_selector",
        "timeframe_selector",
        "ordered_asset_inputs",
        "rendered_asset_labels",
    },
}


def validate_wizard_browser_auth_observation(
    observation: object,
    *,
    expected_route_kind: str | None = None,
    now: datetime | None = None,
    max_age_hours: float | None = None,
) -> dict[str, Any]:
    """Validate independent protected-route and member-shell evidence."""

    payload = observation if isinstance(observation, dict) else {}
    checked_at = _as_utc(now)
    requested_url = _text(payload.get("requested_url"))
    final_url = _text(payload.get("final_url"))
    requested_kind = _route_kind(requested_url)
    final_kind = _route_kind(final_url)
    declared_kind = _text(payload.get("route_kind"))
    expected_kind = _text(expected_route_kind) or declared_kind
    captured_at = _parse_timestamp(payload.get("captured_at"))
    markers = {
        _text(value) for value in payload.get("protected_content_markers", []) if _text(value)
    }
    member_targets = _member_targets(payload.get("member_navigation_targets"))
    protected_targets = {target for target in member_targets if _route_kind(target) != ""}
    required_markers = ROUTE_MARKERS.get(final_kind, set())
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: object = "") -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "detail": _text(detail),
            }
        )

    check(
        "observation_schema",
        payload.get("schema_version") == OBSERVATION_SCHEMA_VERSION,
        payload.get("schema_version"),
    )
    check("capture_timestamp_valid", captured_at is not None, payload.get("captured_at"))
    if captured_at is not None:
        future_seconds = (captured_at - checked_at).total_seconds()
        check(
            "capture_not_materially_future_dated",
            future_seconds <= 300,
            f"future_seconds={future_seconds:.3f}",
        )
        if max_age_hours is not None:
            age_hours = max((checked_at - captured_at).total_seconds() / 3600.0, 0.0)
            check(
                "capture_fresh",
                age_hours <= max_age_hours,
                f"age_hours={age_hours:.6f};max={max_age_hours}",
            )
    check(
        "requested_route_explicit",
        payload.get("requested_url_source") == "capture_argument",
        payload.get("requested_url_source"),
    )
    check("requested_route_protected", bool(requested_kind), requested_url)
    check("final_route_protected", bool(final_kind), final_url)
    check(
        "declared_route_matches_final",
        bool(final_kind) and declared_kind == final_kind,
        f"declared={declared_kind};final={final_kind}",
    )
    check(
        "expected_route_matches_final",
        not expected_kind or expected_kind == final_kind,
        f"expected={expected_kind};final={final_kind}",
    )
    check(
        "member_navigation_count",
        len(protected_targets) >= 2,
        f"protected_targets={len(protected_targets)}",
    )
    check(
        "account_navigation_present",
        any(
            urlparse(target).path.rstrip("/") == "/wizards/account" for target in protected_targets
        ),
        ";".join(sorted(protected_targets)),
    )
    check(
        "route_specific_content",
        bool(required_markers) and required_markers.issubset(markers),
        "missing=" + ";".join(sorted(required_markers - markers)),
    )
    check(
        "sign_in_form_absent",
        payload.get("sign_in_form_present") is False,
        payload.get("sign_in_form_present"),
    )
    check(
        "verification_form_absent",
        payload.get("verification_form_present") is False,
        payload.get("verification_form_present"),
    )
    check(
        "public_marketing_shell_absent",
        payload.get("public_marketing_shell_present") is False,
        payload.get("public_marketing_shell_present"),
    )
    check(
        "browser_storage_not_accessed",
        payload.get("browser_storage_accessed") is False,
        payload.get("browser_storage_accessed"),
    )
    check(
        "credentials_excluded",
        payload.get("no_credentials_or_browser_storage_captured") is True,
        payload.get("no_credentials_or_browser_storage_captured"),
    )
    blockers = [row["check"] for row in checks if row["status"] != "PASS"]
    observation_hash = sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    observation_id = "wizardbrowserauth_" + observation_hash[:20]
    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "status": "PASS" if not blockers else "BLOCKED",
        "blocker": ";".join(blockers),
        "checks": checks,
        "checks_total": len(checks),
        "checks_passed": len(checks) - len(blockers),
        "observation_id": observation_id,
        "observation_sha256": observation_hash,
        "captured_at": captured_at.isoformat() if captured_at is not None else "",
        "requested_url": requested_url,
        "final_url": final_url,
        "route_kind": final_kind,
        "member_navigation_target_count": len(protected_targets),
        "authenticated_route_proven": not blockers,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def build_wizard_browser_auth_readiness(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    observation_paths: Iterable[Path] | None = None,
) -> CommandResult:
    """Build a source-hash-bound readiness receipt for scanner and pair routes."""

    checked_at = _as_utc(now)
    contract, contract_blocker = _read_contract(root)
    max_age_hours = float(contract.get("max_age_hours", DEFAULT_MAX_AGE_HOURS))
    required_route_kinds = tuple(contract.get("required_route_kinds", REQUIRED_ROUTE_KINDS))
    paths = list(observation_paths) if observation_paths is not None else _candidate_paths(root)
    assessments: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    for path in paths:
        relative = _relative(path, root)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            check_rows.append(
                {
                    "source_path": relative,
                    "route_kind": "",
                    "observation_id": "",
                    "check": "source_readable_json",
                    "status": "FAIL",
                    "detail": type(exc).__name__,
                }
            )
            continue
        for observation in _extract_observations(payload):
            assessment = validate_wizard_browser_auth_observation(
                observation,
                now=checked_at,
                max_age_hours=max_age_hours,
            )
            assessment["source_path"] = relative
            assessment["source_sha256"] = sha256(path.read_bytes()).hexdigest()
            assessments.append(assessment)
            for row in assessment["checks"]:
                check_rows.append(
                    {
                        "source_path": relative,
                        "route_kind": assessment["route_kind"],
                        "observation_id": assessment["observation_id"],
                        **row,
                    }
                )

    selected: list[dict[str, Any]] = []
    blockers: list[str] = []
    if contract_blocker:
        blockers.append(contract_blocker)
    for route_kind in required_route_kinds:
        eligible = [
            item
            for item in assessments
            if item["route_kind"] == route_kind and item["status"] == "PASS"
        ]
        eligible.sort(key=lambda item: (item["captured_at"], item["source_path"]), reverse=True)
        if not eligible:
            blockers.append(f"fresh_authenticated_{route_kind}_observation_missing")
        else:
            selected.append(eligible[0])
    selected_paths = [item["source_path"] for item in selected]
    if len(selected_paths) != len(set(selected_paths)):
        blockers.append("required_routes_not_independently_captured")
    selected_capture_times = [
        parsed
        for item in selected
        if (parsed := _parse_timestamp(item.get("captured_at"))) is not None
    ]
    valid_until = (
        min(selected_capture_times) + timedelta(hours=max_age_hours)
        if len(selected_capture_times) == len(required_route_kinds)
        else None
    )

    active = root / "reports" / "active"
    checks_path = active / "wizard_browser_auth_checks.csv"
    status_path = active / "wizard_browser_auth_readiness.json"
    markdown_path = active / "wizard_browser_auth_readiness.md"
    _atomic_csv(
        pd.DataFrame(
            check_rows,
            columns=[
                "source_path",
                "route_kind",
                "observation_id",
                "check",
                "status",
                "detail",
            ],
        ),
        checks_path,
    )
    selected_evidence = [
        {
            "route_kind": item["route_kind"],
            "captured_at": item["captured_at"],
            "requested_url": item["requested_url"],
            "final_url": item["final_url"],
            "observation_id": item["observation_id"],
            "observation_sha256": item["observation_sha256"],
            "source_path": item["source_path"],
            "source_sha256": item["source_sha256"],
        }
        for item in sorted(selected, key=lambda item: item["route_kind"])
    ]
    core: dict[str, Any] = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "status": "PASS_AUTHENTICATED_BROWSER_ROUTES" if not blockers else "BLOCKED",
        "authenticated_routes_ready": not blockers,
        "max_age_hours": max_age_hours,
        "required_route_kinds": list(required_route_kinds),
        "selected_evidence": selected_evidence,
        "valid_until_utc": valid_until.isoformat() if valid_until is not None else "",
        "blockers": sorted(set(blockers)),
        "research_only": True,
        "promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    if not blockers:
        receipt_id = (
            "wizardbrowserreadiness_"
            + sha256(_canonical_json(core).encode("utf-8")).hexdigest()[:20]
        )
        receipt = {**core, "receipt_id": receipt_id}
        receipt_path = (
            root / "data" / "research" / "wizard_browser_auth_readiness" / f"{receipt_id}.json"
        )
        _write_immutable_json(receipt, receipt_path)
        receipt_sha = sha256(receipt_path.read_bytes()).hexdigest()
        summary = {
            **receipt,
            "checked_at_utc": checked_at.isoformat(),
            "receipt_path": _relative(receipt_path, root),
            "receipt_sha256": receipt_sha,
            "checks_path": _relative(checks_path, root),
        }
    else:
        receipt_path = None
        summary = {
            **core,
            "receipt_id": "",
            "checked_at_utc": checked_at.isoformat(),
            "receipt_path": "",
            "receipt_sha256": "",
            "checks_path": _relative(checks_path, root),
        }
    _atomic_json(summary, status_path)
    _atomic_text(_markdown(summary), markdown_path)
    result_paths = {
        "checks": checks_path,
        "status": status_path,
        "summary": markdown_path,
    }
    if receipt_path is not None:
        result_paths["immutable_receipt"] = receipt_path
    return CommandResult(paths=result_paths, summary=summary)


def validate_wizard_browser_auth_readiness(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    status_path: Path | None = None,
) -> dict[str, Any]:
    """Verify the active pointer, immutable receipt, hashes, and freshness."""

    status_path = status_path or root / "reports/active/wizard_browser_auth_readiness.json"
    checked_at = _as_utc(now)
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("schema_version") != READINESS_SCHEMA_VERSION:
            raise ValueError("readiness schema mismatch")
        if status.get("status") != "PASS_AUTHENTICATED_BROWSER_ROUTES":
            blockers = status.get("blockers", [])
            if not isinstance(blockers, list) or not blockers:
                raise ValueError("authenticated browser routes not ready")
            return {
                "status": "BLOCKED",
                "blocker": "browser_auth_not_ready:"
                + ";".join(sorted(_text(value) for value in blockers if _text(value))),
            }
        if status.get("authenticated_routes_ready") is not True:
            raise ValueError("authenticated browser readiness false")
        receipt_path = _safe_existing_path(root, status.get("receipt_path"))
        if receipt_path is None:
            raise ValueError("receipt path invalid")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = dict(status)
        for key in ("checked_at_utc", "receipt_path", "receipt_sha256", "checks_path"):
            expected.pop(key, None)
        if receipt != expected:
            raise ValueError("active pointer receipt mismatch")
        if sha256(receipt_path.read_bytes()).hexdigest() != status.get("receipt_sha256"):
            raise ValueError("receipt hash mismatch")
        max_age_hours = float(receipt["max_age_hours"])
        route_kinds: set[str] = set()
        source_paths: set[str] = set()
        for item in receipt.get("selected_evidence", []):
            source_path = _safe_existing_path(root, item.get("source_path"))
            if source_path is None:
                raise ValueError("source path invalid")
            if sha256(source_path.read_bytes()).hexdigest() != item.get("source_sha256"):
                raise ValueError("source hash mismatch")
            captured_at = _parse_timestamp(item.get("captured_at"))
            if captured_at is None:
                raise ValueError("source timestamp invalid")
            age_hours = max((checked_at - captured_at).total_seconds() / 3600.0, 0.0)
            if age_hours > max_age_hours or (captured_at - checked_at).total_seconds() > 300:
                raise ValueError("source observation stale or future dated")
            route_kinds.add(_text(item.get("route_kind")))
            source_paths.add(_text(item.get("source_path")))
        if route_kinds != set(receipt.get("required_route_kinds", [])):
            raise ValueError("required route set mismatch")
        if len(source_paths) != len(route_kinds):
            raise ValueError("required routes not independently captured")
        if (
            receipt.get("promotion_authority") is not False
            or receipt.get("testnet_order_authority") is not False
            or receipt.get("live_trading_authorized") is not False
        ):
            raise ValueError("browser receipt granted prohibited authority")
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return {
            "status": "BLOCKED",
            "blocker": f"invalid_wizard_browser_auth_readiness:{type(exc).__name__}",
        }
    return {
        "status": "PASS",
        "blocker": "",
        "receipt_id": str(status["receipt_id"]),
        "receipt_path": _relative(receipt_path, root),
        "receipt_sha256": str(status["receipt_sha256"]),
        "route_kinds": sorted(route_kinds),
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _extract_observations(payload: object) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    if payload.get("schema_version") == OBSERVATION_SCHEMA_VERSION:
        return [payload]
    nested = payload.get("browser_auth_observation")
    return [nested] if isinstance(nested, dict) else []


def _candidate_paths(root: Path) -> list[Path]:
    patterns = ("data/raw/crypto_wizards/browser_auth/**/*.json",)
    return sorted(
        {
            path
            for pattern in patterns
            for path in root.glob(pattern)
            if path.is_file() and not path.is_symlink()
        }
    )


def _read_contract(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "config" / "wizard_browser_auth_contract.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != CONTRACT_SCHEMA_VERSION:
            raise ValueError("contract schema mismatch")
        max_age = float(payload.get("max_age_hours"))
        route_kinds = payload.get("required_route_kinds")
        if not (0 < max_age <= 168):
            raise ValueError("max_age_hours out of range")
        if route_kinds != list(REQUIRED_ROUTE_KINDS):
            raise ValueError("required route kinds mismatch")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {
            "max_age_hours": DEFAULT_MAX_AGE_HOURS,
            "required_route_kinds": list(REQUIRED_ROUTE_KINDS),
        }, "wizard_browser_auth_contract_invalid"
    return payload, ""


def _member_targets(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    targets: set[str] = set()
    for item in value:
        raw = item.get("href") if isinstance(item, dict) else item
        url = _text(raw)
        parsed = urlparse(url)
        if not parsed.scheme and url.startswith("/"):
            url = "https://cryptowizards.net" + url
        if _route_kind(url):
            targets.add(url)
    return targets


def _route_kind(value: object) -> str:
    try:
        parsed = urlparse(_text(value))
    except ValueError:
        return ""
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in ALLOWED_HOSTS:
        return ""
    path = parsed.path.rstrip("/") or "/"
    if path == "/wizards/zscore/scanner":
        return "scanner"
    if path.startswith("/wizards/zscore/pair/") and len(path.split("/")) == 5:
        return "pair_detail"
    if path == "/wizards/account":
        return "account"
    if path.startswith("/wizards/"):
        return "member"
    return ""


def _parse_timestamp(value: object) -> datetime | None:
    token = _text(value)
    if not token:
        return None
    try:
        parsed = datetime.fromisoformat(token)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _safe_existing_path(root: Path, relative: object) -> Path | None:
    token = _text(relative)
    if not token:
        return None
    candidate = (root / token).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    if not candidate.is_file() or candidate.is_symlink():
        return None
    return candidate


def _write_immutable_json(payload: dict[str, Any], path: Path) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"immutable receipt collision: {path}")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False)
    promote_staged_file(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", path)


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    promote_staged_file(temporary, path)


def _markdown(summary: dict[str, Any]) -> str:
    blockers = summary.get("blockers", [])
    evidence = summary.get("selected_evidence", [])
    return "\n".join(
        [
            "# Crypto Wizards Browser Authentication Readiness",
            "",
            f"- Status: `{summary['status']}`",
            f"- Checked at: `{summary['checked_at_utc']}`",
            f"- Max evidence age: `{summary['max_age_hours']}` hours",
            f"- Proven routes: `{', '.join(item['route_kind'] for item in evidence) or 'none'}`",
            f"- Receipt: `{summary.get('receipt_id') or 'not issued'}`",
            f"- Blockers: `{';'.join(blockers) or 'none'}`",
            "- Authority: `research evidence only; no promotion or order authority`",
            "",
        ]
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _as_utc(value: datetime | None) -> datetime:
    current = value or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(UTC)


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()
