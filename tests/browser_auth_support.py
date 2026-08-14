from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path


def write_browser_auth_binding(
    root: Path,
    *,
    captured_at: datetime,
    required_at: datetime,
    max_age_hours: float = 24.0,
) -> dict[str, object]:
    captured_at = _as_utc(captured_at)
    required_at = _as_utc(required_at)
    valid_until = captured_at + timedelta(hours=max_age_hours)
    selected_evidence: list[dict[str, object]] = []
    for route_kind in ("pair_detail", "scanner"):
        pair = route_kind == "pair_detail"
        url = (
            "https://cryptowizards.net/wizards/zscore/pair/6"
            if pair
            else "https://cryptowizards.net/wizards/zscore/scanner"
        )
        markers = (
            [
                "pair_mode_selector",
                "timeframe_selector",
                "ordered_asset_inputs",
                "rendered_asset_labels",
            ]
            if pair
            else [
                "scanner_filter_controls",
                "scanner_strategy_control",
                "scanner_exchange_control",
                "scanner_results_surface",
            ]
        )
        observation = {
            "schema_version": "wizard_browser_auth_observation.v1",
            "captured_at": captured_at.isoformat(),
            "requested_url": url,
            "requested_url_source": "capture_argument",
            "final_url": url,
            "route_kind": route_kind,
            "member_navigation_targets": [
                "https://cryptowizards.net/wizards/account",
                "https://cryptowizards.net/wizards/zscore/scanner",
                "https://cryptowizards.net/wizards/zscore/trades",
            ],
            "protected_content_markers": markers,
            "sign_in_form_present": False,
            "verification_form_present": False,
            "public_marketing_shell_present": False,
            "browser_storage_accessed": False,
            "no_credentials_or_browser_storage_captured": True,
        }
        source_relative = f"data/raw/crypto_wizards/browser_auth/{route_kind}.json"
        source = root / source_relative
        _write_json(source, observation)
        observation_sha = sha256(_canonical_json(observation).encode("utf-8")).hexdigest()
        selected_evidence.append(
            {
                "route_kind": route_kind,
                "captured_at": captured_at.isoformat(),
                "requested_url": url,
                "final_url": url,
                "observation_id": "wizardbrowserauth_" + observation_sha[:20],
                "observation_sha256": observation_sha,
                "source_path": source_relative,
                "source_sha256": sha256(source.read_bytes()).hexdigest(),
            }
        )

    core = {
        "schema_version": "thewiz.wizard_browser_auth_readiness.v1",
        "status": "PASS_AUTHENTICATED_BROWSER_ROUTES",
        "authenticated_routes_ready": True,
        "max_age_hours": max_age_hours,
        "required_route_kinds": ["scanner", "pair_detail"],
        "selected_evidence": selected_evidence,
        "valid_until_utc": valid_until.isoformat(),
        "blockers": [],
        "research_only": True,
        "promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    receipt_id = (
        "wizardbrowserreadiness_" + sha256(_canonical_json(core).encode("utf-8")).hexdigest()[:20]
    )
    receipt = {**core, "receipt_id": receipt_id}
    receipt_relative = f"data/research/wizard_browser_auth_readiness/{receipt_id}.json"
    receipt_path = root / receipt_relative
    _write_json(receipt_path, receipt)
    receipt_sha = sha256(receipt_path.read_bytes()).hexdigest()
    return {
        "browser_auth_required_at_utc": required_at.isoformat(),
        "browser_auth_valid_at_capture_window": valid_until >= required_at,
        "browser_auth_valid_until_utc": valid_until.isoformat(),
        "browser_auth_receipt_id": receipt_id,
        "browser_auth_receipt_path": receipt_relative,
        "browser_auth_receipt_sha256": receipt_sha,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
