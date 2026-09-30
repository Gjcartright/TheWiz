from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from quant_platform.orchestration.corrective_wizard_browser_auth import (
    build_wizard_browser_auth_readiness,
    validate_wizard_browser_auth_observation,
    validate_wizard_browser_auth_readiness,
)

NOW = datetime(2026, 8, 13, 5, 0, tzinfo=UTC)


def _observation(route_kind: str) -> dict[str, object]:
    pair = route_kind == "pair_detail"
    final_url = (
        "https://cryptowizards.net/wizards/zscore/pair/6?origin=scanner"
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
    return {
        "schema_version": "wizard_browser_auth_observation.v1",
        "captured_at": "2026-08-13T04:55:00+00:00",
        "requested_url": final_url,
        "requested_url_source": "capture_argument",
        "final_url": final_url,
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


def _write_contract(root: Path) -> None:
    config = root / "config"
    config.mkdir(parents=True)
    (config / "wizard_browser_auth_contract.json").write_text(
        json.dumps(
            {
                "schema_version": "thewiz.wizard_browser_auth_contract.v1",
                "max_age_hours": 24,
                "required_route_kinds": ["scanner", "pair_detail"],
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )


def test_observation_requires_multiple_independent_auth_signals() -> None:
    result = validate_wizard_browser_auth_observation(
        _observation("pair_detail"), expected_route_kind="pair_detail", now=NOW
    )

    assert result["status"] == "PASS"
    assert result["authenticated_route_proven"] is True
    assert result["checks_passed"] == result["checks_total"]
    assert result["promotion_authority"] is False
    assert result["live_trading_authorized"] is False


def test_absent_sign_in_text_alone_cannot_prove_authentication() -> None:
    observation = _observation("pair_detail")
    observation["member_navigation_targets"] = []
    observation["protected_content_markers"] = []

    result = validate_wizard_browser_auth_observation(observation, now=NOW)

    assert result["status"] == "BLOCKED"
    assert "member_navigation_count" in result["blocker"]
    assert "route_specific_content" in result["blocker"]


def test_public_redirect_cannot_masquerade_as_authenticated_route() -> None:
    observation = _observation("scanner")
    observation["final_url"] = "https://cryptowizards.net/"

    result = validate_wizard_browser_auth_observation(observation, now=NOW)

    assert result["status"] == "BLOCKED"
    assert "final_route_protected" in result["blocker"]
    assert "declared_route_matches_final" in result["blocker"]


def test_fallback_current_location_is_not_explicit_requested_route_proof() -> None:
    observation = _observation("scanner")
    observation["requested_url_source"] = "current_location_fallback"

    result = validate_wizard_browser_auth_observation(observation, now=NOW)

    assert result["status"] == "BLOCKED"
    assert "requested_route_explicit" in result["blocker"]


def test_sign_in_or_verification_forms_block_authentication() -> None:
    for field in ("sign_in_form_present", "verification_form_present"):
        observation = _observation("scanner")
        observation[field] = True

        result = validate_wizard_browser_auth_observation(observation, now=NOW)

        assert result["status"] == "BLOCKED"
        assert field.replace("_present", "_absent") in result["blocker"]


def test_readiness_requires_fresh_independent_scanner_and_pair_sources(tmp_path: Path) -> None:
    _write_contract(tmp_path)
    raw = tmp_path / "data" / "raw" / "crypto_wizards" / "browser_auth"
    raw.mkdir(parents=True)
    scanner_path = raw / "scanner.json"
    pair_path = raw / "pair.json"
    scanner_path.write_text(json.dumps(_observation("scanner")), encoding="utf-8")
    pair_path.write_text(json.dumps(_observation("pair_detail")), encoding="utf-8")

    result = build_wizard_browser_auth_readiness(root=tmp_path, now=NOW)
    validation = validate_wizard_browser_auth_readiness(root=tmp_path, now=NOW)

    assert result.summary["status"] == "PASS_AUTHENTICATED_BROWSER_ROUTES"
    assert result.summary["authenticated_routes_ready"] is True
    assert len(result.summary["selected_evidence"]) == 2
    assert result.summary["valid_until_utc"] == "2026-08-14T04:55:00+00:00"
    assert validation["status"] == "PASS"
    assert validation["route_kinds"] == ["pair_detail", "scanner"]
    assert result.summary["testnet_order_authority"] is False


def test_readiness_blocks_missing_pair_route(tmp_path: Path) -> None:
    _write_contract(tmp_path)
    raw = tmp_path / "data" / "raw" / "crypto_wizards" / "browser_auth"
    raw.mkdir(parents=True)
    (raw / "scanner.json").write_text(json.dumps(_observation("scanner")), encoding="utf-8")

    result = build_wizard_browser_auth_readiness(root=tmp_path, now=NOW)

    assert result.summary["status"] == "BLOCKED"
    assert "fresh_authenticated_pair_detail_observation_missing" in result.summary["blockers"]
    assert "immutable_receipt" not in result.paths
    validation = validate_wizard_browser_auth_readiness(root=tmp_path, now=NOW)
    assert validation["status"] == "BLOCKED"
    assert validation["blocker"] == (
        "browser_auth_not_ready:fresh_authenticated_pair_detail_observation_missing"
    )


def test_readiness_blocks_stale_observation(tmp_path: Path) -> None:
    _write_contract(tmp_path)
    raw = tmp_path / "data" / "raw" / "crypto_wizards" / "browser_auth"
    raw.mkdir(parents=True)
    for route_kind in ("scanner", "pair_detail"):
        observation = _observation(route_kind)
        observation["captured_at"] = "2026-08-11T04:55:00+00:00"
        (raw / f"{route_kind}.json").write_text(json.dumps(observation), encoding="utf-8")

    result = build_wizard_browser_auth_readiness(root=tmp_path, now=NOW)

    assert result.summary["status"] == "BLOCKED"
    assert len(result.summary["selected_evidence"]) == 0


def test_readiness_is_valid_through_ttl_boundary_then_expires(tmp_path: Path) -> None:
    _write_contract(tmp_path)
    raw = tmp_path / "data" / "raw" / "crypto_wizards" / "browser_auth"
    raw.mkdir(parents=True)
    for route_kind in ("scanner", "pair_detail"):
        (raw / f"{route_kind}.json").write_text(
            json.dumps(_observation(route_kind)), encoding="utf-8"
        )
    build_wizard_browser_auth_readiness(root=tmp_path, now=NOW)

    at_boundary = validate_wizard_browser_auth_readiness(
        root=tmp_path,
        now=datetime(2026, 8, 14, 4, 55, tzinfo=UTC),
    )
    after_boundary = validate_wizard_browser_auth_readiness(
        root=tmp_path,
        now=datetime(2026, 8, 14, 4, 55, 1, tzinfo=UTC),
    )

    assert at_boundary["status"] == "PASS"
    assert after_boundary["status"] == "BLOCKED"
    assert after_boundary["blocker"] == ("invalid_wizard_browser_auth_readiness:ValueError")


def test_readiness_detects_source_tampering(tmp_path: Path) -> None:
    _write_contract(tmp_path)
    raw = tmp_path / "data" / "raw" / "crypto_wizards" / "browser_auth"
    raw.mkdir(parents=True)
    scanner_path = raw / "scanner.json"
    pair_path = raw / "pair.json"
    scanner_path.write_text(json.dumps(_observation("scanner")), encoding="utf-8")
    pair_path.write_text(json.dumps(_observation("pair_detail")), encoding="utf-8")
    build_wizard_browser_auth_readiness(root=tmp_path, now=NOW)

    tampered = deepcopy(_observation("scanner"))
    tampered["final_url"] = "https://cryptowizards.net/"
    scanner_path.write_text(json.dumps(tampered), encoding="utf-8")

    validation = validate_wizard_browser_auth_readiness(root=tmp_path, now=NOW)
    assert validation["status"] == "BLOCKED"
    assert validation["blocker"] == "invalid_wizard_browser_auth_readiness:ValueError"


def test_same_source_cannot_supply_both_required_routes(tmp_path: Path) -> None:
    _write_contract(tmp_path)
    path = tmp_path / "combined.json"
    path.write_text(
        json.dumps(
            {
                "browser_auth_observation": _observation("scanner"),
            }
        ),
        encoding="utf-8",
    )
    result = build_wizard_browser_auth_readiness(
        root=tmp_path,
        now=NOW,
        observation_paths=[path],
    )
    assert result.summary["status"] == "BLOCKED"


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("max_age_hours", 169),
        ("required_route_kinds", ["pair_detail", "scanner"]),
    ],
)
def test_readiness_rejects_self_consistent_forged_contract_fields(
    tmp_path: Path, field: str, forged_value: object
) -> None:
    _write_contract(tmp_path)
    raw = tmp_path / "data" / "raw" / "crypto_wizards" / "browser_auth"
    raw.mkdir(parents=True)
    for route_kind in ("scanner", "pair_detail"):
        (raw / f"{route_kind}.json").write_text(
            json.dumps(_observation(route_kind)), encoding="utf-8"
        )
    build_wizard_browser_auth_readiness(root=tmp_path, now=NOW)

    status_path = tmp_path / "reports/active/wizard_browser_auth_readiness.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    receipt_path = tmp_path / status["receipt_path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = forged_value
    status[field] = forged_value
    receipt_bytes = json.dumps(receipt, sort_keys=True).encode("utf-8")
    receipt_path.chmod(0o600)
    receipt_path.write_bytes(receipt_bytes)
    status["receipt_sha256"] = sha256(receipt_bytes).hexdigest()
    status_path.write_text(json.dumps(status), encoding="utf-8")

    validation = validate_wizard_browser_auth_readiness(root=tmp_path, now=NOW)
    assert validation["status"] == "BLOCKED"
    assert validation["blocker"] == "invalid_wizard_browser_auth_readiness:ValueError"
