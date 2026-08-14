from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.crypto_wizards_sweep import build_wizard_sweep_cells
from quant_platform.orchestration.exhaustive_wizard_api_refresh import (
    build_exhaustive_wizard_api_refresh_delta,
)
from quant_platform.orchestration.corrective_wizard_browser_auth import (
    build_wizard_browser_auth_readiness,
)


def _auth_observation(route_kind: str) -> dict[str, object]:
    pair = route_kind == "pair_detail"
    url = (
        "https://cryptowizards.net/wizards/zscore/pair/6"
        if pair
        else "https://cryptowizards.net/wizards/zscore/scanner"
    )
    return {
        "schema_version": "wizard_browser_auth_observation.v1",
        "captured_at": "2026-08-08T12:04:00+00:00",
        "requested_url": url,
        "requested_url_source": "capture_argument",
        "final_url": url,
        "route_kind": route_kind,
        "member_navigation_targets": [
            "https://cryptowizards.net/wizards/account",
            "https://cryptowizards.net/wizards/zscore/scanner",
            "https://cryptowizards.net/wizards/zscore/trades",
        ],
        "protected_content_markers": (
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
        ),
        "sign_in_form_present": False,
        "verification_form_present": False,
        "public_marketing_shell_present": False,
        "browser_storage_accessed": False,
        "no_credentials_or_browser_storage_captured": True,
    }


def _write_browser_auth_readiness(root: Path) -> None:
    config = root / "config"
    config.mkdir(exist_ok=True)
    (config / "wizard_browser_auth_contract.json").write_text(
        json.dumps(
            {
                "schema_version": "thewiz.wizard_browser_auth_contract.v1",
                "max_age_hours": 24,
                "required_route_kinds": ["scanner", "pair_detail"],
            }
        ),
        encoding="utf-8",
    )
    raw = root / "data" / "raw" / "crypto_wizards" / "browser_auth"
    raw.mkdir(parents=True, exist_ok=True)
    paths = []
    for route_kind in ("scanner", "pair_detail"):
        path = raw / f"{route_kind}.json"
        path.write_text(json.dumps(_auth_observation(route_kind)), encoding="utf-8")
        paths.append(path)
    build_wizard_browser_auth_readiness(
        root=root,
        now=datetime(2026, 8, 8, 12, 5, tzinfo=timezone.utc),
        observation_paths=paths,
    )


def _write_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    active = root / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "asset": "BTC",
                "asset_index": 0,
                "universe_name": "BTC",
                "max_leverage": 20,
                "only_isolated": False,
                "tradable_perp": True,
                "checked_at_utc": "2026-08-08T12:00:00+00:00",
                "fetch_blocker": "",
            },
            {
                "asset": "ETH",
                "asset_index": 1,
                "universe_name": "ETH",
                "max_leverage": 10,
                "only_isolated": False,
                "tradable_perp": True,
                "checked_at_utc": "2026-08-08T12:00:00+00:00",
                "fetch_blocker": "",
            },
        ]
    ).to_csv(active / "hyperliquid_testnet_market_inventory.csv", index=False)
    sweep_id = "20260808T120000000000Z"
    candidates: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for index, cell in enumerate(build_wizard_sweep_cells(sweep_id=sweep_id)):
        evidence = root / "data" / "raw" / f"{cell.request_id}.json"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text("{}", encoding="utf-8")
        candidates.append(
            {
                "sweep_id": sweep_id,
                "request_id": cell.request_id,
                "sweep_rank": 1,
                "sweep_priority": cell.priority,
                "sweep_strategy": cell.strategy,
                "sweep_exchange": cell.exchange,
                "sweep_interval": cell.interval,
                "sweep_captured_at": "2026-08-08T12:00:00+00:00",
                "sweep_source_timestamp": "2026-08-08T11:59:00+00:00",
                "sweep_evidence_path": str(evidence.relative_to(root)),
                "sweep_complete": True,
                "discovery_authority": "complete_discovery",
                "symbol_1": "BTCUSDT",
                "symbol_2": "ETHUSDT",
                "pair_id": f"pair-{index}",
                "spread_id": index,
                "strategy": (
                    "copula"
                    if cell.strategy == "Copula"
                    else "zscore_roll"
                    if cell.strategy == "ZScoreRoll"
                    else "zscore"
                ),
                "spread_type": "static",
                "ou_optimal": index % 2 == 0,
                "sharpe": 1.0 + index / 100,
                "returns_total": 0.1 + index / 1000,
            }
        )
        manifest.append(
            {
                "request_id": cell.request_id,
                "status": "completed",
                "row_count": 1,
                "evidence_path": str(evidence.relative_to(root)),
            }
        )
    candidates_path = active / "wizard_sweep_candidates.csv"
    manifest_path = active / "wizard_sweep_manifest.csv"
    summary_path = active / "wizard_sweep_summary.json"
    pd.DataFrame(candidates).to_csv(candidates_path, index=False)
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "sweep_id": sweep_id,
                "sweep_complete": True,
                "discovery_authority": "complete_discovery",
                "completed_cells": 30,
                "candidate_rows": 30,
            }
        ),
        encoding="utf-8",
    )

    frozen_dir = (
        root
        / "reports"
        / "snapshots"
        / "exhaustive_wizard_hyperliquid"
        / "ewhl_test"
    )
    frozen_dir.mkdir(parents=True)
    frozen_path = frozen_dir / "pair_ledger.csv"
    pd.DataFrame(
        [
            {
                "pair_group_key": "binance|daily|BTC|ETH",
                "wizard_exchange": "binance",
                "timeframe": "daily",
                "source_row_count": 2,
                "observed_orientations": "BTC/ETH;ETH/BTC",
                "observed_sharpe_min": 1.1,
                "observed_sharpe_max": 1.5,
                "observed_returns_total_min": 0.10,
                "observed_returns_total_max": 0.20,
                "hyperliquid_pair_ready": True,
                "hyperliquid_mapping_blocker": "",
                "source_evidence_paths": "frozen/binance.csv",
            },
            {
                "pair_group_key": "dydx|daily|SOL|WLD",
                "wizard_exchange": "dydx",
                "timeframe": "daily",
                "source_row_count": 1,
                "observed_orientations": "SOL/WLD",
                "observed_sharpe_min": 2.0,
                "observed_sharpe_max": 2.0,
                "observed_returns_total_min": 0.30,
                "observed_returns_total_max": 0.30,
                "hyperliquid_pair_ready": True,
                "hyperliquid_mapping_blocker": "",
                "source_evidence_paths": "frozen/dydx.csv",
            },
        ]
    ).to_csv(frozen_path, index=False)
    exhaustive_manifest_path = active / "exhaustive_wizard_hyperliquid_run_manifest.json"
    exhaustive_manifest_path.write_text(
        json.dumps(
            {
                "run_id": "ewhl_test",
                "immutable_snapshot": {
                    "artifacts": {
                        "pair_ledger": str(frozen_path.relative_to(root)),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return candidates_path, manifest_path, summary_path, exhaustive_manifest_path


def test_refresh_accounts_for_every_row_and_queues_every_current_pair(tmp_path: Path) -> None:
    _write_inputs(tmp_path)

    result = build_exhaustive_wizard_api_refresh_delta(
        root=tmp_path,
        now=datetime(2026, 8, 8, 12, 5, tzinfo=timezone.utc),
    )

    assert result.summary["api_source_rows"] == 30
    assert result.summary["api_source_rows_accounted"] == 30
    assert result.summary["api_source_row_ids_unique"] == 30
    assert result.summary["ou_optimal_true_source_rows"] == 15
    assert result.summary["api_pair_groups"] == 10
    assert result.summary["frozen_pair_groups"] == 2
    assert result.summary["union_pair_groups"] == 11
    assert result.summary["matched_frozen_pairs"] == 1
    assert result.summary["new_api_discoveries"] == 9
    assert result.summary["missing_from_api_refresh"] == 1
    assert result.summary["pair_detail_queue_rows"] == 10
    assert result.summary["hyperliquid_pair_groups_mapped"] == 11
    assert result.summary["hyperliquid_ready_pair_groups"] == 10
    assert result.summary["hyperliquid_blocked_pair_groups"] == 1
    assert result.summary["current_api_hyperliquid_pair_groups_mapped"] == 10
    assert result.summary["current_api_hyperliquid_ready_pair_groups"] == 10
    assert result.summary["current_api_hyperliquid_blocked_pair_groups"] == 0
    assert result.summary["promotion_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    assert result.summary["pair_detail_browser_route_status"] == (
        "AUTHENTICATION_RECEIPT_NOT_PROVEN"
    )
    assert "OU (Optimal)" not in result.summary["exact_modes_required"]
    assert result.summary["scanner_overlays_required"] == ["OU (Optimal)"]

    accounting = pd.read_csv(result.paths["source_accounting"], keep_default_na=False)
    delta = pd.read_csv(result.paths["delta"], keep_default_na=False)
    queue = pd.read_csv(result.paths["pair_detail_queue"], keep_default_na=False)
    acquisition = pd.read_csv(
        result.paths["pair_detail_acquisition_plan"], keep_default_na=False
    )
    mapping = pd.read_csv(result.paths["hyperliquid_mapping"], keep_default_na=False)
    validation = pd.read_csv(result.paths["validation"], keep_default_na=False)
    assert len(accounting) == 30
    assert accounting["api_source_row_id"].nunique() == 30
    assert accounting["api_ou_optimal"].astype(bool).sum() == 15
    assert accounting["discovery_prefilter_applied"].eq(False).all()  # noqa: E712
    assert len(delta) == 11
    assert set(delta["membership_status"]) == {
        "MATCHED_FROZEN_PAIR",
        "NEW_API_DISCOVERY",
        "MISSING_FROM_API_REFRESH",
    }
    assert len(queue) == 10
    assert not queue["required_exact_modes"].str.contains(
        r"OU \(Optimal\)", regex=True
    ).any()
    assert queue["required_scanner_overlays"].eq("ou_optimal").all()
    assert queue["promotion_authority"].eq(False).all()  # noqa: E712
    assert queue["live_trading_authorized"].eq(False).all()  # noqa: E712
    assert len(acquisition) == 6
    assert len(mapping) == 11
    current_mapping = mapping[mapping["in_api_refresh"].astype(bool)]
    assert len(current_mapping) == 10
    assert current_mapping["hyperliquid_pair_ready"].eq(True).all()  # noqa: E712
    assert pd.to_numeric(
        current_mapping["hyperliquid_pair_max_leverage"], errors="coerce"
    ).eq(10.0).all()
    assert acquisition["execute_now"].eq(False).all()  # noqa: E712
    browser = acquisition.loc[
        acquisition["lane"].eq("authenticated_dashboard_pair_page")
    ].iloc[0]
    assert browser["full_queue_credits"] == 0
    assert browser["operational_capacity_pairs_now"] == 0
    get_bundle = acquisition.loc[
        acquisition["lane"].eq("documented_get_full_bundle")
    ].iloc[0]
    assert get_bundle["credits_per_pair"] == 174
    assert get_bundle["full_queue_credits"] == 1740
    assert validation["status"].eq("PASS").all()
    assert result.paths["snapshot_api_candidates"].exists()
    assert result.paths["snapshot_manifest"].exists()


def test_refresh_uses_fresh_authenticated_browser_inspector_status(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    _write_browser_auth_readiness(tmp_path)

    result = build_exhaustive_wizard_api_refresh_delta(
        root=tmp_path,
        now=datetime(2026, 8, 8, 12, 5, tzinfo=timezone.utc),
    )

    acquisition = pd.read_csv(
        result.paths["pair_detail_acquisition_plan"], keep_default_na=False
    )
    browser = acquisition.loc[
        acquisition["lane"].eq("authenticated_dashboard_pair_page")
    ].iloc[0]
    assert result.summary["pair_detail_browser_route_operational"] is True
    assert result.summary["pair_detail_browser_route_status"] == (
        "AVAILABLE_AUTHENTICATED_RECEIPT"
    )
    assert browser["operational_capacity_pairs_now"] == 10


def test_refresh_rejects_legacy_mutable_authenticated_boolean(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    status_path = tmp_path / "reports" / "active" / "crypto_wizards_inspector_status.json"
    status_path.write_text(
        json.dumps(
            {
                "observed_at": "2026-08-08T12:04:00+00:00",
                "authenticated": True,
                "pair_detail_route_status": "AVAILABLE_AUTHENTICATED_MANUAL_CAPTURE",
            }
        ),
        encoding="utf-8",
    )

    result = build_exhaustive_wizard_api_refresh_delta(
        root=tmp_path,
        now=datetime(2026, 8, 8, 12, 5, tzinfo=timezone.utc),
    )

    assert result.summary["pair_detail_browser_route_operational"] is False
    assert result.summary["pair_detail_browser_route_status"] == (
        "AUTHENTICATION_RECEIPT_NOT_PROVEN"
    )


def test_refresh_refuses_incomplete_sweep(tmp_path: Path) -> None:
    _, manifest_path, _, _ = _write_inputs(tmp_path)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    manifest.loc[0, "status"] = "failed"
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="incomplete request cell"):
        build_exhaustive_wizard_api_refresh_delta(root=tmp_path)


def test_refresh_refuses_silent_candidate_drop(tmp_path: Path) -> None:
    _, manifest_path, _, _ = _write_inputs(tmp_path)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    manifest.loc[0, "row_count"] = 2
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="candidate count does not match"):
        build_exhaustive_wizard_api_refresh_delta(root=tmp_path)


def test_refresh_retains_explicit_hyperliquid_mapping_blockers(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    inventory_path = (
        tmp_path / "reports" / "active" / "hyperliquid_testnet_market_inventory.csv"
    )
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    inventory = inventory[inventory["asset"].ne("ETH")]
    inventory.to_csv(inventory_path, index=False)

    result = build_exhaustive_wizard_api_refresh_delta(root=tmp_path)

    mapping = pd.read_csv(result.paths["hyperliquid_mapping"], keep_default_na=False)
    assert len(mapping) == 11
    assert mapping["hyperliquid_pair_ready"].eq(False).all()  # noqa: E712
    current_mapping = mapping[mapping["in_api_refresh"].astype(bool)]
    assert current_mapping["hyperliquid_mapping_blocker"].str.contains(
        "missing_hyperliquid_testnet_perp:ETH"
    ).all()
    assert mapping["hyperliquid_mapping_blocker"].str.len().gt(0).all()
    assert result.summary["hyperliquid_ready_pair_groups"] == 0
    assert result.summary["hyperliquid_blocked_pair_groups"] == 11
    assert result.summary["current_api_hyperliquid_blocked_pair_groups"] == 10
