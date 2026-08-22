from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from quant_platform.orchestration.exhaustive_wizard_hyperliquid_run import (
    EXACT_MODES,
    ORIENTATIONS,
    build_exhaustive_wizard_hyperliquid_run,
    build_exhaustive_wizard_hyperliquid_mapping_refresh,
)


def _write_inventory(root, assets):
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "asset": asset,
                "asset_index": index,
                "universe_name": asset,
                "max_leverage": 10,
                "only_isolated": False,
                "tradable_perp": True,
                "checked_at_utc": "2026-08-07T12:00:00+00:00",
                "fetch_blocker": "",
            }
            for index, asset in enumerate(assets)
        ]
    ).to_csv(active / "hyperliquid_testnet_market_inventory.csv", index=False)
    pd.DataFrame([{"status": "completed"}]).to_csv(
        active / "wizard_sweep_manifest.csv", index=False
    )
    return active


def test_exhaustive_run_keeps_low_score_duplicates_and_unavailable_pairs(tmp_path):
    active = _write_inventory(tmp_path, ["BTC", "ETH", "DOGE"])
    rows = [
        {
            "sweep_id": "sweep-1",
            "request_id": "request-1",
            "sweep_rank": 1,
            "sweep_exchange": "Dydx",
            "sweep_interval": "Daily",
            "sweep_captured_at": "2026-08-07T12:30:00+00:00",
            "sweep_complete": True,
            "symbol_1": "BTC-USD",
            "symbol_2": "ETH-USD",
            "sharpe": -1.25,
            "returns_total": -0.40,
            "sweep_evidence_path": "data/raw/request-1.json",
        },
        {
            "sweep_id": "sweep-1",
            "request_id": "request-2",
            "sweep_rank": 7,
            "sweep_exchange": "Dydx",
            "sweep_interval": "Daily",
            "sweep_complete": True,
            "symbol_1": "BTC-USD",
            "symbol_2": "ETH-USD",
            "sharpe": -1.25,
            "returns_total": -0.40,
            "sweep_evidence_path": "data/raw/request-2.json",
        },
        {
            "sweep_id": "sweep-1",
            "request_id": "request-3",
            "sweep_rank": 1,
            "sweep_exchange": "Binance",
            "sweep_interval": "Hourly",
            "sweep_complete": True,
            "symbol_1": "DOGEUSDT",
            "symbol_2": "NOPEUSDT",
            "sharpe": 9.5,
            "returns_total": 3.0,
            "sweep_evidence_path": "data/raw/request-3.json",
        },
    ]
    pd.DataFrame(rows).to_csv(active / "wizard_sweep_candidates.csv", index=False)

    result = build_exhaustive_wizard_hyperliquid_run(
        root=tmp_path,
        now=datetime(2026, 8, 7, 13, tzinfo=timezone.utc),
    )

    source = pd.read_csv(result.paths["source_row_ledger"])
    pairs = pd.read_csv(result.paths["pair_ledger"])
    detail_queue = pd.read_csv(result.paths["pair_detail_capture_queue"])
    matrix = pd.read_csv(result.paths["experiment_matrix"])
    validation = pd.read_csv(result.paths["coverage_validation"])
    manifest = json.loads(result.paths["manifest"].read_text(encoding="utf-8"))

    assert len(source) == len(rows)
    assert source["source_row_id"].nunique() == len(rows)
    assert not source["discovery_prefilter_applied"].astype(bool).any()
    assert source["sharpe"].min() == -1.25
    assert len(pairs) == 2
    assert pairs.loc[pairs["pair"].eq("BTC-USD-ETH-USD"), "source_row_count"].item() == 2
    blocked = pairs.loc[pairs["pair"].eq("DOGE-USD-NOPE-USD")].iloc[0]
    assert blocked["hyperliquid_pair_ready"] in (False, 0)
    assert "missing_hyperliquid_testnet_perp:NOPE" in blocked["hyperliquid_mapping_blocker"]
    assert len(detail_queue) == len(pairs)
    assert detail_queue["pair_group_id"].nunique() == len(pairs)
    assert detail_queue["capture_status"].eq("NOT_CAPTURED").all()
    assert (
        detail_queue.loc[
            detail_queue["pair"].eq("BTC-USD-ETH-USD"),
            "scanner_capture_timestamps",
        ].item()
        == "2026-08-07T12:30:00+00:00"
    )
    assert detail_queue["modes_to_capture"].str.contains("Copula").all()
    assert (
        detail_queue["capture_order_policy"]
        .eq("scanner_cell_first_then_hyperliquid_ready_no_rows_filtered")
        .all()
    )
    assert detail_queue["pair_presence_recheck_required"].astype(bool).all()
    assert (
        detail_queue["capture_cadence_policy"]
        .eq("capture_pair_detail_immediately_within_same_live_scanner_cell")
        .all()
    )
    assert not detail_queue["live_trading_authorized"].astype(bool).any()
    assert len(matrix) == len(pairs) * len(EXACT_MODES) * len(ORIENTATIONS)
    assert set(matrix["exact_mode"]) == set(EXACT_MODES)
    assert set(matrix["orientation"]) == set(ORIENTATIONS)
    assert not matrix["live_trading_authorized"].astype(bool).any()
    assert (
        validation.loc[validation["check"].eq("every_source_row_accounted"), "status"].item()
        == "PASS"
    )
    assert (
        validation.loc[validation["check"].eq("all_dashboard_pages_proven"), "status"].item()
        == "BLOCKED"
    )
    assert manifest["authority"] == "PARTIAL_CAPTURE_RESEARCH_ONLY"
    assert manifest["canonical_replay_leverage"] == 1.0
    assert manifest["live_trading_authorized"] is False
    snapshot = manifest["immutable_snapshot"]
    snapshot_dir = tmp_path / snapshot["directory"]
    assert snapshot_dir.name == manifest["run_id"]
    assert result.paths["snapshot_manifest"].exists()
    assert (tmp_path / snapshot["source"]).read_bytes() == (
        active / "wizard_sweep_candidates.csv"
    ).read_bytes()
    assert (tmp_path / snapshot["sweep_manifest"]).read_bytes() == (
        active / "wizard_sweep_manifest.csv"
    ).read_bytes()
    assert (tmp_path / snapshot["hyperliquid_inventory"]).read_bytes() == (
        active / "hyperliquid_testnet_market_inventory.csv"
    ).read_bytes()
    assert all((tmp_path / path).exists() for path in snapshot["artifacts"].values())


def test_exhaustive_authority_requires_complete_page_evidence(tmp_path):
    active = _write_inventory(tmp_path, ["BTC", "ETH"])
    pd.DataFrame(
        [
            {
                "sweep_id": "sweep-2",
                "request_id": "request-1",
                "sweep_rank": rank,
                "sweep_exchange": "Dydx",
                "sweep_interval": "Daily",
                "sweep_complete": True,
                "sweep_page": page,
                "sweep_total_pages": 2,
                "sweep_page_complete": True,
                "symbol_1": "BTC-USD",
                "symbol_2": "ETH-USD",
            }
            for rank, page in enumerate((1, 2), start=1)
        ]
    ).to_csv(active / "wizard_sweep_candidates.csv", index=False)

    result = build_exhaustive_wizard_hyperliquid_run(root=tmp_path)

    assert result.summary["all_dashboard_pages_proven"] is True
    assert result.summary["authority"] == "EXHAUSTIVE_RESEARCH_DEFINITION_READY"
    assert result.summary["planned_experiments"] == len(EXACT_MODES) * len(ORIENTATIONS)


def test_exhaustive_run_blocks_cross_quote_rows_that_collapse_to_one_perp(tmp_path):
    active = _write_inventory(tmp_path, ["ALGO"])
    pd.DataFrame(
        [
            {
                "sweep_id": "sweep-cross-quote",
                "request_id": "request-cross-quote",
                "sweep_exchange": "Coinbase",
                "sweep_interval": "Daily",
                "sweep_complete": True,
                "symbol_1": "ALGO-USD",
                "symbol_2": "ALGO-EUR",
            }
        ]
    ).to_csv(active / "wizard_sweep_candidates.csv", index=False)

    result = build_exhaustive_wizard_hyperliquid_run(root=tmp_path)

    source = pd.read_csv(result.paths["source_row_ledger"])
    pairs = pd.read_csv(result.paths["pair_ledger"])
    assert source["normalization_blocker"].eq("identical_canonical_assets").all()
    assert not pairs["hyperliquid_pair_ready"].astype(bool).any()
    assert pairs["hyperliquid_mapping_blocker"].str.contains("identical_canonical_assets").all()
    assert result.summary["hyperliquid_ready_pair_groups"] == 0

    refresh = build_exhaustive_wizard_hyperliquid_mapping_refresh(root=tmp_path)
    mapping = pd.read_csv(refresh.paths["mapping"])
    assert not mapping["current_pair_ready"].astype(bool).any()
    assert mapping["current_mapping_blocker"].str.contains("identical_canonical_assets").all()
    assert refresh.summary["current_ready_pair_groups"] == 0


def test_mapping_refresh_preserves_frozen_run_and_reports_stable_inventory(tmp_path):
    active = _write_inventory(tmp_path, ["BTC", "ETH"])
    pd.DataFrame(
        [
            {
                "sweep_id": "sweep-stable",
                "request_id": "request-stable",
                "sweep_exchange": "Dydx",
                "sweep_interval": "Daily",
                "sweep_complete": True,
                "symbol_1": "BTC-USD",
                "symbol_2": "ETH-USD",
            }
        ]
    ).to_csv(active / "wizard_sweep_candidates.csv", index=False)
    run = build_exhaustive_wizard_hyperliquid_run(root=tmp_path)
    frozen_inventory = Path(
        tmp_path / run.summary["immutable_snapshot"]["hyperliquid_inventory"]
    ).read_bytes()

    refresh = build_exhaustive_wizard_hyperliquid_mapping_refresh(root=tmp_path)

    mapping = pd.read_csv(refresh.paths["mapping"])
    drift = pd.read_csv(refresh.paths["inventory_drift"])
    assert refresh.summary["current_ready_pair_groups"] == 1
    assert refresh.summary["mapping_drift_pair_groups"] == 0
    assert refresh.summary["inventory_drift_rows"] == 0
    assert mapping["mapping_status"].eq("READY").all()
    assert not mapping["mapping_drift"].astype(bool).any()
    assert drift["drift_status"].eq("UNCHANGED").all()
    assert not mapping["live_trading_authorized"].astype(bool).any()
    assert refresh.paths["snapshot_inventory"].exists()
    assert (
        Path(tmp_path / run.summary["immutable_snapshot"]["hyperliquid_inventory"]).read_bytes()
        == frozen_inventory
    )


def test_mapping_refresh_exposes_market_removal_and_rule_drift(tmp_path):
    active = _write_inventory(tmp_path, ["BTC", "ETH"])
    pd.DataFrame(
        [
            {
                "sweep_id": "sweep-drift",
                "request_id": "request-drift",
                "sweep_exchange": "Dydx",
                "sweep_interval": "Daily",
                "sweep_complete": True,
                "symbol_1": "BTC-USD",
                "symbol_2": "ETH-USD",
            }
        ]
    ).to_csv(active / "wizard_sweep_candidates.csv", index=False)
    build_exhaustive_wizard_hyperliquid_run(root=tmp_path)
    pd.DataFrame(
        [
            {
                "asset": "BTC",
                "asset_index": 3,
                "universe_name": "BTC",
                "max_leverage": 40,
                "only_isolated": True,
                "tradable_perp": True,
                "checked_at_utc": "2026-08-08T05:00:00+00:00",
                "fetch_blocker": "",
            },
            {
                "asset": "SOL",
                "asset_index": 0,
                "universe_name": "SOL",
                "max_leverage": 10,
                "only_isolated": False,
                "tradable_perp": True,
                "checked_at_utc": "2026-08-08T05:00:00+00:00",
                "fetch_blocker": "",
            },
        ]
    ).to_csv(active / "hyperliquid_testnet_market_inventory.csv", index=False)

    refresh = build_exhaustive_wizard_hyperliquid_mapping_refresh(root=tmp_path)

    mapping = pd.read_csv(refresh.paths["mapping"]).iloc[0]
    drift = pd.read_csv(refresh.paths["inventory_drift"]).set_index("asset")
    assert mapping["mapping_status"] == "BLOCKED"
    assert bool(mapping["mapping_drift"])
    assert "pair_readiness_changed" in mapping["mapping_drift_reasons"]
    assert "missing_hyperliquid_testnet_perp:ETH" in mapping["current_mapping_blocker"]
    assert drift.loc["BTC", "drift_status"] == "CHANGED"
    assert drift.loc["ETH", "drift_status"] == "REMOVED"
    assert drift.loc["SOL", "drift_status"] == "ADDED"
    assert refresh.summary["inventory_drift_rows"] == 3
    assert refresh.summary["live_trading_authorized"] is False
