from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from quant_platform.orchestration.corrective_testnet_cohort_readiness import (
    build_testnet_prospective_cohort_readiness,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


def _write_sources(tmp_path) -> None:
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "experiment_id": "exp_eth_pyth",
                "pair_group_key": "dydx|daily|ETH|PYTH",
                "pair": "ETH-USD-PYTH-USD",
                "asset_x": "ETH",
                "asset_y": "PYTH",
                "registered_contract_candidate": True,
                "stage_two_candidate": True,
            }
        ]
    ).to_csv(active / "corrective_l2_capture_candidates.csv", index=False)
    pd.DataFrame(
        [
            {
                "market_type": "perp",
                "asset": "ETH",
                "asset_index": 4,
                "sz_decimals": 4,
                "max_leverage": 25,
                "is_delisted": False,
                "tradable_perp": True,
                "mid_price": 1885.25,
                "checked_at_utc": "2026-08-11T11:00:00+00:00",
                "fetch_blocker": "",
            },
            {
                "market_type": "perp",
                "asset": "PYTH",
                "asset_index": 61,
                "sz_decimals": 0,
                "max_leverage": 10,
                "is_delisted": False,
                "tradable_perp": True,
                "mid_price": 0.042382,
                "checked_at_utc": "2026-08-11T11:00:00+00:00",
                "fetch_blocker": "",
            },
        ]
    ).to_csv(active / "hyperliquid_testnet_market_inventory.csv", index=False)
    pd.DataFrame(
        [
            {
                "cost_model_id": "hlpaircost_test",
                "model_as_of_utc": "2026-08-11T11:30:00+00:00",
                "pair_group_key": "dydx|daily|ETH|PYTH",
                "strict_observed_cost_ready": True,
                "cost_model_status": "STRICT_OBSERVED",
                "funding_x_complete": True,
                "funding_y_complete": True,
            }
        ]
    ).to_csv(processed / "hyperliquid_pair_cost_models.csv", index=False)


def test_prospective_cohort_readiness_is_immutable_and_zero_authority(tmp_path):
    _write_sources(tmp_path)

    first = build_testnet_prospective_cohort_readiness(root=tmp_path, now=NOW)
    second = build_testnet_prospective_cohort_readiness(root=tmp_path, now=NOW)

    assert first.summary["status"] == "PASS_PROSPECTIVE_NO_AUTHORITY"
    assert first.summary["cohort_pairs"] == 1
    assert first.summary["prospectively_ready_pairs"] == 1
    assert first.paths["prospective_cohort_readiness_receipt"] == second.paths[
        "prospective_cohort_readiness_receipt"
    ]
    assert first.paths["prospective_cohort_readiness_receipt"].is_file()
    assert first.summary["candidate_selection_performed"] is False
    assert first.summary["collateral_transfer_attempted"] is False
    assert first.summary["order_submission_performed"] is False
    assert first.summary["testnet_order_authority"] is False
    assert first.summary["live_trading_authorized"] is False
    frame = pd.read_csv(first.paths["prospective_cohort_readiness"])
    row = frame.iloc[0]
    assert row["prospective_readiness_status"] == "PASS_PROSPECTIVE_NO_AUTHORITY"
    assert row["paired_size_precision_feasible"]
    assert 10.0 <= row["asset_x_indicative_notional_usd"] <= 12.5
    assert 10.0 <= row["asset_y_indicative_notional_usd"] <= 12.5
    assert row["indicative_pair_notional_usd"] <= 25.0
    assert row["asset_x_indicative_size"] == pytest.approx(0.0056)
    assert row["asset_y_indicative_size"] == 248


def test_missing_testnet_leg_blocks_without_granting_authority(tmp_path):
    _write_sources(tmp_path)
    inventory = tmp_path / "reports" / "active" / "hyperliquid_testnet_market_inventory.csv"
    frame = pd.read_csv(inventory)
    frame.loc[frame["asset"].eq("PYTH")].to_csv(inventory, index=False)

    result = build_testnet_prospective_cohort_readiness(root=tmp_path, now=NOW)

    assert result.summary["status"] == "BLOCKED"
    row = pd.read_csv(result.paths["prospective_cohort_readiness"]).iloc[0]
    assert "testnet_market_identity_missing_or_ambiguous:ETH" in row["blocker"]
    assert not row["testnet_order_authority"]


@pytest.mark.parametrize(
    ("column", "value", "blocker"),
    [
        (
            "checked_at_utc",
            "2026-08-10T10:00:00+00:00",
            "testnet_market_inventory_stale_or_future:ETH",
        ),
        (
            "checked_at_utc",
            "2026-08-11T13:00:00+00:00",
            "testnet_market_inventory_stale_or_future:ETH",
        ),
        ("sz_decimals", -1, "testnet_size_precision_missing:ETH"),
        ("mid_price", 0, "testnet_reference_price_missing:ETH"),
    ],
)
def test_invalid_market_evidence_blocks_fail_closed(
    tmp_path, column, value, blocker
):
    _write_sources(tmp_path)
    inventory = tmp_path / "reports" / "active" / "hyperliquid_testnet_market_inventory.csv"
    frame = pd.read_csv(inventory)
    frame.loc[frame["asset"].eq("ETH"), column] = value
    frame.to_csv(inventory, index=False)

    result = build_testnet_prospective_cohort_readiness(root=tmp_path, now=NOW)

    assert result.summary["status"] == "BLOCKED"
    row = pd.read_csv(result.paths["prospective_cohort_readiness"]).iloc[0]
    assert blocker in row["blocker"]
    assert not result.summary["candidate_promotion_authority"]


@pytest.mark.parametrize(
    ("column", "value", "blocker"),
    [
        (
            "model_as_of_utc",
            "2026-08-11T09:00:00+00:00",
            "strict_pair_cost_model_stale_or_unready",
        ),
        ("funding_y_complete", False, "funding_history_incomplete:PYTH"),
    ],
)
def test_stale_cost_or_incomplete_funding_blocks(tmp_path, column, value, blocker):
    _write_sources(tmp_path)
    cost = tmp_path / "data" / "processed" / "hyperliquid_pair_cost_models.csv"
    frame = pd.read_csv(cost)
    frame.loc[0, column] = value
    frame.to_csv(cost, index=False)

    result = build_testnet_prospective_cohort_readiness(root=tmp_path, now=NOW)

    assert result.summary["status"] == "BLOCKED"
    row = pd.read_csv(result.paths["prospective_cohort_readiness"]).iloc[0]
    assert blocker in row["blocker"]
    assert row["must_revalidate_after_candidate_selection"]


def test_immutable_receipt_tampering_is_rejected(tmp_path):
    _write_sources(tmp_path)
    result = build_testnet_prospective_cohort_readiness(root=tmp_path, now=NOW)
    path = result.paths["prospective_cohort_readiness_receipt"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["testnet_order_authority"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="receipt collision"):
        build_testnet_prospective_cohort_readiness(root=tmp_path, now=NOW)
