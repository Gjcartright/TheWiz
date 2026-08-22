from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd

from quant_platform.orchestration.current_wizard_hyperliquid_evidence_command_center import (
    build_current_wizard_hyperliquid_evidence_command_center,
)

NOW = datetime(2026, 8, 16, 12, tzinfo=UTC)


def test_command_center_separates_frozen_snapshot_from_rolling_coverage(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    selected = ["coinbase|daily|BTC|ETH", "coinbase|daily|ETH|SOL"]
    (active / "current_wizard_hyperliquid_cost_manifest.json").write_text(
        json.dumps(
            {
                "cost_evidence_id": "cwcost-test",
                "selected_pair_group_keys": selected,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            _pair_cost(selected[0], "BTC-USD-ETH-USD", "BTC", "ETH"),
            _pair_cost(selected[1], "ETH-USD-SOL-USD", "ETH", "SOL"),
        ]
    ).to_csv(active / "current_wizard_hyperliquid_pair_cost_evidence.csv", index=False)
    pd.DataFrame([_funding(asset) for asset in ("BTC", "ETH", "SOL")]).to_csv(
        active / "current_wizard_hyperliquid_funding_asset_results.csv", index=False
    )
    samples = []
    for asset in ("BTC", "ETH"):
        samples.extend(
            _sample(asset, NOW - timedelta(minutes=5 * offset))
            for offset in range(1, 13)
        )
    samples.extend(
        _sample("SOL", NOW - timedelta(hours=10, minutes=5 * offset))
        for offset in range(3)
    )
    pd.DataFrame(samples).to_csv(
        processed / "hyperliquid_l2_slippage_samples.csv", index=False
    )
    pd.DataFrame(
        [
            _failure("exp-1", selected[0], "handoff", "handoff_blocked", False),
            _failure("exp-2", selected[0], "hyperliquid_mapping", "mapping_blocked", False),
            _failure("exp-3", selected[1], "cost_evidence", "strict_l2_incomplete", False),
            _failure("exp-4", selected[1], "", "", True),
        ]
    ).to_csv(
        active / "current_wizard_hyperliquid_failure_attribution.csv", index=False
    )
    (active / "current_wizard_hyperliquid_failure_attribution_manifest.json").write_text(
        json.dumps({"failure_attribution_id": "failure-test"}), encoding="utf-8"
    )

    result = build_current_wizard_hyperliquid_evidence_command_center(
        root=tmp_path, now=NOW
    )
    pairs = pd.read_csv(result.paths["pairs"])
    assets = pd.read_csv(result.paths["assets"])
    funnel = pd.read_csv(result.paths["funnel"])
    validation = pd.read_csv(result.paths["validation"])

    assert result.summary["rolling_strict_ready_assets"] == 2
    assert result.summary["rolling_provisional_ready_assets"] == 3
    assert result.summary["rolling_strict_ready_pairs"] == 1
    assert result.summary["rolling_provisional_ready_pairs"] == 2
    assert result.summary["current_snapshot_strict_cost_ready_pairs"] == 0
    assert not pairs["current_snapshot_repairable_with_future_l2"].any()
    assert assets.set_index("asset").loc["SOL", "capture_due"]
    handoff = funnel.set_index("stage").loc["handoff"]
    cost = funnel.set_index("stage").loc["cost_evidence"]
    assert handoff["reached_stage"] == 4
    assert handoff["passed_stage"] == 3
    assert cost["reached_stage"] == 2
    assert cost["passed_stage"] == 1
    assert validation["status"].eq("PASS").all()
    assert result.summary["live_trading_authorized"] is False


def _pair_cost(key: str, pair: str, asset_x: str, asset_y: str) -> dict[str, object]:
    return {
        "pair_group_key": key,
        "pair": pair,
        "wizard_exchange": "coinbase",
        "timeframe": "daily",
        "asset_x": asset_x,
        "asset_y": asset_y,
        "selected_for_cost_evidence": True,
        "strict_l2_samples_x": 0,
        "strict_l2_samples_y": 0,
        "provisional_l2_samples_x": 0,
        "provisional_l2_samples_y": 0,
        "cost_evidence_status": "BLOCKED_COST_EVIDENCE",
        "cost_blocker": "strict_l2_calibration_incomplete",
        "cost_acceptance_ready": False,
        "evidence_path": "reports/snapshots/frozen/pair.csv",
    }


def _funding(asset: str) -> dict[str, object]:
    return {
        "asset": asset,
        "funding_rows": 1000,
        "latest_funding_at": (NOW - timedelta(hours=1)).isoformat(),
        "fetch_complete_flag": True,
        "funding_status": "COMPLETE",
    }


def _sample(asset: str, timestamp: datetime) -> dict[str, object]:
    return {
        "asset": asset,
        "notional_usd": 1000.0,
        "source_timestamp": timestamp.isoformat(),
        "one_way_slippage_bps": 1.0,
        "buy_complete": True,
        "sell_complete": True,
        "blocker": "",
    }


def _failure(
    experiment_id: str,
    pair_group_key: str,
    stage: str,
    blocker: str,
    accepted: bool,
) -> dict[str, object]:
    asset_x, asset_y = pair_group_key.split("|")[-2:]
    return {
        "experiment_id": experiment_id,
        "pair_group_key": pair_group_key,
        "pair": f"{asset_x}-USD-{asset_y}-USD",
        "wizard_exchange": "coinbase",
        "wizard_timeframe": "daily",
        "asset_x": asset_x,
        "asset_y": asset_y,
        "overall_research_rank": int(experiment_id[-1]),
        "first_blocking_stage": stage,
        "first_blocker": blocker,
        "next_action": "test",
        "execution_acceptance_ready": accepted,
        "evidence_path": "reports/active/failure.csv",
    }
