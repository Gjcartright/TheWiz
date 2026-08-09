from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pandas as pd

from quant_platform.youtube_hypothesis_validation import run_youtube_hypothesis_validation
from quant_platform.orchestration.mini_agents import build_mini_agent_orchestration


def _write_candidate(root) -> None:
    agents = root / "reports" / "agents"
    agents.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "hypothesis_id": "yth_test",
                "pair": "BNB-USD/OP-USD",
                "asset_x": "BNB-USD",
                "asset_y": "OP-USD",
                "wizard_exchange": "dydx",
                "timeframe": "Daily",
                "exact_mode": "Static (Spread)",
                "wizard_sharpe": 2.1,
                "wizard_returns_total_pct": 22.0,
                "wizard_discovery_pass": True,
            }
        ]
    ).to_csv(agents / "youtube_brain_hypotheses.csv", index=False)


def _write_history(root, *, start: datetime, rows: int = 80) -> None:
    detail = root / "data" / "raw" / "pair_details"
    detail.mkdir(parents=True)
    history = [
        {
            "timestamp": (start + timedelta(days=index)).isoformat(),
            "price_x": 500.0 + index,
            "price_y": 1.0 + index / 100.0,
        }
        for index in range(rows)
    ]
    (detail / "pair_bnb_op_hyperliquid_1d_derived_history.json").write_text(
        json.dumps({"asset_x": "BNB", "asset_y": "OP", "interval": "1d", "history": history}),
        encoding="utf-8",
    )


def _write_venue_reports(root, *, slippage_ready: bool) -> None:
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"pair": "BNB-USD/OP-USD", "funding_ready": True, "funding_coverage_pct": 100.0}]
    ).to_csv(active / "hyperliquid_funding_coverage.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BNB-USD/OP-USD",
                "cost_model_ready": True,
                "slippage_model_ready": slippage_ready,
                "slippage_samples_x": 12 if slippage_ready else 1,
                "slippage_samples_y": 12 if slippage_ready else 1,
                "required_slippage_samples": 12,
                "estimated_pair_round_trip_cost_bps": 18.0,
            }
        ]
    ).to_csv(active / "hyperliquid_pair_cost_model.csv", index=False)
    pd.DataFrame(
        [
            {"asset": "BNB", "tradable": True},
            {"asset": "OP", "tradable": True},
        ]
    ).to_csv(active / "hyperliquid_market_context.csv", index=False)
    pd.DataFrame(
        [
            {"asset": "BNB", "tradable_perp": True},
            {"asset": "OP", "tradable_perp": True},
        ]
    ).to_csv(active / "hyperliquid_testnet_market_inventory.csv", index=False)


def _write_scanner(root) -> None:
    active = root / "reports" / "active"
    raw = root / "data" / "raw" / "crypto_wizards_scanner" / "capture.json"
    raw.parent.mkdir(parents=True)
    raw.write_text(
        json.dumps(
            [
                {
                    "pair_id": 99,
                    "spread_id": 3,
                    "strategy_id": 1,
                    "hedge_ratio": 0.8,
                    "x_weighting": 0.5,
                    "y_weighting": 0.5,
                    "zscore_window": 20,
                    "period": 320,
                }
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "pair": "BNB-USD/OP-USD",
                "asset_x": "BNB-USD",
                "asset_y": "OP-USD",
                "pair_id": 99,
                "timeframe": "Daily",
                "exact_mode": "Static (Spread)",
                "period": 320,
                "entry_level": 2.0,
                "exit_level": 0.0,
                "raw_source_path": str(raw.relative_to(root)),
                "capture_timestamp_utc": "2026-08-06T12:00:00Z",
            }
        ]
    ).to_csv(active / "crypto_wizards_live_scanner_capture.csv", index=False)


def test_validation_prefills_observed_fields_but_refuses_unconfirmed_exact_mode(tmp_path):
    _write_candidate(tmp_path)
    _write_history(tmp_path, start=datetime(2025, 1, 1, tzinfo=timezone.utc), rows=150)
    _write_venue_reports(tmp_path, slippage_ready=False)
    _write_scanner(tmp_path)

    result = run_youtube_hypothesis_validation(
        root=tmp_path,
        as_of=datetime(2026, 8, 6, 13, tzinfo=timezone.utc),
    )
    validation = pd.read_csv(result.paths["validation_queue"])
    row = validation.iloc[0]
    capture = pd.read_csv(result.paths["settings_capture_queue"]).iloc[0]
    regime = pd.read_csv(result.paths["regime_matrix"])

    assert row["validation_status"] == "BLOCKED_MISSING_EXACT_SETTINGS"
    assert "wizard_exact_settings_not_live_confirmed" in row["blocker"]
    assert "hyperliquid_l2_slippage_not_calibrated" in row["blocker"]
    assert row["observed_hedge_ratio"] == 0.8
    assert not row["acceptance_eligible"]
    assert not row["trade_authorized"]
    assert capture["hedge_ratio"] == 0.8
    assert not capture["capture_confirmed"]
    assert len(regime) == 12
    assert set(regime["validation_status"]) == {"BLOCKED_PREREQUISITES"}


def test_complete_capture_stages_walk_forward_but_never_grants_acceptance(tmp_path):
    _write_candidate(tmp_path)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _write_history(tmp_path, start=start, rows=150)
    _write_venue_reports(tmp_path, slippage_ready=True)
    _write_scanner(tmp_path)
    active = tmp_path / "reports" / "active"
    capture_timestamp = start + timedelta(days=40)
    pd.DataFrame(
        [
            {
                "pair": "BNB-USD/OP-USD",
                "asset_x": "BNB-USD",
                "asset_y": "OP-USD",
                "exchange": "dydx",
                "interval": "daily",
                "period": 320,
                "exact_mode": "Static (Spread)",
                "capture_timestamp_utc": capture_timestamp.isoformat(),
                "capture_evidence_path": "reports/evidence/pair-page.png",
                "capture_confirmed": True,
                "pair_page_url": "https://cryptowizards.net/wizards/zscore/pair/99",
                "entry_long_operator": "<=",
                "entry_long_value": -2.0,
                "entry_long_position": "long_x_short_y",
                "entry_short_operator": ">=",
                "entry_short_value": 2.0,
                "entry_short_position": "short_x_long_y",
                "exit_long_operator": ">=",
                "exit_long_value": 0.0,
                "exit_short_operator": "<=",
                "exit_short_value": 0.0,
                "x_weighting": 0.5,
                "y_weighting": 0.5,
                "hedge_ratio": 0.8,
                "zscore_window": 20,
            }
        ]
    ).to_csv(active / "crypto_wizards_pair_page_capture_settings.csv", index=False)

    result = run_youtube_hypothesis_validation(
        root=tmp_path,
        as_of=datetime(2026, 8, 6, 13, tzinfo=timezone.utc),
    )
    row = pd.read_csv(result.paths["validation_queue"]).iloc[0]
    regime = pd.read_csv(result.paths["regime_matrix"])

    assert row["validation_status"] == "READY_FOR_COSTED_WALK_FORWARD"
    assert row["local_mode_replay_ready"]
    assert row["point_in_time_settings_ready"]
    assert row["costed_walk_forward_ready"]
    assert row["hyperliquid_testnet_pair_tradable"]
    assert not row["acceptance_eligible"]
    assert not row["trade_authorized"]
    assert set(regime["validation_status"]) == {"READY_TO_RUN"}
    assert regime["trades"].isna().all()


def test_validation_routes_exact_settings_and_slippage_to_separate_agents(tmp_path):
    _write_candidate(tmp_path)
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BNB-USD/OP-USD",
                "exact_mode": "Static (Spread)",
                "validation_status": "BLOCKED_MISSING_EXACT_SETTINGS",
                "slippage_model_ready": False,
                "blocker": "wizard_exact_settings_not_live_confirmed;hyperliquid_l2_slippage_not_calibrated",
            }
        ]
    ).to_csv(active / "youtube_hypothesis_validation_queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BNB-USD/OP-USD",
                "wizard_exchange": "dydx",
                "readiness_status": "ready_for_replay",
                "next_step": "run_dydx_exact_mode_replay",
            }
        ]
    ).to_csv(active / "multi_venue_history_readiness.csv", index=False)

    result = build_mini_agent_orchestration(root=tmp_path)
    queue = pd.read_csv(result.paths["next_action_queue"])

    pair_tasks = queue.loc[queue["pair"].eq("BNB-USD/OP-USD")]
    assert {
        "capture_youtube_hypothesis_exact_mode",
        "calibrate_hyperliquid_l2_slippage",
    }.issubset(set(pair_tasks["task_type"]))
    assert "run_dydx_exact_mode_replay" not in set(queue["next_step"])
