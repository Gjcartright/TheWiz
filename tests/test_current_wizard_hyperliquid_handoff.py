from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.orchestration.current_wizard_hyperliquid_cadence import (
    build_current_wizard_hyperliquid_operating_cadence,
)
from quant_platform.orchestration.current_wizard_hyperliquid_concentration import (
    build_current_wizard_hyperliquid_concentration,
)
from quant_platform.orchestration.current_wizard_hyperliquid_costs import (
    materialize_current_wizard_hyperliquid_cost_evidence,
)
from quant_platform.orchestration.current_wizard_hyperliquid_failure_attribution import (
    build_current_wizard_hyperliquid_failure_attribution,
)
from quant_platform.orchestration.current_wizard_hyperliquid_handoff import (
    build_current_wizard_hyperliquid_handoff,
)
from quant_platform.orchestration.current_wizard_hyperliquid_learning import (
    build_current_wizard_hyperliquid_learning_ledger,
)
from quant_platform.orchestration.current_wizard_hyperliquid_leverage import (
    build_current_wizard_hyperliquid_leverage_surface,
)
from quant_platform.orchestration.current_wizard_hyperliquid_observed_replay import (
    run_current_wizard_hyperliquid_observed_cost_replay,
)
from quant_platform.orchestration.current_wizard_hyperliquid_regimes import (
    build_current_wizard_hyperliquid_regime_attribution,
)
from quant_platform.orchestration.current_wizard_hyperliquid_replay import (
    materialize_current_wizard_hyperliquid_history,
    run_current_wizard_hyperliquid_canonical_replay,
)
from quant_platform.orchestration.current_wizard_hyperliquid_robustness import (
    run_current_wizard_hyperliquid_robustness,
)
from quant_platform.orchestration.current_wizard_hyperliquid_storage import (
    build_current_wizard_hyperliquid_storage_reclamation_plan,
)
from quant_platform.orchestration.current_wizard_hyperliquid_validation import (
    validate_current_wizard_hyperliquid_chain,
)
from quant_platform.orchestration.current_wizard_hyperliquid_walkforward import (
    run_current_wizard_hyperliquid_walkforward,
)
from quant_platform.orchestration.snapshot_lineage import (
    verified_snapshot_reference,
)

READY_PAIR = "binance|daily|ETH|WIF"
BLOCKED_PAIR = "binance|daily|AAA|BBB"


def _write_inputs(root: Path) -> None:
    active = root / "reports" / "active"
    active.mkdir(parents=True)
    (active / "exhaustive_wizard_api_refresh_manifest.json").write_text(
        json.dumps(
            {
                "refresh_id": "ewapi_test",
                "discovery_policy": "exhaustive_no_prefilter",
                "api_source_rows": 3,
                "api_source_rows_accounted": 3,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"check": "fixture_refresh_complete", "status": "PASS"}]
    ).to_csv(active / "exhaustive_wizard_api_refresh_validation.csv", index=False)
    pd.DataFrame(
        [
            {
                "queue_position": 1,
                "pair_group_key": READY_PAIR,
                "pair": "ETH-USD-WIF-USD",
                "wizard_exchange": "binance",
                "timeframe": "daily",
                "asset_a": "ETH",
                "asset_b": "WIF",
                "api_pair_ids": "1",
                "api_exact_modes_observed": "Copula;Dyn (ZScoreR)",
                "api_observed_orientations": "ETH/WIF",
                "api_captured_at": "2026-08-08T11:09:20Z",
                "api_sharpe_min": 0.1,
                "api_sharpe_max": 0.5,
                "api_returns_total_min": 0.02,
                "api_returns_total_max": 0.12,
            },
            {
                "queue_position": 2,
                "pair_group_key": BLOCKED_PAIR,
                "pair": "AAA-USD-BBB-USD",
                "wizard_exchange": "binance",
                "timeframe": "daily",
                "asset_a": "AAA",
                "asset_b": "BBB",
                "api_pair_ids": "2",
                "api_exact_modes_observed": "Static (Spread)",
                "api_observed_orientations": "AAA/BBB",
                "api_captured_at": "2026-08-08T11:09:10Z",
                "api_sharpe_min": -1.0,
                "api_sharpe_max": -1.0,
                "api_returns_total_min": -0.2,
                "api_returns_total_max": -0.2,
            },
        ]
    ).to_csv(
        active / "exhaustive_wizard_api_refresh_pair_detail_queue.csv", index=False
    )
    pd.DataFrame(
        [
            {"pair_group_key": READY_PAIR, "api_exact_mode": "Copula"},
            {"pair_group_key": READY_PAIR, "api_exact_mode": "Dyn (ZScoreR)"},
            {"pair_group_key": BLOCKED_PAIR, "api_exact_mode": "Static (Spread)"},
        ]
    ).to_csv(
        active / "exhaustive_wizard_api_refresh_source_accounting.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "pair_group_key": READY_PAIR,
                "hyperliquid_market_a": "ETH#4",
                "hyperliquid_market_b": "WIF#78",
                "hyperliquid_pair_ready": True,
                "hyperliquid_pair_max_leverage": 5.0,
                "hyperliquid_only_isolated": False,
                "hyperliquid_inventory_checked_at": "2026-08-08T10:11:20Z",
                "hyperliquid_mapping_blocker": "",
            },
            {
                "pair_group_key": BLOCKED_PAIR,
                "hyperliquid_market_a": "",
                "hyperliquid_market_b": "",
                "hyperliquid_pair_ready": False,
                "hyperliquid_pair_max_leverage": "",
                "hyperliquid_only_isolated": False,
                "hyperliquid_inventory_checked_at": "2026-08-08T10:11:20Z",
                "hyperliquid_mapping_blocker": (
                    "missing_hyperliquid_testnet_perp:AAA;"
                    "missing_hyperliquid_testnet_perp:BBB"
                ),
            },
        ]
    ).to_csv(
        active / "exhaustive_wizard_api_refresh_hyperliquid_mapping.csv",
        index=False,
    )
    (active / "wizard_pair_detail_api_pilot_manifest.json").write_text(
        json.dumps(
            {
                "pilot_id": "pilot_test",
                "pair_group_key": READY_PAIR,
                "pilot_complete": True,
                "completed_endpoints": 6,
                "observed_fields": 85,
                "coverage_passes": 11,
                "coverage_checks": 15,
                "ecm_fields_found": False,
                "dashboard_pair_detail_complete": False,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"endpoint": name, "status": "COMPLETED"} for name in range(6)]
    ).to_csv(active / "wizard_pair_detail_api_pilot_manifest.csv", index=False)
    pd.DataFrame(
        [
            {"field_group": "backtest_return", "status": "FOUND"},
            {"field_group": "ecm_x", "status": "MISSING"},
            {"field_group": "ecm_y", "status": "MISSING"},
            {"field_group": "ecm_strength", "status": "MISSING"},
            {
                "field_group": "dashboard_entry_exit_controls",
                "status": "MISSING",
            },
        ]
    ).to_csv(active / "wizard_pair_detail_api_pilot_coverage.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair_group_key": READY_PAIR,
                "capture_status": "COMPLETE",
                "captured_planned_cells": 14,
                "pair_detail_evidence_paths": "historical.json",
            }
        ]
    ).to_csv(
        active / "exhaustive_wizard_pair_detail_capture_progress.csv", index=False
    )


def test_current_handoff_accounts_all_pairs_modes_orientations_and_blockers(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)

    result = build_current_wizard_hyperliquid_handoff(
        root=tmp_path,
        now=datetime(2026, 8, 8, 12, 30, tzinfo=timezone.utc),
    )

    pairs = pd.read_csv(result.paths["pair_status"], keep_default_na=False)
    experiments = pd.read_csv(result.paths["experiments"], keep_default_na=False)
    history = pd.read_csv(result.paths["pair_history_queue"], keep_default_na=False)
    assets = pd.read_csv(result.paths["asset_fetch_queue"], keep_default_na=False)
    validation = pd.read_csv(result.paths["validation"], keep_default_na=False)

    assert len(pairs) == 2
    assert pairs["pair_group_key"].nunique() == 2
    ready = pairs[pairs["pair_group_key"].eq(READY_PAIR)].iloc[0]
    blocked = pairs[pairs["pair_group_key"].eq(BLOCKED_PAIR)].iloc[0]
    assert ready["api_schema_pilot_pair"]
    assert ready["vendor_pair_detail_status"] == (
        "API_ANALYTICS_COMPLETE_DASHBOARD_FIELDS_MISSING"
    )
    assert "missing_ecm_x" in ready["vendor_pair_detail_blocker"]
    assert ready["local_research_status"] == "READY_FOR_POINT_IN_TIME_HISTORY"
    assert blocked["local_research_status"] == "BLOCKED_HYPERLIQUID_MAPPING"
    assert blocked["local_research_blocker"]

    assert len(experiments) == 32
    assert experiments["experiment_id"].nunique() == 32
    counts = experiments["experiment_status"].value_counts().to_dict()
    assert counts["READY_FOR_POINT_IN_TIME_HISTORY"] == 14
    assert counts["NOT_APPLICABLE_VENDOR_MODE"] == 2
    assert counts["BLOCKED_HYPERLIQUID_MAPPING"] == 16
    assert len(history) == 2
    assert history["history_request_status"].eq("READY_TO_FETCH").sum() == 1
    assert len(assets) == 2
    assert validation["status"].eq("PASS").all()
    assert result.summary["pair_groups_accounted"] == 2
    assert result.summary["planned_experiments"] == 32
    assert result.summary["promotion_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    assert Path(result.paths["snapshot_manifest"]).exists()


def test_completed_api_pilot_does_not_block_a_later_board_where_pair_is_absent(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    active = tmp_path / "reports" / "active"
    manifest_path = active / "wizard_pair_detail_api_pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pair_group_key"] = "binance|daily|OLD|PAIR"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = build_current_wizard_hyperliquid_handoff(
        root=tmp_path,
        now=datetime(2026, 8, 9, 12, 30, tzinfo=timezone.utc),
    )
    pairs = pd.read_csv(result.paths["pair_status"], keep_default_na=False)
    validation = pd.read_csv(result.paths["validation"], keep_default_na=False)

    assert not pairs["api_schema_pilot_pair"].astype(bool).any()
    check = validation.loc[
        validation["check"].eq("api_pilot_overlay_applied_when_pair_present")
    ].iloc[0]
    assert check["status"] == "PASS"
    assert "pilot_pair_on_current_board=False" in check["evidence"]


def _correlated_candle_fetcher():
    rng = np.random.default_rng(712)
    rows = 900
    log_wif = -2.0 + np.cumsum(rng.normal(0.0, 0.012, rows))
    residual = np.zeros(rows)
    for index in range(1, rows):
        residual[index] = 0.78 * residual[index - 1] + rng.normal(0.0, 0.012)
    log_eth = 8.0 + 1.20 * log_wif + residual
    prices = {"ETH": np.exp(log_eth), "WIF": np.exp(log_wif)}

    def fetcher(*, coin, interval, days, output_dir, end_time):
        del days
        timestamps = pd.date_range(end=end_time, periods=rows, freq="1D", tz="UTC")
        candles = [
            {
                "startedAt": timestamp.isoformat(),
                "ticker": f"{coin}-USD",
                "resolution": interval,
                "open": float(price),
                "high": float(price * 1.01),
                "low": float(price * 0.99),
                "close": float(price),
                "baseVolume": 100_000.0,
                "usdVolume": float(price * 100_000.0),
                "source": "hyperliquid",
            }
            for timestamp, price in zip(timestamps, prices[coin])
        ]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{coin}_{interval}_candles.json"
        output.write_text(
            json.dumps({"source": "hyperliquid", "candles": candles}),
            encoding="utf-8",
        )
        return output

    return fetcher


def test_bounded_current_history_and_replay_account_for_every_cell(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    build_current_wizard_hyperliquid_handoff(
        root=tmp_path,
        now=datetime(2026, 8, 8, 12, 30, tzinfo=timezone.utc),
    )

    history = materialize_current_wizard_hyperliquid_history(
        root=tmp_path,
        pair_group_keys=(READY_PAIR,),
        now=datetime(2026, 8, 8, 13, 0, tzinfo=timezone.utc),
        available_disk_bytes=2 * 1024 * 1024 * 1024,
        fetcher=_correlated_candle_fetcher(),
        sleep=lambda _: None,
    )
    pairs = pd.read_csv(history.paths["pair_results"], keep_default_na=False)
    assets = pd.read_csv(history.paths["asset_results"], keep_default_na=False)
    ready = pairs.loc[pairs["pair_group_key"].eq(READY_PAIR)].iloc[0]
    blocked = pairs.loc[pairs["pair_group_key"].eq(BLOCKED_PAIR)].iloc[0]

    assert len(pairs) == 2
    assert len(assets) == 2
    assert assets["history_status"].eq("COMPLETE").all()
    assert ready["history_status"] == "READY_FOR_CANONICAL_1X_REPLAY"
    assert ready["history_rows"] == 900
    assert ready["post_cutoff_rows"] == 0
    assert blocked["history_status"] == "BLOCKED_HANDOFF_REQUEST"
    assert history.summary["network_scope_explicit_and_bounded"] is True
    assert history.summary["live_trading_authorized"] is False

    replay = run_current_wizard_hyperliquid_canonical_replay(
        root=tmp_path,
        now=datetime(2026, 8, 8, 13, 30, tzinfo=timezone.utc),
    )
    results = pd.read_csv(replay.paths["results"], keep_default_na=False)
    validation = pd.read_csv(replay.paths["validation"], keep_default_na=False)
    ready_results = results.loc[results["pair_group_key"].eq(READY_PAIR)]
    completed = ready_results.loc[
        ready_results["replay_status"].eq("RESEARCH_REPLAY_COMPLETE")
    ]

    assert len(results) == 32
    assert results["experiment_id"].nunique() == 32
    assert len(completed) == 14
    assert ready_results["replay_status"].eq(
        "NOT_APPLICABLE_VENDOR_MODE"
    ).sum() == 2
    assert results.loc[results["pair_group_key"].eq(BLOCKED_PAIR), "replay_status"].eq(
        "BLOCKED_HYPERLIQUID_MAPPING"
    ).all()
    assert completed["train_rows"].eq(630).all()
    assert completed["test_rows"].eq(270).all()
    assert completed["canonical_replay_leverage"].eq(1.0).all()
    assert completed["acceptance_status"].eq("BLOCKED").all()
    assert not results["live_trading_authorized"].astype(bool).any()
    assert validation["status"].eq("PASS").all()
    assert replay.summary["acceptance_eligible_replays"] == 0
    assert replay.summary["replay_status_count_total"] == 32
    assert sum(replay.summary["replay_status_counts"].values()) == 32

    static = completed.loc[
        completed["exact_mode"].eq("Static (ZScoreR)")
        & completed["orientation"].eq("original")
    ].iloc[0]
    dynamic = completed.loc[
        completed["exact_mode"].eq("Dyn (ZScoreR)")
        & completed["orientation"].eq("original")
    ].iloc[0]
    reverse_static = completed.loc[
        completed["exact_mode"].eq("Static (ZScoreR)")
        & completed["orientation"].eq("reverse")
    ].iloc[0]
    static_settings = json.loads(static["settings_json"])
    dynamic_settings = json.loads(dynamic["settings_json"])
    assert static_settings["entry_long_position"] == "short_x_long_y"
    assert static_settings["entry_short_position"] == "long_x_short_y"
    assert dynamic_settings["entry_long_position"] == "long_x_short_y"
    assert dynamic_settings["entry_short_position"] == "short_x_long_y"
    assert not np.isclose(float(static["hedge_ratio"]), float(reverse_static["hedge_ratio"]))


def test_bounded_history_storage_gate_fetches_nothing_and_keeps_accounting(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    build_current_wizard_hyperliquid_handoff(root=tmp_path)
    calls: list[str] = []

    def forbidden_fetcher(**kwargs):
        calls.append(str(kwargs.get("coin")))
        raise AssertionError("storage preflight must prevent network fetches")

    history = materialize_current_wizard_hyperliquid_history(
        root=tmp_path,
        pair_group_keys=(READY_PAIR,),
        available_disk_bytes=1,
        fetcher=forbidden_fetcher,
    )
    pairs = pd.read_csv(history.paths["pair_results"], keep_default_na=False)
    assets = pd.read_csv(history.paths["asset_results"], keep_default_na=False)

    assert calls == []
    assert len(pairs) == 2
    assert pairs.loc[pairs["pair_group_key"].eq(READY_PAIR), "history_status"].iloc[0] == (
        "BLOCKED_STORAGE_PREFLIGHT"
    )
    assert assets["history_status"].eq("BLOCKED_STORAGE_PREFLIGHT").all()
    assert history.summary["storage_preflight_passed"] is False


def test_current_cost_evidence_is_exact_pair_bounded_and_fully_accounted(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    build_current_wizard_hyperliquid_handoff(
        root=tmp_path,
        now=datetime(2026, 8, 8, 12, 30, tzinfo=timezone.utc),
    )
    materialize_current_wizard_hyperliquid_history(
        root=tmp_path,
        pair_group_keys=(READY_PAIR,),
        now=datetime(2026, 8, 8, 13, 0, tzinfo=timezone.utc),
        available_disk_bytes=2 * 1024 * 1024 * 1024,
        fetcher=_correlated_candle_fetcher(),
        sleep=lambda _: None,
    )
    run_current_wizard_hyperliquid_canonical_replay(
        root=tmp_path,
        now=datetime(2026, 8, 8, 13, 30, tzinfo=timezone.utc),
    )
    config = tmp_path / "config"
    config.mkdir(parents=True)
    (config / "hyperliquid_perp_cost_profile.json").write_text(
        json.dumps(
            {
                "profile_id": "test",
                "taker_fee_bps": 4.5,
                "execution_risk_bps": 2.0,
                "fee_source_url": "https://example.test/fees",
                "fee_source_checked_at": "2026-08-05T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    cutoff = pd.Timestamp("2026-08-08T11:09:20Z")
    l2_rows = []
    for asset, slippage in (("ETH", 0.7), ("WIF", 3.5)):
        for index, timestamp in enumerate(
            pd.date_range(end=cutoff, periods=12, freq="10min", tz="UTC")
        ):
            for duplicate_pair in ("ETH-USD-WIF-USD", "OTHER-USD-WIF-USD"):
                l2_rows.append(
                    {
                        "sample_id": f"{asset}-{index}-{duplicate_pair}",
                        "pair": duplicate_pair,
                        "asset": asset,
                        "notional_usd": 1000.0,
                        "source_timestamp": timestamp.isoformat(),
                        "one_way_slippage_bps": slippage + index / 100.0,
                        "buy_complete": True,
                        "sell_complete": True,
                        "blocker": "",
                    }
                )
    pd.DataFrame(l2_rows).to_csv(
        processed / "hyperliquid_l2_slippage_samples.csv", index=False
    )

    cached_dir = tmp_path / "reports" / "snapshots" / "funding_cache"
    cached_dir.mkdir(parents=True)
    cached_eth = cached_dir / "ETH_funding.json"
    cached_records = [
        {
            "coin": "ETH",
            "time": int(timestamp.timestamp() * 1000),
            "fundingRate": 0.00001,
        }
        for timestamp in pd.date_range(end=cutoff, periods=900, freq="1D", tz="UTC")
    ]
    cached_eth.write_text(
        json.dumps(
            {"coin": "ETH", "funding": cached_records, "fetch_complete": True}
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "asset": "ETH",
                "funding_status": "COMPLETE",
                "funding_path": str(cached_eth.relative_to(tmp_path)),
                "fetch_end_at": cutoff.isoformat(),
            }
        ]
    ).to_csv(
        tmp_path
        / "reports"
        / "active"
        / "exhaustive_wizard_hyperliquid_funding_asset_results.csv",
        index=False,
    )
    network_calls: list[str] = []

    def funding_fetcher(*, coin, days, output_dir, end_time):
        del days
        network_calls.append(coin)
        timestamps = pd.date_range(end=end_time, periods=900, freq="1D", tz="UTC")
        records = [
            {
                "coin": coin,
                "time": int(timestamp.timestamp() * 1000),
                "fundingRate": 0.00001,
            }
            for timestamp in timestamps
        ]
        output = Path(output_dir) / f"{coin}_funding.json"
        output.write_text(
            json.dumps({"coin": coin, "funding": records, "fetch_complete": True}),
            encoding="utf-8",
        )
        return output

    result = materialize_current_wizard_hyperliquid_cost_evidence(
        root=tmp_path,
        pair_group_keys=(READY_PAIR,),
        now=datetime(2026, 8, 8, 14, 0, tzinfo=timezone.utc),
        funding_fetcher=funding_fetcher,
    )
    pairs = pd.read_csv(result.paths["pairs"], keep_default_na=False)
    experiments = pd.read_csv(result.paths["experiments"], keep_default_na=False)
    assets = pd.read_csv(result.paths["assets"], keep_default_na=False)
    validation = pd.read_csv(result.paths["validation"], keep_default_na=False)
    ready = pairs.loc[pairs["pair_group_key"].eq(READY_PAIR)].iloc[0]
    blocked = pairs.loc[pairs["pair_group_key"].eq(BLOCKED_PAIR)].iloc[0]

    assert len(pairs) == 2
    assert len(experiments) == 32
    assert len(assets) == 2
    assert network_calls == ["WIF"]
    assert assets.set_index("asset").loc["ETH", "funding_source"] == (
        "reused_point_in_time_cache"
    )
    assert assets.set_index("asset").loc["WIF", "funding_source"] == (
        "hyperliquid_public_api"
    )
    assert assets["post_cutoff_rows"].eq(0).all()
    assert ready["strict_l2_samples_x"] == 12
    assert ready["strict_l2_samples_y"] == 12
    assert ready["strict_slippage_ready"]
    assert ready["funding_both_coverage"] == 1.0
    assert ready["cost_evidence_status"] == "STRICT_COST_EVIDENCE_READY"
    assert ready["cost_acceptance_ready"]
    assert blocked["cost_evidence_status"] == "DEFERRED_NOT_SELECTED"
    assert experiments["cost_replay_status"].eq(
        "READY_FOR_OBSERVED_COST_RESEARCH"
    ).sum() == 14
    assert validation["status"].eq("PASS").all()
    assert result.summary["pair_groups_accounted"] == 2
    assert result.summary["experiments_accounted"] == 32
    assert result.summary["funding_assets_reused_from_cache"] == 1
    assert result.summary["funding_assets_fetched_from_network"] == 1
    assert result.summary["live_trading_authorized"] is False

    replay = run_current_wizard_hyperliquid_observed_cost_replay(
        root=tmp_path,
        now=datetime(2026, 8, 8, 14, 30, tzinfo=timezone.utc),
    )
    replay_rows = pd.read_csv(replay.paths["results"], keep_default_na=False)
    replay_validation = pd.read_csv(replay.paths["validation"], keep_default_na=False)
    complete = replay_rows.loc[
        replay_rows["replay_status"].eq("OBSERVED_COST_RESEARCH_REPLAY_COMPLETE")
    ]
    assert len(replay_rows) == 32
    assert len(complete) == 14
    assert complete["funding_evidence_attached"].all()
    assert complete["slippage_evidence_attached"].all()
    assert complete["funding_policy"].eq("signed_realized").all()
    assert complete["acceptance_status"].eq("BLOCKED").all()
    assert replay_validation["status"].eq("PASS").all()
    assert replay.summary["acceptance_eligible_replays"] == 0
    assert replay.summary["live_trading_authorized"] is False

    walkforward = run_current_wizard_hyperliquid_walkforward(
        root=tmp_path,
        now=datetime(2026, 8, 8, 15, 0, tzinfo=timezone.utc),
    )
    walk_status = pd.read_csv(walkforward.paths["status"], keep_default_na=False)
    walk_candidates = pd.read_csv(
        walkforward.paths["candidates"], keep_default_na=False
    )
    walk_folds = pd.read_csv(walkforward.paths["folds"], keep_default_na=False)
    walk_validation = pd.read_csv(
        walkforward.paths["validation"], keep_default_na=False
    )
    assert len(walk_status) == 32
    assert len(walk_candidates) == 14
    assert len(walk_folds) == 70
    assert walk_folds.groupby("experiment_id").size().eq(5).all()
    assert walk_folds["embargo_rows"].eq(20).all()
    assert not walk_folds["fit_uses_test_data"].astype(bool).any()
    assert walk_folds["open_trades"].eq(0).all()
    assert walk_status["acceptance_status"].eq("BLOCKED").all()
    assert not walk_status["live_trading_authorized"].astype(bool).any()
    assert walk_validation["status"].eq("PASS").all()
    assert walkforward.summary["experiments_accounted"] == 32
    assert walkforward.summary["folds_complete"] == 70
    assert walkforward.summary["selection_hindsight_used_for_fold_execution"] is False

    regimes = build_current_wizard_hyperliquid_regime_attribution(
        root=tmp_path,
        now=datetime(2026, 8, 8, 15, 30, tzinfo=timezone.utc),
    )
    regime_status = pd.read_csv(regimes.paths["status"], keep_default_na=False)
    regime_bars = pd.read_csv(regimes.paths["bars"], keep_default_na=False)
    regime_trades = pd.read_csv(regimes.paths["trades"], keep_default_na=False)
    regime_validation = pd.read_csv(
        regimes.paths["validation"], keep_default_na=False
    )
    assert len(regime_status) == 32
    assert regime_status["acceptance_status"].eq("BLOCKED").all()
    assert not regime_status["live_trading_authorized"].astype(bool).any()
    if not regime_bars.empty:
        assert not regime_bars["regime_uses_future_data"].astype(bool).any()
    if not regime_trades.empty:
        assert not regime_trades["regime_uses_future_data"].astype(bool).any()
    assert regime_validation["status"].eq("PASS").all()
    assert regimes.summary["experiments_accounted"] == 32
    assert regimes.summary["point_in_time_regime_thresholds"] is True

    robustness = run_current_wizard_hyperliquid_robustness(
        root=tmp_path,
        now=datetime(2026, 8, 8, 16, 0, tzinfo=timezone.utc),
    )
    robust_status = pd.read_csv(robustness.paths["status"], keep_default_na=False)
    robust_candidates = pd.read_csv(
        robustness.paths["candidates"], keep_default_na=False
    )
    robust_scenarios = pd.read_csv(
        robustness.paths["scenarios"], keep_default_na=False
    )
    robust_folds = pd.read_csv(robustness.paths["folds"], keep_default_na=False)
    robust_validation = pd.read_csv(
        robustness.paths["validation"], keep_default_na=False
    )
    assert len(robust_status) == 32
    assert len(robust_scenarios) == len(robust_candidates) * 11
    assert len(robust_folds) == len(robust_scenarios) * 5
    if not robust_candidates.empty:
        assert robust_candidates["baseline_reconciliation_error"].le(1e-10).all()
    assert not robust_folds.get(
        "fit_uses_test_data", pd.Series(dtype=bool)
    ).astype(bool).any()
    assert robust_status["acceptance_status"].eq("BLOCKED").all()
    assert not robust_status["live_trading_authorized"].astype(bool).any()
    assert robust_validation["status"].eq("PASS").all()
    assert robustness.summary["experiments_accounted"] == 32

    concentration = build_current_wizard_hyperliquid_concentration(
        root=tmp_path,
        now=datetime(2026, 8, 8, 16, 15, tzinfo=timezone.utc),
    )
    concentration_status = pd.read_csv(
        concentration.paths["status"], keep_default_na=False
    )
    concentration_cohorts = pd.read_csv(
        concentration.paths["cohorts"], keep_default_na=False
    )
    concentration_validation = pd.read_csv(
        concentration.paths["validation"], keep_default_na=False
    )
    assert len(concentration_status) == 32
    assert concentration_status["experiment_id"].nunique() == 32
    assert len(concentration_cohorts) == 3
    assert concentration_status["acceptance_status"].eq("BLOCKED").all()
    assert not concentration_status["live_trading_authorized"].astype(bool).any()
    assert concentration_validation["status"].eq("PASS").all()
    assert concentration.summary["experiments_accounted"] == 32
    assert concentration.summary["live_trading_authorized"] is False
    assert set(concentration.summary["input_snapshot_modes"].values()) == {
        "verified_upstream_reference"
    }
    assert all(
        "/inputs/" not in path
        for path in concentration.summary["input_snapshots"].values()
    )
    assert concentration.summary["referenced_upstream_bytes"] > 0
    assert concentration.summary["locally_copied_input_bytes"] == 0

    attribution = build_current_wizard_hyperliquid_failure_attribution(
        root=tmp_path,
        now=datetime(2026, 8, 8, 16, 30, tzinfo=timezone.utc),
    )
    attribution_rows = pd.read_csv(
        attribution.paths["attribution"], keep_default_na=False
    )
    attribution_routes = pd.read_csv(
        attribution.paths["route_index"], keep_default_na=False
    )
    attribution_validation = pd.read_csv(
        attribution.paths["validation"], keep_default_na=False
    )
    assert len(attribution_rows) == 32
    assert len(attribution_routes) == 32
    assert attribution_routes["experiment_id"].is_unique
    assert set(attribution_routes.columns) == {
        "experiment_id",
        "pair_group_key",
        "pair",
        "asset_x",
        "asset_y",
        "overall_research_rank",
    }
    assert attribution_rows["experiment_id"].nunique() == 32
    assert attribution_rows["first_blocker"].astype(str).ne("").all()
    assert attribution_rows["next_action"].astype(str).ne("").all()
    survivors = attribution_rows["one_x_research_survivor"].astype(bool)
    assert (
        ~survivors
        | attribution_rows["walkforward_status"].eq("PASS_RESEARCH_WALK_FORWARD")
    ).all()
    assert not attribution_rows["testnet_order_authority"].astype(bool).any()
    assert attribution_rows["acceptance_status"].eq("BLOCKED").all()
    assert not attribution_rows["live_trading_authorized"].astype(bool).any()
    assert attribution_validation["status"].eq("PASS").all()
    assert attribution.summary["experiments_accounted"] == 32
    assert set(attribution.summary["input_snapshot_modes"].values()) == {
        "verified_upstream_reference"
    }
    assert all(
        "/failure_attribution/" not in path
        for path in attribution.summary["input_snapshots"].values()
    )
    assert attribution.summary["referenced_upstream_bytes"] > 0
    assert attribution.summary["locally_copied_input_bytes"] == 0
    assert attribution.summary["schema_version"].endswith(".v1")
    assert "route_index" not in attribution.summary["artifacts"]
    assert json.loads(
        attribution.paths["manifest"].read_text(encoding="utf-8")
    ) == attribution.summary
    route_manifest = json.loads(
        attribution.paths["route_manifest"].read_text(encoding="utf-8")
    )
    assert route_manifest["route_index_rows"] == 32
    assert route_manifest["route_index_unique_experiment_ids"] == 32
    assert route_manifest["source_failure_attribution_id"] == attribution.summary[
        "failure_attribution_id"
    ]
    assert route_manifest["route_index_sha256"] == sha256(
        attribution.paths["route_index"].read_bytes()
    ).hexdigest()
    assert route_manifest["testnet_order_authority"] is False
    assert route_manifest["live_trading_authorized"] is False

    leverage = build_current_wizard_hyperliquid_leverage_surface(
        root=tmp_path,
        now=datetime(2026, 8, 8, 17, 0, tzinfo=timezone.utc),
    )
    leverage_status = pd.read_csv(leverage.paths["status"], keep_default_na=False)
    leverage_candidates = pd.read_csv(
        leverage.paths["candidates"], keep_default_na=False
    )
    leverage_scenarios = pd.read_csv(
        leverage.paths["scenarios"], keep_default_na=False
    )
    leverage_validation = pd.read_csv(
        leverage.paths["validation"], keep_default_na=False
    )
    assert len(leverage_status) == 32
    assert leverage_status["experiment_id"].nunique() == 32
    assert len(leverage_scenarios) == len(leverage_candidates) * 240
    assert leverage_status["acceptance_status"].eq("BLOCKED").all()
    assert not leverage_status["testnet_order_authority"].astype(bool).any()
    assert not leverage_status["live_trading_authorized"].astype(bool).any()
    assert leverage_validation["status"].eq("PASS").all()
    assert leverage.summary["experiments_accounted"] == 32
    assert set(leverage.summary["input_snapshot_modes"].values()) == {
        "verified_upstream_reference",
        "local_external_evidence_copy",
    }
    assert leverage.summary["input_snapshot_modes"]["failure_attribution"] == (
        "verified_upstream_reference"
    )
    local_external_inputs = [
        name
        for name, mode in leverage.summary["input_snapshot_modes"].items()
        if mode == "local_external_evidence_copy"
    ]
    assert local_external_inputs
    assert leverage.summary["referenced_upstream_bytes"] > 0
    assert leverage.summary["locally_copied_input_bytes"] > 0

    learning = build_current_wizard_hyperliquid_learning_ledger(
        root=tmp_path,
        now=datetime(2026, 8, 8, 17, 15, tzinfo=timezone.utc),
    )
    learning_rows = pd.read_csv(learning.paths["ledger"], keep_default_na=False)
    learning_validation = pd.read_csv(
        learning.paths["validation"], keep_default_na=False
    )
    assert len(learning_rows) == 32
    assert learning_rows["experiment_id"].nunique() == 32
    assert learning_rows["learning_record_id"].nunique() == 32
    assert learning_rows["backtest_label"].astype(str).ne("").all()
    assert learning_rows["paper_label"].astype(str).eq("").all()
    assert learning_rows["live_label"].astype(str).eq("").all()
    assert not learning_rows["training_eligible"].astype(bool).any()
    assert not learning_rows["order_submission_performed"].astype(bool).any()
    assert not learning_rows["live_trading_authorized"].astype(bool).any()
    assert learning_validation["status"].eq("PASS").all()
    assert learning.summary["records"] == 32
    assert learning.summary["training_eligible_records"] == 0
    assert set(learning.summary["input_snapshot_modes"].values()) == {
        "verified_upstream_reference"
    }
    assert all(
        "/learning/" not in path
        for path in learning.summary["input_snapshots"].values()
    )
    assert learning.summary["referenced_upstream_bytes"] > 0
    assert learning.summary["locally_copied_input_bytes"] == 0

    active = tmp_path / "reports" / "active"
    pd.DataFrame(
        [
            {
                "ready_for_no_order_preflight": True,
                "submit_orders_enabled": False,
                "blockers": "",
            }
        ]
    ).to_csv(active / "hyperliquid_testnet_preflight.csv", index=False)
    pd.DataFrame(
        [
            {
                "status": "BLOCKED",
                "blockers": "testnet_usdc_requires_spot_to_perp_transfer",
            }
        ]
    ).to_csv(active / "hyperliquid_testnet_margin_snapshot.csv", index=False)
    pd.DataFrame(
        [{"status": "BLOCKED", "execution_allowed": False}]
    ).to_csv(active / "hyperliquid_testnet_lifecycle_gate.csv", index=False)

    handoff_manifest = json.loads(
        (active / "current_wizard_hyperliquid_handoff_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    (active / "current_wizard_ou_optimal_overlay_manifest.json").write_text(
        json.dumps(
            {
                "overlay_id": "cwouoverlay_test",
                "refresh_id": "ewapi_test",
                "handoff_id": handoff_manifest["handoff_id"],
                "scanner_overlay": "ou_optimal",
                "pair_page_exact_modes": [
                    "Static (Spread)",
                    "Static (ZScoreR)",
                    "Dyn (Spread)",
                    "Dyn (ZScoreR)",
                    "OU (Spread)",
                    "OU (ZScoreR)",
                    "Copula",
                ],
                "source_rows_accounted": 3,
                "ou_optimal_true_rows": 1,
                "ou_optimal_false_rows": 2,
                "pair_page_ou_optimal_captured_rows": 0,
                "acceptance_authority": False,
                "promotion_authority": False,
                "order_submission_performed": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"check": "fixture_overlay_complete", "status": "PASS"}]
    ).to_csv(active / "current_wizard_ou_optimal_overlay_validation.csv", index=False)

    chain = validate_current_wizard_hyperliquid_chain(
        root=tmp_path,
        now=datetime(2026, 8, 8, 17, 30, tzinfo=timezone.utc),
    )
    chain_validation = pd.read_csv(
        chain.paths["validation"], keep_default_na=False
    )
    assert chain_validation["status"].eq("PASS").all()
    assert chain.summary["chain_status"] == "PASS"
    assert chain.summary["stages_complete"] == 14
    assert chain.summary["experiment_authority_count"] == 32
    assert chain.summary["trade_eligibility_status"] == (
        "BLOCKED_NO_ONE_X_RESEARCH_SURVIVOR"
    )
    assert chain.summary["testnet_no_order_preflight_ready"] is True
    assert chain.summary["testnet_submit_orders_enabled"] is False
    assert chain.summary["order_submission_performed"] is False
    assert chain.summary["live_trading_authorized"] is False

    cadence = build_current_wizard_hyperliquid_operating_cadence(
        root=tmp_path,
        now=datetime(2026, 8, 8, 17, 45, tzinfo=timezone.utc),
        available_disk_bytes=4 * 1024**3,
        wizard_api_key_present=True,
    )
    cadence_rows = pd.read_csv(cadence.paths["cadence"], keep_default_na=False)
    live_lock = pd.read_csv(cadence.paths["live_lock"], keep_default_na=False)
    storage_efficiency = pd.read_csv(
        cadence.paths["storage_efficiency"], keep_default_na=False
    )
    cadence_validation = pd.read_csv(
        cadence.paths["validation"], keep_default_na=False
    )
    assert len(cadence_rows) == 19
    assert cadence_rows["sequence"].tolist() == list(range(1, 20))
    assert not cadence_rows["scheduled_order_submission"].astype(bool).any()
    assert not cadence_rows["testnet_order_authority"].astype(bool).any()
    overlay_stage = cadence_rows.loc[
        cadence_rows["stage"].eq("ou_optimal_outcome_stratification")
    ].iloc[0]
    assert overlay_stage["sequence"] == 11
    assert "build-current-wizard-ou-optimal-overlay" in overlay_stage["command"]
    assert not bool(overlay_stage["promotion_authority"])
    leverage_stage = cadence_rows.loc[
        cadence_rows["stage"].eq("conditional_leverage_surface")
    ].iloc[0]
    assert leverage_stage["run_status"] == "BLOCKED"
    assert leverage_stage["blocker"] == "no_one_x_research_survivor"
    assert live_lock.iloc[0]["lock_state"] == "PERMANENT_RESEARCH_ONLY"
    assert not bool(live_lock.iloc[0]["unlock_authority_in_this_pipeline"])
    assert cadence_validation["status"].eq("PASS").all()
    assert len(storage_efficiency) == 4
    assert storage_efficiency["referenced_upstream_bytes"].astype(int).gt(0).all()
    assert not storage_efficiency["live_trading_authorized"].astype(bool).any()
    assert cadence.summary["recursive_copy_bytes_avoided"] > 0
    assert cadence.summary["daily_research_run_ready"] is True
    assert cadence.summary["order_submission_performed"] is False
    assert cadence.summary["live_trading_authorized"] is False

    leverage_status.iloc[:-1].to_csv(
        tmp_path
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_leverage_status.csv",
        index=False,
    )
    with pytest.raises(ValueError, match="accounting mismatch"):
        build_current_wizard_hyperliquid_learning_ledger(root=tmp_path)
    with pytest.raises(
        ValueError,
        match="leverage_status_active_snapshot_parity",
    ):
        validate_current_wizard_hyperliquid_chain(root=tmp_path)


def test_verified_snapshot_reference_fails_closed_for_invalid_lineage(
    tmp_path: Path,
) -> None:
    active = tmp_path / "reports" / "active" / "evidence.csv"
    snapshot = tmp_path / "reports" / "snapshots" / "run" / "evidence.csv"
    active.parent.mkdir(parents=True)
    snapshot.parent.mkdir(parents=True)
    active.write_text("value\n1\n", encoding="utf-8")
    snapshot.write_text("value\n1\n", encoding="utf-8")
    manifest = {
        "artifacts": {
            "snapshot_data": str(snapshot.relative_to(tmp_path)),
        }
    }

    assert verified_snapshot_reference(
        root=tmp_path,
        active_path=active,
        upstream_manifest=manifest,
        artifact_key="data",
    ) == snapshot

    active.write_text("value\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifacts differ"):
        verified_snapshot_reference(
            root=tmp_path,
            active_path=active,
            upstream_manifest=manifest,
            artifact_key="data",
        )

    active.write_text("value\n1\n", encoding="utf-8")
    snapshot.unlink()
    with pytest.raises(FileNotFoundError, match="reference is missing"):
        verified_snapshot_reference(
            root=tmp_path,
            active_path=active,
            upstream_manifest=manifest,
            artifact_key="data",
        )

    outside = tmp_path / "outside.csv"
    outside.write_text("value\n1\n", encoding="utf-8")
    manifest["artifacts"]["snapshot_data"] = str(outside.relative_to(tmp_path))
    with pytest.raises(ValueError, match="outside immutable snapshot storage"):
        verified_snapshot_reference(
            root=tmp_path,
            active_path=active,
            upstream_manifest=manifest,
            artifact_key="data",
        )


def test_storage_reclamation_plan_protects_current_lineage_and_moves_nothing(
    tmp_path: Path,
) -> None:
    active = tmp_path / "reports" / "active"
    snapshots = (
        tmp_path / "reports" / "snapshots" / "current_wizard_hyperliquid"
    )
    current = snapshots / "run" / "cwcadence_current"
    old = snapshots / "run" / "cwconcentration_old"
    old_child = old / "failure_attribution" / "cwfailure_old"
    incomplete = snapshots / "run" / "cwfailure_incomplete"
    exhaustive = (
        tmp_path / "reports" / "snapshots" / "exhaustive_wizard_hyperliquid"
    )
    exhaustive_current = exhaustive / "ewhl_current"
    exhaustive_old = exhaustive / "ewhl_old"
    for path in (
        active,
        current,
        old_child,
        incomplete,
        exhaustive_current,
        exhaustive_old,
    ):
        path.mkdir(parents=True, exist_ok=True)
    (current / "manifest.json").write_text("{}", encoding="utf-8")
    (old / "manifest.json").write_text("{}", encoding="utf-8")
    (old / "payload.csv").write_text("value\n1\n", encoding="utf-8")
    (old_child / "manifest.json").write_text("{}", encoding="utf-8")
    (incomplete / "partial.csv").write_text("value\n2\n", encoding="utf-8")
    (exhaustive_current / "manifest.json").write_text("{}", encoding="utf-8")
    (exhaustive_old / "manifest.json").write_text("{}", encoding="utf-8")
    (exhaustive_old / "payload.csv").write_text("value\n3\n", encoding="utf-8")
    (active / "current_wizard_hyperliquid_operating_cadence_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "snapshot_manifest": str(
                        (current / "manifest.json").relative_to(tmp_path)
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    (active / "exhaustive_wizard_hyperliquid_run_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "snapshot_manifest": str(
                        (exhaustive_current / "manifest.json").relative_to(tmp_path)
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    bookkeeping_reference = str((old / "manifest.json").relative_to(tmp_path))
    (active / "current_wizard_hyperliquid_archive_copy_manifest.json").write_text(
        json.dumps({"candidate_receipts": [{"snapshot_path": bookkeeping_reference}]}),
        encoding="utf-8",
    )
    (active / "current_wizard_hyperliquid_archive_release_manifest.json").write_text(
        json.dumps({"release_plan": [{"snapshot_path": bookkeeping_reference}]}),
        encoding="utf-8",
    )

    result = build_current_wizard_hyperliquid_storage_reclamation_plan(
        root=tmp_path,
        now=datetime(2026, 8, 8, 18, 0, tzinfo=timezone.utc),
        available_disk_bytes=100,
    )
    plan = pd.read_csv(result.paths["plan"], keep_default_na=False)
    validation = pd.read_csv(result.paths["validation"], keep_default_na=False)

    assert set(plan["snapshot_id"]) == {
        "cwconcentration_old",
        "cwfailure_incomplete",
        "ewhl_old",
    }
    assert "cwcadence_current" not in set(plan["snapshot_id"])
    assert "ewhl_current" not in set(plan["snapshot_id"])
    complete = plan.loc[plan["snapshot_id"].eq("cwconcentration_old")].iloc[0]
    partial = plan.loc[plan["snapshot_id"].eq("cwfailure_incomplete")].iloc[0]
    assert complete["safe_to_archive_later"]
    assert not partial["safe_to_archive_later"]
    assert not plan["move_performed"].astype(bool).any()
    assert not plan["delete_performed"].astype(bool).any()
    safe = plan.loc[plan["safe_to_archive_later"].astype(bool)]
    assert safe["hash_verified"].astype(bool).all()
    assert safe["manifest_sha256"].str.len().eq(64).all()
    assert safe["tree_sha256"].str.len().eq(64).all()
    assert not partial["hash_verified"]
    assert validation["status"].eq("PASS").all()
    assert result.summary["safe_archive_candidate_branches"] == 2
    assert result.summary["hash_verified_candidate_branches"] == 2
    assert result.summary["manual_review_branches"] == 1
    assert result.summary["archive_destination_configured"] is False
    assert result.summary["archive_destination_status"] == "BLOCKED"
    assert (
        result.summary["archive_destination_blocker"]
        == "off_volume_archive_destination_not_configured"
    )
    assert result.summary["archive_copy_preflight_ready"] is False
    assert result.summary["archive_release_preflight_ready"] is False
    assert result.summary["move_performed"] is False
    assert result.summary["delete_performed"] is False

    same_device = build_current_wizard_hyperliquid_storage_reclamation_plan(
        root=tmp_path,
        now=datetime(2026, 8, 8, 18, 0, tzinfo=timezone.utc),
        available_disk_bytes=100,
        archive_destination=tmp_path,
    )
    assert same_device.summary["archive_destination_configured"] is True
    assert same_device.summary["archive_destination_same_device"] is True
    assert (
        "archive_destination_is_same_device_as_project"
        in same_device.summary["archive_destination_blocker"]
    )
    assert same_device.summary["archive_copy_preflight_ready"] is False
    assert same_device.summary["archive_release_preflight_ready"] is False
    assert same_device.summary["move_performed"] is False
    assert same_device.summary["delete_performed"] is False
