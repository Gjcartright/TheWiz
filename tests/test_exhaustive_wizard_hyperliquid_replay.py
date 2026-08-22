from datetime import datetime, timedelta, timezone
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration import exhaustive_wizard_hyperliquid_validation
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_replay import (
    build_exhaustive_wizard_hyperliquid_replay_preflight,
    materialize_exhaustive_wizard_hyperliquid_history,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_canonical_replay import (
    _unique_mode_rows,
    run_exhaustive_wizard_hyperliquid_canonical_replay,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_cost_evidence import (
    materialize_exhaustive_hyperliquid_funding_evidence,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_cost_bridge import (
    build_exhaustive_wizard_hyperliquid_cost_evidence,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_observed_cost_replay import (
    run_exhaustive_wizard_hyperliquid_observed_cost_replay,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_walkforward import (
    FOLD_COUNT,
    _attach_causal_entry_features,
    run_exhaustive_wizard_hyperliquid_walkforward,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_regimes import (
    build_exhaustive_wizard_hyperliquid_regime_attribution,
    build_causal_pair_regime_features,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_robustness import (
    run_exhaustive_wizard_hyperliquid_robustness,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_concentration import (
    _concentration_metrics,
    build_exhaustive_wizard_hyperliquid_concentration,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_leverage import (
    _margin_metrics,
    _simulate_leverage_path,
    build_exhaustive_wizard_hyperliquid_leverage_surface,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_run import (
    EXACT_MODES,
    ORIENTATIONS,
    build_exhaustive_wizard_hyperliquid_mapping_refresh,
    build_exhaustive_wizard_hyperliquid_run,
)


def test_walkforward_entry_features_are_causal_and_label_separated():
    timestamps = pd.date_range("2025-01-01", periods=80, freq="h", tz="UTC")
    history = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price_x": 100.0 + np.arange(80) * 0.3,
            "price_y": 80.0 + np.arange(80) * 0.2 + np.sin(np.arange(80) / 5),
            "hedge_ratio": np.linspace(0.9, 1.1, 80),
            "funding_bps_per_day": np.linspace(-0.2, 0.2, 80),
            "slippage_x_model_bps": 1.5,
            "slippage_y_model_bps": 2.0,
        }
    )
    metric = pd.Series(np.linspace(0.05, 0.95, 80), index=history.index)
    closed = pd.DataFrame(
        [
            {
                "trade_id": 1,
                "entry_timestamp": timestamps[35],
                "exit_timestamp": timestamps[40],
                "profit_after_cost": 0.01,
            }
        ]
    )
    bars = pd.DataFrame(
        [{"timestamp": timestamps[35], "net_return": 0.0}]
    )

    trades, _ = _attach_causal_entry_features(
        closed=closed,
        bars=bars,
        signal_history=history,
        mode_metric=metric,
        metric_name="u1_given_u2",
        settings={"hedge_ratio": 1.0},
        exact_mode="Copula",
    )
    changed = history.copy()
    changed.loc[36:, ["price_x", "price_y", "hedge_ratio"]] *= 10.0
    changed_metric = metric.copy()
    changed_metric.loc[36:] = 0.0
    changed_trades, _ = _attach_causal_entry_features(
        closed=closed,
        bars=bars,
        signal_history=changed,
        mode_metric=changed_metric,
        metric_name="u1_given_u2",
        settings={"hedge_ratio": 1.0},
        exact_mode="Copula",
    )

    row = trades.iloc[0]
    changed_row = changed_trades.iloc[0]
    assert row["feature_timestamp"] == timestamps[35].isoformat()
    assert row["label_timestamp"] == timestamps[40]
    assert row["mode_metric"] == pytest.approx(metric.iloc[35])
    assert row["u1_given_u2"] == pytest.approx(metric.iloc[35])
    assert bool(row["feature_uses_future_data"]) is False
    assert bool(row["feature_known_at_or_before_entry"]) is True
    for column in (
        "mode_metric",
        "spread",
        "spread_slope",
        "realized_volatility_percentile",
        "correlation",
        "hedge_ratio",
        "funding_bps_per_day",
        "liquidity_score",
    ):
        assert changed_row[column] == pytest.approx(row[column])


def _build_fixture(root, *, asset_y="ETH", mapped=True):
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    assets = ["BTC", asset_y] if mapped else ["BTC"]
    pd.DataFrame(
        [
            {
                "asset": asset,
                "asset_index": index,
                "universe_name": asset,
                "max_leverage": 10,
                "margin_table_id": 10,
                "only_isolated": False,
                "tradable_perp": True,
                "checked_at_utc": "2026-08-07T12:00:00+00:00",
                "fetch_blocker": "",
            }
            for index, asset in enumerate(assets)
        ]
    ).to_csv(active / "hyperliquid_testnet_market_inventory.csv", index=False)
    pd.DataFrame(
        [
            {
                "margin_table_id": 10,
                "tier_number": 0,
                "lower_bound_usd": 0.0,
                "upper_bound_usd": "",
                "max_leverage": 10,
                "maintenance_margin_rate": 0.05,
                "maintenance_deduction_usd": 0.0,
                "checked_at_utc": "2026-08-07T12:00:00+00:00",
                "source_system": "hyperliquid_testnet_meta",
                "source_url": (
                    "https://hyperliquid.gitbook.io/hyperliquid-docs/trading/margin-tiers"
                ),
                "blocker": "",
            }
        ]
    ).to_csv(active / "hyperliquid_testnet_margin_tiers.csv", index=False)
    pd.DataFrame([{"status": "completed"}]).to_csv(
        active / "wizard_sweep_manifest.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "sweep_id": "sweep-1",
                "request_id": "request-1",
                "sweep_exchange": "Dydx",
                "sweep_interval": "Daily",
                "sweep_complete": True,
                "symbol_1": "BTC-USD",
                "symbol_2": f"{asset_y}-USD",
            }
        ]
    ).to_csv(active / "wizard_sweep_candidates.csv", index=False)
    run = build_exhaustive_wizard_hyperliquid_run(root=root)
    build_exhaustive_wizard_hyperliquid_mapping_refresh(root=root)
    pair = pd.read_csv(run.paths["pair_ledger"]).iloc[0]
    pd.DataFrame(
        [
            {
                "exhaustive_run_id": run.summary["run_id"],
                "pair_group_id": pair["pair_group_id"],
                "timeframe": "daily",
                "capture_status": "COMPLETE",
                "scanner_capture_timestamps": "2026-08-07T12:30:00+00:00",
            }
        ]
    ).to_csv(active / "exhaustive_wizard_pair_detail_capture_progress.csv", index=False)
    mode_rows = []
    if mapped:
        for mode in EXACT_MODES:
            for orientation in ORIENTATIONS:
                mode_rows.append(
                    {
                        "pair_group_id": pair["pair_group_id"],
                        "exact_mode": mode,
                        "orientation": orientation,
                        "capture_status": "CAPTURED",
                        "capture_blocker": "",
                        "orientation_verified": True,
                        "entry_long": 0.20 if mode == "Copula" else -0.50,
                        "entry_short": 0.80 if mode == "Copula" else 0.50,
                        "exit_long": 0.45 if mode == "Copula" else 0.0,
                        "exit_short": 0.55 if mode == "Copula" else 0.0,
                        "entry_long_operator": "Lte" if mode != "Copula" else "",
                        "entry_short_operator": "Gte" if mode != "Copula" else "",
                        "exit_long_operator": "Gte" if mode != "Copula" else "",
                        "exit_short_operator": "Lte" if mode != "Copula" else "",
                        "rolling_window": 20 if mode.endswith("ZScoreR)") else "",
                        "hedge_ratio": 1.5,
                        "ou_mu": 0.0 if mode == "OU (Spread)" else "",
                        "ou_sigma": 1.0 if mode == "OU (Spread)" else "",
                        "copula_family": "gaussian",
                        "evidence_path": "data/raw/pair_bundle.json",
                    }
                )
    pd.DataFrame(
        mode_rows,
        columns=[
            "pair_group_id",
            "exact_mode",
            "orientation",
            "capture_status",
            "capture_blocker",
            "orientation_verified",
            "entry_long",
            "entry_short",
            "exit_long",
            "exit_short",
            "entry_long_operator",
            "entry_short_operator",
            "exit_long_operator",
            "exit_short_operator",
            "rolling_window",
            "hedge_ratio",
            "ou_mu",
            "ou_sigma",
            "copula_family",
            "evidence_path",
        ],
    ).to_csv(active / "exhaustive_wizard_pair_detail_mode_ledger.csv", index=False)
    mainnet_assets = ["BTC", asset_y] if mapped else ["BTC"]
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"asset": asset, "tradable": True} for asset in mainnet_assets]).to_csv(
        root / "data" / "processed" / "hyperliquid_market_context.csv", index=False
    )
    return run


def test_replay_preflight_accounts_every_cell_and_deduplicates_only_fetches(tmp_path):
    run = _build_fixture(tmp_path)

    result = build_exhaustive_wizard_hyperliquid_replay_preflight(
        root=tmp_path,
        now=datetime(2026, 8, 8, 5, tzinfo=timezone.utc),
    )

    experiments = pd.read_csv(result.paths["experiment_preflight"])
    validation_queue = pd.read_csv(result.paths["current_cycle_validation_queue"])
    pairs = pd.read_csv(result.paths["pair_history_queue"])
    fetches = pd.read_csv(result.paths["asset_fetch_queue"])
    assert len(experiments) == len(EXACT_MODES) * len(ORIENTATIONS)
    assert experiments["experiment_id"].nunique() == len(experiments)
    assert experiments["preflight_status"].eq("READY_FOR_HISTORY").sum() == 14
    assert experiments["preflight_status"].eq("NOT_APPLICABLE_WIZARD_MODE").sum() == 0
    assert len(validation_queue) == 10
    assert not validation_queue["exact_mode"].str.startswith("OU").any()
    ou_rows = experiments.loc[experiments["exact_mode"].str.startswith("OU")]
    assert ou_rows["validation_lane"].eq("DIAGNOSTIC_ONLY").all()
    assert ou_rows["acceptance_policy_status"].eq("BLOCKED").all()
    assert ou_rows["acceptance_policy_blocker"].eq("ou_family_terminal_research_only").all()
    assert pairs["history_request_status"].eq("READY_TO_FETCH").all()
    assert pairs["ready_mode_orientation_cells"].item() == 10
    assert pairs["current_cycle_planned_mode_orientation_cells"].item() == 10
    assert pairs["diagnostic_only_mode_orientation_cells"].item() == 4
    assert set(fetches["asset"]) == {"BTC", "ETH"}
    assert fetches["hyperliquid_interval"].eq("1d").all()
    assert fetches["fetch_end_at"].str.startswith("2026-08-07T12:30:00").all()
    assert result.summary["experiments"] == run.summary["planned_experiments"]
    assert result.summary["network_request_deduplication_only"] is True
    assert result.summary["current_cycle_validation_experiments"] == 10
    assert result.summary["ou_diagnostic_only_experiments"] == 4
    assert not experiments["discovery_prefilter_applied"].astype(bool).any()
    assert not experiments["live_trading_authorized"].astype(bool).any()
    assert result.paths["snapshot_manifest"].exists()


def test_replay_preflight_keeps_mapping_blocked_pair_without_fetch_substitution(tmp_path):
    run = _build_fixture(tmp_path, asset_y="NOPE", mapped=False)

    result = build_exhaustive_wizard_hyperliquid_replay_preflight(root=tmp_path)

    experiments = pd.read_csv(result.paths["experiment_preflight"])
    pairs = pd.read_csv(result.paths["pair_history_queue"])
    fetches = pd.read_csv(result.paths["asset_fetch_queue"])
    assert len(experiments) == run.summary["planned_experiments"]
    assert experiments["preflight_status"].eq("BLOCKED_HYPERLIQUID_MAPPING").all()
    assert (
        experiments["preflight_blocker"].str.contains("missing_hyperliquid_testnet_perp:NOPE").all()
    )
    assert pairs["history_request_status"].eq("BLOCKED").all()
    assert fetches.empty
    assert result.summary["asset_interval_fetch_requests"] == 0
    assert result.summary["live_trading_authorized"] is False


def _fake_candle_fetcher(*, rows):
    def fetcher(*, coin, interval, days, output_dir, end_time):
        del days
        output_dir.mkdir(parents=True, exist_ok=True)
        step = timedelta(days=1) if interval == "1d" else timedelta(hours=1)
        latest = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
        start = latest - step * (rows - 1)
        base = 100.0 if coin == "BTC" else 50.0
        candles = []
        for index in range(rows):
            timestamp = start + step * index
            price = base + index * 0.01
            candles.append(
                {
                    "startedAt": timestamp.isoformat(),
                    "ticker": f"{coin}-USD",
                    "resolution": interval,
                    "open": price,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price,
                    "baseVolume": 10.0,
                    "usdVolume": price * 10.0,
                    "source": "hyperliquid",
                }
            )
        output = output_dir / f"{coin}_{interval}_candles.json"
        output.write_text(json.dumps({"candles": candles}), encoding="utf-8")
        return output

    return fetcher


def _fake_funding_fetcher(*, coin, days, output_dir, end_time, page_interval_seconds):
    del page_interval_seconds
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
    records = [
        {
            "coin": coin,
            "time": int((latest - timedelta(days=index)).timestamp() * 1000),
            "fundingRate": "0.0001" if index % 2 == 0 else "-0.00005",
            "premium": "0.0",
        }
        for index in range(days)
    ]
    output = output_dir / f"{coin}_funding.json"
    output.write_text(
        json.dumps(
            {
                "coin": coin,
                "funding": records,
                "fetch_complete": True,
                "fetched_at": end_time.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    return output


def test_causal_regime_features_are_future_prefix_invariant():
    rng = np.random.default_rng(7)
    timestamps = pd.date_range("2024-01-01", periods=320, freq="D", tz="UTC")
    returns_x = rng.normal(0.0005, 0.02, len(timestamps))
    returns_y = 0.65 * returns_x + rng.normal(0.0003, 0.012, len(timestamps))
    history = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price_x": 100.0 * np.cumprod(1.0 + returns_x),
            "price_y": 80.0 * np.cumprod(1.0 + returns_y),
        },
        index=timestamps,
    )
    altered = history.copy()
    altered.loc[timestamps[250]:, "price_x"] *= np.linspace(1.0, 4.0, len(timestamps) - 250)
    altered.loc[timestamps[250]:, "price_y"] *= np.linspace(1.0, 0.25, len(timestamps) - 250)

    baseline_features = build_causal_pair_regime_features(history)
    altered_features = build_causal_pair_regime_features(altered)

    pd.testing.assert_frame_equal(
        baseline_features.iloc[:250].reset_index(drop=True),
        altered_features.iloc[:250].reset_index(drop=True),
    )
    assert not baseline_features["regime_uses_future_data"].astype(bool).any()


def test_cross_cell_concentration_rejects_one_dominant_profit_source():
    grouped = pd.DataFrame(
        [
            {
                "dimension_value": "BTC/ETH",
                "candidate_count": 1,
                "trade_count": 80,
                "net_profit": 90.0,
            },
            {
                "dimension_value": "SOL/ETH",
                "candidate_count": 1,
                "trade_count": 10,
                "net_profit": 5.0,
            },
            {
                "dimension_value": "DOGE/XRP",
                "candidate_count": 1,
                "trade_count": 10,
                "net_profit": 5.0,
            },
        ]
    )
    summary, contributors = _concentration_metrics(
        grouped,
        cohort="test",
        dimension="canonical_pair",
        candidate_count=3,
        concentration_id="concentration-test",
        thresholds={
            "minimum_values": 3,
            "maximum_positive_profit_share": 0.35,
            "maximum_trade_share": 0.40,
            "minimum_effective_profit_breadth": 3.0,
        },
    )

    assert summary["dimension_gate_pass"] is False
    assert "max_positive_profit_share>0.35" in summary["dimension_blocker"]
    assert "max_trade_share>0.4" in summary["dimension_blocker"]
    assert len(contributors) == 3
    assert max(row["positive_profit_share"] for row in contributors) == 0.9


def test_hyperliquid_leverage_margin_uses_pair_tiers_and_integer_setting():
    tiers = pd.DataFrame(
        [
            {
                "margin_table_id": 10,
                "lower_bound_usd": 0.0,
                "upper_bound_usd": "",
                "max_leverage": 10,
                "maintenance_margin_rate": 0.05,
                "maintenance_deduction_usd": 0.0,
                "blocker": "",
            }
        ]
    )
    market = {"max_leverage": 10, "margin_table_id": 10, "only_isolated": False}

    isolated = _margin_metrics(
        requested_leverage=1.5,
        equity_usd=1000.0,
        hedge_ratio_abs=1.0,
        margin_mode="isolated",
        market_x=market,
        market_y=market,
        tiers=tiers,
    )

    assert isolated["effective_gross_leverage"] == 1.5
    assert isolated["required_exchange_leverage_setting"] == 2
    assert isolated["leg_x_notional_usd"] == 750.0
    assert isolated["leg_y_notional_usd"] == 750.0
    assert isolated["total_initial_margin_usd"] == 750.0
    assert isolated["total_maintenance_margin_usd"] == 75.0
    assert isolated["maintenance_buffer_ratio"] == 0.925
    assert isolated["minimum_liquidation_distance_pct"] == pytest.approx(
        (375.0 - 37.5) / 750.0 / 1.05
    )


def test_hyperliquid_leverage_path_scales_canonical_returns_without_changing_signal():
    bars = pd.DataFrame(
        [
            {
                "fold_number": 1,
                "target_position": 1,
                "net_return": -0.02,
                "gross_return": -0.018,
                "funding": 0.001,
                "slippage": 0.001,
            },
            {
                "fold_number": 1,
                "target_position": 1,
                "net_return": 0.03,
                "gross_return": 0.032,
                "funding": 0.001,
                "slippage": 0.001,
            },
        ]
    )

    baseline = _simulate_leverage_path(
        bars,
        effective_leverage=2.0,
        stress={"stress_scenario": "baseline"},
        maximum_leg_weight=0.5,
    )
    orphan = _simulate_leverage_path(
        bars,
        effective_leverage=2.0,
        stress={"stress_scenario": "orphan", "orphan_leg_loss": 0.05},
        maximum_leg_weight=0.5,
    )

    assert baseline["aggregate_total_return"] == pytest.approx((1 - 0.04) * (1 + 0.06) - 1)
    assert baseline["minimum_bar_return"] == -0.04
    assert baseline["ruin_triggered"] is False
    assert orphan["aggregate_total_return"] < baseline["aggregate_total_return"]


def test_validation_runner_uses_fixed_stage_order_and_fail_closed_manifest(
    tmp_path, monkeypatch
):
    calls = []
    stages = [
        ("build_exhaustive_wizard_hyperliquid_cost_evidence", "cost_evidence_id"),
        ("run_exhaustive_wizard_hyperliquid_observed_cost_replay", "observed_cost_replay_id"),
        ("run_exhaustive_wizard_hyperliquid_walkforward", "walkforward_id"),
        ("build_exhaustive_wizard_hyperliquid_regime_attribution", "regime_attribution_id"),
        ("run_exhaustive_wizard_hyperliquid_robustness", "robustness_id"),
        ("build_exhaustive_wizard_hyperliquid_concentration", "concentration_id"),
        ("build_exhaustive_wizard_hyperliquid_leverage_surface", "leverage_surface_id"),
    ]

    for index, (attribute, identity_key) in enumerate(stages, start=1):
        def fake_stage(*, root, now, position=index, key=identity_key, name=attribute):
            calls.append(name)
            manifest = root / "reports" / "active" / f"stage_{position}.json"
            snapshot_manifest = (
                root / "reports" / "snapshots" / "run-1" / f"stage_{position}.json"
            )
            manifest.parent.mkdir(parents=True, exist_ok=True)
            snapshot_manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("{}", encoding="utf-8")
            snapshot_manifest.write_text("{}", encoding="utf-8")
            return CommandResult(
                paths={"manifest": manifest, "snapshot_manifest": snapshot_manifest},
                summary={
                    "run_id": "run-1",
                    "experiments": 14,
                    "experiment_status_accounted": True,
                    key: f"stage-id-{position}",
                    "acceptance_eligible_replays": 0,
                    "live_trading_authorized": False,
                },
            )

        monkeypatch.setattr(exhaustive_wizard_hyperliquid_validation, attribute, fake_stage)

    def fake_learning(*, root, now, validation_id, validation_stages):
        calls.append("build_exhaustive_wizard_hyperliquid_learning_ledger")
        manifest = root / "reports" / "active" / "learning.json"
        snapshot_manifest = (
            root
            / "reports"
            / "snapshots"
            / "run-1"
            / "validations"
            / validation_id
            / "learning"
            / "manifest.json"
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        snapshot_manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}", encoding="utf-8")
        snapshot_manifest.write_text("{}", encoding="utf-8")
        assert len(validation_stages) == 7
        return CommandResult(
            paths={"manifest": manifest, "snapshot_manifest": snapshot_manifest},
            summary={
                "run_id": "run-1",
                "learning_ledger_id": "learning-1",
                "records": 14,
                "unique_experiment_ids": 14,
                "experiment_status_accounted": True,
                "outcome_type": "backtest_research_outcome",
                "record_granularity": "experiment_summary",
                "training_eligible_records": 0,
                "paper_label_records": 0,
                "live_label_records": 0,
                "live_trading_authorized": False,
            },
        )

    monkeypatch.setattr(
        exhaustive_wizard_hyperliquid_validation,
        "build_exhaustive_wizard_hyperliquid_learning_ledger",
        fake_learning,
    )

    result = exhaustive_wizard_hyperliquid_validation.run_exhaustive_wizard_hyperliquid_validation(
        root=tmp_path,
        now=datetime(2026, 8, 8, 9, tzinfo=timezone.utc),
    )

    assert calls == [
        *[attribute for attribute, _ in stages],
        "build_exhaustive_wizard_hyperliquid_learning_ledger",
    ]
    assert result.summary["stages_complete"] == 7
    assert result.summary["stage_accounting_complete"] is True
    assert result.summary["order_submission_performed"] is False
    assert result.summary["live_trading_authorized"] is False
    assert result.summary["learning_ledger"]["records"] == 14
    assert result.summary["learning_ledger"]["training_eligible_records"] == 0
    assert all(
        row["manifest_path"].startswith("reports/snapshots/")
        for row in result.summary["stages"]
    )
    assert result.paths["snapshot_manifest"].exists()


def test_materialize_history_builds_bounded_pair_history(tmp_path):
    _build_fixture(tmp_path)
    build_exhaustive_wizard_hyperliquid_replay_preflight(
        root=tmp_path,
        now=datetime(2026, 8, 8, 5, tzinfo=timezone.utc),
    )

    result = materialize_exhaustive_wizard_hyperliquid_history(
        root=tmp_path,
        now=datetime(2026, 8, 8, 6, tzinfo=timezone.utc),
        fetcher=_fake_candle_fetcher(rows=800),
        sleep=lambda _: None,
    )

    assets = pd.read_csv(result.paths["asset_results"])
    pairs = pd.read_csv(result.paths["pair_results"])
    assert len(assets) == 2
    assert assets["history_status"].eq("COMPLETE").all()
    assert assets["history_rows"].eq(800).all()
    assert assets["timestamp_bound_valid"].astype(bool).all()
    assert len(pairs) == 1
    assert pairs["history_status"].eq("READY_FOR_CANONICAL_REPLAY").all()
    assert pairs["history_rows"].eq(800).all()
    assert pairs["canonical_replay_leverage"].eq(1.0).all()
    assert result.summary["post_snapshot_asset_violations"] == 0
    assert result.summary["post_snapshot_pair_violations"] == 0
    assert result.summary["live_trading_authorized"] is False
    assert result.paths["snapshot_manifest"].exists()


def test_materialize_history_retains_insufficient_history_blockers(tmp_path):
    _build_fixture(tmp_path)
    build_exhaustive_wizard_hyperliquid_replay_preflight(root=tmp_path)

    result = materialize_exhaustive_wizard_hyperliquid_history(
        root=tmp_path,
        fetcher=_fake_candle_fetcher(rows=20),
        sleep=lambda _: None,
    )

    assets = pd.read_csv(result.paths["asset_results"])
    pairs = pd.read_csv(result.paths["pair_results"])
    assert assets["history_status"].eq("BLOCKED").all()
    assert assets["history_blocker"].eq("insufficient_point_in_time_history").all()
    assert pairs["history_status"].eq("BLOCKED").all()
    assert pairs["history_blocker"].str.contains("insufficient_point_in_time_history").all()
    assert pairs["history_path"].fillna("").eq("").all()
    assert result.summary["pair_histories_ready"] == 0
    assert result.summary["pair_histories_blocked"] == 1


def test_short_history_research_lane_does_not_weaken_acceptance(tmp_path):
    run = _build_fixture(tmp_path)
    build_exhaustive_wizard_hyperliquid_replay_preflight(
        root=tmp_path,
        now=datetime(2026, 8, 8, 5, tzinfo=timezone.utc),
    )

    history = materialize_exhaustive_wizard_hyperliquid_history(
        root=tmp_path,
        now=datetime(2026, 8, 8, 6, tzinfo=timezone.utc),
        fetcher=_fake_candle_fetcher(rows=627),
        sleep=lambda _: None,
    )

    assets = pd.read_csv(history.paths["asset_results"])
    pairs = pd.read_csv(history.paths["pair_results"])
    assert assets["history_status"].eq("COMPLETE_RESEARCH_ONLY").all()
    assert assets["history_lane"].eq("SHORT_HISTORY_RESEARCH_ONLY").all()
    assert not assets["acceptance_history_ready"].astype(bool).any()
    assert assets["research_history_ready"].astype(bool).all()
    assert pairs["history_status"].eq(
        "READY_FOR_SHORT_HISTORY_RESEARCH_REPLAY"
    ).all()
    assert pairs["history_rows"].eq(627).all()
    assert pairs["history_path"].astype(str).str.len().gt(0).all()
    assert history.summary["pair_histories_ready"] == 0
    assert history.summary["pair_histories_short_research_ready"] == 1

    replay = run_exhaustive_wizard_hyperliquid_canonical_replay(
        root=tmp_path,
        now=datetime(2026, 8, 8, 7, tzinfo=timezone.utc),
    )
    results = pd.read_csv(replay.paths["results"], keep_default_na=False)
    completed = results.loc[
        results["replay_status"].eq("SHORT_HISTORY_RESEARCH_REPLAY_COMPLETE")
    ]
    assert len(completed) == 10
    assert completed["history_validation_lane"].eq(
        "SHORT_HISTORY_RESEARCH_ONLY"
    ).all()
    assert completed["acceptance_status"].eq("BLOCKED").all()
    assert completed["acceptance_reason"].str.contains(
        "short_history_research_only"
    ).all()
    assert not completed["research_rank_eligible"].astype(bool).any()
    assert replay.summary["short_history_research_replays_complete"] == 10
    assert replay.summary["acceptance_eligible_replays"] == 0
    assert len(results) == run.summary["planned_experiments"]


def test_short_history_lane_propagates_through_cost_and_walkforward(tmp_path):
    _build_fixture(tmp_path)
    active = tmp_path / "reports" / "active"
    build_exhaustive_wizard_hyperliquid_replay_preflight(root=tmp_path)
    materialize_exhaustive_wizard_hyperliquid_history(
        root=tmp_path,
        fetcher=_fake_candle_fetcher(rows=627),
        sleep=lambda _: None,
    )
    run_exhaustive_wizard_hyperliquid_canonical_replay(root=tmp_path)

    funding = materialize_exhaustive_hyperliquid_funding_evidence(
        root=tmp_path,
        fetcher=_fake_funding_fetcher,
        sleep=lambda _: None,
    )
    funding_queue = pd.read_csv(funding.paths["queue"], keep_default_na=False)
    funding_pairs = pd.read_csv(funding.paths["pair_coverage"], keep_default_na=False)
    assert set(funding_queue["asset"]) == {"BTC", "ETH"}
    assert funding_pairs["funding_status"].eq(
        "READY_FOR_SHORT_HISTORY_COST_RESEARCH"
    ).all()
    assert funding_pairs["funding_research_ready"].astype(bool).all()
    assert not funding_pairs["funding_acceptance_ready"].astype(bool).any()

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "asset_x": "BTC",
                "asset_y": "ETH",
                "fee_profile_id": "test-fee-v1",
                "taker_fee_bps": 4.5,
                "execution_risk_bps": 2.0,
                "fee_source_checked_at": "2026-08-07",
                "cost_model_status": "official_base_tier_conservative_fee_profile",
                "cost_model_ready": True,
                "slippage_model_status": "insufficient_l2_depth_samples",
                "slippage_model_ready": False,
                "slippage_samples_x": 2,
                "slippage_samples_y": 2,
                "required_slippage_samples": 12,
                "slippage_window_hours": 24,
                "slippage_x_p95_bps": 1.0,
                "slippage_y_p95_bps": 2.0,
                "pair_one_way_slippage_bps": 1.5,
                "estimated_pair_round_trip_cost_bps": 16.0,
                "freshest_sample_at": "2026-08-08T04:30:00+00:00",
                "evidence_path": "data/processed/l2.csv",
            }
        ]
    ).to_csv(active / "hyperliquid_pair_cost_model.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "status": "waiting_for_next_capture",
                "next_sample_due_at": "2026-08-08T04:40:00+00:00",
                "evidence_path": "reports/active/cadence.csv",
            }
        ]
    ).to_csv(active / "hyperliquid_evidence_cadence.csv", index=False)

    costs = build_exhaustive_wizard_hyperliquid_cost_evidence(
        root=tmp_path,
        now=datetime(2026, 8, 8, 5, tzinfo=timezone.utc),
    )
    pair_costs = pd.read_csv(costs.paths["pair_cost_evidence"], keep_default_na=False)
    experiment_costs = pd.read_csv(
        costs.paths["experiment_cost_readiness"], keep_default_na=False
    )
    assert pair_costs["cost_evidence_status"].eq(
        "READY_FOR_SHORT_HISTORY_COST_RESEARCH"
    ).all()
    assert pair_costs["history_lane"].eq("SHORT_HISTORY_RESEARCH_ONLY").all()
    assert not pair_costs["cost_acceptance_ready"].astype(bool).any()
    assert experiment_costs["cost_replay_status"].eq(
        "READY_FOR_SHORT_HISTORY_COST_RESEARCH"
    ).sum() == 14

    replay = run_exhaustive_wizard_hyperliquid_observed_cost_replay(
        root=tmp_path,
        now=datetime(2026, 8, 8, 5, 5, tzinfo=timezone.utc),
    )
    replay_rows = pd.read_csv(replay.paths["results"], keep_default_na=False)
    completed = replay_rows.loc[
        replay_rows["replay_status"].eq(
            "SHORT_HISTORY_OBSERVED_COST_RESEARCH_REPLAY_COMPLETE"
        )
    ]
    assert len(completed) == 10
    assert completed["history_validation_lane"].eq(
        "SHORT_HISTORY_RESEARCH_ONLY"
    ).all()
    assert not completed["research_rank_eligible"].astype(bool).any()
    assert completed["research_rank_blocker"].str.contains(
        "short_history_research_only"
    ).all()
    assert replay.summary["short_history_observed_cost_replays_complete"] == 10

    walkforward = run_exhaustive_wizard_hyperliquid_walkforward(
        root=tmp_path,
        now=datetime(2026, 8, 8, 5, 10, tzinfo=timezone.utc),
    )
    statuses = pd.read_csv(walkforward.paths["status"], keep_default_na=False)
    candidates = pd.read_csv(walkforward.paths["candidates"], keep_default_na=False)
    ranked = pd.read_csv(walkforward.paths["ranked"], keep_default_na=False)
    assert len(statuses) == len(EXACT_MODES) * len(ORIENTATIONS)
    assert len(candidates) == 10
    assert candidates["history_validation_lane"].eq(
        "SHORT_HISTORY_RESEARCH_ONLY"
    ).all()
    assert not candidates["walkforward_rank_eligible"].astype(bool).any()
    assert candidates["walkforward_rank_blocker"].eq(
        "short_history_research_only"
    ).all()
    assert ranked["walkforward_rank"].fillna("").astype(str).str.len().eq(0).all()
    assert candidates["acceptance_reason"].str.contains(
        "short_history_research_only"
    ).all()
    assert walkforward.summary["short_history_walkforward_candidates"] == 10
    assert walkforward.summary["acceptance_eligible_replays"] == 0
    assert walkforward.summary["live_trading_authorized"] is False


def test_canonical_replay_accounts_for_every_experiment_without_accepting_research(tmp_path):
    run = _build_fixture(tmp_path)
    build_exhaustive_wizard_hyperliquid_replay_preflight(
        root=tmp_path,
        now=datetime(2026, 8, 8, 5, tzinfo=timezone.utc),
    )
    materialize_exhaustive_wizard_hyperliquid_history(
        root=tmp_path,
        now=datetime(2026, 8, 8, 6, tzinfo=timezone.utc),
        fetcher=_fake_candle_fetcher(rows=800),
        sleep=lambda _: None,
    )

    result = run_exhaustive_wizard_hyperliquid_canonical_replay(
        root=tmp_path,
        now=datetime(2026, 8, 8, 7, tzinfo=timezone.utc),
    )

    results = pd.read_csv(result.paths["results"], keep_default_na=False)
    ranked = pd.read_csv(result.paths["ranked"], keep_default_na=False)
    assert len(results) == run.summary["planned_experiments"]
    assert results["experiment_id"].nunique() == len(results)
    assert results["replay_status"].eq("RESEARCH_REPLAY_COMPLETE").sum() == 10
    assert results["replay_status"].eq("BLOCKED_DYNAMIC_EXPOSURE_RULE").sum() == 4
    assert results["replay_status"].eq("NOT_APPLICABLE_WIZARD_MODE").sum() == 0
    assert len(ranked) == 10
    assert {"research_rank", "research_rank_eligible", "research_rank_blocker"}.issubset(
        ranked.columns
    )
    assert ranked["canonical_replay_leverage"].eq(1.0).all()
    assert ranked["acceptance_status"].eq("BLOCKED").all()
    assert ranked["cost_evidence_status"].eq("PROVISIONAL_CONSERVATIVE_DEFAULTS").all()
    eligible = ranked.loc[ranked["research_rank_eligible"].astype(bool)]
    assert pd.to_numeric(eligible["total_return"], errors="coerce").gt(0.0).all()
    assert pd.to_numeric(eligible["expectancy"], errors="coerce").gt(0.0).all()
    assert result.summary["acceptance_eligible_replays"] == 0
    assert result.summary["live_trading_authorized"] is False


def test_mode_lookup_scopes_historical_rows_but_rejects_active_duplicates():
    frame = pd.DataFrame(
        [
            {"pair_group_id": "active", "exact_mode": "Copula", "orientation": "original"},
            {"pair_group_id": "", "exact_mode": "Copula", "orientation": "original"},
            {"pair_group_id": "", "exact_mode": "Copula", "orientation": "original"},
        ]
    )

    lookup = _unique_mode_rows(frame, allowed_pair_group_ids={"active"})
    assert list(lookup) == [("active", "Copula", "original")]

    active_duplicate = pd.concat([frame.iloc[[0]], frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate mode-ledger identity"):
        _unique_mode_rows(active_duplicate, allowed_pair_group_ids={"active"})


def test_funding_evidence_aligns_without_mutating_canonical_history(tmp_path):
    _build_fixture(tmp_path)
    build_exhaustive_wizard_hyperliquid_replay_preflight(root=tmp_path)
    history_result = materialize_exhaustive_wizard_hyperliquid_history(
        root=tmp_path,
        fetcher=_fake_candle_fetcher(rows=800),
        sleep=lambda _: None,
    )
    canonical_pair = pd.read_csv(history_result.paths["pair_results"]).iloc[0]
    canonical_path = tmp_path / canonical_pair["history_path"]
    canonical_hash = canonical_pair["history_sha256"]

    result = materialize_exhaustive_hyperliquid_funding_evidence(
        root=tmp_path,
        fetcher=_fake_funding_fetcher,
        sleep=lambda _: None,
    )

    assets = pd.read_csv(result.paths["asset_results"], keep_default_na=False)
    pairs = pd.read_csv(result.paths["pair_coverage"], keep_default_na=False)
    assert len(assets) == 2
    assert assets["funding_status"].eq("COMPLETE").all()
    assert pairs["funding_status"].eq("READY_FOR_COST_REPLAY").all()
    assert pairs["funding_both_coverage"].ge(0.95).all()
    assert pairs["funding_longest_contiguous_rows"].ge(250).all()
    enriched = json.loads((tmp_path / pairs.iloc[0]["enriched_history_path"]).read_text())
    assert all(row["funding_x_bps"] is not None for row in enriched["history"])
    assert all(row["funding_y_bps"] is not None for row in enriched["history"])
    assert all(row["funding_x_realized_bps"] is not None for row in enriched["history"])
    assert all(row["funding_y_realized_bps"] is not None for row in enriched["history"])
    assert enriched["funding_rate_semantics"] == "realized_bps_per_aligned_candle"
    assert canonical_path.exists()
    assert canonical_hash == hashlib.sha256(canonical_path.read_bytes()).hexdigest()
    assert result.summary["pair_status_accounted"] is True
    assert result.summary["pairs_unclassified_status"] == 0
    assert sum(result.summary["pair_status_counts"].values()) == result.summary["pair_work_items"]
    assert result.summary["live_trading_authorized"] is False


def test_funding_evidence_resumes_without_dropping_pending_assets(tmp_path):
    _build_fixture(tmp_path)
    build_exhaustive_wizard_hyperliquid_replay_preflight(root=tmp_path)
    materialize_exhaustive_wizard_hyperliquid_history(
        root=tmp_path,
        fetcher=_fake_candle_fetcher(rows=800),
        sleep=lambda _: None,
    )

    first = materialize_exhaustive_hyperliquid_funding_evidence(
        root=tmp_path,
        max_assets=1,
        fetcher=_fake_funding_fetcher,
        sleep=lambda _: None,
    )
    first_assets = pd.read_csv(first.paths["asset_results"], keep_default_na=False)
    first_pairs = pd.read_csv(first.paths["pair_coverage"], keep_default_na=False)
    assert first_assets["funding_status"].eq("COMPLETE").sum() == 1
    assert first_assets["funding_status"].eq("PENDING").sum() == 1
    assert first_pairs["funding_status"].eq("PENDING_ASSET_FUNDING").all()

    second = materialize_exhaustive_hyperliquid_funding_evidence(
        root=tmp_path,
        max_assets=1,
        fetcher=_fake_funding_fetcher,
        sleep=lambda _: None,
    )
    second_assets = pd.read_csv(second.paths["asset_results"], keep_default_na=False)
    second_pairs = pd.read_csv(second.paths["pair_coverage"], keep_default_na=False)
    assert second_assets["funding_status"].eq("COMPLETE").all()
    assert second_pairs["funding_status"].eq("READY_FOR_COST_REPLAY").all()


def test_cost_bridge_accounts_all_pairs_and_experiments_without_promoting_provisional_data(
    tmp_path,
):
    _build_fixture(tmp_path)
    active = tmp_path / "reports" / "active"
    mode_path = active / "exhaustive_wizard_pair_detail_mode_ledger.csv"
    mode_rows = pd.read_csv(mode_path, keep_default_na=False)
    retained_historical = mode_rows.iloc[[0]].copy()
    retained_historical["pair_group_id"] = ""
    pd.concat(
        [mode_rows, retained_historical, retained_historical],
        ignore_index=True,
    ).to_csv(mode_path, index=False)
    build_exhaustive_wizard_hyperliquid_replay_preflight(root=tmp_path)
    materialize_exhaustive_wizard_hyperliquid_history(
        root=tmp_path,
        fetcher=_fake_candle_fetcher(rows=800),
        sleep=lambda _: None,
    )
    run_exhaustive_wizard_hyperliquid_canonical_replay(root=tmp_path)
    materialize_exhaustive_hyperliquid_funding_evidence(
        root=tmp_path,
        fetcher=_fake_funding_fetcher,
        sleep=lambda _: None,
    )
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "asset_x": "BTC",
                "asset_y": "ETH",
                "fee_profile_id": "test-fee-v1",
                "taker_fee_bps": 4.5,
                "execution_risk_bps": 2.0,
                "fee_source_checked_at": "2026-08-07",
                "cost_model_status": "official_base_tier_conservative_fee_profile",
                "cost_model_ready": True,
                "slippage_model_status": "insufficient_l2_depth_samples",
                "slippage_model_ready": False,
                "slippage_samples_x": 2,
                "slippage_samples_y": 2,
                "required_slippage_samples": 12,
                "slippage_window_hours": 24,
                "slippage_x_p95_bps": 1.0,
                "slippage_y_p95_bps": 2.0,
                "pair_one_way_slippage_bps": 1.5,
                "estimated_pair_round_trip_cost_bps": 16.0,
                "freshest_sample_at": "2026-08-08T04:30:00+00:00",
                "evidence_path": "data/processed/l2.csv",
            }
        ]
    ).to_csv(active / "hyperliquid_pair_cost_model.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "status": "waiting_for_next_capture",
                "next_sample_due_at": "2026-08-08T04:40:00+00:00",
                "evidence_path": "reports/active/cadence.csv",
            }
        ]
    ).to_csv(active / "hyperliquid_evidence_cadence.csv", index=False)

    result = build_exhaustive_wizard_hyperliquid_cost_evidence(
        root=tmp_path,
        now=datetime(2026, 8, 8, 5, tzinfo=timezone.utc),
    )

    pairs = pd.read_csv(result.paths["pair_cost_evidence"], keep_default_na=False)
    experiments = pd.read_csv(result.paths["experiment_cost_readiness"], keep_default_na=False)
    assert len(pairs) == 1
    assert pairs["provisional_cost_research_ready"].map(bool).all()
    assert not pairs["cost_acceptance_ready"].map(bool).any()
    assert pairs["cost_evidence_status"].eq("READY_FOR_PROVISIONAL_COST_RESEARCH").all()
    assert len(experiments) == len(EXACT_MODES) * len(ORIENTATIONS)
    assert experiments["experiment_id"].nunique() == len(experiments)
    assert experiments["cost_replay_status"].eq("READY_FOR_PROVISIONAL_COST_RESEARCH").sum() == 14
    assert experiments["cost_replay_status"].eq("NOT_APPLICABLE_WIZARD_MODE").sum() == 0
    assert result.summary["pair_status_accounted"] is True
    assert result.summary["experiment_status_accounted"] is True
    assert result.summary["pairs_ready_for_cost_calibrated_replay"] == 0
    assert result.summary["live_trading_authorized"] is False

    replay = run_exhaustive_wizard_hyperliquid_observed_cost_replay(
        root=tmp_path,
        now=datetime(2026, 8, 8, 5, 5, tzinfo=timezone.utc),
    )
    replay_rows = pd.read_csv(replay.paths["results"], keep_default_na=False)
    assert len(replay_rows) == len(EXACT_MODES) * len(ORIENTATIONS)
    assert replay_rows["experiment_id"].nunique() == len(replay_rows)
    assert replay_rows["replay_status"].eq("OBSERVED_COST_RESEARCH_REPLAY_COMPLETE").sum() == 10
    assert replay_rows["replay_status"].eq("BLOCKED_DYNAMIC_EXPOSURE_RULE").sum() == 4
    assert replay_rows["replay_status"].eq("NOT_APPLICABLE_WIZARD_MODE").sum() == 0
    completed = replay_rows.loc[
        replay_rows["replay_status"].eq("OBSERVED_COST_RESEARCH_REPLAY_COMPLETE")
    ]
    assert completed["funding_policy"].eq("signed_realized").all()
    assert completed["acceptance_status"].eq("BLOCKED").all()
    assert not replay_rows["live_trading_authorized"].astype(bool).any()
    assert replay.summary["experiment_status_accounted"] is True
    assert replay.summary["acceptance_eligible_replays"] == 0

    walkforward = run_exhaustive_wizard_hyperliquid_walkforward(
        root=tmp_path,
        now=datetime(2026, 8, 8, 5, 10, tzinfo=timezone.utc),
    )
    statuses = pd.read_csv(walkforward.paths["status"], keep_default_na=False)
    candidates = pd.read_csv(walkforward.paths["candidates"], keep_default_na=False)
    folds = pd.read_csv(walkforward.paths["folds"], keep_default_na=False)
    assert len(statuses) == len(EXACT_MODES) * len(ORIENTATIONS)
    assert statuses["experiment_id"].nunique() == len(statuses)
    assert len(candidates) == 10
    assert len(folds) == len(candidates) * FOLD_COUNT
    assert folds.groupby("experiment_id")["fold_number"].nunique().eq(FOLD_COUNT).all()
    assert not folds["fit_uses_test_data"].astype(bool).any()
    copula_folds = folds.loc[folds["exact_mode"].eq("Copula")]
    assert pd.to_numeric(copula_folds["fitted_hedge_ratio"], errors="coerce").notna().all()
    assert candidates["walkforward_status"].isin(
        {"PASS_RESEARCH_WALK_FORWARD", "FAIL_RESEARCH_WALK_FORWARD"}
    ).all()
    assert candidates["acceptance_status"].eq("BLOCKED").all()
    assert {
        "fold_return_raw_pvalue",
        "bh_qvalue",
        "parameter_stability_status",
        "deflated_sharpe_status",
        "statistical_selection_status",
        "statistical_selection_blocker",
    }.issubset(candidates.columns)
    assert not statuses["live_trading_authorized"].astype(bool).any()
    assert walkforward.summary["experiment_status_accounted"] is True
    assert walkforward.summary["selection_hindsight_used_for_fold_execution"] is False
    assert walkforward.summary["acceptance_eligible_replays"] == 0
    evaluated_walkforward = statuses["walkforward_status"].isin(
        {"PASS_RESEARCH_WALK_FORWARD", "FAIL_RESEARCH_WALK_FORWARD"}
    )
    assert statuses.loc[evaluated_walkforward, "acceptance_reason"].str.contains(
        "current_l2_depth_is_point_in_time_not_historical_execution_evidence"
    ).all()
    assert not statuses["acceptance_reason"].str.contains(
        "l2_slippage_calibration_incomplete"
    ).any()

    regimes = build_exhaustive_wizard_hyperliquid_regime_attribution(
        root=tmp_path,
        now=datetime(2026, 8, 8, 5, 15, tzinfo=timezone.utc),
    )
    regime_status = pd.read_csv(regimes.paths["status"], keep_default_na=False)
    assert len(regime_status) == len(EXACT_MODES) * len(ORIENTATIONS)
    assert regime_status["experiment_id"].nunique() == len(regime_status)
    assert regimes.summary["experiment_status_accounted"] is True
    assert regimes.summary["acceptance_eligible_replays"] == 0

    robustness = run_exhaustive_wizard_hyperliquid_robustness(
        root=tmp_path,
        now=datetime(2026, 8, 8, 5, 20, tzinfo=timezone.utc),
    )
    robustness_status = pd.read_csv(robustness.paths["status"], keep_default_na=False)
    assert len(robustness_status) == len(EXACT_MODES) * len(ORIENTATIONS)
    assert robustness_status["experiment_id"].nunique() == len(robustness_status)
    assert robustness.summary["experiment_status_accounted"] is True
    assert robustness.summary["acceptance_eligible_replays"] == 0
    assert robustness.summary["live_trading_authorized"] is False

    concentration = build_exhaustive_wizard_hyperliquid_concentration(
        root=tmp_path,
        now=datetime(2026, 8, 8, 5, 25, tzinfo=timezone.utc),
    )
    concentration_status = pd.read_csv(
        concentration.paths["status"], keep_default_na=False
    )
    concentration_dimensions = pd.read_csv(
        concentration.paths["dimensions"], keep_default_na=False
    )
    assert len(concentration_status) == len(EXACT_MODES) * len(ORIENTATIONS)
    assert concentration_status["experiment_id"].nunique() == len(concentration_status)
    assert len(concentration_dimensions) == 3 * 6
    assert concentration.summary["experiment_status_accounted"] is True
    assert concentration.summary["promotion_concentration_pass"] is False
    assert concentration.summary["ready_for_leverage_gate"] is False
    assert concentration.summary["acceptance_eligible_replays"] == 0
    assert concentration.summary["live_trading_authorized"] is False

    leverage = build_exhaustive_wizard_hyperliquid_leverage_surface(
        root=tmp_path,
        now=datetime(2026, 8, 8, 5, 30, tzinfo=timezone.utc),
    )
    leverage_status = pd.read_csv(leverage.paths["status"], keep_default_na=False)
    leverage_scenarios = pd.read_csv(
        leverage.paths["scenarios"], keep_default_na=False
    )
    assert len(leverage_status) == len(EXACT_MODES) * len(ORIENTATIONS)
    assert leverage_status["experiment_id"].nunique() == len(leverage_status)
    assert leverage_scenarios.empty
    assert leverage.summary["experiment_status_accounted"] is True
    assert leverage.summary["concentration_ready_candidates"] == 0
    assert leverage.summary["testnet_1x_lifecycle_ready"] == 0
    assert leverage.summary["acceptance_eligible_replays"] == 0
    assert leverage.summary["live_trading_authorized"] is False


def test_zero_survivor_chain_writes_schema_bearing_artifacts(tmp_path):
    _build_fixture(tmp_path)
    build_exhaustive_wizard_hyperliquid_replay_preflight(root=tmp_path)
    materialize_exhaustive_wizard_hyperliquid_history(
        root=tmp_path,
        fetcher=_fake_candle_fetcher(rows=800),
        sleep=lambda _: None,
    )
    run_exhaustive_wizard_hyperliquid_canonical_replay(root=tmp_path)
    materialize_exhaustive_hyperliquid_funding_evidence(
        root=tmp_path,
        fetcher=_fake_funding_fetcher,
        sleep=lambda _: None,
    )
    build_exhaustive_wizard_hyperliquid_cost_evidence(root=tmp_path)
    observed = run_exhaustive_wizard_hyperliquid_observed_cost_replay(root=tmp_path)
    assert observed.summary["observed_cost_replays_complete"] == 0

    walkforward = run_exhaustive_wizard_hyperliquid_walkforward(root=tmp_path)
    candidates = pd.read_csv(walkforward.paths["candidates"], keep_default_na=False)
    folds = pd.read_csv(walkforward.paths["folds"], keep_default_na=False)
    bars = pd.read_csv(walkforward.paths["bars"], keep_default_na=False)
    assert candidates.empty and "experiment_id" in candidates.columns
    assert folds.empty and "fold_number" in folds.columns
    assert bars.empty and "timestamp" in bars.columns

    regimes = build_exhaustive_wizard_hyperliquid_regime_attribution(root=tmp_path)
    regime_candidates = pd.read_csv(regimes.paths["candidates"], keep_default_na=False)
    assert regime_candidates.empty and "experiment_id" in regime_candidates.columns

    robustness = run_exhaustive_wizard_hyperliquid_robustness(root=tmp_path)
    robustness_candidates = pd.read_csv(
        robustness.paths["candidates"], keep_default_na=False
    )
    scenarios = pd.read_csv(robustness.paths["scenarios"], keep_default_na=False)
    assert robustness_candidates.empty and "experiment_id" in robustness_candidates.columns
    assert scenarios.empty and "scenario" in scenarios.columns

    concentration = build_exhaustive_wizard_hyperliquid_concentration(root=tmp_path)
    assert concentration.summary["experiment_status_accounted"] is True
    assert concentration.summary["ready_for_leverage_gate"] is False

    leverage = build_exhaustive_wizard_hyperliquid_leverage_surface(root=tmp_path)
    assert leverage.summary["experiment_status_accounted"] is True
    assert leverage.summary["testnet_1x_lifecycle_ready"] == 0

    validation = (
        exhaustive_wizard_hyperliquid_validation.run_exhaustive_wizard_hyperliquid_validation(
            root=tmp_path
        )
    )
    assert validation.summary["stage_accounting_complete"] is True
    assert validation.summary["learning_ledger"]["records"] == len(EXACT_MODES) * len(
        ORIENTATIONS
    )
    assert validation.summary["learning_ledger"]["training_eligible_records"] == 0
