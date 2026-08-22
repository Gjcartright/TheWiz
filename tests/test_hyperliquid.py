import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from quant_platform.hyperliquid import (
    build_hyperliquid_capacity_curve,
    build_hyperliquid_evidence_cadence,
    build_hyperliquid_lane_report,
    build_hyperliquid_pair_cost_model,
    build_hyperliquid_pair_history,
    build_hyperliquid_research_bundle,
    normalize_hyperliquid_candles,
    normalize_hyperliquid_funding_history,
    refresh_hyperliquid_execution_cost_snapshot,
    refresh_hyperliquid_funding_history,
    refresh_hyperliquid_market_context,
)


def test_normalize_hyperliquid_candles_outputs_canonical_candles():
    payload = [
        {"t": 1761091200000, "o": "10", "h": "12", "l": "9", "c": "11", "v": "100"},
        {"t": 1761177600000, "o": "11", "h": "13", "l": "10", "c": "12", "v": "200"},
    ]

    candles = normalize_hyperliquid_candles(payload, coin="HYPE", interval="1d")

    assert candles[0]["ticker"] == "HYPE-USD"
    assert candles[0]["resolution"] == "1d"
    assert candles[0]["close"] == 11.0
    assert candles[0]["usdVolume"] == 1100.0
    assert candles[0]["source"] == "hyperliquid"


def test_normalize_hyperliquid_funding_history_converts_hourly_rate_to_bps():
    rows = normalize_hyperliquid_funding_history(
        [{"coin": "BTC", "fundingRate": "0.0000125", "premium": "-0.0002", "time": 1761091200000}],
        coin="BTC",
        evidence_path="raw/btc.json",
    )

    assert len(rows) == 1
    assert rows[0]["market"] == "BTC-USD"
    assert rows[0]["funding_bps"] == 0.125
    assert rows[0]["timestamp"] == "2025-10-22T00:00:00+00:00"
    assert rows[0]["evidence_path"] == "raw/btc.json"


def test_build_hyperliquid_pair_history_marks_exchange_and_reuses_pair_engine(tmp_path):
    candle_dir = tmp_path / "candles"
    candle_dir.mkdir()
    timestamps = [1761091200000, 1761177600000, 1761264000000, 1761350400000, 1761436800000]
    left = normalize_hyperliquid_candles(
        [{"t": ts, "o": 10 + idx, "h": 11 + idx, "l": 9 + idx, "c": 10 + idx, "v": 100} for idx, ts in enumerate(timestamps)],
        coin="HYPE",
        interval="1d",
    )
    right = normalize_hyperliquid_candles(
        [{"t": ts, "o": 2 + idx, "h": 3 + idx, "l": 1 + idx, "c": 2 + idx, "v": 200} for idx, ts in enumerate(timestamps)],
        coin="TRX",
        interval="1d",
    )
    (candle_dir / "HYPE_1d_candles.json").write_text(json.dumps({"candles": left}), encoding="utf-8")
    (candle_dir / "TRX_1d_candles.json").write_text(json.dumps({"candles": right}), encoding="utf-8")

    path = build_hyperliquid_pair_history(
        asset_x="HYPE-USD",
        asset_y="TRX-USD",
        interval="1d",
        candle_dir=candle_dir,
        output_dir=tmp_path,
        zscore_window=3,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["exchange"] == "hyperliquid"
    assert payload["asset_x"] == "HYPE-USD"
    assert payload["asset_y"] == "TRX-USD"
    assert len(payload["history"]) == 5
    assert {
        "price_x",
        "price_y",
        "spread",
        "zscore",
        "research_proxy_ecm_x",
        "research_proxy_ecm_y",
        "research_proxy_ecm_strength",
    }.issubset(payload["history"][0])
    assert "ecm_x" not in payload["history"][0]


def test_hyperliquid_lane_report_blocks_without_daily_history(tmp_path):
    reports = tmp_path / "reports" / "active"
    reports.mkdir(parents=True)
    (reports / "venue_lane_classification.csv").write_text(
        "asset,best_lane,dydx_lane,hyperliquid_lane,blockers,next_action\n"
        "HYPE,hyperliquid_research_candidate,blocked_liquidity,hyperliquid_research_candidate,missing_hyperliquid_local_replay,build_hyperliquid_history_and_cost_model\n",
        encoding="utf-8",
    )

    result = build_hyperliquid_lane_report(root=tmp_path)

    text = result.paths["hyperliquid_lane_readiness"].read_text(encoding="utf-8")
    assert "missing_hyperliquid_daily_history" in text


def test_refresh_hyperliquid_market_context_captures_public_snapshot_without_promotion(tmp_path):
    captured_at = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
    payload = [
        {"universe": [{"name": "BTC"}, {"name": "ETH", "isDelisted": True}]},
        [
            {"markPx": "100", "openInterest": "2", "dayNtlVlm": "1000000", "funding": "0.0001"},
            {"markPx": "10", "openInterest": "3", "dayNtlVlm": "200000", "funding": "-0.0002"},
        ],
    ]

    result = refresh_hyperliquid_market_context(root=tmp_path, payload=payload, captured_at=captured_at)

    frame = pd.read_csv(result.paths["hyperliquid_market_context"])
    btc = frame.set_index("asset").loc["BTC"]
    eth = frame.set_index("asset").loc["ETH"]
    assert btc["open_interest_usd"] == 200.0
    assert btc["funding_rate"] == 0.0001
    assert bool(btc["promotion_allowed"]) is False
    assert bool(eth["tradable"]) is False
    assert "requires_pair_history_cost_slippage_and_preflight" in btc["blocker"]
    assert result.paths["raw_snapshot"].exists()


def test_hyperliquid_research_bundle_builds_fresh_local_history_without_execution_authority(tmp_path):
    active = tmp_path / "reports" / "active"
    candle_dir = tmp_path / "data" / "raw" / "hyperliquid_candles"
    active.mkdir(parents=True)
    candle_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "asset_x": "BTC",
                "asset_y": "ETH",
                "both_legs_testnet_perp": True,
            }
        ]
    ).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for coin, offset in (("BTC", 100.0), ("ETH", 50.0)):
        candles = normalize_hyperliquid_candles(
            [
                {
                    "t": int((start + timedelta(days=index)).timestamp() * 1000),
                    "o": offset + index,
                    "h": offset + index + 1,
                    "l": offset + index - 1,
                    "c": offset + index + 0.5,
                    "v": 1000,
                }
                for index in range(125)
            ],
            coin=coin,
            interval="1d",
        )
        (candle_dir / f"{coin}_1d_candles.json").write_text(
            json.dumps({"candles": candles, "fetched_at": datetime.now(timezone.utc).isoformat()}),
            encoding="utf-8",
        )

    result = build_hyperliquid_research_bundle(
        root=tmp_path,
        max_pairs=1,
        intervals=("1d",),
        refresh=False,
    )

    frame = pd.read_csv(result.paths["hyperliquid_research_bundle"])
    assert len(frame) == 1
    assert bool(frame.loc[0, "history_ready"]) is True
    assert frame.loc[0, "history_status"] == "history_ready_for_local_replay"
    assert frame.loc[0, "next_step"] == "collect_venue_cost_slippage_funding_and_preflight"
    assert (tmp_path / frame.loc[0, "history_path"]).exists()


def test_explicit_wizard_symbols_route_to_hyperliquid_without_fallback(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    candle_dir = tmp_path / "data" / "raw" / "hyperliquid_candles"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    candle_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"asset": "BTC", "tradable": True},
            {"asset": "EIGEN", "tradable": True},
        ]
    ).to_csv(processed / "hyperliquid_market_context.csv", index=False)
    candidate_path = active / "wizard_candidates.csv"
    pd.DataFrame(
        [
            {
                "pair": "BTCUSDT / EIGENPERP",
                "asset_x": "BTCUSDT",
                "asset_y": "EIGENPERP",
            }
        ]
    ).to_csv(candidate_path, index=False)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for coin, offset in (("BTC", 100.0), ("EIGEN", 10.0)):
        candles = normalize_hyperliquid_candles(
            [
                {
                    "t": int((start + timedelta(hours=index)).timestamp() * 1000),
                    "o": offset + index / 100,
                    "h": offset + index / 100 + 0.1,
                    "l": offset + index / 100 - 0.1,
                    "c": offset + index / 100 + 0.05,
                    "v": 1000,
                }
                for index in range(1100)
            ],
            coin=coin,
            interval="1h",
        )
        (candle_dir / f"{coin}_1h_candles.json").write_text(
            json.dumps({"candles": candles, "fetched_at": datetime.now(timezone.utc).isoformat()}),
            encoding="utf-8",
        )

    result = build_hyperliquid_research_bundle(
        root=tmp_path,
        max_pairs=2,
        intervals=("1h",),
        refresh=False,
        candidate_path=candidate_path,
    )
    frame = pd.read_csv(result.paths["hyperliquid_research_bundle"])

    assert frame[["asset_x", "asset_y"]].to_dict(orient="records") == [
        {"asset_x": "BTC", "asset_y": "EIGEN"}
    ]
    assert frame.loc[0, "candidate_source"] == "explicit_hyperliquid_candidate_file"


def test_explicit_hyperliquid_candidates_never_fall_back_to_unrelated_pairs(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "ETH-USD-SOL-USD",
                "asset_x": "ETH",
                "asset_y": "SOL",
                "both_legs_testnet_perp": True,
            }
        ]
    ).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)
    candidate_path = active / "wizard_candidates.csv"
    pd.DataFrame(
        [{"pair": "UNKNOWNUSDT / MISSINGPERP", "asset_x": "UNKNOWNUSDT", "asset_y": "MISSINGPERP"}]
    ).to_csv(candidate_path, index=False)

    result = build_hyperliquid_research_bundle(
        root=tmp_path,
        max_pairs=2,
        intervals=("1h",),
        refresh=False,
        candidate_path=candidate_path,
    )
    frame = pd.read_csv(result.paths["hyperliquid_research_bundle"])

    assert frame.empty


def test_refresh_hyperliquid_funding_history_merges_daily_rates_without_future_fill(tmp_path):
    active = tmp_path / "reports" / "active"
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    active.mkdir(parents=True)
    pair_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"asset": "BTC", "tradable_perp": True},
            {"asset": "ETH", "tradable_perp": True},
        ]
    ).to_csv(active / "hyperliquid_testnet_market_inventory.csv", index=False)
    pd.DataFrame(
        [{"pair": "BTC-USD-ETH-USD", "asset_x": "BTC", "asset_y": "ETH", "both_legs_testnet_perp": True}]
    ).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)
    candidate_path = active / "candidates.csv"
    pd.DataFrame([{"dashboard_rank": 1, "pair": "BTC-USD/ETH-USD", "asset_x": "BTC-USD", "asset_y": "ETH-USD"}]).to_csv(
        candidate_path, index=False
    )
    history_path = pair_dir / "pair_btc_eth_hyperliquid_1d_derived_history.json"
    history_path.write_text(
        json.dumps(
            {
                "pair": "BTC-USD-ETH-USD",
                "asset_x": "BTC-USD",
                "asset_y": "ETH-USD",
                "exchange": "hyperliquid",
                "interval": "1d",
                "history": [
                    {"timestamp": "2025-10-21T00:00:00Z", "funding_bps_per_day": 0.0},
                    {"timestamp": "2025-10-22T00:00:00Z", "funding_bps_per_day": 0.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    payloads = {
        "BTC": [
            {"coin": "BTC", "fundingRate": "0.00001", "time": 1761091200000},
            {"coin": "BTC", "fundingRate": "0.00002", "time": 1761094800000},
        ],
        "ETH": [{"coin": "ETH", "fundingRate": "-0.00003", "time": 1761091200000}],
    }

    result = refresh_hyperliquid_funding_history(
        root=tmp_path,
        max_pairs=1,
        candidate_path=candidate_path,
        funding_payloads=payloads,
        end_time=datetime(2025, 10, 23, tzinfo=timezone.utc),
    )

    merged = json.loads(history_path.read_text(encoding="utf-8"))
    assert merged["history"][0]["funding_x_bps"] is None
    assert merged["history"][0]["funding_y_bps"] is None
    assert merged["history"][1]["funding_x_bps"] == pytest.approx(0.3)
    assert merged["history"][1]["funding_y_bps"] == pytest.approx(-0.3)
    assert merged["history"][1]["funding_bps_per_day"] == pytest.approx(0.6)
    assert merged["funding_alignment"] == "same_utc_day_no_future_fill"
    coverage = pd.read_csv(result.paths["hyperliquid_funding_coverage"])
    assert coverage.loc[0, "funding_coverage_pct"] == 50.0
    assert bool(coverage.loc[0, "funding_ready"]) is False
    assert "hyperliquid_funding_coverage_below_95pct" in coverage.loc[0, "blocker"]


def test_hyperliquid_cost_snapshot_requires_repeated_depth_samples_before_slippage_ready(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "asset_x": "BTC",
                "asset_y": "ETH",
                "both_legs_testnet_perp": True,
            }
        ]
    ).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)
    start = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)

    def book(coin: str, timestamp: datetime, bid: float, ask: float) -> dict[str, object]:
        return {
            "coin": coin,
            "time": int(timestamp.timestamp() * 1000),
            "levels": [
                [{"px": str(bid), "sz": "100"}, {"px": str(bid - 1), "sz": "100"}],
                [{"px": str(ask), "sz": "100"}, {"px": str(ask + 1), "sz": "100"}],
            ],
        }

    first = refresh_hyperliquid_execution_cost_snapshot(
        root=tmp_path,
        max_pairs=1,
        notionals=(1000.0,),
        captured_at=start,
        book_payloads={"BTC": book("BTC", start, 99.0, 101.0), "ETH": book("ETH", start, 49.0, 51.0)},
    )
    initial = pd.read_csv(first.paths["hyperliquid_pair_cost_model"])
    assert bool(initial.loc[0, "cost_model_ready"]) is True
    assert bool(initial.loc[0, "slippage_model_ready"]) is False
    assert initial.loc[0, "slippage_samples_x"] == 1
    assert initial.loc[0, "slippage_samples_y"] == 1
    assert initial.loc[0, "taker_fee_bps"] == 4.5

    for step in range(1, 12):
        timestamp = start + timedelta(minutes=5 * step)
        refresh_hyperliquid_execution_cost_snapshot(
            root=tmp_path,
            max_pairs=1,
            notionals=(1000.0,),
            captured_at=timestamp,
            book_payloads={"BTC": book("BTC", timestamp, 99.0, 101.0), "ETH": book("ETH", timestamp, 49.0, 51.0)},
        )

    result = build_hyperliquid_pair_cost_model(root=tmp_path, max_pairs=1, leg_notional_usd=1000.0, as_of=start + timedelta(hours=1))
    calibrated = pd.read_csv(result.paths["hyperliquid_pair_cost_model"])
    assert bool(calibrated.loc[0, "slippage_model_ready"]) is True
    assert calibrated.loc[0, "slippage_model_status"] == "l2_depth_calibrated_p95"
    assert calibrated.loc[0, "slippage_samples_x"] == 12
    assert calibrated.loc[0, "slippage_samples_y"] == 12
    assert calibrated.loc[0, "estimated_pair_round_trip_cost_bps"] > 0


def test_hyperliquid_capacity_curve_keeps_notional_and_funding_gates_separate(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "asset_x": "BTC",
                "asset_y": "ETH",
                "both_legs_testnet_perp": True,
            }
        ]
    ).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "asset_x": "BTC",
                "asset_y": "ETH",
                "funding_coverage_pct": 100.0,
                "funding_ready": True,
            }
        ]
    ).to_csv(active / "hyperliquid_funding_coverage.csv", index=False)
    now = datetime(2026, 8, 5, 12, 30, tzinfo=timezone.utc)
    rows = []
    for asset in ("BTC", "ETH"):
        for notional in (100.0, 1000.0):
            for minute in (10, 20):
                rows.append(
                    {
                        "pair": "BTC-USD-ETH-USD",
                        "asset": asset,
                        "notional_usd": notional,
                        "source_timestamp": now.replace(minute=minute).isoformat(),
                        "buy_complete": True,
                        "sell_complete": True,
                        "one_way_slippage_bps": 1.0 + notional / 1000.0,
                    }
                )
    pd.DataFrame(rows).to_csv(
        processed / "hyperliquid_l2_slippage_samples.csv", index=False
    )

    result = build_hyperliquid_capacity_curve(
        root=tmp_path,
        max_pairs=1,
        notionals=(100.0, 500.0, 1000.0),
        min_samples=2,
        as_of=now,
    )
    curve = pd.read_csv(result.paths["hyperliquid_capacity_curve"])

    assert set(curve["leg_notional_usd"]) == {100.0, 500.0, 1000.0}
    assert curve.loc[curve["leg_notional_usd"].isin([100.0, 1000.0]), "depth_cost_ready"].all()
    assert curve.loc[curve["leg_notional_usd"].eq(500.0), "funding_ready"].all()
    assert not curve.loc[curve["leg_notional_usd"].eq(500.0), "depth_cost_ready"].any()
    assert result.summary["acceptance_ready_rows"] == 2


def test_hyperliquid_snapshot_model_includes_books_returned_after_request_start(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "asset_x": "BTC",
                "asset_y": "ETH",
                "both_legs_testnet_perp": True,
            }
        ]
    ).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)
    request_started = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
    book_timestamp = request_started + timedelta(seconds=30)

    def book(coin: str, bid: str, ask: str) -> dict[str, object]:
        return {
            "coin": coin,
            "time": int(book_timestamp.timestamp() * 1000),
            "levels": [[{"px": bid, "sz": "100"}], [{"px": ask, "sz": "100"}]],
        }

    result = refresh_hyperliquid_execution_cost_snapshot(
        root=tmp_path,
        max_pairs=1,
        notionals=(1000.0,),
        captured_at=request_started,
        book_payloads={"BTC": book("BTC", "99", "101"), "ETH": book("ETH", "49", "51")},
        min_samples=1,
        window_hours=2,
    )
    model = pd.read_csv(result.paths["hyperliquid_pair_cost_model"])

    assert model.loc[0, "slippage_samples_x"] == 1
    assert model.loc[0, "slippage_samples_y"] == 1
    assert bool(model.loc[0, "slippage_model_ready"]) is True
    assert model.loc[0, "freshest_sample_at"] == book_timestamp.isoformat()
    assert result.summary["model_as_of"] == book_timestamp.isoformat()


def test_explicit_cost_capture_retains_other_pair_models(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "asset_x": "BTC",
                "asset_y": "ETH",
                "both_legs_testnet_perp": True,
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "asset_x": "SOL",
                "asset_y": "HYPE",
                "both_legs_testnet_perp": True,
            },
        ]
    ).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)
    first_candidates = tmp_path / "first_candidates.csv"
    second_candidates = tmp_path / "second_candidates.csv"
    pd.DataFrame(
        [{"pair": "BTC-USD-ETH-USD", "asset_x": "BTC", "asset_y": "ETH"}]
    ).to_csv(first_candidates, index=False)
    pd.DataFrame(
        [{"pair": "SOL-USD-HYPE-USD", "asset_x": "SOL", "asset_y": "HYPE"}]
    ).to_csv(second_candidates, index=False)
    start = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)

    def book(coin: str, timestamp: datetime, bid: str, ask: str) -> dict[str, object]:
        return {
            "coin": coin,
            "time": int(timestamp.timestamp() * 1000),
            "levels": [[{"px": bid, "sz": "1000"}], [{"px": ask, "sz": "1000"}]],
        }

    refresh_hyperliquid_execution_cost_snapshot(
        root=tmp_path,
        max_pairs=1,
        notionals=(1000.0,),
        captured_at=start,
        book_payloads={
            "BTC": book("BTC", start, "99", "101"),
            "ETH": book("ETH", start, "49", "51"),
        },
        candidate_path=first_candidates,
        min_samples=2,
        window_hours=2,
    )
    result = refresh_hyperliquid_execution_cost_snapshot(
        root=tmp_path,
        max_pairs=1,
        notionals=(1000.0,),
        captured_at=start + timedelta(minutes=5),
        book_payloads={
            "SOL": book("SOL", start + timedelta(minutes=5), "19", "21"),
            "HYPE": book("HYPE", start + timedelta(minutes=5), "9", "11"),
        },
        candidate_path=second_candidates,
        min_samples=2,
        window_hours=2,
    )

    model = pd.read_csv(result.paths["hyperliquid_pair_cost_model"])
    by_pair = model.set_index("pair")
    assert set(by_pair.index) == {"BTC-USD-ETH-USD", "SOL-USD-HYPE-USD"}
    assert by_pair.loc["BTC-USD-ETH-USD", "slippage_samples_x"] == 1
    assert by_pair.loc["BTC-USD-ETH-USD", "slippage_samples_y"] == 1
    assert by_pair.loc["SOL-USD-HYPE-USD", "slippage_samples_x"] == 1
    assert by_pair.loc["SOL-USD-HYPE-USD", "slippage_samples_y"] == 1
    assert not bool(by_pair["slippage_model_ready"].any())


def test_explicit_mainnet_candidate_is_kept_when_testnet_is_missing_and_failed_discovery_is_filtered(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    candidate_path = tmp_path / "candidates.csv"
    pd.DataFrame(
        [
            {
                "pair": "BNB-USD/ETHFI-USD",
                "asset_x": "BNB-USD",
                "asset_y": "ETHFI-USD",
                "wizard_discovery_pass": True,
            },
            {
                "pair": "BNB-USD/FAIL-USD",
                "asset_x": "BNB-USD",
                "asset_y": "FAIL-USD",
                "wizard_discovery_pass": False,
            },
        ]
    ).to_csv(candidate_path, index=False)
    pd.DataFrame(
        [
            {"asset": "BNB", "tradable": True},
            {"asset": "ETHFI", "tradable": True},
            {"asset": "FAIL", "tradable": True},
        ]
    ).to_csv(processed / "hyperliquid_market_context.csv", index=False)
    pd.DataFrame([{"asset": "BNB", "tradable_perp": True}]).to_csv(
        active / "hyperliquid_testnet_market_inventory.csv", index=False
    )

    result = build_hyperliquid_pair_cost_model(
        root=tmp_path,
        max_pairs=10,
        candidate_path=candidate_path,
    )
    frame = pd.read_csv(result.paths["hyperliquid_pair_cost_model"])

    assert frame["pair"].tolist() == ["BNB-USD/ETHFI-USD"]
    assert bool(frame.loc[0, "cost_model_ready"]) is True
    assert bool(frame.loc[0, "slippage_model_ready"]) is False


def test_hyperliquid_evidence_cadence_marks_next_l2_capture_due(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [{"pair": "BTC-USD-ETH-USD", "asset_x": "BTC", "asset_y": "ETH", "both_legs_testnet_perp": True}]
    ).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)
    captured = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
    payloads = {
        "BTC": {
            "coin": "BTC",
            "time": int(captured.timestamp() * 1000),
            "levels": [[{"px": "99", "sz": "100"}], [{"px": "101", "sz": "100"}]],
        },
        "ETH": {
            "coin": "ETH",
            "time": int(captured.timestamp() * 1000),
            "levels": [[{"px": "49", "sz": "100"}], [{"px": "51", "sz": "100"}]],
        },
    }
    refresh_hyperliquid_execution_cost_snapshot(
        root=tmp_path,
        max_pairs=1,
        notionals=(1000.0,),
        captured_at=captured,
        book_payloads=payloads,
    )

    result = build_hyperliquid_evidence_cadence(
        root=tmp_path,
        target_samples=2,
        cadence_minutes=10,
        window_hours=2,
        as_of=captured + timedelta(minutes=10),
    )

    cadence = pd.read_csv(result.paths["hyperliquid_evidence_cadence"])
    assert cadence.loc[0, "status"] == "capture_due"
    assert cadence.loc[0, "required_samples"] == 1
    assert cadence.loc[0, "projected_captures_to_calibration"] == 1
    assert bool(cadence.loc[0, "projected_capture_feasible"]) is True
    assert cadence.loc[0, "next_step"] == "run refresh-hyperliquid-execution-cost-snapshot"


def test_hyperliquid_cadence_projection_accounts_for_sample_expiry(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [{"pair": "BTC-USD-ETH-USD", "asset_x": "BTC", "asset_y": "ETH", "both_legs_testnet_perp": True}]
    ).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)
    start = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)

    def payloads(timestamp: datetime) -> dict[str, object]:
        return {
            "BTC": {
                "coin": "BTC",
                "time": int(timestamp.timestamp() * 1000),
                "levels": [[{"px": "99", "sz": "100"}], [{"px": "101", "sz": "100"}]],
            },
            "ETH": {
                "coin": "ETH",
                "time": int(timestamp.timestamp() * 1000),
                "levels": [[{"px": "49", "sz": "100"}], [{"px": "51", "sz": "100"}]],
            },
        }

    for timestamp in (start, start + timedelta(minutes=95), start + timedelta(minutes=115)):
        refresh_hyperliquid_execution_cost_snapshot(
            root=tmp_path,
            max_pairs=1,
            notionals=(1000.0,),
            captured_at=timestamp,
            book_payloads=payloads(timestamp),
            min_samples=4,
            window_hours=2,
        )

    result = build_hyperliquid_evidence_cadence(
        root=tmp_path,
        target_samples=4,
        cadence_minutes=10,
        window_hours=2,
        as_of=start + timedelta(minutes=120),
    )
    cadence = pd.read_csv(result.paths["hyperliquid_evidence_cadence"])

    assert cadence.loc[0, "minimum_samples"] == 3
    assert cadence.loc[0, "required_samples"] == 1
    assert cadence.loc[0, "projected_captures_to_calibration"] == 2
    assert cadence.loc[0, "projected_calibration_at"] == (start + timedelta(minutes=135)).isoformat()


def test_hyperliquid_cadence_blocks_an_infeasible_schedule(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [{"pair": "BTC-USD-ETH-USD", "asset_x": "BTC", "asset_y": "ETH", "both_legs_testnet_perp": True}]
    ).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)
    captured = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
    refresh_hyperliquid_execution_cost_snapshot(
        root=tmp_path,
        max_pairs=1,
        notionals=(1000.0,),
        captured_at=captured,
        book_payloads={
            "BTC": {
                "coin": "BTC",
                "time": int(captured.timestamp() * 1000),
                "levels": [[{"px": "99", "sz": "100"}], [{"px": "101", "sz": "100"}]],
            },
            "ETH": {
                "coin": "ETH",
                "time": int(captured.timestamp() * 1000),
                "levels": [[{"px": "49", "sz": "100"}], [{"px": "51", "sz": "100"}]],
            },
        },
        min_samples=14,
        window_hours=2,
    )

    result = build_hyperliquid_evidence_cadence(
        root=tmp_path,
        target_samples=14,
        cadence_minutes=10,
        window_hours=2,
        as_of=captured,
    )
    cadence = pd.read_csv(result.paths["hyperliquid_evidence_cadence"])

    assert cadence.loc[0, "status"] == "calibration_schedule_infeasible"
    assert cadence.loc[0, "blocker"] == "target_samples_exceed_rolling_window_cadence_capacity"
    assert bool(cadence.loc[0, "projected_capture_feasible"]) is False
    assert result.summary["schedule_infeasible"] == 1


def test_hyperliquid_cadence_accepts_mixed_iso_timestamp_precision(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "asset_x": "BTC",
                "asset_y": "ETH",
                "leg_notional_usd": 1000.0,
                "required_slippage_samples": 2,
                "slippage_window_hours": 2.0,
                "slippage_samples_x": 2,
                "slippage_samples_y": 2,
                "slippage_model_ready": True,
            }
        ]
    ).to_csv(active / "hyperliquid_pair_cost_model.csv", index=False)
    rows = []
    for asset in ("BTC", "ETH"):
        for source_timestamp in (
            "2026-08-05T12:00:00.123000+00:00",
            "2026-08-05T12:10:00+00:00",
        ):
            rows.append(
                {
                    "pair": "BTC-USD-ETH-USD",
                    "asset": asset,
                    "notional_usd": 1000.0,
                    "source_timestamp": source_timestamp,
                    "buy_complete": True,
                    "sell_complete": True,
                    "one_way_slippage_bps": 1.0,
                }
            )
    pd.DataFrame(rows).to_csv(processed / "hyperliquid_l2_slippage_samples.csv", index=False)

    result = build_hyperliquid_evidence_cadence(
        root=tmp_path,
        target_samples=2,
        cadence_minutes=10,
        window_hours=2,
        as_of=datetime(2026, 8, 5, 12, 10, tzinfo=timezone.utc),
    )
    cadence = pd.read_csv(result.paths["hyperliquid_evidence_cadence"])

    assert cadence.loc[0, "samples_x"] == 2
    assert cadence.loc[0, "samples_y"] == 2
    assert cadence.loc[0, "status"] == "calibrated"


def test_hyperliquid_cost_model_uses_rolling_window_and_excludes_future_samples(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "asset_x": "BTC",
                "asset_y": "ETH",
                "both_legs_testnet_perp": True,
            }
        ]
    ).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)
    start = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)

    def book(coin: str, timestamp: datetime, bid: str, ask: str) -> dict[str, object]:
        return {
            "coin": coin,
            "time": int(timestamp.timestamp() * 1000),
            "levels": [[{"px": bid, "sz": "100"}], [{"px": ask, "sz": "100"}]],
        }

    for timestamp in (start, start + timedelta(minutes=30), start + timedelta(hours=3)):
        refresh_hyperliquid_execution_cost_snapshot(
            root=tmp_path,
            max_pairs=1,
            notionals=(1000.0,),
            captured_at=timestamp,
            book_payloads={
                "BTC": book("BTC", timestamp, "99", "101"),
                "ETH": book("ETH", timestamp, "49", "51"),
            },
            min_samples=2,
            window_hours=2,
        )

    result = build_hyperliquid_pair_cost_model(
        root=tmp_path,
        max_pairs=1,
        leg_notional_usd=1000.0,
        min_samples=2,
        window_hours=2,
        as_of=start + timedelta(hours=1),
    )
    model = pd.read_csv(result.paths["hyperliquid_pair_cost_model"])

    assert model.loc[0, "slippage_samples_x"] == 2
    assert model.loc[0, "slippage_samples_y"] == 2
    assert bool(model.loc[0, "slippage_model_ready"]) is True
    assert model.loc[0, "freshest_sample_at"] == (start + timedelta(minutes=30)).isoformat()


def test_hyperliquid_cadence_rolls_old_samples_out_without_permanent_expiry(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "asset_x": "BTC",
                "asset_y": "ETH",
                "both_legs_testnet_perp": True,
            }
        ]
    ).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)
    start = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)

    def payloads(timestamp: datetime) -> dict[str, object]:
        return {
            "BTC": {
                "coin": "BTC",
                "time": int(timestamp.timestamp() * 1000),
                "levels": [[{"px": "99", "sz": "100"}], [{"px": "101", "sz": "100"}]],
            },
            "ETH": {
                "coin": "ETH",
                "time": int(timestamp.timestamp() * 1000),
                "levels": [[{"px": "49", "sz": "100"}], [{"px": "51", "sz": "100"}]],
            },
        }

    refresh_hyperliquid_execution_cost_snapshot(
        root=tmp_path,
        max_pairs=1,
        notionals=(1000.0,),
        captured_at=start,
        book_payloads=payloads(start),
        min_samples=2,
        window_hours=2,
    )
    recent = start + timedelta(hours=3)
    refresh_hyperliquid_execution_cost_snapshot(
        root=tmp_path,
        max_pairs=1,
        notionals=(1000.0,),
        captured_at=recent,
        book_payloads=payloads(recent),
        min_samples=2,
        window_hours=2,
    )

    result = build_hyperliquid_evidence_cadence(
        root=tmp_path,
        target_samples=2,
        cadence_minutes=10,
        window_hours=2,
        as_of=recent + timedelta(minutes=10),
    )
    cadence = pd.read_csv(result.paths["hyperliquid_evidence_cadence"])

    assert cadence.loc[0, "samples_x"] == 1
    assert cadence.loc[0, "samples_y"] == 1
    assert cadence.loc[0, "required_samples"] == 1
    assert cadence.loc[0, "status"] == "capture_due"
    assert cadence.loc[0, "first_sample_at"] == recent.isoformat()
    assert result.summary["expired"] == 0


def test_hyperliquid_cadence_requires_matching_model_before_calibrated(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "asset_x": "BTC",
                "asset_y": "ETH",
                "both_legs_testnet_perp": True,
            }
        ]
    ).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)
    start = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)

    def book(coin: str, timestamp: datetime, bid: str, ask: str) -> dict[str, object]:
        return {
            "coin": coin,
            "time": int(timestamp.timestamp() * 1000),
            "levels": [[{"px": bid, "sz": "100"}], [{"px": ask, "sz": "100"}]],
        }

    for step in range(12):
        timestamp = start + timedelta(minutes=10 * step)
        refresh_hyperliquid_execution_cost_snapshot(
            root=tmp_path,
            max_pairs=1,
            notionals=(1000.0,),
            captured_at=timestamp,
            book_payloads={
                "BTC": book("BTC", timestamp, "99", "101"),
                "ETH": book("ETH", timestamp, "49", "51"),
            },
            min_samples=12,
            window_hours=2,
        )
    end = start + timedelta(minutes=110)

    calibrated_result = build_hyperliquid_evidence_cadence(
        root=tmp_path,
        target_samples=12,
        cadence_minutes=10,
        window_hours=2,
        as_of=end,
    )
    calibrated = pd.read_csv(calibrated_result.paths["hyperliquid_evidence_cadence"])
    assert calibrated.loc[0, "status"] == "calibrated"
    assert bool(calibrated.loc[0, "model_rebuild_required"]) is False

    build_hyperliquid_pair_cost_model(
        root=tmp_path,
        max_pairs=1,
        leg_notional_usd=1000.0,
        min_samples=12,
        window_hours=24,
        as_of=end,
    )
    stale_result = build_hyperliquid_evidence_cadence(
        root=tmp_path,
        target_samples=12,
        cadence_minutes=10,
        window_hours=2,
        as_of=end,
    )
    stale = pd.read_csv(stale_result.paths["hyperliquid_evidence_cadence"])
    assert stale.loc[0, "status"] == "model_rebuild_due"
    assert bool(stale.loc[0, "model_window_matches"]) is False
    assert bool(stale.loc[0, "model_rebuild_required"]) is True
