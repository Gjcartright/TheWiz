import json

import pandas as pd

from quant_platform.dydx_candles import (
    _attach_leg_funding_to_rows,
    archive_dydx_candles,
    backfill_provisional_pair_history_features,
    build_pair_history_from_candles,
    build_pair_history_from_windowed_candles,
    dydx_two_leg_request_rows,
    import_dydx_candle_bundle,
    load_loose_candle_payload,
    merge_dydx_candle_windows,
)


def test_funding_merge_does_not_leak_across_out_of_order_candles():
    rows = [
        {"timestamp": "2026-06-18T00:15:00Z"},
        {"timestamp": "2026-06-18T00:00:00Z"},
    ]
    funding = pd.DataFrame(
        [{"market": "AAA-USD", "timestamp": "2026-06-18T00:10:00Z", "funding_bps": 1.0}]
    )

    _attach_leg_funding_to_rows(rows, funding, "AAA-USD", "funding_x_bps")

    assert rows[0]["funding_x_bps"] == 1.0
    assert "funding_x_bps" not in rows[1]


def test_load_loose_candle_payload_accepts_pasted_response_fragment(tmp_path):
    source = tmp_path / "pasted.txt"
    source.write_text(
        """
        {
          "startedAt": "2026-06-18T00:00:00.000Z",
          "ticker": "BNB-USD",
          "resolution": "5MINS",
          "close": "600.0",
          "usdVolume": "1000"
        },
        {
          "startedAt": "2026-06-18T00:05:00.000Z",
          "ticker": "BNB-USD",
          "resolution": "5MINS",
          "close": "601.0",
          "usdVolume": "1100"
        }
        """,
        encoding="utf-8",
    )

    candles = load_loose_candle_payload(source)

    assert len(candles) == 2
    assert candles[0]["ticker"] == "BNB-USD"
    assert candles[0]["resolution"] == "5MINS"


def test_archive_dydx_candles_writes_ticker_resolution_file(tmp_path):
    source = tmp_path / "bnb.json"
    source.write_text(
        json.dumps(
            {
                "candles": [
                    {
                        "startedAt": "2026-06-18T00:00:00.000Z",
                        "ticker": "BNB-USD",
                        "resolution": "5MINS",
                        "close": "600",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output = archive_dydx_candles(source, tmp_path / "out")

    assert output.name == "BNB-USD_5MINS_candles.json"
    archived = json.loads(output.read_text(encoding="utf-8"))
    assert archived["candles"][0]["close"] == "600"


def test_build_pair_history_from_windowed_candles_merges_dedupes_and_sorts(tmp_path):
    source = tmp_path / "long" / "sol_link"
    for window, minutes in (("window_002", [0, 5, 10]), ("window_001", [10, 15, 20])):
        window_dir = source / window
        window_dir.mkdir(parents=True)
        for market, base in (("SOL-USD", 100), ("LINK-USD", 20)):
            candles = [
                {
                    "startedAt": f"2026-06-18T00:{minute:02d}:00.000Z",
                    "ticker": market,
                    "resolution": "5MINS",
                    "close": str(base + minute / 5),
                }
                for minute in minutes
            ]
            (window_dir / f"{market}_5MINS_candles.json").write_text(
                json.dumps({"candles": candles}), encoding="utf-8"
            )

    paths = build_pair_history_from_windowed_candles(
        input_dir=source,
        output_dir=tmp_path / "merged",
        pair_output_dir=tmp_path / "pairs",
        pair_id="sol_link",
        asset_x="SOL-USD",
        asset_y="LINK-USD",
        hedge_ratio=None,
        beta=None,
        zscore_window=3,
        derive_hedge_ratio=True,
    )

    merged = json.loads(paths["left_candles"].read_text(encoding="utf-8"))
    pair = json.loads(paths["pair_history"].read_text(encoding="utf-8"))
    assert [row["startedAt"] for row in merged["candles"]] == [
        "2026-06-18T00:00:00.000Z",
        "2026-06-18T00:05:00.000Z",
        "2026-06-18T00:10:00.000Z",
        "2026-06-18T00:15:00.000Z",
        "2026-06-18T00:20:00.000Z",
    ]
    assert len(pair["history"]) == 5
    assert pair["hedge_ratio_source"] == "derived_log_y_on_log_x_ols"


def test_merge_dydx_candle_windows_skips_empty_windows(tmp_path):
    source = tmp_path / "long" / "btc_eth"
    source.mkdir(parents=True)

    good_window = source / "window_001"
    bad_window = source / "window_002"
    good_window.mkdir()
    bad_window.mkdir()

    for path in (
        good_window / "BTC-USD_5MINS_candles.json",
        good_window / "ETH-USD_5MINS_candles.json",
    ):
        path.write_text(
            json.dumps(
                {
                    "candles": [
                        {
                            "startedAt": "2026-06-18T00:00:00.000Z",
                            "ticker": "BTC-USD",
                            "resolution": "5MINS",
                            "close": "60000",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    bad_window.joinpath("BTC-USD_5MINS_candles.json").write_text(
        '{"candles": []}', encoding="utf-8"
    )
    bad_window.joinpath("ETH-USD_5MINS_candles.json").write_text(
        '{"candles": []}', encoding="utf-8"
    )

    path = merge_dydx_candle_windows(
        input_dir=source,
        market="BTC-USD",
        resolution="5MINS",
        output_dir=tmp_path / "merged",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(payload["candles"]) == 1
    assert payload["candles"][0]["startedAt"] == "2026-06-18T00:00:00.000Z"


def test_dydx_two_leg_request_rows_builds_candle_funding_and_local_steps():
    rows = dydx_two_leg_request_rows(
        asset_x="BNB-USD", asset_y="STX-USD", pair_id="1", hedge_ratio=1.36
    )

    assert [row["request_name"] for row in rows] == [
        "asset_x_candles_5mins",
        "asset_x_historical_funding",
        "asset_y_candles_5mins",
        "asset_y_historical_funding",
        "build_two_leg_pair_history",
        "merge_funding_and_rerun_research",
    ]
    assert "/v4/candles/perpetualMarkets/BNB-USD" in rows[0]["url"]
    assert "resolution=5MINS" in rows[0]["url"]
    assert "/v4/historicalFunding/STX-USD" in rows[3]["url"]
    assert "build-dydx-pair-history" in rows[4]["import_command"]
    assert "--hedge-ratio 1.36" in rows[4]["import_command"]
    assert "funded-research-spine" in rows[5]["import_command"]


def test_build_pair_history_from_5min_candles_namespaces_proxies_and_adds_math_v2(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    timestamps = [f"2026-06-18T00:{minute:02d}:00.000Z" for minute in range(0, 30, 5)]
    left.write_text(
        json.dumps(
            {
                "candles": [
                    {
                        "startedAt": timestamp,
                        "ticker": "BNB-USD",
                        "resolution": "5MINS",
                        "close": str(600 + idx),
                        "usdVolume": "1000",
                    }
                    for idx, timestamp in enumerate(timestamps)
                ]
            }
        ),
        encoding="utf-8",
    )
    right.write_text(
        json.dumps(
            {
                "candles": [
                    {
                        "startedAt": timestamp,
                        "ticker": "STX-USD",
                        "resolution": "5MINS",
                        "close": str(0.2 + idx * 0.001),
                        "usdVolume": "100",
                    }
                    for idx, timestamp in enumerate(timestamps)
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "pair.json"

    path = build_pair_history_from_candles(
        left_path=left,
        right_path=right,
        output_path=output,
        pair_id="1",
        asset_x="BNB-USD",
        asset_y="STX-USD",
        hedge_ratio=1.36,
        interval="5mins",
        zscore_window=4,
        min_zscore_window=2,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["interval"] == "5mins"
    assert len(payload["history"]) == 6
    assert {
        "price_x",
        "price_y",
        "spread",
        "zscore",
        "research_proxy_ecm_x",
        "research_proxy_ecm_y",
        "research_proxy_ecm_strength",
        "math_v2_zscore_ddof0",
        "math_v2_zscore_ddof1",
    }.issubset(payload["history"][0])
    assert "ecm_strength" not in payload["history"][0]
    assert "conditional_probability_distortion" not in payload["history"][0]
    assert payload["math_version"] == "math-v2.3-venue-clock-execution"
    assert payload["math_v2_signal_use_status"].startswith("blocked")
    assert "funding_x_bps" not in payload["history"][0]
    assert "funding_y_bps" not in payload["history"][0]
    assert "Funding is not fabricated" in payload["source_note"]
    assert payload["hedge_ratio_source"] == "operator_beta_y_on_x"
    assert payload["hedge_ratio_orientation"] == "beta_y_on_x"
    assert payload["spread_contract"] == "log_y_minus_beta_y_on_x_times_log_x"
    assert payload["beta_source"] == "derived_return_covariance"
    assert payload["ecm_derivation"]["native_crypto_wizards_ecm"] is False


def test_build_pair_history_can_derive_hedge_ratio_and_beta_from_candles(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    timestamps = [f"2026-06-18T00:{minute:02d}:00.000Z" for minute in range(0, 40, 5)]
    left_prices = [100 + idx for idx, _ in enumerate(timestamps)]
    right_prices = [3.0 * value**2 for value in left_prices]
    left.write_text(
        json.dumps(
            {
                "candles": [
                    {
                        "startedAt": timestamp,
                        "ticker": "AAA-USD",
                        "resolution": "5MINS",
                        "close": str(price),
                    }
                    for timestamp, price in zip(timestamps, left_prices)
                ]
            }
        ),
        encoding="utf-8",
    )
    right.write_text(
        json.dumps(
            {
                "candles": [
                    {
                        "startedAt": timestamp,
                        "ticker": "BBB-USD",
                        "resolution": "5MINS",
                        "close": str(price),
                    }
                    for timestamp, price in zip(timestamps, right_prices)
                ]
            }
        ),
        encoding="utf-8",
    )

    path = build_pair_history_from_candles(
        left_path=left,
        right_path=right,
        output_path=tmp_path / "pair.json",
        pair_id="derived",
        asset_x="AAA-USD",
        asset_y="BBB-USD",
        hedge_ratio=None,
        beta=None,
        interval="5mins",
        zscore_window=4,
        min_zscore_window=2,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert round(payload["hedge_ratio"], 6) == 2.0
    assert payload["hedge_ratio_source"] == "derived_log_y_on_log_x_ols"
    assert payload["spread_point_in_time_status"] == "full_sample_hindsight_research_only"
    assert payload["beta_source"] == "derived_return_covariance"
    assert round(payload["history"][0]["hedge_ratio"], 6) == 2.0


def test_build_pair_history_prefers_orderbook_mid_price_for_thin_markets(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    timestamps = [f"2026-06-18T00:{minute:02d}:00.000Z" for minute in range(0, 20, 5)]
    left.write_text(
        json.dumps(
            {
                "candles": [
                    {
                        "startedAt": timestamp,
                        "ticker": "AAA-USD",
                        "resolution": "5MINS",
                        "close": "100.0",
                        "orderbookMidPriceClose": str(100.0 + idx),
                    }
                    for idx, timestamp in enumerate(timestamps)
                ]
            }
        ),
        encoding="utf-8",
    )
    right.write_text(
        json.dumps(
            {
                "candles": [
                    {
                        "startedAt": timestamp,
                        "ticker": "BBB-USD",
                        "resolution": "5MINS",
                        "close": "10.0",
                        "orderbookMidPriceClose": str(10.0 + idx),
                    }
                    for idx, timestamp in enumerate(timestamps)
                ]
            }
        ),
        encoding="utf-8",
    )

    path = build_pair_history_from_candles(
        left_path=left,
        right_path=right,
        output_path=tmp_path / "pair_mid.json",
        pair_id="mid_pref",
        asset_x="AAA-USD",
        asset_y="BBB-USD",
        hedge_ratio=1.0,
        beta=1.0,
        interval="5mins",
        zscore_window=4,
        min_zscore_window=2,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [row["price_x"] for row in payload["history"]] == [100.0, 101.0, 102.0, 103.0]
    assert [row["price_y"] for row in payload["history"]] == [10.0, 11.0, 12.0, 13.0]


def test_import_dydx_candle_bundle_writes_pair_histories(tmp_path):
    timestamps = [f"2026-06-18T00:{minute:02d}:00.000Z" for minute in range(0, 25, 5)]

    def leg(market, base):
        return {
            "market": market,
            "ok": True,
            "status": 200,
            "json": {
                "candles": [
                    {
                        "startedAt": timestamp,
                        "ticker": market,
                        "resolution": "5MINS",
                        "close": str(base + idx),
                        "usdVolume": "1000",
                    }
                    for idx, timestamp in enumerate(timestamps)
                ]
            },
        }

    bundle = {
        "resolution": "5MINS",
        "pairs": [
            {
                "pair_id": "1",
                "pair": "BNB-USD-STX-USD",
                "asset_x": "BNB-USD",
                "asset_y": "STX-USD",
                "legs": {"asset_x": leg("BNB-USD", 600), "asset_y": leg("STX-USD", 0.2)},
            },
            {
                "pair_id": "2",
                "pair": "ETH-USD-BTC-USD",
                "asset_x": "ETH-USD",
                "asset_y": "BTC-USD",
                "legs": {"asset_x": leg("ETH-USD", 3000), "asset_y": leg("BTC-USD", 100000)},
            },
        ],
    }
    source = tmp_path / "bundle.json"
    source.write_text(json.dumps(bundle), encoding="utf-8")

    paths = import_dydx_candle_bundle(
        source,
        candle_output_dir=tmp_path / "candles",
        pair_output_dir=tmp_path / "pairs",
        zscore_window=3,
    )

    assert len(paths) == 2
    assert (tmp_path / "candles" / "BNB-USD_5MINS_candles.json").exists()
    first_pair = json.loads(paths[0].read_text(encoding="utf-8"))
    assert first_pair["interval"] == "5mins"
    assert len(first_pair["history"]) == 5
    assert {"price_x", "price_y", "spread", "zscore"}.issubset(first_pair["history"][0])


def test_backfill_provisional_pair_history_features_updates_existing_files(tmp_path):
    pair_dir = tmp_path / "pairs"
    pair_dir.mkdir()
    path = pair_dir / "pair_demo_5mins_dydx_candles_derived_history.json"
    path.write_text(
        json.dumps(
            {
                "pair": "BTC-USD-ETH-USD",
                "history": [
                    {
                        "timestamp": "2026-06-18T00:00:00.000Z",
                        "price_x": 100,
                        "price_y": 50,
                        "spread": 50,
                        "zscore": 2.0,
                    }
                ],
                "source_note": "original",
            }
        ),
        encoding="utf-8",
    )

    written = backfill_provisional_pair_history_features(pair_dir)

    assert written == [path]
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["history"][0]
    assert "conditional_probability_distortion" in row
    assert "half_life" in row
    assert "ml_confidence" in row
    assert "provisional derived features" in payload["source_note"]


def test_backfill_provisional_pair_history_features_generates_row_varying_signal_scores(tmp_path):
    pair_dir = tmp_path / "pairs"
    pair_dir.mkdir()
    path = pair_dir / "pair_demo_5mins_dydx_candles_derived_history.json"
    history = []
    for idx in range(24):
        price_x = 100.0 + idx * 0.6 + (1.5 if idx % 4 == 0 else -0.8)
        price_y = 48.0 + idx * 0.25 + (0.9 if idx % 5 == 0 else -0.4)
        spread = price_x - price_y
        history.append(
            {
                "timestamp": f"2026-06-18T{idx // 12:02d}:{(idx % 12) * 5:02d}:00.000Z",
                "price_x": price_x,
                "price_y": price_y,
                "spread": spread,
                "zscore": ((idx % 7) - 3) / 1.4,
            }
        )
    path.write_text(
        json.dumps(
            {
                "pair": "BTC-USD-SOL-USD",
                "history": history,
                "source_note": "original",
            }
        ),
        encoding="utf-8",
    )

    backfill_provisional_pair_history_features(pair_dir)

    payload = json.loads(path.read_text(encoding="utf-8"))
    enriched = payload["history"]
    ml_values = {round(float(row["ml_confidence"]), 6) for row in enriched}
    profile_values = {round(float(row["profile_match"]), 6) for row in enriched}
    ou_values = {round(float(row["ou_optimal"]), 6) for row in enriched}
    assert len(ml_values) > 1
    assert len(profile_values) > 1
    assert len(ou_values) > 1


def test_pair_history_funding_merge_keeps_pre_observation_rows_unknown(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    timestamps = [f"2026-06-18T00:{minute:02d}:00Z" for minute in range(0, 30, 5)]
    for path, ticker, base in (
        (left, "AAA-USD", 100.0),
        (right, "BBB-USD", 50.0),
    ):
        path.write_text(
            json.dumps(
                {
                    "candles": [
                        {
                            "startedAt": timestamp,
                            "ticker": ticker,
                            "resolution": "5MINS",
                            "close": base + index,
                        }
                        for index, timestamp in enumerate(timestamps)
                    ]
                }
            ),
            encoding="utf-8",
        )
    funding = pd.DataFrame(
        [
            {
                "market": market,
                "timestamp": "2026-06-18T00:10:00Z",
                "funding_bps": value,
            }
            for market, value in (("AAA-USD", 1.0), ("BBB-USD", 2.0))
        ]
    )

    output = build_pair_history_from_candles(
        left_path=left,
        right_path=right,
        output_path=tmp_path / "pair.json",
        pair_id="aaa_bbb",
        asset_x="AAA-USD",
        asset_y="BBB-USD",
        hedge_ratio=1.0,
        beta=1.0,
        interval="5mins",
        zscore_window=3,
        min_zscore_window=2,
        funding_rows=funding,
    )

    history = json.loads(output.read_text(encoding="utf-8"))["history"]
    assert "funding_x_bps" not in history[0]
    assert "funding_y_bps" not in history[1]
    assert history[2]["funding_x_bps"] == 1.0
    assert history[2]["funding_y_bps"] == 2.0


def test_pair_history_does_not_assign_undated_funding_to_past_candles(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    timestamps = [f"2026-06-18T00:{minute:02d}:00Z" for minute in range(0, 30, 5)]
    for path, ticker, base in ((left, "AAA-USD", 100.0), (right, "BBB-USD", 50.0)):
        path.write_text(
            json.dumps(
                {
                    "candles": [
                        {
                            "startedAt": timestamp,
                            "ticker": ticker,
                            "resolution": "5MINS",
                            "close": base + index,
                        }
                        for index, timestamp in enumerate(timestamps)
                    ]
                }
            ),
            encoding="utf-8",
        )
    funding = pd.DataFrame(
        [
            {"market": "AAA-USD", "timestamp": None, "funding_bps": 1.0},
            {"market": "BBB-USD", "timestamp": None, "funding_bps": 2.0},
        ]
    )
    output = build_pair_history_from_candles(
        left_path=left,
        right_path=right,
        output_path=tmp_path / "pair.json",
        pair_id="aaa_bbb",
        asset_x="AAA-USD",
        asset_y="BBB-USD",
        hedge_ratio=1.0,
        beta=1.0,
        interval="5mins",
        zscore_window=3,
        min_zscore_window=2,
        funding_rows=funding,
    )
    history = json.loads(output.read_text(encoding="utf-8"))["history"]
    assert all("funding_x_bps" not in row and "funding_y_bps" not in row for row in history)
