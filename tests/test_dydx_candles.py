import json

from quant_platform.dydx_candles import (
    backfill_provisional_pair_history_features,
    archive_dydx_candles,
    build_pair_history_from_windowed_candles,
    build_pair_history_from_candles,
    dydx_two_leg_request_rows,
    import_dydx_candle_bundle,
    load_loose_candle_payload,
)


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
        json.dumps({"candles": [{"startedAt": "2026-06-18T00:00:00.000Z", "ticker": "BNB-USD", "resolution": "5MINS", "close": "600"}]}),
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
            (window_dir / f"{market}_5MINS_candles.json").write_text(json.dumps({"candles": candles}), encoding="utf-8")

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
    assert pair["hedge_ratio_source"] == "derived_price_ols"


def test_dydx_two_leg_request_rows_builds_candle_funding_and_local_steps():
    rows = dydx_two_leg_request_rows(asset_x="BNB-USD", asset_y="STX-USD", pair_id="1", hedge_ratio=1.36)

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


def test_build_pair_history_from_5min_candles_adds_spread_zscore_and_provisional_ecm(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    timestamps = [f"2026-06-18T00:{minute:02d}:00.000Z" for minute in range(0, 30, 5)]
    left.write_text(
        json.dumps(
            {
                "candles": [
                    {"startedAt": timestamp, "ticker": "BNB-USD", "resolution": "5MINS", "close": str(600 + idx), "usdVolume": "1000"}
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
                    {"startedAt": timestamp, "ticker": "STX-USD", "resolution": "5MINS", "close": str(0.2 + idx * 0.001), "usdVolume": "100"}
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
    assert {"price_x", "price_y", "spread", "zscore", "ecm_x", "ecm_y", "ecm_strength"}.issubset(payload["history"][0])
    assert "funding_x_bps" not in payload["history"][0]
    assert "funding_y_bps" not in payload["history"][0]
    assert "Funding is not fabricated" in payload["source_note"]
    assert payload["hedge_ratio_source"] == "operator"
    assert payload["beta_source"] == "derived_return_covariance"
    assert payload["ecm_derivation"]["native_crypto_wizards_ecm"] is False


def test_build_pair_history_can_derive_hedge_ratio_and_beta_from_candles(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    timestamps = [f"2026-06-18T00:{minute:02d}:00.000Z" for minute in range(0, 40, 5)]
    right_prices = [100 + idx for idx, _ in enumerate(timestamps)]
    left_prices = [10 + 2 * value for value in right_prices]
    left.write_text(
        json.dumps(
            {
                "candles": [
                    {"startedAt": timestamp, "ticker": "AAA-USD", "resolution": "5MINS", "close": str(price)}
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
                    {"startedAt": timestamp, "ticker": "BBB-USD", "resolution": "5MINS", "close": str(price)}
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
    assert payload["hedge_ratio_source"] == "derived_price_ols"
    assert payload["beta_source"] == "derived_return_covariance"
    assert round(payload["history"][0]["hedge_ratio"], 6) == 2.0


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
