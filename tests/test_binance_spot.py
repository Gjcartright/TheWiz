import json

from quant_platform.binance_spot import build_binance_spot_pair_history, normalize_binance_spot_klines


def test_normalize_binance_spot_klines_outputs_canonical_candles():
    payload = [
        [1761091200000, "10", "12", "9", "11", "100", 1761177599999, "1100"],
        [1761177600000, "11", "13", "10", "12", "200", 1761263999999, "2400"],
    ]

    candles = normalize_binance_spot_klines(payload, symbol="ETHUSDT", interval="1d")

    assert candles[0]["ticker"] == "ETHUSDT"
    assert candles[0]["resolution"] == "1d"
    assert candles[0]["close"] == 11.0
    assert candles[0]["usdVolume"] == 1100.0
    assert candles[0]["source"] == "binance_spot"


def test_build_binance_spot_pair_history_marks_exchange_and_reuses_pair_engine(tmp_path):
    candle_dir = tmp_path / "candles"
    candle_dir.mkdir()
    timestamps = [1761091200000, 1761177600000, 1761264000000, 1761350400000, 1761436800000]
    left = normalize_binance_spot_klines(
        [[ts, 10 + idx, 11 + idx, 9 + idx, 10 + idx, 100, ts + 1, 1000] for idx, ts in enumerate(timestamps)],
        symbol="ETHUSDT",
        interval="1d",
    )
    right = normalize_binance_spot_klines(
        [[ts, 2 + idx, 3 + idx, 1 + idx, 2 + idx, 200, ts + 1, 400] for idx, ts in enumerate(timestamps)],
        symbol="YFIUSDT",
        interval="1d",
    )
    (candle_dir / "ETHUSDT_1d_candles.json").write_text(json.dumps({"candles": left}), encoding="utf-8")
    (candle_dir / "YFIUSDT_1d_candles.json").write_text(json.dumps({"candles": right}), encoding="utf-8")

    path = build_binance_spot_pair_history(
        asset_x="ETHUSDT",
        asset_y="YFIUSDT",
        interval="1d",
        candle_dir=candle_dir,
        output_dir=tmp_path,
        zscore_window=3,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["exchange"] == "binance_spot"
    assert payload["asset_x"] == "ETHUSDT"
    assert payload["asset_y"] == "YFIUSDT"
    assert len(payload["history"]) == 5
    assert {"price_x", "price_y", "spread", "zscore", "ecm_x", "ecm_y", "ecm_strength"}.issubset(payload["history"][0])
