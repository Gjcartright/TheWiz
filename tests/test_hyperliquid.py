import json

from quant_platform.hyperliquid import (
    build_hyperliquid_lane_report,
    build_hyperliquid_pair_history,
    normalize_hyperliquid_candles,
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
    assert {"price_x", "price_y", "spread", "zscore", "ecm_x", "ecm_y", "ecm_strength"}.issubset(payload["history"][0])


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
