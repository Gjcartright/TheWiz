import json

import pandas as pd

from quant_platform.yahoo_crypto import (
    backfill_yahoo_crypto_funding,
    build_yahoo_crypto_lane_report,
    build_yahoo_crypto_pair_history,
    normalize_yahoo_crypto_chart,
    refresh_yahoo_research_candidates,
    refresh_yahoo_crypto_pair_history,
)


def test_normalize_yahoo_crypto_chart_outputs_canonical_candles():
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1761091200, 1761177600],
                    "indicators": {
                        "quote": [
                            {
                                "open": [10, 11],
                                "high": [12, 13],
                                "low": [9, 10],
                                "close": [11, 12],
                                "volume": [100, 200],
                            }
                        ]
                    },
                }
            ]
        }
    }

    candles = normalize_yahoo_crypto_chart(payload, symbol="ETH-USD", interval="1d")

    assert candles[0]["ticker"] == "ETH-USD"
    assert candles[0]["resolution"] == "1d"
    assert candles[0]["close"] == 11.0
    assert candles[0]["usdVolume"] == 1100.0
    assert candles[0]["source"] == "yahoo_crypto"


def test_build_yahoo_crypto_pair_history_marks_exchange_and_reuses_pair_engine(tmp_path):
    candle_dir = tmp_path / "candles"
    candle_dir.mkdir()
    timestamps = [1761091200, 1761177600, 1761264000, 1761350400, 1761436800]
    left = normalize_yahoo_crypto_chart(
        {
            "chart": {
                "result": [
                    {
                        "timestamp": timestamps,
                        "indicators": {
                            "quote": [
                                {
                                    "open": [10, 11, 12, 13, 14],
                                    "high": [11, 12, 13, 14, 15],
                                    "low": [9, 10, 11, 12, 13],
                                    "close": [10, 11, 12, 13, 14],
                                    "volume": [100, 101, 102, 103, 104],
                                }
                            ]
                        },
                    }
                ]
            }
        },
        symbol="ETH-USD",
        interval="1d",
    )
    right = normalize_yahoo_crypto_chart(
        {
            "chart": {
                "result": [
                    {
                        "timestamp": timestamps,
                        "indicators": {
                            "quote": [
                                {
                                    "open": [2, 3, 4, 5, 6],
                                    "high": [3, 4, 5, 6, 7],
                                    "low": [1, 2, 3, 4, 5],
                                    "close": [2, 3, 4, 5, 6],
                                    "volume": [200, 201, 202, 203, 204],
                                }
                            ]
                        },
                    }
                ]
            }
        },
        symbol="SOL-USD",
        interval="1d",
    )
    (candle_dir / "ETH-USD_1d_candles.json").write_text(json.dumps({"candles": left}), encoding="utf-8")
    (candle_dir / "SOL-USD_1d_candles.json").write_text(json.dumps({"candles": right}), encoding="utf-8")

    path = build_yahoo_crypto_pair_history(
        asset_x="ETH-USD",
        asset_y="SOL-USD",
        interval="1d",
        candle_dir=candle_dir,
        output_dir=tmp_path,
        zscore_window=3,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["exchange"] == "yahoo_crypto"
    assert payload["asset_x"] == "ETH-USD"
    assert payload["asset_y"] == "SOL-USD"
    assert len(payload["history"]) == 5


def test_refresh_yahoo_crypto_pair_history_fetches_both_legs_before_build(tmp_path, monkeypatch):
    calls: list[tuple[str, str, str]] = []

    def fake_fetch(*, symbol, interval, lookback_range, output_dir=None, **_kwargs):
        from quant_platform.yahoo_crypto import _to_yahoo_symbol

        calls.append((symbol, interval, lookback_range))
        candle_dir = output_dir or (tmp_path / "candles")
        candle_dir.mkdir(parents=True, exist_ok=True)
        normalized = _to_yahoo_symbol(symbol)
        payload = {
            "candles": [
                {"startedAt": "2026-01-01T00:00:00Z", "ticker": normalized, "resolution": interval, "open": 1, "high": 1, "low": 1, "close": 1, "baseVolume": 1, "usdVolume": 1, "source": "yahoo_crypto"},
                {"startedAt": "2026-01-02T00:00:00Z", "ticker": normalized, "resolution": interval, "open": 2, "high": 2, "low": 2, "close": 2, "baseVolume": 2, "usdVolume": 4, "source": "yahoo_crypto"},
                {"startedAt": "2026-01-03T00:00:00Z", "ticker": normalized, "resolution": interval, "open": 3, "high": 3, "low": 3, "close": 3, "baseVolume": 3, "usdVolume": 9, "source": "yahoo_crypto"},
            ]
        }
        path = candle_dir / f"{normalized}_{interval}_candles.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    monkeypatch.setattr("quant_platform.yahoo_crypto.fetch_yahoo_crypto_candles", fake_fetch)

    path = refresh_yahoo_crypto_pair_history(
        asset_x="ETHUSDT",
        asset_y="SOLUSDT",
        interval="1d",
        lookback_range="5y",
        candle_dir=tmp_path / "candles",
        output_dir=tmp_path,
        zscore_window=2,
    )

    assert calls == [("ETHUSDT", "1d", "5y"), ("SOLUSDT", "1d", "5y")]
    assert path.exists()


def test_build_yahoo_crypto_lane_report_detects_research_only_symbols(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    (tmp_path / "data" / "raw" / "yahoo_crypto_candles").mkdir(parents=True)
    (active / "multi_venue_history_readiness_2026-06-25.csv").write_text(
        "wizard_exchange,asset_x,asset_y\nbinance,ETHUSDT,TRUMPUSDT\ndydx,ETH-USD,SOL-USD\ncoinbase,BTC-USD,ETH-USD\n",
        encoding="utf-8",
    )
    (tmp_path / "data" / "raw" / "yahoo_crypto_candles" / "ETH-USD_1d_candles.json").write_text(
        json.dumps({"candles": [{}] * 140}),
        encoding="utf-8",
    )

    result = build_yahoo_crypto_lane_report(root=tmp_path)

    payload = result.paths["yahoo_crypto_history_readiness"]
    rows = payload.read_text(encoding="utf-8")
    assert "ETH-USD" in rows
    assert "TRUMP-USD" in rows
    assert "dydx" not in rows


def test_refresh_yahoo_research_candidates_runs_ready_research_pairs(tmp_path, monkeypatch):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    (active / "multi_venue_history_readiness_2026-06-25.csv").write_text(
        "\n".join(
            [
                "rank,pair,wizard_exchange,asset_x,asset_y,readiness_status",
                "1,ETHUSDT-TRUMPUSDT,binance,ETHUSDT,TRUMPUSDT,ready_to_fetch",
                "2,BTC-USD-ETH-USD,dydx,BTC-USD,ETH-USD,ready_for_replay",
                "3,SOLUSDT-DOGEUSDT,bybit,SOLUSDT,DOGEUSDT,ready_to_fetch",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, str]] = []

    def fake_refresh(**kwargs):
        calls.append((kwargs["asset_x"], kwargs["asset_y"]))
        out = tmp_path / f'{kwargs["asset_x"]}_{kwargs["asset_y"]}.json'
        out.write_text("{}", encoding="utf-8")
        return out

    monkeypatch.setattr("quant_platform.yahoo_crypto.refresh_yahoo_crypto_pair_history", fake_refresh)

    result = refresh_yahoo_research_candidates(root=tmp_path, max_pairs=5, interval="1d", lookback_range="5y", zscore_window=120)

    assert calls == [("ETHUSDT", "TRUMPUSDT"), ("SOLUSDT", "DOGEUSDT")]
    text = result.paths["yahoo_crypto_research_refresh"].read_text(encoding="utf-8")
    assert "ETHUSDT-TRUMPUSDT" in text
    assert "SOLUSDT-DOGEUSDT" in text
    assert "BTC-USD-ETH-USD" not in text


def test_refresh_yahoo_research_candidates_falls_back_to_lane_derived_pairs(tmp_path, monkeypatch):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    (active / "multi_venue_history_readiness_2026-06-25.csv").write_text(
        "rank,pair,wizard_exchange,asset_x,asset_y,readiness_status\n",
        encoding="utf-8",
    )
    (active / "venue_lane_classification.csv").write_text(
        "\n".join(
            [
                "asset,best_lane,dydx_lane,hyperliquid_lane,blockers,next_action",
                "BTC,dydx_execution_candidate,dydx_context,hyperliquid_research_candidate,,",
                "ETH,dydx_execution_candidate,dydx_context,hyperliquid_research_candidate,,",
                "LINK,hyperliquid_research_candidate,dydx_research_only,hyperliquid_research_candidate,,",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, str]] = []

    def fake_refresh(**kwargs):
        calls.append((kwargs["asset_x"], kwargs["asset_y"]))
        out = tmp_path / f'{kwargs["asset_x"]}_{kwargs["asset_y"]}.json'
        out.write_text("{}", encoding="utf-8")
        return out

    monkeypatch.setattr("quant_platform.yahoo_crypto.refresh_yahoo_crypto_pair_history", fake_refresh)

    result = refresh_yahoo_research_candidates(root=tmp_path, max_pairs=2, interval="1d", lookback_range="5y", zscore_window=120)

    assert calls == [("BTC-USD", "LINK-USD"), ("ETH-USD", "LINK-USD")]
    text = result.paths["yahoo_crypto_research_refresh"].read_text(encoding="utf-8")
    assert "BTC-USD/LINK-USD" in text
    assert "ETH-USD/LINK-USD" in text


def test_backfill_yahoo_crypto_funding_injects_leg_funding_columns(tmp_path):
    pair_dir = tmp_path / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True)
    reports = tmp_path / "reports" / "active"
    reports.mkdir(parents=True)
    payload = {
        "pair": "BTC-USD-ETH-USD",
        "history": [
            {"timestamp": "2026-06-23T00:00:00Z", "price_x": 1.0, "price_y": 2.0, "hedge_ratio": 1.0, "beta": 1.0},
            {"timestamp": "2026-06-23T01:00:00Z", "price_x": 1.5, "price_y": 2.5, "hedge_ratio": 1.0, "beta": 1.0},
        ],
    }
    target = pair_dir / "pair_btc-usd_eth-usd_yahoo_crypto_1d_yahoo_crypto_1d_derived_history.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    funding_path = tmp_path / "funding.csv"
    funding_path.write_text(
        "\n".join(
            [
                "market,funding_bps,timestamp",
                "BTC-USD,1.5,2026-06-23T00:00:00Z",
                "ETH-USD,2.5,2026-06-23T00:00:00Z",
            ]
        ),
        encoding="utf-8",
    )

    result = backfill_yahoo_crypto_funding(funding_path=funding_path, root=tmp_path, pair_dir=pair_dir)

    updated = json.loads(target.read_text(encoding="utf-8"))
    assert updated["history"][0]["funding_x_bps"] == 1.5
    assert updated["history"][0]["funding_y_bps"] == 2.5
    assert result.summary["backfilled"] == 1


def test_yahoo_pair_quality_counts_research_execution_inputs(tmp_path):
    from quant_platform.pair_detail_ingestion import pair_detail_quality_report

    pair_dir = tmp_path / "pairs"
    pair_dir.mkdir()
    payload = {
        "pair": "BTC-USD-HYPE-USD",
        "asset_x": "BTC-USD",
        "asset_y": "HYPE-USD",
        "source_note": "Derived from Yahoo Finance public crypto history. This is research evidence only.",
        "history": [
            {
                "timestamp": f"2026-06-23T{idx % 24:02d}:00:00Z",
                "price_x": 1.0 + idx,
                "price_y": 2.0 + idx * 0.5,
                "spread": 0.1 + idx * 0.01,
                "zscore": 0.2 + idx * 0.001,
                "hedge_ratio": 1.0,
                "beta": 1.0,
                "funding_x_bps": 1.0,
                "funding_y_bps": 2.0,
                "slippage_bps": 5.0,
                "bid_ask_spread_bps": 5.0,
                "volume_x_usd": 100.0 + idx,
                "volume_y_usd": 100.0 + idx,
            }
            for idx in range(100)
        ],
    }
    (pair_dir / "pair_btc-usd_hype-usd_yahoo_crypto_1d_yahoo_crypto_1d_derived_history.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    row = pd.DataFrame(pair_detail_quality_report(pair_dir)).iloc[0]
    assert bool(row["research_execution_usable"]) is True
