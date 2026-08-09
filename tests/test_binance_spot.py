import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from quant_platform.binance_spot import build_binance_spot_lane_report, build_binance_spot_pair_history, normalize_binance_spot_klines
from quant_platform.pair_detail_ingestion import pair_detail_payload_quality


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
    assert "execution assumptions are placeholders" in payload["source_note"]

    quality = pair_detail_payload_quality(payload, path)

    assert quality["research_execution_usable"] is False
    assert quality["execution_usable"] is False
    assert "placeholder_execution_assumptions" in quality["quality_blockers"]


def test_binance_lane_report_uses_current_wizard_shortlist_and_keeps_pair_research_only(tmp_path):
    active = tmp_path / "reports" / "active"
    candles = tmp_path / "data" / "raw" / "binance_spot_candles"
    active.mkdir(parents=True)
    candles.mkdir(parents=True)
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    history = [
        {"startedAt": (end - timedelta(days=364 - index)).isoformat().replace("+00:00", "Z")}
        for index in range(365)
    ]
    for symbol in ["ETHUSDT", "FIDAUSDT"]:
        (candles / f"{symbol}_1d_candles.json").write_text(json.dumps({"candles": history}), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT/FIDAUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "FIDAUSDT",
                "exchange": "binance",
                "exact_mode": "Copula",
                "period": 365,
                "source_timestamp": end.isoformat(),
                "source_fresh": True,
                "backtest_settings_complete": False,
                "backtest_settings_missing": "copula_thresholds;wizard_cost_assumptions",
                "evidence_path": "reports/active/wizard_discovery_shortlist.csv",
            }
        ]
    ).to_csv(active / "wizard_discovery_shortlist.csv", index=False)

    result = build_binance_spot_lane_report(root=tmp_path)
    symbols = pd.read_csv(result.paths["binance_spot_history_readiness"])
    pairs = pd.read_csv(result.paths["binance_spot_pair_readiness"])

    assert set(symbols["symbol"]) == {"ETHUSDT", "FIDAUSDT"}
    assert set(symbols["status"]) == {"history_ready"}
    row = pairs.iloc[0]
    assert row["history_status"] == "history_ready_needs_settings_and_cost_model"
    assert row["wizard_settings_status"] == "missing_pair_detail_settings"
    assert row["research_execution_status"] == "research_only_execution_blocked"
