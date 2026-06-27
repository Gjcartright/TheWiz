import pytest

import pandas as pd

from quant_platform.experiments import PairDataset
from quant_platform.funding import (
    enrich_pair_dataset_with_funding,
    funding_coverage_for_pairs,
    funding_market_requirements,
    funding_rows_from_dydx_payload,
    normalize_funding_rows,
)


def test_normalize_funding_rows_converts_decimal_rate_to_bps():
    rows = [{"ticker": "ETH-USD", "effectiveAt": "2026-01-01T00:00:00Z", "rate": "0.0001"}]

    normalized = normalize_funding_rows(rows)

    assert list(normalized["market"]) == ["ETH-USD"]
    assert normalized["funding_bps"].iloc[0] == 1.0


def test_funding_rows_from_dydx_payload_adds_default_market_to_historical_rows():
    payload = {
        "historicalFunding": [
            {"effectiveAt": "2026-01-01T00:00:00Z", "rate": "0.0001"},
            {"effectiveAt": "2026-01-01T01:00:00Z", "rate": "0.0002"},
        ]
    }

    rows = funding_rows_from_dydx_payload(payload, market="ETH-USD")
    normalized = normalize_funding_rows(rows)

    assert list(normalized["market"]) == ["ETH-USD", "ETH-USD"]
    assert list(normalized["funding_bps"]) == [1.0, 2.0]


def test_funding_rows_from_dydx_payload_keeps_row_market_when_present():
    payload = {
        "historicalFunding": [
            {"ticker": "BTC-USD", "effectiveAt": "2026-01-01T00:00:00Z", "rate": "0.0003"},
        ]
    }

    rows = funding_rows_from_dydx_payload(payload, market="ETH-USD")

    assert rows[0]["ticker"] == "BTC-USD"


def test_funding_rows_from_dydx_payload_reads_nested_indexer_payload():
    payload = {
        "market": "ETH-USD",
        "payload": {
            "historicalFunding": [
                {"effectiveAt": "2026-01-01T00:00:00Z", "rate": "0.0001"},
            ]
        },
    }

    rows = funding_rows_from_dydx_payload(payload)
    normalized = normalize_funding_rows(rows)

    assert list(normalized["market"]) == ["ETH-USD"]
    assert list(normalized["funding_bps"]) == [1.0]


def test_funding_rows_from_dydx_payload_reads_alternative_funding_fields():
    payload = {
        "result": [
            {"ticker": "SOL-USD", "nextFundingRate": "0.0002", "scrapedAt": "2026-01-01T00:00:00Z"},
        ]
    }

    rows = funding_rows_from_dydx_payload(payload)
    normalized = normalize_funding_rows(rows)

    assert list(normalized["market"]) == ["SOL-USD"]
    assert list(normalized["funding_bps"]) == [2.0]


def test_funding_rows_from_dydx_payload_keeps_zero_apify_next_funding():
    payload = [
        {"ticker": "LINK-USD", "next_funding": 0, "scrapedAt": "2026-01-01T00:00:00Z"},
    ]

    rows = funding_rows_from_dydx_payload(payload)
    normalized = normalize_funding_rows(rows)

    assert list(normalized["market"]) == ["LINK-USD"]
    assert list(normalized["funding_bps"]) == [0.0]


def test_funding_rows_from_dydx_payload_accepts_epoch_timestamps():
    rows = [
        {"market": "LINK-USD", "funding_rate": 0.0003, "timeMs": 1_640_995_200_000},
    ]

    normalized = normalize_funding_rows(rows)

    assert list(normalized["market"]) == ["LINK-USD"]
    assert normalized["funding_bps"].iloc[0] == pytest.approx(3.0)


def test_normalize_funding_rows_accepts_mixed_timestamp_formats():
    rows = [
        {"market": "SOL-USD", "funding_rate": 0.0002, "timestamp": "2026-06-19 12:00:00.177000+00:00"},
        {"market": "SOL-USD", "funding_rate": 0.0003, "timestamp": "2022-01-01T00:00:00Z"},
    ]

    normalized = normalize_funding_rows(rows)

    assert len(normalized) == 2
    assert normalized["timestamp"].notna().all()


def test_funding_coverage_for_pairs_reports_missing_leg_funding():
    funding = pd.DataFrame([{"market": "ETH-USD", "funding_bps": 2.0}])

    coverage = funding_coverage_for_pairs(["ETH-BTC"], funding)
    row = coverage.iloc[0]

    assert row["market_x"] == "ETH-USD"
    assert row["market_y"] == "BTC-USD"
    assert bool(row["funding_x_available"]) is True
    assert bool(row["funding_y_available"]) is False
    assert bool(row["ready"]) is False
    assert row["missing"] == "funding_y"
    assert row["missing_markets"] == "BTC-USD"
    assert row["required_markets"] == "ETH-USD;BTC-USD"


def test_funding_market_requirements_lists_dydx_leg_markets():
    requirements = funding_market_requirements(["ETH-BTC", "BNB-USD-STX-USD"])
    rows = requirements.set_index("pair")

    assert bool(rows.loc["ETH-BTC", "valid"]) is True
    assert rows.loc["ETH-BTC", "required_markets"] == "ETH-USD;BTC-USD"
    assert bool(rows.loc["BNB-USD-STX-USD", "valid"]) is True
    assert rows.loc["BNB-USD-STX-USD", "required_markets"] == "BNB-USD;STX-USD"


def test_funding_market_requirements_reports_invalid_pairs():
    requirements = funding_market_requirements(["ETH-BTC-SOL"])
    row = requirements.iloc[0]

    assert bool(row["valid"]) is False
    assert row["required_markets"] == ""
    assert "pair must be" in row["error"]


def test_funding_coverage_for_pairs_accepts_usd_market_pair_names():
    funding = pd.DataFrame(
        [
            {"market": "BNB-USD", "timestamp": "2026-01-01T00:00:00Z", "funding_bps": 2.0},
            {"market": "STX-USD", "timestamp": "2026-01-01T00:00:00Z", "funding_bps": 3.0},
        ]
    )

    coverage = funding_coverage_for_pairs(["BNB-USD-STX-USD"], funding)
    row = coverage.iloc[0]

    assert bool(row["ready"]) is True
    assert row["funding_x_rows"] == 1
    assert row["funding_y_rows"] == 1
    assert row["funding_x_timestamped_rows"] == 1
    assert row["funding_y_timestamped_rows"] == 1
    assert row["missing_markets"] == ""
    assert row["required_markets"] == "BNB-USD;STX-USD"


def test_enrich_pair_dataset_with_static_latest_funding():
    dataset = PairDataset(
        pair="ETH-BTC",
        frame=pd.DataFrame({"spread": [0.1, 0.2], "price_x": [100.0, 101.0], "price_y": [50.0, 51.0]}),
    )
    funding = pd.DataFrame(
        [
            {"market": "ETH-USD", "funding_bps": 2.0},
            {"market": "BTC-USD", "funding_bps": 3.0},
        ]
    )

    enriched = enrich_pair_dataset_with_funding(dataset, funding)

    assert list(enriched.frame["funding_x_bps"]) == [2.0, 2.0]
    assert list(enriched.frame["funding_y_bps"]) == [3.0, 3.0]


def test_enrich_pair_dataset_with_timestamped_asof_funding():
    dataset = PairDataset(
        pair="ETH-BTC",
        frame=pd.DataFrame(
            {
                "timestamp": ["2026-01-01T01:00:00Z", "2026-01-01T03:00:00Z"],
                "spread": [0.1, 0.2],
            }
        ),
    )
    funding = pd.DataFrame(
        [
            {"market": "ETH-USD", "timestamp": "2026-01-01T00:00:00Z", "funding_bps": 1.0},
            {"market": "ETH-USD", "timestamp": "2026-01-01T02:00:00Z", "funding_bps": 2.0},
            {"market": "BTC-USD", "timestamp": "2026-01-01T00:00:00Z", "funding_bps": 3.0},
            {"market": "BTC-USD", "timestamp": "2026-01-01T02:00:00Z", "funding_bps": 4.0},
        ]
    )

    enriched = enrich_pair_dataset_with_funding(dataset, funding)

    assert list(enriched.frame["funding_x_bps"]) == [1.0, 2.0]
    assert list(enriched.frame["funding_y_bps"]) == [3.0, 4.0]


def test_enrich_pair_dataset_with_timestamp_precision_mismatch():
    dataset = PairDataset(
        pair="ETH-BTC",
        frame=pd.DataFrame(
            {
                "timestamp": ["2026-01-01T01:00:00.123456+00:00", "2026-01-01T03:00:00.123456+00:00"],
                "spread": [0.1, 0.2],
            }
        ),
    )
    funding = pd.DataFrame(
        {
            "market": ["ETH-USD", "ETH-USD", "BTC-USD", "BTC-USD"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00.123456+00:00",
                    "2026-01-01T02:00:00.123456+00:00",
                    "2026-01-01T00:00:00.123456+00:00",
                    "2026-01-01T02:00:00.123456+00:00",
                ],
                utc=True,
            ).astype("datetime64[us, UTC]"),
            "funding_bps": [1.0, 2.0, 3.0, 4.0],
        }
    )

    enriched = enrich_pair_dataset_with_funding(dataset, funding)

    assert list(enriched.frame["funding_x_bps"]) == [1.0, 2.0]
    assert list(enriched.frame["funding_y_bps"]) == [3.0, 4.0]


def test_enrich_pair_dataset_preserves_existing_funding_columns():
    dataset = PairDataset(
        pair="ETH-BTC",
        frame=pd.DataFrame({"funding_x_bps": [9.0], "funding_y_bps": [8.0]}),
    )
    funding = pd.DataFrame(
        [
            {"market": "ETH-USD", "funding_bps": 2.0},
            {"market": "BTC-USD", "funding_bps": 3.0},
        ]
    )

    enriched = enrich_pair_dataset_with_funding(dataset, funding)

    assert list(enriched.frame["funding_x_bps"]) == [9.0]
    assert list(enriched.frame["funding_y_bps"]) == [8.0]
