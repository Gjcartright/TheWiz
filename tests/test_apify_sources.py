from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.apify_sources import _extract_usage_details, parse_apify_sources_from_mcp_url, refresh_apify_sources
from quant_platform.active_pipeline import _planned_source_rows


def test_parse_apify_sources_from_mcp_url_handles_actors_docs_and_slugged_actors():
    url = (
        "https://mcp.apify.com/?tools=actors,docs,"
        "parseforge/dydx-v4-perpetual-markets-scraper,"
        "parseforge/dydx-markets-scraper,"
        "muhammetakkurtt/coinmarketcap-scraper"
    )
    sources = parse_apify_sources_from_mcp_url(url)
    assert "apify/actors" in sources
    assert "apify/docs" in sources
    assert "parseforge/dydx-v4-perpetual-markets-scraper" in sources
    assert "muhammetakkurtt/coinmarketcap-scraper" in sources
    assert len(sources) == len(set(sources))


def test_refresh_apify_sources_without_fetch_generates_coverage(tmp_path: Path):
    mcp_url = "https://mcp.apify.com/?tools=actors,docs,parseforge/dydx-markets-scraper"
    result = refresh_apify_sources(root=tmp_path, mcp_url=mcp_url, do_fetch=False)

    assert result.coverage_path.exists()
    assert result.manifest_path.exists()
    assert result.source_count == 3
    assert result.sampled_count == 0
    frame = pd.read_csv(result.coverage_path)
    assert set(frame["source_id"]) == {"apify/actors", "apify/docs", "parseforge/dydx-markets-scraper"}


def test_planned_source_rows_includes_all_sources_and_marks_context_only_not_promotable():
    source_coverage = pd.DataFrame(
        [
            {
                "source_id": "fraktalapi/funding-pulse",
                "category": "cross_exchange_perp_feed",
                "status": "checked",
                "sample_status": "not_sampled",
                "evidence": "reports/active/apify_mcp_source_coverage_2026-06-25.csv",
                "limitations": "",
            },
            {
                "source_id": "parseforge/dydx-v4-perpetual-markets-scraper",
                "category": "perp_market_feed",
                "status": "checked",
                "sample_status": "sampled",
                "limitations": "requires dYdX pair feed",
            },
            {
                "source_id": "parseforge/gmx-arbitrum-stats-scraper",
                "category": "defi_perp_feed",
                "status": "checked",
                "sample_status": "not_sampled",
                "limitations": "requires defi context",
            },
            {
                "source_id": "bybit-screener",
                "category": "supplemental",
                "status": "checked",
                "sample_status": "needs_api_key",
                "limitations": "",
            },
        ]
    )

    rows = _planned_source_rows(source_coverage)
    assert len(rows) >= 3
    byrow = {row["venue"]: row for row in rows}
    assert byrow["cross_exchange"]["source_system"] == "funding_pulse"
    assert byrow["dydx"]["execution_authority"] is True
    assert byrow["gmx"]["promotion_allowed"] is False
    assert byrow["bybit"]["promotion_allowed"] is False


def test_extract_usage_details_reads_common_apify_fields():
    payload = {
        "id": "run_123",
        "startedAt": "2026-06-27T00:00:00.000Z",
        "finishedAt": "2026-06-27T00:00:05.123Z",
        "stats": {"runTimeMillis": 5120},
        "meta": {"datasetId": "ds_1"},
        "billing": {"cost": 1.42, "currency": "USD", "credits": 142},
        "status": "SUCCEEDED",
    }

    usage = _extract_usage_details(payload, "run_123")

    assert usage["run_id"] == "run_123"
    assert usage["started_at_utc"] == "2026-06-27T00:00:00.000Z"
    assert usage["finished_at_utc"] == "2026-06-27T00:00:05.123Z"
    assert usage["duration_ms"] == 5120
    assert usage["usage_currency"] == "USD"
    assert usage["usage_amount"] == 1.42
    assert usage["usage_credits"] == 142
    assert "datasetId" in str(usage["usage_meta"])
