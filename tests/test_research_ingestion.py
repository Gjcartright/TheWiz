from pathlib import Path

import pandas as pd
from types import SimpleNamespace

from quant_platform.research_extract_youtube import extract_youtube_research
from quant_platform.research_extract_ccxt import _parse_ccxt_spec, extract_research_knowledge
from quant_platform.research_ingestion import (
    active_research_sources,
    build_research_source_registry,
    ingest_research_source,
)
from quant_platform.research_knowledge_store import build_research_knowledge_store


def test_research_registry_quarantine_excludes_source_from_refresh(tmp_path):
    audit = tmp_path / "audit.md"
    audit.write_text(
        "\n".join(
            [
                "# Audit",
                "",
                "### Dynamic Pair Trading",
                "URL: https://example.test/video",
                "Why it matters:",
                "- Cointegration and pair trading remain useful.",
                "Repo implication:",
                "- Use OU mode for research.",
            ]
        ),
        encoding="utf-8",
    )

    ingest_research_source(
        source_type="youtube",
        title="Crypto Wizards Audit",
        source_path_or_url=str(audit),
        root=tmp_path,
        confidence=0.8,
        status="active",
        quarantine_status="active",
    )
    active = active_research_sources(tmp_path)
    assert len(active) == 1

    manifest = next((tmp_path / "data" / "external" / "research_sources" / "manifests").glob("*.json"))
    payload = manifest.read_text(encoding="utf-8").replace('"quarantine_status": "active"', '"quarantine_status": "quarantined"')
    manifest.write_text(payload, encoding="utf-8")

    build_research_source_registry(root=tmp_path)
    refreshed = active_research_sources(tmp_path)
    assert refreshed.empty


def test_youtube_refresh_builds_structured_knowledge_rows(tmp_path):
    audit = tmp_path / "audit.md"
    audit.write_text(
        "\n".join(
            [
                "# Audit",
                "",
                "### Copula Pair Trading Review",
                "URL: https://example.test/copula",
                "Why it matters:",
                "- Pair trading and cointegration can work with funding-aware filters.",
                "Repo implication:",
                "- Prioritize copula mode for dynamic regimes.",
                "- Do not promote without paper trading evidence.",
                "- Track funding as a feature.",
                "- Warn if execution automation is unreliable.",
            ]
        ),
        encoding="utf-8",
    )
    ingest_research_source(
        source_type="youtube",
        title="Copula Review",
        source_path_or_url=str(audit),
        root=tmp_path,
        confidence=0.9,
    )

    extract_result = extract_youtube_research(root=tmp_path)
    knowledge_result = build_research_knowledge_store(root=tmp_path)

    rows = pd.read_csv(extract_result.paths["youtube_research_rows"])
    assert not rows.empty
    assert set(rows["row_type"]).issubset(
        {
            "strategy_hint",
            "feature_idea",
            "risk_prior",
            "mode_preference",
            "regime_condition",
            "pair_filter",
            "anti_pattern",
            "execution_warning",
        }
    )
    assert {"source_id", "source_type", "source_title", "evidence_path"}.issubset(rows.columns)

    summary = pd.read_csv(knowledge_result.paths["research_knowledge_summary"])
    assert set(summary["table"]) >= {"research_rules", "research_features", "research_strategy_hints"}


def test_ccxt_source_type_and_parser_without_ccxt_dependency(tmp_path):
    ingest_research_source(
        source_type="ccxt",
        title="ccxt smoke source",
        source_path_or_url="ccxt://binance:BTC/USDT,ETH/USDT;timeframe=5m;limit=120",
        root=tmp_path,
        confidence=0.6,
    )
    source = _parse_ccxt_spec("ccxt://binance:BTC/USDT,ETH/USDT;timeframe=1m,5m;limit=200")
    assert source is not None
    assert source.exchange == "binance"
    assert source.symbols == ["BTC/USDT", "ETH/USDT"]
    assert source.timeframes == ["1m", "5m"]
    assert source.limit == 200

    result = extract_research_knowledge(root=tmp_path)
    rows = pd.read_csv(result.paths["ccxt_research_rows"])
    assert len(rows) == 0
    assert result.summary["sources"] == 1
    assert result.summary["ccxt_rows"] == 0


def test_ccxt_extractor_maps_structured_rows_from_research_feed(tmp_path, monkeypatch):
    def _build_ohlcv(rows: int, base: float, drift: float, volume: float) -> list[list[float]]:
        payload: list[list[float]] = []
        price = base
        for idx in range(rows):
            payload.append([idx, price, price * 1.01, price * 0.99, price, volume])
            price *= 1 + drift
        return payload

    class _MockBinance:
        def __init__(self, *_, **__):
            pass

        def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int):
            if "BTC" in symbol:
                return _build_ohlcv(120, 30000.0, 0.001, 0.25)
            return _build_ohlcv(120, 2000.0, -0.0002, 0.25)

    class _MockCcxt(SimpleNamespace):
        binance = _MockBinance

    monkeypatch.setattr("quant_platform.research_extract_ccxt.ccxt", _MockCcxt)

    ingest_research_source(
        source_type="ccxt",
        title="Mock CCXT Source",
        source_path_or_url="ccxt://binance:BTC/USDT,ETH/USDT;timeframe=5m;limit=120",
        root=tmp_path,
        confidence=0.7,
    )
    result = extract_research_knowledge(root=tmp_path)
    rows = pd.read_csv(result.paths["ccxt_research_rows"])
    assert len(rows) > 0
    assert set(rows["row_type"]).issuperset({"risk_prior", "regime_condition", "feature_idea", "pair_filter", "anti_pattern"})
    assert any("BTC-USDT" in str(value) for value in rows["pair_filter"])
    assert rows["pair_filter"].str.contains("ETH-USDT").any()
    assert rows[rows["row_type"] == "anti_pattern"]["anti_pattern"].str.contains("low_liquidity").any()
    knowledge = build_research_knowledge_store(root=tmp_path)
    features = pd.read_csv(knowledge.paths["research_features"])
    risks = pd.read_csv(knowledge.paths["research_risk_priors"])
    rules = pd.read_csv(knowledge.paths["research_rules"])
    assert not features.empty
    assert not risks.empty
    assert not rules.empty
