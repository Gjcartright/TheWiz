from pathlib import Path

import pandas as pd

from quant_platform.orchestration.phase00_ccxt_evidence import (
    CcxtEvidenceContract,
    CcxtEvidenceRuntime,
)
from quant_platform.research_extract_ccxt import _parse_ccxt_spec, extract_research_knowledge
from quant_platform.research_extract_youtube import extract_youtube_research
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
    assert source.lane == ""
    assert source.since_ms is None
    assert source.end_ms is None
    explicit = _parse_ccxt_spec(
        "ccxt://binance:BTC/USDT;lane=binance_spot;timeframe=5m;limit=8;"
        "since_ms=0;end_ms=2400000;captured_at_ms=2400000;force_market_reload=true"
    )
    assert explicit is not None
    assert explicit.lane == "binance_spot"
    assert explicit.since_ms == 0
    assert explicit.end_ms == 2_400_000
    assert explicit.captured_at_ms == 2_400_000
    assert explicit.force_market_reload is True
    assert _parse_ccxt_spec("ccxt://binance:BTC/USDT;limit=invalid") is None
    assert _parse_ccxt_spec("ccxt://binance:BTC/USDT;unknown=value") is None

    result = extract_research_knowledge(root=tmp_path)
    rows = pd.read_csv(result.paths["ccxt_research_rows"])
    assert len(rows) == 0
    assert result.summary["sources"] == 1
    assert result.summary["ccxt_rows"] == 0
    assert result.summary["processed"] is False
    registry = active_research_sources(tmp_path)
    assert registry.iloc[0]["processed_at"] == ""


def test_ccxt_zero_row_capture_writes_failure_evidence_without_marking_processed(tmp_path):
    class _EmptyBinance:
        id = "binance"

        def __init__(self):
            self.has = {"fetchOHLCV": True}
            self.timeframes = {"5m": "5m"}

        def load_markets(self, *, reload: bool = False):
            del reload
            return {
                "BTC/USDT": {
                    "id": "BTCUSDT",
                    "symbol": "BTC/USDT",
                    "base": "BTC",
                    "quote": "USDT",
                    "type": "spot",
                    "spot": True,
                    "swap": False,
                    "future": False,
                    "contract": False,
                    "limits": {},
                    "precision": {},
                }
            }

        def fetch_ohlcv(self, **kwargs):
            del kwargs
            return []

    exchange = _EmptyBinance()
    contract_path = Path(__file__).resolve().parents[1] / "config" / "ccxt_evidence_contract.json"
    runtime = CcxtEvidenceRuntime(
        contract=CcxtEvidenceContract.from_path(contract_path),
        exchange_factory=lambda exchange_id, options: exchange,
        package_version="4.4.26",
        source_revision="c" * 64,
        runtime_fingerprint={"test": "zero_row_rejection"},
    )
    ingest_research_source(
        source_type="ccxt",
        title="Empty CCXT Source",
        source_path_or_url=(
            "ccxt://binance:BTC/USDT;lane=binance_spot;timeframe=5m;limit=8;"
            "since_ms=0;end_ms=2400000;captured_at_ms=2400000"
        ),
        root=tmp_path,
    )

    result = extract_research_knowledge(root=tmp_path, evidence_runtime=runtime)

    assert result.summary["ccxt_rows"] == 0
    assert result.summary["accepted_sources"] == 0
    assert result.summary["failed_sources"] == 1
    assert result.summary["processed"] is False
    registry = active_research_sources(tmp_path)
    assert registry.iloc[0]["processed_at"] == ""
    receipts = list((tmp_path / "data" / "external" / "ccxt" / "evidence").glob("*/receipt.json"))
    assert len(receipts) == 1
    assert '"status": "empty"' in receipts[0].read_text(encoding="utf-8")


def test_ccxt_extractor_maps_structured_rows_from_accepted_evidence(tmp_path):
    def _build_ohlcv(rows: int, base: float, drift: float, volume: float) -> list[list[float]]:
        payload: list[list[float]] = []
        price = base
        for idx in range(rows):
            payload.append([idx * 300_000, price, price * 1.01, price * 0.99, price, volume])
            price *= 1 + drift
        return payload

    class _MockBinance:
        id = "binance"

        def __init__(self):
            self.has = {"fetchOHLCV": True}
            self.timeframes = {"5m": "5m"}

        def load_markets(self, *, reload: bool = False):
            del reload
            return {
                symbol: {
                    "id": symbol.replace("/", ""),
                    "symbol": symbol,
                    "base": symbol.split("/")[0],
                    "quote": "USDT",
                    "type": "spot",
                    "spot": True,
                    "swap": False,
                    "future": False,
                    "contract": False,
                    "settle": None,
                    "contractSize": None,
                    "linear": None,
                    "inverse": None,
                    "marginModes": {},
                    "active": True,
                    "limits": {},
                    "precision": {},
                }
                for symbol in ("BTC/USDT", "ETH/USDT")
            }

        def fetch_ohlcv(
            self,
            *,
            symbol: str,
            timeframe: str,
            since: int,
            limit: int,
            params: dict[str, object],
        ):
            del timeframe, params
            if "BTC" in symbol:
                payload = _build_ohlcv(120, 30000.0, 0.001, 0.25)
            else:
                payload = _build_ohlcv(120, 2000.0, -0.0002, 0.25)
            return [row for row in payload if row[0] >= since][:limit]

    exchange = _MockBinance()
    contract_path = Path(__file__).resolve().parents[1] / "config" / "ccxt_evidence_contract.json"
    runtime = CcxtEvidenceRuntime(
        contract=CcxtEvidenceContract.from_path(contract_path),
        exchange_factory=lambda exchange_id, options: exchange,
        package_version="4.4.26",
        source_revision="b" * 64,
        runtime_fingerprint={"test": "research_ingestion"},
    )

    ingest_research_source(
        source_type="ccxt",
        title="Mock CCXT Source",
        source_path_or_url=(
            "ccxt://binance:BTC/USDT,ETH/USDT;lane=binance_spot;timeframe=5m;"
            "limit=120;since_ms=0;end_ms=36000000;captured_at_ms=36000000"
        ),
        root=tmp_path,
        confidence=0.7,
    )
    result = extract_research_knowledge(root=tmp_path, evidence_runtime=runtime)
    rows = pd.read_csv(result.paths["ccxt_research_rows"])
    assert len(rows) > 0
    assert set(rows["row_type"]).issuperset(
        {"risk_prior", "regime_condition", "feature_idea", "pair_filter", "anti_pattern"}
    )
    assert any("BTC-USDT" in str(value) for value in rows["pair_filter"])
    assert rows["pair_filter"].str.contains("ETH-USDT").any()
    assert (
        rows[rows["row_type"] == "anti_pattern"]["anti_pattern"].str.contains("low_liquidity").any()
    )
    assert rows["evidence_path"].str.contains("data/external/ccxt/evidence").all()
    assert rows["evidence_path"].map(lambda value: Path(value).exists()).all()
    assert result.summary["accepted_sources"] == 1
    assert result.summary["failed_sources"] == 0
    assert result.summary["processed"] is True
    registry = active_research_sources(tmp_path)
    assert registry.iloc[0]["processed_at"] != ""
    knowledge = build_research_knowledge_store(root=tmp_path)
    features = pd.read_csv(knowledge.paths["research_features"])
    risks = pd.read_csv(knowledge.paths["research_risk_priors"])
    rules = pd.read_csv(knowledge.paths["research_rules"])
    assert not features.empty
    assert not risks.empty
    assert not rules.empty
