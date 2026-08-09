from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_platform.research_extract_youtube import extract_youtube_research
from quant_platform.research_ingestion import ingest_research_source
from quant_platform.youtube_brain import (
    _collection_due,
    build_youtube_brain,
    build_youtube_brain_dashboard,
    build_youtube_pair_hypotheses,
    refresh_youtube_collection,
    refresh_youtube_outcome_memory,
    youtube_brain_paths,
)


def _write_catalog(path: Path, entries: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"channel": "Crypto Wizards", "channel_id": "channel-1", "entries": entries}),
        encoding="utf-8",
    )
    return path


def _write_audit(root: Path) -> None:
    audit = root / "reports" / "research" / "youtube_audit.md"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(
        "\n".join(
            [
                "# Audit",
                "",
                "## Section Without Direct Evidence",
                "",
                "### Python Course and Automation",
                "URL: https://www.youtube.com/watch?v=video000001",
                "Why it matters:",
                "- Pair trading execution needs cost controls.",
                "Repo implication:",
                "- Do not promote without forward testing and slippage evidence.",
                "",
                "### OU Spread Research",
                "URL: https://www.youtube.com/watch?v=video000002",
                "Why it matters:",
                "- OU spread uses half-life and Hurst evidence.",
                "Repo implication:",
                "- Prioritize OU mode for research only.",
            ]
        ),
        encoding="utf-8",
    )
    ingest_research_source(
        source_type="youtube",
        title="YouTube Audit",
        source_path_or_url=str(audit),
        root=root,
        channel_or_publisher="Crypto Wizards",
        review_status="reviewed",
        confidence=0.85,
    )


def _write_formula_catalog(root: Path) -> None:
    path = root / "reports" / "research" / "cryptowizards_formula_catalog_2026-07-01.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "formula_id": "ou_process",
                "formula_name": "Ornstein-Uhlenbeck",
                "category": "mean_reversion",
                "math": "dX=theta(mu-X)dt+sigma dW",
                "video_count": 1,
                "top_source_videos": "OU Spread Research (https://www.youtube.com/watch?v=video000002)",
                "repo_status": "implemented",
            },
            {
                "formula_id": "half_life",
                "formula_name": "Half-life",
                "category": "mean_reversion",
                "math": "-ln(2)/ln(phi)",
                "video_count": 1,
                "top_source_videos": "OU Spread Research (https://www.youtube.com/watch?v=video000002)",
                "repo_status": "implemented",
            },
            {
                "formula_id": "hurst",
                "formula_name": "Hurst",
                "category": "mean_reversion",
                "math": "H<0.5",
                "video_count": 1,
                "top_source_videos": "OU Spread Research (https://www.youtube.com/watch?v=video000002)",
                "repo_status": "implemented",
            },
            {
                "formula_id": "adf_stationarity",
                "formula_name": "ADF",
                "category": "stationarity",
                "math": "reject unit root",
                "video_count": 1,
                "top_source_videos": "OU Spread Research (https://www.youtube.com/watch?v=video000002)",
                "repo_status": "implemented",
            },
        ]
    ).to_csv(path, index=False)


def test_youtube_extraction_requires_direct_video_and_avoids_ou_substring(tmp_path):
    _write_audit(tmp_path)

    result = extract_youtube_research(root=tmp_path)
    rows = pd.read_csv(result.paths["youtube_research_rows"]).fillna("")
    registry = pd.read_csv(tmp_path / "data" / "external" / "research_sources" / "research_source_registry.csv").fillna("")

    assert rows["notes"].str.contains("youtube.com/watch", regex=False).all()
    course_rows = rows[rows["notes"].str.contains("Python Course", regex=False)]
    assert not course_rows.empty
    assert not course_rows["strategy_family"].astype(str).eq("ou").any()
    assert registry.loc[registry["source_type"].eq("youtube"), "processed_at"].astype(str).ne("").all()


def test_incremental_collection_detects_new_and_changed_videos(tmp_path):
    first = _write_catalog(
        tmp_path / "catalog-1.json",
        [{"id": "video000001", "title": "First title", "duration": 100}],
    )
    second = _write_catalog(
        tmp_path / "catalog-2.json",
        [
            {"id": "video000001", "title": "Updated title", "duration": 100},
            {"id": "video000002", "title": "New video", "duration": 120},
        ],
    )

    refresh_youtube_collection(root=tmp_path, catalog_path=first)
    refresh_youtube_collection(root=tmp_path, catalog_path=second)
    registry = pd.read_csv(youtube_brain_paths(tmp_path)["video_registry"])

    assert len(registry) == 2
    assert registry.loc[registry["video_id"].eq("video000002"), "is_new"].astype(bool).all()
    assert registry.loc[registry["video_id"].eq("video000001"), "metadata_changed"].astype(bool).all()


def test_collection_cadence_uses_last_live_refresh_not_local_registry_updates():
    status = pd.DataFrame([{"last_live_refresh_at": "2026-08-06T12:00:00+00:00"}])

    assert not _collection_due(status, now="2026-08-06T13:00:00+00:00", refresh_hours=24)
    assert _collection_due(status, now="2026-08-07T12:00:00+00:00", refresh_hours=24)


def test_youtube_brain_builds_safe_hypotheses_and_idempotent_outcome_memory(tmp_path):
    _write_audit(tmp_path)
    _write_formula_catalog(tmp_path)
    catalog = _write_catalog(
        tmp_path / "data" / "external" / "youtube" / "channel_videos.json",
        [
            {"id": "video000001", "title": "Python Course and Automation", "duration": 100},
            {"id": "video000002", "title": "OU Spread Research", "duration": 120},
        ],
    )
    refresh_youtube_collection(root=tmp_path, catalog_path=catalog)
    build_youtube_brain(root=tmp_path)

    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD/ETH-USD",
                "asset_x": "BTC-USD",
                "asset_y": "ETH-USD",
                "exchange": "dydx",
                "timeframe": "Daily",
                "exact_mode": "OU (Spread)",
                "sharpe": 2.1,
                "returns_total_pct": 25.0,
                "entry_level": 2.0,
                "exit_level": 0.0,
                "evidence_path": "wizard.csv",
            }
        ]
    ).to_csv(active / "crypto_wizards_live_scanner_capture.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD/ETH-USD",
                "timeframe": "1d",
                "wizard_exact_mode": "OU (Spread)",
                "test_trades": 10,
                "test_profit_factor": 1.8,
                "test_sharpe": 1.4,
                "oos_metric_pass": True,
                "acceptance": False,
            }
        ]
    ).to_csv(active / "workflow_walk_forward_ranked.csv", index=False)

    hypothesis_result = build_youtube_pair_hypotheses(root=tmp_path)
    hypotheses = pd.read_csv(hypothesis_result.paths["hypotheses"])
    assert len(hypotheses) == 1
    assert hypotheses.iloc[0]["mode_key"] == "ou_spread"
    assert bool(hypotheses.iloc[0]["wizard_discovery_pass"])
    assert not bool(hypotheses.iloc[0]["trade_authorized"])
    assert hypotheses.iloc[0]["promotion_authority"] == "none_research_only"

    first = refresh_youtube_outcome_memory(root=tmp_path)
    memory_path = first.paths["memory_jsonl"]
    first_lines = memory_path.read_text(encoding="utf-8").splitlines()
    refresh_youtube_outcome_memory(root=tmp_path)
    second_lines = memory_path.read_text(encoding="utf-8").splitlines()
    scorecard = pd.read_csv(first.paths["replication_scorecard"])
    assert scorecard.iloc[0]["lifecycle"] == "REPLICATING"
    assert len(first_lines) == len(second_lines) == 1

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD/ETH-USD",
                "venue": "binance",
                "network": "testnet",
                "lifecycle_status": "closed",
                "realized_return": 0.03,
            }
        ]
    ).to_csv(tmp_path / "reports" / "paper_trading_journal.csv", index=False)
    refresh_youtube_outcome_memory(root=tmp_path)
    scorecard = pd.read_csv(first.paths["replication_scorecard"])
    assert scorecard.iloc[0]["lifecycle"] == "REPLICATING"

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD/ETH-USD",
                "venue": "hyperliquid",
                "network": "testnet",
                "lifecycle_status": "closed",
                "realized_return": 0.03,
            }
        ]
    ).to_csv(tmp_path / "reports" / "paper_trading_journal.csv", index=False)
    refresh_youtube_outcome_memory(root=tmp_path)
    scorecard = pd.read_csv(first.paths["replication_scorecard"])
    assert scorecard.iloc[0]["lifecycle"] == "TESTNET_SUPPORTED"
    assert scorecard.iloc[0]["verified_paper_outcomes"] == 1

    dashboard = build_youtube_brain_dashboard(root=tmp_path)
    status = pd.read_csv(dashboard.paths["youtube_brain_status"])
    assert status.iloc[0]["promotion_authority"] == "none_research_only"
