from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_platform.active_pipeline import CommandResult
from quant_platform.research_extract_udemy import (
    extract_udemy_research,
    refresh_udemy_research,
    search_udemy_research,
)
from quant_platform.research_knowledge_store import query_research_knowledge
from quant_platform.rl.base_rl_lane import run_augmented_rl


def _write_udemy_sources(root, *, include_transcript_body: bool = False):
    reports = root / "reports" / "research"
    reports.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "course": "Financial Econometrics",
            "section": "Cointegration",
            "video_title": "Error Correction Models",
            "review_status": "reviewed",
            "evidence_source": "udemy_transcript",
            "transcript_word_count": 1200,
            "transcript_cue_count": 80,
            "topic_tags": "cointegration|ecm|granger",
            "lecture_url": "https://www.udemy.com/course/example/learn/lecture/1#overview",
        },
        {
            "course": "Machine Learning Applied to Trading",
            "section": "Reinforcement Learning",
            "video_title": "PPO Trading Environment",
            "review_status": "reviewed",
            "evidence_source": "udemy_ai_assistant",
            "transcript_word_count": 0,
            "transcript_cue_count": 0,
            "topic_tags": "reinforcement|risk_sizing",
            "lecture_url": "https://www.udemy.com/course/example/learn/lecture/2#overview",
        },
    ]
    if include_transcript_body:
        for row in rows:
            row["transcript_text"] = "copyrighted lecture text"
    coverage = reports / "shaun_mcdonogh_udemy_video_review.csv"
    pd.DataFrame(rows).to_csv(coverage, index=False)
    review = reports / "shaun_mcdonogh_udemy_implementation_review.md"
    review.write_text(
        "\n".join(
            [
                "# Course Review",
                "",
                "## Quantitative Findings",
                "",
                "- Use rolling point-in-time econometrics and copula diagnostics.",
                "",
                "## Changes Recommended For TheWiz",
                "",
                "1. Add costed walk-forward feature ablations.",
            ]
        ),
        encoding="utf-8",
    )
    return coverage, review


def test_udemy_refresh_builds_research_only_knowledge_and_preserves_provenance(tmp_path):
    coverage, review = _write_udemy_sources(tmp_path)

    result = refresh_udemy_research(root=tmp_path, coverage_path=coverage, review_path=review)

    lecture_index = pd.read_csv(result.paths["lecture_index"]).fillna("")
    assert len(lecture_index) == 2
    assert set(lecture_index["evidence_source"]) == {"udemy_transcript", "udemy_ai_assistant"}
    assert lecture_index["transcript_text_stored"].astype(bool).sum() == 0
    assert lecture_index["live_signal_eligible"].astype(bool).sum() == 0
    assert set(lecture_index["promotion_authority"]) == {"none_research_only"}

    knowledge = pd.read_csv(result.paths["udemy_research_rows"]).fillna("")
    assert not knowledge.empty
    assert set(knowledge["source_type"]) == {"udemy"}
    assert knowledge["live_signal_eligible"].astype(bool).sum() == 0
    assert set(knowledge["promotion_authority"]) == {"none_research_only"}
    assert not {"transcript", "transcript_text", "caption_text", "full_text"}.intersection(knowledge.columns)

    hypotheses = pd.read_csv(result.paths["hypotheses"]).fillna("")
    assert len(hypotheses) == 8
    assert set(hypotheses["status"]) == {"UNVERIFIED"}
    assert hypotheses["trade_authorized"].astype(bool).sum() == 0
    assert {"copula", "reinforcement_learning", "execution"}.issubset(set(hypotheses["strategy_family"]))

    source_summary = pd.read_csv(result.paths["research_knowledge_source_summary"]).fillna("")
    assert "udemy" in set(source_summary["source_type"])
    assert source_summary.loc[source_summary["source_type"] == "udemy", "live_signal_eligible_rows"].sum() == 0


def test_udemy_retrieval_finds_lecture_metadata_and_structured_knowledge(tmp_path):
    coverage, review = _write_udemy_sources(tmp_path)
    refresh_udemy_research(root=tmp_path, coverage_path=coverage, review_path=review)

    lecture_results = search_udemy_research("error correction", root=tmp_path)
    assert len(lecture_results) == 1
    assert lecture_results.iloc[0]["evidence_source"] == "udemy_transcript"

    knowledge_results = query_research_knowledge(root=tmp_path, source_type="udemy", text="copula")
    assert not knowledge_results.empty
    assert set(knowledge_results["source_type"]) == {"udemy"}
    assert knowledge_results["live_signal_eligible"].astype(bool).sum() == 0


def test_udemy_ingestion_rejects_transcript_body_columns(tmp_path):
    coverage, review = _write_udemy_sources(tmp_path, include_transcript_body=True)

    with pytest.raises(ValueError, match="udemy_ledger_contains_transcript_content"):
        extract_udemy_research(root=tmp_path, coverage_path=coverage, review_path=review)


def test_augmented_rl_observes_udemy_context_without_execution_authority(tmp_path, monkeypatch):
    coverage, review = _write_udemy_sources(tmp_path)
    refresh_udemy_research(root=tmp_path, coverage_path=coverage, review_path=review)
    reports = tmp_path / "reports" / "rl"
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"pair": "BTC-USD-ETH-USD", "rl_policy_name": "policy_a", "rl_supported": True}]
    ).to_csv(reports / "base_rl_pair_coverage.csv", index=False)
    monkeypatch.setattr(
        "quant_platform.rl.base_rl_lane.run_base_rl",
        lambda root=tmp_path, pair_id="": CommandResult(paths={}, summary={}),
    )

    result = run_augmented_rl(root=tmp_path)

    assert result.summary["udemy_research_rows"] > 0
    assert "udemy" in result.summary["research_source_types"].split(";")
    evidence = json.loads(result.paths["augmented_evidence"].read_text(encoding="utf-8"))
    assert evidence["educational_evidence_authority"] == "research_only_no_trade_authority"
    assert evidence["udemy_research_rows"] > 0
