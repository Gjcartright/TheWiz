from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from quant_platform.research_paper_math_audit import run_paper_critical_checks
from quant_platform.research_paper_reviews import build_paper_adversarial_reviews

REPO_ROOT = Path(__file__).resolve().parents[1]


def _seed_paper_inputs(root: Path) -> None:
    report_dir = root / "reports" / "research" / "papers"
    report_dir.mkdir(parents=True)
    config_dir = root / "config"
    config_dir.mkdir()
    config_path = config_dir / "research_paper_assessments.yaml"
    shutil.copy2(
        REPO_ROOT / "config" / "research_paper_assessments.yaml",
        config_path,
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    papers = payload["papers"]
    inventory_rows = []
    queue_rows = []
    page_dir = report_dir / "fixture_pages"
    page_dir.mkdir()
    for paper_id, assessment in papers.items():
        page_path = page_dir / f"{paper_id}.jsonl"
        page_path.write_text(
            json.dumps(
                {
                    "page_number": 1,
                    "text": str(assessment["claim_locator"]),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        title = f"Synthetic fixture for {paper_id}"
        inventory_rows.append(
            {
                "paper_id": paper_id,
                "title": title,
                "source_path": f"fixture://{paper_id}",
                "sha256": "a" * 64,
                "doi": "",
                "source_authenticity": "synthetic_test_fixture",
                "pages_path": str(page_path),
            }
        )
        queue_rows.append({"paper_id": paper_id, "title": title})
    pd.DataFrame(inventory_rows).to_csv(report_dir / "paper_inventory.csv", index=False)
    pd.DataFrame(queue_rows).to_csv(report_dir / "paper_review_queue.csv", index=False)


def test_paper_adversarial_review_covers_every_ingested_paper(tmp_path):
    _seed_paper_inputs(tmp_path)

    result = build_paper_adversarial_reviews(root=tmp_path)

    summary = pd.read_csv(result.paths["paper_review_summary"])
    assert len(summary) == 23
    assert not summary["live_signal_eligible"].astype(bool).any()
    assert set(summary["promotion_status"]) == {"blocked_research_only"}
    assert result.summary["gap_analyses"] == 23
    assert result.summary["premortems"] == 23
    assert result.summary["red_team_reviews"] == 23


def test_known_lookahead_and_math_risks_are_preserved(tmp_path):
    _seed_paper_inputs(tmp_path)

    result = build_paper_adversarial_reviews(root=tmp_path)

    reviews = pd.read_csv(result.paths["paper_review_summary"])
    assert (
        reviews.loc[reviews["paper_id"].eq("paper_8ee8ed6d01c3aa01"), "evidence_verdict"].item()
        == "contradicted"
    )
    gaps = pd.read_csv(result.paths["paper_gap_analysis"])
    assert gaps.loc[gaps["paper_id"].eq("paper_f0b8fced125866b5"), "severity"].item() == "critical"
    hypotheses = pd.read_csv(result.paths["paper_hypothesis_queue"])
    assert not hypotheses["execution_eligible"].astype(bool).any()
    knowledge = pd.read_csv(result.paths["paper_research_rows"])
    assert not knowledge["live_signal_eligible"].astype(bool).any()
    assert set(knowledge["promotion_authority"]) == {"local_point_in_time_validation_only"}


def test_critical_checks_block_formula_and_temporal_lineage_failures(tmp_path):
    _seed_paper_inputs(tmp_path)

    result = run_paper_critical_checks(root=tmp_path)

    checks = pd.read_csv(result.paths["paper_math_reconciliation"])
    assert set(checks["status"]) == {"fail", "blocked"}
    assert checks["promotion_blocker"].astype(bool).all()
    assert result.summary["discovery_test_overlap_days"] > 700
    assert result.summary["execution_eligible"] == 0
