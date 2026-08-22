from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quant_platform.active_pipeline import ROOT, CommandResult
from quant_platform.research_extraction import normalize_extraction_rows

REQUIRED_ASSESSMENT_FIELDS = {
    "relevance",
    "strategy_families",
    "evidence_verdict",
    "guidance",
    "primary_claim",
    "claim_locator",
    "good",
    "bad",
    "do",
    "dont",
    "neither",
    "gap",
    "gap_severity",
    "premortem",
    "warning",
    "mitigation",
    "red_attack",
    "red_test",
    "math_focus",
    "hypothesis",
    "data_needed",
    "confidence",
}


def _report_base(root: Path) -> Path:
    return root / "reports" / "research" / "papers"


def _config_path(root: Path) -> Path:
    return root / "config" / "research_paper_assessments.yaml"


def _load_assessments(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = _config_path(root)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    papers = payload.get("papers") or {}
    if not isinstance(papers, dict):
        raise TypeError("paper assessment config must contain a papers mapping")
    for paper_id, assessment in papers.items():
        if not isinstance(assessment, dict):
            raise TypeError(f"paper assessment must be a mapping: {paper_id}")
        missing = sorted(REQUIRED_ASSESSMENT_FIELDS - set(assessment))
        if missing:
            raise ValueError(f"paper assessment missing fields for {paper_id}: {','.join(missing)}")
    return payload, papers


def _pages(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _source_pages(pages: list[dict[str, Any]], locator: str) -> str:
    target = str(locator).casefold().strip()
    if not target:
        return ""
    matches = [
        int(page["page_number"]) for page in pages if target in str(page.get("text", "")).casefold()
    ]
    return ";".join(str(number) for number in matches[:6])


def _write_markdown(
    *,
    path: Path,
    inventory: pd.Series,
    assessment: dict[str, Any],
    source_pages: str,
    review_meta: dict[str, Any],
) -> None:
    families = ", ".join(str(value) for value in assessment["strategy_families"])
    hypothesis = str(assessment["hypothesis"] or "No trading hypothesis authorized.")
    lines = [
        f"# {inventory['title']}",
        "",
        f"- paper id: `{inventory['paper_id']}`",
        f"- source: `{inventory['source_path']}`",
        f"- sha256: `{inventory['sha256']}`",
        f"- DOI: {inventory['doi'] or 'not recorded'}",
        f"- source pages for primary locator: {source_pages or 'not located automatically'}",
        f"- review version: `{review_meta.get('review_version', '')}`",
        f"- relevance: `{assessment['relevance']}`",
        f"- strategy families: {families or 'none'}",
        f"- evidence verdict: `{assessment['evidence_verdict']}`",
        f"- guidance: `{assessment['guidance']}`",
        "- live-signal eligibility: `false`",
        "",
        "## Primary Claim",
        "",
        str(assessment["primary_claim"]),
        "",
        "## Good",
        "",
        str(assessment["good"]),
        "",
        "## Bad",
        "",
        str(assessment["bad"]),
        "",
        "## Do",
        "",
        str(assessment["do"]),
        "",
        "## Don't",
        "",
        str(assessment["dont"]),
        "",
        "## Neither",
        "",
        str(assessment["neither"]),
        "",
        "## Gap Analysis",
        "",
        f"- severity: `{assessment['gap_severity']}`",
        f"- gap: {assessment['gap']}",
        "",
        "## Premortem",
        "",
        f"- predicted failure: {assessment['premortem']}",
        f"- early warning: {assessment['warning']}",
        f"- mitigation: {assessment['mitigation']}",
        "",
        "## Red Team",
        "",
        f"- attack: {assessment['red_attack']}",
        f"- falsification test: {assessment['red_test']}",
        "",
        "## Math Review",
        "",
        str(assessment["math_focus"]),
        "",
        "Status: scoped, not yet independently reproduced.",
        "",
        "## Hypothesis Disposition",
        "",
        hypothesis,
        "",
        f"Required data: {assessment['data_needed'] or 'none for current project scope'}",
        "",
        "## Governance",
        "",
        (
            "This is an adversarial research assessment, not acceptance evidence. The paper has "
            "no authority to alter a model, signal, position, or execution policy."
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "\n".join(lines) + "\n", encoding="utf-8")


def _contradictions() -> pd.DataFrame:
    rows = [
        {
            "topic": "stationarity_and_adaptation",
            "paper_a": "paper_8ee8ed6d01c3aa01",
            "paper_b": "paper_83bedd65a3817a87",
            "relationship": "tension",
            "finding": "Adaptive RL claims no stationarity requirement, while fractional cointegration evidence shows equilibrium persistence and leadership are regime-dependent.",
            "resolution_test": "Compare causal RL against break-aware deterministic and fractional-cointegration baselines on identical folds.",
        },
        {
            "topic": "tail_risk_objective",
            "paper_a": "paper_f0b8fced125866b5",
            "paper_b": "paper_9acd25bc0eb74b5f",
            "relationship": "complement_and_tension",
            "finding": "A mean-variance reward suppresses ordinary volatility but does not directly model scarce catastrophic tails addressed by EVT.",
            "resolution_test": "Compare variance, empirical CVaR, EVT-CVaR, and hard-limit rewards with matched take-rate.",
        },
        {
            "topic": "cost_realism",
            "paper_a": "paper_bd860d3475116e96",
            "paper_b": "paper_9d33d77526db17f3",
            "relationship": "complement",
            "finding": "Proportional-cost sensitivity supports selective trading, while no-trade-region theory suggests explicit state-dependent rebalancing boundaries.",
            "resolution_test": "Use observed L2 costs to compare predictive selectivity with static and dynamic no-trade bands.",
        },
        {
            "topic": "pair_discovery_authority",
            "paper_a": "paper_a2e2cdcb8d0f2fac",
            "paper_b": "paper_c60e9f5f6a175ce5",
            "relationship": "complement",
            "finding": "Characteristic clustering and graph dependencies may improve discovery, but neither establishes tradable stationarity.",
            "resolution_test": "Score discovery lift only by future local acceptance and costed outcomes.",
        },
        {
            "topic": "distribution_and_crisis_regime",
            "paper_a": "paper_5202676052f4c33f",
            "paper_b": "paper_2dda18bfbe5c0988",
            "relationship": "complement",
            "finding": "Multifractal distribution shape and Bell-violation dependence are candidate regime descriptors with no demonstrated incremental trading value.",
            "resolution_test": "Compare both against volatility, drawdown, rank dependence, and funding baselines in sealed folds.",
        },
        {
            "topic": "reported_rl_validity",
            "paper_a": "paper_8ee8ed6d01c3aa01",
            "paper_b": "paper_bd860d3475116e96",
            "relationship": "contradiction_in_evidence_quality",
            "finding": "The clustering-RL paper contains direct discovery look-ahead, while the panel-prediction paper documents stronger cost and inference controls but still omits bid-ask spread.",
            "resolution_test": "Quarantine the former results and independently reproduce the latter before any model comparison.",
        },
    ]
    return pd.DataFrame(rows)


def build_paper_adversarial_reviews(*, root: Path = ROOT) -> CommandResult:
    report_base = _report_base(root)
    inventory_path = report_base / "paper_inventory.csv"
    if not inventory_path.exists():
        raise FileNotFoundError("paper inventory missing; run ingest-paper-library first")
    inventory = pd.read_csv(inventory_path).fillna("")
    review_meta, assessments = _load_assessments(root)
    inventory_ids = set(inventory["paper_id"].astype(str))
    assessment_ids = set(assessments)
    missing = sorted(inventory_ids - assessment_ids)
    unexpected = sorted(assessment_ids - inventory_ids)
    if missing or unexpected:
        raise ValueError(
            f"paper assessment coverage mismatch; missing={missing}; unexpected={unexpected}"
        )

    summary_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    premortem_rows: list[dict[str, Any]] = []
    red_rows: list[dict[str, Any]] = []
    math_rows: list[dict[str, Any]] = []
    guidance_rows: list[dict[str, Any]] = []
    hypothesis_rows: list[dict[str, Any]] = []
    review_dir = report_base / "reviews"

    for _, paper in inventory.iterrows():
        paper_id = str(paper["paper_id"])
        assessment = assessments[paper_id]
        pages = _pages(str(paper["pages_path"]))
        source_pages = _source_pages(pages, str(assessment["claim_locator"]))
        review_path = review_dir / f"{paper_id}.md"
        _write_markdown(
            path=review_path,
            inventory=paper,
            assessment=assessment,
            source_pages=source_pages,
            review_meta=review_meta,
        )
        common = {
            "paper_id": paper_id,
            "title": paper["title"],
            "source_pages": source_pages,
            "source_authenticity": paper.get("source_authenticity", "unverified"),
            "evidence_path": str(review_path),
            "live_signal_eligible": False,
        }
        summary_rows.append(
            {
                **common,
                "relevance": assessment["relevance"],
                "strategy_families": ";".join(assessment["strategy_families"]),
                "evidence_verdict": assessment["evidence_verdict"],
                "guidance": assessment["guidance"],
                "confidence": float(assessment["confidence"]),
                "gap_severity": assessment["gap_severity"],
                "math_review_status": "scoped_pending_independent_reproduction",
                "local_replication_status": "not_started",
                "promotion_status": "blocked_research_only",
            }
        )
        claim_rows.append(
            {
                **common,
                "claim_id": f"{paper_id}_primary_claim",
                "author_claim_paraphrase": assessment["primary_claim"],
                "our_interpretation": assessment["do"],
                "evidence_verdict": assessment["evidence_verdict"],
                "confidence": float(assessment["confidence"]),
                "uses_future_data": "unknown_pending_replication",
                "promotion_authority": "local_point_in_time_validation_only",
            }
        )
        gap_rows.append(
            {
                **common,
                "severity": assessment["gap_severity"],
                "gap": assessment["gap"],
                "blocking": assessment["relevance"] != "not_applicable",
                "resolution_status": "open",
            }
        )
        premortem_rows.append(
            {
                **common,
                "failure_mode": assessment["premortem"],
                "early_warning": assessment["warning"],
                "mitigation": assessment["mitigation"],
                "stop_condition": "promotion_gate_fails_or_warning_is_observed",
            }
        )
        red_rows.append(
            {
                **common,
                "attack": assessment["red_attack"],
                "falsification_test": assessment["red_test"],
                "status": "required_before_promotion",
            }
        )
        math_rows.append(
            {
                **common,
                "math_focus": assessment["math_focus"],
                "status": "scoped_pending_independent_reproduction",
                "blocking": assessment["relevance"] != "not_applicable",
            }
        )
        for category in ("good", "bad", "do", "dont", "neither"):
            guidance_rows.append(
                {
                    **common,
                    "category": category,
                    "finding": assessment[category],
                }
            )
        if str(assessment["hypothesis"]).strip():
            hypothesis_rows.append(
                {
                    **common,
                    "hypothesis_id": f"{paper_id}_h1",
                    "hypothesis": assessment["hypothesis"],
                    "required_data": assessment["data_needed"],
                    "status": "proposed_unregistered",
                    "acceptance_authority": "local_costed_walkforward_only",
                    "execution_eligible": False,
                }
            )

    outputs = {
        "paper_review_summary": report_base / "paper_review_summary.csv",
        "paper_claim_ledger": report_base / "paper_claim_ledger.csv",
        "paper_gap_analysis": report_base / "paper_gap_analysis.csv",
        "paper_premortem": report_base / "paper_premortem.csv",
        "paper_red_team": report_base / "paper_red_team.csv",
        "paper_math_review": report_base / "paper_math_review.csv",
        "paper_guidance": report_base / "paper_good_bad_do_dont_neither.csv",
        "paper_hypothesis_queue": report_base / "paper_hypothesis_queue.csv",
        "paper_contradictions": report_base / "paper_contradiction_matrix.csv",
    }
    frames = {
        "paper_review_summary": pd.DataFrame(summary_rows),
        "paper_claim_ledger": pd.DataFrame(claim_rows),
        "paper_gap_analysis": pd.DataFrame(gap_rows),
        "paper_premortem": pd.DataFrame(premortem_rows),
        "paper_red_team": pd.DataFrame(red_rows),
        "paper_math_review": pd.DataFrame(math_rows),
        "paper_guidance": pd.DataFrame(guidance_rows),
        "paper_hypothesis_queue": pd.DataFrame(hypothesis_rows),
        "paper_contradictions": _contradictions(),
    }
    for key, frame in frames.items():
        atomic_write_csv(frame, outputs[key], index=False)

    summary_md = report_base / "paper_library_review.md"
    atomic_write_text(summary_md, _summary_markdown(frames, inventory), encoding="utf-8")
    outputs["paper_library_review_md"] = summary_md

    knowledge_rows = _knowledge_rows(inventory, assessments, review_dir)
    knowledge_path = root / "data" / "processed" / "research_knowledge" / "paper_research_rows.csv"
    knowledge_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(knowledge_rows, knowledge_path, index=False)
    outputs["paper_research_rows"] = knowledge_path

    review_queue_path = report_base / "paper_review_queue.csv"
    review_queue = pd.read_csv(review_queue_path).fillna("")
    review_queue["review_status"] = "initial_adversarial_review_complete"
    review_queue["gap_analysis_status"] = "complete"
    review_queue["premortem_status"] = "complete"
    review_queue["red_team_status"] = "complete"
    review_queue["math_review_status"] = "scoped_pending_independent_reproduction"
    review_queue["local_replication_status"] = "not_started"
    review_queue["blocking_reason"] = "math_and_local_replication_required"
    atomic_write_csv(review_queue, review_queue_path, index=False)
    outputs["paper_review_queue"] = review_queue_path

    return CommandResult(
        paths=outputs,
        summary={
            "papers_reviewed": len(summary_rows),
            "claims": len(claim_rows),
            "gap_analyses": len(gap_rows),
            "premortems": len(premortem_rows),
            "red_team_reviews": len(red_rows),
            "hypotheses_proposed": len(hypothesis_rows),
            "knowledge_rows": len(knowledge_rows),
            "critical_gaps": sum(row["severity"] == "critical" for row in gap_rows),
            "execution_eligible": 0,
        },
    )


def _knowledge_rows(
    inventory: pd.DataFrame,
    assessments: dict[str, dict[str, Any]],
    review_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, paper in inventory.iterrows():
        paper_id = str(paper["paper_id"])
        assessment = assessments[paper_id]
        families = assessment["strategy_families"]
        family = str(families[0]) if families else ""
        common = {
            "strategy_family": family,
            "confidence": float(assessment["confidence"]),
            "source_id": paper_id,
            "source_type": "pdf",
            "source_title": paper["title"],
            "evidence_path": str(review_dir / f"{paper_id}.md"),
            "evidence_source": "paper_adversarial_review",
            "point_in_time_status": "paper_claim_not_locally_validated",
            "live_signal_eligible": False,
            "promotion_authority": "local_point_in_time_validation_only",
            "source_url": paper.get("doi", ""),
            "review_status": "initial_adversarial_review_complete",
            "notes": (
                f"verdict={assessment['evidence_verdict']};guidance={assessment['guidance']};"
                f"source_authenticity={paper.get('source_authenticity', 'unverified')}"
            ),
        }
        rows.append(
            {
                **common,
                "row_id": f"{paper_id}_do",
                "row_type": "strategy_hint",
                "entry_logic": assessment["do"],
            }
        )
        rows.append(
            {
                **common,
                "row_id": f"{paper_id}_dont",
                "row_type": "anti_pattern",
                "anti_pattern": assessment["dont"],
            }
        )
        rows.append(
            {
                **common,
                "row_id": f"{paper_id}_risk",
                "row_type": "risk_prior",
                "risk_rule": assessment["warning"],
            }
        )
        if str(assessment["hypothesis"]).strip():
            rows.append(
                {
                    **common,
                    "row_id": f"{paper_id}_hypothesis",
                    "row_type": "feature_idea",
                    "feature_name": f"paper_hypothesis:{paper_id}",
                    "entry_logic": assessment["hypothesis"],
                }
            )
    return normalize_extraction_rows(rows)


def _summary_markdown(frames: dict[str, pd.DataFrame], inventory: pd.DataFrame) -> str:
    summary = frames["paper_review_summary"]
    guidance = frames["paper_guidance"]
    hypotheses = frames["paper_hypothesis_queue"]
    lines = [
        "# Wizard Paper Library Review",
        "",
        f"- papers ingested: {len(inventory)}",
        f"- papers adversarially reviewed: {len(summary)}",
        f"- gap analyses: {len(frames['paper_gap_analysis'])}",
        f"- premortems: {len(frames['paper_premortem'])}",
        f"- red-team reviews: {len(frames['paper_red_team'])}",
        f"- guidance findings: {len(guidance)}",
        f"- proposed hypotheses: {len(hypotheses)}",
        f"- critical open gaps: {int(summary['gap_severity'].eq('critical').sum())}",
        "- execution-eligible findings: 0",
        "",
        "## Evidence Verdicts",
        "",
    ]
    for verdict, count in summary["evidence_verdict"].value_counts().sort_index().items():
        lines.append(f"- {verdict}: {count}")
    lines.extend(["", "## Guidance", ""])
    for category, count in guidance["category"].value_counts().sort_index().items():
        lines.append(f"- {category}: {count}")
    lines.extend(
        [
            "",
            "## Current Gate",
            "",
            (
                "The source library is preserved and its initial adversarial screen is complete. "
                "No paper has passed independent mathematical reproduction, source-authenticity "
                "verification, local point-in-time replication, costed walk-forward testing, or "
                "execution validation. Every finding remains research-only."
            ),
        ]
    )
    return "\n".join(lines) + "\n"
