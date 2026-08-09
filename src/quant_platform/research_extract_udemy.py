from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.research_extraction import EXTRACTION_COLUMNS, normalize_extraction_rows
from quant_platform.research_ingestion import ingest_research_source, mark_research_sources_processed
from quant_platform.research_knowledge_store import build_research_knowledge_store


UDEMY_SOURCE_ID = "udemy_shaun_mcdonogh_review"
LEDGER_COLUMNS = (
    "course",
    "section",
    "video_title",
    "review_status",
    "evidence_source",
    "transcript_word_count",
    "transcript_cue_count",
    "topic_tags",
    "lecture_url",
)
REQUIRED_LEDGER_COLUMNS = set(LEDGER_COLUMNS)
ALLOWED_TRANSCRIPT_METADATA_COLUMNS = {"transcript_word_count", "transcript_cue_count"}
PROMOTION_AUTHORITY = "none_research_only"


KNOWLEDGE_SEEDS: tuple[dict[str, str], ...] = (
    {"row_type": "feature_idea", "feature_name": "integration_order", "strategy_family": "cointegration", "notes": "Verify integration order before interpreting cointegration evidence."},
    {"row_type": "feature_idea", "feature_name": "rolling_adf", "strategy_family": "cointegration", "notes": "Estimate ADF evidence on rolling point-in-time windows."},
    {"row_type": "feature_idea", "feature_name": "residual_stationarity", "strategy_family": "cointegration", "notes": "Test fitted hedge residuals rather than treating correlation as stationarity."},
    {"row_type": "feature_idea", "feature_name": "ecm_strength", "strategy_family": "cointegration", "notes": "Use ECM strength as adjustment-speed evidence, not an automatic signal."},
    {"row_type": "feature_idea", "feature_name": "granger_diagnostics", "strategy_family": "cointegration", "notes": "Track directional diagnostics without treating Granger results as trade authorization."},
    {"row_type": "feature_idea", "feature_name": "symmetric_asymmetric_garch", "strategy_family": "regime", "notes": "Compare symmetric and asymmetric volatility estimates for sizing and crisis filters."},
    {"row_type": "feature_idea", "feature_name": "empirical_tail_diagnostics", "strategy_family": "copula", "notes": "Record skew, kurtosis, empirical CDF, Student-t fit, Q-Q, and mixture-density evidence."},
    {"row_type": "strategy_hint", "strategy_family": "copula", "entry_logic": "Use point-in-time marginal transforms and conditional-probability dislocation.", "exit_logic": "Exit on conditional-probability convergence or a deterministic risk exit.", "notes": "Keep copula dislocation independent from generic z-score modes."},
    {"row_type": "mode_preference", "strategy_family": "copula", "mode_preference": "copula_dislocation", "notes": "Do not translate copula dislocation into a generic z-score proxy."},
    {"row_type": "strategy_hint", "strategy_family": "mean_reversion", "exit_logic": "Test half-life timeout alongside mean-cross and partial convergence exits.", "notes": "Half-life is both a candidate feature and a time-exit reference."},
    {"row_type": "regime_condition", "strategy_family": "regime", "regime_condition": "HMM state labels must remain stable across rolling refits.", "notes": "HMM is discovery, filtering, or sizing evidence until walk-forward proof."},
    {"row_type": "feature_idea", "feature_name": "kmeans_pair_cluster", "strategy_family": "discovery", "notes": "Use K-means to cluster behaviorally similar assets before pair testing."},
    {"row_type": "feature_idea", "feature_name": "pca_residual_factor", "strategy_family": "discovery", "notes": "Use chronological preprocessing and train/test separation for PCA features."},
    {"row_type": "strategy_hint", "strategy_family": "trade_gate", "entry_logic": "Use supervised models to score a strategy-valid candidate, not create one.", "notes": "Evaluate XGBoost, random forest, and explainable baselines as trade-quality gates."},
    {"row_type": "risk_prior", "strategy_family": "trade_gate", "risk_rule": "Require purged chronological folds, costed labels, take-rate floors, concentration tests, and regime attribution.", "notes": "Economic validation outranks accuracy-only metrics."},
    {"row_type": "anti_pattern", "strategy_family": "trade_gate", "anti_pattern": "Model confidence must not create a trade that failed strategy or safety gates.", "notes": "The model may only skip or resize an otherwise valid proposal."},
    {"row_type": "feature_idea", "feature_name": "rl_after_cost_reward", "strategy_family": "reinforcement_learning", "notes": "Reward after-cost portfolio outcomes with drawdown, turnover, and concentration penalties."},
    {"row_type": "risk_prior", "strategy_family": "reinforcement_learning", "risk_rule": "Penalize stale data and invalid actions; deterministic controls retain final authority.", "notes": "RL proposes, ranks, resizes, or recommends experiments only."},
    {"row_type": "execution_warning", "strategy_family": "execution", "execution_warning": "Reconcile intended, submitted, open, filled, and expected position state for both legs.", "notes": "Execution state must be persisted on every cycle."},
    {"row_type": "risk_prior", "strategy_family": "execution", "risk_rule": "Qualify both legs for liquidity before submitting either leg.", "notes": "A liquid first leg does not make an illiquid second leg executable."},
    {"row_type": "execution_warning", "strategy_family": "execution", "execution_warning": "Keep close-all, leg-repair, timeout, crisis-exit, and kill-switch paths deterministic.", "notes": "Venue adapters must not bypass orchestration safety."},
    {"row_type": "strategy_hint", "strategy_family": "exit_design", "exit_logic": "Compare mean-cross, partial convergence, half-life timeout, volatility shock, hedge break, liquidity deterioration, and crisis hard exit.", "notes": "Exit styles require separate costed walk-forward attribution."},
    {"row_type": "risk_prior", "strategy_family": "acceptance", "risk_rule": "Dashboard and educational examples are hypothesis sources only; require local point-in-time Hyperliquid replay and testnet evidence.", "notes": "Educational evidence has no promotion authority."},
)


HYPOTHESIS_SEEDS: tuple[dict[str, str], ...] = (
    {"hypothesis_family": "econometrics_features", "strategy_family": "cointegration", "proposed_change": "Add rolling integration-order, ADF, residual-stationarity, ECM, Granger, and GARCH features.", "required_test": "Ablate each feature in purged walk-forward folds and report pair/regime concentration.", "acceptance_gate": "Incremental after-cost improvement without unstable fold dependence."},
    {"hypothesis_family": "distribution_tail_features", "strategy_family": "copula", "proposed_change": "Add empirical distribution and tail diagnostics to pair detail and copula reports.", "required_test": "Verify marginal calibration and tail diagnostics out of sample.", "acceptance_gate": "Calibrated conditional probabilities and stable costed performance."},
    {"hypothesis_family": "unsupervised_discovery", "strategy_family": "discovery", "proposed_change": "Use HMM, K-means, and PCA only in discovery or feature-generation lanes.", "required_test": "Compare rolling refit stability and downstream incremental value against deterministic baselines.", "acceptance_gate": "Stable labels/features and positive out-of-sample contribution."},
    {"hypothesis_family": "trade_gate_validation", "strategy_family": "trade_gate", "proposed_change": "Harden trade-gate evaluation with chronological purging, costed labels, take-rate floors, and attribution.", "required_test": "Compare raw, rule-filtered, model-gated, and model-sized strategies out of sample.", "acceptance_gate": "Better after-cost drawdown/expectancy without take-rate collapse."},
    {"hypothesis_family": "rl_reward_guardrails", "strategy_family": "reinforcement_learning", "proposed_change": "Redesign RL rewards around after-cost portfolio outcomes and invalid-action penalties.", "required_test": "Run chronological shadow episodes with deterministic venue and risk controls held fixed.", "acceptance_gate": "Improved shadow outcomes without increased invalid actions or concentration."},
    {"hypothesis_family": "order_reconciliation", "strategy_family": "execution", "proposed_change": "Add a two-leg order-reconciliation node across intent, submission, open orders, fills, and positions.", "required_test": "Inject partial fill, stale order, reject, timeout, and leg mismatch failures on testnet.", "acceptance_gate": "Every injected mismatch resolves or fails closed with an auditable reason."},
    {"hypothesis_family": "exit_matrix", "strategy_family": "exit_design", "proposed_change": "Expand exact-mode exit testing across convergence, time, volatility, hedge, liquidity, and crisis exits.", "required_test": "Run costed walk-forward exit ablations by pair, mode, timeframe, and regime.", "acceptance_gate": "Exit improvement survives multiple pairs and regimes after costs."},
    {"hypothesis_family": "source_authority", "strategy_family": "acceptance", "proposed_change": "Keep dashboard and course evidence in hypothesis generation only.", "required_test": "Audit every promoted setup for point-in-time Hyperliquid replay and testnet evidence.", "acceptance_gate": "Zero promotions sourced only from educational or dashboard evidence."},
)


def udemy_paths(root: Path = ROOT) -> dict[str, Path]:
    processed = root / "data" / "processed" / "research_knowledge"
    reports = root / "reports" / "research"
    agents = root / "reports" / "agents"
    return {
        "coverage_source": reports / "shaun_mcdonogh_udemy_video_review.csv",
        "review_source": reports / "shaun_mcdonogh_udemy_implementation_review.md",
        "lecture_index": processed / "udemy_lecture_index.csv",
        "udemy_research_rows": processed / "udemy_research_rows.csv",
        "udemy_research_features": processed / "udemy_research_features.csv",
        "udemy_research_rules": processed / "udemy_research_rules.csv",
        "udemy_strategy_hints": processed / "udemy_strategy_hints.csv",
        "course_coverage": reports / "udemy_course_coverage.csv",
        "ingestion_audit": reports / "udemy_ingestion_audit.csv",
        "ingestion_audit_md": reports / "udemy_ingestion_audit.md",
        "hypotheses": agents / "udemy_research_hypotheses.csv",
        "agent_context": agents / "udemy_research_context.csv",
    }


def refresh_udemy_research(
    *,
    root: Path = ROOT,
    coverage_path: Path | None = None,
    review_path: Path | None = None,
) -> CommandResult:
    paths = udemy_paths(root)
    coverage = coverage_path or paths["coverage_source"]
    review = review_path or paths["review_source"]
    if not coverage.exists() or not review.exists():
        missing = [str(path) for path in (coverage, review) if not path.exists()]
        raise FileNotFoundError(f"missing_udemy_research_artifacts:{';'.join(missing)}")
    registry = ingest_research_source(
        source_type="udemy",
        title="Shaun McDonogh course implementation review",
        source_path_or_url=str(review),
        root=root,
        author="Shaun McDonogh",
        channel_or_publisher="Udemy",
        topic_tags="pairs_trading;econometrics;machine_learning;reinforcement_learning;execution",
        review_status="reviewed_synthesis",
        confidence=0.65,
        notes=f"coverage_ledger={coverage};research_only;no_transcript_text_stored",
        source_id=UDEMY_SOURCE_ID,
    )
    extraction = extract_udemy_research(root=root, coverage_path=coverage, review_path=review)
    knowledge = build_research_knowledge_store(root=root)
    return CommandResult(
        paths={**registry.paths, **extraction.paths, **knowledge.paths},
        summary={
            **extraction.summary,
            "udemy_knowledge_rows": int(extraction.summary.get("knowledge_rows", 0)),
            "combined_knowledge_rows": int(knowledge.summary.get("rows", 0)),
        },
    )


def extract_udemy_research(
    *,
    root: Path = ROOT,
    coverage_path: Path | None = None,
    review_path: Path | None = None,
) -> CommandResult:
    paths = udemy_paths(root)
    coverage = coverage_path or paths["coverage_source"]
    review = review_path or paths["review_source"]
    if not coverage.exists() or not review.exists():
        return _write_blocked_outputs(paths, coverage=coverage, review=review)

    ledger = pd.read_csv(coverage).fillna("")
    missing_columns = sorted(REQUIRED_LEDGER_COLUMNS - set(ledger.columns))
    forbidden_columns = sorted(
        column
        for column in ledger.columns
        if column not in ALLOWED_TRANSCRIPT_METADATA_COLUMNS
        and any(token in column.casefold() for token in ("transcript", "caption_text", "full_text", "lecture_text"))
    )
    if missing_columns:
        raise ValueError(f"udemy_ledger_missing_columns:{';'.join(missing_columns)}")
    if forbidden_columns:
        raise ValueError(f"udemy_ledger_contains_transcript_content:{';'.join(forbidden_columns)}")
    review_text = review.read_text(encoding="utf-8")
    if "## Quantitative Findings" not in review_text or "## Changes Recommended For TheWiz" not in review_text:
        raise ValueError("udemy_review_missing_required_sections")

    lecture_index = _lecture_index(ledger, coverage)
    research_rows = _research_rows(review)
    course_coverage = _course_coverage(lecture_index)
    hypotheses = _hypotheses(review)
    context = _agent_context(hypotheses)
    audit = _audit(lecture_index, coverage=coverage, review=review)

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    lecture_index.to_csv(paths["lecture_index"], index=False)
    research_rows.to_csv(paths["udemy_research_rows"], index=False)
    research_rows[research_rows["row_type"] == "feature_idea"].to_csv(paths["udemy_research_features"], index=False)
    research_rows[research_rows["row_type"].isin(["risk_prior", "regime_condition", "anti_pattern", "execution_warning"])].to_csv(paths["udemy_research_rules"], index=False)
    research_rows[research_rows["row_type"].isin(["strategy_hint", "mode_preference"])].to_csv(paths["udemy_strategy_hints"], index=False)
    course_coverage.to_csv(paths["course_coverage"], index=False)
    audit.to_csv(paths["ingestion_audit"], index=False)
    paths["ingestion_audit_md"].write_text(_audit_markdown(audit, course_coverage), encoding="utf-8")
    hypotheses.to_csv(paths["hypotheses"], index=False)
    context.to_csv(paths["agent_context"], index=False)
    mark_research_sources_processed("udemy", root=root)
    return CommandResult(
        paths={key: value for key, value in paths.items() if key not in {"coverage_source", "review_source"}},
        summary={
            "lectures": int(len(lecture_index)),
            "direct_transcript_reviews": int(lecture_index["evidence_source"].eq("udemy_transcript").sum()),
            "assistant_fallback_reviews": int(lecture_index["evidence_source"].eq("udemy_ai_assistant").sum()),
            "knowledge_rows": int(len(research_rows)),
            "hypotheses": int(len(hypotheses)),
            "trade_authorized": False,
        },
    )


def search_udemy_research(query: str, *, root: Path = ROOT, limit: int = 25) -> pd.DataFrame:
    paths = udemy_paths(root)
    lectures = _read_csv(paths["lecture_index"])
    knowledge = _read_csv(paths["udemy_research_rows"])
    rows: list[dict[str, object]] = []
    for _, row in lectures.iterrows():
        rows.append(
            {
                "record_type": "lecture",
                "record_id": row.get("lecture_id", ""),
                "title": row.get("video_title", ""),
                "strategy_family": "",
                "topic_tags": row.get("topic_tags", ""),
                "evidence_source": row.get("evidence_source", ""),
                "evidence_path": row.get("evidence_path", ""),
                "source_url": row.get("lecture_url", ""),
                "live_signal_eligible": False,
            }
        )
    for _, row in knowledge.iterrows():
        title = row.get("feature_name") or row.get("strategy_family") or row.get("row_type")
        rows.append(
            {
                "record_type": "knowledge",
                "record_id": row.get("row_id", ""),
                "title": title,
                "strategy_family": row.get("strategy_family", ""),
                "topic_tags": row.get("notes", ""),
                "evidence_source": row.get("evidence_source", ""),
                "evidence_path": row.get("evidence_path", ""),
                "source_url": row.get("source_url", ""),
                "live_signal_eligible": False,
            }
        )
    frame = pd.DataFrame(rows)
    terms = [token.casefold() for token in str(query).split() if token.strip()]
    if terms and not frame.empty:
        haystack = frame.astype(str).agg(" ".join, axis=1).str.casefold()
        frame = frame[haystack.map(lambda value: all(term in value for term in terms))]
    return frame.head(max(0, int(limit))).reset_index(drop=True)


def _lecture_index(frame: pd.DataFrame, evidence_path: Path) -> pd.DataFrame:
    result = frame[list(LEDGER_COLUMNS)].copy()
    result["lecture_id"] = result.apply(
        lambda row: "udl_" + sha256(f"{row['course']}|{row['section']}|{row['video_title']}|{row['lecture_url']}".encode()).hexdigest()[:16],
        axis=1,
    )
    result["source_type"] = "udemy"
    result["evidence_quality"] = result["evidence_source"].map(
        {"udemy_transcript": "direct_transcript_review", "udemy_ai_assistant": "assistant_fallback_review"}
    ).fillna("unknown_review_method")
    result["transcript_text_stored"] = False
    result["point_in_time_status"] = "reference_only_not_market_data"
    result["live_signal_eligible"] = False
    result["promotion_authority"] = PROMOTION_AUTHORITY
    result["evidence_path"] = str(evidence_path)
    return result


def _research_rows(review_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(KNOWLEDGE_SEEDS, start=1):
        rows.append(
            {
                **seed,
                "row_id": f"{UDEMY_SOURCE_ID}_{index:04d}",
                "confidence": 0.65,
                "source_id": UDEMY_SOURCE_ID,
                "source_type": "udemy",
                "source_title": "Shaun McDonogh course implementation review",
                "evidence_path": str(review_path),
                "evidence_source": "reviewed_course_synthesis",
                "point_in_time_status": "reference_only_not_market_data",
                "live_signal_eligible": False,
                "promotion_authority": PROMOTION_AUTHORITY,
                "source_url": "",
                "review_status": "reviewed_synthesis",
            }
        )
    return normalize_extraction_rows(rows)


def _course_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["course", "videos", "direct_transcript", "udemy_ai_fallback", "blocked"])
    rows = []
    for course, group in frame.groupby("course", sort=True):
        rows.append(
            {
                "course": course,
                "videos": int(len(group)),
                "direct_transcript": int(group["evidence_source"].eq("udemy_transcript").sum()),
                "udemy_ai_fallback": int(group["evidence_source"].eq("udemy_ai_assistant").sum()),
                "blocked": int((~group["review_status"].astype(str).str.casefold().eq("reviewed")).sum()),
            }
        )
    return pd.DataFrame(rows)


def _hypotheses(review_path: Path) -> pd.DataFrame:
    rows = []
    for index, seed in enumerate(HYPOTHESIS_SEEDS, start=1):
        rows.append(
            {
                "hypothesis_id": f"udemy_hypothesis_{index:03d}",
                **seed,
                "status": "UNVERIFIED",
                "trade_authorized": False,
                "live_signal_eligible": False,
                "promotion_authority": PROMOTION_AUTHORITY,
                "source_type": "udemy",
                "evidence_source": "reviewed_course_synthesis",
                "evidence_path": str(review_path),
                "next_step": "implement_in_research_branch_then_run_costed_walk_forward_ablation",
            }
        )
    return pd.DataFrame(rows)


def _agent_context(hypotheses: pd.DataFrame) -> pd.DataFrame:
    if hypotheses.empty:
        return pd.DataFrame(columns=["context_id", "context_type", "strategy_family", "summary", "required_test", "guardrail", "evidence_path"])
    return pd.DataFrame(
        {
            "context_id": hypotheses["hypothesis_id"],
            "context_type": "research_hypothesis",
            "strategy_family": hypotheses["strategy_family"],
            "summary": hypotheses["proposed_change"],
            "required_test": hypotheses["required_test"],
            "guardrail": "research_only_no_trade_authority",
            "evidence_path": hypotheses["evidence_path"],
        }
    )


def _audit(frame: pd.DataFrame, *, coverage: Path, review: Path) -> pd.DataFrame:
    reviewed = frame["review_status"].astype(str).str.casefold().eq("reviewed") if not frame.empty else pd.Series(dtype=bool)
    return pd.DataFrame(
        [
            {"metric": "ingestion_status", "value": "ready", "detail": "research_only"},
            {"metric": "lectures_total", "value": int(len(frame)), "detail": ""},
            {"metric": "lectures_reviewed", "value": int(reviewed.sum()), "detail": ""},
            {"metric": "direct_transcript_reviews", "value": int(frame["evidence_source"].eq("udemy_transcript").sum()), "detail": "coverage metadata only"},
            {"metric": "assistant_fallback_reviews", "value": int(frame["evidence_source"].eq("udemy_ai_assistant").sum()), "detail": "preserved as lower-quality provenance"},
            {"metric": "transcript_text_rows_stored", "value": int(frame["transcript_text_stored"].map(bool).sum()), "detail": "must remain zero"},
            {"metric": "live_signal_eligible_rows", "value": int(frame["live_signal_eligible"].map(bool).sum()), "detail": "must remain zero"},
            {"metric": "coverage_sha256", "value": _hash_file(coverage), "detail": str(coverage)},
            {"metric": "review_sha256", "value": _hash_file(review), "detail": str(review)},
        ]
    )


def _write_blocked_outputs(paths: dict[str, Path], *, coverage: Path, review: Path) -> CommandResult:
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_csv(paths["lecture_index"], index=False)
    pd.DataFrame(columns=EXTRACTION_COLUMNS).to_csv(paths["udemy_research_rows"], index=False)
    for key in ("udemy_research_features", "udemy_research_rules", "udemy_strategy_hints"):
        pd.DataFrame(columns=EXTRACTION_COLUMNS).to_csv(paths[key], index=False)
    pd.DataFrame(columns=["course", "videos", "direct_transcript", "udemy_ai_fallback", "blocked"]).to_csv(paths["course_coverage"], index=False)
    audit = pd.DataFrame([{"metric": "ingestion_status", "value": "blocked_missing_source", "detail": f"coverage={coverage};review={review}"}])
    audit.to_csv(paths["ingestion_audit"], index=False)
    paths["ingestion_audit_md"].write_text(_audit_markdown(audit, pd.DataFrame()), encoding="utf-8")
    pd.DataFrame(columns=["hypothesis_id"]).to_csv(paths["hypotheses"], index=False)
    pd.DataFrame(columns=["context_id"]).to_csv(paths["agent_context"], index=False)
    return CommandResult(paths={key: value for key, value in paths.items() if key not in {"coverage_source", "review_source"}}, summary={"lectures": 0, "knowledge_rows": 0, "hypotheses": 0, "blocker": "missing_udemy_research_artifacts"})


def _audit_markdown(audit: pd.DataFrame, course_coverage: pd.DataFrame) -> str:
    lines = ["# Udemy Research Ingestion Audit", ""]
    for _, row in audit.iterrows():
        detail = f" ({row['detail']})" if str(row.get("detail", "")).strip() else ""
        lines.append(f"- {row['metric']}: {row['value']}{detail}")
    lines.extend(["", "## Course Coverage", ""])
    if course_coverage.empty:
        lines.append("- no course coverage available")
    else:
        for _, row in course_coverage.iterrows():
            lines.append(f"- {row['course']}: {row['videos']} videos, {row['direct_transcript']} direct transcript, {row['udemy_ai_fallback']} assistant fallback")
    lines.extend(["", "Educational evidence is research-only and cannot authorize or promote a trade.", ""])
    return "\n".join(lines)


def _hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return pd.DataFrame()
