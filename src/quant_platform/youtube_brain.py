from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd
import yaml

from quant_platform.active_pipeline import ROOT, CommandResult
from quant_platform.orchestration.corrective_runtime import (
    atomic_append_text,
    atomic_write_csv,
    atomic_write_parquet,
    atomic_write_text,
)
from quant_platform.research_extract_youtube import extract_youtube_research
from quant_platform.research_knowledge_store import build_research_knowledge_store
from quant_platform.youtube_caption_insights import build_youtube_caption_insights

YOUTUBE_BRAIN_AGENT = "youtube_research_brain"
YOUTUBE_BRAIN_SCHEMA_VERSION = "youtube_brain.v1"
DEFAULT_CHANNEL_URL = "https://www.youtube.com/@CryptoWizards"
DEFAULT_REFRESH_HOURS = 24

MODE_FORMULAS: dict[str, tuple[str, ...]] = {
    "static_spread": ("spread_static", "engle_granger", "adf_stationarity", "beta"),
    "static_zscorer": ("spread_static", "rolling_zscore", "zero_crossings", "engle_granger"),
    "dynamic_spread": ("dynamic_spread_kalman", "beta", "engle_granger", "half_life"),
    "dynamic_zscorer": ("dynamic_spread_kalman", "rolling_zscore", "zero_crossings", "half_life"),
    "ou_spread": ("ou_process", "half_life", "hurst", "adf_stationarity"),
    "ou_zscorer": ("ou_process", "rolling_zscore", "half_life", "hurst"),
    "copula": ("copula_core", "conditional_copula", "kendall_spearman_pearson", "var_cvar"),
}


def youtube_brain_paths(root: Path = ROOT) -> dict[str, Path]:
    external = root / "data" / "external" / "youtube" / "brain"
    processed = root / "data" / "processed" / "youtube_brain"
    agents = root / "reports" / "agents"
    dashboard = root / "reports" / "dashboard"
    return {
        "external": external,
        "snapshots": external / "snapshots",
        "subtitles": external / "subtitles",
        "video_registry": external / "video_registry.csv",
        "caption_manifest": external / "caption_manifest.csv",
        "collection_status": agents / "youtube_brain_collection_status.csv",
        "claims": processed / "claims.csv",
        "claims_parquet": processed / "claims.parquet",
        "external_research_priors": processed / "external_research_priors.csv",
        "formulas": processed / "formulas.csv",
        "formulas_parquet": processed / "formulas.parquet",
        "strategy_memory": processed / "strategy_memory.csv",
        "warning_memory": processed / "warning_memory.csv",
        "caption_insight_candidates": processed / "caption_insight_candidates.csv",
        "knowledge_nodes": processed / "knowledge_nodes.csv",
        "knowledge_edges": processed / "knowledge_edges.csv",
        "hypotheses": agents / "youtube_brain_hypotheses.csv",
        "hypotheses_jsonl": agents / "youtube_brain_hypotheses.jsonl",
        "replication_scorecard": agents / "youtube_brain_replication_scorecard.csv",
        "outcome_memory": processed / "outcome_memory.csv",
        "memory_jsonl": root / "data" / "agent_memory" / "youtube_research_brain.jsonl",
        "brain_status": agents / "youtube_brain_status.csv",
        "recommendations": agents / "youtube_brain_recommendations.csv",
        "recommendations_markdown": agents / "youtube_brain_recommendations.md",
        "dashboard_status": dashboard / "youtube_brain_status.csv",
        "dashboard_hypotheses": dashboard / "youtube_brain_hypotheses.csv",
        "dashboard_replication": dashboard / "youtube_brain_replication_scorecard.csv",
        "dashboard_recommendations": dashboard / "youtube_brain_recommendations.csv",
        "dashboard_recommendations_markdown": dashboard / "youtube_brain_recommendations.md",
        "dashboard_summary": dashboard / "youtube_brain.md",
    }


def refresh_youtube_collection(
    *,
    root: Path = ROOT,
    channel_url: str | None = None,
    fetch_live: bool = False,
    force: bool = False,
    fetch_captions: bool | None = None,
    refresh_hours: int | None = None,
    max_new_caption_fetches: int | None = None,
    caption_languages: tuple[str, ...] | None = None,
    catalog_path: Path | None = None,
) -> CommandResult:
    config = _brain_config(root)
    channel_url = channel_url or _text(config.get("channel_url")) or DEFAULT_CHANNEL_URL
    fetch_captions = bool(config.get("fetch_captions", True)) if fetch_captions is None else fetch_captions
    refresh_hours = int(refresh_hours or config.get("refresh_hours") or DEFAULT_REFRESH_HOURS)
    max_new_caption_fetches = int(
        max_new_caption_fetches or config.get("max_new_caption_fetches_per_cycle") or 50
    )
    configured_languages = config.get("caption_languages", ["en", "en-orig"])
    caption_languages = caption_languages or tuple(str(value) for value in configured_languages)
    paths = youtube_brain_paths(root)
    for key in ("external", "snapshots", "subtitles"):
        paths[key].mkdir(parents=True, exist_ok=True)
    now = _now()
    previous = _read_csv(paths["video_registry"])
    previous_status = _read_csv(paths["collection_status"])
    due = _collection_due(previous_status, now=now, refresh_hours=refresh_hours)
    live_status = "not_requested"
    live_error = ""
    selected_catalog = catalog_path or _latest_local_catalog(root)

    if fetch_live and (force or due):
        try:
            selected_catalog = _fetch_live_catalog(channel_url, paths["snapshots"])
            live_status = "live_catalog_refreshed"
        except (RuntimeError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            live_status = "live_fetch_failed_fallback_local"
            live_error = f"{type(exc).__name__}:{exc}"
    elif fetch_live:
        live_status = "not_due_reused_registry"

    payload = _read_json(selected_catalog) if selected_catalog and selected_catalog.exists() else {}
    current = _normalize_catalog(payload, source_path=selected_catalog, captured_at=now)
    if live_status == "live_catalog_refreshed" and (
        current.empty or (not previous.empty and len(current) < max(1, int(len(previous) * 0.5)))
    ):
        live_status = "live_catalog_invalid_fallback_registry"
        live_error = _join_nonempty(
            live_error,
            f"catalog_sanity_check_failed:previous={len(previous)}:current={len(current)}",
        )
        current = previous.copy()
    if current.empty and not previous.empty:
        current = previous.copy()
    registry = _merge_registry(previous, current, now=now)

    new_ids = registry.loc[registry["is_new"].map(_boolish), "video_id"].astype(str).tolist() if not registry.empty else []
    if fetch_live and fetch_captions and new_ids and live_status == "live_catalog_refreshed":
        try:
            _fetch_captions(
                new_ids,
                paths["subtitles"],
                languages=caption_languages,
                limit=max_new_caption_fetches,
            )
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            live_error = _join_nonempty(live_error, f"caption_fetch:{type(exc).__name__}:{exc}")

    caption_manifest = _build_caption_manifest(root, paths["subtitles"])
    if not registry.empty:
        caption_counts = caption_manifest.groupby("video_id").size().to_dict() if not caption_manifest.empty else {}
        registry["caption_file_count"] = registry["video_id"].map(caption_counts).fillna(0).astype(int)
        registry["caption_status"] = registry["caption_file_count"].map(lambda count: "available" if count else "missing")
    _write_csv(registry, paths["video_registry"])
    _write_csv(caption_manifest, paths["caption_manifest"])

    last_live_refresh_at = _latest_text(previous_status, "last_live_refresh_at")
    if live_status == "live_catalog_refreshed":
        last_live_refresh_at = now
    next_refresh_due_at = _next_live_refresh_due_at(
        last_live_refresh_at,
        now=now,
        refresh_hours=refresh_hours,
    )

    status = pd.DataFrame(
        [
            {
                "checked_at": now,
                "channel_url": channel_url,
                "fetch_live_requested": fetch_live,
                "live_status": live_status,
                "live_error": live_error,
                "catalog_path": str(selected_catalog or ""),
                "videos": len(registry),
                "new_videos": int(registry.get("is_new", pd.Series(dtype=bool)).map(_boolish).sum()),
                "changed_videos": int(registry.get("metadata_changed", pd.Series(dtype=bool)).map(_boolish).sum()),
                "videos_with_captions": int(registry.get("caption_status", pd.Series(dtype=str)).astype(str).eq("available").sum()),
                "caption_files": len(caption_manifest),
                "refresh_hours": refresh_hours,
                "last_live_refresh_at": last_live_refresh_at,
                "next_refresh_due_at": next_refresh_due_at,
                "collector_ready": _yt_dlp_available(),
                "promotion_authority": "none_research_only",
            }
        ]
    )
    _write_csv(status, paths["collection_status"])
    return CommandResult(
        paths={
            "video_registry": paths["video_registry"],
            "caption_manifest": paths["caption_manifest"],
            "collection_status": paths["collection_status"],
        },
        summary={"videos": len(registry), "new_videos": len(new_ids), "live_status": live_status},
    )


def build_youtube_brain(*, root: Path = ROOT) -> CommandResult:
    paths = youtube_brain_paths(root)
    extraction = extract_youtube_research(root=root)
    build_research_knowledge_store(root=root)
    rows = _read_csv(extraction.paths["youtube_research_rows"])
    rows = _dedupe_research_rows(rows)
    formulas = _load_formula_catalog(root)
    video_registry = _read_csv(paths["video_registry"])

    native_claims = _claim_rows(rows)
    external_claims = _external_research_prior_claims(root)
    claims = pd.concat([native_claims, external_claims], ignore_index=True, sort=False)
    if not claims.empty:
        claims = claims.drop_duplicates(subset=["claim_id"], keep="last").reset_index(drop=True)
    strategy_memory = _strategy_rows(claims)
    warning_memory = _warning_rows(claims)
    nodes, edges = _knowledge_graph(claims, formulas, video_registry)
    insights = build_youtube_caption_insights(root=root)

    _write_csv_and_parquet(claims, paths["claims"], paths["claims_parquet"])
    _write_csv(external_claims, paths["external_research_priors"])
    _write_csv_and_parquet(formulas, paths["formulas"], paths["formulas_parquet"])
    _write_csv(strategy_memory, paths["strategy_memory"])
    _write_csv(warning_memory, paths["warning_memory"])
    _write_csv(nodes, paths["knowledge_nodes"])
    _write_csv(edges, paths["knowledge_edges"])

    status = pd.DataFrame(
        [
            {
                "built_at": _now(),
                "videos": len(video_registry),
                "claims": len(claims),
                "native_channel_claims": len(native_claims),
                "external_research_priors": len(external_claims),
                "hudson_thames_must_test_priors": int(
                    external_claims.get("priority", pd.Series(dtype=str)).astype(str).eq("must_test").sum()
                ),
                "formulas": len(formulas),
                "strategy_memories": len(strategy_memory),
                "warning_memories": len(warning_memory),
                "caption_insight_candidates": int(insights.summary.get("candidate_windows", 0)),
                "caption_recommendations": int(insights.summary.get("recommendations", 0)),
                "knowledge_nodes": len(nodes),
                "knowledge_edges": len(edges),
                "duplicate_claim_ids": int(claims.get("claim_id", pd.Series(dtype=str)).duplicated().sum()),
                "claims_without_video_url": int(claims.get("video_url", pd.Series(dtype=str)).astype(str).eq("").sum()),
                "promotion_authority": "none_research_only",
                "status": "ready" if not claims.empty else "blocked_no_claims",
            }
        ]
    )
    _write_csv(status, paths["brain_status"])
    return CommandResult(
        paths={
            "claims": paths["claims"],
            "external_research_priors": paths["external_research_priors"],
            "formulas": paths["formulas"],
            "strategy_memory": paths["strategy_memory"],
            "warning_memory": paths["warning_memory"],
            "caption_insight_candidates": paths["caption_insight_candidates"],
            "recommendations": paths["recommendations"],
            "recommendations_markdown": paths["recommendations_markdown"],
            "knowledge_nodes": paths["knowledge_nodes"],
            "knowledge_edges": paths["knowledge_edges"],
            "brain_status": paths["brain_status"],
        },
        summary={
            "claims": len(claims),
            "external_research_priors": len(external_claims),
            "formulas": len(formulas),
            "nodes": len(nodes),
            "edges": len(edges),
            "recommendations": int(insights.summary.get("recommendations", 0)),
        },
    )


def build_youtube_pair_hypotheses(
    *,
    root: Path = ROOT,
    candidate_path: Path | None = None,
    max_pairs: int = 100,
) -> CommandResult:
    paths = youtube_brain_paths(root)
    candidates = _load_wizard_candidates(root, candidate_path)
    claims = _read_csv(paths["claims"])
    formulas = _read_csv(paths["formulas"])
    warnings = _read_csv(paths["warning_memory"])
    collection = _read_csv(paths["collection_status"])
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()

    for _, candidate in candidates.head(max_pairs).iterrows():
        pair = _pair(candidate)
        exact_mode = _text(candidate.get("exact_mode") or candidate.get("dashboard_recommended_strategy"))
        timeframe = _text(candidate.get("timeframe") or candidate.get("interval"))
        key = (pair, exact_mode, timeframe)
        if not pair or key in seen:
            continue
        seen.add(key)
        mode_key = _mode_key(exact_mode)
        formula_ids = MODE_FORMULAS.get(mode_key, ())
        formula_subset = formulas[formulas.get("formula_id", pd.Series(dtype=str)).astype(str).isin(formula_ids)].copy()
        relevant_claims = _relevant_claims(claims, mode_key=mode_key, formula_ids=formula_ids)
        relevant_warnings = _relevant_warnings(warnings, mode_key=mode_key)
        source_confidence = _mean_numeric(relevant_claims.get("confidence", pd.Series(dtype=float)))
        formula_coverage = len(formula_subset) / len(formula_ids) if formula_ids else 0.0
        direct_link_coverage = (
            relevant_claims.get("video_url", pd.Series(dtype=str)).astype(str).ne("").mean()
            if not relevant_claims.empty
            else 0.0
        )
        freshness = _collection_freshness(collection)
        priority = round(0.35 * source_confidence + 0.30 * formula_coverage + 0.20 * direct_link_coverage + 0.15 * freshness, 4)
        sharpe = _number(candidate.get("sharpe"))
        returns_total = _number(candidate.get("returns_total_pct"), fallback=_number(candidate.get("returns_total")) * 100)
        discovery_pass = sharpe >= 1.75 and returns_total >= 10.0
        hypothesis_id = _stable_id("yth", pair, exact_mode, timeframe, ";".join(formula_ids))
        rows.append(
            {
                "schema_version": YOUTUBE_BRAIN_SCHEMA_VERSION,
                "hypothesis_id": hypothesis_id,
                "pair": pair,
                "asset_x": _text(candidate.get("asset_x")),
                "asset_y": _text(candidate.get("asset_y")),
                "wizard_exchange": _text(candidate.get("exchange")),
                "timeframe": timeframe,
                "exact_mode": exact_mode,
                "mode_key": mode_key,
                "wizard_sharpe": sharpe,
                "wizard_returns_total_pct": returns_total,
                "wizard_discovery_pass": discovery_pass,
                "formula_ids": ";".join(formula_ids),
                "formula_names": ";".join(formula_subset.get("formula_name", pd.Series(dtype=str)).astype(str).tolist()),
                "claim_ids": ";".join(relevant_claims.get("claim_id", pd.Series(dtype=str)).astype(str).head(10).tolist()),
                "video_urls": ";".join(_unique(relevant_claims.get("video_url", pd.Series(dtype=str)).astype(str).tolist())[:10]),
                "warning_ids": ";".join(relevant_warnings.get("claim_id", pd.Series(dtype=str)).astype(str).head(8).tolist()),
                "source_confidence": round(source_confidence, 4),
                "formula_coverage": round(formula_coverage, 4),
                "direct_link_coverage": round(float(direct_link_coverage), 4),
                "source_freshness_score": round(freshness, 4),
                "research_priority_score": priority,
                "entry_hypothesis": _entry_hypothesis(candidate, exact_mode),
                "exit_hypothesis": _exit_hypothesis(candidate, exact_mode),
                "test_plan": _test_plan(exact_mode),
                "lifecycle": "UNVERIFIED",
                "trade_authorized": False,
                "promotion_authority": "none_research_only",
                "blocker": _hypothesis_blocker(discovery_pass, relevant_claims, formula_subset),
                "next_step": "run_exact_mode_costed_walk_forward_and_regime_tests",
                "created_at": _now(),
                "evidence_path": _join_nonempty(
                    _text(candidate.get("evidence_path") or candidate.get("source_path")),
                    str(paths["claims"]),
                    str(paths["formulas"]),
                ),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["wizard_discovery_pass", "research_priority_score", "wizard_sharpe"], ascending=[False, False, False])
    _write_csv(frame, paths["hypotheses"])
    _write_jsonl(frame, paths["hypotheses_jsonl"])
    return CommandResult(
        paths={"hypotheses": paths["hypotheses"], "hypotheses_jsonl": paths["hypotheses_jsonl"]},
        summary={"hypotheses": len(frame), "discovery_passes": int(frame.get("wizard_discovery_pass", pd.Series(dtype=bool)).map(_boolish).sum())},
    )


def refresh_youtube_outcome_memory(*, root: Path = ROOT) -> CommandResult:
    paths = youtube_brain_paths(root)
    hypotheses = _read_csv(paths["hypotheses"])
    walk = _read_csv(root / "reports" / "active" / "workflow_walk_forward_ranked.csv")
    validation_path = root / "reports" / "active" / "youtube_hypothesis_validation_queue.csv"
    validation = _read_csv(validation_path)
    paper = _read_csv(root / "reports" / "paper_trading_journal.csv")
    score_rows: list[dict[str, object]] = []
    memory_events: list[dict[str, object]] = []

    for _, hypothesis in hypotheses.iterrows():
        pair = _text(hypothesis.get("pair"))
        mode = _text(hypothesis.get("exact_mode"))
        matched_walk = _match_walk_forward(walk, pair=pair, mode=mode)
        matched_validation = _match_walk_forward(validation, pair=pair, mode=mode)
        matched_paper = _match_closed_outcomes(paper, pair=pair)
        lifecycle, reason = _replication_lifecycle(matched_walk, matched_paper)
        validation_status = _first(matched_validation, "validation_status", "")
        validation_blocker = _first(matched_validation, "blocker", "")
        if matched_walk.empty and matched_paper.empty and not matched_validation.empty:
            if matched_validation.get("costed_walk_forward_ready", pd.Series(False, index=matched_validation.index)).map(_boolish).any():
                lifecycle = "READY_FOR_WALK_FORWARD"
                reason = "exact_mode_and_venue_prerequisites_ready_for_walk_forward"
            else:
                lifecycle = "VALIDATION_BLOCKED"
                reason = _first(matched_validation, "acceptance_reason", "validation_prerequisites_incomplete")
        best_pf = _max_numeric(matched_walk.get("test_profit_factor", pd.Series(dtype=float)))
        best_sharpe = _max_numeric(matched_walk.get("test_sharpe", pd.Series(dtype=float)))
        max_trades = int(_max_numeric(matched_walk.get("test_trades", pd.Series(dtype=float))))
        oos_passes = int(matched_walk.get("oos_metric_pass", pd.Series(dtype=bool)).map(_boolish).sum()) if not matched_walk.empty else 0
        accepted = int(matched_walk.get("acceptance", pd.Series(dtype=bool)).map(_boolish).sum()) if not matched_walk.empty else 0
        score_rows.append(
            {
                "hypothesis_id": hypothesis.get("hypothesis_id", ""),
                "pair": pair,
                "exact_mode": mode,
                "timeframe": hypothesis.get("timeframe", ""),
                "lifecycle": lifecycle,
                "replication_reason": reason,
                "validation_rows": len(matched_validation),
                "validation_status": validation_status,
                "validation_blocker": validation_blocker,
                "history_ready": _first(matched_validation, "history_ready", False),
                "funding_ready": _first(matched_validation, "funding_ready", False),
                "cost_model_ready": _first(matched_validation, "cost_model_ready", False),
                "slippage_model_ready": _first(matched_validation, "slippage_model_ready", False),
                "hyperliquid_testnet_pair_tradable": _first(
                    matched_validation, "hyperliquid_testnet_pair_tradable", False
                ),
                "walk_forward_rows": len(matched_walk),
                "oos_metric_passes": oos_passes,
                "accepted_rows": accepted,
                "best_test_profit_factor": best_pf,
                "best_test_sharpe": best_sharpe,
                "max_test_trades": max_trades,
                "verified_paper_outcomes": len(matched_paper),
                "trade_authorized": False,
                "promotion_authority": "none_research_only",
                "evaluated_at": _now(),
                "evidence_path": _join_nonempty(
                    str(root / "reports" / "active" / "workflow_walk_forward_ranked.csv"),
                    str(validation_path),
                    str(root / "reports" / "paper_trading_journal.csv"),
                ),
            }
        )
        memory_events.append(
            {
                "schema_version": YOUTUBE_BRAIN_SCHEMA_VERSION,
                "event_id": _stable_id(
                    "ytm",
                    hypothesis.get("hypothesis_id", ""),
                    lifecycle,
                    reason,
                    best_pf,
                    best_sharpe,
                    max_trades,
                    accepted,
                    len(matched_paper),
                    validation_status,
                    validation_blocker,
                ),
                "timestamp": _now(),
                "agent": YOUTUBE_BRAIN_AGENT,
                "hypothesis_id": hypothesis.get("hypothesis_id", ""),
                "pair": pair,
                "exact_mode": mode,
                "outcome_known": bool(len(matched_walk) or len(matched_paper)),
                "outcome_label": lifecycle if len(matched_walk) or len(matched_paper) else "",
                "learning_note": reason,
                "validation_status": validation_status,
                "validation_blocker": validation_blocker,
                "best_test_profit_factor": best_pf,
                "best_test_sharpe": best_sharpe,
                "max_test_trades": max_trades,
                "accepted_rows": accepted,
                "verified_hyperliquid_testnet_outcomes": len(matched_paper),
                "evidence_path": score_rows[-1]["evidence_path"],
            }
        )

    scorecard = pd.DataFrame(score_rows)
    _write_csv(scorecard, paths["replication_scorecard"])
    _write_csv(scorecard, paths["outcome_memory"])
    _append_unique_jsonl(memory_events, paths["memory_jsonl"])
    return CommandResult(
        paths={
            "replication_scorecard": paths["replication_scorecard"],
            "outcome_memory": paths["outcome_memory"],
            "memory_jsonl": paths["memory_jsonl"],
        },
        summary={
            "rows": len(scorecard),
            "outcomes_known": int(
                (
                    scorecard.get("walk_forward_rows", pd.Series(0, index=scorecard.index)).gt(0)
                    | scorecard.get("verified_paper_outcomes", pd.Series(0, index=scorecard.index)).gt(0)
                ).sum()
            ),
        },
    )


def build_youtube_brain_dashboard(*, root: Path = ROOT) -> CommandResult:
    paths = youtube_brain_paths(root)
    collection = _read_csv(paths["collection_status"])
    brain = _read_csv(paths["brain_status"])
    hypotheses = _read_csv(paths["hypotheses"])
    replication = _read_csv(paths["replication_scorecard"])
    recommendations = _read_csv(paths["recommendations"])
    caption_insights = _read_csv(paths["caption_insight_candidates"])
    validation = _read_csv(root / "reports" / "active" / "youtube_hypothesis_validation_queue.csv")
    status = pd.DataFrame(
        [
            {
                "generated_at": _now(),
                "collection_status": _first(collection, "live_status", "missing"),
                "collector_ready": _first(collection, "collector_ready", False),
                "videos": _first(collection, "videos", 0),
                "videos_with_captions": _first(collection, "videos_with_captions", 0),
                "claims": _first(brain, "claims", 0),
                "formulas": _first(brain, "formulas", 0),
                "caption_insight_candidates": len(caption_insights),
                "caption_recommendations": len(recommendations),
                "hypotheses": len(hypotheses),
                "current_wizard_passes": int(hypotheses.get("wizard_discovery_pass", pd.Series(dtype=bool)).map(_boolish).sum()),
                "walk_forward_supported": int(replication.get("lifecycle", pd.Series(dtype=str)).astype(str).eq("WALK_FORWARD_SUPPORTED").sum()),
                "testnet_supported": int(replication.get("lifecycle", pd.Series(dtype=str)).astype(str).eq("TESTNET_SUPPORTED").sum()),
                "validation_blocked": int(
                    validation.get("costed_walk_forward_ready", pd.Series(False, index=validation.index)).map(_boolish).eq(False).sum()
                ),
                "hyperliquid_testnet_compatible": int(
                    validation.get("hyperliquid_testnet_pair_tradable", pd.Series(False, index=validation.index)).map(_boolish).sum()
                ),
                "slippage_calibrated": int(
                    validation.get("slippage_model_ready", pd.Series(False, index=validation.index)).map(_boolish).sum()
                ),
                "trade_authorized": False,
                "promotion_authority": "none_research_only",
                "status": "research_ready" if not brain.empty else "blocked_brain_not_built",
            }
        ]
    )
    _write_csv(status, paths["dashboard_status"])
    _write_csv(hypotheses, paths["dashboard_hypotheses"])
    _write_csv(replication, paths["dashboard_replication"])
    _write_csv(recommendations, paths["dashboard_recommendations"])
    paths["dashboard_summary"].parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(paths["dashboard_summary"], _dashboard_markdown(status, hypotheses, replication, recommendations), encoding="utf-8")
    return CommandResult(
        paths={
            "youtube_brain_status": paths["dashboard_status"],
            "youtube_brain_hypotheses": paths["dashboard_hypotheses"],
            "youtube_brain_replication": paths["dashboard_replication"],
            "youtube_brain_recommendations": paths["dashboard_recommendations"],
            "youtube_brain_summary": paths["dashboard_summary"],
        },
        summary={"hypotheses": len(hypotheses), "replication_rows": len(replication)},
    )


def run_youtube_brain_cycle(
    *,
    root: Path = ROOT,
    fetch_live: bool | None = None,
    force: bool = False,
    candidate_path: Path | None = None,
) -> CommandResult:
    if fetch_live is None:
        fetch_live = _brain_config(root).get("auto_refresh_enabled", False)
    collection = refresh_youtube_collection(root=root, fetch_live=fetch_live, force=force)
    brain = build_youtube_brain(root=root)
    hypotheses = build_youtube_pair_hypotheses(root=root, candidate_path=candidate_path)
    outcomes = refresh_youtube_outcome_memory(root=root)
    dashboard = build_youtube_brain_dashboard(root=root)
    return CommandResult(
        paths={**collection.paths, **brain.paths, **hypotheses.paths, **outcomes.paths, **dashboard.paths},
        summary={
            "videos": int(collection.summary.get("videos", 0)),
            "claims": int(brain.summary.get("claims", 0)),
            "recommendations": int(brain.summary.get("recommendations", 0)),
            "hypotheses": int(hypotheses.summary.get("hypotheses", 0)),
            "replication_rows": int(outcomes.summary.get("rows", 0)),
            "promotion_authority": "none_research_only",
        },
    )


def _latest_local_catalog(root: Path) -> Path | None:
    candidates = list((root / "data" / "external" / "youtube").glob("**/channel_videos*.json"))
    candidates.extend((root / "data" / "external" / "youtube").glob("*priority_videos.json"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _brain_config(root: Path) -> dict[str, object]:
    path = root / "config" / "youtube_brain.yaml"
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _fetch_live_catalog(channel_url: str, snapshot_dir: Path) -> Path:
    if not _yt_dlp_available():
        raise RuntimeError("yt_dlp_not_installed_install_project_youtube_extra")
    videos_url = channel_url.rstrip("/")
    if not videos_url.endswith("/videos"):
        videos_url = f"{videos_url}/videos"
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--flat-playlist", "--dump-single-json", "--no-warnings", videos_url],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    payload = json.loads(result.stdout)
    output = snapshot_dir / f"channel_videos_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    atomic_write_text(output, json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def _fetch_captions(
    video_ids: list[str],
    output_dir: Path,
    *,
    languages: tuple[str, ...] = ("en", "en-orig"),
    limit: int = 50,
) -> None:
    if not _yt_dlp_available():
        raise RuntimeError("yt_dlp_not_installed_install_project_youtube_extra")
    output_dir.mkdir(parents=True, exist_ok=True)
    urls = [f"https://www.youtube.com/watch?v={video_id}" for video_id in video_ids[:limit]]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs",
            ",".join(languages),
            "--sub-format",
            "vtt",
            "--ignore-errors",
            "--no-warnings",
            "-o",
            str(output_dir / "%(id)s.%(ext)s"),
            *urls,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )


def _normalize_catalog(payload: object, *, source_path: Path | None, captured_at: str) -> pd.DataFrame:
    entries = payload.get("entries", []) if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    channel = _text(payload.get("channel") if isinstance(payload, dict) else "")
    channel_id = _text(payload.get("channel_id") if isinstance(payload, dict) else "")
    rows: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if _text(entry.get("_type")).lower() in {"playlist", "multi_video"}:
            continue
        video_id = _text(entry.get("id"))
        if not _valid_youtube_video_id(video_id):
            continue
        title = _text(entry.get("title"))
        url = _text(entry.get("webpage_url")) or f"https://www.youtube.com/watch?v={video_id}"
        published_at = _timestamp_iso(entry.get("timestamp") or entry.get("release_timestamp"))
        metadata = {
            "video_id": video_id,
            "title": title,
            "url": url,
            "duration_seconds": _number(entry.get("duration")),
            "published_at": published_at,
            "channel": _text(entry.get("channel")) or channel,
            "channel_id": _text(entry.get("channel_id")) or channel_id,
            "description": _text(entry.get("description")),
        }
        rows.append(
            {
                **metadata,
                "metadata_hash": _stable_id("meta", json.dumps(metadata, sort_keys=True)),
                "source_snapshot": str(source_path or ""),
                "captured_at": captured_at,
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["video_id"], keep="last") if rows else pd.DataFrame()


def _merge_registry(previous: pd.DataFrame, current: pd.DataFrame, *, now: str) -> pd.DataFrame:
    if current.empty:
        return previous
    previous_by_id = (
        {
            str(row["video_id"]): row
            for _, row in previous.iterrows()
            if _valid_youtube_video_id(_text(row.get("video_id")))
        }
        if not previous.empty
        else {}
    )
    rows: list[dict[str, object]] = []
    current_ids: set[str] = set()
    for _, row in current.iterrows():
        record = row.to_dict()
        video_id = str(record.get("video_id", ""))
        current_ids.add(video_id)
        old = previous_by_id.get(video_id)
        record["first_seen_at"] = _text(old.get("first_seen_at")) if old is not None else now
        record["last_seen_at"] = now
        record["is_new"] = old is None
        record["metadata_changed"] = bool(old is not None and _text(old.get("metadata_hash")) != _text(record.get("metadata_hash")))
        rows.append(record)
    for video_id, old in previous_by_id.items():
        if video_id in current_ids:
            continue
        record = old.to_dict()
        record["is_new"] = False
        record["metadata_changed"] = False
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["is_new", "title"], ascending=[False, True]).reset_index(drop=True)


def _valid_youtube_video_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{11}", value))


def _build_caption_manifest(root: Path, brain_subtitles: Path) -> pd.DataFrame:
    directories = [brain_subtitles, root / "data" / "external" / "youtube" / "cryptowizards_full_scrub" / "subtitles"]
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.vtt")):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            parts = path.name.split(".")
            rows.append(
                {
                    "video_id": parts[0],
                    "language": parts[1] if len(parts) > 2 else "",
                    "variant": ".".join(parts[1:-1]),
                    "path": str(path),
                    "sha256": _file_hash(path),
                    "bytes": path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                }
            )
    return pd.DataFrame(rows)


def _claim_rows(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        claim_text = _claim_text(row)
        video_url = _video_url(row)
        if not claim_text or not video_url:
            continue
        output.append(
            {
                "schema_version": YOUTUBE_BRAIN_SCHEMA_VERSION,
                "claim_id": _stable_id("ytc", row.get("source_id", ""), row.get("row_type", ""), claim_text, video_url),
                "row_type": row.get("row_type", ""),
                "claim_text": claim_text,
                "strategy_family": row.get("strategy_family", ""),
                "mode_preference": row.get("mode_preference", ""),
                "feature_name": row.get("feature_name", ""),
                "timeframe_preference": row.get("timeframe_preference", ""),
                "regime_condition": row.get("regime_condition", ""),
                "confidence": _number(row.get("confidence")),
                "source_id": row.get("source_id", ""),
                "source_title": row.get("source_title", ""),
                "video_url": video_url,
                "evidence_path": row.get("evidence_path", ""),
                "review_status": row.get("review_status", ""),
                "lifecycle": "UNVERIFIED",
                "promotion_authority": "none_research_only",
            }
        )
    frame = pd.DataFrame(output)
    return frame.drop_duplicates(subset=["claim_id"], keep="last").reset_index(drop=True) if not frame.empty else frame


def _external_research_prior_claims(root: Path) -> pd.DataFrame:
    """Normalize reviewed external video research into non-authoritative priors."""
    source = root / "reports" / "research" / "hudson_thames_youtube_recommendations.csv"
    recommendations = _read_csv(source)
    output: list[dict[str, object]] = []
    for _, row in recommendations.iterrows():
        recommendation = _text(row.get("recommendation"))
        source_urls = _urls(_text(row.get("source_videos")))
        if not recommendation or not source_urls:
            continue
        theme = _text(row.get("theme")).lower()
        row_type = "risk_prior" if theme in {"validation", "risk", "execution"} else "strategy_hint"
        source_id = _text(row.get("recommendation_id"))
        output.append({
            "schema_version": YOUTUBE_BRAIN_SCHEMA_VERSION,
            "claim_id": _stable_id("ytx", source_id, recommendation, source_urls[0]),
            "row_type": row_type,
            "claim_text": recommendation,
            "strategy_family": _text(row.get("strategy_families")),
            "mode_preference": _text(row.get("strategy_families")),
            "feature_name": theme,
            "timeframe_preference": "",
            "regime_condition": "",
            "confidence": _number(row.get("confidence")),
            "source_id": source_id,
            "source_title": _text(row.get("theme_label")),
            "source_channel": "Hudson & Thames",
            "video_url": source_urls[0],
            "source_urls": ";".join(source_urls),
            "source_video_count": int(_number(row.get("source_video_count"))),
            "priority": _text(row.get("priority")),
            "interesting_suggestion": _text(row.get("interesting_suggestion")),
            "local_validation_test": _text(row.get("local_validation_test")),
            "evidence_basis": _text(row.get("evidence_basis")),
            "evidence_path": str(source),
            "review_status": _text(row.get("review_status")),
            "lifecycle": "UNVERIFIED",
            "promotion_authority": "none_research_only",
            "trade_authorized": False,
        })
    return pd.DataFrame(output)


def _strategy_rows(claims: pd.DataFrame) -> pd.DataFrame:
    if claims.empty:
        return claims.copy()
    mask = claims["row_type"].astype(str).isin({"strategy_hint", "mode_preference"})
    return claims.loc[mask].copy().reset_index(drop=True)


def _warning_rows(claims: pd.DataFrame) -> pd.DataFrame:
    if claims.empty:
        return claims.copy()
    mask = claims["row_type"].astype(str).isin({"risk_prior", "anti_pattern", "execution_warning"})
    return claims.loc[mask].copy().reset_index(drop=True)


def _knowledge_graph(claims: pd.DataFrame, formulas: pd.DataFrame, videos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    video_node_ids: set[str] = set()
    for _, video in videos.iterrows():
        video_node_id = f"video:{video.get('video_id', '')}"
        video_node_ids.add(video_node_id)
        nodes.append({"node_id": video_node_id, "node_type": "video", "label": video.get("title", ""), "attributes": video.get("url", "")})
    for _, formula in formulas.iterrows():
        formula_id = _text(formula.get("formula_id"))
        nodes.append({"node_id": f"formula:{formula_id}", "node_type": "formula", "label": formula.get("formula_name", formula_id), "attributes": formula.get("math", "")})
        for url in _urls(_text(formula.get("top_source_videos"))):
            video_id = _video_id_from_url(url)
            if video_id:
                edges.append({"edge_id": _stable_id("yte", formula_id, video_id), "source_node": f"formula:{formula_id}", "target_node": f"video:{video_id}", "relation": "supported_by"})
    for _, claim in claims.iterrows():
        claim_id = _text(claim.get("claim_id"))
        nodes.append({"node_id": f"claim:{claim_id}", "node_type": "claim", "label": claim.get("claim_text", ""), "attributes": claim.get("row_type", "")})
        claim_video_urls = _urls(_text(claim.get("source_urls"))) or [_text(claim.get("video_url"))]
        for claim_video_url in claim_video_urls:
            video_id = _video_id_from_url(claim_video_url)
            if not video_id:
                continue
            video_node_id = f"video:{video_id}"
            if video_node_id not in video_node_ids:
                video_node_ids.add(video_node_id)
                nodes.append({
                    "node_id": video_node_id,
                    "node_type": "video",
                    "label": claim.get("source_title", "external research video"),
                    "attributes": claim_video_url,
                })
            edges.append({"edge_id": _stable_id("yte", claim_id, video_id), "source_node": f"claim:{claim_id}", "target_node": f"video:{video_id}", "relation": "extracted_from"})
        for formula_id in _formula_matches(claim, formulas):
            edges.append({"edge_id": _stable_id("yte", claim_id, formula_id), "source_node": f"claim:{claim_id}", "target_node": f"formula:{formula_id}", "relation": "references"})
    node_frame = pd.DataFrame(nodes).drop_duplicates(subset=["node_id"], keep="last") if nodes else pd.DataFrame()
    edge_frame = pd.DataFrame(edges).drop_duplicates(subset=["edge_id"], keep="last") if edges else pd.DataFrame()
    return node_frame, edge_frame


def _load_formula_catalog(root: Path) -> pd.DataFrame:
    source = root / "reports" / "research" / "cryptowizards_formula_catalog_2026-07-01.csv"
    frame = _read_csv(source)
    if frame.empty:
        return pd.DataFrame(columns=["formula_id", "formula_name", "category", "math", "video_count", "top_source_videos", "repo_status"])
    frame = frame.drop_duplicates(subset=["formula_id"], keep="last").copy()
    frame["schema_version"] = YOUTUBE_BRAIN_SCHEMA_VERSION
    frame["source_path"] = str(source)
    frame["promotion_authority"] = "none_reference_only"
    return frame


def _formula_matches(claim: pd.Series, formulas: pd.DataFrame) -> list[str]:
    text = " ".join(_text(claim.get(column)) for column in ["claim_text", "strategy_family", "mode_preference", "feature_name"]).lower()
    matches: list[str] = []
    for _, formula in formulas.iterrows():
        formula_id = _text(formula.get("formula_id"))
        tokens = [token for token in re.split(r"[_\s/-]+", formula_id.lower()) if len(token) >= 3]
        if any(re.search(rf"\b{re.escape(token)}\b", text) for token in tokens):
            matches.append(formula_id)
    return matches


def _mode_key(value: object) -> str:
    text = _text(value).lower()
    if "copula" in text:
        return "copula"
    suffix = "zscorer" if "zscore" in text else "spread"
    if re.search(r"\b(?:ou|ornstein)", text):
        return f"ou_{suffix}"
    if "dyn" in text or "dynamic" in text or "kalman" in text:
        return f"dynamic_{suffix}"
    if "static" in text:
        return f"static_{suffix}"
    return "unmapped"


def _relevant_claims(claims: pd.DataFrame, *, mode_key: str, formula_ids: tuple[str, ...]) -> pd.DataFrame:
    if claims.empty:
        return claims
    keywords = set(mode_key.split("_"))
    for formula_id in formula_ids:
        keywords.update(token for token in formula_id.split("_") if len(token) >= 3)
    text = claims[[column for column in ["claim_text", "strategy_family", "mode_preference", "feature_name"] if column in claims]].astype(str).agg(" ".join, axis=1).str.lower()
    mask = pd.Series(False, index=claims.index)
    for keyword in sorted(keywords):
        mask |= text.str.contains(rf"\b{re.escape(keyword)}\b", regex=True, na=False)
    return claims.loc[mask].sort_values("confidence", ascending=False).reset_index(drop=True)


def _relevant_warnings(warnings: pd.DataFrame, *, mode_key: str) -> pd.DataFrame:
    if warnings.empty:
        return warnings
    text = warnings.get("claim_text", pd.Series("", index=warnings.index)).astype(str).str.lower()
    mode_terms = [term for term in mode_key.split("_") if term not in {"spread", "zscorer"}]
    mask = text.str.contains("risk|cost|slippage|drawdown|execution|forward|paper", regex=True, na=False)
    for term in mode_terms:
        mask |= text.str.contains(rf"\b{re.escape(term)}\b", regex=True, na=False)
    return warnings.loc[mask].sort_values("confidence", ascending=False).reset_index(drop=True)


def _load_wizard_candidates(root: Path, explicit: Path | None) -> pd.DataFrame:
    candidates = [
        explicit,
        root / "reports" / "active" / "crypto_wizards_live_scanner_capture.csv",
        root / "reports" / "active" / "wizard_discovery_current.csv",
    ]
    for path in candidates:
        if path and path.exists():
            frame = _read_csv(path)
            if not frame.empty:
                return frame
    return pd.DataFrame()


def _match_walk_forward(frame: pd.DataFrame, *, pair: str, mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    pair_mask = frame.get("pair", pd.Series("", index=frame.index)).astype(str).map(_canonical_pair).eq(_canonical_pair(pair))
    mode_series = frame.get("wizard_exact_mode", frame.get("exact_mode", pd.Series("", index=frame.index))).astype(str).map(_mode_key)
    return frame[pair_mask & mode_series.eq(_mode_key(mode))].copy()


def _match_closed_outcomes(frame: pd.DataFrame, *, pair: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    pair_mask = frame.get("pair", pd.Series("", index=frame.index)).astype(str).map(_canonical_pair).eq(_canonical_pair(pair))
    lifecycle = frame.get("lifecycle_status", pd.Series("", index=frame.index)).astype(str).str.lower()
    label = frame.get("outcome_label", pd.Series("", index=frame.index)).astype(str).str.strip()
    realized = pd.to_numeric(frame.get("realized_return", pd.Series(float("nan"), index=frame.index)), errors="coerce")
    venue = frame.get("execution_venue", frame.get("venue", pd.Series("", index=frame.index))).astype(str).str.lower()
    network = frame.get("network", frame.get("environment", pd.Series("", index=frame.index))).astype(str).str.lower()
    verified = lifecycle.isin({"closed", "settled", "completed"}) & (label.ne("") | realized.notna())
    hyperliquid_testnet = venue.str.contains(r"\bhyperliquid\b", regex=True, na=False) & network.str.contains(
        r"\btestnet\b", regex=True, na=False
    )
    return frame[pair_mask & verified & hyperliquid_testnet].copy()


def _replication_lifecycle(walk: pd.DataFrame, paper: pd.DataFrame) -> tuple[str, str]:
    if not paper.empty:
        profitable = pd.to_numeric(paper.get("realized_return", pd.Series(dtype=float)), errors="coerce").fillna(0).gt(0).any()
        return ("TESTNET_SUPPORTED", "verified_testnet_outcome_positive") if profitable else ("REJECTED", "verified_testnet_outcome_not_positive")
    if walk.empty:
        return "UNVERIFIED", "no_matching_local_walk_forward_evidence"
    accepted = walk.get("acceptance", pd.Series(False, index=walk.index)).map(_boolish).any()
    if accepted:
        return "WALK_FORWARD_SUPPORTED", "local_acceptance_rows_present"
    oos = walk.get("oos_metric_pass", pd.Series(False, index=walk.index)).map(_boolish).any()
    return ("REPLICATING", "proxy_oos_metrics_pass_but_acceptance_blocked") if oos else ("REJECTED", "matching_walk_forward_rows_failed_oos_metrics")


def _entry_hypothesis(candidate: pd.Series, exact_mode: str) -> str:
    level = candidate.get("entry_level", "")
    if pd.notna(level) and _text(level):
        return f"wizard_documented_or_captured_entry_level={level};confirm_direction_mapping"
    return f"capture_exact_{_mode_key(exact_mode)}_entry_threshold_and_direction"


def _exit_hypothesis(candidate: pd.Series, exact_mode: str) -> str:
    level = candidate.get("exit_level", "")
    if pd.notna(level) and _text(level):
        return f"wizard_documented_or_captured_exit_level={level};test_entry_only_and_hard_exit_regime_overlays"
    return f"capture_exact_{_mode_key(exact_mode)}_exit_threshold;test_entry_only_and_hard_exit"


def _test_plan(exact_mode: str) -> str:
    return _join_nonempty(
        f"exact_mode={_mode_key(exact_mode)}",
        "regimes=all,range_only,exclude_crisis,calm_vol_only,range_low_tail,stable_hedge",
        "regime_behaviors=entry_only,hard_exit",
        "validation=costed_walk_forward,hyperliquid_funding,hyperliquid_l2_slippage",
    )


def _hypothesis_blocker(discovery_pass: bool, claims: pd.DataFrame, formulas: pd.DataFrame) -> str:
    blockers = []
    if not discovery_pass:
        blockers.append("wizard_discovery_gate_not_passed")
    if claims.empty:
        blockers.append("youtube_claim_support_missing")
    if formulas.empty:
        blockers.append("youtube_formula_mapping_missing")
    blockers.append("youtube_research_has_no_promotion_authority")
    return ";".join(blockers)


def _claim_text(row: pd.Series) -> str:
    fields = {
        "strategy_hint": "entry_logic",
        "feature_idea": "feature_name",
        "risk_prior": "risk_rule",
        "mode_preference": "mode_preference",
        "regime_condition": "regime_condition",
        "pair_filter": "pair_filter",
        "anti_pattern": "anti_pattern",
        "execution_warning": "execution_warning",
    }
    return _text(row.get(fields.get(_text(row.get("row_type")), "")))


def _video_url(row: pd.Series) -> str:
    match = re.search(r"https?://(?:www\.)?youtube\.com/watch\?v=[^;\s]+", _text(row.get("notes")), flags=re.IGNORECASE)
    return match.group(0) if match else ""


def _dashboard_markdown(
    status: pd.DataFrame,
    hypotheses: pd.DataFrame,
    replication: pd.DataFrame,
    recommendations: pd.DataFrame | None = None,
) -> str:
    row = status.iloc[0].to_dict() if not status.empty else {}
    lines = [
        "# YouTube Research Brain",
        "",
        f"- status: {row.get('status', 'missing')}",
        f"- videos: {row.get('videos', 0)}",
        f"- videos with captions: {row.get('videos_with_captions', 0)}",
        f"- claims: {row.get('claims', 0)}",
        f"- formulas: {row.get('formulas', 0)}",
        f"- caption insight candidates: {row.get('caption_insight_candidates', 0)}",
        f"- caption recommendations: {row.get('caption_recommendations', 0)}",
        f"- current hypotheses: {len(hypotheses)}",
        f"- walk-forward supported: {row.get('walk_forward_supported', 0)}",
        f"- Testnet supported: {row.get('testnet_supported', 0)}",
        "- promotion authority: none; research prioritization only",
        "",
        "## Lifecycle",
        "",
    ]
    if replication.empty:
        lines.append("- no replication outcomes yet")
    else:
        for lifecycle, count in replication["lifecycle"].value_counts().items():
            lines.append(f"- {lifecycle}: {count}")
    lines.extend(["", "## Research Recommendations", ""])
    if recommendations is None or recommendations.empty:
        lines.append("- no caption-derived recommendations yet")
    else:
        ranked = recommendations.copy()
        ranked["priority_rank"] = ranked.get("priority", pd.Series("useful", index=ranked.index)).map(
            {"must_test": 0, "useful": 1}
        ).fillna(2)
        ranked = ranked.sort_values(["priority_rank", "priority_score"], ascending=[True, False])
        for item in ranked.head(8).itertuples(index=False):
            lines.append(f"- **{item.theme_label}:** {item.interesting_suggestion}")
    lines.extend(
        [
            "",
            "Caption recommendations are machine-assisted inferences that require human review and local validation.",
            "YouTube claims cannot authorize Testnet or live execution.\n",
        ]
    )
    return "\n".join(lines)


def _collection_due(status: pd.DataFrame, *, now: str, refresh_hours: int = DEFAULT_REFRESH_HOURS) -> bool:
    if status.empty or "last_live_refresh_at" not in status:
        return True
    last = pd.to_datetime(status["last_live_refresh_at"], utc=True, errors="coerce").max()
    current = pd.Timestamp(now)
    return pd.isna(last) or current - last >= pd.Timedelta(hours=refresh_hours)


def _next_live_refresh_due_at(last_live_refresh_at: str, *, now: str, refresh_hours: int) -> str:
    last = pd.to_datetime(last_live_refresh_at, utc=True, errors="coerce")
    if pd.isna(last):
        return now
    return (last + pd.Timedelta(hours=refresh_hours)).isoformat()


def _latest_text(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return ""
    values = frame[column].astype(str).str.strip()
    values = values[values.ne("")]
    return values.iloc[-1] if not values.empty else ""


def _collection_freshness(collection: pd.DataFrame) -> float:
    if collection.empty:
        return 0.0
    checked = pd.to_datetime(collection.get("checked_at", pd.Series(dtype=str)), utc=True, errors="coerce").max()
    if pd.isna(checked):
        return 0.0
    age = pd.Timestamp.now(tz="UTC") - checked
    if age <= pd.Timedelta(hours=48):
        return 1.0
    if age <= pd.Timedelta(days=14):
        return 0.5
    return 0.1


def _yt_dlp_available() -> bool:
    return importlib.util.find_spec("yt_dlp") is not None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return pd.DataFrame()


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(frame, path, index=False)
    return path


def _write_csv_and_parquet(frame: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    _write_csv(frame, csv_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(frame, parquet_path, index=False)


def _write_jsonl(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(
        json.dumps(row, sort_keys=True, default=str) + "\n"
        for row in frame.to_dict("records")
    )
    atomic_write_text(path, encoded, encoding="utf-8")
    return path


def _append_unique_jsonl(events: list[dict[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                existing.add(_text(json.loads(line).get("event_id")))
            except (json.JSONDecodeError, AttributeError):
                continue
    appended: list[str] = []
    for event in events:
        event_id = _text(event.get("event_id"))
        if not event_id or event_id in existing:
            continue
        appended.append(json.dumps(event, sort_keys=True, default=str) + "\n")
        existing.add(event_id)
    if appended:
        atomic_append_text(path, "".join(appended), encoding="utf-8")
    return path


def _dedupe_research_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    populated = frame.get("row_id", pd.Series("", index=frame.index)).astype(str).str.strip().ne("")
    return pd.concat(
        [frame[populated].drop_duplicates(subset=["row_id"], keep="last"), frame[~populated].drop_duplicates(keep="last")],
        ignore_index=True,
    )


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(_text(part) for part in parts)
    return f"{prefix}_{sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _pair(row: pd.Series) -> str:
    pair = _text(row.get("pair"))
    if pair:
        return _canonical_pair(pair)
    left, right = _text(row.get("asset_x")), _text(row.get("asset_y"))
    return _canonical_pair(f"{left}/{right}") if left and right else ""


def _canonical_pair(value: object) -> str:
    text = _text(value).upper().replace("_", "-")
    if "/" in text:
        return text
    marker = "-USD-"
    if marker in text:
        left, right = text.split(marker, 1)
        return f"{left}-USD/{right}"
    return text


def _timestamp_iso(value: object) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
        except (ValueError, OSError, OverflowError):
            return ""
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    return "" if pd.isna(timestamp) else timestamp.isoformat()


def _video_id_from_url(url: str) -> str:
    match = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", url)
    return match.group(1) if match else ""


def _urls(value: str) -> list[str]:
    return re.findall(r"https?://[^\s|)]+", value)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _join_nonempty(*values: object) -> str:
    return ";".join(_unique([_text(value) for value in values]))


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _number(value: object, *, fallback: float = 0.0) -> float:
    try:
        numeric = float(value)
        return numeric if pd.notna(numeric) else fallback
    except (TypeError, ValueError, OverflowError):
        return fallback


def _mean_numeric(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else 0.0


def _max_numeric(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else 0.0


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y", "pass", "accepted"}


def _first(frame: pd.DataFrame, column: str, default: object) -> object:
    return frame.iloc[0].get(column, default) if not frame.empty else default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
