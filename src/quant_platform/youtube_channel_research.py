from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.youtube_caption_insights import (
    build_youtube_theme_recommendations,
    mine_youtube_caption_insights,
)


HUDSON_THAMES_KEY = "hudson_thames"
HUDSON_THAMES_CHANNEL_URL = "https://www.youtube.com/@hudsonthamesresearch"
CHANNEL_RESEARCH_SCHEMA_VERSION = "youtube_channel_research.v1"

TITLE_TOPICS: dict[str, tuple[str, ...]] = {
    "pairs_trading": ("pairs trading", "pair trading", "arbitrage", "cointegration", "ou model", "copula"),
    "portfolio_risk": ("portfolio", "position sizing", "risk", "allocation", "optimization"),
    "machine_learning": ("machine learning", "meta-label", "feature importance", "model", "cross validation", "sample weights"),
    "reinforcement_learning": ("reinforcement learning",),
    "market_structure": ("microstructure", "order book", "market impact", "execution"),
    "data_validation": ("backtest", "cross validation", "sampling", "label concurrency", "structural breaks"),
}


@dataclass(frozen=True)
class CuratedIdea:
    key: str
    theme: str
    label: str
    recommendation: str
    interesting_suggestion: str
    local_test: str
    strategy_families: str
    sources: tuple[tuple[str, str, int], ...]
    evidence_basis: str


HUDSON_THAMES_CURATED_IDEAS: tuple[CuratedIdea, ...] = (
    CuratedIdea(
        "copula_logic_matrix",
        "copula",
        "Copula entry and exit logic matrix",
        "Treat AND/OR entry and exit rules as explicit strategy parameters rather than implementation details.",
        "Prioritize AND-open/OR-exit as a robustness challenger, while preserving the full four-combination matrix for local replay.",
        "Replay AND/AND, AND/OR, OR/AND, and OR/OR with identical point-in-time copula fits, costs, and signal thresholds.",
        "copula",
        (
            ("Advanced Pairs Trading: Intro to the Copula Approach", "hLV5Roa_Bek", 855),
            ("Advanced Pairs Trading: Variations on the Copula Based Mispricing Index Strategy.", "t99uCFQL5KI", 1215),
        ),
        "The reviewed captions say performance is highly sensitive to Boolean signal assembly and describe AND-open/OR-exit as a more stable variation.",
    ),
    CuratedIdea(
        "copula_model_dependence",
        "copula",
        "Copula model-dependence audit",
        "Require signal direction and trade lifecycle to survive several statistically plausible copula families.",
        "Create a copula-consensus score and use family disagreement as a veto, size reduction, or separate regime state.",
        "Compare top information-criterion families, parameter perturbations, threshold bands, flag reset rules, and divergence stop-loss rules.",
        "copula",
        (
            ("Advanced Pairs Trading: Intro to the Copula Approach", "hLV5Roa_Bek", 1845),
            ("Advanced Pairs Trading: Variations on the Copula Based Mispricing Index Strategy.", "t99uCFQL5KI", 990),
        ),
        "The reviewed captions repeatedly warn that mispricing-index signals and position flips depend materially on the selected copula and logic.",
    ),
    CuratedIdea(
        "vine_cohort_and_exit",
        "copula",
        "Vine Copula cohort selection and exit repair",
        "Use Vine Copulas only after point-in-time cohort selection demonstrates stable multivariate dependency.",
        "Test extrema-based cohort selection and replace the documented early Bollinger exit with conditional-density normalization and risk exits.",
        "Compare random, sector, dependency-clustered, and extrema-selected cohorts; attribute gains separately to selection and exit logic.",
        "vine_copula",
        (("Advanced Pairs Trading: Vine Copula Trading Strategy", "FB6tNgCZSbo", 1080),),
        "The reviewed caption reports weak performance for arbitrary groups, stronger extrema selection, and a tendency for Bollinger exits to close too early.",
    ),
    CuratedIdea(
        "ou_cost_aware_thresholds",
        "spread_signal",
        "Cost-aware OU threshold optimization",
        "Derive OU entries and exits from expected first-passage behavior and transaction costs instead of fixing every trade at plus or minus two.",
        "Optimize return and Sharpe per unit time, then compare the analytic thresholds with the regular fixed-threshold styles already in the project.",
        "Refit OU parameters on trailing training windows, freeze thresholds for the next fold, and run parameter-sensitivity bands after costs.",
        "ou_spread;ou_zscorer",
        (("Advanced Pairs Trading: Optimal Trading Thresholds for the O-U Process", "C7iZLMXyIOQ", 315),),
        "The reviewed caption derives entry/exit boundaries from first-passage time, risk-free rate, and round-trip transaction cost.",
    ),
    CuratedIdea(
        "minimum_profit_frequency",
        "spread_signal",
        "Minimum-profit versus trade-frequency optimization",
        "Select spread boundaries by balancing minimum profit per trade against expected trade frequency and holding time.",
        "Add a minimum-profit feasibility check that rejects thresholds whose gross boundary value cannot clear fees, funding, and slippage.",
        "Estimate first-passage duration and inter-trade interval point-in-time, then compare profit-per-time with fixed z-score entries.",
        "static_spread;dynamic_spread;ou_spread",
        (("Pairs Trading: The Cointegration Approach and Minimum Profit Optimization", "1zz91G0nR14", 1036),),
        "The reviewed caption explicitly frames threshold choice as a trade-off between profit per trade and trade frequency under transaction costs.",
    ),
    CuratedIdea(
        "hedge_error_stationarity",
        "stationarity",
        "Hedge-error stationarity and hedge-ratio tournament",
        "Evaluate hedge-ratio estimators by the stationarity and boundedness of realized hedge error, not regression fit alone.",
        "Run OLS-differences, OLS-levels, ECM, PCA, and dynamic estimators as competing exact hedge-construction modes.",
        "Freeze each estimator per fold and compare hedge-error stationarity, beta drift, turnover, costs, and spread strategy outcomes.",
        "static_spread;dynamic_spread;ou_spread",
        (("Advanced Pairs Trading: Hedge Ratio Estimation Methods", "odh5rH3WYJM", 137),),
        "The reviewed caption distinguishes static and dynamic hedge methods and identifies non-stationary, unbounded hedge error as the central failure.",
    ),
    CuratedIdea(
        "meta_label_calibrated_sizing",
        "machine_learning",
        "Calibrated meta-label trade gate and sizing",
        "Use a secondary model only after a base strategy proposes a discrete trade, then calibrate its output before using it for size.",
        "Audit extreme confidence deciles for reversal and compare trade/no-trade gating with risk-constrained Kelly sizing.",
        "Require enough candidate trades, a non-overfit primary model, calibrated probabilities, take-rate floors, and drawdown attribution.",
        "model_gate;all_pair_strategies",
        (
            ("Meta-Labeling: Solving for Non Stationarity and Position Sizing", "WbgglcXfEzA", 1261),
            ("Meta-Labeling: Calibration and Position Sizing", "BIBSv_gwBgs", 3331),
        ),
        "The reviewed captions distinguish model confidence from probability, show calibration failures at extreme deciles, and warn against daily vectorized use or tiny trade samples.",
    ),
    CuratedIdea(
        "purged_combinatorial_validation",
        "validation",
        "Purged, embargoed, and combinatorial validation",
        "Purge overlapping trade labels and embargo observations adjacent to test folds before evaluating any learned gate.",
        "Use combinatorial purged cross-validation to produce a distribution of backtest paths instead of trusting one walk-forward path.",
        "Compare walk-forward, purged K-fold, and combinatorial purged folds; report leakage sensitivity, path dispersion, and recent-regime coverage.",
        "model_gate;rl_research;all_pair_strategies",
        (
            ("Modelling: Label Concurrency and Cross Validation", "lDTSGK4JMYk", 900),
            ("Cross-Validation in Finance and Backtesting", "kyjmHHoMc80", 855),
        ),
        "The reviewed captions explain label concurrency, purging, embargo, and the benefit of multiple combinatorial backtest paths.",
    ),
    CuratedIdea(
        "clustered_feature_importance",
        "machine_learning",
        "Clustered feature-importance stability",
        "Measure feature importance at the dependency-cluster level so correlated substitutes do not hide or duplicate signal.",
        "Cluster Wizard, spread, copula, regime, and execution features using several dependence metrics and compare cluster stability across folds.",
        "Run clustered MDA/MDI with purged folds, multiple random seeds, and a report of informative, redundant, and noise clusters.",
        "model_gate",
        (
            ("Clustered Feature Importance Algorithms in Financial Machine Learning: Part 2", "rY4JWAz194Q", 46),
            ("Feature Importance Algorithms in Financial Machine Learning: Part 1", "-A7yrsOihNM", 720),
        ),
        "The reviewed captions address masking and substitution effects by grouping similar features before measuring importance.",
    ),
    CuratedIdea(
        "conditioned_policy_challengers",
        "reinforcement_learning",
        "Conditioned portfolio and RL challengers",
        "Use regime-conditioned optimization and RL as challengers to deterministic strategy decisions, not as execution authority.",
        "Train policies across multiple reward functions, action spaces, asset clusters, and regime features; always retain equal-weight and rule-based baselines.",
        "Freeze environments, include costs and risk in rewards, compare discrete versus continuous sizing, and report baseline-relative out-of-sample failures.",
        "rl_research;portfolio_sizing",
        (
            ("Deep Reinforcement Learning for Trading", "PwoTb-SxoC0", 900),
            ("Conditional Portfolio Optimization", "vyjgR5Ln57s", 2025),
        ),
        "The reviewed captions emphasize state/action/reward design and also show a simple equal-weight baseline beating the conditioned optimizer out of sample.",
    ),
)


def youtube_channel_research_paths(
    *,
    root: Path = ROOT,
    source_key: str = HUDSON_THAMES_KEY,
) -> dict[str, Path]:
    external = root / "data" / "external" / "youtube" / source_key
    processed = root / "data" / "processed" / "youtube_channels" / source_key
    research = root / "reports" / "research"
    dashboard = root / "reports" / "dashboard"
    return {
        "external": external,
        "snapshots": external / "snapshots",
        "subtitles": external / "subtitles",
        "video_registry": external / "video_registry.csv",
        "caption_manifest": external / "caption_manifest.csv",
        "collection_status": external / "collection_status.csv",
        "inventory": research / f"{source_key}_youtube_inventory.csv",
        "inventory_markdown": research / f"{source_key}_youtube_inventory.md",
        "candidates": processed / "caption_insight_candidates.csv",
        "recommendations": research / f"{source_key}_youtube_recommendations.csv",
        "recommendations_markdown": research / f"{source_key}_youtube_recommendations.md",
        "dashboard": dashboard / f"{source_key}_youtube_research.csv",
        "dashboard_markdown": dashboard / f"{source_key}_youtube_research.md",
        "sources_dashboard": dashboard / "youtube_research_sources.csv",
        "sources_dashboard_markdown": dashboard / "youtube_research_sources.md",
    }


def refresh_hudson_thames_youtube_collection(
    *,
    root: Path = ROOT,
    fetch_live: bool = True,
    fetch_captions: bool = True,
    catalog_path: Path | None = None,
) -> CommandResult:
    paths = youtube_channel_research_paths(root=root)
    for key in ("external", "snapshots", "subtitles"):
        paths[key].mkdir(parents=True, exist_ok=True)

    now = _now()
    previous_status = _read_csv(paths["collection_status"])
    live_status = "not_requested"
    live_error = ""
    selected_catalog = catalog_path or _latest_snapshot(paths["snapshots"])
    if fetch_live:
        try:
            selected_catalog = _fetch_live_catalog(HUDSON_THAMES_CHANNEL_URL, paths["snapshots"])
            live_status = "live_catalog_refreshed"
        except (RuntimeError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            live_status = "live_fetch_failed_fallback_local"
            live_error = f"{type(exc).__name__}:{exc}"

    catalog_captured_at = now if live_status == "live_catalog_refreshed" else _snapshot_timestamp(selected_catalog) or now
    payload = _read_json(selected_catalog) if selected_catalog else {}
    registry = _normalize_catalog(payload, selected_catalog, captured_at=catalog_captured_at)
    previous = _read_csv(paths["video_registry"])
    if registry.empty and not previous.empty:
        registry = previous.copy()
    registry = _merge_registry(previous, registry, now=now)

    caption_fetch_requested = 0
    caption_fetch_errors = ""
    if fetch_captions and not registry.empty:
        existing_ids = _caption_video_ids(paths["subtitles"])
        missing_ids = [video_id for video_id in registry["video_id"].astype(str) if video_id not in existing_ids]
        caption_fetch_requested = len(missing_ids)
        if missing_ids:
            caption_fetch_errors = _fetch_captions(missing_ids, paths["subtitles"])

    manifest = _caption_manifest(paths["subtitles"])
    if not registry.empty:
        counts = manifest.groupby("video_id").size().to_dict() if not manifest.empty else {}
        registry["caption_file_count"] = registry["video_id"].map(counts).fillna(0).astype(int)
        registry["caption_status"] = registry["caption_file_count"].map(lambda count: "available" if count else "missing")
    _write_csv(registry, paths["video_registry"])
    _write_csv(manifest, paths["caption_manifest"])

    status = pd.DataFrame(
        [
            {
                "checked_at": now,
                "source_key": HUDSON_THAMES_KEY,
                "channel": "Hudson & Thames",
                "channel_url": HUDSON_THAMES_CHANNEL_URL,
                "live_status": live_status,
                "live_error": live_error,
                "catalog_path": str(selected_catalog or ""),
                "videos": len(registry),
                "videos_with_captions": int(registry.get("caption_status", pd.Series(dtype=str)).eq("available").sum()),
                "caption_files": len(manifest),
                "caption_fetch_requested": caption_fetch_requested,
                "caption_fetch_errors": caption_fetch_errors,
                "collector_ready": _yt_dlp_available(),
                "last_live_refresh_at": (
                    now
                    if live_status == "live_catalog_refreshed"
                    else _snapshot_timestamp(selected_catalog) or _first_text(previous_status, "last_live_refresh_at")
                ),
                "promotion_authority": "none_research_only",
            }
        ]
    )
    _write_csv(status, paths["collection_status"])
    return CommandResult(
        paths={key: paths[key] for key in ("video_registry", "caption_manifest", "collection_status")},
        summary={
            "videos": len(registry),
            "videos_with_captions": int(status.iloc[0]["videos_with_captions"]),
            "caption_fetch_requested": caption_fetch_requested,
            "live_status": live_status,
        },
    )


def build_hudson_thames_youtube_research(*, root: Path = ROOT) -> CommandResult:
    paths = youtube_channel_research_paths(root=root)
    registry = _read_csv(paths["video_registry"])
    manifest = _read_csv(paths["caption_manifest"])
    selected, candidates = mine_youtube_caption_insights(registry, manifest)
    recommendations = build_youtube_theme_recommendations(candidates)
    recommendations = _append_curated_recommendations(recommendations, registry)
    inventory = _build_inventory(registry)

    _write_csv(inventory, paths["inventory"])
    _write_csv(candidates, paths["candidates"])
    _write_csv(recommendations, paths["recommendations"])
    _write_csv(recommendations, paths["dashboard"])
    _write_text(_inventory_markdown(inventory), paths["inventory_markdown"])
    report = _recommendations_markdown(inventory, candidates, recommendations)
    _write_text(report, paths["recommendations_markdown"])
    _write_text(report, paths["dashboard_markdown"])
    sources_dashboard = _build_sources_dashboard(root, inventory, candidates, recommendations)
    _write_csv(sources_dashboard, paths["sources_dashboard"])
    _write_text(_sources_dashboard_markdown(sources_dashboard), paths["sources_dashboard_markdown"])
    return CommandResult(
        paths={
            "inventory": paths["inventory"],
            "inventory_markdown": paths["inventory_markdown"],
            "candidates": paths["candidates"],
            "recommendations": paths["recommendations"],
            "recommendations_markdown": paths["recommendations_markdown"],
            "dashboard": paths["dashboard"],
            "dashboard_markdown": paths["dashboard_markdown"],
            "sources_dashboard": paths["sources_dashboard"],
            "sources_dashboard_markdown": paths["sources_dashboard_markdown"],
        },
        summary={
            "videos": len(inventory),
            "captions_scanned": len(selected),
            "candidate_windows": len(candidates),
            "recommendations": len(recommendations),
            "source_videos": int(candidates.get("video_id", pd.Series(dtype=str)).nunique()),
            "promotion_authority": "none_research_only",
        },
    )


def run_hudson_thames_youtube_research(
    *,
    root: Path = ROOT,
    fetch_live: bool = True,
) -> CommandResult:
    collection = refresh_hudson_thames_youtube_collection(
        root=root,
        fetch_live=fetch_live,
        fetch_captions=fetch_live,
    )
    research = build_hudson_thames_youtube_research(root=root)
    return CommandResult(
        paths={**collection.paths, **research.paths},
        summary={**collection.summary, **research.summary},
    )


def _fetch_live_catalog(channel_url: str, snapshot_dir: Path) -> Path:
    if not _yt_dlp_available():
        raise RuntimeError("yt_dlp_not_installed")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "--flat-playlist",
            "--dump-single-json",
            "--no-warnings",
            f"{channel_url.rstrip('/')}/videos",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    payload = json.loads(result.stdout)
    output = snapshot_dir / f"channel_videos_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def _fetch_captions(video_ids: list[str], output_dir: Path) -> str:
    if not _yt_dlp_available():
        return "yt_dlp_not_installed"
    errors: list[str] = []
    for start in range(0, len(video_ids), 20):
        batch = video_ids[start : start + 20]
        urls = [f"https://www.youtube.com/watch?v={video_id}" for video_id in batch]
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "--skip-download",
                "--write-subs",
                "--write-auto-subs",
                "--sub-langs",
                "en,en-orig",
                "--sub-format",
                "vtt",
                "--ignore-errors",
                "--no-overwrites",
                "--no-warnings",
                "-o",
                str(output_dir / "%(id)s.%(ext)s"),
                *urls,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1200,
        )
        if result.returncode:
            errors.append(f"batch_{start // 20 + 1}:exit_{result.returncode}")
    return ";".join(errors)


def _normalize_catalog(payload: object, source_path: Path | None, *, captured_at: str) -> pd.DataFrame:
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    channel = str(payload.get("channel", "Hudson & Thames")) if isinstance(payload, dict) else "Hudson & Thames"
    channel_id = str(payload.get("channel_id", "")) if isinstance(payload, dict) else ""
    rows: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        video_id = str(entry.get("id", "")).strip()
        if len(video_id) != 11:
            continue
        title = str(entry.get("title", "")).strip()
        rows.append(
            {
                "video_id": video_id,
                "title": title,
                "url": str(entry.get("url", "")).strip() or f"https://www.youtube.com/watch?v={video_id}",
                "duration_seconds": entry.get("duration", ""),
                "published_at": entry.get("timestamp", entry.get("upload_date", "")),
                "channel": channel,
                "channel_id": channel_id,
                "source_snapshot": str(source_path or ""),
                "captured_at": captured_at,
                "metadata_hash": _stable_hash(video_id, title, entry.get("duration", "")),
            }
        )
    return pd.DataFrame(rows)


def _merge_registry(previous: pd.DataFrame, current: pd.DataFrame, *, now: str) -> pd.DataFrame:
    if current.empty:
        return current
    previous_map = previous.set_index("video_id").to_dict("index") if not previous.empty and "video_id" in previous else {}
    rows: list[dict[str, Any]] = []
    for row in current.to_dict("records"):
        old = previous_map.get(str(row["video_id"]), {})
        row["first_seen_at"] = old.get("first_seen_at", now)
        row["last_seen_at"] = now
        row["is_new"] = not bool(old)
        row["metadata_changed"] = bool(old) and old.get("metadata_hash") != row.get("metadata_hash")
        rows.append(row)
    return pd.DataFrame(rows)


def _caption_manifest(directory: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.vtt")):
        parts = path.name.split(".")
        rows.append(
            {
                "video_id": parts[0],
                "language": parts[1] if len(parts) > 2 else "",
                "variant": ".".join(parts[1:-1]),
                "path": str(path.resolve()),
                "sha256": _file_hash(path),
                "bytes": path.stat().st_size,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def _caption_video_ids(directory: Path) -> set[str]:
    return {path.name.split(".")[0] for path in directory.glob("*.vtt")}


def _build_inventory(registry: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_key",
        "channel",
        "video_id",
        "title",
        "url",
        "duration_seconds",
        "caption_status",
        "research_topics",
        "research_relevance_score",
        "research_priority",
        "promotion_authority",
    ]
    rows: list[dict[str, object]] = []
    for _, video in registry.iterrows():
        title = str(video.get("title", ""))
        lowered = title.lower()
        topics = [topic for topic, terms in TITLE_TOPICS.items() if any(term in lowered for term in terms)]
        score = sum(sum(1 for term in terms if term in lowered) for terms in TITLE_TOPICS.values())
        priority = "must_review" if score >= 2 else "useful" if score == 1 else "optional"
        rows.append(
            {
                "source_key": HUDSON_THAMES_KEY,
                "channel": "Hudson & Thames",
                "video_id": video.get("video_id", ""),
                "title": title,
                "url": video.get("url", ""),
                "duration_seconds": video.get("duration_seconds", ""),
                "caption_status": video.get("caption_status", "missing"),
                "research_topics": ";".join(topics),
                "research_relevance_score": score,
                "research_priority": priority,
                "promotion_authority": "none_research_only",
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["research_relevance_score", "title"], ascending=[False, True]
    ).reset_index(drop=True)


def _append_curated_recommendations(recommendations: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    available = set(
        registry.loc[
            registry.get("caption_status", pd.Series("missing", index=registry.index)).astype(str).eq("available"),
            "video_id",
        ].astype(str)
    ) if not registry.empty else set()
    rows = recommendations.to_dict("records")
    columns = recommendations.columns.tolist()
    for idea in HUDSON_THAMES_CURATED_IDEAS:
        if not all(video_id in available for _, video_id, _ in idea.sources):
            continue
        source_links = [
            (
                title,
                f"https://www.youtube.com/watch?v={video_id}",
                _format_timestamp(seconds),
                f"https://www.youtube.com/watch?v={video_id}&t={seconds}s",
            )
            for title, video_id, seconds in idea.sources
        ]
        source_videos = " | ".join(f"{title} ({url})" for title, url, _, _ in source_links)
        rows.append(
            {
                "schema_version": "youtube_caption_insights.v1",
                "recommendation_id": f"htr_{_stable_hash(idea.key, source_videos)[:20]}",
                "theme": idea.theme,
                "theme_label": idea.label,
                "recommendation_type": "caption_supported_research_hypothesis",
                "recommendation": idea.recommendation,
                "interesting_suggestion": idea.interesting_suggestion,
                "local_validation_test": idea.local_test,
                "strategy_families": idea.strategy_families,
                "priority": "must_test",
                "priority_score": 0.0,
                "confidence": 0.85,
                "source_video_count": len(source_links),
                "source_videos": source_videos,
                "source_timestamps": " | ".join(
                    f"{title} [{timestamp}]({url})" for title, _, timestamp, url in source_links
                ),
                "evidence_excerpts": "",
                "evidence_basis": idea.evidence_basis,
                "directness": "assistant_synthesis_from_manually_reviewed_caption_windows",
                "review_status": "assistant_reviewed_2026-08-06",
                "promotion_authority": "none_research_only",
                "trade_authorized": False,
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    frame["priority_rank"] = frame.get("priority", pd.Series("useful", index=frame.index)).map(
        {"must_test": 0, "useful": 1}
    ).fillna(2)
    return frame.sort_values(["priority_rank", "priority_score"], ascending=[True, False]).drop(
        columns="priority_rank"
    ).reset_index(drop=True)


def _inventory_markdown(inventory: pd.DataFrame) -> str:
    lines = [
        "# Hudson & Thames YouTube Inventory",
        "",
        f"- videos: {len(inventory)}",
        f"- videos with captions: {int(inventory.get('caption_status', pd.Series(dtype=str)).eq('available').sum())}",
        f"- must-review titles: {int(inventory.get('research_priority', pd.Series(dtype=str)).eq('must_review').sum())}",
        "- promotion authority: none; research source only",
        "",
        "## Highest-Relevance Videos",
        "",
    ]
    for row in inventory.head(30).itertuples(index=False):
        lines.append(f"- [{row.title}]({row.url}) - {row.research_topics or 'general_quant_research'}")
    return "\n".join(lines) + "\n"


def _build_sources_dashboard(
    root: Path,
    inventory: pd.DataFrame,
    candidates: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    crypto_collection = _read_csv(root / "reports" / "agents" / "youtube_brain_collection_status.csv")
    crypto_recommendations = _read_csv(root / "reports" / "agents" / "youtube_brain_recommendations.csv")
    crypto_candidates = _read_csv(root / "data" / "processed" / "youtube_brain" / "caption_insight_candidates.csv")
    if not crypto_collection.empty:
        rows.append(
            {
                "source_key": "crypto_wizards",
                "channel": "Crypto Wizards",
                "channel_url": str(crypto_collection.iloc[-1].get("channel_url", "")),
                "videos": _safe_int(crypto_collection.iloc[-1].get("videos", 0)),
                "videos_with_captions": _safe_int(crypto_collection.iloc[-1].get("videos_with_captions", 0)),
                "caption_insight_windows": len(crypto_candidates),
                "recommendations": len(crypto_recommendations),
                "curated_must_test": int(
                    crypto_recommendations.get("priority", pd.Series(dtype=str)).astype(str).eq("must_test").sum()
                ),
                "report_path": str(root / "reports" / "agents" / "youtube_brain_recommendations.md"),
                "promotion_authority": "none_research_only",
            }
        )
    rows.append(
        {
            "source_key": HUDSON_THAMES_KEY,
            "channel": "Hudson & Thames",
            "channel_url": HUDSON_THAMES_CHANNEL_URL,
            "videos": len(inventory),
            "videos_with_captions": int(inventory.get("caption_status", pd.Series(dtype=str)).eq("available").sum()),
            "caption_insight_windows": len(candidates),
            "recommendations": len(recommendations),
            "curated_must_test": int(
                recommendations.get("priority", pd.Series(dtype=str)).astype(str).eq("must_test").sum()
            ),
            "report_path": str(root / "reports" / "research" / "hudson_thames_youtube_recommendations.md"),
            "promotion_authority": "none_research_only",
        }
    )
    return pd.DataFrame(rows)


def _sources_dashboard_markdown(sources: pd.DataFrame) -> str:
    lines = [
        "# YouTube Research Sources",
        "",
        "Each channel is an independent research lane. Findings can prioritize local tests but cannot authorize execution.",
        "",
    ]
    for row in sources.itertuples(index=False):
        lines.extend(
            [
                f"## {row.channel}",
                "",
                f"- videos: {row.videos}",
                f"- videos with captions: {row.videos_with_captions}",
                f"- caption insight windows: {row.caption_insight_windows}",
                f"- recommendations: {row.recommendations}",
                f"- curated must-test hypotheses: {row.curated_must_test}",
                f"- report: `{row.report_path}`",
                "- promotion authority: none; local validation required",
                "",
            ]
        )
    return "\n".join(lines)


def _recommendations_markdown(
    inventory: pd.DataFrame,
    candidates: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> str:
    lines = [
        "# Hudson & Thames YouTube Research Recommendations",
        "",
        "These are machine-assisted inferences from Hudson & Thames captions. They require manual review and local validation.",
        "",
        f"- videos inventoried: {len(inventory)}",
        f"- caption insight windows: {len(candidates)}",
        f"- source videos represented: {candidates.get('video_id', pd.Series(dtype=str)).nunique()}",
        f"- recommendation rows: {len(recommendations)}",
        "- promotion authority: none; no item can authorize Testnet or live execution",
        "",
    ]
    for priority, heading in (("must_test", "Curated Must-Test Hypotheses"), ("useful", "Theme Recommendations")):
        subset = recommendations[recommendations.get("priority", pd.Series(dtype=str)).astype(str).eq(priority)]
        if subset.empty:
            continue
        lines.extend([f"## {heading}", ""])
        for row in subset.itertuples(index=False):
            lines.extend(
                [
                    f"### {row.theme_label}",
                    "",
                    f"**Recommendation:** {row.recommendation}",
                    "",
                    f"**Interesting test idea:** {row.interesting_suggestion}",
                    "",
                    f"**Local validation:** {row.local_validation_test}",
                    "",
                    f"**Evidence basis:** {row.evidence_basis}",
                    "",
                    f"**Source windows:** {row.source_timestamps}",
                    "",
                    f"Confidence: {row.confidence:.2f}; review status: {row.review_status}.",
                    "",
                ]
            )
    return "\n".join(lines)


def _format_timestamp(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _latest_snapshot(directory: Path) -> Path | None:
    candidates = sorted(directory.glob("channel_videos_*.json"))
    return candidates[-1] if candidates else None


def _snapshot_timestamp(path: Path | None) -> str:
    if path is None:
        return ""
    match = re.search(r"(\d{8}T\d{6}Z)", path.name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return ""


def _yt_dlp_available() -> bool:
    return importlib.util.find_spec("yt_dlp") is not None


def _stable_hash(*parts: object) -> str:
    return sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:24]


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path | None) -> object:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _first_text(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return ""
    values = frame[column].astype(str).str.strip()
    values = values[values.ne("")]
    return values.iloc[-1] if not values.empty else ""


def _safe_int(value: object) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    return 0 if pd.isna(numeric) else int(numeric)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
