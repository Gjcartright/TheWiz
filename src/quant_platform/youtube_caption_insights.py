from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

from dataclasses import dataclass
from hashlib import sha256
from html import unescape
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT


CAPTION_INSIGHT_SCHEMA_VERSION = "youtube_caption_insights.v1"
WINDOW_SECONDS = 45
MAX_EXCERPT_WORDS = 12


@dataclass(frozen=True)
class Theme:
    key: str
    label: str
    terms: tuple[str, ...]
    recommendation: str
    interesting_suggestion: str
    local_test: str
    strategy_families: str


@dataclass(frozen=True)
class CuratedIdea:
    key: str
    theme: str
    title: str
    recommendation: str
    interesting_suggestion: str
    local_test: str
    strategy_families: str
    sources: tuple[tuple[str, str, int], ...]
    evidence_basis: str


THEMES: tuple[Theme, ...] = (
    Theme(
        "stationarity",
        "Stationarity and cointegration",
        ("stationarity", "stationary", "cointegration", "engle granger", "johansen", "unit root", "adf"),
        "Gate pair hypotheses with point-in-time stationarity and cointegration diagnostics before spread testing.",
        "Treat disagreement between Engle-Granger, Johansen, and ADF as a regime feature instead of silently choosing the favorable test.",
        "Compare expanding-window Engle-Granger, Johansen, and ADF results; measure break frequency and out-of-sample survival.",
        "static_spread;static_zscorer;dynamic_spread;dynamic_zscorer;ou_spread;ou_zscorer",
    ),
    Theme(
        "dependency",
        "Dependency structure",
        ("correlation", "pearson", "spearman", "kendall", "dependency", "dependence", "ecm", "error correction"),
        "Use correlation and ECM fields as diagnostics, not as standalone proof that a spread will mean-revert.",
        "Create a dependency-consensus feature that records Pearson, Spearman, Kendall, ECM-X, ECM-Y, and ECM-strength agreement.",
        "Ablate each dependency field in walk-forward tests and compare consensus, disagreement, and single-metric variants.",
        "all_pair_strategies",
    ),
    Theme(
        "spread_signal",
        "Spread and signal design",
        ("z score", "zscore", "spread", "mean reversion", "ornstein", "uhlenbeck", "kalman", "half life", "zero crossing"),
        "Preserve each Wizard spread and ZScoreR mode as a separate hypothesis with its own entry and exit behavior.",
        "Run an exact-mode tournament across Static Spread, Static ZScoreR, Dyn Spread, Dyn ZScoreR, OU Spread, and OU ZScoreR.",
        "Replay identical pair/timeframe folds for all six modes with entry-only and hard-exit regime overlays.",
        "static_spread;static_zscorer;dynamic_spread;dynamic_zscorer;ou_spread;ou_zscorer",
    ),
    Theme(
        "copula",
        "Copula dislocation",
        ("copula", "conditional distribution", "conditional probability", "tail dependence", "vine copula", "mispricing index"),
        "Keep copula dislocation as a distinct strategy family rather than treating it as another z-score.",
        "Test copula signals both alone and as confirmation for spread entries, with tail-specific exits and asymmetric risk limits.",
        "Fit copulas only on prior data; compare standalone, confirmation, and veto roles under costed walk-forward replay.",
        "copula",
    ),
    Theme(
        "risk_sizing",
        "Risk and position sizing",
        ("risk", "position size", "position sizing", "volatility", "garch", "drawdown", "value at risk", "var", "cvar"),
        "Size pair legs from hedge exposure and volatility rather than equal notional alone.",
        "Compare beta-neutral, volatility-balanced, and tail-risk-budgeted sizing while preserving a hard pair-level loss budget.",
        "Replay all accepted hypotheses under equal-notional, beta-neutral, inverse-volatility, and CVaR-budgeted sizing.",
        "all_pair_strategies",
    ),
    Theme(
        "regime",
        "Regime and serial behavior",
        ("regime", "hidden markov", "hmm", "gaussian mixture", "autocorrelation", "serial correlation", "market state"),
        "Estimate regime and serial-dependence features only from information available before each entry.",
        "Use regime-model disagreement as an uncertainty flag and reduce size instead of forcing a single state label.",
        "Compare existing deterministic overlays with expanding-window HMM/GMM and autocorrelation features using purged folds.",
        "all_pair_strategies",
    ),
    Theme(
        "validation",
        "Backtest validation",
        ("backtest", "walk forward", "out of sample", "overfit", "look ahead", "forward test", "paper trading", "win rate"),
        "Require costed walk-forward and Testnet evidence before promoting any video-derived strategy idea.",
        "Track how often a strong full-sample result fails after purging, costs, and parameter freezing; make that decay a model feature.",
        "Run purged walk-forward folds with frozen parameters, sensitivity bands, after-cost metrics, and minimum trade-count gates.",
        "all_pair_strategies",
    ),
    Theme(
        "machine_learning",
        "Machine learning",
        ("machine learning", "xgboost", "feature importance", "random forest", "gradient boosting", "classification", "prediction"),
        "Use machine learning as an explainable trade-quality gate after the base strategy produces a candidate.",
        "Measure feature-importance stability across folds and pairs; unstable importance should lower confidence even when headline metrics improve.",
        "Compare raw, rule-filtered, model-gated, and model-sized variants with take-rate and concentration controls.",
        "model_gate",
    ),
    Theme(
        "reinforcement_learning",
        "Reinforcement learning",
        ("reinforcement learning", "deep reinforcement", "reward function", "policy gradient", "q learning"),
        "Use reinforcement learning to propose and challenge hypotheses, not to bypass deterministic safety and acceptance gates.",
        "Let the RL research agent search entry, exit, sizing, and regime combinations, then cluster its ideas to find similar Wizard pairs.",
        "Evaluate policies on frozen candidate sets with turnover, drawdown, and skipped-opportunity penalties before Testnet handoff.",
        "rl_research",
    ),
    Theme(
        "execution",
        "Execution and market frictions",
        ("execution", "slippage", "liquidity", "order book", "funding", "fee", "bid ask", "market impact", "exchange"),
        "Calibrate fees, funding, and slippage from the exact Hyperliquid market and network used for the replay.",
        "Maintain separate research and execution feasibility scores so thin markets remain visible without becoming tradable by accident.",
        "Replay fills against timestamped L2 snapshots and realized Testnet fills; compare assumed versus observed cost distributions.",
        "hyperliquid_execution",
    ),
)

RECOMMENDATION_SIGNALS: tuple[str, ...] = (
    "should",
    "need to",
    "important",
    "recommend",
    "one way",
    "useful",
    "make sure",
    "do not",
    "don't",
    "avoid",
    "problem",
    "issue",
    "improve",
    "better",
    "boost",
    "fail",
)

CURATED_IDEAS: tuple[CuratedIdea, ...] = (
    CuratedIdea(
        "mode_specific_feature_profiles",
        "machine_learning",
        "Mode-specific feature profiles",
        "Train and evaluate feature gates separately for each exact Wizard mode because predictive features differ by strategy construction.",
        "Use a shared feature catalog but require mode-specific stability reports: low half-life mattered for Dynamic Spread, crossing behavior mattered for OU, and copula dependency strength mattered for Copula.",
        "Fit per-mode explainable gates on purged folds, then compare feature rank stability and cross-mode transfer failure.",
        "dynamic_spread;ou_spread;ou_zscorer;copula",
        (("180,000 Backtests Meet XGBOOST...Here are the Important Features", "hJam2VHK3-A", 360),),
        "The reviewed caption distinguishes feature importance across Dynamic Spread, OU, and Copula tests.",
    ),
    CuratedIdea(
        "raw_vs_rolling_adaptivity",
        "spread_signal",
        "Raw spread versus rolling ZScoreR adaptivity",
        "Test raw and rolling-ZScoreR variants independently; do not assume a rolling z-score improves an already dynamic spread.",
        "Add an adaptivity-overlap diagnostic: Kalman-updated spreads plus rolling normalization may create choppier signals than the raw dynamic spread.",
        "Compare Dynamic Spread versus Dyn ZScoreR and OU Spread versus OU ZScoreR on identical frozen folds, costs, and trade thresholds.",
        "dynamic_spread;dynamic_zscorer;ou_spread;ou_zscorer",
        (("UNEXPECTED Results - Statistical Arbitrage Forward testing", "p28JpBi55Ko", 675),),
        "The reviewed forward-test caption says Dynamic rolling ZScoreR may be redundant and reports materially different behavior by mode.",
    ),
    CuratedIdea(
        "feature_transfer_decay",
        "validation",
        "Backtest-to-forward feature transfer decay",
        "Measure whether features that explain backtest profitability retain direction and rank in forward tests.",
        "Create a feature-transfer-decay score and penalize modes whose important backtest features disappear or reverse out of sample.",
        "For every exact mode, compare SHAP direction, rank correlation, and predictive lift between training and forward-only folds.",
        "all_pair_strategies;model_gate",
        (("UNEXPECTED Results - Statistical Arbitrage Forward testing", "p28JpBi55Ko", 810),),
        "The reviewed caption reports that some OU backtest drivers did not resemble the forward-test drivers.",
    ),
    CuratedIdea(
        "copula_calibration_stability",
        "copula",
        "Copula family and calibration-window stability",
        "Require copula family, dependency strength, and signal direction to remain stable across several trailing calibration windows.",
        "Treat a family switch between Clayton, Gaussian, Student-t, and Gumbel as model uncertainty; test it as a veto or size reduction.",
        "Refit point-in-time over multiple trailing windows, record family switches, and compare standalone, confirmation, and veto roles.",
        "copula",
        (
            ("180,000 Backtests Meet XGBOOST...Here are the Important Features", "hJam2VHK3-A", 1080),
            ("Copulas - A Powerful Tool in Statistical Arbitrage", "ZRCpGNiv0Xo", 1050),
            ("Vine Copulas in Statistical Arbitrage - Introduction", "MOK4nb1l5RA", 1980),
        ),
        "The reviewed captions show family-specific tail structure and warn that changing the data period can materially change copula shape.",
    ),
    CuratedIdea(
        "trend_stationarity_routing",
        "stationarity",
        "Trend-stationarity strategy routing",
        "Route Engle-Granger-with-trend results to rolling or dynamic research lanes instead of treating them as ordinary static mean reversion.",
        "Use the Wizard orange/green stationarity state as a strategy-routing variable and test whether trend-adjusted pairs need moving centers or earlier exits.",
        "Compare static, rolling-ZScoreR, and dynamic modes separately for green, orange, and unflagged point-in-time stationarity states.",
        "static_spread;static_zscorer;dynamic_spread;dynamic_zscorer",
        (("Engle Granger Cointegration With and Without a Trend", "96z6UOIzrlU", 0),),
        "The reviewed caption defines orange as stationarity around a trend and suggests a rolling z-score or adjusted base may be more appropriate.",
    ),
    CuratedIdea(
        "dependency_consensus",
        "dependency",
        "Stationarity and dependency consensus",
        "Combine Engle-Granger, Johansen, zero-crossing, half-life, Hurst, and ECM evidence without allowing any one favorable metric to dominate.",
        "Score both consensus and disagreement; a pair can be research-interesting when tests disagree, but disagreement should lower execution confidence.",
        "Run component ablations and a disagreement interaction across exact modes using only expanding-window values.",
        "all_pair_strategies",
        (
            ("Statistical Arbitrage for the Uninitiated (no fluff)", "-Fr-Nz-uO2U", 225),
            ("180,000 Backtests Meet XGBOOST...Here are the Important Features", "hJam2VHK3-A", 1395),
        ),
        "The reviewed captions distinguish spread stationarity, Johansen structure, crossing behavior, Hurst, half-life, and ECM directionality.",
    ),
    CuratedIdea(
        "garch_dcc_pair_sizing",
        "risk_sizing",
        "GARCH volatility and DCC correlation sizing",
        "Compare inverse-volatility leg allocation with beta-neutral sizing, then add dynamic correlation only if it improves realized pair-risk control.",
        "Use DCC-GARCH as a research challenger for periods when volatility and correlation cluster together, not as an automatic sizing authority.",
        "Replay fixed-notional, beta-neutral, inverse-GARCH-volatility, and DCC-adjusted sizing with turnover and drawdown attribution.",
        "all_pair_strategies",
        (("A Useful Technique to Model Risk and Allocate Capital for Pairs Traders", "XVzC9__kQXI", 0),),
        "The reviewed caption derives inverse-volatility allocation and points to DCC-GARCH for time-varying cross-asset correlation.",
    ),
    CuratedIdea(
        "rl_reward_audit",
        "reinforcement_learning",
        "RL state and reward audit",
        "Treat state design and reward design as first-class model artifacts, with deterministic penalties for costs, drawdown, turnover, and unsafe actions.",
        "Use a reward red team that searches for policies exploiting Sharpe, unfilled orders, stale marks, or episode boundaries without creating real after-cost edge.",
        "Evaluate reward variants on frozen environments and require policy behavior, not only cumulative reward, to survive out of sample.",
        "rl_research",
        (("Deep Reinforcement Learning Applied to Crypto and Stock Trading - Beginner Insights", "A3eWVrM3cvI", 405),),
        "The reviewed caption emphasizes that environment state and agent incentives can matter more than the chosen RL algorithm.",
    ),
)


def youtube_caption_insight_paths(root: Path = ROOT) -> dict[str, Path]:
    processed = root / "data" / "processed" / "youtube_brain"
    agents = root / "reports" / "agents"
    dashboard = root / "reports" / "dashboard"
    return {
        "candidates": processed / "caption_insight_candidates.csv",
        "recommendations": agents / "youtube_brain_recommendations.csv",
        "recommendations_markdown": agents / "youtube_brain_recommendations.md",
        "dashboard_recommendations": dashboard / "youtube_brain_recommendations.csv",
        "dashboard_recommendations_markdown": dashboard / "youtube_brain_recommendations.md",
    }


def build_youtube_caption_insights(*, root: Path = ROOT) -> CommandResult:
    registry = _read_csv(root / "data" / "external" / "youtube" / "brain" / "video_registry.csv")
    manifest = _read_csv(root / "data" / "external" / "youtube" / "brain" / "caption_manifest.csv")
    selected, candidates = mine_youtube_caption_insights(registry, manifest)
    recommendations = _build_recommendations(
        candidates,
        available_video_ids=set(selected.get("video_id", pd.Series(dtype=str)).astype(str)),
    )
    paths = youtube_caption_insight_paths(root)

    _write_csv(candidates, paths["candidates"])
    _write_csv(recommendations, paths["recommendations"])
    _write_csv(recommendations, paths["dashboard_recommendations"])
    markdown = _recommendations_markdown(recommendations, candidates)
    _write_text(markdown, paths["recommendations_markdown"])
    _write_text(markdown, paths["dashboard_recommendations_markdown"])
    return CommandResult(
        paths=paths,
        summary={
            "caption_files_scanned": len(selected),
            "candidate_windows": len(candidates),
            "recommendations": len(recommendations),
            "source_videos": int(candidates.get("video_id", pd.Series(dtype=str)).nunique()),
            "promotion_authority": "none_research_only",
        },
    )


def mine_youtube_caption_insights(
    registry: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = _preferred_captions(manifest)
    return selected, _mine_candidates(registry, selected)


def build_youtube_theme_recommendations(candidates: pd.DataFrame) -> pd.DataFrame:
    return _build_recommendations(candidates, available_video_ids=set())


def _preferred_captions(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return manifest.copy()
    frame = manifest.copy()
    frame["variant_priority"] = frame.get("variant", "").astype(str).map(
        lambda value: 0 if value == "en-orig" else 1 if value == "en" else 2
    )
    frame["path_exists"] = frame.get("path", "").astype(str).map(lambda value: Path(value).exists())
    frame = frame[frame["path_exists"]].sort_values(["video_id", "variant_priority", "bytes"], ascending=[True, True, False])
    return frame.drop_duplicates(subset=["video_id"], keep="first").reset_index(drop=True)


def _mine_candidates(registry: pd.DataFrame, captions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "schema_version",
        "candidate_id",
        "theme",
        "theme_label",
        "signal_type",
        "score",
        "matched_terms",
        "matched_recommendation_signals",
        "video_id",
        "video_title",
        "video_url",
        "timestamp_seconds",
        "timestamp",
        "timestamped_video_url",
        "evidence_excerpt",
        "caption_path",
        "caption_sha256",
        "review_status",
        "promotion_authority",
        "trade_authorized",
    ]
    if captions.empty:
        return pd.DataFrame(columns=columns)
    metadata = registry.set_index("video_id").to_dict("index") if not registry.empty and "video_id" in registry else {}
    rows: list[dict[str, object]] = []
    for _, caption in captions.iterrows():
        video_id = str(caption.get("video_id", "")).strip()
        path = Path(str(caption.get("path", "")).strip())
        cues = _parse_vtt(path)
        if not cues:
            continue
        video = metadata.get(video_id, {})
        title = str(video.get("title", "")).strip()
        base_url = str(video.get("url", "")).strip() or f"https://www.youtube.com/watch?v={video_id}"
        for start, end, text in _caption_windows(cues):
            caption_text = text.lower()
            signal_matches = _matches(caption_text, RECOMMENDATION_SIGNALS)
            for theme in THEMES:
                term_matches = _matches(caption_text, theme.terms)
                if not term_matches:
                    continue
                title_matches = _matches(title.lower(), theme.terms)
                if theme.key == "copula" and not _matches(
                    f"{title} {text}".lower(),
                    ("copula", "vine copula", "tail dependence", "mispricing index"),
                ):
                    continue
                score = 2 * len(term_matches) + len(signal_matches) + min(2, len(title_matches))
                if score < 3:
                    continue
                signal_type = "recommendation_language" if signal_matches else "topic_evidence"
                timestamp_seconds = int(start)
                rows.append(
                    {
                        "schema_version": CAPTION_INSIGHT_SCHEMA_VERSION,
                        "candidate_id": _stable_id("ytci", video_id, theme.key, timestamp_seconds, text),
                        "theme": theme.key,
                        "theme_label": theme.label,
                        "signal_type": signal_type,
                        "score": score,
                        "matched_terms": ";".join(term_matches),
                        "matched_recommendation_signals": ";".join(signal_matches),
                        "video_id": video_id,
                        "video_title": title,
                        "video_url": base_url,
                        "timestamp_seconds": timestamp_seconds,
                        "timestamp": _format_timestamp(timestamp_seconds),
                        "timestamped_video_url": _timestamped_url(base_url, timestamp_seconds),
                        "evidence_excerpt": _short_excerpt(text, (*term_matches, *signal_matches)),
                        "caption_path": str(path),
                        "caption_sha256": str(caption.get("sha256", "")),
                        "review_status": "machine_extracted_needs_human_review",
                        "promotion_authority": "none_research_only",
                        "trade_authorized": False,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["theme", "score", "video_title", "timestamp_seconds"], ascending=[True, False, True, True])
    frame = frame.drop_duplicates(subset=["theme", "video_id"], keep="first")
    return frame.groupby("theme", group_keys=False).head(8).reset_index(drop=True)[columns]


def _build_recommendations(candidates: pd.DataFrame, *, available_video_ids: set[str]) -> pd.DataFrame:
    columns = [
        "schema_version",
        "recommendation_id",
        "theme",
        "theme_label",
        "recommendation_type",
        "recommendation",
        "interesting_suggestion",
        "local_validation_test",
        "strategy_families",
        "priority",
        "priority_score",
        "confidence",
        "source_video_count",
        "source_videos",
        "source_timestamps",
        "evidence_excerpts",
        "evidence_basis",
        "directness",
        "review_status",
        "promotion_authority",
        "trade_authorized",
    ]
    rows: list[dict[str, object]] = []
    for theme in THEMES:
        evidence = candidates[candidates.get("theme", pd.Series(dtype=str)).astype(str).eq(theme.key)].copy()
        if evidence.empty:
            continue
        evidence = evidence.sort_values("score", ascending=False).head(3)
        mean_score = float(pd.to_numeric(evidence["score"], errors="coerce").fillna(0).mean())
        confidence = min(0.65, round(0.35 + 0.02 * mean_score + 0.02 * len(evidence), 2))
        source_videos = " | ".join(
            f"{row.video_title} ({row.video_url})" for row in evidence.itertuples(index=False)
        )
        source_timestamps = " | ".join(
            f"{row.video_title} [{row.timestamp}]({row.timestamped_video_url})" for row in evidence.itertuples(index=False)
        )
        excerpts = " | ".join(str(value) for value in evidence["evidence_excerpt"].tolist())
        rows.append(
            {
                "schema_version": CAPTION_INSIGHT_SCHEMA_VERSION,
                "recommendation_id": _stable_id("ytr", theme.key, theme.recommendation, source_videos),
                "theme": theme.key,
                "theme_label": theme.label,
                "recommendation_type": "inferred_research_recommendation",
                "recommendation": theme.recommendation,
                "interesting_suggestion": theme.interesting_suggestion,
                "local_validation_test": theme.local_test,
                "strategy_families": theme.strategy_families,
                "priority": "useful",
                "priority_score": round(mean_score, 2),
                "confidence": confidence,
                "source_video_count": len(evidence),
                "source_videos": source_videos,
                "source_timestamps": source_timestamps,
                "evidence_excerpts": excerpts,
                "evidence_basis": "deterministic theme extraction from caption windows",
                "directness": "inferred_from_caption_topic_and_recommendation_signals",
                "review_status": "machine_extracted_needs_human_review",
                "promotion_authority": "none_research_only",
                "trade_authorized": False,
            }
        )
    for idea in CURATED_IDEAS:
        if not all(video_id in available_video_ids for _, video_id, _ in idea.sources):
            continue
        source_links = [
            (
                title,
                f"https://www.youtube.com/watch?v={video_id}",
                _format_timestamp(seconds),
                _timestamped_url(f"https://www.youtube.com/watch?v={video_id}", seconds),
            )
            for title, video_id, seconds in idea.sources
        ]
        source_videos = " | ".join(f"{title} ({url})" for title, url, _, _ in source_links)
        source_timestamps = " | ".join(f"{title} [{timestamp}]({url})" for title, _, timestamp, url in source_links)
        rows.append(
            {
                "schema_version": CAPTION_INSIGHT_SCHEMA_VERSION,
                "recommendation_id": _stable_id("ytrc", idea.key, source_videos),
                "theme": idea.theme,
                "theme_label": idea.title,
                "recommendation_type": "caption_supported_research_hypothesis",
                "recommendation": idea.recommendation,
                "interesting_suggestion": idea.interesting_suggestion,
                "local_validation_test": idea.local_test,
                "strategy_families": idea.strategy_families,
                "priority": "must_test",
                "priority_score": 0.0,
                "confidence": 0.8,
                "source_video_count": len(source_links),
                "source_videos": source_videos,
                "source_timestamps": source_timestamps,
                "evidence_excerpts": "",
                "evidence_basis": idea.evidence_basis,
                "directness": "assistant_synthesis_from_manually_reviewed_caption_windows",
                "review_status": "assistant_reviewed_2026-08-06",
                "promotion_authority": "none_research_only",
                "trade_authorized": False,
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    frame["priority_rank"] = frame["priority"].map({"must_test": 0, "useful": 1}).fillna(2)
    return frame.sort_values(["priority_rank", "priority_score"], ascending=[True, False]).drop(columns="priority_rank").reset_index(drop=True)


def _parse_vtt(path: Path) -> list[tuple[float, float, str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    cues: list[tuple[float, float, str]] = []
    index = 0
    previous = ""
    while index < len(lines):
        match = re.match(r"\s*(\d{2}:\d{2}:\d{2}\.\d+)\s+-->\s+(\d{2}:\d{2}:\d{2}\.\d+)", lines[index])
        if not match:
            index += 1
            continue
        start, end = _vtt_seconds(match.group(1)), _vtt_seconds(match.group(2))
        index += 1
        text_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            clean = _clean_caption_text(lines[index])
            if clean:
                text_lines.append(clean)
            index += 1
        text = text_lines[-1] if text_lines else ""
        if text and text != previous:
            cues.append((start, end, text))
            previous = text
    return cues


def _caption_windows(cues: Iterable[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    groups: dict[int, list[tuple[float, float, str]]] = {}
    for cue in cues:
        groups.setdefault(int(cue[0] // WINDOW_SECONDS), []).append(cue)
    windows: list[tuple[float, float, str]] = []
    for bucket in sorted(groups):
        items = groups[bucket]
        phrases: list[str] = []
        for _, _, text in items:
            if not phrases or text != phrases[-1]:
                phrases.append(text)
        windows.append((items[0][0], items[-1][1], " ".join(phrases)))
    return windows


def _clean_caption_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", unescape(value))
    return re.sub(r"\s+", " ", text).strip()


def _vtt_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _matches(text: str, terms: Iterable[str]) -> list[str]:
    found = []
    for term in terms:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, flags=re.IGNORECASE):
            found.append(term)
    return found


def _short_excerpt(text: str, anchors: Iterable[str]) -> str:
    words = text.split()
    if not words:
        return ""
    lowered = text.lower()
    start = 0
    for anchor in anchors:
        position = lowered.find(anchor.lower())
        if position >= 0:
            start = max(0, len(text[:position].split()) - 2)
            break
    return " ".join(words[start : start + MAX_EXCERPT_WORDS])


def _format_timestamp(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _timestamped_url(url: str, seconds: int) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}t={seconds}s"


def _recommendations_markdown(recommendations: pd.DataFrame, candidates: pd.DataFrame) -> str:
    lines = [
        "# YouTube Research Recommendations",
        "",
        "These are machine-assisted research inferences from Crypto Wizards captions. They are not direct endorsements, acceptance evidence, or trade authorization.",
        "",
        f"- caption insight windows: {len(candidates)}",
        f"- source videos represented: {candidates.get('video_id', pd.Series(dtype=str)).nunique()}",
        f"- recommendation rows: {len(recommendations)}",
        "- promotion authority: none; every idea requires local point-in-time, costed walk-forward validation",
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


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return f"{prefix}_{sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(frame, path, index=False)


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, text, encoding="utf-8")
