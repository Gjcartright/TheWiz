from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

from pathlib import Path
import re

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.research_extraction import EXTRACTION_COLUMNS, normalize_extraction_rows
from quant_platform.research_ingestion import active_research_sources, mark_research_sources_processed


def youtube_extraction_paths(root: Path = ROOT) -> dict[str, Path]:
    base = root / "data" / "processed" / "research_knowledge"
    return {
        "youtube_research_rows": base / "youtube_research_rows.csv",
        "youtube_research_rules": base / "youtube_research_rules.csv",
        "youtube_research_features": base / "youtube_research_features.csv",
        "youtube_strategy_hints": base / "youtube_strategy_hints.csv",
    }


def extract_youtube_research(root: Path = ROOT) -> CommandResult:
    registry = active_research_sources(root)
    youtube = registry[registry["source_type"].astype(str) == "youtube"].copy() if not registry.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, row in youtube.iterrows():
        path = Path(str(row.get("source_path_or_url", "")).strip())
        if not path.exists():
            continue
        if path.suffix.lower() != ".md":
            continue
        rows.extend(_extract_markdown_rows(path, row))
    normalized = normalize_extraction_rows(rows)
    paths = youtube_extraction_paths(root)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(normalized, paths["youtube_research_rows"], index=False)
    atomic_write_csv(normalized[normalized["row_type"].isin(["risk_prior", "mode_preference", "regime_condition", "pair_filter", "anti_pattern", "execution_warning"])], paths["youtube_research_rules"], index=False)
    atomic_write_csv(normalized[normalized["row_type"] == "feature_idea"], paths["youtube_research_features"], index=False)
    atomic_write_csv(normalized[normalized["row_type"] == "strategy_hint"], paths["youtube_strategy_hints"], index=False)
    if not youtube.empty:
        mark_research_sources_processed("youtube", root=root)
    return CommandResult(paths=paths, summary={"rows": int(len(normalized)), "sources": int(len(youtube))})


def _extract_markdown_rows(path: Path, source_row: pd.Series) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    lines = [line.rstrip() for line in text.splitlines()]
    rows: list[dict[str, object]] = []
    current_url = ""
    current_title = ""
    current_why = ""
    current_implications: list[str] = []
    row_counter = 0

    def flush() -> None:
        nonlocal row_counter, current_why, current_implications, current_url, current_title
        if not current_title and not current_implications and not current_why:
            return
        derived = _derive_rows(
            title=current_title,
            url=current_url,
            why=current_why,
            implications=current_implications,
            source_row=source_row,
            evidence_path=path,
        )
        for item in derived:
            row_counter += 1
            item["row_id"] = f"{source_row['source_id']}_{row_counter:04d}"
            rows.append(item)
        current_why = ""
        current_implications = []

    in_repo_implication = False
    in_why = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("### ") or stripped.startswith("#### "):
            if current_title:
                flush()
            current_title = stripped.lstrip("#").strip()
            current_url = ""
            in_repo_implication = False
            in_why = False
            continue
        if stripped.startswith("URL:"):
            current_url = stripped.split("URL:", 1)[1].strip().strip("<>")
            continue
        if stripped == "Why it matters:":
            in_why = True
            in_repo_implication = False
            continue
        if stripped == "Repo implication:":
            in_repo_implication = True
            in_why = False
            continue
        if stripped.startswith("### ") or stripped.startswith("## "):
            in_repo_implication = False
            in_why = False
        if in_why and stripped.startswith("- "):
            current_why = f"{current_why} {stripped[2:].strip()}".strip()
        if in_repo_implication and stripped.startswith("- "):
            current_implications.append(stripped[2:].strip())
    if current_title:
        flush()
    return rows


def _derive_rows(
    *,
    title: str,
    url: str,
    why: str,
    implications: list[str],
    source_row: pd.Series,
    evidence_path: Path,
) -> list[dict[str, object]]:
    if not re.match(r"^https?://", str(url or "").strip(), flags=re.IGNORECASE):
        return []
    rows: list[dict[str, object]] = []
    base = {
        "strategy_family": "",
        "mode_preference": "",
        "entry_logic": "",
        "exit_logic": "",
        "risk_rule": "",
        "feature_name": "",
        "pair_filter": "",
        "timeframe_preference": "",
        "regime_condition": "",
        "anti_pattern": "",
        "execution_warning": "",
        "confidence": float(source_row.get("confidence", 0.5) or 0.5),
        "source_id": str(source_row.get("source_id", "")),
        "source_type": str(source_row.get("source_type", "")),
        "source_title": str(source_row.get("title", title)),
        "evidence_path": f"{evidence_path}#{title}",
        "review_status": str(source_row.get("review_status", "unreviewed")),
        "notes": url,
    }
    lowered = f"{title} {why} {' '.join(implications)}".lower()
    if any(token in lowered for token in ["copula", "cointegration", "kalman", "ou", "zscore", "statistical arbitrage", "pair trading"]):
        rows.append(
            {
                **base,
                "row_type": "strategy_hint",
                "strategy_family": _strategy_family(lowered),
                "entry_logic": _first_match(implications, ["use", "compare", "capture", "prioritize"]) or why,
                "notes": f"{url};title={title}",
            }
        )
    if any(token in lowered for token in ["risk", "slippage", "drawdown", "cost", "paper trading", "forward testing"]):
        rows.append(
            {
                **base,
                "row_type": "risk_prior",
                "risk_rule": _first_match(implications, ["do not", "keep", "supports", "warns"]) or why,
                "notes": f"{url};title={title}",
            }
        )
    for feature in ["cointegration", "adf", "zero crossings", "hurst", "half-life", "ecm", "copula", "funding", "liquidity"]:
        if feature in lowered:
            rows.append(
                {
                    **base,
                    "row_type": "feature_idea",
                    "feature_name": feature.replace(" ", "_"),
                    "notes": f"{url};title={title}",
                }
            )
    if any(token in lowered for token in ["dynamic", "static", "ou", "copula"]):
        rows.append(
            {
                **base,
                "row_type": "mode_preference",
                "mode_preference": _mode_preference(lowered),
                "notes": f"{url};title={title}",
            }
        )
    if any(token in lowered for token in ["5-minute", "5 minute", "daily", "timeframe"]):
        rows.append(
            {
                **base,
                "row_type": "regime_condition",
                "timeframe_preference": "5m_and_daily",
                "regime_condition": why or "timeframe-aware signal",
                "notes": f"{url};title={title}",
            }
        )
    if any("do not" in item.lower() or "warn" in item.lower() or "fail" in item.lower() for item in implications):
        rows.append(
            {
                **base,
                "row_type": "anti_pattern",
                "anti_pattern": _first_match(implications, ["do not", "warn", "fail"]) or why,
                "notes": f"{url};title={title}",
            }
        )
    if any(token in lowered for token in ["dydx", "execution", "api", "automation", "telegram", "submission"]):
        rows.append(
            {
                **base,
                "row_type": "execution_warning",
                "execution_warning": _first_match(implications, ["prioritize", "keep", "do not", "warn"]) or why,
                "notes": f"{url};title={title}",
            }
        )
    return rows


def _strategy_family(lowered: str) -> str:
    for token, name in [
        ("copula", "copula"),
        ("cointegration", "cointegration"),
        ("kalman", "kalman"),
        ("ou", "ou"),
        ("zscore", "zscore"),
        ("pair trading", "pair_trading"),
    ]:
        if _contains_concept(lowered, token):
            return name
    return "research"


def _mode_preference(lowered: str) -> str:
    parts = [token for token in ["dynamic", "static", "ou", "copula"] if _contains_concept(lowered, token)]
    return ";".join(parts) if parts else ""


def _contains_concept(text: str, concept: str) -> bool:
    if concept == "ou":
        return bool(re.search(r"\b(?:ou|ornstein[- ]uhlenbeck)\b", text, flags=re.IGNORECASE))
    return bool(re.search(rf"\b{re.escape(concept)}\b", text, flags=re.IGNORECASE))


def _first_match(lines: list[str], tokens: list[str]) -> str:
    for token in tokens:
        for line in lines:
            if token in line.lower():
                return line
    return lines[0] if lines else ""
