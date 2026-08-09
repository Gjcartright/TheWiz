from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.research_extraction import EXTRACTION_COLUMNS, normalize_extraction_rows


KNOWLEDGE_OUTPUTS = {
    "research_rules": "research_rules.csv",
    "research_features": "research_features.csv",
    "research_risk_priors": "research_risk_priors.csv",
    "research_strategy_hints": "research_strategy_hints.csv",
    "research_mode_preferences": "research_mode_preferences.csv",
}


def knowledge_base(root: Path = ROOT) -> Path:
    return root / "data" / "processed" / "research_knowledge"


def build_research_knowledge_store(root: Path = ROOT) -> CommandResult:
    base = knowledge_base(root)
    base.mkdir(parents=True, exist_ok=True)
    source_files = sorted(base.glob("*research_rows.csv"))
    frames = []
    seen: set[Path] = set()
    for path in source_files:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            frame = pd.read_csv(path).fillna("")
        except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
            continue
        if not frame.empty:
            frames.append(frame)
    combined = normalize_extraction_rows(pd.concat(frames, ignore_index=True).to_dict("records")) if frames else pd.DataFrame(columns=EXTRACTION_COLUMNS)
    if not combined.empty:
        populated_ids = combined["row_id"].astype(str).str.strip().ne("")
        with_ids = combined[populated_ids].drop_duplicates(subset=["row_id"], keep="last")
        without_ids = combined[~populated_ids].drop_duplicates(keep="last")
        combined = pd.concat([with_ids, without_ids], ignore_index=True)
    outputs = {
        "research_rules": combined[combined["row_type"].isin(["risk_prior", "regime_condition", "pair_filter", "anti_pattern", "execution_warning"])].copy(),
        "research_features": combined[combined["row_type"] == "feature_idea"].copy(),
        "research_risk_priors": combined[combined["row_type"] == "risk_prior"].copy(),
        "research_strategy_hints": combined[combined["row_type"] == "strategy_hint"].copy(),
        "research_mode_preferences": combined[combined["row_type"] == "mode_preference"].copy(),
    }
    written: dict[str, Path] = {}
    for key, filename in KNOWLEDGE_OUTPUTS.items():
        output = base / filename
        outputs[key].to_csv(output, index=False)
        written[key] = output
    summary_csv = root / "reports" / "research" / "research_knowledge_summary.csv"
    summary_md = root / "reports" / "research" / "research_knowledge_summary.md"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [{"table": key, "rows": int(len(frame)), "path": str(written[key])} for key, frame in outputs.items()],
        columns=["table", "rows", "path"],
    )
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(_summary_markdown(summary), encoding="utf-8")
    source_summary_path = root / "reports" / "research" / "research_knowledge_source_summary.csv"
    source_summary = _source_summary(combined)
    source_summary.to_csv(source_summary_path, index=False)
    return CommandResult(
        paths={
            **written,
            "research_knowledge_summary": summary_csv,
            "research_knowledge_summary_md": summary_md,
            "research_knowledge_source_summary": source_summary_path,
        },
        summary={"tables": len(outputs), "rows": int(len(combined))},
    )


def research_knowledge_summary(root: Path = ROOT) -> CommandResult:
    return build_research_knowledge_store(root=root)


def _summary_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Research Knowledge Summary", ""]
    if frame.empty:
        lines.append("- no knowledge rows")
    else:
        for _, row in frame.iterrows():
            lines.append(f"- {row['table']}: {row['rows']} rows")
    return "\n".join(lines) + "\n"


def query_research_knowledge(
    *,
    root: Path = ROOT,
    source_type: str = "",
    strategy_family: str = "",
    row_types: tuple[str, ...] = (),
    text: str = "",
    limit: int = 100,
) -> pd.DataFrame:
    """Return research rows for agents without granting execution authority."""

    base = knowledge_base(root)
    frames: list[pd.DataFrame] = []
    for path in sorted(base.glob("*research_rows.csv")):
        try:
            frame = pd.read_csv(path).fillna("")
        except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=EXTRACTION_COLUMNS)
    result = normalize_extraction_rows(pd.concat(frames, ignore_index=True).to_dict("records"))
    if source_type:
        result = result[result["source_type"].astype(str).str.casefold().eq(source_type.casefold())]
    if strategy_family:
        result = result[result["strategy_family"].astype(str).str.casefold().eq(strategy_family.casefold())]
    if row_types:
        result = result[result["row_type"].astype(str).isin(row_types)]
    terms = [token.casefold() for token in text.split() if token.strip()]
    if terms:
        haystack = result.astype(str).agg(" ".join, axis=1).str.casefold()
        result = result[haystack.map(lambda value: all(term in value for term in terms))]
    return result.head(max(0, int(limit))).reset_index(drop=True)


def _source_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["source_type", "row_type", "rows", "live_signal_eligible_rows", "promotion_authorities"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for (source_type, row_type), group in frame.groupby(["source_type", "row_type"], dropna=False):
        live = group.get("live_signal_eligible", pd.Series(False, index=group.index)).map(_boolish)
        authorities = sorted(
            str(value)
            for value in group.get("promotion_authority", pd.Series(dtype=str)).astype(str).unique()
            if str(value).strip()
        )
        rows.append(
            {
                "source_type": source_type,
                "row_type": row_type,
                "rows": int(len(group)),
                "live_signal_eligible_rows": int(live.sum()),
                "promotion_authorities": ";".join(authorities),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _boolish(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "on"}
