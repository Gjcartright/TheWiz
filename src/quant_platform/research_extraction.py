from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

from pathlib import Path
from typing import Any

import pandas as pd

ROW_TYPES = {
    "strategy_hint",
    "feature_idea",
    "risk_prior",
    "mode_preference",
    "regime_condition",
    "pair_filter",
    "anti_pattern",
    "execution_warning",
}

EXTRACTION_COLUMNS = [
    "row_id",
    "row_type",
    "strategy_family",
    "mode_preference",
    "entry_logic",
    "exit_logic",
    "risk_rule",
    "feature_name",
    "pair_filter",
    "timeframe_preference",
    "regime_condition",
    "anti_pattern",
    "execution_warning",
    "confidence",
    "source_id",
    "source_type",
    "source_title",
    "evidence_path",
    "evidence_source",
    "point_in_time_status",
    "live_signal_eligible",
    "promotion_authority",
    "source_url",
    "review_status",
    "notes",
]


def normalize_extraction_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=EXTRACTION_COLUMNS)
    normalized = frame.reindex(columns=EXTRACTION_COLUMNS, fill_value="").fillna("")
    normalized["row_type"] = normalized["row_type"].astype(str).str.strip()
    invalid = sorted(set(normalized.loc[~normalized["row_type"].isin(ROW_TYPES), "row_type"].astype(str)))
    if invalid:
        raise ValueError(f"unsupported_research_row_types:{','.join(invalid)}")
    normalized["confidence"] = pd.to_numeric(normalized["confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return normalized


def write_extraction_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_extraction_rows(frame.to_dict("records"))
    atomic_write_csv(normalized, path, index=False)
    return path
