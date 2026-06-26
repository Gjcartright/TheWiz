from __future__ import annotations

from pathlib import Path

import pandas as pd


def file_ready(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing_artifact"
    if path.is_file() and path.stat().st_size == 0:
        return False, "empty_artifact"
    return True, ""


def pair_has_local_acceptance(pair_universe: pd.DataFrame, pair_id: str = "") -> tuple[bool, str]:
    if pair_universe.empty:
        return False, "missing_pair_universe"
    frame = pair_universe
    if pair_id and "pair" in frame.columns:
        frame = frame[frame["pair"].astype(str).str.replace("/", "-").str.contains(pair_id.replace("/", "-"), case=False, regex=False)]
    promoted = frame[frame.get("decision_bucket", pd.Series(dtype=str)).astype(str) == "PROMOTE"]
    if promoted.empty:
        return False, "local_acceptance_not_promoted"
    if "acceptance_score" in promoted and (pd.to_numeric(promoted["acceptance_score"], errors="coerce") < 70).any():
        return False, "promotion_without_acceptance_score"
    return True, ""


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
