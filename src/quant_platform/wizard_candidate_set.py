from __future__ import annotations

from hashlib import sha256

import pandas as pd


def wizard_hyperliquid_candidate_set_id(frame: pd.DataFrame) -> str:
    """Return one stable identity for a frozen Wizard-to-Hyperliquid queue."""

    if frame.empty:
        return ""
    material: set[str] = set()
    for _, row in frame.iterrows():
        config_hash = _text(row.get("candidate_config_hash", ""))
        interval = _text(row.get("local_interval", row.get("interval", row.get("timeframe", ""))))
        # Uppercase M denotes months; lowercasing it would alias minute rows.
        interval_key = interval if interval.endswith("M") and interval[:-1].isdigit() else interval.lower()
        identity = config_hash or "|".join(
            [
                _text(row.get("pair", "")).upper(),
                interval_key,
                _text(row.get("exact_mode", "")).lower(),
            ]
        )
        if identity.strip("|"):
            material.add(identity)
    if not material:
        return ""
    digest = sha256("\n".join(sorted(material)).encode("utf-8")).hexdigest()
    return f"whlset_{digest[:20]}"


def _text(value: object) -> str:
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return ""
    return str(value).strip()
