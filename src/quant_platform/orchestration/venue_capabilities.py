"""Versioned, read-only venue capability evidence for dynamic shadow routing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from quant_platform.orchestration.venue_gate import VenuePolicy


ROOT = Path(__file__).resolve().parents[3]
CAPABILITY_FRESHNESS_SECONDS = 2 * 60 * 60 + 30 * 60
CAPABILITY_FILENAME = "venue_account_capabilities.csv"
CAPABILITY_SCHEMA_VERSION = "venue_capabilities.v1"


def load_venue_policy(*, venue: str, root: Path = ROOT, now: datetime | None = None) -> VenuePolicy:
    """Load the latest verified capability row, otherwise return a fail-closed policy."""

    now = now or datetime.now(timezone.utc)
    path = root / "reports" / "active" / CAPABILITY_FILENAME
    relative_path = str(path.relative_to(root))
    if not path.exists():
        return VenuePolicy(venue=venue, capability_status="missing", capability_evidence_path=relative_path)
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return VenuePolicy(venue=venue, capability_status="invalid", capability_evidence_path=relative_path)
    required = {"schema_version", "venue", "account_eligible", "product_type", "supports_short_leg", "captured_at_utc"}
    if not required.issubset(frame.columns):
        return VenuePolicy(venue=venue, capability_status="invalid", capability_evidence_path=relative_path)
    rows = frame.loc[frame["venue"].astype(str).str.lower().eq(venue.lower())].copy()
    rows = rows.loc[rows["schema_version"].astype(str).eq(CAPABILITY_SCHEMA_VERSION)]
    if rows.empty:
        return VenuePolicy(venue=venue, capability_status="missing", capability_evidence_path=relative_path)
    rows["captured_at_utc"] = pd.to_datetime(rows["captured_at_utc"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["captured_at_utc"]).sort_values("captured_at_utc")
    if rows.empty:
        return VenuePolicy(venue=venue, capability_status="invalid", capability_evidence_path=relative_path)
    row = rows.iloc[-1]
    captured_at = row["captured_at_utc"].to_pydatetime()
    if captured_at < now - timedelta(seconds=CAPABILITY_FRESHNESS_SECONDS) or captured_at > now:
        return VenuePolicy(venue=venue, capability_status="stale", capability_evidence_path=relative_path)
    return VenuePolicy(
        venue=venue,
        account_eligible=_as_bool(row["account_eligible"]),
        product_type=str(row["product_type"]),
        supports_short_leg=_as_bool(row["supports_short_leg"]),
        capability_status="verified",
        capability_evidence_path=relative_path,
    )


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
