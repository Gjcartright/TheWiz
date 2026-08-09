from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from quant_platform.orchestration.venue_capabilities import CAPABILITY_SCHEMA_VERSION, load_venue_policy


def test_verified_fresh_venue_capability_can_be_loaded(tmp_path):
    active = tmp_path / "reports" / "active"; active.mkdir(parents=True)
    now = datetime(2026, 7, 24, tzinfo=timezone.utc)
    pd.DataFrame([{"schema_version": CAPABILITY_SCHEMA_VERSION, "venue": "dydx", "account_eligible": True, "product_type": "perp", "supports_short_leg": True, "captured_at_utc": now.isoformat()}]).to_csv(active / "venue_account_capabilities.csv", index=False)
    policy = load_venue_policy(venue="dydx", root=tmp_path, now=now)
    assert policy.capability_status == "verified"
    assert policy.account_eligible


def test_stale_or_missing_capability_fails_closed(tmp_path):
    now = datetime(2026, 7, 24, tzinfo=timezone.utc)
    assert load_venue_policy(venue="dydx", root=tmp_path, now=now).capability_status == "missing"
    active = tmp_path / "reports" / "active"; active.mkdir(parents=True)
    pd.DataFrame([{"schema_version": CAPABILITY_SCHEMA_VERSION, "venue": "dydx", "account_eligible": True, "product_type": "perp", "supports_short_leg": True, "captured_at_utc": (now - timedelta(hours=3)).isoformat()}]).to_csv(active / "venue_account_capabilities.csv", index=False)
    assert load_venue_policy(venue="dydx", root=tmp_path, now=now).capability_status == "stale"
