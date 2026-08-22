from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from quant_platform.evidence_registry import SourceRule, build_evidence_registry


def test_evidence_registry_uses_payload_time_and_never_grants_order_authority(tmp_path):
    source = tmp_path / "captures"
    source.mkdir()
    (source / "fresh.json").write_text(
        json.dumps({"captured_at": "2026-08-15T19:30:00+00:00", "payload": {"x": 1}}),
        encoding="utf-8",
    )
    (source / "._fresh.json").write_bytes(b"appledouble")
    rule = SourceRule(
        "test_l2",
        "hyperliquid",
        ("captures/*.json",),
        2.0,
        True,
        True,
        "test",
    )

    result = build_evidence_registry(
        root=tmp_path,
        now=datetime(2026, 8, 15, 20, 0, tzinfo=timezone.utc),
        rules=(rule,),
    )
    frame = pd.read_csv(result["registry"])

    assert len(frame) == 1
    assert frame.loc[0, "freshness_status"] == "fresh"
    assert bool(frame.loc[0, "acceptance_input_eligible"]) is True
    assert bool(frame.loc[0, "testnet_authority"]) is False
    assert bool(frame.loc[0, "live_authority"]) is False


def test_evidence_registry_marks_missing_market_timestamp_unknown(tmp_path):
    source = tmp_path / "captures"
    source.mkdir()
    (source / "unknown.json").write_text(json.dumps({"value": 1}), encoding="utf-8")
    rule = SourceRule("test", "apify", ("captures/*.json",), 24.0, True, False, "test")

    result = build_evidence_registry(root=tmp_path, rules=(rule,))
    frame = pd.read_csv(result["registry"])

    assert frame.loc[0, "point_in_time_status"] == "unknown"
    assert frame.loc[0, "freshness_status"] == "unknown"
    assert not bool(frame.loc[0, "acceptance_input_eligible"])
