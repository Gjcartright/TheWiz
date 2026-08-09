from __future__ import annotations

import json

import pandas as pd

from quant_platform.rl.features import (
    FEATURE_COLUMNS,
    RL_POLICY_SCHEMA_VERSION,
    WIZARD_DASHBOARD_NON_FEATURE_COLUMNS,
    attach_copula_dashboard_features,
    build_rl_feature_frame,
    write_feature_schema,
)


def _candidate(timestamp: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair": "ARB-USD/DOGE-USD",
                "timeframe": "Daily",
                "feature_timestamp": timestamp,
            }
        ]
    )


def _journal() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair": "ARB-USD/DOGE-USD",
                "timeframe": "Daily",
                "capture_timestamp_utc": "2026-07-15T09:00:00Z",
                "copula_x_given_y_pct": 99.3,
                "copula_y_given_x_pct": 0.1,
                "copula_probability_gap_pct": 99.2,
                "copula_best_fit": "studentt",
                "johansen_badge_state": "not_confirmed",
                "engle_granger_badge_state": "trending",
                "copula_signal_status": "strong_asymmetric_dislocation",
                "copula_journal_status": "research_only",
                "copula_execution_blockers": "engle_granger_trending",
                "return_total": 999.0,
                "sharpe": 99.0,
                "evidence_path": "reports/active/wizard_research_journal.csv",
            },
            {
                "pair": "ARB-USD/DOGE-USD",
                "timeframe": "Daily",
                "capture_timestamp_utc": "2026-07-15T11:00:00Z",
                "copula_x_given_y_pct": 1.0,
                "copula_y_given_x_pct": 99.0,
                "evidence_path": "future.csv",
            },
        ]
    )


def test_copula_features_use_latest_snapshot_known_at_entry_only():
    enriched, audit = attach_copula_dashboard_features(_candidate("2026-07-15T10:00:00Z"), _journal())

    assert audit.loc[0, "join_status"] == "attached"
    assert audit.loc[0, "uses_future_data"] == False
    assert enriched.loc[0, "wizard_copula_available"] == 1.0
    assert enriched.loc[0, "wizard_copula_x_given_y"] == 0.993
    assert enriched.loc[0, "wizard_copula_y_given_x"] == 0.001
    assert enriched.loc[0, "wizard_copula_engle_granger_trending"] == 1.0
    assert enriched.loc[0, "wizard_copula_execution_blocked"] == 1.0
    assert enriched.loc[0, "wizard_copula_snapshot_timestamp"].startswith("2026-07-15T09:00:00")


def test_copula_features_reject_stale_and_missing_timestamps():
    stale, stale_audit = attach_copula_dashboard_features(_candidate("2026-07-15T14:00:00Z"), _journal())
    missing, missing_audit = attach_copula_dashboard_features(_candidate("not-a-timestamp"), _journal())

    assert stale_audit.loc[0, "join_status"] == "stale_snapshot"
    assert stale.loc[0, "wizard_copula_available"] == 0.0
    assert stale.loc[0, "wizard_copula_stale_snapshot"] == 1.0
    assert missing_audit.loc[0, "join_status"] == "missing_feature_timestamp"
    assert missing.loc[0, "wizard_copula_available"] == 0.0


def test_dashboard_return_and_sharpe_never_become_rl_features(tmp_path):
    enriched, _ = attach_copula_dashboard_features(_candidate("2026-07-15T10:00:00Z"), _journal())
    features = build_rl_feature_frame(enriched)
    schema_path = write_feature_schema(tmp_path / "feature_schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert WIZARD_DASHBOARD_NON_FEATURE_COLUMNS.isdisjoint(FEATURE_COLUMNS)
    assert "return_total" not in features.columns
    assert "sharpe" not in features.columns
    assert schema["schema_version"] == RL_POLICY_SCHEMA_VERSION
    assert schema["copula_dashboard_policy"]["requires_snapshot_at_or_before_feature_timestamp"] is True
    assert schema["copula_dashboard_policy"]["excludes_dashboard_return_and_sharpe"] is True
