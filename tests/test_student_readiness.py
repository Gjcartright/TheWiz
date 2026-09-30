from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from quant_platform.orchestration.student_readiness import (
    STUDENT_TRAINING_COLUMNS,
    StudentReadinessPolicy,
    audit_student_training_dataset,
    write_student_training_readiness,
)
from quant_platform.orchestration.teacher_contracts import ExactMode
from quant_platform.statistics.math_v2 import MATH_VERSION


NOW = datetime(2026, 8, 6, 18, tzinfo=timezone.utc)


def _dataset() -> pd.DataFrame:
    rows = []
    for index, mode in enumerate(ExactMode):
        rows.append(
            {
                "training_event_id": f"event-{index}",
                "context_id": f"context-{index}",
                "candidate_id": f"candidate-{index}",
                "pair": f"ASSET{index % 3}-USD/OTHER{index % 3}-USD",
                "timeframe": "1d" if index % 2 == 0 else "2h",
                "exact_mode": mode.value,
                "proposed_action": "short_x_long_y",
                "feature_timestamp": NOW - timedelta(hours=2),
                "label_timestamp": NOW,
                "point_in_time_status": "confirmed",
                "math_version": MATH_VERSION,
                "source_system": "hyperliquid_local_replay",
                "label_source": "backtest_label",
                "uses_wizard_as_label": False,
                "uses_dashboard_hindsight": False,
                "profit_after_cost": 0.01 if index % 2 == 0 else -0.01,
                "good_trade": index % 2,
                "max_adverse_excursion": 0.01,
                "max_favorable_excursion": 0.02,
                "hold_bars": 5,
                "exit_reason": "mean_reversion",
                "action_propensity": 0.5,
                "evidence_path": f"reports/trades/{index}.csv",
            }
        )
    return pd.DataFrame(rows, columns=STUDENT_TRAINING_COLUMNS)


def test_valid_student_dataset_passes_supervised_and_bandit_checks():
    audit = audit_student_training_dataset(
        _dataset(),
        policy=StudentReadinessPolicy(min_rows=7, min_pairs=3, min_timeframes=2, min_rows_per_mode=1, max_pair_share=0.5, max_mode_share=0.2),
    )
    assert set(audit["status"]) == {"PASS"}


def test_leakage_wizard_labels_math_v1_and_missing_propensity_are_blocked():
    dataset = _dataset()
    dataset.loc[0, "label_timestamp"] = NOW - timedelta(hours=3)
    dataset.loc[1, "uses_wizard_as_label"] = True
    dataset.loc[2, "uses_dashboard_hindsight"] = True
    dataset.loc[3, "math_version"] = "math-v1"
    dataset.loc[4, "action_propensity"] = None
    audit = audit_student_training_dataset(
        dataset,
        policy=StudentReadinessPolicy(min_rows=7, min_pairs=3, min_timeframes=2, min_rows_per_mode=1, max_pair_share=0.5, max_mode_share=0.2),
    )
    blockers = set(audit.loc[audit["status"] == "BLOCKED", "blocker"])
    assert "future_leakage_or_invalid_timestamp" in blockers
    assert "wizard_hindsight_used_as_label" in blockers
    assert "dashboard_hindsight_feature_present" in blockers
    assert "unverified_math_version" in blockers
    assert "bandit_off_policy_evaluation_not_ready" in blockers


def test_deterministic_propensity_is_logged_but_not_bandit_ready():
    dataset = _dataset()
    dataset["propensity_source"] = "logged_deterministic_behavior_policy"
    dataset["behavior_policy_exploratory"] = False

    audit = audit_student_training_dataset(
        dataset,
        policy=StudentReadinessPolicy(min_rows=7, min_pairs=3, min_timeframes=2, min_rows_per_mode=1, max_pair_share=0.5, max_mode_share=0.2),
    )

    blockers = set(audit.loc[audit["status"] == "BLOCKED", "blocker"])
    assert "synthetic_or_missing_propensity_lineage" in blockers
    assert "deterministic_behavior_policy_has_no_counterfactual_support" in blockers


def test_inverse_frequency_weights_control_effective_mode_concentration():
    dataset = pd.concat([_dataset(), pd.concat([_dataset().iloc[[0]]] * 20, ignore_index=True)], ignore_index=True)
    counts = dataset["exact_mode"].value_counts()
    dataset["sample_weight"] = dataset["exact_mode"].map(lambda value: 1.0 / counts[value])

    audit = audit_student_training_dataset(
        dataset,
        policy=StudentReadinessPolicy(min_rows=7, min_pairs=3, min_timeframes=2, min_rows_per_mode=1, max_pair_share=0.5, max_mode_share=0.2),
    )

    mode_check = audit.loc[audit["check"] == "mode_concentration"].iloc[0]
    assert mode_check["status"] == "PASS"
    assert "raw=" in str(mode_check["observed"])


def test_missing_training_dataset_writes_explicit_blocker(tmp_path):
    result = write_student_training_readiness(root=tmp_path)
    audit = pd.read_csv(result["audit"])
    assert result["supervised_status"] == "BLOCKED"
    assert audit.loc[0, "blocker"] == "missing_student_training_dataset"
