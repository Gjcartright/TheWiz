from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from quant_platform.orchestration.versioned_learning_comparison import (
    VERSION_LABELS,
    build_versioned_learning_comparison,
)


def _contract(*, execution_authority: bool = False) -> dict[str, object]:
    return {
        "schema_version": "versioned_learning_comparison.v1",
        "comparison_runtime": "shared_current_engine_feature_gated",
        "paper_ingestion_boundary": "2026-08-16T04:55:46+00:00",
        "minimum_prospective_shadow_outcomes": 100,
        "minimum_incremental_balanced_accuracy": 0.02,
        "versions": {
            "V1": {
                "parent_version": "",
                "version_role": "champion",
                "historical_code_ref": "deadbeef",
                "historical_boundary": "before_research_paper_ingestion",
                "paper_features_allowed": False,
                "paper_teacher_weight": 0.0,
                "shadow_only": True,
                "promotion_authority": False,
                "execution_authority": False,
                "notes": "pre-paper reference",
            },
            "V1.1": {
                "parent_version": "V1",
                "version_role": "challenger",
                "historical_code_ref": "WORKTREE",
                "historical_boundary": "paper_augmented_research_layer",
                "paper_features_allowed": True,
                "paper_teacher_weight": 0.0,
                "shadow_only": True,
                "promotion_authority": False,
                "execution_authority": execution_authority,
                "notes": "paper challenger",
            },
        },
    }


def _write_csv(root: Path, relative: str, rows: list[dict[str, object]]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _fixture_root(tmp_path: Path) -> Path:
    config_path = tmp_path / "config/research_version_contracts.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(_contract(), sort_keys=False), encoding="utf-8")

    _write_csv(
        tmp_path,
        "reports/dashboard/teacher_council_decisions.csv",
        [
            {
                "context_id": "context-1",
                "pair": "BTC-USD/ETH-USD",
                "timeframe": "1d",
                "selected_mode": "OU Spread",
                "action": "abstain",
                "status": "BLOCKED",
                "blocker_codes": "local_gate_failed",
                "evidence_paths": "evidence/teacher.csv",
            }
        ],
    )
    _write_csv(
        tmp_path,
        "reports/orchestration/teacher_council/student_training_readiness.csv",
        [
            {"scope": "supervised_student", "check": "all", "status": "PASS"},
            {"scope": "contextual_bandit", "check": "propensity", "status": "BLOCKED"},
        ],
    )
    _write_csv(
        tmp_path,
        "reports/orchestration/teacher_council/teacher_proposals.csv",
        [{"proposal_id": "proposal-1"}],
    )
    _write_csv(
        tmp_path,
        "reports/agents/rl_ideas.csv",
        [
            {
                "idea_id": "idea-1",
                "pair": "BTC-USD-ETH-USD",
                "timeframe": "1d",
                "strategy": "OU Spread",
                "confidence_score": 0.6,
                "status": "candidate",
                "evidence_path": "evidence/rl.csv",
            }
        ],
    )
    reproduction_dir = tmp_path / "reports/research/papers/reproductions"
    _write_csv(
        tmp_path,
        "reports/research/papers/reproductions/crypto_characteristic_candidates.csv",
        [
            {
                "pair": "SOL-USD-WLD-USD",
                "discovery_score": 0.7,
                "point_in_time_vintage": False,
                "decision_bucket": "RESEARCH_ONLY",
                "evidence_path": "evidence/paper.csv",
            }
        ],
    )
    _write_csv(
        tmp_path,
        "reports/research/papers/reproductions/paper_reproduction_gate.csv",
        [
            {
                "experiment": "panel",
                "exact_paper_reproduction": False,
                "signal_eligible": False,
                "execution_eligible": False,
            }
        ],
    )
    _write_csv(
        tmp_path,
        "reports/research/papers/reproductions/paper_reproduction_data_audit.csv",
        [{"asset": "BTC", "point_in_time_vintage": False}],
    )
    _write_csv(
        tmp_path,
        "reports/research/papers/reproductions/panel_residual_walkforward_metrics.csv",
        [
            {
                "model": "pooled_logistic",
                "population": "whole_asset_holdout",
                "balanced_accuracy": 0.50,
            },
            {
                "model": "mean_reversion_sign",
                "population": "whole_asset_holdout",
                "balanced_accuracy": 0.53,
            },
        ],
    )
    for filename in (
        "panel_residual_walkforward_predictions.csv",
        "one_sided_stability_diagnostics.csv",
        "dynamic_no_trade_band_results.csv",
    ):
        (reproduction_dir / filename).write_text("placeholder\n1\n", encoding="utf-8")
    return tmp_path


def test_builds_paired_v1_and_v1_1_comparisons_with_zero_authority(tmp_path: Path):
    root = _fixture_root(tmp_path)

    result = build_versioned_learning_comparison(root=root)

    contracts = pd.read_csv(result.paths["contracts_csv"])
    features = pd.read_csv(result.paths["features"])
    teacher = pd.read_csv(result.paths["teacher"])
    rl = pd.read_csv(result.paths["rl"])
    shadow = pd.read_csv(result.paths["shadow"])
    promotion = pd.read_csv(result.paths["promotion"])

    assert tuple(contracts["research_version"]) == VERSION_LABELS
    assert contracts.loc[contracts["research_version"].eq("V1.1"), "parent_version"].item() == "V1"
    challenger_contract = contracts.loc[contracts["research_version"].eq("V1.1")].iloc[0]
    assert not bool(challenger_contract["historical_code_ref_verified"])
    assert bool(challenger_contract["runtime_materialized"])
    for frame in (teacher, rl, shadow):
        assert set(frame["research_version"]) == set(VERSION_LABELS)
        assert not frame["execution_authority"].any()
        assert not frame["promotion_authority"].any()
    paired = shadow.groupby("comparison_key")["research_version"].agg(set)
    assert paired.map(lambda labels: labels == set(VERSION_LABELS)).all()
    assert shadow["order_action"].eq("abstain").all()
    assert not shadow["outcome_known"].any()
    assert not promotion["promotion_authority"].any()
    assert result.summary["v1_1_promoted"] is False

    v1_paper = features[features["research_version"].eq("V1") & features["is_paper_feature"]]
    challenger_paper = features[
        features["research_version"].eq("V1.1") & features["is_paper_feature"]
    ]
    assert not v1_paper["feature_available"].any()
    assert challenger_paper["feature_available"].all()
    assert not challenger_paper["student_training_eligible"].any()
    assert not challenger_paper["rl_observation_eligible"].any()
    assert not challenger_paper["reward_eligible"].any()
    assert not challenger_paper["teacher_vote_eligible"].any()
    base_exact_mode = features[
        features["research_version"].eq("V1") & features["feature_name"].eq("exact_mode_features")
    ].iloc[0]
    assert bool(base_exact_mode["student_training_eligible"])
    assert not bool(base_exact_mode["rl_observation_eligible"])
    assert not bool(base_exact_mode["reward_eligible"])
    assert bool(base_exact_mode["teacher_vote_eligible"])


def test_memory_is_version_partitioned_and_deterministic(tmp_path: Path):
    root = _fixture_root(tmp_path)
    first = build_versioned_learning_comparison(root=root)
    first_rows = [json.loads(line) for line in first.paths["memory"].read_text().splitlines()]
    second = build_versioned_learning_comparison(root=root)
    second_rows = [json.loads(line) for line in second.paths["memory"].read_text().splitlines()]

    assert len(first_rows) == first.summary["shadow_comparison_rows"]
    assert {row["research_version"] for row in first_rows} == set(VERSION_LABELS)
    assert all(row["memory_partition"] == row["research_version"] for row in first_rows)
    assert not any(row["cross_version_memory_pooling"] for row in first_rows)
    assert len({row["event_id"] for row in first_rows}) == len(first_rows)
    assert [row["event_id"] for row in first_rows] == [row["event_id"] for row in second_rows]


def test_red_team_preserves_paper_failures(tmp_path: Path):
    result = build_versioned_learning_comparison(root=_fixture_root(tmp_path))
    red_team = pd.read_csv(result.paths["red_team"])
    challenger = red_team[red_team["research_version"].eq("V1.1")].set_index("check")

    assert challenger.loc["paper_reproduction_signal_gate", "status"] == "FAIL"
    assert challenger.loc["paper_point_in_time_vintages", "status"] == "FAIL"
    assert challenger.loc["paper_incremental_accuracy", "status"] == "FAIL"
    assert challenger.loc["contextual_bandit_support", "status"] == "BLOCKED"
    assert challenger.loc["challenger_promotion_firewall", "status"] == "PASS"


def test_missing_sources_fail_closed_but_still_emit_paired_rows(tmp_path: Path):
    config_path = tmp_path / "config/research_version_contracts.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(_contract(), sort_keys=False), encoding="utf-8")

    result = build_versioned_learning_comparison(root=tmp_path)
    teacher = pd.read_csv(result.paths["teacher"])
    shadow = pd.read_csv(result.paths["shadow"])

    assert len(teacher) == 2
    assert teacher["pair"].eq("SOURCE_MISSING").all()
    assert set(shadow["research_version"]) == set(VERSION_LABELS)
    assert shadow["order_action"].eq("abstain").all()
    assert not shadow["execution_authority"].any()


def test_contract_rejects_v1_1_execution_authority(tmp_path: Path):
    config_path = tmp_path / "config/research_version_contracts.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(_contract(execution_authority=True), sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="execution_authority"):
        build_versioned_learning_comparison(root=tmp_path)
