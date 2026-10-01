from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.rl.base_rl_lane import _gate_ready
from quant_platform.rl.brain_contract import (
    BRAIN_MEMORY_AGENT,
    _overall_readiness_row,
    valid_memory_event,
)
from quant_platform.rl.quantization import export_rl_policy
from quant_platform.rl.rl_idea_engine import run_rl_idea_scout
from quant_platform.rl.rl_learning_agent import _select_policy_candidate
from quant_platform.rl.rl_strategy_selector import approved_strategy_choices


def _write_idea_dataset(root, rows):
    path = root / "data" / "ml" / "trade_training_dataset.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_persisted_false_does_not_open_base_or_overall_readiness(tmp_path):
    readiness = pd.DataFrame([{"gate": "paper_execution_gate", "ready": "False "}])
    assert _gate_ready(readiness, "paper_execution_gate") is False

    overall = _overall_readiness_row(
        tmp_path,
        {"ready": "True", "status": "ready"},
        {"ready": "False ", "status": "blocked"},
        {"ready": "True", "status": "ready"},
    )
    assert overall["ready"] is False
    assert overall["status"] == "blocked"


def test_memory_outcome_known_false_rejects_nonempty_label():
    event = {
        "timestamp": "2026-06-28T00:00:00Z",
        "agent": BRAIN_MEMORY_AGENT,
        "task_id": "brain:001",
        "task_type": "run_brain_cycle",
        "cycle_id": "20260628",
        "pair": "ETH-BTC",
        "outcome_known": "False ",
        "outcome_label": "passed",
        "blocker": "",
        "next_step": "review",
    }
    assert valid_memory_event(event) == (False, "outcome_label_without_outcome_known")


def test_persisted_false_does_not_approve_strategy_or_policy():
    choices = pd.DataFrame(
        [
            {"pair": "blocked", "accepted": "False "},
            {"pair": "eligible", "accepted": "True"},
        ]
    )
    assert approved_strategy_choices(choices)["pair"].tolist() == ["eligible"]

    policies = pd.DataFrame(
        [
            {"policy_name": "blocked", "validation_eligible": "False ", "validation_gate_count": 10,
             "profit_factor": 10.0, "sharpe": 10.0, "max_drawdown": 0.0, "trades": 100},
            {"policy_name": "eligible", "validation_eligible": "True", "validation_gate_count": 1,
             "profit_factor": 1.0, "sharpe": 1.0, "max_drawdown": 0.1, "trades": 10},
        ]
    )
    assert _select_policy_candidate(policies)["policy_name"].iloc[0] == "eligible"


def test_persisted_false_acceptance_remains_false_in_export_report(tmp_path):
    path = tmp_path / "reports" / "rl" / "rl_acceptance_report.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"accepted": "False "}]).to_csv(path, index=False)

    result = export_rl_policy(root=tmp_path)

    assert result.summary["accepted"] is False
    assert result.summary["exported"] is False
    assert result.summary["blocker"] == "rl_acceptance_not_passed"


@pytest.mark.parametrize(
    ("extra", "blocker"),
    [
        ({}, "missing_out_of_sample_prediction_score"),
        ({"predicted_return": 0.25}, "missing_out_of_sample_prediction_provenance"),
    ],
)
def test_idea_scout_blocks_realized_only_or_unproven_predictions(tmp_path, extra, blocker):
    _write_idea_dataset(
        tmp_path,
        [{"pair": "BTC-USD/ETH-USD", "timeframe": "1d", "strategy": "Static Spread",
          "profit_after_cost": 0.8, **extra}],
    )

    result = run_rl_idea_scout(root=tmp_path)

    assert result.summary["ideas"] == 0
    assert result.summary["blocker"] == blocker
    assert pd.read_csv(result.paths["rl_ideas"]).empty
    assert pd.read_csv(result.paths["rl_idea_summary"])["blocker"].iloc[0] == blocker


def test_idea_scout_ranks_only_forward_oos_predictions(tmp_path):
    _write_idea_dataset(
        tmp_path,
        [
            {"pair": "excluded", "timeframe": "1d", "strategy": "Static Spread",
             "profit_after_cost": 1.0, "predicted_return": 1.0, "prediction_is_oos": "False ",
             "predicted_risk": 0.01},
            {"pair": "realized_winner", "timeframe": "1d", "strategy": "Static Spread",
             "profit_after_cost": 0.8, "predicted_return": 0.1, "prediction_is_oos": "True",
             "predicted_risk": 0.02},
            {"pair": "forward_winner", "timeframe": "1d", "strategy": "Static Spread",
             "profit_after_cost": -0.5, "predicted_return": 0.25, "prediction_is_oos": "True",
             "predicted_risk": 0.03, "predicted_trade_count": 7},
        ],
    )

    result = run_rl_idea_scout(root=tmp_path, top_ideas=1)
    ideas = pd.read_csv(result.paths["rl_ideas"])

    assert result.summary["blocker"] == ""
    assert len(ideas) == 1
    assert ideas.loc[0, "pair"] == "forward_winner"
    assert ideas.loc[0, "expected_return"] == pytest.approx(0.25)
    assert ideas.loc[0, "risk_proxy"] == pytest.approx(0.03)
    assert ideas.loc[0, "expected_trade_count"] == 7
    assert ideas.loc[0, "source"] == "rl_oos_prediction_hypothesis"
    assert ideas.loc[0, "status"] == "hypothesis_only"


def test_idea_scout_requires_finite_score_and_accepts_held_out_split(tmp_path):
    _write_idea_dataset(
        tmp_path,
        [
            {"pair": "in_sample", "predicted_return": 0.9, "evaluation_split": "diagnostic_full_sample"},
            {"pair": "nonfinite", "predicted_return": "inf", "evaluation_split": "held_out_test"},
            {"pair": "held_out", "predicted_return": 0.2, "evaluation_split": "held_out_test"},
        ],
    )

    result = run_rl_idea_scout(root=tmp_path, top_ideas=3)
    ideas = pd.read_csv(result.paths["rl_ideas"])

    assert result.summary["ideas"] == 1
    assert ideas["pair"].tolist() == ["held_out"]
