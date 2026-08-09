from __future__ import annotations

from datetime import timedelta
import json

import numpy as np
import pandas as pd

from quant_platform.math_v2_acceptance import build_math_v2_acceptance
from quant_platform.orchestration.hyperliquid_learning_and_risk import (
    build_portfolio_critic,
    materialize_council_learning_dataset,
)
from quant_platform.orchestration.hyperliquid_research_validation import (
    build_hyperliquid_research_family_controls,
    build_hyperliquid_walkforward_validation,
)
from quant_platform.orchestration.hyperliquid_run_manifest import (
    build_hyperliquid_authority_state,
    build_hyperliquid_run_manifest,
)
from quant_platform.orchestration.teacher_adapters import build_teacher_evidence_adapters
from quant_platform.orchestration.teacher_control_plane import build_teacher_council_control_plane
from quant_platform.orchestration.teacher_evidence_materializer import materialize_teacher_evidence


def _write_fixture(root):
    active = root / "reports" / "active"
    history_dir = root / "data" / "raw" / "pair_details"
    active.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(29)
    rows = 701
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="1D", tz="UTC")
    log_y = 4.0 + np.cumsum(rng.normal(0.0, 0.012, rows))
    residual = np.zeros(rows)
    for index in range(1, rows):
        residual[index] = 0.82 * residual[index - 1] + rng.normal(0.0, 0.015)
    log_x = 0.4 + 1.1 * log_y + residual
    history = [
        {
            "timestamp": timestamp.isoformat(),
            "price_x": float(np.exp(x)),
            "price_y": float(np.exp(y)),
            "funding_x_bps": 0.2,
            "funding_y_bps": -0.1,
        }
        for timestamp, x, y in zip(timestamps, log_x, log_y)
    ]
    history_path = history_dir / "pair_aaa_bbb_hyperliquid_1d_derived_history.json"
    history_path.write_text(json.dumps({"history": history}), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "pair": "AAA-USD/BBB-USD",
                "asset_x": "AAA",
                "asset_y": "BBB",
                "venue": "hyperliquid",
                "interval": "1d",
                "history_path": str(history_path.relative_to(root)),
                "history_rows": rows,
                "history_ready": True,
            }
        ]
    ).to_csv(active / "hyperliquid_research_bundle.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "AAA-USD/BBB-USD",
                "fee_profile_id": "test-cost-v1",
                "leg_notional_usd": 1000,
                "taker_fee_bps": 1.0,
                "pair_one_way_slippage_bps": 0.5,
                "execution_risk_bps": 0.5,
                "cost_model_ready": True,
                "slippage_model_ready": True,
            }
        ]
    ).to_csv(active / "workflow_hyperliquid_pair_cost_model.csv", index=False)
    pd.DataFrame(
        [
            {"asset": "AAA", "tradable_perp": True},
            {"asset": "BBB", "tradable_perp": True},
        ]
    ).to_csv(active / "hyperliquid_testnet_market_inventory.csv", index=False)
    pd.DataFrame(
        [{"pair": "AAA-USD/BBB-USD", "exact_mode": "OU (Spread)"}]
    ).to_csv(active / "hyperliquid_wizard_hypothesis_queue.csv", index=False)
    return timestamps[-1].to_pydatetime()


def test_materializer_builds_seven_teachers_and_six_independent_critics(tmp_path):
    latest = _write_fixture(tmp_path)
    build_math_v2_acceptance(root=tmp_path)
    result = materialize_teacher_evidence(root=tmp_path, now=latest + timedelta(hours=1))
    teachers = pd.read_csv(result["teacher_inputs"])
    critics = pd.read_csv(result["critic_inputs"])

    assert result["status"] == "MATERIALIZED"
    assert len(teachers) == 7
    assert teachers["exact_mode"].nunique() == 7
    assert teachers["history_hash"].str.len().eq(64).all()
    assert teachers["point_in_time_status"].eq("confirmed").all()
    assert teachers["mode_fidelity_status"].eq("local_validated_estimator").all()
    assert teachers["wizard_nominated"].sum() == 1
    assert len(critics) == 6
    assert set(critics["critic_type"]) == {"dependency", "regime", "risk", "cost", "execution", "outcome"}
    actions = pd.read_csv(result["next_actions"])
    assert actions["priority"].is_monotonic_increasing
    assert "materialize_candidate_trade_rows_with_action_propensities" in set(actions["action"])

    adapter = build_teacher_evidence_adapters(root=tmp_path)
    assert adapter["status"] == "READY_FOR_COUNCIL"
    assert adapter["proposal_count"] == 7
    assert adapter["assessment_count"] == 6


def test_materializer_refuses_to_emit_without_machine_math_marker(tmp_path):
    latest = _write_fixture(tmp_path)
    result = materialize_teacher_evidence(root=tmp_path, now=latest + timedelta(hours=1))
    assert result["status"] == "BLOCKED"
    assert result["teacher_rows"] == 0
    assert result["critic_rows"] == 0


def test_run_manifest_fails_closed_when_slippage_or_candidate_gate_is_blocked(tmp_path):
    latest = _write_fixture(tmp_path)
    costs_path = tmp_path / "reports" / "active" / "workflow_hyperliquid_pair_cost_model.csv"
    costs = pd.read_csv(costs_path)
    costs["slippage_model_ready"] = False
    costs.to_csv(costs_path, index=False)
    queue_path = tmp_path / "reports" / "active" / "hyperliquid_wizard_hypothesis_queue.csv"
    queue = pd.read_csv(queue_path)
    queue["blocker"] = "insufficient_closed_trades_for_paid_proof"
    queue.to_csv(queue_path, index=False)

    manifest = build_hyperliquid_run_manifest(root=tmp_path, now=latest + timedelta(hours=1))

    assert manifest["status"] == "BLOCKED"
    assert manifest["cost_evidence_ready"] is False
    assert manifest["candidate_blockers_clear"] is False


def test_canonical_cycle_preserves_lineage_and_stays_fail_closed(tmp_path):
    latest = _write_fixture(tmp_path)
    now = latest + timedelta(hours=1)
    build_math_v2_acceptance(root=tmp_path)

    first = build_hyperliquid_run_manifest(root=tmp_path, now=now)
    costs_path = tmp_path / "reports" / "active" / "workflow_hyperliquid_pair_cost_model.csv"
    costs = pd.read_csv(costs_path)
    costs.insert(0, "candidate_set_id", first["candidate_set_id"])
    costs.insert(0, "run_id", first["run_id"])
    costs.to_csv(costs_path, index=False)
    second = build_hyperliquid_run_manifest(root=tmp_path, now=now)

    assert second["run_id"] == first["run_id"]
    assert second["candidate_set_id"] == first["candidate_set_id"]

    walkforward = build_hyperliquid_walkforward_validation(root=tmp_path)
    assert walkforward["fold_rows"] == 35
    assert walkforward["mode_rows"] == 7
    assert walkforward["trade_rows"] > 0
    summary = pd.read_csv(walkforward["summary"])
    selection = pd.read_csv(walkforward["selection"])
    assert summary["evidence_path"].eq(
        "reports/orchestration/teacher_council/walkforward_fold_evidence.csv"
    ).all()
    assert selection["evidence_path"].eq(
        "reports/orchestration/teacher_council/walkforward_mode_summary.csv"
    ).all()

    materialized = materialize_teacher_evidence(root=tmp_path, now=now)
    teachers = pd.read_csv(materialized["teacher_inputs"])
    assert teachers["run_id"].eq(first["run_id"]).all()
    assert teachers["candidate_set_id"].eq(first["candidate_set_id"]).all()
    assert teachers["mode_setup_identity"].nunique() == 7

    build_teacher_evidence_adapters(root=tmp_path)
    build_teacher_council_control_plane(root=tmp_path)
    learning = materialize_council_learning_dataset(root=tmp_path)
    training = pd.read_csv(learning["dataset"])
    assert not training.empty
    assert training["record_granularity"].eq("trade_entry").all()
    assert training["propensity_source"].eq("logged_deterministic_behavior_policy").all()
    assert training["action_propensity"].eq(1.0).all()
    assert training["behavior_policy_exploratory"].eq(False).all()

    build_teacher_council_control_plane(root=tmp_path)
    portfolio = build_portfolio_critic(root=tmp_path)
    authority = build_hyperliquid_authority_state(root=tmp_path, now=now)
    assert portfolio["status"] == "VETO"
    assert authority["identity_match"] is True
    assert authority["execution_allowed"] is False
    assert authority["status"] != "LIVE_READY"


def test_research_family_controls_pool_timeframes_and_spend_alpha_idempotently(tmp_path):
    council = tmp_path / "reports" / "orchestration" / "teacher_council"
    council.mkdir(parents=True)
    base = {
        "run_id": "run-1",
        "setup_identity": "setup-1",
        "pair": "BTC-USD/ETH-USD",
        "exact_mode": "Static Spread",
        "family_tests": 1,
        "false_discovery_rate": 0.10,
        "deflated_sharpe_status": "PASS",
        "parameter_stability_status": "PASS",
        "selection_status": "PASS",
        "selection_blocker": "",
    }
    pd.DataFrame(
        [{**base, "candidate_set_id": "set-1", "timeframe": "1d", "raw_pvalue": 0.04, "bh_qvalue": 0.04}]
    ).to_csv(council / "statistical_selection_controls.csv", index=False)
    pd.DataFrame(
        [{**base, "candidate_set_id": "set-1:aux:abc", "timeframe": "4h", "raw_pvalue": 0.90, "bh_qvalue": 0.90}]
    ).to_csv(council / "auxiliary_4h_walkforward_selection_controls.csv", index=False)

    first = build_hyperliquid_research_family_controls(root=tmp_path)
    controls = pd.read_csv(first["selection"])
    assert first["attempt_number"] == 1
    assert first["tests"] == 2
    assert first["alpha_spend_limit"] == 0.05
    assert controls.loc[controls["timeframe"] == "1d", "research_family_bh_qvalue"].iloc[0] == 0.08
    assert controls["selection_status"].eq("BLOCKED").all()

    repeated = build_hyperliquid_research_family_controls(root=tmp_path)
    assert repeated["attempt_number"] == 1
    assert len(pd.read_csv(repeated["registry"])) == 1

    auxiliary = pd.read_csv(council / "auxiliary_4h_walkforward_selection_controls.csv")
    auxiliary["raw_pvalue"] = 0.001
    auxiliary.to_csv(council / "auxiliary_4h_walkforward_selection_controls.csv", index=False)
    changed = build_hyperliquid_research_family_controls(root=tmp_path)
    assert changed["attempt_number"] == 2
    assert changed["alpha_spend_limit"] < first["alpha_spend_limit"]
    assert len(pd.read_csv(changed["registry"])) == 2
