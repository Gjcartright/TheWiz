from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.active_pipeline import (
    CommandResult,
    _model_gain_concentration,
    _model_gated_acceptance,
    _model_gated_comparison,
    _model_pair_concentration,
    _score_bucket_report,
)
from quant_platform.ml_filter import (
    expected_walkforward_prediction_membership,
    model_selection_leaderboard,
)
from quant_platform.orchestration.corrective_registered_learning import (
    _registered_rl_protocol,
    _registered_supervised_protocol,
    _registered_survivor_rl_attribution_frame,
    _selection_concentration_within_limit,
    build_registered_exact_mode_trade_dataset,
    latest_verified_registered_learning,
    run_registered_learning_research,
)
from quant_platform.orchestration.corrective_registered_learning_protocol import (
    build_registered_stage5_protocol,
    validate_registered_stage5_protocol,
)
from quant_platform.orchestration.corrective_registered_rerun import (
    CONCLUSION_SCHEMA_VERSION,
)
from quant_platform.orchestration.corrective_release_gates import (
    _validated_registered_learning,
)
from quant_platform.rl.rl_acceptance import return_summary, rl_acceptance_report
from quant_platform.rl.rl_learning_agent import _chronological_rl_partitions
from quant_platform.wizard_mode_replay import CANONICAL_WIZARD_MODES

NOW = datetime(2026, 8, 10, 1, tzinfo=UTC)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _payload_hash(payload: dict) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _conclusion_receipt(payload: dict) -> dict:
    core = {"schema_version": CONCLUSION_SCHEMA_VERSION, **payload}
    return {
        **core,
        "conclusion_id": "registeredconclusion_" + _payload_hash(core)[:20],
    }


def _stage5_protocol_fixture(root: Path, *, registered_at: datetime | None = None):
    sources = [
        (
            "agent_learning_governance",
            "src/quant_platform/orchestration/corrective_agent_governance.py",
        ),
        (
            "registered_learning_orchestrator",
            "src/quant_platform/orchestration/corrective_registered_learning.py",
        ),
        (
            "registered_stage4_contract",
            "src/quant_platform/orchestration/corrective_registered_rerun.py",
        ),
        (
            "registered_stage4_execution_validator",
            "src/quant_platform/orchestration/corrective_registered_rerun_executor.py",
        ),
        ("dataset_and_model_gate_pipeline", "src/quant_platform/active_pipeline.py"),
        ("exact_mode_schema", "src/quant_platform/wizard_mode_replay.py"),
        ("model_training_and_selection", "src/quant_platform/ml_filter.py"),
        (
            "registered_stage5_protocol_registry",
            "src/quant_platform/orchestration/corrective_registered_learning_protocol.py",
        ),
        ("rl_feature_construction", "src/quant_platform/rl/features.py"),
        (
            "rl_research_and_execution_simulator",
            "src/quant_platform/rl/rl_backtest.py",
        ),
        ("rl_acceptance", "src/quant_platform/rl/rl_acceptance.py"),
        (
            "rl_chronological_partitioning",
            "src/quant_platform/rl/rl_learning_agent.py",
        ),
    ]
    for role, relative in sources:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{role}\n", encoding="utf-8")
    _write_json(
        root / "config" / "registered_stage5_protocol.json",
        {
            "schema_version": "thewiz.registered_stage5_protocol_config.v1",
            "protocol_version": "test-2026-08-10.1",
            "thresholds_changed_after_results": False,
            "protocol_contract": {
                "cohort": ("immutable_accepted_registered_stage4_full_family_strict_cost_trades"),
                "features": "entry_time_causal_no_dashboard_hindsight",
                "supervised": {
                    "candidate_families": [
                        "logistic_regression",
                        "regularized_random_forest",
                        "boosted_tree_with_deterministic_fallback",
                    ],
                    "walkforward_splits": 5,
                    "minimum_train_rows": 20,
                    "embargo_periods": 1,
                    "selection_scheme": (
                        "chronological_model_selection_then_untouched_evaluation_v1"
                    ),
                    "global_split_scheme": (
                        "globally_purged_embargoed_pair_aware_timestamp_groups_v2"
                    ),
                    "prediction_membership": "exact_registered_row_fold_phase_v1",
                    "threshold_calibration": ("training_only_minimum_participation_v1"),
                    "threshold_grid": {
                        "start": 0.5,
                        "stop_inclusive": 0.8,
                        "step": 0.05,
                    },
                    "minimum_training_take_rate": 0.1,
                    "minimum_oos_take_rate": 0.1,
                    "model_selection_uses_untouched_evaluation": False,
                    "score_bucket_monotonicity_required": True,
                    "pair_regime_timeframe_concentration_required": True,
                    "selection_concentration_dimensions": [
                        "pair",
                        "timeframe",
                        "regime",
                    ],
                },
                "reinforcement_learning": {
                    "policy": "simulated_quantile_hold_policy",
                    "entry_threshold_quantile": 0.7,
                    "calibration_split": "globally_purged_train",
                    "validation_split": "policy_selection",
                    "test_split": "untouched_evaluation",
                    "train_fraction": 0.6,
                    "validation_fraction": 0.2,
                    "minimum_rows": [50, 30, 30],
                    "minimum_take_rate": 0.1,
                    "positive_validation_after_cost_return_required": True,
                    "positive_test_after_cost_return_required": True,
                    "pair_regime_timeframe_concentration_required": True,
                },
                "selection_rule": (
                    "both_supervised_and_rl_oos_gates_must_pass_without_trade_count_collapse"
                ),
                "failed_run_action": ("research_only_no_quantization_no_testnet_candidate"),
            },
            "source_contracts": [{"role": role, "path": relative} for role, relative in sources],
            "promotion_authority": False,
            "testnet_candidate_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    return build_registered_stage5_protocol(
        root=root,
        now=registered_at or NOW - timedelta(hours=1),
    )


def _registered_stage4_fixture(
    root: Path,
    *,
    accepted: bool = True,
    stage4_accepted: bool | None = None,
    checkpoint_pass: bool | None = None,
    support_policy_overrides: dict[str, object] | None = None,
) -> Path:
    stage4_accepted = accepted if stage4_accepted is None else stage4_accepted
    checkpoint_pass = stage4_accepted if checkpoint_pass is None else checkpoint_pass
    contract_id = "registered-contract-learning"
    execution_id = "registered-execution-learning"
    _write_json(
        root / "config" / "acceptance_policy_manifest.json",
        {
            "schema_version": "thewiz.acceptance_policy.v1",
            "research_gates": {"minimum_independent_supporting_clusters": 3},
        },
    )
    support_policy = {
        "schema_version": ("thewiz.registered_survivor_oos_support_policy.v4"),
        "policy_version": "test-v5",
        "effective_at_utc": NOW.isoformat(),
        "minimum_dataset_rows": 10,
        "minimum_model_selection_folds": 2,
        "minimum_oos_prediction_rows": 10,
        "minimum_oos_folds": 3,
        "minimum_oos_regimes": 2,
        "minimum_oos_taken_rows": 5,
        "minimum_oos_take_rate": 0.1,
        "minimum_oos_taken_return_sum_exclusive": 0.0,
        "minimum_model_filtered_trades": 20,
        "minimum_model_gated_trades": 20,
        "minimum_nonempty_score_buckets": 2,
        "maximum_model_gain_concentration": 0.55,
        "maximum_model_selection_concentration": 0.67,
        "return_basis": "realized_return_after_cost",
        "minimum_rl_rows_per_split": 5,
        "minimum_rl_entered_rows_per_split": 2,
        "minimum_rl_take_rate_per_split": 0.1,
        "minimum_rl_trades": 20,
        "maximum_rl_concentration": 0.65,
        "minimum_rl_validation_return_sum_exclusive": 0.0,
        "minimum_rl_test_return_sum_exclusive": 0.0,
        "rl_return_basis": "net_after_cost_strategy_return",
        "thresholds_changed_after_results": False,
        "promotion_authority": False,
        "testnet_candidate_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    support_policy.update(support_policy_overrides or {})
    _write_json(
        root / "config" / "registered_survivor_oos_support_policy.json",
        support_policy,
    )
    _stage5_protocol_fixture(root)
    contract_path = (
        root / "data" / "research" / "registered_rerun_contracts" / f"{contract_id}.json"
    )
    contract = {
        "contract_id": contract_id,
        "source_family_sha256": "a" * 64,
        "source_family_rows": 14,
        "acceptance_policy_id": "acceptance-1",
        "holdout_policy_id": "holdout-1",
        "registered_candidates": [
            {
                "semantic_hypothesis_id": "hypothesis-1",
                "source_experiment_id": "experiment-1",
                "pair_group_key": "hyperliquid|1h|ETH|BTC",
                "pair": "ETH-USD-BTC-USD",
                "exact_mode": "Static (Spread)",
                "orientation": "original",
            },
            {
                "semantic_hypothesis_id": "hypothesis-2",
                "source_experiment_id": "experiment-2",
                "pair_group_key": "hyperliquid|1h|SOL|TURBO",
                "pair": "SOL-USD-TURBO-USD",
                "exact_mode": "Static (Spread)",
                "orientation": "reverse",
            },
            {
                "semantic_hypothesis_id": "hypothesis-3",
                "source_experiment_id": "experiment-3",
                "pair_group_key": "hyperliquid|1h|WLD|DOGE",
                "pair": "WLD-USD-DOGE-USD",
                "exact_mode": "Static (ZScoreR)",
                "orientation": "original",
            },
        ],
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    _write_json(contract_path, contract)
    active = root / "reports" / "active"
    _write_json(active / "registered_research_rerun_contract.json", contract)
    pd.DataFrame(
        [
            {
                "stage": 4,
                "status": "PASS" if checkpoint_pass else "IN_PROGRESS",
                "evidence_progress": (
                    f"active_contract_id={contract_id};"
                    "final_receipt_conclusion_bound=True;"
                    "stage4_terminal_outcome="
                    + (
                        "ACCEPTED_VALID_INDEPENDENT_SURVIVORS"
                        if checkpoint_pass
                        else "INCOMPLETE_CURRENT_FAMILY"
                    )
                ),
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ]
    ).to_csv(active / "seven_stage_goal_checkpoint.csv", index=False)

    ready_path = root / "data" / "research" / "registered_rerun_ready" / f"{contract_id}.json"
    _write_json(
        ready_path,
        {
            "contract_id": contract_id,
            "parity_sha256": "b" * 64,
            "promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    snapshot = root / "reports" / "snapshots" / "registered-learning"
    snapshot.mkdir(parents=True)
    trades_path = snapshot / "walkforward_trades.csv"
    bars_path = snapshot / "walkforward_bars.csv"
    costs_path = snapshot / "pair_costs.csv"
    regime_path = snapshot / "regime_trades.csv"
    trade_rows = []
    bar_rows = []
    regime_rows = []
    start = NOW - timedelta(days=30)
    counter = 0
    for mode in CANONICAL_WIZARD_MODES:
        for orientation in ("original", "reverse"):
            counter += 1
            entry = start + timedelta(hours=counter * 3)
            exit_at = entry + timedelta(hours=2)
            experiment_id = f"experiment-{counter}"
            accepted_pair = {
                1: ("hyperliquid|1h|ETH|BTC", "ETH-USD-BTC-USD", "ETH", "BTC"),
                2: ("hyperliquid|1h|SOL|TURBO", "SOL-USD-TURBO-USD", "SOL", "TURBO"),
                3: ("hyperliquid|1h|WLD|DOGE", "WLD-USD-DOGE-USD", "WLD", "DOGE"),
            }.get(
                counter,
                ("hyperliquid|1h|ETH|BTC", "ETH-USD-BTC-USD", "ETH", "BTC"),
            )
            trade_rows.append(
                {
                    "experiment_id": experiment_id,
                    "pair_group_key": accepted_pair[0],
                    "pair": accepted_pair[1],
                    "hyperliquid_interval": "1h",
                    "exact_mode": mode,
                    "orientation": orientation,
                    "asset_x": accepted_pair[2],
                    "asset_y": accepted_pair[3],
                    "fold_number": 1,
                    "trade_id": 1,
                    "side": -1 if orientation == "original" else 1,
                    "entry_timestamp": entry.isoformat(),
                    "exit_timestamp": exit_at.isoformat(),
                    "label_timestamp": exit_at.isoformat(),
                    "exit_reason": "signal_exit",
                    "bars": 2,
                    "profit_after_cost": 0.01 if counter % 2 else -0.005,
                    "feature_timestamp": entry.isoformat(),
                    "mode_metric": 2.1 if counter % 2 else -2.1,
                    "mode_metric_name": "u1_given_u2" if mode == "Copula" else "zscore",
                    "spread": 0.2,
                    "spread_slope": 0.01,
                    "realized_volatility_percentile": 0.5,
                    "correlation": 0.8,
                    "hedge_ratio": 1.1,
                    "hedge_ratio_stability": 0.9,
                    "funding_bps_per_day": 0.2,
                    "liquidity_score": 0.8,
                    "feature_known_at_or_before_entry": True,
                    "feature_uses_future_data": False,
                    "uses_dashboard_hindsight": False,
                    "strategy": mode,
                    "strategy_name": mode,
                    "family": mode,
                    "entry_style": "wizard_exact_mode_captured_thresholds",
                    "exit_style": "wizard_exact_mode_captured_thresholds",
                    "live_trading_authorized": False,
                }
            )
            for timestamp, net_return in ((entry, -0.001), (exit_at, 0.002)):
                bar_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "fold_number": 1,
                        "timestamp": timestamp.isoformat(),
                        "net_return": net_return,
                    }
                )
            regime_rows.append(
                {
                    "experiment_id": experiment_id,
                    "fold_number": 1,
                    "trade_id": 1,
                    "entry_timestamp": entry.isoformat(),
                    "regime": "calm" if counter % 2 else "high_vol",
                    "regime_uses_future_data": False,
                }
            )
    pd.DataFrame(trade_rows).to_csv(trades_path, index=False)
    pd.DataFrame(bar_rows).to_csv(bars_path, index=False)
    pd.DataFrame(
        [
            {
                "pair_group_key": pair_group_key,
                "pair": pair,
                "hyperliquid_interval": "1h",
                "cost_model_id": f"strict-cost-{index}",
                "cost_acceptance_ready": True,
            }
            for index, (pair_group_key, pair) in enumerate(
                (
                    ("hyperliquid|1h|ETH|BTC", "ETH-USD-BTC-USD"),
                    ("hyperliquid|1h|SOL|TURBO", "SOL-USD-TURBO-USD"),
                    ("hyperliquid|1h|WLD|DOGE", "WLD-USD-DOGE-USD"),
                ),
                start=1,
            )
        ]
    ).to_csv(costs_path, index=False)
    pd.DataFrame(regime_rows).to_csv(regime_path, index=False)

    def stage(name: str, artifacts: dict[str, Path]) -> dict:
        manifest_path = snapshot / f"{name}_manifest.json"
        manifest = {
            "schema_version": f"test.{name}.v1",
            "experiments_accounted": 14,
            "artifacts": {key: str(path.relative_to(root)) for key, path in artifacts.items()},
            "promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        _write_json(manifest_path, manifest)
        bound = {**artifacts, "snapshot_manifest": manifest_path}
        return {
            "stage": name,
            "stage_identity": f"{name}-1",
            "experiments_accounted": 14,
            "manifest_path": str(manifest_path.relative_to(root)),
            "manifest_sha256": _file_hash(manifest_path),
            "immutable_artifact_hashes": {
                str(path.relative_to(root)): _file_hash(path) for path in bound.values()
            },
            "order_submission_performed": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }

    stages = [
        stage(
            "walkforward",
            {"snapshot_trades": trades_path, "snapshot_bars": bars_path},
        ),
        stage("cost_evidence", {"snapshot_pairs": costs_path}),
        stage("regime_attribution", {"snapshot_trades": regime_path}),
    ]
    survivor_path = snapshot / "final_1x_survivor_receipt.json"
    _write_json(
        survivor_path,
        {
            "schema_version": "thewiz.final_one_x_survivor_receipt.v1",
            "receipt_status": "PASS" if stage4_accepted else "ZERO_SURVIVORS",
            "acceptance_policy_id": "acceptance-1",
            "holdout_policy_id": "holdout-1",
            "independent_supporting_clusters": 3 if stage4_accepted else 0,
            "independent_full_survivor_clusters": 3 if stage4_accepted else 0,
            "independent_supporting_pairs": 3 if stage4_accepted else 0,
            "independent_full_survivor_pairs": 3 if stage4_accepted else 0,
            "final_one_x_survivors": 3 if stage4_accepted else 0,
            "final_experiment_ids": (
                ["experiment-1", "experiment-2", "experiment-3"] if stage4_accepted else []
            ),
            "final_canonical_pairs": (
                ["BTC-ETH", "DOGE-WLD", "SOL-TURBO"] if stage4_accepted else []
            ),
            "blockers": [] if stage4_accepted else ["zero_final_survivors"],
            "thresholds_changed_after_results": False,
            "testnet_candidate_authority": stage4_accepted,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    conclusion_path = (
        root / "data" / "research" / "registered_rerun_conclusions" / f"{contract_id}.json"
    )
    _write_json(
        conclusion_path,
        _conclusion_receipt(
            {
                "contract_id": contract_id,
                "conclusion_status": (
                    "ACCEPTED_REGISTERED_SURVIVORS"
                    if accepted
                    else "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS"
                ),
                "accepted_registered_hypotheses": 3 if accepted else 0,
                "rejected_registered_hypotheses": 0 if accepted else 3,
                "registered_hypotheses": 3,
                "outcomes": {
                    f"hypothesis-{index}": (
                        "ACCEPTED_SURVIVOR" if accepted else "REJECTED_BY_FROZEN_GATES"
                    )
                    for index in range(1, 4)
                },
                "final_survivor_receipt_sha256": _file_hash(survivor_path),
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
    )
    execution_path = (
        root / "data" / "research" / "registered_rerun_executions" / f"{contract_id}.json"
    )
    execution = {
        "schema_version": "thewiz.corrective_registered_rerun_execution.v1",
        "execution_id": execution_id,
        "contract_id": contract_id,
        "started_at_utc": (NOW - timedelta(minutes=30)).isoformat(),
        "completed_at_utc": NOW.isoformat(),
        "status": "PASS_REGISTERED_RERUN_ACCOUNTED",
        "source_family_sha256": "a" * 64,
        "source_family_rows": 14,
        "acceptance_policy_id": "acceptance-1",
        "holdout_policy_id": "holdout-1",
        "ready_receipt_path": str(ready_path.relative_to(root)),
        "ready_receipt_sha256": _file_hash(ready_path),
        "stages": stages,
        "conclusion_path": str(conclusion_path.relative_to(root)),
        "conclusion_sha256": _file_hash(conclusion_path),
        "stage4_survivor_receipt_path": str(survivor_path.relative_to(root)),
        "stage4_survivor_receipt_sha256": _file_hash(survivor_path),
        "stage4_independent_supporting_clusters": 3 if stage4_accepted else 0,
        "stage4_independent_full_survivor_clusters": 3 if stage4_accepted else 0,
        "stage4_independent_supporting_pairs": 3 if stage4_accepted else 0,
        "stage4_independent_full_survivor_pairs": 3 if stage4_accepted else 0,
        "stage4_final_one_x_survivors": 3 if stage4_accepted else 0,
        "stage4_final_experiment_ids": (
            ["experiment-1", "experiment-2", "experiment-3"] if stage4_accepted else []
        ),
        "stage4_final_canonical_pairs": (
            ["BTC-ETH", "DOGE-WLD", "SOL-TURBO"] if stage4_accepted else []
        ),
        "conclusion_status": (
            "ACCEPTED_REGISTERED_SURVIVORS" if accepted else "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS"
        ),
        "accepted_registered_hypotheses": 3 if accepted else 0,
        "rejected_registered_hypotheses": 0 if accepted else 3,
        "order_submission_performed": False,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    execution["receipt_sha256"] = _payload_hash(execution)
    _write_json(execution_path, execution)
    return execution_path


def test_registered_learning_dataset_is_exact_mode_cost_and_lineage_bound(tmp_path):
    execution = _registered_stage4_fixture(tmp_path)

    result = build_registered_exact_mode_trade_dataset(
        root=tmp_path, execution_receipt_path=execution
    )
    dataset = pd.read_csv(result.paths["dataset_csv"])
    audit = pd.read_csv(result.paths["leakage_audit"], keep_default_na=False)
    receipt = json.loads(result.paths["receipt"].read_text())

    assert len(dataset) == 14
    assert dataset["trade_id"].nunique() == 14
    assert set(dataset["exact_mode"]) == set(CANONICAL_WIZARD_MODES)
    assert set(dataset["orientation"]) == {"original", "reverse"}
    assert dataset["registered_candidate"].astype(bool).sum() == 3
    assert dataset["accepted_stage4_survivor"].astype(bool).sum() == 3
    accepted = dataset.loc[dataset["accepted_stage4_survivor"].astype(bool)]
    assert set(accepted["experiment_id"]) == {
        "experiment-1",
        "experiment-2",
        "experiment-3",
    }
    assert set(accepted["registered_semantic_hypothesis_id"]) == {
        "hypothesis-1",
        "hypothesis-2",
        "hypothesis-3",
    }
    assert accepted["registered_hypothesis_outcome"].eq("ACCEPTED_SURVIVOR").all()
    assert dataset["source_venue"].eq("hyperliquid").all()
    assert not audit["uses_future_data"].astype(bool).any()
    assert audit["leakage_blocker"].eq("").all()
    assert receipt["registered_execution_id"] == "registered-execution-learning"
    assert receipt["strict_cost_coverage_complete"] is True
    assert receipt["causal_entry_features_proven"] is True
    assert receipt["research_acceptance_blockers"] == []
    assert receipt["accepted_survivor_dataset_rows"] == 3
    assert receipt["testnet_order_authority"] is False


def test_stage5_protocol_is_immutable_reusable_and_zero_authority(tmp_path):
    _write_json(
        tmp_path / "config" / "registered_survivor_oos_support_policy.json",
        {"schema_version": "test-support-policy"},
    )
    first = _stage5_protocol_fixture(tmp_path)
    second = build_registered_stage5_protocol(root=tmp_path, now=NOW)
    receipt = json.loads(first.paths["protocol_receipt"].read_text(encoding="utf-8"))

    assert first.summary["protocol_id"] == second.summary["protocol_id"]
    assert first.paths["protocol_receipt"] == second.paths["protocol_receipt"]
    assert receipt["registered_at_utc"] == (NOW - timedelta(hours=1)).isoformat()
    assert receipt["testnet_candidate_authority"] is False
    assert receipt["testnet_order_authority"] is False
    assert receipt["live_trading_authorized"] is False


def test_registered_learning_blocks_missing_or_drifted_stage5_protocol(tmp_path):
    execution = _registered_stage4_fixture(tmp_path)
    pointer = tmp_path / "reports" / "active" / "registered_stage5_protocol.json"
    pointer.unlink()

    missing = run_registered_learning_research(
        root=tmp_path,
        now=NOW,
        execute=False,
        execution_receipt_path=execution,
    )
    assert missing.summary["status"] == "BLOCKED_STAGE5_PROTOCOL"
    assert "pointer is missing or blocked" in missing.summary["blocker"]

    _stage5_protocol_fixture(tmp_path)
    source = tmp_path / "src" / "quant_platform" / "ml_filter.py"
    source.write_text(source.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
    drifted = run_registered_learning_research(
        root=tmp_path,
        now=NOW,
        execute=False,
        execution_receipt_path=execution,
    )
    assert drifted.summary["status"] == "BLOCKED_STAGE5_PROTOCOL"
    assert "source_bindings" in drifted.summary["blocker"]
    assert drifted.summary["testnet_order_authority"] is False


def test_stage5_protocol_must_predate_stage4_completion(tmp_path):
    _write_json(
        tmp_path / "config" / "registered_survivor_oos_support_policy.json",
        {"schema_version": "test-support-policy"},
    )
    _stage5_protocol_fixture(tmp_path, registered_at=NOW)

    with pytest.raises(ValueError, match="created after Stage 4 completion"):
        validate_registered_stage5_protocol(
            root=tmp_path,
            stage4_completed_at_utc=(NOW - timedelta(seconds=1)).isoformat(),
        )


def test_registered_learning_rejects_tampered_stage4_snapshot(tmp_path):
    execution = _registered_stage4_fixture(tmp_path)
    payload = json.loads(execution.read_text())
    walk = next(row for row in payload["stages"] if row["stage"] == "walkforward")
    trades_path = next(
        tmp_path / relative
        for relative in walk["immutable_artifact_hashes"]
        if relative.endswith("walkforward_trades.csv")
    )
    trades_path.write_text(trades_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="stage artifact hash mismatch"):
        build_registered_exact_mode_trade_dataset(root=tmp_path, execution_receipt_path=execution)


def test_registered_learning_rejects_relocated_stage4_receipt_before_runners(tmp_path):
    execution = _registered_stage4_fixture(tmp_path)
    relocated = tmp_path / "reports" / "active" / "copied_stage4_execution.json"
    relocated.parent.mkdir(parents=True, exist_ok=True)
    relocated.write_bytes(execution.read_bytes())
    calls: list[dict] = []

    result = run_registered_learning_research(
        root=tmp_path,
        now=NOW,
        execute=True,
        execution_receipt_path=relocated,
        dataset_builder=lambda **kwargs: calls.append(kwargs),
    )

    assert result.summary["status"] == "BLOCKED_STAGE4"
    assert "execution receipt path is not canonical" in result.summary["blocker"]
    assert result.summary["stage5_research_gate_pass"] is False
    assert result.summary["testnet_order_authority"] is False
    assert calls == []


def test_registered_learning_rejects_sparse_mode_orientation_family(tmp_path):
    execution = _registered_stage4_fixture(tmp_path)
    payload = json.loads(execution.read_text())
    walk = next(row for row in payload["stages"] if row["stage"] == "walkforward")
    trades_relative = next(
        relative
        for relative in walk["immutable_artifact_hashes"]
        if relative.endswith("walkforward_trades.csv")
    )
    trades_path = tmp_path / trades_relative
    trades = pd.read_csv(trades_path)
    trades = trades.loc[~(trades["exact_mode"].eq("Copula") & trades["orientation"].eq("reverse"))]
    trades.to_csv(trades_path, index=False)
    walk["immutable_artifact_hashes"][trades_relative] = _file_hash(trades_path)
    manifest_path = tmp_path / walk["manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["snapshot_trades"] = trades_relative
    _write_json(manifest_path, manifest)
    walk["manifest_sha256"] = _file_hash(manifest_path)
    payload["receipt_sha256"] = _payload_hash(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    _write_json(execution, payload)

    with pytest.raises(ValueError, match="mode-orientation family is incomplete"):
        build_registered_exact_mode_trade_dataset(root=tmp_path, execution_receipt_path=execution)


def test_registered_learning_rejects_missing_accepted_survivor_trades(tmp_path):
    execution = _registered_stage4_fixture(tmp_path)
    payload = json.loads(execution.read_text())
    walk = next(row for row in payload["stages"] if row["stage"] == "walkforward")
    trades_relative = next(
        relative
        for relative in walk["immutable_artifact_hashes"]
        if relative.endswith("walkforward_trades.csv")
    )
    trades_path = tmp_path / trades_relative
    trades = pd.read_csv(trades_path)
    trades.loc[trades["experiment_id"].eq("experiment-3"), "experiment_id"] = "full-family-forged"
    trades.to_csv(trades_path, index=False)
    walk["immutable_artifact_hashes"][trades_relative] = _file_hash(trades_path)
    manifest_path = tmp_path / walk["manifest_path"]
    walk["manifest_sha256"] = _file_hash(manifest_path)
    payload["receipt_sha256"] = _payload_hash(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    _write_json(execution, payload)

    with pytest.raises(
        ValueError,
        match="accepted Stage 4 survivor trade support is incomplete",
    ):
        build_registered_exact_mode_trade_dataset(root=tmp_path, execution_receipt_path=execution)


@pytest.mark.parametrize(
    ("field", "first_experiment", "second_experiment"),
    [
        ("pair", "experiment-1", "experiment-2"),
        ("pair_group_key", "experiment-1", "experiment-2"),
        ("exact_mode", "experiment-1", "experiment-3"),
        ("orientation", "experiment-1", "experiment-2"),
    ],
)
def test_registered_learning_rejects_stage4_trade_identity_substitution(
    tmp_path,
    field,
    first_experiment,
    second_experiment,
):
    execution = _registered_stage4_fixture(tmp_path)
    payload = json.loads(execution.read_text())
    walk = next(row for row in payload["stages"] if row["stage"] == "walkforward")
    trades_relative = next(
        relative
        for relative in walk["immutable_artifact_hashes"]
        if relative.endswith("walkforward_trades.csv")
    )
    trades_path = tmp_path / trades_relative
    trades = pd.read_csv(trades_path)
    first_mask = trades["experiment_id"].eq(first_experiment)
    second_mask = trades["experiment_id"].eq(second_experiment)
    first_value = trades.loc[first_mask, field].iloc[0]
    second_value = trades.loc[second_mask, field].iloc[0]
    trades.loc[first_mask, field] = second_value
    trades.loc[second_mask, field] = first_value
    trades.to_csv(trades_path, index=False)
    walk["immutable_artifact_hashes"][trades_relative] = _file_hash(trades_path)
    payload["receipt_sha256"] = _payload_hash(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    _write_json(execution, payload)

    with pytest.raises(
        ValueError,
        match="trade identity does not match Stage 4 contract",
    ):
        build_registered_exact_mode_trade_dataset(
            root=tmp_path,
            execution_receipt_path=execution,
        )


def test_registered_learning_blocks_conclusive_zero_survivor_stage4(tmp_path):
    execution = _registered_stage4_fixture(tmp_path, accepted=False)
    calls = []

    result = run_registered_learning_research(
        root=tmp_path,
        now=NOW,
        execute=True,
        execution_receipt_path=execution,
        dataset_builder=lambda **kwargs: calls.append(kwargs),
    )

    assert result.summary["status"] == ("BLOCKED_STAGE4_NO_ACCEPTED_SURVIVOR")
    assert result.summary["accepted_registered_hypotheses"] == 0
    assert calls == []
    assert not result.summary["stage5_research_gate_pass"]
    assert not result.summary["testnet_order_authority"]
    with pytest.raises(ValueError, match="requires an accepted Stage 4 survivor"):
        build_registered_exact_mode_trade_dataset(root=tmp_path, execution_receipt_path=execution)


def test_registered_learning_blocks_partial_accepted_contract_without_final_breadth(
    tmp_path,
):
    execution = _registered_stage4_fixture(
        tmp_path,
        accepted=True,
        stage4_accepted=False,
    )

    result = run_registered_learning_research(
        root=tmp_path,
        now=NOW,
        execute=True,
        execution_receipt_path=execution,
    )

    assert result.summary["status"] == "BLOCKED_STAGE4"
    assert "not the exact accepted registered cohort" in result.summary["blocker"]
    assert result.summary["registered_execution_id"] == ""
    assert result.summary["stage5_research_gate_pass"] is False
    assert result.summary["testnet_order_authority"] is False


def test_registered_learning_direct_command_requires_whole_cohort_checkpoint(
    tmp_path,
):
    execution = _registered_stage4_fixture(
        tmp_path,
        accepted=True,
        stage4_accepted=True,
        checkpoint_pass=False,
    )

    result = run_registered_learning_research(
        root=tmp_path,
        now=NOW,
        execute=True,
        execution_receipt_path=execution,
    )

    assert result.summary["status"] == ("BLOCKED_STAGE4_CURRENT_FAMILY_INCOMPLETE")
    assert result.summary["blocker"] == (
        "registered_stage4_checkpoint_does_not_prove_whole_cohort_accepted_independent_survivors"
    )
    assert result.summary["stage5_research_gate_pass"] is False
    assert result.summary["testnet_order_authority"] is False


def test_registered_learning_plan_invokes_no_learning_stages(tmp_path):
    execution = _registered_stage4_fixture(tmp_path)

    result = run_registered_learning_research(
        root=tmp_path,
        now=NOW,
        execute=False,
        execution_receipt_path=execution,
    )

    assert result.summary["status"] == "PLANNED"
    assert result.summary["testnet_order_authority"] is False
    assert not (tmp_path / "data" / "ml" / "candidate_trade_dataset.json").exists()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("minimum_model_selection_folds", 0),
        ("minimum_oos_folds", 0),
        ("minimum_model_gated_trades", 0),
        ("maximum_model_gain_concentration", 0.0),
        ("maximum_model_selection_concentration", 0.0),
        ("maximum_model_selection_concentration", 1.01),
        ("maximum_rl_concentration", 1.01),
    ],
)
def test_registered_learning_blocks_invalid_survivor_support_policy(tmp_path, field, invalid_value):
    execution = _registered_stage4_fixture(tmp_path)
    policy_path = tmp_path / "config" / "registered_survivor_oos_support_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy[field] = invalid_value
    _write_json(policy_path, policy)

    result = run_registered_learning_research(
        root=tmp_path,
        now=NOW,
        execute=False,
        execution_receipt_path=execution,
    )

    assert result.summary["status"] == "BLOCKED_STAGE5_SUPPORT_POLICY"
    assert field in result.summary["blocker"]
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_registered_learning_blocks_invalid_survivor_support_policy_timestamp(tmp_path):
    execution = _registered_stage4_fixture(tmp_path)
    policy_path = tmp_path / "config" / "registered_survivor_oos_support_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["effective_at_utc"] = "not-a-timestamp"
    _write_json(policy_path, policy)

    result = run_registered_learning_research(
        root=tmp_path,
        now=NOW,
        execute=False,
        execution_receipt_path=execution,
    )

    assert result.summary["status"] == "BLOCKED_STAGE5_SUPPORT_POLICY"
    assert "effective_at_utc" in result.summary["blocker"]
    assert result.summary["stage5_research_gate_pass"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


@pytest.mark.parametrize(
    "dimensions",
    [
        [],
        ["pair", "regime"],
        ["pair", "regime", "timeframe"],
        ["pair", "timeframe", "regime", "strategy"],
    ],
)
def test_registered_supervised_protocol_freezes_selection_dimensions(
    tmp_path,
    dimensions,
):
    _registered_stage4_fixture(tmp_path)
    _, receipt = validate_registered_stage5_protocol(root=tmp_path)
    receipt["protocol_contract"]["supervised"]["selection_concentration_dimensions"] = dimensions

    with pytest.raises(ValueError, match="selection_concentration_dimensions"):
        _registered_supervised_protocol(receipt)


@pytest.mark.parametrize(
    ("section", "field", "invalid_value", "parser"),
    [
        (
            "contract",
            "selection_rule",
            "supervised_only",
            _registered_supervised_protocol,
        ),
        ("supervised", "minimum_oos_take_rate", 0.0, _registered_supervised_protocol),
        (
            "supervised",
            "threshold_grid",
            {"start": 0.5, "stop_inclusive": 0.9, "step": 0.05},
            _registered_supervised_protocol,
        ),
        (
            "reinforcement_learning",
            "minimum_take_rate",
            0.0,
            _registered_rl_protocol,
        ),
        (
            "reinforcement_learning",
            "positive_validation_after_cost_return_required",
            False,
            _registered_rl_protocol,
        ),
        (
            "reinforcement_learning",
            "positive_test_after_cost_return_required",
            False,
            _registered_rl_protocol,
        ),
        (
            "reinforcement_learning",
            "pair_regime_timeframe_concentration_required",
            False,
            _registered_rl_protocol,
        ),
    ],
)
def test_registered_stage5_protocol_rejects_weakened_oos_controls(
    tmp_path,
    section,
    field,
    invalid_value,
    parser,
):
    _registered_stage4_fixture(tmp_path)
    _, receipt = validate_registered_stage5_protocol(root=tmp_path)
    contract = receipt["protocol_contract"]
    target = contract if section == "contract" else contract[section]
    target[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        parser(receipt)


def test_stage5_model_and_rl_take_rate_floor_is_ten_percent():
    comparison = pd.DataFrame(
        [
            {
                "variant": "raw_strategy",
                "profit_factor": 1.2,
                "sharpe": 0.5,
                "max_drawdown": 0.25,
                "total_return": 0.1,
                "trades": 200,
                "take_rate": 1.0,
            },
            {
                "variant": "model_gated_strategy",
                "profit_factor": 1.4,
                "sharpe": 0.6,
                "max_drawdown": 0.20,
                "total_return": 0.12,
                "trades": 20,
                "take_rate": 0.09,
            },
        ]
    )
    assert not bool(_model_gated_acceptance(comparison).iloc[0]["accepted"])
    comparison.loc[comparison["variant"].eq("model_gated_strategy"), "take_rate"] = 0.10
    assert bool(_model_gated_acceptance(comparison).iloc[0]["accepted"])

    rows = []
    for split in ("validation", "held_out_test"):
        rows.extend(
            [
                {
                    "evaluation_split": split,
                    "variant": "non_rl_baseline",
                    "profit_factor": 1.2,
                    "max_drawdown": 0.25,
                    "sharpe": 0.5,
                    "total_return": 0.05,
                    "trades": 200,
                    "take_rate": 1.0,
                    "pair_concentration": 0.5,
                    "pair_pnl_concentration": 0.5,
                    "timeframe_concentration": 0.5,
                    "timeframe_pnl_concentration": 0.5,
                    "regime_concentration": 0.5,
                    "regime_pnl_concentration": 0.5,
                },
                {
                    "evaluation_split": split,
                    "variant": "safe_rl_policy",
                    "profit_factor": 1.4,
                    "max_drawdown": 0.20,
                    "sharpe": 0.6,
                    "total_return": 0.06,
                    "trades": 20,
                    "take_rate": 0.09,
                    "pair_concentration": 0.5,
                    "pair_pnl_concentration": 0.5,
                    "timeframe_concentration": 0.5,
                    "timeframe_pnl_concentration": 0.5,
                    "regime_concentration": 0.5,
                    "regime_pnl_concentration": 0.5,
                },
            ]
        )
    evaluation = pd.DataFrame(rows)
    assert not bool(rl_acceptance_report(evaluation).iloc[0]["accepted"])
    evaluation.loc[evaluation["variant"].eq("safe_rl_policy"), "take_rate"] = 0.10
    assert bool(rl_acceptance_report(evaluation).iloc[0]["accepted"])


def test_stage5_rl_rejects_negative_after_cost_split_despite_relative_improvement():
    rows = []
    for split in ("validation", "held_out_test"):
        rows.extend(
            [
                {
                    "evaluation_split": split,
                    "variant": "non_rl_baseline",
                    "profit_factor": 0.7,
                    "max_drawdown": 0.25,
                    "sharpe": -0.5,
                    "total_return": -0.10,
                    "trades": 200,
                    "take_rate": 1.0,
                    "pair_concentration": 0.5,
                    "pair_pnl_concentration": 0.5,
                    "timeframe_concentration": 0.5,
                    "timeframe_pnl_concentration": 0.5,
                    "regime_concentration": 0.5,
                    "regime_pnl_concentration": 0.5,
                },
                {
                    "evaluation_split": split,
                    "variant": "safe_rl_policy",
                    "profit_factor": 0.9,
                    "max_drawdown": 0.20,
                    "sharpe": -0.3,
                    "total_return": -0.01,
                    "trades": 20,
                    "take_rate": 0.10,
                    "pair_concentration": 0.5,
                    "pair_pnl_concentration": 0.5,
                    "timeframe_concentration": 0.5,
                    "timeframe_pnl_concentration": 0.5,
                    "regime_concentration": 0.5,
                    "regime_pnl_concentration": 0.5,
                },
            ]
        )

    report = rl_acceptance_report(pd.DataFrame(rows)).iloc[0]

    assert not bool(report["accepted"])
    assert report["validation_gate_failures"] == "positive_after_cost_return"
    assert report["held_out_test_gate_failures"] == "positive_after_cost_return"


def test_stage5_rl_survivor_requires_positive_validation_and_test_returns():
    identity = {
        "registered_semantic_hypothesis_id": "hypothesis-1",
        "accepted_stage4_survivor": True,
        "experiment_id": "experiment-1",
        "pair": "ETH-PYTH",
        "exact_mode": "OU Spread",
        "orientation": "x_on_y",
    }
    dataset = pd.DataFrame([{**identity, "trade_id": f"trade-{index}"} for index in range(10)])
    execution_rows = []
    for split, entered_return in (("validation", -0.01), ("held_out_test", 0.01)):
        for index in range(5):
            entered = index < 2
            execution_rows.append(
                {
                    **identity,
                    "evaluation_split": split,
                    "simulation_reason": "entered" if entered else "skipped",
                    "simulated_return": entered_return if entered else 0.0,
                    "return_basis": "net_after_cost_strategy_return",
                }
            )
    policy = {
        "support_policy_id": "survivor-policy-1",
        "policy_version": "test-v2",
        "minimum_rl_rows_per_split": 5,
        "minimum_rl_entered_rows_per_split": 2,
        "minimum_rl_take_rate_per_split": 0.1,
        "minimum_rl_validation_return_sum_exclusive": 0.0,
        "minimum_rl_test_return_sum_exclusive": 0.0,
        "rl_return_basis": "net_after_cost_strategy_return",
    }

    attribution = _registered_survivor_rl_attribution_frame(
        dataset=dataset,
        execution=pd.DataFrame(execution_rows),
        survivor_support_policy=policy,
    )

    assert attribution.iloc[0]["support_status"] == ("BLOCKED_RL_VALIDATION_AFTER_COST_RETURN")
    assert not bool(attribution.iloc[0]["testnet_candidate_authority"])

    validation_entered = pd.DataFrame(execution_rows)["evaluation_split"].eq(
        "validation"
    ) & pd.DataFrame(execution_rows)["simulation_reason"].eq("entered")
    repaired = pd.DataFrame(execution_rows)
    repaired.loc[validation_entered, "simulated_return"] = 0.01
    attribution = _registered_survivor_rl_attribution_frame(
        dataset=dataset,
        execution=repaired,
        survivor_support_policy=policy,
    )

    assert attribution.iloc[0]["support_status"] == "READY"


def test_stage5_selection_concentration_requires_complete_normalized_dimensions():
    rows = [
        {"dimension": dimension, "value": value, "share_of_taken": share}
        for dimension in ("pair", "timeframe", "regime")
        for value, share in (("a", 0.6), ("b", 0.4))
    ]
    evidence = pd.DataFrame(rows)
    dimensions = ["pair", "timeframe", "regime"]

    assert _selection_concentration_within_limit(
        evidence,
        dimensions=dimensions,
        maximum=0.65,
    )
    assert not _selection_concentration_within_limit(
        evidence.loc[evidence["dimension"].ne("regime")],
        dimensions=dimensions,
        maximum=0.65,
    )

    malformed = evidence.copy()
    malformed.loc[
        malformed["dimension"].eq("timeframe"),
        "share_of_taken",
    ] = [0.7, 0.4]
    assert not _selection_concentration_within_limit(
        malformed,
        dimensions=dimensions,
        maximum=0.65,
    )


def _registered_learning_runners(
    root: Path,
    calls: list[str],
    *,
    model_accepted: bool = True,
    unsafe_stage: str = "",
    omit_artifact: str = "",
    attack: str = "",
) -> dict[str, object]:
    dataset_id = "dataset-registered"
    dataset_start = NOW - timedelta(days=250)

    def accepted_identity(index: int) -> tuple[str, str, str]:
        survivor_number = (index // 2) % 3 + 1
        pair = {
            1: "ETH-PYTH",
            2: "SOL-TURBO",
            3: "DOGE-WLD",
        }[survivor_number]
        return (
            f"experiment-{survivor_number}",
            f"hypothesis-{survivor_number}",
            pair,
        )

    def artifact(name: str, payload: str = "evidence\n") -> Path:
        path = root / "fake-learning" / name
        if name != omit_artifact:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")
        return path

    def json_artifact(name: str, payload: dict) -> Path:
        return artifact(name, json.dumps(payload, indent=2, sort_keys=True))

    def csv_artifact(name: str, frame: pd.DataFrame) -> Path:
        path = root / "fake-learning" / name
        if name != omit_artifact:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
        return path

    def summary(name: str, **extra) -> dict:
        payload = {
            "promotion_authority": False,
            "testnet_candidate_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
            **extra,
        }
        if name == unsafe_stage:
            payload["testnet_candidate_authority"] = True
        return payload

    def dataset_builder(**_):
        calls.append("dataset_builder")
        receipt = artifact("dataset_receipt.json", '{"research_acceptance_blockers": []}')
        pointer = artifact("candidate_pointer.json", "{}")
        return CommandResult(
            paths={"receipt": receipt, "candidate_pointer": pointer},
            summary=summary("dataset_builder"),
        )

    def promoter(**_):
        calls.append("promoter")
        dataset_rows = []
        for index in range(200):
            accepted_survivor = index % 2 == 0
            experiment_id, semantic_id, accepted_pair = accepted_identity(index)
            realized_return = 0.01 if accepted_survivor else (-0.01 if index % 4 == 1 else 0.02)
            entry = dataset_start + timedelta(days=index)
            exit_at = entry + timedelta(hours=1)
            if attack == "rl_forged_split_audit" and index == 109:
                exit_at = dataset_start + timedelta(days=130)
            dataset_rows.append(
                {
                    "trade_id": f"trade-{index}",
                    "experiment_id": (
                        experiment_id if accepted_survivor else f"full-family-{index}"
                    ),
                    "pair": (
                        accepted_pair
                        if accepted_survivor
                        else ("BTC-DOGE", "SOL-WLD", "ETH-LINK")[(index // 3) % 3]
                    ),
                    "source_venue": "hyperliquid",
                    "strategy_id": (
                        "copula-dislocation"
                        if accepted_survivor
                        else ("static-zscore", "ou-zscore")[(index // 3) % 2]
                    ),
                    "strategy_name": (
                        "copula" if accepted_survivor else ("zscore", "ou")[(index // 3) % 2]
                    ),
                    "family": (
                        "copula" if accepted_survivor else ("static", "ou")[(index // 3) % 2]
                    ),
                    "backtest_mode": "strict_cost",
                    "exact_mode": "Copula" if accepted_survivor else "Static (Spread)",
                    "orientation": "reverse" if accepted_survivor else "original",
                    "registered_contract_id": "registered-contract-learning",
                    "registered_execution_id": "registered-execution-learning",
                    "registered_semantic_hypothesis_id": (semantic_id if accepted_survivor else ""),
                    "registered_hypothesis_outcome": (
                        "ACCEPTED_SURVIVOR" if accepted_survivor else "UNREGISTERED_FULL_FAMILY"
                    ),
                    "registered_candidate": accepted_survivor,
                    "accepted_stage4_survivor": accepted_survivor,
                    "timeframe": ("1h", "4h")[index % 2],
                    "regime": (
                        "range"
                        if attack == "model_single_regime_registered_survivor" and accepted_survivor
                        else ("range" if index % 4 == 0 or index % 8 == 3 else "trend")
                    ),
                    "feature_timestamp": entry.isoformat(),
                    "entry_timestamp": entry.isoformat(),
                    "label_timestamp": exit_at.isoformat(),
                    "exit_timestamp": exit_at.isoformat(),
                    "profit_after_cost": realized_return,
                    "realized_return": realized_return,
                    "label_profitable": int(realized_return > 0.0),
                }
            )
        dataset = csv_artifact(
            "active_dataset.csv",
            pd.DataFrame(dataset_rows),
        )
        pointer = json_artifact(
            "active_pointer.json",
            {
                "status": "ACTIVE_RESEARCH_DATASET",
                "dataset_id": dataset_id,
                "active_dataset_path": str(dataset.relative_to(root)),
                "active_dataset_sha256": _file_hash(dataset),
            },
        )
        return CommandResult(
            paths={"active_pointer": pointer},
            summary=summary("promoter", dataset_id=dataset_id),
        )

    def trainer(**_):
        calls.append("trainer")
        metrics = {
            "accepted": model_accepted,
            "best_model": (
                "forged_model" if attack == "model_forged_best_model" else "gradient_boosting"
            ),
            "training_dataset_id": dataset_id,
            "evaluation_scheme": ("globally_purged_embargoed_pair_aware_timestamp_groups_v2"),
            "selection_isolation_scheme": (
                "chronological_model_selection_then_untouched_evaluation_v1"
            ),
            "selection_evaluation_boundary_scheme": ("label_complete_chronology_gap_v1"),
            "chronology_gap_folds": "",
            "chronology_gap_fold_count": 0,
            "selection_label_end_boundary": (
                dataset_start + timedelta(days=100, hours=1)
            ).isoformat(),
            "untouched_evaluation_start_boundary": (
                dataset_start + timedelta(days=101)
            ).isoformat(),
            "model_selection_isolation_proven": True,
            "model_winner_replayed": True,
            "selection_folds": 2,
            "untouched_evaluation_folds": 3,
            "median_take_rate": 1 / 3,
            "total_filtered_trades": 30,
            "score_buckets_monotonic": True,
            "top_pair_positive_gain_share": 0.25,
            "top_timeframe_positive_gain_share": 0.5,
            "top_regime_positive_gain_share": 0.5,
            "top_strategy_positive_gain_share": 0.5,
            "failing_checks": "" if model_accepted else "forced_rejection",
        }
        lineage = {
            "training_dataset_id": dataset_id,
            "accepted": model_accepted,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        return CommandResult(
            paths={
                "metrics": json_artifact("model_metrics.json", metrics),
                "lineage": json_artifact("model_lineage.json", lineage),
            },
            summary=summary("trainer", accepted=model_accepted),
        )

    def model_backtester(**_):
        calls.append("model_backtester")
        pairs = [
            "ETH-PYTH",
            "SOL-TURBO",
            "DOGE-WLD",
            "BTC-DOGE",
            "SOL-WLD",
            "ETH-LINK",
        ]
        active_dataset = pd.read_csv(root / "fake-learning" / "active_dataset.csv")
        membership = expected_walkforward_prediction_membership(
            active_dataset,
            n_splits=5,
            min_train_rows=20,
            embargo_periods=1,
        )
        prediction_rows = []
        for member in membership.to_dict(orient="records"):
            index = int(str(member["trade_id"]).split("-")[-1])
            accepted_survivor = index % 2 == 0
            experiment_id, semantic_id, accepted_pair = accepted_identity(index)
            realized_return = 0.01 if accepted_survivor else (-0.01 if index % 4 == 1 else 0.02)
            prediction_rows.append(
                {
                    "trade_id": f"trade-{index}",
                    "model_name": "gradient_boosting",
                    "fold": int(member["fold"]),
                    "selection_phase": str(member["selection_phase"]),
                    "walkforward_splits_requested": int(member["walkforward_splits_requested"]),
                    "minimum_train_rows_requested": int(member["minimum_train_rows_requested"]),
                    "selection_isolation_scheme": (
                        "chronological_model_selection_then_untouched_evaluation_v1"
                    ),
                    "selection_evaluation_boundary_scheme": ("label_complete_chronology_gap_v1"),
                    "chronology_gap_folds": "",
                    "selection_label_end_boundary": (
                        dataset_start + timedelta(days=100, hours=1)
                    ).isoformat(),
                    "untouched_evaluation_start_boundary": (
                        dataset_start + timedelta(days=101)
                    ).isoformat(),
                    "split_scheme": ("globally_purged_embargoed_pair_aware_timestamp_groups_v2"),
                    "embargo_periods": int(member["embargo_periods"]),
                    "global_label_purge": True,
                    "global_label_overlap_rows_after_purge": 0,
                    "test_start": "2026-02-01T00:00:00+00:00",
                    "train_label_end_max": "2026-01-31T23:00:00+00:00",
                    "probability_profitable": (0.8 if accepted_survivor or index % 4 == 3 else 0.4),
                    "threshold": 0.7,
                    "shadow_take": accepted_survivor or index % 4 == 3,
                    "profit_after_cost": realized_return,
                    "realized_return": realized_return,
                    "label_profitable": int(realized_return > 0.0),
                    "pair": (
                        accepted_pair
                        if accepted_survivor
                        else ("BTC-DOGE", "SOL-WLD", "ETH-LINK")[(index // 3) % 3]
                    ),
                    "source_venue": "hyperliquid",
                    "strategy_id": (
                        "copula-dislocation"
                        if accepted_survivor
                        else ("static-zscore", "ou-zscore")[(index // 3) % 2]
                    ),
                    "strategy_name": (
                        "copula" if accepted_survivor else ("zscore", "ou")[(index // 3) % 2]
                    ),
                    "family": (
                        "copula" if accepted_survivor else ("static", "ou")[(index // 3) % 2]
                    ),
                    "backtest_mode": "strict_cost",
                    "timeframe": ("1h", "4h")[index % 2],
                    "regime": (
                        "range"
                        if attack == "model_single_regime_registered_survivor" and accepted_survivor
                        else ("range" if index % 4 == 0 or index % 8 == 3 else "trend")
                    ),
                    "exact_mode": "Copula" if accepted_survivor else "Static (Spread)",
                    "orientation": "reverse" if accepted_survivor else "original",
                    "registered_contract_id": "registered-contract-learning",
                    "registered_execution_id": "registered-execution-learning",
                    "experiment_id": (
                        experiment_id if accepted_survivor else f"full-family-{index}"
                    ),
                    "registered_semantic_hypothesis_id": (semantic_id if accepted_survivor else ""),
                    "registered_hypothesis_outcome": (
                        "ACCEPTED_SURVIVOR" if accepted_survivor else "UNREGISTERED_FULL_FAMILY"
                    ),
                    "registered_candidate": accepted_survivor,
                    "accepted_stage4_survivor": accepted_survivor,
                    "feature_timestamp": (dataset_start + timedelta(days=index)).isoformat(),
                    "entry_timestamp": (dataset_start + timedelta(days=index)).isoformat(),
                    "label_timestamp": (
                        dataset_start + timedelta(days=index) + timedelta(hours=1)
                    ).isoformat(),
                    "exit_timestamp": (
                        dataset_start + timedelta(days=index) + timedelta(hours=1)
                    ).isoformat(),
                }
            )
        predictions = pd.DataFrame(prediction_rows)
        if attack == "model_skips_registered_survivor":
            mask = predictions["accepted_stage4_survivor"].astype(bool)
            predictions.loc[mask, "shadow_take"] = False
        if attack == "model_single_fold_registered_survivor":
            mask = predictions["accepted_stage4_survivor"].astype(bool)
            predictions.loc[mask, "fold"] = 0
        if attack == "model_timeframe_selection_concentration":
            predictions.loc[predictions["timeframe"].ne("1h"), "shadow_take"] = False
        if attack == "model_regime_selection_concentration":
            predictions.loc[predictions["regime"].ne("range"), "shadow_take"] = False
        if attack == "model_global_label_overlap":
            predictions.loc[:, "train_label_end_max"] = "2026-02-01T00:01:00+00:00"
        if attack == "model_prediction_trade_id_forged":
            predictions.loc[predictions.index[0], "trade_id"] = "forged-trade"
        if attack == "model_prediction_outcome_forged":
            predictions.loc[predictions.index[0], "realized_return"] = 0.99
        if attack == "model_selection_phase_overlap":
            predictions.loc[predictions.index[-1], "selection_phase"] = "model_selection"
        if attack == "model_prediction_row_omitted":
            removable = predictions.loc[
                predictions["selection_phase"].eq("untouched_evaluation")
                & ~predictions["accepted_stage4_survivor"].astype(bool)
                & predictions["realized_return"].lt(0.0)
            ]
            predictions = predictions.drop(index=removable.index[:1]).reset_index(drop=True)
        evaluation_predictions = predictions.loc[
            predictions["selection_phase"].eq("untouched_evaluation")
        ].copy()
        acceptance = _model_gated_acceptance(_model_gated_comparison(evaluation_predictions))
        accepted_value = model_accepted and attack != "model_artifact_rejects"
        acceptance.loc[:, "accepted"] = accepted_value
        acceptance.loc[:, "blocker"] = "" if accepted_value else "forced_rejection"
        if attack == "model_take_rate_collapse":
            acceptance.loc[:, "gated_take_rate"] = 0.01
        if attack == "model_acceptance_mismatch":
            acceptance.loc[:, "gated_drawdown"] = (
                pd.to_numeric(acceptance["gated_drawdown"], errors="coerce") + 0.05
            )
        model_backtest = _model_gated_comparison(evaluation_predictions)
        score_buckets = _score_bucket_report(evaluation_predictions)
        pair_concentration = _model_pair_concentration(evaluation_predictions)
        gain_concentration = _model_gain_concentration(evaluation_predictions)
        if attack == "model_backtest_mismatch":
            model_backtest.loc[model_backtest.index[0], "expectancy"] += 0.001
        if attack == "model_score_bucket_mismatch":
            score_buckets.loc[score_buckets.index[0], "rows"] += 1
        if attack == "model_pair_artifact_mismatch":
            pair_concentration.loc[pair_concentration.index[0], "taken_rows"] += 1
        if attack == "model_gain_artifact_mismatch":
            gain_concentration.loc[gain_concentration.index[0], "value"] = "forged-value"
        if attack == "model_gain_concentration":
            pair_gain = gain_concentration["dimension"].astype(str).eq("pair")
            pair_indexes = gain_concentration.index[pair_gain].tolist()
            gain_concentration.loc[pair_indexes[0], "share_of_positive_returns"] = 0.75
            gain_concentration.loc[pair_indexes[1:], "share_of_positive_returns"] = 0.25 / max(
                len(pair_indexes) - 1, 1
            )
        selection_leaderboard = model_selection_leaderboard(predictions)
        if attack == "model_selection_leaderboard_forged":
            selection_leaderboard.loc[selection_leaderboard.index[0], "selection_score"] += 99.0
        return CommandResult(
            paths={
                "predictions": csv_artifact("model_predictions.csv", predictions),
                "selection_leaderboard": csv_artifact(
                    "model_selection_leaderboard.csv",
                    selection_leaderboard,
                ),
                "backtest": csv_artifact(
                    "model_backtest.csv",
                    model_backtest,
                ),
                "acceptance": csv_artifact("model_acceptance.csv", acceptance),
                "score_buckets": csv_artifact("model_score_buckets.csv", score_buckets),
                "pair_concentration": csv_artifact(
                    "model_pair_concentration.csv", pair_concentration
                ),
                "gain_concentration": csv_artifact(
                    "model_gain_concentration.csv", gain_concentration
                ),
                "failures": csv_artifact(
                    "model_failure_attribution.csv",
                    pd.DataFrame(
                        [
                            {
                                "reason": "none",
                                "promotion_authority": False,
                                "testnet_order_authority": False,
                                "live_trading_authorized": False,
                            }
                        ]
                    ),
                ),
                "model_gate_pair_support_report": csv_artifact(
                    "model_pair_support.csv",
                    pd.DataFrame(
                        [
                            {
                                "pair": pair,
                                "promotion_authority": False,
                                "testnet_order_authority": False,
                                "live_trading_authorized": False,
                            }
                            for pair in pairs
                        ]
                    ),
                ),
            },
            summary=summary("model_backtester", accepted=model_accepted),
        )

    def rl_runner(**_):
        calls.append("rl_runner")
        active_pointer = json.loads(
            (root / "fake-learning" / "active_pointer.json").read_text(encoding="utf-8")
        )
        active_dataset = pd.read_csv(root / active_pointer["active_dataset_path"])
        _, partitions, split_audit = _chronological_rl_partitions(active_dataset)
        if attack == "rl_forged_split_audit":
            train_row = split_audit["split"].eq("train")
            split_audit.loc[train_row, "purged_overlap_rows"] = 0
        rl_execution = pd.DataFrame(
            [
                {
                    "trade_id": str(row["trade_id"]),
                    "evaluation_split": split,
                    "pair": str(row["pair"]),
                    "timeframe": str(row["timeframe"]),
                    "regime": str(row["regime"]),
                    "exact_mode": str(row["exact_mode"]),
                    "orientation": str(row["orientation"]),
                    "experiment_id": str(row["experiment_id"]),
                    "registered_semantic_hypothesis_id": str(
                        row["registered_semantic_hypothesis_id"]
                    ),
                    "accepted_stage4_survivor": bool(row["accepted_stage4_survivor"]),
                    "feature_timestamp": str(row["feature_timestamp"]),
                    "entry_timestamp": str(row["entry_timestamp"]),
                    "exit_timestamp": str(row["exit_timestamp"]),
                    "simulation_reason": (
                        "entered"
                        if bool(row["accepted_stage4_survivor"])
                        or int(str(row["trade_id"]).split("-")[-1]) % 3 != 0
                        else "below_entry_threshold"
                    ),
                    "policy_name": "simulated_quantile_hold_policy",
                    "entry_threshold_calibration_quantile": 0.7,
                    "base_return": float(row["profit_after_cost"]),
                    "simulated_return": (
                        -0.005 if int(str(row["trade_id"]).split("-")[-1]) % 7 == 0 else 0.02
                    ),
                    "return_basis": "net_after_cost_strategy_return",
                    "promotion_authority": False,
                    "testnet_candidate_authority": False,
                    "testnet_order_authority": False,
                    "live_trading_authorized": False,
                }
                for split, partition_name in (
                    ("validation", "validation"),
                    ("held_out_test", "test"),
                )
                for _, row in partitions[partition_name].iterrows()
            ]
        )
        if attack == "rl_missing_test":
            rl_execution = rl_execution.loc[~rl_execution["evaluation_split"].eq("held_out_test")]
        if attack == "rl_concentration":
            entered = rl_execution["simulation_reason"].eq("entered")
            rl_execution.loc[entered, ["pair", "timeframe", "regime"]] = [
                "ETH-PYTH",
                "1h",
                "range",
            ]
        if attack == "rl_missing_registered_survivor_test":
            survivor_test = rl_execution["evaluation_split"].eq("held_out_test") & rl_execution[
                "accepted_stage4_survivor"
            ].astype(bool)
            rl_execution.loc[survivor_test, "registered_semantic_hypothesis_id"] = ""
            rl_execution.loc[survivor_test, "accepted_stage4_survivor"] = False
        if attack == "rl_execution_membership_swap":
            first_test = rl_execution.index[rl_execution["evaluation_split"].eq("held_out_test")][0]
            rl_execution.loc[first_test, "trade_id"] = str(partitions["train"].iloc[0]["trade_id"])
        if attack == "rl_policy_quantile_drift":
            rl_execution.loc[:, "entry_threshold_calibration_quantile"] = 0.6

        evaluation_rows = []
        for split in ("validation", "held_out_test"):
            source = rl_execution.loc[rl_execution["evaluation_split"].eq(split)]
            if source.empty:
                continue
            entered = source["simulation_reason"].eq("entered")
            baseline = return_summary(
                "non_rl_baseline",
                source,
                source["base_return"],
                len(source),
                source_frame=source,
            )
            policy = return_summary(
                "safe_rl_policy",
                source.loc[entered],
                source.loc[entered, "simulated_return"],
                len(source),
                source_frame=source,
            )
            baseline["evaluation_split"] = split
            policy["evaluation_split"] = split
            evaluation_rows.extend((baseline, policy))
        evaluation = pd.DataFrame(evaluation_rows)
        if attack == "rl_forged_evaluation_summary":
            safe_test = evaluation["variant"].eq("safe_rl_policy") & evaluation[
                "evaluation_split"
            ].eq("held_out_test")
            evaluation.loc[safe_test, "pair_concentration"] = 0.01
        rl_acceptance = rl_acceptance_report(evaluation)
        if attack == "rl_acceptance_mismatch":
            rl_acceptance.loc[:, "rl_profit_factor"] = (
                pd.to_numeric(rl_acceptance["rl_profit_factor"], errors="coerce") + 0.25
            )
        active_pointer = json.loads(
            (root / "fake-learning" / "active_pointer.json").read_text(encoding="utf-8")
        )
        lineage = {
            "training_dataset_id": dataset_id,
            "training_dataset_sha256": active_pointer.get("active_dataset_sha256", ""),
            "accepted": True,
            "registered_policy_parameters": {
                "train_fraction": 0.6,
                "validation_fraction": 0.2,
                "minimum_rows": [50, 30, 30],
                "entry_threshold_quantile": (0.6 if attack == "rl_policy_quantile_drift" else 0.7),
            },
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        return CommandResult(
            paths={
                "acceptance_report": csv_artifact("rl_acceptance.csv", rl_acceptance),
                "evaluation_report": csv_artifact("rl_evaluation.csv", evaluation),
                "execution_backtest": csv_artifact("rl_execution_backtest.csv", rl_execution),
                "split_audit": csv_artifact("rl_split_audit.csv", split_audit),
                "leakage_audit": csv_artifact(
                    "rl_leakage.csv",
                    pd.DataFrame(
                        [
                            {
                                "uses_future_data": False,
                                "global_label_purge": True,
                                "split_status": "ready",
                                "leakage_blocker": "",
                            }
                        ]
                    ),
                ),
                "feature_schema": json_artifact(
                    "rl_feature_schema.json",
                    {
                        "forbidden_policy_action_inputs": [
                            "hold_bars",
                            "trade_bars",
                            "max_adverse_excursion",
                            "max_favorable_excursion",
                            "exit_timestamp",
                        ],
                        "testnet_order_authority": False,
                        "live_trading_authorized": False,
                    },
                ),
                "lineage_report_json": json_artifact("rl_lineage.json", lineage),
            },
            summary=summary("rl_runner", accepted=True),
        )

    def governance_builder(**_):
        calls.append("governance_builder")
        authority_payload = (
            {}
            if attack == "empty_governance"
            else {
                "schema_version": "thewiz.agent_learning_governance.v1",
                "active_dataset_id": dataset_id,
                "training_dataset_lineage_ready": True,
                "model_active_dataset_lineage_matches": True,
                "model_artifact_hash_matches": True,
                "rl_active_dataset_lineage_matches": True,
                "out_of_sample_incremental_edge_accepted": True,
                "exact_mode_trade_provenance_ready": True,
                "strict_cost_training_evidence_ready": True,
                "training_dataset_lineage_blockers": [],
                "rl_global_label_purge_ready": True,
                "rl_validation_passed": True,
                "rl_held_out_test_passed": True,
                "rl_out_of_sample_accepted": True,
                "model_authority": "RESEARCH_ONLY",
                "blockers": (
                    ["realized_testnet_sample_not_available"]
                    if attack == "deferred_testnet_blocker"
                    else []
                ),
                "quantization_authorized": False,
                "promotion_authority": False,
                "testnet_candidate_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
        return CommandResult(
            paths={
                "model_authority": json_artifact(
                    "model_authority.json",
                    authority_payload,
                )
            },
            summary=summary("governance_builder", status="PASS"),
        )

    return {
        "dataset_builder": dataset_builder,
        "promoter": promoter,
        "trainer": trainer,
        "model_backtester": model_backtester,
        "rl_runner": rl_runner,
        "governance_builder": governance_builder,
    }


def test_registered_learning_executes_once_and_binds_required_artifacts(tmp_path):
    execution = _registered_stage4_fixture(tmp_path)
    calls: list[str] = []
    kwargs = {
        "root": tmp_path,
        "now": NOW,
        "execute": True,
        "execution_receipt_path": execution,
        **_registered_learning_runners(tmp_path, calls),
    }

    first = run_registered_learning_research(**kwargs)
    second = run_registered_learning_research(**kwargs)
    receipt = json.loads(first.paths["learning_receipt"].read_text())

    assert first.summary["status"] == "PASS_RESEARCH_LEARNING_GATES"
    assert second.summary["status"] == "ALREADY_COMPLETE"
    assert calls == [
        "dataset_builder",
        "promoter",
        "trainer",
        "model_backtester",
        "rl_runner",
        "governance_builder",
    ]
    assert receipt["training_accepted_oos"] is True
    assert receipt["stage5_research_gate_pass"] is True
    assert len(receipt["artifact_hashes"]) == 26
    assert "registered_survivor_support_policy" in receipt["artifact_roles"]
    assert "registered_stage5_protocol" in receipt["artifact_roles"]
    assert "rl_execution_backtest" in receipt["artifact_roles"]
    assert "model_selection_leaderboard" in receipt["artifact_roles"]
    assert "registered_survivor_rl_attribution" in receipt["artifact_roles"]
    assert receipt["testnet_candidate_authority"] is False
    assert receipt["testnet_order_authority"] is False
    learning, model_authority, stage4, learning_path, blocker = _validated_registered_learning(
        root=tmp_path
    )
    assert blocker == ""
    assert learning["learning_id"] == receipt["learning_id"]
    assert model_authority["model_authority"] == "RESEARCH_ONLY"
    assert stage4["execution_id"] == receipt["registered_execution_id"]
    assert learning_path == first.paths["learning_receipt"]
    audit = latest_verified_registered_learning(root=tmp_path)
    assert audit["status"] == "PASS_VERIFIED_REGISTERED_LEARNING_ACCEPTANCE"
    assert audit["evidence_valid"] is True
    assert audit["stage5_research_gate_pass"] is True
    assert audit["registered_execution_id"] == receipt["registered_execution_id"]
    assert audit["testnet_order_authority"] is False
    assert audit["live_trading_authorized"] is False


def test_registered_learning_uses_frozen_model_trade_threshold(tmp_path):
    execution = _registered_stage4_fixture(
        tmp_path,
        support_policy_overrides={"minimum_model_gated_trades": 1_000},
    )

    result = run_registered_learning_research(
        root=tmp_path,
        now=NOW,
        execute=True,
        execution_receipt_path=execution,
        **_registered_learning_runners(tmp_path, []),
    )
    receipt = json.loads(result.paths["learning_receipt"].read_text())

    assert result.summary["status"] == "REJECTED_RESEARCH_LEARNING_GATES"
    assert result.summary["stage5_research_gate_pass"] is False
    assert "model.acceptance_artifact_trades" in receipt["acceptance_blockers"]
    assert "model.gated_trades" in receipt["acceptance_blockers"]
    assert receipt["testnet_candidate_authority"] is False
    assert receipt["testnet_order_authority"] is False
    assert receipt["live_trading_authorized"] is False


def test_release_gate_rejects_rehashed_learning_with_relocated_stage4_receipt(
    tmp_path,
):
    execution = _registered_stage4_fixture(tmp_path)
    calls: list[str] = []
    result = run_registered_learning_research(
        root=tmp_path,
        now=NOW,
        execute=True,
        execution_receipt_path=execution,
        **_registered_learning_runners(tmp_path, calls),
    )
    learning_path = result.paths["learning_receipt"]
    learning = json.loads(learning_path.read_text(encoding="utf-8"))
    original_relative = learning["artifact_roles"]["stage4_execution_receipt"]
    relocated = (
        tmp_path / "data" / "research" / "registered_rerun_executions" / "relocated-stage4.json"
    )
    relocated.write_bytes((tmp_path / original_relative).read_bytes())
    relocated_relative = str(relocated.relative_to(tmp_path))
    learning["artifact_roles"]["stage4_execution_receipt"] = relocated_relative
    learning["artifact_hashes"].pop(original_relative)
    learning["artifact_hashes"][relocated_relative] = _file_hash(relocated)
    learning.pop("receipt_sha256")
    learning["receipt_sha256"] = _payload_hash(learning)
    _write_json(learning_path, learning)

    active_path = tmp_path / "reports" / "active" / "registered_learning_research_status.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))
    active["learning_receipt_sha256"] = _file_hash(learning_path)
    _write_json(active_path, active)

    validated, model, stage4, receipt_path, blocker = _validated_registered_learning(root=tmp_path)

    assert validated == {}
    assert model == {}
    assert stage4 == {}
    assert receipt_path is None
    assert "execution receipt path is not canonical" in blocker


def test_registered_learning_rejection_remains_research_only(tmp_path):
    execution = _registered_stage4_fixture(tmp_path)
    result = run_registered_learning_research(
        root=tmp_path,
        now=NOW,
        execute=True,
        execution_receipt_path=execution,
        **_registered_learning_runners(tmp_path, [], model_accepted=False),
    )
    receipt = json.loads(result.paths["learning_receipt"].read_text())

    assert result.summary["status"] == "REJECTED_RESEARCH_LEARNING_GATES"
    assert result.summary["stage5_research_gate_pass"] is False
    assert receipt["model_accepted_oos"] is False
    assert receipt["testnet_candidate_authority"] is False
    assert receipt["live_trading_authorized"] is False
    audit = latest_verified_registered_learning(root=tmp_path)
    assert audit["status"] == "PASS_VERIFIED_REGISTERED_LEARNING_REJECTION"
    assert audit["evidence_valid"] is True
    assert audit["stage5_research_gate_pass"] is False
    assert audit["acceptance_blockers"]


def test_registered_learning_allows_only_the_deferred_stage6_sample_blocker(tmp_path):
    execution = _registered_stage4_fixture(tmp_path)
    result = run_registered_learning_research(
        root=tmp_path,
        now=NOW,
        execute=True,
        execution_receipt_path=execution,
        **_registered_learning_runners(tmp_path, [], attack="deferred_testnet_blocker"),
    )
    receipt = json.loads(result.paths["learning_receipt"].read_text())

    assert result.summary["status"] == "PASS_RESEARCH_LEARNING_GATES"
    assert receipt["stage5_research_gate_pass"] is True
    assert receipt["testnet_candidate_authority"] is False


@pytest.mark.parametrize(
    ("attack", "expected_blocker"),
    [
        ("model_artifact_rejects", "model.acceptance_artifact_accepted"),
        ("model_take_rate_collapse", "model.acceptance_artifact_take_rate"),
        ("model_gain_concentration", "model.gain_artifact_within_limit"),
        (
            "model_timeframe_selection_concentration",
            "training.pair_timeframe_regime_selection_concentration_within_limit",
        ),
        (
            "model_regime_selection_concentration",
            "model.pair_timeframe_regime_selection_artifact_within_limit",
        ),
        (
            "model_global_label_overlap",
            "training.globally_purged_pair_aware_oos",
        ),
        (
            "model_skips_registered_survivor",
            "model.registered_survivor_taken_support",
        ),
        (
            "model_single_fold_registered_survivor",
            "model.registered_survivor_oos_fold_depth",
        ),
        (
            "model_single_regime_registered_survivor",
            "model.registered_survivor_oos_regime_depth",
        ),
        (
            "model_prediction_trade_id_forged",
            "model.predictions_match_active_dataset",
        ),
        (
            "model_prediction_outcome_forged",
            "model.predictions_match_active_dataset",
        ),
        (
            "model_prediction_row_omitted",
            "model.prediction_membership_matches_registered_protocol",
        ),
        (
            "model_selection_phase_overlap",
            "training.model_selection_isolation",
        ),
        (
            "model_forged_best_model",
            "training.best_model_replayed",
        ),
        (
            "model_selection_leaderboard_forged",
            "training.model_selection_leaderboard_matches",
        ),
        (
            "model_acceptance_mismatch",
            "model.acceptance_artifact_matches_recomputed",
        ),
        (
            "model_backtest_mismatch",
            "model.backtest_artifact_matches_recomputed",
        ),
        (
            "model_score_bucket_mismatch",
            "model.score_bucket_artifact_matches_recomputed",
        ),
        (
            "model_pair_artifact_mismatch",
            "model.pair_artifact_matches_recomputed",
        ),
        (
            "model_gain_artifact_mismatch",
            "model.gain_artifact_matches_recomputed",
        ),
        ("rl_missing_test", "rl.held_out_test_passed"),
        (
            "rl_missing_registered_survivor_test",
            "rl.registered_survivor_validation_and_test_ready",
        ),
        ("rl_concentration", "rl.concentration_within_limit"),
        (
            "rl_acceptance_mismatch",
            "rl.acceptance_artifact_matches_recomputed",
        ),
        (
            "rl_forged_evaluation_summary",
            "rl.evaluation_artifact_matches_raw_execution",
        ),
        (
            "rl_forged_split_audit",
            "rl.split_audit_matches_active_dataset",
        ),
        (
            "rl_execution_membership_swap",
            "rl.execution_membership_matches_active_dataset",
        ),
        (
            "rl_policy_quantile_drift",
            "rl.registered_protocol_lineage",
        ),
        ("empty_governance", "governance.authority_schema"),
    ],
)
def test_registered_learning_reconciles_adversarial_evidence(tmp_path, attack, expected_blocker):
    execution = _registered_stage4_fixture(tmp_path)
    result = run_registered_learning_research(
        root=tmp_path,
        now=NOW,
        execute=True,
        execution_receipt_path=execution,
        **_registered_learning_runners(tmp_path, [], attack=attack),
    )
    receipt = json.loads(result.paths["learning_receipt"].read_text())

    assert result.summary["status"] == "REJECTED_RESEARCH_LEARNING_GATES"
    assert expected_blocker in result.summary["blocker"]
    assert receipt["acceptance_checks"][expected_blocker] is False
    assert receipt["stage5_research_gate_pass"] is False
    assert receipt["testnet_candidate_authority"] is False
    assert receipt["testnet_order_authority"] is False
    assert receipt["live_trading_authorized"] is False


def test_registered_learning_rejects_rehashed_forged_acceptance_receipt(tmp_path):
    execution = _registered_stage4_fixture(tmp_path)
    calls: list[str] = []
    kwargs = {
        "root": tmp_path,
        "now": NOW,
        "execute": True,
        "execution_receipt_path": execution,
        **_registered_learning_runners(tmp_path, calls),
    }
    first = run_registered_learning_research(**kwargs)
    receipt_path = Path(first.paths["learning_receipt"])
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["acceptance_checks"]["model.gated_take_rate"] = False
    payload["receipt_sha256"] = _payload_hash(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    _write_json(receipt_path, payload)
    active_path = tmp_path / "reports" / "active" / "registered_learning_research_status.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))
    active["learning_receipt_sha256"] = _file_hash(receipt_path)
    _write_json(active_path, active)

    audit = latest_verified_registered_learning(root=tmp_path)
    assert audit["status"] == "BLOCKED_REGISTERED_LEARNING_EVIDENCE"
    assert audit["evidence_valid"] is False
    assert "acceptance reconciliation mismatch" in audit["blockers"][0]

    with pytest.raises(ValueError, match="acceptance reconciliation mismatch: acceptance_checks"):
        run_registered_learning_research(**kwargs)


def test_registered_learning_rejects_rehashed_forged_survivor_attribution(
    tmp_path,
):
    execution = _registered_stage4_fixture(tmp_path)
    kwargs = {
        "root": tmp_path,
        "now": NOW,
        "execute": True,
        "execution_receipt_path": execution,
        **_registered_learning_runners(tmp_path, []),
    }
    first = run_registered_learning_research(**kwargs)
    receipt_path = Path(first.paths["learning_receipt"])
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    attribution_path = tmp_path / payload["artifact_roles"]["registered_survivor_attribution"]
    attribution = pd.read_csv(attribution_path)
    attribution.loc[:, "oos_taken_rows"] = 0
    attribution.loc[:, "support_status"] = "READY"
    attribution.to_csv(attribution_path, index=False)
    relative = str(attribution_path.relative_to(tmp_path))
    payload["artifact_hashes"][relative] = _file_hash(attribution_path)
    payload["receipt_sha256"] = _payload_hash(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    _write_json(receipt_path, payload)

    with pytest.raises(
        ValueError,
        match="acceptance reconciliation mismatch",
    ):
        run_registered_learning_research(**kwargs)


@pytest.mark.parametrize(
    ("runner_options", "match"),
    [
        ({"unsafe_stage": "dataset_builder"}, "testnet_candidate_authority"),
        ({"omit_artifact": "model_metrics.json"}, "evidence missing: metrics"),
    ],
)
def test_registered_learning_fails_closed_and_releases_lock(tmp_path, runner_options, match):
    execution = _registered_stage4_fixture(tmp_path)
    with pytest.raises(ValueError, match=match):
        run_registered_learning_research(
            root=tmp_path,
            now=NOW,
            execute=True,
            execution_receipt_path=execution,
            **_registered_learning_runners(tmp_path, [], **runner_options),
        )

    status = json.loads(
        (tmp_path / "reports" / "active" / "registered_learning_research_status.json").read_text()
    )
    assert status["status"] == "FAILED"
    assert status["testnet_order_authority"] is False
    assert not (tmp_path / "reports" / "active" / ".corrective_registered_learning.lock").exists()
