from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from quant_platform import active_pipeline as active_pipeline_module
from quant_platform.active_pipeline import (
    _canonical_hyperliquid_state_rows,
    _active_layer_state_row,
    _command_center_markdown,
    _configured_env_key_present,
    _data_health_rows,
    _filter_predictions_to_model,
    _filter_predictions_to_untouched_evaluation,
    _filter_shortlist_to_visible_dydx_markets,
    _global_label_purge_proven,
    _model_gain_concentration,
    _model_authority_state_row,
    _model_pair_concentration,
    _model_selection_isolation_proven,
    _normalize_registered_history_features,
    _score_bucket_report,
    _select_hyperliquid_training_datasets,
    _trade_gate_metrics,
    _training_threshold_calibration_proven,
    _seven_stage_state_rows,
    _venue_lane_test_rows,
    _venue_route_state_row,
    _wizard_pair_metrics,
    archive_from_index,
    build_artifact_index,
    build_command_dashboard,
    build_market_venue_context,
    build_multi_venue_history_readiness,
    build_pair_universe,
    build_trade_dataset,
    build_venue_lane_test_plan,
    build_venue_route_scorecard,
    export_trade_gate_model,
    focused_paper_validation_rows,
    model_gate_pair_support_report,
    paper_candidate_shortlist_rows,
    promote_trade_dataset,
    system_check,
)
from quant_platform.experiments import PairDataset


def test_canonical_recovery_command_uses_current_savepoint_receipt():
    commands = active_pipeline_module.CANONICAL_COMMANDS
    assert "python scripts/ops/nightly_savepoint.py --status" in commands
    assert not any("build_current_recovery_checkpoint.py" in command for command in commands)
    assert (Path(__file__).resolve().parents[1] / "scripts/ops/nightly_savepoint.py").is_file()


def test_trade_gate_diagnostics_are_scoped_to_selected_model_only():
    summary = pd.DataFrame(
        [
            {
                "model_name": "selected",
                "promising": True,
                "selection_score": 10.0,
                "profit_factor_delta": 0.5,
                "sharpe_delta": 0.5,
                "selection_isolation_scheme": (
                    "chronological_model_selection_then_untouched_evaluation_v1"
                ),
                "selection_folds": 1,
                "untouched_evaluation_folds": 2,
                "evaluation_scheme": (
                    "globally_purged_embargoed_pair_aware_timestamp_groups_v2"
                ),
                "evaluation_median_filtered_profit_factor": 1.5,
                "evaluation_median_filtered_sharpe": 1.0,
                "evaluation_worst_filtered_drawdown": 0.2,
                "evaluation_profit_factor_delta": 0.5,
                "evaluation_sharpe_delta": 0.5,
                "evaluation_drawdown_delta": -0.1,
                "evaluation_median_take_rate": 0.2,
                "evaluation_total_filtered_trades": 30,
            },
            {
                "model_name": "rejected",
                "promising": False,
                "selection_score": 0.0,
                "profit_factor_delta": -0.5,
                "sharpe_delta": -0.5,
            },
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "trade_id": f"selected-{index}",
                "model_name": "selected",
                "pair": "A-B",
                "probability_profitable": probability,
                "realized_return": realized_return,
                "shadow_take": probability >= 0.7,
                "split_scheme": (
                    "globally_purged_embargoed_pair_aware_timestamp_groups_v2"
                ),
                "global_label_purge": True,
                "global_label_overlap_rows_after_purge": 0,
                "test_start": "2026-02-01T00:00:00+00:00",
                "train_label_end_max": "2026-01-31T23:00:00+00:00",
                "selection_phase": (
                    "model_selection"
                    if index == 0
                    else "untouched_evaluation"
                ),
                "selection_isolation_scheme": (
                    "chronological_model_selection_then_untouched_evaluation_v1"
                ),
                "selection_evaluation_boundary_scheme": (
                    "label_complete_chronology_gap_v1"
                ),
                "chronology_gap_folds": "",
                "selection_label_end_boundary": (
                    pd.Timestamp("2026-01-01 01:00", tz="UTC").isoformat()
                ),
                "untouched_evaluation_start_boundary": (
                    pd.Timestamp("2026-01-02", tz="UTC").isoformat()
                ),
                "fold": index,
                "entry_timestamp": (
                    pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(days=index)
                ).isoformat(),
                "exit_timestamp": (
                    pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(days=index, hours=1)
                ).isoformat(),
            }
            for index, (probability, realized_return) in enumerate(
                [(0.2, -0.03), (0.6, 0.01), (0.8, 0.04)]
            )
        ]
        + [
            {
                "trade_id": f"rejected-{index}",
                "model_name": "rejected",
                "pair": "X-Y",
                "probability_profitable": probability,
                "realized_return": realized_return,
                "shadow_take": probability >= 0.7,
                "split_scheme": (
                    "globally_purged_embargoed_pair_aware_timestamp_groups_v2"
                ),
                "global_label_purge": True,
                "global_label_overlap_rows_after_purge": 0,
                "test_start": "2026-02-01T00:00:00+00:00",
                "train_label_end_max": "2026-01-31T23:00:00+00:00",
                "selection_phase": (
                    "model_selection"
                    if index == 0
                    else "untouched_evaluation"
                ),
                "selection_isolation_scheme": (
                    "chronological_model_selection_then_untouched_evaluation_v1"
                ),
                "selection_evaluation_boundary_scheme": (
                    "label_complete_chronology_gap_v1"
                ),
                "chronology_gap_folds": "",
                "selection_label_end_boundary": (
                    pd.Timestamp("2026-01-01 01:00", tz="UTC").isoformat()
                ),
                "untouched_evaluation_start_boundary": (
                    pd.Timestamp("2026-01-02", tz="UTC").isoformat()
                ),
                "fold": index,
                "entry_timestamp": (
                    pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(days=index)
                ).isoformat(),
                "exit_timestamp": (
                    pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(days=index, hours=1)
                ).isoformat(),
            }
            for index, (probability, realized_return) in enumerate(
                [(0.2, 0.04), (0.6, 0.01), (0.8, -0.03)]
            )
        ]
    )

    metrics = _trade_gate_metrics(summary, predictions)
    selected_model = _filter_predictions_to_model(
        predictions, metrics["best_model"]
    )
    selected = _filter_predictions_to_untouched_evaluation(selected_model)
    buckets = _score_bucket_report(selected)
    concentration = _model_pair_concentration(selected)

    assert metrics["best_model"] == "selected"
    assert metrics["model_winner_replayed"] is True
    assert metrics["selection_candidate_eligible"] is False
    assert "selection_candidate_eligible" in metrics["failing_checks"]
    assert metrics["accepted"] is False
    assert metrics["score_buckets_monotonic"] is True
    assert metrics["diagnostic_prediction_rows"] == 2
    assert metrics["all_model_prediction_rows"] == 6
    assert buckets["rows"].sum() == 2
    assert buckets["model_name"].eq("selected").all()
    assert concentration["pair"].tolist() == ["A-B"]
    assert concentration["model_name"].eq("selected").all()
    assert _global_label_purge_proven(selected) is True
    assert _model_selection_isolation_proven(selected_model) is True
    missing_boundary = selected_model.drop(
        columns=["selection_evaluation_boundary_scheme"]
    )
    assert _model_selection_isolation_proven(missing_boundary) is False
    tampered_boundary = selected_model.copy()
    tampered_boundary["untouched_evaluation_start_boundary"] = (
        "2026-01-01T00:30:00+00:00"
    )
    assert _model_selection_isolation_proven(tampered_boundary) is False
    assert _training_threshold_calibration_proven(selected_model) is False

    calibrated = selected_model.assign(
        threshold_calibration_scheme="training_only_minimum_participation_v1",
        minimum_training_take_rate=0.10,
        training_take_rate_at_threshold=0.10,
        training_take_rate_floor_pass=True,
    )
    assert _training_threshold_calibration_proven(calibrated) is True
    calibrated.loc[calibrated.index[0], "training_take_rate_at_threshold"] = 0.01
    assert _training_threshold_calibration_proven(calibrated) is False

    low_participation = summary.copy()
    low_participation.loc[
        low_participation["model_name"].eq("selected"),
        "evaluation_median_take_rate",
    ] = 0.075
    low_participation_metrics = _trade_gate_metrics(
        low_participation, predictions
    )
    assert "take_rate_min" in low_participation_metrics["failing_checks"]


def test_global_label_purge_proof_fails_on_overlapping_training_label():
    predictions = pd.DataFrame(
        [
            {
                "split_scheme": (
                    "globally_purged_embargoed_pair_aware_timestamp_groups_v2"
                ),
                "global_label_purge": True,
                "global_label_overlap_rows_after_purge": 0,
                "test_start": "2026-02-01T00:00:00+00:00",
                "train_label_end_max": "2026-02-01T00:01:00+00:00",
            }
        ]
    )

    assert _global_label_purge_proven(predictions) is False


def test_missing_selected_model_predictions_fail_closed():
    predictions = pd.DataFrame(
        [{"model_name": "other", "probability_profitable": 0.9}]
    )

    filtered = _filter_predictions_to_model(predictions, "missing")

    assert filtered.empty


def test_model_gain_concentration_exposes_pair_and_timeframe_dominance():
    predictions = pd.DataFrame(
        [
            {
                "model_name": "selected",
                "pair": pair,
                "timeframe": timeframe,
                "regime": "range",
                "strategy_name": "zscore",
                "realized_return": value,
                "shadow_take": True,
            }
            for pair, timeframe, value in [
                ("A-B", "1d", 0.80),
                ("A-B", "1d", 0.10),
                ("C-D", "1h", 0.10),
            ]
        ]
    )

    report = _model_gain_concentration(predictions)
    pair_top = report[report["dimension"].eq("pair")].iloc[0]
    timeframe_top = report[report["dimension"].eq("timeframe")].iloc[0]

    assert pair_top["value"] == "A-B"
    assert pair_top["share_of_positive_returns"] == pytest.approx(0.9)
    assert timeframe_top["value"] == "1d"
    assert timeframe_top["share_of_positive_returns"] == pytest.approx(0.9)


def test_training_dataset_selects_one_deepest_hyperliquid_history_per_cell():
    def dataset(rows, venue, source):
        return PairDataset(
            "ETH-PYTH",
            pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2026-01-01", periods=rows, freq="h", tz="UTC"
                    ),
                    "exchange": venue,
                    "interval": "1h",
                    "source_path": source,
                }
            ),
        )

    selected, audit = _select_hyperliquid_training_datasets(
        [
            dataset(20, "hyperliquid", "short.json"),
            dataset(50, "hyperliquid", "deep.json"),
            dataset(100, "dydx", "other-venue.json"),
        ]
    )

    assert len(selected) == 1
    assert len(selected[0].frame) == 50
    assert audit["selection_status"].eq("SELECTED_CANONICAL").sum() == 1
    assert audit["selection_status"].eq("REJECTED_OVERLAPPING_HISTORY").sum() == 1
    assert audit["selection_status"].eq("REJECTED_OUTSIDE_TARGET_VENUE").sum() == 1
    assert not audit["testnet_order_authority"].any()


def test_registered_history_feature_normalization_keeps_provenance():
    frame = pd.DataFrame(
        {
            "math_v2_half_life": [12.0, None],
            "research_proxy_half_life": [20.0, 30.0],
            "research_proxy_ecm_x": [-0.2, -0.1],
        }
    )

    normalized = _normalize_registered_history_features(frame)

    assert normalized["half_life"].tolist() == [12.0, 30.0]
    assert normalized["half_life_feature_source"].tolist() == [
        "math_v2_half_life",
        "research_proxy_half_life",
    ]
    assert normalized["ecm_x_feature_source"].eq(
        "research_proxy_ecm_x"
    ).all()


def test_trade_dataset_promotion_preserves_prior_dataset_and_verifies_hashes(
    tmp_path,
):
    data_ml = tmp_path / "data" / "ml"
    reports = tmp_path / "reports" / "ml"
    build = data_ml / "dataset_builds" / "tradedataset-test"
    build.mkdir(parents=True)
    reports.mkdir(parents=True)
    prior = data_ml / "trade_training_dataset.csv"
    pd.DataFrame([{"trade_id": "old", "source_venue": "hyperliquid"}]).to_csv(
        prior, index=False
    )
    artifacts = {
        "dataset": build / "trade_training_dataset.csv",
        "leakage_audit": build / "leakage_audit.csv",
        "source_selection": build / "trade_dataset_source_selection.csv",
        "history_registry_audit": build
        / "trade_dataset_history_registry_audit.csv",
        "summary": build / "trade_dataset_summary.csv",
    }
    pd.DataFrame(
        [
            {
                "trade_id": "new",
                "source_venue": "hyperliquid",
                "pair": "ETH-PYTH",
                "exact_mode": "Copula",
            }
        ]
    ).to_csv(artifacts["dataset"], index=False)
    pd.DataFrame([{"trade_id": "new", "leakage_blocker": ""}]).to_csv(
        artifacts["leakage_audit"], index=False
    )
    pd.DataFrame(
        [
            {
                "selection_status": "SELECTED_CANONICAL",
                "strict_observed_cost_ready": True,
            }
        ]
    ).to_csv(
        artifacts["source_selection"], index=False
    )
    pd.DataFrame([{"registry_status": "HASH_VERIFIED_READY"}]).to_csv(
        artifacts["history_registry_audit"], index=False
    )
    pd.DataFrame([{"rows": 1}]).to_csv(artifacts["summary"], index=False)
    execution_receipt = build / "registered_execution_receipt.json"
    execution_receipt.write_text(
        json.dumps({"execution_id": "registered-execution-1"}), encoding="utf-8"
    )

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    pointer = {
        "status": "VALIDATED_CANDIDATE",
        "dataset_id": "tradedataset-test",
        "dataset_path": str(artifacts["dataset"].relative_to(tmp_path)),
        "leakage_audit_path": str(
            artifacts["leakage_audit"].relative_to(tmp_path)
        ),
        "source_selection_path": str(
            artifacts["source_selection"].relative_to(tmp_path)
        ),
        "history_registry_audit_path": str(
            artifacts["history_registry_audit"].relative_to(tmp_path)
        ),
        "summary_path": str(artifacts["summary"].relative_to(tmp_path)),
        "parquet_path": "",
        "dataset_sha256": digest(artifacts["dataset"]),
        "leakage_audit_sha256": digest(artifacts["leakage_audit"]),
        "source_selection_sha256": digest(artifacts["source_selection"]),
        "history_registry_audit_sha256": digest(
            artifacts["history_registry_audit"]
        ),
        "exact_modes": ["Copula"],
        "strict_cost_history_intersections": 1,
        "research_acceptance_blockers": [],
        "registered_contract_id": "registered-contract-1",
        "registered_execution_id": "registered-execution-1",
        "registered_execution_receipt_path": str(
            execution_receipt.relative_to(tmp_path)
        ),
        "registered_execution_receipt_sha256": digest(execution_receipt),
        "source_family_sha256": "a" * 64,
        "exact_mode_parity_sha256": "b" * 64,
        "full_family_accounted": True,
        "causal_entry_features_proven": True,
        "strict_cost_coverage_complete": True,
        "cost_model_scope": "pair_specific_strict_observed_costs",
    }
    pointer_path = data_ml / "candidate_trade_dataset.json"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    result = promote_trade_dataset(root=tmp_path)
    active = json.loads(result.paths["active_pointer"].read_text(encoding="utf-8"))

    assert active["dataset_id"] == "tradedataset-test"
    assert active["model_retraining_required"] is True
    assert pd.read_csv(result.paths["dataset_csv"])["trade_id"].tolist() == ["new"]
    assert result.paths["dataset_csv"].stat().st_nlink == 1
    assert (
        result.paths["dataset_csv"].stat().st_ino
        != artifacts["dataset"].stat().st_ino
    )
    superseded = tmp_path / active["superseded_dataset_receipt"]
    assert superseded.is_file()
    preserved = pd.read_csv(superseded.parent / "trade_training_dataset.csv")
    assert preserved["trade_id"].tolist() == ["old"]


def test_trade_dataset_promotion_honors_declared_research_blockers(tmp_path):
    data_ml = tmp_path / "data" / "ml"
    reports = tmp_path / "reports" / "ml"
    build = data_ml / "dataset_builds" / "tradedataset-blocked"
    build.mkdir(parents=True)
    reports.mkdir(parents=True)
    dataset = build / "trade_training_dataset.csv"
    audit = build / "leakage_audit.csv"
    selection = build / "trade_dataset_source_selection.csv"
    registry = build / "trade_dataset_history_registry_audit.csv"
    summary = build / "trade_dataset_summary.csv"
    pd.DataFrame(
        [
            {
                "trade_id": "trade-1",
                "source_venue": "hyperliquid",
                "pair": "ETH-PYTH",
                "exact_mode": "Copula",
            }
        ]
    ).to_csv(dataset, index=False)
    pd.DataFrame([{"trade_id": "trade-1", "leakage_blocker": ""}]).to_csv(
        audit, index=False
    )
    pd.DataFrame(
        [
            {
                "selection_status": "SELECTED_CANONICAL",
                "strict_observed_cost_ready": True,
            }
        ]
    ).to_csv(selection, index=False)
    pd.DataFrame([{"registry_status": "HASH_VERIFIED_READY"}]).to_csv(
        registry, index=False
    )
    pd.DataFrame([{"rows": 1}]).to_csv(summary, index=False)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    pointer = {
        "status": "VALIDATED_CANDIDATE",
        "dataset_id": "tradedataset-blocked",
        "dataset_path": str(dataset.relative_to(tmp_path)),
        "leakage_audit_path": str(audit.relative_to(tmp_path)),
        "source_selection_path": str(selection.relative_to(tmp_path)),
        "history_registry_audit_path": str(registry.relative_to(tmp_path)),
        "summary_path": str(summary.relative_to(tmp_path)),
        "parquet_path": "",
        "dataset_sha256": digest(dataset),
        "leakage_audit_sha256": digest(audit),
        "source_selection_sha256": digest(selection),
        "history_registry_audit_sha256": digest(registry),
        "exact_modes": ["Copula"],
        "strict_cost_history_intersections": 1,
        "research_acceptance_blockers": [
            "registered_exact_mode_lineage_missing"
        ],
    }
    pointer_path = data_ml / "candidate_trade_dataset.json"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(SystemExit, match="registered_exact_mode_lineage_missing"):
        promote_trade_dataset(root=tmp_path)


def test_canonical_current_state_uses_decision_and_audit_truth(tmp_path):
    council = tmp_path / "reports" / "orchestration" / "teacher_council"
    council.mkdir(parents=True)
    pd.DataFrame([{"selection_status": "BLOCKED"}]).to_csv(council / "statistical_selection_controls.csv", index=False)
    pd.DataFrame([{"status": "BLOCKED", "action": "abstain"}]).to_csv(council / "council_decisions.csv", index=False)
    pd.DataFrame([{"status": "PASS"}, {"status": "BLOCKED"}]).to_csv(council / "student_training_readiness.csv", index=False)
    pd.DataFrame([{"verdict": "veto"}]).to_csv(council / "portfolio_critic.csv", index=False)
    authority = pd.Series(
        {
            "run_id": "run-1",
            "candidate_set_id": "set-1",
            "status": "RESEARCH_ONLY",
            "research_ready": True,
            "paper_ready": False,
            "execution_allowed": False,
            "blocker": "evidence_not_ready",
        }
    )

    rows = {row["area"]: row for row in _canonical_hyperliquid_state_rows(tmp_path, authority)}

    assert rows["teacher_council"]["ready"] is False
    assert rows["student_learning"]["ready"] is False
    assert rows["portfolio_critic"]["ready"] is False
    assert rows["hyperliquid_walkforward"]["ready"] is False


def test_canonical_current_state_scopes_bandit_blockers_away_from_supervised_student(tmp_path):
    council = tmp_path / "reports" / "orchestration" / "teacher_council"
    council.mkdir(parents=True)
    pd.DataFrame([{"selection_status": "BLOCKED"}]).to_csv(council / "statistical_selection_controls.csv", index=False)
    pd.DataFrame([{"status": "BLOCKED", "action": "abstain"}]).to_csv(council / "council_decisions.csv", index=False)
    pd.DataFrame(
        [
            {"scope": "all_learning", "status": "PASS"},
            {"scope": "supervised_student", "status": "PASS"},
            {"scope": "contextual_bandit", "status": "BLOCKED"},
        ]
    ).to_csv(council / "student_training_readiness.csv", index=False)
    pd.DataFrame([{"verdict": "veto"}]).to_csv(council / "portfolio_critic.csv", index=False)
    authority = pd.Series(
        {
            "run_id": "run-1",
            "candidate_set_id": "set-1",
            "status": "RESEARCH_ONLY",
            "execution_allowed": False,
        }
    )

    rows = {row["area"]: row for row in _canonical_hyperliquid_state_rows(tmp_path, authority)}

    assert rows["student_learning"]["ready"] is True


def test_artifact_index_classifies_without_moving_files(tmp_path):
    (tmp_path / "src" / "quant_platform").mkdir(parents=True)
    (tmp_path / "src" / "quant_platform" / "cli.py").write_text("", encoding="utf-8")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "evidence.csv").write_text("value\n1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Test repository\n", encoding="utf-8")
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "dependency.py").write_text("", encoding="utf-8")
    (tmp_path / "node_modules" / "package").mkdir(parents=True)
    (tmp_path / "node_modules" / "package" / "index.js").write_text("", encoding="utf-8")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache" / "README.md").write_text("", encoding="utf-8")
    (tmp_path / ".venv313" / "lib").mkdir(parents=True)
    (tmp_path / ".venv313" / "lib" / "dependency.py").write_text("", encoding="utf-8")
    (tmp_path / "pytest-of-user" / "case").mkdir(parents=True)
    (tmp_path / "pytest-of-user" / "case" / "evidence.csv").write_text("", encoding="utf-8")
    (tmp_path / "data" / "research").mkdir(parents=True)
    (tmp_path / "data" / "research" / "receipt.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".DS_Store").write_text("", encoding="utf-8")

    result = build_artifact_index(root=tmp_path)

    assert all(path.is_relative_to(tmp_path) for path in result.paths.values())
    frame = pd.read_csv(result.paths["artifact_index"])

    assert not frame.empty
    assert {"active", "historical_evidence", "do_not_move"}.issubset(set(frame["status"]))
    assert Path("src/quant_platform/cli.py").as_posix() in set(frame["path"])
    assert Path("README.md").as_posix() in set(frame["path"])
    assert not frame["path"].str.startswith(
        (".venv/", ".venv313/", "node_modules/", ".pytest_cache/", "pytest-of-user/")
    ).any()
    assert ".DS_Store" not in set(frame["path"])
    research_row = frame.loc[frame["path"].eq("data/research/receipt.json")].iloc[0]
    assert research_row["status"] == "historical_evidence"


def test_active_layer_state_rejects_polluted_artifact_index(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "path": ".venv/lib/dependency.py",
                "artifact_type": "py",
                "status": "unknown",
                "source_system": "unknown",
                "created_or_modified_at": "2026-08-11T00:00:00Z",
                "used_by_active_pipeline": False,
                "evidence_value": "none",
                "safe_to_archive_later": False,
                "reason": "unclassified",
                "notes": "",
            }
        ]
    ).to_csv(active / "artifact_index.csv", index=False)

    row = _active_layer_state_row(tmp_path)

    assert row["ready"] is False
    assert "generated_dependency_paths:1" in row["blocker"]


def test_model_state_uses_authority_instead_of_artifact_existence(tmp_path):
    active = tmp_path / "reports" / "active"
    model_dir = tmp_path / "models" / "trade_gate"
    active.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    (model_dir / "model.pkl").write_bytes(b"research model")
    (active / "model_authority_status.json").write_text(
        json.dumps(
            {
                "model_authority": "RESEARCH_ONLY",
                "out_of_sample_incremental_edge_accepted": False,
                "score_buckets_monotonic": False,
                "rl_out_of_sample_accepted": False,
                "blockers": ["model_incremental_edge_not_accepted"],
            }
        ),
        encoding="utf-8",
    )

    row = _model_authority_state_row(tmp_path)

    assert row["ready"] is False
    assert row["status"] == "RESEARCH_ONLY"
    assert row["blocker"] == "model_incremental_edge_not_accepted"
    assert "artifact_exists=True" in row["detail"]


def test_seven_stage_state_rows_preserve_canonical_gate_status(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "stage": 1,
                "objective": "daily receipts",
                "status": "IN_PROGRESS",
                "evidence_progress": "2/7",
                "blocker": "five_days_missing",
                "next_action": "wait_for_next_daily_receipt",
            },
            {
                "stage": 2,
                "objective": "cost evidence",
                "status": "PASS",
                "evidence_progress": "ready",
                "blocker": "",
                "next_action": "monitor",
            },
        ]
    ).to_csv(active / "seven_stage_goal_checkpoint.csv", index=False)

    rows = {row["area"]: row for row in _seven_stage_state_rows(tmp_path)}

    assert rows["seven_stage_1"]["ready"] is False
    assert rows["seven_stage_1"]["status"] == "IN_PROGRESS"
    assert rows["seven_stage_1"]["blocker"] == "five_days_missing"
    assert rows["seven_stage_2"]["ready"] is True
    assert rows["seven_stage_2"]["status"] == "PASS"


def test_pair_universe_separates_discovery_from_acceptance_and_blocks_vendor_only_promote(
    tmp_path,
):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT/FIDAUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "FIDAUSDT",
                "exchange": "binance",
                "exact_mode": "Copula",
                "mode_valid": True,
                "sharpe": 2.4,
                "returns_total": 0.3,
                "source_authority": "discovery_only",
                "source_timestamp": "2026-08-05T16:00:00Z",
                "source_fresh": True,
                "source_health": "healthy",
                "evidence_path": "data/processed/wizard_evidence.csv",
            }
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)

    result = build_pair_universe(root=tmp_path)

    assert all(path.is_relative_to(tmp_path) for path in result.paths.values())
    frame = pd.read_csv(result.paths["pair_universe"])

    assert not frame.empty
    assert {"discovery_score", "acceptance_score", "decision_bucket", "decision_reason", "evidence_path"}.issubset(frame.columns)
    promoted = frame[frame["decision_bucket"] == "PROMOTE"]
    if not promoted.empty:
        assert (promoted["acceptance_score"] >= 70).all()
    assert not frame["decision_reason"].astype(str).str.contains("dashboard_only_promote", case=False).any()


def test_wizard_pair_metrics_matches_normalized_legs_and_excludes_stale_score_inputs():
    tables = {
        "evidence": pd.DataFrame(
            [
                {
                    "pair": "ETHUSDT/FIDAUSDT",
                    "asset_x": "ETHUSDT",
                    "asset_y": "FIDAUSDT",
                    "exchange": "binance",
                    "exact_mode": "Copula",
                    "mode_valid": True,
                    "sharpe": 9.0,
                    "returns_total": 0.9,
                    "source_authority": "discovery_only",
                    "source_timestamp": "2026-07-01T00:00:00Z",
                    "source_fresh": False,
                    "source_health": "healthy",
                },
                {
                    "pair": "ETHUSDT/FIDAUSDT",
                    "asset_x": "ETHUSDT",
                    "asset_y": "FIDAUSDT",
                    "exchange": "binance",
                    "exact_mode": "Copula",
                    "mode_valid": True,
                    "sharpe": 2.4,
                    "returns_total": 0.3,
                    "source_authority": "discovery_only",
                    "source_timestamp": "2026-08-05T16:00:00Z",
                    "source_fresh": True,
                    "source_health": "healthy",
                    "stationarity_status": "engle_granger",
                    "engle_granger_cointegrated": True,
                    "engle_granger_trend": False,
                    "johansen_cointegrated": False,
                    "zscore_last": -2.1,
                    "zscore_roll_last": -1.2,
                    "volume_min": 185_000,
                },
            ]
        ),
        "diagnostics": pd.DataFrame(
            [
                {
                    "pair": "ETHUSDT/FIDAUSDT",
                    "asset_x": "ETHUSDT",
                    "asset_y": "FIDAUSDT",
                    "exact_mode": "Copula",
                    "wizard_diagnostic_score": 42.0,
                    "source_fresh": False,
                    "source_health": "healthy",
                },
                {
                    "pair": "ETHUSDT/FIDAUSDT",
                    "asset_x": "ETHUSDT",
                    "asset_y": "FIDAUSDT",
                    "exact_mode": "Copula",
                    "wizard_diagnostic_score": 12.0,
                    "source_fresh": True,
                    "source_health": "healthy",
                },
            ]
        ),
        "hypotheses": pd.DataFrame(
            [
                {
                    "pair": "ETHUSDT/FIDAUSDT",
                    "asset_x": "ETHUSDT",
                    "asset_y": "FIDAUSDT",
                    "exact_mode": "Copula",
                    "hypothesis_status": "DISCOVERY_ONLY",
                    "source_fresh": True,
                    "source_health": "healthy",
                }
            ]
        ),
        "parity": pd.DataFrame(),
    }

    metrics = _wizard_pair_metrics("ETH-USD-FIDA-USD", tables)

    assert metrics["best_wizard_sharpe"] == 2.4
    assert metrics["best_wizard_exchange"] == "binance"
    assert metrics["wizard_evidence_state"] == "DISCOVERY_ONLY"
    assert metrics["zscore_score"] == 7.2
    assert metrics["wizard_diagnostic_score"] == 12.0
    assert metrics["cointegration_score"] == 4.0
    assert metrics["wizard_hypothesis_status"] == "DISCOVERY_ONLY"
    assert metrics["wizard_zscore_last"] == -2.1


def test_pair_universe_routes_fresh_wizard_candidate_to_its_research_venue(tmp_path):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT/FIDAUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "FIDAUSDT",
                "exchange": "binance",
                "interval": "daily",
                "exact_mode": "Copula",
                "mode_valid": True,
                "sharpe": 2.4,
                "returns_total": 0.3,
                "source_authority": "discovery_only",
                "source_timestamp": "2026-08-05T16:00:00Z",
                "source_fresh": True,
                "source_health": "healthy",
                "stationarity_status": "engle_granger",
            }
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)

    result = build_pair_universe(root=tmp_path)
    frame = pd.read_csv(result.paths["pair_universe"])
    row = frame.iloc[0]

    assert row["pair"] == "ETHUSDT-FIDAUSDT"
    assert row["exchange"] == "binance"
    assert row["best_wizard_exchange"] == "binance"
    assert row["field_freshness"] == "fresh_wizard_discovery"
    assert row["wizard_evidence_state"] == "DISCOVERY_ONLY"
    assert row["decision_bucket"] == "FETCH_MORE_DATA"
    assert "wizard_discovery_only_needs_pair_detail_and_local_replay" in row["missing_data_reason"]


def test_market_venue_context_keeps_sources_authority_aware(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "asset": "BTC",
                "venue": "dydx",
                "seen_or_tradable": True,
                "execution_decision": "dydx_execution_ok",
                "source": "parseforge/dydx-markets-scraper",
                "reported_24h_volume": 1_000_000,
                "reported_open_interest_native": 100,
                "reported_open_interest_usd": 500_000,
                "funding_rate": 0.0001,
                "liquidity_bucket": "deep",
            }
        ]
    ).to_csv(active / "multi_exchange_liquidity_test_2026-06-25.csv", index=False)

    result = build_market_venue_context(root=tmp_path)

    frame = pd.read_csv(result.paths["market_venue_context"])
    lanes = pd.read_csv(result.paths["venue_lanes"])

    assert not frame.empty
    assert {
        "asset",
        "venue",
        "source_system",
        "execution_authority",
        "promotion_allowed",
        "venue_lane",
        "funding_pulse_status",
        "blocker",
        "evidence_path",
    }.issubset(frame.columns)
    assert "funding_pulse_needs_api_key" in set(frame["blocker"].astype(str))
    context_only = frame[frame["source_system"].isin(["coinglass", "gmx", "dexscreener", "funding_pulse"])]
    assert not context_only["promotion_allowed"].astype(bool).any()
    hyperliquid = frame[frame["venue"].astype(str).str.lower() == "hyperliquid"]
    if not hyperliquid.empty:
        historical_hyperliquid = hyperliquid[hyperliquid["source_system"].astype(str) != "hyperliquid_public_api"]
        if not historical_hyperliquid.empty:
            assert historical_hyperliquid["blocker"].astype(str).str.contains("missing_hyperliquid_local_replay").all()
        public_hyperliquid = hyperliquid[hyperliquid["source_system"].astype(str) == "hyperliquid_public_api"]
        if not public_hyperliquid.empty:
            assert public_hyperliquid["blocker"].astype(str).str.contains("requires_pair_history_cost_slippage_and_preflight").all()
            assert not public_hyperliquid["promotion_allowed"].astype(bool).any()
    assert not lanes.empty
    assert {"asset", "best_lane", "next_action"}.issubset(lanes.columns)


def test_venue_lane_test_plan_routes_hyperliquid_without_promoting(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {"asset": "AAA", "best_lane": "hyperliquid_research_candidate", "blockers": "", "next_action": "collect_history"},
            {"asset": "BBB", "best_lane": "hyperliquid_research_candidate", "blockers": "", "next_action": "collect_history"},
        ]
    ).to_csv(active / "venue_lane_classification.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "AAA-USD/BBB-USD",
                "asset_x": "AAA",
                "asset_y": "BBB",
                "exchange": "hyperliquid",
                "exact_mode": "OU (Spread)",
                "mode_valid": True,
                "source_fresh": True,
                "passes_sharpe_gate": True,
                "sharpe": 2.1,
                "returns_total": 0.12,
                "discovery_min_returns_total": 0.10,
                "source_system": "crypto_wizards_live_scanner_capture",
                "source_authority": "discovery_only",
                "evidence_path": "reports/active/wizard.csv",
            }
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)
    result = build_venue_lane_test_plan(root=tmp_path)

    frame = pd.read_csv(result.paths["venue_lane_test_plan"])

    assert not frame.empty
    assert {"pair_lane", "test_status", "funding_pulse_status", "next_step", "evidence_path"}.issubset(frame.columns)
    assert set(frame["funding_pulse_status"].dropna().unique()) == {"needs_api_key"}
    hyperliquid_rows = frame[frame["pair_lane"].astype(str).str.contains("hyperliquid", na=False)]
    if not hyperliquid_rows.empty:
        assert hyperliquid_rows["test_status"].astype(str).str.contains("hyperliquid").all()
        assert not hyperliquid_rows["next_step"].astype(str).str.contains("promote", case=False, na=False).any()


def test_venue_route_scorecard_requires_current_complete_two_leg_evidence(tmp_path):
    processed = tmp_path / "data" / "processed"
    active = tmp_path / "reports" / "active"
    processed.mkdir(parents=True)
    active.mkdir(parents=True)
    timestamp = "2026-08-05T12:00:00+00:00"
    pair = "BTC-USD-ETH-USD"

    pd.DataFrame(
        [
            {
                "pair": pair,
                "asset_x": "BTC-USD",
                "asset_y": "ETH-USD",
                "combined_score": 91.0,
                "available_venues": "dydx;hyperliquid",
                "funding_drag_bps": 1.1,
                "field_freshness": "current_snapshot",
            }
        ]
    ).to_csv(processed / "pair_universe.csv", index=False)
    context_rows = []
    for venue in ("dydx", "hyperliquid"):
        for asset in ("BTC", "ETH"):
            context_rows.append(
                {
                    "asset": asset,
                    "venue": venue,
                    "tradable": True,
                    "execution_authority": True,
                    "source_timestamp": timestamp,
                    "volume_24h": 2_000_000.0,
                    "open_interest_usd": 10_000_000.0,
                    "funding_rate": "0.0001",
                    "evidence_path": f"reports/active/{venue}_snapshot.csv",
                }
            )
    pd.DataFrame(context_rows).to_csv(processed / "market_venue_context.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": pair,
                "venue": "dydx",
                "execution_ready": True,
                "ready_for_submission": False,
                "cost_model_aligned": True,
                "cost_model_profile": "dydx_perpetual_cost_model",
                "evidence": "reports/paper_venue_preflight.csv",
            },
            {
                "pair": pair,
                "venue": "hyperliquid",
                "execution_ready": True,
                "ready_for_submission": False,
                "cost_model_aligned": False,
                "cost_model_profile": "missing_pair_cost_model",
                "evidence": "reports/paper_venue_preflight.csv",
            },
        ]
    ).to_csv(tmp_path / "reports" / "paper_venue_preflight.csv", index=False)
    pd.DataFrame(
        [
            {"pair": pair, "asset_x": "BTC-USD", "asset_y": "ETH-USD", "wizard_exchange": "dydx", "test_status": "dydx_replayed"},
            {"pair": pair, "asset_x": "BTC-USD", "asset_y": "ETH-USD", "wizard_exchange": "hyperliquid", "test_status": "hyperliquid_history_needed"},
        ]
    ).to_csv(active / "venue_lane_test_plan.csv", index=False)
    pd.DataFrame().to_csv(active / "multi_venue_history_readiness_2026-06-25.csv", index=False)

    result = build_venue_route_scorecard(root=tmp_path, as_of=pd.Timestamp("2026-08-05T12:30:00+00:00"))
    scorecard = pd.read_csv(result.paths["venue_route_scorecard"])
    recommendations = pd.read_csv(result.paths["venue_route_recommendations"])
    by_venue = scorecard.set_index("venue")

    assert bool(by_venue.loc["dydx", "validation_route_ready"]) is True
    assert bool(by_venue.loc["hyperliquid", "research_route_ready"]) is True
    assert bool(by_venue.loc["hyperliquid", "validation_route_ready"]) is False
    assert "missing_pair_cost_model" in by_venue.loc["hyperliquid", "blockers"]
    assert recommendations.loc[0, "recommended_execution_venue"] == "dydx"
    assert pd.isna(recommendations.loc[0, "recommended_paper_venue"])

    stale = build_venue_route_scorecard(root=tmp_path, as_of=pd.Timestamp("2026-08-07T12:30:00+00:00"))
    stale_recommendations = pd.read_csv(stale.paths["venue_route_recommendations"])
    assert pd.isna(stale_recommendations.loc[0, "recommended_execution_venue"])
    assert "stale_or_missing_venue_context" in stale_recommendations.loc[0, "blockers"]


def test_venue_route_scorecard_credits_fresh_hyperliquid_bundle_but_not_missing_costs(tmp_path):
    processed = tmp_path / "data" / "processed"
    active = tmp_path / "reports" / "active"
    processed.mkdir(parents=True)
    active.mkdir(parents=True)
    pair = "BTC-USD-ETH-USD"
    timestamp = "2026-08-05T12:00:00+00:00"
    pd.DataFrame(
        [
            {
                "pair": pair,
                "asset_x": "BTC-USD",
                "asset_y": "ETH-USD",
                "combined_score": 75.0,
                "available_venues": "hyperliquid",
            }
        ]
    ).to_csv(processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "asset": asset,
                "venue": "hyperliquid",
                "tradable": True,
                "execution_authority": True,
                "source_timestamp": timestamp,
                "volume_24h": 2_000_000.0,
                "open_interest_usd": 10_000_000.0,
                "funding_rate": "0.0001",
                "evidence_path": "data/raw/hyperliquid_market_snapshots/snapshot.json",
            }
            for asset in ("BTC", "ETH")
        ]
    ).to_csv(processed / "market_venue_context.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": pair,
                "venue": "hyperliquid",
                "execution_ready": True,
                "ready_for_submission": False,
                "cost_model_aligned": False,
                "cost_model_profile": "missing_pair_cost_model",
                "evidence": "reports/paper_venue_preflight.csv",
            }
        ]
    ).to_csv(tmp_path / "reports" / "paper_venue_preflight.csv", index=False)
    pd.DataFrame().to_csv(active / "venue_lane_test_plan.csv", index=False)
    pd.DataFrame().to_csv(active / "multi_venue_history_readiness_2026-06-25.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": pair,
                "asset_x": "BTC",
                "asset_y": "ETH",
                "venue": "hyperliquid",
                "history_ready": True,
                "history_status": "history_ready_for_local_replay",
                "evidence_path": "reports/active/hyperliquid_research_bundle.csv",
            }
        ]
    ).to_csv(active / "hyperliquid_research_bundle.csv", index=False)

    result = build_venue_route_scorecard(root=tmp_path, as_of=pd.Timestamp("2026-08-05T12:30:00+00:00"))
    row = pd.read_csv(result.paths["venue_route_scorecard"]).iloc[0]

    assert bool(row["history_ready"]) is True
    assert row["history_status"] == "history_ready_for_local_replay"
    assert bool(row["validation_route_ready"]) is False
    assert "missing_pair_cost_model" in row["blockers"]
    assert "missing_venue_specific_slippage_calibration" in row["blockers"]
    assert "hyperliquid_research_bundle.csv" in row["evidence_path"]


def test_venue_route_scorecard_uses_fee_profile_without_confusing_it_for_slippage_calibration(tmp_path):
    processed = tmp_path / "data" / "processed"
    active = tmp_path / "reports" / "active"
    processed.mkdir(parents=True)
    active.mkdir(parents=True)
    pair = "BTC-USD-ETH-USD"
    timestamp = "2026-08-05T12:00:00+00:00"
    pd.DataFrame(
        [{"pair": pair, "asset_x": "BTC-USD", "asset_y": "ETH-USD", "combined_score": 75.0, "available_venues": "hyperliquid"}]
    ).to_csv(processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "asset": asset,
                "venue": "hyperliquid",
                "tradable": True,
                "execution_authority": True,
                "source_timestamp": timestamp,
                "volume_24h": 2_000_000.0,
                "open_interest_usd": 10_000_000.0,
                "funding_rate": "0.0001",
                "evidence_path": "data/raw/hyperliquid_market_snapshots/snapshot.json",
            }
            for asset in ("BTC", "ETH")
        ]
    ).to_csv(processed / "market_venue_context.csv", index=False)
    pd.DataFrame(
        [{"pair": pair, "venue": "hyperliquid", "execution_ready": True, "ready_for_submission": False, "cost_model_aligned": False, "cost_model_profile": "missing_pair_cost_model"}]
    ).to_csv(tmp_path / "reports" / "paper_venue_preflight.csv", index=False)
    pd.DataFrame().to_csv(active / "venue_lane_test_plan.csv", index=False)
    pd.DataFrame().to_csv(active / "multi_venue_history_readiness_2026-06-25.csv", index=False)
    pd.DataFrame(
        [{"pair": pair, "asset_x": "BTC", "asset_y": "ETH", "venue": "hyperliquid", "history_ready": True, "history_status": "history_ready_for_local_replay"}]
    ).to_csv(active / "hyperliquid_research_bundle.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": pair,
                "asset_x": "BTC",
                "asset_y": "ETH",
                "venue": "hyperliquid",
                "cost_model_ready": True,
                "cost_model_status": "official_base_tier_conservative_fee_profile",
                "slippage_model_ready": False,
                "slippage_model_status": "insufficient_l2_depth_samples",
                "evidence_path": "reports/active/hyperliquid_pair_cost_model.csv",
            }
        ]
    ).to_csv(active / "hyperliquid_pair_cost_model.csv", index=False)

    result = build_venue_route_scorecard(root=tmp_path, as_of=pd.Timestamp("2026-08-05T12:30:00+00:00"))
    row = pd.read_csv(result.paths["venue_route_scorecard"]).iloc[0]

    assert bool(row["cost_model_ready"]) is True
    assert row["cost_model_status"] == "official_base_tier_conservative_fee_profile"
    assert bool(row["slippage_model_ready"]) is False
    assert row["slippage_model_status"] == "insufficient_l2_depth_samples"
    assert bool(row["validation_route_ready"]) is False
    assert "missing_pair_cost_model" not in row["blockers"]
    assert "insufficient_l2_depth_samples" in row["blockers"]


def test_venue_route_dashboard_treats_csv_nan_as_no_route():
    recommendations = pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "recommended_execution_venue": float("nan"),
                "recommended_paper_venue": float("nan"),
                "blockers": "missing_matching_venue_history",
            }
        ]
    )
    health = _data_health_rows(pd.DataFrame(), pd.DataFrame(), recommendations)
    route_health = health[health["area"] == "venue_routes_without_execution_evidence"].iloc[0]
    markdown = _command_center_markdown(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        health,
        route_recommendations=recommendations,
    )

    assert bool(route_health["ready"]) is False
    assert "pairs_without_execution_route=1" in route_health["blocker"]
    assert "- evidence-complete execution routes: 0" in markdown
    assert "- paper-submission-ready routes: 0" in markdown


def test_venue_route_current_state_is_research_only_until_execution_evidence(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "recommended_research_venue": "hyperliquid",
                "recommended_execution_venue": float("nan"),
                "recommended_paper_venue": float("nan"),
                "blockers": "missing_matching_venue_history",
            }
        ]
    ).to_csv(active / "venue_route_recommendations.csv", index=False)

    row = _venue_route_state_row(tmp_path)

    assert row["ready"] is False
    assert row["status"] == "research_only"
    assert row["blocker"] == "missing_matching_venue_history"
    assert "execution_routes=0" in row["detail"]


def test_venue_lane_test_plan_falls_back_to_gated_wizard_evidence(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT-TRUMPUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "TRUMPUSDT",
                "exchange": "binance",
                "interval": "daily",
                "exact_mode": "OU (Spread)",
                "mode_valid": True,
                "sharpe": 2.25,
                "returns_total": 0.24,
                "passes_sharpe_gate": True,
                "passes_returns_total_gt_20pct": True,
                "evidence_path": "reports/active/wizard_evidence.csv",
            }
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)

    result = build_venue_lane_test_plan(root=tmp_path)
    frame = pd.read_csv(result.paths["venue_lane_test_plan"])

    assert frame["pair"].tolist() == ["ETHUSDT-TRUMPUSDT"]
    assert frame.iloc[0]["test_status"] == "binance_research_only"
    assert "wizard_evidence.csv" in str(frame.iloc[0]["evidence_path"])


def test_venue_lane_test_plan_keeps_fresh_wizard_modes_when_legacy_queue_exists(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "OLDUSDT/LEGACYUSDT",
                "asset_x": "OLDUSDT",
                "asset_y": "LEGACYUSDT",
                "wizard_exchange": "binance",
                "wizard_sharpe": 2.0,
                "wizard_returns_total": 0.2,
            }
        ]
    ).to_csv(active / "crypto_wizards_next_best_sharpe_returns_queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT/FIDAUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "FIDAUSDT",
                "exchange": "binance",
                "interval": "daily",
                "exact_mode": "Copula",
                "mode_valid": True,
                "sharpe": 2.4,
                "returns_total": 0.3,
                "passes_sharpe_gate": True,
                "passes_returns_total_gt_20pct": True,
                "source_fresh": True,
                "source_authority": "discovery_only",
                "source_timestamp": "2026-08-05T12:00:00Z",
                "hypothesis_status": "DISCOVERY_ONLY",
                "evidence_path": "reports/active/wizard_evidence.csv",
            },
            {
                "pair": "ETHUSDT/FIDAUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "FIDAUSDT",
                "exchange": "binance",
                "interval": "daily",
                "exact_mode": "OU (Spread)",
                "mode_valid": True,
                "sharpe": 2.1,
                "returns_total": 0.25,
                "passes_sharpe_gate": True,
                "passes_returns_total_gt_20pct": True,
                "source_fresh": True,
                "source_authority": "discovery_only",
                "source_timestamp": "2026-08-05T12:00:00Z",
                "hypothesis_status": "DISCOVERY_ONLY",
                "evidence_path": "reports/active/wizard_evidence.csv",
            },
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)

    result = build_venue_lane_test_plan(root=tmp_path)
    frame = pd.read_csv(result.paths["venue_lane_test_plan"])
    fresh = frame[frame["pair"].eq("ETHUSDT/FIDAUSDT")]

    assert set(fresh["exact_mode"]) == {"Copula", "OU (Spread)"}
    assert fresh["source_fresh"].map(bool).all()
    assert set(fresh["source_authority"]) == {"discovery_only"}


def test_venue_lane_test_plan_routes_wizard_non_dydx_as_research_only():
    queue = pd.DataFrame(
        [
            {
                "pair": "ETHUSDT-TRUMPUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "TRUMPUSDT",
                "wizard_exchange": "binance",
                "asset_x_normalized": "ETH-USDT",
                "asset_y_normalized": "TRUMP-USDT",
                "normalized_pair": "ETH-USDT-TRUMP-USDT",
                "wizard_sharpe": 3.37,
                "wizard_returns_total": 0.205,
                "source_path": "reports/active/crypto_wizards_exchange_sample_2026-06-25.csv",
            },
            {
                "pair": "PEPE-USD-ETH-BTC",
                "asset_x": "PEPE-USD",
                "asset_y": "ETH-BTC",
                "scanner_exchange": "coinbase",
                "asset_x_normalized": "PEPE-USD",
                "asset_y_normalized": "ETH-BTC",
                "normalized_pair": "PEPE-USD-ETH-BTC",
                "wizard_sharpe": 2.95,
                "wizard_returns_total": 0.22,
                "source_path": "reports/active/crypto_wizards_exchange_sample_2026-06-25.csv",
            },
            {
                "pair": "ETHUSDT-TRUMPUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "TRUMPUSDT",
                "wizard_exchange": "bybit",
                "asset_x_normalized": "ETH-USDT",
                "asset_y_normalized": "TRUMP-USDT",
                "normalized_pair": "ETH-USDT-TRUMP-USDT",
                "wizard_sharpe": 2.9,
                "wizard_returns_total": 0.18,
                "source_path": "reports/active/crypto_wizards_exchange_sample_2026-06-25.csv",
            },
        ]
    )

    frame = pd.DataFrame(_venue_lane_test_rows(pd.DataFrame(), pd.DataFrame(), queue))

    assert set(frame["wizard_exchange"]) == {"binance", "coinbase", "bybit"}
    assert set(frame["pair_lane"]) == {"binance_research_lane", "coinbase_research_lane", "bybit_research_lane"}
    assert set(frame["test_status"]) == {"binance_research_only", "coinbase_research_only", "bybit_research_only"}
    assert len(frame[frame["normalized_pair"] == "ETH-USDT-TRUMP-USDT"]) == 2
    assert frame["next_step"].astype(str).str.contains("do_not_promote_until").all()


def test_multi_venue_history_readiness_ranks_fetchable_research_candidates(tmp_path, monkeypatch):
    from quant_platform import active_pipeline

    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    monkeypatch.setattr(active_pipeline, "ACTIVE", active)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT-TRUMPUSDT",
                "wizard_exchange": "binance",
                "asset_x": "ETHUSDT",
                "asset_y": "TRUMPUSDT",
                "asset_x_normalized": "ETH-USDT",
                "asset_y_normalized": "TRUMP-USDT",
                "normalized_pair": "ETH-USDT-TRUMP-USDT",
                "sharpe": 3.37,
                "zscore_norm": -0.64,
                "zscore_roll": 0.98,
            },
            {
                "pair": "BNB-USD-ETH-USD",
                "wizard_exchange": "dydx",
                "asset_x": "BNB-USD",
                "asset_y": "ETH-USD",
                "asset_x_normalized": "BNB-USD",
                "asset_y_normalized": "ETH-USD",
                "normalized_pair": "BNB-USD-ETH-USD",
                "sharpe": 1.98,
                "zscore_norm": -0.89,
                "zscore_roll": -0.41,
            },
        ]
    ).to_csv(active / "crypto_wizards_multi_venue_sharpe_rows_2026-06-25.csv", index=False)

    result = build_multi_venue_history_readiness(root=tmp_path, top_n=10)
    frame = pd.read_csv(result.paths["multi_venue_history_readiness"])

    assert {"readiness_status", "history_source_status", "cost_model_status", "next_step"}.issubset(frame.columns)
    binance = frame[frame["wizard_exchange"] == "binance"].iloc[0]
    dydx = frame[frame["wizard_exchange"] == "dydx"].iloc[0]
    assert binance["readiness_status"] == "ready_to_fetch"
    assert "needs_spot_fee_model" in binance["blockers"]
    assert dydx["readiness_status"] == "ready_for_replay"


def test_multi_venue_history_readiness_falls_back_to_queue_when_primary_rows_missing(tmp_path, monkeypatch):
    from quant_platform import active_pipeline

    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    monkeypatch.setattr(active_pipeline, "ACTIVE", active)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT-TRUMPUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "TRUMPUSDT",
                "wizard_exchange": "binance",
                "asset_x_normalized": "ETH-USDT",
                "asset_y_normalized": "TRUMP-USDT",
                "normalized_pair": "ETH-USDT-TRUMP-USDT",
                "sharpe": 3.37,
                "returns_total": 0.205,
            }
        ]
    ).to_csv(active / "crypto_wizards_next_best_sharpe_returns_queue.csv", index=False)

    result = build_multi_venue_history_readiness(root=tmp_path, top_n=10)
    frame = pd.read_csv(result.paths["multi_venue_history_readiness"])

    assert len(frame) == 1
    assert frame.iloc[0]["wizard_exchange"] == "binance"
    assert frame.iloc[0]["readiness_status"] == "ready_to_fetch"
    assert "crypto_wizards_next_best_sharpe_returns_queue.csv" in str(frame.iloc[0]["evidence_path"])


def test_multi_venue_history_readiness_prefers_current_wizard_shortlist(tmp_path, monkeypatch):
    from quant_platform import active_pipeline

    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    monkeypatch.setattr(active_pipeline, "ACTIVE", active)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT/FIDAUSDT",
                "asset_x": "ETHUSDT",
                "asset_y": "FIDAUSDT",
                "exchange": "binance",
                "interval": "daily",
                "period": 365,
                "setup_identity": "ETHUSDT|FIDAUSDT|daily|365|copula",
                "exact_mode": "Copula",
                "sharpe": 2.46,
                "returns_total": 0.41,
                "source_authority": "discovery_only",
                "source_timestamp": "2026-08-05T16:13:59Z",
                "source_fresh": True,
                "discovery_screen_status": "SCREEN_PASS",
                "research_quality_status": "RESEARCH_READY",
                "research_blockers": "",
                "zscore_last": 0.0,
                "zscore_roll_last": 0.0,
            }
        ]
    ).to_csv(active / "wizard_discovery_shortlist.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "STALEUSDT/OLDUSDT",
                "asset_x": "STALEUSDT",
                "asset_y": "OLDUSDT",
                "wizard_exchange": "binance",
                "sharpe": 99.0,
            }
        ]
    ).to_csv(active / "crypto_wizards_multi_venue_sharpe_rows_2026-06-25.csv", index=False)

    result = build_multi_venue_history_readiness(root=tmp_path, top_n=10)
    frame = pd.read_csv(result.paths["multi_venue_history_readiness"])

    assert result.paths["multi_venue_history_readiness"].name == "multi_venue_history_readiness.csv"
    assert list(frame["pair"]) == ["ETHUSDT/FIDAUSDT"]
    assert frame.iloc[0]["wizard_exchange"] == "binance"
    assert frame.iloc[0]["exact_mode"] == "Copula"
    assert bool(frame.iloc[0]["source_fresh"])
    assert frame.iloc[0]["candidate_source_kind"] == "current_wizard_shortlist"


def test_trade_dataset_writes_leakage_audit_and_required_labels(
    tmp_path, monkeypatch
):
    source = PairDataset(
        "ETH-PYTH",
        pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2026-01-01", periods=20, freq="h", tz="UTC"
                ),
                "exchange": "hyperliquid",
                "timeframe": "1h",
                "source_path": "data/raw/pair_details/eth_pyth.json",
                "spread": range(20),
                "zscore": 0.0,
            }
        ),
    )
    registry_audit = pd.DataFrame(
        [{"registry_status": "HASH_VERIFIED_READY", "blocker": ""}]
    )
    trade_row = pd.DataFrame(
        [
            {
                "trade_id": "trade-1",
                "pair": "ETH-PYTH",
                "timeframe": "1h",
                "source_venue": "hyperliquid",
                "source_path": "data/raw/pair_details/eth_pyth.json",
                "exact_mode": "OU Spread",
                "orientation": "x_on_y",
                "strategy_id": 1,
                "strategy_name": "zscore",
                "family": "mean_reversion",
                "entry_timestamp": "2026-01-01T01:00:00+00:00",
                "exit_timestamp": "2026-01-01T02:00:00+00:00",
                "trade_bars": 2,
                "label_profitable": 1,
                "realized_return": 0.01,
                "max_adverse_excursion": -0.002,
                "max_favorable_excursion": 0.012,
                "return_aggregation": "compounded_bar_returns_zero_floor",
                "return_unit": "fraction_of_equity",
            }
        ]
    )
    monkeypatch.setattr(
        active_pipeline_module,
        "_registered_hyperliquid_training_datasets",
        lambda _root: ([source], registry_audit),
    )
    monkeypatch.setattr(
        active_pipeline_module,
        "build_trade_filter_dataset",
        lambda _datasets: trade_row,
    )
    monkeypatch.setattr(
        active_pipeline_module,
        "classify_regimes",
        lambda frame, _config: frame.assign(regime="range"),
    )
    from quant_platform import three_brain_system

    monkeypatch.setattr(
        three_brain_system, "build_shared_outcome_memory", lambda root: None
    )
    monkeypatch.setattr(
        three_brain_system,
        "build_native_outcome_feature_memory",
        lambda root: None,
    )

    result = build_trade_dataset(root=tmp_path)

    dataset = pd.read_csv(result.paths["dataset_csv"])
    audit = pd.read_csv(result.paths["leakage_audit"])

    assert {"good_trade", "profit_after_cost", "max_adverse_excursion", "max_favorable_excursion", "hold_bars", "exit_reason"}.issubset(dataset.columns)
    assert {"uses_future_data", "uses_dashboard_hindsight", "feature_completeness_score", "leakage_blocker", "evidence_path"}.issubset(audit.columns)
    assert not audit["uses_future_data"].astype(bool).any()
    assert dataset["profit_after_cost"].ge(-1.0).all()
    assert dataset["return_aggregation"].eq("compounded_bar_returns_zero_floor").all()
    assert dataset["return_unit"].eq("fraction_of_equity").all()
    assert audit["leakage_blocker"].fillna("").eq("").all()


def test_system_check_reports_active_artifacts(tmp_path):
    for folder in ["data/raw", "data/processed", "reports", "src/quant_platform"]:
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)

    result = system_check(root=tmp_path)

    assert all(path.is_relative_to(tmp_path) for path in result.paths.values())
    frame = pd.read_csv(result.paths["system_check"])

    assert not frame.empty
    assert {"check", "ready", "blocker", "evidence_path", "next_action"}.issubset(frame.columns)
    assert "artifact:pair_universe.csv" in set(frame["check"])
    assert "artifact:wizard_mode_matrix_capture_queue.csv" in set(frame["check"])
    assert "artifact:wizard_mode_comparison.csv" in set(frame["check"])
    assert "artifact:current_wizard_ou_optimal_overlay_ledger.csv" in set(
        frame["check"]
    )
    assert "storage:deep_dashboard_refresh" in set(frame["check"])
    assert "storage:off_volume_archive_destination" in set(frame["check"])
    assert "storage:verified_archive_copy" in set(frame["check"])
    assert "storage:archive_release_dry_run" in set(frame["check"])
    assert "scheduler:live_runtime_contract" in set(frame["check"])
    assert "hyperliquid_testnet:deterministic_lifecycle_protocol" in set(
        frame["check"]
    )


def test_system_check_accepts_only_zero_authority_scheduler_runtime_receipt(
    tmp_path,
):
    for folder in ["data/raw", "data/processed", "reports/active", "src/quant_platform"]:
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    runtime_path = tmp_path / "reports" / "active" / "scheduler_runtime_readiness.json"
    payload = {
        "status": "PASS_SCHEDULER_RUNTIME_READY",
        "agents_ready": 3,
        "agents_expected": 3,
        "checks_passed": 61,
        "checks_total": 61,
        "operational_warnings": ["system_volume_free_space_low"],
        "blockers": [],
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "orders_submitted": 0,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    runtime_path.write_text(json.dumps(payload), encoding="utf-8")

    result = system_check(root=tmp_path)
    frame = pd.read_csv(result.paths["system_check"])
    row = frame.loc[frame["check"].eq("scheduler:live_runtime_contract")].iloc[0]

    assert bool(row["ready"])
    assert row["status"] == "ready_with_operational_warnings"
    assert pd.isna(row["blocker"])

    runtime_path.write_text(
        json.dumps({**payload, "testnet_order_authority": True}),
        encoding="utf-8",
    )
    blocked = system_check(root=tmp_path)
    blocked_frame = pd.read_csv(blocked.paths["system_check"])
    blocked_row = blocked_frame.loc[
        blocked_frame["check"].eq("scheduler:live_runtime_contract")
    ].iloc[0]
    assert not bool(blocked_row["ready"])
    assert "scheduler_runtime_authority_not_zero" in blocked_row["blocker"]


def test_configured_env_key_presence_is_consistent_without_loading_secret(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("HYPERLIQUID_MASTER_ADDRESS", raising=False)
    (tmp_path / ".env.local").write_text(
        "HYPERLIQUID_MASTER_ADDRESS=0xconfigured\nEMPTY_KEY=\n",
        encoding="utf-8",
    )

    assert _configured_env_key_present(tmp_path, "HYPERLIQUID_MASTER_ADDRESS")
    assert not _configured_env_key_present(tmp_path, "EMPTY_KEY")
    assert not _configured_env_key_present(tmp_path, "MISSING_KEY")


def test_atomic_csv_write_preserves_prior_evidence_on_storage_failure(
    tmp_path, monkeypatch
):
    target = tmp_path / "canonical.csv"
    prior = b"pair,status\nBTC-ETH,accepted\n"
    target.write_bytes(prior)

    def fail_after_partial_write(self, destination, *args, **kwargs):
        Path(destination).write_bytes(b"pair,status\nBTC-ETH")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_after_partial_write)

    with pytest.raises(OSError, match="No space left on device"):
        active_pipeline_module._write_csv(pd.DataFrame([{"pair": "SOL-WLD"}]), target)

    assert target.read_bytes() == prior
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_optional_parquet_write_preserves_prior_evidence_on_storage_failure(
    tmp_path, monkeypatch
):
    target = tmp_path / "canonical.parquet"
    prior = b"prior-parquet-evidence"
    target.write_bytes(prior)

    def fail_after_partial_write(self, destination, *args, **kwargs):
        Path(destination).write_bytes(b"partial")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_after_partial_write)

    result = active_pipeline_module._write_parquet_if_available(
        pd.DataFrame([{"pair": "SOL-WLD"}]), target
    )

    assert result == "not_written:OSError"
    assert target.read_bytes() == prior
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_export_trade_gate_blocks_without_model_acceptance(tmp_path, monkeypatch):
    from quant_platform import active_pipeline

    monkeypatch.setattr(active_pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(active_pipeline, "ML_REPORTS", tmp_path / "reports" / "ml")
    monkeypatch.setattr(active_pipeline, "MODELS", tmp_path / "models" / "trade_gate")

    result = export_trade_gate_model(root=tmp_path)

    report = result.paths["export_report"].read_text(encoding="utf-8")
    assert "model_gated_backtest_not_accepted" in report


def test_dashboard_rows_include_reason_blocker_freshness_and_evidence(tmp_path):
    build_pair_universe(root=tmp_path)
    active = tmp_path / "reports" / "active"
    pd.DataFrame(
        [
            {
                "exact_mode": "OU ZScoreR",
                "history_match_status": "MATCHED",
                "acceptance_eligible": False,
                "paper_or_execution_eligible": False,
                "evidence_path": "fixture://wizard-mode-comparison",
            }
        ]
    ).to_csv(active / "wizard_mode_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "cost_case": "observed",
                "execution_risk_bps": 0.0,
                "scenario_status": "RESEARCH_ONLY",
                "acceptance_eligible": False,
                "paper_or_execution_eligible": False,
                "evidence_path": "fixture://wizard-cost-sensitivity",
            }
        ]
    ).to_csv(active / "wizard_exploratory_cost_sensitivity.csv", index=False)
    pd.DataFrame(
        [
            {
                "exact_mode": "Copula",
                "source_fresh": True,
                "candidate_source_kind": "wizard_discovery",
                "readiness_status": "RESEARCH_ONLY",
            }
        ]
    ).to_csv(active / "multi_venue_history_readiness.csv", index=False)
    pd.DataFrame(
        [
            {
                "history_status": "MISSING",
                "wizard_settings_status": "CAPTURE_REQUIRED",
                "research_execution_status": "RESEARCH_ONLY",
                "next_action": "capture_history",
            }
        ]
    ).to_csv(active / "binance_spot_pair_readiness.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "ETH-PYTH",
                "status": "collecting",
                "minimum_samples": 11,
                "model_required_samples": 12,
                "projected_captures_to_calibration": 1,
                "projected_calibration_at": "2026-08-11T00:00:00Z",
            }
        ]
    ).to_csv(active / "hyperliquid_evidence_cadence.csv", index=False)
    (active / "current_wizard_ou_optimal_overlay_manifest.json").write_text(
        json.dumps(
            {
                "scanner_overlay": "ou_optimal",
                "source_rows_accounted": 2,
                "ou_optimal_true_rows": 1,
                "ou_optimal_false_rows": 1,
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (active / "exhaustive_wizard_hyperliquid_run_manifest.json").write_text(
        json.dumps({"run_id": "fixture-run", "source_rows": 1, "pair_groups": 1}),
        encoding="utf-8",
    )
    (active / "exhaustive_wizard_api_refresh_manifest.json").write_text(
        json.dumps({"refresh_id": "fixture-refresh", "api_source_rows": 1}),
        encoding="utf-8",
    )
    (active / "current_wizard_hyperliquid_handoff_manifest.json").write_text(
        json.dumps({"handoff_id": "fixture-handoff", "pair_groups": 1}),
        encoding="utf-8",
    )
    result = build_command_dashboard(root=tmp_path, refresh_profile="monitor")

    assert all(path.is_relative_to(tmp_path) for path in result.paths.values())
    live = pd.read_csv(result.paths["live_signals"])
    wizard_discovery = pd.read_csv(result.paths["wizard_discovery"])
    wizard_shortlist = pd.read_csv(result.paths["wizard_discovery_shortlist"])
    wizard_copula = pd.read_csv(result.paths["wizard_copula_discovery"])
    capture_queue = pd.read_csv(result.paths["wizard_pair_detail_capture_queue"])
    replay_handoff = pd.read_csv(result.paths["wizard_replay_handoff"])
    mode_comparison = pd.read_csv(result.paths["wizard_mode_comparison"])
    exploratory_cost_sensitivity = pd.read_csv(result.paths["wizard_exploratory_cost_sensitivity"])
    multi_venue = pd.read_csv(result.paths["multi_venue_history_readiness"])
    binance_readiness = pd.read_csv(result.paths["binance_spot_pair_readiness"])
    wizard_control = pd.read_csv(result.paths["wizard_control_plane"])
    candidate_ranking = pd.read_csv(result.paths["candidate_ranking"])
    data_health = pd.read_csv(result.paths["data_health"])
    command_center = result.paths["command_center"].read_text(encoding="utf-8")

    assert result.paths["hyperliquid_auxiliary_4h_walkforward"].exists()
    assert result.paths["student_mode_training_manifest"].exists()
    assert result.paths["hyperliquid_testnet_margin"].exists()
    assert result.paths["hyperliquid_testnet_preflight"].exists()
    assert result.paths["hyperliquid_evidence_cadence"].exists()
    assert result.paths["exhaustive_hyperliquid_concentration"].exists()
    assert result.paths["exhaustive_hyperliquid_leverage_status"].exists()
    assert result.paths["exhaustive_hyperliquid_leverage_candidates"].exists()
    assert result.paths["exhaustive_hyperliquid_leverage_scenarios"].exists()
    assert result.paths["exhaustive_hyperliquid_learning"].exists()
    assert result.paths["wizard_sweep_settings_capture_queue"].exists()
    assert result.paths["wizard_api_contract"].exists()
    assert result.paths["wizard_config_lineage"].exists()
    assert result.paths["current_wizard_ou_optimal_overlay_ledger"].exists()
    assert result.paths["current_wizard_ou_optimal_overlay_coverage"].exists()
    assert result.paths["current_wizard_ou_optimal_overlay_validation"].exists()
    assert "Wizard pair-page modes / scanner overlays: 7 / ou_optimal" in command_center
    ou_accounting = re.search(
        r"OU Optimal source rows accounted / true / false: (\d+) / (\d+) / (\d+)",
        command_center,
    )
    assert ou_accounting is not None
    total_rows, true_rows, false_rows = map(int, ou_accounting.groups())
    assert total_rows > 0
    assert total_rows == true_rows + false_rows
    assert "projected L2 captures remaining after rolling expiry" in command_center
    assert "family-wide statistical selection passes" in command_center
    assert "latest frozen validation" in command_center
    assert "dated research learning records" in command_center
    assert "training-eligible experiment summaries" in command_center

    assert {"reason", "blocker", "feature_timestamp", "evidence_path"}.issubset(live.columns)
    assert {"check", "status", "blocker", "evidence_path"}.issubset(wizard_control.columns)
    assert {"wizard_ranking_ready", "wizard_ranking_blocker", "wizard_rank"}.issubset(
        candidate_ranking.columns
    )
    assert "wizard_ranking_authority" in set(data_health["area"])
    assert {"source_fresh", "discovery_screen_status", "research_quality_status", "next_step"}.issubset(wizard_discovery.columns)
    assert {"exact_mode", "sharpe", "returns_total_pct", "research_quality_status"}.issubset(wizard_shortlist.columns)
    assert {"copula", "u1_given_u2", "u2_given_u1", "entry_signal_status"}.issubset(wizard_copula.columns)
    assert {"priority", "required_settings", "capture_status", "next_step"}.issubset(capture_queue.columns)
    assert {
        "settings_status",
        "venue_history_status",
        "exploratory_replay_status",
        "replay_status",
        "acceptance_authority",
    }.issubset(replay_handoff.columns)
    assert {
        "cost_case",
        "execution_risk_bps",
        "scenario_status",
        "acceptance_eligible",
        "paper_or_execution_eligible",
        "evidence_path",
    }.issubset(exploratory_cost_sensitivity.columns)
    assert {"exact_mode", "source_fresh", "candidate_source_kind", "readiness_status"}.issubset(multi_venue.columns)
    assert {"history_status", "wizard_settings_status", "research_execution_status", "next_action"}.issubset(binance_readiness.columns)
    assert {"exact_mode", "history_match_status", "acceptance_eligible", "paper_or_execution_eligible", "evidence_path"}.issubset(mode_comparison.columns)
    assert "## Wizard Discovery" in command_center
    assert "## Exhaustive Wizard To Hyperliquid" in command_center
    assert "rolling sample floor" in command_center
    assert " of 12" in command_center
    assert "## Copula Research" in command_center
    assert "## Exact-Mode Replay Handoff" in command_center
    assert "## Exploratory Cost Sensitivity" in command_center
    assert "## Cross-Mode Research Comparison" in command_center
    if "deep refresh blocked by storage" in command_center:
        assert "deep profile rebuilds evidence" not in command_center
    assert "## All-Venue Research Lanes" in command_center
    assert "Copula conditional values are a hypothesis signal, not a trade direction" in command_center
    assert "public spot history is research evidence only" in command_center


def test_paper_shortlist_and_validation_include_rl_winner_context(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    reports_rl = tmp_path / "reports" / "rl"
    reports_rl.mkdir(parents=True)
    reports_ml = tmp_path / "reports" / "ml"
    reports_ml.mkdir(parents=True)
    reports_root = tmp_path / "reports"
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "best_execution_venue": "dydx",
                "combined_score": 92.0,
                "acceptance_score": 81.0,
                "funding_drag_bps": 1.5,
                "available_timeframes": "1d",
                "decision_bucket": "PROMOTE",
                "decision_reason": "ready",
                "evidence_path": "reports/example.csv",
            },
            {
                "pair": "SOL-USD-TRX-USD",
                "best_execution_venue": "dydx",
                "combined_score": 88.0,
                "acceptance_score": 79.0,
                "funding_drag_bps": 2.0,
                "available_timeframes": "1d",
                "decision_bucket": "PROMOTE",
                "decision_reason": "ready",
                "evidence_path": "reports/example2.csv",
            },
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy_name": "learning_policy_005",
                "winner": 1,
                "profit_factor": 6.0,
                "stop_loss_pct": 0.07,
                "take_profit_pct": 0.14,
                "session_loss_cap_pct": 0.10,
                "top_pairs_entered": "BTC-USD-ETH-USD:15;SOL-USD-TRX-USD:7",
            }
        ]
    ).to_csv(reports_rl / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame([{"accepted": True, "blocker": ""}]).to_csv(reports_ml / "model_gated_acceptance.csv", index=False)
    pd.DataFrame([{"ready": True}, {"ready": True}]).to_csv(reports_root / "paper_execution_preflight.csv", index=False)

    shortlist = paper_candidate_shortlist_rows(root=tmp_path)
    focused = focused_paper_validation_rows(root=tmp_path)

    assert {"rl_policy_name", "rl_pair_focus_count", "rl_stop_loss_pct", "rl_take_profit_pct", "rl_session_loss_cap_pct"}.issubset(shortlist.columns)
    assert shortlist.iloc[0]["pair"] == "BTC-USD-ETH-USD"
    assert int(shortlist.iloc[0]["rl_pair_focus_count"]) == 15
    assert "rl_focus" in str(shortlist.iloc[0]["shortlist_reason"])
    assert {"rl_policy_winner", "rl_pair_focus_count", "rl_stop_loss_pct"}.issubset(focused.columns)
    assert focused.iloc[0]["validation_status"] == "hold_for_pair_specific_model_support"
    assert "pair-specific model support" in str(focused.iloc[0]["next_action"])


def test_paper_shortlist_prefers_quality_eligible_pairs(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True)
    (reports_root / "rl").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-TRX-USD",
                "best_execution_venue": "dydx",
                "combined_score": 99.0,
                "acceptance_score": 80.0,
                "funding_drag_bps": 1.0,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_only",
                "evidence_path": "reports/high_score.csv",
            },
            {
                "pair": "BTC-USD-ETH-USD",
                "best_execution_venue": "dydx",
                "combined_score": 95.0,
                "acceptance_score": 79.0,
                "funding_drag_bps": 1.2,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_and_quality",
                "evidence_path": "reports/quality.csv",
            },
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-TRX-USD", "research_execution_usable": False, "execution_usable": False},
            {"pair": "BTC-USD-ETH-USD", "research_execution_usable": True, "execution_usable": True},
        ]
    ).to_csv(reports_root / "pair_detail_quality_report.csv", index=False)

    shortlist = paper_candidate_shortlist_rows(root=tmp_path)

    assert shortlist["pair"].tolist() == ["BTC-USD-ETH-USD"]


def test_paper_shortlist_execution_filter_can_be_disabled(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "best_execution_venue": "injective",
                "combined_score": 99.0,
                "acceptance_score": 80.0,
                "funding_drag_bps": 1.0,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_and_quality",
                "evidence_path": "reports/candidate.csv",
            }
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)

    # Without a compatibility table row, this pair should be marked as incompatible and filtered out
    shortlist_filtered = paper_candidate_shortlist_rows(root=tmp_path)
    # With the optional gate disabled, we should still get the raw candidate row for diagnostics.
    shortlist_unfiltered = paper_candidate_shortlist_rows(root=tmp_path, require_execution_compatible=False)

    assert shortlist_filtered.empty is True
    assert not shortlist_unfiltered.empty
    assert shortlist_unfiltered.iloc[0]["pair"] == "BTC-USD-ETH-USD"
    assert bool(shortlist_unfiltered.iloc[0]["execution_compatible"]) is False


def test_paper_shortlist_accepts_confirmed_hyperliquid_pair_route(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    active = tmp_path / "reports" / "active"
    data_processed.mkdir(parents=True)
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "best_execution_venue": "hyperliquid",
                "execution_venue_ready": True,
                "combined_score": 99.0,
                "acceptance_score": 80.0,
                "funding_drag_bps": 1.0,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "route_valid",
                "evidence_path": "reports/hyperliquid_candidate.csv",
            }
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-ETH-USD",
                "testnet_perp_x": "BTC#3",
                "testnet_perp_y": "ETH#4",
                "mirrorable_for_paper": True,
                "mirror_blocker": "",
            }
        ]
    ).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)

    shortlist = paper_candidate_shortlist_rows(root=tmp_path)

    assert shortlist["pair"].tolist() == ["BTC-USD-ETH-USD"]
    assert bool(shortlist.iloc[0]["execution_compatible"]) is True
    assert str(shortlist.iloc[0]["execution_blocker"]) == ""


def test_filter_shortlist_to_visible_dydx_markets_keeps_pairs_when_market_status_unknown(tmp_path, monkeypatch):
    class _Adapter:
        def __init__(self, payloads):
            self.payloads = payloads

        def market_data(self, market: str):
            if market == "ETH-USD":
                raise RuntimeError("indexer timeout")
            payload = self.payloads.get(market)
            if payload is None:
                return {"payload": {"markets": {}}}
            return payload

    def _adapter_factory(_config):
        return _Adapter(
            {
                "BNB-USD": {"payload": {"markets": {"BNB-USD": {"status": "ACTIVE"}}}},
            }
        )

    monkeypatch.setattr("quant_platform.active_pipeline.build_dydx_indexer_adapter", _adapter_factory)

    shortlist = pd.DataFrame(
        [
            {"pair": "BNB-USD-ETH-USD", "best_execution_venue": "dydx"},
            {"pair": "ETH-USD-TRX-USD", "best_execution_venue": "binance"},
        ]
    )

    filtered = _filter_shortlist_to_visible_dydx_markets(shortlist)

    assert len(filtered) == 2
    assert set(filtered["pair"]) == {"BNB-USD-ETH-USD", "ETH-USD-TRX-USD"}


def test_filter_shortlist_to_visible_dydx_markets_rejects_inactive_markets(tmp_path, monkeypatch):
    class _Adapter:
        def market_data(self, market: str):
            rows = {
                "BNB-USD": {"payload": {"markets": {"BNB-USD": {"status": "ACTIVE"}}}},
                "ETH-USD": {"payload": {"markets": {"ETH-USD": {"status": "INACTIVE"}}}},
            }
            return rows.get(market, {"payload": {"markets": {}}})

    monkeypatch.setattr("quant_platform.active_pipeline.build_dydx_indexer_adapter", lambda _config: _Adapter())

    shortlist = pd.DataFrame([{"pair": "BNB-USD-ETH-USD", "best_execution_venue": "dydx"}])

    filtered = _filter_shortlist_to_visible_dydx_markets(shortlist)

    assert filtered.empty


def test_paper_shortlist_prefers_strong_pair_model_support_over_higher_rl_focus(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True)
    (reports_root / "rl").mkdir(parents=True)
    (reports_root / "brain").mkdir(parents=True)
    (reports_root / "ml").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "best_execution_venue": "dydx",
                "combined_score": 99.0,
                "acceptance_score": 80.0,
                "funding_drag_bps": 1.0,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_and_quality",
                "evidence_path": "reports/btc.csv",
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "best_execution_venue": "dydx",
                "combined_score": 103.0,
                "acceptance_score": 75.0,
                "funding_drag_bps": 1.2,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_and_quality",
                "evidence_path": "reports/sol.csv",
            },
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy_name": "learning_policy_005",
                "winner": 1,
                "profit_factor": 6.0,
                "stop_loss_pct": 0.07,
                "take_profit_pct": 0.14,
                "session_loss_cap_pct": 0.10,
                "top_pairs_entered": "BTC-USD-HYPE-USD:8;SOL-USD-HYPE-USD:14",
            }
        ]
    ).to_csv(reports_root / "rl" / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "research_execution_usable": True, "execution_usable": True},
            {"pair": "SOL-USD-HYPE-USD", "research_execution_usable": True, "execution_usable": True},
        ]
    ).to_csv(reports_root / "pair_detail_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "pair_model_support_status": "strong_model_support",
                "pair_model_taken_trades": 22,
                "pair_model_profit_factor": 4.39,
                "pair_model_mean_return": 0.0145,
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "pair_model_support_status": "no_model_support",
                "pair_model_taken_trades": 0,
                "pair_model_profit_factor": 0.0,
                "pair_model_mean_return": 0.0,
            },
        ]
    ).to_csv(reports_root / "brain" / "native_focus_repair_report.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "candidate_id": "native:BTC", "setup_identity": "native:BTC", "setup_role": "primary", "paper_credible": True},
            {"pair": "SOL-USD-HYPE-USD", "candidate_id": "native:SOL", "setup_identity": "native:SOL", "setup_role": "primary", "paper_credible": True},
        ]
    ).to_csv(reports_root / "brain" / "promotion_ladder.csv", index=False)

    shortlist = paper_candidate_shortlist_rows(root=tmp_path)

    assert shortlist["pair"].tolist() == ["BTC-USD-HYPE-USD"]
    assert shortlist.iloc[0]["pair_model_support_status"] == "strong_model_support"
    assert "strong_pair_model_support" in str(shortlist.iloc[0]["shortlist_reason"])


def test_paper_shortlist_can_promote_watch_candidate_when_route_valid_and_strong_model_supported(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True)
    (reports_root / "rl").mkdir(parents=True)
    (reports_root / "brain").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-LDO-USD",
                "best_execution_venue": "dydx",
                "combined_score": 44.6,
                "acceptance_score": 70.0,
                "funding_drag_bps": 1.0,
                "available_timeframes": "5m",
                "decision_bucket": "WATCH",
                "decision_reason": "route_valid_watch",
                "evidence_path": "reports/ldo.csv",
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "best_execution_venue": "dydx",
                "combined_score": 102.9,
                "acceptance_score": 74.9,
                "funding_drag_bps": 1.2,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_and_quality",
                "evidence_path": "reports/sol.csv",
            },
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-LDO-USD", "research_execution_usable": True, "execution_usable": True},
            {"pair": "SOL-USD-HYPE-USD", "research_execution_usable": True, "execution_usable": True},
        ]
    ).to_csv(reports_root / "pair_detail_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-LDO-USD",
                "pair_model_support_status": "strong_model_support",
                "pair_model_taken_trades": 21,
                "pair_model_profit_factor": 2.56,
                "pair_model_mean_return": 0.036,
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "pair_model_support_status": "no_model_support",
                "pair_model_taken_trades": 0,
                "pair_model_profit_factor": 0.0,
                "pair_model_mean_return": 0.0,
            },
        ]
    ).to_csv(reports_root / "brain" / "native_focus_repair_report.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-LDO-USD", "candidate_id": "native:LDO", "setup_identity": "native:LDO", "setup_role": "primary", "paper_credible": True},
            {"pair": "SOL-USD-HYPE-USD", "candidate_id": "native:SOL", "setup_identity": "native:SOL", "setup_role": "primary", "paper_credible": True},
        ]
    ).to_csv(reports_root / "brain" / "promotion_ladder.csv", index=False)

    shortlist = paper_candidate_shortlist_rows(root=tmp_path)

    assert shortlist.iloc[0]["pair"] == "BTC-USD-LDO-USD"
    assert shortlist.iloc[0]["pair_model_support_status"] == "strong_model_support"


def test_paper_shortlist_does_not_let_rl_focus_override_explicitly_unsupported_pairs(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True)
    (reports_root / "rl").mkdir(parents=True)
    (reports_root / "brain").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "best_execution_venue": "dydx",
                "combined_score": 113.7,
                "acceptance_score": 75.0,
                "funding_drag_bps": 1.0,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_and_quality",
                "evidence_path": "reports/btc.csv",
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "best_execution_venue": "dydx",
                "combined_score": 102.9,
                "acceptance_score": 74.9,
                "funding_drag_bps": 1.2,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_and_quality",
                "evidence_path": "reports/sol.csv",
            },
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy_name": "learning_policy_005",
                "winner": 1,
                "profit_factor": 6.0,
                "stop_loss_pct": 0.07,
                "take_profit_pct": 0.14,
                "session_loss_cap_pct": 0.10,
                "top_pairs_entered": "SOL-USD-HYPE-USD:14",
            }
        ]
    ).to_csv(reports_root / "rl" / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "research_execution_usable": True, "execution_usable": True},
            {"pair": "SOL-USD-HYPE-USD", "research_execution_usable": True, "execution_usable": True},
        ]
    ).to_csv(reports_root / "pair_detail_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "pair_model_support_status": "no_model_support",
                "pair_model_taken_trades": 0,
                "pair_model_profit_factor": 0.0,
                "pair_model_mean_return": 0.0,
            },
            {
                "pair": "SOL-USD-HYPE-USD",
                "pair_model_support_status": "no_model_support",
                "pair_model_taken_trades": 0,
                "pair_model_profit_factor": 0.0,
                "pair_model_mean_return": 0.0,
            },
        ]
    ).to_csv(reports_root / "brain" / "native_focus_repair_report.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "candidate_id": "native:BTC", "setup_identity": "native:BTC", "setup_role": "primary", "paper_credible": True},
            {"pair": "SOL-USD-HYPE-USD", "candidate_id": "native:SOL", "setup_identity": "native:SOL", "setup_role": "primary", "paper_credible": True},
        ]
    ).to_csv(reports_root / "brain" / "promotion_ladder.csv", index=False)

    shortlist = paper_candidate_shortlist_rows(root=tmp_path)

    assert shortlist.iloc[0]["pair"] == "BTC-USD-HYPE-USD"
    assert shortlist.iloc[0]["pair_model_support_status"] == "no_model_support"


def test_focused_paper_validation_blocks_when_pair_model_support_is_not_strong(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True)
    (reports_root / "rl").mkdir(parents=True)
    (reports_root / "brain").mkdir(parents=True)
    (reports_root / "ml").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "SOL-USD-HYPE-USD",
                "best_execution_venue": "dydx",
                "combined_score": 103.0,
                "acceptance_score": 75.0,
                "funding_drag_bps": 1.2,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "score_and_quality",
                "evidence_path": "reports/sol.csv",
            },
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy_name": "learning_policy_005",
                "winner": 1,
                "profit_factor": 6.0,
                "stop_loss_pct": 0.07,
                "take_profit_pct": 0.14,
                "session_loss_cap_pct": 0.10,
                "top_pairs_entered": "SOL-USD-HYPE-USD:14",
            }
        ]
    ).to_csv(reports_root / "rl" / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame(
        [{"pair": "SOL-USD-HYPE-USD", "research_execution_usable": True, "execution_usable": True}]
    ).to_csv(reports_root / "pair_detail_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "SOL-USD-HYPE-USD",
                "pair_model_support_status": "no_model_support",
                "pair_model_taken_trades": 0,
                "pair_model_profit_factor": 0.0,
                "pair_model_mean_return": 0.0,
            },
        ]
    ).to_csv(reports_root / "brain" / "native_focus_repair_report.csv", index=False)
    pd.DataFrame(
        [{"pair": "SOL-USD-HYPE-USD", "candidate_id": "native:SOL", "setup_identity": "native:SOL", "setup_role": "primary", "paper_credible": True}]
    ).to_csv(reports_root / "brain" / "promotion_ladder.csv", index=False)
    pd.DataFrame([{"accepted": True, "blocker": ""}]).to_csv(reports_root / "ml" / "model_gated_acceptance.csv", index=False)
    pd.DataFrame([{"ready": True}, {"ready": True}]).to_csv(reports_root / "paper_execution_preflight.csv", index=False)

    focused = focused_paper_validation_rows(root=tmp_path)

    assert focused.iloc[0]["pair_model_support_status"] == "no_model_support"
    assert focused.iloc[0]["validation_status"] == "hold_for_pair_specific_model_support"
    assert "pair-specific model support" in str(focused.iloc[0]["next_action"])


def test_model_gate_pair_support_report_ranks_strong_anchor_candidate(tmp_path):
    reports_ml = tmp_path / "reports" / "ml"
    reports_ml.mkdir(parents=True)
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "combined_score": 113.7},
            {"pair": "SOL-USD-HYPE-USD", "combined_score": 102.9},
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {"accepted": False, "failing_checks": "filtered_profit_factor_min;filtered_sharpe_positive"},
        ]
    ).to_csv(reports_ml / "model_gated_acceptance.csv", index=False)
    pd.DataFrame(
        [
            *[
                {"pair": "BTC-USD-HYPE-USD", "shadow_take": True, "realized_return": value}
                for value in [0.05, 0.04, 0.03, 0.02, 0.01, 0.02, 0.03, 0.01, 0.02, -0.01]
            ],
            *[
                {"pair": "BTC-USD-HYPE-USD", "shadow_take": False, "realized_return": value}
                for value in [-0.02, 0.01]
            ],
            *[
                {"pair": "SOL-USD-HYPE-USD", "shadow_take": False, "realized_return": value}
                for value in [-0.04, 0.02, -0.01]
            ],
        ]
    ).to_csv(reports_ml / "model_walkforward_predictions.csv", index=False)

    result = model_gate_pair_support_report(root=tmp_path)
    report = pd.read_csv(result.paths["model_gate_pair_support_report"])

    assert report.iloc[0]["pair"] == "BTC-USD-HYPE-USD"
    assert report.iloc[0]["support_status"] == "strong_model_support"
    assert "preserve strong pair-specific model support" in str(report.iloc[0]["recommended_repair_action"])


def test_model_gate_pair_support_report_prefers_current_shortlist_anchor_over_global_stronger_pair(tmp_path):
    reports_ml = tmp_path / "reports" / "ml"
    reports_ml.mkdir(parents=True)
    reports_rl = tmp_path / "reports" / "rl"
    reports_rl.mkdir(parents=True)
    reports_brain = tmp_path / "reports" / "brain"
    reports_brain.mkdir(parents=True)
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    reports_root = tmp_path / "reports"
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "best_execution_venue": "dydx",
                "combined_score": 113.7,
                "acceptance_score": 74.9,
                "funding_drag_bps": 0.01,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "ready",
                "evidence_path": "reports/btc.csv",
            },
            {
                "pair": "BTCUSDT-SOLUSDT",
                "best_execution_venue": "dydx",
                "combined_score": 70.0,
                "acceptance_score": 60.0,
                "funding_drag_bps": 0.01,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "ready",
                "evidence_path": "reports/other.csv",
            },
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy_name": "learning_policy_005",
                "winner": 1,
                "profit_factor": 6.0,
                "stop_loss_pct": 0.07,
                "take_profit_pct": 0.14,
                "session_loss_cap_pct": 0.10,
                "top_pairs_entered": "",
            }
        ]
    ).to_csv(reports_rl / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame(
        [
            {"pair": "BTC-USD-HYPE-USD", "research_execution_usable": True, "execution_usable": True},
            {"pair": "BTCUSDT-SOLUSDT", "research_execution_usable": True, "execution_usable": True},
        ]
    ).to_csv(reports_root / "pair_detail_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "pair_model_support_status": "strong_model_support",
                "pair_model_taken_trades": 22,
                "pair_model_profit_factor": 4.39,
                "pair_model_mean_return": 0.0145,
            },
        ]
    ).to_csv(reports_brain / "native_focus_repair_report.csv", index=False)
    pd.DataFrame(
        [{"pair": "BTC-USD-HYPE-USD", "candidate_id": "native:BTC", "setup_identity": "native:BTC", "setup_role": "primary", "paper_credible": True}]
    ).to_csv(reports_brain / "promotion_ladder.csv", index=False)
    pd.DataFrame(
        [{"accepted": False, "failing_checks": "filtered_profit_factor_min", "blocker": "model_gated_backtest_not_accepted"}]
    ).to_csv(reports_ml / "model_gated_acceptance.csv", index=False)
    pd.DataFrame(
        [
            *[
                {"pair": "BTCUSDT-SOLUSDT", "shadow_take": True, "realized_return": value}
                for value in [0.05] * 25
            ],
            *[
                {"pair": "BTC-USD-HYPE-USD", "shadow_take": True, "realized_return": value}
                for value in [0.03] * 10
            ],
        ]
    ).to_csv(reports_ml / "model_walkforward_predictions.csv", index=False)

    report = pd.read_csv(model_gate_pair_support_report(root=tmp_path).paths["model_gate_pair_support_report"])

    assert report.iloc[0]["pair"] == "BTC-USD-HYPE-USD"


def test_focused_paper_validation_uses_model_gate_anchor_action_for_strong_support_candidate(tmp_path):
    data_processed = tmp_path / "data" / "processed"
    data_processed.mkdir(parents=True)
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True)
    (reports_root / "rl").mkdir(parents=True)
    (reports_root / "brain").mkdir(parents=True)
    (reports_root / "ml").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "best_execution_venue": "dydx",
                "combined_score": 113.7,
                "acceptance_score": 74.9,
                "funding_drag_bps": 0.01,
                "available_timeframes": "5m",
                "decision_bucket": "PROMOTE",
                "decision_reason": "ready",
                "evidence_path": "reports/btc.csv",
            },
        ]
    ).to_csv(data_processed / "pair_universe.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy_name": "learning_policy_005",
                "winner": 1,
                "profit_factor": 6.0,
                "stop_loss_pct": 0.07,
                "take_profit_pct": 0.14,
                "session_loss_cap_pct": 0.10,
                "top_pairs_entered": "",
            }
        ]
    ).to_csv(reports_root / "rl" / "rl_learning_cycle_summary.csv", index=False)
    pd.DataFrame(
        [{"pair": "BTC-USD-HYPE-USD", "research_execution_usable": True, "execution_usable": True}]
    ).to_csv(reports_root / "pair_detail_quality_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD-HYPE-USD",
                "pair_model_support_status": "strong_model_support",
                "pair_model_taken_trades": 22,
                "pair_model_profit_factor": 4.39,
                "pair_model_mean_return": 0.0145,
            },
        ]
    ).to_csv(reports_root / "brain" / "native_focus_repair_report.csv", index=False)
    pd.DataFrame(
        [{"pair": "BTC-USD-HYPE-USD", "candidate_id": "native:BTC", "setup_identity": "native:BTC", "setup_role": "primary", "paper_credible": True}]
    ).to_csv(reports_root / "brain" / "promotion_ladder.csv", index=False)
    pd.DataFrame(
        [{"accepted": False, "failing_checks": "filtered_profit_factor_min;filtered_sharpe_positive", "blocker": "model_acceptance_gates_not_met"}]
    ).to_csv(reports_root / "ml" / "model_gated_acceptance.csv", index=False)
    pd.DataFrame(
        [{"pair": "BTC-USD-HYPE-USD", "shadow_take": True, "realized_return": value} for value in [0.03] * 10]
    ).to_csv(reports_root / "ml" / "model_walkforward_predictions.csv", index=False)
    pd.DataFrame([{"ready": True}, {"ready": True}]).to_csv(reports_root / "paper_execution_preflight.csv", index=False)
    model_gate_pair_support_report(root=tmp_path)

    focused = focused_paper_validation_rows(root=tmp_path)

    assert focused.iloc[0]["pair_model_support_status"] == "strong_model_support"
    assert focused.iloc[0]["validation_status"] == "hold_for_model_or_gate"
    assert "preserve strong pair-specific model support" in str(focused.iloc[0]["next_action"])


def test_archive_dry_run_writes_manifest_and_moves_nothing(tmp_path):
    (tmp_path / "work").mkdir(parents=True)
    (tmp_path / "work" / "scratch.log").write_text("scratch\n", encoding="utf-8")
    build_artifact_index(root=tmp_path)
    result = archive_from_index(dry_run=True, root=tmp_path)

    assert all(path.is_relative_to(tmp_path) for path in result.paths.values())
    frame = pd.read_csv(result.paths["archive_manifest"])

    assert "planned_action" in frame.columns
    assert set(frame["planned_action"].dropna().unique()).issubset({"would_archive"})
