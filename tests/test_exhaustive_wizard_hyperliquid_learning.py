from datetime import datetime, timezone
import json

import pandas as pd
import pytest

from quant_platform.orchestration.exhaustive_wizard_hyperliquid_learning import (
    build_exhaustive_wizard_hyperliquid_learning_ledger,
)


STAGE_IDENTITIES = {
    "cost_evidence": "cost-1",
    "observed_cost_replay": "observed-1",
    "walkforward": "walk-1",
    "regime_attribution": "regime-1",
    "robustness": "robust-1",
    "concentration": "concentration-1",
    "leverage_surface": "leverage-1",
}


def _write_inputs(root, *, omit_leverage_experiment=False):
    active = root / "reports" / "active"
    active.mkdir(parents=True)
    experiment_ids = ["experiment-1", "experiment-2"]
    base_rows = [
        {
            "exhaustive_run_id": "run-1",
            "experiment_id": experiment_id,
            "pair_group_id": f"pair-{index}",
            "pair": "BTC-USD-ETH-USD",
            "wizard_exchange": "dydx",
            "timeframe": "daily",
            "exact_mode": "Static (Spread)",
            "orientation": "original" if index == 1 else "reverse",
            "asset_x": "BTC",
            "asset_y": "ETH",
            "hyperliquid_pair_ready": True,
            "hyperliquid_pair_max_leverage": 10,
            "hyperliquid_mapping_blocker": "",
            "source_row_ids": f"source-{index}",
            "evidence_path": f"raw/source-{index}.json",
        }
        for index, experiment_id in enumerate(experiment_ids, start=1)
    ]
    pd.DataFrame(base_rows).to_csv(
        active / "exhaustive_wizard_experiment_matrix.csv", index=False
    )

    def rows(extra):
        return [
            {"experiment_id": experiment_id, **extra(index)}
            for index, experiment_id in enumerate(experiment_ids, start=1)
        ]

    pd.DataFrame(
        rows(
            lambda _: {
                "exhaustive_run_id": "run-1",
                "cost_evidence_id": "cost-1",
                "preflight_status": "READY_FOR_HISTORY",
                "cost_evidence_status": "READY_FOR_COST_CALIBRATED_REPLAY",
                "provisional_cost_research_ready": True,
                "cost_acceptance_ready": True,
                "cost_replay_status": "READY_FOR_COST_CALIBRATED_REPLAY",
                "cost_replay_blocker": "",
            }
        )
    ).to_csv(
        active / "exhaustive_wizard_hyperliquid_experiment_cost_readiness.csv",
        index=False,
    )
    pd.DataFrame(
        rows(
            lambda index: {
                "exhaustive_run_id": "run-1",
                "observed_cost_replay_id": "observed-1",
                "scanner_cutoff_at": "2026-08-08T08:00:00+00:00",
                "hyperliquid_interval": "1d",
                "canonical_replay_leverage": 1.0,
                "observed_funding_rows": 800,
                "funding_both_coverage": 1.0,
                "taker_fee_bps": 4.5,
                "slippage_x_p95_bps": 1.0,
                "slippage_y_p95_bps": 1.5,
                "execution_risk_bps": 2.0,
                "replay_status": "OBSERVED_COST_RESEARCH_REPLAY_COMPLETE",
                "replay_blocker": "",
                "mode_fidelity_status": "local_formula_approximation",
                "mode_fidelity_reason": "research_only",
                "math_version": "math-v2.1-y-on-x",
                "trades": 20,
                "profit_factor": 1.2,
                "expectancy": 0.01,
                "sharpe": 1.1,
                "max_drawdown": 0.1,
                "total_return": 0.2,
                "research_rank_eligible": True,
                "research_rank_blocker": "",
            }
        )
    ).to_csv(
        active / "exhaustive_wizard_hyperliquid_observed_cost_replay.csv",
        index=False,
    )
    pd.DataFrame(
        rows(
            lambda index: {
                "walkforward_id": "walk-1",
                "walkforward_status": (
                    "PASS_RESEARCH_WALK_FORWARD"
                    if index == 1
                    else "FAIL_RESEARCH_WALK_FORWARD"
                ),
                "walkforward_blocker": "" if index == 1 else "weak_fold",
                "folds_complete": 5,
                "positive_folds": 4 if index == 1 else 1,
                "aggregate_trades": 20,
                "aggregate_profit_factor": 1.2,
                "aggregate_expectancy": 0.01,
                "aggregate_sharpe": 1.1,
                "aggregate_max_drawdown": 0.1,
                "aggregate_total_return": 0.2,
                "fold_return_raw_pvalue": 0.02,
                "bh_qvalue": 0.20,
                "parameter_stability_status": "PASS",
                "deflated_sharpe_status": "PASS",
                "statistical_selection_status": "BLOCKED",
                "statistical_selection_blocker": "false_discovery_gate_failed",
            }
        )
    ).to_csv(
        active / "exhaustive_wizard_hyperliquid_walkforward_status.csv", index=False
    )
    pd.DataFrame(
        rows(
            lambda _: {
                "regime_attribution_id": "regime-1",
                "regime_status": "REGIME_ATTRIBUTION_COMPLETE",
                "regime_blocker": "",
                "regimes_observed": 3,
                "regimes_with_minimum_trades": 2,
                "positive_expectancy_regimes": 2,
                "regime_profit_concentration": 0.7,
                "worst_regime_expectancy": -0.01,
                "crisis_trades": 2,
                "crisis_expectancy": -0.02,
                "regime_stability_status": "PASS_RESEARCH_REGIME_STABILITY",
                "regime_stability_blocker": "",
            }
        )
    ).to_csv(active / "exhaustive_wizard_hyperliquid_regime_status.csv", index=False)
    pd.DataFrame(
        rows(
            lambda _: {
                "robustness_id": "robust-1",
                "robustness_status": "ROBUSTNESS_COMPLETE",
                "robustness_blocker": "",
                "scenarios_complete": 11,
                "parameter_positive_ratio": 0.8,
                "parameter_pass_ratio": 0.7,
                "worst_parameter_drawdown": 0.15,
                "research_robustness_status": "PASS_RESEARCH_ROBUSTNESS",
                "research_robustness_blocker": "",
                "promotion_readiness": "BLOCKED",
                "promotion_blocker": "statistical_selection_not_passed",
            }
        )
    ).to_csv(
        active / "exhaustive_wizard_hyperliquid_robustness_status.csv", index=False
    )
    pd.DataFrame(
        rows(
            lambda _: {
                "concentration_id": "concentration-1",
                "concentration_status": "NOT_SELECTED_PRIOR_RESEARCH_GATES",
                "concentration_blocker": "statistical_selection_not_passed",
                "concentration_gate_pass": False,
                "ready_for_leverage_gate": False,
            }
        )
    ).to_csv(
        active / "exhaustive_wizard_hyperliquid_concentration_status.csv", index=False
    )
    leverage_rows = rows(
        lambda _: {
            "leverage_surface_id": "leverage-1",
            "leverage_surface_status": "NOT_SELECTED_PRIOR_CONCENTRATION_GATE",
            "leverage_surface_blocker": "statistical_selection_not_passed",
            "ready_for_testnet_1x_lifecycle": False,
            "acceptance_status": "BLOCKED",
            "acceptance_reason": "research_only",
            "acceptance_eligible": False,
        }
    )
    if omit_leverage_experiment:
        leverage_rows.pop()
    pd.DataFrame(leverage_rows).to_csv(
        active / "exhaustive_wizard_hyperliquid_leverage_status.csv", index=False
    )
    stages = []
    for stage, identity in STAGE_IDENTITIES.items():
        manifest = root / "snapshots" / stage / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps({"stage": stage, "stage_identity": identity}), encoding="utf-8"
        )
        stages.append(
            {
                "stage": stage,
                "stage_identity": identity,
                "manifest_path": str(manifest.relative_to(root)),
            }
        )
    return stages


def test_learning_ledger_accounts_every_experiment_without_creating_trade_labels(
    tmp_path,
):
    stages = _write_inputs(tmp_path)
    result = build_exhaustive_wizard_hyperliquid_learning_ledger(
        root=tmp_path,
        now=datetime(2026, 8, 8, 10, tzinfo=timezone.utc),
        validation_id="validation-1",
        validation_stages=stages,
    )

    ledger = pd.read_csv(result.paths["ledger"], keep_default_na=False)
    assert len(ledger) == 2
    assert ledger["experiment_id"].nunique() == 2
    assert not ledger["training_eligible"].astype(bool).any()
    assert not ledger["realized_outcome_available"].astype(bool).any()
    assert ledger["paper_label"].eq("").all()
    assert ledger["live_label"].eq("").all()
    assert not ledger["wizard_is_label_authority"].astype(bool).any()
    assert not ledger["live_trading_authorized"].astype(bool).any()
    assert set(ledger["research_outcome_label"]) == {
        "PRACTICAL_WALKFORWARD_PASS_NOT_PROMOTED",
        "RESEARCH_REPLAY_EVALUATED_NOT_PROMOTED",
    }
    assert result.summary["experiment_status_accounted"] is True
    assert result.summary["training_eligible_records"] == 0
    assert result.summary["paper_label_records"] == 0
    assert result.summary["live_label_records"] == 0
    assert result.paths["snapshot_manifest"].exists()
    assert (
        result.paths["snapshot_manifest"].parent
        / "inputs"
        / "leverage_surface_manifest.json"
    ).exists()
    assert not (tmp_path / "data" / "meta_learning" / "trades.jsonl").exists()
    records = [
        json.loads(line)
        for line in result.paths["dataset_jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert all(record["outcome_type"] == "backtest_research_outcome" for record in records)


def test_learning_ledger_fails_when_a_stage_silently_drops_an_experiment(tmp_path):
    stages = _write_inputs(tmp_path, omit_leverage_experiment=True)

    with pytest.raises(ValueError, match="leverage experiment accounting mismatch"):
        build_exhaustive_wizard_hyperliquid_learning_ledger(
            root=tmp_path,
            validation_id="validation-1",
            validation_stages=stages,
        )
