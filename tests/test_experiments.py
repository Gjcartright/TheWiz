from dataclasses import replace

import numpy as np
import pandas as pd

from quant_platform.backtest import BacktestResult, CostModel
from quant_platform.experiments import (
    AcceptanceGate,
    CostBucket,
    ExperimentConfig,
    ExperimentHarness,
    PairDataset,
    strategy_acceptance_report,
)
from quant_platform.strategies import STRATEGIES


def experiment_frame(n: int = 160) -> pd.DataFrame:
    x = np.linspace(0, 12 * np.pi, n)
    spread = np.sin(x)
    frame = pd.DataFrame({"spread": spread})
    frame["zscore"] = frame["spread"] * 2.5
    frame["conditional_probability_distortion"] = np.tanh(frame["zscore"] / 3.0)
    frame["regime"] = np.where(frame.index < n / 2, "range", "crisis")
    return frame


def two_leg_experiment_frame(n: int = 160) -> pd.DataFrame:
    frame = experiment_frame(n)
    frame["price_x"] = 100 + np.linspace(0, 5, n) + frame["spread"]
    frame["price_y"] = 50 + np.linspace(0, 2, n) - frame["spread"]
    frame["hedge_ratio"] = 1.2
    frame["beta"] = 0.8
    frame["funding_x_bps"] = 2.0
    frame["funding_y_bps"] = 3.0
    return frame


def reconstructed_zscore_experiment_frame(n: int = 220) -> pd.DataFrame:
    frame = experiment_frame(n).copy()
    # emulate reconstructed z-score availability when raw zscore is missing
    frame["zscore"] = pd.NA
    frame["zscore_reconstructed"] = frame["spread"] * 2.5
    return frame


def test_harness_runs_executable_strategies_and_marks_missing_ones_skipped():
    config = ExperimentConfig(
        cost_buckets=(
            CostBucket(
                "base",
                CostModel(
                    taker_fee_bps=0, slippage_bps=0, execution_risk_bps=0, funding_bps_per_day=0
                ),
            ),
        ),
        gate=AcceptanceGate(min_profit_factor=0.1, min_sharpe=-99, max_drawdown=1.0, min_trades=1),
        min_rows=20,
    )
    harness = ExperimentHarness(strategies=STRATEGIES[:3], config=config)

    results = harness.run([PairDataset("ETH-BTC", experiment_frame())])

    assert {"ALL", "crisis", "range"} == set(results["regime"])
    assert set(results["status"]) == {"evaluated", "skipped"}
    assert (results["strategy_name"] == "Classic ZScore Mean Reversion").any()
    assert results["reason"].str.contains("missing_columns:ecm_strength", regex=False).any()
    assert "rank_score" in results
    assert "rank" in results
    evaluated = results[results["status"] == "evaluated"]
    assert set(evaluated["backtest_mode"]) == {"spread"}
    assert not evaluated["has_price_x"].any()
    assert not evaluated["has_price_y"].any()


def test_harness_marks_two_leg_backtest_mode_when_leg_prices_exist():
    config = ExperimentConfig(
        cost_buckets=(
            CostBucket(
                "base",
                CostModel(
                    taker_fee_bps=0, slippage_bps=0, execution_risk_bps=0, funding_bps_per_day=0
                ),
            ),
        ),
        gate=AcceptanceGate(min_profit_factor=0.1, min_sharpe=-99, max_drawdown=1.0, min_trades=1),
        min_rows=20,
    )
    harness = ExperimentHarness(strategies=(STRATEGIES[0],), config=config)

    results = harness.run([PairDataset("ETH-BTC", two_leg_experiment_frame())])
    evaluated = results[results["status"] == "evaluated"]

    assert set(evaluated["backtest_mode"]) == {"two_leg"}
    assert evaluated["has_price_x"].all()
    assert evaluated["has_price_y"].all()
    assert evaluated["has_hedge_ratio"].all()
    assert evaluated["has_beta"].all()
    assert evaluated["has_funding_x"].all()
    assert evaluated["has_funding_y"].all()


def test_harness_routes_incomplete_two_leg_inputs_to_spread_only_research():
    frame = two_leg_experiment_frame().drop(columns=["hedge_ratio"])
    config = ExperimentConfig(
        cost_buckets=(
            CostBucket(
                "base",
                CostModel(
                    taker_fee_bps=0,
                    slippage_bps=0,
                    execution_risk_bps=0,
                    funding_bps_per_day=0,
                ),
            ),
        ),
        gate=AcceptanceGate(
            min_profit_factor=0.1,
            min_sharpe=-99,
            max_drawdown=1.0,
            min_trades=1,
        ),
        min_rows=20,
    )
    harness = ExperimentHarness(strategies=(STRATEGIES[0],), config=config)

    results = harness.run([PairDataset("ETH-BTC", frame)])
    evaluated = results[results["status"] == "evaluated"]

    assert not evaluated.empty
    assert set(evaluated["backtest_mode"]) == {"spread"}
    assert not evaluated["has_hedge_ratio"].any()


def test_harness_accepts_reconstructed_zscore_when_provider_zscore_is_missing():
    config = ExperimentConfig(
        cost_buckets=(
            CostBucket(
                "base",
                CostModel(
                    taker_fee_bps=0, slippage_bps=0, execution_risk_bps=0, funding_bps_per_day=0
                ),
            ),
        ),
        gate=AcceptanceGate(min_profit_factor=0.1, min_sharpe=-99, max_drawdown=1.0, min_trades=1),
        min_rows=20,
    )
    harness = ExperimentHarness(strategies=(STRATEGIES[0],), config=config)

    results = harness.run([PairDataset("BTC-ETH", reconstructed_zscore_experiment_frame())])
    evaluated = results[(results["status"] == "evaluated") & (results["strategy_id"] == 1)]

    assert not evaluated.empty
    assert not evaluated["reason"].str.contains("missing_columns", regex=False).any()


def test_harness_writes_all_report_files(tmp_path):
    config = ExperimentConfig(
        cost_buckets=(CostBucket("base", CostModel()),),
        gate=AcceptanceGate(min_profit_factor=0.1, min_sharpe=-99, max_drawdown=1.0, min_trades=1),
        min_rows=20,
    )
    harness = ExperimentHarness(strategies=(STRATEGIES[0], STRATEGIES[1]), config=config)
    results = harness.run([PairDataset("SOL-ETH", experiment_frame())])

    paths = harness.write_reports(results, tmp_path)

    assert set(paths) == {
        "ablation",
        "all_results",
        "acceptance",
        "strategy_summary",
        "regime_summary",
        "regime_pair_strategy",
        "coverage",
    }
    assert all(path.exists() for path in paths.values())
    written = pd.read_csv(paths["all_results"])
    assert len(written) == len(results)
    assert "backtest_mode" in written.columns
    summary = pd.read_csv(paths["strategy_summary"])
    cost_columns = {
        "median_gross_return",
        "median_total_return",
        "median_cost_drag",
        "total_fees",
        "total_slippage",
        "total_funding",
        "total_execution_risk",
        "total_partial_fill_cost",
        "median_avg_gross_exposure",
    }
    assert {"two_leg_runs", "spread_runs", *cost_columns}.issubset(summary.columns)
    regime_summary = pd.read_csv(paths["regime_summary"])
    assert cost_columns.issubset(regime_summary.columns)


def test_acceptance_gate_rejects_under_sampled_results():
    config = ExperimentConfig(
        cost_buckets=(
            CostBucket(
                "base",
                CostModel(
                    taker_fee_bps=0, slippage_bps=0, execution_risk_bps=0, funding_bps_per_day=0
                ),
            ),
        ),
        gate=AcceptanceGate(
            min_profit_factor=0.1, min_sharpe=-99, max_drawdown=1.0, min_trades=10_000
        ),
        min_rows=20,
    )
    harness = ExperimentHarness(strategies=(STRATEGIES[0],), config=config)

    results = harness.run([PairDataset("BTC-ETH", experiment_frame())])

    evaluated = results[results["status"] == "evaluated"]
    assert not evaluated["eligible"].any()
    assert evaluated["reason"].str.contains("trades<10000", regex=False).all()


def accepted_strategy_rows() -> pd.DataFrame:
    rows = []
    for pair in ("ETH-BTC", "SOL-ETH"):
        for cost_bucket in ("base", "stress"):
            rows.append(
                {
                    "pair": pair,
                    "strategy_id": 1,
                    "strategy_name": "Classic ZScore Mean Reversion",
                    "family": "zscore",
                    "regime": "ALL",
                    "cost_bucket": cost_bucket,
                    "status": "evaluated",
                    "eligible": True,
                    "reason": "passed",
                    "trades": 130,
                    "profit_factor": 2.1,
                    "expectancy": 0.01,
                    "sharpe": 1.6,
                    "max_drawdown": 0.08,
                    "win_rate": 0.56,
                    "total_return": 0.12,
                    "open_trades": 0,
                    "sharpe_status": "valid",
                    "reconciliation_error": 0.0,
                    "expectancy_lower_95": 0.005,
                    "observations": 500,
                    "backtest_mode": "two_leg",
                    "has_price_x": True,
                    "has_price_y": True,
                    "has_hedge_ratio": True,
                    "has_beta": True,
                    "has_funding_x": True,
                    "has_funding_y": True,
                    "rank_score": 100,
                    "rank": 1,
                }
            )
    return pd.DataFrame(rows)


def spread_only_accepted_strategy_rows() -> pd.DataFrame:
    frame = accepted_strategy_rows().copy()
    frame["backtest_mode"] = "spread"
    return frame


def incomplete_two_leg_input_rows() -> pd.DataFrame:
    frame = accepted_strategy_rows().copy()
    frame["has_beta"] = False
    frame["has_funding_x"] = False
    frame["has_funding_y"] = False
    return frame


def test_strategy_acceptance_requires_multi_pair_and_required_cost_buckets():
    gate = AcceptanceGate()

    accepted = strategy_acceptance_report(accepted_strategy_rows(), gate)
    missing_pair = strategy_acceptance_report(
        accepted_strategy_rows().query("pair == 'ETH-BTC'"), gate
    )
    missing_stress = strategy_acceptance_report(
        accepted_strategy_rows().query("cost_bucket == 'base'"), gate
    )

    assert bool(accepted["production_eligible"].iloc[0]) is True
    assert bool(accepted["preferred_eligible"].iloc[0]) is True
    assert bool(accepted["research_eligible"].iloc[0]) is True
    assert accepted["two_leg_pairs_tested"].iloc[0] == 2
    assert accepted["two_leg_passing_pairs"].iloc[0] == 2
    assert accepted["acceptance_reason"].iloc[0] == "passed"
    assert bool(missing_pair["production_eligible"].iloc[0]) is False
    assert "pairs_tested<2" in missing_pair["acceptance_reason"].iloc[0]
    assert bool(missing_stress["production_eligible"].iloc[0]) is False
    assert "missing_cost_buckets:stress" in missing_stress["acceptance_reason"].iloc[0]


def test_acceptance_gate_rejects_nonfinite_and_unreconciled_open_backtest():
    gate = AcceptanceGate()
    valid = BacktestResult(
        trades=250, profit_factor=2.1, expectancy=0.01, sharpe=1.8,
        max_drawdown=0.05, win_rate=0.55, total_return=0.1,
        sharpe_status="valid", expectancy_lower_95=0.005,
    )
    assert gate.evaluate(valid) == (True, "passed")
    invalid = replace(
        valid, profit_factor=float("inf"), sharpe_status="blocked",
        open_trades=1, reconciliation_error=0.1, expectancy_lower_95=None,
    )
    accepted, reason = gate.evaluate(invalid)
    assert not accepted
    assert "profit_factor_nonfinite" in reason
    assert "sharpe_invalid_or_unknown_interval" in reason
    assert "open_trades_at_end" in reason
    assert "expectancy_lower_95<=0_or_missing" in reason
    assert "reconciliation_error>" in reason


def test_strategy_acceptance_rechecks_claimed_eligible_rows():
    rows = accepted_strategy_rows()
    rows.loc[0, "open_trades"] = 1
    report = strategy_acceptance_report(rows, AcceptanceGate())
    assert not bool(report["production_eligible"].iloc[0])
    assert "unverified_eligible_runs:1" in report["acceptance_reason"].iloc[0]

    missing = accepted_strategy_rows().drop(columns=["expectancy_lower_95"])
    report = strategy_acceptance_report(missing, AcceptanceGate())
    assert not bool(report["production_eligible"].iloc[0])
    assert "unverified_eligible_runs:4" in report["acceptance_reason"].iloc[0]

    malformed = accepted_strategy_rows()
    malformed["expectancy_lower_95"] = malformed["expectancy_lower_95"].astype(object)
    malformed.at[0, "expectancy_lower_95"] = [0.005]
    report = strategy_acceptance_report(malformed, AcceptanceGate())
    assert not bool(report["production_eligible"].iloc[0])
    assert "unverified_eligible_runs:1" in report["acceptance_reason"].iloc[0]


def test_strategy_acceptance_counts_repriced_trades_once():
    rows = accepted_strategy_rows()
    rows["trades"] = 70
    report = strategy_acceptance_report(rows, AcceptanceGate(min_trades=50))
    row = report.iloc[0]
    assert bool(row["production_eligible"])
    assert row["total_trades"] == 140
    assert not bool(row["preferred_eligible"])
    assert "total_trades<250" in row["preferred_reason"]


def test_strategy_acceptance_rejects_string_boolean_execution_flags():
    rows = accepted_strategy_rows()
    rows["has_beta"] = "False"
    report = strategy_acceptance_report(rows, AcceptanceGate())
    assert not bool(report["production_eligible"].iloc[0])
    assert "two_leg_execution_input_pairs<2" in report["acceptance_reason"].iloc[0]

    rows = accepted_strategy_rows()
    rows["eligible"] = "True"
    report = strategy_acceptance_report(rows, AcceptanceGate())
    assert not bool(report["production_eligible"].iloc[0])
    assert "passing_pairs<2" in report["acceptance_reason"].iloc[0]


def test_strategy_acceptance_rejects_spread_only_results_for_production():
    gate = AcceptanceGate()

    report = strategy_acceptance_report(spread_only_accepted_strategy_rows(), gate)

    assert bool(report["production_eligible"].iloc[0]) is False
    assert report["two_leg_pairs_tested"].iloc[0] == 0
    assert report["two_leg_passing_pairs"].iloc[0] == 0
    assert "two_leg_pairs<2" in report["acceptance_reason"].iloc[0]


def test_strategy_acceptance_rejects_incomplete_two_leg_execution_inputs():
    gate = AcceptanceGate()

    report = strategy_acceptance_report(incomplete_two_leg_input_rows(), gate)

    assert bool(report["production_eligible"].iloc[0]) is False
    assert report["two_leg_pairs_tested"].iloc[0] == 2
    assert report["two_leg_execution_input_pairs"].iloc[0] == 0
    assert "two_leg_execution_input_pairs<2" in report["acceptance_reason"].iloc[0]
    assert "two_leg_missing_inputs" in report["acceptance_reason"].iloc[0]


def test_write_reports_includes_acceptance_report(tmp_path):
    config = ExperimentConfig(gate=AcceptanceGate(min_pairs=2))
    harness = ExperimentHarness(strategies=(STRATEGIES[0],), config=config)

    paths = harness.write_reports(accepted_strategy_rows(), tmp_path)

    assert "acceptance" in paths
    report = pd.read_csv(paths["acceptance"])
    assert list(report.columns) == [
        "strategy_id",
        "strategy_name",
        "family",
        "implementation_kind",
        "acceptance_authority",
        "production_eligible",
        "preferred_eligible",
        "research_eligible",
        "research_tier",
        "acceptance_reason",
        "preferred_reason",
        "research_reason",
        "evaluated_runs",
        "passing_runs",
        "pairs_tested",
        "passing_pairs",
        "research_pairs_tested",
        "research_passing_pairs",
        "two_leg_pairs_tested",
        "two_leg_execution_input_pairs",
        "two_leg_passing_pairs",
        "research_score",
        "research_conviction_score",
        "research_stability_score",
        "research_breadth_score",
        "research_strengths",
        "research_concerns",
        "research_setup_mix",
        "research_top_pairs",
        "research_next_step",
        "required_cost_buckets",
        "required_backtest_mode",
        "required_two_leg_inputs",
        "total_trades",
        "median_profit_factor",
        "median_sharpe",
        "worst_drawdown",
    ]
    assert bool(report["production_eligible"].iloc[0])


def test_strategy_acceptance_reports_elastic_research_tier_for_promising_small_sample():
    frame = accepted_strategy_rows().copy()
    frame["eligible"] = False
    frame["reason"] = "trades<100"
    frame["trades"] = 28
    frame["profit_factor"] = 1.35
    frame["expectancy"] = 0.03
    frame["sharpe"] = 1.05
    frame["max_drawdown"] = 0.45

    report = strategy_acceptance_report(frame, AcceptanceGate())

    row = report.iloc[0]
    assert bool(row["production_eligible"]) is False
    assert bool(row["research_eligible"]) is True
    assert row["research_tier"] in {"research_watch", "research_explore", "research_candidate"}
    assert row["research_pairs_tested"] == 2
    assert row["research_conviction_score"] >= row["research_score"]
    assert row["research_next_step"] in {
        "promote_to_focused_paper_validation",
        "expand_history_and_retest",
        "tighten_risk_controls_then_retest",
        "refine_filters_and_compare_variants",
        "monitor_in_research_queue",
    }
