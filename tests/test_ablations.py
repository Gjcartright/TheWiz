import pandas as pd

from quant_platform.ablations import AblationSpec, ablation_report
from quant_platform.experiments import ExperimentHarness


def result_row(strategy_id, name, pf, sharpe, drawdown, expectancy=0.01, pair="ETH-BTC"):
    return {
        "pair": pair,
        "strategy_id": strategy_id,
        "strategy_name": name,
        "family": "test",
        "regime": "ALL",
        "cost_bucket": "base",
        "status": "evaluated",
        "eligible": pf >= 1.8,
        "reason": "passed",
        "trades": 120,
        "profit_factor": pf,
        "expectancy": expectancy,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "win_rate": 0.55,
        "total_return": 0.10,
        "observations": 500,
        "rank_score": 100,
        "rank": 1,
    }


def test_ablation_report_detects_incremental_value():
    results = pd.DataFrame(
        [
            result_row(1, "Classic ZScore Mean Reversion", 1.5, 1.0, 0.12),
            result_row(2, "ZScore + ECM", 2.0, 1.4, 0.08),
        ]
    )

    report = ablation_report(results, (AblationSpec("ecm_vs_z", 2, 1, "ecm", "test"),))

    assert report["status"].iloc[0] == "evaluated"
    assert report["median_pf_delta"].iloc[0] > 0
    assert report["median_drawdown_improvement"].iloc[0] > 0
    assert report["conclusion"].iloc[0] == "adds_value"


def test_ablation_report_detects_harmful_component():
    results = pd.DataFrame(
        [
            result_row(1, "Classic ZScore Mean Reversion", 2.0, 1.4, 0.08),
            result_row(3, "ZScore + Copula", 1.2, 0.7, 0.18),
        ]
    )

    report = ablation_report(results, (AblationSpec("copula_vs_z", 3, 1, "copula", "test"),))

    assert report["status"].iloc[0] == "evaluated"
    assert report["median_pf_delta"].iloc[0] < 0
    assert report["median_drawdown_improvement"].iloc[0] < 0
    assert report["conclusion"].iloc[0] == "hurts"


def test_ablation_report_marks_missing_baseline():
    results = pd.DataFrame([result_row(2, "ZScore + ECM", 2.0, 1.4, 0.08)])

    report = ablation_report(results, (AblationSpec("ecm_vs_z", 2, 1, "ecm", "test"),))

    assert report["status"].iloc[0] == "missing_baseline_strategy"
    assert report["matched_runs"].iloc[0] == 0


def test_experiment_report_bundle_includes_ablation_report(tmp_path):
    results = pd.DataFrame(
        [
            result_row(1, "Classic ZScore Mean Reversion", 1.5, 1.0, 0.12),
            result_row(2, "ZScore + ECM", 2.0, 1.4, 0.08),
        ]
    )
    harness = ExperimentHarness(strategies=())

    paths = harness.write_reports(results, tmp_path)

    assert "ablation" in paths
    written = pd.read_csv(paths["ablation"])
    assert "incremental_score" in written.columns

