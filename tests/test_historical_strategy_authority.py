"""Historical strategy repairs must remain causal and research scoped."""

import numpy as np
import pandas as pd
import pytest

from quant_platform.backtest import CostModel
from quant_platform.experiments import (
    AcceptanceGate,
    CostBucket,
    ExperimentConfig,
    ExperimentHarness,
    PairDataset,
    strategy_acceptance_report,
)
from quant_platform.runtime_types import strict_bool
from quant_platform.strategies import (
    STRATEGY_BY_ID,
    STRATEGIES,
    _stateful,
    gmm_regime_signal,
    hmm_regime_signal,
    kmeans_regime_signal,
    pair_ranking_signal,
    portfolio_rotation_signal,
    risk_adjusted_ranking_signal,
    meta_model_proxy_signal,
)

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
                }
            )
    return pd.DataFrame(rows)


def experiment_frame(*, two_leg: bool) -> pd.DataFrame:
    index = np.arange(160)
    spread = np.sin(np.linspace(0, 12 * np.pi, len(index)))
    frame = pd.DataFrame({"spread": spread, "zscore": spread * 2.5})
    if two_leg:
        frame["price_x"] = 100 + index * 0.02 + spread
        frame["price_y"] = 50 + index * 0.01 - spread
        frame["hedge_ratio"] = 1.2
        frame["beta"] = 0.8
        frame["funding_x_bps"] = 2.0
        frame["funding_y_bps"] = 3.0
    return frame


def experiment_harness() -> ExperimentHarness:
    return ExperimentHarness(
        strategies=(STRATEGIES[0],),
        config=ExperimentConfig(
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
            gate=AcceptanceGate(min_profit_factor=0.1, min_sharpe=-99, max_drawdown=1.0, min_trades=1),
        ),
    )


@pytest.mark.parametrize("signal", (hmm_regime_signal, gmm_regime_signal, kmeans_regime_signal))
def test_regime_signal_prefix_does_not_change_with_future_rows(signal):
    rows = 40
    prefix = pd.DataFrame(
        {
            "returns": np.sin(np.arange(rows)) * 0.02,
            "zscore": np.where(np.arange(rows) % 2 == 0, 3.0, 0.0),
        }
    )
    future = pd.DataFrame(
        {
            "returns": [0.9, -0.9] * 20,
            "zscore": np.tile([3.0, 0.0], 20),
        }
    )
    extended = pd.concat((prefix, future), ignore_index=True)
    pd.testing.assert_series_equal(signal(prefix), signal(extended).iloc[:rows])


@pytest.mark.parametrize(
    "signal",
    (pair_ranking_signal, portfolio_rotation_signal, risk_adjusted_ranking_signal, meta_model_proxy_signal),
)
def test_missing_model_or_rank_inputs_do_not_create_trades(signal):
    frame = pd.DataFrame({"zscore": [3.0] * 40})
    assert not signal(frame).any()


def test_string_false_exit_mask_does_not_close_position():
    result = _stateful(pd.Series([1.0, 0.0, 0.0]), pd.Series(["False", "False", "True"]))
    assert result.tolist() == [1.0, 1.0, 0.0]
    assert strict_bool("False") is False
    assert strict_bool("True") is True


def test_spread_only_harness_row_is_never_acceptance_eligible():
    result = experiment_harness().run([PairDataset("ETH-BTC", experiment_frame(two_leg=False))])
    assert result["status"].eq("evaluated").all()
    assert not result["eligible"].any()
    assert result["reason"].str.contains("spread_only_pnl_has_no_acceptance_notional").all()


@pytest.mark.parametrize("bad_value", (np.nan, True))
def test_present_but_incomplete_funding_is_skipped(bad_value):
    frame = experiment_frame(two_leg=True)
    if isinstance(bad_value, bool):
        frame["funding_x_bps"] = frame["funding_x_bps"].astype(object)
    frame.loc[10, "funding_x_bps"] = bad_value
    result = experiment_harness().run([PairDataset("ETH-BTC", frame)])
    assert result["status"].eq("skipped").all()
    assert result["reason"].str.contains("incomplete_numeric_inputs:funding_x_bps").all()


def test_present_but_incomplete_zscore_is_skipped():
    frame = experiment_frame(two_leg=True)
    frame.loc[40, "zscore"] = np.nan
    result = experiment_harness().run([PairDataset("ETH-BTC", frame)])
    assert result["status"].eq("skipped").all()
    assert result["reason"].str.contains("incomplete_strategy_inputs:zscore").all()


@pytest.mark.parametrize("strategy_id", (3, 14, 23, 30))
def test_proxy_strategy_cannot_gain_production_authority_from_score(strategy_id):
    strategy = STRATEGY_BY_ID[strategy_id]
    rows = accepted_strategy_rows()
    rows["strategy_id"] = strategy_id
    rows["strategy_name"] = strategy.name
    rows["family"] = strategy.family
    report = strategy_acceptance_report(rows, AcceptanceGate())
    row = report.iloc[0]
    assert not row["production_eligible"]
    assert not row["preferred_eligible"]
    assert row["acceptance_authority"].startswith("research_only")
    assert "acceptance_authority:" in row["acceptance_reason"]


@pytest.mark.parametrize("kind", ("unknown", "mismatched"))
def test_unregistered_or_spoofed_strategy_cannot_gain_authority(kind):
    rows = accepted_strategy_rows()
    if kind == "unknown":
        rows["strategy_id"] = 9999
    else:
        rows["strategy_name"] = "Different strategy"
    row = strategy_acceptance_report(rows, AcceptanceGate()).iloc[0]
    assert not row["production_eligible"]
    assert "acceptance_authority:" in row["acceptance_reason"]
