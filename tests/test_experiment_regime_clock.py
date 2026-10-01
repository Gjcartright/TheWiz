"""Synthetic regime-clock admission, preserving the complete experiment grid."""
from dataclasses import asdict, replace
from itertools import product

import pandas as pd
import pytest

from quant_platform import experiments
from quant_platform.backtest import backtest_pair, backtest_two_leg_spread
from quant_platform.experiments import (
    ExperimentConfig,
    ExperimentHarness,
    PairDataset,
    strategy_acceptance_report,
)
from quant_platform.regimes import RegimeConfig, classify_regimes
from quant_platform.strategies import STRATEGIES


def _frame(offsets=(0, 1, 2, 3, 4, 5), *, two_leg=True):
    count = len(offsets)
    spread = [-2.0, -1.0, 0.0, 1.0, 2.0, 0.0]
    frame = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=value) for value in offsets],
        "spread": [spread[index % 6] for index in range(count)],
        "zscore": [(-2.2, -1.0, 0.0, 1.0, 2.2, 0.0)[index % 6] for index in range(count)],
    })
    if two_leg:
        frame = frame.assign(price_x=[100.0 + index for index in range(count)],
                             price_y=[50.0 + index * .5 for index in range(count)],
                             hedge_ratio=1.2, funding_x_bps=2.0, funding_y_bps=3.0)
    return frame


def _harness(*, overall=True, min_rows=1, strategies=None):
    if strategies is None:
        strategies = (STRATEGIES[0], replace(STRATEGIES[0], id=10001, name="Second synthetic strategy"))
    return ExperimentHarness(strategies=strategies, config=ExperimentConfig(
        min_rows=min_rows, include_overall_regime=overall,
    ))


def _identities(results):
    return set(results[["strategy_id", "regime", "cost_bucket"]].itertuples(index=False, name=None))


@pytest.mark.parametrize("two_leg", [False, True])
def test_original_six_row_inferred_regimes_retain_every_cell(two_leg):
    # Original CLI fixture: no explicit regime. The real classifier creates
    # singleton slices alongside evaluable ALL/bull slices.
    frame = classify_regimes(_frame(two_leg=two_leg), RegimeConfig(preserve_existing=True))
    original = frame.copy(deep=True)
    harness = _harness()
    results = harness.run([PairDataset("ETH-BTC", frame)])
    regimes = {"ALL", *frame.regime.unique()}
    expected = set(product([strategy.id for strategy in harness.strategies], regimes,
                           [bucket.name for bucket in harness.config.cost_buckets]))
    assert _identities(results) == expected and len(results) == len(expected)
    sizes = frame.groupby("regime").size().to_dict() | {"ALL": len(frame)}
    assert all(row.observations == sizes[row.regime] for row in results.itertuples())
    invalid = results[results.observations < 3]
    assert not invalid.empty
    assert invalid.status.eq("skipped").all()
    assert invalid.reason.eq("funding clock is not valid: insufficient_timestamp_observations").all()
    assert invalid.backtest_mode.eq("not_run").all() and not invalid.eligible.any()
    assert invalid.rank_score.eq(-1_000_000).all()
    assert results[results.regime.eq("ALL")].status.eq("evaluated").all()
    pd.testing.assert_frame_equal(frame, original)


def test_mixed_good_irregular_and_small_regimes_continue_in_native_validation_order():
    frame = _frame(offsets=(0, 1, 2, 3, 4, 6, 7, 8))
    frame["regime"] = ["good"] * 3 + ["irregular"] * 3 + ["small"] * 2
    calls = []
    def signal(selected):
        calls.append(tuple(selected.index))
        return pd.Series(0.0, index=selected.index)
    strategy = replace(STRATEGIES[0], signal_function=signal)
    harness = _harness(overall=False, strategies=(strategy,))
    results = harness.run([PairDataset("ETH-BTC", frame)])
    assert len(results) == 6 and len(calls) == 6
    assert set(calls) == {(0, 1, 2), (3, 4, 5), (6, 7)}
    assert results[results.regime.eq("good")].status.eq("evaluated").all()
    irregular = results[results.regime.eq("irregular")]
    assert irregular.reason.eq("funding clock is not valid: irregular_timestamp_grid").all()
    assert irregular.observations.eq(3).all()
    assert results[results.regime.eq("small")].reason.str.endswith("insufficient_timestamp_observations").all()
    assert not results[results.regime.ne("good")].eligible.any()


def test_all_invalid_population_cannot_become_accepted_science():
    frame = _frame(offsets=(0, 1)).assign(regime="small")
    harness = _harness()
    results = harness.run([PairDataset("ETH-BTC", frame)])
    assert len(results) == 8
    assert results.status.eq("skipped").all()
    assert not results.eligible.any() and results.backtest_mode.eq("not_run").all()
    acceptance = strategy_acceptance_report(results, harness.config.gate)
    assert len(acceptance) == 2 and acceptance.evaluated_runs.eq(0).all()
    assert not acceptance[["production_eligible", "preferred_eligible", "research_eligible"]].any().any()


@pytest.mark.parametrize("two_leg", [False, True])
def test_valid_cells_numerically_match_unmodified_direct_backtest(two_leg):
    frame = _frame(two_leg=two_leg)
    harness = _harness(overall=False, strategies=(STRATEGIES[0],))
    frame["regime"] = "valid"
    results = harness.run([PairDataset("ETH-BTC", frame)])
    prepared = harness.feature_engine.score_frame(experiments._coalesce_signal_zscore(frame))
    prepared = experiments._coalesce_signal_zscore(prepared)
    signal = STRATEGIES[0].signal_function(prepared)
    for bucket in harness.config.cost_buckets:
        reference = (backtest_two_leg_spread if two_leg else backtest_pair)(prepared, signal, bucket.cost_model)
        row = results[results.cost_bucket.eq(bucket.name)].iloc[0]
        assert row.status == "evaluated"
        for key, value in asdict(reference).items():
            if key in results.columns:
                if pd.isna(value):
                    assert pd.isna(row[key])
                else:
                    assert row[key] == value


def test_realized_two_leg_funding_retains_original_clock_bypass():
    frame = _frame(offsets=(0, 1)).assign(
        regime="realized", funding_x_realized_bps=0.1, funding_y_realized_bps=-0.2,
    )
    harness = _harness(overall=False, strategies=(STRATEGIES[0],))
    results = harness.run([PairDataset("ETH-BTC", frame)])
    assert results.status.eq("evaluated").all()
    assert results.observations.eq(2).all()


def test_absent_timestamp_clock_preserves_existing_declared_cost_divisor():
    frame = _frame(offsets=(0, 1), two_leg=False).drop(columns="timestamp").assign(regime="legacy")
    results = _harness(overall=False).run([PairDataset("ETH-BTC", frame)])
    assert results.status.eq("evaluated").all()




def test_existing_min_rows_disposition_takes_precedence():
    results = _harness(min_rows=20).run([PairDataset("ETH-BTC", _frame().assign(regime="small"))])
    assert results.reason.eq("rows<20").all()
    assert results.observations.eq(6).all()


@pytest.mark.parametrize("message", [
    "synthetic signal failure on insufficient grid",
    "funding clock is not valid: insufficient_timestamp_observations",
    "funding clock is not valid: irregular_timestamp_grid",
])
def test_insufficient_grid_never_masks_signal_failure_even_with_matching_error_text(message):
    frame = _frame(offsets=(0, 1)).assign(regime="small")
    def broken(_frame):
        raise ValueError(message)
    strategy = replace(STRATEGIES[0], signal_function=broken)
    with pytest.raises(ValueError) as caught:
        _harness(overall=False, strategies=(strategy,)).run([PairDataset("ETH-BTC", frame)])
    assert str(caught.value) == message




def test_insufficient_grid_does_not_mask_incomplete_realized_funding_contract():
    frame = _frame(offsets=(0, 1)).assign(regime="small", funding_x_realized_bps=0.1)
    with pytest.raises(ValueError, match="realized funding requires both leg columns"):
        _harness(overall=False, strategies=(STRATEGIES[0],)).run([PairDataset("ETH-BTC", frame)])
