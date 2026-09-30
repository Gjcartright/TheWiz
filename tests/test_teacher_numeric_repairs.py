"""Advisory model numeric domains, no services or publication."""
from datetime import datetime, timezone
import math
import pytest
from pydantic import ValidationError
from quant_platform.orchestration.teacher_contracts import (
    EXACT_MODES, StudentRouterPrediction, StudentOutcomeForecast,
    TeacherAction, TeacherProposal, CouncilContext, EvidenceAuthority,
)
from quant_platform.orchestration.contracts import CandidateIdentity
from quant_platform.orchestration.teacher_council import _teacher_vote, _weighted_average, TeacherCouncilPolicy
from quant_platform.performance_math import MATH_VERSION


def probabilities():
    result = {mode.value: 0. for mode in EXACT_MODES}
    result[TeacherAction.ABSTAIN.value] = 1.
    return result


@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf])
def test_router_probability_must_be_finite(bad):
    values = probabilities()
    values[next(iter(values))] = bad
    with pytest.raises(ValidationError):
        StudentRouterPrediction(prediction_id='p', context_id='c', model_version='m', feature_schema_version='f',
                                mode_probabilities=values, uncertainty=.1, training_support_score=.8,
                                feature_completeness_score=1., evidence_paths=('research',))


def forecast():
    return dict(forecast_id='f', context_id='c', model_version='m', probability_positive=.6,
                expected_net_return=.01, lower_bound_net_return=-.01, median_net_return=.005, upper_bound_net_return=.02,
                expected_mae=.03, expected_mfe=.05, expected_holding_bars=5,
                structural_break_probability=.1, execution_failure_probability=.1, uncertainty=.2,
                evidence_paths=('research',))


@pytest.mark.parametrize('field', ['expected_net_return', 'expected_mae', 'expected_mfe', 'expected_holding_bars'])
@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf])
def test_forecasts_do_not_accept_nonfinite_numbers(field, bad):
    with pytest.raises(ValidationError):
        StudentOutcomeForecast(**{**forecast(), field: bad})


def proposal(confidence, uncertainty):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    context = CouncilContext(pair='BTC-USD/ETH-USD', venue='dydx', timeframe='1h', lookback=100,
                             source_snapshot_id='snapshot', source_timestamp=now)
    return TeacherProposal(proposal_id='p', context=context,
        candidate=CandidateIdentity(pair=context.pair, venue=context.venue, strategy_family=EXACT_MODES[0].value,
                                    timeframe=context.timeframe, lookback=context.lookback, formula_version='local'),
        teacher_id='teacher', exact_mode=EXACT_MODES[0], proposed_action=TeacherAction.SHORT_X_LONG_Y,
        confidence=confidence, uncertainty=uncertainty, expected_net_return=.01, lower_bound_net_return=-.01,
        entry_style='entry', exit_style='exit', invalidation_condition='invalid', source_system='local',
        authority=EvidenceAuthority.LOCAL_POINT_IN_TIME, point_in_time_status='confirmed',
        formula_version='local', math_version=MATH_VERSION, source_timestamp=now, evidence_paths=('research',))


@pytest.mark.parametrize('confidence,uncertainty', [(0., 0.), (1., 1.), (0., 1.)])
def test_zero_usable_vote_deterministically_abstains(confidence, uncertainty):
    result = _teacher_vote((proposal(confidence, uncertainty),))
    assert result['action'] == TeacherAction.ABSTAIN
    assert result['selected'] is None
    assert result['confidence'] == 0.


def test_valid_advisory_values_remain_supported():
    result = StudentOutcomeForecast(**forecast())
    assert result.shadow_only is True
    vote = _teacher_vote((proposal(.8, .1),))
    assert vote['action'] == TeacherAction.SHORT_X_LONG_Y


@pytest.mark.parametrize('field', ['max_evidence_age_hours', 'max_teacher_disagreement', 'min_weighted_confidence'])
@pytest.mark.parametrize('bad', [math.nan, math.inf, -1., True, '0.5', None, [], {}])
def test_invalid_thresholds_do_not_disable_numeric_checks(field, bad):
    with pytest.raises(ValueError):
        TeacherCouncilPolicy(**{field: bad})


def test_duplicate_proposals_are_not_additional_votes():
    p = proposal(.8, .1)
    result = _teacher_vote((p, p))
    assert result['action'] == TeacherAction.ABSTAIN
    assert result['selected'] is None


@pytest.mark.parametrize('value', [1e308, -1e308, float.fromhex('0x1.fffffffffffffp+1023')])
def test_weighted_average_is_finite_at_representational_extremes(value):
    assert _weighted_average([(value, .72)] * 7, default=None) == pytest.approx(value)


def test_convex_average_of_large_opposite_values_is_not_overflow():
    result = _weighted_average([(1e308, .75), (-1e308, .25)], default=None)
    assert result == pytest.approx(5e307)


@pytest.mark.parametrize('rows', [[(math.inf, 1.)], [(1., math.inf)], [(1., -1.)], [(None, 1.)]])
def test_invalid_weighted_values_remain_unavailable(rows):
    assert _weighted_average(rows, default=None) is None


def test_zero_weight_extreme_cannot_change_average_scale():
    assert _weighted_average([(1e308, 0.), (1e-308, 1.)], default=None) == 1e-308


def test_cancellation_does_not_erase_a_small_weighted_observation():
    assert _weighted_average([(1e308, 1.), (-1e308, 1.), (3e-308, 1.)], default=None) == 1e-308
