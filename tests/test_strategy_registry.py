from quant_platform.strategies import STRATEGIES


def test_all_required_strategies_are_registered():
    assert len(STRATEGIES) == 37
    assert [strategy.id for strategy in STRATEGIES] == list(range(1, 38))


def test_deterministic_strategy_families_have_signal_functions():
    executable_ids = {strategy.id for strategy in STRATEGIES if strategy.signal_function is not None}

    assert executable_ids == set(range(1, 38))
