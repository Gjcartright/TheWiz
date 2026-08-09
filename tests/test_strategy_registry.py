from quant_platform.strategies import ALL_STRATEGIES, LEGACY_RESEARCH_STRATEGIES, OFFICIAL_CRYPTO_WIZARDS_STRATEGIES, STRATEGIES


def test_full_strategy_library_is_preserved():
    assert len(ALL_STRATEGIES) == 37
    assert [strategy.id for strategy in ALL_STRATEGIES] == list(range(1, 38))


def test_active_registry_uses_crypto_wizards_official_lane():
    assert STRATEGIES == OFFICIAL_CRYPTO_WIZARDS_STRATEGIES
    assert [strategy.id for strategy in STRATEGIES] == [1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 20, 28, 29, 33, 34, 35, 36]
    assert [strategy.id for strategy in LEGACY_RESEARCH_STRATEGIES] == [7, 9, 10, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 37]


def test_registered_strategies_have_signal_functions():
    executable_ids = {strategy.id for strategy in STRATEGIES if strategy.signal_function is not None}

    assert executable_ids == {strategy.id for strategy in STRATEGIES}
