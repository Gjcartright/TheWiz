from quant_platform import pair_market_utils


def test_pair_markets_from_pair_handles_four_segment_dydx_format():
    assert pair_market_utils.pair_markets_from_pair("BTC-USD-ETH-USD") == ["BTC-USD", "ETH-USD"]


def test_pair_markets_from_pair_handles_two_segment_pair():
    assert pair_market_utils.pair_markets_from_pair("DOGE/LTC") == ["DOGE-USD", "LTC-USD"]


def test_pair_markets_from_pair_returns_empty_when_invalid():
    assert pair_market_utils.pair_markets_from_pair("") == []


def test_normalize_dydx_market_defaults_to_usd_quote():
    assert pair_market_utils.normalize_dydx_market("eth") == "ETH-USD"
