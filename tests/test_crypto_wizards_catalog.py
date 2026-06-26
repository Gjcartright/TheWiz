from quant_platform.crypto_wizards_catalog import BASE_URL, ENDPOINTS, endpoint_rows


def test_crypto_wizards_catalog_contains_documented_v1beta_endpoints():
    names = {endpoint.name for endpoint in ENDPOINTS}

    assert BASE_URL == "https://api.cryptowizards.net"
    assert len(ENDPOINTS) == 15
    assert {
        "backtest_get",
        "cointegration_get",
        "copula_get",
        "correlations_get",
        "credits_used_get",
        "prescanned_get",
        "spread_get",
        "zscores_get",
        "backtest_post",
        "cointegration_post",
        "copula_post",
        "correlations_post",
        "spread_post",
        "zscores_post",
        "pair_detail_dashboard",
    } == names
    assert all(endpoint.path.startswith("/v1beta/") for endpoint in ENDPOINTS if endpoint.method != "DASHBOARD")


def test_endpoint_rows_are_csv_ready():
    rows = endpoint_rows()

    assert rows[0]["name"] == "backtest_get"
    assert isinstance(rows[0]["required"], str)
