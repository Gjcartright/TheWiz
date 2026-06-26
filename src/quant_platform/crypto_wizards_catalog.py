from __future__ import annotations

from dataclasses import dataclass


BASE_URL = "https://api.cryptowizards.net"


@dataclass(frozen=True)
class CryptoWizardsEndpoint:
    name: str
    method: str
    path: str
    credits: int
    required: tuple[str, ...]
    notes: str


ENDPOINTS: tuple[CryptoWizardsEndpoint, ...] = (
    CryptoWizardsEndpoint(
        "backtest_get",
        "GET",
        "/v1beta/backtest",
        6,
        ("symbol_1", "symbol_2", "exchange", "interval", "period", "strategy"),
        "Data-provided backtest metrics; optional with_history returns spread stats and returns.",
    ),
    CryptoWizardsEndpoint(
        "cointegration_get",
        "GET",
        "/v1beta/cointegration",
        5,
        ("symbol_1", "symbol_2", "exchange", "interval", "period"),
        "Engle-Granger cointegration with optional history.",
    ),
    CryptoWizardsEndpoint(
        "copula_get",
        "GET",
        "/v1beta/copula",
        5,
        ("symbol_1", "symbol_2", "exchange", "interval", "period"),
        "Copula family and conditional probabilities.",
    ),
    CryptoWizardsEndpoint(
        "correlations_get",
        "GET",
        "/v1beta/correlations",
        5,
        ("symbol_1", "symbol_2", "exchange", "interval", "period"),
        "Pearson, Spearman, and Kendall correlation metrics.",
    ),
    CryptoWizardsEndpoint(
        "credits_used_get",
        "GET",
        "/v1beta/credits-used",
        0,
        (),
        "Current UTC-day credit usage for the API key.",
    ),
    CryptoWizardsEndpoint(
        "prescanned_get",
        "GET",
        "/v1beta/prescanned",
        10,
        ("priority", "strategy"),
        "Top prescanned pair opportunities sorted by priority.",
    ),
    CryptoWizardsEndpoint(
        "spread_get",
        "GET",
        "/v1beta/spread",
        5,
        ("symbol_1", "symbol_2", "exchange", "interval", "period"),
        "Spread, z-score, hedge ratio, half-life, and Hurst statistics.",
    ),
    CryptoWizardsEndpoint(
        "zscores_get",
        "GET",
        "/v1beta/zscores",
        5,
        ("symbol_1", "symbol_2", "exchange", "interval", "period"),
        "Latest z-score and rolling z-score, optional history.",
    ),
    CryptoWizardsEndpoint(
        "backtest_post",
        "POST",
        "/v1beta/backtest",
        2,
        ("params", "bt_inputs"),
        "Lower-credit backtest using caller-supplied price data.",
    ),
    CryptoWizardsEndpoint(
        "cointegration_post",
        "POST",
        "/v1beta/cointegration",
        1,
        ("series_1_closes", "series_2_closes"),
        "Lower-credit cointegration using caller-supplied closes.",
    ),
    CryptoWizardsEndpoint(
        "copula_post",
        "POST",
        "/v1beta/copula",
        1,
        ("series_1_closes", "series_2_closes"),
        "Lower-credit copula using caller-supplied closes.",
    ),
    CryptoWizardsEndpoint(
        "correlations_post",
        "POST",
        "/v1beta/correlations",
        1,
        ("series_1_closes", "series_2_closes"),
        "Lower-credit correlations using caller-supplied closes.",
    ),
    CryptoWizardsEndpoint(
        "spread_post",
        "POST",
        "/v1beta/spread",
        1,
        ("series_1_closes", "series_2_closes"),
        "Lower-credit spread statistics using caller-supplied closes.",
    ),
    CryptoWizardsEndpoint(
        "zscores_post",
        "POST",
        "/v1beta/zscores",
        1,
        ("series_1_closes", "series_2_closes"),
        "Lower-credit z-score metrics using caller-supplied closes.",
    ),
    CryptoWizardsEndpoint(
        "pair_detail_dashboard",
        "DASHBOARD",
        "https://cryptowizards.net/wizards/zscore/pair/<pair_id>",
        0,
        ("authenticated_browser_session", "pair_id"),
        "Authenticated dashboard research surface with ECM chart options, copula stats, dependency views, and backtest/risk metrics.",
    ),
)


def endpoint_rows() -> list[dict[str, object]]:
    return [
        {
            "name": endpoint.name,
            "method": endpoint.method,
            "path": endpoint.path,
            "credits": endpoint.credits,
            "required": ";".join(endpoint.required),
            "notes": endpoint.notes,
        }
        for endpoint in ENDPOINTS
    ]
