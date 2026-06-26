import json

import pytest
import pandas as pd

from quant_platform.experiments import AcceptanceGate, CostBucket, ExperimentConfig, ExperimentHarness
from quant_platform.backtest import CostModel
from quant_platform.fixture_ingestion import (
    datasets_from_fixtures,
    discovered_field_rows,
    load_fixture_payloads,
    normalize_crypto_wizards_records,
    write_fixture_field_dictionary,
)
from quant_platform.strategies import STRATEGIES


def write_fixture_files(tmp_path):
    json_payload = {
        "endpoint": "pair_metrics",
        "data": [
            {
                "timestamp": "2026-06-15T00:00:00Z",
                "pair": "ETH/BTC",
                "metrics": {
                    "spread": -0.5,
                    "zScore": -2.1,
                    "copula": {"u1_given_u2": 0.2, "u2_given_u1": 0.7},
                    "ecm": {"ecm_x": -0.1, "ecm_y": -0.02, "ecm_strength": 0.7},
                    "regime": "range",
                },
            },
            {
                "timestamp": "2026-06-15T01:00:00Z",
                "pair": "ETH/BTC",
                "metrics": {
                    "spread": -0.2,
                    "zScore": -0.8,
                    "copula": {"u1_given_u2": 0.35, "u2_given_u1": 0.55},
                    "regime": "range",
                },
            },
        ],
    }
    (tmp_path / "pair_metrics.json").write_text(json.dumps(json_payload), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "timestamp": "2026-06-15T02:00:00Z",
                "pair": "SOL-ETH",
                "spread": 0.8,
                "zscore": 2.2,
                "u1_given_u2": 0.8,
                "u2_given_u1": 0.25,
                "regime": "bull",
            },
            {
                "timestamp": "2026-06-15T03:00:00Z",
                "pair": "SOL-ETH",
                "spread": 0.1,
                "zscore": 0.2,
                "u1_given_u2": 0.51,
                "u2_given_u1": 0.49,
                "regime": "bull",
            },
        ]
    ).to_csv(tmp_path / "pair_metrics.csv", index=False)


def test_loads_json_and_csv_fixture_payloads(tmp_path):
    write_fixture_files(tmp_path)

    payloads = load_fixture_payloads(tmp_path)

    assert {payload.path.suffix for payload in payloads} == {".json", ".csv"}
    assert {payload.endpoint for payload in payloads} == {"pair_metrics"}


def test_fixture_ingestion_normalizes_pair_datasets_and_copula_distortion(tmp_path):
    write_fixture_files(tmp_path)

    datasets = datasets_from_fixtures(tmp_path)

    assert {dataset.pair for dataset in datasets} == {"ETH-BTC", "SOL-ETH"}
    eth = next(dataset for dataset in datasets if dataset.pair == "ETH-BTC")
    assert {"spread", "zscore", "conditional_probability_distortion", "regime"}.issubset(eth.frame.columns)
    assert eth.frame["conditional_probability_distortion"].iloc[0] == pytest.approx(-0.5)


def test_fixture_field_dictionary_has_required_columns(tmp_path):
    write_fixture_files(tmp_path)

    rows = discovered_field_rows(tmp_path)
    output = write_fixture_field_dictionary(tmp_path, tmp_path / "field_dictionary.csv")
    written = pd.read_csv(output)

    assert rows
    assert list(written.columns) == [
        "name",
        "description",
        "type",
        "example_value",
        "endpoint",
        "importance_score",
        "notes",
    ]
    assert "spread" in set(written["name"])
    assert "zscore" in set(written["name"])


def test_fixture_datasets_flow_into_experiment_harness(tmp_path):
    write_fixture_files(tmp_path)
    datasets = datasets_from_fixtures(tmp_path)
    config = ExperimentConfig(
        cost_buckets=(CostBucket("base", CostModel()),),
        gate=AcceptanceGate(min_profit_factor=0.1, min_sharpe=-99, max_drawdown=1.0, min_trades=1),
        min_rows=1,
    )
    harness = ExperimentHarness(strategies=(STRATEGIES[0], STRATEGIES[4]), config=config)

    results = harness.run(datasets)

    assert not results.empty
    assert set(results["pair"]) == {"ETH-BTC", "SOL-ETH"}
    assert set(results["status"]) == {"evaluated"}


def test_fixture_ingestion_derives_beta_from_leg_prices_when_missing():
    records = [
        {"pair": "ETH/BTC", "spread": -0.4, "price_x": 100.0, "price_y": 50.0},
        {"pair": "ETH/BTC", "spread": -0.2, "price_x": 102.0, "price_y": 51.0},
        {"pair": "ETH/BTC", "spread": 0.1, "price_x": 101.0, "price_y": 50.0},
        {"pair": "ETH/BTC", "spread": 0.3, "price_x": 104.0, "price_y": 52.0},
        {"pair": "ETH/BTC", "spread": 0.0, "price_x": 103.0, "price_y": 51.5},
    ]

    normalized = normalize_crypto_wizards_records(records)

    assert "beta" in normalized.columns
    assert "beta_source" in normalized.columns
    assert set(normalized["beta_source"]) == {"derived_from_price_returns"}
