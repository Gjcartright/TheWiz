import pandas as pd

from quant_platform.regimes import (
    RegimeConfig,
    classify_regimes,
    regime_distribution,
    regime_pair_strategy_report,
    write_regime_dataset_report,
)
from quant_platform.experiments import PairDataset


def test_classify_regimes_outputs_required_columns_and_labels():
    frame = pd.DataFrame(
        {
            "market_return": [0.01, 0.01, 0.01, -0.04, -0.05, 0.08, -0.12, 0.01],
            "spread": range(8),
        }
    )

    classified = classify_regimes(frame, RegimeConfig(lookback=2, trend_threshold=0.015, crisis_drawdown_threshold=0.05))

    assert {"classified_regime", "regime", "regime_rolling_return", "regime_rolling_volatility", "regime_drawdown"}.issubset(
        classified.columns
    )
    assert set(classified["regime"]).issubset({"bull", "bear", "range", "crisis"})
    assert "crisis" in set(classified["regime"])


def test_regime_distribution_and_dataset_report(tmp_path):
    frame = pd.DataFrame({"regime": ["range", "range", "bull"]})

    distribution = regime_distribution(frame)
    path = write_regime_dataset_report([PairDataset("ETH-BTC", frame)], tmp_path / "regimes.csv")

    assert distribution["observations"].sum() == 3
    written = pd.read_csv(path)
    assert set(written["regime"]) == {"range", "bull"}


def test_regime_pair_strategy_report_groups_per_pair_regime_strategy():
    results = pd.DataFrame(
        [
            {
                "pair": "ETH-BTC",
                "regime": "range",
                "strategy_id": 1,
                "strategy_name": "Classic",
                "family": "zscore",
                "status": "evaluated",
                "eligible": True,
                "profit_factor": 2.0,
                "sharpe": 1.4,
                "max_drawdown": 0.05,
                "trades": 120,
            },
            {
                "pair": "ETH-BTC",
                "regime": "range",
                "strategy_id": 1,
                "strategy_name": "Classic",
                "family": "zscore",
                "status": "evaluated",
                "eligible": False,
                "profit_factor": 1.0,
                "sharpe": 0.5,
                "max_drawdown": 0.20,
                "trades": 80,
            },
        ]
    )

    report = regime_pair_strategy_report(results)

    assert len(report) == 1
    assert report["runs"].iloc[0] == 2
    assert report["eligible_runs"].iloc[0] == 1
    assert report["total_trades"].iloc[0] == 200

