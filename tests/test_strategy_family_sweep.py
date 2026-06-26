from __future__ import annotations

import pandas as pd

from quant_platform import cli
from quant_platform.experiments import PairDataset


def _dataset(pair: str, sign: float) -> PairDataset:
    rows = []
    for idx in range(80):
        z = 2.2 * sign if idx % 20 == 0 else (0.0 if idx % 20 == 10 else 0.3 * sign)
        rows.append(
            {
                "timestamp": f"2026-01-01T{idx:02d}:00:00Z",
                "pair": pair,
                "spread": (0.05 * idx) * sign,
                "zscore": z,
                "price_x": 100.0 + idx,
                "price_y": 50.0 + idx * 0.5,
                "hedge_ratio": 1.0,
                "beta": 1.0,
                "funding_x_bps": 0.1,
                "funding_y_bps": 0.1,
            }
        )
    return PairDataset(pair, pd.DataFrame(rows))


def test_strategy_family_sweep_report_writes_expected_outputs(tmp_path, monkeypatch):
    datasets = [
        _dataset("BTC-USD-SOL-USD", 1.0),
        _dataset("DOGE-USD-SOL-USD", -1.0),
        _dataset("SOL-USD-XRP-USD", 1.0),
        _dataset("SOL-USD-LINK-USD", -1.0),
    ]
    monkeypatch.setattr(cli, "datasets_from_pair_detail_snapshots", lambda input_dir, require_research_usable=True: datasets)
    monkeypatch.setattr(cli, "_enrich_datasets_with_funding", lambda ds, funding_path: ds)

    output_dir = tmp_path / "family_sweep"
    paths = cli.strategy_family_sweep_report(input_dir=tmp_path, output_dir=output_dir)

    for path in paths.values():
        assert path.exists()

    summary = pd.read_csv(paths["summary"])
    assert {"strategy_name", "family", "passing_pairs", "median_sharpe"}.issubset(summary.columns)

    ranked = pd.read_csv(paths["ranked"])
    assert {"strategy", "family", "pairs_tested", "production_eligible"}.issubset(ranked.columns)

    best = pd.read_csv(paths["best_by_family"])
    assert "strategy_name" in best.columns

    shortlist = pd.read_csv(paths["promotion_shortlist"])
    assert "promotion_reason" in shortlist.columns

    failure = pd.read_csv(paths["failure_attribution"])
    assert {"family", "diagnosis", "top_blockers"}.issubset(failure.columns)


def test_family_failure_attribution_report_counts_gate_blockers(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    sweep_dir = tmp_path / "reports" / "strategy_family_sweep"
    sweep_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "strategy_name": "Classic ZScore Mean Reversion",
                "family": "zscore",
                "evaluated_runs": 40,
                "passing_pairs": 0,
                "total_trades": 12,
                "median_profit_factor": 0.9,
                "median_sharpe": -0.2,
                "worst_drawdown": 0.4,
                "acceptance_reason": "passing_pairs<2",
                "preferred_reason": "not_production_eligible;median_profit_factor<2.0;median_sharpe<1.5;worst_drawdown>0.1;total_trades<250",
            },
            {
                "strategy_name": "Dynamic Threshold Model",
                "family": "zscore",
                "evaluated_runs": 40,
                "passing_pairs": 0,
                "total_trades": 8,
                "median_profit_factor": 0.8,
                "median_sharpe": -0.3,
                "worst_drawdown": 0.5,
                "acceptance_reason": "passing_pairs<2",
                "preferred_reason": "not_production_eligible;median_profit_factor<2.0;median_sharpe<1.5;worst_drawdown>0.1;total_trades<250",
            },
        ]
    ).to_csv(sweep_dir / "strategy_family_sweep_summary.csv", index=False)

    frame = cli.family_failure_attribution_report(sweep_dir)

    row = frame.iloc[0]
    assert row["family"] == "zscore"
    assert row["diagnosis"] == "no_passing_pairs"
    assert row["strategies_blocked_by_passing_pairs"] == 2
    assert row["strategies_blocked_by_total_trades"] == 2
