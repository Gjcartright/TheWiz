import pandas as pd

from quant_platform.research_quantization import quantize_family_matrix


def test_quantize_family_matrix_writes_ranked_outputs(tmp_path):
    separate = pd.DataFrame(
        [
            {
                "family": "regime",
                "best_strategy_id": 24,
                "best_strategy_name": "Regime Filtered Stat-Arb",
                "production_eligible": False,
                "preferred_eligible": False,
                "acceptance_reason": "passing_pairs<2",
                "preferred_reason": "median_sharpe<1.5",
                "passing_pairs": 1,
                "total_trades": 180,
                "median_profit_factor": 1.6,
                "median_sharpe": 0.8,
                "worst_drawdown": 0.12,
            },
            {
                "family": "zscore",
                "best_strategy_id": 1,
                "best_strategy_name": "Classic ZScore Mean Reversion",
                "production_eligible": False,
                "preferred_eligible": False,
                "acceptance_reason": "passing_pairs<2",
                "preferred_reason": "median_profit_factor<2.0",
                "passing_pairs": 0,
                "total_trades": 120,
                "median_profit_factor": 1.05,
                "median_sharpe": 0.1,
                "worst_drawdown": 0.22,
            },
        ]
    )
    combos = pd.DataFrame(
        [
            {
                "combo_name": "combo_2way_1",
                "combo_size": 2,
                "families": "mean_reversion;regime",
                "strategies": "Half-Life Optimized;Regime Filtered Stat-Arb",
                "strategy_ids": "11;24",
                "production_eligible": True,
                "preferred_eligible": True,
                "acceptance_reason": "",
                "preferred_reason": "",
                "passing_pairs": 2,
                "total_trades": 320,
                "median_profit_factor": 2.2,
                "median_sharpe": 1.7,
                "worst_drawdown": 0.08,
            }
        ]
    )
    separate.to_csv(tmp_path / "family_separate_summary.csv", index=False)
    combos.to_csv(tmp_path / "family_combo_summary.csv", index=False)

    paths = quantize_family_matrix(tmp_path, top_n=2)

    ranked = pd.read_csv(paths["ranked"])
    top = pd.read_csv(paths["top"])
    summary = pd.read_csv(paths["summary"])

    assert set(paths) == {"ranked", "top", "summary", "runbook"}
    assert ranked.iloc[0]["decision_bucket"] == "promote_now"
    assert ranked.iloc[0]["candidate_type"] == "combo"
    assert top.shape[0] == 2
    assert int(summary.loc[0, "promote_now"]) == 1
    assert int(summary.loc[0, "shadow_ready"]) + int(summary.loc[0, "watchlist"]) >= 1


def test_quantize_family_matrix_parses_string_booleans(tmp_path):
    separate = pd.DataFrame(
        [
            {
                "family": "regime",
                "best_strategy_id": 24,
                "best_strategy_name": "Regime Filtered Stat-Arb",
                "production_eligible": "False",
                "preferred_eligible": "False",
                "acceptance_reason": "passing_pairs<2",
                "preferred_reason": "not_production_eligible",
                "passing_pairs": 0,
                "total_trades": 144,
                "median_profit_factor": 2.4,
                "median_sharpe": 0.6,
                "worst_drawdown": 0.3,
            }
        ]
    )
    separate.to_csv(tmp_path / "family_separate_summary.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "family_combo_summary.csv", index=False)

    paths = quantize_family_matrix(tmp_path)
    ranked = pd.read_csv(paths["ranked"])
    summary = pd.read_csv(paths["summary"])

    assert ranked.loc[0, "decision_bucket"] == "watchlist"
    assert int(summary.loc[0, "promote_now"]) == 0
