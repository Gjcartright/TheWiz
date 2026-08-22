from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_platform.orchestration.current_wizard_hyperliquid_math_comparison import (
    build_current_wizard_hyperliquid_math_comparison,
)


def test_math_comparison_requires_same_inputs_and_records_eligibility_flips(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    old_results = snapshot / "old.csv"
    new_results = snapshot / "new.csv"
    base = {
        "experiment_id": "exp-1",
        "pair_group_key": "dydx|daily|A|B",
        "pair": "A-B",
        "wizard_exchange": "dydx",
        "timeframe": "daily",
        "exact_mode": "Static (ZScoreR)",
        "orientation": "original",
        "replay_status": "RESEARCH_REPLAY_COMPLETE",
        "acceptance_status": "BLOCKED",
        "sharpe_status": "valid",
        "research_rank_blocker": "closed_trades<10",
        "engle_granger_pvalue": 0.01,
        "hedge_ratio": 1.2,
        "ou_phi": 0.9,
        "ou_half_life": 6.5,
        "trades": 9,
        "open_trades": 0,
        "profit_factor": 1.1,
        "expectancy": 0.01,
        "expectancy_lower_95": -0.01,
        "sharpe": 0.5,
        "max_drawdown": 0.2,
        "win_rate": 0.5,
        "total_return": 0.1,
        "gross_return": 0.12,
        "total_fees": 0.01,
        "total_slippage": 0.01,
        "total_funding": 0.0,
        "total_execution_risk": 0.0,
        "total_partial_fill_cost": 0.0,
        "avg_gross_exposure": 1.0,
        "reconciliation_error": 0.0,
        "closed_trade_return_std": 0.1,
    }
    pd.DataFrame([{**base, "research_rank_eligible": False}]).to_csv(old_results, index=False)
    pd.DataFrame(
        [
            {
                **base,
                "trades": 10,
                "sharpe": 0.4,
                "research_rank_eligible": True,
                "research_rank_blocker": "",
            }
        ]
    ).to_csv(new_results, index=False)
    input_hashes = {"experiment_matrix": "a", "pair_history_results": "b"}
    old_manifest = snapshot / "old.json"
    new_manifest = snapshot / "new.json"
    old_manifest.write_text(
        json.dumps(
            {
                "canonical_replay_id": "old",
                "settings_version": "v1",
                "input_hashes": input_hashes,
                "trade_ledger_rows": 9,
                "artifacts": {"snapshot_results": str(old_results.relative_to(tmp_path))},
            }
        ),
        encoding="utf-8",
    )
    new_manifest.write_text(
        json.dumps(
            {
                "canonical_replay_id": "new",
                "settings_version": "v2",
                "input_hashes": input_hashes,
                "trade_ledger_rows": 10,
                "artifacts": {"snapshot_results": str(new_results.relative_to(tmp_path))},
            }
        ),
        encoding="utf-8",
    )

    result = build_current_wizard_hyperliquid_math_comparison(
        root=tmp_path,
        old_manifest_path=old_manifest,
        new_manifest_path=new_manifest,
        output_stem="comparison",
    )

    assert result.summary["experiments_compared"] == 1
    assert result.summary["eligibility_gained"] == 1
    assert result.summary["rows_with_metric_changes"] == 1
    assert pd.read_csv(result.paths["eligibility_flips"]).shape[0] == 1
    assert pd.read_csv(result.paths["mode_summary"]).iloc[0]["exact_mode"] == "Static (ZScoreR)"
    assert pd.read_csv(result.paths["pair_summary"]).iloc[0]["pair_group_key"] == ("dydx|daily|A|B")
