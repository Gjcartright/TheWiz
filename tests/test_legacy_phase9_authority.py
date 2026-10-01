"""The historical Phase 9 report must never grant current trading authority."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_evidence_pipeline_phase9_final_promotion.py"


def load_script():
    spec = importlib.util.spec_from_file_location("legacy_phase9", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_historical_winning_row_is_research_only() -> None:
    module = load_script()
    label, reason, next_action = module.final_label(
        "ETH-SOL",
        pd.Series({"walk_forward_pass": True}),
        stress_pf=1.6,
        stress_pass=True,
    )
    assert module.LEGACY_PIPELINE_AUTHORITY == "HISTORICAL_RESEARCH_ONLY"
    assert label == "research_only"
    assert "no acceptance or paper-trade authority" in reason
    assert "canonical V2" in next_action


def test_historical_report_cannot_emit_promotion_authority(tmp_path: Path) -> None:
    module = load_script()
    module.REPORTS = tmp_path
    pd.DataFrame(
        [
            {
                "pair": "ETH-SOL",
                "phase8_rank": 1,
                "strategy": "static",
                "timeframe": "1h",
                "regime": "all",
                "entry_style": "zscore",
                "exit_style": "mean",
                "trades": 100,
                "profit_factor": 2.0,
                "sharpe": 2.0,
                "max_drawdown": 0.1,
                "test_profit_factor": 1.5,
                "walk_forward_pass": True,
            }
        ]
    ).to_csv(tmp_path / "phase8_pair_specific_ranked.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "ETH-SOL",
                "cost_bucket": "high",
                "profit_factor": 1.6,
                "stress_pass": True,
            }
        ]
    ).to_csv(tmp_path / "phase7_cost_stress_results.csv", index=False)

    module.main()

    labels = pd.read_csv(tmp_path / "phase9_final_promotion_labels.csv")
    assert set(labels["final_label"]) <= {"watchlist", "research_only", "reject"}
    assert labels.loc[labels["pair"].eq("ETH-SOL"), "final_label"].item() == "research_only"
    assert not labels["promotion_authority"].any()
    report = (tmp_path / "phase9_final_promotion_report.md").read_text()
    assert "no promotion, paper-trade, or execution authority" in report
