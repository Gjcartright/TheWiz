from __future__ import annotations

from pathlib import Path

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.rl.rl_backtest import run_rl_research


def evaluate_research_policy(root: Path = ROOT, pair_id: str = "") -> CommandResult:
    return run_rl_research(root=root, pair_id=pair_id)
