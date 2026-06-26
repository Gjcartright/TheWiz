from __future__ import annotations

from pathlib import Path

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.rl.rl_backtest import run_rl_research


def train_ppo_research_policy(root: Path = ROOT, pair_id: str = "") -> CommandResult:
    """Run the research scaffold and record whether PPO dependencies are available."""
    result = run_rl_research(root=root, pair_id=pair_id)
    report = root / "reports" / "rl" / "rl_ppo_dependency_report.csv"
    try:
        import stable_baselines3  # noqa: F401

        status = "available"
        blocker = "ppo_training_still_blocked_until_rl_dataset_acceptance"
    except Exception:
        status = "missing"
        blocker = "missing_optional_rl_dependency:stable-baselines3"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "dependency,status,blocker,live_enabled\n"
        f"stable-baselines3,{status},{blocker},False\n",
        encoding="utf-8",
    )
    result.paths["ppo_dependency_report"] = report
    result.summary["ppo_dependency_status"] = status
    return result
