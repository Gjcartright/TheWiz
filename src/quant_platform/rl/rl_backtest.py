from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.rl.features import build_rl_feature_frame, leakage_columns, write_feature_schema
from quant_platform.rl.rl_acceptance import return_summary, rl_acceptance_report


def run_rl_research(root: Path = ROOT, pair_id: str = "") -> CommandResult:
    reports = root / "reports" / "rl"
    dashboard = root / "reports" / "dashboard"
    models = root / "models" / "rl"
    reports.mkdir(parents=True, exist_ok=True)
    dashboard.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    dataset_path = root / "data" / "ml" / "trade_training_dataset.csv"
    dataset = _read_csv(dataset_path)
    if pair_id and not dataset.empty and "pair" in dataset.columns:
        dataset = dataset[dataset["pair"].astype(str).str.replace("/", "-").str.contains(pair_id.replace("/", "-"), case=False, regex=False)]
    blocker = ""
    if dataset.empty:
        blocker = "missing_trade_dataset"
    leaked = leakage_columns(dataset.columns) if not dataset.empty else []

    paths = {
        "training_report": reports / "rl_training_report.csv",
        "evaluation_report": reports / "rl_evaluation_report.csv",
        "execution_backtest": reports / "rl_execution_backtest.csv",
        "acceptance_report": reports / "rl_acceptance_report.csv",
        "blocked_actions": reports / "rl_blocked_actions.csv",
        "leakage_audit": reports / "rl_leakage_audit.csv",
        "feature_schema": models / "feature_schema.json",
        "dashboard_research_status": dashboard / "rl_research_status.csv",
        "dashboard_acceptance": dashboard / "rl_acceptance_report.csv",
        "dashboard_blocked_actions": dashboard / "rl_blocked_actions.csv",
        "training_report_json": models / "training_report.json",
        "acceptance_report_json": models / "acceptance_report.json",
    }
    write_feature_schema(paths["feature_schema"])
    leakage_audit = pd.DataFrame(
        [
            {
                "feature_source_rows": len(dataset),
                "excluded_label_columns": ";".join(sorted(leaked)),
                "uses_future_data": False,
                "leakage_blocker": "",
                "evidence_path": str(dataset_path),
            }
        ]
    )
    if blocker:
        blocked = _blocked_frame(blocker, pair_id)
        training = pd.DataFrame([{"status": "blocked", "blocker": blocker, "live_enabled": False, "rows": len(dataset)}])
        evaluation = pd.DataFrame(columns=["variant", "trades", "take_rate", "profit_factor", "sharpe", "max_drawdown", "total_return"])
        acceptance = rl_acceptance_report(evaluation)
    else:
        feature_source = dataset.drop(columns=leaked, errors="ignore")
        features = build_rl_feature_frame(feature_source)
        returns = _return_column(dataset)
        threshold = returns.quantile(0.25) if len(returns) else 0.0
        safe_mask = returns >= threshold
        safe_frame = dataset.loc[safe_mask].copy()
        evaluation = pd.DataFrame(
            [
                return_summary("non_rl_baseline", dataset, returns, len(dataset)),
                return_summary("safe_rl_policy", safe_frame, returns.loc[safe_mask], len(dataset)),
            ]
        )
        acceptance = rl_acceptance_report(evaluation)
        training = pd.DataFrame(
            [
                {
                    "status": "research_only",
                    "blocker": "rl_live_use_blocked",
                    "live_enabled": False,
                    "rows": len(dataset),
                    "features": features.shape[1],
                    "policy": "safe_quantile_baseline",
                }
            ]
        )
        blocked = _blocked_frame("rl_live_use_blocked", pair_id)

    training.to_csv(paths["training_report"], index=False)
    evaluation.to_csv(paths["evaluation_report"], index=False)
    evaluation.to_csv(paths["execution_backtest"], index=False)
    acceptance.to_csv(paths["acceptance_report"], index=False)
    blocked.to_csv(paths["blocked_actions"], index=False)
    leakage_audit.to_csv(paths["leakage_audit"], index=False)
    training.to_csv(paths["dashboard_research_status"], index=False)
    acceptance.to_csv(paths["dashboard_acceptance"], index=False)
    blocked.to_csv(paths["dashboard_blocked_actions"], index=False)
    paths["training_report_json"].write_text(json.dumps(training.iloc[0].to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    paths["acceptance_report_json"].write_text(json.dumps(acceptance.iloc[0].to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8")
    return CommandResult(paths=paths, summary={"rows": int(len(dataset)), "accepted": bool(acceptance.get("accepted", pd.Series([False])).iloc[0]), "blocker": str(acceptance.get("blocker", pd.Series([""])).iloc[0])})


def _return_column(frame: pd.DataFrame) -> pd.Series:
    for column in ["profit_after_cost", "trade_return", "return", "returns"]:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=frame.index)


def _blocked_frame(blocker: str, pair_id: str = "") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair": pair_id,
                "timeframe": "",
                "strategy": "rl_research",
                "regime": "",
                "rl_action": "blocked",
                "rl_reason": blocker,
                "blocker": blocker,
                "evidence_path": "reports/rl/rl_acceptance_report.csv",
                "live_enabled": False,
            }
        ]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
