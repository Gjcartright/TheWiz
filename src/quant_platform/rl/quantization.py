from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT


def export_rl_policy(root: Path = ROOT) -> CommandResult:
    reports = root / "reports" / "rl"
    models = root / "models" / "rl"
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    acceptance = _read_csv(reports / "rl_acceptance_report.csv")
    accepted = bool(not acceptance.empty and acceptance.get("accepted", pd.Series([False])).astype(bool).iloc[0])
    blocker = "" if accepted else "rl_acceptance_not_passed"
    export_report = {
        "accepted": accepted,
        "exported": False,
        "blocker": blocker or "onnx_export_not_implemented_for_unproven_policy",
        "policy_original": "models/rl/policy_original.zip",
        "policy_onnx": "models/rl/policy_onnx.onnx",
        "policy_int8": "models/rl/policy_int8.onnx",
        "feature_schema": "models/rl/feature_schema.json",
    }
    parity = pd.DataFrame(
        [
            {
                "accepted": False,
                "blocker": blocker or "missing_policy_artifacts",
                "original_policy": export_report["policy_original"],
                "onnx_policy": export_report["policy_onnx"],
                "int8_policy": export_report["policy_int8"],
                "decision_match_rate": 0.0,
            }
        ]
    )
    export_path = models / "export_report.json"
    parity_json = models / "parity_report.json"
    parity_csv = reports / "rl_quantization_parity.csv"
    export_path.write_text(json.dumps(export_report, indent=2, sort_keys=True), encoding="utf-8")
    parity_json.write_text(json.dumps(parity.iloc[0].to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    parity.to_csv(parity_csv, index=False)
    return CommandResult(
        paths={"export_report": export_path, "parity_report": parity_json, "parity_csv": parity_csv},
        summary=export_report,
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
