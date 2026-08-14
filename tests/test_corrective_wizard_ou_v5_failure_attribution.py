from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd

from quant_platform.orchestration.corrective_wizard_ou_v5_failure_attribution import (
    build_ou_v5_failure_attribution,
)


def test_v5_failure_attribution_is_immutable_and_cannot_register_successor(
    tmp_path: Path,
) -> None:
    _write_failed_v5(tmp_path)

    result = build_ou_v5_failure_attribution(
        root=tmp_path,
        now=datetime(2026, 8, 14, tzinfo=UTC),
    )
    detail = pd.read_csv(result.paths["attribution"])

    assert result.summary["status"] == "PASS_FAILURE_ATTRIBUTION_COMPLETE"
    assert result.summary["mode_cells"] == 8
    assert result.summary["independent_orientations"] == 4
    assert result.summary["failed_orientations"] == 4
    assert result.summary["transform_selector_orientations_passed"] == 3
    assert result.summary["trend_selector_orientations_passed"] == 3
    assert result.summary["profile_branch_selector_orientations_passed"] == 3
    assert result.summary["formula_kernel_orientations_passed"] == 3
    assert result.summary["v5_holdout_reuse_allowed"] is False
    assert result.summary["successor_design_automatic"] is False
    assert result.summary["successor_registration_automatic"] is False
    assert result.summary["candidate_promotion_authority"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    assert len(detail) == 4
    assert detail["duplicate_mode_cells"].eq(2).all()
    assert detail["raw_bindings_valid"].all()
    assert result.paths["immutable_attribution"].is_file()


def test_v5_failure_attribution_blocks_tampered_immutable_evaluation(
    tmp_path: Path,
) -> None:
    _write_failed_v5(tmp_path)
    status_path = tmp_path / "reports/active/wizard_ou_v5_holdout_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    (tmp_path / status["immutable_result_path"]).write_text(
        '{"tampered":true}\n',
        encoding="utf-8",
    )

    result = build_ou_v5_failure_attribution(root=tmp_path)

    assert result.summary["status"] == "BLOCKED"
    assert "ou_v5_immutable_result_hash_mismatch" in result.summary["blockers"]


def test_v5_failure_attribution_preserves_passing_orientations(
    tmp_path: Path,
) -> None:
    _write_failed_v5(tmp_path)
    evaluation_path = tmp_path / "reports/active/wizard_ou_v5_holdout_evaluation.csv"
    evaluation = pd.read_csv(evaluation_path)
    passing = evaluation["pair"].eq("SOL-BNB")
    for column in (
        "transform_selector_parity_passed",
        "trend_selector_parity_passed",
        "profile_branch_selector_parity_passed",
        "formula_parity_passed",
    ):
        evaluation.loc[passing, column] = True
    evaluation.loc[passing, "cell_status"] = "PASS"
    evaluation.to_csv(evaluation_path, index=False)
    _write_source_binding(tmp_path, evaluation)

    result = build_ou_v5_failure_attribution(root=tmp_path)
    detail = pd.read_csv(result.paths["attribution"])

    assert result.summary["status"] == "PASS_FAILURE_ATTRIBUTION_COMPLETE"
    assert result.summary["failed_orientations"] == 3
    assert detail["orientation_status"].value_counts().to_dict() == {"FAIL": 3, "PASS": 1}
    passing_orientation = detail.loc[detail["orientation_status"].eq("PASS")].iloc[0]
    assert passing_orientation["root_cause"] == "none"


def test_v5_failure_attribution_blocks_active_evaluation_summary_drift(
    tmp_path: Path,
) -> None:
    _write_failed_v5(tmp_path)
    evaluation_path = tmp_path / "reports/active/wizard_ou_v5_holdout_evaluation.csv"
    evaluation = pd.read_csv(evaluation_path)
    evaluation.loc[0, "cell_status"] = "PASS"
    evaluation.to_csv(evaluation_path, index=False)

    result = build_ou_v5_failure_attribution(root=tmp_path)

    assert result.summary["status"] == "BLOCKED"
    assert "ou_v5_source_immutable_summary_mismatch:passed_cells" in result.summary["blockers"]
    assert "ou_v5_source_active_summary_mismatch:passed_cells" in result.summary["blockers"]


def _write_failed_v5(root: Path) -> None:
    active = root / "reports/active"
    active.mkdir(parents=True)
    cases = [
        ("SOL-BNB", "SOL-BNB", "original", False, True, True, True),
        ("SOL-BNB", "BNB-SOL", "reverse", True, False, True, True),
        ("DOGE-FET", "DOGE-FET", "original", True, True, False, True),
        ("DOGE-FET", "FET-DOGE", "reverse", True, True, True, False),
    ]
    rows = []
    for group, pair, orientation, transform, trend, branch, formula in cases:
        for mode in ("OU (Spread)", "OU (ZScoreR)"):
            slug = f"{pair}_{orientation}_{mode}".replace(" ", "_").replace("/", "_")
            request = root / "data/raw/requests" / f"{slug}.json"
            response = root / "data/raw/responses" / f"{slug}.json"
            request.parent.mkdir(parents=True, exist_ok=True)
            response.parent.mkdir(parents=True, exist_ok=True)
            request.write_text('{"params":{}}\n', encoding="utf-8")
            response.write_text('{"history":{}}\n', encoding="utf-8")
            rows.append(
                {
                    "pair_group": group,
                    "pair": pair,
                    "interval": "1h",
                    "orientation": orientation,
                    "exact_mode": mode,
                    "cell_status": "FAIL",
                    "predicted_log_used": True,
                    "vendor_log_used": transform,
                    "transform_selector_parity_passed": transform,
                    "predicted_inc_trend": True,
                    "vendor_inc_trend": trend,
                    "trend_selector_parity_passed": trend,
                    "predicted_profile_branch": "zero_mean",
                    "inferred_vendor_profile_branch": ("zero_mean" if branch else "intercept"),
                    "profile_branch_selector_parity_passed": branch,
                    "formula_parity_passed": formula,
                    "request_path": str(request.relative_to(root)),
                    "request_sha256": sha256(request.read_bytes()).hexdigest(),
                    "response_path": str(response.relative_to(root)),
                    "response_sha256": sha256(response.read_bytes()).hexdigest(),
                }
            )
    evaluation = pd.DataFrame(rows)
    evaluation.to_csv(active / "wizard_ou_v5_holdout_evaluation.csv", index=False)
    _write_source_binding(root, evaluation)


def _write_source_binding(root: Path, evaluation: pd.DataFrame) -> None:
    active = root / "reports/active"
    immutable = root / "data/research/wizard_ou_v5_holdout_evaluations/result.json"
    immutable.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "FAIL",
        "result_id": "failed-v5",
        "required_cells": len(evaluation),
        "passed_cells": int(evaluation["cell_status"].eq("PASS").sum()),
        "failed_cells": int(evaluation["cell_status"].ne("PASS").sum()),
        "transform_selector_parity_passed_cells": int(
            evaluation["transform_selector_parity_passed"].sum()
        ),
        "trend_selector_parity_passed_cells": int(evaluation["trend_selector_parity_passed"].sum()),
        "profile_branch_selector_parity_passed_cells": int(
            evaluation["profile_branch_selector_parity_passed"].sum()
        ),
        "formula_parity_passed_cells": int(evaluation["formula_parity_passed"].sum()),
        "response_sha256s": sorted(evaluation["response_sha256"].astype(str)),
    }
    immutable.write_text(json.dumps(summary), encoding="utf-8")
    status = {
        **summary,
        "immutable_result_path": str(immutable.relative_to(root)),
        "immutable_result_sha256": sha256(immutable.read_bytes()).hexdigest(),
    }
    (active / "wizard_ou_v5_holdout_status.json").write_text(
        json.dumps(status),
        encoding="utf-8",
    )
