from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd

from quant_platform.orchestration import corrective_wizard_ou_v4_failure_attribution
from quant_platform.orchestration.corrective_wizard_ou_v4_failure_attribution import (
    build_ou_v4_failure_attribution,
)


def test_failure_attribution_collapses_modes_and_forbids_holdout_reuse(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_failed_holdout(tmp_path)

    def fake_formula(*, response, inc_trend, **_kwargs):
        branch = "zero_mean" if inc_trend else "intercept"
        preferred = response["preferred_branch"]
        passed = branch == preferred
        error = 1e-8 if passed else 0.1
        return {
            "formula_parity_passed": passed,
            "local_hedge_ratio": response["branch_beta"][branch],
            "hedge_ratio_abs_error": error,
            "spread_max_abs_error": error,
        }

    monkeypatch.setattr(
        corrective_wizard_ou_v4_failure_attribution,
        "_evaluate_formula",
        fake_formula,
    )
    result = build_ou_v4_failure_attribution(
        root=tmp_path,
        now=datetime(2026, 8, 13, tzinfo=UTC),
    )
    detail = pd.read_csv(result.paths["attribution"])

    assert result.summary["status"] == "PASS_FAILURE_ATTRIBUTION_COMPLETE"
    assert result.summary["mode_cells"] == 8
    assert result.summary["independent_orientations"] == 4
    assert result.summary["transform_selector_orientations_passed"] == 2
    assert result.summary["trend_selector_orientations_passed"] == 4
    assert result.summary["profile_branch_selector_orientations_passed"] == 2
    assert result.summary["formula_kernel_orientations_attributed"] == 4
    assert result.summary["v4_evidence_role"] == "derivation_only"
    assert result.summary["v4_holdout_reuse_allowed"] is False
    assert len(detail) == 4
    assert detail["duplicate_mode_cells"].eq(2).all()
    assert detail["holdout_reuse_allowed"].eq(False).all()  # noqa: E712
    assert result.summary["candidate_promotion_authority"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    assert Path(result.paths["immutable_attribution"]).is_file()


def test_failure_attribution_blocks_tampered_immutable_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_failed_holdout(tmp_path)
    immutable = tmp_path / "data/research/wizard_ou_v4_holdout_evaluations/result.json"
    immutable.write_text('{"tampered":true}\n', encoding="utf-8")
    monkeypatch.setattr(
        corrective_wizard_ou_v4_failure_attribution,
        "_evaluate_formula",
        lambda **_kwargs: {
            "formula_parity_passed": True,
            "local_hedge_ratio": 1.0,
            "hedge_ratio_abs_error": 0.0,
            "spread_max_abs_error": 0.0,
        },
    )

    result = build_ou_v4_failure_attribution(root=tmp_path)

    assert result.summary["status"] == "BLOCKED"
    assert "ou_v4_immutable_result_hash_mismatch" in result.summary["blockers"]


def _write_failed_holdout(root: Path) -> None:
    active = root / "reports/active"
    active.mkdir(parents=True)
    immutable = root / "data/research/wizard_ou_v4_holdout_evaluations/result.json"
    immutable.parent.mkdir(parents=True)
    immutable.write_text('{"result_id":"failed-v4"}\n', encoding="utf-8")
    status = {
        "status": "FAIL",
        "result_id": "failed-v4",
        "immutable_result_path": str(immutable.relative_to(root)),
        "immutable_result_sha256": sha256(immutable.read_bytes()).hexdigest(),
        "implementation_source_sha256": "a" * 64,
    }
    (active / "wizard_ou_v4_holdout_status.json").write_text(
        json.dumps(status), encoding="utf-8"
    )
    cases = [
        ("PAIR1", "A-B", "original", True, True, True, "intercept"),
        ("PAIR1", "B-A", "reverse", False, True, True, "zero_mean"),
        ("PAIR2", "C-D", "original", False, False, False, "zero_mean"),
        ("PAIR2", "D-C", "reverse", True, False, False, "intercept"),
    ]
    rows = []
    for pair_group, pair, orientation, transform_match, vendor_trend, local_trend, branch in cases:
        for mode in ("OU (Spread)", "OU (ZScoreR)"):
            slug = f"{pair}_{orientation}_{mode}".replace(" ", "_").replace("/", "_")
            request = root / "data/raw/requests" / f"{slug}.json"
            response = root / "data/raw/responses" / f"{slug}.json"
            request.parent.mkdir(parents=True, exist_ok=True)
            response.parent.mkdir(parents=True, exist_ok=True)
            request.write_text('{"params":{}}\n', encoding="utf-8")
            response.write_text(
                json.dumps(
                    {
                        "preferred_branch": branch,
                        "branch_beta": {"zero_mean": 2.0, "intercept": 0.5},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows.append(
                {
                    "pair_group": pair_group,
                    "pair": pair,
                    "interval": "1h",
                    "exact_mode": mode,
                    "orientation": orientation,
                    "vendor_log_used": True,
                    "local_predicted_log_used": transform_match,
                    "transform_selector_parity_passed": transform_match,
                    "vendor_inc_trend": vendor_trend,
                    "local_predicted_inc_trend": local_trend,
                    "trend_selector_parity_passed": vendor_trend == local_trend,
                    "vendor_hedge_ratio": 2.0 if branch == "zero_mean" else 0.5,
                    "request_path": str(request.relative_to(root)),
                    "request_sha256": sha256(request.read_bytes()).hexdigest(),
                    "response_path": str(response.relative_to(root)),
                    "response_sha256": sha256(response.read_bytes()).hexdigest(),
                }
            )
    pd.DataFrame(rows).to_csv(
        active / "wizard_ou_v4_holdout_evaluation.csv", index=False
    )
