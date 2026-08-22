from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_platform.performance_math import MATH_VERSION
from quant_platform.v2_math_diagnostic import build_v2_math_diagnostic

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_v2_math_diagnostic_records_contract_incidents_and_blocks_release(tmp_path: Path):
    proof_path = tmp_path / "fixtures" / "vendor_static_proofs.csv"
    proof_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "exact_mode": "Static (Spread)" if index % 2 == 0 else "Static (ZScoreR)",
                "vendor_formula_parity_status": "exact_reconstruction",
                "vendor_spread_max_abs_error": 0.0,
                "vendor_zscore_max_abs_error": 0.0,
                "vendor_zscore_roll_max_abs_error": 0.0,
                "vendor_spread_formula": "ols_y_on_x",
            }
            for index in range(10)
        ]
    ).to_csv(proof_path, index=False)

    result = build_v2_math_diagnostic(
        root=tmp_path,
        source_root=REPOSITORY_ROOT,
        vendor_static_proof_path=proof_path,
    )

    assert result.summary["math_version"] == MATH_VERSION
    assert result.summary["local_core_status"] == "PASS"
    assert result.summary["wizard_static_reconstruction"] == "PASS_DIAGNOSTIC_ONLY"
    assert result.summary["release_decision"] == "BLOCKED_PENDING_CONTROLLED_REGENERATION"
    assert result.summary["paper_or_live_authority"] is False

    checks = pd.read_csv(result.paths["diagnostic_checks"])
    local = checks[checks["authority_domain"].isin(["local_core", "runtime", "source_contract"])]
    assert local["status"].eq("PASS").all()
    assert (
        checks.loc[checks["check_id"].eq("WIZARD_STATIC_EXACT_RECONSTRUCTION"), "status"]
        .eq("PASS")
        .all()
    )
    assert (
        checks.loc[checks["check_id"].eq("WIZARD_OU_REMAINS_BLOCKED"), "status"]
        .eq("EXPECTED_BLOCK")
        .all()
    )

    incidents = pd.read_csv(result.paths["error_ledger"])
    orientation = incidents.loc[incidents["incident_id"].eq("MATH-001")].iloc[0]
    assert orientation["status"] == "FIXED_THIS_AUDIT"
    assert "log(X) on log(Y)" in orientation["wrong_or_inconsistent_math"]
    assert "MATH-021" in set(incidents["incident_id"])
    assert {"MATH-023", "MATH-024"}.issubset(set(incidents["incident_id"]))

    inventory = pd.read_csv(result.paths["formula_inventory"])
    assert {
        "LOCAL_EG_Y_ON_X",
        "LOCAL_EG_RESIDUAL",
        "LOCAL_TWO_LEG_WEIGHTS",
        "LOCAL_INGESTED_SPREAD",
        "MODE_DYNAMIC_ZSCORER",
        "MODE_COPULA",
        "WIZARD_STATIC_OBSERVED",
        "WIZARD_BACKTEST_ACCOUNTING_UNKNOWN",
    }.issubset(set(inventory["formula_id"]))

    comparison = pd.read_csv(result.paths["math_comparison"])
    assert {
        "pair_orientation_and_hedge_ratio",
        "dynamic_zscorer",
        "ou_spread",
        "copula",
        "ml_rl_student_labels",
    }.issubset(set(comparison["component"]))
    assert comparison["current_authority"].ne("live").all()

    assert {
        "LEGACY_INGESTION_SPREAD_REMOVED",
        "TWO_LEG_DISPATCH_REQUIRES_HEDGE_RATIO",
        "SEVEN_MODE_SHARED_ECONOMIC_CONTRACT",
    }.issubset(set(checks["check_id"]))

    authority = json.loads(result.paths["authority"].read_text(encoding="utf-8"))
    assert authority["promotion_authority"] is False
    assert authority["live_trading_authorized"] is False
    assert authority["artifacts"]["report"]["sha256"]


def test_v2_math_diagnostic_missing_vendor_proof_blocks_only_vendor_domain(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    for relative in (
        "src/quant_platform/orchestration/current_wizard_hyperliquid_replay.py",
        "src/quant_platform/orchestration/teacher_evidence_materializer.py",
        "src/quant_platform/orchestration/hyperliquid_research_validation.py",
        "src/quant_platform/backtest.py",
    ):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")

    result = build_v2_math_diagnostic(root=tmp_path / "output", source_root=source)
    checks = pd.read_csv(result.paths["diagnostic_checks"])

    static = checks.loc[checks["check_id"].eq("WIZARD_STATIC_EXACT_RECONSTRUCTION")].iloc[0]
    assert static["status"] == "BLOCKED"
    assert static["observed"] == "proof file missing"
    assert result.summary["wizard_static_reconstruction"] == "BLOCKED"
