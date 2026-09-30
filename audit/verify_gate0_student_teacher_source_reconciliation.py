#!/usr/bin/env python3
"""Bind the student/teacher historical review to selected bytes and the full suite."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "6de2ee3016a8413030f87996a7153f91154cce1e"
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
OUTPUT = ROOT / "audit/GATE0_STUDENT_TEACHER_SOURCE_RECONCILIATION_2026-09-30.json"
JUNIT = Path(
    "/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30/"
    "student-teacher-reconciliation-junit.xml"
)
VARIANT_ROOT = Path("/Users/gregc/Backups/TheWiz/source-delta-evidence/2026-09-30/variants")
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")
SELECTED = {
    "src/quant_platform/orchestration/student_readiness.py": "0d5d71fea0079d757a20c09bc768d56d5a760485bf043590fdc0cfedc05e678b",
    "src/quant_platform/orchestration/teacher_evidence_materializer.py": "9d1199d65b95564370255d5ddef4ffe44dc725a210aa3a15a9e9e5996a4b095f",
    "tests/test_student_readiness.py": "e711ca46c2d9fc7107a1b09a2de53354e532dabff2790e6336251d984540aac4",
    "tests/test_teacher_council.py": "af32632581af98969faf198d71f6e2ae6268e256fc0f46d01c87840434d652e2",
    "tests/test_teacher_adapters.py": "ad0fc94a5cfe7b98ad75617a8a02c84e3cdf3efa34f757b51ae8633baec14e60",
    "tests/test_teacher_evidence_materializer.py": "c8e183f5fbdc192f3487079c54eaee48e30832abae4adbf5d18f15f071d57e84",
    "tests/test_experiments.py": "d279569e05570bc31e0c91d89b2ea4c47820c4ad1ec78c75c48586ef4f17e5a2",
    "audit/verify_gate0_dydx_ml_source_reconciliation.py": "ca803ad351efb0960b5606a865e9ebe7d17b8f90ff784a4126a1cb833afdb17b",
}
SAVEPOINTS = {
    "src/quant_platform/orchestration/student_readiness.py": "7544803621ca3371ec5db8f8fb766d0dfe8d84d4631d3df5b31ea0acd618f965",
    "tests/test_student_readiness.py": "f4849fa06f8690b43c0470c1d9f3436ca0b34d4d5999aaf0ba10d42b39e6af29",
    "tests/test_teacher_council.py": "50e4603195ce593423ea991f1c81bfe1f52128eb1f2b76665fa616c87d222158",
}
VARIANTS = {
    "src/quant_platform/orchestration/teacher_evidence_materializer.py": [
        "b6fb274cba5b4556960018b547b3f22878fe3ce0862b2d4ebd1ddbcd79d40188",
        "dba45be2f62ee814617620eca96e6e2c0cb435d0eaa12a6844511d034390bc78",
    ],
    "tests/test_teacher_evidence_materializer.py": [
        "2c2e1d1b9c0af859b7ae3946df23c4ddf4663407f2f1e410b4292fa780f5ef3a",
        "60196f5f0557ffaae905bb262fa6d93028ab3cdb2ab6af18024d4d2912f2ea85",
    ],
    "tests/test_experiments.py": [
        "0dbd0816630de56cc6e4fb414bc9007a4b9528026fc321929b8a7bc22637ffed",
    ],
}
EXPERIMENT_TEST_SUCCESSORS = {
    "test_acceptance_gate_requires_uncertainty_lifecycle_and_reconciliation":
        "tests/test_experiments.py:test_acceptance_gate_rejects_nonfinite_and_unreconciled_open_backtest",
    "test_harness_rejects_internal_strategy_feature_gaps_after_warmup":
        "tests/test_historical_strategy_authority.py:test_present_but_incomplete_zscore_is_skipped",
    "test_harness_rejects_present_but_incomplete_funding_columns":
        "tests/test_historical_strategy_authority.py:test_present_but_incomplete_funding_is_skipped",
    "test_strategy_acceptance_blocks_proxy_without_proven_authority":
        "tests/test_historical_strategy_authority.py:test_proxy_strategy_cannot_gain_production_authority_from_score",
    "test_strategy_acceptance_does_not_double_count_cost_scenario_trades":
        "tests/test_experiments.py:test_strategy_acceptance_counts_repriced_trades_once",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_names(path: Path) -> set[str]:
    return {
        node.name
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def main() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    for relative, expected in SELECTED.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError(f"selected source, test, or verifier drift: {relative}")
    for relative, expected in SAVEPOINTS.items():
        path = SAVEPOINT / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"local runtime savepoint drift: {relative}")
    for relative, hashes in VARIANTS.items():
        active = function_names(ROOT / relative)
        for expected in hashes:
            variant = VARIANT_ROOT / expected
            if not variant.is_file() or digest(variant) != expected:
                raise ValueError(f"preserved historical variant drift: {expected}")
            old_only = function_names(variant) - active
            if relative == "tests/test_experiments.py":
                if old_only != set(EXPERIMENT_TEST_SUCCESSORS):
                    raise ValueError("historical experiment test accounting changed")
            elif old_only:
                raise ValueError(f"historical functions unaccounted for: {relative}: {old_only}")
    for successor in EXPERIMENT_TEST_SUCCESSORS.values():
        relative, name = successor.split(":", 1)
        if name not in function_names(ROOT / relative):
            raise ValueError(f"historical experiment test successor missing: {successor}")

    suite = next(ET.parse(JUNIT).getroot().iter("testsuite"))
    counts = {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")}
    if counts != {"tests": 2604, "failures": 0, "errors": 0, "skipped": 0}:
        raise ValueError(f"isolated reconciliation suite is not green: {counts}")
    with QUEUE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 811 or len({row["relative_path"] for row in rows}) != 811:
        raise ValueError("union source queue changed size or contains duplicates")
    queue = {row["relative_path"]: row for row in rows}
    reviewed_paths = set(SELECTED) - {"audit/verify_gate0_dydx_ml_source_reconciliation.py"}
    for relative in reviewed_paths:
        row = queue[relative]
        if row["custody_status"] != "REVIEWED_STUDENT_TEACHER_HISTORICAL_VARIANTS":
            raise ValueError(f"reviewed path not closed: {relative}")
        if OUTPUT.name not in row["decision_evidence"]:
            raise ValueError(f"reviewed path lacks decision evidence: {relative}")
        if relative in VARIANTS and set(row["historical_variant_sha256"].split(";")) != set(VARIANTS[relative]):
            raise ValueError(f"historical variant set mismatch: {relative}")
    pending = sum(
        row["custody_status"] in {
            "PRESERVED_REVIEW_REQUIRED_NO_PORT",
            "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED",
        }
        for row in rows
    )
    reviewed = sum(row["custody_status"].startswith("REVIEWED_") for row in rows)
    if (pending, reviewed, len(rows) - pending - reviewed) != (660, 133, 18):
        raise ValueError("Gate 0 source queue accounting changed")
    report = {
        "schema_version": "thewiz.gate0.student_teacher_source_reconciliation.v1",
        "decision": "HISTORICAL_VARIANTS_RECONCILED_EXACT_COMMIT_VERIFICATION_REQUIRED",
        "base_commit": BASE,
        "selected_file_sha256": SELECTED,
        "local_runtime_savepoint_sha256": SAVEPOINTS,
        "preserved_historical_variant_sha256": VARIANTS,
        "historical_experiment_test_successors": EXPERIMENT_TEST_SUCCESSORS,
        "isolated_suite": {**counts, "junit_sha256": digest(JUNIT)},
        "review_decisions": [
            "require_explicit_logged_exploratory_behavior_policy_for_bandit_readiness",
            "reject_ambiguous_student_identifiers_timestamps_labels_and_provenance",
            "retain_v23_y_on_x_teacher_orientation_and_positive_hedge_ratio_gate",
            "retain_economic_tail_actions_profit_factor_blocker_and_atomic_teacher_outputs",
            "retain_global_scoped_publication_authority_for_teacher_adapter_tests",
            "block_prior_math_version_council_evidence_after_repair",
            "map_historical_experiment_assertions_to_current_contract_regressions",
        ],
        "union_source_queue": {
            "file_sha256": digest(QUEUE),
            "distinct_paths": len(rows),
            "reviewed_paths": reviewed,
            "nonsource_paths": len(rows) - pending - reviewed,
            "remaining_semantic_review_paths": pending,
        },
        "research_acceptance": "BLOCKED",
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS student_teacher_historical_source_reconciliation")


if __name__ == "__main__":
    main()
