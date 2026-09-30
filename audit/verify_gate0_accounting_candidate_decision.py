#!/usr/bin/env python3
"""Verify the preserved, dependency-blocked accounting source decision."""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30")
DIAGNOSTIC = Path(
    "/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30/"
    "accounting-candidate"
)
OUTPUT = AUDIT / "GATE0_ACCOUNTING_CANDIDATE_DECISION_2026-09-30.json"
DEFERRED = (
    "src/quant_platform/risk_metrics.py",
    "src/quant_platform/protective_exits.py",
    "tests/test_risk_metrics.py",
    "tests/test_accounting_repairs.py",
)
OPEN_DEPENDENCIES = (
    "src/quant_platform/economic_contract.py",
    "src/quant_platform/performance_math.py",
    "src/quant_platform/trade_ledger.py",
    "src/quant_platform/backtest.py",
    "tests/test_backtest.py",
    "tests/test_performance_math_v2.py",
)
EXPECTED_FAILURES = {
    "tests.test_dydx_candles::test_build_pair_history_from_5min_candles_namespaces_proxies_and_adds_math_v2",
    "tests.test_ml_filter::test_build_trade_filter_dataset_normalizes_timeframe_aliases",
    "tests.test_student_readiness::test_valid_student_dataset_passes_supervised_and_bandit_checks",
    "tests.test_teacher_adapters::test_adapter_emits_only_a_complete_seven_teacher_six_critic_context",
    "tests.test_teacher_adapters::test_adapter_clears_streams_when_context_is_incomplete",
    "tests.test_teacher_council::test_complete_math_v2_council_can_only_authorize_shadow_test",
    "tests.test_teacher_council::test_student_router_can_only_add_an_abstention",
    "tests.test_teacher_evidence_materializer::test_materializer_builds_seven_teachers_and_six_independent_critics",
    "tests.test_teacher_evidence_materializer::test_canonical_cycle_preserves_lineage_and_stays_fail_closed",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suite_result(path: Path) -> tuple[int, int, int, set[str]]:
    root = ET.parse(path).getroot()
    suite = next(root.iter("testsuite"))
    failures = {
        f"{case.attrib['classname']}::{case.attrib['name']}"
        for case in root.iter("testcase") if case.find("failure") is not None
    }
    return (
        int(suite.attrib["tests"]), int(suite.attrib["failures"]),
        int(suite.attrib["errors"]), failures,
    )


def main() -> None:
    with (AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        queue = {row["relative_path"]: row for row in csv.DictReader(stream)}
    saved = json.loads((SAVEPOINT / "SAVEPOINT_MANIFEST.json").read_text())
    saved_hashes = {
        item["path"]: item["sha256"]
        for item in saved["entries"] if item["type"] == "file"
    }
    hashes: dict[str, str] = {}
    for relative in DEFERRED:
        row = queue[relative]
        if (row["working_vs_runtime"] != "RUNTIME_ONLY"
            or row["custody_status"] != "REVIEWED_ACCOUNTING_CANDIDATE_DEFERRED_NO_PORT"
            or OUTPUT.name not in row["decision_evidence"]
            or (ROOT / relative).exists()):
            raise ValueError(f"accounting candidate unexpectedly active: {relative}")
        expected = row["runtime_sha256_at_freeze"]
        if (digest(RUNTIME / relative) != expected
            or digest(SAVEPOINT / "local_runtime" / relative) != expected
            or saved_hashes[f"local_runtime/{relative}"] != expected):
            raise ValueError(f"accounting candidate custody drift: {relative}")
        hashes[relative] = expected
    for relative in OPEN_DEPENDENCIES:
        if queue[relative]["custody_status"] != "PRESERVED_REVIEW_REQUIRED_NO_PORT":
            raise ValueError(f"dependent source was approved prematurely: {relative}")

    diagnostic_path = DIAGNOSTIC / "ACCOUNTING_CANDIDATE_DIAGNOSTIC.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if (diagnostic["active_base_commit"] != "0e44233b890eb217fb62861676272881480913f4"
        or diagnostic["disposition"] != "PRESERVE_COHERENT_CANDIDATE_NO_ACTIVE_PORT"
        or diagnostic["active_math_version"] != "math-v2.1-y-on-x"
        or diagnostic["candidate_math_version"] != "math-v2.2-validity-accounting"):
        raise ValueError("accounting diagnostic identity or version drift")
    source_receipt = json.loads((DIAGNOSTIC / "SOURCE_RECEIPT.json").read_text())
    if digest(DIAGNOSTIC / "SOURCE_RECEIPT.json") != diagnostic["source_receipt_sha256"]:
        raise ValueError("candidate source receipt drift")
    for relative, expected in source_receipt["runtime_source_hashes"].items():
        if (digest(RUNTIME / relative) != expected
            or digest(DIAGNOSTIC / "repo" / relative) != expected):
            raise ValueError(f"candidate source copy drift: {relative}")
    for name, expected in source_receipt["runtime_test_hashes"].items():
        original = "test_" + name.removeprefix("test_candidate_")
        if (digest(RUNTIME / "tests" / original) != expected
            or digest(DIAGNOSTIC / "repo" / "tests" / name) != expected):
            raise ValueError(f"candidate test copy drift: {name}")
    for label, expected in (
        ("coherent_focused", (167, 0, 0)),
        ("downstream_unadjusted", (220, 4, 0)),
        ("downstream_clock_skip", (220, 1, 0)),
        ("full_disposable_selection", (2509, 9, 0)),
    ):
        result = diagnostic["test_runs"][label]
        path = Path(result["path"])
        actual = suite_result(path)
        if (actual[:3] != expected or digest(path) != result["sha256"]
            or (result["tests"], result["failures"], result["errors"]) != expected):
            raise ValueError(f"candidate JUnit drift: {label}")
        if label == "full_disposable_selection" and actual[3] != EXPECTED_FAILURES:
            raise ValueError("full candidate failure set changed")
    patch = diagnostic["disposable_integration_adjustments_patch"]
    if digest(Path(patch["path"])) != patch["sha256"]:
        raise ValueError("disposable adjustment patch drift")
    for label, record in (
        ("profit_factor_basis_probe", diagnostic["profit_factor_basis_probe"]),
        ("boolean_input_probe", diagnostic["boolean_input_probe"]),
    ):
        pairs = [("script_path", "script_sha256")]
        if label == "profit_factor_basis_probe":
            pairs.extend((("active_path", "active_sha256"), ("runtime_path", "runtime_sha256")))
        else:
            pairs.append(("result_path", "result_sha256"))
        for path_key, hash_key in pairs:
            if digest(Path(record[path_key])) != record[hash_key]:
                raise ValueError(f"{label} evidence drift: {path_key}")
    profit = diagnostic["profit_factor_basis_probe"]
    if (profit["active_result"]["normalized_return_profit_factor"] != 2.0
        or profit["runtime_result"]["same_currency_profit_factor"] != 1.0
        or profit["active_result"]["portfolio_return"] != 0.0):
        raise ValueError("profit-factor basis probe changed")
    boolean = diagnostic["boolean_input_probe"]["result"]
    if any(value["status"] != "valid" for value in boolean.values()):
        raise ValueError("candidate boolean-input defect changed")

    report = {
        "schema_version": "thewiz.gate0.accounting_candidate_decision.v1",
        "decision": "PRESERVE_ACCOUNTING_CANDIDATE_AND_REPAIR_VERSIONED_INTEGRATION_BEFORE_PORT",
        "reviewed_runtime_only_path_sha256": dict(sorted(hashes.items())),
        "dependent_paths_still_pending": list(OPEN_DEPENDENCIES),
        "diagnostic_path": str(diagnostic_path),
        "diagnostic_sha256": digest(diagnostic_path),
        "coherent_focused_tests_passed": 167,
        "full_disposable_tests": 2509,
        "full_disposable_failures": 9,
        "observed_profit_factor_disagreement": {
            "entry_normalized_return_basis": 2.0,
            "same_currency_net_pnl_basis": 1.0,
        },
        "candidate_boolean_input_defect": True,
        "interpretation": "Preserve the four exact runtime-only paths without active adoption. The six-module accounting candidate changes Math V2.1 to V2.2; its full integration fails one funding-clock fixture and eight version-bound research/teacher checks. Its pure risk metrics also accept boolean inputs as financial values. Repair those contracts and requalify downstream evidence before selecting the candidate. No research acceptance or order authority follows.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS four_deferred_accounting_source_decisions")


if __name__ == "__main__":
    main()
