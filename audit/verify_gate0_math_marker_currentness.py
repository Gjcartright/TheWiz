#!/usr/bin/env python3
"""Verify the bound math marker repair and its source reconciliation evidence."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "6cba715163709f82fd49aa9f49afd4b7b8e0abde"
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
OUTPUT = ROOT / "audit/GATE0_MATH_MARKER_CURRENTNESS_2026-09-30.json"
EVIDENCE = Path("/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30")
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")
VARIANTS = Path("/Users/gregc/Backups/TheWiz/source-delta-evidence/2026-09-30/variants")
SELECTED = {
    "audit/verify_gate0_teacher_numeric_guards.py": "b11422e2c2868de8cc9bf8c585daa9eb7ad46c8da6b37b485e4104de502df4a0",
    "src/quant_platform/math_v2_acceptance.py": "a08855cebdab2e101f8211404c3a01d208bb57a31f786e4294a689ae639f9551",
    "src/quant_platform/orchestration/math_acceptance_currentness.py": "b56fe4c6866cbc3a4b5d18691a9b5464c9e45d5fabd14be5c4956e32fde904a8",
    "src/quant_platform/orchestration/teacher_adapters.py": "41bdf698a655c837c9ce5fdcb496dc7ca8e79de7febbed859559ce24117ca63c",
    "src/quant_platform/orchestration/teacher_control_plane.py": "c86fbf75bb72b586da496de8ba86172b69712fd53389e4f376f99445d5b313e5",
    "tests/test_math_acceptance_currentness.py": "6d6b47a9e4a9d561bd0c214a492dddf3bba1bb343e75a16edfe3ed66a48430ca",
    "tests/test_math_v2_acceptance.py": "9599313e1ede12b2aae4a670b10c112987d65d1782e5c39a26ec5b268b11ef1e",
    "tests/test_teacher_control_plane.py": "0a31f36c5142a2891ec8235adf4636fb972a96ffa57a42dc521e80f68426dd73",
}
SAVEPOINTS = {
    "src/quant_platform/math_v2_acceptance.py": "a01ef2e6e2d81794ed6b37bd71b02b05378d38c983328f8a49929963c8c135f1",
    "src/quant_platform/orchestration/math_acceptance_currentness.py": "ae4f18066ac63f764ca7c662856396387708cd7c783760131011a533c992b37b",
    "src/quant_platform/orchestration/teacher_adapters.py": "2155c3a7e956148da1a487e406519a34dfaca47369f79f3fa25dc70a3a425a2d",
    "src/quant_platform/orchestration/teacher_control_plane.py": "ab8a5682d5cdb0290a05479d66f61e8cb46ec89112980a1cf5717e6fde59ca42",
    "tests/test_math_v2_acceptance.py": "1bd29afbeff2a82c56947146201ecaf505cb12187c2cf32a878ba2efb1a5ba9a",
    "tests/test_teacher_control_plane.py": "1cf55a7c3ab9e834db9433091c0f81f35f9521b4bc67b4c934e043d61094814b",
}
HISTORICAL = {
    "src/quant_platform/math_v2_acceptance.py": "c32dfe8409c0b299f4b8a38d0dcdefb3f79e88b05a91b8c445a473e760c4d090",
    "src/quant_platform/orchestration/teacher_control_plane.py": "b99eba103e1746522e69e707707a39cb35e5e10a7e5fb155d7de22ac58ac0ecc",
}
REVIEWED = frozenset({
    "src/quant_platform/math_v2_acceptance.py",
    "src/quant_platform/orchestration/math_acceptance_currentness.py",
    "src/quant_platform/orchestration/teacher_adapters.py",
    "src/quant_platform/orchestration/teacher_control_plane.py",
    "tests/test_math_v2_acceptance.py",
    "tests/test_teacher_control_plane.py",
})
BASELINE_SHA256 = "9fcbaed895be03793e6d5395ee9ea1aa164cf1c67797fb3ee04f62b9294f7fe1"
WHEEL_SHA256 = "6d3d8d2a3fb62c3dedf1f610be730f785c8ff7d693485378b27d179002f89bf4"
JUNIT_SHA256 = "335250b6a66269903bb8ea61a6d11dc8c28768d16a225c6c7f094666bc14c1f8"
QUEUE_SHA256 = "64c8ff2a572ccc9e749c6a06a5a353c93ee631f937027990509ddcab69e4780e"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def top_level_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def main() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    for relative, expected in SELECTED.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError(f"selected source, test, or verifier drift: {relative}")
    for relative, expected in SAVEPOINTS.items():
        path = SAVEPOINT / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"preserved runtime savepoint drift: {relative}")
    for relative, expected in HISTORICAL.items():
        variant = VARIANTS / expected
        if not variant.is_file() or digest(variant) != expected:
            raise ValueError(f"preserved forensic variant drift: {relative}")
        if top_level_functions(variant) - top_level_functions(ROOT / relative):
            raise ValueError(f"historical function unaccounted for: {relative}")
    for relative in ("tests/test_math_v2_acceptance.py", "tests/test_teacher_control_plane.py"):
        old_only = top_level_functions(SAVEPOINT / relative) - top_level_functions(ROOT / relative)
        expected = {"_scoped_research_publication"} if relative.endswith("test_teacher_control_plane.py") else set()
        if old_only != expected:
            raise ValueError(f"historical test unaccounted for: {relative}: {old_only}")

    baseline_path = EVIDENCE / "math-marker-baseline.json"
    if digest(baseline_path) != BASELINE_SHA256:
        raise ValueError("baseline marker probe drift")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("source_commit") != BASE or not all(
        baseline.get(key) is True for key in (
            "adapter_accepts_marker_without_reports_or_source_binding",
            "control_plane_accepts_marker_without_reports_or_source_binding",
        )
    ) or baseline.get("order_authority_granted") is not False:
        raise ValueError("baseline false-positive observation changed")

    wheel_path = EVIDENCE / "math-marker-wheel-probe/QUALIFICATION.json"
    if digest(wheel_path) != WHEEL_SHA256:
        raise ValueError("installed-wheel qualification drift")
    wheel = json.loads(wheel_path.read_text(encoding="utf-8"))
    if (wheel.get("source_binding_count") != 9
            or wheel.get("bound_marker_currentness") != "PASS"
            or wheel.get("publication_authority") != "SCOPED_SYNTHETIC_RESEARCH_ONLY"
            or wheel.get("network_or_order_effects_allowed") is not False
            or wheel.get("source_hashes", {}).get("src/quant_platform/math_v2_acceptance.py") != SELECTED["src/quant_platform/math_v2_acceptance.py"]
            or wheel.get("source_hashes", {}).get("src/quant_platform/orchestration/math_acceptance_currentness.py") != SELECTED["src/quant_platform/orchestration/math_acceptance_currentness.py"]):
        raise ValueError("installed-wheel source marker qualification changed")

    junit_path = EVIDENCE / "math-marker-currentness-final-junit.xml"
    if digest(junit_path) != JUNIT_SHA256:
        raise ValueError("isolated full-suite evidence drift")
    suite = next(ET.parse(junit_path).getroot().iter("testsuite"))
    counts = {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")}
    if counts != {"tests": 2619, "failures": 0, "errors": 0, "skipped": 0}:
        raise ValueError(f"isolated full suite is not green: {counts}")

    if digest(QUEUE) != QUEUE_SHA256:
        raise ValueError("union source queue drift")
    with QUEUE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 811 or len({row["relative_path"] for row in rows}) != 811:
        raise ValueError("union source queue size or uniqueness changed")
    queue = {row["relative_path"]: row for row in rows}
    for relative in REVIEWED:
        row = queue[relative]
        if (row["custody_status"] != "REVIEWED_MATH_MARKER_CURRENTNESS"
                or OUTPUT.name not in row["decision_evidence"]):
            raise ValueError(f"math marker source decision missing: {relative}")
    pending = sum(row["custody_status"] in {
        "PRESERVED_REVIEW_REQUIRED_NO_PORT",
        "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED",
    } for row in rows)
    reviewed = sum(row["custody_status"].startswith("REVIEWED_") for row in rows)
    if (pending, reviewed, len(rows) - pending - reviewed) != (652, 141, 18):
        raise ValueError("Gate 0 source queue accounting changed")

    report = {
        "schema_version": "thewiz.gate0.math_marker_currentness.v1",
        "decision": "BOUND_MATH_MARKER_SOURCE_REPORT_AND_CONSUMER_CONTRACT_EXACT_COMMIT_VERIFICATION_REQUIRED",
        "base_commit": BASE,
        "selected_file_sha256": SELECTED,
        "local_runtime_savepoint_sha256": SAVEPOINTS,
        "preserved_historical_variant_sha256": HISTORICAL,
        "baseline_probe_sha256": BASELINE_SHA256,
        "baseline_false_positives": {
            "adapter_accepts_unbound_marker": baseline["adapter_accepts_marker_without_reports_or_source_binding"],
            "control_plane_accepts_unbound_marker": baseline["control_plane_accepts_marker_without_reports_or_source_binding"],
        },
        "installed_wheel_probe_sha256": WHEEL_SHA256,
        "isolated_suite": {**counts, "junit_sha256": JUNIT_SHA256},
        "review_decisions": [
            "reject_unbound_self_reported_math_markers_in_both_teacher_consumers",
            "bind_math_marker_to_loaded_package_source_and_exact_report_checks",
            "require_scoped_publication_authority_for_acceptance_reports",
            "retain_current_v23_math_checks_and_atomic_publication",
            "account_for_historical_per_file_test_publication_fixtures",
        ],
        "union_source_queue": {
            "file_sha256": QUEUE_SHA256,
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
    print("PASS math_marker_currentness_source_reconciliation")


if __name__ == "__main__":
    main()
