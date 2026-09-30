#!/usr/bin/env python3
"""Verify the evidence-blocked Math Desk source selection for Gate 0."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")
DIAGNOSTICS = Path("/Users/gregc/Backups/TheWiz/mathdesk-diagnostics/2026-09-30")
AVAILABILITY = AUDIT / "MATH_DESK_SOURCE_AVAILABILITY_2026-09-30.json"
OUTPUT = AUDIT / "GATE0_MATH_DESK_DECISION_2026-09-30.json"
MODULE_PREFIX = "src/quant_platform/math_desk/"
TEST_PREFIX = "tests/test_math_desk"
RELATED_TEST = "tests/test_node_contract_repairs.py"
RELATED_JUNIT = Path(
    "/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30/"
    "node-contract-candidate-junit.xml"
)
STATUS = "REVIEWED_GOVERNED_EVIDENCE_BLOCKED_MATH_DESK_NO_PORT"
REQUIRED_PATHS = (
    "docs/math_desk_constitution.md",
    "docs/math_desk_m3_core_contract.md",
    "reports/plans/2026-08-22_math_desk_m0_m2_implementation_plan.md",
    "reports/plans/2026-08-23_math_desk_10_of_10_corrective_plan.md",
    "reports/plans/2026-08-23_math_desk_m3_core_implementation_plan.md",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suite(path: Path) -> dict[str, str]:
    return next(ET.parse(path).getroot().iter("testsuite")).attrib


def main() -> None:
    with (AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        all_rows = list(csv.DictReader(stream))
        queue = [
            row for row in all_rows
            if row["relative_path"].startswith((MODULE_PREFIX, TEST_PREFIX))
        ]
    related = next(row for row in all_rows if row["relative_path"] == RELATED_TEST)
    modules = [row for row in queue if row["relative_path"].startswith(MODULE_PREFIX)]
    tests = [row for row in queue if row["relative_path"].startswith(TEST_PREFIX)]
    if (
        len(queue) != 38
        or len(modules) != 27
        or len(tests) != 11
        or len({row["relative_path"] for row in queue}) != 38
    ):
        raise ValueError("Math Desk source/test queue changed")
    manifest = json.loads(
        (SAVEPOINT.parent / "SAVEPOINT_MANIFEST.json").read_text(encoding="utf-8")
    )
    saved_files = {
        entry["path"]: entry["sha256"]
        for entry in manifest["entries"]
        if entry["type"] == "file"
    }
    hashes = {}
    for row in (*queue, related):
        relative = row["relative_path"]
        if (
            row["custody_status"] != STATUS
            or row["working_vs_runtime"] != "RUNTIME_ONLY"
            or row["historical_variant_count"] != "0"
        ):
            raise ValueError(f"Math Desk row status changed: {relative}")
        if (ROOT / relative).exists():
            raise ValueError(f"unselected Math Desk source became active: {relative}")
        candidate, saved = RUNTIME / relative, SAVEPOINT / relative
        candidate_hash = digest(candidate)
        if (
            candidate_hash != row["runtime_sha256_at_freeze"]
            or candidate_hash != digest(saved)
            or saved_files.get(f"local_runtime/{relative}") != candidate_hash
        ):
            raise ValueError(f"Math Desk custody drift: {relative}")
        hashes[relative] = candidate_hash

    availability = json.loads(AVAILABILITY.read_text(encoding="utf-8"))
    if (
        availability.get("conclusion")
        != "UNAVAILABLE_IN_ACCESSIBLE_GATE0_SOURCE_NOT_PROVEN_DELETED"
        or tuple(availability.get("required_relative_paths", ())) != REQUIRED_PATHS
        or availability.get("private_nightly_backup_tree_truncated") is not False
        or availability.get("private_nightly_backup_matching_paths")
    ):
        raise ValueError("governed-source availability result changed")
    for source_root, found in availability["known_source_roots_checked"].items():
        if found:
            raise ValueError(f"recorded root contains required source: {source_root}")
        if any((Path(source_root) / relative).exists() for relative in REQUIRED_PATHS):
            raise ValueError(f"required Math Desk source appeared: {source_root}")
    if availability["savepoint_manifest_matching_paths"]:
        raise ValueError("Math Desk source appeared in savepoint manifest")
    if any(Path(entry["path"]).name in {Path(x).name for x in REQUIRED_PATHS}
           for entry in manifest["entries"]):
        raise ValueError("current savepoint manifest contains required source")
    objects = subprocess.run(
        ["git", "rev-list", "--all", "--objects"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if any(
        line.endswith("/" + Path(relative).name)
        for line in objects.splitlines()
        for relative in REQUIRED_PATHS
    ):
        raise ValueError("required Math Desk source appeared in reachable Git objects")
    if availability["reachable_source_git_matching_paths"]:
        raise ValueError("recorded Git object search changed")

    baseline = suite(DIAGNOSTICS / "mathdesk-junit.xml")
    followup = suite(DIAGNOSTICS / "risk-census-junit.xml")
    node_contract = suite(RELATED_JUNIT)
    if any(
        baseline[key] != value
        for key, value in (("tests", "174"), ("failures", "18"), ("errors", "67"))
    ):
        raise ValueError("Math Desk baseline outcome changed")
    if any(
        followup[key] != value
        for key, value in (("tests", "12"), ("failures", "0"), ("errors", "4"))
    ):
        raise ValueError("Math Desk risk-census outcome changed")
    if any(
        node_contract[key] != value
        for key, value in (("tests", "57"), ("failures", "0"), ("errors", "0"))
    ):
        raise ValueError("Math Desk related node-contract test outcome changed")
    diagnostic = json.loads(
        (AUDIT / "MATH_DESK_DIAGNOSTIC_2026-09-30.json").read_text(
            encoding="utf-8"
        )
    )
    if set(diagnostic["lecture_index_missing_fields"]) != {
        "review_status", "topic_tags", "transcript_text_stored"
    }:
        raise ValueError("Math Desk lecture-index schema gap changed")
    if "math-research = [" not in (RUNTIME / "pyproject.toml").read_text(
        encoding="utf-8"
    ) or "math-research = [" in (ROOT / "pyproject.toml").read_text(
        encoding="utf-8"
    ):
        raise ValueError("Math Desk optional dependency-profile difference changed")
    summary = {
        "schema_version": "thewiz.gate0.math_desk_decision.v1",
        "decision": STATUS,
        "path_sha256": dict(sorted(hashes.items())),
        "candidate_module_count": len(modules),
        "candidate_test_file_count": len(tests),
        "related_node_contract_test_file_count": 1,
        "related_node_contract_junit_sha256": digest(RELATED_JUNIT),
        "related_node_contract_tests_passed": 57,
        "source_availability_sha256": digest(AVAILABILITY),
        "required_governed_paths": list(REQUIRED_PATHS),
        "baseline_junit_sha256": digest(DIAGNOSTICS / "mathdesk-junit.xml"),
        "baseline_tests": 174,
        "baseline_passed": 89,
        "baseline_failed": 18,
        "baseline_errors": 67,
        "risk_census_junit_sha256": digest(DIAGNOSTICS / "risk-census-junit.xml"),
        "risk_census_tests_passed": 8,
        "risk_census_errors": 4,
        "lecture_index_missing_fields": diagnostic["lecture_index_missing_fields"],
        "interpretation": "Retain all 27 exact Math Desk modules, 11 matching test files, and one related pure node-contract metadata test as a preserved, unselected candidate for Gate 0. The related test passes 57 cases. Five governed source documents are unavailable in accessible source locations but are not proven deleted; the optional dependency profile and lecture-index schema also differ from active source. Existing focused passes do not establish algorithm acceptance, MATLAB parity, strategy promotion, or trading authority. Revisit only with independently sourced governed evidence and a coherent dependency/data contract.",
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS thirty_nine_math_desk_source_and_test_decisions")


if __name__ == "__main__":
    main()
