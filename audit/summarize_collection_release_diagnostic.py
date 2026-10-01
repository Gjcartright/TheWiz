"""Summarize isolated collection-release tests and process import-order probe."""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
ISOLATED = Path("/tmp/thewiz-runtime-mathdesk.G5oKC2")
DIAGNOSTIC = Path("/Users/gregc/Backups/TheWiz/collection-release-diagnostics/2026-09-30")
PACKAGE = DIAGNOSTIC / "collection-release-junit.xml"
INTEGRATION = DIAGNOSTIC / "integration-junit.xml"
ALONE = DIAGNOSTIC / "process-unittest-stdout.txt"
OUTPUT = AUDIT / "COLLECTION_RELEASE_NONPASS_2026-09-30.csv"
SUMMARY = AUDIT / "COLLECTION_RELEASE_DIAGNOSTIC_2026-09-30.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suite(path: Path) -> tuple[dict[str, str], list[ET.Element]]:
    root = ET.parse(path).getroot()
    metadata = next(root.iter("testsuite")).attrib
    return metadata, list(root.iter("testcase"))


def main() -> None:
    package_meta, _package_cases = suite(PACKAGE)
    integration_meta, integration_cases = suite(INTEGRATION)
    nonpass = []
    for case in integration_cases:
        issue = case.find("failure")
        if issue is None:
            issue = case.find("error")
        if issue is None:
            continue
        message = issue.attrib.get("message", "")
        if "collection_process_identity_bootstrap_order_or_source" not in message:
            raise ValueError(f"Unexpected release failure: {case.attrib['name']}: {message}")
        nonpass.append(
            {
                "test_id": case.attrib["classname"] + "::" + case.attrib["name"],
                "diagnostic_cause": "PYTEST_CONFTEST_IMPORTS_SCHEDULER_BEFORE_COLLECTION_BARRIER",
            }
        )
    if len(nonpass) != 10 or package_meta["failures"] != "0" or integration_meta["failures"] != "10":
        raise ValueError("Unexpected collection release test census")
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(nonpass[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(nonpass)
    direct = ALONE.read_text(encoding="utf-8")
    if "Ran 12 tests" not in direct or "\nOK\n" not in direct:
        raise ValueError("Standalone process identity control did not pass")
    package_files = list((RUNTIME / "src/quant_platform/orchestration/collection_release").glob("*.py"))
    package_files += [
        RUNTIME / "tests/conftest.py",
        RUNTIME / "tests/test_process_integration700.py",
        RUNTIME / "src/quant_platform/orchestration/corrective_scheduler_lock.py",
        RUNTIME / "src/quant_platform/orchestration/collection_admission_barrier.py",
    ]
    drift = [
        str(file.relative_to(RUNTIME))
        for file in package_files
        if digest(file) != digest(ISOLATED / file.relative_to(RUNTIME))
    ]
    if drift:
        raise ValueError(f"Runtime candidate file drift: {drift}")
    summary = {
        "schema_version": "thewiz.gate0.collection_release_diagnostic.v1",
        "authority": "DISPOSABLE_LOCAL_TEST_NO_EXTERNAL_OR_ORDER",
        "package_junit_path": str(PACKAGE),
        "package_junit_sha256": digest(PACKAGE),
        "package_tests": int(package_meta["tests"]),
        "package_failures": int(package_meta["failures"]),
        "package_errors": int(package_meta["errors"]),
        "package_pytest_summary": "89 passed, 221 subtests passed",
        "integration_junit_path": str(INTEGRATION),
        "integration_junit_sha256": digest(INTEGRATION),
        "integration_tests": int(integration_meta["tests"]),
        "integration_failures": int(integration_meta["failures"]),
        "integration_errors": int(integration_meta["errors"]),
        "integration_pytest_summary": "157 passed, 10 failed, 29 subtests passed",
        "integration_failure_causes": dict(Counter(row["diagnostic_cause"] for row in nonpass)),
        "standalone_unittest_path": str(ALONE),
        "standalone_unittest_sha256": digest(ALONE),
        "standalone_process_identity_tests_passed": 12,
        "source_and_test_files_matching_runtime": len(package_files),
        "interpretation": "Package tests pass; broader pytest import order conflicts with dedicated collector bootstrap, while standalone process tests pass. Active-checkout integration remains unproven.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
