"""Bind the isolated publication-stack cohort to exact candidate bytes."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
DIAGNOSTICS = Path(
    "/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30"
)
SNAPSHOT = DIAGNOSTICS / "publication-candidate-source-snapshot"
SNAPSHOT_RECEIPT = SNAPSHOT / "SOURCE_SNAPSHOT_RECEIPT.json"
OUTPUT = AUDIT / "PUBLICATION_STACK_DIAGNOSTIC_2026-09-30.json"
SOURCES = (
    "src/quant_platform/orchestration/corrective_runtime.py",
    "src/quant_platform/orchestration/corrective_scheduler_lock.py",
    "src/quant_platform/orchestration/effect_authority.py",
)
TESTS = (
    "tests/test_corrective_runtime.py",
    "tests/test_corrective_scheduler_lock.py",
    "tests/test_exact_publication_targets.py",
)
CONFTEST = "tests/conftest.py"
BASELINE = DIAGNOSTICS / "publication-candidate-junit.xml"
ADJUSTED = DIAGNOSTICS / "publication-candidate-with-fixture-junit.xml"
UNMANAGED = DIAGNOSTICS / "unmanaged-directory-publication-diagnostic.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suite(path: Path) -> tuple[dict[str, str], list[str]]:
    root = ET.parse(path).getroot()
    element = next(root.iter("testsuite"))
    nonpasses = []
    for case in root.iter("testcase"):
        issue = case.find("failure")
        if issue is None:
            issue = case.find("error")
        if issue is None:
            continue
        message = issue.attrib.get("message", "") + " " + (issue.text or "")
        if "publication_authority_session_missing" not in message:
            raise ValueError(f"unclassified publication failure: {case.attrib['name']}")
        nonpasses.append(case.attrib["classname"] + "::" + case.attrib["name"])
    return element.attrib, nonpasses


def kwonly_names(path: Path, function: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function
    ]
    if len(functions) != 1:
        raise ValueError(f"missing function {function} in {path}")
    return {argument.arg for argument in functions[0].args.kwonlyargs}


def top_level_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }


def main() -> None:
    with (AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        queue = {row["relative_path"]: row for row in csv.DictReader(stream)}
    snapshot_receipt = json.loads(SNAPSHOT_RECEIPT.read_text(encoding="utf-8"))
    receipt_entries = {
        item["relative_path"]: item for item in snapshot_receipt["files"]
    }
    if set(receipt_entries) != set((*SOURCES, *TESTS, CONFTEST)):
        raise ValueError("publication source snapshot file set changed")
    hashes = {}
    for relative in (*SOURCES, *TESTS, CONFTEST):
        runtime, saved = RUNTIME / relative, SNAPSHOT / relative
        runtime_hash = digest(runtime)
        if runtime_hash != digest(saved):
            raise ValueError(f"saved candidate source/test file drift: {relative}")
        if (
            receipt_entries[relative]["sha256"] != runtime_hash
            or receipt_entries[relative]["size_bytes"] != saved.stat().st_size
        ):
            raise ValueError(f"candidate source snapshot receipt drift: {relative}")
        if relative in SOURCES:
            row = queue[relative]
            if row["custody_status"] != "PRESERVED_REVIEW_REQUIRED_NO_PORT":
                raise ValueError(f"publication source was prematurely selected: {relative}")
            if runtime_hash != row["runtime_sha256_at_freeze"]:
                raise ValueError(f"candidate changed since source freeze: {relative}")
            if digest(ROOT / relative) != row["working_sha256_at_freeze"]:
                raise ValueError(f"active source changed since source freeze: {relative}")
        hashes[relative] = runtime_hash

    fixture = (RUNTIME / CONFTEST).read_text(encoding="utf-8")
    if any(
        f"'{Path(relative).name}'" in fixture
        for relative in (TESTS[0], TESTS[2])
    ):
        raise ValueError("candidate fixture already includes both selected test files")
    baseline, baseline_nonpasses = suite(BASELINE)
    adjusted, adjusted_nonpasses = suite(ADJUSTED)
    if (
        baseline.get("tests") != "156"
        or baseline.get("failures") != "13"
        or baseline.get("errors") != "0"
        or len(baseline_nonpasses) != 13
        or adjusted.get("tests") != "156"
        or adjusted.get("failures") != "0"
        or adjusted.get("errors") != "0"
        or adjusted_nonpasses
    ):
        raise ValueError("publication cohort outcome changed")
    nonpass_modules = Counter(item.split("::", 1)[0] for item in baseline_nonpasses)
    if nonpass_modules != {
        "tests.test_corrective_runtime": 4,
        "tests.test_exact_publication_targets": 9,
    }:
        raise ValueError("unexpected publication fixture failure distribution")
    active_runtime = ROOT / SOURCES[0]
    candidate_runtime = RUNTIME / SOURCES[0]
    active_kw = kwonly_names(active_runtime, "promote_staged_directory")
    candidate_kw = kwonly_names(candidate_runtime, "promote_staged_directory")
    if "immutable" in active_kw or "immutable" not in candidate_kw:
        raise ValueError("directory promotion API relationship changed")
    required_candidate_symbols = {
        SOURCES[1]: "validate_current_directory_inventory",
        SOURCES[2]: "authorize_directory_publication",
    }
    for relative, function in required_candidate_symbols.items():
        if function in top_level_functions(ROOT / relative):
            raise ValueError(f"candidate publication symbol became active: {function}")
        if function not in top_level_functions(RUNTIME / relative):
            raise ValueError(f"candidate publication symbol missing: {function}")
    unmanaged = json.loads(UNMANAGED.read_text(encoding="utf-8"))
    if (
        unmanaged.get("source_commit")
        != "82a9da8d3580c375767b76e5f7137b4a2bd16b6a"
        or unmanaged.get("observed", {}).get("published") is not True
    ):
        raise ValueError("unmanaged-path observation changed")

    summary = {
        "schema_version": "thewiz.gate0.publication_stack_diagnostic.v1",
        "decision": "PARTIAL_REVIEW_NO_SOURCE_SELECTION",
        "source_and_test_sha256": hashes,
        "source_snapshot_receipt_sha256": digest(SNAPSHOT_RECEIPT),
        "source_snapshot_root": str(SNAPSHOT),
        "baseline_junit_sha256": digest(BASELINE),
        "baseline_tests": 156,
        "baseline_passed": 143,
        "baseline_failed_missing_synthetic_authority": 13,
        "baseline_nonpass_modules": dict(sorted(nonpass_modules.items())),
        "fixture_adjustment": "In the disposable copy only, add test_corrective_runtime.py and test_exact_publication_targets.py to the existing synthetic publication-authority fixture file list; restore conftest byte-for-byte afterward.",
        "fixture_restored_sha256": hashes[CONFTEST],
        "adjusted_junit_sha256": digest(ADJUSTED),
        "adjusted_tests_passed": 156,
        "active_directory_promotion_kwonly": sorted(active_kw),
        "candidate_directory_promotion_kwonly": sorted(candidate_kw),
        "candidate_only_publication_symbols": required_candidate_symbols,
        "unmanaged_path_diagnostic_sha256": digest(UNMANAGED),
        "scope_limit": "The passing candidate cohort uses synthetic test authority and does not certify the full candidate suite or justify porting the larger publication/effect stack. The unmanaged-path probe covers a disposable path outside a governed repository only.",
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS publication_stack_diagnostic_156_adjusted_tests")


if __name__ == "__main__":
    main()
