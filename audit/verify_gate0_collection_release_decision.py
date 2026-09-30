"""Verify the dependency-blocked selection of the collection release package."""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")
MANIFEST = SAVEPOINT.parent / "SAVEPOINT_MANIFEST.json"
DIAGNOSTIC = Path(
    "/Users/gregc/Backups/TheWiz/collection-release-diagnostics/2026-09-30"
)
OUTPUT = AUDIT / "GATE0_COLLECTION_RELEASE_DECISION_2026-09-30.json"
PREFIX = "src/quant_platform/orchestration/collection_release/"
STATUS = "REVIEWED_DEPENDENCY_BLOCKED_COLLECTION_RELEASE_NO_PORT"
EXPECTED_DEPENDENCIES = {
    "src/quant_platform/orchestration/collection_owner_binding.py",
    "src/quant_platform/orchestration/collection_runtime_route.py",
    "src/quant_platform/orchestration/corrective_trusted_artifact.py",
    "src/quant_platform/orchestration/precollection_admission.py",
    "src/quant_platform/orchestration/precollection_campaign_io.py",
    "src/quant_platform/orchestration/scheduler_terminal_paths.py",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suite(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    return next(root.iter("testsuite")).attrib


def main() -> None:
    with (AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        queue = [
            row for row in csv.DictReader(stream)
            if row["relative_path"].startswith(PREFIX)
        ]
    if len(queue) != 34 or len({row["relative_path"] for row in queue}) != 34:
        raise ValueError("collection release queue has changed")
    saved_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    saved_files = {
        entry["path"]: entry["sha256"]
        for entry in saved_manifest["entries"]
        if entry["type"] == "file"
    }
    hashes = {}
    test_files = []
    for row in queue:
        relative = row["relative_path"]
        if (
            row["custody_status"] != STATUS
            or row["working_vs_runtime"] != "RUNTIME_ONLY"
            or row["historical_variant_count"] != "0"
        ):
            raise ValueError(f"collection release row not selected as expected: {relative}")
        if (ROOT / relative).exists():
            raise ValueError(f"unselected collection release file became active: {relative}")
        candidate, saved = RUNTIME / relative, SAVEPOINT / relative
        candidate_hash, saved_hash = digest(candidate), digest(saved)
        if candidate_hash != row["runtime_sha256_at_freeze"]:
            raise ValueError(f"runtime collection release source drift: {relative}")
        if candidate_hash != saved_hash:
            raise ValueError(f"Mac savepoint collection release drift: {relative}")
        if saved_files.get(f"local_runtime/{relative}") != saved_hash:
            raise ValueError(f"Mac savepoint manifest drift: {relative}")
        hashes[relative] = candidate_hash
        if Path(relative).name.startswith("test_"):
            test_files.append(relative)
    if len(test_files) != 8:
        raise ValueError("collection release in-package test count changed")

    with (AUDIT / "evidence_freeze_2026-09-29/missing_import_edges.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        edges = [
            row for row in csv.DictReader(stream)
            if row["source_runtime_only"].startswith(PREFIX)
        ]
    dependencies = {row["runtime_dependency_file"] for row in edges}
    if (
        len(edges) != 11
        or len({row["source_runtime_only"] for row in edges}) != 10
        or dependencies != EXPECTED_DEPENDENCIES
    ):
        raise ValueError("collection release dependency closure changed")
    for relative in dependencies:
        if (ROOT / relative).exists() or not (RUNTIME / relative).is_file():
            raise ValueError(f"collection release dependency presence changed: {relative}")

    package = suite(DIAGNOSTIC / "collection-release-junit.xml")
    integration = suite(DIAGNOSTIC / "integration-junit.xml")
    if any(
        package[key] != value
        for key, value in (("tests", "310"), ("failures", "0"), ("errors", "0"))
    ):
        raise ValueError("collection release package test result changed")
    if any(
        integration[key] != value
        for key, value in (("tests", "196"), ("failures", "10"), ("errors", "0"))
    ):
        raise ValueError("collection release integration test result changed")
    process_output = (DIAGNOSTIC / "process-unittest-stdout.txt").read_text(
        encoding="utf-8"
    )
    if "Ran 12 tests" not in process_output or "\nOK\n" not in process_output:
        raise ValueError("standalone process control did not pass")
    conftest = (RUNTIME / "tests/conftest.py").read_text(encoding="utf-8")
    process_test = (
        RUNTIME / "tests/test_process_integration700.py"
    ).read_text(encoding="utf-8")
    if (
        "from quant_platform.orchestration.corrective_scheduler_lock import"
        not in conftest
        or "collection_admission_barrier as barrier" not in process_test
    ):
        raise ValueError("process bootstrap import-order evidence changed")
    summary = {
        "schema_version": "thewiz.gate0.collection_release_decision.v1",
        "decision": STATUS,
        "path_sha256": dict(sorted(hashes.items())),
        "in_package_test_files": sorted(test_files),
        "missing_active_dependency_edges": len(edges),
        "missing_active_dependency_paths": sorted(dependencies),
        "package_junit_sha256": digest(DIAGNOSTIC / "collection-release-junit.xml"),
        "package_tests_passed": 310,
        "integration_junit_sha256": digest(DIAGNOSTIC / "integration-junit.xml"),
        "integration_tests_failed": 10,
        "standalone_process_tests_passed": 12,
        "standalone_process_stdout_sha256": digest(
            DIAGNOSTIC / "process-unittest-stdout.txt"
        ),
        "interpretation": "Retain the 34-file package as exact preserved candidate source for Gate 0. Its imports require six absent active modules; package tests pass, but broader process tests fail under the shared pytest bootstrap order. Do not port a partial release stack or grant collection/provider/order authority. Revisit with a dependency-complete candidate and current-checkout integration test.",
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS thirty_four_collection_release_source_decisions")


if __name__ == "__main__":
    main()
