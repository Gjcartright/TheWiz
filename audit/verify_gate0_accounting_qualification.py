#!/usr/bin/env python3
"""Bind the disposable accounting migration to exact source and test evidence."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
EVIDENCE = Path(
    "/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30/accounting-candidate"
)
QUALIFIED = EVIDENCE / "qualified-source"
MANIFEST = EVIDENCE / "QUALIFIED_SOURCE_MANIFEST.json"
JUNIT = EVIDENCE / "full-qualified-junit.xml"
LOG = EVIDENCE / "full-qualified.log"
BOOLEAN = EVIDENCE / "risk-metric-boolean-qualified.json"
OUTPUT = AUDIT / "GATE0_ACCOUNTING_QUALIFICATION_2026-09-30.json"
BASE = "0e44233b890eb217fb62861676272881480913f4"
FROZEN_HEAD = "989a67c4833422da57065863d4fc6edf218da7d7"
PRESERVED_PATHS = (
    "src/quant_platform/risk_metrics.py",
    "src/quant_platform/protective_exits.py",
    "tests/test_risk_metrics.py",
    "tests/test_accounting_repairs.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["base_commit"] != BASE or len(manifest["files"]) != 20:
        raise ValueError("accounting qualified source manifest changed")
    paths = {entry["path"] for entry in manifest["files"]}
    if len(paths) != 20:
        raise ValueError("accounting qualified source paths are duplicated")
    for entry in manifest["files"]:
        path = QUALIFIED / entry["path"]
        if digest(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"accounting qualified source drift: {entry['path']}")

    suite = next(ET.parse(JUNIT).getroot().iter("testsuite"))
    tests = int(suite.attrib["tests"])
    failures = int(suite.attrib["failures"])
    errors = int(suite.attrib["errors"])
    skipped = int(suite.attrib.get("skipped", "0"))
    if (tests, failures, errors, skipped) != (2510, 0, 0, 0):
        raise ValueError("accounting qualified suite is not green")
    if "2510 passed" not in LOG.read_text(encoding="utf-8"):
        raise ValueError("accounting qualified test log is incomplete")

    boolean = json.loads(BOOLEAN.read_text(encoding="utf-8"))
    if len(boolean) != 3 or any(
        item["status"] != "unavailable"
        or any(value is not None for key, value in item.items() if key != "status")
        for item in boolean.values()
    ):
        raise ValueError("boolean financial inputs were not rejected")

    frozen_queue = subprocess.check_output(
        ["git", "show", f"{FROZEN_HEAD}:audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"],
        cwd=ROOT,
        text=True,
    )
    with io.StringIO(frozen_queue) as stream:
        queue = {row["relative_path"]: row for row in csv.DictReader(stream)}
    for path in PRESERVED_PATHS:
        if (
            queue[path]["custody_status"] != "REVIEWED_ACCOUNTING_CANDIDATE_DEFERRED_NO_PORT"
            or subprocess.run(
                ["git", "cat-file", "-e", f"{FROZEN_HEAD}:{path}"],
                cwd=ROOT, capture_output=True,
            ).returncode == 0
        ):
            raise ValueError(f"qualified candidate was ported without a source decision: {path}")

    report = {
        "schema_version": "thewiz.gate0.accounting_qualification.v1",
        "decision": "DISPOSABLE_INTEGRATION_QUALIFIED_ACTIVE_PORT_DEFERRED",
        "base_commit": BASE,
        "manifest_path": str(MANIFEST),
        "manifest_sha256": digest(MANIFEST),
        "qualified_source_files": len(paths),
        "full_suite": {
            "tests": tests,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "junit_path": str(JUNIT),
            "junit_sha256": digest(JUNIT),
            "log_path": str(LOG),
            "log_sha256": digest(LOG),
        },
        "boolean_probe_path": str(BOOLEAN),
        "boolean_probe_sha256": digest(BOOLEAN),
        "qualification_scope": (
            "The preserved accounting modules and dependent V2.2 fixtures pass in a "
            "disposable exact-base restore. Invalid funding clocks produce ineligible "
            "experiment rows; unavailable profit factor is a teacher blocker; boolean "
            "risk inputs are rejected. This does not grant research acceptance, "
            "order authority, or approval to replace unresolved historical variants."
        ),
        "next_source_decision": (
            "Review the remaining historical variants and active audit contracts, "
            "then port the qualified group as one versioned change and requalify "
            "the exact active commit."
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS disposable_accounting_integration_qualification")


if __name__ == "__main__":
    main()
