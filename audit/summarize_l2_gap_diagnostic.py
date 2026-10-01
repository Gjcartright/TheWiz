"""Summarize an isolated historical L2-gap policy test and authority control."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
ACTIVE = AUDIT.parent
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
ISOLATED = Path("/tmp/thewiz-runtime-mathdesk.G5oKC2")
OFFDRIVE = Path("/Users/gregc/Backups/TheWiz/l2-gap-diagnostics/2026-09-30")
BASELINE = OFFDRIVE / "l2-gap-junit.xml"
SYNTHETIC = OFFDRIVE / "l2-gap-with-synthetic-authority-junit.xml"
OUTPUT = AUDIT / "L2_GAP_CANDIDATE_DIAGNOSTIC_2026-09-30.json"
PATHS = (
    "config/acceptance_policy_manifest.json",
    "src/quant_platform/orchestration/corrective_data_evidence.py",
    "src/quant_platform/orchestration/corrective_acceptance_policy_validation.py",
    "tests/test_corrective_data_evidence.py",
    "tests/test_corrective_acceptance_policy_validation.py",
    "tests/conftest.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suite(path: Path) -> tuple[dict[str, str], list[ET.Element]]:
    root = ET.parse(path).getroot()
    item = next(root.iter("testsuite"))
    return item.attrib, list(item.iter("testcase"))


def main() -> None:
    baseline, cases = suite(BASELINE)
    synthetic, _cases = suite(SYNTHETIC)
    if (baseline["tests"], baseline["failures"], baseline["errors"]) != (
        "2", "1", "0"
    ):
        raise ValueError("unexpected baseline L2 test result")
    if (synthetic["tests"], synthetic["failures"], synthetic["errors"]) != (
        "1", "0", "0"
    ):
        raise ValueError("synthetic-authority L2 test did not pass")
    failures = [case for case in cases if case.find("failure") is not None]
    if len(failures) != 1 or "unmanaged_publication_authority_session_missing" not in failures[0].find("failure").attrib.get("message", ""):
        raise ValueError("unexpected baseline L2 failure cause")
    matching = {}
    for relative in PATHS:
        runtime = digest(RUNTIME / relative)
        isolated = digest(ISOLATED / relative)
        if runtime != isolated:
            raise ValueError(f"disposable candidate was not restored: {relative}")
        matching[relative] = runtime
    active_policy = json.loads((ACTIVE / PATHS[0]).read_text(encoding="utf-8"))
    runtime_policy = json.loads((RUNTIME / PATHS[0]).read_text(encoding="utf-8"))
    if "maximum_strict_l2_gap_minutes" in active_policy["cost_gates"]:
        raise ValueError("active policy already contains candidate L2 gap field")
    if runtime_policy["cost_gates"].get("maximum_strict_l2_gap_minutes") != 15:
        raise ValueError("runtime L2 gap policy changed")
    summary = {
        "schema_version": "thewiz.gate0.l2_gap_candidate_diagnostic.v1",
        "authority": "DISPOSABLE_SYNTHETIC_FILE_PUBLICATION_NO_EXTERNAL_OR_ORDER",
        "baseline_junit_path": str(BASELINE),
        "baseline_junit_sha256": digest(BASELINE),
        "baseline_policy_validation_passed": 1,
        "baseline_data_evidence_failed_on_publication_authority": 1,
        "synthetic_authority_junit_path": str(SYNTHETIC),
        "synthetic_authority_junit_sha256": digest(SYNTHETIC),
        "synthetic_authority_data_evidence_passed": 1,
        "candidate_source_test_and_restored_harness_sha256": matching,
        "runtime_maximum_strict_l2_gap_minutes": 15,
        "active_policy_has_gap_field": False,
        "interpretation": "The candidate's bounded-gap policy and one data-evidence case pass when the disposable pytest harness supplies its synthetic file-publication authority. The unchanged baseline test fails before its assertion. Active-source policy, evidence logic, validator, and tests need a coordinated review before any port.",
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS historical_l2_gap_diagnostic_recorded")


if __name__ == "__main__":
    main()
