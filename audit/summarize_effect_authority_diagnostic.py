"""Summarize the isolated LocalRuntime effect-authority regression cohort."""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
ISOLATED = Path("/tmp/thewiz-runtime-mathdesk.G5oKC2")
JUNIT = Path(
    "/Users/gregc/Backups/TheWiz/effect-authority-diagnostics/2026-09-30/"
    "effect-authority-junit.xml"
)
OUTPUT = AUDIT / "EFFECT_AUTHORITY_NONPASS_2026-09-30.csv"
SUMMARY = AUDIT / "EFFECT_AUTHORITY_DIAGNOSTIC_2026-09-30.json"
FILES = (
    "src/quant_platform/orchestration/corrective_phase00_closure.py",
    "tests/test_phase00_effect_authority.py",
    "tests/test_phase00_effect_guard.py",
    "tests/test_gate00g_order_authority.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for relative in FILES:
        if digest(RUNTIME / relative) != digest(ISOLATED / relative):
            raise ValueError(f"Isolated source/test drift: {relative}")
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    nonpass = []
    for test in ET.parse(JUNIT).getroot().iter("testcase"):
        issue = test.find("failure")
        outcome = "FAIL" if issue is not None else "PASS"
        if issue is None:
            issue = test.find("error")
            if issue is not None:
                outcome = "ERROR"
        group = test.attrib["classname"]
        groups[group][outcome] += 1
        if issue is not None:
            message = issue.attrib.get("message", "")
            if "_contained_pytest_command" in message:
                cause = "TEST_CALLS_REMOVED_CONTAINMENT_HELPER"
            elif "toolchain_root" in message and "output_root" in message:
                cause = "TEST_USES_OLD_SANITIZED_ENV_SIGNATURE"
            else:
                raise ValueError(f"Unclassified authority failure: {group}::{test.attrib['name']}")
            nonpass.append(
                {
                    "test_id": group + "::" + test.attrib["name"],
                    "outcome": outcome,
                    "diagnostic_cause": cause,
                }
            )
    if sum(sum(group.values()) for group in groups.values()) != 93 or len(nonpass) != 2:
        raise ValueError("Unexpected effect-authority test census")
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(nonpass[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(nonpass)
    summary = {
        "schema_version": "thewiz.gate0.effect_authority_diagnostic.v1",
        "authority": "DISPOSABLE_LOCAL_TEST_NO_EXTERNAL_OR_ORDER",
        "junit_path": str(JUNIT),
        "junit_sha256": digest(JUNIT),
        "source_and_test_hash_matches": len(FILES),
        "tests": 93,
        "passed": 91,
        "failed": 2,
        "errors": 0,
        "per_module": {key: dict(value) for key, value in sorted(groups.items())},
        "failure_causes": dict(Counter(item["diagnostic_cause"] for item in nonpass)),
        "interpretation": "Candidate safety code passes 91 focused checks but its preserved tests call an obsolete closure API; candidate certification is not established.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
