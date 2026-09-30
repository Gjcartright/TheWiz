"""Classify the isolated Math Desk JUnit runs without granting run authority."""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
DIAGNOSTIC = Path("/Users/gregc/Backups/TheWiz/mathdesk-diagnostics/2026-09-30")
BASELINE = DIAGNOSTIC / "mathdesk-junit.xml"
FOLLOWUP = DIAGNOSTIC / "risk-census-junit.xml"
INDEX = Path(
    "/Users/gregc/TheWiz-LocalRuntime/data/processed/research_knowledge/udemy_lecture_index.csv"
)
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
ISOLATED = Path("/tmp/thewiz-runtime-mathdesk.G5oKC2")
OUTPUT = AUDIT / "MATH_DESK_NONPASS_2026-09-30.csv"
SUMMARY = AUDIT / "MATH_DESK_DIAGNOSTIC_2026-09-30.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases(path: Path) -> list[dict[str, str]]:
    root = ET.parse(path).getroot()
    rows = []
    for testcase in root.iter("testcase"):
        event = testcase.find("failure")
        outcome = "failure"
        if event is None:
            event = testcase.find("error")
            outcome = "error" if event is not None else "pass"
        rows.append(
            {
                "test_id": testcase.attrib["classname"] + "::" + testcase.attrib["name"],
                "outcome": outcome,
                "message": event.attrib.get("message", "") if event is not None else "",
            }
        )
    return rows


def category(row: dict[str, str]) -> str:
    message = row["message"]
    if "governed" in message or "docs/math_desk_constitution.md" in message:
        return "MISSING_GOVERNED_SOURCES"
    if "subprocess.CalledProcessError" in message:
        # The exact subprocess was independently rerun and raised the five-doc error.
        return "MISSING_GOVERNED_SOURCES_INDIRECT"
    if "wizard_research_journal.csv" in message:
        return "MISSING_ACTIVE_JOURNAL"
    if "udemy_lecture_index.csv" in message:
        return "MISSING_INDEX_IN_ISOLATED_COPY"
    raise ValueError(f"Unclassified Math Desk nonpass: {row['test_id']}: {message}")


def main() -> None:
    baseline = cases(BASELINE)
    followup = {row["test_id"]: row for row in cases(FOLLOWUP)}
    nonpass = [row for row in baseline if row["outcome"] != "pass"]
    if len(baseline) != 174 or len(nonpass) != 85:
        raise ValueError("Unexpected Math Desk baseline test census")
    output = []
    for row in nonpass:
        cause = category(row)
        later = followup.get(row["test_id"])
        if cause == "MISSING_INDEX_IN_ISOLATED_COPY":
            if later is None or later["outcome"] != "error" or "review_status" not in later["message"]:
                raise ValueError(f"Index follow-up changed unexpectedly: {row['test_id']}")
            cause = "INDEX_SCHEMA_MISMATCH_AFTER_EXACT_INDEX_COPY"
        output.append(
            {
                "test_id": row["test_id"],
                "baseline_outcome": row["outcome"],
                "diagnostic_cause": cause,
                "followup_outcome": later["outcome"] if later is not None else "NOT_RERUN",
            }
        )
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    with INDEX.open(newline="", encoding="utf-8") as stream:
        actual_fields = csv.DictReader(stream).fieldnames or []
    expected_fields = [
        "lecture_id", "course", "video_title", "lecture_url", "review_status",
        "section", "topic_tags", "transcript_text_stored",
    ]
    source_files = list((RUNTIME / "src/quant_platform/math_desk").glob("*.py"))
    source_files += list((RUNTIME / "tests").glob("test_math_desk_*.py"))
    mismatches = [
        str(path.relative_to(RUNTIME))
        for path in source_files
        if digest(path) != digest(ISOLATED / path.relative_to(RUNTIME))
    ]
    if mismatches:
        raise ValueError(f"Isolated source drift: {mismatches}")
    summary = {
        "schema_version": "thewiz.gate0.mathdesk_diagnostic.v1",
        "authority": "RESEARCH_ONLY_NO_EXTERNAL_OR_ORDER",
        "baseline_junit": str(BASELINE),
        "baseline_junit_sha256": digest(BASELINE),
        "baseline_tests": len(baseline),
        "baseline_passed": sum(row["outcome"] == "pass" for row in baseline),
        "baseline_failed": sum(row["outcome"] == "failure" for row in baseline),
        "baseline_errors": sum(row["outcome"] == "error" for row in baseline),
        "nonpass_cause_counts": dict(sorted(Counter(row["diagnostic_cause"] for row in output).items())),
        "followup_junit": str(FOLLOWUP),
        "followup_junit_sha256": digest(FOLLOWUP),
        "followup_tests": len(followup),
        "followup_passed": sum(row["outcome"] == "pass" for row in followup.values()),
        "followup_errors": sum(row["outcome"] == "error" for row in followup.values()),
        "lecture_index_sha256": digest(INDEX),
        "lecture_index_actual_fields": actual_fields,
        "lecture_index_missing_fields": [field for field in expected_fields if field not in actual_fields],
        "isolated_source_and_test_files_matching_runtime": len(source_files),
        "interpretation": "Nonpasses are missing governed inputs, a missing journal, or an incompatible data-index schema; algorithm acceptance is unproven.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
