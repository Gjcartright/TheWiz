#!/usr/bin/env python3
"""Verify finite teacher contracts and current-math shadow boundaries."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "c3f4d9d628f471eb993655d91806f9fe5b56039c"
SELECTED_COMMIT = "6cba715163709f82fd49aa9f49afd4b7b8e0abde"
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
OUTPUT = ROOT / "audit/GATE0_TEACHER_NUMERIC_GUARDS_2026-09-30.json"
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")
BASELINE = Path("/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30/teacher-numeric-baseline.json")
JUNIT = Path("/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30/teacher-numeric-guards-junit.xml")
SELECTED = {
    "src/quant_platform/orchestration/student_readiness.py": "c930ce3cf42c920b1e5e32679ffd51348bfa9a5b19f2caa0280efd4c3c3e8a32",
    "src/quant_platform/orchestration/teacher_contracts.py": "b1d9acc42dc36be51d7351f0327267cc33d86c3836daa988f835baf94d86aec9",
    "src/quant_platform/orchestration/teacher_council.py": "3ba243194a70d284c57ec0bdf4f7e08148c396abe83dabd12140ff63c21ecc16",
    "tests/test_teacher_council.py": "322b46f5c2fbdb06238c6535ef0ec731c5c66815c33ecb79c1ebefcf978c0374",
    "audit/verify_gate0_student_teacher_source_reconciliation.py": "26d5d501e7646e16efcb39bd7d2d146f4f6578a176485aad2a6950c20bb2d5b8",
}
SAVEPOINTS = {
    "src/quant_platform/orchestration/teacher_contracts.py": "2d492ba7480bca3d3281da3429775b7a84277e5f35b51068fddc0040b4f5b2d6",
    "src/quant_platform/orchestration/teacher_council.py": "6083b211042f6dc6694d3b017b5fbcf04d90b61cef97d90bc741a5598544c624",
}
BASELINE_SHA256 = "d4e0f80bb68e2c3b601f1a6265f596ae2a158123030f1178370ad38421542258"
REVIEWED = {
    "src/quant_platform/orchestration/teacher_contracts.py",
    "src/quant_platform/orchestration/teacher_council.py",
}
REQUALIFIED = {
    "src/quant_platform/orchestration/student_readiness.py",
    "tests/test_teacher_council.py",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_bytes(relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{SELECTED_COMMIT}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout


def main() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    subprocess.run(["git", "merge-base", "--is-ancestor", SELECTED_COMMIT, "HEAD"], cwd=ROOT, check=True)
    for relative, expected in SELECTED.items():
        if hashlib.sha256(selected_bytes(relative)).hexdigest() != expected:
            raise ValueError(f"selected source, test, or verifier drift: {relative}")
    for relative, expected in SAVEPOINTS.items():
        path = SAVEPOINT / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"local runtime savepoint drift: {relative}")
    if digest(BASELINE) != BASELINE_SHA256:
        raise ValueError("baseline numeric probe drift")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if baseline["source_commit"] != BASE or not all(
        baseline[key]
        for key in (
            "nonfinite_teacher_proposal_accepted",
            "nonfinite_packet_lower_bound_is_infinite",
            "nan_policy_threshold_accepted",
            "previous_math_student_policy_accepted",
        )
    ) or baseline["nonfinite_packet_status"] != "SHADOW_TEST" or baseline["previous_math_policy_status"] != "SHADOW_TEST":
        raise ValueError("baseline false-positive observations changed")
    suite = next(ET.parse(JUNIT).getroot().iter("testsuite"))
    counts = {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")}
    if counts != {"tests": 2615, "failures": 0, "errors": 0, "skipped": 0}:
        raise ValueError(f"isolated numeric-guard suite is not green: {counts}")
    queue_bytes = selected_bytes(QUEUE.relative_to(ROOT).as_posix())
    rows = list(csv.DictReader(queue_bytes.decode("utf-8").splitlines()))
    if len(rows) != 811 or len({row["relative_path"] for row in rows}) != 811:
        raise ValueError("union source queue changed size or contains duplicates")
    queue = {row["relative_path"]: row for row in rows}
    for relative in REVIEWED:
        row = queue[relative]
        if row["custody_status"] != "REVIEWED_TEACHER_NUMERIC_GUARDS" or OUTPUT.name not in row["decision_evidence"]:
            raise ValueError(f"teacher source decision missing: {relative}")
    for relative in REQUALIFIED:
        row = queue[relative]
        if row["custody_status"] != "REVIEWED_STUDENT_TEACHER_HISTORICAL_VARIANTS" or OUTPUT.name not in row["decision_evidence"]:
            raise ValueError(f"requalified prior source decision missing: {relative}")
    pending = sum(
        row["custody_status"] in {
            "PRESERVED_REVIEW_REQUIRED_NO_PORT",
            "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED",
        }
        for row in rows
    )
    reviewed = sum(row["custody_status"].startswith("REVIEWED_") for row in rows)
    if (pending, reviewed, len(rows) - pending - reviewed) != (658, 135, 18):
        raise ValueError("Gate 0 source queue accounting changed")
    report = {
        "schema_version": "thewiz.gate0.teacher_numeric_guards.v1",
        "decision": "FINITE_TEACHER_AND_CURRENT_MATH_GUARDS_EXACT_COMMIT_VERIFICATION_REQUIRED",
        "base_commit": BASE,
        "selected_file_sha256": SELECTED,
        "local_runtime_savepoint_sha256": SAVEPOINTS,
        "baseline_probe_sha256": BASELINE_SHA256,
        "baseline_false_positives": {
            "nonfinite_packet_status": baseline["nonfinite_packet_status"],
            "nonfinite_packet_lower_bound_is_infinite": baseline["nonfinite_packet_lower_bound_is_infinite"],
            "nan_policy_threshold_accepted": baseline["nan_policy_threshold_accepted"],
            "previous_math_policy_status": baseline["previous_math_policy_status"],
            "previous_math_student_policy_accepted": baseline["previous_math_student_policy_accepted"],
        },
        "isolated_suite": {**counts, "junit_sha256": digest(JUNIT)},
        "review_decisions": [
            "reject_nonfinite_teacher_and_student_contract_numbers",
            "require_finite_council_thresholds_and_current_math_version",
            "require_current_math_version_in_student_readiness_policy",
            "reject_duplicate_teacher_ids_before_vote_aggregation",
            "compute_finite_weighted_teacher_means_without_intermediate_overflow",
            "retain_shadow_only_council_boundary",
        ],
        "union_source_queue": {
            "file_sha256": hashlib.sha256(queue_bytes).hexdigest(),
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
    print("PASS finite_teacher_numeric_and_current_math_guards")


if __name__ == "__main__":
    main()
