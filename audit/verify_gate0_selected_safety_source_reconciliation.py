#!/usr/bin/env python3
"""Verify selected math, L2 cadence and scheduler safety source decisions."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTED_COMMIT = "8a00bf88f40fc214005195bc6d730c8ea4dfe208"
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
OUTPUT = ROOT / "audit/GATE0_SELECTED_SAFETY_SOURCE_RECONCILIATION_2026-09-30.json"
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")
DIAGNOSTICS = Path("/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30")
SELECTED = {
    "config/acceptance_policy_manifest.json": "65d88e28ee4f1201cea95a0777afd5aebcd26b2dd638764a2393f5ab17e4d923",
    "src/quant_platform/orchestration/corrective_data_evidence.py": "6950fe07d5019d546c80eed65145d268dd3af3cd9c924e425f2a952752686461",
    "tests/test_corrective_data_evidence.py": "c326b550ca1bbdd3d7498e236c5b4257d9dcd1fb5bbfd337be981b2688cdc993",
    "tests/pair_cost_bundle_support.py": "23e66c76413910b6f64c7d51c345baa2ad90d505fb42ae22cbd3c56b6b6438a4",
    "src/quant_platform/statistics/math_v2.py": "f40fbd88078e96c9793242bb1b73cd8ce9853942707899d3c0c87e6c897257cb",
    "tests/test_statistics_math_v2.py": "27954f793f0f0609bd350a49aa8cbb0d41c033b329d13be4a772375245900516",
    "tests/test_performance_math_v2.py": "7f9f90b2e4b4d96dfe0f68784c92230bd2a7d81b6851ec6699ddadd58d41ebf1",
    "tests/test_static_desk_ecm_repairs.py": "efba8cf0be0c34d4e404c712a6fdd9f3557c3e80d24d5efacd3e5ebb97d70550",
    "src/quant_platform/orchestration/corrective_scheduler_supervisor.py": "077bd2e2ff1ce9f4cfc5acfe883d5f80e3fc098a4801bff3f6d894a99c92a65b",
    "tests/test_phase00_scheduler_supervisor.py": "f511fac4081770ce3d8f241a61a9112f5b4f21318d2f68d347aa374f58a00c8d",
}
DEFERRED = {
    "src/quant_platform/orchestration/corrective_acceptance_policy_validation.py": "e887549d64db2fdac8c8df5efc52ae8977929bd4789975ab9f581f2742df1497",
    "tests/test_corrective_acceptance_policy_validation.py": "84c18c0f46988b893e4017584fb3cd7a8a219e39edc9c6a9ae2ae74a43e68ee7",
}
REVIEWED = frozenset(SELECTED) - frozenset({
    "src/quant_platform/orchestration/corrective_data_evidence.py",
    "tests/test_corrective_data_evidence.py",
    "src/quant_platform/orchestration/corrective_scheduler_supervisor.py",
    "tests/test_phase00_scheduler_supervisor.py",
}) | frozenset(DEFERRED)
PARTIAL = frozenset(SELECTED) - REVIEWED
QUEUE_SHA256 = "c78390549444e7dee3fe1c76eff441e9c1fa3b8ec2910711eb2c097a8eacb062"
JUNIT = {
    "math_baseline": (
        DIAGNOSTICS / "statistics-math-gate0-baseline/statistics-math-baseline-junit.xml",
        "8f66d277563ac9a0e55f7c09997dd5029480e33a469292098baf7d8098b059db",
        (77, 53, 0),
    ),
    "math_focused": (
        DIAGNOSTICS / "statistics-math-gate0-candidate/statistics-math-focused-final-junit.xml",
        "5c42847522f3bd0005f19eb3da35d5dc0092140e56a31f21db4f39c18be14b39",
        (89, 0, 0),
    ),
    "math_full": (
        DIAGNOSTICS / "statistics-math-gate0-candidate/statistics-math-full-final-junit.xml",
        "920cfca1c189456c02f1aef7926ae0acda6f8ae5205df746dbe5ef512650cfba",
        (2690, 0, 0),
    ),
    "supervisor_full": (
        DIAGNOSTICS / "supervisor-failclosed-i9xjzq/full-after-junit.xml",
        "3ddd31d8ed18613ecb70f58910f1f3922ae0d34e6f307666389157a01fe0e531",
        (2621, 0, 0),
    ),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selected_bytes(relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{SELECTED_COMMIT}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout


def selected_exists(relative: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{SELECTED_COMMIT}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    ).returncode == 0


def main() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", SELECTED_COMMIT, "HEAD"], cwd=ROOT, check=True)
    for relative, expected in SELECTED.items():
        if digest(selected_bytes(relative)) != expected:
            raise ValueError(f"selected safety source changed: {relative}")
    for relative, expected in DEFERRED.items():
        path = SAVEPOINT / relative
        if selected_exists(relative) or not path.is_file() or digest(path.read_bytes()) != expected:
            raise ValueError(f"deferred validator source custody changed: {relative}")
    policy = json.loads(selected_bytes("config/acceptance_policy_manifest.json"))
    if policy["cost_gates"]["maximum_strict_l2_gap_minutes"] != 15:
        raise ValueError("strict L2 gap bound changed")
    source = selected_bytes("src/quant_platform/statistics/math_v2.py").decode("utf-8")
    if "EG_ECM_COMPONENT_CONTRACT" not in source or "sample_identity" not in source:
        raise ValueError("paired sample and relationship contract missing")
    supervisor = selected_bytes(
        "src/quant_platform/orchestration/corrective_scheduler_supervisor.py"
    ).decode("utf-8")
    if ("provider_effect_absence_proven" not in supervisor
            or "scheduler_crash_retry_requires_manual_reauthorization" not in supervisor
            or "retryable = False" not in supervisor):
        raise ValueError("scheduler recovery guard missing")

    junit_report = {}
    for label, (path, expected_sha, expected_counts) in JUNIT.items():
        if not path.is_file() or digest(path.read_bytes()) != expected_sha:
            raise ValueError(f"diagnostic JUnit drift: {label}")
        suite = next(ET.parse(path).getroot().iter("testsuite"))
        counts = tuple(int(suite.attrib[key]) for key in ("tests", "failures", "errors"))
        if counts != expected_counts:
            raise ValueError(f"diagnostic JUnit counts changed: {label}")
        junit_report[label] = {"sha256": expected_sha, "tests": counts[0], "failures": counts[1], "errors": counts[2]}

    if digest(QUEUE.read_bytes()) != QUEUE_SHA256:
        raise ValueError("union source queue drift")
    with QUEUE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 811 or len({row["relative_path"] for row in rows}) != 811:
        raise ValueError("union source queue changed")
    queue = {row["relative_path"]: row for row in rows}
    for relative in REVIEWED | PARTIAL:
        row = queue[relative]
        expected = "REVIEWED_" if relative in REVIEWED else "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED"
        if (not row["custody_status"].startswith(expected)
                or OUTPUT.name not in row["decision_evidence"]):
            raise ValueError(f"safety source queue decision missing: {relative}")
    pending = sum(row["custody_status"] in {
        "PRESERVED_REVIEW_REQUIRED_NO_PORT",
        "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED",
    } for row in rows)
    reviewed = sum(row["custody_status"].startswith("REVIEWED_") for row in rows)
    if (pending, reviewed, len(rows) - pending - reviewed) != (631, 162, 18):
        raise ValueError("source queue accounting changed")

    report = {
        "schema_version": "thewiz.gate0.selected_safety_source_reconciliation.v1",
        "decision": "SELECTED_SAFETY_REPAIRS_PORTED_WITH_REMAINDER_DEFERRED",
        "selected_commit": SELECTED_COMMIT,
        "selected_source_sha256": SELECTED,
        "deferred_validator_savepoint_sha256": DEFERRED,
        "reviewed_paths": sorted(REVIEWED),
        "partial_remaining_paths": sorted(PARTIAL),
        "diagnostic_junit": junit_report,
        "reason_codes": [
            "MATH_COMPLETE_PAIRED_GRID_REQUIRED",
            "HISTORICAL_JOHANSEN_VARIANT_FAIL_OPEN_NO_PORT",
            "STRICT_L2_MAXIMUM_INTERIOR_GAP_AND_LATEST_AGE_15_MINUTES",
            "DORMANT_POLICY_VALIDATOR_CANONICAL_STATUS_V2_DEPENDENCY_NO_PORT",
            "ABANDONED_SCHEDULER_RETRY_REQUIRES_MANUAL_REAUTHORIZATION",
            "UNPROVEN_PROVIDER_EFFECT_ABSENCE_KEEPS_RESERVATION_OPEN",
        ],
        "union_source_queue": {
            "sha256": QUEUE_SHA256,
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
    print("PASS selected_safety_source_reconciliation")


if __name__ == "__main__":
    main()
