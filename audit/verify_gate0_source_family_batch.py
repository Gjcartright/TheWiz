#!/usr/bin/env python3
"""Verify the frozen collection, decide, and desk source-family decisions."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "ff07b8b66c115e6a6d0ed3bb084b9df71ffc8483"
DECISION_COMMIT = "8784a63753277736786f7a2c80419027e32e3908"
QUEUE_PATH = "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
QUEUE_SHA256 = "f0b61f0f8843bace02b6d65729adfe7c1e031c48fc97a6d331d9a22bd93c7e2e"
REPORT_SHA256 = {
    "audit/GATE0_DECIDE_SOURCE_FAMILY_RECONCILIATION_2026-09-30.json": "091ef504383e7e44f1c9445969ee97749aa6df0704244a8a88e36aa930c436e4",
    "audit/GATE0_DESK_CONNECTIONS_SOURCE_RECONCILIATION_2026-09-30.json": "7d4e14aa441442e22295157d54d40a3fd726a476e1aebae66b1c3cd9adf5ac92",
    "audit/GATE0_COLLECTION_FAMILY_REVIEW_2026-09-30.json": "2d929f9d68e04936fb79a6a0889cacdba9fe528fb4a621ab9b2f043621f43e64",
}
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selected_bytes(relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{DECISION_COMMIT}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout


def base_exists(relative: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{BASE_COMMIT}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    ).returncode == 0


def main() -> None:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", DECISION_COMMIT, "HEAD"],
        cwd=ROOT,
        check=True,
    )
    queue_bytes = selected_bytes(QUEUE_PATH)
    if sha256(queue_bytes) != QUEUE_SHA256:
        raise ValueError("frozen source-family decision queue drift")
    rows = list(csv.DictReader(queue_bytes.decode("utf-8").splitlines()))
    if len(rows) != 811 or len({row["relative_path"] for row in rows}) != 811:
        raise ValueError("source queue identity or count changed")
    queue = {row["relative_path"]: row for row in rows}
    expected: dict[str, tuple[str, str, str, str]] = {}

    def record(path: str, digest: str, status: str, rationale: str, report: str) -> None:
        if path in expected or not rationale or not status.startswith("REVIEWED_"):
            raise ValueError(f"invalid family decision: {path}")
        expected[path] = digest, status, rationale, report

    for report_path, digest in REPORT_SHA256.items():
        report_bytes = selected_bytes(report_path)
        if sha256(report_bytes) != digest:
            raise ValueError(f"family report drift: {report_path}")
        report = json.loads(report_bytes)
        if "DECIDE" in report_path:
            if report["active_source_commit"] != BASE_COMMIT:
                raise ValueError("decide base commit drift")
            for item in report["source_decisions"] + report["direct_test_decisions"]:
                if item["local_runtime_sha256"] != item["queue_recovery_sha256_at_freeze"]:
                    raise ValueError(f"decide frozen hash drift: {item['relative_path']}")
                record(item["relative_path"], item["local_runtime_sha256"],
                       item["queue_status"], item["reason"], report_path)
        elif "DESK_CONNECTIONS" in report_path:
            if report["base_commit"] != BASE_COMMIT:
                raise ValueError("desk base commit drift")
            for item in report["source_paths"]:
                if item["runtime_sha256"] != item["savepoint_sha256"]:
                    raise ValueError(f"desk frozen hash drift: {item['relative_path']}")
                record(item["relative_path"], item["runtime_sha256"],
                       item["proposed_queue_status"], item["decision_rationale"], report_path)
        else:
            if report["active_commit"] != BASE_COMMIT:
                raise ValueError("collection base commit drift")
            for path, item in report["files"].items():
                if item["runtime_sha256"] != item["savepoint_sha256"]:
                    raise ValueError(f"collection frozen hash drift: {path}")
                record(path, item["runtime_sha256"],
                       item["recommended_status"], item["rationale"], report_path)

    if len(expected) != 50:
        raise ValueError(f"source family decision count changed: {len(expected)}")
    for path, (digest, status, rationale, report_path) in expected.items():
        row = queue[path]
        if (row["custody_status"] != status
                or row["decision_rationale"] != rationale
                or row["decision_evidence"] != report_path
                or digest not in (row["runtime_sha256_at_freeze"],
                                  row["recovery_sha256_at_freeze"])
                or base_exists(path)):
            raise ValueError(f"source family queue/base mismatch: {path}")
        preserved = SAVEPOINT / path
        if not preserved.is_file() or sha256(preserved.read_bytes()) != digest:
            raise ValueError(f"source family savepoint drift: {path}")
    if sum(row["custody_status"] == "PRESERVED_REVIEW_REQUIRED_NO_PORT" for row in rows) != 541:
        raise ValueError("source-family queue pending count changed")
    print("PASS source_family_batch: 50 reviewed; 541 preserved-review-required remain")


if __name__ == "__main__":
    main()
