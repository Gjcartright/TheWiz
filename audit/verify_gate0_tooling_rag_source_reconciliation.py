#!/usr/bin/env python3
"""Verify the frozen tooling and research RAG source-family decisions."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "ff07b8b66c115e6a6d0ed3bb084b9df71ffc8483"
DECISION_COMMIT = "a541b424d310a932f938293e711478183f9dda74"
REPORT_PATH = "audit/GATE0_TOOLING_RAG_SOURCE_RECONCILIATION_2026-09-30.json"
REPORT_SHA256 = "e94fa9aade97a5ee330537a552b4f591e1bcde7c0d7659bf9cf5618bd8f3d996"
QUEUE_PATH = "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
QUEUE_SHA256 = "147faf71e8aa2776b29b844dcd291e9dfb66143b3e675cb36e7e67f8244e5e9f"
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
    report_bytes = selected_bytes(REPORT_PATH)
    queue_bytes = selected_bytes(QUEUE_PATH)
    if sha256(report_bytes) != REPORT_SHA256 or sha256(queue_bytes) != QUEUE_SHA256:
        raise ValueError("tooling/RAG evidence or queue drift")
    report = json.loads(report_bytes)
    if (report["base_commit"] != BASE_COMMIT
            or report["decision"] != "DEFER_BOTH_PRESERVED_PACKAGES_NO_ACTIVE_PORT"
            or len(report["source_paths"]) != 18):
        raise ValueError("tooling/RAG review scope changed")
    rows = list(csv.DictReader(queue_bytes.decode("utf-8").splitlines()))
    queue = {row["relative_path"]: row for row in rows}
    if len(rows) != 811 or len(queue) != 811:
        raise ValueError("tooling/RAG queue identity changed")
    for item in report["source_paths"]:
        path = item["relative_path"]
        row = queue[path]
        saved = SAVEPOINT / path
        if (base_exists(path)
                or item["active_path_present"]
                or item["runtime_sha256"] != item["mac_savepoint_sha256"]
                or item["runtime_sha256"] != row["runtime_sha256_at_freeze"]
                or item["runtime_sha256"] != row["recovery_sha256_at_freeze"]
                or not saved.is_file()
                or sha256(saved.read_bytes()) != item["runtime_sha256"]
                or row["custody_status"] != item["proposed_queue_status"]
                or row["decision_rationale"] != item["decision_rationale"]
                or row["decision_evidence"] != REPORT_PATH):
            raise ValueError(f"tooling/RAG custody mismatch: {path}")
    if (report["frozen_data_review"]["rag_existing_unique_roots"] != 0
            or report["frozen_data_review"]["index_exists_in_active_runtime_or_savepoint"]
            or report["isolated_math_adapter_probe"]["tests_executed"] != 0):
        raise ValueError("tooling/RAG blocker evidence changed")
    if sum(row["custody_status"] == "PRESERVED_REVIEW_REQUIRED_NO_PORT" for row in rows) != 520:
        raise ValueError("tooling/RAG queue pending count changed")
    print("PASS tooling_rag_source_reconciliation")


if __name__ == "__main__":
    main()
