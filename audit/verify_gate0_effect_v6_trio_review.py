#!/usr/bin/env python3
"""Verify preserved v6 authority dependency review and queue decisions."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTED_COMMIT = "394911783c455013b300a51a1b51aefba51c6acf"
QUEUE_COMMIT = "ca809cdb7d01118174bf226efe4d081726fa440b"
REPORT = ROOT / "audit/GATE0_EFFECT_V6_TRIO_REVIEW_2026-09-30.json"
REPORT_SHA256 = "098711ca219b53276ce7126ca3f9cc43c0e04daa1a8c91f198ca410a8fedcd08"
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
QUEUE_SHA256 = "931a0d780d4c86aa2e59872b31d89c8f543bf4974936b4966a1373f35f41da29"
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selected_bytes(relative: str, commit: str = SELECTED_COMMIT) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
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
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SELECTED_COMMIT, "HEAD"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", QUEUE_COMMIT, "HEAD"],
        cwd=ROOT,
        check=True,
    )
    if not REPORT.is_file() or sha256(REPORT.read_bytes()) != REPORT_SHA256:
        raise ValueError("effect v6 review report drift")
    report = json.loads(REPORT.read_bytes())
    if (report["canonical_commit"] != SELECTED_COMMIT
            or report["contract_versions"]["canonical"]["journal_schema"] != 2
            or report["contract_versions"]["preserved"]["journal_schema"] != 6):
        raise ValueError("effect journal migration boundary changed")
    methods = {item["method"] for item in report["api_breaks"]}
    if methods != {
        "register_external_reservation", "require_open_external_reservation",
        "external_reservation_retry_safe", "close_external_reservation",
    }:
        raise ValueError("effect reservation API break inventory changed")
    entries = report["source_and_primary_test_rows"] + report["direct_barrier_test_rows"]
    if len(entries) != 17 or len({item["relative_path"] for item in entries}) != 17:
        raise ValueError("effect v6 direct source/test inventory changed")
    for item in entries:
        relative = item["relative_path"]
        preserved = SAVEPOINT / relative
        expected = item["preserved_sha256"]
        if (not preserved.is_file()
                or item["mac_savepoint_sha256"] != expected
                or sha256(preserved.read_bytes()) != expected):
            raise ValueError(f"preserved effect source/test drift: {relative}")
        canonical_sha = item["canonical_sha256"]
        if canonical_sha is None:
            if selected_exists(relative):
                raise ValueError(f"runtime-only effect file was present: {relative}")
        elif sha256(selected_bytes(relative)) != canonical_sha:
            raise ValueError(f"selected canonical effect source/test drift: {relative}")

    queue_bytes = selected_bytes(QUEUE.relative_to(ROOT).as_posix(), QUEUE_COMMIT)
    if sha256(queue_bytes) != QUEUE_SHA256:
        raise ValueError("union source queue drift")
    rows = list(csv.DictReader(queue_bytes.decode("utf-8").splitlines()))
    if len(rows) != 811 or len({row["relative_path"] for row in rows}) != 811:
        raise ValueError("union source queue changed")
    queue = {row["relative_path"]: row for row in rows}
    for item in entries:
        relative = item["relative_path"]
        row = queue[relative]
        reviewed = item["proposed_disposition"].startswith("REVIEWED_")
        status = (
            "REVIEWED_EFFECT_V6_DEPENDENCY_NO_PORT"
            if reviewed else "PRESERVED_REVIEW_REQUIRED_NO_PORT"
        )
        if (row["custody_status"] != status
                or REPORT.name not in row["decision_evidence"]
                or row["decision_rationale"] != item["rationale"]):
            raise ValueError(f"effect v6 queue decision missing: {relative}")
    pending = sum(row["custody_status"] in {
        "PRESERVED_REVIEW_REQUIRED_NO_PORT",
        "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED",
    } for row in rows)
    reviewed = sum(row["custody_status"].startswith("REVIEWED_") for row in rows)
    if (pending, reviewed, len(rows) - pending - reviewed) != (611, 182, 18):
        raise ValueError("source queue accounting changed")
    if (report["verification"]["effects_activated"]
            or report["verification"]["orders_activated"]
            or report["verification"]["canonical_source_modified"]):
        raise ValueError("historical review exceeded read-only scope")
    print("PASS effect_v6_trio_review")


if __name__ == "__main__":
    main()
