#!/usr/bin/env python3
"""Verify legacy encrypted workspace review and retirement custody."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTED_COMMIT = "785b8da316cae8541922b2b51a5738de35cbd179"
REPORT = ROOT / "audit/GATE0_WORKSPACE_OPS_REVIEW_2026-09-30.json"
REPORT_SHA256 = "2f235bcbe76d9d7a0babd3bb20a843ad3aa89ae50d2ae33860b2ad23ac78e057"
RETIREMENT = ROOT / "audit/GATE0_LEGACY_ENCRYPTED_MOUNT_RETIREMENT_2026-09-30.json"
RETIREMENT_SHA256 = "f6251237f5581bd885fe4f289bdb838177970124c91a6d6fb3a2cbc7f57444d4"
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
QUEUE_SHA256 = "d0fdecdf5df180298f5ff83c8ca6676fbfe8f720cce6f9afb167d21e03bb53ad"
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")


def sha256(data: bytes) -> str:
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
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SELECTED_COMMIT, "HEAD"],
        cwd=ROOT,
        check=True,
    )
    if (sha256(REPORT.read_bytes()) != REPORT_SHA256
            or sha256(RETIREMENT.read_bytes()) != RETIREMENT_SHA256):
        raise ValueError("workspace operations review or retirement receipt drift")
    report = json.loads(REPORT.read_bytes())
    retired = json.loads(RETIREMENT.read_bytes())
    source = report["sources"]
    if (report["clone_head"] != SELECTED_COMMIT
            or not report["active_mount_script_absent"]
            or selected_exists("scripts/ops/mount_workspace.sh")
            or sha256(selected_bytes("scripts/ops/validate_workspace.sh"))
            != source["active_validate"]["sha256"]
            or sha256(selected_bytes("scripts/ops/nightly_savepoint.py"))
            != source["active_nightly_source"]["sha256"]):
        raise ValueError("selected active workspace source changed")
    for key, relative in (
        ("runtime_mount", "scripts/ops/mount_workspace.sh"),
        ("runtime_validate", "scripts/ops/validate_workspace.sh"),
    ):
        path = SAVEPOINT / relative
        if not path.is_file() or sha256(path.read_bytes()) != source[key]["sha256"]:
            raise ValueError(f"historical workspace source drift: {relative}")
    for key in ("forensic_validate", "external_mount_agent_plist", "external_mount_agent_script", "installed_nightly_script"):
        item = source[key]
        path = Path(item["path"])
        if not path.is_file() or sha256(path.read_bytes()) != item["sha256"]:
            raise ValueError(f"workspace source custody drift: {key}")
    if (retired["decision"] != "OBSOLETE_ENCRYPTED_MOUNT_AGENT_DISABLED_REVERSIBLY"
            or not retired["after"]["disabled"]
            or not retired["after"]["service_unloaded"]
            or retired["after"]["launchctl_print_exit_code"] != 113
            or not retired["nightly_backup"]["loaded"]
            or retired["nightly_backup"]["last_exit_code"] != 0
            or retired["nightly_backup"]["installed_script_sha256"]
            != source["installed_nightly_script"]["sha256"]
            or retired["encrypted_bundle_modified"]
            or retired["passcode_accessed"]
            or retired["backup_schedule_modified"]):
        raise ValueError("legacy mount retirement or nightly backup boundary changed")

    if sha256(QUEUE.read_bytes()) != QUEUE_SHA256:
        raise ValueError("union source queue drift")
    with QUEUE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 811 or len({row["relative_path"] for row in rows}) != 811:
        raise ValueError("union source queue changed")
    queue = {row["relative_path"]: row for row in rows}
    for relative, decision in report["decision_recommendations"].items():
        row = queue[relative]
        if (row["custody_status"] != decision["status"]
                or row["decision_rationale"] != decision["rationale"]
                or REPORT.name not in row["decision_evidence"]):
            raise ValueError(f"workspace ops queue decision missing: {relative}")
    pending = sum(row["custody_status"] in {
        "PRESERVED_REVIEW_REQUIRED_NO_PORT",
        "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED",
    } for row in rows)
    reviewed = sum(row["custody_status"].startswith("REVIEWED_") for row in rows)
    if (pending, reviewed, len(rows) - pending - reviewed) != (601, 192, 18):
        raise ValueError("source queue accounting changed")
    print("PASS workspace_ops_review_and_mount_retirement")


if __name__ == "__main__":
    main()
