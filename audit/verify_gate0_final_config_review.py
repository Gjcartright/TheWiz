#!/usr/bin/env python3
"""Verify the remaining two historical config decisions."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTED_COMMIT = "14dc97e22cfa298b6d058e7b493c0ccd4aab8d61"
QUEUE_COMMIT = "79e314d17fe572ebff0ac37596ddebe643facc94"
REPORT = ROOT / "audit/GATE0_FINAL_CONFIG_REVIEW_2026-09-30.json"
REPORT_SHA256 = "5f3904d497462ab847d0f2f0b877509753bef0e95397d1f166cbb40053a7255a"
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
QUEUE_SHA256 = "db787932d609c620ce2a58cf72050090cbb97cd5e593fe2b228540d9a87a9bf9"
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
        raise ValueError("final config review report drift")
    report = json.loads(REPORT.read_bytes())
    files = report["files"]
    if report["clone_head"] != SELECTED_COMMIT or report["source_changes"] or len(files) != 2:
        raise ValueError("final config review scope changed")
    for relative, item in files.items():
        path = SAVEPOINT / relative
        if (item["active_exists"]
                or selected_exists(relative)
                or not item["copies_match"]
                or not path.is_file()
                or item["runtime_sha256"] != item["mac_savepoint_sha256"]
                or item["runtime_sha256"] != item["recovery_local_runtime_sha256"]
                or sha256(path.read_bytes()) != item["runtime_sha256"]):
            raise ValueError(f"historical final config custody drift: {relative}")
        consumer = SAVEPOINT / item["preserved_consumer"]
        if not consumer.is_file() or sha256(consumer.read_bytes()) != item["preserved_consumer_sha256"]:
            raise ValueError(f"preserved config consumer drift: {relative}")
    personal = files["config/personal_self_directed_declaration.example.json"]
    for relative, expected in personal["active_policy_sha256"].items():
        if sha256(selected_bytes(relative)) != expected:
            raise ValueError(f"selected authority policy drift: {relative}")
    risk = files["config/risk_desk_visual_reviews.csv"]
    if (risk["row_count"] != 9
            or len(risk["evidence_paths"]) != 9
            or risk["authority_values"] != [["False", "False"]]):
        raise ValueError("risk visual review evidence/authority changed")
    for evidence in risk["evidence_paths"]:
        relative = evidence["path"]
        if (evidence["present_in_active_clone"]
                or evidence["present_in_mac_savepoint"]
                or evidence["present_in_runtime"]
                or selected_exists(relative)
                or (SAVEPOINT / relative).exists()):
            raise ValueError(f"risk visual source unexpectedly available: {relative}")

    queue_bytes = selected_bytes(QUEUE.relative_to(ROOT).as_posix(), QUEUE_COMMIT)
    if sha256(queue_bytes) != QUEUE_SHA256:
        raise ValueError("union source queue drift")
    rows = list(csv.DictReader(queue_bytes.decode("utf-8").splitlines()))
    if len(rows) != 811 or len({row["relative_path"] for row in rows}) != 811:
        raise ValueError("union source queue changed")
    queue = {row["relative_path"]: row for row in rows}
    for relative, item in files.items():
        row = queue[relative]
        if (row["custody_status"] != item["decision"]
                or row["decision_rationale"] != item["rationale"]
                or REPORT.name not in row["decision_evidence"]):
            raise ValueError(f"final config queue decision missing: {relative}")
    pending = sum(row["custody_status"] in {
        "PRESERVED_REVIEW_REQUIRED_NO_PORT",
        "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED",
    } for row in rows)
    reviewed = sum(row["custody_status"].startswith("REVIEWED_") for row in rows)
    if (pending, reviewed, len(rows) - pending - reviewed) != (595, 198, 18):
        raise ValueError("source queue accounting changed")
    print("PASS final_config_review")


if __name__ == "__main__":
    main()
