#!/usr/bin/env python3
"""Verify eight research/math-desk configuration no-port decisions."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTED_COMMIT = "8a00bf88f40fc214005195bc6d730c8ea4dfe208"
REPORT = ROOT / "audit/GATE0_RESEARCH_MATH_DESK_CONFIG_SOURCE_RECONCILIATION_2026-09-30.json"
REPORT_SHA256 = "691a68b809556a8444821020ebdeab93548e1e38513eb3204534275356852f16"
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
QUEUE_SHA256 = "71238c8239b935a04911191838bf9b5707e8225f083ada47186a16c038722e17"
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")
VARIANTS = Path("/Users/gregc/Backups/TheWiz/source-delta-evidence/2026-09-30/variants")


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
    if not REPORT.is_file() or sha256(REPORT.read_bytes()) != REPORT_SHA256:
        raise ValueError("research config review report drift")
    report = json.loads(REPORT.read_bytes())
    decisions = report["decisions"]
    if (report["active_source_commit"] != SELECTED_COMMIT
            or len(decisions) != 8
            or report["source_files_changed"] != 0
            or report["provider_calls"] != 0
            or report["expansion_checkout_edited"]
            or report["shared_queue_edited"]):
        raise ValueError("research config source review scope changed")
    for item in decisions:
        relative = item["relative_path"]
        active = item["active"]
        runtime = item["local_runtime"]
        if item["port_runtime_or_forensic_bytes"] or any(item["authority_granted_by_this_review"].values()):
            raise ValueError(f"research config authority or port changed: {relative}")
        if active["exists"]:
            if not selected_exists(relative) or sha256(selected_bytes(relative)) != active["sha256"]:
                raise ValueError(f"selected research config drift: {relative}")
        elif selected_exists(relative):
            raise ValueError(f"runtime-only config was active: {relative}")
        runtime_path = SAVEPOINT / relative
        if (not runtime["exists"] or not runtime_path.is_file()
                or sha256(runtime_path.read_bytes()) != runtime["sha256"]):
            raise ValueError(f"historical research config drift: {relative}")
        for path, exists in item["active_consumer_paths"].items():
            if selected_exists(path) != exists:
                raise ValueError(f"active config consumer inventory drift: {path}")
        for path, exists in item["local_runtime_consumer_paths"].items():
            if (SAVEPOINT / path).is_file() != exists:
                raise ValueError(f"runtime config consumer inventory drift: {path}")
        forensic = item.get("forensic_b")
        if forensic is not None:
            variant = VARIANTS / forensic["sha256"]
            if not variant.is_file() or sha256(variant.read_bytes()) != forensic["sha256"]:
                raise ValueError(f"forensic research config variant drift: {relative}")
    assessment = next(item for item in decisions if item["relative_path"] == "config/research_paper_assessments.yaml")
    paper_counts = assessment["counts_and_checks"]
    if (paper_counts["active_papers"] != 23
            or paper_counts["forensic_b_papers"] != 25
            or len(paper_counts["forensic_b_added_paper_ids"]) != 2
            or paper_counts["forensic_b_modified_shared_paper_ids"]):
        raise ValueError("paper assessment delta changed")

    if sha256(QUEUE.read_bytes()) != QUEUE_SHA256:
        raise ValueError("union source queue drift")
    with QUEUE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 811 or len({row["relative_path"] for row in rows}) != 811:
        raise ValueError("union source queue changed")
    queue = {row["relative_path"]: row for row in rows}
    for item in decisions:
        row = queue[item["relative_path"]]
        if (row["custody_status"] != item["decision"]
                or row["decision_rationale"] != item["rationale"]
                or REPORT.name not in row["decision_evidence"]):
            raise ValueError(f"research config queue decision missing: {item['relative_path']}")
    pending = sum(row["custody_status"] in {
        "PRESERVED_REVIEW_REQUIRED_NO_PORT",
        "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED",
    } for row in rows)
    reviewed = sum(row["custody_status"].startswith("REVIEWED_") for row in rows)
    if (pending, reviewed, len(rows) - pending - reviewed) != (603, 190, 18):
        raise ValueError("source queue accounting changed")
    print("PASS research_math_desk_config_source_reconciliation")


if __name__ == "__main__":
    main()
