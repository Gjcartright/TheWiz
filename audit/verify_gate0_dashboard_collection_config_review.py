#!/usr/bin/env python3
"""Verify four deferred dashboard/collection configuration decisions."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTED_COMMIT = "80cc6220899209ba05db1bf8abe20971f8944444"
REPORT = ROOT / "audit/GATE0_DASHBOARD_COLLECTION_CONFIG_REVIEW_2026-09-30.json"
REPORT_SHA256 = "cd280627fe7bb05476bf84fece439d44c5b34abf109411eae107093cbc303048"
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
QUEUE_SHA256 = "1a5eee0818a0ee57e176d079ecc36842b82297ca481b7e1b3b2bad56aaf12409"
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
    if not REPORT.is_file() or sha256(REPORT.read_bytes()) != REPORT_SHA256:
        raise ValueError("dashboard config review report drift")
    report = json.loads(REPORT.read_bytes())
    files = report["files"]
    if (report["clone_head"] != SELECTED_COMMIT
            or report["source_changes"]
            or len(files) != 4):
        raise ValueError("dashboard config review scope changed")
    for relative, item in files.items():
        historical = SAVEPOINT / relative
        if (item["active_exists"]
                or selected_exists(relative)
                or not historical.is_file()
                or item["mac_savepoint_sha256"] != item["runtime_sha256"]
                or sha256(historical.read_bytes()) != item["runtime_sha256"]):
            raise ValueError(f"historical dashboard config custody changed: {relative}")
    human = json.loads((SAVEPOINT / "config/human_evidence_research_sources.json").read_bytes())
    budget = json.loads((SAVEPOINT / "config/seven_day_wizard_credit_budget.json").read_bytes())
    example = json.loads((SAVEPOINT / "config/wizard_source_route_adapter.example.json").read_bytes())
    if (len(human["sources"]) != 5
            or budget["protected_reserve"] != 200
            or budget["scheduled_operating_ceiling"] != 800
            or example["max_requests"] != 7):
        raise ValueError("preserved dashboard config semantics changed")
    cross = report["cross_file_evidence"]
    for relative, key in (
        ("src/quant_platform/wizard_credit_budget.py", "active_wizard_budget_source_sha256"),
        ("tests/test_wizard_credit_budget.py", "active_budget_test_sha256"),
    ):
        if sha256(selected_bytes(relative)) != cross[key]:
            raise ValueError(f"active budget source/test drift: {relative}")
    for path, key in (
        (Path(cross["l2_calibration_report_path"]), "l2_calibration_report_sha256"),
        (Path(cross["l2_calibration_report_path"]).with_name("calibration-receipt.json"), "l2_calibration_receipt_sha256"),
        (Path(cross["seven_day_checkpoint_path"]), "seven_day_checkpoint_sha256"),
    ):
        if not path.is_file() or sha256(path.read_bytes()) != cross[key]:
            raise ValueError(f"dashboard historical calibration/checkpoint drift: {path}")
    if ("IN_PROGRESS 5/7" not in cross["seven_day_checkpoint_stage1"]
            or "cadence credit zero" not in cross["l2_calibration_interpretation"]):
        raise ValueError("historical seven-day or L2 authority interpretation changed")

    if sha256(QUEUE.read_bytes()) != QUEUE_SHA256:
        raise ValueError("union source queue drift")
    with QUEUE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 811 or len({row["relative_path"] for row in rows}) != 811:
        raise ValueError("union source queue changed")
    queue = {row["relative_path"]: row for row in rows}
    for relative, item in files.items():
        row = queue[relative]
        if (row["custody_status"] != item["decision"]
                or row["decision_rationale"] != item["rationale"]
                or REPORT.name not in row["decision_evidence"]):
            raise ValueError(f"dashboard config queue decision missing: {relative}")
    pending = sum(row["custody_status"] in {
        "PRESERVED_REVIEW_REQUIRED_NO_PORT",
        "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED",
    } for row in rows)
    reviewed = sum(row["custody_status"].startswith("REVIEWED_") for row in rows)
    if (pending, reviewed, len(rows) - pending - reviewed) != (597, 196, 18):
        raise ValueError("source queue accounting changed")
    print("PASS dashboard_collection_config_review")


if __name__ == "__main__":
    main()
