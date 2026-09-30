#!/usr/bin/env python3
"""Verify that historical workspace claims are dated and output is ignored."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "audit/GATE0_WORKSPACE_ALIGNMENT_REPAIR_2026-09-30.json"
REPORT_SHA256 = "16ecb5edc6067dcf36527c70858658e3243a97f979345303026f8db49169573c"
SELECTED_COMMIT = "80cc6220899209ba05db1bf8abe20971f8944444"
HISTORICAL_COMMIT = "c0f7f5c3d1aaa707f1f7b61bdf0444cd2bf93f65"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_bytes(commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout


def main() -> None:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SELECTED_COMMIT, "HEAD"],
        cwd=ROOT,
        check=True,
    )
    if not REPORT.is_file() or sha256(REPORT.read_bytes()) != REPORT_SHA256:
        raise ValueError("workspace alignment repair receipt drift")
    report = json.loads(REPORT.read_bytes())
    if (report["selected_commit"] != SELECTED_COMMIT
            or report["historical_plan_commit"] != HISTORICAL_COMMIT):
        raise ValueError("workspace alignment commit binding changed")
    selected = {}
    for relative, expected in report["selected_file_sha256"].items():
        content = git_bytes(SELECTED_COMMIT, relative)
        if sha256(content) != expected:
            raise ValueError(f"selected workspace alignment source drift: {relative}")
        selected[relative] = content.decode("utf-8")
    old_plan = git_bytes(HISTORICAL_COMMIT, "docs/workspace_alignment_plan.md")
    old_report = git_bytes(HISTORICAL_COMMIT, "reports/workspace_alignment_status.md")
    if (b"/Users/gregc/Documents/Codex/TheWiz-publish-20260625" not in old_plan
            or b".env.local present: **YES**" not in old_report):
        raise ValueError("historical workspace claims not preserved in Git")
    plan = selected["docs/workspace_alignment_plan.md"]
    status = selected["reports/workspace_alignment_status.md"]
    validator = selected["scripts/ops/validate_workspace.sh"]
    if ("June 2026 Historical Record" not in plan
            or "/Volumes/Expansion/Crypto Wizard" not in plan
            or "Historical Placeholder" not in status
            or "not current evidence" not in status
            or 'REPORT_PATH="$REPORT_DIR/active/workspace_alignment_status.md"' not in validator
            or 'cat > "$REPORT_PATH"' not in validator):
        raise ValueError("current versus historical workspace guidance changed")
    subprocess.run(["bash", "-n", "scripts/ops/validate_workspace.sh"], cwd=ROOT, check=True)
    ignored = subprocess.run(
        ["git", "check-ignore", "reports/active/workspace_alignment_status.md"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if ignored.returncode != 0:
        raise ValueError("generated workspace status is not Git ignored")

    original_columns = report["csv_original_columns_preserved"]
    old_header = next(csv.reader(git_bytes(HISTORICAL_COMMIT, "work/legacy_workspace_inventory.csv").decode("utf-8").splitlines()))
    lines = selected["work/legacy_workspace_inventory.csv"].splitlines()
    header = next(csv.reader(lines))
    rows = list(csv.DictReader(lines))
    if (old_header != original_columns
            or header != [*original_columns, "inventory_as_of"]
            or len(rows) != 7
            or {row["inventory_as_of"] for row in rows} != {"2026-06-26"}):
        raise ValueError("dated workspace inventory format changed")

    probe = report["validator_probe"]
    log = Path(probe["log_path"])
    generated = Path(probe["generated_report_path"])
    if (probe["commit"] != SELECTED_COMMIT
            or not log.is_file()
            or sha256(log.read_bytes()) != probe["log_sha256"]
            or not generated.is_file()
            or sha256(generated.read_bytes()) != probe["generated_report_sha256"]
            or not probe["generated_report_git_ignored"]
            or not probe["tracked_worktree_clean_after_probe"]):
        raise ValueError("isolated workspace validator probe drift")
    if (f"active_head={SELECTED_COMMIT[:7]}" not in log.read_text(encoding="utf-8")
            or "Credential values inspected: **NO**" not in generated.read_text(encoding="utf-8")):
        raise ValueError("workspace validator probe content changed")
    retired = ROOT / "audit/GATE0_LEGACY_ENCRYPTED_MOUNT_RETIREMENT_2026-09-30.json"
    if sha256(retired.read_bytes()) != report["legacy_mount_retirement_receipt_sha256"]:
        raise ValueError("legacy mount retirement receipt drift")
    if (report["credential_readiness"]
            or report["research_acceptance"] != "BLOCKED"
            or report["testnet_order_authority"]
            or report["live_trading_authorized"]):
        raise ValueError("workspace alignment repair exceeded its authority")
    print("PASS workspace_alignment_repair")


if __name__ == "__main__":
    main()
