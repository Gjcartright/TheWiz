#!/usr/bin/env python3
"""Verify the obsolete three-hour research loop remains fail closed."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTED_COMMIT = "1c50d7efc67d7d53da4bae722dc0acae41b0fe94"
BASE_COMMIT = "14dc97e22cfa298b6d058e7b493c0ccd4aab8d61"
SCRIPT = "scripts/run_three_hour_bh_loop.sh"
REPORT = ROOT / "audit/GATE0_LEGACY_BH_LOOP_RETIREMENT_2026-09-30.json"
REPORT_SHA256 = "641ae7e8218938844191f8001a3553dfcea842709734819ab8006d29c87b6a86"
HISTORICAL_SHA256 = "27dd3688439ecef6c5543b743e820d7b1c47a848cc2aaa8bdcd54653d4b3d9cb"
RETIRED_SHA256 = "d5156e29299fa90f1a0f8c52c6121dce20f5ba006b65204a330999b9505ae302"


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
    subprocess.run(["git", "merge-base", "--is-ancestor", SELECTED_COMMIT, "HEAD"], cwd=ROOT, check=True)
    if not REPORT.is_file() or sha256(REPORT.read_bytes()) != REPORT_SHA256:
        raise ValueError("legacy loop retirement report drift")
    report = json.loads(REPORT.read_bytes())
    old = git_bytes(BASE_COMMIT, SCRIPT)
    retired = git_bytes(SELECTED_COMMIT, SCRIPT)
    if (report["base_commit"] != BASE_COMMIT
            or report["decision"] != "RETIRED_FAIL_CLOSED"
            or sha256(old) != HISTORICAL_SHA256
            or sha256(retired) != RETIRED_SHA256
            or report["historical_sha256"] != HISTORICAL_SHA256
            or report["retirement_stub_sha256"] != RETIRED_SHA256
            or sha256((ROOT / SCRIPT).read_bytes()) != RETIRED_SHA256):
        raise ValueError("historical or active three-hour loop source changed")
    if (b"/Users/gregc/Documents/Codex/TheWiz-publish-20260625" not in old
            or b"run-dydx-pair-expansion" not in old
            or b"|| true" not in old
            or b"run-dydx-pair-expansion" in retired
            or b"exit 78" not in retired):
        raise ValueError("legacy API route or fail-closed retirement differs")
    subprocess.run(["bash", "-n", SCRIPT], cwd=ROOT, check=True)
    probe = subprocess.run(["bash", SCRIPT], cwd=ROOT, capture_output=True, text=True, check=False)
    qualification = report["qualification"]
    if (probe.returncode != 78
            or probe.stdout != ""
            or probe.stderr.strip() != qualification["isolated_stub_stderr"]
            or qualification["provider_calls_during_review"]
            or qualification["loop_started_during_review"]):
        raise ValueError("retired loop invocation did not fail closed")
    if (report["historical_source_reconciliation"]["union_queue_row_for_script"] != 0
            or report["runtime_observation"]["legacy_root_exists"]
            or report["runtime_observation"]["tracked_functional_callers"]):
        raise ValueError("legacy loop source inventory changed")
    print("PASS legacy_bh_loop_retirement")


if __name__ == "__main__":
    main()
