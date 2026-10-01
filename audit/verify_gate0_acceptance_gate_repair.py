#!/usr/bin/env python3
"""Bind the partial Gate 0 acceptance repair to source and off-drive diagnostics."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
DIAGNOSTICS = Path("/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30")
BASE = "56377e002d98636ea23cd9bc8066b1edc8277187"
FROZEN_HEAD = "989a67c4833422da57065863d4fc6edf218da7d7"
SOURCE = "src/quant_platform/experiments.py"
TEST = "tests/test_experiments.py"
OUTPUT = AUDIT / "GATE0_ACCEPTANCE_GATE_REPAIR_2026-09-30.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def baseline_digest(path: str) -> str:
    return digest(subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT))


def frozen_digest(path: str) -> str:
    return digest(subprocess.check_output(["git", "show", f"{FROZEN_HEAD}:{path}"], cwd=ROOT))


def main() -> None:
    frozen_queue = subprocess.check_output(
        ["git", "show", f"{FROZEN_HEAD}:audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"],
        cwd=ROOT, text=True,
    )
    with io.StringIO(frozen_queue) as stream:
        queue = {row["relative_path"]: row for row in csv.DictReader(stream)}
    if any(queue[path]["custody_status"] != "PRESERVED_REVIEW_REQUIRED_NO_PORT" for path in (SOURCE, TEST)):
        raise ValueError("partial acceptance source review must stay pending")
    source_baseline = baseline_digest(SOURCE)
    test_baseline = baseline_digest(TEST)
    if source_baseline != queue[SOURCE]["working_sha256_at_freeze"]:
        raise ValueError("acceptance baseline differs from frozen working source")
    if source_baseline != "e53fbb31bb781b100899d65da3181e91197308b3d3f217c2be26d4af3325110f":
        raise ValueError("acceptance baseline changed")
    baseline_path = DIAGNOSTICS / "acceptance-probe-baseline.json"
    patched_path = DIAGNOSTICS / "acceptance-probe-patched.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    patched = json.loads(patched_path.read_text(encoding="utf-8"))
    if baseline != {
        "invalid_result_accepted": True,
        "invalid_result_reason": "passed",
        "unsubstantiated_rows_production_eligible": True,
        "unsubstantiated_rows_reason": "passed",
        "unsubstantiated_rows_total_trades": 520,
    }:
        raise ValueError("baseline false-positive probe changed")
    if (patched["invalid_result_accepted"] is not False
        or patched["unsubstantiated_rows_production_eligible"] is not False
        or patched["unsubstantiated_rows_total_trades"] != 260
        or "unverified_eligible_runs:4" not in patched["unsubstantiated_rows_reason"]):
        raise ValueError("patched false-positive probe changed")
    for blocker in (
        "profit_factor_nonfinite", "sharpe_invalid_or_unknown_interval",
        "expectancy_lower_95<=0_or_missing", "open_trades_at_end", "reconciliation_error>",
    ):
        if blocker not in patched["invalid_result_reason"]:
            raise ValueError(f"patched probe missing blocker: {blocker}")

    junit = DIAGNOSTICS / "acceptance-fix-final-junit.xml"
    suite = next(ET.parse(junit).getroot().iter("testsuite"))
    if tuple(int(suite.attrib[key]) for key in ("tests", "failures", "errors")) != (26, 0, 0):
        raise ValueError("focused acceptance suite is not green")
    report = {
        "schema_version": "thewiz.gate0.acceptance_gate_repair.v1",
        "status": "PARTIAL_SOURCE_REVIEW_REPAIR_COMMITTED_PENDING_REMAINDER",
        "baseline_commit": BASE,
        "source_sha256": {
            SOURCE: {
                "baseline": source_baseline,
                "patched": frozen_digest(SOURCE),
                "runtime_candidate": queue[SOURCE]["runtime_sha256_at_freeze"],
                "forensic_variant": queue[SOURCE]["historical_variant_sha256"],
            },
            TEST: {
                "baseline": test_baseline,
                "patched": frozen_digest(TEST),
                "forensic_variant": queue[TEST]["historical_variant_sha256"],
            },
        },
        "probe": {
            "script_path": str(DIAGNOSTICS / "acceptance-probe.py"),
            "script_sha256": file_digest(DIAGNOSTICS / "acceptance-probe.py"),
            "baseline_path": str(baseline_path),
            "baseline_sha256": file_digest(baseline_path),
            "patched_path": str(patched_path),
            "patched_sha256": file_digest(patched_path),
            "baseline_result": baseline,
            "patched_result": patched,
        },
        "focused_junit_path": str(junit),
        "focused_junit_sha256": file_digest(junit),
        "focused_tests_passed": 26,
        "decision": "Fail closed on invalid run metrics and unsubstantiated eligibility; count required cost-scenario trades conservatively once. Preserve the remaining source variants for separate review.",
        "remaining_source_decisions": [
            "forensic strategy registry and authority contract",
            "runtime funding-clock exception behavior",
            "historical funding-feature and proxy-authority tests",
            "backtest, math v2, and strategy source dependencies",
            "event-identity evidence for truly independent trade counts",
        ],
        "authority": "No strategy acceptance, Testnet order, or live order authority is granted.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS partial_acceptance_gate_repair_and_evidence")


if __name__ == "__main__":
    main()
