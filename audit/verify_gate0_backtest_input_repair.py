#!/usr/bin/env python3
"""Verify the narrow backtest input repair and retain the broader source queue."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
DIAGNOSTICS = Path("/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30")
BASE = "f8cb4a8efcab3b539575505acab65ec43e946d7a"
SOURCE = "src/quant_platform/backtest.py"
TEST = "tests/test_backtest.py"
OUTPUT = AUDIT / "GATE0_BACKTEST_INPUT_REPAIR_2026-09-30.json"
CASES = (
    "boolean_signal", "fractional_bars_per_day", "missing_signal_row",
    "nan_funding_divisor", "negative_leg_slippage",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prior_sha(path: str) -> str:
    content = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    with (AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        queue = {row["relative_path"]: row for row in csv.DictReader(stream)}
    if any(queue[path]["custody_status"] != "PRESERVED_REVIEW_REQUIRED_NO_PORT" for path in (SOURCE, TEST)):
        raise ValueError("broader backtest source review must remain pending")
    for path in (SOURCE, TEST):
        if prior_sha(path) != queue[path]["working_sha256_at_freeze"]:
            raise ValueError(f"frozen source baseline changed: {path}")

    paths = {
        "script": DIAGNOSTICS / "backtest-input-probe.py",
        "active_baseline": DIAGNOSTICS / "backtest-input-active.json",
        "runtime_candidate": DIAGNOSTICS / "backtest-input-runtime.json",
        "patched": DIAGNOSTICS / "backtest-input-patched.json",
        "focused_junit": DIAGNOSTICS / "backtest-input-fix-junit.xml",
    }
    probes = {name: json.loads(paths[name].read_text(encoding="utf-8"))
              for name in ("active_baseline", "runtime_candidate", "patched")}
    for name in CASES:
        if probes["active_baseline"][name]["outcome"] != "accepted":
            raise ValueError(f"baseline did not accept {name}")
        for label in ("runtime_candidate", "patched"):
            if (probes[label][name]["outcome"] != "rejected"
                or probes[label][name]["exception_type"] != "ValueError"):
                raise ValueError(f"{label} did not reject {name}")
    if probes["runtime_candidate"] != probes["patched"]:
        raise ValueError("patched input behavior differs from preserved runtime")
    suite = next(ET.parse(paths["focused_junit"]).getroot().iter("testsuite"))
    if tuple(int(suite.attrib[key]) for key in ("tests", "failures", "errors")) != (37, 0, 0):
        raise ValueError("backtest and acceptance focused suite is not green")

    report = {
        "schema_version": "thewiz.gate0.backtest_input_repair.v1",
        "status": "PARTIAL_SOURCE_REVIEW_REPAIR_COMMITTED_PENDING_REMAINDER",
        "baseline_commit": BASE,
        "source_sha256": {
            SOURCE: {
                "baseline": prior_sha(SOURCE), "patched": sha(ROOT / SOURCE),
                "runtime_candidate": queue[SOURCE]["runtime_sha256_at_freeze"],
                "historical_variants": queue[SOURCE]["historical_variant_sha256"].split(";"),
            },
            TEST: {
                "baseline": prior_sha(TEST), "patched": sha(ROOT / TEST),
                "runtime_candidate": queue[TEST]["runtime_sha256_at_freeze"],
                "historical_variant": queue[TEST]["historical_variant_sha256"],
            },
        },
        "diagnostic_files": {name: {"path": str(path), "sha256": sha(path)}
                             for name, path in paths.items()},
        "focused_tests_passed": 37,
        "decision": "Fail closed on unaligned, missing, boolean, and complex signals; invalid funding clock divisors; and negative leg slippage. Retain the broader backtest implementation variants for dependency-complete review.",
        "remaining_source_decisions": [
            "profit-factor numerator and denominator currency contract",
            "two-leg ledger and return accounting, including fixed-units holding",
            "hedge-ratio kind, spread orientation, and rebalance policy",
            "funding-clock and annualization policy",
            "historical backtest tests and shared math dependencies",
        ],
        "authority": "No strategy acceptance, Testnet order, or live order authority is granted.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS five_backtest_input_rejections_and_partial_source_decision")


if __name__ == "__main__":
    main()
