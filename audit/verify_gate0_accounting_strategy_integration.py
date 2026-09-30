#!/usr/bin/env python3
"""Verify the exact disposable V2.3 accounting and strategy candidate."""

from __future__ import annotations

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(
    "/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30/"
    "accounting-historical-integration"
)
MANIFEST = EVIDENCE / "QUALIFIED_V23_SOURCE_MANIFEST.json"
SNAPSHOT = EVIDENCE / "qualified-v23-source"
JUNIT = EVIDENCE / "full-v23-release-junit.xml"
LOG = EVIDENCE / "full-v23-release.log"
PREFIX = EVIDENCE / "strategy-prefix-causality-result.json"
RUFF = EVIDENCE / "v23-ruff.log"
OUTPUT = ROOT / "audit/GATE0_ACCOUNTING_STRATEGY_INTEGRATION_2026-09-30.json"
BASE = "6806d00682079a1d91f00b903263794fe009beb8"
EXPECTED_MANIFEST_SHA256 = "1234259c9904550905da1a01ec9e8d7518cf2a5c00c89a78819d1aabc0be60ec"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if digest(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("qualified V2.3 source manifest changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["base_commit"] != BASE or len(manifest["files"]) != 27:
        raise ValueError("qualified V2.3 source identity changed")
    paths: set[str] = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[0] not in {"src", "tests"}
            or entry["path"] in paths
        ):
            raise ValueError(f"unsafe or duplicate source path: {entry['path']}")
        paths.add(entry["path"])
        path = SNAPSHOT / relative
        if path.is_symlink() or digest(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"qualified source drift: {entry['path']}")

    suite = next(ET.parse(JUNIT).getroot().iter("testsuite"))
    tests = int(suite.attrib["tests"])
    failures = int(suite.attrib["failures"])
    errors = int(suite.attrib["errors"])
    skipped = int(suite.attrib.get("skipped", "0"))
    if (tests, failures, errors, skipped) != (2584, 0, 0, 0):
        raise ValueError("V2.3 full disposable suite is not green")
    if "2584 passed" not in LOG.read_text(encoding="utf-8"):
        raise ValueError("V2.3 full suite log is incomplete")
    classes = {case.attrib["classname"] for case in ET.parse(JUNIT).getroot().iter("testcase")}
    for required in (
        "test_historical_accounting_input_contract",
        "test_historical_strategy_authority",
        "test_historical_backtest_v23",
        "test_historical_performance_math_v23",
    ):
        if not any(name.endswith(required) for name in classes):
            raise ValueError(f"historical repair tests absent: {required}")
    if RUFF.read_text(encoding="utf-8").strip() != "All checks passed!":
        raise ValueError("V2.3 correctness lint is not green")

    prefix = json.loads(PREFIX.read_text(encoding="utf-8"))
    if (
        prefix["active"]["changed_prefix_rows"] != [2, 14, 16, 18, 20, 26, 30, 36, 38]
        or prefix["forensic_b"]["changed_prefix_rows"] != []
    ):
        raise ValueError("regime lookahead probe changed")
    original = subprocess.check_output(
        ["git", "show", f"{BASE}:src/quant_platform/strategies.py"], cwd=ROOT
    )
    active_strategy_path = str(EVIDENCE / "repo/src/quant_platform/strategies.py")
    if hashlib.sha256(original).hexdigest() != prefix["source_sha256"][active_strategy_path]:
        raise ValueError("lookahead probe is not bound to the original source")
    for path, expected in prefix["source_sha256"].items():
        if path != active_strategy_path and digest(Path(path)) != expected:
            raise ValueError(f"historical lookahead source drift: {path}")

    report = {
        "schema_version": "thewiz.gate0.accounting_strategy_integration.v1",
        "decision": "DISPOSABLE_V23_INTEGRATION_QUALIFIED_ACTIVE_PORT_PENDING",
        "base_commit": BASE,
        "manifest_path": str(MANIFEST),
        "manifest_sha256": digest(MANIFEST),
        "qualified_source_files": len(paths),
        "full_suite": {
            "tests": tests,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "junit_path": str(JUNIT),
            "junit_sha256": digest(JUNIT),
            "log_path": str(LOG),
            "log_sha256": digest(LOG),
        },
        "correctness_lint_sha256": digest(RUFF),
        "lookahead_probe_sha256": digest(PREFIX),
        "historical_repair_scope": [
            "same-currency PnL and risk metrics",
            "fixed-units execution and explicit spread/hedge orientation",
            "venue interval and calendar-month distinction",
            "complete numeric inputs and spread-only nonacceptance",
            "causal regime thresholds and proxy strategy authority labels",
        ],
        "remaining_gate0_source_paths": 678,
        "qualification_scope": (
            "This exact disposable source passed integration diagnostics. The Expansion "
            "checkout has not selected these files. No research acceptance, Testnet "
            "order, or live trading authority follows."
        ),
        "next_source_decision": (
            "Review the V2.3 source transition and historical verifier lifecycle, "
            "then port as one versioned group and requalify the exact active commit."
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS disposable_v23_accounting_strategy_integration")


if __name__ == "__main__":
    main()
