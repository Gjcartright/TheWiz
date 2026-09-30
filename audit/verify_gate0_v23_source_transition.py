#!/usr/bin/env python3
"""Verify the selected V2.3 source transition and remaining custody queue."""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
EVIDENCE = Path(
    "/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30/"
    "accounting-historical-integration"
)
MANIFEST = EVIDENCE / "QUALIFIED_V23_SOURCE_MANIFEST.json"
JUNIT = EVIDENCE / "full-v23-release-junit.xml"
QUEUE = AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
OUTPUT = AUDIT / "GATE0_V23_SOURCE_TRANSITION_2026-09-30.json"
BASE = "989a67c4833422da57065863d4fc6edf218da7d7"
MANIFEST_SHA256 = "1234259c9904550905da1a01ec9e8d7518cf2a5c00c89a78819d1aabc0be60ec"

REVIEWED_PORTED = {
    "src/quant_platform/backtest.py",
    "src/quant_platform/economic_contract.py",
    "src/quant_platform/experiments.py",
    "src/quant_platform/performance_math.py",
    "src/quant_platform/protective_exits.py",
    "src/quant_platform/risk_metrics.py",
    "src/quant_platform/runtime_types.py",
    "src/quant_platform/strategies.py",
    "src/quant_platform/trade_ledger.py",
}
PARTIAL_REVIEW = {
    "src/quant_platform/orchestration/teacher_evidence_materializer.py",
    "tests/test_dydx_candles.py",
    "tests/test_experiments.py",
    "tests/test_ml_filter.py",
    "tests/test_student_readiness.py",
    "tests/test_teacher_adapters.py",
    "tests/test_teacher_council.py",
    "tests/test_teacher_evidence_materializer.py",
}
REPLACED_TESTS = {
    "tests/test_risk_metrics.py": "tests/test_risk_metrics_v23.py",
    "tests/test_accounting_repairs.py": "tests/test_accounting_repairs_v23.py",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if digest(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("qualified V2.3 manifest changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["base_commit"] != "6806d00682079a1d91f00b903263794fe009beb8":
        raise ValueError("disposable candidate base changed")
    files = manifest["files"]
    if len(files) != 27 or len({item["path"] for item in files}) != 27:
        raise ValueError("V2.3 source set changed")
    for item in files:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in {"src", "tests"}:
            raise ValueError(f"unsafe source path: {relative}")
        active = ROOT / relative
        if active.is_symlink() or digest(active) != item["sha256"] or active.stat().st_size != item["bytes"]:
            raise ValueError(f"selected V2.3 source drift: {relative}")

    suite = next(ET.parse(JUNIT).getroot().iter("testsuite"))
    if tuple(int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")) != (
        2584, 0, 0, 0
    ):
        raise ValueError("V2.3 qualification is not green")

    with QUEUE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 811 or len({row["relative_path"] for row in rows}) != 811:
        raise ValueError("union source queue changed size or contains duplicates")
    queue = {row["relative_path"]: row for row in rows}
    expected_status = {
        **{path: "REVIEWED_PORTED_V23_SOURCE" for path in REVIEWED_PORTED},
        **{path: "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED" for path in PARTIAL_REVIEW},
        **{path: "REVIEWED_REPLACED_BY_VERSIONED_V23_TEST" for path in REPLACED_TESTS},
    }
    for path, status in expected_status.items():
        row = queue[path]
        if row["custody_status"] != status or OUTPUT.name not in row["decision_evidence"]:
            raise ValueError(f"V2.3 queue decision drift: {path}")
    for old, new in REPLACED_TESTS.items():
        if (ROOT / old).exists() or not (ROOT / new).is_file():
            raise ValueError(f"runtime test was copied under an unversioned name: {old}")
    pending = sum(
        row["custody_status"] in {
            "PRESERVED_REVIEW_REQUIRED_NO_PORT",
            "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED",
        }
        for row in rows
    )
    reviewed = sum(row["custody_status"].startswith("REVIEWED_") for row in rows)
    nonsource = len(rows) - pending - reviewed
    if (pending, reviewed, nonsource) != (671, 122, 18):
        raise ValueError("Gate 0 source queue accounting changed")

    report = {
        "schema_version": "thewiz.gate0.v23_source_transition.v1",
        "decision": "VERSIONED_V23_SOURCE_SELECTED_EXACT_COMMIT_REQUALIFICATION_REQUIRED",
        "base_commit": BASE,
        "qualified_manifest_sha256": digest(MANIFEST),
        "qualified_junit_sha256": digest(JUNIT),
        "selected_source_and_test_files": len(files),
        "reviewed_ported_paths": sorted(REVIEWED_PORTED),
        "partial_historical_review_paths": sorted(PARTIAL_REVIEW),
        "replaced_runtime_tests": REPLACED_TESTS,
        "union_source_queue": {
            "file_sha256": digest(QUEUE),
            "distinct_paths": len(rows),
            "reviewed_paths": reviewed,
            "nonsource_paths": nonsource,
            "remaining_semantic_review_paths": pending,
        },
        "research_acceptance": "BLOCKED",
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "next_check": "Run full diagnostics, CI, wheel, graph dry run, and off-drive restore on the exact selected commit.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS versioned_v23_source_transition_queue_and_hashes")


if __name__ == "__main__":
    main()
