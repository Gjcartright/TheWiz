#!/usr/bin/env python3
"""Verify the reviewed dYdX and ML historical source decisions."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
BASE = "2f8ff5a0271f31e9504da38bd0c1f04845cd5c65"
VARIANT_ROOT = Path("/Users/gregc/Backups/TheWiz/source-delta-evidence/2026-09-30/variants")
JUNIT = Path(
    "/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30/"
    "dydx-ml-reconciliation-final-junit.xml"
)
QUEUE = AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
OUTPUT = AUDIT / "GATE0_DYDX_ML_SOURCE_RECONCILIATION_2026-09-30.json"
EXPECTED = {
    "src/quant_platform/dydx_candles.py": "050828a317398fd49fee82966344123dc9a7fc1baf0ba2c8f716bde7447b0271",
    "src/quant_platform/ml_filter.py": "755d04a12a3adead6cd3301706a09c8b434512536d64584fa8906523117a3679",
    "tests/test_dydx_candles.py": "e1b9261b4109e14c76414476e58125f87b29d6317c94554957fc89ea3f884a40",
    "tests/test_ml_filter.py": "8257f2f3e4f5cf72b9e6edaf29a7c5d57726f9d6ca0a602b115bfcf82e9e9436",
    "audit/verify_gate0_ml_funding_repair.py": "b49859f3def9f12088c5b257cfab38dc07d502054615d29fbf929951e35de7e9",
}
VARIANTS = {
    "src/quant_platform/dydx_candles.py": [
        "78161bb66b85aa1ce860cc34030e883f6312034fc844db6b2254d9d16de47cc6"
    ],
    "src/quant_platform/ml_filter.py": [
        "07f3f824e77f36302808b78a34f1bbb79dfe26ccb5fd81beeda02f242e8fe123",
        "bb156cf2c686aa3cafcf77744f4948cd211bef967654c33b6b6aa45fdbd5d063",
    ],
    "tests/test_dydx_candles.py": [
        "2447c8aabc0953cb802e22a7c106548db4f4149b15b129b84b3a124bd72d9f8b"
    ],
    "tests/test_ml_filter.py": [
        "6d9fac418448eeef73b0718f7b8232b8188a129fa0f4f5f4a6bb1fe03809cca4",
        "79b14fff82db187e296c2f9b50bbe49556362bc13e94c1176dfdb40f9ec50af3",
    ],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_names(path: Path) -> set[str]:
    return {
        node.name
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def main() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError(f"selected source or test drift: {relative}")
    for relative, historical in VARIANTS.items():
        active_names = function_names(ROOT / relative)
        for expected in historical:
            variant = VARIANT_ROOT / expected
            if not variant.is_file() or digest(variant) != expected:
                raise ValueError(f"preserved historical variant drift: {expected}")
            if not function_names(variant).issubset(active_names):
                raise ValueError(f"historical function name unaccounted for: {relative}")

    suite = next(ET.parse(JUNIT).getroot().iter("testsuite"))
    counts = {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")}
    if counts != {"tests": 2593, "failures": 0, "errors": 0, "skipped": 0}:
        raise ValueError(f"isolated reconciliation suite is not green: {counts}")
    if digest(QUEUE) != "29bd1fc68b0a2dc42883e2f8319d8d93af516d8550794918793665d692c53afd":
        raise ValueError("source queue bytes changed")
    with QUEUE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 811 or len({row["relative_path"] for row in rows}) != 811:
        raise ValueError("union source queue changed size or contains duplicates")
    queue = {row["relative_path"]: row for row in rows}
    for relative in VARIANTS:
        row = queue[relative]
        if row["custody_status"] != "REVIEWED_HISTORICAL_VARIANTS_RECONCILED":
            raise ValueError(f"reviewed path not closed: {relative}")
        if OUTPUT.name not in row["decision_evidence"]:
            raise ValueError(f"reviewed path lacks decision evidence: {relative}")
        if set(row["historical_variant_sha256"].split(";")) != set(VARIANTS[relative]):
            raise ValueError(f"historical variant set mismatch: {relative}")
    pending = sum(
        row["custody_status"] in {
            "PRESERVED_REVIEW_REQUIRED_NO_PORT",
            "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED",
        }
        for row in rows
    )
    reviewed = sum(row["custody_status"].startswith("REVIEWED_") for row in rows)
    if (pending, reviewed, len(rows) - pending - reviewed) != (667, 126, 18):
        raise ValueError("Gate 0 source queue accounting changed")

    report = {
        "schema_version": "thewiz.gate0.dydx_ml_source_reconciliation.v1",
        "decision": "HISTORICAL_VARIANTS_RECONCILED_EXACT_COMMIT_VERIFICATION_REQUIRED",
        "base_commit": BASE,
        "selected_file_sha256": EXPECTED,
        "preserved_historical_variant_sha256": VARIANTS,
        "isolated_suite": {**counts, "junit_sha256": digest(JUNIT)},
        "review_decisions": [
            "retain_log_y_on_x_spread_and_explicit_hedge_ratio_orientation",
            "preserve_calendar_month_identity_across_dydx_requests_and_pair_history",
            "namespace_provisional_backfill_without_overwriting_native_features",
            "preserve_unknown_funding_features_in_ml_and_explicit_cost_assumptions_in_backtest",
            "exclude_wizard_outcome_fields_from_ml_features",
            "retain_entry_time_allowlist_and_atomic_model_outputs",
            "historical_test_functions_accounted_for_with_current_contract_assertions",
        ],
        "union_source_queue": {
            "file_sha256": digest(QUEUE),
            "distinct_paths": len(rows),
            "reviewed_paths": reviewed,
            "nonsource_paths": len(rows) - pending - reviewed,
            "remaining_semantic_review_paths": pending,
        },
        "research_acceptance": "BLOCKED",
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS dydx_ml_historical_source_reconciliation")


if __name__ == "__main__":
    main()
