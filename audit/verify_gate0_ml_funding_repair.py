#!/usr/bin/env python3
"""Bind the isolated funding and ML leakage repair to its source and test evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
BASE = "039611077e125624fe9cb26ffc9a5f93d025d29a"
JUNIT = Path(
    "/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30/"
    "ml-funding-candidate-junit.xml"
)
OUTPUT = AUDIT / "GATE0_ML_FUNDING_REPAIR_2026-09-30.json"
EXPECTED = {
    "src/quant_platform/dydx_candles.py": "d7a0543bb7523b3d026be472af0bf0b9192e50441a6731b5f2c605b6a30b0725",
    "src/quant_platform/ml_filter.py": "94b935fa9e4c798ca749893318490b102bd26b43d072830a0f3e5e44e133d977",
    "tests/test_dydx_candles.py": "cfd7f965690187218d8d68352d8a94f04106b32fcbcb0eeb8a09447eda1efc6a",
    "tests/test_ml_filter.py": "2428cca07501adbb487e07aa08a8575f7030b52599c76306cb41f5d97a430f00",
    "audit/verify_gate0_v23_source_transition.py": "eed2364941c1c3533bcd611859c0b48a0b0a9ab075ab770bac0921841fbd02ab",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError(f"repair source or test drift: {relative}")

    suite = next(ET.parse(JUNIT).getroot().iter("testsuite"))
    counts = {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")}
    if counts != {"tests": 2590, "failures": 0, "errors": 0, "skipped": 0}:
        raise ValueError(f"isolated repair suite is not green: {counts}")

    with (AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 811 or len({row["relative_path"] for row in rows}) != 811:
        raise ValueError("union source queue changed")
    queue = {row["relative_path"]: row for row in rows}
    for relative in EXPECTED:
        if relative.startswith("src/") and queue[relative]["custody_status"] != (
            "PRESERVED_REVIEW_REQUIRED_NO_PORT"
        ):
            raise ValueError("full historical source review was claimed prematurely")
    pending = sum(
        row["custody_status"] in {
            "PRESERVED_REVIEW_REQUIRED_NO_PORT",
            "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED",
        }
        for row in rows
    )
    if pending != 671:
        raise ValueError("remaining Gate 0 source review count changed")

    report = {
        "schema_version": "thewiz.gate0.ml_funding_repair.v1",
        "decision": "ISOLATED_REPAIR_QUALIFIED_EXACT_COMMIT_VERIFICATION_REQUIRED",
        "base_commit": BASE,
        "file_sha256": EXPECTED,
        "isolated_suite": {**counts, "junit_sha256": digest(JUNIT)},
        "repairs": [
            "funding_observations_cannot_fill_earlier_or_out_of_order_candles",
            "undated_funding_cannot_be_retroactively_assigned",
            "spread_point_only_trade_labels_are_rejected",
            "optional_entry_features_preserve_missingness_and_boolean_values_are_rejected",
            "model_features_use_an_entry_time_allowlist_excluding_future_and_hindsight_columns",
            "prior_v23_source_receipt_reads_its_frozen_selected_commit",
        ],
        "remaining_semantic_review_paths": pending,
        "research_acceptance": "BLOCKED",
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS isolated_funding_and_ml_repair_qualified")


if __name__ == "__main__":
    main()
