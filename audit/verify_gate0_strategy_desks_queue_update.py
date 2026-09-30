#!/usr/bin/env python3
"""Verify the bounded strategy/desk/RL queue proposal against preserved bytes.

This verifier is read-only. It does not query providers or update the queue.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
TABLE = AUDIT / "GATE0_STRATEGY_DESKS_27_PATH_QUEUE_UPDATE_2026-09-30.csv"
TABLE_SHA256 = "7afab0178f66d4263731a266e1684e8272f07bcbff5d899e38fc6b0f3cdc6993"
REPORT_SHA256 = {
    "GATE0_STRATEGY_DESKS_SOURCE_RECONCILIATION_2026-09-30.json": "6cc980be93a8d3dcde6cae28be50bfa8b8c0d08af80bbba7fc3312fbf974357d",
    "GATE0_RL_FORENSIC_VARIANT_SUPPLEMENT_2026-09-30.json": "e91b509bcaf7302c6ccf9b05ac903ab6ceda2dbfddadd93c778896259d2364cc",
    "GATE0_RL_VISUAL_VARIANT_AND_CALIBRATION_SUPPLEMENT_2026-09-30.json": "d3d5998f7da41f9c3b645acf00ba8d4cecf4c96089f071217641a88924ff3a66",
}
FULL_JUNIT_SHA256 = "3dd0054246f550edba13ebac7056806973a4a9dfe07fcdd04d6e7301fa1dcaab"
EXPECTED_QUEUE_STATES = {
    "REVIEWED_SELECTIVE_PORT": 8,
    "REVIEWED_EXACT_PORT": 5,
    "REVIEWED_NO_PORT": 6,
    "PENDING_BLOCKED": 5,
    "PENDING_DEFERRED": 3,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hash(errors: list[str], label: str, path: Path, expected: str) -> None:
    if not expected:
        return
    if not path.is_file():
        errors.append(f"missing:{label}:{path}")
        return
    actual = sha256(path)
    if actual != expected:
        errors.append(f"hash_mismatch:{label}:{path}:{actual}!={expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-junit", action="store_true", help="also verify the local full-suite JUnit file")
    parser.add_argument("--junit", type=Path, default=Path("/tmp/strategy-desks-calibration-full.xml"))
    args = parser.parse_args()
    errors: list[str] = []

    check_hash(errors, "queue_update_table", TABLE, TABLE_SHA256)
    for name, expected in REPORT_SHA256.items():
        check_hash(errors, "source_report", AUDIT / name, expected)
    if errors:
        print(json.dumps({"verified": False, "errors": errors}, indent=2, sort_keys=True))
        return 1

    with TABLE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    base = json.loads((AUDIT / "GATE0_STRATEGY_DESKS_SOURCE_RECONCILIATION_2026-09-30.json").read_text())
    visual = json.loads((AUDIT / "GATE0_RL_VISUAL_VARIANT_AND_CALIBRATION_SUPPLEMENT_2026-09-30.json").read_text())
    runtime_root = Path(base["roots"]["local_runtime"])
    visual_root = Path(visual["roots"]["visual_recovery"])
    candidate_root = Path(visual["roots"]["source_candidate"])
    forensic_root = Path(visual["roots"]["forensic_b"])
    queue_rows = [row for row in rows if row["entry_type"] == "QUEUE_PATH"]
    addenda = [row for row in rows if row["entry_type"] == "VISUAL_VARIANT_ADDENDUM"]
    table_paths = [row["relative_path"] for row in rows]
    base_paths = {row["path"] for row in base["paths"]}
    visual_variants = {
        row["path"]: row["visual_recovery_sha256"]
        for row in visual["paths"]
        if row["visual_is_distinct_variant"]
    }
    table_visual = {
        row["relative_path"]: row["visual_recovery_variant_sha256"]
        for row in rows
        if row["visual_recovery_variant_sha256"]
    }

    if len(rows) != 31 or len(queue_rows) != 27 or len(addenda) != 4:
        errors.append(f"unexpected_row_counts:{len(rows)}:{len(queue_rows)}:{len(addenda)}")
    if len(table_paths) != len(set(table_paths)):
        errors.append("duplicate_table_path")
    if {row["relative_path"] for row in queue_rows} != base_paths:
        errors.append("queue_paths_do_not_match_27_path_report")
    if table_visual != visual_variants or len(table_visual) != 10:
        errors.append("visual_variant_set_or_hash_mismatch")
    states = Counter(row["review_state"] for row in queue_rows)
    if dict(states) != EXPECTED_QUEUE_STATES:
        errors.append(f"queue_state_count_mismatch:{dict(states)}")

    for row in rows:
        path = row["relative_path"]
        selected = ROOT / path
        expected_selected = row["selected_sha256"]
        if expected_selected:
            check_hash(errors, "selected", selected, expected_selected)
        elif selected.exists():
            errors.append(f"unexpected_selected_path:{path}")
        check_hash(errors, "local_runtime", runtime_root / path, row["local_runtime_sha256"])
        check_hash(errors, "source_candidate", candidate_root / path, row["source_candidate_sha256"])
        check_hash(errors, "visual_recovery", visual_root / path, row["visual_recovery_variant_sha256"])
        forensic_hashes = [value for value in row["forensic_variant_sha256"].split(";") if value]
        if forensic_hashes:
            actual = sha256(forensic_root / path) if (forensic_root / path).is_file() else ""
            if actual not in forensic_hashes:
                errors.append(f"forensic_variant_hash_mismatch:{path}:{actual}")
        state = row["review_state"]
        if state.startswith("PENDING_") and expected_selected:
            errors.append(f"pending_path_has_selected_bytes:{path}")
        if state == "REVIEWED_EXACT_PORT" and expected_selected != row["local_runtime_sha256"]:
            errors.append(f"exact_port_differs_from_runtime:{path}")

    selected_src = ROOT / "src"
    imported_desks = []
    for source in selected_src.rglob("*.py"):
        contents = source.read_text(encoding="utf-8")
        if any(f"quant_platform.{desk}_desk" in contents for desk in ("static", "dynamic", "copula")):
            imported_desks.append(str(source.relative_to(ROOT)))
    if imported_desks:
        errors.append(f"unexpected_active_desk_imports:{','.join(imported_desks)}")

    junit_evidence = {"sha256": FULL_JUNIT_SHA256, "tests": 2841, "failures": 0, "errors": 0, "static_signal_tests": 39}
    if args.check_junit:
        check_hash(errors, "full_junit", args.junit, FULL_JUNIT_SHA256)
        if args.junit.is_file():
            tree = ElementTree.parse(args.junit)
            suite = tree.find(".//testsuite")
            actual_counts = {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors")}
            if actual_counts != {key: junit_evidence[key] for key in actual_counts}:
                errors.append(f"full_junit_count_mismatch:{actual_counts}")
            signal_cases = [
                case for case in tree.findall(".//testcase")
                if "test_static_desk_signal_repairs" in case.get("classname", "")
            ]
            if len(signal_cases) != 39:
                errors.append(f"static_signal_junit_count_mismatch:{len(signal_cases)}")

    output = {
        "schema_version": "thewiz.gate0.strategy_desks_queue_verifier.v1",
        "verified": not errors,
        "queue_paths": len(queue_rows),
        "visual_variant_addenda": len(addenda),
        "visual_variant_hashes": len(table_visual),
        "review_states": dict(sorted(states.items())),
        "full_suite_evidence": junit_evidence,
        "full_junit_checked": args.check_junit,
        "errors": errors,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
