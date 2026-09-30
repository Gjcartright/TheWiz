#!/usr/bin/env python3
"""Apply or verify exactly nine reviewed residual Gate 0 queue decisions.

All frozen hashes, copy metadata, row order, and unrelated rows are preserved.
The source evidence is pinned to BASE in the accompanying builder.
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = "109de903c3631f057f8a9b8712907266e214f95d"
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
PROPOSAL = ROOT / "audit/GATE0_NINE_RESIDUAL_QUEUE_PROPOSAL_2026-09-30.csv"
BUILDER = ROOT / "scripts/build_gate0_nine_residual_review.py"
DECISION_FIELDS = ("custody_status", "decision_rationale", "decision_evidence")
PROPOSAL_FIELDS = {
    "custody_status": "proposed_custody_status",
    "decision_rationale": "proposed_decision_rationale",
    "decision_evidence": "proposed_decision_evidence",
}


def read_csv(data: bytes) -> tuple[list[str], list[dict[str, str]]]:
    handle = io.StringIO(data.decode("utf-8"), newline="")
    reader = csv.DictReader(handle)
    rows = list(reader)
    if reader.fieldnames is None:
        raise AssertionError("CSV header missing")
    if len(rows) != len({row["relative_path"] for row in rows}):
        raise AssertionError("duplicate relative_path")
    return reader.fieldnames, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the nine decision fields to this isolated queue")
    args = parser.parse_args()
    subprocess.run([sys.executable, str(BUILDER), "--verify"], cwd=ROOT, check=True)
    baseline = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{BASE}:{QUEUE.relative_to(ROOT)}"]
    )
    base_fields, base_rows = read_csv(baseline)
    current_fields, rows = read_csv(QUEUE.read_bytes())
    if current_fields != base_fields or len(rows) != len(base_rows):
        raise AssertionError("queue schema or membership changed")
    if [row["relative_path"] for row in rows] != [row["relative_path"] for row in base_rows]:
        raise AssertionError("queue row order changed")
    base_by_path = {row["relative_path"]: row for row in base_rows}
    _, proposal_rows = read_csv(PROPOSAL.read_bytes())
    if len(proposal_rows) != 9:
        raise AssertionError("proposal must contain exactly nine unique paths")
    proposed = {row["relative_path"]: row for row in proposal_rows}
    if not proposed.keys() <= base_by_path.keys():
        raise AssertionError("proposal path absent from baseline")

    for row in rows:
        path = row["relative_path"]
        if path not in proposed:
            continue
        before = base_by_path[path]
        decision = proposed[path]
        if before["custody_status"] != decision["queue_current_status_at_base"]:
            raise AssertionError(f"base queue status mismatch: {path}")
        if any(row[field] != before[field] for field in base_fields if field not in DECISION_FIELDS):
            raise AssertionError(f"frozen custody metadata changed: {path}")
        target = {field: decision[PROPOSAL_FIELDS[field]] for field in DECISION_FIELDS}
        prior_decision = {field: before[field] for field in DECISION_FIELDS}
        current_decision = {field: row[field] for field in DECISION_FIELDS}
        if args.apply:
            if current_decision not in (prior_decision, target):
                raise AssertionError(f"queue decision was independently edited: {path}")
            row.update(target)
        elif current_decision != target:
            raise AssertionError(f"queue decision mismatch: {path}")
        if "REVIEW_REQUIRED" in row["custody_status"]:
            raise AssertionError(f"review-required status remains: {path}")

    if args.apply:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=base_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        QUEUE.write_bytes(output.getvalue().encode("utf-8"))
    print(f"PASS {'applied' if args.apply else 'verified'} nine residual Gate 0 queue decisions")


if __name__ == "__main__":
    main()
