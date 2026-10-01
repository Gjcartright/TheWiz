"""Apply the frozen 107-path test/support review to the current source queue."""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
PROPOSAL = ROOT / "audit/GATE0_REMAINING_PENDING_TEST_SUPPORT_QUEUE_UPDATE_2026-09-30.csv"
BASE = "6fc39c3"
DECISION = {"custody_status", "decision_rationale", "decision_evidence"}
DEADLINE = "tests/test_daily_stage_deadline.py"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expected() -> dict[str, dict[str, str]]:
    rows = read(PROPOSAL)
    if len(rows) != 107 or len({row["relative_path"] for row in rows}) != 107:
        raise ValueError("107-path proposal membership changed")
    result = {row["relative_path"]: row for row in rows}
    row = result[DEADLINE]
    if row["custody_status"] != "REVIEWED_DEPENDENCY_BLOCKED_TEST_NO_PORT":
        raise ValueError("deadline proposal no longer reflects the pre-port review")
    row["custody_status"] = "REVIEWED_SELECTIVE_DAILY_TIMEOUT_TEST_PORT"
    row["decision_rationale"] += " Focused deadline regression and source repair were ported in 85612e8."
    row["decision_evidence"] += "; selective source/test repair 85612e8"
    return result


def run(*, apply: bool) -> None:
    data = subprocess.check_output(
        ["git", "show", f"{BASE}:{QUEUE.relative_to(ROOT)}"], cwd=ROOT
    )
    baseline = {row["relative_path"]: row for row in csv.DictReader(io.StringIO(data.decode()))}
    current = read(QUEUE)
    proposed = expected()
    if len(current) != len(baseline) or not proposed.keys() <= baseline.keys():
        raise ValueError("queue membership changed")
    updated = 0
    for row in current:
        path = row["relative_path"]
        if path not in proposed:
            continue
        prior = baseline[path]
        target = proposed[path]
        if prior["custody_status"] != "PRESERVED_REVIEW_REQUIRED_NO_PORT":
            raise ValueError(f"base decision not pending: {path}")
        for key in prior.keys() - DECISION:
            if row[key] != prior[key] or target[key] != prior[key]:
                raise ValueError(f"frozen custody field changed: {path} {key}")
        if target["custody_status"] == "PRESERVED_REVIEW_REQUIRED_NO_PORT":
            # Thirteen source-dependent regression candidates await separate
            # qualification; this review does not silently close them.
            continue
        if apply:
            if any(row[key] != prior[key] for key in DECISION) and any(
                row[key] != target[key] for key in DECISION
            ):
                raise ValueError(f"decision edited unexpectedly: {path}")
            row.update({key: target[key] for key in DECISION})
        elif any(row[key] != target[key] for key in DECISION):
            raise ValueError(f"decision mismatch: {path}")
        updated += 1
    if updated != 94:
        raise ValueError(f"expected 94 closed decisions, saw {updated}")
    if apply:
        with QUEUE.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=current[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(current)
    print(f"PASS {'applied' if apply else 'verified'} {updated} decisions; 13 selective candidates remain")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    run(apply=parser.parse_args().apply)
