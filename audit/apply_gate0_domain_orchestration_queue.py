"""Apply and verify two frozen original-811 reconciliation decisions.

Only decision fields for the 19 domain tests and 45 orchestration sources are
changed. The queue's hashes, copy roots, and other cohort rows stay intact.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
DOMAIN = ROOT / "audit/GATE0_DOMAIN_TEST_QUEUE_PROPOSAL_2026-09-30.csv"
ORCHESTRATION = ROOT / "audit/GATE0_ORIGINAL_ORCHESTRATION_PENDING_RECONCILIATION_2026-09-30.json"
BASE = "9a1b36a"
FIELDS = ("custody_status", "decision_rationale", "decision_evidence")

# These repairs were independently reviewed, committed, and tested. The
# proposal remains a faithful record of the pre-repair review state.
DOMAIN_PORTS = {
    "tests/test_research_ingestion.py": (
        "REVIEWED_SELECTIVE_APPLEDOUBLE_PORT_REMAINDER_NO_PORT",
        "AppleDouble sidecar exclusion and regression ported in 543d415; older wholesale tests remain rejected.",
        "543d415",
    ),
    "tests/test_wizard_candidate_set.py": (
        "REVIEWED_SELECTIVE_TIMEFRAME_PORT_REMAINDER_NO_PORT",
        "Minute/monthly candidate identity repair and regression ported in 4152402; older wholesale tests remain rejected.",
        "4152402",
    ),
    "tests/test_wizard_evidence.py": (
        "REVIEWED_SELECTIVE_TIMEFRAME_PORT_REMAINDER_NO_PORT",
        "Minute/monthly bars-per-day repair and regression ported in 4152402; older wholesale tests remain rejected.",
        "4152402",
    ),
    "tests/test_youtube_brain.py": (
        "REVIEWED_SELECTIVE_NATIVE_READINESS_PORT_REMAINDER_NO_PORT",
        "Native video evidence readiness repair and regression ported in 7776c58; older wholesale tests remain rejected.",
        "7776c58",
    ),
}

ORCHESTRATION_PORTS = {
    "src/quant_platform/orchestration/dynamic_stage_runner.py",
    "src/quant_platform/orchestration/venue_gate.py",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def decisions() -> dict[str, dict[str, str]]:
    proposed = {row["relative_path"]: row for row in read_csv(DOMAIN)}
    if len(proposed) != 19 or not DOMAIN_PORTS.keys() <= proposed.keys():
        raise ValueError("domain proposal membership changed")
    result = {}
    for path, row in proposed.items():
        fields = {key: row[key] for key in FIELDS}
        if path in DOMAIN_PORTS:
            status, note, commit = DOMAIN_PORTS[path]
            fields["custody_status"] = status
            fields["decision_rationale"] += " " + note
            fields["decision_evidence"] += "; selective source/test repair " + commit
        result[path] = fields

    report = json.loads(ORCHESTRATION.read_text(encoding="utf-8"))
    rows = report["rows"]
    if len(rows) != 45 or len({row["relative_path"] for row in rows}) != 45:
        raise ValueError("orchestration report membership changed")
    if ORCHESTRATION_PORTS != {
        row["relative_path"] for row in rows
        if row["recommended_queue_status_now"] == "PRESERVED_REVIEW_REQUIRED_NO_PORT"
    }:
        raise ValueError("orchestration port candidates changed")
    for row in rows:
        path = row["relative_path"]
        status = row["recommended_queue_status_after_guard_candidate"]
        rationale = row["recommended_decision_rationale"]
        evidence = row["recommended_decision_evidence_append"]
        if path in ORCHESTRATION_PORTS:
            rationale += " Strict persisted boolean guard and regression ported in dfa008e."
            evidence += "; selective source/test repair dfa008e"
        previous = row["existing_queue_decision_evidence"]
        if previous:
            evidence = previous + "; " + evidence
        if path in result:
            raise ValueError(f"overlapping cohort: {path}")
        result[path] = dict(zip(FIELDS, (status, rationale, evidence)))
    if len(result) != 64:
        raise ValueError("expected 64 exact decisions")
    return result


def base_rows() -> dict[str, dict[str, str]]:
    data = subprocess.check_output(["git", "show", f"{BASE}:{QUEUE.relative_to(ROOT)}"], cwd=ROOT)
    return {row["relative_path"]: row for row in csv.DictReader(data.decode("utf-8").splitlines())}


def run(*, apply: bool) -> None:
    expected = decisions()
    rows = read_csv(QUEUE)
    baseline = base_rows()
    if len(rows) != len(baseline) or len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("queue membership or uniqueness changed")
    for row in rows:
        path = row["relative_path"]
        if path not in expected:
            continue
        old = baseline[path]
        for key in old.keys() - set(FIELDS):
            if row[key] != old[key]:
                raise ValueError(f"frozen custody field changed: {path} {key}")
        if old["custody_status"] != "PRESERVED_REVIEW_REQUIRED_NO_PORT":
            raise ValueError(f"base queue cohort was not pending: {path}")
        if apply:
            if all(row[key] == expected[path][key] for key in FIELDS):
                continue
            if any(row[key] != old[key] for key in FIELDS):
                raise ValueError(f"decision already edited unexpectedly: {path}")
            row.update(expected[path])
        elif any(row[key] != expected[path][key] for key in FIELDS):
            raise ValueError(f"decision mismatch: {path}")
    if apply:
        with QUEUE.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(f"PASS {'applied' if apply else 'verified'} 64 domain/orchestration decisions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    run(apply=parser.parse_args().apply)
