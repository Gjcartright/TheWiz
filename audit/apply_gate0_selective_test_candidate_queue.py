"""Apply or verify only the 13 qualified pending test decisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
FROZEN = ROOT / "audit/GATE0_ORIGINAL_811_SOURCE_QUEUE_SNAPSHOT_2026-09-30.csv"
PREVIOUS = ROOT / "audit/GATE0_REMAINING_PENDING_TEST_SUPPORT_RECONCILIATION_2026-09-30.json"
REPORT = ROOT / "audit/GATE0_SELECTIVE_TEST_CANDIDATE_QUALIFICATION_2026-09-30.json"
BASE = "2968ac3d3ffeb418b82c4e860ed6397d867ee90b"
BASE_QUEUE_SHA256 = "a4e1afe755532f8e1e872da634b71de5e04534d3ea76e0777cd1732e9c2ccc21"
FROZEN_SHA256 = "db787932d609c620ce2a58cf72050090cbb97cd5e593fe2b228540d9a87a9bf9"
PREVIOUS_SHA256 = "79c73324d32f1abf429e8c434208e3484da67d9f05ae9e465e24ddd0b0ee2d33"
REPORT_SHA256 = "607e76272ea5b7da7a545a8a0bb2b198852de07aec676b2f214009bbdfc818a3"
DECISION = ("custody_status", "decision_rationale", "decision_evidence")
REPORT_REF = "audit/GATE0_SELECTIVE_TEST_CANDIDATE_QUALIFICATION_2026-09-30.json"
VERIFICATION_REF = "audit/GATE0_SELECTIVE_TEST_CANDIDATE_VERIFICATION_2026-09-30.json"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def rows_from_bytes(payload: bytes) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    assert reader.fieldnames is not None
    return reader.fieldnames, list(reader)


def unique(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    mapping = {row["relative_path"]: row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("duplicate queue path")
    return mapping


def targets() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    frozen_bytes = FROZEN.read_bytes()
    previous_bytes = PREVIOUS.read_bytes()
    report_bytes = REPORT.read_bytes()
    if (sha(frozen_bytes), sha(previous_bytes), sha(report_bytes)) != (
        FROZEN_SHA256, PREVIOUS_SHA256, REPORT_SHA256,
    ):
        raise ValueError("frozen decision evidence changed")
    _, frozen_rows = rows_from_bytes(frozen_bytes)
    frozen = unique(frozen_rows)
    if len(frozen) != 811:
        raise ValueError("original union membership changed")
    previous = json.loads(previous_bytes)
    report = json.loads(report_bytes)
    if previous["scope"]["selected_active_commit"] != "58d4f95b826fd1abca61ed5cd80a8adadd2c4455":
        raise ValueError("previous review freeze changed")
    pending = {
        item["relative_path"] for item in previous["rows"]
        if item["decision"] == "selective-port-candidate"
    }
    qualification_rows = report["rows"]
    paths = {item["relative_path"] for item in qualification_rows}
    if len(qualification_rows) != 13 or len(paths) != 13 or paths != pending or not paths <= frozen.keys():
        raise ValueError("qualified 13-path cohort changed")
    if report["scope"]["exact_test_only_ports"] != 12 or report["scope"]["partial_test_only_ports"] != 1:
        raise ValueError("port classification changed")
    result: dict[str, dict[str, str]] = {}
    for item in qualification_rows:
        path = item["relative_path"]
        if item["provider_calls_made"] is not False or item["authority_granted"] is not False:
            raise ValueError(f"authority claim changed: {path}")
        test_sha = sha((ROOT / path).read_bytes())
        if test_sha != item["candidate_port_test_sha256"]:
            raise ValueError(f"selected test bytes changed: {path}")
        full = item["full_preserved_test_result"]
        if item["port_kind"] == "exact-test-only":
            if item["queue_status_recommendation"] != "REVIEWED_SELECTIVE_TEST_PORTED":
                raise ValueError(f"exact port decision changed: {path}")
            if full["fail"] or full["error"] or full["skipped"] or full["pass"] != full["run"]:
                raise ValueError(f"exact port focused result changed: {path}")
            summary = f"Focused preserved regression {full['pass']}/{full['run']} passed."
        elif item["port_kind"] == "partial-test-only" and path == "tests/test_experiment_regime_clock.py":
            if item["queue_status_recommendation"] != "REVIEWED_SELECTIVE_TEST_PARTIAL_REMAINDER_BLOCKED":
                raise ValueError("partial port decision changed")
            subset = item["selected_subset_result"]
            if subset != {"run": 13, "pass": 13, "fail": 0, "error": 0, "skipped": 0}:
                raise ValueError("partial port focused result changed")
            if full != {"run": 22, "pass": 15, "fail": 7, "error": 0, "skipped": 0}:
                raise ValueError("full preserved experiment result changed")
            if len(item["blocked_remainder_exact_failures"]) != 7:
                raise ValueError("blocked experiment remainder changed")
            summary = "Compatible subset 13/13 passed; seven preserved cases remain blocked."
        else:
            raise ValueError(f"unrecognized port: {path}")
        result[path] = {
            "custody_status": item["queue_status_recommendation"],
            "decision_rationale": f"{item['rationale']} {summary} Selected test SHA-256 {test_sha}. No authority granted.",
            "decision_evidence": f"{REPORT_REF}; {VERIFICATION_REF}; selected test SHA-256 {test_sha}",
        }
    return result, frozen


def run(*, apply: bool) -> dict[str, object]:
    baseline_bytes = subprocess.check_output(
        ["git", "show", f"{BASE}:{QUEUE.relative_to(ROOT)}"], cwd=ROOT
    )
    if sha(baseline_bytes) != BASE_QUEUE_SHA256:
        raise ValueError("integrated pre-apply queue baseline changed")
    fields, baseline_rows = rows_from_bytes(baseline_bytes)
    baseline = unique(baseline_rows)
    current_fields, current = rows_from_bytes(QUEUE.read_bytes())
    if fields != current_fields or len(current) != len(baseline) or len(current) != 891:
        raise ValueError("queue schema or membership changed")
    current_map = unique(current)
    if current_map.keys() != baseline.keys():
        raise ValueError("queue path set changed")
    expected, frozen = targets()
    changed = 0
    for path in sorted(expected):
        old = baseline[path]
        row = current_map[path]
        original = frozen[path]
        target = expected[path]
        if old["custody_status"] != "PRESERVED_REVIEW_REQUIRED_NO_PORT":
            raise ValueError(f"pre-apply decision not pending: {path}")
        for key in fields:
            if key not in DECISION and (old[key] != original[key] or row[key] != old[key]):
                raise ValueError(f"frozen custody field changed: {path} {key}")
        old_decision = {key: old[key] for key in DECISION}
        now_decision = {key: row[key] for key in DECISION}
        if apply:
            if now_decision == old_decision:
                row.update(target)
                changed += 1
            elif now_decision != target:
                raise ValueError(f"unexpected intermediate decision: {path}")
        elif now_decision != target:
            raise ValueError(f"decision mismatch: {path}")
    if apply and changed:
        with QUEUE.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(current)
    receipt = {
        "status": "PASS",
        "mode": "apply" if apply else "verify",
        "selected_pre_apply_commit": BASE,
        "qualified_paths": len(expected),
        "changed_paths": changed,
        "exact_ports": 12,
        "partial_ports_with_blocked_remainder": 1,
        "queue_sha256": sha(QUEUE.read_bytes()),
        "report_sha256": REPORT_SHA256,
        "authority_granted": False,
    }
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(apply=args.apply), sort_keys=True))
