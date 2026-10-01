"""Verify the frozen YouTube Brain source variants and narrow readiness repair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = "src/quant_platform/youtube_brain.py"
TEST = "tests/test_youtube_brain.py"
QUEUE = "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
FREEZE = "audit/evidence_freeze_2026-09-29/source_reconciliation.csv"
MANIFEST = "audit/GATE0_EXTENDED_VARIANT_INPUTS_2026-09-30.csv"
REPORT = ROOT / "audit/GATE0_YOUTUBE_BRAIN_SOURCE_DECISION_2026-09-30.json"
REVIEW_COMMIT = "5fe8f5e"
REPAIRED_COMMIT = "ac28775"
STATUS = "REVIEWED_SELECTIVE_NATIVE_FORMULA_READINESS_PORT_REMAINDER_NO_PORT"
EVIDENCE = "audit/GATE0_YOUTUBE_BRAIN_SOURCE_DECISION_2026-09-30.json; selective ports 7776c58 and ac28775"


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def row(data: bytes, path: str) -> dict[str, str]:
    matches = [item for item in csv.DictReader(io.StringIO(data.decode("utf-8"))) if item["relative_path"] == path]
    if len(matches) != 1:
        raise ValueError(f"expected one frozen row for {path}")
    return matches[0]


def build() -> dict[str, object]:
    queue = row(git_bytes(REVIEW_COMMIT, QUEUE), PATH)
    freeze = row(git_bytes(REVIEW_COMMIT, FREEZE), PATH)
    if queue["custody_status"] != "PRESERVED_REVIEW_REQUIRED_NO_PORT":
        raise ValueError("YouTube source was not pending at review")
    selected_before = git_bytes(REVIEW_COMMIT, PATH)
    selected_after = git_bytes(REPAIRED_COMMIT, PATH)
    test_after = git_bytes(REPAIRED_COMMIT, TEST)
    if sha(selected_before) != freeze["working_sha256"]:
        raise ValueError("four-root selected source drift")
    if len({freeze[key] for key in ("working_sha256", "recovery_sha256", "runtime_sha256")}) != 1:
        raise ValueError("four-root copies differ")
    manifest = list(csv.DictReader(io.StringIO(git_bytes(REVIEW_COMMIT, MANIFEST).decode("utf-8"))))
    copies = []
    for entry in manifest:
        if entry["relative_path"] != PATH or entry["status"] != "hashed":
            continue
        file = Path(entry["absolute_path"])
        if not file.is_file() or file.is_symlink():
            raise ValueError(f"preserved copy missing: {file}")
        actual = sha(file.read_bytes())
        if actual != entry["sha256"]:
            raise ValueError(f"preserved copy hash drift: {file}")
        copies.append({"root": entry["root_label"], "absolute_path": str(file), "sha256": actual})
    historical = {item["sha256"] for item in copies} - {sha(selected_before)}
    if historical != set(queue["historical_variant_sha256"].split(";")) - {""} or len(historical) != 2:
        raise ValueError("two historical variant hashes changed")
    text = selected_after.decode("utf-8")
    required = (
        "native_evidence_ready = not native_claims.empty and not formulas.empty",
        '"blocked_external_priors_only"',
        '"blocked_native_formulas_missing"',
        "atomic_write_csv(frame, path, index=False)",
        "atomic_write_parquet(frame, parquet_path, index=False)",
    )
    if any(fragment not in text for fragment in required):
        raise ValueError("selected readiness/publication anchors changed")
    if "test_youtube_native_claims_without_formulas_do_not_mark_research_ready" not in test_after.decode("utf-8"):
        raise ValueError("native-formula regression missing")
    return {
        "schema_version": "thewiz.gate0.youtube_brain_source_decision.v1",
        "path": PATH,
        "review_commit": REVIEW_COMMIT,
        "repaired_commit": REPAIRED_COMMIT,
        "selected_before_sha256": sha(selected_before),
        "selected_after_sha256": sha(selected_after),
        "selected_test_after_sha256": sha(test_after),
        "four_root_freeze_sha256": {key: freeze[key] for key in ("working_sha256", "recovery_sha256", "runtime_sha256")},
        "queue_historical_variant_sha256": sorted(historical),
        "preserved_copies": sorted(copies, key=lambda item: item["root"]),
        "decision": "SELECTIVE_NATIVE_AND_FORMULA_READINESS_PORT_RETAIN_ATOMIC_PUBLICATION",
        "decision_rationale": "The forensic_b variant distinguishes native claims plus formula evidence from external priors and incomplete native extraction, but both historical variants publish directly. The selected source now carries the narrow readiness rules while retaining atomic CSV/Parquet publication. No whole-file historical port or trading authority is granted.",
        "selected_focused_tests": "7 passed in tests/test_youtube_brain.py at ac28775",
        "external_effects_or_orders_executed": False,
    }


def check_queue(report: dict[str, object], *, apply: bool) -> None:
    source = ROOT / QUEUE
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len([item for item in rows if item["relative_path"] == PATH]) != 1:
        raise ValueError("YouTube source queue identity changed")
    before = row(git_bytes(REVIEW_COMMIT, QUEUE), PATH)
    current = next(item for item in rows if item["relative_path"] == PATH)
    for key in before.keys() - {"custody_status", "decision_rationale", "decision_evidence"}:
        if current[key] != before[key]:
            raise ValueError(f"frozen queue field changed: {key}")
    expected = {
        "custody_status": STATUS,
        "decision_rationale": str(report["decision_rationale"]),
        "decision_evidence": EVIDENCE,
    }
    if apply:
        if current["custody_status"] not in {before["custody_status"], STATUS}:
            raise ValueError("YouTube source decision already changed")
        current.update(expected)
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    elif any(current[key] != value for key, value in expected.items()):
        raise ValueError("YouTube source queue decision differs from receipt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--apply-queue", action="store_true")
    args = parser.parse_args()
    report = build()
    data = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.write:
        REPORT.write_text(data, encoding="utf-8")
    elif REPORT.read_text(encoding="utf-8") != data:
        raise ValueError("YouTube Brain source receipt differs from frozen evidence")
    if args.apply_queue:
        check_queue(report, apply=True)
    elif not args.write:
        check_queue(report, apply=False)
    print("PASS YouTube Brain: two historical variants, narrow native/formula repair")
