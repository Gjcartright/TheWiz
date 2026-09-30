"""Verify the frozen 19-path Wizard/Hyperliquid source reconciliation.

This reads local files and Git objects only. It never imports the project or
executes a producer, scheduler, provider, or order path.
"""

from __future__ import annotations

import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "audit/GATE0_CURRENT_WIZARD_HYPERLIQUID_COHORT_2026-09-30.json"
PREFIX = "src/quant_platform/orchestration/current_wizard_hyperliquid_"
PENDING = "PRESERVED_REVIEW_REQUIRED_NO_PORT"


def _git_bytes(commit: str, relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def verify() -> dict[str, int]:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    base = report["base_commit"]
    records = report["paths"]
    queue = _rows(_git_bytes(base, report["queue_path"]))
    current_queue = {
        row["relative_path"]: row
        for row in _rows((ROOT / report["queue_path"]).read_bytes())
    }
    freeze = {
        row["relative_path"]: row
        for row in _rows(_git_bytes(base, report["four_root_freeze_path"]))
    }
    expected = {
        row["relative_path"]: row
        for row in queue
        if row["relative_path"].startswith(PREFIX)
        and row["custody_status"] == PENDING
    }
    assert len(expected) == 19, f"expected 19 pending paths, got {len(expected)}"
    assert {row["relative_path"] for row in records} == set(expected)
    assert len(records) == len(expected), "duplicate report path"

    variant_count = 0
    for record in records:
        relative_path = record["relative_path"]
        queue_row = expected[relative_path]
        assert _sha(_git_bytes(base, relative_path)) == record["active_sha256"]
        frozen = freeze[relative_path]
        assert frozen["working_sha256"] == record["active_sha256"]
        for label in ("working", "recovery", "runtime"):
            entry = record["four_root_copies"][label]
            assert entry["sha256"] == frozen[f"{label}_sha256"]
            assert _sha(Path(entry["absolute_path"]).read_bytes()) == entry["sha256"]
            variant_count += 1

        history_hashes = set()
        for entry in record["extended_copies"]:
            assert _sha(Path(entry["absolute_path"]).read_bytes()) == entry["sha256"]
            if entry["root_label"] != "source_candidate":
                history_hashes.add(entry["sha256"])
            variant_count += 1
        assert history_hashes == set(
            filter(None, queue_row["historical_variant_sha256"].split(";"))
        ), relative_path
        assert len(history_hashes) == int(queue_row["historical_variant_count"])
        decision = record["decision"]
        expected_status = (
            "REVIEWED_SELECTIVE_PERSISTED_BOOL_PORT"
            if decision == "SELECTIVE_PORT_READY"
            else "REVIEWED_LEGACY_HYPERLIQUID_DEPENDENCY_DEFERRED_NO_PORT"
            if decision.startswith("DEFER_")
            else "REVIEWED_ACTIVE_LEGACY_HYPERLIQUID_NO_PORT"
        )
        current = current_queue[relative_path]
        if current["custody_status"] != expected_status:
            raise AssertionError(f"current queue decision not closed: {relative_path}")
        if "Historical decision " + decision + ": " + record["semantic_rationale"] not in current["decision_rationale"]:
            raise AssertionError(f"current queue rationale missing: {relative_path}")
        if REPORT.relative_to(ROOT).as_posix() not in current["decision_evidence"].split("; "):
            raise AssertionError(f"current queue report citation missing: {relative_path}")
        if current["historical_variant_sha256"] != queue_row["historical_variant_sha256"]:
            raise AssertionError(f"current queue lost frozen SHA: {relative_path}")

    return {"paths": len(records), "verified_copies": variant_count}


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
