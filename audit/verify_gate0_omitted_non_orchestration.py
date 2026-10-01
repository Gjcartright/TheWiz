#!/usr/bin/env python3
"""Verify the 47 non-orchestration source variants omitted from the first queue."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "7eecd20d45232ce4a4658c16158efa1307318019"
REPORT = ROOT / "audit/GATE0_OMITTED_UNION_NON_ORCHESTRATION_RECONCILIATION_2026-09-30.json"
PROPOSAL = ROOT / "audit/GATE0_OMITTED_UNION_NON_ORCHESTRATION_QUEUE_UPDATE_2026-09-30.csv"
INPUTS = ROOT / "audit/GATE0_UNION_OMITTED_VARIANT_INPUTS_2026-09-30.csv"
BASELINE = ROOT / "audit/evidence_freeze_2026-09-29/source_reconciliation.csv"
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
REPORT_SHA = "5144abddd4aeea00f12ac7745d83702f8a6b0651f37b75c27cb90efa5d90b4d7"
PROPOSAL_SHA = "7aefca1b000c48a4be0176dde758cc5f8f0c16a8eeaea0bb4bb99d25aae7a38e"
LATER_SELECTED_TEST_REPAIR = {
    # The historical visual copy remains a no-port decision. This selected
    # test later gained a focused persisted-readiness regression at 0311956.
    "tests/test_v2_run.py": "ddaea665932d18a1edfaa028bb338ad4a7b468bce5598eac99d0ed8509127511",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def selected(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{BASE}:{path}"], cwd=ROOT,
        check=True, capture_output=True,
    ).stdout


def verify() -> dict[str, int]:
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    if sha(REPORT.read_bytes()) != REPORT_SHA or sha(PROPOSAL.read_bytes()) != PROPOSAL_SHA:
        raise ValueError("non-orchestration decision evidence hash changed")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if "no source/test bytes changed" not in report["no_authority_claim"] or len(report["rows"]) != 47:
        raise ValueError("report scope or authority changed")
    proposed = {row["relative_path"]: row for row in rows(PROPOSAL.read_bytes())}
    queue = {row["relative_path"]: row for row in rows(QUEUE.read_bytes())}
    baseline = {row["relative_path"]: row for row in rows(BASELINE.read_bytes())}
    copies: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows(INPUTS.read_bytes()):
        if not row["relative_path"].startswith("src/quant_platform/orchestration/"):
            copies[row["relative_path"]].append(row)
    if set(copies) != set(proposed) or set(copies) != {r["relative_path"] for r in report["rows"]}:
        raise ValueError("47-path decision set differs from omitted frozen manifest")
    checked = 0
    for record in report["rows"]:
        path = record["relative_path"]
        proposal = proposed[path]
        current = queue[path]
        frozen = baseline[path]
        if sha(selected(path)) != record["selected_active_sha256"]:
            raise ValueError(f"selected base source changed: {path}")
        if sha((ROOT / path).read_bytes()) not in {
            record["selected_active_sha256"],
            LATER_SELECTED_TEST_REPAIR.get(path, record["selected_active_sha256"]),
        }:
            raise ValueError(f"selected current source changed: {path}")
        if frozen["decision"] != "NO_PORT_NEEDED" or frozen["working_vs_runtime"] != "SAME":
            raise ValueError(f"four-root status mismatch: {path}")
        if frozen["working_sha256"] != record["four_root_selected_baseline_sha256"]:
            raise ValueError(f"four-root selected SHA mismatch: {path}")
        if record["recommendation"] != "no-port" or record["authority_granted"] is not False:
            raise ValueError(f"unreviewed authority claim: {path}")
        variant_sha = record["historical_variant_sha256"]
        if current["historical_variant_sha256"] != variant_sha:
            raise ValueError(f"queue lost historical SHA: {path}")
        if current["custody_status"] != proposal["custody_status"]:
            raise ValueError(f"queue decision mismatch: {path}")
        if proposal["decision_rationale"] not in current["decision_rationale"]:
            raise ValueError(f"queue rationale missing: {path}")
        if REPORT.relative_to(ROOT).as_posix() not in current["decision_evidence"].split("; "):
            raise ValueError(f"queue citation missing: {path}")
        if {r["root_label"] for r in copies[path]} != {"visual_recovery", "forensic_b"}:
            raise ValueError(f"copy roots differ: {path}")
        for copy in copies[path]:
            if copy["sha256"] != variant_sha or sha(Path(copy["absolute_path"]).read_bytes()) != variant_sha:
                raise ValueError(f"frozen copy changed: {path} @ {copy['root_label']}")
            checked += 1
    return {"paths": len(copies), "verified_historical_copies": checked}


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
