#!/usr/bin/env python3
"""Verify 25 nonvenue source decisions and the 11 deliberately open variants."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DECISION_COMMIT = "9e9a2c5b0cbb0b955b366480c3f276d5a271003c"
SOURCE_REPORT = "audit/GATE0_TOP_LEVEL_NONVENUE_SOURCE_RECONCILIATION_2026-09-30.json"
DECISIONS = "audit/GATE0_TOP_LEVEL_NONVENUE_25_DECISIONS_2026-09-30.json"
QUEUE = "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
EXPECTED_SHA256 = {
    SOURCE_REPORT: "97731389fae2b5a42a82ccb7fcefb5e7e8bc6be56763f5014fbe4c15f575f1fc",
    DECISIONS: "0b52cd529b0b79775aab4cba55d7006f12ddee25025b4c92cbd573e525bc9e01",
    QUEUE: "d5f77589e1c95aa85d352b82b0a642daa530b44b2ea0f3ca60523177fa63b282",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selected_bytes(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{DECISION_COMMIT}:{path}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout


def main() -> None:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", DECISION_COMMIT, "HEAD"],
        cwd=ROOT,
        check=True,
    )
    frozen = {path: selected_bytes(path) for path in EXPECTED_SHA256}
    for path, expected in EXPECTED_SHA256.items():
        if sha256(frozen[path]) != expected:
            raise ValueError(f"Nonvenue decision drift: {path}")
    source = json.loads(frozen[SOURCE_REPORT])
    decisions = json.loads(frozen[DECISIONS])
    queue_rows = list(csv.DictReader(io.StringIO(frozen[QUEUE].decode("utf-8"))))
    queue = {row["relative_path"]: row for row in queue_rows}
    if (
        source["schema"] != "thewiz.gate0.top_level_nonvenue_source_reconciliation.v1"
        or decisions["schema_version"] != "gate0.top_level_nonvenue_25_decisions.v1"
        or decisions["source_report_sha256"] != EXPECTED_SHA256[SOURCE_REPORT]
        or len(source["paths"]) != 36
        or len(decisions["selected"]) != decisions["selected_count"]
        or len(decisions["selected"]) != 25
        or len(decisions["held"]) != decisions["held_for_visual_variant_review_count"]
        or len(decisions["held"]) != 11
        or len(queue_rows) != len(queue)
        or len(queue) != 811
    ):
        raise ValueError("Nonvenue decision scope changed")
    scoped = {row["relative_path"]: row for row in source["paths"]}
    selected = {row["relative_path"]: row for row in decisions["selected"]}
    held = {row["relative_path"]: row for row in decisions["held"]}
    if len(scoped) != 36 or len(selected) != 25 or len(held) != 11 or set(scoped) != set(selected) | set(held):
        raise ValueError("Nonvenue path accounting changed")

    verified_copies = 0
    for path, item in scoped.items():
        for copy in item["all_extended_manifest_variants"]:
            candidate = Path(copy["absolute_path"])
            if (
                not candidate.is_file()
                or sha256(candidate.read_bytes()) != copy["manifest_sha256"]
                or copy["actual_sha256"] != copy["manifest_sha256"]
                or copy["exact_match"] is not True
            ):
                raise ValueError(f"Nonvenue preserved copy drift: {candidate}")
            verified_copies += 1
        row = queue[path]
        if path in selected:
            decision = selected[path]
            if (
                row["custody_status"] != decision["reviewed_status"]
                or row["decision_rationale"][: len(decision["rationale"])] != decision["rationale"]
                or DECISIONS not in row["decision_evidence"]
            ):
                raise ValueError(f"Nonvenue decision not recorded: {path}")
            active = subprocess.run(
                ["git", "cat-file", "-e", f"{DECISION_COMMIT}:{path}"],
                cwd=ROOT,
                capture_output=True,
            ).returncode == 0
            expected = decision["active_head_sha256_at_decision"]
            if active != bool(expected) or active and sha256(selected_bytes(path)) != expected:
                raise ValueError(f"Nonvenue selected source drift: {path}")
        else:
            decision = held[path]
            if row["custody_status"] != "PRESERVED_REVIEW_REQUIRED_NO_PORT":
                raise ValueError(f"Nonvenue historical variant silently decided: {path}")
            if not set(decision["additional_visual_sha256"]).issubset(
                set(row["historical_variant_sha256"].split(";"))
            ):
                raise ValueError(f"Nonvenue visual variant not in queue: {path}")
    if verified_copies != 116:
        raise ValueError("Nonvenue frozen copy count changed")
    print("PASS top_level_nonvenue_25_decisions: 25 reviewed, 11 open, 116 frozen copies")


if __name__ == "__main__":
    main()
