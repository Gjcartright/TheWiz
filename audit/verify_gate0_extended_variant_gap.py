#!/usr/bin/env python3
"""Verify the frozen extended-copy census and the reopened Gate 0 queue."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DECISION_COMMIT = "99df4d1f3b3f34a6c6c50c2d111a2344c05c6a61"
REPORT = "audit/GATE0_EXTENDED_VARIANT_GAP_2026-09-30.json"
FILTERED = "audit/GATE0_EXTENDED_VARIANT_INPUTS_2026-09-30.csv"
BASELINE = "audit/evidence_freeze_2026-09-29/source_reconciliation.csv"
QUEUE = "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
EXPECTED_SHA256 = {
    REPORT: "843abe2a6f4c2234ee1acad4d78c53e13e4ebbc9ef375a402dc6ef1daf136383",
    FILTERED: "ca055e5cfbd6c5cdcd1e42b712b394aac2701bcf01a173fef151248e9eb046b3",
    BASELINE: "cecdd2603471da26db24b826d0a6aafe1ab11b36823f014c1abfe02ec2406943",
    QUEUE: "6e0beb2f87635f5e7af344519a333ef06b9da8c5ace6175e911380b79791b9b5",
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


def csv_rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def main() -> None:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", DECISION_COMMIT, "HEAD"],
        cwd=ROOT,
        check=True,
    )
    snapshot = {path: selected_bytes(path) for path in EXPECTED_SHA256}
    for path, expected in EXPECTED_SHA256.items():
        if sha256(snapshot[path]) != expected:
            raise ValueError(f"Gate 0 extended census drift: {path}")

    report = json.loads(snapshot[REPORT])
    queue = {row["relative_path"]: row for row in csv_rows(snapshot[QUEUE])}
    baseline = {row["relative_path"]: row for row in csv_rows(snapshot[BASELINE])}
    copies = csv_rows(snapshot[FILTERED])
    if (
        report["schema_version"] != "gate0.extended_variant_gap.v1"
        or len(queue) != report["queue_paths"]
        or len(queue) != 811
        or len(copies) != report["filtered_manifest_rows"]
        or len(copies) != 1412
        or len(report["rows"]) != report["unlisted_variant_paths"]
        or len(report["rows"]) != 103
        or report["unlisted_variant_sha256_count"] != 103
        or report["prior_reviewed_rows_to_reopen"] != 43
        or report["prior_pending_rows"] != 60
        or report["filtered_manifest_sha256"] != EXPECTED_SHA256[FILTERED]
        or report["four_root_source_reconciliation_sha256"] != EXPECTED_SHA256[BASELINE]
    ):
        raise ValueError("Gate 0 extended census scope changed")

    missing = {item["relative_path"]: item for item in report["rows"]}
    if len(missing) != 103:
        raise ValueError("Duplicate missing-variant path")
    copy_index: dict[tuple[str, str], list[dict[str, str]]] = {}
    for copy in copies:
        if copy["status"] != "hashed" or copy["relative_path"] not in queue:
            raise ValueError("Unexpected extended-manifest copy")
        path = Path(copy["absolute_path"])
        if not path.is_file() or sha256(path.read_bytes()) != copy["sha256"]:
            raise ValueError(f"Frozen extended copy drift: {path}")
        copy_index.setdefault((copy["relative_path"], copy["sha256"]), []).append(copy)

    reopened = 0
    for path, row in queue.items():
        base = baseline.get(path, {})
        baseline_hashes = {
            base.get(key, "")
            for key in ("working_sha256", "recovery_sha256", "runtime_sha256")
        }
        queue_hashes = {
            row[key]
            for key in (
                "working_sha256_at_freeze",
                "recovery_sha256_at_freeze",
                "runtime_sha256_at_freeze",
            )
        }
        historical = set(filter(None, row["historical_variant_sha256"].split(";")))
        if len(historical) != int(row["historical_variant_count"]):
            raise ValueError(f"Historical variant count mismatch: {path}")
        for (copy_path, copy_hash) in copy_index:
            if copy_path == path and copy_hash not in baseline_hashes | queue_hashes | historical:
                raise ValueError(f"Unclassified frozen variant: {path}")
        item = missing.get(path)
        if item is None:
            continue
        variants = item["unlisted_variants"]
        if len(variants) != 1 or variants[0]["sha256"] not in historical:
            raise ValueError(f"Gap variant not recorded: {path}")
        variant = variants[0]
        indexed = copy_index.get((path, variant["sha256"]), [])
        if {
            (copy["root_label"], copy["absolute_path"], copy["sha256"])
            for copy in indexed
        } != {
            (copy["root"], copy["absolute_path"], copy["frozen_sha256"])
            for copy in variant["copies"]
        }:
            raise ValueError(f"Gap copy custody mismatch: {path}")
        if (
            int(row["historical_variant_count"])
            != int(item["prior_historical_variant_count"]) + 1
            or "audit/GATE0_EXTENDED_VARIANT_GAP_2026-09-30.json"
            not in row["decision_evidence"]
        ):
            raise ValueError(f"Gap decision mismatch: {path}")
        if item["prior_custody_status"] != "PRESERVED_REVIEW_REQUIRED_NO_PORT":
            reopened += 1
            if row["custody_status"] != "PRESERVED_ADDITIONAL_VARIANT_REVIEW_REQUIRED":
                raise ValueError(f"Reviewed path was not reopened: {path}")
        elif row["custody_status"] != "PRESERVED_REVIEW_REQUIRED_NO_PORT":
            raise ValueError(f"Pending path was silently decided: {path}")
    if reopened != 43:
        raise ValueError("Reopened path accounting changed")
    print("PASS extended_variant_gap: 1412 frozen copies, 103 new variants, 43 reopened")


if __name__ == "__main__":
    main()
