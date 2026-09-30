"""Join the two dated Gate 0 source-difference inventories by relative path.

This is a custody queue. It never decides that a historical file is safe to port.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
DELTA = AUDIT / "GATE0_SOURCE_DELTA_REGISTER_2026-09-30.csv"
ROOTS = AUDIT / "evidence_freeze_2026-09-29" / "source_reconciliation.csv"
OUTPUT = AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
SUMMARY = AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    delta = defaultdict(list)
    for row in read_rows(DELTA):
        if row["decision"] == "SEMANTIC_REVIEW_REQUIRED_NO_PORT":
            delta[row["relative_path"]].append(row)
    roots = {
        row["relative_path"]: row
        for row in read_rows(ROOTS)
        if row["decision"] == "REVIEW_REQUIRED"
    }
    paths = sorted(set(delta) | set(roots))
    fieldnames = [
        "relative_path",
        "in_same_path_drift",
        "in_four_root_reconciliation",
        "historical_variant_count",
        "historical_variant_sha256",
        "historical_copy_roots",
        "working_vs_runtime",
        "working_sha256_at_freeze",
        "recovery_sha256_at_freeze",
        "runtime_sha256_at_freeze",
        "custody_status",
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for path in paths:
            variants = delta.get(path, [])
            root = roots.get(path, {})
            writer.writerow(
                {
                    "relative_path": path,
                    "in_same_path_drift": bool(variants),
                    "in_four_root_reconciliation": bool(root),
                    "historical_variant_count": len(variants),
                    "historical_variant_sha256": ";".join(
                        sorted(row["copy_sha256"] for row in variants)
                    ),
                    "historical_copy_roots": ";".join(
                        sorted(
                            {
                                copy_root
                                for row in variants
                                for copy_root in row["copy_roots"].split(";")
                            }
                        )
                    ),
                    "working_vs_runtime": root.get("working_vs_runtime", ""),
                    "working_sha256_at_freeze": root.get("working_sha256", ""),
                    "recovery_sha256_at_freeze": root.get("recovery_sha256", ""),
                    "runtime_sha256_at_freeze": root.get("runtime_sha256", ""),
                    "custody_status": "PRESERVED_REVIEW_REQUIRED_NO_PORT",
                }
            )
    overlap = set(delta) & set(roots)
    summary = {
        "schema_version": "thewiz.gate0.union_source_queue.v1",
        "snapshot_date": "2026-09-30",
        "source_registers": [str(DELTA.relative_to(AUDIT)), str(ROOTS.relative_to(AUDIT))],
        "same_path_review_variants": sum(map(len, delta.values())),
        "same_path_review_paths": len(delta),
        "four_root_review_paths": len(roots),
        "overlap_paths": len(overlap),
        "same_path_only": len(set(delta) - set(roots)),
        "four_root_only": len(set(roots) - set(delta)),
        "union_review_paths": len(paths),
        "meaning": "Custody and queue counts only; semantic decisions and ports remain open.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
