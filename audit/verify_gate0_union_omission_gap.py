#!/usr/bin/env python3
"""Verify source paths omitted by the first Gate 0 union queue.

The committed filtered manifest makes the finding portable within the source
checkout. If the full second-pass manifest is available, verify completeness
against it as well. This script never imports a historical source variant.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "589a168eaf20e8ff2da88afb12cf1e3e7e7d364d"
QUEUE = "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
BASELINE = "audit/evidence_freeze_2026-09-29/source_reconciliation.csv"
FILTERED = "audit/GATE0_UNION_OMITTED_VARIANT_INPUTS_2026-09-30.csv"
FULL = "audit/SECOND_PASS_EXTENDED_MANIFEST_2026-09-29.csv"
REPORT = ROOT / "audit/GATE0_UNION_OMISSION_GAP_2026-09-30.json"
BASE_QUEUE_SHA = "d5f77589e1c95aa85d352b82b0a642daa530b44b2ea0f3ca60523177fa63b282"
BASELINE_SHA = "cecdd2603471da26db24b826d0a6aafe1ab11b36823f014c1abfe02ec2406943"
FULL_SHA = "7042b2e8fd82ebeb4922d82f39d2823cc350ed38642ef1c734d4efd5e9237918"
SOURCE_PREFIXES = ("src/", "tests/", "scripts/", "config/")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def git_bytes(spec: str) -> bytes:
    return subprocess.run(
        ["git", "show", spec], cwd=ROOT, check=True, capture_output=True,
    ).stdout


def omitted_copies(
    manifest: list[dict[str, str]],
    original_paths: set[str],
    baseline: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    result = []
    for row in manifest:
        path = row["relative_path"]
        if (
            row["status"] != "hashed"
            or path in original_paths
            or path not in baseline
            or not path.startswith(SOURCE_PREFIXES)
        ):
            continue
        base = baseline[path]
        if row["sha256"] not in {
            base["working_sha256"], base["recovery_sha256"], base["runtime_sha256"]
        }:
            result.append(row)
    return sorted(result, key=lambda row: (row["relative_path"], row["root_label"]))


def build() -> dict[str, object]:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE, "HEAD"],
        cwd=ROOT, check=True,
    )
    original_bytes = git_bytes(f"{BASE}:{QUEUE}")
    baseline_bytes = (ROOT / BASELINE).read_bytes()
    filtered_bytes = (ROOT / FILTERED).read_bytes()
    if digest(original_bytes) != BASE_QUEUE_SHA or digest(baseline_bytes) != BASELINE_SHA:
        raise ValueError("frozen baseline or original queue changed")
    original = {row["relative_path"] for row in rows(original_bytes)}
    baseline = {row["relative_path"]: row for row in rows(baseline_bytes)}
    filtered = rows(filtered_bytes)
    expected = omitted_copies(filtered, original, baseline)
    if filtered != expected:
        raise ValueError("filtered manifest includes baseline copies, duplicate order, or queued paths")
    full_path = ROOT / FULL
    if full_path.is_file():
        full_bytes = full_path.read_bytes()
        if digest(full_bytes) != FULL_SHA:
            raise ValueError("full extended manifest hash changed")
        full_expected = omitted_copies(rows(full_bytes), original, baseline)
        if filtered != full_expected:
            raise ValueError("filtered manifest omits a source variant from the full manifest")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in filtered:
        grouped[row["relative_path"]].append(row)
    if len(grouped) != 80 or len(filtered) != 160:
        raise ValueError("unexpected omitted source path or copy count")
    current = {row["relative_path"]: row for row in rows((ROOT / QUEUE).read_bytes())}
    result = []
    for path, copies in sorted(grouped.items()):
        shas = {row["sha256"] for row in copies}
        roots = {row["root_label"] for row in copies}
        if len(shas) != 1 or roots != {"forensic_b", "visual_recovery"}:
            raise ValueError(f"historical variant cardinality changed: {path}")
        variant_sha = next(iter(shas))
        for copy in copies:
            frozen = Path(copy["absolute_path"])
            if not frozen.is_file() or frozen.is_symlink() or digest(frozen.read_bytes()) != variant_sha:
                raise ValueError(f"frozen source copy changed: {path} @ {copy['root_label']}")
        base = baseline[path]
        if base["working_vs_runtime"] != "SAME" or base["decision"] != "NO_PORT_NEEDED":
            raise ValueError(f"four-root baseline category changed: {path}")
        selected_sha = digest(git_bytes(f"{BASE}:{path}"))
        queued = current.get(path)
        if queued is None or (
            queued["in_same_path_drift"] != "True"
            or queued["in_four_root_reconciliation"] != "False"
            or queued["historical_variant_count"] != "1"
            or queued["historical_variant_sha256"] != variant_sha
            or set(queued["historical_copy_roots"].split(";")) != roots
            or REPORT.relative_to(ROOT).as_posix() not in queued["decision_evidence"].split("; ")
        ):
            raise ValueError(f"omitted path not fully represented in queue: {path}")
        result.append({
            "relative_path": path,
            "selected_sha256_at_base_commit": selected_sha,
            "four_root_baseline_sha256": {
                key: base[key] for key in
                ("working_sha256", "recovery_sha256", "runtime_sha256")
            },
            "historical_variant_sha256": variant_sha,
            "copies": [
                {"root": copy["root_label"], "absolute_path": copy["absolute_path"]}
                for copy in copies
            ],
        })
    return {
        "schema_version": "thewiz.gate0.union_omission_gap.v1",
        "base_commit": BASE,
        "original_queue_sha256": BASE_QUEUE_SHA,
        "four_root_baseline_sha256": BASELINE_SHA,
        "full_extended_manifest_sha256": FULL_SHA,
        "filtered_manifest_sha256": digest(filtered_bytes),
        "scope": "src, tests, scripts, config paths with hashes absent from the four-root baseline and original union queue",
        "omitted_paths": len(result),
        "distinct_historical_variants": len({row["historical_variant_sha256"] for row in result}),
        "frozen_copy_count": len(filtered),
        "selected_source_changes": 0,
        "rows": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    encoded = json.dumps(build(), indent=2, sort_keys=True) + "\n"
    if args.write_report:
        REPORT.write_text(encoded, encoding="utf-8")
    elif REPORT.read_text(encoding="utf-8") != encoded:
        raise ValueError("committed omission report differs from frozen source evidence")
    print("PASS union omission gap: 80 paths, 80 variants, 160 frozen copies")


if __name__ == "__main__":
    main()
