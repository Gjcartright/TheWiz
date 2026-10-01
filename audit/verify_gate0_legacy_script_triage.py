#!/usr/bin/env python3
"""Verify the frozen legacy crawler, Phase 9, and dashboard-builder triage."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "da40a10d03e5dd5e06381a672fecf967a121a4b7"
DECISION_COMMIT = "3b1b767ea078a0b2799d86a1bcaa84cafa6d6db1"
REPORT_PATH = "audit/GATE0_LEGACY_SCRIPT_TRIAGE_2026-09-30.json"
REPORT_SHA256 = "3c871a3a38dc01b7701ed9a8ccef5379e61de7941c5fa94a531778b79ead32b0"
QUEUE_PATH = "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
QUEUE_SHA256 = "e03029ac6dacdec3df662fa9ab7b4b28a91b1153ece4c2978a70e1ab541edd57"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selected_bytes(commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
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
    report_bytes = selected_bytes(DECISION_COMMIT, REPORT_PATH)
    queue_bytes = selected_bytes(DECISION_COMMIT, QUEUE_PATH)
    if sha256(report_bytes) != REPORT_SHA256 or sha256(queue_bytes) != QUEUE_SHA256:
        raise ValueError("legacy-script decision evidence drift")
    report = json.loads(report_bytes)
    if report["source_commit_before"] != BASE_COMMIT or len(report["scope"]) != 3:
        raise ValueError("legacy-script decision scope changed")
    rows = list(csv.DictReader(queue_bytes.decode("utf-8").splitlines()))
    queue = {row["relative_path"]: row for row in rows}
    if len(rows) != 811 or len(queue) != 811:
        raise ValueError("legacy-script queue identity changed")
    for path, item in report["scope"].items():
        preserved = Path(item["preserved_root"]) / path
        row = queue[path]
        if (item["base_commit"] != BASE_COMMIT
                or sha256(selected_bytes(BASE_COMMIT, path)) != item["base_source_sha256"]
                or sha256(selected_bytes(DECISION_COMMIT, path)) != item["selected_source_sha256"]
                or not preserved.is_file()
                or sha256(preserved.read_bytes()) != item["preserved_sha256"]
                or row["custody_status"] != item["decision"]
                or row["decision_rationale"] != item["rationale"]
                or row["decision_evidence"] != REPORT_PATH):
            raise ValueError(f"legacy-script source/queue drift: {path}")
        frozen = row["runtime_sha256_at_freeze"] or row["historical_variant_sha256"]
        if frozen != item["preserved_sha256"]:
            raise ValueError(f"legacy-script preserved hash mismatch: {path}")
    crawler = selected_bytes(DECISION_COMMIT, "scripts/crawl_crypto_wizards_with_curl.sh")
    phase9 = selected_bytes(DECISION_COMMIT, "scripts/run_evidence_pipeline_phase9_final_promotion.py")
    dashboard = selected_bytes(DECISION_COMMIT, "scripts/build_dashboard_full_inventory.py")
    if (b"exit 78" not in crawler
            or b"curl --fail" in crawler
            or b'"promotion_authority": False' not in phase9
            or b"legacy_snapshot_builder_disabled" not in dashboard):
        raise ValueError("legacy-script safety boundary changed")
    if sum(row["custody_status"] == "PRESERVED_REVIEW_REQUIRED_NO_PORT" for row in rows) != 538:
        raise ValueError("legacy-script queue pending count changed")
    print("PASS legacy_script_triage")


if __name__ == "__main__":
    main()
