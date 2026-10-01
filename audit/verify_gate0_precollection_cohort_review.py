#!/usr/bin/env python3
"""Verify the historical precollection family was reviewed without activation."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISION_COMMIT = "d8a1efc236a94fa7ad210758c8836573859c0dba"
REPORT_PATH = "audit/GATE0_PRECOLLECTION_COHORT_REVIEW_2026-09-30.json"
REPORT_SHA256 = "f553adf9f43fc94a3edd76615f705f462478f082f703e78b8ba9544a321f940c"
QUEUE_PATH = "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
QUEUE_SHA256 = "38c92c2363febe074fa43d6a57557e68d30f28f54ea9b0f5262eb4db4d6ebc90"
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")
RECOVERY = Path("/Volumes/Expansion/TheWiz-Workspace/project")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selected_bytes(relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{DECISION_COMMIT}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout


def module_imports(data: bytes) -> list[str]:
    imports = set()
    for node in ast.walk(ast.parse(data)):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("quant_platform."):
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names if alias.name.startswith("quant_platform."))
    return sorted(imports)


def exists_at(relative: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{DECISION_COMMIT}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    ).returncode == 0


def main() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", DECISION_COMMIT, "HEAD"], cwd=ROOT, check=True)
    report_bytes = selected_bytes(REPORT_PATH)
    queue_bytes = selected_bytes(QUEUE_PATH)
    if sha256(report_bytes) != REPORT_SHA256 or sha256(queue_bytes) != QUEUE_SHA256:
        raise ValueError("precollection evidence or decision queue drift")
    report = json.loads(report_bytes)
    if (report["base_commit"] != "2d5ebb7506f9b8929ffdb53302f7619ff359cd16"
            or report["decision"] != "PRESERVE_PRECOLLECTION_COHORT_NO_GATE0_PORT"
            or len(report["records"]) != 17):
        raise ValueError("precollection review scope changed")
    rows = list(csv.DictReader(queue_bytes.decode("utf-8").splitlines()))
    queue = {row["relative_path"]: row for row in rows}
    if len(rows) != 811 or len(queue) != 811:
        raise ValueError("precollection queue identity changed")
    active_matches = subprocess.run(
        ["git", "grep", "-n", "precollection_", DECISION_COMMIT, "--", "src", "scripts", "tests", "config"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if active_matches.returncode != 1 or active_matches.stdout:
        raise ValueError("precollection acquired an active consumer at decision commit")
    seen = set()
    for item in report["records"]:
        path = item["relative_path"]
        row = queue[path]
        if path in seen or exists_at(path):
            raise ValueError(f"precollection active/duplicate path: {path}")
        seen.add(path)
        saved = SAVEPOINT / path
        if (not saved.is_file()
                or sha256(saved.read_bytes()) != item["runtime_sha256"]
                or row["runtime_sha256_at_freeze"] != item["runtime_sha256"]
                or row["recovery_sha256_at_freeze"] != item["recovery_sha256_at_freeze"]
                or module_imports(saved.read_bytes()) != item["quant_platform_imports"]):
            raise ValueError(f"precollection runtime custody drift: {path}")
        recovery = RECOVERY / path
        if item["recovery_sha256_at_freeze"]:
            if not recovery.is_file() or sha256(recovery.read_bytes()) != item["recovery_sha256_at_freeze"]:
                raise ValueError(f"precollection recovery custody drift: {path}")
        elif recovery.exists():
            raise ValueError(f"unexpected precollection recovery path: {path}")
        variant = item["historical_variant"]
        if variant:
            historical = Path(variant["path"])
            if (row["historical_variant_sha256"] != variant["sha256"]
                    or not historical.is_file()
                    or sha256(historical.read_bytes()) != variant["sha256"]):
                raise ValueError(f"precollection historical variant drift: {path}")
        elif row["historical_variant_sha256"]:
            raise ValueError(f"unreviewed precollection variant: {path}")
        if (row["custody_status"] != item["recommended_status"]
                or row["decision_rationale"] != item["rationale"]
                or row["decision_evidence"] != REPORT_PATH):
            raise ValueError(f"precollection decision mismatch: {path}")
    if (len(seen) != 17
            or sum(row["custody_status"] == "PRESERVED_REVIEW_REQUIRED_NO_PORT" for row in rows) != 349):
        raise ValueError("precollection queue accounting changed")
    print("PASS precollection_cohort_review")


if __name__ == "__main__":
    main()
