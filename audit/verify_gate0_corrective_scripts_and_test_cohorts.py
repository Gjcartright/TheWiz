#!/usr/bin/env python3
"""Verify frozen corrective variants, legacy scripts, and missing-module tests."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "ff07b8b66c115e6a6d0ed3bb084b9df71ffc8483"
DECISION_COMMIT = "92e9792273f50602dce38546c7ec53ef1e0b7122"
QUEUE_PATH = "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
QUEUE_SHA256 = "53ee32c17fba8e8769a47ee26bb662d07c96a4895b06026924b1af3a6f4a3d0f"
REPORT_SHA256 = {
    "audit/GATE0_CORRECTIVE_COHORT_REVIEW_2026-09-30.json": "938909da7a21f3ebf1e4add5ec209d7025e04580b4f2e4cd5b32163a226996dd",
    "audit/GATE0_CORRECTIVE_HISTORICAL_VARIANTS_SUPPLEMENT_2026-09-30.json": "0a04ace10fd44a47c7a1830846fc66bde21cc975a8147b3c441409f48f5b62b7",
    "audit/GATE0_MISSING_MODULE_TEST_REVIEW_2026-09-30.json": "ef60bd8d028fc8987dae3bfcaafe8f3d02a9776c1b917d1b0808a8033ab91216",
    "audit/GATE0_REMAINING_SCRIPTS_SOURCE_RECONCILIATION_2026-09-30.json": "c0179c40bdcdf222804670a861d8326a85087f31a1d177d106a925162b771b02",
}
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selected_bytes(relative: str, commit: str = DECISION_COMMIT) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout


def exists_at(relative: str, commit: str = BASE_COMMIT) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    ).returncode == 0


def check_corrective(
    queue: dict[str, dict[str, str]], primary: dict, supplement: dict
) -> None:
    primary_path, supplement_path = list(REPORT_SHA256)[:2]
    if (primary["baseline_commit"] != BASE_COMMIT
            or supplement["baseline_commit"] != BASE_COMMIT
            or len(primary["files"]) != 57
            or len(supplement["files"]) != 23):
        raise ValueError("corrective cohort scope changed")
    variants = {item["path"]: item for item in supplement["files"]}
    seen = set()
    for item in primary["files"]:
        path = item["path"]
        row = queue[path]
        if path in seen or item["runtime_sha256"] != item["savepoint_sha256"]:
            raise ValueError(f"corrective cohort duplicate/hash mismatch: {path}")
        seen.add(path)
        saved = SAVEPOINT / path
        if not saved.is_file() or sha256(saved.read_bytes()) != item["runtime_sha256"]:
            raise ValueError(f"corrective savepoint drift: {path}")
        base_digest = sha256(selected_bytes(path, BASE_COMMIT)) if exists_at(path) else None
        if base_digest != item["active_sha256_at_baseline"]:
            raise ValueError(f"corrective base source drift: {path}")
        if row["runtime_sha256_at_freeze"] and row["runtime_sha256_at_freeze"] != item["runtime_sha256"]:
            raise ValueError(f"corrective frozen runtime drift: {path}")
        rationale = item["rationale"]
        evidence = primary_path
        if path in variants:
            historical = variants[path]
            frozen = set(row["historical_variant_sha256"].split(";"))
            if frozen != set(historical["frozen_historical_variant_sha256_members"]):
                raise ValueError(f"corrective historical hash set drift: {path}")
            for variant in historical["variants"]:
                if variant["sha256"] not in frozen:
                    raise ValueError(f"unlisted corrective variant: {path}")
                for copy in variant["copy_paths"]:
                    if not Path(copy).is_file() or sha256(Path(copy).read_bytes()) != variant["sha256"]:
                        raise ValueError(f"corrective historical copy drift: {copy}")
            rationale += " Historical variant: " + historical["rationale"]
            evidence += "; " + supplement_path
        elif row["historical_variant_sha256"]:
            raise ValueError(f"unreviewed corrective variant: {path}")
        if (row["custody_status"] != item["recommended_queue_status"]
                or row["decision_rationale"] != rationale
                or row["decision_evidence"] != evidence):
            raise ValueError(f"corrective queue mismatch: {path}")
    if set(variants) - seen:
        raise ValueError("corrective supplement contains out-of-scope path")
    candidate = primary["browser_auth_candidate"]
    browser = "src/quant_platform/orchestration/corrective_wizard_browser_auth.py"
    if sha256(selected_bytes(browser)) != candidate["source_sha256"]:
        raise ValueError("selected browser-auth guard drift")


def check_missing_module_tests(queue: dict[str, dict[str, str]], report: dict) -> None:
    if report["base_commit"] != "932108584a4ec4b5c9c5260fd310adb499e6600b" or len(report["records"]) != 65:
        raise ValueError("missing-module test cohort scope changed")
    seen = set()
    for item in report["records"]:
        path = item["relative_path"]
        row = queue[path]
        saved = SAVEPOINT / path
        if path in seen or not saved.is_file() or exists_at(path):
            raise ValueError(f"missing-module test selection drift: {path}")
        seen.add(path)
        data = saved.read_bytes()
        if (sha256(data) != item["preserved_sha256"]
                or row["runtime_sha256_at_freeze"] != item["preserved_sha256"]
                or row["historical_variant_sha256"]):
            raise ValueError(f"missing-module test frozen bytes drift: {path}")
        imports = set()
        for node in ast.walk(ast.parse(data, filename=path)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("quant_platform."):
                imports.add(node.module)
            elif isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names if alias.name.startswith("quant_platform."))
        if sorted(imports) != item["imported_quant_platform_modules"]:
            raise ValueError(f"missing-module test import map drift: {path}")
        for missing in item["missing_active_modules"]:
            if (missing["module"] not in imports
                    or exists_at(missing["expected_py"])
                    or exists_at(missing["expected_package_init"])):
                raise ValueError(f"missing-module test blocker drift: {path}")
        if (row["custody_status"] != item["decision"]
                or row["decision_rationale"] != item["rationale"]
                or row["decision_evidence"] != "audit/GATE0_MISSING_MODULE_TEST_REVIEW_2026-09-30.json"):
            raise ValueError(f"missing-module test queue mismatch: {path}")


def check_scripts(queue: dict[str, dict[str, str]], report: dict) -> None:
    if report["canonical_base_head"] != "da40a10d03e5dd5e06381a672fecf967a121a4b7" or len(report["paths"]) != 34:
        raise ValueError("legacy scripts review scope changed")
    seen = set()
    for item in report["paths"]:
        path = item["relative_path"]
        row = queue[path]
        saved = SAVEPOINT / path
        if path in seen or exists_at(path) or not saved.is_file():
            raise ValueError(f"legacy script selection drift: {path}")
        seen.add(path)
        if (sha256(saved.read_bytes()) != item["runtime_sha256"]
                or item["mac_savepoint_sha256"] != item["runtime_sha256"]
                or row["runtime_sha256_at_freeze"] != item["runtime_sha256"]):
            raise ValueError(f"legacy script frozen bytes drift: {path}")
        variant = item["source_candidate_distinct_variant"]
        rationale = item["reason"]
        if variant:
            if (variant["sha256"] != row["historical_variant_sha256"]
                    or not Path(variant["path"]).is_file()
                    or sha256(Path(variant["path"]).read_bytes()) != variant["sha256"]):
                raise ValueError(f"legacy script variant drift: {path}")
            rationale += " Historical variant: " + item["path_specific_note"]
        elif row["historical_variant_sha256"]:
            raise ValueError(f"unreviewed legacy script variant: {path}")
        if (row["custody_status"] != item["proposed_queue_status"]
                or row["decision_rationale"] != rationale
                or row["decision_evidence"] != "audit/GATE0_REMAINING_SCRIPTS_SOURCE_RECONCILIATION_2026-09-30.json"):
            raise ValueError(f"legacy script queue mismatch: {path}")


def main() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", DECISION_COMMIT, "HEAD"], cwd=ROOT, check=True)
    queue_bytes = selected_bytes(QUEUE_PATH)
    if sha256(queue_bytes) != QUEUE_SHA256:
        raise ValueError("decision queue drift")
    rows = list(csv.DictReader(queue_bytes.decode("utf-8").splitlines()))
    queue = {row["relative_path"]: row for row in rows}
    if len(rows) != 811 or len(queue) != 811:
        raise ValueError("decision queue identity changed")
    reports = {}
    for path, digest in REPORT_SHA256.items():
        data = selected_bytes(path)
        if sha256(data) != digest:
            raise ValueError(f"report drift: {path}")
        reports[path] = json.loads(data)
    check_corrective(queue, reports[list(REPORT_SHA256)[0]], reports[list(REPORT_SHA256)[1]])
    check_missing_module_tests(queue, reports[list(REPORT_SHA256)[2]])
    check_scripts(queue, reports[list(REPORT_SHA256)[3]])
    pending = sum(row["custody_status"] == "PRESERVED_REVIEW_REQUIRED_NO_PORT" for row in rows)
    partial = sum(row["custody_status"] == "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED" for row in rows)
    if (pending, partial) != (366, 2):
        raise ValueError("decision queue remaining count changed")
    print("PASS corrective_scripts_and_test_cohorts: 156 reviewed; 368 pending/partial")


if __name__ == "__main__":
    main()
