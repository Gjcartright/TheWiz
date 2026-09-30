#!/usr/bin/env python3
"""Verify the Gate 0 dependency and wheel-build source selection."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30")
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
VARIANTS = Path("/Users/gregc/Backups/TheWiz/source-delta-evidence/2026-09-30")
DIAGNOSTIC = Path(
    "/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30/wheel-probe"
)
OUTPUT = AUDIT / "GATE0_DEPENDENCY_LOCK_DECISION_2026-09-30.json"
PROJECT_VARIANTS = (
    "a311ff77f4662dc08089616b7ec07a0dea4a4d214fc1ff94d7340ee59f56b9af",
    "be967304aa53051a530bbeec7aeab41253ce160176bbdafae02c7698f709fd1d",
)
LOCK_VARIANT = "f203179ba6e53f36ca60ddf50287587a740a5b06e30303ed21c8152bc2d381d9"
MANIFEST_STATUS = "REVIEWED_ACTIVE_LOCK_RETAIN_HISTORICAL_NO_PORT"
BUILD_STATUS = "REVIEWED_REPLACED_BY_COMMIT_BOUND_WHEEL_BUILDER_NO_PORT"
BUILD_PATHS = (
    "scripts/build_clean_wheel.py",
    "src/quant_platform/package_build.py",
    "tests/test_package_build.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parsed(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def package_set(lock: dict) -> set[str]:
    return {f"{package['name']}=={package['version']}" for package in lock["package"]}


def main() -> None:
    with (AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        queue = {row["relative_path"]: row for row in csv.DictReader(stream)}
    selected = {"pyproject.toml": MANIFEST_STATUS, "uv.lock": MANIFEST_STATUS}
    selected.update(dict.fromkeys(BUILD_PATHS, BUILD_STATUS))
    for relative, status in selected.items():
        row = queue[relative]
        if row["custody_status"] != status or OUTPUT.name not in row["decision_evidence"]:
            raise ValueError(f"source queue decision drift: {relative}")

    saved_manifest = json.loads((SAVEPOINT / "SAVEPOINT_MANIFEST.json").read_text())
    saved_hashes = {
        entry["path"]: entry["sha256"]
        for entry in saved_manifest["entries"]
        if entry["type"] == "file"
    }
    custody = {}
    for relative in ("pyproject.toml", "uv.lock"):
        active_hash = digest(ROOT / relative)
        runtime_hash = digest(RUNTIME / relative)
        if active_hash != saved_hashes[f"project/{relative}"]:
            raise ValueError(f"active savepoint mismatch: {relative}")
        if runtime_hash != saved_hashes[f"local_runtime/{relative}"]:
            raise ValueError(f"runtime savepoint mismatch: {relative}")
        if digest(SAVEPOINT / "project" / relative) != active_hash:
            raise ValueError(f"saved active bytes mismatch: {relative}")
        if digest(SAVEPOINT / "local_runtime" / relative) != runtime_hash:
            raise ValueError(f"saved runtime bytes mismatch: {relative}")
        custody[relative] = {"active_sha256": active_hash, "runtime_sha256": runtime_hash}

    variant_manifest = json.loads((VARIANTS / "MANIFEST.json").read_text())
    variant_entries = {
        (entry["relative_path"], entry["sha256"]): entry
        for entry in variant_manifest["entries"]
    }
    for relative, hashes in (("pyproject.toml", PROJECT_VARIANTS), ("uv.lock", (LOCK_VARIANT,))):
        row_hashes = queue[relative]["historical_variant_sha256"].split(";")
        if row_hashes != list(hashes):
            raise ValueError(f"variant queue drift: {relative}")
        for value in hashes:
            entry = variant_entries[(relative, value)]
            if digest(VARIANTS / entry["stored_as"]) != value:
                raise ValueError(f"preserved variant bytes mismatch: {relative} {value}")
        custody[relative]["historical_variant_sha256"] = list(hashes)

    for relative in BUILD_PATHS:
        row = queue[relative]
        if (ROOT / relative).exists() or row["working_vs_runtime"] != "RUNTIME_ONLY":
            raise ValueError(f"runtime-only package build path became active: {relative}")
        candidate_hash = digest(RUNTIME / relative)
        if candidate_hash != row["runtime_sha256_at_freeze"]:
            raise ValueError(f"candidate build source drift: {relative}")
        if digest(SAVEPOINT / "local_runtime" / relative) != candidate_hash:
            raise ValueError(f"candidate build savepoint drift: {relative}")
        if saved_hashes[f"local_runtime/{relative}"] != candidate_hash:
            raise ValueError(f"candidate build manifest drift: {relative}")
        custody[relative] = {"runtime_sha256": candidate_hash}

    active_project = parsed(ROOT / "pyproject.toml")
    runtime_project = parsed(RUNTIME / "pyproject.toml")
    historical_projects = [parsed(VARIANTS / "variants" / value) for value in PROJECT_VARIANTS]
    active_dependencies = active_project["project"]["dependencies"]
    if "scipy>=1.11,<1.18" not in active_dependencies or "ccxt==4.4.26" not in active_dependencies:
        raise ValueError("active numerical or exchange client contract changed")
    if not any("dydx-v4-client" in dep for dep in active_project["project"]["optional-dependencies"]["dev"]):
        raise ValueError("active development adapter contract changed")
    if any(
        not any(dep.startswith("scipy==1.18.0;") for dep in project["project"]["dependencies"])
        for project in historical_projects
    ):
        raise ValueError("historical SciPy resolution changed")
    if "exclude" not in historical_projects[0]["tool"]["setuptools"]["packages"]["find"]:
        raise ValueError("historical package-discovery exclusion changed")
    if "exclude" in historical_projects[1]["tool"]["setuptools"]["packages"]["find"]:
        raise ValueError("second historical project variant changed")
    if "math-research" not in runtime_project["project"]["optional-dependencies"]:
        raise ValueError("runtime Math Desk dependency profile changed")

    active_lock = package_set(parsed(ROOT / "uv.lock"))
    historical_lock = package_set(parsed(VARIANTS / "variants" / LOCK_VARIANT))
    runtime_lock = package_set(parsed(RUNTIME / "uv.lock"))
    if (len(active_lock), len(historical_lock), len(runtime_lock)) != (140, 129, 157):
        raise ValueError("lock package cardinality changed")
    if historical_lock - active_lock != {"scipy==1.18.0"}:
        raise ValueError("historical lock difference changed")
    if active_lock - runtime_lock or len(runtime_lock - active_lock) != 17:
        raise ValueError("runtime optional dependency difference changed")
    subprocess.run(["uv", "lock", "--check", "--offline"], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    wheel_diagnostic = json.loads((DIAGNOSTIC / "DIAGNOSTIC.json").read_text())
    runs = wheel_diagnostic["runs"]
    if (runs["expansion_clean"]["result"] != "FAIL"
        or runs["mac_clean"]["result"] != "PASS"
        or runs["staged_from_expansion_commit"]["result"] != "PASS"):
        raise ValueError("filesystem build control changed")
    for name in ("mac_clean", "mac_appledouble", "staged_from_expansion_commit"):
        if digest(Path(runs[name]["wheel"])) != runs[name]["wheel_sha256"]:
            raise ValueError(f"wheel receipt mismatch: {name}")
    if digest(Path(runs["expansion_clean"]["build_stderr"])) != runs["expansion_clean"]["build_stderr_sha256"]:
        raise ValueError("Expansion failure log drift")
    junit_path = DIAGNOSTIC / "candidate-package-build-junit.xml"
    suite = next(ET.parse(junit_path).getroot().iter("testsuite"))
    if any(suite.attrib[key] != value for key, value in (
        ("tests", "2"), ("failures", "0"), ("errors", "0"))):
        raise ValueError("candidate build tests did not pass")
    helper = (ROOT / "scripts/ops/build_wheel_from_commit.py").read_text()
    for marker in ("git\", \"archive", "--offline", "extractall", "package.testzip()",
                   "uv\", \"pip", "import quant_platform"):
        if marker not in helper:
            raise ValueError(f"active commit-bound build helper changed: {marker}")

    report = {
        "schema_version": "thewiz.gate0.dependency_lock_decision.v1",
        "decision": "RETAIN_ACTIVE_PROJECT_AND_LOCK_USE_COMMIT_BOUND_INTERNAL_WHEEL_STAGE",
        "custody": dict(sorted(custody.items())),
        "active_lock_packages": len(active_lock),
        "historical_lock_packages": len(historical_lock),
        "runtime_lock_packages": len(runtime_lock),
        "historical_lock_only": sorted(historical_lock - active_lock),
        "active_lock_only_vs_historical": sorted(active_lock - historical_lock),
        "runtime_lock_only_vs_active": sorted(runtime_lock - active_lock),
        "wheel_diagnostic_path": str(DIAGNOSTIC / "DIAGNOSTIC.json"),
        "wheel_diagnostic_sha256": digest(DIAGNOSTIC / "DIAGNOSTIC.json"),
        "candidate_package_build_junit_sha256": digest(junit_path),
        "candidate_package_build_tests_passed": 2,
        "interpretation": "Retain the active dependency manifest and lock. The preserved manifest/lock pair resolves SciPy 1.18 for Python 3.12 and omits active ccxt dependencies. The runtime Math Desk and collection extras belong to unselected source. A clean Expansion wheel build fails at setuptools install_scripts, while the same committed source succeeds on Mac internal staging; the historical package-discovery exclude did not establish a fix. Use the commit-bound staged builder and keep the three runtime-only helper/test files as exact preserved candidates.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS five_dependency_and_build_source_decisions")


if __name__ == "__main__":
    main()
