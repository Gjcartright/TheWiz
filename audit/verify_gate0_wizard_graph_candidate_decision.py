#!/usr/bin/env python3
"""Verify the seven-file Wizard validation stage candidate remains unselected."""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30")
VARIANTS = Path("/Users/gregc/Backups/TheWiz/source-delta-evidence/2026-09-30")
DIAGNOSTIC = Path(
    "/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30/"
    "wizard-stage-candidate-ykzl2pf4"
)
OUTPUT = AUDIT / "GATE0_WIZARD_GRAPH_CANDIDATE_DECISION_2026-09-30.json"
NODES = "src/quant_platform/orchestration/nodes.py"
RUNTIME_ONLY = (
    "docs/wizard_frozen_source_qualification.md",
    "docs/wizard_research_validation_spine.md",
    "src/quant_platform/orchestration/wizard_frozen_source_qualification.py",
    "src/quant_platform/orchestration/wizard_research_validation_spine.py",
    "tests/test_wizard_frozen_source_qualification.py",
    "tests/test_wizard_research_validation_spine.py",
)
VARIANT_HASH = "4eb064001bf1c011bdf6666f4d74153bdcf990f60bc4f61c02e869270e35f1ff"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suite(path: Path) -> dict[str, str]:
    return next(ET.parse(path).getroot().iter("testsuite")).attrib


def main() -> None:
    with (AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        queue = {row["relative_path"]: row for row in csv.DictReader(stream)}
    saved_manifest = json.loads((SAVEPOINT / "SAVEPOINT_MANIFEST.json").read_text())
    saved_hashes = {
        entry["path"]: entry["sha256"]
        for entry in saved_manifest["entries"] if entry["type"] == "file"
    }
    hashes = {}
    for relative in (NODES, *RUNTIME_ONLY):
        row = queue[relative]
        if OUTPUT.name not in row["decision_evidence"]:
            raise ValueError(f"Wizard stage decision missing: {relative}")
        runtime_hash = digest(RUNTIME / relative)
        if runtime_hash != row["runtime_sha256_at_freeze"]:
            raise ValueError(f"Wizard stage runtime source drift: {relative}")
        if runtime_hash != saved_hashes[f"local_runtime/{relative}"]:
            raise ValueError(f"Wizard stage savepoint manifest drift: {relative}")
        if digest(SAVEPOINT / "local_runtime" / relative) != runtime_hash:
            raise ValueError(f"Wizard stage saved bytes drift: {relative}")
        if relative == NODES:
            if row["custody_status"] != "REVIEWED_ACTIVE_STAGE_MAP_RETAIN_WIZARD_CANDIDATE_NO_PORT":
                raise ValueError("active stage map decision drift")
            active_hash = digest(ROOT / relative)
            if (active_hash != row["working_sha256_at_freeze"]
                or active_hash != saved_hashes[f"project/{relative}"]):
                raise ValueError("active stage map drift")
            hashes[relative] = {"active_sha256": active_hash, "runtime_sha256": runtime_hash}
        else:
            if (row["custody_status"] != "REVIEWED_EVIDENCE_GATED_WIZARD_VALIDATION_NO_PORT"
                or row["working_vs_runtime"] != "RUNTIME_ONLY" or (ROOT / relative).exists()):
                raise ValueError(f"unselected Wizard source became active: {relative}")
            hashes[relative] = {"runtime_sha256": runtime_hash}

    variant_manifest = json.loads((VARIANTS / "MANIFEST.json").read_text())
    entry = next(
        item for item in variant_manifest["entries"]
        if item["relative_path"] == NODES and item["sha256"] == VARIANT_HASH
    )
    preserved = VARIANTS / entry["stored_as"]
    if digest(preserved) != VARIANT_HASH or queue[NODES]["historical_variant_sha256"] != VARIANT_HASH:
        raise ValueError("historical node stage variant drift")
    historical = preserved.read_text(encoding="utf-8")
    active = (ROOT / NODES).read_text(encoding="utf-8")
    candidate = (RUNTIME / NODES).read_text(encoding="utf-8")
    if ('reason=str(exc)' not in historical or 'safe_exception_code(exc)' not in active
        or 'wizard_frozen_source_qualification' not in candidate
        or 'wizard_research_validation_spine' not in candidate
        or 'wizard_frozen_source_qualification' in active):
        raise ValueError("stage-map safety distinction changed")
    hashes[NODES]["historical_variant_sha256"] = VARIANT_HASH

    for relative in RUNTIME_ONLY[:2]:
        if "no" not in (RUNTIME / relative).read_text(encoding="utf-8").lower():
            raise ValueError(f"candidate guide changed: {relative}")
    source_code = [
        (RUNTIME / relative).read_text(encoding="utf-8")
        for relative in RUNTIME_ONLY[2:4]
    ]
    for source in source_code:
        if ("Literal[False]" not in source or "atomic_write_text" not in source
            or "execution_allowed" not in source or "collection_authority" not in source):
            raise ValueError("candidate zero-authority or publication contract changed")
    for relative in (
        "data/research/wizard_source_qualification/INPUT.json",
        "data/research/wizard_validation_spine/CURRENT.json",
    ):
        if (ROOT / relative).exists():
            raise ValueError(f"unreviewed active Wizard input appeared: {relative}")

    diagnostic = json.loads((DIAGNOSTIC / "DIAGNOSTIC.json").read_text())
    if diagnostic["active_base_commit"] != "c2113c0d05e77bae19d992e872130af4a8dee57a":
        raise ValueError("candidate active base changed")
    results = diagnostic["test_results"]
    for label, expected in (
        ("runtime", (24, 0, 0)),
        ("active_baseline", (24, 2, 0)),
        ("active_adjusted_fixture", (24, 0, 0)),
    ):
        recorded = results[label]
        if (recorded["tests"], recorded["failures"], recorded["errors"]) != expected:
            raise ValueError(f"candidate diagnostic result drift: {label}")
        path = Path(recorded["junit_path"])
        if digest(path) != recorded["junit_sha256"]:
            raise ValueError(f"candidate JUnit hash drift: {label}")
        actual = suite(path)
        if tuple(int(actual[key]) for key in ("tests", "failures", "errors")) != expected:
            raise ValueError(f"candidate JUnit content drift: {label}")
    if (digest(DIAGNOSTIC / "repo/tests/conftest.py") != diagnostic["fixture_original_sha256"]
        or diagnostic["fixture_original_sha256"] != diagnostic["fixture_restored_sha256"]):
        raise ValueError("disposable fixture was not restored")
    copied = json.loads((DIAGNOSTIC / "SOURCE_RECEIPT.json").read_text())
    for relative, value in copied["copied_candidate_paths"].items():
        if digest(RUNTIME / relative) != value or digest(DIAGNOSTIC / "repo" / relative) != value:
            raise ValueError(f"candidate copy hash drift: {relative}")

    report = {
        "schema_version": "thewiz.gate0.wizard_graph_candidate_decision.v1",
        "decision": "RETAIN_ACTIVE_GRAPH_STAGE_MAP_PRESERVE_EVIDENCE_GATED_WIZARD_CANDIDATE",
        "path_sha256": dict(sorted(hashes.items())),
        "diagnostic_path": str(DIAGNOSTIC / "DIAGNOSTIC.json"),
        "diagnostic_sha256": digest(DIAGNOSTIC / "DIAGNOSTIC.json"),
        "runtime_candidate_tests_passed": 24,
        "active_disposable_baseline_passed": 22,
        "active_disposable_baseline_failed": 2,
        "active_disposable_adjusted_tests_passed": 24,
        "interpretation": "Keep the active 25-stage graph and its safe exception redaction. Preserve the six runtime-only source, doc and test files and the alternate nodes.py bytes without partial promotion. Candidate modules are local-only and encode zero collection, promotion, paper and execution authority; they pass 24 tests under a disposable active fixture adjustment. The active checkout has neither required input file. Independent source evidence and a versioned current-checkout fixture/integration decision are needed before enabling these stages.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS seven_wizard_graph_candidate_source_decisions")


if __name__ == "__main__":
    main()
