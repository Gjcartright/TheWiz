#!/usr/bin/env python3
"""Verify the retained LangGraph guide and atomic publication source."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30")
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
VARIANTS = Path("/Users/gregc/Backups/TheWiz/source-delta-evidence/2026-09-30")
HEAD_AT_PROBE = "9099a2d39322ed4125b7801e8a072a87db270035"
DIAGNOSTIC = Path("/Users/gregc/Backups/TheWiz/gate0-checkpoints") / HEAD_AT_PROBE
OUTPUT = AUDIT / "GATE0_LANGGRAPH_CORE_DECISION_2026-09-30.json"
DOC = "docs/langgraph_agent_workflow.md"
WORKFLOW = "src/quant_platform/orchestration/langgraph_workflow.py"
ASSISTANT = "src/quant_platform/orchestration/orchestrator_assistant.py"
VARIANT_HASHES = {
    WORKFLOW: "ea43d042cf6cec5238d241c885db9f813db9e287167ce5080ddfd5a9e08bde78",
    ASSISTANT: "673f8f1bd8cd72beabdab753bf2cefb2e07da36422e69f1d7a4c014a99971e30",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with (AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        queue = {row["relative_path"]: row for row in csv.DictReader(stream)}
    for relative in (DOC, WORKFLOW, ASSISTANT):
        row = queue[relative]
        if not row["custody_status"].startswith("REVIEWED_") or OUTPUT.name not in row["decision_evidence"]:
            raise ValueError(f"LangGraph core queue drift: {relative}")

    saved = json.loads((SAVEPOINT / "SAVEPOINT_MANIFEST.json").read_text())
    saved_hashes = {
        entry["path"]: entry["sha256"]
        for entry in saved["entries"]
        if entry["type"] == "file"
    }
    variant_manifest = json.loads((VARIANTS / "MANIFEST.json").read_text())
    variant_entries = {
        (entry["relative_path"], entry["sha256"]): entry
        for entry in variant_manifest["entries"]
    }
    custody = {}
    for relative in (DOC, WORKFLOW, ASSISTANT):
        active = ROOT / relative
        runtime = RUNTIME / relative
        active_hash = digest(active)
        runtime_hash = digest(runtime)
        # Historical-only queue rows do not carry four-root freeze hashes.
        if (queue[relative]["in_four_root_reconciliation"] == "True"
            and runtime_hash != queue[relative]["runtime_sha256_at_freeze"]):
            raise ValueError(f"runtime freeze changed: {relative}")
        if runtime_hash != saved_hashes[f"local_runtime/{relative}"]:
            raise ValueError(f"runtime savepoint changed: {relative}")
        if digest(SAVEPOINT / "local_runtime" / relative) != runtime_hash:
            raise ValueError(f"runtime saved bytes changed: {relative}")
        if digest(SAVEPOINT / "project" / relative) != saved_hashes[f"project/{relative}"]:
            raise ValueError(f"active saved bytes changed: {relative}")
        if relative != DOC and (active_hash != runtime_hash or active_hash != saved_hashes[f"project/{relative}"]):
            raise ValueError(f"retained active/runtime graph source changed: {relative}")
        custody[relative] = {
            "active_sha256": active_hash,
            "runtime_sha256": runtime_hash,
            "saved_project_sha256": saved_hashes[f"project/{relative}"],
        }
        if relative in VARIANT_HASHES:
            value = VARIANT_HASHES[relative]
            if queue[relative]["historical_variant_sha256"] != value:
                raise ValueError(f"variant queue changed: {relative}")
            preserved = VARIANTS / variant_entries[(relative, value)]["stored_as"]
            if digest(preserved) != value:
                raise ValueError(f"variant source changed: {relative}")
            custody[relative]["historical_variant_sha256"] = value

    if queue[DOC]["working_sha256_at_freeze"] != saved_hashes[f"project/{DOC}"]:
        raise ValueError("guide freeze identity changed")
    guide = (ROOT / DOC).read_text(encoding="utf-8")
    old_guide = (RUNTIME / DOC).read_text(encoding="utf-8")
    if "scripts/ops/run_langgraph_dry_run.py" not in guide or "./.venv311/bin/python" not in old_guide:
        raise ValueError("graph invocation guide changed")
    for relative in (WORKFLOW, ASSISTANT):
        active = (ROOT / relative).read_text(encoding="utf-8")
        old = (VARIANTS / "variants" / VARIANT_HASHES[relative]).read_text(encoding="utf-8")
        if "atomic_write" not in active or "strict_bool" not in old or "strict_bool" in active:
            raise ValueError(f"atomic/boolean variant distinction changed: {relative}")
    if "def strict_bool" in (ROOT / "src/quant_platform/runtime_types.py").read_text():
        raise ValueError("strict_bool became active; reassess historical variants")

    dry_run = json.loads((DIAGNOSTIC / "LANGGRAPH_DRY_RUN_RECEIPT.json").read_text())
    if dry_run["source_commit"] != HEAD_AT_PROBE:
        raise ValueError("graph diagnostic commit changed")
    subprocess.run(["git", "merge-base", "--is-ancestor", HEAD_AT_PROBE, "HEAD"],
                   cwd=ROOT, check=True)
    if (dry_run["status_rows"], dry_run["agent_lanes"], dry_run["workflow_edges"],
        dry_run["blocked"], dry_run["failed"]) != (25, 8, 8, 0, 0):
        raise ValueError("graph dry-run result changed")
    if dry_run["statuses"] != ["dry_run"]:
        raise ValueError("graph status was not dry-run-only")
    reports = DIAGNOSTIC / "restore-rehearsal" / "reports/active"
    for filename, value in dry_run["reports_sha256"].items():
        if digest(reports / filename) != value:
            raise ValueError(f"graph report changed: {filename}")
    bare = dry_run["bare_cli_without_publication_authority"]
    if (bare["result"] != "FAIL_CLOSED"
        or digest(Path(bare["stderr_path"])) != bare["stderr_sha256"]
        or bare["reason"] not in Path(bare["stderr_path"]).read_text()):
        raise ValueError("bare graph CLI no longer fails closed")
    focused = dry_run["focused_tests"]
    suite = next(ET.parse(focused["junit_path"]).getroot().iter("testsuite"))
    if (focused["passed"] != 4 or digest(Path(focused["junit_path"])) != focused["junit_sha256"]
        or any(suite.attrib[key] != value for key, value in (
            ("tests", "4"), ("failures", "0"), ("errors", "0")))):
        raise ValueError("focused graph tests changed")

    summary = {
        "schema_version": "thewiz.gate0.langgraph_core_decision.v1",
        "decision": "RETAIN_ACTIVE_SCOPED_DRY_RUN_AND_ATOMIC_GRAPH_PUBLICATION",
        "custody": custody,
        "diagnostic_commit": HEAD_AT_PROBE,
        "diagnostic_receipt_path": str(DIAGNOSTIC / "LANGGRAPH_DRY_RUN_RECEIPT.json"),
        "diagnostic_receipt_sha256": digest(DIAGNOSTIC / "LANGGRAPH_DRY_RUN_RECEIPT.json"),
        "dry_run_stages": 25,
        "dry_run_graph_blockers": 0,
        "focused_tests_passed": 4,
        "open_contracts": [
            "nodes.py candidate adds two unselected Wizard validation stages",
            "historical strict_bool flag coercion needs separate typed-input review",
            "dry-run evidence does not authorize stage actions or trading",
        ],
        "interpretation": "Retain the current guide and active/runtime-identical graph and assistant source. Their preserved forensic variants replace governed atomic writes with direct writes and rely on an absent strict_bool helper. The exact committed restore produced seven reports for 25 dry-run stages, eight lanes, and eight edges; bare CLI report publication failed closed without a session. Four focused tests passed. This verifies the scoped diagnostic route only.",
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS three_langgraph_core_source_decisions")


if __name__ == "__main__":
    main()
