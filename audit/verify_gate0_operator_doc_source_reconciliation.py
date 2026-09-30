#!/usr/bin/env python3
"""Verify operator documentation decisions against preserved source copies."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "4362c54b55afabe38f522289c2f9e84788e9ab67"
SELECTED_COMMIT = "0b60eadf4f6a8d92cf9653c952f184aa3ea97146"
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
OUTPUT = ROOT / "audit/GATE0_OPERATOR_DOC_SOURCE_RECONCILIATION_2026-09-30.json"
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")
VARIANT_ROOT = Path("/Users/gregc/Backups/TheWiz/source-delta-evidence/2026-09-30/variants")
SELECTED = {
    "audit/verify_gate0_math_marker_currentness.py": "54c2cf9e643998798982f52c7b61a8f1ac7a79ed4e6230446a4dcdda3759f70c",
    "docs/architecture.md": "204fa1db829fd8ed8184fc15d289eef6bf4efe49724bbfe64b84515fd4e8a0a8",
    "docs/current_corrective_operations.md": "5f73606074d48cfef0ffb4b6315d69a262989b7702b1516825eb4b62bdcaadf6",
    "docs/registered_evidence_continuity.md": "5952f0d379a3878f7cdb87555fdeab8d05f8e99db8c0634c78cca2c20d16da44",
    "docs/strategy_registry.csv": "7d371db27b21b65a6a1b2a3764aeae4fff48c344c7cec07666d5902685188f56",
    "docs/wizard_hyperliquid_mode_fidelity.md": "1c6a9c07f0963f62fe78ecd9077c9f3c7be0732ef8512f89e92687c78d03aebe",
    "src/quant_platform/strategies.py": "ceaac4be116c63a1a7bb2e43e5ffabfdb72722e9d327f8ffc27339f48fb661df",
    "src/quant_platform/orchestration/corrective_data_evidence.py": "77de431827997e15a97c074d06aacd49badc674857f34c9ff836d361e46f1610",
    "src/quant_platform/orchestration/corrective_wizard_parity.py": "30a39d62ffd1f5c9857e8eeaedcf478f749e6318a19d219bdfda773e25757b7f",
}
RUNTIME_ONLY = {
    "docs/seven_day_release_handoff.md": "38d78817527969e6e93821eb8b3bedbda36885fae0359642d9b482f4b544051c",
    "docs/single_owner_research_mode.md": "ac52acb017a4f8758f3c8b2a48bb3bc631064356ff5baa89ce17a08b7156179a",
    "docs/wizard_hyperliquid_frozen_adapter.md": "0b4e0dffce7863d81a01e7bacf22bba71d1ca83a4de624a5a4e45a82092aa80c",
    "docs/wizard_hyperliquid_l2_calibration_runner.md": "a34e7055f0d7466b7e150e1d9959083e4335400684879ef8fa8e711ce759cf92",
}
VARIANTS = {
    "docs/current_corrective_operations.md": "fcb93e7a0666eabbbcf5e79f6815458c180cbb1a58ffa926e75b4c00d8230ef9",
    "docs/strategy_registry.csv": "7bf6bc248b39a782bef3bb77f7f53f9d0d63837adb3b7946569dd77cbf4f9297",
}
RETAIN = frozenset({
    "docs/architecture.md",
    "docs/current_corrective_operations.md",
    "docs/registered_evidence_continuity.md",
    "docs/strategy_registry.csv",
    "docs/wizard_hyperliquid_mode_fidelity.md",
})
DEFER = frozenset(RUNTIME_ONLY)
BASE_OPERATIONS_SHA256 = "5441eec3c383334096e92a9f3d4aca9944e690db712a3ebbee5e7e144f680785"
QUEUE_SHA256 = "0324b59290b1672e3c08af4589429e9a62edaaebc04bcaee0ddd52edd0e632ed"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_bytes(relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{SELECTED_COMMIT}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout


def selected_exists(relative: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{SELECTED_COMMIT}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    ).returncode == 0


def strategy_counts() -> tuple[int, tuple[int, ...]]:
    tree = ast.parse(selected_bytes("src/quant_platform/strategies.py").decode("utf-8"))
    catalog = None
    official = None
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id == "ALL_STRATEGIES" and isinstance(node.value, ast.Tuple):
            catalog = len(node.value.elts)
        elif node.target.id == "OFFICIAL_CRYPTO_WIZARDS_STRATEGY_IDS":
            official = tuple(ast.literal_eval(node.value))
    if catalog is None or official is None:
        raise ValueError("strategy source declarations missing")
    return catalog, official


def main() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    subprocess.run(["git", "merge-base", "--is-ancestor", SELECTED_COMMIT, "HEAD"], cwd=ROOT, check=True)
    for relative, expected in SELECTED.items():
        if hashlib.sha256(selected_bytes(relative)).hexdigest() != expected:
            raise ValueError(f"selected documentation or source drift: {relative}")
    for relative, expected in RUNTIME_ONLY.items():
        path = SAVEPOINT / relative
        if selected_exists(relative) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"deferred runtime-only document changed: {relative}")
    for relative, expected in VARIANTS.items():
        path = VARIANT_ROOT / expected
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"historical document variant drift: {relative}")
    base_operations = subprocess.run(
        ["git", "show", f"{BASE}:docs/current_corrective_operations.md"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    if hashlib.sha256(base_operations).hexdigest() != BASE_OPERATIONS_SHA256:
        raise ValueError("pre-repair operator command evidence changed")

    architecture = selected_bytes("docs/architecture.md").decode("utf-8")
    operations = selected_bytes("docs/current_corrective_operations.md").decode("utf-8")
    continuity = selected_bytes("docs/registered_evidence_continuity.md").decode("utf-8")
    mode_fidelity = selected_bytes("docs/wizard_hyperliquid_mode_fidelity.md").decode("utf-8")
    if not all(word in architecture for word in ("UV_PROJECT_ENVIRONMENT", "Node.js", "LangGraph Agent Workflow")):
        raise ValueError("current architecture runtime distinctions missing")
    if (operations.count("uv run --locked python") != 26
            or "PYTHONPATH=src .venv/bin/python3" in operations
            or "UV_PROJECT_ENVIRONMENT" not in operations
            or "/Volumes/Expansion/Crypto Wizard" not in operations
            or "/Volumes/CodexWorkspace" in operations):
        raise ValueError("operator command or recovery guidance changed")
    if ("current_wizard_cost_selected_collection" not in continuity
            or "current_wizard_cost_selected_collection" not in
            selected_bytes("src/quant_platform/orchestration/corrective_data_evidence.py").decode("utf-8")):
        raise ValueError("third read-only cost lane is unaccounted for")
    if ("Golden dashboard captures: `0`" not in mode_fidelity
            or "Golden dashboard captures: `14`" not in
            (SAVEPOINT / "docs/wizard_hyperliquid_mode_fidelity.md").read_text(encoding="utf-8")):
        raise ValueError("current versus historical dashboard capture posture changed")

    registry = list(csv.DictReader(selected_bytes("docs/strategy_registry.csv").decode("utf-8").splitlines()))
    catalog, official = strategy_counts()
    if catalog != len(registry) or [int(row["id"]) for row in registry] != list(range(1, 38)) or len(official) != 18:
        raise ValueError("full strategy catalog and official active lane differ")
    with (VARIANT_ROOT / VARIANTS["docs/strategy_registry.csv"]).open(newline="", encoding="utf-8") as stream:
        old_subset = list(csv.DictReader(stream))
    if tuple(int(row["id"]) for row in old_subset) != official:
        raise ValueError("historical official-lane subset differs from source")
    for module in (
        "src/quant_platform/orchestration/wizard_hyperliquid_frozen_adapter.py",
        "src/quant_platform/orchestration/wizard_hyperliquid_l2_calibration_runner.py",
    ):
        if selected_exists(module) or not (SAVEPOINT / module).is_file():
            raise ValueError(f"deferred document's runtime module custody changed: {module}")

    queue_bytes = selected_bytes(QUEUE.relative_to(ROOT).as_posix())
    if hashlib.sha256(queue_bytes).hexdigest() != QUEUE_SHA256:
        raise ValueError("union source queue drift")
    rows = list(csv.DictReader(queue_bytes.decode("utf-8").splitlines()))
    if len(rows) != 811 or len({row["relative_path"] for row in rows}) != 811:
        raise ValueError("union source queue size or uniqueness changed")
    queue = {row["relative_path"]: row for row in rows}
    for relative in RETAIN | DEFER:
        status = "REVIEWED_OPERATOR_DOC_RETAIN" if relative in RETAIN else "REVIEWED_DEPENDENT_DOCUMENT_DEFERRED_NO_PORT"
        row = queue[relative]
        if row["custody_status"] != status or OUTPUT.name not in row["decision_evidence"]:
            raise ValueError(f"operator documentation source decision missing: {relative}")
    pending = sum(row["custody_status"] in {
        "PRESERVED_REVIEW_REQUIRED_NO_PORT",
        "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED",
    } for row in rows)
    reviewed = sum(row["custody_status"].startswith("REVIEWED_") for row in rows)
    if (pending, reviewed, len(rows) - pending - reviewed) != (643, 150, 18):
        raise ValueError("Gate 0 source queue accounting changed")

    report = {
        "schema_version": "thewiz.gate0.operator_doc_source_reconciliation.v1",
        "decision": "NINE_OPERATOR_DOC_PATHS_REVIEWED_EXACT_COMMIT_VERIFICATION_REQUIRED",
        "base_commit": BASE,
        "selected_file_sha256": SELECTED,
        "runtime_only_savepoint_sha256": RUNTIME_ONLY,
        "preserved_historical_variant_sha256": VARIANTS,
        "base_operator_commands_sha256": BASE_OPERATIONS_SHA256,
        "review_decisions": [
            "retain_current_expansion_and_langgraph_architecture_instead_of_old_runtime_doc",
            "replace_26_incomplete_on_drive_virtualenv_command_examples_with_locked_uv",
            "retain_three_distinct_read_only_l2_collection_lanes",
            "retain_37_strategy_catalog_and_preserve_18_strategy_active_subset",
            "preserve_historical_14_vendor_capture_claim_without_current_parity_authority",
            "defer_four_runtime_only_operational_documents_until_their_source_and_custody_are_reviewed",
        ],
        "union_source_queue": {
            "file_sha256": QUEUE_SHA256,
            "distinct_paths": len(rows),
            "reviewed_paths": reviewed,
            "nonsource_paths": len(rows) - pending - reviewed,
            "remaining_semantic_review_paths": pending,
        },
        "research_acceptance": "BLOCKED",
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS operator_documentation_source_reconciliation")


if __name__ == "__main__":
    main()
