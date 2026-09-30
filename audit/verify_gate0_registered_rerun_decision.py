"""Verify the guarded registered-rerun source selection."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
VARIANTS = Path("/Users/gregc/Backups/TheWiz/source-delta-evidence/2026-09-30")
JUNIT = Path("/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30/registered-rerun-active-junit.xml")
RELATIVE = "src/quant_platform/orchestration/corrective_registered_rerun_executor.py"
RUNTIME_API = "src/quant_platform/orchestration/corrective_runtime.py"
OUTPUT = AUDIT / "GATE0_REGISTERED_RERUN_DECISION_2026-09-30.json"
STATUS = "REVIEWED_ACTIVE_RETAIN_CANDIDATE_DEFERRED_NO_PORT"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kwonly_names(path: Path, function: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function
    ]
    if len(matches) != 1:
        raise ValueError(f"missing runtime function: {function}")
    return {arg.arg for arg in matches[0].args.kwonlyargs}


def main() -> None:
    with (AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = [row for row in csv.DictReader(stream) if row["relative_path"] == RELATIVE]
    if len(rows) != 1 or rows[0]["custody_status"] != STATUS:
        raise ValueError("registered-rerun source decision missing or duplicated")
    row = rows[0]
    historical_hash = row["historical_variant_sha256"]
    if int(row["historical_variant_count"]) != 1:
        raise ValueError("unexpected registered-rerun variant count")
    manifest = json.loads((VARIANTS / "MANIFEST.json").read_text())
    entries = [
        entry
        for entry in manifest["entries"]
        if entry["relative_path"] == RELATIVE and entry["sha256"] == historical_hash
    ]
    if len(entries) != 1:
        raise ValueError("historical registered-rerun variant absent from custody")
    historical = VARIANTS / entries[0]["stored_as"]
    if digest(historical) != historical_hash:
        raise ValueError("historical registered-rerun bytes changed")
    active = ROOT / RELATIVE
    runtime = RUNTIME / RELATIVE
    if digest(active) != row["working_sha256_at_freeze"]:
        raise ValueError("active registered-rerun source changed")
    if digest(runtime) != row["runtime_sha256_at_freeze"]:
        raise ValueError("runtime registered-rerun candidate changed")
    active_text = active.read_text(encoding="utf-8")
    runtime_text = runtime.read_text(encoding="utf-8")
    historical_text = historical.read_text(encoding="utf-8")
    if "promote_staged_directory(temporary, workspace)" not in active_text:
        raise ValueError("active guarded directory publication absent")
    if "safe_exception_code" not in active_text or "atomic_write_bytes" not in active_text:
        raise ValueError("active guarded file or redaction path absent")
    if "promote_staged_directory(temporary, workspace, immutable=False)" not in runtime_text:
        raise ValueError("runtime mutable-workspace candidate changed")
    if any(token not in historical_text for token in (
        "shutil.copy2(source, target)",
        "temporary.replace(workspace)",
        "from quant_platform.runtime_types import strict_bool",
    )):
        raise ValueError("historical bypass candidate changed")
    active_kwonly = kwonly_names(ROOT / RUNTIME_API, "promote_staged_directory")
    runtime_kwonly = kwonly_names(RUNTIME / RUNTIME_API, "promote_staged_directory")
    if "immutable" in active_kwonly or "immutable" not in runtime_kwonly:
        raise ValueError("directory promotion API relationship changed")
    active_runtime_types = ast.parse(
        (ROOT / "src/quant_platform/runtime_types.py").read_text(encoding="utf-8")
    )
    if any(
        isinstance(node, ast.FunctionDef) and node.name == "strict_bool"
        for node in active_runtime_types.body
    ):
        raise ValueError("historical strict_bool dependency became active")
    junit = next(ET.parse(JUNIT).getroot().iter("testsuite"))
    if any(junit.attrib[key] != value for key, value in (("tests", "49"), ("failures", "0"), ("errors", "0"))):
        raise ValueError("focused registered-rerun tests did not pass")
    summary = {
        "schema_version": "thewiz.gate0.registered_rerun_decision.v1",
        "decision": STATUS,
        "source_path": RELATIVE,
        "active_sha256": digest(active),
        "runtime_sha256": digest(runtime),
        "preserved_forensic_sha256": historical_hash,
        "active_directory_promotion_kwonly": sorted(active_kwonly),
        "runtime_directory_promotion_kwonly": sorted(runtime_kwonly),
        "active_focused_tests_passed": 49,
        "focused_junit_sha256": digest(JUNIT),
        "interpretation": "Retain active guarded publication and exception redaction. The forensic candidate bypasses those guards and has an absent strict_bool dependency; the LocalRuntime mutable-directory call needs its matching runtime API and further safety review.",
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS registered_rerun_source_decision")


if __name__ == "__main__":
    main()
