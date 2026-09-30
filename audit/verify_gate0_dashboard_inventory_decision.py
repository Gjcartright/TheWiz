"""Verify the conservative dashboard inventory source selection."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from gate0_queue_receipts import original_decision_queue


AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")
JUNIT = Path(
    "/Users/gregc/Backups/TheWiz/dashboard-inventory-diagnostics/2026-09-30/inventory-junit.xml"
)
OUTPUT = AUDIT / "GATE0_DASHBOARD_INVENTORY_DECISION_2026-09-30.json"
PATHS = (
    "config/wizard_surface_inventory_contract.json",
    "src/quant_platform/orchestration/corrective_wizard_surface_inventory.py",
    "tests/test_corrective_wizard_surface_inventory.py",
)
EXPECTED_OMITTED_GAPS = {
    "Authenticated dashboard preflight",
    "Daily and Hourly browser capture completion",
    "Authenticated inspector status",
}
EXPECTED_OMITTED_CAPTURES = {"Inspector route receipt"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def literal_assignment(path: Path, name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, set):
                raise ValueError(f"{name} is not a static set")
            return value
    raise ValueError(f"missing static set: {name}")


def fixture_counts(path: Path) -> tuple[int, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [key.value if isinstance(key, ast.Constant) else None for key in node.keys]
        if "missing_integration_count" in keys and "capture_opportunity_count" in keys:
            value = ast.literal_eval(node)
            return value["missing_integration_count"], value["capture_opportunity_count"]
    raise ValueError("missing dashboard inventory test fixture")


def main() -> None:
    rows = original_decision_queue(ROOT)
    hashes = {}
    for relative in PATHS:
        row = rows[relative]
        if row["custody_status"] != "REVIEWED_ACTIVE_RETAIN_RUNTIME_NO_PORT":
            raise ValueError(f"dashboard source decision not recorded: {relative}")
        active, runtime, saved = (root / relative for root in (ROOT, RUNTIME, SAVEPOINT))
        hashes[relative] = {
            "active_sha256": digest(active),
            "runtime_sha256": digest(runtime),
            "saved_runtime_sha256": digest(saved),
        }
        if hashes[relative]["active_sha256"] != row["working_sha256_at_freeze"]:
            raise ValueError(f"active dashboard source drift: {relative}")
        if hashes[relative]["runtime_sha256"] != row["runtime_sha256_at_freeze"]:
            raise ValueError(f"runtime dashboard source drift: {relative}")
        if hashes[relative]["saved_runtime_sha256"] != row["runtime_sha256_at_freeze"]:
            raise ValueError(f"saved dashboard candidate drift: {relative}")
        head = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(head).hexdigest() != hashes[relative]["active_sha256"]:
            raise ValueError(f"active dashboard file differs from HEAD: {relative}")
    source = PATHS[1]
    active_gaps = literal_assignment(ROOT / source, "MISSING_INTEGRATIONS")
    runtime_gaps = literal_assignment(RUNTIME / source, "MISSING_INTEGRATIONS")
    active_captures = literal_assignment(ROOT / source, "CAPTURE_OPPORTUNITIES")
    runtime_captures = literal_assignment(RUNTIME / source, "CAPTURE_OPPORTUNITIES")
    if active_gaps - runtime_gaps != EXPECTED_OMITTED_GAPS or runtime_gaps - active_gaps:
        raise ValueError("unexpected dashboard gap delta")
    if active_captures - runtime_captures != EXPECTED_OMITTED_CAPTURES or runtime_captures - active_captures:
        raise ValueError("unexpected dashboard capture delta")
    active_contract = json.loads((ROOT / PATHS[0]).read_text(encoding="utf-8"))
    runtime_contract = json.loads((RUNTIME / PATHS[0]).read_text(encoding="utf-8"))
    for contract, gaps, captures, root in (
        (active_contract, active_gaps, active_captures, ROOT),
        (runtime_contract, runtime_gaps, runtime_captures, RUNTIME),
    ):
        counts = (contract["missing_integration_count"], contract["capture_opportunity_count"])
        if counts != (len(gaps), len(captures)) or fixture_counts(root / PATHS[2]) != counts:
            raise ValueError("dashboard contract/source/test counts disagree")
        if any(
            contract.get(key) is not False
            for key in ("candidate_promotion_authority", "testnet_order_authority", "live_trading_authorized")
        ):
            raise ValueError("dashboard contract unexpectedly grants authority")
    junit = next(ET.parse(JUNIT).getroot().iter("testsuite"))
    if any(junit.attrib[key] != value for key, value in (("tests", "6"), ("failures", "0"), ("errors", "0"))):
        raise ValueError("current dashboard inventory tests did not pass")
    summary = {
        "schema_version": "thewiz.gate0.dashboard_inventory_decision.v1",
        "decision": "REVIEWED_ACTIVE_RETAIN_RUNTIME_NO_PORT",
        "paths": hashes,
        "active_recorded_missing_integrations": len(active_gaps),
        "runtime_recorded_missing_integrations": len(runtime_gaps),
        "active_capture_opportunities": len(active_captures),
        "runtime_capture_opportunities": len(runtime_captures),
        "gaps_omitted_in_runtime": sorted(EXPECTED_OMITTED_GAPS),
        "capture_opportunities_omitted_in_runtime": sorted(EXPECTED_OMITTED_CAPTURES),
        "focused_current_tests_passed": 6,
        "focused_junit_sha256": digest(JUNIT),
        "interpretation": "Retain the active static inventory for Gate 0; the runtime candidate's lower counts do not by themselves prove these dashboard inspection gaps were completed.",
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS three_dashboard_inventory_paths_six_tests")


if __name__ == "__main__":
    main()
