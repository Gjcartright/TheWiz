"""Verify the two recovery-script source dispositions and preserved variants."""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
VARIANTS = Path("/Users/gregc/Backups/TheWiz/source-delta-evidence/2026-09-30")
JUNIT = Path("/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30/active-focused-junit.xml")
OUTPUT = AUDIT / "GATE0_RECOVERY_SCRIPT_DECISIONS_2026-09-30.json"
PATHS = {
    "scripts/build_corrective_checkpoint.py": "REVIEWED_ACTIVE_RETAIN_RUNTIME_AND_FORENSIC_NO_PORT",
    "scripts/build_current_recovery_checkpoint.py": "REVIEWED_REPAIRED_ACTIVE_RETAIN_FORENSIC_NO_PORT",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with (AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        queue = {row["relative_path"]: row for row in csv.DictReader(stream)}
    manifest = json.loads((VARIANTS / "MANIFEST.json").read_text(encoding="utf-8"))
    custody_receipt = json.loads((VARIANTS / "RECEIPT.json").read_text(encoding="utf-8"))
    canonical_manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        custody_receipt["status"] != "PASS"
        or custody_receipt["manifest_sha256"] != canonical_manifest_sha256
    ):
        raise ValueError("historical variant custody receipt mismatch")
    decisions = {}
    for relative, status in PATHS.items():
        row = queue[relative]
        if row["custody_status"] != status or int(row["historical_variant_count"]) != 1:
            raise ValueError(f"recovery script decision missing: {relative}")
        historical_hash = row["historical_variant_sha256"]
        entries = [
            item
            for item in manifest["entries"]
            if item["relative_path"] == relative and item["sha256"] == historical_hash
        ]
        if len(entries) != 1 or digest(VARIANTS / entries[0]["stored_as"]) != historical_hash:
            raise ValueError(f"historical recovery variant changed: {relative}")
        runtime_hash = digest(RUNTIME / relative)
        if runtime_hash != row["runtime_sha256_at_freeze"]:
            raise ValueError(f"runtime recovery candidate changed: {relative}")
        decisions[relative] = {
            "decision": status,
            "frozen_active_sha256": row["working_sha256_at_freeze"],
            "current_active_sha256": digest(ROOT / relative),
            "runtime_sha256": runtime_hash,
            "preserved_historical_sha256": historical_hash,
            "historical_copy_roots": row["historical_copy_roots"],
        }
    baseline = ROOT / "scripts/build_corrective_checkpoint.py"
    packager = ROOT / "scripts/build_current_recovery_checkpoint.py"
    if decisions[str(baseline.relative_to(ROOT))]["current_active_sha256"] != decisions[
        str(baseline.relative_to(ROOT))
    ]["frozen_active_sha256"]:
        raise ValueError("active baseline selector changed unexpectedly")
    if decisions[str(packager.relative_to(ROOT))]["current_active_sha256"] == decisions[
        str(packager.relative_to(ROOT))
    ]["frozen_active_sha256"]:
        raise ValueError("clean-checkout recovery repair was not applied")
    baseline_source = baseline.read_text(encoding="utf-8")
    packager_source = packager.read_text(encoding="utf-8")
    if '"apps/the-ave/"' not in baseline_source or '"apps/the-ave/"' not in packager_source:
        raise ValueError("auxiliary app was admitted into the quant release")
    if any(token not in packager_source for token in (
        "empty source manifest requires a verified clean checkout",
        "python = Path(sys.executable)",
        "restored Git head differs from checkpoint head",
    )):
        raise ValueError("recovery repair contract is incomplete")
    if 'ROOT / ".venv312"' in packager_source:
        raise ValueError("missing legacy interpreter is still required")
    junit = next(ET.parse(JUNIT).getroot().iter("testsuite"))
    if any(junit.attrib[key] != value for key, value in (("tests", "58"), ("failures", "0"), ("errors", "0"))):
        raise ValueError("focused recovery and active-pipeline tests did not pass")
    route = json.loads((AUDIT / "GATE0_RECOVERY_ROUTE_DIAGNOSTIC_2026-09-30.json").read_text())
    if route["fixed_restore_gate"] != "PASS" or route["fixed_smoke_tests_passed"] != 15:
        raise ValueError("disposable recovery package was not verified")
    summary = {
        "schema_version": "thewiz.gate0.recovery_script_decisions.v1",
        "decisions": decisions,
        "variant_manifest_sha256": digest(VARIANTS / "MANIFEST.json"),
        "variant_custody_receipt_canonical_sha256": canonical_manifest_sha256,
        "route_diagnostic_sha256": digest(AUDIT / "GATE0_RECOVERY_ROUTE_DIAGNOSTIC_2026-09-30.json"),
        "focused_tests_passed": 58,
        "focused_junit_sha256": digest(JUNIT),
        "disposable_restore_smoke_tests_passed": 15,
        "interpretation": "Retain the current auxiliary-app exclusion, leave encrypted-workspace keyword/evidence additions in preserved custody, and use the repaired active optional packager with its verified clean-checkout restore. The scheduled midnight backup remains the primary current route.",
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS two_recovery_script_decisions")


if __name__ == "__main__":
    main()
