"""Summarize the clean-checkout recovery repair and current nightly route."""

from __future__ import annotations

import csv
import hashlib
import json
import plistlib
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
OFFDRIVE = Path("/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30")
NIGHTLY = Path("/Users/gregc/Backups/TheWiz/state/2026-09-30.json")
PLIST = Path("/Users/gregc/Library/LaunchAgents/com.thewiz.nightly-savepoint.plist")
INSTALLED = Path("/Users/gregc/Library/Application Support/TheWizBackup/nightly_savepoint.py")
OUTPUT = AUDIT / "GATE0_RECOVERY_ROUTE_DIAGNOSTIC_2026-09-30.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    initial_baseline = json.loads((OFFDRIVE / "baseline-stdout.txt").read_text())
    fixed_baseline = json.loads((OFFDRIVE / "fixed-v2-baseline-stdout.txt").read_text())
    initial_failure = (OFFDRIVE / "recovery-stdout.txt").read_text()
    if (
        initial_baseline["release_status"] != "READY_CLEAN_CHECKOUT"
        or initial_baseline["commit_candidate_files"] != 0
        or "ValueError: baseline source manifest is empty" not in initial_failure
    ):
        raise ValueError("initial clean-checkout failure was not reproduced")
    if (
        fixed_baseline["release_status"] != "READY_CLEAN_CHECKOUT"
        or fixed_baseline["commit_candidate_files"] != 0
    ):
        raise ValueError("fixed baseline was not a clean checkout")
    fixed_result = json.loads((OFFDRIVE / "fixed-v2-recovery-stdout.txt").read_text())
    if fixed_result["status"] != "PASS":
        raise ValueError("fixed recovery command did not pass")
    package = Path(fixed_result["checkpoint"])
    manifest_path = package / "recovery_manifest.json"
    drill_path = package / "restore_drill.json"
    manifest = json.loads(manifest_path.read_text())
    drill = json.loads(drill_path.read_text())
    if not (
        manifest["source_files"] == 0
        and manifest["secret_blockers"] == 0
        and manifest["restore_gate"] == "PASS"
        and manifest["research_only"] is True
        and manifest["orders_submitted"] == 0
        and manifest["testnet_order_authority"] is False
        and manifest["live_trading_authorized"] is False
        and drill["status"] == "PASS"
        and drill["bundle_verified"] is True
        and drill["compileall_passed"] is True
        and drill["restored_git_head"] == manifest["git_head"]
        and "15 passed" in drill["smoke_test_output"]
    ):
        raise ValueError("fixed restore package did not satisfy the recovery contract")
    with (package / "recovery_files.csv").open(newline="", encoding="utf-8") as stream:
        artifacts = list(csv.DictReader(stream))
    if len(artifacts) != 8 or any(
        digest(package / row["path"]) != row["sha256"] for row in artifacts
    ):
        raise ValueError("recovery package artifact hashes differ from its manifest")
    nightly = json.loads(NIGHTLY.read_text())
    if any(nightly[key] != "PASS" for key in ("expansion", "mac_internal", "github")):
        raise ValueError("nightly savepoint destination is not passing")
    for key in ("expansion_path", "mac_internal_path"):
        snapshot = Path(nightly[key])
        saved = json.loads((snapshot / "SAVEPOINT_MANIFEST.json").read_text())
        if saved["manifest_sha256"] != nightly["manifest_sha256"]:
            raise ValueError(f"nightly manifest mismatch: {key}")
    if digest(INSTALLED) != digest(ROOT / "scripts/ops/nightly_savepoint.py"):
        raise ValueError("installed nightly script differs from source")
    launch_agent = plistlib.loads(PLIST.read_bytes())
    if launch_agent["StartCalendarInterval"] != {"Hour": 0, "Minute": 0}:
        raise ValueError("nightly LaunchAgent schedule changed")
    summary = {
        "schema_version": "thewiz.gate0.recovery_route_diagnostic.v1",
        "initial_clean_checkout_baseline": "READY_CLEAN_CHECKOUT_ZERO_CHANGED_SOURCE",
        "initial_recovery_failure": "BASELINE_SOURCE_MANIFEST_EMPTY",
        "fixed_disposable_package_path": str(package),
        "fixed_disposable_head": manifest["git_head"],
        "fixed_package_manifest_sha256": digest(manifest_path),
        "fixed_restore_drill_sha256": digest(drill_path),
        "fixed_package_artifacts_rehashed": len(artifacts),
        "fixed_clean_checkout_changed_source_rows": 0,
        "fixed_smoke_tests_passed": 15,
        "fixed_restore_gate": "PASS",
        "fixed_baseline_artifact_and_git_state_binding": "PASS",
        "nightly_date": nightly["date"],
        "nightly_expansion": nightly["expansion"],
        "nightly_mac_internal": nightly["mac_internal"],
        "nightly_private_github": nightly["github"],
        "nightly_manifest_sha256": nightly["manifest_sha256"],
        "nightly_source_and_installed_script_sha256": digest(INSTALLED),
        "nightly_calendar": "00:00 America/New_York",
        "interpretation": "The current scheduled savepoint is passing. The optional redacted packager failed on a clean checkout before the repair and passed a complete disposable restore drill afterward. Historical workspace-specific source variants remain under review.",
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS recovery_route_diagnostic_recorded")


if __name__ == "__main__":
    main()
