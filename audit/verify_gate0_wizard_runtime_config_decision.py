"""Verify the source decision for two LocalRuntime-only Wizard configs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")
OUTPUT = AUDIT / "GATE0_WIZARD_RUNTIME_CONFIG_DECISION_2026-09-30.json"
STATUS = "REVIEWED_DEPENDENT_RUNTIME_CONFIG_DEFERRED_NO_PORT"
PATHS = (
    "config/crypto_wizards_dashboard_research_specialist.json",
    "config/crypto_wizards_keychain_reference.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with (AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = {row["relative_path"]: row for row in csv.DictReader(stream)}
    manifest = json.loads(
        (SAVEPOINT.parent / "SAVEPOINT_MANIFEST.json").read_text(encoding="utf-8")
    )
    manifest_files = {
        entry["path"]: entry["sha256"]
        for entry in manifest["entries"]
        if entry["type"] == "file"
    }
    evidence = {}
    for relative in PATHS:
        row = rows[relative]
        if row["custody_status"] != STATUS or row["working_vs_runtime"] != "RUNTIME_ONLY":
            raise ValueError(f"config source decision missing: {relative}")
        if (ROOT / relative).exists():
            raise ValueError(f"unselected config unexpectedly became active: {relative}")
        candidate, saved = RUNTIME / relative, SAVEPOINT / relative
        if digest(candidate) != row["runtime_sha256_at_freeze"]:
            raise ValueError(f"LocalRuntime candidate changed: {relative}")
        if digest(saved) != row["runtime_sha256_at_freeze"]:
            raise ValueError(f"Mac savepoint copy changed: {relative}")
        if manifest_files.get(f"local_runtime/{relative}") != digest(saved):
            raise ValueError(f"Mac savepoint manifest differs: {relative}")
        evidence[relative] = {
            "runtime_sha256": digest(candidate),
            "savepoint_sha256": digest(saved),
        }

    specialist = json.loads((RUNTIME / PATHS[0]).read_text(encoding="utf-8"))
    if specialist.get("schema_version") != "thewiz.crypto_wizards_dashboard_research_specialist.v1":
        raise ValueError("dashboard specialist schema changed")
    if specialist.get("status") != "configured_waiting_for_authenticated_browser_capture":
        raise ValueError("dashboard specialist activation claim changed")
    boundaries = specialist["hard_boundaries"]
    if not boundaries or any(value is not False for value in boundaries.values()):
        raise ValueError("dashboard specialist gained authority")
    keychain = json.loads((RUNTIME / PATHS[1]).read_text(encoding="utf-8"))
    if set(keychain) != {"schema_version", "service"}:
        raise ValueError("Keychain selector includes unreviewed fields")
    if keychain["schema_version"] != "thewiz.crypto_wizards_keychain_reference.v1":
        raise ValueError("Keychain selector schema changed")
    if not isinstance(keychain["service"], str) or not keychain["service"]:
        raise ValueError("Keychain service selector invalid")

    runtime_mini_agents = RUNTIME / "src/quant_platform/orchestration/mini_agents.py"
    runtime_daily_runner = RUNTIME / "src/quant_platform/orchestration/current_wizard_hyperliquid_daily_runner.py"
    if specialist["agent"] not in runtime_mini_agents.read_text(encoding="utf-8"):
        raise ValueError("dashboard specialist runtime dependency missing")
    if "WIZARD_KEYCHAIN_REFERENCE = Path(\"config/crypto_wizards_keychain_reference.json\")" not in runtime_daily_runner.read_text(encoding="utf-8"):
        raise ValueError("Keychain selector runtime dependency missing")
    active_daily_runner = ROOT / runtime_daily_runner.relative_to(RUNTIME)
    if not active_daily_runner.is_file():
        raise ValueError("active daily runner missing")
    if "WIZARD_KEYCHAIN_REFERENCE" in active_daily_runner.read_text(encoding="utf-8"):
        raise ValueError("Keychain selector unexpectedly became active")
    active_mini_agents = ROOT / runtime_mini_agents.relative_to(RUNTIME)
    if specialist["agent"] in active_mini_agents.read_text(encoding="utf-8"):
        raise ValueError("dashboard specialist unexpectedly became active")

    summary = {
        "schema_version": "thewiz.gate0.wizard_runtime_config_decision.v1",
        "decision": STATUS,
        "paths": evidence,
        "dashboard_specialist_status": specialist["status"],
        "dashboard_specialist_hard_boundaries": boundaries,
        "keychain_config_contains_secret": False,
        "runtime_consumers": [
            str(runtime_mini_agents.relative_to(RUNTIME)),
            str(runtime_daily_runner.relative_to(RUNTIME)),
        ],
        "interpretation": "Both files are preserved as runtime-only source. The dashboard specialist is a research configuration awaiting authenticated capture, and its agent registry entry is absent from active source. The Keychain file is a non-secret selector: the daily runner exists in active source, but its Keychain selector and credential flow exist only in LocalRuntime. Neither config is selected alone without its matching behavior review.",
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS two_wizard_runtime_config_decisions")


if __name__ == "__main__":
    main()
