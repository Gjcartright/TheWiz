"""Verify the dated no-port decisions against exact backup bytes and Git ignore."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    decisions = rows(AUDIT / "GATE0_SOURCE_DECISIONS_2026-09-30.csv")
    source = {
        row["relative_path"]: row
        for row in rows(AUDIT / "evidence_freeze_2026-09-29/source_reconciliation.csv")
    }
    manifest = json.loads((SAVEPOINT / "SAVEPOINT_MANIFEST.json").read_text())
    entries = {row["path"]: row for row in manifest["entries"]}
    seen = set()
    for row in decisions:
        relative = row["relative_path"]
        if relative in seen or relative not in source:
            raise ValueError(f"duplicate_or_missing_source:{relative}")
        seen.add(relative)
        if row["decision"] == "RETAIN_HISTORICAL_LOG_NO_PORT":
            valid_path = relative.startswith("scripts/schedule_logs/") and relative.endswith(".log")
        elif row["decision"] == "GENERATED_PACKAGE_METADATA_NO_PORT":
            valid_path = relative.startswith("src/quantized_stat_arb_platform.egg-info/")
        else:
            valid_path = False
        if not valid_path:
            raise ValueError(f"invalid_nonsource_decision:{relative}")
        backup_path = SAVEPOINT / "local_runtime" / relative
        expected = source[relative]["runtime_sha256"]
        if entries["local_runtime/" + relative]["sha256"] != expected:
            raise ValueError(f"manifest_hash_mismatch:{relative}")
        if hashlib.sha256(backup_path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"backup_byte_mismatch:{relative}")
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", relative], cwd=ROOT, check=False
        ).returncode == 0
        if not ignored:
            raise ValueError(f"source_path_not_ignored:{relative}")
    print(f"PASS exact_backup_and_ignored_paths={len(decisions)}")


if __name__ == "__main__":
    main()
