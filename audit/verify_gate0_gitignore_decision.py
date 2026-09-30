"""Verify the reviewed .gitignore selection against preserved variant bytes."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
VARIANTS = Path("/Users/gregc/Backups/TheWiz/source-delta-evidence/2026-09-30")
OUTPUT = AUDIT / "GATE0_GITIGNORE_DECISION_2026-09-30.json"
REQUIRED_IGNORED = (
    "._garbage.py",
    "pytest-of-1/foo",
    ".runtime_logs/example",
    ".runtime_locks/example",
    ".runtime_control/example",
)
REQUIRED_INCLUDED = (
    "reports/supreme_team/2026-08-20_v2_corrective_roadmap.md",
    "reports/supreme_team/2026-08-20_v2_corrective_work_items.csv",
    "reports/supreme_team/2026-08-20_v2_definition_of_done.csv",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pattern(path: str, expected_prefix: str) -> str:
    result = subprocess.run(
        ["git", "check-ignore", "-v", path],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    pattern = result.stdout.split("\t", 1)[0].split(":", 2)[-1]
    if not pattern.startswith(expected_prefix):
        raise ValueError(f"unexpected ignore policy for {path}: {pattern}")
    return pattern


def main() -> None:
    with (AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if row["relative_path"] == ".gitignore"]
    if len(matches) != 1 or matches[0]["custody_status"] != "REVIEWED_ACTIVE_RETAIN_HISTORICAL_NO_PORT":
        raise ValueError(".gitignore decision is absent or duplicated")
    row = matches[0]
    manifest = json.loads((VARIANTS / "MANIFEST.json").read_text(encoding="utf-8"))
    entries = [entry for entry in manifest["entries"] if entry["relative_path"] == ".gitignore"]
    expected = set(row["historical_variant_sha256"].split(";"))
    actual = {entry["sha256"] for entry in entries}
    if len(entries) != 2 or actual != expected:
        raise ValueError("historical .gitignore variants do not match the queue")
    for entry in entries:
        if digest(VARIANTS / entry["stored_as"]) != entry["sha256"]:
            raise ValueError("preserved .gitignore variant changed")
    active = ROOT / ".gitignore"
    head = subprocess.run(
        ["git", "show", "HEAD:.gitignore"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    if hashlib.sha256(head).hexdigest() != digest(active):
        raise ValueError("active .gitignore differs from HEAD")
    ignored = {path: check_pattern(path, "") for path in REQUIRED_IGNORED}
    if any(pattern.startswith("!") for pattern in ignored.values()):
        raise ValueError("required runtime artifact is not ignored")
    included = {path: check_pattern(path, "!") for path in REQUIRED_INCLUDED}
    summary = {
        "schema_version": "thewiz.gate0.gitignore_decision.v1",
        "decision": row["custody_status"],
        "active_sha256": digest(active),
        "historical_variants": entries,
        "variant_manifest_sha256": digest(VARIANTS / "MANIFEST.json"),
        "ignored_probes": ignored,
        "included_probes": included,
        "review_note": "Both preserved variants omit runtime-control ignores and the corrective-plan exception. The forensic_b variant alone adds build/ and dist/ ignores; those optional generated-output patterns do not justify replacing the active policy.",
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS active_policy_and_two_preserved_variants")


if __name__ == "__main__":
    main()
