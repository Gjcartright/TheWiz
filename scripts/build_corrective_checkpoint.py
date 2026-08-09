#!/usr/bin/env python3
"""Build a redacted source-control baseline for the corrective plan."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "reports" / "active"

ROOT_INCLUDES = {
    ".gitignore",
    ".env.example",
    "README.md",
    "project_objective.md",
    "pyproject.toml",
}
INCLUDE_PREFIXES = ("src/quant_platform/", "tests/", "config/", "docs/")
SCRIPT_KEYWORDS = (
    "wizard",
    "hyperliquid",
    "three_hour",
    "dydx",
    "crypto",
    "quant",
    "research",
    "dashboard",
    "apify",
    "funding",
    "corrective",
)
EXCLUDE_PREFIXES = (
    ".venv312/",
    ".venv/",
    ".venv311/",
    "data/",
    "reports/",
    "runs/",
    "apps/the-ave/",
    "scripts/schedule_logs/",
    "work/",
    "archive/",
    "models/",
)
EXCLUDE_EXACT = {"tmp_subs.en-orig.vtt"}

SECRET_PATTERNS = (
    ("pem_private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("github_token", re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("apify_token", re.compile(r"\bapify_api_[A-Za-z0-9_-]{20,}\b", re.I)),
    (
        "possible_hex_private_key",
        re.compile(
            r"(?i)(?:private[_ -]?key|secret[_ -]?key)\s*[:=]\s*[\"']?0x[a-f0-9]{64}\b"
        ),
    ),
)
PLACEHOLDER_MARKERS = ("example", "placeholder", "changeme", "your_", "<", "${", "dummy", "test")


def _git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=text)


def _worktree_entries() -> list[tuple[str, str]]:
    cached_status = Path("/tmp/thewiz_corrective_status.bin")
    if cached_status.exists():
        raw = cached_status.read_bytes()
    else:
        raw = _git("status", "--porcelain=v1", "-z", "-uall", text=False)
        assert isinstance(raw, bytes)
    parts = raw.decode("utf-8", "surrogateescape").split("\0")
    entries: list[tuple[str, str]] = []
    index = 0
    while index < len(parts):
        item = parts[index]
        if not item:
            index += 1
            continue
        status = item[:2]
        path = item[3:]
        if status[0] in {"R", "C"}:
            index += 1
            if index < len(parts) and parts[index]:
                path = parts[index]
        entries.append((status, path))
        index += 1
    return entries


def _classify(path: str) -> tuple[str, str]:
    if path in ROOT_INCLUDES or path.startswith(INCLUDE_PREFIXES):
        return "commit", "active_source_config_test_or_documentation"
    if path.startswith("scripts/"):
        name = Path(path).name.lower()
        if any(keyword in name for keyword in SCRIPT_KEYWORDS):
            return "commit", "active_quant_research_operator_script"
        return "review", "script_scope_not_yet_confirmed"
    if path in EXCLUDE_EXACT or path.startswith(EXCLUDE_PREFIXES):
        return "exclude", "generated_secret_environment_or_unrelated_scope"
    return "review", "outside_confirmed_quant_platform_scope"


def _file_hash(path: Path) -> tuple[str, int]:
    if not path.is_file() or path.is_symlink():
        return "", 0
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ACTIVE.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for status, relative_path in _worktree_entries():
        disposition, reason = _classify(relative_path)
        path = ROOT / relative_path
        if path.exists() and disposition == "commit":
            digest, size = _file_hash(path)
        elif path.exists() and path.is_file():
            digest, size = "", path.stat().st_size
        else:
            digest, size = "", 0
        rows.append(
            {
                "path": relative_path,
                "git_status": status,
                "disposition": disposition,
                "reason": reason,
                "exists": path.exists(),
                "size_bytes": size,
                "sha256": digest,
            }
        )

    manifest_path = ACTIVE / "corrective_source_commit_manifest.csv"
    manifest_fields = [
        "path",
        "git_status",
        "disposition",
        "reason",
        "exists",
        "size_bytes",
        "sha256",
    ]
    _write_csv(manifest_path, rows, manifest_fields)

    include_rows = [row for row in rows if row["disposition"] == "commit" and row["exists"]]
    files_path = ACTIVE / "corrective_baseline_files.csv"
    file_fields = ["path", "git_status", "size_bytes", "sha256"]
    _write_csv(
        files_path,
        [{field: row[field] for field in file_fields} for row in include_rows],
        file_fields,
    )

    findings: list[dict[str, object]] = []
    for row in include_rows:
        path = ROOT / str(row["path"])
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(content.splitlines(), 1):
            for detector, pattern in SECRET_PATTERNS:
                if not pattern.search(line):
                    continue
                lower = line.lower()
                placeholder = any(marker in lower for marker in PLACEHOLDER_MARKERS)
                findings.append(
                    {
                        "path": row["path"],
                        "line": line_number,
                        "detector": detector,
                        "severity": "review" if placeholder else "block",
                        "value_redacted": True,
                    }
                )

    scan_path = ACTIVE / "corrective_secret_scan.csv"
    scan_fields = ["path", "line", "detector", "severity", "value_redacted"]
    _write_csv(scan_path, findings, scan_fields)

    counts: dict[str, int] = {}
    for row in rows:
        disposition = str(row["disposition"])
        counts[disposition] = counts.get(disposition, 0) + 1

    manifest = {
        "run_id": "corrective_baseline_20260809T034347Z",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "git_head": str(_git("rev-parse", "HEAD")).strip(),
        "git_branch": str(_git("branch", "--show-current")).strip(),
        "git_remote": str(_git("remote", "get-url", "origin")).strip(),
        "worktree_entries": len(rows),
        "disposition_counts": counts,
        "commit_candidate_files": len(include_rows),
        "commit_candidate_bytes": sum(int(row["size_bytes"]) for row in include_rows),
        "secret_findings_total": len(findings),
        "secret_blockers": sum(1 for finding in findings if finding["severity"] == "block"),
        "latest_completion_audit": str(
            ACTIVE / "current_wizard_hyperliquid_completion_audit_summary.md"
        ),
        "latest_gap_analysis": str(ROOT / "reports/gap_analysis/latest_gap_analysis.md"),
        "latest_pre_mortem": str(ROOT / "reports/pre_mortem/latest_pre_mortem.md"),
        "latest_post_mortem": str(ROOT / "reports/post_mortem/latest_post_mortem.md"),
        "latest_red_team": str(ROOT / "reports/red_team/latest_red_team.md"),
        "corrective_plan": str(ACTIVE / "corrective_execution_plan.md"),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "baseline_files_sha256": hashlib.sha256(files_path.read_bytes()).hexdigest(),
        "secret_scan_sha256": hashlib.sha256(scan_path.read_bytes()).hexdigest(),
    }
    output = ACTIVE / "corrective_baseline_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
