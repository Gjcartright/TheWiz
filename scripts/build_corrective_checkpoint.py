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
    "uv.lock",
}
INCLUDE_PREFIXES = ("src/quant_platform/", "tests/", "config/", "docs/", ".github/")
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
    "recovery",
    "udemy",
    "transcript",
    "evidence",
    "statistical",
    "canonical",
)
EXCLUDE_PREFIXES = (
    ".venv312/",
    ".venv/",
    ".venv311/",
    ".venv313/",
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
    release_index_path = ACTIVE / "quant_release_index.csv"
    _write_csv(release_index_path, rows, manifest_fields)

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

    created_at = datetime.now(timezone.utc)
    lock_path = ROOT / "uv.lock"
    lock_tracked = bool(str(_git("ls-files", "--", "uv.lock")).strip())
    lock_check = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    clean_checkout_reproducible = bool(
        lock_path.is_file()
        and lock_tracked
        and lock_check.returncode == 0
        and not include_rows
        and not any(row["disposition"] == "review" for row in rows)
    )
    manifest = {
        "run_id": f"corrective_baseline_{created_at.strftime('%Y%m%dT%H%M%SZ')}",
        "created_at_utc": created_at.isoformat(),
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
        "uv_lock_exists": lock_path.is_file(),
        "uv_lock_tracked": lock_tracked,
        "uv_lock_check_passed": lock_check.returncode == 0,
        "uv_lock_check_output": lock_check.stdout.strip(),
        "clean_checkout_reproducible": clean_checkout_reproducible,
        "release_status": (
            "READY_CLEAN_CHECKOUT"
            if clean_checkout_reproducible
            else "BLOCKED_UNCOMMITTED_OR_UNREVIEWED_CANDIDATE"
        ),
        "release_blockers": [
            blocker
            for blocker, blocked in (
                ("uv_lock_missing", not lock_path.is_file()),
                ("uv_lock_untracked", not lock_tracked),
                ("uv_lock_inconsistent", lock_check.returncode != 0),
                ("quant_source_changes_uncommitted", bool(include_rows)),
                (
                    "unclassified_release_paths_require_review",
                    any(row["disposition"] == "review" for row in rows),
                ),
                (
                    "secret_scan_blocker",
                    any(finding["severity"] == "block" for finding in findings),
                ),
            )
            if blocked
        ],
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
    release_md = ACTIVE / "quant_release_index.md"
    release_md.write_text(
        "\n".join(
            [
                "# Quant Release Index",
                "",
                "The existing repo is left intact. This index is the reviewed active release boundary.",
                "Generated evidence and `apps/the-ave` remain outside the quant release.",
                "",
                f"- Status: `{manifest['release_status']}`",
                f"- Commit candidates: `{len(include_rows)}`",
                f"- Review paths: `{counts.get('review', 0)}`",
                f"- Excluded paths: `{counts.get('exclude', 0)}`",
                f"- `uv.lock` exists / tracked / consistent: `{lock_path.is_file()}` / `{lock_tracked}` / `{lock_check.returncode == 0}`",
                f"- Secret blockers: `{manifest['secret_blockers']}`",
                f"- Clean-checkout reproducible: `{clean_checkout_reproducible}`",
                f"- Blockers: `{' ; '.join(manifest['release_blockers']) or 'none'}`",
                "",
                "A local passing environment is not clean-checkout proof. The release stays blocked until the reviewed source set and lockfile are committed and CI passes from that commit.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
