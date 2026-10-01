"""Reconcile the dated same-path custody scan against the active Git lineage.

This produces a source-decision queue. It never copies historical code into the
active checkout and does not treat a hash difference as a semantic regression.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "audit/SECOND_PASS_SAME_PATH_DRIFT_2026-09-29.csv"
OUTPUT = ROOT / "audit/GATE0_SOURCE_DELTA_REGISTER_2026-09-30.csv"
SUMMARY = ROOT / "audit/GATE0_SOURCE_DELTA_SUMMARY_2026-09-30.json"
ANCESTOR = "56a19218414173d687d57f4eaf0e2abd321b15df"
HEAD = "b98d5b9447a109938b57b239a7d458f47606d7c9"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def committed_sha(ref: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, check=False
    )
    return digest(result.stdout) if result.returncode == 0 else ""


def kind(path: str) -> str:
    if path.startswith(("src/", "scripts/", "config/")) or path in {
        "pyproject.toml", "uv.lock", ".gitignore"
    }:
        return "source_or_dependency"
    if path.startswith("tests/"):
        return "test"
    if path.startswith("docs/"):
        return "documentation"
    return "historical_evidence"


def main() -> None:
    with INPUT.open(newline="") as handle:
        original = list(csv.DictReader(handle))
    variants: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in original:
        variants[(row["relative_path"], row["copy_sha256"])].add(row["copy_root"])
    paths = {path for path, _ in variants}
    hashes: dict[str, tuple[str, str, str]] = {}
    for path in sorted(paths):
        active = ROOT / path
        working_sha = digest(active.read_bytes()) if active.is_file() else ""
        hashes[path] = (working_sha, committed_sha(HEAD, path), committed_sha(ANCESTOR, path))
    rows = []
    for (path, copy_sha), roots in sorted(variants.items()):
        working_sha, head_sha, ancestor_sha = hashes[path]
        category = kind(path)
        if copy_sha == working_sha:
            decision = "MATCHES_ACTIVE_WORKTREE"
        elif copy_sha == head_sha:
            decision = "MATCHES_ACTIVE_HEAD"
        elif copy_sha == ancestor_sha:
            decision = "ANCESTOR_VERSION_RETAIN_NO_PORT"
        elif category == "historical_evidence":
            decision = "RETAIN_ARCHIVE_NOT_CANONICAL_SOURCE"
        else:
            decision = "SEMANTIC_REVIEW_REQUIRED_NO_PORT"
        rows.append({
            "relative_path": path,
            "category": category,
            "copy_sha256": copy_sha,
            "copy_roots": ";".join(sorted(roots)),
            "active_worktree_sha256": working_sha,
            "active_head_sha256": head_sha,
            "ancestor_56a1921_sha256": ancestor_sha,
            "decision": decision,
        })
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "thewiz.gate0.source_delta.v1",
        "canonical_candidate": str(ROOT),
        "source_scan": str(INPUT.relative_to(ROOT)),
        "active_head": HEAD,
        "historical_ancestor": ANCESTOR,
        "original_drift_rows": len(original),
        "unique_paths": len(paths),
        "unique_path_hash_variants": len(rows),
        "decision_counts": dict(sorted(Counter(row["decision"] for row in rows).items())),
        "category_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
        "meaning": "Hash identity is custody evidence; semantic review remains open for non-ancestor source and test variants.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
