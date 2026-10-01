"""Verify exact custody bytes for the original-811 pending test/support cohort."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import subprocess
from collections import Counter
from pathlib import Path


REPORT = "GATE0_REMAINING_PENDING_TEST_SUPPORT_RECONCILIATION_2026-09-30.json"
PROPOSAL = "GATE0_REMAINING_PENDING_TEST_SUPPORT_QUEUE_UPDATE_2026-09-30.csv"
QUEUE = "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
ORIGINAL_QUEUE_SNAPSHOT = "audit/GATE0_ORIGINAL_811_SOURCE_QUEUE_SNAPSHOT_2026-09-30.csv"
MANIFEST_COHORT_SNAPSHOT = (
    "audit/GATE0_REMAINING_PENDING_TEST_SUPPORT_MANIFEST_SNAPSHOT_2026-09-30.csv"
)
MANIFEST_COHORT_SHA256 = "b0b4f115d6f8b024410da59a1d1c37414be05a62438147f4ab716e4e5370378d"
FULL_MANIFEST_ORIGIN_SHA256 = "7042b2e8fd82ebeb4922d82f39d2823cc350ed38642ef1c734d4efd5e9237918"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_bytes(repo: Path, commit: str, path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def verify(repo: Path, active: Path, report_path: Path, proposal_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    assert report["schema_version"] == "gate0.remaining_pending_test_support.v1"
    scope = report["scope"]
    commit = scope["selected_active_commit"]
    # These snapshots keep the original membership and in-scope variant rows
    # stable after the live queue and source tree move forward.
    original_bytes = (repo / ORIGINAL_QUEUE_SNAPSHOT).read_bytes()
    selected_bytes = git_bytes(active, commit, QUEUE)
    manifest_bytes = (repo / MANIFEST_COHORT_SNAPSHOT).read_bytes()
    assert selected_bytes is not None
    assert sha(original_bytes) == scope["original_811_queue_sha256"]
    assert sha(selected_bytes) == scope["selected_active_891_queue_sha256"]
    assert sha(manifest_bytes) == MANIFEST_COHORT_SHA256
    assert scope["extended_manifest_sha256"] == FULL_MANIFEST_ORIGIN_SHA256
    original = {r["relative_path"] for r in csv.DictReader(io.StringIO(original_bytes.decode()))}
    selected_rows = list(csv.DictReader(io.StringIO(selected_bytes.decode())))
    selected = {r["relative_path"]: r for r in selected_rows}
    assert len(original) == 811 and len(selected_rows) == 891
    excluded = scope["excluded_prefixes"]
    assert excluded == [
        "test_collection_", "test_corrective_", "test_wizard_",
        "test_current_wizard_", "test_hyperliquid_", "test_binance_",
        "test_pair_", "test_research_", "test_v2_", "test_youtube_",
        "test_execution",
    ]
    paths = sorted(
        p for p in original
        if p.startswith("tests/")
        and selected[p]["custody_status"] == "PRESERVED_REVIEW_REQUIRED_NO_PORT"
        and not any(p.startswith("tests/" + prefix) for prefix in excluded)
    )
    assert len(paths) == scope["reviewed_paths"] == 107
    rows = report["rows"]
    with proposal_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(selected_rows[0])
        proposed = list(reader)
    assert len(rows) == len(proposed) == 107
    assert [r["relative_path"] for r in rows] == paths
    assert [r["relative_path"] for r in proposed] == paths
    manifest = {}
    manifest_rows = list(csv.DictReader(io.StringIO(manifest_bytes.decode())))
    assert len(manifest_rows) == 164
    for row in manifest_rows:
        assert row["status"] == "hashed"
        assert row["relative_path"] in paths
        manifest.setdefault(row["relative_path"], []).append(row)
    runtime = Path(scope["runtime_root"])
    source_candidate = Path(scope["source_candidate_root"])
    decisions = Counter()
    copies = 0
    active_present = 0
    for row, update in zip(rows, proposed):
        path = row["relative_path"]
        base = selected[path]
        selected_bytes = git_bytes(active, commit, path)
        runtime_bytes = (runtime / path).read_bytes()
        candidate_path = source_candidate / path
        candidate_bytes = candidate_path.read_bytes() if candidate_path.exists() else None
        assert row["selected_active_test_sha256"] == (
            sha(selected_bytes) if selected_bytes is not None else None
        )
        active_present += selected_bytes is not None
        assert row["runtime_test_sha256"] == sha(runtime_bytes)
        assert row["source_candidate_test_sha256"] == (
            sha(candidate_bytes) if candidate_bytes is not None else None
        )
        assert row["queue_working_sha256_at_freeze"] == base["working_sha256_at_freeze"]
        assert row["selected_drifted_from_queue_freeze"] == bool(
            selected_bytes is not None
            and base["working_sha256_at_freeze"]
            and sha(selected_bytes) != base["working_sha256_at_freeze"]
        )
        if base["runtime_sha256_at_freeze"]:
            assert sha(runtime_bytes) == base["runtime_sha256_at_freeze"]
        assert row["queue_historical_variant_sha256"] == base["historical_variant_sha256"]
        if path.endswith(".py"):
            count = sum(
                isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name.startswith("test_")
                for n in ast.walk(ast.parse(runtime_bytes))
            )
            assert row["runtime_test_function_count"] == count
        else:
            assert row["runtime_test_function_count"] is None
        copies_for_path = sorted(
            manifest.get(path, []), key=lambda r: (r["root_label"], r["sha256"])
        )
        assert len(row["preserved_manifest_copies"]) == len(copies_for_path)
        for actual, recorded in zip(copies_for_path, row["preserved_manifest_copies"]):
            assert recorded == {
                "root_label": actual["root_label"],
                "sha256": actual["sha256"],
                "absolute_path": actual["absolute_path"],
            }
            assert sha(Path(actual["absolute_path"]).read_bytes()) == actual["sha256"]
            copies += 1
        history = set(base["historical_variant_sha256"].split(";")) - {""}
        assert history <= {r["sha256"] for r in copies_for_path}
        target = row["primary_target"]
        if target and "/" in target and not target.startswith("quant_platform."):
            target_bytes = git_bytes(active, commit, target)
            assert row["primary_target_selected_sha256"] == (
                sha(target_bytes) if target_bytes is not None else None
            )
        assert row["authority_granted"] is False
        assert row["provider_calls_made"] is False
        assert row["semantic_rationale"]
        assert update["decision_rationale"] == row["semantic_rationale"]
        assert update["decision_evidence"] == f"audit/{REPORT}"
        assert update["custody_status"] == row["queue_custody_status"]
        for field, value in base.items():
            if field not in {"custody_status", "decision_rationale", "decision_evidence"}:
                assert update[field] == value, (path, field)
        decision = row["decision"]
        decisions[decision] += 1
        if decision == "no-port":
            assert selected_bytes is not None
            assert row["port_recommendation"] == "retain-selected"
            assert row["queue_custody_status"] == "REVIEWED_RETAIN_SELECTED_TEST_NO_PORT"
        elif decision == "dependency-blocked":
            assert row["port_recommendation"] == "defer-until-source-dependency-reviewed"
            assert row["queue_custody_status"] == "REVIEWED_DEPENDENCY_BLOCKED_TEST_NO_PORT"
        elif decision == "selective-port-candidate":
            assert row["port_recommendation"] == "focused-qualification-before-port"
            assert row["queue_custody_status"] == "PRESERVED_REVIEW_REQUIRED_NO_PORT"
            assert row["primary_target_selected_sha256"] is not None
        else:
            raise AssertionError((path, decision))
    assert dict(sorted(decisions.items())) == report["decision_counts"] == {
        "dependency-blocked": 76,
        "no-port": 18,
        "selective-port-candidate": 13,
    }
    assert active_present == scope["selected_active_tests_present"] == 34
    return {
        "status": "PASS",
        "selected_active_commit": commit,
        "reviewed_paths": len(rows),
        "preserved_manifest_copies": copies,
        "decision_counts": dict(sorted(decisions.items())),
        "report_sha256": sha(report_path.read_bytes()),
        "proposal_sha256": sha(proposal_path.read_bytes()),
        "provider_calls_made": False,
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-repo", type=Path, default=Path("/Volumes/Expansion/Crypto Wizard"))
    args = parser.parse_args()
    print(json.dumps(verify(repo, args.active_repo, repo / "audit" / REPORT, repo / "audit" / PROPOSAL), sort_keys=True))


if __name__ == "__main__":
    main()
