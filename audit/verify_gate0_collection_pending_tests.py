"""Read-only verifier for the pending original-811 collection test cohort."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

REPORT_NAME = "GATE0_COLLECTION_PENDING_TEST_RECONCILIATION_2026-09-30.json"
PROPOSAL_NAME = "GATE0_COLLECTION_PENDING_TEST_QUEUE_UPDATE_2026-09-30.csv"
QUEUE_PATH = "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_bytes(repo: Path, commit: str, relative_path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _defined_names(tree: ast.AST) -> dict[str, ast.AST]:
    names: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names[target.id] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names[alias.asname or alias.name] = node
    return names


def verify(
    *,
    active_repo: Path,
    runtime_root: Path,
    source_candidate_root: Path,
    report_path: Path,
    proposal_path: Path,
) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "gate0.collection_pending_tests.v1"
    commit = report["scope"]["selected_active_commit"]
    queue_bytes = _git_bytes(active_repo, commit, QUEUE_PATH)
    assert queue_bytes is not None
    assert _sha(queue_bytes) == report["scope"]["selected_active_891_queue_sha256"]
    queue_rows = list(csv.DictReader(io.StringIO(queue_bytes.decode("utf-8"))))
    queue = {row["relative_path"]: row for row in queue_rows}
    assert len(queue_rows) == 891
    pending = sorted(
        path
        for path, row in queue.items()
        if path.startswith("tests/test_collection_")
        and row["custody_status"] == "PRESERVED_REVIEW_REQUIRED_NO_PORT"
    )
    assert len(pending) == 42
    assert report["scope"]["pending_reviewed"] == 42
    assert report["scope"]["original_collection_test_rows"] == 57
    assert report["scope"]["already_reviewed_excluded"] == 15
    records = report["rows"]
    with proposal_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(queue_rows[0])
        proposal = list(reader)
    assert len(records) == len(proposal) == 42
    assert [item["relative_path"] for item in records] == pending
    assert [item["relative_path"] for item in proposal] == pending

    distinct = 0
    missing_types: dict[str, int] = {}
    for item, proposed in zip(records, proposal, strict=True):
        path = item["relative_path"]
        baseline = queue[path]
        assert _git_bytes(active_repo, commit, path) is None
        runtime = (runtime_root / path).read_bytes()
        candidate = (source_candidate_root / path).read_bytes()
        assert _sha(runtime) == item["runtime_test_sha256"]
        assert _sha(candidate) == item["source_candidate_test_sha256"]
        assert _sha(runtime) == baseline["runtime_sha256_at_freeze"]
        assert _sha(runtime) == baseline["recovery_sha256_at_freeze"]
        assert baseline["working_sha256_at_freeze"] == ""
        assert baseline["working_vs_runtime"] == "RUNTIME_ONLY"
        is_distinct = runtime != candidate
        assert is_distinct == item["source_candidate_distinct"]
        if is_distinct:
            distinct += 1
            assert _sha(candidate) == baseline["historical_variant_sha256"]
            assert baseline["historical_copy_roots"] == "source_candidate"
            assert item["source_candidate_variant_rationale"]
        else:
            assert baseline["historical_variant_sha256"] == ""
            assert item["source_candidate_variant_rationale"] == ""
        tests = sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            for node in ast.walk(ast.parse(runtime))
        )
        assert tests == item["runtime_test_function_count"]
        assert item["decision"] == "dependency-blocked"
        assert item["port_recommendation"] == "no-port-until-coordinated-source-adoption"
        assert item["selected_active_test_exists"] is False
        assert item["authority_granted"] is False
        assert item["provider_calls_made"] is False
        assert proposed["custody_status"] == item["queue_custody_status"]
        assert proposed["custody_status"] == "REVIEWED_DEPENDENCY_BLOCKED_COLLECTION_TEST_NO_PORT"
        assert proposed["decision_evidence"] == f"audit/{REPORT_NAME}"
        assert proposed["decision_rationale"]
        for field, value in baseline.items():
            if field not in {"custody_status", "decision_rationale", "decision_evidence"}:
                assert proposed[field] == value, (path, field)

        dependency = item["primary_dependency"]
        kind = dependency["kind"]
        source_path = dependency["relative_path"]
        source = _git_bytes(active_repo, commit, source_path)
        if kind in {"module", "support_test"}:
            assert source is None, (path, source_path)
            assert dependency["selected_active_sha256"] is None
        else:
            assert source is not None, (path, source_path)
            assert _sha(source) == dependency["selected_active_sha256"]
            named = _defined_names(ast.parse(source))
            if kind == "symbol":
                assert dependency["subject"] not in named, (path, dependency)
            elif kind == "parameter":
                function_name, parameter_name = dependency["subject"].split(":", 1)
                function = named.get(function_name)
                assert isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
                args = function.args
                parameters = [arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs]
                assert parameter_name not in parameters, (path, dependency)
            else:
                raise AssertionError((path, kind))
        assert dependency["source_queue_status"] == queue.get(source_path, {}).get(
            "custody_status", "not_queued"
        )
        missing_types[kind] = missing_types.get(kind, 0) + 1

    assert distinct == 2
    assert report["scope"]["distinct_source_candidate_variants"] == distinct
    assert report["decision_counts"] == {
        "dependency-blocked": 42,
        "no-port": 0,
        "selective-port-candidate": 0,
    }
    return {
        "status": "PASS",
        "selected_active_commit": commit,
        "validated_paths": len(records),
        "distinct_source_candidate_variants": distinct,
        "primary_dependency_types": dict(sorted(missing_types.items())),
        "report_sha256": _sha(report_path.read_bytes()),
        "proposal_sha256": _sha(proposal_path.read_bytes()),
        "provider_calls_made": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--active-repo", type=Path, default=repo_root)
    parser.add_argument("--runtime-root", type=Path, default=Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime"))
    parser.add_argument("--source-candidate-root", type=Path, default=Path("/Volumes/Expansion/TheWiz-Workspace/reconciliation-inputs/current-source-candidate"))
    parser.add_argument("--report", type=Path, default=repo_root / "audit" / REPORT_NAME)
    parser.add_argument("--proposal", type=Path, default=repo_root / "audit" / PROPOSAL_NAME)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                active_repo=args.active_repo,
                runtime_root=args.runtime_root,
                source_candidate_root=args.source_candidate_root,
                report_path=args.report,
                proposal_path=args.proposal,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
