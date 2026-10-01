"""Verify selected test ports against preserved runtime bytes and focused results."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree


REPORT = "audit/GATE0_SELECTIVE_TEST_CANDIDATE_QUALIFICATION_2026-09-30.json"
PREVIOUS = "audit/GATE0_REMAINING_PENDING_TEST_SUPPORT_RECONCILIATION_2026-09-30.json"


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


def junit_counts(paths: list[Path]) -> dict[str, dict[str, int]]:
    results: dict[str, dict[str, int]] = defaultdict(
        lambda: {"run": 0, "pass": 0, "fail": 0, "error": 0, "skipped": 0}
    )
    for path in paths:
        for case in ElementTree.parse(path).getroot().iter("testcase"):
            name = case.attrib["classname"].split(".")[-1]
            if name == "test_phase00_expansion_interfaces_runtime_candidate":
                name = "test_phase00_expansion_interfaces"
            result = results[f"tests/{name}.py"]
            result["run"] += 1
            disposition = "pass"
            for child in case:
                if child.tag in {"failure", "error", "skipped"}:
                    disposition = "fail" if child.tag == "failure" else child.tag
            result[disposition] += 1
    return dict(results)


def verify(repo: Path, runtime_root: Path) -> dict[str, object]:
    report_path = repo / REPORT
    report = json.loads(report_path.read_text())
    assert report["schema_version"] == "gate0.selective_test_candidate_qualification.v1"
    scope = report["scope"]
    assert scope["selected_base_commit"] == "ce7131b94ebcd18eb324dd83ab51d213eac0ff66"
    assert scope["qualification_tip_commit"] == "4d8b6a24de89cd51103031a49573b31bba2c02d5"
    assert sha((repo / PREVIOUS).read_bytes()) == scope["source_cohort_report_sha256"]
    previous = json.loads((repo / PREVIOUS).read_text())
    candidate_paths = sorted(
        row["relative_path"] for row in previous["rows"]
        if row["decision"] == "selective-port-candidate"
    )
    rows = report["rows"]
    assert [row["relative_path"] for row in rows] == candidate_paths
    assert len(rows) == scope["paths"] == 13
    assert Path(scope["runtime_root"]) == runtime_root

    for key, digest in report["focused_evidence"].items():
        assert sha((repo / report["focused_evidence_paths"][key]).read_bytes()) == digest
    evidence = repo / "audit/gate0_selective_candidate_evidence"
    full_results = junit_counts([
        evidence / "phase00.xml", evidence / "math.xml", evidence / "teacher.xml",
        evidence / "rest.xml",
    ])
    partial_results = junit_counts([evidence / "experiment_partial.xml"])
    capture = (evidence / "capture.tap").read_text()
    assert "# tests 17\n" in capture
    assert "# pass 17\n" in capture
    assert "# fail 0\n" in capture
    full_results["tests/test_capture_input_minimization.cjs"] = {
        "run": 17, "pass": 17, "fail": 0, "error": 0, "skipped": 0,
    }
    exact = partial = 0
    for row in rows:
        path = row["relative_path"]
        runtime = (runtime_root / path).read_bytes()
        current = (repo / path).read_bytes()
        before = git_bytes(repo, scope["selected_base_commit"], path)
        source = git_bytes(repo, scope["selected_base_commit"], row["selected_base_source_path"])
        committed = git_bytes(repo, row["qualifying_commit"], path)
        assert source is not None
        assert sha(runtime) == row["preserved_runtime_test_sha256"]
        assert sha(current) == row["candidate_port_test_sha256"]
        # Cherry-picks create new commit IDs in the receiving repository. The
        # original candidate commit is optional provenance there.
        if committed is not None:
            assert sha(committed) == row["candidate_port_test_sha256"]
        assert row["selected_base_test_sha256"] == (sha(before) if before is not None else None)
        assert row["selected_base_source_sha256"] == sha(source)
        assert row["full_preserved_test_result"] == full_results[path]
        assert row["authority_granted"] is False
        assert row["provider_calls_made"] is False
        assert row["rationale"]
        if row["port_kind"] == "exact-test-only":
            exact += 1
            assert row["queue_status_recommendation"] == "REVIEWED_SELECTIVE_TEST_PORTED"
            assert runtime == current
            assert row["safe_to_port_exact_test_alone"] is True
            assert row["full_preserved_test_result"]["fail"] == 0
        else:
            partial += 1
            assert path == "tests/test_experiment_regime_clock.py"
            assert row["port_kind"] == "partial-test-only"
            assert row["queue_status_recommendation"] == (
                "REVIEWED_SELECTIVE_TEST_PARTIAL_REMAINDER_BLOCKED"
            )
            assert runtime != current
            assert row["safe_to_port_exact_test_alone"] is False
            assert row["full_preserved_test_result"] == {
                "run": 22, "pass": 15, "fail": 7, "error": 0, "skipped": 0,
            }
            assert row["selected_subset_result"] == partial_results[path]
            assert partial_results[path] == {
                "run": 13, "pass": 13, "fail": 0, "error": 0, "skipped": 0,
            }
            names = {node.name for node in ast.walk(ast.parse(current)) if isinstance(node, ast.FunctionDef)}
            for omitted in row["omitted_functions"]:
                assert omitted["name"] not in names
            assert sum(item["failing_cases"] for item in row["omitted_functions"]) == 7
            failures = [
                case.attrib["name"]
                for case in ElementTree.parse(evidence / "rest.xml").getroot().iter("testcase")
                if case.attrib["classname"].endswith("test_experiment_regime_clock")
                and any(child.tag in {"failure", "error"} for child in case)
            ]
            assert row["blocked_remainder_exact_failures"] == failures
            conflict = row["selected_skip_policy_conflict"]
            conflict_bytes = git_bytes(
                repo, scope["selected_base_commit"], conflict["relative_path"]
            )
            assert conflict_bytes is not None
            assert sha(conflict_bytes) == conflict["selected_base_sha256"]
            conflict_names = {
                node.name for node in ast.walk(ast.parse(conflict_bytes))
                if isinstance(node, ast.FunctionDef)
            }
            assert conflict["test_name"] in conflict_names
    assert exact == scope["exact_test_only_ports"] == 12
    assert partial == scope["partial_test_only_ports"] == 1
    broad = report["broad_suite"]
    assert (broad["run"], broad["passed"], broad["failed"], broad["errors"], broad["skipped"]) == (
        3132, 3131, 1, 0, 0,
    )
    compressed = (evidence / "full_suite.xml.gz").read_bytes()
    xml_bytes = gzip.decompress(compressed)
    assert sha(compressed) == broad["junit_gzip_sha256"]
    assert sha(xml_bytes) == broad["junit_xml_sha256"]
    suite = ElementTree.fromstring(xml_bytes).find("testsuite")
    assert suite is not None
    assert [int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")] == [
        3132, 1, 0, 0,
    ]
    failures = [
        case for case in suite.iter("testcase")
        if any(child.tag in {"failure", "error"} for child in case)
    ]
    assert len(failures) == 1
    assert broad["failure"] == (
        failures[0].attrib["classname"].replace(".", "/") + ".py::" + failures[0].attrib["name"]
    )
    assert sha((evidence / "full_suite.log").read_bytes()) == broad["suite_log_sha256"]
    assert b"1 failed, 3131 passed" in (evidence / "full_suite.log").read_bytes()
    baseline_xml = (evidence / "base_financial_inventory.xml").read_bytes()
    baseline_log = (evidence / "base_financial_inventory.log").read_bytes()
    assert sha(baseline_xml) == broad["baseline_failure_junit_sha256"]
    assert sha(baseline_log) == broad["baseline_failure_log_sha256"]
    assert b"assert 17 == 19" in baseline_log
    baseline_cases = list(ElementTree.fromstring(baseline_xml).iter("testcase"))
    assert len(baseline_cases) == 1
    assert any(child.tag == "failure" for child in baseline_cases[0])
    tip_check = subprocess.run(
        ["git", "cat-file", "-e", scope["qualification_tip_commit"] + "^{commit}"],
        cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    if tip_check.returncode == 0:
        unchanged = subprocess.run(
            ["git", "diff", "--quiet", scope["selected_base_commit"],
             scope["qualification_tip_commit"], "--",
             "src", "tests/test_corrective_financial_effect_registry.py"],
            cwd=repo, check=False,
        )
        assert unchanged.returncode == 0
    assert broad["source_changes_from_selected_base"] == 0
    assert broad["failing_test_changes_from_selected_base"] == 0
    return {
        "status": "PASS",
        "paths": len(rows),
        "exact_test_only_ports": exact,
        "partial_test_only_ports": partial,
        "broad_suite": {"passed": 3131, "failed_preexisting": 1},
        "report_sha256": sha(report_path.read_bytes()),
        "provider_calls_made": False,
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-root", type=Path,
        default=Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime"),
    )
    args = parser.parse_args()
    print(json.dumps(verify(repo, args.runtime_root), sort_keys=True))


if __name__ == "__main__":
    main()
