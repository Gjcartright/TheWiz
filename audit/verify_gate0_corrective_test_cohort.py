"""Verify the frozen 25-path corrective test source reconciliation.

This reads Git objects and local files only. It never imports project modules,
runs pytest, starts a scheduler, or calls a provider.
"""

from __future__ import annotations

import ast
import csv
import difflib
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "audit/GATE0_CORRECTIVE_TEST_COHORT_2026-09-30.json"
PROPOSAL = ROOT / "audit/GATE0_CORRECTIVE_TEST_COHORT_QUEUE_PROPOSAL_2026-09-30.csv"
PREFIX = "tests/test_corrective_"
PENDING = "PRESERVED_REVIEW_REQUIRED_NO_PORT"


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _git_bytes(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def _test_names(data: bytes) -> list[str]:
    if not data:
        return []
    return sorted(
        node.name
        for node in ast.parse(data.decode("utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def _diff(a: bytes, b: bytes, label: str) -> dict[str, object]:
    text = "".join(
        difflib.unified_diff(
            a.decode("utf-8").splitlines(keepends=True),
            b.decode("utf-8").splitlines(keepends=True),
            fromfile="active",
            tofile=label,
            n=1,
        )
    )
    return {"sha256": _sha(text.encode("utf-8")), "lines": len(text.splitlines())}


def verify() -> dict[str, int]:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    base = report["base_commit"]
    queue = {
        row["relative_path"]: row
        for row in _rows(_git_bytes(base, report["queue_path"]))
        if row["relative_path"].startswith(PREFIX)
        and row["custody_status"] == PENDING
    }
    records = report["paths"]
    assert len(queue) == len(records) == 25
    assert {record["relative_path"] for record in records} == set(queue)
    freeze = {
        row["relative_path"]: row
        for row in _rows(_git_bytes(base, report["four_root_freeze_path"]))
    }
    proposal = {
        row["relative_path"]: row for row in csv.DictReader(PROPOSAL.open(newline=""))
    }
    current_queue = {
        row["relative_path"]: row
        for row in _rows((ROOT / report["queue_path"]).read_bytes())
    }
    assert set(proposal) == set(queue)

    present_copies = 0
    for record in records:
        path = record["relative_path"]
        row = queue[path]
        frozen = freeze[path]
        active = _git_bytes(base, path) if record["active_sha256"] else b""
        assert _sha(active) == record["active_sha256"] if active else not record["active_sha256"]
        assert record["working_vs_runtime"] == frozen["working_vs_runtime"]
        assert row["in_four_root_reconciliation"] == "True"
        assert row["working_sha256_at_freeze"] == frozen["working_sha256"]
        assert row["runtime_sha256_at_freeze"] == frozen["runtime_sha256"]
        frozen_hashes = set()
        for label in ("working", "recovery", "runtime"):
            entry = record["four_root_copies"][label]
            assert entry["sha256_at_freeze"] == frozen[f"{label}_sha256"]
            local = Path(entry["absolute_path"])
            observed = _sha(local.read_bytes()) if local.is_file() else ""
            assert observed == entry["sha256_now"]
            assert local.is_file() == entry["exists"]
            if entry["frozen_git_blob_commit"]:
                assert _sha(_git_bytes(entry["frozen_git_blob_commit"], path)) == entry["sha256_at_freeze"]
            else:
                assert observed == entry["sha256_at_freeze"]
            if observed:
                present_copies += 1
            if entry["sha256_at_freeze"]:
                frozen_hashes.add(entry["sha256_at_freeze"])

        extended_hashes = set()
        variants = {}
        for entry in record["extended_copies"]:
            assert _sha(Path(entry["absolute_path"]).read_bytes()) == entry["sha256"]
            variants[entry["root_label"]] = entry
            if entry["root_label"] != "source_candidate" and entry["sha256"] not in frozen_hashes:
                extended_hashes.add(entry["sha256"])
            present_copies += 1
        assert sorted(extended_hashes) == record["historical_variant_sha256"]
        assert sorted(extended_hashes) == sorted(filter(None, row["historical_variant_sha256"].split(";")))
        assert len(extended_hashes) == int(row["historical_variant_count"])
        runtime = Path(record["four_root_copies"]["runtime"]["absolute_path"]).read_bytes()
        assert _test_names(active) == record["active_test_names"]
        assert _test_names(runtime) == record["runtime_test_names"]
        assert sorted(set(_test_names(runtime)) - set(_test_names(active))) == record["runtime_only_test_names"]
        assert record["comparison_diffs"]["runtime"] == _diff(active, runtime, "runtime")
        if "forensic_b" in variants:
            historical = Path(variants["forensic_b"]["absolute_path"]).read_bytes()
            assert record["comparison_diffs"]["forensic_b"] == _diff(active, historical, "forensic_b")

        source = record["primary_source_contract"]
        # The report describes the source contract at its frozen base commit.
        # Later reviewed ports may change the active file without changing this
        # historical comparison.
        try:
            selected_source = _git_bytes(base, source["relative_path"])
        except subprocess.CalledProcessError:
            selected_source = None
        runtime_source = Path("/Users/gregc/TheWiz-LocalRuntime") / source["relative_path"]
        assert (selected_source is not None) == source["active_exists"]
        assert runtime_source.is_file() == source["runtime_exists"]
        assert (_sha(selected_source) if selected_source is not None else "") == source["active_sha256"]
        assert _sha(runtime_source.read_bytes()) == source["runtime_sha256"]
        for key in (
            "active_sha256",
            "working_vs_runtime",
            "queue_current_status",
            "proposed_custody_status",
            "proposed_decision_rationale",
            "proposed_decision_evidence",
        ):
            assert proposal[path][key] == record[key]
        current = current_queue[path]
        assert current["custody_status"] == proposal[path]["proposed_custody_status"]
        assert proposal[path]["proposed_decision_rationale"] in current["decision_rationale"]
        for evidence in proposal[path]["proposed_decision_evidence"].split(";"):
            assert evidence in current["decision_evidence"].split("; ")
    return {"paths": len(records), "verified_present_copies": present_copies}


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
