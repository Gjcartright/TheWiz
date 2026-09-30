"""Verify the 33 added orchestration union rows from frozen local evidence.

Only local Git objects and files are read; no project code is imported.
"""

from __future__ import annotations

import csv
import difflib
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "audit/GATE0_ORCHESTRATION_UNION_OMISSION33_2026-09-30.json"
PROPOSAL = (
    ROOT / "audit/GATE0_ORCHESTRATION_UNION_OMISSION33_QUEUE_PROPOSAL_2026-09-30.csv"
)
PREFIX = "src/quant_platform/orchestration/"


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _git_bytes(commit: str, relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def verify() -> dict[str, int]:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    base = report["base_commit"]
    old = {
        row["relative_path"]
        for row in _rows(_git_bytes(base, "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"))
    }
    expanded = {
        row["relative_path"]: row
        for row in _rows(Path(report["expanded_union_queue_path"]).read_bytes())
    }
    expected = {path for path in expanded.keys() - old if path.startswith(PREFIX)}
    records = report["paths"]
    assert len(records) == len(expected) == 33
    assert {record["relative_path"] for record in records} == expected
    freeze = {
        row["relative_path"]: row
        for row in _rows(
            _git_bytes(base, "audit/evidence_freeze_2026-09-29/source_reconciliation.csv")
        )
    }
    proposal = {
        row["relative_path"]: row for row in csv.DictReader(PROPOSAL.open(newline=""))
    }
    assert set(proposal) == expected

    copies = 0
    for record in records:
        path = record["relative_path"]
        queue = expanded[path]
        frozen = freeze[path]
        active = _git_bytes(base, path)
        assert _sha(active) == record["active_sha256"]
        assert frozen["working_vs_runtime"] == "SAME"
        assert frozen["decision"] == "NO_PORT_NEEDED"
        assert queue["in_four_root_reconciliation"] == "False"
        assert queue["historical_variant_count"] == "1"
        assert queue["historical_variant_sha256"] == record["historical_sha256"]
        for label in ("working", "recovery", "runtime"):
            entry = record["four_root_copies"][label]
            assert entry["sha256"] == frozen[f"{label}_sha256"]
            assert _sha(Path(entry["absolute_path"]).read_bytes()) == entry["sha256"]
            copies += 1
        variants = {}
        for entry in record["extended_copies"]:
            assert _sha(Path(entry["absolute_path"]).read_bytes()) == entry["sha256"]
            variants[entry["root_label"]] = entry
            copies += 1
        assert set(variants) == {"visual_recovery", "forensic_b", "source_candidate"}
        assert variants["source_candidate"]["sha256"] == record["active_sha256"]
        assert variants["visual_recovery"]["sha256"] == record["historical_sha256"]
        assert variants["forensic_b"]["sha256"] == record["historical_sha256"]
        historical = Path(variants["forensic_b"]["absolute_path"]).read_text()
        diff = "".join(
            difflib.unified_diff(
                active.decode("utf-8").splitlines(keepends=True),
                historical.splitlines(keepends=True),
                fromfile="active",
                tofile="forensic_b",
                n=1,
            )
        )
        assert _sha(diff.encode("utf-8")) == record["unified_diff_sha256"]
        assert len(diff.splitlines()) == record["unified_diff_lines"]
        for field in (
            "active_sha256",
            "historical_sha256",
            "proposed_custody_status",
            "proposed_decision_rationale",
            "proposed_decision_evidence",
        ):
            assert proposal[path][field] == record[field]
    return {"paths": len(records), "verified_copies": copies}


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
