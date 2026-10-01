"""Apply exact original pending non-orchestration source decisions."""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
PROPOSAL = ROOT / "audit/GATE0_NONORCHESTRATION_SOURCE11_QUEUE_PROPOSAL_2026-09-30.csv"
BASE = "f248e59"
FIELDS = {"custody_status", "decision_rationale", "decision_evidence"}
PORTS = {
    "src/quant_platform/binance_testnet.py": (
        "REVIEWED_CANONICAL_PAIR_METHOD_GUARD_PORT_REMAINDER_NO_WHOLE_FILE_PORT",
        " The canonical entry and rollback method guard and regressions were ported in ad98345; broader historical attestation remains a separate dependency family.",
        "ad98345",
    ),
    "src/quant_platform/env.py": (
        "REVIEWED_SECRET_PROVENANCE_GUARD_PORT_REMAINDER_NO_WHOLE_FILE_PORT",
        " The owner-only regular-file and inode-bound pre-read guard with regressions was ported in f248e59 while retaining both active loader APIs.",
        "f248e59",
    ),
    "src/quant_platform/execution.py": (
        "REVIEWED_NARROW_AUTHORITY_PORT_REMAINDER_DEFERRED_NO_WHOLE_FILE_PORT",
        " The live dYdX account snapshot now remains authoritative over browser claims after ad98345; broader runtime adapter attestation remains a separate dependency family.",
        "ad98345",
    ),
    "src/quant_platform/meta_learning.py": (
        "REVIEWED_MODEL_READINESS_REPAIR_PORT_REMAINDER_NO_WHOLE_FILE_PORT",
        " Unverified return claims can no longer count as modeling outcomes; the report and P5 flags fail closed after dab88aa, 2ddaeeb, and 8a87d4b. A receipt-bound outcome verifier remains a later-gate dependency.",
        "dab88aa;2ddaeeb;8a87d4b",
    ),
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(*, apply: bool) -> None:
    proposal = read(PROPOSAL)
    if len(proposal) != 11 or len({item["relative_path"] for item in proposal}) != 11:
        raise ValueError("source11 proposal membership changed")
    expected = {item["relative_path"]: item for item in proposal}
    if PORTS.keys() - expected.keys():
        raise ValueError("ported source absent from proposal")
    for path, (status, note, commits) in PORTS.items():
        item = expected[path]
        item["custody_status"] = status
        item["decision_rationale"] += note
        item["decision_evidence"] += "; selective source/test repairs " + commits
    base_data = subprocess.check_output(
        ["git", "show", f"{BASE}:{QUEUE.relative_to(ROOT)}"], cwd=ROOT
    )
    baseline = {item["relative_path"]: item for item in csv.DictReader(io.StringIO(base_data.decode()))}
    rows = read(QUEUE)
    if len(rows) != len(baseline) or not expected.keys() <= baseline.keys():
        raise ValueError("queue membership changed")
    for item in rows:
        path = item["relative_path"]
        if path not in expected:
            continue
        before = baseline[path]
        target = expected[path]
        if before["custody_status"] != "PRESERVED_REVIEW_REQUIRED_NO_PORT":
            raise ValueError(f"source was not pending: {path}")
        for key in before.keys() - FIELDS:
            if item[key] != before[key]:
                raise ValueError(f"frozen custody field changed: {path} {key}")
        if apply:
            if any(item[key] != before[key] for key in FIELDS) and any(
                item[key] != target[key] for key in FIELDS
            ):
                raise ValueError(f"decision edited unexpectedly: {path}")
            item.update({key: target[key] for key in FIELDS})
        elif any(item[key] != target[key] for key in FIELDS):
            raise ValueError(f"decision mismatch: {path}")
    if apply:
        with QUEUE.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(f"PASS {'applied' if apply else 'verified'} 11 non-orchestration source decisions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    run(apply=parser.parse_args().apply)
