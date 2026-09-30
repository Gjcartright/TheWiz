#!/usr/bin/env python3
"""Verify the historical collection runtime/configuration no-port decision."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTED_COMMIT = "0b60eadf4f6a8d92cf9653c952f184aa3ea97146"
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")
QUEUE = ROOT / "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"
OUTPUT = ROOT / "audit/GATE0_COLLECTION_RUNTIME_CONFIG_SOURCE_RECONCILIATION_2026-09-30.json"
SOURCES = {
    "config/collection_launch_runtime.json": "b5abb8736eda54c79155febe3f0246a331833a29990bf26e4f6722d1f945ba84",
    "config/collection_prestart_runtime_candidate.json": "48dc1134a1489b9cb066ea55481490693077e25a049d4b06288bd796e3c0c715",
    "config/collection_verification_batches.json": "3f9f44b2572e26504e74bcb23fcf8a28281ee76916da72b3394e485844d69b8d",
    "config/collection_verification_scope.json": "3cd182480eba6f926f3b48dd03c2856db28fadefc3ebfe7bbe9abdd4e23749e8",
}
OLD_LOCK_SHA256 = "bc67c17f3873991bc3dfb7a3bf782153d81838402a9c6057de4be9a854549a9a"
SELECTED_LOCK_SHA256 = "c8955d2a94ad5d7c97da861cab9db9a32d3d489d17825052632ee9b96dcad634"
QUEUE_SHA256 = "2be83b0b1f9a599b11618ddacb76347dd5979a049ac6a2707cf8fe2d882562f4"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selected_bytes(relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{SELECTED_COMMIT}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout


def selected_exists(relative: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{SELECTED_COMMIT}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    ).returncode == 0


def main() -> None:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SELECTED_COMMIT, "HEAD"],
        cwd=ROOT,
        check=True,
    )
    source_data = {}
    for relative, expected in SOURCES.items():
        path = SAVEPOINT / relative
        if selected_exists(relative) or not path.is_file() or sha256(path.read_bytes()) != expected:
            raise ValueError(f"historical collection config custody changed: {relative}")
        source_data[relative] = json.loads(path.read_bytes())

    if sha256(selected_bytes("uv.lock")) != SELECTED_LOCK_SHA256:
        raise ValueError("selected canonical lock changed")
    launch = source_data["config/collection_launch_runtime.json"]
    candidate = source_data["config/collection_prestart_runtime_candidate.json"]
    for config in (launch, candidate):
        if (config["lockfile"]["sha256"] != OLD_LOCK_SHA256
                or config["lockfile"]["sha256"] == SELECTED_LOCK_SHA256
                or not config["interpreter"].startswith("/Users/gregc/TheWiz-LocalRuntime/")):
            raise ValueError("historical runtime is not separated from canonical runtime")
    if (launch["status"] != "VERIFIED_FOR_BOUNDED_COLLECTION_LAUNCH"
            or launch["scope"] != "seven-day read-only research collection only"
            or any(launch["authority"].values())
            or candidate["status"] != "CANDIDATE_NOT_ADOPTED"
            or any(candidate["prohibitions"].values())):
        raise ValueError("historical runtime authority scope changed")

    scope = source_data["config/collection_verification_scope.json"]
    batches = source_data["config/collection_verification_batches.json"]
    selected_nodeids = scope["selected_nodeids"]
    test_files = set(scope["test_files"])
    selected_paths = {nodeid.split("::", 1)[0] for nodeid in selected_nodeids}
    flattened = [nodeid for batch in batches["batches"] for nodeid in batch["nodeids"]]
    if (len(selected_nodeids) != 2411
            or len(set(selected_nodeids)) != 2411
            or len(test_files) != 129
            or selected_paths != test_files
            or flattened != selected_nodeids
            or len(batches["batches"]) != 82):
        raise ValueError("historical test scope or batch relationship changed")
    missing_paths = {path for path in test_files if not selected_exists(path)}
    if len(missing_paths) != 118:
        raise ValueError("historical test scope no longer matches selected commit")

    if sha256(QUEUE.read_bytes()) != QUEUE_SHA256:
        raise ValueError("union source queue drift")
    with QUEUE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 811 or len({row["relative_path"] for row in rows}) != 811:
        raise ValueError("union source queue changed")
    queue = {row["relative_path"]: row for row in rows}
    for relative in SOURCES:
        row = queue[relative]
        if (row["custody_status"] != "REVIEWED_STALE_COLLECTION_RUNTIME_CONFIG_NO_PORT"
                or OUTPUT.name not in row["decision_evidence"]):
            raise ValueError(f"collection config decision missing: {relative}")
    pending = sum(row["custody_status"] in {
        "PRESERVED_REVIEW_REQUIRED_NO_PORT",
        "PORTED_V23_PARTIAL_HISTORICAL_REVIEW_REQUIRED",
    } for row in rows)
    reviewed = sum(row["custody_status"].startswith("REVIEWED_") for row in rows)
    if (pending, reviewed, len(rows) - pending - reviewed) != (639, 154, 18):
        raise ValueError("source queue accounting changed")

    report = {
        "schema_version": "thewiz.gate0.collection_runtime_config_source_reconciliation.v1",
        "decision": "FOUR_STALE_COLLECTION_CONFIGS_REVIEWED_NO_PORT",
        "selected_commit": SELECTED_COMMIT,
        "historical_config_sha256": SOURCES,
        "historical_lock_sha256": OLD_LOCK_SHA256,
        "selected_canonical_lock_sha256": SELECTED_LOCK_SHA256,
        "historical_verification_scope": {
            "selected_nodeids": len(selected_nodeids),
            "batches": len(batches["batches"]),
            "test_file_paths": len(test_files),
            "paths_absent_in_selected_commit": len(missing_paths),
        },
        "union_source_queue": {
            "sha256": QUEUE_SHA256,
            "distinct_paths": len(rows),
            "reviewed_paths": reviewed,
            "nonsource_paths": len(rows) - pending - reviewed,
            "remaining_semantic_review_paths": pending,
        },
        "collection_activation_authorized": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS collection_runtime_config_source_reconciliation")


if __name__ == "__main__":
    main()
