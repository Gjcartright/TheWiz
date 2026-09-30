#!/usr/bin/env python3
"""Bind the narrow Engle-Granger grid repair to its controlled probe."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
DIAGNOSTICS = Path("/Users/gregc/Backups/TheWiz/recovery-route-diagnostics/2026-09-30")
BASE = "6ce7bc7f6e47a27f847b98ee76093efdd31bb67e"
SELECTED_COMMIT = "4362c54b55afabe38f522289c2f9e84788e9ab67"
SOURCE = "src/quant_platform/statistics/math_v2.py"
TEST = "tests/test_statistics_math_v2.py"
OUTPUT = AUDIT / "GATE0_MATH_GRID_REPAIR_2026-09-30.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prior_sha(path: str) -> str:
    content = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)
    return hashlib.sha256(content).hexdigest()


def selected_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{SELECTED_COMMIT}:{path}"], cwd=ROOT)


def main() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", SELECTED_COMMIT, "HEAD"], cwd=ROOT, check=True)
    queue_bytes = selected_bytes("audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv")
    queue = {row["relative_path"]: row for row in csv.DictReader(queue_bytes.decode("utf-8").splitlines())}
    if any(queue[path]["custody_status"] != "PRESERVED_REVIEW_REQUIRED_NO_PORT" for path in (SOURCE, TEST)):
        raise ValueError("broader Math V2 review must remain pending")
    for path in (SOURCE, TEST):
        if prior_sha(path) != queue[path]["working_sha256_at_freeze"]:
            raise ValueError(f"frozen Math V2 baseline changed: {path}")

    paths = {
        "script": DIAGNOSTICS / "math-grid-probe.py",
        "active_baseline": DIAGNOSTICS / "math-grid-active.json",
        "runtime_candidate": DIAGNOSTICS / "math-grid-runtime.json",
        "patched": DIAGNOSTICS / "math-grid-patched.json",
        "focused_junit": DIAGNOSTICS / "math-grid-fix-junit.xml",
    }
    probes = {name: json.loads(paths[name].read_text(encoding="utf-8"))
              for name in ("active_baseline", "runtime_candidate", "patched")}
    if any(probes[label]["aligned_regular"]["status"] != "valid" for label in probes):
        raise ValueError("regular control must stay valid")
    for name, reason in (
        ("shifted_index", "price_index_identity_mismatch"),
        ("irregular_index", "irregular_observation_grid"),
        ("incomplete_price", "non_finite_prices"),
    ):
        if probes["active_baseline"][name]["status"] != "valid":
            raise ValueError(f"baseline did not accept {name}")
        for label in ("runtime_candidate", "patched"):
            if (probes[label][name]["status"], probes[label][name]["reason"]) != (
                "invalid", reason
            ):
                raise ValueError(f"{label} did not reject {name}")
    if probes["runtime_candidate"] != probes["patched"]:
        raise ValueError("patched math-grid behavior differs from preserved runtime")
    suite = next(ET.parse(paths["focused_junit"]).getroot().iter("testsuite"))
    if tuple(int(suite.attrib[key]) for key in ("tests", "failures", "errors")) != (35, 0, 0):
        raise ValueError("math-grid, backtest and acceptance focused suite is not green")
    report = {
        "schema_version": "thewiz.gate0.math_grid_repair.v1",
        "status": "PARTIAL_SOURCE_REVIEW_REPAIR_COMMITTED_PENDING_REMAINDER",
        "baseline_commit": BASE,
        "source_sha256": {
            path: {
                "baseline": prior_sha(path), "patched": hashlib.sha256(selected_bytes(path)).hexdigest(),
                "runtime_candidate": queue[path]["runtime_sha256_at_freeze"],
                "historical_variants": queue[path]["historical_variant_sha256"].split(";"),
            }
            for path in (SOURCE, TEST)
        },
        "diagnostic_files": {name: {"path": str(path), "sha256": sha(path)}
                             for name, path in paths.items()},
        "focused_tests_passed": 35,
        "decision": "Require exact pair-index identity, a regular ordered observation grid, and complete finite positive paired prices before Engle-Granger estimation. Preserve other Math V2 variants for dependency-complete review.",
        "remaining_source_decisions": [
            "Engle-Granger identified-design and finite-inference contract",
            "paired sample and relationship identities across EG and ECM",
            "OU observation-grid and missing-sample treatment",
            "historical I(1) diagnostic and Johansen estimator",
            "shared performance math and full strategy integration",
        ],
        "authority": "No cointegration strategy acceptance, Testnet order, or live order authority is granted.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS three_engle_granger_grid_rejections_and_partial_math_decision")


if __name__ == "__main__":
    main()
