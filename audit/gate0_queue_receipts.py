"""Read the original Gate 0 source decision snapshot after later queue reviews."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path


DECISION_COMMIT = "12695b48cfe6301240eb95d04c757792670ba03e"
QUEUE_PATH = "audit/GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv"


def original_decision_queue(root: Path) -> dict[str, dict[str, str]]:
    """Return immutable earlier decisions; new variants have their own verifier."""

    subprocess.run(
        ["git", "merge-base", "--is-ancestor", DECISION_COMMIT, "HEAD"],
        cwd=root,
        check=True,
    )
    data = subprocess.run(
        ["git", "show", f"{DECISION_COMMIT}:{QUEUE_PATH}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = list(csv.DictReader(data.splitlines()))
    queue = {row["relative_path"]: row for row in rows}
    if len(rows) != 811 or len(queue) != 811:
        raise ValueError("Gate 0 original queue identity changed")
    return queue
