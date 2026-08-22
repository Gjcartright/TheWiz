from __future__ import annotations

import json
from pathlib import Path

from quant_platform.orchestration.corrective_runtime import atomic_append_text
from quant_platform.orchestration.state import StageResult


def append_stage_event(path: Path, *, run_id: str, pair_id: str, result: StageResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_append_text(
        path,
        json.dumps(result.to_row(run_id, pair_id), sort_keys=True) + "\n",
    )
    return path
