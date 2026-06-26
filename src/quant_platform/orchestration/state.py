from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4


class StageStatus(StrEnum):
    READY = "ready"
    DRY_RUN = "dry_run"
    PASSED = "passed"
    BLOCKED = "blocked"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: StageStatus
    blocker: str = ""
    reason: str = ""
    evidence_path: str = ""
    next_step: str = ""
    rows: int = 0

    def to_row(self, run_id: str, pair_id: str = "") -> dict[str, object]:
        return {
            "run_id": run_id,
            "pair_id": pair_id,
            "stage": self.stage,
            "status": self.status.value,
            "blocker": self.blocker,
            "reason": self.reason,
            "evidence_path": self.evidence_path,
            "next_step": self.next_step,
            "rows": self.rows,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


@dataclass
class OrchestratorState:
    run_id: str = field(default_factory=lambda: uuid4().hex[:12])
    stage_group: str = "all"
    pair_id: str = ""
    dry_run: bool = False
    force_refresh: bool = False
    fail_fast: bool = False
    report_only: bool = False
    root: Path | None = None
    results: list[StageResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, result: StageResult) -> None:
        self.results.append(result)

    @property
    def blocked(self) -> bool:
        return any(result.status in {StageStatus.BLOCKED, StageStatus.FAILED} for result in self.results)
