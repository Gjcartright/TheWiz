"""Executable entrypoint for dynamic controls without importing the ML stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_platform.orchestration.dynamic_stage_runner import dynamic_stages_for_group, run_dynamic_stage
from quant_platform.orchestration.state import OrchestratorState


ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run shadow-only dynamic-agent controls.")
    parser.add_argument(
        "--stage",
        default="dynamic_rollout",
        choices=("copula_shadow", "dynamic_rollout", "dynamic_supreme_team", "math_v2", "teacher_council"),
    )
    parser.add_argument("--pair-id", default="")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    state = OrchestratorState(stage_group=args.stage, pair_id=args.pair_id, dry_run=args.dry_run, report_only=True, root=args.root)
    results = [run_dynamic_stage(stage, state, args.root) for stage in dynamic_stages_for_group(args.stage)]
    print(json.dumps([result.to_row(state.run_id, state.pair_id) for result in results], indent=2, default=str))
    return 0 if all(result.status.value not in {"failed"} for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
