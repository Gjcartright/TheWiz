"""Stdlib-only interpreter guard for governed scheduler entry points."""

from __future__ import annotations

import json
import runpy
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

BOOTSTRAP_SCHEMA_VERSION = "thewiz.scheduler_bootstrap.v1"
BOOTSTRAP_EXIT_RUNTIME_MISMATCH = 78


def canonical_runtime_blockers(
    root: Path,
    *,
    observed_executable: Path | None = None,
) -> list[str]:
    """Return blockers without importing any project or numerical module."""

    canonical_root = root.resolve()
    expected = canonical_root / ".venv" / "bin" / "python3"
    observed = Path(observed_executable or sys.executable)
    blockers: list[str] = []
    if not expected.is_file():
        blockers.append("canonical_scheduler_interpreter_missing")
    else:
        try:
            if observed.resolve() != expected.resolve():
                blockers.append("wrong_scheduler_interpreter")
        except OSError:
            blockers.append("scheduler_interpreter_resolution_failed")
    if not (canonical_root / ".venv" / "pyvenv.cfg").is_file():
        blockers.append("canonical_scheduler_environment_metadata_missing")
    if not (canonical_root / "src" / "quant_platform").is_dir():
        blockers.append("scheduler_workspace_source_missing")
    return blockers


def run_scheduler_bootstrap(
    argv: Sequence[str],
    *,
    observed_executable: Path | None = None,
    module_runner: Callable[..., dict[str, Any]] = runpy.run_module,
) -> int:
    """Validate runtime identity, then execute one registered scheduler module."""

    if len(argv) != 2:
        return _blocked(
            root="",
            contract_key="",
            blockers=["scheduler_bootstrap_arguments_invalid"],
        )
    root = Path(argv[0]).resolve()
    contract_key = argv[1].strip()
    blockers = canonical_runtime_blockers(
        root,
        observed_executable=observed_executable,
    )
    if blockers:
        return _blocked(
            root=str(root),
            contract_key=contract_key,
            blockers=blockers,
        )

    from quant_platform.orchestration.corrective_runtime import scheduler_contract

    try:
        contract = scheduler_contract(contract_key)
    except KeyError:
        return _blocked(
            root=str(root),
            contract_key=contract_key,
            blockers=["scheduler_bootstrap_contract_unknown"],
        )
    sys.argv = [contract.module, contract.action]
    module_runner(contract.module, run_name="__main__", alter_sys=False)
    return 0


def _blocked(*, root: str, contract_key: str, blockers: list[str]) -> int:
    payload = {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "status": "BLOCKED_BEFORE_SCHEDULER_IMPORT",
        "root": root,
        "contract_key": contract_key,
        "blockers": blockers,
        "repair_command": "uv sync --frozen",
        "order_submission_performed": False,
        "live_trading_authorized": False,
    }
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)
    return BOOTSTRAP_EXIT_RUNTIME_MISMATCH


def main() -> None:
    raise SystemExit(run_scheduler_bootstrap(sys.argv[1:]))


if __name__ == "__main__":
    main()
