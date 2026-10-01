#!/usr/bin/env python3
"""Run the graph's stage-only diagnostic with scoped report publication.

The authority profile permits file publication under reports/active only. No
network, credential, provider-credit, account, or order effect is permitted.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from quant_platform.orchestration.effect_authority import (
    PHASE00_REPAIR_PROFILE,
    EffectAuthority,
    publication_authority_session,
)
from quant_platform.orchestration.langgraph_workflow import run_langgraph_agent_workflow


def _hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a stage-only LangGraph dry-run")
    parser.add_argument("--stage", default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run_id = f"gate0_graph_{uuid4().hex}"
    report_dir = root / "reports" / "active"
    authority = EffectAuthority(
        root=root,
        secret=secrets.token_bytes(32),
        issuer_id="gate0_graph_dry_run",
        profile=PHASE00_REPAIR_PROFILE,
    )
    with publication_authority_session(
        authority=authority,
        run_id=run_id,
        intended_slot_id=run_id,
        policy_version="gate0.graph_dry_run.v1",
        source_fingerprint_sha256=_hash_file(
            root / "src/quant_platform/orchestration/langgraph_workflow.py"
        ),
        runtime_fingerprint_sha256=_hash_file(Path(sys.executable).resolve()),
        configuration_fingerprint_sha256=_hash_file(root / "uv.lock"),
        allowed_scopes=frozenset({"research"}),
        allowed_target_prefixes=(report_dir,),
        max_total_bytes=8 * 1024**2,
    ):
        result = run_langgraph_agent_workflow(
            stage=args.stage, dry_run=True, root=root
        )
    print(
        json.dumps(
            {"summary": result.summary, "paths": {key: str(path) for key, path in result.paths.items()}},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
