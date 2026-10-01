"""Read-only integrity verifier for the multi-asset seven-day evidence lane."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from quant_platform.orchestration.multi_asset_capture_job import (
    _read_receipt,
    _validate_full_predecessor_chain,
)


def verify_multi_asset_store(*, root: Path, now: datetime | None = None) -> dict[str, object]:
    base = root / "data" / "research" / "multi_asset_captures"
    if base.is_symlink():
        raise ValueError("multi_asset_capture_store_must_not_be_symlink")
    capture_paths = (
        tuple(sorted(path for path in base.iterdir() if path.name.startswith("universe-")))
        if base.exists()
        else ()
    )
    if any(path.is_symlink() or not path.is_dir() for path in capture_paths):
        raise ValueError("multi_asset_capture_entry_invalid")
    capture_ids = tuple(path.name for path in capture_paths)
    if not capture_ids:
        return {
            "status": "EMPTY",
            "capture_count": 0,
            "authority": {"execution": False, "wizard_credits": 0},
        }
    terminal = capture_ids[-1]
    chain_ids = _validate_full_predecessor_chain(root, terminal)
    if set(chain_ids) != set(capture_ids):
        raise ValueError("multi_asset_capture_chain_incomplete")
    receipt = _read_receipt(root, terminal)
    observed = datetime.fromisoformat(str(receipt["observed_at_utc"]))
    checked_at = now or datetime.now(UTC)
    if (
        observed.tzinfo is None
        or observed.utcoffset() is None
        or checked_at.tzinfo is None
        or checked_at.utcoffset() is None
    ):
        raise ValueError("multi_asset_terminal_time_invalid")
    age_minutes = (checked_at.astimezone(UTC) - observed.astimezone(UTC)).total_seconds() / 60
    if age_minutes < 0:
        raise ValueError("multi_asset_terminal_time_invalid")
    assets = receipt.get("assets")
    if (
        not isinstance(assets, list)
        or not 2 <= len(assets) <= 30
        or not all(isinstance(asset, str) and asset for asset in assets)
        or len(set(assets)) != len(assets)
        or receipt.get("interval") != "1h"
        or receipt.get("candidate_pair_count") != len(assets) * (len(assets) - 1) // 2
    ):
        raise ValueError("multi_asset_terminal_universe_invalid")
    authority = receipt.get("authority")
    if (
        not isinstance(authority, dict)
        or authority.get("credentials_used") is not False
        or authority.get("orders") != 0
        or authority.get("wizard_credits") != 0
        or authority.get("paper_trading") is not False
        or authority.get("execution") is not False
    ):
        raise ValueError("multi_asset_terminal_authority_invalid")
    return {
        "status": "PASS",
        "capture_count": len(capture_ids),
        "terminal_capture_id": terminal,
        "asset_count": len(assets),
        "candidate_pair_count": receipt.get("candidate_pair_count"),
        "age_minutes": age_minutes,
        "stale": age_minutes > 75 * 60,
        "authority": authority,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify retained multi-asset collection evidence")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_multi_asset_store(root=args.root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
