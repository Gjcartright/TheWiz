from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from quant_platform.orchestration.current_wizard_hyperliquid_cadence import (
    build_current_wizard_hyperliquid_operating_cadence,
)
from quant_platform.orchestration.current_wizard_ou_optimal_overlay import (
    _experiment_orientation,
    build_current_wizard_ou_optimal_overlay,
)


PAIR = "binance|daily|ETH|WIF"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_lineage(
    *,
    root: Path,
    active_name: str,
    snapshot_dir: str,
    snapshot_name: str,
    frame: pd.DataFrame,
) -> str:
    active = root / "reports" / "active" / active_name
    snapshot = root / "reports" / "snapshots" / snapshot_dir / snapshot_name
    active.parent.mkdir(parents=True, exist_ok=True)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(active, index=False)
    snapshot.write_bytes(active.read_bytes())
    return str(snapshot.relative_to(root))


def _write_inputs(root: Path) -> None:
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)

    candidates_path = active / "wizard_sweep_candidates.csv"
    pd.DataFrame(
        [
            {
                "pair": "ETH-USD-WIF-USD",
                "exchange": "binance",
                "timeframe": "daily",
                "strategy": "Static (Spread)",
                "symbol_1": "ETH-USD",
                "symbol_2": "WIF-USD",
                "ou_optimal": True,
            },
            {
                "pair": "ETH-USD-WIF-USD",
                "exchange": "binance",
                "timeframe": "daily",
                "strategy": "Copula",
                "symbol_1": "WIF-USD",
                "symbol_2": "ETH-USD",
                "ou_optimal": False,
            },
        ]
    ).to_csv(candidates_path, index=False)
    candidates = pd.read_csv(candidates_path, keep_default_na=False)
    source_rows = []
    for index, candidate in candidates.iterrows():
        values = {str(key): value.item() if hasattr(value, "item") else value for key, value in candidate.items()}
        fingerprint = sha256(_canonical_json(values).encode("utf-8")).hexdigest()
        source_rows.append(
            {
                "api_source_row_id": f"row_{index}",
                "api_source_row_index": index,
                "source_row_fingerprint": fingerprint,
                "pair_group_key": PAIR,
                "pair": "ETH-USD-WIF-USD",
                "wizard_exchange": "binance",
                "timeframe": "daily",
                "asset_x": "ETH" if index == 0 else "WIF",
                "asset_y": "WIF" if index == 0 else "ETH",
                "orientation": "ETH/WIF" if index == 0 else "WIF/ETH",
                "api_exact_mode": candidate["strategy"],
                "api_ou_optimal": candidate["ou_optimal"],
                "sweep_captured_at": "2026-08-08T12:00:00+00:00",
                "sweep_source_timestamp": "2026-08-08T11:59:00+00:00",
                "evidence_path": "reports/snapshots/refresh/api_candidates.csv",
            }
        )
    source = pd.DataFrame(source_rows)
    source_snapshot = _write_lineage(
        root=root,
        active_name="exhaustive_wizard_api_refresh_source_accounting.csv",
        snapshot_dir="refresh",
        snapshot_name="source_accounting.csv",
        frame=source,
    )
    candidate_snapshot = _write_lineage(
        root=root,
        active_name="wizard_sweep_candidates.csv",
        snapshot_dir="refresh",
        snapshot_name="api_candidates.csv",
        frame=candidates,
    )
    (active / "exhaustive_wizard_api_refresh_manifest.json").write_text(
        json.dumps(
            {
                "refresh_id": "refresh_test",
                "artifacts": {
                    "snapshot_source_accounting": source_snapshot,
                    "snapshot_api_candidates": candidate_snapshot,
                },
            }
        ),
        encoding="utf-8",
    )

    pair_status = pd.DataFrame(
        [
            {
                "pair_group_key": PAIR,
                "asset_a": "ETH",
                "asset_b": "WIF",
                "hyperliquid_pair_ready": True,
            }
        ]
    )
    experiments = pd.DataFrame(
        [
            {
                "experiment_id": "static_original",
                "pair_group_key": PAIR,
                "exact_mode": "Static (Spread)",
                "orientation": "original",
                "experiment_status": "READY_FOR_POINT_IN_TIME_HISTORY",
                "experiment_blocker": "",
            },
            {
                "experiment_id": "copula_reverse",
                "pair_group_key": PAIR,
                "exact_mode": "Copula",
                "orientation": "reverse",
                "experiment_status": "READY_FOR_POINT_IN_TIME_HISTORY",
                "experiment_blocker": "",
            },
        ]
    )
    pair_snapshot = _write_lineage(
        root=root,
        active_name="current_wizard_pair_detail_status.csv",
        snapshot_dir="handoff",
        snapshot_name="pair_status.csv",
        frame=pair_status,
    )
    experiment_snapshot = _write_lineage(
        root=root,
        active_name="current_wizard_hyperliquid_experiment_matrix.csv",
        snapshot_dir="handoff",
        snapshot_name="experiments.csv",
        frame=experiments,
    )
    handoff_snapshot_manifest = root / "reports" / "snapshots" / "handoff" / "manifest.json"
    handoff_snapshot_manifest.write_text("{}", encoding="utf-8")
    (active / "current_wizard_hyperliquid_handoff_manifest.json").write_text(
        json.dumps(
            {
                "handoff_id": "handoff_test",
                "artifacts": {
                    "snapshot_pair_status": pair_snapshot,
                    "snapshot_experiments": experiment_snapshot,
                    "snapshot_manifest": str(handoff_snapshot_manifest.relative_to(root)),
                },
            }
        ),
        encoding="utf-8",
    )

    downstream = {
        "canonical": (
            "current_wizard_hyperliquid_canonical_replay.csv",
            "current_wizard_hyperliquid_canonical_replay_manifest.json",
            "results",
            pd.DataFrame(
                [
                    {
                        "experiment_id": "static_original",
                        "replay_status": "RESEARCH_REPLAY_COMPLETE",
                        "research_rank_eligible": True,
                    },
                    {
                        "experiment_id": "copula_reverse",
                        "replay_status": "RESEARCH_REPLAY_COMPLETE",
                        "research_rank_eligible": True,
                    },
                ]
            ),
        ),
        "observed": (
            "current_wizard_hyperliquid_observed_cost_replay.csv",
            "current_wizard_hyperliquid_observed_cost_replay_manifest.json",
            "results",
            pd.DataFrame(
                [
                    {"experiment_id": "static_original", "replay_status": "COMPLETE"},
                    {"experiment_id": "copula_reverse", "replay_status": "COMPLETE"},
                ]
            ),
        ),
        "walkforward": (
            "current_wizard_hyperliquid_walkforward_status.csv",
            "current_wizard_hyperliquid_walkforward_manifest.json",
            "status",
            pd.DataFrame(
                [
                    {
                        "experiment_id": "static_original",
                        "walkforward_status": "PASS_RESEARCH_WALK_FORWARD",
                        "statistical_selection_status": "PASS",
                    },
                    {
                        "experiment_id": "copula_reverse",
                        "walkforward_status": "FAIL_RESEARCH_WALK_FORWARD",
                        "statistical_selection_status": "FAIL",
                    },
                ]
            ),
        ),
    }
    for snapshot_dir, (active_name, manifest_name, artifact_key, frame) in downstream.items():
        snapshot_path = _write_lineage(
            root=root,
            active_name=active_name,
            snapshot_dir=snapshot_dir,
            snapshot_name=f"{artifact_key}.csv",
            frame=frame,
        )
        (active / manifest_name).write_text(
            json.dumps({"artifacts": {f"snapshot_{artifact_key}": snapshot_path}}),
            encoding="utf-8",
        )

    pd.DataFrame(
        [
            {
                "exact_mode": "OU (Optimal)",
                "capture_status": "NOT_AVAILABLE_ON_PAIR_PAGE",
            }
        ]
    ).to_csv(active / "exhaustive_wizard_pair_detail_mode_ledger.csv", index=False)


def test_overlay_accounts_for_scanner_flag_without_inventing_a_mode(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)

    result = build_current_wizard_ou_optimal_overlay(
        root=tmp_path,
        now=datetime(2026, 8, 8, 13, 0, tzinfo=timezone.utc),
    )

    ledger = pd.read_csv(result.paths["ledger"], keep_default_na=False)
    validation = pd.read_csv(result.paths["validation"], keep_default_na=False)

    assert len(ledger) == 2
    assert ledger["ou_optimal"].astype(bool).sum() == 1
    assert set(ledger["base_experiment_id"]) == {"static_original", "copula_reverse"}
    assert not ledger["independent_pair_page_mode"].astype(bool).any()
    assert not ledger["acceptance_authority"].astype(bool).any()
    assert not ledger["live_trading_authorized"].astype(bool).any()
    assert validation["status"].eq("PASS").all()
    assert result.summary["source_rows_accounted"] == 2
    assert result.summary["ou_optimal_true_rows"] == 1
    assert result.summary["ou_optimal_false_rows"] == 1
    assert result.summary["pair_page_ou_optimal_captured_rows"] == 0


def test_self_pair_orientation_is_accounted_but_not_made_tradable() -> None:
    orientation = _experiment_orientation(
        {"asset_x": "DOGE", "asset_y": "DOGE"},
        {"asset_a": "DOGE", "asset_b": ""},
    )

    assert orientation == "original"


def test_operating_cadence_requires_overlay_stratification(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    overlay = build_current_wizard_ou_optimal_overlay(
        root=tmp_path,
        now=datetime(2026, 8, 8, 13, 0, tzinfo=timezone.utc),
    )
    active = tmp_path / "reports" / "active"
    refresh_path = active / "exhaustive_wizard_api_refresh_manifest.json"
    refresh = json.loads(refresh_path.read_text(encoding="utf-8"))
    refresh.update(
        {
            "discovery_policy": "exhaustive_no_prefilter",
            "api_source_rows": 2,
            "api_source_rows_accounted": 2,
        }
    )
    refresh_path.write_text(json.dumps(refresh), encoding="utf-8")

    chain_snapshot = tmp_path / "reports" / "snapshots" / "chain" / "manifest.json"
    chain_snapshot.parent.mkdir(parents=True, exist_ok=True)
    chain_snapshot.write_text("{}", encoding="utf-8")
    (active / "current_wizard_hyperliquid_chain_validation_manifest.json").write_text(
        json.dumps(
            {
                "validation_id": "validation_test",
                "chain_status": "PASS",
                "experiment_authority_count": 4,
                "one_x_research_survivors": 0,
                "trade_eligibility_status": "BLOCKED_NO_ONE_X_RESEARCH_SURVIVOR",
                "artifacts": {
                    "snapshot_manifest": str(chain_snapshot.relative_to(tmp_path))
                },
            }
        ),
        encoding="utf-8",
    )
    for filename, stage_id, value in (
        (
            "current_wizard_hyperliquid_concentration_manifest.json",
            "concentration_id",
            "concentration_test",
        ),
        (
            "current_wizard_hyperliquid_failure_attribution_manifest.json",
            "failure_attribution_id",
            "failure_test",
        ),
        (
            "current_wizard_hyperliquid_leverage_manifest.json",
            "leverage_surface_id",
            "leverage_test",
        ),
        (
            "current_wizard_hyperliquid_learning_manifest.json",
            "learning_ledger_id",
            "learning_test",
        ),
    ):
        (active / filename).write_text(
            json.dumps(
                {
                    stage_id: value,
                    "referenced_upstream_bytes": 1,
                    "locally_copied_input_bytes": 0,
                }
            ),
            encoding="utf-8",
        )

    cadence = build_current_wizard_hyperliquid_operating_cadence(
        root=tmp_path,
        now=datetime(2026, 8, 8, 13, 30, tzinfo=timezone.utc),
        available_disk_bytes=4 * 1024**3,
        wizard_api_key_present=True,
    )
    rows = pd.read_csv(cadence.paths["cadence"], keep_default_na=False)
    validation = pd.read_csv(cadence.paths["validation"], keep_default_na=False)
    stage = rows.loc[rows["stage"].eq("ou_optimal_outcome_stratification")].iloc[0]

    assert len(rows) == 19
    assert rows["sequence"].tolist() == list(range(1, 20))
    assert stage["sequence"] == 11
    assert "build-current-wizard-ou-optimal-overlay" in stage["command"]
    assert not bool(stage["testnet_order_authority"])
    assert validation["status"].eq("PASS").all()
    assert cadence.summary["ou_optimal_overlay_id"] == overlay.summary["overlay_id"]
    assert cadence.summary["ou_optimal_source_rows_accounted"] == 2
