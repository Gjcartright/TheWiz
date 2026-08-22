from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import numpy as np
import pandas as pd

from quant_platform.apify_sources import infer_apify_venue
from quant_platform.execution import DydxNetworkConfig, build_dydx_indexer_adapter
from quant_platform.experiments import PairDataset
from quant_platform.ml_filter import (
    GLOBAL_PURGED_SPLIT_SCHEME,
    MINIMUM_TRAINING_TAKE_RATE,
    MODEL_SELECTION_BOUNDARY_SCHEME,
    MODEL_SELECTION_ISOLATION_SCHEME,
    MODEL_SELECTION_PHASE,
    NON_FEATURE_COLUMNS,
    RETURN_COLUMN,
    TARGET_COLUMN,
    THRESHOLD_CALIBRATION_SCHEME,
    TIMESTAMP_COLUMN,
    UNTOUCHED_EVALUATION_PHASE,
    build_trade_filter_dataset,
    model_selection_leaderboard,
    train_trade_filter_walkforward,
)
from quant_platform.orchestration.corrective_runtime import (
    atomic_copy_file,
    atomic_write_csv,
    atomic_write_parquet,
    atomic_write_text,
    promote_staged_directory,
    promote_staged_file,
    write_immutable_bytes,
)
from quant_platform.pair_detail_ingestion import (
    add_derived_beta_from_prices,
    datasets_from_pair_detail_snapshots,
    extract_history_rows,
    load_pair_detail_payload,
    snapshot_from_payload,
)
from quant_platform.pair_market_utils import pair_markets_from_pair
from quant_platform.regimes import RegimeConfig, classify_regimes
from quant_platform.runtime_types import ROOT, CommandResult
from quant_platform.wizard_symbols import (
    normalize_wizard_exchange,
    normalize_wizard_symbol,
    wizard_exchange_lane,
)

ACTIVE = ROOT / "reports" / "active"
DASHBOARD = ROOT / "reports" / "dashboard"
ML_REPORTS = ROOT / "reports" / "ml"
DATA_ML = ROOT / "data" / "ml"
MODELS = ROOT / "models" / "trade_gate"
TRADE_DATASET_BUILD_SCHEMA = "thewiz.trade_dataset_build.v3"
MINIMUM_MODEL_GATED_TAKE_RATE = 0.10
TRADE_DATASET_REGISTRIES = (
    "current_wizard_hyperliquid_pair_history_results.csv",
    "exhaustive_wizard_hyperliquid_pair_history_results.csv",
)
POST_OUTCOME_MEMORY_COLUMNS = {
    "wizard_history_feature_count",
    "wizard_same_regime_count",
    "wizard_same_strategy_family_count",
    "wizard_same_venue_count",
    "wizard_same_regime_strategy_venue_count",
    "wizard_verified_same_regime_strategy_venue_count",
    "wizard_same_regime_strategy_venue_win_rate",
    "wizard_same_regime_strategy_venue_mean_return",
    "wizard_same_regime_strategy_venue_mean_drawdown",
    "wizard_learning_feature_state",
    "native_promotion_basis",
    "shared_outcome_count",
    "shared_verified_outcome_count",
    "shared_outcome_win_rate",
    "shared_outcome_mean_return",
    "shared_outcome_mean_drawdown",
}
MULTI_VENUE_HISTORY_READINESS_FILENAME = "multi_venue_history_readiness.csv"
LEGACY_MULTI_VENUE_HISTORY_READINESS_FILENAME = "multi_venue_history_readiness_2026-06-25.csv"
DASHBOARD_REFRESH_PROFILES = {"monitor", "deep"}
DASHBOARD_DEEP_MIN_FREE_BYTES = 3 * 1024 * 1024 * 1024
DASHBOARD_REFRESH_COLUMNS = [
    "refresh_profile",
    "stage",
    "status",
    "reason",
    "evidence_path",
    "timestamp_utc",
]


def current_multi_venue_history_readiness_path(root: Path = ROOT) -> Path:
    """Return current readiness when available, falling back only for legacy evidence readers."""
    active = root / "reports" / "active"
    current = active / MULTI_VENUE_HISTORY_READINESS_FILENAME
    legacy = active / LEGACY_MULTI_VENUE_HISTORY_READINESS_FILENAME
    return current if current.exists() or not legacy.exists() else legacy


def dashboard_storage_preflight(root: Path = ROOT) -> dict[str, object]:
    """Report whether a deep refresh has enough working storage to complete safely."""

    usage = shutil.disk_usage(root)
    free_bytes = int(usage.free)
    required_bytes = DASHBOARD_DEEP_MIN_FREE_BYTES
    return {
        "ready": free_bytes >= required_bytes,
        "free_bytes": free_bytes,
        "required_bytes": required_bytes,
        "free_gib": round(free_bytes / 1024**3, 2),
        "required_gib": round(required_bytes / 1024**3, 2),
        "blocker": "" if free_bytes >= required_bytes else "insufficient_free_space_for_deep_dashboard_refresh",
    }

ARTIFACT_COLUMNS = [
    "path",
    "artifact_type",
    "status",
    "source_system",
    "created_or_modified_at",
    "used_by_active_pipeline",
    "evidence_value",
    "safe_to_archive_later",
    "reason",
    "notes",
]
ARTIFACT_EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".runtime_agents",
    ".runtime_tmp",
    ".venv",
    ".venv311",
    ".venv312",
    "__pycache__",
    "node_modules",
    "venv",
}
ARTIFACT_EXCLUDED_DIR_PREFIXES = (".venv", "pytest-of-")
ARTIFACT_EXCLUDED_FILES = {".DS_Store"}

PAIR_UNIVERSE_COLUMNS = [
    "pair",
    "asset_x",
    "asset_y",
    "exchange",
    "dydx_tradable",
    "best_execution_venue",
    "execution_venue_ready",
    "available_venues",
    "venue_decision_reason",
    "available_timeframes",
    "wizards_pair_id",
    "cointegration_score",
    "copula_score",
    "zscore_score",
    "half_life",
    "hurst",
    "correlation",
    "funding_drag_bps",
    "volume_usd",
    "open_interest_usd",
    "local_backtest_score",
    "discovery_score",
    "acceptance_score",
    "combined_score",
    "decision_bucket",
    "decision_reason",
    "missing_data_reason",
    "source_timestamp",
    "field_freshness",
    "stale_reason",
    "evidence_path",
    "best_wizard_exact_mode",
    "best_wizard_spread_id",
    "best_wizard_strategy_id",
    "best_wizard_local_strategy_id",
    "best_wizard_local_strategy_name",
    "best_wizard_local_strategy_family",
    "best_wizard_strategy_mapping_status",
    "best_wizard_sharpe",
    "best_wizard_returns_total",
    "best_wizard_exchange",
    "best_wizard_source_authority",
    "best_wizard_source_timestamp",
    "best_wizard_source_fresh",
    "best_wizard_source_health",
    "wizard_evidence_state",
    "wizard_stationarity_status",
    "wizard_engle_granger_cointegrated",
    "wizard_engle_granger_trend",
    "wizard_johansen_cointegrated",
    "wizard_zscore_last",
    "wizard_zscore_roll_last",
    "wizard_leg_volume_min",
    "wizard_diagnostic_score",
    "wizard_hypothesis_status",
    "local_mode_confirmation_status",
    "wizard_local_parity_status",
    "promotion_blocker",
]

CANONICAL_COMMANDS = [
    "PYTHONPATH=src python -m quant_platform.cli system-check",
    "PYTHONPATH=src python -m quant_platform.cli build-artifact-index",
    "PYTHONPATH=src python -m quant_platform.cli current-state",
    "PYTHONPATH=src python -m quant_platform.cli complete-corrective-plan",
    "PYTHONPATH=src python -m quant_platform.cli build-scheduler-runtime-readiness",
    "PYTHONPATH=src python -m quant_platform.cli build-wizard-reset-readiness",
    "PYTHONPATH=src python -m quant_platform.cli build-stage4-handoff-readiness",
    "PYTHONPATH=src python -m quant_platform.cli build-corrective-agent-governance",
    "PYTHONPATH=src python -m quant_platform.cli run-corrective-l2-capture",
    (
        "PYTHONPATH=src python -m quant_platform.cli "
        "build-current-wizard-hyperliquid-evidence-command-center"
    ),
    "python scripts/build_corrective_checkpoint.py",
    "python scripts/build_current_recovery_checkpoint.py --destination /Volumes/TheWizRecovery",
    "PYTHONPATH=src python -m quant_platform.cli build-wizard-research-pack",
    "PYTHONPATH=src python -m quant_platform.cli build-wizard-mode-matrix-capture-queue",
    "PYTHONPATH=src python -m quant_platform.cli build-wizard-pair-settings-capture-template",
    "PYTHONPATH=src python -m quant_platform.cli build-wizard-mode-replay-capability",
    "PYTHONPATH=src python -m quant_platform.cli build-wizard-mode-comparison",
    "PYTHONPATH=src python -m quant_platform.cli build-wizard-exploratory-cost-sensitivity",
    "PYTHONPATH=src python -m quant_platform.cli refresh-hyperliquid-market-context",
    "PYTHONPATH=src python -m quant_platform.cli build-hyperliquid-research-bundle --max-pairs 5",
    "PYTHONPATH=src python -m quant_platform.cli build-hyperliquid-wizard-hypothesis-queue",
    "PYTHONPATH=src python -m quant_platform.cli run-hyperliquid-research-cycle",
    "PYTHONPATH=src python -m quant_platform.cli run-hyperliquid-research-cycle --collect-l2",
    "PYTHONPATH=src python -m quant_platform.cli run-hyperliquid-auxiliary-timeframe-validation --interval 4h --intraday-days 800",
    "PYTHONPATH=src python -m quant_platform.cli hyperliquid-testnet-margin-snapshot",
    "PYTHONPATH=src python -m quant_platform.cli hyperliquid-testnet-smoke-approval-template",
    "PYTHONPATH=src python -m quant_platform.cli hyperliquid-testnet-collateral-transfer-preflight --transfer-amount-usd 25",
    "PYTHONPATH=src python -m quant_platform.cli run-hyperliquid-testnet-collateral-transfer --transfer-preflight-id <immutable_preflight_id> --transfer-approval-id <signed_one_run_approval_id> --transfer-amount-usd 25",
    "PYTHONPATH=src python -m quant_platform.cli build-hyperliquid-testnet-lifecycle-gate",
    "PYTHONPATH=src python -m quant_platform.cli capture-hyperliquid-testnet-lifecycle-evidence",
    (
        "PYTHONPATH=src python -m quant_platform.cli "
        "hyperliquid-testnet-recover-pair-state "
        "--order-approval-id <signed_one_run_approval_id>"
    ),
    "PYTHONPATH=src python -m quant_platform.cli run-hyperliquid-wizard-mode-proofs",
    "PYTHONPATH=src python -m quant_platform.cli refresh-hyperliquid-execution-cost-snapshot --max-pairs 5",
    "PYTHONPATH=src python -m quant_platform.cli build-hyperliquid-evidence-cadence",
    "PYTHONPATH=src python -m quant_platform.cli build-current-wizard-hyperliquid-handoff",
    "PYTHONPATH=src python -m quant_platform.cli materialize-current-wizard-hyperliquid-history",
    "PYTHONPATH=src python -m quant_platform.cli run-current-wizard-hyperliquid-canonical-replay",
    "PYTHONPATH=src python -m quant_platform.cli materialize-current-wizard-hyperliquid-cost-evidence",
    "PYTHONPATH=src python -m quant_platform.cli run-current-wizard-hyperliquid-observed-cost-replay",
    "PYTHONPATH=src python -m quant_platform.cli run-current-wizard-hyperliquid-walkforward",
    "PYTHONPATH=src python -m quant_platform.cli build-current-wizard-hyperliquid-regime-attribution",
    "PYTHONPATH=src python -m quant_platform.cli run-current-wizard-hyperliquid-robustness",
    "PYTHONPATH=src python -m quant_platform.cli build-current-wizard-hyperliquid-concentration",
    "PYTHONPATH=src python -m quant_platform.cli build-current-wizard-hyperliquid-failure-attribution",
    "PYTHONPATH=src python -m quant_platform.cli build-current-wizard-hyperliquid-leverage-surface",
    "PYTHONPATH=src python -m quant_platform.cli build-current-wizard-hyperliquid-learning-ledger",
    "PYTHONPATH=src python -m quant_platform.cli validate-current-wizard-hyperliquid-chain",
    "PYTHONPATH=src python -m quant_platform.cli build-current-wizard-hyperliquid-operating-cadence",
    "PYTHONPATH=src python -m quant_platform.cli run-current-wizard-hyperliquid-daily-pipeline",
    (
        "PYTHONPATH=src python -m quant_platform.cli "
        "run-current-wizard-hyperliquid-daily-pipeline --execute-daily-pipeline"
    ),
    (
        "PYTHONPATH=src python -m quant_platform.cli "
        "build-current-wizard-hyperliquid-completion-audit"
    ),
    (
        "PYTHONPATH=src python -m quant_platform.cli "
        "build-current-wizard-hyperliquid-storage-reclamation-plan"
    ),
    (
        "PYTHONPATH=src python -m quant_platform.cli "
        "stage-current-wizard-hyperliquid-archive-copy "
        "--wizard-archive-destination /Volumes/<drive>/TheWizArchive "
        "--archive-copy-approval-id <one_run_approval_id>"
    ),
    (
        "PYTHONPATH=src python -m quant_platform.cli "
        "plan-current-wizard-hyperliquid-archive-release"
    ),
    (
        "PYTHONPATH=src python -m quant_platform.cli "
        "validate-current-wizard-hyperliquid-testnet-protocol"
    ),
    "PYTHONPATH=src python -m quant_platform.cli build-current-wizard-ou-optimal-overlay",
    "PYTHONPATH=src python -m quant_platform.cli build-market-venue-context",
    "PYTHONPATH=src python -m quant_platform.cli build-venue-lane-test-plan",
    "PYTHONPATH=src python -m quant_platform.cli build-multi-venue-history-readiness",
    "PYTHONPATH=src python -m quant_platform.cli build-venue-route-scorecard",
    "PYTHONPATH=src python -m quant_platform.cli hyperliquid-lane-readiness",
    "PYTHONPATH=src python -m quant_platform.cli build-pair-universe",
    "PYTHONPATH=src python -m quant_platform.cli build-trade-dataset",
    "PYTHONPATH=src python -m quant_platform.cli train-trade-gate",
    "PYTHONPATH=src python -m quant_platform.cli run-model-gated-backtest",
    "PYTHONPATH=src python -m quant_platform.cli export-trade-gate-model",
    "PYTHONPATH=src python -m quant_platform.cli build-command-dashboard",
    "PYTHONPATH=src python -m quant_platform.cli build-v2-preflight-run",
    "PYTHONPATH=src python -m quant_platform.cli validate-v2-run --run-id <run_id>",
    "PYTHONPATH=src python -m quant_platform.cli publish-v2-run-status --run-id <run_id>",
    "PYTHONPATH=src python -m quant_platform.cli apify-source-summary",
    "PYTHONPATH=src python -m quant_platform.cli refresh-apify-sources",
    "PYTHONPATH=src python -m quant_platform.cli archive-from-index --dry-run",
]


def build_artifact_index(root: Path = ROOT) -> CommandResult:
    active = root / "reports" / "active"
    rows = [_artifact_row(path, root) for path in _iter_repo_files(root)]
    frame = pd.DataFrame(rows, columns=ARTIFACT_COLUMNS).sort_values("path").reset_index(drop=True)
    csv_path = active / "artifact_index.csv"
    md_path = active / "artifact_index.md"
    commands_path = active / "canonical_commands.md"
    _write_csv(frame, csv_path)
    _write_text(md_path, _artifact_index_markdown(frame))
    _write_text(commands_path, _canonical_commands_markdown())
    return CommandResult(
        paths={"artifact_index": csv_path, "artifact_index_md": md_path, "canonical_commands": commands_path},
        summary={
            "artifacts": int(len(frame)),
            "active": int((frame["status"] == "active").sum()),
            "historical_evidence": int((frame["status"] == "historical_evidence").sum()),
            "do_not_move": int((frame["status"] == "do_not_move").sum()),
            "safe_to_archive_later": int(frame["safe_to_archive_later"].astype(bool).sum()),
        },
    )


def current_state(root: Path = ROOT) -> CommandResult:
    from quant_platform.rl.brain_contract import build_phase1_readiness_surfaces

    active = root / "reports" / "active"
    dashboard = root / "reports" / "dashboard"
    data_ml = root / "data" / "ml"
    readiness_paths = build_phase1_readiness_surfaces(root)
    base_rl_handoff = _read_csv(root / "reports" / "rl" / "base_rl_paper_handoff_status.csv")
    handoff_ready = bool(
        not base_rl_handoff.empty
        and base_rl_handoff.get("paper_authorized", pd.Series([False])).fillna(False).astype(bool).iloc[0]
    )
    handoff_status = (
        str(base_rl_handoff.get("status", pd.Series(["missing"])).iloc[0]) if not base_rl_handoff.empty else "missing"
    )
    handoff_blocker = (
        str(base_rl_handoff.get("blocker", pd.Series(["missing_base_rl_handoff"])).iloc[0]) if not base_rl_handoff.empty else "missing_base_rl_handoff"
    )
    handoff_detail = (
        f"status={handoff_status};paper_authorized={handoff_ready};"
        f"journal_monitoring_authorized={str(base_rl_handoff.get('journal_monitoring_authorized', pd.Series([False])).iloc[0]) if not base_rl_handoff.empty else False};"
        f"journal_learning_ready={str(base_rl_handoff.get('journal_learning_ready', pd.Series([False])).iloc[0]) if not base_rl_handoff.empty else False};"
        f"execution_truth_mode={str(base_rl_handoff.get('execution_truth_mode', pd.Series(['paper_only'])).iloc[0]) if not base_rl_handoff.empty else 'paper_only'};"
        f"injective_mirrorable_pairs={int(base_rl_handoff.get('injective_mirrorable_pairs', pd.Series([0])).fillna(0).iloc[0]) if not base_rl_handoff.empty else 0}"
        if not base_rl_handoff.empty
        else "base RL handoff missing"
    )
    injective_ready = bool(
        not base_rl_handoff.empty
        and str(base_rl_handoff.get("injective_checked", pd.Series([False])).iloc[0]).strip().lower() in {"true", "1", "yes"}
    )
    injective_blocker = (
        str(base_rl_handoff.get("injective_blocker", pd.Series(["injective_execution_compatibility_table_missing"])).iloc[0])
        if not base_rl_handoff.empty
        else "injective_execution_compatibility_table_missing"
    )
    injective_mirrorable_pairs = (
        int(base_rl_handoff.get("injective_mirrorable_pairs", pd.Series([0])).fillna(0).iloc[0])
        if not base_rl_handoff.empty
        else 0
    )
    gmx_shortlist = _read_csv(active / "gmx_testnet_candidate_shortlist.csv")
    gmx_inventory = _read_csv(active / "gmx_testnet_market_inventory.csv")
    gmx_supported_pairs = int(len(gmx_shortlist)) if not gmx_shortlist.empty else 0
    gmx_market_rows = int(len(gmx_inventory)) if not gmx_inventory.empty else 0
    gmx_fetch_blocker = ""
    if not gmx_inventory.empty and "fetch_blocker" in gmx_inventory.columns:
        blockers = [
            str(value).strip()
            for value in gmx_inventory["fetch_blocker"].dropna().astype(str).tolist()
            if str(value).strip() and str(value).strip().lower() != "nan"
        ]
        gmx_fetch_blocker = ";".join(sorted(set(blockers)))
    hyperliquid_compatibility = _read_csv(active / "hyperliquid_execution_market_compatibility.csv")
    hyperliquid_shortlist = _read_csv(active / "hyperliquid_testnet_candidate_shortlist.csv")
    hyperliquid_inventory = _read_csv(active / "hyperliquid_testnet_market_inventory.csv")
    hyperliquid_supported_pairs = (
        int(hyperliquid_compatibility.get("mirrorable_for_paper", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        if not hyperliquid_compatibility.empty
        else (int(len(hyperliquid_shortlist)) if not hyperliquid_shortlist.empty else 0)
    )
    hyperliquid_market_rows = (
        int(hyperliquid_inventory.get("tradable_perp", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        if not hyperliquid_inventory.empty
        else 0
    )
    hyperliquid_fetch_blocker = ""
    if not hyperliquid_inventory.empty and "fetch_blocker" in hyperliquid_inventory.columns:
        blockers = [
            str(value).strip()
            for value in hyperliquid_inventory["fetch_blocker"].dropna().astype(str).tolist()
            if str(value).strip() and str(value).strip().lower() != "nan"
        ]
        hyperliquid_fetch_blocker = ";".join(sorted(set(blockers)))
    hyperliquid_preflight_ready = False
    hyperliquid_preflight_blocker = "hyperliquid_testnet_candidate_shortlist_missing"
    if not hyperliquid_shortlist.empty:
        hyperliquid_preflight_ready = bool(
            hyperliquid_shortlist.get("order_preflight_ready", pd.Series(dtype=bool)).fillna(False).astype(bool).any()
        )
        blockers = [
            str(value).strip()
            for value in hyperliquid_shortlist.get("order_preflight_blocker", pd.Series(dtype=object)).dropna().astype(str).tolist()
            if str(value).strip() and str(value).strip().lower() != "nan"
        ]
        hyperliquid_preflight_blocker = ";".join(sorted(set(blockers)))
    venue_route_state = _venue_route_state_row(root)
    rows = [
        *_seven_stage_state_rows(root),
        _active_layer_state_row(root),
        _state_row(
            "pair_universe",
            _exists(root / "data" / "processed" / "pair_universe.csv"),
            "pair universe exists" if (root / "data" / "processed" / "pair_universe.csv").exists() else "pair universe missing",
            root / "data" / "processed" / "pair_universe.csv",
            "run build-pair-universe",
        ),
        _state_row(
            "market_venue_context",
            _exists(root / "data" / "processed" / "market_venue_context.csv"),
            "market venue context exists"
            if (root / "data" / "processed" / "market_venue_context.csv").exists()
            else "market venue context missing",
            root / "data" / "processed" / "market_venue_context.csv",
            "run build-market-venue-context",
        ),
        _state_row(
            "venue_lane_test_plan",
            _exists(active / "venue_lane_test_plan.csv"),
            "venue lane test plan exists" if (active / "venue_lane_test_plan.csv").exists() else "venue lane test plan missing",
            active / "venue_lane_test_plan.csv",
            "run build-venue-lane-test-plan",
        ),
        _state_row(
            "hyperliquid_lane",
            _exists(active / "hyperliquid_lane_readiness.csv"),
            "hyperliquid lane readiness exists"
            if (active / "hyperliquid_lane_readiness.csv").exists()
            else "hyperliquid lane readiness missing",
            active / "hyperliquid_lane_readiness.csv",
            "run hyperliquid-lane-readiness",
        ),
        _hyperliquid_public_context_state_row(root),
        _hyperliquid_research_bundle_state_row(root),
        _hyperliquid_wizard_hypothesis_state_row(root),
        _hyperliquid_wizard_mode_proof_state_row(root),
        _wizard_ou_v6_terminal_state_row(root),
        _hyperliquid_slippage_calibration_state_row(root),
        venue_route_state,
        _strategy_acceptance_state_row(root),
        _state_row(
            "wizard_mode_matrix_capture",
            _exists(active / "wizard_mode_matrix_capture_queue.csv"),
            "seven-mode capture matrix exists"
            if (active / "wizard_mode_matrix_capture_queue.csv").exists()
            else "seven-mode capture matrix missing",
            active / "wizard_mode_matrix_capture_queue.csv",
            "run build-wizard-mode-matrix-capture-queue",
        ),
        _state_row(
            "wizard_mode_replay_capability",
            _exists(active / "wizard_mode_replay_capability.csv"),
            "mode replay capability board exists"
            if (active / "wizard_mode_replay_capability.csv").exists()
            else "mode replay capability board missing",
            active / "wizard_mode_replay_capability.csv",
            "run build-wizard-mode-replay-capability",
        ),
        _state_row(
            "wizard_mode_comparison",
            _exists(active / "wizard_mode_comparison.csv"),
            "cross-mode research comparison exists"
            if (active / "wizard_mode_comparison.csv").exists()
            else "cross-mode research comparison missing",
            active / "wizard_mode_comparison.csv",
            "run build-wizard-mode-comparison after confirmed settings capture",
        ),
        _state_row(
            "trade_dataset",
            _exists(data_ml / "trade_training_dataset.csv"),
            "trade dataset exists" if (data_ml / "trade_training_dataset.csv").exists() else "trade dataset missing",
            data_ml / "trade_training_dataset.csv",
            "run build-trade-dataset",
        ),
        _state_row(
            "youtube_research_brain",
            _exists(root / "reports" / "agents" / "youtube_brain_status.csv"),
            "YouTube research brain artifacts exist"
            if (root / "reports" / "agents" / "youtube_brain_status.csv").exists()
            else "YouTube research brain artifacts missing",
            root / "reports" / "agents" / "youtube_brain_status.csv",
            "run run-youtube-brain --no-fetch or enable the daily collector",
        ),
        _model_authority_state_row(root),
        {
            "area": "base_rl_handoff",
            "ready": handoff_ready,
            "status": "ready" if handoff_ready else handoff_status,
            "blocker": "" if handoff_ready else handoff_blocker,
            "detail": handoff_detail,
            "pair": "",
            "candidate_id": "",
            "setup_identity": "",
            "setup_role": "",
            "setup_status": handoff_status,
            "setup_blocker": "" if handoff_ready else handoff_blocker,
            "evidence_path": root / "reports" / "rl" / "base_rl_paper_handoff_status.csv",
            "next_action": "run run-base-rl or base-rl-paper-handoff",
        },
        {
            "area": "injective_mirror_lane",
            "ready": injective_ready and injective_mirrorable_pairs > 0,
            "status": "ready" if injective_ready and injective_mirrorable_pairs > 0 else ("checked" if injective_ready else "missing"),
            "blocker": "" if injective_ready and injective_mirrorable_pairs > 0 else injective_blocker,
            "detail": f"injective_mirrorable_pairs={injective_mirrorable_pairs}",
            "pair": "",
            "candidate_id": "",
            "setup_identity": "",
            "setup_role": "",
            "setup_status": "spot_supported" if injective_mirrorable_pairs > 0 else "paper_only",
            "setup_blocker": "" if injective_mirrorable_pairs > 0 else injective_blocker,
            "evidence_path": root / "reports" / "active" / "injective_execution_market_compatibility.csv",
            "next_action": "refresh Injective compatibility and queue before trusting Injective spot routing",
        },
        {
            "area": "gmx_testnet_lane",
            "ready": bool(gmx_supported_pairs > 0 and not gmx_fetch_blocker),
            "status": "market_supported" if gmx_supported_pairs > 0 and not gmx_fetch_blocker else ("checked" if gmx_market_rows > 0 else "missing"),
            "blocker": "" if gmx_supported_pairs > 0 and not gmx_fetch_blocker else (gmx_fetch_blocker or "gmx_no_supported_current_pairs"),
            "detail": f"gmx_testnet_markets={gmx_market_rows};gmx_supported_pairs={gmx_supported_pairs}",
            "pair": "",
            "candidate_id": "",
            "setup_identity": "",
            "setup_role": "",
            "setup_status": "testnet_perp_supported" if gmx_supported_pairs > 0 else "read_only_checked",
            "setup_blocker": "" if gmx_supported_pairs > 0 and not gmx_fetch_blocker else (gmx_fetch_blocker or "gmx_no_supported_current_pairs"),
            "evidence_path": active / "gmx_testnet_candidate_shortlist.csv",
            "next_action": "use GMX as a read-only testnet venue filter until an order adapter and wallet preflight are proven",
        },
        {
            "area": "hyperliquid_testnet_lane",
            "ready": bool(hyperliquid_supported_pairs > 0 and not hyperliquid_fetch_blocker),
            "status": "market_supported"
            if hyperliquid_supported_pairs > 0 and not hyperliquid_fetch_blocker
            else ("checked" if hyperliquid_market_rows > 0 else "missing"),
            "blocker": ""
            if hyperliquid_supported_pairs > 0 and not hyperliquid_fetch_blocker
            else (hyperliquid_fetch_blocker or "hyperliquid_no_supported_current_pairs"),
            "detail": (
                f"hyperliquid_testnet_perp_markets={hyperliquid_market_rows};"
                f"hyperliquid_supported_pairs={hyperliquid_supported_pairs};"
                f"order_preflight_ready={hyperliquid_preflight_ready}"
            ),
            "pair": "",
            "candidate_id": "",
            "setup_identity": "",
            "setup_role": "",
            "setup_status": "testnet_perp_supported"
            if hyperliquid_supported_pairs > 0
            else "read_only_checked",
            "setup_blocker": "" if hyperliquid_preflight_ready else hyperliquid_preflight_blocker,
            "evidence_path": active / "hyperliquid_execution_market_compatibility.csv",
            "next_action": "use Hyperliquid as a testnet venue filter; configure signed adapter and wallet preflight before any submit",
        },
        _readiness_state_row(
            "wizard_readiness",
            readiness_paths["wizard_readiness"],
            "run brain-readiness-report to refresh Wizard current state",
        ),
        _readiness_state_row(
            "native_readiness",
            readiness_paths["native_readiness"],
            "emit contract-aligned native candidate packets to refresh Native current state",
        ),
        _readiness_state_row(
            "overall_readiness",
            readiness_paths["overall_readiness"],
            "refresh Wizard, Native, and paper aliases before trusting the overall answer",
        ),
        _readiness_state_row(
            "paper_status",
            readiness_paths["paper_status"],
            "run base-rl-paper-handoff to refresh the current paper answer",
        ),
        _readiness_state_row(
            "orchestrator_status",
            root / "reports" / "brain" / "orchestrator_status.csv",
            "run the three-brain system build to refresh orchestrator truth",
        ),
        _state_row(
            "dashboard",
            _exists(dashboard / "command_center.md"),
            "command dashboard exists" if (dashboard / "command_center.md").exists() else "command dashboard missing",
            dashboard / "command_center.md",
            "run build-command-dashboard",
        ),
    ]
    frame = pd.DataFrame(rows)
    authority = _read_csv(active / "hyperliquid_authority_state.csv")
    if not authority.empty:
        authority_row = authority.iloc[0]
        canonical_rows = _canonical_hyperliquid_state_rows(root, authority_row)
        legacy_areas = {
            "strategy_acceptance",
            "base_rl_handoff",
            "injective_mirror_lane",
            "gmx_testnet_lane",
            "wizard_readiness",
            "native_readiness",
            "overall_readiness",
            "paper_status",
            "orchestrator_status",
        }
        legacy = frame["area"].astype(str).isin(legacy_areas)
        frame.loc[legacy, "ready"] = False
        frame.loc[legacy, "status"] = "historical_non_authoritative"
        frame.loc[legacy, "blocker"] = "retired_from_hyperliquid_execution_authority"
        frame.loc[legacy, "setup_blocker"] = "retired_from_hyperliquid_execution_authority"
        frame.loc[legacy, "next_action"] = "retain for historical comparison; do not use for current execution decisions"
        frame = pd.concat([pd.DataFrame(canonical_rows), frame], ignore_index=True, sort=False)
    csv_path = active / "current_state.csv"
    md_path = active / "current_state.md"
    _write_csv(frame, csv_path)
    _write_text(md_path, _current_state_markdown(frame))
    return CommandResult(paths={"current_state": csv_path, "current_state_md": md_path}, summary={"rows": len(frame)})


def system_check(root: Path = ROOT) -> CommandResult:
    ACTIVE = root / "reports" / "active"
    DASHBOARD = root / "reports" / "dashboard"
    rows = []
    for folder in ["data", "data/raw", "data/processed", "reports", "src/quant_platform"]:
        path = root / folder
        rows.append(_check_row(f"folder:{folder}", path.exists(), str(path), "create folder or restore repo state"))
    for key in [
        "CRYPTO_WIZARDS_API_KEY",
        "CRYPTO_WIZARDS_BASE_URL",
        "HYPERLIQUID_NETWORK",
        "HYPERLIQUID_TESTNET_BASE_URL",
        "HYPERLIQUID_MASTER_ADDRESS",
        "HYPERLIQUID_AGENT_ADDRESS",
        "HYPERLIQUID_TESTNET_AGENT_KEYCHAIN_SERVICE",
        "HYPERLIQUID_TESTNET_APPROVAL_KEYCHAIN_SERVICE",
        "HYPERLIQUID_TESTNET_APPROVAL_KEYCHAIN_ACCOUNT",
        "HYPERLIQUID_TESTNET_REQUESTED_LEVERAGE",
        "HYPERLIQUID_TESTNET_MARGIN_MODE",
        "HYPERLIQUID_TESTNET_ONE_X_PROOF_ID",
        "HYPERLIQUID_TESTNET_LEVERAGE_SCENARIO_ID",
        "HYPERLIQUID_TESTNET_SUBMIT_ORDERS",
    ]:
        present = _configured_env_key_present(root, key)
        required = key not in {
            "CRYPTO_WIZARDS_API_KEY",
            "HYPERLIQUID_TESTNET_APPROVAL_KEYCHAIN_SERVICE",
            "HYPERLIQUID_TESTNET_APPROVAL_KEYCHAIN_ACCOUNT",
            "HYPERLIQUID_TESTNET_REQUESTED_LEVERAGE",
            "HYPERLIQUID_TESTNET_MARGIN_MODE",
            "HYPERLIQUID_TESTNET_ONE_X_PROOF_ID",
            "HYPERLIQUID_TESTNET_LEVERAGE_SCENARIO_ID",
            "HYPERLIQUID_TESTNET_SUBMIT_ORDERS",
        }
        rows.append(
            {
                "check": f"env:{key}",
                "ready": present or not required,
                "status": "present" if present else ("missing_required" if required else "missing_optional"),
                "blocker": "" if present or not required else f"missing_{key.lower()}",
                "evidence_path": ".env.local/.env.example",
                "next_action": "fill .env.local if this integration is needed",
            }
        )
    for package in ["numpy", "pandas", "sklearn", "statsmodels", "requests", "yaml"]:
        rows.append(_package_check_row(package))
    scheduler_runtime_path = ACTIVE / "scheduler_runtime_readiness.json"
    scheduler_runtime = _read_json(scheduler_runtime_path)
    scheduler_agents_ready = int(scheduler_runtime.get("agents_ready", 0) or 0)
    scheduler_agents_expected = int(
        scheduler_runtime.get("agents_expected", 3) or 3
    )
    scheduler_authority_safe = all(
        scheduler_runtime.get(field) is False
        for field in (
            "candidate_promotion_authority",
            "order_submission_included",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    ) and int(scheduler_runtime.get("orders_submitted", -1) or 0) == 0
    scheduler_runtime_ready = bool(
        scheduler_runtime.get("status") == "PASS_SCHEDULER_RUNTIME_READY"
        and scheduler_agents_expected == 3
        and scheduler_agents_ready == scheduler_agents_expected
        and int(scheduler_runtime.get("checks_total", 0) or 0) > 0
        and int(scheduler_runtime.get("checks_passed", 0) or 0)
        == int(scheduler_runtime.get("checks_total", 0) or 0)
        and scheduler_authority_safe
    )
    scheduler_runtime_blockers = scheduler_runtime.get("blockers", [])
    if not isinstance(scheduler_runtime_blockers, list):
        scheduler_runtime_blockers = [str(scheduler_runtime_blockers)]
    if not scheduler_runtime:
        scheduler_runtime_blockers = ["scheduler_runtime_receipt_missing"]
    elif not scheduler_authority_safe:
        scheduler_runtime_blockers.append("scheduler_runtime_authority_not_zero")
    scheduler_runtime_warnings = scheduler_runtime.get("operational_warnings", [])
    if not isinstance(scheduler_runtime_warnings, list):
        scheduler_runtime_warnings = [str(scheduler_runtime_warnings)]
    rows.append(
        {
            "check": "scheduler:live_runtime_contract",
            "ready": scheduler_runtime_ready,
            "status": (
                "ready_with_operational_warnings"
                if scheduler_runtime_ready and scheduler_runtime_warnings
                else "ready"
                if scheduler_runtime_ready
                else str(scheduler_runtime.get("status", "missing"))
            ),
            "blocker": (
                ""
                if scheduler_runtime_ready
                else ";".join(
                    str(value) for value in scheduler_runtime_blockers if str(value)
                )
                or "scheduler_runtime_contract_not_ready"
            ),
            "evidence_path": str(scheduler_runtime_path),
            "next_action": (
                "monitor_scheduler_runtime_and_system_volume_warning"
                if scheduler_runtime_ready and scheduler_runtime_warnings
                else "monitor_scheduler_runtime_receipt"
                if scheduler_runtime_ready
                else "run build-scheduler-runtime-readiness and repair every blocker"
            ),
        }
    )
    storage = dashboard_storage_preflight(root)
    rows.append(
        {
            "check": "storage:deep_dashboard_refresh",
            "ready": bool(storage["ready"]),
            "status": "ready" if storage["ready"] else "blocked_insufficient_free_space",
            "blocker": str(storage["blocker"]),
            "evidence_path": str(root),
            "next_action": (
                "free_at_least_"
                f"{storage['required_gib']}GiB_before_running_deep_dashboard_refresh"
                if not storage["ready"]
                else "run build-command-dashboard"
            ),
        }
    )
    reclamation_manifest_path = (
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_storage_reclamation_manifest.json"
    )
    reclamation = _read_json(reclamation_manifest_path)
    archive_copy_ready = bool(reclamation.get("archive_copy_preflight_ready", False))
    archive_destination_status = str(
        reclamation.get("archive_destination_status", "BLOCKED")
    )
    archive_destination_blocker = str(
        reclamation.get(
            "archive_destination_blocker",
            "missing_storage_reclamation_plan",
        )
    )
    rows.append(
        {
            "check": "storage:off_volume_archive_destination",
            "ready": archive_copy_ready,
            "status": archive_destination_status,
            "blocker": "" if archive_copy_ready else archive_destination_blocker,
            "evidence_path": str(reclamation_manifest_path),
            "next_action": (
                "review_verified_copy_plan; archive_apply_remains_disabled"
                if archive_copy_ready
                else "configure_or_mount_off_volume_archive_destination_and_rebuild_reclamation_plan"
            ),
        }
    )
    archive_copy_manifest_path = (
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_archive_copy_manifest.json"
    )
    archive_copy = _read_json(archive_copy_manifest_path)
    archive_copy_completed = bool(archive_copy.get("archive_copy_completed", False))
    rows.append(
        {
            "check": "storage:verified_archive_copy",
            "ready": archive_copy_completed,
            "status": "READY" if archive_copy_completed else "BLOCKED",
            "blocker": "" if archive_copy_completed else "verified_off_volume_archive_copy_missing",
            "evidence_path": str(archive_copy_manifest_path),
            "next_action": (
                "review_copy_receipt; source_release_remains_separately_disabled"
                if archive_copy_completed
                else "run_explicitly_approved_copy_only_archive_stage_after_destination_preflight"
            ),
        }
    )
    archive_release_manifest_path = (
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_archive_release_manifest.json"
    )
    archive_release = _read_json(archive_release_manifest_path)
    archive_release_ready = bool(archive_release.get("release_dry_run_ready", False))
    rows.append(
        {
            "check": "storage:archive_release_dry_run",
            "ready": archive_release_ready,
            "status": str(archive_release.get("release_status", "BLOCKED")),
            "blocker": (
                "" if archive_release_ready else str(
                    archive_release.get(
                        "release_blocker",
                        "archive_release_dry_run_missing",
                    )
                )
            ),
            "evidence_path": str(archive_release_manifest_path),
            "next_action": (
                "review_exact_release_plan; source_release_remains_unauthorized"
                if archive_release_ready
                else "complete_verified_archive_copy_then_rebuild_release_dry_run"
            ),
        }
    )
    testnet_protocol_manifest_path = (
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_testnet_protocol_manifest.json"
    )
    testnet_protocol = _read_json(testnet_protocol_manifest_path)
    testnet_protocol_ready = bool(
        testnet_protocol.get("protocol_status") == "PASS"
        and testnet_protocol.get("simulation_only") is True
        and testnet_protocol.get("simulation_is_testnet_proof") is False
        and testnet_protocol.get("order_submission_performed") is False
        and testnet_protocol.get("live_trading_authorized") is False
    )
    rows.append(
        {
            "check": "hyperliquid_testnet:deterministic_lifecycle_protocol",
            "ready": testnet_protocol_ready,
            "status": "READY" if testnet_protocol_ready else "BLOCKED",
            "blocker": (
                ""
                if testnet_protocol_ready
                else "deterministic_testnet_protocol_validation_missing_or_failed"
            ),
            "evidence_path": str(testnet_protocol_manifest_path),
            "next_action": (
                "retain_as_simulation_prerequisite; actual_testnet_proof_remains_separate"
                if testnet_protocol_ready
                else "run_validate_current_wizard_hyperliquid_testnet_protocol"
            ),
        }
    )
    for artifact in [
        ACTIVE / "artifact_index.csv",
        root / "data" / "processed" / "pair_universe.csv",
        root / "data" / "processed" / "market_venue_context.csv",
        root / "data" / "processed" / "hyperliquid_market_context.csv",
        ACTIVE / "venue_lane_test_plan.csv",
        ACTIVE / "hyperliquid_lane_readiness.csv",
        ACTIVE / "hyperliquid_research_bundle.csv",
        ACTIVE / "hyperliquid_wizard_hypothesis_queue.csv",
        ACTIVE / "hyperliquid_wizard_vendor_mode_proofs.csv",
        ACTIVE / "hyperliquid_l2_slippage_samples.csv",
        ACTIVE / "hyperliquid_pair_cost_model.csv",
        ACTIVE / "hyperliquid_evidence_cadence.csv",
        ACTIVE / "hyperliquid_run_manifest.json",
        ACTIVE / "hyperliquid_run_candidates.csv",
        ACTIVE / "hyperliquid_run_manifest_validation.csv",
        ACTIVE / "hyperliquid_authority_state.csv",
        root / "reports" / "orchestration" / "teacher_council" / "research_family_selection_controls.csv",
        root / "reports" / "orchestration" / "teacher_council" / "research_family_registry.csv",
        ACTIVE / "hyperliquid_testnet_smoke_approval.json",
        ACTIVE / "hyperliquid_testnet_lifecycle_gate.csv",
        ACTIVE / "hyperliquid_testnet_lifecycle_evidence_capture.csv",
        ACTIVE / "current_wizard_hyperliquid_daily_run_status.csv",
        ACTIVE / "current_wizard_hyperliquid_daily_run_manifest.json",
        ACTIVE / "current_wizard_hyperliquid_completion_audit.csv",
        ACTIVE / "current_wizard_hyperliquid_completion_audit_manifest.json",
        ACTIVE / "exhaustive_wizard_api_refresh_manifest.json",
        ACTIVE / "exhaustive_wizard_api_refresh_delta.csv",
        ACTIVE / "exhaustive_wizard_api_refresh_hyperliquid_mapping.csv",
        ACTIVE / "exhaustive_wizard_api_refresh_pair_detail_queue.csv",
        ACTIVE / "exhaustive_wizard_pair_detail_acquisition_plan.csv",
        ACTIVE / "exhaustive_wizard_api_refresh_validation.csv",
        ACTIVE / "wizard_pair_detail_api_pilot_manifest.json",
        ACTIVE / "wizard_pair_detail_api_pilot_manifest.csv",
        ACTIVE / "wizard_pair_detail_api_pilot_fields.csv",
        ACTIVE / "wizard_pair_detail_api_pilot_coverage.csv",
        ACTIVE / "wizard_pair_detail_api_pilot_attempt_history.csv",
        ACTIVE / "current_wizard_pair_detail_status.csv",
        ACTIVE / "current_wizard_hyperliquid_experiment_matrix.csv",
        ACTIVE / "current_wizard_hyperliquid_pair_history_queue.csv",
        ACTIVE / "current_wizard_hyperliquid_asset_fetch_queue.csv",
        ACTIVE / "current_wizard_hyperliquid_handoff_validation.csv",
        ACTIVE / "current_wizard_hyperliquid_handoff_manifest.json",
        ACTIVE / "current_wizard_hyperliquid_asset_history_results.csv",
        ACTIVE / "current_wizard_hyperliquid_pair_history_results.csv",
        ACTIVE / "current_wizard_hyperliquid_history_validation.csv",
        ACTIVE / "current_wizard_hyperliquid_history_manifest.json",
        ACTIVE / "current_wizard_hyperliquid_canonical_replay.csv",
        ACTIVE / "current_wizard_hyperliquid_canonical_replay_ranked.csv",
        ACTIVE / "current_wizard_hyperliquid_replay_pair_status.csv",
        ACTIVE / "current_wizard_hyperliquid_replay_validation.csv",
        ACTIVE / "current_wizard_hyperliquid_canonical_replay_manifest.json",
        ACTIVE / "current_wizard_hyperliquid_funding_asset_results.csv",
        ACTIVE / "current_wizard_hyperliquid_pair_cost_evidence.csv",
        ACTIVE / "current_wizard_hyperliquid_experiment_cost_readiness.csv",
        ACTIVE / "current_wizard_hyperliquid_cost_validation.csv",
        ACTIVE / "current_wizard_hyperliquid_cost_manifest.json",
        ACTIVE / "current_wizard_hyperliquid_observed_cost_replay.csv",
        ACTIVE / "current_wizard_hyperliquid_observed_cost_replay_ranked.csv",
        ACTIVE / "current_wizard_hyperliquid_observed_cost_replay_validation.csv",
        ACTIVE / "current_wizard_hyperliquid_observed_cost_replay_manifest.json",
        ACTIVE / "current_wizard_hyperliquid_walkforward_status.csv",
        ACTIVE / "current_wizard_hyperliquid_walkforward_candidates.csv",
        ACTIVE / "current_wizard_hyperliquid_walkforward_validation.csv",
        ACTIVE / "current_wizard_hyperliquid_walkforward_manifest.json",
        ACTIVE / "current_wizard_hyperliquid_regime_status.csv",
        ACTIVE / "current_wizard_hyperliquid_regime_candidates.csv",
        ACTIVE / "current_wizard_hyperliquid_regime_validation.csv",
        ACTIVE / "current_wizard_hyperliquid_regime_manifest.json",
        ACTIVE / "current_wizard_hyperliquid_robustness_status.csv",
        ACTIVE / "current_wizard_hyperliquid_robustness_candidates.csv",
        ACTIVE / "current_wizard_hyperliquid_robustness_validation.csv",
        ACTIVE / "current_wizard_hyperliquid_robustness_manifest.json",
        ACTIVE / "current_wizard_hyperliquid_concentration_status.csv",
        ACTIVE / "current_wizard_hyperliquid_concentration_cohorts.csv",
        ACTIVE / "current_wizard_hyperliquid_concentration_dimensions.csv",
        ACTIVE / "current_wizard_hyperliquid_concentration_contributors.csv",
        ACTIVE / "current_wizard_hyperliquid_concentration_validation.csv",
        ACTIVE / "current_wizard_hyperliquid_concentration_manifest.json",
        ACTIVE / "current_wizard_hyperliquid_failure_attribution.csv",
        ACTIVE / "current_wizard_hyperliquid_failure_attribution_summary.csv",
        ACTIVE / "current_wizard_hyperliquid_failure_attribution_validation.csv",
        ACTIVE / "current_wizard_hyperliquid_failure_attribution_manifest.json",
        ACTIVE / "current_wizard_hyperliquid_leverage_status.csv",
        ACTIVE / "current_wizard_hyperliquid_leverage_candidates.csv",
        ACTIVE / "current_wizard_hyperliquid_leverage_scenarios.csv",
        ACTIVE / "current_wizard_hyperliquid_leverage_validation.csv",
        ACTIVE / "current_wizard_hyperliquid_leverage_manifest.json",
        ACTIVE / "current_wizard_hyperliquid_learning_ledger.csv.gz",
        ACTIVE / "current_wizard_hyperliquid_learning_validation.csv",
        ACTIVE / "current_wizard_hyperliquid_learning_manifest.json",
        ACTIVE / "current_wizard_hyperliquid_chain_validation.csv",
        ACTIVE / "current_wizard_hyperliquid_chain_validation_manifest.json",
        ACTIVE / "current_wizard_hyperliquid_chain_validation_summary.md",
        ACTIVE / "current_wizard_hyperliquid_operating_cadence.csv",
        ACTIVE / "current_wizard_hyperliquid_operating_cadence_validation.csv",
        ACTIVE / "current_wizard_hyperliquid_operating_cadence_manifest.json",
        ACTIVE / "current_wizard_hyperliquid_live_lock.csv",
        ACTIVE / "current_wizard_hyperliquid_storage_efficiency.csv",
        ACTIVE / "current_wizard_hyperliquid_storage_reclamation_plan.csv",
        ACTIVE / "current_wizard_hyperliquid_storage_reclamation_validation.csv",
        ACTIVE / "current_wizard_hyperliquid_storage_reclamation_manifest.json",
        ACTIVE / "current_wizard_hyperliquid_storage_reclamation_summary.md",
        ACTIVE / "current_wizard_ou_optimal_overlay_ledger.csv",
        ACTIVE / "current_wizard_ou_optimal_overlay_coverage.csv",
        ACTIVE / "current_wizard_ou_optimal_overlay_validation.csv",
        ACTIVE / "current_wizard_ou_optimal_overlay_manifest.json",
        ACTIVE / "current_wizard_ou_optimal_overlay_summary.md",
        root
        / "data"
        / "meta_learning"
        / "current_wizard_hyperliquid_research_outcomes.csv.gz",
        ACTIVE / "exhaustive_wizard_hyperliquid_validation_manifest.json",
        ACTIVE / "exhaustive_wizard_hyperliquid_learning_ledger.csv",
        ACTIVE / "exhaustive_wizard_hyperliquid_learning_manifest.json",
        root
        / "data"
        / "meta_learning"
        / "exhaustive_wizard_hyperliquid_research_outcomes.csv",
        ACTIVE / "venue_route_scorecard.csv",
        ACTIVE / "venue_route_recommendations.csv",
        ACTIVE / "wizard_replay_handoff.csv",
        ACTIVE / "wizard_mode_matrix_capture_queue.csv",
        ACTIVE / "wizard_mode_replay_capability.csv",
        ACTIVE / "wizard_mode_comparison.csv",
        ACTIVE / "wizard_exploratory_cost_sensitivity.csv",
        DATA_ML / "trade_training_dataset.csv",
        MODELS / "model.pkl",
        DASHBOARD / "command_center.md",
    ]:
        rows.append(_check_row(f"artifact:{artifact.name}", artifact.exists(), str(artifact), "run the corresponding active pipeline command"))
    authority = _read_csv(ACTIVE / "hyperliquid_authority_state.csv")
    if authority.empty:
        rows.append(_check_row("readiness:canonical_hyperliquid_authority", False, str(ACTIVE / "hyperliquid_authority_state.csv"), "run the Hyperliquid research cycle"))
    else:
        record = authority.iloc[0]
        for layer, column in [
            ("infrastructure", "infrastructure_ready"),
            ("data", "data_ready"),
            ("research", "research_ready"),
            ("paper", "paper_ready"),
            ("live", "live_ready"),
        ]:
            ready = _truthy(record.get(column, False))
            rows.append(
                _check_row(
                    f"readiness:{layer}",
                    ready,
                    str(ACTIVE / "hyperliquid_authority_state.csv"),
                    "resolve the canonical Hyperliquid authority blockers",
                )
            )
    frame = pd.DataFrame(rows)
    csv_path = ACTIVE / "system_check.csv"
    md_path = ACTIVE / "system_check.md"
    _write_csv(frame, csv_path)
    _write_text(md_path, _simple_report_markdown("System Check", frame))
    return CommandResult(
        paths={"system_check": csv_path, "system_check_md": md_path},
        summary={"checks": len(frame), "ready": int(frame["ready"].astype(bool).sum()), "blocked": int((~frame["ready"].astype(bool)).sum())},
    )


def _canonical_hyperliquid_state_rows(root: Path, authority: pd.Series) -> list[dict[str, object]]:
    active = root / "reports" / "active"
    council = root / "reports" / "orchestration" / "teacher_council"
    run_id = str(authority.get("run_id", "") or "")
    candidate_set_id = str(authority.get("candidate_set_id", "") or "")
    common = {
        "pair": "",
        "candidate_id": candidate_set_id,
        "setup_identity": run_id,
        "setup_role": "canonical_hyperliquid_authority",
    }
    family_selection_path = council / "research_family_selection_controls.csv"
    selection_path = family_selection_path if family_selection_path.exists() else council / "statistical_selection_controls.csv"
    selection = _read_csv(selection_path)
    primary_selection = selection
    if not selection.empty and "research_lane" in selection.columns:
        primary_selection = selection.loc[selection["research_lane"].astype(str).eq("primary_daily")].copy()
    decisions = _read_csv(council / "council_decisions.csv")
    student = _read_csv(council / "student_training_readiness.csv")
    portfolio = _read_csv(council / "portfolio_critic.csv")
    selection_ready = bool(
        not primary_selection.empty
        and primary_selection.get("selection_status", pd.Series(dtype=str)).astype(str).eq("PASS").any()
    )
    council_ready = bool(
        not decisions.empty
        and decisions.get("status", pd.Series(dtype=str)).astype(str).eq("SHADOW_TEST").all()
        and decisions.get("action", pd.Series(dtype=str)).astype(str).ne("abstain").all()
    )
    supervised_student = student
    if not student.empty and "scope" in student.columns:
        supervised_student = student[
            student["scope"].astype(str).isin({"all_learning", "supervised_student"})
        ]
    student_ready = bool(
        not supervised_student.empty
        and not supervised_student.get("status", pd.Series(dtype=str)).astype(str).eq("BLOCKED").any()
    )
    portfolio_ready = bool(
        not portfolio.empty
        and portfolio.get("verdict", pd.Series(dtype=str)).astype(str).eq("pass").all()
    )
    portfolio_blockers = (
        _text_value(portfolio.iloc[0].get("blocker_codes", ""))
        if not portfolio.empty
        else "portfolio_evidence_missing"
    )
    definitions = [
        (
            "canonical_hyperliquid_authority",
            _truthy(authority.get("execution_allowed", False)),
            str(authority.get("status", "BLOCKED")),
            str(authority.get("blocker", "")),
            active / "hyperliquid_authority_state.csv",
            "follow the deterministic Hyperliquid research-cycle receipt",
        ),
        (
            "hyperliquid_walkforward",
            selection_ready,
            "ready" if selection_ready else "research_evidence",
            "" if selection_ready else "selection_controls_must_pass_before_promotion",
            selection_path,
            "refresh candidates or revise hypotheses, then rerun the existing five-fold tests without lowering gates",
        ),
        (
            "teacher_council",
            council_ready,
            "ready_for_shadow_test" if council_ready else "blocked",
            "" if council_ready else "teacher_council_not_clear",
            council / "council_decisions.csv",
            "resolve teacher blockers and critic vetoes",
        ),
        (
            "student_learning",
            student_ready,
            "ready_for_shadow_training" if student_ready else "research_only",
            "" if student_ready else "trade_level_outcomes_and_logged_propensities_required",
            root / "data" / "ml" / "student_teacher_training_dataset.csv",
            (
                "train and compare the seven supervised shadow specialists; keep the contextual bandit blocked"
                if student_ready
                else "collect trade-level shadow outcomes and logged behavior propensities"
            ),
        ),
        (
            "portfolio_critic",
            portfolio_ready,
            "ready" if portfolio_ready else "blocked",
            "" if portfolio_ready else portfolio_blockers or "portfolio_critic_not_clear",
            council / "portfolio_critic.csv",
            (
                "portfolio evidence is ready"
                if portfolio_ready
                else "resolve the recorded council-decision and testnet spot-to-perp margin blockers"
            ),
        ),
    ]
    rows = []
    for area, ready, status, blocker, evidence, next_action in definitions:
        rows.append(
            {
                "area": area,
                "ready": ready,
                "status": status,
                "blocker": blocker,
                "detail": f"run_id={run_id};candidate_set_id={candidate_set_id};execution_truth_mode=hyperliquid_testnet",
                **common,
                "setup_status": status,
                "setup_blocker": blocker,
                "evidence_path": evidence,
                "next_action": next_action,
            }
        )
    return rows


MARKET_VENUE_CONTEXT_COLUMNS = [
    "asset",
    "venue",
    "tradable",
    "volume_24h",
    "open_interest",
    "open_interest_usd",
    "funding_rate",
    "liquidity_usd",
    "transaction_count_24h",
    "market_cap",
    "source_timestamp",
    "source_system",
    "source_status",
    "source_role",
    "execution_authority",
    "promotion_allowed",
    "venue_lane",
    "liquidity_bucket",
    "funding_pulse_status",
    "blocker",
    "evidence_path",
    "notes",
]


def build_market_venue_context(root: Path = ROOT) -> CommandResult:
    active = root / "reports" / "active"
    liquidity_path = active / "multi_exchange_liquidity_test_2026-06-25.csv"
    liquidity = _read_csv(liquidity_path)
    source_coverage = _read_csv(active / "apify_mcp_source_coverage_2026-06-25.csv")
    hyperliquid_context = _read_csv(root / "data" / "processed" / "hyperliquid_market_context.csv")
    rows = _market_venue_rows_from_liquidity(liquidity, evidence_path=liquidity_path)
    rows.extend(_planned_source_rows(source_coverage))
    rows.extend(_market_venue_rows_from_hyperliquid_context(hyperliquid_context))
    frame = pd.DataFrame(rows, columns=MARKET_VENUE_CONTEXT_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["asset", "venue", "source_system"]).reset_index(drop=True)
    output = root / "data" / "processed" / "market_venue_context.csv"
    snapshot = root / "data" / "processed" / "market_venue_context_snapshots" / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M')}.csv"
    summary = active / "market_venue_context_summary.csv"
    summary_md = active / "market_venue_context_summary.md"
    lanes = active / "venue_lane_classification.csv"
    _write_csv(frame, output)
    _write_csv(frame, snapshot)
    lane_frame = _venue_lane_classification(frame)
    _write_csv(_market_venue_summary(frame), summary)
    _write_csv(lane_frame, lanes)
    _write_text(summary_md, _market_venue_context_markdown(frame, lane_frame))
    return CommandResult(
        paths={
            "market_venue_context": output,
            "snapshot": snapshot,
            "summary": summary,
            "summary_md": summary_md,
            "venue_lanes": lanes,
        },
        summary={
            "rows": len(frame),
            "assets": int(frame["asset"].nunique()) if not frame.empty else 0,
            "venues": int(frame["venue"].nunique()) if not frame.empty else 0,
            "promotion_allowed_rows": int(frame["promotion_allowed"].astype(bool).sum()) if not frame.empty else 0,
            "blocked_rows": int(frame["blocker"].astype(str).ne("").sum()) if not frame.empty else 0,
        },
    )


VENUE_LANE_TEST_COLUMNS = [
    "pair",
    "asset_x",
    "asset_y",
    "wizard_exchange",
    "asset_x_normalized",
    "asset_y_normalized",
    "normalized_pair",
    "pair_lane",
    "test_action",
    "test_status",
    "acceptance",
    "acceptance_reason",
    "local_sharpe",
    "local_profit_factor",
    "local_total_return",
    "local_max_drawdown",
    "local_trades",
    "local_closed_trades",
    "wizard_sharpe",
    "wizard_returns_total",
    "exact_mode",
    "source_system",
    "source_authority",
    "source_timestamp",
    "source_fresh",
    "hypothesis_status",
    "dydx_blockers",
    "hyperliquid_blockers",
    "funding_pulse_status",
    "next_step",
    "evidence_path",
]

MULTI_VENUE_HISTORY_READINESS_COLUMNS = [
    "rank",
    "pair",
    "wizard_exchange",
    "venue_type",
    "asset_x",
    "asset_y",
    "asset_x_normalized",
    "asset_y_normalized",
    "asset_x_base",
    "asset_y_base",
    "asset_x_quote",
    "asset_y_quote",
    "normalized_pair",
    "wizard_sharpe",
    "wizard_returns_total",
    "exact_mode",
    "setup_identity",
    "source_authority",
    "source_timestamp",
    "source_fresh",
    "discovery_screen_status",
    "research_quality_status",
    "research_blockers",
    "candidate_source_kind",
    "zscore_norm",
    "zscore_roll",
    "symbol_mapping_status",
    "history_source_status",
    "cost_model_status",
    "slippage_model_status",
    "funding_or_borrow_status",
    "readiness_status",
    "blockers",
    "next_step",
    "evidence_path",
]

VENUE_ROUTE_SCORECARD_COLUMNS = [
    "pair",
    "asset_x",
    "asset_y",
    "venue",
    "venue_rank",
    "both_legs_present",
    "both_legs_tradable",
    "execution_capable",
    "source_fresh",
    "source_timestamp",
    "volume_x_usd",
    "volume_y_usd",
    "pair_liquidity_usd",
    "open_interest_x_usd",
    "open_interest_y_usd",
    "pair_open_interest_usd",
    "history_status",
    "history_ready",
    "cost_model_status",
    "cost_model_ready",
    "slippage_model_status",
    "slippage_model_ready",
    "funding_or_borrow_status",
    "funding_or_borrow_ready",
    "venue_preflight_status",
    "venue_preflight_ready",
    "research_route_ready",
    "validation_route_ready",
    "paper_submission_ready",
    "route_score",
    "selection_status",
    "selected_for_research",
    "selected_for_execution",
    "selected_for_paper",
    "blockers",
    "next_step",
    "evidence_path",
]

VENUE_ROUTE_RECOMMENDATION_COLUMNS = [
    "pair",
    "asset_x",
    "asset_y",
    "recommended_research_venue",
    "recommended_execution_venue",
    "recommended_paper_venue",
    "research_route_score",
    "execution_route_score",
    "paper_route_score",
    "route_status",
    "selection_reason",
    "blockers",
    "next_step",
    "evidence_path",
]


def build_venue_lane_test_plan(root: Path = ROOT) -> CommandResult:
    active = root / "reports" / "active"
    lanes = _read_csv(active / "venue_lane_classification.csv")
    verification = _read_csv(active / "wizard_local_verification_batch.csv")
    queue = _read_csv(active / "crypto_wizards_next_best_sharpe_returns_queue.csv")
    wizard_evidence = _read_csv(root / "data" / "processed" / "wizard_evidence.csv")
    rows = _venue_lane_test_rows(lanes, verification, queue, wizard_evidence)
    frame = pd.DataFrame(rows, columns=VENUE_LANE_TEST_COLUMNS)
    if not frame.empty:
        frame["_source_fresh"] = frame.get("source_fresh", pd.Series(False, index=frame.index)).map(_boolish).astype(int)
        frame = frame.sort_values(["_source_fresh", "test_status", "pair_lane", "wizard_sharpe"], ascending=[False, True, True, False]).drop(columns=["_source_fresh"]).reset_index(drop=True)
    csv_path = active / "venue_lane_test_plan.csv"
    md_path = active / "venue_lane_test_plan.md"
    _write_csv(frame, csv_path)
    _write_text(md_path, _venue_lane_test_markdown(frame))
    return CommandResult(
        paths={"venue_lane_test_plan": csv_path, "venue_lane_test_plan_md": md_path},
        summary={
            "pairs": len(frame),
            "dydx_replayed": int(frame["test_status"].astype(str).eq("dydx_replayed").sum()) if not frame.empty else 0,
            "hyperliquid_build_needed": int(frame["test_status"].astype(str).eq("hyperliquid_history_needed").sum()) if not frame.empty else 0,
            "blocked": int(frame["test_status"].astype(str).str.contains("blocked", na=False).sum()) if not frame.empty else 0,
        },
    )


def build_multi_venue_history_readiness(root: Path = ROOT, top_n: int = 25) -> CommandResult:
    active = root / "reports" / "active"
    current_shortlist_path = active / "wizard_discovery_shortlist.csv"
    rows = _read_csv(current_shortlist_path)
    evidence_path = current_shortlist_path
    source_kind = "current_wizard_shortlist"
    if current_shortlist_path.exists():
        rows = _multi_venue_candidate_columns(rows)
    else:
        rows_path = active / "crypto_wizards_multi_venue_sharpe_rows_2026-06-25.csv"
        rows = _read_csv(rows_path)
        evidence_path = rows_path
        source_kind = "legacy_historical_rows"
        if rows.empty:
            fallback_path = active / "crypto_wizards_next_best_sharpe_returns_queue.csv"
            fallback = _read_csv(fallback_path)
            if not fallback.empty:
                rows = fallback.copy()
                evidence_path = fallback_path
                source_kind = "legacy_queue_fallback"
    readiness_rows = _multi_venue_history_readiness_rows(rows, evidence_path, top_n=top_n)
    frame = pd.DataFrame(readiness_rows, columns=MULTI_VENUE_HISTORY_READINESS_COLUMNS)
    if not frame.empty:
        frame["candidate_source_kind"] = source_kind
    csv_path = active / MULTI_VENUE_HISTORY_READINESS_FILENAME
    md_path = active / "multi_venue_history_readiness.md"
    _write_csv(frame, csv_path)
    _write_text(md_path, _multi_venue_history_readiness_markdown(frame))
    return CommandResult(
        paths={"multi_venue_history_readiness": csv_path, "multi_venue_history_readiness_md": md_path},
        summary={
            "rows": len(frame),
            "source_kind": source_kind,
            "ready_to_fetch": int(frame["readiness_status"].astype(str).eq("ready_to_fetch").sum()) if not frame.empty else 0,
            "ready_for_replay": int(frame["readiness_status"].astype(str).eq("ready_for_replay").sum()) if not frame.empty else 0,
            "blocked": int(frame["readiness_status"].astype(str).str.contains("blocked", na=False).sum()) if not frame.empty else 0,
        },
    )


def build_venue_route_scorecard(
    root: Path = ROOT,
    max_pairs: int = 100,
    *,
    as_of: pd.Timestamp | None = None,
) -> CommandResult:
    """Rank every supported venue for each pair without creating execution authority.

    A route needs more than listed markets. It must have fresh two-leg context,
    matching local history, a venue cost/slippage model, and funding or borrow
    evidence before it can be recommended for execution. Submission readiness is
    kept separate because an account may intentionally remain disabled.
    """
    active = root / "reports" / "active"
    reports = root / "reports"
    universe = _read_csv(root / "data" / "processed" / "pair_universe.csv")
    context = _read_csv(root / "data" / "processed" / "market_venue_context.csv")
    preflight = _read_csv(reports / "paper_venue_preflight.csv")
    lane_plan = _read_csv(active / "venue_lane_test_plan.csv")
    history_readiness = _read_csv(current_multi_venue_history_readiness_path(root))
    hyperliquid_bundle = _read_csv(active / "hyperliquid_research_bundle.csv")
    hyperliquid_cost_model = _read_csv(active / "hyperliquid_pair_cost_model.csv")
    now = pd.Timestamp(as_of or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")

    candidates = _venue_route_candidates(universe, max_pairs=max_pairs)
    context_index = _venue_route_context_index(context)
    preflight_index = _venue_route_preflight_index(preflight)
    lane_index = _venue_route_lane_index(lane_plan, history_readiness, hyperliquid_bundle)
    cost_index = _venue_route_cost_index(hyperliquid_cost_model)
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        pair = str(candidate.get("pair", "") or "")
        assets = _venue_route_assets(candidate)
        if not pair or assets is None:
            continue
        asset_x, asset_y = assets
        pair_key = _venue_route_pair_key(pair, asset_x, asset_y)
        venues = _venue_route_venues(candidate, asset_x, asset_y, context_index, preflight_index)
        if not venues:
            rows.append(
                _venue_route_row(
                    pair=pair,
                    asset_x=asset_x,
                    asset_y=asset_y,
                    venue="",
                    left=None,
                    right=None,
                    preflight_row=None,
                    cost_evidence=None,
                    history_status="missing_matching_history",
                    history_ready=False,
                    now=now,
                    funding_proxy_available=False,
                    evidence_paths=[str(root / "data" / "processed" / "market_venue_context.csv")],
                )
            )
            continue
        funding_proxy_available = _venue_route_funding_proxy_available(candidate)
        for venue in venues:
            left = context_index.get((asset_x, venue))
            right = context_index.get((asset_y, venue))
            preflight_row = preflight_index.get((pair_key, venue))
            cost_evidence = cost_index.get((pair_key, venue))
            history_status, history_ready, history_evidence = lane_index.get(
                (pair_key, venue),
                ("missing_matching_history", False, ""),
            )
            evidence_paths = [
                str((left or {}).get("evidence_path", "")),
                str((right or {}).get("evidence_path", "")),
                str((preflight_row or {}).get("evidence", "")),
                str((cost_evidence or {}).get("evidence_path", "")),
                str(root / "reports" / "active" / "venue_lane_test_plan.csv"),
                history_evidence,
            ]
            rows.append(
                _venue_route_row(
                    pair=pair,
                    asset_x=asset_x,
                    asset_y=asset_y,
                    venue=venue,
                    left=left,
                    right=right,
                    preflight_row=preflight_row,
                    cost_evidence=cost_evidence,
                    history_status=history_status,
                    history_ready=history_ready,
                    now=now,
                    funding_proxy_available=funding_proxy_available,
                    evidence_paths=evidence_paths,
                )
            )

    scorecard = pd.DataFrame(rows, columns=VENUE_ROUTE_SCORECARD_COLUMNS)
    if not scorecard.empty:
        scorecard = scorecard.sort_values(
            ["pair", "paper_submission_ready", "validation_route_ready", "research_route_ready", "route_score", "venue"],
            ascending=[True, False, False, False, False, True],
        ).reset_index(drop=True)
        scorecard["venue_rank"] = scorecard.groupby("pair").cumcount() + 1
    recommendations = _venue_route_recommendations(scorecard)
    if not scorecard.empty and not recommendations.empty:
        chosen = recommendations.set_index("pair")
        scorecard["selected_for_research"] = scorecard.apply(
            lambda row: row["venue"] == chosen.loc[row["pair"], "recommended_research_venue"] if row["pair"] in chosen.index else False,
            axis=1,
        )
        scorecard["selected_for_execution"] = scorecard.apply(
            lambda row: row["venue"] == chosen.loc[row["pair"], "recommended_execution_venue"] if row["pair"] in chosen.index else False,
            axis=1,
        )
        scorecard["selected_for_paper"] = scorecard.apply(
            lambda row: row["venue"] == chosen.loc[row["pair"], "recommended_paper_venue"] if row["pair"] in chosen.index else False,
            axis=1,
        )
    scorecard = scorecard.reindex(columns=VENUE_ROUTE_SCORECARD_COLUMNS, fill_value="")
    recommendations = recommendations.reindex(columns=VENUE_ROUTE_RECOMMENDATION_COLUMNS, fill_value="")

    scorecard_path = active / "venue_route_scorecard.csv"
    recommendations_path = active / "venue_route_recommendations.csv"
    markdown_path = active / "venue_route_recommendations.md"
    _write_csv(scorecard, scorecard_path)
    _write_csv(recommendations, recommendations_path)
    _write_text(markdown_path, _venue_route_recommendations_markdown(recommendations))
    return CommandResult(
        paths={
            "venue_route_scorecard": scorecard_path,
            "venue_route_recommendations": recommendations_path,
            "venue_route_recommendations_md": markdown_path,
        },
        summary={
            "pairs": int(len(recommendations)),
            "venue_rows": int(len(scorecard)),
            "research_routes": int(recommendations["recommended_research_venue"].astype(str).ne("").sum()) if not recommendations.empty else 0,
            "execution_routes": int(recommendations["recommended_execution_venue"].astype(str).ne("").sum()) if not recommendations.empty else 0,
            "paper_routes": int(recommendations["recommended_paper_venue"].astype(str).ne("").sum()) if not recommendations.empty else 0,
        },
    )


def _venue_route_candidates(universe: pd.DataFrame, *, max_pairs: int) -> list[dict[str, object]]:
    if universe.empty or "pair" not in universe.columns:
        return []
    frame = universe.copy()
    frame["_combined_score"] = pd.to_numeric(frame.get("combined_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    return frame.sort_values(["_combined_score", "pair"], ascending=[False, True]).head(max_pairs).to_dict("records")


def _venue_route_assets(candidate: dict[str, object]) -> tuple[str, str] | None:
    asset_x = _venue_route_asset_key(candidate.get("asset_x", ""))
    asset_y = _venue_route_asset_key(candidate.get("asset_y", ""))
    if asset_x and asset_y:
        return asset_x, asset_y
    parsed = _assets_from_pair(str(candidate.get("pair", "")))
    if parsed is None:
        return None
    return _venue_route_asset_key(parsed[0]), _venue_route_asset_key(parsed[1])


def _venue_route_asset_key(value: object) -> str:
    text = str(value or "").upper().replace("_", "-").replace("/", "-").strip()
    if not text or text in {"NAN", "NONE", "NULL"}:
        return ""
    for quote in ("-USDT", "-USDC", "-USD", "-BUSD", "-DAI"):
        if text.endswith(quote):
            return text[: -len(quote)]
    for quote in ("USDT", "USDC", "BUSD", "USD"):
        if text.endswith(quote) and len(text) > len(quote):
            return text[: -len(quote)]
    return text.split("-", 1)[0]


def _venue_route_pair_key(pair: object, asset_x: str = "", asset_y: str = "") -> str:
    assets = [asset_x, asset_y] if asset_x and asset_y else []
    if not assets:
        parsed = _assets_from_pair(str(pair or ""))
        assets = list(parsed or ())
    normalized = sorted(_venue_route_asset_key(asset) for asset in assets if _venue_route_asset_key(asset))
    return "|".join(normalized)


def _venue_route_name(value: object) -> str:
    text = str(value or "").lower().strip().replace("_", " ")
    aliases = {
        "coinglass hyperliquid": "hyperliquid",
        "hyperliquid testnet": "hyperliquid",
        "binance us": "binanceus",
        "binance.us": "binanceus",
    }
    return aliases.get(text, text.replace(" ", ""))


def _venue_route_bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def _venue_route_number(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if np.isfinite(numeric) else 0.0


def _venue_route_text(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _venue_route_context_index(context: pd.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    if context.empty or not {"asset", "venue"}.issubset(context.columns):
        return {}
    frame = context.copy()
    frame["_asset"] = frame["asset"].map(_venue_route_asset_key)
    frame["_venue"] = frame["venue"].map(_venue_route_name)
    frame["_timestamp"] = pd.to_datetime(frame.get("source_timestamp"), utc=True, errors="coerce")
    frame = frame[(frame["_asset"] != "") & (frame["_asset"] != "ALL") & (frame["_venue"] != "")]
    frame = frame.sort_values("_timestamp", na_position="first")
    return {
        (str(row["_asset"]), str(row["_venue"])): row.to_dict()
        for _, row in frame.drop_duplicates(["_asset", "_venue"], keep="last").iterrows()
    }


def _venue_route_preflight_index(preflight: pd.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    if preflight.empty or not {"pair", "venue"}.issubset(preflight.columns):
        return {}
    index: dict[tuple[str, str], dict[str, object]] = {}
    for _, row in preflight.iterrows():
        record = row.to_dict()
        key = (_venue_route_pair_key(record.get("pair", "")), _venue_route_name(record.get("venue", "")))
        if all(key):
            index[key] = record
    return index


def _venue_route_cost_index(cost_models: pd.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    if cost_models.empty or not {"pair", "venue"}.issubset(cost_models.columns):
        return {}
    index: dict[tuple[str, str], dict[str, object]] = {}
    for _, row in cost_models.iterrows():
        record = row.to_dict()
        key = (
            _venue_route_pair_key(record.get("pair", ""), record.get("asset_x", ""), record.get("asset_y", "")),
            _venue_route_name(record.get("venue", "")),
        )
        if all(key):
            index[key] = record
    return index


def _venue_route_lane_index(
    lane_plan: pd.DataFrame,
    readiness: pd.DataFrame,
    hyperliquid_bundle: pd.DataFrame | None = None,
) -> dict[tuple[str, str], tuple[str, bool, str]]:
    index: dict[tuple[str, str], tuple[str, bool, str]] = {}
    if not lane_plan.empty:
        for _, row in lane_plan.iterrows():
            record = row.to_dict()
            pair_key = _venue_route_pair_key(record.get("pair", ""), record.get("asset_x", ""), record.get("asset_y", ""))
            status = _venue_route_text(record.get("test_status", ""))
            inferred_venue = _venue_route_name(record.get("wizard_exchange", ""))
            if status.startswith("dydx_"):
                inferred_venue = "dydx"
            elif status.startswith("hyperliquid_"):
                inferred_venue = "hyperliquid"
            ready = any(token in status for token in ("replayed", "verified", "ready_for_replay"))
            if pair_key and inferred_venue:
                index[(pair_key, inferred_venue)] = (
                    status or "missing_matching_history",
                    ready,
                    _venue_route_text(record.get("evidence_path", "")),
                )
    if not readiness.empty:
        for _, row in readiness.iterrows():
            record = row.to_dict()
            pair_key = _venue_route_pair_key(record.get("pair", ""), record.get("asset_x", ""), record.get("asset_y", ""))
            venue = _venue_route_name(record.get("wizard_exchange", ""))
            status = _venue_route_text(record.get("readiness_status", ""))
            ready = status == "ready_for_replay"
            if pair_key and venue and (pair_key, venue) not in index:
                index[(pair_key, venue)] = (
                    status or "missing_matching_history",
                    ready,
                    _venue_route_text(record.get("evidence_path", "")),
                )
    if hyperliquid_bundle is not None and not hyperliquid_bundle.empty:
        for _, row in hyperliquid_bundle.iterrows():
            record = row.to_dict()
            if _venue_route_name(record.get("venue", "")) != "hyperliquid":
                continue
            pair_key = _venue_route_pair_key(record.get("pair", ""), record.get("asset_x", ""), record.get("asset_y", ""))
            ready = _venue_route_bool(record.get("history_ready"))
            status = _venue_route_text(record.get("history_status", ""))
            if pair_key and ready:
                index[(pair_key, "hyperliquid")] = (
                    status or "hyperliquid_local_history_ready",
                    True,
                    _venue_route_text(record.get("evidence_path", "")),
                )
    return index


def _venue_route_venues(
    candidate: dict[str, object],
    asset_x: str,
    asset_y: str,
    context: dict[tuple[str, str], dict[str, object]],
    preflight: dict[tuple[str, str], dict[str, object]],
) -> list[str]:
    pair_key = _venue_route_pair_key(candidate.get("pair", ""), asset_x, asset_y)
    venues = {
        venue
        for (asset, venue) in context
        if asset in {asset_x, asset_y}
    }
    venues.update(venue for (key, venue) in preflight if key == pair_key)
    for field in ("best_execution_venue", "exchange", "available_venues"):
        for venue in str(candidate.get(field, "") or "").split(";"):
            normalized = _venue_route_name(venue)
            if normalized:
                venues.add(normalized)
    return sorted(venues)


def _venue_route_fresh(record: dict[str, object] | None, now: pd.Timestamp) -> bool:
    if not record:
        return False
    captured = pd.to_datetime(record.get("source_timestamp"), utc=True, errors="coerce")
    return bool(pd.notna(captured) and now - captured <= pd.Timedelta(hours=24))


def _venue_route_funding_proxy_available(candidate: dict[str, object]) -> bool:
    value = candidate.get("funding_drag_bps")
    return bool(_venue_route_text(value) != "" and _venue_route_text(candidate.get("field_freshness")) == "current_snapshot")


def _venue_route_row(
    *,
    pair: str,
    asset_x: str,
    asset_y: str,
    venue: str,
    left: dict[str, object] | None,
    right: dict[str, object] | None,
    preflight_row: dict[str, object] | None,
    cost_evidence: dict[str, object] | None,
    history_status: str,
    history_ready: bool,
    now: pd.Timestamp,
    funding_proxy_available: bool,
    evidence_paths: list[str],
) -> dict[str, object]:
    left = left or {}
    right = right or {}
    both_legs_present = bool(left and right)
    both_legs_tradable = both_legs_present and _venue_route_bool(left.get("tradable")) and _venue_route_bool(right.get("tradable"))
    execution_capable = both_legs_present and _venue_route_bool(left.get("execution_authority")) and _venue_route_bool(right.get("execution_authority"))
    source_fresh = _venue_route_fresh(left, now) and _venue_route_fresh(right, now)
    volume_x = _venue_route_number(left.get("volume_24h"))
    volume_y = _venue_route_number(right.get("volume_24h"))
    oi_x = _venue_route_number(left.get("open_interest_usd"))
    oi_y = _venue_route_number(right.get("open_interest_usd"))
    pair_liquidity = min(volume_x, volume_y) if both_legs_present else 0.0
    pair_oi = min(oi_x, oi_y) if both_legs_present else 0.0
    preflight_cost_ready = _venue_route_bool((preflight_row or {}).get("cost_model_aligned"))
    preflight_cost_status = _venue_route_text((preflight_row or {}).get("cost_model_profile")) or "missing_pair_cost_model"
    evidence_cost_ready = _venue_route_bool((cost_evidence or {}).get("cost_model_ready"))
    evidence_cost_status = _venue_route_text((cost_evidence or {}).get("cost_model_status"))
    cost_ready = evidence_cost_ready or preflight_cost_ready
    cost_status = evidence_cost_status or preflight_cost_status
    evidence_slippage_ready = _venue_route_bool((cost_evidence or {}).get("slippage_model_ready"))
    evidence_slippage_status = _venue_route_text((cost_evidence or {}).get("slippage_model_status"))
    slippage_ready = evidence_slippage_ready if cost_evidence else bool(history_ready and cost_ready)
    slippage_status = evidence_slippage_status or (
        "costed_local_replay" if slippage_ready else "missing_venue_specific_slippage_calibration"
    )
    left_funding = _venue_route_text(left.get("funding_rate"))
    right_funding = _venue_route_text(right.get("funding_rate"))
    if left_funding and right_funding:
        funding_ready = True
        funding_status = "current_venue_funding_available"
    elif funding_proxy_available and venue == "dydx":
        funding_ready = True
        funding_status = "current_local_dydx_funding_drag_available"
    elif venue in {"binance", "binanceus", "coinbase"}:
        funding_ready = False
        funding_status = "missing_spot_borrow_cost"
    else:
        funding_ready = False
        funding_status = "missing_venue_funding_evidence"
    preflight_ready = _venue_route_bool((preflight_row or {}).get("execution_ready"))
    paper_ready = _venue_route_bool((preflight_row or {}).get("ready_for_submission"))
    research_ready = bool(both_legs_tradable and source_fresh and pair_liquidity > 0)
    validation_ready = bool(
        research_ready and execution_capable and history_ready and cost_ready and slippage_ready and funding_ready
    )
    paper_submission_ready = bool(validation_ready and paper_ready)
    blockers: list[str] = []
    if not both_legs_present:
        blockers.append("missing_two_leg_venue_context")
    elif not both_legs_tradable:
        blockers.append("one_or_both_legs_not_tradable")
    if both_legs_present and not source_fresh:
        blockers.append("stale_or_missing_venue_context")
    if both_legs_present and pair_liquidity <= 0:
        blockers.append("missing_two_leg_liquidity_evidence")
    if not execution_capable:
        blockers.append("venue_execution_capability_unconfirmed")
    if not history_ready:
        blockers.append("missing_matching_venue_history")
    if not cost_ready:
        blockers.append("missing_pair_cost_model")
    if not slippage_ready:
        blockers.append(slippage_status or "missing_venue_specific_slippage_calibration")
    if not funding_ready:
        blockers.append(funding_status)
    if not paper_ready:
        blockers.append("venue_paper_preflight_not_ready")
    score = 0.0
    score += 25.0 if both_legs_tradable else 0.0
    score += 10.0 if source_fresh else 0.0
    score += min(pair_liquidity / 1_000_000.0, 15.0)
    score += min(pair_oi / 5_000_000.0, 5.0)
    score += 15.0 if execution_capable else 0.0
    score += 15.0 if history_ready else 0.0
    score += 12.0 if cost_ready else 0.0
    score += 8.0 if slippage_ready else 0.0
    score += 10.0 if funding_ready else 0.0
    if paper_ready:
        score += 5.0
    score -= min(len(set(blockers)) * 2.0, 20.0)
    if paper_submission_ready:
        selection_status = "paper_ready"
        next_step = "paper_route_is_evidence_complete_but_requires_runtime_trade_approval"
    elif validation_ready:
        selection_status = "execution_ready"
        next_step = "run_named_testnet_pair_rehearsal_before_any_paper_submission"
    elif research_ready:
        selection_status = "research_ready"
        next_step = "collect_missing_history_cost_slippage_and_funding_evidence"
    else:
        selection_status = "blocked"
        next_step = "repair_two_leg_venue_listing_liquidity_or_freshness"
    timestamps = [
        _venue_route_text(left.get("source_timestamp")),
        _venue_route_text(right.get("source_timestamp")),
    ]
    return {
        "pair": pair,
        "asset_x": asset_x,
        "asset_y": asset_y,
        "venue": venue,
        "venue_rank": 0,
        "both_legs_present": both_legs_present,
        "both_legs_tradable": both_legs_tradable,
        "execution_capable": execution_capable,
        "source_fresh": source_fresh,
        "source_timestamp": ";".join(value for value in timestamps if value),
        "volume_x_usd": volume_x,
        "volume_y_usd": volume_y,
        "pair_liquidity_usd": pair_liquidity,
        "open_interest_x_usd": oi_x,
        "open_interest_y_usd": oi_y,
        "pair_open_interest_usd": pair_oi,
        "history_status": history_status,
        "history_ready": history_ready,
        "cost_model_status": cost_status,
        "cost_model_ready": cost_ready,
        "slippage_model_status": slippage_status,
        "slippage_model_ready": slippage_ready,
        "funding_or_borrow_status": funding_status,
        "funding_or_borrow_ready": funding_ready,
        "venue_preflight_status": "ready" if paper_ready else "blocked_or_missing",
        "venue_preflight_ready": preflight_ready,
        "research_route_ready": research_ready,
        "validation_route_ready": validation_ready,
        "paper_submission_ready": paper_submission_ready,
        "route_score": round(score, 3),
        "selection_status": selection_status,
        "selected_for_research": False,
        "selected_for_execution": False,
        "selected_for_paper": False,
        "blockers": ";".join(sorted(set(blockers))),
        "next_step": next_step,
        "evidence_path": ";".join(dict.fromkeys(path for path in evidence_paths if _venue_route_text(path))),
    }


def _venue_route_recommendations(scorecard: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if scorecard.empty:
        return pd.DataFrame(rows, columns=VENUE_ROUTE_RECOMMENDATION_COLUMNS)
    for pair, group in scorecard.groupby("pair", sort=True):
        ranked = group.sort_values(
            ["paper_submission_ready", "validation_route_ready", "research_route_ready", "route_score", "venue"],
            ascending=[False, False, False, False, True],
        )
        research = ranked[ranked["research_route_ready"]]
        execution = ranked[ranked["validation_route_ready"]]
        paper = ranked[ranked["paper_submission_ready"]]
        top = ranked.iloc[0]
        research_row = research.iloc[0] if not research.empty else None
        execution_row = execution.iloc[0] if not execution.empty else None
        paper_row = paper.iloc[0] if not paper.empty else None
        if paper_row is not None:
            status = "paper_route_ready"
            reason = "all_route_evidence_and_paper_preflight_checks_passed"
        elif execution_row is not None:
            status = "execution_route_ready"
            reason = "route_has_current_market_history_cost_slippage_and_funding_evidence"
        elif research_row is not None:
            status = "research_route_only"
            reason = "listed_and_liquid_but_execution_evidence_is_incomplete"
        else:
            status = "no_supported_route"
            reason = "no_venue_has_fresh_two_leg_liquidity_evidence"
        rows.append(
            {
                "pair": pair,
                "asset_x": top["asset_x"],
                "asset_y": top["asset_y"],
                "recommended_research_venue": research_row["venue"] if research_row is not None else "",
                "recommended_execution_venue": execution_row["venue"] if execution_row is not None else "",
                "recommended_paper_venue": paper_row["venue"] if paper_row is not None else "",
                "research_route_score": research_row["route_score"] if research_row is not None else 0.0,
                "execution_route_score": execution_row["route_score"] if execution_row is not None else 0.0,
                "paper_route_score": paper_row["route_score"] if paper_row is not None else 0.0,
                "route_status": status,
                "selection_reason": reason,
                "blockers": top["blockers"],
                "next_step": top["next_step"],
                "evidence_path": top["evidence_path"],
            }
        )
    return pd.DataFrame(rows, columns=VENUE_ROUTE_RECOMMENDATION_COLUMNS)


def _venue_route_recommendations_markdown(recommendations: pd.DataFrame) -> str:
    lines = [
        "# Venue Route Recommendations",
        "",
        "A venue is selected for execution only after two-leg listing, freshness, matching history, costs, slippage, and funding or borrow evidence all pass.",
        "Submission remains a separate runtime approval decision.",
        "",
    ]
    if recommendations.empty:
        lines.append("No pair-universe candidates were available for venue routing.")
    else:
        lines.extend(["", recommendations.to_string(index=False), ""])
    return "\n".join(lines)


def _multi_venue_candidate_columns(rows: pd.DataFrame) -> pd.DataFrame:
    """Adapt fresh Wizard shortlist rows to the venue-history readiness contract."""
    if rows.empty:
        return rows.copy()
    frame = rows.copy()
    exchange = frame.get("exchange", pd.Series("", index=frame.index)).map(_text_value)
    wizard_exchange = frame.get("wizard_exchange", pd.Series("", index=frame.index)).map(_text_value)
    frame["wizard_exchange"] = wizard_exchange.where(wizard_exchange.ne(""), exchange)
    if "zscore_norm" not in frame:
        frame["zscore_norm"] = frame.get("zscore_last", pd.Series("", index=frame.index))
    if "zscore_roll" not in frame:
        frame["zscore_roll"] = frame.get("zscore_roll_last", pd.Series("", index=frame.index))
    return frame


def _multi_venue_history_readiness_rows(rows: pd.DataFrame, evidence_path: Path, top_n: int = 25) -> list[dict[str, object]]:
    if rows.empty:
        return []
    frame = rows.copy()
    if "sharpe" not in frame:
        frame["sharpe"] = 0.0
    frame["sharpe"] = pd.to_numeric(frame["sharpe"], errors="coerce").fillna(0.0)
    if "wizard_exchange" not in frame:
        frame["wizard_exchange"] = "dydx"
    frame = frame.sort_values("sharpe", ascending=False).head(top_n).reset_index(drop=True)
    output: list[dict[str, object]] = []
    for idx, row in frame.iterrows():
        exchange = normalize_wizard_exchange(row.get("wizard_exchange"), default="dydx") or "unknown"
        asset_x = str(row.get("asset_x", "") or "").upper()
        asset_y = str(row.get("asset_y", "") or "").upper()
        x_symbol = normalize_wizard_symbol(row.get("asset_x_normalized") or asset_x, exchange)
        y_symbol = normalize_wizard_symbol(row.get("asset_y_normalized") or asset_y, exchange)
        asset_x_normalized = _text_value(row.get("asset_x_normalized")) or x_symbol.normalized_symbol or ""
        asset_y_normalized = _text_value(row.get("asset_y_normalized")) or y_symbol.normalized_symbol or ""
        normalized_pair = _text_value(row.get("normalized_pair")) or (
            f"{asset_x_normalized}-{asset_y_normalized}" if asset_x_normalized and asset_y_normalized else ""
        )
        mapping_status = _symbol_mapping_status(exchange, x_symbol, y_symbol)
        history_status = _history_source_status(exchange)
        cost_status = _cost_model_status(exchange, x_symbol, y_symbol)
        slippage_status = _slippage_model_status(exchange)
        funding_status = _funding_or_borrow_status(exchange, x_symbol, y_symbol)
        readiness, blockers, next_step = _history_readiness_decision(
            exchange,
            mapping_status,
            history_status,
            cost_status,
            slippage_status,
            funding_status,
        )
        output.append(
            {
                "rank": idx + 1,
                "pair": row.get("pair", "") or f"{asset_x}/{asset_y}",
                "wizard_exchange": exchange,
                "venue_type": _venue_type(exchange, x_symbol, y_symbol),
                "asset_x": asset_x,
                "asset_y": asset_y,
                "asset_x_normalized": asset_x_normalized,
                "asset_y_normalized": asset_y_normalized,
                "asset_x_base": x_symbol.base_asset or "",
                "asset_y_base": y_symbol.base_asset or "",
                "asset_x_quote": x_symbol.quote_asset or "",
                "asset_y_quote": y_symbol.quote_asset or "",
                "normalized_pair": normalized_pair,
                "wizard_sharpe": row.get("sharpe", ""),
                "wizard_returns_total": row.get("returns_total", ""),
                "exact_mode": row.get("exact_mode", ""),
                "setup_identity": row.get("setup_identity", ""),
                "source_authority": row.get("source_authority", ""),
                "source_timestamp": row.get("source_timestamp", ""),
                "source_fresh": row.get("source_fresh", ""),
                "discovery_screen_status": row.get("discovery_screen_status", ""),
                "research_quality_status": row.get("research_quality_status", ""),
                "research_blockers": row.get("research_blockers", ""),
                "candidate_source_kind": "",
                "zscore_norm": row.get("zscore_norm", ""),
                "zscore_roll": row.get("zscore_roll", ""),
                "symbol_mapping_status": mapping_status,
                "history_source_status": history_status,
                "cost_model_status": cost_status,
                "slippage_model_status": slippage_status,
                "funding_or_borrow_status": funding_status,
                "readiness_status": readiness,
                "blockers": ";".join(blockers),
                "next_step": next_step,
                "evidence_path": str(evidence_path),
            }
        )
    return output


def _symbol_mapping_status(exchange: str, x_symbol: object, y_symbol: object) -> str:
    if not getattr(x_symbol, "normalized_symbol", None) or not getattr(y_symbol, "normalized_symbol", None):
        return "blocked_needs_symbol_mapping"
    if exchange in {"binance", "binanceus", "bybit"} and (
        getattr(x_symbol, "symbol_format", "") == "unknown" or getattr(y_symbol, "symbol_format", "") == "unknown"
    ):
        return "blocked_needs_symbol_mapping"
    if exchange == "coinbase" and (
        getattr(x_symbol, "quote_asset", None) not in {"USD", "USDT", "USDC", "BTC", "ETH", "EUR", "GBP"}
        or getattr(y_symbol, "quote_asset", None) not in {"USD", "USDT", "USDC", "BTC", "ETH", "EUR", "GBP"}
    ):
        return "blocked_needs_symbol_mapping"
    return "mapped"


def _history_source_status(exchange: str) -> str:
    if exchange in {"binance", "binanceus", "coinbase"}:
        return "ready_public_candles"
    if exchange == "bybit":
        return "ready_public_or_apify_candles"
    if exchange == "dydx":
        return "ready_dydx_or_apify_candles"
    return "blocked_no_history_source"


def _cost_model_status(exchange: str, x_symbol: object, y_symbol: object) -> str:
    if exchange == "dydx":
        return "available_dydx_cost_model"
    if _venue_type(exchange, x_symbol, y_symbol) == "spot":
        return "needs_spot_fee_model"
    if _venue_type(exchange, x_symbol, y_symbol) == "perp_or_mixed":
        return "needs_perp_fee_and_funding_model"
    return "needs_cost_model"


def _slippage_model_status(exchange: str) -> str:
    if exchange == "dydx":
        return "available_or_existing_dydx_slippage_guard"
    return "needs_orderbook_or_volume_slippage_model"


def _funding_or_borrow_status(exchange: str, x_symbol: object, y_symbol: object) -> str:
    venue_type = _venue_type(exchange, x_symbol, y_symbol)
    if exchange == "dydx":
        return "available_dydx_funding"
    if venue_type == "perp_or_mixed":
        return "needs_funding_rates"
    if venue_type == "spot":
        return "borrow_short_cost_tracked_not_research_gate"
    return "funding_or_borrow_cost_tracked_not_research_gate"


def _history_readiness_decision(
    exchange: str,
    mapping_status: str,
    history_status: str,
    cost_status: str,
    slippage_status: str,
    funding_status: str,
) -> tuple[str, list[str], str]:
    blockers = []
    if mapping_status != "mapped":
        blockers.append(mapping_status)
    if history_status.startswith("blocked"):
        blockers.append(history_status)
    if blockers:
        return "blocked_needs_mapping_or_source", blockers, "repair_symbol_mapping_or_add_history_source"
    if exchange == "dydx" and cost_status.startswith("available") and funding_status.startswith("available"):
        return "ready_for_replay", blockers, "run_dydx_exact_mode_replay"
    soft_blockers = [cost_status, slippage_status, funding_status]
    return "ready_to_fetch", soft_blockers, f"fetch_{exchange}_candles_then_track_cost_slippage_funding_or_borrow_assumptions"


def _venue_type(exchange: str, x_symbol: object, y_symbol: object) -> str:
    quotes = {getattr(x_symbol, "quote_asset", None), getattr(y_symbol, "quote_asset", None)}
    normalized = {
        str(getattr(x_symbol, "normalized_symbol", "") or "").upper(),
        str(getattr(y_symbol, "normalized_symbol", "") or "").upper(),
    }
    if "PERP" in quotes or any(symbol.endswith("-PERP") for symbol in normalized):
        return "perp_or_mixed"
    if exchange in {"binance", "binanceus", "coinbase"}:
        return "spot"
    if exchange == "bybit":
        return "perp_or_mixed" if "PERP" in "".join(normalized) else "spot_or_perp_unknown"
    if exchange == "dydx":
        return "perp"
    return "unknown"


def _text_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _match_row(frame: pd.DataFrame, value: str, column: str) -> pd.Series | None:
    if frame.empty or column not in frame.columns:
        return None
    matches = frame[frame[column].astype(str) == str(value)]
    if matches.empty:
        return None
    return matches.iloc[0]


def _multi_venue_history_readiness_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Multi-Venue History Readiness\n\nNo multi-venue Wizard rows were available.\n"
    counts = frame["readiness_status"].value_counts().reset_index()
    counts.columns = ["readiness_status", "rows"]
    view_cols = [
        "rank",
        "wizard_exchange",
        "asset_x",
        "asset_y",
        "wizard_sharpe",
        "readiness_status",
        "blockers",
        "next_step",
    ]
    return "\n".join(
        [
            "# Multi-Venue History Readiness",
            "",
            "This report ranks the current Crypto Wizards multi-venue candidates and decides whether each can move to candle fetching.",
            "",
            "## Status Counts",
            "",
            counts.to_markdown(index=False),
            "",
            "## Top Candidates",
            "",
            frame[view_cols].to_markdown(index=False),
            "",
            "## Rules",
            "",
            "- `ready_to_fetch` means symbol mapping and a plausible candle source are available, but local replay still needs venue-specific cost, slippage, and funding/borrow assumptions.",
            "- `ready_for_replay` is only allowed when history, costs, and funding assumptions already exist.",
            "- Non-dYdX venues remain research-only until local replay exists.",
            "",
        ]
    )


def _venue_lane_test_rows(
    lanes: pd.DataFrame,
    verification: pd.DataFrame,
    queue: pd.DataFrame,
    wizard_evidence: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    lane_map = {
        str(row.get("asset", "")).upper(): row.to_dict()
        for _, row in lanes.iterrows()
        if str(row.get("asset", "") or "").strip()
    }
    candidates = _venue_lane_candidates(verification, queue, wizard_evidence)
    rows = []
    for candidate in candidates:
        asset_x = str(candidate.get("asset_x", "") or "").upper()
        asset_y = str(candidate.get("asset_y", "") or "").upper()
        wizard_exchange = _candidate_wizard_exchange(candidate)
        x_symbol = normalize_wizard_symbol(asset_x, wizard_exchange)
        y_symbol = normalize_wizard_symbol(asset_y, wizard_exchange)
        asset_x_normalized = candidate.get("asset_x_normalized", "") or x_symbol.normalized_symbol or ""
        asset_y_normalized = candidate.get("asset_y_normalized", "") or y_symbol.normalized_symbol or ""
        normalized_pair = candidate.get("normalized_pair", "") or (
            f"{asset_x_normalized}-{asset_y_normalized}" if asset_x_normalized and asset_y_normalized else ""
        )
        left = lane_map.get(asset_x, {})
        right = lane_map.get(asset_y, {})
        pair_lane = _pair_lane(left, right, candidate)
        test_status, test_action, next_step = _pair_lane_test_status(pair_lane, candidate)
        rows.append(
            {
                "pair": candidate.get("pair", ""),
                "asset_x": asset_x,
                "asset_y": asset_y,
                "wizard_exchange": wizard_exchange,
                "asset_x_normalized": asset_x_normalized,
                "asset_y_normalized": asset_y_normalized,
                "normalized_pair": normalized_pair,
                "pair_lane": pair_lane,
                "test_action": test_action,
                "test_status": test_status,
                "acceptance": candidate.get("acceptance", ""),
                "acceptance_reason": candidate.get("acceptance_reason", ""),
                "local_sharpe": candidate.get("local_sharpe", ""),
                "local_profit_factor": candidate.get("local_profit_factor", ""),
                "local_total_return": candidate.get("local_total_return", ""),
                "local_max_drawdown": candidate.get("local_max_drawdown", ""),
                "local_trades": candidate.get("local_trades", ""),
                "local_closed_trades": candidate.get("local_closed_trades", ""),
                "wizard_sharpe": candidate.get("wizard_sharpe", ""),
                "wizard_returns_total": candidate.get("wizard_returns_total", ""),
                "exact_mode": candidate.get("exact_mode", ""),
                "source_system": candidate.get("source_system", ""),
                "source_authority": candidate.get("source_authority", ""),
                "source_timestamp": candidate.get("source_timestamp", ""),
                "source_fresh": candidate.get("source_fresh", ""),
                "hypothesis_status": candidate.get("hypothesis_status", ""),
                "dydx_blockers": _asset_blockers(left, right, "dydx"),
                "hyperliquid_blockers": _asset_blockers(left, right, "hyperliquid"),
                "funding_pulse_status": "needs_api_key",
                "next_step": next_step,
                "evidence_path": _candidate_evidence_path(candidate),
            }
        )
    return rows


def _venue_lane_candidates(
    verification: pd.DataFrame,
    queue: pd.DataFrame,
    wizard_evidence: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    if not verification.empty:
        for _, row in verification.iterrows():
            candidate = row.to_dict()
            key = _candidate_dedupe_key(candidate)
            if key and key not in seen:
                seen.add(key)
                candidates.append(candidate)
    if not queue.empty:
        for _, row in queue.iterrows():
            candidate = row.to_dict()
            key = _candidate_dedupe_key(candidate)
            if key and key not in seen:
                seen.add(key)
                candidates.append(candidate)
    if wizard_evidence is None or wizard_evidence.empty:
        return candidates

    evidence = wizard_evidence.copy()
    if "mode_valid" in evidence.columns:
        evidence = evidence[evidence["mode_valid"].map(_boolish)].copy()
    if "source_fresh" in evidence.columns:
        evidence = evidence[evidence["source_fresh"].map(_boolish)].copy()
    if "passes_sharpe_gate" in evidence.columns:
        sharpe_gate = evidence["passes_sharpe_gate"].map(_boolish)
    else:
        sharpe_gate = pd.to_numeric(evidence.get("sharpe", pd.Series(dtype=float)), errors="coerce").ge(1.75)
    returns = pd.to_numeric(evidence.get("returns_total", pd.Series(dtype=float)), errors="coerce")
    minimum_returns = pd.to_numeric(
        evidence.get("discovery_min_returns_total", pd.Series(0.10, index=evidence.index)),
        errors="coerce",
    ).fillna(0.10)
    returns_gate = returns.ge(minimum_returns)
    evidence = evidence[sharpe_gate & returns_gate].copy()
    for _, row in evidence.iterrows():
        candidate = row.to_dict()
        candidate["wizard_exchange"] = _text_value(candidate.get("exchange")) or "dydx"
        candidate["wizard_sharpe"] = candidate.get("sharpe", "")
        candidate["wizard_returns_total"] = candidate.get("returns_total", "")
        key = _candidate_dedupe_key(candidate)
        if key and key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return candidates


def _candidate_dedupe_key(candidate: dict[str, object]) -> str:
    exchange = _candidate_wizard_exchange(candidate) or "unknown"
    pair = str(candidate.get("normalized_pair", "") or candidate.get("pair", "") or "").upper()
    if not pair:
        asset_x = str(candidate.get("asset_x", "") or "").upper()
        asset_y = str(candidate.get("asset_y", "") or "").upper()
        pair = f"{asset_x}/{asset_y}"
    exact_mode = _text_value(candidate.get("exact_mode")).lower()
    return f"{exchange}:{pair}:{exact_mode}"


def _pair_lane(left: dict[str, object], right: dict[str, object], candidate: dict[str, object] | None = None) -> str:
    candidate = candidate or {}
    candidate_exchange = _candidate_wizard_exchange(candidate)
    if candidate_exchange and candidate_exchange != "dydx":
        return wizard_exchange_lane(candidate_exchange)
    verification_status = str(candidate.get("verification_status", "") or "")
    execution_bucket = str(candidate.get("execution_bucket", "") or "")
    execution_blockers = str(candidate.get("execution_blockers", "") or "")
    if verification_status == "verified":
        return "dydx_local_replayed_unclassified"
    if "REJECT_MISSING_MARKET" in execution_bucket or "missing_from_apify_dydx" in execution_blockers:
        return "blocked_missing_market"
    if "REJECT_EXECUTION_NOW" in execution_bucket:
        return "blocked_execution_now"
    lanes = {str(left.get("best_lane", "")), str(right.get("best_lane", ""))}
    if "blocked_liquidity" in lanes:
        if "hyperliquid_research_candidate" in lanes or "hyperliquid_watch" in lanes:
            return "mixed_blocked_and_hyperliquid_research"
        return "blocked_liquidity"
    if lanes and all(lane == "dydx_execution_candidate" for lane in lanes):
        return "dydx_exact_mode_replay"
    if any(lane == "dydx_execution_watch" for lane in lanes) and all(
        lane in {"dydx_execution_candidate", "dydx_execution_watch"} for lane in lanes
    ):
        return "dydx_size_limited_replay"
    if any(lane == "hyperliquid_research_candidate" for lane in lanes):
        return "hyperliquid_research_lane"
    if any(lane == "hyperliquid_watch" for lane in lanes):
        return "hyperliquid_watch_lane"
    if any(lane == "dydx_research_only" for lane in lanes):
        return "dydx_research_only"
    return "needs_more_data"


def _pair_lane_test_status(pair_lane: str, candidate: dict[str, object]) -> tuple[str, str, str]:
    verification_status = str(candidate.get("verification_status", "") or "")
    acceptance = str(candidate.get("acceptance", "") or "")
    if verification_status == "verified":
        return "dydx_replayed", "review_local_acceptance", "promote_only_if_local_acceptance_passes" if acceptance == "ACCEPT" else "keep_or_reject_from_local_replay"
    if pair_lane in {"dydx_exact_mode_replay", "dydx_size_limited_replay"}:
        if verification_status == "verified":
            return "dydx_replayed", "review_local_acceptance", "promote_only_if_local_acceptance_passes" if acceptance == "ACCEPT" else "keep_or_reject_from_local_replay"
        return "dydx_replay_blocked", "run_or_repair_dydx_exact_mode_replay", "collect_missing_dydx_history_or_exact_mode_capture"
    if pair_lane in {"blocked_missing_market", "blocked_execution_now"}:
        return "blocked_execution", "do_not_run_acceptance", "refresh_market_context_or_find_alternate_venue"
    if pair_lane == "hyperliquid_research_lane":
        return "hyperliquid_history_needed", "build_hyperliquid_history_and_cost_model", "do_not_promote_until_hyperliquid_local_replay_exists"
    if pair_lane == "hyperliquid_watch_lane":
        return "hyperliquid_liquidity_watch", "collect_more_hyperliquid_depth_and_slippage_evidence", "watch_until_depth_and_replay_are_ready"
    if pair_lane == "mixed_blocked_and_hyperliquid_research":
        return "hyperliquid_history_needed", "route_to_hyperliquid_research_not_dydx", "build_hyperliquid_history_and_cost_model"
    if pair_lane in {"binance_research_lane", "binanceus_research_lane", "bybit_research_lane", "coinbase_research_lane"}:
        venue = pair_lane.replace("_research_lane", "")
        return (
            f"{venue}_research_only",
            f"build_{venue}_history_cost_and_symbol_mapping",
            f"do_not_promote_until_{venue}_local_replay_exists",
        )
    if pair_lane in {"forex_out_of_scope_lane", "stocks_out_of_scope_lane"}:
        return "out_of_crypto_scope", "do_not_run_active_crypto_acceptance", "exclude_unless_project_scope_changes"
    if pair_lane == "blocked_liquidity":
        return "blocked_liquidity", "do_not_run_acceptance", "wait_for_new_venue_or_liquidity_source"
    return "needs_more_data", "collect_missing_lane_evidence", "rerun_after_market_venue_context_refresh"


def _candidate_wizard_exchange(candidate: dict[str, object]) -> str | None:
    for key in ("wizard_exchange", "scanner_exchange", "exchange"):
        value = candidate.get(key)
        if value is not None and not pd.isna(value) and str(value).strip():
            return normalize_wizard_exchange(value, default="dydx")
    return "dydx"


def _asset_blockers(left: dict[str, object], right: dict[str, object], venue: str) -> str:
    values = []
    for row in [left, right]:
        text = str(row.get("blockers", "") or "")
        for blocker in text.split(";"):
            if blocker and (venue in blocker or (venue == "dydx" and blocker.startswith("thin_"))):
                values.append(blocker)
    return ";".join(sorted(dict.fromkeys(values)))


def _candidate_evidence_path(candidate: dict[str, object]) -> str:
    paths = [
        str(candidate.get("summary_path", "") or ""),
        str(candidate.get("history_path", "") or ""),
        str(candidate.get("source_path", "") or ""),
        str(candidate.get("evidence_path", "") or ""),
    ]
    return ";".join(path for path in paths if path)


def _venue_lane_test_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Venue Lane Test Plan\n\nNo candidate rows were available.\n"
    counts = frame["test_status"].value_counts().reset_index()
    counts.columns = ["test_status", "rows"]
    view_cols = [
        "pair",
        "pair_lane",
        "test_status",
        "acceptance",
        "acceptance_reason",
        "next_step",
    ]
    parts = [
        "# Venue Lane Test Plan",
        "",
        "This report reroutes the current Wizard/local candidates through the venue-aware context layer.",
        "",
        "## Status Counts",
        "",
        counts.to_markdown(index=False),
        "",
        "## Candidate Actions",
        "",
        frame[view_cols].to_markdown(index=False),
        "",
        "## Rules",
        "",
        "- dYdX candidates can only move forward after exact-mode local replay.",
        "- Hyperliquid candidates are research-only until Hyperliquid history and cost replay exist.",
        "- Binance, Binance US, ByBit, and Coinbase Wizard candidates are research-only until venue-specific history, symbol mapping, costs, slippage, and funding/borrow assumptions exist.",
        "- Funding Pulse remains `needs_api_key` and cannot promote trades.",
        "- Context-only sources route research but do not authorize execution.",
        "",
    ]
    return "\n".join(parts)


def _market_venue_rows_from_liquidity(frame: pd.DataFrame, *, evidence_path: Path) -> list[dict[str, object]]:
    if frame.empty:
        return []
    source_timestamp = _historical_snapshot_timestamp(evidence_path)
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        asset = str(row.get("asset", "") or "").upper().strip()
        venue = str(row.get("venue", "") or "").strip()
        if not asset or not venue:
            continue
        execution_decision = str(row.get("execution_decision", "") or "")
        source = str(row.get("source", "") or "")
        source_system = _source_system_from_market_source(source, venue)
        source_role = _market_source_role(source_system, venue)
        execution_authority = _market_execution_authority(source_system, venue)
        lane = _venue_lane(asset, venue, execution_decision, source_system)
        blocker = _market_context_blocker(venue, execution_decision, source_system, row)
        promotion_allowed = bool(execution_authority and not blocker and venue.lower() == "dydx")
        rows.append(
            {
                "asset": asset,
                "venue": venue,
                "tradable": _truthy(row.get("seen_or_tradable")),
                "volume_24h": _clean_numeric(row.get("reported_24h_volume")),
                "open_interest": _clean_numeric(row.get("reported_open_interest_native")),
                "open_interest_usd": _clean_numeric(row.get("reported_open_interest_usd")),
                "funding_rate": _clean_numeric(row.get("funding_rate")),
                "liquidity_usd": "",
                "transaction_count_24h": "",
                "market_cap": "",
                "source_timestamp": source_timestamp,
                "source_system": source_system,
                "source_status": "historical_snapshot",
                "source_role": source_role,
                "execution_authority": execution_authority,
                "promotion_allowed": promotion_allowed,
                "venue_lane": lane,
                "liquidity_bucket": row.get("liquidity_bucket", ""),
                "funding_pulse_status": "needs_api_key",
                "blocker": blocker,
                "evidence_path": _market_context_evidence_path(source),
                "notes": row.get("notes", ""),
            }
        )
    return rows


def _market_venue_rows_from_hyperliquid_context(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Use an already captured public snapshot without making a network request."""
    if frame.empty:
        return []
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        record = row.to_dict()
        asset = _venue_route_asset_key(record.get("asset", ""))
        if not asset:
            continue
        rows.append(
            {
                "asset": asset,
                "venue": "hyperliquid",
                "tradable": _venue_route_bool(record.get("tradable")),
                "volume_24h": _clean_numeric(record.get("volume_24h")),
                "open_interest": _clean_numeric(record.get("open_interest")),
                "open_interest_usd": _clean_numeric(record.get("open_interest_usd")),
                "funding_rate": _clean_numeric(record.get("funding_rate")),
                "liquidity_usd": _clean_numeric(record.get("liquidity_usd")),
                "transaction_count_24h": _clean_numeric(record.get("transaction_count_24h")),
                "market_cap": _clean_numeric(record.get("market_cap")),
                "source_timestamp": _text_value(record.get("source_timestamp", "")),
                "source_system": _text_value(record.get("source_system", "")) or "hyperliquid_public_api",
                "source_status": _text_value(record.get("source_status", "")) or "captured",
                "source_role": _text_value(record.get("source_role", "")) or "market_data_funding_context",
                "execution_authority": _venue_route_bool(record.get("execution_authority")),
                "promotion_allowed": False,
                "venue_lane": _text_value(record.get("venue_lane", "")) or "hyperliquid_research_candidate",
                "liquidity_bucket": _text_value(record.get("liquidity_bucket", "")),
                "funding_pulse_status": _text_value(record.get("funding_pulse_status", "")),
                "blocker": _text_value(record.get("blocker", "")) or "requires_pair_history_cost_slippage_and_preflight",
                "evidence_path": _text_value(record.get("evidence_path", "")),
                "notes": _text_value(record.get("notes", "")),
            }
        )
    return rows


def _historical_snapshot_timestamp(path: Path) -> str:
    match = re.search(r"(20\d{2}-\d{2}-\d{2})(?:[_-](\d{4,6}))?", path.name)
    if not match:
        return ""
    date_part, time_part = match.groups()
    if not time_part:
        return f"{date_part}T00:00:00+00:00"
    padded = time_part.ljust(6, "0")
    return f"{date_part}T{padded[:2]}:{padded[2:4]}:{padded[4:6]}+00:00"


def _planned_source_rows(source_coverage: pd.DataFrame) -> list[dict[str, object]]:
    if source_coverage.empty or "source_id" not in source_coverage:
        return [_funding_pulse_placeholder()]
    rows: list[dict[str, object]] = []
    funding_rows = source_coverage[source_coverage["source_id"].astype(str) == "fraktalapi/funding-pulse"]
    if funding_rows.empty:
        rows.append(_funding_pulse_placeholder())
    else:
        source = funding_rows.iloc[0]
        rows.append(
            {
                "asset": "ALL",
                "venue": "cross_exchange",
                "tradable": False,
                "volume_24h": "",
                "open_interest": "",
                "open_interest_usd": "",
                "funding_rate": "",
                "liquidity_usd": "",
                "transaction_count_24h": "",
                "market_cap": "",
                "source_timestamp": _now(),
                "source_system": "funding_pulse",
                "source_status": "planned_needs_api_key",
                "source_role": "funding_crowding_risk_layer",
                "execution_authority": False,
                "promotion_allowed": False,
                "venue_lane": "planned_cross_exchange_risk_layer",
                "liquidity_bucket": "",
                "funding_pulse_status": "needs_api_key",
                "blocker": "funding_pulse_needs_api_key",
                "evidence_path": str(source.get("evidence", "reports/active/apify_mcp_source_coverage_2026-06-25.csv")),
                "notes": "Funding Pulse remains planned; pipeline must continue without using it for promotion.",
            }
        )
    for _, row in source_coverage.iterrows():
        source_id = str(row.get("source_id", "")).strip()
        if not source_id:
            continue
        if source_id == "fraktalapi/funding-pulse":
            continue
        sample_status = str(row.get("sample_status", "not_sampled")).strip()
        venue = infer_apify_venue(source_id)
        source_system = _source_system_from_market_source(source_id, venue)
        execution_authority = source_system in {"dydx", "hyperliquid"} and sample_status == "sampled"
        lane = _venue_lane(
            "ALL",
            venue,
            "dydx_execution_ok" if execution_authority else "research_only",
            source_system,
        )
        blocker = (
            "needs_api_key"
            if sample_status == "needs_api_key"
            else "context_only_not_promotion_authority"
            if source_system in {"funding_pulse", "coinglass", "gmx", "dexscreener"}
            else ("missing_hyperliquid_local_replay" if source_system == "hyperliquid" else "")
        )
        rows.append(
            {
                "asset": "ALL",
                "venue": venue,
                "tradable": False,
                "volume_24h": "",
                "open_interest": "",
                "open_interest_usd": "",
                "funding_rate": "",
                "liquidity_usd": "",
                "transaction_count_24h": "",
                "market_cap": "",
                "source_timestamp": _now(),
                "source_system": source_system,
                "source_status": sample_status,
                "source_role": _market_source_role(source_system, venue),
                "execution_authority": execution_authority,
                "promotion_allowed": bool(execution_authority and not blocker),
                "venue_lane": lane,
                "liquidity_bucket": "",
                "funding_pulse_status": "needs_api_key" if source_system == "funding_pulse" else "",
                "blocker": blocker,
                "evidence_path": str(row.get("evidence", "reports/active/apify_mcp_source_coverage_2026-06-25.csv")),
                "notes": str(row.get("limitations", "")),
            }
        )
    return rows


def _funding_pulse_placeholder() -> dict[str, object]:
    return {
        "asset": "ALL",
        "venue": "cross_exchange",
        "tradable": False,
        "volume_24h": "",
        "open_interest": "",
        "open_interest_usd": "",
        "funding_rate": "",
        "liquidity_usd": "",
        "transaction_count_24h": "",
        "market_cap": "",
        "source_timestamp": _now(),
        "source_system": "funding_pulse",
        "source_status": "planned_needs_api_key",
        "source_role": "funding_crowding_risk_layer",
        "execution_authority": False,
        "promotion_allowed": False,
        "venue_lane": "planned_cross_exchange_risk_layer",
        "liquidity_bucket": "",
        "funding_pulse_status": "needs_api_key",
        "blocker": "funding_pulse_needs_api_key",
        "evidence_path": "reports/active/apify_mcp_source_coverage_2026-06-25.csv",
        "notes": "Funding Pulse remains planned; pipeline must continue without using it for promotion.",
    }


def _source_system_from_market_source(source: str, venue: str) -> str:
    text = f"{source} {venue}".lower()
    if "funding-pulse" in text or "funding pulse" in text:
        return "funding_pulse"
    if "hyperliquid" in text:
        return "hyperliquid"
    if "coinglass" in text:
        return "coinglass"
    if "dexscreener" in text:
        return "dexscreener"
    if "gmx" in text:
        return "gmx"
    if "dydx" in text:
        return "dydx"
    return "apify"


def _market_source_role(source_system: str, venue: str) -> str:
    if source_system in {"dydx", "hyperliquid"}:
        return "perp_market_snapshot"
    if source_system == "coinglass":
        return "cross_exchange_context"
    if source_system == "dexscreener":
        return "dex_liquidity_discovery"
    if source_system == "gmx":
        return "defi_derivatives_or_oracle_context"
    if source_system == "funding_pulse":
        return "funding_crowding_risk_layer"
    if venue.lower() in {"coinglass dydx", "coinglass hyperliquid"}:
        return "cross_exchange_context"
    return "supplemental_context"


def _market_execution_authority(source_system: str, venue: str) -> bool:
    return source_system in {"dydx", "hyperliquid"} and venue.lower() in {"dydx", "hyperliquid"}


def _venue_lane(asset: str, venue: str, execution_decision: str, source_system: str) -> str:
    decision = execution_decision.lower()
    venue_lower = venue.lower()
    if source_system == "coinglass":
        if "hyperliquid" in venue_lower:
            return "hyperliquid_research_candidate"
        if "dydx" in venue_lower:
            return "dydx_context"
        return f"{venue_lower.replace(' ', '_')}_context"
    if source_system == "gmx":
        return "gmx_supplemental_context"
    if venue_lower == "dydx":
        if decision == "dydx_execution_ok":
            return "dydx_execution_candidate"
        if decision == "dydx_execution_watch":
            return "dydx_execution_watch"
        if "research_only" in decision:
            return "dydx_research_only"
        return "blocked_liquidity"
    if "hyperliquid" in venue_lower:
        if "watch" in decision:
            return "hyperliquid_watch"
        return "hyperliquid_research_candidate"
    return "supplemental_context"


def _market_context_blocker(venue: str, execution_decision: str, source_system: str, row: pd.Series) -> str:
    decision = execution_decision.lower()
    venue_lower = venue.lower()
    blockers: list[str] = []
    if source_system == "coinglass":
        blockers.append("context_only_not_promotion_authority")
    if source_system == "gmx":
        blockers.append("supplemental_only_not_promotion_authority")
        if not _clean_numeric(row.get("reported_24h_volume")) and not _clean_numeric(row.get("reported_open_interest_usd")):
            blockers.append("missing_gmx_volume_or_open_interest")
    if venue_lower == "dydx":
        if decision == "dydx_execution_watch":
            blockers.append("limited_dydx_liquidity_requires_size_slippage_check")
        elif "research_only" in decision:
            blockers.append("thin_dydx_liquidity")
        elif "blocked" in decision:
            blockers.append("dydx_blocked_or_missing_market")
    if "hyperliquid" in venue_lower:
        blockers.append("missing_hyperliquid_local_replay")
        if "watch" in decision:
            blockers.append("hyperliquid_liquidity_watch")
    if source_system == "dexscreener":
        blockers.append("dex_discovery_only_requires_activity_filters")
    return ";".join(dict.fromkeys(blockers))


def _market_context_evidence_path(source: str) -> str:
    text = str(source or "")
    if "eTJs7sXrurA0fywZN" in text or "dydx-markets-scraper" in text:
        return "reports/active/multi_exchange_liquidity_test_2026-06-25.csv"
    if "2sv7gycjD9jbRD8VX" in text or "hyperliquid-perp-funding-scraper" in text:
        return "reports/active/multi_exchange_liquidity_test_2026-06-25.csv"
    if "coinglass" in text.lower():
        return "reports/active/multi_exchange_liquidity_test_2026-06-25.csv"
    if "gmx" in text.lower():
        return "reports/active/multi_exchange_liquidity_test_2026-06-25.csv"
    return "reports/active/multi_exchange_liquidity_test_2026-06-25.csv"


def _venue_lane_classification(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "asset",
        "best_lane",
        "dydx_lane",
        "hyperliquid_lane",
        "blockers",
        "next_action",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for asset, subset in frame[frame["asset"].astype(str) != "ALL"].groupby("asset"):
        lanes = set(subset["venue_lane"].astype(str))
        blockers = sorted({part for value in subset["blocker"].dropna().astype(str) for part in value.split(";") if part})
        dydx_lane = next((lane for lane in lanes if lane.startswith("dydx_") or lane == "blocked_liquidity"), "")
        hyper_lane = next((lane for lane in lanes if lane.startswith("hyperliquid_")), "")
        best_lane = _best_asset_lane(lanes)
        rows.append(
            {
                "asset": asset,
                "best_lane": best_lane,
                "dydx_lane": dydx_lane,
                "hyperliquid_lane": hyper_lane,
                "blockers": ";".join(blockers),
                "next_action": _venue_lane_next_action(best_lane, blockers),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["best_lane", "asset"]
    ).reset_index(drop=True)


def _best_asset_lane(lanes: set[str]) -> str:
    priority = [
        "dydx_execution_candidate",
        "dydx_execution_watch",
        "hyperliquid_research_candidate",
        "hyperliquid_watch",
        "dydx_research_only",
        "blocked_liquidity",
    ]
    for lane in priority:
        if lane in lanes:
            return lane
    return sorted(lanes)[0] if lanes else "needs_more_data"


def _venue_lane_next_action(best_lane: str, blockers: list[str]) -> str:
    if best_lane == "dydx_execution_candidate":
        return "run_dydx_exact_mode_local_replay"
    if best_lane == "dydx_execution_watch":
        return "run_dydx_replay_with_size_and_slippage_limits"
    if best_lane == "hyperliquid_research_candidate":
        return "build_hyperliquid_history_and_cost_model"
    if best_lane == "hyperliquid_watch":
        return "collect_more_hyperliquid_depth_and_slippage_evidence"
    if best_lane == "dydx_research_only":
        return "do_not_promote_on_dydx_without_liquidity_improvement"
    if best_lane == "blocked_liquidity":
        return "keep_blocked_until_new_venue_or_liquidity_source"
    if blockers:
        return "resolve_blockers_before_testing"
    return "review_source_context"


def _market_venue_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["metric", "value"])
    return pd.DataFrame(
        [
            {"metric": "rows", "value": len(frame)},
            {"metric": "assets", "value": frame["asset"].nunique()},
            {"metric": "venues", "value": frame["venue"].nunique()},
            {"metric": "promotion_allowed_rows", "value": int(frame["promotion_allowed"].astype(bool).sum())},
            {"metric": "blocked_rows", "value": int(frame["blocker"].astype(str).ne("").sum())},
            {"metric": "funding_pulse_status", "value": "needs_api_key"},
            {
                "metric": "hyperliquid_research_candidates",
                "value": int(frame["venue_lane"].astype(str).eq("hyperliquid_research_candidate").sum()),
            },
            {"metric": "dydx_execution_candidates", "value": int(frame["venue_lane"].astype(str).eq("dydx_execution_candidate").sum())},
        ]
    )


def _market_venue_context_markdown(frame: pd.DataFrame, lanes: pd.DataFrame) -> str:
    if frame.empty:
        return "# Market Venue Context\n\nNo market venue context rows were built.\n"
    summary = _market_venue_summary(frame)
    lane_counts = frame["venue_lane"].value_counts().reset_index()
    lane_counts.columns = ["venue_lane", "rows"]
    parts = [
        "# Market Venue Context",
        "",
        "This report normalizes the current venue/liquidity evidence before pair testing.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Lane Counts",
        "",
        lane_counts.to_markdown(index=False),
        "",
        "## Asset Lane Classification",
        "",
        lanes.to_markdown(index=False) if not lanes.empty else "No assets classified.",
        "",
        "## Rules Enforced",
        "",
        "- dYdX rows can only support dYdX venue testing.",
        "- Hyperliquid rows route assets into a Hyperliquid research lane until local Hyperliquid replay exists.",
        "- CoinGlass, DexScreener, CoinGecko, CoinMarketCap, KuCoin, and GMX are context/discovery sources, not promotion authority.",
        "- Funding Pulse remains planned with `needs_api_key` and cannot promote trades.",
        "- Blockers are carried forward instead of hidden behind scores.",
        "",
        "## Next Step",
        "",
        "Rerun candidate tests by venue lane, starting with dYdX exact-mode replay for BTC, ETH, SOL, and size-limited DOGE, while building Hyperliquid history/cost support for WLD, HYPE, LINK, TRX, and TAO.",
        "",
    ]
    return "\n".join(parts)


def _clean_numeric(value: object) -> object:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return ""


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"yes", "true", "1", "available", "partial"}


def build_pair_universe(root: Path = ROOT) -> CommandResult:
    active = root / "reports" / "active"
    pair_rows = []
    experiment = _read_csv(root / "reports" / "experiment_results.csv")
    acceptance = _read_csv(root / "reports" / "acceptance_report.csv")
    funding = _read_csv(root / "data" / "processed" / "dydx_funding.csv")
    wizard = _wizard_research_tables(root)
    pairs = _candidate_pairs(root)
    candle_index = _dydx_candle_index(root)
    market_snapshot = _latest_apify_markets(root)
    venue_context = _read_csv(root / "data" / "processed" / "market_venue_context.csv")
    market_context = _venue_context_by_pair(venue_context)
    for pair, assets, evidence_paths in pairs:
        asset_x, asset_y = assets
        metrics = _local_pair_metrics(pair, experiment, acceptance)
        wizard_metrics = _wizard_pair_metrics(pair, wizard)
        metrics.update(wizard_metrics)
        _apply_market_snapshot_metrics(metrics, asset_x, asset_y, market_snapshot)
        funding_drag = _funding_drag(asset_x, asset_y, funding, market_snapshot)
        timeframes = sorted(set(candle_index.get(asset_x, set())) & set(candle_index.get(asset_y, set())))
        venue_profile = _venue_profile_for_pair(market_context, asset_x, asset_y)
        if not venue_profile:
            wizard_venue = normalize_wizard_exchange(metrics.get("best_wizard_exchange"), default=None)
            wizard_is_current = _boolish(metrics.get("best_wizard_source_fresh", False)) and str(
                metrics.get("best_wizard_source_health", "") or ""
            ).lower() == "healthy"
            if wizard_venue and wizard_venue != "dydx" and wizard_is_current:
                venue_profile = {
                    "best_venue": wizard_venue,
                    "execution_ready": False,
                    "available_venues": wizard_venue,
                    "reason": "wizard_discovery_venue_requires_local_context",
                }
            else:
                venue_profile = _venue_profile_from_dydx(candle_index, timeframes, market_snapshot, asset_x, asset_y)
        execution_venue = str(venue_profile.get("best_venue", "hyperliquid")).lower()
        execution_ready = bool(venue_profile.get("execution_ready", bool(candle_index.get(asset_x) and candle_index.get(asset_y))))
        available_venues = str(venue_profile.get("available_venues", ""))
        venue_reason = str(venue_profile.get("reason", ""))
        discovery_components = _discovery_components(execution_ready, timeframes, metrics, funding_drag, available_venues=available_venues)
        acceptance_components = _acceptance_components(execution_ready, metrics, funding_drag, best_venue=execution_venue, venue_ready=venue_profile.get("execution_ready", False))
        discovery_score = round(sum(discovery_components.values()), 3)
        acceptance_score = round(sum(acceptance_components.values()), 3)
        combined_score = round(discovery_score + acceptance_score, 3)
        bucket, reason = _decision_bucket(
            discovery_score,
            acceptance_score,
            metrics,
            execution_ready,
            timeframes,
            best_venue=execution_venue,
            available_venues=available_venues,
        )
        missing = _missing_pair_data(execution_ready, timeframes, metrics, funding_drag, best_venue=execution_venue)
        wizard_state = str(metrics.get("wizard_evidence_state", "") or "")
        wizard_source_fresh = _boolish(metrics.get("best_wizard_source_fresh", False))
        if wizard_source_fresh:
            field_freshness = "fresh_wizard_discovery"
        elif wizard_state == "STALE_EVIDENCE":
            field_freshness = "stale_wizard_evidence"
        else:
            field_freshness = "current_snapshot"
        stale_reasons = []
        if wizard_state in {"STALE_EVIDENCE", "CAPTURE_UNHEALTHY", "INVALID_EXACT_MODE"}:
            stale_reasons.append(wizard_state.lower())
        if missing:
            stale_reasons.append("missing_or_partial_inputs")
        pair_rows.append(
            {
                "pair": pair,
                "asset_x": asset_x,
                "asset_y": asset_y,
                "exchange": execution_venue,
                "dydx_tradable": bool(execution_ready),
                "best_execution_venue": execution_venue,
                "execution_venue_ready": bool(execution_ready),
                "available_venues": available_venues,
                "venue_decision_reason": venue_reason,
                "available_timeframes": ";".join(timeframes),
                "wizards_pair_id": "",
                "cointegration_score": metrics.get("cointegration_score", 0.0),
                "copula_score": metrics.get("copula_score", 0.0),
                "zscore_score": metrics.get("zscore_score", 0.0),
                "half_life": metrics.get("half_life", ""),
                "hurst": metrics.get("hurst", ""),
                "correlation": metrics.get("correlation", ""),
                "funding_drag_bps": funding_drag,
                "volume_usd": metrics.get("volume_usd", ""),
                "open_interest_usd": metrics.get("open_interest_usd", ""),
                "local_backtest_score": metrics.get("local_backtest_score", 0.0),
                "discovery_score": discovery_score,
                "acceptance_score": acceptance_score,
                "combined_score": combined_score,
                "decision_bucket": bucket,
                "decision_reason": reason,
                "missing_data_reason": missing,
                "source_timestamp": metrics.get("best_wizard_source_timestamp", "") or _now(),
                "field_freshness": field_freshness,
                "stale_reason": ";".join(stale_reasons),
                "evidence_path": ";".join(sorted(evidence_paths | _market_evidence_paths(asset_x, asset_y, market_snapshot))),
                "best_wizard_exact_mode": metrics.get("best_wizard_exact_mode", ""),
                "best_wizard_spread_id": metrics.get("best_wizard_spread_id", ""),
                "best_wizard_strategy_id": metrics.get("best_wizard_strategy_id", ""),
                "best_wizard_local_strategy_id": metrics.get("best_wizard_local_strategy_id", ""),
                "best_wizard_local_strategy_name": metrics.get("best_wizard_local_strategy_name", ""),
                "best_wizard_local_strategy_family": metrics.get("best_wizard_local_strategy_family", ""),
                "best_wizard_strategy_mapping_status": metrics.get("best_wizard_strategy_mapping_status", ""),
                "best_wizard_sharpe": metrics.get("best_wizard_sharpe", ""),
                "best_wizard_returns_total": metrics.get("best_wizard_returns_total", ""),
                "best_wizard_exchange": metrics.get("best_wizard_exchange", ""),
                "best_wizard_source_authority": metrics.get("best_wizard_source_authority", ""),
                "best_wizard_source_timestamp": metrics.get("best_wizard_source_timestamp", ""),
                "best_wizard_source_fresh": metrics.get("best_wizard_source_fresh", ""),
                "best_wizard_source_health": metrics.get("best_wizard_source_health", ""),
                "wizard_evidence_state": metrics.get("wizard_evidence_state", ""),
                "wizard_stationarity_status": metrics.get("wizard_stationarity_status", ""),
                "wizard_engle_granger_cointegrated": metrics.get("wizard_engle_granger_cointegrated", ""),
                "wizard_engle_granger_trend": metrics.get("wizard_engle_granger_trend", ""),
                "wizard_johansen_cointegrated": metrics.get("wizard_johansen_cointegrated", ""),
                "wizard_zscore_last": metrics.get("wizard_zscore_last", ""),
                "wizard_zscore_roll_last": metrics.get("wizard_zscore_roll_last", ""),
                "wizard_leg_volume_min": metrics.get("wizard_leg_volume_min", ""),
                "wizard_diagnostic_score": metrics.get("wizard_diagnostic_score", ""),
                "wizard_hypothesis_status": metrics.get("wizard_hypothesis_status", ""),
                "local_mode_confirmation_status": metrics.get("local_mode_confirmation_status", ""),
                "wizard_local_parity_status": metrics.get("wizard_local_parity_status", ""),
                "promotion_blocker": _promotion_blocker(bucket, metrics),
            }
        )
    frame = pd.DataFrame(pair_rows, columns=PAIR_UNIVERSE_COLUMNS)
    if not frame.empty:
        frame["_wizard_strategy_rank"] = frame.get("best_wizard_local_strategy_id", pd.Series("", index=frame.index)).astype(str).str.strip().ne("").astype(int)
        frame = frame.sort_values(
            ["decision_bucket", "_wizard_strategy_rank", "acceptance_score", "discovery_score"],
            ascending=[True, False, False, False],
        ).drop(columns=["_wizard_strategy_rank"])
    output = root / "data" / "processed" / "pair_universe.csv"
    snapshot = root / "data" / "processed" / "pair_universe_snapshots" / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M')}.csv"
    summary = active / "pair_universe_summary.csv"
    summary_md = active / "pair_universe_summary.md"
    components = active / "pair_score_components.csv"
    decisions = active / "pair_decision_buckets.csv"
    _write_csv(frame, output)
    _write_csv(frame, snapshot)
    _write_csv(_pair_summary(frame), summary)
    _write_text(summary_md, _pair_universe_markdown(frame))
    _write_csv(_pair_score_components(frame), components)
    _write_csv(frame[["pair", "decision_bucket", "decision_reason", "missing_data_reason", "evidence_path"]], decisions)
    return CommandResult(
        paths={"pair_universe": output, "snapshot": snapshot, "summary": summary, "summary_md": summary_md, "components": components, "decisions": decisions},
        summary={"pairs": len(frame), "promote": int((frame["decision_bucket"] == "PROMOTE").sum()) if not frame.empty else 0},
    )


def build_trade_dataset(root: Path = ROOT, input_dir: Path | None = None, funding_path: Path | None = None) -> CommandResult:
    from quant_platform.three_brain_system import (
        build_native_outcome_feature_memory,
        build_shared_outcome_memory,
    )

    source = input_dir or root / "data" / "raw" / "pair_details"
    if input_dir is None:
        source_datasets, registry_audit = (
            _registered_hyperliquid_training_datasets(root)
        )
        if not source_datasets:
            raise SystemExit(
                "no hash-verified READY histories exist in the Stage 2 Hyperliquid registries"
            )
    else:
        source_datasets = datasets_from_pair_detail_snapshots(
            source, require_research_usable=True
        )
        registry_audit = pd.DataFrame(
            [
                {
                    "registry_path": str(source),
                    "registry_status": "EXPLICIT_INPUT_DIRECTORY",
                    "eligible_for_candidate_dataset": True,
                    "blocker": "",
                }
            ]
        )
    datasets, source_selection = _select_hyperliquid_training_datasets(
        source_datasets
    )
    source_selection = _annotate_trade_dataset_cost_coverage(
        root, source_selection
    )
    if not datasets:
        raise SystemExit(
            "no canonical Hyperliquid histories are eligible for the trade dataset"
        )
    datasets = [
        PairDataset(dataset.pair, classify_regimes(_ensure_trade_dataset_inputs(dataset.frame), RegimeConfig(preserve_existing=True)))
        for dataset in datasets
    ]
    frame = build_trade_filter_dataset(datasets)
    if frame.empty:
        raise SystemExit(f"no leakage-safe trade rows could be built from {source}")
    hardened = _harden_trade_dataset(frame)
    build_shared_outcome_memory(root=root)
    build_native_outcome_feature_memory(root=root)
    audit = _leakage_audit(hardened)
    blocked_audit = audit["leakage_blocker"].fillna("").astype(str).ne("")
    if blocked_audit.any():
        blockers = audit.loc[blocked_audit, "trade_id"].head(5).tolist()
        raise SystemExit(f"leakage audit failed for trade rows: {blockers}")
    staged = _stage_trade_dataset_candidate(
        root=root,
        dataset=hardened,
        leakage_audit=audit,
        source_selection=source_selection,
        registry_audit=registry_audit,
    )
    return CommandResult(
        paths=staged["paths"],
        summary={
            "rows": len(hardened),
            "parquet_status": staged["parquet_status"],
            "canonical_hyperliquid_histories": len(datasets),
            "dataset_id": staged["dataset_id"],
            "status": "VALIDATED_CANDIDATE",
            "active_dataset_unchanged": True,
        },
    )


def _registered_hyperliquid_training_datasets(
    root: Path,
) -> tuple[list[PairDataset], pd.DataFrame]:
    """Load only hash-verified Stage 2 histories registered as replay-ready."""

    datasets: list[PairDataset] = []
    audit_rows: list[dict[str, object]] = []
    ready_statuses = {"READY", "READY_FOR_CANONICAL_1X_REPLAY"}
    for registry_name in TRADE_DATASET_REGISTRIES:
        registry_path = root / "reports" / "active" / registry_name
        registry = _read_csv(registry_path)
        if registry.empty:
            audit_rows.append(
                {
                    "registry_path": str(registry_path),
                    "registry_status": "MISSING_OR_EMPTY",
                    "eligible_for_candidate_dataset": False,
                    "blocker": "registered_history_ledger_missing_or_empty",
                }
            )
            continue
        for _, row in registry.iterrows():
            pair = str(row.get("pair", "")).strip()
            timeframe = str(
                row.get("hyperliquid_interval", row.get("timeframe", ""))
            ).strip().lower()
            registered_status = str(row.get("history_status", "")).strip()
            raw_history_path = str(row.get("history_path", "")).strip()
            expected_hash = str(row.get("history_sha256", "")).strip().lower()
            expected_rows = int(
                pd.to_numeric(
                    pd.Series([row.get("history_rows", 0)]), errors="coerce"
                )
                .fillna(0)
                .iloc[0]
            )
            blocker = ""
            actual_hash = ""
            actual_rows = 0
            history_path = Path(raw_history_path) if raw_history_path else Path()
            if raw_history_path and not history_path.is_absolute():
                history_path = root / history_path
            if registered_status not in ready_statuses:
                blocker = "registered_history_not_ready"
            elif not raw_history_path or not history_path.is_file():
                blocker = "registered_history_file_missing"
            elif not expected_hash:
                blocker = "registered_history_hash_missing"
            else:
                actual_hash = _sha256_file(history_path)
                if actual_hash != expected_hash:
                    blocker = "registered_history_hash_mismatch"
            if not blocker:
                try:
                    payload = load_pair_detail_payload(history_path)
                    snapshot = snapshot_from_payload(payload)
                    history = extract_history_rows(payload)
                    actual_rows = len(history)
                    if not history:
                        blocker = "registered_history_rows_missing"
                    elif expected_rows and actual_rows != expected_rows:
                        blocker = "registered_history_row_count_mismatch"
                    elif pair and snapshot.pair and pair != snapshot.pair:
                        blocker = "registered_history_pair_mismatch"
                    else:
                        frame = pd.DataFrame(history)
                        frame["source_path"] = raw_history_path
                        frame["source_registry_path"] = str(
                            registry_path.relative_to(root)
                        )
                        frame["registered_history_sha256"] = actual_hash
                        for column, value in snapshot.to_row().items():
                            if column not in frame.columns and value is not None:
                                frame[column] = value
                        frame["exchange"] = "hyperliquid"
                        if timeframe:
                            frame["timeframe"] = timeframe
                            frame["interval"] = timeframe
                        frame = add_derived_beta_from_prices(frame)
                        frame = _normalize_registered_history_features(frame)
                        if "regime" not in frame.columns:
                            frame["regime"] = "unknown"
                        datasets.append(
                            PairDataset(pair=pair or snapshot.pair, frame=frame)
                        )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    blocker = f"registered_history_parse_failed:{type(exc).__name__}"
            audit_rows.append(
                {
                    "registry_path": str(registry_path.relative_to(root)),
                    "pair": pair,
                    "timeframe": timeframe,
                    "registered_history_status": registered_status,
                    "history_path": raw_history_path,
                    "expected_history_sha256": expected_hash,
                    "actual_history_sha256": actual_hash,
                    "expected_rows": expected_rows,
                    "actual_rows": actual_rows,
                    "registry_status": "HASH_VERIFIED_READY" if not blocker else "REJECTED",
                    "eligible_for_candidate_dataset": not blocker,
                    "blocker": blocker,
                    "promotion_authority": False,
                    "testnet_order_authority": False,
                    "live_trading_authorized": False,
                }
            )
    return datasets, pd.DataFrame(audit_rows)


def _normalize_registered_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose canonical feature names without hiding math-v2/proxy provenance."""

    normalized = frame.copy()
    mappings: dict[str, tuple[str, ...]] = {
        "conditional_probability_distortion": (
            "math_v2_conditional_probability_distortion",
            "research_proxy_conditional_probability_distortion",
        ),
        "u1_given_u2": (
            "math_v2_u1_given_u2",
            "research_proxy_u1_given_u2",
        ),
        "u2_given_u1": (
            "math_v2_u2_given_u1",
            "research_proxy_u2_given_u1",
        ),
        "ecm_strength": (
            "math_v2_ecm_strength",
            "research_proxy_ecm_strength",
        ),
        "ecm_x": ("research_proxy_ecm_x",),
        "ecm_y": ("research_proxy_ecm_y",),
        "half_life": ("math_v2_half_life", "research_proxy_half_life"),
        "hurst": ("math_v2_hurst", "research_proxy_hurst"),
        "ou_optimal": ("research_proxy_ou_optimal",),
        "tail_dependence": ("research_proxy_tail_dependence",),
        "cvar": ("research_proxy_cvar",),
        "regime_strategy_match": ("research_proxy_regime_strategy_match",),
        "cointegration_pvalue": (
            "math_v2_cointegration_pvalue",
            "research_proxy_cointegration_pvalue",
        ),
        "copula_calibration_score": (
            "research_proxy_copula_calibration_score",
        ),
        "composite_score": ("research_proxy_composite_score",),
        "bid_ask_spread_bps": ("research_proxy_bid_ask_spread_bps",),
        "funding_bps_per_day": ("research_proxy_funding_bps_per_day",),
    }
    for target, sources in mappings.items():
        values = pd.to_numeric(
            normalized.get(target, pd.Series(np.nan, index=normalized.index)),
            errors="coerce",
        )
        provenance = pd.Series(
            np.where(values.notna(), target, ""), index=normalized.index
        )
        for source in sources:
            if source not in normalized.columns:
                continue
            candidate = pd.to_numeric(normalized[source], errors="coerce")
            use = values.isna() & candidate.notna()
            values = values.where(~use, candidate)
            provenance = provenance.where(~use, source)
        normalized[target] = values
        normalized[f"{target}_feature_source"] = provenance
    return normalized


def _annotate_trade_dataset_cost_coverage(
    root: Path, source_selection: pd.DataFrame
) -> pd.DataFrame:
    annotated = source_selection.copy()
    costs = _read_csv(root / "data" / "processed" / "hyperliquid_pair_cost_models.csv")
    ready_pairs: set[str] = set()
    if not costs.empty and "pair" in costs.columns:
        ready = costs.get(
            "strict_observed_cost_ready", pd.Series(False, index=costs.index)
        ).map(_truthy)
        ready_pairs = set(costs.loc[ready, "pair"].astype(str))
    annotated["strict_observed_cost_ready"] = annotated.get(
        "pair", pd.Series("", index=annotated.index)
    ).astype(str).isin(ready_pairs)
    annotated["strict_cost_model_path"] = np.where(
        annotated["strict_observed_cost_ready"],
        "data/processed/hyperliquid_pair_cost_models.csv",
        "",
    )
    return annotated


def _stage_trade_dataset_candidate(
    *,
    root: Path,
    dataset: pd.DataFrame,
    leakage_audit: pd.DataFrame,
    source_selection: pd.DataFrame,
    registry_audit: pd.DataFrame,
    receipt_context: dict[str, object] | None = None,
    additional_acceptance_blockers: Iterable[str] = (),
) -> dict[str, object]:
    context = dict(receipt_context or {})
    protected_context_keys = {
        "schema_version",
        "dataset_id",
        "status",
        "dataset_sha256",
        "leakage_audit_sha256",
        "source_selection_sha256",
        "history_registry_audit_sha256",
        "research_acceptance_blockers",
        "promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    }
    forbidden_context = protected_context_keys.intersection(context)
    if forbidden_context:
        raise ValueError(
            "trade dataset receipt context overrides protected fields: "
            + ";".join(sorted(forbidden_context))
        )
    context_hash = hashlib.sha256(
        json.dumps(
            context, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()
    data_ml = root / "data" / "ml"
    ml_reports = root / "reports" / "ml"
    builds = data_ml / "dataset_builds"
    temporary = builds / f".candidate-{uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    csv_path = temporary / "trade_training_dataset.csv"
    parquet_path = temporary / "trade_training_dataset.parquet"
    audit_path = temporary / "leakage_audit.csv"
    selection_path = temporary / "trade_dataset_source_selection.csv"
    registry_path = temporary / "trade_dataset_history_registry_audit.csv"
    summary_path = temporary / "trade_dataset_summary.csv"
    _write_csv(dataset, csv_path)
    parquet_status = _write_parquet_if_available(dataset, parquet_path)
    _write_csv(leakage_audit, audit_path)
    _write_csv(source_selection, selection_path)
    _write_csv(registry_audit, registry_path)
    _write_csv(
        _trade_dataset_summary(dataset, leakage_audit, parquet_status),
        summary_path,
    )
    dataset_hash = _sha256_file(csv_path)
    audit_hash = _sha256_file(audit_path)
    selection_hash = _sha256_file(selection_path)
    registry_hash = _sha256_file(registry_path)
    identity = hashlib.sha256(
        "|".join(
            (
                TRADE_DATASET_BUILD_SCHEMA,
                dataset_hash,
                audit_hash,
                selection_hash,
                registry_hash,
                context_hash,
            )
        ).encode("utf-8")
    ).hexdigest()
    dataset_id = f"tradedataset_{identity[:20]}"
    final_dir = builds / dataset_id
    selected = source_selection.get(
        "selection_status", pd.Series("", index=source_selection.index)
    ).astype(str).eq("SELECTED_CANONICAL")
    strict = source_selection.get(
        "strict_observed_cost_ready", pd.Series(False, index=source_selection.index)
    ).map(_truthy)
    exact_modes = sorted(
        value
        for value in dataset.get("exact_mode", pd.Series(dtype=object))
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
        if value
    )
    strict_cost_intersections = int((selected & strict).sum())
    selected_histories = int(selected.sum())
    acceptance_blockers = [
        str(value).strip()
        for value in additional_acceptance_blockers
        if str(value).strip()
    ]
    if not exact_modes:
        acceptance_blockers.append("exact_mode_trade_provenance_missing")
    if selected_histories <= 0:
        acceptance_blockers.append("canonical_history_selection_missing")
    elif strict_cost_intersections != selected_histories:
        acceptance_blockers.append("strict_observed_cost_history_coverage_incomplete")
    if not str(context.get("registered_execution_id", "")).strip():
        acceptance_blockers.append("registered_stage4_execution_lineage_missing")
    receipt = {
        "schema_version": TRADE_DATASET_BUILD_SCHEMA,
        "dataset_id": dataset_id,
        "created_at_utc": _now(),
        "status": "VALIDATED_CANDIDATE",
        "rows": len(dataset),
        "pairs": int(dataset.get("pair", pd.Series(dtype=object)).nunique()),
        "timeframes": sorted(
            dataset.get("timeframe", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "source_venues": sorted(
            dataset.get("source_venue", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "strategy_names": sorted(
            dataset.get("strategy_name", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "strategy_families": sorted(
            dataset.get("family", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "exact_modes": exact_modes,
        "canonical_hyperliquid_histories": int(selected.sum()),
        "strict_cost_history_intersections": strict_cost_intersections,
        "selected_canonical_histories": selected_histories,
        "cost_model_scope": (
            "pair_specific_strict_observed_costs"
            if selected_histories > 0
            and strict_cost_intersections == selected_histories
            else "incomplete_or_provisional_cost_coverage"
        ),
        "receipt_context_sha256": context_hash,
        **context,
        "research_acceptance_blockers": sorted(set(acceptance_blockers)),
        "dataset_sha256": dataset_hash,
        "leakage_audit_sha256": audit_hash,
        "source_selection_sha256": selection_hash,
        "history_registry_audit_sha256": registry_hash,
        "dataset_path": str((final_dir / csv_path.name).relative_to(root)),
        "parquet_path": str((final_dir / parquet_path.name).relative_to(root)),
        "leakage_audit_path": str((final_dir / audit_path.name).relative_to(root)),
        "source_selection_path": str((final_dir / selection_path.name).relative_to(root)),
        "history_registry_audit_path": str((final_dir / registry_path.name).relative_to(root)),
        "summary_path": str((final_dir / summary_path.name).relative_to(root)),
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    _write_json(temporary / "dataset_receipt.json", receipt)
    if final_dir.exists():
        existing = _read_json(final_dir / "dataset_receipt.json")
        if existing.get("dataset_sha256") != dataset_hash:
            shutil.rmtree(temporary)
            raise SystemExit(f"dataset build identity collision: {dataset_id}")
        shutil.rmtree(temporary)
    else:
        promote_staged_directory(temporary, final_dir)
    pointer = {
        **receipt,
        "candidate_receipt_path": str(
            (final_dir / "dataset_receipt.json").relative_to(root)
        ),
        "active_dataset_unchanged": True,
    }
    candidate_pointer = data_ml / "candidate_trade_dataset.json"
    _write_json(candidate_pointer, pointer)
    _write_json(ml_reports / "trade_dataset_candidate_receipt.json", pointer)
    _write_csv(
        source_selection,
        ml_reports / "candidate_trade_dataset_source_selection.csv",
    )
    _write_csv(
        registry_audit,
        ml_reports / "candidate_trade_dataset_history_registry_audit.csv",
    )
    paths = {
        "dataset_csv": final_dir / csv_path.name,
        "dataset_parquet": final_dir / parquet_path.name,
        "summary": final_dir / summary_path.name,
        "leakage_audit": final_dir / audit_path.name,
        "source_selection": final_dir / selection_path.name,
        "history_registry_audit": final_dir / registry_path.name,
        "receipt": final_dir / "dataset_receipt.json",
        "candidate_pointer": candidate_pointer,
    }
    return {
        "paths": paths,
        "dataset_id": dataset_id,
        "parquet_status": parquet_status,
    }


def promote_trade_dataset(
    root: Path = ROOT, candidate_pointer_path: Path | None = None
) -> CommandResult:
    """Promote a validated candidate while preserving the prior active evidence."""

    data_ml = root / "data" / "ml"
    ml_reports = root / "reports" / "ml"
    pointer_path = candidate_pointer_path or data_ml / "candidate_trade_dataset.json"
    candidate = _read_json(pointer_path)
    if candidate.get("status") != "VALIDATED_CANDIDATE":
        raise SystemExit("validated candidate trade dataset pointer is missing")
    required_paths = {
        "dataset": _rooted_path(root, candidate.get("dataset_path")),
        "leakage_audit": _rooted_path(root, candidate.get("leakage_audit_path")),
        "source_selection": _rooted_path(root, candidate.get("source_selection_path")),
        "history_registry_audit": _rooted_path(
            root, candidate.get("history_registry_audit_path")
        ),
        "summary": _rooted_path(root, candidate.get("summary_path")),
    }
    expected_hashes = {
        "dataset": str(candidate.get("dataset_sha256", "")),
        "leakage_audit": str(candidate.get("leakage_audit_sha256", "")),
        "source_selection": str(candidate.get("source_selection_sha256", "")),
        "history_registry_audit": str(
            candidate.get("history_registry_audit_sha256", "")
        ),
    }
    blockers = []
    declared_blockers = candidate.get("research_acceptance_blockers", [])
    if isinstance(declared_blockers, str):
        declared_blockers = [
            value.strip()
            for value in declared_blockers.replace(",", ";").split(";")
            if value.strip()
        ]
    if not isinstance(declared_blockers, list):
        declared_blockers = ["invalid_research_acceptance_blockers"]
    blockers.extend(
        f"candidate_research_acceptance_blocker:{value}"
        for value in declared_blockers
        if str(value).strip()
    )
    for name, path in required_paths.items():
        if not path.is_file():
            blockers.append(f"candidate_artifact_missing:{name}")
        elif name in expected_hashes and _sha256_file(path) != expected_hashes[name]:
            blockers.append(f"candidate_artifact_hash_mismatch:{name}")
    leakage = _read_csv(required_paths["leakage_audit"])
    leakage_blockers = int(
        leakage.get(
            "leakage_blocker", pd.Series("", index=leakage.index)
        )
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
    if leakage.empty:
        blockers.append("candidate_leakage_audit_empty")
    if leakage_blockers:
        blockers.append("candidate_leakage_audit_blocked")
    dataset = _read_csv(required_paths["dataset"])
    forbidden = sorted(POST_OUTCOME_MEMORY_COLUMNS.intersection(dataset.columns))
    if forbidden:
        blockers.append("post_outcome_memory_features_present")
    venues = dataset.get(
        "source_venue", pd.Series("", index=dataset.index)
    ).fillna("").astype(str).str.lower()
    if dataset.empty or venues.ne("hyperliquid").any():
        blockers.append("candidate_dataset_not_hyperliquid_only")
    observed_modes = {
        value
        for value in dataset.get("exact_mode", pd.Series(dtype=object))
        .fillna("")
        .astype(str)
        .str.strip()
        if value
    }
    declared_modes = {
        str(value).strip()
        for value in candidate.get("exact_modes", [])
        if str(value).strip()
    }
    if not observed_modes:
        blockers.append("candidate_exact_mode_trade_provenance_missing")
    elif observed_modes != declared_modes:
        blockers.append("candidate_exact_mode_receipt_mismatch")
    source_selection = _read_csv(required_paths["source_selection"])
    selected = source_selection.get(
        "selection_status", pd.Series("", index=source_selection.index)
    ).astype(str).eq("SELECTED_CANONICAL")
    strict = source_selection.get(
        "strict_observed_cost_ready", pd.Series(False, index=source_selection.index)
    ).map(_truthy)
    selected_count = int(selected.sum())
    strict_selected_count = int((selected & strict).sum())
    if selected_count <= 0:
        blockers.append("candidate_canonical_history_selection_missing")
    elif strict_selected_count != selected_count:
        blockers.append("candidate_strict_cost_coverage_incomplete")
    declared_strict_count = int(candidate.get("strict_cost_history_intersections", -1))
    if declared_strict_count != strict_selected_count:
        blockers.append("candidate_strict_cost_receipt_mismatch")
    for field in (
        "registered_contract_id",
        "registered_execution_id",
        "source_family_sha256",
        "exact_mode_parity_sha256",
    ):
        if not str(candidate.get(field, "")).strip():
            blockers.append(f"candidate_registered_lineage_missing:{field}")
    for field in (
        "full_family_accounted",
        "causal_entry_features_proven",
        "strict_cost_coverage_complete",
    ):
        if not _truthy(candidate.get(field)):
            blockers.append(f"candidate_registered_lineage_not_proven:{field}")
    if candidate.get("cost_model_scope") != "pair_specific_strict_observed_costs":
        blockers.append("candidate_cost_model_scope_not_strict_pair_specific")
    execution_receipt = _rooted_path(
        root, candidate.get("registered_execution_receipt_path")
    )
    execution_receipt_hash = str(
        candidate.get("registered_execution_receipt_sha256", "")
    )
    if not execution_receipt.is_file():
        blockers.append("candidate_registered_execution_receipt_missing")
    elif (
        not execution_receipt_hash
        or _sha256_file(execution_receipt) != execution_receipt_hash
    ):
        blockers.append("candidate_registered_execution_receipt_hash_mismatch")
    if blockers:
        raise SystemExit(
            "candidate trade dataset promotion blocked: " + ";".join(blockers)
        )

    superseded = _snapshot_active_trade_dataset(root)
    canonical = {
        "dataset": data_ml / "trade_training_dataset.csv",
        "leakage_audit": ml_reports / "leakage_audit.csv",
        "source_selection": ml_reports / "trade_dataset_source_selection.csv",
        "history_registry_audit": ml_reports
        / "trade_dataset_history_registry_audit.csv",
        "summary": ml_reports / "trade_dataset_summary.csv",
    }
    for name, destination in canonical.items():
        _link_or_copy_atomic(required_paths[name], destination)
    candidate_parquet = _rooted_path(root, candidate.get("parquet_path"))
    canonical_parquet = data_ml / "trade_training_dataset.parquet"
    if candidate_parquet.is_file():
        _link_or_copy_atomic(candidate_parquet, canonical_parquet)
    active_pointer = {
        **candidate,
        "status": "ACTIVE_RESEARCH_DATASET",
        "promoted_at_utc": _now(),
        "active_dataset_path": str(canonical["dataset"].relative_to(root)),
        "active_dataset_sha256": _sha256_file(canonical["dataset"]),
        "superseded_dataset_receipt": (
            str(superseded.relative_to(root)) if superseded else ""
        ),
        "model_retraining_required": True,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    active_pointer_path = data_ml / "active_trade_dataset.json"
    _write_json(active_pointer_path, active_pointer)
    _write_json(ml_reports / "active_trade_dataset_receipt.json", active_pointer)
    return CommandResult(
        paths={
            "dataset_csv": canonical["dataset"],
            "dataset_parquet": canonical_parquet,
            "leakage_audit": canonical["leakage_audit"],
            "source_selection": canonical["source_selection"],
            "history_registry_audit": canonical["history_registry_audit"],
            "summary": canonical["summary"],
            "active_pointer": active_pointer_path,
        },
        summary={
            "dataset_id": candidate.get("dataset_id", ""),
            "rows": len(dataset),
            "status": "ACTIVE_RESEARCH_DATASET",
            "model_retraining_required": True,
            "promotion_authority": False,
        },
    )


def _snapshot_active_trade_dataset(root: Path) -> Path | None:
    data_ml = root / "data" / "ml"
    current = data_ml / "trade_training_dataset.csv"
    if not current.is_file():
        return None
    current_hash = _sha256_file(current)
    snapshot_dir = data_ml / "dataset_builds" / f"superseded_{current_hash[:20]}"
    receipt_path = snapshot_dir / "dataset_receipt.json"
    if receipt_path.is_file():
        return receipt_path
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "trade_training_dataset.csv": current,
        "trade_training_dataset.parquet": data_ml / "trade_training_dataset.parquet",
        "leakage_audit.csv": root / "reports" / "ml" / "leakage_audit.csv",
        "trade_dataset_source_selection.csv": root
        / "reports"
        / "ml"
        / "trade_dataset_source_selection.csv",
        "trade_dataset_summary.csv": root
        / "reports"
        / "ml"
        / "trade_dataset_summary.csv",
    }
    preserved = []
    for name, source in artifacts.items():
        if source.is_file():
            destination = snapshot_dir / name
            _link_or_copy(source, destination)
            preserved.append(name)
    _write_json(
        receipt_path,
        {
            "schema_version": TRADE_DATASET_BUILD_SCHEMA,
            "dataset_id": f"superseded_{current_hash[:20]}",
            "status": "SUPERSEDED_PRESERVED",
            "preserved_at_utc": _now(),
            "dataset_sha256": current_hash,
            "artifacts": preserved,
            "promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    return receipt_path


def _rooted_path(root: Path, value: object) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else root / path


def _link_or_copy(source: Path, destination: Path) -> None:
    write_immutable_bytes(destination, source.read_bytes())


def _link_or_copy_atomic(source: Path, destination: Path) -> None:
    atomic_copy_file(source, destination)


def _select_hyperliquid_training_datasets(
    datasets: list[PairDataset],
) -> tuple[list[PairDataset], pd.DataFrame]:
    """Select one deepest point-in-time Hyperliquid history per pair/timeframe."""

    records: list[dict[str, object]] = []
    for index, dataset in enumerate(datasets):
        frame = dataset.frame
        venue_values = frame.get("exchange", pd.Series(dtype=object)).dropna().astype(str)
        venue = venue_values.iloc[0].strip().lower() if not venue_values.empty else ""
        venue = "hyperliquid" if "hyperliquid" in venue else venue
        timeframe_values = pd.Series(dtype=object)
        for column in ("timeframe", "interval"):
            if column in frame.columns:
                timeframe_values = frame[column].dropna().astype(str)
                if not timeframe_values.empty:
                    break
        timeframe = (
            timeframe_values.iloc[0].strip().lower()
            if not timeframe_values.empty
            else ""
        )
        timeframe = {
            "5mins": "5m",
            "5min": "5m",
            "daily": "1d",
            "1day": "1d",
            "4hour": "4h",
            "1hour": "1h",
        }.get(timeframe.replace(" ", ""), timeframe)
        source_values = frame.get("source_path", pd.Series(dtype=object)).dropna().astype(str)
        source_path = source_values.iloc[0] if not source_values.empty else ""
        timestamps = pd.to_datetime(
            frame.get("timestamp", pd.Series(dtype=object)),
            utc=True,
            errors="coerce",
            format="mixed",
        ).dropna()
        records.append(
            {
                "dataset_index": index,
                "pair": dataset.pair,
                "source_venue": venue,
                "timeframe": timeframe,
                "source_path": source_path,
                "rows": len(frame),
                "unique_timestamps": int(timestamps.nunique()),
                "history_start": timestamps.min().isoformat() if not timestamps.empty else "",
                "history_end": timestamps.max().isoformat() if not timestamps.empty else "",
                "eligible_target_venue": venue == "hyperliquid",
            }
        )
    audit = pd.DataFrame(records)
    if audit.empty:
        return [], audit
    audit["selection_rank"] = 0
    audit["selection_status"] = "REJECTED_OUTSIDE_TARGET_VENUE"
    audit["blocker"] = "source_venue_not_hyperliquid"
    selected_indices: list[int] = []
    eligible = audit.loc[audit["eligible_target_venue"]].copy()
    for _, group in eligible.groupby(
        ["pair", "source_venue", "timeframe"], dropna=False, sort=True
    ):
        ranked = group.sort_values(
            ["unique_timestamps", "rows", "history_end", "source_path"],
            ascending=[False, False, False, True],
        )
        for rank, row_index in enumerate(ranked.index, start=1):
            audit.loc[row_index, "selection_rank"] = rank
            if rank == 1:
                audit.loc[row_index, "selection_status"] = "SELECTED_CANONICAL"
                audit.loc[row_index, "blocker"] = ""
                selected_indices.append(int(audit.loc[row_index, "dataset_index"]))
            else:
                audit.loc[row_index, "selection_status"] = (
                    "REJECTED_OVERLAPPING_HISTORY"
                )
                audit.loc[row_index, "blocker"] = (
                    "deeper_canonical_history_selected_for_same_pair_venue_timeframe"
                )
    audit["promotion_authority"] = False
    audit["testnet_order_authority"] = False
    audit["live_trading_authorized"] = False
    selected = [datasets[index] for index in selected_indices]
    return selected, audit.sort_values(
        ["eligible_target_venue", "pair", "timeframe", "selection_rank"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def _training_dataset_lineage(
    root: Path, input_path: Path | None
) -> tuple[Path, dict[str, object]]:
    data_ml = root / "data" / "ml"
    active_pointer_path = data_ml / "active_trade_dataset.json"
    active = _read_json(active_pointer_path)
    if input_path is None and active.get("status") == "ACTIVE_RESEARCH_DATASET":
        source = _rooted_path(root, active.get("active_dataset_path"))
        expected_hash = str(active.get("active_dataset_sha256", ""))
        dataset_id = str(active.get("dataset_id", ""))
        pointer_used = str(active_pointer_path.relative_to(root))
    else:
        source = input_path or data_ml / "trade_training_dataset.csv"
        expected_hash = ""
        dataset_id = ""
        pointer_used = ""
    if not source.is_file():
        raise SystemExit(f"trade gate dataset is missing: {source}")
    actual_hash = _sha256_file(source)
    if expected_hash and actual_hash != expected_hash:
        raise SystemExit("active trade dataset hash does not match its pointer")
    if not dataset_id:
        dataset_id = f"external_{actual_hash[:20]}"
    return source, {
        "training_dataset_id": dataset_id,
        "training_dataset_sha256": actual_hash,
        "training_dataset_path": str(
            source.relative_to(root) if source.is_relative_to(root) else source
        ),
        "training_dataset_pointer": pointer_used,
    }


def _snapshot_trade_gate_model(root: Path) -> Path | None:
    model_dir = root / "models" / "trade_gate"
    model_path = model_dir / "model.pkl"
    if not model_path.is_file():
        return None
    model_hash = _sha256_file(model_path)
    snapshot_dir = model_dir / "model_builds" / f"superseded_{model_hash[:20]}"
    receipt_path = snapshot_dir / "model_receipt.json"
    if receipt_path.is_file():
        return receipt_path
    artifacts = {
        "model.pkl": model_path,
        "metrics.json": model_dir / "metrics.json",
        "feature_schema.json": model_dir / "feature_schema.json",
        "model_walkforward_predictions.csv": root
        / "reports"
        / "ml"
        / "model_walkforward_predictions.csv",
        "model_backtest_comparison.csv": root
        / "reports"
        / "ml"
        / "model_backtest_comparison.csv",
        "model_gated_acceptance.csv": root
        / "reports"
        / "ml"
        / "model_gated_acceptance.csv",
        "ml_trade_filter_manifest.json": model_dir
        / "ml_trade_filter_manifest.json",
    }
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    preserved = []
    for name, source in artifacts.items():
        if source.is_file():
            _link_or_copy(source, snapshot_dir / name)
            preserved.append(name)
    _write_json(
        receipt_path,
        {
            "schema_version": "thewiz.trade_gate_model_build.v1",
            "model_id": f"superseded_{model_hash[:20]}",
            "model_sha256": model_hash,
            "status": "SUPERSEDED_PRESERVED",
            "preserved_at_utc": _now(),
            "artifacts": preserved,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    return receipt_path


def train_trade_gate(
    root: Path = ROOT,
    input_path: Path | None = None,
    walkforward_splits: int = 5,
    min_train_rows: int = 100,
    embargo_periods: int = 1,
) -> CommandResult:
    source, dataset_lineage = _training_dataset_lineage(root, input_path)
    raw_dataset = _read_csv(source)
    forbidden = sorted(POST_OUTCOME_MEMORY_COLUMNS.intersection(raw_dataset.columns))
    if forbidden:
        raise SystemExit(
            "trade gate dataset contains post-outcome aggregate features: "
            + ";".join(forbidden)
        )
    dataset = _model_dataset_from_hardened(raw_dataset)
    if dataset.empty:
        raise SystemExit(f"trade gate dataset is empty: {source}")
    ml_reports = root / "reports" / "ml"
    model_root = root / "models" / "trade_gate"
    study_dir = ml_reports / "trade_gate"
    superseded_model_receipt = _snapshot_trade_gate_model(root)
    try:
        paths = train_trade_filter_walkforward(
            dataset,
            output_dir=study_dir,
            n_splits=walkforward_splits,
            min_train_rows=min_train_rows,
            embargo_periods=embargo_periods,
        )
    except ValueError as exc:
        class_counts = dataset[TARGET_COLUMN].value_counts(dropna=False).to_dict() if TARGET_COLUMN in dataset else {}
        positive_trades = int(class_counts.get(1, 0))
        negative_trades = int(class_counts.get(0, 0))
        metrics = {
            "accepted": False,
            "best_model": "",
            "profit_factor_delta": 0.0,
            "filtered_profit_factor": 0.0,
            "filtered_sharpe": 0.0,
            "filtered_drawdown": 1.0,
            "sharpe_delta": 0.0,
            "drawdown_delta": 0.0,
            "median_take_rate": 0.0,
            "total_filtered_trades": 0,
            "score_buckets_monotonic": False,
            "blocker": f"train_trade_filter_walkforward_blocked: {exc}",
            "created_at": _now(),
            "label_source": "backtest_trained",
            "positive_trades": positive_trades,
            "negative_trades": negative_trades,
            "class_balance_blocker": "requires both profitable and unprofitable labels",
            **dataset_lineage,
        }
        model_dir = study_dir / "failure"
        model_dir.mkdir(parents=True, exist_ok=True)
        empty_predictions = model_dir / "model_walkforward_predictions.csv"
        _write_json(model_dir / "train_failure.json", {"error": str(exc), "source": str(source), "class_counts": class_counts})
        failure_predictions = ml_reports / "model_walkforward_predictions.csv"
        empty_backtest_summary = ml_reports / "model_backtest_comparison.csv"
        empty_score_bucket = ml_reports / "score_bucket_report.csv"
        empty_pair_conc = ml_reports / "model_pair_concentration.csv"
        empty_gain_conc = ml_reports / "model_gain_concentration.csv"
        empty_selection_leaderboard = (
            ml_reports / "model_selection_leaderboard.csv"
        )
        _write_csv(pd.DataFrame(), empty_predictions)
        _write_csv(pd.DataFrame(), model_dir / "model_backtest_comparison.csv")
        _write_csv(pd.DataFrame(), model_dir / "score_bucket_report.csv")
        _write_csv(pd.DataFrame(), model_dir / "model_pair_concentration.csv")
        _write_csv(pd.DataFrame(), failure_predictions)
        _write_csv(pd.DataFrame(), empty_backtest_summary)
        _write_csv(pd.DataFrame(), empty_score_bucket)
        _write_csv(pd.DataFrame(), empty_pair_conc)
        _write_csv(pd.DataFrame(), empty_gain_conc)
        _write_csv(pd.DataFrame(), empty_selection_leaderboard)
        _write_json(model_dir / "train_failure.json", {"error": str(exc), "source": str(source), "class_counts": class_counts})
        feature_schema = {**_feature_schema(dataset), **dataset_lineage}
        _write_json(model_root / "feature_schema.json", feature_schema)
        _write_json(model_root / "metrics.json", metrics)
        _write_csv(
            _model_acceptance_frame(metrics),
            ml_reports / "model_gated_acceptance.csv",
        )
        return CommandResult(
            paths={
                "model": model_root / "model.pkl",
                "feature_schema": model_root / "feature_schema.json",
                "metrics": model_root / "metrics.json",
                "predictions": failure_predictions,
                "selection_leaderboard": empty_selection_leaderboard,
            },
            summary={"accepted": False, "best_model": "", "blocker": str(exc)},
        )
    model_root.mkdir(parents=True, exist_ok=True)
    model_path = model_root / "model.pkl"
    walkforward_manifest_path = model_root / "ml_trade_filter_manifest.json"
    _link_or_copy_atomic(paths["best_model"], model_path)
    _link_or_copy_atomic(paths["manifest"], walkforward_manifest_path)
    walkforward_manifest = _read_json(walkforward_manifest_path)
    walkforward_manifest_sha256 = _sha256_file(walkforward_manifest_path)
    summary = _read_csv(paths["summary"])
    predictions = _read_csv(paths["predictions"])
    selection_leaderboard = _read_csv(paths["selection_leaderboard"])
    metrics = {**_trade_gate_metrics(summary, predictions), **dataset_lineage}
    model_hash = _sha256_file(model_path)
    model_id = f"tradegate_{model_hash[:20]}"
    metrics["model_id"] = model_id
    metrics["model_sha256"] = model_hash
    metrics["walkforward_manifest_path"] = str(
        walkforward_manifest_path.relative_to(root)
    )
    metrics["walkforward_manifest_sha256"] = walkforward_manifest_sha256
    metrics["superseded_model_receipt"] = (
        str(superseded_model_receipt.relative_to(root))
        if superseded_model_receipt
        else ""
    )
    selected_model_predictions = _filter_predictions_to_model(
        predictions, str(metrics.get("best_model", ""))
    )
    selected_predictions = _filter_predictions_to_untouched_evaluation(
        selected_model_predictions
    )
    feature_schema = {
        **_feature_schema(dataset),
        **dataset_lineage,
        "model_id": model_id,
        "model_sha256": model_hash,
        "walkforward_manifest_path": str(
            walkforward_manifest_path.relative_to(root)
        ),
        "walkforward_manifest_sha256": walkforward_manifest_sha256,
        "selection_evaluation_boundary_scheme": str(
            walkforward_manifest.get(
                "selection_evaluation_boundary_scheme", ""
            )
        ),
        "chronology_gap_folds": walkforward_manifest.get(
            "chronology_gap_folds", []
        ),
        "selection_label_end_boundary": str(
            walkforward_manifest.get("selection_label_end_boundary", "")
        ),
        "untouched_evaluation_start_boundary": str(
            walkforward_manifest.get(
                "untouched_evaluation_start_boundary", ""
            )
        ),
    }
    acceptance = _model_acceptance_frame(metrics)
    bucket_report = _score_bucket_report(selected_predictions)
    concentration = _model_pair_concentration(selected_predictions)
    gain_concentration = _model_gain_concentration(selected_predictions)
    _write_json(model_root / "feature_schema.json", feature_schema)
    _write_json(model_root / "metrics.json", metrics)
    _write_json(
        model_root / "model_lineage.json",
        {
            "schema_version": "thewiz.trade_gate_model_build.v1",
            "model_id": model_id,
            "model_sha256": model_hash,
            "walkforward_manifest_path": str(
                walkforward_manifest_path.relative_to(root)
            ),
            "walkforward_manifest_sha256": walkforward_manifest_sha256,
            "selection_isolation_scheme": str(
                walkforward_manifest.get("selection_isolation_scheme", "")
            ),
            "selection_evaluation_boundary_scheme": str(
                walkforward_manifest.get(
                    "selection_evaluation_boundary_scheme", ""
                )
            ),
            "chronology_gap_folds": walkforward_manifest.get(
                "chronology_gap_folds", []
            ),
            "selection_label_end_boundary": str(
                walkforward_manifest.get("selection_label_end_boundary", "")
            ),
            "untouched_evaluation_start_boundary": str(
                walkforward_manifest.get(
                    "untouched_evaluation_start_boundary", ""
                )
            ),
            **dataset_lineage,
            "accepted": bool(metrics.get("accepted", False)),
            "created_at_utc": _now(),
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    _write_csv(summary, ml_reports / "model_backtest_comparison.csv")
    _write_csv(predictions, ml_reports / "model_walkforward_predictions.csv")
    _write_csv(
        selection_leaderboard,
        ml_reports / "model_selection_leaderboard.csv",
    )
    _write_csv(bucket_report, ml_reports / "score_bucket_report.csv")
    _write_csv(concentration, ml_reports / "model_pair_concentration.csv")
    _write_csv(gain_concentration, ml_reports / "model_gain_concentration.csv")
    _write_csv(acceptance, ml_reports / "model_gated_acceptance.csv")
    return CommandResult(
        paths={
            "model": model_path,
            "feature_schema": model_root / "feature_schema.json",
            "metrics": model_root / "metrics.json",
            "lineage": model_root / "model_lineage.json",
            "selection_leaderboard": (
                ml_reports / "model_selection_leaderboard.csv"
            ),
        },
        summary={
            "accepted": bool(metrics["accepted"]),
            "best_model": metrics.get("best_model", ""),
            "model_id": model_id,
            "training_dataset_id": dataset_lineage["training_dataset_id"],
        },
    )


def run_model_gated_backtest(root: Path = ROOT) -> CommandResult:
    ml_reports = root / "reports" / "ml"
    model_root = root / "models" / "trade_gate"
    all_predictions = _read_csv(ml_reports / "model_walkforward_predictions.csv")
    summary = _read_csv(ml_reports / "model_backtest_comparison.csv")
    existing_metrics = _read_json(model_root / "metrics.json")
    metrics = _trade_gate_metrics(summary, all_predictions)
    selection_leaderboard = model_selection_leaderboard(all_predictions)
    for key in (
        "model_id",
        "model_sha256",
        "training_dataset_id",
        "training_dataset_sha256",
        "training_dataset_path",
        "training_dataset_pointer",
        "superseded_model_receipt",
        "walkforward_manifest_path",
        "walkforward_manifest_sha256",
    ):
        if key in existing_metrics:
            metrics[key] = existing_metrics[key]
    selected_model_predictions = _filter_predictions_to_model(
        all_predictions, str(metrics.get("best_model", ""))
    )
    predictions = _filter_predictions_to_untouched_evaluation(
        selected_model_predictions
    )
    _write_json(model_root / "metrics.json", metrics)
    _write_csv(
        selection_leaderboard,
        ml_reports / "model_selection_leaderboard.csv",
    )
    _write_csv(
        _score_bucket_report(predictions), ml_reports / "score_bucket_report.csv"
    )
    _write_csv(
        _model_pair_concentration(predictions),
        ml_reports / "model_pair_concentration.csv",
    )
    _write_csv(
        _model_gain_concentration(predictions),
        ml_reports / "model_gain_concentration.csv",
    )
    if predictions.empty:
        blocker = str(
            metrics.get(
                "blocker",
                "selected model walk-forward predictions missing; run train-trade-gate first",
            )
        )
        acceptance = pd.DataFrame(
            [
                {
                    "accepted": False,
                    "reason": blocker,
                    "factor": "",
                    "raw_pf": 0.0,
                    "raw_sharpe": 0.0,
                    "raw_maxdd": 0.0,
                    "raw_trades": 0,
                    "model_pf": 0.0,
                    "model_sharpe": 0.0,
                    "model_maxdd": 0.0,
                    "model_trades": 0,
                    "median_pf": 0.0,
                    "median_sharpe": 0.0,
                    "median_maxdd": 0.0,
                    "median_trades": 0,
                    "confidence": "none",
                    "improvement_pf": 0.0,
                    "improvement_sharpe": 0.0,
                    "improvement_drawdown": 0.0,
                }
            ]
        )
        _write_csv(acceptance, ml_reports / "model_gated_acceptance.csv")
        _write_csv(predictions, ml_reports / "model_gated_backtest.csv")
        _write_csv(pd.DataFrame(), ml_reports / "model_failure_attribution.csv")
        model_gate_pair_support_report(root=root)
        return CommandResult(
            paths={
                "backtest": ml_reports / "model_gated_backtest.csv",
                "acceptance": ml_reports / "model_gated_acceptance.csv",
                "predictions": ml_reports / "model_walkforward_predictions.csv",
                "failures": ml_reports / "model_failure_attribution.csv",
                "pair_support": ml_reports / "model_gate_pair_support_report.csv",
                "score_buckets": ml_reports / "score_bucket_report.csv",
                "pair_concentration": ml_reports / "model_pair_concentration.csv",
                "gain_concentration": ml_reports / "model_gain_concentration.csv",
                "selection_leaderboard": (
                    ml_reports / "model_selection_leaderboard.csv"
                ),
                "metrics": model_root / "metrics.json",
            },
            summary={"accepted": False, "blocker": blocker},
        )
    comparison = _model_gated_comparison(predictions)
    acceptance = _model_gated_acceptance(comparison)
    failures = _model_failure_attribution(predictions, acceptance, root=root)
    _write_csv(comparison, ml_reports / "model_gated_backtest.csv")
    _write_csv(acceptance, ml_reports / "model_gated_acceptance.csv")
    _write_csv(failures, ml_reports / "model_failure_attribution.csv")
    pair_support = model_gate_pair_support_report(root=root)
    return CommandResult(
        paths={
            "backtest": ml_reports / "model_gated_backtest.csv",
            "acceptance": ml_reports / "model_gated_acceptance.csv",
            "predictions": ml_reports / "model_walkforward_predictions.csv",
            "failures": ml_reports / "model_failure_attribution.csv",
            "score_buckets": ml_reports / "score_bucket_report.csv",
            "pair_concentration": ml_reports / "model_pair_concentration.csv",
            "gain_concentration": ml_reports / "model_gain_concentration.csv",
            "selection_leaderboard": (
                ml_reports / "model_selection_leaderboard.csv"
            ),
            "metrics": model_root / "metrics.json",
            **pair_support.paths,
        },
        summary={"accepted": bool(acceptance["accepted"].iloc[0]) if not acceptance.empty else False},
    )


def export_trade_gate_model(root: Path = ROOT) -> CommandResult:
    acceptance = _read_csv(ML_REPORTS / "model_gated_acceptance.csv")
    accepted = bool(not acceptance.empty and acceptance.get("accepted", pd.Series([False])).astype(bool).iloc[0])
    MODELS.mkdir(parents=True, exist_ok=True)
    report = {
        "accepted": accepted,
        "onnx_exported": False,
        "int8_exported": False,
        "blocker": "" if accepted else "model_gated_backtest_not_accepted",
        "created_at": _now(),
    }
    if accepted:
        report["blocker"] = "onnx_export_not_implemented_for_current_sklearn_pipeline"
    _write_json(MODELS / "export_report.json", report)
    return CommandResult(paths={"export_report": MODELS / "export_report.json"}, summary=report)


def build_command_dashboard(root: Path = ROOT, *, refresh_profile: str = "deep") -> CommandResult:
    from quant_platform.wizard_control_plane import build_wizard_control_plane

    ACTIVE = root / "reports" / "active"
    DASHBOARD = root / "reports" / "dashboard"
    ML_REPORTS = root / "reports" / "ml"
    if refresh_profile not in DASHBOARD_REFRESH_PROFILES:
        raise ValueError(f"unsupported dashboard refresh profile: {refresh_profile}")
    DASHBOARD.mkdir(parents=True, exist_ok=True)
    requested_refresh_profile = refresh_profile
    effective_refresh_profile = refresh_profile
    storage = dashboard_storage_preflight(root)
    if refresh_profile == "deep" and not storage["ready"]:
        # A deep rebuild can create enough temporary artifacts to leave a mixed,
        # incomplete dashboard when the volume is nearly full. Reuse evidence
        # only, and make the blocked deep refresh unmistakable in the output.
        effective_refresh_profile = "monitor"
        refresh_status = _refresh_dashboard_dependencies(root, refresh_profile="monitor")
        storage_block = pd.DataFrame(
            [
                {
                    "refresh_profile": "deep",
                    "stage": "storage_preflight",
                    "status": "blocked_insufficient_free_space",
                    "reason": (
                        f"free_gib={storage['free_gib']};required_gib={storage['required_gib']};"
                        "deep_refresh_not_started"
                    ),
                    "evidence_path": str(root),
                    "timestamp_utc": _now(),
                }
            ],
            columns=DASHBOARD_REFRESH_COLUMNS,
        )
        refresh_status = pd.concat([storage_block, refresh_status], ignore_index=True)
        _write_csv(refresh_status, root / "reports" / "active" / "dashboard_refresh_status.csv")
    else:
        refresh_status = _refresh_dashboard_dependencies(root, refresh_profile=refresh_profile)
    route_result = build_venue_route_scorecard(root)
    wizard_control = build_wizard_control_plane(root=root)
    current_state(root)
    system_check(root)
    pair_universe = _read_csv(root / "data" / "processed" / "pair_universe.csv")
    venue_routes = _read_csv(route_result.paths["venue_route_scorecard"])
    route_recommendations = _normalize_venue_route_recommendations(
        _read_csv(route_result.paths["venue_route_recommendations"])
    )
    dashboard_universe = _attach_venue_route_recommendations(pair_universe, route_recommendations)
    wizard_ranking_ready = bool(wizard_control.summary.get("ready", False))
    dashboard_universe["wizard_ranking_ready"] = wizard_ranking_ready
    dashboard_universe["wizard_ranking_blocker"] = str(wizard_control.summary.get("blocker", ""))
    dashboard_universe["wizard_rank"] = (
        dashboard_universe["combined_score"].rank(method="first", ascending=False).astype("Int64")
        if wizard_ranking_ready and "combined_score" in dashboard_universe
        else pd.Series(pd.NA, index=dashboard_universe.index, dtype="Int64")
    )
    current = _read_csv(ACTIVE / "current_state.csv")
    model_acceptance = _read_csv(ML_REPORTS / "model_gated_acceptance.csv")
    system = _read_csv(ACTIVE / "system_check.csv")
    live = _live_signal_rows(dashboard_universe, model_acceptance)
    blocked = live[live["blocker"].astype(str) != ""].copy() if not live.empty else pd.DataFrame()
    data_health = _data_health_rows(system, dashboard_universe, route_recommendations)
    wizard_health = _read_csv(wizard_control.paths["wizard_control_plane_health"])
    if not wizard_health.empty:
        wizard_health_rows = pd.DataFrame(
            {
                "area": "wizard_" + wizard_health["check"].astype(str),
                "ready": wizard_health["status"].astype(str).eq("pass"),
                "blocker": wizard_health["blocker"].fillna("").astype(str),
                "evidence_path": wizard_health["evidence_path"].fillna("").astype(str),
            }
        )
        data_health = pd.concat([data_health, wizard_health_rows], ignore_index=True)
    paths = {
        "pair_universe": DASHBOARD / "pair_universe_dashboard.csv",
        "candidate_ranking": DASHBOARD / "candidate_ranking_dashboard.csv",
        "venue_route_scorecard": DASHBOARD / "venue_route_scorecard.csv",
        "venue_route_recommendations": DASHBOARD / "venue_route_recommendations.csv",
        "strategy_tests": DASHBOARD / "strategy_tests_dashboard.csv",
        "wizard_local_verification": DASHBOARD / "wizard_local_verification_dashboard.csv",
        "wizard_strategy_alignment": DASHBOARD / "wizard_strategy_alignment_dashboard.csv",
        "wizard_discovery": DASHBOARD / "wizard_discovery_dashboard.csv",
        "wizard_control_plane": DASHBOARD / "wizard_control_plane_health.csv",
        "wizard_sweep_settings_capture_queue": DASHBOARD / "wizard_sweep_settings_capture_queue.csv",
        "wizard_api_contract": DASHBOARD / "wizard_api_contract_report.csv",
        "wizard_config_lineage": DASHBOARD / "wizard_config_lineage_audit.csv",
        "wizard_freshness": DASHBOARD / "wizard_freshness_blockers.csv",
        "wizard_discovery_shortlist": DASHBOARD / "wizard_discovery_shortlist_dashboard.csv",
        "wizard_copula_discovery": DASHBOARD / "wizard_copula_discovery_dashboard.csv",
        "wizard_pair_detail_capture_queue": DASHBOARD / "wizard_pair_detail_capture_queue_dashboard.csv",
        "wizard_mode_matrix_capture_queue": DASHBOARD / "wizard_mode_matrix_capture_queue_dashboard.csv",
        "wizard_pair_settings_capture_validation": DASHBOARD / "wizard_pair_settings_capture_validation_dashboard.csv",
        "wizard_replay_handoff": DASHBOARD / "wizard_replay_handoff_dashboard.csv",
        "wizard_mode_replay_capability": DASHBOARD / "wizard_mode_replay_capability_dashboard.csv",
        "wizard_mode_comparison": DASHBOARD / "wizard_mode_comparison_dashboard.csv",
        "wizard_exploratory_cost_sensitivity": DASHBOARD / "wizard_exploratory_cost_sensitivity_dashboard.csv",
        "multi_venue_history_readiness": DASHBOARD / "multi_venue_history_readiness_dashboard.csv",
        "binance_spot_history_readiness": DASHBOARD / "binance_spot_history_readiness.csv",
        "binance_spot_pair_readiness": DASHBOARD / "binance_spot_pair_readiness.csv",
        "hyperliquid_wizard_hypothesis": DASHBOARD / "hyperliquid_wizard_hypothesis_dashboard.csv",
        "hyperliquid_wizard_mode_proofs": DASHBOARD / "hyperliquid_wizard_vendor_mode_proofs_dashboard.csv",
        "wizard_ou_v6_terminal_outcome": DASHBOARD / "wizard_ou_v6_terminal_outcome.csv",
        "wizard_ou_v6_failure_attribution": DASHBOARD
        / "wizard_ou_v6_failure_attribution.csv",
        "wizard_ou_v6_orientation_policy": DASHBOARD / "wizard_ou_v6_orientation_policy.csv",
        "hyperliquid_run_candidates": DASHBOARD / "hyperliquid_run_candidates.csv",
        "hyperliquid_authority": DASHBOARD / "hyperliquid_authority_state.csv",
        "hyperliquid_walkforward": DASHBOARD / "hyperliquid_walkforward_mode_summary.csv",
        "hyperliquid_auxiliary_4h_walkforward": DASHBOARD / "hyperliquid_auxiliary_4h_walkforward_mode_summary.csv",
        "hyperliquid_selection_controls": DASHBOARD / "hyperliquid_statistical_selection_controls.csv",
        "hyperliquid_research_family_controls": DASHBOARD / "hyperliquid_research_family_selection_controls.csv",
        "hyperliquid_research_family_registry": DASHBOARD / "hyperliquid_research_family_registry.csv",
        "teacher_council_decisions": DASHBOARD / "teacher_council_decisions.csv",
        "student_training_readiness": DASHBOARD / "student_training_readiness.csv",
        "student_mode_training_manifest": DASHBOARD / "student_mode_training_manifest.csv",
        "portfolio_critic": DASHBOARD / "portfolio_critic.csv",
        "hyperliquid_testnet_margin": DASHBOARD / "hyperliquid_testnet_margin_snapshot.csv",
        "hyperliquid_testnet_lifecycle": DASHBOARD / "hyperliquid_testnet_lifecycle_gate.csv",
        "hyperliquid_testnet_lifecycle_evidence": DASHBOARD
        / "hyperliquid_testnet_lifecycle_evidence_capture.csv",
        "current_wizard_hyperliquid_daily_run": DASHBOARD
        / "current_wizard_hyperliquid_daily_run_status.csv",
        "current_wizard_hyperliquid_completion_audit": DASHBOARD
        / "current_wizard_hyperliquid_completion_audit.csv",
        "hyperliquid_testnet_preflight": DASHBOARD / "hyperliquid_testnet_preflight.csv",
        "hyperliquid_evidence_cadence": DASHBOARD / "hyperliquid_evidence_cadence.csv",
        "exhaustive_wizard_api_refresh_delta": DASHBOARD
        / "exhaustive_wizard_api_refresh_delta.csv",
        "exhaustive_wizard_api_refresh_hyperliquid_mapping": DASHBOARD
        / "exhaustive_wizard_api_refresh_hyperliquid_mapping.csv",
        "exhaustive_wizard_api_refresh_pair_detail_queue": DASHBOARD
        / "exhaustive_wizard_api_refresh_pair_detail_queue.csv",
        "exhaustive_wizard_pair_detail_acquisition_plan": DASHBOARD
        / "exhaustive_wizard_pair_detail_acquisition_plan.csv",
        "exhaustive_wizard_api_refresh_validation": DASHBOARD
        / "exhaustive_wizard_api_refresh_validation.csv",
        "wizard_pair_detail_api_manifest": DASHBOARD
        / "wizard_pair_detail_api_pilot_manifest.csv",
        "wizard_pair_detail_api_fields": DASHBOARD
        / "wizard_pair_detail_api_pilot_fields.csv",
        "wizard_pair_detail_api_coverage": DASHBOARD
        / "wizard_pair_detail_api_pilot_coverage.csv",
        "wizard_pair_detail_api_attempt_history": DASHBOARD
        / "wizard_pair_detail_api_pilot_attempt_history.csv",
        "current_wizard_pair_detail_status": DASHBOARD
        / "current_wizard_pair_detail_status.csv",
        "current_wizard_hyperliquid_experiments": DASHBOARD
        / "current_wizard_hyperliquid_experiment_matrix.csv",
        "current_wizard_hyperliquid_pair_history_queue": DASHBOARD
        / "current_wizard_hyperliquid_pair_history_queue.csv",
        "current_wizard_hyperliquid_asset_fetch_queue": DASHBOARD
        / "current_wizard_hyperliquid_asset_fetch_queue.csv",
        "current_wizard_hyperliquid_handoff_validation": DASHBOARD
        / "current_wizard_hyperliquid_handoff_validation.csv",
        "current_wizard_hyperliquid_asset_history": DASHBOARD
        / "current_wizard_hyperliquid_asset_history_results.csv",
        "current_wizard_hyperliquid_pair_history": DASHBOARD
        / "current_wizard_hyperliquid_pair_history_results.csv",
        "current_wizard_hyperliquid_history_validation": DASHBOARD
        / "current_wizard_hyperliquid_history_validation.csv",
        "current_wizard_hyperliquid_canonical_replay": DASHBOARD
        / "current_wizard_hyperliquid_canonical_replay.csv",
        "current_wizard_hyperliquid_canonical_replay_ranked": DASHBOARD
        / "current_wizard_hyperliquid_canonical_replay_ranked.csv",
        "current_wizard_hyperliquid_replay_pair_status": DASHBOARD
        / "current_wizard_hyperliquid_replay_pair_status.csv",
        "current_wizard_hyperliquid_replay_validation": DASHBOARD
        / "current_wizard_hyperliquid_replay_validation.csv",
        "current_wizard_hyperliquid_funding_assets": DASHBOARD
        / "current_wizard_hyperliquid_funding_asset_results.csv",
        "current_wizard_hyperliquid_pair_cost_evidence": DASHBOARD
        / "current_wizard_hyperliquid_pair_cost_evidence.csv",
        "current_wizard_hyperliquid_experiment_cost_readiness": DASHBOARD
        / "current_wizard_hyperliquid_experiment_cost_readiness.csv",
        "current_wizard_hyperliquid_cost_validation": DASHBOARD
        / "current_wizard_hyperliquid_cost_validation.csv",
        "current_wizard_hyperliquid_observed_cost_replay": DASHBOARD
        / "current_wizard_hyperliquid_observed_cost_replay.csv",
        "current_wizard_hyperliquid_observed_cost_replay_ranked": DASHBOARD
        / "current_wizard_hyperliquid_observed_cost_replay_ranked.csv",
        "current_wizard_hyperliquid_observed_cost_replay_validation": DASHBOARD
        / "current_wizard_hyperliquid_observed_cost_replay_validation.csv",
        "current_wizard_hyperliquid_walkforward_status": DASHBOARD
        / "current_wizard_hyperliquid_walkforward_status.csv",
        "current_wizard_hyperliquid_walkforward_candidates": DASHBOARD
        / "current_wizard_hyperliquid_walkforward_candidates.csv",
        "current_wizard_hyperliquid_walkforward_validation": DASHBOARD
        / "current_wizard_hyperliquid_walkforward_validation.csv",
        "current_wizard_hyperliquid_regime_status": DASHBOARD
        / "current_wizard_hyperliquid_regime_status.csv",
        "current_wizard_hyperliquid_regime_candidates": DASHBOARD
        / "current_wizard_hyperliquid_regime_candidates.csv",
        "current_wizard_hyperliquid_regime_validation": DASHBOARD
        / "current_wizard_hyperliquid_regime_validation.csv",
        "current_wizard_hyperliquid_robustness_status": DASHBOARD
        / "current_wizard_hyperliquid_robustness_status.csv",
        "current_wizard_hyperliquid_robustness_candidates": DASHBOARD
        / "current_wizard_hyperliquid_robustness_candidates.csv",
        "current_wizard_hyperliquid_robustness_validation": DASHBOARD
        / "current_wizard_hyperliquid_robustness_validation.csv",
        "current_wizard_hyperliquid_concentration_status": DASHBOARD
        / "current_wizard_hyperliquid_concentration_status.csv",
        "current_wizard_hyperliquid_concentration_cohorts": DASHBOARD
        / "current_wizard_hyperliquid_concentration_cohorts.csv",
        "current_wizard_hyperliquid_concentration_dimensions": DASHBOARD
        / "current_wizard_hyperliquid_concentration_dimensions.csv",
        "current_wizard_hyperliquid_concentration_contributors": DASHBOARD
        / "current_wizard_hyperliquid_concentration_contributors.csv",
        "current_wizard_hyperliquid_concentration_validation": DASHBOARD
        / "current_wizard_hyperliquid_concentration_validation.csv",
        "current_wizard_hyperliquid_failure_attribution": DASHBOARD
        / "current_wizard_hyperliquid_failure_attribution.csv",
        "current_wizard_hyperliquid_failure_attribution_summary": DASHBOARD
        / "current_wizard_hyperliquid_failure_attribution_summary.csv",
        "current_wizard_hyperliquid_failure_attribution_validation": DASHBOARD
        / "current_wizard_hyperliquid_failure_attribution_validation.csv",
        "current_wizard_hyperliquid_leverage_status": DASHBOARD
        / "current_wizard_hyperliquid_leverage_status.csv",
        "current_wizard_hyperliquid_leverage_candidates": DASHBOARD
        / "current_wizard_hyperliquid_leverage_candidates.csv",
        "current_wizard_hyperliquid_leverage_scenarios": DASHBOARD
        / "current_wizard_hyperliquid_leverage_scenarios.csv",
        "current_wizard_hyperliquid_leverage_validation": DASHBOARD
        / "current_wizard_hyperliquid_leverage_validation.csv",
        "current_wizard_hyperliquid_learning_validation": DASHBOARD
        / "current_wizard_hyperliquid_learning_validation.csv",
        "current_wizard_hyperliquid_chain_validation": DASHBOARD
        / "current_wizard_hyperliquid_chain_validation.csv",
        "current_wizard_hyperliquid_operating_cadence": DASHBOARD
        / "current_wizard_hyperliquid_operating_cadence.csv",
        "current_wizard_hyperliquid_operating_cadence_validation": DASHBOARD
        / "current_wizard_hyperliquid_operating_cadence_validation.csv",
        "current_wizard_hyperliquid_live_lock": DASHBOARD
        / "current_wizard_hyperliquid_live_lock.csv",
        "current_wizard_hyperliquid_storage_efficiency": DASHBOARD
        / "current_wizard_hyperliquid_storage_efficiency.csv",
        "current_wizard_hyperliquid_storage_reclamation_plan": DASHBOARD
        / "current_wizard_hyperliquid_storage_reclamation_plan.csv",
        "current_wizard_hyperliquid_storage_reclamation_validation": DASHBOARD
        / "current_wizard_hyperliquid_storage_reclamation_validation.csv",
        "current_wizard_hyperliquid_archive_copy_receipt": DASHBOARD
        / "current_wizard_hyperliquid_archive_copy_receipt.csv",
        "current_wizard_hyperliquid_archive_copy_validation": DASHBOARD
        / "current_wizard_hyperliquid_archive_copy_validation.csv",
        "current_wizard_hyperliquid_archive_release_plan": DASHBOARD
        / "current_wizard_hyperliquid_archive_release_plan.csv",
        "current_wizard_hyperliquid_archive_release_validation": DASHBOARD
        / "current_wizard_hyperliquid_archive_release_validation.csv",
        "current_wizard_hyperliquid_testnet_protocol_scenarios": DASHBOARD
        / "current_wizard_hyperliquid_testnet_protocol_scenarios.csv",
        "current_wizard_hyperliquid_testnet_protocol_transitions": DASHBOARD
        / "current_wizard_hyperliquid_testnet_protocol_transitions.csv",
        "current_wizard_hyperliquid_testnet_protocol_candidate_coverage": DASHBOARD
        / "current_wizard_hyperliquid_testnet_protocol_candidate_coverage.csv",
        "current_wizard_hyperliquid_testnet_protocol_validation": DASHBOARD
        / "current_wizard_hyperliquid_testnet_protocol_validation.csv",
        "current_wizard_ou_optimal_overlay_ledger": DASHBOARD
        / "current_wizard_ou_optimal_overlay_ledger.csv",
        "current_wizard_ou_optimal_overlay_coverage": DASHBOARD
        / "current_wizard_ou_optimal_overlay_coverage.csv",
        "current_wizard_ou_optimal_overlay_validation": DASHBOARD
        / "current_wizard_ou_optimal_overlay_validation.csv",
        "exhaustive_hyperliquid_concentration": DASHBOARD
        / "exhaustive_wizard_hyperliquid_concentration_status.csv",
        "exhaustive_hyperliquid_leverage_status": DASHBOARD
        / "exhaustive_wizard_hyperliquid_leverage_status.csv",
        "exhaustive_hyperliquid_leverage_candidates": DASHBOARD
        / "exhaustive_wizard_hyperliquid_leverage_candidates.csv",
        "exhaustive_hyperliquid_leverage_scenarios": DASHBOARD
        / "exhaustive_wizard_hyperliquid_leverage_scenarios.csv",
        "exhaustive_hyperliquid_learning": DASHBOARD
        / "exhaustive_wizard_hyperliquid_learning_ledger.csv",
        "model_training": DASHBOARD / "model_training_dashboard.csv",
        "model_gate_pair_support": DASHBOARD / "model_gate_pair_support.csv",
        "youtube_brain_status": DASHBOARD / "youtube_brain_status.csv",
        "youtube_brain_hypotheses": DASHBOARD / "youtube_brain_hypotheses.csv",
        "youtube_brain_replication": DASHBOARD / "youtube_brain_replication_scorecard.csv",
        "youtube_hypothesis_validation": DASHBOARD / "youtube_hypothesis_validation.csv",
        "youtube_hypothesis_regime_matrix": DASHBOARD / "youtube_hypothesis_regime_matrix.csv",
        "youtube_brain_summary": DASHBOARD / "youtube_brain.md",
        "feature_readiness": DASHBOARD / "feature_readiness_dashboard.csv",
        "orchestrator_run_status": DASHBOARD / "orchestrator_run_status.csv",
        "supreme_team_checkpoint": DASHBOARD / "supreme_team_checkpoint.csv",
        "project_spine_audit": DASHBOARD / "project_spine_audit.md",
        "rl_research_status": DASHBOARD / "rl_research_status.csv",
        "rl_execution_backtest": DASHBOARD / "rl_execution_backtest.csv",
        "rl_position_sizing": DASHBOARD / "rl_position_sizing_results.csv",
        "rl_strategy_selector": DASHBOARD / "rl_strategy_selector_results.csv",
        "rl_blocked_actions": DASHBOARD / "rl_blocked_actions.csv",
        "rl_acceptance": DASHBOARD / "rl_acceptance_report.csv",
        "quantization_readiness": DASHBOARD / "quantization_readiness.csv",
        "live_signals": DASHBOARD / "live_signals_dashboard.csv",
        "blocked_trades": DASHBOARD / "blocked_trades_dashboard.csv",
        "data_health": DASHBOARD / "data_health_dashboard.csv",
        "refresh_status": DASHBOARD / "dashboard_refresh_status.csv",
        "api_credit_usage": DASHBOARD / "api_credit_usage_dashboard.csv",
        "apify_cost_audit": DASHBOARD / "apify_cost_audit.csv",
        "paper_candidate_shortlist": DASHBOARD / "paper_candidate_shortlist.csv",
        "focused_paper_validation": DASHBOARD / "focused_paper_validation.csv",
        "command_center": DASHBOARD / "command_center.md",
        "scoring_audit": DASHBOARD / "scoring_audit.csv",
        "brain_readiness_report": DASHBOARD / "brain_readiness_report.csv",
        "brain_candidate_rollup": DASHBOARD / "brain_candidate_rollup.csv",
        "brain_readiness_trend": DASHBOARD / "brain_readiness_trend.csv",
        "wizard_readiness": DASHBOARD / "wizard_readiness.csv",
        "native_readiness": DASHBOARD / "native_readiness.csv",
        "overall_readiness": DASHBOARD / "overall_readiness.csv",
        "paper_status": DASHBOARD / "paper_status.csv",
        "three_brain_contract": DASHBOARD / "three_brain_contract.json",
        "wizard_dashboard_investigation": DASHBOARD / "wizard_dashboard_investigation.csv",
        "forward_walk_strength_report": DASHBOARD / "forward_walk_strength_report.csv",
        "native_quality_report": DASHBOARD / "native_quality_report.csv",
        "native_candidate_diagnosis": DASHBOARD / "native_candidate_diagnosis.csv",
        "wizard_candidate_repair_report": DASHBOARD / "wizard_candidate_repair_report.csv",
        "wizard_hourly_priority_capture_queue": DASHBOARD / "wizard_hourly_priority_capture_queue.csv",
        "native_focus_repair_report": DASHBOARD / "native_focus_repair_report.csv",
        "wizard_hourly_repair_targets": DASHBOARD / "wizard_hourly_repair_targets.csv",
        "candidate_repair_loop": DASHBOARD / "candidate_repair_loop.csv",
        "shadow_rl_usefulness_report": DASHBOARD / "shadow_rl_usefulness_report.csv",
        "shadow_rl_candidate_set": DASHBOARD / "shadow_rl_candidate_set.csv",
        "shadow_rl_intake_report": DASHBOARD / "shadow_rl_intake_report.csv",
        "overall_arbitration_scorecard": DASHBOARD / "overall_arbitration_scorecard.csv",
        "blocker_persistence_report": DASHBOARD / "blocker_persistence_report.csv",
        "verified_paper_outcome_report": DASHBOARD / "verified_paper_outcome_report.csv",
        "overall_brain_summary": DASHBOARD / "overall_brain_summary.csv",
        "promotion_ladder": DASHBOARD / "promotion_ladder.csv",
        "orchestrator_status": DASHBOARD / "orchestrator_status.csv",
        "base_rl_training_report": DASHBOARD / "base_rl_training_report.csv",
        "base_rl_evaluation_report": DASHBOARD / "base_rl_evaluation_report.csv",
        "base_rl_pair_coverage": DASHBOARD / "base_rl_pair_coverage.csv",
        "base_rl_blocked_actions": DASHBOARD / "base_rl_blocked_actions.csv",
        "base_rl_paper_handoff_status": DASHBOARD / "base_rl_paper_handoff_status.csv",
        "base_rl_comparison_report": DASHBOARD / "base_rl_comparison_report.csv",
        "base_rl_promotion_readiness": DASHBOARD / "base_rl_promotion_readiness.csv",
        "base_rl_promotion_decision": DASHBOARD / "base_rl_promotion_decision.csv",
        "base_rl_blocker_delta": DASHBOARD / "base_rl_blocker_delta_report.csv",
        "base_rl_pair_blocker": DASHBOARD / "base_rl_pair_blocker_report.csv",
    }
    _write_csv(dashboard_universe, paths["pair_universe"])
    ranking = (
        dashboard_universe.sort_values("combined_score", ascending=False)
        if wizard_ranking_ready and "combined_score" in dashboard_universe
        else dashboard_universe
    )
    _write_csv(ranking, paths["candidate_ranking"])
    _write_csv(venue_routes, paths["venue_route_scorecard"])
    _write_csv(route_recommendations, paths["venue_route_recommendations"])
    _write_csv(_strategy_tests_dashboard(root), paths["strategy_tests"])
    batch_verification = root / "reports" / "active" / "wizard_local_verification_batch.csv"
    single_verification = root / "reports" / "active" / "bnb_stx_daily_320_static_spread_after_cost.csv"
    _write_csv(_read_csv(batch_verification if batch_verification.exists() else single_verification), paths["wizard_local_verification"])
    _write_csv(_read_csv(root / "reports" / "active" / "wizard_strategy_alignment_report.csv"), paths["wizard_strategy_alignment"])
    _write_csv(_read_csv(root / "reports" / "active" / "wizard_discovery_current.csv"), paths["wizard_discovery"])
    _write_csv(wizard_health, paths["wizard_control_plane"])
    _write_csv(
        _read_csv(wizard_control.paths["wizard_sweep_settings_capture_queue"]),
        paths["wizard_sweep_settings_capture_queue"],
    )
    _write_csv(_read_csv(wizard_control.paths["wizard_api_contract_report"]), paths["wizard_api_contract"])
    _write_csv(_read_csv(wizard_control.paths["wizard_config_lineage_audit"]), paths["wizard_config_lineage"])
    _write_csv(_read_csv(wizard_control.paths["wizard_freshness_blockers"]), paths["wizard_freshness"])
    _write_csv(_read_csv(root / "reports" / "active" / "wizard_discovery_shortlist.csv"), paths["wizard_discovery_shortlist"])
    _write_csv(_read_csv(root / "reports" / "active" / "wizard_copula_discovery_triage.csv"), paths["wizard_copula_discovery"])
    _write_csv(_read_csv(root / "reports" / "active" / "wizard_pair_detail_capture_queue.csv"), paths["wizard_pair_detail_capture_queue"])
    _write_csv(_read_csv(root / "reports" / "active" / "wizard_mode_matrix_capture_queue.csv"), paths["wizard_mode_matrix_capture_queue"])
    _write_csv(_read_csv(root / "reports" / "active" / "wizard_pair_settings_capture_validation.csv"), paths["wizard_pair_settings_capture_validation"])
    _write_csv(_read_csv(root / "reports" / "active" / "wizard_replay_handoff.csv"), paths["wizard_replay_handoff"])
    _write_csv(_read_csv(root / "reports" / "active" / "wizard_mode_replay_capability.csv"), paths["wizard_mode_replay_capability"])
    _write_csv(_read_csv(root / "reports" / "active" / "wizard_mode_comparison.csv"), paths["wizard_mode_comparison"])
    _write_csv(_read_csv(root / "reports" / "active" / "wizard_exploratory_cost_sensitivity.csv"), paths["wizard_exploratory_cost_sensitivity"])
    _write_csv(_read_csv(current_multi_venue_history_readiness_path(root)), paths["multi_venue_history_readiness"])
    _write_csv(_read_csv(root / "reports" / "active" / "binance_spot_history_readiness.csv"), paths["binance_spot_history_readiness"])
    _write_csv(_read_csv(root / "reports" / "active" / "binance_spot_pair_readiness.csv"), paths["binance_spot_pair_readiness"])
    _write_csv(_read_csv(root / "reports" / "active" / "hyperliquid_wizard_hypothesis_queue.csv"), paths["hyperliquid_wizard_hypothesis"])
    _write_csv(_read_csv(root / "reports" / "active" / "hyperliquid_wizard_vendor_mode_proofs.csv"), paths["hyperliquid_wizard_mode_proofs"])
    _write_csv(
        _read_csv(root / "reports" / "active" / "wizard_ou_v6_terminal_outcome.csv"),
        paths["wizard_ou_v6_terminal_outcome"],
    )
    _write_csv(
        _read_csv(root / "reports" / "active" / "wizard_ou_v6_failure_attribution.csv"),
        paths["wizard_ou_v6_failure_attribution"],
    )
    _write_csv(
        _read_csv(root / "reports" / "active" / "wizard_ou_v6_orientation_policy.csv"),
        paths["wizard_ou_v6_orientation_policy"],
    )
    _write_csv(_read_csv(root / "reports" / "active" / "hyperliquid_run_candidates.csv"), paths["hyperliquid_run_candidates"])
    _write_csv(_read_csv(root / "reports" / "active" / "hyperliquid_authority_state.csv"), paths["hyperliquid_authority"])
    teacher_council_dir = root / "reports" / "orchestration" / "teacher_council"
    _write_csv(_read_csv(teacher_council_dir / "walkforward_mode_summary.csv"), paths["hyperliquid_walkforward"])
    _write_csv(
        _read_csv(teacher_council_dir / "auxiliary_4h_walkforward_mode_summary.csv"),
        paths["hyperliquid_auxiliary_4h_walkforward"],
    )
    _write_csv(_read_csv(teacher_council_dir / "statistical_selection_controls.csv"), paths["hyperliquid_selection_controls"])
    _write_csv(_read_csv(teacher_council_dir / "research_family_selection_controls.csv"), paths["hyperliquid_research_family_controls"])
    _write_csv(_read_csv(teacher_council_dir / "research_family_registry.csv"), paths["hyperliquid_research_family_registry"])
    _write_csv(_read_csv(teacher_council_dir / "council_decisions.csv"), paths["teacher_council_decisions"])
    _write_csv(_read_csv(teacher_council_dir / "student_training_readiness.csv"), paths["student_training_readiness"])
    _write_csv(_read_csv(teacher_council_dir / "student_mode_training_manifest.csv"), paths["student_mode_training_manifest"])
    _write_csv(_read_csv(teacher_council_dir / "portfolio_critic.csv"), paths["portfolio_critic"])
    _write_csv(_read_csv(root / "reports" / "active" / "hyperliquid_testnet_margin_snapshot.csv"), paths["hyperliquid_testnet_margin"])
    _write_csv(_read_csv(root / "reports" / "active" / "hyperliquid_testnet_lifecycle_gate.csv"), paths["hyperliquid_testnet_lifecycle"])
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "hyperliquid_testnet_lifecycle_evidence_capture.csv"
        ),
        paths["hyperliquid_testnet_lifecycle_evidence"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "current_wizard_hyperliquid_daily_run_status.csv"
        ),
        paths["current_wizard_hyperliquid_daily_run"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "current_wizard_hyperliquid_completion_audit.csv"
        ),
        paths["current_wizard_hyperliquid_completion_audit"],
    )
    _write_csv(
        _read_csv(root / "reports" / "active" / "hyperliquid_testnet_preflight.csv"),
        paths["hyperliquid_testnet_preflight"],
    )
    _write_csv(
        _read_csv(root / "reports" / "active" / "hyperliquid_evidence_cadence.csv"),
        paths["hyperliquid_evidence_cadence"],
    )
    _write_csv(
        _read_csv(
            root / "reports" / "active" / "exhaustive_wizard_api_refresh_delta.csv"
        ),
        paths["exhaustive_wizard_api_refresh_delta"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "exhaustive_wizard_api_refresh_hyperliquid_mapping.csv"
        ),
        paths["exhaustive_wizard_api_refresh_hyperliquid_mapping"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "exhaustive_wizard_api_refresh_pair_detail_queue.csv"
        ),
        paths["exhaustive_wizard_api_refresh_pair_detail_queue"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "exhaustive_wizard_pair_detail_acquisition_plan.csv"
        ),
        paths["exhaustive_wizard_pair_detail_acquisition_plan"],
    )
    _write_csv(
        _read_csv(
            root / "reports" / "active" / "exhaustive_wizard_api_refresh_validation.csv"
        ),
        paths["exhaustive_wizard_api_refresh_validation"],
    )
    _write_csv(
        _read_csv(
            root / "reports" / "active" / "wizard_pair_detail_api_pilot_manifest.csv"
        ),
        paths["wizard_pair_detail_api_manifest"],
    )
    _write_csv(
        _read_csv(
            root / "reports" / "active" / "wizard_pair_detail_api_pilot_fields.csv"
        ),
        paths["wizard_pair_detail_api_fields"],
    )
    _write_csv(
        _read_csv(
            root / "reports" / "active" / "wizard_pair_detail_api_pilot_coverage.csv"
        ),
        paths["wizard_pair_detail_api_coverage"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "wizard_pair_detail_api_pilot_attempt_history.csv"
        ),
        paths["wizard_pair_detail_api_attempt_history"],
    )
    _write_csv(
        _read_csv(
            root / "reports" / "active" / "current_wizard_pair_detail_status.csv"
        ),
        paths["current_wizard_pair_detail_status"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "current_wizard_hyperliquid_experiment_matrix.csv"
        ),
        paths["current_wizard_hyperliquid_experiments"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "current_wizard_hyperliquid_pair_history_queue.csv"
        ),
        paths["current_wizard_hyperliquid_pair_history_queue"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "current_wizard_hyperliquid_asset_fetch_queue.csv"
        ),
        paths["current_wizard_hyperliquid_asset_fetch_queue"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "current_wizard_hyperliquid_handoff_validation.csv"
        ),
        paths["current_wizard_hyperliquid_handoff_validation"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "current_wizard_hyperliquid_asset_history_results.csv"
        ),
        paths["current_wizard_hyperliquid_asset_history"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "current_wizard_hyperliquid_pair_history_results.csv"
        ),
        paths["current_wizard_hyperliquid_pair_history"],
    )
    _write_csv(
        _read_csv(
            root / "reports" / "active" / "current_wizard_hyperliquid_history_validation.csv"
        ),
        paths["current_wizard_hyperliquid_history_validation"],
    )
    _write_csv(
        _read_csv(
            root / "reports" / "active" / "current_wizard_hyperliquid_canonical_replay.csv"
        ),
        paths["current_wizard_hyperliquid_canonical_replay"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "current_wizard_hyperliquid_canonical_replay_ranked.csv"
        ),
        paths["current_wizard_hyperliquid_canonical_replay_ranked"],
    )
    _write_csv(
        _read_csv(
            root / "reports" / "active" / "current_wizard_hyperliquid_replay_pair_status.csv"
        ),
        paths["current_wizard_hyperliquid_replay_pair_status"],
    )
    _write_csv(
        _read_csv(
            root / "reports" / "active" / "current_wizard_hyperliquid_replay_validation.csv"
        ),
        paths["current_wizard_hyperliquid_replay_validation"],
    )
    _write_csv(
        _read_csv(
            root / "reports" / "active" / "current_wizard_hyperliquid_funding_asset_results.csv"
        ),
        paths["current_wizard_hyperliquid_funding_assets"],
    )
    _write_csv(
        _read_csv(
            root / "reports" / "active" / "current_wizard_hyperliquid_pair_cost_evidence.csv"
        ),
        paths["current_wizard_hyperliquid_pair_cost_evidence"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "current_wizard_hyperliquid_experiment_cost_readiness.csv"
        ),
        paths["current_wizard_hyperliquid_experiment_cost_readiness"],
    )
    _write_csv(
        _read_csv(
            root / "reports" / "active" / "current_wizard_hyperliquid_cost_validation.csv"
        ),
        paths["current_wizard_hyperliquid_cost_validation"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "current_wizard_hyperliquid_observed_cost_replay.csv"
        ),
        paths["current_wizard_hyperliquid_observed_cost_replay"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "current_wizard_hyperliquid_observed_cost_replay_ranked.csv"
        ),
        paths["current_wizard_hyperliquid_observed_cost_replay_ranked"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "current_wizard_hyperliquid_observed_cost_replay_validation.csv"
        ),
        paths["current_wizard_hyperliquid_observed_cost_replay_validation"],
    )
    for source_name, path_key in (
        (
            "current_wizard_hyperliquid_walkforward_status.csv",
            "current_wizard_hyperliquid_walkforward_status",
        ),
        (
            "current_wizard_hyperliquid_walkforward_candidates.csv",
            "current_wizard_hyperliquid_walkforward_candidates",
        ),
        (
            "current_wizard_hyperliquid_walkforward_validation.csv",
            "current_wizard_hyperliquid_walkforward_validation",
        ),
        (
            "current_wizard_hyperliquid_regime_status.csv",
            "current_wizard_hyperliquid_regime_status",
        ),
        (
            "current_wizard_hyperliquid_regime_candidates.csv",
            "current_wizard_hyperliquid_regime_candidates",
        ),
        (
            "current_wizard_hyperliquid_regime_validation.csv",
            "current_wizard_hyperliquid_regime_validation",
        ),
        (
            "current_wizard_hyperliquid_robustness_status.csv",
            "current_wizard_hyperliquid_robustness_status",
        ),
        (
            "current_wizard_hyperliquid_robustness_candidates.csv",
            "current_wizard_hyperliquid_robustness_candidates",
        ),
        (
            "current_wizard_hyperliquid_robustness_validation.csv",
            "current_wizard_hyperliquid_robustness_validation",
        ),
        (
            "current_wizard_hyperliquid_concentration_status.csv",
            "current_wizard_hyperliquid_concentration_status",
        ),
        (
            "current_wizard_hyperliquid_concentration_cohorts.csv",
            "current_wizard_hyperliquid_concentration_cohorts",
        ),
        (
            "current_wizard_hyperliquid_concentration_dimensions.csv",
            "current_wizard_hyperliquid_concentration_dimensions",
        ),
        (
            "current_wizard_hyperliquid_concentration_contributors.csv",
            "current_wizard_hyperliquid_concentration_contributors",
        ),
        (
            "current_wizard_hyperliquid_concentration_validation.csv",
            "current_wizard_hyperliquid_concentration_validation",
        ),
        (
            "current_wizard_hyperliquid_failure_attribution.csv",
            "current_wizard_hyperliquid_failure_attribution",
        ),
        (
            "current_wizard_hyperliquid_failure_attribution_summary.csv",
            "current_wizard_hyperliquid_failure_attribution_summary",
        ),
        (
            "current_wizard_hyperliquid_failure_attribution_validation.csv",
            "current_wizard_hyperliquid_failure_attribution_validation",
        ),
        (
            "current_wizard_hyperliquid_leverage_status.csv",
            "current_wizard_hyperliquid_leverage_status",
        ),
        (
            "current_wizard_hyperliquid_leverage_candidates.csv",
            "current_wizard_hyperliquid_leverage_candidates",
        ),
        (
            "current_wizard_hyperliquid_leverage_scenarios.csv",
            "current_wizard_hyperliquid_leverage_scenarios",
        ),
        (
            "current_wizard_hyperliquid_leverage_validation.csv",
            "current_wizard_hyperliquid_leverage_validation",
        ),
        (
            "current_wizard_hyperliquid_learning_validation.csv",
            "current_wizard_hyperliquid_learning_validation",
        ),
        (
            "current_wizard_hyperliquid_chain_validation.csv",
            "current_wizard_hyperliquid_chain_validation",
        ),
        (
            "current_wizard_hyperliquid_operating_cadence.csv",
            "current_wizard_hyperliquid_operating_cadence",
        ),
        (
            "current_wizard_hyperliquid_operating_cadence_validation.csv",
            "current_wizard_hyperliquid_operating_cadence_validation",
        ),
        (
            "current_wizard_hyperliquid_live_lock.csv",
            "current_wizard_hyperliquid_live_lock",
        ),
        (
            "current_wizard_hyperliquid_storage_efficiency.csv",
            "current_wizard_hyperliquid_storage_efficiency",
        ),
        (
            "current_wizard_hyperliquid_storage_reclamation_plan.csv",
            "current_wizard_hyperliquid_storage_reclamation_plan",
        ),
        (
            "current_wizard_hyperliquid_storage_reclamation_validation.csv",
            "current_wizard_hyperliquid_storage_reclamation_validation",
        ),
        (
            "current_wizard_hyperliquid_archive_copy_receipt.csv",
            "current_wizard_hyperliquid_archive_copy_receipt",
        ),
        (
            "current_wizard_hyperliquid_archive_copy_validation.csv",
            "current_wizard_hyperliquid_archive_copy_validation",
        ),
        (
            "current_wizard_hyperliquid_archive_release_plan.csv",
            "current_wizard_hyperliquid_archive_release_plan",
        ),
        (
            "current_wizard_hyperliquid_archive_release_validation.csv",
            "current_wizard_hyperliquid_archive_release_validation",
        ),
        (
            "current_wizard_hyperliquid_testnet_protocol_scenarios.csv",
            "current_wizard_hyperliquid_testnet_protocol_scenarios",
        ),
        (
            "current_wizard_hyperliquid_testnet_protocol_transitions.csv",
            "current_wizard_hyperliquid_testnet_protocol_transitions",
        ),
        (
            "current_wizard_hyperliquid_testnet_protocol_candidate_coverage.csv",
            "current_wizard_hyperliquid_testnet_protocol_candidate_coverage",
        ),
        (
            "current_wizard_hyperliquid_testnet_protocol_validation.csv",
            "current_wizard_hyperliquid_testnet_protocol_validation",
        ),
        (
            "current_wizard_ou_optimal_overlay_ledger.csv",
            "current_wizard_ou_optimal_overlay_ledger",
        ),
        (
            "current_wizard_ou_optimal_overlay_coverage.csv",
            "current_wizard_ou_optimal_overlay_coverage",
        ),
        (
            "current_wizard_ou_optimal_overlay_validation.csv",
            "current_wizard_ou_optimal_overlay_validation",
        ),
    ):
        _write_csv(
            _read_csv(root / "reports" / "active" / source_name),
            paths[path_key],
        )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "exhaustive_wizard_hyperliquid_concentration_status.csv"
        ),
        paths["exhaustive_hyperliquid_concentration"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "exhaustive_wizard_hyperliquid_leverage_status.csv"
        ),
        paths["exhaustive_hyperliquid_leverage_status"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "exhaustive_wizard_hyperliquid_leverage_candidates.csv"
        ),
        paths["exhaustive_hyperliquid_leverage_candidates"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "exhaustive_wizard_hyperliquid_leverage_scenarios.csv"
        ),
        paths["exhaustive_hyperliquid_leverage_scenarios"],
    )
    _write_csv(
        _read_csv(
            root
            / "reports"
            / "active"
            / "exhaustive_wizard_hyperliquid_learning_ledger.csv"
        ),
        paths["exhaustive_hyperliquid_learning"],
    )
    _write_csv(_model_training_dashboard(model_acceptance), paths["model_training"])
    _write_csv(_read_csv(root / "reports" / "ml" / "model_gate_pair_support_report.csv"), paths["model_gate_pair_support"])
    _write_csv(_read_csv(root / "reports" / "agents" / "youtube_brain_status.csv"), paths["youtube_brain_status"])
    _write_csv(_read_csv(root / "reports" / "agents" / "youtube_brain_hypotheses.csv"), paths["youtube_brain_hypotheses"])
    _write_csv(_read_csv(root / "reports" / "agents" / "youtube_brain_replication_scorecard.csv"), paths["youtube_brain_replication"])
    _write_csv(_read_csv(root / "reports" / "active" / "youtube_hypothesis_validation_queue.csv"), paths["youtube_hypothesis_validation"])
    _write_csv(_read_csv(root / "reports" / "active" / "youtube_hypothesis_regime_matrix.csv"), paths["youtube_hypothesis_regime_matrix"])
    _write_text(paths["youtube_brain_summary"], _read_text(root / "reports" / "dashboard" / "youtube_brain.md"))
    _write_csv(_feature_readiness_dashboard(root), paths["feature_readiness"])
    _write_csv(_read_csv(root / "reports" / "active" / "orchestrator_run_status.csv"), paths["orchestrator_run_status"])
    _write_csv(_supreme_team_checkpoint_rows(root), paths["supreme_team_checkpoint"])
    _write_text(paths["project_spine_audit"], _read_text(root / "reports" / "active" / "project_spine_audit.md"))
    _write_csv(_read_csv(root / "reports" / "rl" / "rl_training_report.csv"), paths["rl_research_status"])
    _write_csv(_read_csv(root / "reports" / "rl" / "rl_execution_backtest.csv"), paths["rl_execution_backtest"])
    _write_csv(_read_csv(root / "reports" / "rl" / "rl_evaluation_report.csv"), paths["rl_position_sizing"])
    _write_csv(_read_csv(root / "reports" / "rl" / "rl_evaluation_report.csv"), paths["rl_strategy_selector"])
    _write_csv(_read_csv(root / "reports" / "rl" / "rl_blocked_actions.csv"), paths["rl_blocked_actions"])
    _write_csv(_read_csv(root / "reports" / "rl" / "rl_acceptance_report.csv"), paths["rl_acceptance"])
    _write_csv(_quantization_readiness_rows(root), paths["quantization_readiness"])
    _write_csv(live, paths["live_signals"])
    _write_csv(blocked, paths["blocked_trades"])
    _write_csv(data_health, paths["data_health"])
    _write_csv(refresh_status, paths["refresh_status"])
    _write_csv(_api_credit_usage_rows(), paths["api_credit_usage"])
    _write_csv(apify_cost_audit_rows(root), paths["apify_cost_audit"])
    _write_csv(paper_candidate_shortlist_rows(root), paths["paper_candidate_shortlist"])
    _write_csv(focused_paper_validation_rows(root), paths["focused_paper_validation"])
    _write_csv(live, paths["scoring_audit"])
    _write_csv(_read_csv(root / "reports" / "brain" / "brain_readiness_report.csv"), paths["brain_readiness_report"])
    _write_csv(_read_csv(root / "reports" / "brain" / "brain_candidate_rollup.csv"), paths["brain_candidate_rollup"])
    _write_csv(_read_csv(root / "reports" / "brain" / "paper_readiness_trend.csv"), paths["brain_readiness_trend"])
    _write_csv(_read_csv(root / "reports" / "brain" / "wizard_readiness.csv"), paths["wizard_readiness"])
    _write_csv(_read_csv(root / "reports" / "brain" / "native_readiness.csv"), paths["native_readiness"])
    _write_csv(_read_csv(root / "reports" / "brain" / "overall_readiness.csv"), paths["overall_readiness"])
    _write_csv(_read_csv(root / "reports" / "brain" / "paper_status.csv"), paths["paper_status"])
    _write_text(paths["three_brain_contract"], _read_text(root / "reports" / "brain" / "three_brain_contract.json"))
    _write_csv(_read_csv(root / "reports" / "brain" / "wizard_dashboard_investigation.csv"), paths["wizard_dashboard_investigation"])
    _write_csv(_read_csv(root / "reports" / "brain" / "forward_walk_strength_report.csv"), paths["forward_walk_strength_report"])
    _write_csv(_read_csv(root / "reports" / "brain" / "native_quality_report.csv"), paths["native_quality_report"])
    _write_csv(_read_csv(root / "reports" / "brain" / "native_candidate_diagnosis.csv"), paths["native_candidate_diagnosis"])
    _write_csv(_read_csv(root / "reports" / "brain" / "wizard_candidate_repair_report.csv"), paths["wizard_candidate_repair_report"])
    _write_csv(_read_csv(root / "reports" / "active" / "wizard_hourly_priority_capture_queue.csv"), paths["wizard_hourly_priority_capture_queue"])
    _write_csv(_read_csv(root / "reports" / "brain" / "native_focus_repair_report.csv"), paths["native_focus_repair_report"])
    _write_csv(_read_csv(root / "reports" / "brain" / "wizard_hourly_repair_targets.csv"), paths["wizard_hourly_repair_targets"])
    _write_csv(_read_csv(root / "reports" / "brain" / "candidate_repair_loop.csv"), paths["candidate_repair_loop"])
    _write_csv(_read_csv(root / "reports" / "brain" / "shadow_rl_usefulness_report.csv"), paths["shadow_rl_usefulness_report"])
    _write_csv(_read_csv(root / "reports" / "brain" / "shadow_rl_candidate_set.csv"), paths["shadow_rl_candidate_set"])
    _write_csv(_read_csv(root / "reports" / "brain" / "shadow_rl_intake_report.csv"), paths["shadow_rl_intake_report"])
    _write_csv(_read_csv(root / "reports" / "brain" / "overall_arbitration_scorecard.csv"), paths["overall_arbitration_scorecard"])
    _write_csv(_read_csv(root / "reports" / "brain" / "blocker_persistence_report.csv"), paths["blocker_persistence_report"])
    _write_csv(_read_csv(root / "reports" / "brain" / "verified_paper_outcome_report.csv"), paths["verified_paper_outcome_report"])
    _write_csv(_read_csv(root / "reports" / "brain" / "overall_brain_summary.csv"), paths["overall_brain_summary"])
    _write_csv(_read_csv(root / "reports" / "brain" / "promotion_ladder.csv"), paths["promotion_ladder"])
    _write_csv(_read_csv(root / "reports" / "brain" / "orchestrator_status.csv"), paths["orchestrator_status"])
    _write_csv(_read_csv(root / "reports" / "rl" / "base_rl_training_report.csv"), paths["base_rl_training_report"])
    _write_csv(_read_csv(root / "reports" / "rl" / "base_rl_evaluation_report.csv"), paths["base_rl_evaluation_report"])
    _write_csv(_read_csv(root / "reports" / "rl" / "base_rl_pair_coverage.csv"), paths["base_rl_pair_coverage"])
    _write_csv(_read_csv(root / "reports" / "rl" / "base_rl_blocked_actions.csv"), paths["base_rl_blocked_actions"])
    _write_csv(_read_csv(root / "reports" / "rl" / "base_rl_paper_handoff_status.csv"), paths["base_rl_paper_handoff_status"])
    _write_csv(_read_csv(root / "reports" / "rl" / "base_vs_augmented_pair_comparison.csv"), paths["base_rl_comparison_report"])
    _write_csv(_read_csv(root / "reports" / "rl" / "base_rl_promotion_readiness.csv"), paths["base_rl_promotion_readiness"])
    _write_csv(_read_csv(root / "reports" / "rl" / "base_rl_promotion_decision.csv"), paths["base_rl_promotion_decision"])
    _write_csv(_read_csv(root / "reports" / "rl" / "base_rl_blocker_delta_report.csv"), paths["base_rl_blocker_delta"])
    _write_csv(_read_csv(root / "reports" / "rl" / "base_rl_pair_blocker_report.csv"), paths["base_rl_pair_blocker"])
    _write_text(
        paths["command_center"],
        _command_center_markdown(
            dashboard_universe,
            current,
            model_acceptance,
            data_health,
            route_recommendations=route_recommendations,
            refresh_status=refresh_status,
            root=root,
        ),
    )
    return CommandResult(
        paths=paths,
        summary={
            "dashboard_files": len(paths),
            "blocked_rows": len(blocked),
            "refresh_profile": requested_refresh_profile,
            "effective_refresh_profile": effective_refresh_profile,
            "deep_refresh_blocked_by_storage": requested_refresh_profile == "deep" and not bool(storage["ready"]),
            "wizard_ranking_ready": wizard_ranking_ready,
            "wizard_ranking_blocker": str(wizard_control.summary.get("blocker", "")),
        },
    )


def archive_from_index(dry_run: bool = True, root: Path = ROOT) -> CommandResult:
    index = _read_csv(root / "reports" / "active" / "artifact_index.csv")
    if index.empty:
        raise SystemExit("artifact index missing; run build-artifact-index first")
    candidates = index[index["safe_to_archive_later"].astype(bool)].copy()
    candidates["planned_action"] = "would_archive" if dry_run else "blocked_by_policy"
    candidates["archive_path"] = candidates["path"].map(lambda p: f"archive/pending/{p}")
    out = root / "archive" / "archive_manifest.csv"
    _write_csv(candidates, out)
    if not dry_run:
        raise SystemExit("archive apply is intentionally blocked until active lineage is reviewed")
    return CommandResult(paths={"archive_manifest": out}, summary={"dry_run": dry_run, "candidates": len(candidates)})


def _iter_repo_files(root: Path) -> Iterable[Path]:
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not _artifact_dir_is_excluded(name)]
        base = Path(directory)
        for filename in filenames:
            if filename in ARTIFACT_EXCLUDED_FILES:
                continue
            yield base / filename


def _artifact_dir_is_excluded(name: str) -> bool:
    return name in ARTIFACT_EXCLUDED_DIRS or name.startswith(ARTIFACT_EXCLUDED_DIR_PREFIXES)


def _artifact_row(path: Path, root: Path) -> dict[str, object]:
    rel = path.relative_to(root).as_posix()
    status = _artifact_status(rel)
    stat = path.stat()
    return {
        "path": rel,
        "artifact_type": _artifact_type(rel),
        "status": status,
        "source_system": _source_system(rel),
        "created_or_modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "used_by_active_pipeline": status in {"active", "do_not_move"} or rel.startswith("data/raw") or rel.startswith("data/processed"),
        "evidence_value": _evidence_value(rel, status),
        "safe_to_archive_later": status in {"scratch", "superseded"},
        "reason": _artifact_reason(rel, status),
        "notes": "non_destructive_index_only",
    }


def _artifact_status(rel: str) -> str:
    if rel in {
        ".env.local",
        ".env.example",
        ".gitignore",
        "mcp.example.json",
        "pyproject.toml",
        "README.md",
        "project_objective.md",
        "memory.md",
        "uv.lock",
    }:
        return "do_not_move"
    if rel.startswith(
        ("src/", "tests/", "config/", "scripts/", "docs/", "apps/", ".github/", ".vscode/")
    ):
        return "active"
    if rel.startswith(
        (
            "reports/active/",
            "reports/dashboard/",
            "reports/ml/",
            "data/agent_memory/",
            "data/fixtures/",
            "data/ml/",
            "models/trade_gate/",
        )
    ):
        return "active"
    if rel.startswith(("data/", "models/", "reports/", "runs/", "archive/")):
        return "historical_evidence"
    if rel.startswith(("work/", "outputs/", "tmp_")) or rel.endswith((".log", ".tmp")):
        return "scratch"
    if ".pytest_cache/" in rel or rel.endswith(".pyc"):
        return "superseded"
    return "unknown"


def _artifact_type(rel: str) -> str:
    suffix = Path(rel).suffix.lower().lstrip(".")
    if rel.startswith("src/"):
        return "source_code"
    if rel.startswith("tests/"):
        return "test"
    if rel.startswith("reports/"):
        return "report"
    if rel.startswith("data/"):
        return "data"
    if rel.startswith("docs/"):
        return "documentation"
    if rel.startswith("scripts/"):
        return "script"
    return suffix or "file"


def _source_system(rel: str) -> str:
    text = rel.lower()
    if "crypto_wizards" in text or "wizards" in text:
        return "crypto_wizards"
    if "dydx" in text:
        return "dydx"
    if "apify" in text:
        return "apify"
    if "ml" in text or "model" in text:
        return "modeling"
    return "local"


def _evidence_value(rel: str, status: str) -> str:
    if status == "historical_evidence":
        return "preserve_for_lineage"
    if status == "active":
        return "current_pipeline_or_contract"
    if status == "do_not_move":
        return "configuration_or_repo_contract"
    return "low_until_reviewed"


def _artifact_reason(rel: str, status: str) -> str:
    if status == "active":
        return "part_of_active_code_docs_or_outputs"
    if status == "historical_evidence":
        return "existing research evidence; keep until active lineage supersedes it"
    if status == "scratch":
        return "working artifact; candidate for later archive only"
    if status == "superseded":
        return "cache or generated byproduct"
    if status == "do_not_move":
        return "repo contract or local configuration"
    return "needs manual classification"


def _candidate_pairs(root: Path) -> list[tuple[str, tuple[str, str], set[str]]]:
    found: dict[str, tuple[tuple[str, str], set[str]]] = {}
    for path in (root / "data" / "processed" / "evidence_pipeline").glob("*_pair_history.csv"):
        stem = path.name.replace("_pair_history.csv", "")
        parts = stem.split("_")
        if len(parts) >= 3:
            asset_x, asset_y = _asset(parts[0]), _asset(parts[1])
            pair = f"{asset_x}-{asset_y}"
            found.setdefault(pair, ((asset_x, asset_y), set()))[1].add(path.relative_to(root).as_posix())
    for report in [
        root / "reports" / "dydx_pair_research_priority_ranked.csv",
        root / "reports" / "strategy_fit_tests" / "ranked_report.csv",
        root / "reports" / "experiment_results.csv",
    ]:
        frame = _read_csv(report)
        if frame.empty or "pair" not in frame:
            continue
        for pair_text in frame["pair"].dropna().astype(str).head(300):
            assets = _assets_from_pair(pair_text)
            if assets:
                pair = f"{assets[0]}-{assets[1]}"
                found.setdefault(pair, (assets, set()))[1].add(report.relative_to(root).as_posix())
    for pair, assets, evidence_path in _apify_market_candidate_pairs(root):
        found.setdefault(pair, (assets, set()))[1].add(evidence_path)
    wizard = _read_csv(root / "data" / "processed" / "wizard_evidence.csv")
    if not wizard.empty and "pair" in wizard:
        for _, row in wizard.head(500).iterrows():
            assets = _assets_from_pair(str(row.get("pair", "")))
            if not assets:
                left = str(row.get("asset_x", "") or "")
                right = str(row.get("asset_y", "") or "")
                assets = (left, right) if _valid_asset_text(left) and _valid_asset_text(right) else None
            if not assets:
                continue
            pair = f"{assets[0]}-{assets[1]}"
            path = str(row.get("evidence_path", "data/processed/wizard_evidence.csv") or "data/processed/wizard_evidence.csv")
            found.setdefault(pair, (assets, set()))[1].add(path)
    return [(pair, assets, paths) for pair, (assets, paths) in found.items()]


def _venue_context_by_pair(context: pd.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    if context.empty:
        return {}
    table = context.copy()
    if {"asset", "venue"}.issubset(table.columns):
        venue_map: dict[tuple[str, str], dict[str, object]] = {}
        for asset, group in table.groupby("asset"):
            normalized_asset = str(asset or "").upper().strip()
            if not normalized_asset:
                continue
            for _, row in group.iterrows():
                venue = str(row.get("venue", "")).lower().strip()
                if not venue:
                    continue
                key = (normalized_asset, venue)
                venue_map[key] = {
                    "asset": normalized_asset,
                    "venue": venue,
                    "tradable": bool(row.get("tradable")),
                    "execution_authority": bool(row.get("execution_authority")),
                    "funding_pulse_status": str(row.get("funding_pulse_status", "")),
                    "blocker": str(row.get("blocker", "") or "").strip(),
                    "liquidity_24h": _clean_numeric(row.get("volume_24h")),
                    "open_interest_usd": _clean_numeric(row.get("open_interest_usd")),
                    "evidence_path": str(row.get("evidence_path", "")),
                }
        return venue_map
    return {}


def _venue_profile_from_dydx(candle_index: dict[str, set[str]], timeframes: list[str], market_snapshot: dict[str, dict[str, object]], asset_x: str, asset_y: str) -> dict[str, object]:
    dydx_ready = bool(candle_index.get(asset_x) and candle_index.get(asset_y)) or (
        asset_x in market_snapshot and asset_y in market_snapshot
    )
    return {
        "best_venue": "hyperliquid",
        "execution_ready": dydx_ready,
        "available_venues": "hyperliquid",
        "reason": "hyperliquid_fallback" if dydx_ready else "no_multi_venue_context_available",
    }


def _venue_profile_for_pair(context: dict[tuple[str, str], dict[str, object]], asset_x: str, asset_y: str) -> dict[str, object]:
    def _venue_asset_key(value: str) -> str:
        candidate = str(value or "").upper().replace("_", "-")
        return candidate[:-4] if candidate.endswith("-USD") else candidate

    normalized_x = _venue_asset_key(asset_x)
    normalized_y = _venue_asset_key(asset_y)
    candidates: list[tuple[str, float, dict[str, str]]] = []
    venues = set(
        row.get("venue")
        for key, row in context.items()
        if _venue_asset_key(key[0]) in {normalized_x, normalized_y}
    )
    venues = {str(venue) for venue in venues if venue}
    for venue in sorted(venues):
        left = context.get((normalized_x, venue), {})
        right = context.get((normalized_y, venue), {})
        if not left:
            left = context.get((f"{normalized_x}-USD", venue), {})
        if not right:
            right = context.get((f"{normalized_y}-USD", venue), {})
        if not left or not right:
            continue
        left_blockers = set(str(left.get("blocker", "")).split(";")) if left.get("blocker") else set()
        right_blockers = set(str(right.get("blocker", "")).split(";")) if right.get("blocker") else set()
        combined_blockers = left_blockers | right_blockers
        execution_ready = bool(left.get("execution_authority")) and bool(right.get("execution_authority"))
        tradable = bool(left.get("tradable")) and bool(right.get("tradable"))
        liquidity = 0.0
        for row in (left, right):
            try:
                liquidity += float(row.get("liquidity_24h") or 0.0)
            except (TypeError, ValueError):
                pass
        try:
            oi = float(left.get("open_interest_usd") or 0.0) + float(right.get("open_interest_usd") or 0.0)
        except (TypeError, ValueError):
            oi = 0.0
        blocker_penalty = 0 if not combined_blockers else -10.0
        score = 120.0 if execution_ready else (30.0 if tradable else 0.0)
        score += min(liquidity / 1_000_000.0, 20.0)
        score += min(oi / 5_000_000.0, 10.0)
        score += blocker_penalty
        candidates.append((venue, score, {"reason": ";".join(sorted(combined_blockers)) if combined_blockers else "tradeable", "ready": execution_ready}))
    if not candidates:
        return {}
    best_venue, _, info = max(candidates, key=lambda row: row[1])
    return {
        "best_venue": best_venue,
        "execution_ready": bool(info.get("ready", False)),
        "available_venues": ";".join(sorted(venue for venue, _, _ in candidates)),
        "reason": str(info.get("reason", "")),
    }


def _apify_market_candidate_pairs(root: Path, max_markets: int = 10) -> list[tuple[str, tuple[str, str], str]]:
    markets = _latest_apify_markets(root)
    ranked = sorted(
        markets.values(),
        key=lambda row: (float(row.get("volume24H", 0.0) or 0.0), float(row.get("trades24H", 0.0) or 0.0)),
        reverse=True,
    )[:max_markets]
    rows: list[tuple[str, tuple[str, str], str]] = []
    for idx, left in enumerate(ranked):
        for right in ranked[idx + 1 :]:
            asset_x = str(left.get("ticker", "")).upper()
            asset_y = str(right.get("ticker", "")).upper()
            if not asset_x or not asset_y:
                continue
            pair = f"{asset_x}-{asset_y}"
            rows.append((pair, (asset_x, asset_y), str(left.get("_evidence_path", "data/raw/dydx_inbox/apify_dydx_markets_snapshot_latest.json"))))
    return rows


def _asset(text: str) -> str:
    if not text or str(text).strip().lower() in {"nan", "none", "null"}:
        return ""
    upper = text.upper().replace("-", "_")
    if upper.endswith("_USD"):
        return upper.replace("_", "-")
    return f"{upper}-USD"


def _assets_from_pair(pair_text: str) -> tuple[str, str] | None:
    if not pair_text or str(pair_text).strip().lower() in {"nan", "none", "null"}:
        return None
    pieces = [piece for piece in pair_text.replace("/", "-").split("-") if piece]
    if any(piece.strip().lower() in {"nan", "none", "null"} for piece in pieces):
        return None
    usd_positions = [i for i, piece in enumerate(pieces) if piece.upper() == "USD"]
    if len(usd_positions) >= 2:
        return (f"{pieces[0].upper()}-USD", f"{pieces[usd_positions[0] + 1].upper()}-USD")
    if "_" in pair_text:
        parts = pair_text.split("_")
        if len(parts) >= 2:
            assets = (_asset(parts[0]), _asset(parts[1]))
            return assets if all(assets) else None
    return None


def _valid_asset_text(value: object) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and text not in {"nan", "none", "null", "unknown"}


def _dydx_candle_index(root: Path) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for path in (root / "data" / "raw" / "dydx_candles").glob("*_candles.json"):
        name = path.name.replace("_candles.json", "")
        pieces = name.split("_")
        if len(pieces) >= 2:
            market = pieces[0]
            timeframe = pieces[1]
            index.setdefault(market, set()).add(timeframe)
    return index


def _local_pair_metrics(pair: str, experiment: pd.DataFrame, acceptance: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {"local_backtest_score": 0.0}
    subset = pd.DataFrame()
    if not experiment.empty and "pair" in experiment:
        subset = experiment[experiment["pair"].astype(str).str.upper() == pair.upper()]
    if not subset.empty:
        pf = _numeric(subset.get("profit_factor", pd.Series(dtype=float))).median()
        sharpe = _numeric(subset.get("sharpe", pd.Series(dtype=float))).median()
        dd = _numeric(subset.get("max_drawdown", pd.Series(dtype=float))).median()
        trades = _numeric(subset.get("trades", subset.get("trade_count", pd.Series(dtype=float)))).median()
        metrics["local_backtest_score"] = float(np.clip((pf - 1.0) * 15 + max(sharpe, 0) * 10 + min(trades, 100) / 5 - dd * 30, 0, 60))
        metrics["zscore_score"] = float(np.clip(max(sharpe, 0) * 10, 0, 15))
    if not acceptance.empty and "production_eligible" in acceptance:
        eligible = acceptance["production_eligible"].astype(str).str.lower().isin({"true", "1", "yes"}).any()
        if eligible:
            metrics["local_backtest_score"] = max(metrics["local_backtest_score"], 40.0)
    return metrics


def _wizard_research_tables(root: Path) -> dict[str, pd.DataFrame]:
    evidence_path = root / "data" / "processed" / "wizard_evidence.csv"
    if not evidence_path.exists():
        try:
            from quant_platform.wizard_evidence import build_wizard_research_pack

            build_wizard_research_pack(root)
        except Exception:
            pass
    return {
        "evidence": _read_csv(evidence_path),
        "hypotheses": _read_csv(root / "reports" / "active" / "wizard_hypotheses.csv"),
        "diagnostics": _read_csv(root / "reports" / "active" / "wizard_diagnostic_confirmation.csv"),
        "parity": _read_csv(root / "reports" / "active" / "wizard_vs_local_parity_report.csv"),
    }


def _wizard_pair_metrics(pair: str, tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    metrics: dict[str, object] = {}
    evidence = _wizard_pair_subset(tables.get("evidence", pd.DataFrame()), pair)
    if evidence.empty:
        return metrics
    ranked = evidence.copy()
    ranked["_rank_source_fresh"] = ranked.get("source_fresh", pd.Series(False, index=ranked.index)).map(_boolish).astype(int)
    ranked["_rank_source_healthy"] = ranked.get("source_health", pd.Series("", index=ranked.index)).astype(str).str.lower().eq("healthy").astype(int)
    ranked["_rank_mode_valid"] = ranked.get("mode_valid", pd.Series(False, index=ranked.index)).map(_boolish).astype(int)
    ranked["_rank_sharpe"] = _numeric(ranked.get("sharpe", pd.Series(dtype=float))).fillna(-999)
    ranked["_rank_return"] = _numeric(ranked.get("returns_total", pd.Series(dtype=float))).fillna(-999)
    if "passes_sharpe_gate" in ranked:
        ranked["_rank_sharpe_gate"] = ranked["passes_sharpe_gate"].map(_boolish).astype(int)
    else:
        ranked["_rank_sharpe_gate"] = ranked["_rank_sharpe"].ge(1.75).astype(int)
    if "passes_returns_total_gt_20pct" in ranked:
        ranked["_rank_return_gate"] = ranked["passes_returns_total_gt_20pct"].map(_boolish).astype(int)
    else:
        ranked["_rank_return_gate"] = ranked["_rank_return"].gt(0.10).astype(int)
    ranked = ranked.sort_values(
        [
            "_rank_source_fresh",
            "_rank_source_healthy",
            "_rank_mode_valid",
            "_rank_sharpe_gate",
            "_rank_return_gate",
            "_rank_sharpe",
            "_rank_return",
        ],
        ascending=[False, False, False, False, False, False, False],
    )
    best = ranked.iloc[0]
    eligible = ranked[
        ranked["_rank_source_fresh"].eq(1)
        & ranked["_rank_source_healthy"].eq(1)
        & ranked["_rank_mode_valid"].eq(1)
    ].copy()
    if eligible.empty:
        if not ranked["_rank_source_fresh"].eq(1).any():
            evidence_state = "STALE_EVIDENCE"
        elif not ranked.loc[ranked["_rank_source_fresh"].eq(1), "_rank_source_healthy"].eq(1).any():
            evidence_state = "CAPTURE_UNHEALTHY"
        else:
            evidence_state = "INVALID_EXACT_MODE"
    else:
        best = eligible.iloc[0]
        authority = str(best.get("source_authority", "") or "")
        evidence_state = "DISCOVERY_ONLY" if authority == "discovery_only" else "CURRENT_RESEARCH_EVIDENCE"
    metrics.update(
        {
            "best_wizard_exact_mode": best.get("exact_mode", ""),
            "best_wizard_spread_id": best.get("spread_id", ""),
            "best_wizard_strategy_id": best.get("strategy_id", ""),
            "best_wizard_local_strategy_id": best.get("local_strategy_id", ""),
            "best_wizard_local_strategy_name": best.get("local_strategy_name", ""),
            "best_wizard_local_strategy_family": best.get("local_strategy_family", ""),
            "best_wizard_strategy_mapping_status": best.get("strategy_mapping_status", ""),
            "best_wizard_sharpe": best.get("sharpe", ""),
            "best_wizard_returns_total": best.get("returns_total", ""),
            "best_wizard_exchange": best.get("exchange", ""),
            "best_wizard_source_authority": best.get("source_authority", ""),
            "best_wizard_source_timestamp": best.get("source_timestamp", ""),
            "best_wizard_source_fresh": _boolish(best.get("source_fresh", False)),
            "best_wizard_source_health": best.get("source_health", ""),
            "wizard_evidence_state": evidence_state,
            "wizard_stationarity_status": best.get("stationarity_status", ""),
            "wizard_engle_granger_cointegrated": best.get("engle_granger_cointegrated", ""),
            "wizard_engle_granger_trend": best.get("engle_granger_trend", ""),
            "wizard_johansen_cointegrated": best.get("johansen_cointegrated", ""),
            "wizard_zscore_last": best.get("zscore_last", ""),
            "wizard_zscore_roll_last": best.get("zscore_roll_last", ""),
            "wizard_leg_volume_min": best.get("volume_min", ""),
            "zscore_score": (
                round(float(np.clip(float(_numeric(pd.Series([best.get("sharpe", 0)])).fillna(0).iloc[0]) * 3, 0.0, 15.0)), 3)
                if not eligible.empty
                else 0.0
            ),
        }
    )
    diagnostics = _wizard_pair_subset(tables.get("diagnostics", pd.DataFrame()), pair)
    if not diagnostics.empty and not eligible.empty and "wizard_diagnostic_score" in diagnostics:
        if "source_fresh" in diagnostics:
            diagnostics = diagnostics[diagnostics["source_fresh"].map(_boolish)].copy()
        if "source_health" in diagnostics:
            diagnostics = diagnostics[diagnostics["source_health"].astype(str).str.lower().eq("healthy")].copy()
        exact_mode = str(best.get("exact_mode", "") or "")
        if exact_mode and "exact_mode" in diagnostics:
            diagnostics = diagnostics[diagnostics["exact_mode"].astype(str).eq(exact_mode)].copy()
        if not diagnostics.empty:
            metrics["wizard_diagnostic_score"] = float(_numeric(diagnostics["wizard_diagnostic_score"]).max())
            metrics["cointegration_score"] = min(float(metrics["wizard_diagnostic_score"]) / 3, 20.0)
    hypotheses = _wizard_pair_subset(tables.get("hypotheses", pd.DataFrame()), pair)
    if not hypotheses.empty:
        ranked_hypotheses = hypotheses.copy()
        if "source_fresh" in ranked_hypotheses:
            ranked_hypotheses = ranked_hypotheses[ranked_hypotheses["source_fresh"].map(_boolish)].copy()
        if "source_health" in ranked_hypotheses:
            ranked_hypotheses = ranked_hypotheses[ranked_hypotheses["source_health"].astype(str).str.lower().eq("healthy")].copy()
        exact_mode = str(best.get("exact_mode", "") or "")
        if exact_mode and "exact_mode" in ranked_hypotheses:
            ranked_hypotheses = ranked_hypotheses[ranked_hypotheses["exact_mode"].astype(str).eq(exact_mode)].copy()
        if ranked_hypotheses.empty:
            ranked_hypotheses = pd.DataFrame()
    if not hypotheses.empty and not ranked_hypotheses.empty:
        ranked_hypotheses["_rank_exact_mode"] = ranked_hypotheses.get("exact_mode", pd.Series("", index=ranked_hypotheses.index)).astype(str).str.strip().ne("").astype(int)
        ranked_hypotheses["_rank_ready"] = ranked_hypotheses.get("hypothesis_status", pd.Series("", index=ranked_hypotheses.index)).astype(str).eq("HYPOTHESIS_READY").astype(int)
        ranked_hypotheses = ranked_hypotheses.sort_values(["_rank_ready", "_rank_exact_mode"], ascending=[False, False])
        metrics["wizard_hypothesis_status"] = str(ranked_hypotheses.iloc[0].get("hypothesis_status", ""))
    else:
        metrics["wizard_hypothesis_status"] = evidence_state
    parity = _wizard_pair_subset(tables.get("parity", pd.DataFrame()), pair)
    if not parity.empty and not eligible.empty:
        exact_mode = str(best.get("exact_mode", "") or "")
        if exact_mode and "exact_mode" in parity:
            parity = parity[parity["exact_mode"].astype(str).eq(exact_mode)].copy()
    if not parity.empty and not eligible.empty:
        status = str(parity.iloc[0].get("parity_status", ""))
        metrics["wizard_local_parity_status"] = status
        metrics["local_mode_confirmation_status"] = "confirmed" if status in {"MATCH", "CLOSE"} else "unconfirmed"
    return metrics


def _wizard_pair_subset(frame: pd.DataFrame, pair: str) -> pd.DataFrame:
    if frame.empty or "pair" not in frame:
        return pd.DataFrame()
    target_compact = _compact_pair_identity(pair)
    if target_compact:
        source_compact = frame["pair"].map(_compact_pair_identity)
        if {"asset_x", "asset_y"}.issubset(frame.columns):
            leg_compact = frame.apply(
                lambda row: _compact_pair_identity(f"{row.get('asset_x', '')}/{row.get('asset_y', '')}"),
                axis=1,
            )
        else:
            leg_compact = pd.Series("", index=frame.index)
        direct = frame[(source_compact == target_compact) | (leg_compact == target_compact)]
        if not direct.empty:
            return direct
    target_assets = _assets_from_pair(pair)
    if not target_assets:
        return frame[frame["pair"].astype(str).str.upper() == pair.upper()]
    target_legs = _wizard_canonical_legs(target_assets[0], target_assets[1])
    if not target_legs:
        return pd.DataFrame()
    matches = frame.apply(
        lambda row: _wizard_canonical_legs(row.get("asset_x", ""), row.get("asset_y", "")) == target_legs,
        axis=1,
    )
    return frame[matches]


def _compact_pair_identity(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _wizard_canonical_legs(asset_x: object, asset_y: object) -> tuple[str, str] | None:
    left = normalize_wizard_symbol(asset_x)
    right = normalize_wizard_symbol(asset_y)
    left_key = left.canonical_usd_symbol or left.normalized_symbol
    right_key = right.canonical_usd_symbol or right.normalized_symbol
    if not left_key or not right_key:
        return None
    return left_key, right_key


def _promotion_blocker(bucket: str, metrics: dict[str, object]) -> str:
    if bucket == "PROMOTE":
        return ""
    blockers = []
    if metrics.get("best_wizard_sharpe") not in {"", None} and metrics.get("local_backtest_score", 0.0) <= 0:
        blockers.append("wizard_discovery_only_needs_local_acceptance")
    parity = str(metrics.get("wizard_local_parity_status", ""))
    if parity and parity not in {"MATCH", "CLOSE"}:
        blockers.append(f"wizard_local_parity_{parity.lower()}")
    status = str(metrics.get("wizard_hypothesis_status", ""))
    if status in {
        "RESEARCH_BLOCKED",
        "NEEDS_LOCAL_DATA",
        "STALE_EVIDENCE",
        "CAPTURE_UNHEALTHY",
        "INVALID_EXACT_MODE",
        "DISCOVERY_ONLY",
    }:
        blockers.append(status.lower())
    return ";".join(blockers)


def _latest_apify_markets(root: Path) -> dict[str, dict[str, object]]:
    candidates = sorted((root / "data" / "raw" / "dydx_inbox").glob("apify_dydx_markets_snapshot*.json"))
    if not candidates:
        return {}
    latest = next((path for path in candidates if path.name.endswith("_latest.json")), candidates[-1])
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, dict):
        items = payload.get("items", [])
    elif isinstance(payload, list):
        items = payload
    else:
        return {}
    markets: dict[str, dict[str, object]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).upper()
        if not ticker:
            continue
        enriched = dict(item)
        enriched["_evidence_path"] = latest.relative_to(root).as_posix()
        markets[ticker] = enriched
    return markets


def _apply_market_snapshot_metrics(metrics: dict[str, float], asset_x: str, asset_y: str, markets: dict[str, dict[str, object]]) -> None:
    left = markets.get(asset_x, {})
    right = markets.get(asset_y, {})
    if not left and not right:
        return
    volume = float(left.get("volume24H", 0.0) or 0.0) + float(right.get("volume24H", 0.0) or 0.0)
    open_interest = float(left.get("openInterest", 0.0) or 0.0) + float(right.get("openInterest", 0.0) or 0.0)
    trades = float(left.get("trades24H", 0.0) or 0.0) + float(right.get("trades24H", 0.0) or 0.0)
    metrics["volume_usd"] = volume
    metrics["open_interest_usd"] = open_interest
    metrics["liquidity_hint"] = min(volume / 1_000_000.0, 15.0) + min(trades / 2_000.0, 5.0)


def _market_evidence_paths(asset_x: str, asset_y: str, markets: dict[str, dict[str, object]]) -> set[str]:
    paths = set()
    for asset in [asset_x, asset_y]:
        path = markets.get(asset, {}).get("_evidence_path")
        if path:
            paths.add(str(path))
    return paths


def _funding_drag(asset_x: str, asset_y: str, funding: pd.DataFrame, markets: dict[str, dict[str, object]] | None = None) -> float:
    markets = markets or {}
    market_values = []
    for asset in [asset_x, asset_y]:
        if asset in markets:
            rate = float(markets[asset].get("nextFundingRate", 0.0) or 0.0)
            market_values.append(abs(rate * 10_000.0))
    if market_values:
        return round(sum(market_values), 6)
    if funding.empty:
        return 0.0
    market_col = next((c for c in ["market", "symbol", "ticker"] if c in funding), None)
    value_col = next((c for c in ["funding_bps", "funding_rate_bps", "rate_bps"] if c in funding), None)
    if not market_col or not value_col:
        return 0.0
    values = []
    for asset in [asset_x, asset_y]:
        rows = funding[funding[market_col].astype(str).str.upper() == asset.upper()]
        if not rows.empty:
            values.append(abs(float(_numeric(rows[value_col]).dropna().tail(1).iloc[0])))
    return round(sum(values), 6)


def _discovery_components(
    execution_ready: bool,
    timeframes: list[str],
    metrics: dict[str, float],
    funding_drag: float,
    *,
    available_venues: str | None = None,
) -> dict[str, float]:
    return {
        "mean_reversion_hint": 8.0 if timeframes else 0.0,
        "cointegration_hint": metrics.get("cointegration_score", 0.0),
        "copula_hint": metrics.get("copula_score", 0.0),
        "liquidity_hint": max(metrics.get("liquidity_hint", 0.0), 8.0 if execution_ready else 0.0),
        "dashboard_confirmation_hint": min(metrics.get("local_backtest_score", 0.0) / 5, 10.0),
        "obvious_instability_penalty": -min(funding_drag / 5, 5.0),
        "venue_diversity_hint": min(len([value for value in str(available_venues or "").split(";") if value]), 2.0),
    }


def _acceptance_components(
    execution_ready: bool,
    metrics: dict[str, float],
    funding_drag: float,
    *,
    best_venue: str | None = None,
    venue_ready: bool | None = None,
) -> dict[str, float]:
    return {
        "local_backtest_score": metrics.get("local_backtest_score", 0.0),
        "walk_forward_score": 0.0,
        "regime_stability_score": 0.0,
        "trade_count_score": 0.0,
        "tradeability_score": 15.0 if (venue_ready if venue_ready is not None else execution_ready) else 0.0,
        "venue_fallback_penalty": -5.0 if (best_venue or "hyperliquid") != "hyperliquid" and not (venue_ready or False) else 0.0,
        "funding_drag_penalty": -min(funding_drag / 2, 10.0),
        "drawdown_penalty": 0.0,
        "cost_failure_penalty": 0.0,
        "stale_data_penalty": 0.0,
    }


def _decision_bucket(
    discovery: float,
    acceptance: float,
    metrics: dict[str, float],
    execution_ready: bool,
    timeframes: list[str],
    *,
    best_venue: str | None = None,
    available_venues: str | None = None,
) -> tuple[str, str]:
    venue = (best_venue or "").lower() or "hyperliquid"
    if acceptance >= 70 and execution_ready and (not available_venues or venue in available_venues):
        return "PROMOTE", f"local_acceptance_score_passed_with_tradeability_venue_{venue}"
    if not execution_ready or not timeframes:
        return "FETCH_MORE_DATA", f"missing_{venue}_tradeability_or_timeframe_history"
    if acceptance >= 25 or discovery >= 20:
        return "WATCH", f"promising_discovery_or_partial_local_evidence_but_not_promoted_for_{venue}"
    return "REJECT", "insufficient_local_acceptance_evidence"


def _missing_pair_data(
    execution_ready: bool,
    timeframes: list[str],
    metrics: dict[str, float],
    funding_drag: float,
    *,
    best_venue: str | None = None,
) -> str:
    missing = []
    venue = (best_venue or "dydx").lower()
    if not execution_ready:
        missing.append(f"{venue}_leg_readiness")
    if not timeframes:
        missing.append("common_timeframes")
    if metrics.get("local_backtest_score", 0.0) <= 0:
        missing.append("local_backtest_metrics")
    if funding_drag == 0:
        missing.append("funding_drag_observation")
    wizard_state = str(metrics.get("wizard_evidence_state", "") or "")
    if wizard_state == "DISCOVERY_ONLY":
        missing.append("wizard_discovery_only_needs_pair_detail_and_local_replay")
    elif wizard_state == "STALE_EVIDENCE":
        missing.append("wizard_evidence_stale")
    elif wizard_state == "CAPTURE_UNHEALTHY":
        missing.append("wizard_capture_unhealthy")
    elif wizard_state == "INVALID_EXACT_MODE":
        missing.append("wizard_exact_mode_invalid")
    return ";".join(missing)


def _harden_trade_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    hardened = frame.copy()
    hardened["good_trade"] = hardened.get(TARGET_COLUMN, 0)
    hardened["profit_after_cost"] = hardened.get(RETURN_COLUMN, 0.0)
    fallback_adverse = pd.Series(np.minimum(hardened["profit_after_cost"].astype(float), 0.0), index=hardened.index)
    fallback_favorable = pd.Series(np.maximum(hardened["profit_after_cost"].astype(float), 0.0), index=hardened.index)
    hardened["max_adverse_excursion"] = pd.to_numeric(
        hardened.get("max_adverse_excursion", fallback_adverse), errors="coerce"
    ).fillna(fallback_adverse)
    hardened["max_favorable_excursion"] = pd.to_numeric(
        hardened.get("max_favorable_excursion", fallback_favorable), errors="coerce"
    ).fillna(fallback_favorable)
    hardened["return_aggregation"] = hardened.get("return_aggregation", "legacy_terminal_return")
    hardened["return_unit"] = hardened.get("return_unit", "fraction_of_equity")
    hardened["hold_bars"] = hardened.get("trade_bars", 0)
    hardened["exit_reason"] = "strategy_exit"
    hardened["feature_timestamp"] = hardened.get("entry_timestamp", "")
    hardened["label_timestamp"] = hardened.get("exit_timestamp", "")
    hardened["uses_dashboard_hindsight"] = False
    hardened["feature_completeness_score"] = hardened.notna().mean(axis=1).round(4)
    hardened["evidence_path"] = hardened.get(
        "source_path", "data/raw/pair_details"
    )
    hardened["backtest_label"] = hardened["good_trade"]
    hardened["paper_label"] = ""
    hardened["live_label"] = ""
    return hardened


def _enrich_trade_dataset_with_outcome_memory(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    enriched = frame.copy()
    native_features = _read_csv(root / "reports" / "brain" / "native_outcome_feature_memory.csv")
    shared = _read_csv(root / "reports" / "brain" / "shared_outcome_memory.csv")

    if not native_features.empty and "pair" in enriched.columns and "pair" in native_features.columns:
        feature_columns = [
            "pair",
            "wizard_history_feature_count",
            "wizard_same_regime_count",
            "wizard_same_strategy_family_count",
            "wizard_same_venue_count",
            "wizard_same_regime_strategy_venue_count",
            "wizard_verified_same_regime_strategy_venue_count",
            "wizard_same_regime_strategy_venue_win_rate",
            "wizard_same_regime_strategy_venue_mean_return",
            "wizard_same_regime_strategy_venue_mean_drawdown",
            "wizard_learning_feature_state",
            "native_promotion_basis",
        ]
        available = [column for column in feature_columns if column in native_features.columns]
        native_feature_frame = native_features.loc[:, available].drop_duplicates(subset=["pair"], keep="last")
        enriched = enriched.merge(native_feature_frame, how="left", on="pair")

    shared_summary = _shared_outcome_summary_by_pair(shared)
    if not shared_summary.empty and "pair" in enriched.columns:
        enriched = enriched.merge(shared_summary, how="left", on="pair")

    defaults: dict[str, object] = {
        "wizard_history_feature_count": 0,
        "wizard_same_regime_count": 0,
        "wizard_same_strategy_family_count": 0,
        "wizard_same_venue_count": 0,
        "wizard_same_regime_strategy_venue_count": 0,
        "wizard_verified_same_regime_strategy_venue_count": 0,
        "wizard_same_regime_strategy_venue_win_rate": 0.0,
        "wizard_same_regime_strategy_venue_mean_return": 0.0,
        "wizard_same_regime_strategy_venue_mean_drawdown": 0.0,
        "wizard_learning_feature_state": "cold_start",
        "native_promotion_basis": "native_evidence_only",
        "shared_outcome_count": 0,
        "shared_verified_outcome_count": 0,
        "shared_outcome_win_rate": 0.0,
        "shared_outcome_mean_return": 0.0,
        "shared_outcome_mean_drawdown": 0.0,
    }
    for column, default in defaults.items():
        if column not in enriched.columns:
            enriched[column] = default
        else:
            enriched[column] = enriched[column].fillna(default)
    return enriched


def _shared_outcome_summary_by_pair(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "pair" not in frame.columns:
        return pd.DataFrame(
            columns=[
                "pair",
                "shared_outcome_count",
                "shared_verified_outcome_count",
                "shared_outcome_win_rate",
                "shared_outcome_mean_return",
                "shared_outcome_mean_drawdown",
            ]
        )
    working = frame.copy()
    working["pair"] = working.get("pair", pd.Series(dtype=object)).astype(str)
    working["verification_status"] = working.get("verification_status", pd.Series(dtype=object)).astype(str)
    working["outcome_label"] = working.get("outcome_label", pd.Series(dtype=object)).astype(str)
    working["realized_return"] = pd.to_numeric(working.get("realized_return", pd.Series(dtype=float)), errors="coerce")
    working["drawdown"] = pd.to_numeric(working.get("drawdown", pd.Series(dtype=float)), errors="coerce")

    rows: list[dict[str, object]] = []
    for pair, group in working.groupby("pair", dropna=False):
        verified = group[group["verification_status"].str.lower() == "verified"].copy()
        decided = verified[verified["outcome_label"].isin(["win", "loss"])].copy()
        rows.append(
            {
                "pair": str(pair),
                "shared_outcome_count": int(len(group)),
                "shared_verified_outcome_count": int(len(verified)),
                "shared_outcome_win_rate": round(float(decided["outcome_label"].eq("win").mean()), 6) if not decided.empty else 0.0,
                "shared_outcome_mean_return": round(float(verified["realized_return"].dropna().mean()), 6) if not verified["realized_return"].dropna().empty else 0.0,
                "shared_outcome_mean_drawdown": round(float(verified["drawdown"].dropna().mean()), 6) if not verified["drawdown"].dropna().empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _ensure_trade_dataset_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in ["funding_x_bps", "funding_y_bps", "funding_bps_per_day"]:
        if column not in data:
            data[column] = 0.0
    for column in ["hedge_ratio", "beta"]:
        if column not in data:
            data[column] = 1.0
    return data


def _leakage_audit(frame: pd.DataFrame) -> pd.DataFrame:
    trade_ids = frame.get(
        "trade_id", pd.Series(range(len(frame)), index=frame.index)
    ).astype(str)
    duplicate_identity = trade_ids.duplicated(keep=False)
    entry = pd.to_datetime(frame.get("feature_timestamp", pd.Series(dtype=str)), utc=True, errors="coerce", format="mixed")
    label = pd.to_datetime(frame.get("label_timestamp", pd.Series(dtype=str)), utc=True, errors="coerce", format="mixed")
    returns = pd.to_numeric(frame.get("profit_after_cost", pd.Series(np.nan, index=frame.index)), errors="coerce")
    uses_future = entry.notna() & label.notna() & (entry >= label)
    invalid_entry = entry.isna() | ~entry.dt.year.between(2009, pd.Timestamp.now(tz="UTC").year + 1)
    invalid_label = label.isna() | ~label.dt.year.between(2009, pd.Timestamp.now(tz="UTC").year + 1)
    invalid_return = returns.isna() | ~np.isfinite(returns) | returns.lt(-1.0 - 1e-9)
    blocker = np.select(
        [duplicate_identity, invalid_entry, invalid_label, uses_future, invalid_return],
        [
            "duplicate_trade_identity",
            "invalid_or_missing_feature_timestamp",
            "invalid_or_missing_label_timestamp",
            "feature_timestamp_not_before_label_timestamp",
            "invalid_fractional_equity_return_below_minus_one",
        ],
        default="",
    )
    return pd.DataFrame(
        {
            "trade_id": trade_ids,
            "source_venue": frame.get("source_venue", ""),
            "source_path": frame.get("source_path", ""),
            "feature_timestamp": frame.get("feature_timestamp", ""),
            "label_timestamp": frame.get("label_timestamp", ""),
            "uses_future_data": uses_future,
            "uses_dashboard_hindsight": frame.get("uses_dashboard_hindsight", False),
            "feature_completeness_score": frame.get("feature_completeness_score", 0.0),
            "profit_after_cost": returns,
            "return_aggregation": frame.get("return_aggregation", ""),
            "return_unit": frame.get("return_unit", ""),
            "leakage_blocker": blocker,
            "evidence_path": frame.get("evidence_path", ""),
        }
    )


def _model_dataset_from_hardened(frame: pd.DataFrame) -> pd.DataFrame:
    dataset = frame.copy()
    if TARGET_COLUMN not in dataset and "good_trade" in dataset:
        dataset[TARGET_COLUMN] = dataset["good_trade"].astype(int)
    if RETURN_COLUMN not in dataset and "profit_after_cost" in dataset:
        dataset[RETURN_COLUMN] = dataset["profit_after_cost"].astype(float)
    if TIMESTAMP_COLUMN not in dataset and "feature_timestamp" in dataset:
        dataset[TIMESTAMP_COLUMN] = dataset["feature_timestamp"]
    audit_only = [
        "good_trade",
        "profit_after_cost",
        "max_adverse_excursion",
        "max_favorable_excursion",
        "hold_bars",
        "exit_reason",
        "feature_timestamp",
        "label_timestamp",
        "uses_dashboard_hindsight",
        "feature_completeness_score",
        "evidence_path",
        "backtest_label",
        "paper_label",
        "live_label",
    ]
    dataset = dataset.drop(columns=[column for column in audit_only if column in dataset.columns], errors="ignore")
    return dataset


def _trade_gate_metrics(summary: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, object]:
    if summary.empty:
        return {"accepted": False, "blocker": "missing_model_summary"}
    if "selection_score" not in summary.columns:
        return {
            "accepted": False,
            "blocker": "model_selection_isolation_summary_missing",
        }
    replayed_leaderboard = model_selection_leaderboard(predictions)
    if replayed_leaderboard.empty:
        return {
            "accepted": False,
            "blocker": "model_selection_leaderboard_replay_failed",
        }
    best_model = str(replayed_leaderboard.iloc[0]["model_name"])
    best_rows = summary.loc[
        summary.get("model_name", pd.Series("", index=summary.index))
        .astype(str)
        .eq(best_model)
    ]
    if len(best_rows) != 1:
        return {
            "accepted": False,
            "blocker": "selected_model_summary_row_missing_or_duplicated",
        }
    best = best_rows.iloc[0]
    all_selected_predictions = _filter_predictions_to_model(predictions, best_model)
    selected_predictions = _filter_predictions_to_untouched_evaluation(
        all_selected_predictions
    )
    gain_concentration = _model_gain_concentration(selected_predictions)

    def top_gain_share(dimension: str) -> float:
        scoped = gain_concentration.loc[
            gain_concentration.get(
                "dimension", pd.Series("", index=gain_concentration.index)
            ).astype(str).eq(dimension)
        ]
        if scoped.empty:
            return 1.0
        return float(scoped["share_of_positive_returns"].max())

    pair_gain_share = top_gain_share("pair")
    timeframe_gain_share = top_gain_share("timeframe")
    regime_gain_share = top_gain_share("regime")
    strategy_gain_share = top_gain_share("strategy")
    concentration_limit = 0.50
    checks = {
        "selected_model_predictions_present": bool(not selected_predictions.empty),
        "model_winner_replayed": bool(
            _truthy(replayed_leaderboard.iloc[0].get("chosen_model"))
            and int(replayed_leaderboard.iloc[0].get("selection_rank", 0)) == 1
        ),
        "selection_candidate_eligible": _truthy(
            replayed_leaderboard.iloc[0].get("promising")
        ),
        "model_selection_isolation_proven": bool(
            str(best.get("selection_isolation_scheme", ""))
            == MODEL_SELECTION_ISOLATION_SCHEME
            and _model_selection_isolation_proven(all_selected_predictions)
        ),
        "training_only_threshold_calibration_proven": (
            _training_threshold_calibration_proven(all_selected_predictions)
        ),
        "globally_purged_pair_aware_evaluation": bool(
            str(best.get("evaluation_scheme", ""))
            == GLOBAL_PURGED_SPLIT_SCHEME
            and _global_label_purge_proven(selected_predictions)
        ),
        "profit_factor_delta_positive": bool(
            best.get("evaluation_profit_factor_delta", 0.0) > 0
        ),
        "filtered_profit_factor_min": bool(
            best.get("evaluation_median_filtered_profit_factor", 0.0) >= 1.2
        ),
        "filtered_sharpe_positive": bool(
            best.get("evaluation_median_filtered_sharpe", 0.0) > 0.0
        ),
        "filtered_drawdown_max": bool(
            best.get("evaluation_worst_filtered_drawdown", 1.0) <= 0.30
        ),
        "drawdown_delta_nonpositive": bool(
            best.get("evaluation_drawdown_delta", 0.0) <= 0
        ),
        "take_rate_min": bool(
            best.get("evaluation_median_take_rate", 0.0)
            >= MINIMUM_MODEL_GATED_TAKE_RATE
        ),
        "filtered_trades_min": bool(
            best.get("evaluation_total_filtered_trades", 0.0) >= 20
        ),
        "score_buckets_monotonic": bool(
            _score_buckets_monotonic(selected_predictions)
        ),
        "pair_gain_concentration_max": pair_gain_share <= concentration_limit,
        "timeframe_gain_concentration_max": timeframe_gain_share
        <= concentration_limit,
        "regime_gain_concentration_max": regime_gain_share
        <= concentration_limit,
        "strategy_gain_concentration_max": strategy_gain_share
        <= concentration_limit,
    }
    accepted = bool(all(checks.values()))
    failing_checks = [name for name, passed in checks.items() if not passed]
    return {
        "accepted": accepted,
        "best_model": best_model,
        "diagnostic_prediction_scope": "selected_model_untouched_evaluation_only",
        "diagnostic_prediction_rows": len(selected_predictions),
        "all_model_prediction_rows": len(predictions),
        "selection_prediction_rows": len(all_selected_predictions)
        - len(selected_predictions),
        "models_ranked_for_selection": int(len(replayed_leaderboard)),
        "selection_score": float(
            replayed_leaderboard.iloc[0].get("selection_score", 0.0)
        ),
        "model_winner_replayed": checks["model_winner_replayed"],
        "selection_candidate_eligible": checks[
            "selection_candidate_eligible"
        ],
        "profit_factor_delta": float(
            best.get("evaluation_profit_factor_delta", 0.0)
        ),
        "filtered_profit_factor": float(
            best.get("evaluation_median_filtered_profit_factor", 0.0)
        ),
        "filtered_sharpe": float(
            best.get("evaluation_median_filtered_sharpe", 0.0)
        ),
        "filtered_drawdown": float(
            best.get("evaluation_worst_filtered_drawdown", 0.0)
        ),
        "sharpe_delta": float(best.get("evaluation_sharpe_delta", 0.0)),
        "drawdown_delta": float(best.get("evaluation_drawdown_delta", 0.0)),
        "median_take_rate": float(
            best.get("evaluation_median_take_rate", 0.0)
        ),
        "total_filtered_trades": int(
            best.get("evaluation_total_filtered_trades", 0)
        ),
        "score_buckets_monotonic": checks["score_buckets_monotonic"],
        "maximum_gain_concentration": concentration_limit,
        "top_pair_positive_gain_share": pair_gain_share,
        "top_timeframe_positive_gain_share": timeframe_gain_share,
        "top_regime_positive_gain_share": regime_gain_share,
        "top_strategy_positive_gain_share": strategy_gain_share,
        "evaluation_scheme": str(best.get("evaluation_scheme", "")),
        "selection_isolation_scheme": str(
            best.get("selection_isolation_scheme", "")
        ),
        "selection_evaluation_boundary_scheme": str(
            best.get("selection_evaluation_boundary_scheme", "")
        ),
        "chronology_gap_folds": str(best.get("chronology_gap_folds", "")),
        "chronology_gap_fold_count": int(
            best.get("chronology_gap_fold_count", 0) or 0
        ),
        "selection_label_end_boundary": str(
            best.get("selection_label_end_max", "")
        ),
        "untouched_evaluation_start_boundary": str(
            best.get("untouched_evaluation_start", "")
        ),
        "selection_folds": int(best.get("selection_folds", 0) or 0),
        "untouched_evaluation_folds": int(
            best.get("untouched_evaluation_folds", 0) or 0
        ),
        "model_selection_isolation_proven": checks[
            "model_selection_isolation_proven"
        ],
        "training_only_threshold_calibration_proven": checks[
            "training_only_threshold_calibration_proven"
        ],
        "global_label_purge_proven": checks[
            "globally_purged_pair_aware_evaluation"
        ],
        "failing_checks": ";".join(failing_checks),
        "blocker": "" if accepted else "model_acceptance_gates_not_met",
        "created_at": _now(),
        "label_source": "backtest_trained",
    }


def _training_threshold_calibration_proven(predictions: pd.DataFrame) -> bool:
    required = {
        "threshold_calibration_scheme",
        "minimum_training_take_rate",
        "training_take_rate_at_threshold",
        "training_take_rate_floor_pass",
    }
    if predictions.empty or not required.issubset(predictions.columns):
        return False
    minimum = pd.to_numeric(
        predictions["minimum_training_take_rate"], errors="coerce"
    )
    observed = pd.to_numeric(
        predictions["training_take_rate_at_threshold"], errors="coerce"
    )
    floor_pass = predictions["training_take_rate_floor_pass"].map(_truthy)
    return bool(
        predictions["threshold_calibration_scheme"]
        .astype(str)
        .eq(THRESHOLD_CALIBRATION_SCHEME)
        .all()
        and minimum.notna().all()
        and minimum.eq(MINIMUM_TRAINING_TAKE_RATE).all()
        and observed.notna().all()
        and observed.ge(minimum).all()
        and floor_pass.all()
    )


def _global_label_purge_proven(predictions: pd.DataFrame) -> bool:
    required = {
        "split_scheme",
        "global_label_purge",
        "global_label_overlap_rows_after_purge",
        "test_start",
        "train_label_end_max",
    }
    if predictions.empty or not required.issubset(predictions.columns):
        return False
    test_start = pd.to_datetime(
        predictions["test_start"], utc=True, errors="coerce", format="mixed"
    )
    train_label_end = pd.to_datetime(
        predictions["train_label_end_max"],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    overlap_after = pd.to_numeric(
        predictions["global_label_overlap_rows_after_purge"], errors="coerce"
    )
    return bool(
        predictions["split_scheme"].astype(str).eq(
            GLOBAL_PURGED_SPLIT_SCHEME
        ).all()
        and predictions["global_label_purge"].map(_truthy).all()
        and overlap_after.notna().all()
        and overlap_after.eq(0).all()
        and test_start.notna().all()
        and train_label_end.notna().all()
        and train_label_end.lt(test_start).all()
    )


def _feature_schema(dataset: pd.DataFrame) -> dict[str, object]:
    excluded = set(NON_FEATURE_COLUMNS) | {"good_trade", "profit_after_cost", "paper_label", "live_label", "backtest_label"}
    features = [column for column in dataset.columns if column not in excluded]
    return {"schema_version": "trade_gate_v1", "features": features, "created_at": _now(), "label_source": "backtest_trained"}


def _model_acceptance_frame(metrics: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "accepted": metrics.get("accepted", False),
                "blocker": metrics.get("blocker", ""),
                "best_model": metrics.get("best_model", ""),
                "profit_factor_delta": metrics.get("profit_factor_delta", 0.0),
                "filtered_profit_factor": metrics.get("filtered_profit_factor", 0.0),
                "sharpe_delta": metrics.get("sharpe_delta", 0.0),
                "filtered_sharpe": metrics.get("filtered_sharpe", 0.0),
                "drawdown_delta": metrics.get("drawdown_delta", 0.0),
                "filtered_drawdown": metrics.get("filtered_drawdown", 0.0),
                "take_rate": metrics.get("median_take_rate", 0.0),
                "trades": metrics.get("total_filtered_trades", 0),
                "score_buckets_monotonic": metrics.get("score_buckets_monotonic", False),
                "maximum_gain_concentration": metrics.get(
                    "maximum_gain_concentration", 0.50
                ),
                "top_pair_positive_gain_share": metrics.get(
                    "top_pair_positive_gain_share", 1.0
                ),
                "top_timeframe_positive_gain_share": metrics.get(
                    "top_timeframe_positive_gain_share", 1.0
                ),
                "top_regime_positive_gain_share": metrics.get(
                    "top_regime_positive_gain_share", 1.0
                ),
                "top_strategy_positive_gain_share": metrics.get(
                    "top_strategy_positive_gain_share", 1.0
                ),
                "failing_checks": metrics.get("failing_checks", ""),
                "acceptance_reason": "passed"
                if metrics.get("accepted", False)
                else metrics.get("failing_checks", "") or metrics.get("blocker", "blocked"),
            }
        ]
    )


def model_gate_pair_support_report(root: Path = ROOT) -> CommandResult:
    predictions = _read_csv(root / "reports" / "ml" / "model_walkforward_predictions.csv")
    predictions = _filter_predictions_to_selected_model(root, predictions)
    acceptance = _read_csv(root / "reports" / "ml" / "model_gated_acceptance.csv")
    pair_universe = _read_csv(root / "data" / "processed" / "pair_universe.csv")
    training = _read_csv(root / "data" / "ml" / "trade_training_dataset.csv")
    shortlist = paper_candidate_shortlist_rows(root)
    columns = [
        "model_gate_anchor_rank",
        "pair",
        "rows",
        "taken_trades",
        "take_rate",
        "profit_factor",
        "mean_return",
        "support_status",
        "global_model_gate_accepted",
        "global_model_gate_blocker",
        "recommended_repair_action",
        "evidence_path",
    ]
    path = root / "reports" / "ml" / "model_gate_pair_support_report.csv"
    if predictions.empty or "pair" not in predictions.columns:
        frame = pd.DataFrame(columns=columns)
        _write_csv(frame, path)
        return CommandResult(paths={"model_gate_pair_support_report": path}, summary={"rows": 0})

    report = _model_gate_pair_support_frame(predictions, acceptance)
    shortlist_pairs = shortlist.get("pair", pd.Series(dtype=object)).astype(str) if not shortlist.empty and "pair" in shortlist.columns else pd.Series(dtype=object)
    if not shortlist_pairs.empty:
        known_pairs = set(report.get("pair", pd.Series(dtype=object)).astype(str))
        training_pairs = set(training.get("pair", pd.Series(dtype=object)).astype(str)) if not training.empty and "pair" in training.columns else set()
        missing_shortlist_pairs = [pair for pair in shortlist_pairs if pair not in known_pairs]
        if missing_shortlist_pairs:
            additions = []
            for pair in missing_shortlist_pairs:
                additions.append(
                    {
                        "pair": pair,
                        "rows": int(training[training["pair"].astype(str) == pair].shape[0]) if pair in training_pairs else 0,
                        "taken_trades": 0,
                        "take_rate": 0.0,
                        "profit_factor": 0.0,
                        "mean_return": 0.0,
                        "support_status": "pair_missing_from_model_predictions" if pair in training_pairs else "no_model_support",
                        "global_model_gate_accepted": bool(not acceptance.empty and acceptance.get("accepted", pd.Series([False])).astype(bool).iloc[0]),
                        "global_model_gate_blocker": (
                            _text_value(acceptance.iloc[0].get("blocker", "")) if not acceptance.empty else "model_gated_acceptance_missing"
                        ),
                        "recommended_repair_action": (
                            "investigate_walkforward_prediction_coverage_for_pair"
                            if pair in training_pairs
                            else "wait_for_pair_specific_model_support"
                        ),
                        "evidence_path": "data/ml/trade_training_dataset.csv;reports/ml/model_walkforward_predictions.csv;reports/ml/model_gated_acceptance.csv",
                    }
                )
            report = pd.concat([report, pd.DataFrame(additions)], ignore_index=True)
    report = _attach_wizard_daily_pair_support(root, report, acceptance)
    if not pair_universe.empty and "pair" in pair_universe.columns:
        pair_scores = pair_universe[["pair"]].copy()
        pair_scores["combined_score"] = pd.to_numeric(pair_universe.get("combined_score", pd.Series(dtype=object)), errors="coerce").fillna(0.0)
        pair_scores = pair_scores.drop_duplicates(subset=["pair"])
        report = report.merge(pair_scores, on="pair", how="left")
    else:
        report["combined_score"] = 0.0
    shortlist_pairs = set(shortlist_pairs.astype(str)) if not shortlist_pairs.empty else set()
    report["_shortlist_anchor_rank"] = report["pair"].map(lambda value: 0 if str(value) in shortlist_pairs else 1).fillna(1).astype(int)
    support_rank = {"strong_model_support": 0, "weak_model_support": 1, "no_model_support": 2}
    report["_support_rank"] = report["support_status"].map(lambda value: support_rank.get(str(value), 3)).fillna(3)
    report = report.sort_values(
        ["_shortlist_anchor_rank", "_support_rank", "taken_trades", "profit_factor", "mean_return", "combined_score"],
        ascending=[True, True, False, False, False, False],
    ).reset_index(drop=True)
    report.insert(0, "model_gate_anchor_rank", report.index + 1)
    report = report[columns]
    _write_csv(report, path)
    return CommandResult(paths={"model_gate_pair_support_report": path}, summary={"rows": int(len(report))})


def _model_gate_pair_support_frame(predictions: pd.DataFrame, acceptance: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "pair" not in predictions.columns:
        return pd.DataFrame(
            columns=[
                "pair",
                "rows",
                "taken_trades",
                "take_rate",
                "profit_factor",
                "mean_return",
                "support_status",
                "global_model_gate_accepted",
                "global_model_gate_blocker",
                "recommended_repair_action",
                "evidence_path",
            ]
        )
    frame = predictions.copy()
    frame["pair"] = frame.get("pair", pd.Series(dtype=object)).astype(str)
    take_col = "shadow_take" if "shadow_take" in frame.columns else "model_take" if "model_take" in frame.columns else ""
    return_col = RETURN_COLUMN if RETURN_COLUMN in frame.columns else "realized_return" if "realized_return" in frame.columns else ""
    if not take_col:
        frame["_fallback_take"] = False
        take_col = "_fallback_take"
    else:
        frame[take_col] = frame.get(take_col, pd.Series(dtype=object)).fillna(False).astype(bool)
    if not return_col:
        frame["_fallback_return"] = pd.Series(0.0, index=frame.index, dtype=float)
        return_col = "_fallback_return"
    else:
        frame[return_col] = pd.to_numeric(frame.get(return_col, pd.Series(dtype=object)), errors="coerce")

    accepted = False
    blocker = ""
    if not acceptance.empty:
        acceptance_row = acceptance.iloc[0]
        accepted = bool(acceptance_row.get("accepted", False))
        blocker = (
            _text_value(acceptance_row.get("failing_checks", ""))
            or _text_value(acceptance_row.get("blocker", ""))
            or _text_value(acceptance_row.get("acceptance_reason", ""))
        )

    rows: list[dict[str, object]] = []
    for pair, subset in frame.groupby("pair", dropna=False):
        taken = subset[subset[take_col]].copy()
        returns = pd.to_numeric(taken[return_col], errors="coerce").dropna()
        gross_profit = float(returns[returns > 0].sum()) if not returns.empty else 0.0
        gross_loss = float(-returns[returns < 0].sum()) if not returns.empty else 0.0
        profit_factor = float("inf") if gross_loss == 0.0 and gross_profit > 0.0 else (gross_profit / gross_loss if gross_loss > 0.0 else 0.0)
        mean_return = float(returns.mean()) if not returns.empty else 0.0
        take_rate = float(len(taken) / len(subset)) if len(subset) else 0.0
        support_status = "weak_model_support"
        if len(taken) == 0:
            support_status = "no_model_support"
        elif len(returns) >= 10 and profit_factor >= 1.2 and mean_return > 0:
            support_status = "strong_model_support"
        if support_status == "strong_model_support" and not accepted:
            action = "preserve strong pair-specific model support while improving global model gate acceptance"
        elif support_status == "strong_model_support":
            action = "pair_support_ready_for_global_model_gate_confirmation"
        elif support_status == "weak_model_support":
            action = "improve pair-specific model support quality"
        else:
            action = "wait_for_pair_specific_model_support"
        rows.append(
            {
                "pair": str(pair),
                "rows": int(len(subset)),
                "taken_trades": int(len(returns)),
                "take_rate": take_rate,
                "profit_factor": profit_factor,
                "mean_return": mean_return,
                "support_status": support_status,
                "global_model_gate_accepted": accepted,
                "global_model_gate_blocker": blocker,
                "recommended_repair_action": action,
                "evidence_path": "reports/ml/model_walkforward_predictions.csv;reports/ml/model_gated_acceptance.csv",
            }
        )
    return pd.DataFrame(rows)


def _attach_wizard_daily_pair_support(root: Path, report: pd.DataFrame, acceptance: pd.DataFrame) -> pd.DataFrame:
    readiness = _read_csv(root / "reports" / "active" / "wizard_ou_spread_daily_paper_readiness_latest.csv")
    verification = _read_csv(root / "reports" / "active" / "wizard_local_verification_batch.csv")
    if readiness.empty or verification.empty:
        return report

    accepted = bool(not acceptance.empty and acceptance.get("accepted", pd.Series([False])).astype(bool).iloc[0])
    blocker = _text_value(acceptance.iloc[0].get("blocker", "")) if not acceptance.empty else "model_gated_acceptance_missing"
    if not blocker and not acceptance.empty:
        blocker = (
            _text_value(acceptance.iloc[0].get("failing_checks", ""))
            or _text_value(acceptance.iloc[0].get("acceptance_reason", ""))
        )

    readiness_pairs = readiness.copy()
    readiness_pairs["pair"] = readiness_pairs.get("pair", pd.Series(dtype=object)).astype(str)
    verification_pairs = verification.copy()
    verification_pairs["pair"] = verification_pairs.get("pair", pd.Series(dtype=object)).astype(str)
    verification_index = verification_pairs.set_index("pair", drop=False)

    rows = report.copy()
    known_pairs = set(rows.get("pair", pd.Series(dtype=object)).astype(str))
    additions: list[dict[str, object]] = []
    for _, row in readiness_pairs.iterrows():
        if str(row.get("local_acceptance", "")).upper() != "ACCEPT":
            continue
        if str(row.get("forward_walk_status", "")).lower() != "pass":
            continue
        pair = str(row.get("pair", "")).strip()
        if not pair:
            continue
        verification_row = verification_index.loc[pair] if pair in verification_index.index else None
        if isinstance(verification_row, pd.DataFrame):
            verification_row = verification_row.iloc[0]
        taken_trades = int(
            pd.to_numeric(
                pd.Series([verification_row.get("local_closed_trades", 0) if verification_row is not None else 0]),
                errors="coerce",
            ).fillna(0).iloc[0]
        )
        profit_factor = float(pd.to_numeric(pd.Series([row.get("local_profit_factor", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        mean_return = float(pd.to_numeric(pd.Series([row.get("local_total_return", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        take_rate = 1.0 if taken_trades > 0 else 0.0
        payload = {
            "pair": pair,
            "rows": taken_trades,
            "taken_trades": taken_trades,
            "take_rate": take_rate,
            "profit_factor": profit_factor,
            "mean_return": mean_return,
            "support_status": "strong_model_support" if taken_trades >= 10 and profit_factor >= 1.2 and mean_return > 0 else "weak_model_support",
            "global_model_gate_accepted": accepted,
            "global_model_gate_blocker": blocker,
            "recommended_repair_action": (
                "preserve wizard-validated pair support while improving global model gate acceptance"
                if taken_trades >= 10 and profit_factor >= 1.2 and mean_return > 0
                else "improve wizard pair-specific support quality"
            ),
            "evidence_path": "reports/active/wizard_ou_spread_daily_paper_readiness_latest.csv;reports/active/wizard_local_verification_batch.csv",
        }
        if pair in known_pairs:
            mask = rows["pair"].astype(str) == pair
            existing = rows.loc[mask].iloc[0]
            existing_rank = 0 if str(existing.get("support_status", "")) == "strong_model_support" else 1
            candidate_rank = 0 if payload["support_status"] == "strong_model_support" else 1
            existing_trades = int(pd.to_numeric(pd.Series([existing.get("taken_trades", 0)]), errors="coerce").fillna(0).iloc[0])
            if candidate_rank < existing_rank or (candidate_rank == existing_rank and taken_trades > existing_trades):
                for key, value in payload.items():
                    rows.loc[mask, key] = value
        else:
            additions.append(payload)

    if additions:
        rows = pd.concat([rows, pd.DataFrame(additions)], ignore_index=True)
    return rows


def _filter_predictions_to_selected_model(root: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "model_name" not in predictions.columns:
        return predictions
    manifest_path = root / "reports" / "ml" / "trade_gate" / "ml_trade_filter_manifest.json"
    chosen_model = ""
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            chosen_model = str(payload.get("chosen_model", "") or "").strip()
        except Exception:
            chosen_model = ""
    if not chosen_model:
        metrics_path = root / "models" / "trade_gate" / "metrics.json"
        if metrics_path.exists():
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                chosen_model = str(payload.get("best_model", "") or "").strip()
            except Exception:
                chosen_model = ""
    if not chosen_model:
        return predictions
    selected = _filter_predictions_to_model(predictions, chosen_model)
    if "selection_phase" not in selected.columns:
        return selected
    return _filter_predictions_to_untouched_evaluation(selected)


def _filter_predictions_to_model(
    predictions: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    if (
        predictions.empty
        or not model_name
        or "model_name" not in predictions.columns
    ):
        return pd.DataFrame(columns=predictions.columns)
    return predictions.loc[
        predictions["model_name"].astype(str).eq(model_name)
    ].copy()


def _filter_predictions_to_untouched_evaluation(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    if predictions.empty or "selection_phase" not in predictions.columns:
        return pd.DataFrame(columns=predictions.columns)
    return predictions.loc[
        predictions["selection_phase"]
        .astype(str)
        .eq(UNTOUCHED_EVALUATION_PHASE)
    ].copy()


def _model_selection_isolation_proven(
    predictions: pd.DataFrame,
    *,
    minimum_selection_folds: int = 1,
    minimum_evaluation_folds: int = 1,
) -> bool:
    required = {
        "trade_id",
        "fold",
        "selection_phase",
        "selection_isolation_scheme",
        "selection_evaluation_boundary_scheme",
        "chronology_gap_folds",
        "selection_label_end_boundary",
        "untouched_evaluation_start_boundary",
        "entry_timestamp",
        "exit_timestamp",
    }
    if predictions.empty or not required.issubset(predictions.columns):
        return False
    if (
        predictions["trade_id"].fillna("").astype(str).str.strip().eq("").any()
        or predictions["trade_id"].astype(str).duplicated().any()
        or not predictions["selection_isolation_scheme"]
        .astype(str)
        .eq(MODEL_SELECTION_ISOLATION_SCHEME)
        .all()
        or not predictions["selection_evaluation_boundary_scheme"]
        .astype(str)
        .eq(MODEL_SELECTION_BOUNDARY_SCHEME)
        .all()
    ):
        return False
    phases = predictions["selection_phase"].astype(str)
    if set(phases) != {MODEL_SELECTION_PHASE, UNTOUCHED_EVALUATION_PHASE}:
        return False
    selection = predictions.loc[phases.eq(MODEL_SELECTION_PHASE)].copy()
    evaluation = predictions.loc[
        phases.eq(UNTOUCHED_EVALUATION_PHASE)
    ].copy()
    selection_folds = set(selection["fold"].dropna().astype(str)) - {""}
    evaluation_folds = set(evaluation["fold"].dropna().astype(str)) - {""}
    if (
        len(selection_folds) < minimum_selection_folds
        or len(evaluation_folds) < minimum_evaluation_folds
        or selection_folds.intersection(evaluation_folds)
    ):
        return False
    selection_entry = pd.to_datetime(
        selection["entry_timestamp"], utc=True, errors="coerce", format="mixed"
    )
    selection_exit = pd.to_datetime(
        selection["exit_timestamp"], utc=True, errors="coerce", format="mixed"
    )
    evaluation_entry = pd.to_datetime(
        evaluation["entry_timestamp"], utc=True, errors="coerce", format="mixed"
    )
    evaluation_exit = pd.to_datetime(
        evaluation["exit_timestamp"], utc=True, errors="coerce", format="mixed"
    )
    selection_boundary = pd.to_datetime(
        predictions["selection_label_end_boundary"],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    evaluation_boundary = pd.to_datetime(
        predictions["untouched_evaluation_start_boundary"],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    if any(
        series.isna().any()
        for series in (
            selection_entry,
            selection_exit,
            evaluation_entry,
            evaluation_exit,
            selection_boundary,
            evaluation_boundary,
        )
    ):
        return False
    if (
        selection_boundary.nunique() != 1
        or evaluation_boundary.nunique() != 1
        or selection_boundary.iloc[0] >= evaluation_boundary.iloc[0]
        or selection_exit.max() != selection_boundary.iloc[0]
        or evaluation_entry.min() != evaluation_boundary.iloc[0]
    ):
        return False
    return bool(
        selection_entry.max() < evaluation_entry.min()
        and selection_exit.max() < evaluation_entry.min()
        and selection_entry.lt(selection_exit).all()
        and evaluation_entry.lt(evaluation_exit).all()
    )


def _score_bucket_report(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "probability_profitable" not in predictions:
        return pd.DataFrame(columns=["score_bucket", "rows", "mean_return", "monotonic"])
    frame = predictions.copy()
    frame["score_bucket"] = pd.cut(frame["probability_profitable"], bins=[-0.01, 0.55, 0.70, 1.01], labels=["skip", "reduced", "full"])
    grouped = frame.groupby("score_bucket", observed=False)[RETURN_COLUMN].agg(["count", "mean"]).reset_index()
    grouped = grouped.rename(columns={"count": "rows", "mean": "mean_return"})
    grouped["monotonic"] = _score_buckets_monotonic(predictions)
    grouped["model_name"] = (
        str(predictions["model_name"].iloc[0])
        if "model_name" in predictions and not predictions.empty
        else ""
    )
    grouped["diagnostic_scope"] = "selected_model_oos_only"
    return grouped


def _score_buckets_monotonic(predictions: pd.DataFrame) -> bool:
    report = _score_bucket_report_raw(predictions)
    if len(report) < 2:
        return False
    means = report["mean_return"].tolist()
    return all(later >= earlier for earlier, later in zip(means, means[1:]))


def _score_bucket_report_raw(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "probability_profitable" not in predictions or RETURN_COLUMN not in predictions:
        return pd.DataFrame()
    frame = predictions.copy()
    frame["score_bucket"] = pd.cut(frame["probability_profitable"], bins=[-0.01, 0.55, 0.70, 1.01], labels=["skip", "reduced", "full"])
    return frame.groupby("score_bucket", observed=False)[RETURN_COLUMN].mean().dropna().reset_index(name="mean_return")


def _model_pair_concentration(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "pair" not in predictions:
        return pd.DataFrame(columns=["pair", "rows", "taken_rows", "share_of_taken"])
    take_col = "shadow_take" if "shadow_take" in predictions else "model_take"
    if take_col not in predictions:
        if "probability_profitable" in predictions:
            predictions = predictions.assign(model_take=predictions["probability_profitable"] >= 0.70)
            take_col = "model_take"
        else:
            predictions = predictions.assign(model_take=False)
            take_col = "model_take"
    grouped = predictions.groupby("pair").agg(rows=("pair", "size"), taken_rows=(take_col, "sum")).reset_index()
    total = max(float(grouped["taken_rows"].sum()), 1.0)
    grouped["share_of_taken"] = grouped["taken_rows"] / total
    grouped["model_name"] = (
        str(predictions["model_name"].iloc[0])
        if "model_name" in predictions and not predictions.empty
        else ""
    )
    grouped["diagnostic_scope"] = "selected_model_oos_only"
    return grouped.sort_values("share_of_taken", ascending=False)


def _model_gain_concentration(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dimension",
        "value",
        "rows",
        "taken_rows",
        "taken_return_sum",
        "positive_return_sum",
        "share_of_taken",
        "share_of_positive_returns",
        "model_name",
        "diagnostic_scope",
    ]
    if predictions.empty or RETURN_COLUMN not in predictions:
        return pd.DataFrame(columns=columns)
    frame = predictions.copy()
    if "shadow_take" in frame:
        taken = frame["shadow_take"].map(_truthy)
    elif "probability_profitable" in frame:
        threshold = pd.to_numeric(
            frame.get("threshold", pd.Series(0.70, index=frame.index)),
            errors="coerce",
        ).fillna(0.70)
        taken = pd.to_numeric(
            frame["probability_profitable"], errors="coerce"
        ).fillna(0.0).ge(threshold)
    else:
        taken = pd.Series(False, index=frame.index)
    frame["_taken"] = taken
    frame["_return"] = pd.to_numeric(
        frame[RETURN_COLUMN], errors="coerce"
    ).fillna(0.0)
    frame["_positive_return"] = frame["_return"].clip(lower=0.0).where(
        frame["_taken"], 0.0
    )
    total_taken = max(int(frame["_taken"].sum()), 1)
    total_positive = float(frame["_positive_return"].sum())
    model_name = (
        str(frame["model_name"].iloc[0]) if "model_name" in frame else ""
    )
    rows: list[dict[str, object]] = []
    dimensions = {
        "pair": "pair",
        "timeframe": "timeframe",
        "regime": "regime",
        "strategy": "strategy_name",
    }
    for dimension, column in dimensions.items():
        if column not in frame.columns:
            continue
        for value, group in frame.groupby(column, dropna=False, sort=True):
            group_taken = group["_taken"]
            positive_sum = float(group["_positive_return"].sum())
            rows.append(
                {
                    "dimension": dimension,
                    "value": str(value),
                    "rows": len(group),
                    "taken_rows": int(group_taken.sum()),
                    "taken_return_sum": float(
                        group.loc[group_taken, "_return"].sum()
                    ),
                    "positive_return_sum": positive_sum,
                    "share_of_taken": float(group_taken.sum() / total_taken),
                    "share_of_positive_returns": (
                        positive_sum / total_positive if total_positive > 0 else 1.0
                    ),
                    "model_name": model_name,
                    "diagnostic_scope": "selected_model_oos_only",
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["dimension", "share_of_positive_returns"],
        ascending=[True, False],
    )


def _model_gated_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    threshold_col = "threshold"
    take = predictions["probability_profitable"] >= predictions.get(threshold_col, pd.Series(0.70, index=predictions.index))
    rows = []
    for name, mask in [
        ("raw_strategy", pd.Series(True, index=predictions.index)),
        ("model_gated_strategy", take),
        ("model_sized_strategy", take | (predictions["probability_profitable"] >= 0.55)),
    ]:
        returns = predictions.loc[mask, RETURN_COLUMN].astype(float)
        rows.append(_return_summary(name, returns, len(predictions), predictions.loc[mask]))
    return pd.DataFrame(rows)


def _return_summary(name: str, returns: pd.Series, total_rows: int, subset: pd.DataFrame) -> dict[str, object]:
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = float(gains / losses) if losses else float("inf") if gains > 0 else 0.0
    sample_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / sample_std * np.sqrt(len(returns))) if sample_std else 0.0
    drawdown = _max_drawdown(returns)
    return {
        "variant": name,
        "trades": int(len(returns)),
        "take_rate": float(len(returns) / max(total_rows, 1)),
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "expectancy": float(returns.mean()) if len(returns) else 0.0,
        "total_return": float((1.0 + returns).clip(lower=0.0).prod() - 1.0) if len(returns) else 0.0,
        "cost_drag": float(subset.get("trade_cost_drag", pd.Series(dtype=float)).astype(float).sum()) if not subset.empty else 0.0,
        "regime": "mixed",
        "pair": "mixed",
        "timeframe": "mixed",
        "strategy": "mixed",
        "acceptance_reason": "",
    }


def _model_gated_acceptance(comparison: pd.DataFrame) -> pd.DataFrame:
    raw = comparison[comparison["variant"] == "raw_strategy"].iloc[0]
    gated = comparison[comparison["variant"] == "model_gated_strategy"].iloc[0]
    accepted = bool(
        gated["profit_factor"] > raw["profit_factor"]
        and gated["profit_factor"] >= 1.2
        and gated["sharpe"] > 0.0
        and gated["max_drawdown"] <= 0.30
        and gated["total_return"] > 0.0
        and gated["max_drawdown"] <= raw["max_drawdown"]
        and gated["trades"] >= 20
        and gated["take_rate"] >= MINIMUM_MODEL_GATED_TAKE_RATE
    )
    return pd.DataFrame(
        [
            {
                "accepted": accepted,
                "blocker": "" if accepted else "model_gated_backtest_not_accepted",
                "raw_profit_factor": raw["profit_factor"],
                "gated_profit_factor": gated["profit_factor"],
                "raw_drawdown": raw["max_drawdown"],
                "gated_drawdown": gated["max_drawdown"],
                "gated_trades": gated["trades"],
                "gated_take_rate": gated["take_rate"],
                "acceptance_reason": "passed" if accepted else "model_did_not_clear_incremental_edge_gates",
            }
        ]
    )


def _model_failure_attribution(predictions: pd.DataFrame, acceptance: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    accepted = bool(not acceptance.empty and acceptance["accepted"].astype(bool).iloc[0])
    rows = []
    if not accepted:
        blocker = acceptance["blocker"].iloc[0] if not acceptance.empty else "missing_acceptance"
        failing_checks = _text_value(acceptance["failing_checks"].iloc[0]) if not acceptance.empty and "failing_checks" in acceptance.columns else ""
        detail = blocker
        if failing_checks:
            detail = f"{detail}; failing_checks={failing_checks}"
        rows.append({"failure": "incremental_edge_gate", "detail": detail})
    if not _score_buckets_monotonic(predictions):
        rows.append({"failure": "score_bucket_monotonicity", "detail": "higher_scores_did_not_imply_better_returns"})
    concentration = _model_pair_concentration(predictions)
    if not concentration.empty and concentration["share_of_taken"].iloc[0] > 0.65:
        rows.append({"failure": "pair_concentration", "detail": str(concentration["pair"].iloc[0])})
    pair_support_frame = _model_gate_pair_support_frame(predictions, acceptance)
    if not pair_support_frame.empty:
        shortlist = paper_candidate_shortlist_rows(root)
        shortlist_pairs = set(shortlist.get("pair", pd.Series(dtype=object)).astype(str)) if not shortlist.empty and "pair" in shortlist.columns else set()
        support_rank = {"strong_model_support": 0, "weak_model_support": 1, "no_model_support": 2}
        pair_support_frame = pair_support_frame.copy()
        pair_support_frame["_shortlist_anchor_rank"] = pair_support_frame["pair"].map(lambda value: 0 if str(value) in shortlist_pairs else 1).fillna(1)
        pair_support_frame["_support_rank"] = pair_support_frame["support_status"].map(lambda value: support_rank.get(str(value), 3)).fillna(3)
        top_anchor = pair_support_frame.sort_values(
            ["_shortlist_anchor_rank", "_support_rank", "taken_trades", "profit_factor", "mean_return"],
            ascending=[True, True, False, False, False],
        ).iloc[0]
        rows.append(
            {
                "failure": "anchor_candidate_repair_path",
                "detail": (
                    f"{_text_value(top_anchor.get('pair', ''))}; "
                    f"support={_text_value(top_anchor.get('support_status', ''))}; "
                    f"action={_text_value(top_anchor.get('recommended_repair_action', ''))}"
                ),
            }
        )
    return pd.DataFrame(rows or [{"failure": "", "detail": "no_failure_detected"}])


def _attach_venue_route_recommendations(
    pair_universe: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the current independent route decision without mutating source evidence."""
    frame = pair_universe.copy()
    route_columns = {
        "recommended_research_venue": "route_recommended_research_venue",
        "recommended_execution_venue": "route_recommended_execution_venue",
        "recommended_paper_venue": "route_recommended_paper_venue",
        "route_status": "route_status",
        "selection_reason": "route_selection_reason",
        "blockers": "route_blockers",
        "next_step": "route_next_step",
        "evidence_path": "route_evidence_path",
    }
    if frame.empty:
        return frame.reindex(columns=[*frame.columns, *route_columns.values()])
    recommendations = _normalize_venue_route_recommendations(recommendations)
    if recommendations.empty or "pair" not in recommendations.columns:
        for column in route_columns.values():
            frame[column] = ""
        return frame
    available = ["pair", *[column for column in route_columns if column in recommendations.columns]]
    routes = recommendations.loc[:, available].drop_duplicates("pair", keep="last").rename(columns=route_columns)
    return frame.merge(routes, on="pair", how="left").fillna({column: "" for column in route_columns.values()})


def _normalize_venue_route_recommendations(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Keep blank CSV route fields blank after pandas round-trips them as NaN."""
    frame = recommendations.copy()
    for column in (
        "recommended_research_venue",
        "recommended_execution_venue",
        "recommended_paper_venue",
        "route_status",
        "selection_reason",
        "blockers",
        "next_step",
        "evidence_path",
    ):
        if column in frame.columns:
            frame[column] = frame[column].map(_text_value)
    return frame


def _live_signal_rows(pair_universe: pd.DataFrame, acceptance: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "pair",
        "strategy",
        "timeframe",
        "feature_timestamp",
        "model_version",
        "schema_version",
        "feature_completeness_score",
        "trade_quality_score",
        "action",
        "reason",
        "blocker",
        "recommended_execution_venue",
        "route_status",
        "evidence_path",
    ]
    if pair_universe.empty:
        return pd.DataFrame(columns=columns)
    model_ok = bool(not acceptance.empty and acceptance.get("accepted", pd.Series([False])).astype(bool).iloc[0])
    rows = []
    for _, row in pair_universe.head(100).iterrows():
        blocker = ""
        action = "watch"
        route_venue = _text_value(row.get("route_recommended_execution_venue", ""))
        route_status = _text_value(row.get("route_status", ""))
        route_blocker = _text_value(row.get("route_blockers", ""))
        route_reason = _text_value(row.get("route_selection_reason", ""))
        reason = route_reason or str(row.get("decision_reason", ""))
        if row.get("decision_bucket") != "PROMOTE":
            blocker = str(row.get("missing_data_reason", "")) or "not_promoted"
            action = "blocked"
        elif not route_venue:
            blocker = route_blocker or "no_evidence_complete_execution_route"
            action = "blocked"
        elif not model_ok:
            blocker = "model_gate_not_accepted"
            action = "blocked"
        rows.append(
            {
                "pair": row.get("pair", ""),
                "strategy": "candidate_strategy",
                "timeframe": row.get("available_timeframes", ""),
                "feature_timestamp": row.get("source_timestamp", ""),
                "model_version": "trade_gate_v1",
                "schema_version": "trade_gate_v1",
                "feature_completeness_score": 1.0 if not blocker else 0.5,
                "trade_quality_score": "",
                "action": action,
                "reason": reason,
                "blocker": blocker,
                "recommended_execution_venue": route_venue,
                "route_status": route_status,
                "evidence_path": ";".join(
                    value
                    for value in [
                        _text_value(row.get("evidence_path", "")),
                        _text_value(row.get("route_evidence_path", "")),
                    ]
                    if value
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _strategy_tests_dashboard(root: Path) -> pd.DataFrame:
    acceptance = _read_csv(root / "reports" / "acceptance_report.csv")
    checklist = _read_csv(root / "reports" / "strategy_acceptance_checklist.csv")
    if not acceptance.empty:
        return acceptance
    return checklist


def _model_training_dashboard(acceptance: pd.DataFrame) -> pd.DataFrame:
    if acceptance.empty:
        return pd.DataFrame([{"accepted": False, "blocker": "model_gated_acceptance_missing", "reason": "run train-trade-gate and run-model-gated-backtest"}])
    return acceptance


def _feature_readiness_dashboard(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ml_comparison = _read_csv(root / "reports" / "ml" / "model_backtest_comparison.csv")
    if ml_comparison.empty:
        rows.append(
            {
                "area": "feature_importance_models",
                "ready": False,
                "blocker": "model_backtest_comparison_missing",
                "detail": "run train-trade-gate and run-model-gated-backtest",
            }
        )
    else:
        unavailable = sorted(
            {
                str(reason)
                for reason in ml_comparison.get("unavailable_reason", pd.Series(dtype=str)).dropna().astype(str)
                if reason.strip()
            }
        )
        rows.append(
            {
                "area": "feature_importance_models",
                "ready": not unavailable,
                "blocker": ";".join(unavailable),
                "detail": "xgboost/shap-style diagnostics ready" if not unavailable else "install optional ML backends to match Crypto Wizards feature-ranking workflow",
            }
        )
    strategy = _read_csv(root / "reports" / "strategy_summary.csv")
    if strategy.empty:
        rows.append(
            {
                "area": "strategy_feature_gap",
                "ready": False,
                "blocker": "strategy_summary_missing",
                "detail": "rerun acceptance evidence",
            }
        )
    else:
        top = strategy.sort_values("total_trades", ascending=False).head(5)
        top_names = ",".join(top["strategy_name"].astype(str).tolist())
        rows.append(
            {
                "area": "strategy_feature_gap",
                "ready": True,
                "blocker": "",
                "detail": f"highest-activity strategies for feature diagnosis: {top_names}",
            }
        )
    return pd.DataFrame(rows)


def _data_health_rows(
    system: pd.DataFrame,
    pair_universe: pd.DataFrame,
    route_recommendations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    if not system.empty:
        for _, row in system.iterrows():
            rows.append({"area": row.get("check", ""), "ready": row.get("ready", False), "blocker": row.get("blocker", ""), "evidence_path": row.get("evidence_path", "")})
    if not pair_universe.empty:
        stale_reason = pair_universe.get("stale_reason")
        if stale_reason is None:
            stale_reason = pd.Series([""] * len(pair_universe), index=pair_universe.index, dtype=str)
        else:
            stale_reason = stale_reason.astype(str)
        stale = pair_universe[stale_reason != ""]
        rows.append({"area": "pair_universe_stale_rows", "ready": stale.empty, "blocker": f"stale_rows={len(stale)}" if not stale.empty else "", "evidence_path": "data/processed/pair_universe.csv"})
    if route_recommendations is not None and not route_recommendations.empty:
        route_recommendations = _normalize_venue_route_recommendations(route_recommendations)
        missing_execution = route_recommendations[
            route_recommendations.get("recommended_execution_venue", pd.Series("", index=route_recommendations.index)).map(_text_value).eq("")
        ]
        stale_routes = route_recommendations[
            route_recommendations.get("blockers", pd.Series("", index=route_recommendations.index)).astype(str).str.contains("stale_or_missing_venue_context", na=False)
        ]
        rows.append(
            {
                "area": "venue_routes_without_execution_evidence",
                "ready": missing_execution.empty,
                "blocker": f"pairs_without_execution_route={len(missing_execution)}" if not missing_execution.empty else "",
                "evidence_path": "reports/active/venue_route_recommendations.csv",
            }
        )
        rows.append(
            {
                "area": "venue_routes_with_stale_context",
                "ready": stale_routes.empty,
                "blocker": f"stale_route_pairs={len(stale_routes)}" if not stale_routes.empty else "",
                "evidence_path": "reports/active/venue_route_recommendations.csv",
            }
        )
    return pd.DataFrame(rows)


def _api_credit_usage_rows() -> pd.DataFrame:

    def _is_present(value: object) -> bool:
        if pd.isna(value):
            return False
        text = str(value).strip().lower()
        return text not in {"", "nan", "none", "null"}

    rows: list[dict[str, object]] = [
        {"source": "crypto_wizards", "usage_known": False, "blocker": "credit_usage_not_captured", "evidence_path": "reports/dashboard_integration_summary.md"},
        {"source": "dydx", "usage_known": True, "blocker": "", "evidence_path": "reports/dydx_live_market_selector.csv"},
    ]

    manifest = _read_csv(ROOT / "reports" / "active" / "apify_source_capture_manifest.csv")
    if manifest.empty:
        rows.append({"source": "apify", "usage_known": False, "blocker": "credit_usage_not_captured", "evidence_path": "docs/apify_integration.md"})
        return pd.DataFrame(rows)

    for _, row in manifest.iterrows():
        source_id = str(row.get("source_id", "")).strip()
        source_category = source_id.split("/")[-1] or source_id
        usage_currency = str(row.get("usage_currency", "")).strip()
        usage_amount = pd.to_numeric(row.get("usage_amount", pd.NA), errors="coerce")
        usage_credits = pd.to_numeric(row.get("usage_credits", pd.NA), errors="coerce")
        usage_known = _is_present(usage_amount) or _is_present(usage_credits) or _is_present(usage_currency) or _is_present(row.get("run_id"))
        rows.append(
            {
                "source": f"apify:{source_category}",
                "usage_known": bool(usage_known),
                "blocker": "" if usage_known else "credit_usage_not_captured",
                "evidence_path": str(row.get("evidence", "docs/apify_integration.md")),
                "last_run_status": str(row.get("run_status", "")),
                "run_id": str(row.get("run_id", "")),
                "usage_currency": usage_currency,
                "usage_amount": None if pd.isna(usage_amount) else float(usage_amount),
                "usage_credits": None if pd.isna(usage_credits) else float(usage_credits),
                "sample_rows": int(row.get("sample_rows", 0) or 0),
            }
        )

    return pd.DataFrame(rows)


def _refresh_dashboard_dependencies(root: Path = ROOT, *, refresh_profile: str = "deep") -> pd.DataFrame:
    if refresh_profile not in DASHBOARD_REFRESH_PROFILES:
        raise ValueError(f"unsupported dashboard refresh profile: {refresh_profile}")

    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    timestamp = _now()
    rows: list[dict[str, object]] = []

    def record(stage: str, status: str, reason: str, evidence: Path) -> None:
        rows.append(
            {
                "refresh_profile": refresh_profile,
                "stage": stage,
                "status": status,
                "reason": reason,
                "evidence_path": str(evidence),
                "timestamp_utc": timestamp,
            }
        )

    if root != ROOT:
        record("dependency_refresh", "not_auto_refreshed", "noncanonical_root_uses_existing_test_artifacts", active)
        frame = pd.DataFrame(rows, columns=DASHBOARD_REFRESH_COLUMNS)
        _write_csv(frame, active / "dashboard_refresh_status.csv")
        return frame

    reports = root / "reports"
    if refresh_profile == "deep":
        from quant_platform.binance_spot import build_binance_spot_lane_report
        from quant_platform.wizard_evidence import (
            build_wizard_exploratory_cost_sensitivity,
            build_wizard_mode_replay_capability,
            build_wizard_replay_handoff,
            build_wizard_research_pack,
        )
        from quant_platform.wizard_mode_comparison import build_wizard_mode_comparison

        wizard = build_wizard_research_pack(root)
        record("wizard_research_pack", "rebuilt", "local_capture_evidence_reprocessed", wizard.paths["wizard_evidence"])
        record(
            "wizard_mode_matrix_capture_queue",
            "rebuilt",
            "seven_mode_capture_work_recomputed",
            wizard.paths["wizard_mode_matrix_capture_queue"],
        )
        multi_venue = build_multi_venue_history_readiness(root)
        record("multi_venue_history_readiness", "rebuilt", "current_wizard_shortlist_used", multi_venue.paths["multi_venue_history_readiness"])
        universe = build_pair_universe(root)
        record("pair_universe", "rebuilt", "current_discovery_and_local_evidence_scored", universe.paths["pair_universe"])
        binance = build_binance_spot_lane_report(root)
        record("binance_spot_lane", "rebuilt", "history_and_cost_readiness_rechecked", binance.paths["binance_spot_pair_readiness"])
        replay_handoff = build_wizard_replay_handoff(root)
        record("wizard_replay_handoff", "rebuilt", "exact_mode_replay_inputs_rechecked", replay_handoff.paths["wizard_replay_handoff"])
        mode_capability = build_wizard_mode_replay_capability(root)
        record(
            "wizard_mode_replay_capability",
            "rebuilt",
            "mode_specific_local_replay_inputs_rechecked",
            mode_capability.paths["wizard_mode_replay_capability"],
        )
        mode_comparison = build_wizard_mode_comparison(root)
        record(
            "wizard_mode_comparison",
            "rebuilt",
            "confirmed_mode_settings_compared_on_matching_local_history_when_available",
            mode_comparison.paths["wizard_mode_comparison"],
        )
        sensitivity = build_wizard_exploratory_cost_sensitivity(root)
        record(
            "wizard_exploratory_cost_sensitivity",
            "rebuilt",
            "provisional_cost_scenarios_rechecked",
            sensitivity.paths["wizard_exploratory_cost_sensitivity"],
        )
    else:
        monitor_artifacts = {
            "wizard_discovery": active / "wizard_discovery_current.csv",
            "multi_venue_history_readiness": current_multi_venue_history_readiness_path(root),
            "pair_universe": root / "data" / "processed" / "pair_universe.csv",
            "binance_spot_lane": active / "binance_spot_pair_readiness.csv",
            "wizard_mode_matrix_capture_queue": active / "wizard_mode_matrix_capture_queue.csv",
            "wizard_mode_replay_capability": active / "wizard_mode_replay_capability.csv",
            "wizard_mode_comparison": active / "wizard_mode_comparison.csv",
            "wizard_exploratory_cost_sensitivity": active / "wizard_exploratory_cost_sensitivity.csv",
        }
        for stage, path in monitor_artifacts.items():
            record(
                stage,
                "reused_current_artifact" if path.exists() else "missing_current_artifact",
                "monitor_profile_does_not_rebuild_research_evidence",
                path,
            )

    from quant_platform.cli import (
        paper_execution_preflight_report,
        paper_venue_preflight_report,
        priority_gap_test_report,
        priority_readiness_report,
        strategy_acceptance_checklist_report,
    )

    strategy_acceptance_checklist_report(
        reports / "strategy_acceptance_checklist.csv",
        root=root,
    )
    record("strategy_acceptance", "rebuilt", "acceptance_blockers_recomputed", reports / "strategy_acceptance_checklist.csv")
    readiness = priority_readiness_report(
        reports / "priority_readiness.csv",
        root=root,
    )
    record("priority_readiness", "rebuilt", "priority_gates_recomputed", reports / "priority_readiness.csv")
    paper_venue_preflight_report(
        output_path=reports / "paper_venue_preflight.csv",
        root=root,
    )
    record("paper_venue_preflight", "rebuilt", "submission_gate_recomputed", reports / "paper_venue_preflight.csv")
    paper_execution_preflight_report(
        reports / "paper_execution_preflight.csv",
        root=root,
    )
    record("paper_execution_preflight", "rebuilt", "paper_gate_recomputed", reports / "paper_execution_preflight.csv")
    priority_gap_test_report(
        readiness,
        reports / "priority_gap_test.csv",
        root=root,
    )
    record("priority_gap_test", "rebuilt", "open_gaps_recomputed", reports / "priority_gap_test.csv")
    frame = pd.DataFrame(rows, columns=DASHBOARD_REFRESH_COLUMNS)
    _write_csv(frame, active / "dashboard_refresh_status.csv")
    return frame


def _quality_eligible_pairs(root: Path = ROOT) -> set[str]:
    quality = _read_csv(root / "reports" / "pair_detail_quality_report.csv")
    if quality.empty or "pair" not in quality.columns:
        return set()

    research_execution = quality.get("research_execution_usable", pd.Series(dtype=bool)).fillna(False).astype(bool)
    execution = quality.get("execution_usable", pd.Series(dtype=bool)).fillna(False).astype(bool)
    eligible = quality[research_execution | execution].copy()
    return {
        str(pair).replace("_", "-").replace("/", "-").upper().strip()
        for pair in eligible.get("pair", pd.Series(dtype=str)).dropna().astype(str)
    }


def apify_cost_audit_rows(root: Path = ROOT) -> pd.DataFrame:
    manifest = _read_csv(root / "reports" / "active" / "apify_source_capture_manifest.csv")
    columns = [
        "source_id",
        "run_status",
        "sample_rows",
        "usage_usd",
        "cost_rank",
        "usage_classification",
        "pipeline_value",
        "recommendation",
        "rationale",
        "output_path",
        "run_id",
    ]
    if manifest.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    paid = manifest.copy()
    paid["usage_usd"] = pd.to_numeric(paid.get("usage_amount", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    paid = paid.sort_values(["usage_usd", "sample_rows"], ascending=[False, False]).reset_index(drop=True)
    rank = 0
    for _, row in paid.iterrows():
        source_id = str(row.get("source_id", "")).strip()
        usage_usd = float(row.get("usage_usd", 0.0) or 0.0)
        if usage_usd > 0:
            rank += 1
        recommendation = "pause"
        pipeline_value = "unproven"
        rationale = "No direct downstream usage evidence in active dashboards."
        if "coinglass" in source_id:
            recommendation = "keep"
            pipeline_value = "direct_context_usage"
            rationale = "Feeds market venue context and multi-exchange liquidity evidence."
        elif "funding-pulse" in source_id:
            recommendation = "keep_if_integrated"
            pipeline_value = "partial_integration"
            rationale = "Supports funding assumptions, but current pair-quality reports still show missing funding fields."
        elif "coinmarketcap" in source_id or "cryptocurrency-market-data-scraper" in source_id:
            recommendation = "pause"
            pipeline_value = "stored_only"
            rationale = "Snapshot exists, but it is not clearly driving pair selection or paper-readiness outputs."
        elif usage_usd <= 0:
            recommendation = "ignore"
            pipeline_value = "not_billed"
            rationale = "No recorded billed usage in the local run ledger."

        rows.append(
            {
                "source_id": source_id,
                "run_status": str(row.get("run_status", "")),
                "sample_rows": int(row.get("sample_rows", 0) or 0),
                "usage_usd": usage_usd,
                "cost_rank": rank if usage_usd > 0 else "",
                "usage_classification": "billed" if usage_usd > 0 else "not_billed",
                "pipeline_value": pipeline_value,
                "recommendation": recommendation,
                "rationale": rationale,
                "output_path": str(row.get("output_path", "")),
                "run_id": str(row.get("run_id", "")),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def paper_candidate_shortlist_rows(
    root: Path = ROOT,
    max_pairs: int = 5,
    require_execution_compatible: bool = True,
) -> pd.DataFrame:
    pair_universe = _read_csv(root / "data" / "processed" / "pair_universe.csv")
    rl_summary = _read_csv(root / "reports" / "rl" / "rl_learning_cycle_summary.csv")
    promotion = _read_csv(root / "reports" / "brain" / "promotion_ladder.csv")
    native_focus = _read_csv(root / "reports" / "brain" / "native_focus_repair_report.csv")
    columns = [
        "shortlist_rank",
        "pair",
        "candidate_id",
        "setup_identity",
        "setup_role",
        "best_execution_venue",
        "combined_score",
        "acceptance_score",
        "funding_drag_bps",
        "rl_policy_name",
        "rl_policy_winner",
        "rl_pair_focus_count",
        "rl_stop_loss_pct",
        "rl_take_profit_pct",
        "rl_session_loss_cap_pct",
        "available_timeframes",
        "decision_bucket",
        "decision_reason",
        "evidence_path",
        "shortlist_reason",
        "pair_model_support_status",
        "pair_model_taken_trades",
        "pair_model_profit_factor",
        "pair_model_mean_return",
    ]
    wizard_validated = _wizard_daily_validated_shortlist_rows(root=root)
    if pair_universe.empty:
        return wizard_validated.reindex(columns=columns, fill_value="").head(max_pairs).copy() if not wizard_validated.empty else pd.DataFrame(columns=columns)

    universe = pair_universe.copy()
    universe["pair_normalized"] = universe.get("pair", pd.Series(dtype=str)).astype(str).map(
        lambda value: value.replace("_", "-").replace("/", "-").upper().strip()
    )
    universe["combined_score"] = pd.to_numeric(universe.get("combined_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    universe["acceptance_score"] = pd.to_numeric(universe.get("acceptance_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    universe["funding_drag_bps"] = pd.to_numeric(universe.get("funding_drag_bps", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    universe = _attach_rl_shortlist_context(universe, rl_summary)
    universe = _attach_native_shortlist_context(universe, native_focus, root=root)
    unsupported_model_support_statuses = {
        "no_model_support",
        "pair_missing_from_model_predictions",
        "model_predictions_missing",
    }
    decision_bucket = universe.get("decision_bucket", pd.Series("", index=universe.index, dtype=str)).astype(str)
    shortlist = universe[decision_bucket == "PROMOTE"].copy()
    if "execution_venue_ready" in universe.columns:
        route_ready = universe["execution_venue_ready"].map(
            lambda value: True if not _text_value(value) else _boolish(value)
        )
    else:
        route_ready = pd.Series(True, index=universe.index, dtype=bool)
    route_valid_strong_support = universe[
        universe.get("best_execution_venue", pd.Series("", index=universe.index, dtype=str)).astype(str).str.strip().ne("")
        & route_ready
        & universe.get("pair_model_support_status", pd.Series("", index=universe.index, dtype=str)).astype(str).eq("strong_model_support")
    ].copy()
    if not route_valid_strong_support.empty:
        shortlist = route_valid_strong_support
    quality_pairs = _quality_eligible_pairs(root)
    quality_shortlist = shortlist[shortlist["pair_normalized"].isin(quality_pairs)].copy() if quality_pairs else pd.DataFrame()
    if not quality_shortlist.empty:
        shortlist = quality_shortlist
    elif shortlist.empty:
        shortlist = universe.sort_values(["combined_score", "acceptance_score"], ascending=[False, False]).head(max_pairs).copy()

    prioritized_by_model_support = False
    strong_model_supported = shortlist[
        shortlist.get("pair_model_support_status", pd.Series(dtype=object)).astype(str) == "strong_model_support"
    ].copy()
    if not strong_model_supported.empty:
        shortlist = strong_model_supported
        prioritized_by_model_support = True

    rl_supported_shortlist = shortlist[
        (shortlist.get("rl_pair_focus_count", pd.Series(dtype=int)).fillna(0).astype(int) > 0)
        & ~shortlist.get("pair_model_support_status", pd.Series(dtype=object)).astype(str).isin(unsupported_model_support_statuses)
    ].copy()
    if not prioritized_by_model_support:
        if not rl_supported_shortlist.empty:
            shortlist = rl_supported_shortlist
        else:
            paper_route_universe = universe.copy()
            if "execution_venue_ready" in paper_route_universe.columns:
                route_ready_candidates = paper_route_universe[
                    paper_route_universe["execution_venue_ready"].map(
                        lambda value: True if not _text_value(value) else _boolish(value)
                    )
                ].copy()
                if not route_ready_candidates.empty:
                    paper_route_universe = route_ready_candidates
            route_decision_bucket = paper_route_universe.get(
                "decision_bucket",
                pd.Series("", index=paper_route_universe.index, dtype=str),
            ).astype(str)
            non_rejected = paper_route_universe[route_decision_bucket != "REJECT"].copy()
            if not non_rejected.empty:
                paper_route_universe = non_rejected
            rl_supported_universe = paper_route_universe[
                paper_route_universe.get("rl_pair_focus_count", pd.Series(dtype=int)).fillna(0).astype(int) > 0
            ].copy()
            if not rl_supported_universe.empty:
                rl_supported_universe = rl_supported_universe[
                    ~rl_supported_universe.get("pair_model_support_status", pd.Series(dtype=object)).astype(str).isin(
                        unsupported_model_support_statuses
                    )
                ].copy()
            if quality_pairs:
                quality_rl_supported = rl_supported_universe[rl_supported_universe["pair_normalized"].isin(quality_pairs)].copy()
                if not quality_rl_supported.empty:
                    rl_supported_universe = quality_rl_supported
            if not rl_supported_universe.empty:
                shortlist = rl_supported_universe

    if not wizard_validated.empty:
        wizard_validated = wizard_validated.copy()
        wizard_validated["pair_normalized"] = wizard_validated.get("pair", pd.Series(dtype=str)).astype(str).map(
            lambda value: value.replace("_", "-").replace("/", "-").upper().strip()
        )
        shortlist_pairs = set(shortlist.get("pair", pd.Series(dtype=object)).astype(str)) if not shortlist.empty else set()
        wizard_additions = wizard_validated[~wizard_validated.get("pair", pd.Series(dtype=object)).astype(str).isin(shortlist_pairs)].copy()
        shortlist = pd.concat([wizard_additions, shortlist], ignore_index=True, sort=False) if not wizard_additions.empty else shortlist
        shortlist = _attach_rl_shortlist_context(shortlist, rl_summary)
        shortlist = _attach_native_shortlist_context(
            shortlist, native_focus, root=root
        )

    shortlisted_visible = _filter_shortlist_to_visible_dydx_markets(shortlist)
    if not shortlisted_visible.empty:
        shortlist = shortlisted_visible

    shortlist = _attach_execution_compatibility_context(shortlist, root)
    if require_execution_compatible:
        shortlist = shortlist[
            shortlist.get("execution_compatible", pd.Series(dtype=bool)).fillna(False).astype(bool)
        ].copy()

    shortlist["_wizard_priority"] = shortlist.get("shortlist_reason", pd.Series(dtype=object)).astype(str).str.startswith("wizard_validated_daily_")
    shortlist["_shortlist_rl_focus_rank"] = shortlist.apply(
        lambda row: int(pd.to_numeric(pd.Series([row.get("rl_pair_focus_count", 0)]), errors="coerce").fillna(0).iloc[0])
        if str(row.get("pair_model_support_status", "") or "") not in unsupported_model_support_statuses
        else 0,
        axis=1,
    )
    shortlist = shortlist.sort_values(
        ["_wizard_priority", "pair_model_support_rank", "_shortlist_rl_focus_rank", "combined_score", "acceptance_score"],
        ascending=[False, True, False, False, False],
    ).head(max_pairs).copy()
    shortlist = _attach_setup_shortlist_context(shortlist, promotion)
    shortlist = shortlist.reset_index(drop=True)
    shortlist.insert(0, "shortlist_rank", shortlist.index + 1)
    shortlist["shortlist_reason"] = shortlist.apply(_shortlist_reason, axis=1)
    output_columns = [
        "shortlist_rank",
        "pair",
        "candidate_id",
        "setup_identity",
        "setup_role",
        "best_execution_venue",
        "combined_score",
        "acceptance_score",
        "funding_drag_bps",
        "rl_policy_name",
        "rl_policy_winner",
        "rl_pair_focus_count",
        "rl_stop_loss_pct",
        "rl_take_profit_pct",
        "rl_session_loss_cap_pct",
        "available_timeframes",
        "decision_bucket",
        "decision_reason",
        "evidence_path",
        "shortlist_reason",
        "execution_compatible",
        "execution_blocker",
        "confirmed_execution_markets",
        "missing_execution_markets",
        "wizard_timeframe_capture_complete",
        "wizard_missing_timeframes",
        "pair_model_support_status",
        "pair_model_taken_trades",
        "pair_model_profit_factor",
        "pair_model_mean_return",
    ]
    return shortlist.reindex(columns=output_columns, fill_value="").copy()


def _attach_execution_compatibility_context(shortlist: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    enriched = shortlist.copy()
    if enriched.empty or "pair" not in enriched.columns:
        for column in (
            "execution_compatible",
            "execution_blocker",
            "confirmed_execution_markets",
            "missing_execution_markets",
            "wizard_timeframe_capture_complete",
            "wizard_missing_timeframes",
        ):
            enriched[column] = pd.Series(dtype=object)
        return enriched

    compatibility = _read_csv(root / "reports" / "active" / "dydx_execution_market_compatibility.csv")
    compatibility_lookup: dict[str, bool] = {}
    if not compatibility.empty and "market" in compatibility.columns:
        working = compatibility.copy()
        working["market"] = working["market"].astype(str).str.upper().str.strip()
        for _, row in working.iterrows():
            compatibility_lookup[str(row.get("market", "")).upper().strip()] = _boolish(
                row.get("compatible_for_paper_submit", False)
            )

    hyperliquid = _read_csv(root / "reports" / "active" / "hyperliquid_execution_market_compatibility.csv")
    hyperliquid_lookup: dict[str, dict[str, object]] = {}
    if not hyperliquid.empty and "pair" in hyperliquid.columns:
        for _, row in hyperliquid.iterrows():
            key = _normalize_pair_symbol(row.get("pair", "")).replace("/", "-")
            if key:
                hyperliquid_lookup[key] = {
                    "compatible": _boolish(row.get("mirrorable_for_paper", False)),
                    "blocker": str(row.get("mirror_blocker", "") or "").strip(),
                    "markets": ";".join(
                        value
                        for value in [
                            str(row.get("testnet_perp_x", "") or "").strip(),
                            str(row.get("testnet_perp_y", "") or "").strip(),
                        ]
                        if value
                    ),
                }

    detail = _read_csv(root / "reports" / "active" / "wizard_research_pair_detail_capture.csv")
    timeframe_truth: dict[str, set[str]] = {}
    if not detail.empty and "pair" in detail.columns:
        working = detail.copy()
        working["pair"] = working["pair"].astype(str).map(_normalize_pair_symbol)
        working["timeframe"] = working.get("timeframe", pd.Series(dtype=object)).astype(str).str.strip()
        working["capture_status"] = working.get("capture_status", pd.Series(dtype=object)).astype(str).str.strip()
        captured = working[
            working["capture_status"].eq("captured")
            & working["timeframe"].isin(["Daily", "4 Hour", "1 Hour", "5 Min"])
        ].copy()
        for pair, frame in captured.groupby("pair", sort=False):
            timeframe_truth[str(pair)] = set(frame.get("timeframe", pd.Series(dtype=object)).astype(str).tolist())

    required_timeframes = ("Daily", "4 Hour", "1 Hour", "5 Min")
    execution_compatible: list[bool] = []
    execution_blockers: list[str] = []
    confirmed_markets: list[str] = []
    missing_markets: list[str] = []
    timeframe_complete: list[bool] = []
    timeframe_missing_labels: list[str] = []

    for _, row in enriched.iterrows():
        pair = _normalize_pair_symbol(row.get("pair", ""))
        markets = _active_pipeline_pair_to_markets(pair)
        venue = str(row.get("best_execution_venue", "") or "").strip().lower()
        if venue == "hyperliquid":
            hyperliquid_row = hyperliquid_lookup.get(pair.replace("/", "-"), {})
            compatible = bool(hyperliquid_row.get("compatible", False))
            confirmed = str(hyperliquid_row.get("markets", "")).split(";") if compatible else []
            missing = [] if compatible else markets
            blocker = str(hyperliquid_row.get("blocker", "") or "") or "hyperliquid_pair_execution_unconfirmed"
        elif venue and venue != "dydx":
            compatible = _boolish(row.get("execution_venue_ready", False))
            confirmed = markets if compatible else []
            missing = [] if compatible else markets
            blocker = "" if compatible else "venue_pair_execution_unconfirmed"
        else:
            if compatibility_lookup:
                confirmed = [market for market in markets if compatibility_lookup.get(market, False)]
                missing = [market for market in markets if not compatibility_lookup.get(market, False)]
                compatible = bool(markets) and not missing
                blocker = "" if compatible else ("unsupported_execution_legs" if markets else "pair_markets_unresolved")
            else:
                # The actual dYdX submit preflight remains authoritative; this preserves
                # legacy shortlist diagnostics before a compatibility artifact is built.
                confirmed = markets
                missing = []
                compatible = bool(markets)
                blocker = "" if compatible else "pair_markets_unresolved"
        execution_compatible.append(compatible)
        execution_blockers.append("" if compatible else blocker)
        confirmed_markets.append(";".join(confirmed))
        missing_markets.append(";".join(missing))

        captured_timeframes = timeframe_truth.get(pair, set())
        missing_timeframes = [label for label in required_timeframes if label not in captured_timeframes]
        timeframe_complete.append(not missing_timeframes if captured_timeframes else False)
        timeframe_missing_labels.append(";".join(missing_timeframes))

    enriched["execution_compatible"] = execution_compatible
    enriched["execution_blocker"] = execution_blockers
    enriched["confirmed_execution_markets"] = confirmed_markets
    enriched["missing_execution_markets"] = missing_markets
    enriched["wizard_timeframe_capture_complete"] = timeframe_complete
    enriched["wizard_missing_timeframes"] = timeframe_missing_labels
    return enriched


def _filter_shortlist_to_visible_dydx_markets(shortlist: pd.DataFrame) -> pd.DataFrame:
    if shortlist.empty or "pair" not in shortlist.columns or "best_execution_venue" not in shortlist.columns:
        return shortlist

    config = DydxNetworkConfig.paper_testnet_from_env()
    adapter = build_dydx_indexer_adapter(config)
    if adapter is None:
        return shortlist

    market_visibility_cache: dict[str, bool | None] = {}

    def _market_visible(market: str) -> bool | None:
        key = str(market or "").upper().strip()
        if not key:
            return False
        if key in market_visibility_cache:
            return market_visibility_cache[key]
        try:
            payload = adapter.market_data(key)
            markets = ((payload or {}).get("payload") or {}).get("markets") or {}
            row = markets.get(key) if isinstance(markets, dict) else None
            if not isinstance(row, dict):
                # Missing payload for this leg is treated as transient/unknown.
                # We keep the candidate unless an explicit inactive status is
                # observed.
                visible = None
            else:
                status = str(row.get("status", "")).upper().strip()
                if status == "ACTIVE":
                    visible = True
                elif status:
                    visible = False
                else:
                    visible = None
        except Exception:
            # Adapter failures should not silently cull candidates. Keep the pair
            # unless we can explicitly prove a market is inactive.
            visible = None
        market_visibility_cache[key] = visible
        return visible

    keep_mask: list[bool] = []
    for _, row in shortlist.iterrows():
        venue = str(row.get("best_execution_venue", "") or "").strip().lower()
        if venue != "dydx":
            keep_mask.append(True)
            continue
        pair = str(row.get("pair", "") or "")
        market_states = [_market_visible(market) for market in _active_pipeline_pair_to_markets(pair)]
        keep_mask.append(not any(state is False for state in market_states))

    filtered = shortlist.loc[pd.Series(keep_mask, index=shortlist.index)].copy()
    return filtered


def _active_pipeline_pair_to_markets(pair: str) -> list[str]:
    return pair_markets_from_pair(pair)


def _normalize_pair_symbol(value: object) -> str:
    text = str(value or "").strip().upper().replace("_", "-").replace("/", "-")
    markets = pair_markets_from_pair(text)
    if len(markets) == 2:
        return f"{markets[0]}/{markets[1]}"
    return text


def _boolish(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text in {"true", "1", "yes", "y"}


def focused_paper_validation_rows(root: Path = ROOT) -> pd.DataFrame:
    shortlist = paper_candidate_shortlist_rows(root)
    preflight = _read_csv(root / "reports" / "paper_execution_preflight.csv")
    acceptance = _read_csv(root / "reports" / "ml" / "model_gated_acceptance.csv")
    pair_support = _read_csv(root / "reports" / "ml" / "model_gate_pair_support_report.csv")
    global_paper_ready = bool(not preflight.empty and preflight.get("ready", pd.Series(dtype=bool)).fillna(False).astype(bool).all())
    model_accepted = bool(not acceptance.empty and acceptance.get("accepted", pd.Series([False])).astype(bool).iloc[0])
    global_model_blocker = ""
    if not acceptance.empty:
        acceptance_row = acceptance.iloc[0]
        global_model_blocker = (
            str(acceptance_row.get("failing_checks", "") or "")
            or str(acceptance_row.get("blocker", "") or "")
            or str(acceptance_row.get("acceptance_reason", "") or "")
        )

    columns = [
        "shortlist_rank",
        "pair",
        "candidate_id",
        "setup_identity",
        "setup_role",
        "best_execution_venue",
        "paper_gate_ready",
        "model_gate_accepted",
        "rl_policy_winner",
        "rl_pair_focus_count",
        "rl_stop_loss_pct",
        "rl_take_profit_pct",
        "rl_session_loss_cap_pct",
        "pair_model_support_status",
        "pair_model_taken_trades",
        "pair_model_profit_factor",
        "pair_model_mean_return",
        "global_model_gate_blocker",
        "model_gate_repair_action",
        "validation_status",
        "next_action",
        "evidence_path",
    ]
    if shortlist.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for _, row in shortlist.iterrows():
        pair_model_support = (
            str(row.get("pair_model_support_status", "") or "").strip()
            or "model_predictions_missing"
        )
        pair_support_row = _match_row(pair_support, str(row.get("pair", "")), "pair")
        model_gate_repair_action = (
            str(pair_support_row.get("recommended_repair_action", "") or "")
            if pair_support_row is not None
            else ""
        )
        if pair_model_support != "strong_model_support":
            validation_status = "hold_for_pair_specific_model_support"
            next_action = "improve pair-specific model support before focused paper validation"
        elif global_paper_ready and model_accepted:
            validation_status = "ready_for_focused_paper_validation"
            next_action = "run focused paper-plan validation on shortlist pair with learned RL risk controls"
        else:
            validation_status = "hold_for_model_or_gate"
            next_action = model_gate_repair_action or "improve model acceptance and refresh dashboard before paper validation"
        if float(pd.to_numeric(pd.Series([row.get("rl_pair_focus_count", 0)]), errors="coerce").fillna(0.0).iloc[0]) <= 0:
            next_action = f"{next_action}; confirm RL pair coverage before paper handoff"
        rows.append(
            {
                "shortlist_rank": int(row.get("shortlist_rank", 0) or 0),
                "pair": str(row.get("pair", "")),
                "candidate_id": str(row.get("candidate_id", "")),
                "setup_identity": str(row.get("setup_identity", "")),
                "setup_role": str(row.get("setup_role", "")),
                "best_execution_venue": str(row.get("best_execution_venue", "")),
                "paper_gate_ready": global_paper_ready,
                "model_gate_accepted": model_accepted,
                "rl_policy_winner": str(row.get("rl_policy_name", "")),
                "rl_pair_focus_count": int(pd.to_numeric(pd.Series([row.get("rl_pair_focus_count", 0)]), errors="coerce").fillna(0).iloc[0]),
                "rl_stop_loss_pct": float(pd.to_numeric(pd.Series([row.get("rl_stop_loss_pct", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                "rl_take_profit_pct": float(pd.to_numeric(pd.Series([row.get("rl_take_profit_pct", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                "rl_session_loss_cap_pct": float(pd.to_numeric(pd.Series([row.get("rl_session_loss_cap_pct", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                "pair_model_support_status": pair_model_support,
                "pair_model_taken_trades": int(pd.to_numeric(pd.Series([row.get("pair_model_taken_trades", 0)]), errors="coerce").fillna(0).iloc[0]),
                "pair_model_profit_factor": float(pd.to_numeric(pd.Series([row.get("pair_model_profit_factor", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                "pair_model_mean_return": float(pd.to_numeric(pd.Series([row.get("pair_model_mean_return", 0.0)]), errors="coerce").fillna(0.0).iloc[0]),
                "global_model_gate_blocker": global_model_blocker,
                "model_gate_repair_action": model_gate_repair_action,
                "validation_status": validation_status,
                "next_action": next_action,
                "evidence_path": ";".join(
                    part
                    for part in [
                        str(row.get("evidence_path", "")),
                        "reports/ml/model_gate_pair_support_report.csv" if pair_support_row is not None else "",
                    ]
                    if part
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _attach_native_shortlist_context(
    shortlist: pd.DataFrame,
    native_focus: pd.DataFrame,
    *,
    root: Path = ROOT,
) -> pd.DataFrame:
    enriched = shortlist.copy()
    enriched["pair_model_support_status"] = "model_predictions_missing"
    enriched["pair_model_taken_trades"] = 0
    enriched["pair_model_profit_factor"] = 0.0
    enriched["pair_model_mean_return"] = 0.0
    enriched["pair_model_support_rank"] = 3
    focus_sources: list[pd.DataFrame] = []
    if not native_focus.empty and "pair" in native_focus.columns:
        focus_sources.append(native_focus.copy())
    model_gate_pairs = _read_csv(
        root / "reports" / "ml" / "model_gate_pair_support_report.csv"
    )
    if not model_gate_pairs.empty and "pair" in model_gate_pairs.columns:
        focus_sources.append(
            model_gate_pairs.rename(
                columns={
                    "support_status": "pair_model_support_status",
                    "taken_trades": "pair_model_taken_trades",
                    "profit_factor": "pair_model_profit_factor",
                    "mean_return": "pair_model_mean_return",
                }
            ).copy()
        )
    if not focus_sources:
        return enriched

    focus = pd.concat(focus_sources, ignore_index=True, sort=False)
    focus["pair"] = focus.get("pair", pd.Series(dtype=object)).astype(str)
    focus = focus.drop_duplicates(subset=["pair"], keep="first")
    focus_index = focus.set_index("pair", drop=False)
    rank_map = {
        "strong_model_support": 0,
        "weak_model_support": 1,
        "no_model_support": 2,
        "pair_missing_from_model_predictions": 3,
        "model_predictions_missing": 3,
    }

    def _focus_row(pair: object) -> pd.Series | None:
        key = str(pair or "")
        if key in focus_index.index:
            row = focus_index.loc[key]
            if isinstance(row, pd.DataFrame):
                return row.iloc[0]
            return row
        return None

    enriched["pair_model_support_status"] = enriched["pair"].map(
        lambda pair: str(
            _focus_row(pair).get(
                "pair_model_support_status", "model_predictions_missing"
            )
            or "model_predictions_missing"
        )
        if _focus_row(pair) is not None
        else "model_predictions_missing"
    )
    enriched["pair_model_taken_trades"] = enriched["pair"].map(
        lambda pair: int(pd.to_numeric(pd.Series([_focus_row(pair).get("pair_model_taken_trades", 0) if _focus_row(pair) is not None else 0]), errors="coerce").fillna(0).iloc[0])
    )
    enriched["pair_model_profit_factor"] = enriched["pair"].map(
        lambda pair: float(pd.to_numeric(pd.Series([_focus_row(pair).get("pair_model_profit_factor", 0.0) if _focus_row(pair) is not None else 0.0]), errors="coerce").fillna(0.0).iloc[0])
    )
    enriched["pair_model_mean_return"] = enriched["pair"].map(
        lambda pair: float(pd.to_numeric(pd.Series([_focus_row(pair).get("pair_model_mean_return", 0.0) if _focus_row(pair) is not None else 0.0]), errors="coerce").fillna(0.0).iloc[0])
    )
    enriched["pair_model_support_rank"] = enriched["pair_model_support_status"].map(lambda value: rank_map.get(str(value), 3)).fillna(3).astype(int)
    return enriched


def _attach_setup_shortlist_context(shortlist: pd.DataFrame, promotion: pd.DataFrame) -> pd.DataFrame:
    enriched = shortlist.copy()
    enriched["candidate_id"] = ""
    enriched["setup_identity"] = ""
    enriched["setup_role"] = ""
    if promotion.empty or "pair" not in promotion.columns:
        return enriched
    for idx, row in enriched.iterrows():
        pair = str(row.get("pair", ""))
        pair_rows = promotion[promotion["pair"].astype(str) == pair].copy()
        if pair_rows.empty:
            continue
        pair_rows["_credible"] = pair_rows.get("paper_credible", pd.Series(dtype=object)).astype(bool)
        pair_rows["_primary"] = pair_rows.get("setup_role", pd.Series(dtype=object)).astype(str).eq("primary")
        pair_rows = pair_rows.sort_values(["_credible", "_primary"], ascending=[False, False])
        best = pair_rows.iloc[0]
        enriched.at[idx, "candidate_id"] = str(best.get("candidate_id", "") or "")
        enriched.at[idx, "setup_identity"] = str(best.get("setup_identity", "") or "")
        enriched.at[idx, "setup_role"] = str(best.get("setup_role", "") or "")
    return enriched


def _wizard_daily_validated_shortlist_rows(root: Path = ROOT) -> pd.DataFrame:
    readiness = _read_csv(root / "reports" / "active" / "wizard_ou_spread_daily_paper_readiness_latest.csv")
    verification = _read_csv(root / "reports" / "active" / "wizard_local_verification_batch.csv")
    columns = [
        "pair",
        "candidate_id",
        "setup_identity",
        "setup_role",
        "best_execution_venue",
        "combined_score",
        "acceptance_score",
        "funding_drag_bps",
        "rl_policy_name",
        "rl_policy_winner",
        "rl_pair_focus_count",
        "rl_stop_loss_pct",
        "rl_take_profit_pct",
        "rl_session_loss_cap_pct",
        "available_timeframes",
        "decision_bucket",
        "decision_reason",
        "evidence_path",
        "shortlist_reason",
        "pair_model_support_status",
        "pair_model_taken_trades",
        "pair_model_profit_factor",
        "pair_model_mean_return",
        "pair_model_support_rank",
    ]
    if readiness.empty or verification.empty:
        return pd.DataFrame(columns=columns)

    verification = verification.copy()
    verification["pair"] = verification.get("pair", pd.Series(dtype=object)).astype(str)
    verification_index = verification.set_index("pair", drop=False)
    rows: list[dict[str, object]] = []
    for _, row in readiness.iterrows():
        if str(row.get("local_acceptance", "")).upper() != "ACCEPT":
            continue
        if str(row.get("forward_walk_status", "")).lower() != "pass":
            continue
        pair = str(row.get("pair", ""))
        if not pair:
            continue
        verification_row = verification_index.loc[pair] if pair in verification_index.index else None
        if isinstance(verification_row, pd.DataFrame):
            verification_row = verification_row.iloc[0]
        setup_identity = str(verification_row.get("setup_identity", "") or "") if verification_row is not None else ""
        setup_role = str(verification_row.get("setup_role", "") or "primary") if verification_row is not None else "primary"
        local_pf = float(pd.to_numeric(pd.Series([row.get("local_profit_factor", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        local_sharpe = float(pd.to_numeric(pd.Series([row.get("local_sharpe", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        fw_pf = float(pd.to_numeric(pd.Series([row.get("forward_walk_profit_factor", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        fw_sharpe = float(pd.to_numeric(pd.Series([row.get("forward_walk_sharpe", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        combined_score = round(local_pf * 25.0 + local_sharpe * 10.0 + fw_pf * 20.0 + fw_sharpe * 5.0, 3)
        acceptance_score = round(local_pf * 40.0 + local_sharpe * 15.0, 3)
        rows.append(
            {
                "pair": pair,
                "candidate_id": str(row.get("candidate_id", "") or ""),
                "setup_identity": setup_identity,
                "setup_role": setup_role,
                "best_execution_venue": "dydx",
                "combined_score": combined_score,
                "acceptance_score": acceptance_score,
                "funding_drag_bps": 0.0,
                "rl_policy_name": "",
                "rl_policy_winner": False,
                "rl_pair_focus_count": 0,
                "rl_stop_loss_pct": 0.0,
                "rl_take_profit_pct": 0.0,
                "rl_session_loss_cap_pct": 0.0,
                "available_timeframes": "daily",
                "decision_bucket": "PROMOTE",
                "decision_reason": "wizard_daily_local_acceptance_and_forward_walk_pass",
                "evidence_path": str(root / "reports" / "active" / "wizard_ou_spread_daily_paper_readiness_latest.csv"),
                "shortlist_reason": "wizard_validated_daily_route_candidate",
                "pair_model_support_status": "",
                "pair_model_taken_trades": 0,
                "pair_model_profit_factor": 0.0,
                "pair_model_mean_return": 0.0,
                "pair_model_support_rank": 3,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _attach_rl_shortlist_context(shortlist: pd.DataFrame, rl_summary: pd.DataFrame) -> pd.DataFrame:
    enriched = shortlist.copy()
    enriched["rl_policy_name"] = ""
    enriched["rl_policy_winner"] = False
    enriched["rl_pair_focus_count"] = 0
    enriched["rl_stop_loss_pct"] = 0.0
    enriched["rl_take_profit_pct"] = 0.0
    enriched["rl_session_loss_cap_pct"] = 0.0
    if rl_summary.empty:
        return enriched

    summary = rl_summary.copy()
    winner_series = summary.get("winner", pd.Series(dtype=int, index=summary.index))
    winner_rows = summary.loc[winner_series.fillna(0).astype(int) == 1]
    if not winner_rows.empty:
        winner = winner_rows.iloc[0]
    elif "profit_factor" in summary.columns:
        winner = summary.sort_values("profit_factor", ascending=False).iloc[0]
    else:
        winner = summary.iloc[0]
    pair_focus = _parse_top_pairs_entered(str(winner.get("top_pairs_entered", "")))
    enriched["rl_policy_name"] = str(winner.get("policy_name", ""))
    enriched["rl_policy_winner"] = True
    enriched["rl_stop_loss_pct"] = float(pd.to_numeric(pd.Series([winner.get("stop_loss_pct", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    enriched["rl_take_profit_pct"] = float(pd.to_numeric(pd.Series([winner.get("take_profit_pct", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    enriched["rl_session_loss_cap_pct"] = float(pd.to_numeric(pd.Series([winner.get("session_loss_cap_pct", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    enriched["rl_pair_focus_count"] = enriched["pair"].map(lambda value: pair_focus.get(str(value), 0)).fillna(0).astype(int)
    enriched = enriched.sort_values(
        ["rl_pair_focus_count", "combined_score", "acceptance_score"],
        ascending=[False, False, False],
    )
    return enriched


def _parse_top_pairs_entered(value: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    text = str(value or "")
    for part in text.split(";"):
        token = part.strip()
        if not token or ":" not in token:
            continue
        pair, count = token.rsplit(":", 1)
        try:
            mapping[pair.strip()] = int(float(count))
        except (TypeError, ValueError):
            continue
    return mapping


def _shortlist_reason(row: pd.Series) -> str:
    venue = str(row.get("best_execution_venue", "unknown_venue"))
    focus = int(pd.to_numeric(pd.Series([row.get("rl_pair_focus_count", 0)]), errors="coerce").fillna(0).iloc[0])
    pair_model_support = str(row.get("pair_model_support_status", "") or "")
    if str(row.get("shortlist_reason", "") or "").startswith("wizard_validated_daily_"):
        return "wizard_validated_daily_route_candidate"
    if pair_model_support == "strong_model_support" and focus > 0:
        return f"top_promoted_pair_for_{venue}_with_strong_pair_model_support_and_rl_focus_{focus}"
    if pair_model_support == "strong_model_support":
        return f"top_promoted_pair_for_{venue}_with_strong_pair_model_support"
    if focus > 0:
        return f"top_promoted_pair_for_{venue}_with_rl_focus_{focus}"
    return f"top_promoted_pair_for_{venue}"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_text(path: Path) -> str:
    if not path.exists():
        return "# Project Spine Audit\n\nNot generated yet.\n"
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return "# Project Spine Audit\n\nCould not read audit.\n"


def _quantization_readiness_rows(root: Path) -> pd.DataFrame:
    trade_gate = _read_json(root / "models" / "trade_gate" / "export_report.json")
    rl_acceptance = _read_csv(root / "reports" / "rl" / "rl_acceptance_report.csv")
    rl_ok = bool(not rl_acceptance.empty and rl_acceptance.get("accepted", pd.Series([False])).astype(bool).iloc[0])
    return pd.DataFrame(
        [
            {
                "model": "trade_gate",
                "ready": bool(trade_gate.get("accepted", False) or trade_gate.get("exported", False)),
                "blocker": trade_gate.get("blocker", ""),
                "evidence_path": "models/trade_gate/export_report.json",
            },
            {
                "model": "rl_policy",
                "ready": rl_ok,
                "blocker": "" if rl_ok else "rl_acceptance_not_passed",
                "evidence_path": "reports/rl/rl_acceptance_report.csv",
            },
        ]
    )


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configured_env_key_present(root: Path, key: str) -> bool:
    """Check process or local-file configuration without loading secret values."""

    if os.getenv(key, "").strip():
        return True
    env_path = root / ".env.local"
    if not env_path.is_file():
        return False
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        candidate, value = line.split("=", 1)
        if candidate.strip() == key and value.strip().strip("'\""):
            return True
    return False


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    atomic_write_csv(frame, path, index=False)
    return path


def _write_text(path: Path, text: str) -> Path:
    atomic_write_text(path, text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    return _write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _write_parquet_if_available(frame: pd.DataFrame, path: Path) -> str:
    try:
        atomic_write_parquet(frame, path, index=False)
        return "written"
    except Exception as exc:
        return f"not_written:{type(exc).__name__}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    drawdown = (peak - equity) / peak.replace(0, np.nan)
    return float(drawdown.max(skipna=True) or 0.0)


def _exists(path: Path) -> bool:
    return path.exists()


def _seven_stage_state_rows(root: Path) -> list[dict[str, object]]:
    path = root / "reports" / "active" / "seven_stage_goal_checkpoint.csv"
    frame = _read_csv(path)
    required = {"stage", "objective", "status", "evidence_progress", "blocker", "next_action"}
    if frame.empty or not required.issubset(frame.columns):
        return [
            _state_row(
                "seven_stage_checkpoint",
                False,
                "seven_stage_checkpoint_missing_or_invalid",
                path,
                "run the corrective program checkpoint builder",
            )
        ]

    rows: list[dict[str, object]] = []
    for _, source in frame.sort_values("stage", key=lambda values: pd.to_numeric(values, errors="coerce")).iterrows():
        stage = str(source.get("stage", "")).strip()
        status = str(source.get("status", "BLOCKED")).strip().upper() or "BLOCKED"
        ready = status == "PASS"
        blocker = str(source.get("blocker", "") or "").strip()
        evidence_progress = str(source.get("evidence_progress", "") or "").strip()
        rows.append(
            {
                "area": f"seven_stage_{stage}",
                "ready": ready,
                "status": status,
                "blocker": "" if ready else blocker or f"stage_{stage}_{status.lower()}",
                "detail": evidence_progress,
                "pair": "",
                "candidate_id": "",
                "setup_identity": "",
                "setup_role": "",
                "setup_status": status,
                "setup_blocker": "" if ready else blocker,
                "evidence_path": str(path),
                "next_action": str(source.get("next_action", "") or "").strip(),
            }
        )
    return rows


def _active_layer_state_row(root: Path) -> dict[str, object]:
    path = root / "reports" / "active" / "artifact_index.csv"
    frame = _read_csv(path)
    if frame.empty or not set(ARTIFACT_COLUMNS).issubset(frame.columns):
        return _state_row(
            "active_layer",
            False,
            "artifact_index_missing_or_invalid",
            path,
            "run build-artifact-index",
        )
    indexed_paths = frame["path"].fillna("").astype(str)
    polluted = indexed_paths.map(lambda value: any(_artifact_dir_is_excluded(part) for part in Path(value).parts))
    pollution_count = int(polluted.sum())
    unknown_count = int(frame["status"].fillna("").astype(str).eq("unknown").sum())
    ready = pollution_count == 0
    return {
        "area": "active_layer",
        "ready": ready,
        "status": "ready" if ready else "blocked",
        "blocker": "" if ready else f"artifact_index_contains_generated_dependency_paths:{pollution_count}",
        "detail": f"artifacts={len(frame)};unknown={unknown_count};excluded_path_violations={pollution_count}",
        "pair": "",
        "candidate_id": "",
        "setup_identity": "",
        "setup_role": "",
        "setup_status": "ready" if ready else "blocked",
        "setup_blocker": "" if ready else "artifact_index_contains_generated_dependency_paths",
        "evidence_path": str(path),
        "next_action": "review unknown classifications" if ready and unknown_count else "run build-artifact-index",
    }


def _model_authority_state_row(root: Path) -> dict[str, object]:
    authority_path = root / "reports" / "active" / "model_authority_status.json"
    model_path = root / "models" / "trade_gate" / "model.pkl"
    authority = _read_json(authority_path)
    if not authority:
        return _state_row(
            "trade_gate_model",
            False,
            "model_authority_status_missing",
            authority_path,
            "run corrective agent governance after training",
        )

    authority_status = str(authority.get("model_authority", "RESEARCH_ONLY")).strip().upper()
    blockers = authority.get("blockers", [])
    if not isinstance(blockers, list):
        blockers = [str(blockers)]
    blockers = [str(value).strip() for value in blockers if str(value).strip()]
    ready = bool(
        model_path.exists()
        and authority_status not in {"", "RESEARCH_ONLY", "BLOCKED", "NOT_AUTHORIZED"}
        and authority.get("out_of_sample_incremental_edge_accepted") is True
        and authority.get("score_buckets_monotonic") is True
        and authority.get("rl_out_of_sample_accepted") is True
    )
    blocker = "" if ready else ";".join(blockers) or f"model_authority_{authority_status.lower()}"
    return {
        "area": "trade_gate_model",
        "ready": ready,
        "status": authority_status or "BLOCKED",
        "blocker": blocker,
        "detail": (
            f"artifact_exists={model_path.exists()};authority={authority_status};"
            f"oos_edge={bool(authority.get('out_of_sample_incremental_edge_accepted', False))};"
            f"monotonic={bool(authority.get('score_buckets_monotonic', False))};"
            f"rl_oos={bool(authority.get('rl_out_of_sample_accepted', False))}"
        ),
        "pair": "",
        "candidate_id": "",
        "setup_identity": "",
        "setup_role": "",
        "setup_status": authority_status or "BLOCKED",
        "setup_blocker": blocker,
        "evidence_path": str(authority_path),
        "next_action": (
            "retain the model for research only and resolve every recorded blocker"
            if not ready
            else "retain accepted model lineage and monitor out-of-sample evidence"
        ),
    }


def _acceptance_ready(root: Path) -> bool:
    frame = _read_csv(root / "reports" / "strategy_acceptance_checklist.csv")
    return bool(not frame.empty and frame.get("ready", pd.Series(dtype=bool)).astype(bool).all())


def _acceptance_status(root: Path) -> str:
    frame = _read_csv(root / "reports" / "strategy_acceptance_checklist.csv")
    if frame.empty:
        return "strategy acceptance checklist missing"
    blockers = frame.loc[~frame.get("ready", pd.Series(True, index=frame.index)).astype(bool), "blocker"].dropna().astype(str).tolist()
    return "ready" if not blockers else "blocked:" + ";".join(blockers[:3])


def _hyperliquid_public_context_state_row(root: Path) -> dict[str, object]:
    path = root / "data" / "processed" / "hyperliquid_market_context.csv"
    frame = _read_csv(path)
    if frame.empty:
        return _state_row(
            "hyperliquid_public_market_context",
            False,
            "hyperliquid public market context missing",
            path,
            "run refresh-hyperliquid-market-context",
        )
    timestamps = pd.to_datetime(frame.get("source_timestamp", pd.Series(dtype=object)), utc=True, errors="coerce")
    fresh = timestamps.notna() & (pd.Timestamp.now(tz="UTC") - timestamps <= pd.Timedelta(hours=24))
    fresh_rows = int(fresh.sum())
    ready = fresh_rows > 0
    status = "research_context_ready" if ready else "stale"
    blocker = "" if ready else "hyperliquid_public_market_context_stale"
    return {
        "area": "hyperliquid_public_market_context",
        "ready": ready,
        "status": status,
        "blocker": blocker,
        "detail": f"markets={len(frame)};fresh_markets={fresh_rows};promotion_authority=false",
        "pair": "",
        "candidate_id": "",
        "setup_identity": "",
        "setup_role": "public_market_research",
        "setup_status": status,
        "setup_blocker": blocker,
        "evidence_path": path,
        "next_action": "build-hyperliquid-research-bundle" if ready else "refresh-hyperliquid-market-context",
    }


def _hyperliquid_research_bundle_state_row(root: Path) -> dict[str, object]:
    path = root / "reports" / "active" / "hyperliquid_research_bundle.csv"
    frame = _read_csv(path)
    if frame.empty:
        return _state_row(
            "hyperliquid_fresh_pair_history",
            False,
            "hyperliquid fresh pair-history bundle missing",
            path,
            "run build-hyperliquid-research-bundle --max-pairs 5",
        )
    ready_series = frame.get("history_ready", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    ready_rows = int(ready_series.sum())
    ready_pairs = int(frame.loc[ready_series, "pair"].nunique()) if ready_rows and "pair" in frame.columns else 0
    ready = ready_rows > 0
    blockers = sorted(
        {
            _text_value(value)
            for value in frame.loc[~ready_series, "blocker"].tolist()
            if _text_value(value)
        }
    ) if "blocker" in frame.columns else []
    blocker = "" if ready else ";".join(blockers) or "no_hyperliquid_pair_history_ready"
    return {
        "area": "hyperliquid_fresh_pair_history",
        "ready": ready,
        "status": "research_history_ready" if ready else "blocked",
        "blocker": blocker,
        "detail": f"ready_pair_timeframes={ready_rows};ready_pairs={ready_pairs};validation_authority=false",
        "pair": "",
        "candidate_id": "",
        "setup_identity": "",
        "setup_role": "two_leg_local_research",
        "setup_status": "research_history_ready" if ready else "blocked",
        "setup_blocker": blocker,
        "evidence_path": path,
        "next_action": "collect venue costs, slippage, funding, then rebuild venue route scorecard" if ready else "repair_hyperliquid_history_evidence",
    }


def _hyperliquid_wizard_hypothesis_state_row(root: Path) -> dict[str, object]:
    path = root / "reports" / "active" / "hyperliquid_wizard_hypothesis_queue.csv"
    frame = _read_csv(path)
    if frame.empty:
        return _state_row(
            "hyperliquid_wizard_exact_mode_intake",
            False,
            "Hyperliquid to Wizard exact-mode hypothesis queue missing",
            path,
            "run build-hyperliquid-wizard-hypothesis-queue",
        )
    eligible = int(frame.get("vendor_custom_series_eligible", pd.Series(False, index=frame.index)).fillna(False).astype(bool).sum())
    total = len(frame)
    blocker_rows = frame.loc[
        ~frame.get("vendor_custom_series_eligible", pd.Series(False, index=frame.index)).fillna(False).astype(bool),
        "blocker",
    ] if "blocker" in frame.columns else pd.Series(dtype=object)
    blockers = sorted({_text_value(value) for value in blocker_rows.tolist() if _text_value(value)})
    ready = eligible > 0
    blocker = "" if ready else (";".join(blockers[:3]) or "no_fresh_exact_mode_hypothesis")
    return {
        "area": "hyperliquid_wizard_exact_mode_intake",
        "ready": ready,
        "status": "vendor_mode_proof_ready" if ready else "blocked",
        "blocker": blocker,
        "detail": f"hypotheses={total};vendor_mode_proof_eligible={eligible};promotion_authority=false",
        "pair": "",
        "candidate_id": "",
        "setup_identity": "",
        "setup_role": "wizard_hypothesis_intake",
        "setup_status": "vendor_mode_proof_ready" if ready else "blocked",
        "setup_blocker": blocker,
        "evidence_path": path,
        "next_action": (
            "run bounded vendor custom-series proof, then local after-cost replay"
            if ready
            else "refresh the current Wizard capture with exact mode and full backtest settings"
        ),
    }


def _hyperliquid_wizard_mode_proof_state_row(root: Path) -> dict[str, object]:
    path = root / "reports" / "active" / "hyperliquid_wizard_vendor_mode_proofs.csv"
    frame = _read_csv(path)
    if frame.empty:
        return _state_row(
            "hyperliquid_wizard_vendor_mode_proof",
            False,
            "bounded Wizard vendor mode proof has not been preflighted",
            path,
            "run run-hyperliquid-wizard-mode-proofs; it is no-credit unless explicitly enabled",
        )
    status = frame.get("mode_proof_status", pd.Series("", index=frame.index)).astype(str)
    completed = int(status.eq("completed").sum())
    preflight = int(status.eq("preflight_ready").sum())
    blockers = sorted(
        {
            _text_value(value)
            for value in frame.get("blocker", pd.Series("", index=frame.index)).tolist()
            if _text_value(value)
        }
    )
    ready = completed > 0
    status_label = "vendor_mode_proof_complete" if ready else ("vendor_mode_proof_preflight_ready" if preflight else "blocked")
    blocker = "" if ready else (";".join(blockers[:3]) or "no_completed_vendor_mode_proof")
    return {
        "area": "hyperliquid_wizard_vendor_mode_proof",
        "ready": ready,
        "status": status_label,
        "blocker": blocker,
        "detail": f"completed={completed};preflight_ready={preflight};promotion_authority=false",
        "pair": "",
        "candidate_id": "",
        "setup_identity": "",
        "setup_role": "wizard_vendor_mode_proof",
        "setup_status": status_label,
        "setup_blocker": blocker,
        "evidence_path": path,
        "next_action": (
            "inspect the vendor proof, then run a separate local after-cost replay"
            if ready
            else "refresh a complete Wizard capture and run the bounded no-credit preflight"
        ),
    }


def _wizard_ou_v6_terminal_state_row(root: Path) -> dict[str, object]:
    path = root / "reports" / "active" / "wizard_ou_v6_terminal_closure.json"
    payload = _read_json(path)
    if not payload:
        return _state_row(
            "wizard_ou_v6_terminal_outcome",
            False,
            "OU-v6 terminal closure missing",
            path,
            "run close-wizard-ou-v6-terminal",
        )
    closed = (
        payload.get("status") == "CLOSED_TERMINAL_FAILURE"
        and payload.get("evidence_locked") is True
        and payload.get("general_ou_v6_activation") is False
        and payload.get("v7_registration_authorized") is False
    )
    evidence_blockers = payload.get("blockers")
    blocker = (
        "ou_v6_terminal_holdout_failed"
        if closed
        else ";".join(str(item) for item in evidence_blockers or [])
        or "ou_v6_terminal_closure_invalid"
    )
    return {
        "area": "wizard_ou_v6_terminal_outcome",
        "ready": False,
        "status": "terminal_failure_closed" if closed else "blocked_evidence_invalid",
        "blocker": blocker,
        "detail": (
            f"passed_cells={payload.get('passed_cells', 0)}/8;"
            f"evidence_locked={str(payload.get('evidence_locked') is True).lower()};"
            "activation=false;v7_registration=false"
        ),
        "pair": "",
        "candidate_id": "",
        "setup_identity": "",
        "setup_role": "wizard_ou_comparator_research",
        "setup_status": "terminal_failure_closed" if closed else "blocked_evidence_invalid",
        "setup_blocker": blocker,
        "evidence_path": path,
        "next_action": payload.get("next_action", "keep_ou_v6_blocked"),
    }


def _hyperliquid_slippage_calibration_state_row(root: Path) -> dict[str, object]:
    model_path = root / "reports" / "active" / "hyperliquid_pair_cost_model.csv"
    cadence_path = root / "reports" / "active" / "hyperliquid_evidence_cadence.csv"
    model = _read_csv(model_path)
    cadence = _read_csv(cadence_path)
    if model.empty:
        return _state_row(
            "hyperliquid_slippage_calibration",
            False,
            "hyperliquid pair cost model missing",
            model_path,
            "run refresh-hyperliquid-execution-cost-snapshot --max-pairs 5",
        )
    calibrated = int(model.get("slippage_model_ready", pd.Series(False, index=model.index)).fillna(False).astype(bool).sum())
    cost_ready = int(model.get("cost_model_ready", pd.Series(False, index=model.index)).fillna(False).astype(bool).sum())
    total = len(model)
    ready = calibrated == total and total > 0
    cadence_rows = cadence[cadence.get("status", pd.Series("", index=cadence.index)).astype(str) != "calibrated"] if not cadence.empty else pd.DataFrame()
    next_step = "run_costed_hyperliquid_local_replay" if ready else "run refresh-hyperliquid-execution-cost-snapshot when cadence is due"
    if not cadence_rows.empty and "next_step" in cadence_rows.columns:
        next_step = str(cadence_rows.iloc[0].get("next_step", "") or next_step)
    blockers = sorted(
        {
            _text_value(value)
            for value in model.loc[~model.get("slippage_model_ready", pd.Series(False, index=model.index)).fillna(False).astype(bool), "blocker"].tolist()
            if _text_value(value)
        }
    ) if "blocker" in model.columns else []
    blocker = "" if ready else ";".join(blockers) or "hyperliquid_l2_slippage_calibration_incomplete"
    return {
        "area": "hyperliquid_slippage_calibration",
        "ready": ready,
        "status": "calibrated" if ready else "collecting",
        "blocker": blocker,
        "detail": f"pairs={total};fee_profiles_ready={cost_ready};slippage_models_ready={calibrated};target_pairs={total}",
        "pair": "",
        "candidate_id": "",
        "setup_identity": "",
        "setup_role": "public_l2_depth_calibration",
        "setup_status": "calibrated" if ready else "collecting",
        "setup_blocker": blocker,
        "evidence_path": f"{model_path};{cadence_path}",
        "next_action": next_step,
    }


def _strategy_acceptance_state_row(root: Path) -> dict[str, object]:
    readiness = _read_csv(root / "reports" / "priority_readiness.csv")
    checklist = _read_csv(root / "reports" / "strategy_acceptance_checklist.csv")

    readiness_row = None
    if not readiness.empty and "gate" in readiness.columns:
        matches = readiness[readiness["gate"].astype(str) == "strategy_acceptance"]
        if not matches.empty:
            readiness_row = matches.iloc[0]

    ready = _acceptance_ready(root)
    detail = _acceptance_status(root)
    evidence_path = root / "reports" / "strategy_acceptance_checklist.csv"
    next_action = "review acceptance blockers before paper trading"

    checklist_blockers = []
    if not checklist.empty:
        checklist_blockers = (
            checklist.loc[~checklist.get("ready", pd.Series(True, index=checklist.index)).astype(bool), "blocker"]
            .dropna()
            .astype(str)
            .tolist()
        )

    if readiness_row is not None:
        ready = bool(readiness_row.get("ready", False))
        gate_status = str(readiness_row.get("status", "")).strip() or ("ready" if ready else "blocked")
        gate_evidence = str(readiness_row.get("evidence", "")).strip()
        detail_parts = [gate_status]
        if gate_evidence:
            detail_parts.append(gate_evidence)
        if checklist_blockers:
            detail_parts.append("checklist_blockers=" + ";".join(checklist_blockers[:3]))
        detail = ";".join(part for part in detail_parts if part)
        next_action = str(readiness_row.get("next_action", "")).strip() or next_action
        evidence_path = root / "reports" / "priority_readiness.csv"

    return _state_row(
        "strategy_acceptance",
        ready,
        detail,
        evidence_path,
        next_action,
    )


def _state_row(area: str, ready: bool, status: str, evidence_path: Path, next_action: str) -> dict[str, object]:
    return {
        "area": area,
        "ready": ready,
        "status": "ready" if ready else "blocked",
        "blocker": "" if ready else status,
        "detail": status,
        "pair": "",
        "candidate_id": "",
        "setup_identity": "",
        "setup_role": "",
        "setup_status": "",
        "setup_blocker": "",
        "evidence_path": str(evidence_path),
        "next_action": next_action,
    }


def _readiness_state_row(area: str, alias_path: Path, next_action: str) -> dict[str, object]:
    frame = _read_csv(alias_path)
    if frame.empty:
        return _state_row(area, False, f"{area}_alias_missing", alias_path, next_action)
    row = frame.iloc[0]
    ready = bool(row.get("ready", False))
    detail = str(row.get("summary", "")).strip() or str(row.get("status", "")).strip() or f"{area}_status_unknown"
    blocker = str(row.get("blocker", "")).strip() or detail
    return {
        "area": area,
        "ready": ready,
        "status": str(row.get("status", "blocked")).strip() or ("ready" if ready else "blocked"),
        "blocker": "" if ready else blocker,
        "detail": detail,
        "pair": str(row.get("pair", "") or ""),
        "candidate_id": str(row.get("candidate_id", "") or ""),
        "setup_identity": str(row.get("setup_identity", "") or ""),
        "setup_role": str(row.get("setup_role", "") or ""),
        "setup_status": str(row.get("setup_status", "") or ""),
        "setup_blocker": str(row.get("setup_blocker", "") or ""),
        "evidence_path": str(alias_path),
        "next_action": next_action,
    }


def _venue_route_state_row(root: Path) -> dict[str, object]:
    recommendations_path = root / "reports" / "active" / "venue_route_recommendations.csv"
    recommendations = _normalize_venue_route_recommendations(_read_csv(recommendations_path))
    if recommendations.empty:
        return {
            "area": "venue_route_scorecard",
            "ready": False,
            "status": "missing",
            "blocker": "venue_route_recommendations_missing",
            "detail": "pairs_reviewed=0;research_routes=0;execution_routes=0;paper_routes=0",
            "pair": "",
            "candidate_id": "",
            "setup_identity": "",
            "setup_role": "",
            "setup_status": "missing",
            "setup_blocker": "venue_route_recommendations_missing",
            "evidence_path": recommendations_path,
            "next_action": "run build-venue-route-scorecard",
        }
    research_count = int(
        recommendations.get("recommended_research_venue", pd.Series("", index=recommendations.index)).map(_text_value).ne("").sum()
    )
    execution_count = int(
        recommendations.get("recommended_execution_venue", pd.Series("", index=recommendations.index)).map(_text_value).ne("").sum()
    )
    paper_count = int(
        recommendations.get("recommended_paper_venue", pd.Series("", index=recommendations.index)).map(_text_value).ne("").sum()
    )
    if paper_count:
        status = "paper_route_ready"
    elif execution_count:
        status = "execution_route_ready"
    elif research_count:
        status = "research_only"
    else:
        status = "blocked"
    blocker = ""
    if not execution_count:
        blockers = [
            _text_value(value)
            for value in recommendations.get("blockers", pd.Series(dtype=object)).tolist()
            if _text_value(value)
        ]
        blocker = blockers[0] if blockers else "no_evidence_complete_execution_route"
    return {
        "area": "venue_route_scorecard",
        "ready": bool(execution_count),
        "status": status,
        "blocker": blocker,
        "detail": (
            f"pairs_reviewed={len(recommendations)};research_routes={research_count};"
            f"execution_routes={execution_count};paper_routes={paper_count}"
        ),
        "pair": "",
        "candidate_id": "",
        "setup_identity": "",
        "setup_role": "",
        "setup_status": status,
        "setup_blocker": blocker,
        "evidence_path": recommendations_path,
        "next_action": (
            "continue paper readiness only after runtime trade approval"
            if paper_count
            else (
                "complete matching history, cost, slippage, and funding checks for the selected venue"
                if research_count
                else "repair two-leg venue listing, liquidity, and freshness evidence"
            )
        ),
    }


def _check_row(name: str, ready: bool, evidence: str, next_action: str) -> dict[str, object]:
    return {"check": name, "ready": ready, "status": "ready" if ready else "blocked", "blocker": "" if ready else name, "evidence_path": evidence, "next_action": next_action}


def _package_check_row(package: str) -> dict[str, object]:
    try:
        __import__(package)
        ready = True
        blocker = ""
    except Exception as exc:
        ready = False
        blocker = f"missing_package:{package}:{type(exc).__name__}"
    return {"check": f"package:{package}", "ready": ready, "status": "ready" if ready else "blocked", "blocker": blocker, "evidence_path": "pyproject.toml", "next_action": "install project dependencies"}


def _pair_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame([{"pairs": 0, "promote": 0, "watch": 0, "fetch_more_data": 0, "reject": 0}])
    counts = frame["decision_bucket"].value_counts()
    return pd.DataFrame(
        [
            {
                "pairs": len(frame),
                "promote": int(counts.get("PROMOTE", 0)),
                "watch": int(counts.get("WATCH", 0)),
                "fetch_more_data": int(counts.get("FETCH_MORE_DATA", 0)),
                "reject": int(counts.get("REJECT", 0)),
                "top_pair": frame.sort_values("combined_score", ascending=False)["pair"].iloc[0],
            }
        ]
    )


def _pair_score_components(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["pair", "component", "score", "source", "reason"])
    rows = []
    for _, row in frame.iterrows():
        rows.extend(
            [
                {"pair": row["pair"], "component": "discovery_score", "score": row["discovery_score"], "source": "wizards_apify_local_hints", "reason": "discovery_only_not_promotion_authority"},
                {"pair": row["pair"], "component": "acceptance_score", "score": row["acceptance_score"], "source": "local_point_in_time_evidence", "reason": "promotion_authority"},
                {"pair": row["pair"], "component": "funding_drag_penalty", "score": -float(row.get("funding_drag_bps", 0.0) or 0.0), "source": "dydx_funding", "reason": "cost_drag"},
            ]
        )
    return pd.DataFrame(rows)


def _trade_dataset_summary(frame: pd.DataFrame, audit: pd.DataFrame, parquet_status: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rows": len(frame),
                "pairs": frame.get("pair", pd.Series(dtype=str)).nunique(),
                "future_leakage_rows": int(audit["uses_future_data"].astype(bool).sum()),
                "dashboard_hindsight_rows": int(audit["uses_dashboard_hindsight"].astype(bool).sum()),
                "mean_feature_completeness": float(audit["feature_completeness_score"].mean()),
                "parquet_status": parquet_status,
                "label_source": "backtest_label",
            }
        ]
    )


def _artifact_index_markdown(frame: pd.DataFrame) -> str:
    counts = frame["status"].value_counts().to_dict() if not frame.empty else {}
    lines = ["# Artifact Index", "", "Non-destructive classification. No files were moved.", "", "## Status Counts", ""]
    for status in ["active", "historical_evidence", "scratch", "superseded", "unknown", "do_not_move"]:
        lines.append(f"- {status}: {counts.get(status, 0)}")
    lines.extend(["", "## Rule", "", "Archive later only from this index, and only after active lineage is stable."])
    return "\n".join(lines) + "\n"


def _canonical_commands_markdown() -> str:
    lines = ["# Canonical Commands", ""]
    for command in CANONICAL_COMMANDS:
        lines.append(f"- `{command}`")
    return "\n".join(lines) + "\n"


def _current_state_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Current State", "", "| Area | Status | Setup | Blocker | Evidence | Next Action |", "| --- | --- | --- | --- | --- | --- |"]
    for _, row in frame.iterrows():
        setup = str(row.get("setup_identity", "") or row.get("candidate_id", "") or "").strip()
        lines.append(f"| {row['area']} | {row['status']} | {setup} | {row['blocker']} | {row['evidence_path']} | {row['next_action']} |")
    return "\n".join(lines) + "\n"


def _simple_report_markdown(title: str, frame: pd.DataFrame) -> str:
    lines = [f"# {title}", "", f"- rows: {len(frame)}", ""]
    if "ready" in frame:
        lines.append(f"- ready: {int(frame['ready'].astype(bool).sum())}")
        lines.append(f"- blocked: {int((~frame['ready'].astype(bool)).sum())}")
    return "\n".join(lines) + "\n"


def _pair_universe_markdown(frame: pd.DataFrame) -> str:
    summary = _pair_summary(frame)
    lines = ["# Pair Universe Summary", "", summary.to_markdown(index=False), "", "## Promotion Rule", "", "Crypto Wizards/dashboard-only evidence cannot produce PROMOTE."]
    return "\n".join(lines) + "\n"


def _command_center_markdown(
    pair_universe: pd.DataFrame,
    current: pd.DataFrame,
    acceptance: pd.DataFrame,
    data_health: pd.DataFrame,
    route_recommendations: pd.DataFrame | None = None,
    refresh_status: pd.DataFrame | None = None,
    root: Path = ROOT,
) -> str:
    lines = ["# Command Center", "", "## Current State", ""]
    if current.empty:
        lines.append("- current state not built")
    else:
        for _, row in current.iterrows():
            setup = _text_value(row.get("setup_identity", "")) or _text_value(row.get("candidate_id", ""))
            setup_text = f" [{setup}]" if setup else ""
            blocker = _text_value(row.get("blocker", ""))
            lines.append(f"- {_text_value(row.get('area', ''))}: {_text_value(row.get('status', ''))}{setup_text} {blocker}")
    authority = _read_csv(root / "reports" / "active" / "hyperliquid_authority_state.csv")
    run_candidates = _read_csv(root / "reports" / "active" / "hyperliquid_run_candidates.csv")
    family_selection_path = root / "reports" / "orchestration" / "teacher_council" / "research_family_selection_controls.csv"
    selection = _read_csv(
        family_selection_path
        if family_selection_path.exists()
        else root / "reports" / "orchestration" / "teacher_council" / "statistical_selection_controls.csv"
    )
    primary_selection = selection
    if not selection.empty and "research_lane" in selection.columns:
        primary_selection = selection.loc[selection["research_lane"].astype(str).eq("primary_daily")].copy()
    council = _read_csv(root / "reports" / "orchestration" / "teacher_council" / "council_decisions.csv")
    portfolio = _read_csv(root / "reports" / "orchestration" / "teacher_council" / "portfolio_critic.csv")
    student = _read_csv(root / "reports" / "orchestration" / "teacher_council" / "student_training_readiness.csv")
    student_modes = _read_csv(root / "reports" / "orchestration" / "teacher_council" / "student_mode_training_manifest.csv")
    auxiliary_4h = _read_csv(root / "reports" / "orchestration" / "teacher_council" / "auxiliary_4h_walkforward_mode_summary.csv")
    margin = _read_csv(root / "reports" / "active" / "hyperliquid_testnet_margin_snapshot.csv")
    lifecycle = _read_csv(root / "reports" / "active" / "hyperliquid_testnet_lifecycle_gate.csv")
    lines.extend(["", "## Canonical Hyperliquid Authority", ""])
    if authority.empty:
        lines.append("- blocked: canonical Hyperliquid authority state has not been built")
    else:
        row = authority.iloc[0]
        lines.append(f"- run: {_text_value(row.get('run_id', ''))}")
        lines.append(f"- candidate set: {_text_value(row.get('candidate_set_id', ''))}")
        lines.append(f"- status: {_text_value(row.get('status', 'BLOCKED'))}")
        lines.append(f"- candidates: {len(run_candidates)}")
        lines.append(f"- primary selection-control passes: {int(primary_selection.get('selection_status', pd.Series(dtype=str)).astype(str).eq('PASS').sum())}")
        if not selection.empty and "research_family_id" in selection.columns:
            family_row = selection.iloc[0]
            lines.append(f"- research-family tests: {int(family_row.get('research_family_tests', len(selection)) or len(selection))}")
            lines.append(f"- research attempt: {int(family_row.get('research_attempt_number', 0) or 0)}")
            lines.append(f"- alpha spend limit: {float(family_row.get('alpha_spend_limit', 0.0) or 0.0):.6f}")
        lines.append(f"- council shadow-test decisions: {int(council.get('status', pd.Series(dtype=str)).astype(str).eq('SHADOW_TEST').sum())}")
        lines.append(f"- portfolio critic: {_text_value(portfolio.iloc[0].get('verdict', 'missing')) if not portfolio.empty else 'missing'}")
        supervised_student = student
        if not student.empty and "scope" in student.columns:
            supervised_student = student[
                student["scope"].astype(str).isin({"all_learning", "supervised_student"})
            ]
        student_status = (
            "BLOCKED"
            if not supervised_student.empty
            and supervised_student.get("status", pd.Series(dtype=str)).astype(str).eq("BLOCKED").any()
            else "PASS"
            if not supervised_student.empty
            else "missing"
        )
        bandit_status = (
            "BLOCKED"
            if not student.empty
            and "scope" in student.columns
            and student.loc[student["scope"].astype(str).eq("contextual_bandit"), "status"].astype(str).eq("BLOCKED").any()
            else "PASS"
            if not student.empty and "scope" in student.columns
            else "missing"
        )
        mode_ready = int(
            student_modes.get("status", pd.Series(dtype=str)).astype(str).eq("READY_FOR_SHADOW_TRAINING").sum()
        )
        lines.append(f"- supervised student readiness: {student_status}")
        lines.append(f"- exact-mode specialists ready for shadow training: {mode_ready} of {len(student_modes)}")
        lines.append(f"- contextual-bandit readiness: {bandit_status}")
        lines.append(f"- auxiliary 4h mode tests: {len(auxiliary_4h)}")
        if not margin.empty:
            margin_row = margin.iloc[0]
            lines.append(f"- testnet perp account value: ${float(margin_row.get('account_value_usd', 0.0) or 0.0):.2f}")
            lines.append(f"- testnet spot USDC: ${float(margin_row.get('spot_usdc_usd', 0.0) or 0.0):.2f}")
            lines.append(f"- testnet margin blocker: {_text_value(margin_row.get('blockers', ''))}")
        lifecycle_passes = int(lifecycle.get("status", pd.Series(dtype=str)).astype(str).eq("PASS").sum())
        lines.append(f"- testnet lifecycle checks passed: {lifecycle_passes} of {len(lifecycle)}")
        lines.append(f"- execution allowed: {_boolish(row.get('execution_allowed', False))}")
        lines.append(f"- blocker: {_text_value(row.get('blocker', ''))}")
        lines.append("- Crypto Wizards is discovery-only; this authority is driven by identity-bound local Hyperliquid evidence")
    exhaustive_run = _read_json(
        root / "reports" / "active" / "exhaustive_wizard_hyperliquid_run_manifest.json"
    )
    exhaustive_api_refresh = _read_json(
        root / "reports" / "active" / "exhaustive_wizard_api_refresh_manifest.json"
    )
    pair_detail_api_pilot = _read_json(
        root / "reports" / "active" / "wizard_pair_detail_api_pilot_manifest.json"
    )
    current_wizard_handoff = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_handoff_manifest.json"
    )
    current_wizard_history = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_history_manifest.json"
    )
    current_wizard_replay = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_canonical_replay_manifest.json"
    )
    current_wizard_costs = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_cost_manifest.json"
    )
    current_wizard_observed = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_observed_cost_replay_manifest.json"
    )
    current_wizard_walkforward = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_walkforward_manifest.json"
    )
    current_wizard_regime = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_regime_manifest.json"
    )
    current_wizard_robustness = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_robustness_manifest.json"
    )
    current_wizard_concentration = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_concentration_manifest.json"
    )
    current_wizard_failure_attribution = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_failure_attribution_manifest.json"
    )
    current_wizard_leverage = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_leverage_manifest.json"
    )
    current_wizard_learning = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_learning_manifest.json"
    )
    current_wizard_chain_validation = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_chain_validation_manifest.json"
    )
    current_wizard_cadence = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_operating_cadence_manifest.json"
    )
    current_wizard_storage_reclamation = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_storage_reclamation_manifest.json"
    )
    current_wizard_archive_copy = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_archive_copy_manifest.json"
    )
    current_wizard_archive_release = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_archive_release_manifest.json"
    )
    current_wizard_testnet_protocol = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_hyperliquid_testnet_protocol_manifest.json"
    )
    current_wizard_ou_optimal_overlay = _read_json(
        root
        / "reports"
        / "active"
        / "current_wizard_ou_optimal_overlay_manifest.json"
    )
    concentration = _read_json(
        root
        / "reports"
        / "active"
        / "exhaustive_wizard_hyperliquid_concentration_manifest.json"
    )
    leverage = _read_json(
        root / "reports" / "active" / "exhaustive_wizard_hyperliquid_leverage_manifest.json"
    )
    validation = _read_json(
        root / "reports" / "active" / "exhaustive_wizard_hyperliquid_validation_manifest.json"
    )
    exhaustive_learning = _read_json(
        root / "reports" / "active" / "exhaustive_wizard_hyperliquid_learning_manifest.json"
    )
    exhaustive_walkforward = _read_json(
        root / "reports" / "active" / "exhaustive_wizard_hyperliquid_walkforward_manifest.json"
    )
    exhaustive_regime = _read_json(
        root / "reports" / "active" / "exhaustive_wizard_hyperliquid_regime_manifest.json"
    )
    exhaustive_robustness = _read_json(
        root / "reports" / "active" / "exhaustive_wizard_hyperliquid_robustness_manifest.json"
    )
    cadence = _read_csv(root / "reports" / "active" / "hyperliquid_evidence_cadence.csv")
    testnet_preflight = _read_csv(
        root / "reports" / "active" / "hyperliquid_testnet_preflight.csv"
    )
    lines.extend(["", "## Exhaustive Wizard To Hyperliquid", ""])
    if not exhaustive_run:
        lines.append("- blocked: exhaustive Wizard-to-Hyperliquid run has not been built")
    else:
        lines.append(f"- run: {_text_value(exhaustive_run.get('run_id', ''))}")
        lines.append(f"- source rows accounted: {int(exhaustive_run.get('source_rows', 0) or 0)}")
        lines.append(f"- pair groups accounted: {int(exhaustive_run.get('pair_groups', 0) or 0)}")
        if exhaustive_api_refresh:
            lines.append(
                "- current API refresh: "
                f"{_text_value(exhaustive_api_refresh.get('refresh_id', 'missing'))}"
            )
            lines.append(
                "- current API rows accounted: "
                f"{int(exhaustive_api_refresh.get('api_source_rows_accounted', 0) or 0)} of "
                f"{int(exhaustive_api_refresh.get('api_source_rows', 0) or 0)}"
            )
            lines.append(
                "- current API / frozen UI / union pair groups: "
                f"{int(exhaustive_api_refresh.get('api_pair_groups', 0) or 0)} / "
                f"{int(exhaustive_api_refresh.get('frozen_pair_groups', 0) or 0)} / "
                f"{int(exhaustive_api_refresh.get('union_pair_groups', 0) or 0)}"
            )
            lines.append(
                "- new API pair-detail captures required: "
                f"{int(exhaustive_api_refresh.get('new_api_discoveries', 0) or 0)}"
            )
            lines.append(
                "- current API pair-detail queue complete: "
                f"{_boolish(exhaustive_api_refresh.get('pair_detail_queue_complete', False))}"
            )
            lines.append(
                "- union pairs mapped / Hyperliquid-ready / blocked: "
                f"{int(exhaustive_api_refresh.get('hyperliquid_pair_groups_mapped', 0) or 0)} / "
                f"{int(exhaustive_api_refresh.get('hyperliquid_ready_pair_groups', 0) or 0)} / "
                f"{int(exhaustive_api_refresh.get('hyperliquid_blocked_pair_groups', 0) or 0)}"
            )
            lines.append(
                "- current API pairs mapped / Hyperliquid-ready / blocked: "
                f"{int(exhaustive_api_refresh.get('current_api_hyperliquid_pair_groups_mapped', 0) or 0)} / "
                f"{int(exhaustive_api_refresh.get('current_api_hyperliquid_ready_pair_groups', 0) or 0)} / "
                f"{int(exhaustive_api_refresh.get('current_api_hyperliquid_blocked_pair_groups', 0) or 0)}"
            )
            lines.append(
                "- pair-detail browser route: "
                f"{_text_value(exhaustive_api_refresh.get('pair_detail_browser_route_status', 'missing'))}"
            )
            lines.append(
                "- estimated Wizard credits available after reserve: "
                f"{int(exhaustive_api_refresh.get('credits_available_after_reserve_estimate', 0) or 0)}"
            )
            if pair_detail_api_pilot:
                lines.append(
                    "- pair-detail API pilot endpoints complete: "
                    f"{int(pair_detail_api_pilot.get('completed_endpoints', 0) or 0)} of "
                    f"{int(pair_detail_api_pilot.get('planned_endpoints', 0) or 0)}"
                )
                lines.append(
                    "- pair-detail API fields / coverage groups: "
                    f"{int(pair_detail_api_pilot.get('observed_fields', 0) or 0)} / "
                    f"{int(pair_detail_api_pilot.get('coverage_passes', 0) or 0)} of "
                    f"{int(pair_detail_api_pilot.get('coverage_checks', 0) or 0)}"
                )
                lines.append(
                    "- pair-detail API retry credits planned / observed: "
                    f"{int(pair_detail_api_pilot.get('planned_credits', 0) or 0)} / "
                    f"{int(pair_detail_api_pilot.get('observed_credit_delta', 0) or 0)}"
                )
                lines.append(
                    "- current Wizard credits used / available after reserve: "
                    f"{int(pair_detail_api_pilot.get('credits_used_after', 0) or 0)} / "
                    f"{int(pair_detail_api_pilot.get('credits_available_after_reserve', 0) or 0)}"
                )
                lines.append(
                    "- pair-detail API ECM fields found: "
                    f"{_boolish(pair_detail_api_pilot.get('ecm_fields_found', False))}"
                )
                lines.append(
                    "- pair-detail API authority / blocker: "
                    f"{_text_value(pair_detail_api_pilot.get('authority', 'missing'))} / "
                    f"{_text_value(pair_detail_api_pilot.get('blocker', 'missing'))}"
                )
            if current_wizard_handoff:
                lines.append(
                    "- current-board pair groups accounted: "
                    f"{int(current_wizard_handoff.get('pair_groups_accounted', 0) or 0)} of "
                    f"{int(current_wizard_handoff.get('pair_groups', 0) or 0)}"
                )
                lines.append(
                    "- current complete vendor details / historical UI / API pilots: "
                    f"{int(current_wizard_handoff.get('current_vendor_pair_detail_complete', 0) or 0)} / "
                    f"{int(current_wizard_handoff.get('historical_ui_pair_groups', 0) or 0)} / "
                    f"{int(current_wizard_handoff.get('api_schema_pilot_pairs', 0) or 0)}"
                )
                lines.append(
                    "- current-board Hyperliquid-ready / blocked pairs: "
                    f"{int(current_wizard_handoff.get('hyperliquid_ready_pair_groups', 0) or 0)} / "
                    f"{int(current_wizard_handoff.get('hyperliquid_blocked_pair_groups', 0) or 0)}"
                )
                lines.append(
                    "- current-board mode/orientation experiments planned / history-ready: "
                    f"{int(current_wizard_handoff.get('planned_experiments', 0) or 0)} / "
                    f"{int(current_wizard_handoff.get('ready_for_point_in_time_history_experiments', 0) or 0)}"
                )
                lines.append(
                    "- current-board pair histories / asset requests ready: "
                    f"{int(current_wizard_handoff.get('pair_history_ready_to_fetch', 0) or 0)} / "
                    f"{int(current_wizard_handoff.get('asset_interval_fetch_requests', 0) or 0)}"
                )
                if current_wizard_history:
                    lines.append(
                        "- bounded history selected / ready / all pairs accounted: "
                        f"{int(current_wizard_history.get('selected_pair_groups', 0) or 0)} / "
                        f"{int(current_wizard_history.get('pair_histories_ready', 0) or 0)} / "
                        f"{int(current_wizard_history.get('pair_groups_accounted', 0) or 0)}"
                    )
                    lines.append(
                        "- bounded history storage preflight / live authority: "
                        f"{_boolish(current_wizard_history.get('storage_preflight_passed', False))} / "
                        f"{_boolish(current_wizard_history.get('live_trading_authorized', False))}"
                    )
                if current_wizard_replay:
                    lines.append(
                        "- current-board canonical 1x cells complete / deferred / accounted: "
                        f"{int(current_wizard_replay.get('research_replays_complete', 0) or 0)} / "
                        f"{int(current_wizard_replay.get('deferred_point_in_time_history', 0) or 0)} / "
                        f"{int(current_wizard_replay.get('experiments_accounted', 0) or 0)}"
                    )
                    lines.append(
                        "- current-board acceptance eligible / live authority: "
                        f"{int(current_wizard_replay.get('acceptance_eligible_replays', 0) or 0)} / "
                        f"{_boolish(current_wizard_replay.get('live_trading_authorized', False))}"
                    )
                if current_wizard_costs:
                    lines.append(
                        "- current-board observed-cost research / strict calibration: "
                        f"{int(current_wizard_costs.get('pairs_ready_for_observed_cost_research', 0) or 0)} / "
                        f"{int(current_wizard_costs.get('pairs_strict_cost_calibrated', 0) or 0)}"
                    )
                    lines.append(
                        "- current-board cost pair/experiment accounting and live authority: "
                        f"{int(current_wizard_costs.get('pair_groups_accounted', 0) or 0)} / "
                        f"{int(current_wizard_costs.get('experiments_accounted', 0) or 0)} / "
                        f"{_boolish(current_wizard_costs.get('live_trading_authorized', False))}"
                    )
                if current_wizard_observed:
                    lines.append(
                        "- current-board observed-cost cells complete / research-rank eligible: "
                        f"{int(current_wizard_observed.get('replays_complete', 0) or 0)} / "
                        f"{int(current_wizard_observed.get('research_rank_eligible_replays', 0) or 0)}"
                    )
                    lines.append(
                        "- current-board strict-cost cells / acceptance / live authority: "
                        f"{int(current_wizard_observed.get('strict_cost_calibrated_replays', 0) or 0)} / "
                        f"{int(current_wizard_observed.get('acceptance_eligible_replays', 0) or 0)} / "
                        f"{_boolish(current_wizard_observed.get('live_trading_authorized', False))}"
                    )
                if current_wizard_walkforward:
                    lines.append(
                        "- current-board walk-forward candidates / practical passes / statistical passes: "
                        f"{int(current_wizard_walkforward.get('walkforward_candidates_completed', 0) or 0)} / "
                        f"{int(current_wizard_walkforward.get('walkforward_passes', 0) or 0)} / "
                        f"{int(current_wizard_walkforward.get('statistical_selection_passes', 0) or 0)}"
                    )
                    lines.append(
                        "- current-board walk-forward fold accounting / live authority: "
                        f"{int(current_wizard_walkforward.get('folds_complete', 0) or 0)} / "
                        f"{_boolish(current_wizard_walkforward.get('live_trading_authorized', False))}"
                    )
                if current_wizard_regime:
                    lines.append(
                        "- current-board regime attributions / stability passes: "
                        f"{int(current_wizard_regime.get('regime_attributions_complete', 0) or 0)} / "
                        f"{int(current_wizard_regime.get('regime_stability_passes', 0) or 0)}"
                    )
                    lines.append(
                        "- current-board causal regime thresholds / live authority: "
                        f"{_boolish(current_wizard_regime.get('point_in_time_regime_thresholds', False))} / "
                        f"{_boolish(current_wizard_regime.get('live_trading_authorized', False))}"
                    )
                if current_wizard_robustness:
                    lines.append(
                        "- current-board robustness candidates / research passes / statistical passes: "
                        f"{int(current_wizard_robustness.get('robustness_candidates_complete', 0) or 0)} / "
                        f"{int(current_wizard_robustness.get('research_robustness_passes', 0) or 0)} / "
                        f"{int(current_wizard_robustness.get('statistically_selected_robustness_passes', 0) or 0)}"
                    )
                    lines.append(
                        "- current-board robustness scenarios / folds / live authority: "
                        f"{int(current_wizard_robustness.get('scenario_rows', 0) or 0)} / "
                        f"{int(current_wizard_robustness.get('fold_rows', 0) or 0)} / "
                        f"{_boolish(current_wizard_robustness.get('live_trading_authorized', False))}"
                    )
                if current_wizard_concentration:
                    lines.append(
                        "- current-board concentration practical / robustness / statistical cohorts: "
                        f"{int(current_wizard_concentration.get('practical_walkforward_candidates', 0) or 0)} / "
                        f"{int(current_wizard_concentration.get('research_robustness_candidates', 0) or 0)} / "
                        f"{int(current_wizard_concentration.get('statistically_selected_robustness_candidates', 0) or 0)}"
                    )
                    lines.append(
                        "- current-board concentration promotion gate / leverage-ready / live authority: "
                        f"{_boolish(current_wizard_concentration.get('promotion_concentration_pass', False))} / "
                        f"{_boolish(current_wizard_concentration.get('ready_for_leverage_gate', False))} / "
                        f"{_boolish(current_wizard_concentration.get('live_trading_authorized', False))}"
                    )
                if current_wizard_failure_attribution:
                    lines.append(
                        "- current-board all-cell failure attribution / 1x survivors / leverage-research eligible: "
                        f"{int(current_wizard_failure_attribution.get('experiments_accounted', 0) or 0)} / "
                        f"{int(current_wizard_failure_attribution.get('one_x_research_survivors', 0) or 0)} / "
                        f"{int(current_wizard_failure_attribution.get('leverage_research_eligible', 0) or 0)}"
                    )
                    lines.append(
                        "- current-board execution-ready / Testnet-preflight eligible / live authority: "
                        f"{int(current_wizard_failure_attribution.get('execution_acceptance_ready', 0) or 0)} / "
                        f"{int(current_wizard_failure_attribution.get('testnet_preflight_eligible', 0) or 0)} / "
                        f"{_boolish(current_wizard_failure_attribution.get('live_trading_authorized', False))}"
                    )
                if current_wizard_leverage:
                    lines.append(
                        "- current-board leverage 1x survivors / candidate surfaces / scenario rows: "
                        f"{int(current_wizard_leverage.get('one_x_research_survivors_selected', 0) or 0)} / "
                        f"{int(current_wizard_leverage.get('leverage_candidates_complete', 0) or 0)} / "
                        f"{int(current_wizard_leverage.get('scenario_rows', 0) or 0)}"
                    )
                    lines.append(
                        "- current-board leverage Testnet-ready / market evidence fresh / live authority: "
                        f"{int(current_wizard_leverage.get('testnet_1x_lifecycle_ready', 0) or 0)} / "
                        f"{_boolish(current_wizard_leverage.get('market_evidence_fresh', False))} / "
                        f"{_boolish(current_wizard_leverage.get('live_trading_authorized', False))}"
                    )
                if current_wizard_testnet_protocol:
                    lines.append(
                        "- current-board Testnet protocol simulation / scenarios passed / "
                        "actual Testnet proof: "
                        f"{_text_value(current_wizard_testnet_protocol.get('protocol_status', 'missing'))} / "
                        f"{int(current_wizard_testnet_protocol.get('scenarios_passed', 0) or 0)} of "
                        f"{int(current_wizard_testnet_protocol.get('scenarios', 0) or 0)} / "
                        f"{_boolish(current_wizard_testnet_protocol.get('simulation_is_testnet_proof', False))}"
                    )
                    lines.append(
                        "- current-board Testnet protocol candidates / order submission / live authority: "
                        f"{int(current_wizard_testnet_protocol.get('actual_testnet_candidates', 0) or 0)} / "
                        f"{_boolish(current_wizard_testnet_protocol.get('order_submission_performed', False))} / "
                        f"{_boolish(current_wizard_testnet_protocol.get('live_trading_authorized', False))}"
                    )
                if current_wizard_learning:
                    lines.append(
                        "- current-board dated learning records / training eligible: "
                        f"{int(current_wizard_learning.get('records', 0) or 0)} / "
                        f"{int(current_wizard_learning.get('training_eligible_records', 0) or 0)}"
                    )
                    lines.append(
                        "- current-board paper labels / live labels / order submission: "
                        f"{int(current_wizard_learning.get('paper_label_records', 0) or 0)} / "
                        f"{int(current_wizard_learning.get('live_label_records', 0) or 0)} / "
                        f"{_boolish(current_wizard_learning.get('order_submission_performed', False))}"
                    )
                if current_wizard_chain_validation:
                    lines.append(
                        "- current-board frozen chain status / stages / checks: "
                        f"{_text_value(current_wizard_chain_validation.get('chain_status', 'missing'))} / "
                        f"{int(current_wizard_chain_validation.get('stages_complete', 0) or 0)} of "
                        f"{int(current_wizard_chain_validation.get('stages_expected', 0) or 0)} / "
                        f"{int(current_wizard_chain_validation.get('checks_passed', 0) or 0)} of "
                        f"{int(current_wizard_chain_validation.get('checks', 0) or 0)}"
                    )
                    lines.append(
                        "- current-board trade eligibility / order submission / live authority: "
                        f"{_text_value(current_wizard_chain_validation.get('trade_eligibility_status', 'missing'))} / "
                        f"{_boolish(current_wizard_chain_validation.get('order_submission_performed', False))} / "
                        f"{_boolish(current_wizard_chain_validation.get('live_trading_authorized', False))}"
                    )
                if current_wizard_cadence:
                    lines.append(
                        "- current-board operating cadence stages / daily run ready / blocker: "
                        f"{int(current_wizard_cadence.get('stages', 0) or 0)} / "
                        f"{_boolish(current_wizard_cadence.get('daily_research_run_ready', False))} / "
                        f"{_text_value(current_wizard_cadence.get('daily_research_run_blocker', '')) or 'none'}"
                    )
                    lines.append(
                        "- current-board scheduled Testnet authority / order submission / live authority: "
                        f"{_boolish(current_wizard_cadence.get('scheduled_testnet_order_authority', False))} / "
                        f"{_boolish(current_wizard_cadence.get('order_submission_performed', False))} / "
                        f"{_boolish(current_wizard_cadence.get('live_trading_authorized', False))}"
                    )
                    lines.append(
                        "- current-board immutable input bytes referenced / copied / recursive copies avoided: "
                        f"{int(current_wizard_cadence.get('referenced_upstream_bytes', 0) or 0)} / "
                        f"{int(current_wizard_cadence.get('locally_copied_input_bytes', 0) or 0)} / "
                        f"{int(current_wizard_cadence.get('recursive_copy_bytes_avoided', 0) or 0)}"
                    )
                if current_wizard_storage_reclamation:
                    lines.append(
                        "- current-board safe archive candidates / hash verified / "
                        "reclaimable bytes / projected storage floor met: "
                        f"{int(current_wizard_storage_reclamation.get('safe_archive_candidate_branches', 0) or 0)} / "
                        f"{int(current_wizard_storage_reclamation.get('hash_verified_candidate_branches', 0) or 0)} / "
                        f"{int(current_wizard_storage_reclamation.get('reclaimable_bytes_if_archived_off_volume', 0) or 0)} / "
                        f"{_boolish(current_wizard_storage_reclamation.get('storage_floor_met_after_safe_archive', False))}"
                    )
                    lines.append(
                        "- current-board archive destination / free bytes / required bytes / copy preflight: "
                        f"{_text_value(current_wizard_storage_reclamation.get('archive_destination_path', '')) or 'not configured'} / "
                        f"{int(current_wizard_storage_reclamation.get('archive_destination_free_bytes', 0) or 0)} / "
                        f"{int(current_wizard_storage_reclamation.get('archive_destination_required_bytes', 0) or 0)} / "
                        f"{_boolish(current_wizard_storage_reclamation.get('archive_copy_preflight_ready', False))}"
                    )
                    lines.append(
                        "- current-board archive destination status / blocker / release preflight: "
                        f"{_text_value(current_wizard_storage_reclamation.get('archive_destination_status', 'BLOCKED'))} / "
                        f"{_text_value(current_wizard_storage_reclamation.get('archive_destination_blocker', '')) or 'none'} / "
                        f"{_boolish(current_wizard_storage_reclamation.get('archive_release_preflight_ready', False))}"
                    )
                    lines.append(
                        "- current-board archive authority / move performed / delete performed: "
                        f"{_boolish(current_wizard_storage_reclamation.get('archive_apply_authority', False))} / "
                        f"{_boolish(current_wizard_storage_reclamation.get('move_performed', False))} / "
                        f"{_boolish(current_wizard_storage_reclamation.get('delete_performed', False))}"
                    )
                if current_wizard_archive_copy:
                    lines.append(
                        "- current-board archive copy / branches / bytes verified / completed: "
                        f"{_text_value(current_wizard_archive_copy.get('copy_id', 'missing'))} / "
                        f"{int(current_wizard_archive_copy.get('candidate_branches', 0) or 0)} / "
                        f"{int(current_wizard_archive_copy.get('bytes_verified', 0) or 0)} / "
                        f"{_boolish(current_wizard_archive_copy.get('archive_copy_completed', False))}"
                    )
                    lines.append(
                        "- current-board source release / source move / source delete: "
                        f"{_boolish(current_wizard_archive_copy.get('source_release_authorized', False))} / "
                        f"{_boolish(current_wizard_archive_copy.get('source_move_performed', False))} / "
                        f"{_boolish(current_wizard_archive_copy.get('source_delete_performed', False))}"
                    )
                else:
                    lines.append(
                        "- current-board archive copy / branches / bytes verified / completed: "
                        "missing / 0 / 0 / False"
                    )
                    lines.append(
                        "- current-board source release / source move / source delete: "
                        "False / False / False"
                    )
                if current_wizard_archive_release:
                    lines.append(
                        "- current-board archive release dry run / candidates / bytes / ready: "
                        f"{_text_value(current_wizard_archive_release.get('release_id', 'missing'))} / "
                        f"{int(current_wizard_archive_release.get('release_candidates', 0) or 0)} / "
                        f"{int(current_wizard_archive_release.get('recoverable_bytes', 0) or 0)} / "
                        f"{_boolish(current_wizard_archive_release.get('release_dry_run_ready', False))}"
                    )
                    lines.append(
                        "- current-board archive release blocker / authorized / delete performed: "
                        f"{_text_value(current_wizard_archive_release.get('release_blocker', '')) or 'none'} / "
                        f"{_boolish(current_wizard_archive_release.get('source_release_authorized', False))} / "
                        f"{_boolish(current_wizard_archive_release.get('source_delete_performed', False))}"
                    )
                else:
                    lines.append(
                        "- current-board archive release dry run / candidates / bytes / ready: "
                        "missing / 0 / 0 / False"
                    )
                    lines.append(
                        "- current-board archive release blocker / authorized / delete performed: "
                        "archive_release_dry_run_missing / False / False"
                    )
                if current_wizard_ou_optimal_overlay:
                    lines.append(
                        "- Wizard pair-page modes / scanner overlays: 7 / "
                        f"{_text_value(current_wizard_ou_optimal_overlay.get('scanner_overlay', 'missing'))}"
                    )
                    lines.append(
                        "- OU Optimal source rows accounted / true / false: "
                        f"{int(current_wizard_ou_optimal_overlay.get('source_rows_accounted', 0) or 0)} / "
                        f"{int(current_wizard_ou_optimal_overlay.get('ou_optimal_true_rows', 0) or 0)} / "
                        f"{int(current_wizard_ou_optimal_overlay.get('ou_optimal_false_rows', 0) or 0)}"
                    )
                    lines.append(
                        "- OU Optimal true base replays / walk-forward passes / selected: "
                        f"{int(current_wizard_ou_optimal_overlay.get('ou_optimal_true_base_replays_complete', 0) or 0)} / "
                        f"{int(current_wizard_ou_optimal_overlay.get('ou_optimal_true_walkforward_passes', 0) or 0)} / "
                        f"{int(current_wizard_ou_optimal_overlay.get('ou_optimal_true_statistically_selected', 0) or 0)}"
                    )
                    lines.append(
                        "- independent OU Optimal pair-page captures / promotion authority / live authority: "
                        f"{int(current_wizard_ou_optimal_overlay.get('pair_page_ou_optimal_captured_rows', 0) or 0)} / "
                        f"{_boolish(current_wizard_ou_optimal_overlay.get('promotion_authority', False))} / "
                        f"{_boolish(current_wizard_ou_optimal_overlay.get('live_trading_authorized', False))}"
                    )
        lines.append(
            "- frozen browser-run exact-mode/orientation experiments accounted: "
            f"{int(exhaustive_run.get('planned_experiments', 0) or 0)}"
        )
        lines.append(
            "- latest frozen validation: "
            f"{_text_value(validation.get('validation_id', 'missing'))}"
        )
        lines.append(
            "- validation stages complete: "
            f"{int(validation.get('stages_complete', 0) or 0)} of "
            f"{int(validation.get('expected_stages', 0) or 0)}"
        )
        lines.append(
            "- dated research learning records: "
            f"{int(exhaustive_learning.get('records', 0) or 0)}"
        )
        lines.append(
            "- learning experiment accounting complete: "
            f"{_boolish(exhaustive_learning.get('experiment_status_accounted', False))}"
        )
        lines.append(
            "- training-eligible experiment summaries: "
            f"{int(exhaustive_learning.get('training_eligible_records', 0) or 0)}"
        )
        lines.append(
            "- realized paper/live labels in research ledger: "
            f"{int(exhaustive_learning.get('paper_label_records', 0) or 0)} / "
            f"{int(exhaustive_learning.get('live_label_records', 0) or 0)}"
        )
        lines.append(
            "- practical walk-forward passes: "
            f"{int(exhaustive_walkforward.get('walkforward_passes', 0) or 0)}"
        )
        lines.append(
            "- family-wide statistical selection passes: "
            f"{int(exhaustive_walkforward.get('statistical_selection_passes', 0) or 0)}"
        )
        lines.append(
            "- regime-stability diagnostic passes: "
            f"{int(exhaustive_regime.get('regime_stability_passes', 0) or 0)}"
        )
        lines.append(
            "- robustness diagnostic passes: "
            f"{int(exhaustive_robustness.get('research_robustness_passes', 0) or 0)}"
        )
        lines.append(
            "- promotion concentration gate: "
            f"{'PASS' if _boolish(concentration.get('promotion_concentration_pass', False)) else 'BLOCKED'}"
        )
        lines.append(
            "- concentration-ready leverage candidates: "
            f"{int(leverage.get('concentration_ready_candidates', 0) or 0)}"
        )
        lines.append(
            "- leverage scenario rows: "
            f"{int(leverage.get('scenario_rows', 0) or 0)}"
        )
        lines.append(
            "- Testnet 1x lifecycle-ready candidates: "
            f"{int(leverage.get('testnet_1x_lifecycle_ready', 0) or 0)}"
        )
        lines.append(
            "- leverage venue evidence fresh: "
            f"{_boolish(leverage.get('market_evidence_fresh', False))}"
        )
        if cadence.empty:
            lines.append("- L2 calibration: missing")
        else:
            calibrated = int(cadence["status"].astype(str).eq("calibrated").sum())
            minimum = pd.to_numeric(cadence.get("minimum_samples"), errors="coerce").min()
            required = pd.to_numeric(
                cadence.get("model_required_samples"), errors="coerce"
            ).max()
            lines.append(
                f"- L2 calibration: {calibrated} of {len(cadence)} pairs calibrated; "
                f"rolling sample floor {int(minimum or 0)} of {int(required or 0)}"
            )
            projected_captures = pd.to_numeric(
                cadence.get(
                    "projected_captures_to_calibration",
                    pd.Series(index=cadence.index, dtype=float),
                ),
                errors="coerce",
            ).max()
            projected_at = _text_value(
                cadence.get("projected_calibration_at", pd.Series(dtype=str)).max()
            )
            if pd.notna(projected_captures):
                lines.append(
                    "- projected L2 captures remaining after rolling expiry: "
                    f"{int(projected_captures)}"
                )
            if projected_at:
                lines.append(f"- projected L2 calibration time: {projected_at}")
        if testnet_preflight.empty:
            lines.append("- Testnet no-order preflight: missing")
        else:
            preflight_row = testnet_preflight.iloc[0]
            lines.append(
                "- Testnet no-order preflight: "
                f"{'PASS' if _boolish(preflight_row.get('ready_for_no_order_preflight', False)) else 'BLOCKED'}"
            )
            lines.append(
                "- Testnet submissions enabled: "
                f"{_boolish(preflight_row.get('submit_orders_enabled', False))}"
            )
        lines.append(
            "- live trading authorized: "
            f"{_boolish(leverage.get('live_trading_authorized', False))}"
        )
    lines.extend(["", "## Dashboard Refresh", ""])
    if refresh_status is None or refresh_status.empty:
        lines.append("- refresh status unavailable")
    else:
        profile = _text_value(refresh_status.get("refresh_profile", pd.Series("", index=refresh_status.index)).iloc[0])
        rebuilt = int(refresh_status.get("status", pd.Series("", index=refresh_status.index)).astype(str).eq("rebuilt").sum())
        reused = int(refresh_status.get("status", pd.Series("", index=refresh_status.index)).astype(str).eq("reused_current_artifact").sum())
        missing = int(refresh_status.get("status", pd.Series("", index=refresh_status.index)).astype(str).str.contains("missing", na=False).sum())
        storage_blocked = refresh_status.get("status", pd.Series("", index=refresh_status.index)).astype(str).eq("blocked_insufficient_free_space")
        lines.append(f"- profile: {profile or 'unknown'}")
        lines.append(f"- rebuilt stages: {rebuilt}")
        lines.append(f"- reused current artifacts: {reused}")
        lines.append(f"- missing current artifacts: {missing}")
        if storage_blocked.any():
            storage_reason = _text_value(refresh_status.loc[storage_blocked, "reason"].iloc[0])
            lines.append(f"- deep refresh blocked by storage: {storage_reason}")
            lines.append("- this is a monitor-only snapshot; it does not create fresh research or acceptance authority")
        elif profile == "monitor":
            lines.append("- monitor profile never turns reused evidence into new acceptance authority")
        else:
            lines.append("- deep profile rebuilds evidence but still requires local point-in-time acceptance proof")
    lines.extend(["", "## Pair Universe", ""])
    if pair_universe.empty:
        lines.append("- pair universe missing")
    else:
        counts = pair_universe["decision_bucket"].value_counts()
        for bucket in ["PROMOTE", "WATCH", "FETCH_MORE_DATA", "REJECT"]:
            lines.append(f"- {bucket}: {int(counts.get(bucket, 0))}")
    wizard_current = _read_csv(root / "reports" / "active" / "wizard_discovery_current.csv")
    wizard_shortlist = _read_csv(root / "reports" / "active" / "wizard_discovery_shortlist.csv")
    wizard_copula = _read_csv(root / "reports" / "active" / "wizard_copula_discovery_triage.csv")
    capture_queue = _read_csv(root / "reports" / "active" / "wizard_pair_detail_capture_queue.csv")
    mode_matrix_queue = _read_csv(root / "reports" / "active" / "wizard_mode_matrix_capture_queue.csv")
    settings_validation = _read_csv(root / "reports" / "active" / "wizard_pair_settings_capture_validation.csv")
    replay_handoff = _read_csv(root / "reports" / "active" / "wizard_replay_handoff.csv")
    mode_capability = _read_csv(root / "reports" / "active" / "wizard_mode_replay_capability.csv")
    mode_comparison = _read_csv(root / "reports" / "active" / "wizard_mode_comparison.csv")
    exploratory_cost_sensitivity = _read_csv(root / "reports" / "active" / "wizard_exploratory_cost_sensitivity.csv")
    multi_venue_readiness = _read_csv(current_multi_venue_history_readiness_path(root))
    binance_readiness = _read_csv(root / "reports" / "active" / "binance_spot_pair_readiness.csv")
    lines.extend(["", "## Wizard Discovery", ""])
    if wizard_current.empty:
        lines.append("- blocked: no current Wizard discovery evidence")
    else:
        screen_pass = wizard_current.get("discovery_screen_status", pd.Series("", index=wizard_current.index)).astype(str).eq("SCREEN_PASS")
        research_ready = wizard_shortlist.get("research_quality_status", pd.Series("", index=wizard_shortlist.index)).astype(str).eq("RESEARCH_READY")
        lines.append(f"- current scanner rows: {len(wizard_current)}")
        lines.append(f"- Sharpe/return screen passes: {int(screen_pass.sum())}")
        lines.append(f"- research-ready shortlist rows: {int(research_ready.sum())} of {len(wizard_shortlist)}")
        lines.append("- authority: discovery only; local point-in-time replay remains required for acceptance")
    lines.extend(["", "## Copula Research", ""])
    lines.append("- Copula conditional values are a hypothesis signal, not a trade direction")
    if wizard_copula.empty:
        lines.append("- no current Copula rows captured")
    else:
        copula_shortlist = wizard_shortlist.get("exact_mode", pd.Series("", index=wizard_shortlist.index)).astype(str).str.lower().eq("copula")
        threshold_missing = wizard_copula.get("entry_signal_status", pd.Series("", index=wizard_copula.index)).astype(str).eq("COPULA_THRESHOLD_CAPTURE_REQUIRED")
        lines.append(f"- current Copula rows: {len(wizard_copula)}")
        lines.append(f"- Copula rows passing Sharpe/return screen: {int(copula_shortlist.sum())}")
        lines.append(f"- rows requiring directional threshold capture: {int(threshold_missing.sum())}")
    lines.extend(["", "## Settings Capture", ""])
    if capture_queue.empty:
        lines.append("- no current pair-detail settings captures required")
    else:
        priority_one = capture_queue.get("priority", pd.Series("", index=capture_queue.index)).astype(str).eq("P1")
        lines.append(f"- queued pair-detail captures: {len(capture_queue)}")
        lines.append(f"- priority-one captures: {int(priority_one.sum())}")
        if priority_one.any():
            row = capture_queue.loc[priority_one].iloc[0]
            lines.append(
                "- next capture: "
                f"{_text_value(row.get('pair', ''))} on {_text_value(row.get('exchange', ''))} "
                f"({_text_value(row.get('exact_mode', ''))})"
            )
        lines.append("- missing settings block a local mode-matched replay; values are never guessed")
        lines.append("- capture template: reports/templates/wizard_pair_settings_capture_template.csv")
    lines.extend(["", "## Seven-Mode Capture Matrix", ""])
    if mode_matrix_queue.empty:
        lines.append("- no seven-mode capture queue built yet")
    else:
        matrix_required = mode_matrix_queue.get(
            "capture_status", pd.Series("", index=mode_matrix_queue.index)
        ).astype(str).eq("CAPTURE_REQUIRED")
        matrix_pairs = mode_matrix_queue[[column for column in ["pair", "exchange", "interval", "period"] if column in mode_matrix_queue.columns]].drop_duplicates()
        lines.append(f"- pair/timeframe groups: {len(matrix_pairs)}")
        lines.append(f"- seven-mode capture rows: {len(mode_matrix_queue)}")
        lines.append(f"- modes still requiring visible settings capture: {int(matrix_required.sum())}")
        lines.append("- scanner mode is discovery evidence only; comparison modes must be selected and captured independently")
    if not settings_validation.empty:
        accepted = settings_validation.get(
            "accepted_into_active_capture", pd.Series(False, index=settings_validation.index)
        ).map(_boolish)
        lines.append(f"- latest imported settings rows: {len(settings_validation)}")
        lines.append(f"- validated settings rows: {int(accepted.sum())}")
        invalid = settings_validation.loc[~accepted, "validation_blockers"].dropna().astype(str).head(1).tolist()
        if invalid:
            lines.append(f"- latest capture blocker: {invalid[0]}")
    lines.extend(["", "## Exact-Mode Replay Handoff", ""])
    if replay_handoff.empty:
        lines.append("- no exact-mode replay handoff rows")
    else:
        exploratory_ready = replay_handoff.get("exploratory_replay_status", pd.Series("", index=replay_handoff.index)).astype(str).eq("READY_FOR_EXPLORATORY_LOCAL_REPLAY")
        replay_ready = replay_handoff.get("replay_status", pd.Series("", index=replay_handoff.index)).astype(str).eq("READY_FOR_COSTED_LOCAL_REPLAY")
        settings_missing = replay_handoff.get("settings_status", pd.Series("", index=replay_handoff.index)).astype(str).ne("CURRENT_PAIR_DETAIL_SETTINGS_CAPTURED")
        research_review = replay_handoff.get("research_quality_status", pd.Series("", index=replay_handoff.index)).astype(str).ne("RESEARCH_READY")
        economics_blocked = replay_handoff.get("replay_blockers", pd.Series("", index=replay_handoff.index)).astype(str).str.contains(
            "venue_cost_model_not_validated|venue_slippage_not_calibrated|venue_funding_or_borrow_not_validated",
            regex=True,
            na=False,
        )
        lines.append(f"- current exact-mode setups: {len(replay_handoff)}")
        lines.append(f"- ready for exploratory local replay: {int(exploratory_ready.sum())}")
        lines.append(f"- ready for costed local replay: {int(replay_ready.sum())}")
        lines.append(f"- settings captures still required: {int(settings_missing.sum())}")
        lines.append(f"- research-review setups: {int(research_review.sum())}")
        lines.append(f"- setups with unresolved venue economics: {int(economics_blocked.sum())}")
        lines.append("- exploratory replay uses explicit cost sensitivity; execution remains separately blocked")
    lines.extend(["", "## Mode-Matched Local Replay", ""])
    if mode_capability.empty:
        lines.append("- no mode capability board built yet")
    else:
        mode_ready = mode_capability.get(
            "mode_capability_status", pd.Series("", index=mode_capability.index)
        ).astype(str).eq("READY_FOR_RESEARCH_MODE_REPLAY")
        mode_settings_blocked = mode_capability.get(
            "mode_capability_status", pd.Series("", index=mode_capability.index)
        ).astype(str).eq("BLOCKED_SETTINGS_CAPTURE")
        mode_input_blocked = mode_capability.get(
            "mode_capability_status", pd.Series("", index=mode_capability.index)
        ).astype(str).eq("BLOCKED_MODE_INPUTS")
        lines.append(f"- exact-mode capability rows: {len(mode_capability)}")
        lines.append(f"- ready for local research formula replay: {int(mode_ready.sum())}")
        lines.append(f"- blocked awaiting settings capture: {int(mode_settings_blocked.sum())}")
        lines.append(f"- blocked on mode-specific inputs: {int(mode_input_blocked.sum())}")
        lines.append("- local formula results remain approximations; vendor custom-series proof is required for vendor-exact labeling")
    lines.extend(["", "## Cross-Mode Research Comparison", ""])
    if mode_comparison.empty:
        lines.append("- no valid confirmed mode-settings captures plus matching local history are available yet")
    else:
        comparison_groups = mode_comparison.get("comparison_group_id", pd.Series("", index=mode_comparison.index)).astype(str).nunique()
        research_ready = mode_comparison.get("mode_replay_status", pd.Series("", index=mode_comparison.index)).astype(str).eq("READY_FOR_RESEARCH_REPLAY")
        venue_mismatch = mode_comparison.get("history_match_status", pd.Series("", index=mode_comparison.index)).astype(str).str.contains("VENUE_MISMATCH|VENUE_UNKNOWN", regex=True, na=False)
        lines.append(f"- comparison groups: {comparison_groups}")
        lines.append(f"- provisional replay rows: {int(research_ready.sum())}")
        lines.append(f"- rows with venue mismatch or unknown venue: {int(venue_mismatch.sum())}")
        lines.append("- comparisons are research-only; no row can promote, paper, or execute a trade")
        lines.append("- same-venue point-in-time history, vendor custom-series parity, and validated venue economics remain acceptance requirements")
    lines.extend(["", "## Exploratory Cost Sensitivity", ""])
    if exploratory_cost_sensitivity.empty:
        lines.append("- no cost-sensitivity plan built yet")
    else:
        sensitivity_ready = exploratory_cost_sensitivity.get(
            "scenario_status", pd.Series("", index=exploratory_cost_sensitivity.index)
        ).astype(str).eq("READY_FOR_PROVISIONAL_COST_SENSITIVITY")
        profile_ids = sorted(
            {
                _text_value(value)
                for value in exploratory_cost_sensitivity.get(
                    "cost_profile_id", pd.Series("", index=exploratory_cost_sensitivity.index)
                )
                if _text_value(value)
            }
        )
        execution_risk_included = "execution_risk_bps" in exploratory_cost_sensitivity.columns
        lines.append(f"- planned sensitivity rows: {len(exploratory_cost_sensitivity)}")
        lines.append(f"- replay-ready sensitivity rows: {int(sensitivity_ready.sum())}")
        lines.append(f"- profiles: {', '.join(profile_ids) or 'unknown'}")
        lines.append(f"- execution-risk component included: {execution_risk_included}")
        lines.append("- every scenario is research-only; validated venue economics remain required for acceptance")
    lines.extend(["", "## All-Venue Research Lanes", ""])
    if multi_venue_readiness.empty:
        lines.append("- no current venue-history readiness rows")
    else:
        ready_to_fetch = multi_venue_readiness.get("readiness_status", pd.Series("", index=multi_venue_readiness.index)).astype(str).eq("ready_to_fetch")
        ready_for_replay = multi_venue_readiness.get("readiness_status", pd.Series("", index=multi_venue_readiness.index)).astype(str).eq("ready_for_replay")
        source_kinds = sorted({value for value in multi_venue_readiness.get("candidate_source_kind", pd.Series("", index=multi_venue_readiness.index)).map(_text_value) if value})
        lines.append(f"- current pair/mode rows: {len(multi_venue_readiness)}")
        lines.append(f"- venue histories ready to fetch: {int(ready_to_fetch.sum())}")
        lines.append(f"- local replay routes ready: {int(ready_for_replay.sum())}")
        lines.append(f"- candidate source: {', '.join(source_kinds) or 'legacy reference'}")
        lines.append("- venue readiness authorizes research collection only, never trade submission")
    lines.extend(["", "## Binance Research Lane", ""])
    if binance_readiness.empty:
        lines.append("- no Binance readiness rows")
    else:
        histories_ready = binance_readiness.get("history_status", pd.Series("", index=binance_readiness.index)).astype(str).eq("history_ready_needs_settings_and_cost_model")
        execution_blocked = binance_readiness.get("research_execution_status", pd.Series("", index=binance_readiness.index)).astype(str).eq("research_only_execution_blocked")
        lines.append(f"- current Binance pair/mode rows: {len(binance_readiness)}")
        lines.append(f"- daily histories ready for research: {int(histories_ready.sum())}")
        lines.append(f"- execution-blocked rows: {int(execution_blocked.sum())}")
    lines.append("- public spot history is research evidence only until costs, shortability, and slippage are calibrated")
    lines.extend(["", "## Venue Routing", ""])
    if route_recommendations is None or route_recommendations.empty:
        lines.append("- blocked: venue route scorecard missing")
    else:
        route_recommendations = _normalize_venue_route_recommendations(route_recommendations)
        execution_ready = route_recommendations.get("recommended_execution_venue", pd.Series("", index=route_recommendations.index)).map(_text_value).ne("")
        paper_ready = route_recommendations.get("recommended_paper_venue", pd.Series("", index=route_recommendations.index)).map(_text_value).ne("")
        lines.append(f"- pairs reviewed: {len(route_recommendations)}")
        lines.append(f"- evidence-complete execution routes: {int(execution_ready.sum())}")
        lines.append(f"- paper-submission-ready routes: {int(paper_ready.sum())}")
        lines.append("- authority: reports/active/venue_route_recommendations.csv")
    lines.extend(["", "## YouTube Research Brain", ""])
    youtube_status = _read_csv(root / "reports" / "agents" / "youtube_brain_status.csv")
    youtube_hypotheses = _read_csv(root / "reports" / "agents" / "youtube_brain_hypotheses.csv")
    youtube_replication = _read_csv(root / "reports" / "agents" / "youtube_brain_replication_scorecard.csv")
    if youtube_status.empty:
        lines.append("- blocked: YouTube research brain has not been built")
    else:
        row = youtube_status.iloc[0]
        lines.append(f"- status: {_text_value(row.get('status', ''))}")
        lines.append(f"- videos: {int(_clean_numeric(row.get('videos', 0)) or 0)}")
        lines.append(f"- claims: {int(_clean_numeric(row.get('claims', 0)) or 0)}")
        lines.append(f"- formulas: {int(_clean_numeric(row.get('formulas', 0)) or 0)}")
        lines.append(f"- current pair hypotheses: {len(youtube_hypotheses)}")
        supported = youtube_replication.get("lifecycle", pd.Series("", index=youtube_replication.index)).astype(str).isin(
            {"WALK_FORWARD_SUPPORTED", "TESTNET_SUPPORTED", "LIVE_VALIDATED"}
        )
        lines.append(f"- evidence-supported hypotheses: {int(supported.sum())}")
        lines.append("- authority: research prioritization only; no trade authorization")
    lines.extend(["", "## Model Gate", ""])
    if acceptance.empty:
        lines.append("- blocked: model acceptance missing")
    else:
        lines.append(f"- accepted: {bool(acceptance.get('accepted', pd.Series([False])).astype(bool).iloc[0])}")
    lines.extend(["", "## Base RL", ""])
    base_rl = _read_csv(root / "reports" / "rl" / "base_rl_paper_handoff_status.csv")
    if base_rl.empty:
        lines.append("- blocked: base RL handoff missing")
    else:
        row = base_rl.iloc[0]
        lines.append(f"- status: {row.get('status', '')}")
        lines.append(f"- blocker: {row.get('blocker', '') or 'none'}")
        lines.append(f"- paper authorized: {bool(row.get('paper_authorized', False))}")
    promotion = _read_csv(root / "reports" / "brain" / "promotion_ladder.csv")
    lines.extend(["", "## Current Setup Truth", ""])
    current_setup = pd.DataFrame()
    if not current.empty and "area" in current.columns:
        current_setup = current[current["area"].astype(str).isin(["paper_status", "orchestrator_status", "overall_readiness"])].copy()
    if not current_setup.empty:
        current_setup["_area_rank"] = current_setup["area"].astype(str).map({"paper_status": 0, "orchestrator_status": 1, "overall_readiness": 2}).fillna(9)
        current_setup = current_setup.sort_values(["_area_rank"])
        row = current_setup.iloc[0]
        lines.append(f"- pair: {_text_value(row.get('pair', ''))}")
        lines.append(f"- setup: {_text_value(row.get('setup_identity', '')) or _text_value(row.get('candidate_id', ''))}")
        lines.append(f"- stage: {_text_value(row.get('setup_status', '')) or _text_value(row.get('status', ''))}")
        lines.append(f"- blocker: {_text_value(row.get('setup_blocker', '')) or _text_value(row.get('blocker', '')) or 'none'}")
    elif promotion.empty:
        lines.append("- no setup-specific promotion truth yet")
    else:
        preferred = promotion.copy()
        preferred["_credible"] = preferred.get("paper_credible", pd.Series(dtype=object)).astype(bool)
        preferred["_stage_rank"] = preferred.get("promotion_stage", pd.Series(dtype=object)).astype(str).map(
            {
                "paper_eligible": 0,
                "orchestrator_review": 1,
                "candidate_validated": 2,
                "rl_interpretation": 3,
                "forward_walk_blocked": 4,
            }
        ).fillna(9)
        preferred["_primary"] = preferred.get("setup_role", pd.Series(dtype=object)).astype(str).eq("primary")
        preferred = preferred.sort_values(["_credible", "_stage_rank", "_primary", "pair"], ascending=[False, True, False, True])
        row = preferred.iloc[0]
        blocker = _text_value(row.get("blocker", ""))
        if not blocker and _text_value(row.get("paper_status", "")) == "research_only":
            blocker = _acceptance_status(root)
        lines.append(f"- pair: {_text_value(row.get('pair', ''))}")
        lines.append(f"- setup: {_text_value(row.get('setup_identity', '')) or _text_value(row.get('candidate_id', ''))}")
        lines.append(f"- stage: {_text_value(row.get('promotion_stage', ''))}")
        lines.append(f"- blocker: {blocker or 'none'}")
    comparison = _read_csv(root / "reports" / "rl" / "base_vs_augmented_pair_comparison.csv")
    lines.extend(["", "## Base RL Comparison", ""])
    if comparison.empty:
        lines.append("- no base vs augmented comparison yet")
    else:
        supported = comparison[comparison.get("agreement", pd.Series(dtype=bool)).astype(str).str.lower() == "true"].shape[0]
        total = len(comparison)
        lines.append(f"- pairs compared: {total}")
        lines.append(f"- agreements: {supported}")
    promotion = _read_csv(root / "reports" / "rl" / "base_rl_promotion_readiness.csv")
    lines.extend(["", "## Base RL Promotion", ""])
    if promotion.empty:
        lines.append("- no promotion readiness rows yet")
    else:
        ready = int(promotion[promotion.get("paper_eligible", pd.Series(dtype=bool)).astype(bool)].shape[0]) if "paper_eligible" in promotion.columns else 0
        lines.append(f"- promotion rows: {len(promotion)}")
        lines.append(f"- likely eligible rows: {ready}")
    lines.extend(["", "## Supreme Team", ""])
    team = _supreme_team_checkpoint_rows(root)
    if team.empty:
        lines.append("- latest checkpoint not yet run")
    else:
        rows = int(len(team))
        open_actions = int(team["open_actions"].astype(float).iloc[0]) if "open_actions" in team.columns else rows
        pass_gates = int(team["pass_gates"].astype(float).iloc[0]) if "pass_gates" in team.columns else 0
        lines.append(f"- open actions: {open_actions}")
        lines.append(f"- pass gates: {pass_gates}")
        lines.append(f"- rows: {rows}")
    lines.extend(["", "## Data Health", f"- rows: {len(data_health)}", "", "Blockers are intentionally visible. No stale row is actionable."])
    return "\n".join(lines) + "\n"


def _supreme_team_checkpoint_rows(root: Path) -> pd.DataFrame:
    latest = root / "reports" / "supreme_team" / "latest_supreme_team.md"
    team_index = _read_csv(root / "reports" / "supreme_team_index.csv")
    if team_index.empty:
        return pd.DataFrame([{"status": "missing", "evidence": latest.as_posix(), "run_id": "", "source": "reports/supreme_team_index.csv"}])
    latest_row = team_index.sort_values("timestamp_utc").iloc[-1]
    open_actions = int(latest_row.get("open_actions", 0) or 0)
    pass_gates = int(latest_row.get("pass_gates", 0) or 0)
    critical = int(latest_row.get("critical", 0) or 0)
    high = int(latest_row.get("high", 0) or 0)
    medium = int(latest_row.get("medium", 0) or 0)
    return pd.DataFrame(
        [
            {
                "status": "ready" if open_actions == 0 else "open_actions_pending",
                "run_id": str(latest_row.get("run_id", "")),
                "open_actions": open_actions,
                "pass_gates": pass_gates,
                "critical": critical,
                "high": high,
                "medium": medium,
                "evidence": str(latest),
            }
        ]
    )
