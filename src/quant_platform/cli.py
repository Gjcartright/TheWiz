from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from quant_platform.active_pipeline import (
    CommandResult,
    apify_cost_audit_rows,
    archive_from_index,
    build_artifact_index,
    build_command_dashboard,
    build_market_venue_context,
    build_multi_venue_history_readiness,
    build_pair_universe,
    build_trade_dataset,
    build_venue_lane_test_plan,
    build_venue_route_scorecard,
    current_state,
    export_trade_gate_model,
    focused_paper_validation_rows,
    paper_candidate_shortlist_rows,
    promote_trade_dataset,
    run_model_gated_backtest,
    system_check,
    train_trade_gate,
)
from quant_platform.api_extraction import (
    CryptoWizardsExtractor,
    CryptoWizardsFetchError,
    CryptoWizardsLiveConfig,
    parse_endpoint_specs,
)
from quant_platform.apify_sources import (
    infer_apify_venue,
    parse_apify_sources_from_mcp_url,
    refresh_apify_sources,
)
from quant_platform.backtest import CostModel, backtest_pair, backtest_two_leg_spread
from quant_platform.binance_spot import (
    build_binance_spot_lane_report,
    build_binance_spot_pair_history,
    fetch_binance_spot_candles,
)
from quant_platform.binance_testnet import (
    BinanceTestnetConfig,
    binance_testnet_pair_preflight,
    binance_testnet_preflight,
)
from quant_platform.crypto_wizards_catalog import endpoint_rows
from quant_platform.crypto_wizards_dashboard_capture import (
    ingest_exhaustive_wizard_dashboard_captures,
)
from quant_platform.crypto_wizards_history import (
    CryptoWizardsHistoryRequest,
    crawl_prescanned_backtest_histories,
    crawl_prescanned_zscores_histories,
    official_min5_request_rows,
    write_backtest_pair_payload,
    write_zscores_pair_payload,
)
from quant_platform.crypto_wizards_scanner import load_scanner_rows, write_scanner_reports
from quant_platform.crypto_wizards_sweep import (
    restore_complete_wizard_sweep_from_raw,
    run_authorized_wizard_discovery_sweep,
    run_wizard_discovery_sweep,
)
from quant_platform.dydx_candles import (
    archive_dydx_candles,
    backfill_provisional_pair_history_features,
    build_pair_history_from_candles,
    build_pair_history_from_windowed_candles,
    dydx_two_leg_request_rows,
    import_dydx_candle_bundle,
    load_loose_candle_payload,
)
from quant_platform.env import load_env_file
from quant_platform.execution import (
    DydxNetworkConfig,
    OrderIntent,
    SpreadOrderPlan,
    _lookup_dashboard_trade_snapshot,
    append_paper_outcome_record,
    append_paper_trading_record,
    block_paper_plan_for_execution_config,
    build_dydx_indexer_adapter,
    build_dydx_order_client_adapter,
    build_execution_venue,
    build_research_gated_paper_plan,
    build_venue_order_client_adapter,
    dydx_execution_compatibility_snapshot,
    dydx_readiness_report,
    effective_dydx_account_state_snapshot,
    hyperliquid_testnet_order_preflight_status,
    normalize_venue_name,
    paper_trading_record,
    refresh_current_paper_watch_positions,
    refresh_dydx_execution_compatibility_table,
    refresh_gmx_execution_compatibility_table,
    refresh_gmx_testnet_candidate_shortlist,
    refresh_gmx_testnet_market_inventory,
    refresh_hyperliquid_execution_compatibility_table,
    refresh_hyperliquid_testnet_candidate_shortlist,
    refresh_hyperliquid_testnet_market_inventory,
    refresh_injective_execution_compatibility_table,
    refresh_injective_mirror_candidate_queue,
    refresh_injective_spot_first_candidate_shortlist,
    refresh_injective_spot_supported_pair_universe,
    refresh_live_paper_trade_monitor,
    refresh_non_eth_route_submit_queue,
    refresh_paper_trade_decision_report,
    refresh_paper_trade_price_journal,
    submit_paper_plan,
    validate_dydx_order_client_adapter,
    validate_venue_order_client_adapter,
    venue_has_paper_adapter,
)
from quant_platform.experiments import (
    AcceptanceGate,
    ExperimentConfig,
    ExperimentHarness,
    PairDataset,
    strategy_acceptance_report,
)
from quant_platform.family_matrix import run_family_matrix
from quant_platform.field_registry import field_rows
from quant_platform.fixture_ingestion import (
    CANONICAL_ALIASES,
    datasets_from_fixtures,
    snake_case,
    write_fixture_field_dictionary,
)
from quant_platform.formula_registry import FORMULAS
from quant_platform.funding import (
    enrich_pair_dataset_with_funding,
    funding_coverage_for_pairs,
    funding_market_requirements,
    funding_rows_from_dydx_payload,
    normalize_funding_rows,
)
from quant_platform.hyperliquid import (
    DEFAULT_SLIPPAGE_CALIBRATION_CADENCE_MINUTES,
    DEFAULT_SLIPPAGE_CALIBRATION_MIN_SAMPLES,
    DEFAULT_SLIPPAGE_CALIBRATION_WINDOW_HOURS,
    build_hyperliquid_evidence_cadence,
    build_hyperliquid_lane_report,
    build_hyperliquid_pair_cost_model,
    build_hyperliquid_pair_history,
    build_hyperliquid_research_bundle,
    fetch_hyperliquid_candles,
    refresh_hyperliquid_execution_cost_snapshot,
    refresh_hyperliquid_funding_history,
    refresh_hyperliquid_market_context,
)
from quant_platform.hyperliquid_testnet import (
    HYPERLIQUID_TESTNET_EXECUTION_STATE_JSON,
    HyperliquidTestnetConfig,
    HyperliquidTestnetPairExecutor,
    write_hyperliquid_testnet_margin_snapshot,
    write_hyperliquid_testnet_preflight_report,
)
from quant_platform.math_v2_acceptance import build_math_v2_acceptance
from quant_platform.meta_learning import (
    JsonlTradeStore,
    TradeRecord,
    write_learning_event_summary_report,
)
from quant_platform.runtime_types import strict_bool
from quant_platform.ml_filter import (
    build_trade_filter_dataset,
    shadow_model_branch_comparison,
    shadow_trade_filter_predictions,
    train_trade_filter_walkforward,
)
from quant_platform.orchestration import run_langgraph_agent_workflow, run_orchestrator
from quant_platform.orchestration.corrective_agent_governance import (
    build_corrective_agent_governance,
)
from quant_platform.orchestration.corrective_artifact_retention import (
    run_corrective_artifact_retention,
)
from quant_platform.orchestration.corrective_canonical_status import (
    build_canonical_program_status,
)
from quant_platform.orchestration.corrective_daily_scheduler import build_corrective_daily_cadence
from quant_platform.orchestration.corrective_data_evidence import build_corrective_data_evidence
from quant_platform.orchestration.corrective_governance import build_corrective_governance
from quant_platform.orchestration.corrective_l2_scheduler import (
    install_corrective_l2_launch_agent,
    run_corrective_l2_capture,
)
from quant_platform.orchestration.corrective_live_canary import _policy_id
from quant_platform.orchestration.corrective_live_canary_execution import (
    run_live_canary_executor,
)
from quant_platform.orchestration.corrective_live_canary_executor import (
    build_live_canary_executor_preflight,
)
from quant_platform.orchestration.corrective_live_parity_capture import (
    capture_live_input_parity_evidence,
)
from quant_platform.orchestration.corrective_phase00_control import (
    build_phase00_quiesced_checkpoint,
    resume_phase00_maintenance,
    seal_phase00_active_artifact_lineage,
    start_phase00_maintenance,
)
from quant_platform.orchestration.corrective_phase00_descendants import (
    build_phase00_descendant_invalidation,
)
from quant_platform.orchestration.corrective_phase00_closure import (
    build_phase00_closure_verification,
)
from quant_platform.orchestration.corrective_program import complete_corrective_plan
from quant_platform.orchestration.corrective_registered_learning_protocol import (
    build_registered_stage5_protocol,
)
from quant_platform.orchestration.corrective_release_gates import build_corrective_release_gates
from quant_platform.orchestration.corrective_runtime import (
    atomic_copy_file,
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_text,
    build_canonical_scheduler_runtime_contract,
    promote_staged_file,
)
from quant_platform.orchestration.corrective_scheduler_lock import (
    GovernedEvidenceLockBusy,
    GovernedEvidenceMaintenanceActive,
    governed_evidence_write_lock,
)
from quant_platform.orchestration.corrective_scheduler_runtime_readiness import (
    build_corrective_scheduler_runtime_readiness,
)
from quant_platform.orchestration.corrective_stage4_handoff_readiness import (
    build_corrective_stage4_handoff_readiness,
)
from quant_platform.orchestration.corrective_statistical_remediation import (
    build_corrective_statistical_remediation,
)
from quant_platform.orchestration.corrective_testnet_collateral_transfer import (
    build_testnet_collateral_transfer_preflight,
    run_testnet_collateral_transfer,
)
from quant_platform.orchestration.corrective_testnet_pair_execution import (
    build_testnet_pair_execution_preflight,
    run_testnet_pair_execution,
)
from quant_platform.orchestration.corrective_wizard_api_credit_receipt import (
    capture_wizard_api_credit_receipt,
)
from quant_platform.orchestration.corrective_wizard_capture_manifest import (
    build_corrective_wizard_capture_manifest,
    validate_ou_v4_capture_manifest_contract,
    validate_ou_v5_capture_manifest_contract,
)
from quant_platform.orchestration.corrective_wizard_capture_reconciliation import (
    reconcile_corrective_wizard_capture_manifest,
)
from quant_platform.orchestration.corrective_wizard_comparator_review_control import (
    build_corrective_wizard_comparator_review_control,
)
from quant_platform.orchestration.corrective_wizard_copula_behavioral import (
    register_copula_behavioral_v2,
    run_current_copula_behavioral_proofs,
)
from quant_platform.orchestration.corrective_wizard_dynamic_holdout import (
    build_dynamic_v2_reviewed_activation,
)
from quant_platform.orchestration.corrective_wizard_dynamic_supreme_review import (
    build_dynamic_v2_supreme_review,
)
from quant_platform.orchestration.corrective_wizard_ou_holdout import (
    build_ou_v3_reviewed_activation,
    evaluate_ou_v2_blind_holdout,
    register_ou_trend_selector_v1_holdout,
    run_ou_v3_prospective_holdout,
)
from quant_platform.orchestration.corrective_wizard_ou_v4_failure_attribution import (
    build_ou_v4_failure_attribution,
)
from quant_platform.orchestration.corrective_wizard_ou_v4_holdout import (
    register_ou_v4_prospective_holdout,
    run_ou_v4_prospective_holdout,
)
from quant_platform.orchestration.corrective_wizard_ou_v4_supreme_review import (
    build_ou_v4_supreme_review,
)
from quant_platform.orchestration.corrective_wizard_ou_v5_failure_attribution import (
    build_ou_v5_failure_attribution,
)
from quant_platform.orchestration.corrective_wizard_ou_v5_holdout import (
    register_ou_v5_prospective_holdout,
    run_ou_v5_prospective_holdout,
)
from quant_platform.orchestration.corrective_wizard_ou_v5_supreme_review import (
    build_ou_v5_supreme_review,
)
from quant_platform.orchestration.corrective_wizard_ou_v6_holdout import (
    register_ou_v6_prospective_holdout,
)
from quant_platform.orchestration.corrective_wizard_ou_v6_supreme_review import (
    build_ou_v6_supreme_review,
)
from quant_platform.orchestration.corrective_wizard_ou_v6_terminal_closure import (
    build_ou_v6_terminal_closure,
)
from quant_platform.orchestration.corrective_wizard_parity import build_corrective_wizard_parity
from quant_platform.orchestration.corrective_wizard_reset_readiness import (
    build_corrective_wizard_reset_readiness,
)
from quant_platform.orchestration.current_wizard_hyperliquid_archive import (
    stage_current_wizard_hyperliquid_archive_copy,
)
from quant_platform.orchestration.current_wizard_hyperliquid_archive_release import (
    build_current_wizard_hyperliquid_archive_release_dry_run,
)
from quant_platform.orchestration.current_wizard_hyperliquid_cadence import (
    build_current_wizard_hyperliquid_operating_cadence,
)
from quant_platform.orchestration.current_wizard_hyperliquid_completion_audit import (
    build_current_wizard_hyperliquid_completion_audit,
)
from quant_platform.orchestration.current_wizard_hyperliquid_concentration import (
    build_current_wizard_hyperliquid_concentration,
)
from quant_platform.orchestration.current_wizard_hyperliquid_costs import (
    materialize_current_wizard_hyperliquid_cost_evidence,
)
from quant_platform.orchestration.current_wizard_hyperliquid_daily_runner import (
    run_current_wizard_hyperliquid_daily_pipeline,
)
from quant_platform.orchestration.current_wizard_hyperliquid_evidence_command_center import (
    build_current_wizard_hyperliquid_evidence_command_center,
)
from quant_platform.orchestration.current_wizard_hyperliquid_failure_attribution import (
    build_current_wizard_hyperliquid_failure_attribution,
    build_current_wizard_hyperliquid_failure_routing_index,
)
from quant_platform.orchestration.current_wizard_hyperliquid_handoff import (
    build_current_wizard_hyperliquid_handoff,
)
from quant_platform.orchestration.current_wizard_hyperliquid_learning import (
    build_current_wizard_hyperliquid_learning_ledger,
)
from quant_platform.orchestration.current_wizard_hyperliquid_leverage import (
    build_current_wizard_hyperliquid_leverage_surface,
)
from quant_platform.orchestration.current_wizard_hyperliquid_math_comparison import (
    reevaluate_current_wizard_hyperliquid_math,
)
from quant_platform.orchestration.current_wizard_hyperliquid_observed_replay import (
    run_current_wizard_hyperliquid_observed_cost_replay,
)
from quant_platform.orchestration.current_wizard_hyperliquid_regimes import (
    build_current_wizard_hyperliquid_regime_attribution,
)
from quant_platform.orchestration.current_wizard_hyperliquid_replay import (
    materialize_current_wizard_hyperliquid_history,
    run_current_wizard_hyperliquid_canonical_replay,
)
from quant_platform.orchestration.current_wizard_hyperliquid_robustness import (
    run_current_wizard_hyperliquid_robustness,
)
from quant_platform.orchestration.current_wizard_hyperliquid_storage import (
    build_current_wizard_hyperliquid_storage_reclamation_plan,
)
from quant_platform.orchestration.current_wizard_hyperliquid_testnet_protocol import (
    validate_current_wizard_hyperliquid_testnet_protocol,
)
from quant_platform.orchestration.current_wizard_hyperliquid_validation import (
    validate_current_wizard_hyperliquid_chain,
)
from quant_platform.orchestration.current_wizard_hyperliquid_walkforward import (
    run_current_wizard_hyperliquid_walkforward,
)
from quant_platform.orchestration.current_wizard_ou_optimal_overlay import (
    build_current_wizard_ou_optimal_overlay,
)
from quant_platform.orchestration.exhaustive_wizard_api_refresh import (
    build_exhaustive_wizard_api_refresh_delta,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_canonical_replay import (
    run_exhaustive_wizard_hyperliquid_canonical_replay,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_concentration import (
    build_exhaustive_wizard_hyperliquid_concentration,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_cost_bridge import (
    build_exhaustive_wizard_hyperliquid_cost_evidence,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_cost_evidence import (
    materialize_exhaustive_hyperliquid_funding_evidence,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_learning import (
    build_exhaustive_wizard_hyperliquid_learning_ledger,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_leverage import (
    build_exhaustive_wizard_hyperliquid_leverage_surface,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_observed_cost_replay import (
    run_exhaustive_wizard_hyperliquid_observed_cost_replay,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_regimes import (
    build_exhaustive_wizard_hyperliquid_regime_attribution,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_replay import (
    build_exhaustive_wizard_hyperliquid_replay_preflight,
    materialize_exhaustive_wizard_hyperliquid_history,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_robustness import (
    run_exhaustive_wizard_hyperliquid_robustness,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_run import (
    build_exhaustive_wizard_hyperliquid_mapping_refresh,
    build_exhaustive_wizard_hyperliquid_run,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_validation import (
    run_exhaustive_wizard_hyperliquid_validation,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_walkforward import (
    run_exhaustive_wizard_hyperliquid_walkforward,
)
from quant_platform.orchestration.hyperliquid_learning_and_risk import (
    build_testnet_lifecycle_gate,
    sign_testnet_smoke_approval,
    write_testnet_smoke_approval_template,
)
from quant_platform.orchestration.hyperliquid_research_cycle import run_hyperliquid_research_cycle
from quant_platform.orchestration.hyperliquid_research_validation import (
    build_hyperliquid_auxiliary_timeframe_validation,
)
from quant_platform.orchestration.hyperliquid_testnet_lifecycle_evidence import (
    capture_hyperliquid_testnet_lifecycle_evidence,
)
from quant_platform.orchestration.interactive_mixtape_graph import (
    build_interactive_mixtape_solution,
)
from quant_platform.orchestration.mini_agents import build_mini_agent_orchestration
from quant_platform.orchestration.orchestrator_assistant import build_orchestrator_assistant
from quant_platform.orchestration.specialist_scoreboard import build_specialist_scoreboard
from quant_platform.orchestration.versioned_learning_comparison import (
    build_versioned_learning_comparison,
)
from quant_platform.orchestration.wizard_pair_detail_api_pilot import (
    run_wizard_pair_detail_api_pilot,
)
from quant_platform.pair_detail_ingestion import (
    ECM_FIELD_SOURCE,
    PAIR_DETAIL_CAPTURE_AUDIT_COLUMNS,
    PAIR_DETAIL_CAPTURE_CHECKLIST_COLUMNS,
    PAIR_DETAIL_QUALITY_COLUMNS,
    datasets_from_pair_detail_snapshots,
    extract_history_rows,
    load_or_refresh_pair_detail_evidence_cache,
    load_pair_detail_snapshots,
    pair_detail_capture_audit,
    pair_detail_capture_checklist,
    pair_detail_history_coverage,
    pair_detail_payload_capture_audit,
    pair_detail_payload_capture_checklist,
    pair_detail_payload_history_coverage,
    pair_detail_quality_report,
    snapshot_from_payload,
    write_pair_detail_reports,
)
from quant_platform.pair_market_utils import normalize_dydx_market, pair_markets_from_pair
from quant_platform.regimes import RegimeConfig, classify_regimes, write_regime_dataset_report
from quant_platform.research_extract_ccxt import extract_research_knowledge
from quant_platform.research_extract_udemy import extract_udemy_research, refresh_udemy_research
from quant_platform.research_extract_youtube import extract_youtube_research
from quant_platform.research_ingestion import (
    build_research_source_registry,
    ingest_research_source,
    research_source_audit,
)
from quant_platform.research_knowledge_store import (
    build_research_knowledge_store,
    research_knowledge_summary,
)
from quant_platform.research_paper_math_audit import run_paper_critical_checks
from quant_platform.research_paper_reproduction import run_paper_reproduction_suite
from quant_platform.research_paper_reviews import build_paper_adversarial_reviews
from quant_platform.research_papers import ingest_paper_library, verify_paper_sources
from quant_platform.research_quantization import quantize_family_matrix
from quant_platform.rl import (
    base_rl_paper_handoff_report,
    build_brain_readiness_report,
    evaluate_base_rl,
    export_rl_policy,
    refresh_base_rl_feedback,
    run_augmented_rl,
    run_base_rl,
    run_brain_cycle,
    run_magicka_learning_cycle,
    run_rl_idea_scout,
    run_rl_research,
    run_sequential_thinking_magicka,
)
from quant_platform.rl.train_ppo import train_ppo_research_policy
from quant_platform.strategies import STRATEGIES, strategy_rows, zscore_signal
from quant_platform.trade_timing import (
    TRADE_TIMING_TEMPLATE_COLUMNS,
    load_trade_timing_history,
    trade_timing_comparison_report_frame,
    trade_timing_comparison_summary,
    write_trade_timing_template,
)
from quant_platform.v2_math_diagnostic import build_v2_math_diagnostic
from quant_platform.v2_run import build_v2_preflight_run, publish_v2_run_status, validate_v2_run
from quant_platform.wizard_control_plane import build_wizard_control_plane
from quant_platform.wizard_credit_budget import build_wizard_credit_budget_contract
from quant_platform.wizard_evidence import (
    build_wizard_diagnostic_confirmation,
    build_wizard_discovery_triage,
    build_wizard_evidence,
    build_wizard_exact_mode_capture_queue,
    build_wizard_exploratory_cost_sensitivity,
    build_wizard_hypotheses,
    build_wizard_local_parity,
    build_wizard_mode_matrix_capture_queue,
    build_wizard_mode_replay_capability,
    build_wizard_pair_detail_capture_queue,
    build_wizard_pair_settings_capture_template,
    build_wizard_replay_handoff,
    build_wizard_research_pack,
    import_wizard_pair_settings_capture,
)
from quant_platform.wizard_hyperliquid_bridge import build_hyperliquid_wizard_hypothesis_queue
from quant_platform.wizard_hyperliquid_mode_proof import (
    build_exhaustive_wizard_mode_proof_queue,
    refresh_activated_dynamic_v2_proofs,
    refresh_activated_ou_v3_proofs,
    refresh_activated_ou_v4_proofs,
    refresh_activated_ou_v5_proofs,
    refresh_activated_ou_v6_proofs,
    run_hyperliquid_wizard_mode_proofs,
)
from quant_platform.wizard_local_verification import (
    build_wizard_local_verification_batch,
    verify_wizard_local_mode,
)
from quant_platform.wizard_mode_comparison import build_wizard_mode_comparison
from quant_platform.wizard_ou_v4_comparator_activation import (
    build_reviewed_ou_v4_activation,
)
from quant_platform.wizard_ou_v5_comparator_activation import (
    build_reviewed_ou_v5_activation,
)
from quant_platform.wizard_ou_v6_comparator_activation import (
    build_reviewed_ou_v6_activation,
)
from quant_platform.wizard_pair_detail_ui_bundle import (
    ingest_wizard_pair_detail_ui_bundles,
)
from quant_platform.wizard_research_journal import build_wizard_research_journal
from quant_platform.yahoo_crypto import (
    backfill_yahoo_crypto_funding,
    build_yahoo_crypto_lane_report,
    build_yahoo_crypto_pair_history,
    fetch_yahoo_crypto_candles,
    refresh_yahoo_crypto_pair_history,
    refresh_yahoo_research_candidates,
)
from quant_platform.youtube_brain import (
    build_youtube_brain,
    build_youtube_brain_dashboard,
    build_youtube_pair_hypotheses,
    refresh_youtube_collection,
    refresh_youtube_outcome_memory,
    run_youtube_brain_cycle,
)
from quant_platform.youtube_caption_insights import build_youtube_caption_insights
from quant_platform.youtube_channel_research import run_hudson_thames_youtube_research
from quant_platform.youtube_hypothesis_validation import run_youtube_hypothesis_validation

ROOT = Path(__file__).resolve().parents[2]
SWEEP_MODES = ("light", "deep", "paid")
PROJECT_OBJECTIVE_PATH = ROOT / "project_objective.md"
NON_DYDX_ENRICHMENT_SOURCES = ("hyperliquid", "gmx", "dexscreener")
LEARNING_OUTCOME_TEMPLATE_COLUMNS = [
    "trade_id",
    "pair",
    "strategy_id",
    "realized_return",
    "signal",
    "hedge_ratio",
    "beta",
    "notional_usd",
    "regime",
]
TRADE_TIMING_DEFAULT_TEMPLATE = ROOT / "data" / "meta_learning" / "trade_timing_template.csv"
DEFAULT_INDEXER_BASE = os.getenv("QPA_INDEXER_BASE", "https://indexer.dydx.trade").strip()


@contextmanager
def _local_env_for_reports(root: Path | None = None):
    effective_root = root or ROOT
    loaded = load_env_file(effective_root / ".env.local", override=False)
    try:
        yield
    finally:
        # Report helpers may be called in-process by notebooks or tests. Do not
        # let values read only for one report become hidden global state.
        for key, value in loaded.items():
            if os.environ.get(key) == value:
                os.environ.pop(key, None)


DEFAULT_FAMILY_SWEEP_PAIRS = (
    "BTC-USD-SOL-USD",
    "DOGE-USD-SOL-USD",
    "SOL-USD-XRP-USD",
    "SOL-USD-LINK-USD",
)


def _acceptance_report_path(root: Path | None = None) -> Path:
    effective_root = root or ROOT
    configured = os.getenv("QPA_ACCEPTANCE_REPORT_PATH", "").strip()
    if not configured:
        return effective_root / "reports" / "acceptance_report.csv"
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = effective_root / path
    return path


def _native_acceptance_bridge_row(reports: Path) -> dict[str, object] | None:
    packets = _read_csv_or_empty(reports / "brain" / "native_candidate_packets.csv")
    forward = _read_csv_or_empty(reports / "brain" / "native_forward_walk.csv")
    if packets.empty or forward.empty:
        return None

    packets = packets.copy()
    packets["candidate_id"] = packets.get("candidate_id", pd.Series(dtype=object)).astype(str)
    packets["venue"] = packets.get("venue", pd.Series(dtype=object)).astype(str).str.lower()
    packets["blocker_state"] = (
        packets.get("blocker_state", pd.Series(dtype=object)).fillna("").astype(str)
    )
    eligible_packets = packets[packets["venue"].eq("dydx") & packets["blocker_state"].eq("")].copy()
    if eligible_packets.empty:
        return None

    forward = forward.copy()
    forward["candidate_id"] = forward.get("candidate_id", pd.Series(dtype=object)).astype(str)
    forward["forward_walk_status"] = (
        forward.get("forward_walk_status", pd.Series(dtype=object)).astype(str).str.lower()
    )
    merged = eligible_packets.merge(
        forward, on="candidate_id", how="inner", suffixes=("_packet", "_fw")
    )
    if merged.empty:
        return None

    merged["oos_sharpe"] = pd.to_numeric(
        merged.get("oos_sharpe", pd.Series(dtype=float)), errors="coerce"
    )
    merged["oos_profit_factor"] = pd.to_numeric(
        merged.get("oos_profit_factor", pd.Series(dtype=float)), errors="coerce"
    )
    merged["oos_max_drawdown"] = pd.to_numeric(
        merged.get("oos_max_drawdown", pd.Series(dtype=float)), errors="coerce"
    )
    merged["oos_trade_count"] = pd.to_numeric(
        merged.get("oos_trade_count", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)

    passing = merged[merged["forward_walk_status"].eq("pass")].copy()
    pairs_tested = (
        int(merged.get("pair", pd.Series(dtype=object)).astype(str).nunique())
        if "pair" in merged.columns
        else 0
    )
    passing_pairs = (
        int(passing.get("pair", pd.Series(dtype=object)).astype(str).nunique())
        if not passing.empty
        else 0
    )
    total_trades = int(passing["oos_trade_count"].sum()) if not passing.empty else 0
    median_pf = float(passing["oos_profit_factor"].median()) if not passing.empty else 0.0
    median_sharpe = float(passing["oos_sharpe"].median()) if not passing.empty else 0.0
    worst_drawdown = float(passing["oos_max_drawdown"].max()) if not passing.empty else 0.0

    production_eligible = passing_pairs >= 2
    preferred_failures: list[str] = []
    if not production_eligible:
        preferred_failures.append("not_production_eligible")
    if median_pf < 1.25:
        preferred_failures.append("median_profit_factor<1.25")
    if median_sharpe < 0.15:
        preferred_failures.append("median_sharpe<0.15")
    if worst_drawdown > 0.35:
        preferred_failures.append("worst_drawdown>0.35")
    if total_trades < 100:
        preferred_failures.append("total_trades<100")

    top_pairs = (
        ",".join(passing.get("pair", pd.Series(dtype=object)).astype(str).head(5).tolist())
        if not passing.empty
        else ""
    )
    concern = []
    if passing_pairs < 2:
        concern.append("thin_pair_support")
    if worst_drawdown > 0.35:
        concern.append("elevated_drawdown")
    if median_sharpe < 0.15:
        concern.append("fragile_sharpe")

    return {
        "strategy_id": 9001,
        "strategy_name": "Native Local Math",
        "family": "native",
        "production_eligible": production_eligible,
        "preferred_eligible": not preferred_failures,
        "research_eligible": passing_pairs >= 1,
        "research_tier": "research_ready"
        if production_eligible
        else ("research_watch" if passing_pairs >= 1 else "research_explore"),
        "acceptance_reason": "passed" if production_eligible else "passing_pairs<2",
        "preferred_reason": "passed" if not preferred_failures else ";".join(preferred_failures),
        "research_reason": "passed" if passing_pairs >= 1 else "research_passing_pairs<1",
        "evaluated_runs": len(merged),
        "passing_runs": len(passing),
        "pairs_tested": pairs_tested,
        "passing_pairs": passing_pairs,
        "research_pairs_tested": pairs_tested,
        "research_passing_pairs": passing_pairs,
        "two_leg_pairs_tested": pairs_tested,
        "two_leg_execution_input_pairs": pairs_tested,
        "two_leg_passing_pairs": passing_pairs,
        "research_score": round(
            min(100.0, 40.0 + passing_pairs * 10.0 + median_pf * 10.0 + median_sharpe * 20.0), 2
        ),
        "research_conviction_score": round(min(100.0, median_pf * 25.0), 2),
        "research_stability_score": round(max(0.0, 100.0 - worst_drawdown * 100.0), 2),
        "research_breadth_score": round(min(100.0, pairs_tested * 25.0), 2),
        "research_strengths": "native_forward_walk_passes" if passing_pairs >= 1 else "",
        "research_concerns": ",".join(concern),
        "research_setup_mix": "native_dydx_forward_walk",
        "research_top_pairs": top_pairs,
        "research_next_step": "promote_native_candidates_into_strategy_acceptance"
        if production_eligible
        else "expand_native_pair_support",
        "required_cost_buckets": "base;stress",
        "required_backtest_mode": "two_leg",
        "required_two_leg_inputs": "price_x;price_y;hedge_ratio;beta;funding_x;funding_y",
        "total_trades": total_trades,
        "median_profit_factor": round(median_pf, 6),
        "median_sharpe": round(median_sharpe, 6),
        "worst_drawdown": round(worst_drawdown, 6),
    }


def _augmented_acceptance_frame(
    reports: Path | None = None,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    effective_root = root or ROOT
    reports_dir = reports or (effective_root / "reports")
    acceptance = _read_csv_or_empty(_acceptance_report_path(effective_root)).copy()
    native_row = _native_acceptance_bridge_row(reports_dir)
    if native_row is None:
        return acceptance
    if acceptance.empty:
        return pd.DataFrame([native_row])
    acceptance = acceptance[
        acceptance.get("strategy_id", pd.Series(dtype=object)).astype(str)
        != str(native_row["strategy_id"])
    ].copy()
    return pd.concat([acceptance, pd.DataFrame([native_row])], ignore_index=True)


def _project_objective_snippet(max_chars: int = 1200) -> str:
    if not PROJECT_OBJECTIVE_PATH.exists():
        return f"missing_project_objective={PROJECT_OBJECTIVE_PATH}"
    text = PROJECT_OBJECTIVE_PATH.read_text(encoding="utf-8").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def _project_objective_spine_status() -> tuple[str, str]:
    if not PROJECT_OBJECTIVE_PATH.exists():
        return "blocked", f"project_objective_missing={PROJECT_OBJECTIVE_PATH}"
    return "completed", f"project_objective_loaded={PROJECT_OBJECTIVE_PATH}"


def _parse_pair_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    items = [item.strip() for item in re.split(r"[,\n;]+", value) if item.strip()]
    return tuple(items)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _raw_root() -> Path:
    return ROOT / "data" / "raw"


def _normalized_root() -> Path:
    return ROOT / "data" / "normalized"


def _dydx_inbox_dir() -> Path:
    return _raw_root() / "dydx_inbox"


def _dydx_manual_dir() -> Path:
    return _raw_root() / "dydx_manual"


def _enrichment_raw_root() -> Path:
    return _raw_root() / "enrichment"


def _canonical_source_name(source: str | None) -> str:
    normalized = str(source or "").strip().lower()
    if not normalized:
        return ""
    aliases = {
        "dydx": "dydx",
        "hyperliquid": "hyperliquid",
        "gmx": "gmx",
        "dexscreener": "dexscreener",
    }
    return aliases.get(normalized, normalized)


def _normalized_enrichment_dir(source: str) -> Path:
    canonical = _canonical_source_name(source)
    if canonical not in NON_DYDX_ENRICHMENT_SOURCES:
        raise SystemExit(f"unsupported enrichment source: {source}")
    return _normalized_root() / canonical


def _raw_enrichment_dir(source: str) -> Path:
    canonical = _canonical_source_name(source)
    if canonical not in NON_DYDX_ENRICHMENT_SOURCES:
        raise SystemExit(f"unsupported enrichment source: {source}")
    return _enrichment_raw_root() / canonical


def _assert_dydx_raw_target(path: Path, source: str | None) -> None:
    canonical = _canonical_source_name(source)
    resolved = path.resolve()
    dydx_dirs = (_dydx_inbox_dir(), _dydx_manual_dir())
    if any(
        _is_relative_to(resolved, folder) or resolved == folder.resolve() for folder in dydx_dirs
    ):
        if canonical != "dydx":
            raise SystemExit(
                f"only dYdX actors may write into {_dydx_inbox_dir()} or {_dydx_manual_dir()}; "
                f"use data/raw/enrichment/<source> for {source or 'non-dydx'} payloads"
            )


def _assert_fixture_experiment_input_normalized(input_dir: Path) -> None:
    resolved = input_dir.resolve()
    enrichment_root = _enrichment_raw_root()
    raw_root = _raw_root()
    if _is_relative_to(resolved, enrichment_root) or resolved == enrichment_root.resolve():
        raise SystemExit(
            f"raw enrichment feeds may not go directly into experiments: {input_dir}. "
            "Run normalize-enrichment-fixtures first."
        )
    if resolved == raw_root.resolve():
        raise SystemExit(
            f"raw root may mix dYdX and enrichment payloads: {input_dir}. "
            "Use a normalized enrichment folder or the dYdX pair-detail path."
        )
    if _is_relative_to(resolved, _dydx_inbox_dir()) or _is_relative_to(
        resolved, _dydx_manual_dir()
    ):
        raise SystemExit(
            f"dYdX raw folders are not fixture experiment inputs: {input_dir}. "
            "Use the dYdX pair-detail build path instead."
        )


def _dns_fallback_ip_candidates(hostname: str) -> list[str]:
    ip_candidates: list[str] = []
    if not hostname:
        return ip_candidates

    explicit_map = os.getenv("QPA_INDEXER_HOST_IP_HINTS", "").strip()
    if explicit_map:
        # format:
        #   host:ip1,ip2;other-host:ip3
        # or newline/comma separated host:ip strings for convenience.
        host_entries = re.split(r"[;\n]", explicit_map)
        for entry in host_entries:
            if not entry.strip():
                continue
            if ":" not in entry:
                continue
            mapped_host, mapped_ips = entry.split(":", 1)
            if mapped_host.strip() == hostname:
                ip_candidates.extend([ip.strip() for ip in mapped_ips.split(",") if ip.strip()])

    known = {
        # Keep legacy fallback for mainnet only; testnet DNS is currently reachable
        # in this environment and should be used directly when available.
        "indexer.dydx.trade": ["172.66.166.30", "104.20.40.161"],
        "indexer.v4testnet.dydx.exchange": ["104.18.24.136"],
    }
    ip_candidates.extend(known.get(hostname, []))

    deduped: list[str] = []
    for ip in ip_candidates:
        if ip not in deduped:
            deduped.append(ip)
    return deduped


def _normalize_indexer_base(indexer_base: str, default_scheme: str = "https") -> str:
    raw_base = indexer_base.strip().rstrip("/")
    if not raw_base:
        return raw_base
    parsed = urlparse(raw_base)
    if parsed.scheme in {"http", "https"}:
        return raw_base
    if "://" in raw_base:
        return raw_base
    return f"{default_scheme}://{raw_base}"


def _candidate_curl_paths() -> list[str]:
    candidates = [
        "/opt/anaconda3/bin/curl",
        "/usr/bin/curl",
        "curl",
    ]
    valid: list[str] = []
    for path in candidates:
        if "/" in path and shutil.which(path):
            valid.append(path)
        elif path == "curl" and shutil.which(path):
            valid.append(shutil.which(path) or path)
    return valid


def _request_variants_for_url(url: str) -> list[tuple[str, dict[str, str], bool]]:
    """Build resilient requests.get variants for a single URL.

    Returns (url, headers, verify_ssl) tuples.
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return [(url, {"Content-Type": "application/json"}, True)]

    base_headers = {"Content-Type": "application/json"}
    host = parsed.hostname or ""
    variants: list[tuple[str, dict[str, str], bool]] = [(url, base_headers, True)]

    if not host:
        return variants

    for ip in _dns_fallback_ip_candidates(host):
        alt = parsed._replace(netloc=f"{ip}:{parsed.port}" if parsed.port else ip)
        alt_url = alt.geturl()
        headers = {**base_headers, "Host": host}
        if (alt_url, tuple(sorted(headers.items())), True) not in {
            (u, tuple(sorted(h.items())), v) for u, h, v in variants
        }:
            variants.append((alt_url, headers, True))
        if parsed.scheme == "https":
            if (alt_url, tuple(sorted(headers.items())), False) not in {
                (u, tuple(sorted(h.items())), v) for u, h, v in variants
            }:
                variants.append((alt_url, headers, False))

    return variants


def _indexer_base_candidates(indexer_base: str) -> list[str]:
    raw = os.getenv("QPA_INDEXER_BASES", "").strip()
    if indexer_base:
        if raw:
            raw = f"{raw},{indexer_base}"
        else:
            raw = indexer_base
    if not raw:
        return [DEFAULT_INDEXER_BASE]
    bases: list[str] = []
    for entry in re.split(r"[,\n;]", raw):
        base = _normalize_indexer_base(entry.strip())
        if base:
            bases.append(base)
    if not bases:
        return [_normalize_indexer_base(DEFAULT_INDEXER_BASE)]
    return list(dict.fromkeys(bases))


def _indexer_url_variants(url: str, indexer_bases: list[str]) -> list[str]:
    parsed = urlparse(url)
    if not (parsed.path and parsed.scheme and parsed.netloc):
        return [url]
    path_query = parsed.path
    if parsed.query:
        path_query += f"?{parsed.query}"
    if not indexer_bases:
        return [url]
    variants: list[str] = []
    if parsed.netloc:
        variants.append(url)
    for base in indexer_bases:
        parsed_base = urlparse(base)
        if not (parsed_base.scheme and parsed_base.netloc):
            continue
        candidate = f"{base}{path_query}"
        variants.append(candidate)
    deduped: list[str] = []
    for candidate in variants:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped or [url]


def _fetch_url_scheme_variants(url: str, forced_scheme: str | None = None) -> list[str]:
    forced_scheme = (forced_scheme or os.getenv("QPA_INDEXER_SCHEME", "")).strip().lower()
    if forced_scheme in {"http", "https"}:
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            return [parsed._replace(scheme=forced_scheme).geturl()]
        return [url]
    if os.getenv("QPA_DISABLE_SCHEME_FALLBACK", "").lower() in {"1", "true", "yes"}:
        return [url]
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return [url]
    variants: list[str] = []
    for scheme in (parsed.scheme, "https" if parsed.scheme == "http" else "http"):
        parsed = parsed._replace(scheme=scheme)
        candidate = parsed.geturl()
        if candidate not in variants:
            variants.append(candidate)
    return variants


def _resolve_host_via_doh(hostname: str) -> str | None:
    if not hostname:
        return None

    servers = [
        ("https://1.1.1.1/dns-query", {"accept": "application/dns-json"}),
        ("https://dns.google/resolve", {}),
    ]
    for endpoint, headers in servers:
        try:
            response = requests.get(
                endpoint,
                headers={**({"accept": "application/dns-json"} if headers else {})},
                params={"name": hostname, "type": "A"},
                timeout=5.0,
            )
            response.raise_for_status()
            for answer in response.json().get("Answer", []):
                if answer.get("type") == 1 and isinstance(answer.get("data"), str):
                    return answer["data"]
        except requests.exceptions.RequestException:
            continue
    return None


def _project_objective_runbook_lines() -> list[str]:
    if not PROJECT_OBJECTIVE_PATH.exists():
        return [
            "## Project Objective",
            "",
            f"Objective file missing: `{PROJECT_OBJECTIVE_PATH}`",
            "Add this file and rerun `priority-runbook` to include it in the spine narrative.",
            "",
        ]
    return [
        "## Project Objective",
        "",
        f"Source: `{PROJECT_OBJECTIVE_PATH}`",
        "```text",
        _project_objective_snippet(),
        "```",
        "",
    ]


def _read_positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    try:
        parsed = int(str(value).strip())
    except ValueError:
        return default
    return max(parsed, 1)


def _read_positive_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return max(parsed, 0.0)


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _apply_indexer_scheme_env(scheme: str | None) -> None:
    if not scheme:
        return
    normalized = scheme.strip().lower()
    if normalized and normalized not in {"http", "https"}:
        raise SystemExit("--indexer-scheme must be either http or https")
    os.environ["QPA_INDEXER_SCHEME"] = normalized


def _indexer_base_with_scheme(indexer_base: str, indexer_scheme: str | None) -> str:
    indexer_base = _normalize_indexer_base(indexer_base)
    normalized = str(indexer_scheme or "").strip().lower()
    if normalized not in {"http", "https"}:
        return indexer_base
    parsed = urlparse(indexer_base)
    if parsed.scheme not in {"http", "https"}:
        return indexer_base
    return parsed._replace(scheme=normalized).geturl()


def _acceptance_gate_from_env(base_gate: AcceptanceGate | None = None) -> AcceptanceGate:
    base = base_gate or AcceptanceGate()
    return AcceptanceGate(
        min_profit_factor=_read_positive_float_env("QPA_MIN_PROFIT_FACTOR", base.min_profit_factor),
        preferred_profit_factor=_read_positive_float_env(
            "QPA_PREFERRED_PROFIT_FACTOR", base.preferred_profit_factor
        ),
        min_sharpe=_read_positive_float_env("QPA_MIN_SHARPE", base.min_sharpe),
        preferred_sharpe=_read_positive_float_env("QPA_PREFERRED_SHARPE", base.preferred_sharpe),
        max_drawdown=_read_positive_float_env("QPA_MAX_DRAWDOWN", base.max_drawdown),
        preferred_max_drawdown=_read_positive_float_env(
            "QPA_PREFERRED_MAX_DRAWDOWN", base.preferred_max_drawdown
        ),
        min_trades=_read_positive_int_env("QPA_MIN_TRADES", base.min_trades),
        preferred_trades=_read_positive_int_env("QPA_PREFERRED_TRADES", base.preferred_trades),
        min_pairs=_read_positive_int_env("QPA_MIN_PAIRS", base.min_pairs),
        required_cost_buckets=tuple(
            bucket.strip().lower()
            for bucket in os.getenv(
                "QPA_REQUIRED_COST_BUCKETS", ",".join(base.required_cost_buckets)
            ).split(",")
            if bucket.strip()
        )
        or base.required_cost_buckets,
        required_regime=os.getenv("QPA_REQUIRED_REGIME", base.required_regime).strip()
        or base.required_regime,
        require_positive_expectancy=_read_bool_env(
            "QPA_REQUIRE_POSITIVE_EXPECTANCY", base.require_positive_expectancy
        ),
        require_two_leg_backtests=_read_bool_env(
            "QPA_REQUIRE_TWO_LEG_BACKTESTS", base.require_two_leg_backtests
        ),
        require_two_leg_execution_inputs=_read_bool_env(
            "QPA_REQUIRE_TWO_LEG_EXECUTION_INPUTS",
            base.require_two_leg_execution_inputs,
        ),
    )


def _experiment_harness(
    *,
    min_rows: int | None = None,
    gate: AcceptanceGate | None = None,
    strategy_ids: tuple[int, ...] | None = None,
) -> ExperimentHarness:
    base = ExperimentConfig()
    return ExperimentHarness(
        config=ExperimentConfig(
            cost_buckets=base.cost_buckets,
            min_rows=min_rows if min_rows is not None else base.min_rows,
            include_overall_regime=base.include_overall_regime,
            regime_column=base.regime_column,
            gate=gate or _acceptance_gate_from_env(base.gate),
        ),
        strategies=tuple(
            spec for spec in STRATEGIES if strategy_ids is None or spec.id in strategy_ids
        ),
    )


LEARNING_OUTCOME_REQUIRED_COLUMNS = ["pair", "strategy_id", "realized_return"]
FUNDING_TEMPLATE_COLUMNS = ["market", "timestamp", "funding_bps"]
FUNDING_TEMPLATE_REQUIRED_COLUMNS = ["market", "funding_bps"]
DEFAULT_DYDX_EXPANSION_PAIRS = (
    ("BTC-USD", "ETH-USD"),
    ("BTC-USD", "SOL-USD"),
    ("ETH-USD", "SOL-USD"),
    ("ETH-USD", "AVAX-USD"),
    ("ETH-USD", "LINK-USD"),
    ("SOL-USD", "AVAX-USD"),
    ("SOL-USD", "LINK-USD"),
    ("BTC-USD", "AVAX-USD"),
    ("BTC-USD", "LINK-USD"),
    ("AAVE-USD", "UNI-USD"),
    ("ARB-USD", "OP-USD"),
    ("MATIC-USD", "ARB-USD"),
    ("DOGE-USD", "XRP-USD"),
    ("DOGE-USD", "LTC-USD"),
    ("ETH-USD", "MKR-USD"),
)
DYDX_PAIR_EXPANSION_HUNT_PATH = ROOT / "reports" / "dydx_pair_expansion_plan_hunt.csv"
DYDX_LIVE_MARKET_SELECTOR_CUSTOM_PATH = ROOT / "reports" / "dydx_live_market_selector_custom.csv"
DEFAULT_DYDX_LIVE_SELECTOR_ANCHORS = ("BTC-USD", "ETH-USD", "SOL-USD")
DEFAULT_DYDX_LIVE_SELECTOR_EXCLUDED_MARKETS = {
    "DAI-USD",
    "EUR-USD",
    "EURC-USD",
    "PAXG-USD",
    "WTI-USD",
}


def _write_csv_atomic(frame: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    promote_staged_file(tmp, output)
    return output


def build_dictionaries() -> None:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    atomic_write_csv(pd.DataFrame(field_rows()), docs / "field_dictionary.csv", index=False)
    atomic_write_csv(pd.DataFrame(strategy_rows()), docs / "strategy_registry.csv", index=False)
    atomic_write_csv(pd.DataFrame(endpoint_rows()), docs / "crypto_wizards_endpoint_catalog.csv", index=False)

    formula_lines = ["# Formula Dictionary", ""]
    for name, info in FORMULAS.items():
        formula_lines.extend(
            [
                f"## {name}",
                f"- Formula: {info['formula']}",
                f"- Market interpretation: {info['interpretation']}",
                f"- Use case: {info['use_case']}",
                f"- Failure mode: {info['failure_mode']}",
                "",
            ]
        )
    atomic_write_text(docs / "formula_dictionary.md", "\n".join(formula_lines), encoding="utf-8")

    brain_lines = ["# Quant Brain", ""]
    for row in field_rows():
        formula = FORMULAS.get(row["name"], {})
        brain_lines.extend(
            [
                f"## {row['name']}",
                f"- Measures: {row['description']}",
                f"- Why it exists: {formula.get('interpretation', 'Research field from API; validate empirically.')}",
                f"- How it may create edge: {formula.get('use_case', 'Only if walk-forward tests show incremental predictive power.')}",
                f"- When it fails: {formula.get('failure_mode', 'Unknown until tested across regimes.')}",
                f"- Research role: {row['role']}",
                f"- Required tests: {row['required_tests']}",
                "",
            ]
        )
    atomic_write_text(docs / "quant_brain.md", "\n".join(brain_lines), encoding="utf-8")


def run_demo_backtest() -> None:
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    n = 1200
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.96 * spread[i - 1] + rng.normal(0, 0.02)
    frame = pd.DataFrame({"spread": spread})
    frame["zscore"] = (frame["spread"] - frame["spread"].rolling(7).mean()) / frame[
        "spread"
    ].rolling(7).std()
    frame = frame.dropna().reset_index(drop=True)
    result = backtest_pair(frame, zscore_signal(frame), CostModel())
    atomic_write_csv(pd.DataFrame([result.__dict__]), reports / "demo_backtest.csv", index=False)
    print(result)


def _demo_pair_frame(seed: int, n: int, phi: float, noise: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = phi * spread[i - 1] + rng.normal(0, noise)
    frame = pd.DataFrame({"spread": spread})
    frame["zscore"] = (frame["spread"] - frame["spread"].rolling(7).mean()) / frame[
        "spread"
    ].rolling(7).std()
    frame["conditional_probability_distortion"] = np.tanh(frame["zscore"].fillna(0.0) / 3.0)
    return classify_regimes(
        frame.dropna().reset_index(drop=True), RegimeConfig(lookback=40, trend_threshold=0.03)
    )


def run_demo_experiments() -> None:
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    datasets = [
        PairDataset("ETH-BTC", _demo_pair_frame(seed=11, n=1600, phi=0.94, noise=0.02)),
        PairDataset("SOL-ETH", _demo_pair_frame(seed=17, n=1600, phi=0.97, noise=0.025)),
    ]
    write_regime_dataset_report(datasets, reports / "regime_dataset_report.csv")
    harness = ExperimentHarness()
    results = harness.run(datasets)
    paths = harness.write_reports(results, reports)
    print(f"wrote {len(results)} experiment rows")
    for name, path in paths.items():
        print(f"{name}: {path}")


def ingest_fixtures(input_dir: Path | None = None) -> None:
    input_dir = input_dir or ROOT / "data" / "raw"
    docs = ROOT / "docs"
    reports = ROOT / "reports"
    docs.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    field_path = write_fixture_field_dictionary(
        input_dir, docs / "crypto_wizards_fixture_field_dictionary.csv"
    )
    datasets = datasets_from_fixtures(input_dir)
    datasets = [
        PairDataset(
            dataset.pair, classify_regimes(dataset.frame, RegimeConfig(preserve_existing=True))
        )
        for dataset in datasets
    ]
    write_regime_dataset_report(datasets, reports / "regime_dataset_report.csv")
    atomic_write_csv(pd.DataFrame(
        [
            {
                "pair": dataset.pair,
                "rows": len(dataset.frame),
                "columns": ";".join(dataset.frame.columns),
            }
            for dataset in datasets
        ]
    ), reports / "fixture_ingestion_summary.csv", index=False)
    print(f"field_dictionary: {field_path}")
    print(f"datasets: {len(datasets)}")


def normalize_enrichment_fixtures(
    source: str,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    canonical = _canonical_source_name(source)
    if canonical not in NON_DYDX_ENRICHMENT_SOURCES:
        raise SystemExit(
            f"normalize-enrichment-fixtures requires one of: {', '.join(NON_DYDX_ENRICHMENT_SOURCES)}"
        )
    source_dir = input_dir or _raw_enrichment_dir(canonical)
    _assert_dydx_raw_target(source_dir, canonical)
    datasets = datasets_from_fixtures(source_dir)
    if not datasets:
        raise SystemExit(f"no experiment-ready fixture datasets found in {source_dir}")
    output_base = output_dir or _normalized_enrichment_dir(canonical)
    _assert_dydx_raw_target(output_base, canonical)
    output_base.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [dataset.frame.assign(pair=dataset.pair, source=canonical) for dataset in datasets],
        ignore_index=True,
    )
    output_path = output_base / f"{canonical}_normalized_pairs.csv"
    atomic_write_csv(combined, output_path, index=False)
    report = pd.DataFrame(
        [
            {
                "source": canonical,
                "input_dir": str(source_dir),
                "output_path": str(output_path),
                "pairs": combined["pair"].nunique() if "pair" in combined.columns else 0,
                "rows": len(combined),
                "status": "normalized",
            }
        ]
    )
    _write_csv_atomic(report, ROOT / "reports" / f"{canonical}_normalization_report.csv")
    return output_path


def run_fixture_experiments(
    input_dir: Path | None = None, funding_path: Path | None = None
) -> None:
    input_dir = input_dir or _normalized_root()
    _assert_fixture_experiment_input_normalized(input_dir)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    datasets = datasets_from_fixtures(input_dir)
    datasets = _enrich_datasets_with_funding(datasets, funding_path)
    datasets = [
        PairDataset(
            dataset.pair, classify_regimes(dataset.frame, RegimeConfig(preserve_existing=True))
        )
        for dataset in datasets
    ]
    if not datasets:
        raise SystemExit(f"no experiment-ready fixture datasets found in {input_dir}")
    write_regime_dataset_report(datasets, reports / "regime_dataset_report.csv")
    harness = _experiment_harness(min_rows=1)
    results = harness.run(datasets)
    paths = harness.write_reports(results, reports)
    print(f"loaded {len(datasets)} fixture datasets")
    print(f"wrote {len(results)} experiment rows")
    for name, path in paths.items():
        print(f"{name}: {path}")


def ingest_pair_details(input_dir: Path | None = None) -> None:
    input_dir = input_dir or ROOT / "data" / "raw" / "pair_details"
    reports = ROOT / "reports"
    paths = write_pair_detail_reports(input_dir, reports)
    snapshots = load_pair_detail_snapshots(input_dir)
    print(f"pair_detail_snapshots: {len(snapshots)}")
    for name, path in paths.items():
        print(f"{name}: {path}")


def ingest_crypto_wizards_scanner(input_dir: Path | None = None) -> None:
    input_dir = input_dir or ROOT / "data" / "raw" / "crypto_wizards_scanner"
    reports = ROOT / "reports"
    paths = write_scanner_reports(input_dir, reports)
    rows = load_scanner_rows(input_dir)
    print(f"crypto_wizards_scanner_rows: {len(rows)}")
    for name, path in paths.items():
        print(f"{name}: {path}")


def _print_capture_checklist_summary(checklist: dict[str, object]) -> None:
    grouped_fields = [
        "missing_baseline_fields",
        "missing_ecm_fields",
        "missing_two_leg_fields",
        "missing_execution_assumption_fields",
    ]
    source_fields = [
        "capture_fetches",
        "capture_xhrs",
        "capture_worker_messages",
        "capture_wasm_extracts",
        "capture_har_entries",
        "capture_har_response_texts",
        "capture_har_dydx_candle_requests",
        "capture_storage_items",
        "capture_indexeddb_databases",
        "capture_scripts",
        "capture_resources",
    ]

    for field in grouped_fields:
        value = str(checklist.get(field, "") or "")
        print(f"{field}: {value if value else 'none'}")

    print(f"capture_completeness_score: {checklist.get('capture_completeness_score', 0)}")
    print(f"capture_payload_sources: {checklist.get('capture_payload_sources', '') or 'none'}")
    for field in source_fields:
        print(f"{field}: {checklist.get(field, 0)}")
    print(f"capture_operator_hint: {checklist.get('capture_operator_hint', '') or 'none'}")


def import_pair_detail_capture(input_path: Path, output_name: str | None = None) -> None:
    if not input_path.exists():
        raise SystemExit(f"pair-detail capture JSON not found: {input_path}")
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"input is not valid JSON: {input_path}: {exc}") from exc

    snapshot = snapshot_from_payload(payload)
    pair_id = (
        snapshot.pair_id if snapshot.pair_id and snapshot.pair_id != "unknown" else input_path.stem
    )
    filename = output_name or f"pair_{pair_id}_capture.json"
    if not filename.endswith(".json"):
        filename = f"{filename}.json"
    output_dir = ROOT / "data" / "raw" / "pair_details"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    atomic_write_text(output_path, json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    rows = extract_history_rows(payload)
    paths = write_pair_detail_reports(output_dir, ROOT / "reports")
    coverage = pd.DataFrame(pair_detail_history_coverage(output_dir))
    imported_row = (
        coverage[coverage["path"] == str(output_path)] if not coverage.empty else pd.DataFrame()
    )
    print(f"imported_pair_detail_capture: {output_path}")
    print(f"pair: {snapshot.pair}")
    print(f"history_rows_detected: {len(rows)}")
    if not imported_row.empty:
        print(f"experiment_ready: {bool(imported_row['experiment_ready'].iloc[0])}")
        print(f"ecm_history_ready: {bool(imported_row['ecm_history_ready'].iloc[0])}")
        print(f"two_leg_execution_ready: {bool(imported_row['two_leg_execution_ready'].iloc[0])}")
        missing_baseline = str(imported_row["missing_for_baseline_backtest"].iloc[0])
        missing_ecm = str(imported_row["missing_for_ecm_backtest"].iloc[0])
        missing_two_leg = str(imported_row["missing_for_two_leg_backtest"].iloc[0])
        assumption_notes = str(imported_row["execution_assumption_notes"].iloc[0])
        print(f"missing_for_baseline_backtest: {missing_baseline if missing_baseline else 'none'}")
        print(f"missing_for_ecm_backtest: {missing_ecm if missing_ecm else 'none'}")
        print(f"missing_for_two_leg_backtest: {missing_two_leg if missing_two_leg else 'none'}")
        print(f"execution_assumption_notes: {assumption_notes if assumption_notes else 'none'}")
    audit = pd.DataFrame(pair_detail_capture_audit(output_dir))
    imported_audit = audit[audit["path"] == str(output_path)] if not audit.empty else pd.DataFrame()
    if not imported_audit.empty:
        ready_paths = imported_audit[imported_audit["experiment_ready"]]["json_path"].tolist()
        ecm_ready_paths = imported_audit[imported_audit["ecm_history_ready"]]["json_path"].tolist()
        two_leg_ready_paths = imported_audit[imported_audit["two_leg_execution_ready"]][
            "json_path"
        ].tolist()
        print(f"capture_candidate_paths: {len(imported_audit)}")
        print(f"experiment_ready_paths: {','.join(ready_paths) if ready_paths else 'none'}")
        print(f"ecm_ready_paths: {','.join(ecm_ready_paths) if ecm_ready_paths else 'none'}")
        print(
            f"two_leg_ready_paths: {','.join(two_leg_ready_paths) if two_leg_ready_paths else 'none'}"
        )
    checklist = pair_detail_payload_capture_checklist(payload, output_path)
    print(f"found_required_fields: {checklist['found_required_fields'] or 'none'}")
    print(f"missing_required_fields: {checklist['missing_required_fields'] or 'none'}")
    _print_capture_checklist_summary(checklist)
    print(f"next_capture_focus: {checklist['next_capture_focus']}")
    for name, path in paths.items():
        print(f"{name}: {path}")


def import_latest_pair_detail_download(
    download_dir: Path | None = None, output_name: str | None = None
) -> Path:
    download_dir = download_dir or Path.home() / "Downloads"
    if not download_dir.exists():
        raise SystemExit(f"download directory not found: {download_dir}")

    patterns = [
        "crypto_wizards_pair_*_capture.json",
        "crypto_wizards_pair_*_capture*.json",
        "crypto_wizards_pair_*_capture.har",
        "crypto_wizards_pair_*_capture*.har",
        "*crypto*wizards*pair*capture*.json",
        "*crypto*wizards*pair*capture*.har",
        "*pair*capture*.json",
        "*pair*capture*.har",
    ]
    candidates: dict[Path, float] = {}
    for pattern in patterns:
        for path in download_dir.glob(pattern):
            if path.is_file():
                candidates[path] = path.stat().st_mtime

    if not candidates:
        raise SystemExit(f"no Crypto Wizards pair capture JSON found in: {download_dir}")

    latest = max(candidates, key=candidates.get)
    print(f"latest_pair_detail_download: {latest}")
    import_pair_detail_capture(latest, output_name)
    return latest


def import_dydx_candles(input_path: Path, output_dir: Path | None = None) -> Path:
    if not input_path.exists():
        raise SystemExit(f"dYdX candle response not found: {input_path}")
    try:
        output = archive_dydx_candles(
            input_path, output_dir or ROOT / "data" / "raw" / "dydx_candles"
        )
        candles = load_loose_candle_payload(output)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"input is not a valid dYdX candle response: {input_path}: {exc}") from exc
    first = candles[0]
    last = candles[-1]
    print(f"dydx_candles: {output}")
    print(f"ticker: {first.get('ticker', 'UNKNOWN')}")
    print(f"resolution: {first.get('resolution', 'UNKNOWN')}")
    print(f"rows: {len(candles)}")
    print(f"first: {first.get('startedAt')} close={first.get('close')}")
    print(f"last: {last.get('startedAt')} close={last.get('close')}")
    return output


def build_dydx_pair_history(
    *,
    left_candles: Path,
    right_candles: Path,
    asset_x: str,
    asset_y: str,
    pair_id: str,
    hedge_ratio: float,
    beta: float | None,
    interval: str | None,
    zscore_window: int,
    output_path: Path | None = None,
    derive_hedge_ratio: bool = False,
    funding_path: Path | None = None,
) -> Path:
    output = output_path or (
        ROOT
        / "data"
        / "raw"
        / "pair_details"
        / f"pair_{pair_id}_{(interval or 'candles').lower()}_dydx_candles_derived_history.json"
    )
    try:
        path = build_pair_history_from_candles(
            left_path=left_candles,
            right_path=right_candles,
            output_path=output,
            pair_id=pair_id,
            asset_x=asset_x,
            asset_y=asset_y,
            hedge_ratio=None if derive_hedge_ratio else hedge_ratio,
            beta=None if derive_hedge_ratio else beta,
            interval=interval,
            zscore_window=zscore_window,
            funding_path=funding_path,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"could not build dYdX pair history: {exc}") from exc
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("history", [])
    print(f"dydx_pair_history: {path}")
    print(f"pair: {payload.get('pair')}")
    print(f"interval: {payload.get('interval')}")
    print(f"rows: {len(history)}")
    print(f"hedge_ratio: {payload.get('hedge_ratio')}")
    print(f"ecm_derivation: {payload.get('ecm_derivation', {}).get('method', 'none')}")
    return path


def build_dydx_long_history_pair(
    *,
    input_dir: Path | None,
    asset_x: str,
    asset_y: str,
    pair_id: str,
    hedge_ratio: float,
    beta: float | None,
    interval: str | None,
    zscore_window: int,
    derive_hedge_ratio: bool = False,
    run_research: bool = False,
    funding_path: Path | None = None,
) -> dict[str, Path]:
    source_dir = input_dir or ROOT / "data" / "raw" / "dydx_long_history" / pair_id
    try:
        paths = build_pair_history_from_windowed_candles(
            input_dir=source_dir,
            output_dir=ROOT / "data" / "raw" / "dydx_candles",
            pair_output_dir=ROOT / "data" / "raw" / "pair_details",
            pair_id=pair_id,
            asset_x=asset_x,
            asset_y=asset_y,
            hedge_ratio=None if derive_hedge_ratio else hedge_ratio,
            beta=None if derive_hedge_ratio else beta,
            resolution=(interval or "5MINS").upper(),
            interval=(interval or "5mins").lower(),
            zscore_window=zscore_window,
            derive_hedge_ratio=derive_hedge_ratio,
            funding_path=funding_path,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"could not build dYdX long-history pair: {exc}") from exc

    payload = json.loads(paths["pair_history"].read_text(encoding="utf-8"))
    history = payload.get("history", [])
    print(f"dydx_long_history_pair: {paths['pair_history']}")
    print(f"source_dir: {source_dir}")
    print(f"left_candles: {paths['left_candles']}")
    print(f"right_candles: {paths['right_candles']}")
    print(f"pair: {payload.get('pair')}")
    print(f"interval: {payload.get('interval')}")
    print(f"rows: {len(history)}")
    print(f"hedge_ratio: {payload.get('hedge_ratio')}")
    print(f"ecm_derivation: {payload.get('ecm_derivation', {}).get('method', 'none')}")
    if run_research:
        resolved_funding_path = funding_path or ROOT / "data" / "processed" / "dydx_funding.csv"
        if resolved_funding_path.exists():
            rerun_p2_acceptance_evidence(
                input_dir=ROOT / "data" / "raw" / "pair_details",
                funding_path=resolved_funding_path,
            )
        else:
            print(f"research_spine_skipped: missing funding file {resolved_funding_path}")
    return paths


def fetch_dydx_long_history_windows(
    *,
    plan_path: Path | None = None,
    max_windows: int | None = None,
    skip_existing: bool = True,
    indexer_base: str = DEFAULT_INDEXER_BASE,
    indexer_scheme: str = "",
    allow_stale_fetch: bool = False,
    required_pair_id: str | None = None,
    required_asset_x: str | None = None,
    required_asset_y: str | None = None,
) -> pd.DataFrame:
    plan_file = plan_path or ROOT / "reports" / "dydx_long_history_plan.csv"
    if not plan_file.exists():
        raise SystemExit(f"long-history plan not found: {plan_file}")
    plan = pd.read_csv(plan_file)
    if plan.empty:
        raise SystemExit(f"long-history plan is empty: {plan_file}")
    if "method" not in plan.columns or "url" not in plan.columns or "save_as" not in plan.columns:
        raise SystemExit(f"long-history plan is missing required columns: {plan_file}")

    if required_pair_id or required_asset_x or required_asset_y:
        required_pair_id = _md_text(required_pair_id or "")
        required_asset_x = _normalize_dydx_market(_md_text(required_asset_x or ""))
        required_asset_y = _normalize_dydx_market(_md_text(required_asset_y or ""))
        matched_plan = plan.copy()
        if required_pair_id:
            matched_plan = matched_plan[matched_plan.get("pair_id").astype(str) == required_pair_id]
        if required_asset_x and required_asset_y:
            matched_plan = matched_plan[
                (matched_plan.get("asset_x").astype(str).str.upper() == required_asset_x)
                & (matched_plan.get("asset_y").astype(str).str.upper() == required_asset_y)
            ]
        elif required_asset_x:
            matched_plan = matched_plan[
                matched_plan.get("asset_x").astype(str).str.upper() == required_asset_x
            ]
        elif required_asset_y:
            matched_plan = matched_plan[
                matched_plan.get("asset_y").astype(str).str.upper() == required_asset_y
            ]

        if matched_plan.empty:
            expected = _md_text(
                f"pair_id={required_pair_id or 'any'}|asset_x={required_asset_x or 'any'}|asset_y={required_asset_y or 'any'}"
            )
            raise SystemExit(
                f"long-history plan {plan_file} did not include requested target ({expected})"
            )
        plan = matched_plan.copy()

    rows = plan[plan["method"].astype(str).str.upper() == "GET"].copy()
    if "request_name" in rows.columns:
        rows = rows[rows["request_name"].astype(str).str.contains("candles", case=False, na=False)]
    if "window" in rows.columns:
        rows["window"] = pd.to_numeric(rows["window"], errors="coerce")
        rows = rows.sort_values(["window", "request_name"], na_position="last")
    if max_windows is not None and max_windows > 0 and "window" in rows.columns:
        rows = rows[rows["window"] <= max_windows]

    indexer_bases = _indexer_base_candidates(indexer_base)
    fetched_rows: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        url = str(row.get("url") or "").strip()
        save_as = str(row.get("save_as") or "").strip()
        if not url or not save_as:
            continue
        path = Path(save_as)
        if not path.is_absolute():
            path = ROOT / path
        if skip_existing and path.exists() and path.stat().st_size > 0:
            fetched_rows.append(
                {
                    "window": row.get("window", ""),
                    "pair_id": row.get("pair_id", ""),
                    "asset_x": row.get("asset_x", ""),
                    "asset_y": row.get("asset_y", ""),
                    "request_name": row.get("request_name", ""),
                    "save_as": str(path),
                    "url": url,
                    "status": "existing",
                }
            )
            continue
        status = "failed"
        error = ""
        last_error: Exception | None = None
        for request_url in _indexer_url_variants(url, indexer_bases):
            try:
                _fetch_public_json(
                    request_url,
                    path,
                    allow_stale_fetch=allow_stale_fetch,
                    fetch_scheme=indexer_scheme,
                )
                status = "fetched"
                error = ""
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            error = str(last_error)
        fetched_rows.append(
            {
                "window": row.get("window", ""),
                "pair_id": row.get("pair_id", ""),
                "asset_x": row.get("asset_x", ""),
                "asset_y": row.get("asset_y", ""),
                "request_name": row.get("request_name", ""),
                "save_as": str(path),
                "url": url,
                "status": status,
                "error": error,
            }
        )
    frame = pd.DataFrame(fetched_rows)
    _write_csv_atomic(frame, ROOT / "reports" / "dydx_long_history_fetch.csv")
    return frame


def run_dydx_long_history(
    *,
    pair: str | None = None,
    asset_x: str | None = None,
    asset_y: str | None = None,
    pair_id: str | None = None,
    windows: int = 12,
    limit: int = 1000,
    resolution: str = "5MINS",
    indexer_base: str = DEFAULT_INDEXER_BASE,
    indexer_scheme: str = "",
    to_iso: str | None = None,
    derive_hedge_ratio: bool = True,
    run_research: bool = False,
    funding_path: Path | None = None,
    allow_stale_fetch: bool = False,
) -> dict[str, Path]:
    requested_indexer_base = _indexer_base_with_scheme(indexer_base, indexer_scheme)
    plan = dydx_long_history_plan_report(
        pair=pair,
        asset_x=asset_x,
        asset_y=asset_y,
        pair_id=pair_id,
        windows=windows,
        limit=limit,
        resolution=resolution,
        indexer_base=indexer_base,
        indexer_scheme=indexer_scheme,
        to_iso=to_iso,
    )
    resolved_pair_id = (
        str(plan.iloc[0]["pair_id"]) if not plan.empty else (pair_id or "long_history")
    )
    resolved_asset_x = str(plan.iloc[0]["asset_x"]) if not plan.empty else (asset_x or "")
    resolved_asset_y = str(plan.iloc[0]["asset_y"]) if not plan.empty else (asset_y or "")
    plan_path = ROOT / "reports" / "dydx_long_history_plan.csv"

    fetch_frame = fetch_dydx_long_history_windows(
        plan_path=plan_path,
        max_windows=windows,
        indexer_base=requested_indexer_base,
        indexer_scheme=indexer_scheme,
        allow_stale_fetch=allow_stale_fetch,
        required_pair_id=resolved_pair_id,
        required_asset_x=resolved_asset_x,
        required_asset_y=resolved_asset_y,
    )
    if fetch_frame.empty:
        raise SystemExit("long-history fetch produced no candle files")
    paths = build_dydx_long_history_pair(
        input_dir=ROOT / "data" / "raw" / "dydx_long_history" / resolved_pair_id,
        asset_x=resolved_asset_x,
        asset_y=resolved_asset_y,
        pair_id=resolved_pair_id,
        hedge_ratio=1.0,
        beta=1.0,
        interval=resolution.lower(),
        zscore_window=7,
        derive_hedge_ratio=derive_hedge_ratio,
        run_research=run_research,
        funding_path=funding_path,
    )
    paths["plan"] = plan_path
    paths["fetch"] = ROOT / "reports" / "dydx_long_history_fetch.csv"
    return paths


def import_dydx_candle_bundle_from_cli(
    input_path: Path, output_dir: Path | None = None, zscore_window: int = 7
) -> list[Path]:
    if not input_path.exists():
        raise SystemExit(f"dYdX candle bundle not found: {input_path}")
    pair_dir = output_dir or ROOT / "data" / "raw" / "pair_details"
    try:
        paths = import_dydx_candle_bundle(
            input_path,
            candle_output_dir=ROOT / "data" / "raw" / "dydx_candles",
            pair_output_dir=pair_dir,
            default_hedge_ratio=1.0,
            zscore_window=zscore_window,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"input is not a valid dYdX candle bundle: {input_path}: {exc}") from exc
    print(f"dydx_candle_bundle: {input_path}")
    print(f"pair_histories_written: {len(paths)}")
    for path in paths:
        print(f"pair_history: {path}")
    if paths:
        report_paths = write_pair_detail_reports(pair_dir, ROOT / "reports")
        quality = pd.DataFrame(
            pair_detail_quality_report(pair_dir), columns=PAIR_DETAIL_QUALITY_COLUMNS
        )
        imported_quality = (
            quality[quality["path"].isin({str(path) for path in paths})]
            if not quality.empty
            else quality
        )
        if not imported_quality.empty:
            columns = [
                "pair",
                "interval",
                "history_rows",
                "research_usable",
                "execution_usable",
                "quality_blockers",
            ]
            print(imported_quality[columns].to_string(index=False))
        print(f"pair_detail_quality_report: {report_paths['quality']}")
    return paths


def dydx_two_leg_request_template_report(
    *,
    pair: str | None = None,
    asset_x: str | None = None,
    asset_y: str | None = None,
    pair_id: str = "manual",
    hedge_ratio: float = 1.0,
    beta: float | None = None,
    zscore_window: int = 7,
    limit: int = 100,
    indexer_base: str = DEFAULT_INDEXER_BASE,
    indexer_scheme: str = "",
    output_path: Path | None = None,
) -> pd.DataFrame:
    left_asset, right_asset = _resolve_two_leg_assets(pair=pair, asset_x=asset_x, asset_y=asset_y)
    requested_indexer_base = _indexer_base_with_scheme(indexer_base, indexer_scheme)
    rows = dydx_two_leg_request_rows(
        asset_x=left_asset,
        asset_y=right_asset,
        pair_id=pair_id,
        hedge_ratio=hedge_ratio,
        beta=beta,
        limit=limit,
        indexer_base=requested_indexer_base,
        zscore_window=zscore_window,
    )
    frame = pd.DataFrame(rows)
    output = output_path or ROOT / "reports" / "dydx_two_leg_data_requests.csv"
    _write_csv_atomic(frame, output)
    return frame


def print_dydx_two_leg_request_template(
    *,
    pair: str | None = None,
    asset_x: str | None = None,
    asset_y: str | None = None,
    pair_id: str = "manual",
    hedge_ratio: float = 1.0,
    beta: float | None = None,
    zscore_window: int = 7,
    limit: int = 100,
    indexer_base: str = DEFAULT_INDEXER_BASE,
    indexer_scheme: str = "",
    output_path: Path | None = None,
) -> None:
    frame = dydx_two_leg_request_template_report(
        pair=pair,
        asset_x=asset_x,
        asset_y=asset_y,
        pair_id=pair_id,
        hedge_ratio=hedge_ratio,
        beta=beta,
        zscore_window=zscore_window,
        limit=limit,
        indexer_base=indexer_base,
        indexer_scheme=indexer_scheme,
        output_path=output_path,
    )
    output = output_path or ROOT / "reports" / "dydx_two_leg_data_requests.csv"
    print(frame[["request_name", "url", "save_as", "notes"]].to_string(index=False))
    print(f"dydx_two_leg_data_requests: {output}")


def fetch_dydx_two_leg_data(
    *,
    pair: str | None = None,
    asset_x: str | None = None,
    asset_y: str | None = None,
    pair_id: str = "manual",
    hedge_ratio: float = 1.0,
    beta: float | None = None,
    zscore_window: int = 7,
    indexer_base: str = DEFAULT_INDEXER_BASE,
    indexer_scheme: str = "",
    limit: int = 100,
    output_dir: Path | None = None,
    run_research: bool = False,
    derive_hedge_ratio: bool = False,
    allow_stale_fetch: bool = False,
    skip_fetch: bool = False,
    funding_path: Path | None = None,
) -> dict[str, Path]:
    left_asset, right_asset = _resolve_two_leg_assets(pair=pair, asset_x=asset_x, asset_y=asset_y)
    requested_indexer_base = _indexer_base_with_scheme(indexer_base, indexer_scheme)
    manual_dir = output_dir or ROOT / "data" / "raw" / "dydx_manual"
    _assert_dydx_raw_target(manual_dir, "dydx")
    indexer_bases = _indexer_base_candidates(requested_indexer_base)
    rows = dydx_two_leg_request_rows(
        asset_x=left_asset,
        asset_y=right_asset,
        pair_id=pair_id,
        hedge_ratio=hedge_ratio,
        beta=beta,
        indexer_base=requested_indexer_base,
        output_dir=manual_dir,
        limit=limit,
        zscore_window=zscore_window,
    )
    request_report = ROOT / "reports" / "dydx_two_leg_data_requests.csv"
    _write_csv_atomic(pd.DataFrame(rows), request_report)

    saved: dict[str, Path] = {}
    for row in rows:
        if row.get("method") != "GET":
            continue
        path = Path(str(row["save_as"]))
        if not path.is_absolute():
            path = ROOT / path
        if skip_fetch:
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(
                    f"skip_fetch is enabled, but required payload file is missing or empty: {path}"
                )
            saved[str(row["request_name"])] = path
            continue
        last_error: Exception | None = None
        for request_url in _indexer_url_variants(str(row["url"]), indexer_bases):
            try:
                _fetch_public_json(
                    request_url,
                    path,
                    allow_stale_fetch=allow_stale_fetch,
                    fetch_scheme=indexer_scheme,
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        saved[str(row["request_name"])] = path

    result_paths: dict[str, Path] = {"request_report": request_report}
    result_paths.update(saved)
    try:
        left_candles = archive_dydx_candles(
            saved["asset_x_candles_5mins"], ROOT / "data" / "raw" / "dydx_candles"
        )
        right_candles = archive_dydx_candles(
            saved["asset_y_candles_5mins"], ROOT / "data" / "raw" / "dydx_candles"
        )
    except ValueError:
        return result_paths
    funding_csv = export_dydx_funding_payload(
        manual_dir, funding_path or ROOT / "data" / "processed" / "dydx_funding.csv"
    )
    pair_history = build_dydx_pair_history(
        left_candles=left_candles,
        right_candles=right_candles,
        asset_x=left_asset,
        asset_y=right_asset,
        pair_id=pair_id,
        hedge_ratio=None if derive_hedge_ratio else hedge_ratio,
        beta=None if derive_hedge_ratio else beta,
        interval="5mins",
        zscore_window=zscore_window,
        derive_hedge_ratio=derive_hedge_ratio,
        funding_path=funding_csv,
    )
    coverage = funding_coverage_report(funding_csv, pairs=[f"{left_asset}-{right_asset}"])
    if run_research:
        rerun_p2_acceptance_evidence(
            input_dir=ROOT / "data" / "raw" / "pair_details",
            funding_path=funding_csv,
        )
    result_paths.update(
        {
            "left_candles": left_candles,
            "right_candles": right_candles,
            "pair_history": pair_history,
            "funding_csv": funding_csv,
            "funding_coverage": ROOT / "reports" / "funding_coverage.csv",
        }
    )
    return result_paths


def print_fetch_dydx_two_leg_data(
    *,
    pair: str | None = None,
    asset_x: str | None = None,
    asset_y: str | None = None,
    pair_id: str = "manual",
    hedge_ratio: float = 1.0,
    beta: float | None = None,
    zscore_window: int = 7,
    indexer_base: str = DEFAULT_INDEXER_BASE,
    indexer_scheme: str = "",
    limit: int = 100,
    output_dir: Path | None = None,
    run_research: bool = False,
    derive_hedge_ratio: bool = False,
    allow_stale_fetch: bool = False,
    skip_fetch: bool = False,
    funding_path: Path | None = None,
) -> None:
    paths = fetch_dydx_two_leg_data(
        pair=pair,
        asset_x=asset_x,
        asset_y=asset_y,
        pair_id=pair_id,
        hedge_ratio=hedge_ratio,
        beta=beta,
        zscore_window=zscore_window,
        indexer_base=indexer_base,
        indexer_scheme=indexer_scheme,
        limit=limit,
        output_dir=output_dir,
        run_research=run_research,
        derive_hedge_ratio=derive_hedge_ratio,
        allow_stale_fetch=allow_stale_fetch,
        skip_fetch=skip_fetch,
        funding_path=funding_path,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


def _fetch_public_json(
    url: str,
    output_path: Path,
    timeout: float = 12.0,
    max_retries: int = 2,
    allow_stale_fetch: bool = False,
    fetch_scheme: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    allow_stale = allow_stale_fetch or os.getenv("QPA_ALLOW_STALE_FETCH", "").lower() in {
        "1",
        "true",
        "yes",
    }
    use_requests = os.getenv("QPA_USE_REQUESTS_FETCH", "").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if allow_stale and output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    last_exc: Exception | None = None
    url_candidates = _fetch_url_scheme_variants(url, forced_scheme=fetch_scheme)
    last_curl_error: str | None = None
    if use_requests:
        for attempt in range(1, max_retries + 1):
            for fetch_url in url_candidates:
                for request_url, headers, verify_ssl in _request_variants_for_url(fetch_url):
                    try:
                        get_kwargs = {"headers": headers, "timeout": (3.0, timeout)}
                        if not verify_ssl:
                            get_kwargs["verify"] = False
                        response = requests.get(request_url, **get_kwargs)
                        response.raise_for_status()
                        try:
                            payload = response.json()
                            atomic_write_text(output_path, json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                        except ValueError:
                            atomic_write_text(output_path, response.text, encoding="utf-8")
                        return output_path
                    except requests.exceptions.RequestException as exc:
                        last_exc = exc
                        continue
            if attempt < max_retries:
                continue

    # On flaky network stacks, fallback to curl with DNS-over-HTTPS resolution.
    # This allows hostnames to be resolved even when local resolution intermittently fails.
    for request_url in url_candidates:
        fallback_host = urlparse(request_url).hostname
        parse = urlparse(request_url)

        resolved_ip: str | None = None
        dns_error: str | None = None
        if fallback_host:
            resolved_ip = _resolve_host_via_doh(fallback_host)
            if resolved_ip is None:
                dns_error = "all_doh_endpoints_failed"

        fallback_hosts = _dns_fallback_ip_candidates(fallback_host) if fallback_host else []
        fallback_targets = [resolved_ip] if resolved_ip else []
        for host in fallback_hosts:
            if host not in fallback_targets:
                fallback_targets.append(host)

        # Keep a final raw-URL attempt even when DNS/IP fallbacks are defined.
        # In many constrained environments, direct hostname resolution may work
        # from curl even when Python-side DNS resolution is failing.
        if None not in fallback_targets:
            fallback_targets.append(None)

        curl_candidates = _candidate_curl_paths()
        if not curl_candidates:
            raise RuntimeError(
                f"failed to fetch dYdX indexer URL {request_url}; "
                "fallback is unavailable because no curl binary is installed"
            ) from last_exc

        curl_exc: Exception | None = None
        curl_trace: list[str] = []
        for cmd in curl_candidates:
            for target_ip in fallback_targets:
                resolved_url = request_url
                curl_cmd = [
                    cmd,
                    "-L",
                    "--fail",
                    "--show-error",
                    "--silent",
                    "--retry",
                    "3",
                    "--retry-delay",
                    "2",
                    "--max-time",
                    str(int(timeout)),
                    "-H",
                    "Content-Type: application/json",
                    "--output",
                    str(output_path),
                ]

                if target_ip:
                    connect_port = (
                        str(parse.port)
                        if parse.port
                        else ("443" if parse.scheme == "https" else "80")
                    )
                    resolve_host = f"{fallback_host}:{connect_port}:{target_ip}"
                    # Keep host-based URL to preserve TLS SNI while routing via explicit DNS fallback.
                    # `curl --resolve` handles host-to-IP mapping without forcing IP into URL path.
                    curl_cmd.extend(
                        ["--http1.1", "--resolve", resolve_host, "-H", f"Host: {fallback_host}"]
                    )
                curl_cmd.extend([resolved_url, "--insecure"])

                curl_trace.append(f"{cmd} target={target_ip or 'default'}")
                try:
                    subprocess.run(
                        curl_cmd,
                        check=True,
                    )
                    curl_exc = None
                    break
                except (OSError, subprocess.CalledProcessError) as exc:
                    curl_exc = exc
                    last_curl_error = str(exc)
                    continue

            if curl_exc is None:
                break

        if curl_exc is None:
            break
    if curl_exc is not None:
        details = []
        if last_exc:
            details.append(f"requests={type(last_exc).__name__}:{last_exc}")
        if dns_error:
            details.append(f"doh={dns_error}")
        if last_curl_error:
            details.append(f"curl_last={last_curl_error}")
        if curl_trace:
            details.append(f"curl_traces={'; '.join(curl_trace)}")
        raise RuntimeError(
            f"failed to fetch dYdX indexer URL {url}; "
            f"attempts={min(max_retries, 3)}; {'; '.join(details)}"
        ) from (curl_exc or last_exc)
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        atomic_write_text(output_path, json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except json.JSONDecodeError:
        pass
    return output_path


def _resolve_two_leg_assets(
    *, pair: str | None, asset_x: str | None, asset_y: str | None
) -> tuple[str, str]:
    if asset_x and asset_y:
        return asset_x, asset_y
    if pair:
        requirements = funding_market_requirements([pair])
        if not requirements.empty and bool(requirements.iloc[0].get("valid", False)):
            return str(requirements.iloc[0]["market_x"]), str(requirements.iloc[0]["market_y"])
        error = (
            str(requirements.iloc[0].get("error", "invalid pair"))
            if not requirements.empty
            else "invalid pair"
        )
        raise SystemExit(f"could not resolve dYdX markets from --pair {pair}: {error}")
    raise SystemExit(
        "dydx-two-leg-request-template requires --pair or both --asset-x and --asset-y"
    )


def inspect_pair_detail_capture(input_path: Path) -> None:
    if not input_path.exists():
        raise SystemExit(f"pair-detail capture JSON not found: {input_path}")
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"input is not valid JSON: {input_path}: {exc}") from exc

    snapshot = snapshot_from_payload(payload)
    rows = extract_history_rows(payload)
    coverage = pair_detail_payload_history_coverage(payload, input_path)
    audit = pair_detail_payload_capture_audit(payload, input_path)
    checklist = pair_detail_payload_capture_checklist(payload, input_path)
    ready_paths = [row["json_path"] for row in audit if bool(row.get("experiment_ready"))]
    ecm_ready_paths = [row["json_path"] for row in audit if bool(row.get("ecm_history_ready"))]
    two_leg_ready_paths = [
        row["json_path"] for row in audit if bool(row.get("two_leg_execution_ready"))
    ]

    print(f"inspected_pair_detail_capture: {input_path}")
    print(f"pair: {snapshot.pair}")
    print(f"history_rows_detected: {len(rows)}")
    print(f"experiment_ready: {bool(coverage['experiment_ready'])}")
    print(f"ecm_history_ready: {bool(coverage['ecm_history_ready'])}")
    print(f"two_leg_execution_ready: {bool(coverage['two_leg_execution_ready'])}")
    print(f"missing_for_baseline_backtest: {coverage['missing_for_baseline_backtest'] or 'none'}")
    print(f"missing_for_ecm_backtest: {coverage['missing_for_ecm_backtest'] or 'none'}")
    print(f"missing_for_two_leg_backtest: {coverage['missing_for_two_leg_backtest'] or 'none'}")
    print(f"execution_assumption_notes: {coverage['execution_assumption_notes'] or 'none'}")
    print(f"capture_candidate_paths: {len(audit)}")
    print(f"experiment_ready_paths: {','.join(ready_paths) if ready_paths else 'none'}")
    print(f"ecm_ready_paths: {','.join(ecm_ready_paths) if ecm_ready_paths else 'none'}")
    print(
        f"two_leg_ready_paths: {','.join(two_leg_ready_paths) if two_leg_ready_paths else 'none'}"
    )
    print(f"found_required_fields: {checklist['found_required_fields'] or 'none'}")
    print(f"missing_required_fields: {checklist['missing_required_fields'] or 'none'}")
    _print_capture_checklist_summary(checklist)
    print(f"next_capture_focus: {checklist['next_capture_focus']}")


def pair_detail_capture_preflight(
    input_path: Path, output_path: Path | None = None
) -> pd.DataFrame:
    if not input_path.exists():
        raise SystemExit(f"pair-detail capture JSON not found: {input_path}")
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"input is not valid JSON: {input_path}: {exc}") from exc
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "pair_detail_capture_preflight.csv"
    checklist = pair_detail_payload_capture_checklist(payload, input_path)
    frame = pd.DataFrame([checklist], columns=PAIR_DETAIL_CAPTURE_CHECKLIST_COLUMNS)
    _write_csv_atomic(frame, output)
    return frame


def print_pair_detail_capture_preflight(
    input_path: Path | None, output_path: Path | None = None
) -> None:
    if input_path is None:
        raise SystemExit("capture-preflight requires --json-path")
    output = output_path or ROOT / "reports" / "pair_detail_capture_preflight.csv"
    frame = pair_detail_capture_preflight(input_path, output)
    print(frame.to_string(index=False))
    print(f"pair_detail_capture_preflight: {output}")


def write_pair_detail_capture_checklist(input_dir: Path | None = None) -> None:
    input_dir = input_dir or ROOT / "data" / "raw" / "pair_details"
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = reports / "pair_detail_capture_checklist.csv"
    frame = pd.DataFrame(
        pair_detail_capture_checklist(input_dir), columns=PAIR_DETAIL_CAPTURE_CHECKLIST_COLUMNS
    )
    _write_csv_atomic(frame, output)
    print(frame.to_string(index=False))
    print(f"pair_detail_capture_checklist: {output}")


def write_pair_detail_quality_report(input_dir: Path | None = None) -> None:
    input_dir = input_dir or ROOT / "data" / "raw" / "pair_details"
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = reports / "pair_detail_quality_report.csv"
    frame = pd.DataFrame(pair_detail_quality_report(input_dir), columns=PAIR_DETAIL_QUALITY_COLUMNS)
    _write_csv_atomic(frame, output)
    print(frame.to_string(index=False))
    print(f"pair_detail_quality_report: {output}")


def run_pair_detail_experiments(
    input_dir: Path | None = None,
    funding_path: Path | None = None,
    require_research_usable: bool = False,
    pair_filter: tuple[str, ...] = (),
    strategy_ids: tuple[int, ...] | None = None,
) -> None:
    input_dir = input_dir or ROOT / "data" / "raw" / "pair_details"
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    datasets = datasets_from_pair_detail_snapshots(
        input_dir, require_research_usable=require_research_usable
    )
    if pair_filter:
        requested = {_normalize_pair_for_filter(pair) for pair in pair_filter if pair}
        filtered: list[PairDataset] = []
        for dataset in datasets:
            if _normalize_pair_for_filter(dataset.pair) in requested:
                filtered.append(dataset)
        datasets = filtered
    datasets = _enrich_datasets_with_funding(datasets, funding_path)
    datasets = [
        PairDataset(
            dataset.pair, classify_regimes(dataset.frame, RegimeConfig(preserve_existing=True))
        )
        for dataset in datasets
    ]
    if not datasets:
        raise SystemExit(f"no experiment-ready pair-detail history datasets found in {input_dir}")
    write_regime_dataset_report(datasets, reports / "regime_dataset_report.csv")
    harness = _experiment_harness(
        strategy_ids=tuple(int(v) for v in strategy_ids) if strategy_ids else None,
    )
    results = harness.run(datasets)
    paths = harness.write_reports(results, reports)
    print(f"loaded {len(datasets)} pair-detail history dataset(s)")
    print(f"wrote {len(results)} experiment rows")
    for name, path in paths.items():
        print(f"{name}: {path}")


def _normalize_pair_for_filter(pair: str) -> str:
    if not pair:
        return ""
    normalized = pair.strip().lower().replace("/", "-").replace("_", "-")
    normalized = normalized.replace(" ", "")
    parts = [part for part in normalized.split("-") if part]
    # Accept both pair-id style (btc_eth) and full market style (BTC-USD-ETH-USD)
    # by reducing to the pair of market tickers, sorted for deterministic comparisons.
    symbols = [part for part in parts if part != "usd"]
    if not symbols:
        return ""
    symbols = sorted(set(symbols))
    if len(symbols) >= 2:
        return f"{symbols[0]}_{symbols[1]}"
    return symbols[0]


def _recommended_strategies_from_mode(strategy_mode: str | None) -> list[int]:
    text = (strategy_mode or "").strip().lower()
    if not text:
        return []
    if "copula" in text:
        return [33, 34, 35, 36, 37, 5, 6, 7, 28, 29]
    if "ou" in text:
        return [14, 1, 2, 11, 12, 13, 20]
    if "dynamic" in text:
        return [20, 1, 2, 11, 12, 13]
    if "mean" in text or "reversion" in text or "zscore" in text or text == "static":
        return [1, 2, 11, 12, 13, 14, 20]
    return [1, 2, 11, 12, 13, 14, 20]


def _extract_recommended_strategy_ids(payload: dict) -> list[int]:
    """Best-effort strategy IDs from common pair detail recommendation fields."""

    recommended: list[int] = []

    def _append_ids(ids: list[int]) -> None:
        for sid in ids:
            if sid and sid not in recommended:
                recommended.append(sid)

    def _to_int(value: object) -> int | None:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    for field in ("strategy_id", "local_strategy_id"):
        sid = _to_int(payload.get(field))
        if sid is not None:
            _append_ids([sid])

    for key in (
        "dashboard_recommended_strategy",
        "strategy_mode",
        "exact_mode",
        "strategy_hint",
        "strategy",
    ):
        _append_ids(_recommended_strategies_from_mode(str(payload.get(key, ""))))

    return recommended


def _pair_expansion_recommended_and_all_strategy_ids(
    input_dir: Path,
    pair_filter: tuple[str, ...],
) -> tuple[int, ...]:
    ordered: list[int] = []
    seen: set[int] = set()
    for sid in _strategy_id_priority_for_pair_details(input_dir=input_dir, pair_filter=pair_filter):
        if sid not in seen:
            ordered.append(sid)
            seen.add(sid)
    for spec in STRATEGIES:
        sid = int(spec.id)
        if sid not in seen:
            ordered.append(sid)
            seen.add(sid)
    return tuple(ordered)


def _strategy_id_priority_for_pair_details(
    input_dir: Path,
    pair_filter: tuple[str, ...],
) -> list[int]:
    requested = {_normalize_pair_for_filter(pair) for pair in pair_filter} if pair_filter else None
    prioritized: list[int] = []

    for path in sorted(Path(input_dir).glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        pair = _normalize_pair_for_filter(str(payload.get("pair", "")))
        if requested is not None and pair not in requested:
            continue
        if not pair:
            continue
        for strategy_id in _extract_recommended_strategy_ids(payload):
            if strategy_id not in prioritized:
                prioritized.append(strategy_id)

    if not prioritized:
        return [1]
    return prioritized


def build_ml_trade_filter_dataset_report(
    input_dir: Path | None = None,
    funding_path: Path | None = None,
    output_path: Path | None = None,
    require_research_usable: bool = True,
) -> Path:
    source = input_dir or ROOT / "data" / "raw" / "pair_details"
    output = output_path or ROOT / "reports" / "ml_trade_filter_dataset.csv"
    datasets = datasets_from_pair_detail_snapshots(
        source, require_research_usable=require_research_usable
    )
    datasets = _enrich_datasets_with_funding(datasets, funding_path)
    datasets = [
        PairDataset(
            dataset.pair, classify_regimes(dataset.frame, RegimeConfig(preserve_existing=True))
        )
        for dataset in datasets
    ]
    frame = build_trade_filter_dataset(datasets)
    if frame.empty:
        raise SystemExit(f"no ML trade-filter candidate rows could be built from {source}")
    _write_csv_atomic(frame, output)
    return output


def print_build_ml_trade_filter_dataset(
    input_dir: Path | None = None,
    funding_path: Path | None = None,
    output_path: Path | None = None,
    require_research_usable: bool = True,
) -> None:
    output = build_ml_trade_filter_dataset_report(
        input_dir=input_dir,
        funding_path=funding_path,
        output_path=output_path,
        require_research_usable=require_research_usable,
    )
    frame = pd.read_csv(output)
    print(frame.head(10).to_string(index=False))
    print(f"ml_trade_filter_dataset: {output}")


def _load_ml_trade_filter_dataset(
    input_dir: Path | None,
    funding_path: Path | None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    source = input_dir or ROOT / "reports" / "ml_trade_filter_dataset.csv"
    if source.exists() and source.is_file() and source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    dataset_path = build_ml_trade_filter_dataset_report(
        input_dir=source,
        funding_path=funding_path,
        output_path=output_path or ROOT / "reports" / "ml_trade_filter_dataset.csv",
    )
    return pd.read_csv(dataset_path)


def print_train_ml_trade_filter(
    input_dir: Path | None = None,
    funding_path: Path | None = None,
    output_dir: Path | None = None,
    walkforward_splits: int = 5,
    min_train_rows: int = 100,
) -> None:
    dataset = _load_ml_trade_filter_dataset(input_dir, funding_path)
    output = output_dir or ROOT / "reports" / "ml_trade_filter"
    paths = train_trade_filter_walkforward(
        dataset,
        output_dir=output,
        n_splits=walkforward_splits,
        min_train_rows=min_train_rows,
    )
    summary = pd.read_csv(paths["summary"])
    print(summary.to_string(index=False))
    for name, path in paths.items():
        print(f"{name}: {path}")


def print_shadow_ml_trade_filter(
    input_dir: Path | None = None,
    funding_path: Path | None = None,
    model_path: Path | None = None,
    output_path: Path | None = None,
    model_sha256: str = "",
) -> None:
    dataset = _load_ml_trade_filter_dataset(input_dir, funding_path)
    artifact = model_path or ROOT / "reports" / "ml_trade_filter" / "ml_trade_filter_best_model.pkl"
    output = output_path or ROOT / "reports" / "ml_trade_filter_shadow_predictions.csv"
    path = shadow_trade_filter_predictions(
        dataset,
        model_artifact_path=artifact,
        output_path=output,
        expected_model_sha256=model_sha256,
    )
    frame = pd.read_csv(path)
    print(frame.head(20).to_string(index=False))
    print(f"ml_trade_filter_shadow_predictions: {path}")


def print_compare_ml_shadow_models(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    pair_list: tuple[str, ...] = (),
) -> None:
    predictions_path = (
        input_dir
        or ROOT / "reports" / "ml_trade_filter" / "ml_trade_filter_walkforward_predictions.csv"
    )
    if not predictions_path.exists():
        raise SystemExit(f"ML walk-forward predictions not found: {predictions_path}")
    output = output_dir or predictions_path.parent
    predictions = pd.read_csv(predictions_path)
    model_report, pair_report = shadow_model_branch_comparison(predictions, pairs=pair_list or None)
    model_path = _write_csv_atomic(
        model_report, output / "ml_trade_filter_branch_model_comparison.csv"
    )
    pair_path = _write_csv_atomic(
        pair_report, output / "ml_trade_filter_branch_pair_comparison.csv"
    )
    print(model_report.to_string(index=False))
    print(f"ml_trade_filter_branch_model_comparison: {model_path}")
    print(f"ml_trade_filter_branch_pair_comparison: {pair_path}")


def _best_requested_pair_datasets(
    input_dir: Path,
    funding_path: Path | None,
    requested_pairs: tuple[str, ...],
) -> list[PairDataset]:
    requested = requested_pairs or DEFAULT_FAMILY_SWEEP_PAIRS
    datasets = datasets_from_pair_detail_snapshots(input_dir, require_research_usable=True)
    datasets = _enrich_datasets_with_funding(datasets, funding_path)
    classified = [
        PairDataset(
            dataset.pair, classify_regimes(dataset.frame, RegimeConfig(preserve_existing=True))
        )
        for dataset in datasets
    ]
    best_by_pair: dict[str, PairDataset] = {}
    for dataset in classified:
        if dataset.pair not in requested:
            continue
        existing = best_by_pair.get(dataset.pair)
        if existing is None or len(dataset.frame) > len(existing.frame):
            best_by_pair[dataset.pair] = dataset
    return [best_by_pair[pair] for pair in requested if pair in best_by_pair]


def strategy_family_sweep_report(
    input_dir: Path | None = None,
    funding_path: Path | None = None,
    output_dir: Path | None = None,
    pair_list: tuple[str, ...] = (),
) -> dict[str, Path]:
    root = input_dir or ROOT / "data" / "raw" / "pair_details"
    output = output_dir or ROOT / "reports" / "strategy_family_sweep"
    output.mkdir(parents=True, exist_ok=True)
    selected_pairs = pair_list or DEFAULT_FAMILY_SWEEP_PAIRS
    datasets = _best_requested_pair_datasets(root, funding_path, selected_pairs)
    if not datasets:
        raise SystemExit(f"no requested pair datasets found in {root}")

    harness = _experiment_harness()
    results = harness.run(datasets)
    harness.write_reports(results, output / "base_reports")
    detail = results.copy()
    detail_path = _write_csv_atomic(detail, output / "strategy_family_sweep_detail.csv")

    summary = strategy_acceptance_report(results, harness.config.gate).copy()
    summary["rank_key"] = list(
        zip(
            ~summary["production_eligible"].fillna(False).astype(bool),
            ~summary["preferred_eligible"].fillna(False).astype(bool),
            -pd.to_numeric(summary["passing_pairs"], errors="coerce").fillna(0),
            -pd.to_numeric(summary["median_sharpe"], errors="coerce").fillna(0.0),
            -pd.to_numeric(summary["median_profit_factor"], errors="coerce").fillna(0.0),
            -pd.to_numeric(summary["total_trades"], errors="coerce").fillna(0),
            pd.to_numeric(summary["worst_drawdown"], errors="coerce").fillna(0.0),
        )
    )
    summary = summary.sort_values("rank_key").drop(columns=["rank_key"]).reset_index(drop=True)
    summary.insert(0, "strategy_rank", range(1, len(summary) + 1))
    summary.insert(3, "pairs_requested", len(selected_pairs))
    summary_path = _write_csv_atomic(summary, output / "strategy_family_sweep_summary.csv")

    ranked = summary[
        [
            "strategy_rank",
            "strategy_name",
            "family",
            "pairs_requested",
            "passing_pairs",
            "total_trades",
            "median_profit_factor",
            "median_sharpe",
            "worst_drawdown",
            "production_eligible",
            "preferred_eligible",
            "acceptance_reason",
            "preferred_reason",
        ]
    ].rename(columns={"strategy_name": "strategy", "pairs_requested": "pairs_tested"})
    ranked_path = _write_csv_atomic(ranked, output / "strategy_family_ranked_comparison.csv")

    best_by_family = (
        summary.sort_values(
            [
                "production_eligible",
                "preferred_eligible",
                "passing_pairs",
                "median_sharpe",
                "median_profit_factor",
                "total_trades",
                "worst_drawdown",
            ],
            ascending=[False, False, False, False, False, False, True],
        )
        .groupby("family", as_index=False)
        .first()
        .sort_values(
            [
                "production_eligible",
                "preferred_eligible",
                "passing_pairs",
                "median_sharpe",
                "median_profit_factor",
                "total_trades",
                "worst_drawdown",
            ],
            ascending=[False, False, False, False, False, False, True],
        )
        .reset_index(drop=True)
    )
    best_by_family.insert(0, "family_rank", range(1, len(best_by_family) + 1))
    best_by_family_path = _write_csv_atomic(
        best_by_family, output / "strategy_family_best_by_family.csv"
    )

    shortlist = best_by_family[
        (
            best_by_family["production_eligible"].fillna(False).astype(bool)
            | best_by_family["preferred_eligible"].fillna(False).astype(bool)
            | (pd.to_numeric(best_by_family["passing_pairs"], errors="coerce").fillna(0) > 0)
        )
    ].copy()
    if shortlist.empty:
        shortlist = best_by_family.head(min(3, len(best_by_family))).copy()
    shortlist.insert(
        1, "promotion_reason", shortlist.apply(_strategy_family_promotion_reason, axis=1)
    )
    shortlist_path = _write_csv_atomic(
        shortlist, output / "strategy_family_promotion_shortlist.csv"
    )

    notes_path = output / "strategy_family_sweep_notes.md"
    atomic_write_text(notes_path, _strategy_family_sweep_notes(selected_pairs, summary, best_by_family, shortlist), encoding="utf-8")
    failure_attribution_path = output / "strategy_family_failure_attribution.csv"
    family_failure_attribution_report(output, failure_attribution_path)
    failure_notes_path = output / "strategy_family_failure_attribution.md"
    atomic_write_text(failure_notes_path, _strategy_family_failure_notes(output), encoding="utf-8")

    return {
        "detail": detail_path,
        "summary": summary_path,
        "ranked": ranked_path,
        "best_by_family": best_by_family_path,
        "promotion_shortlist": shortlist_path,
        "notes": notes_path,
        "failure_attribution": failure_attribution_path,
        "failure_notes": failure_notes_path,
    }


def _strategy_family_promotion_reason(row: pd.Series) -> str:
    if bool(row.get("production_eligible", False)):
        return "production_eligible"
    if bool(row.get("preferred_eligible", False)):
        return "preferred_eligible"
    passing_pairs = int(
        pd.to_numeric(pd.Series([row.get("passing_pairs")]), errors="coerce").fillna(0).iloc[0]
    )
    if passing_pairs > 0:
        return "positive_passing_pairs"
    return "top_family_placeholder"


def _strategy_family_sweep_notes(
    selected_pairs: tuple[str, ...],
    summary: pd.DataFrame,
    best_by_family: pd.DataFrame,
    shortlist: pd.DataFrame,
) -> str:
    lines = [
        "# Strategy Family Sweep Notes",
        "",
        "## Pair Pack",
        "",
    ]
    lines.extend([f"- `{pair}`" for pair in selected_pairs])
    lines.extend(
        [
            "",
            "## Sweep Readout",
            "",
            f"- strategies_run: {len(summary)}",
            f"- families_seen: {summary['family'].nunique() if not summary.empty else 0}",
            f"- production_eligible_strategies: {int(summary['production_eligible'].fillna(False).astype(bool).sum()) if not summary.empty else 0}",
            f"- preferred_eligible_strategies: {int(summary['preferred_eligible'].fillna(False).astype(bool).sum()) if not summary.empty else 0}",
            "",
            "## Best Families",
            "",
        ]
    )
    for _, row in best_by_family.head(5).iterrows():
        lines.append(
            f"- `{row['family']}` -> `{row['strategy_name']}` "
            f"(passing_pairs={int(row['passing_pairs'])}, sharpe={float(row['median_sharpe']):.3f}, "
            f"pf={float(row['median_profit_factor']):.3f}, dd={float(row['worst_drawdown']):.3f})"
        )
    lines.extend(["", "## Promotion Shortlist", ""])
    for _, row in shortlist.iterrows():
        lines.append(
            f"- `{row['family']}` / `{row['strategy_name']}` because `{row['promotion_reason']}`"
        )
    return "\n".join(lines) + "\n"


def print_strategy_family_sweep(
    input_dir: Path | None = None,
    funding_path: Path | None = None,
    output_dir: Path | None = None,
    pair_list: tuple[str, ...] = (),
) -> None:
    paths = strategy_family_sweep_report(
        input_dir=input_dir,
        funding_path=funding_path,
        output_dir=output_dir,
        pair_list=pair_list,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


def strategy_family_matrix_report(
    input_dir: Path | None = None,
    funding_path: Path | None = None,
    output_dir: Path | None = None,
    pair_list: tuple[str, ...] = (),
    max_combo_size: int = 4,
) -> dict[str, Path]:
    root = input_dir or ROOT / "data" / "raw" / "pair_details"
    output = output_dir or ROOT / "reports" / "strategy_family_matrix"
    output.mkdir(parents=True, exist_ok=True)
    selected_pairs = pair_list or DEFAULT_FAMILY_SWEEP_PAIRS
    datasets = _best_requested_pair_datasets(root, funding_path, selected_pairs)
    if not datasets:
        raise SystemExit(f"no requested pair datasets found in {root}")
    return run_family_matrix(datasets, output_dir=output, max_combo_size=max_combo_size)


def print_strategy_family_matrix(
    input_dir: Path | None = None,
    funding_path: Path | None = None,
    output_dir: Path | None = None,
    pair_list: tuple[str, ...] = (),
    max_combo_size: int = 4,
) -> None:
    paths = strategy_family_matrix_report(
        input_dir=input_dir,
        funding_path=funding_path,
        output_dir=output_dir,
        pair_list=pair_list,
        max_combo_size=max_combo_size,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


def research_quantization_report(
    family_matrix_dir: Path | None = None,
    output_dir: Path | None = None,
    top_n: int = 10,
) -> dict[str, Path]:
    source = family_matrix_dir or ROOT / "reports" / "strategy_family_matrix_canonical"
    output = output_dir or source / "quantized"
    return quantize_family_matrix(source, output_dir=output, top_n=top_n)


def print_research_quantization(
    family_matrix_dir: Path | None = None,
    output_dir: Path | None = None,
    top_n: int = 10,
) -> None:
    paths = research_quantization_report(
        family_matrix_dir=family_matrix_dir,
        output_dir=output_dir,
        top_n=top_n,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


def family_failure_attribution_report(
    sweep_dir: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    base = sweep_dir or ROOT / "reports" / "strategy_family_sweep"
    output = output_path or base / "strategy_family_failure_attribution.csv"
    summary = _read_csv_or_empty(base / "strategy_family_sweep_summary.csv")
    if summary.empty:
        frame = pd.DataFrame(
            [
                {
                    "family": "",
                    "best_strategy": "",
                    "diagnosis": "missing_family_sweep_summary",
                    "next_action": "run strategy-family-sweep first",
                }
            ]
        )
        _write_csv_atomic(frame, output)
        return frame
    for column, default in (
        ("production_eligible", False),
        ("preferred_eligible", False),
        ("evaluated_runs", 0),
        ("passing_pairs", 0),
        ("total_trades", 0),
        ("median_profit_factor", 0.0),
        ("median_sharpe", 0.0),
        ("worst_drawdown", 0.0),
        ("acceptance_reason", ""),
        ("preferred_reason", ""),
    ):
        if column not in summary.columns:
            summary[column] = default

    rows: list[dict[str, object]] = []
    for family, group in summary.groupby("family", dropna=False):
        ranked = group.sort_values(
            [
                "production_eligible",
                "preferred_eligible",
                "passing_pairs",
                "median_sharpe",
                "median_profit_factor",
                "total_trades",
                "worst_drawdown",
            ],
            ascending=[False, False, False, False, False, False, True],
        ).reset_index(drop=True)
        best = ranked.iloc[0]
        blocker_counts = _family_sweep_blocker_counts(group)
        diagnosis = _family_sweep_diagnosis(best, blocker_counts)
        rows.append(
            {
                "family": family,
                "best_strategy": best.get("strategy_name", ""),
                "strategies_in_family": len(group),
                "evaluated_runs_best_strategy": int(
                    pd.to_numeric(pd.Series([best.get("evaluated_runs")]), errors="coerce")
                    .fillna(0)
                    .iloc[0]
                ),
                "best_passing_pairs": int(
                    pd.to_numeric(pd.Series([best.get("passing_pairs")]), errors="coerce")
                    .fillna(0)
                    .iloc[0]
                ),
                "best_total_trades": int(
                    pd.to_numeric(pd.Series([best.get("total_trades")]), errors="coerce")
                    .fillna(0)
                    .iloc[0]
                ),
                "best_median_profit_factor": float(
                    pd.to_numeric(pd.Series([best.get("median_profit_factor")]), errors="coerce")
                    .fillna(0.0)
                    .iloc[0]
                ),
                "best_median_sharpe": float(
                    pd.to_numeric(pd.Series([best.get("median_sharpe")]), errors="coerce")
                    .fillna(0.0)
                    .iloc[0]
                ),
                "best_worst_drawdown": float(
                    pd.to_numeric(pd.Series([best.get("worst_drawdown")]), errors="coerce")
                    .fillna(0.0)
                    .iloc[0]
                ),
                "strategies_blocked_by_passing_pairs": blocker_counts.get("passing_pairs", 0),
                "strategies_blocked_by_total_trades": blocker_counts.get("total_trades", 0),
                "strategies_blocked_by_median_profit_factor": blocker_counts.get(
                    "median_profit_factor", 0
                ),
                "strategies_blocked_by_median_sharpe": blocker_counts.get("median_sharpe", 0),
                "strategies_blocked_by_worst_drawdown": blocker_counts.get("worst_drawdown", 0),
                "strategies_blocked_by_no_evaluated_runs": blocker_counts.get(
                    "no_evaluated_runs", 0
                ),
                "top_blockers": ";".join(
                    f"{name}:{count}" for name, count in _sorted_counter_items(blocker_counts)[:5]
                ),
                "diagnosis": diagnosis,
                "next_action": _strategy_failure_next_action(diagnosis),
            }
        )
    frame = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "best_passing_pairs",
                "best_median_sharpe",
                "best_median_profit_factor",
                "best_total_trades",
                "best_worst_drawdown",
            ],
            ascending=[False, False, False, False, True],
        )
        .reset_index(drop=True)
    )
    _write_csv_atomic(frame, output)
    return frame


def _family_sweep_blocker_counts(group: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in ("acceptance_reason", "preferred_reason"):
        if column not in group.columns:
            continue
        for value in group[column].fillna("").astype(str):
            for item in value.split(";"):
                key = _normalize_family_sweep_blocker(item)
                if key:
                    counts[key] = counts.get(key, 0) + 1
    return counts


def _normalize_family_sweep_blocker(value: str) -> str:
    text = value.strip()
    if not text or text in {"passed", "not_production_eligible"}:
        return ""
    if text.startswith("passing_pairs<"):
        return "passing_pairs"
    if text.startswith("total_trades<"):
        return "total_trades"
    if text.startswith("median_profit_factor<"):
        return "median_profit_factor"
    if text.startswith("median_sharpe<"):
        return "median_sharpe"
    if text.startswith("worst_drawdown>"):
        return "worst_drawdown"
    if text.startswith("no_evaluated_runs"):
        return "no_evaluated_runs"
    return text


def _sorted_counter_items(counts: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _family_sweep_diagnosis(best: pd.Series, blocker_counts: dict[str, int]) -> str:
    best_passing_pairs = int(
        pd.to_numeric(pd.Series([best.get("passing_pairs")]), errors="coerce").fillna(0).iloc[0]
    )
    best_total_trades = int(
        pd.to_numeric(pd.Series([best.get("total_trades")]), errors="coerce").fillna(0).iloc[0]
    )
    best_pf = float(
        pd.to_numeric(pd.Series([best.get("median_profit_factor")]), errors="coerce")
        .fillna(0.0)
        .iloc[0]
    )
    best_sharpe = float(
        pd.to_numeric(pd.Series([best.get("median_sharpe")]), errors="coerce").fillna(0.0).iloc[0]
    )
    best_dd = float(
        pd.to_numeric(pd.Series([best.get("worst_drawdown")]), errors="coerce").fillna(0.0).iloc[0]
    )
    if blocker_counts.get("no_evaluated_runs", 0) > 0 and best_total_trades == 0:
        return "no_evaluated_runs"
    if best_passing_pairs == 0 and best_total_trades < 10:
        return "too_few_trades_and_no_passing_pairs"
    if best_passing_pairs == 0:
        return "no_passing_pairs"
    if best_pf < 1.8:
        return "profit_factor_below_gate"
    if best_sharpe < 1.2:
        return "sharpe_below_gate"
    if best_dd > 0.15:
        return "drawdown_above_gate"
    return "acceptance_failed_unknown"


def _strategy_family_failure_notes(sweep_dir: Path | None = None) -> str:
    base = sweep_dir or ROOT / "reports" / "strategy_family_sweep"
    frame = family_failure_attribution_report(
        base, base / "strategy_family_failure_attribution.csv"
    )
    if frame.empty:
        return "# Strategy Family Failure Attribution\n\nNo attribution data available.\n"
    lines = [
        "# Strategy Family Failure Attribution",
        "",
        "## Dominant Family Failures",
        "",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"- `{row['family']}` -> `{row['diagnosis']}` "
            f"(best_strategy=`{row['best_strategy']}`, blockers=`{row['top_blockers']}`)"
        )
    return "\n".join(lines) + "\n"


def print_strategy_family_failure_attribution(
    sweep_dir: Path | None = None,
    output_path: Path | None = None,
) -> None:
    output = (
        output_path
        or (sweep_dir or ROOT / "reports" / "strategy_family_sweep")
        / "strategy_family_failure_attribution.csv"
    )
    frame = family_failure_attribution_report(sweep_dir, output)
    print(frame.to_string(index=False))
    print(f"strategy_family_failure_attribution: {output}")


def _enrich_datasets_with_funding(
    datasets: list[PairDataset], funding_path: Path | None
) -> list[PairDataset]:
    if funding_path is None:
        return datasets
    funding = _load_funding_rows(funding_path)
    normalized = normalize_funding_rows(funding)
    return [enrich_pair_dataset_with_funding(dataset, normalized) for dataset in datasets]


def _load_funding_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"funding file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for key in ("historicalFunding", "funding", "data", "rows", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return pd.DataFrame(value)
            return pd.DataFrame([payload])
    raise SystemExit(f"unsupported funding file type: {path}")


def export_dydx_funding_payload(
    input_path: Path, output_path: Path | None = None, market: str | None = None
) -> Path:
    if not input_path.exists():
        raise SystemExit(f"dYdX funding payload not found: {input_path}")
    rows = _dydx_funding_rows_from_path(input_path, market)
    normalized = normalize_funding_rows(rows)
    if normalized.empty:
        raise SystemExit(f"no funding rows found in {input_path}")
    output = output_path or ROOT / "data" / "processed" / "dydx_funding.csv"
    _write_csv_atomic(normalized, output)
    return output


def fetch_dydx_funding(markets: list[str], output_path: Path | None = None) -> Path:
    if not markets:
        raise SystemExit(
            "fetch-dydx-funding requires --market, with comma-separated markets allowed"
        )
    config = DydxNetworkConfig.paper_testnet_from_env()
    adapter = build_dydx_indexer_adapter(config)
    if adapter is None:
        raise SystemExit(
            "dYdX indexer adapter is not available; install/wire the official v4 client first"
        )
    rows: list[dict[str, object]] = []
    for market in markets:
        try:
            payload = adapter.funding(market)
        except Exception as exc:
            raise SystemExit(f"failed to fetch dYdX funding for {market}: {exc}") from exc
        rows.extend(funding_rows_from_dydx_payload(payload, market=market))
    normalized = normalize_funding_rows(rows)
    if normalized.empty:
        raise SystemExit(f"no funding rows returned for markets: {','.join(markets)}")
    output = output_path or ROOT / "data" / "processed" / "dydx_funding.csv"
    _write_csv_atomic(normalized, output)
    return output


def print_fetch_dydx_funding(market: str | None, output_path: Path | None = None) -> None:
    markets = _market_args(market)
    path = fetch_dydx_funding(markets, output_path)
    print(f"dydx_funding_csv: {path}")


def _market_args(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def rerun_p2_acceptance_evidence(
    input_dir: Path | None = None,
    funding_path: Path | None = None,
) -> dict[str, Path]:
    input_dir = input_dir or ROOT / "data" / "raw" / "pair_details"
    resolved_funding_path = funding_path or ROOT / "data" / "processed" / "dydx_funding.csv"
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    run_pair_detail_experiments(input_dir=input_dir, funding_path=resolved_funding_path)
    strategy_acceptance_checklist_report(reports / "strategy_acceptance_checklist.csv")
    strategy_failure_attribution_report(reports / "strategy_failure_attribution.csv")
    research_unblock_plan_report(reports / "research_unblock_plan.csv")
    readiness = priority_readiness_report(reports / "priority_readiness.csv")
    paper_execution_preflight_report(reports / "paper_execution_preflight.csv")
    priority_gap_test_report(readiness, reports / "priority_gap_test.csv")

    return {
        "experiment_results": reports / "experiment_results.csv",
        "acceptance_report": _acceptance_report_path(),
        "strategy_acceptance_checklist": reports / "strategy_acceptance_checklist.csv",
        "strategy_failure_attribution": reports / "strategy_failure_attribution.csv",
        "research_unblock_plan": reports / "research_unblock_plan.csv",
        "priority_readiness": reports / "priority_readiness.csv",
        "priority_gap_test": reports / "priority_gap_test.csv",
        "paper_execution_preflight": reports / "paper_execution_preflight.csv",
    }


def _dydx_funding_rows_from_path(
    input_path: Path, market: str | None = None
) -> list[dict[str, object]]:
    if input_path.is_dir():
        rows: list[dict[str, object]] = []
        paths = sorted(path for path in input_path.glob("*.json") if path.is_file())
        if not paths:
            raise SystemExit(f"no JSON funding payloads found in {input_path}")
        for path in paths:
            rows.extend(
                _dydx_funding_rows_from_file(path, market or _market_from_funding_filename(path))
            )
        return rows
    return _dydx_funding_rows_from_file(
        input_path, market or _market_from_funding_filename(input_path)
    )


def _dydx_funding_rows_from_file(
    input_path: Path, market: str | None = None
) -> list[dict[str, object]]:
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"input is not valid JSON: {input_path}: {exc}") from exc
    return funding_rows_from_dydx_payload(payload, market=market)


def _market_from_funding_filename(path: Path) -> str | None:
    stem = re.sub(
        r"(?i)(?:^funding[_-]?|[_-]?funding$|[_-]?historical$|[_-]?history$)", "", path.stem
    )
    normalized = stem.replace("_", "-").upper()
    match = re.search(r"([A-Z0-9]+-USD)", normalized)
    return match.group(1) if match else None


def funding_coverage_report(
    funding_path: Path,
    pairs: list[str] | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    funding = _load_funding_rows(funding_path)
    selected_pairs = pairs or _pairs_from_latest_experiment_results()
    if not selected_pairs:
        raise SystemExit("no pairs supplied and no experiment_results.csv pairs found")
    frame = funding_coverage_for_pairs(selected_pairs, funding)
    output = output_path or ROOT / "reports" / "funding_coverage.csv"
    _write_csv_atomic(frame, output)
    return frame


def funding_requirements_report(
    pairs: list[str] | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    selected_pairs = pairs or _pairs_from_latest_experiment_results()
    if not selected_pairs:
        raise SystemExit("no pairs supplied and no experiment_results.csv pairs found")
    frame = funding_market_requirements(selected_pairs)
    output = output_path or ROOT / "reports" / "funding_requirements.csv"
    _write_csv_atomic(frame, output)
    return frame


def print_funding_requirements(pair: str | None = None, output_path: Path | None = None) -> None:
    pairs = [pair] if pair else None
    output = output_path or ROOT / "reports" / "funding_requirements.csv"
    frame = funding_requirements_report(pairs, output)
    print(frame.to_string(index=False))
    valid_rows = frame[frame.get("valid", pd.Series(dtype=bool)).fillna(False).astype(bool)]
    required_markets = _semicolon_values(valid_rows.get("required_markets", pd.Series(dtype=str)))
    invalid_pairs = sorted(
        str(value) for value in frame.loc[~frame.index.isin(valid_rows.index), "pair"].dropna()
    )
    print(f"funding_required_markets: {';'.join(required_markets) if required_markets else 'none'}")
    print(
        f"fetch_dydx_funding_market_arg: {','.join(required_markets) if required_markets else 'none'}"
    )
    print(f"funding_invalid_pairs: {';'.join(invalid_pairs) if invalid_pairs else 'none'}")
    print(f"funding_requirements: {output}")


def funding_template_report(
    pairs: list[str] | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    requirements = funding_requirements_report(pairs)
    valid_rows = requirements[
        requirements.get("valid", pd.Series(dtype=bool)).fillna(False).astype(bool)
    ]
    markets = _semicolon_values(valid_rows.get("required_markets", pd.Series(dtype=str)))
    frame = pd.DataFrame(
        [{"market": market, "timestamp": "", "funding_bps": ""} for market in markets],
        columns=["market", "timestamp", "funding_bps"],
    )
    output = output_path or ROOT / "data" / "processed" / "dydx_funding_template.csv"
    _write_csv_atomic(frame, output)
    return frame


def funding_template_check_report(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    source = input_path or ROOT / "data" / "processed" / "dydx_funding_template.csv"
    output = output_path or ROOT / "reports" / "funding_template_check.csv"
    if not source.exists():
        frame = pd.DataFrame(
            [
                {
                    "path": str(source),
                    "rows": 0,
                    "ready_rows": 0,
                    "blocked_rows": 0,
                    "required_markets": ";".join(_funding_required_markets()),
                    "ready_markets": "",
                    "missing_markets": ";".join(_funding_required_markets()),
                    "missing_columns": ";".join(FUNDING_TEMPLATE_REQUIRED_COLUMNS),
                    "invalid_rows": "",
                    "ready_to_import": False,
                    "next_action": "create funding template and fill real funding_bps values",
                }
            ]
        )
        _write_csv_atomic(frame, output)
        return frame
    try:
        data = pd.read_csv(source, dtype=str).fillna("")
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        data = pd.DataFrame()
    missing_columns, ready_indices, invalid_rows = _funding_template_validation(data)
    normalized = (
        normalize_funding_rows(data.loc[ready_indices])
        if ready_indices and not missing_columns
        else pd.DataFrame()
    )
    required_markets = _funding_required_markets()
    ready_markets = (
        sorted(normalized["market"].dropna().astype(str).unique()) if not normalized.empty else []
    )
    missing_markets = sorted(set(required_markets).difference(ready_markets))
    ready_to_import = bool(
        ready_indices and not invalid_rows and not missing_columns and not missing_markets
    )
    frame = pd.DataFrame(
        [
            {
                "path": str(source),
                "rows": len(data),
                "ready_rows": len(ready_indices),
                "blocked_rows": max(len(data) - len(ready_indices), 0)
                if not missing_columns
                else len(data),
                "required_markets": ";".join(required_markets),
                "ready_markets": ";".join(ready_markets),
                "missing_markets": ";".join(missing_markets),
                "missing_columns": ";".join(missing_columns),
                "invalid_rows": ";".join(invalid_rows),
                "ready_to_import": ready_to_import,
                "next_action": "import funding template to data/processed/dydx_funding.csv"
                if ready_to_import
                else "fill required markets with numeric funding_bps values",
            }
        ]
    )
    _write_csv_atomic(frame, output)
    return frame


def import_funding_template(
    input_path: Path | None = None,
    output_path: Path | None = None,
    report_path: Path | None = None,
) -> pd.DataFrame:
    source = input_path or ROOT / "data" / "processed" / "dydx_funding_template.csv"
    funding_output = output_path or ROOT / "data" / "processed" / "dydx_funding.csv"
    report_output = report_path or ROOT / "reports" / "funding_template_import_report.csv"
    check = funding_template_check_report(source, ROOT / "reports" / "funding_template_check.csv")
    row = check.iloc[0] if not check.empty else pd.Series(dtype=object)
    if not bool(row.get("ready_to_import", False)):
        frame = pd.DataFrame(
            [
                {
                    "path": str(source),
                    "imported_rows": 0,
                    "funding_output": str(funding_output),
                    "status": "blocked",
                    "blocker": row.get("invalid_rows", "")
                    or row.get("missing_markets", "")
                    or row.get("missing_columns", ""),
                    "next_action": row.get("next_action", "fill required funding template rows"),
                }
            ]
        )
        _write_csv_atomic(frame, report_output)
        return frame
    data = pd.read_csv(source, dtype=str).fillna("")
    _, ready_indices, _ = _funding_template_validation(data)
    normalized = normalize_funding_rows(data.loc[ready_indices])
    _write_csv_atomic(normalized, funding_output)
    frame = pd.DataFrame(
        [
            {
                "path": str(source),
                "imported_rows": len(normalized),
                "funding_output": str(funding_output),
                "status": "imported",
                "blocker": "",
                "next_action": "run funding-coverage, then funded-research-spine",
            }
        ]
    )
    _write_csv_atomic(frame, report_output)
    return frame


def print_funding_template(pair: str | None = None, output_path: Path | None = None) -> None:
    pairs = [pair] if pair else None
    output = output_path or ROOT / "data" / "processed" / "dydx_funding_template.csv"
    frame = funding_template_report(pairs, output)
    print(frame.to_string(index=False))
    print(f"funding_template_rows: {len(frame)}")
    print(f"funding_template: {output}")


def print_funding_template_check(
    input_path: Path | None = None, output_path: Path | None = None
) -> None:
    output = output_path or ROOT / "reports" / "funding_template_check.csv"
    frame = funding_template_check_report(input_path, output)
    print(frame.to_string(index=False))
    print(f"funding_template_check: {output}")


def print_import_funding_template(
    input_path: Path | None = None, output_path: Path | None = None
) -> None:
    frame = import_funding_template(input_path=input_path, output_path=output_path)
    report = ROOT / "reports" / "funding_template_import_report.csv"
    print(frame.to_string(index=False))
    print(f"funding_template_import_report: {report}")


def print_funding_coverage(
    funding_path: Path | None, pair: str | None = None, output_path: Path | None = None
) -> None:
    if funding_path is None:
        raise SystemExit("funding-coverage requires --funding-path")
    pairs = [pair] if pair else None
    output = output_path or ROOT / "reports" / "funding_coverage.csv"
    frame = funding_coverage_report(funding_path, pairs, output)
    print(frame.to_string(index=False))
    if not frame.empty:
        ready_pairs = (
            int(frame["ready"].fillna(False).astype(bool).sum()) if "ready" in frame.columns else 0
        )
        required_markets = _semicolon_values(frame.get("required_markets", pd.Series(dtype=str)))
        missing_markets = _semicolon_values(frame.get("missing_markets", pd.Series(dtype=str)))
        print(f"funding_pairs_ready: {ready_pairs}/{len(frame)}")
        print(
            f"funding_required_markets: {';'.join(required_markets) if required_markets else 'none'}"
        )
        print(
            f"funding_missing_markets: {';'.join(missing_markets) if missing_markets else 'none'}"
        )
    print(f"funding_coverage: {output}")


def _semicolon_values(series: pd.Series) -> list[str]:
    values: set[str] = set()
    for item in series.dropna().astype(str):
        for value in item.split(";"):
            cleaned = value.strip()
            if cleaned:
                values.add(cleaned)
    return sorted(values)


def _funding_required_markets() -> list[str]:
    try:
        requirements = funding_requirements_report()
    except SystemExit:
        return []
    valid_rows = requirements[
        requirements.get("valid", pd.Series(dtype=bool)).fillna(False).astype(bool)
    ]
    return _semicolon_values(valid_rows.get("required_markets", pd.Series(dtype=str)))


def _funding_template_validation(frame: pd.DataFrame) -> tuple[list[str], list[int], list[str]]:
    missing_columns = [
        column for column in FUNDING_TEMPLATE_REQUIRED_COLUMNS if column not in frame.columns
    ]
    invalid_rows: list[str] = []
    ready_indices: list[int] = []
    if missing_columns or frame.empty:
        return missing_columns, ready_indices, invalid_rows
    for index, row in frame.iterrows():
        missing = [
            column
            for column in FUNDING_TEMPLATE_REQUIRED_COLUMNS
            if str(row.get(column, "")).strip() == ""
        ]
        invalid = []
        market = str(row.get("market", "")).strip()
        if market:
            try:
                normalize_funding_rows(pd.DataFrame([{"market": market, "funding_bps": 0.0}]))
            except Exception:
                invalid.append("market")
        funding_value = str(row.get("funding_bps", "")).strip()
        if funding_value:
            try:
                float(funding_value)
            except ValueError:
                invalid.append("funding_bps")
        if missing or invalid:
            invalid_rows.append(
                f"row_{index + 2}[missing={'+'.join(missing) or 'none'},invalid={'+'.join(invalid) or 'none'}]"
            )
        else:
            ready_indices.append(int(index))
    return missing_columns, ready_indices, invalid_rows


def _pairs_from_latest_experiment_results() -> list[str]:
    results = _read_csv_or_empty(ROOT / "reports" / "experiment_results.csv")
    if results.empty or "pair" not in results.columns:
        return []
    return sorted(str(pair) for pair in results["pair"].dropna().unique())


def research_spine(
    input_dir: Path | None = None,
    require_two_leg: bool = True,
    funding_path: Path | None = None,
) -> pd.DataFrame:
    input_dir = input_dir or ROOT / "data" / "raw" / "pair_details"
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    objective_status, objective_detail = _project_objective_spine_status()
    rows.append(
        _spine_row(step="project_objective", status=objective_status, detail=objective_detail)
    )

    try:
        paths = write_pair_detail_reports(input_dir, reports)
        snapshots = load_pair_detail_snapshots(input_dir)
        rows.append(
            _spine_row(
                step="ingest_pair_details",
                status="completed",
                detail=f"snapshots={len(snapshots)};reports={','.join(paths)}",
            )
        )
    except Exception as exc:
        rows.append(_spine_row(step="ingest_pair_details", status="failed", detail=str(exc)))
        frame = pd.DataFrame(rows)
        _write_csv_atomic(frame, reports / "research_spine.csv")
        return frame

    readiness = priority_readiness_report()
    rows.append(
        _spine_row(
            step="priority_readiness", status="completed", detail="reports/priority_readiness.csv"
        )
    )

    gates = readiness.set_index("gate") if not readiness.empty else pd.DataFrame()
    history_gate = _gate_ready(gates, "pair_detail_history")
    two_leg_gate = _gate_ready(gates, "pair_detail_two_leg_execution_history")
    if not history_gate:
        rows.append(
            _spine_row(
                step="run_pair_detail_experiments",
                status="skipped",
                detail=_gate_blocker(gates, "pair_detail_history")
                or "pair_detail_history_not_ready",
            )
        )
    elif require_two_leg and not two_leg_gate:
        rows.append(
            _spine_row(
                step="run_pair_detail_experiments",
                status="skipped",
                detail=_gate_blocker(gates, "pair_detail_two_leg_execution_history")
                or "two_leg_history_not_ready",
            )
        )
    else:
        datasets = datasets_from_pair_detail_snapshots(
            input_dir, require_research_usable=require_two_leg
        )
        datasets = _enrich_datasets_with_funding(datasets, funding_path)
        datasets = [
            PairDataset(
                dataset.pair, classify_regimes(dataset.frame, RegimeConfig(preserve_existing=True))
            )
            for dataset in datasets
        ]
        if not datasets:
            rows.append(
                _spine_row(
                    step="run_pair_detail_experiments",
                    status="skipped",
                    detail=f"no experiment-ready pair-detail history datasets found in {input_dir}",
                )
            )
        else:
            write_regime_dataset_report(datasets, reports / "regime_dataset_report.csv")
            harness = _experiment_harness()
            results = harness.run(datasets)
            paths = harness.write_reports(results, reports)
            rows.append(
                _spine_row(
                    step="run_pair_detail_experiments",
                    status="completed",
                    detail=(
                        f"datasets={len(datasets)};experiment_rows={len(results)};"
                        f"funding_path={funding_path or 'none'};reports={','.join(paths)}"
                    ),
                )
            )
            priority_readiness_report()
            rows.append(
                _spine_row(
                    step="priority_readiness_after_experiments",
                    status="completed",
                    detail="reports/priority_readiness.csv",
                )
            )

    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, reports / "research_spine.csv")
    return frame


def print_research_spine(
    input_dir: Path | None = None,
    require_two_leg: bool = True,
    funding_path: Path | None = None,
) -> None:
    frame = research_spine(
        input_dir=input_dir, require_two_leg=require_two_leg, funding_path=funding_path
    )
    print(frame.to_string(index=False))
    print(f"research_spine_report: {ROOT / 'reports' / 'research_spine.csv'}")


def _spine_row(step: str, status: str, detail: str) -> dict[str, object]:
    return {"step": step, "status": status, "detail": detail}


def funded_research_spine(
    funding_path: Path | None,
    input_dir: Path | None = None,
    require_two_leg: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "funded_research_spine.csv"
    rows: list[dict[str, object]] = []
    if funding_path is None:
        rows.append(
            _spine_row(
                step="funding_coverage",
                status="blocked",
                detail="missing_funding_path",
            )
        )
        rows.append(
            _spine_row(step="research_spine", status="skipped", detail="funding_coverage_not_ready")
        )
        frame = pd.DataFrame(rows)
        _write_csv_atomic(frame, output)
        return frame
    try:
        coverage = funding_coverage_report(
            funding_path, output_path=reports / "funding_coverage.csv"
        )
    except SystemExit as exc:
        rows.append(_spine_row(step="funding_coverage", status="blocked", detail=str(exc)))
        rows.append(
            _spine_row(step="research_spine", status="skipped", detail="funding_coverage_not_ready")
        )
        frame = pd.DataFrame(rows)
        _write_csv_atomic(frame, output)
        return frame
    ready_pairs = int(coverage.get("ready", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    total_pairs = len(coverage)
    blocked_pairs = [
        str(row.get("pair", ""))
        for _, row in coverage.iterrows()
        if not bool(row.get("ready", False))
    ]
    if ready_pairs < total_pairs or total_pairs == 0:
        missing_markets = _semicolon_values(coverage.get("missing_markets", pd.Series(dtype=str)))
        rows.append(
            _spine_row(
                step="funding_coverage",
                status="blocked",
                detail=(
                    f"ready_pairs={ready_pairs}/{total_pairs};"
                    f"blocked_pairs={';'.join(blocked_pairs) if blocked_pairs else 'none'};"
                    f"missing_markets={';'.join(missing_markets) if missing_markets else 'none'}"
                ),
            )
        )
        rows.append(
            _spine_row(step="research_spine", status="skipped", detail="funding_coverage_not_ready")
        )
        frame = pd.DataFrame(rows)
        _write_csv_atomic(frame, output)
        return frame
    rows.append(
        _spine_row(
            step="funding_coverage",
            status="completed",
            detail=f"ready_pairs={ready_pairs}/{total_pairs};funding_path={funding_path}",
        )
    )
    spine = research_spine(
        input_dir=input_dir, require_two_leg=require_two_leg, funding_path=funding_path
    )
    spine_status = (
        "completed" if not spine.empty and not (spine["status"] == "failed").any() else "failed"
    )
    refreshed_coverage = funding_coverage_report(
        funding_path, output_path=reports / "funding_coverage.csv"
    )
    refreshed_ready_pairs = int(
        refreshed_coverage.get("ready", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
    )
    refreshed_total_pairs = len(refreshed_coverage)
    rows.append(
        _spine_row(
            step="research_spine",
            status=spine_status,
            detail=(
                "reports/research_spine.csv;"
                f"post_research_funding_ready_pairs={refreshed_ready_pairs}/{refreshed_total_pairs}"
            ),
        )
    )
    acceptance = strategy_acceptance_checklist_report(reports / "strategy_acceptance_checklist.csv")
    ready_steps = int(
        acceptance.get("ready", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
    )
    rows.append(
        _spine_row(
            step="strategy_acceptance_checklist",
            status="completed",
            detail=f"ready_steps={ready_steps}/{len(acceptance)};reports/strategy_acceptance_checklist.csv",
        )
    )
    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, output)
    return frame


def print_funded_research_spine(
    funding_path: Path | None,
    input_dir: Path | None = None,
    require_two_leg: bool = True,
    output_path: Path | None = None,
) -> None:
    output = output_path or ROOT / "reports" / "funded_research_spine.csv"
    frame = funded_research_spine(
        funding_path, input_dir=input_dir, require_two_leg=require_two_leg, output_path=output
    )
    print(frame.to_string(index=False))
    print(f"funded_research_spine_report: {output}")


def _gate_ready(gates: pd.DataFrame, gate: str) -> bool:
    if gates.empty or gate not in gates.index:
        return False
    return bool(gates.loc[gate, "ready"])


def _gate_blocker(gates: pd.DataFrame, gate: str) -> str:
    if gates.empty or gate not in gates.index:
        return ""
    return str(gates.loc[gate, "blocker"])


def check_live_config(endpoint_specs: list[str] | None = None) -> None:
    endpoints = _cli_endpoint_specs(endpoint_specs)
    config = CryptoWizardsLiveConfig.from_env(endpoints=endpoints)
    missing = config.missing_requirements()
    if missing:
        print(f"Crypto Wizards live config missing: {', '.join(missing)}")
    else:
        print(f"Crypto Wizards live config ready: {len(config.endpoints)} endpoint(s)")


def diagnose_crypto_wizards(
    endpoint_specs: list[str] | None = None,
    output_path: Path | None = None,
) -> None:
    endpoints = _cli_endpoint_specs(endpoint_specs)
    config = CryptoWizardsLiveConfig.from_env(endpoints=endpoints)
    missing = config.missing_requirements()
    print(f"base_url_present: {bool(config.base_url)}")
    print(f"api_key_present: {bool(config.api_key)}")
    print(f"endpoint_count: {len(config.endpoints)}")
    if missing:
        print(f"missing: {','.join(missing)}")
        return
    extractor = CryptoWizardsExtractor.from_live_config(config, ROOT / "data" / "raw")
    diagnostics = extractor.diagnose_all(config.endpoints)
    frame = pd.DataFrame([asdict(diagnostic) for diagnostic in diagnostics])
    print(frame.to_string(index=False))
    output = output_path or ROOT / "reports" / "crypto_wizards_diagnostic.csv"
    _write_csv_atomic(frame, output)
    print(f"diagnostic_report: {output}")


def crypto_wizards_min5_request_template_report(
    *,
    asset_x: str | None = None,
    asset_y: str | None = None,
    priority: str = "Sharpe",
    cw_strategy: str = "Spread",
    exchange: str = "Dydx",
    period: int = 320,
    spread_type: str = "Static",
    roll_w: int = 42,
    asset: str | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    rows = official_min5_request_rows(
        symbol_1=asset_x,
        symbol_2=asset_y,
        priority=priority,
        strategy=cw_strategy,
        exchange=exchange,
        interval="Min5",
        period=period,
        spread_type=spread_type,
        roll_w=roll_w,
        asset=asset,
    )
    frame = pd.DataFrame(rows)
    output = output_path or ROOT / "reports" / "crypto_wizards_min5_api_requests.csv"
    _write_csv_atomic(frame, output)
    return frame


def print_crypto_wizards_min5_request_template(
    *,
    asset_x: str | None = None,
    asset_y: str | None = None,
    priority: str = "Sharpe",
    cw_strategy: str = "Spread",
    exchange: str = "Dydx",
    period: int = 320,
    spread_type: str = "Static",
    roll_w: int = 42,
    asset: str | None = None,
    output_path: Path | None = None,
) -> None:
    frame = crypto_wizards_min5_request_template_report(
        asset_x=asset_x,
        asset_y=asset_y,
        priority=priority,
        cw_strategy=cw_strategy,
        exchange=exchange,
        period=period,
        spread_type=spread_type,
        roll_w=roll_w,
        asset=asset,
        output_path=output_path,
    )
    output = output_path or ROOT / "reports" / "crypto_wizards_min5_api_requests.csv"
    print(frame[["request_name", "url", "save_as", "notes"]].to_string(index=False))
    print(f"crypto_wizards_min5_api_requests: {output}")


def crawl_crypto_wizards(endpoint_specs: list[str] | None = None) -> None:
    endpoints = _cli_endpoint_specs(endpoint_specs)
    config = CryptoWizardsLiveConfig.from_env(endpoints=endpoints)
    extractor = CryptoWizardsExtractor.from_live_config(config, ROOT / "data" / "raw")
    try:
        payloads = extractor.fetch_all(config.endpoints)
    except CryptoWizardsFetchError as exc:
        raise SystemExit(f"Crypto Wizards crawl failed: {exc}") from exc
    output = ROOT / "docs" / "crypto_wizards_live_field_dictionary.csv"
    output.parent.mkdir(exist_ok=True)
    CryptoWizardsExtractor.write_discovered_fields(payloads, output)
    print(f"archived {len(payloads)} Crypto Wizards endpoint response(s)")
    print(f"field_dictionary: {output}")


def crawl_crypto_wizards_min5(
    *,
    max_pairs: int,
    priority: str,
    cw_strategy: str,
    exchange: str,
    period: int,
    spread_type: str,
    roll_w: int,
    asset: str | None,
    run_research: bool = False,
    output_dir: Path | None = None,
) -> list[Path]:
    pair_dir = output_dir or ROOT / "data" / "raw" / "pair_details"
    api_key = os.getenv("CRYPTO_WIZARDS_API_KEY")
    try:
        paths = crawl_prescanned_zscores_histories(
            api_key=api_key,
            output_dir=pair_dir,
            max_pairs=max_pairs,
            priority=priority,
            strategy=cw_strategy,
            exchange=exchange,
            interval="Min5",
            period=period,
            spread_type=spread_type,
            roll_w=roll_w,
            asset=asset,
        )
    except CryptoWizardsFetchError as exc:
        raise SystemExit(f"Crypto Wizards Min5 crawl failed: {exc}") from exc
    _print_imported_pair_quality(paths, pair_dir)
    if run_research and paths:
        print("running_pair_detail_experiments: true")
        run_pair_detail_experiments(pair_dir)
        print_strategy_acceptance_checklist()
    return paths


def crawl_crypto_wizards_min5_backtests(
    *,
    max_pairs: int,
    priority: str,
    cw_strategy: str,
    exchange: str,
    period: int,
    spread_type: str,
    roll_w: int,
    asset: str | None,
    run_research: bool = False,
    output_dir: Path | None = None,
) -> list[Path]:
    pair_dir = output_dir or ROOT / "data" / "raw" / "pair_details"
    api_key = os.getenv("CRYPTO_WIZARDS_API_KEY")
    try:
        paths = crawl_prescanned_backtest_histories(
            api_key=api_key,
            output_dir=pair_dir,
            max_pairs=max_pairs,
            priority=priority,
            strategy=cw_strategy,
            exchange=exchange,
            interval="Min5",
            period=period,
            spread_type=spread_type,
            roll_w=roll_w,
            asset=asset,
        )
    except CryptoWizardsFetchError as exc:
        raise SystemExit(f"Crypto Wizards Min5 backtest crawl failed: {exc}") from exc
    _print_imported_pair_quality(
        paths, pair_dir, label="crypto_wizards_min5_backtest_histories_written"
    )
    if run_research and paths:
        print("running_pair_detail_experiments: true")
        run_pair_detail_experiments(pair_dir)
        print_strategy_acceptance_checklist()
    return paths


def _print_imported_pair_quality(
    paths: list[Path],
    pair_dir: Path,
    *,
    label: str = "crypto_wizards_min5_histories_written",
) -> None:
    report_paths = write_pair_detail_reports(pair_dir, ROOT / "reports")
    quality = pd.DataFrame(
        pair_detail_quality_report(pair_dir), columns=PAIR_DETAIL_QUALITY_COLUMNS
    )
    imported_quality = (
        quality[quality["path"].isin({str(path) for path in paths})]
        if not quality.empty
        else quality
    )
    print(f"{label}: {len(paths)}")
    for path in paths:
        print(f"pair_history: {path}")
    if not imported_quality.empty:
        print(
            imported_quality[
                [
                    "pair",
                    "interval",
                    "history_rows",
                    "research_usable",
                    "execution_usable",
                    "quality_blockers",
                ]
            ].to_string(index=False)
        )
    print(f"pair_detail_quality_report: {report_paths['quality']}")


def import_crypto_wizards_zscores_history(
    input_path: Path,
    *,
    asset_x: str,
    asset_y: str,
    exchange: str,
    interval: str,
    period: int,
    spread_type: str,
    roll_w: int,
    output_dir: Path | None = None,
    run_research: bool = False,
) -> Path:
    if not input_path.exists():
        raise SystemExit(f"Crypto Wizards zscores JSON not found: {input_path}")
    try:
        response = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"input is not valid JSON: {input_path}: {exc}") from exc
    if not isinstance(response, dict):
        raise SystemExit(
            f"input JSON must be an object response from /v1beta/zscores: {input_path}"
        )

    pair_dir = output_dir or ROOT / "data" / "raw" / "pair_details"
    request = CryptoWizardsHistoryRequest(
        symbol_1=asset_x,
        symbol_2=asset_y,
        exchange=exchange,
        interval=interval,
        period=period,
        spread_type=spread_type,
        roll_w=roll_w,
    )
    path = write_zscores_pair_payload(request, response, pair_dir)
    report_paths = write_pair_detail_reports(pair_dir, ROOT / "reports")
    quality = pd.DataFrame(
        pair_detail_quality_report(pair_dir), columns=PAIR_DETAIL_QUALITY_COLUMNS
    )
    imported_quality = quality[quality["path"] == str(path)] if not quality.empty else quality
    print(f"imported_crypto_wizards_zscores_history: {path}")
    if not imported_quality.empty:
        print(
            imported_quality[
                [
                    "pair",
                    "interval",
                    "history_rows",
                    "research_usable",
                    "execution_usable",
                    "quality_blockers",
                ]
            ].to_string(index=False)
        )
    print(f"pair_detail_quality_report: {report_paths['quality']}")
    if run_research:
        print("running_pair_detail_experiments: true")
        run_pair_detail_experiments(pair_dir)
        print_strategy_acceptance_checklist()
    return path


def import_crypto_wizards_backtest_history(
    input_path: Path,
    *,
    asset_x: str,
    asset_y: str,
    exchange: str,
    interval: str,
    period: int,
    spread_type: str,
    roll_w: int,
    output_dir: Path | None = None,
    run_research: bool = False,
) -> Path:
    if not input_path.exists():
        raise SystemExit(f"Crypto Wizards backtest JSON not found: {input_path}")
    try:
        response = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"input is not valid JSON: {input_path}: {exc}") from exc
    if not isinstance(response, dict):
        raise SystemExit(
            f"input JSON must be an object response from /v1beta/backtest: {input_path}"
        )

    pair_dir = output_dir or ROOT / "data" / "raw" / "pair_details"
    request = CryptoWizardsHistoryRequest(
        symbol_1=asset_x,
        symbol_2=asset_y,
        exchange=exchange,
        interval=interval,
        period=period,
        spread_type=spread_type,
        roll_w=roll_w,
    )
    path = write_backtest_pair_payload(request, response, pair_dir)
    _print_imported_pair_quality(
        [path], pair_dir, label="imported_crypto_wizards_backtest_histories"
    )
    if run_research:
        print("running_pair_detail_experiments: true")
        run_pair_detail_experiments(pair_dir)
        print_strategy_acceptance_checklist()
    return path


def verify_crypto_wizards_live_artifacts() -> None:
    raw_dir = ROOT / "data" / "raw"
    live_dictionary = ROOT / "docs" / "crypto_wizards_live_field_dictionary.csv"
    raw_payloads = sorted(
        path
        for path in raw_dir.glob("*.json")
        if not path.name.startswith("crypto_wizards_pair_metrics_sample")
    )

    print(f"live_payload_count: {len(raw_payloads)}")
    for path in raw_payloads:
        print(f"live_payload: {path}")
    print(f"live_field_dictionary_exists: {live_dictionary.exists()}")
    if live_dictionary.exists():
        frame = pd.read_csv(live_dictionary)
        print(f"live_field_dictionary_rows: {len(frame)}")
        print(f"live_field_dictionary_columns: {','.join(frame.columns)}")

    if not raw_payloads:
        raise SystemExit("missing live Crypto Wizards raw payloads in data/raw")
    if not live_dictionary.exists():
        raise SystemExit("missing docs/crypto_wizards_live_field_dictionary.csv")
    report = crypto_wizards_live_coverage_report()
    ecm_fields = report[
        (report["type"] == "field") & report["name"].isin({"ecm_x", "ecm_y", "ecm_strength"})
    ]
    missing_ecm = sorted(ecm_fields.loc[~ecm_fields["present_in_live"], "name"].astype(str))
    print(f"live_ecm_fields_present: {not missing_ecm}")
    print(f"live_ecm_missing_fields: {','.join(missing_ecm) if missing_ecm else 'none'}")
    print(f"coverage_report: {ROOT / 'reports' / 'crypto_wizards_live_coverage.csv'}")


def crypto_wizards_live_coverage_report() -> pd.DataFrame:
    live_dictionary = ROOT / "docs" / "crypto_wizards_live_field_dictionary.csv"
    if not live_dictionary.exists():
        raise SystemExit("missing docs/crypto_wizards_live_field_dictionary.csv")

    live_fields = pd.read_csv(live_dictionary)
    normalized_live, live_sources = _canonical_live_fields(
        live_fields["field"].dropna().astype(str)
    )
    pair_detail_fields, pair_detail_sources = _pair_detail_live_fields()
    all_fields = set(normalized_live) | set(pair_detail_fields)
    all_sources = _merge_sources(live_sources, pair_detail_sources)

    rows: list[dict[str, object]] = []
    for row in field_rows():
        field = str(row["name"])
        present = field in all_fields
        rows.append(
            {
                "type": "field",
                "name": field,
                "present_in_live": present,
                "missing_fields": "" if present else field,
                "prescanned_present": field in normalized_live,
                "pair_detail_present": field in pair_detail_fields,
                "source_fields": ";".join(all_sources.get(field, [])),
                "notes": row["description"],
            }
        )

    from quant_platform.strategies import STRATEGY_REQUIRED_COLUMNS

    for strategy in STRATEGIES:
        required = STRATEGY_REQUIRED_COLUMNS.get(strategy.id, set())
        missing = sorted(column for column in required if column not in all_fields)
        rows.append(
            {
                "type": "strategy",
                "name": f"{strategy.id}: {strategy.name}",
                "present_in_live": not missing,
                "missing_fields": ";".join(missing),
                "prescanned_present": False,
                "pair_detail_present": bool(required)
                and set(required).issubset(pair_detail_fields),
                "source_fields": "",
                "notes": strategy.family,
            }
        )

    report = pd.DataFrame(rows)
    output = ROOT / "reports" / "crypto_wizards_live_coverage.csv"
    _write_csv_atomic(report, output)
    return report


def write_crypto_wizards_live_coverage_report() -> None:
    report = crypto_wizards_live_coverage_report()
    output = ROOT / "reports" / "crypto_wizards_live_coverage.csv"
    print(f"coverage_report: {output}")
    print(report.to_string(index=False))


def _normalize_live_field(field: str) -> str:
    normalized = field.replace("[]", "").replace("[].", "").strip(".")
    return normalized.split(".")[-1]


def _canonical_live_fields(fields: pd.Series) -> tuple[set[str], dict[str, list[str]]]:
    alias_to_canonical: dict[str, str] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        alias_to_canonical[snake_case(canonical)] = canonical
        for alias in aliases:
            alias_to_canonical[snake_case(alias)] = canonical

    canonical_fields: set[str] = set()
    sources: dict[str, list[str]] = {}
    for field in fields:
        leaf = _normalize_live_field(str(field))
        canonical = alias_to_canonical.get(snake_case(leaf), snake_case(leaf))
        canonical_fields.add(canonical)
        sources.setdefault(canonical, []).append(str(field))

    if {"u1_given_u2", "u2_given_u1"}.issubset(canonical_fields):
        canonical_fields.update({"conditional_probability_distortion", "conditional_probabilities"})
        source_pair = sources.get("u1_given_u2", []) + sources.get("u2_given_u1", [])
        sources.setdefault("conditional_probability_distortion", source_pair)
        sources.setdefault("conditional_probabilities", source_pair)
    if "conditional_probability_distortion" in canonical_fields:
        canonical_fields.add("conditional_probabilities")
        sources.setdefault(
            "conditional_probabilities", sources.get("conditional_probability_distortion", [])
        )
    if "conditional_probabilities" in canonical_fields:
        canonical_fields.add("conditional_probability_distortion")
        sources.setdefault(
            "conditional_probability_distortion", sources.get("conditional_probabilities", [])
        )
    return canonical_fields, sources


def _pair_detail_live_fields() -> tuple[set[str], dict[str, list[str]]]:
    fields: set[str] = set()
    sources: dict[str, list[str]] = {}
    pair_detail_dir = ROOT / "data" / "raw" / "pair_details"
    snapshots = load_pair_detail_snapshots(pair_detail_dir) if pair_detail_dir.exists() else []
    for snapshot in snapshots:
        row = snapshot.to_row()
        for field, value in row.items():
            if value is None:
                continue
            canonical = _pair_detail_field_to_canonical(field)
            fields.add(canonical)
            sources.setdefault(canonical, []).append(f"pair_detail:{snapshot.pair}:{field}")
        if snapshot.ecm_x_available:
            fields.add("ecm_x")
            sources.setdefault("ecm_x", []).append(ECM_FIELD_SOURCE["ecm_x"])
        if snapshot.ecm_y_available:
            fields.add("ecm_y")
            sources.setdefault("ecm_y", []).append(ECM_FIELD_SOURCE["ecm_y"])
        if snapshot.ecm_strength_available:
            fields.add("ecm_strength")
            sources.setdefault("ecm_strength", []).append(ECM_FIELD_SOURCE["ecm_strength"])
        if snapshot.u1_given_u2 is not None and snapshot.u2_given_u1 is not None:
            fields.update({"conditional_probabilities", "conditional_probability_distortion"})
            source_pair = [
                f"pair_detail:{snapshot.pair}:u1_given_u2",
                f"pair_detail:{snapshot.pair}:u2_given_u1",
            ]
            sources.setdefault("conditional_probabilities", source_pair)
            sources.setdefault("conditional_probability_distortion", source_pair)
    return fields, sources


def _pair_detail_field_to_canonical(field: str) -> str:
    mapping = {
        "returns_total": "returns_total",
        "annual_return": "returns_total",
        "drawdown": "drawdown",
        "corr_copula": "copula",
        "closed_trades": "closed",
    }
    return mapping.get(field, field)


def _merge_sources(*source_maps: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for source_map in source_maps:
        for field, values in source_map.items():
            merged.setdefault(field, [])
            merged[field].extend(values)
    return {field: list(dict.fromkeys(values)) for field, values in merged.items()}


def import_crypto_wizards_payload(input_path: Path, endpoint_name: str = "manual") -> None:
    if not input_path.exists():
        raise SystemExit(f"input JSON not found: {input_path}")
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"input is not valid JSON: {input_path}: {exc}") from exc

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_payload = raw_dir / f"{endpoint_name}.json"
    atomic_copy_file(input_path, output_payload)

    dictionary_path = ROOT / "docs" / "crypto_wizards_live_field_dictionary.csv"
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    CryptoWizardsExtractor.write_discovered_fields({endpoint_name: payload}, dictionary_path)
    print(f"imported_payload: {output_payload}")
    print(f"field_dictionary: {dictionary_path}")


def check_dydx_config() -> None:
    config = DydxNetworkConfig.paper_testnet_from_env()
    order_client, order_adapter_error = _load_dydx_order_client_adapter()
    adapter_contract = validate_dydx_order_client_adapter()
    order_adapter_loaded = order_client is not None and not order_adapter_error
    report = dydx_readiness_report(
        config=config,
        order_client_wired=order_adapter_loaded,
        indexer_adapter_wired=build_dydx_indexer_adapter(config) is not None,
    )
    if order_adapter_error:
        report["blockers"] = [
            *report["blockers"],
            f"invalid_dydx_order_client_adapter:{order_adapter_error}",
        ]
    elif (
        adapter_contract["configured"]
        and adapter_contract["valid"]
        and not adapter_contract["exchange_submission_capable"]
    ):
        report["blockers"] = [*report["blockers"], "record_only_dydx_order_client_adapter"]
    report["ready_for_paper_submission"] = len(report["blockers"]) == 0
    for key, value in report.items():
        if isinstance(value, list):
            value = ",".join(value) if value else "none"
        print(f"{key}: {value}")


def _load_dydx_order_client_adapter():
    try:
        return build_dydx_order_client_adapter(), ""
    except Exception as exc:
        return None, str(exc)


def _load_venue_order_client_adapter(venue: str):
    try:
        return build_venue_order_client_adapter(venue), ""
    except Exception as exc:
        return None, str(exc)


def dydx_order_adapter_contract_report(output_path: Path | None = None) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "dydx_order_adapter_contract.csv"
    row = validate_dydx_order_client_adapter()
    frame = pd.DataFrame([row])
    _write_csv_atomic(frame, output)
    return frame


def print_dydx_order_adapter_contract(output_path: Path | None = None) -> None:
    output = output_path or ROOT / "reports" / "dydx_order_adapter_contract.csv"
    frame = dydx_order_adapter_contract_report(output)
    print(frame.to_string(index=False))
    print(f"dydx_order_adapter_contract: {output}")


def dydx_execution_checklist_report(
    output_path: Path | None = None,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    effective_root = root or ROOT
    with _local_env_for_reports(effective_root):
        return _dydx_execution_checklist_report(output_path, root=effective_root)


def _dydx_execution_checklist_report(
    output_path: Path | None = None,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    effective_root = root or ROOT
    reports = effective_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "dydx_execution_checklist.csv"
    config = DydxNetworkConfig.paper_testnet_from_env()
    indexer_wired = build_dydx_indexer_adapter(config) is not None
    order_client, order_adapter_error = _load_dydx_order_client_adapter()
    adapter_contract = validate_dydx_order_client_adapter()
    adapter_submission_capable = bool(
        adapter_contract["valid"] and adapter_contract["exchange_submission_capable"]
    )
    readiness = dydx_readiness_report(
        config=config,
        order_client_wired=order_client is not None and adapter_submission_capable,
        indexer_adapter_wired=indexer_wired,
    )
    acceptance_path = _acceptance_report_path(effective_root)
    strategy_ready = False
    two_leg_passing_pairs = 0
    production_eligible = 0
    if acceptance_path.exists():
        acceptance = _augmented_acceptance_frame(reports, root=effective_root)
        production_eligible = int(
            acceptance.get("production_eligible", pd.Series(dtype=bool)).fillna(False).sum()
        )
        two_leg_passing_pairs = _max_int_column(acceptance, "two_leg_passing_pairs")
        strategy_ready = production_eligible > 0 and two_leg_passing_pairs > 0

    rows = [
        _execution_check_row(
            step="indexer_market_data",
            ready=bool(readiness["dydx_indexer_adapter_wired"]),
            blocker=""
            if readiness["dydx_indexer_adapter_wired"]
            else "missing_dydx_indexer_adapter",
            evidence=f"rest_indexer={readiness['rest_indexer']};websocket_indexer={readiness['websocket_indexer']}",
            next_action="read market/funding data from testnet indexer"
            if readiness["dydx_indexer_adapter_wired"]
            else "install/wire official dYdX v4 indexer client",
        ),
        _execution_check_row(
            step="testnet_credentials",
            ready=bool(readiness["wallet_address_present"] and readiness["private_key_present"]),
            blocker=_join_missing(
                [
                    ("missing_wallet_address", not readiness["wallet_address_present"]),
                    ("missing_private_key", not readiness["private_key_present"]),
                ]
            ),
            evidence=(
                f"wallet_address_present={readiness['wallet_address_present']};"
                f"private_key_present={readiness['private_key_present']}"
            ),
            next_action="keep secrets in .env.local only"
            if readiness["wallet_address_present"] and readiness["private_key_present"]
            else "set DYDX_TESTNET_WALLET_ADDRESS and DYDX_TESTNET_PRIVATE_KEY in .env.local",
        ),
        _execution_check_row(
            step="dydx_sdk",
            ready=bool(readiness["dydx_v4_client_installed"]),
            blocker="" if readiness["dydx_v4_client_installed"] else "missing_dydx_v4_client",
            evidence=f"dydx_v4_client_installed={readiness['dydx_v4_client_installed']}",
            next_action="continue adapter wiring"
            if readiness["dydx_v4_client_installed"]
            else 'install optional dependency with pip install -e ".[dev,dydx]"',
        ),
        _execution_check_row(
            step="submit_flag",
            ready=bool(readiness["submit_orders"]),
            blocker="" if readiness["submit_orders"] else "submit_orders_false",
            evidence=f"submit_orders={readiness['submit_orders']}",
            next_action="submit flag is enabled; order adapter gate still applies"
            if readiness["submit_orders"]
            else "leave DYDX_TESTNET_SUBMIT_ORDERS=false until research and adapter gates pass",
        ),
        _execution_check_row(
            step="order_client_adapter",
            ready=bool(readiness["dydx_order_client_adapter_wired"])
            and not order_adapter_error
            and bool(adapter_contract["valid"])
            and bool(adapter_contract["exchange_submission_capable"]),
            blocker=""
            if (
                readiness["dydx_order_client_adapter_wired"]
                and not order_adapter_error
                and adapter_contract["valid"]
                and adapter_contract["exchange_submission_capable"]
            )
            else (
                "record_only_dydx_order_client_adapter"
                if adapter_contract["valid"] and not adapter_contract["exchange_submission_capable"]
                else (
                    "invalid_dydx_order_client_adapter"
                    if order_adapter_error or adapter_contract["configured"]
                    else "missing_dydx_order_client_adapter"
                )
            ),
            evidence=(
                f"order_adapter={readiness['dydx_order_client_adapter_wired']};"
                f"contract_valid={adapter_contract['valid']};"
                f"signature_accepts_intent_config={adapter_contract['signature_accepts_intent_config']};"
                f"exchange_submission_capable={adapter_contract['exchange_submission_capable']};"
                f"record_only={adapter_contract['record_only']};"
                f"adapter_error={order_adapter_error or adapter_contract['error'] or 'none'}"
            ),
            next_action="authenticated testnet order client is injected"
            if (
                readiness["dydx_order_client_adapter_wired"]
                and not order_adapter_error
                and adapter_contract["valid"]
                and adapter_contract["exchange_submission_capable"]
            )
            else (
                "replace record-only adapter with an authenticated dYdX testnet order adapter"
                if adapter_contract["valid"] and not adapter_contract["exchange_submission_capable"]
                else (
                    "fix DYDX_TESTNET_ORDER_CLIENT_ADAPTER module:object path"
                    if order_adapter_error or adapter_contract["configured"]
                    else "set DYDX_TESTNET_ORDER_CLIENT_ADAPTER to a module:object implementing place_order"
                )
            ),
        ),
        _execution_check_row(
            step="research_acceptance",
            ready=strategy_ready,
            blocker="" if strategy_ready else "no_research_accepted_two_leg_strategy",
            evidence=(
                f"acceptance_report_exists={acceptance_path.exists()};"
                f"production_eligible={production_eligible};two_leg_passing_pairs={two_leg_passing_pairs}"
            ),
            next_action="allow only production-eligible two-leg strategies"
            if strategy_ready
            else "run real two-leg experiments until acceptance gates pass",
        ),
    ]
    adapter_ready = (
        bool(readiness["dydx_order_client_adapter_wired"])
        and not order_adapter_error
        and bool(adapter_contract["valid"])
        and bool(adapter_contract["exchange_submission_capable"])
    )
    paper_ready = bool(readiness["ready_for_paper_submission"]) and adapter_ready and strategy_ready
    blockers = [str(row["blocker"]) for row in rows if str(row["blocker"])]
    rows.append(
        _execution_check_row(
            step="paper_submission_gate",
            ready=paper_ready,
            blocker="" if paper_ready else ";".join(blockers),
            evidence=f"dydx_ready={readiness['ready_for_paper_submission']};strategy_ready={strategy_ready}",
            next_action="paper submission is allowed by current gates"
            if paper_ready
            else "do not submit paper orders",
        )
    )
    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, output)
    return frame


def _venue_cost_model_profile(venue: str) -> tuple[str, bool, str]:
    mapping = {
        "dydx": ("dydx_research_model", True, "dydx_cost_model_ready"),
        "hyperliquid": (
            "hyperliquid_research_model_pending",
            False,
            "cost_model_alignment_pending",
        ),
        "binance": (
            "binance_spot_fee_or_borrow_not_implemented",
            False,
            "cost_model_alignment_pending",
        ),
        "binanceus": (
            "binanceus_spot_fee_or_borrow_not_implemented",
            False,
            "cost_model_alignment_pending",
        ),
        "coinbase": ("coinbase_venue_model_not_wired", False, "cost_model_alignment_pending"),
        "bybit": ("bybit_venue_model_not_wired", False, "cost_model_alignment_pending"),
    }
    normalized = normalize_venue_name(venue)
    profile, ready, next_action = mapping.get(
        normalized, ("unknown_venue_cost_model", False, "wire_exchange_cost_model_before_paper")
    )
    return profile, ready, next_action


def _normalize_report_pair(value: object) -> str:
    return str(value or "").replace("_", "-").replace("/", "-").upper().strip()


def _check_exchange_cost_model_alignment_from_quality(
    results: pd.DataFrame,
    *,
    root: Path | None = None,
) -> dict[str, object]:
    effective_root = root or ROOT
    quality = _read_csv_or_empty(
        effective_root / "reports" / "pair_detail_quality_report.csv"
    )
    if quality.empty:
        return {
            "ready": False,
            "blocker": "pair_detail_quality_report_missing",
            "evidence": "pair_detail_quality_report_missing;must_validate_cost_model_alignment",
            "next_action": "run pair-detail quality report to verify execution model coverage",
            "missing_pairs": "",
        }

    if (
        quality.empty
        or "pair" not in quality.columns
        or (
            "execution_usable" not in quality.columns
            and "research_execution_usable" not in quality.columns
        )
    ):
        return {
            "ready": False,
            "blocker": "pair_detail_quality_report_incomplete",
            "evidence": "pair_detail_quality_report_missing_or_incomplete;required_execution_usable_flags_missing",
            "next_action": "rebuild pair_detail_quality_report.csv via pair_detail quality path",
            "missing_pairs": "",
        }

    tested_pairs = {
        str(pair) for pair in results.get("pair", pd.Series(dtype=str)).dropna().astype(str)
    }
    if not tested_pairs:
        return {
            "ready": True,
            "blocker": "",
            "evidence": "no_pairs_to_validate_cost_model_alignment",
            "next_action": "run experiments first",
            "missing_pairs": "",
        }

    q = quality.copy()
    q["pair_normalized"] = q["pair"].map(_normalize_report_pair)
    tested_normalized = {_normalize_report_pair(pair) for pair in tested_pairs}
    aligned = True
    blockers: list[str] = []
    missing: list[str] = []
    for pair in tested_normalized:
        matched = q[q["pair_normalized"] == pair]
        if matched.empty:
            aligned = False
            blockers.append(f"missing_cost_quality_row:{pair}")
            missing.append(pair)
            continue
        execution_ready = bool(
            matched.get("execution_usable", pd.Series(dtype=bool)).fillna(False).any()
        )
        research_ready = bool(
            matched.get("research_execution_usable", pd.Series(dtype=bool)).fillna(False).any()
        )
        if not execution_ready and not research_ready:
            aligned = False
            blockers.append(f"missing_cost_execution_alignment:{pair}")
            missing.append(pair)

    return {
        "ready": aligned,
        "blocker": ";".join(sorted(set(blockers))),
        "evidence": (
            f"tested_pairs={len(tested_normalized)};"
            f"quality_rows={len(quality)};"
            f"misaligned_pairs={';'.join(missing) or 'none'}"
        ),
        "next_action": (
            "refresh pair_detail_quality_report and fill venue-specific cost/slippage/funding assumptions"
            if blockers
            else "cost-model alignment has venue-specific evidence in quality report"
        ),
        "missing_pairs": ";".join(sorted(set(missing))),
    }


def strategy_acceptance_checklist_report(
    output_path: Path | None = None,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    effective_root = root or ROOT
    reports = effective_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "strategy_acceptance_checklist.csv"
    acceptance_path = _acceptance_report_path(effective_root)
    results_path = reports / "experiment_results.csv"
    funding_coverage_path = reports / "funding_coverage.csv"
    funding_requirements_path = reports / "funding_requirements.csv"
    acceptance = _augmented_acceptance_frame(reports, root=effective_root)
    results = _read_csv_or_empty(results_path)
    funding_coverage = _read_csv_or_empty(funding_coverage_path)

    evaluated_runs = (
        int((results.get("status", pd.Series(dtype=str)) == "evaluated").sum())
        if not results.empty
        else 0
    )
    two_leg_runs = (
        int((results.get("backtest_mode", pd.Series(dtype=str)) == "two_leg").sum())
        if not results.empty
        else 0
    )
    spread_runs = (
        int((results.get("backtest_mode", pd.Series(dtype=str)) == "spread").sum())
        if not results.empty
        else 0
    )
    production_eligible = _sum_bool_column(acceptance, "production_eligible")
    preferred_eligible = _sum_bool_column(acceptance, "preferred_eligible")
    max_two_leg_pairs_tested = _max_int_column(acceptance, "two_leg_pairs_tested")
    max_two_leg_execution_input_pairs = _max_int_column(acceptance, "two_leg_execution_input_pairs")
    max_two_leg_passing_pairs = _max_int_column(acceptance, "two_leg_passing_pairs")
    max_total_trades = _max_int_column(acceptance, "total_trades")
    required_cost_buckets = _required_cost_buckets_from_acceptance(acceptance)
    missing_cost_buckets = sorted(
        required_cost_buckets.difference(_cost_buckets_from_results(results))
    )
    blocker_counts = _acceptance_blocker_counts(acceptance)
    top_blockers = ";".join(f"{name}:{count}" for name, count in blocker_counts[:5])
    two_leg_input_blocker = _two_leg_execution_input_blocker(acceptance)
    missing_two_leg_inputs = _missing_two_leg_inputs_from_acceptance(acceptance)
    funding_missing = "funding_x" in missing_two_leg_inputs or "funding_y" in missing_two_leg_inputs
    funding_requirements = _funding_requirements_for_preflight(
        funding_missing, funding_requirements_path
    )
    funding_preflight = _funding_preflight_status(
        funding_coverage,
        funding_missing,
        funding_coverage_path,
        funding_requirements,
        funding_requirements_path,
    )
    cost_model_alignment = _check_exchange_cost_model_alignment_from_quality(
        results,
        root=effective_root,
    )

    rows = [
        _execution_check_row(
            step="experiment_results",
            ready=evaluated_runs > 0,
            blocker="" if evaluated_runs > 0 else "missing_evaluated_experiments",
            evidence=f"results_exists={results_path.exists()};evaluated_runs={evaluated_runs};spread_runs={spread_runs};two_leg_runs={two_leg_runs}",
            next_action="continue acceptance diagnostics"
            if evaluated_runs > 0
            else "run experiments on real pair-detail history",
        ),
        _execution_check_row(
            step="two_leg_coverage",
            ready=max_two_leg_pairs_tested >= 2 and two_leg_runs > 0,
            blocker=""
            if max_two_leg_pairs_tested >= 2 and two_leg_runs > 0
            else "missing_two_leg_backtests",
            evidence=f"two_leg_runs={two_leg_runs};max_two_leg_pairs_tested={max_two_leg_pairs_tested}",
            next_action="evaluate strategy acceptance on two-leg results"
            if max_two_leg_pairs_tested >= 2 and two_leg_runs > 0
            else "capture price_x/price_y and rerun two-leg experiments",
        ),
        _execution_check_row(
            step="two_leg_execution_assumptions",
            ready=max_two_leg_execution_input_pairs >= 2,
            blocker="" if max_two_leg_execution_input_pairs >= 2 else two_leg_input_blocker,
            evidence=(
                f"max_two_leg_execution_input_pairs={max_two_leg_execution_input_pairs};"
                f"required_inputs={_required_two_leg_inputs_from_acceptance(acceptance)}"
            ),
            next_action="two-leg economics include hedge ratio, beta, and funding inputs"
            if max_two_leg_execution_input_pairs >= 2
            else (
                "fetch/export dYdX funding for required markets, then run funding-coverage"
                if funding_missing and missing_two_leg_inputs.issubset({"funding_x", "funding_y"})
                else "capture or map hedge_ratio, beta, funding_x_bps, and funding_y_bps for each tested pair"
            ),
        ),
        _execution_check_row(
            step="exchange_cost_model_alignment",
            ready=bool(cost_model_alignment["ready"]),
            blocker=str(cost_model_alignment["blocker"]),
            evidence=cost_model_alignment["evidence"],
            next_action=cost_model_alignment["next_action"],
        ),
        _execution_check_row(
            step="funding_preflight",
            ready=funding_preflight["ready"],
            blocker=funding_preflight["blocker"],
            evidence=funding_preflight["evidence"],
            next_action=funding_preflight["next_action"],
        ),
        _execution_check_row(
            step="cost_bucket_coverage",
            ready=not missing_cost_buckets and bool(required_cost_buckets),
            blocker=""
            if not missing_cost_buckets and bool(required_cost_buckets)
            else "missing_required_cost_buckets",
            evidence=(
                f"required_cost_buckets={';'.join(sorted(required_cost_buckets)) or 'unknown'};"
                f"missing_cost_buckets={';'.join(missing_cost_buckets) or 'none'}"
            ),
            next_action="base/stress cost buckets are represented"
            if not missing_cost_buckets and bool(required_cost_buckets)
            else "rerun experiments with required base and stress cost buckets",
        ),
        _execution_check_row(
            step="production_eligibility",
            ready=production_eligible > 0,
            blocker="" if production_eligible > 0 else "no_production_eligible_strategy",
            evidence=(
                f"acceptance_exists={acceptance_path.exists()};strategies={len(acceptance)};"
                f"production_eligible={production_eligible};max_two_leg_passing_pairs={max_two_leg_passing_pairs};"
                f"max_total_trades={max_total_trades};top_blockers={top_blockers or 'none'}"
            ),
            next_action="allow only production eligible strategies into paper planning"
            if production_eligible > 0
            else "resolve top blockers before paper execution",
        ),
        _execution_check_row(
            step="preferred_readiness",
            ready=preferred_eligible > 0,
            blocker="" if preferred_eligible > 0 else "no_preferred_eligible_strategy",
            evidence=f"preferred_eligible={preferred_eligible};top_blockers={top_blockers or 'none'}",
            next_action="preferred deployment criteria have at least one candidate"
            if preferred_eligible > 0
            else "collect more robust trades and improve drawdown/sharpe before production deployment",
        ),
    ]
    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, output)
    return frame


def print_strategy_acceptance_checklist() -> None:
    output = ROOT / "reports" / "strategy_acceptance_checklist.csv"
    frame = strategy_acceptance_checklist_report(output)
    print(frame.to_string(index=False))
    print(f"strategy_acceptance_checklist: {output}")


def strategy_failure_attribution_report(
    output_path: Path | None = None,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    effective_root = root or ROOT
    reports = effective_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "strategy_failure_attribution.csv"
    results = _read_csv_or_empty(reports / "experiment_results.csv")
    acceptance = _augmented_acceptance_frame(reports, root=effective_root)
    if results.empty:
        frame = pd.DataFrame(
            [
                {
                    "strategy_id": "",
                    "strategy_name": "",
                    "family": "",
                    "diagnosis": "missing_experiment_results",
                    "next_action": "run funded-research-spine with real two-leg data",
                }
            ]
        )
        _write_csv_atomic(frame, output)
        return frame

    rows: list[dict[str, object]] = []
    acceptance_by_id = (
        acceptance.set_index("strategy_id")
        if "strategy_id" in acceptance.columns and not acceptance.empty
        else pd.DataFrame()
    )
    for (strategy_id, strategy_name, family), group in results.groupby(
        ["strategy_id", "strategy_name", "family"], dropna=False
    ):
        evaluated = group[group.get("status", pd.Series(dtype=str)) == "evaluated"].copy()
        skipped = group[group.get("status", pd.Series(dtype=str)) != "evaluated"].copy()
        eligible = (
            int(evaluated.get("eligible", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
            if not evaluated.empty
            else 0
        )
        reason_counts = (
            _reason_counts(evaluated.get("reason", pd.Series(dtype=str)))
            if not evaluated.empty
            else []
        )
        skip_counts = (
            _reason_counts(skipped.get("reason", pd.Series(dtype=str))) if not skipped.empty else []
        )
        missing_columns = sorted(
            {
                item
                for reason, _ in skip_counts
                if reason.startswith("missing_columns:")
                for item in reason.replace("missing_columns:", "").split(",")
                if item
            }
        )
        pair_count = (
            int(evaluated.get("pair", pd.Series(dtype=str)).nunique()) if not evaluated.empty else 0
        )
        total_trades = int(
            pd.to_numeric(evaluated.get("trades", pd.Series(dtype=float)), errors="coerce")
            .fillna(0)
            .sum()
        )
        max_trades = (
            int(
                pd.to_numeric(evaluated.get("trades", pd.Series(dtype=float)), errors="coerce")
                .fillna(0)
                .max()
            )
            if not evaluated.empty
            else 0
        )
        median_pf = _median_numeric(evaluated, "profit_factor")
        median_sharpe = _median_numeric(evaluated, "sharpe")
        worst_drawdown = _max_numeric(evaluated, "max_drawdown")
        median_expectancy = _median_numeric(evaluated, "expectancy")
        median_cost_drag = _median_cost_drag(evaluated)
        acceptance_reason = ""
        preferred_reason = ""
        lookup_id = strategy_id if strategy_id in acceptance_by_id.index else str(strategy_id)
        if not acceptance_by_id.empty and lookup_id in acceptance_by_id.index:
            acceptance_row = acceptance_by_id.loc[lookup_id]
            if isinstance(acceptance_row, pd.DataFrame):
                acceptance_row = acceptance_row.iloc[0]
            acceptance_reason = _md_text(acceptance_row.get("acceptance_reason", ""))
            preferred_reason = _md_text(acceptance_row.get("preferred_reason", ""))

        diagnosis = _strategy_failure_diagnosis(
            evaluated_runs=len(evaluated),
            eligible_runs=eligible,
            total_trades=total_trades,
            max_trades=max_trades,
            median_profit_factor=median_pf,
            median_sharpe=median_sharpe,
            median_expectancy=median_expectancy,
            worst_drawdown=worst_drawdown,
            missing_columns=missing_columns,
            acceptance_reason=acceptance_reason,
        )
        rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "family": family,
                "diagnosis": diagnosis,
                "next_action": _strategy_failure_next_action(diagnosis),
                "evaluated_runs": len(evaluated),
                "skipped_runs": len(skipped),
                "pairs_tested": pair_count,
                "eligible_runs": eligible,
                "total_trades": total_trades,
                "max_trades_per_run": max_trades,
                "median_profit_factor": median_pf,
                "median_sharpe": median_sharpe,
                "worst_drawdown": worst_drawdown,
                "median_expectancy": median_expectancy,
                "median_cost_drag": median_cost_drag,
                "top_run_failures": ";".join(
                    f"{reason}:{count}" for reason, count in reason_counts[:5]
                ),
                "top_skip_reasons": ";".join(
                    f"{reason}:{count}" for reason, count in skip_counts[:5]
                ),
                "missing_columns": ";".join(missing_columns),
                "acceptance_reason": acceptance_reason,
                "preferred_reason": preferred_reason,
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["eligible_runs", "total_trades", "median_profit_factor", "median_sharpe"],
        ascending=[False, False, False, False],
    )
    _write_csv_atomic(frame, output)
    return frame


def print_strategy_failure_attribution(output_path: Path | None = None) -> None:
    output = output_path or ROOT / "reports" / "strategy_failure_attribution.csv"
    frame = strategy_failure_attribution_report(output)
    print(frame.to_string(index=False))
    print(f"strategy_failure_attribution: {output}")


def research_unblock_plan_report(
    output_path: Path | None = None,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    effective_root = root or ROOT
    reports = effective_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "research_unblock_plan.csv"
    failures = strategy_failure_attribution_report(
        reports / "strategy_failure_attribution.csv",
        root=effective_root,
    )
    results = _read_csv_or_empty(reports / "experiment_results.csv")
    quality = _read_csv_or_empty(reports / "pair_detail_quality_report.csv")
    acceptance = _augmented_acceptance_frame(reports, root=effective_root)

    rows: list[dict[str, object]] = []
    evaluated = (
        results[results.get("status", pd.Series(dtype=str)) == "evaluated"].copy()
        if not results.empty
        else pd.DataFrame()
    )
    if not evaluated.empty:
        trades = pd.to_numeric(
            evaluated.get("trades", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0)
        observations = pd.to_numeric(
            evaluated.get("observations", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0)
        max_trades = int(trades.max())
        total_trades = int(trades.sum())
        pairs_tested = int(evaluated.get("pair", pd.Series(dtype=str)).nunique())
        best_trade_rows = (
            evaluated.assign(_trades=trades)
            .sort_values(["_trades", "profit_factor", "sharpe"], ascending=[False, False, False])
            .drop_duplicates(["strategy_name", "pair"])
            .head(5)
        )
        best_candidates = ";".join(
            f"{_md_text(row.get('strategy_name', ''))}@{_md_text(row.get('pair', ''))}[trades={int(row.get('_trades', 0))}]"
            for _, row in best_trade_rows.iterrows()
        )
        rows.append(
            {
                "priority": 1,
                "area": "trade_sample_size",
                "blocker": "too_few_completed_trades",
                "evidence": (
                    f"pairs_tested={pairs_tested};total_trades={total_trades};"
                    f"max_trades_per_run={max_trades};max_observations={int(observations.max())}"
                ),
                "target": ">=100 trades minimum and >=250 trades preferred per production candidate",
                "recommended_action": (
                    "run dydx-pair-expansion-plan, collect longer 5-minute histories, and add more candidate pairs before enabling paper trading"
                ),
                "candidate_detail": best_candidates,
                "minimum_history_multiplier_estimate": _history_multiplier(max_trades, 100),
                "preferred_history_multiplier_estimate": _history_multiplier(max_trades, 250),
            }
        )
    threshold_summary = _read_csv_or_empty(reports / "zscore_threshold_sweep_summary.csv")
    if not threshold_summary.empty:
        best_threshold = threshold_summary.copy()
        best_threshold["_max_trades"] = pd.to_numeric(
            best_threshold.get("max_trades", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0)
        best_threshold["_passing_pairs"] = pd.to_numeric(
            best_threshold.get("passing_pairs", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0)
        best_threshold = best_threshold.sort_values(
            ["_passing_pairs", "_max_trades"], ascending=[False, False]
        ).iloc[0]
        passing_pairs = int(best_threshold.get("_passing_pairs", 0))
        max_sweep_trades = int(best_threshold.get("_max_trades", 0))
        rows.append(
            {
                "priority": 1.5,
                "area": "threshold_sensitivity",
                "blocker": "threshold_sweep_has_no_passing_pairs" if passing_pairs == 0 else "",
                "evidence": (
                    f"best_threshold={best_threshold.get('threshold', '')};"
                    f"cost_bucket={best_threshold.get('cost_bucket', '')};"
                    f"max_trades={max_sweep_trades};passing_pairs={passing_pairs};"
                    f"diagnosis={best_threshold.get('diagnosis', '')}"
                ),
                "target": "threshold changes should increase trades without failing PF, Sharpe, drawdown, expectancy, or stress costs",
                "recommended_action": (
                    "do not loosen thresholds as a standalone fix; collect longer histories and add filters/features"
                    if passing_pairs == 0
                    else "inspect passing threshold candidates before considering strategy registration"
                ),
                "candidate_detail": "reports/zscore_threshold_sweep_summary.csv",
                "minimum_history_multiplier_estimate": "",
                "preferred_history_multiplier_estimate": "",
            }
        )

    if not failures.empty and "missing_columns" in failures.columns:
        missing_rows = failures[failures["missing_columns"].fillna("").astype(str) != ""].copy()
        missing_counts: dict[str, dict[str, object]] = {}
        for _, row in missing_rows.iterrows():
            for column in str(row.get("missing_columns", "")).split(";"):
                column = column.strip()
                if not column:
                    continue
                item = missing_counts.setdefault(column, {"strategies": set(), "families": set()})
                item["strategies"].add(_md_text(row.get("strategy_name", "")))
                item["families"].add(_md_text(row.get("family", "")))
        for column, item in sorted(
            missing_counts.items(), key=lambda kv: (-len(kv[1]["strategies"]), kv[0])
        ):
            strategies = sorted(item["strategies"])
            families = sorted(item["families"])
            rows.append(
                {
                    "priority": 2,
                    "area": "missing_feature_coverage",
                    "blocker": f"missing_{column}",
                    "evidence": f"affected_strategies={len(strategies)};families={';'.join(families)}",
                    "target": f"populate {column} for every research-usable 5-minute pair history",
                    "recommended_action": "capture the native Crypto Wizards pair-detail/API field or keep dependent strategies data-blocked",
                    "candidate_detail": ";".join(strategies[:12]),
                    "minimum_history_multiplier_estimate": "",
                    "preferred_history_multiplier_estimate": "",
                }
            )

    if not quality.empty:
        research_usable = int(
            quality.get("research_usable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
        )
        execution_usable = int(
            quality.get("execution_usable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
        )
        rows.append(
            {
                "priority": 3,
                "area": "pair_universe_quality",
                "blocker": "not_enough_execution_usable_histories",
                "evidence": f"research_usable={research_usable};execution_usable={execution_usable};quality_rows={len(quality)}",
                "target": "multiple research-usable and execution-usable pairs with price_x, price_y, hedge ratio, beta, funding, spread, and zscore",
                "recommended_action": "promote stale or incomplete captures only after real dYdX funding and non-stale two-leg candles are present",
                "candidate_detail": _quality_blocker_summary(quality),
                "minimum_history_multiplier_estimate": "",
                "preferred_history_multiplier_estimate": "",
            }
        )

    production_eligible = 0
    if not acceptance.empty and "production_eligible" in acceptance.columns:
        production_eligible = int(
            acceptance["production_eligible"].fillna(False).astype(bool).sum()
        )
    rows.append(
        {
            "priority": 4,
            "area": "paper_trading_gate",
            "blocker": "research_rejected_all_strategies" if production_eligible == 0 else "",
            "evidence": f"production_eligible={production_eligible}",
            "target": "at least one production-eligible two-leg strategy before dYdX paper submission",
            "recommended_action": "keep dYdX submit orders disabled until P2 passes",
            "candidate_detail": "",
            "minimum_history_multiplier_estimate": "",
            "preferred_history_multiplier_estimate": "",
        }
    )

    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, output)
    return frame


def print_research_unblock_plan(output_path: Path | None = None) -> None:
    output = output_path or ROOT / "reports" / "research_unblock_plan.csv"
    frame = research_unblock_plan_report(output)
    print(frame.to_string(index=False))
    print(f"research_unblock_plan: {output}")


def zscore_threshold_sweep_report(
    *,
    input_dir: Path | None = None,
    funding_path: Path | None = None,
    output_path: Path | None = None,
    thresholds: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25),
) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "zscore_threshold_sweep.csv"
    input_dir = input_dir or ROOT / "data" / "raw" / "pair_details"
    funding_path = funding_path or ROOT / "data" / "processed" / "dydx_funding.csv"
    datasets = datasets_from_pair_detail_snapshots(input_dir, require_research_usable=True)
    if funding_path.exists():
        datasets = _enrich_datasets_with_funding(datasets, funding_path)
    datasets = [
        PairDataset(
            dataset.pair, classify_regimes(dataset.frame, RegimeConfig(preserve_existing=True))
        )
        for dataset in datasets
    ]
    rows: list[dict[str, object]] = []
    cost_buckets = ExperimentConfig().cost_buckets
    for dataset in datasets:
        frame = dataset.frame
        if not {"price_x", "price_y", "hedge_ratio", "zscore"}.issubset(frame.columns):
            continue
        for threshold in thresholds:
            signal = zscore_signal(frame, entry=threshold)
            for bucket in cost_buckets:
                result = backtest_two_leg_spread(frame, signal, bucket.cost_model)
                rows.append(
                    {
                        "pair": dataset.pair,
                        "threshold": threshold,
                        "cost_bucket": bucket.name,
                        "trades": result.trades,
                        "profit_factor": result.profit_factor,
                        "expectancy": result.expectancy,
                        "sharpe": result.sharpe,
                        "max_drawdown": result.max_drawdown,
                        "win_rate": result.win_rate,
                        "total_return": result.total_return,
                        "gross_return": result.gross_return,
                        "total_fees": result.total_fees,
                        "total_slippage": result.total_slippage,
                        "total_funding": result.total_funding,
                        "total_execution_risk": result.total_execution_risk,
                        "total_partial_fill_cost": result.total_partial_fill_cost,
                        "observations": len(frame),
                        "passes_trade_gate": result.trades >= 100,
                        "passes_quality_gate": (
                            result.trades >= 100
                            and result.profit_factor >= 1.8
                            and result.sharpe >= 1.2
                            and result.max_drawdown <= 0.15
                            and result.expectancy > 0
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["passes_quality_gate", "passes_trade_gate", "trades", "profit_factor", "sharpe"],
            ascending=[False, False, False, False, False],
        )
    _write_csv_atomic(frame, output)
    summary = zscore_threshold_sweep_summary(frame, reports / "zscore_threshold_sweep_summary.csv")
    return frame


def zscore_threshold_sweep_summary(
    frame: pd.DataFrame, output_path: Path | None = None
) -> pd.DataFrame:
    output = output_path or ROOT / "reports" / "zscore_threshold_sweep_summary.csv"
    if frame.empty:
        summary = pd.DataFrame(
            [
                {
                    "threshold": "",
                    "cost_bucket": "",
                    "pairs": 0,
                    "median_trades": 0,
                    "max_trades": 0,
                    "median_profit_factor": 0.0,
                    "median_sharpe": 0.0,
                    "passing_pairs": 0,
                    "diagnosis": "no_threshold_sweep_rows",
                }
            ]
        )
        _write_csv_atomic(summary, output)
        return summary
    summary = (
        frame.groupby(["threshold", "cost_bucket"], as_index=False)
        .agg(
            pairs=("pair", "nunique"),
            median_trades=("trades", "median"),
            max_trades=("trades", "max"),
            median_profit_factor=("profit_factor", "median"),
            median_sharpe=("sharpe", "median"),
            median_expectancy=("expectancy", "median"),
            worst_drawdown=("max_drawdown", "max"),
            passing_pairs=("passes_quality_gate", "sum"),
        )
        .sort_values(
            ["passing_pairs", "median_trades", "median_profit_factor"],
            ascending=[False, False, False],
        )
    )
    summary["diagnosis"] = summary.apply(_threshold_sweep_diagnosis, axis=1)
    _write_csv_atomic(summary, output)
    return summary


def print_zscore_threshold_sweep(
    input_dir: Path | None = None,
    funding_path: Path | None = None,
    output_path: Path | None = None,
) -> None:
    output = output_path or ROOT / "reports" / "zscore_threshold_sweep.csv"
    frame = zscore_threshold_sweep_report(
        input_dir=input_dir, funding_path=funding_path, output_path=output
    )
    print(frame.to_string(index=False))
    print(f"zscore_threshold_sweep: {output}")
    print(
        f"zscore_threshold_sweep_summary: {ROOT / 'reports' / 'zscore_threshold_sweep_summary.csv'}"
    )


def dydx_pair_expansion_plan_report(
    *,
    max_pairs: int = 10,
    limit: int = 1000,
    indexer_base: str = DEFAULT_INDEXER_BASE,
    indexer_scheme: str = "",
    output_path: Path | None = None,
) -> pd.DataFrame:
    def _iter_hunt_pairs() -> list[tuple[str, str, int, str]]:
        source_path = DYDX_LIVE_MARKET_SELECTOR_CUSTOM_PATH
        if not source_path.exists() and DYDX_PAIR_EXPANSION_HUNT_PATH.exists():
            source_path = DYDX_PAIR_EXPANSION_HUNT_PATH
        if not source_path.exists():
            return []
        hunt = _read_csv_or_empty(source_path)
        if hunt.empty:
            return []
        ranked_rows = []
        for _, row in hunt.iterrows():
            asset_x = str(row.get("asset_x", "")).strip()
            asset_y = str(row.get("asset_y", "")).strip()
            if not asset_x or not asset_y:
                continue
            pair_key = frozenset({_normalize_dydx_market(asset_x), _normalize_dydx_market(asset_y)})
            already_tested = pair_key in tested_pairs
            already_fetched = pair_key in fetched_pairs
            if already_tested or already_fetched:
                continue
            order = row.get("order", "")
            rank = row.get("rank", "")
            hunt_rank = 0
            if str(order).strip().isdigit():
                hunt_rank = int(order)
            elif str(rank).strip().isdigit():
                hunt_rank = int(rank)
            ranked_rows.append((asset_x, asset_y, hunt_rank, "hunt"))
        ranked_rows = sorted(ranked_rows, key=lambda item: item[2])
        return [
            (asset_x, asset_y, 1000 + rank, "hunt") for (asset_x, asset_y, rank, _) in ranked_rows
        ]

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "dydx_pair_expansion_plan.csv"
    tested_pairs = _tested_market_pairs()
    fetched_pairs = _fetched_market_pair_info()
    risky_markets = _stale_market_risk_info()
    covered_markets = _covered_funding_markets()
    unblock = _read_csv_or_empty(reports / "research_unblock_plan.csv")
    sample_note = _trade_sample_note(unblock)

    candidates: list[dict[str, object]] = []
    fresh_candidates: list[dict[str, object]] = []
    all_pairs: list[tuple[str, str, int, str]] = [
        (left, right, order, "default")
        for order, (left, right) in enumerate(DEFAULT_DYDX_EXPANSION_PAIRS)
    ] + _iter_hunt_pairs()

    seen_pairs: set[frozenset[str]] = set()
    unique_pairs: list[tuple[str, str, int, str]] = []
    for asset_x, asset_y, order, source in all_pairs:
        key = frozenset({_normalize_dydx_market(asset_x), _normalize_dydx_market(asset_y)})
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        unique_pairs.append((asset_x, asset_y, order, source))

    for order, (asset_x, asset_y, _, source) in enumerate(
        sorted(unique_pairs, key=lambda item: item[2])
    ):
        left = _normalize_dydx_market(asset_x)
        right = _normalize_dydx_market(asset_y)
        pair_key = frozenset({left, right})
        already_tested = pair_key in tested_pairs
        fetched_info = fetched_pairs.get(pair_key, {})
        already_fetched = bool(fetched_info)
        fresh_candidate = not already_tested and not already_fetched
        risk_reasons = [
            risky_markets[market] for market in (left, right) if market in risky_markets
        ]
        candidate = {
            "order": order,
            "asset_x": left,
            "asset_y": right,
            "pair_key": pair_key,
            "already_tested": already_tested,
            "already_fetched": already_fetched,
            "fresh_candidate": fresh_candidate,
            "fetched_info": fetched_info,
            "market_risk_status": "stale_market_risk" if risk_reasons else "clean",
            "market_risk_reasons": ";".join(risk_reasons),
            "risk_score": 1 if risk_reasons else 0,
        }
        candidates.append(candidate)
        if fresh_candidate:
            fresh_candidates.append(candidate)

    ranked_fresh = sorted(
        fresh_candidates, key=lambda row: (int(row["risk_score"]), int(row["order"]))
    )
    rank_by_key = {
        candidate["pair_key"]: rank for rank, candidate in enumerate(ranked_fresh, start=1)
    }

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        left = str(candidate["asset_x"])
        right = str(candidate["asset_y"])
        pair_key = candidate["pair_key"]
        already_tested = bool(candidate["already_tested"])
        already_fetched = bool(candidate["already_fetched"])
        fresh_candidate = bool(candidate["fresh_candidate"])
        fetched_info = (
            candidate["fetched_info"] if isinstance(candidate["fetched_info"], dict) else {}
        )
        rank = rank_by_key.get(pair_key, "")
        if fresh_candidate and rank and int(rank) > max_pairs:
            continue
        pair_id = _pair_id_from_markets(left, right)
        missing_markets = sorted(
            market for market in (left, right) if market not in covered_markets
        )
        scheme_flag = f" --indexer-scheme {indexer_scheme}" if indexer_scheme else ""
        fetch_command = (
            "PYTHONPATH=src python3 -m quant_platform.cli fetch-dydx-two-leg-data "
            f"--asset-x {left} --asset-y {right} --pair-id {pair_id} --limit {limit} "
            f"--indexer-base {indexer_base}{scheme_flag} --derive-hedge-ratio --run-research"
        )
        template_command = (
            "PYTHONPATH=src python3 -m quant_platform.cli dydx-two-leg-request-template "
            f"--asset-x {left} --asset-y {right} --pair-id {pair_id} --limit {limit} "
            f"--indexer-base {indexer_base}{scheme_flag}"
        )
        rows.append(
            {
                "rank": "" if not fresh_candidate else rank,
                "pair_id": pair_id,
                "asset_x": left,
                "asset_y": right,
                "already_tested": already_tested,
                "already_fetched": already_fetched,
                "quality_status": fetched_info.get("quality_status", ""),
                "quality_blockers": fetched_info.get("quality_blockers", ""),
                "market_risk_status": candidate["market_risk_status"],
                "market_risk_reasons": candidate["market_risk_reasons"],
                "missing_funding_markets": ";".join(missing_markets),
                "fetch_command": fetch_command,
                "request_template_command": template_command,
                "data_goal": "add a new 5-minute two-leg pair with candles, derived hedge ratio/beta, funding, and P2 rerun",
                "sample_size_note": sample_note,
                "notes": "verify dYdX market availability; keep paper trading blocked until strategy acceptance passes",
            }
        )
    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, output)
    return frame


def print_dydx_pair_expansion_plan(
    *,
    max_pairs: int = 10,
    limit: int = 1000,
    indexer_base: str = DEFAULT_INDEXER_BASE,
    indexer_scheme: str = "",
    output_path: Path | None = None,
) -> None:
    output = output_path or ROOT / "reports" / "dydx_pair_expansion_plan.csv"
    frame = dydx_pair_expansion_plan_report(
        max_pairs=max_pairs,
        limit=limit,
        indexer_base=indexer_base,
        indexer_scheme=indexer_scheme,
        output_path=output,
    )
    print(frame.to_string(index=False))
    print(f"dydx_pair_expansion_plan: {output}")


def dydx_long_history_plan_report(
    *,
    pair: str | None = None,
    asset_x: str | None = None,
    asset_y: str | None = None,
    pair_id: str | None = None,
    windows: int = 12,
    limit: int = 1000,
    resolution: str = "5MINS",
    indexer_base: str = DEFAULT_INDEXER_BASE,
    indexer_scheme: str = "",
    to_iso: str | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "dydx_long_history_plan.csv"
    left, right, resolved_pair_id = _resolve_long_history_pair(
        pair=pair, asset_x=asset_x, asset_y=asset_y, pair_id=pair_id
    )
    end_time = _parse_iso_datetime(to_iso) if to_iso else datetime.now(UTC)
    step = _resolution_timedelta(resolution) * limit
    requested_indexer_base = _indexer_base_with_scheme(indexer_base, indexer_scheme)
    rows: list[dict[str, object]] = []
    for window in range(1, windows + 1):
        window_to = end_time - step * (window - 1)
        window_from = window_to - step
        window_dir = (
            ROOT / "data" / "raw" / "dydx_long_history" / resolved_pair_id / f"window_{window:03d}"
        )
        request_rows = dydx_two_leg_request_rows(
            asset_x=left,
            asset_y=right,
            pair_id=resolved_pair_id,
            resolution=resolution,
            limit=limit,
            from_iso=_format_iso_z(window_from),
            to_iso=_format_iso_z(window_to),
            indexer_base=requested_indexer_base,
            output_dir=window_dir,
        )
        for row in request_rows:
            if row.get("method") != "GET" or "candles" not in str(row.get("request_name", "")):
                continue
            rows.append(
                {
                    "window": window,
                    "pair_id": resolved_pair_id,
                    "asset_x": left,
                    "asset_y": right,
                    "resolution": resolution,
                    "limit": limit,
                    "from_iso": _format_iso_z(window_from),
                    "to_iso": _format_iso_z(window_to),
                    **row,
                }
            )
    rows.append(
        {
            "window": "",
            "pair_id": resolved_pair_id,
            "asset_x": left,
            "asset_y": right,
            "resolution": resolution,
            "limit": limit,
            "from_iso": "",
            "to_iso": "",
            "request_name": "long_history_next_step",
            "method": "LOCAL",
            "url": "",
            "curl": "",
            "save_as": str(ROOT / "data" / "raw" / "dydx_long_history" / resolved_pair_id),
            "import_command": (
                "PYTHONPATH=src python3 -m quant_platform.cli run-dydx-long-history "
                f"--asset-x {left} --asset-y {right} --pair-id {resolved_pair_id} "
                f"--windows {windows} --limit {limit} --interval {resolution} "
                "--derive-hedge-ratio --run-research "
                f"{('--indexer-scheme ' + indexer_scheme + ' ') if indexer_scheme else ''}"
                "--research-funding-path data/processed/dydx_funding.csv"
            ),
            "notes": (
                f"{windows} windows x {limit} {resolution} candles targets roughly "
                f"{windows * limit} bars before overlap/deduplication. Current P2 evidence needs longer histories."
            ),
        }
    )
    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, output)
    return frame


def print_dydx_long_history_plan(
    *,
    pair: str | None = None,
    asset_x: str | None = None,
    asset_y: str | None = None,
    pair_id: str | None = None,
    windows: int = 12,
    limit: int = 1000,
    resolution: str = "5MINS",
    indexer_base: str = DEFAULT_INDEXER_BASE,
    indexer_scheme: str = "",
    to_iso: str | None = None,
    output_path: Path | None = None,
) -> None:
    output = output_path or ROOT / "reports" / "dydx_long_history_plan.csv"
    frame = dydx_long_history_plan_report(
        pair=pair,
        asset_x=asset_x,
        asset_y=asset_y,
        pair_id=pair_id,
        windows=windows,
        limit=limit,
        resolution=resolution,
        indexer_base=indexer_base,
        indexer_scheme=indexer_scheme,
        to_iso=to_iso,
        output_path=output,
    )
    print(frame.to_string(index=False))
    print(f"dydx_long_history_plan: {output}")


def dydx_long_history_coverage_report(
    *,
    pair: str | None = None,
    asset_x: str | None = None,
    asset_y: str | None = None,
    pair_id: str | None = None,
    windows: int = 12,
    limit: int = 1000,
    resolution: str = "5MINS",
    indexer_base: str = DEFAULT_INDEXER_BASE,
    indexer_scheme: str = "",
    to_iso: str | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    resolved_pair_id = _resolve_long_history_pair(
        pair=pair, asset_x=asset_x, asset_y=asset_y, pair_id=pair_id
    )[2]
    output = output_path or reports / f"{resolved_pair_id}_dydx_long_history_coverage.csv"
    plan = dydx_long_history_plan_report(
        pair=pair,
        asset_x=asset_x,
        asset_y=asset_y,
        pair_id=pair_id,
        windows=windows,
        limit=limit,
        resolution=resolution,
        indexer_base=indexer_base,
        indexer_scheme=indexer_scheme,
        to_iso=to_iso,
    )
    requests = plan[
        plan.get("method", pd.Series(dtype=str)).astype(str).str.upper() == "GET"
    ].copy()
    if requests.empty:
        raise SystemExit("long-history plan produced no GET rows")
    rows: list[dict[str, object]] = []
    for window, group in requests.groupby("window", dropna=False):
        expected = 0
        existing = 0
        missing_paths: list[str] = []
        present_paths: list[str] = []
        for _, req in group.iterrows():
            save_as = str(req.get("save_as", "")).strip()
            if not save_as:
                continue
            expected += 1
            path = Path(save_as)
            if not path.is_absolute():
                path = ROOT / path
            if path.exists() and path.stat().st_size > 0:
                existing += 1
                present_paths.append(str(path))
            else:
                missing_paths.append(str(path))
        rows.append(
            {
                "pair_id": resolved_pair_id,
                "window": int(window) if pd.notna(window) else "",
                "expected_files": expected,
                "existing_files": existing,
                "missing_files": max(expected - existing, 0),
                "ready": existing == expected and expected > 0,
                "present_paths": ";".join(present_paths),
                "missing_paths": ";".join(missing_paths),
            }
        )
    frame = pd.DataFrame(rows).sort_values("window")
    _write_csv_atomic(frame, output)
    return frame


def print_dydx_long_history_coverage(
    *,
    pair: str | None = None,
    asset_x: str | None = None,
    asset_y: str | None = None,
    pair_id: str | None = None,
    windows: int = 12,
    limit: int = 1000,
    resolution: str = "5MINS",
    indexer_base: str = DEFAULT_INDEXER_BASE,
    indexer_scheme: str = "",
    to_iso: str | None = None,
    output_path: Path | None = None,
) -> None:
    frame = dydx_long_history_coverage_report(
        pair=pair,
        asset_x=asset_x,
        asset_y=asset_y,
        pair_id=pair_id,
        windows=windows,
        limit=limit,
        resolution=resolution,
        indexer_base=indexer_base,
        indexer_scheme=indexer_scheme,
        to_iso=to_iso,
        output_path=output_path,
    )
    ready_windows = int(frame.get("ready", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    print(frame.to_string(index=False))
    print(f"long_history_windows_ready: {ready_windows}/{len(frame)}")
    pair_id_value = str(frame.iloc[0]["pair_id"]) if not frame.empty else "unknown_pair"
    default_output = ROOT / "reports" / f"{pair_id_value}_dydx_long_history_coverage.csv"
    print(f"dydx_long_history_coverage: {output_path or default_output}")


def run_dydx_pair_expansion(
    *,
    max_pairs: int = 1,
    limit: int = 1000,
    indexer_base: str = DEFAULT_INDEXER_BASE,
    indexer_scheme: str = "",
    output_path: Path | None = None,
    run_research: bool = True,
    skip_fetch: bool = False,
    allow_stale_fetch: bool = False,
    pair_ids: tuple[str, ...] = (),
    strategy_ids: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "dydx_pair_expansion_run.csv"
    plan = dydx_pair_expansion_plan_report(
        max_pairs=max_pairs,
        limit=limit,
        indexer_base=indexer_base,
        indexer_scheme=indexer_scheme,
    )
    if pair_ids:
        requested_pairs = {_normalize_pair_for_filter(pair) for pair in pair_ids if pair}
        plan = plan[
            plan["pair_id"]
            .map(lambda value: _normalize_pair_for_filter(str(value)))
            .isin(requested_pairs)
        ]
    tested = pd.Series(
        plan["already_tested"].map(_coerce_bool) if "already_tested" in plan.columns else False,
        index=plan.index,
    )
    fetched = pd.Series(
        plan["already_fetched"].map(_coerce_bool) if "already_fetched" in plan.columns else False,
        index=plan.index,
    )
    fresh = plan[(~tested) & (~fetched)].copy()
    if "rank" in fresh.columns:
        fresh["_rank"] = pd.to_numeric(fresh["rank"], errors="coerce")
        fresh = fresh.sort_values("_rank")
    rows: list[dict[str, object]] = []
    candidates = list(fresh.head(max_pairs).iterrows())
    for index, (_, row) in enumerate(candidates):
        pair_id = _md_text(row.get("pair_id", ""))
        asset_x = _md_text(row.get("asset_x", ""))
        asset_y = _md_text(row.get("asset_y", ""))
        base = {
            "pair_id": pair_id,
            "asset_x": asset_x,
            "asset_y": asset_y,
            "rank": row.get("rank", ""),
            "status": "started",
            "detail": "",
            "pair_history": "",
            "funding_csv": "",
            "funding_coverage": "",
        }
        try:
            paths = fetch_dydx_two_leg_data(
                asset_x=asset_x,
                asset_y=asset_y,
                pair_id=pair_id,
                limit=limit,
                indexer_base=indexer_base,
                allow_stale_fetch=allow_stale_fetch,
                skip_fetch=skip_fetch,
                derive_hedge_ratio=True,
                run_research=run_research,
            )
        except Exception as exc:
            rows.append({**base, "status": "failed", "detail": str(exc)})
            if index < len(candidates) - 1 and not skip_fetch:
                time.sleep(0.8)
            continue
        rows.append(
            {
                **base,
                "status": "completed",
                "detail": "fetched_candles_funding_built_pair_history",
                "pair_history": str(paths.get("pair_history", "")),
                "funding_csv": str(paths.get("funding_csv", "")),
                "funding_coverage": str(paths.get("funding_coverage", "")),
            }
        )
        if index < len(candidates) - 1 and not skip_fetch:
            time.sleep(0.8)
    if not rows:
        rows.append(
            {
                "pair_id": "",
                "asset_x": "",
                "asset_y": "",
                "rank": "",
                "status": "skipped",
                "detail": "no_fresh_ranked_pairs_in_expansion_plan",
                "pair_history": "",
                "funding_csv": "",
                "funding_coverage": "",
            }
        )
    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, output)
    if run_research:
        if pair_ids or strategy_ids:
            if pair_ids:
                pair_filter = tuple(_parse_pair_list(",".join(pair_ids)))
                if strategy_ids is None:
                    strategy_ids = _pair_expansion_recommended_and_all_strategy_ids(
                        ROOT / "data" / "raw" / "pair_details",
                        pair_filter,
                    )
            else:
                pair_filter = tuple(plan.get("pair_id", pd.Series(dtype=str)).astype(str).tolist())
            try:
                run_pair_detail_experiments(
                    input_dir=ROOT / "data" / "raw" / "pair_details",
                    funding_path=ROOT / "data" / "processed" / "dydx_funding.csv",
                    pair_filter=pair_filter,
                    strategy_ids=strategy_ids,
                )
            except SystemExit as exc:
                if "no experiment-ready pair-detail history datasets found" in str(exc):
                    print(
                        "run_dydx_pair_expansion: research experiments skipped "
                        f"(no experiment-ready datasets for pair_filter={pair_filter})"
                    )
                else:
                    raise
        else:
            strategy_failure_attribution_report()
            research_unblock_plan_report()
            priority_readiness_report()
    return frame


def _resolve_research_funding_path(
    research_funding_path: Path | None, funding_path: Path | None
) -> Path | None:
    """Prefer explicitly requested long-history funding path; fallback to legacy alias."""
    return research_funding_path or funding_path


def run_dydx_local_pair_universe(
    *,
    input_dir: Path | None = None,
    pair_output_dir: Path | None = None,
    funding_output_path: Path | None = None,
    zscore_window: int = 7,
    output_path: Path | None = None,
    run_research: bool = True,
) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    manual_dir = input_dir or ROOT / "data" / "raw" / "dydx_manual"
    pair_dir = pair_output_dir or ROOT / "data" / "raw" / "pair_details"
    pair_dir.mkdir(parents=True, exist_ok=True)

    funding_csv = funding_output_path or ROOT / "data" / "processed" / "dydx_funding.csv"
    export_dydx_funding_payload(manual_dir, funding_csv)
    funding_rows = None
    if funding_csv.exists():
        funding_rows = normalize_funding_rows(_load_funding_rows(funding_csv))

    candle_paths = sorted(manual_dir.glob("*_5MINS_candles.json"))
    markets = sorted(
        {_normalize_dydx_market(path.name.split("_5MINS_candles.json")[0]) for path in candle_paths}
    )
    if len(markets) < 2:
        raise SystemExit(f"need at least two 5-minute candle markets in {manual_dir}")

    rows: list[dict[str, object]] = []
    for left, right in combinations(markets, 2):
        pair_id = _pair_id_from_markets(left, right)
        left_path = manual_dir / f"{left}_5MINS_candles.json"
        right_path = manual_dir / f"{right}_5MINS_candles.json"
        output = pair_dir / f"pair_{pair_id}_5mins_dydx_candles_derived_history.json"
        existing = output.exists() and output.stat().st_size > 0
        status = "rebuilt" if existing else "built"
        try:
            build_pair_history_from_candles(
                left_path=left_path,
                right_path=right_path,
                output_path=output,
                pair_id=pair_id,
                asset_x=left,
                asset_y=right,
                hedge_ratio=None,
                beta=None,
                interval="5mins",
                zscore_window=zscore_window,
                funding_path=funding_csv,
                funding_rows=funding_rows,
            )
        except Exception as exc:
            rows.append(
                {
                    "pair_id": pair_id,
                    "asset_x": left,
                    "asset_y": right,
                    "status": "failed",
                    "pair_history": str(output),
                    "funding_csv": str(funding_csv),
                    "detail": str(exc),
                }
            )
            continue
        rows.append(
            {
                "pair_id": pair_id,
                "asset_x": left,
                "asset_y": right,
                "status": status,
                "pair_history": str(output),
                "funding_csv": str(funding_csv),
                "detail": "rebuilt_existing_pair_history" if existing else "",
            }
        )

    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, output_path or reports / "dydx_local_pair_universe_run.csv")

    if run_research:
        research_spine(input_dir=pair_dir, require_two_leg=True, funding_path=funding_csv)
        strategy_acceptance_checklist_report(reports / "strategy_acceptance_checklist.csv")
        priority_readiness_report()
    return frame


def materialize_p2_rerun_subset(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    quality_report_path: Path | None = None,
) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    pair_dir = input_dir or ROOT / "data" / "raw" / "pair_details"
    quality_path = quality_report_path or reports / "pair_detail_quality_report.csv"
    quality = _read_csv_or_empty(quality_path)
    if quality.empty:
        quality_rows = pair_detail_quality_report(pair_dir) if pair_dir.exists() else []
        quality = pd.DataFrame(quality_rows, columns=PAIR_DETAIL_QUALITY_COLUMNS)
        if not quality.empty:
            _write_csv_atomic(quality, quality_path)
    if quality.empty or "research_usable" not in quality.columns or "path" not in quality.columns:
        raise SystemExit(
            f"research-usable quality report is missing required columns: {quality_path}"
        )

    subset_dir = output_dir or ROOT / "work" / "p2_rerun_subset"
    subset_dir.mkdir(parents=True, exist_ok=True)
    for existing in subset_dir.glob("*.json"):
        existing.unlink()

    selected = quality[quality["research_usable"].fillna(False).astype(bool)].copy()
    if selected.empty:
        raise SystemExit(f"no research-usable pair-detail histories found in {quality_path}")
    selected["_execution_usable"] = (
        selected.get("execution_usable", pd.Series(dtype=bool)).fillna(False).astype(bool)
    )
    selected["_history_rows"] = pd.to_numeric(
        selected.get("history_rows", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    selected = selected.sort_values(
        ["pair", "_execution_usable", "_history_rows"], ascending=[True, False, False]
    ).drop_duplicates(["pair"], keep="first")

    rows: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        source_text = str(row.get("path", "")).strip()
        if not source_text:
            continue
        source = Path(source_text)
        if not source.is_absolute():
            source = pair_dir / source
        chosen = source
        replacement = None
        if source.name.endswith("_dydx_candles_derived_history.json"):
            replacement = source.with_name(
                source.name.replace(
                    "_dydx_candles_derived_history.json", "_dydx_long_history_derived_history.json"
                )
            )
            if replacement.exists():
                chosen = replacement
        status = "copied"
        detail = ""
        if not chosen.exists():
            status = "missing"
            detail = "source_missing"
        else:
            target = subset_dir / chosen.name
            atomic_write_bytes(target, chosen.read_bytes())
            detail = (
                "long_history_replacement"
                if replacement is not None and chosen == replacement
                else "quality_report_source"
            )
        rows.append(
            {
                "pair": row.get("pair", ""),
                "selected_path": str(chosen),
                "original_path": str(source),
                "status": status,
                "detail": detail,
                "history_rows": row.get("history_rows", ""),
                "execution_usable": row.get("execution_usable", ""),
            }
        )
    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, reports / "p2_rerun_subset_manifest.csv")
    return frame


def print_materialize_p2_rerun_subset(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    quality_report_path: Path | None = None,
) -> None:
    frame = materialize_p2_rerun_subset(
        input_dir=input_dir, output_dir=output_dir, quality_report_path=quality_report_path
    )
    print(frame.to_string(index=False))
    print(f"p2_rerun_subset: {output_dir or (ROOT / 'work' / 'p2_rerun_subset')}")
    print(f"p2_rerun_subset_manifest: {ROOT / 'reports' / 'p2_rerun_subset_manifest.csv'}")


def print_dydx_local_pair_universe(
    *,
    input_dir: Path | None = None,
    pair_output_dir: Path | None = None,
    funding_output_path: Path | None = None,
    zscore_window: int = 7,
    output_path: Path | None = None,
    run_research: bool = True,
) -> None:
    output = output_path or ROOT / "reports" / "dydx_local_pair_universe_run.csv"
    frame = run_dydx_local_pair_universe(
        input_dir=input_dir,
        pair_output_dir=pair_output_dir,
        funding_output_path=funding_output_path,
        zscore_window=zscore_window,
        output_path=output,
        run_research=run_research,
    )
    print(frame.to_string(index=False))
    print(f"dydx_local_pair_universe: {output}")


def print_run_dydx_pair_expansion(
    *,
    max_pairs: int = 1,
    limit: int = 1000,
    indexer_base: str = DEFAULT_INDEXER_BASE,
    indexer_scheme: str = "",
    output_path: Path | None = None,
    run_research: bool = True,
    skip_fetch: bool = False,
    allow_stale_fetch: bool = False,
    pair_ids: tuple[str, ...] = (),
    strategy_ids: tuple[int, ...] | None = None,
) -> None:
    output = output_path or ROOT / "reports" / "dydx_pair_expansion_run.csv"
    frame = run_dydx_pair_expansion(
        max_pairs=max_pairs,
        limit=limit,
        indexer_base=indexer_base,
        indexer_scheme=indexer_scheme,
        output_path=output,
        run_research=run_research,
        skip_fetch=skip_fetch,
        allow_stale_fetch=allow_stale_fetch,
        pair_ids=pair_ids,
        strategy_ids=strategy_ids,
    )
    print(frame.to_string(index=False))
    print(f"dydx_pair_expansion_run: {output}")


def priority_spine_dashboard_report(
    readiness: pd.DataFrame | None = None,
    output_path: Path | None = None,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    effective_root = root or ROOT
    reports = effective_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "priority_spine_dashboard.csv"
    readiness = (
        readiness
        if readiness is not None
        else priority_readiness_report(root=effective_root)
    )
    gates = readiness.set_index("gate") if not readiness.empty else pd.DataFrame()
    capture = _read_csv_or_empty(reports / "pair_detail_capture_checklist.csv")
    quality = _read_csv_or_empty(reports / "pair_detail_quality_report.csv")
    strategy = _read_csv_or_empty(reports / "strategy_acceptance_checklist.csv")
    research_unblock_path = reports / "research_unblock_plan.csv"
    has_research_unblock = research_unblock_path.exists()
    dydx = _read_csv_or_empty(reports / "dydx_execution_checklist.csv")
    paper = _read_csv_or_empty(reports / "paper_execution_preflight.csv")
    learning = _read_csv_or_empty(reports / "learning_event_summary.csv")
    paper_submission_ready = _checklist_step_ready(paper, "paper_submission_gate")
    paper_submission_blocker = _checklist_step_value(paper, "paper_submission_gate", "blocker")
    paper_submission_next_action = _checklist_step_value(
        paper, "paper_submission_gate", "next_action"
    )
    strategy_dependency_next_action = _checklist_step_value(
        paper, "strategy_acceptance_dependency", "next_action"
    )

    rows = [
        _dashboard_row(
            priority="P1",
            area="crypto_wizards_capture",
            ready=_all_gates_ready(
                gates,
                [
                    "crypto_wizards_live_artifacts",
                    "pair_detail_history",
                    "pair_detail_two_leg_execution_history",
                    "pair_detail_quality",
                    "pair_detail_capture_audit",
                ],
            ),
            blocker=_first_blocker(
                gates,
                [
                    "pair_detail_capture_audit",
                    "pair_detail_quality",
                    "pair_detail_history",
                    "pair_detail_two_leg_execution_history",
                    "crypto_wizards_live_artifacts",
                ],
            ),
            key_metric=f"{_capture_dashboard_metric(capture)};{_capture_quality_dashboard_metric(quality)}",
            source_report="reports/pair_detail_capture_checklist.csv;reports/pair_detail_quality_report.csv",
            next_action=_first_next_action(
                gates,
                [
                    "pair_detail_capture_audit",
                    "pair_detail_quality",
                    "pair_detail_history",
                    "pair_detail_two_leg_execution_history",
                    "crypto_wizards_live_artifacts",
                ],
            ),
        ),
        _dashboard_row(
            priority="P2",
            area="strategy_acceptance",
            ready=_gate_ready_from_index(gates, "strategy_acceptance"),
            blocker=_gate_value(gates, "strategy_acceptance", "blocker"),
            key_metric=_checklist_dashboard_metric(strategy),
            source_report="reports/strategy_acceptance_checklist.csv;reports/research_unblock_plan.csv"
            if has_research_unblock
            else "reports/strategy_acceptance_checklist.csv",
            next_action=_gate_value(gates, "strategy_acceptance", "next_action")
            if has_research_unblock
            else _checklist_first_blocked_next_action(strategy)
            or _gate_value(gates, "strategy_acceptance", "next_action"),
        ),
        _dashboard_row(
            priority="P3",
            area="dydx_testnet_readiness",
            ready=_gate_ready_from_index(gates, "dydx_testnet_readiness"),
            blocker=_gate_value(gates, "dydx_testnet_readiness", "blocker"),
            key_metric=_checklist_dashboard_metric(dydx),
            source_report="reports/dydx_execution_checklist.csv",
            next_action=_gate_value(gates, "dydx_testnet_readiness", "next_action"),
        ),
        _dashboard_row(
            priority="P4",
            area="paper_execution_gate",
            ready=_gate_ready_from_index(gates, "paper_execution_gate")
            if paper_submission_ready is None
            else bool(paper_submission_ready),
            blocker=_gate_value(gates, "paper_execution_gate", "blocker")
            if paper_submission_ready is None or paper_submission_ready
            else paper_submission_blocker,
            key_metric=_checklist_dashboard_metric(paper)
            if not paper.empty
            else _gate_value(gates, "paper_execution_gate", "evidence"),
            source_report="reports/paper_execution_preflight.csv"
            if not paper.empty
            else "reports/priority_readiness.csv",
            next_action=(
                strategy_dependency_next_action
                if not _gate_ready_from_index(gates, "strategy_acceptance")
                and strategy_dependency_next_action
                else paper_submission_next_action
                or _checklist_first_blocked_next_action(paper)
                or _gate_value(gates, "paper_execution_gate", "next_action")
            ),
        ),
        _dashboard_row(
            priority="P5",
            area="learning_event_store",
            ready=_gate_ready_from_index(gates, "learning_event_store"),
            blocker=_gate_value(gates, "learning_event_store", "blocker"),
            key_metric=_learning_dashboard_metric(learning),
            source_report="reports/learning_event_summary.csv",
            next_action=_gate_value(gates, "learning_event_store", "next_action"),
        ),
    ]
    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, output)
    return frame


def print_priority_dashboard() -> None:
    output = ROOT / "reports" / "priority_spine_dashboard.csv"
    readiness = priority_readiness_report(root=ROOT)
    paper_execution_preflight_report(
        ROOT / "reports" / "paper_execution_preflight.csv", root=ROOT
    )
    frame = priority_spine_dashboard_report(readiness, output, root=ROOT)
    print(frame.to_string(index=False))
    print(f"priority_spine_dashboard: {output}")


def priority_runbook(output_path: Path | None = None) -> Path:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "priority_runbook.md"
    readiness = priority_readiness_report(root=ROOT)
    paper_execution_preflight_report(
        reports / "paper_execution_preflight.csv", root=ROOT
    )
    dashboard = priority_spine_dashboard_report(
        readiness, reports / "priority_spine_dashboard.csv", root=ROOT
    )
    gap_test = priority_gap_test_report(
        readiness, reports / "priority_gap_test.csv", root=ROOT
    )
    actions = priority_action_plan(readiness, reports / "priority_action_plan.csv")

    lines = [
        "# Priority Spine Runbook",
        "",
        "Generated from the current P1-P5 readiness reports.",
        "",
        *_project_objective_runbook_lines(),
        "## Current Dashboard",
        "",
        "| Priority | Area | Status | Blocker | Next Action |",
        "|---|---|---|---|---|",
    ]
    for _, row in dashboard.iterrows():
        lines.append(
            "| {priority} | {area} | {status} | {blocker} | {next_action} |".format(
                priority=_md_cell(row.get("priority", "")),
                area=_md_cell(row.get("area", "")),
                status=_md_cell(row.get("status", "")),
                blocker=_md_cell(row.get("blocker", "")),
                next_action=_md_cell(row.get("next_action", "")),
            )
        )

    lines.extend(["", "## Gap Proof Required", ""])
    for _, row in gap_test.iterrows():
        if str(row.get("status", "")) == "pass":
            continue
        lines.extend(
            [
                f"### {_md_text(row.get('priority', ''))}: {_md_text(row.get('area', ''))}",
                f"- Severity: `{_md_text(row.get('severity', ''))}`",
                f"- Current evidence: `{_md_text(row.get('current_evidence', ''))}`",
                f"- Required proof: {_md_text(row.get('required_proof', ''))}",
                f"- Source report: `{_md_text(row.get('source_report', ''))}`",
                f"- Next action: {_md_text(row.get('next_action', ''))}",
                "",
            ]
        )

    lines.extend(["## Ranked Work Queue", ""])
    if actions.empty:
        lines.append("No blocked gates.")
    else:
        lines.extend(
            ["| Rank | Gate | Depends On | Blocker | Command/Action |", "|---:|---|---|---|---|"]
        )
        for index, row in actions.reset_index(drop=True).iterrows():
            lines.append(
                "| {rank} | {gate} | {depends_on} | {blocker} | {next_action} |".format(
                    rank=index + 1,
                    gate=_md_cell(row.get("gate", "")),
                    depends_on=_md_cell(row.get("depends_on", "")),
                    blocker=_md_cell(row.get("blocker", "")),
                    next_action=_md_cell(row.get("next_action", "")),
                )
            )

    lines.extend(
        [
            "",
            "## Operator Commands",
            "",
            "- P0 gap analysis checkpoint: `PYTHONPATH=src python3 -m quant_platform.cli gap-analysis-checklist`",
            "- P1 copy browser capture helper: `./scripts/copy_crypto_wizards_capture_helper.sh`",
            "- P1 capture checklist: `PYTHONPATH=src python3 -m quant_platform.cli pair-detail-capture-checklist`",
            "- P1 browser status after refresh: `await __CW_CAPTURE_STATUS__()`",
            "- P1 browser download after useful status: `await __CW_DOWNLOAD_CAPTURE__()`",
            "- P1 import latest browser download: `PYTHONPATH=src python3 -m quant_platform.cli import-latest-pair-detail-download`",
            "- P1 capture preflight: `PYTHONPATH=src python3 -m quant_platform.cli capture-preflight --json-path /path/to/crypto_wizards_pair_capture.json`",
            "- P2 funding requirements: `PYTHONPATH=src python3 -m quant_platform.cli funding-requirements`",
            "- P2 funding CSV template: `PYTHONPATH=src python3 -m quant_platform.cli funding-template --output-path data/processed/dydx_funding_template.csv`",
            "- P2 funding template check: `PYTHONPATH=src python3 -m quant_platform.cli funding-template-check --input-dir data/processed/dydx_funding_template.csv`",
            "- P2 import funding template: `PYTHONPATH=src python3 -m quant_platform.cli import-funding-template --input-dir data/processed/dydx_funding_template.csv --output-path data/processed/dydx_funding.csv`",
            f"- P2 fetch dYdX funding: `PYTHONPATH=src python3 -m quant_platform.cli fetch-dydx-funding --market {_funding_requirement_market_arg()}`",
            "- P2 funding coverage: `PYTHONPATH=src python3 -m quant_platform.cli funding-coverage --funding-path data/processed/dydx_funding.csv`",
            "- P2 funded research spine: `PYTHONPATH=src python3 -m quant_platform.cli funded-research-spine --funding-path data/processed/dydx_funding.csv`",
            "- P2 strategy acceptance: `PYTHONPATH=src python3 -m quant_platform.cli strategy-acceptance-checklist`",
            "- P2 research unblock plan: `PYTHONPATH=src python3 -m quant_platform.cli research-unblock-plan`",
            "- P2 z-score threshold sweep: `PYTHONPATH=src python3 -m quant_platform.cli zscore-threshold-sweep --funding-path data/processed/dydx_funding.csv`",
            "- P2 dYdX pair expansion plan: `PYTHONPATH=src python3 -m quant_platform.cli dydx-pair-expansion-plan --max-pairs 10 --limit 1000`",
            "- P2 dYdX long-history plan: `PYTHONPATH=src python3 -m quant_platform.cli dydx-long-history-plan --asset-x SOL-USD --asset-y LINK-USD --pair-id sol_link --windows 12 --limit 1000`",
            "- P2 run shell-backed dYdX long-history workflow: `bash scripts/run_dydx_long_history.sh --asset-x SOL-USD --asset-y LINK-USD --pair-id sol_link --windows 12 --limit 1000 --funding-path data/processed/dydx_funding.csv`",
            "- P2 shell-backed strict long-history workflow: `bash scripts/run_dydx_long_history.sh --strict --asset-x SOL-USD --asset-y LINK-USD --pair-id sol_link --windows 12 --limit 1000 --funding-path data/processed/dydx_funding.csv`",
            "- P2 run dYdX long-history workflow: `PYTHONPATH=src python3 -m quant_platform.cli run-dydx-long-history --asset-x SOL-USD --asset-y LINK-USD --pair-id sol_link --windows 12 --limit 1000 --derive-hedge-ratio --run-research --research-funding-path data/processed/dydx_funding.csv`",
            "- P2 build dYdX long-history pair: `PYTHONPATH=src python3 -m quant_platform.cli build-dydx-long-history-pair --asset-x SOL-USD --asset-y LINK-USD --pair-id sol_link --interval 5mins --derive-hedge-ratio --run-research --research-funding-path data/processed/dydx_funding.csv`",
            "- P2 run dYdX pair expansion: `PYTHONPATH=src python3 -m quant_platform.cli run-dydx-pair-expansion --max-pairs 1 --limit 1000 --run-research`",
            "- P3 adapter contract: `PYTHONPATH=src python3 -m quant_platform.cli dydx-order-adapter-contract`",
            "- P3 dYdX readiness: `PYTHONPATH=src python3 -m quant_platform.cli dydx-execution-checklist`",
            "- P4 paper preflight: `PYTHONPATH=src python3 -m quant_platform.cli paper-execution-preflight`",
            "- P4 paper venue preflight: `PYTHONPATH=src python3 -m quant_platform.cli paper-venue-preflight --pair ETH-BTC`",
            "- P5 learning report: `PYTHONPATH=src python3 -m quant_platform.cli learning-report`",
            "- P5 learning outcome template: `PYTHONPATH=src python3 -m quant_platform.cli learning-outcome-template --output-path data/meta_learning/learning_outcome_template.csv`",
            "- P5 learning outcome template check: `PYTHONPATH=src python3 -m quant_platform.cli learning-outcome-template-check --input-dir data/meta_learning/learning_outcome_template.csv`",
            "- P5 import learning outcomes: `PYTHONPATH=src python3 -m quant_platform.cli import-learning-outcomes --input-dir data/meta_learning/learning_outcome_template.csv --output-path reports/learning_outcome_import_report.csv`",
            "- pre-mortem checkpoint: `PYTHONPATH=src python3 -m quant_platform.cli pre-mortem-checklist`",
            "- post-mortem checkpoint: `PYTHONPATH=src python3 -m quant_platform.cli post-mortem-checklist`",
            "- supreme team checkpoint: `PYTHONPATH=src python3 -m quant_platform.cli supreme-team`",
            "- red-team checkpoint: `PYTHONPATH=src python3 -m quant_platform.cli red-team-checklist`",
            "",
        ]
    )
    atomic_write_text(output, "\n".join(lines), encoding="utf-8")
    return output


def _funding_requirement_market_arg() -> str:
    requirements = _read_csv_or_empty(ROOT / "reports" / "funding_requirements.csv")
    markets = _semicolon_values(requirements.get("required_markets", pd.Series(dtype=str)))
    return ",".join(markets) if markets else "ETH-USD,BTC-USD,SOL-USD"


def print_priority_runbook() -> None:
    output = priority_runbook()
    print(f"priority_runbook: {output}")


def paper_execution_preflight_report(
    output_path: Path | None = None,
    readiness: pd.DataFrame | None = None,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    effective_root = root or ROOT
    with _local_env_for_reports(effective_root):
        return _paper_execution_preflight_report(
            output_path=output_path,
            readiness=readiness,
            root=effective_root,
        )


def _paper_execution_preflight_report(
    output_path: Path | None = None,
    readiness: pd.DataFrame | None = None,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    effective_root = root or ROOT
    reports = effective_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "paper_execution_preflight.csv"
    readiness = (
        readiness
        if readiness is not None
        else priority_readiness_report(root=effective_root)
    )
    gates = readiness.set_index("gate") if not readiness.empty else pd.DataFrame()
    strategy = _read_csv_or_empty(reports / "strategy_acceptance_checklist.csv")
    dydx = _read_csv_or_empty(reports / "dydx_execution_checklist.csv")
    venue_preflight = _read_csv_or_empty(reports / "paper_venue_preflight.csv")
    paper_journal = reports / "paper_trading_journal.csv"
    route_candidates = _read_csv_or_empty(reports / "rl" / "base_rl_route_candidates.csv")
    route_markets = _route_markets_from_candidates(route_candidates)
    compatibility = dydx_execution_compatibility_snapshot(
        route_markets,
        root=effective_root,
    )
    account_state = effective_dydx_account_state_snapshot(
        DydxNetworkConfig.paper_testnet_from_env(),
        root=effective_root,
    )
    compatibility_ready = bool(compatibility.get("checked", False)) and not compatibility.get(
        "blocker"
    )
    account_state_ready = bool(account_state.get("checked", False)) and not account_state.get(
        "blocker"
    )

    venue_cost_model_ready = True
    venue_blocker = ""
    venue_evidence = "paper_venue_preflight_missing"
    if not venue_preflight.empty and "cost_model_aligned" in venue_preflight.columns:
        venue_cost_aligned_pairs = int(
            venue_preflight.get("cost_model_aligned", pd.Series(dtype=bool)).map(_coerce_bool).sum()
        )
        venue_rows = len(venue_preflight)
        venue_cost_model_ready = venue_cost_aligned_pairs > 0
        venue_evidence = f"paper_venue_preflight_rows={venue_rows};cost_model_aligned_pairs={venue_cost_aligned_pairs}"
        if not venue_cost_model_ready:
            venue_blocker = "venue_cost_model_alignment_pending"

    strategy_ready = _gate_ready_from_index(gates, "strategy_acceptance")
    dydx_ready = _gate_ready_from_index(gates, "dydx_testnet_readiness")
    paper_ready = _gate_ready_from_index(gates, "paper_execution_gate")
    submission_gate_ready = bool(paper_ready and compatibility_ready and account_state_ready)
    submission_gate_blocker = ""
    if not strategy_ready or not dydx_ready or not paper_ready:
        submission_gate_blocker = "strategy_or_dydx_gate_not_ready"
    elif compatibility.get("blocker"):
        submission_gate_blocker = str(
            compatibility.get("blocker", "execution_compatibility_blocked")
        )
    elif account_state.get("blocker"):
        submission_gate_blocker = str(account_state.get("blocker", "account_state_blocked"))
    strategy_next_action = _checklist_first_blocked_next_action(strategy) or _gate_value(
        gates, "strategy_acceptance", "next_action"
    )
    dydx_next_action = _checklist_first_blocked_next_action(dydx) or _gate_value(
        gates, "dydx_testnet_readiness", "next_action"
    )
    rows = [
        _execution_check_row(
            step="strategy_acceptance_dependency",
            ready=strategy_ready,
            blocker="" if strategy_ready else _gate_value(gates, "strategy_acceptance", "blocker"),
            evidence=_gate_value(gates, "strategy_acceptance", "evidence"),
            next_action=strategy_next_action,
        ),
        _execution_check_row(
            step="dydx_testnet_dependency",
            ready=dydx_ready,
            blocker="" if dydx_ready else _gate_value(gates, "dydx_testnet_readiness", "blocker"),
            evidence=_gate_value(gates, "dydx_testnet_readiness", "evidence"),
            next_action=dydx_next_action,
        ),
        _execution_check_row(
            step="paper_submission_gate",
            ready=submission_gate_ready,
            blocker=submission_gate_blocker,
            evidence=(
                f"{_gate_value(gates, 'paper_execution_gate', 'evidence')};"
                f"compatibility_ready={compatibility_ready};"
                f"account_state_ready={account_state_ready}"
            ),
            next_action=(
                "paper-plan may create and submit research-gated paper orders"
                if submission_gate_ready
                else (
                    "only route paper through markets that have confirmed on exchange"
                    if compatibility.get("blocker")
                    else (
                        "flatten lingering account legs before any new paper submission"
                        if account_state.get("blocker")
                        else "do not submit paper orders until strategy and dYdX dependencies are ready"
                    )
                )
            ),
        ),
        _execution_check_row(
            step="venue_cost_model_readiness",
            ready=bool(venue_cost_model_ready),
            blocker=venue_blocker,
            evidence=venue_evidence,
            next_action=(
                "refresh pair_detail_quality_report and venue cost model mappings"
                if not venue_cost_model_ready
                else "venue cost-model status is ready"
            ),
        ),
        _execution_check_row(
            step="paper_journal",
            ready=paper_journal.exists(),
            blocker="" if paper_journal.exists() else "missing_paper_trading_journal",
            evidence=f"journal_exists={paper_journal.exists()};rows={_csv_row_count(paper_journal)}",
            next_action="paper handoffs are auditable"
            if paper_journal.exists()
            else "paper-plan will create reports/paper_trading_journal.csv on first attempted handoff",
        ),
        _execution_check_row(
            step="execution_compatibility",
            ready=bool(compatibility.get("checked", False)) and not compatibility.get("blocker"),
            blocker=str(compatibility.get("blocker", "")),
            evidence=(
                f"route_markets={','.join(compatibility.get('route_markets', []))};"
                f"compatible={','.join(compatibility.get('compatible_markets', []))};"
                f"incompatible={','.join(compatibility.get('incompatible_markets', []))};"
                f"missing={','.join(compatibility.get('missing_markets', []))}"
            ),
            next_action=(
                "only route paper through markets that have confirmed on exchange"
                if compatibility.get("blocker")
                else "execution compatibility is aligned with the route candidate set"
            ),
        ),
        _execution_check_row(
            step="account_state_clean",
            ready=bool(account_state.get("checked", False)) and not account_state.get("blocker"),
            blocker=str(account_state.get("blocker", "")),
            evidence=(
                f"open_markets={','.join(account_state.get('open_markets', []))};"
                f"positions={json.dumps(account_state.get('positions', []), sort_keys=True)};"
                f"source={account_state.get('source', '')};"
                f"browser_override_confirmed_at_utc={account_state.get('browser_override_confirmed_at_utc', '')}"
            ),
            next_action=(
                "flatten lingering account legs before any new paper submission"
                if account_state.get("blocker")
                else "account state is flat and ready for fresh paper submissions"
            ),
        ),
    ]
    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, output)
    return frame


def _route_markets_from_candidates(frame: pd.DataFrame) -> list[str]:
    if frame.empty or "pair" not in frame.columns:
        return []
    markets: list[str] = []
    for pair in frame.get("pair", pd.Series(dtype=object)).astype(str).tolist():
        markets.extend(_pair_to_markets(pair))
    return sorted({market for market in markets if market})


def _pair_to_markets(pair: str) -> list[str]:
    return pair_markets_from_pair(pair)


def print_paper_execution_preflight() -> None:
    output = ROOT / "reports" / "paper_execution_preflight.csv"
    frame = paper_execution_preflight_report(output)
    print(frame.to_string(index=False))
    print(f"paper_execution_preflight: {output}")


def refresh_execution_truth_surfaces(root: Path = ROOT) -> dict[str, str]:
    reports = root / "reports"
    active = reports / "active"
    rl = reports / "rl"
    compatibility = refresh_dydx_execution_compatibility_table(root=root)
    injective_compatibility = refresh_injective_execution_compatibility_table(root=root)
    refresh_injective_mirror_candidate_queue(root=root)
    refresh_injective_spot_supported_pair_universe(root=root)
    refresh_injective_spot_first_candidate_shortlist(root=root)
    gmx_inventory = _read_csv_or_empty(
        active / "gmx_testnet_market_inventory.csv"
    )
    gmx_compatibility = refresh_gmx_execution_compatibility_table(root=root)
    refresh_gmx_testnet_candidate_shortlist(root=root)
    hyperliquid_inventory = _read_csv_or_empty(
        active / "hyperliquid_testnet_market_inventory.csv"
    )
    hyperliquid_compatibility = refresh_hyperliquid_execution_compatibility_table(root=root)
    refresh_hyperliquid_testnet_candidate_shortlist(root=root)
    refresh_non_eth_route_submit_queue(root=root)
    paper_execution_preflight_report(reports / "paper_execution_preflight.csv")
    journal_result = build_wizard_research_journal(root=root)
    refresh_paper_trade_price_journal(root=root)
    shortlist = paper_candidate_shortlist_rows(root=root)
    shortlist_path = active / "compatibility_first_candidate_shortlist.csv"
    atomic_write_csv(shortlist, shortlist_path, index=False)
    refresh_paper_trade_decision_report(root=root)
    live_monitor_paths = refresh_live_paper_trade_monitor(root=root)
    base_rl_paper_handoff_report(root=root)
    current_state(root=root)
    return {
        "execution_compatibility": str(active / "dydx_execution_market_compatibility.csv"),
        "injective_execution_compatibility": str(
            active / "injective_execution_market_compatibility.csv"
        ),
        "injective_mirror_candidate_queue": str(active / "injective_mirror_candidate_queue.csv"),
        "injective_spot_supported_pair_universe": str(
            active / "injective_spot_supported_pair_universe.csv"
        ),
        "injective_spot_first_candidate_shortlist": str(
            active / "injective_spot_first_candidate_shortlist.csv"
        ),
        "gmx_testnet_market_inventory": str(active / "gmx_testnet_market_inventory.csv"),
        "gmx_execution_compatibility": str(active / "gmx_execution_market_compatibility.csv"),
        "gmx_testnet_candidate_shortlist": str(active / "gmx_testnet_candidate_shortlist.csv"),
        "hyperliquid_testnet_market_inventory": str(
            active / "hyperliquid_testnet_market_inventory.csv"
        ),
        "hyperliquid_execution_compatibility": str(
            active / "hyperliquid_execution_market_compatibility.csv"
        ),
        "hyperliquid_testnet_candidate_shortlist": str(
            active / "hyperliquid_testnet_candidate_shortlist.csv"
        ),
        "route_submit_queue": str(active / "non_eth_route_submit_queue.csv"),
        "compatibility_first_candidate_shortlist": str(shortlist_path),
        "paper_trade_decision_report": str(active / "paper_trade_decision_report.csv"),
        "live_paper_trade_monitor": str(live_monitor_paths["summary"]),
        "live_paper_trade_timeframe_monitor": str(live_monitor_paths["timeframe"]),
        "live_paper_trade_monitor_md": str(live_monitor_paths["markdown"]),
        "paper_execution_preflight": str(reports / "paper_execution_preflight.csv"),
        "base_rl_paper_handoff_status": str(rl / "base_rl_paper_handoff_status.csv"),
        "current_state": str(active / "current_state.csv"),
        "wizard_research_scanner_capture": str(
            journal_result.paths["wizard_research_scanner_capture"]
        ),
        "wizard_research_pair_detail_capture": str(
            journal_result.paths["wizard_research_pair_detail_capture"]
        ),
        "wizard_research_journal": str(journal_result.paths["wizard_research_journal"]),
        "wizard_research_journal_json": str(journal_result.paths["wizard_research_journal_json"]),
        "compatible_markets": str(
            int(
                compatibility.get("compatible_for_paper_submit", pd.Series(dtype=bool))
                .fillna(False)
                .astype(bool)
                .sum()
            )
        )
        if not compatibility.empty
        else "0",
        "injective_mirrorable_pairs": str(
            int(
                injective_compatibility.get("mirrorable_for_paper", pd.Series(dtype=bool))
                .fillna(False)
                .astype(bool)
                .sum()
            )
        )
        if not injective_compatibility.empty
        else "0",
        "gmx_testnet_markets": str(len(gmx_inventory)) if not gmx_inventory.empty else "0",
        "gmx_mirrorable_pairs": str(
            int(
                gmx_compatibility.get("mirrorable_for_paper", pd.Series(dtype=bool))
                .fillna(False)
                .astype(bool)
                .sum()
            )
        )
        if not gmx_compatibility.empty
        else "0",
        "hyperliquid_testnet_markets": str(
            int(
                hyperliquid_inventory.get("tradable_perp", pd.Series(dtype=bool))
                .fillna(False)
                .astype(bool)
                .sum()
            )
        )
        if not hyperliquid_inventory.empty
        else "0",
        "hyperliquid_mirrorable_pairs": str(
            int(
                hyperliquid_compatibility.get("mirrorable_for_paper", pd.Series(dtype=bool))
                .fillna(False)
                .astype(bool)
                .sum()
            )
        )
        if not hyperliquid_compatibility.empty
        else "0",
    }


def paper_venue_preflight_report(
    pair: str | None = None,
    output_path: Path | None = None,
    max_pairs: int = 25,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    """Build compact per-venue paper readiness for one pair or top pairs."""
    effective_root = root or ROOT
    reports = effective_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "paper_venue_preflight.csv"
    universe = _read_csv_or_empty(
        effective_root / "data" / "processed" / "pair_universe.csv"
    )
    rows: list[dict[str, object]] = []

    if pair:
        pair_rows: list[str] = [_normalize_dydx_pair(pair)]
    else:
        pair_rows = []
        if not universe.empty and "pair" in universe.columns:
            score_col = pd.to_numeric(
                universe.get("combined_score", pd.Series(dtype=float)), errors="coerce"
            )
            ranked = universe.copy()
            ranked["combined_score"] = score_col
            pair_rows = (
                ranked.sort_values("combined_score", ascending=False, na_position="last")["pair"]
                .dropna()
                .astype(str)
                .head(max_pairs)
                .tolist()
            )

    if not pair_rows:
        frame = pd.DataFrame(
            [
                {
                    "pair": _md_text(pair or ""),
                    "venue": "",
                    "preference": "",
                    "venue_lanes": "",
                    "execution_ready": False,
                    "adapter_ready": False,
                    "ready_for_submission": False,
                    "contract_configured": False,
                    "contract_valid": False,
                    "exchange_submission_capable": False,
                    "record_only": False,
                    "contract_error": "no_candidate_pairs_found",
                    "blockers": "no_market_venue_context_or_pair_universe",
                    "evidence": "missing_market_venue_context_or_pair_universe",
                }
            ]
        )
        _write_csv_atomic(frame, output)
        return frame

    for candidate_pair in pair_rows:
        options = _build_paper_venue_options(candidate_pair, root=effective_root)
        if not options:
            rows.append(
                {
                    "pair": candidate_pair,
                    "venue": "dydx",
                    "preference": "candidate",
                    "venue_lanes": "",
                    "execution_ready": False,
                    "adapter_ready": False,
                    "ready_for_submission": False,
                    "cost_model_profile": "unknown_venue_cost_model",
                    "cost_model_aligned": False,
                    "cost_model_next_action": "wire_exchange_cost_model_report_before_execution",
                    "contract_configured": False,
                    "contract_valid": False,
                    "exchange_submission_capable": False,
                    "record_only": False,
                    "contract_error": "missing_market_venue_context",
                    "blockers": "no_market_venue_context",
                    "evidence": "pair_venue_context_missing_in_market_venue_context.csv",
                }
            )
            continue

        for venue_row in options:
            venue = str(venue_row.get("venue", "")).strip().lower()
            venue_lanes = str(venue_row.get("venue_lanes", ""))
            preference = str(venue_row.get("preference", "candidate"))
            execution_ready = bool(venue_row.get("execution_ready", False))
            blockers: list[str] = [
                item for item in str(venue_row.get("blockers", "")).split(";") if item
            ]

            if venue == "dydx":
                config = DydxNetworkConfig.paper_testnet_from_env()
                indexer_ready = build_dydx_indexer_adapter(config) is not None
                order_client, order_adapter_error = _load_dydx_order_client_adapter()
                adapter_contract = validate_dydx_order_client_adapter()
                adapter_ready = (
                    order_client is not None
                    and not bool(order_adapter_error)
                    and bool(adapter_contract.get("valid"))
                    and bool(adapter_contract.get("exchange_submission_capable"))
                )
                blockers.extend(config.paper_trading_blockers())
                if order_adapter_error:
                    blockers.append(f"invalid_dydx_order_client_adapter:{order_adapter_error}")
                elif not adapter_contract.get("configured"):
                    blockers.append("missing_dydx_order_client_adapter")
                elif not adapter_contract.get("valid"):
                    blockers.append(
                        f"dydx_order_client_adapter_invalid:{adapter_contract.get('error')}"
                    )
                if not indexer_ready:
                    blockers.append("missing_dydx_indexer_adapter")
                if not adapter_ready:
                    blockers.append("dydx_not_submission_ready")
                ready_for_submission = (
                    execution_ready
                    and adapter_ready
                    and indexer_ready
                    and not bool(config.paper_trading_blockers())
                )

                rows.append(
                    {
                        "pair": candidate_pair,
                        "venue": venue,
                        "preference": preference,
                        "venue_lanes": venue_lanes,
                        "execution_ready": execution_ready,
                        "adapter_ready": bool(adapter_ready),
                        "ready_for_submission": bool(ready_for_submission),
                        "cost_model_profile": _venue_cost_model_profile(venue)[0],
                        "cost_model_aligned": bool(_venue_cost_model_profile(venue)[1]),
                        "cost_model_next_action": _venue_cost_model_profile(venue)[2],
                        "contract_configured": bool(adapter_contract.get("configured")),
                        "contract_valid": bool(adapter_contract.get("valid")),
                        "exchange_submission_capable": bool(
                            adapter_contract.get("exchange_submission_capable")
                        ),
                        "record_only": bool(adapter_contract.get("record_only")),
                        "contract_error": str(adapter_contract.get("error") or ""),
                        "blockers": ";".join(sorted(set([item for item in blockers if item]))),
                        "evidence": (
                            f"dydx_submit_orders={config.submit_orders};"
                            f"dydx_indexer_ready={indexer_ready};"
                            f"dydx_order_adapter_ready={order_client is not None}"
                        ),
                    }
                )
                continue

            if venue == "hyperliquid":
                config = HyperliquidTestnetConfig.paper_testnet_from_env()
                preflight = hyperliquid_testnet_order_preflight_status()
                adapter_contract = validate_venue_order_client_adapter(venue)
                adapter_ready = bool(adapter_contract.get("valid")) and bool(
                    preflight.get("pair_executor_available")
                )
                preflight_blockers = [
                    item for item in str(preflight.get("blocker") or "").split(";") if item
                ]
                blockers.extend(preflight_blockers)
                if not adapter_ready:
                    blockers.append("hyperliquid_pair_executor_not_available")
                if not bool(preflight.get("ready")):
                    blockers.append("hyperliquid_not_submission_ready")
                ready_for_submission = execution_ready and bool(preflight.get("ready"))

                rows.append(
                    {
                        "pair": candidate_pair,
                        "venue": venue,
                        "preference": preference,
                        "venue_lanes": venue_lanes,
                        "execution_ready": execution_ready,
                        "adapter_ready": adapter_ready,
                        "ready_for_submission": ready_for_submission,
                        "cost_model_profile": _venue_cost_model_profile(venue)[0],
                        "cost_model_aligned": bool(_venue_cost_model_profile(venue)[1]),
                        "cost_model_next_action": _venue_cost_model_profile(venue)[2],
                        "contract_configured": bool(adapter_contract.get("configured")),
                        "contract_valid": bool(adapter_contract.get("valid")),
                        "exchange_submission_capable": bool(
                            preflight.get("pair_executor_available")
                        ),
                        "record_only": bool(adapter_contract.get("record_only")),
                        "contract_error": str(adapter_contract.get("error") or ""),
                        "blockers": ";".join(sorted(set(item for item in blockers if item))),
                        "evidence": (
                            f"hyperliquid_submit_orders={config.submit_orders};"
                            f"pair_executor_available={preflight.get('pair_executor_available')};"
                            f"single_leg_order_path_blocked={preflight.get('single_leg_order_path_blocked')}"
                        ),
                    }
                )
                continue

            order_client, order_adapter_error = _load_venue_order_client_adapter(venue)
            adapter_contract = validate_venue_order_client_adapter(venue)
            adapter_ready = (
                order_client is not None
                and not bool(order_adapter_error)
                and bool(adapter_contract.get("valid"))
                and bool(adapter_contract.get("exchange_submission_capable"))
            )
            if order_adapter_error:
                blockers.append(f"invalid_{venue}_order_client_adapter:{order_adapter_error}")
            elif not adapter_contract.get("configured"):
                blockers.append(f"missing_{venue}_order_client_adapter")
            elif not adapter_contract.get("valid"):
                blockers.append(
                    f"{venue}_order_client_adapter_invalid:{adapter_contract.get('error')}"
                )
            if not adapter_ready:
                blockers.append(f"{venue}_not_submission_ready")
            ready_for_submission = execution_ready and adapter_ready

            rows.append(
                {
                    "pair": candidate_pair,
                    "venue": venue,
                    "preference": preference,
                    "venue_lanes": venue_lanes,
                    "execution_ready": execution_ready,
                    "adapter_ready": bool(adapter_ready),
                    "ready_for_submission": bool(ready_for_submission),
                    "cost_model_profile": _venue_cost_model_profile(venue)[0],
                    "cost_model_aligned": bool(_venue_cost_model_profile(venue)[1]),
                    "cost_model_next_action": _venue_cost_model_profile(venue)[2],
                    "contract_configured": bool(adapter_contract.get("configured")),
                    "contract_valid": bool(adapter_contract.get("valid")),
                    "exchange_submission_capable": bool(
                        adapter_contract.get("exchange_submission_capable")
                    ),
                    "record_only": bool(adapter_contract.get("record_only")),
                    "contract_error": str(adapter_contract.get("error") or ""),
                    "blockers": ";".join(sorted(set([item for item in blockers if item]))),
                    "evidence": (
                        f"adapter_path={adapter_contract.get('adapter_path') or ''};"
                        f"venue_contract={adapter_contract.get('signature_accepts_intent_config')}"
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, output)
    return frame


def print_paper_venue_preflight(pair: str | None = None, max_pairs: int = 25) -> None:
    output = ROOT / "reports" / "paper_venue_preflight.csv"
    frame = paper_venue_preflight_report(pair=pair, output_path=output, max_pairs=max_pairs)
    print(frame.to_string(index=False))
    print(f"paper_venue_preflight: {output}")


def paper_readiness_checkpoint(
    root: Path | None = None, readiness_threshold: float = 0.65
) -> dict[str, object]:
    """Run a no-trade, no-spend checkpoint pass for paper-readiness continuity."""
    if root is None:
        root = ROOT
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output_path = reports / "paper_readiness_checkpoint.csv"
    readiness = priority_readiness_report()
    gates = readiness.set_index("gate") if not readiness.empty else pd.DataFrame()
    brain_readiness = build_brain_readiness_report(root=root, score_threshold=readiness_threshold)
    paper_preflight = paper_execution_preflight_report(reports / "paper_execution_preflight.csv")
    brain_ready = str(brain_readiness.summary.get("score_gate")) == "pass"
    if not readiness.empty:
        priority_ready = True
        priority_blocker = ""
        priority_next = "continue"
    else:
        priority_ready = False
        priority_blocker = "missing_priority_readiness"
        priority_next = "run: PYTHONPATH=src python3 -m quant_platform.cli priority-readiness"

    if brain_ready:
        brain_blocker = ""
        brain_next = "keep running for trend and provider quality improvements"
    else:
        brain_blocker = "brain_readiness_not_ready"
        brain_next = "review paper_readiness_trend.csv"

    paper_gate_ready = _gate_ready_from_index(gates, "paper_execution_gate")
    paper_blocker = _gate_value(gates, "paper_execution_gate", "blocker")
    if paper_gate_ready:
        paper_next = "paper-readiness checkpoint passed; use paper-execution-preflight to continue"
        paper_blocker = ""
    else:
        paper_next = "fix blockers in priority checklist and venue adapters before paper plan"

    paper_preflight_ready = len(paper_preflight) > 0 and bool(
        paper_preflight.get("ready", pd.Series(dtype=bool)).map(_coerce_bool).all()
    )
    paper_cost_blocker = "paper_preflight_not_clean" if not paper_preflight_ready else ""

    rows = []
    rows.append(
        _execution_check_row(
            step="priority_readiness",
            ready=priority_ready,
            blocker=priority_blocker,
            evidence="reports/priority_readiness.csv",
            next_action=priority_next,
        )
    )
    rows.append(
        _execution_check_row(
            step="brain_readiness_score",
            ready=brain_ready,
            blocker=brain_blocker,
            evidence=f"reports/brain/{Path(brain_readiness.paths['brain_readiness_report']).name}",
            next_action=brain_next,
        )
    )
    rows.append(
        _execution_check_row(
            step="paper_execution_preflight",
            ready=paper_gate_ready,
            blocker=paper_blocker,
            evidence="reports/paper_execution_preflight.csv",
            next_action=paper_next,
        )
    )
    rows.append(
        _execution_check_row(
            step="paper_venue_cost_alignment",
            ready=paper_preflight_ready,
            blocker=paper_cost_blocker,
            evidence="reports/paper_venue_preflight.csv",
            next_action="maintain no-live-cost checkpoint mode until venue readiness stabilizes",
        )
    )
    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, output_path)
    return {
        "summary": {
            "readiness_gate": str(brain_readiness.summary.get("score_gate")),
            "readiness_score": float(brain_readiness.summary.get("readiness_score", 0.0)),
            "ready_rows": int(frame["status"].eq("ready").sum()),
            "blocked_rows": int(frame["status"].eq("blocked").sum()),
            "checkpoint_path": str(output_path),
        },
        "paths": {
            "paper_readiness_checkpoint": str(output_path),
            "paper_execution_preflight": str(reports / "paper_execution_preflight.csv"),
            "brain_readiness_report": str(brain_readiness.paths["brain_readiness_report"]),
            "brain_readiness_trend": str(reports / "brain" / "paper_readiness_trend.csv"),
        },
    }


def _sweep_row(
    step: str, status: str, detail: str, artifact_path: str = "", mode: str = ""
) -> dict[str, object]:
    return {
        "mode": mode,
        "step": step,
        "status": status,
        "detail": detail,
        "artifact_path": artifact_path,
    }


def run_research_sweep(
    *,
    mode: str = "light",
    root: Path = ROOT,
    mcp_url: str | None = None,
    api_token: str | None = None,
    source_filter: str | None = None,
    wait_seconds: int = 90,
    do_fetch: bool = True,
    apify_actor_credit_ceiling: int = 0,
    readiness_threshold: float = 0.65,
) -> dict[str, object]:
    if mode not in SWEEP_MODES:
        raise SystemExit(f"unsupported sweep mode: {mode}")

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output_path = reports / "research_sweep_status.csv"
    summary_path = reports / "research_sweep_summary.json"
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%SZ")
    rows: list[dict[str, object]] = []

    current = current_state()
    rows.append(
        _sweep_row(
            "current_state",
            "completed",
            f"rows={current.summary.get('rows', 0)}",
            str(current.paths.get("current_state", "")),
            mode,
        )
    )
    system = system_check()
    rows.append(
        _sweep_row(
            "system_check",
            "completed",
            f"checks={system.summary.get('checks', 0)};blocked={system.summary.get('blocked', 0)}",
            str(system.paths.get("system_check", "")),
            mode,
        )
    )

    if mode == "paid":
        if not mcp_url:
            rows.append(
                _sweep_row(
                    "apify_refresh",
                    "skipped",
                    "missing_mcp_url_or_apify_server",
                    str(root / "reports" / "active" / "apify_source_capture_manifest.csv"),
                    mode,
                )
            )
        else:
            apify = refresh_apify_sources(
                root=root,
                mcp_url=mcp_url,
                source_filter=source_filter,
                do_fetch=do_fetch,
                api_token=api_token if api_token else None,
                wait_seconds=wait_seconds,
                actor_credit_ceiling=apify_actor_credit_ceiling,
            )
            rows.append(
                _sweep_row(
                    "apify_refresh",
                    "completed",
                    f"sources={apify.source_count};sampled={apify.sampled_count};failed={apify.failed_count};needs_key={apify.needs_key_count}",
                    str(apify.manifest_path),
                    mode,
                )
            )
    else:
        rows.append(
            _sweep_row(
                "apify_refresh",
                "skipped",
                "mode_does_not_run_paid_confirmation",
                str(root / "reports" / "active" / "apify_source_capture_manifest.csv"),
                mode,
            )
        )

    venue_context = build_market_venue_context()
    rows.append(
        _sweep_row(
            "market_venue_context",
            "completed",
            f"rows={venue_context.summary.get('rows', 0)}",
            str(venue_context.paths.get("market_venue_context", "")),
            mode,
        )
    )
    pair_universe = build_pair_universe()
    rows.append(
        _sweep_row(
            "pair_universe",
            "completed",
            f"pairs={pair_universe.summary.get('pairs', 0)};promoted={pair_universe.summary.get('promoted', 0)}",
            str(pair_universe.paths.get("pair_universe", "")),
            mode,
        )
    )

    if mode in {"deep", "paid"}:
        dataset = build_trade_dataset()
        rows.append(
            _sweep_row(
                "trade_dataset",
                "completed",
                f"rows={dataset.summary.get('rows', 0)}",
                str(dataset.paths.get("dataset_csv", "")),
                mode,
            )
        )
        trade_gate = train_trade_gate()
        rows.append(
            _sweep_row(
                "train_trade_gate",
                "completed",
                f"accepted={trade_gate.summary.get('accepted', False)}",
                str(trade_gate.paths.get("metrics_json", trade_gate.paths.get("metrics", ""))),
                mode,
            )
        )
        gated = run_model_gated_backtest()
        rows.append(
            _sweep_row(
                "model_gated_backtest",
                "completed",
                f"accepted={gated.summary.get('accepted', False)}",
                str(gated.paths.get("acceptance", "")),
                mode,
            )
        )
    else:
        rows.append(
            _sweep_row(
                "trade_dataset",
                "skipped",
                "mode_does_not_run_deep_modeling",
                str(root / "data" / "ml" / "trade_training_dataset.csv"),
                mode,
            )
        )
        rows.append(
            _sweep_row(
                "train_trade_gate",
                "skipped",
                "mode_does_not_run_deep_modeling",
                str(root / "models" / "trade_gate" / "metrics.json"),
                mode,
            )
        )
        rows.append(
            _sweep_row(
                "model_gated_backtest",
                "skipped",
                "mode_does_not_run_deep_modeling",
                str(root / "reports" / "ml" / "model_gated_acceptance.csv"),
                mode,
            )
        )

    brain = build_brain_readiness_report(root=root, score_threshold=readiness_threshold)
    rows.append(
        _sweep_row(
            "brain_readiness",
            "completed",
            f"score_gate={brain.summary.get('score_gate', '')};score={brain.summary.get('readiness_score', 0.0)}",
            str(brain.paths.get("brain_readiness_report", "")),
            mode,
        )
    )

    dashboard = build_command_dashboard(refresh_profile="monitor" if mode == "light" else "deep")
    rows.append(
        _sweep_row(
            "command_dashboard",
            "completed",
            f"dashboard_files={dashboard.summary.get('dashboard_files', 0)};blocked_rows={dashboard.summary.get('blocked_rows', 0)}",
            str(dashboard.paths.get("command_center", "")),
            mode,
        )
    )
    readiness = priority_readiness_report(reports / "priority_readiness.csv")
    ready_gates = (
        int(readiness.get("ready", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        if not readiness.empty
        else 0
    )
    rows.append(
        _sweep_row(
            "priority_readiness",
            "completed",
            f"ready_gates={ready_gates}/{len(readiness)}",
            str(reports / "priority_readiness.csv"),
            mode,
        )
    )
    paper_execution_preflight_report(reports / "paper_execution_preflight.csv")
    checkpoint = paper_readiness_checkpoint(root=root, readiness_threshold=readiness_threshold)
    rows.append(
        _sweep_row(
            "paper_readiness_checkpoint",
            "completed",
            f"readiness_gate={checkpoint['summary'].get('readiness_gate', '')};ready_rows={checkpoint['summary'].get('ready_rows', 0)};blocked_rows={checkpoint['summary'].get('blocked_rows', 0)}",
            str(checkpoint["paths"].get("paper_readiness_checkpoint", "")),
            mode,
        )
    )

    if mode in {"deep", "paid"}:
        supreme_csv, supreme_md = print_supreme_team_checkpoint()
        rows.append(
            _sweep_row("supreme_team", "completed", "checkpoint_refreshed", str(supreme_md), mode)
        )
    else:
        rows.append(
            _sweep_row(
                "supreme_team",
                "skipped",
                "mode_does_not_run_full_audit",
                str(root / "reports" / "supreme_team" / "latest_supreme_team.md"),
                mode,
            )
        )

    frame = pd.DataFrame(rows, columns=["mode", "step", "status", "detail", "artifact_path"])
    _write_csv_atomic(frame, output_path)
    blocked = (
        readiness[~readiness.get("ready", pd.Series(dtype=bool)).fillna(False).astype(bool)].copy()
        if not readiness.empty
        else pd.DataFrame()
    )
    paper_gate_ready = bool(
        not readiness.empty
        and readiness[readiness["gate"] == "paper_execution_gate"]["ready"]
        .fillna(False)
        .astype(bool)
        .any()
    )
    summary = {
        "mode": mode,
        "run_id": f"research_sweep_{mode}_{timestamp}",
        "completed_steps": int((frame["status"] == "completed").sum()),
        "skipped_steps": int((frame["status"] == "skipped").sum()),
        "blocked_gates": len(blocked),
        "paper_trading_ready": paper_gate_ready,
        "next_action": "paper trading lane is ready"
        if paper_gate_ready
        else ";".join(blocked["next_action"].dropna().astype(str).head(3).tolist())
        or "review readiness blockers",
    }
    atomic_write_text(summary_path, json.dumps(
            {
                "summary": summary,
                "paths": {
                    "research_sweep_status": str(output_path),
                    "research_sweep_summary": str(summary_path),
                    "priority_readiness": str(reports / "priority_readiness.csv"),
                    "paper_execution_preflight": str(reports / "paper_execution_preflight.csv"),
                    "command_dashboard": str(root / "reports" / "dashboard" / "command_center.md"),
                },
            },
            indent=2,
        ), encoding="utf-8")
    return {
        "summary": summary,
        "paths": {
            "research_sweep_status": str(output_path),
            "research_sweep_summary": str(summary_path),
            "priority_readiness": str(reports / "priority_readiness.csv"),
            "paper_execution_preflight": str(reports / "paper_execution_preflight.csv"),
            "command_dashboard": str(root / "reports" / "dashboard" / "command_center.md"),
        },
    }


def _priority_gap_frame_from_dashboard(dashboard: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in dashboard.iterrows():
        priority = str(row["priority"])
        area = str(row["area"])
        ready = bool(row["ready"])
        blocker = str(row["blocker"] or "")
        rows.append(
            {
                "priority": priority,
                "area": area,
                "status": "pass" if ready else "gap",
                "severity": "none" if ready else _gap_severity(priority),
                "gap": "" if ready else blocker,
                "current_evidence": row["key_metric"],
                "required_proof": _required_gap_proof(area),
                "source_report": row["source_report"],
                "next_action": row["next_action"],
            }
        )
    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, output)
    return frame


def _append_current_state_operational_gaps(
    frame: pd.DataFrame,
    output: Path,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    effective_root = root or ROOT
    current_path = effective_root / "reports" / "active" / "current_state.csv"
    current = _read_csv_or_empty(current_path)
    if current.empty or "area" not in current.columns:
        return frame

    canonical_authority = _read_csv_or_empty(
        effective_root / "reports" / "active" / "hyperliquid_authority_state.csv"
    )
    critical_areas = (
        [
            "canonical_hyperliquid_authority",
            "hyperliquid_wizard_exact_mode_intake",
            "hyperliquid_walkforward",
            "teacher_council",
            "student_learning",
            "portfolio_critic",
        ]
        if not canonical_authority.empty
        else [
            "base_rl_handoff",
            "paper_status",
            "injective_mirror_lane",
            "wizard_readiness",
            "overall_readiness",
        ]
    )
    working = current[current["area"].astype(str).isin(critical_areas)].copy()
    if working.empty:
        return frame

    rows: list[dict[str, object]] = []
    priority_index = 1
    for area in critical_areas:
        matches = working[working["area"].astype(str) == area]
        if matches.empty:
            continue
        row = matches.iloc[0]
        ready = str(row.get("ready", "")).strip().lower() in {"true", "1", "yes"}
        if ready:
            continue
        blocker = str(row.get("blocker", "") or "").strip() or str(row.get("status", "blocked"))
        rows.append(
            {
                "priority": f"CS{priority_index}",
                "area": f"current_state_{area}",
                "status": "gap",
                "severity": "critical"
                if area
                in {"canonical_hyperliquid_authority", "hyperliquid_wizard_exact_mode_intake"}
                else (
                    "high"
                    if area
                    in {
                        "hyperliquid_walkforward",
                        "teacher_council",
                        "student_learning",
                        "portfolio_critic",
                        "base_rl_handoff",
                        "paper_status",
                        "overall_readiness",
                    }
                    else "medium"
                ),
                "gap": blocker,
                "current_evidence": str(row.get("detail", "") or ""),
                "required_proof": "current_state_ready_true_with_matching_evidence",
                "source_report": str(row.get("evidence_path", "") or current_path),
                "next_action": str(
                    row.get("next_action", "")
                    or "repair current-state blocker and rerun current-state"
                ),
            }
        )
        priority_index += 1

    if not rows:
        return frame
    augmented = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True, sort=False)
    _write_csv_atomic(augmented, output)
    return augmented


def _seven_stage_priority_gap_frame(root: Path = ROOT) -> pd.DataFrame:
    """Project the active seven-stage checkpoint into the assessment schema."""

    checkpoint_path = root / "reports" / "active" / "seven_stage_goal_checkpoint.csv"
    checkpoint = _read_csv_or_empty(checkpoint_path)
    required = {
        "stage",
        "objective",
        "status",
        "evidence_progress",
        "blocker",
        "next_action",
    }
    if checkpoint.empty or not required.issubset(checkpoint.columns):
        return pd.DataFrame()
    stages = pd.to_numeric(checkpoint["stage"], errors="coerce")
    if stages.isna().any() or set(stages.astype(int)) != set(range(1, 8)):
        return pd.DataFrame()
    if stages.astype(int).duplicated().any():
        return pd.DataFrame()

    severity = {
        1: "medium",
        2: "critical",
        3: "critical",
        4: "critical",
        5: "high",
        6: "high",
        7: "high",
    }
    rows: list[dict[str, object]] = []
    for _, row in checkpoint.assign(_stage=stages.astype(int)).sort_values("_stage").iterrows():
        stage = int(row["_stage"])
        passed = str(row["status"]).strip().upper() == "PASS"
        rows.append(
            {
                "priority": f"S{stage}",
                "area": f"seven_stage_{stage}_{str(row['objective']).strip()}",
                "status": "pass" if passed else "gap",
                "severity": "none" if passed else severity[stage],
                "gap": "" if passed else str(row["blocker"] or "").strip(),
                "current_evidence": str(row["evidence_progress"] or "").strip(),
                "required_proof": f"stage_{stage}_pass_with_immutable_evidence",
                "source_report": str(checkpoint_path.relative_to(root)),
                "next_action": str(row["next_action"] or "").strip(),
            }
        )
    return pd.DataFrame(rows)


def priority_gap_test_report(
    readiness: pd.DataFrame | None = None,
    output_path: Path | None = None,
    refresh_paper_preflight: bool = True,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    effective_root = root or ROOT
    reports = effective_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "priority_gap_test.csv"
    seven_stage = _seven_stage_priority_gap_frame(effective_root)
    if not seven_stage.empty:
        _write_csv_atomic(seven_stage, output)
        return seven_stage
    readiness = (
        readiness
        if readiness is not None
        else priority_readiness_report(root=effective_root)
    )
    if refresh_paper_preflight:
        paper_execution_preflight_report(
            reports / "paper_execution_preflight.csv",
            root=effective_root,
        )
    dashboard = priority_spine_dashboard_report(readiness, root=effective_root)
    frame = _priority_gap_frame_from_dashboard(dashboard, output)
    canonical_authority = _read_csv_or_empty(
        effective_root / "reports" / "active" / "hyperliquid_authority_state.csv"
    )
    if not canonical_authority.empty and "area" in frame.columns:
        historical_mask = frame["area"].astype(str).eq("crypto_wizards_capture")
        frame.loc[historical_mask, "area"] = "crypto_wizards_historical_capture"
        frame.loc[historical_mask, "priority"] = "P1H"
        frame.loc[historical_mask, "required_proof"] = (
            "historical_capture_corpus_complete_with_lineage"
        )
        frame.loc[historical_mask, "next_action"] = (
            "preserve as historical discovery evidence; do not treat it as current candidate readiness"
        )
        legacy_areas = {
            "strategy_acceptance",
            "dydx_testnet_readiness",
            "paper_execution_gate",
            "learning_event_store",
        }
        frame = frame.loc[~frame["area"].astype(str).isin(legacy_areas)].copy()
        _write_csv_atomic(frame, output)
    return _append_current_state_operational_gaps(
        frame,
        output,
        root=effective_root,
    )


def print_gap_test() -> None:
    output = ROOT / "reports" / "priority_gap_test.csv"
    readiness = priority_readiness_report()
    paper_execution_preflight_report(
        ROOT / "reports" / "paper_execution_preflight.csv", readiness=readiness
    )
    frame = priority_gap_test_report(readiness, output, refresh_paper_preflight=False)
    print(frame.to_string(index=False))
    print(f"priority_gap_test: {output}")


def print_gap_analysis_checklist(run_dir: Path | None = None) -> tuple[Path, Path]:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir or (reports / "gap_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%SZ")
    run_id = f"gap_analysis_{timestamp}"

    gap_frame = priority_gap_test_report()
    if gap_frame.empty:
        gap_rows = [
            {
                "run_id": run_id,
                "timestamp_utc": timestamp,
                "priority": "N/A",
                "area": "unknown",
                "status": "blocked",
                "severity": "high",
                "gap": "priority_gap_test_report_empty",
                "current_evidence": "",
                "required_proof": "rerun_priority_gap_report",
                "source_report": str(reports / "priority_gap_test.csv"),
                "next_action": "rerun gap-test and then rebuild checklist",
                "done": False,
            }
        ]
        gap_table = pd.DataFrame(gap_rows)
    else:
        gap_table = gap_frame.copy()
        gap_table["run_id"] = run_id
        gap_table["timestamp_utc"] = timestamp
        gap_table["done"] = gap_table["status"].eq("pass")

    selected_cols = [
        "run_id",
        "timestamp_utc",
        "priority",
        "area",
        "status",
        "severity",
        "gap",
        "current_evidence",
        "required_proof",
        "source_report",
        "next_action",
        "done",
    ]
    checklist_frame = gap_table[selected_cols]

    csv_path = output_dir / f"{run_id}.csv"
    _write_csv_atomic(checklist_frame, csv_path)

    open_count = int((checklist_frame["status"] == "gap").sum())
    pass_count = int((checklist_frame["status"] == "pass").sum())
    critical_count = int((checklist_frame["severity"] == "critical").sum())
    high_count = int((checklist_frame["severity"] == "high").sum())
    medium_count = int((checklist_frame["severity"] == "medium").sum())
    lines: list[str] = [
        "# Gap Analysis Checkpoint",
        "",
        f"run_id: {run_id}",
        f"created_utc: {timestamp}",
        f"open_gaps: {open_count} / {len(checklist_frame)}",
        f"pass_gates: {pass_count}",
        f"critical: {critical_count}",
        f"high: {high_count}",
        f"medium: {medium_count}",
        "",
        "## Checklist",
        "",
    ]
    for _, row in checklist_frame.iterrows():
        status = str(row["status"])
        area = str(row["area"])
        priority = str(row["priority"])
        gap = str(row["gap"])
        next_action = str(row["next_action"])
        if status == "pass":
            lines.append(f"- [x] {priority} {area}: PASS ({row['gap']})")
        else:
            lines.append(f"- [ ] {priority} {area}: GAP ({gap}) -> {next_action}")
            lines.append(f"  - evidence: {row['current_evidence']}")
            lines.append(f"  - required proof: {row['required_proof']}")
            lines.append(f"  - source report: {row['source_report']}")
        lines.append("")

    latest_md = output_dir / "latest_gap_analysis.md"
    checkpoint_md = output_dir / f"{run_id}.md"
    atomic_write_text(checkpoint_md, "\n".join(lines), encoding="utf-8")
    atomic_write_text(latest_md, checkpoint_md.read_text(encoding="utf-8"), encoding="utf-8")

    index_path = reports / "gap_analysis_index.csv"
    index_frame = _read_csv_or_empty(index_path)
    if index_frame.empty:
        index_frame = pd.DataFrame(
            columns=[
                "run_id",
                "timestamp_utc",
                "open_gaps",
                "pass_gates",
                "critical",
                "high",
                "medium",
            ]
        )
    index_frame = pd.concat(
        [
            index_frame,
            pd.DataFrame(
                [
                    {
                        "run_id": run_id,
                        "timestamp_utc": timestamp,
                        "open_gaps": open_count,
                        "pass_gates": pass_count,
                        "critical": critical_count,
                        "high": high_count,
                        "medium": medium_count,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    _write_csv_atomic(index_frame, index_path)

    print(f"gap_analysis_checklist_csv: {csv_path}")
    print(f"gap_analysis_checklist_md: {checkpoint_md}")
    print(f"gap_analysis_checkpoint: {latest_md}")
    print(f"gap_analysis_index: {index_path}")
    return csv_path, checkpoint_md


def print_pre_mortem_checklist(
    run_dir: Path | None = None,
    readiness: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir or (reports / "pre_mortem")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%SZ")
    run_id = f"pre_mortem_{timestamp}"

    readiness = readiness if readiness is not None else priority_readiness_report()
    paper_execution_preflight_report(reports / "paper_execution_preflight.csv", readiness=readiness)
    try:
        gap_frame = priority_gap_test_report(readiness, refresh_paper_preflight=False)
    except TypeError:
        gap_frame = priority_gap_test_report(readiness)
    if gap_frame.empty:
        pm_frame = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "timestamp_utc": timestamp,
                    "priority": "N/A",
                    "area": "unknown",
                    "status": "blocked",
                    "severity": "high",
                    "gap": "priority_gap_test_report_empty",
                    "current_evidence": "",
                    "required_proof": "rerun_priority_gap_report",
                    "source_report": str(reports / "priority_gap_test.csv"),
                    "pre_mortem_question": "Can we trust execution readiness if no gap evidence is present?",
                    "failure_mode": "No evidence exists, so readiness decisions become guesswork.",
                    "prevention": "Re-run gap test and rerun pre-mortem before any acceptance changes.",
                    "done": False,
                }
            ]
        )
    else:
        pm_rows: list[dict[str, object]] = []
        for _, row in gap_frame.iterrows():
            priority = str(row["priority"])
            area = str(row["area"])
            status = str(row["status"])
            severity = str(row["severity"])
            gap = str(row["gap"] or "")
            required_proof = str(row["required_proof"] or "")
            evidence = str(row["current_evidence"] or "")
            source_report = str(row["source_report"] or "")
            pm_rows.append(
                {
                    "run_id": run_id,
                    "timestamp_utc": timestamp,
                    "priority": priority,
                    "area": area,
                    "status": status,
                    "severity": severity,
                    "gap": gap,
                    "current_evidence": evidence,
                    "required_proof": required_proof,
                    "source_report": source_report,
                    "pre_mortem_question": _pre_mortem_question(priority, area, severity),
                    "failure_mode": _pre_mortem_failure_mode(area, gap),
                    "prevention": _pre_mortem_prevention(area, required_proof),
                    "done": status == "pass",
                }
            )
        pm_frame = pd.DataFrame(pm_rows)

    selected_cols = [
        "run_id",
        "timestamp_utc",
        "priority",
        "area",
        "status",
        "severity",
        "gap",
        "current_evidence",
        "required_proof",
        "source_report",
        "pre_mortem_question",
        "failure_mode",
        "prevention",
        "done",
    ]
    pre_mortem_report = pm_frame[selected_cols]

    csv_path = output_dir / f"{run_id}.csv"
    _write_csv_atomic(pre_mortem_report, csv_path)

    open_count = int((pre_mortem_report["status"] == "gap").sum())
    pass_count = int((pre_mortem_report["status"] == "pass").sum())
    critical_count = int((pre_mortem_report["severity"] == "critical").sum())
    high_count = int((pre_mortem_report["severity"] == "high").sum())
    medium_count = int((pre_mortem_report["severity"] == "medium").sum())

    lines: list[str] = [
        "# Pre-Mortem Checkpoint",
        "",
        f"run_id: {run_id}",
        f"created_utc: {timestamp}",
        f"open_gaps: {open_count} / {len(pre_mortem_report)}",
        f"pass_gates: {pass_count}",
        f"critical: {critical_count}",
        f"high: {high_count}",
        f"medium: {medium_count}",
        "",
        "## Checklist",
        "",
    ]
    for _, row in pre_mortem_report.iterrows():
        status = str(row["status"])
        area = str(row["area"])
        priority = str(row["priority"])
        gap = str(row["gap"])
        if status == "pass":
            lines.append(f"- [x] {priority} {area}: PASS ({gap or 'no pre-mortem blocker'})")
        else:
            lines.append(f"- [ ] {priority} {area}: PRE-MORTEM RISK ({gap})")
            lines.append(f"  - preemptive question: {row['pre_mortem_question']}")
            lines.append(f"  - failure_mode: {row['failure_mode']}")
            lines.append(f"  - prevention: {row['prevention']}")
            lines.append(f"  - required proof: {row['required_proof']}")
            lines.append(f"  - evidence: {row['current_evidence']}")
            lines.append(f"  - source report: {row['source_report']}")
        lines.append("")

    latest_md = output_dir / "latest_pre_mortem.md"
    checkpoint_md = output_dir / f"{run_id}.md"
    atomic_write_text(checkpoint_md, "\n".join(lines), encoding="utf-8")
    atomic_write_text(latest_md, checkpoint_md.read_text(encoding="utf-8"), encoding="utf-8")

    index_path = reports / "pre_mortem_index.csv"
    index_frame = _read_csv_or_empty(index_path)
    if index_frame.empty:
        index_frame = pd.DataFrame(
            columns=[
                "run_id",
                "timestamp_utc",
                "open_gaps",
                "pass_gates",
                "critical",
                "high",
                "medium",
            ]
        )
    index_frame = pd.concat(
        [
            index_frame,
            pd.DataFrame(
                [
                    {
                        "run_id": run_id,
                        "timestamp_utc": timestamp,
                        "open_gaps": open_count,
                        "pass_gates": pass_count,
                        "critical": critical_count,
                        "high": high_count,
                        "medium": medium_count,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    _write_csv_atomic(index_frame, index_path)

    print(f"pre_mortem_checklist_csv: {csv_path}")
    print(f"pre_mortem_checklist_md: {checkpoint_md}")
    print(f"pre_mortem_checkpoint: {latest_md}")
    print(f"pre_mortem_index: {index_path}")
    return csv_path, checkpoint_md


def print_post_mortem_checklist(
    run_dir: Path | None = None,
    readiness: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir or (reports / "post_mortem")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%SZ")
    run_id = f"post_mortem_{timestamp}"

    previous_path = reports / "post_mortem" / "latest_post_mortem.csv"
    previous = _read_csv_or_empty(previous_path)
    previous_by_area: dict[str, str] = {}
    if not previous.empty and "area" in previous.columns and "status" in previous.columns:
        previous_by_area = {
            str(area): str(status)
            for area, status in zip(previous["area"].astype(str), previous["status"].astype(str))
        }

    readiness = readiness if readiness is not None else priority_readiness_report()
    paper_execution_preflight_report(reports / "paper_execution_preflight.csv", readiness=readiness)
    try:
        gap_frame = priority_gap_test_report(readiness, refresh_paper_preflight=False)
    except TypeError:
        gap_frame = priority_gap_test_report(readiness)
    if gap_frame.empty:
        pm_frame = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "timestamp_utc": timestamp,
                    "priority": "N/A",
                    "area": "unknown",
                    "status": "blocked",
                    "severity": "high",
                    "gap": "priority_gap_test_report_empty",
                    "current_evidence": "",
                    "required_proof": "rerun_priority_gap_report",
                    "source_report": str(reports / "priority_gap_test.csv"),
                    "incident_observed": "none",
                    "trajectory": "unknown",
                    "post_mortem_insight": "No evidence exists; run gap-test before post-mortem review.",
                    "prevention_from_pre_mortem": _post_mortem_prevention("unknown", ""),
                    "done": False,
                }
            ]
        )
    else:
        rows: list[dict[str, object]] = []
        for _, row in gap_frame.iterrows():
            priority = str(row["priority"])
            area = str(row["area"])
            status = str(row["status"])
            severity = str(row["severity"])
            gap = str(row["gap"] or "")
            required_proof = str(row["required_proof"] or "")
            evidence = str(row["current_evidence"] or "")
            source_report = str(row["source_report"] or "")
            prev_status = str(previous_by_area.get(area, "unknown"))
            trajectory = _post_mortem_status_trajectory(
                area=area, current_status=status, previous_status=prev_status
            )
            rows.append(
                {
                    "run_id": run_id,
                    "timestamp_utc": timestamp,
                    "priority": priority,
                    "area": area,
                    "status": status,
                    "severity": severity,
                    "gap": gap,
                    "current_evidence": evidence,
                    "required_proof": required_proof,
                    "source_report": source_report,
                    "incident_observed": _post_mortem_incident(area, gap),
                    "trajectory": trajectory,
                    "post_mortem_insight": _post_mortem_insight(area, trajectory, evidence),
                    "prevention_from_pre_mortem": _post_mortem_prevention(area, required_proof),
                    "done": status == "pass",
                }
            )
        pm_frame = pd.DataFrame(rows)

    selected_cols = [
        "run_id",
        "timestamp_utc",
        "priority",
        "area",
        "status",
        "severity",
        "gap",
        "current_evidence",
        "required_proof",
        "source_report",
        "incident_observed",
        "trajectory",
        "post_mortem_insight",
        "prevention_from_pre_mortem",
        "done",
    ]
    post_mortem_report = pm_frame[selected_cols]

    csv_path = output_dir / f"{run_id}.csv"
    _write_csv_atomic(post_mortem_report, csv_path)

    open_count = int((post_mortem_report["status"] == "gap").sum())
    pass_count = int((post_mortem_report["status"] == "pass").sum())
    critical_count = int((post_mortem_report["severity"] == "critical").sum())
    high_count = int((post_mortem_report["severity"] == "high").sum())
    medium_count = int((post_mortem_report["severity"] == "medium").sum())

    lines: list[str] = [
        "# Post-Mortem Checkpoint",
        "",
        f"run_id: {run_id}",
        f"created_utc: {timestamp}",
        f"open_gaps: {open_count} / {len(post_mortem_report)}",
        f"pass_gates: {pass_count}",
        f"critical: {critical_count}",
        f"high: {high_count}",
        f"medium: {medium_count}",
        "",
        "## Checklist",
        "",
    ]
    for _, row in post_mortem_report.iterrows():
        status = str(row["status"])
        area = str(row["area"])
        priority = str(row["priority"])
        gap = str(row["gap"])
        if status == "pass":
            lines.append(f"- [x] {priority} {area}: PASS ({gap or 'resolved'})")
        else:
            lines.append(f"- [ ] {priority} {area}: POST-MORTEM GATE ({gap})")
            lines.append(f"  - trajectory: {row['trajectory']}")
            lines.append(f"  - incident observed: {row['incident_observed']}")
            lines.append(f"  - postmortem insight: {row['post_mortem_insight']}")
            lines.append(f"  - prevention evidence source: {row['prevention_from_pre_mortem']}")
            lines.append(f"  - required proof: {row['required_proof']}")
            lines.append(f"  - evidence: {row['current_evidence']}")
            lines.append(f"  - source report: {row['source_report']}")
        lines.append("")

    latest_md = output_dir / "latest_post_mortem.md"
    checkpoint_md = output_dir / f"{run_id}.md"
    atomic_write_text(checkpoint_md, "\n".join(lines), encoding="utf-8")
    atomic_write_text(latest_md, checkpoint_md.read_text(encoding="utf-8"), encoding="utf-8")

    index_path = reports / "post_mortem_index.csv"
    index_frame = _read_csv_or_empty(index_path)
    if index_frame.empty:
        index_frame = pd.DataFrame(
            columns=[
                "run_id",
                "timestamp_utc",
                "open_gaps",
                "pass_gates",
                "critical",
                "high",
                "medium",
            ]
        )
    index_frame = pd.concat(
        [
            index_frame,
            pd.DataFrame(
                [
                    {
                        "run_id": run_id,
                        "timestamp_utc": timestamp,
                        "open_gaps": open_count,
                        "pass_gates": pass_count,
                        "critical": critical_count,
                        "high": high_count,
                        "medium": medium_count,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    _write_csv_atomic(index_frame, index_path)

    print(f"post_mortem_checklist_csv: {csv_path}")
    print(f"post_mortem_checklist_md: {checkpoint_md}")
    print(f"post_mortem_checkpoint: {latest_md}")
    print(f"post_mortem_index: {index_path}")
    return csv_path, checkpoint_md


def print_red_team_checklist(run_dir: Path | None = None) -> tuple[Path, Path]:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir or (reports / "red_team")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%SZ")
    run_id = f"red_team_{timestamp}"

    gap_frame = priority_gap_test_report()
    if gap_frame.empty:
        rt_frame = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "timestamp_utc": timestamp,
                    "priority": "N/A",
                    "area": "unknown",
                    "status": "blocked",
                    "severity": "high",
                    "gap": "priority_gap_test_report_empty",
                    "current_evidence": "",
                    "required_proof": "rerun_priority_gap_report",
                    "source_report": str(reports / "priority_gap_test.csv"),
                    "red_team_hypothesis": "No evidence present; do not proceed until gap evidence exists.",
                    "attack_vector": "Unknown",
                    "adversarial_question": "Can we trust this run for production decisions?",
                    "control_test": "Re-run gap test and complete readiness evidence before strategy deployment.",
                    "done": False,
                }
            ]
        )
    else:
        rows: list[dict[str, object]] = []
        for _, row in gap_frame.iterrows():
            priority = str(row["priority"])
            area = str(row["area"])
            status = str(row["status"])
            severity = str(row["severity"])
            gap = str(row["gap"] or "")
            required_proof = str(row["required_proof"] or "")
            evidence = str(row["current_evidence"] or "")
            source_report = str(row["source_report"] or "")
            rows.append(
                {
                    "run_id": run_id,
                    "timestamp_utc": timestamp,
                    "priority": priority,
                    "area": area,
                    "status": status,
                    "severity": severity,
                    "gap": gap,
                    "current_evidence": evidence,
                    "required_proof": required_proof,
                    "source_report": source_report,
                    "red_team_hypothesis": f"Could `{area}` be gamed by stale, leveraged, or mislabeled evidence?",
                    "attack_vector": (
                        "adversarial data assumptions, silent venue drift, model overfit, or operational bypass"
                        if status == "gap"
                        else "No active attack vector while gate is passed."
                    ),
                    "adversarial_question": (
                        f"What specific adversarial scenario could produce false confidence in `{area}` despite `{required_proof}`?"
                    ),
                    "control_test": _pre_mortem_prevention(area, required_proof),
                    "done": status == "pass",
                }
            )
        rt_frame = pd.DataFrame(rows)

    selected_cols = [
        "run_id",
        "timestamp_utc",
        "priority",
        "area",
        "status",
        "severity",
        "gap",
        "current_evidence",
        "required_proof",
        "source_report",
        "red_team_hypothesis",
        "attack_vector",
        "adversarial_question",
        "control_test",
        "done",
    ]
    red_team_report = rt_frame[selected_cols]

    csv_path = output_dir / f"{run_id}.csv"
    _write_csv_atomic(red_team_report, csv_path)

    open_count = int((red_team_report["status"] == "gap").sum())
    pass_count = int((red_team_report["status"] == "pass").sum())
    critical_count = int((red_team_report["severity"] == "critical").sum())
    high_count = int((red_team_report["severity"] == "high").sum())
    medium_count = int((red_team_report["severity"] == "medium").sum())

    lines: list[str] = [
        "# Red Team Checkpoint",
        "",
        f"run_id: {run_id}",
        f"created_utc: {timestamp}",
        f"open_gaps: {open_count} / {len(red_team_report)}",
        f"pass_gates: {pass_count}",
        f"critical: {critical_count}",
        f"high: {high_count}",
        f"medium: {medium_count}",
        "",
        "## Checklist",
        "",
    ]
    for _, row in red_team_report.iterrows():
        status = str(row["status"])
        area = str(row["area"])
        priority = str(row["priority"])
        gap = str(row["gap"])
        if status == "pass":
            lines.append(f"- [x] {priority} {area}: PASS ({gap or 'no red-team blocker'})")
        else:
            lines.append(f"- [ ] {priority} {area}: RED TEAM CHALLENGE ({gap})")
            lines.append(f"  - hypothesis: {row['red_team_hypothesis']}")
            lines.append(f"  - attack vector: {row['attack_vector']}")
            lines.append(f"  - adversarial question: {row['adversarial_question']}")
            lines.append(f"  - control test: {row['control_test']}")
            lines.append(f"  - required proof: {row['required_proof']}")
            lines.append(f"  - evidence: {row['current_evidence']}")
            lines.append(f"  - source report: {row['source_report']}")
        lines.append("")

    latest_md = output_dir / "latest_red_team.md"
    checkpoint_md = output_dir / f"{run_id}.md"
    atomic_write_text(checkpoint_md, "\n".join(lines), encoding="utf-8")
    atomic_write_text(latest_md, checkpoint_md.read_text(encoding="utf-8"), encoding="utf-8")

    index_path = reports / "red_team_index.csv"
    index_frame = _read_csv_or_empty(index_path)
    if index_frame.empty:
        index_frame = pd.DataFrame(
            columns=[
                "run_id",
                "timestamp_utc",
                "open_gaps",
                "pass_gates",
                "critical",
                "high",
                "medium",
            ]
        )
    index_frame = pd.concat(
        [
            index_frame,
            pd.DataFrame(
                [
                    {
                        "run_id": run_id,
                        "timestamp_utc": timestamp,
                        "open_gaps": open_count,
                        "pass_gates": pass_count,
                        "critical": critical_count,
                        "high": high_count,
                        "medium": medium_count,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    _write_csv_atomic(index_frame, index_path)

    print(f"red_team_checklist_csv: {csv_path}")
    print(f"red_team_checklist_md: {checkpoint_md}")
    print(f"red_team_checkpoint: {latest_md}")
    print(f"red_team_index: {index_path}")
    return csv_path, checkpoint_md


def print_supreme_team_checkpoint(run_dir: Path | None = None) -> tuple[Path, Path]:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir or (reports / "supreme_team")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%SZ")
    run_id = f"supreme_team_{timestamp}"

    gap_csv, _ = print_gap_analysis_checklist()
    pm_csv, _ = print_pre_mortem_checklist()
    post_csv, _ = print_post_mortem_checklist()
    rt_csv, _ = print_red_team_checklist()

    def _checkpoint_rows(path: Path, source: str) -> pd.DataFrame:
        frame = _read_csv_or_empty(path)
        if frame.empty:
            return pd.DataFrame(
                [
                    {
                        "run_id": run_id,
                        "timestamp_utc": timestamp,
                        "source_checkpoint": source,
                        "source_run_id": "",
                        "priority": "N/A",
                        "area": "unknown",
                        "status": "blocked",
                        "severity": "high",
                        "gap": f"{source}_missing_rows",
                        "current_evidence": "",
                        "required_proof": f"rerun {source.replace('_', '-')}",
                        "source_report": f"reports/{source}_index.csv",
                        "source_row": "",
                        "next_action": f"rerun {source.replace('_', '-')}",
                        "done": False,
                    }
                ]
            )
        local = frame.copy()
        local["run_id"] = (
            local.get("run_id", pd.Series(dtype=str)).fillna("").astype(str).replace({"": run_id})
        )
        local["source_checkpoint"] = source
        local["timestamp_utc"] = local.get(
            "timestamp_utc", pd.Series([timestamp] * len(local))
        ).fillna(timestamp)
        local["source_run_id"] = local["run_id"]
        local["source_row"] = source
        if "next_action" in local.columns:
            local["next_action"] = local["next_action"].fillna("")
        elif "prevention" in local.columns:
            local["next_action"] = local["prevention"].fillna("")
        elif source == "post_mortem":
            local["next_action"] = local["post_mortem_insight"].fillna("")
        elif source == "red_team":
            local["next_action"] = local["control_test"].fillna("")
        else:
            local["next_action"] = ""
        return local

    all_checkpoints = pd.concat(
        [
            _checkpoint_rows(gap_csv, "gap_analysis"),
            _checkpoint_rows(pm_csv, "pre_mortem"),
            _checkpoint_rows(post_csv, "post_mortem"),
            _checkpoint_rows(rt_csv, "red_team"),
        ],
        ignore_index=True,
    )
    seven_stage_path = reports / "active" / "seven_stage_goal_checkpoint.csv"
    seven_stage = _read_csv_or_empty(seven_stage_path)
    expected_stages = {str(value) for value in range(1, 8)}
    if (
        not seven_stage.empty
        and {"stage", "objective", "status", "evidence_progress", "blocker", "next_action"}
        <= set(seven_stage.columns)
        and set(seven_stage["stage"].astype(str)) == expected_stages
        and not seven_stage["stage"].astype(str).duplicated().any()
        and set(seven_stage["status"].astype(str)) <= {"PASS", "IN_PROGRESS", "BLOCKED"}
    ):
        legacy_current_state = (
            all_checkpoints.get("priority", pd.Series(dtype=str))
            .fillna("")
            .astype(str)
            .str.startswith("CS")
        )
        all_checkpoints = all_checkpoints.loc[~legacy_current_state].copy()
        canonical_rows: list[dict[str, object]] = []
        severity_by_stage = {
            "1": "medium",
            "2": "critical",
            "3": "critical",
            "4": "critical",
            "5": "high",
            "6": "high",
            "7": "high",
        }
        for _, stage_row in seven_stage.sort_values(
            "stage", key=lambda values: values.astype(int)
        ).iterrows():
            stage = str(stage_row["stage"])
            passed = str(stage_row["status"]) == "PASS"
            canonical_rows.append(
                {
                    "run_id": run_id,
                    "timestamp_utc": timestamp,
                    "source_checkpoint": "seven_stage",
                    "source_run_id": "",
                    "priority": f"S{stage}",
                    "area": f"seven_stage_{stage}_{stage_row['objective']}",
                    "status": "pass" if passed else "gap",
                    "severity": "none" if passed else severity_by_stage[stage],
                    "gap": ""
                    if passed
                    else str(stage_row.get("blocker", "") or stage_row["status"]),
                    "current_evidence": str(stage_row.get("evidence_progress", "") or ""),
                    "required_proof": f"stage_{stage}_pass_with_immutable_evidence",
                    "source_report": str(seven_stage_path),
                    "source_row": f"stage_{stage}",
                    "next_action": str(stage_row.get("next_action", "") or ""),
                    "done": passed,
                }
            )
        all_checkpoints = pd.concat(
            [all_checkpoints, pd.DataFrame(canonical_rows)],
            ignore_index=True,
            sort=False,
        )

    def _severity_rank(value: object) -> int:
        return {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4, "": 5}.get(
            str(value).strip().lower(), 5
        )

    worklist = all_checkpoints[all_checkpoints["status"] != "pass"].copy()
    if not worklist.empty:
        worklist["_priority_rank"] = worklist["priority"].map(_priority_sort_key)
        worklist["_severity_rank"] = worklist["severity"].map(_severity_rank)
        worklist = worklist.sort_values(
            ["_severity_rank", "_priority_rank", "source_checkpoint", "area"]
        ).reset_index(drop=True)
        worklist["rank"] = list(range(1, len(worklist) + 1))
        worklist_rows = [
            {
                "run_id": run_id,
                "timestamp_utc": timestamp,
                "rank": int(row["rank"]),
                "source_checkpoint": row["source_checkpoint"],
                "source_run_id": row.get("source_run_id", ""),
                "priority": row.get("priority", ""),
                "area": row.get("area", ""),
                "status": row.get("status", ""),
                "severity": row.get("severity", ""),
                "gap": row.get("gap", ""),
                "current_evidence": row.get("current_evidence", ""),
                "required_proof": row.get("required_proof", ""),
                "next_action": row.get("next_action", ""),
                "source_report": row.get("source_report", ""),
                "done": bool(row.get("done", False)),
                "source_row": row.get("source_row", ""),
            }
            for _, row in worklist.iterrows()
        ]
    else:
        worklist_rows = []

    plan_frame = pd.DataFrame(
        worklist_rows,
        columns=[
            "run_id",
            "timestamp_utc",
            "rank",
            "source_checkpoint",
            "source_run_id",
            "priority",
            "area",
            "status",
            "severity",
            "gap",
            "current_evidence",
            "required_proof",
            "next_action",
            "source_report",
            "done",
            "source_row",
        ],
    )

    open_count = int((plan_frame["status"] != "pass").sum()) if not plan_frame.empty else 0
    pass_count = int((all_checkpoints["status"] == "pass").sum())
    critical_count = int((all_checkpoints["severity"] == "critical").sum())
    high_count = int((all_checkpoints["severity"] == "high").sum())
    medium_count = int((all_checkpoints["severity"] == "medium").sum())

    csv_path = output_dir / f"{run_id}.csv"
    _write_csv_atomic(plan_frame, csv_path)

    lines: list[str] = [
        "# Supreme Team Checkpoint",
        "",
        f"run_id: {run_id}",
        f"created_utc: {timestamp}",
        f"open_actions: {open_count}",
        f"pass_gates: {pass_count}",
        f"critical: {critical_count}",
        f"high: {high_count}",
        f"medium: {medium_count}",
        "",
        "## Checkpoint Artifacts",
        f"- gap_analysis_checklist_csv: {gap_csv}",
        f"- pre_mortem_checklist_csv: {pm_csv}",
        f"- post_mortem_checklist_csv: {post_csv}",
        f"- red_team_checklist_csv: {rt_csv}",
        f"- seven_stage_checkpoint_csv: {seven_stage_path}",
        "",
        "## Supreme Team Next Actions",
    ]

    if plan_frame.empty:
        lines.extend(["- [x] No open actions; all checkpoints are passing."])
    else:
        for _, row in plan_frame.iterrows():
            rank = int(row["rank"])
            area = row["area"]
            priority = row["priority"]
            source = row["source_checkpoint"]
            status = row["status"]
            lines.append(
                f"- [ ] {rank}. {priority} {area} ({source}/{status}) "
                f"=> {row['severity']} | {row['gap']}"
            )
            lines.append(f"  - required proof: {row['required_proof']}")
            lines.append(f"  - action: {row['next_action']}")
            lines.append(f"  - evidence: {row['current_evidence']}")
            lines.append(f"  - source report: {row['source_report']}")
            lines.append("")

    checkpoint_md = output_dir / f"{run_id}.md"
    latest_md = output_dir / "latest_supreme_team.md"
    atomic_write_text(checkpoint_md, "\n".join(lines), encoding="utf-8")
    atomic_write_text(latest_md, checkpoint_md.read_text(encoding="utf-8"), encoding="utf-8")

    index_path = reports / "supreme_team_index.csv"
    index_frame = _read_csv_or_empty(index_path)
    if index_frame.empty:
        index_frame = pd.DataFrame(
            columns=[
                "run_id",
                "timestamp_utc",
                "open_actions",
                "pass_gates",
                "critical",
                "high",
                "medium",
            ]
        )
    index_frame = pd.concat(
        [
            index_frame,
            pd.DataFrame(
                [
                    {
                        "run_id": run_id,
                        "timestamp_utc": timestamp,
                        "open_actions": open_count,
                        "pass_gates": pass_count,
                        "critical": critical_count,
                        "high": high_count,
                        "medium": medium_count,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    _write_csv_atomic(index_frame, index_path)

    print(f"supreme_team_checklist_csv: {csv_path}")
    print(f"supreme_team_checklist_md: {checkpoint_md}")
    print(f"supreme_team_checkpoint: {latest_md}")
    print(f"supreme_team_index: {index_path}")
    return csv_path, checkpoint_md


def strategy_trade_count_gap_report(
    experiment_path: Path | None = None,
    required_trades: int = 100,
    output_path: Path | None = None,
) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "strategy_trade_count_gap.csv"
    frame = _read_csv_or_empty(experiment_path or (reports / "experiment_results.csv"))
    if frame.empty:
        empty = pd.DataFrame(
            [
                {
                    "strategy_id": "",
                    "strategy_name": "",
                    "pair": "",
                    "required_trades": required_trades,
                    "status": "blocked",
                    "base_trades": 0,
                    "stress_trades": 0,
                    "base_multiplier": 0,
                    "stress_multiplier": 0,
                    "pair_multiplier": 0,
                    "missing_cost_buckets": "base;stress",
                    "notes": "experiment_results missing or empty",
                }
            ]
        )
        _write_csv_atomic(empty, output)
        return empty

    evaluated = frame[frame["status"] == "evaluated"].copy()
    if evaluated.empty:
        empty = pd.DataFrame(
            [
                {
                    "strategy_id": "",
                    "strategy_name": "",
                    "pair": "",
                    "required_trades": required_trades,
                    "status": "blocked",
                    "base_trades": 0,
                    "stress_trades": 0,
                    "base_multiplier": 0,
                    "stress_multiplier": 0,
                    "pair_multiplier": 0,
                    "missing_cost_buckets": "base;stress",
                    "notes": "no evaluated rows in experiment_results",
                }
            ]
        )
        _write_csv_atomic(empty, output)
        return empty

    required_buckets = {"base", "stress"}
    grouped = evaluated.groupby(
        ["strategy_id", "strategy_name", "pair", "cost_bucket"], as_index=False
    ).agg(
        trades=("trades", "max"),
        observations=("observations", "max"),
    )

    rows: list[dict[str, object]] = []
    for (strategy_id, strategy_name, pair), scope in grouped.groupby(
        ["strategy_id", "strategy_name", "pair"]
    ):
        base = scope[scope["cost_bucket"] == "base"]
        stress = scope[scope["cost_bucket"] == "stress"]
        base_trades = int(base["trades"].max()) if not base.empty else 0
        stress_trades = int(stress["trades"].max()) if not stress.empty else 0
        base_obs = int(base["observations"].max()) if not base.empty else 0
        stress_obs = int(stress["observations"].max()) if not stress.empty else 0

        base_multiplier = 0
        stress_multiplier = 0
        if base_trades > 0:
            base_multiplier = max(1, int((required_trades + base_trades - 1) // base_trades))
        if stress_trades > 0:
            stress_multiplier = max(1, int((required_trades + stress_trades - 1) // stress_trades))

        present_buckets = set(str(v) for v in scope["cost_bucket"].dropna().unique())
        missing_cost_buckets = (
            ";".join(sorted(required_buckets.difference(present_buckets))) or "none"
        )
        if missing_cost_buckets != "none":
            notes = "missing_required_cost_bucket"
            pair_multiplier = 0
        elif base_trades >= required_trades and stress_trades >= required_trades:
            pair_multiplier = 1
            notes = "ready"
        elif base_trades > 0 and stress_trades > 0:
            pair_multiplier = max(base_multiplier, stress_multiplier)
            notes = "trade_frequency_limited"
        else:
            pair_multiplier = 0
            notes = "near_zero_signals_in_bucket"

        rows.append(
            {
                "strategy_id": int(strategy_id),
                "strategy_name": str(strategy_name),
                "pair": str(pair),
                "required_trades": required_trades,
                "status": "ready" if notes == "ready" else "gap",
                "base_trades": base_trades,
                "stress_trades": stress_trades,
                "base_observations": base_obs,
                "stress_observations": stress_obs,
                "base_multiplier": base_multiplier,
                "stress_multiplier": stress_multiplier,
                "pair_multiplier": pair_multiplier,
                "missing_cost_buckets": missing_cost_buckets,
                "notes": notes,
            }
        )

    if not rows:
        result = pd.DataFrame()
    else:
        result = pd.DataFrame(rows).sort_values(
            ["pair_multiplier", "strategy_id", "strategy_name", "pair"],
            ascending=[True, True, True, True],
        )
    _write_csv_atomic(result, output)
    return result


def print_strategy_trade_count_gap(
    experiment_path: Path | None = None,
    required_trades: int = 100,
    output_path: Path | None = None,
) -> None:
    output = output_path or ROOT / "reports" / "strategy_trade_count_gap.csv"
    frame = strategy_trade_count_gap_report(experiment_path, required_trades, output)
    if frame.empty:
        print("strategy_trade_count_gap: no rows found")
    else:
        status_counts = frame["status"].value_counts().to_dict()
        ready_count = int(status_counts.get("ready", 0))
        gap_count = int(status_counts.get("gap", 0))
        print(
            f"strategy_trade_count_gap: total_rows={len(frame)} ready={ready_count} gap={gap_count}"
        )
        print("ready_examples=", end=" ")
        ready_examples = (
            frame[frame["status"] == "ready"]
            .head(5)[["strategy_id", "strategy_name", "pair", "base_trades", "pair_multiplier"]]
            .to_string(index=False)
        )
        print(ready_examples if ready_examples.strip() else "(none)")
        print("top_gaps=", end=" ")
        top_gaps = (
            frame[frame["status"] == "gap"]
            .head(5)[
                ["strategy_id", "strategy_name", "pair", "base_trades", "stress_trades", "notes"]
            ]
            .to_string(index=False)
        )
        print(top_gaps if top_gaps.strip() else "(none)")
    print(f"strategy_trade_count_gap: {output}")


def print_dydx_execution_checklist(output_path: Path | None = None) -> None:
    output = output_path or ROOT / "reports" / "dydx_execution_checklist.csv"
    frame = dydx_execution_checklist_report(output)
    print(frame.to_string(index=False))
    print(f"dydx_execution_checklist: {output}")


def priority_readiness_report(
    output_path: Path | None = None,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    effective_root = root or ROOT
    with _local_env_for_reports(effective_root):
        return _priority_readiness_report(output_path, root=effective_root)


def _priority_readiness_report(
    output_path: Path | None = None,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    effective_root = root or ROOT
    reports = effective_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "priority_readiness.csv"
    rows: list[dict[str, object]] = []

    live_dictionary = effective_root / "docs" / "crypto_wizards_live_field_dictionary.csv"
    raw_payloads = sorted(
        path
        for path in (effective_root / "data" / "raw").glob("*.json")
        if not path.name.startswith("crypto_wizards_pair_metrics_sample")
    )
    live_ready = bool(raw_payloads) and live_dictionary.exists()
    rows.append(
        _readiness_row(
            priority="P1",
            gate="crypto_wizards_live_artifacts",
            ready=live_ready,
            evidence=f"payloads={len(raw_payloads)};dictionary_exists={live_dictionary.exists()}",
            blocker="" if live_ready else "missing_live_payload_or_dictionary",
            next_action="crawl or import Crypto Wizards payloads"
            if not live_ready
            else "continue field coverage checks",
        )
    )

    pair_detail_dir = effective_root / "data" / "raw" / "pair_details"
    pair_detail_cache_status = "NO_SOURCE_DIRECTORY"
    if pair_detail_dir.exists():
        pair_detail_evidence = load_or_refresh_pair_detail_evidence_cache(
            pair_detail_dir,
            reports,
        )
        history_rows = list(pair_detail_evidence["history_rows"])
        quality_rows = list(pair_detail_evidence["quality_rows"])
        pair_detail_cache_status = str(pair_detail_evidence["cache_status"])
    else:
        history_rows = []
        quality_rows = []
    experiment_ready = [row for row in history_rows if bool(row.get("experiment_ready"))]
    ecm_ready = [row for row in history_rows if bool(row.get("ecm_history_ready"))]
    two_leg_ready = [row for row in history_rows if bool(row.get("two_leg_execution_ready"))]
    research_usable = [row for row in quality_rows if bool(row.get("research_usable"))]
    execution_usable = [row for row in quality_rows if bool(row.get("execution_usable"))]
    rows.append(
        _readiness_row(
            priority="P1",
            gate="pair_detail_history",
            ready=bool(experiment_ready and ecm_ready),
            evidence=(
                f"snapshots={len(history_rows)};experiment_ready={len(experiment_ready)};"
                f"ecm_ready={len(ecm_ready)};cache={pair_detail_cache_status}"
            ),
            blocker=""
            if experiment_ready and ecm_ready
            else "missing_spread_zscore_or_ecm_history",
            next_action="import authenticated pair-detail capture with spread/zscore/ecm arrays"
            if not (experiment_ready and ecm_ready)
            else "run pair-detail experiments",
        )
    )

    rows.append(
        _readiness_row(
            priority="P1",
            gate="pair_detail_two_leg_execution_history",
            ready=bool(two_leg_ready),
            evidence=f"snapshots={len(history_rows)};two_leg_ready={len(two_leg_ready)}",
            blocker="" if two_leg_ready else "missing_price_x_or_price_y_history",
            next_action="capture price_x and price_y arrays for execution-realistic two-leg backtests"
            if not two_leg_ready
            else "run execution-realistic pair-detail experiments",
        )
    )

    rows.append(
        _readiness_row(
            priority="P1",
            gate="pair_detail_quality",
            ready=bool(research_usable),
            evidence=(
                f"snapshots={len(quality_rows)};"
                f"research_usable={len(research_usable)};"
                f"execution_usable={len(execution_usable)};"
                f"cache={pair_detail_cache_status}"
            ),
            blocker="" if research_usable else "no_research_usable_pair_detail_history",
            next_action="run strategy research on quality-accepted histories"
            if research_usable
            else "capture more 5-minute pairs or reject stale/illiquid pairs",
        )
    )

    capture_report_path = reports / "pair_detail_capture_audit.csv"
    cached_capture = _read_csv_or_empty(capture_report_path)
    if not cached_capture.empty:
        capture_rows = [
            {
                **row,
                "experiment_ready": _coerce_bool(row.get("experiment_ready")),
                "ecm_history_ready": _coerce_bool(row.get("ecm_history_ready")),
                "two_leg_execution_ready": _coerce_bool(row.get("two_leg_execution_ready")),
            }
            for row in cached_capture.to_dict("records")
        ]
    else:
        capture_checklist_path = reports / "pair_detail_capture_checklist.csv"
        cached_checklist = _read_csv_or_empty(capture_checklist_path)
        if not cached_checklist.empty:
            capture_rows = []
            for row in cached_checklist.to_dict("records"):
                path = _md_text(row.get("path") or row.get("source_path"))
                if not path:
                    continue
                capture_rows.append(
                    {
                        "path": path,
                        "pair": _md_text(row.get("pair", "")),
                        "json_path": path,
                        "candidate_type": "cached_checklist",
                        "row_count": int(
                            float(pd.to_numeric(row.get("history_rows", 0), errors="coerce") or 0)
                        ),
                        "columns": "",
                        "experiment_ready": _coerce_bool(
                            row.get("experiment_ready", row.get("baseline_ready", False))
                        ),
                        "missing_for_baseline_backtest": "",
                        "ecm_history_ready": _coerce_bool(
                            row.get("ecm_history_ready", row.get("ecm_ready", False))
                        ),
                        "missing_for_ecm_backtest": "",
                        "two_leg_execution_ready": _coerce_bool(
                            row.get("two_leg_execution_ready", row.get("two_leg_ready", False))
                        ),
                        "missing_for_two_leg_backtest": "",
                        "hedge_ratio_available": "",
                        "beta_available": "",
                        "funding_columns_available": "",
                        "execution_assumption_notes": "",
                    }
                )
        else:
            capture_rows = (
                pair_detail_capture_audit(pair_detail_dir) if pair_detail_dir.exists() else []
            )
    if capture_rows:
        _write_csv_atomic(
            pd.DataFrame(capture_rows, columns=PAIR_DETAIL_CAPTURE_AUDIT_COLUMNS),
            capture_report_path,
        )
    capture_experiment_ready = [row for row in capture_rows if bool(row.get("experiment_ready"))]
    capture_ecm_ready = [row for row in capture_rows if bool(row.get("ecm_history_ready"))]
    capture_two_leg_ready = [
        row for row in capture_rows if bool(row.get("two_leg_execution_ready"))
    ]
    rows.append(
        _readiness_row(
            priority="P1",
            gate="pair_detail_capture_audit",
            ready=bool(capture_experiment_ready and capture_ecm_ready and capture_two_leg_ready),
            evidence=(
                f"candidate_paths={len(capture_rows)};"
                f"experiment_ready_paths={len(capture_experiment_ready)};"
                f"ecm_ready_paths={len(capture_ecm_ready)};"
                f"two_leg_ready_paths={len(capture_two_leg_ready)}"
            ),
            blocker=""
            if capture_experiment_ready and capture_ecm_ready and capture_two_leg_ready
            else "no_nested_execution_ready_history_candidate_detected",
            next_action="run updated browser capture helper on authenticated pair page"
            if not (capture_experiment_ready and capture_ecm_ready and capture_two_leg_ready)
            else "import capture and run experiments",
        )
    )

    acceptance_path = _acceptance_report_path(effective_root)
    acceptance_checklist_path = reports / "strategy_acceptance_checklist.csv"
    strategy_acceptance_checklist_report(
        acceptance_checklist_path,
        root=effective_root,
    )
    research_unblock_path = reports / "research_unblock_plan.csv"
    research_unblock_plan_report(research_unblock_path, root=effective_root)
    if acceptance_path.exists():
        acceptance = _augmented_acceptance_frame(reports, root=effective_root)
        production_ready = int(
            acceptance.get("production_eligible", pd.Series(dtype=bool)).fillna(False).sum()
        )
        preferred_ready = int(
            acceptance.get("preferred_eligible", pd.Series(dtype=bool)).fillna(False).sum()
        )
        two_leg_pairs_tested = _max_int_column(acceptance, "two_leg_pairs_tested")
        two_leg_passing_pairs = _max_int_column(acceptance, "two_leg_passing_pairs")
        total_strategies = len(acceptance)
        strategy_ready = production_ready > 0
        strategy_evidence = (
            f"strategies={total_strategies};production_eligible={production_ready};"
            f"preferred_eligible={preferred_ready};max_two_leg_pairs_tested={two_leg_pairs_tested};"
            f"max_two_leg_passing_pairs={two_leg_passing_pairs};"
            f"checklist={acceptance_checklist_path}"
        )
    else:
        strategy_ready = False
        strategy_evidence = f"acceptance_report_exists=False;checklist={acceptance_checklist_path}"
    rows.append(
        _readiness_row(
            priority="P2",
            gate="strategy_acceptance",
            ready=strategy_ready,
            evidence=strategy_evidence,
            blocker="" if strategy_ready else "no_strategy_passes_production_gates",
            next_action=f"review {research_unblock_path} and collect the highest-impact missing history/features"
            if not strategy_ready
            else "allow research-gated paper plans",
        )
    )

    config: object = DydxNetworkConfig.paper_testnet_from_env()
    order_client, order_adapter_error = _load_dydx_order_client_adapter()
    adapter_contract = validate_dydx_order_client_adapter()
    order_adapter_loaded = order_client is not None and not order_adapter_error
    order_adapter_ready = (
        order_client is not None
        and not order_adapter_error
        and bool(adapter_contract["valid"])
        and bool(adapter_contract["exchange_submission_capable"])
    )
    dydx_report = dydx_readiness_report(
        config=config,
        order_client_wired=order_adapter_loaded,
        indexer_adapter_wired=build_dydx_indexer_adapter(config) is not None,
    )
    dydx_checklist_path = reports / "dydx_execution_checklist.csv"
    dydx_execution_checklist_report(dydx_checklist_path, root=effective_root)
    dydx_blockers = list(dydx_report.get("blockers", []))
    if order_adapter_error or (adapter_contract["configured"] and not adapter_contract["valid"]):
        dydx_blockers.append("invalid_dydx_order_client_adapter")
    elif (
        adapter_contract["configured"]
        and adapter_contract["valid"]
        and not adapter_contract["exchange_submission_capable"]
    ):
        dydx_blockers.append("record_only_dydx_order_client_adapter")
    dydx_ready = len(dydx_blockers) == 0
    dydx_report["ready_for_paper_submission"] = dydx_ready
    rows.append(
        _readiness_row(
            priority="P3",
            gate="dydx_testnet_readiness",
            ready=bool(dydx_ready),
            evidence=(
                f"indexer={dydx_report['dydx_indexer_adapter_wired']};"
                f"order_adapter={dydx_report['dydx_order_client_adapter_wired']};"
                f"adapter_contract_valid={adapter_contract['valid']};"
                f"exchange_submission_capable={adapter_contract['exchange_submission_capable']};"
                f"record_only={adapter_contract['record_only']};"
                f"submit_orders={dydx_report['submit_orders']};"
                f"checklist={dydx_checklist_path}"
            ),
            blocker=";".join(dydx_blockers),
            next_action="keep order submission disabled until research passes and order adapter is injected"
            if dydx_blockers
            else "submit only research-accepted paper plans",
        )
    )

    paper_gate_ready = strategy_ready and bool(dydx_ready) and order_adapter_ready
    rows.append(
        _readiness_row(
            priority="P4",
            gate="paper_execution_gate",
            ready=paper_gate_ready,
            evidence=f"strategy_ready={strategy_ready};dydx_ready={dydx_report['ready_for_paper_submission']}",
            blocker="" if paper_gate_ready else "strategy_or_dydx_gate_not_ready",
            next_action="paper trade only accepted strategies"
            if paper_gate_ready
            else "do not submit paper orders yet",
        )
    )

    paper_journal = reports / "paper_trading_journal.csv"
    trade_store = effective_root / "data" / "meta_learning" / "trades.jsonl"
    learning_summary_path = reports / "learning_event_summary.csv"
    write_learning_event_summary_report(paper_journal, trade_store, learning_summary_path)
    learning_summary = _read_csv_or_empty(learning_summary_path)
    combined_learning = learning_summary[
        learning_summary.get("source", pd.Series(dtype=str)) == "combined"
    ]
    combined_row = (
        combined_learning.iloc[0] if not combined_learning.empty else pd.Series(dtype=object)
    )
    paper_journal_rows = _csv_row_count(paper_journal)
    trade_store_rows = _jsonl_row_count(trade_store)
    learning_events = int(combined_row.get("events", 0) or 0)
    learning_outcomes = int(combined_row.get("outcome_events", 0) or 0)
    learning_audit_only = int(combined_row.get("audit_only_events", 0) or 0)
    learning_outcomes_remaining = int(combined_row.get("outcome_events_remaining", 100) or 0)
    learning_ready_for_modeling = strict_bool(combined_row.get("ready_for_modeling", False))
    learning_ready = learning_ready_for_modeling
    learning_blocker = (
        "missing_learning_events" if learning_events == 0 else "missing_model_ready_outcomes"
    )
    rows.append(
        _readiness_row(
            priority="P5",
            gate="learning_event_store",
            ready=learning_ready,
            evidence=(
                f"paper_journal_exists={paper_journal.exists()};paper_journal_rows={paper_journal_rows};"
                f"trade_store_exists={trade_store.exists()};trade_store_rows={trade_store_rows};"
                f"events={learning_events};outcomes={learning_outcomes};audit_only={learning_audit_only};"
                f"outcomes_remaining={learning_outcomes_remaining};ready_for_modeling={learning_ready_for_modeling};"
                f"summary_report={learning_summary_path}"
            ),
            blocker="" if learning_ready else learning_blocker,
            next_action="train outcome/feature-importance models from recorded events"
            if learning_ready
            else "append realized trade outcomes once research-gated paper signals exist",
        )
    )

    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, output)
    _write_csv_atomic(priority_action_plan(frame), reports / "priority_action_plan.csv")
    dashboard = priority_spine_dashboard_report(
        frame,
        reports / "priority_spine_dashboard.csv",
        root=effective_root,
    )
    _priority_gap_frame_from_dashboard(dashboard, reports / "priority_gap_test.csv")
    return frame


def print_priority_readiness(output_path: Path | None = None) -> None:
    frame = priority_readiness_report(output_path)
    print(frame.to_string(index=False))
    print(
        f"priority_readiness_report: {output_path or ROOT / 'reports' / 'priority_readiness.csv'}"
    )
    print(f"priority_action_plan: {ROOT / 'reports' / 'priority_action_plan.csv'}")


def priority_action_plan(
    readiness: pd.DataFrame | None = None, output_path: Path | None = None
) -> pd.DataFrame:
    readiness = readiness if readiness is not None else priority_readiness_report()
    blocked = readiness[~readiness["ready"].astype(bool)].copy()
    if blocked.empty:
        frame = pd.DataFrame(
            columns=["rank", "priority", "gate", "blocker", "next_action", "evidence", "depends_on"]
        )
    else:
        blocked["priority_rank"] = blocked["priority"].map(_priority_sort_key)
        blocked["gate_rank"] = blocked["gate"].map(_gate_sort_key)
        blocked = blocked.sort_values(["priority_rank", "gate_rank", "gate"]).reset_index(drop=True)
        rows = []
        for index, row in blocked.iterrows():
            rows.append(
                {
                    "rank": index + 1,
                    "priority": row["priority"],
                    "gate": row["gate"],
                    "blocker": row["blocker"],
                    "next_action": row["next_action"],
                    "evidence": row["evidence"],
                    "depends_on": _gate_dependency(str(row["gate"])),
                }
            )
        frame = pd.DataFrame(rows)
    if output_path is not None:
        _write_csv_atomic(frame, output_path)
    return frame


def print_priority_actions() -> None:
    output = ROOT / "reports" / "priority_action_plan.csv"
    frame = priority_action_plan(output_path=output)
    print(frame.to_string(index=False))
    print(f"priority_action_plan: {output}")


def write_learning_report() -> None:
    reports = ROOT / "reports"
    output = reports / "learning_event_summary.csv"
    path = write_learning_event_summary_report(
        reports / "paper_trading_journal.csv",
        ROOT / "data" / "meta_learning" / "trades.jsonl",
        output,
    )
    print(pd.read_csv(path).to_string(index=False))
    print(f"learning_event_summary: {path}")


def learning_outcome_template_report(output_path: Path | None = None) -> pd.DataFrame:
    output = output_path or ROOT / "data" / "meta_learning" / "learning_outcome_template.csv"
    frame = pd.DataFrame(
        [
            {
                "trade_id": "",
                "pair": "",
                "strategy_id": "",
                "realized_return": "",
                "signal": "",
                "hedge_ratio": "",
                "beta": "",
                "notional_usd": "",
                "regime": "unknown",
            }
        ],
        columns=LEARNING_OUTCOME_TEMPLATE_COLUMNS,
    )
    _write_csv_atomic(frame, output)
    return frame


def print_learning_outcome_template(output_path: Path | None = None) -> None:
    output = output_path or ROOT / "data" / "meta_learning" / "learning_outcome_template.csv"
    frame = learning_outcome_template_report(output)
    print(frame.to_string(index=False))
    print(f"learning_outcome_template_rows: {len(frame)}")
    print(f"learning_outcome_template: {output}")


def seed_learning_outcome_template_from_paper_journal(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    source = input_path or ROOT / "reports" / "paper_trading_journal.csv"
    output = output_path or ROOT / "data" / "meta_learning" / "learning_outcome_template.csv"
    journal = _read_csv_or_empty(source)
    rows: list[dict[str, object]] = []
    if not journal.empty:
        statuses = journal.get("plan_status", pd.Series(dtype=str)).fillna("").astype(str)
        eligible = statuses.isin(
            ["paper_ready", "paper_submitted", "confirmed_on_exchange", "paper_completed"]
        )
        for _, row in journal.loc[eligible].iterrows():
            fill_statuses: list[str] = []
            try:
                parsed_fills = json.loads(str(row.get("fills_json", "")) or "[]")
            except json.JSONDecodeError:
                parsed_fills = []
            if isinstance(parsed_fills, list):
                for item in parsed_fills:
                    if isinstance(item, dict) and item.get("status") is not None:
                        fill_statuses.append(str(item["status"]))
            if not any(
                status in {"paper_submitted", "confirmed_on_exchange"} for status in fill_statuses
            ):
                continue
            pair = str(row.get("pair", "")).strip()
            strategy_id = str(row.get("strategy_id", "")).strip()
            timestamp = str(row.get("timestamp_utc", "")).strip()
            if not pair or not strategy_id:
                continue
            trade_id = f"{timestamp}_{pair}_{strategy_id}" if timestamp else f"{pair}_{strategy_id}"
            rows.append(
                {
                    "trade_id": trade_id,
                    "pair": pair,
                    "strategy_id": strategy_id,
                    "realized_return": "",
                    "signal": "",
                    "hedge_ratio": "",
                    "beta": "",
                    "notional_usd": "",
                    "regime": "unknown",
                }
            )
    frame = pd.DataFrame(rows, columns=LEARNING_OUTCOME_TEMPLATE_COLUMNS)
    _write_csv_atomic(frame, output)
    return frame


def print_seed_learning_outcome_template_from_paper_journal(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> None:
    output = output_path or ROOT / "data" / "meta_learning" / "learning_outcome_template.csv"
    source = input_path or ROOT / "reports" / "paper_trading_journal.csv"
    frame = seed_learning_outcome_template_from_paper_journal(
        input_path=input_path, output_path=output
    )
    print(frame.to_string(index=False))
    print(f"seeded_rows: {len(frame)}")
    print(f"paper_trading_journal_source: {source}")
    print(f"learning_outcome_template: {output}")


def print_trade_timing_template(output_path: Path | None = None) -> None:
    output = output_path or TRADE_TIMING_DEFAULT_TEMPLATE
    path = write_trade_timing_template(output)
    print(pd.DataFrame(columns=TRADE_TIMING_TEMPLATE_COLUMNS).to_string(index=False))
    print(f"trade_timing_template: {path}")


def trade_timing_comparison_report(
    trades_path: Path | None = None,
    history_path: Path | None = None,
    output_path: Path | None = None,
    *,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
) -> pd.DataFrame:
    if trades_path is None:
        raise SystemExit(
            "trade-timing-comparison-report requires --input-dir pointing to a trades CSV"
        )
    if history_path is None:
        raise SystemExit("trade-timing-comparison-report requires --history-path")
    trades = pd.read_csv(trades_path, dtype=str).fillna("")
    history = load_trade_timing_history(history_path)
    report = trade_timing_comparison_report_frame(
        trades,
        history,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
    )
    output = output_path or ROOT / "reports" / "trade_timing_comparison_report.csv"
    _write_csv_atomic(report, output)
    summary = trade_timing_comparison_summary(report)
    _write_csv_atomic(summary, output.with_name(f"{output.stem}_summary.csv"))
    return report


def print_trade_timing_comparison_report(
    trades_path: Path | None = None,
    history_path: Path | None = None,
    output_path: Path | None = None,
    *,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
) -> None:
    output = output_path or ROOT / "reports" / "trade_timing_comparison_report.csv"
    report = trade_timing_comparison_report(
        trades_path=trades_path,
        history_path=history_path,
        output_path=output,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
    )
    summary_path = output.with_name(f"{output.stem}_summary.csv")
    summary = _read_csv_or_empty(summary_path)
    print(report.to_string(index=False))
    if not summary.empty:
        print(summary.to_string(index=False))
    print(f"trade_timing_comparison_report: {output}")
    print(f"trade_timing_comparison_summary: {summary_path}")


def learning_outcome_template_check_report(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    source = input_path or ROOT / "data" / "meta_learning" / "learning_outcome_template.csv"
    output = output_path or ROOT / "reports" / "learning_outcome_template_check.csv"
    if not source.exists():
        frame = pd.DataFrame(
            [
                {
                    "path": str(source),
                    "rows": 0,
                    "ready_rows": 0,
                    "blocked_rows": 0,
                    "missing_columns": ";".join(LEARNING_OUTCOME_REQUIRED_COLUMNS),
                    "invalid_rows": "",
                    "ready_to_append": False,
                    "next_action": "create learning outcome template and fill realized outcomes",
                }
            ]
        )
        _write_csv_atomic(frame, output)
        return frame
    try:
        data = pd.read_csv(source, dtype=str).fillna("")
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        data = pd.DataFrame()
    missing_columns, ready_indices, invalid_rows = _learning_outcome_template_validation(data)
    ready_rows = len(ready_indices)
    blocked_rows = max(len(data) - ready_rows, 0) if not missing_columns else len(data)
    next_action = (
        "append ready rows with append-learning-outcome"
        if ready_rows and not invalid_rows and not missing_columns
        else "fill required columns: pair,strategy_id,realized_return"
    )
    frame = pd.DataFrame(
        [
            {
                "path": str(source),
                "rows": len(data),
                "ready_rows": ready_rows,
                "blocked_rows": blocked_rows,
                "missing_columns": ";".join(missing_columns),
                "invalid_rows": ";".join(invalid_rows),
                "ready_to_append": bool(ready_rows and not invalid_rows and not missing_columns),
                "next_action": next_action,
            }
        ]
    )
    _write_csv_atomic(frame, output)
    return frame


def print_learning_outcome_template_check(
    input_path: Path | None = None, output_path: Path | None = None
) -> None:
    output = output_path or ROOT / "reports" / "learning_outcome_template_check.csv"
    frame = learning_outcome_template_check_report(input_path, output)
    print(frame.to_string(index=False))
    print(f"learning_outcome_template_check: {output}")


def import_learning_outcomes_from_template(
    input_path: Path | None = None,
    trade_store_path: Path | None = None,
    report_path: Path | None = None,
) -> pd.DataFrame:
    source = input_path or ROOT / "data" / "meta_learning" / "learning_outcome_template.csv"
    output = report_path or ROOT / "reports" / "learning_outcome_import_report.csv"
    if not source.exists():
        frame = pd.DataFrame(
            [
                {
                    "path": str(source),
                    "rows": 0,
                    "imported_rows": 0,
                    "blocked_rows": 0,
                    "missing_columns": ";".join(LEARNING_OUTCOME_REQUIRED_COLUMNS),
                    "invalid_rows": "",
                    "trade_store": str(
                        trade_store_path or ROOT / "data" / "meta_learning" / "trades.jsonl"
                    ),
                    "status": "blocked",
                    "next_action": "create learning outcome template and fill realized outcomes",
                }
            ]
        )
        _write_csv_atomic(frame, output)
        return frame
    try:
        data = pd.read_csv(source, dtype=str).fillna("")
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        data = pd.DataFrame()
    missing_columns, ready_indices, invalid_rows = _learning_outcome_template_validation(data)
    store_path = trade_store_path or ROOT / "data" / "meta_learning" / "trades.jsonl"
    existing_trade_ids = JsonlTradeStore(store_path).trade_ids()
    imported = 0
    duplicate_rows: list[str] = []
    if not missing_columns:
        for index in ready_indices:
            row = data.loc[index]
            trade_id = str(row.get("trade_id", "")).strip() or None
            if trade_id is not None and trade_id in existing_trade_ids:
                duplicate_rows.append(f"row_{index + 2}[trade_id={trade_id}]")
                continue
            append_learning_outcome(
                pair=str(row.get("pair", "")).strip(),
                strategy_id=int(float(str(row.get("strategy_id", "")).strip())),
                realized_return=float(str(row.get("realized_return", "")).strip()),
                signal=_optional_float(row.get("signal")),
                hedge_ratio=_optional_float(row.get("hedge_ratio")),
                beta=_optional_float(row.get("beta")),
                notional_usd=_optional_float(row.get("notional_usd")),
                regime=str(row.get("regime", "") or "unknown").strip() or "unknown",
                trade_id=trade_id,
                trade_store_path=store_path,
            )
            imported += 1
            if trade_id is not None:
                existing_trade_ids.add(trade_id)
    blocked_rows = len(invalid_rows) if not missing_columns else len(data)
    status = "imported" if imported and not invalid_rows and not missing_columns else "blocked"
    if imported == 0 and duplicate_rows and not invalid_rows and not missing_columns:
        status = "skipped_duplicates"
    frame = pd.DataFrame(
        [
            {
                "path": str(source),
                "rows": len(data),
                "imported_rows": imported,
                "blocked_rows": blocked_rows,
                "duplicate_rows": len(duplicate_rows),
                "missing_columns": ";".join(missing_columns),
                "invalid_rows": ";".join(invalid_rows),
                "duplicate_details": ";".join(duplicate_rows),
                "trade_store": str(store_path),
                "status": status,
                "next_action": "rerun learning-report"
                if imported or duplicate_rows
                else "fill required columns: pair,strategy_id,realized_return",
            }
        ]
    )
    _write_csv_atomic(frame, output)
    return frame


def print_import_learning_outcomes(
    input_path: Path | None = None, output_path: Path | None = None
) -> None:
    frame = import_learning_outcomes_from_template(input_path=input_path, report_path=output_path)
    output = output_path or ROOT / "reports" / "learning_outcome_import_report.csv"
    print(frame.to_string(index=False))
    print(f"learning_outcome_import_report: {output}")


def _learning_outcome_template_validation(
    frame: pd.DataFrame,
) -> tuple[list[str], list[int], list[str]]:
    missing_columns = [
        column for column in LEARNING_OUTCOME_REQUIRED_COLUMNS if column not in frame.columns
    ]
    invalid_rows: list[str] = []
    ready_indices: list[int] = []
    if missing_columns or frame.empty:
        return missing_columns, ready_indices, invalid_rows
    for index, row in frame.iterrows():
        missing = [
            column
            for column in LEARNING_OUTCOME_REQUIRED_COLUMNS
            if str(row.get(column, "")).strip() == ""
        ]
        numeric_errors = []
        for column in ("strategy_id", "realized_return"):
            value = str(row.get(column, "")).strip()
            if value:
                try:
                    int(float(value)) if column == "strategy_id" else float(value)
                except ValueError:
                    numeric_errors.append(column)
        if missing or numeric_errors:
            invalid_rows.append(
                f"row_{index + 2}[missing={'+'.join(missing) or 'none'},invalid={'+'.join(numeric_errors) or 'none'}]"
            )
        else:
            ready_indices.append(int(index))
    return missing_columns, ready_indices, invalid_rows


def _optional_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    return float(text)


def append_learning_outcome(
    pair: str,
    strategy_id: int,
    realized_return: float,
    signal: float | None = None,
    hedge_ratio: float | None = None,
    beta: float | None = None,
    notional_usd: float | None = None,
    regime: str = "unknown",
    trade_id: str | None = None,
    trade_store_path: Path | None = None,
) -> Path:
    if not pair:
        raise SystemExit("append-learning-outcome requires --pair")
    strategy = next((spec for spec in STRATEGIES if spec.id == strategy_id), None)
    strategy_name = strategy.name if strategy is not None else f"strategy_{strategy_id}"
    timestamp = pd.Timestamp.now(tz="UTC").to_pydatetime()
    record = TradeRecord(
        trade_id=trade_id or f"{pair}-{strategy_id}-{timestamp.isoformat()}",
        timestamp=timestamp,
        pair=pair,
        strategy=strategy_name,
        regime=regime,
        features={
            "hedge_ratio": float(hedge_ratio) if hedge_ratio is not None else 1.0,
            "beta": float(beta) if beta is not None else 1.0,
        },
        signal={"value": float(signal) if signal is not None else 0.0},
        execution={
            "venue": "dydx_testnet",
            "notional_usd": float(notional_usd) if notional_usd is not None else 0.0,
        },
        outcome={"realized_return": float(realized_return)},
    )
    path = trade_store_path or ROOT / "data" / "meta_learning" / "trades.jsonl"
    store = JsonlTradeStore(path)
    if trade_id:
        store.append_if_new(record)
    else:
        store.append(record)
    return path


def run_append_learning_outcome(
    pair: str | None,
    strategy_id: int | None,
    realized_return: float | None,
    signal: float | None,
    hedge_ratio: float | None,
    beta: float | None,
    notional_usd: float | None,
    regime: str,
    trade_id: str | None,
    output_path: Path | None,
) -> None:
    if pair is None or strategy_id is None or realized_return is None:
        raise SystemExit(
            "append-learning-outcome requires --pair, --strategy-id, and --realized-return"
        )
    path = append_learning_outcome(
        pair=pair,
        strategy_id=strategy_id,
        realized_return=realized_return,
        signal=signal,
        hedge_ratio=hedge_ratio,
        beta=beta,
        notional_usd=notional_usd,
        regime=regime,
        trade_id=trade_id,
        trade_store_path=output_path,
    )
    print(f"learning_trade_store: {path}")


def _readiness_row(
    priority: str,
    gate: str,
    ready: bool,
    evidence: str,
    blocker: str,
    next_action: str,
) -> dict[str, object]:
    return {
        "priority": priority,
        "gate": gate,
        "ready": ready,
        "status": "ready" if ready else "blocked",
        "evidence": evidence,
        "blocker": blocker,
        "next_action": next_action,
    }


def _execution_check_row(
    step: str, ready: bool, blocker: str, evidence: str, next_action: str
) -> dict[str, object]:
    return {
        "step": step,
        "ready": ready,
        "status": "ready" if ready else "blocked",
        "blocker": blocker,
        "evidence": evidence,
        "next_action": next_action,
    }


def _dashboard_row(
    priority: str,
    area: str,
    ready: bool,
    blocker: str,
    key_metric: str,
    source_report: str,
    next_action: str,
) -> dict[str, object]:
    return {
        "priority": priority,
        "area": area,
        "ready": ready,
        "status": "ready" if ready else "blocked",
        "blocker": blocker,
        "key_metric": key_metric,
        "source_report": source_report,
        "next_action": next_action,
    }


def _all_gates_ready(gates: pd.DataFrame, gate_names: list[str]) -> bool:
    return all(_gate_ready_from_index(gates, gate) for gate in gate_names)


def _gate_ready_from_index(gates: pd.DataFrame, gate: str) -> bool:
    if gates.empty or gate not in gates.index or "ready" not in gates.columns:
        return False
    return bool(gates.loc[gate, "ready"])


def _gate_value(gates: pd.DataFrame, gate: str, column: str) -> str:
    if gates.empty or gate not in gates.index or column not in gates.columns:
        return ""
    value = gates.loc[gate, column]
    return "" if pd.isna(value) else str(value)


def _md_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _md_cell(value: object) -> str:
    return _md_text(value).replace("|", "\\|").replace("\n", " ")


def _first_blocker(gates: pd.DataFrame, gate_names: list[str]) -> str:
    for gate in gate_names:
        blocker = _gate_value(gates, gate, "blocker")
        if blocker:
            return blocker
    return ""


def _first_next_action(gates: pd.DataFrame, gate_names: list[str]) -> str:
    for gate in gate_names:
        if not _gate_ready_from_index(gates, gate):
            return _gate_value(gates, gate, "next_action")
    return ""


def _capture_dashboard_metric(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "captures=0"
    ready = int(
        frame.get("research_spine_ready", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
    )
    completeness = pd.to_numeric(
        frame.get("capture_completeness_score", pd.Series(dtype=float)), errors="coerce"
    )
    if completeness.dropna().empty:
        best_row = frame.iloc[0]
    else:
        best_row = frame.loc[completeness.idxmax()]
    next_focus = str(best_row.get("next_capture_focus", "unknown") or "unknown")
    missing_value = best_row.get("missing_required_fields", "")
    missing = "" if pd.isna(missing_value) else str(missing_value or "")
    best_score = (
        "" if completeness.dropna().empty else f";best_completeness={float(completeness.max()):.2f}"
    )
    return f"captures={len(frame)};research_spine_ready={ready}{best_score};next_focus={next_focus};missing={missing or 'none'}"


def _capture_quality_dashboard_metric(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "quality_rows=0"
    research_usable = int(
        frame.get("research_usable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
    )
    execution_usable = int(
        frame.get("execution_usable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
    )
    blocked = frame[~frame.get("research_usable", pd.Series(dtype=bool)).fillna(False).astype(bool)]
    first_blocker = ""
    if not blocked.empty and "quality_blockers" in blocked.columns:
        first_blocker = str(blocked["quality_blockers"].fillna("").iloc[0])
    return (
        f"quality_rows={len(frame)};research_usable={research_usable};"
        f"execution_usable={execution_usable};first_quality_blocker={first_blocker or 'none'}"
    )


def _checklist_dashboard_metric(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "steps=0"
    ready = int(frame.get("ready", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    total = len(frame)
    blocked = frame[~frame.get("ready", pd.Series(dtype=bool)).fillna(False).astype(bool)]
    first_blocker = ""
    if not blocked.empty and "blocker" in blocked.columns:
        first_blocker = str(blocked["blocker"].fillna("").iloc[0])
    return f"steps_ready={ready}/{total};first_blocker={first_blocker or 'none'}"


def _checklist_first_blocked_next_action(frame: pd.DataFrame) -> str:
    if frame.empty or "ready" not in frame.columns or "next_action" not in frame.columns:
        return ""
    blocked = frame[~frame["ready"].fillna(False).astype(bool)]
    if blocked.empty:
        return ""
    return str(blocked["next_action"].fillna("").iloc[0])


def _checklist_step_ready(frame: pd.DataFrame, step: str) -> bool | None:
    if frame.empty or "step" not in frame.columns or "ready" not in frame.columns:
        return None
    matches = frame[frame["step"].astype(str) == step]
    if matches.empty:
        return None
    return bool(matches["ready"].fillna(False).astype(bool).iloc[0])


def _checklist_step_value(frame: pd.DataFrame, step: str, column: str) -> str:
    if frame.empty or "step" not in frame.columns or column not in frame.columns:
        return ""
    matches = frame[frame["step"].astype(str) == step]
    if matches.empty:
        return ""
    value = matches[column].fillna("").iloc[0]
    return "" if pd.isna(value) else str(value)


def _learning_dashboard_metric(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "events=0"
    combined = frame[frame.get("source", pd.Series(dtype=str)) == "combined"]
    row = combined.iloc[0] if not combined.empty else frame.iloc[-1]
    return (
        f"events={int(row.get('events', 0) or 0)};"
        f"outcomes={int(row.get('outcome_events', 0) or 0)};"
        f"outcomes_remaining={int(row.get('outcome_events_remaining', 0) or 0)};"
        f"ready_for_modeling={bool(row.get('ready_for_modeling', False))}"
    )


def _gap_severity(priority: str) -> str:
    return {
        "P1": "critical",
        "P2": "critical",
        "P3": "high",
        "P4": "high",
        "P5": "medium",
    }.get(priority, "medium")


def _required_gap_proof(area: str) -> str:
    proofs = {
        "crypto_wizards_capture": "pair-detail capture with spread,zscore,ecm_x,ecm_y,ecm_strength,price_x,price_y history",
        "crypto_wizards_historical_capture": "historical_capture_corpus_complete_with_lineage",
        "current_state_hyperliquid_wizard_exact_mode_intake": "same-day exact-mode settings for every current candidate",
        "strategy_acceptance": "production-eligible strategy with required two-leg base/stress results across multiple pairs",
        "dydx_testnet_readiness": "submit flag, credentials, SDK, indexer, and authenticated order adapter all ready",
        "paper_execution_gate": "strategy_acceptance ready and dydx_testnet_readiness ready",
        "learning_event_store": "paper journal or trade store contains outcome events for later modeling",
    }
    return proofs.get(area, "documented readiness evidence")


def _pre_mortem_question(priority: str, area: str, severity: str) -> str:
    if severity == "critical":
        return f"{priority} {area}: If we move to execution without this, what hard failure would most likely break us first?"
    if severity == "high":
        return f"{priority} {area}: What would fail and what signal would we watch first?"
    return f"{priority} {area}: What is the most likely downside risk we are accepting by skipping this?"


def _pre_mortem_failure_mode(area: str, gap: str) -> str:
    if area in {
        "crypto_wizards_capture",
        "crypto_wizards_historical_capture",
        "current_state_hyperliquid_wizard_exact_mode_intake",
    }:
        return (
            "Selection decisions may be based on incomplete spread/score history, producing false positives and "
            "pairs that are untradeable in practice."
        )
    if area == "strategy_acceptance":
        return "A strategy might appear good in dashboards but be rejected by production-style gates after deployment."
    if area == "dydx_testnet_readiness":
        if "submit" in str(gap or "").lower():
            return "Paper/live requests can fail at submit time and silently degrade into partial or dropped hedges."
        return "Venue adapter drift or config gaps can cause wrong orders or invalid venue assumptions."
    if area == "paper_execution_gate":
        return "Paper gating might pass with assumptions that do not hold in real or execution-tied backtests."
    if area == "learning_event_store":
        return "No realized outcomes means no feedback loop; model quality and execution bias drift undetected."
    return "A hidden evidence gap may only appear during live conditions and invalidate recent decisions."


def _pre_mortem_prevention(area: str, required_proof: str) -> str:
    preventions = {
        "crypto_wizards_capture": "Require capture completeness on spread/z-score/ECM metrics before any research promotion.",
        "crypto_wizards_historical_capture": "Keep historical capture coverage separate from current candidate readiness and promotion authority.",
        "current_state_hyperliquid_wizard_exact_mode_intake": "Require same-day exact-mode settings and timestamps for every current candidate; never inherit settings from historical captures.",
        "strategy_acceptance": "Keep strategy acceptance gates as mandatory before moving any pair into execution lanes.",
        "dydx_testnet_readiness": "Keep all venue execution checks and adapter wiring as hard blockers before paper/live signals.",
        "paper_execution_gate": "Run paper preflight as a strict step with explicit block reasons and no override path.",
        "learning_event_store": "Collect learning outcomes before model score updates and prevent model retraining on missing labels.",
    }
    return preventions.get(area, f"Use required proof: {required_proof}")


def _post_mortem_status_trajectory(area: str, current_status: str, previous_status: str) -> str:
    if current_status == "pass" and previous_status == "gap":
        return "resolved"
    if current_status == "gap" and previous_status == "pass":
        return "regressed"
    if current_status == "gap" and previous_status == "gap":
        return "persistent"
    if current_status == "gap" and previous_status == "unknown":
        return "new"
    return "unchanged" if current_status == previous_status else "unknown"


def _post_mortem_incident(area: str, gap: str) -> str:
    if area in {
        "crypto_wizards_capture",
        "crypto_wizards_historical_capture",
        "current_state_hyperliquid_wizard_exact_mode_intake",
    }:
        return (
            "We likely selected a candidate on stale or incomplete pair-structure data and entered with misleading "
            "spread/z-score/ECM signal quality."
        )
    if area == "strategy_acceptance":
        return "A strategy likely performed in simulation but failed production-grade constraints not met in earlier runs."
    if area == "dydx_testnet_readiness":
        if "submit" in str(gap or "").lower():
            return "Order submission path likely failed or dropped in paper/live paths."
        return "Venue or adapter readiness assumptions likely diverged from current execution environment."
    if area == "paper_execution_gate":
        return "Paper validation became out of sync with actual execution readiness and generated false-positive acceptance."
    if area == "learning_event_store":
        return "No realized outcomes blocked learning feedback; weak or unsafe settings were not corrected in time."
    return "A non-blocked gate likely failed to transfer into reliable real-world outcomes."


def _post_mortem_insight(area: str, trajectory: str, evidence: str) -> str:
    if trajectory in {"resolved", "new"}:
        return (
            f"{area} is actionable to investigate now; evidence is available (`{evidence}`), "
            "so a concrete regression cause can be logged."
        )
    if trajectory == "regressed":
        return (
            f"{area} was previously passing and regressed; this indicates a recent process or dependency change. "
            "Prioritize root-cause containment and replay controls."
        )
    if trajectory == "persistent":
        return f"{area} remained open in previous checks and continues to pose operational risk until evidence is completed."
    return "No recent trajectory comparison data was available; treat this as first-observed post-run evidence."


def _post_mortem_prevention(area: str, required_proof: str) -> str:
    return _pre_mortem_prevention(area, required_proof)


def _join_missing(items: list[tuple[str, bool]]) -> str:
    return ";".join(name for name, missing in items if missing)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return pd.DataFrame()


def _sum_bool_column(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns or frame.empty:
        return 0
    return int(frame[column].fillna(False).map(_coerce_bool).sum())


def _required_cost_buckets_from_acceptance(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "required_cost_buckets" not in frame.columns:
        return set()
    buckets: set[str] = set()
    for value in frame["required_cost_buckets"].dropna().astype(str):
        buckets.update(item for item in value.split(";") if item)
    return buckets


def _cost_buckets_from_results(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "cost_bucket" not in frame.columns:
        return set()
    return {str(value) for value in frame["cost_bucket"].dropna().unique()}


def _required_two_leg_inputs_from_acceptance(frame: pd.DataFrame) -> str:
    if frame.empty or "required_two_leg_inputs" not in frame.columns:
        return "price_x;price_y;hedge_ratio;beta;funding_x;funding_y"
    inputs: set[str] = set()
    for value in frame["required_two_leg_inputs"].dropna().astype(str):
        inputs.update(item for item in value.split(";") if item)
    return ";".join(sorted(inputs)) if inputs else "unknown"


def _funding_requirements_for_preflight(funding_missing: bool, output_path: Path) -> pd.DataFrame:
    if not funding_missing:
        return pd.DataFrame()
    try:
        return funding_requirements_report(output_path=output_path)
    except SystemExit:
        return _read_csv_or_empty(output_path)


def _funding_preflight_status(
    coverage: pd.DataFrame,
    funding_missing: bool,
    coverage_path: Path,
    requirements: pd.DataFrame | None = None,
    requirements_path: Path | None = None,
) -> dict[str, object]:
    if not funding_missing:
        return {
            "ready": True,
            "blocker": "",
            "evidence": "funding_inputs_not_current_blocker",
            "next_action": "funding inputs already represented in acceptance evidence",
        }
    requirements = requirements if requirements is not None else pd.DataFrame()
    required_markets = _semicolon_values(requirements.get("required_markets", pd.Series(dtype=str)))
    fetch_market_arg = ",".join(required_markets) if required_markets else "unknown"
    requirements_note = (
        f";requirements_exists={bool(requirements_path and requirements_path.exists())};"
        f"required_markets={';'.join(required_markets) if required_markets else 'unknown'}"
    )
    if coverage.empty:
        return {
            "ready": False,
            "blocker": "missing_funding_coverage_report",
            "evidence": f"coverage_exists={coverage_path.exists()};ready_pairs=0;blocked_pairs=unknown{requirements_note}",
            "next_action": (
                "fetch/export dYdX funding for "
                f"{fetch_market_arg}, then run funding-coverage with the dYdX funding CSV"
            ),
        }
    if "ready" not in coverage.columns:
        return {
            "ready": False,
            "blocker": "invalid_funding_coverage_report",
            "evidence": f"coverage_exists=True;columns={';'.join(coverage.columns)}",
            "next_action": "regenerate funding coverage with python -m quant_platform.cli funding-coverage",
        }
    ready_values = coverage["ready"].fillna(False).astype(bool)
    ready_pairs = int(ready_values.sum())
    total_pairs = len(coverage)
    blocked = coverage[~ready_values]
    blocked_pairs = ";".join(
        str(pair) for pair in blocked.get("pair", pd.Series(dtype=str)).dropna().unique()
    )
    missing = ";".join(
        str(value) for value in blocked.get("missing", pd.Series(dtype=str)).dropna().unique()
    )
    missing_markets = _semicolon_values(blocked.get("missing_markets", pd.Series(dtype=str)))
    missing_market_arg = ",".join(missing_markets)
    all_ready = total_pairs > 0 and ready_pairs == total_pairs
    return {
        "ready": all_ready,
        "blocker": "" if all_ready else "incomplete_funding_coverage",
        "evidence": (
            f"coverage_exists=True;pairs={total_pairs};ready_pairs={ready_pairs};"
            f"blocked_pairs={blocked_pairs or 'none'};missing={missing or 'none'};"
            f"missing_markets={';'.join(missing_markets) if missing_markets else 'none'}"
        ),
        "next_action": "rerun experiments with --funding-path"
        if all_ready
        else (
            f"fetch/export dYdX funding for {missing_market_arg}, rerun funding-coverage, then rerun experiments"
            if missing_markets
            else "add missing dYdX funding markets, rerun funding-coverage, then rerun experiments"
        ),
    }


def _two_leg_execution_input_blocker(frame: pd.DataFrame) -> str:
    missing = _missing_two_leg_inputs_from_acceptance(frame)
    if not missing:
        return "missing_hedge_beta_or_funding_inputs"
    if missing.issubset({"funding_x", "funding_y"}):
        return "missing_funding_inputs"
    if missing.issubset({"beta", "funding_x", "funding_y"}):
        return "missing_beta_or_funding_inputs"
    return "missing_hedge_beta_or_funding_inputs"


def _missing_two_leg_inputs_from_acceptance(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "acceptance_reason" not in frame.columns:
        return set()
    missing: set[str] = set()
    for value in frame["acceptance_reason"].dropna().astype(str):
        for match in re.finditer(r"\[([^\]]+)\]", value):
            missing.update(item for item in match.group(1).split("+") if item)
    return missing


def _acceptance_blocker_counts(frame: pd.DataFrame) -> list[tuple[str, int]]:
    if frame.empty or "acceptance_reason" not in frame.columns:
        return []
    counts: dict[str, int] = {}
    for value in frame["acceptance_reason"].dropna().astype(str):
        if value == "passed":
            continue
        for item in value.split(";"):
            if not item:
                continue
            counts[item] = counts.get(item, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _reason_counts(values: pd.Series) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for value in values.dropna().astype(str):
        for item in value.split(";"):
            if item and item != "passed":
                counts[item] = counts.get(item, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _median_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.dropna().empty:
        return 0.0
    return float(values.median())


def _max_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.dropna().empty:
        return 0.0
    return float(values.max())


def _median_cost_drag(frame: pd.DataFrame) -> float:
    if frame.empty or "gross_return" not in frame.columns or "total_return" not in frame.columns:
        return 0.0
    gross = pd.to_numeric(frame["gross_return"], errors="coerce").fillna(0.0)
    net = pd.to_numeric(frame["total_return"], errors="coerce").fillna(0.0)
    return float((gross - net).median())


def _strategy_failure_diagnosis(
    *,
    evaluated_runs: int,
    eligible_runs: int,
    total_trades: int,
    max_trades: int,
    median_profit_factor: float,
    median_sharpe: float,
    median_expectancy: float,
    worst_drawdown: float,
    missing_columns: list[str],
    acceptance_reason: str,
) -> str:
    if eligible_runs > 0:
        return "has_eligible_runs_but_strategy_acceptance_still_failed"
    if evaluated_runs == 0 and missing_columns:
        return "missing_required_feature_columns"
    if evaluated_runs == 0:
        return "no_evaluated_runs"
    if total_trades == 0 or max_trades == 0:
        return "no_trade_generation"
    if "passing_pairs<" in acceptance_reason:
        if max_trades < 10:
            return "too_few_trades_and_no_passing_pairs"
        return "no_passing_pairs"
    if median_expectancy <= 0:
        return "negative_or_zero_expectancy"
    if median_profit_factor < 1.8:
        return "profit_factor_below_gate"
    if median_sharpe < 1.2:
        return "sharpe_below_gate"
    if worst_drawdown > 0.15:
        return "drawdown_above_gate"
    return "acceptance_failed_unknown"


def _strategy_failure_next_action(diagnosis: str) -> str:
    return {
        "missing_required_feature_columns": "collect Crypto Wizards fields needed by this strategy or keep it data-blocked",
        "no_evaluated_runs": "inspect strategy required columns and signal function coverage",
        "no_trade_generation": "do not deploy; research whether thresholds are too strict before changing them",
        "too_few_trades_and_no_passing_pairs": "collect longer histories and test threshold families without relaxing production gates",
        "no_passing_pairs": "reject for now; search for pairs/regimes where the strategy passes both base and stress costs",
        "negative_or_zero_expectancy": "reject for now; costs and signal direction consume the edge",
        "profit_factor_below_gate": "reject for now; analyze feature ablations before changing sizing",
        "sharpe_below_gate": "reject for now; volatility-adjusted returns are not robust",
        "drawdown_above_gate": "reject for now; tighten risk filter or avoid the regime",
        "has_eligible_runs_but_strategy_acceptance_still_failed": "inspect pair coverage and required cost bucket coverage",
    }.get(diagnosis, "inspect acceptance_report and experiment_results")


def _history_multiplier(max_trades: int, target_trades: int) -> str:
    if max_trades <= 0:
        return "unknown_no_trades"
    multiplier = int(np.ceil(target_trades / max_trades))
    return f"{multiplier}x_current_history"


def _quality_blocker_summary(frame: pd.DataFrame) -> str:
    if frame.empty or "quality_blockers" not in frame.columns:
        return ""
    counts: dict[str, int] = {}
    for value in frame["quality_blockers"].dropna().astype(str):
        for item in value.split(";"):
            item = item.strip()
            if item:
                counts[item] = counts.get(item, 0) + 1
    return ";".join(
        f"{blocker}:{count}"
        for blocker, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    )


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _numeric_cell(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _tested_market_pairs() -> set[frozenset[str]]:
    reports = ROOT / "reports"
    pairs: set[frozenset[str]] = set()
    for path in (reports / "funding_coverage.csv", reports / "funding_requirements.csv"):
        frame = _read_csv_or_empty(path)
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            left = _md_text(row.get("market_x", ""))
            right = _md_text(row.get("market_y", ""))
            if left and right:
                pairs.add(frozenset({_normalize_dydx_market(left), _normalize_dydx_market(right)}))
    results = _read_csv_or_empty(reports / "experiment_results.csv")
    if not results.empty and "pair" in results.columns:
        for value in results["pair"].dropna().astype(str).unique():
            parsed = _markets_from_pair_name(value)
            if parsed:
                pairs.add(frozenset(parsed))
    return pairs


def _fetched_market_pair_info() -> dict[frozenset[str], dict[str, str]]:
    reports = ROOT / "reports"
    frame = _read_csv_or_empty(reports / "pair_detail_quality_report.csv")
    info: dict[frozenset[str], dict[str, str]] = {}
    if frame.empty or "pair" not in frame.columns:
        return info
    parsed_candidates: dict[frozenset[str], dict[str, object]] = {}
    for _, row in frame.iterrows():
        parsed = _markets_from_pair_name(_md_text(row.get("pair", "")))
        if not parsed:
            continue
        pair_key = frozenset(parsed)
        research_usable = _coerce_bool(row.get("research_usable", False))
        execution_usable = _coerce_bool(row.get("execution_usable", False))
        history_rows = int(_numeric_cell(row.get("history_rows", 0.0), 0.0))
        blockers = _md_text(row.get("quality_blockers", ""))
        status = "research_usable" if research_usable else "quality_blocked"
        candidate = {
            "quality_status": status,
            "quality_blockers": blockers,
            "research_usable": research_usable,
            "execution_usable": execution_usable,
            "history_rows": history_rows,
        }
        existing = parsed_candidates.get(pair_key)
        if existing is None:
            parsed_candidates[pair_key] = candidate
            continue

        existing_exec = bool(existing.get("execution_usable"))
        new_exec = execution_usable
        existing_research = bool(existing.get("research_usable"))
        new_research = research_usable
        existing_rows = int(existing.get("history_rows") or 0)
        # Prefer execution-ready rows, then research-ready, then more rows.
        if existing_exec and (not new_exec):
            continue
        if (not existing_exec) and new_exec:
            parsed_candidates[pair_key] = candidate
            continue
        if existing_research and (not new_research):
            continue
        if (not existing_research) and new_research:
            parsed_candidates[pair_key] = candidate
            continue
        if history_rows < existing_rows:
            continue
        parsed_candidates[pair_key] = candidate

    for pair_key, candidate in parsed_candidates.items():
        info[pair_key] = {
            "quality_status": str(candidate.get("quality_status", "")),
            "quality_blockers": str(candidate.get("quality_blockers", "")),
        }
    return info


def _stale_market_risk_info() -> dict[str, str]:
    reports = ROOT / "reports"
    frame = _read_csv_or_empty(reports / "pair_detail_quality_report.csv")
    risks: dict[str, str] = {}
    if frame.empty or "pair" not in frame.columns:
        return risks
    selected: dict[frozenset[str], dict[str, object]] = {}
    for _, row in frame.iterrows():
        parsed = _markets_from_pair_name(_md_text(row.get("pair", "")))
        if not parsed:
            continue
        pair_key = frozenset(parsed)
        history_rows = int(_numeric_cell(row.get("history_rows", 0.0), 0.0))
        execution_usable = _coerce_bool(row.get("execution_usable", False))
        research_usable = _coerce_bool(row.get("research_usable", False))
        existing = selected.get(pair_key)
        if existing is not None:
            existing_exec = bool(existing.get("execution_usable"))
            existing_research = bool(existing.get("research_usable"))
            existing_rows = int(existing.get("history_rows") or 0)
            if existing_exec and not execution_usable:
                continue
            if not existing_exec and execution_usable:
                selected[pair_key] = {
                    "row": row,
                    "execution_usable": execution_usable,
                    "research_usable": research_usable,
                    "history_rows": history_rows,
                }
                continue
            if existing_research and not research_usable:
                continue
            if not existing_research and research_usable:
                selected[pair_key] = {
                    "row": row,
                    "execution_usable": execution_usable,
                    "research_usable": research_usable,
                    "history_rows": history_rows,
                }
                continue
            if history_rows < existing_rows:
                continue
        selected[pair_key] = {
            "row": row,
            "execution_usable": execution_usable,
            "research_usable": research_usable,
            "history_rows": history_rows,
        }

    for selected_entry in selected.values():
        row = selected_entry.get("row")
        if row is None or not hasattr(row, "get"):
            continue
        parsed = _markets_from_pair_name(_md_text(row.get("pair", "")))
        if not parsed:
            continue
        left, right = parsed
        blockers = _md_text(row.get("quality_blockers", ""))
        stale_x = _numeric_cell(row.get("stale_price_x_rate", 0.0), 0.0)
        stale_y = _numeric_cell(row.get("stale_price_y_rate", 0.0), 0.0)
        if "price_x_stale_above_90pct" in blockers or stale_x > 0.90:
            risks[left] = f"{left}:stale_price_x"
        if "price_y_stale_above_90pct" in blockers or stale_y > 0.90:
            risks[right] = f"{right}:stale_price_y"
    return risks


def _covered_funding_markets() -> set[str]:
    reports = ROOT / "reports"
    markets: set[str] = set()
    coverage = _read_csv_or_empty(reports / "funding_coverage.csv")
    for column in ("market_x", "market_y"):
        if column in coverage.columns:
            markets.update(
                _normalize_dydx_market(value)
                for value in coverage[column].dropna().astype(str)
                if value
            )
    return markets


def _markets_from_pair_name(pair: str) -> tuple[str, str] | None:
    markets = pair_markets_from_pair(pair)
    if len(markets) >= 2:
        return markets[0], markets[1]
    return None


def _normalize_dydx_market(asset: str) -> str:
    return normalize_dydx_market(asset)


def _normalize_dydx_pair(pair: str) -> str:
    text = str(pair).upper().replace("/", "-").strip()
    parsed = _markets_from_pair_name(text)
    if parsed:
        return f"{parsed[0]}-{parsed[1]}"
    parts = [part for part in re.split(r"[-_/]", text) if part]
    if len(parts) == 2 and parts[1] == "USD":
        return f"{parts[0]}-USD"
    if len(parts) == 2:
        return "-".join(parts)
    return text


def _pair_id_from_markets(left: str, right: str) -> str:
    return f"{left.replace('-USD', '').lower()}_{right.replace('-USD', '').lower()}"


def _trade_sample_note(unblock: pd.DataFrame) -> str:
    if unblock.empty or "area" not in unblock.columns:
        return "collect enough 5-minute history to satisfy 100/250-trade acceptance gates"
    sample = unblock[unblock["area"].astype(str) == "trade_sample_size"]
    if sample.empty:
        return "collect enough 5-minute history to satisfy 100/250-trade acceptance gates"
    row = sample.iloc[0]
    minimum = _md_text(row.get("minimum_history_multiplier_estimate", ""))
    preferred = _md_text(row.get("preferred_history_multiplier_estimate", ""))
    evidence = _md_text(row.get("evidence", ""))
    return f"{evidence};minimum={minimum or 'unknown'};preferred={preferred or 'unknown'}"


def _local_cached_dydx_markets() -> set[str]:
    manual_dir = ROOT / "data" / "raw" / "dydx_manual"
    if not manual_dir.exists():
        return set()
    markets: set[str] = set()
    for path in manual_dir.glob("*_5MINS_candles.json"):
        name = path.name.replace("_5MINS_candles.json", "")
        if name:
            markets.add(_normalize_dydx_market(name))
    return markets


def _fetch_live_dydx_market_catalog(
    indexer_base: str = DEFAULT_INDEXER_BASE, max_markets: int = 300
) -> dict[str, dict[str, object]]:
    requested_indexer_base = _indexer_base_with_scheme(indexer_base, "")
    url = f"{requested_indexer_base}/v4/perpetualMarkets?limit={max_markets}"
    response = requests.get(url, headers={"Content-Type": "application/json"}, timeout=20.0)
    response.raise_for_status()
    payload = response.json()
    markets = payload.get("markets", {})
    return markets if isinstance(markets, dict) else {}


def _live_market_selector_score(trades_24h: float, volume_24h: float) -> float:
    # Favor markets that are both frequently traded and meaningfully liquid.
    return round(
        (np.log10(max(trades_24h, 0.0) + 1.0) * 0.6) + (np.log10(max(volume_24h, 0.0) + 1.0) * 0.4),
        6,
    )


def dydx_live_market_selector_report(
    *,
    max_pairs: int = 10,
    indexer_base: str = DEFAULT_INDEXER_BASE,
    output_path: Path | None = None,
) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "dydx_live_market_selector.csv"
    markets = _fetch_live_dydx_market_catalog(indexer_base=indexer_base)
    tested_pairs = _tested_market_pairs()
    fetched_pairs = _fetched_market_pair_info()
    risky_markets = _stale_market_risk_info()
    cached_markets = _local_cached_dydx_markets()

    candidates: list[dict[str, object]] = []
    for raw_market, meta in markets.items():
        market = _normalize_dydx_market(raw_market)
        if market in DEFAULT_DYDX_LIVE_SELECTOR_EXCLUDED_MARKETS:
            continue
        if market in cached_markets:
            continue
        if "," in raw_market:
            continue
        if str(meta.get("status", "")).upper() != "ACTIVE":
            continue
        if str(meta.get("marketType", "")).upper() != "CROSS":
            continue
        try:
            trades_24h = float(meta.get("trades24H") or 0.0)
            volume_24h = float(meta.get("volume24H") or 0.0)
            oracle_price = float(meta.get("oraclePrice") or 0.0)
        except (TypeError, ValueError):
            continue
        if trades_24h <= 0 or volume_24h <= 0 or oracle_price <= 0:
            continue
        for anchor in DEFAULT_DYDX_LIVE_SELECTOR_ANCHORS:
            if market == anchor:
                continue
            pair_key = frozenset({anchor, market})
            if pair_key in tested_pairs:
                continue
            if pair_key in fetched_pairs:
                continue
            if market in risky_markets:
                continue
            candidates.append(
                {
                    "anchor_market": anchor,
                    "candidate_market": market,
                    "pair_id": _pair_id_from_markets(anchor, market),
                    "pair_key": pair_key,
                    "candidate_trades_24h": trades_24h,
                    "candidate_volume_24h": volume_24h,
                    "candidate_oracle_price": oracle_price,
                    "selector_score": _live_market_selector_score(trades_24h, volume_24h),
                    "indexer_base": indexer_base,
                }
            )

    rows = sorted(
        candidates,
        key=lambda row: (
            -float(row["selector_score"]),
            -float(row["candidate_trades_24h"]),
            -float(row["candidate_volume_24h"]),
            str(row["anchor_market"]),
            str(row["candidate_market"]),
        ),
    )[: max(max_pairs, 1)]

    sample_note = _trade_sample_note(_read_csv_or_empty(reports / "research_unblock_plan.csv"))
    plan_rows: list[dict[str, object]] = []
    for rank, row in enumerate(rows, start=1):
        anchor = str(row["anchor_market"])
        market = str(row["candidate_market"])
        pair_id = str(row["pair_id"])
        plan_rows.append(
            {
                "rank": rank,
                "pair_id": pair_id,
                "asset_x": anchor,
                "asset_y": market,
                "selector_score": row["selector_score"],
                "candidate_trades_24h": row["candidate_trades_24h"],
                "candidate_volume_24h": row["candidate_volume_24h"],
                "candidate_oracle_price": row["candidate_oracle_price"],
                "fetch_command": (
                    "PYTHONPATH=src python3 -m quant_platform.cli fetch-dydx-two-leg-data "
                    f"--asset-x {anchor} --asset-y {market} --pair-id {pair_id} --limit 1000 "
                    f"--indexer-base {indexer_base} --derive-hedge-ratio"
                ),
                "request_template_command": (
                    "PYTHONPATH=src python3 -m quant_platform.cli dydx-two-leg-request-template "
                    f"--asset-x {anchor} --asset-y {market} --pair-id {pair_id} --limit 1000 "
                    f"--indexer-base {indexer_base}"
                ),
                "sample_size_note": sample_note,
                "notes": "live selector candidate: not locally cached, not already tested, active CROSS market, positive 24h volume/trades",
            }
        )

    frame = pd.DataFrame(plan_rows)
    _write_csv_atomic(frame, output)
    return frame


def dydx_anchor_sweep_report(
    *,
    anchors: list[str] | None = None,
    max_pairs: int = 10,
    indexer_base: str = DEFAULT_INDEXER_BASE,
    output_path: Path | None = None,
) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "dydx_anchor_sweep.csv"
    anchor_markets = [
        _normalize_dydx_market(anchor)
        for anchor in (anchors or list(DEFAULT_DYDX_LIVE_SELECTOR_ANCHORS))
        if str(anchor).strip()
    ]
    if not anchor_markets:
        raise SystemExit("dydx-anchor-sweep requires at least one anchor market")

    selector = dydx_live_market_selector_report(
        max_pairs=max(max_pairs * len(anchor_markets) * 8, 50),
        indexer_base=indexer_base,
        output_path=reports / "dydx_live_market_selector.csv",
    )
    if selector.empty:
        _write_csv_atomic(selector, output)
        return selector

    filtered = selector[selector["asset_x"].isin(anchor_markets)].copy()
    if filtered.empty:
        _write_csv_atomic(filtered, output)
        return filtered

    filtered["_selector_score"] = pd.to_numeric(filtered["selector_score"], errors="coerce").fillna(
        0.0
    )
    filtered["_candidate_trades_24h"] = pd.to_numeric(
        filtered["candidate_trades_24h"], errors="coerce"
    ).fillna(0.0)
    filtered["_candidate_volume_24h"] = pd.to_numeric(
        filtered["candidate_volume_24h"], errors="coerce"
    ).fillna(0.0)
    filtered = filtered.sort_values(
        ["asset_x", "_selector_score", "_candidate_trades_24h", "_candidate_volume_24h", "asset_y"],
        ascending=[True, False, False, False, True],
    )
    filtered["anchor_rank"] = filtered.groupby("asset_x").cumcount() + 1
    filtered = filtered[filtered["anchor_rank"] <= max(max_pairs, 1)].copy()
    filtered = filtered.sort_values(
        ["_selector_score", "_candidate_trades_24h", "_candidate_volume_24h", "asset_x", "asset_y"],
        ascending=[False, False, False, True, True],
    )
    filtered["overall_rank"] = range(1, len(filtered) + 1)
    filtered.insert(0, "anchor", filtered["asset_x"])
    frame = filtered[
        [
            "overall_rank",
            "anchor_rank",
            "anchor",
            "pair_id",
            "asset_x",
            "asset_y",
            "selector_score",
            "candidate_trades_24h",
            "candidate_volume_24h",
            "candidate_oracle_price",
            "fetch_command",
            "request_template_command",
            "sample_size_note",
            "notes",
        ]
    ].reset_index(drop=True)
    _write_csv_atomic(frame, output)
    return frame


def dydx_live_market_counts_report(
    *,
    indexer_base: str = DEFAULT_INDEXER_BASE,
    output_path: Path | None = None,
) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "dydx_live_market_counts.csv"

    markets = _fetch_live_dydx_market_catalog(indexer_base=indexer_base)
    normalized_markets = {
        _normalize_dydx_market(name): meta for name, meta in markets.items() if "," not in str(name)
    }
    total_markets = len(normalized_markets)
    active_markets = {
        market: meta
        for market, meta in normalized_markets.items()
        if str(meta.get("status", "")).upper() == "ACTIVE"
    }
    active_cross_markets = {
        market: meta
        for market, meta in active_markets.items()
        if str(meta.get("marketType", "")).upper() == "CROSS"
    }
    excluded_markets = {
        market
        for market in active_cross_markets
        if market in DEFAULT_DYDX_LIVE_SELECTOR_EXCLUDED_MARKETS
    }
    cached_markets = _local_cached_dydx_markets()
    risky_markets = _stale_market_risk_info()
    tested_pairs = _tested_market_pairs()
    fetched_pairs = _fetched_market_pair_info()

    candidate_markets: set[str] = set()
    candidate_pairs = 0
    for market, meta in active_cross_markets.items():
        if market in excluded_markets or market in cached_markets or market in risky_markets:
            continue
        try:
            trades_24h = float(meta.get("trades24H") or 0.0)
            volume_24h = float(meta.get("volume24H") or 0.0)
            oracle_price = float(meta.get("oraclePrice") or 0.0)
        except (TypeError, ValueError):
            continue
        if trades_24h <= 0 or volume_24h <= 0 or oracle_price <= 0:
            continue
        candidate_markets.add(market)
        for anchor in DEFAULT_DYDX_LIVE_SELECTOR_ANCHORS:
            if market == anchor:
                continue
            pair_key = frozenset({anchor, market})
            if pair_key in tested_pairs or pair_key in fetched_pairs:
                continue
            candidate_pairs += 1

    frame = pd.DataFrame(
        [
            {
                "indexer_base": indexer_base,
                "total_markets": total_markets,
                "active_markets": len(active_markets),
                "active_cross_markets": len(active_cross_markets),
                "excluded_markets": len(excluded_markets),
                "cached_markets": len(cached_markets),
                "risky_markets": len(risky_markets),
                "anchor_markets": len(DEFAULT_DYDX_LIVE_SELECTOR_ANCHORS),
                "untested_candidate_markets": len(candidate_markets),
                "untested_candidate_pairs": candidate_pairs,
                "tested_pairs": len(tested_pairs),
                "fetched_pairs": len(fetched_pairs),
            }
        ]
    )
    _write_csv_atomic(frame, output)
    return frame


def print_dydx_live_market_counts(
    *,
    indexer_base: str = DEFAULT_INDEXER_BASE,
    output_path: Path | None = None,
) -> None:
    output = output_path or ROOT / "reports" / "dydx_live_market_counts.csv"
    frame = dydx_live_market_counts_report(indexer_base=indexer_base, output_path=output)
    print(frame.to_string(index=False))
    print(f"dydx_live_market_counts: {output}")


def print_dydx_live_market_selector(
    *,
    max_pairs: int = 10,
    indexer_base: str = DEFAULT_INDEXER_BASE,
    output_path: Path | None = None,
) -> None:
    output = output_path or ROOT / "reports" / "dydx_live_market_selector.csv"
    frame = dydx_live_market_selector_report(
        max_pairs=max_pairs,
        indexer_base=indexer_base,
        output_path=output,
    )
    print(frame.to_string(index=False))
    print(f"dydx_live_market_selector: {output}")


def print_dydx_anchor_sweep(
    *,
    anchors: list[str] | None = None,
    max_pairs: int = 10,
    indexer_base: str = DEFAULT_INDEXER_BASE,
    output_path: Path | None = None,
) -> None:
    output = output_path or ROOT / "reports" / "dydx_anchor_sweep.csv"
    frame = dydx_anchor_sweep_report(
        anchors=anchors,
        max_pairs=max_pairs,
        indexer_base=indexer_base,
        output_path=output,
    )
    print(frame.to_string(index=False))
    print(f"dydx_anchor_sweep: {output}")


def _resolve_long_history_pair(
    *,
    pair: str | None,
    asset_x: str | None,
    asset_y: str | None,
    pair_id: str | None,
) -> tuple[str, str, str]:
    if pair or (asset_x and asset_y):
        left, right = _resolve_two_leg_assets(pair=pair, asset_x=asset_x, asset_y=asset_y)
        return left, right, pair_id or _pair_id_from_markets(left, right)
    plan = dydx_pair_expansion_plan_report(max_pairs=1)
    if not plan.empty:
        tested = (
            plan["already_tested"].map(_coerce_bool)
            if "already_tested" in plan.columns
            else pd.Series(False, index=plan.index)
        )
        fetched = (
            plan["already_fetched"].map(_coerce_bool)
            if "already_fetched" in plan.columns
            else pd.Series(False, index=plan.index)
        )
        fresh = plan[(~tested) & (~fetched)].copy()
        if "rank" in fresh.columns:
            fresh["_rank"] = pd.to_numeric(fresh["rank"], errors="coerce")
            fresh = fresh.sort_values("_rank")
        if not fresh.empty:
            row = fresh.iloc[0]
            left = _md_text(row.get("asset_x", ""))
            right = _md_text(row.get("asset_y", ""))
            return (
                left,
                right,
                pair_id or _md_text(row.get("pair_id", "")) or _pair_id_from_markets(left, right),
            )
    raise SystemExit(
        "dydx-long-history-plan requires --pair or --asset-x/--asset-y when no fresh expansion pair exists"
    )


def _parse_iso_datetime(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_iso_z(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolution_timedelta(resolution: str) -> timedelta:
    text = str(resolution).upper()
    if text.endswith("MINS"):
        return timedelta(minutes=int(text.replace("MINS", "")))
    if text.endswith("MIN"):
        return timedelta(minutes=int(text.replace("MIN", "")))
    if text.endswith("HOUR") or text.endswith("HOURS"):
        return timedelta(hours=int(text.replace("HOURS", "").replace("HOUR", "")))
    if text.endswith("DAY") or text.endswith("DAYS"):
        return timedelta(days=int(text.replace("DAYS", "").replace("DAY", "")))
    raise SystemExit(f"unsupported dYdX candle resolution for long-history plan: {resolution}")


def _threshold_sweep_diagnosis(row: pd.Series) -> str:
    passing_pairs = int(row.get("passing_pairs", 0) or 0)
    max_trades = float(row.get("max_trades", 0) or 0)
    median_pf = float(row.get("median_profit_factor", 0) or 0)
    median_sharpe = float(row.get("median_sharpe", 0) or 0)
    worst_drawdown = float(row.get("worst_drawdown", 0) or 0)
    if passing_pairs > 0:
        return "candidate_threshold_has_passing_pairs"
    if max_trades < 100:
        return "threshold_still_trade_sparse"
    if median_pf < 1.8:
        return "more_trades_but_profit_factor_fails"
    if median_sharpe < 1.2:
        return "more_trades_but_sharpe_fails"
    if worst_drawdown > 0.15:
        return "more_trades_but_drawdown_fails"
    return "threshold_not_production_ready"


def _priority_sort_key(priority: object) -> int:
    text = str(priority).strip().upper()
    if text.startswith("P"):
        try:
            return int(text[1:])
        except ValueError:
            return 999
    return 999


def _gate_dependency(gate: str) -> str:
    dependencies = {
        "pair_detail_history": "crypto_wizards_live_artifacts",
        "pair_detail_two_leg_execution_history": "pair_detail_history",
        "pair_detail_capture_audit": "crypto_wizards_live_artifacts",
        "strategy_acceptance": "pair_detail_two_leg_execution_history",
        "dydx_testnet_readiness": "strategy_acceptance",
        "paper_execution_gate": "strategy_acceptance;dydx_testnet_readiness",
        "learning_event_store": "paper_execution_gate",
    }
    return dependencies.get(gate, "")


def _gate_sort_key(gate: object) -> int:
    order = {
        "crypto_wizards_live_artifacts": 10,
        "pair_detail_capture_audit": 20,
        "pair_detail_history": 30,
        "pair_detail_two_leg_execution_history": 40,
        "strategy_acceptance": 50,
        "dydx_testnet_readiness": 60,
        "paper_execution_gate": 70,
        "learning_event_store": 80,
    }
    return order.get(str(gate), 999)


def _max_int_column(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return int(values.max()) if not values.empty else 0


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(pd.read_csv(path))
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return 0


def _jsonl_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _assert_ready_for_exploration() -> None:
    if _read_bool_env("QPA_ALLOW_BLOCKED_EXPLORATION", False):
        return
    readiness = priority_readiness_report()
    gates = readiness.set_index("gate") if not readiness.empty else pd.DataFrame()
    blocked: list[str] = []
    if not _gate_ready_from_index(gates, "strategy_acceptance"):
        blocked.append("strategy_acceptance")
    if not _gate_ready_from_index(gates, "dydx_testnet_readiness"):
        blocked.append("dydx_testnet_readiness")
    if not _gate_ready_from_index(gates, "learning_event_store"):
        blocked.append("learning_event_store")
    if blocked:
        raise SystemExit(f"exploration_blocked:{','.join(blocked)}")


def build_paper_plan_from_cli(
    pair: str,
    strategy_id: int,
    signal: float,
    hedge_ratio: float,
    beta: float,
    notional_usd: float,
    acceptance_path: Path | None = None,
    venue: str = "dydx",
) -> tuple[SpreadOrderPlan, list[dict[str, object]]]:
    venue = (venue or "").lower()
    if acceptance_path is None:
        acceptance = _augmented_acceptance_frame(ROOT / "reports")
    else:
        if not acceptance_path.exists():
            raise SystemExit(f"acceptance report not found: {acceptance_path}")
        acceptance = pd.read_csv(acceptance_path)
    signal_row = {
        "pair": pair,
        "strategy_id": strategy_id,
        "signal": signal,
        "hedge_ratio": hedge_ratio,
        "beta": beta,
    }
    if venue == "dydx":
        config = DydxNetworkConfig.paper_testnet_from_env()
        indexer = build_dydx_indexer_adapter(config)
        if indexer is not None:
            left, right = _split_pair_assets(pair)
            left_market = left if left.endswith("-USD") else f"{left}-USD"
            right_market = right if right.endswith("-USD") else f"{right}-USD"
            left_payload = indexer.market_data(left_market).get("payload", {})
            right_payload = indexer.market_data(right_market).get("payload", {})
            left_meta = left_payload.get("markets", {}).get(left_market, {})
            right_meta = right_payload.get("markets", {}).get(right_market, {})
            signal_row["price_x"] = float(left_meta.get("oraclePrice") or 0.0)
            signal_row["price_y"] = float(right_meta.get("oraclePrice") or 0.0)
            signal_row["step_size_x"] = float(left_meta.get("stepSize") or 0.0)
            signal_row["step_size_y"] = float(right_meta.get("stepSize") or 0.0)
    plan = build_research_gated_paper_plan(
        signal_row,
        acceptance,
        notional_usd=notional_usd,
        venue=venue,
    )
    return plan, [_intent_row(intent) for intent in plan.intents]


def _venue_asset_key(value: str) -> str:
    text = str(value or "").replace("/", "-").upper().strip()
    text = text.removesuffix("-USD")
    text = text.removesuffix("_USD")
    return text


def _build_paper_venue_options(
    pair: str,
    *,
    root: Path | None = None,
) -> list[dict[str, object]]:
    effective_root = root or ROOT
    parts = _split_pair_assets(pair)
    if len(parts) != 2:
        return []
    x, y = parts

    context = _read_csv_or_empty(
        effective_root / "data" / "processed" / "market_venue_context.csv"
    )
    if not context.empty:
        context["asset_key"] = context["asset"].map(_venue_asset_key)
        context["venue"] = context["venue"].astype(str)

    context_by_asset: dict[str, dict[str, pd.Series]] = {}
    if not context.empty and {"asset_key", "venue"}.issubset(context.columns):
        for (asset_key, venue), row in context.groupby(["asset_key", "venue"]):
            context_by_asset.setdefault(asset_key, {})[str(venue).lower()] = row.iloc[0]

    universe = _read_csv_or_empty(
        effective_root / "data" / "processed" / "pair_universe.csv"
    )
    row = _lookup_pair_universe_row(universe, pair)
    preferred = ""
    preferreds: list[str] = []
    if not row.empty:
        preferred = str(row.iloc[0].get("best_execution_venue", "") or "").strip().lower()
        available = str(row.iloc[0].get("available_venues", "") or "")
        preferreds = [venue.strip().lower() for venue in available.split(";") if venue.strip()]

    recommendations = _read_csv_or_empty(
        effective_root / "reports" / "active" / "venue_route_recommendations.csv"
    )
    route_row = _lookup_pair_universe_row(recommendations, pair)
    if not route_row.empty:
        route_preferreds = []
        for column in (
            "recommended_paper_venue",
            "recommended_execution_venue",
            "recommended_research_venue",
        ):
            venue = str(route_row.iloc[0].get(column, "") or "").strip().lower()
            if venue and venue not in {"nan", "none", "null"}:
                route_preferreds.append(venue)
        if route_preferreds:
            preferred = route_preferreds[0]
            preferreds = list(dict.fromkeys([*route_preferreds, *preferreds]))

    left_map = context_by_asset.get(x, {})
    right_map = context_by_asset.get(y, {})
    context_venues = set(left_map) | set(right_map)
    venues = sorted(context_venues | set(preferreds))
    if not venues:
        return []

    options: list[dict[str, object]] = []
    for venue in venues:
        normalized_venue = venue.lower()
        left = left_map.get(normalized_venue)
        right = right_map.get(normalized_venue)
        blockers: list[str] = []

        left_exists = isinstance(left, pd.Series)
        right_exists = isinstance(right, pd.Series)
        if not left_exists or not right_exists:
            if not left_exists:
                blockers.append(f"missing_{x}_venue_data")
            if not right_exists:
                blockers.append(f"missing_{y}_venue_data")
            options.append(
                {
                    "venue": normalized_venue,
                    "executable": False,
                    "execution_ready": False,
                    "research_ready": left_exists and right_exists,
                    "blockers": ";".join(sorted(blockers)),
                    "preference": "preferred" if venue == preferred else "candidate",
                    "venue_lanes": ";".join(
                        sorted(
                            {
                                str(item)
                                for item in [
                                    left.get("venue_lane") if left_exists else None,
                                    right.get("venue_lane") if right_exists else None,
                                ]
                                if item
                            }
                        )
                    ),
                }
            )
            continue

        left_authority = bool(_coerce_bool(left.get("execution_authority", False)))
        right_authority = bool(_coerce_bool(right.get("execution_authority", False)))
        left_tradable = bool(_coerce_bool(left.get("tradable", False)))
        right_tradable = bool(_coerce_bool(right.get("tradable", False)))
        leg_blockers = [
            value
            for value in [
                str(left.get("blocker", "")).strip(),
                str(right.get("blocker", "")).strip(),
            ]
            if value and value.lower() not in {"nan", "none"}
        ]
        if not left_authority or not right_authority:
            blockers.append("execution_not_authorized")
        if not left_tradable or not right_tradable:
            blockers.append("thin_venue_liquidity")
        blockers.extend(leg_blockers)

        venue_context_ready = (
            left_authority
            and right_authority
            and left_tradable
            and right_tradable
            and not leg_blockers
        )
        generic_execution_supported = (
            venue_has_paper_adapter(normalized_venue) or normalized_venue == "dydx"
        )
        if normalized_venue == "hyperliquid":
            adapter_contract = validate_venue_order_client_adapter(normalized_venue)
            generic_execution_supported = bool(
                adapter_contract.get("valid")
                and adapter_contract.get("exchange_submission_capable")
                and adapter_contract.get("pair_submission_capable")
            )
            if venue_context_ready and not generic_execution_supported:
                blockers.append("hyperliquid_pair_submission_not_supported")
        execution_ready = venue_context_ready and generic_execution_supported
        if (
            venue_context_ready
            and not generic_execution_supported
            and normalized_venue != "hyperliquid"
        ):
            blockers.append("paper_execution_not_implemented")
        options.append(
            {
                "venue": normalized_venue,
                "executable": venue_context_ready and generic_execution_supported,
                "execution_ready": execution_ready,
                "research_ready": left_tradable and right_tradable,
                "blockers": ";".join(sorted(set(blockers))),
                "preference": "preferred" if venue == preferred else "candidate",
                "venue_lanes": ";".join(
                    sorted(
                        {
                            str(item)
                            for item in [
                                str(left.get("venue_lane", "")).strip(),
                                str(right.get("venue_lane", "")).strip(),
                            ]
                            if item
                        }
                    )
                ),
            }
        )

    if not options:
        return []

    def _score(row: dict[str, object]) -> tuple[int, int, str]:
        executable = 2 if bool(row["executable"]) else 0
        preferred_weight = 1 if str(row["preference"]) == "preferred" else 0
        venue = str(row["venue"])
        in_preferred = 1 if venue in preferreds else 0
        return (executable, preferred_weight + in_preferred, venue)

    return sorted(options, key=_score, reverse=True)


def _format_paper_venue_options(options: list[dict[str, object]]) -> str:
    if not options:
        return "none"
    rendered: list[str] = []
    for row in options:
        venue = str(row.get("venue", ""))
        executable = bool(row.get("executable", False))
        reason = str(row.get("blockers", "")).strip() or "ready"
        preference = str(row.get("preference", "candidate"))
        rendered.append(f"{venue}(pref={preference},executable={executable},reason={reason})")
    return "; ".join(rendered)


def _lookup_pair_universe_row(universe: pd.DataFrame, pair: str) -> pd.DataFrame:
    if universe.empty or "pair" not in universe.columns:
        return pd.DataFrame()
    normalized = pair.replace("/", "-").upper().strip()
    rows = universe[universe["pair"].astype(str).str.upper() == normalized]
    if rows.empty:
        normalized_alt = normalized.replace("-", "/")
        rows = universe[universe["pair"].astype(str).str.upper() == normalized_alt]
    return rows


def _split_pair_assets(pair: str) -> list[str]:
    if not pair:
        return []
    normalized = pair.replace("/", "-").upper().strip()
    parsed = _markets_from_pair_name(normalized)
    if parsed:
        return [_venue_asset_key(parsed[0]), _venue_asset_key(parsed[1])]
    left, sep, right = normalized.partition("-")
    if not sep or not left or not right:
        return []
    return [_venue_asset_key(left), _venue_asset_key(right)]


def run_paper_plan(
    pair: str,
    strategy_id: int,
    signal: float,
    hedge_ratio: float,
    beta: float,
    notional_usd: float,
    acceptance_path: Path | None = None,
    journal_path: Path | None = None,
    venue: str | None = None,
    order_approval_id: str | None = None,
) -> None:
    effective_journal_path = journal_path or ROOT / "reports" / "paper_trading_journal.csv"
    watch_output_path = (
        None
        if journal_path is None
        else effective_journal_path.with_name("current_paper_watch_positions.csv")
    )
    venue = (venue or "").lower().strip() or "auto"
    selected_venue = _resolve_paper_venue(pair, venue)
    venue_options = _build_paper_venue_options(pair)
    print(f"paper_plan_requested_venue: {venue}")
    if venue_options:
        print(f"paper_plan_venue_options: {_format_paper_venue_options(venue_options)}")
    plan, intent_rows = build_paper_plan_from_cli(
        pair=pair,
        strategy_id=strategy_id,
        signal=signal,
        hedge_ratio=hedge_ratio,
        beta=beta,
        notional_usd=notional_usd,
        acceptance_path=acceptance_path,
        venue=selected_venue,
    )
    selected_venue = str(plan.venue or selected_venue).lower()
    print(f"paper_plan_status: {plan.status}")
    print(f"paper_plan_reason: {plan.reason}")
    print(f"paper_plan_venue: {selected_venue}")
    if intent_rows:
        print(pd.DataFrame(intent_rows).to_string(index=False))
    dashboard_snapshot = _lookup_dashboard_trade_snapshot(pair=pair)
    if plan.status != "paper_ready":
        path = append_paper_trading_record(
            paper_trading_record(plan, dashboard_snapshot=dashboard_snapshot),
            effective_journal_path,
        )
        watch_path = refresh_current_paper_watch_positions(
            effective_journal_path,
            output_path=watch_output_path,
        )
        print(f"paper_trading_journal: {path}")
        print(f"current_paper_watch_positions: {watch_path}")
        return

    blockers: list[str] = []
    config = DydxNetworkConfig.paper_testnet_from_env()
    order_client: object | None = None
    if selected_venue == "dydx":
        config = DydxNetworkConfig.paper_testnet_from_env()
        blockers = config.paper_trading_blockers()
        order_client, order_adapter_error = _load_dydx_order_client_adapter()
        adapter_contract = validate_dydx_order_client_adapter()
        if order_adapter_error:
            blockers.append("invalid_dydx_order_client_adapter")
        elif (
            adapter_contract["configured"]
            and adapter_contract["valid"]
            and not adapter_contract["exchange_submission_capable"]
        ):
            blockers.append("record_only_dydx_order_client_adapter")
        if order_client is None and "missing_dydx_v4_client" not in blockers:
            blockers.append("missing_dydx_order_client_adapter")
    elif selected_venue == "hyperliquid":
        config = replace(
            HyperliquidTestnetConfig.paper_testnet_from_env(),
            order_approval_id=str(order_approval_id or "").strip() or None,
        )
        preflight = hyperliquid_testnet_order_preflight_status()
        blockers.extend(item for item in str(preflight.get("blocker") or "").split(";") if item)
        order_client, order_adapter_error = _load_venue_order_client_adapter(selected_venue)
        adapter_contract = validate_venue_order_client_adapter(selected_venue)
        if order_adapter_error:
            blockers.append(f"invalid_hyperliquid_order_client_adapter:{order_adapter_error}")
        elif not bool(adapter_contract.get("pair_submission_capable")):
            blockers.append("hyperliquid_pair_submission_not_supported")
        elif not bool(adapter_contract.get("exchange_submission_capable")):
            blockers.append("record_only_hyperliquid_order_client_adapter")
        if not config.order_approval_id:
            blockers.append("missing_explicit_hyperliquid_order_approval")
    else:
        order_client, order_adapter_error = _load_venue_order_client_adapter(selected_venue)
        adapter_contract = validate_venue_order_client_adapter(selected_venue)
        if order_adapter_error:
            blockers.append(f"invalid_{selected_venue}_order_client_adapter")
        else:
            if not adapter_contract["configured"]:
                blockers.append(f"missing_{selected_venue}_order_client_adapter")
            elif not adapter_contract["valid"]:
                blockers.append(
                    f"{selected_venue}_order_client_adapter_invalid:{adapter_contract.get('error')}"
                )
            elif not adapter_contract["exchange_submission_capable"]:
                blockers.append(f"record_only_{selected_venue}_order_client_adapter")
    if order_client is None and not blockers:
        blockers.append(f"paper_execution_not_implemented_for_{selected_venue}")

    if blockers:
        print(f"execution_blockers: {','.join(blockers)}")
        blocked_plan = block_paper_plan_for_execution_config(plan, blockers)
        path = append_paper_trading_record(
            paper_trading_record(
                blocked_plan, blockers=blockers, dashboard_snapshot=dashboard_snapshot
            ),
            effective_journal_path,
        )
        watch_path = refresh_current_paper_watch_positions(
            effective_journal_path,
            output_path=watch_output_path,
        )
        print(f"paper_trading_journal: {path}")
        print(f"current_paper_watch_positions: {watch_path}")
        return
    market_data_client = build_dydx_indexer_adapter(config) if selected_venue == "dydx" else None
    execution = build_execution_venue(
        selected_venue,
        config=config,
        order_client=order_client,
        market_data_client=market_data_client,
    )
    fills = submit_paper_plan(plan, execution)
    if fills:
        print(pd.DataFrame([fill.__dict__ for fill in fills]).to_string(index=False))
    path = append_paper_trading_record(
        paper_trading_record(
            plan, fills=fills, blockers=blockers, dashboard_snapshot=dashboard_snapshot
        ),
        effective_journal_path,
    )
    watch_path = refresh_current_paper_watch_positions(
        effective_journal_path,
        output_path=watch_output_path,
    )
    print(f"paper_trading_journal: {path}")
    print(f"current_paper_watch_positions: {watch_path}")


def print_current_paper_watch(
    journal_path: Path | None = None, output_path: Path | None = None
) -> None:
    source = journal_path or ROOT / "reports" / "paper_trading_journal.csv"
    output = refresh_current_paper_watch_positions(source, output_path=output_path)
    frame = _read_csv_or_empty(output)
    print(frame.to_string(index=False))
    print(f"current_paper_watch_positions: {output}")


def run_close_paper_trade(
    *,
    realized_return: float,
    trade_id: str | None = None,
    pair: str | None = None,
    strategy_id: int | None = None,
    journal_path: Path | None = None,
    output_path: Path | None = None,
) -> None:
    source = journal_path or ROOT / "reports" / "paper_trading_journal.csv"
    path, record = append_paper_outcome_record(
        source,
        trade_id=trade_id,
        pair=pair,
        strategy_id=strategy_id,
        realized_return=realized_return,
    )
    watch_path = refresh_current_paper_watch_positions(source, output_path=output_path)
    print(pd.DataFrame([asdict(record)]).to_string(index=False))
    print(f"paper_trading_journal: {path}")
    print(f"current_paper_watch_positions: {watch_path}")


def _intent_row(intent: OrderIntent) -> dict[str, object]:
    return {
        "market": intent.market,
        "side": intent.side,
        "size": intent.size,
        "limit_price": intent.limit_price,
        "reduce_only": intent.reduce_only,
    }


def _cli_endpoint_specs(endpoint_specs: list[str] | None):
    if not endpoint_specs:
        return None
    return parse_endpoint_specs(",".join(endpoint_specs))


def refresh_augmented_research(root: Path = ROOT) -> CommandResult:
    registry = build_research_source_registry(root=root)
    youtube_extraction = extract_youtube_research(root=root)
    udemy_extraction = extract_udemy_research(root=root)
    ccxt_extraction = extract_research_knowledge(root=root)
    knowledge = build_research_knowledge_store(root=root)
    audit = research_source_audit(root=root)
    summary = research_knowledge_summary(root=root)
    return CommandResult(
        paths={
            **registry.paths,
            **youtube_extraction.paths,
            **udemy_extraction.paths,
            **ccxt_extraction.paths,
            **knowledge.paths,
            **audit.paths,
            **summary.paths,
        },
        summary={
            "sources": int(registry.summary.get("sources", 0)),
            "extracted_rows": (
                int(youtube_extraction.summary.get("rows", 0))
                + int(udemy_extraction.summary.get("knowledge_rows", 0))
                + int(ccxt_extraction.summary.get("rows", 0))
            ),
            "knowledge_rows": int(knowledge.summary.get("rows", 0)),
            "knowledge_tables": int(summary.summary.get("tables", 0)),
        },
    )


def _ou_v4_manual_capture_blockers(
    *, root: Path, now: datetime | None = None
) -> tuple[list[str], Path | None]:
    """Require the governed frozen manifest before manual OU-v4 execution."""

    status_path = root / "reports/active/corrective_wizard_next_capture_manifest.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return ["wizard_capture_manifest_status_missing_or_invalid"], None

    blockers: list[str] = []
    if status.get("schema_version") != "thewiz.corrective_wizard_next_capture_manifest.v1":
        blockers.append("wizard_capture_manifest_status_schema_invalid")
    if status.get("status") != "PASS" or status.get("blockers") not in ([], None):
        blockers.append("wizard_capture_manifest_status_not_passed")

    manifest_id = str(status.get("manifest_id", "")).strip()
    relative = str(status.get("immutable_manifest_path", "")).strip()
    relative_path = Path(relative)
    manifest_path: Path | None = None
    if (
        not manifest_id
        or not relative
        or relative_path.is_absolute()
        or ".." in relative_path.parts
    ):
        blockers.append("wizard_capture_manifest_identity_or_path_invalid")
    else:
        candidate = (root / relative_path).resolve()
        expected_root = (root / "data/research/wizard_capture_manifests").resolve()
        try:
            candidate.relative_to(expected_root)
        except ValueError:
            blockers.append("wizard_capture_manifest_path_outside_immutable_root")
        else:
            manifest_path = candidate

    manifest: dict[str, object] = {}
    if manifest_path is None or not manifest_path.is_file():
        blockers.append("wizard_capture_manifest_immutable_receipt_missing")
    else:
        supplied_hash = str(status.get("immutable_manifest_sha256", "")).strip()
        observed_hash = sha256(manifest_path.read_bytes()).hexdigest()
        if supplied_hash != observed_hash:
            blockers.append("wizard_capture_manifest_immutable_hash_mismatch")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            blockers.append("wizard_capture_manifest_immutable_receipt_invalid")

    calls = manifest.get("calls") if isinstance(manifest, dict) else None
    ou_calls = (
        [row for row in calls if isinstance(row, dict) and row.get("lane") == "ou_v4_holdout"]
        if isinstance(calls, list)
        else []
    )
    if (
        manifest.get("manifest_id") != manifest_id
        or not isinstance(calls, list)
        or len(calls) != 8
        or len(ou_calls) != 8
        or int(manifest.get("pending_calls", -1) or -1) != 8
        or int(manifest.get("planned_credits", -1) or -1) != 16
        or sum(int(row.get("credit_cost", 0) or 0) for row in ou_calls) != 16
    ):
        blockers.append("wizard_capture_manifest_not_exact_frozen_ou_v4_cohort")
    semantic_binding = validate_ou_v4_capture_manifest_contract(
        root=root,
        manifest=manifest if isinstance(manifest, dict) else {},
    )
    if semantic_binding.get("status") != "PASS":
        blockers.extend(str(value) for value in semantic_binding.get("blockers", []))

    eligible_at = pd.to_datetime(
        status.get("next_external_attempt_eligible_at"), utc=True, errors="coerce"
    )
    timestamp = pd.Timestamp(now or datetime.now(UTC))
    if pd.isna(eligible_at):
        blockers.append("wizard_capture_manifest_eligibility_time_invalid")
    elif timestamp < eligible_at:
        blockers.append("wizard_capture_manifest_not_yet_eligible")
    return list(dict.fromkeys(blockers)), manifest_path


def _ou_v5_manual_capture_blockers(
    *, root: Path, now: datetime | None = None
) -> tuple[list[str], Path | None]:
    """Require the governed frozen manifest before manual OU-v5 execution."""

    status_path = root / "reports/active/corrective_wizard_next_capture_manifest.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return ["wizard_capture_manifest_status_missing_or_invalid"], None

    blockers: list[str] = []
    if status.get("schema_version") != "thewiz.corrective_wizard_next_capture_manifest.v1":
        blockers.append("wizard_capture_manifest_status_schema_invalid")
    if status.get("status") != "PASS" or status.get("blockers") not in ([], None):
        blockers.append("wizard_capture_manifest_status_not_passed")

    manifest_id = str(status.get("manifest_id", "")).strip()
    relative = str(status.get("immutable_manifest_path", "")).strip()
    relative_path = Path(relative)
    manifest_path: Path | None = None
    if (
        not manifest_id
        or not relative
        or relative_path.is_absolute()
        or ".." in relative_path.parts
    ):
        blockers.append("wizard_capture_manifest_identity_or_path_invalid")
    else:
        candidate = (root / relative_path).resolve()
        expected_root = (root / "data/research/wizard_capture_manifests").resolve()
        try:
            candidate.relative_to(expected_root)
        except ValueError:
            blockers.append("wizard_capture_manifest_path_outside_immutable_root")
        else:
            manifest_path = candidate

    manifest: dict[str, object] = {}
    if manifest_path is None or not manifest_path.is_file():
        blockers.append("wizard_capture_manifest_immutable_receipt_missing")
    else:
        supplied_hash = str(status.get("immutable_manifest_sha256", "")).strip()
        observed_hash = sha256(manifest_path.read_bytes()).hexdigest()
        if supplied_hash != observed_hash:
            blockers.append("wizard_capture_manifest_immutable_hash_mismatch")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            blockers.append("wizard_capture_manifest_immutable_receipt_invalid")

    calls = manifest.get("calls") if isinstance(manifest, dict) else None
    ou_calls = (
        [row for row in calls if isinstance(row, dict) and row.get("lane") == "ou_v5_holdout"]
        if isinstance(calls, list)
        else []
    )
    if (
        manifest.get("manifest_id") != manifest_id
        or not isinstance(calls, list)
        or len(calls) != 8
        or len(ou_calls) != 8
        or int(manifest.get("pending_calls", -1) or -1) != 8
        or int(manifest.get("planned_credits", -1) or -1) != 16
        or sum(int(row.get("credit_cost", 0) or 0) for row in ou_calls) != 16
    ):
        blockers.append("wizard_capture_manifest_not_exact_frozen_ou_v5_cohort")
    semantic_binding = validate_ou_v5_capture_manifest_contract(
        root=root,
        manifest=manifest if isinstance(manifest, dict) else {},
    )
    if semantic_binding.get("status") != "PASS":
        blockers.extend(str(value) for value in semantic_binding.get("blockers", []))

    eligible_at = pd.to_datetime(
        status.get("next_external_attempt_eligible_at"), utc=True, errors="coerce"
    )
    timestamp = pd.Timestamp(now or datetime.now(UTC))
    if pd.isna(eligible_at):
        blockers.append("wizard_capture_manifest_eligibility_time_invalid")
    elif timestamp < eligible_at:
        blockers.append("wizard_capture_manifest_not_yet_eligible")
    return list(dict.fromkeys(blockers)), manifest_path


def _blocked_direct_wizard_external_execution(
    *,
    command: str,
    blockers: list[str] | None = None,
    evidence_path: Path | None = None,
) -> CommandResult:
    """Keep every chargeable Wizard call inside the shared proof scheduler."""

    all_blockers = list(blockers or [])
    all_blockers.append("shared_credit_proof_scheduler_required")
    paths = {"capture_manifest": evidence_path} if evidence_path is not None else {}
    return CommandResult(
        paths=paths,
        summary={
            "status": "BLOCKED_SHARED_CREDIT_ORCHESTRATION",
            "command": command,
            "blockers": list(dict.fromkeys(all_blockers)),
            "calls_made": 0,
            "responses_captured": 0,
            "credits_attempted": 0,
            "external_requests": 0,
            "research_only": True,
            "candidate_promotion_authority": False,
            "order_submission_included": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    load_environment: bool = True,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "system-check",
            "build-corrective-governance",
            "build-corrective-data-evidence",
            "build-corrective-wizard-parity",
            "review-wizard-dynamic-v2",
            "build-wizard-dynamic-v2-supreme-review",
            "build-wizard-ou-v4-supreme-review",
            "build-wizard-ou-v5-supreme-review",
            "build-wizard-ou-v6-supreme-review",
            "close-wizard-ou-v6-terminal",
            "build-wizard-ou-v4-failure-attribution",
            "build-wizard-ou-v5-failure-attribution",
            "evaluate-wizard-ou-v2-holdout",
            "register-wizard-ou-trend-selector-v1",
            "build-wizard-credit-budget",
            "build-wizard-next-capture-manifest",
            "build-wizard-reset-readiness",
            "phase00-maintenance-start",
            "phase00-checkpoint",
            "phase00-lineage-seal",
            "phase00-descendant-invalidate",
            "phase00-closure-verify",
            "phase00-maintenance-resume",
            "build-canonical-runtime-contract",
            "build-scheduler-runtime-readiness",
            "build-wizard-comparator-review-control",
            "build-stage4-handoff-readiness",
            "reconcile-wizard-capture-manifest",
            "run-wizard-ou-v3-holdout",
            "review-wizard-ou-v3",
            "register-wizard-ou-v4-holdout",
            "run-wizard-ou-v4-holdout",
            "register-wizard-ou-v5-holdout",
            "run-wizard-ou-v5-holdout",
            "register-wizard-ou-v6-holdout",
            "review-wizard-ou-v4",
            "review-wizard-ou-v5",
            "review-wizard-ou-v6",
            "register-wizard-copula-v2",
            "run-wizard-copula-proof",
            "build-corrective-statistical-remediation",
            "build-corrective-daily-cadence",
            "run-corrective-l2-capture",
            "install-corrective-l2-cadence",
            "build-corrective-agent-governance",
            "register-stage5-protocol",
            "build-corrective-release-gates",
            "capture-live-input-parity-evidence",
            "build-live-canary-executor-preflight",
            "run-live-canary-executor",
            "complete-corrective-plan",
            "build-canonical-program-status",
            "capture-wizard-api-credit-receipt",
            "corrective-artifact-retention",
            "build-artifact-index",
            "current-state",
            "build-wizard-research-journal",
            "build-pair-universe",
            "build-trade-dataset",
            "promote-trade-dataset",
            "train-trade-gate",
            "run-model-gated-backtest",
            "export-trade-gate-model",
            "build-command-dashboard",
            "build-v2-math-diagnostic",
            "reevaluate-current-wizard-hyperliquid-math",
            "build-math-v2-acceptance",
            "build-v2-preflight-run",
            "validate-v2-run",
            "publish-v2-run-status",
            "apify-cost-audit",
            "paper-candidate-shortlist",
            "focused-paper-validation",
            "run-langgraph-agent-workflow",
            "run-orchestrator",
            "interactive-mixtape-solution",
            "build-mini-agent-orchestration",
            "build-orchestrator-assistant",
            "build-specialist-scoreboard",
            "run-rl-research",
            "run-rl-idea-scout",
            "run-brain-cycle",
            "brain-readiness-report",
            "run-magicka-learning",
            "run-sequential-thinking-magicka",
            "train-rl-ppo",
            "export-rl-policy",
            "ingest-research-source",
            "ingest-paper-library",
            "verify-paper-sources",
            "build-paper-adversarial-reviews",
            "run-paper-critical-checks",
            "run-paper-reproduction-suite",
            "build-v1-v1-1-comparison",
            "build-research-source-registry",
            "research-source-audit",
            "extract-research-knowledge",
            "build-research-knowledge-store",
            "research-knowledge-summary",
            "refresh-augmented-research",
            "refresh-udemy-research",
            "refresh-youtube-collection",
            "build-youtube-caption-insights",
            "build-youtube-brain",
            "build-youtube-hypotheses",
            "refresh-youtube-outcomes",
            "build-youtube-brain-dashboard",
            "run-youtube-brain",
            "run-hudson-thames-youtube-research",
            "run-youtube-hypothesis-validation",
            "run-base-rl",
            "evaluate-base-rl",
            "base-rl-paper-handoff",
            "refresh-base-rl-feedback",
            "run-augmented-rl",
            "compare-base-vs-augmented-rl",
            "promotion-readiness-report",
            "archive-from-index",
            "build-wizard-evidence",
            "build-wizard-hypotheses",
            "build-wizard-diagnostic-confirmation",
            "build-wizard-discovery-triage",
            "build-wizard-local-parity",
            "build-wizard-pair-detail-capture-queue",
            "build-wizard-mode-matrix-capture-queue",
            "build-wizard-pair-settings-capture-template",
            "import-wizard-pair-settings-capture",
            "build-wizard-replay-handoff",
            "build-wizard-mode-replay-capability",
            "build-wizard-mode-comparison",
            "build-wizard-exploratory-cost-sensitivity",
            "build-wizard-exact-mode-capture-queue",
            "build-wizard-research-pack",
            "ingest-exhaustive-wizard-dashboard-captures",
            "build-exhaustive-wizard-hyperliquid-run",
            "build-exhaustive-wizard-api-refresh-delta",
            "run-wizard-pair-detail-api-pilot",
            "build-current-wizard-hyperliquid-handoff",
            "materialize-current-wizard-hyperliquid-history",
            "run-current-wizard-hyperliquid-canonical-replay",
            "materialize-current-wizard-hyperliquid-cost-evidence",
            "run-current-wizard-hyperliquid-observed-cost-replay",
            "run-current-wizard-hyperliquid-walkforward",
            "build-current-wizard-hyperliquid-regime-attribution",
            "run-current-wizard-hyperliquid-robustness",
            "build-current-wizard-hyperliquid-concentration",
            "build-current-wizard-hyperliquid-failure-attribution",
            "build-current-wizard-hyperliquid-failure-routing-index",
            "build-current-wizard-hyperliquid-leverage-surface",
            "build-current-wizard-hyperliquid-learning-ledger",
            "validate-current-wizard-hyperliquid-chain",
            "build-current-wizard-hyperliquid-operating-cadence",
            "run-current-wizard-hyperliquid-daily-pipeline",
            "build-current-wizard-hyperliquid-completion-audit",
            "build-current-wizard-hyperliquid-evidence-command-center",
            "build-current-wizard-hyperliquid-storage-reclamation-plan",
            "stage-current-wizard-hyperliquid-archive-copy",
            "plan-current-wizard-hyperliquid-archive-release",
            "validate-current-wizard-hyperliquid-testnet-protocol",
            "build-current-wizard-ou-optimal-overlay",
            "build-exhaustive-wizard-hyperliquid-mapping-refresh",
            "build-exhaustive-wizard-hyperliquid-replay-preflight",
            "materialize-exhaustive-wizard-hyperliquid-history",
            "run-exhaustive-wizard-hyperliquid-canonical-replay",
            "materialize-exhaustive-hyperliquid-funding-evidence",
            "build-exhaustive-wizard-hyperliquid-cost-evidence",
            "run-exhaustive-wizard-hyperliquid-observed-cost-replay",
            "run-exhaustive-wizard-hyperliquid-walkforward",
            "build-exhaustive-wizard-hyperliquid-regime-attribution",
            "run-exhaustive-wizard-hyperliquid-robustness",
            "build-exhaustive-wizard-hyperliquid-concentration",
            "build-exhaustive-wizard-hyperliquid-leverage-surface",
            "build-exhaustive-wizard-hyperliquid-learning-ledger",
            "run-exhaustive-wizard-hyperliquid-validation",
            "ingest-wizard-pair-detail-ui-bundles",
            "build-market-venue-context",
            "build-venue-lane-test-plan",
            "build-multi-venue-history-readiness",
            "build-venue-route-scorecard",
            "refresh-hyperliquid-market-context",
            "refresh-hyperliquid-execution-cost-snapshot",
            "refresh-hyperliquid-funding-history",
            "fetch-hyperliquid-candles",
            "build-hyperliquid-pair-history",
            "build-hyperliquid-research-bundle",
            "build-hyperliquid-wizard-hypothesis-queue",
            "build-exhaustive-wizard-mode-proof-queue",
            "run-hyperliquid-wizard-mode-proofs",
            "build-hyperliquid-pair-cost-model",
            "build-hyperliquid-evidence-cadence",
            "hyperliquid-lane-readiness",
            "hyperliquid-testnet-market-inventory",
            "hyperliquid-testnet-preflight",
            "hyperliquid-testnet-margin-snapshot",
            "hyperliquid-testnet-collateral-transfer-preflight",
            "run-hyperliquid-testnet-collateral-transfer",
            "hyperliquid-testnet-pair-execution-preflight",
            "run-hyperliquid-testnet-pair-execution",
            "hyperliquid-testnet-smoke-approval-template",
            "build-hyperliquid-testnet-lifecycle-gate",
            "capture-hyperliquid-testnet-lifecycle-evidence",
            "hyperliquid-testnet-recover-pair-state",
            "hyperliquid-testnet-sign-smoke-approval",
            "hyperliquid-execution-compatibility",
            "hyperliquid-testnet-candidate-shortlist",
            "run-hyperliquid-research-cycle",
            "run-hyperliquid-auxiliary-timeframe-validation",
            "gmx-testnet-market-inventory",
            "gmx-execution-compatibility",
            "gmx-testnet-candidate-shortlist",
            "fetch-binance-spot-candles",
            "build-binance-spot-pair-history",
            "binance-spot-history-readiness",
            "binance-testnet-preflight",
            "binance-testnet-adapter-contract",
            "binance-testnet-pair-preflight",
            "fetch-yahoo-crypto-candles",
            "build-yahoo-crypto-pair-history",
            "refresh-yahoo-crypto-pair-history",
            "yahoo-crypto-history-readiness",
            "refresh-yahoo-research-candidates",
            "backfill-yahoo-crypto-funding",
            "verify-wizard-local-mode",
            "build-wizard-local-verification-batch",
            "build-dictionaries",
            "ingest-fixtures",
            "ingest-crypto-wizards-scanner",
            "normalize-enrichment-fixtures",
            "materialize-p2-rerun-subset",
            "ingest-pair-details",
            "pair-detail-capture-checklist",
            "pair-detail-quality",
            "run-demo-backtest",
            "run-demo-experiments",
            "run-fixture-experiments",
            "run-pair-detail-experiments",
            "list-strategies",
            "list-crypto-wizards-endpoints",
            "check-live-config",
            "diagnose-crypto-wizards",
            "crypto-wizards-min5-request-template",
            "import-crypto-wizards-payload",
            "import-crypto-wizards-zscores",
            "import-crypto-wizards-backtest",
            "import-pair-detail-capture",
            "import-latest-pair-detail-download",
            "import-dydx-candles",
            "import-dydx-candle-bundle",
            "dydx-two-leg-request-template",
            "fetch-dydx-two-leg-data",
            "build-dydx-pair-history",
            "build-dydx-long-history-pair",
            "inspect-pair-detail-capture",
            "capture-preflight",
            "verify-crypto-wizards-live-artifacts",
            "crypto-wizards-live-coverage",
            "paper-readiness-checkpoint",
            "research-sweep",
            "check-dydx-config",
            "dydx-order-adapter-contract",
            "dydx-execution-checklist",
            "funding-requirements",
            "funding-template",
            "funding-template-check",
            "import-funding-template",
            "fetch-dydx-funding",
            "export-dydx-funding",
            "funding-coverage",
            "funded-research-spine",
            "refresh-apify-sources",
            "apify-source-summary",
            "strategy-acceptance-checklist",
            "strategy-failure-attribution",
            "research-unblock-plan",
            "zscore-threshold-sweep",
            "strategy-trade-count-gap",
            "dydx-pair-expansion-plan",
            "dydx-live-market-selector",
            "dydx-anchor-sweep",
            "dydx-live-market-counts",
            "dydx-local-pair-universe",
            "dydx-long-history-plan",
            "dydx-long-history-coverage",
            "fetch-dydx-long-history-windows",
            "run-dydx-long-history",
            "run-dydx-pair-expansion",
            "backfill-dydx-pair-history-features",
            "strategy-family-sweep",
            "strategy-family-matrix",
            "research-quantization",
            "strategy-family-sweep-failure-attribution",
            "priority-readiness",
            "priority-actions",
            "priority-dashboard",
            "priority-runbook",
            "paper-execution-preflight",
            "paper-venue-preflight",
            "gap-test",
            "gap-analysis-checklist",
            "pre-mortem-checklist",
            "post-mortem-checklist",
            "postmortem-checklist",
            "supreme-team",
            "supreme-team-checklist",
            "red-team-checklist",
            "redteam-checklist",
            "learning-report",
            "build-ml-trade-filter-dataset",
            "train-ml-trade-filter",
            "shadow-ml-trade-filter",
            "compare-ml-shadow-models",
            "trade-timing-template",
            "trade-timing-comparison-report",
            "learning-outcome-template",
            "seed-learning-outcome-template",
            "learning-outcome-template-check",
            "import-learning-outcomes",
            "append-learning-outcome",
            "paper-watch",
            "research-spine",
            "crawl-crypto-wizards",
            "crypto-wizards-full-sweep",
            "restore-wizard-sweep-from-raw",
            "wizard-control-plane",
            "crawl-crypto-wizards-min5",
            "crawl-crypto-wizards-min5-backtest",
            "paper-plan",
            "close-paper-trade",
        ],
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument(
        "--old-replay-manifest",
        type=Path,
        default=None,
        help="Frozen legacy canonical replay manifest used for math re-evaluation.",
    )
    parser.add_argument("--run-id", default=None, help="Immutable V2 run identifier.")
    parser.add_argument(
        "--execute-live-canary",
        action="store_true",
        help="Explicitly request the one-use live canary path; absent means status-only.",
    )
    parser.add_argument(
        "--recover-live-canary",
        action="store_true",
        help="Recover an already-reserved live canary without retrying its entry.",
    )
    parser.add_argument("--authorization-sha256", default="")
    parser.add_argument("--approval-id", default="")
    parser.add_argument("--live-canary-acknowledgement", default="")
    parser.add_argument(
        "--phase00-reason",
        default="approved_phase00_corrective_repair",
        help="Operator reason recorded in the immutable Phase 00 maintenance receipt.",
    )
    parser.add_argument("--phase00-maintenance-id", default="")
    parser.add_argument("--phase00-ttl-seconds", type=int, default=7200)
    parser.add_argument("--phase00-wait-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--execute-testnet-collateral-transfer",
        action="store_true",
        help=(
            "Explicitly request the one-use Testnet spot-to-perp collateral "
            "transfer; absent means status-only."
        ),
    )
    parser.add_argument("--transfer-preflight-id", default="")
    parser.add_argument("--transfer-approval-id", default="")
    parser.add_argument("--transfer-amount-usd", type=float, default=25.0)
    parser.add_argument("--testnet-collateral-transfer-acknowledgement", default="")
    parser.add_argument(
        "--testnet-pair-action",
        choices=("entry", "exit"),
        default="entry",
        help="Exact governed Hyperliquid Testnet pair action.",
    )
    parser.add_argument("--testnet-pair-preflight-id", default="")
    parser.add_argument(
        "--execute-hyperliquid-testnet-pair",
        action="store_true",
        help=(
            "Explicitly request one sealed Hyperliquid Testnet pair action; "
            "absent means status-only."
        ),
    )
    parser.add_argument("--testnet-pair-acknowledgement", default="")
    parser.add_argument("--queue-path", type=Path, default=None)
    parser.add_argument(
        "--candidate-path",
        type=Path,
        default=None,
        help="Explicit candidate CSV for bounded Hyperliquid funding, history, and L2 evidence jobs.",
    )
    parser.add_argument(
        "--wizard-source-path",
        type=Path,
        default=None,
        help="Wizard snapshot used to build the exhaustive no-prefilter research run.",
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        default=None,
        help="Crypto Wizards endpoint as name=/path or name=https://host/path. May be repeated.",
    )
    parser.add_argument("--pair", default=None)
    parser.add_argument("--pair-ids", default="")
    parser.add_argument("--source", default=None)
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=None,
        help="Directory of PDF papers to inventory and ingest as research-only evidence.",
    )
    parser.add_argument("--source-type", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--author", default="")
    parser.add_argument("--channel-or-publisher", default="")
    parser.add_argument("--topic-tags", default="")
    parser.add_argument("--status", default="active")
    parser.add_argument("--review-status", default="unreviewed")
    parser.add_argument("--quarantine-status", default="active")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--notes", default="")
    parser.add_argument("--source-id", default="")
    parser.add_argument("--stage", default="all")
    parser.add_argument("--strategy-id", type=int, default=None)
    parser.add_argument("--signal", type=float, default=None)
    parser.add_argument("--hedge-ratio", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--left-candles", type=Path, default=None)
    parser.add_argument("--right-candles", type=Path, default=None)
    parser.add_argument("--asset-x", default=None)
    parser.add_argument("--asset-y", default=None)
    parser.add_argument("--pair-id", default="1")
    parser.add_argument("--interval", default=None)
    parser.add_argument("--zscore-window", type=int, default=7)
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument(
        "--strategy-ids",
        default="",
        help="Comma-separated strategy IDs to run explicitly, overriding default all strategies.",
    )
    parser.add_argument(
        "--recommended-strategy-priority",
        action="store_true",
        help="Prioritize recommended strategies (from pair strategy_mode) by running those first.",
    )
    parser.add_argument("--to-iso", default=None)
    parser.add_argument("--similarity-k", type=int, default=6)
    parser.add_argument("--priority", default="Sharpe")
    parser.add_argument("--cw-strategy", default="Spread")
    parser.add_argument(
        "--wizard-exchanges",
        default="Binance,BinanceUs,ByBit,Coinbase,Dydx",
        help="Comma-separated Crypto Wizards crypto venues for crypto-wizards-full-sweep.",
    )
    parser.add_argument(
        "--wizard-intervals",
        default="Daily,Hourly",
        help="Comma-separated prescanned intervals for crypto-wizards-full-sweep.",
    )
    parser.add_argument(
        "--wizard-strategies",
        default="Spread,ZScoreRoll,Copula",
        help="Comma-separated prescanned strategy families for crypto-wizards-full-sweep.",
    )
    parser.add_argument(
        "--wizard-priorities",
        default="Sharpe",
        help="Comma-separated prescanned priorities for crypto-wizards-full-sweep.",
    )
    parser.add_argument("--wizard-daily-credit-limit", type=int, default=1000)
    parser.add_argument("--wizard-reserved-credits", type=int, default=100)
    parser.add_argument(
        "--wizard-credit-lane",
        default="exhaustive_discovery_sweep",
        choices=("exhaustive_discovery_sweep", "daily_discovery_refresh"),
        help=(
            "Immutable credit-reservation lane for a Wizard sweep. Use "
            "daily_discovery_refresh only for an explicitly authorized second "
            "Daily snapshot after the exhaustive lane has been reconciled."
        ),
    )
    parser.add_argument(
        "--wizard-pair-group-key",
        default="binance|daily|ETH|WIF",
        help="Exact pair_group_key for the bounded Wizard pair-detail API pilot.",
    )
    parser.add_argument(
        "--execute-wizard-detail-pilot",
        action="store_true",
        help="Execute the bounded six-endpoint Wizard pair-detail API schema pilot.",
    )
    parser.add_argument(
        "--wizard-detail-endpoints",
        default="",
        help=(
            "Optional comma-separated endpoint subset for a credit-bounded pair-detail "
            "retry, for example backtest. Empty means all six endpoints."
        ),
    )
    parser.add_argument(
        "--current-pair-group-keys",
        default="binance|daily|ETH|WIF",
        help=(
            "Comma-separated exact current-board pair_group_key values. History network "
            "and disk work is limited to this explicit selection."
        ),
    )
    parser.add_argument(
        "--minimum-free-disk-mib",
        type=int,
        default=512,
        help="Free-disk reserve that bounded current-board history materialization must preserve.",
    )
    parser.add_argument(
        "--execute-wizard-sweep",
        action="store_true",
        help="Execute the full Wizard sweep after credit preflight; otherwise write a zero-credit plan.",
    )
    parser.add_argument(
        "--execute-daily-pipeline",
        action="store_true",
        help=(
            "Execute the storage-gated 19-stage current Wizard research cadence. "
            "Without this flag the command is plan-only."
        ),
    )
    parser.add_argument(
        "--wizard-attempt-only",
        action="store_true",
        help="Write dated/latest-attempt Wizard evidence without replacing the canonical active sweep.",
    )
    parser.add_argument("--exchange", default="Dydx")
    parser.add_argument("--period", type=int, default=320)
    parser.add_argument("--spread-type", default="Static")
    parser.add_argument("--roll-w", type=int, default=42)
    parser.add_argument("--asset", default=None)
    parser.add_argument("--coin", default=None)
    parser.add_argument("--days", type=int, default=500)
    parser.add_argument(
        "--intraday-days",
        type=int,
        default=30,
        help="Days of intraday history for build-hyperliquid-research-bundle.",
    )
    parser.add_argument(
        "--max-wizard-age-hours",
        type=float,
        default=24.0,
        help="Maximum age of a Wizard capture before it is blocked from exact-mode proof.",
    )
    parser.add_argument(
        "--execute-wizard-proof",
        action="store_true",
        help="Request a bounded Crypto Wizards custom-series proof. It also requires QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF=true.",
    )
    parser.add_argument(
        "--apply-dynamic-v2",
        action="store_true",
        help=(
            "Explicitly apply the reviewed, research-only Dynamic-v2 comparator; "
            "without this flag the command is preflight-only."
        ),
    )
    parser.add_argument(
        "--dynamic-v2-reviewer",
        default="",
        help="Reviewer identity required for explicit Dynamic-v2 activation.",
    )
    parser.add_argument(
        "--dynamic-v2-review-note",
        default="",
        help="Substantive review note required for explicit Dynamic-v2 activation.",
    )
    parser.add_argument(
        "--dynamic-v2-review-packet-id",
        default="",
        help="Exact immutable Dynamic-v2 review packet ID required for activation.",
    )
    parser.add_argument(
        "--apply-ou-v3",
        action="store_true",
        help=(
            "Explicitly apply the reviewed, research-only OU-v3 comparator; "
            "without this flag the command is preflight-only."
        ),
    )
    parser.add_argument(
        "--ou-v3-reviewer",
        default="",
        help="Reviewer identity required for explicit OU-v3 activation.",
    )
    parser.add_argument(
        "--ou-v3-review-note",
        default="",
        help="Substantive review note required for explicit OU-v3 activation.",
    )
    parser.add_argument(
        "--ou-v3-review-packet-id",
        default="",
        help="Exact immutable OU-v3 review packet ID required for activation.",
    )
    parser.add_argument(
        "--apply-ou-v4",
        action="store_true",
        help=(
            "Explicitly apply the reviewed, research-only OU-v4 comparator; "
            "without this flag the command is preflight-only."
        ),
    )
    parser.add_argument(
        "--ou-v4-reviewer",
        default="",
        help="Reviewer identity required for explicit OU-v4 activation.",
    )
    parser.add_argument(
        "--ou-v4-review-note",
        default="",
        help="Substantive review note required for explicit OU-v4 activation.",
    )
    parser.add_argument(
        "--ou-v4-review-packet-id",
        default="",
        help="Exact immutable OU-v4 review packet ID required for activation.",
    )
    parser.add_argument(
        "--apply-ou-v5",
        action="store_true",
        help=(
            "Explicitly apply the reviewed, research-only OU-v5 comparator; "
            "without this flag the command is preflight-only."
        ),
    )
    parser.add_argument(
        "--ou-v5-reviewer",
        default="",
        help="Reviewer identity required for explicit OU-v5 activation.",
    )
    parser.add_argument(
        "--ou-v5-review-note",
        default="",
        help="Substantive review note required for explicit OU-v5 activation.",
    )
    parser.add_argument(
        "--ou-v5-review-packet-id",
        default="",
        help="Exact immutable OU-v5 review packet ID required for activation.",
    )
    parser.add_argument(
        "--apply-ou-v6",
        action="store_true",
        help=(
            "Explicitly apply the reviewed, research-only terminal OU-v6 comparator; "
            "without this flag the command is preflight-only."
        ),
    )
    parser.add_argument(
        "--ou-v6-reviewer",
        default="",
        help="Reviewer identity required for explicit OU-v6 activation.",
    )
    parser.add_argument(
        "--ou-v6-review-note",
        default="",
        help="Substantive review note required for explicit OU-v6 activation.",
    )
    parser.add_argument(
        "--ou-v6-review-packet-id",
        default="",
        help="Exact immutable terminal OU-v6 review packet ID required for activation.",
    )
    parser.add_argument(
        "--wizard-proof-queue-source",
        choices=("hypothesis", "exact-mode-parity"),
        default="hypothesis",
        help="Use the score-gated hypothesis queue or the unfiltered exact-mode parity queue.",
    )
    parser.add_argument(
        "--pair-group-id",
        default=None,
        help="Optional exact pair-group identity for the Wizard mode parity queue.",
    )
    parser.add_argument("--range", dest="lookback_range", default="max")
    parser.add_argument(
        "--run-research",
        action="store_true",
        help="After an official Crypto Wizards Min5 crawl, run pair-detail experiments and strategy acceptance.",
    )
    parser.add_argument(
        "--research-funding-path",
        type=Path,
        default=None,
        help="Optional funding CSV used when a long-history build reruns the guarded research spine.",
    )
    parser.add_argument(
        "--derive-hedge-ratio",
        action="store_true",
        help="Derive hedge ratio and beta from dYdX candles instead of using manual CLI values.",
    )
    parser.add_argument("--notional-usd", type=float, default=1000.0)
    parser.add_argument(
        "--max-assets",
        type=int,
        default=0,
        help="Maximum new asset funding fetches for a resumable exhaustive checkpoint; zero means all pending assets.",
    )
    parser.add_argument(
        "--notionals",
        default="250,1000,5000",
        help="Comma-separated per-leg notionals for public Hyperliquid L2 slippage sampling.",
    )
    parser.add_argument(
        "--slippage-min-samples",
        type=int,
        default=DEFAULT_SLIPPAGE_CALIBRATION_MIN_SAMPLES,
        help="Complete public L2 samples required for each leg in the rolling calibration window.",
    )
    parser.add_argument(
        "--slippage-window-hours",
        type=float,
        default=DEFAULT_SLIPPAGE_CALIBRATION_WINDOW_HOURS,
        help="Rolling public L2 calibration-window length in hours.",
    )
    parser.add_argument(
        "--slippage-cadence-minutes",
        type=int,
        default=DEFAULT_SLIPPAGE_CALIBRATION_CADENCE_MINUTES,
        help="Minutes between public L2 evidence snapshots.",
    )
    parser.add_argument("--acceptance-path", type=Path, default=None)
    parser.add_argument("--journal-path", type=Path, default=None)
    parser.add_argument("--history-path", type=Path, default=None)
    parser.add_argument("--funding-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--brief-path", type=Path, default=None)
    parser.add_argument("--entry-threshold", type=float, default=2.0)
    parser.add_argument("--exit-threshold", type=float, default=0.0)
    parser.add_argument(
        "--required-trades",
        type=int,
        default=100,
        help="Required trades for strategy-trade-count-gap reporting (used by strategy-trade-count-gap).",
    )
    parser.add_argument(
        "--experiment-path",
        type=Path,
        default=None,
        help="Experiment CSV path for strategy-trade-count-gap reporting.",
    )
    parser.add_argument("--walkforward-splits", type=int, default=5)
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--max-combo-size", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--readiness-threshold", type=float, default=0.65)
    parser.add_argument("--sweep-mode", choices=list(SWEEP_MODES), default="light")
    parser.add_argument("--dashboard-refresh-profile", choices=["monitor", "deep"], default="deep")
    parser.add_argument(
        "--wizard-archive-destination",
        type=Path,
        default=None,
        help="Existing off-volume directory for Wizard archive preflight or an explicitly approved copy-only stage.",
    )
    parser.add_argument(
        "--archive-copy-approval-id",
        default="",
        help="Explicit one-run approval identifier required by the copy-only archive stage.",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--model-sha256",
        default="",
        help="Expected SHA-256 required before loading a shadow model artifact.",
    )
    parser.add_argument("--market", default=None)
    parser.add_argument(
        "--venue",
        default=None,
        help="Execution venue to target: dydx|hyperliquid|binance|binanceus|coinbase|bybit|auto",
    )
    parser.add_argument(
        "--order-approval-id",
        default=None,
        help="One-run approval identifier required for a Hyperliquid Testnet pair submission.",
    )
    parser.add_argument(
        "--indexer-base",
        default="",
        help="Override the dYdX indexer base URL. If omitted, uses QPA_INDEXER_BASE or https://indexer.dydx.trade.",
    )
    parser.add_argument(
        "--indexer-scheme",
        default="",
        help="Force indexer URL scheme (http/https) without editing scripts or urls.",
    )
    parser.add_argument(
        "--allow-stale-fetch",
        action="store_true",
        help="Use existing payload files if fetches fail (for offline reruns or flaky DNS).",
    )
    parser.add_argument(
        "--allow-blocked-exploration",
        action="store_true",
        help="Run data-exploration commands even when readiness gates are blocked.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip network fetches for fetch-dydx-two-leg-data and require existing payload files in --download-dir.",
    )
    parser.add_argument(
        "--collect-l2",
        action="store_true",
        help="Collect one public Hyperliquid L2 snapshot for the frozen candidate set during the no-order research cycle.",
    )
    parser.add_argument("--diagnostic-output", type=Path, default=None)
    parser.add_argument("--json-path", type=Path, default=None)
    parser.add_argument("--download-dir", type=Path, default=None)
    parser.add_argument(
        "--mcp-url", default=None, help="Apify MCP server URL; defaults to APIFY_MCP_SERVER_URL"
    )
    parser.add_argument(
        "--source-filter",
        dest="source_filter",
        default=None,
        help="Limit Apify source refresh to a single source_id",
    )
    parser.add_argument(
        "--wait-seconds", type=int, default=90, help="Actor fetch timeout for Apify (seconds)"
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip actor runs while building the Apify source coverage table",
    )
    parser.add_argument("--apify-token", default=None, help="Optional APIFY_API_TOKEN override")
    parser.add_argument(
        "--apify-actor-credit-ceiling",
        type=int,
        default=0,
        help="Required per-actor reserved credit ceiling for authorized Apify fetches",
    )
    parser.add_argument("--endpoint-name", default="manual")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--realized-return", type=float, default=None)
    parser.add_argument("--regime", default="unknown")
    parser.add_argument("--trade-id", default=None)
    parser.add_argument(
        "--allow-spread-only",
        action="store_true",
        help="Allow research-spine to run pair-detail experiments without two-leg price history.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="For archive-from-index or run-orchestrator, write planned actions without executing stages.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="For archive-from-index, request apply mode. Apply is intentionally blocked until reviewed.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=ROOT / ".env.local",
        help="Local env file with API keys. Defaults to .env.local.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="For run-orchestrator, ignore reusable stage artifacts where supported.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="For run-orchestrator, stop at the first blocked or failed stage.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="For run-orchestrator, only build reporting stages.",
    )
    args = parser.parse_args(argv)
    authority_gated_environment_commands = {
        "crypto-wizards-full-sweep",
        "hyperliquid-testnet-market-inventory",
        "phase00-checkpoint",
        "phase00-closure-verify",
        "phase00-descendant-invalidate",
        "phase00-lineage-seal",
        "phase00-maintenance-resume",
        "phase00-maintenance-start",
        "refresh-apify-sources",
    }
    phase00_no_external_mode = (
        os.getenv("QPA_PHASE00_EXTERNAL_EFFECT_MODE", "").strip()
        == "NO_EXTERNAL_NO_ORDER"
    )
    if (
        load_environment
        and not phase00_no_external_mode
        and args.command not in authority_gated_environment_commands
    ):
        load_env_file(args.env_file)
    if not args.indexer_base:
        args.indexer_base = (
            os.getenv("QPA_INDEXER_BASE", "https://indexer.dydx.trade").strip()
            or "https://indexer.dydx.trade"
        )
    if not args.indexer_scheme:
        args.indexer_scheme = os.getenv("QPA_INDEXER_SCHEME", "").strip()
    if args.command == "system-check":
        result = system_check()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-corrective-governance":
        result = build_corrective_governance(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-corrective-data-evidence":
        result = build_corrective_data_evidence(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-corrective-wizard-parity":
        result = build_corrective_wizard_parity(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "review-wizard-dynamic-v2":
        activation = build_dynamic_v2_reviewed_activation(
            root=ROOT,
            apply=args.apply_dynamic_v2,
            reviewer=args.dynamic_v2_reviewer,
            review_note=args.dynamic_v2_review_note,
            review_packet_id=args.dynamic_v2_review_packet_id,
        )
        refresh = None
        if (
            args.apply_dynamic_v2
            and activation.summary.get("status") == "APPLIED_RESEARCH_COMPARATOR_ONLY"
        ):
            refresh = refresh_activated_dynamic_v2_proofs(root=ROOT)
        print(
            json.dumps(
                {
                    "activation": activation.summary,
                    "proof_refresh": refresh.summary if refresh is not None else None,
                    "paths": {
                        **{key: str(value) for key, value in activation.paths.items()},
                        **(
                            {f"refresh_{key}": str(value) for key, value in refresh.paths.items()}
                            if refresh is not None
                            else {}
                        ),
                    },
                },
                indent=2,
            )
        )
        if (
            args.apply_dynamic_v2
            and activation.summary.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY"
        ):
            raise SystemExit(2)
    elif args.command == "build-wizard-dynamic-v2-supreme-review":
        result = build_dynamic_v2_supreme_review(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-wizard-ou-v4-supreme-review":
        result = build_ou_v4_supreme_review(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-wizard-ou-v5-supreme-review":
        result = build_ou_v5_supreme_review(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-wizard-ou-v6-supreme-review":
        result = build_ou_v6_supreme_review(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "close-wizard-ou-v6-terminal":
        result = build_ou_v6_terminal_closure(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-wizard-ou-v4-failure-attribution":
        result = build_ou_v4_failure_attribution(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-wizard-ou-v5-failure-attribution":
        result = build_ou_v5_failure_attribution(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "evaluate-wizard-ou-v2-holdout":
        result = evaluate_ou_v2_blind_holdout(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "register-wizard-ou-trend-selector-v1":
        result = register_ou_trend_selector_v1_holdout(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-wizard-next-capture-manifest":
        result = build_corrective_wizard_capture_manifest(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-wizard-credit-budget":
        result = build_wizard_credit_budget_contract(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "reconcile-wizard-capture-manifest":
        result = reconcile_corrective_wizard_capture_manifest(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "run-wizard-ou-v3-holdout":
        result = (
            _blocked_direct_wizard_external_execution(command=args.command)
            if args.execute_wizard_proof
            else run_ou_v3_prospective_holdout(root=ROOT, execute=False)
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "register-wizard-ou-v4-holdout":
        result = register_ou_v4_prospective_holdout(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "register-wizard-ou-v5-holdout":
        result = register_ou_v5_prospective_holdout(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "register-wizard-ou-v6-holdout":
        result = register_ou_v6_prospective_holdout(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "run-wizard-ou-v4-holdout":
        blockers, manifest_path = (
            _ou_v4_manual_capture_blockers(root=ROOT) if args.execute_wizard_proof else ([], None)
        )
        result = (
            _blocked_direct_wizard_external_execution(
                command=args.command,
                blockers=blockers,
                evidence_path=manifest_path,
            )
            if args.execute_wizard_proof
            else run_ou_v4_prospective_holdout(
                root=ROOT,
                execute=False,
            )
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "run-wizard-ou-v5-holdout":
        blockers, manifest_path = (
            _ou_v5_manual_capture_blockers(root=ROOT) if args.execute_wizard_proof else ([], None)
        )
        result = (
            _blocked_direct_wizard_external_execution(
                command=args.command,
                blockers=blockers,
                evidence_path=manifest_path,
            )
            if args.execute_wizard_proof
            else run_ou_v5_prospective_holdout(
                root=ROOT,
                execute=False,
            )
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "review-wizard-ou-v3":
        activation = build_ou_v3_reviewed_activation(
            root=ROOT,
            apply=args.apply_ou_v3,
            reviewer=args.ou_v3_reviewer,
            review_note=args.ou_v3_review_note,
            review_packet_id=args.ou_v3_review_packet_id,
        )
        refresh = None
        if (
            args.apply_ou_v3
            and activation.summary.get("status") == "APPLIED_RESEARCH_COMPARATOR_ONLY"
        ):
            refresh = refresh_activated_ou_v3_proofs(root=ROOT)
        print(
            json.dumps(
                {
                    "activation": activation.summary,
                    "proof_refresh": refresh.summary if refresh is not None else None,
                    "paths": {
                        **{key: str(value) for key, value in activation.paths.items()},
                        **(
                            {f"refresh_{key}": str(value) for key, value in refresh.paths.items()}
                            if refresh is not None
                            else {}
                        ),
                    },
                },
                indent=2,
            )
        )
        if (
            args.apply_ou_v3
            and activation.summary.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY"
        ):
            raise SystemExit(2)
    elif args.command == "review-wizard-ou-v4":
        activation = build_reviewed_ou_v4_activation(
            root=ROOT,
            apply=args.apply_ou_v4,
            reviewer=args.ou_v4_reviewer,
            review_note=args.ou_v4_review_note,
            review_packet_id=args.ou_v4_review_packet_id,
        )
        refresh = None
        if (
            args.apply_ou_v4
            and activation.summary.get("status") == "APPLIED_RESEARCH_COMPARATOR_ONLY"
        ):
            refresh = refresh_activated_ou_v4_proofs(root=ROOT)
        print(
            json.dumps(
                {
                    "activation": activation.summary,
                    "proof_refresh": (refresh.summary if refresh is not None else None),
                    "paths": {
                        **{key: str(value) for key, value in activation.paths.items()},
                        **(
                            {f"refresh_{key}": str(value) for key, value in refresh.paths.items()}
                            if refresh is not None
                            else {}
                        ),
                    },
                },
                indent=2,
            )
        )
        if (
            args.apply_ou_v4
            and activation.summary.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY"
        ):
            raise SystemExit(2)
    elif args.command == "review-wizard-ou-v5":
        activation = build_reviewed_ou_v5_activation(
            root=ROOT,
            apply=args.apply_ou_v5,
            reviewer=args.ou_v5_reviewer,
            review_note=args.ou_v5_review_note,
            review_packet_id=args.ou_v5_review_packet_id,
        )
        refresh = None
        if (
            args.apply_ou_v5
            and activation.summary.get("status") == "APPLIED_RESEARCH_COMPARATOR_ONLY"
        ):
            refresh = refresh_activated_ou_v5_proofs(root=ROOT)
        print(
            json.dumps(
                {
                    "activation": activation.summary,
                    "proof_refresh": refresh.summary if refresh is not None else None,
                    "paths": {
                        **{key: str(value) for key, value in activation.paths.items()},
                        **(
                            {f"refresh_{key}": str(value) for key, value in refresh.paths.items()}
                            if refresh is not None
                            else {}
                        ),
                    },
                },
                indent=2,
            )
        )
        if (
            args.apply_ou_v5
            and activation.summary.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY"
        ):
            raise SystemExit(2)
    elif args.command == "review-wizard-ou-v6":
        activation = build_reviewed_ou_v6_activation(
            root=ROOT,
            apply=args.apply_ou_v6,
            reviewer=args.ou_v6_reviewer,
            review_note=args.ou_v6_review_note,
            review_packet_id=args.ou_v6_review_packet_id,
        )
        refresh = None
        if (
            args.apply_ou_v6
            and activation.summary.get("status") == "APPLIED_RESEARCH_COMPARATOR_ONLY"
        ):
            refresh = refresh_activated_ou_v6_proofs(root=ROOT)
        print(
            json.dumps(
                {
                    "activation": activation.summary,
                    "proof_refresh": refresh.summary if refresh is not None else None,
                    "paths": {
                        **{key: str(value) for key, value in activation.paths.items()},
                        **(
                            {f"refresh_{key}": str(value) for key, value in refresh.paths.items()}
                            if refresh is not None
                            else {}
                        ),
                    },
                },
                indent=2,
            )
        )
        if (
            args.apply_ou_v6
            and activation.summary.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY"
        ):
            raise SystemExit(2)
    elif args.command == "register-wizard-copula-v2":
        result = register_copula_behavioral_v2(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "run-wizard-copula-proof":
        result = (
            _blocked_direct_wizard_external_execution(command=args.command)
            if args.execute_wizard_proof
            else run_current_copula_behavioral_proofs(root=ROOT, execute=False)
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-corrective-statistical-remediation":
        result = build_corrective_statistical_remediation(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-corrective-daily-cadence":
        try:
            with governed_evidence_write_lock(ROOT, blocking=False):
                result = build_corrective_daily_cadence(root=ROOT, install=True)
        except (GovernedEvidenceMaintenanceActive, GovernedEvidenceLockBusy) as exc:
            result = CommandResult(
                paths={},
                summary={
                    "status": "DEFERRED_PHASE00_OR_GOVERNED_LOCK",
                    "blockers": [str(exc)],
                    "live_trading_authorized": False,
                },
            )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "run-corrective-l2-capture":
        try:
            with governed_evidence_write_lock(ROOT, blocking=False):
                result = run_corrective_l2_capture(root=ROOT)
        except (GovernedEvidenceMaintenanceActive, GovernedEvidenceLockBusy) as exc:
            result = CommandResult(
                paths={},
                summary={
                    "status": "DEFERRED_PHASE00_OR_GOVERNED_LOCK",
                    "blockers": [str(exc)],
                    "live_trading_authorized": False,
                },
            )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-current-wizard-hyperliquid-evidence-command-center":
        result = build_current_wizard_hyperliquid_evidence_command_center(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "install-corrective-l2-cadence":
        try:
            with governed_evidence_write_lock(ROOT, blocking=False):
                result = install_corrective_l2_launch_agent(root=ROOT)
        except (GovernedEvidenceMaintenanceActive, GovernedEvidenceLockBusy) as exc:
            result = {
                "status": "DEFERRED_PHASE00_OR_GOVERNED_LOCK",
                "blockers": [str(exc)],
                "live_trading_authorized": False,
            }
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "build-corrective-agent-governance":
        result = build_corrective_agent_governance(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "register-stage5-protocol":
        result = build_registered_stage5_protocol(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-reset-readiness":
        result = build_corrective_wizard_reset_readiness(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "phase00-maintenance-start":
        result = start_phase00_maintenance(
            root=ROOT,
            reason=str(args.phase00_reason),
            ttl_seconds=int(args.phase00_ttl_seconds),
            wait_timeout_seconds=float(args.phase00_wait_timeout_seconds),
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "phase00-checkpoint":
        result = build_phase00_quiesced_checkpoint(
            root=ROOT,
            wait_timeout_seconds=float(args.phase00_wait_timeout_seconds),
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "phase00-lineage-seal":
        result = seal_phase00_active_artifact_lineage(
            root=ROOT,
            wait_timeout_seconds=float(args.phase00_wait_timeout_seconds),
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {
                        key: str(value) for key, value in result.paths.items()
                    },
                },
                indent=2,
            )
        )
    elif args.command == "phase00-descendant-invalidate":
        result = build_phase00_descendant_invalidation(
            root=ROOT,
            wait_timeout_seconds=float(args.phase00_wait_timeout_seconds),
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {
                        key: str(value) for key, value in result.paths.items()
                    },
                },
                indent=2,
            )
        )
    elif args.command == "phase00-closure-verify":
        result = build_phase00_closure_verification(
            root=ROOT,
            wait_timeout_seconds=float(args.phase00_wait_timeout_seconds),
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {
                        key: str(value) for key, value in result.paths.items()
                    },
                },
                indent=2,
            )
        )
    elif args.command == "phase00-maintenance-resume":
        result = resume_phase00_maintenance(
            root=ROOT,
            maintenance_id=str(args.phase00_maintenance_id),
            wait_timeout_seconds=float(args.phase00_wait_timeout_seconds),
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-canonical-runtime-contract":
        result = build_canonical_scheduler_runtime_contract(
            root=ROOT,
            now_iso=datetime.now(UTC).isoformat(),
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-scheduler-runtime-readiness":
        result = build_corrective_scheduler_runtime_readiness(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-comparator-review-control":
        result = build_corrective_wizard_comparator_review_control(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-stage4-handoff-readiness":
        result = build_corrective_stage4_handoff_readiness(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-corrective-release-gates":
        result = build_corrective_release_gates(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "capture-live-input-parity-evidence":
        result = capture_live_input_parity_evidence(root=ROOT)
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "build-live-canary-executor-preflight":
        candidate_path = ROOT / "reports" / "active" / "testnet_candidate_receipt.json"
        policy_path = ROOT / "config" / "live_canary_policy.json"
        candidate = json.loads(candidate_path.read_text()) if candidate_path.is_file() else {}
        policy = json.loads(policy_path.read_text()) if policy_path.is_file() else {}
        result = build_live_canary_executor_preflight(
            root=ROOT,
            candidate=candidate,
            policy_id=_policy_id(policy),
        )
        print(json.dumps({"preflight": str(result)}, indent=2))
    elif args.command == "run-live-canary-executor":
        result = run_live_canary_executor(
            root=ROOT,
            execute=bool(args.execute_live_canary),
            recover_only=bool(args.recover_live_canary),
            authorization_sha256=str(args.authorization_sha256 or ""),
            approval_id=str(args.approval_id or ""),
            acknowledgement=str(args.live_canary_acknowledgement or ""),
        )
        print(json.dumps(result.as_dict(), indent=2))
    elif args.command == "complete-corrective-plan":
        result = complete_corrective_plan(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-canonical-program-status":
        result = build_canonical_program_status(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "capture-wizard-api-credit-receipt":
        result = capture_wizard_api_credit_receipt(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "corrective-artifact-retention":
        result = run_corrective_artifact_retention(root=ROOT, apply=bool(args.apply))
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-artifact-index":
        result = build_artifact_index()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "current-state":
        result = current_state()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-research-journal":
        result = build_wizard_research_journal(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-pair-universe":
        result = build_pair_universe()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-market-venue-context":
        result = build_market_venue_context()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-venue-lane-test-plan":
        result = build_venue_lane_test_plan()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-multi-venue-history-readiness":
        result = build_multi_venue_history_readiness(top_n=args.top_n)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-venue-route-scorecard":
        result = build_venue_route_scorecard(max_pairs=args.max_pairs)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "refresh-hyperliquid-market-context":
        result = refresh_hyperliquid_market_context()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "refresh-hyperliquid-execution-cost-snapshot":
        try:
            notionals = tuple(
                float(value.strip()) for value in args.notionals.split(",") if value.strip()
            )
        except ValueError as exc:
            raise SystemExit(f"--notionals must be comma-separated numbers: {exc}") from exc
        result = refresh_hyperliquid_execution_cost_snapshot(
            max_pairs=args.max_pairs,
            notionals=notionals,
            candidate_path=args.candidate_path,
            min_samples=args.slippage_min_samples,
            window_hours=args.slippage_window_hours,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "refresh-hyperliquid-funding-history":
        result = refresh_hyperliquid_funding_history(
            max_pairs=args.max_pairs,
            candidate_path=args.candidate_path,
            days=args.days,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "fetch-hyperliquid-candles":
        coin = args.coin or args.asset or args.market
        if not coin:
            raise SystemExit("--coin, --asset, or --market is required")
        path = fetch_hyperliquid_candles(coin=coin, interval=args.interval or "1d", days=args.days)
        print(
            json.dumps(
                {"path": str(path), "coin": coin, "interval": args.interval or "1d"}, indent=2
            )
        )
    elif args.command == "build-hyperliquid-pair-history":
        if not args.asset_x or not args.asset_y:
            raise SystemExit("--asset-x and --asset-y are required")
        path = build_hyperliquid_pair_history(
            asset_x=args.asset_x,
            asset_y=args.asset_y,
            interval=args.interval or "1d",
            pair_id=args.pair_id if args.pair_id != "1" else None,
            hedge_ratio=None if args.derive_hedge_ratio else args.hedge_ratio,
            beta=None if args.derive_hedge_ratio else args.beta,
            zscore_window=args.zscore_window,
        )
        print(
            json.dumps(
                {
                    "path": str(path),
                    "asset_x": args.asset_x,
                    "asset_y": args.asset_y,
                    "interval": args.interval or "1d",
                },
                indent=2,
            )
        )
    elif args.command == "build-hyperliquid-research-bundle":
        intervals = tuple(
            value.strip() for value in (args.interval or "1d,5m").split(",") if value.strip()
        )
        result = build_hyperliquid_research_bundle(
            max_pairs=args.max_pairs,
            intervals=intervals,
            daily_days=args.days,
            intraday_days=args.intraday_days,
            refresh=not args.skip_fetch,
            candidate_path=args.candidate_path,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-hyperliquid-wizard-hypothesis-queue":
        result = build_hyperliquid_wizard_hypothesis_queue(
            max_wizard_age_hours=args.max_wizard_age_hours
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-exhaustive-wizard-mode-proof-queue":
        result = build_exhaustive_wizard_mode_proof_queue(
            root=ROOT,
            pair_group_id=args.pair_group_id,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "run-hyperliquid-wizard-mode-proofs":
        proof_queue_path = None
        if args.wizard_proof_queue_source == "exact-mode-parity":
            proof_queue = build_exhaustive_wizard_mode_proof_queue(
                root=ROOT,
                pair_group_id=args.pair_group_id,
            )
            proof_queue_path = proof_queue.paths["queue"]
        proof_kwargs = {
            "root": ROOT,
            "max_pairs": args.max_pairs,
            "execute": args.execute_wizard_proof,
        }
        if proof_queue_path is not None:
            proof_kwargs["queue_path"] = proof_queue_path
        result = (
            _blocked_direct_wizard_external_execution(command=args.command)
            if args.execute_wizard_proof
            else run_hyperliquid_wizard_mode_proofs(**proof_kwargs)
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-hyperliquid-pair-cost-model":
        result = build_hyperliquid_pair_cost_model(
            max_pairs=args.max_pairs,
            leg_notional_usd=args.notional_usd,
            candidate_path=args.candidate_path,
            min_samples=args.slippage_min_samples,
            window_hours=args.slippage_window_hours,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-hyperliquid-evidence-cadence":
        result = build_hyperliquid_evidence_cadence(
            target_samples=args.slippage_min_samples,
            cadence_minutes=args.slippage_cadence_minutes,
            window_hours=args.slippage_window_hours,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "hyperliquid-lane-readiness":
        result = build_hyperliquid_lane_report()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "hyperliquid-testnet-market-inventory":
        frame = refresh_hyperliquid_testnet_market_inventory(root=ROOT)
        output = ROOT / "reports" / "active" / "hyperliquid_testnet_market_inventory.csv"
        evidence = (
            ROOT
            / "reports"
            / "active"
            / "hyperliquid_testnet_market_inventory_evidence.json"
        )
        checked = pd.to_datetime(
            frame.get("checked_at_utc", pd.Series(dtype=str)), utc=True, errors="coerce"
        )
        print(
            json.dumps(
                {
                    "summary": {
                        "schema_version": "hyperliquid_testnet_market_inventory.v1",
                        "rows": len(frame),
                        "tradable_perps": int(
                            frame.get("tradable_perp", pd.Series(False, index=frame.index))
                            .fillna(False)
                            .astype(str)
                            .str.strip()
                            .str.lower()
                            .isin({"1", "true", "yes", "y", "pass", "ready"})
                            .sum()
                        ),
                        "fetch_blocked_rows": int(
                            frame.get("fetch_blocker", pd.Series("", index=frame.index))
                            .fillna("")
                            .astype(str)
                            .str.strip()
                            .ne("")
                            .sum()
                        ),
                        "checked_at_min_utc": (
                            checked.min().isoformat() if checked.notna().any() else ""
                        ),
                        "checked_at_max_utc": (
                            checked.max().isoformat() if checked.notna().any() else ""
                        ),
                        "order_submission_included": False,
                        "testnet_order_authority": False,
                        "live_trading_authorized": False,
                    },
                    "paths": {
                        "inventory": str(output),
                        "inventory_evidence": str(evidence),
                    },
                },
                indent=2,
            )
        )
    elif args.command == "hyperliquid-testnet-preflight":
        frame = write_hyperliquid_testnet_preflight_report(
            root=ROOT, config=HyperliquidTestnetConfig.paper_testnet_from_env()
        )
        output = ROOT / "reports" / "active" / "hyperliquid_testnet_preflight.csv"
        print(frame.to_string(index=False))
        print(f"hyperliquid_testnet_preflight: {output}")
    elif args.command == "hyperliquid-testnet-margin-snapshot":
        frame = write_hyperliquid_testnet_margin_snapshot(
            root=ROOT, config=HyperliquidTestnetConfig.paper_testnet_from_env()
        )
        output = ROOT / "reports" / "active" / "hyperliquid_testnet_margin_snapshot.csv"
        print(frame.to_string(index=False))
        print(f"hyperliquid_testnet_margin_snapshot: {output}")
    elif args.command == "hyperliquid-testnet-collateral-transfer-preflight":
        result = build_testnet_collateral_transfer_preflight(
            root=ROOT,
            config=HyperliquidTestnetConfig.paper_testnet_from_env(),
            approval_id=str(args.transfer_approval_id or ""),
            amount_usd=float(args.transfer_amount_usd),
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "run-hyperliquid-testnet-collateral-transfer":
        result = run_testnet_collateral_transfer(
            root=ROOT,
            config=HyperliquidTestnetConfig.paper_testnet_from_env(),
            preflight_id=str(args.transfer_preflight_id or ""),
            approval_id=str(args.transfer_approval_id or ""),
            amount_usd=float(args.transfer_amount_usd),
            acknowledgement=str(args.testnet_collateral_transfer_acknowledgement or ""),
            execute=bool(args.execute_testnet_collateral_transfer),
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "hyperliquid-testnet-pair-execution-preflight":
        if not str(args.order_approval_id or "").strip():
            raise SystemExit(
                "hyperliquid-testnet-pair-execution-preflight requires --order-approval-id"
            )
        result = build_testnet_pair_execution_preflight(
            root=ROOT,
            action=str(args.testnet_pair_action),
            approval_id=str(args.order_approval_id).strip(),
            config=HyperliquidTestnetConfig.paper_testnet_from_env(),
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
                default=str,
            )
        )
    elif args.command == "run-hyperliquid-testnet-pair-execution":
        if not str(args.order_approval_id or "").strip():
            raise SystemExit("run-hyperliquid-testnet-pair-execution requires --order-approval-id")
        if not str(args.testnet_pair_preflight_id or "").strip():
            raise SystemExit(
                "run-hyperliquid-testnet-pair-execution requires --testnet-pair-preflight-id"
            )
        result = run_testnet_pair_execution(
            root=ROOT,
            action=str(args.testnet_pair_action),
            preflight_id=str(args.testnet_pair_preflight_id).strip(),
            approval_id=str(args.order_approval_id).strip(),
            acknowledgement=str(args.testnet_pair_acknowledgement or ""),
            execute=bool(args.execute_hyperliquid_testnet_pair),
            config=HyperliquidTestnetConfig.paper_testnet_from_env(),
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
                default=str,
            )
        )
    elif args.command == "hyperliquid-testnet-smoke-approval-template":
        result = write_testnet_smoke_approval_template(root=ROOT)
        print(
            json.dumps(
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in result.items()
                },
                indent=2,
            )
        )
    elif args.command == "build-hyperliquid-testnet-lifecycle-gate":
        result = build_testnet_lifecycle_gate(root=ROOT)
        print(
            json.dumps(
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in result.items()
                },
                indent=2,
            )
        )
    elif args.command == "capture-hyperliquid-testnet-lifecycle-evidence":
        result = capture_hyperliquid_testnet_lifecycle_evidence(root=ROOT)
        print(
            json.dumps(
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in result.items()
                },
                indent=2,
            )
        )
    elif args.command == "hyperliquid-testnet-sign-smoke-approval":
        result = sign_testnet_smoke_approval(root=ROOT)
        print(
            json.dumps(
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in result.items()
                },
                indent=2,
            )
        )
    elif args.command == "hyperliquid-testnet-recover-pair-state":
        if not str(args.order_approval_id or "").strip():
            raise SystemExit("hyperliquid-testnet-recover-pair-state requires --order-approval-id")
        config = replace(
            HyperliquidTestnetConfig.paper_testnet_from_env(),
            order_approval_id=str(args.order_approval_id).strip(),
        )
        result = HyperliquidTestnetPairExecutor(
            state_path=HYPERLIQUID_TESTNET_EXECUTION_STATE_JSON
        ).recover_incomplete_pair(config)
        print(json.dumps(asdict(result), indent=2))
    elif args.command == "hyperliquid-execution-compatibility":
        frame = refresh_hyperliquid_execution_compatibility_table(root=ROOT)
        output = ROOT / "reports" / "active" / "hyperliquid_execution_market_compatibility.csv"
        print(frame.to_string(index=False))
        print(f"hyperliquid_execution_compatibility: {output}")
    elif args.command == "hyperliquid-testnet-candidate-shortlist":
        frame = refresh_hyperliquid_testnet_candidate_shortlist(root=ROOT, max_pairs=args.max_pairs)
        output = ROOT / "reports" / "active" / "hyperliquid_testnet_candidate_shortlist.csv"
        print(frame.to_string(index=False))
        print(f"hyperliquid_testnet_candidate_shortlist: {output}")
    elif args.command == "run-hyperliquid-research-cycle":
        result = run_hyperliquid_research_cycle(root=ROOT, collect_l2=args.collect_l2)
        print(
            json.dumps(
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in result.items()
                },
                indent=2,
            )
        )
    elif args.command == "run-hyperliquid-auxiliary-timeframe-validation":
        result = build_hyperliquid_auxiliary_timeframe_validation(
            root=ROOT,
            interval=args.interval or "4h",
            intraday_days=args.intraday_days,
        )
        print(
            json.dumps(
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in result.items()
                },
                indent=2,
            )
        )
    elif args.command == "gmx-testnet-market-inventory":
        frame = refresh_gmx_testnet_market_inventory(root=ROOT)
        output = ROOT / "reports" / "active" / "gmx_testnet_market_inventory.csv"
        print(frame.to_string(index=False))
        print(f"gmx_testnet_market_inventory: {output}")
    elif args.command == "gmx-execution-compatibility":
        frame = refresh_gmx_execution_compatibility_table(root=ROOT)
        output = ROOT / "reports" / "active" / "gmx_execution_market_compatibility.csv"
        print(frame.to_string(index=False))
        print(f"gmx_execution_compatibility: {output}")
    elif args.command == "gmx-testnet-candidate-shortlist":
        frame = refresh_gmx_testnet_candidate_shortlist(root=ROOT, max_pairs=args.max_pairs)
        output = ROOT / "reports" / "active" / "gmx_testnet_candidate_shortlist.csv"
        print(frame.to_string(index=False))
        print(f"gmx_testnet_candidate_shortlist: {output}")
    elif args.command == "fetch-binance-spot-candles":
        symbol = args.market or args.asset or args.coin
        if not symbol:
            raise SystemExit("--market, --asset, or --coin is required")
        path = fetch_binance_spot_candles(
            symbol=symbol, interval=args.interval or "1d", limit=args.limit
        )
        print(
            json.dumps(
                {"path": str(path), "symbol": symbol, "interval": args.interval or "1d"}, indent=2
            )
        )
    elif args.command == "build-binance-spot-pair-history":
        if not args.asset_x or not args.asset_y:
            raise SystemExit("--asset-x and --asset-y are required")
        path = build_binance_spot_pair_history(
            asset_x=args.asset_x,
            asset_y=args.asset_y,
            interval=args.interval or "1d",
            pair_id=args.pair_id if args.pair_id != "1" else None,
            hedge_ratio=None if args.derive_hedge_ratio else args.hedge_ratio,
            beta=None if args.derive_hedge_ratio else args.beta,
            zscore_window=args.zscore_window,
        )
        print(
            json.dumps(
                {
                    "path": str(path),
                    "asset_x": args.asset_x,
                    "asset_y": args.asset_y,
                    "interval": args.interval or "1d",
                },
                indent=2,
            )
        )
    elif args.command == "binance-spot-history-readiness":
        result = build_binance_spot_lane_report()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "binance-testnet-preflight":
        frame = binance_testnet_preflight(root=ROOT)
        print(frame.to_string(index=False))
        print(
            f"binance_testnet_preflight: {ROOT / 'reports' / 'active' / 'binance_testnet_preflight.csv'}"
        )
    elif args.command == "binance-testnet-adapter-contract":
        rows = [
            validate_venue_order_client_adapter("binance_spot_testnet"),
            validate_venue_order_client_adapter("binance_usdm_testnet"),
        ]
        frame = pd.DataFrame(rows)
        output = ROOT / "reports" / "active" / "binance_testnet_adapter_contract.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(frame, output, index=False)
        print(frame.to_string(index=False))
        print(f"binance_testnet_adapter_contract: {output}")
    elif args.command == "binance-testnet-pair-preflight":
        if not args.asset_x or not args.asset_y:
            raise SystemExit("binance-testnet-pair-preflight requires --asset-x and --asset-y")
        lane = normalize_venue_name(args.venue or "binance_usdm_testnet")
        if lane == "binance_spot_testnet":
            config = BinanceTestnetConfig.spot_testnet_from_env()
        elif lane == "binance_usdm_testnet":
            config = BinanceTestnetConfig.usdm_testnet_from_env()
        else:
            raise SystemExit("--venue must be binance_spot_testnet or binance_usdm_testnet")
        frame = binance_testnet_pair_preflight(
            asset_x=args.asset_x,
            asset_y=args.asset_y,
            config=config,
            root=ROOT,
        )
        print(frame.to_string(index=False))
        print(
            f"binance_testnet_pair_preflight: {ROOT / 'reports' / 'active' / 'binance_testnet_pair_preflight.csv'}"
        )
    elif args.command == "fetch-yahoo-crypto-candles":
        symbol = args.market or args.asset or args.coin
        if not symbol:
            raise SystemExit("--market, --asset, or --coin is required")
        path = fetch_yahoo_crypto_candles(
            symbol=symbol,
            interval=args.interval or "1d",
            lookback_range=args.lookback_range or "max",
        )
        print(
            json.dumps(
                {
                    "path": str(path),
                    "symbol": symbol,
                    "interval": args.interval or "1d",
                    "range": args.lookback_range or "max",
                },
                indent=2,
            )
        )
    elif args.command == "build-yahoo-crypto-pair-history":
        if not args.asset_x or not args.asset_y:
            raise SystemExit("--asset-x and --asset-y are required")
        path = build_yahoo_crypto_pair_history(
            asset_x=args.asset_x,
            asset_y=args.asset_y,
            interval=args.interval or "1d",
            pair_id=args.pair_id if args.pair_id != "1" else None,
            hedge_ratio=None if args.derive_hedge_ratio else args.hedge_ratio,
            beta=None if args.derive_hedge_ratio else args.beta,
            zscore_window=args.zscore_window,
        )
        print(
            json.dumps(
                {
                    "path": str(path),
                    "asset_x": args.asset_x,
                    "asset_y": args.asset_y,
                    "interval": args.interval or "1d",
                },
                indent=2,
            )
        )
    elif args.command == "refresh-yahoo-crypto-pair-history":
        if not args.asset_x or not args.asset_y:
            raise SystemExit("--asset-x and --asset-y are required")
        path = refresh_yahoo_crypto_pair_history(
            asset_x=args.asset_x,
            asset_y=args.asset_y,
            interval=args.interval or "1d",
            lookback_range=args.lookback_range or "max",
            pair_id=args.pair_id if args.pair_id != "1" else None,
            hedge_ratio=None if args.derive_hedge_ratio else args.hedge_ratio,
            beta=None if args.derive_hedge_ratio else args.beta,
            zscore_window=args.zscore_window,
        )
        print(
            json.dumps(
                {
                    "path": str(path),
                    "asset_x": args.asset_x,
                    "asset_y": args.asset_y,
                    "interval": args.interval or "1d",
                    "range": args.lookback_range or "max",
                },
                indent=2,
            )
        )
    elif args.command == "yahoo-crypto-history-readiness":
        result = build_yahoo_crypto_lane_report()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "refresh-yahoo-research-candidates":
        result = refresh_yahoo_research_candidates(
            max_pairs=args.max_pairs,
            interval=args.interval or "1d",
            lookback_range=args.lookback_range or "max",
            zscore_window=args.zscore_window,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "backfill-yahoo-crypto-funding":
        funding_path = args.funding_path or ROOT / "data" / "processed" / "dydx_funding.csv"
        result = backfill_yahoo_crypto_funding(funding_path=funding_path)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-trade-dataset":
        result = build_trade_dataset(input_dir=args.input_dir, funding_path=args.funding_path)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "promote-trade-dataset":
        result = promote_trade_dataset(candidate_pointer_path=args.input_dir)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "train-trade-gate":
        result = train_trade_gate(
            input_path=args.input_dir,
            walkforward_splits=args.walkforward_splits,
            min_train_rows=args.min_train_rows,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "run-model-gated-backtest":
        result = run_model_gated_backtest()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "export-trade-gate-model":
        result = export_trade_gate_model()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-command-dashboard":
        result = build_command_dashboard(refresh_profile=args.dashboard_refresh_profile)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-v2-math-diagnostic":
        result = build_v2_math_diagnostic(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "reevaluate-current-wizard-hyperliquid-math":
        old_manifest = args.old_replay_manifest or (
            ROOT
            / "reports"
            / "active"
            / "current_wizard_hyperliquid_canonical_replay_manifest.json"
        )
        result = reevaluate_current_wizard_hyperliquid_math(
            root=ROOT,
            old_manifest_path=old_manifest,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-math-v2-acceptance":
        result = build_math_v2_acceptance(root=ROOT)
        print(
            json.dumps(
                {
                    **{key: value for key, value in result.items() if not isinstance(value, Path)},
                    "paths": {
                        key: str(value) for key, value in result.items() if isinstance(value, Path)
                    },
                },
                indent=2,
            )
        )
    elif args.command == "build-v2-preflight-run":
        result = build_v2_preflight_run(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "validate-v2-run":
        if not args.run_id:
            raise SystemExit("--run-id is required")
        result = validate_v2_run(root=ROOT, run_id=args.run_id)
        print(json.dumps({**result, "run_dir": str(result["run_dir"])}, indent=2))
    elif args.command == "publish-v2-run-status":
        if not args.run_id:
            raise SystemExit("--run-id is required")
        result = publish_v2_run_status(root=ROOT, run_id=args.run_id)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "apify-cost-audit":
        frame = apify_cost_audit_rows()
        output = ROOT / "reports" / "dashboard" / "apify_cost_audit.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(frame, output, index=False)
        print(frame.to_string(index=False))
        print(f"apify_cost_audit: {output}")
    elif args.command == "paper-candidate-shortlist":
        frame = paper_candidate_shortlist_rows()
        output = ROOT / "reports" / "dashboard" / "paper_candidate_shortlist.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(frame, output, index=False)
        print(frame.to_string(index=False))
        print(f"paper_candidate_shortlist: {output}")
    elif args.command == "focused-paper-validation":
        frame = focused_paper_validation_rows()
        output = ROOT / "reports" / "dashboard" / "focused_paper_validation.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(frame, output, index=False)
        print(frame.to_string(index=False))
        print(f"focused_paper_validation: {output}")
    elif args.command == "run-langgraph-agent-workflow":
        pair_id = "" if args.pair_id == "1" else (args.pair_id or "")
        result = run_langgraph_agent_workflow(
            stage=args.stage,
            pair_id=pair_id,
            dry_run=args.dry_run,
            force_refresh=args.force_refresh,
            fail_fast=args.fail_fast,
            report_only=args.report_only,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "run-orchestrator":
        pair_id = "" if args.pair_id == "1" else (args.pair_id or "")
        result = run_orchestrator(
            stage=args.stage,
            pair_id=pair_id,
            dry_run=args.dry_run,
            force_refresh=args.force_refresh,
            fail_fast=args.fail_fast,
            report_only=args.report_only,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "interactive-mixtape-solution":
        brief = args.notes or ""
        if args.brief_path is not None:
            brief = args.brief_path.read_text(encoding="utf-8")
        if not brief.strip():
            raise SystemExit("interactive-mixtape-solution requires --brief-path or --notes")
        result = build_interactive_mixtape_solution(
            brief=brief,
            brand_name=args.title or "Interactive Mixtape",
            target_platform=args.source or "Shopify",
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-mini-agent-orchestration":
        result = build_mini_agent_orchestration()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-orchestrator-assistant":
        result = build_orchestrator_assistant()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-specialist-scoreboard":
        result = build_specialist_scoreboard()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "run-rl-research":
        pair_id = "" if args.pair_id == "1" else (args.pair_id or "")
        result = run_rl_research(pair_id=pair_id)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "run-rl-idea-scout":
        result = run_rl_idea_scout(
            pair_filter=args.pair,
            top_ideas=args.top_n,
            similarity_k=args.similarity_k,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "run-brain-cycle":
        pair_id = "" if args.pair_id == "1" else (args.pair_id or "")
        policy_candidates = max(1, args.top_n)
        max_recommendations = max(1, args.top_n)
        result = run_brain_cycle(
            root=ROOT,
            pair_id=pair_id,
            policy_candidates=policy_candidates,
            max_recommendations=max_recommendations,
            readiness_threshold=args.readiness_threshold,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "brain-readiness-report":
        result = build_brain_readiness_report(
            root=ROOT,
            score_threshold=args.readiness_threshold,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "run-magicka-learning":
        pair_id = "" if args.pair_id == "1" else (args.pair_id or "")
        policy_candidates = max(1, args.top_n)
        result = run_magicka_learning_cycle(
            root=ROOT, pair_id=pair_id, policy_candidates=policy_candidates
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "run-sequential-thinking-magicka":
        pair_id = "" if args.pair_id == "1" else (args.pair_id or "")
        max_recommendations = max(1, args.top_n)
        result = run_sequential_thinking_magicka(
            root=ROOT, pair_id=pair_id, max_recommendations=max_recommendations
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "train-rl-ppo":
        pair_id = "" if args.pair_id == "1" else (args.pair_id or "")
        result = train_ppo_research_policy(pair_id=pair_id)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "export-rl-policy":
        result = export_rl_policy()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "ingest-research-source":
        if not args.source_type or not args.title or not args.source:
            raise SystemExit("ingest-research-source requires --source-type, --title, and --source")
        result = ingest_research_source(
            source_type=args.source_type,
            title=args.title,
            source_path_or_url=args.source,
            root=ROOT,
            author=args.author,
            channel_or_publisher=args.channel_or_publisher,
            topic_tags=args.topic_tags,
            status=args.status,
            review_status=args.review_status,
            quarantine_status=args.quarantine_status,
            confidence=args.confidence,
            notes=args.notes,
            source_id=args.source_id,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "ingest-paper-library":
        if args.paper_dir is None:
            raise SystemExit("ingest-paper-library requires --paper-dir")
        result = ingest_paper_library(source_dir=args.paper_dir, root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "verify-paper-sources":
        result = verify_paper_sources(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-paper-adversarial-reviews":
        result = build_paper_adversarial_reviews(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "run-paper-critical-checks":
        result = run_paper_critical_checks(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "run-paper-reproduction-suite":
        result = run_paper_reproduction_suite(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-v1-v1-1-comparison":
        result = build_versioned_learning_comparison(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-research-source-registry":
        result = build_research_source_registry(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "research-source-audit":
        result = research_source_audit(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "extract-research-knowledge":
        youtube_result = extract_youtube_research(root=ROOT)
        udemy_result = extract_udemy_research(root=ROOT)
        ccxt_result = extract_research_knowledge(root=ROOT)
        result = CommandResult(
            paths={**youtube_result.paths, **udemy_result.paths, **ccxt_result.paths},
            summary={
                "rows": (
                    int(youtube_result.summary.get("rows", 0))
                    + int(udemy_result.summary.get("knowledge_rows", 0))
                    + int(ccxt_result.summary.get("rows", 0))
                ),
                "youtube_rows": int(youtube_result.summary.get("rows", 0)),
                "udemy_rows": int(udemy_result.summary.get("knowledge_rows", 0)),
                "ccxt_rows": int(ccxt_result.summary.get("ccxt_rows", 0)),
            },
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-research-knowledge-store":
        result = build_research_knowledge_store(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "research-knowledge-summary":
        result = research_knowledge_summary(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "refresh-augmented-research":
        result = refresh_augmented_research(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "refresh-udemy-research":
        result = refresh_udemy_research(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "refresh-youtube-collection":
        result = refresh_youtube_collection(
            root=ROOT,
            fetch_live=not args.no_fetch,
            force=args.force_refresh,
            catalog_path=Path(args.source) if args.source else None,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-youtube-caption-insights":
        result = build_youtube_caption_insights(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-youtube-brain":
        result = build_youtube_brain(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-youtube-hypotheses":
        result = build_youtube_pair_hypotheses(
            root=ROOT, candidate_path=Path(args.candidate_path) if args.candidate_path else None
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "refresh-youtube-outcomes":
        result = refresh_youtube_outcome_memory(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-youtube-brain-dashboard":
        result = build_youtube_brain_dashboard(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "run-youtube-brain":
        result = run_youtube_brain_cycle(
            root=ROOT,
            fetch_live=False if args.no_fetch else None,
            force=args.force_refresh,
            candidate_path=Path(args.candidate_path) if args.candidate_path else None,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "run-hudson-thames-youtube-research":
        result = run_hudson_thames_youtube_research(root=ROOT, fetch_live=not args.no_fetch)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "run-youtube-hypothesis-validation":
        result = run_youtube_hypothesis_validation(
            root=ROOT,
            candidate_path=Path(args.candidate_path) if args.candidate_path else None,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "run-base-rl":
        pair_id = "" if args.pair_id == "1" else (args.pair_id or "")
        result = run_base_rl(root=ROOT, pair_id=pair_id)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "evaluate-base-rl":
        result = evaluate_base_rl(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "base-rl-paper-handoff":
        frame = base_rl_paper_handoff_report(root=ROOT)
        print(frame.to_string(index=False))
        print(
            f"base_rl_paper_handoff: {ROOT / 'reports' / 'rl' / 'base_rl_paper_handoff_status.csv'}"
        )
    elif args.command == "run-augmented-rl":
        pair_id = "" if args.pair_id == "1" else (args.pair_id or "")
        result = run_augmented_rl(root=ROOT, pair_id=pair_id)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "compare-base-vs-augmented-rl":
        result = evaluate_base_rl(root=ROOT)
        report = ROOT / "reports" / "rl" / "base_vs_augmented_pair_comparison.csv"
        print(
            json.dumps({"summary": result.summary, "paths": {"comparison": str(report)}}, indent=2)
        )
    elif args.command == "promotion-readiness-report":
        frame = (
            pd.read_csv(ROOT / "reports" / "rl" / "base_rl_promotion_readiness.csv")
            if (ROOT / "reports" / "rl" / "base_rl_promotion_readiness.csv").exists()
            else pd.DataFrame()
        )
        output = ROOT / "reports" / "rl" / "base_rl_promotion_readiness.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(frame, output, index=False)
        print(frame.to_string(index=False))
        print(f"promotion_readiness_report: {output}")
    elif args.command == "refresh-base-rl-feedback":
        result = refresh_base_rl_feedback(root=ROOT)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "archive-from-index":
        result = archive_from_index(dry_run=not args.apply)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-evidence":
        result = build_wizard_evidence()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-hypotheses":
        result = build_wizard_hypotheses()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-diagnostic-confirmation":
        result = build_wizard_diagnostic_confirmation()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-discovery-triage":
        result = build_wizard_discovery_triage()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-local-parity":
        result = build_wizard_local_parity()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-pair-detail-capture-queue":
        result = build_wizard_pair_detail_capture_queue()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-mode-matrix-capture-queue":
        result = build_wizard_mode_matrix_capture_queue()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-pair-settings-capture-template":
        result = build_wizard_pair_settings_capture_template()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "import-wizard-pair-settings-capture":
        if args.input_dir is None:
            raise SystemExit(
                "import-wizard-pair-settings-capture requires --input-dir pointing to a completed CSV"
            )
        result = import_wizard_pair_settings_capture(args.input_dir)
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-replay-handoff":
        result = build_wizard_replay_handoff()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-mode-replay-capability":
        result = build_wizard_mode_replay_capability()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-mode-comparison":
        result = build_wizard_mode_comparison()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-exploratory-cost-sensitivity":
        result = build_wizard_exploratory_cost_sensitivity()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-exact-mode-capture-queue":
        result = build_wizard_exact_mode_capture_queue()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-research-pack":
        result = build_wizard_research_pack()
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "ingest-exhaustive-wizard-dashboard-captures":
        result = ingest_exhaustive_wizard_dashboard_captures(
            root=ROOT,
            input_dir=args.input_dir,
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-exhaustive-wizard-hyperliquid-run":
        result = build_exhaustive_wizard_hyperliquid_run(
            root=ROOT,
            source_path=args.wizard_source_path,
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-exhaustive-wizard-api-refresh-delta":
        result = build_exhaustive_wizard_api_refresh_delta(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "run-wizard-pair-detail-api-pilot":
        wizard_detail_endpoints = tuple(
            value.strip() for value in args.wizard_detail_endpoints.split(",") if value.strip()
        )
        result = run_wizard_pair_detail_api_pilot(
            root=ROOT,
            pair_group_key=args.wizard_pair_group_key,
            execute=args.execute_wizard_detail_pilot,
            api_key=os.getenv("CRYPTO_WIZARDS_API_KEY"),
            daily_credit_limit=args.wizard_daily_credit_limit,
            reserved_credits=args.wizard_reserved_credits,
            endpoint_names=wizard_detail_endpoints or None,
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-current-wizard-hyperliquid-handoff":
        result = build_current_wizard_hyperliquid_handoff(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "materialize-current-wizard-hyperliquid-history":
        selected_pair_keys = tuple(
            value.strip() for value in args.current_pair_group_keys.split(",") if value.strip()
        )
        result = materialize_current_wizard_hyperliquid_history(
            root=ROOT,
            pair_group_keys=selected_pair_keys,
            minimum_free_disk_bytes=args.minimum_free_disk_mib * 1024 * 1024,
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "run-current-wizard-hyperliquid-canonical-replay":
        result = run_current_wizard_hyperliquid_canonical_replay(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "materialize-current-wizard-hyperliquid-cost-evidence":
        selected_pair_keys = tuple(
            value.strip() for value in args.current_pair_group_keys.split(",") if value.strip()
        )
        result = materialize_current_wizard_hyperliquid_cost_evidence(
            root=ROOT,
            pair_group_keys=selected_pair_keys,
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "run-current-wizard-hyperliquid-observed-cost-replay":
        result = run_current_wizard_hyperliquid_observed_cost_replay(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "run-current-wizard-hyperliquid-walkforward":
        result = run_current_wizard_hyperliquid_walkforward(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-current-wizard-hyperliquid-regime-attribution":
        result = build_current_wizard_hyperliquid_regime_attribution(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "run-current-wizard-hyperliquid-robustness":
        result = run_current_wizard_hyperliquid_robustness(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-current-wizard-hyperliquid-concentration":
        result = build_current_wizard_hyperliquid_concentration(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-current-wizard-hyperliquid-failure-attribution":
        result = build_current_wizard_hyperliquid_failure_attribution(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-current-wizard-hyperliquid-failure-routing-index":
        result = build_current_wizard_hyperliquid_failure_routing_index(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-current-wizard-hyperliquid-leverage-surface":
        result = build_current_wizard_hyperliquid_leverage_surface(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-current-wizard-hyperliquid-learning-ledger":
        result = build_current_wizard_hyperliquid_learning_ledger(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "validate-current-wizard-hyperliquid-chain":
        result = validate_current_wizard_hyperliquid_chain(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-current-wizard-hyperliquid-operating-cadence":
        result = build_current_wizard_hyperliquid_operating_cadence(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "run-current-wizard-hyperliquid-daily-pipeline":
        result = run_current_wizard_hyperliquid_daily_pipeline(
            root=ROOT,
            execute=args.execute_daily_pipeline,
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-current-wizard-hyperliquid-completion-audit":
        result = build_current_wizard_hyperliquid_completion_audit(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-current-wizard-hyperliquid-storage-reclamation-plan":
        result = build_current_wizard_hyperliquid_storage_reclamation_plan(
            root=ROOT,
            archive_destination=args.wizard_archive_destination,
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "stage-current-wizard-hyperliquid-archive-copy":
        result = stage_current_wizard_hyperliquid_archive_copy(
            root=ROOT,
            archive_destination=args.wizard_archive_destination,
            approval_id=args.archive_copy_approval_id,
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "plan-current-wizard-hyperliquid-archive-release":
        result = build_current_wizard_hyperliquid_archive_release_dry_run(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "validate-current-wizard-hyperliquid-testnet-protocol":
        result = validate_current_wizard_hyperliquid_testnet_protocol(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-current-wizard-ou-optimal-overlay":
        result = build_current_wizard_ou_optimal_overlay(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-exhaustive-wizard-hyperliquid-mapping-refresh":
        result = build_exhaustive_wizard_hyperliquid_mapping_refresh(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-exhaustive-wizard-hyperliquid-replay-preflight":
        result = build_exhaustive_wizard_hyperliquid_replay_preflight(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "materialize-exhaustive-wizard-hyperliquid-history":
        result = materialize_exhaustive_wizard_hyperliquid_history(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "run-exhaustive-wizard-hyperliquid-canonical-replay":
        result = run_exhaustive_wizard_hyperliquid_canonical_replay(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "materialize-exhaustive-hyperliquid-funding-evidence":
        result = materialize_exhaustive_hyperliquid_funding_evidence(
            root=ROOT,
            max_assets=args.max_assets,
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-exhaustive-wizard-hyperliquid-cost-evidence":
        result = build_exhaustive_wizard_hyperliquid_cost_evidence(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "run-exhaustive-wizard-hyperliquid-observed-cost-replay":
        result = run_exhaustive_wizard_hyperliquid_observed_cost_replay(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "run-exhaustive-wizard-hyperliquid-walkforward":
        result = run_exhaustive_wizard_hyperliquid_walkforward(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-exhaustive-wizard-hyperliquid-regime-attribution":
        result = build_exhaustive_wizard_hyperliquid_regime_attribution(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "run-exhaustive-wizard-hyperliquid-robustness":
        result = run_exhaustive_wizard_hyperliquid_robustness(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-exhaustive-wizard-hyperliquid-concentration":
        result = build_exhaustive_wizard_hyperliquid_concentration(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-exhaustive-wizard-hyperliquid-leverage-surface":
        result = build_exhaustive_wizard_hyperliquid_leverage_surface(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "build-exhaustive-wizard-hyperliquid-learning-ledger":
        result = build_exhaustive_wizard_hyperliquid_learning_ledger(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "run-exhaustive-wizard-hyperliquid-validation":
        result = run_exhaustive_wizard_hyperliquid_validation(root=ROOT)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "ingest-wizard-pair-detail-ui-bundles":
        result = ingest_wizard_pair_detail_ui_bundles(
            root=ROOT,
            input_dir=args.input_dir,
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "verify-wizard-local-mode":
        result = verify_wizard_local_mode(
            history_path=args.history_path,
            wizard_capture_path=args.json_path,
            output_name=args.output_name or "bnb_stx_daily_320_static_spread",
            entry_threshold=args.entry_threshold,
            exit_threshold=args.exit_threshold,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-wizard-local-verification-batch":
        result = build_wizard_local_verification_batch(
            queue_path=args.queue_path or args.input_dir,
            max_pairs=args.max_pairs,
        )
        print(
            json.dumps(
                {"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}},
                indent=2,
            )
        )
    elif args.command == "build-dictionaries":
        build_dictionaries()
    elif args.command == "ingest-fixtures":
        ingest_fixtures(args.input_dir)
    elif args.command == "ingest-crypto-wizards-scanner":
        ingest_crypto_wizards_scanner(args.input_dir)
    elif args.command == "normalize-enrichment-fixtures":
        output = normalize_enrichment_fixtures(args.source, args.input_dir, args.output_path)
        print(f"normalized_enrichment_feed: {output}")
        print(
            f"normalization_report: {ROOT / 'reports' / f'{_canonical_source_name(args.source)}_normalization_report.csv'}"
        )
    elif args.command == "materialize-p2-rerun-subset":
        print_materialize_p2_rerun_subset(args.input_dir, args.output_path)
    elif args.command == "ingest-pair-details":
        ingest_pair_details(args.input_dir)
    elif args.command == "pair-detail-capture-checklist":
        write_pair_detail_capture_checklist(args.input_dir)
    elif args.command == "pair-detail-quality":
        write_pair_detail_quality_report(args.input_dir)
    elif args.command == "run-demo-backtest":
        run_demo_backtest()
    elif args.command == "run-demo-experiments":
        run_demo_experiments()
    elif args.command == "run-fixture-experiments":
        run_fixture_experiments(args.input_dir, args.funding_path)
    elif args.command == "run-pair-detail-experiments":
        pairs_arg = args.pair_ids if args.pair_ids else args.pair
        pair_filter = tuple(
            pair.strip() for pair in (pairs_arg.split(",") if pairs_arg else []) if pair.strip()
        )
        strategy_ids = None
        if args.strategy_ids.strip():
            strategy_ids = tuple(
                int(x) for x in args.strategy_ids.split(",") if x.strip().isdigit()
            )
        elif args.recommended_strategy_priority:
            strategy_ids = tuple(
                _strategy_id_priority_for_pair_details(
                    args.input_dir or ROOT / "data" / "raw" / "pair_details", pair_filter
                )
            )
        run_pair_detail_experiments(
            args.input_dir,
            args.funding_path,
            pair_filter=pair_filter,
            strategy_ids=strategy_ids,
        )
    elif args.command == "list-strategies":
        for strategy in STRATEGIES:
            print(f"{strategy.id:02d} {strategy.name}")
    elif args.command == "list-crypto-wizards-endpoints":
        print(pd.DataFrame(endpoint_rows()).to_string(index=False))
    elif args.command == "check-live-config":
        check_live_config(args.endpoint)
    elif args.command == "diagnose-crypto-wizards":
        diagnose_crypto_wizards(args.endpoint, args.diagnostic_output)
    elif args.command == "crypto-wizards-min5-request-template":
        print_crypto_wizards_min5_request_template(
            asset_x=args.asset_x,
            asset_y=args.asset_y,
            priority=args.priority,
            cw_strategy=args.cw_strategy,
            exchange=args.exchange,
            period=args.period,
            spread_type=args.spread_type,
            roll_w=args.roll_w,
            asset=args.asset,
            output_path=args.output_path,
        )
    elif args.command == "import-crypto-wizards-payload":
        if args.json_path is None:
            raise SystemExit("import-crypto-wizards-payload requires --json-path")
        import_crypto_wizards_payload(args.json_path, args.endpoint_name)
    elif args.command == "import-crypto-wizards-zscores":
        if args.json_path is None:
            raise SystemExit("import-crypto-wizards-zscores requires --json-path")
        if not args.asset_x or not args.asset_y:
            raise SystemExit("import-crypto-wizards-zscores requires --asset-x and --asset-y")
        import_crypto_wizards_zscores_history(
            args.json_path,
            asset_x=args.asset_x,
            asset_y=args.asset_y,
            exchange=args.exchange,
            interval=args.interval or "Min5",
            period=args.period,
            spread_type=args.spread_type,
            roll_w=args.roll_w,
            output_dir=args.output_path,
            run_research=args.run_research,
        )
    elif args.command == "import-crypto-wizards-backtest":
        if args.json_path is None:
            raise SystemExit("import-crypto-wizards-backtest requires --json-path")
        if not args.asset_x or not args.asset_y:
            raise SystemExit("import-crypto-wizards-backtest requires --asset-x and --asset-y")
        import_crypto_wizards_backtest_history(
            args.json_path,
            asset_x=args.asset_x,
            asset_y=args.asset_y,
            exchange=args.exchange,
            interval=args.interval or "Min5",
            period=args.period,
            spread_type=args.spread_type,
            roll_w=args.roll_w,
            output_dir=args.output_path,
            run_research=args.run_research,
        )
    elif args.command == "import-pair-detail-capture":
        if args.json_path is None:
            raise SystemExit("import-pair-detail-capture requires --json-path")
        import_pair_detail_capture(args.json_path, args.output_name)
    elif args.command == "import-latest-pair-detail-download":
        import_latest_pair_detail_download(args.download_dir, args.output_name)
    elif args.command == "import-dydx-candles":
        if args.json_path is None:
            raise SystemExit("import-dydx-candles requires --json-path")
        import_dydx_candles(args.json_path, args.output_path)
    elif args.command == "import-dydx-candle-bundle":
        if args.json_path is None:
            raise SystemExit("import-dydx-candle-bundle requires --json-path")
        import_dydx_candle_bundle_from_cli(args.json_path, args.output_path, args.zscore_window)
    elif args.command == "dydx-two-leg-request-template":
        print_dydx_two_leg_request_template(
            pair=args.pair,
            asset_x=args.asset_x,
            asset_y=args.asset_y,
            pair_id=args.pair_id,
            hedge_ratio=args.hedge_ratio,
            beta=args.beta,
            zscore_window=args.zscore_window,
            limit=args.limit,
            indexer_base=args.indexer_base,
            indexer_scheme=args.indexer_scheme,
            output_path=args.output_path,
        )
    elif args.command == "fetch-dydx-two-leg-data":
        print_fetch_dydx_two_leg_data(
            pair=args.pair,
            asset_x=args.asset_x,
            asset_y=args.asset_y,
            pair_id=args.pair_id,
            hedge_ratio=args.hedge_ratio,
            beta=args.beta,
            zscore_window=args.zscore_window,
            indexer_base=args.indexer_base,
            indexer_scheme=args.indexer_scheme,
            limit=args.limit,
            output_dir=args.download_dir,
            run_research=args.run_research,
            derive_hedge_ratio=args.derive_hedge_ratio,
            allow_stale_fetch=args.allow_stale_fetch,
            skip_fetch=args.skip_fetch,
            funding_path=args.funding_path,
        )
    elif args.command == "build-dydx-pair-history":
        if args.left_candles is None or args.right_candles is None:
            raise SystemExit("build-dydx-pair-history requires --left-candles and --right-candles")
        if not args.asset_x or not args.asset_y:
            raise SystemExit("build-dydx-pair-history requires --asset-x and --asset-y")
        build_dydx_pair_history(
            left_candles=args.left_candles,
            right_candles=args.right_candles,
            asset_x=args.asset_x,
            asset_y=args.asset_y,
            pair_id=args.pair_id,
            hedge_ratio=args.hedge_ratio,
            beta=args.beta,
            interval=args.interval,
            zscore_window=args.zscore_window,
            output_path=args.output_path,
            derive_hedge_ratio=args.derive_hedge_ratio,
            funding_path=args.funding_path,
        )
    elif args.command == "build-dydx-long-history-pair":
        if not args.asset_x or not args.asset_y:
            raise SystemExit("build-dydx-long-history-pair requires --asset-x and --asset-y")
        resolved_research_funding_path = _resolve_research_funding_path(
            args.research_funding_path, args.funding_path
        )
        build_dydx_long_history_pair(
            input_dir=args.input_dir,
            asset_x=args.asset_x,
            asset_y=args.asset_y,
            pair_id=args.pair_id,
            hedge_ratio=args.hedge_ratio,
            beta=args.beta,
            interval=args.interval,
            zscore_window=args.zscore_window,
            derive_hedge_ratio=args.derive_hedge_ratio,
            run_research=args.run_research,
            funding_path=resolved_research_funding_path,
        )
    elif args.command == "inspect-pair-detail-capture":
        if args.json_path is None:
            raise SystemExit("inspect-pair-detail-capture requires --json-path")
        inspect_pair_detail_capture(args.json_path)
    elif args.command == "capture-preflight":
        print_pair_detail_capture_preflight(args.json_path, args.output_path)
    elif args.command == "verify-crypto-wizards-live-artifacts":
        verify_crypto_wizards_live_artifacts()
    elif args.command == "crypto-wizards-live-coverage":
        write_crypto_wizards_live_coverage_report()
    elif args.command == "check-dydx-config":
        check_dydx_config()
    elif args.command == "dydx-order-adapter-contract":
        print_dydx_order_adapter_contract(args.output_path)
    elif args.command == "dydx-execution-checklist":
        print_dydx_execution_checklist(args.output_path)
    elif args.command == "funding-requirements":
        print_funding_requirements(args.pair, args.output_path)
    elif args.command == "funding-template":
        print_funding_template(args.pair, args.output_path)
    elif args.command == "funding-template-check":
        print_funding_template_check(args.input_dir, args.output_path)
    elif args.command == "import-funding-template":
        print_import_funding_template(args.input_dir, args.output_path)
    elif args.command == "fetch-dydx-funding":
        print_fetch_dydx_funding(args.market, args.output_path)
    elif args.command == "export-dydx-funding":
        if args.json_path is None:
            raise SystemExit("export-dydx-funding requires --json-path")
        path = export_dydx_funding_payload(args.json_path, args.output_path, args.market)
        print(f"dydx_funding_csv: {path}")
    elif args.command == "funding-coverage":
        print_funding_coverage(args.funding_path, args.pair, args.output_path)
    elif args.command == "funded-research-spine":
        print_funded_research_spine(
            args.funding_path,
            input_dir=args.input_dir,
            require_two_leg=not args.allow_spread_only,
            output_path=args.output_path,
        )
    elif args.command == "refresh-apify-sources":
        mcp_url = args.mcp_url or os.getenv("APIFY_MCP_SERVER_URL", "").strip()
        if not mcp_url:
            raise SystemExit("refresh-apify-sources requires --mcp-url or APIFY_MCP_SERVER_URL")
        result = refresh_apify_sources(
            root=ROOT,
            mcp_url=mcp_url,
            source_filter=args.source_filter,
            do_fetch=not args.no_fetch,
            api_token=args.apify_token,
            wait_seconds=args.wait_seconds,
            actor_credit_ceiling=args.apify_actor_credit_ceiling,
        )
        print(
            json.dumps(
                {
                    "coverage_path": str(result.coverage_path),
                    "manifest_path": str(result.manifest_path),
                    "source_count": result.source_count,
                    "sampled_count": result.sampled_count,
                    "needs_api_key_count": result.needs_key_count,
                    "failed_count": result.failed_count,
                },
                indent=2,
            )
        )
    elif args.command == "apify-source-summary":
        mcp_url = args.mcp_url or os.getenv("APIFY_MCP_SERVER_URL", "").strip()
        if not mcp_url:
            raise SystemExit("apify-source-summary requires --mcp-url or APIFY_MCP_SERVER_URL")
        sources = parse_apify_sources_from_mcp_url(mcp_url)
        print(
            json.dumps(
                [
                    {
                        "source_id": source,
                        "venue": infer_apify_venue(source),
                        "type": "utility"
                        if source.startswith("apify/")
                        else "market_or_context_feed",
                    }
                    for source in sources
                ],
                indent=2,
            )
        )
    elif args.command == "strategy-acceptance-checklist":
        print_strategy_acceptance_checklist()
    elif args.command == "strategy-failure-attribution":
        print_strategy_failure_attribution(args.output_path)
    elif args.command == "research-unblock-plan":
        print_research_unblock_plan(args.output_path)
    elif args.command == "zscore-threshold-sweep":
        _assert_ready_for_exploration()
        print_zscore_threshold_sweep(args.input_dir, args.funding_path, args.output_path)
    elif args.command == "strategy-trade-count-gap":
        print_strategy_trade_count_gap(
            experiment_path=args.experiment_path,
            required_trades=args.required_trades,
            output_path=args.output_path,
        )
    elif args.command == "dydx-pair-expansion-plan":
        print_dydx_pair_expansion_plan(
            max_pairs=args.max_pairs,
            limit=args.limit,
            indexer_base=args.indexer_base,
            indexer_scheme=args.indexer_scheme,
            output_path=args.output_path,
        )
    elif args.command == "dydx-live-market-selector":
        print_dydx_live_market_selector(
            max_pairs=args.max_pairs,
            indexer_base=args.indexer_base,
            output_path=args.output_path,
        )
    elif args.command == "dydx-anchor-sweep":
        anchors = None
        if args.asset:
            anchors = [item.strip() for item in str(args.asset).split(",") if item.strip()]
        print_dydx_anchor_sweep(
            anchors=anchors,
            max_pairs=args.max_pairs,
            indexer_base=args.indexer_base,
            output_path=args.output_path,
        )
    elif args.command == "dydx-live-market-counts":
        print_dydx_live_market_counts(
            indexer_base=args.indexer_base,
            output_path=args.output_path,
        )
    elif args.command == "dydx-local-pair-universe":
        print_dydx_local_pair_universe(
            input_dir=args.input_dir,
            pair_output_dir=args.download_dir,
            funding_output_path=args.funding_path,
            zscore_window=args.zscore_window,
            output_path=args.output_path,
            run_research=args.run_research,
        )
    elif args.command == "dydx-long-history-plan":
        print_dydx_long_history_plan(
            pair=args.pair,
            asset_x=args.asset_x,
            asset_y=args.asset_y,
            pair_id=args.pair_id,
            windows=args.windows,
            limit=args.limit,
            resolution=args.interval or "5MINS",
            indexer_base=args.indexer_base,
            indexer_scheme=args.indexer_scheme,
            to_iso=args.to_iso,
            output_path=args.output_path,
        )
    elif args.command == "dydx-long-history-coverage":
        print_dydx_long_history_coverage(
            pair=args.pair,
            asset_x=args.asset_x,
            asset_y=args.asset_y,
            pair_id=args.pair_id,
            windows=args.windows,
            limit=args.limit,
            resolution=args.interval or "5MINS",
            indexer_base=args.indexer_base,
            indexer_scheme=args.indexer_scheme,
            to_iso=args.to_iso,
            output_path=args.output_path,
        )
    elif args.command == "fetch-dydx-long-history-windows":
        required_pair_id = args.pair_id if args.pair_id != "1" else None
        frame = fetch_dydx_long_history_windows(
            plan_path=args.input_dir,
            max_windows=args.windows,
            indexer_base=args.indexer_base,
            indexer_scheme=args.indexer_scheme,
            allow_stale_fetch=args.allow_stale_fetch,
            required_pair_id=required_pair_id,
            required_asset_x=args.asset_x,
            required_asset_y=args.asset_y,
        )
        print(frame.to_string(index=False))
        print(f"dydx_long_history_fetch: {ROOT / 'reports' / 'dydx_long_history_fetch.csv'}")
    elif args.command == "run-dydx-long-history":
        if not args.asset_x or not args.asset_y:
            raise SystemExit("run-dydx-long-history requires --asset-x and --asset-y")
        resolved_research_funding_path = _resolve_research_funding_path(
            args.research_funding_path, args.funding_path
        )
        paths = run_dydx_long_history(
            pair=args.pair,
            asset_x=args.asset_x,
            asset_y=args.asset_y,
            pair_id=args.pair_id,
            windows=args.windows,
            limit=args.limit,
            resolution=args.interval or "5MINS",
            indexer_base=args.indexer_base,
            indexer_scheme=args.indexer_scheme,
            to_iso=args.to_iso,
            derive_hedge_ratio=args.derive_hedge_ratio,
            run_research=args.run_research,
            funding_path=resolved_research_funding_path,
            allow_stale_fetch=args.allow_stale_fetch,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
    elif args.command == "run-dydx-pair-expansion":
        if not args.allow_blocked_exploration:
            _assert_ready_for_exploration()
        requested_pairs = _parse_pair_list(args.pair_ids if args.pair_ids else args.pair)
        requested_strategies = None
        if args.strategy_ids.strip():
            requested_strategies = tuple(
                int(x) for x in args.strategy_ids.split(",") if x.strip().isdigit()
            )
        print_run_dydx_pair_expansion(
            max_pairs=args.max_pairs,
            limit=args.limit,
            indexer_base=args.indexer_base,
            indexer_scheme=args.indexer_scheme,
            output_path=args.output_path,
            run_research=args.run_research,
            skip_fetch=args.skip_fetch,
            allow_stale_fetch=args.allow_stale_fetch,
            pair_ids=requested_pairs,
            strategy_ids=requested_strategies,
        )
    elif args.command == "backfill-dydx-pair-history-features":
        input_dir = args.input_dir or ROOT / "data" / "raw" / "pair_details"
        written = backfill_provisional_pair_history_features(input_dir)
        print(pd.DataFrame({"path": [str(path) for path in written]}).to_string(index=False))
        print(f"backfilled_pair_histories: {len(written)}")
    elif args.command == "strategy-family-sweep":
        print_strategy_family_sweep(
            input_dir=args.input_dir,
            funding_path=args.funding_path,
            output_dir=args.output_dir or args.output_path,
            pair_list=_parse_pair_list(args.pair),
        )
    elif args.command == "strategy-family-matrix":
        print_strategy_family_matrix(
            input_dir=args.input_dir,
            funding_path=args.funding_path,
            output_dir=args.output_dir or args.output_path,
            pair_list=_parse_pair_list(args.pair),
            max_combo_size=args.max_combo_size,
        )
    elif args.command == "research-quantization":
        print_research_quantization(
            family_matrix_dir=args.input_dir,
            output_dir=args.output_dir or args.output_path,
            top_n=args.top_n,
        )
    elif args.command == "strategy-family-sweep-failure-attribution":
        print_strategy_family_failure_attribution(
            sweep_dir=args.input_dir,
            output_path=args.output_path,
        )
    elif args.command == "priority-readiness":
        print_priority_readiness()
    elif args.command == "priority-actions":
        print_priority_actions()
    elif args.command == "priority-dashboard":
        print_priority_dashboard()
    elif args.command == "priority-runbook":
        print_priority_runbook()
    elif args.command == "paper-execution-preflight":
        print_paper_execution_preflight()
    elif args.command == "paper-readiness-checkpoint":
        result = paper_readiness_checkpoint(readiness_threshold=args.readiness_threshold)
        print(json.dumps(result, indent=2))
    elif args.command == "research-sweep":
        result = run_research_sweep(
            mode=args.sweep_mode,
            root=ROOT,
            mcp_url=args.mcp_url or os.getenv("APIFY_MCP_SERVER_URL", "").strip() or None,
            api_token=args.apify_token,
            source_filter=args.source_filter,
            wait_seconds=args.wait_seconds,
            do_fetch=not args.no_fetch,
            apify_actor_credit_ceiling=args.apify_actor_credit_ceiling,
            readiness_threshold=args.readiness_threshold,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "paper-venue-preflight":
        print_paper_venue_preflight(pair=args.pair, max_pairs=args.max_pairs)
    elif args.command == "gap-test":
        print_gap_test()
    elif args.command == "gap-analysis-checklist":
        print_gap_analysis_checklist(args.output_path)
    elif args.command == "pre-mortem-checklist":
        print_pre_mortem_checklist(args.output_path)
    elif args.command in {"post-mortem-checklist", "postmortem-checklist"}:
        print_post_mortem_checklist(args.output_path)
    elif args.command in {"supreme-team", "supreme-team-checklist"}:
        print_supreme_team_checkpoint()
    elif args.command in {"red-team-checklist", "redteam-checklist"}:
        print_red_team_checklist(args.output_path)
    elif args.command == "learning-report":
        write_learning_report()
    elif args.command == "build-ml-trade-filter-dataset":
        print_build_ml_trade_filter_dataset(
            input_dir=args.input_dir,
            funding_path=args.funding_path,
            output_path=args.output_path,
        )
    elif args.command == "train-ml-trade-filter":
        print_train_ml_trade_filter(
            input_dir=args.input_dir,
            funding_path=args.funding_path,
            output_dir=args.output_dir,
            walkforward_splits=args.walkforward_splits,
            min_train_rows=args.min_train_rows,
        )
    elif args.command == "shadow-ml-trade-filter":
        print_shadow_ml_trade_filter(
            input_dir=args.input_dir,
            funding_path=args.funding_path,
            model_path=args.model_path,
            output_path=args.output_path,
            model_sha256=args.model_sha256,
        )
    elif args.command == "compare-ml-shadow-models":
        print_compare_ml_shadow_models(
            input_dir=args.input_dir,
            output_dir=args.output_dir or args.output_path,
            pair_list=_parse_pair_list(args.pair),
        )
    elif args.command == "trade-timing-template":
        print_trade_timing_template(args.output_path)
    elif args.command == "trade-timing-comparison-report":
        print_trade_timing_comparison_report(
            trades_path=args.input_dir,
            history_path=args.history_path,
            output_path=args.output_path,
            entry_threshold=args.entry_threshold,
            exit_threshold=args.exit_threshold,
        )
    elif args.command == "learning-outcome-template":
        print_learning_outcome_template(args.output_path)
    elif args.command == "seed-learning-outcome-template":
        print_seed_learning_outcome_template_from_paper_journal(args.input_dir, args.output_path)
    elif args.command == "learning-outcome-template-check":
        print_learning_outcome_template_check(args.input_dir, args.output_path)
    elif args.command == "import-learning-outcomes":
        print_import_learning_outcomes(args.input_dir, args.output_path)
    elif args.command == "append-learning-outcome":
        run_append_learning_outcome(
            pair=args.pair,
            strategy_id=args.strategy_id,
            realized_return=args.realized_return,
            signal=args.signal,
            hedge_ratio=args.hedge_ratio,
            beta=args.beta,
            notional_usd=args.notional_usd,
            regime=args.regime,
            trade_id=args.trade_id,
            output_path=args.output_path,
        )
    elif args.command == "paper-watch":
        print_current_paper_watch(journal_path=args.journal_path, output_path=args.output_path)
    elif args.command == "research-spine":
        print_research_spine(
            args.input_dir,
            require_two_leg=not args.allow_spread_only,
            funding_path=args.funding_path,
        )
    elif args.command == "crawl-crypto-wizards":
        crawl_crypto_wizards(args.endpoint)
    elif args.command == "crypto-wizards-full-sweep":
        sweep_kwargs = {
            "root": ROOT,
            "api_key": None,
            "exchanges": tuple(
                value.strip() for value in args.wizard_exchanges.split(",") if value.strip()
            ),
            "intervals": tuple(
                value.strip() for value in args.wizard_intervals.split(",") if value.strip()
            ),
            "strategies": tuple(
                value.strip() for value in args.wizard_strategies.split(",") if value.strip()
            ),
            "priorities": tuple(
                value.strip() for value in args.wizard_priorities.split(",") if value.strip()
            ),
            "daily_credit_limit": args.wizard_daily_credit_limit,
            "reserved_credits": args.wizard_reserved_credits,
            "credit_lane": args.wizard_credit_lane,
            "publish_active": not args.wizard_attempt_only,
        }
        if args.execute_wizard_sweep:
            result = run_authorized_wizard_discovery_sweep(**sweep_kwargs)
        else:
            result = run_wizard_discovery_sweep(execute=False, **sweep_kwargs)
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "restore-wizard-sweep-from-raw":
        result = restore_complete_wizard_sweep_from_raw(
            root=ROOT,
            sweep_id=args.run_id,
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "wizard-control-plane":
        result = build_wizard_control_plane(
            root=ROOT,
            max_age_hours=args.max_wizard_age_hours,
        )
        print(
            json.dumps(
                {
                    "summary": result.summary,
                    "paths": {key: str(value) for key, value in result.paths.items()},
                },
                indent=2,
            )
        )
    elif args.command == "crawl-crypto-wizards-min5":
        crawl_crypto_wizards_min5(
            max_pairs=args.max_pairs,
            priority=args.priority,
            cw_strategy=args.cw_strategy,
            exchange=args.exchange,
            period=args.period,
            spread_type=args.spread_type,
            roll_w=args.roll_w,
            asset=args.asset,
            run_research=args.run_research,
            output_dir=args.output_path,
        )
    elif args.command == "crawl-crypto-wizards-min5-backtest":
        crawl_crypto_wizards_min5_backtests(
            max_pairs=args.max_pairs,
            priority=args.priority,
            cw_strategy=args.cw_strategy,
            exchange=args.exchange,
            period=args.period,
            spread_type=args.spread_type,
            roll_w=args.roll_w,
            asset=args.asset,
            run_research=args.run_research,
            output_dir=args.output_path,
        )
    elif args.command == "paper-plan":
        if args.pair is None or args.strategy_id is None or args.signal is None:
            raise SystemExit("paper-plan requires --pair, --strategy-id, and --signal")
        run_paper_plan(
            pair=args.pair,
            strategy_id=args.strategy_id,
            signal=args.signal,
            hedge_ratio=args.hedge_ratio,
            beta=args.beta,
            notional_usd=args.notional_usd,
            acceptance_path=args.acceptance_path,
            journal_path=args.journal_path,
            venue=args.venue,
            order_approval_id=args.order_approval_id,
        )
    elif args.command == "close-paper-trade":
        if args.realized_return is None:
            raise SystemExit("close-paper-trade requires --realized-return")
        if args.trade_id is None and args.pair is None:
            raise SystemExit("close-paper-trade requires --trade-id or --pair")
        run_close_paper_trade(
            trade_id=args.trade_id,
            pair=args.pair,
            strategy_id=args.strategy_id,
            realized_return=args.realized_return,
            journal_path=args.journal_path,
            output_path=args.output_path,
        )


def _resolve_paper_venue(pair: str, requested_venue: str = "auto") -> str:
    requested = (requested_venue or "").lower().strip()
    if requested and requested != "auto":
        return requested

    options = _build_paper_venue_options(pair)
    for option in options:
        if bool(option.get("executable", False)):
            return str(option.get("venue", "dydx"))
    if options:
        return str(options[0].get("venue", "dydx"))

    target = (pair or "").replace("/", "-").upper()
    universe = _read_csv_or_empty(ROOT / "data" / "processed" / "pair_universe.csv")
    if (
        not universe.empty
        and "pair" in universe.columns
        and "best_execution_venue" in universe.columns
    ):
        universe_pairs = universe[universe["pair"].astype(str).str.upper() == target]
        if universe_pairs.empty:
            alt = target.replace("-", "/")
            universe_pairs = universe[universe["pair"].astype(str).str.upper() == alt]
        if not universe_pairs.empty:
            best = universe_pairs.iloc[0]
            venue = (
                str(best.get("best_execution_venue", "") or best.get("exchange", "") or "")
                .strip()
                .lower()
            )
            if venue:
                return venue
            if bool(best.get("dydx_tradable", False)):
                return "dydx"
            if str(best.get("decision_bucket", "")).upper() == "PROMOTE":
                return "dydx"
    return "dydx"


if __name__ == "__main__":
    main()
