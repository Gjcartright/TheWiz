from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import shutil
from typing import Iterable

import numpy as np
import pandas as pd

from quant_platform.experiments import PairDataset
from quant_platform.ml_filter import (
    NON_FEATURE_COLUMNS,
    RETURN_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    build_trade_filter_dataset,
    train_trade_filter_walkforward,
)
from quant_platform.pair_detail_ingestion import datasets_from_pair_detail_snapshots
from quant_platform.regimes import RegimeConfig, classify_regimes
from quant_platform.wizard_symbols import normalize_wizard_exchange, normalize_wizard_symbol, wizard_exchange_lane


ROOT = Path(__file__).resolve().parents[2]
ACTIVE = ROOT / "reports" / "active"
DASHBOARD = ROOT / "reports" / "dashboard"
ML_REPORTS = ROOT / "reports" / "ml"
DATA_ML = ROOT / "data" / "ml"
MODELS = ROOT / "models" / "trade_gate"

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

PAIR_UNIVERSE_COLUMNS = [
    "pair",
    "asset_x",
    "asset_y",
    "exchange",
    "dydx_tradable",
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
    "best_wizard_sharpe",
    "best_wizard_returns_total",
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
    "PYTHONPATH=src python -m quant_platform.cli build-wizard-research-pack",
    "PYTHONPATH=src python -m quant_platform.cli build-market-venue-context",
    "PYTHONPATH=src python -m quant_platform.cli build-venue-lane-test-plan",
    "PYTHONPATH=src python -m quant_platform.cli build-multi-venue-history-readiness",
    "PYTHONPATH=src python -m quant_platform.cli hyperliquid-lane-readiness",
    "PYTHONPATH=src python -m quant_platform.cli build-pair-universe",
    "PYTHONPATH=src python -m quant_platform.cli build-trade-dataset",
    "PYTHONPATH=src python -m quant_platform.cli train-trade-gate",
    "PYTHONPATH=src python -m quant_platform.cli run-model-gated-backtest",
    "PYTHONPATH=src python -m quant_platform.cli export-trade-gate-model",
    "PYTHONPATH=src python -m quant_platform.cli build-command-dashboard",
    "PYTHONPATH=src python -m quant_platform.cli archive-from-index --dry-run",
]


@dataclass(frozen=True)
class CommandResult:
    paths: dict[str, Path]
    summary: dict[str, object]


def build_artifact_index(root: Path = ROOT) -> CommandResult:
    rows = [_artifact_row(path, root) for path in _iter_repo_files(root)]
    frame = pd.DataFrame(rows, columns=ARTIFACT_COLUMNS).sort_values("path").reset_index(drop=True)
    csv_path = ACTIVE / "artifact_index.csv"
    md_path = ACTIVE / "artifact_index.md"
    commands_path = ACTIVE / "canonical_commands.md"
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
    rows = [
        _state_row(
            "active_layer",
            _exists(ACTIVE / "artifact_index.csv"),
            "artifact index exists" if (ACTIVE / "artifact_index.csv").exists() else "artifact index missing",
            ACTIVE / "artifact_index.csv",
            "run build-artifact-index",
        ),
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
            _exists(ACTIVE / "venue_lane_test_plan.csv"),
            "venue lane test plan exists" if (ACTIVE / "venue_lane_test_plan.csv").exists() else "venue lane test plan missing",
            ACTIVE / "venue_lane_test_plan.csv",
            "run build-venue-lane-test-plan",
        ),
        _state_row(
            "hyperliquid_lane",
            _exists(ACTIVE / "hyperliquid_lane_readiness.csv"),
            "hyperliquid lane readiness exists"
            if (ACTIVE / "hyperliquid_lane_readiness.csv").exists()
            else "hyperliquid lane readiness missing",
            ACTIVE / "hyperliquid_lane_readiness.csv",
            "run hyperliquid-lane-readiness",
        ),
        _state_row(
            "strategy_acceptance",
            _acceptance_ready(root),
            _acceptance_status(root),
            root / "reports" / "strategy_acceptance_checklist.csv",
            "review acceptance blockers before paper trading",
        ),
        _state_row(
            "trade_dataset",
            _exists(DATA_ML / "trade_training_dataset.csv"),
            "trade dataset exists" if (DATA_ML / "trade_training_dataset.csv").exists() else "trade dataset missing",
            DATA_ML / "trade_training_dataset.csv",
            "run build-trade-dataset",
        ),
        _state_row(
            "trade_gate_model",
            _exists(MODELS / "model.pkl"),
            "model artifact exists" if (MODELS / "model.pkl").exists() else "model artifact missing",
            MODELS / "metrics.json",
            "run train-trade-gate after dataset is ready",
        ),
        _state_row(
            "dashboard",
            _exists(DASHBOARD / "command_center.md"),
            "command dashboard exists" if (DASHBOARD / "command_center.md").exists() else "command dashboard missing",
            DASHBOARD / "command_center.md",
            "run build-command-dashboard",
        ),
    ]
    frame = pd.DataFrame(rows)
    csv_path = ACTIVE / "current_state.csv"
    md_path = ACTIVE / "current_state.md"
    _write_csv(frame, csv_path)
    _write_text(md_path, _current_state_markdown(frame))
    return CommandResult(paths={"current_state": csv_path, "current_state_md": md_path}, summary={"rows": len(frame)})


def system_check(root: Path = ROOT) -> CommandResult:
    rows = []
    for folder in ["data", "data/raw", "data/processed", "reports", "src/quant_platform"]:
        path = root / folder
        rows.append(_check_row(f"folder:{folder}", path.exists(), str(path), "create folder or restore repo state"))
    for key in [
        "CRYPTO_WIZARDS_API_KEY",
        "CRYPTO_WIZARDS_BASE_URL",
        "APIFY_API_TOKEN",
        "APIFY_MCP_SERVER_URL",
        "DYDX_TESTNET_WALLET_ADDRESS",
        "DYDX_TESTNET_PRIVATE_KEY",
        "DYDX_TESTNET_SUBMIT_ORDERS",
    ]:
        present = bool(os.getenv(key, "").strip())
        required = key in {"CRYPTO_WIZARDS_BASE_URL", "DYDX_TESTNET_SUBMIT_ORDERS"}
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
    for artifact in [
        ACTIVE / "artifact_index.csv",
        root / "data" / "processed" / "pair_universe.csv",
        root / "data" / "processed" / "market_venue_context.csv",
        ACTIVE / "venue_lane_test_plan.csv",
        ACTIVE / "hyperliquid_lane_readiness.csv",
        DATA_ML / "trade_training_dataset.csv",
        MODELS / "model.pkl",
        DASHBOARD / "command_center.md",
    ]:
        rows.append(_check_row(f"artifact:{artifact.name}", artifact.exists(), str(artifact), "run the corresponding active pipeline command"))
    frame = pd.DataFrame(rows)
    csv_path = ACTIVE / "system_check.csv"
    md_path = ACTIVE / "system_check.md"
    _write_csv(frame, csv_path)
    _write_text(md_path, _simple_report_markdown("System Check", frame))
    return CommandResult(
        paths={"system_check": csv_path, "system_check_md": md_path},
        summary={"checks": len(frame), "ready": int(frame["ready"].astype(bool).sum()), "blocked": int((~frame["ready"].astype(bool)).sum())},
    )


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
    liquidity = _read_csv(ACTIVE / "multi_exchange_liquidity_test_2026-06-25.csv")
    source_coverage = _read_csv(ACTIVE / "apify_mcp_source_coverage_2026-06-25.csv")
    rows = _market_venue_rows_from_liquidity(liquidity)
    rows.extend(_planned_source_rows(source_coverage))
    frame = pd.DataFrame(rows, columns=MARKET_VENUE_CONTEXT_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["asset", "venue", "source_system"]).reset_index(drop=True)
    output = root / "data" / "processed" / "market_venue_context.csv"
    snapshot = root / "data" / "processed" / "market_venue_context_snapshots" / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M')}.csv"
    summary = ACTIVE / "market_venue_context_summary.csv"
    summary_md = ACTIVE / "market_venue_context_summary.md"
    lanes = ACTIVE / "venue_lane_classification.csv"
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


def build_venue_lane_test_plan(root: Path = ROOT) -> CommandResult:
    lanes = _read_csv(ACTIVE / "venue_lane_classification.csv")
    verification = _read_csv(ACTIVE / "wizard_local_verification_batch.csv")
    queue = _read_csv(ACTIVE / "crypto_wizards_next_best_sharpe_returns_queue.csv")
    rows = _venue_lane_test_rows(lanes, verification, queue)
    frame = pd.DataFrame(rows, columns=VENUE_LANE_TEST_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["test_status", "pair_lane", "wizard_sharpe"], ascending=[True, True, False]).reset_index(drop=True)
    csv_path = ACTIVE / "venue_lane_test_plan.csv"
    md_path = ACTIVE / "venue_lane_test_plan.md"
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
    rows_path = ACTIVE / "crypto_wizards_multi_venue_sharpe_rows_2026-06-25.csv"
    rows = _read_csv(rows_path)
    readiness_rows = _multi_venue_history_readiness_rows(rows, rows_path, top_n=top_n)
    frame = pd.DataFrame(readiness_rows, columns=MULTI_VENUE_HISTORY_READINESS_COLUMNS)
    csv_path = ACTIVE / "multi_venue_history_readiness_2026-06-25.csv"
    md_path = ACTIVE / "multi_venue_history_readiness_2026-06-25.md"
    _write_csv(frame, csv_path)
    _write_text(md_path, _multi_venue_history_readiness_markdown(frame))
    return CommandResult(
        paths={"multi_venue_history_readiness": csv_path, "multi_venue_history_readiness_md": md_path},
        summary={
            "rows": len(frame),
            "ready_to_fetch": int(frame["readiness_status"].astype(str).eq("ready_to_fetch").sum()) if not frame.empty else 0,
            "ready_for_replay": int(frame["readiness_status"].astype(str).eq("ready_for_replay").sum()) if not frame.empty else 0,
            "blocked": int(frame["readiness_status"].astype(str).str.contains("blocked", na=False).sum()) if not frame.empty else 0,
        },
    )


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


def _venue_lane_test_rows(lanes: pd.DataFrame, verification: pd.DataFrame, queue: pd.DataFrame) -> list[dict[str, object]]:
    lane_map = {
        str(row.get("asset", "")).upper(): row.to_dict()
        for _, row in lanes.iterrows()
        if str(row.get("asset", "") or "").strip()
    }
    candidates = _venue_lane_candidates(verification, queue)
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
                "dydx_blockers": _asset_blockers(left, right, "dydx"),
                "hyperliquid_blockers": _asset_blockers(left, right, "hyperliquid"),
                "funding_pulse_status": "needs_api_key",
                "next_step": next_step,
                "evidence_path": _candidate_evidence_path(candidate),
            }
        )
    return rows


def _venue_lane_candidates(verification: pd.DataFrame, queue: pd.DataFrame) -> list[dict[str, object]]:
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
    return candidates


def _candidate_dedupe_key(candidate: dict[str, object]) -> str:
    exchange = _candidate_wizard_exchange(candidate) or "unknown"
    pair = str(candidate.get("normalized_pair", "") or candidate.get("pair", "") or "").upper()
    if not pair:
        asset_x = str(candidate.get("asset_x", "") or "").upper()
        asset_y = str(candidate.get("asset_y", "") or "").upper()
        pair = f"{asset_x}/{asset_y}"
    return f"{exchange}:{pair}"


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


def _market_venue_rows_from_liquidity(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
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
                "source_timestamp": _now(),
                "source_system": source_system,
                "source_status": "sampled",
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
    if frame.empty:
        return pd.DataFrame(columns=["asset", "best_lane", "dydx_lane", "hyperliquid_lane", "blockers", "next_action"])
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
    return pd.DataFrame(rows).sort_values(["best_lane", "asset"]).reset_index(drop=True)


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
    pair_rows = []
    experiment = _read_csv(root / "reports" / "experiment_results.csv")
    acceptance = _read_csv(root / "reports" / "acceptance_report.csv")
    funding = _read_csv(root / "data" / "processed" / "dydx_funding.csv")
    wizard = _wizard_research_tables(root)
    pairs = _candidate_pairs(root)
    candle_index = _dydx_candle_index(root)
    market_snapshot = _latest_apify_markets(root)
    for pair, assets, evidence_paths in pairs:
        asset_x, asset_y = assets
        metrics = _local_pair_metrics(pair, experiment, acceptance)
        wizard_metrics = _wizard_pair_metrics(pair, wizard)
        metrics.update(wizard_metrics)
        _apply_market_snapshot_metrics(metrics, asset_x, asset_y, market_snapshot)
        funding_drag = _funding_drag(asset_x, asset_y, funding, market_snapshot)
        timeframes = sorted(set(candle_index.get(asset_x, set())) & set(candle_index.get(asset_y, set())))
        dydx_tradable = bool(candle_index.get(asset_x) and candle_index.get(asset_y)) or (
            asset_x in market_snapshot and asset_y in market_snapshot
        )
        discovery_components = _discovery_components(dydx_tradable, timeframes, metrics, funding_drag)
        acceptance_components = _acceptance_components(dydx_tradable, metrics, funding_drag)
        discovery_score = round(sum(discovery_components.values()), 3)
        acceptance_score = round(sum(acceptance_components.values()), 3)
        combined_score = round(discovery_score + acceptance_score, 3)
        bucket, reason = _decision_bucket(discovery_score, acceptance_score, metrics, dydx_tradable, timeframes)
        missing = _missing_pair_data(dydx_tradable, timeframes, metrics, funding_drag)
        pair_rows.append(
            {
                "pair": pair,
                "asset_x": asset_x,
                "asset_y": asset_y,
                "exchange": "dydx",
                "dydx_tradable": dydx_tradable,
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
                "source_timestamp": _now(),
                "field_freshness": "current_snapshot",
                "stale_reason": "" if not missing else "missing_or_partial_inputs",
                "evidence_path": ";".join(sorted(evidence_paths | _market_evidence_paths(asset_x, asset_y, market_snapshot))),
                "best_wizard_exact_mode": metrics.get("best_wizard_exact_mode", ""),
                "best_wizard_spread_id": metrics.get("best_wizard_spread_id", ""),
                "best_wizard_strategy_id": metrics.get("best_wizard_strategy_id", ""),
                "best_wizard_sharpe": metrics.get("best_wizard_sharpe", ""),
                "best_wizard_returns_total": metrics.get("best_wizard_returns_total", ""),
                "wizard_diagnostic_score": metrics.get("wizard_diagnostic_score", ""),
                "wizard_hypothesis_status": metrics.get("wizard_hypothesis_status", ""),
                "local_mode_confirmation_status": metrics.get("local_mode_confirmation_status", ""),
                "wizard_local_parity_status": metrics.get("wizard_local_parity_status", ""),
                "promotion_blocker": _promotion_blocker(bucket, metrics),
            }
        )
    frame = pd.DataFrame(pair_rows, columns=PAIR_UNIVERSE_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["decision_bucket", "acceptance_score", "discovery_score"], ascending=[True, False, False])
    output = root / "data" / "processed" / "pair_universe.csv"
    snapshot = root / "data" / "processed" / "pair_universe_snapshots" / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M')}.csv"
    summary = ACTIVE / "pair_universe_summary.csv"
    summary_md = ACTIVE / "pair_universe_summary.md"
    components = ACTIVE / "pair_score_components.csv"
    decisions = ACTIVE / "pair_decision_buckets.csv"
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
    source = input_dir or root / "data" / "raw" / "pair_details"
    datasets = datasets_from_pair_detail_snapshots(source, require_research_usable=True)
    datasets = [
        PairDataset(dataset.pair, classify_regimes(_ensure_trade_dataset_inputs(dataset.frame), RegimeConfig(preserve_existing=True)))
        for dataset in datasets
    ]
    frame = build_trade_filter_dataset(datasets)
    if frame.empty:
        raise SystemExit(f"no leakage-safe trade rows could be built from {source}")
    hardened = _harden_trade_dataset(frame)
    audit = _leakage_audit(hardened)
    if audit["uses_future_data"].astype(bool).any():
        blockers = audit.loc[audit["uses_future_data"].astype(bool), "trade_id"].head(5).tolist()
        raise SystemExit(f"leakage audit failed for trade rows: {blockers}")
    DATA_ML.mkdir(parents=True, exist_ok=True)
    ML_REPORTS.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_ML / "trade_training_dataset.csv"
    parquet_path = DATA_ML / "trade_training_dataset.parquet"
    summary_path = ML_REPORTS / "trade_dataset_summary.csv"
    audit_path = ML_REPORTS / "leakage_audit.csv"
    _write_csv(hardened, csv_path)
    parquet_status = _write_parquet_if_available(hardened, parquet_path)
    _write_csv(audit, audit_path)
    _write_csv(_trade_dataset_summary(hardened, audit, parquet_status), summary_path)
    return CommandResult(
        paths={"dataset_csv": csv_path, "dataset_parquet": parquet_path, "summary": summary_path, "leakage_audit": audit_path},
        summary={"rows": len(hardened), "parquet_status": parquet_status},
    )


def train_trade_gate(root: Path = ROOT, input_path: Path | None = None, walkforward_splits: int = 5, min_train_rows: int = 100) -> CommandResult:
    source = input_path or DATA_ML / "trade_training_dataset.csv"
    dataset = _model_dataset_from_hardened(_read_csv(source))
    if dataset.empty:
        raise SystemExit(f"trade gate dataset is empty: {source}")
    study_dir = ML_REPORTS / "trade_gate"
    paths = train_trade_filter_walkforward(dataset, output_dir=study_dir, n_splits=walkforward_splits, min_train_rows=min_train_rows)
    MODELS.mkdir(parents=True, exist_ok=True)
    model_path = MODELS / "model.pkl"
    shutil.copyfile(paths["best_model"], model_path)
    summary = _read_csv(paths["summary"])
    predictions = _read_csv(paths["predictions"])
    metrics = _trade_gate_metrics(summary, predictions)
    feature_schema = _feature_schema(dataset)
    acceptance = _model_acceptance_frame(metrics)
    bucket_report = _score_bucket_report(predictions)
    concentration = _model_pair_concentration(predictions)
    _write_json(MODELS / "feature_schema.json", feature_schema)
    _write_json(MODELS / "metrics.json", metrics)
    _write_csv(summary, ML_REPORTS / "model_backtest_comparison.csv")
    _write_csv(predictions, ML_REPORTS / "model_walkforward_predictions.csv")
    _write_csv(bucket_report, ML_REPORTS / "score_bucket_report.csv")
    _write_csv(concentration, ML_REPORTS / "model_pair_concentration.csv")
    _write_csv(acceptance, ML_REPORTS / "model_gated_acceptance.csv")
    return CommandResult(
        paths={"model": model_path, "feature_schema": MODELS / "feature_schema.json", "metrics": MODELS / "metrics.json"},
        summary={"accepted": bool(metrics["accepted"]), "best_model": metrics.get("best_model", "")},
    )


def run_model_gated_backtest(root: Path = ROOT) -> CommandResult:
    predictions = _read_csv(ML_REPORTS / "model_walkforward_predictions.csv")
    if predictions.empty:
        raise SystemExit("model walk-forward predictions missing; run train-trade-gate first")
    comparison = _model_gated_comparison(predictions)
    acceptance = _model_gated_acceptance(comparison)
    failures = _model_failure_attribution(predictions, acceptance)
    _write_csv(comparison, ML_REPORTS / "model_gated_backtest.csv")
    _write_csv(acceptance, ML_REPORTS / "model_gated_acceptance.csv")
    _write_csv(failures, ML_REPORTS / "model_failure_attribution.csv")
    return CommandResult(
        paths={"backtest": ML_REPORTS / "model_gated_backtest.csv", "acceptance": ML_REPORTS / "model_gated_acceptance.csv", "failures": ML_REPORTS / "model_failure_attribution.csv"},
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


def build_command_dashboard(root: Path = ROOT) -> CommandResult:
    DASHBOARD.mkdir(parents=True, exist_ok=True)
    pair_universe = _read_csv(root / "data" / "processed" / "pair_universe.csv")
    current = _read_csv(ACTIVE / "current_state.csv")
    model_acceptance = _read_csv(ML_REPORTS / "model_gated_acceptance.csv")
    system = _read_csv(ACTIVE / "system_check.csv")
    live = _live_signal_rows(pair_universe, model_acceptance)
    blocked = live[live["blocker"].astype(str) != ""].copy() if not live.empty else pd.DataFrame()
    data_health = _data_health_rows(system, pair_universe)
    paths = {
        "pair_universe": DASHBOARD / "pair_universe_dashboard.csv",
        "candidate_ranking": DASHBOARD / "candidate_ranking_dashboard.csv",
        "strategy_tests": DASHBOARD / "strategy_tests_dashboard.csv",
        "wizard_local_verification": DASHBOARD / "wizard_local_verification_dashboard.csv",
        "model_training": DASHBOARD / "model_training_dashboard.csv",
        "orchestrator_run_status": DASHBOARD / "orchestrator_run_status.csv",
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
        "api_credit_usage": DASHBOARD / "api_credit_usage_dashboard.csv",
        "command_center": DASHBOARD / "command_center.md",
        "scoring_audit": DASHBOARD / "scoring_audit.csv",
    }
    _write_csv(pair_universe, paths["pair_universe"])
    _write_csv(pair_universe.sort_values("combined_score", ascending=False) if "combined_score" in pair_universe else pair_universe, paths["candidate_ranking"])
    _write_csv(_strategy_tests_dashboard(root), paths["strategy_tests"])
    batch_verification = root / "reports" / "active" / "wizard_local_verification_batch.csv"
    single_verification = root / "reports" / "active" / "bnb_stx_daily_320_static_spread_after_cost.csv"
    _write_csv(_read_csv(batch_verification if batch_verification.exists() else single_verification), paths["wizard_local_verification"])
    _write_csv(_model_training_dashboard(model_acceptance), paths["model_training"])
    _write_csv(_read_csv(root / "reports" / "active" / "orchestrator_run_status.csv"), paths["orchestrator_run_status"])
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
    _write_csv(_api_credit_usage_rows(), paths["api_credit_usage"])
    _write_csv(live, paths["scoring_audit"])
    _write_text(paths["command_center"], _command_center_markdown(pair_universe, current, model_acceptance, data_health))
    return CommandResult(paths=paths, summary={"dashboard_files": len(paths), "blocked_rows": len(blocked)})


def archive_from_index(dry_run: bool = True) -> CommandResult:
    index = _read_csv(ACTIVE / "artifact_index.csv")
    if index.empty:
        raise SystemExit("artifact index missing; run build-artifact-index first")
    candidates = index[index["safe_to_archive_later"].astype(bool)].copy()
    candidates["planned_action"] = "would_archive" if dry_run else "blocked_by_policy"
    candidates["archive_path"] = candidates["path"].map(lambda p: f"archive/pending/{p}")
    out = ROOT / "archive" / "archive_manifest.csv"
    _write_csv(candidates, out)
    if not dry_run:
        raise SystemExit("archive apply is intentionally blocked until active lineage is reviewed")
    return CommandResult(paths={"archive_manifest": out}, summary={"dry_run": dry_run, "candidates": len(candidates)})


def _iter_repo_files(root: Path) -> Iterable[Path]:
    skipped = {".git", "__pycache__"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in skipped for part in rel_parts):
            continue
        yield path


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
    if rel in {".env.local", ".env.example", "pyproject.toml", "README.md", "project_objective.md", "memory.md"}:
        return "do_not_move"
    if rel.startswith(("src/", "tests/", "config/", "scripts/", "docs/")):
        return "active"
    if rel.startswith(("reports/active/", "reports/dashboard/", "reports/ml/", "data/ml/")):
        return "active"
    if rel.startswith(("data/raw/", "data/processed/", "data/meta_learning/", "reports/")):
        return "historical_evidence"
    if rel.startswith(("work/", "outputs/")) or rel.endswith((".log", ".tmp")):
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
    ranked["_rank_mode_valid"] = ranked.get("mode_valid", pd.Series(False, index=ranked.index)).astype(bool).astype(int)
    ranked["_rank_sharpe"] = _numeric(ranked.get("sharpe", pd.Series(dtype=float))).fillna(-999)
    ranked["_rank_return"] = _numeric(ranked.get("returns_total", pd.Series(dtype=float))).fillna(-999)
    if "passes_sharpe_gate" in ranked:
        ranked["_rank_sharpe_gate"] = ranked["passes_sharpe_gate"].astype(bool)
    else:
        ranked["_rank_sharpe_gate"] = ranked["_rank_sharpe"].ge(1.75)
    ranked = ranked.sort_values(["_rank_mode_valid", "_rank_sharpe_gate", "passes_returns_total_gt_20pct", "_rank_sharpe", "_rank_return"], ascending=[False, False, False, False, False])
    best = ranked.iloc[0]
    metrics.update(
        {
            "best_wizard_exact_mode": best.get("exact_mode", ""),
            "best_wizard_spread_id": best.get("spread_id", ""),
            "best_wizard_strategy_id": best.get("strategy_id", ""),
            "best_wizard_sharpe": best.get("sharpe", ""),
            "best_wizard_returns_total": best.get("returns_total", ""),
            "zscore_score": float(np.clip(float(_numeric(pd.Series([best.get("sharpe", 0)])).fillna(0).iloc[0]) * 3, 0.0, 15.0)),
        }
    )
    diagnostics = _wizard_pair_subset(tables.get("diagnostics", pd.DataFrame()), pair)
    if not diagnostics.empty and "wizard_diagnostic_score" in diagnostics:
        metrics["wizard_diagnostic_score"] = float(_numeric(diagnostics["wizard_diagnostic_score"]).max())
        metrics["cointegration_score"] = min(float(metrics["wizard_diagnostic_score"]) / 3, 20.0)
    hypotheses = _wizard_pair_subset(tables.get("hypotheses", pd.DataFrame()), pair)
    if not hypotheses.empty:
        ranked_hypotheses = hypotheses.copy()
        ranked_hypotheses["_rank_exact_mode"] = ranked_hypotheses.get("exact_mode", pd.Series("", index=ranked_hypotheses.index)).astype(str).str.strip().ne("").astype(int)
        ranked_hypotheses["_rank_ready"] = ranked_hypotheses.get("hypothesis_status", pd.Series("", index=ranked_hypotheses.index)).astype(str).eq("HYPOTHESIS_READY").astype(int)
        ranked_hypotheses = ranked_hypotheses.sort_values(["_rank_ready", "_rank_exact_mode"], ascending=[False, False])
        metrics["wizard_hypothesis_status"] = str(ranked_hypotheses.iloc[0].get("hypothesis_status", ""))
    parity = _wizard_pair_subset(tables.get("parity", pd.DataFrame()), pair)
    if not parity.empty:
        status = str(parity.iloc[0].get("parity_status", ""))
        metrics["wizard_local_parity_status"] = status
        metrics["local_mode_confirmation_status"] = "confirmed" if status in {"MATCH", "CLOSE"} else "unconfirmed"
    return metrics


def _wizard_pair_subset(frame: pd.DataFrame, pair: str) -> pd.DataFrame:
    if frame.empty or "pair" not in frame:
        return pd.DataFrame()
    target_assets = _assets_from_pair(pair)
    if not target_assets:
        return frame[frame["pair"].astype(str).str.upper() == pair.upper()]
    target = f"{target_assets[0]}/{target_assets[1]}".upper()
    alt = f"{target_assets[0]}-{target_assets[1]}".upper()
    values = frame["pair"].astype(str).str.upper()
    return frame[(values == target) | (values == alt)]


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
    if status in {"RESEARCH_BLOCKED", "NEEDS_LOCAL_DATA"}:
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
    items = payload.get("items", payload if isinstance(payload, list) else [])
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


def _discovery_components(dydx_tradable: bool, timeframes: list[str], metrics: dict[str, float], funding_drag: float) -> dict[str, float]:
    return {
        "mean_reversion_hint": 8.0 if timeframes else 0.0,
        "cointegration_hint": metrics.get("cointegration_score", 0.0),
        "copula_hint": metrics.get("copula_score", 0.0),
        "liquidity_hint": max(metrics.get("liquidity_hint", 0.0), 8.0 if dydx_tradable else 0.0),
        "dashboard_confirmation_hint": min(metrics.get("local_backtest_score", 0.0) / 5, 10.0),
        "obvious_instability_penalty": -min(funding_drag / 5, 5.0),
    }


def _acceptance_components(dydx_tradable: bool, metrics: dict[str, float], funding_drag: float) -> dict[str, float]:
    return {
        "local_backtest_score": metrics.get("local_backtest_score", 0.0),
        "walk_forward_score": 0.0,
        "regime_stability_score": 0.0,
        "trade_count_score": 0.0,
        "dydx_tradeability_score": 15.0 if dydx_tradable else 0.0,
        "funding_drag_penalty": -min(funding_drag / 2, 10.0),
        "drawdown_penalty": 0.0,
        "cost_failure_penalty": 0.0,
        "stale_data_penalty": 0.0,
    }


def _decision_bucket(discovery: float, acceptance: float, metrics: dict[str, float], dydx_tradable: bool, timeframes: list[str]) -> tuple[str, str]:
    if acceptance >= 70 and dydx_tradable:
        return "PROMOTE", "local_acceptance_score_passed_with_dydx_tradeability"
    if not dydx_tradable or not timeframes:
        return "FETCH_MORE_DATA", "missing_dydx_tradeability_or_timeframe_history"
    if acceptance >= 25 or discovery >= 20:
        return "WATCH", "promising_discovery_or_partial_local_evidence_but_not_promoted"
    return "REJECT", "insufficient_local_acceptance_evidence"


def _missing_pair_data(dydx_tradable: bool, timeframes: list[str], metrics: dict[str, float], funding_drag: float) -> str:
    missing = []
    if not dydx_tradable:
        missing.append("dydx_leg_candles")
    if not timeframes:
        missing.append("common_timeframes")
    if metrics.get("local_backtest_score", 0.0) <= 0:
        missing.append("local_backtest_metrics")
    if funding_drag == 0:
        missing.append("funding_drag_observation")
    return ";".join(missing)


def _harden_trade_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    hardened = frame.copy()
    hardened["good_trade"] = hardened.get(TARGET_COLUMN, 0)
    hardened["profit_after_cost"] = hardened.get(RETURN_COLUMN, 0.0)
    hardened["max_adverse_excursion"] = np.minimum(hardened["profit_after_cost"].astype(float), 0.0)
    hardened["max_favorable_excursion"] = np.maximum(hardened["profit_after_cost"].astype(float), 0.0)
    hardened["hold_bars"] = hardened.get("trade_bars", 0)
    hardened["exit_reason"] = "strategy_exit"
    hardened["feature_timestamp"] = hardened.get("entry_timestamp", "")
    hardened["label_timestamp"] = hardened.get("exit_timestamp", "")
    hardened["uses_dashboard_hindsight"] = False
    hardened["feature_completeness_score"] = hardened.notna().mean(axis=1).round(4)
    hardened["evidence_path"] = "data/raw/pair_details"
    hardened["backtest_label"] = hardened["good_trade"]
    hardened["paper_label"] = ""
    hardened["live_label"] = ""
    return hardened


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
    entry = pd.to_datetime(frame.get("feature_timestamp", pd.Series(dtype=str)), utc=True, errors="coerce")
    label = pd.to_datetime(frame.get("label_timestamp", pd.Series(dtype=str)), utc=True, errors="coerce")
    uses_future = entry.notna() & label.notna() & (entry >= label)
    return pd.DataFrame(
        {
            "trade_id": frame.get("trade_id", pd.Series(range(len(frame)))).astype(str),
            "feature_timestamp": frame.get("feature_timestamp", ""),
            "label_timestamp": frame.get("label_timestamp", ""),
            "uses_future_data": uses_future,
            "uses_dashboard_hindsight": frame.get("uses_dashboard_hindsight", False),
            "feature_completeness_score": frame.get("feature_completeness_score", 0.0),
            "leakage_blocker": np.where(uses_future, "feature_timestamp_not_before_label_timestamp", ""),
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
    best = summary.sort_values(["promising", "profit_factor_delta", "sharpe_delta"], ascending=[False, False, False]).iloc[0]
    accepted = bool(
        best.get("profit_factor_delta", 0.0) > 0
        and best.get("median_filtered_profit_factor", 0.0) >= 1.2
        and best.get("median_filtered_sharpe", 0.0) > 0.0
        and best.get("worst_filtered_drawdown", 1.0) <= 0.30
        and best.get("drawdown_delta", 0.0) <= 0
        and best.get("median_take_rate", 0.0) >= 0.05
        and best.get("total_filtered_trades", 0.0) >= 20
    )
    if not _score_buckets_monotonic(predictions):
        accepted = False
    return {
        "accepted": accepted,
        "best_model": str(best.get("model_name", "")),
        "profit_factor_delta": float(best.get("profit_factor_delta", 0.0)),
        "filtered_profit_factor": float(best.get("median_filtered_profit_factor", 0.0)),
        "filtered_sharpe": float(best.get("median_filtered_sharpe", 0.0)),
        "filtered_drawdown": float(best.get("worst_filtered_drawdown", 0.0)),
        "sharpe_delta": float(best.get("sharpe_delta", 0.0)),
        "drawdown_delta": float(best.get("drawdown_delta", 0.0)),
        "median_take_rate": float(best.get("median_take_rate", 0.0)),
        "total_filtered_trades": int(best.get("total_filtered_trades", 0)),
        "score_buckets_monotonic": _score_buckets_monotonic(predictions),
        "blocker": "" if accepted else "model_acceptance_gates_not_met",
        "created_at": _now(),
        "label_source": "backtest_trained",
    }


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
                "profit_factor_delta": metrics.get("profit_factor_delta", 0.0),
                "sharpe_delta": metrics.get("sharpe_delta", 0.0),
                "drawdown_delta": metrics.get("drawdown_delta", 0.0),
                "take_rate": metrics.get("median_take_rate", 0.0),
                "trades": metrics.get("total_filtered_trades", 0),
                "acceptance_reason": "passed" if metrics.get("accepted", False) else metrics.get("blocker", "blocked"),
            }
        ]
    )


def _score_bucket_report(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "probability_profitable" not in predictions:
        return pd.DataFrame(columns=["score_bucket", "rows", "mean_return", "monotonic"])
    frame = predictions.copy()
    frame["score_bucket"] = pd.cut(frame["probability_profitable"], bins=[-0.01, 0.55, 0.70, 1.01], labels=["skip", "reduced", "full"])
    grouped = frame.groupby("score_bucket", observed=False)[RETURN_COLUMN].agg(["count", "mean"]).reset_index()
    grouped = grouped.rename(columns={"count": "rows", "mean": "mean_return"})
    grouped["monotonic"] = _score_buckets_monotonic(predictions)
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
    return grouped.sort_values("share_of_taken", ascending=False)


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
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(len(returns))) if len(returns) > 1 and returns.std(ddof=0) else 0.0
    drawdown = _max_drawdown(returns)
    return {
        "variant": name,
        "trades": int(len(returns)),
        "take_rate": float(len(returns) / max(total_rows, 1)),
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "expectancy": float(returns.mean()) if len(returns) else 0.0,
        "total_return": float(returns.sum()) if len(returns) else 0.0,
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
        and gated["take_rate"] >= 0.05
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


def _model_failure_attribution(predictions: pd.DataFrame, acceptance: pd.DataFrame) -> pd.DataFrame:
    accepted = bool(not acceptance.empty and acceptance["accepted"].astype(bool).iloc[0])
    rows = []
    if not accepted:
        rows.append({"failure": "incremental_edge_gate", "detail": acceptance["blocker"].iloc[0] if not acceptance.empty else "missing_acceptance"})
    if not _score_buckets_monotonic(predictions):
        rows.append({"failure": "score_bucket_monotonicity", "detail": "higher_scores_did_not_imply_better_returns"})
    concentration = _model_pair_concentration(predictions)
    if not concentration.empty and concentration["share_of_taken"].iloc[0] > 0.65:
        rows.append({"failure": "pair_concentration", "detail": str(concentration["pair"].iloc[0])})
    return pd.DataFrame(rows or [{"failure": "", "detail": "no_failure_detected"}])


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
        "evidence_path",
    ]
    if pair_universe.empty:
        return pd.DataFrame(columns=columns)
    model_ok = bool(not acceptance.empty and acceptance.get("accepted", pd.Series([False])).astype(bool).iloc[0])
    rows = []
    for _, row in pair_universe.head(100).iterrows():
        blocker = ""
        action = "watch"
        reason = str(row.get("decision_reason", ""))
        if row.get("decision_bucket") != "PROMOTE":
            blocker = str(row.get("missing_data_reason", "")) or "not_promoted"
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
                "evidence_path": row.get("evidence_path", ""),
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


def _data_health_rows(system: pd.DataFrame, pair_universe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not system.empty:
        for _, row in system.iterrows():
            rows.append({"area": row.get("check", ""), "ready": row.get("ready", False), "blocker": row.get("blocker", ""), "evidence_path": row.get("evidence_path", "")})
    if not pair_universe.empty:
        stale = pair_universe[pair_universe.get("stale_reason", pd.Series(dtype=str)).astype(str) != ""]
        rows.append({"area": "pair_universe_stale_rows", "ready": stale.empty, "blocker": f"stale_rows={len(stale)}" if not stale.empty else "", "evidence_path": "data/processed/pair_universe.csv"})
    return pd.DataFrame(rows)


def _api_credit_usage_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"source": "crypto_wizards", "usage_known": False, "blocker": "credit_usage_not_captured", "evidence_path": "reports/dashboard_integration_summary.md"},
            {"source": "apify", "usage_known": False, "blocker": "credit_usage_not_captured", "evidence_path": "docs/apify_integration.md"},
            {"source": "dydx", "usage_known": True, "blocker": "", "evidence_path": "reports/dydx_live_market_selector.csv"},
        ]
    )


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


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_parquet_if_available(frame: pd.DataFrame, path: Path) -> str:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
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
    peak = equity.cummax()
    drawdown = (peak - equity) / peak.replace(0, np.nan)
    return float(drawdown.max(skipna=True) or 0.0)


def _exists(path: Path) -> bool:
    return path.exists()


def _acceptance_ready(root: Path) -> bool:
    frame = _read_csv(root / "reports" / "strategy_acceptance_checklist.csv")
    return bool(not frame.empty and frame.get("ready", pd.Series(dtype=bool)).astype(bool).all())


def _acceptance_status(root: Path) -> str:
    frame = _read_csv(root / "reports" / "strategy_acceptance_checklist.csv")
    if frame.empty:
        return "strategy acceptance checklist missing"
    blockers = frame.loc[~frame.get("ready", pd.Series(True, index=frame.index)).astype(bool), "blocker"].dropna().astype(str).tolist()
    return "ready" if not blockers else "blocked:" + ";".join(blockers[:3])


def _state_row(area: str, ready: bool, status: str, evidence_path: Path, next_action: str) -> dict[str, object]:
    return {
        "area": area,
        "ready": ready,
        "status": "ready" if ready else "blocked",
        "blocker": "" if ready else status,
        "detail": status,
        "evidence_path": str(evidence_path),
        "next_action": next_action,
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
    lines = ["# Current State", "", "| Area | Status | Blocker | Evidence | Next Action |", "| --- | --- | --- | --- | --- |"]
    for _, row in frame.iterrows():
        lines.append(f"| {row['area']} | {row['status']} | {row['blocker']} | {row['evidence_path']} | {row['next_action']} |")
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


def _command_center_markdown(pair_universe: pd.DataFrame, current: pd.DataFrame, acceptance: pd.DataFrame, data_health: pd.DataFrame) -> str:
    lines = ["# Command Center", "", "## Current State", ""]
    if current.empty:
        lines.append("- current state not built")
    else:
        for _, row in current.iterrows():
            lines.append(f"- {row.get('area', '')}: {row.get('status', '')} {row.get('blocker', '')}")
    lines.extend(["", "## Pair Universe", ""])
    if pair_universe.empty:
        lines.append("- pair universe missing")
    else:
        counts = pair_universe["decision_bucket"].value_counts()
        for bucket in ["PROMOTE", "WATCH", "FETCH_MORE_DATA", "REJECT"]:
            lines.append(f"- {bucket}: {int(counts.get(bucket, 0))}")
    lines.extend(["", "## Model Gate", ""])
    if acceptance.empty:
        lines.append("- blocked: model acceptance missing")
    else:
        lines.append(f"- accepted: {bool(acceptance.get('accepted', pd.Series([False])).astype(bool).iloc[0])}")
    lines.extend(["", "## Data Health", f"- rows: {len(data_health)}", "", "Blockers are intentionally visible. No stale row is actionable."])
    return "\n".join(lines) + "\n"
