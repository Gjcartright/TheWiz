from __future__ import annotations

import argparse
from itertools import combinations
from datetime import datetime, timedelta, timezone
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import shutil
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from quant_platform.active_pipeline import (
    archive_from_index,
    build_artifact_index,
    build_command_dashboard,
    build_market_venue_context,
    build_multi_venue_history_readiness,
    build_pair_universe,
    build_trade_dataset,
    build_venue_lane_test_plan,
    current_state,
    export_trade_gate_model,
    run_model_gated_backtest,
    system_check,
    train_trade_gate,
)
from quant_platform.apify_sources import infer_apify_venue, parse_apify_sources_from_mcp_url, refresh_apify_sources
from quant_platform.api_extraction import (
    CryptoWizardsExtractor,
    CryptoWizardsFetchError,
    CryptoWizardsLiveConfig,
    parse_endpoint_specs,
)
from quant_platform.backtest import CostModel, backtest_pair, backtest_two_leg_spread
from quant_platform.binance_spot import (
    build_binance_spot_lane_report,
    build_binance_spot_pair_history,
    fetch_binance_spot_candles,
)
from quant_platform.crypto_wizards_catalog import endpoint_rows
from quant_platform.crypto_wizards_history import (
    CryptoWizardsHistoryRequest,
    crawl_prescanned_backtest_histories,
    crawl_prescanned_zscores_histories,
    official_min5_request_rows,
    write_backtest_pair_payload,
    write_zscores_pair_payload,
)
from quant_platform.crypto_wizards_scanner import load_scanner_rows, write_scanner_reports
from quant_platform.dydx_candles import (
    archive_dydx_candles,
    backfill_provisional_pair_history_features,
    build_pair_history_from_windowed_candles,
    build_pair_history_from_candles,
    dydx_two_leg_request_rows,
    import_dydx_candle_bundle,
    load_loose_candle_payload,
)
from quant_platform.execution import (
    DydxNetworkConfig,
    OrderIntent,
    SpreadOrderPlan,
    append_paper_trading_record,
    block_paper_plan_for_execution_config,
    build_execution_venue,
    build_dydx_indexer_adapter,
    build_dydx_order_client_adapter,
    build_research_gated_paper_plan,
    dydx_readiness_report,
    validate_dydx_order_client_adapter,
    build_venue_order_client_adapter,
    validate_venue_order_client_adapter,
    venue_has_paper_adapter,
    paper_trading_record,
    submit_paper_plan,
    PaperDydxExecution,
)
from quant_platform.experiments import AcceptanceGate, ExperimentConfig, ExperimentHarness, PairDataset, strategy_acceptance_report
from quant_platform.env import load_env_file
from quant_platform.family_matrix import run_family_matrix
from quant_platform.field_registry import field_rows
from quant_platform.fixture_ingestion import CANONICAL_ALIASES, datasets_from_fixtures, snake_case, write_fixture_field_dictionary
from quant_platform.formula_registry import FORMULAS
from quant_platform.funding import (
    enrich_pair_dataset_with_funding,
    funding_coverage_for_pairs,
    funding_market_requirements,
    funding_rows_from_dydx_payload,
    normalize_funding_rows,
)
from quant_platform.hyperliquid import (
    build_hyperliquid_lane_report,
    build_hyperliquid_pair_history,
    fetch_hyperliquid_candles,
)
from quant_platform.meta_learning import JsonlTradeStore, TradeRecord, write_learning_event_summary_report
from quant_platform.ml_filter import (
    build_trade_filter_dataset,
    shadow_trade_filter_predictions,
    shadow_model_branch_comparison,
    train_trade_filter_walkforward,
)
from quant_platform.pair_detail_ingestion import (
    ECM_FIELD_SOURCE,
    datasets_from_pair_detail_snapshots,
    extract_history_rows,
    load_pair_detail_snapshots,
    pair_detail_capture_audit,
    pair_detail_capture_checklist,
    PAIR_DETAIL_CAPTURE_AUDIT_COLUMNS,
    PAIR_DETAIL_CAPTURE_CHECKLIST_COLUMNS,
    PAIR_DETAIL_QUALITY_COLUMNS,
    pair_detail_history_coverage,
    pair_detail_quality_report,
    pair_detail_payload_capture_audit,
    pair_detail_payload_capture_checklist,
    pair_detail_payload_history_coverage,
    snapshot_from_payload,
    write_pair_detail_reports,
)
from quant_platform.research_quantization import quantize_family_matrix
from quant_platform.regimes import RegimeConfig, classify_regimes, write_regime_dataset_report
from quant_platform.orchestration import run_orchestrator
from quant_platform.orchestration.mini_agents import build_mini_agent_orchestration
from quant_platform.orchestration.orchestrator_assistant import build_orchestrator_assistant
from quant_platform.orchestration.specialist_scoreboard import build_specialist_scoreboard
from quant_platform.rl import export_rl_policy, run_rl_idea_scout, run_rl_research
from quant_platform.rl.train_ppo import train_ppo_research_policy
from quant_platform.strategies import STRATEGIES, strategy_rows, zscore_signal
from quant_platform.trade_timing import (
    TRADE_TIMING_TEMPLATE_COLUMNS,
    load_trade_timing_history,
    trade_timing_comparison_report_frame,
    trade_timing_comparison_summary,
    write_trade_timing_template,
)
from quant_platform.wizard_evidence import (
    build_wizard_diagnostic_confirmation,
    build_wizard_evidence,
    build_wizard_exact_mode_capture_queue,
    build_wizard_hypotheses,
    build_wizard_local_parity,
    build_wizard_research_pack,
)
from quant_platform.wizard_local_verification import build_wizard_local_verification_batch, verify_wizard_local_mode


ROOT = Path(__file__).resolve().parents[2]
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
DEFAULT_FAMILY_SWEEP_PAIRS = (
    "BTC-USD-SOL-USD",
    "DOGE-USD-SOL-USD",
    "SOL-USD-XRP-USD",
    "SOL-USD-LINK-USD",
)


def _acceptance_report_path() -> Path:
    configured = os.getenv("QPA_ACCEPTANCE_REPORT_PATH", "").strip()
    if not configured:
        return ROOT / "reports" / "acceptance_report.csv"
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path



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
    if any(_is_relative_to(resolved, folder) or resolved == folder.resolve() for folder in dydx_dirs):
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
    if _is_relative_to(resolved, _dydx_inbox_dir()) or _is_relative_to(resolved, _dydx_manual_dir()):
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
        if (alt_url, tuple(sorted(headers.items())), True) not in {(u, tuple(sorted(h.items())), v) for u, h, v in variants}:
            variants.append((alt_url, headers, True))
        if parsed.scheme == "https":
            if (alt_url, tuple(sorted(headers.items())), False) not in {(u, tuple(sorted(h.items())), v) for u, h, v in variants}:
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
        min_profit_factor=base.min_profit_factor,
        preferred_profit_factor=base.preferred_profit_factor,
        min_sharpe=base.min_sharpe,
        preferred_sharpe=base.preferred_sharpe,
        max_drawdown=base.max_drawdown,
        preferred_max_drawdown=base.preferred_max_drawdown,
        min_trades=_read_positive_int_env("QPA_MIN_TRADES", base.min_trades),
        preferred_trades=base.preferred_trades,
        min_pairs=_read_positive_int_env("QPA_MIN_PAIRS", base.min_pairs),
        required_cost_buckets=base.required_cost_buckets,
        required_regime=base.required_regime,
        require_positive_expectancy=base.require_positive_expectancy,
        require_two_leg_backtests=base.require_two_leg_backtests,
        require_two_leg_execution_inputs=base.require_two_leg_execution_inputs,
    )


def _experiment_harness(
    *,
    min_rows: int | None = None,
    gate: AcceptanceGate | None = None,
) -> ExperimentHarness:
    base = ExperimentConfig()
    return ExperimentHarness(
        config=ExperimentConfig(
            cost_buckets=base.cost_buckets,
            min_rows=min_rows if min_rows is not None else base.min_rows,
            include_overall_regime=base.include_overall_regime,
            regime_column=base.regime_column,
            gate=gate or _acceptance_gate_from_env(base.gate),
        )
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
DEFAULT_DYDX_LIVE_SELECTOR_ANCHORS = ("BTC-USD", "ETH-USD", "SOL-USD")
DEFAULT_DYDX_LIVE_SELECTOR_EXCLUDED_MARKETS = {"DAI-USD", "EUR-USD", "EURC-USD", "PAXG-USD", "WTI-USD"}


def _write_csv_atomic(frame: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(output)
    return output


def build_dictionaries() -> None:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    pd.DataFrame(field_rows()).to_csv(docs / "field_dictionary.csv", index=False)
    pd.DataFrame(strategy_rows()).to_csv(docs / "strategy_registry.csv", index=False)
    pd.DataFrame(endpoint_rows()).to_csv(docs / "crypto_wizards_endpoint_catalog.csv", index=False)

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
    (docs / "formula_dictionary.md").write_text("\n".join(formula_lines), encoding="utf-8")

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
    (docs / "quant_brain.md").write_text("\n".join(brain_lines), encoding="utf-8")


def run_demo_backtest() -> None:
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    n = 1200
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.96 * spread[i - 1] + rng.normal(0, 0.02)
    frame = pd.DataFrame({"spread": spread})
    frame["zscore"] = (frame["spread"] - frame["spread"].rolling(80).mean()) / frame["spread"].rolling(80).std()
    frame = frame.dropna().reset_index(drop=True)
    result = backtest_pair(frame, zscore_signal(frame), CostModel())
    pd.DataFrame([result.__dict__]).to_csv(reports / "demo_backtest.csv", index=False)
    print(result)


def _demo_pair_frame(seed: int, n: int, phi: float, noise: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = phi * spread[i - 1] + rng.normal(0, noise)
    frame = pd.DataFrame({"spread": spread})
    frame["zscore"] = (frame["spread"] - frame["spread"].rolling(80).mean()) / frame["spread"].rolling(80).std()
    frame["conditional_probability_distortion"] = np.tanh(frame["zscore"].fillna(0.0) / 3.0)
    return classify_regimes(frame.dropna().reset_index(drop=True), RegimeConfig(lookback=40, trend_threshold=0.03))


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
    field_path = write_fixture_field_dictionary(input_dir, docs / "crypto_wizards_fixture_field_dictionary.csv")
    datasets = datasets_from_fixtures(input_dir)
    datasets = [PairDataset(dataset.pair, classify_regimes(dataset.frame, RegimeConfig(preserve_existing=True))) for dataset in datasets]
    write_regime_dataset_report(datasets, reports / "regime_dataset_report.csv")
    pd.DataFrame(
        [{"pair": dataset.pair, "rows": len(dataset.frame), "columns": ";".join(dataset.frame.columns)} for dataset in datasets]
    ).to_csv(reports / "fixture_ingestion_summary.csv", index=False)
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
    combined.to_csv(output_path, index=False)
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


def run_fixture_experiments(input_dir: Path | None = None, funding_path: Path | None = None) -> None:
    input_dir = input_dir or _normalized_root()
    _assert_fixture_experiment_input_normalized(input_dir)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    datasets = datasets_from_fixtures(input_dir)
    datasets = _enrich_datasets_with_funding(datasets, funding_path)
    datasets = [PairDataset(dataset.pair, classify_regimes(dataset.frame, RegimeConfig(preserve_existing=True))) for dataset in datasets]
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
    pair_id = snapshot.pair_id if snapshot.pair_id and snapshot.pair_id != "unknown" else input_path.stem
    filename = output_name or f"pair_{pair_id}_capture.json"
    if not filename.endswith(".json"):
        filename = f"{filename}.json"
    output_dir = ROOT / "data" / "raw" / "pair_details"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    rows = extract_history_rows(payload)
    paths = write_pair_detail_reports(output_dir, ROOT / "reports")
    coverage = pd.DataFrame(pair_detail_history_coverage(output_dir))
    imported_row = coverage[coverage["path"] == str(output_path)] if not coverage.empty else pd.DataFrame()
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
        two_leg_ready_paths = imported_audit[imported_audit["two_leg_execution_ready"]]["json_path"].tolist()
        print(f"capture_candidate_paths: {len(imported_audit)}")
        print(f"experiment_ready_paths: {','.join(ready_paths) if ready_paths else 'none'}")
        print(f"ecm_ready_paths: {','.join(ecm_ready_paths) if ecm_ready_paths else 'none'}")
        print(f"two_leg_ready_paths: {','.join(two_leg_ready_paths) if two_leg_ready_paths else 'none'}")
    checklist = pair_detail_payload_capture_checklist(payload, output_path)
    print(f"found_required_fields: {checklist['found_required_fields'] or 'none'}")
    print(f"missing_required_fields: {checklist['missing_required_fields'] or 'none'}")
    _print_capture_checklist_summary(checklist)
    print(f"next_capture_focus: {checklist['next_capture_focus']}")
    for name, path in paths.items():
        print(f"{name}: {path}")


def import_latest_pair_detail_download(download_dir: Path | None = None, output_name: str | None = None) -> Path:
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
        output = archive_dydx_candles(input_path, output_dir or ROOT / "data" / "raw" / "dydx_candles")
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
) -> pd.DataFrame:
    plan_file = plan_path or ROOT / "reports" / "dydx_long_history_plan.csv"
    if not plan_file.exists():
        raise SystemExit(f"long-history plan not found: {plan_file}")
    plan = pd.read_csv(plan_file)
    if plan.empty:
        raise SystemExit(f"long-history plan is empty: {plan_file}")
    if "method" not in plan.columns or "url" not in plan.columns or "save_as" not in plan.columns:
        raise SystemExit(f"long-history plan is missing required columns: {plan_file}")

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
    fetch_frame = fetch_dydx_long_history_windows(
        max_windows=windows,
        indexer_base=requested_indexer_base,
        indexer_scheme=indexer_scheme,
        allow_stale_fetch=allow_stale_fetch,
    )
    if fetch_frame.empty:
        raise SystemExit("long-history fetch produced no candle files")
    resolved_pair_id = str(plan.iloc[0]["pair_id"]) if not plan.empty else (pair_id or "long_history")
    resolved_asset_x = str(plan.iloc[0]["asset_x"]) if not plan.empty else (asset_x or "")
    resolved_asset_y = str(plan.iloc[0]["asset_y"]) if not plan.empty else (asset_y or "")
    paths = build_dydx_long_history_pair(
        input_dir=ROOT / "data" / "raw" / "dydx_long_history" / resolved_pair_id,
        asset_x=resolved_asset_x,
        asset_y=resolved_asset_y,
        pair_id=resolved_pair_id,
        hedge_ratio=1.0,
        beta=1.0,
        interval=resolution.lower(),
        zscore_window=320,
        derive_hedge_ratio=derive_hedge_ratio,
        run_research=run_research,
        funding_path=funding_path,
    )
    paths["plan"] = ROOT / "reports" / "dydx_long_history_plan.csv"
    paths["fetch"] = ROOT / "reports" / "dydx_long_history_fetch.csv"
    return paths


def import_dydx_candle_bundle_from_cli(input_path: Path, output_dir: Path | None = None, zscore_window: int = 320) -> list[Path]:
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
        quality = pd.DataFrame(pair_detail_quality_report(pair_dir), columns=PAIR_DETAIL_QUALITY_COLUMNS)
        imported_quality = quality[quality["path"].isin({str(path) for path in paths})] if not quality.empty else quality
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
    zscore_window: int = 320,
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
    zscore_window: int = 320,
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
    zscore_window: int = 320,
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
        left_candles = archive_dydx_candles(saved["asset_x_candles_5mins"], ROOT / "data" / "raw" / "dydx_candles")
        right_candles = archive_dydx_candles(saved["asset_y_candles_5mins"], ROOT / "data" / "raw" / "dydx_candles")
    except ValueError:
        return result_paths
    funding_csv = export_dydx_funding_payload(manual_dir, funding_path or ROOT / "data" / "processed" / "dydx_funding.csv")
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
    zscore_window: int = 320,
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
    timeout: float = 30.0,
    max_retries: int = 3,
    allow_stale_fetch: bool = False,
    fetch_scheme: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    allow_stale = allow_stale_fetch or os.getenv("QPA_ALLOW_STALE_FETCH", "").lower() in {"1", "true", "yes"}
    use_requests = os.getenv("QPA_USE_REQUESTS_FETCH", "").lower() not in {"0", "false", "no", "off"}
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
                        get_kwargs = {"headers": headers, "timeout": timeout}
                        if not verify_ssl:
                            get_kwargs["verify"] = False
                        response = requests.get(request_url, **get_kwargs)
                        response.raise_for_status()
                        try:
                            payload = response.json()
                            output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                        except ValueError:
                            output_path.write_text(response.text, encoding="utf-8")
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
                    connect_port = str(parse.port) if parse.port else ("443" if parse.scheme == "https" else "80")
                    resolve_host = f"{fallback_host}:{connect_port}:{target_ip}"
                    # Keep host-based URL to preserve TLS SNI while routing via explicit DNS fallback.
                    # `curl --resolve` handles host-to-IP mapping without forcing IP into URL path.
                    curl_cmd.extend(["--http1.1", "--resolve", resolve_host, "-H", f"Host: {fallback_host}"])
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
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except json.JSONDecodeError:
        pass
    return output_path


def _resolve_two_leg_assets(*, pair: str | None, asset_x: str | None, asset_y: str | None) -> tuple[str, str]:
    if asset_x and asset_y:
        return asset_x, asset_y
    if pair:
        requirements = funding_market_requirements([pair])
        if not requirements.empty and bool(requirements.iloc[0].get("valid", False)):
            return str(requirements.iloc[0]["market_x"]), str(requirements.iloc[0]["market_y"])
        error = str(requirements.iloc[0].get("error", "invalid pair")) if not requirements.empty else "invalid pair"
        raise SystemExit(f"could not resolve dYdX markets from --pair {pair}: {error}")
    raise SystemExit("dydx-two-leg-request-template requires --pair or both --asset-x and --asset-y")


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
    two_leg_ready_paths = [row["json_path"] for row in audit if bool(row.get("two_leg_execution_ready"))]

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
    print(f"two_leg_ready_paths: {','.join(two_leg_ready_paths) if two_leg_ready_paths else 'none'}")
    print(f"found_required_fields: {checklist['found_required_fields'] or 'none'}")
    print(f"missing_required_fields: {checklist['missing_required_fields'] or 'none'}")
    _print_capture_checklist_summary(checklist)
    print(f"next_capture_focus: {checklist['next_capture_focus']}")


def pair_detail_capture_preflight(input_path: Path, output_path: Path | None = None) -> pd.DataFrame:
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


def print_pair_detail_capture_preflight(input_path: Path | None, output_path: Path | None = None) -> None:
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
    frame = pd.DataFrame(pair_detail_capture_checklist(input_dir), columns=PAIR_DETAIL_CAPTURE_CHECKLIST_COLUMNS)
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
) -> None:
    input_dir = input_dir or ROOT / "data" / "raw" / "pair_details"
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    datasets = datasets_from_pair_detail_snapshots(input_dir, require_research_usable=require_research_usable)
    datasets = _enrich_datasets_with_funding(datasets, funding_path)
    datasets = [PairDataset(dataset.pair, classify_regimes(dataset.frame, RegimeConfig(preserve_existing=True))) for dataset in datasets]
    if not datasets:
        raise SystemExit(f"no experiment-ready pair-detail history datasets found in {input_dir}")
    write_regime_dataset_report(datasets, reports / "regime_dataset_report.csv")
    harness = _experiment_harness()
    results = harness.run(datasets)
    paths = harness.write_reports(results, reports)
    print(f"loaded {len(datasets)} pair-detail history dataset(s)")
    print(f"wrote {len(results)} experiment rows")
    for name, path in paths.items():
        print(f"{name}: {path}")


def build_ml_trade_filter_dataset_report(
    input_dir: Path | None = None,
    funding_path: Path | None = None,
    output_path: Path | None = None,
    require_research_usable: bool = True,
) -> Path:
    source = input_dir or ROOT / "data" / "raw" / "pair_details"
    output = output_path or ROOT / "reports" / "ml_trade_filter_dataset.csv"
    datasets = datasets_from_pair_detail_snapshots(source, require_research_usable=require_research_usable)
    datasets = _enrich_datasets_with_funding(datasets, funding_path)
    datasets = [PairDataset(dataset.pair, classify_regimes(dataset.frame, RegimeConfig(preserve_existing=True))) for dataset in datasets]
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
) -> None:
    dataset = _load_ml_trade_filter_dataset(input_dir, funding_path)
    artifact = model_path or ROOT / "reports" / "ml_trade_filter" / "ml_trade_filter_best_model.pkl"
    output = output_path or ROOT / "reports" / "ml_trade_filter_shadow_predictions.csv"
    path = shadow_trade_filter_predictions(dataset, model_artifact_path=artifact, output_path=output)
    frame = pd.read_csv(path)
    print(frame.head(20).to_string(index=False))
    print(f"ml_trade_filter_shadow_predictions: {path}")


def print_compare_ml_shadow_models(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    pair_list: tuple[str, ...] = (),
) -> None:
    predictions_path = input_dir or ROOT / "reports" / "ml_trade_filter" / "ml_trade_filter_walkforward_predictions.csv"
    if not predictions_path.exists():
        raise SystemExit(f"ML walk-forward predictions not found: {predictions_path}")
    output = output_dir or predictions_path.parent
    predictions = pd.read_csv(predictions_path)
    model_report, pair_report = shadow_model_branch_comparison(predictions, pairs=pair_list or None)
    model_path = _write_csv_atomic(model_report, output / "ml_trade_filter_branch_model_comparison.csv")
    pair_path = _write_csv_atomic(pair_report, output / "ml_trade_filter_branch_pair_comparison.csv")
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
    classified = [PairDataset(dataset.pair, classify_regimes(dataset.frame, RegimeConfig(preserve_existing=True))) for dataset in datasets]
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
    best_by_family_path = _write_csv_atomic(best_by_family, output / "strategy_family_best_by_family.csv")

    shortlist = best_by_family[
        (
            best_by_family["production_eligible"].fillna(False).astype(bool)
            | best_by_family["preferred_eligible"].fillna(False).astype(bool)
            | (pd.to_numeric(best_by_family["passing_pairs"], errors="coerce").fillna(0) > 0)
        )
    ].copy()
    if shortlist.empty:
        shortlist = best_by_family.head(min(3, len(best_by_family))).copy()
    shortlist.insert(1, "promotion_reason", shortlist.apply(_strategy_family_promotion_reason, axis=1))
    shortlist_path = _write_csv_atomic(shortlist, output / "strategy_family_promotion_shortlist.csv")

    notes_path = output / "strategy_family_sweep_notes.md"
    notes_path.write_text(_strategy_family_sweep_notes(selected_pairs, summary, best_by_family, shortlist), encoding="utf-8")
    failure_attribution_path = output / "strategy_family_failure_attribution.csv"
    family_failure_attribution_report(output, failure_attribution_path)
    failure_notes_path = output / "strategy_family_failure_attribution.md"
    failure_notes_path.write_text(_strategy_family_failure_notes(output), encoding="utf-8")

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
    passing_pairs = int(pd.to_numeric(pd.Series([row.get("passing_pairs")]), errors="coerce").fillna(0).iloc[0])
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
        lines.append(f"- `{row['family']}` / `{row['strategy_name']}` because `{row['promotion_reason']}`")
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
                "strategies_in_family": int(len(group)),
                "evaluated_runs_best_strategy": int(pd.to_numeric(pd.Series([best.get("evaluated_runs")]), errors="coerce").fillna(0).iloc[0]),
                "best_passing_pairs": int(pd.to_numeric(pd.Series([best.get("passing_pairs")]), errors="coerce").fillna(0).iloc[0]),
                "best_total_trades": int(pd.to_numeric(pd.Series([best.get("total_trades")]), errors="coerce").fillna(0).iloc[0]),
                "best_median_profit_factor": float(pd.to_numeric(pd.Series([best.get("median_profit_factor")]), errors="coerce").fillna(0.0).iloc[0]),
                "best_median_sharpe": float(pd.to_numeric(pd.Series([best.get("median_sharpe")]), errors="coerce").fillna(0.0).iloc[0]),
                "best_worst_drawdown": float(pd.to_numeric(pd.Series([best.get("worst_drawdown")]), errors="coerce").fillna(0.0).iloc[0]),
                "strategies_blocked_by_passing_pairs": blocker_counts.get("passing_pairs", 0),
                "strategies_blocked_by_total_trades": blocker_counts.get("total_trades", 0),
                "strategies_blocked_by_median_profit_factor": blocker_counts.get("median_profit_factor", 0),
                "strategies_blocked_by_median_sharpe": blocker_counts.get("median_sharpe", 0),
                "strategies_blocked_by_worst_drawdown": blocker_counts.get("worst_drawdown", 0),
                "strategies_blocked_by_no_evaluated_runs": blocker_counts.get("no_evaluated_runs", 0),
                "top_blockers": ";".join(f"{name}:{count}" for name, count in _sorted_counter_items(blocker_counts)[:5]),
                "diagnosis": diagnosis,
                "next_action": _strategy_failure_next_action(diagnosis),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        [
            "best_passing_pairs",
            "best_median_sharpe",
            "best_median_profit_factor",
            "best_total_trades",
            "best_worst_drawdown",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
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
    best_passing_pairs = int(pd.to_numeric(pd.Series([best.get("passing_pairs")]), errors="coerce").fillna(0).iloc[0])
    best_total_trades = int(pd.to_numeric(pd.Series([best.get("total_trades")]), errors="coerce").fillna(0).iloc[0])
    best_pf = float(pd.to_numeric(pd.Series([best.get("median_profit_factor")]), errors="coerce").fillna(0.0).iloc[0])
    best_sharpe = float(pd.to_numeric(pd.Series([best.get("median_sharpe")]), errors="coerce").fillna(0.0).iloc[0])
    best_dd = float(pd.to_numeric(pd.Series([best.get("worst_drawdown")]), errors="coerce").fillna(0.0).iloc[0])
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
    frame = family_failure_attribution_report(base, base / "strategy_family_failure_attribution.csv")
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
    output = output_path or (sweep_dir or ROOT / "reports" / "strategy_family_sweep") / "strategy_family_failure_attribution.csv"
    frame = family_failure_attribution_report(sweep_dir, output)
    print(frame.to_string(index=False))
    print(f"strategy_family_failure_attribution: {output}")


def _enrich_datasets_with_funding(datasets: list[PairDataset], funding_path: Path | None) -> list[PairDataset]:
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


def export_dydx_funding_payload(input_path: Path, output_path: Path | None = None, market: str | None = None) -> Path:
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
        raise SystemExit("fetch-dydx-funding requires --market, with comma-separated markets allowed")
    config = DydxNetworkConfig.paper_testnet_from_env()
    adapter = build_dydx_indexer_adapter(config)
    if adapter is None:
        raise SystemExit("dYdX indexer adapter is not available; install/wire the official v4 client first")
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


def _dydx_funding_rows_from_path(input_path: Path, market: str | None = None) -> list[dict[str, object]]:
    if input_path.is_dir():
        rows: list[dict[str, object]] = []
        paths = sorted(path for path in input_path.glob("*.json") if path.is_file())
        if not paths:
            raise SystemExit(f"no JSON funding payloads found in {input_path}")
        for path in paths:
            rows.extend(_dydx_funding_rows_from_file(path, market or _market_from_funding_filename(path)))
        return rows
    return _dydx_funding_rows_from_file(input_path, market or _market_from_funding_filename(input_path))


def _dydx_funding_rows_from_file(input_path: Path, market: str | None = None) -> list[dict[str, object]]:
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"input is not valid JSON: {input_path}: {exc}") from exc
    return funding_rows_from_dydx_payload(payload, market=market)


def _market_from_funding_filename(path: Path) -> str | None:
    stem = re.sub(r"(?i)(?:^funding[_-]?|[_-]?funding$|[_-]?historical$|[_-]?history$)", "", path.stem)
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
    invalid_pairs = sorted(str(value) for value in frame.loc[~frame.index.isin(valid_rows.index), "pair"].dropna())
    print(f"funding_required_markets: {';'.join(required_markets) if required_markets else 'none'}")
    print(f"fetch_dydx_funding_market_arg: {','.join(required_markets) if required_markets else 'none'}")
    print(f"funding_invalid_pairs: {';'.join(invalid_pairs) if invalid_pairs else 'none'}")
    print(f"funding_requirements: {output}")


def funding_template_report(
    pairs: list[str] | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    requirements = funding_requirements_report(pairs)
    valid_rows = requirements[requirements.get("valid", pd.Series(dtype=bool)).fillna(False).astype(bool)]
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
    normalized = normalize_funding_rows(data.loc[ready_indices]) if ready_indices and not missing_columns else pd.DataFrame()
    required_markets = _funding_required_markets()
    ready_markets = sorted(normalized["market"].dropna().astype(str).unique()) if not normalized.empty else []
    missing_markets = sorted(set(required_markets).difference(ready_markets))
    ready_to_import = bool(ready_indices and not invalid_rows and not missing_columns and not missing_markets)
    frame = pd.DataFrame(
        [
            {
                "path": str(source),
                "rows": len(data),
                "ready_rows": len(ready_indices),
                "blocked_rows": max(len(data) - len(ready_indices), 0) if not missing_columns else len(data),
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
                    "blocker": row.get("invalid_rows", "") or row.get("missing_markets", "") or row.get("missing_columns", ""),
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


def print_funding_template_check(input_path: Path | None = None, output_path: Path | None = None) -> None:
    output = output_path or ROOT / "reports" / "funding_template_check.csv"
    frame = funding_template_check_report(input_path, output)
    print(frame.to_string(index=False))
    print(f"funding_template_check: {output}")


def print_import_funding_template(input_path: Path | None = None, output_path: Path | None = None) -> None:
    frame = import_funding_template(input_path=input_path, output_path=output_path)
    report = ROOT / "reports" / "funding_template_import_report.csv"
    print(frame.to_string(index=False))
    print(f"funding_template_import_report: {report}")


def print_funding_coverage(funding_path: Path | None, pair: str | None = None, output_path: Path | None = None) -> None:
    if funding_path is None:
        raise SystemExit("funding-coverage requires --funding-path")
    pairs = [pair] if pair else None
    output = output_path or ROOT / "reports" / "funding_coverage.csv"
    frame = funding_coverage_report(funding_path, pairs, output)
    print(frame.to_string(index=False))
    if not frame.empty:
        ready_pairs = int(frame["ready"].fillna(False).astype(bool).sum()) if "ready" in frame.columns else 0
        required_markets = _semicolon_values(frame.get("required_markets", pd.Series(dtype=str)))
        missing_markets = _semicolon_values(frame.get("missing_markets", pd.Series(dtype=str)))
        print(f"funding_pairs_ready: {ready_pairs}/{len(frame)}")
        print(f"funding_required_markets: {';'.join(required_markets) if required_markets else 'none'}")
        print(f"funding_missing_markets: {';'.join(missing_markets) if missing_markets else 'none'}")
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
    valid_rows = requirements[requirements.get("valid", pd.Series(dtype=bool)).fillna(False).astype(bool)]
    return _semicolon_values(valid_rows.get("required_markets", pd.Series(dtype=str)))


def _funding_template_validation(frame: pd.DataFrame) -> tuple[list[str], list[int], list[str]]:
    missing_columns = [column for column in FUNDING_TEMPLATE_REQUIRED_COLUMNS if column not in frame.columns]
    invalid_rows: list[str] = []
    ready_indices: list[int] = []
    if missing_columns or frame.empty:
        return missing_columns, ready_indices, invalid_rows
    for index, row in frame.iterrows():
        missing = [column for column in FUNDING_TEMPLATE_REQUIRED_COLUMNS if str(row.get(column, "")).strip() == ""]
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
    rows.append(_spine_row(step="project_objective", status=objective_status, detail=objective_detail))

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
    rows.append(_spine_row(step="priority_readiness", status="completed", detail="reports/priority_readiness.csv"))

    gates = readiness.set_index("gate") if not readiness.empty else pd.DataFrame()
    history_gate = _gate_ready(gates, "pair_detail_history")
    two_leg_gate = _gate_ready(gates, "pair_detail_two_leg_execution_history")
    if not history_gate:
        rows.append(
            _spine_row(
                step="run_pair_detail_experiments",
                status="skipped",
                detail=_gate_blocker(gates, "pair_detail_history") or "pair_detail_history_not_ready",
            )
        )
    elif require_two_leg and not two_leg_gate:
        rows.append(
            _spine_row(
                step="run_pair_detail_experiments",
                status="skipped",
                detail=_gate_blocker(gates, "pair_detail_two_leg_execution_history") or "two_leg_history_not_ready",
            )
        )
    else:
        datasets = datasets_from_pair_detail_snapshots(input_dir, require_research_usable=require_two_leg)
        datasets = _enrich_datasets_with_funding(datasets, funding_path)
        datasets = [
            PairDataset(dataset.pair, classify_regimes(dataset.frame, RegimeConfig(preserve_existing=True)))
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
                _spine_row(step="priority_readiness_after_experiments", status="completed", detail="reports/priority_readiness.csv")
            )

    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, reports / "research_spine.csv")
    return frame


def print_research_spine(
    input_dir: Path | None = None,
    require_two_leg: bool = True,
    funding_path: Path | None = None,
) -> None:
    frame = research_spine(input_dir=input_dir, require_two_leg=require_two_leg, funding_path=funding_path)
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
        rows.append(_spine_row(step="research_spine", status="skipped", detail="funding_coverage_not_ready"))
        frame = pd.DataFrame(rows)
        _write_csv_atomic(frame, output)
        return frame
    try:
        coverage = funding_coverage_report(funding_path, output_path=reports / "funding_coverage.csv")
    except SystemExit as exc:
        rows.append(_spine_row(step="funding_coverage", status="blocked", detail=str(exc)))
        rows.append(_spine_row(step="research_spine", status="skipped", detail="funding_coverage_not_ready"))
        frame = pd.DataFrame(rows)
        _write_csv_atomic(frame, output)
        return frame
    ready_pairs = int(coverage.get("ready", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    total_pairs = int(len(coverage))
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
        rows.append(_spine_row(step="research_spine", status="skipped", detail="funding_coverage_not_ready"))
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
    spine = research_spine(input_dir=input_dir, require_two_leg=require_two_leg, funding_path=funding_path)
    spine_status = "completed" if not spine.empty and not (spine["status"] == "failed").any() else "failed"
    refreshed_coverage = funding_coverage_report(funding_path, output_path=reports / "funding_coverage.csv")
    refreshed_ready_pairs = int(
        refreshed_coverage.get("ready", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
    )
    refreshed_total_pairs = int(len(refreshed_coverage))
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
    ready_steps = int(acceptance.get("ready", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
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
    frame = funded_research_spine(funding_path, input_dir=input_dir, require_two_leg=require_two_leg, output_path=output)
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
    _print_imported_pair_quality(paths, pair_dir, label="crypto_wizards_min5_backtest_histories_written")
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
    quality = pd.DataFrame(pair_detail_quality_report(pair_dir), columns=PAIR_DETAIL_QUALITY_COLUMNS)
    imported_quality = quality[quality["path"].isin({str(path) for path in paths})] if not quality.empty else quality
    print(f"{label}: {len(paths)}")
    for path in paths:
        print(f"pair_history: {path}")
    if not imported_quality.empty:
        print(
            imported_quality[
                ["pair", "interval", "history_rows", "research_usable", "execution_usable", "quality_blockers"]
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
        raise SystemExit(f"input JSON must be an object response from /v1beta/zscores: {input_path}")

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
    quality = pd.DataFrame(pair_detail_quality_report(pair_dir), columns=PAIR_DETAIL_QUALITY_COLUMNS)
    imported_quality = quality[quality["path"] == str(path)] if not quality.empty else quality
    print(f"imported_crypto_wizards_zscores_history: {path}")
    if not imported_quality.empty:
        print(
            imported_quality[
                ["pair", "interval", "history_rows", "research_usable", "execution_usable", "quality_blockers"]
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
        raise SystemExit(f"input JSON must be an object response from /v1beta/backtest: {input_path}")

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
    _print_imported_pair_quality([path], pair_dir, label="imported_crypto_wizards_backtest_histories")
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
    ecm_fields = report[(report["type"] == "field") & report["name"].isin({"ecm_x", "ecm_y", "ecm_strength"})]
    missing_ecm = sorted(ecm_fields.loc[~ecm_fields["present_in_live"], "name"].astype(str))
    print(f"live_ecm_fields_present: {not missing_ecm}")
    print(f"live_ecm_missing_fields: {','.join(missing_ecm) if missing_ecm else 'none'}")
    print(f"coverage_report: {ROOT / 'reports' / 'crypto_wizards_live_coverage.csv'}")


def crypto_wizards_live_coverage_report() -> pd.DataFrame:
    live_dictionary = ROOT / "docs" / "crypto_wizards_live_field_dictionary.csv"
    if not live_dictionary.exists():
        raise SystemExit("missing docs/crypto_wizards_live_field_dictionary.csv")

    live_fields = pd.read_csv(live_dictionary)
    normalized_live, live_sources = _canonical_live_fields(live_fields["field"].dropna().astype(str))
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
                "pair_detail_present": bool(required) and set(required).issubset(pair_detail_fields),
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
        sources.setdefault("conditional_probabilities", sources.get("conditional_probability_distortion", []))
    if "conditional_probabilities" in canonical_fields:
        canonical_fields.add("conditional_probability_distortion")
        sources.setdefault("conditional_probability_distortion", sources.get("conditional_probabilities", []))
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
            source_pair = [f"pair_detail:{snapshot.pair}:u1_given_u2", f"pair_detail:{snapshot.pair}:u2_given_u1"]
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
    shutil.copyfile(input_path, output_payload)

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
        report["blockers"] = [*report["blockers"], f"invalid_dydx_order_client_adapter:{order_adapter_error}"]
    elif adapter_contract["configured"] and adapter_contract["valid"] and not adapter_contract["exchange_submission_capable"]:
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


def dydx_execution_checklist_report(output_path: Path | None = None) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "dydx_execution_checklist.csv"
    config = DydxNetworkConfig.paper_testnet_from_env()
    indexer_wired = build_dydx_indexer_adapter(config) is not None
    order_client, order_adapter_error = _load_dydx_order_client_adapter()
    adapter_contract = validate_dydx_order_client_adapter()
    adapter_submission_capable = bool(adapter_contract["valid"] and adapter_contract["exchange_submission_capable"])
    readiness = dydx_readiness_report(
        config=config,
        order_client_wired=order_client is not None and adapter_submission_capable,
        indexer_adapter_wired=indexer_wired,
    )
    acceptance_path = _acceptance_report_path()
    strategy_ready = False
    two_leg_passing_pairs = 0
    production_eligible = 0
    if acceptance_path.exists():
        acceptance = pd.read_csv(acceptance_path)
        production_eligible = int(acceptance.get("production_eligible", pd.Series(dtype=bool)).fillna(False).sum())
        two_leg_passing_pairs = _max_int_column(acceptance, "two_leg_passing_pairs")
        strategy_ready = production_eligible > 0 and two_leg_passing_pairs > 0

    rows = [
        _execution_check_row(
            step="indexer_market_data",
            ready=bool(readiness["dydx_indexer_adapter_wired"]),
            blocker="" if readiness["dydx_indexer_adapter_wired"] else "missing_dydx_indexer_adapter",
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
            next_action="paper submission is allowed by current gates" if paper_ready else "do not submit paper orders",
        )
    )
    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, output)
    return frame


def strategy_acceptance_checklist_report(output_path: Path | None = None) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "strategy_acceptance_checklist.csv"
    acceptance_path = _acceptance_report_path()
    results_path = reports / "experiment_results.csv"
    funding_coverage_path = reports / "funding_coverage.csv"
    funding_requirements_path = reports / "funding_requirements.csv"
    acceptance = _read_csv_or_empty(acceptance_path)
    results = _read_csv_or_empty(results_path)
    funding_coverage = _read_csv_or_empty(funding_coverage_path)

    evaluated_runs = int((results.get("status", pd.Series(dtype=str)) == "evaluated").sum()) if not results.empty else 0
    two_leg_runs = int((results.get("backtest_mode", pd.Series(dtype=str)) == "two_leg").sum()) if not results.empty else 0
    spread_runs = int((results.get("backtest_mode", pd.Series(dtype=str)) == "spread").sum()) if not results.empty else 0
    production_eligible = _sum_bool_column(acceptance, "production_eligible")
    preferred_eligible = _sum_bool_column(acceptance, "preferred_eligible")
    max_two_leg_pairs_tested = _max_int_column(acceptance, "two_leg_pairs_tested")
    max_two_leg_execution_input_pairs = _max_int_column(acceptance, "two_leg_execution_input_pairs")
    max_two_leg_passing_pairs = _max_int_column(acceptance, "two_leg_passing_pairs")
    max_total_trades = _max_int_column(acceptance, "total_trades")
    required_cost_buckets = _required_cost_buckets_from_acceptance(acceptance)
    missing_cost_buckets = sorted(required_cost_buckets.difference(_cost_buckets_from_results(results)))
    blocker_counts = _acceptance_blocker_counts(acceptance)
    top_blockers = ";".join(f"{name}:{count}" for name, count in blocker_counts[:5])
    two_leg_input_blocker = _two_leg_execution_input_blocker(acceptance)
    missing_two_leg_inputs = _missing_two_leg_inputs_from_acceptance(acceptance)
    funding_missing = "funding_x" in missing_two_leg_inputs or "funding_y" in missing_two_leg_inputs
    funding_requirements = _funding_requirements_for_preflight(funding_missing, funding_requirements_path)
    funding_preflight = _funding_preflight_status(
        funding_coverage,
        funding_missing,
        funding_coverage_path,
        funding_requirements,
        funding_requirements_path,
    )

    rows = [
        _execution_check_row(
            step="experiment_results",
            ready=evaluated_runs > 0,
            blocker="" if evaluated_runs > 0 else "missing_evaluated_experiments",
            evidence=f"results_exists={results_path.exists()};evaluated_runs={evaluated_runs};spread_runs={spread_runs};two_leg_runs={two_leg_runs}",
            next_action="continue acceptance diagnostics" if evaluated_runs > 0 else "run experiments on real pair-detail history",
        ),
        _execution_check_row(
            step="two_leg_coverage",
            ready=max_two_leg_pairs_tested >= 2 and two_leg_runs > 0,
            blocker="" if max_two_leg_pairs_tested >= 2 and two_leg_runs > 0 else "missing_two_leg_backtests",
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
            step="funding_preflight",
            ready=funding_preflight["ready"],
            blocker=funding_preflight["blocker"],
            evidence=funding_preflight["evidence"],
            next_action=funding_preflight["next_action"],
        ),
        _execution_check_row(
            step="cost_bucket_coverage",
            ready=not missing_cost_buckets and bool(required_cost_buckets),
            blocker="" if not missing_cost_buckets and bool(required_cost_buckets) else "missing_required_cost_buckets",
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


def strategy_failure_attribution_report(output_path: Path | None = None) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "strategy_failure_attribution.csv"
    results = _read_csv_or_empty(reports / "experiment_results.csv")
    acceptance = _read_csv_or_empty(_acceptance_report_path())
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
    acceptance_by_id = acceptance.set_index("strategy_id") if "strategy_id" in acceptance.columns and not acceptance.empty else pd.DataFrame()
    for (strategy_id, strategy_name, family), group in results.groupby(["strategy_id", "strategy_name", "family"], dropna=False):
        evaluated = group[group.get("status", pd.Series(dtype=str)) == "evaluated"].copy()
        skipped = group[group.get("status", pd.Series(dtype=str)) != "evaluated"].copy()
        eligible = int(evaluated.get("eligible", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not evaluated.empty else 0
        reason_counts = _reason_counts(evaluated.get("reason", pd.Series(dtype=str))) if not evaluated.empty else []
        skip_counts = _reason_counts(skipped.get("reason", pd.Series(dtype=str))) if not skipped.empty else []
        missing_columns = sorted(
            {
                item
                for reason, _ in skip_counts
                if reason.startswith("missing_columns:")
                for item in reason.replace("missing_columns:", "").split(",")
                if item
            }
        )
        pair_count = int(evaluated.get("pair", pd.Series(dtype=str)).nunique()) if not evaluated.empty else 0
        total_trades = int(pd.to_numeric(evaluated.get("trades", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        max_trades = int(pd.to_numeric(evaluated.get("trades", pd.Series(dtype=float)), errors="coerce").fillna(0).max()) if not evaluated.empty else 0
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
                "evaluated_runs": int(len(evaluated)),
                "skipped_runs": int(len(skipped)),
                "pairs_tested": pair_count,
                "eligible_runs": eligible,
                "total_trades": total_trades,
                "max_trades_per_run": max_trades,
                "median_profit_factor": median_pf,
                "median_sharpe": median_sharpe,
                "worst_drawdown": worst_drawdown,
                "median_expectancy": median_expectancy,
                "median_cost_drag": median_cost_drag,
                "top_run_failures": ";".join(f"{reason}:{count}" for reason, count in reason_counts[:5]),
                "top_skip_reasons": ";".join(f"{reason}:{count}" for reason, count in skip_counts[:5]),
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


def research_unblock_plan_report(output_path: Path | None = None) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "research_unblock_plan.csv"
    failures = strategy_failure_attribution_report(reports / "strategy_failure_attribution.csv")
    results = _read_csv_or_empty(reports / "experiment_results.csv")
    quality = _read_csv_or_empty(reports / "pair_detail_quality_report.csv")
    acceptance = _read_csv_or_empty(_acceptance_report_path())

    rows: list[dict[str, object]] = []
    evaluated = results[results.get("status", pd.Series(dtype=str)) == "evaluated"].copy() if not results.empty else pd.DataFrame()
    if not evaluated.empty:
        trades = pd.to_numeric(evaluated.get("trades", pd.Series(dtype=float)), errors="coerce").fillna(0)
        observations = pd.to_numeric(evaluated.get("observations", pd.Series(dtype=float)), errors="coerce").fillna(0)
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
        best_threshold["_max_trades"] = pd.to_numeric(best_threshold.get("max_trades", pd.Series(dtype=float)), errors="coerce").fillna(0)
        best_threshold["_passing_pairs"] = pd.to_numeric(
            best_threshold.get("passing_pairs", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0)
        best_threshold = best_threshold.sort_values(["_passing_pairs", "_max_trades"], ascending=[False, False]).iloc[0]
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
        for column, item in sorted(missing_counts.items(), key=lambda kv: (-len(kv[1]["strategies"]), kv[0])):
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
        research_usable = int(quality.get("research_usable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        execution_usable = int(quality.get("execution_usable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
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
        production_eligible = int(acceptance["production_eligible"].fillna(False).astype(bool).sum())
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
    datasets = [PairDataset(dataset.pair, classify_regimes(dataset.frame, RegimeConfig(preserve_existing=True))) for dataset in datasets]
    rows: list[dict[str, object]] = []
    cost_buckets = ExperimentConfig().cost_buckets
    for dataset in datasets:
        frame = dataset.frame
        if not {"price_x", "price_y", "zscore"}.issubset(frame.columns):
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


def zscore_threshold_sweep_summary(frame: pd.DataFrame, output_path: Path | None = None) -> pd.DataFrame:
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
        .sort_values(["passing_pairs", "median_trades", "median_profit_factor"], ascending=[False, False, False])
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
    frame = zscore_threshold_sweep_report(input_dir=input_dir, funding_path=funding_path, output_path=output)
    print(frame.to_string(index=False))
    print(f"zscore_threshold_sweep: {output}")
    print(f"zscore_threshold_sweep_summary: {ROOT / 'reports' / 'zscore_threshold_sweep_summary.csv'}")


def dydx_pair_expansion_plan_report(
    *,
    max_pairs: int = 10,
    limit: int = 1000,
    indexer_base: str = DEFAULT_INDEXER_BASE,
    indexer_scheme: str = "",
    output_path: Path | None = None,
) -> pd.DataFrame:
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
    for order, (asset_x, asset_y) in enumerate(DEFAULT_DYDX_EXPANSION_PAIRS):
        left = _normalize_dydx_market(asset_x)
        right = _normalize_dydx_market(asset_y)
        pair_key = frozenset({left, right})
        already_tested = pair_key in tested_pairs
        fetched_info = fetched_pairs.get(pair_key, {})
        already_fetched = bool(fetched_info)
        fresh_candidate = not already_tested and not already_fetched
        risk_reasons = [risky_markets[market] for market in (left, right) if market in risky_markets]
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

    ranked_fresh = sorted(fresh_candidates, key=lambda row: (int(row["risk_score"]), int(row["order"])))
    rank_by_key = {candidate["pair_key"]: rank for rank, candidate in enumerate(ranked_fresh, start=1)}

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        left = str(candidate["asset_x"])
        right = str(candidate["asset_y"])
        pair_key = candidate["pair_key"]
        already_tested = bool(candidate["already_tested"])
        already_fetched = bool(candidate["already_fetched"])
        fresh_candidate = bool(candidate["fresh_candidate"])
        fetched_info = candidate["fetched_info"] if isinstance(candidate["fetched_info"], dict) else {}
        rank = rank_by_key.get(pair_key, "")
        if fresh_candidate and rank and int(rank) > max_pairs:
            continue
        pair_id = _pair_id_from_markets(left, right)
        missing_markets = sorted(market for market in (left, right) if market not in covered_markets)
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
    left, right, resolved_pair_id = _resolve_long_history_pair(pair=pair, asset_x=asset_x, asset_y=asset_y, pair_id=pair_id)
    end_time = _parse_iso_datetime(to_iso) if to_iso else datetime.now(timezone.utc)
    step = _resolution_timedelta(resolution) * limit
    requested_indexer_base = _indexer_base_with_scheme(indexer_base, indexer_scheme)
    rows: list[dict[str, object]] = []
    for window in range(1, windows + 1):
        window_to = end_time - step * (window - 1)
        window_from = window_to - step
        window_dir = ROOT / "data" / "raw" / "dydx_long_history" / resolved_pair_id / f"window_{window:03d}"
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
    resolved_pair_id = _resolve_long_history_pair(pair=pair, asset_x=asset_x, asset_y=asset_y, pair_id=pair_id)[2]
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
    requests = plan[plan.get("method", pd.Series(dtype=str)).astype(str).str.upper() == "GET"].copy()
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
    tested = plan["already_tested"].map(_coerce_bool) if "already_tested" in plan.columns else pd.Series(False, index=plan.index)
    fetched = plan["already_fetched"].map(_coerce_bool) if "already_fetched" in plan.columns else pd.Series(False, index=plan.index)
    fresh = plan[(~tested) & (~fetched)].copy()
    if "rank" in fresh.columns:
        fresh["_rank"] = pd.to_numeric(fresh["rank"], errors="coerce")
        fresh = fresh.sort_values("_rank")
    rows: list[dict[str, object]] = []
    for _, row in fresh.head(max_pairs).iterrows():
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
        strategy_failure_attribution_report()
        research_unblock_plan_report()
        priority_readiness_report()
    return frame


def run_dydx_local_pair_universe(
    *,
    input_dir: Path | None = None,
    pair_output_dir: Path | None = None,
    funding_output_path: Path | None = None,
    zscore_window: int = 320,
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
    markets = sorted({_normalize_dydx_market(path.name.split("_5MINS_candles.json")[0]) for path in candle_paths})
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
        raise SystemExit(f"research-usable quality report is missing required columns: {quality_path}")

    subset_dir = output_dir or ROOT / "work" / "p2_rerun_subset"
    subset_dir.mkdir(parents=True, exist_ok=True)
    for existing in subset_dir.glob("*.json"):
        existing.unlink()

    selected = quality[quality["research_usable"].fillna(False).astype(bool)].copy()
    if selected.empty:
        raise SystemExit(f"no research-usable pair-detail histories found in {quality_path}")
    selected["_execution_usable"] = selected.get("execution_usable", pd.Series(dtype=bool)).fillna(False).astype(bool)
    selected["_history_rows"] = pd.to_numeric(selected.get("history_rows", pd.Series(dtype=float)), errors="coerce").fillna(0)
    selected = (
        selected.sort_values(["pair", "_execution_usable", "_history_rows"], ascending=[True, False, False])
        .drop_duplicates(["pair"], keep="first")
    )

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
            replacement = source.with_name(source.name.replace("_dydx_candles_derived_history.json", "_dydx_long_history_derived_history.json"))
            if replacement.exists():
                chosen = replacement
        status = "copied"
        detail = ""
        if not chosen.exists():
            status = "missing"
            detail = "source_missing"
        else:
            target = subset_dir / chosen.name
            target.write_bytes(chosen.read_bytes())
            detail = "long_history_replacement" if replacement is not None and chosen == replacement else "quality_report_source"
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
    frame = materialize_p2_rerun_subset(input_dir=input_dir, output_dir=output_dir, quality_report_path=quality_report_path)
    print(frame.to_string(index=False))
    print(f"p2_rerun_subset: {output_dir or (ROOT / 'work' / 'p2_rerun_subset')}")
    print(f"p2_rerun_subset_manifest: {ROOT / 'reports' / 'p2_rerun_subset_manifest.csv'}")


def print_dydx_local_pair_universe(
    *,
    input_dir: Path | None = None,
    pair_output_dir: Path | None = None,
    funding_output_path: Path | None = None,
    zscore_window: int = 320,
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
    )
    print(frame.to_string(index=False))
    print(f"dydx_pair_expansion_run: {output}")


def priority_spine_dashboard_report(
    readiness: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "priority_spine_dashboard.csv"
    readiness = readiness if readiness is not None else priority_readiness_report()
    gates = readiness.set_index("gate") if not readiness.empty else pd.DataFrame()
    capture = _read_csv_or_empty(reports / "pair_detail_capture_checklist.csv")
    quality = _read_csv_or_empty(reports / "pair_detail_quality_report.csv")
    strategy = _read_csv_or_empty(reports / "strategy_acceptance_checklist.csv")
    research_unblock_path = reports / "research_unblock_plan.csv"
    has_research_unblock = research_unblock_path.exists()
    dydx = _read_csv_or_empty(reports / "dydx_execution_checklist.csv")
    paper = _read_csv_or_empty(reports / "paper_execution_preflight.csv")
    learning = _read_csv_or_empty(reports / "learning_event_summary.csv")

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
            ready=_gate_ready_from_index(gates, "paper_execution_gate"),
            blocker=_gate_value(gates, "paper_execution_gate", "blocker"),
            key_metric=_checklist_dashboard_metric(paper) if not paper.empty else _gate_value(gates, "paper_execution_gate", "evidence"),
            source_report="reports/paper_execution_preflight.csv" if not paper.empty else "reports/priority_readiness.csv",
            next_action=_checklist_first_blocked_next_action(paper)
            or _gate_value(gates, "paper_execution_gate", "next_action"),
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
    readiness = priority_readiness_report()
    paper_execution_preflight_report(ROOT / "reports" / "paper_execution_preflight.csv")
    frame = priority_spine_dashboard_report(readiness, output)
    print(frame.to_string(index=False))
    print(f"priority_spine_dashboard: {output}")


def priority_runbook(output_path: Path | None = None) -> Path:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "priority_runbook.md"
    readiness = priority_readiness_report()
    paper_execution_preflight_report(reports / "paper_execution_preflight.csv")
    dashboard = priority_spine_dashboard_report(readiness, reports / "priority_spine_dashboard.csv")
    gap_test = priority_gap_test_report(readiness, reports / "priority_gap_test.csv")
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
        lines.extend(["| Rank | Gate | Depends On | Blocker | Command/Action |", "|---:|---|---|---|---|"])
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
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _funding_requirement_market_arg() -> str:
    requirements = _read_csv_or_empty(ROOT / "reports" / "funding_requirements.csv")
    markets = _semicolon_values(requirements.get("required_markets", pd.Series(dtype=str)))
    return ",".join(markets) if markets else "ETH-USD,BTC-USD,SOL-USD"


def print_priority_runbook() -> None:
    output = priority_runbook()
    print(f"priority_runbook: {output}")


def paper_execution_preflight_report(output_path: Path | None = None) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "paper_execution_preflight.csv"
    readiness = priority_readiness_report()
    gates = readiness.set_index("gate") if not readiness.empty else pd.DataFrame()
    strategy = _read_csv_or_empty(reports / "strategy_acceptance_checklist.csv")
    dydx = _read_csv_or_empty(reports / "dydx_execution_checklist.csv")
    paper_journal = reports / "paper_trading_journal.csv"

    strategy_ready = _gate_ready_from_index(gates, "strategy_acceptance")
    dydx_ready = _gate_ready_from_index(gates, "dydx_testnet_readiness")
    paper_ready = _gate_ready_from_index(gates, "paper_execution_gate")
    strategy_next_action = (
        _checklist_first_blocked_next_action(strategy) or _gate_value(gates, "strategy_acceptance", "next_action")
    )
    dydx_next_action = _checklist_first_blocked_next_action(dydx) or _gate_value(gates, "dydx_testnet_readiness", "next_action")
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
            ready=paper_ready,
            blocker="" if paper_ready else _gate_value(gates, "paper_execution_gate", "blocker"),
            evidence=_gate_value(gates, "paper_execution_gate", "evidence"),
            next_action="paper-plan may create and submit research-gated paper orders"
            if paper_ready
            else "do not submit paper orders until strategy and dYdX dependencies are ready",
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
    ]
    frame = pd.DataFrame(rows)
    _write_csv_atomic(frame, output)
    return frame


def print_paper_execution_preflight() -> None:
    output = ROOT / "reports" / "paper_execution_preflight.csv"
    frame = paper_execution_preflight_report(output)
    print(frame.to_string(index=False))
    print(f"paper_execution_preflight: {output}")


def paper_venue_preflight_report(
    pair: str | None = None,
    output_path: Path | None = None,
    max_pairs: int = 25,
) -> pd.DataFrame:
    """Build compact per-venue paper readiness for one pair or top pairs."""
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "paper_venue_preflight.csv"
    universe = _read_csv_or_empty(ROOT / "data" / "processed" / "pair_universe.csv")
    rows: list[dict[str, object]] = []

    if pair:
        pair_rows: list[str] = [_normalize_dydx_pair(pair)]
    else:
        pair_rows = []
        if not universe.empty and "pair" in universe.columns:
            score_col = pd.to_numeric(universe.get("combined_score", pd.Series(dtype=float)), errors="coerce")
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
        options = _build_paper_venue_options(candidate_pair)
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
            blockers: list[str] = [item for item in str(venue_row.get("blockers", "")).split(";") if item]

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
                    blockers.append(f"dydx_order_client_adapter_invalid:{adapter_contract.get('error')}")
                if not indexer_ready:
                    blockers.append("missing_dydx_indexer_adapter")
                if not adapter_ready:
                    blockers.append("dydx_not_submission_ready")
                ready_for_submission = execution_ready and adapter_ready and indexer_ready and not bool(config.paper_trading_blockers())

                rows.append(
                    {
                        "pair": candidate_pair,
                        "venue": venue,
                        "preference": preference,
                        "venue_lanes": venue_lanes,
                        "execution_ready": execution_ready,
                        "adapter_ready": bool(adapter_ready),
                        "ready_for_submission": bool(ready_for_submission),
                        "contract_configured": bool(adapter_contract.get("configured")),
                        "contract_valid": bool(adapter_contract.get("valid")),
                        "exchange_submission_capable": bool(adapter_contract.get("exchange_submission_capable")),
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
                blockers.append(f"{venue}_order_client_adapter_invalid:{adapter_contract.get('error')}")
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
                    "contract_configured": bool(adapter_contract.get("configured")),
                    "contract_valid": bool(adapter_contract.get("valid")),
                    "exchange_submission_capable": bool(adapter_contract.get("exchange_submission_capable")),
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


def priority_gap_test_report(
    readiness: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "priority_gap_test.csv"
    readiness = readiness if readiness is not None else priority_readiness_report()
    dashboard = priority_spine_dashboard_report(readiness)
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


def print_gap_test() -> None:
    output = ROOT / "reports" / "priority_gap_test.csv"
    readiness = priority_readiness_report()
    paper_execution_preflight_report(ROOT / "reports" / "paper_execution_preflight.csv")
    frame = priority_gap_test_report(readiness, output)
    print(frame.to_string(index=False))
    print(f"priority_gap_test: {output}")


def print_gap_analysis_checklist(run_dir: Path | None = None) -> tuple[Path, Path]:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir or (reports / "gap_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")
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
    checkpoint_md.write_text("\n".join(lines), encoding="utf-8")
    latest_md.write_text(checkpoint_md.read_text(encoding="utf-8"), encoding="utf-8")

    index_path = reports / "gap_analysis_index.csv"
    index_frame = _read_csv_or_empty(index_path)
    if index_frame.empty:
        index_frame = pd.DataFrame(columns=["run_id", "timestamp_utc", "open_gaps", "pass_gates", "critical", "high", "medium"])
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


def print_pre_mortem_checklist(run_dir: Path | None = None) -> tuple[Path, Path]:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir or (reports / "pre_mortem")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")
    run_id = f"pre_mortem_{timestamp}"

    gap_frame = priority_gap_test_report()
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
    checkpoint_md.write_text("\n".join(lines), encoding="utf-8")
    latest_md.write_text(checkpoint_md.read_text(encoding="utf-8"), encoding="utf-8")

    index_path = reports / "pre_mortem_index.csv"
    index_frame = _read_csv_or_empty(index_path)
    if index_frame.empty:
        index_frame = pd.DataFrame(columns=["run_id", "timestamp_utc", "open_gaps", "pass_gates", "critical", "high", "medium"])
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


def print_post_mortem_checklist(run_dir: Path | None = None) -> tuple[Path, Path]:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir or (reports / "post_mortem")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")
    run_id = f"post_mortem_{timestamp}"

    previous_path = reports / "post_mortem" / "latest_post_mortem.csv"
    previous = _read_csv_or_empty(previous_path)
    previous_by_area: dict[str, str] = {}
    if not previous.empty and "area" in previous.columns and "status" in previous.columns:
        previous_by_area = {
            str(area): str(status)
            for area, status in zip(previous["area"].astype(str), previous["status"].astype(str))
        }

    gap_frame = priority_gap_test_report()
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
            trajectory = _post_mortem_status_trajectory(area=area, current_status=status, previous_status=prev_status)
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
    checkpoint_md.write_text("\n".join(lines), encoding="utf-8")
    latest_md.write_text(checkpoint_md.read_text(encoding="utf-8"), encoding="utf-8")

    index_path = reports / "post_mortem_index.csv"
    index_frame = _read_csv_or_empty(index_path)
    if index_frame.empty:
        index_frame = pd.DataFrame(
            columns=["run_id", "timestamp_utc", "open_gaps", "pass_gates", "critical", "high", "medium"]
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
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")
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
    checkpoint_md.write_text("\n".join(lines), encoding="utf-8")
    latest_md.write_text(checkpoint_md.read_text(encoding="utf-8"), encoding="utf-8")

    index_path = reports / "red_team_index.csv"
    index_frame = _read_csv_or_empty(index_path)
    if index_frame.empty:
        index_frame = pd.DataFrame(columns=["run_id", "timestamp_utc", "open_gaps", "pass_gates", "critical", "high", "medium"])
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
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")
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
        local["run_id"] = local.get("run_id", pd.Series(dtype=str)).fillna("").astype(str).replace({"": run_id})
        local["source_checkpoint"] = source
        local["timestamp_utc"] = local.get("timestamp_utc", pd.Series([timestamp] * len(local))).fillna(timestamp)
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

    def _severity_rank(value: object) -> int:
        return {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4, "": 5}.get(str(value).strip().lower(), 5)

    worklist = all_checkpoints[all_checkpoints["status"] != "pass"].copy()
    if not worklist.empty:
        worklist["_priority_rank"] = worklist["priority"].map(_priority_sort_key)
        worklist["_severity_rank"] = worklist["severity"].map(_severity_rank)
        worklist = worklist.sort_values(["_severity_rank", "_priority_rank", "source_checkpoint", "area"]).reset_index(drop=True)
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
    checkpoint_md.write_text("\n".join(lines), encoding="utf-8")
    latest_md.write_text(checkpoint_md.read_text(encoding="utf-8"), encoding="utf-8")

    index_path = reports / "supreme_team_index.csv"
    index_frame = _read_csv_or_empty(index_path)
    if index_frame.empty:
        index_frame = pd.DataFrame(
            columns=["run_id", "timestamp_utc", "open_actions", "pass_gates", "critical", "high", "medium"]
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
    grouped = (
        evaluated.groupby(["strategy_id", "strategy_name", "pair", "cost_bucket"], as_index=False)
        .agg(
            trades=("trades", "max"),
            observations=("observations", "max"),
        )
    )

    rows: list[dict[str, object]] = []
    for (strategy_id, strategy_name, pair), scope in grouped.groupby(["strategy_id", "strategy_name", "pair"]):
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
        missing_cost_buckets = ";".join(sorted(required_buckets.difference(present_buckets))) or "none"
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
    print(frame.to_string(index=False))
    print(f"strategy_trade_count_gap: {output}")


def print_dydx_execution_checklist(output_path: Path | None = None) -> None:
    output = output_path or ROOT / "reports" / "dydx_execution_checklist.csv"
    frame = dydx_execution_checklist_report(output)
    print(frame.to_string(index=False))
    print(f"dydx_execution_checklist: {output}")


def priority_readiness_report(output_path: Path | None = None) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "priority_readiness.csv"
    rows: list[dict[str, object]] = []

    live_dictionary = ROOT / "docs" / "crypto_wizards_live_field_dictionary.csv"
    raw_payloads = sorted(
        path
        for path in (ROOT / "data" / "raw").glob("*.json")
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
            next_action="crawl or import Crypto Wizards payloads" if not live_ready else "continue field coverage checks",
        )
    )

    pair_detail_dir = ROOT / "data" / "raw" / "pair_details"
    history_rows = pair_detail_history_coverage(pair_detail_dir) if pair_detail_dir.exists() else []
    experiment_ready = [row for row in history_rows if bool(row.get("experiment_ready"))]
    ecm_ready = [row for row in history_rows if bool(row.get("ecm_history_ready"))]
    two_leg_ready = [row for row in history_rows if bool(row.get("two_leg_execution_ready"))]
    quality_rows = pair_detail_quality_report(pair_detail_dir) if pair_detail_dir.exists() else []
    _write_csv_atomic(
        pd.DataFrame(quality_rows, columns=PAIR_DETAIL_QUALITY_COLUMNS),
        reports / "pair_detail_quality_report.csv",
    )
    research_usable = [row for row in quality_rows if bool(row.get("research_usable"))]
    execution_usable = [row for row in quality_rows if bool(row.get("execution_usable"))]
    rows.append(
        _readiness_row(
            priority="P1",
            gate="pair_detail_history",
            ready=bool(experiment_ready and ecm_ready),
            evidence=f"snapshots={len(history_rows)};experiment_ready={len(experiment_ready)};ecm_ready={len(ecm_ready)}",
            blocker="" if experiment_ready and ecm_ready else "missing_spread_zscore_or_ecm_history",
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
                f"execution_usable={len(execution_usable)}"
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
            {**row, "experiment_ready": _coerce_bool(row.get("experiment_ready")), "ecm_history_ready": _coerce_bool(row.get("ecm_history_ready")), "two_leg_execution_ready": _coerce_bool(row.get("two_leg_execution_ready"))}
            for row in cached_capture.to_dict("records")
        ]
    else:
        capture_rows = pair_detail_capture_audit(pair_detail_dir) if pair_detail_dir.exists() else []
    if capture_rows:
            _write_csv_atomic(
                pd.DataFrame(capture_rows, columns=PAIR_DETAIL_CAPTURE_AUDIT_COLUMNS),
                capture_report_path,
            )
    capture_experiment_ready = [row for row in capture_rows if bool(row.get("experiment_ready"))]
    capture_ecm_ready = [row for row in capture_rows if bool(row.get("ecm_history_ready"))]
    capture_two_leg_ready = [row for row in capture_rows if bool(row.get("two_leg_execution_ready"))]
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

    acceptance_path = _acceptance_report_path()
    acceptance_checklist_path = reports / "strategy_acceptance_checklist.csv"
    strategy_acceptance_checklist_report(acceptance_checklist_path)
    research_unblock_path = reports / "research_unblock_plan.csv"
    research_unblock_plan_report(research_unblock_path)
    if acceptance_path.exists():
        acceptance = pd.read_csv(acceptance_path)
        production_ready = int(acceptance.get("production_eligible", pd.Series(dtype=bool)).fillna(False).sum())
        preferred_ready = int(acceptance.get("preferred_eligible", pd.Series(dtype=bool)).fillna(False).sum())
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

    config = DydxNetworkConfig.paper_testnet_from_env()
    order_client, order_adapter_error = _load_dydx_order_client_adapter()
    adapter_contract = validate_dydx_order_client_adapter()
    order_adapter_loaded = (
        order_client is not None
        and not order_adapter_error
    )
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
    dydx_execution_checklist_report(dydx_checklist_path)
    dydx_blockers = list(dydx_report.get("blockers", []))
    if order_adapter_error or (adapter_contract["configured"] and not adapter_contract["valid"]):
        dydx_blockers.append("invalid_dydx_order_client_adapter")
    elif adapter_contract["configured"] and adapter_contract["valid"] and not adapter_contract["exchange_submission_capable"]:
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
            next_action="paper trade only accepted strategies" if paper_gate_ready else "do not submit paper orders yet",
        )
    )

    paper_journal = reports / "paper_trading_journal.csv"
    trade_store = ROOT / "data" / "meta_learning" / "trades.jsonl"
    learning_summary_path = reports / "learning_event_summary.csv"
    write_learning_event_summary_report(paper_journal, trade_store, learning_summary_path)
    learning_summary = _read_csv_or_empty(learning_summary_path)
    combined_learning = learning_summary[learning_summary.get("source", pd.Series(dtype=str)) == "combined"]
    combined_row = combined_learning.iloc[0] if not combined_learning.empty else pd.Series(dtype=object)
    paper_journal_rows = _csv_row_count(paper_journal)
    trade_store_rows = _jsonl_row_count(trade_store)
    learning_events = int(combined_row.get("events", 0) or 0)
    learning_outcomes = int(combined_row.get("outcome_events", 0) or 0)
    learning_audit_only = int(combined_row.get("audit_only_events", 0) or 0)
    learning_outcomes_remaining = int(combined_row.get("outcome_events_remaining", 100) or 0)
    learning_ready_for_modeling = bool(combined_row.get("ready_for_modeling", False))
    learning_ready = learning_ready_for_modeling
    learning_blocker = "missing_learning_events" if learning_events == 0 else "missing_model_ready_outcomes"
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
    priority_spine_dashboard_report(frame, reports / "priority_spine_dashboard.csv")
    priority_gap_test_report(frame, reports / "priority_gap_test.csv")
    return frame


def print_priority_readiness(output_path: Path | None = None) -> None:
    frame = priority_readiness_report(output_path)
    print(frame.to_string(index=False))
    print(f"priority_readiness_report: {output_path or ROOT / 'reports' / 'priority_readiness.csv'}")
    print(f"priority_action_plan: {ROOT / 'reports' / 'priority_action_plan.csv'}")


def priority_action_plan(readiness: pd.DataFrame | None = None, output_path: Path | None = None) -> pd.DataFrame:
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
        for _, row in journal.loc[statuses == "paper_ready"].iterrows():
            fill_statuses: list[str] = []
            try:
                parsed_fills = json.loads(str(row.get("fills_json", "")) or "[]")
            except json.JSONDecodeError:
                parsed_fills = []
            if isinstance(parsed_fills, list):
                for item in parsed_fills:
                    if isinstance(item, dict) and item.get("status") is not None:
                        fill_statuses.append(str(item["status"]))
            if not any(status == "paper_submitted" for status in fill_statuses):
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
    frame = seed_learning_outcome_template_from_paper_journal(input_path=input_path, output_path=output)
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
        raise SystemExit("trade-timing-comparison-report requires --input-dir pointing to a trades CSV")
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


def print_learning_outcome_template_check(input_path: Path | None = None, output_path: Path | None = None) -> None:
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
                    "trade_store": str(trade_store_path or ROOT / "data" / "meta_learning" / "trades.jsonl"),
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


def print_import_learning_outcomes(input_path: Path | None = None, output_path: Path | None = None) -> None:
    frame = import_learning_outcomes_from_template(input_path=input_path, report_path=output_path)
    output = output_path or ROOT / "reports" / "learning_outcome_import_report.csv"
    print(frame.to_string(index=False))
    print(f"learning_outcome_import_report: {output}")


def _learning_outcome_template_validation(frame: pd.DataFrame) -> tuple[list[str], list[int], list[str]]:
    missing_columns = [column for column in LEARNING_OUTCOME_REQUIRED_COLUMNS if column not in frame.columns]
    invalid_rows: list[str] = []
    ready_indices: list[int] = []
    if missing_columns or frame.empty:
        return missing_columns, ready_indices, invalid_rows
    for index, row in frame.iterrows():
        missing = [column for column in LEARNING_OUTCOME_REQUIRED_COLUMNS if str(row.get(column, "")).strip() == ""]
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
    timestamp = pd.Timestamp.utcnow().to_pydatetime()
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
        raise SystemExit("append-learning-outcome requires --pair, --strategy-id, and --realized-return")
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


def _execution_check_row(step: str, ready: bool, blocker: str, evidence: str, next_action: str) -> dict[str, object]:
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
    ready = int(frame.get("research_spine_ready", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    completeness = pd.to_numeric(frame.get("capture_completeness_score", pd.Series(dtype=float)), errors="coerce")
    if completeness.dropna().empty:
        best_row = frame.iloc[0]
    else:
        best_row = frame.loc[completeness.idxmax()]
    next_focus = str(best_row.get("next_capture_focus", "unknown") or "unknown")
    missing_value = best_row.get("missing_required_fields", "")
    missing = "" if pd.isna(missing_value) else str(missing_value or "")
    best_score = "" if completeness.dropna().empty else f";best_completeness={float(completeness.max()):.2f}"
    return f"captures={len(frame)};research_spine_ready={ready}{best_score};next_focus={next_focus};missing={missing or 'none'}"


def _capture_quality_dashboard_metric(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "quality_rows=0"
    research_usable = int(frame.get("research_usable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    execution_usable = int(frame.get("execution_usable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
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
        "strategy_acceptance": "production-eligible strategy with required two-leg base/stress results across multiple pairs",
        "dydx_testnet_readiness": "submit flag, credentials, SDK, indexer, and authenticated order adapter all ready",
        "paper_execution_gate": "strategy_acceptance ready and dydx_testnet_readiness ready",
        "learning_event_store": "paper journal or trade store contains outcome events for later modeling",
    }
    return proofs.get(area, "documented readiness evidence")


def _pre_mortem_question(priority: str, area: str, severity: str) -> str:
    if severity == "critical":
        return (
            f"{priority} {area}: If we move to execution without this, what hard failure would most likely break us first?"
        )
    if severity == "high":
        return f"{priority} {area}: What would fail and what signal would we watch first?"
    return f"{priority} {area}: What is the most likely downside risk we are accepting by skipping this?"


def _pre_mortem_failure_mode(area: str, gap: str) -> str:
    if area == "crypto_wizards_capture":
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
    if area == "crypto_wizards_capture":
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
        return (
            f"{area} remained open in previous checks and continues to pose operational risk until evidence is completed."
        )
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
    total_pairs = int(len(coverage))
    blocked = coverage[~ready_values]
    blocked_pairs = ";".join(str(pair) for pair in blocked.get("pair", pd.Series(dtype=str)).dropna().unique())
    missing = ";".join(str(value) for value in blocked.get("missing", pd.Series(dtype=str)).dropna().unique())
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
    return ";".join(f"{blocker}:{count}" for blocker, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8])


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
    for _, row in frame.iterrows():
        parsed = _markets_from_pair_name(_md_text(row.get("pair", "")))
        if not parsed:
            continue
        research_usable = _coerce_bool(row.get("research_usable", False))
        blockers = _md_text(row.get("quality_blockers", ""))
        status = "research_usable" if research_usable else "quality_blocked"
        info[frozenset(parsed)] = {
            "quality_status": status,
            "quality_blockers": blockers,
        }
    return info


def _stale_market_risk_info() -> dict[str, str]:
    reports = ROOT / "reports"
    frame = _read_csv_or_empty(reports / "pair_detail_quality_report.csv")
    risks: dict[str, str] = {}
    if frame.empty or "pair" not in frame.columns:
        return risks
    for _, row in frame.iterrows():
        parsed = _markets_from_pair_name(_md_text(row.get("pair", "")))
        if not parsed:
            continue
        left, right = parsed
        blockers = _md_text(row.get("quality_blockers", ""))
        stale_x = _numeric_cell(row.get("stale_price_x_rate", 0.0))
        stale_y = _numeric_cell(row.get("stale_price_y_rate", 0.0))
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
            markets.update(_normalize_dydx_market(value) for value in coverage[column].dropna().astype(str) if value)
    return markets


def _markets_from_pair_name(pair: str) -> tuple[str, str] | None:
    text = str(pair).upper()
    if "-USD-" in text:
        left, right = text.split("-USD-", 1)
        if left and right:
            return _normalize_dydx_market(left), _normalize_dydx_market(right)
    parts = [part for part in re.split(r"[-_/]", text) if part and part != "USD"]
    if len(parts) >= 2:
        return _normalize_dydx_market(parts[0]), _normalize_dydx_market(parts[1])
    return None


def _normalize_dydx_market(asset: str) -> str:
    text = str(asset).upper().replace("/", "-").strip()
    parts = [part for part in text.split("-") if part]
    if len(parts) >= 2 and parts[-1] == "USD":
        return f"{parts[0]}-USD"
    if len(parts) == 1:
        return f"{parts[0]}-USD"
    return text


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


def _fetch_live_dydx_market_catalog(indexer_base: str = DEFAULT_INDEXER_BASE, max_markets: int = 300) -> dict[str, dict[str, object]]:
    requested_indexer_base = _indexer_base_with_scheme(indexer_base, "")
    url = f"{requested_indexer_base}/v4/perpetualMarkets?limit={max_markets}"
    response = requests.get(url, headers={"Content-Type": "application/json"}, timeout=20.0)
    response.raise_for_status()
    payload = response.json()
    markets = payload.get("markets", {})
    return markets if isinstance(markets, dict) else {}


def _live_market_selector_score(trades_24h: float, volume_24h: float) -> float:
    # Favor markets that are both frequently traded and meaningfully liquid.
    return round((np.log10(max(trades_24h, 0.0) + 1.0) * 0.6) + (np.log10(max(volume_24h, 0.0) + 1.0) * 0.4), 6)


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


def dydx_live_market_counts_report(
    *,
    indexer_base: str = DEFAULT_INDEXER_BASE,
    output_path: Path | None = None,
) -> pd.DataFrame:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = output_path or reports / "dydx_live_market_counts.csv"

    markets = _fetch_live_dydx_market_catalog(indexer_base=indexer_base)
    normalized_markets = {_normalize_dydx_market(name): meta for name, meta in markets.items() if "," not in str(name)}
    total_markets = len(normalized_markets)
    active_markets = {
        market: meta for market, meta in normalized_markets.items() if str(meta.get("status", "")).upper() == "ACTIVE"
    }
    active_cross_markets = {
        market: meta for market, meta in active_markets.items() if str(meta.get("marketType", "")).upper() == "CROSS"
    }
    excluded_markets = {market for market in active_cross_markets if market in DEFAULT_DYDX_LIVE_SELECTOR_EXCLUDED_MARKETS}
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
        tested = plan["already_tested"].map(_coerce_bool) if "already_tested" in plan.columns else pd.Series(False, index=plan.index)
        fetched = plan["already_fetched"].map(_coerce_bool) if "already_fetched" in plan.columns else pd.Series(False, index=plan.index)
        fresh = plan[(~tested) & (~fetched)].copy()
        if "rank" in fresh.columns:
            fresh["_rank"] = pd.to_numeric(fresh["rank"], errors="coerce")
            fresh = fresh.sort_values("_rank")
        if not fresh.empty:
            row = fresh.iloc[0]
            left = _md_text(row.get("asset_x", ""))
            right = _md_text(row.get("asset_y", ""))
            return left, right, pair_id or _md_text(row.get("pair_id", "")) or _pair_id_from_markets(left, right)
    raise SystemExit("dydx-long-history-plan requires --pair or --asset-x/--asset-y when no fresh expansion pair exists")


def _parse_iso_datetime(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
        return int(len(pd.read_csv(path)))
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
    acceptance_path = acceptance_path or _acceptance_report_path()
    if not acceptance_path.exists():
        raise SystemExit(f"acceptance report not found: {acceptance_path}")
    venue = (venue or "").lower()
    acceptance = pd.read_csv(acceptance_path)
    plan = build_research_gated_paper_plan(
        {
            "pair": pair,
            "strategy_id": strategy_id,
            "signal": signal,
            "hedge_ratio": hedge_ratio,
            "beta": beta,
        },
        acceptance,
        notional_usd=notional_usd,
        venue=venue,
    )
    return plan, [_intent_row(intent) for intent in plan.intents]


def _venue_asset_key(value: str) -> str:
    text = str(value or "").replace("/", "-").upper().strip()
    if text.endswith("-USD"):
        text = text[:-4]
    if text.endswith("_USD"):
        text = text[:-4]
    return text


def _build_paper_venue_options(pair: str) -> list[dict[str, object]]:
    parts = _split_pair_assets(pair)
    if len(parts) != 2:
        return []
    x, y = parts

    context = _read_csv_or_empty(ROOT / "data" / "processed" / "market_venue_context.csv")
    if not context.empty:
        context["asset_key"] = context["asset"].map(_venue_asset_key)
        context["venue"] = context["venue"].astype(str)

    context_by_asset: dict[str, dict[str, pd.Series]] = {}
    if not context.empty and {"asset_key", "venue"}.issubset(context.columns):
        for (asset_key, venue), row in context.groupby(["asset_key", "venue"]):
            context_by_asset.setdefault(asset_key, {})[str(venue).lower()] = row.iloc[0]

    universe = _read_csv_or_empty(ROOT / "data" / "processed" / "pair_universe.csv")
    row = _lookup_pair_universe_row(universe, pair)
    preferred = ""
    preferreds: list[str] = []
    if not row.empty:
        preferred = str(row.iloc[0].get("best_execution_venue", "") or "").strip().lower()
        available = str(row.iloc[0].get("available_venues", "") or "")
        preferreds = [venue.strip().lower() for venue in available.split(";") if venue.strip()]

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

        has_execution_support = venue_has_paper_adapter(normalized_venue) or normalized_venue == "dydx"
        execution_ready = left_authority and right_authority and left_tradable and right_tradable and has_execution_support
        if (left_authority and right_authority and left_tradable and right_tradable) and not has_execution_support:
            blockers.append("paper_execution_not_implemented")
        options.append(
            {
                "venue": normalized_venue,
                "executable": execution_ready,
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
) -> None:
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
    if plan.status != "paper_ready":
        path = append_paper_trading_record(
            paper_trading_record(plan),
            journal_path or ROOT / "reports" / "paper_trading_journal.csv",
        )
        print(f"paper_trading_journal: {path}")
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
        elif adapter_contract["configured"] and adapter_contract["valid"] and not adapter_contract["exchange_submission_capable"]:
            blockers.append("record_only_dydx_order_client_adapter")
        if order_client is None and "missing_dydx_v4_client" not in blockers:
            blockers.append("missing_dydx_order_client_adapter")
    else:
        order_client, order_adapter_error = _load_venue_order_client_adapter(selected_venue)
        adapter_contract = validate_venue_order_client_adapter(selected_venue)
        if order_adapter_error:
            blockers.append(f"invalid_{selected_venue}_order_client_adapter")
        else:
            if not adapter_contract["configured"]:
                blockers.append(f"missing_{selected_venue}_order_client_adapter")
            elif not adapter_contract["valid"]:
                blockers.append(f"{selected_venue}_order_client_adapter_invalid:{adapter_contract.get('error')}")
            elif not adapter_contract["exchange_submission_capable"]:
                blockers.append(f"record_only_{selected_venue}_order_client_adapter")
    if order_client is None and not blockers:
        blockers.append(f"paper_execution_not_implemented_for_{selected_venue}")

    if blockers:
        print(f"execution_blockers: {','.join(blockers)}")
        blocked_plan = block_paper_plan_for_execution_config(plan, blockers)
        path = append_paper_trading_record(
            paper_trading_record(blocked_plan, blockers=blockers),
            journal_path or ROOT / "reports" / "paper_trading_journal.csv",
        )
        print(f"paper_trading_journal: {path}")
        return
    execution = build_execution_venue(selected_venue, config=config, order_client=order_client, market_data_client=build_dydx_indexer_adapter(config))
    fills = submit_paper_plan(plan, execution)
    if fills:
        print(pd.DataFrame([fill.__dict__ for fill in fills]).to_string(index=False))
    path = append_paper_trading_record(
        paper_trading_record(plan, fills=fills, blockers=blockers),
        journal_path or ROOT / "reports" / "paper_trading_journal.csv",
    )
    print(f"paper_trading_journal: {path}")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "system-check",
            "build-artifact-index",
            "current-state",
            "build-pair-universe",
            "build-trade-dataset",
            "train-trade-gate",
            "run-model-gated-backtest",
            "export-trade-gate-model",
            "build-command-dashboard",
            "run-orchestrator",
            "build-mini-agent-orchestration",
            "build-orchestrator-assistant",
            "build-specialist-scoreboard",
            "run-rl-research",
            "run-rl-idea-scout",
            "train-rl-ppo",
            "export-rl-policy",
            "archive-from-index",
            "build-wizard-evidence",
            "build-wizard-hypotheses",
            "build-wizard-diagnostic-confirmation",
            "build-wizard-local-parity",
            "build-wizard-exact-mode-capture-queue",
            "build-wizard-research-pack",
            "build-market-venue-context",
            "build-venue-lane-test-plan",
            "build-multi-venue-history-readiness",
            "fetch-hyperliquid-candles",
            "build-hyperliquid-pair-history",
            "hyperliquid-lane-readiness",
            "fetch-binance-spot-candles",
            "build-binance-spot-pair-history",
            "binance-spot-history-readiness",
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
            "research-spine",
            "crawl-crypto-wizards",
            "crawl-crypto-wizards-min5",
            "crawl-crypto-wizards-min5-backtest",
            "paper-plan",
        ],
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--queue-path", type=Path, default=None)
    parser.add_argument(
        "--endpoint",
        action="append",
        default=None,
        help="Crypto Wizards endpoint as name=/path or name=https://host/path. May be repeated.",
    )
    parser.add_argument("--pair", default=None)
    parser.add_argument("--source", default=None)
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
    parser.add_argument("--zscore-window", type=int, default=320)
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--to-iso", default=None)
    parser.add_argument("--similarity-k", type=int, default=6)
    parser.add_argument("--priority", default="Sharpe")
    parser.add_argument("--cw-strategy", default="Spread")
    parser.add_argument("--exchange", default="Dydx")
    parser.add_argument("--period", type=int, default=320)
    parser.add_argument("--spread-type", default="Static")
    parser.add_argument("--roll-w", type=int, default=42)
    parser.add_argument("--asset", default=None)
    parser.add_argument("--coin", default=None)
    parser.add_argument("--days", type=int, default=500)
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
    parser.add_argument("--acceptance-path", type=Path, default=None)
    parser.add_argument("--journal-path", type=Path, default=None)
    parser.add_argument("--history-path", type=Path, default=None)
    parser.add_argument("--funding-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--entry-threshold", type=float, default=2.0)
    parser.add_argument("--exit-threshold", type=float, default=0.0)
    parser.add_argument("--walkforward-splits", type=int, default=5)
    parser.add_argument("--min-train-rows", type=int, default=100)
    parser.add_argument("--max-combo-size", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--market", default=None)
    parser.add_argument("--venue", default=None, help="Execution venue to target: dydx|hyperliquid|binance|binanceus|coinbase|bybit|auto")
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
    parser.add_argument("--diagnostic-output", type=Path, default=None)
    parser.add_argument("--json-path", type=Path, default=None)
    parser.add_argument("--download-dir", type=Path, default=None)
    parser.add_argument("--mcp-url", default=None, help="Apify MCP server URL; defaults to APIFY_MCP_SERVER_URL")
    parser.add_argument("--source-filter", dest="source_filter", default=None, help="Limit Apify source refresh to a single source_id")
    parser.add_argument("--wait-seconds", type=int, default=90, help="Actor fetch timeout for Apify (seconds)")
    parser.add_argument("--no-fetch", action="store_true", help="Skip actor runs while building the Apify source coverage table")
    parser.add_argument("--apify-token", default=None, help="Optional APIFY_API_TOKEN override")
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
    parser.add_argument("--force-refresh", action="store_true", help="For run-orchestrator, ignore reusable stage artifacts where supported.")
    parser.add_argument("--fail-fast", action="store_true", help="For run-orchestrator, stop at the first blocked or failed stage.")
    parser.add_argument("--report-only", action="store_true", help="For run-orchestrator, only build reporting stages.")
    args = parser.parse_args()
    load_env_file(args.env_file)
    if not args.indexer_base:
        args.indexer_base = os.getenv("QPA_INDEXER_BASE", "https://indexer.dydx.trade").strip() or "https://indexer.dydx.trade"
    if not args.indexer_scheme:
        args.indexer_scheme = os.getenv("QPA_INDEXER_SCHEME", "").strip()
    if args.command == "system-check":
        result = system_check()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-artifact-index":
        result = build_artifact_index()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "current-state":
        result = current_state()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-pair-universe":
        result = build_pair_universe()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-market-venue-context":
        result = build_market_venue_context()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-venue-lane-test-plan":
        result = build_venue_lane_test_plan()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-multi-venue-history-readiness":
        result = build_multi_venue_history_readiness(top_n=args.top_n)
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "fetch-hyperliquid-candles":
        coin = args.coin or args.asset or args.market
        if not coin:
            raise SystemExit("--coin, --asset, or --market is required")
        path = fetch_hyperliquid_candles(coin=coin, interval=args.interval or "1d", days=args.days)
        print(json.dumps({"path": str(path), "coin": coin, "interval": args.interval or "1d"}, indent=2))
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
        print(json.dumps({"path": str(path), "asset_x": args.asset_x, "asset_y": args.asset_y, "interval": args.interval or "1d"}, indent=2))
    elif args.command == "hyperliquid-lane-readiness":
        result = build_hyperliquid_lane_report()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "fetch-binance-spot-candles":
        symbol = args.market or args.asset or args.coin
        if not symbol:
            raise SystemExit("--market, --asset, or --coin is required")
        path = fetch_binance_spot_candles(symbol=symbol, interval=args.interval or "1d", limit=args.limit)
        print(json.dumps({"path": str(path), "symbol": symbol, "interval": args.interval or "1d"}, indent=2))
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
        print(json.dumps({"path": str(path), "asset_x": args.asset_x, "asset_y": args.asset_y, "interval": args.interval or "1d"}, indent=2))
    elif args.command == "binance-spot-history-readiness":
        result = build_binance_spot_lane_report()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-trade-dataset":
        result = build_trade_dataset(input_dir=args.input_dir, funding_path=args.funding_path)
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "train-trade-gate":
        result = train_trade_gate(input_path=args.input_dir, walkforward_splits=args.walkforward_splits, min_train_rows=args.min_train_rows)
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "run-model-gated-backtest":
        result = run_model_gated_backtest()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "export-trade-gate-model":
        result = export_trade_gate_model()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-command-dashboard":
        result = build_command_dashboard()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
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
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-mini-agent-orchestration":
        result = build_mini_agent_orchestration()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-orchestrator-assistant":
        result = build_orchestrator_assistant()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-specialist-scoreboard":
        result = build_specialist_scoreboard()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "run-rl-research":
        pair_id = "" if args.pair_id == "1" else (args.pair_id or "")
        result = run_rl_research(pair_id=pair_id)
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "run-rl-idea-scout":
        result = run_rl_idea_scout(
            pair_filter=args.pair,
            top_ideas=args.top_n,
            similarity_k=args.similarity_k,
        )
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "train-rl-ppo":
        pair_id = "" if args.pair_id == "1" else (args.pair_id or "")
        result = train_ppo_research_policy(pair_id=pair_id)
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "export-rl-policy":
        result = export_rl_policy()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "archive-from-index":
        result = archive_from_index(dry_run=not args.apply)
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-wizard-evidence":
        result = build_wizard_evidence()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-wizard-hypotheses":
        result = build_wizard_hypotheses()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-wizard-diagnostic-confirmation":
        result = build_wizard_diagnostic_confirmation()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-wizard-local-parity":
        result = build_wizard_local_parity()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-wizard-exact-mode-capture-queue":
        result = build_wizard_exact_mode_capture_queue()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-wizard-research-pack":
        result = build_wizard_research_pack()
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "verify-wizard-local-mode":
        result = verify_wizard_local_mode(
            history_path=args.history_path,
            wizard_capture_path=args.json_path,
            output_name=args.output_name or "bnb_stx_daily_320_static_spread",
            entry_threshold=args.entry_threshold,
            exit_threshold=args.exit_threshold,
        )
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-wizard-local-verification-batch":
        result = build_wizard_local_verification_batch(
            queue_path=args.queue_path or args.input_dir,
            max_pairs=args.max_pairs,
        )
        print(json.dumps({"summary": result.summary, "paths": {k: str(v) for k, v in result.paths.items()}}, indent=2))
    elif args.command == "build-dictionaries":
        build_dictionaries()
    elif args.command == "ingest-fixtures":
        ingest_fixtures(args.input_dir)
    elif args.command == "ingest-crypto-wizards-scanner":
        ingest_crypto_wizards_scanner(args.input_dir)
    elif args.command == "normalize-enrichment-fixtures":
        output = normalize_enrichment_fixtures(args.source, args.input_dir, args.output_path)
        print(f"normalized_enrichment_feed: {output}")
        print(f"normalization_report: {ROOT / 'reports' / f'{_canonical_source_name(args.source)}_normalization_report.csv'}")
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
        run_pair_detail_experiments(args.input_dir, args.funding_path)
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
            funding_path=args.research_funding_path,
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
        token = args.apify_token or os.getenv("APIFY_API_TOKEN", "").strip()
        result = refresh_apify_sources(
            root=ROOT,
            mcp_url=mcp_url,
            source_filter=args.source_filter,
            do_fetch=not args.no_fetch,
            api_token=token if token else None,
            wait_seconds=args.wait_seconds,
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
                        "type": "utility" if source.startswith("apify/") else "market_or_context_feed",
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
        print_strategy_trade_count_gap()
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
        frame = fetch_dydx_long_history_windows(
            plan_path=args.input_dir,
            max_windows=args.windows,
            indexer_base=args.indexer_base,
            indexer_scheme=args.indexer_scheme,
            allow_stale_fetch=args.allow_stale_fetch,
        )
        print(frame.to_string(index=False))
        print(f"dydx_long_history_fetch: {ROOT / 'reports' / 'dydx_long_history_fetch.csv'}")
    elif args.command == "run-dydx-long-history":
        if not args.asset_x or not args.asset_y:
            raise SystemExit("run-dydx-long-history requires --asset-x and --asset-y")
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
            funding_path=args.research_funding_path,
            allow_stale_fetch=args.allow_stale_fetch,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
    elif args.command == "run-dydx-pair-expansion":
        if not args.allow_blocked_exploration:
            _assert_ready_for_exploration()
        print_run_dydx_pair_expansion(
            max_pairs=args.max_pairs,
            limit=args.limit,
            indexer_base=args.indexer_base,
            indexer_scheme=args.indexer_scheme,
            output_path=args.output_path,
            run_research=args.run_research,
            skip_fetch=args.skip_fetch,
            allow_stale_fetch=args.allow_stale_fetch,
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
            output_dir=args.output_path,
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
    elif args.command == "research-spine":
        print_research_spine(args.input_dir, require_two_leg=not args.allow_spread_only, funding_path=args.funding_path)
    elif args.command == "crawl-crypto-wizards":
        crawl_crypto_wizards(args.endpoint)
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
    if not universe.empty and "pair" in universe.columns and "best_execution_venue" in universe.columns:
        universe_pairs = universe[universe["pair"].astype(str).str.upper() == target]
        if universe_pairs.empty:
            alt = target.replace("-", "/")
            universe_pairs = universe[universe["pair"].astype(str).str.upper() == alt]
        if not universe_pairs.empty:
            best = universe_pairs.iloc[0]
            venue = str(best.get("best_execution_venue", "") or best.get("exchange", "") or "").strip().lower()
            if venue:
                return venue
            if bool(best.get("dydx_tradable", False)):
                return "dydx"
            if str(best.get("decision_bucket", "")).upper() == "PROMOTE":
                return "dydx"
    return "dydx"


if __name__ == "__main__":
    main()
