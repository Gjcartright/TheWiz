from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.pair_detail_ingestion import extract_history_rows, snapshot_from_payload


ROOT = Path(__file__).resolve().parents[2]
ACTIVE = ROOT / "reports" / "active"
DATA_PROCESSED = ROOT / "data" / "processed"

DISCOVERY_MIN_SHARPE = 1.75
DISCOVERY_MIN_RETURNS_TOTAL = 0.10

WIZARD_MODE_MAP: dict[tuple[int, int], str] = {
    (3, 1): "Static (Spread)",
    (3, 2): "Static (ZScoreR)",
    (1, 1): "Dyn (Spread)",
    (1, 2): "Dyn (ZScoreR)",
    (2, 1): "OU (Spread)",
    (2, 2): "OU (ZScoreR)",
    (1, 3): "Copula",
}

MODE_TO_IDS = {mode: ids for ids, mode in WIZARD_MODE_MAP.items()}

WIZARD_EVIDENCE_COLUMNS = [
    "pair",
    "asset_x",
    "asset_y",
    "exchange",
    "interval",
    "period",
    "exact_mode",
    "spread_id",
    "strategy_id",
    "mode_valid",
    "mode_source",
    "mode_blocker",
    "sharpe",
    "returns_total",
    "returns_total_pct",
    "discovery_min_sharpe",
    "discovery_min_returns_total",
    "passes_sharpe_gate",
    "passes_sharpe_gt_2",
    "passes_returns_total_gt_20pct",
    "hurst",
    "half_life",
    "pearson",
    "spearman",
    "kendall",
    "copula",
    "corr_copula",
    "u1_given_u2",
    "u2_given_u1",
    "ecm_x_available",
    "ecm_y_available",
    "ecm_strength_available",
    "hedge_ratio",
    "closed_trades",
    "drawdown",
    "source_system",
    "source_path",
    "evidence_path",
]

HYPOTHESIS_COLUMNS = [
    "pair",
    "asset_x",
    "asset_y",
    "interval",
    "exact_mode",
    "spread_id",
    "strategy_id",
    "sharpe",
    "returns_total",
    "hypothesis_status",
    "hypothesis_reason",
    "next_step",
    "local_data_available",
    "evidence_path",
]

DIAGNOSTIC_COLUMNS = [
    "pair",
    "interval",
    "exact_mode",
    "correlation_status",
    "ecm_status",
    "copula_status",
    "mean_reversion_status",
    "wizard_diagnostic_score",
    "diagnostic_blocker",
    "evidence_path",
]

PARITY_COLUMNS = [
    "pair",
    "interval",
    "exact_mode",
    "wizard_source_path",
    "local_source_path",
    "wizard_zscore_last",
    "local_zscore_last",
    "wizard_rolling_zscore_last",
    "local_rolling_zscore_last",
    "zscore_delta",
    "rolling_zscore_delta",
    "parity_status",
    "parity_reason",
    "evidence_path",
]

EXACT_MODE_CAPTURE_QUEUE_COLUMNS = [
    "priority_rank",
    "pair",
    "asset_x",
    "asset_y",
    "interval",
    "sharpe",
    "returns_total",
    "returns_total_pct",
    "mode_blocker",
    "pair_page_url",
    "required_capture_fields",
    "operator_action",
    "source_path",
    "evidence_path",
]


@dataclass(frozen=True)
class ModeResolution:
    exact_mode: str
    spread_id: int | None
    strategy_id: int | None
    mode_valid: bool
    mode_source: str
    mode_blocker: str


def mode_from_ids(spread_id: int | None, strategy_id: int | None) -> str:
    if spread_id is None or strategy_id is None:
        return ""
    return WIZARD_MODE_MAP.get((int(spread_id), int(strategy_id)), "")


def ids_from_exact_mode(exact_mode: str | None) -> tuple[int | None, int | None]:
    if not exact_mode:
        return None, None
    normalized = _normalize_mode_label(exact_mode)
    for label, ids in MODE_TO_IDS.items():
        if _normalize_mode_label(label) == normalized:
            return ids
    return None, None


def build_wizard_evidence(root: Path = ROOT) -> CommandResult:
    rows = _wizard_evidence_rows(root)
    frame = pd.DataFrame(rows, columns=WIZARD_EVIDENCE_COLUMNS)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["pair", "interval", "exact_mode", "source_path"], keep="first")
        frame = frame.sort_values(["passes_sharpe_gate", "returns_total", "sharpe"], ascending=[False, False, False])
    output = root / "data" / "processed" / "wizard_evidence.csv"
    summary = root / "reports" / "active" / "wizard_evidence_summary.md"
    _write_csv(frame, output)
    _write_text(summary, _wizard_evidence_markdown(frame))
    return CommandResult(
        paths={"wizard_evidence": output, "summary_md": summary},
        summary={"rows": len(frame), "mode_valid": int(frame["mode_valid"].astype(bool).sum()) if not frame.empty else 0},
    )


def build_wizard_hypotheses(root: Path = ROOT) -> CommandResult:
    evidence = _ensure_wizard_evidence(root)
    local_index = _local_history_index(root)
    rows = []
    for _, row in evidence.iterrows():
        local_available = _local_data_available(local_index, row)
        status, reason, next_step = _hypothesis_status(row, local_available)
        rows.append(
            {
                "pair": row.get("pair", ""),
                "asset_x": row.get("asset_x", ""),
                "asset_y": row.get("asset_y", ""),
                "interval": row.get("interval", ""),
                "exact_mode": row.get("exact_mode", ""),
                "spread_id": row.get("spread_id", ""),
                "strategy_id": row.get("strategy_id", ""),
                "sharpe": row.get("sharpe", ""),
                "returns_total": row.get("returns_total", ""),
                "hypothesis_status": status,
                "hypothesis_reason": reason,
                "next_step": next_step,
                "local_data_available": local_available,
                "evidence_path": row.get("evidence_path", ""),
            }
        )
    frame = pd.DataFrame(rows, columns=HYPOTHESIS_COLUMNS)
    output = root / "reports" / "active" / "wizard_hypotheses.csv"
    summary = root / "reports" / "active" / "wizard_hypotheses.md"
    _write_csv(frame, output)
    _write_text(summary, _simple_markdown("Wizard Hypotheses", frame))
    return CommandResult(paths={"hypotheses": output, "summary_md": summary}, summary={"rows": len(frame)})


def build_wizard_diagnostic_confirmation(root: Path = ROOT) -> CommandResult:
    evidence = _ensure_wizard_evidence(root)
    rows = []
    for _, row in evidence.iterrows():
        corr_status, corr_score = _correlation_status(row)
        ecm_status, ecm_score = _ecm_status(row)
        copula_status, copula_score = _copula_status(row)
        mr_status, mr_score = _mean_reversion_status(row)
        blockers = [
            status
            for status in [corr_status, ecm_status, copula_status, mr_status]
            if status.startswith("missing") or status.startswith("weak")
        ]
        rows.append(
            {
                "pair": row.get("pair", ""),
                "interval": row.get("interval", ""),
                "exact_mode": row.get("exact_mode", ""),
                "correlation_status": corr_status,
                "ecm_status": ecm_status,
                "copula_status": copula_status,
                "mean_reversion_status": mr_status,
                "wizard_diagnostic_score": round(corr_score + ecm_score + copula_score + mr_score, 3),
                "diagnostic_blocker": ";".join(blockers),
                "evidence_path": row.get("evidence_path", ""),
            }
        )
    frame = pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)
    output = root / "reports" / "active" / "wizard_diagnostic_confirmation.csv"
    _write_csv(frame, output)
    return CommandResult(paths={"diagnostics": output}, summary={"rows": len(frame)})


def build_wizard_local_parity(root: Path = ROOT) -> CommandResult:
    evidence = _ensure_wizard_evidence(root)
    local_index = _local_history_index(root)
    rows = []
    for _, row in evidence.iterrows():
        local_path = _matching_local_path(local_index, row)
        wizard_last = _history_last_values(Path(str(row.get("source_path", ""))))
        local_last = _history_last_values(local_path) if local_path else {}
        z_delta = _delta(wizard_last.get("zscore"), local_last.get("zscore"))
        rz_delta = _delta(wizard_last.get("rolling_zscore"), local_last.get("rolling_zscore"))
        status, reason = _parity_status(local_path, z_delta, rz_delta)
        rows.append(
            {
                "pair": row.get("pair", ""),
                "interval": row.get("interval", ""),
                "exact_mode": row.get("exact_mode", ""),
                "wizard_source_path": row.get("source_path", ""),
                "local_source_path": str(local_path or ""),
                "wizard_zscore_last": wizard_last.get("zscore", ""),
                "local_zscore_last": local_last.get("zscore", ""),
                "wizard_rolling_zscore_last": wizard_last.get("rolling_zscore", ""),
                "local_rolling_zscore_last": local_last.get("rolling_zscore", ""),
                "zscore_delta": z_delta if z_delta is not None else "",
                "rolling_zscore_delta": rz_delta if rz_delta is not None else "",
                "parity_status": status,
                "parity_reason": reason,
                "evidence_path": ";".join([p for p in [str(row.get("source_path", "")), str(local_path or "")] if p]),
            }
        )
    frame = pd.DataFrame(rows, columns=PARITY_COLUMNS)
    output = root / "reports" / "active" / "wizard_vs_local_parity_report.csv"
    summary = root / "reports" / "active" / "wizard_vs_local_parity_report.md"
    _write_csv(frame, output)
    _write_text(summary, _simple_markdown("Wizard vs Local Parity", frame))
    return CommandResult(paths={"parity": output, "summary_md": summary}, summary={"rows": len(frame)})


def build_wizard_exact_mode_capture_queue(root: Path = ROOT) -> CommandResult:
    evidence = _ensure_wizard_evidence(root)
    if evidence.empty:
        frame = pd.DataFrame(columns=EXACT_MODE_CAPTURE_QUEUE_COLUMNS)
    else:
        valid_pairs = set(
            _evidence_pair_interval_key(row)
            for _, row in evidence[evidence.get("mode_valid", pd.Series(False, index=evidence.index)).astype(bool)].iterrows()
        )
        blocked = evidence[
            (~evidence.get("mode_valid", pd.Series(False, index=evidence.index)).astype(bool))
            & _passes_sharpe_gate(evidence).astype(bool)
            & evidence.get("passes_returns_total_gt_20pct", pd.Series(False, index=evidence.index)).astype(bool)
        ].copy()
        if not blocked.empty and valid_pairs:
            blocked["_pair_interval"] = [_evidence_pair_interval_key(row) for _, row in blocked.iterrows()]
            blocked = blocked[~blocked["_pair_interval"].isin(valid_pairs)]
        if not blocked.empty:
            blocked["_rank_sharpe"] = _numeric_series(blocked.get("sharpe")).fillna(-999)
            blocked["_rank_return"] = _numeric_series(blocked.get("returns_total")).fillna(-999)
            blocked = blocked.sort_values(["_rank_sharpe", "_rank_return"], ascending=[False, False])
            blocked = blocked.drop_duplicates(subset=["pair", "interval"], keep="first")
        rows = []
        for rank, (_, row) in enumerate(blocked.iterrows(), start=1):
            source_path = str(row.get("source_path", "") or "")
            pair_page_url = _pair_page_url_from_source(root, source_path)
            rows.append(
                {
                    "priority_rank": rank,
                    "pair": row.get("pair", ""),
                    "asset_x": row.get("asset_x", ""),
                    "asset_y": row.get("asset_y", ""),
                    "interval": row.get("interval", ""),
                    "sharpe": row.get("sharpe", ""),
                    "returns_total": row.get("returns_total", ""),
                    "returns_total_pct": row.get("returns_total_pct", ""),
                    "mode_blocker": row.get("mode_blocker", "missing_exact_mode"),
                    "pair_page_url": pair_page_url,
                    "required_capture_fields": "selected_strategy_value;spread_id;strategy_id;exact_mode;period;interval;backtest_settings",
                    "operator_action": _capture_action(pair_page_url),
                    "source_path": source_path,
                    "evidence_path": row.get("evidence_path", ""),
                }
            )
        frame = pd.DataFrame(rows, columns=EXACT_MODE_CAPTURE_QUEUE_COLUMNS)
    output = root / "reports" / "active" / "wizard_exact_mode_capture_queue.csv"
    summary = root / "reports" / "active" / "wizard_exact_mode_capture_queue.md"
    _write_csv(frame, output)
    _write_text(summary, _simple_markdown("Wizard Exact Mode Capture Queue", frame))
    return CommandResult(paths={"exact_mode_capture_queue": output, "summary_md": summary}, summary={"rows": len(frame)})


def build_wizard_research_pack(root: Path = ROOT) -> CommandResult:
    evidence = build_wizard_evidence(root)
    hypotheses = build_wizard_hypotheses(root)
    diagnostics = build_wizard_diagnostic_confirmation(root)
    parity = build_wizard_local_parity(root)
    queue = build_wizard_exact_mode_capture_queue(root)
    return CommandResult(
        paths={
            "wizard_evidence": evidence.paths["wizard_evidence"],
            "wizard_evidence_summary": evidence.paths["summary_md"],
            "wizard_hypotheses": hypotheses.paths["hypotheses"],
            "wizard_hypotheses_summary": hypotheses.paths["summary_md"],
            "wizard_diagnostics": diagnostics.paths["diagnostics"],
            "wizard_parity": parity.paths["parity"],
            "wizard_parity_summary": parity.paths["summary_md"],
            "wizard_exact_mode_capture_queue": queue.paths["exact_mode_capture_queue"],
            "wizard_exact_mode_capture_queue_summary": queue.paths["summary_md"],
        },
        summary={
            "evidence_rows": evidence.summary["rows"],
            "hypotheses_rows": hypotheses.summary["rows"],
            "diagnostic_rows": diagnostics.summary["rows"],
            "parity_rows": parity.summary["rows"],
            "exact_mode_capture_queue_rows": queue.summary["rows"],
        },
    )


def _wizard_evidence_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for active_path in sorted((root / "reports" / "active").glob("crypto_wizards*.csv")):
        frame = _read_csv(active_path)
        for _, active_row in frame.iterrows():
            source_path = str(active_row.get("source_path", active_row.get("raw_path", "")) or "")
            if source_path:
                seen_paths.add(source_path)
            payload_row = _row_from_source_path(root, source_path) if source_path else {}
            merged = {**payload_row, **{k: v for k, v in active_row.to_dict().items() if pd.notna(v)}}
            rows.append(_normalize_evidence_row(merged, active_path, source_path or str(active_path), "crypto_wizards_active_report"))
    pair_detail_dir = root / "data" / "raw" / "pair_details"
    for path in sorted(pair_detail_dir.glob("*.json")):
        rel = path.relative_to(root).as_posix() if _is_relative_to(path, root) else str(path)
        if rel in seen_paths or str(path) in seen_paths:
            continue
        try:
            row = _row_from_payload_path(path)
        except Exception:
            continue
        if _looks_like_wizard_source(row, path):
            rows.append(_normalize_evidence_row(row, path, rel, "crypto_wizards_pair_detail"))
    return rows


def _row_from_source_path(root: Path, source_path: str) -> dict[str, object]:
    path = Path(source_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists() or path.suffix.lower() != ".json":
        return {}
    return _row_from_payload_path(path)


def _row_from_payload_path(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    snapshot = snapshot_from_payload(payload)
    row = snapshot.to_row()
    normalized_payload = {str(k).lower(): v for k, v in payload.items()} if isinstance(payload, dict) else {}
    for field in ["spread_id", "strategy_id", "exact_mode", "zscore_last", "zscore_roll_last"]:
        if field in normalized_payload:
            row[field] = normalized_payload[field]
    row["source_path"] = str(path)
    row["history_rows"] = len(extract_history_rows(payload)) if isinstance(payload, dict) else 0
    return row


def _normalize_evidence_row(row: dict[str, object], evidence_path: Path, source_path: str, source_system: str) -> dict[str, object]:
    spread_id = _as_int(_first_present(row, ["spread_id"]))
    strategy_id = _as_int(_first_present(row, ["strategy_id"]))
    exact_mode = str(_first_present(row, ["exact_mode", "mode", "wizard_exact_mode"]) or "")
    strategy_hint = str(_first_present(row, ["strategy_family_note", "strategy_mode", "spread_type"]) or "")
    z_or_spread_hint = str(_first_present(row, ["z_or_spread_note", "cw_strategy"]) or "")
    mode = _resolve_mode(spread_id, strategy_id, exact_mode, strategy_hint, z_or_spread_hint)
    returns_total = _as_float(_first_present(row, ["returns_total", "returns_total_decimal"]))
    if returns_total is None:
        returns_pct = _as_float(_first_present(row, ["returns_total_pct"]))
        returns_total = returns_pct / 100 if returns_pct is not None else None
    sharpe = _as_float(_first_present(row, ["sharpe", "sharpe_ratio"]))
    pair = _pair_value(row)
    asset_x, asset_y = _asset_values(row, pair)
    ecm_x = _as_bool(_first_present(row, ["ecm_x_available"]))
    ecm_y = _as_bool(_first_present(row, ["ecm_y_available"]))
    ecm_strength = _as_bool(_first_present(row, ["ecm_strength_available"]))
    return {
        "pair": pair,
        "asset_x": asset_x,
        "asset_y": asset_y,
        "exchange": str(_first_present(row, ["exchange"]) or ""),
        "interval": str(_first_present(row, ["interval", "timeframe"]) or ""),
        "period": _as_int(_first_present(row, ["period"])),
        "exact_mode": mode.exact_mode,
        "spread_id": mode.spread_id,
        "strategy_id": mode.strategy_id,
        "mode_valid": mode.mode_valid,
        "mode_source": mode.mode_source,
        "mode_blocker": mode.mode_blocker,
        "sharpe": sharpe,
        "returns_total": returns_total,
        "returns_total_pct": returns_total * 100 if returns_total is not None else None,
        "discovery_min_sharpe": DISCOVERY_MIN_SHARPE,
        "discovery_min_returns_total": DISCOVERY_MIN_RETURNS_TOTAL,
        "passes_sharpe_gate": bool(sharpe is not None and sharpe >= DISCOVERY_MIN_SHARPE),
        "passes_sharpe_gt_2": bool(sharpe is not None and sharpe >= DISCOVERY_MIN_SHARPE),
        "passes_returns_total_gt_20pct": bool(returns_total is not None and returns_total > DISCOVERY_MIN_RETURNS_TOTAL),
        "hurst": _as_float(_first_present(row, ["hurst"])),
        "half_life": _as_float(_first_present(row, ["half_life", "halflife"])),
        "pearson": _as_float(_first_present(row, ["pearson"])),
        "spearman": _as_float(_first_present(row, ["spearman"])),
        "kendall": _as_float(_first_present(row, ["kendall"])),
        "copula": str(_first_present(row, ["copula"]) or ""),
        "corr_copula": _as_float(_first_present(row, ["corr_copula", "copula_correlation"])),
        "u1_given_u2": _as_float(_first_present(row, ["u1_given_u2"])),
        "u2_given_u1": _as_float(_first_present(row, ["u2_given_u1"])),
        "ecm_x_available": ecm_x,
        "ecm_y_available": ecm_y,
        "ecm_strength_available": ecm_strength,
        "hedge_ratio": _as_float(_first_present(row, ["hedge_ratio"])),
        "closed_trades": _as_int(_first_present(row, ["closed_trades"])),
        "drawdown": _as_float(_first_present(row, ["drawdown", "max_drawdown"])),
        "source_system": source_system,
        "source_path": source_path,
        "evidence_path": str(evidence_path),
    }


def _resolve_mode(
    spread_id: int | None,
    strategy_id: int | None,
    exact_mode: str,
    strategy_hint: str,
    z_or_spread_hint: str,
) -> ModeResolution:
    if spread_id is not None and strategy_id is not None:
        label = mode_from_ids(spread_id, strategy_id)
        if label:
            return ModeResolution(label, spread_id, strategy_id, True, "spread_id_strategy_id", "")
        return ModeResolution("", spread_id, strategy_id, False, "spread_id_strategy_id", "unknown_spread_strategy_id")
    ids = ids_from_exact_mode(exact_mode)
    if ids != (None, None):
        return ModeResolution(WIZARD_MODE_MAP[ids], ids[0], ids[1], True, "exact_mode", "")
    inferred = _infer_exact_mode(strategy_hint, z_or_spread_hint)
    if inferred:
        ids = ids_from_exact_mode(inferred)
        return ModeResolution(inferred, ids[0], ids[1], True, "inferred_from_strategy_and_signal", "")
    return ModeResolution("", spread_id, strategy_id, False, "missing", "missing_exact_mode")


def _infer_exact_mode(strategy_hint: str, z_or_spread_hint: str) -> str:
    engine = strategy_hint.strip().lower()
    signal = z_or_spread_hint.strip().lower()
    if "copula" in engine or "copula" in signal:
        return "Copula"
    spread_label = ""
    if "static" in engine:
        spread_label = "Static"
    elif engine in {"dyn", "dynamic"} or "dynamic" in engine:
        spread_label = "Dyn"
    elif "ou" in engine:
        spread_label = "OU"
    signal_label = ""
    if "zscorer" in signal or "zscore r" in signal or "roll" in signal:
        signal_label = "ZScoreR"
    elif "spread" in signal:
        signal_label = "Spread"
    if spread_label and signal_label:
        return f"{spread_label} ({signal_label})"
    return ""


def _ensure_wizard_evidence(root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "wizard_evidence.csv"
    if not path.exists():
        build_wizard_evidence(root)
    return _read_csv(path)


def _passes_sharpe_gate(frame: pd.DataFrame) -> pd.Series:
    if "passes_sharpe_gate" in frame:
        return frame["passes_sharpe_gate"].astype(bool)
    if "sharpe" in frame:
        sharpe = _numeric_series(frame["sharpe"])
        return sharpe.ge(DISCOVERY_MIN_SHARPE).fillna(False)
    if "passes_sharpe_gt_2" in frame:
        return frame["passes_sharpe_gt_2"].astype(bool)
    return pd.Series(False, index=frame.index)


def _row_passes_sharpe_gate(row: pd.Series) -> bool:
    if "passes_sharpe_gate" in row.index:
        return bool(row.get("passes_sharpe_gate", False))
    sharpe = _as_float(row.get("sharpe"))
    if sharpe is not None:
        return sharpe >= DISCOVERY_MIN_SHARPE
    return bool(row.get("passes_sharpe_gt_2", False))


def _hypothesis_status(row: pd.Series, local_available: bool) -> tuple[str, str, str]:
    if not bool(row.get("mode_valid", False)):
        return "RESEARCH_BLOCKED", "missing_exact_mode", "capture_pair_page_exact_mode"
    if not _row_passes_sharpe_gate(row):
        return "REJECT", f"sharpe_below_{DISCOVERY_MIN_SHARPE:g}", "return_to_wizard_scanner"
    if not bool(row.get("passes_returns_total_gt_20pct", False)):
        return "REJECT", f"returns_total_not_above_{int(DISCOVERY_MIN_RETURNS_TOTAL * 100)}pct", "return_to_wizard_scanner"
    if not local_available:
        return "NEEDS_LOCAL_DATA", "wizard_only_evidence_cannot_promote", "fetch_local_dydx_history"
    return "HYPOTHESIS_READY", "wizard_candidate_ready_for_local_after_cost_test", "run_local_mode_separated_backtest"


def _correlation_status(row: pd.Series) -> tuple[str, float]:
    values = [_as_float(row.get(name)) for name in ["pearson", "spearman", "kendall"]]
    values = [abs(v) for v in values if v is not None]
    if not values:
        return "missing_correlation", 0.0
    if max(values) >= 0.70 and sum(v >= 0.50 for v in values) >= 2:
        return "confirmed_correlation", 15.0
    if max(values) >= 0.50:
        return "partial_correlation", 8.0
    return "weak_correlation", -5.0


def _ecm_status(row: pd.Series) -> tuple[str, float]:
    flags = [bool(row.get(name, False)) for name in ["ecm_x_available", "ecm_y_available", "ecm_strength_available"]]
    if all(flags):
        return "confirmed_ecm", 15.0
    if any(flags):
        return "partial_ecm", 5.0
    return "missing_ecm", 0.0


def _copula_status(row: pd.Series) -> tuple[str, float]:
    exact = str(row.get("exact_mode", ""))
    has_copula = bool(str(row.get("copula", "")).strip()) or _as_float(row.get("corr_copula")) is not None
    has_conditionals = _as_float(row.get("u1_given_u2")) is not None and _as_float(row.get("u2_given_u1")) is not None
    if has_copula and has_conditionals:
        return "confirmed_copula", 15.0
    if has_copula:
        return "partial_copula", 8.0
    if exact == "Copula":
        return "missing_copula_for_copula_mode", -10.0
    return "missing_copula", 0.0


def _mean_reversion_status(row: pd.Series) -> tuple[str, float]:
    hurst = _as_float(row.get("hurst"))
    half_life = _as_float(row.get("half_life"))
    if hurst is None and half_life is None:
        return "missing_mean_reversion", 0.0
    if hurst is not None and hurst >= 0.60:
        return "weak_hurst_mean_reversion", -10.0
    if half_life is not None and half_life <= 0:
        return "weak_half_life", -5.0
    return "confirmed_mean_reversion", 15.0


def _local_history_index(root: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for path in sorted((root / "data" / "raw" / "pair_details").glob("*dydx*derived_history.json")):
        try:
            row = _row_from_payload_path(path)
        except Exception:
            continue
        pair = _canonical_pair_key(str(row.get("asset_x", "") or ""), str(row.get("asset_y", "") or ""), str(row.get("pair", "")))
        interval = str(row.get("interval", "") or "")
        if pair and (pair, interval) not in index:
            index[(pair, interval)] = path
    return index


def _local_data_available(index: dict[tuple[str, str], Path], row: pd.Series) -> bool:
    return _matching_local_path(index, row) is not None


def _matching_local_path(index: dict[tuple[str, str], Path], row: pd.Series) -> Path | None:
    pair = _canonical_pair_key(str(row.get("asset_x", "") or ""), str(row.get("asset_y", "") or ""), str(row.get("pair", "")))
    interval = str(row.get("interval", "") or "")
    return index.get((pair, interval)) or index.get((pair, ""))


def _history_last_values(path: Path | None) -> dict[str, float]:
    if not path or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = extract_history_rows(payload)
    except Exception:
        return {}
    if not rows:
        return {}
    last = rows[-1]
    return {
        "zscore": _as_float(last.get("zscore")),
        "rolling_zscore": _as_float(last.get("rolling_zscore", last.get("zscore_roll"))),
    }


def _parity_status(local_path: Path | None, z_delta: float | None, rz_delta: float | None) -> tuple[str, str]:
    if local_path is None:
        return "MISSING_LOCAL_DATA", "no_matching_local_dydx_history"
    deltas = [abs(v) for v in [z_delta, rz_delta] if v is not None]
    if not deltas:
        return "NOT_REPLICATED", "no_comparable_zscore_fields"
    if max(deltas) <= 0.05:
        return "MATCH", "wizard_and_local_zscore_close"
    if max(deltas) <= 0.25:
        return "CLOSE", "small_zscore_difference"
    return "MISMATCH", "mode_or_formula_difference_needs_review"


def _pair_page_url_from_source(root: Path, source_path: str) -> str:
    if not source_path:
        return ""
    path = Path(source_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists() or path.suffix.lower() != ".json":
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("source_url") or payload.get("url") or "")


def _capture_action(pair_page_url: str) -> str:
    if pair_page_url:
        return "open_pair_page_run_capture_helper_click_recalculate_download_json_import_pair_details"
    return "find_pair_page_from_scanner_then_capture_exact_strategy_dropdown_and_backtest_settings"


def _delta(left: Any, right: Any) -> float | None:
    left_f = _as_float(left)
    right_f = _as_float(right)
    if left_f is None or right_f is None:
        return None
    return round(left_f - right_f, 8)


def _looks_like_wizard_source(row: dict[str, object], path: Path) -> bool:
    text = path.as_posix().lower()
    return "cw" in text or "wizard" in text or _as_float(row.get("sharpe")) is not None or _as_float(row.get("returns_total")) is not None


def _pair_value(row: dict[str, object]) -> str:
    pair = str(_first_present(row, ["pair"]) or "")
    if pair and pair.upper() != "UNKNOWN":
        return pair
    asset_x, asset_y = _asset_values(row, "")
    return f"{asset_x}/{asset_y}" if asset_x and asset_y else ""


def _asset_values(row: dict[str, object], pair: str) -> tuple[str, str]:
    asset_x = str(_first_present(row, ["asset_x"]) or "")
    asset_y = str(_first_present(row, ["asset_y"]) or "")
    if asset_x and asset_y:
        return asset_x, asset_y
    if "/" in pair:
        left, right = pair.split("/", 1)
        return left, right
    return asset_x, asset_y


def _canonical_pair(pair: str) -> str:
    return pair.upper().replace("_", "-").replace(" / ", "/").strip()


def _canonical_pair_key(asset_x: str, asset_y: str, pair: str = "") -> str:
    left = str(asset_x or "").upper().replace("_", "-").strip()
    right = str(asset_y or "").upper().replace("_", "-").strip()
    if left and right:
        return f"{left}|{right}"
    text = _canonical_pair(pair)
    if "/" in text:
        pieces = [piece for piece in text.split("/") if piece]
        if len(pieces) == 2:
            return f"{pieces[0]}|{pieces[1]}"
    pieces = [piece for piece in text.split("-") if piece]
    usd_positions = [i for i, piece in enumerate(pieces) if piece == "USD"]
    if len(usd_positions) >= 2:
        return f"{pieces[0]}-USD|{pieces[usd_positions[0] + 1]}-USD"
    return text


def _evidence_pair_interval_key(row: pd.Series) -> str:
    return f"{_canonical_pair_key(str(row.get('asset_x', '') or ''), str(row.get('asset_y', '') or ''), str(row.get('pair', '') or ''))}|{str(row.get('interval', '') or '')}"


def _normalize_mode_label(value: str | None) -> str:
    return str(value or "").lower().replace(" ", "").replace("-", "").replace("_", "")


def _first_present(row: dict[str, object], names: list[str]) -> object | None:
    lower = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        value = lower.get(name.lower())
        if value is not None and not (isinstance(value, float) and pd.isna(value)) and str(value).strip() != "":
            return value
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return int(number) if number is not None else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "available"}


def _numeric_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _wizard_evidence_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Wizard Evidence Summary", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]
    lines.append(f"- rows: {len(frame)}")
    if frame.empty:
        return "\n".join(lines) + "\n"
    lines.append(f"- exact mode valid: {int(frame['mode_valid'].astype(bool).sum())}")
    lines.append(f"- Sharpe >= {DISCOVERY_MIN_SHARPE:g}: {int(_passes_sharpe_gate(frame).astype(bool).sum())}")
    lines.append(f"- returns_total > {DISCOVERY_MIN_RETURNS_TOTAL:.0%}: {int(frame['passes_returns_total_gt_20pct'].astype(bool).sum())}")
    lines.extend(["", "## Top Rows", ""])
    preview_cols = ["pair", "interval", "exact_mode", "sharpe", "returns_total", "mode_blocker", "source_path"]
    lines.append(frame[preview_cols].head(20).to_markdown(index=False))
    return "\n".join(lines) + "\n"


def _simple_markdown(title: str, frame: pd.DataFrame) -> str:
    lines = [f"# {title}", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", "", f"- rows: {len(frame)}", ""]
    if not frame.empty:
        lines.append(frame.head(30).to_markdown(index=False))
    return "\n".join(lines) + "\n"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
