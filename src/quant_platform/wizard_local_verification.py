from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.backtest import BacktestResult, CostModel, backtest_two_leg_spread
from quant_platform.wizard_evidence import _ensure_wizard_evidence, _wizard_setup_identity
from quant_platform.wizard_mode_replay import WizardModeReplayResult, build_local_mode_signal


DEFAULT_HISTORY = ROOT / "data" / "raw" / "pair_details" / "pair_bnb_stx_daily_320_fresh_1day_dydx_long_history_derived_history.json"
DEFAULT_WIZARD_CAPTURE = ROOT / "data" / "raw" / "pair_details" / "pair_BNB-USD_STX-USD_Dydx_Daily_320_exact_mode_capture.json"
DEFAULT_QUEUE = ROOT / "reports" / "active" / "crypto_wizards_next_best_sharpe_returns_queue.csv"


def build_wizard_local_verification_batch(
    *,
    root: Path = ROOT,
    queue_path: Path | None = None,
    max_pairs: int = 20,
    current_date: str = "2026-06-25",
) -> CommandResult:
    """Build the Wizard-to-local verification board for the current candidate queue."""
    queue_file = queue_path or DEFAULT_QUEUE
    rows: list[dict[str, object]] = []
    for candidate in _candidate_rows(root=root, queue_file=queue_file, max_pairs=max_pairs):
        rows.append(_verify_candidate(root=root, candidate=candidate, current_date=current_date))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["verification_status", "wizard_sharpe", "wizard_returns_total"], ascending=[True, False, False])
    reports = root / "reports" / "active"
    output = reports / "wizard_local_verification_batch.csv"
    md_output = reports / "wizard_local_verification_batch.md"
    _write_csv(frame, output)
    _write_text(md_output, _batch_markdown(frame, queue_file))
    verified = int((frame.get("verification_status", pd.Series(dtype=str)) == "verified").sum()) if not frame.empty else 0
    accepted = int((frame.get("acceptance", pd.Series(dtype=str)) == "ACCEPT").sum()) if not frame.empty else 0
    blocked = int((frame.get("verification_status", pd.Series(dtype=str)) != "verified").sum()) if not frame.empty else 0
    return CommandResult(
        paths={"batch": output, "batch_md": md_output},
        summary={
            "candidates": int(len(frame)),
            "verified": verified,
            "accepted": accepted,
            "blocked": blocked,
            "queue_path": _rel(queue_file),
        },
    )


def verify_wizard_local_mode(
    *,
    root: Path = ROOT,
    history_path: Path | None = None,
    wizard_capture_path: Path | None = None,
    output_name: str = "bnb_stx_daily_320_static_spread",
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    current_date: str = "2026-06-25",
    exact_mode: str | None = None,
    mode_settings: Mapping[str, object] | None = None,
) -> CommandResult:
    history_file = history_path or DEFAULT_HISTORY
    wizard_file = wizard_capture_path or DEFAULT_WIZARD_CAPTURE
    payload = _read_json(history_file)
    try:
        history = _history_frame(payload)
    except ValueError as exc:
        reports = root / "reports" / "active"
        output_stem = f"{output_name}_after_cost"
        summary_path = reports / f"{output_stem}.csv"
        cost_path = reports / f"{output_stem}_cost_comparison.csv"
        trade_path = reports / f"{output_stem}_trade_log.csv"
        md_path = reports / f"{output_stem}.md"
        summary = {
            "pair": payload.get("pair", "unknown"),
            "asset_x": payload.get("asset_x", ""),
            "asset_y": payload.get("asset_y", ""),
            "interval": payload.get("interval", ""),
            "local_observations": 0,
            "local_last_timestamp": "",
            "local_stale_days": "",
            "pair_sharpe": "",
            "total_return": "",
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "closed_trades": 0,
            "trades": 0,
            "acceptance": "BLOCKED",
            "acceptance_reason": str(exc),
            "evidence_path": str(history_file),
            "wizard_capture_path": str(wizard_file),
            "exact_mode": "",
            "source": "orchestrator_verify_wizard_local_mode",
            "error": str(exc),
        }
        _write_csv(pd.DataFrame([summary]), summary_path)
        _write_csv(pd.DataFrame(), cost_path)
        _write_csv(pd.DataFrame(), trade_path)
        _write_text(md_path, f"# Local Wizard Verification Blocked\n\n- reason: {exc}\n- history_path: {history_file}\n- wizard_capture_path: {wizard_file}\n")
        return CommandResult(
            paths={
                "summary": summary_path,
                "cost_comparison": cost_path,
                "trade_log": trade_path,
                "summary_md": md_path,
            },
            summary={
                "pair": payload.get("pair", "unknown"),
                "rows": 0,
                "acceptance": "BLOCKED",
                "acceptance_reason": str(exc),
                "closed_trades": 0,
                "profit_factor": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
            },
        )
    wizard_payload = _read_wizard_capture_payload(wizard_file, payload) if wizard_file.exists() else {}
    mode = exact_mode or _mode_from_payload(wizard_payload, payload)
    vendor_fidelity_status, vendor_fidelity_reason = _mode_fidelity(payload, wizard_payload, mode)
    resolved_mode_settings = dict(mode_settings or {})
    local_mode_result: WizardModeReplayResult | None = None
    if vendor_fidelity_status == "vendor_exact":
        signal, trade_log = static_spread_signal(history["zscore"], entry_threshold=entry_threshold, exit_threshold=exit_threshold)
        mode_fidelity_status = vendor_fidelity_status
        mode_fidelity_reason = vendor_fidelity_reason
        signal_metadata = _generic_signal_metadata(
            mode_replay_status="VENDOR_CUSTOM_SERIES_PROVENANCE",
            signal_source="Crypto Wizards custom-series backtest provenance for the captured exact mode",
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )
    elif resolved_mode_settings:
        local_mode_result = build_local_mode_signal(history, resolved_mode_settings, exact_mode=mode)
        signal, trade_log = local_mode_result.signal, local_mode_result.trades
        mode_fidelity_status = local_mode_result.mode_fidelity_status
        mode_fidelity_reason = local_mode_result.mode_fidelity_reason
        signal_metadata = _local_mode_signal_metadata(local_mode_result, resolved_mode_settings)
    else:
        signal, trade_log = static_spread_signal(history["zscore"], entry_threshold=entry_threshold, exit_threshold=exit_threshold)
        mode_fidelity_status = vendor_fidelity_status
        mode_fidelity_reason = vendor_fidelity_reason
        signal_metadata = _generic_signal_metadata(
            mode_replay_status="GENERIC_PROXY_ONLY",
            signal_source=f"local zscore column used as a non-comparable {mode} proxy",
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )
    cost_buckets = _cost_buckets()
    cost_rows = []
    for name, model in cost_buckets.items():
        result = backtest_two_leg_spread(history, signal, model)
        cost_rows.append(_cost_row(name, model, result))
    cost_frame = pd.DataFrame(cost_rows)
    base_result = cost_frame[cost_frame["cost_case"] == "base_cost_used"].iloc[0].to_dict()
    trade_frame = pd.DataFrame(trade_log)
    if not trade_frame.empty:
        trade_frame = _attach_trade_returns(trade_frame, history, signal, cost_buckets["base_cost_used"])

    local_last = pd.to_datetime(history["timestamp"].iloc[-1], utc=True)
    as_of = pd.Timestamp(current_date, tz="UTC")
    age_days = max(0, int((as_of.normalize() - local_last.normalize()).days))
    acceptance, reason = _acceptance(base_result, len(history), age_days, int((trade_frame.get("exit_reason", pd.Series(dtype=str)) != "open_at_end_of_history").sum()))
    if mode_fidelity_status != "vendor_exact":
        acceptance = "BLOCKED"
        reason = _append_reason(reason, mode_fidelity_reason)
    summary = _summary_row(
        payload=payload,
        wizard_payload=wizard_payload,
        history_path=history_file,
        wizard_path=wizard_file,
        rows=len(history),
        trade_frame=trade_frame,
        base_result=base_result,
        acceptance=acceptance,
        reason=reason,
        local_last=local_last,
        current_date=current_date,
        age_days=age_days,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        exact_mode=mode,
        mode_fidelity_status=mode_fidelity_status,
        mode_fidelity_reason=mode_fidelity_reason,
        signal_metadata=signal_metadata,
    )
    reports = root / "reports" / "active"
    summary_path = reports / f"{output_name}_after_cost.csv"
    cost_path = reports / f"{output_name}_cost_comparison.csv"
    trade_path = reports / f"{output_name}_trade_log.csv"
    md_path = reports / f"{output_name}_after_cost.md"
    _write_csv(pd.DataFrame([summary]), summary_path)
    _write_csv(cost_frame, cost_path)
    _write_csv(trade_frame, trade_path)
    _write_text(md_path, _markdown(summary, cost_frame, trade_frame))
    return CommandResult(
        paths={
            "summary": summary_path,
            "cost_comparison": cost_path,
            "trade_log": trade_path,
            "summary_md": md_path,
        },
        summary={
            "pair": summary["pair"],
            "rows": int(summary["local_observations"]),
            "acceptance": acceptance,
            "acceptance_reason": reason,
            "closed_trades": int(summary["closed_trades"]),
            "profit_factor": float(summary["profit_factor"]),
            "sharpe": float(summary["sharpe"]),
            "max_drawdown": float(summary["max_drawdown"]),
        },
    )


def _candidate_rows(*, root: Path, queue_file: Path, max_pairs: int) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    wizard_evidence = _ensure_wizard_evidence(root)
    evidence_index = _primary_wizard_evidence_index(wizard_evidence)
    packet_index = _wizard_candidate_packet_index(root)
    hourly_target_index = _wizard_hourly_target_index(root)
    if queue_file.exists():
        queue = pd.read_csv(queue_file).head(max_pairs)
        for _, row in queue.iterrows():
            candidate = _candidate_from_queue_row(
                row,
                root=root,
                evidence_index=evidence_index,
                packet_index=packet_index,
                hourly_target_index=hourly_target_index,
            )
            key = (str(candidate.get("pair", "")), str(candidate.get("history_path", "")))
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    bnb_history = root / "data" / "raw" / "pair_details" / DEFAULT_HISTORY.name
    bnb_capture = root / "data" / "raw" / "pair_details" / DEFAULT_WIZARD_CAPTURE.name
    if bnb_history.exists() or bnb_capture.exists():
        bnb = {
            "pair": "BNB-USD/STX-USD",
            "asset_x": "BNB-USD",
            "asset_y": "STX-USD",
            "interval": "daily",
            "period": 320,
            "wizard_sharpe": "",
            "wizard_returns_total": "",
            "wizard_returns_total_pct": "",
            "exact_mode": "Static (Spread)",
            "setup_identity": _wizard_setup_identity("BNB-USD/STX-USD", "daily", 320, "Static (Spread)"),
            "setup_role": "primary",
            "spread_id": 3,
            "strategy_id": 1,
            "history_path": bnb_history,
            "wizard_capture_path": bnb_capture,
            "source_row_path": _rel(bnb_capture),
            "candidate_source": "exact_mode_capture",
        }
        key = (str(bnb["pair"]), str(bnb["history_path"]))
        if key not in seen:
            candidates.append(bnb)
    return candidates[:max_pairs]


def _candidate_from_queue_row(
    row: pd.Series,
    *,
    root: Path,
    evidence_index: dict[str, dict[str, object]] | None = None,
    packet_index: dict[str, dict[str, object]] | None = None,
    hourly_target_index: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    pair_text = _text_value(row.get("pair", ""))
    evidence_row = _match_primary_wizard_evidence(pair_text, row.get("asset_x", ""), row.get("asset_y", ""), evidence_index or {})
    packet_row = _match_indexed_pair_row(pair_text, row.get("asset_x", ""), row.get("asset_y", ""), packet_index or {})
    hourly_target_row = _match_indexed_pair_row(pair_text, row.get("asset_x", ""), row.get("asset_y", ""), hourly_target_index or {})
    asset_x = _text_value(row.get("asset_x", "")) or _text_value((evidence_row or {}).get("asset_x", ""))
    asset_y = _text_value(row.get("asset_y", "")) or _text_value((evidence_row or {}).get("asset_y", ""))
    interval = (
        _text_value(row.get("interval", ""))
        or _text_value((evidence_row or {}).get("interval", ""))
        or _text_value((packet_row or {}).get("timeframe", ""))
        or _text_value((hourly_target_row or {}).get("timeframe", ""))
        or "daily"
    )
    explicit_history_path = _path_from_value(row.get("pair_history_path", ""), root=root)
    fallback_history_path = _find_local_history_path(root=root, asset_x=str(asset_x), asset_y=str(asset_y), interval=str(interval))
    history_path = explicit_history_path
    if history_path is not None and history_path.exists() and _path_is_usable_local_history(history_path):
        pass
    elif fallback_history_path is not None and fallback_history_path.exists():
        history_path = fallback_history_path
    elif history_path is not None and history_path.exists():
        pass
    else:
        history_path = None
    source_hint = (
        _text_value((evidence_row or {}).get("source_path", ""))
        or _text_value((packet_row or {}).get("source_path", ""))
        or _text_value(row.get("source_path", ""))
    )
    wizard_source_path = _path_from_value(source_hint, root=root)
    source_payload = _read_json(history_path) if history_path and history_path.exists() else {}
    exact_mode = (
        _text_value(row.get("exact_mode"))
        or _text_value((evidence_row or {}).get("exact_mode", ""))
        or _text_value((packet_row or {}).get("strategy_mode", ""))
        or _text_value((hourly_target_row or {}).get("matched_hourly_strategy", ""))
        or _text_value(row.get("dashboard_recommended_strategy"))
        or _text_value((evidence_row or {}).get("dashboard_recommended_strategy", ""))
        or _text_value(source_payload.get("exact_mode"))
        or _mode_from_strategy(row.get("strategy", row.get("strategy_family_note", source_payload.get("strategy_mode", ""))))
    )
    period = (
        source_payload.get("period")
        or (evidence_row or {}).get("period")
        or (packet_row or {}).get("period")
        or row.get("period", "")
    )
    wizard_capture_path = _find_wizard_capture_path(
        root=root,
        pair=pair_text,
        asset_x=str(asset_x),
        asset_y=str(asset_y),
        interval=str(interval or source_payload.get("interval", "")),
        exact_mode=str(exact_mode),
        source_path=wizard_source_path,
    )
    return {
        "pair": pair_text or _text_value((evidence_row or {}).get("pair", "")),
        "asset_x": asset_x or source_payload.get("asset_x", ""),
        "asset_y": asset_y or source_payload.get("asset_y", ""),
        "interval": interval or source_payload.get("interval", ""),
        "period": period,
        "wizard_sharpe": _first_nonblank(row.get("sharpe"), (evidence_row or {}).get("sharpe"), source_payload.get("sharpe", "")),
        "wizard_returns_total": _first_nonblank(row.get("returns_total"), row.get("return_pct"), (evidence_row or {}).get("returns_total"), source_payload.get("returns_total", "")),
        "wizard_returns_total_pct": _first_nonblank(row.get("returns_total_pct"), row.get("return_pct"), (evidence_row or {}).get("returns_total_pct"), ""),
        "exact_mode": exact_mode,
        "setup_identity": _text_value((evidence_row or {}).get("setup_identity", "")) or _text_value((packet_row or {}).get("setup_identity", "")) or _wizard_setup_identity(str(pair_text or (evidence_row or {}).get("pair", "")), str(interval or source_payload.get("interval", "")), _maybe_int(period), str(exact_mode)),
        "setup_role": _text_value((evidence_row or {}).get("setup_role", "")) or _text_value((packet_row or {}).get("setup_role", "")) or "primary",
        "spread_id": _first_nonblank(row.get("spread_id"), (evidence_row or {}).get("spread_id"), (packet_row or {}).get("spread_id"), source_payload.get("spread_id", "")),
        "strategy_id": _first_nonblank(row.get("strategy_id"), (evidence_row or {}).get("strategy_id"), (packet_row or {}).get("strategy_id"), source_payload.get("strategy_id", "")),
        "history_path": history_path,
        "wizard_capture_path": wizard_capture_path,
        "source_row_path": _text_value(row.get("source_path", "")) or _text_value((evidence_row or {}).get("source_path", "")),
        "candidate_source": row.get("source_group", "wizard_queue"),
        "execution_bucket": row.get("execution_bucket", ""),
        "execution_blockers": row.get("execution_blockers", ""),
        "research_blockers_only": row.get("research_blockers_only", ""),
        "recommended_action": row.get("recommended_action", ""),
    }


def _verify_candidate(*, root: Path, candidate: dict[str, object], current_date: str) -> dict[str, object]:
    mode_settings = _validated_mode_settings_for_candidate(root, candidate)
    blockers = _candidate_blockers(candidate, has_mode_settings=bool(mode_settings))
    base = _candidate_base_row(candidate)
    if blockers:
        return {
            **base,
            "verification_status": "blocked",
            "verification_blocker": ";".join(blockers),
            "acceptance": "BLOCKED",
            "acceptance_reason": "local_verification_not_run",
        }
    output_name = _candidate_output_name(candidate)
    try:
        result = verify_wizard_local_mode(
            root=root,
            history_path=Path(candidate["history_path"]),
            wizard_capture_path=Path(candidate["wizard_capture_path"]),
            output_name=output_name,
            current_date=current_date,
            exact_mode=str(candidate.get("exact_mode", "")),
            mode_settings=mode_settings or None,
        )
        summary = pd.read_csv(result.paths["summary"]).iloc[0].to_dict()
        mode_fidelity_status = _text_value(summary.get("mode_fidelity_status", ""))
        exact_verified = mode_fidelity_status == "vendor_exact"
        return {
            **base,
            "verification_status": "verified" if exact_verified else "proxy_only",
            "verification_blocker": "" if exact_verified else _text_value(summary.get("mode_fidelity_reason", "mode_fidelity_not_exact")),
            "mode_fidelity_status": mode_fidelity_status,
            "mode_fidelity_reason": _text_value(summary.get("mode_fidelity_reason", "")),
            "mode_replay_status": _text_value(summary.get("mode_replay_status", "")),
            "mode_metric_name": _text_value(summary.get("mode_metric_name", "")),
            "mode_missing_inputs": _text_value(summary.get("mode_missing_inputs", "")),
            "acceptance": summary.get("acceptance", ""),
            "acceptance_reason": summary.get("acceptance_reason", ""),
            "local_observations": summary.get("local_observations", ""),
            "local_sharpe": summary.get("sharpe", ""),
            "local_profit_factor": summary.get("profit_factor", ""),
            "local_total_return": summary.get("total_return", ""),
            "local_max_drawdown": summary.get("max_drawdown", ""),
            "local_trades": summary.get("trades", ""),
            "local_closed_trades": summary.get("closed_trades", ""),
            "local_stale_reason": summary.get("stale_reason", ""),
            "summary_path": _rel(result.paths["summary"]),
            "trade_log_path": _rel(result.paths["trade_log"]),
            "cost_comparison_path": _rel(result.paths["cost_comparison"]),
        }
    except Exception as exc:  # pragma: no cover - defensive report hygiene
        return {
            **base,
            "verification_status": "error",
            "verification_blocker": f"{type(exc).__name__}:{exc}",
            "acceptance": "BLOCKED",
            "acceptance_reason": "local_verification_error",
        }


def _candidate_base_row(candidate: dict[str, object]) -> dict[str, object]:
    return {
        "pair": candidate.get("pair", ""),
        "asset_x": candidate.get("asset_x", ""),
        "asset_y": candidate.get("asset_y", ""),
        "interval": candidate.get("interval", ""),
        "period": candidate.get("period", ""),
        "setup_identity": candidate.get("setup_identity", ""),
        "setup_role": candidate.get("setup_role", "primary"),
        "exact_mode": candidate.get("exact_mode", ""),
        "spread_id": candidate.get("spread_id", ""),
        "strategy_id": candidate.get("strategy_id", ""),
        "wizard_sharpe": candidate.get("wizard_sharpe", ""),
        "wizard_returns_total": candidate.get("wizard_returns_total", ""),
        "wizard_returns_total_pct": candidate.get("wizard_returns_total_pct", ""),
        "candidate_source": candidate.get("candidate_source", ""),
        "execution_bucket": candidate.get("execution_bucket", ""),
        "execution_blockers": candidate.get("execution_blockers", ""),
        "research_blockers_only": candidate.get("research_blockers_only", ""),
        "recommended_action": candidate.get("recommended_action", ""),
        "history_path": _rel(Path(candidate["history_path"])) if candidate.get("history_path") else "",
        "wizard_capture_path": _rel(Path(candidate["wizard_capture_path"])) if candidate.get("wizard_capture_path") else "",
        "source_row_path": candidate.get("source_row_path", ""),
    }


def _validated_mode_settings_for_candidate(root: Path, candidate: dict[str, object]) -> dict[str, object]:
    """Return a validated dashboard capture for one exact setup, if present."""

    path = root / "reports" / "active" / "crypto_wizards_pair_page_capture_settings.csv"
    if not path.exists():
        return {}
    try:
        captures = pd.read_csv(path, dtype=object).fillna("")
    except (OSError, pd.errors.EmptyDataError):
        return {}
    if captures.empty:
        return {}
    setup_identity = _text_value(candidate.get("setup_identity", ""))
    if setup_identity and "setup_identity" in captures.columns:
        exact = captures[captures["setup_identity"].map(_text_value).eq(setup_identity)]
        if len(exact) == 1:
            return exact.iloc[0].to_dict()
    pair = _normalize_pair(str(candidate.get("pair", "")), str(candidate.get("asset_x", "")), str(candidate.get("asset_y", "")))
    interval = _normalize_interval(candidate.get("interval", ""))
    mode = str(candidate.get("exact_mode", "")).strip().lower()
    period = _maybe_int(candidate.get("period", ""))
    matches: list[dict[str, object]] = []
    for _, row in captures.iterrows():
        row_pair = _normalize_pair(str(row.get("pair", "")), str(row.get("asset_x", "")), str(row.get("asset_y", "")))
        row_interval = _normalize_interval(row.get("interval", ""))
        row_mode = str(row.get("exact_mode", "")).strip().lower()
        row_period = _maybe_int(row.get("period", ""))
        if row_pair != pair or (interval and row_interval != interval) or (mode and row_mode != mode):
            continue
        if period is not None and row_period is not None and row_period != period:
            continue
        matches.append(row.to_dict())
    return matches[0] if len(matches) == 1 else {}


def _candidate_blockers(candidate: dict[str, object], *, has_mode_settings: bool = False) -> list[str]:
    blockers: list[str] = []
    history_path = candidate.get("history_path")
    wizard_path = candidate.get("wizard_capture_path")
    exact_mode = str(candidate.get("exact_mode", "")).strip().lower()
    if not exact_mode:
        blockers.append("missing_exact_mode_capture")
    elif exact_mode not in {
        "static (spread)",
        "static spread",
        "ou (spread)",
        "ou spread",
        "copula",
        "static (zscorer)",
        "static zscorer",
        "static (zscore)",
        "static zscore",
        "dyn (zscorer)",
        "dyn zscorer",
        "dynamic (zscorer)",
        "dynamic zscorer",
        "dyn (zscore)",
        "dyn zscore",
        "dynamic (zscore)",
        "dynamic zscore",
        "ou (zscorer)",
        "ou zscorer",
        "ou (zscore)",
        "ou zscore",
    }:
        blockers.append(f"unsupported_exact_mode:{candidate.get('exact_mode')}")
    if not history_path:
        blockers.append("missing_local_history_path")
    elif not Path(history_path).exists():
        blockers.append("local_history_file_missing")
    else:
        payload = _read_json(Path(history_path))
        history = pd.DataFrame(payload.get("history", []))
        required = {"timestamp", "price_x", "price_y"}
        missing = sorted(required - set(history.columns))
        if not missing and not has_mode_settings and not _has_zscore_like_columns(history):
            missing.append("zscore_for_generic_proxy")
        if missing:
            blockers.append(f"local_history_missing_columns:{','.join(missing)}")
    if not wizard_path or not Path(wizard_path).exists():
        blockers.append("missing_wizard_capture_path")
    return blockers


def _candidate_output_name(candidate: dict[str, object]) -> str:
    pair = str(candidate.get("pair", "candidate")).replace("/", "_").replace("-", "").lower()
    mode = str(candidate.get("exact_mode", "mode")).replace("(", "").replace(")", "").replace(" ", "_").lower()
    return f"{pair}_{mode}_verification"


def _maybe_int(value: object) -> int | None:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _has_zscore_like_columns(frame: pd.DataFrame) -> bool:
    return bool({"zscore", "zscore_reconstructed", "rolling_zscore"}.intersection(frame.columns))


def _path_from_value(value: object, *, root: Path) -> Path | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else root / path


def _find_local_history_path(*, root: Path, asset_x: str, asset_y: str, interval: str) -> Path | None:
    if not asset_x or not asset_y:
        return None
    pair_dir = root / "data" / "raw" / "pair_details"
    left_tokens = _asset_match_tokens(asset_x)
    right_tokens = _asset_match_tokens(asset_y)
    candidates = []
    for path in sorted(pair_dir.glob("*.json")):
        text = path.name.lower()
        if any(token in text for token in left_tokens) and any(token in text for token in right_tokens):
            candidates.append(path)
    daily = str(interval).strip().lower() in {"daily", "1day", "day"}
    matches: list[tuple[int, pd.Timestamp, Path]] = []
    for path in candidates:
        payload = _read_json(path)
        history = pd.DataFrame(payload.get("history", []))
        if {"timestamp", "price_x", "price_y"}.issubset(history.columns) and _has_zscore_like_columns(history):
            text = path.name.lower()
            if not daily or "1day" in text or "daily" in text:
                last = pd.to_datetime(history["timestamp"], utc=True, errors="coerce").max()
                matches.append((len(history), last if pd.notna(last) else pd.Timestamp.min.tz_localize("UTC"), path))
    if matches:
        return sorted(matches, key=lambda item: (item[0], item[1]), reverse=True)[0][2]
    fallback_matches: list[tuple[int, pd.Timestamp, Path]] = []
    for path in candidates:
        payload = _read_json(path)
        history = pd.DataFrame(payload.get("history", []))
        if {"timestamp", "price_x", "price_y"}.issubset(history.columns) and _has_zscore_like_columns(history):
            last = pd.to_datetime(history["timestamp"], utc=True, errors="coerce").max()
            fallback_matches.append((len(history), last if pd.notna(last) else pd.Timestamp.min.tz_localize("UTC"), path))
    return sorted(fallback_matches, key=lambda item: (item[0], item[1]), reverse=True)[0][2] if fallback_matches else None


def _asset_match_tokens(asset: str) -> set[str]:
    text = _text_value(asset).lower()
    if not text:
        return set()
    compact = text.replace("-", "").replace("_", "")
    underscore = text.replace("-", "_")
    base = text.replace("-usd", "").replace("_usd", "").replace("usd", "").strip("-_")
    return {token for token in {text, compact, underscore, base} if token}


def _path_is_usable_local_history(path: Path) -> bool:
    try:
        payload = _read_json(path)
    except Exception:
        return False
    history = pd.DataFrame(payload.get("history", []))
    return {"timestamp", "price_x", "price_y"}.issubset(history.columns) and _has_zscore_like_columns(history)


def _primary_wizard_evidence_index(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    if frame.empty:
        return {}
    working = frame.copy()
    primary = working.get("primary_wizard_setup", pd.Series(False, index=working.index)).map(_as_bool_like)
    if primary.any():
        working = working.loc[primary].copy()
    index: dict[str, dict[str, object]] = {}
    for _, row in working.iterrows():
        key = _normalize_pair(str(row.get("pair", "")), str(row.get("asset_x", "")), str(row.get("asset_y", "")))
        if key and key not in index:
            index[key] = row.to_dict()
    return index


def _match_primary_wizard_evidence(pair: str, asset_x: object, asset_y: object, evidence_index: dict[str, dict[str, object]]) -> dict[str, object] | None:
    key = _normalize_pair(pair, _text_value(asset_x), _text_value(asset_y))
    return evidence_index.get(key)


def _mode_from_strategy(value: object) -> str:
    text = str(value).strip().lower().replace("_", "").replace(" ", "")
    if "copula" in text:
        return "Copula"
    if "kalman" in text or "dynamic" in text or text in {"dynspread", "dyn"}:
        return "Dyn (Spread)"
    if text in {"ouspread", "ou"}:
        return "OU (Spread)"
    if text in {"ouzscorer", "ouzscore", "ouzscoreroll"}:
        return "OU (ZScoreR)"
    if text in {"staticspread", "static"}:
        return "Static (Spread)"
    if text in {"staticzscorer", "staticzscore", "staticzscoreroll"}:
        return "Static (ZScoreR)"
    if "zscore" in text:
        return "Static (ZScoreR)"
    return ""


def _mode_from_payload(wizard_payload: dict[str, object], local_payload: dict[str, object]) -> str:
    explicit = str(wizard_payload.get("exact_mode", "") or local_payload.get("exact_mode", "")).strip()
    if explicit:
        return explicit
    strategy_hint = (
        wizard_payload.get("dashboard_recommended_strategy")
        or wizard_payload.get("selected_strategy_value")
        or wizard_payload.get("strategy_name")
        or wizard_payload.get("strategy_mode")
        or local_payload.get("dashboard_recommended_strategy")
        or local_payload.get("strategy_mode", "")
    )
    return _mode_from_strategy(strategy_hint) or "Static (Spread)"


def _mode_fidelity(
    local_payload: dict[str, object],
    wizard_payload: dict[str, object],
    exact_mode: str,
) -> tuple[str, str]:
    """Return whether the local calculation is a real vendor-mode replay or a proxy."""

    source = _text_value(
        local_payload.get("mode_computation_source")
        or wizard_payload.get("mode_computation_source")
        or local_payload.get("source_mode_provenance")
        or wizard_payload.get("source_mode_provenance")
    ).lower()
    recorded_mode = _text_value(local_payload.get("exact_mode") or wizard_payload.get("exact_mode"))
    if source == "crypto_wizards_custom_series_backtest" and _same_mode(recorded_mode, exact_mode):
        return "vendor_exact", ""
    if not recorded_mode:
        return "generic_zscore_proxy", "exact_mode_not_recorded_in_local_mode_engine"
    if not _same_mode(recorded_mode, exact_mode):
        return "generic_zscore_proxy", "local_mode_does_not_match_wizard_exact_mode"
    return "generic_zscore_proxy", "local_zscore_proxy_not_vendor_custom_series_replay"


def _same_mode(left: str, right: str) -> bool:
    normalize = lambda value: "".join(character for character in str(value).lower() if character.isalnum())
    return bool(normalize(left) and normalize(left) == normalize(right))


def _append_reason(reason: str, addition: str) -> str:
    values = [value for value in [str(reason or "").strip(), str(addition or "").strip()] if value]
    return ";".join(dict.fromkeys(values))


def _find_wizard_capture_path(
    *,
    root: Path,
    pair: str,
    asset_x: str,
    asset_y: str,
    interval: str,
    exact_mode: str,
    source_path: Path | None,
) -> Path | None:
    if source_path and source_path.exists():
        return source_path
    for candidate_path in [
        root / "reports" / "active" / "crypto_wizards_pair_page_capture.csv",
        root / "reports" / "active" / "wizard_scanner_dependency_capture.csv",
    ]:
        matched = _capture_csv_matches(
            candidate_path,
            pair=pair,
            asset_x=asset_x,
            asset_y=asset_y,
            interval=interval,
            exact_mode=exact_mode,
        )
        if matched:
            return candidate_path
    return None


def _capture_csv_matches(path: Path, *, pair: str, asset_x: str, asset_y: str, interval: str, exact_mode: str) -> bool:
    if not path.exists():
        return False
    try:
        capture = pd.read_csv(path)
    except Exception:
        return False
    if capture.empty:
        return False
    pair_norm = _normalize_pair(pair, asset_x, asset_y)
    interval_norm = str(interval).strip().lower()
    exact_norm = str(exact_mode).strip().lower()
    for _, row in capture.iterrows():
        row_pair = _normalize_pair(str(row.get("pair", "")), str(row.get("asset_x", "")), str(row.get("asset_y", "")))
        row_interval = str(row.get("interval", "") or row.get("timeframe", "")).strip().lower()
        row_exact = str(
            row.get("exact_mode", "")
            or row.get("dashboard_recommended_strategy", "")
            or row.get("strategy_mode", "")
            or row.get("matched_hourly_strategy", "")
        ).strip().lower()
        if row_pair == pair_norm and (not interval_norm or not row_interval or row_interval == interval_norm) and (not exact_norm or row_exact == exact_norm):
            return True
    return False


def _normalize_pair(pair: str, asset_x: str, asset_y: str) -> str:
    text = str(pair).strip()
    if text:
        return text.replace("/", "-").upper()
    if asset_x and asset_y:
        return f"{asset_x}-{asset_y}".replace("/", "-").upper()
    return ""


def _read_wizard_capture_payload(path: Path, local_payload: dict[str, object]) -> dict[str, object]:
    if path.suffix.lower() == ".csv":
        try:
            capture = pd.read_csv(path)
        except Exception:
            return {}
        if capture.empty:
            return {}
        local_pair = _normalize_pair(
            str(local_payload.get("pair", "")),
            str(local_payload.get("asset_x", "")),
            str(local_payload.get("asset_y", "")),
        )
        local_interval = str(local_payload.get("interval", "")).strip().lower()
        pair_only_match: dict[str, object] | None = None
        for _, row in capture.iterrows():
            row_pair = _normalize_pair(str(row.get("pair", "")), str(row.get("asset_x", "")), str(row.get("asset_y", "")))
            row_interval = str(row.get("interval", "") or row.get("timeframe", "")).strip().lower()
            if row_pair == local_pair and (not local_interval or row_interval == local_interval):
                return row.to_dict()
            if row_pair == local_pair and pair_only_match is None:
                pair_only_match = row.to_dict()
        if pair_only_match is not None:
            return pair_only_match
        return capture.iloc[0].to_dict()
    return _read_json(path)


def _wizard_candidate_packet_index(root: Path) -> dict[str, dict[str, object]]:
    path = root / "reports" / "brain" / "wizard_candidate_packets.csv"
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    index: dict[str, dict[str, object]] = {}
    for _, row in frame.iterrows():
        key = _normalize_pair(str(row.get("pair", "")), "", "")
        if key and key not in index:
            index[key] = row.to_dict()
    return index


def _wizard_hourly_target_index(root: Path) -> dict[str, dict[str, object]]:
    path = root / "reports" / "brain" / "wizard_hourly_repair_targets.csv"
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    index: dict[str, dict[str, object]] = {}
    for _, row in frame.iterrows():
        key = _normalize_pair(str(row.get("pair", "")), "", "")
        if key and key not in index:
            index[key] = row.to_dict()
    return index


def _match_indexed_pair_row(pair: str, asset_x: object, asset_y: object, index: dict[str, dict[str, object]]) -> dict[str, object] | None:
    key = _normalize_pair(pair, _text_value(asset_x), _text_value(asset_y))
    return index.get(key)


def _text_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _as_bool_like(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = _text_value(value).lower()
    return text in {"1", "true", "yes", "y"}


def _first_nonblank(*values: object) -> object:
    for value in values:
        text = _text_value(value)
        if text:
            return value
    return ""


def _ids_from_mode(exact_mode: str) -> tuple[int | str, int | str]:
    text = exact_mode.strip().lower()
    if text in {"dynamic (spread)", "dynamic spread"}:
        return 1, 1
    if text in {"dynamic (zscorer)", "dynamic zscorer", "dynamic zscore"}:
        return 1, 2
    if text in {"ou (spread)", "ou spread"}:
        return 2, 1
    if text in {"ou (zscorer)", "ou zscorer", "ou zscore"}:
        return 2, 2
    if text in {"static (spread)", "static spread"}:
        return 3, 1
    if text in {"static (zscorer)", "static zscorer", "static zscore"}:
        return 3, 2
    if text == "copula":
        return 1, 3
    return "", ""


def _batch_markdown(frame: pd.DataFrame, queue_file: Path) -> str:
    lines = [
        "# Wizard Local Verification Batch",
        "",
        f"- Queue: `{_rel(queue_file)}`",
        f"- Candidates: `{len(frame)}`",
    ]
    if frame.empty:
        return "\n".join(lines + ["", "No candidates found.", ""])
    display_frame = frame.reindex(
        columns=[
            "pair",
            "exact_mode",
            "wizard_sharpe",
            "wizard_returns_total",
            "verification_status",
            "acceptance",
            "acceptance_reason",
            "verification_blocker",
            "local_sharpe",
            "local_total_return",
            "local_max_drawdown",
            "local_closed_trades",
        ],
        fill_value="",
    )
    verified = int((frame["verification_status"] == "verified").sum())
    accepted = int((frame["acceptance"] == "ACCEPT").sum())
    lines.extend(
        [
            f"- Verified locally: `{verified}`",
            f"- Accepted: `{accepted}`",
            "",
            "## Ranked Board",
            "",
            display_frame.to_markdown(index=False),
            "",
        ]
    )
    return "\n".join(lines)


def static_spread_signal(
    zscore: pd.Series,
    *,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
) -> tuple[pd.Series, list[dict[str, object]]]:
    signal: list[float] = []
    state = 0.0
    current: dict[str, object] | None = None
    trades: list[dict[str, object]] = []
    timestamps = zscore.index
    for i, z_raw in enumerate(pd.to_numeric(zscore, errors="coerce")):
        z = float(z_raw) if pd.notna(z_raw) else np.nan
        timestamp = timestamps[i]
        if state == 0.0 and pd.notna(z):
            if z >= entry_threshold:
                state = -1.0
                current = {
                    "trade_id": len(trades) + 1,
                    "entry_timestamp": timestamp,
                    "entry_zscore": z,
                    "direction": "long_x_short_y",
                    "start_i": i,
                }
            elif z <= -entry_threshold:
                state = 1.0
                current = {
                    "trade_id": len(trades) + 1,
                    "entry_timestamp": timestamp,
                    "entry_zscore": z,
                    "direction": "short_x_long_y",
                    "start_i": i,
                }
        elif state == -1.0 and pd.notna(z) and z <= exit_threshold:
            if current is not None:
                current.update(
                    {
                        "exit_timestamp": timestamp,
                        "exit_zscore": z,
                        "end_i": i,
                        "bars_held": i - int(current["start_i"]) + 1,
                        "exit_reason": "static_spread_zero_cross",
                    }
                )
                trades.append(current)
            current = None
            state = 0.0
        elif state == 1.0 and pd.notna(z) and z >= -exit_threshold:
            if current is not None:
                current.update(
                    {
                        "exit_timestamp": timestamp,
                        "exit_zscore": z,
                        "end_i": i,
                        "bars_held": i - int(current["start_i"]) + 1,
                        "exit_reason": "static_spread_zero_cross",
                    }
                )
                trades.append(current)
            current = None
            state = 0.0
        signal.append(state)
    if current is not None:
        current.update(
            {
                "exit_timestamp": "",
                "exit_zscore": "",
                "end_i": len(signal) - 1,
                "bars_held": len(signal) - int(current["start_i"]),
                "exit_reason": "open_at_end_of_history",
            }
        )
        trades.append(current)
    return pd.Series(signal, index=zscore.index, dtype="float64"), trades


def _history_frame(payload: dict[str, object]) -> pd.DataFrame:
    history = pd.DataFrame(payload.get("history", []))
    if history.empty:
        raise ValueError("local history has no rows")
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    history = history.sort_values("timestamp").set_index("timestamp", drop=False)
    for column in ["price_x", "price_y", "zscore", "spread", "hedge_ratio", "beta", "funding_x_bps", "funding_y_bps", "funding_bps_per_day"]:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")
    if "funding_bps_per_day" in history.columns:
        history["funding_x_bps"] = history.get("funding_x_bps", history["funding_bps_per_day"])
        history["funding_y_bps"] = history.get("funding_y_bps", history["funding_bps_per_day"])
    return history


def _cost_buckets() -> dict[str, CostModel]:
    return {
        "zero_cost": CostModel(
            taker_fee_bps=0.0,
            slippage_bps=0.0,
            execution_risk_bps=0.0,
            funding_bps_per_day=0.0,
            bars_per_day=1,
            partial_fill_probability=0.0,
            partial_fill_penalty_bps=0.0,
        ),
        "base_cost_used": CostModel(bars_per_day=1),
        "stress_cost": CostModel(
            taker_fee_bps=7.5,
            slippage_bps=8.0,
            execution_risk_bps=4.0,
            funding_bps_per_day=3.0,
            bars_per_day=1,
        ),
    }


def _cost_row(name: str, model: CostModel, result: BacktestResult) -> dict[str, object]:
    row = {"cost_case": name, **asdict(model), **asdict(result)}
    row["total_cost_drag"] = row["total_fees"] + row["total_slippage"] + row["total_funding"] + row["total_execution_risk"] + row["total_partial_fill_cost"]
    return row


def _attach_trade_returns(
    trade_frame: pd.DataFrame,
    history: pd.DataFrame,
    signal: pd.Series,
    cost_model: CostModel,
) -> pd.DataFrame:
    data = history.copy()
    data["signal"] = signal.reindex(data.index).fillna(0.0)
    price_x = pd.to_numeric(data["price_x"], errors="coerce").ffill()
    price_y = pd.to_numeric(data["price_y"], errors="coerce").ffill()
    returns_x = price_x.pct_change().fillna(0.0)
    returns_y = price_y.pct_change().fillna(0.0)
    hedge_ratio = pd.to_numeric(data.get("hedge_ratio", 1.0), errors="coerce").fillna(1.0)
    signal_position = data["signal"].shift(1).fillna(0.0)
    gross_scale = 1.0 + hedge_ratio.abs()
    weight_y = signal_position / gross_scale
    weight_x = -signal_position * hedge_ratio / gross_scale
    gross_return = weight_x * returns_x + weight_y * returns_y
    target_weight_y = data["signal"] / gross_scale
    target_weight_x = -data["signal"] * hedge_ratio / gross_scale
    turnover = target_weight_x.diff().abs().fillna(target_weight_x.abs()) + target_weight_y.diff().abs().fillna(target_weight_y.abs())
    funding_x = _series_or_default(data, "funding_x_bps", cost_model.funding_bps_per_day)
    funding_y = _series_or_default(data, "funding_y_bps", cost_model.funding_bps_per_day)
    costs = (
        turnover * cost_model.taker_fee_bps / 10_000.0
        + turnover * cost_model.slippage_bps / 10_000.0
        + turnover * cost_model.execution_risk_bps / 10_000.0
        + turnover * cost_model.partial_fill_probability * (1.0 - cost_model.partial_fill_fraction) * cost_model.partial_fill_penalty_bps / 10_000.0
        + (weight_x.abs() * funding_x.abs() / 10_000.0 / cost_model.bars_per_day + weight_y.abs() * funding_y.abs() / 10_000.0 / cost_model.bars_per_day)
    )
    net_return = gross_return - costs
    rows = []
    for _, trade in trade_frame.iterrows():
        row = trade.to_dict()
        if row.get("exit_reason") == "open_at_end_of_history":
            row["profit_after_cost"] = ""
        else:
            start = int(row["start_i"])
            end = int(row["end_i"])
            row["profit_after_cost"] = float((1.0 + net_return.iloc[start : end + 1]).prod() - 1.0)
        rows.append(row)
    return pd.DataFrame(rows).drop(columns=["start_i", "end_i"], errors="ignore")


def _series_or_default(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _acceptance(base_result: dict[str, object], rows: int, age_days: int, closed_trades: int) -> tuple[str, str]:
    blockers = []
    if rows < 320:
        blockers.append("local_history_rows<320")
    if float(base_result.get("total_return", 0.0)) <= 0:
        blockers.append("total_return<=0")
    sharpe = float(base_result.get("sharpe", float("nan")))
    if not np.isfinite(sharpe):
        blockers.append("sharpe_invalid_or_unknown_interval")
    elif sharpe < 1.2:
        blockers.append("sharpe<1.2")
    pf = float(base_result.get("profit_factor", 0.0))
    if not np.isinf(pf) and pf < 1.5:
        blockers.append("profit_factor<1.5")
    if float(base_result.get("max_drawdown", 0.0)) > 0.25:
        blockers.append("max_drawdown>25pct")
    if age_days > 2:
        blockers.append("stale_data")
    if closed_trades < 3:
        blockers.append("thin_trade_count")
    if blockers:
        return "REJECT", ";".join(blockers)
    return "ACCEPT", "passed_local_after_cost_gate"


def _summary_row(
    *,
    payload: dict[str, object],
    wizard_payload: dict[str, object],
    history_path: Path,
    wizard_path: Path,
    rows: int,
    trade_frame: pd.DataFrame,
    base_result: dict[str, object],
    acceptance: str,
    reason: str,
    local_last: pd.Timestamp,
    current_date: str,
    age_days: int,
    entry_threshold: float,
    exit_threshold: float,
    exact_mode: str,
    mode_fidelity_status: str,
    mode_fidelity_reason: str,
    signal_metadata: dict[str, object],
) -> dict[str, object]:
    closed = int((trade_frame.get("exit_reason", pd.Series(dtype=str)) != "open_at_end_of_history").sum()) if not trade_frame.empty else 0
    spread_id, strategy_id = _ids_from_mode(exact_mode)
    return {
        "pair": f"{payload.get('asset_x', '')}/{payload.get('asset_y', '')}",
        "asset_x": payload.get("asset_x", ""),
        "asset_y": payload.get("asset_y", ""),
        "timeframe": _normalize_interval(payload.get("interval", "")),
        "wizard_periods": wizard_payload.get("period", payload.get("period", 320)),
        "local_observations": rows,
        "exact_mode": exact_mode,
        "spread_id": spread_id,
        "strategy_id": strategy_id,
        "mode_fidelity_status": mode_fidelity_status,
        "mode_fidelity_reason": mode_fidelity_reason,
        "mode_replay_status": signal_metadata.get("mode_replay_status", ""),
        "mode_metric_name": signal_metadata.get("mode_metric_name", ""),
        "mode_missing_inputs": signal_metadata.get("mode_missing_inputs", ""),
        "mode_computation_notes": signal_metadata.get("mode_computation_notes", ""),
        "mode_settings_evidence_path": signal_metadata.get("mode_settings_evidence_path", ""),
        "signal_source": signal_metadata.get("signal_source", ""),
        "entry_long_x": signal_metadata.get("entry_long_x", f">= {entry_threshold:.2f}"),
        "entry_short_x": signal_metadata.get("entry_short_x", f"<= {-entry_threshold:.2f}"),
        "exit_long_x": signal_metadata.get("exit_long_x", f"<= {exit_threshold:.2f}"),
        "exit_short_x": signal_metadata.get("exit_short_x", f">= {exit_threshold:.2f}"),
        "backtest_mode": "two_leg_daily",
        "trades": int(base_result.get("trades", 0)),
        "entries": int(len(trade_frame)),
        "closed_trades": closed,
        "open_trades": int(len(trade_frame) - closed),
        "profit_factor": base_result.get("profit_factor", 0.0),
        "sharpe": base_result.get("sharpe", 0.0),
        "max_drawdown": base_result.get("max_drawdown", 0.0),
        "win_rate": base_result.get("win_rate", 0.0),
        "total_return": base_result.get("total_return", 0.0),
        "gross_return": base_result.get("gross_return", 0.0),
        "expectancy": base_result.get("expectancy", 0.0),
        "total_fees": base_result.get("total_fees", 0.0),
        "total_slippage": base_result.get("total_slippage", 0.0),
        "total_funding": base_result.get("total_funding", 0.0),
        "total_execution_risk": base_result.get("total_execution_risk", 0.0),
        "total_partial_fill_cost": base_result.get("total_partial_fill_cost", 0.0),
        "acceptance": acceptance,
        "acceptance_reason": reason,
        "promotion_allowed": bool(mode_fidelity_status == "vendor_exact" and acceptance == "ACCEPT"),
        "wizard_evidence_path": _rel(wizard_path),
        "local_evidence_path": _rel(history_path),
        "local_last_timestamp": local_last.isoformat(),
        "current_date_assumed": current_date,
        "data_age_days": age_days,
        "stale_reason": "" if age_days <= 2 else f"local_daily_history_ends_{age_days}_days_before_current_date",
    }


def _generic_signal_metadata(
    *,
    mode_replay_status: str,
    signal_source: str,
    entry_threshold: float,
    exit_threshold: float,
) -> dict[str, object]:
    return {
        "mode_replay_status": mode_replay_status,
        "mode_metric_name": "zscore",
        "mode_missing_inputs": "",
        "mode_computation_notes": "",
        "mode_settings_evidence_path": "",
        "signal_source": signal_source,
        "entry_long_x": f">= {entry_threshold:.2f}",
        "entry_short_x": f"<= {-entry_threshold:.2f}",
        "exit_long_x": f"<= {exit_threshold:.2f}",
        "exit_short_x": f">= {exit_threshold:.2f}",
    }


def _local_mode_signal_metadata(
    result: WizardModeReplayResult,
    settings: Mapping[str, object],
) -> dict[str, object]:
    return {
        "mode_replay_status": result.mode_replay_status,
        "mode_metric_name": result.metric_name,
        "mode_missing_inputs": ";".join(result.missing_inputs),
        "mode_computation_notes": ";".join(result.computation_notes),
        "mode_settings_evidence_path": _text_value(settings.get("capture_evidence_path", "")),
        "signal_source": f"captured-settings local {result.exact_mode} formula approximation; never vendor-exact",
        "entry_long_x": _captured_rule(settings, "entry_long", include_position=True),
        "entry_short_x": _captured_rule(settings, "entry_short", include_position=True),
        "exit_long_x": _captured_rule(settings, "exit_long"),
        "exit_short_x": _captured_rule(settings, "exit_short"),
    }


def _captured_rule(settings: Mapping[str, object], prefix: str, *, include_position: bool = False) -> str:
    operator = _text_value(settings.get(f"{prefix}_operator", ""))
    value = _text_value(settings.get(f"{prefix}_value", ""))
    position = _text_value(settings.get(f"{prefix}_position", "")) if include_position else ""
    rule = " ".join(part for part in [operator, value] if part)
    return f"{rule} -> {position}" if position else rule


def _markdown(summary: dict[str, object], cost_frame: pd.DataFrame, trade_frame: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Wizard Local Verification",
            "",
            f"- Pair: `{summary['pair']}`",
            f"- Exact mode: `{summary['exact_mode']}`",
            f"- Mode fidelity: `{summary['mode_fidelity_status']}`",
            f"- Mode fidelity reason: `{summary['mode_fidelity_reason'] or 'none'}`",
            f"- Local observations: `{summary['local_observations']}`",
            f"- Acceptance: `{summary['acceptance']}`",
            f"- Reason: `{summary['acceptance_reason']}`",
            f"- Profit factor: `{float(summary['profit_factor']):.4f}`",
            f"- Sharpe: `{float(summary['sharpe']):.4f}`",
            f"- Max drawdown: `{float(summary['max_drawdown']):.2%}`",
            f"- Total return: `{float(summary['total_return']):.2%}`",
            "",
            "## Cost Comparison",
            "",
            cost_frame[["cost_case", "trades", "profit_factor", "sharpe", "max_drawdown", "total_return", "gross_return", "total_cost_drag"]].to_markdown(index=False),
            "",
            "## Trade Log Preview",
            "",
            trade_frame.head(20).to_markdown(index=False) if not trade_frame.empty else "No trades.",
            "",
        ]
    )


def _normalize_interval(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"1day", "daily", "day", "days"}:
        return "daily"
    return text


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
