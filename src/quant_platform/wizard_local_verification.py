from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.backtest import BacktestResult, CostModel, backtest_two_leg_spread


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
) -> CommandResult:
    history_file = history_path or DEFAULT_HISTORY
    wizard_file = wizard_capture_path or DEFAULT_WIZARD_CAPTURE
    payload = _read_json(history_file)
    history = _history_frame(payload)
    signal, trade_log = static_spread_signal(history["zscore"], entry_threshold=entry_threshold, exit_threshold=exit_threshold)
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

    wizard_payload = _read_json(wizard_file) if wizard_file.exists() else {}
    mode = exact_mode or _mode_from_payload(wizard_payload, payload)
    local_last = pd.to_datetime(history["timestamp"].iloc[-1], utc=True)
    as_of = pd.Timestamp(current_date, tz="UTC")
    age_days = max(0, int((as_of.normalize() - local_last.normalize()).days))
    acceptance, reason = _acceptance(base_result, len(history), age_days, int((trade_frame.get("exit_reason", pd.Series(dtype=str)) != "open_at_end_of_history").sum()))
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
    if queue_file.exists():
        queue = pd.read_csv(queue_file).head(max_pairs)
        for _, row in queue.iterrows():
            candidate = _candidate_from_queue_row(row, root=root)
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
            "wizard_sharpe": "",
            "wizard_returns_total": "",
            "wizard_returns_total_pct": "",
            "exact_mode": "Static (Spread)",
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


def _candidate_from_queue_row(row: pd.Series, *, root: Path) -> dict[str, object]:
    asset_x = row.get("asset_x", "")
    asset_y = row.get("asset_y", "")
    interval = row.get("interval", "daily")
    if pd.isna(interval) or str(interval).strip() == "":
        interval = "daily"
    source_path = _path_from_value(row.get("pair_history_path", row.get("source_path", "")), root=root)
    if source_path is None or not source_path.exists():
        source_path = _find_local_history_path(root=root, asset_x=str(asset_x), asset_y=str(asset_y), interval=str(interval))
    source_payload = _read_json(source_path) if source_path and source_path.exists() else {}
    exact_mode = source_payload.get("exact_mode", "") or _mode_from_strategy(row.get("strategy", row.get("strategy_family_note", source_payload.get("strategy_mode", ""))))
    return {
        "pair": row.get("pair", ""),
        "asset_x": asset_x or source_payload.get("asset_x", ""),
        "asset_y": asset_y or source_payload.get("asset_y", ""),
        "interval": interval or source_payload.get("interval", ""),
        "wizard_sharpe": row.get("sharpe", source_payload.get("sharpe", "")),
        "wizard_returns_total": row.get("returns_total", row.get("return_pct", source_payload.get("returns_total", ""))),
        "wizard_returns_total_pct": row.get("returns_total_pct", row.get("return_pct", "")),
        "exact_mode": exact_mode,
        "spread_id": source_payload.get("spread_id", ""),
        "strategy_id": source_payload.get("strategy_id", ""),
        "history_path": source_path,
        "wizard_capture_path": source_path,
        "source_row_path": row.get("source_path", ""),
        "candidate_source": row.get("source_group", "wizard_queue"),
        "execution_bucket": row.get("execution_bucket", ""),
        "execution_blockers": row.get("execution_blockers", ""),
        "research_blockers_only": row.get("research_blockers_only", ""),
        "recommended_action": row.get("recommended_action", ""),
    }


def _verify_candidate(*, root: Path, candidate: dict[str, object], current_date: str) -> dict[str, object]:
    blockers = _candidate_blockers(candidate)
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
        )
        summary = pd.read_csv(result.paths["summary"]).iloc[0].to_dict()
        return {
            **base,
            "verification_status": "verified",
            "verification_blocker": "",
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


def _candidate_blockers(candidate: dict[str, object]) -> list[str]:
    blockers: list[str] = []
    history_path = candidate.get("history_path")
    wizard_path = candidate.get("wizard_capture_path")
    exact_mode = str(candidate.get("exact_mode", "")).strip().lower()
    if not exact_mode:
        blockers.append("missing_exact_mode_capture")
    elif exact_mode not in {"static (spread)", "static spread", "ou (spread)", "ou spread"}:
        blockers.append(f"unsupported_exact_mode:{candidate.get('exact_mode')}")
    if not history_path:
        blockers.append("missing_local_history_path")
    elif not Path(history_path).exists():
        blockers.append("local_history_file_missing")
    else:
        payload = _read_json(Path(history_path))
        history = pd.DataFrame(payload.get("history", []))
        required = {"timestamp", "price_x", "price_y", "zscore"}
        missing = sorted(required - set(history.columns))
        if missing:
            blockers.append(f"local_history_missing_columns:{','.join(missing)}")
    if not wizard_path or not Path(wizard_path).exists():
        blockers.append("missing_wizard_capture_path")
    return blockers


def _candidate_output_name(candidate: dict[str, object]) -> str:
    pair = str(candidate.get("pair", "candidate")).replace("/", "_").replace("-", "").lower()
    mode = str(candidate.get("exact_mode", "mode")).replace("(", "").replace(")", "").replace(" ", "_").lower()
    return f"{pair}_{mode}_verification"


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
    left = asset_x.replace("-USD", "").replace("-", "_").lower()
    right = asset_y.replace("-USD", "").replace("-", "_").lower()
    candidates = sorted((root / "data" / "raw" / "pair_details").glob(f"*{left}*{right}*.json"))
    candidates.extend(sorted((root / "data" / "raw" / "pair_details").glob(f"*{right}*{left}*.json")))
    daily = str(interval).strip().lower() in {"daily", "1day", "day"}
    matches: list[tuple[int, pd.Timestamp, Path]] = []
    for path in candidates:
        payload = _read_json(path)
        history = pd.DataFrame(payload.get("history", []))
        if {"timestamp", "price_x", "price_y", "zscore"}.issubset(history.columns):
            text = path.name.lower()
            if not daily or "1day" in text or "daily" in text:
                last = pd.to_datetime(history["timestamp"], utc=True, errors="coerce").max()
                matches.append((len(history), last if pd.notna(last) else pd.Timestamp.min.tz_localize("UTC"), path))
    if matches:
        return sorted(matches, key=lambda item: (item[0], item[1]), reverse=True)[0][2]
    if daily:
        return None
    fallback_matches: list[tuple[int, pd.Timestamp, Path]] = []
    for path in candidates:
        payload = _read_json(path)
        history = pd.DataFrame(payload.get("history", []))
        if {"timestamp", "price_x", "price_y", "zscore"}.issubset(history.columns):
            last = pd.to_datetime(history["timestamp"], utc=True, errors="coerce").max()
            fallback_matches.append((len(history), last if pd.notna(last) else pd.Timestamp.min.tz_localize("UTC"), path))
    return sorted(fallback_matches, key=lambda item: (item[0], item[1]), reverse=True)[0][2] if fallback_matches else None


def _mode_from_strategy(value: object) -> str:
    text = str(value).strip().lower().replace("_", "").replace(" ", "")
    if text in {"ouspread", "ou"}:
        return "OU (Spread)"
    if text in {"ouzscorer", "ouzscore", "ouzscoreroll"}:
        return "OU (ZScoreR)"
    if text in {"staticspread", "static"}:
        return "Static (Spread)"
    if text in {"staticzscorer", "staticzscore", "staticzscoreroll"}:
        return "Static (ZScoreR)"
    return ""


def _mode_from_payload(wizard_payload: dict[str, object], local_payload: dict[str, object]) -> str:
    explicit = str(wizard_payload.get("exact_mode", "") or local_payload.get("exact_mode", "")).strip()
    if explicit:
        return explicit
    return _mode_from_strategy(wizard_payload.get("strategy_mode", local_payload.get("strategy_mode", ""))) or "Static (Spread)"


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
    verified = int((frame["verification_status"] == "verified").sum())
    accepted = int((frame["acceptance"] == "ACCEPT").sum())
    lines.extend(
        [
            f"- Verified locally: `{verified}`",
            f"- Accepted: `{accepted}`",
            "",
            "## Ranked Board",
            "",
            frame[
                [
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
                ]
            ].to_markdown(index=False),
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
    beta = pd.to_numeric(data.get("beta", 1.0), errors="coerce").fillna(1.0).replace(0, 1.0).abs()
    signal_position = data["signal"].shift(1).fillna(0.0)
    gross_scale = 1.0 + hedge_ratio.abs() * beta
    weight_y = signal_position / gross_scale
    weight_x = -signal_position * hedge_ratio * beta / gross_scale
    gross_return = weight_x * returns_x + weight_y * returns_y
    target_weight_y = data["signal"] / gross_scale
    target_weight_x = -data["signal"] * hedge_ratio * beta / gross_scale
    turnover = target_weight_x.diff().abs().fillna(target_weight_x.abs()) + target_weight_y.diff().abs().fillna(target_weight_y.abs())
    funding_x = pd.to_numeric(data.get("funding_x_bps", cost_model.funding_bps_per_day), errors="coerce").fillna(cost_model.funding_bps_per_day)
    funding_y = pd.to_numeric(data.get("funding_y_bps", cost_model.funding_bps_per_day), errors="coerce").fillna(cost_model.funding_bps_per_day)
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


def _acceptance(base_result: dict[str, object], rows: int, age_days: int, closed_trades: int) -> tuple[str, str]:
    blockers = []
    if rows < 320:
        blockers.append("local_history_rows<320")
    if float(base_result.get("total_return", 0.0)) <= 0:
        blockers.append("total_return<=0")
    if float(base_result.get("sharpe", 0.0)) < 1.2:
        blockers.append("sharpe<1.2")
    pf = float(base_result.get("profit_factor", 0.0))
    if not np.isinf(pf) and pf < 1.8:
        blockers.append("profit_factor<1.8")
    if float(base_result.get("max_drawdown", 0.0)) > 0.15:
        blockers.append("max_drawdown>15pct")
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
        "signal_source": f"local zscore column used as {exact_mode} sigma proxy",
        "entry_long_x": f">= {entry_threshold:.2f}",
        "entry_short_x": f"<= {-entry_threshold:.2f}",
        "exit_long_x": f"<= {exit_threshold:.2f}",
        "exit_short_x": f">= {exit_threshold:.2f}",
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
        "wizard_evidence_path": _rel(wizard_path),
        "local_evidence_path": _rel(history_path),
        "local_last_timestamp": local_last.isoformat(),
        "current_date_assumed": current_date,
        "data_age_days": age_days,
        "stale_reason": "" if age_days <= 2 else f"local_daily_history_ends_{age_days}_days_before_current_date",
    }


def _markdown(summary: dict[str, object], cost_frame: pd.DataFrame, trade_frame: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Wizard Local Verification",
            "",
            f"- Pair: `{summary['pair']}`",
            f"- Exact mode: `{summary['exact_mode']}`",
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
