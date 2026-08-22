from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

from math import isfinite
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.rl.brain_contract import (
    BRAIN_SCHEMA_VERSION,
    append_brain_memory,
    brain_output_paths,
    normalize_pair,
    write_cycle_summary,
    write_suggestion_frame,
)
from quant_platform.rl.rl_learning_agent import run_magicka_learning_cycle, run_sequential_thinking_magicka


DEFAULT_BRAIN_READINESS_THRESHOLD = 0.65
DEFAULT_BRAIN_READY_WEIGHT = 0.55
DEFAULT_BRAIN_CONFIDENCE_WEIGHT = 0.45
READINESS_SCORE_WARNING_LOW_CONFIDENCE = 0.5
READINESS_SCORE_WARNING_LOW_PROVIDER_QUALITY = 0.7


def _sorted_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    if not candidates:
        return candidates
    frame = pd.DataFrame(candidates)
    confidence = pd.to_numeric(frame.get("confidence", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    provider_quality = pd.to_numeric(frame.get("provider_quality_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    status_rank = frame.get("status", pd.Series(dtype=str)).astype(str).eq("ready").astype(int)
    frame["candidate_priority"] = status_rank * 2 + confidence * 2 + provider_quality * 0.8
    return frame.sort_values(by=["candidate_priority", "confidence", "provider_quality_score"], ascending=[False, False, False]).to_dict("records")


def run_brain_cycle(
    root: Path = ROOT,
    *,
    pair_id: str = "",
    policy_candidates: int = 12,
    max_recommendations: int = 12,
    readiness_threshold: float = DEFAULT_BRAIN_READINESS_THRESHOLD,
) -> CommandResult:
    """Run a shadow Brain cycle over magicka + sequential-thinking and emit review-ready candidates."""
    cycle_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pair_filter = pair_id.replace("/", "-").upper().strip() if pair_id else ""
    paths = brain_output_paths(root, cycle_id)

    learning_result = run_magicka_learning_cycle(root=root, pair_id=pair_filter, policy_candidates=policy_candidates)
    sequential_result = run_sequential_thinking_magicka(
        root=root,
        pair_id=pair_filter,
        max_recommendations=max_recommendations,
    )

    learning_summary = _read_csv(learning_result.paths.get("summary", Path()))
    learning_backtests = _read_csv(learning_result.paths.get("backtests", Path()))
    recommendations = _read_csv(sequential_result.paths.get("recommendations", Path()))
    if learning_summary.empty and "summary" in learning_result.paths:
        learning_summary = _read_csv(learning_result.paths["summary"])
    if learning_backtests.empty and "backtests" in learning_result.paths:
        learning_backtests = _read_csv(learning_result.paths["backtests"])
    if recommendations.empty and "recommendations" in sequential_result.paths:
        recommendations = _read_csv(sequential_result.paths["recommendations"])

    provider_context = _provider_context_from_frame(learning_summary, recommendations, learning_result, sequential_result)
    candidate_rows: list[dict[str, object]] = []
    if not learning_summary.empty:
        winner_flags = (
            pd.Series(False, index=learning_summary.index)
            if "winner" not in learning_summary.columns
            else learning_summary["winner"].fillna(0).astype(int).eq(1)
        )
        winner_rows = learning_summary.loc[winner_flags]
        winner = winner_rows.iloc[0] if not winner_rows.empty else learning_summary.sort_values("profit_factor", ascending=False).iloc[0]
        candidate_rows.append(_policy_row_from_learning(winner, pair_filter, provider_context=provider_context))
    if not recommendations.empty and {"pair_id", "proposed_change", "focus_area", "focus_area"}.issubset(set(recommendations.columns)):
        for idx, row in recommendations.head(max_recommendations).iterrows():
            candidate_rows.append(_policy_row_from_recommendation(row, idx, pair_filter, provider_context=provider_context))

    if not candidate_rows:
        candidate_rows.append(
            _blocked_candidate(
                cycle_id,
                pair_filter,
                learning_result.summary.get("status"),
                provider_context=provider_context,
            )
        )

    candidates = pd.DataFrame(_sorted_candidates(candidate_rows), columns=_candidate_columns())
    valid, reason = write_suggestion_candidates(candidates, paths["candidate_csv"])
    cycle_readiness = score_brain_cycle_readiness(candidates)
    final_gate = "pass" if float(cycle_readiness["readiness_score"]) >= float(readiness_threshold) else "hold"
    candidate_status = "ready" if reason == "written" else "blocked"
    final_status = "ready" if candidate_status == "ready" and final_gate == "pass" else "blocked"
    write_cycle_summary(
        paths["cycle_summary"],
        {
            "cycle_id": cycle_id,
            "pair_filter": pair_filter,
            "candidate_count": len(candidate_rows),
            "readiness_threshold": readiness_threshold,
            "candidate_status": candidate_status,
            "status": final_status,
            "blocker": "" if reason == "written" else reason,
            "learning_status": learning_result.summary.get("status", "unknown"),
            "sequential_status": sequential_result.summary.get("status", "unknown"),
            "created_at": now_utc_iso(),
        },
    )
    atomic_write_csv(candidates, paths["candidate_rollup"], index=False)
    writeable_rollup = _coerce_candidate_rollup(candidates)
    atomic_write_csv(writeable_rollup, paths["candidate_rollup"], index=False)
    readiness_gate = "pass" if float(cycle_readiness["readiness_score"]) >= float(readiness_threshold) else "hold"
    _append_readiness_trend(
        root=root,
        cycle_id=cycle_id,
        summary=cycle_readiness,
        readiness_threshold=readiness_threshold,
        status=final_status,
        readiness_gate=readiness_gate,
    )

    _append_brain_cycle_memory(
        paths["memory_jsonl"],
        cycle_id=cycle_id,
        pair_filter=pair_filter,
        status="passed" if final_status == "ready" else "blocked",
        blocker=reason if final_status != "ready" else "",
        recommendation_count=int(len(candidate_rows)),
    )
    return CommandResult(
        paths={
            "candidate_csv": paths["candidate_csv"],
            "candidate_rollup": paths["candidate_rollup"],
            "cycle_summary": paths["cycle_summary"],
            "memory_jsonl": paths["memory_jsonl"],
            "learning_summary": learning_result.paths.get("summary", Path()) if hasattr(learning_result, "paths") else Path(),
            "learning_backtests": learning_result.paths.get("backtests", Path()) if hasattr(learning_result, "paths") else Path(),
            "sequential_recommendations": sequential_result.paths.get("recommendations", Path()) if hasattr(sequential_result, "paths") else Path(),
        },
        summary={
            "cycle_id": cycle_id,
            "status": final_status,
            "candidate_count": int(len(candidate_rows)),
            "readiness_score": float(cycle_readiness["readiness_score"]),
            "readiness_gate": readiness_gate,
            "ready_count": int(cycle_readiness["ready_count"]),
            "blocked_count": int(cycle_readiness["blocked_count"]),
            "mean_confidence": float(cycle_readiness["mean_confidence"]),
            "readiness_warnings": cycle_readiness.get("readiness_warnings", ""),
            "blocker": reason if reason != "written" else "",
        },
    )


def build_brain_readiness_report(
    root: Path = ROOT,
    *,
    candidate_rollup_path: Path | None = None,
    ready_weight: float = DEFAULT_BRAIN_READY_WEIGHT,
    confidence_weight: float = DEFAULT_BRAIN_CONFIDENCE_WEIGHT,
    score_threshold: float = DEFAULT_BRAIN_READINESS_THRESHOLD,
) -> CommandResult:
    """Compute a lightweight readiness score over the latest brain candidate rollup."""
    rollup_path = candidate_rollup_path or (root / "reports" / "brain" / "brain_candidate_rollup.csv")
    frame = _read_csv(rollup_path)
    cycle_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not frame.empty:
        latest_cycle = str(frame["cycle_id"].iloc[-1]) if "cycle_id" in frame.columns else cycle_id
        frame = frame[frame["cycle_id"] == latest_cycle]
    score_payload = score_brain_cycle_readiness(
        frame,
        ready_weight=ready_weight,
        confidence_weight=confidence_weight,
    )
    score_payload.update(
        {
            "score_threshold": score_threshold,
            "cycle_id": str(score_payload["cycle_id"] or cycle_id),
            "pair_filter": str(score_payload["pair"]) or "",
            "score_gate": "pass" if score_payload["readiness_score"] >= score_threshold else "hold",
        }
    )
    score_payload.update(_provider_rollup_from_frame(frame))
    if score_payload["score_gate"] == "pass":
        score_payload["readiness_alert_level"] = "green"
    elif score_payload["readiness_score"] >= score_threshold * 0.9:
        score_payload["readiness_alert_level"] = "yellow"
    else:
        score_payload["readiness_alert_level"] = "red"
    report_dir = root / "reports" / "brain"
    report_path = report_dir / f"brain_readiness_report_{cycle_id}.csv"
    latest_report_path = report_dir / "brain_readiness_report.csv"
    summary_path = root / "reports" / "brain" / f"brain_readiness_summary_{cycle_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    score_payload["candidate_rows"] = int(len(frame))
    report_frame = pd.DataFrame([score_payload])
    atomic_write_csv(report_frame, report_path, index=False)
    atomic_write_csv(report_frame, latest_report_path, index=False)
    write_cycle_summary(
        summary_path,
        {
            **{k: str(v) for k, v in score_payload.items() if isinstance(v, (str, int, float, bool))},
            "written_at": now_utc_iso(),
        },
    )
    _append_readiness_trend(
        root=root,
        cycle_id=cycle_id,
        summary=score_payload,
        readiness_threshold=score_threshold,
        status="pass" if score_payload["score_gate"] == "pass" else "blocked",
        readiness_gate=score_payload["score_gate"],
    )
    return CommandResult(
        paths={
            "brain_readiness_report": report_path,
            "brain_readiness_summary": summary_path,
            "brain_candidate_rollup": rollup_path,
        },
        summary=score_payload,
    )


def score_brain_cycle_readiness(
    frame: pd.DataFrame,
    *,
    ready_weight: float = DEFAULT_BRAIN_READY_WEIGHT,
    confidence_weight: float = DEFAULT_BRAIN_CONFIDENCE_WEIGHT,
) -> dict[str, object]:
    """Score a candidate frame on readiness mix and confidence."""
    if frame.empty:
        return {
            "cycle_id": "",
            "pair": "",
            "candidate_count": 0,
            "ready_count": 0,
            "blocked_count": 0,
            "ready_ratio": 0.0,
            "mean_confidence": 0.0,
            "readiness_score": 0.0,
            "readiness_gate": "hold",
            "readiness_warnings": "no_candidates",
        }
    candidate_count = len(frame)
    ready = (frame["status"] == "ready").sum()
    blocked = candidate_count - ready
    mean_confidence = float(pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.0).mean())
    ready_ratio = ready / candidate_count if candidate_count else 0.0
    readiness_score = max(
        0.0, min(1.0, (ready_ratio * ready_weight) + (mean_confidence * confidence_weight))
    )
    pair = str(frame["pair"].iloc[0]) if not frame.empty and "pair" in frame.columns else ""
    provider_payload = _provider_rollup_from_frame(frame)
    warnings: list[str] = []
    provider_quality_score = _safe_numeric(provider_payload.get("provider_quality_score", 0.0))
    if provider_quality_score < READINESS_SCORE_WARNING_LOW_PROVIDER_QUALITY:
        warnings.append("low_provider_quality")
    if float(mean_confidence) <= READINESS_SCORE_WARNING_LOW_CONFIDENCE:
        warnings.append("low_mean_confidence")
    has_provider_trace = any(
        field in frame.columns
        for field in (
            "provider_rows",
            "provider",
            "provider_name",
            "provider_source",
            "provider_quality_score",
            "market_data_provider",
            "venue",
        )
    )
    if has_provider_trace and provider_payload.get("provider_rows", 0) <= 0:
        warnings.append("missing_provider_rows")
    if not _ready_status_diversity(frame):
        warnings.append("single_status_profile")

    if not warnings:
        warnings.append("none")

    return {
        "cycle_id": str(frame["cycle_id"].iloc[0]) if "cycle_id" in frame.columns else "",
        "pair": normalize_pair(pair),
        "candidate_count": int(candidate_count),
        "ready_count": int(ready),
        "blocked_count": int(blocked),
        "ready_ratio": round(ready_ratio, 4),
        "mean_confidence": round(mean_confidence, 4),
        "readiness_score": round(readiness_score, 4),
        "readiness_gate": "pass" if readiness_score >= (ready_weight + confidence_weight) / 2.0 else "hold",
        "readiness_warnings": ";".join(warnings),
        **provider_payload,
    }


def _policy_row_from_learning(row: pd.Series, pair_filter: str, *, provider_context: dict[str, object] | None = None) -> dict[str, object]:
    pair = normalize_pair(row.get("pair", pair_filter or ""))
    if not pair:
        pair = normalize_pair(pair_filter) if pair_filter else "UNKNOWN"
    entry_threshold = row.get("entry_threshold", "")
    hold_cap_pct = float(row.get("hold_cap_pct", 0.0) or 0.0)
    status = str(row.get("status", "ready"))
    confidence = _confidence_from_profit_factor(row.get("profit_factor", 0.0))
    return {
        "cycle_id": row.get("cycle_id", ""),
        "pair": pair,
        "strategy_name": "brain_candidate",
        "variant": str(row.get("policy_name", "learning_policy")),
        "entry_logic": f"entry_abs_zscore >= {entry_threshold}",
        "exit_logic": f"hold_cap_pct={hold_cap_pct:.3f};max_drawdown_pressure_guard",
        "entry_threshold": entry_threshold if pd.notna(entry_threshold) else 0.0,
        "hold_bars_min": 1,
        "hold_bars_max": max(1, int(100 * hold_cap_pct)) if hold_cap_pct > 0 else 12,
        "confidence": confidence,
        "expected_return_delta": _safe_numeric(row.get("total_return", 0.0)),
        "expected_drawdown_delta": -abs(_safe_numeric(row.get("max_drawdown", 0.0))),
        "evidence_path": "reports/rl/rl_learning_cycle_summary.csv",
        "blocker": "",
        "status": "ready" if str(status).lower() == "ready" else "blocked",
        "created_at": now_utc_iso(),
        "schema_version": BRAIN_SCHEMA_VERSION,
        **_provider_row_fields(row, provider_context),
    }


def _policy_row_from_recommendation(
    row: pd.Series,
    idx: int,
    pair_filter: str,
    *,
    provider_context: dict[str, object] | None = None,
) -> dict[str, object]:
    pair = normalize_pair(row.get("pair_id", row.get("pair", pair_filter or "")))
    if not pair:
        pair = normalize_pair(pair_filter) if pair_filter else "UNKNOWN"
    return {
        "cycle_id": row.get("cycle_id", ""),
        "pair": pair,
        "strategy_name": "brain_recommendation",
        "variant": f"rec_{idx + 1:03d}",
        "entry_logic": str(row.get("proposed_change", "")),
        "exit_logic": str(row.get("expected_effect", "")),
        "entry_threshold": 0.0,
        "hold_bars_min": 1,
        "hold_bars_max": 12,
        "confidence": 0.55,
        "expected_return_delta": 0.0,
        "expected_drawdown_delta": 0.0,
        "evidence_path": "reports/rl/sequential_thinking_magicka_recommendations.csv",
        "blocker": "",
        "status": "ready" if str(row.get("priority", "")).lower() in {"high", "medium", "low"} else "blocked",
        "created_at": now_utc_iso(),
        "schema_version": BRAIN_SCHEMA_VERSION,
        **_provider_row_fields(row, provider_context),
    }


def _blocked_candidate(
    cycle_id: str,
    pair_filter: str,
    blocker: object | None,
    *,
    provider_context: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "cycle_id": cycle_id,
        "pair": normalize_pair(pair_filter) if pair_filter else "UNKNOWN",
        "strategy_name": "brain_blocked",
        "variant": "no_candidate",
        "entry_logic": "defer",
        "exit_logic": "defer",
        "entry_threshold": 0.0,
        "hold_bars_min": 0,
        "hold_bars_max": 0,
        "confidence": 0.0,
        "expected_return_delta": 0.0,
        "expected_drawdown_delta": 0.0,
        "evidence_path": "",
        "blocker": str(blocker or "no_candidates"),
        "status": "blocked",
        "created_at": now_utc_iso(),
        "schema_version": BRAIN_SCHEMA_VERSION,
        **_provider_row_fields({}, provider_context),
    }


def write_suggestion_candidates(path_like: pd.DataFrame | list[dict[str, object]], path: Path) -> tuple[bool, str]:
    frame = pd.DataFrame(path_like)
    frame = frame[_candidate_columns()]
    valid, reason = _validate_candidate_frame(frame)
    if not valid:
        return False, reason
    write_suggestion_frame(frame, path)
    return True, "written"


def _coerce_candidate_rollup(frame: pd.DataFrame) -> pd.DataFrame:
    rollup = frame.copy()
    rollup["ready_count"] = int((rollup["status"] == "ready").sum())
    rollup["blocked_count"] = int((rollup["status"] != "ready").sum())
    rollup["generated_at"] = now_utc_iso()
    return rollup


def _append_readiness_trend(
    *,
    root: Path,
    cycle_id: str,
    summary: dict[str, object],
    readiness_threshold: float,
    status: str,
    readiness_gate: str,
) -> None:
    trend_path = root / "reports" / "brain" / "paper_readiness_trend.csv"
    payload = {
        "cycle_id": cycle_id,
        "pair": str(summary.get("pair", "")),
        "candidate_count": int(summary.get("candidate_count", 0)),
        "ready_count": int(summary.get("ready_count", 0)),
        "blocked_count": int(summary.get("blocked_count", 0)),
        "readiness_score": float(summary.get("readiness_score", 0.0)),
        "readiness_gate": str(readiness_gate),
        "status": status,
        "readiness_threshold": float(readiness_threshold),
        "provider": str(summary.get("provider", "")),
        "provider_source": str(summary.get("provider_source", "")),
        "provider_rows": int(summary.get("provider_rows", 0)),
        "provider_quality_score": float(summary.get("provider_quality_score", 0.0)),
        "readiness_warnings": str(summary.get("readiness_warnings", "")),
        "created_at": now_utc_iso(),
    }
    trend_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame()
    if trend_path.exists():
        try:
            existing = pd.read_csv(trend_path)
        except Exception:
            existing = pd.DataFrame()
    atomic_write_csv(pd.concat([existing, pd.DataFrame([payload])], ignore_index=True), trend_path, index=False)


def _ready_status_diversity(frame: pd.DataFrame) -> bool:
    if "status" not in frame.columns:
        return False
    unique_statuses = set(str(value).lower() for value in frame["status"].dropna().tolist())
    return len(unique_statuses) > 1


def _candidate_columns() -> list[str]:
    return [
        "cycle_id",
        "pair",
        "strategy_name",
        "variant",
        "entry_logic",
        "exit_logic",
        "entry_threshold",
        "hold_bars_min",
        "hold_bars_max",
        "confidence",
        "expected_return_delta",
        "expected_drawdown_delta",
        "evidence_path",
        "blocker",
        "status",
        "created_at",
        "schema_version",
        "provider",
        "provider_source",
        "provider_rows",
        "provider_quality_score",
    ]


def _first_non_empty(*candidates: object) -> str:
    for value in candidates:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def _first_non_empty_numeric(value: object, fallback: float = 0.0) -> float:
    try:
        if value is None:
            return fallback
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return fallback
        return float(text)
    except (TypeError, ValueError):
        return fallback


def _provider_row_fields(row: pd.Series | dict[str, object], fallback: dict[str, object] | None) -> dict[str, object]:
    source_row = row if isinstance(row, pd.Series) else pd.Series(row)
    return {
        "provider": _first_non_empty(
            source_row.get("provider"),
            source_row.get("provider_name"),
            source_row.get("market_data_provider"),
            source_row.get("venue"),
            fallback.get("provider") if fallback else "",
        ),
        "provider_source": _first_non_empty(
            source_row.get("provider_source"),
            source_row.get("source"),
            source_row.get("source_system"),
            source_row.get("source_group"),
            fallback.get("provider_source") if fallback else "",
        ),
        "provider_rows": _first_non_empty_numeric(
            source_row.get("provider_rows"),
            _first_non_empty_numeric(fallback.get("provider_rows") if fallback else 0, 0),
        ),
        "provider_quality_score": _first_non_empty_numeric(
            source_row.get("provider_quality_score"),
            _first_non_empty_numeric(fallback.get("provider_quality_score") if fallback else 0.0, 0.0),
        ),
    }


def _provider_context_from_frame(
    learning_summary: pd.DataFrame,
    recommendations: pd.DataFrame,
    *_unused: CommandResult,
) -> dict[str, object]:
    frame = pd.DataFrame()
    if not learning_summary.empty:
        frame = learning_summary
    elif not recommendations.empty:
        frame = recommendations
    if frame.empty:
        return {}
    provider_value = _first_non_empty(
        frame.get("provider", pd.Series(dtype=object)).iloc[0] if "provider" in frame.columns else "",
        frame.get("provider_name", pd.Series(dtype=object)).iloc[0] if "provider_name" in frame.columns else "",
        frame.get("market_data_provider", pd.Series(dtype=object)).iloc[0] if "market_data_provider" in frame.columns else "",
        frame.get("venue", pd.Series(dtype=object)).iloc[0] if "venue" in frame.columns else "",
    )
    provider_source = _first_non_empty(
        frame.get("provider_source", pd.Series(dtype=object)).iloc[0] if "provider_source" in frame.columns else "",
        frame.get("source", pd.Series(dtype=object)).iloc[0] if "source" in frame.columns else "",
        frame.get("source_system", pd.Series(dtype=object)).iloc[0] if "source_system" in frame.columns else "",
        frame.get("source_group", pd.Series(dtype=object)).iloc[0] if "source_group" in frame.columns else "",
    )
    return {
        "provider": provider_value,
        "provider_source": provider_source,
        "provider_rows": _first_non_empty_numeric(
            frame.get("provider_rows", pd.Series(dtype=object)).sum() if "provider_rows" in frame.columns else 0,
            len(frame),
        ),
        "provider_quality_score": _first_non_empty_numeric(
            frame.get("provider_quality_score", pd.Series(dtype=object)).mean()
            if "provider_quality_score" in frame.columns
            else 0.0
        ),
    }


def _provider_rollup_from_frame(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "provider": "",
            "provider_source": "",
            "provider_rows": 0,
            "provider_quality_score": 0.0,
            "provider_mix": "",
        }
    providers = [
        str(value).strip() for value in frame.get("provider", pd.Series(dtype=object)).tolist() if str(value).strip()
    ]
    provider_sources = [
        str(value).strip() for value in frame.get("provider_source", pd.Series(dtype=object)).tolist() if str(value).strip()
    ]
    providers = sorted(set(providers))
    provider_sources = sorted(set(provider_sources))
    return {
        "provider": ";".join(providers),
        "provider_source": ";".join(provider_sources),
        "provider_rows": int(_safe_int_or_zero(frame.get("provider_rows", pd.Series(dtype=object)).sum(), int(len(frame)))),
        "provider_quality_score": round(
            float(
                pd.to_numeric(frame.get("provider_quality_score", pd.Series(dtype=object)), errors="coerce")
                .fillna(0.0)
                .mean()
            ),
            4,
        ),
        "provider_mix": "mixed" if len(providers) > 1 or len(provider_sources) > 1 else (providers[0] if providers else (provider_sources[0] if provider_sources else "")),
    }


def _safe_int_or_zero(value: object, fallback: int) -> int:
    try:
        if value is None:
            return fallback
        if isinstance(value, int):
            return value
        parsed = int(float(value))
        return parsed if parsed >= 0 else fallback
    except (TypeError, ValueError):
        return fallback


def _validate_candidate_frame(frame: pd.DataFrame) -> tuple[bool, str]:
    if frame.empty:
        return False, "no_brain_candidates"
    for _, row in frame.iterrows():
        if str(row.get("pair", "")) in {"", "UNKNOWN"}:
            return False, "missing_pair"
    return True, "valid"


def _append_brain_cycle_memory(
    memory_path: Path,
    *,
    cycle_id: str,
    pair_filter: str,
    status: str,
    blocker: str,
    recommendation_count: int,
) -> None:
    append_brain_memory(
        memory_path,
        {
            "agent": "brain_entity",
            "task_id": f"brain_cycle:{cycle_id}",
            "task_type": "run_brain_cycle",
            "cycle_id": cycle_id,
            "pair": pair_filter,
            "outcome_known": True,
            "outcome_label": status,
            "blocker": blocker,
            "next_step": f"review {recommendation_count} suggestions",
        },
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_numeric(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not isfinite(number):
        return 0.0
    return number


def _confidence_from_profit_factor(value: object) -> float:
    pf = _safe_numeric(value)
    if pf <= 0:
        return 0.0
    return max(0.0, min(1.0, (pf - 1.0) / 3.0))


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
