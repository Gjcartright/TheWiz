"""Bounded Crypto Wizards pair-detail API schema pilot.

The documented API is not assumed to equal the authenticated dashboard. A
pilot inventories the returned fields, preserves exact request lineage, and
keeps all results discovery-only until dashboard and local parity are proven.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_platform.active_pipeline import CommandResult
from quant_platform.api_extraction import CryptoWizardsExtractor
from quant_platform.crypto_wizards_catalog import BASE_URL
from quant_platform.crypto_wizards_history import fetch_credits_used
from quant_platform.crypto_wizards_sweep import parse_wizard_credit_usage


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "wizard_pair_detail_api_pilot.v2"
PAIR_GROUP_DEFAULT = "binance|daily|ETH|WIF"
EndpointFetcher = Callable[..., Any]
CreditsFetcher = Callable[..., Any]
BACKTEST_CONTRACT_SOURCE = (
    "https://api.cryptowizards.net/docsv1beta/backtest-get.mdx/"
)
BACKTEST_LEVELS = {
    "Spread": (2.0, 0.0),
    "ZScoreRoll": (1.5, 0.0),
    "Copula": (0.05, 0.50),
}

ENDPOINTS = (
    ("spread", "/v1beta/spread", 5),
    ("zscores", "/v1beta/zscores", 5),
    ("backtest", "/v1beta/backtest", 6),
    ("cointegration", "/v1beta/cointegration", 5),
    ("copula", "/v1beta/copula", 5),
    ("correlations", "/v1beta/correlations", 5),
)
PLANNED_CREDITS = sum(cost for _, _, cost in ENDPOINTS)


def run_wizard_pair_detail_api_pilot(
    *,
    root: Path = ROOT,
    pair_group_key: str = PAIR_GROUP_DEFAULT,
    execute: bool = False,
    api_key: str | None = None,
    base_url: str = BASE_URL,
    daily_credit_limit: int = 1000,
    reserved_credits: int = 100,
    timeout: float = 30.0,
    now: datetime | None = None,
    endpoint_names: tuple[str, ...] | None = None,
    credits_fetcher: CreditsFetcher | None = None,
    endpoint_fetcher: EndpointFetcher | None = None,
) -> CommandResult:
    """Plan or execute a credit-bounded API schema probe."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    queue_path = active / "exhaustive_wizard_api_refresh_pair_detail_queue.csv"
    accounting_path = active / "exhaustive_wizard_api_refresh_source_accounting.csv"
    refresh_manifest_path = active / "exhaustive_wizard_api_refresh_manifest.json"
    queue = _read_csv_required(queue_path)
    accounting = _read_csv_required(accounting_path)
    refresh_manifest = _read_json_required(refresh_manifest_path)
    selected_queue = queue[queue["pair_group_key"].astype(str).eq(pair_group_key)]
    selected_sources = accounting[accounting["pair_group_key"].astype(str).eq(pair_group_key)]
    if selected_queue.empty or selected_sources.empty:
        raise ValueError(f"Pair group is not in the current API refresh: {pair_group_key}")
    source = _select_source_variant(selected_sources)
    request_config = _request_config(source)
    config_hash = sha256(_canonical_json(request_config).encode()).hexdigest()
    refresh_id = _text(refresh_manifest.get("refresh_id"))
    pilot_id = "wapipilot_" + sha256(
        f"{refresh_id}|{config_hash}".encode()
    ).hexdigest()[:20]
    attempt_id = (
        "wapipilotattempt_"
        + as_of.strftime("%Y%m%dT%H%M%S%fZ")
        + "_"
        + config_hash[:8]
    )
    selected_endpoints = _select_endpoints(endpoint_names)
    selected_endpoint_names = tuple(item[0] for item in selected_endpoints)
    planned_credits = sum(item[2] for item in selected_endpoints)
    raw_dir = (
        root
        / "data"
        / "raw"
        / "crypto_wizards"
        / "pair_detail_api"
        / as_of.date().isoformat()
        / pilot_id
        / attempt_id
    )
    raw_dir.mkdir(parents=True, exist_ok=True)

    credits_before: int | None = None
    credits_remaining_before: int | None = None
    credit_blocker = "execution_not_requested"
    if execute:
        if not api_key:
            credit_blocker = "missing_crypto_wizards_api_key"
        else:
            payload = (credits_fetcher or fetch_credits_used)(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
            )
            usage = parse_wizard_credit_usage(payload, configured_limit=daily_credit_limit)
            credits_before = usage.used
            credits_remaining_before = usage.remaining
            if not usage.known:
                credit_blocker = "credit_usage_unknown"
            elif planned_credits > max((usage.remaining or 0) - reserved_credits, 0):
                credit_blocker = "insufficient_credits_after_reserve"
            else:
                credit_blocker = ""

    manifest_rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []
    attempted_credits = 0
    completed_credits = 0
    for endpoint_name, endpoint_path, credit_cost in selected_endpoints:
        params = _endpoint_params(endpoint_name, request_config)
        status = "PLANNED" if not execute else f"BLOCKED_{credit_blocker.upper()}"
        error = "" if not execute else credit_blocker
        response_hash = ""
        evidence_path = ""
        observed_field_count = 0
        requested_at = ""
        completed_at = ""
        if execute and not credit_blocker:
            requested_at = datetime.now(timezone.utc).isoformat()
            attempted_credits += credit_cost
            try:
                payload = (endpoint_fetcher or _fetch_endpoint)(
                    endpoint_path=endpoint_path,
                    params=params,
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                )
                response_hash = sha256(_canonical_json(payload).encode()).hexdigest()
                raw_path = raw_dir / f"{endpoint_name}.json"
                envelope = {
                    "capture_metadata": {
                        "schema_version": SCHEMA_VERSION,
                        "pilot_id": pilot_id,
                        "attempt_id": attempt_id,
                        "endpoint": endpoint_name,
                        "captured_at": requested_at,
                        "credit_cost": credit_cost,
                        "request_status": "COMPLETED",
                        "response_sha256": response_hash,
                        "promotion_authority": False,
                        "live_trading_authorized": False,
                    },
                    "request": params,
                    "response": payload,
                }
                raw_path.write_text(
                    json.dumps(envelope, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                evidence_path = _relative(raw_path, root)
                discovered = CryptoWizardsExtractor.discover_fields(payload)
                observed_field_count = len(discovered)
                for field in discovered:
                    field_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "pilot_id": pilot_id,
                            "attempt_id": attempt_id,
                            "pair_group_key": pair_group_key,
                            "endpoint": endpoint_name,
                            "field": field.get("field", ""),
                            "type": field.get("type", ""),
                            "example": field.get("example", ""),
                            "evidence_path": evidence_path,
                            "promotion_authority": False,
                            "live_trading_authorized": False,
                        }
                    )
                completed_at = datetime.now(timezone.utc).isoformat()
                completed_credits += credit_cost
                status = "COMPLETED"
                error = ""
            except (requests.RequestException, ValueError, TypeError, json.JSONDecodeError) as exc:
                completed_at = datetime.now(timezone.utc).isoformat()
                status = "FAILED"
                error = _safe_error(exc)
                failure = _failure_payload(exc, api_key=api_key)
                response_hash = sha256(_canonical_json(failure).encode()).hexdigest()
                raw_path = raw_dir / f"{endpoint_name}.failure.json"
                raw_path.write_text(
                    json.dumps(
                        {
                            "capture_metadata": {
                                "schema_version": SCHEMA_VERSION,
                                "pilot_id": pilot_id,
                                "attempt_id": attempt_id,
                                "endpoint": endpoint_name,
                                "captured_at": completed_at,
                                "credit_cost": credit_cost,
                                "request_status": "FAILED",
                                "response_sha256": response_hash,
                                "promotion_authority": False,
                                "live_trading_authorized": False,
                            },
                            "request": params,
                            "failure": failure,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                evidence_path = _relative(raw_path, root)
        manifest_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "pilot_id": pilot_id,
                "attempt_id": attempt_id,
                "refresh_id": refresh_id,
                "pair_group_key": pair_group_key,
                "endpoint": endpoint_name,
                "endpoint_path": endpoint_path,
                "credit_cost": credit_cost,
                "status": status,
                "requested_at": requested_at,
                "completed_at": completed_at,
                "request_config_hash": config_hash,
                "request_params": _canonical_json(params),
                "response_sha256": response_hash,
                "observed_field_count": observed_field_count,
                "evidence_path": evidence_path,
                "error": error,
                "credit_charge_status": "NOT_ATTEMPTED",
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )

    credits_after: int | None = None
    if execute and api_key and not credit_blocker:
        after_payload = (credits_fetcher or fetch_credits_used)(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        credits_after = parse_wizard_credit_usage(
            after_payload,
            configured_limit=daily_credit_limit,
        ).used
    observed_credit_delta = (
        credits_after - credits_before
        if credits_after is not None and credits_before is not None
        else None
    )
    attempt_manifest = pd.DataFrame(manifest_rows)
    attempted_credit_accounting_matches = bool(
        observed_credit_delta is not None
        and observed_credit_delta == attempted_credits
    )
    if not attempt_manifest.empty:
        completed_mask = attempt_manifest["status"].eq("COMPLETED")
        failed_mask = attempt_manifest["status"].eq("FAILED")
        attempt_manifest.loc[completed_mask, "credit_charge_status"] = (
            "CHARGED_SUCCESS"
            if observed_credit_delta is not None
            else "SUCCESS_CHARGE_UNOBSERVED"
        )
        failed_charge_status = "FAILED_CHARGE_UNKNOWN"
        if observed_credit_delta is None:
            failed_charge_status = "FAILED_CHARGE_UNOBSERVED"
        elif attempted_credit_accounting_matches:
            failed_charge_status = "CHARGED_FAILED_CONFIRMED"
        attempt_manifest.loc[failed_mask, "credit_charge_status"] = (
            failed_charge_status
        )
    canonical_manifest_path = active / "wizard_pair_detail_api_pilot_manifest.csv"
    preserve_canonical_active = bool(
        canonical_manifest_path.exists()
        and (not execute or (credit_blocker and attempted_credits == 0))
    )
    artifact_prefix = (
        "wizard_pair_detail_api_pilot_plan"
        if preserve_canonical_active
        else "wizard_pair_detail_api_pilot"
    )
    completed_attempt_endpoints = tuple(
        attempt_manifest.loc[
            attempt_manifest["status"].eq("COMPLETED"), "endpoint"
        ].astype(str)
    )
    manifest = (
        attempt_manifest
        if preserve_canonical_active
        else _merge_active_manifest(
            canonical_manifest_path,
            attempt_manifest,
            pilot_id=pilot_id,
            selected_endpoint_names=selected_endpoint_names,
        )
    )
    fields = pd.DataFrame(
        field_rows,
        columns=[
            "schema_version",
            "pilot_id",
            "attempt_id",
            "pair_group_key",
            "endpoint",
            "field",
            "type",
            "example",
            "evidence_path",
            "promotion_authority",
            "live_trading_authorized",
        ],
    )
    if not preserve_canonical_active:
        fields = _merge_active_fields(
            active / "wizard_pair_detail_api_pilot_fields.csv",
            fields,
            pilot_id=pilot_id,
            completed_endpoint_names=completed_attempt_endpoints,
        )
    coverage = _coverage_rows(fields, pilot_id=pilot_id, pair_group_key=pair_group_key)
    completed_endpoints = int(manifest["status"].eq("COMPLETED").sum())
    complete = bool(execute and completed_endpoints == len(ENDPOINTS))
    if not execute:
        authority = "PREFLIGHT_ONLY"
        blocker = "execution_not_requested"
    elif credit_blocker:
        authority = "BLOCKED_SCHEMA_PROBE"
        blocker = credit_blocker
    elif not complete:
        authority = "PARTIAL_SCHEMA_PROBE"
        blocker = f"failed_endpoints:{len(ENDPOINTS) - completed_endpoints}"
    else:
        authority = "COMPLETE_SCHEMA_PROBE_ONLY"
        blocker = "dashboard_ecm_and_settings_still_required"

    paths = {
        "manifest_csv": active / f"{artifact_prefix}_manifest.csv",
        "fields": active / f"{artifact_prefix}_fields.csv",
        "coverage": active / f"{artifact_prefix}_coverage.csv",
        "manifest_json": active / f"{artifact_prefix}_manifest.json",
        "summary_md": active / f"{artifact_prefix}_summary.md",
        "attempt_history": active / "wizard_pair_detail_api_pilot_attempt_history.csv",
    }
    if execute and attempted_credits:
        _append_attempt_history(
            paths["attempt_history"],
            active_manifest_path=canonical_manifest_path,
            attempt_manifest=attempt_manifest,
        )
    manifest.to_csv(paths["manifest_csv"], index=False)
    fields.to_csv(paths["fields"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "pilot_id": pilot_id,
        "attempt_id": attempt_id,
        "created_at": as_of.isoformat(),
        "refresh_id": refresh_id,
        "pair_group_key": pair_group_key,
        "request_config": request_config,
        "request_config_hash": config_hash,
        "backtest_contract_source": BACKTEST_CONTRACT_SOURCE,
        "execute": execute,
        "requested_endpoints": list(selected_endpoint_names),
        "requested_endpoint_count": len(selected_endpoints),
        "planned_endpoints": len(ENDPOINTS),
        "completed_endpoints": completed_endpoints,
        "planned_credits": planned_credits,
        "full_bundle_credits": PLANNED_CREDITS,
        "attempted_credits": attempted_credits,
        "completed_credits": completed_credits,
        "charged_failed_credits": max(
            (observed_credit_delta or 0) - completed_credits,
            0,
        ),
        "attempted_credit_accounting_matches": (
            attempted_credit_accounting_matches
        ),
        "credits_used_before": credits_before,
        "credits_remaining_before": credits_remaining_before,
        "credits_used_after": credits_after,
        "credits_remaining_after": (
            max(daily_credit_limit - credits_after, 0)
            if credits_after is not None
            else None
        ),
        "credits_available_after_reserve": (
            max(daily_credit_limit - credits_after - reserved_credits, 0)
            if credits_after is not None
            else None
        ),
        "observed_credit_delta": observed_credit_delta,
        "daily_credit_limit": daily_credit_limit,
        "reserved_credits": reserved_credits,
        "pilot_complete": complete,
        "observed_fields": int(len(fields)),
        "coverage_checks": int(len(coverage)),
        "coverage_passes": int(coverage["status"].eq("FOUND").sum()),
        "ecm_fields_found": bool(
            coverage.loc[
                coverage["field_group"].isin(["ecm_x", "ecm_y", "ecm_strength"]),
                "status",
            ].eq("FOUND").all()
        ),
        "dashboard_pair_detail_complete": False,
        "authority": authority,
        "blocker": blocker,
        "promotion_authority": False,
        "live_trading_authorized": False,
        "inputs": {
            "queue": _relative(queue_path, root),
            "source_accounting": _relative(accounting_path, root),
            "refresh_manifest": _relative(refresh_manifest_path, root),
        },
        "artifacts": {key: _relative(path, root) for key, path in paths.items()},
        "raw_snapshot_directory": _relative(raw_dir, root),
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    paths["manifest_json"].write_text(manifest_text, encoding="utf-8")
    paths["summary_md"].write_text(_summary_markdown(summary), encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _select_source_variant(sources: pd.DataFrame) -> pd.Series:
    priority = {
        "Copula": 0,
        "Dyn (ZScoreR)": 1,
        "OU (ZScoreR)": 2,
        "Static (ZScoreR)": 3,
        "Dyn (Spread)": 4,
        "OU (Spread)": 5,
        "Static (Spread)": 6,
    }
    selected = sources.copy()
    selected["_priority"] = selected["api_exact_mode"].map(priority).fillna(99)
    selected = selected.sort_values(["_priority", "api_source_row_id"], kind="mergesort")
    return selected.iloc[0]


def _request_config(source: pd.Series) -> dict[str, object]:
    spread_type = _api_spread_type(source.get("spread_type"))
    exact_mode = _text(source.get("api_exact_mode"))
    if exact_mode == "Copula":
        strategy = "Copula"
    elif "ZScoreR" in exact_mode:
        strategy = "ZScoreRoll"
    else:
        strategy = "Spread"
    return {
        "api_source_row_id": _text(source.get("api_source_row_id")),
        "pair_id": _text(source.get("pair_id")),
        "symbol_1": _text(source.get("api_symbol_1")),
        "symbol_2": _text(source.get("api_symbol_2")),
        "exchange": _text(source.get("api_exchange")),
        "interval": _text(source.get("api_interval")),
        "period": int(float(source.get("api_period") or 365)),
        "spread_type": spread_type,
        "roll_w": int(float(source.get("api_roll_window") or 42)),
        "strategy": strategy,
        "exact_mode": exact_mode,
        "with_history": True,
    }


def _endpoint_params(endpoint: str, config: dict[str, object]) -> dict[str, object]:
    params: dict[str, object] = {
        "symbol_1": config["symbol_1"],
        "symbol_2": config["symbol_2"],
        "exchange": config["exchange"],
        "interval": config["interval"],
        "period": config["period"],
    }
    if endpoint in {"spread", "zscores"} or (
        endpoint == "backtest" and config["strategy"] != "Copula"
    ):
        params.update(
            {
                "spread_type": config["spread_type"],
                "roll_w": config["roll_w"],
            }
        )
    if endpoint in {"spread", "zscores", "backtest"}:
        params["with_history"] = "true"
    if endpoint in {"cointegration", "copula", "correlations"}:
        params["with_history"] = "true"
    if endpoint == "backtest":
        entry_level, exit_level = BACKTEST_LEVELS[str(config["strategy"])]
        params.update(
            {
                "strategy": config["strategy"],
                "entry_level": entry_level,
                "exit_level": exit_level,
                "x_weighting": 0.5,
                "slippage_rate": 0.0005,
                "commission_rate": 0.0005,
                "stop_loss_rate": 0.10,
            }
        )
    return params


def _select_endpoints(
    endpoint_names: tuple[str, ...] | None,
) -> tuple[tuple[str, str, int], ...]:
    if endpoint_names is None:
        return ENDPOINTS
    requested = tuple(dict.fromkeys(_text(name).lower() for name in endpoint_names))
    known = {item[0] for item in ENDPOINTS}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"Unknown Wizard pair-detail endpoints: {','.join(unknown)}")
    if not requested:
        raise ValueError("At least one Wizard pair-detail endpoint is required")
    return tuple(item for item in ENDPOINTS if item[0] in requested)


def _fetch_endpoint(
    *,
    endpoint_path: str,
    params: dict[str, object],
    api_key: str | None,
    base_url: str,
    timeout: float,
) -> Any:
    response = requests.get(
        f"{base_url.rstrip('/')}{endpoint_path}",
        params=params,
        headers={"X-api-key": api_key or "", "Content-Type": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _failure_payload(exc: Exception, *, api_key: str | None) -> dict[str, object]:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    content_type = ""
    response_body: object = ""
    if response is not None:
        content_type = _text(getattr(response, "headers", {}).get("Content-Type"))
        try:
            response_body = response.json()
        except (ValueError, json.JSONDecodeError):
            response_body = _redact_text(
                _text(getattr(response, "text", ""))[:10_000],
                api_key,
            )
    return {
        "exception_type": type(exc).__name__,
        "message": _redact_text(_safe_error(exc), api_key),
        "http_status": status_code,
        "content_type": content_type,
        "response_body": response_body,
    }


def _merge_active_manifest(
    path: Path,
    attempt_manifest: pd.DataFrame,
    *,
    pilot_id: str,
    selected_endpoint_names: tuple[str, ...],
) -> pd.DataFrame:
    if not path.exists():
        return attempt_manifest
    previous = pd.read_csv(path, keep_default_na=False)
    if previous.empty or not previous["pilot_id"].astype(str).eq(pilot_id).any():
        return attempt_manifest
    previous = previous[previous["pilot_id"].astype(str).eq(pilot_id)].copy()
    previous_completed = set(
        previous.loc[previous["status"].eq("COMPLETED"), "endpoint"].astype(str)
    )
    attempted_completed = set(
        attempt_manifest.loc[
            attempt_manifest["status"].eq("COMPLETED"), "endpoint"
        ].astype(str)
    )
    replace_endpoints = attempted_completed | (
        set(selected_endpoint_names) - previous_completed
    )
    previous = previous[
        ~previous["endpoint"].astype(str).isin(replace_endpoints)
    ]
    current = attempt_manifest[
        attempt_manifest["endpoint"].astype(str).isin(replace_endpoints)
    ]
    merged = pd.concat([previous, current], ignore_index=True, sort=False)
    endpoint_order = {name: index for index, (name, _, _) in enumerate(ENDPOINTS)}
    merged["_endpoint_order"] = merged["endpoint"].map(endpoint_order).fillna(999)
    return merged.sort_values("_endpoint_order", kind="mergesort").drop(
        columns="_endpoint_order"
    )


def _merge_active_fields(
    path: Path,
    attempt_fields: pd.DataFrame,
    *,
    pilot_id: str,
    completed_endpoint_names: tuple[str, ...],
) -> pd.DataFrame:
    if not path.exists():
        return attempt_fields
    previous = pd.read_csv(path, keep_default_na=False)
    if previous.empty or not previous["pilot_id"].astype(str).eq(pilot_id).any():
        return attempt_fields
    previous = previous[
        previous["pilot_id"].astype(str).eq(pilot_id)
        & ~previous["endpoint"].astype(str).isin(completed_endpoint_names)
    ]
    current = attempt_fields[
        attempt_fields["endpoint"].astype(str).isin(completed_endpoint_names)
    ]
    return pd.concat([previous, current], ignore_index=True, sort=False)


def _append_attempt_history(
    path: Path,
    *,
    active_manifest_path: Path,
    attempt_manifest: pd.DataFrame,
) -> None:
    frames: list[pd.DataFrame] = []
    if path.exists():
        frames.append(pd.read_csv(path, keep_default_na=False))
    elif active_manifest_path.exists():
        frames.append(pd.read_csv(active_manifest_path, keep_default_na=False))
    frames.append(attempt_manifest)
    history = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    dedupe_columns = [
        name
        for name in (
            "pilot_id",
            "attempt_id",
            "endpoint",
            "requested_at",
            "status",
            "response_sha256",
        )
        if name in history.columns
    ]
    if dedupe_columns:
        history = history.drop_duplicates(subset=dedupe_columns, keep="last")
    history.to_csv(path, index=False)


def _coverage_rows(
    fields: pd.DataFrame,
    *,
    pilot_id: str,
    pair_group_key: str,
) -> pd.DataFrame:
    checks = {
        "spread_series": ("spread",),
        "zscore_series": ("zscore",),
        "pearson": ("pearson",),
        "spearman": ("spearman",),
        "kendall": ("kendall",),
        "cointegration": ("is_coint", "p_value", "t_stat"),
        "copula_family": ("copula",),
        "copula_conditionals": ("u1_given_u2", "u2_given_u1"),
        "backtest_return": ("total_return", "annual_return", "strat_returns"),
        "backtest_sharpe": ("sharpe",),
        "backtest_drawdown": ("drawdown",),
        "ecm_x": ("ecm_x",),
        "ecm_y": ("ecm_y",),
        "ecm_strength": ("ecm_strength",),
        "dashboard_entry_exit_controls": (
            "entry_long_operator",
            "exit_long_operator",
            "copula_entry_lower",
        ),
    }
    field_values = fields.get("field", pd.Series(dtype=str)).astype(str).str.lower()
    rows: list[dict[str, object]] = []
    for group, patterns in checks.items():
        matched = sorted(
            {
                value
                for value in field_values
                if any(pattern in value for pattern in patterns)
            }
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "pilot_id": pilot_id,
                "pair_group_key": pair_group_key,
                "field_group": group,
                "status": "FOUND" if matched else "MISSING",
                "matched_fields": ";".join(matched),
                "required_for_complete_dashboard_pair_detail": True,
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows)


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Wizard Pair-Detail API Pilot",
            "",
            f"- Pilot: `{summary['pilot_id']}`",
            f"- Pair group: `{summary['pair_group_key']}`",
            f"- Authority: `{summary['authority']}`",
            f"- Endpoints complete: {summary['completed_endpoints']} / {summary['planned_endpoints']}",
            f"- Credits planned / observed: {summary['planned_credits']} / {summary['observed_credit_delta']}",
            "- Current credits used / available after reserve: "
            f"{summary['credits_used_after']} / {summary['credits_available_after_reserve']}",
            f"- Observed fields: {summary['observed_fields']}",
            f"- ECM fields found: `{str(summary['ecm_fields_found']).lower()}`",
            "- Dashboard pair detail complete: `false`",
            "- Promotion authority: `false`",
            "- Live trading authorized: `false`",
            f"- Blocker: `{summary['blocker']}`",
            "",
            "This is a bounded API schema probe. It cannot replace authenticated dashboard ECM/settings capture or local Hyperliquid acceptance evidence.",
            "",
        ]
    )


def _api_spread_type(value: object) -> str:
    return {
        "dynamic": "Dynamic",
        "dyn": "Dynamic",
        "ou": "OU",
        "static": "Static",
    }.get(_text(value).lower(), "Static")


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, keep_default_na=False)
    if frame.empty:
        raise ValueError(f"Required CSV is empty: {path}")
    return frame


def _read_json_required(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Required JSON object is invalid: {path}")
    return payload


def _safe_error(exc: Exception) -> str:
    return str(exc).replace("\n", " ")[:500]


def _redact_text(value: str, api_key: str | None) -> str:
    if api_key:
        return value.replace(api_key, "[REDACTED]")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
