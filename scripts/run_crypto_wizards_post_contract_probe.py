from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import pandas as pd

from quant_platform.api_extraction import CryptoWizardsExtractor
from quant_platform.crypto_wizards_history import (
    CryptoWizardsCustomSeriesAnalyticsRequest,
    fetch_credits_used,
    fetch_custom_series_analytics,
)
from quant_platform.crypto_wizards_sweep import parse_wizard_credit_usage
from quant_platform.env import load_env_file
from quant_platform.wizard_credit_ledger import (
    PROOF_LANE,
    reconcile_wizard_credit_lane,
)


ROOT = Path(__file__).resolve().parents[1]
ENDPOINTS = ("cointegration", "correlations", "spread", "zscores")
PLANNED_CREDITS = len(ENDPOINTS)
SCHEMA_VERSION = "crypto_wizards_post_contract_probe.v1"


def main() -> None:
    now = datetime.now(UTC)
    load_env_file(ROOT / ".env.local", override=False)
    api_key = os.getenv("CRYPTO_WIZARDS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("CRYPTO_WIZARDS_API_KEY is required")

    summary_path = ROOT / "reports" / "active" / "crypto_wizards_post_contract_probe.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            previous.get("credit_date_utc") == now.date().isoformat()
            and previous.get("status") == "PASS"
        ):
            print(json.dumps(previous, indent=2, sort_keys=True))
            return

    reservation_path = (
        ROOT
        / "data"
        / "research"
        / "wizard_credit_ledger"
        / now.date().isoformat()
        / "reservations"
        / f"{PROOF_LANE}.json"
    )
    if not reservation_path.exists():
        raise SystemExit("current proof-lane credit reservation is missing")
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    reservation_id = str(reservation.get("reservation_id", ""))
    if not reservation_id:
        raise SystemExit("proof-lane credit reservation has no reservation_id")

    source_path, request = _frozen_request()
    request_hash = sha256(
        json.dumps(request.payload("spread"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    probe_id = "cwpostprobe_" + request_hash[:20]
    raw_dir = (
        ROOT
        / "data"
        / "raw"
        / "crypto_wizards"
        / "post_contract_probe"
        / now.date().isoformat()
        / probe_id
    )
    raw_dir.mkdir(parents=True, exist_ok=True)

    before = parse_wizard_credit_usage(fetch_credits_used(api_key=api_key))
    if not before.known or before.remaining is None:
        raise SystemExit("Crypto Wizards credit usage is unknown")
    protected_reserve = int(reservation.get("protected_reserve", 100))
    if before.remaining - protected_reserve < PLANNED_CREDITS:
        raise SystemExit("insufficient Crypto Wizards credits after protected reserve")

    rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []
    completed = 0
    for endpoint in ENDPOINTS:
        payload = request.payload(endpoint)
        request_path = raw_dir / f"{endpoint}_request.json"
        request_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "probe_id": probe_id,
                    "endpoint": endpoint,
                    "captured_at": now.isoformat(),
                    "source_history_path": _relative(source_path),
                    "request": payload,
                    "promotion_authority": False,
                    "live_trading_authorized": False,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        response_path = raw_dir / f"{endpoint}_response.json"
        error = ""
        status = "FAILED"
        fields = 0
        try:
            response = fetch_custom_series_analytics(endpoint, request, api_key=api_key)
            discovered = CryptoWizardsExtractor.discover_fields(response)
            fields = len(discovered)
            response_path.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "probe_id": probe_id,
                        "endpoint": endpoint,
                        "captured_at": datetime.now(UTC).isoformat(),
                        "response": response,
                        "promotion_authority": False,
                        "live_trading_authorized": False,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            for field in discovered:
                field_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "probe_id": probe_id,
                        "endpoint": endpoint,
                        **field,
                        "evidence_path": _relative(response_path),
                        "promotion_authority": False,
                        "live_trading_authorized": False,
                    }
                )
            status = "COMPLETED"
            completed += 1
        except Exception as exc:  # Preserve the bounded vendor contract failure.
            error = " ".join(str(exc).split())[:1000]
            response_path = raw_dir / f"{endpoint}_failure.json"
            response_path.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "probe_id": probe_id,
                        "endpoint": endpoint,
                        "captured_at": datetime.now(UTC).isoformat(),
                        "error_type": type(exc).__name__,
                        "error": error,
                        "promotion_authority": False,
                        "live_trading_authorized": False,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "probe_id": probe_id,
                "endpoint": endpoint,
                "method": "POST",
                "path": f"/v1beta/{endpoint}",
                "credit_cost": 1,
                "status": status,
                "observed_field_count": fields,
                "request_path": _relative(request_path),
                "response_path": _relative(response_path),
                "error": error,
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )

    after = parse_wizard_credit_usage(fetch_credits_used(api_key=api_key))
    observed_delta = (
        after.used - before.used
        if after.known and after.used is not None and before.used is not None
        else None
    )
    reconciliation = reconcile_wizard_credit_lane(
        root=ROOT,
        lane=PROOF_LANE,
        reservation_id=reservation_id,
        reconciliation_key=f"{SCHEMA_VERSION}|{probe_id}",
        attempted_credits=PLANNED_CREDITS,
        completed_credits=completed,
        external_requests=PLANNED_CREDITS,
        observed_used_before=before.used,
        now=now,
    )
    status = (
        "PASS"
        if completed == PLANNED_CREDITS
        and observed_delta == PLANNED_CREDITS
        and reconciliation.summary.get("status") in {"PASS_RECONCILED", "REUSED_RECONCILIATION"}
        else "BLOCKED"
    )
    active = ROOT / "reports" / "active"
    manifest_path = active / "crypto_wizards_post_contract_probe.csv"
    fields_path = active / "crypto_wizards_post_contract_probe_fields.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    pd.DataFrame(field_rows).to_csv(fields_path, index=False)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "probe_id": probe_id,
        "status": status,
        "credit_date_utc": now.date().isoformat(),
        "captured_at": now.isoformat(),
        "endpoints": list(ENDPOINTS),
        "planned_credits": PLANNED_CREDITS,
        "completed_endpoints": completed,
        "credits_used_before": before.used,
        "credits_used_after": after.used,
        "observed_credit_delta": observed_delta,
        "reservation_id": reservation_id,
        "reconciliation_id": reconciliation.summary.get("reconciliation_id", ""),
        "reconciliation_status": reconciliation.summary.get("status", ""),
        "reconciliation_path": _relative(reconciliation.paths["reconciliation"]),
        "source_history_path": _relative(source_path),
        "manifest_path": _relative(manifest_path),
        "fields_path": _relative(fields_path),
        "raw_directory": _relative(raw_dir),
        "promotion_authority": False,
        "live_trading_authorized": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _frozen_request() -> tuple[Path, CryptoWizardsCustomSeriesAnalyticsRequest]:
    candidates = sorted(
        (
            ROOT
            / "reports"
            / "snapshots"
            / "exhaustive_wizard_hyperliquid"
            / "ewhl_eb77ddc97799a978655d"
        ).glob("**/*derived_history.json")
    )
    for path in reversed(candidates):
        payload = json.loads(path.read_text(encoding="utf-8"))
        history = payload.get("history", [])
        pairs = [
            (float(row.get("price_x")), float(row.get("price_y")))
            for row in history
            if _positive_finite(row.get("price_x")) and _positive_finite(row.get("price_y"))
        ]
        if len(pairs) >= 360:
            selected = pairs[-360:]
            return path, CryptoWizardsCustomSeriesAnalyticsRequest(
                series_1_closes=tuple(value[0] for value in selected),
                series_2_closes=tuple(value[1] for value in selected),
                spread_type="Static",
                roll_w=42,
                with_history=True,
            )
    raise SystemExit("no frozen 360-row Hyperliquid history is available")


def _positive_finite(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
