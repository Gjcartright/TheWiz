from __future__ import annotations

from pathlib import Path

import pytest

from quant_platform.orchestration.corrective_financial_effect_registry import (
    financial_effect_surface_rows,
    unfenced_financial_effect_surface_ids,
)


def test_unknown_financial_effect_surface_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "src" / "quant_platform" / "new_venue.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def submit(exchange, order):\n    return exchange.create_order(order)\n",
        encoding="utf-8",
    )

    rows = financial_effect_surface_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["method"] == "create_order"
    assert rows[0]["migration_state"] == "UNMIGRATED"
    assert rows[0]["blocker"] == "financial_effect_surface_not_reviewed"


def test_appledouble_sidecar_is_not_scanned_as_source(tmp_path: Path) -> None:
    source = tmp_path / "src" / "quant_platform" / "new_venue.py"
    source.parent.mkdir(parents=True)
    source.write_text("def submit(exchange):\n    return exchange.create_order({})\n")
    source.with_name("._new_venue.py").write_bytes(b"\x00\x05\x16\x07Mac OS X")

    rows = financial_effect_surface_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["source_path"] == "src/quant_platform/new_venue.py"


def test_reviewed_surface_without_fence_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    source = (
        tmp_path
        / "src"
        / "quant_platform"
        / "dydx_sdk_order_adapter.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        "async def _place_order(node, wallet, order):\n"
        "    return await node.place_order(wallet, order)\n",
        encoding="utf-8",
    )

    rows = financial_effect_surface_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["classification"] == "direct_order_sink"
    assert rows[0]["migration_state"] == "UNMIGRATED"
    assert rows[0]["blocker"] == "financial_effect_fence_evidence_missing_or_late"


@pytest.mark.parametrize(
    "decoy",
    [
        '    marker = "claim_effect_dispatch"\n',
        "    marker = authority.claim_effect_dispatch\n",
        "    marker = lambda: claim_effect_dispatch(authorization)\n",
        "    if enabled:\n        claim_effect_dispatch(authorization)\n",
    ],
)
def test_fence_decoys_do_not_certify_financial_sink(
    tmp_path: Path,
    decoy: str,
) -> None:
    source = tmp_path / "src" / "quant_platform" / "dydx_sdk_order_adapter.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "async def _place_order(node, wallet, order, authority, authorization, enabled):\n"
        + decoy
        + "    return await node.place_order(wallet, order)\n",
        encoding="utf-8",
    )

    rows = financial_effect_surface_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["migration_state"] == "UNMIGRATED"
    assert rows[0]["blocker"] == "financial_effect_fence_evidence_missing_or_late"


def test_actual_dominating_fence_call_certifies_reviewed_sink(tmp_path: Path) -> None:
    source = tmp_path / "src" / "quant_platform" / "dydx_sdk_order_adapter.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "async def _place_order(node, wallet, order, authorization):\n"
        "    claim_effect_dispatch(authorization)\n"
        "    return await node.place_order(wallet, order)\n",
        encoding="utf-8",
    )

    rows = financial_effect_surface_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["migration_state"] == "MIGRATED"
    assert rows[0]["blocker"] == ""


def test_canonical_binance_function_call_requires_pair_adapter_fence(tmp_path: Path) -> None:
    source = tmp_path / "src" / "quant_platform" / "binance_testnet.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def _gate00g_binance_pair_order_call(client, intent, config):\n"
        "    return _CANONICAL_BINANCE_PLACE_ORDER(client, intent, config)\n",
        encoding="utf-8",
    )
    rows = financial_effect_surface_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["migration_state"] == "UNMIGRATED"
    assert rows[0]["blocker"] == "financial_effect_fence_evidence_missing_or_late"

    source.write_text(
        "def _gate00g_binance_pair_order_call(client, intent, config):\n"
        "    canonical = _require_gate00g_binance_pair_adapter(client)\n"
        "    return _CANONICAL_BINANCE_PLACE_ORDER(canonical, intent, config)\n",
        encoding="utf-8",
    )
    rows = financial_effect_surface_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["migration_state"] == "MIGRATED"
    assert rows[0]["blocker"] == ""


def test_current_tree_financial_effect_inventory_is_fully_fenced() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = financial_effect_surface_rows(root)

    assert len(rows) == 18
    assert any(
        row["source_path"] == "src/quant_platform/binance_testnet.py"
        and row["function"] == "_gate00g_binance_pair_order_call"
        and row["method"] == "place_order"
        for row in rows
    )
    assert unfenced_financial_effect_surface_ids(root) == []
    assert {str(row["effect_kind"]) for row in rows} == {
        "cancel",
        "close",
        "leverage",
        "order",
        "transfer",
    }
