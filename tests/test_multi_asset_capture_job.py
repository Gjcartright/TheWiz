import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest

from quant_platform.orchestration import multi_asset_capture_job as capture_job
from quant_platform.orchestration.corrective_external_effects import (
    RESEARCH_EXTERNAL_EFFECT_PROFILE,
    ExternalEffectCallContract,
    external_effect_authority_session,
)
from quant_platform.orchestration.effect_authority import (
    _CURRENT_PUBLICATION_AUTHORITY,
    EffectAuthority,
    EffectAuthorityError,
)
from quant_platform.orchestration.multi_asset_capture_job import capture_multi_asset_universe
from quant_platform.orchestration.multi_asset_capture_verify import verify_multi_asset_store

NOW = datetime(2026, 9, 19, 12, tzinfo=UTC)
HOUR_MS = 60 * 60 * 1000
FINGERPRINT = "a" * 64


@pytest.fixture(autouse=True)
def multi_asset_public_network_authority(tmp_path, request):
    if request.node.get_closest_marker("without_public_network_authority"):
        yield
        return
    authority = EffectAuthority(
        root=tmp_path,
        secret=b"multi-asset-network-authority-test-secret-32-bytes",
        issuer_id="multi_asset_test_supervisor",
        profile=RESEARCH_EXTERNAL_EFFECT_PROFILE,
    )
    contracts = frozenset(
        ExternalEffectCallContract(
            operation=f"HYPERLIQUID_MAINNET_{suffix}",
            method="POST",
            target=capture_job.INFO_URL,
            credit_units_per_request=0,
        )
        for suffix in ("META_AND_ASSET_CONTEXTS", "CANDLE_SNAPSHOT")
    )
    with external_effect_authority_session(
        authority=authority,
        run_id="multi-asset-test-run",
        intended_slot_id="multi-asset-test-slot",
        source_fingerprint_sha256=FINGERPRINT,
        runtime_fingerprint_sha256=FINGERPRINT,
        configuration_fingerprint_sha256=FINGERPRINT,
        provider_id="hyperliquid_public",
        account_scope_id="hyperliquid:public_research",
        reservation_id="multi-asset-test-reservation",
        reservation_sha256=sha256(b"multi-asset-test-reservation").hexdigest(),
        allowed_targets=frozenset({capture_job.INFO_URL}),
        allowed_credential_keys=frozenset(),
        max_total_requests=128,
        max_total_credits=0,
        allowed_call_contracts=contracts,
    ):
        yield


def _candle(symbol: str, start_ms: int, *, close: str = "10", **changes):
    row = {
        "s": symbol,
        "i": "1h",
        "t": start_ms,
        "T": start_ms + HOUR_MS - 1,
        "o": "9",
        "h": "11",
        "l": "8",
        "c": close,
        "v": "100",
        "n": 3,
    }
    row.update(changes)
    return row


def _candles(symbol: str, now: datetime = NOW) -> list[dict[str, object]]:
    latest_start = int(now.timestamp() * 1000) - 3 * HOUR_MS
    return [_candle(symbol, latest_start + index * HOUR_MS) for index in range(3)]


def _fetch(payload, *, candle_transform=None):
    if payload["type"] == "metaAndAssetCtxs":
        return json.dumps(
            [
                {"universe": [{"name": "BTC"}, {"name": "ETH"}, {"name": "SOL"}]},
                [
                    {"dayNtlVlm": "40e6", "markPx": "1", "openInterest": "40e6"},
                    {"dayNtlVlm": "30e6", "markPx": "1", "openInterest": "30e6"},
                    {"dayNtlVlm": "20e6", "markPx": "1", "openInterest": "20e6"},
                ],
            ]
        ).encode()
    coin = payload["req"]["coin"]
    rows = _candles(coin)
    return json.dumps(candle_transform(coin, rows) if candle_transform else rows).encode()


def test_raw_hyperliquid_response_is_read_before_context_closes(monkeypatch):
    class ClosingResponse:
        status = 200
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            self.closed = True

        def read(self):
            if self.closed:
                raise ValueError("response already closed")
            return b"{}"

    monkeypatch.setattr(capture_job, "urlopen", lambda *_args, **_kwargs: ClosingResponse())
    assert capture_job._raw_hyperliquid_info_call({"type": "metaAndAssetCtxs"}) == b"{}"


def test_multi_asset_capture_is_raw_first_and_zero_credit(tmp_path):
    result = capture_multi_asset_universe(root=tmp_path, now=NOW, fetch=_fetch, max_assets=3)
    assert result["assets"] == ("BTC", "ETH", "SOL")
    assert result["candidate_pair_count"] == 3
    assert result["network_calls"] == 4
    assert result["wizard_credits"] == 0
    assert not result["credentials_used"]


def test_publication_authority_is_required_before_any_fetch(tmp_path):
    called = False

    def unexpected_fetch(_payload):
        nonlocal called
        called = True
        raise AssertionError("fetch must not run without publication authority")

    token = _CURRENT_PUBLICATION_AUTHORITY.set(None)
    try:
        with pytest.raises(EffectAuthorityError, match="publication_authority_session_missing"):
            capture_multi_asset_universe(
                root=tmp_path,
                now=NOW,
                fetch=unexpected_fetch,
                max_assets=3,
            )
    finally:
        _CURRENT_PUBLICATION_AUTHORITY.reset(token)
    assert not called
    assert not (tmp_path / "data").exists()


@pytest.mark.without_public_network_authority
def test_public_network_authority_is_required_before_any_fetch(tmp_path):
    called = False

    def unexpected_fetch(_payload):
        nonlocal called
        called = True
        raise AssertionError("fetch must not run without public-network authority")

    with pytest.raises(EffectAuthorityError, match="session_missing"):
        capture_multi_asset_universe(
            root=tmp_path,
            now=NOW,
            fetch=unexpected_fetch,
            max_assets=3,
        )
    assert not called
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("dayNtlVlm", "NaN"),
        ("dayNtlVlm", "Infinity"),
        ("markPx", "Infinity"),
        ("openInterest", "NaN"),
    ],
)
def test_nonfinite_market_metadata_is_excluded_from_universe(field, value):
    raw = json.dumps(
        [
            {"universe": [{"name": "BAD"}, {"name": "ETH"}, {"name": "SOL"}]},
            [
                {"dayNtlVlm": "40e6", "markPx": "1", "openInterest": "40e6"},
                {"dayNtlVlm": "30e6", "markPx": "1", "openInterest": "30e6"},
                {"dayNtlVlm": "20e6", "markPx": "1", "openInterest": "20e6"},
            ],
        ]
    ).encode()
    decoded = json.loads(raw)
    decoded[1][0][field] = value
    assets = capture_job._parse_metadata(
        json.dumps(decoded).encode(),
        minimum_notional=10,
        minimum_open_interest_notional=5,
        max_assets=3,
    )
    assert assets == ("ETH", "SOL")


def test_overflowing_open_interest_notional_is_excluded_from_universe():
    decoded = json.loads(_fetch({"type": "metaAndAssetCtxs"}))
    decoded[1][0]["markPx"] = "1e308"
    decoded[1][0]["openInterest"] = "1e308"
    assets = capture_job._parse_metadata(
        json.dumps(decoded).encode(),
        minimum_notional=10,
        minimum_open_interest_notional=5,
        max_assets=3,
    )
    assert assets == ("ETH", "SOL")


def test_duplicate_asset_names_in_metadata_are_rejected():
    decoded = json.loads(_fetch({"type": "metaAndAssetCtxs"}))
    decoded[0]["universe"][1]["name"] = "BTC"
    with pytest.raises(ValueError, match="multi_asset_metadata_duplicate_asset"):
        capture_job._parse_metadata(
            json.dumps(decoded).encode(),
            minimum_notional=10,
            minimum_open_interest_notional=5,
            max_assets=3,
        )


def test_boolean_metadata_numbers_are_rejected():
    decoded = json.loads(_fetch({"type": "metaAndAssetCtxs"}))
    decoded[1][0]["dayNtlVlm"] = True
    with pytest.raises(ValueError, match="multi_asset_metadata_value_invalid"):
        capture_job._parse_metadata(
            json.dumps(decoded).encode(),
            minimum_notional=10,
            minimum_open_interest_notional=5,
            max_assets=3,
        )


def test_asset_filename_label_cannot_escape_raw_directory():
    label = capture_job._asset_raw_label("../../outside\\file")
    assert "/" not in label
    assert "\\" not in label
    assert label.startswith("candles_")


def test_nonfinite_capture_threshold_is_rejected_before_fetch(tmp_path):
    with pytest.raises(ValueError, match="multi_asset_capture_configuration_invalid"):
        capture_multi_asset_universe(
            root=tmp_path,
            now=NOW,
            fetch=lambda _payload: (_ for _ in ()).throw(AssertionError("fetch must not run")),
            max_assets=3,
            minimum_notional=float("nan"),
        )


def test_official_verifier_accepts_a_complete_synthetic_capture(tmp_path):
    capture_multi_asset_universe(root=tmp_path, now=NOW, fetch=_fetch, max_assets=3)
    result = verify_multi_asset_store(root=tmp_path, now=NOW + timedelta(minutes=1))
    assert result["status"] == "PASS"
    assert result["capture_count"] == 1
    assert result["age_minutes"] == 1
    assert result["authority"]["execution"] is False


def test_official_verifier_rejects_unlinked_capture_directory(tmp_path):
    first = capture_multi_asset_universe(root=tmp_path, now=NOW, fetch=_fetch, max_assets=3)
    capture_multi_asset_universe(
        root=tmp_path,
        prior_capture_id=first["capture_id"],
        now=NOW + timedelta(hours=1),
        fetch=_fetch,
        max_assets=3,
    )
    orphan = tmp_path / "data" / "research" / "multi_asset_captures" / "universe-20260919T110000Z"
    orphan.mkdir()
    (orphan / "receipt.json").write_text("{}")
    with pytest.raises(ValueError, match="multi_asset_capture_chain_incomplete"):
        verify_multi_asset_store(root=tmp_path, now=NOW + timedelta(hours=2))


def test_official_verifier_rejects_future_or_naive_verification_time(tmp_path):
    capture_multi_asset_universe(root=tmp_path, now=NOW, fetch=_fetch, max_assets=3)
    with pytest.raises(ValueError, match="multi_asset_terminal_time_invalid"):
        verify_multi_asset_store(root=tmp_path, now=NOW - timedelta(minutes=1))
    with pytest.raises(ValueError, match="multi_asset_terminal_time_invalid"):
        verify_multi_asset_store(root=tmp_path, now=NOW.replace(tzinfo=None))


def test_multi_asset_capture_requires_forward_predecessor(tmp_path):
    first = capture_multi_asset_universe(root=tmp_path, now=NOW, fetch=_fetch, max_assets=3)
    with pytest.raises(ValueError, match="multi_asset_nonforward_observation"):
        capture_multi_asset_universe(
            root=tmp_path, prior_capture_id=first["capture_id"], now=NOW, fetch=_fetch, max_assets=3
        )
    second = capture_multi_asset_universe(
        root=tmp_path,
        prior_capture_id=first["capture_id"],
        now=NOW + timedelta(hours=1),
        fetch=_fetch,
        max_assets=3,
    )
    assert second["network_calls"] == 4


def test_multi_asset_capture_rejects_new_root_when_store_is_not_empty(tmp_path):
    capture_multi_asset_universe(root=tmp_path, now=NOW, fetch=_fetch, max_assets=3)
    called = False

    def unexpected_fetch(_payload):
        nonlocal called
        called = True
        raise AssertionError("capture must not fork the receipt chain")

    with pytest.raises(ValueError, match="multi_asset_predecessor_required"):
        capture_multi_asset_universe(
            root=tmp_path,
            now=NOW + timedelta(hours=2),
            fetch=unexpected_fetch,
            max_assets=3,
        )
    assert not called


def test_multi_asset_capture_rejects_branch_from_nonterminal_predecessor(tmp_path):
    first = capture_multi_asset_universe(root=tmp_path, now=NOW, fetch=_fetch, max_assets=3)
    second = capture_multi_asset_universe(
        root=tmp_path,
        prior_capture_id=first["capture_id"],
        now=NOW + timedelta(hours=1),
        fetch=_fetch,
        max_assets=3,
    )
    called = False

    def unexpected_fetch(_payload):
        nonlocal called
        called = True
        raise AssertionError("capture must extend the terminal receipt")

    with pytest.raises(ValueError, match="multi_asset_predecessor_not_terminal"):
        capture_multi_asset_universe(
            root=tmp_path,
            prior_capture_id=first["capture_id"],
            now=NOW + timedelta(hours=2),
            fetch=unexpected_fetch,
            max_assets=3,
        )
    assert not called
    assert second["capture_id"] == "universe-20260919T130000Z"


def test_duplicate_capture_id_is_rejected_before_fetch(tmp_path):
    capture_multi_asset_universe(root=tmp_path, now=NOW, fetch=_fetch, max_assets=3)
    with pytest.raises(FileExistsError, match="multi_asset_capture_target_exists"):
        capture_multi_asset_universe(
            root=tmp_path,
            now=NOW,
            fetch=lambda _payload: (_ for _ in ()).throw(AssertionError("fetch must not run")),
            max_assets=3,
        )


def test_tampered_predecessor_raw_is_rejected_before_fetch(tmp_path):
    first = capture_multi_asset_universe(root=tmp_path, now=NOW, fetch=_fetch, max_assets=3)
    raw = (
        tmp_path
        / "data"
        / "research"
        / "multi_asset_captures"
        / first["capture_id"]
        / "raw"
        / "metadata.json"
    )
    raw.chmod(0o600)
    raw.write_bytes(b"tampered")
    called = False

    def unexpected_fetch(_payload):
        nonlocal called
        called = True
        raise AssertionError("fetch must not run")

    with pytest.raises(ValueError, match="multi_asset_predecessor_raw_hash_mismatch"):
        capture_multi_asset_universe(
            root=tmp_path,
            prior_capture_id=first["capture_id"],
            now=NOW + timedelta(hours=1),
            fetch=unexpected_fetch,
            max_assets=3,
        )
    assert not called


def test_broken_ancestor_chain_is_rejected_before_fetch(tmp_path):
    first = capture_multi_asset_universe(root=tmp_path, now=NOW, fetch=_fetch, max_assets=3)
    receipt_path = (
        tmp_path
        / "data"
        / "research"
        / "multi_asset_captures"
        / first["capture_id"]
        / "receipt.json"
    )
    receipt_path.chmod(0o600)
    receipt = json.loads(receipt_path.read_text())
    receipt["prior_capture_id"] = "universe-20260918T110000Z"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    called = False

    def unexpected_fetch(_payload):
        nonlocal called
        called = True
        raise AssertionError("fetch must not run")

    with pytest.raises(ValueError, match="multi_asset_predecessor_missing_or_invalid"):
        capture_multi_asset_universe(
            root=tmp_path,
            prior_capture_id=first["capture_id"],
            now=NOW + timedelta(hours=1),
            fetch=unexpected_fetch,
            max_assets=3,
        )
    assert not called


@pytest.mark.parametrize(
    "changes",
    [
        {"i": "30m"},
        {"t": int(NOW.timestamp() * 1000) - 3 * HOUR_MS + 1},
        {"o": "NaN"},
        {"h": "9"},
        {"v": "-1"},
        {"n": -1},
        {"c": "Infinity"},
    ],
)
def test_invalid_candle_schema_is_rejected(tmp_path, changes):
    def malformed(_coin, rows):
        rows[1].update(changes)
        return rows

    with pytest.raises(ValueError, match="multi_asset_candle_(identity|value)_invalid"):
        capture_multi_asset_universe(
            root=tmp_path,
            now=NOW,
            fetch=lambda payload: _fetch(payload, candle_transform=malformed),
            max_assets=3,
        )


@pytest.mark.parametrize("field,value", [("h", "12"), ("v", "99"), ("n", 4), ("o", "9.1")])
def test_closed_candle_any_field_revision_is_rejected(tmp_path, field, value):
    first = capture_multi_asset_universe(root=tmp_path, now=NOW, fetch=_fetch, max_assets=3)

    def revised(_coin, rows):
        rows[0][field] = value
        return rows

    with pytest.raises(ValueError, match="multi_asset_closed_candle_revision_unexplained"):
        capture_multi_asset_universe(
            root=tmp_path,
            prior_capture_id=first["capture_id"],
            now=NOW + timedelta(hours=1),
            fetch=lambda payload: _fetch(payload, candle_transform=revised),
            max_assets=3,
        )


def test_missing_previously_closed_candle_is_rejected(tmp_path):
    first = capture_multi_asset_universe(root=tmp_path, now=NOW, fetch=_fetch, max_assets=3)

    def missing(_coin, rows):
        return rows[1:]

    with pytest.raises(ValueError, match="multi_asset_closed_candle_revision_unexplained"):
        capture_multi_asset_universe(
            root=tmp_path,
            prior_capture_id=first["capture_id"],
            now=NOW + timedelta(hours=1),
            fetch=lambda payload: _fetch(payload, candle_transform=missing),
            max_assets=3,
        )


def test_interrupted_publication_leaves_only_hidden_staging_evidence(tmp_path, monkeypatch):
    def fail_fsync(_path):
        raise OSError("simulated directory sync failure")

    monkeypatch.setattr(capture_job, "_fsync_directory", fail_fsync)
    with pytest.raises(OSError, match="simulated directory sync failure"):
        capture_multi_asset_universe(root=tmp_path, now=NOW, fetch=_fetch, max_assets=3)

    base = tmp_path / "data" / "research" / "multi_asset_captures"
    assert not (base / "universe-20260919T120000Z").exists()
    staging = tuple(base.glob(".universe-20260919T120000Z.staging-*"))
    assert len(staging) == 1
    assert (staging[0] / "receipt.json").exists()


def test_publication_permission_failure_never_exposes_final_capture(tmp_path, monkeypatch):
    original_chmod = capture_job.os.chmod

    def fail_raw_directory(path, mode, *args, **kwargs):
        if str(path).endswith("/raw"):
            raise PermissionError("simulated directory permission failure")
        return original_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(capture_job.os, "chmod", fail_raw_directory)
    with pytest.raises(PermissionError, match="simulated directory permission failure"):
        capture_multi_asset_universe(root=tmp_path, now=NOW, fetch=_fetch, max_assets=3)

    base = tmp_path / "data" / "research" / "multi_asset_captures"
    assert not (base / "universe-20260919T120000Z").exists()
    assert tuple(base.glob(".universe-20260919T120000Z.staging-*"))


def test_naive_observation_time_is_rejected_before_fetch(tmp_path):
    with pytest.raises(ValueError, match="multi_asset_capture_time_must_be_timezone_aware"):
        capture_multi_asset_universe(
            root=tmp_path,
            now=NOW.replace(tzinfo=None),
            fetch=lambda _payload: (_ for _ in ()).throw(AssertionError("fetch must not run")),
            max_assets=3,
        )
