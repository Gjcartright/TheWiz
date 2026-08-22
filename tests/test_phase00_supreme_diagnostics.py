from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import UTC, date, datetime, time, timedelta
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import quant_platform.orchestration.corrective_active_artifact_envelopes as envelopes
import quant_platform.orchestration.corrective_phase00_closure as closure
import quant_platform.orchestration.corrective_phase00_control as phase00_control
import quant_platform.orchestration.corrective_phase00_descendants as descendants
from quant_platform.orchestration import effect_authority
from quant_platform.orchestration.corrective_artifact_retention import (
    run_corrective_artifact_retention,
)
from quant_platform.orchestration.corrective_external_effects import (
    RESEARCH_EXTERNAL_EFFECT_PROFILE,
    external_effect_authority_session,
    read_authorized_credential,
    run_authorized_credit_call,
)
from quant_platform.orchestration.corrective_phase00_diagnostics import (
    evaluate_cadence_slo,
    exhaustive_run_control_model,
)
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_text,
    scheduler_contract,
)
from quant_platform.orchestration.corrective_scheduler_bootstrap import (
    BOOTSTRAP_EXIT_RUNTIME_MISMATCH,
    run_scheduler_bootstrap,
)
from quant_platform.orchestration.corrective_scheduler_supervisor import (
    intended_scheduler_slot,
)
from quant_platform.orchestration.effect_authority import (
    PHASE00_REPAIR_PROFILE,
    EffectAuthority,
    publication_authority_session,
)
from quant_platform.wizard_credit_ledger import (
    DISCOVERY_LANE,
    PROOF_LANE,
    reconcile_wizard_credit_lane,
    reserve_wizard_credit_lane,
)
from tests.test_corrective_phase00_closure import _publish_fixture_closure
from tests.test_corrective_phase00_control import (
    _git_fixture,
    build_phase00_quiesced_checkpoint,
    start_phase00_maintenance,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HASH = "a" * 64
TARGET = "https://api.cryptowizards.net/v1/custom-series"
NOW = datetime(2026, 8, 22, 16, 0, tzinfo=UTC)


def test_p00_sl_004_child_cannot_publish_after_launcher_exit(
    tmp_path: Path,
) -> None:
    parent_target = tmp_path / "parent.json"
    child_target = tmp_path / "child.json"
    authority = EffectAuthority(
        root=tmp_path,
        secret=b"phase00-child-publication-authority-secret",
        issuer_id="phase00-child-diagnostic",
        profile=PHASE00_REPAIR_PROFILE,
    )
    inherited = effect_authority._CURRENT_PUBLICATION_AUTHORITY.set(None)
    try:
        with publication_authority_session(
            authority=authority,
            run_id="launcher-run",
            intended_slot_id="launcher-slot",
            policy_version="phase00.child.v1",
            source_fingerprint_sha256=HASH,
            runtime_fingerprint_sha256=HASH,
            configuration_fingerprint_sha256=HASH,
            allowed_scopes=frozenset({"research"}),
            allowed_target_prefixes=(tmp_path,),
            max_total_bytes=1024,
        ):
            atomic_write_text(
                parent_target,
                '{"publisher":"parent"}\n',
                publication_scope="research",
            )
    finally:
        effect_authority._CURRENT_PUBLICATION_AUTHORITY.reset(inherited)

    script = """from pathlib import Path
from quant_platform.orchestration.corrective_runtime import atomic_write_text
target = Path(__import__('sys').argv[1])
atomic_write_text(target, '{"publisher":"child"}\\n', publication_scope='research')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(child_target)],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert json.loads(parent_target.read_text(encoding="utf-8")) == {
        "publisher": "parent"
    }
    assert completed.returncode != 0
    assert "publication_authority_session_missing" in completed.stderr
    assert not child_target.exists()


def test_t00_001_source_freeze_manifest_is_repeatable_and_quiesced(
    tmp_path: Path,
) -> None:
    root = _git_fixture(tmp_path)
    start_phase00_maintenance(root=root, reason="supreme diagnostic", now=NOW)

    first = build_phase00_quiesced_checkpoint(root=root, now=NOW)
    second = build_phase00_quiesced_checkpoint(
        root=root,
        now=NOW + timedelta(seconds=1),
    )
    manifest = json.loads(
        second.paths["stable_manifest"].read_text(encoding="utf-8")
    )
    runtime = manifest["runtime_observation"]

    assert first.summary["snapshot_stable"] is True
    assert second.summary["snapshot_stable"] is True
    assert first.summary["first_manifest_sha256"] == first.summary[
        "second_manifest_sha256"
    ]
    assert second.summary["first_manifest_sha256"] == second.summary[
        "second_manifest_sha256"
    ]
    assert first.summary["freshness_manifest_sha256"] == second.summary[
        "freshness_manifest_sha256"
    ]
    assert runtime["processes"]["status"] == "PASS_NO_PRODUCERS"
    assert runtime["processes"]["producer_count"] == 0
    assert all(
        row["status"] == "PASS_UNLOADED" and row["loaded"] is False
        for row in runtime["launchd"]
    )


def test_t00_010_atomic_writer_concurrency_exact_10000(tmp_path: Path) -> None:
    target = tmp_path / "atomic-concurrency.json"

    def publish(generation: int) -> int:
        digest = sha256(str(generation).encode("ascii")).hexdigest()
        atomic_write_text(
            target,
            json.dumps(
                {"generation": generation, "generation_sha256": digest},
                sort_keys=True,
            )
            + "\n",
            publication_scope="research",
        )
        return generation

    with ThreadPoolExecutor(max_workers=48) as executor:
        futures = [
            executor.submit(copy_context().run, publish, generation)
            for generation in range(10_000)
        ]
        completed = [future.result() for future in futures]

    observed = json.loads(target.read_text(encoding="utf-8"))
    assert len(completed) == 10_000
    assert sorted(completed) == list(range(10_000))
    assert 0 <= observed["generation"] < 10_000
    assert observed["generation_sha256"] == sha256(
        str(observed["generation"]).encode("ascii")
    ).hexdigest()
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []


def test_t00_020_cadence_slo_replay() -> None:
    contract = scheduler_contract("daily_research")
    zone = ZoneInfo("America/New_York")
    expected: dict[str, datetime] = {}
    completions: list[dict[str, object]] = []
    for offset in range(7):
        local_due = datetime.combine(
            date(2026, 8, 10) + timedelta(days=offset),
            time(6, 15),
            tzinfo=zone,
        )
        expected_at = local_due.astimezone(UTC)
        slot = intended_scheduler_slot(
            contract,
            now=expected_at + timedelta(minutes=1),
        )
        expected[slot] = expected_at
        completions.append(
            {
                "intended_slot": slot,
                "completed_at_utc": (
                    expected_at + timedelta(minutes=2 + offset)
                ).isoformat(),
                "trigger_provenance": "launchd",
                "terminal_status": "PASS",
                "intended_slot_credit": True,
            }
        )
    completions.append(
        {
            "intended_slot": next(iter(expected)),
            "completed_at_utc": NOW.isoformat(),
            "trigger_provenance": "manual",
            "terminal_status": "PASS",
            "intended_slot_credit": True,
        }
    )

    passed = evaluate_cadence_slo(
        expected_slots=expected,
        completions=completions,
        maximum_delay_seconds=15 * 60,
        minimum_expected_slots=7,
    )
    first_slot, second_slot, *_, last_slot = expected
    attacked = [row for row in completions if row["intended_slot"] != last_slot]
    attacked.append(dict(attacked[0]))
    attacked[1] = {
        **attacked[1],
        "completed_at_utc": (expected[second_slot] + timedelta(hours=1)).isoformat(),
    }
    attacked.append(
        {
            "intended_slot": "unexpected-slot",
            "completed_at_utc": NOW.isoformat(),
            "trigger_provenance": "launchd",
            "terminal_status": "PASS",
            "intended_slot_credit": True,
        }
    )
    blocked = evaluate_cadence_slo(
        expected_slots=expected,
        completions=attacked,
        maximum_delay_seconds=15 * 60,
        minimum_expected_slots=7,
    )

    assert passed["status"] == "PASS_CADENCE_SLO"
    assert passed["availability"] == 1.0
    assert passed["ineligible_records"] == 1
    assert blocked["status"] == "BLOCKED_CADENCE_SLO"
    assert first_slot in blocked["duplicate_slots"]
    assert second_slot in blocked["delayed_slots"]
    assert last_slot in blocked["missing_slots"]
    assert blocked["unexplained_slots"] == ["unexpected-slot"]
    assert blocked["promotion_authority"] is False
    assert blocked["order_authority"] is False


def test_t00_021_calendar_boundary_matrix() -> None:
    daily = scheduler_contract("daily_research")
    interval = scheduler_contract("hyperliquid_l2")
    zone = ZoneInfo("America/New_York")
    boundary_windows = (
        [date(2026, 3, 6) + timedelta(days=index) for index in range(5)],
        [date(2026, 10, 30) + timedelta(days=index) for index in range(5)],
        [date(2028, 2, 27) + timedelta(days=index) for index in range(4)],
    )
    all_slots: set[str] = set()
    for window in boundary_windows:
        observed_utc: list[datetime] = []
        for local_date in window:
            local_now = datetime.combine(local_date, time(12, 0), tzinfo=zone)
            now = local_now.astimezone(UTC)
            slot = intended_scheduler_slot(daily, now=now)
            slot_time = datetime.fromisoformat(slot.rsplit("/", 1)[0])
            local_slot = slot_time.astimezone(zone)
            assert local_slot.date() == local_date
            assert (local_slot.hour, local_slot.minute) == (6, 15)
            assert slot_time <= now
            assert slot not in all_slots
            all_slots.add(slot)
            observed_utc.append(slot_time)
        assert all(
            later > earlier
            and (later - earlier).total_seconds() in {23 * 3600, 24 * 3600, 25 * 3600}
            for earlier, later in pairwise(observed_utc)
        )

    base = datetime(2026, 11, 1, 5, 58, tzinfo=UTC)
    wall_clock_sequence = (
        base,
        base + timedelta(minutes=4),
        base + timedelta(minutes=2),
        base + timedelta(minutes=7),
        base + timedelta(hours=2),
    )
    interval_slots = [
        intended_scheduler_slot(interval, now=observed)
        for observed in wall_clock_sequence
    ]
    for observed, slot in zip(wall_clock_sequence, interval_slots, strict=True):
        assert datetime.fromisoformat(slot.rsplit("/", 1)[0]) <= observed
    assert interval_slots[1] == interval_slots[2]
    assert len(set(interval_slots)) < len(interval_slots)


def test_t00_022_credit_reset_boundary(tmp_path: Path) -> None:
    before_reset = datetime(2026, 8, 22, 23, 59, 58, tzinfo=UTC)
    reservation = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        planned_credits=20,
        now=before_reset,
    )
    crash_retry = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        planned_credits=20,
        now=before_reset + timedelta(seconds=1),
    )
    reconciliation_kwargs = {
        "root": tmp_path,
        "lane": PROOF_LANE,
        "reservation_id": reservation.summary["reservation_id"],
        "reconciliation_key": "boundary-call",
        "attempted_credits": 4,
        "completed_credits": 2,
        "external_requests": 2,
        "observed_used_before": 100,
        "observed_used_after": 104,
        "activity_rows": [
            {
                "lane": "boundary_calls",
                "external_requests": 2,
                "credit_cost": 2,
                "attempted_credits": 4,
                "completed_credits": 2,
            }
        ],
        "now": before_reset + timedelta(seconds=1),
    }
    reconciled = reconcile_wizard_credit_lane(**reconciliation_kwargs)
    replayed = reconcile_wizard_credit_lane(**reconciliation_kwargs)
    next_day = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=PROOF_LANE,
        planned_credits=20,
        now=before_reset + timedelta(seconds=3),
    )

    assert reservation.summary["status"] == "PASS"
    assert crash_retry.summary["status"] == "REUSED"
    assert crash_retry.summary["reservation_id"] == reservation.summary[
        "reservation_id"
    ]
    assert crash_retry.summary["external_spend_authorized"] is False
    assert reconciled.summary["status"] == "PASS_RECONCILED"
    assert replayed.summary["status"] == "REUSED_RECONCILIATION"
    assert replayed.summary["reconciliation_id"] == reconciled.summary[
        "reconciliation_id"
    ]
    assert next_day.summary["status"] == "PASS"
    assert next_day.summary["reservation_id"] != reservation.summary[
        "reservation_id"
    ]
    assert len(list(reservation.paths["reservation"].parent.glob("*.json"))) == 1
    assert len(list(next_day.paths["reservation"].parent.glob("*.json"))) == 1


def test_t00_023_provider_failure_matrix(tmp_path: Path) -> None:
    scenarios = (
        ("http_429", "callback"),
        ("http_500", "callback"),
        ("timeout", "callback"),
        ("disconnect", "callback"),
        ("malformed", "recorder"),
        ("partial", "recorder"),
        ("replayed_response", "recorder"),
    )
    for scenario, failure_stage in scenarios:
        root = tmp_path / scenario
        authority = EffectAuthority(
            root=root,
            secret=b"provider-failure-matrix-authority-secret",
            issuer_id=f"provider-matrix-{scenario}",
            profile=RESEARCH_EXTERNAL_EFFECT_PROFILE,
        )
        callback_calls = 0

        def callback(
            *,
            _failure_stage: str = failure_stage,
            _scenario: str = scenario,
        ) -> dict[str, str]:
            nonlocal callback_calls
            callback_calls += 1
            if _failure_stage == "callback":
                raise RuntimeError(f"provider_{_scenario}")
            return {"scenario": _scenario}

        def recorder(_result: object, *, _scenario: str = scenario) -> str:
            raise ValueError(f"evidence_{_scenario}_not_complete")

        with external_effect_authority_session(
            authority=authority,
            run_id=f"run-{scenario}",
            intended_slot_id=f"slot-{scenario}",
            source_fingerprint_sha256=HASH,
            runtime_fingerprint_sha256=HASH,
            configuration_fingerprint_sha256=HASH,
            provider_id="crypto_wizards",
            account_scope_id="wizard-research-test",
            reservation_id=f"reservation-{scenario}",
            reservation_sha256=sha256(scenario.encode("ascii")).hexdigest(),
            allowed_targets=frozenset({TARGET}),
            allowed_credential_keys=frozenset({"CRYPTO_WIZARDS_API_KEY"}),
            max_total_requests=1,
            max_total_credits=2,
        ) as session:
            read_authorized_credential(
                "CRYPTO_WIZARDS_API_KEY",
                reader=lambda _key: "fixture-secret",
            )
            with pytest.raises((RuntimeError, ValueError)):
                run_authorized_credit_call(
                    target=TARGET,
                    operation="POST",
                    method="POST",
                    request_payload=b"{}",
                    request_count=1,
                    credit_cost=2,
                    callback=callback,
                    result_recorder=recorder,
                )
        accounting = authority.run_accounting(
            run_id=f"run-{scenario}",
            intended_slot_id=f"slot-{scenario}",
        )
        assert callback_calls == 1
        assert accounting["unknown_effects"] == 2
        assert accounting["accounting_complete"] is False
        assert session.consumed_requests == 1
        assert session.consumed_credits == 2


def test_t00_026_warning_policy_is_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning = subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            "-c",
            "import warnings; warnings.warn('unclassified', RuntimeWarning)",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    clean = subprocess.run(
        [sys.executable, "-W", "error", "-c", "print('clean')"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    python = tmp_path / ".venv/bin/python"
    ruff = tmp_path / ".venv/bin/ruff"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    ruff.write_text("", encoding="utf-8")
    source = tmp_path / "src/fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    commands: list[list[str]] = []

    def contained(command):
        return list(command), {
            "status": "ENFORCED",
            "mechanism": "fixture",
            "network_denied": True,
            "keychain_security_exec_denied": True,
        }

    def runner(command, **_kwargs):
        command = list(command)
        commands.append(command)
        for argument in command:
            if str(argument).startswith("--junitxml="):
                Path(str(argument).split("=", 1)[1]).write_text(
                    '<testsuite tests="1" failures="0" errors="0" skipped="0" />',
                    encoding="utf-8",
                )
        return subprocess.CompletedProcess(command, 0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr(closure, "_contained_pytest_command", contained)
    monkeypatch.setattr(closure, "_phase00_ruff_scope", lambda _root: ["src/fixture.py"])
    result = closure._run_verification(
        tmp_path,
        runner=runner,
        timeout_seconds=30,
    )
    pytest_command = next(command for command in commands if "pytest" in command)

    assert warning.returncode != 0
    assert "RuntimeWarning" in warning.stderr
    assert clean.returncode == 0
    assert "-W" in pytest_command
    assert pytest_command[pytest_command.index("-W") + 1] == "error"
    assert result["blockers"] == []


def test_t00_027_wrong_interpreter_fails_before_numerical_imports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = tmp_path / ".venv/bin/python3"
    expected.parent.mkdir(parents=True)
    expected.write_text("", encoding="utf-8")
    (tmp_path / ".venv/pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    (tmp_path / "src/quant_platform").mkdir(parents=True)
    module_imported = False

    def module_runner(*_args, **_kwargs):
        nonlocal module_imported
        module_imported = True
        return {}

    exit_code = run_scheduler_bootstrap(
        [str(tmp_path), "daily_research"],
        observed_executable=tmp_path / "wrong/python",
        module_runner=module_runner,
    )
    stderr = json.loads(capsys.readouterr().err)

    assert exit_code == BOOTSTRAP_EXIT_RUNTIME_MISMATCH
    assert module_imported is False
    assert stderr["status"] == "BLOCKED_BEFORE_SCHEDULER_IMPORT"
    assert stderr["blockers"] == ["wrong_scheduler_interpreter"]
    assert stderr["repair_command"] == "uv sync --frozen"
    assert stderr["live_trading_authorized"] is False


def test_t00_028_schema_fuzz_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = {
        "duplicate": '{"status":"PASS","status":"BLOCKED"}\n',
        "nan": '{"value":NaN}\n',
        "infinity": '{"value":Infinity}\n',
        "array_root": "[]\n",
    }
    for name, payload in malformed.items():
        path = tmp_path / f"{name}.json"
        path.write_text(payload, encoding="utf-8")
        with pytest.raises((TypeError, ValueError)) as error:
            descendants._load_strict_object(path)
        assert len(str(error.value)) < 256

    huge = tmp_path / "huge.json"
    with huge.open("wb") as handle:
        handle.truncate(descendants.MAX_SOURCE_BYTES + 1)
    with pytest.raises(ValueError, match="source_too_large"):
        descendants._load_strict_object(huge)

    original = tmp_path / "original.json"
    original.write_text("{}\n", encoding="utf-8")
    hardlink = tmp_path / "hardlink.json"
    os.link(original, hardlink)
    with pytest.raises(ValueError, match="source_not_single_regular_file"):
        descendants._load_strict_object(hardlink)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(original)
    with pytest.raises(ValueError, match="source_not_single_regular_file"):
        descendants._load_strict_object(symlink)
    with pytest.raises(ValueError, match="unsafe_relative_path"):
        descendants._safe_relative_path(tmp_path, "../../outside.json")

    permissive_candidate = tmp_path / "control.json"
    permissive_candidate.write_text(malformed["duplicate"], encoding="utf-8")
    assert phase00_control._read_json(permissive_candidate) == {}
    active_huge = tmp_path / "active_huge.json"
    with active_huge.open("wb") as handle:
        handle.truncate(envelopes.MAX_ACTIVE_ARTIFACT_BYTES + 1)
    with pytest.raises(ValueError, match="size limit"):
        envelopes._load_strict_object(active_huge, require_single_link=False)

    closure_root = tmp_path / "closure"
    closure_root.mkdir()
    result = _publish_fixture_closure(closure_root, monkeypatch)
    pointer = json.loads(result.paths["active_closure"].read_text(encoding="utf-8"))
    pointer["status"] = "UNKNOWN_ENUM"
    result.paths["active_closure"].write_text(
        json.dumps(pointer, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    observation = closure.observe_phase00_closure(
        closure_root,
        verify_current_tree=False,
    )
    assert observation["validation_status"] == "BLOCKED"
    assert any("status_invalid" in blocker for blocker in observation["blockers"])
    assert observation["runtime_activation_authorized"] is False


def test_t00_029_idempotent_replay(tmp_path: Path) -> None:
    reservation = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=DISCOVERY_LANE,
        planned_credits=50,
        now=NOW,
    )
    reservation_replay = reserve_wizard_credit_lane(
        root=tmp_path,
        lane=DISCOVERY_LANE,
        planned_credits=50,
        now=NOW + timedelta(minutes=1),
    )
    arguments = {
        "root": tmp_path,
        "lane": DISCOVERY_LANE,
        "reservation_id": reservation.summary["reservation_id"],
        "reconciliation_key": "same-source-slot",
        "attempted_credits": 0,
        "completed_credits": 0,
        "external_requests": 0,
        "now": NOW,
    }
    reconciliation = reconcile_wizard_credit_lane(**arguments)
    reconciliation_replay = reconcile_wizard_credit_lane(**arguments)
    mismatch = reconcile_wizard_credit_lane(
        **{
            **arguments,
                "attempted_credits": 1,
                "completed_credits": 1,
                "external_requests": 1,
                "observed_used_before": 0,
                "observed_used_after": 1,
                "activity_rows": [
                {
                    "lane": "replay_mismatch",
                    "external_requests": 1,
                    "credit_cost": 1,
                    "attempted_credits": 1,
                    "completed_credits": 1,
                }
            ],
        }
    )

    assert reservation_replay.summary["status"] == "REUSED"
    assert reservation_replay.summary["reservation_id"] == reservation.summary[
        "reservation_id"
    ]
    assert reconciliation_replay.summary["status"] == "REUSED_RECONCILIATION"
    assert reconciliation_replay.summary["reconciliation_id"] == reconciliation.summary[
        "reconciliation_id"
    ]
    assert mismatch.summary["status"] == "BLOCKED"
    assert mismatch.summary["blocker"] == "existing_reconciliation_contract_mismatch"
    assert len(list(reservation.paths["reservation"].parent.glob("*.json"))) == 1
    assert len(list(reconciliation.paths["reconciliation"].parent.glob("*.json"))) == 1
    assert reservation.summary["testnet_order_authority"] is False
    assert reconciliation.summary["live_trading_authorized"] is False


def test_t00_031_one_year_resource_retention_simulation(tmp_path: Path) -> None:
    policy = tmp_path / "config/corrective_artifact_retention.json"
    policy.parent.mkdir(parents=True)
    policy.write_text(
        json.dumps(
            {
                "schema_version": "thewiz.corrective_artifact_retention_policy.v1",
                "operational_receipt_directories": ["data/research/operational"],
                "receipt_retention_days": 14,
                "minimum_newest_per_directory": 1,
                "log_paths": ["reports/active/schedule_logs/*.log"],
                "log_rotation_threshold_bytes": 1024,
                "log_rotations_to_keep": 2,
                "never_archive_directories": ["data/research/scientific"],
            }
        ),
        encoding="utf-8",
    )
    start = datetime(2025, 8, 16, tzinfo=UTC)
    paths: list[Path] = []
    for index in range(365):
        observed = start + timedelta(days=index)
        path = tmp_path / "data/research/operational" / f"receipt_{index:03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"day": index}) + "\n", encoding="utf-8")
        os.utime(path, (observed.timestamp(), observed.timestamp()))
        paths.append(path)
    referenced = paths[::30]
    active = tmp_path / "reports/active/retained_lineage.json"
    active.parent.mkdir(parents=True)
    active.write_text(
        json.dumps(
            {
                "evidence_paths": [
                    path.relative_to(tmp_path).as_posix() for path in referenced
                ]
            }
        ),
        encoding="utf-8",
    )

    result = run_corrective_artifact_retention(
        root=tmp_path,
        now=datetime(2026, 8, 16, tzinfo=UTC),
        apply=False,
    )
    rows = {row["source_path"]: row for row in result.summary["receipt_rows"]}

    assert result.summary["status"] == "PASS_RETENTION_DRY_RUN"
    assert result.summary["files_scanned"] == 365
    assert result.summary["archive_candidates"] > 330
    assert result.summary["retention_horizon_days"] == 14
    assert result.summary["projected_reclaimable_bytes"] > 0
    assert result.summary["projected_retained_bytes"] > 0
    assert result.summary["capacity_forecast_status"] == "PASS_POLICY_BOUNDED"
    assert result.summary["dry_run_moved_nothing"] is True
    assert all(path.exists() for path in paths)
    assert all(
        rows[path.relative_to(tmp_path).as_posix()]["eligible"] is False
        for path in referenced
    )
    assert result.summary["scientific_evidence_directories_touched"] == []


def test_t00_032_bounded_state_machine_model_check() -> None:
    result = exhaustive_run_control_model(maximum_depth=8)

    assert result["status"] == "PASS_BOUNDED_MODEL"
    assert result["reachable_states"] >= 15
    assert result["valid_transitions"] > 0
    assert result["rejected_interleavings"] > result["valid_transitions"]
    assert result["violations"] == []
    assert result["promotion_authority"] is False
    assert result["order_authority"] is False


def test_t00_033_active_pointer_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _publish_fixture_closure(tmp_path, monkeypatch)
    pointer_path = result.paths["active_closure"]
    pristine = pointer_path.read_bytes()
    pointer = json.loads(pristine)

    assert closure.observe_phase00_closure(
        tmp_path,
        verify_current_tree=False,
    )["validation_status"] == "PASS"
    attacks = []
    dangling = dict(pointer)
    dangling["closure_id"] = "phase00closure_" + "f" * 24
    attacks.append(dangling)
    cross_run = dict(pointer)
    cross_run["closure_receipt_path"] = (
        "data/research/phase00_control/closures/"
        "phase00closure_aaaaaaaaaaaaaaaaaaaaaaaa/closure_receipt.json"
    )
    attacks.append(cross_run)
    wrong_hash = dict(pointer)
    wrong_hash["closure_receipt_sha256"] = "0" * 64
    attacks.append(wrong_hash)

    for attacked in attacks:
        pointer_path.write_text(
            json.dumps(attacked, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        observation = closure.observe_phase00_closure(
            tmp_path,
            verify_current_tree=False,
        )
        assert observation["validation_status"] == "BLOCKED"
        assert observation["runtime_activation_authorized"] is False
        pointer_path.write_bytes(pristine)

    assert closure.observe_phase00_closure(
        tmp_path,
        verify_current_tree=False,
    )["validation_status"] == "PASS"
