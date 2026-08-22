#!/usr/bin/env python3
"""Build and restore-test a redacted recovery checkpoint for the current worktree."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "reports" / "active"
BASELINE_MANIFEST = ACTIVE / "corrective_baseline_manifest.json"
BASELINE_FILES = ACTIVE / "corrective_baseline_files.csv"
SECRET_SCAN = ACTIVE / "corrective_secret_scan.csv"
FORBIDDEN_SOURCE_PREFIXES = (
    ".git/",
    ".venv",
    "apps/the-ave/",
    "archive/",
    "data/",
    "models/",
    "reports/",
    "runs/",
    "work/",
)
FORBIDDEN_SOURCE_FILES = {".env", ".env.local"}
EVIDENCE_FILES = (
    "reports/active/seven_stage_goal_checkpoint.csv",
    "reports/active/seven_stage_goal_checkpoint.md",
    "reports/active/model_authority_status.json",
    "reports/active/current_state.csv",
    "reports/active/current_state.md",
    "reports/active/system_check.csv",
    "reports/active/system_check.md",
    "reports/active/artifact_index.csv",
    "reports/active/quant_release_index.csv",
    "reports/active/quant_release_index.md",
    "reports/active/evidence_registry.csv",
    "reports/active/evidence_registry.md",
    "reports/active/evidence_source_health.csv",
    "reports/active/canonical_wizard_hyperliquid_contract.json",
    "reports/active/canonical_wizard_hyperliquid_contract.md",
    "reports/active/canonical_wizard_hyperliquid_stages.csv",
    "reports/research/udemy_transcript_vault_manifest.csv",
    "reports/research/udemy_transcript_capture_queue.csv",
    "reports/active/corrective_plan_completion.json",
    "reports/active/corrective_plan_completion.md",
    "reports/active/corrective_plan_phase_status.csv",
    "reports/active/corrective_l2_capture_status.json",
    "reports/active/corrective_wizard_proof_launcher_status.json",
    "reports/active/scheduler_runtime_readiness.json",
    "reports/active/wizard_reset_readiness.json",
    "reports/active/wizard_reset_readiness_checks.csv",
    "reports/active/stage4_handoff_readiness.json",
    "reports/active/stage4_handoff_readiness_checks.csv",
    "reports/active/full_project_self_inspection_2026-08-11.md",
    "reports/active/full_project_self_inspection_2026-08-11.csv",
    "reports/active/corrective_baseline_manifest.json",
    "reports/active/corrective_baseline_files.csv",
    "reports/active/corrective_secret_scan.csv",
    "reports/active/corrective_recovery_runbook.md",
)
SMOKE_TESTS = (
    "tests/test_active_pipeline.py::test_artifact_index_classifies_without_moving_files",
    "tests/test_wizard_research_journal.py::test_detail_records_preserve_top_metrics_separately_from_detail_metrics",
    "tests/test_ml_filter.py::test_shadow_trade_filter_predictions_rejects_tampered_model_before_deserialization",
    "tests/test_corrective_wizard_reset_readiness.py::test_reset_readiness_passes_without_granting_authority",
    "tests/test_udemy_transcript_vault.py",
    "tests/test_statistical_validation.py",
    "tests/test_evidence_registry.py",
    "tests/test_canonical_wizard_hyperliquid_contract.py",
)


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(*args: str, cwd: Path = ROOT, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _load_source_rows() -> list[dict[str, str]]:
    if not BASELINE_MANIFEST.is_file() or not BASELINE_FILES.is_file() or not SECRET_SCAN.is_file():
        raise ValueError("run scripts/build_corrective_checkpoint.py before recovery packaging")
    baseline = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    if int(baseline.get("secret_blockers", -1)) != 0:
        raise ValueError("baseline secret scan has blocking findings")
    with BASELINE_FILES.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("baseline source manifest is empty")
    for row in rows:
        relative = str(row.get("path", "")).strip()
        if (
            not relative
            or relative in FORBIDDEN_SOURCE_FILES
            or relative.startswith(FORBIDDEN_SOURCE_PREFIXES)
        ):
            raise ValueError(f"forbidden recovery source path: {relative}")
        source = ROOT / relative
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"recovery source is not a regular file: {relative}")
        if _hash(source) != str(row.get("sha256", "")):
            raise ValueError(f"recovery source changed after baseline manifest: {relative}")
    return rows


def _write_tar(path: Path, relative_paths: list[str]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for relative in sorted(relative_paths):
            source = ROOT / relative
            if source.is_file() and not source.is_symlink():
                archive.add(source, arcname=relative, recursive=False)


def _verify_restored_sources(restore_root: Path, rows: list[dict[str, str]]) -> None:
    for row in rows:
        relative = str(row["path"])
        restored = restore_root / relative
        if not restored.is_file() or _hash(restored) != str(row["sha256"]):
            raise ValueError(f"restored source hash mismatch: {relative}")


def _restore_drill(
    *,
    bundle_path: Path,
    source_snapshot: Path,
    rows: list[dict[str, str]],
    branch: str,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="thewiz-current-restore-") as directory:
        restore_root = Path(directory) / "repo"
        bundle_verify = _run("git", "bundle", "verify", str(bundle_path))
        _run("git", "clone", "--quiet", "--branch", branch, str(bundle_path), str(restore_root))
        with tarfile.open(source_snapshot, "r:gz") as archive:
            archive.extractall(restore_root, filter="data")
        _verify_restored_sources(restore_root, rows)
        compile_result = _run(
            str(ROOT / ".venv312" / "bin" / "python"),
            "-m",
            "compileall",
            "-q",
            "src",
            cwd=restore_root,
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        smoke = _run(
            str(ROOT / ".venv312" / "bin" / "python"),
            "-m",
            "pytest",
            "-q",
            *SMOKE_TESTS,
            cwd=restore_root,
            env=env,
        )
        return {
            "status": "PASS",
            "bundle_verified": True,
            "source_files_verified": len(rows),
            "compileall_passed": compile_result.returncode == 0,
            "smoke_tests": list(SMOKE_TESTS),
            "smoke_test_output": smoke.stdout.strip(),
            "bundle_verify_output": bundle_verify.stdout.strip().replace(
                str(bundle_path), bundle_path.name
            ),
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }


def build_checkpoint(destination: Path) -> Path:
    rows = _load_source_rows()
    now = datetime.now(timezone.utc)
    head = _run("git", "rev-parse", "HEAD").stdout.strip()
    branch = _run("git", "branch", "--show-current").stdout.strip()
    checkpoint_id = f"checkpoint-{head[:7]}-worktree-{now.strftime('%Y%m%dT%H%M%SZ')}"
    destination.mkdir(parents=True, exist_ok=True)
    final_dir = destination / checkpoint_id
    if final_dir.exists():
        raise FileExistsError(f"recovery checkpoint already exists: {final_dir}")

    staging = destination / f".{checkpoint_id}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        bundle_path = staging / "TheWiz-committed.bundle"
        source_snapshot = staging / "current_source_snapshot.tar.gz"
        evidence_snapshot = staging / "current_evidence_snapshot.tar.gz"
        source_manifest = staging / "source_manifest.csv"
        secret_scan = staging / "secret_scan.csv"
        baseline_manifest = staging / "baseline_manifest.json"

        _run("git", "bundle", "create", str(bundle_path), branch)
        _write_tar(source_snapshot, [str(row["path"]) for row in rows])
        _write_tar(evidence_snapshot, [path for path in EVIDENCE_FILES if (ROOT / path).is_file()])
        shutil.copy2(BASELINE_FILES, source_manifest)
        shutil.copy2(SECRET_SCAN, secret_scan)
        shutil.copy2(BASELINE_MANIFEST, baseline_manifest)

        restore = _restore_drill(
            bundle_path=bundle_path,
            source_snapshot=source_snapshot,
            rows=rows,
            branch=branch,
        )
        restore_path = staging / "restore_drill.json"
        restore_path.write_text(json.dumps(restore, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        checkpoint = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
        recovery_manifest = {
            "schema_version": "thewiz.current_recovery_checkpoint.v1",
            "checkpoint_id": checkpoint_id,
            "created_at_utc": now.isoformat(),
            "git_head": head,
            "git_branch": branch,
            "worktree_baseline_id": checkpoint.get("run_id", ""),
            "source_files": len(rows),
            "secret_blockers": 0,
            "git_bundle_sha256": _hash(bundle_path),
            "source_snapshot_sha256": _hash(source_snapshot),
            "evidence_snapshot_sha256": _hash(evidence_snapshot),
            "source_manifest_sha256": _hash(source_manifest),
            "restore_drill_sha256": _hash(restore_path),
            "restore_gate": restore["status"],
            "research_only": True,
            "orders_submitted": 0,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        manifest_path = staging / "recovery_manifest.json"
        manifest_path.write_text(
            json.dumps(recovery_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        artifact_rows = []
        for artifact in sorted(staging.iterdir()):
            if artifact.is_file():
                artifact_rows.append(
                    {
                        "path": artifact.name,
                        "size_bytes": artifact.stat().st_size,
                        "sha256": _hash(artifact),
                    }
                )
        with (staging / "recovery_files.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256"])
            writer.writeheader()
            writer.writerows(artifact_rows)
        staging.replace(final_dir)
        latest_link = destination / "latest-current"
        pending_link = destination / ".latest-current.pending"
        pending_link.unlink(missing_ok=True)
        pending_link.symlink_to(final_dir.name, target_is_directory=True)
        pending_link.replace(latest_link)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = build_checkpoint(args.destination.expanduser().resolve())
    print(json.dumps({"status": "PASS", "checkpoint": str(checkpoint)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
