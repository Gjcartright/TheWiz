"""Verify the source decision for unadopted strategy desk documents."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
RUNTIME = Path("/Users/gregc/TheWiz-LocalRuntime")
SAVEPOINT = Path("/Users/gregc/Backups/TheWiz/savepoints/2026-09-30/local_runtime")
OUTPUT = AUDIT / "GATE0_STRATEGY_DESK_ARCHIVE_DECISION_2026-09-30.json"
STATUS = "REVIEWED_UNADOPTED_STRATEGY_DESK_ARCHIVE_NO_PORT"
INDEX = "config/strategy_desk_index.json"
PROFILE_STEMS = (
    "copula_research_20260909",
    "dynamic_spread_research_20260909",
    "dynamic_zscore_research_20260909",
    "static_spread_research_20260909",
    "static_zscore_research_20260909",
)
REVIEWS = (
    "config/strategy_desk_reviews/ou_spread_retired_20260909.json",
    "config/strategy_desk_reviews/ou_zscore_retired_20260909.json",
)
PROFILES = tuple(f"config/strategy_desks/{stem}.json" for stem in PROFILE_STEMS)
MANIFESTS = tuple(
    f"config/strategy_desks/{stem}.manifest.json" for stem in PROFILE_STEMS
)
PATHS = (INDEX, *PROFILES, *MANIFESTS, *REVIEWS)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_false_fields(payload: dict, fields: tuple[str, ...], context: str) -> None:
    for field in fields:
        if payload.get(field) is not False:
            raise ValueError(f"{context} has unexpected authority: {field}")


def main() -> None:
    with (AUDIT / "GATE0_UNION_SOURCE_QUEUE_2026-09-30.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = {row["relative_path"]: row for row in csv.DictReader(stream)}
    manifest = json.loads(
        (SAVEPOINT.parent / "SAVEPOINT_MANIFEST.json").read_text(encoding="utf-8")
    )
    saved_hashes = {
        entry["path"]: entry["sha256"]
        for entry in manifest["entries"]
        if entry["type"] == "file"
    }
    path_hashes = {}
    for relative in PATHS:
        row = rows[relative]
        if row["custody_status"] != STATUS or row["working_vs_runtime"] != "RUNTIME_ONLY":
            raise ValueError(f"unreviewed strategy desk path: {relative}")
        if (ROOT / relative).exists():
            raise ValueError(f"strategy desk candidate became active: {relative}")
        candidate, saved = RUNTIME / relative, SAVEPOINT / relative
        candidate_hash, saved_hash = digest(candidate), digest(saved)
        if candidate_hash != row["runtime_sha256_at_freeze"]:
            raise ValueError(f"runtime strategy desk source drift: {relative}")
        if candidate_hash != saved_hash:
            raise ValueError(f"Mac savepoint strategy desk byte drift: {relative}")
        if saved_hashes.get(f"local_runtime/{relative}") != saved_hash:
            raise ValueError(f"Mac savepoint manifest drift: {relative}")
        path_hashes[relative] = candidate_hash

    index = json.loads((RUNTIME / INDEX).read_text(encoding="utf-8"))
    if index.get("schema_version") != "thewiz.strategy_desk_organization.v1":
        raise ValueError("strategy desk index schema changed")
    if index.get("document_kind") != "non_executable_desk_organization":
        raise ValueError("strategy desk index became executable")
    if index.get("runtime_consumer") is not None:
        raise ValueError("strategy desk index gained a runtime consumer")
    require_false_fields(
        index,
        (
            "dashboard_implemented",
            "active_strategy_configuration_changed",
            "live_trading_authorized",
            "gate_completion_claim",
        ),
        INDEX,
    )
    if len(index.get("desks", [])) != 7:
        raise ValueError("seven-mode organization changed")

    profiles = []
    if len(PROFILES) != len(MANIFESTS):
        raise ValueError("strategy profile/manifest count mismatch")
    for profile_path, manifest_path in zip(PROFILES, MANIFESTS):
        profile = json.loads((RUNTIME / profile_path).read_text(encoding="utf-8"))
        binding = json.loads((RUNTIME / manifest_path).read_text(encoding="utf-8"))
        if profile.get("scope") != "offline_research_scenario":
            raise ValueError(f"research profile scope changed: {profile_path}")
        require_false_fields(
            profile,
            ("live_trading_authorized", "paper_trading_authorized", "workflow_qualified"),
            profile_path,
        )
        require_false_fields(
            binding,
            (
                "account_adopted",
                "workflow_qualified",
                "paper_trading_authorized",
                "live_trading_authorized",
            ),
            manifest_path,
        )
        if binding.get("profile_path") != profile_path:
            raise ValueError(f"strategy profile manifest path mismatch: {manifest_path}")
        if binding.get("profile_sha256") != path_hashes[profile_path]:
            raise ValueError(f"strategy profile manifest hash mismatch: {manifest_path}")
        if binding.get("profile_id") != profile.get("profile_id"):
            raise ValueError(f"strategy profile ID mismatch: {manifest_path}")
        profiles.append({
            "path": profile_path,
            "profile_id": profile["profile_id"],
            "desk_id": profile["desk_id"],
            "scope": profile["scope"],
            "manifest_path": manifest_path,
            "profile_sha256": path_hashes[profile_path],
        })
    if len({item["desk_id"] for item in profiles}) != len(profiles):
        raise ValueError("duplicate strategy desk ID")

    retired_reviews = []
    for relative in REVIEWS:
        review = json.loads((RUNTIME / relative).read_text(encoding="utf-8"))
        if review.get("review_status") != "ARCHIVE_REVIEW_COMPLETE_NOT_RUNTIME_CONFIGURATION":
            raise ValueError(f"retired OU review status changed: {relative}")
        if review.get("runtime_consumer") is not None:
            raise ValueError(f"retired OU review gained runtime consumer: {relative}")
        if review.get("terminal_disposition", {}).get("status") != "CLOSED_TERMINAL_FAILURE":
            raise ValueError(f"retired OU failure disposition changed: {relative}")
        authority = review.get("authority", {})
        if not authority or any(value is not False for value in authority.values()):
            raise ValueError(f"retired OU review gained authority: {relative}")
        retired_reviews.append({
            "path": relative,
            "review_id": review["review_id"],
            "terminal_status": review["terminal_disposition"]["status"],
        })

    summary = {
        "schema_version": "thewiz.gate0.strategy_desk_archive_decision.v1",
        "decision": STATUS,
        "path_sha256": path_hashes,
        "index_document_kind": index["document_kind"],
        "index_runtime_consumer": index["runtime_consumer"],
        "organized_exact_modes": len(index["desks"]),
        "research_profiles": profiles,
        "retired_ou_reviews": retired_reviews,
        "interpretation": "Retain all 13 exact files as preserved, unadopted research and archive records. Profile numbers are scenario inputs, not active risk or trading policy. Five profile manifests bind to exact profile bytes; both retired OU records retain terminal failure and no authority. Review against current data, math, risk, and execution contracts before any later adoption.",
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS thirteen_unadopted_strategy_desk_paths")


if __name__ == "__main__":
    main()
