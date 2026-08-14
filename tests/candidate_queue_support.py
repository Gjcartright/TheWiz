from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from quant_platform.orchestration.corrective_release_gates import (
    QUEUE_RECEIPT_SCHEMA_VERSION,
    QUEUE_ROW_SCHEMA_VERSION,
    _payload_hash,
    _seal_testnet_candidate_receipt,
    _write_immutable_json,
)


def seal_candidate_with_valid_queue(
    *,
    root: Path,
    receipt: dict[str, Any],
    support_candidates: tuple[dict[str, str], ...] | None = None,
) -> dict[str, Any]:
    candidate = dict(receipt)
    candidate.setdefault(
        "registered_semantic_hypothesis_id", "fixture-selected-hypothesis"
    )
    candidate.setdefault(
        "registered_stage4_contract_id", "fixture-stage4-contract"
    )
    candidate.setdefault("registered_stage4_conclusion_sha256", "c" * 64)
    candidate.setdefault("immutable_stage4_candidate_identity_ready", True)
    support_candidates = support_candidates or (
        {
            "experiment_id": "fixture-support-1",
            "pair": "SOL-USD-DOGE-USD",
            "asset_x": "SOL",
            "asset_y": "DOGE",
        },
        {
            "experiment_id": "fixture-support-2",
            "pair": "XRP-USD-ADA-USD",
            "asset_x": "XRP",
            "asset_y": "ADA",
        },
    )
    selected = _queue_row(
        experiment_id=str(candidate["candidate_experiment_id"]),
        semantic_hypothesis_id=str(
            candidate["registered_semantic_hypothesis_id"]
        ),
        stage4_contract_id=str(candidate["registered_stage4_contract_id"]),
        pair_group_key=str(candidate["pair_group_key"]),
        pair=str(candidate["pair"]),
        asset_x=str(candidate["asset_x"]),
        asset_y=str(candidate["asset_y"]),
        timeframe=str(candidate["timeframe"]),
        exact_mode=str(candidate["exact_mode"]),
        orientation=str(candidate["orientation"]),
        cost_model_id=str(candidate["cost_model_id"]),
        samples=0,
        rank=1,
        selected=True,
    )
    rows = [selected]
    for rank, support in enumerate(support_candidates, start=2):
        experiment_id = support["experiment_id"]
        pair = support["pair"]
        asset_x = support["asset_x"]
        asset_y = support["asset_y"]
        rows.append(
            _queue_row(
                experiment_id=experiment_id,
                semantic_hypothesis_id=f"fixture-support-hypothesis-{rank}",
                stage4_contract_id=str(candidate["registered_stage4_contract_id"]),
                pair_group_key=f"hyperliquid|daily|{asset_x}|{asset_y}",
                pair=pair,
                asset_x=asset_x,
                asset_y=asset_y,
                timeframe=str(candidate["timeframe"]),
                exact_mode=str(candidate["exact_mode"]),
                orientation="original",
                cost_model_id=f"fixture-cost-{rank}",
                samples=rank - 1,
                rank=rank,
                selected=False,
            )
        )
    source_hashes: dict[str, str] = {}
    source_artifacts = dict(candidate.get("source_artifact_hashes", {}))
    if not source_artifacts:
        source_artifacts = {
            "data/research/fixture_stage5_receipt.json": "fixture"
        }
    for relative in source_artifacts:
        source_path = root / str(relative)
        if not source_path.is_file():
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(
                f"fixture candidate source: {relative}\n",
                encoding="utf-8",
            )
        source_hashes[str(relative)] = sha256(source_path.read_bytes()).hexdigest()
    learning_relative = str(
        candidate.get("registered_learning_receipt_path", "")
    ).strip()
    if not learning_relative or learning_relative not in source_hashes:
        learning_relative = next(iter(source_hashes))
    candidate["registered_learning_id"] = str(
        candidate.get("registered_learning_id", "fixture-learning")
    )
    candidate["registered_learning_receipt_path"] = learning_relative
    candidate["registered_learning_receipt_sha256"] = source_hashes[
        learning_relative
    ]
    candidate.setdefault("registered_stage5_protocol_id", "fixture-stage5-protocol")
    candidate.setdefault("registered_stage5_protocol_sha256", "d" * 64)
    candidate.setdefault("registered_execution_id", "fixture-stage4-execution")
    queue_core = {
        "schema_version": QUEUE_RECEIPT_SCHEMA_VERSION,
        "registered_stage4_contract_id": str(
            candidate["registered_stage4_contract_id"]
        ),
        "registered_stage4_conclusion_sha256": str(
            candidate["registered_stage4_conclusion_sha256"]
        ),
        "immutable_stage4_candidate_identity_ready": True,
        "survivor_receipt_id": str(candidate["survivor_receipt_id"]),
        "model_training_dataset_id": str(candidate["model_training_dataset_id"]),
        "model_artifact_sha256": str(candidate["model_artifact_sha256"]),
        "registered_learning_id": str(candidate["registered_learning_id"]),
        "registered_learning_receipt_path": str(
            candidate["registered_learning_receipt_path"]
        ),
        "registered_learning_receipt_sha256": str(
            candidate["registered_learning_receipt_sha256"]
        ),
        "registered_stage5_protocol_id": str(
            candidate["registered_stage5_protocol_id"]
        ),
        "registered_stage5_protocol_sha256": str(
            candidate["registered_stage5_protocol_sha256"]
        ),
        "registered_execution_id": str(candidate["registered_execution_id"]),
        "testnet_sample_policy_id": "fixture-policy-three-independent-pairs",
        "required_candidate_pairs": 3,
        "ready_candidate_pairs": 3,
        "rows": rows,
        "global_blockers": [],
        "candidate_selection_ready": True,
        "source_artifact_hashes": source_hashes,
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    queue_id = "testnetcandidatequeue_" + _payload_hash(queue_core)[:20]
    queue = {**queue_core, "candidate_queue_id": queue_id}
    queue["receipt_sha256"] = _payload_hash(queue)
    relative = f"data/testnet/candidate_queues/{queue_id}.json"
    path = root / relative
    _write_immutable_json(queue, path)
    queue_sha256 = sha256(path.read_bytes()).hexdigest()
    candidate.update(
        {
            "candidate_queue_id": queue_id,
            "candidate_queue_path": relative,
            "candidate_queue_sha256": queue_sha256,
            "candidate_queue_selection_rank": 1,
            "validated_current_model_candidate_samples": 0,
            "validated_current_model_pair_samples": 0,
            "ready_candidate_pairs": 3,
            "required_candidate_pairs": 3,
        }
    )
    source_hashes[relative] = queue_sha256
    candidate["source_artifact_hashes"] = source_hashes
    candidate["evidence_path"] = ";".join(source_hashes)
    return _seal_testnet_candidate_receipt(receipt=candidate, root=root)


def _queue_row(
    *,
    experiment_id: str,
    semantic_hypothesis_id: str,
    stage4_contract_id: str,
    pair_group_key: str,
    pair: str,
    asset_x: str,
    asset_y: str,
    timeframe: str,
    exact_mode: str,
    orientation: str,
    cost_model_id: str,
    samples: int,
    rank: int,
    selected: bool,
) -> dict[str, Any]:
    return {
        "schema_version": QUEUE_ROW_SCHEMA_VERSION,
        "candidate_experiment_id": experiment_id,
        "registered_semantic_hypothesis_id": semantic_hypothesis_id,
        "registered_stage4_contract_id": stage4_contract_id,
        "pair_group_key": pair_group_key,
        "pair": pair,
        "asset_x": asset_x,
        "asset_y": asset_y,
        "timeframe": timeframe,
        "exact_mode": exact_mode,
        "orientation": orientation,
        "cost_model_id": cost_model_id,
        "strict_cost_model_current": True,
        "hyperliquid_markets_current": True,
        "immutable_stage4_candidate_identity_ready": True,
        "validated_current_model_candidate_samples": samples,
        "validated_current_model_pair_samples": samples,
        "local_blockers": "",
        "queue_status": "READY",
        "selection_rank": rank,
        "selected_active": selected,
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
