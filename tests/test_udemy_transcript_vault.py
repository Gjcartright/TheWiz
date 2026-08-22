from __future__ import annotations

import json
from hashlib import sha256

import pandas as pd

from quant_platform.udemy_transcript_vault import build_udemy_transcript_vault_inventory


def _lecture_index(root):
    path = root / "data" / "processed" / "research_knowledge" / "udemy_lecture_index.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "lecture_id": "udl_one",
                "course": "Course A",
                "section": "Section A",
                "video_title": "Lecture A",
                "lecture_url": "https://www.udemy.com/course/a/learn/lecture/1#overview",
                "evidence_source": "udemy_transcript",
            },
            {
                "lecture_id": "udl_two",
                "course": "Course A",
                "section": "Section A",
                "video_title": "Lecture B",
                "lecture_url": "https://www.udemy.com/course/a/learn/lecture/2#overview",
                "evidence_source": "udemy_ai_assistant",
            },
        ]
    ).to_csv(path, index=False)
    return path


def test_vault_manifest_hashes_local_transcript_without_ingesting_body(tmp_path):
    index = _lecture_index(tmp_path)
    vault = tmp_path / "data" / "external" / "udemy" / "restricted_transcripts"
    vault.mkdir(parents=True)
    body = "first cue\nsecond cue\n"
    (vault / "udl_one.txt").write_text(body, encoding="utf-8")
    (vault / "udl_one.metadata.json").write_text(
        json.dumps(
            {
                "status": "captured",
                "lecture_url": "https://www.udemy.com/course/a/learn/lecture/1#overview",
                "cue_count": 2,
                "captured_at_utc": "2026-08-15T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    result = build_udemy_transcript_vault_inventory(root=tmp_path, lecture_index_path=index)

    manifest = pd.read_csv(result.paths["manifest"]).fillna("")
    captured = manifest.loc[manifest["lecture_id"].eq("udl_one")].iloc[0]
    assert captured["vault_status"] == "captured"
    assert captured["transcript_sha256"] == sha256(body.encode()).hexdigest()
    assert int(captured["transcript_cue_count"]) == 2
    assert bool(captured["independent_rereview_ready"])
    assert not bool(captured["transcript_text_ingested"])
    assert not bool(captured["live_signal_eligible"])
    assert captured["promotion_authority"] == "none_research_only"
    assert body not in result.paths["manifest"].read_text(encoding="utf-8")

    queue = json.loads(result.paths["capture_queue"].read_text(encoding="utf-8"))
    assert [row["lecture_id"] for row in queue] == ["udl_two"]


def test_vault_accepts_verified_unavailable_transcript_as_terminal_status(tmp_path):
    index = _lecture_index(tmp_path)
    vault = tmp_path / "data" / "external" / "udemy" / "restricted_transcripts"
    vault.mkdir(parents=True)
    for lecture_id, url in (
        ("udl_one", "https://www.udemy.com/course/a/learn/lecture/1#overview"),
        ("udl_two", "https://www.udemy.com/course/a/learn/lecture/2#overview"),
    ):
        (vault / f"{lecture_id}.metadata.json").write_text(
            json.dumps(
                {
                    "status": "transcript_unavailable",
                    "lecture_url": url,
                    "blocker": "udemy_transcript_not_available",
                }
            ),
            encoding="utf-8",
        )

    result = build_udemy_transcript_vault_inventory(root=tmp_path, lecture_index_path=index)

    assert result.summary["captures_pending"] == 0
    assert result.summary["transcripts_unavailable"] == 2
    assert result.summary["capture_resolution_complete"] is True
    assert result.summary["all_available_transcripts_rereview_ready"] is True
    assert result.summary["direct_transcript_complete"] is False
    assert result.summary["independent_rereview_ready"] is False

    recheck = build_udemy_transcript_vault_inventory(
        root=tmp_path,
        lecture_index_path=index,
        recheck_unavailable=True,
    )
    queue = json.loads(recheck.paths["capture_queue"].read_text(encoding="utf-8"))
    assert len(queue) == 2
    assert {row["vault_status"] for row in queue} == {"transcript_unavailable"}


def test_vault_blocks_duplicate_transcript_bodies_across_lectures(tmp_path):
    index = _lecture_index(tmp_path)
    vault = tmp_path / "data" / "external" / "udemy" / "restricted_transcripts"
    vault.mkdir(parents=True)
    body = "a transcript body that must not be reused across lecture ids\n"
    for lecture_id, url in (
        ("udl_one", "https://www.udemy.com/course/a/learn/lecture/1#overview"),
        ("udl_two", "https://www.udemy.com/course/a/learn/lecture/2#overview"),
    ):
        (vault / f"{lecture_id}.txt").write_text(body, encoding="utf-8")
        (vault / f"{lecture_id}.metadata.json").write_text(
            json.dumps({"status": "captured", "lecture_url": url, "cue_count": 1}),
            encoding="utf-8",
        )

    result = build_udemy_transcript_vault_inventory(root=tmp_path, lecture_index_path=index)

    manifest = pd.read_csv(result.paths["manifest"]).fillna("")
    assert set(manifest["vault_status"]) == {"blocked_duplicate_transcript_body"}
    assert not manifest["independent_rereview_ready"].astype(bool).any()
    assert result.summary["captures_pending"] == 2
