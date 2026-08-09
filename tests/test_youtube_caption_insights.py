from __future__ import annotations

from hashlib import sha256

import pandas as pd

from quant_platform.youtube_caption_insights import (
    MAX_EXCERPT_WORDS,
    build_youtube_caption_insights,
    youtube_caption_insight_paths,
)


def test_caption_insights_are_source_linked_and_research_only(tmp_path):
    external = tmp_path / "data" / "external" / "youtube" / "brain"
    subtitle_dir = external / "subtitles"
    subtitle_dir.mkdir(parents=True)
    video_id = "video000001"
    caption = subtitle_dir / f"{video_id}.en-orig.vtt"
    caption.write_text(
        "\n".join(
            [
                "WEBVTT",
                "",
                "00:00:10.000 --> 00:00:14.000",
                "you should make sure the spread is stationary before trading",
                "",
                "00:00:16.000 --> 00:00:20.000",
                "compare engle granger johansen and adf out of sample",
                "",
                "00:01:00.000 --> 00:01:05.000",
                "copula tail dependence can reveal a different dislocation signal",
            ]
        ),
        encoding="utf-8",
    )
    digest = sha256(caption.read_bytes()).hexdigest()
    pd.DataFrame(
        [
            {
                "video_id": video_id,
                "title": "Stationarity and Copula Research",
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }
        ]
    ).to_csv(external / "video_registry.csv", index=False)
    pd.DataFrame(
        [
            {
                "video_id": video_id,
                "language": "en-orig",
                "variant": "en-orig",
                "path": str(caption),
                "sha256": digest,
                "bytes": caption.stat().st_size,
                "modified_at": "2026-08-06T00:00:00+00:00",
            }
        ]
    ).to_csv(external / "caption_manifest.csv", index=False)

    result = build_youtube_caption_insights(root=tmp_path)
    paths = youtube_caption_insight_paths(tmp_path)
    candidates = pd.read_csv(paths["candidates"]).fillna("")
    recommendations = pd.read_csv(paths["recommendations"]).fillna("")

    assert result.summary["source_videos"] == 1
    assert {"stationarity", "copula"}.issubset(set(candidates["theme"]))
    assert candidates["timestamped_video_url"].str.contains("&t=", regex=False).all()
    assert candidates["evidence_excerpt"].map(lambda value: len(str(value).split()) <= MAX_EXCERPT_WORDS).all()
    assert recommendations["recommendation_type"].eq("inferred_research_recommendation").all()
    assert recommendations["promotion_authority"].eq("none_research_only").all()
    assert not recommendations["trade_authorized"].astype(bool).any()
    assert paths["recommendations_markdown"].read_text(encoding="utf-8").startswith("# YouTube Research Recommendations")
