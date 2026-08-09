from __future__ import annotations

from hashlib import sha256

import pandas as pd

from quant_platform.youtube_channel_research import (
    build_hudson_thames_youtube_research,
    youtube_channel_research_paths,
)


def test_hudson_thames_research_stays_separate_and_research_only(tmp_path):
    paths = youtube_channel_research_paths(root=tmp_path)
    paths["subtitles"].mkdir(parents=True)
    video_id = "video000001"
    caption = paths["subtitles"] / f"{video_id}.en-orig.vtt"
    caption.write_text(
        "\n".join(
            [
                "WEBVTT",
                "",
                "00:00:05.000 --> 00:00:12.000",
                "copula tail dependence should be validated with cross validation",
                "",
                "00:00:50.000 --> 00:00:58.000",
                "meta labeling can calibrate position sizing without replacing the strategy",
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "video_id": video_id,
                "title": "Copula Arbitrage and Meta-Labeling",
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "duration_seconds": 120,
                "caption_status": "available",
            }
        ]
    ).to_csv(paths["video_registry"], index=False)
    pd.DataFrame(
        [
            {
                "video_id": video_id,
                "language": "en-orig",
                "variant": "en-orig",
                "path": str(caption),
                "sha256": sha256(caption.read_bytes()).hexdigest(),
                "bytes": caption.stat().st_size,
                "modified_at": "2026-08-06T00:00:00+00:00",
            }
        ]
    ).to_csv(paths["caption_manifest"], index=False)

    result = build_hudson_thames_youtube_research(root=tmp_path)
    inventory = pd.read_csv(result.paths["inventory"]).fillna("")
    candidates = pd.read_csv(result.paths["candidates"]).fillna("")
    recommendations = pd.read_csv(result.paths["recommendations"]).fillna("")
    sources_dashboard = pd.read_csv(result.paths["sources_dashboard"]).fillna("")

    assert inventory.iloc[0]["source_key"] == "hudson_thames"
    assert "copula" in set(candidates["theme"])
    assert recommendations["promotion_authority"].eq("none_research_only").all()
    assert not recommendations["trade_authorized"].astype(bool).any()
    assert "hudson_thames" in set(sources_dashboard["source_key"])
    assert "Hudson & Thames" in result.paths["recommendations_markdown"].read_text(encoding="utf-8")
