# YouTube Research Brain

The YouTube Research Brain is a research-only specialist that turns the Crypto Wizards video archive into traceable claims, formulas, warnings, strategy memories, and pair-specific test hypotheses. It cannot authorize a Hyperliquid Testnet or live trade.

## Daily Cycle

```bash
PYTHONPATH=src python -m quant_platform.cli run-youtube-brain
```

The cycle:

1. Refreshes the Crypto Wizards `/videos` catalog when the configured refresh interval is due.
2. Downloads captions only for newly discovered videos.
3. Rebuilds direct-source claims and the formula/strategy/warning knowledge graph.
4. Generates exact-mode hypotheses for the current Wizard candidate snapshot.
5. Matches hypotheses to local walk-forward and verified Hyperliquid Testnet outcomes.
6. Appends changed validation evidence to persistent agent memory.
7. Publishes dashboard tables and a concise status report.

Use `--force-refresh` for an immediate live catalog check and `--no-fetch` for a reproducible local-only rebuild.

## Configuration

Configuration lives in `config/youtube_brain.yaml`:

- `channel_url`: canonical channel URL.
- `auto_refresh_enabled`: enables due-based collection in the full cycle.
- `refresh_hours`: collection cadence.
- `fetch_captions`: whether new-video captions are collected.
- `caption_languages`: accepted subtitle languages.
- `max_new_caption_fetches_per_cycle`: download cap per cycle.
- `promotion_authority`: fixed to `none_research_only`.

## Evidence Lifecycle

- `UNVERIFIED`: no matching local validation evidence.
- `REPLICATING`: proxy out-of-sample metrics pass, but acceptance remains blocked.
- `WALK_FORWARD_SUPPORTED`: matching local costed walk-forward rows pass acceptance.
- `TESTNET_SUPPORTED`: at least one explicitly closed Hyperliquid Testnet outcome is positive.
- `REJECTED`: matching evidence fails its validation threshold.

Wizard metrics and YouTube claims prioritize research. Local costed walk-forward evidence and explicit Hyperliquid Testnet outcomes determine replication state. Even `TESTNET_SUPPORTED` does not by itself authorize live execution.

## Main Artifacts

- `data/external/youtube/brain/video_registry.csv`
- `data/external/youtube/brain/caption_manifest.csv`
- `data/processed/youtube_brain/claims.csv`
- `data/processed/youtube_brain/formulas.csv`
- `data/processed/youtube_brain/strategy_memory.csv`
- `data/processed/youtube_brain/warning_memory.csv`
- `data/processed/youtube_brain/knowledge_nodes.csv`
- `data/processed/youtube_brain/knowledge_edges.csv`
- `data/agent_memory/youtube_research_brain.jsonl`
- `reports/agents/youtube_brain_hypotheses.csv`
- `reports/agents/youtube_brain_replication_scorecard.csv`
- `reports/dashboard/youtube_brain.md`

Every hypothesis includes source URLs, formula IDs, current Wizard evidence, a documented entry/exit research plan, blockers, freshness, and the next local test step.

## Component Commands

```bash
PYTHONPATH=src python -m quant_platform.cli refresh-youtube-collection --force-refresh
PYTHONPATH=src python -m quant_platform.cli build-youtube-brain
PYTHONPATH=src python -m quant_platform.cli build-youtube-hypotheses
PYTHONPATH=src python -m quant_platform.cli refresh-youtube-outcomes
PYTHONPATH=src python -m quant_platform.cli build-youtube-brain-dashboard
```

The specialist is also registered in the deterministic orchestrator under the `youtube_research_brain` stage and in mini-agent planning. Its hypotheses become test tasks, never direct execution tasks.
