# Workspace Alignment Plan (Step 4 Deepening)

## Goal
Keep all active work, code, and command execution in a single canonical repository:

- `/Users/gregc/Documents/Codex/TheWiz-publish-20260625`

while preserving every older folder as historical reference and avoiding accidental
credential or configuration drift.

## Why this needed
We discovered multiple local folders with overlapping project content:

- `/Users/gregc/Documents/Codex/2026-06-15-chief-quantitative-research-architect-you-are/`
- `/Users/gregc/Documents/Codex/2026-06-15-quantized-statistical-arbitrage-ai-agent-master/`
- `/Users/gregc/Documents/Codex/TheWiz-publish-20260625/` (git repo with remote `Gjcartright/TheWiz`)

Only one folder has an active `.git` checkout. Running commands from any non-git folder
kept `.env.local` and artifacts logically disconnected from the active branch.

## Canonical operating rules
1. **Single execution root**: run all CLI commands from
   `/Users/gregc/Documents/Codex/TheWiz-publish-20260625`.
2. **Do not run research or publish commands in folders without `.git`.**
3. **Never copy secrets into git-tracked files.** Keep keys only in `.env.local`.
4. **Before any research run**, execute:
   - `python -m quant_platform.cli check-live-config`
   - `python -m quant_platform.cli system-check`
5. **Record legacy references only as read-only pointers** (not execution source).

## Concrete actions executed
- Added workspace validation command in `scripts/ops/validate_workspace.sh`.
- Added `work/legacy_workspace_inventory.csv` with all local `/Users/Codex` folders.
- Added analyses artifacts under `reports/`:
  - `gap_analysis_workspace_alignment.csv`
  - `pre_mortem_workspace_alignment.md`
  - `post_mortem_workspace_alignment.md`
  - `red_team_workspace_alignment.md`
- Added status report `reports/workspace_alignment_status.md`.
- Added this plan document for repeatable onboarding.

## Ownership
Use `scripts/ops/validate_workspace.sh` at the start of every session to detect
environment mismatch before any command that depends on API credentials.
