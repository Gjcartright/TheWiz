# Workspace Alignment Plan — June 2026 Historical Record

This plan was committed on 2026-06-26 at `c0f7f5c`. Its Documents/Codex paths
and credential observations describe that earlier workspace. They are not current
operating instructions.

As of the 2026-09-30 Gate 0 source review, the designated writable source checkout
is `/Volumes/Expansion/Crypto Wizard`, directly on the mounted Expansion drive.
The encrypted workspace images are preserved historical copies, not active source
or a prerequisite for the current checkout. See
[current corrective operations](current_corrective_operations.md) and the dated
Gate 0 custody receipts for the current source and backup state.

## Historical goal and context

The June plan sought one repository under
`/Users/gregc/Documents/Codex/TheWiz-publish-20260625` while preserving older
folders as references. Several Documents/Codex folders overlapped, and the
inventory found only one Git checkout at that time. Running commands from a
non-Git folder could disconnect `.env.local` and artifacts from the active branch.

## Historical operating rules

The June plan designated the Documents/Codex repository as its execution root,
kept older folders as read-only references, kept secrets out of Git, and called
for `check-live-config` and `system-check` before research runs. That designation
was superseded by the Expansion checkout above. The old `.env.local` observations
in [the inventory](../work/legacy_workspace_inventory.csv) are dated facts, not
current credential evidence or permission to run API-dependent commands.

## Artifacts from that plan

- `work/legacy_workspace_inventory.csv` records the 2026-06-26 folder inventory.
- `reports/workspace_alignment_status.md` points to the original committed status
  placeholder; its former assertions are not current checks.
- The gap analysis, pre-mortem, post-mortem, and red-team reports remain historical
  review artifacts.

## Current local checkout check

`scripts/ops/validate_workspace.sh` records the Git root, branch, commit,
upstream, and working-tree change count of the checkout that runs it. Its
transient output goes to the Git-ignored
`reports/active/workspace_alignment_status.md`. The script does not inspect
credentials, confirm canonical-source selection, or grant research, release,
Testnet, or live trading authority. Use the current operations guide and dated
Gate 0 receipts to identify the designated source and backup state.
