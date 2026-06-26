# Pre-Mortem: Workspace Alignment Failure Modes

## What could fail during alignment
- We could still run research commands from a non-canonical folder because path errors are silent.
- Another `.env.local` in a legacy folder could still be edited and looked “successful” while active run reads old values.
- Validation script could silently pass if only existence checks are performed but not command-execution context.

## Early signals we monitor
- `pwd` does not match `/Users/gregc/Documents/Codex/TheWiz-publish-20260625` before a command.
- `CRYPTO_WIZARDS_API_KEY` not loadable in active repo.
- Active branch drifts from `origin` with unreferenced local CSV outputs only.

## Hard controls to prevent failure
- Require a pre-flight step for every research session:
  - `python -m quant_platform.cli check-live-config`
  - `python -m quant_platform.cli system-check`
  - `bash scripts/ops/validate_workspace.sh`
- Add explicit failure when canonical path not detected for API-dependent commands.

## Why this matters
Without this control, results appear valid but come from stale or unauthorized environments, creating false acceptance signals and wasted experiments.
