# Red Team: Adversarial Failure Review

## Attack / failure scenarios
1. **Path Drift Attack**
   - Someone runs commands from a legacy folder with stale credentials.
   - Impact: fake continuity, false production-ready labels.
2. **Credential Inconsistency**
   - Old key overwritten in local file while active `.env.local` is unchanged.
   - Impact: intermittent API failures and mixed experiment states.
3. **Artifact Contamination**
   - CSV reports generated in legacy folder mistaken as latest proof.
   - Impact: wrong pair shortlist and wrong execution decisions.

## Defensive controls
- `validate_workspace.sh` enforces canonical branch + `.env.local` + execution path checks.
- Weekly/manual reconciliation of `work/legacy_workspace_inventory.csv` with filesystem reality.
- Never mix evidence from outside `TheWiz-publish-20260625/reports` in acceptance decisions.

## Remaining risk
- Human error can still run commands manually from other folders; automation and runbook discipline is required.
- Mitigation: create shell alias/function that first calls validation before dispatching CLI commands.
