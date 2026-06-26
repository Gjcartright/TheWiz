# Post-Mortem: What happened

## Actual issue observed
The user saw different credentials and behavior across sessions because multiple local directories contained project-like material, but only one was an active git repo with remote `Gjcartright/TheWiz`.

## Root technical cause
`TheWiz-publish-20260625` is the only repository with a `.git` directory and branch state.
Other folders were treated as context folders; they can still hold older `.env.local` values.
When commands ran from those folders, configuration used was unrelated to the active tracked branch.

## Why this felt abrupt
The error messages were runtime-level (“API key required”) rather than repository-level (“wrong workspace”), so it seemed like a data source outage instead of a workspace mismatch.

## Fix implemented
- Explicitly documented canonical workspace.
- Created execution validation script.
- Added legacy workspace inventory and analysis docs.
- Added a persistent alignment status report.

## Evidence
- System checks now run from canonical repo path.
- `.env.local` in canonical repo includes required keys.
- Legacy dirs are preserved as evidence, not active execution source.
