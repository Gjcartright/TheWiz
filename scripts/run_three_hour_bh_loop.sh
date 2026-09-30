#!/usr/bin/env bash
set -euo pipefail

# The June 2026 three-hour research loop is retained in Git history.
# Do not revive it by changing its old workspace path: its command sequence
# predates the current source, runtime, and research acceptance contracts.
printf '%s\n' \
  'ERROR: The three-hour BH research loop is retired.' \
  'Its historical implementation is preserved in Git history.' \
  'Use docs/current_corrective_operations.md for current Gate 0 procedures.' >&2
exit 78
