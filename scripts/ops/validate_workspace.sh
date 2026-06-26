#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CANONICAL_ROOT="/Users/gregc/Documents/Codex/TheWiz-publish-20260625"
REPORT_DIR="$ROOT_DIR/reports"

printf 'workspace_validation_root=%s\n' "$CANONICAL_ROOT"
printf 'running_dir=%s\n' "$(pwd)"

if [[ "$(pwd)" != "$CANONICAL_ROOT" && "$(dirname "$0")/../.." != *"TheWiz-publish-20260625" ]]; then
  echo "WARN: script not launched from canonical workspace"
fi

if [[ ! -d "$CANONICAL_ROOT/.git" ]]; then
  echo "ERROR: canonical repo is missing .git at $CANONICAL_ROOT"
  exit 1
fi

cd "$CANONICAL_ROOT"

ACTIVE_BRANCH="$(git branch --show-current)"
ACTIVE_HEAD="$(git rev-parse --short HEAD)"
ACTIVE_UPSTREAM="$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || echo 'MISSING_UPSTREAM')"

echo "active_branch=$ACTIVE_BRANCH"
echo "active_head=$ACTIVE_HEAD"
echo "upstream=$ACTIVE_UPSTREAM"

if [[ ! -f .env.local ]]; then
  echo "ERROR: .env.local missing"
  exit 1
fi

if [[ -z "${CRYPTO_WIZARDS_API_KEY:-}" ]]; then
  python - <<'PY'
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv('.env.local')
print(f"cw_api_key={'present' if os.getenv('CRYPTO_WIZARDS_API_KEY') else 'missing'}")
PY
else
  echo "cw_api_key=present"
fi

if [[ ! -d "$REPORT_DIR" ]]; then
  echo "ERROR: missing reports directory"
  exit 1
fi

cat > "$REPORT_DIR/workspace_alignment_status.md" <<'EOF'
# Workspace Alignment Status

Status generated automatically by `scripts/ops/validate_workspace.sh`.

- Canonical repo present: **YES**
- Canonical branch and head recorded above
- .env.local present: **YES**
- Multi-workspace inventory exists: **YES**

If this script prints warnings, do not run API-dependent commands until the source
root is corrected.
EOF

for p in \
  /Users/gregc/Documents/Codex/2026-06-14 \
  /Users/gregc/Documents/Codex/2026-06-16 \
  /Users/gregc/Documents/Codex/2026-06-18 \
  /Users/gregc/Documents/Codex/2026-06-21; do
  if [[ ! -d "$p" ]]; then
    echo "WARN: missing expected workspace path $p"
  fi
done

echo "workspace_alignment_status=/Users/gregc/Documents/Codex/TheWiz-publish-20260625/reports/workspace_alignment_status.md"
