#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m pip install --no-build-isolation -e ".[dydx]"
python3 - <<'PY'
import importlib.util

installed = importlib.util.find_spec("dydx_v4_client") is not None
print(f"dydx_v4_client_installed={installed}")
if not installed:
    raise SystemExit(1)
PY
