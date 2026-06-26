#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ASSET_X="SOL-USD"
ASSET_Y="LINK-USD"
PAIR_ID="sol_link"
WINDOWS=12
LIMIT=1000
RESOLUTION="5MINS"
TO_ISO=""
FUNDING_PATH="data/processed/dydx_funding.csv"
INDEXER_BASE="${QPA_INDEXER_BASE:-https://indexer.dydx.trade}"
INDEXER_SCHEME="${QPA_INDEXER_SCHEME:-}"
RETRY_COUNT=3
RETRY_DELAY=2
ALLOW_PARTIAL=1
RUN_RESEARCH=1
PYTHON_BIN="/opt/anaconda3/bin/python3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --asset-x)
      ASSET_X="$2"
      shift 2
      ;;
    --asset-y)
      ASSET_Y="$2"
      shift 2
      ;;
    --pair-id)
      PAIR_ID="$2"
      shift 2
      ;;
    --windows)
      WINDOWS="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --resolution|--interval)
      RESOLUTION="$2"
      shift 2
      ;;
    --to-iso)
      TO_ISO="$2"
      shift 2
      ;;
    --funding-path)
      FUNDING_PATH="$2"
      shift 2
      ;;
    --indexer-base)
      INDEXER_BASE="$2"
      shift 2
      ;;
    --indexer-scheme)
      INDEXER_SCHEME="$2"
      shift 2
      ;;
    --retry-count)
      RETRY_COUNT="$2"
      shift 2
      ;;
    --retry-delay)
      RETRY_DELAY="$2"
      shift 2
      ;;
    --strict)
      ALLOW_PARTIAL=0
      shift 1
      ;;
    --skip-research)
      RUN_RESEARCH=0
      shift 1
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

normalize_indexer_base() {
  local base="$1"
  if [[ "$base" == http://* || "$base" == https://* ]]; then
    printf '%s\n' "$base"
  else
    printf 'https://%s\n' "$base"
  fi
}

INDEXER_BASE="$(normalize_indexer_base "$INDEXER_BASE")"

plan_args=(
  "dydx-long-history-plan"
  "--asset-x" "$ASSET_X"
  "--asset-y" "$ASSET_Y"
  "--pair-id" "$PAIR_ID"
  "--windows" "$WINDOWS"
  "--limit" "$LIMIT"
  "--interval" "$RESOLUTION"
  "--indexer-base" "$INDEXER_BASE"
  "--indexer-scheme" "$INDEXER_SCHEME"
)
if [[ -n "$TO_ISO" ]]; then
  plan_args+=("--to-iso" "$TO_ISO")
fi

PYTHONPATH=src "$PYTHON_BIN" -m quant_platform.cli "${plan_args[@]}" >/dev/null

fetch_long_history_window() {
  local url="$1"
  local output="$2"
  local tries=0
  mkdir -p "$(dirname "$output")"
  while (( tries < RETRY_COUNT )); do
    tries=$((tries + 1))
    if URL="$url" OUTPUT_PATH="$output" RETRY_COUNT="$RETRY_COUNT" QPA_INDEXER_SCHEME="${INDEXER_SCHEME}" PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
from quant_platform.cli import _fetch_public_json

url = os.environ["URL"]
output_path = Path(os.environ["OUTPUT_PATH"])
retries = max(1, int(os.environ.get("RETRY_COUNT", "3")))

_fetch_public_json(url, output_path, max_retries=retries, timeout=30.0)
PY
    then
      return 0
    fi
    if [[ "$ALLOW_PARTIAL" -eq 1 ]]; then
      echo "fetch_failed ${tries}/${RETRY_COUNT}: $url" >&2
    fi
    if (( tries >= RETRY_COUNT )); then
      return 1
    fi
    sleep "$RETRY_DELAY"
  done
}

while IFS=$'\t' read -r url save_as; do
  if ! fetch_long_history_window "$url" "$save_as"; then
    if [[ "$ALLOW_PARTIAL" -eq 1 ]]; then
      echo "fetch_failed: $url" >&2
      continue
    fi
    exit 1
  fi
done < <(
PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
import csv
import os
from pathlib import Path

plan = Path("reports/dydx_long_history_plan.csv")
with plan.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if str(row.get("method", "")).upper() != "GET":
            continue
        request_name = str(row.get("request_name", ""))
        if "candles" not in request_name.lower():
            continue
        url = str(row.get("url", "")).strip()
        save_as = str(row.get("save_as", "")).strip()
        if url and save_as:
            print(f"{url}\t{save_as}")
PY
)

build_args=(
  "build-dydx-long-history-pair"
  --asset-x "$ASSET_X"
  --asset-y "$ASSET_Y"
  --pair-id "$PAIR_ID"
  --interval "$RESOLUTION"
  --derive-hedge-ratio
)

if [[ "$RUN_RESEARCH" -eq 1 ]]; then
  build_args+=(
    --run-research
    --research-funding-path "$FUNDING_PATH"
  )
fi

PYTHONPATH=src "$PYTHON_BIN" -m quant_platform.cli "${build_args[@]}"
