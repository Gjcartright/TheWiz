#!/usr/bin/env bash
set -euo pipefail

# Shell-first dYdX two-leg fetcher.
# Purpose: keep pair-data expansion working when the Python requests/curl-in-Python
# path is blocked by resolver differences in the execution environment.

ASSET_X="BTC-USD"
ASSET_Y="ETH-USD"
PAIR_ID="btc_eth"
LIMIT=1000
INDEXER_BASE="${QPA_INDEXER_BASE:-https://indexer.dydx.trade}"
INDEXER_SCHEME="${QPA_INDEXER_SCHEME:-}"
RETRY_COUNT=5
RETRY_DELAY=2
OUT_DIR="data/raw/dydx_manual"
FUNDING_PATH="data/processed/dydx_funding.csv"
RUN_RESEARCH=0
SKIP_FETCH=0
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
    --limit)
      LIMIT="$2"
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
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --funding-path)
      FUNDING_PATH="$2"
      shift 2
      ;;
    --skip-fetch)
      SKIP_FETCH=1
      shift 1
      ;;
    --run-research)
      RUN_RESEARCH=1
      shift 1
      ;;
    --retry-count)
      RETRY_COUNT="$2"
      shift 2
      ;;
    --retry-delay)
      RETRY_DELAY="$2"
      shift 2
      ;;
    --help|-h)
  cat <<EOF
Usage: fetch_dydx_two_leg_shell.sh \
  --asset-x BTC-USD --asset-y ETH-USD --pair-id btc_eth \
  [--limit 1000] [--indexer-base ${QPA_INDEXER_BASE:-https://indexer.dydx.trade}] \
  [--indexer-scheme ${QPA_INDEXER_SCHEME:-http|https}] \
  [--out-dir data/raw/dydx_manual] [--funding-path data/processed/dydx_funding.csv] \
  [--skip-fetch] [--run-research]
EOF
      exit 0
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

if [[ "$INDEXER_SCHEME" == "http" || "$INDEXER_SCHEME" == "https" ]]; then
  INDEXER_BASE="${INDEXER_SCHEME}://${INDEXER_BASE#*://}"
fi

mkdir -p "$OUT_DIR"
LEFT_OUT="$OUT_DIR/${ASSET_X}_5MINS_candles.json"
RIGHT_OUT="$OUT_DIR/${ASSET_Y}_5MINS_candles.json"
LEFT_FUNDING_OUT="$OUT_DIR/${ASSET_X}_funding.json"
RIGHT_FUNDING_OUT="$OUT_DIR/${ASSET_Y}_funding.json"

assert_nonempty_file() {
  local output="$1"
  local label="$2"
  if [[ ! -s "$output" ]]; then
    echo "${label} file is missing or empty: $output" >&2
    exit 1
  fi
}

retry_fetch() {
  local url="$1"
  local output="$2"
  local tries=0
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
    if (( tries >= RETRY_COUNT )); then
      echo "fetch_failed: $url" >&2
      return 1
    fi
    echo "retry ${tries}/${RETRY_COUNT} for $url" >&2
    sleep "$RETRY_DELAY"
  done
}

LEFT_URL="${INDEXER_BASE}/v4/candles/perpetualMarkets/${ASSET_X}?resolution=5MINS&limit=${LIMIT}"
RIGHT_URL="${INDEXER_BASE}/v4/candles/perpetualMarkets/${ASSET_Y}?resolution=5MINS&limit=${LIMIT}"
LEFT_FUNDING_URL="${INDEXER_BASE}/v4/historicalFunding/${ASSET_X}?limit=${LIMIT}"
RIGHT_FUNDING_URL="${INDEXER_BASE}/v4/historicalFunding/${ASSET_Y}?limit=${LIMIT}"

if (( SKIP_FETCH == 1 )); then
  assert_nonempty_file "$LEFT_OUT" "left candle"
  assert_nonempty_file "$RIGHT_OUT" "right candle"
  assert_nonempty_file "$LEFT_FUNDING_OUT" "left funding"
  assert_nonempty_file "$RIGHT_FUNDING_OUT" "right funding"
else
  retry_fetch "$LEFT_URL" "$LEFT_OUT"
  retry_fetch "$RIGHT_URL" "$RIGHT_OUT"
  retry_fetch "$LEFT_FUNDING_URL" "$LEFT_FUNDING_OUT"
  retry_fetch "$RIGHT_FUNDING_URL" "$RIGHT_FUNDING_OUT"
  assert_nonempty_file "$LEFT_OUT" "left candle"
  assert_nonempty_file "$RIGHT_OUT" "right candle"
  assert_nonempty_file "$LEFT_FUNDING_OUT" "left funding"
  assert_nonempty_file "$RIGHT_FUNDING_OUT" "right funding"
fi

PYTHONPATH=src "$PYTHON_BIN" -m quant_platform.cli export-dydx-funding \
  --json-path "$OUT_DIR" \
  --output-path "$FUNDING_PATH"
assert_nonempty_file "$FUNDING_PATH" "funding CSV"

declare -a research_args=()
if (( RUN_RESEARCH == 1 )); then
  research_args+=(--run-research --funding-path "$FUNDING_PATH")
fi

PYTHONPATH=src "$PYTHON_BIN" -m quant_platform.cli build-dydx-pair-history \
  --left-candles "$LEFT_OUT" \
  --right-candles "$RIGHT_OUT" \
  --asset-x "$ASSET_X" \
  --asset-y "$ASSET_Y" \
  --pair-id "$PAIR_ID" \
  --interval 5mins \
  --derive-hedge-ratio \
  --zscore-window 320 \
  "${research_args[@]+"${research_args[@]}"}"
