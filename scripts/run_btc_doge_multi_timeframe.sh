#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="plan"
CONFIG_PATH="config/btc_doge_multi_timeframe.yaml"
PYTHON_BIN="/opt/anaconda3/bin/python3"
STRICT_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --strict)
      STRICT_ARGS=("--strict")
      shift 1
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "plan" && "$MODE" != "run" ]]; then
  echo "--mode must be plan or run" >&2
  exit 1
fi

PYTHONPATH=src "$PYTHON_BIN" - "$CONFIG_PATH" "$MODE" ${STRICT_ARGS[@]+"${STRICT_ARGS[@]}"} <<'PY'
import subprocess
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
mode = sys.argv[2]
strict_args = sys.argv[3:]
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

asset_x = config["asset_x"]
asset_y = config["asset_y"]
pair_id = config["pair_id"]
funding_path = config.get("funding_path", "data/processed/dydx_funding.csv")
indexer_base = config.get("indexer_base", "https://indexer.dydx.trade")

for item in config["timeframes"]:
    resolution = str(item["resolution"])
    windows = str(item.get("windows", 12))
    limit = str(item.get("limit", 1000))
    print(f"{mode} {pair_id} {resolution} windows={windows} limit={limit}", flush=True)
    if mode == "plan":
        command = [
            "/opt/anaconda3/bin/python3",
            "-m",
            "quant_platform.cli",
            "dydx-long-history-plan",
            "--asset-x",
            asset_x,
            "--asset-y",
            asset_y,
            "--pair-id",
            pair_id,
            "--windows",
            windows,
            "--limit",
            limit,
            "--interval",
            resolution,
            "--indexer-base",
            indexer_base,
            "--output-path",
            f"reports/{pair_id}_{resolution.lower()}_long_history_plan.csv",
        ]
    else:
        command = [
            "bash",
            "scripts/run_dydx_long_history.sh",
            "--asset-x",
            asset_x,
            "--asset-y",
            asset_y,
            "--pair-id",
            pair_id,
            "--windows",
            windows,
            "--limit",
            limit,
            "--resolution",
            resolution,
            "--indexer-base",
            indexer_base,
            "--funding-path",
            funding_path,
            *strict_args,
        ]
    subprocess.run(command, check=True)
PY
