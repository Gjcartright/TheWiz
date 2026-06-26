#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="manifest"
CONFIG_PATH="config/candidate_pairs_multi_timeframe.yaml"
MAX_PAIRS=""
OFFSET="0"
PYTHON_BIN="/opt/anaconda3/bin/python3"
STRICT_ARGS=()
DEFER_RESEARCH=0

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
    --max-pairs)
      MAX_PAIRS="$2"
      shift 2
      ;;
    --offset)
      OFFSET="$2"
      shift 2
      ;;
    --strict)
      STRICT_ARGS=("--strict")
      shift 1
      ;;
    --defer-research)
      DEFER_RESEARCH=1
      shift 1
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "manifest" && "$MODE" != "plan" && "$MODE" != "run" ]]; then
  echo "--mode must be manifest, plan, or run" >&2
  exit 1
fi

PYTHONPATH=src "$PYTHON_BIN" - "$CONFIG_PATH" "$MODE" "$MAX_PAIRS" "$OFFSET" "$DEFER_RESEARCH" ${STRICT_ARGS[@]+"${STRICT_ARGS[@]}"} <<'PY'
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

config_path = Path(sys.argv[1])
mode = sys.argv[2]
max_pairs_arg = sys.argv[3]
offset = int(sys.argv[4] or 0)
defer_research = str(sys.argv[5]).strip() == "1"
strict_args = sys.argv[6:]

config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
candidate_path = Path(config["candidate_pairs_csv"])
funding_path = config.get("funding_path", "data/processed/dydx_funding.csv")
indexer_base = config.get("indexer_base", "https://indexer.dydx.trade")
output_dir = Path(config.get("output_dir", "reports/multi_timeframe_plans"))
manifest_path = Path(config.get("manifest_path", "reports/candidate_pairs_multi_timeframe_manifest.csv"))
summary_path = Path(config.get("summary_path", "reports/candidate_pairs_multi_timeframe_summary.csv"))
include_statuses = {str(value) for value in config.get("include_statuses", [])}
timeframes = list(config["timeframes"])

pairs = pd.read_csv(candidate_path)
if include_statuses and "status" in pairs.columns:
    pairs = pairs[pairs["status"].astype(str).isin(include_statuses)].copy()
if "research_priority_score" in pairs.columns:
    pairs["_priority"] = pd.to_numeric(pairs["research_priority_score"], errors="coerce").fillna(-math.inf)
    pairs = pairs.sort_values(["_priority", "pair_id"], ascending=[False, True])
pairs = pairs.iloc[offset:]
if max_pairs_arg:
    pairs = pairs.head(int(max_pairs_arg))

rows = []
for pair_rank, (_, row) in enumerate(pairs.iterrows(), start=offset + 1):
    pair_id = str(row["pair_id"])
    asset_x = str(row["asset_x"])
    asset_y = str(row["asset_y"])
    priority = row.get("research_priority_score", "")
    for item in timeframes:
        resolution = str(item["resolution"])
        windows = int(item.get("windows", 12))
        limit = int(item.get("limit", 1000))
        role = str(item.get("role", ""))
        plan_path = output_dir / f"{pair_rank:04d}_{pair_id}_{resolution.lower()}_long_history_plan.csv"
        run_command = (
            "bash scripts/run_dydx_long_history.sh "
            f"--asset-x {asset_x} --asset-y {asset_y} --pair-id {pair_id} "
            f"--windows {windows} --limit {limit} --resolution {resolution} "
            f"--indexer-base {indexer_base} --funding-path {funding_path}"
        )
        rows.append(
            {
                "pair_rank": pair_rank,
                "pair_id": pair_id,
                "asset_x": asset_x,
                "asset_y": asset_y,
                "research_priority_score": priority,
                "resolution": resolution,
                "windows": windows,
                "limit": limit,
                "role": role,
                "planned_candle_requests": windows * 2,
                "target_bars_per_leg_before_dedup": windows * limit,
                "plan_path": str(plan_path),
                "run_command": run_command,
            }
        )

manifest = pd.DataFrame(rows)
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest.to_csv(manifest_path, index=False)

summary_rows = [
    {"metric": "candidate_pairs", "value": int(manifest["pair_id"].nunique()) if not manifest.empty else 0},
    {"metric": "timeframes_per_pair", "value": len(timeframes)},
    {"metric": "manifest_rows", "value": int(len(manifest))},
    {"metric": "planned_candle_requests", "value": int(manifest["planned_candle_requests"].sum()) if not manifest.empty else 0},
    {"metric": "mode", "value": mode},
    {"metric": "offset", "value": offset},
    {"metric": "max_pairs", "value": max_pairs_arg or "all"},
    {"metric": "defer_research", "value": int(defer_research)},
]
summary = pd.DataFrame(summary_rows)
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(summary_path, index=False)

print(summary.to_string(index=False), flush=True)
print(f"manifest: {manifest_path}", flush=True)
print(f"summary: {summary_path}", flush=True)

if mode == "manifest":
    raise SystemExit(0)

output_dir.mkdir(parents=True, exist_ok=True)
for record in rows:
    print(
        f"{mode} {record['pair_rank']:04d} {record['pair_id']} {record['resolution']} "
        f"windows={record['windows']} limit={record['limit']}",
        flush=True,
    )
    if mode == "plan":
        command = [
            "/opt/anaconda3/bin/python3",
            "-m",
            "quant_platform.cli",
            "dydx-long-history-plan",
            "--asset-x",
            record["asset_x"],
            "--asset-y",
            record["asset_y"],
            "--pair-id",
            record["pair_id"],
            "--windows",
            str(record["windows"]),
            "--limit",
            str(record["limit"]),
            "--interval",
            record["resolution"],
            "--indexer-base",
            indexer_base,
            "--output-path",
            record["plan_path"],
        ]
    else:
        command = [
            "bash",
            "scripts/run_dydx_long_history.sh",
            "--asset-x",
            record["asset_x"],
            "--asset-y",
            record["asset_y"],
            "--pair-id",
            record["pair_id"],
            "--windows",
            str(record["windows"]),
            "--limit",
            str(record["limit"]),
            "--resolution",
            record["resolution"],
            "--indexer-base",
            indexer_base,
            "--funding-path",
            funding_path,
            *strict_args,
        ]
        if defer_research:
            command.append("--skip-research")
    subprocess.run(command, check=True)

if mode == "run" and defer_research and rows:
    print("run final research refresh", flush=True)
    command = [
        "/opt/anaconda3/bin/python3",
        "-m",
        "quant_platform.cli",
        "run-pair-detail-experiments",
        "--funding-path",
        funding_path,
    ]
    subprocess.run(command, check=True)
PY
