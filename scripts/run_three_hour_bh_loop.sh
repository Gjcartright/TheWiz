#!/usr/bin/env bash
set -euo pipefail

ROOT=/Users/gregc/Documents/Codex/TheWiz-publish-20260625
LOG_DIR="$ROOT/scripts/schedule_logs"
LOG_FILE="$LOG_DIR/three_hour_schedule_$(date +"%Y%m%d_%H%M%S").log"
DURATION_MINUTES=180
START_TS=$(date +%s)
END_TS=$((START_TS + DURATION_MINUTES * 60))
INTERVAL_SECONDS=$((15 * 60))
CYCLE=0

{
  echo "=== three-hour run schedule start ==="
  echo "start_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "end_target_ts=$END_TS"
  echo "interval_seconds=$INTERVAL_SECONDS"
  echo "log=$LOG_FILE"

  while [ $(date +%s) -lt "$END_TS" ]; do
    CYCLE=$((CYCLE+1))
    NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "\n--- CYCLE $CYCLE --- $NOW ---"

    cd "$ROOT"

    # Orchestration visibility
    PYTHONPATH=src python -m quant_platform.cli system-check | tee -a "$LOG_FILE" || true
    PYTHONPATH=src python -m quant_platform.cli paper-candidate-shortlist | tee -a "$LOG_FILE" || true
    PYTHONPATH=src python -m quant_platform.cli priority-dashboard | tee -a "$LOG_FILE" || true

    # Candidate discovery + refresh signals for BTC/ETH/SOL
    PYTHONPATH=src python -m quant_platform.cli dydx-live-market-selector --coin BTC --coin ETH --coin SOL | tee -a "$LOG_FILE" || true
    PYTHONPATH=src python -m quant_platform.cli run-dydx-pair-expansion --max-pairs 1 --limit 1000 | tee -a "$LOG_FILE" || true
    PYTHONPATH=src python -m quant_platform.cli run-dydx-pair-expansion --max-pairs 1 --limit 1000 | tee -a "$LOG_FILE" || true
    PYTHONPATH=src python -m quant_platform.cli run-dydx-pair-expansion --max-pairs 1 --limit 1000 | tee -a "$LOG_FILE" || true

    # RL route gate + paper readiness slices
    PYTHONPATH=src python -m quant_platform.cli run-base-rl --pair-id BTC | tee -a "$LOG_FILE" || true
    PYTHONPATH=src python -m quant_platform.cli run-base-rl --pair-id ETH | tee -a "$LOG_FILE" || true
    PYTHONPATH=src python -m quant_platform.cli run-base-rl --pair-id SOL | tee -a "$LOG_FILE" || true
    PYTHONPATH=src python -m quant_platform.cli evaluate-base-rl | tee -a "$LOG_FILE" || true
    PYTHONPATH=src python -m quant_platform.cli base-rl-paper-handoff | tee -a "$LOG_FILE" || true
    PYTHONPATH=src python -m quant_platform.cli compare-base-vs-augmented-rl | tee -a "$LOG_FILE" || true
    PYTHONPATH=src python -m quant_platform.cli promotion-readiness-report | tee -a "$LOG_FILE" || true
    PYTHONPATH=src python -m quant_platform.cli refresh-base-rl-feedback | tee -a "$LOG_FILE" || true

    # Gate posture / completion
    PYTHONPATH=src python -m quant_platform.cli strategy-acceptance-checklist | tee -a "$LOG_FILE" || true
    PYTHONPATH=src python -m quant_platform.cli gap-test | tee -a "$LOG_FILE" || true

    echo "cycle=$CYCLE complete at $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "$LOG_FILE"

    NOW_TS=$(date +%s)
    NEXT=$((NOW_TS + INTERVAL_SECONDS))
    if [ "$NEXT" -ge "$END_TS" ]; then
      break
    fi
    SLEEP_SECONDS=$((END_TS - NOW_TS < INTERVAL_SECONDS ? END_TS - NOW_TS : INTERVAL_SECONDS))
    echo "sleeping $SLEEP_SECONDS seconds until next cycle" | tee -a "$LOG_FILE"
    sleep "$SLEEP_SECONDS"
  done

  echo "=== three-hour schedule complete ==="
  echo "end_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "cycles=$CYCLE"
} | tee "$LOG_FILE"
