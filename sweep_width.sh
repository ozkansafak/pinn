#!/bin/bash
# sweep_width.sh — run PINN training across 10 doubling layer widths
#
# Usage:
#   bash sweep_width.sh                       # siren, widths 4..2048, cap 60k epochs
#   bash sweep_width.sh 4 60000 siren         # explicit
#   bash sweep_width.sh 4 60000 tanh          # tanh baseline

set -euo pipefail

START_WIDTH=${1:-4}
MAX_EPOCHS=${2:-60000}
ACTIVATION=${3:-siren}
OUTPUT="results/width_sweep.csv"
LOG_DIR="results/logs"

mkdir -p "$LOG_DIR"

echo "========================================"
echo "  PINN width sweep"
echo "  activation  : $ACTIVATION"
echo "  start_width : $START_WIDTH"
echo "  max_epochs  : $MAX_EPOCHS"
echo "  output      : $OUTPUT"
echo "========================================"
echo

width=$START_WIDTH
sweep_start=$(date +%s)

for run in $(seq 1 10); do
    log_file="$LOG_DIR/width_${width}_${ACTIVATION}.log"
    echo "──────────────────────────────────────────"
    echo "  Run $run/10 — width=$width  activation=$ACTIVATION"
    echo "  Log → $log_file"
    echo "──────────────────────────────────────────"

    run_start=$(date +%s)

    python train_width.py \
        --width "$width" \
        --max-epochs "$MAX_EPOCHS" \
        --activation "$ACTIVATION" \
        --output "$OUTPUT" \
        2>&1 | tee "$log_file"

    run_end=$(date +%s)
    run_secs=$(( run_end - run_start ))
    echo "  Done — ${run_secs}s ($(( run_secs / 60 ))m $(( run_secs % 60 ))s)"
    echo

    width=$(( width * 2 ))
done

sweep_end=$(date +%s)
total_secs=$(( sweep_end - sweep_start ))
echo "========================================"
echo "  Sweep complete"
echo "  Total time : $(( total_secs / 3600 ))h $(( (total_secs % 3600) / 60 ))m $(( total_secs % 60 ))s"
echo "  Results    : $OUTPUT"
echo "========================================"
