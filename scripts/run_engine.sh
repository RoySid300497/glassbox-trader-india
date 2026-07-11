#!/bin/bash
# running one engine mode with environment setup and timestamped logging
REPO="/Users/sidroy/Desktop/venv/ml/project/glassbox-trader"
LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"
MODE="${1:-daily}"
STAMP=$(date "+%Y-%m-%d %H:%M:%S")

source /Users/sidroy/Desktop/venv/bin/activate
cd "$REPO"
export STOCK_LENS_BASE="$REPO/stock-lens-data"
export PYTHONPATH="$REPO/src"

echo "==== $STAMP mode=$MODE ====" >> "$LOGDIR/engine.log"
python -m engine.run_daily --mode "$MODE" >> "$LOGDIR/engine.log" 2>&1
echo "==== exit=$? ====" >> "$LOGDIR/engine.log"
