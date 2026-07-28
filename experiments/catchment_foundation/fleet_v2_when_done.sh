#!/bin/bash
# Waits for the fleet_v1 pipeline process to exit, then launches the retuned fleet_v2 run.
FF=/Users/isaac/Documents/GitHub/flow-forecast
while pgrep -f "run_training.py --name fleet_v1_swe" >/dev/null; do sleep 120; done
cd "$FF" || exit 1
.venv/bin/python experiments/catchment_foundation/run_training.py --name fleet_v2_swe --swe \
    --epochs 30 --samples-per-epoch 16384 --lr 1e-3
echo "=== FLEET V2 DONE code=$? ==="
