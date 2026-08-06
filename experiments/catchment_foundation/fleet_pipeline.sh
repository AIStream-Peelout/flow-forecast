#!/bin/bash
# Full-fleet pipeline: SNODAS historical backfill (2003-2019) -> series recompile (inside the
# scrape loop) -> manifest rebuild -> full-fleet multi-basin training with SWE seeding.
set -o pipefail
FF=/Users/isaac/Documents/GitHub/flow-forecast
WATER=/Users/isaac/Documents/GitHub/Water
PY=$FF/.venv/bin/python

echo "=== FLEET STAGE 1: SNODAS backfill 2003-2019 ==="
SCRAPE_RANGE_LIST="2003-09-30:2019-09-30" "$WATER/snodas_scrape_loop.sh"

echo "=== FLEET STAGE 2: manifest rebuild ==="
cd "$FF" || exit 1
$PY experiments/catchment_foundation/build_manifest.py \
    --out experiments/catchment_foundation/manifests/co_manifest.json

echo "=== FLEET STAGE 3: full-fleet training (+SWE) ==="
$PY experiments/catchment_foundation/run_training.py --name fleet_v1_swe --swe \
    --epochs 20 --samples-per-epoch 4096
echo "=== FLEET PIPELINE COMPLETE code=$? ==="
