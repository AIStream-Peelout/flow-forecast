#!/bin/bash
# Three-rung forcing ladder on a fixed 10-basin subset. Every setting is identical across rungs --
# same seed, learning rate, schedule and basins -- so the ONLY difference is how precipitation
# reaches the ODE. That makes the comparison causal, which the earlier A/B was not.
#
#   rung 1  physics    : NLDAS precipitation and PET straight into snow-GR4; only the parameter
#                        head learns. This is the bar the neural components must clear.
#   rung 2  multiplier : adds the bounded [0.5, 2.0] learned correction on gridded precipitation.
#   rung 3  asos       : adds the gated station-innovation term for storms the grid understated.
#
# If rung 2 cannot beat rung 1, the Crossformer is not contributing.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
COMMON="--swe --max-basins 10 --epochs 15 --samples-per-epoch 1024 --lr 1e-3 --patience 5 --seed 42"

for rung in "ladder1_physics --anchored --no-multiplier" \
            "ladder2_multiplier --anchored" \
            "ladder3_asos --anchored --asos-gate"; do
  set -- $rung
  name=$1; shift
  echo "=== LADDER RUNG $name ==="
  # Fresh checkpoint per rung: EarlyStopper writes a single working-directory checkpoint.pth, and
  # a stale one can restore another run's weights (and its saved parameter-range buffers).
  rm -f checkpoint.pth
  $PY experiments/catchment_foundation/run_training.py --name "$name" "$@" $COMMON
  echo "=== RUNG $name EXIT=$? ==="
done
rm -f checkpoint.pth
echo "LADDER COMPLETE"
