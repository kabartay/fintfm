#!/usr/bin/env bash
# Continued-pretraining pilot: one arm per real source, every arm identical but for the data.
#
# Every arm starts from the same trained checkpoint (the A6 control, runs/v4-cellattn-labels.pt),
# runs the same steps, seed, learning rate and schedule at the same architecture, and differs
# ONLY in --real-panels. The control arm continues on the synthetic prior alone, so "continued
# training itself" is held fixed and any difference is attributable to the source.
#
# This is a PILOT, sized for CPU on a shared machine: 1,500 steps x 8 tasks at 256-row
# contexts (~12k tasks, a quarter of the checkpoint's original 48k), scored on V4FinBench
# fold 0 under a 200k-row development cap. It ranks sources for the GPU decisive run; it is not
# that run. See docs/results/FINDINGS.md for the result and its limits.
#
# Usage: scripts/candidates/pilot.sh ARM [ARM ...]    (ARM from the list in arm_spec below)
set -euo pipefail
cd "$(dirname "$0")/../.."

STEPS=1500
SHARE=0.3   # fraction of tasks drawn from the real source in a non-control arm

arm_spec() {
  case "$1" in
    control)     echo "" ;;
    edgar)       echo "data/edgar/panel.parquet=$SHARE" ;;
    sba)         echo "data/real/sba/panel.parquet=$SHARE" ;;
    ulb_fraud)   echo "data/real/ulb_fraud/panel.parquet=$SHARE" ;;
    bondora)     echo "data/real/bondora/panel.parquet=$SHARE" ;;
    ppdai)       echo "data/real/ppdai/panel.parquet=$SHARE" ;;
    lendingclub) echo "data/lendingclub/panel.parquet=$SHARE" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

mkdir -p runs/pilot
for arm in "$@"; do
  spec="$(arm_spec "$arm")"
  echo "[$(date +%H:%M:%S)] train $arm  real-panels='${spec}'"
  PYTHONUNBUFFERED=1 nice -n 5 uv run fintfm-train \
    --init-from runs/v4-cellattn-labels.pt \
    --steps "$STEPS" --lr 1e-4 --batch-size 8 --n-rows-choices 256 \
    --d-cell 48 --d-model 128 --n-layers 4 --n-col-layers 2 --n-cell-blocks 1 \
    --column-id-dim 16 --feature-chunk 8 --max-features 136 --max-classes 2 --cell-labels \
    --p-financial 1.0 --real-panels "$spec" \
    --device cpu --threads 5 --seed 0 --checkpoint-every 500 \
    --out "runs/pilot/$arm.pt" > "runs/pilot/$arm.train.log" 2>&1
  echo "[$(date +%H:%M:%S)] score $arm"
  PYTHONUNBUFFERED=1 nice -n 5 uv run fintfm-v4protocol \
    --model "runs/pilot/$arm.pt" --folds 0 --max-rows 200000 --no-boosting \
    --classical logistic_regression --device cpu \
    --out "runs/pilot/$arm.score" > "runs/pilot/$arm.score.log" 2>&1
  echo "[$(date +%H:%M:%S)] done $arm"
done
