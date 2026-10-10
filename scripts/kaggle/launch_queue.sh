#!/usr/bin/env bash
# Run Kaggle kernels one after another in a "lane", unattended: push (retrying while the weekly GPU
# quota is exhausted), wait for the run to finish, fetch its checkpoint, and score it on V4FinBench
# in the background while the lane moves on to its next kernel.
#
# Kaggle allows two concurrent GPU sessions, so run two lanes. A lane stops at the first kernel
# that errors rather than spending quota repeating what is likely a systematic failure.
#
# Each item is DIR:CHECKPOINT -- the kernel directory under scripts/kaggle/ and the file name its
# `--out` writes under /kaggle/working/. Results land in runs/kaggle-<slug>/; the lane's log is
# runs/kaggle_queue/<lane>.log.
#
# Usage (detached, survives the session):
#   nohup scripts/kaggle/launch_queue.sh lane1 task_158_matched_control:v4-cellattn-labels-mc10.pt \
#       fintfm_r_sba_s1:fintfm-r-sba-s1.pt > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."

LANE=$1; shift
LOGDIR=runs/kaggle_queue
mkdir -p "$LOGDIR"
LOG=$LOGDIR/$LANE.log
RETRY_S=${RETRY_S:-1800}   # between pushes refused for quota
POLL_S=${POLL_S:-600}      # between status checks of a running kernel
# Full-panel scorings each hold ~10 GB. Four at once (two lanes, each finishing a kernel while
# the previous score still ran) took this 64 GB machine to 63 GB used and 45/47 GB swap on
# 2026-10-10, one step from the freeze CLAUDE.md records. The cap is across lanes.
MAX_SCORING=${MAX_SCORING:-2}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

for item in "$@"; do
  dir=scripts/kaggle/${item%%:*}
  ckpt=${item#*:}
  id=$(python3 -c "import json; print(json.load(open('$dir/kernel-metadata.json'))['id'])")
  out=runs/kaggle-${id#*/}

  while true; do
    msg=$(kaggle kernels push -p "$dir" 2>&1)
    if grep -qi "quota" <<<"$msg"; then
      log "quota: $id -- retry in ${RETRY_S}s"; sleep "$RETRY_S"; continue
    elif grep -qi "successfully pushed" <<<"$msg"; then
      log "pushed $id"; break
    else
      log "PUSH FAILED $id: $msg -- lane stops"; exit 1
    fi
  done

  sleep "${SETTLE_S:-120}"   # a just-pushed kernel can briefly report the previous version's status
  while true; do
    st=$(kaggle kernels status "$id" 2>&1)
    case "$st" in
      *COMPLETE*) log "complete $id"; break ;;
      *ERROR*|*CANCEL*) log "KERNEL FAILED $id: $st -- lane stops"; exit 1 ;;
      *) sleep "$POLL_S" ;;
    esac
  done

  mkdir -p "$out"
  kaggle kernels output "$id" -p "$out" >> "$LOG" 2>&1
  if [ ! -f "$out/$ckpt" ]; then
    log "MISSING $out/$ckpt after fetch -- lane stops"; exit 1
  fi
  log "fetched $out/$ckpt; scoring in background"
  # The exact protocol of runs/task48-6-kaggle/control_rescore.json, so every result compares
  # with scripts/compare_v4_scores.py against that control, the s1 replicate, or each other.
  ( while [ "$(pgrep -f '.venv/bin/fintfm-v4protocol' | wc -l)" -ge "$MAX_SCORING" ]; do
      sleep 60
    done
    PYTHONUNBUFFERED=1 nice -n 5 uv run fintfm-v4protocol --model "$out/$ckpt" --horizon 0 \
      --folds 0,1,2,3,4 --no-boosting --classical logistic_regression --device cpu \
      --out "$out/score.json" > "$out/score.log" 2>&1
    log "scored $id EXIT=$?" ) &
done
wait
log "lane done"
