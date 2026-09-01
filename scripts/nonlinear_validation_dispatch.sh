#!/bin/bash
# Flock-queue dispatcher for the nonlinear-validation campaign.
#
# usage: nonlinear_validation_dispatch.sh <campaign_dir> <phase> <gpu>[,<gpu>...]
#   phase: positions | fits
#
# One worker per listed GPU pulls lines off the phase queue under an
# exclusive flock. A job whose artifact already exists is skipped, so
# re-running the dispatcher resumes the campaign. Every job writes a log
# and a DONE/FAILED sentinel; the dispatcher exits nonzero if any job
# failed or any queue line is left unaccounted for.
set -u

CAMPAIGN_DIR="$(cd "$1" && pwd)"; shift
PHASE="$1"; shift
IFS=',' read -r -a GPUS <<< "$1"; shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/data/home/gvassilakis/Software/miniconda3/envs/hwo-slaps/bin/python

# Staged configurations carry repo-root-relative asset paths.
cd "$REPO_ROOT" || exit 2

case "$PHASE" in
  positions) QUEUE="$CAMPAIGN_DIR/positions_queue.txt" ;;
  fits) QUEUE="$CAMPAIGN_DIR/fits_queue.txt" ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac
[ -f "$QUEUE" ] || { echo "missing queue: $QUEUE" >&2; exit 2; }

LOGDIR="$CAMPAIGN_DIR/logs"
SENTDIR="$CAMPAIGN_DIR/sentinels"
mkdir -p "$LOGDIR" "$SENTDIR"
CURSOR="$CAMPAIGN_DIR/.${PHASE}_cursor"
LOCK="$CAMPAIGN_DIR/.${PHASE}_lock"
[ -f "$CURSOR" ] || echo 0 > "$CURSOR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1
export HWOSLAPS_NAUTILUS_TRAINING_WORKERS=4
export HWOSLAPS_CAMPAIGN_UUID="$("$PY" -c "
import json,sys
print(json.load(open('$CAMPAIGN_DIR/manifest.json'))['campaign_uuid'])")"

TOTAL=$(grep -c . "$QUEUE")
echo "[dispatch] phase=$PHASE jobs=$TOTAL gpus=${GPUS[*]} uuid=$HWOSLAPS_CAMPAIGN_UUID"

next_line() {
  flock 9 || return 1
  local n
  n=$(cat "$CURSOR")
  if [ "$n" -ge "$TOTAL" ]; then return 1; fi
  echo $((n + 1)) > "$CURSOR"
  sed -n "$((n + 1))p" "$QUEUE"
}

worker() {
  local gpu="$1"
  while true; do
    local line
    line=$(next_line 9>"$LOCK") || break
    [ -n "$line" ] || continue
    read -r -a FIELDS <<< "$line"
    local out tag
    if [ "$PHASE" = positions ]; then
      out="${FIELDS[2]}"
      tag="$(basename "$out")_positions"
      if [ -f "$out/injection_position.json" ]; then
        echo "[gpu$gpu] skip $tag (artifact exists)"; continue
      fi
    else
      out="${FIELDS[3]}"
      tag="$(basename "$out")_${FIELDS[2]}"
      if [ -f "$out/nonlinear_validation_${FIELDS[2]}.json" ]; then
        echo "[gpu$gpu] skip $tag (artifact exists)"; continue
      fi
    fi
    echo "[gpu$gpu] start $tag"
    local rc
    if [ "$PHASE" = positions ]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" \
        "$REPO_ROOT/scripts/extract_injection_positions.py" \
        "${FIELDS[0]}" "${FIELDS[1]}" "$out" \
        > "$LOGDIR/$tag.log" 2>&1
      rc=$?
    else
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" \
        "$REPO_ROOT/scripts/run_nonlinear_validation.py" \
        "${FIELDS[0]}" "${FIELDS[1]}" "${FIELDS[2]}" "$out" \
        > "$LOGDIR/$tag.log" 2>&1
      rc=$?
    fi
    if [ "$rc" -eq 0 ]; then
      rm -f "$SENTDIR/$tag.FAILED"
      touch "$SENTDIR/$tag.DONE"
      echo "[gpu$gpu] DONE $tag"
    else
      touch "$SENTDIR/$tag.FAILED"
      echo "[gpu$gpu] FAILED $tag rc=$rc (log: $LOGDIR/$tag.log)"
    fi
  done
}

PIDS=()
for gpu in "${GPUS[@]}"; do
  worker "$gpu" &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid"; done

# The exit status follows the artifacts alone: a stale FAILED sentinel
# whose job later succeeded on resume must not fail a complete phase.
MISSING=0
while IFS= read -r line; do
  [ -n "$line" ] || continue
  read -r -a FIELDS <<< "$line"
  if [ "$PHASE" = positions ]; then
    [ -f "${FIELDS[2]}/injection_position.json" ] || MISSING=$((MISSING + 1))
  else
    [ -f "${FIELDS[3]}/nonlinear_validation_${FIELDS[2]}.json" ] \
      || MISSING=$((MISSING + 1))
  fi
done < "$QUEUE"

if [ "$MISSING" -eq 0 ]; then
  touch "$SENTDIR/${PHASE}_PHASE_COMPLETE"
  echo "[dispatch] ${PHASE}_PHASE_COMPLETE ($TOTAL jobs)"
  exit 0
fi
echo "[dispatch] phase=$PHASE INCOMPLETE: missing=$MISSING"
exit 1
