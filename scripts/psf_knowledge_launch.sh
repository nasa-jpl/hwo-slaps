#!/bin/bash
set -euo pipefail

PY=/data/home/gvassilakis/Software/miniconda3/envs/hwo-slaps/bin/python
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISPATCHER="$REPO_ROOT/scripts/nonlinear_validation_dispatch.sh"

log() {
  printf '%s [launch] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

usage() {
  log "usage: psf_knowledge_launch.sh <prep|fleet|all> <campaigns_root> <gpu_list> --ladder-root <dir> [--workers-per-gpu N --allow-packing] [--allow-shared-gpus]"
}

if [ "$#" -lt 4 ]; then
  usage
  exit 2
fi

MODE="$1"
CAMPAIGNS_ROOT="$2"
GPU_LIST="$3"
shift 3

case "$MODE" in
  prep|fleet|all) ;;
  *)
    log "unknown mode: $MODE"
    usage
    exit 2
    ;;
esac

LADDER_ROOT=""
WORKERS_PER_GPU=1
ALLOW_PACKING=0
ALLOW_SHARED_GPUS=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --ladder-root)
      if [ "$#" -lt 2 ]; then
        log "--ladder-root requires a directory"
        exit 2
      fi
      LADDER_ROOT="$2"
      shift 2
      ;;
    --workers-per-gpu)
      if [ "$#" -lt 2 ]; then
        log "--workers-per-gpu requires a value"
        exit 2
      fi
      WORKERS_PER_GPU="$2"
      shift 2
      ;;
    --allow-packing)
      ALLOW_PACKING=1
      shift
      ;;
    --allow-shared-gpus)
      ALLOW_SHARED_GPUS=1
      shift
      ;;
    *)
      log "unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

if [ ! -d "$CAMPAIGNS_ROOT" ]; then
  log "campaigns root does not exist: $CAMPAIGNS_ROOT"
  exit 2
fi
if [ -z "$LADDER_ROOT" ] || [ ! -d "$LADDER_ROOT" ]; then
  log "--ladder-root is required and must be a directory"
  exit 2
fi
CAMPAIGNS_ROOT="$(cd "$CAMPAIGNS_ROOT" && pwd)"
LADDER_ROOT="$(cd "$LADDER_ROOT" && pwd)"

if ! [[ "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
  log "--workers-per-gpu must be a positive integer"
  exit 2
fi
if [ "$WORKERS_PER_GPU" -gt 1 ] && [ "$ALLOW_PACKING" -eq 0 ]; then
  log "--workers-per-gpu above 1 requires --allow-packing"
  exit 2
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [ "${#GPUS[@]}" -eq 0 ]; then
  log "gpu list must contain at least one GPU"
  exit 2
fi
if [ "${#GPUS[@]}" -gt 4 ]; then
  log "the PSF knowledge block allows at most four GPUs"
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if ! [[ "$gpu" =~ ^(0|[1-9][0-9]*)$ ]]; then
    log "gpu list entries must be canonical non-negative integers: $gpu"
    exit 2
  fi
done
UNIQUE_GPU_COUNT="$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l | tr -d ' ')"
if [ "$UNIQUE_GPU_COUNT" -ne "${#GPUS[@]}" ]; then
  log "gpu list repeats a GPU; packing is expressed only through --workers-per-gpu"
  exit 2
fi

FISHER="$CAMPAIGNS_ROOT/psf_knowledge_fisher_v1"
NONLINEAR="$CAMPAIGNS_ROOT/psf_knowledge_nonlinear_v1"
STATE="$CAMPAIGNS_ROOT/psf_knowledge_state.json"
LOCK_PATH="$CAMPAIGNS_ROOT/.psf_knowledge_launch.lock"
HEAD=""
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EXPANDED_GPUS=""

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  log "could not acquire exclusive launch lock: $LOCK_PATH"
  exit 2
fi

preflight_fail() {
  log "preflight failed: $*"
  exit 2
}

if [ ! -x "$PY" ]; then
  preflight_fail "environment Python is missing or not executable: $PY"
fi
if [ ! -x "$DISPATCHER" ]; then
  preflight_fail "dispatcher is missing or not executable: $DISPATCHER"
fi
if ! HEAD="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)"; then
  preflight_fail "could not read the repository HEAD"
fi
if ! TREE_STATUS="$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)"; then
  preflight_fail "could not inspect repository status"
fi
if [ -n "$TREE_STATUS" ]; then
  preflight_fail "repository tree is dirty"
fi
if ! FREEZE_CHECK="$(
  cd "$REPO_ROOT"
  "$PY" -c 'from hwoslaps.campaign.design_freeze import load_design_freeze; f = load_design_freeze(); assert f["freeze"]["version"] == 5, f["freeze"]["version"]' 2>&1
)"; then
  preflight_fail "design freeze did not load at version 5: $FREEZE_CHECK"
fi
if [ "${HWOSLAPS_NAUTILUS_TRAINING_WORKERS+x}" = x ]; then
  preflight_fail "HWOSLAPS_NAUTILUS_TRAINING_WORKERS is already set"
fi
if ! AVAILABLE_GB="$(df -BG --output=avail "$CAMPAIGNS_ROOT" | awk 'NR == 2 {gsub(/G/, "", $1); print $1}')"; then
  preflight_fail "could not read free space for $CAMPAIGNS_ROOT"
fi
if ! [[ "$AVAILABLE_GB" =~ ^[0-9]+$ ]] || [ "$AVAILABLE_GB" -le 100 ]; then
  preflight_fail "free space is ${AVAILABLE_GB:-unknown} GB, need above 100 GB"
fi
case "$CAMPAIGNS_ROOT" in
  /nfs/*) preflight_fail "campaigns root $CAMPAIGNS_ROOT is on NFS; use /data" ;;
esac
PGREP_COUNT="$(pgrep -fc nonlinear_validation_dispatch.sh || true)"
if ! [[ "$PGREP_COUNT" =~ ^[0-9]+$ ]] || [ "$PGREP_COUNT" -ne 0 ]; then
  preflight_fail "nonlinear_validation_dispatch.sh process count is ${PGREP_COUNT:-unknown}"
fi
if ! GPU_TABLE="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)"; then
  preflight_fail "nvidia-smi GPU query failed"
fi
for gpu in "${GPUS[@]}"; do
  if ! USED_MIB="$(printf '%s\n' "$GPU_TABLE" | awk -F',' -v requested="$gpu" '$1 + 0 == requested {gsub(/[[:space:]]/, "", $2); print $2; found = 1} END {if (!found) exit 1}')"; then
    preflight_fail "listed GPU $gpu is absent from nvidia-smi"
  fi
  if ! [[ "$USED_MIB" =~ ^[0-9]+$ ]]; then
    preflight_fail "GPU $gpu reported a non-numeric memory value"
  fi
  if [ "$ALLOW_SHARED_GPUS" -eq 0 ] && [ "$USED_MIB" -ge 1024 ]; then
    preflight_fail "GPU $gpu has $USED_MIB MiB in use; pass --allow-shared-gpus to override"
  fi
done
for source_path in \
  "$LADDER_ROOT/ladder_selected_v1/manifest.yaml" \
  "$LADDER_ROOT/ladder_selected_v1/run" \
  "$LADDER_ROOT/ladder_parent_v1/run" \
  "$CAMPAIGNS_ROOT/nonlinear_validation_v1/manifest.json" \
  "$CAMPAIGNS_ROOT/nonlinear_validation_v1/harvest/harvest.json" \
  "$CAMPAIGNS_ROOT/nonlinear_null_v1/harvest/harvest.json"; do
  if [ ! -e "$source_path" ]; then
    preflight_fail "required source is missing: $source_path"
  fi
done
for campaign_dir in "$FISHER" "$NONLINEAR"; do
  manifest="$campaign_dir/manifest.json"
  if [ -f "$manifest" ]; then
    if ! manifest_head="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["code_revision"]["git_hash"])' "$manifest" 2>/dev/null)"; then
      preflight_fail "could not read manifest revision: $manifest"
    fi
    if [ "$manifest_head" != "$HEAD" ]; then
      preflight_fail "$manifest records git hash $manifest_head, expected $HEAD"
    fi
  fi
done

for gpu in "${GPUS[@]}"; do
  for ((worker=1; worker<=WORKERS_PER_GPU; worker++)); do
    if [ -n "$EXPANDED_GPUS" ]; then
      EXPANDED_GPUS+=","
    fi
    EXPANDED_GPUS+="$gpu"
  done
done

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

state_init() {
  "$PY" - "$STATE" init "$$" "$MODE" "$STARTED" "$HEAD" \
    "$CAMPAIGNS_ROOT" "$GPU_LIST" "$WORKERS_PER_GPU" "$LADDER_ROOT" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
state = {
    "launcher_pid": int(sys.argv[3]),
    "mode": sys.argv[4],
    "started": sys.argv[5],
    "head": sys.argv[6],
    "campaigns_root": sys.argv[7],
    "gpu_list": sys.argv[8],
    "workers_per_gpu": int(sys.argv[9]),
    "ladder_root": sys.argv[10],
    "steps": [],
    "status": "RUNNING",
}
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(
    json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
os.replace(temporary, path)
PY
}

state_step_start() {
  "$PY" - "$STATE" step_start "$1" "$2" "$3" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
state = json.loads(path.read_text(encoding="utf-8"))
state["steps"].append({
    "step": sys.argv[3],
    "campaign": None if sys.argv[4] == "-" else sys.argv[4],
    "phase": None if sys.argv[5] == "-" else sys.argv[5],
    "started": sys.argv[6],
    "finished": None,
    "rc": None,
})
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(
    json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
os.replace(temporary, path)
PY
}

state_step_finish() {
  "$PY" - "$STATE" step_finish "$1" "$2" "$3" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$4" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
state = json.loads(path.read_text(encoding="utf-8"))
campaign = None if sys.argv[4] == "-" else sys.argv[4]
phase = None if sys.argv[5] == "-" else sys.argv[5]
for step in reversed(state["steps"]):
    if (
        step["step"] == sys.argv[3]
        and step["campaign"] == campaign
        and step["phase"] == phase
        and step["finished"] is None
    ):
        step["finished"] = sys.argv[6]
        step["rc"] = int(sys.argv[7])
        break
else:
    raise SystemExit("state step to finish was not found")
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(
    json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
os.replace(temporary, path)
PY
}

state_status() {
  "$PY" - "$STATE" status "$1" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
state = json.loads(path.read_text(encoding="utf-8"))
state["status"] = sys.argv[3]
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(
    json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
os.replace(temporary, path)
PY
}

state_exit() {
  "$PY" - "$STATE" "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
state = json.loads(path.read_text(encoding="utf-8"))
exit_rc = int(sys.argv[2])
finished = sys.argv[3]
if state["status"] == "RUNNING":
    state["status"] = "ABORTED"
    for step in state["steps"]:
        if step.get("finished") is None:
            step["finished"] = finished
            step["rc"] = exit_rc
elif state["status"] not in {"SMOKES_READY", "COMPLETE", "INCOMPLETE"}:
    raise ValueError(f"unexpected launcher state status {state['status']!r}")
state["exit_rc"] = exit_rc
state["finished"] = finished
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(
    json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
os.replace(temporary, path)
PY
}

on_exit() {
  local saved=$?
  set +e
  if [ -f "$STATE" ]; then
    state_exit "$saved"
  fi
  exit "$saved"
}

run_logged_step() {
  local step="$1"; local campaign="$2"; local phase="$3"
  shift 3
  state_step_start "$step" "$campaign" "$phase"
  log "start step=$step campaign=$campaign phase=$phase"
  local -a pipeline_status
  if "$@" 2>&1 | while IFS= read -r line; do
    log "$line"
  done; then
    pipeline_status=("${PIPESTATUS[@]}")
  else
    pipeline_status=("${PIPESTATUS[@]}")
  fi
  local rc="${pipeline_status[0]}"
  state_step_finish "$step" "$campaign" "$phase" "$rc"
  log "finish step=$step campaign=$campaign phase=$phase rc=$rc"
  return "$rc"
}

run_dispatch() {
  local campaign="$1"; local phase="$2"; local campaign_name="$3"
  local dispatch_log="$campaign/${phase}_dispatch.log"
  printf '%s [launch] dispatcher invocation campaign=%s phase=%s gpus=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$campaign_name" "$phase" \
    "$EXPANDED_GPUS" >> "$dispatch_log"
  printf '0\n' > "$campaign/.${phase}_cursor"
  rm -f "$campaign/sentinels/${phase}_PHASE_COMPLETE"
  if [ "$phase" = "smokes" ] || [ "$phase" = "maps_smokes" ]; then
    # An approval covers the smoke artifacts it was written against; a new
    # smoke dispatch invalidates it so the fleet gate needs a fresh review.
    rm -f "$campaign/SMOKES_APPROVED"
  fi
  state_step_start "dispatch" "$campaign_name" "$phase"
  log "start step=dispatch campaign=$campaign_name phase=$phase"
  set +e
  "$DISPATCHER" "$campaign" "$phase" "$EXPANDED_GPUS" 2>&1 9>&- |
    tee -a "$dispatch_log" |
    while IFS= read -r line; do
      log "$line"
    done
  local -a pipeline_status=("${PIPESTATUS[@]}")
  local rc="${pipeline_status[0]}"
  set -e
  state_step_finish "dispatch" "$campaign_name" "$phase" "$rc"
  log "finish step=dispatch campaign=$campaign_name phase=$phase rc=$rc"
  return "$rc"
}

run_generator() {
  local campaign="$1"; local campaign_name="$2"
  if [ -f "$campaign/manifest.json" ]; then
    state_step_start "generate" "$campaign_name" "prep"
    log "already staged campaign=$campaign_name manifest=$campaign/manifest.json"
    state_step_finish "generate" "$campaign_name" "prep" 0
    return 0
  fi
  if [ "$campaign_name" = "psf_knowledge_fisher_v1" ]; then
    run_logged_step "generate" "$campaign_name" "prep" \
      "$PY" scripts/generate_psf_knowledge_campaign.py \
      --campaign psf_knowledge_fisher_v1 \
      --selected-run "$LADDER_ROOT/ladder_selected_v1/run" \
      "$campaign"
  else
    run_logged_step "generate" "$campaign_name" "prep" \
      "$PY" scripts/generate_nonlinear_validation_campaign.py \
      --campaign psf_knowledge_nonlinear_v1 \
      --parent-run "$LADDER_ROOT/ladder_parent_v1/run" \
      --selected-run "$LADDER_ROOT/ladder_selected_v1/run" \
      --positions-source-dir "$CAMPAIGNS_ROOT/nonlinear_validation_v1" \
      --null-source-dir "$CAMPAIGNS_ROOT/nonlinear_null_v1" \
      "$campaign"
  fi
}

print_map_smoke_summary() {
  local campaign="$1"; local campaign_name="$2"
  local queue="$campaign/smokes_queue.txt"
  if [ ! -f "$queue" ]; then
    log "map smoke queue missing campaign=$campaign_name path=$queue"
    return 0
  fi
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    local config ladder delta direction out artifact summary
    read -r config ladder delta direction out <<< "$line"
    for artifact in "$out"/psf_knowledge_map_m*_delta${delta}_dir${direction}.npz; do
      [ -f "$artifact" ] || continue
      if summary="$("$PY" - "$artifact" <<'PY'
import sys
import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as record:
    classes = [str(value) for value in np.atleast_1d(record["rung_classes"])]
    print(
        "map_smoke"
        f" artifact={sys.argv[1]}"
        f" logm={float(record['logm'])}"
        f" classes={classes!r}"
        f" production_cells={int(record['production_cells'])}"
        f" matched_cells={int(record['matched_cells'])}"
        f" mismatch_cells={int(record['mismatch_cells'])}"
        f" spurious_cells={int(record['spurious_cells'])}"
        f" measured_draw_rms_nm={float(record['measured_draw_rms_nm'])!r}"
        f" detector_build_seconds={float(record['detector_build_seconds'])!r}"
        f" map_wall_seconds={float(record['map_wall_seconds'])!r}"
    )
PY
      )"; then
        while IFS= read -r summary; do
          log "$summary"
        done <<< "$summary"
      else
        log "could not read map smoke artifact=$artifact"
      fi
    done
  done < "$queue"
}

print_fit_smoke_summary() {
  local campaign="$1"; local campaign_name="$2"
  local queue="$campaign/smokes_queue.txt"
  if [ ! -f "$queue" ]; then
    log "fit smoke queue missing campaign=$campaign_name path=$queue"
    return 0
  fi
  while read -r config positions arm output_dir direction; do
    [ -n "${config:-}" ] || continue
    local artifact
    if [ -n "${direction:-}" ]; then
      artifact="$output_dir/nonlinear_validation_${arm}_dir${direction}.json"
    else
      artifact="$output_dir/nonlinear_validation_${arm}.json"
    fi
    if [ ! -f "$artifact" ]; then
      log "fit smoke artifact missing campaign=$campaign_name path=$artifact"
      continue
    fi
    local summary
    if summary="$("$PY" - "$artifact" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
delta = payload.get("fit_psf_delta")
fields = [
    f"fit_smoke artifact={sys.argv[1]}",
    f"system={payload.get('system_id')}",
    f"arm={payload.get('arm')}",
    f"q_fit={payload.get('q_fit')!r}",
    f"dlogZ={payload.get('delta_log_evidence')!r}",
    f"smooth_status={payload.get('smooth_status')!r}",
    f"subhalo_status={payload.get('subhalo_status')!r}",
    f"quality_flags={payload.get('quality_flags')!r}",
    f"fit_pair_s={(payload.get('timings') or {}).get('fit_pair_s')!r}",
]
if isinstance(delta, dict):
    fields.extend([
        f"fit_psf_delta_amplitude={delta.get('amplitude_rms_nm')!r}",
        f"direction={delta.get('direction')!r}",
        f"measured_draw_rms_nm={delta.get('measured_draw_rms_nm')!r}",
    ])
print(" ".join(fields))
PY
    )"; then
      while IFS= read -r summary; do
        log "$summary"
      done <<< "$summary"
    else
      log "could not read fit smoke artifact=$artifact"
    fi
  done < "$queue"
}

PREP_RC=0
run_prep() {
  local worst=0
  local fisher_ready=1
  local nonlinear_ready=1
  if run_generator "$FISHER" "psf_knowledge_fisher_v1"; then
    :
  else
    local rc=$?
    fisher_ready=0
    [ "$rc" -gt "$worst" ] && worst="$rc"
  fi
  if run_generator "$NONLINEAR" "psf_knowledge_nonlinear_v1"; then
    :
  else
    local rc=$?
    nonlinear_ready=0
    [ "$rc" -gt "$worst" ] && worst="$rc"
  fi
  if [ "$fisher_ready" -eq 1 ]; then
    if run_dispatch "$FISHER" "maps_smokes" "psf_knowledge_fisher_v1"; then
      :
    else
      local rc=$?
      [ "$rc" -gt "$worst" ] && worst="$rc"
    fi
  else
    log "skipping Fisher map smokes after generation failure"
  fi
  if [ "$nonlinear_ready" -eq 1 ]; then
    if run_dispatch "$NONLINEAR" "smokes" "psf_knowledge_nonlinear_v1"; then
      :
    else
      local rc=$?
      [ "$rc" -gt "$worst" ] && worst="$rc"
    fi
  else
    log "skipping nonlinear smokes after generation failure"
  fi
  print_map_smoke_summary "$FISHER" "psf_knowledge_fisher_v1"
  print_fit_smoke_summary "$NONLINEAR" "psf_knowledge_nonlinear_v1"
  log "approval: printf '%s\\n' '<reviewer, date, reason>' > '$FISHER/SMOKES_APPROVED'"
  log "approval: printf '%s\\n' '<reviewer, date, reason>' > '$NONLINEAR/SMOKES_APPROVED'"
  log "SMOKES_READY"
  if [ "$worst" -eq 0 ]; then
    state_status "SMOKES_READY"
  else
    state_status "INCOMPLETE"
  fi
  PREP_RC="$worst"
  return 0
}

smoke_gate() {
  if [ ! -f "$FISHER/sentinels/maps_smokes_PHASE_COMPLETE" ]; then
    log "fleet gate missing $FISHER/sentinels/maps_smokes_PHASE_COMPLETE"
    return 3
  fi
  if [ ! -s "$FISHER/SMOKES_APPROVED" ]; then
    log "fleet gate missing non-empty $FISHER/SMOKES_APPROVED"
    return 3
  fi
  if [ ! -f "$NONLINEAR/sentinels/smokes_PHASE_COMPLETE" ]; then
    log "fleet gate missing $NONLINEAR/sentinels/smokes_PHASE_COMPLETE"
    return 3
  fi
  if [ ! -s "$NONLINEAR/SMOKES_APPROVED" ]; then
    log "fleet gate missing non-empty $NONLINEAR/SMOKES_APPROVED"
    return 3
  fi
  return 0
}

LAST_PLAIN_HARVEST_RC=0
LAST_HARVEST_RC=0
run_harvest_fisher() {
  local plain_rc=0; local fallback_rc=0
  if run_logged_step "harvest_plain" "psf_knowledge_fisher_v1" "harvest" \
    "$PY" scripts/harvest_psf_knowledge.py "$FISHER"; then
    :
  else
    plain_rc=$?
    log "plain Fisher harvest failed rc=$plain_rc; retrying with --allow-incomplete"
    if run_logged_step "harvest_allow_incomplete" \
      "psf_knowledge_fisher_v1" "harvest" \
      "$PY" scripts/harvest_psf_knowledge.py "$FISHER" \
      --allow-incomplete; then
      :
    else
      fallback_rc=$?
    fi
  fi
  LAST_PLAIN_HARVEST_RC="$plain_rc"
  LAST_HARVEST_RC="$plain_rc"
  [ "$fallback_rc" -gt "$LAST_HARVEST_RC" ] && LAST_HARVEST_RC="$fallback_rc"
  return "$plain_rc"
}

run_harvest_nonlinear() {
  local plain_rc=0; local fallback_rc=0
  if run_logged_step "harvest_plain" "psf_knowledge_nonlinear_v1" "harvest" \
    "$PY" scripts/harvest_nonlinear_validation.py "$NONLINEAR"; then
    :
  else
    plain_rc=$?
    log "plain nonlinear harvest failed rc=$plain_rc; retrying with --allow-incomplete"
    if run_logged_step "harvest_allow_incomplete" \
      "psf_knowledge_nonlinear_v1" "harvest" \
      "$PY" scripts/harvest_nonlinear_validation.py "$NONLINEAR" \
      --allow-incomplete; then
      :
    else
      fallback_rc=$?
    fi
  fi
  LAST_PLAIN_HARVEST_RC="$plain_rc"
  LAST_HARVEST_RC="$plain_rc"
  [ "$fallback_rc" -gt "$LAST_HARVEST_RC" ] && LAST_HARVEST_RC="$fallback_rc"
  return "$plain_rc"
}

run_fleet() {
  local worst=0
  if smoke_gate; then
    :
  else
    local gate_rc=$?
    state_status "INCOMPLETE"
    return "$gate_rc"
  fi
  if run_dispatch "$FISHER" "maps" "psf_knowledge_fisher_v1"; then
    :
  else
    local rc=$?
    [ "$rc" -gt "$worst" ] && worst="$rc"
    log "continuing to nonlinear fits after Fisher maps rc=$rc"
  fi
  if run_dispatch "$NONLINEAR" "fits" "psf_knowledge_nonlinear_v1"; then
    :
  else
    local rc=$?
    [ "$rc" -gt "$worst" ] && worst="$rc"
  fi
  if run_harvest_fisher; then
    :
  else
    local rc=$LAST_HARVEST_RC
    [ "$rc" -gt "$worst" ] && worst="$rc"
  fi
  if run_harvest_nonlinear; then
    :
  else
    local rc=$LAST_HARVEST_RC
    [ "$rc" -gt "$worst" ] && worst="$rc"
  fi
  rm -f "$CAMPAIGNS_ROOT/psf_knowledge_COMPLETE" \
    "$CAMPAIGNS_ROOT/psf_knowledge_INCOMPLETE"
  if [ "$worst" -eq 0 ]; then
    touch "$CAMPAIGNS_ROOT/psf_knowledge_COMPLETE"
    state_status "COMPLETE"
  else
    touch "$CAMPAIGNS_ROOT/psf_knowledge_INCOMPLETE"
    state_status "INCOMPLETE"
  fi
  return "$worst"
}

wait_for_smoke_approval() {
  local last_log
  last_log="$(date +%s)"
  log "waiting for non-empty SMOKES_APPROVED in both campaign directories"
  while [ ! -s "$FISHER/SMOKES_APPROVED" ] || \
        [ ! -s "$NONLINEAR/SMOKES_APPROVED" ]; do
    sleep 60
    local now
    now="$(date +%s)"
    if [ "$((now - last_log))" -ge 1800 ]; then
      log "still waiting for both SMOKES_APPROVED sentinels"
      last_log="$now"
    fi
  done
  log "both smoke approvals are present"
}

cd "$REPO_ROOT"
state_init
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
case "$MODE" in
  prep)
    run_prep
    exit "$PREP_RC"
    ;;
  fleet)
    if run_fleet; then
      exit 0
    else
      rc=$?
      exit "$rc"
    fi
    ;;
  all)
    run_prep
    if [ "$PREP_RC" -ne 0 ]; then
      log "prep completed with rc=$PREP_RC; approval wait is not started"
      exit "$PREP_RC"
    fi
    wait_for_smoke_approval
    if run_fleet; then
      exit 0
    else
      rc=$?
      exit "$rc"
    fi
    ;;
esac
