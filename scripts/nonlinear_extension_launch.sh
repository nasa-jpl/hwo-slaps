#!/bin/bash
set -euo pipefail

PY=/data/home/gvassilakis/Software/miniconda3/envs/hwo-slaps/bin/python
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISPATCHER="$REPO_ROOT/scripts/nonlinear_validation_dispatch.sh"

log() {
  printf '%s [launch] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

usage() {
  log "usage: nonlinear_extension_launch.sh <prep|fleet|all> <campaigns_root> <gpu_list> [--workers-per-gpu N] [--allow-shared-gpus]"
}

if [ "$#" -lt 3 ]; then
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

WORKERS_PER_GPU=1
ALLOW_SHARED_GPUS=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --workers-per-gpu)
      if [ "$#" -lt 2 ]; then
        log "--workers-per-gpu requires a value"
        exit 2
      fi
      WORKERS_PER_GPU="$2"
      shift 2
      ;;
    --allow-shared-gpus)
      ALLOW_SHARED_GPUS=1
      shift
      ;;
    *)
      log "unknown argument: $1"
      exit 2
      ;;
  esac
done

if [ ! -d "$CAMPAIGNS_ROOT" ]; then
  log "campaigns root does not exist: $CAMPAIGNS_ROOT"
  exit 2
fi
CAMPAIGNS_ROOT="$(cd "$CAMPAIGNS_ROOT" && pwd)"

if ! [[ "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
  log "--workers-per-gpu must be a positive integer"
  exit 2
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [ "${#GPUS[@]}" -eq 0 ]; then
  log "gpu list must contain at least one GPU"
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    log "gpu list contains a non-negative integer violation: $gpu"
    exit 2
  fi
done

CAMPAIGN_B="$CAMPAIGNS_ROOT/nonlinear_validation100_v1"
CAMPAIGN_A="$CAMPAIGNS_ROOT/nonlinear_null_v1"
STATE="$CAMPAIGNS_ROOT/nonlinear_extension_state.json"
LOCK_PATH="$CAMPAIGNS_ROOT/.nonlinear_extension_launch.lock"
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
  "$PY" -c 'from hwoslaps.campaign.design_freeze import load_design_freeze; f = load_design_freeze(); assert f["freeze"]["version"] == 4, f["freeze"]["version"]' 2>&1
)"; then
  preflight_fail "design freeze did not load at version 4: $FREEZE_CHECK"
fi
if [ "${HWOSLAPS_NAUTILUS_TRAINING_WORKERS+x}" = x ]; then
  preflight_fail "HWOSLAPS_NAUTILUS_TRAINING_WORKERS is already set"
fi
if ! AVAILABLE_GB="$(df -BG --output=avail "$CAMPAIGNS_ROOT" | awk 'NR == 2 {gsub(/G/, "", $1); print $1}')"; then
  preflight_fail "could not read free space for $CAMPAIGNS_ROOT"
fi
if ! [[ "$AVAILABLE_GB" =~ ^[0-9]+$ ]] || [ "$AVAILABLE_GB" -le 500 ]; then
  preflight_fail "free space is ${AVAILABLE_GB:-unknown} GB, need above 500 GB"
fi
# pgrep exits 1 when it prints a count of 0, so the exit status is ignored
# and the printed count is compared numerically.
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
    preflight_fail "GPU $gpu has ${USED_MIB} MiB in use; pass --allow-shared-gpus to override"
  fi
done
for source_dir in \
  "$CAMPAIGNS_ROOT/ladder_parent_v1/run" \
  "$CAMPAIGNS_ROOT/ladder_selected_v1/run" \
  "$CAMPAIGNS_ROOT/ladder_validation_v1/run" \
  "$CAMPAIGNS_ROOT/nonlinear_validation_v1/manifest.json" \
  "$CAMPAIGNS_ROOT/nonlinear_validation_v1/harvest/harvest.json"; do
  if [ ! -e "$source_dir" ]; then
    preflight_fail "required source is missing: $source_dir"
  fi
done
for campaign_dir in "$CAMPAIGN_B" "$CAMPAIGN_A"; do
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
    "$CAMPAIGNS_ROOT" "$GPU_LIST" "$WORKERS_PER_GPU" <<'PY'
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
  if [ "$campaign_name" = "nonlinear_validation100_v1" ]; then
    run_logged_step "generate" "$campaign_name" "prep" \
      "$PY" scripts/generate_nonlinear_validation_campaign.py \
      --campaign nonlinear_validation100_v1 \
      --validation-run "$CAMPAIGNS_ROOT/ladder_validation_v1/run" \
      --pooled-source-dir "$CAMPAIGNS_ROOT/nonlinear_validation_v1" \
      "$campaign"
  else
    run_logged_step "generate" "$campaign_name" "prep" \
      "$PY" scripts/generate_nonlinear_validation_campaign.py \
      --campaign nonlinear_null_v1 \
      --parent-run "$CAMPAIGNS_ROOT/ladder_parent_v1/run" \
      --selected-run "$CAMPAIGNS_ROOT/ladder_selected_v1/run" \
      --positions-source-dir "$CAMPAIGNS_ROOT/nonlinear_validation_v1" \
      "$campaign"
  fi
}

print_smoke_summary() {
  local campaign="$1"; local campaign_name="$2"
  local queue="$campaign/smokes_queue.txt"
  if [ ! -f "$queue" ]; then
    log "smoke queue missing campaign=$campaign_name path=$queue"
    return 0
  fi
  while read -r config positions arm output_dir; do
    [ -n "${config:-}" ] || continue
    local artifact="$output_dir/nonlinear_validation_${arm}.json"
    if [ ! -f "$artifact" ]; then
      log "smoke artifact missing campaign=$campaign_name arm=$arm path=$artifact"
      continue
    fi
    local summary
    if summary="$("$PY" - "$artifact" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
declaration = payload.get("arm_declaration") or {}
fields = [
    f"system={payload.get('system_id')}",
    f"arm={payload.get('arm')}",
    f"q_fit={payload.get('q_fit')!r}",
    f"delta_log_evidence={payload.get('delta_log_evidence')!r}",
    f"smooth_status={payload.get('smooth_status')!r}",
    f"subhalo_status={payload.get('subhalo_status')!r}",
    f"quality_flags={payload.get('quality_flags')!r}",
    f"fit_pair_s={((payload.get('timings') or {}).get('fit_pair_s'))!r}",
    f"noise_seed={payload.get('noise_seed')!r}",
]
if declaration.get("subhalo_in_truth") is True:
    recovery = (payload.get("case") or {}).get("subhalo_recovery")
    if isinstance(recovery, dict):
        for key in sorted(recovery):
            if "log10" in key:
                fields.append(f"{key}={recovery[key]!r}")
print("smoke " + " ".join(fields))
PY
    )"; then
      while IFS= read -r line; do
        log "$line"
      done <<< "$summary"
    else
      log "could not read smoke artifact campaign=$campaign_name path=$artifact"
    fi
  done < "$queue"
}

PREP_RC=0
run_prep() {
  local worst=0
  local b_ready=1; local a_ready=1
  if run_generator "$CAMPAIGN_B" "nonlinear_validation100_v1"; then
    :
  else
    local rc=$?
    b_ready=0
    if [ "$rc" -gt "$worst" ]; then worst="$rc"; fi
  fi
  if run_generator "$CAMPAIGN_A" "nonlinear_null_v1"; then
    :
  else
    local rc=$?
    a_ready=0
    if [ "$rc" -gt "$worst" ]; then worst="$rc"; fi
  fi
  if [ "$b_ready" -eq 1 ]; then
    local positions_ok=1
    if run_dispatch "$CAMPAIGN_B" positions "nonlinear_validation100_v1"; then
      :
    else
      local rc=$?
      positions_ok=0
      if [ "$rc" -gt "$worst" ]; then worst="$rc"; fi
    fi
    if [ "$positions_ok" -eq 1 ]; then
      if run_dispatch "$CAMPAIGN_B" smokes "nonlinear_validation100_v1"; then
        :
      else
        local rc=$?
        if [ "$rc" -gt "$worst" ]; then worst="$rc"; fi
      fi
    else
      log "skipping validation100 smokes after an incomplete positions phase"
    fi
  else
    log "skipping validation100 phases after generation failure"
  fi
  if [ "$a_ready" -eq 1 ]; then
    if run_dispatch "$CAMPAIGN_A" smokes "nonlinear_null_v1"; then
      :
    else
      local rc=$?
      if [ "$rc" -gt "$worst" ]; then worst="$rc"; fi
    fi
  else
    log "skipping nonlinear_null_v1 smoke phase after generation failure"
  fi
  print_smoke_summary "$CAMPAIGN_B" "nonlinear_validation100_v1"
  print_smoke_summary "$CAMPAIGN_A" "nonlinear_null_v1"
  log "approval: printf '%s\\n' '<reviewer, date, reason>' > '$CAMPAIGN_B/SMOKES_APPROVED'"
  log "approval: printf '%s\\n' '<reviewer, date, reason>' > '$CAMPAIGN_A/SMOKES_APPROVED'"
  if [ "$worst" -eq 0 ]; then
    state_status "SMOKES_READY"
  else
    state_status "INCOMPLETE"
  fi
  PREP_RC="$worst"
  return 0
}

smoke_gate() {
  for campaign in "$CAMPAIGN_B" "$CAMPAIGN_A"; do
    if [ ! -f "$campaign/sentinels/smokes_PHASE_COMPLETE" ]; then
      log "fleet gate missing $campaign/sentinels/smokes_PHASE_COMPLETE"
      return 3
    fi
    if [ ! -s "$campaign/SMOKES_APPROVED" ]; then
      log "fleet gate missing non-empty $campaign/SMOKES_APPROVED"
      return 3
    fi
  done
  return 0
}

LAST_PLAIN_HARVEST_RC=0
LAST_HARVEST_RC=0
run_harvest_campaign() {
  local campaign="$1"; local campaign_name="$2"
  local plain_rc=0; local fallback_rc=0
  if run_logged_step "harvest_plain" "$campaign_name" "harvest" \
    "$PY" scripts/harvest_nonlinear_validation.py "$campaign"; then
    :
  else
    plain_rc=$?
    log "plain harvest failed campaign=$campaign_name rc=$plain_rc; retrying with --allow-incomplete"
    if run_logged_step "harvest_allow_incomplete" "$campaign_name" "harvest" \
      "$PY" scripts/harvest_nonlinear_validation.py "$campaign" \
      --allow-incomplete; then
      :
    else
      fallback_rc=$?
    fi
  fi
  LAST_PLAIN_HARVEST_RC="$plain_rc"
  LAST_HARVEST_RC="$plain_rc"
  if [ "$fallback_rc" -gt "$LAST_HARVEST_RC" ]; then
    LAST_HARVEST_RC="$fallback_rc"
  fi
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
  if run_dispatch "$CAMPAIGN_B" fits "nonlinear_validation100_v1"; then
    :
  else
    local rc=$?
    if [ "$rc" -gt "$worst" ]; then worst="$rc"; fi
    log "continuing to nonlinear_null_v1 after validation100 fits rc=$rc"
  fi
  if run_dispatch "$CAMPAIGN_A" fits "nonlinear_null_v1"; then
    :
  else
    local rc=$?
    if [ "$rc" -gt "$worst" ]; then worst="$rc"; fi
  fi
  if run_harvest_campaign "$CAMPAIGN_B" "nonlinear_validation100_v1"; then
    :
  else
    local rc=$LAST_HARVEST_RC
    if [ "$rc" -gt "$worst" ]; then worst="$rc"; fi
  fi
  if run_harvest_campaign "$CAMPAIGN_A" "nonlinear_null_v1"; then
    :
  else
    local rc=$LAST_HARVEST_RC
    if [ "$rc" -gt "$worst" ]; then worst="$rc"; fi
  fi
  rm -f "$CAMPAIGNS_ROOT/nonlinear_extension_COMPLETE" \
    "$CAMPAIGNS_ROOT/nonlinear_extension_INCOMPLETE"
  if [ "$worst" -eq 0 ]; then
    touch "$CAMPAIGNS_ROOT/nonlinear_extension_COMPLETE"
    state_status "COMPLETE"
  else
    touch "$CAMPAIGNS_ROOT/nonlinear_extension_INCOMPLETE"
    state_status "INCOMPLETE"
  fi
  return "$worst"
}

wait_for_smoke_approval() {
  local last_log
  last_log="$(date +%s)"
  log "waiting for non-empty SMOKES_APPROVED in both campaign directories"
  while [ ! -s "$CAMPAIGN_B/SMOKES_APPROVED" ] || \
        [ ! -s "$CAMPAIGN_A/SMOKES_APPROVED" ]; do
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
