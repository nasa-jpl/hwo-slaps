#!/bin/bash
set -u

if [ "$#" -ne 1 ]; then
  printf 'usage: nonlinear_extension_status.sh <campaigns_root>\n'
  exit 0
fi

ROOT="$1"
STATE="$ROOT/nonlinear_extension_state.json"
CAMPAIGNS=(
  "$ROOT/nonlinear_validation100_v1"
  "$ROOT/nonlinear_null_v1"
)

mtime_text() {
  if [ -n "$1" ]; then
    date -u -d "@$1" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null
  else
    printf 'never'
  fi
}

phase_report() {
  local campaign="$1"; local phase="$2"; local queue="$3"
  local total=0; local done=0; local failed=0; local recent=0
  local first=""; local last=""
  local config second arm output_dir extra
  while read -r config second arm output_dir extra; do
    [ -n "${config:-}" ] || continue
    local tag
    if [ "$phase" = "positions" ]; then
      output_dir="$arm"
      tag="$(basename "$output_dir")_positions"
    else
      tag="$(basename "$output_dir")_$arm"
    fi
    local done_path="$campaign/sentinels/$tag.DONE"
    local failed_path="$campaign/sentinels/$tag.FAILED"
    total=$((total + 1))
    if [ -f "$done_path" ]; then
      done=$((done + 1))
      local stamp
      stamp="$(stat -c %Y "$done_path" 2>/dev/null || true)"
      if [[ "$stamp" =~ ^[0-9]+$ ]]; then
        if [ -z "$first" ] || [ "$stamp" -lt "$first" ]; then
          first="$stamp"
        fi
        if [ -z "$last" ] || [ "$stamp" -gt "$last" ]; then
          last="$stamp"
        fi
      fi
      if find "$done_path" -newermt '60 minutes ago' -print -quit 2>/dev/null | grep -q .; then
        recent=$((recent + 1))
      fi
    fi
    if [ -f "$failed_path" ]; then
      failed=$((failed + 1))
    fi
  done < "$queue"
  local remaining=$((total - done))
  local eta
  if [ "$remaining" -eq 0 ]; then
    eta='0 minutes'
  elif [ "$recent" -gt 0 ]; then
    eta="$(((remaining * 60 + recent - 1) / recent)) minutes"
  else
    eta='unknown'
  fi
  local phase_status='incomplete'
  if [ "$done" -eq "$total" ]; then
    phase_status='complete'
  fi
  printf '[%s][%s] status=%s queue=%d DONE=%d FAILED=%d remaining=%d first_DONE=%s last_DONE=%s DONE_last_60m=%d rate=%d_per_hour ETA=%s\n' \
    "$(basename "$campaign")" "$phase" "$phase_status" "$total" "$done" \
    "$failed" "$remaining" "$(mtime_text "$first")" "$(mtime_text "$last")" \
    "$recent" "$recent" "$eta"
}

if [ -f "$STATE" ]; then
  state_status="$(grep -o '"status"[[:space:]]*:[[:space:]]*"[^"]*"' "$STATE" 2>/dev/null | tail -n 1 | sed -E 's/.*"status"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/')"
  state_step="$(grep -o '"step"[[:space:]]*:[[:space:]]*"[^"]*"' "$STATE" 2>/dev/null | tail -n 1 | sed -E 's/.*"step"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/')"
  printf 'state status=%s last_step=%s\n' \
    "${state_status:-unknown}" "${state_step:-unknown}"
else
  printf 'state status=missing last_step=missing\n'
fi

for campaign in "${CAMPAIGNS[@]}"; do
  printf 'campaign=%s\n' "$(basename "$campaign")"
  for phase in positions smokes fits; do
    queue="$campaign/${phase}_queue.txt"
    if [ -f "$queue" ]; then
      phase_report "$campaign" "$phase" "$queue"
    fi
  done
  for marker in SMOKES_APPROVED sentinels/smokes_PHASE_COMPLETE \
    sentinels/fits_PHASE_COMPLETE; do
    if [ -e "$campaign/$marker" ]; then
      printf 'marker %s=present\n' "$marker"
    else
      printf 'marker %s=missing\n' "$marker"
    fi
  done
done

for marker in nonlinear_extension_COMPLETE nonlinear_extension_INCOMPLETE; do
  if [ -e "$ROOT/$marker" ]; then
    printf 'root marker %s=present\n' "$marker"
  else
    printf 'root marker %s=missing\n' "$marker"
  fi
done
exit 0
