#!/bin/bash
set -u

if [ "$#" -ne 1 ]; then
  printf 'usage: psf_knowledge_status.sh <campaigns_root>\n'
  exit 0
fi

ROOT="$1"
STATE="$ROOT/psf_knowledge_state.json"
FISHER="$ROOT/psf_knowledge_fisher_v1"
NONLINEAR="$ROOT/psf_knowledge_nonlinear_v1"

mtime_text() {
  if [ -n "$1" ]; then
    date -u -d "@$1" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null
  else
    printf 'never'
  fi
}

phase_report() {
  local campaign="$1"; local phase="$2"; local queue="$3"; local kind="$4"
  local total=0; local done=0; local failed=0; local recent=0
  local first=""; local last=""
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    read -r -a fields <<< "$line"
    local tag
    if [ "$kind" = "maps" ]; then
      if [ "${#fields[@]}" -ne 5 ]; then
        continue
      fi
      tag="$(basename "${fields[4]}")_delta${fields[2]}_dir${fields[3]}"
    else
      if [ "${#fields[@]}" -lt 4 ]; then
        continue
      fi
      if [ "${#fields[@]}" -ge 5 ]; then
        tag="$(basename "${fields[3]}")_${fields[2]}_dir${fields[4]}"
      else
        tag="$(basename "${fields[3]}")_${fields[2]}"
      fi
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

printf 'campaign=%s\n' "$(basename "$FISHER")"
for phase in smokes maps; do
  if [ "$phase" = smokes ]; then
    queue="$FISHER/smokes_queue.txt"
  else
    queue="$FISHER/maps_queue.txt"
  fi
  if [ -f "$queue" ]; then
    phase_report "$FISHER" "$phase" "$queue" maps
  fi
done
for marker in SMOKES_APPROVED sentinels/maps_smokes_PHASE_COMPLETE sentinels/maps_PHASE_COMPLETE; do
  if [ -e "$FISHER/$marker" ]; then
    printf 'marker %s=present\n' "$marker"
  else
    printf 'marker %s=missing\n' "$marker"
  fi
done

printf 'campaign=%s\n' "$(basename "$NONLINEAR")"
for phase in smokes fits; do
  queue="$NONLINEAR/${phase}_queue.txt"
  if [ -f "$queue" ]; then
    phase_report "$NONLINEAR" "$phase" "$queue" nonlinear
  fi
done
for marker in SMOKES_APPROVED sentinels/smokes_PHASE_COMPLETE sentinels/fits_PHASE_COMPLETE; do
  if [ -e "$NONLINEAR/$marker" ]; then
    printf 'marker %s=present\n' "$marker"
  else
    printf 'marker %s=missing\n' "$marker"
  fi
done

for marker in psf_knowledge_COMPLETE psf_knowledge_INCOMPLETE; do
  if [ -e "$ROOT/$marker" ]; then
    printf 'root marker %s=present\n' "$marker"
  else
    printf 'root marker %s=missing\n' "$marker"
  fi
done
exit 0
