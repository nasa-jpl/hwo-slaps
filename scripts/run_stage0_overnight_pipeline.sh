#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/home/gvassilakis/Software/miniconda3/envs/hwo-slaps/bin/python}"
WORKERS="${WORKERS:-120}"
WORKER_THREADS="${WORKER_THREADS:-1}"
MAX_NFEV="${MAX_NFEV:-12}"
MAIN_MAX_CASES="${MAIN_MAX_CASES:-160}"
BOOST_MAX_CASES="${BOOST_MAX_CASES:-140}"
MAIN_MAX_FALSE_POSITIVE="${MAIN_MAX_FALSE_POSITIVE:-18}"
BOOST_MAX_FALSE_POSITIVE="${BOOST_MAX_FALSE_POSITIVE:-18}"

MAIN_MANIFEST="${MAIN_MANIFEST:-scratch/study/stage0_rasti_overnight_main_manifest.yaml}"
BOOST_MANIFEST="${BOOST_MANIFEST:-scratch/study/stage0_rasti_overnight_near_threshold_manifest.yaml}"

MAIN_OUT="outputs/stage0_rasti_overnight_main"
BOOST_OUT="outputs/stage0_rasti_overnight_near_threshold"

check_results() {
  local results_csv="$1"
  "$PYTHON_BIN" - "$results_csv" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"Missing results CSV: {path}")
rows = list(csv.DictReader(path.open(newline="")))
bad = [row for row in rows if row.get("status") != "success"]
print(f"{path}: rows={len(rows)} failures={len(bad)}")
if not rows or bad:
    for row in bad[:12]:
        print(f"  failed: {row.get('run_name')} {row.get('error')}")
    raise SystemExit(1)
PY
}

select_cases() {
  local out_dir="$1"
  local tag="$2"
  local max_cases="$3"
  local max_false_positive="$4"

  "$PYTHON_BIN" scripts/select_stage0_nonlinear_cases.py \
    "$out_dir/results.csv" \
    --cases-output "$out_dir/${tag}_nonlinear_cases.txt" \
    --false-positive-output "$out_dir/${tag}_false_positive_cases.txt" \
    --max-cases "$max_cases" \
    --max-false-positive-cases "$max_false_positive"
}

run_validation() {
  local out_dir="$1"
  local tag="$2"
  local validation_out="$3"
  local cases_file="$out_dir/${tag}_nonlinear_cases.txt"
  local fp_file="$out_dir/${tag}_false_positive_cases.txt"

  mapfile -t cases < "$cases_file"
  mapfile -t false_positive_cases < "$fp_file"

  if (( ${#cases[@]} == 0 )); then
    echo "No nonlinear cases selected for $tag; skipping validation."
    return 0
  fi

  local args=(
    scripts/run_stage0_spie_plus_validation.py
    --stage0-results "$out_dir/results.csv"
    --config-dir "$out_dir/generated_configs"
    --output-dir "$validation_out"
    --cases "${cases[@]}"
    --workers "$WORKERS"
    --max-nfev "$MAX_NFEV"
  )

  if (( ${#false_positive_cases[@]} > 0 )); then
    args+=(--include-false-positive --false-positive-cases "${false_positive_cases[@]}")
  else
    args+=(--no-false-positive)
  fi

  CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "${args[@]}"
}

echo "Running Fisher block 1/2: $MAIN_MANIFEST"
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" scripts/run_stage0_study.py \
  --manifest "$MAIN_MANIFEST" \
  --workers "$WORKERS" \
  --worker-threads "$WORKER_THREADS" \
  --no-plots

check_results "$MAIN_OUT/results.csv"

echo "Running Fisher block 2/2: $BOOST_MANIFEST"
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" scripts/run_stage0_study.py \
  --manifest "$BOOST_MANIFEST" \
  --workers "$WORKERS" \
  --worker-threads "$WORKER_THREADS" \
  --no-plots

check_results "$BOOST_OUT/results.csv"

echo "Selecting nonlinear validation cases after all Fisher blocks passed."
select_cases "$MAIN_OUT" "main" "$MAIN_MAX_CASES" "$MAIN_MAX_FALSE_POSITIVE"
select_cases "$BOOST_OUT" "near_threshold" "$BOOST_MAX_CASES" "$BOOST_MAX_FALSE_POSITIVE"

echo "Running nonlinear validation block 1/2."
run_validation "$MAIN_OUT" "main" "outputs/stage0_rasti_overnight_main_nonlinear_validation"

echo "Running nonlinear validation block 2/2."
run_validation "$BOOST_OUT" "near_threshold" "outputs/stage0_rasti_overnight_near_threshold_nonlinear_validation"

echo "Overnight pipeline complete."
