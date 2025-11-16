#!/usr/bin/env bash
set -euo pipefail
set -x

# --- dirs & calibration link ---
mkdir -p bin data
test -s data/device_calibration_4070.json
ln -sfn data/device_calibration_4070.json data/device_calibration.json
ls -l data/device_calibration_4070.json data/device_calibration.json

# --- build runner and capture FULL log ---
BUILD_LOG=data/build_4070.log
: > "$BUILD_LOG"
if ! nvcc -std=c++14 -O3 --ptxas-options=-v -lineinfo -arch=sm_89 -DTILE=32 \
      -o bin/runner runner/main.cu >"$BUILD_LOG" 2>&1 ; then
  echo "NVCC failed. Showing last 120 lines of $BUILD_LOG"
  tail -120 "$BUILD_LOG"
  exit 1
fi

# Ensure the binary exists and is executable
if [[ ! -x bin/runner ]]; then
  echo "Build did not produce bin/runner. Showing last 120 lines of $BUILD_LOG"
  tail -120 "$BUILD_LOG"
  exit 1
fi

# --- device tag for filenames ---
DEV_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^ *//;s/ *$//')"
TAG="$(echo "$DEV_NAME" | tr '[:upper:]' '[:lower:]' | sed -E 's/nvidia //; s/geforce //; s/rtx //; s/[ -]//g')"
echo "DEVICE='$DEV_NAME'  TAG='$TAG'"

# --- helper must exist ---
if [[ ! -x scripts/run_trials.sh ]]; then
  echo "scripts/run_trials.sh missing or not executable"; exit 1
fi

# --- run trials: 10 per kernel (each trial repeats via --reps) ---
scripts/run_trials.sh vector_add            "--N 1048576 --block 256 --warmup 20 --reps 100" 12 0 10
scripts/run_trials.sh saxpy                 "--N 1048576 --block 256 --warmup 20 --reps 100" 12 0 10
scripts/run_trials.sh strided_copy_8        "--N 1048576 --block 256 --warmup 20 --reps 100"  8 0 10
scripts/run_trials.sh naive_transpose       "--rows 2048 --cols 2048 --warmup 20 --reps 100"  8 0 10
scripts/run_trials.sh shared_transpose      "--rows 2048 --cols 2048 --warmup 20 --reps 100" 10 4224 10
scripts/run_trials.sh matmul_tiled          "--matN 512 --warmup 10 --reps 50"               37 8192 10
scripts/run_trials.sh matmul_naive          "--matN 512 --warmup 10 --reps 50"               40 0    10
scripts/run_trials.sh reduce_sum            "--N 1048576 --block 256 --warmup 20 --reps 100" 10 1024 10
scripts/run_trials.sh dot_product           "--N 1048576 --block 256 --warmup 20 --reps 100" 15 1024 10
scripts/run_trials.sh histogram             "--N 1048576 --block 256 --warmup 10  --reps 50" 10 1024 10
scripts/run_trials.sh conv2d_3x3            "--H 1024 --W 1024 --warmup 10 --reps 50"        30 0    10
scripts/run_trials.sh conv2d_7x7            "--H 1024 --W 1024 --warmup 10 --reps 50"        40 0    10
scripts/run_trials.sh random_access         "--N 1048576 --block 256 --warmup 20 --reps 100" 10 0    10
scripts/run_trials.sh vector_add_divergent  "--N 1048576 --block 256 --warmup 20 --reps 100" 15 0    10
scripts/run_trials.sh shared_bank_conflict  "--warmup 20 --reps 100"                        206 4096 10
scripts/run_trials.sh atomic_hotspot        "--N 1048576 --block 256 --iters 100 --warmup 10 --reps 50" 7 0 10

# --- verify trials exist for THIS device ---
ls -l data/trials_*__${TAG}.csv

# --- aggregation ---
python3 scripts/aggregate_trials.py data/trials_*__${TAG}.csv > data/runs_agg_4070.csv
cp -f data/runs_agg_4070.csv data/runs_4070.csv

# --- quick peek ---
head -5 data/runs_4070.csv || true
