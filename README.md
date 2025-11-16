

using the scripts you currently have:

* `scripts/run_suite_4070.sh`
* `scripts/run_trials.sh`
* `scripts/static_counts.py`
* `scripts/add_singlethread_baseline.py`
* `scripts/verify_calibration.py`

It produces these final artifacts:

* `data/device_calibration_4070.json` (and a correct non-broken link)
* `data/runs_4070.csv` (10 trials per kernel, averaged)
* `data/runs_4070_with_counts.csv` (adds FLOPs/BYTES)
* `data/runs_4070_final.csv` (adds single-thread baseline + modeled speedup)

---

# A. Clean slate (keep calibration + build logs)

Run this from repo root `~/gpu-perf`:

```bash
# 1) Keep only the essentials; delete old CSV noise
find data -maxdepth 1 -type f \
  \( -name 'trials_*__*.csv' -o -name 'runs_*.csv' -o -name 'runs_agg_*.csv' \) \
  -delete

# 2) Keep calibration outputs and logs; show what's left
ls -l data
```

You should still see (examples):

```
device_calibration_4070.json
props_4070.out
stream_like_4070.out
gemm_cublas_4070.out
ptxas_4070.log
```

---

# B. Fix the symlink once (no more “broken link”)

**Important:** Create the symlink **from inside `data/`** so the target path is correct.

```bash
cd data
ln -sfn device_calibration_4070.json device_calibration.json
ls -l device_calibration_4070.json device_calibration.json
# sanity
jq -r '.device_name,
       .sustained_mem_bandwidth_gbps,
       .sustained_compute_gflops' device_calibration.json
cd ..
```

You should see:

```
NVIDIA GeForce RTX 4070
<non-zero GB/s>
<non-zero GFLOP/s>
```

> If either number is 0.0, the `stream_like_4070.out` / `gemm_cublas_4070.out` are missing their “SUSTAINED_” lines. Fix with:
>
> ```bash
> bw=$(awk -F'=' '/GBps=/{print $NF}' data/stream_like_4070.out | sort -nr | head -1)
> fl=$(awk -F'=' '/GFLOPS=/{print $NF}' data/gemm_cublas_4070.out | sort -nr | head -1)
> grep -q SUSTAINED_MEM_BW_GBPS data/stream_like_4070.out  || echo "SUSTAINED_MEM_BW_GBPS=$bw"  >> data/stream_like_4070.out
> grep -q SUSTAINED_COMPUTE_GFLOPS data/gemm_cublas_4070.out || echo "SUSTAINED_COMPUTE_GFLOPS=$fl" >> data/gemm_cublas_4070.out
> # re-write JSON
> python3 scripts/verify_calibration.py \
>   --props  data/props_4070.out \
>   --stream data/stream_like_4070.out \
>   --gemm   data/gemm_cublas_4070.out \
>   --out    data/device_calibration_4070.json
> (cd data && ln -sfn device_calibration_4070.json device_calibration.json)
> jq -r '.device_name,.sustained_mem_bandwidth_gbps,.sustained_compute_gflops' data/device_calibration.json
> ```

---

# C. Build and run all kernels (creates the per-kernel trial CSVs and the aggregated `runs_4070.csv`)

```bash
# Build + run everything; this script compiles runner and runs 10 trials/kernel
bash scripts/run_suite_4070.sh

# Confirm trials got created for the 4070 tag and the aggregate files exist
ls -l data/trials_*__*.csv
ls -l data/runs_4070.csv data/runs_agg_4070.csv

# Quick peek
head -5 data/runs_4070.csv
```

If the script prints only the compile step and stops, your `run_suite_4070.sh` likely has `set -e` and failed silently on a later command. Run with trace to see the exact line:

```bash
bash -x scripts/run_suite_4070.sh
```

Common fixes:

* Ensure `scripts/run_trials.sh` is executable: `chmod +x scripts/run_trials.sh`
* Make sure `bin/runner` exists: the script compiles it; if not, try compiling once manually:

  ```bash
  nvcc -std=c++14 -O3 --ptxas-options=-v -lineinfo -arch=sm_89 -DTILE=32 \
    -o bin/runner runner/main.cu 2> data/ptxas_4070.log
  test -x bin/runner
  ```

---

# D. Add static FLOPs/BYTES to each row (no extra “join” file needed)

Your `scripts/static_counts.py` reads `runs_4070.csv` and writes to **stdout**. Redirect it:

```bash
python3 scripts/static_counts.py \
  data/runs_4070.csv > data/runs_4070_with_counts.csv

# sanity
head -3 data/runs_4070_with_counts.csv
```

You should now see additional columns like `FLOPs` and `BYTES` appended to each kernel row.

> Note: You **do not** need `kernels_static_4070.csv` or `join_runs_and_counts.py` for this pipeline. We’re keeping it simple.

---

# E. Add the single-thread baseline + modeled speedup (this creates your final CSV)

```bash
python3 scripts/add_singlethread_baseline.py \
  data/runs_4070_with_counts.csv \
  data/device_calibration_4070.json \
  data/runs_4070_final.csv

# sanity
head -3 data/runs_4070_final.csv
tail -3 data/runs_4070_final.csv
```

You’ll now have two extra columns:

* `T1_model_ms`  — modeled runtime for a **single active lane** (we scale device ceilings by warp size = 32; you **don’t** re-run kernels)
* `speedup_model` — `T1_model_ms / mean_ms` for each kernel row

This exactly satisfies the “relative to a single thread” requirement **without** adding more kernel runs.

---

## What to keep vs. what to delete (after it all works)

Keep:

* `data/device_calibration_4070.json`
* `data/device_calibration.json` (the **correct** symlink made inside `data/`)
* `data/props_4070.out`, `data/stream_like_4070.out`, `data/gemm_cublas_4070.out`
* `data/ptxas_4070.log` (useful for audits)
* `data/runs_4070.csv`
* `data/runs_4070_with_counts.csv`
* `data/runs_4070_final.csv`  ← **your final deliverable**

Safe to delete:

* `data/trials_*__*.csv` (raw trials) after you trust `runs_4070.csv`
* Any older `runs_agg_*.csv` from previous attempts

---

## Quick “all good?” checklist

```bash
# link points to the right file (no "broken symbolic link")
ls -l data/device_calibration*.json
file data/device_calibration.json
jq -r '.device_name,.sustained_mem_bandwidth_gbps,.sustained_compute_gflops' data/device_calibration.json

# aggregated runs exist and are non-empty
wc -l data/runs_4070.csv data/runs_4070_with_counts.csv data/runs_4070_final.csv

# sample final lines
tail -5 data/runs_4070_final.csv
```

If any step fails, copy-paste the exact failing command + one-line error and I’ll zero in on it. This flow avoids the earlier symlink pitfall and doesn’t rely on the separate `kernels_static_4070.csv` join step, so it’s tight and repeatable.
