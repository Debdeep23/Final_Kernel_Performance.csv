# Final_Kernel_Performance.csv

Got you. Here’s a clean, **bullet-proof** set of steps that (1) avoids the broken-symlink mess and (2) produces the `*_with_counts.csv` correctly. Run everything **from the repo root** (`~/gpu-perf`).

---

# 0) Clean slate (optional but recommended)

```bash
cd ~/gpu-perf
rm -f data/runs_4070.csv data/runs_4070_with_counts.csv data/runs_4070_final.csv
rm -f data/trials_*__4070.csv
```

---

# 1) Make a **safe** symlink to calibration (no more broken links)

Use an **absolute path** for the target so the link never breaks due to `cwd`.

```bash
cd ~/gpu-perf
# sanity: must exist and be non-empty
test -s data/device_calibration_4070.json

# make a safe absolute symlink
ln -sfn "$(pwd)/data/device_calibration_4070.json" data/device_calibration.json

# verify
ls -l data/device_calibration*.json
readlink -f data/device_calibration.json
jq -r '.device_name,.sustained_mem_bandwidth_gbps,.sustained_compute_gflops' data/device_calibration.json
```

You should see:

* the symlink pointing to an **absolute** path
* the device name “NVIDIA GeForce RTX 4070”
* non-zero GB/s and GFLOP/s

---

# 2) Build the runner

```bash
nvcc -std=c++14 -O3 --ptxas-options=-v -lineinfo -arch=sm_89 -DTILE=32 \
  -o bin/runner runner/main.cu > data/build_4070.log 2>&1 || tail -120 data/build_4070.log

test -x bin/runner
```

---

# 3) Run the full 4070 suite (creates the trial CSVs)

```bash
chmod +x scripts/run_trials.sh
bash scripts/run_suite_4070.sh

# verify the per-kernel trial files exist
ls -l data/trials_*__4070.csv | wc -l
head -3 data/trials_vector_add__4070.csv
```

You should see **15** trial files (one per kernel), each with 10 rows.

---

# 4) Aggregate trials → per-kernel means/std → `runs_4070.csv`

```bash
python3 scripts/aggregate_trials.py data/trials_*__4070.csv > data/runs_4070.csv

# quick peek
head -5 data/runs_4070.csv
```

---

# 5) Join with static counts → `runs_4070_with_counts.csv`

This is where you previously hit the error—make sure to pass **three args** (no redirection).

```bash
# kernels_static_4070.csv must exist (it was created earlier by the static-counts step)
test -s data/kernels_static_4070.csv

python3 scripts/join_runs_and_counts.py \
  data/runs_4070.csv \
  data/kernels_static_4070.csv \
  data/runs_4070_with_counts.csv

# verify
head -3 data/runs_4070_with_counts.csv
```

You should now see columns like `kernel,args,regs,shmem,block,grid,rows,cols,matN,H,W,FLOPs,BYTES,ai,...`

---

# 6) Add single-thread baseline & modeled speedup → `runs_4070_final.csv`

This computes the “relative to a single thread” metric by scaling sustained ceilings by warp size (32). No extra runs needed.

```bash
python3 scripts/add_singlethread_baseline.py \
  data/runs_4070_with_counts.csv \
  data/device_calibration_4070.json \
  data/runs_4070_final.csv

# verify the two extra columns exist
head -1 data/runs_4070_final.csv | tr ',' '\n' | nl | sed -n '1,120p'
tail -5 data/runs_4070_final.csv
```

You should see new columns: `T1_model_ms` and `speedup_model`.

---

## Common gotchas (and the fixes we applied)

* **Broken symlink**: happens if you do `ln -s data/device_calibration_4070.json data/device_calibration.json` from a different directory.
  **Fix used**: `ln -sfn "$(pwd)/data/device_calibration_4070.json" data/device_calibration.json` (absolute path).

* **Empty `*_with_counts.csv` or script error**: occurs if you redirect stdout instead of passing the 3rd arg.
  **Fix used**: call

  ```bash
  python3 scripts/join_runs_and_counts.py data/runs_4070.csv data/kernels_static_4070.csv data/runs_4070_with_counts.csv
  ```

  (three positional args; no `>`).

* **No trial CSVs**: ensure `scripts/run_trials.sh` is executable and `run_suite_4070.sh` actually calls it—your current version does, and you verified `bin/runner` prints timings.

---

### Final artifact checklist

After the steps above, you should have:

* `data/device_calibration.json` → valid symlink to `device_calibration_4070.json`
* `data/trials_*__4070.csv` → raw per-trial timings (10 rows each)
* `data/runs_4070.csv` → kernel-level mean/std from trials
* `data/kernels_static_4070.csv` → static counts + launch config
* `data/runs_4070_with_counts.csv` → joined runs + counts
* `data/runs_4070_final.csv` → plus single-thread baseline & modeled speedup

If anything doesn’t appear, paste the command output and I’ll zero in fast.
