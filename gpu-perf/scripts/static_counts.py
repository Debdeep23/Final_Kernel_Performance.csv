#!/usr/bin/env python3
import csv, sys, re, math

# --- small CLI-arg parser for our "args" column: turns
#     "--N 1048576 --block 256 --iters 100 --rows 2048 --cols 2048 --H 1024 --W 1024 --matN 512"
#     into a dict {"N":1048576, "block":256, "iters":100, "rows":2048, "cols":2048, "H":1024, "W":1024, "matN":512}
def parse_args(argstr: str):
    toks = argstr.strip().split()
    out = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            key = t[2:]
            val = None
            if i+1 < len(toks) and not toks[i+1].startswith("--"):
                val = toks[i+1]
                i += 1
            if val is not None:
                # try int then float
                try:
                    out[key] = int(val)
                except ValueError:
                    try:
                        out[key] = float(val)
                    except ValueError:
                        out[key] = val
            else:
                out[key] = 1
        i += 1
    return out

# --- helpers for safe extraction with defaults
def gi(d, k, default=0):
    v = d.get(k, default)
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default

def gf(d, k, default=0.0):
    v = d.get(k, default)
    try:
        return float(v)
    except Exception:
        return default

# --- FLOPs/BYTES models (simple but nonzero for all kernels)
def counts_for_row(kernel: str, args: dict):
    # Defaults that match our suite
    N     = gi(args, "N", 0)
    block = gi(args, "block", 256)
    rows  = gi(args, "rows", 0)
    cols  = gi(args, "cols", 0)
    H     = gi(args, "H", 0)
    W     = gi(args, "W", 0)
    matN  = gi(args, "matN", 0)
    iters = gi(args, "iters", 0)

    # 4 bytes per float
    B4 = 4

    k = kernel.strip()

    # 1) Pure vector ops
    if k == "vector_add":
        # C = A + B
        flops = N                  # 1 add/element
        bytes_ = 3 * N * B4        # A read, B read, C write
        return flops, bytes_

    if k == "saxpy":
        # C = alpha * A + B
        flops = 2 * N              # 1 mul + 1 add
        bytes_ = 3 * N * B4
        return flops, bytes_

    if k == "vector_add_divergent":
        # same math/bytes as vector_add; divergence hits runtime, not counts
        flops = N
        bytes_ = 3 * N * B4
        return flops, bytes_

    if k == "strided_copy_8":
        # copy every element once (our kernel uses 8-way indexing, but 1 read+1 write per element)
        flops = 0
        bytes_ = 2 * N * B4
        return flops, bytes_

    # 2) Reductions / dot
    if k == "reduce_sum":
        # Each element participates in an add; shared scratch writes are small-ish; keep simple
        flops = N                   # N-1 adds ~ N
        bytes_ = N * B4 + B4        # read N, write one scalar (approx)
        return flops, bytes_

    if k == "dot_product":
        # Sum_i A[i]*B[i]
        flops = 2 * N               # mul + add
        bytes_ = 2 * N * B4 + B4    # read A & B, final scalar write (approx)
        return flops, bytes_

    # 3) Transposes
    if k == "naive_transpose":
        # Read rows*cols, write rows*cols
        n = rows * cols
        flops = 0
        bytes_ = 2 * n * B4
        return flops, bytes_

    if k == "shared_transpose":
        n = rows * cols
        flops = 0
        bytes_ = 2 * n * B4
        return flops, bytes_

    # 4) GEMM
    if k == "matmul_naive" or k == "matmul_tiled":
        n = matN
        # FLOPs = 2*n^3 (mul+add)
        flops = 2 * (n**3)
        # Minimal traffic (ignoring re-use) ~ 3*n^2 floats
        bytes_ = 3 * (n**2) * B4
        return flops, bytes_

    # 5) Conv2D
    if k == "conv2d_3x3":
        K = 3
        flops = 2 * (K*K) * H * W
        bytes_ = (H*W + K*K + H*W) * B4  # read input + kernel + write output
        return flops, bytes_

    if k == "conv2d_7x7":
        K = 7
        flops = 2 * (K*K) * H * W
        bytes_ = (H*W + K*K + H*W) * B4
        return flops, bytes_

    # 6) Histogram
    if k == "histogram":
        # read N uints + update 256 bins (approximate: treat as one write per element amortized)
        flops = 0
        bytes_ = N * B4 + 256 * B4
        return flops, bytes_

    # 7) Random access (gather/scatter)
    if k == "random_access":
        # A[idx[i]] -> B[i] : 1 read + 1 write per element
        flops = 0
        bytes_ = 2 * N * B4
        return flops, bytes_

    # 8) Shared-memory bank conflict microbench
    if k == "shared_bank_conflict":
        # 1 block of 1024 threads; assume one shared load+store per thread
        T = 1024
        flops = 0
        bytes_ = 2 * T * B4
        return flops, bytes_

    # 9) Atomic hotspot
    if k == "atomic_hotspot":
        # Approximate one RMW per atomic: treat as 8B traffic; threads ~= grid*block; grid derived from N
        threads = max(1, ((N + max(1,block) - 1) // max(1,block)) * max(1,block))
        flops = 0
        bytes_ = 8 * threads * max(1, iters)
        return flops, bytes_

    # Fallback: something unknown — avoid (0,0)
    return 0, 1

if __name__ == "__main__":
    runs_csv = sys.argv[1] if len(sys.argv) > 1 else "data/runs_4070.csv"
    out_csv  = sys.argv[2] if len(sys.argv) > 2 else "data/runs_4070_with_counts.csv"

    with open(runs_csv, newline='') as f, open(out_csv, "w", newline='') as g:
        rd = csv.DictReader(f)
        fieldnames = rd.fieldnames + ["FLOPs", "BYTES"]
        w = csv.DictWriter(g, fieldnames=fieldnames)
        w.writeheader()
        for row in rd:
            args = parse_args(row.get("args",""))
            fl, by = counts_for_row(row.get("kernel","").strip(), args)
            row["FLOPs"] = f"{fl}"
            row["BYTES"] = f"{by}"
            w.writerow(row)

    print(f"[OK] wrote {out_csv}")

