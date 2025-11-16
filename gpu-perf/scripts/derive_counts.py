#!/usr/bin/env python3
import csv, sys, re

inp  = sys.argv[1] if len(sys.argv)>1 else "data/runs_4070.csv"
outp = sys.argv[2] if len(sys.argv)>2 else "data/kernels_static_4070.csv"

def parse_args(argstr: str):
    """Extract common params from the 'args' field like '--N 1048576 --rows 2048 --cols 2048 ...'."""
    kv = {}
    toks = argstr.strip().split()
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--") and i+1 < len(toks):
            key = t[2:]
            val = toks[i+1]
            # only keep numeric-looking values
            if re.fullmatch(r"-?\d+(\.\d+)?", val):
                kv[key] = float(val) if "." in val else int(val)
            i += 2
        else:
            i += 1
    return kv

def counts_for_row(kernel: str, kv: dict):
    """
    Return FLOPs, BYTES for the given kernel.
    Approximation models kept consistent with our earlier runs.
    """
    # helpers
    N    = int(kv.get("N", 0))
    rows = int(kv.get("rows", 0))
    cols = int(kv.get("cols", 0))
    H    = int(kv.get("H", 0))
    W    = int(kv.get("W", 0))
    matN = int(kv.get("matN", 0))
    iters= int(kv.get("iters", 0))

    B4 = 4  # sizeof(float)

    k = kernel.strip()

    if k == "vector_add":
        flops = N * 1
        bytes_ = N * (B4 + B4 + B4)  # read A,B; write C
        return flops, bytes_

    if k == "saxpy":
        flops = N * 2               # a*x + y  => 1 mul + 1 add
        bytes_ = N * (B4 + B4 + B4) # read A,B; write C
        return flops, bytes_

    if k == "strided_copy_8":
        # total elements moved is N; each element read+write once
        flops = 0
        bytes_ = N * (B4 + B4)      # read src, write dst
        return flops, bytes_

    if k == "naive_transpose":
        elems = rows * cols
        flops = 0
        bytes_ = elems * (B4 + B4)  # read + write
        return flops, bytes_

    if k == "shared_transpose":
        elems = rows * cols
        flops = 0
        bytes_ = elems * (B4 + B4)
        return flops, bytes_

    if k == "matmul_tiled" or k == "matmul_naive":
        N2 = matN if matN else 0
        flops  = 2 * (N2**3)        # classic GEMM
        bytes_ = 3 * (N2**2) * B4   # A,B read; C write
        return flops, bytes_

    if k == "reduce_sum":
        flops  = max(N - 1, 0)      # ~N adds
        bytes_ = N * B4             # read array (writes of partials negligible)
        return flops, bytes_

    if k == "dot_product":
        flops  = 2 * N              # mul + add
        bytes_ = (2 * N) * B4       # read A and B (partials negligible)
        return flops, bytes_

    if k == "histogram":
        flops  = N                   # ~1 op per element (bin arithmetic)
        bytes_ = N * B4              # read data; bin traffic ignored here
        return flops, bytes_

    if k == "conv2d_3x3":
        elems = H * W
        flops  = elems * (9 + 8)     # 9 mul + 8 adds per output
        bytes_ = elems * (B4 + B4)   # read input pixel neighborhood amortized ~ read+write per px
        return flops, bytes_

    if k == "conv2d_7x7":
        elems = H * W
        flops  = elems * (49 + 48)   # 49 mul + 48 adds
        bytes_ = elems * (B4 + B4)
        return flops, bytes_

    if k == "random_access":
        flops  = 0
        # read A (or dst), read idx, write B  => ~12 bytes/elt
        bytes_ = N * (B4 + B4 + B4)
        return flops, bytes_

    if k == "vector_add_divergent":
        flops  = N * 1
        bytes_ = N * (B4 + B4 + B4)
        return flops, bytes_

    if k == "shared_bank_conflict":
        # small synthetic; keep it finite
        el = 1024
        flops  = el                   # nominal
        bytes_ = el * (B4 + B4)       # read+write one pass
        return flops, bytes_

    if k == "atomic_hotspot":
        # N threads * iters atomic adds (to a single counter)
        flops  = N * iters
        bytes_ = N * iters * B4       # very rough proxy
        return flops, bytes_

    # default safe zero
    return 0, 0

with open(inp, newline='') as f, open(outp, "w", newline='') as g:
    rd = csv.DictReader(f)
    fieldnames = ["kernel","N","rows","cols","H","W","matN","iters","FLOPs","BYTES","args"]
    w = csv.DictWriter(g, fieldnames=fieldnames)
    w.writeheader()
    for r in rd:
        k = r["kernel"].strip()
        kv = parse_args(r.get("args",""))
        fl, by = counts_for_row(k, kv)
        row = {
            "kernel": k,
            "N":    kv.get("N",""),
            "rows": kv.get("rows",""),
            "cols": kv.get("cols",""),
            "H":    kv.get("H",""),
            "W":    kv.get("W",""),
            "matN": kv.get("matN",""),
            "iters":kv.get("iters",""),
            "FLOPs": f"{fl}",
            "BYTES": f"{by}",
            "args": r.get("args","")
        }
        w.writerow(row)

print(f"[OK] wrote {outp}")

