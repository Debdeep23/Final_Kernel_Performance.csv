#!/usr/bin/env python3
import argparse, json, re, sys

def kvs_from_text(txt):
    """
    Extract key=value pairs from the whole text.
    Works even if multiple pairs share a line (e.g., 'major=8 minor=9').
    """
    kv = {}
    for m in re.finditer(r'([A-Za-z0-9_]+)=([^\s]+)', txt):
        k, v = m.group(1), m.group(2)
        kv[k] = v
    return kv

def parse_props(txt):
    kv = kvs_from_text(txt)
    out = {}
    # Basic identity
    out["device_name"] = kv.get("name","")
    # SM version
    out["major"] = int(kv.get("major","0"))
    out["minor"] = int(kv.get("minor","0"))
    # Counts/limits (fall back to 0 if missing)
    out["sm_count"]                = int(kv.get("multiProcessorCount","0"))
    out["max_threads_per_sm"]      = int(kv.get("maxThreadsPerMultiProcessor","0"))
    out["max_blocks_per_sm"]       = int(kv.get("maxBlocksPerMultiProcessor","0"))
    out["registers_per_sm"]        = int(kv.get("regsPerMultiprocessor","0"))
    out["shared_mem_per_sm_bytes"] = int(kv.get("sharedMemPerMultiprocessor","0"))
    out["warpSize"]                = int(kv.get("warpSize","32"))
    return out

def parse_stream(txt):
    # Expect a line 'SUSTAINED_MEM_BW_GBPS=<number>'
    m = re.search(r'SUSTAINED_MEM_BW_GBPS\s*=\s*([0-9.]+)', txt)
    if not m:
        # best-effort: take the max GBps= number
        m = re.search(r'GBps=([0-9.]+)', txt)
    return float(m.group(1)) if m else 0.0

def parse_gemm(txt):
    # Expect a line 'SUSTAINED_COMPUTE_GFLOPS=<number>'
    m = re.search(r'SUSTAINED_COMPUTE_GFLOPS\s*=\s*([0-9.]+)', txt)
    if not m:
        # best-effort: take the max GFLOPS= number
        m = re.search(r'GFLOPS=([0-9.]+)', txt)
    return float(m.group(1)) if m else 0.0

def ok(label, got, want):
    print(f"[OK] {label}: {got} vs {want}") if got == want else print(f"[MISMATCH] {label}: {got} vs {want}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--props",  required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--gemm",   required=True)
    ap.add_argument("--out",    required=True)
    args = ap.parse_args()

    props_txt  = open(args.props,  "r").read()
    stream_txt = open(args.stream, "r").read()
    gemm_txt   = open(args.gemm,   "r").read()

    P = parse_props(props_txt)
    bw = parse_stream(stream_txt)
    fl = parse_gemm(gemm_txt)

    cfg = {
        "device_name": P["device_name"],
        "warp_size":   P["warpSize"],
        "sm_count":    P["sm_count"],
        "sustained_mem_bandwidth_gbps": bw,
        "sustained_compute_gflops":     fl,
        "sm_limits": {
            "max_threads_per_sm":      P["max_threads_per_sm"],
            "max_blocks_per_sm":       P["max_blocks_per_sm"],
            "registers_per_sm":        P["registers_per_sm"],
            "shared_mem_per_sm_bytes": P["shared_mem_per_sm_bytes"]
        }
    }

    # sanity compare against what the CUDA props say
    ok("device_name",          cfg["device_name"], cfg["device_name"])
    ok("sm.max_threads_per_sm",cfg["sm_limits"]["max_threads_per_sm"], P["max_threads_per_sm"])
    ok("sm.max_blocks_per_sm", cfg["sm_limits"]["max_blocks_per_sm"],  P["max_blocks_per_sm"])
    ok("sm.registers_per_sm",  cfg["sm_limits"]["registers_per_sm"],   P["registers_per_sm"])
    ok("sm.shared_mem_per_sm", cfg["sm_limits"]["shared_mem_per_sm_bytes"], P["shared_mem_per_sm_bytes"])
    ok("warp_size",            cfg["warp_size"], P["warpSize"])

    # write out
    with open(args.out,"w") as f:
        json.dump(cfg,f,indent=2)
