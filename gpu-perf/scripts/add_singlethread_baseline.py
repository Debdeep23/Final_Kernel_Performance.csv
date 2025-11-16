#!/usr/bin/env python3
import sys, json, csv, math

def roofline_time_ms(flops, bytes_, peak_gflops, peak_gbps):
    # time = max( FLOPs / PeakCompute , BYTES / PeakBW )
    t_comp = 0.0
    if peak_gflops > 0 and flops > 0:
        t_comp = 1000.0 * (flops / (peak_gflops * 1e9))
    t_mem = 0.0
    if peak_gbps > 0 and bytes_ > 0:
        t_mem  = 1000.0 * (bytes_ / (peak_gbps * 1e9))
    return max(t_comp, t_mem)

if __name__ == "__main__":
    runs_csv   = sys.argv[1] if len(sys.argv)>1 else "data/runs_4070_with_counts.csv"
    calib_json = sys.argv[2] if len(sys.argv)>2 else "data/device_calibration_4070.json"
    out_csv    = sys.argv[3] if len(sys.argv)>3 else "data/runs_4070_final.csv"

    C = json.load(open(calib_json))
    peak_gflops = float(C.get("sustained_compute_gflops", 0.0))
    peak_gbps   = float(C.get("sustained_mem_bandwidth_gbps", 0.0))
    warp_size   = int(C.get("warp_size", 32) or 32)

    # single-active-lane model: scale ceilings by warp_size
    peak_gflops_1 = peak_gflops / warp_size if warp_size>0 else 0.0
    peak_gbps_1   = peak_gbps   / warp_size if warp_size>0 else 0.0

    with open(runs_csv, newline='') as f, open(out_csv, "w", newline='') as g:
        rd = csv.DictReader(f)
        fn = rd.fieldnames + ["T1_model_ms", "speedup_model"]
        w = csv.DictWriter(g, fieldnames=fn)
        w.writeheader()
        for row in rd:
            # measured parallel runtime
            try:
                Tpar = float(row.get("mean_ms","") or row.get("time_ms","") or 0.0)
            except:
                Tpar = 0.0

            # static counts
            try:
                fl = float(row.get("FLOPs","") or 0.0)
            except:
                fl = 0.0
            try:
                by = float(row.get("BYTES","") or 0.0)
            except:
                by = 0.0

            # compute T1 even if FLOPs==0 (memory-bound kernels still have BYTES)
            T1 = roofline_time_ms(fl, by, peak_gflops_1, peak_gbps_1)

            # speedup = T1 / Tpar, but only if both are > 0
            if T1 > 0.0 and Tpar > 0.0:
                speed = T1 / Tpar
                row["speedup_model"] = f"{speed:.6f}"
            else:
                row["speedup_model"] = ""

            row["T1_model_ms"] = f"{T1:.6f}" if T1>0 else ""
            w.writerow(row)

    print(f"[OK] wrote {out_csv}")

