#!/usr/bin/env python3
import sys, csv, math, statistics as st

inputs = sys.argv[1:]  # e.g., data/trials_*__4070.csv
rows=[]
for p in inputs:
    with open(p, newline='') as f:
        rows += list(csv.DictReader(f))

# group key preserves geometry
def key(r):
    return (r["kernel"], r["args"], r["regs"], r["shmem"], r["device_name"],
            r["gx"], r["gy"], r["gz"], r["bx"], r["by"], r["bz"])

groups={}
for r in rows:
    groups.setdefault(key(r), []).append(float(r["time_ms"]))

wr = csv.writer(sys.stdout)
wr.writerow(["kernel","args","regs","shmem","device_name",
             "gx","gy","gz","bx","by","bz","trials","mean_ms","std_ms"])
for k, times in groups.items():
    mean = st.fmean(times)
    std  = st.pstdev(times) if len(times)>1 else 0.0
    wr.writerow([*k, len(times), f"{mean:.6f}", f"{std:.6f}"])

