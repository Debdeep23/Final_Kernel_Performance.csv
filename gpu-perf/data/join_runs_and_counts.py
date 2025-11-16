#!/usr/bin/env python3
import csv, sys, re

runs_csv   = sys.argv[1]  # data/runs_4070.csv
counts_csv = sys.argv[2]  # data/kernels_static_4070.csv
out_csv    = sys.argv[3]  # data/runs_4070_with_counts.csv

def parse_kv(argstr):
    kv={}
    toks=argstr.strip().split()
    i=0
    while i<len(toks):
        t=toks[i]
        if t.startswith("--") and i+1<len(toks):
            key=t[2:]
            val=toks[i+1]
            if re.fullmatch(r"-?\d+(\.\d+)?", val):
                kv[key]=val
            i+=2
        else:
            i+=1
    return kv

# Build a join key from the important shape fields
def make_key(kernel, kv):
    def g(k): return (kv.get(k) or "").strip()
    # priority: N, rows/cols, H/W, matN
    return (
        kernel.strip(),
        g("N"), g("rows"), g("cols"),
        g("H"), g("W"), g("matN"), g("iters")
    )

# load counts
counts={}
with open(counts_csv, newline='') as f:
    rd=csv.DictReader(f)
    for r in rd:
        kv = parse_kv(r.get("args",""))
        key = make_key(r["kernel"], {
            "N": r.get("N",""),
            "rows": r.get("rows",""),
            "cols": r.get("cols",""),
            "H": r.get("H",""),
            "W": r.get("W",""),
            "matN": r.get("matN",""),
            "iters": r.get("iters",""),
        })
        counts[key] = (r.get("FLOPs",""), r.get("BYTES",""))

with open(runs_csv, newline='') as f, open(out_csv,"w",newline='') as g:
    rd=csv.DictReader(f)
    fieldnames = rd.fieldnames + [c for c in ["FLOPs","BYTES"] if c not in rd.fieldnames]
    w=csv.DictWriter(g, fieldnames=fieldnames)
    w.writeheader()
    missing=0
    for r in rd:
        kv = parse_kv(r.get("args",""))
        key = make_key(r["kernel"], kv)
        fl, by = counts.get(key, ("",""))
        if not fl:
            missing += 1
        r["FLOPs"]=fl
        r["BYTES"]=by
        w.writerow(r)
    if missing:
        sys.stderr.write(f"[WARN] counts missing for {missing} row(s)\n")

print(f"[OK] wrote {out_csv}")

