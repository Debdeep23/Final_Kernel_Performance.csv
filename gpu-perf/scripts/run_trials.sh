#!/usr/bin/env bash
set -euo pipefail

KERN="$1"; ARGS="$2"; REGS="$3"; SHMEM="$4"; NUM="$5"
OUT="data/trials_${KERN}__4070.csv"

# header once
if [[ ! -s "$OUT" ]]; then
  echo "kernel,args,regs,shmem,device_name,gx,gy,gz,bx,by,bz,time_ms,trial" > "$OUT"
fi

for t in $(seq 1 "$NUM"); do
  LINE="$(bin/runner --kernel "$KERN" $ARGS)"
  # LINE is: kernel,args,device_name,gx,gy,gz,bx,by,bz,time_ms
  # Prepend regs,shmem and append trial
  IFS=',' read -r k a dev gx gy gz bx by bz ms <<< "$LINE"
  echo "$k,$a,$REGS,$SHMEM,$dev,$gx,$gy,$gz,$bx,$by,$bz,$ms,$t" >> "$OUT"
done

