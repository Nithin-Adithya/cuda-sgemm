#!/usr/bin/env bash
# Profiles each stage with Nsight Compute and exports the counters the roofline
# needs (DRAM bytes moved, floating-point instructions executed).
#
#   scripts/profile.sh [N]        N defaults to 2048
#   FULL=1 scripts/profile.sh     also saves a --set full report per kernel
#                                 (open results/ncu/<kernel>_full.ncu-rep in the ncu UI)
set -euo pipefail

N="${1:-2048}"
BIN="${BIN:-build/sgemm}"
OUT="${OUT:-results}/ncu"
KERNELS="${KERNELS:-naive coalesced tiled regblock vectorized cublas}"

METRICS="dram__bytes_read.sum,dram__bytes_write.sum"
METRICS+=",smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
METRICS+=",smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
METRICS+=",smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
METRICS+=",gpu__time_duration.sum"

mkdir -p "$OUT"
for k in $KERNELS; do
    echo "== $k (N=$N)"
    ncu --metrics "$METRICS" -f -o "$OUT/$k" "$BIN" --profile --kernel "$k" --sizes "$N" > "$OUT/$k.log"
    ncu --import "$OUT/$k.ncu-rep" --csv --page raw --print-units base > "$OUT/$k.csv"
    if [[ "${FULL:-0}" == 1 ]]; then
        ncu --set full -f -o "$OUT/${k}_full" "$BIN" --profile --kernel "$k" --sizes "$N" >> "$OUT/$k.log"
    fi
done
echo "Reports in $OUT. Next: python3 scripts/roofline.py --size $N"
