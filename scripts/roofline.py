#!/usr/bin/env python3
"""Roofline plot and memory-bound / compute-bound classification.

Inputs (all under --results, default "results"):
  device.csv      measured ceilings, written by the benchmark
  bench.csv       measured GFLOP/s per kernel, written by the benchmark
  ncu/<k>.csv     Nsight Compute counters per kernel, written by scripts/profile.sh

Arithmetic intensity = FLOPs executed / bytes moved to and from DRAM, both taken
from Nsight Compute. Without ncu data it falls back to the algorithmic minimum
traffic (read A and B once, write C once), which is an upper bound on intensity.
"""
import argparse
import csv
import os
import sys

ORDER = ["naive", "coalesced", "tiled", "regblock", "vectorized", "cublas"]
FLOP_WEIGHTS = {  # an FMA is a multiply and an add
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 2,
}
BYTE_METRICS = ["dram__bytes_read.sum", "dram__bytes_write.sum"]


def to_float(text):
    try:
        return float(text.replace(",", ""))
    except (AttributeError, ValueError):
        return None


def read_ncu(path):
    """Returns {metric: total over all kernel launches in the report}.

    Handles both ncu CSV layouts: long (Metric Name / Metric Value columns) and
    wide (one column per metric, with a units row under the header).
    """
    totals = {}
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    # ncu may print warnings before the table; the header is the row with "ID".
    start = next((i for i, r in enumerate(rows) if r and r[0] == "ID"), None)
    if start is None:
        return totals
    header, body = rows[start], rows[start + 1:]
    if "Metric Name" in header:
        name_col, value_col = header.index("Metric Name"), header.index("Metric Value")
        for r in body:
            if len(r) > value_col and to_float(r[value_col]) is not None:
                totals[r[name_col]] = totals.get(r[name_col], 0.0) + to_float(r[value_col])
    else:
        for col, name in enumerate(header):
            for r in body:
                if len(r) > col and to_float(r[col]) is not None:
                    totals[name] = totals.get(name, 0.0) + to_float(r[col])
    return totals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--size", type=int, default=2048, help="matrix size N to plot (must be in bench.csv)")
    ap.add_argument("--out", default=None, help="output image (default: <results>/roofline_N.png)")
    args = ap.parse_args()

    with open(os.path.join(args.results, "device.csv"), newline="") as f:
        device = {r["key"]: r["value"] for r in csv.DictReader(f)}
    bandwidth = float(device["bandwidth_gbs"])  # GB/s
    with open(os.path.join(args.results, "bench.csv"), newline="") as f:
        gflops = {r["kernel"]: float(r["gflops"]) for r in csv.DictReader(f) if int(r["n"]) == args.size}
    if not gflops:
        sys.exit(f"no rows for N={args.size} in bench.csv")
    # The FMA microbenchmark is an estimate; never draw the roof below something we measured.
    peak = max(float(device["peak_gflops"]), max(gflops.values()))
    ridge = peak / bandwidth  # FLOP/byte where the two roofs meet

    points, used_fallback = [], False
    for k in [k for k in ORDER if k in gflops]:
        ncu_path = os.path.join(args.results, "ncu", k + ".csv")
        ncu = read_ncu(ncu_path) if os.path.exists(ncu_path) else {}
        flops = sum(w * ncu.get(m, 0.0) for m, w in FLOP_WEIGHTS.items())
        nbytes = sum(ncu.get(m, 0.0) for m in BYTE_METRICS)
        if flops > 0 and nbytes > 0:
            ai, source = flops / nbytes, "ncu"
        else:
            ai, source, used_fallback = args.size / 6.0, "algorithmic", True  # 2N^3 / (3 * 4N^2)
        roof = min(peak, ai * bandwidth)
        points.append((k, ai, gflops[k], roof, source))

    print(f"{device.get('gpu', 'GPU')}, N={args.size}")
    print(f"ceilings: {bandwidth:.1f} GB/s, {peak:.0f} GFLOP/s, ridge at {ridge:.1f} FLOP/byte\n")
    print(f"{'kernel':<11} {'FLOP/byte':>10} {'GFLOP/s':>9} {'roof':>9} {'% of roof':>10}  regime")
    for k, ai, perf, roof, source in points:
        regime = "memory-bound" if ai < ridge else "compute-bound"
        note = "" if source == "ncu" else "  (no ncu data)"
        print(f"{k:<11} {ai:>10.2f} {perf:>9.1f} {roof:>9.1f} {100 * perf / roof:>9.1f}%  {regime}{note}")
    if used_fallback:
        print("\nwarning: kernels marked '(no ncu data)' use the algorithmic intensity N/6, which only\n"
              "bounds the real one from above. Run scripts/profile.sh for measured values.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("\nmatplotlib not installed; skipping the plot (pip install matplotlib)")

    lo = min(0.05, min(p[1] for p in points) / 2)
    hi = max(ridge * 8, max(p[1] for p in points) * 2)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot([lo, ridge, hi], [lo * bandwidth, peak, peak], color="black", lw=2)
    ax.axvline(ridge, color="gray", ls=":", lw=1)
    ax.text(ridge * 0.9, lo * bandwidth * 1.5, "memory-bound", ha="right", color="gray")
    ax.text(ridge * 1.1, lo * bandwidth * 1.5, "compute-bound", ha="left", color="gray")
    ax.text(lo * 1.3, lo * 1.3 * bandwidth * 1.25, f"{bandwidth:.0f} GB/s", rotation=38,
            rotation_mode="anchor", fontsize=9)
    ax.text(hi, peak * 1.08, f"{peak:.0f} GFLOP/s", ha="right", fontsize=9)
    for k, ai, perf, _, source in points:
        ax.scatter(ai, perf, s=60, zorder=3, marker="o" if source == "ncu" else "x")
        ax.annotate(k, (ai, perf), textcoords="offset points", xytext=(7, -3), fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP / DRAM byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title(f"SGEMM roofline, N={args.size}, {device.get('gpu', 'GPU')}")
    ax.grid(True, which="both", alpha=0.25)
    out = args.out or os.path.join(args.results, f"roofline_{args.size}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
