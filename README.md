# cuda-sgemm

A single-precision matrix multiply (`C = A * B`) written in CUDA C++ and optimized in stages,
each benchmarked against cuBLAS and profiled with Nsight Compute to see whether it is limited
by memory or by arithmetic.

| # | Kernel | What changes | What it fixes |
|---|--------|--------------|---------------|
| 0 | `naive` | one thread per output element, `threadIdx.x` indexes the row | baseline: every warp access is 32 separate transactions |
| 1 | `coalesced` | `threadIdx.x` indexes the column | warp reads of `B` and writes of `C` become one 128-byte transaction |
| 2 | `tiled` | 32x32 tiles of `A` and `B` staged in shared memory | each global element is fetched once per block, not once per thread |
| 3 | `regblock` | each thread computes an 8x8 patch held in registers (128x128 block, 8-deep K step) | shared-memory loads per FMA drop from 2 to 1/4 |
| 4 | `vectorized` | `float4` global loads and stores, `A` tile stored transposed | 4x fewer load instructions, contiguous shared-memory reads |

All kernels are in [src/kernels.cuh](src/kernels.cuh); the comment above each one explains the access pattern.

## Build and run (needs an NVIDIA GPU and the CUDA toolkit)

```bash
cmake -B build && cmake --build build -j
```

or without CMake:

```bash
mkdir -p build && nvcc -O3 -std=c++17 -lineinfo src/main.cu -o build/sgemm -lcublas
```

Benchmark every stage at N = 1024, 2048, 4096:

```bash
./build/sgemm
```

This prints time, GFLOP/s and percent of cuBLAS for each stage, checks the result against cuBLAS and against a double-precision host recomputation of sampled entries, and writes
`results/bench.csv` and `results/device.csv`. It also measures the two roofline ceilings on your GPU:
DRAM bandwidth (a triad over 256 MB arrays) and peak FMA rate (register-only FMA chains).

Options: `--sizes 2048,4096`, `--kernel vectorized`, `--reps 20`, `--out DIR`.
Stages 3 and 4 need N to be a multiple of 128 (stage 2: of 32); other sizes are skipped.

## Profile and build the roofline

```bash
scripts/profile.sh 2048
```

```bash
python3 scripts/roofline.py --size 2048
```

`profile.sh` runs each stage once under `ncu`, collecting DRAM bytes read/written and the
floating-point instruction counts, into `results/ncu/`. `roofline.py` combines those with the
benchmark numbers:

- arithmetic intensity = (fadd + fmul + 2 * ffma) / (DRAM bytes read + written)
- a kernel left of the ridge point (peak GFLOP/s / bandwidth) is memory-bound, right of it compute-bound
- "% of roof" says how close it gets to the limit that applies to it

It prints that table and writes `results/roofline_2048.png`.

For the full Nsight Compute view (memory workload analysis, occupancy, source hot spots), run
`FULL=1 scripts/profile.sh 2048` and open `results/ncu/<kernel>_full.ncu-rep` in the `ncu-ui`.

## No GPU?

The kernels' indexing can be checked on any machine. [test/cuda_emu.h](test/cuda_emu.h) emulates
blocks, threads, `__shared__` and `__syncthreads()` with `std::thread` and `std::barrier`, and
[test/emu_test.cpp](test/emu_test.cpp) runs all five kernels through it against a reference loop:

```bash
clang++ -std=c++20 -O2 test/emu_test.cpp -o emu_test && ./emu_test
```

For real numbers, a free Google Colab GPU runtime has `nvcc` and cuBLAS: upload this folder, then run
the build and benchmark commands above in a cell, each prefixed with `!`.

## Results

Fill in from `./build/sgemm` and `scripts/roofline.py` on your GPU.

GPU: ____ , N = ____

| Kernel | GFLOP/s | % of cuBLAS | FLOP/byte | Regime |
|--------|---------|-------------|-----------|--------|
| naive | | | | |
| coalesced | | | | |
| tiled | | | | |
| regblock | | | | |
| vectorized | | | | |
| cuBLAS | | 100 | | |
