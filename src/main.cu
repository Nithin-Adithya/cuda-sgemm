// Benchmark driver: times each SGEMM stage against cuBLAS, checks the result,
// and measures the two roofline ceilings (DRAM bandwidth, peak FMA rate).
//
//   sgemm [--sizes 1024,2048,4096] [--kernel NAME] [--reps 10] [--out results]
//   sgemm --profile --kernel NAME --sizes 2048     (single launch, for ncu)
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "kernels.cuh"

#define CHECK_CUDA(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                         __FILE__, __LINE__);                                         \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

#define CHECK_CUBLAS(call)                                                         \
    do {                                                                           \
        cublasStatus_t st_ = (call);                                               \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                        \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, \
                         __LINE__);                                                \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

// ---------------------------------------------------------------------------
// Stage table
// ---------------------------------------------------------------------------
struct Problem {
    int M, N, K;
    const float *A, *B;
    float *C;
};

static cublasHandle_t g_cublas = nullptr;

static dim3 grid2d(int x, int y, int b) { return dim3((x + b - 1) / b, (y + b - 1) / b); }

static void run_naive(const Problem &p)
{
    sgemm_naive<<<grid2d(p.M, p.N, 32), dim3(32, 32)>>>(p.M, p.N, p.K, p.A, p.B, p.C);
}
static void run_coalesced(const Problem &p)
{
    sgemm_coalesced<<<grid2d(p.N, p.M, 32), dim3(32, 32)>>>(p.M, p.N, p.K, p.A, p.B, p.C);
}
static void run_tiled(const Problem &p)
{
    sgemm_tiled<32><<<dim3(p.N / 32, p.M / 32), dim3(32, 32)>>>(p.M, p.N, p.K, p.A, p.B, p.C);
}

// Block/thread tile shape shared by the last two stages.
enum { BM = 128, BN = 128, BK = 8, TM = 8, TN = 8 };

static void run_regblock(const Problem &p)
{
    sgemm_regblock<BM, BN, BK, TM, TN><<<dim3(p.N / BN, p.M / BM), dim3((BM / TM) * (BN / TN))>>>(
        p.M, p.N, p.K, p.A, p.B, p.C);
}
static void run_vectorized(const Problem &p)
{
    sgemm_vectorized<BM, BN, BK, TM, TN><<<dim3(p.N / BN, p.M / BM), dim3((BM / TM) * (BN / TN))>>>(
        p.M, p.N, p.K, p.A, p.B, p.C);
}
static void run_cublas(const Problem &p)
{
    // cuBLAS is column-major. Our row-major A, B, C are its A^T, B^T, C^T, and
    // C^T = B^T * A^T, so swap the operands.
    const float one = 1.0f, zero = 0.0f;
    CHECK_CUBLAS(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, p.N, p.M, p.K, &one, p.B, p.N,
                             p.A, p.K, &zero, p.C, p.N));
}

struct Stage {
    const char *name;
    void (*run)(const Problem &);
    int multiple; // M, N, K must be multiples of this
};

static const Stage kStages[] = {
    {"cublas", run_cublas, 1}, // first: it is the reference for the others
    {"naive", run_naive, 1},
    {"coalesced", run_coalesced, 1},
    {"tiled", run_tiled, 32},
    {"regblock", run_regblock, 128},
    {"vectorized", run_vectorized, 128},
};

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
struct Timing {
    double best_ms, median_ms;
};

// Times fn() with CUDA events. Stops early once `budget_ms` of GPU time is
// spent (the slow stages take seconds per run at large N), but always does at
// least two reps.
template <typename Fn>
static Timing time_gpu(Fn fn, int reps, double budget_ms = 3000.0)
{
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    std::vector<double> ms;
    double total = 0.0;
    for (int r = 0; r < reps && (r < 2 || total < budget_ms); ++r) {
        CHECK_CUDA(cudaEventRecord(t0));
        fn();
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float t = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&t, t0, t1));
        ms.push_back(t);
        total += t;
    }
    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    std::sort(ms.begin(), ms.end());
    return {ms.front(), ms[ms.size() / 2]};
}

// ---------------------------------------------------------------------------
// Roofline ceilings, measured on this device
// ---------------------------------------------------------------------------
__global__ void triad(float *c, const float *a, const float *b, float s, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + s * b[i];
}

// Eight independent FMA chains per thread, all in registers: no memory traffic,
// so the rate is the arithmetic ceiling.
__global__ void fma_burn(float *out, float a, float b, int iters)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x0 = tid * 1e-7f, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3;
    float x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;
    for (int i = 0; i < iters; ++i) {
        x0 = x0 * a + b; x1 = x1 * a + b; x2 = x2 * a + b; x3 = x3 * a + b;
        x4 = x4 * a + b; x5 = x5 * a + b; x6 = x6 * a + b; x7 = x7 * a + b;
    }
    out[tid] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

static double measure_bandwidth_gbs()
{
    size_t free_b = 0, total_b = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
    size_t n = (size_t)1 << 26; // 256 MB per array: far larger than any cache
    while (3 * n * sizeof(float) > free_b / 2) n >>= 1;

    float *a, *b, *c;
    CHECK_CUDA(cudaMalloc(&a, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&c, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(a, 0, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(b, 0, n * sizeof(float)));
    const unsigned blocks = (unsigned)((n + 255) / 256);
    triad<<<blocks, 256>>>(c, a, b, 2.0f, n); // warm-up
    const Timing t = time_gpu([&] { triad<<<blocks, 256>>>(c, a, b, 2.0f, n); }, 10);
    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));
    return 3.0 * n * sizeof(float) / (t.best_ms * 1e-3) / 1e9; // 2 reads + 1 write
}

static double measure_peak_gflops(const cudaDeviceProp &prop)
{
    const int blocks = prop.multiProcessorCount * 32, threads = 256, iters = 1 << 16;
    float *out;
    CHECK_CUDA(cudaMalloc(&out, (size_t)blocks * threads * sizeof(float)));
    fma_burn<<<blocks, threads>>>(out, 0.999f, 0.001f, iters); // warm-up
    const Timing t = time_gpu([&] { fma_burn<<<blocks, threads>>>(out, 0.999f, 0.001f, iters); }, 10);
    CHECK_CUDA(cudaFree(out));
    const double flops = 2.0 * 8.0 * iters * (double)blocks * threads; // FMA = 2 flops
    return flops / (t.best_ms * 1e-3) / 1e9;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
static double rel_error(const std::vector<float> &x, const std::vector<float> &ref)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double d = (double)x[i] - ref[i];
        num += d * d;
        den += (double)ref[i] * ref[i];
    }
    return std::sqrt(num / den);
}

int main(int argc, char **argv)
{
    std::vector<int> sizes = {1024, 2048, 4096};
    std::string only, out_dir = "results";
    int reps = 10;
    bool profile = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", arg.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--sizes") {
            sizes.clear();
            std::stringstream ss(next());
            for (std::string tok; std::getline(ss, tok, ',');) sizes.push_back(std::atoi(tok.c_str()));
        } else if (arg == "--kernel") {
            only = next();
        } else if (arg == "--reps") {
            reps = std::atoi(next().c_str());
        } else if (arg == "--out") {
            out_dir = next();
        } else if (arg == "--profile") {
            profile = true;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--sizes N1,N2,..] [--kernel NAME] [--reps R] [--out DIR] [--profile]\n"
                         "kernels: cublas naive coalesced tiled regblock vectorized\n",
                         argv[0]);
            return arg == "--help" ? 0 : 1;
        }
    }

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    CHECK_CUBLAS(cublasCreate(&g_cublas));

    // --profile: exactly one launch of one kernel and nothing else on the GPU,
    // so an ncu report of the process contains only that kernel.
    if (profile) {
        if (only.empty() || sizes.size() != 1) {
            std::fprintf(stderr, "--profile needs --kernel NAME and a single --sizes N\n");
            return 1;
        }
    } else {
        std::printf("GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    }

    FILE *csv = nullptr;
    if (!profile) {
        std::filesystem::create_directories(out_dir);
        const double bw = measure_bandwidth_gbs();
        const double peak = measure_peak_gflops(prop);
        std::printf("Measured ceilings: %.1f GB/s DRAM bandwidth, %.0f GFLOP/s FMA peak\n\n", bw, peak);

        FILE *dev = std::fopen((out_dir + "/device.csv").c_str(), "w");
        if (!dev) { std::perror("device.csv"); return 1; }
        std::fprintf(dev, "key,value\ngpu,%s\nbandwidth_gbs,%.3f\npeak_gflops,%.3f\n", prop.name, bw, peak);
        std::fclose(dev);

        csv = std::fopen((out_dir + "/bench.csv").c_str(), "w");
        if (!csv) { std::perror("bench.csv"); return 1; }
        std::fprintf(csv, "kernel,n,ms_best,ms_median,gflops,pct_cublas,rel_err\n");
    }

    bool all_ok = true;
    for (int n : sizes) {
        const size_t count = (size_t)n * n, bytes = count * sizeof(float);
        std::vector<float> hA(count), hB(count), hC(count), hRef;
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : hA) v = dist(rng);
        for (auto &v : hB) v = dist(rng);

        float *dA, *dB, *dC;
        CHECK_CUDA(cudaMalloc(&dA, bytes));
        CHECK_CUDA(cudaMalloc(&dB, bytes));
        CHECK_CUDA(cudaMalloc(&dC, bytes));
        CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
        const Problem p = {n, n, n, dA, dB, dC};
        const double gflop = 2.0 * n * n * (double)n / 1e9;

        if (!profile)
            std::printf("N = %d\n  %-11s %10s %10s %10s %9s %9s\n", n, "kernel", "best ms", "median ms",
                        "GFLOP/s", "% cuBLAS", "rel.err");

        double cublas_gflops = 0.0;
        for (const Stage &s : kStages) {
            const bool is_ref = std::string(s.name) == "cublas";
            if (profile ? only != s.name : (!only.empty() && only != s.name && !is_ref)) continue;
            if (n % s.multiple != 0) {
                std::printf("  %-11s skipped (N must be a multiple of %d)\n", s.name, s.multiple);
                continue;
            }

            if (profile) {
                s.run(p);
                CHECK_CUDA(cudaDeviceSynchronize());
                continue;
            }

            s.run(p); // warm-up
            CHECK_CUDA(cudaDeviceSynchronize());
            const Timing t = time_gpu([&] { s.run(p); }, reps);
            CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

            const double gflops = gflop / (t.best_ms * 1e-3);
            double err = 0.0;
            if (is_ref) {
                hRef = hC;
                cublas_gflops = gflops;
            } else {
                err = rel_error(hC, hRef);
            }
            const bool ok = err < 1e-4;
            all_ok = all_ok && ok;
            const double pct = 100.0 * gflops / cublas_gflops;
            std::printf("  %-11s %10.3f %10.3f %10.1f %8.1f%% %9.1e%s\n", s.name, t.best_ms, t.median_ms,
                        gflops, pct, err, ok ? "" : "  WRONG RESULT");
            std::fprintf(csv, "%s,%d,%.4f,%.4f,%.2f,%.2f,%.3e\n", s.name, n, t.best_ms, t.median_ms, gflops,
                         pct, err);
        }
        if (!profile) std::printf("\n");

        CHECK_CUDA(cudaFree(dA));
        CHECK_CUDA(cudaFree(dB));
        CHECK_CUDA(cudaFree(dC));
    }

    if (csv) {
        std::fclose(csv);
        std::printf("Wrote %s/bench.csv and %s/device.csv\n", out_dir.c_str(), out_dir.c_str());
    }
    CHECK_CUBLAS(cublasDestroy(g_cublas));
    return all_ok ? 0 : 1;
}
