// Runs every kernel in src/kernels.cuh on the CPU emulator and compares the
// result with a straightforward triple loop. Build: see README ("No GPU?").
#include "cuda_emu.h"
#include "../src/kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static std::vector<float> A, B, C, R;
static int M, N, K;

// 16-byte aligned so the float4 accesses are legal, like cudaMalloc memory.
static float *alloc(size_t n)
{
    const size_t bytes = (n * sizeof(float) + 63) / 64 * 64; // aligned_alloc wants a multiple of the alignment
    return static_cast<float *>(aligned_alloc(64, bytes));
}

static void setup(int m, int n, int k)
{
    M = m; N = n; K = k;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    A.resize(M * K); B.resize(K * N); R.assign(M * N, 0.0f);
    for (auto &v : A) v = dist(rng);
    for (auto &v : B) v = dist(rng);
    for (int i = 0; i < M; ++i)
        for (int k2 = 0; k2 < K; ++k2)
            for (int j = 0; j < N; ++j)
                R[i * N + j] += A[i * K + k2] * B[k2 * N + j];
}

template <typename Kernel>
static bool check(const char *name, Kernel kernel, dim3 grid, dim3 block)
{
    float *a = alloc(M * K), *b = alloc(K * N), *c = alloc(M * N);
    std::copy(A.begin(), A.end(), a);
    std::copy(B.begin(), B.end(), b);
    std::fill(c, c + M * N, NAN); // any element the kernel misses stays NaN

    emu::launch(kernel, grid, block, M, N, K, (const float *)a, (const float *)b, c);

    double num = 0, den = 0;
    for (int i = 0; i < M * N; ++i) {
        num += double(c[i] - R[i]) * (c[i] - R[i]);
        den += double(R[i]) * R[i];
    }
    const double err = std::sqrt(num / den);
    const bool ok = err < 1e-5; // false for NaN too
    std::printf("  %-28s %3dx%3dx%3d  rel.err %.2e  %s\n", name, M, N, K, err, ok ? "ok" : "FAIL");
    free(a); free(b); free(c);
    return ok;
}

int main()
{
    bool ok = true;

    // Non-square, and not a multiple of the block size, to exercise the bounds checks.
    setup(50, 70, 30);
    ok &= check("naive", sgemm_naive, dim3((M + 15) / 16, (N + 15) / 16), dim3(16, 16));
    ok &= check("coalesced", sgemm_coalesced, dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16));

    // Non-square so a swapped M/N/K shows up.
    setup(64, 96, 32);
    ok &= check("tiled<16>", sgemm_tiled<16>, dim3(N / 16, M / 16), dim3(16, 16));
    ok &= check("tiled<32>", sgemm_tiled<32>, dim3(N / 32, M / 32), dim3(32, 32));
    ok &= check("regblock<32,32,8,4,4>", sgemm_regblock<32, 32, 8, 4, 4>, dim3(N / 32, M / 32), dim3(64));
    ok &= check("vectorized<32,32,8,4,4>", sgemm_vectorized<32, 32, 8, 4, 4>, dim3(N / 32, M / 32), dim3(64));
    ok &= check("regblock<64,32,16,8,4>", sgemm_regblock<64, 32, 16, 8, 4>, dim3(N / 32, M / 64), dim3(64));
    ok &= check("vectorized<64,32,16,8,4>", sgemm_vectorized<64, 32, 16, 8, 4>, dim3(N / 32, M / 64), dim3(64));

    // The exact configuration the GPU benchmark uses.
    setup(256, 128, 64);
    ok &= check("regblock<128,128,8,8,8>", sgemm_regblock<128, 128, 8, 8, 8>, dim3(N / 128, M / 128), dim3(256));
    ok &= check("vectorized<128,128,8,8,8>", sgemm_vectorized<128, 128, 8, 8, 8>, dim3(N / 128, M / 128), dim3(256));

    std::printf("%s\n", ok ? "ALL PASSED" : "FAILURES");
    return ok ? 0 : 1;
}
