// SGEMM kernels, one per optimization stage. C (MxN) = A (MxK) * B (KxN),
// all matrices row-major, single precision.
//
// This header is compiled by nvcc for the GPU build and by a host compiler for
// the CPU emulation test (test/cuda_emu.h supplies the CUDA builtins there), so
// it must only use: __global__, __shared__, __syncthreads, threadIdx, blockIdx,
// float4 / make_float4.
#pragma once

// ---------------------------------------------------------------------------
// Stage 0: naive. threadIdx.x indexes the *row*, so the 32 threads of a warp
// touch 32 different rows: A reads are strided by K and C writes by N. Every
// warp-level access becomes 32 separate memory transactions.
// Launch: block (B, B), grid (ceil(M/B), ceil(N/B)).
// ---------------------------------------------------------------------------
__global__ void sgemm_naive(int M, int N, int K, const float *A, const float *B, float *C)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Stage 1: global-memory coalescing. Same arithmetic, but threadIdx.x now
// indexes the *column*. Within a warp, A[row*K + k] is a single address
// (broadcast), B[k*N + col] is 32 consecutive floats (one 128-byte
// transaction), and the C write is consecutive too.
// Launch: block (B, B), grid (ceil(N/B), ceil(M/B)).
// ---------------------------------------------------------------------------
__global__ void sgemm_coalesced(int M, int N, int K, const float *A, const float *B, float *C)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Stage 2: shared-memory tiling. Each block computes a TILE x TILE patch of C.
// Per K-step it stages a TILE x TILE tile of A and of B in shared memory, so
// each global element is fetched once per block instead of once per thread.
// Requires M, N, K to be multiples of TILE.
// Launch: block (TILE, TILE), grid (N/TILE, M/TILE).
// ---------------------------------------------------------------------------
template <int TILE>
__global__ void sgemm_tiled(int M, int N, int K, const float *A, const float *B, float *C)
{
    __shared__ float As[TILE * TILE];
    __shared__ float Bs[TILE * TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        As[ty * TILE + tx] = A[row * K + k0 + tx];
        Bs[ty * TILE + tx] = B[(k0 + ty) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            acc += As[ty * TILE + k] * Bs[k * TILE + tx];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Stage 3: register blocking. A block still owns a BM x BN patch of C, but
// each thread now computes a TM x TN sub-patch and keeps it in registers.
// Per k it reads TM values of A and TN values of B from shared memory and
// does TM*TN FMAs, so shared-memory loads per FMA drop from 2 to
// (TM+TN)/(TM*TN). That moves the bottleneck from memory to arithmetic.
// Requires M % BM == 0, N % BN == 0, K % BK == 0.
// Launch: block ((BM/TM)*(BN/TN)), grid (N/BN, M/BM).
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_regblock(int M, int N, int K, const float *A, const float *B, float *C)
{
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int nthreads = (BM / TM) * (BN / TN);
    const int tid = threadIdx.x;
    const int tcol = tid % (BN / TN);
    const int trow = tid / (BN / TN);

    // Move to this block's patch.
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM * TN] = {0.0f};
    float ra[TM], rb[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * BK; i += nthreads)
            As[i] = A[(i / BK) * K + k0 + (i % BK)];
        for (int i = tid; i < BK * BN; i += nthreads)
            Bs[i] = B[(k0 + i / BN) * N + (i % BN)];
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) ra[i] = As[(trow * TM + i) * BK + k];
            for (int j = 0; j < TN; ++j) rb[j] = Bs[k * BN + tcol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] += ra[i] * rb[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            C[(trow * TM + i) * N + tcol * TN + j] = acc[i * TN + j];
}

// ---------------------------------------------------------------------------
// Stage 4: vectorized loads. Same blocking as stage 3, plus:
//   * global -> shared copies go through float4 (one 128-bit LDG per 4 floats),
//   * the A tile is stored transposed (As[k][row]) so the per-k register fill
//     reads contiguous shared memory, which the compiler can also vectorize,
//   * C is written back as float4.
// Requires the stage 3 divisibility, plus BK % 4 == 0, TN % 4 == 0, and
// K % 4 == 0, N % 4 == 0 so every float4 access is 16-byte aligned.
// Launch: block ((BM/TM)*(BN/TN)), grid (N/BN, M/BM).
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_vectorized(int M, int N, int K, const float *A, const float *B, float *C)
{
    __shared__ float As[BK * BM]; // transposed: As[k * BM + row]
    __shared__ float Bs[BK * BN];

    const int nthreads = (BM / TM) * (BN / TN);
    const int tid = threadIdx.x;
    const int tcol = tid % (BN / TN);
    const int trow = tid / (BN / TN);

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM * TN] = {0.0f};
    float ra[TM], rb[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * BK / 4; i += nthreads) {
            const int r = i / (BK / 4);
            const int c = (i % (BK / 4)) * 4;
            const float4 v = reinterpret_cast<const float4 *>(&A[r * K + k0 + c])[0];
            As[(c + 0) * BM + r] = v.x;
            As[(c + 1) * BM + r] = v.y;
            As[(c + 2) * BM + r] = v.z;
            As[(c + 3) * BM + r] = v.w;
        }
        for (int i = tid; i < BK * BN / 4; i += nthreads) {
            const int r = i / (BN / 4);
            const int c = (i % (BN / 4)) * 4;
            reinterpret_cast<float4 *>(&Bs[r * BN + c])[0] =
                reinterpret_cast<const float4 *>(&B[(k0 + r) * N + c])[0];
        }
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) ra[i] = As[k * BM + trow * TM + i];
            for (int j = 0; j < TN; ++j) rb[j] = Bs[k * BN + tcol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] += ra[i] * rb[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; j += 4)
            reinterpret_cast<float4 *>(&C[(trow * TM + i) * N + tcol * TN + j])[0] =
                make_float4(acc[i * TN + j], acc[i * TN + j + 1],
                            acc[i * TN + j + 2], acc[i * TN + j + 3]);
}
