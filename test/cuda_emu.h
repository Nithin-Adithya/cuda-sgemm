// Minimal CPU emulation of the CUDA execution model, just enough to run
// src/kernels.cuh on a machine without an NVIDIA GPU and check the indexing.
//
// Each CUDA thread of a block becomes a std::thread; __syncthreads() is a
// std::barrier across the block. Blocks run one after another, so __shared__
// can be a plain static. This is for correctness only, not performance.
#pragma once

#include <barrier>
#include <thread>
#include <vector>

struct uint3 { unsigned x, y, z; };
struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
struct float4 { float x, y, z, w; };
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }

namespace emu {
inline thread_local uint3 threadIdx, blockIdx;
inline dim3 blockDim, gridDim;
inline std::barrier<> *sync = nullptr;

template <typename Kernel, typename... Args>
void launch(Kernel kernel, dim3 grid, dim3 block, Args... args)
{
    gridDim = grid;
    blockDim = block;
    std::barrier<> bar(block.x * block.y * block.z);
    sync = &bar;
    for (unsigned bz = 0; bz < grid.z; ++bz)
    for (unsigned by = 0; by < grid.y; ++by)
    for (unsigned bx = 0; bx < grid.x; ++bx) {
        std::vector<std::thread> threads;
        for (unsigned tz = 0; tz < block.z; ++tz)
        for (unsigned ty = 0; ty < block.y; ++ty)
        for (unsigned tx = 0; tx < block.x; ++tx)
            threads.emplace_back([=] {
                threadIdx = {tx, ty, tz};
                blockIdx = {bx, by, bz};
                kernel(args...);
            });
        for (auto &t : threads) t.join();
    }
    sync = nullptr;
}
} // namespace emu

using emu::blockDim;
using emu::blockIdx;
using emu::gridDim;
using emu::threadIdx;

#define __global__
#define __shared__ static
#define __syncthreads() emu::sync->arrive_and_wait()
