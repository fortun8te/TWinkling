// Routed-expert decode to BF16 (ported from engine/csrc/prefill/dequant.cu; kernel bodies unchanged).
//
// A prefill that multiplies many tokens per expert can decode the selected experts' matrices once into a
// BF16 scratch buffer and run grouped GEMMs on it. Decoding follows the checkpoint semantics exactly
// (FP32 product, one BF16 rounding), matching model.decode_experts:
//   NVFP4: value = e2m1(code) * e4m3(block scale per 16) * factor[expert]
//   MXFP8: value = e4m3(code) * 2^(e8m0 per 32 - 127)
// One thread decodes one scale block (16 or 32 values) and writes it with 16-byte stores.
#include "kernels/moe/moe.h"
#include "kernels/moe/scale6.cuh"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <stdexcept>

namespace ingine::kernels::moe {
namespace {

__constant__ float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

__global__ void dequant_nvfp4(const uint8_t* __restrict__ codes, const __nv_fp8_e4m3* __restrict__ scales,
                              const float* __restrict__ factor, const int32_t* __restrict__ experts,
                              __nv_bfloat16* __restrict__ out, int64_t blocks_per_expert, int64_t total,
                              int scale_base) {
    int64_t b = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (b >= total) return;
    int64_t slot = b / blocks_per_expert, within = b % blocks_per_expert;
    int64_t e = experts[slot];
    int64_t src = e * blocks_per_expert + within;
    // the reference forms e2m1 * block_scale, then multiplies by the expert factor (both FP32)
    float scale;
    if (scale_base >= 0) {
        __nv_fp8_e4m3 f8;
        f8.__x = unpack1_6bit(reinterpret_cast<const uint32_t*>(scales), uint64_t(src), uint8_t(scale_base));
        scale = float(f8);
    } else {
        scale = float(scales[src]);
    }
    const float f = factor[e];
    uint2 c = *reinterpret_cast<const uint2*>(codes + src * 8);
    uint32_t w[2] = {c.x, c.y};
    __align__(16) __nv_bfloat16 v[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t byte = (w[i / 4] >> (8 * (i % 4))) & 0xFF;
        v[2 * i] = __float2bfloat16((kE2M1[byte & 0xF] * scale) * f);
        v[2 * i + 1] = __float2bfloat16((kE2M1[byte >> 4] * scale) * f);
    }
    uint4* dst = reinterpret_cast<uint4*>(out + (slot * blocks_per_expert + within) * 16);
    dst[0] = reinterpret_cast<uint4*>(v)[0];
    dst[1] = reinterpret_cast<uint4*>(v)[1];
}

__global__ void minmax_init_kernel(uint32_t* out_minmax) {
    out_minmax[0] = 255u;
    out_minmax[1] = 0u;
}

__global__ void minmax_u8_kernel(const uint8_t* __restrict__ data, size_t count, uint32_t* __restrict__ out_minmax) {
    __shared__ uint32_t smin[8], smax[8];
    const size_t nvec = count >> 4;
    const uint4* vec = reinterpret_cast<const uint4*>(data);
    uint32_t lo = 255u, hi = 0u;
    auto update_word = [&](uint32_t w) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t v = (w >> (8 * b)) & 0xffu;
            lo = min(lo, v);
            hi = max(hi, v);
        }
    };
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < nvec; i += stride) {
        const uint4 q = vec[i];
        update_word(q.x);
        update_word(q.y);
        update_word(q.z);
        update_word(q.w);
    }
    if (blockIdx.x == 0) {
        for (size_t j = nvec * 16 + threadIdx.x; j < count; j += blockDim.x) {
            const uint32_t v = data[j];
            lo = min(lo, v);
            hi = max(hi, v);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        lo = min(lo, __shfl_xor_sync(0xffffffffu, lo, o));
        hi = max(hi, __shfl_xor_sync(0xffffffffu, hi, o));
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) smin[warp] = lo, smax[warp] = hi;
    __syncthreads();
    if (threadIdx.x < 8) {
        lo = smin[threadIdx.x], hi = smax[threadIdx.x];
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            lo = min(lo, __shfl_xor_sync(0xffu, lo, o));
            hi = max(hi, __shfl_xor_sync(0xffu, hi, o));
        }
        if (threadIdx.x == 0) {
            atomicMin(&out_minmax[0], lo);
            atomicMax(&out_minmax[1], hi);
        }
    }
}

__global__ void pack_6bit_kernel(const uint8_t* __restrict__ src, uint32_t* __restrict__ dst, size_t ngroups,
                                 uint8_t base) {
    const size_t g = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (g >= ngroups) return;
    if (g == 0) dst[3 * ngroups] = 0u;
    const uint4 q = *reinterpret_cast<const uint4*>(src + g * 16);
    uint8_t bytes[16];
    const uint32_t ws[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        bytes[4 * i + 0] = uint8_t(ws[i] & 0xffu);
        bytes[4 * i + 1] = uint8_t((ws[i] >> 8) & 0xffu);
        bytes[4 * i + 2] = uint8_t((ws[i] >> 16) & 0xffu);
        bytes[4 * i + 3] = uint8_t((ws[i] >> 24) & 0xffu);
    }
    uint32_t packed[3];
    pack16_6bit(bytes, packed, base);
    dst[3 * g + 0] = packed[0];
    dst[3 * g + 1] = packed[1];
    dst[3 * g + 2] = packed[2];
}

__global__ void dequant_mxfp8(const __nv_fp8_e4m3* __restrict__ codes, const uint8_t* __restrict__ scales,
                              const int32_t* __restrict__ experts, __nv_bfloat16* __restrict__ out,
                              int64_t blocks_per_expert, int64_t total) {
    int64_t b = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (b >= total) return;
    int64_t slot = b / blocks_per_expert, within = b % blocks_per_expert;
    int64_t src = experts[slot] * blocks_per_expert + within;
    float s = exp2f(float(scales[src]) - 127.f);
    const uint4* c = reinterpret_cast<const uint4*>(codes + src * 32);
    uint4* dst = reinterpret_cast<uint4*>(out + (slot * blocks_per_expert + within) * 32);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        uint4 raw = c[q];
        const __nv_fp8_e4m3* f = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
        __align__(16) __nv_bfloat16 v[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = __float2bfloat16(float(f[i]) * s);
        dst[2 * q] = reinterpret_cast<uint4*>(v)[0];
        dst[2 * q + 1] = reinterpret_cast<uint4*>(v)[1];
    }
}

}  // namespace

// codes [128, R, C/2] uint8 (NVFP4) or [128, R, C] float8_e4m3fn (MXFP8); experts int32 [n];
// out bf16 [n, R, C].
void dequant_experts(const void* codes, const void* scales, const float* factor, const std::int32_t* experts, int n,
                     int rows, int cols, Format fmt, __nv_bfloat16* out, cudaStream_t stream, int scale_base) {
    const bool nv = fmt == Format::NVFP4;
    if (n < 0 || rows <= 0 || cols % (nv ? 16 : 32) != 0 || (nv && !factor) ||
        scale_base < -1 || scale_base > 192 || (!nv && scale_base >= 0))
        throw std::invalid_argument("moe: dequant_experts: bad arguments");
    if (n == 0) return;
    int64_t per_expert_blocks = int64_t(rows) * (cols / (nv ? 16 : 32));
    int64_t total = n * per_expert_blocks;
    int threads = 256;
    int64_t grid = (total + threads - 1) / threads;
    if (nv) {
        dequant_nvfp4<<<grid, threads, 0, stream>>>(
            static_cast<const uint8_t*>(codes), static_cast<const __nv_fp8_e4m3*>(scales), factor, experts, out,
            per_expert_blocks, total, scale_base);
    } else {
        dequant_mxfp8<<<grid, threads, 0, stream>>>(
            static_cast<const __nv_fp8_e4m3*>(codes), static_cast<const uint8_t*>(scales), experts, out,
            per_expert_blocks, total);
    }
    CUDA_CHECK(cudaGetLastError());
}

void nvfp4_scales_minmax(const std::uint8_t* scales, std::size_t count, std::uint32_t* out_minmax,
                         cudaStream_t stream) {
    if (!out_minmax || (count > 0 && !scales) || (std::uintptr_t(scales) & 15) != 0)
        throw std::invalid_argument("moe: nvfp4_scales_minmax: bad arguments");
    minmax_init_kernel<<<1, 1, 0, stream>>>(out_minmax);
    if (count > 0) {
        const size_t nblocks = std::min<size_t>(1024, std::max<size_t>(1, ((count >> 4) + 255) / 256));
        minmax_u8_kernel<<<unsigned(nblocks), 256, 0, stream>>>(scales, count, out_minmax);
    }
    CUDA_CHECK(cudaGetLastError());
}

void pack_nvfp4_scales_6bit(const std::uint8_t* src, std::uint32_t* dst, std::size_t count, std::uint8_t base,
                            cudaStream_t stream) {
    if (!src || !dst || (count % 16) != 0 || base > 192 || (std::uintptr_t(src) & 15) != 0 ||
        (std::uintptr_t(dst) & 3) != 0)
        throw std::invalid_argument("moe: pack_nvfp4_scales_6bit: bad arguments");
    if (count == 0) return;
    const size_t ngroups = count / 16;
    const size_t grid = (ngroups + 255) / 256;
    pack_6bit_kernel<<<unsigned(grid), 256, 0, stream>>>(src, dst, ngroups, base);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::moe
