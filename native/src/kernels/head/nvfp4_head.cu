#include "kernels/head/nvfp4_head.h"

#include "core/device.h"
#include "core/arena.h"
#include "kernels/common/l2.cuh"
#include "kernels/common/nvfp4.cuh"
#include "kernels/dense/cluster.cuh"
#include "model/tp_layout.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ingine::kernels::head {
namespace {

constexpr int HeadK = model::HIDDEN, Tile = 16, Threads = 256;
constexpr int HeadSplit = 6;
constexpr auto WeightPolicy = l2::Evict::first;

__global__ void matrix_amax(const __nv_bfloat16* w, size_t n, unsigned* amax) {
  __shared__ float warp_max[Threads / 32];
  float v = 0.f;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const float x = __bfloat162float(w[i]);
    v = fmaxf(v, isfinite(x) ? fabsf(x) : CUDART_INF_F);
  }
  for (int off = 16; off; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x / 32] = v;
  __syncthreads();
  if (threadIdx.x < 32) {
    v = threadIdx.x < Threads / 32 ? warp_max[threadIdx.x] : 0.f;
    for (int off = 16; off; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    if (threadIdx.x == 0) atomicMax(amax, __float_as_uint(v));
  }
}

__global__ void pack_rows(const __nv_bfloat16* w, size_t blocks, float g, uint64_t* codes, uint8_t* scales) {
  for (size_t b = size_t(blockIdx.x) * blockDim.x + threadIdx.x; b < blocks; b += size_t(gridDim.x) * blockDim.x) {
    float v[16], maximum = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      v[i] = __bfloat162float(w[16 * b + i]);
      maximum = fmaxf(maximum, fabsf(v[i]));
    }
    const __nv_fp8_e4m3 s(maximum > 0.f && g > 0.f ? __fdiv_rn(maximum, __fmul_rn(6.f, g)) : 0.f);
    scales[b] = s.__x;
    const float denominator = __fmul_rn(g, float(s));
    uint64_t packed = 0;
    if (denominator > 0.f) {
#pragma unroll
      for (int i = 0; i < 16; ++i)
        packed |= uint64_t(__nv_fp4_e2m1(__fdiv_rn(v[i], denominator)).__x & 15u) << (4 * i);
    }
    codes[b] = packed;
  }
}

struct WeightChunk {
  uint4 codes{};
  uint16_t scales = 0;
};

template<int K>
__device__ __forceinline__ WeightChunk load(Nvfp4Matrix w, int row, int chunk, uint64_t policy) {
  WeightChunk v;
  if (row < w.rows) {
    const uint8_t* codes = w.codes + size_t(row) * (K / 2) + 16 * chunk;
    const uint8_t* scales = w.scales + size_t(row) * (K / 16) + 2 * chunk;
    if constexpr (l2::hinted<WeightPolicy>) {
      asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
          : "=r"(v.codes.x), "=r"(v.codes.y), "=r"(v.codes.z), "=r"(v.codes.w) : "l"(codes), "l"(policy));
    } else {
      v.codes = __ldg(reinterpret_cast<const uint4*>(codes));
    }
    v.scales = __ldg(reinterpret_cast<const uint16_t*>(scales));
  }
  return v;
}

__device__ __forceinline__ Candidate better(Candidate a, Candidate b) {
  return b.val > a.val || (b.val == a.val && b.global_id < a.global_id) ? b : a;
}

template<int K, int Split, bool Head, bool Raw = false>
__global__ void __launch_bounds__(Split * 32)
project(const __nv_bfloat16* x, int M, Nvfp4Matrix w, float* logits, Candidate* scratch) {
  __shared__ float partial[Split][32][4];
  __shared__ Candidate candidates[Head ? 8 : 1][Tile];
  constexpr int Chunks = K / (32 * 4 * Split);
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const int row = blockIdx.x * Tile + lane / 4;
  const int token = lane / 4, t = lane & 3;
  const uint64_t policy = l2::policy<WeightPolicy>();
  dense::pdl_wait();
  dense::pdl_trigger();
  float sum[4] = {};
#pragma unroll
  for (int step = 0; step < Chunks; ++step) {
    const int c = (warp * Chunks + step) * 4 + t;
    const WeightChunk lo = load<K>(w, row, c, policy), hi = load<K>(w, row + 8, c, policy);
    uint32_t a[16], b[16];
    nvfp4::decode(lo.codes, lo.scales, w.outer_scale, a);
    nvfp4::decode(hi.codes, hi.scales, w.outer_scale, b);
    const auto* xp = reinterpret_cast<const uint4*>(x + (token < M ? token * K + 32 * c : 0));
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      uint4 in{};
      if (token < M) in = xp[p];
      nvfp4::mma_bf16(sum, a[4 * p], b[4 * p], a[4 * p + 1], b[4 * p + 1], in.x, in.y);
      nvfp4::mma_bf16(sum, a[4 * p + 2], b[4 * p + 2], a[4 * p + 3], b[4 * p + 3], in.z, in.w);
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) partial[warp][lane][i] = sum[i];
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int m = 2 * t + (i & 1), r = row + 8 * (i / 2);
      if (m < M) {
        float dot = 0.f;
#pragma unroll
        for (int j = 0; j < Split; ++j) dot += partial[j][lane][i];
        if constexpr (Head) {
          const float score = r >= w.rows ? -CUDART_INF_F : Raw ? dot :
              30.f * tanhf(__bfloat162float(__float2bfloat16_rn(dot)) / 30.f);
          if (logits && r < w.rows) logits[size_t(m) * w.rows + r] = score;
          if (scratch) candidates[m][r % Tile] = {score, r < w.rows ? w.v_begin + r : INT32_MAX};
        } else if (r < w.rows) {
          logits[size_t(m) * w.rows + r] = __bfloat162float(__float2bfloat16_rn(dot));
        }
      }
    }
  }
  if constexpr (!Head) return;
  if (scratch) {
    __syncthreads();
    if (threadIdx.x < M) {
      Candidate best = candidates[threadIdx.x][0];
#pragma unroll
      for (int i = 1; i < Tile; ++i) best = better(best, candidates[threadIdx.x][i]);
      scratch[size_t(threadIdx.x) * gridDim.x + blockIdx.x] = best;
    }
  }
}

__global__ void reduce_candidates(const Candidate* scratch, Candidate* out, int tiles) {
  __shared__ Candidate best_by_warp[Threads / 32];
  dense::pdl_wait();
  Candidate best{-CUDART_INF_F, INT32_MAX};
  for (int i = threadIdx.x; i < tiles; i += blockDim.x) best = better(best, scratch[blockIdx.x * tiles + i]);
  for (int d = 16; d; d >>= 1)
    best = better(best, {__shfl_xor_sync(0xffffffffu, best.val, d), __shfl_xor_sync(0xffffffffu, best.global_id, d)});
  if ((threadIdx.x & 31) == 0) best_by_warp[threadIdx.x / 32] = best;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (Candidate v : best_by_warp) best = better(best, v);
    out[blockIdx.x] = best;
  }
}

unsigned blocks_for(size_t n) { return unsigned(std::min<size_t>((n + Threads - 1) / Threads, 4096)); }

}  // namespace

float nvfp4_outer_scale(const __nv_bfloat16* weights, size_t elements, cudaStream_t stream) {
  if (!weights || !elements) throw std::invalid_argument("empty NVFP4 calibration matrix");
  DeviceBuffer storage(sizeof(unsigned));
  auto* device = static_cast<unsigned*>(storage.p);
  CUDA_CHECK(cudaMemsetAsync(device, 0, sizeof(unsigned), stream));
  matrix_amax<<<blocks_for(elements), Threads, 0, stream>>>(weights, elements, device);
  CUDA_CHECK(cudaGetLastError());
  unsigned bits;
  CUDA_CHECK(cudaMemcpyAsync(&bits, device, sizeof(bits), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const float amax = std::bit_cast<float>(bits);
  if (!std::isfinite(amax)) throw std::runtime_error("nonfinite NVFP4 weight");
  return amax / (448.f * 6.f);
}

void nvfp4_pack_rows(const __nv_bfloat16* weights, int rows, float outer_scale,
                     uint8_t* codes, uint8_t* scales, cudaStream_t stream) {
  nvfp4_pack_rows(weights, rows, HeadK, outer_scale, codes, scales, stream);
}

void nvfp4_pack_rows(const __nv_bfloat16* weights, int rows, int cols, float outer_scale,
                     uint8_t* codes, uint8_t* scales, cudaStream_t stream) {
  if (!weights || rows <= 0 || cols <= 0 || cols % 16 || !(outer_scale >= 0.f) || !std::isfinite(outer_scale) || !codes || !scales)
    throw std::invalid_argument("invalid NVFP4 packing arguments");
  const size_t n = size_t(rows) * (cols / 16);
  pack_rows<<<blocks_for(n), Threads, 0, stream>>>(weights, n, outer_scale, reinterpret_cast<uint64_t*>(codes), scales);
  CUDA_CHECK(cudaGetLastError());
}

std::size_t nvfp4_candidate_workspace(int M, int rows) {
  return size_t(M) * ((rows + Tile - 1) / Tile) * sizeof(Candidate);
}

void gemv_nvfp4(const __nv_bfloat16* x, int cols, Nvfp4Matrix w, float* out, cudaStream_t stream) {
  if (!x || !out || !w.codes || !w.scales || w.rows < 1)
    throw std::invalid_argument("invalid gemv_nvfp4 arguments");
  const dim3 tiles((w.rows - 1) / Tile + 1);
  if (cols == 2304)
    dense::launch_ex(project<2304, 6, false>, tiles, 6 * 32, 0, 1, true, stream, x, 8, w, out, nullptr);
  else if (cols == 9216)
    dense::launch_ex(project<9216, 12, false>, tiles, 12 * 32, 0, 1, true, stream, x, 8, w, out, nullptr);
  else
    throw std::invalid_argument("gemv_nvfp4 supports input widths 2304 and 9216");
}

void project_nvfp4(const __nv_bfloat16* x, int M, int cols, Nvfp4Matrix w, float* out, cudaStream_t stream) {
  const dim3 tiles((w.rows - 1) / Tile + 1);
  if (cols == 2048)
    dense::launch_ex(project<2048, 8, false>, tiles, 256, 0, 1, false, stream, x, M, w, out, nullptr);
  else if (cols == 2304)
    dense::launch_ex(project<2304, 6, false>, tiles, 192, 0, 1, false, stream, x, M, w, out, nullptr);
  else if (cols == 3456)
    dense::launch_ex(project<3456, 9, false>, tiles, 288, 0, 1, false, stream, x, M, w, out, nullptr);
  else if (cols == 4096)
    dense::launch_ex(project<4096, 8, false>, tiles, 256, 0, 1, false, stream, x, M, w, out, nullptr);
  else if (cols == 6912)
    dense::launch_ex(project<6912, 9, false>, tiles, 288, 0, 1, false, stream, x, M, w, out, nullptr);
  else
    throw std::invalid_argument("Unsupported DFlash2 NVFP4 projection width");
}

void head_nvfp4(const __nv_bfloat16* x, int M, Nvfp4Matrix w, float* logits, Candidate* candidates,
                Candidate* scratch, bool pdl, cudaStream_t stream) {
  if (!x || M < 1 || M > 8 || !w.codes || !w.scales || w.rows < 1 || w.v_begin < 0 ||
      w.v_begin > INT32_MAX - w.rows || (!logits && !candidates) || (candidates && !scratch))
    throw std::invalid_argument("invalid head_nvfp4 arguments");
  const int tiles = (w.rows - 1) / Tile + 1;
  dense::launch_ex(project<HeadK, HeadSplit, true>, dim3(tiles), HeadSplit * 32, 0, 1, pdl, stream, x, M, w, logits,
                   candidates ? scratch : nullptr);
  if (candidates) dense::launch_ex(reduce_candidates, dim3(M), Threads, 0, 1, pdl, stream, scratch, candidates, tiles);
}

void head_nvfp4_raw(const __nv_bfloat16* x, int M, Nvfp4Matrix w, float* dots, Candidate* candidates,
                    Candidate* scratch, cudaStream_t stream) {
  const int tiles = (w.rows - 1) / Tile + 1;
  dense::launch_ex(project<HeadK, HeadSplit, true, true>, dim3(tiles), HeadSplit * 32, 0, 1, false, stream,
                   x, M, w, dots, scratch);
  reduce_candidates<<<M, Threads, 0, stream>>>(scratch, candidates, tiles);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::head
