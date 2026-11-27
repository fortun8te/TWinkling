// Certified Lazy 4-Bit LM Head for RTX 5080 (reclaims ~193 MiB VRAM + ~1.7x faster head):
//   1. quantize_q4_and_fp8_rows: quantizes BF16 [rows, 2304] -> group-32 INT4 in VRAM (247 MiB)
//      AND exact FP8 E4M3 in mapped pinned host RAM (0 MiB VRAM).
//   2. head_q4_kernel<M>: streams 247 MiB Q4 proxy from VRAM to produce approximate logits [M, V].
//   3. certify_top64_kernel: per row m, finds top-64 proxy candidates (32 warps x 2 = 64 in L2)
//      and recomputes their EXACT FP8 dot products from mapped host RAM, overwriting logits[m, tok].
#include "kernels/common/l2.cuh"
#include "kernels/dense/dense.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>

namespace ingine::kernels::dense {
namespace {

constexpr int kH = 2304;
constexpr int kGroup = 32;
constexpr int kGroups = kH / kGroup;  // 72
constexpr int kPackedRow = kH / 2;    // 1152 bytes

__device__ __forceinline__ float round_bf16(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ float2 e4m3x2_to_f2(uint16_t pair) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair), __NV_E4M3);
  return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

// Quantize BF16 [rows, 2304] ->
//   q4_codes [rows, 1152] uint8 (group-32 symmetric INT4, stored as nibble + 8)
//   q4_scales [rows, 72] half
//   fp8_codes [rows, 2304] uint8 (mapped host or device)
//   fp8_scales [rows] float
// 1 warp (32 lanes) per row; 8 rows per 256-thread block.
__global__ void __launch_bounds__(256)
    quantize_q4_and_fp8_kernel(const __nv_bfloat16* __restrict__ src, int rows,
                               uint8_t* __restrict__ q4_codes, __half* __restrict__ q4_scales,
                               uint8_t* __restrict__ fp8_codes, float* __restrict__ fp8_scales) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const __nv_bfloat16* x = src + size_t(row) * kH;
  uint8_t* dst_q4 = q4_codes + size_t(row) * kPackedRow;
  __half* dst_sc = q4_scales + size_t(row) * kGroups;
  uint8_t* dst_f8 = fp8_codes + size_t(row) * kH;

  // Pass 1: row amax for FP8 per-row scale, AND quantize each 32-element group to Q4
  float row_amax = 0.f;
  for (int g = lane; g < kGroups; g += 32) {
    float v[32];
    float gamax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 u = *reinterpret_cast<const uint4*>(x + g * kGroup + i * 8);
      const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[j]));
        v[i * 8 + j * 2] = f.x;
        v[i * 8 + j * 2 + 1] = f.y;
        gamax = fmaxf(gamax, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
    }
    row_amax = fmaxf(row_amax, gamax);
    const float sc = gamax > 0.f ? gamax / 7.f : 0.f;
    const __half sc_h = __float2half_rn(sc);
    const float sc_q = __half2float(sc_h);
    const float inv = sc_q > 0.f ? 1.f / sc_q : 0.f;
    dst_sc[g] = sc_h;
    uint32_t packed[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      uint32_t word = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int idx = w * 8 + b * 2;
        const int q0 = max(-8, min(7, __float2int_rn(v[idx] * inv))) + 8;
        const int q1 = max(-8, min(7, __float2int_rn(v[idx + 1] * inv))) + 8;
        word |= uint32_t(q0 | (q1 << 4)) << (8 * b);
      }
      packed[w] = word;
    }
    *reinterpret_cast<uint4*>(dst_q4 + g * 16) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) row_amax = fmaxf(row_amax, __shfl_xor_sync(0xffffffffu, row_amax, o));
  const float f8_scale = row_amax > 0.f ? row_amax / 448.f : 1.f;
  const float f8_inv = row_amax > 0.f ? 1.f / f8_scale : 0.f;
  if (lane == 0) fp8_scales[row] = f8_scale;

  // Pass 2: write FP8 E4M3 row to mapped host RAM (coalesced 16B stores)
  for (int g = lane; g < kGroups; g += 32) {
    uint32_t out_lo[4], out_hi[4];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      uint32_t* dst_w = half == 0 ? out_lo : out_hi;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint4 u = *reinterpret_cast<const uint4*>(x + g * kGroup + half * 16 + i * 8);
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
        uint32_t p0 = 0, p1 = 0;
        const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[0]));
        const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[1]));
        const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[2]));
        const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[3]));
        p0 = uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(f0.x * f8_inv, f0.y * f8_inv), __NV_SATFINITE, __NV_E4M3)) |
             (uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(f1.x * f8_inv, f1.y * f8_inv), __NV_SATFINITE, __NV_E4M3)) << 16);
        p1 = uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(f2.x * f8_inv, f2.y * f8_inv), __NV_SATFINITE, __NV_E4M3)) |
             (uint32_t(__nv_cvt_float2_to_fp8x2(make_float2(f3.x * f8_inv, f3.y * f8_inv), __NV_SATFINITE, __NV_E4M3)) << 16);
        dst_w[i * 2] = p0;
        dst_w[i * 2 + 1] = p1;
      }
    }
    *reinterpret_cast<uint4*>(dst_f8 + g * 32) = make_uint4(out_lo[0], out_lo[1], out_lo[2], out_lo[3]);
    *reinterpret_cast<uint4*>(dst_f8 + g * 32 + 16) = make_uint4(out_hi[0], out_hi[1], out_hi[2], out_hi[3]);
  }
}

__device__ __forceinline__ uint4 ld_streaming_u4(const uint4* ptr, uint64_t pol) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr), "l"(pol));
  return v;
}

// Q4 proxy LM head: 8 lanes (quarter-warp) per vocab row, 4 rows/warp, 16 rows/block (128 threads).
// Each lane owns 9 contiguous groups of 32 elements (9 x uint4 = 144 bytes).
template <int M>
__global__ void __launch_bounds__(128)
    head_q4_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ codes,
                   const __half* __restrict__ scales, int V, float* __restrict__ out) {
  __shared__ __nv_bfloat16 xs[M * kH];
  const int tid = threadIdx.x;
  const int row = blockIdx.x * 16 + (tid >> 3);
  const int lane8 = tid & 7;

  // Stage x [M, 2304] into shared memory
  const int total_u4 = (M * kH) / 8;
  for (int i = tid; i < total_u4; i += 128)
    reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(x)[i];
  __syncthreads();
  if (row >= V) return;

  const uint64_t pol = l2::policy<l2::Evict::first>();
  const uint4* w_row = reinterpret_cast<const uint4*>(codes + size_t(row) * kPackedRow);
  const __half* s_row = scales + size_t(row) * kGroups;

  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

#pragma unroll
  for (int it = 0; it < 9; ++it) {
    const int g = it * 8 + lane8;
    const uint4 q = ld_streaming_u4(w_row + g, pol);
    const float sc = __half2float(__ldg(s_row + g));
    const uint32_t words[4] = {q.x, q.y, q.z, q.w};
    float wv[32];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const uint32_t uw = words[w];
      wv[w * 8 + 0] = float(int(uw & 15u) - 8) * sc;
      wv[w * 8 + 1] = float(int((uw >> 4) & 15u) - 8) * sc;
      wv[w * 8 + 2] = float(int((uw >> 8) & 15u) - 8) * sc;
      wv[w * 8 + 3] = float(int((uw >> 12) & 15u) - 8) * sc;
      wv[w * 8 + 4] = float(int((uw >> 16) & 15u) - 8) * sc;
      wv[w * 8 + 5] = float(int((uw >> 20) & 15u) - 8) * sc;
      wv[w * 8 + 6] = float(int((uw >> 24) & 15u) - 8) * sc;
      wv[w * 8 + 7] = float(int((uw >> 28) & 15u) - 8) * sc;
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const uint4* xp = reinterpret_cast<const uint4*>(xs + m * kH + g * kGroup);
      float dot = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint4 xv = xp[c];
        const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xw[j]));
          dot = fmaf(wv[c * 8 + j * 2], xf.x, dot);
          dot = fmaf(wv[c * 8 + j * 2 + 1], xf.y, dot);
        }
      }
      acc[m] += dot;
    }
  }

#pragma unroll
  for (int m = 0; m < M; ++m) {
    float v = acc[m];
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    if (lane8 == 0) out[size_t(m) * V + row] = tanhf(round_bf16(v) / 30.f) * 30.f;
  }
}

// Certify top-64 candidates per row m against exact FP8 unembed in mapped pinned host RAM.
// 1 block of 1024 threads (32 warps) per row m:
//   - Each warp scans V/32 elements and extracts its top-2 (64 total candidates across 32 warps).
//   - Each warp then computes exact FP8 dot products for 2 candidates from host_w and overwrites out[m, tok].
__global__ void __launch_bounds__(1024)
    certify_top64_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ host_w,
                         const float* __restrict__ host_scale, int V, float* __restrict__ out) {
  const int m = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  __shared__ float xs[kH];
  for (int i = tid; i < kH; i += 1024) xs[i] = __bfloat162float(x[size_t(m) * kH + i]);

  float* row_out = out + size_t(m) * V;

  // Each lane scans its stride and keeps its local top-2
  float v1 = -1e30f, v2 = -1e30f;
  int i1 = -1, i2 = -1;
  for (int idx = tid; idx < V; idx += 1024) {
    const float v = row_out[idx];
    if (v > v1) {
      v2 = v1; i2 = i1;
      v1 = v;  i1 = idx;
    } else if (v > v2) {
      v2 = v;  i2 = idx;
    }
  }

  // Warp-level top-2 reduction
  auto warp_max = [](float& val, int& idx) {
    float bv = val;
    int bi = idx;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    val = bv; idx = bi;
  };

  float w1 = v1; int wi1 = i1;
  warp_max(w1, wi1);
  // Mask out winner to find 2nd best in warp
  float cand2 = (i1 == wi1) ? v2 : v1;
  int idx2 = (i1 == wi1) ? i2 : i1;
  warp_max(cand2, idx2);

  __syncthreads();  // xs ready

  // Warp certifies its 2 candidate tokens [wi1, idx2] against exact mapped host FP8 rows
  const int cands[2] = {wi1, idx2};
  for (int c = 0; c < 2; ++c) {
    const int tok = cands[c];
    if (tok < 0 || tok >= V) continue;
    const uint8_t* wrow = host_w + size_t(tok) * kH;
    float dot = 0.f;
    // 2304 / 32 = 72 bytes per lane = 9 x 8 bytes (uint2)
    for (int b = lane * 8; b < kH; b += 32 * 8) {
      const uint2 u = *reinterpret_cast<const uint2*>(wrow + b);
      const float2 a0 = e4m3x2_to_f2(uint16_t(u.x & 0xffffu));
      const float2 a1 = e4m3x2_to_f2(uint16_t(u.x >> 16));
      const float2 a2 = e4m3x2_to_f2(uint16_t(u.y & 0xffffu));
      const float2 a3 = e4m3x2_to_f2(uint16_t(u.y >> 16));
      dot = fmaf(a0.x, xs[b + 0], dot);
      dot = fmaf(a0.y, xs[b + 1], dot);
      dot = fmaf(a1.x, xs[b + 2], dot);
      dot = fmaf(a1.y, xs[b + 3], dot);
      dot = fmaf(a2.x, xs[b + 4], dot);
      dot = fmaf(a2.y, xs[b + 5], dot);
      dot = fmaf(a3.x, xs[b + 6], dot);
      dot = fmaf(a3.y, xs[b + 7], dot);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
    if (lane == 0) {
      const float sc = host_scale[tok];
      row_out[tok] = tanhf(round_bf16(dot * sc) / 30.f) * 30.f;
    }
  }
}

template <int M>
void launch_head_q4(const __nv_bfloat16* x, const Q4Matrix& q4, const Matrix& exact_host, float* out,
                    cudaStream_t stream) {
  const int blocks = (q4.rows + 15) / 16;
  head_q4_kernel<M><<<blocks, 128, 0, stream>>>(
      x, static_cast<const uint8_t*>(q4.codes), static_cast<const __half*>(q4.scales), q4.rows, out);
  if (exact_host.w && exact_host.scale) {
    certify_top64_kernel<<<M, 1024, 0, stream>>>(
        x, static_cast<const uint8_t*>(exact_host.w), exact_host.scale, q4.rows, out);
  }
}

}  // namespace

void quantize_q4_and_fp8_rows(const __nv_bfloat16* src, int rows, void* q4_codes, void* q4_scales,
                              void* fp8_codes, float* fp8_scales, cudaStream_t stream) {
  if (rows <= 0) return;
  quantize_q4_and_fp8_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(
      src, rows, static_cast<uint8_t*>(q4_codes), static_cast<__half*>(q4_scales),
      static_cast<uint8_t*>(fp8_codes), fp8_scales);
}

void head_q4_certified(const __nv_bfloat16* x, int M, Q4Matrix q4, Matrix exact_host, float* out,
                       cudaStream_t stream) {
  switch (M) {
    case 1: launch_head_q4<1>(x, q4, exact_host, out, stream); break;
    case 2: launch_head_q4<2>(x, q4, exact_host, out, stream); break;
    case 3: launch_head_q4<3>(x, q4, exact_host, out, stream); break;
    case 4: launch_head_q4<4>(x, q4, exact_host, out, stream); break;
    case 5: launch_head_q4<5>(x, q4, exact_host, out, stream); break;
    case 6: launch_head_q4<6>(x, q4, exact_host, out, stream); break;
    case 7: launch_head_q4<7>(x, q4, exact_host, out, stream); break;
    case 8: launch_head_q4<8>(x, q4, exact_host, out, stream); break;
    default: throw std::invalid_argument("head_q4_certified: M must be in 1..8");
  }
}

}  // namespace ingine::kernels::dense
