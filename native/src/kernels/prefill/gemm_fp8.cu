// Blackwell SM120a block-scaled FP8 dense GEMM for prefill (wproj and wo when dense_fp8 is active).
// Adapted from Siphon's linear_fp8.cu (mxf8f6f4.block_scale.scale_vec::1X.m16n8k32, 4x BF16 tensor core
// throughput on SM120a), extended with per-product output row stride `out_ld` so multi-segment wproj
// projections execute in a single batched launch without dequant_rows or wscratch_.
#include "kernels/prefill/common.cuh"
#include "kernels/prefill/prefill.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>

namespace ingine::kernels::prefill {
namespace {

__device__ __forceinline__ void load8_bf16(const __nv_bfloat16* p, float* x) {
  const uint4 v = *reinterpret_cast<const uint4*>(p);
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[i]));
    x[2 * i] = f.x;
    x[2 * i + 1] = f.y;
  }
}

// Quantize BF16 rows [rows, k] to MXFP8: E4M3 values [rows, k] + UE8M0 scales [rows, k/32].
// One warp per row, 8 rows per 256-thread block. k must be a multiple of 32.
__global__ void __launch_bounds__(256)
    quantize_mxfp8_kernel(const __nv_bfloat16* __restrict__ x, int rows, int k,
                          uint8_t* __restrict__ values, uint8_t* __restrict__ scales) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= rows) return;
  const __nv_bfloat16* src = x + size_t(row) * k;
  uint8_t* dst_v = values + size_t(row) * k;
  uint8_t* dst_s = scales + size_t(row) * (k / 32);
  for (int b = threadIdx.x & 31; b < k / 32; b += 32) {
    float v[32];
#pragma unroll
    for (int i = 0; i < 32; i += 8) load8_bf16(src + b * 32 + i, v + i);
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
    const float raw = __fmul_rn(amax, 1.f / 448.f);
    constexpr float kMinScale = 0x1p-127f;
    int exp = ((__float_as_int(raw) + 0x007fffff) & 0x7f800000) >> 23;
    exp = raw <= kMinScale ? 0 : min(exp, 254);
    exp = raw != raw ? 255 : exp;
    const float scale = fmaxf(__int_as_float(exp << 23), kMinScale);
    float inv = exp == 0 ? 0x1p127f : __frcp_rn(scale);
    inv = exp == 255 ? 0.f : inv;
    dst_s[b] = uint8_t(exp);
    uint32_t words[8];
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      uint32_t word = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float val = __fmul_rn(v[4 * w + i], inv);
        val = val != val ? val : fminf(fmaxf(val, -448.f), 448.f);
        const __nv_fp8_e4m3 q(val);
        word |= uint32_t(q.__x) << (8 * i);
      }
      words[w] = word;
    }
    uint4* dst = reinterpret_cast<uint4*>(dst_v + b * 32);
    dst[0] = make_uint4(words[0], words[1], words[2], words[3]);
    dst[1] = make_uint4(words[4], words[5], words[6], words[7]);
  }
}

namespace gemm8 {

constexpr int kMaxProducts = 8;
constexpr int kBm = 128;
constexpr int kBn = 128;
constexpr int kBk = 128;
constexpr int kWarpsM = 2;
constexpr int kWarpsN = 4;
constexpr int kThreads = 32 * kWarpsM * kWarpsN;
constexpr int kStages = 3;
constexpr int kWarpRows = kBm / kWarpsM;
constexpr int kWarpCols = kBn / kWarpsN;
constexpr int kMi = kWarpRows / 16;
constexpr int kNi = kWarpCols / 8;
constexpr int kStageA = kBm * kBk;
constexpr int kStageB = kBn * kBk;
constexpr int kStageScales = kBm * 4;
constexpr int kStage = kStageA + kStageB + kStageScales;
constexpr int kSmemBytes = kStages * kStage;
static_assert(kSmemBytes <= 99 * 1024);

struct Product {
  const uint8_t* w;
  const float* w_scale;
  const uint8_t* x;
  const uint8_t* x_scale;
  __nv_bfloat16* out;
  int n;
  int out_ld;
  int tile_end;
};

struct Batch {
  Product p[kMaxProducts];
  int count;
  int m;
  int k;
};

__device__ __forceinline__ void cp_async16_u(uint32_t dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void cp_async4_u(uint32_t dst, const void* src, bool valid) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(dst), "l"(src), "r"(valid ? 4 : 0));
}

template <int kByte>
__device__ __forceinline__ void mma_fp8_a(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                          uint32_t b1, uint32_t scale_a) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3."
      "e4m3.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, "
      "{%13}, {%12,%12};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(scale_a),
        "h"(uint16_t(kByte)), "h"(uint16_t(0)), "r"(0x7f7f7f7fu));
}

__device__ __forceinline__ int offset(int row, int chunk) {
  return row * kBk + ((chunk ^ (row & 7)) << 4);
}

__global__ void __launch_bounds__(kThreads, 1) gemm_fp8_128x128(const __grid_constant__ Batch b) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int block = blockIdx.x;
  int s = 0;
  int tile_begin = 0;
  while (s + 1 < b.count && block >= b.p[s].tile_end) {
    tile_begin = b.p[s].tile_end;
    ++s;
  }
  const Product& p = b.p[s];
  const int m = b.m, n = p.n, k = b.k;
  const int tiles_m = (m + kBm - 1) / kBm;
  const int tile = block - tile_begin;
  const int m0 = (tile % tiles_m) * kBm;
  const int n0 = (tile / tiles_m) * kBn;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wn = warp % kWarpsN;
  const int wm = warp / kWarpsN;

  const int chunk = tid & 7;
  const uint8_t* a_src[4];
  const uint8_t* b_src[4];
  uint32_t dst[4];
  uint32_t a_live = 0, b_live = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = (tid >> 3) + 32 * i;
    dst[i] = offset(row, chunk);
    const bool a_ok = m0 + row < m;
    const bool b_ok = n0 + row < n;
    a_live |= uint32_t(a_ok) << i;
    b_live |= uint32_t(b_ok) << i;
    a_src[i] = p.x + size_t(a_ok ? m0 + row : 0) * k + chunk * 16;
    b_src[i] = p.w + size_t(b_ok ? n0 + row : 0) * k + chunk * 16;
  }
  const bool scale_ok = tid < kBm && m0 + tid < m;
  const uint8_t* scale_src = p.x_scale + size_t(scale_ok ? m0 + tid : 0) * (k / 32);

  const uint32_t smem0 = smem_addr(smem);
  auto load = [&](int stage, int step) {
    const uint32_t base = smem0 + stage * kStage;
    const int k0 = step * kBk;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      cp_async16_u(base + dst[i], a_src[i] + k0, (a_live >> i) & 1);
      cp_async16_u(base + kStageA + dst[i], b_src[i] + k0, (b_live >> i) & 1);
    }
    if (tid < kBm) cp_async4_u(base + kStageA + kStageB + 4 * tid, scale_src + step * 4, scale_ok);
  };

  const int a_row = wm * kWarpRows + (lane & 15);
  const int b_row = wn * kWarpCols + (lane & 7) + (lane >> 4) * 8;
  const int scale_row = wm * kWarpRows + (lane >> 2) + 8 * (lane & 1);

  float acc[kMi][kNi][4] = {};
  const int steps = k / kBk;
#pragma unroll
  for (int st = 0; st < kStages - 1; ++st) {
    if (st < steps) load(st, st);
    cp_commit();
  }

  int stage = 0;
  for (int step = 0; step < steps; ++step) {
    cp_wait<kStages - 2>();
    __syncthreads();
    const int next = step + kStages - 1;
    if (next < steps) load((stage + kStages - 1) % kStages, next);
    cp_commit();

    const uint32_t base = smem0 + stage * kStage;
    const uint32_t* scales = reinterpret_cast<const uint32_t*>(smem + stage * kStage + kStageA + kStageB);
    uint32_t sa[kMi];
#pragma unroll
    for (int mi = 0; mi < kMi; ++mi) sa[mi] = scales[scale_row + 16 * mi];

    auto k_step = [&]<int kK>() {
      uint32_t bf[kNi][2];
#pragma unroll
      for (int nj = 0; nj < kNi; nj += 2) {
        uint32_t r[4];
        ldmatrix_x4(r[0], r[1], r[2], r[3], base + kStageA + offset(b_row + nj * 8, 2 * kK + ((lane >> 3) & 1)));
        bf[nj][0] = r[0];
        bf[nj][1] = r[1];
        bf[nj + 1][0] = r[2];
        bf[nj + 1][1] = r[3];
      }
#pragma unroll
      for (int mi = 0; mi < kMi; ++mi) {
        uint32_t af[4];
        ldmatrix_x4(af[0], af[1], af[2], af[3], base + offset(a_row + mi * 16, 2 * kK + (lane >> 4)));
#pragma unroll
        for (int ni = 0; ni < kNi; ++ni) mma_fp8_a<kK>(acc[mi][ni], af, bf[ni][0], bf[ni][1], sa[mi]);
      }
    };
    k_step.template operator()<0>();
    k_step.template operator()<1>();
    k_step.template operator()<2>();
    k_step.template operator()<3>();
    stage = stage + 1 == kStages ? 0 : stage + 1;
  }

  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int ni = 0; ni < kNi; ++ni) {
    const int col = n0 + wn * kWarpCols + ni * 8 + 2 * t;
    if (col >= n) continue;
    const bool second = col + 1 < n;
    const float s0 = p.w_scale[col];
    const float s1 = second ? p.w_scale[col + 1] : 0.f;
#pragma unroll
    for (int mi = 0; mi < kMi; ++mi) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = m0 + wm * kWarpRows + mi * 16 + g + 8 * h;
        if (row >= m) continue;
        __nv_bfloat16* dst_row = p.out + size_t(row) * p.out_ld + col;
        const float v0 = acc[mi][ni][2 * h] * s0;
        const float v1 = acc[mi][ni][2 * h + 1] * s1;
        if (second && (reinterpret_cast<uintptr_t>(dst_row) & 3u) == 0) {
          *reinterpret_cast<__nv_bfloat162*>(dst_row) = __floats2bfloat162_rn(v0, v1);
        } else {
          dst_row[0] = __float2bfloat16_rn(v0);
          if (second) dst_row[1] = __float2bfloat16_rn(v1);
        }
      }
    }
  }
}

}  // namespace gemm8
}  // namespace

void quantize_mxfp8(const __nv_bfloat16* x, int rows, int k, uint8_t* values, uint8_t* scales,
                    cudaStream_t stream) {
  if (rows <= 0) return;
  quantize_mxfp8_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(x, rows, k, values, scales);
}

void gemm_fp8_batch(const Fp8GemmProduct* products, int count, int m, int k, cudaStream_t stream) {
  if (count <= 0 || m <= 0) return;
  if (count > gemm8::kMaxProducts || k % 128 != 0)
    throw std::invalid_argument("prefill gemm_fp8_batch: invalid count or k");
  gemm8::Batch batch{};
  batch.count = count;
  batch.m = m;
  batch.k = k;
  const int tiles_m = (m + gemm8::kBm - 1) / gemm8::kBm;
  int total_tiles = 0;
  for (int i = 0; i < count; ++i) {
    const int tiles_n = (products[i].n + gemm8::kBn - 1) / gemm8::kBn;
    total_tiles += tiles_m * tiles_n;
    batch.p[i] = gemm8::Product{
        static_cast<const uint8_t*>(products[i].w), products[i].w_scale,
        products[i].x, products[i].x_scale,
        products[i].out, products[i].n, products[i].out_ld, total_tiles};
  }
  static const cudaError_t attr =
      cudaFuncSetAttribute(gemm8::gemm_fp8_128x128, cudaFuncAttributeMaxDynamicSharedMemorySize, gemm8::kSmemBytes);
  if (attr != cudaSuccess) throw std::runtime_error("prefill gemm_fp8_128x128: smem attr failed");
  gemm8::gemm_fp8_128x128<<<total_tiles, gemm8::kThreads, gemm8::kSmemBytes, stream>>>(batch);
}

}  // namespace ingine::kernels::prefill
