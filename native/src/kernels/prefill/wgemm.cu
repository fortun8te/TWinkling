// Grouped expert GEMMs of the prefill (model._moe_grouped) over NVFP4 / MXFP8 experts decoded on the fly.
//
// Design. A CTA computes up to 128 activation rows x 128 weight rows, K in steps of 64, mma.m16n8k16 BF16 with
// FP32 accumulation. The 8 warps split the weight rows (16 each) and every warp covers all of the tile's
// activation rows, so each weight element is decoded exactly once per CTA, straight from the staged checkpoint
// codes into the MMA B fragments in registers (no BF16 weight tile in shared memory):
//   NVFP4: bf16((e2m1 * e4m3 block scale) * factor)      (model.decode_experts)
//   MXFP8: bf16(e4m3 * 2^(e8m0 - 127))                    (model.decode_experts)
// Activation rows (gathered by pair for the experts) and the weight codes stream in through 16-byte cp.async:
// activations in 3 stages of 64 elements, the codes in 2 slots of 128-byte (64-byte for the NVFP4 down
// projection) row pieces, so DRAM reads whole lines and small chunks, which are bound by the weight bytes, keep
// enough loads in flight. One barrier per step. Code rows are padded in shared memory so the 8 rows of a fragment
// load hit distinct banks. m16 row blocks past the tile's row count skip their MMAs (zero-filled copies).
// The per-element products and the k order of the FP32 sums are those of the BF16 MMA over decoded rows.
// Epilogues: gate/up (w13 rows interleaved, 2i gate and 2i + 1 up, so each thread holds gate and up of the same
// i): act = bf16(softcap(bf16 g, 4) * sigmoid(bf16 g) * softcap(bf16 u, 25)); down: bf16 per pair placed at
// t * 8 + slot (moe_combine sums the slots in order: deterministic).
#include "kernels/prefill/common.cuh"
#include "kernels/prefill/prefill.h"

#include <cuda_fp4.h>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace ingine::kernels::prefill {
namespace {

enum class Mode { Up, Down };
constexpr int FMT_NVFP4 = 0, FMT_MXFP8 = 1;

constexpr int TN = 128, BK = 64;
constexpr int AS = 3;  // activation stages (BK elements each)
constexpr int NT = 256;
constexpr int KCH = BK / 8;  // 16-byte chunks per activation tile row

__constant__ float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

template <int FMT, int K, int TM>
struct Cfg {
  static constexpr int CODE_ROW = FMT == FMT_NVFP4 ? K / 2 : K;  // checkpoint bytes per weight row
  static constexpr int SCALE_ROW = FMT == FMT_NVFP4 ? K / 16 : K / 32;
  // weight chunk: 64 code bytes per row (NVFP4 128 elements, FP8 64)
  static constexpr int WK = FMT == FMT_NVFP4 ? 128 : 64;
  static constexpr int R = WK / BK;  // activation steps per weight chunk
  static constexpr int KC = K / WK;  // weight chunks
  // weight slots: the chunk of step s + 1 must have landed (and be published by the barrier) at step s, when its
  // B fragments are decoded under step s's MMAs: R (WS - 1) >= AS. Small-row tiles prefetch further.
  static constexpr int WS = TM <= 32 ? (FMT == FMT_NVFP4 ? 6 : 7) : (FMT == FMT_NVFP4 ? 3 : 4);
  static_assert(R * (WS - 1) >= AS);
  static constexpr int WCODE = FMT == FMT_NVFP4 ? WK / 2 : WK;  // code bytes per row per chunk
  static constexpr int WSTRIDE = WCODE + 16;                    // padded: fragment loads on distinct banks
  static constexpr int W_BYTES = TN * WSTRIDE;
  static constexpr int A_BYTES = TM * BK * 2;
  static constexpr int SC_BYTES = TN * SCALE_ROW;               // block scales of the whole tile
  static constexpr int OFF_W = AS * A_BYTES, OFF_SC = OFF_W + WS * W_BYTES;
  static constexpr int OFF_LUT = (OFF_SC + SC_BYTES + 15) / 16 * 16;
  static constexpr int SMEM = OFF_LUT + 64;
  static_assert(K % WK == 0 && SMEM <= 101376);
};

struct GemmArgs {
  const uint8_t* codes = nullptr;   // NVFP4 / MXFP8 expert codes [128, N, ..]
  const uint8_t* scales = nullptr;  // block scales
  const float* factors = nullptr;   // NVFP4 per-expert factor
  const uint16_t* a = nullptr;      // activations (bf16)
  MoeWork w;                        // tiles and pair lists
  uint16_t* out = nullptr;
  int stage_first = 0, stage_n = 0;  // experts read from the staged copies (RTX 5080 port)
  const uint8_t* stage_codes = nullptr;
  const uint8_t* stage_scales = nullptr;
};

__device__ __forceinline__ void cp_async4(void* smem_dst, const void* gmem_src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_addr(smem_dst)), "l"(gmem_src));
}

__device__ __forceinline__ float softcap(float x, float cap) { return tanhf(x * (1.f / cap)) * cap; }

// One activation step of MMAs for the first MT 16-row blocks: per k16 group, the A fragments by ldmatrix, then the
// MMAs, then `after(kk)` (the next step's B decoding, which runs under the issued MMAs).
template <int MT, int TM, class After>
__device__ __forceinline__ void mma_step(float (&acc)[TM / 16][2][4], const uint16_t* A,
                                         const unsigned (&b)[BK / 16][2][2], int lane, After&& after) {
#pragma unroll
  for (int kk = 0; kk < BK / 16; ++kk) {
    unsigned a[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
      ldmatrix_x4(a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                  smem_addr(A + swz<KCH>(mt * 16 + (lane & 15), kk * 2 + (lane >> 4))));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      mma_bf16(acc[mt][0], a[mt], b[kk][0][0], b[kk][0][1]);
      mma_bf16(acc[mt][1], a[mt], b[kk][1][0], b[kk][1][1]);
    }
    after(kk);
  }
}

// Up: A = x rows of the pairs' tokens (K = H), W = w13 (N = 2I), out act [pairs, I].
// Down: A = act rows (K = I), W = w2 (N = H), out down [T * 8, H] at pair_dst.
template <int FMT, Mode MODE, int K, int TM>
__global__ void __launch_bounds__(NT, 1) wgemm_kernel(const GemmArgs g) {
  constexpr int KS = K / BK;
  constexpr int NE = MODE == Mode::Up ? 2 * kInter : kHidden;  // expert weight rows
  using C = Cfg<FMT, K, TM>;
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* a_smem = smem;
  unsigned char* w_smem = smem + C::OFF_W;
  uint8_t* sc_smem = smem + C::OFF_SC;
  float* lut = reinterpret_cast<float*>(smem + C::OFF_LUT);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  // ---- tile: n pair rows (from pb) of expert e x weight rows n0..n0+127
  constexpr int NTILES_N = NE / TN;
  const int tile_idx = blockIdx.x / NTILES_N;
  if (tile_idx >= *g.w.tiles_n) return;
  const int4 tile = g.w.tiles[tile_idx];
  const int e = tile.x, pb = tile.y, n = tile.z;
  const int n0 = (blockIdx.x % NTILES_N) * TN;
  const bool staged = e >= g.stage_first && e < g.stage_first + g.stage_n;
  const int ee = staged ? e - g.stage_first : e;
  const uint8_t* code_tile = (staged ? g.stage_codes : g.codes) + (int64_t(ee) * NE + n0) * C::CODE_ROW;
  const uint8_t* scale_tile = (staged ? g.stage_scales : g.scales) + (int64_t(ee) * NE + n0) * C::SCALE_ROW;
  if (tid < 16) lut[tid] = kE2M1[tid];

  // activation rows of this thread's cp.async chunks: rows tid / 8 + 32 i, chunk tid % 8
  const uint16_t* arow[TM / 32];
#pragma unroll
  for (int i = 0; i < TM / 32; ++i) {
    const int r = (tid >> 3) + 32 * i;
    const int rr = r < n ? r : 0;
    if constexpr (MODE == Mode::Up) arow[i] = g.a + size_t(g.w.pair_token[pb + rr]) * K;
    else arow[i] = g.a + size_t(pb + rr) * K;
  }

  auto load_a = [&](int ks) {
    uint16_t* A = reinterpret_cast<uint16_t*>(a_smem + (ks % AS) * C::A_BYTES);
#pragma unroll
    for (int i = 0; i < TM / 32; ++i) {
      const int r = (tid >> 3) + 32 * i, c = tid & 7;
      cp_async16(A + swz<KCH>(r, c), arow[i] + ks * BK + c * 8, r < n ? 16 : 0);
    }
  };
  auto load_w = [&](int chunk) {
    unsigned char* ws = w_smem + (chunk % C::WS) * C::W_BYTES;
    constexpr int CPR = C::WCODE / 16;  // 16-byte pieces per row
#pragma unroll
    for (int i = 0; i < TN * CPR / NT; ++i) {
      const int id = tid + i * NT, row = id / CPR, ch = id % CPR;
      cp_async16(ws + row * C::WSTRIDE + ch * 16, code_tile + int64_t(row) * C::CODE_ROW + chunk * C::WCODE + ch * 16);
    }
  };
  // group j: activation step j (plus whatever weight chunk was issued since the previous commit)
  auto issue = [&](int j) {
    if (j < KS) load_a(j);
    cp_commit();
  };

  // the tile's block scales, with the first activation group
  for (int id = tid; id < C::SC_BYTES / 4; id += NT)
    cp_async4(sc_smem + id * 4, scale_tile + (id * 4 / C::SCALE_ROW) * int64_t(C::SCALE_ROW) + (id * 4) % C::SCALE_ROW);
#pragma unroll
  for (int j = 0; j < C::WS - 1; ++j)
    if (j < C::KC) load_w(j);
#pragma unroll
  for (int j = 0; j < AS - 1; ++j) issue(j);

  const int c = lane & 3;
  const float factor = FMT == FMT_NVFP4 ? __ldg(g.factors + e) : 1.f;
  const int mact = min((n + 15) / 16, TM / 16);  // active m16 row blocks (same for every warp)
  float acc[TM / 16][2][4];
#pragma unroll
  for (int a = 0; a < TM / 16; ++a)
#pragma unroll
    for (int b = 0; b < 2; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;

  // B fragments of step ks (k16 group kk) decoded from the staged codes of its weight chunk
  auto scales_of = [&](int ks, float (&sc)[2][4]) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int row = warp * 16 + nt * 8 + (lane >> 2);
      if constexpr (FMT == FMT_NVFP4) {
        const uint32_t s4 = *reinterpret_cast<const uint32_t*>(sc_smem + row * C::SCALE_ROW + ks * 4);
        const float2 lo = e4m3x2_to_float2(uint16_t(s4 & 0xffff)), hi = e4m3x2_to_float2(uint16_t(s4 >> 16));
        sc[nt][0] = lo.x, sc[nt][1] = lo.y, sc[nt][2] = hi.x, sc[nt][3] = hi.y;
      } else {
        const uint8_t* ms = sc_smem + row * C::SCALE_ROW + ks * 2;
        sc[nt][0] = sc[nt][1] = exp2f(float(ms[0]) - 127.f);
        sc[nt][2] = sc[nt][3] = exp2f(float(ms[1]) - 127.f);
      }
    }
  };
  auto decode = [&](int ks, int kk, const float (&sc)[2][4], unsigned (&b)[BK / 16][2][2]) {
    const unsigned char* ws = w_smem + ((ks / C::R) % C::WS) * C::W_BYTES;
    const int kc = (ks % C::R) * BK + kk * 16;  // first element of the k16 group inside the chunk
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int row = warp * 16 + nt * 8 + (lane >> 2);
      const float s = sc[nt][kk];
      if constexpr (FMT == FMT_NVFP4) {
        // elements 2c, 2c+1 (byte c) and 8 + 2c, +1 (byte 4 + c) of the k16 group
        const uint2 wd = *reinterpret_cast<const uint2*>(ws + row * C::WSTRIDE + kc / 2);
        const uint32_t b0 = (wd.x >> (8 * c)) & 0xffu, b1 = (wd.y >> (8 * c)) & 0xffu;
        b[kk][nt][0] = pack_bf16x2((lut[b0 & 0xf] * s) * factor, (lut[b0 >> 4] * s) * factor);
        b[kk][nt][1] = pack_bf16x2((lut[b1 & 0xf] * s) * factor, (lut[b1 >> 4] * s) * factor);
      } else {
        const uint4 q = *reinterpret_cast<const uint4*>(ws + row * C::WSTRIDE + kc);
        const uint32_t w0 = (c >> 1) ? q.y : q.x, w1 = (c >> 1) ? q.w : q.z;
        const float2 f0 = e4m3x2_to_float2(uint16_t(w0 >> (16 * (c & 1))));
        const float2 f1 = e4m3x2_to_float2(uint16_t(w1 >> (16 * (c & 1))));
        b[kk][nt][0] = pack_bf16x2(f0.x * s, f0.y * s);
        b[kk][nt][1] = pack_bf16x2(f1.x * s, f1.y * s);
      }
      asm volatile("" : "+r"(b[kk][nt][0]), "+r"(b[kk][nt][1]));
    }
  };

  // Software pipeline: step ks's MMAs run on fragments decoded during step ks - 1; the decoding of step ks + 1
  // is interleaved with them (its chunk has landed: R (WS - 1) >= AS, and the barrier published it).
  unsigned bcur[BK / 16][2][2], bnext[BK / 16][2][2];
#pragma unroll 1
  for (int ks = 0; ks < KS; ++ks) {
    cp_wait<AS - 2>();
    __syncthreads();
    const int chunk = ks / C::R;
    // first step of a chunk: the chunk WS - 1 ahead into the slot the previous chunk used (read before the barrier)
    if (ks % C::R == 0 && chunk + C::WS - 1 < C::KC) load_w(chunk + C::WS - 1);
    issue(ks + AS - 1);
    if (ks == 0) {
      float sc0[2][4];
      scales_of(0, sc0);
#pragma unroll
      for (int kk = 0; kk < BK / 16; ++kk) decode(0, kk, sc0, bcur);
    }
    const uint16_t* A = reinterpret_cast<const uint16_t*>(a_smem + (ks % AS) * C::A_BYTES);
    const bool more = ks + 1 < KS;
    float scn[2][4];
    if (more) scales_of(ks + 1, scn);
    auto after = [&](int kk) {
      if (more) decode(ks + 1, kk, scn, bnext);
    };
    // the row-block count is a template constant inside mma_step: a runtime `mt < mact` test would be if-converted
    // into predicated MMAs, which still occupy the tensor pipe (8x the work for a 16-row tile)
    switch (mact) {
      case 1: if constexpr (1 <= TM / 16) mma_step<1, TM>(acc, A, bcur, lane, after); break;
      case 2: if constexpr (2 <= TM / 16) mma_step<2, TM>(acc, A, bcur, lane, after); break;
      case 3: if constexpr (3 <= TM / 16) mma_step<3, TM>(acc, A, bcur, lane, after); break;
      case 4: if constexpr (4 <= TM / 16) mma_step<4, TM>(acc, A, bcur, lane, after); break;
      case 5: if constexpr (5 <= TM / 16) mma_step<5, TM>(acc, A, bcur, lane, after); break;
      case 6: if constexpr (6 <= TM / 16) mma_step<6, TM>(acc, A, bcur, lane, after); break;
      case 7: if constexpr (7 <= TM / 16) mma_step<7, TM>(acc, A, bcur, lane, after); break;
      default: mma_step<TM / 16, TM>(acc, A, bcur, lane, after); break;
    }
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) bcur[kk][nt][0] = bnext[kk][nt][0], bcur[kk][nt][1] = bnext[kk][nt][1];
  }
  cp_wait<0>();

#pragma unroll
  for (int mt = 0; mt < TM / 16; ++mt) {
    if (mt >= mact) continue;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = mt * 16 + (lane >> 2) + half * 8;
      if (row >= n) continue;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int cl = warp * 16 + nt * 8 + c * 2;  // column inside the tile
        const float c0 = acc[mt][nt][2 * half], c1 = acc[mt][nt][2 * half + 1];
        if constexpr (MODE == Mode::Up) {
          const float gt = round_bf16(c0), u = round_bf16(c1);
          const float sig = 1.f / (1.f + expf(-gt));
          const float v = __fmul_rn(__fmul_rn(softcap(gt, 4.f), sig), softcap(u, 25.f));
          g.out[size_t(pb + row) * kInter + (n0 + cl) / 2] = f2bf_bits(v);
        } else {
          const int dst = g.w.pair_dst[pb + row];
          *reinterpret_cast<uint32_t*>(g.out + size_t(dst) * kHidden + n0 + cl) = pack_bf16x2(c0, c1);
        }
      }
    }
  }
}

template <int FMT, Mode MODE, int K, int TM>
void launch(const GemmArgs& g, int grid, cudaStream_t stream) {
  constexpr int SMEM = Cfg<FMT, K, TM>::SMEM;
  static const cudaError_t attr =
      cudaFuncSetAttribute(wgemm_kernel<FMT, MODE, K, TM>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  if (attr != cudaSuccess) throw std::runtime_error("prefill wgemm: shared memory attribute failed");
  wgemm_kernel<FMT, MODE, K, TM><<<grid, NT, SMEM, stream>>>(g);
}

template <int FMT, Mode MODE, int K>
void launch_rows(const GemmArgs& g, int tm, int grid_per_row_tile, int row_tiles, cudaStream_t stream) {
  if (tm == 32) launch<FMT, MODE, K, 32>(g, grid_per_row_tile * row_tiles, stream);
  else launch<FMT, MODE, K, 128>(g, grid_per_row_tile * row_tiles, stream);
}

GemmArgs expert_args(const void* codes, const void* scales, const float* factors, const __nv_bfloat16* a,
                     const MoeWork& w, __nv_bfloat16* out) {
  GemmArgs g;
  g.codes = static_cast<const uint8_t*>(codes);
  g.scales = static_cast<const uint8_t*>(scales);
  g.factors = factors;
  g.a = reinterpret_cast<const uint16_t*>(a);
  g.w = w;
  g.out = reinterpret_cast<uint16_t*>(out);
  return g;
}

// -----------------------------------------------------------------------------
// Blackwell SM120a native NVFP4 W4A4 grouped MoE (Four-Over-Six activation quant
// + mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64).
// -----------------------------------------------------------------------------
namespace nvfp4_tc {

constexpr int kTileN = 128;
constexpr int kTileT = 128;
constexpr int kStages = 2;
constexpr int kRowBytes = 64;     // 128 E2M1 elements per stage (2 MMA k=64 steps)
constexpr int kStageSteps = 2;
constexpr int kRows = kTileN + kTileT;
constexpr int kStageBytes = kRows * kRowBytes;
constexpr int kSlotRow = 16;      // 16 E4M3 scale bytes = 4 k=64 steps per slot
constexpr int kSlots = 2;
constexpr int kOutStride = kTileN + 8;
constexpr int kSmemBytes = kStages * kStageBytes + kSlots * kRows * kSlotRow;
static_assert(kTileT * kOutStride * 2 <= kSmemBytes);
static_assert(kSmemBytes <= 99 * 1024);

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

__device__ __forceinline__ float e4m3_round(float x) {
  const __nv_fp8_e4m3 q(x);
  return float(q);
}

__device__ __forceinline__ uint32_t e2m1x2(float lo, float hi, float& lo_v, float& hi_v) {
  uint32_t byte, halves;
  asm("{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %3, %2; cvt.u32.u8 %0, b; "
      "cvt.rn.f16x2.e2m1x2 %1, b; }"
      : "=r"(byte), "=r"(halves)
      : "f"(lo), "f"(hi));
  const __half2 h = *reinterpret_cast<const __half2*>(&halves);
  lo_v = __low2float(h);
  hi_v = __high2float(h);
  return byte;
}

__device__ __forceinline__ float fp4_candidate(const float (&x)[16], float scale, float inv, float outer,
                                               uint2& packed) {
  float err[16];
  uint32_t bytes[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float lo, hi;
    bytes[i] = e2m1x2(__fmul_rn(x[2 * i], inv), __fmul_rn(x[2 * i + 1], inv), lo, hi);
    err[2 * i] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(lo, scale), outer), x[2 * i]));
    err[2 * i + 1] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(hi, scale), outer), x[2 * i + 1]));
  }
  packed.x = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
  packed.y = bytes[4] | (bytes[5] << 8) | (bytes[6] << 16) | (bytes[7] << 24);
#pragma unroll
  for (int w = 8; w >= 1; w >>= 1) {
#pragma unroll
    for (int i = 0; i < w; ++i) err[i] = __fadd_rn(err[2 * i], err[2 * i + 1]);
  }
  return err[0];
}

__global__ void __launch_bounds__(256)
    quantize_nvfp4_kernel(const __nv_bfloat16* __restrict__ x, int rows, int k,
                          uint8_t* __restrict__ values, uint8_t* __restrict__ scales,
                          float* __restrict__ outer) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const __nv_bfloat16* src = x + size_t(row) * k;
  uint8_t* dst_v = values + size_t(row) * (k / 2);
  uint8_t* dst_s = scales + size_t(row) * (k / 16);

  float amax = 0.f;
  for (int b = lane; b < k / 16; b += 32) {
    float v[16];
    load8_bf16(src + b * 16, v);
    load8_bf16(src + b * 16 + 8, v + 8);
#pragma unroll
    for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fabsf(v[j]));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));

  if (amax == 0.f) {
    if (lane == 0) outer[row] = 0.f;
    for (int b = lane; b < k / 16; b += 32) {
      dst_s[b] = 0;
      *reinterpret_cast<uint2*>(dst_v + b * 8) = make_uint2(0, 0);
    }
    return;
  }

  constexpr float kFp32Max = 3.4028234663852886e38f;
  const float encoder = fminf(__fdiv_rn(1536.f, amax), kFp32Max);
  const float decoder = __fdiv_rn(1.f, encoder);
  const float outer_scale = __fmul_rn(amax, 1.f / 1536.f);
  if (lane == 0) outer[row] = outer_scale;
  const float encoder6 = __fmul_rn(encoder, 1.f / 6.f);
  const float encoder4 = __fmul_rn(encoder, 0.25f);

  for (int b = lane; b < k / 16; b += 32) {
    float v[16];
    load8_bf16(src + b * 16, v);
    load8_bf16(src + b * 16 + 8, v + 8);
    float bmax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) bmax = fmaxf(bmax, fabsf(v[i]));
    if (bmax == 0.f) {
      dst_s[b] = 0;
      *reinterpret_cast<uint2*>(dst_v + b * 8) = make_uint2(0, 0);
      continue;
    }
    const float scale6 = e4m3_round(fminf(__fmul_rn(bmax, encoder6), 448.f));
    const float scale4 = e4m3_round(fminf(__fmul_rn(bmax, encoder4), 448.f));
    const float inv6 = scale6 > 0.f ? fminf(__fdiv_rn(1.f, __fmul_rn(scale6, decoder)), kFp32Max) : 0.f;
    const float inv4 = scale4 > 0.f ? fminf(__fdiv_rn(1.f, __fmul_rn(scale4, decoder)), kFp32Max) : 0.f;
    uint2 p6, p4;
    const float e6 = fp4_candidate(v, scale6, inv6, outer_scale, p6);
    const float e4 = fp4_candidate(v, scale4, inv4, outer_scale, p4);
    const bool choose4 = e4 < e6;
    const __nv_fp8_e4m3 sc(choose4 ? scale4 : scale6);
    dst_s[b] = sc.__x;
    *reinterpret_cast<uint2*>(dst_v + b * 8) = choose4 ? p4 : p6;
  }
}

__device__ __forceinline__ int q_offset(int row, int u) {
  return row * kRowBytes + ((u ^ ((row >> 1) & 3)) << 4);
}

__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(smem_addr(dst)), "l"(src));
}

template <int kThreadB>
__device__ __forceinline__ void mma_fp4(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                        uint32_t scale_a, uint32_t scale_b) {
  asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1."
      "e2m1.f32.ue4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, "
      "{%13}, {%14,%15};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(scale_a),
        "h"(uint16_t(0)), "h"(uint16_t(0)), "r"(scale_b), "h"(uint16_t(0)),
        "h"(uint16_t(kThreadB)));
}

struct Nvfp4GemmArgs {
  const uint8_t* codes = nullptr;
  const uint8_t* scales = nullptr;
  const float* factors = nullptr;
  const uint8_t* xq = nullptr;
  const uint8_t* xscale = nullptr;
  const float* xouter = nullptr;
  MoeWork w;
  uint16_t* out = nullptr;
  int stage_first = 0, stage_n = 0;
  const uint8_t* stage_codes = nullptr;
  const uint8_t* stage_scales = nullptr;
};

template <Mode MODE, int K>
__global__ void __launch_bounds__(256, 2) wgemm_nvfp4_kernel(const Nvfp4GemmArgs g) {
  constexpr int NE = MODE == Mode::Up ? 2 * kInter : kHidden;
  constexpr int NTILES_N = NE / kTileN;
  constexpr int kRowB = K / 2;
  constexpr int kScaleRowB = K / 16;
  constexpr int kStagesTotal = kRowB / kRowBytes;
  static_assert(kRowB % kRowBytes == 0);

  const int tile_idx = blockIdx.x / NTILES_N;
  if (tile_idx >= *g.w.tiles_n) return;
  const int4 tile = g.w.tiles[tile_idx];
  const int e = tile.x, pb = tile.y, count = tile.z;
  const int n0 = (blockIdx.x % NTILES_N) * kTileN;

  const bool staged = e >= g.stage_first && e < g.stage_first + g.stage_n;
  const int ee = staged ? e - g.stage_first : e;
  const uint8_t* weight_base = (staged ? g.stage_codes : g.codes) + (int64_t(ee) * NE + n0) * kRowB;
  const uint8_t* scale_base = (staged ? g.stage_scales : g.scales) + (int64_t(ee) * NE + n0) * kScaleRowB;

  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* const slots = smem + kStages * kStageBytes;
  __shared__ float pair_outer[kTileT];

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int gr = lane >> 2, t = lane & 3;
  const int col0 = (warp & 3) * 32;
  const int pair0 = (warp >> 2) * 64;

  auto input_of = [&](int pair) {
    const int p = pair < count ? pair : 0;
    if constexpr (MODE == Mode::Up) return g.w.pair_token[pb + p];
    else return pb + p;
  };

  const int copy_unit = tid & 3;
  const uint8_t* w_src[2];
  const uint8_t* x_src[2];
  uint32_t w_dst[2], x_dst[2];
  int x_bytes[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int row = (tid >> 2) + 64 * i;
    w_src[i] = weight_base + size_t(row) * kRowB + copy_unit * 16;
    w_dst[i] = q_offset(row, copy_unit);
    x_src[i] = g.xq + size_t(input_of(row)) * kRowB + copy_unit * 16;
    x_dst[i] = q_offset(kTileN + row, copy_unit);
    x_bytes[i] = row < count ? 16 : 0;
  }
  const uint8_t* sc_src = tid < kTileN ? scale_base + size_t(tid) * kScaleRowB
                                       : g.xscale + size_t(input_of(tid - kTileN)) * kScaleRowB;
  if (tid < kTileT) pair_outer[tid] = g.xouter[input_of(tid)];

  auto load_stage = [&](int index, int stage) {
    uint8_t* base = smem + stage * kStageBytes;
    const int step = index * kStageSteps;
    if ((step & 1) == 0) {
      const int piece = step >> 1;
      uint8_t* dst = slots + ((step >> 2) % kSlots) * (kRows * kSlotRow) + tid * kSlotRow + ((step & 3) >> 1) * 8;
      cp_async8(dst, sc_src + piece * 8);
    }
    const int offset = index * kRowBytes;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      cp_async16(base + w_dst[i], w_src[i] + offset, 16);
      cp_async16(base + x_dst[i], x_src[i] + offset, x_bytes[i]);
    }
  };

  float acc[2][8][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < kStagesTotal) load_stage(s, s);
    cp_commit();
  }

  const uint32_t smem0 = smem_addr(smem);
  const int a_row = col0 + (lane & 7) + 8 * ((lane >> 3) & 1);
  const int b_pair = kTileN + pair0 + (lane & 7) + 8 * (lane >> 4);
  const int scale_row = col0 + gr + 8 * (t & 1);
  const int scale_pair = kTileN + pair0 + lane;

  auto compute = [&]<int kFragments, int kSub>(int step, int stage, int sub) {
    const uint32_t base = smem0 + stage * kStageBytes;
    const uint8_t* slot = slots + ((step >> 2) % kSlots) * (kRows * kSlotRow);
    uint32_t a[2][4];
    ldmatrix_x4(a[0][0], a[0][1], a[0][2], a[0][3], base + q_offset(a_row, 2 * sub + (lane >> 4)));
    ldmatrix_x4(a[1][0], a[1][1], a[1][2], a[1][3], base + q_offset(a_row + 16, 2 * sub + (lane >> 4)));
    const int word = step & 3;
    const uint32_t sa0 = *reinterpret_cast<const uint32_t*>(slot + scale_row * kSlotRow + 4 * word);
    const uint32_t sa1 = *reinterpret_cast<const uint32_t*>(slot + (scale_row + 16) * kSlotRow + 4 * word);
    uint32_t sb[8];
    {
      uint32_t r[4];
      ldmatrix_x4(r[0], r[1], r[2], r[3], smem_addr(slot) + scale_pair * kSlotRow);
      sb[0] = r[0]; sb[1] = r[1]; sb[2] = r[2]; sb[3] = r[3];
      if constexpr (kFragments > 4) {
        ldmatrix_x4(r[0], r[1], r[2], r[3], smem_addr(slot) + (scale_pair + 32) * kSlotRow);
        sb[4] = r[0]; sb[5] = r[1]; sb[6] = r[2]; sb[7] = r[3];
      }
    }
#pragma unroll
    for (int q = 0; q < (kFragments + 1) / 2; ++q) {
      uint32_t b[4];
      ldmatrix_x4(b[0], b[1], b[2], b[3], base + q_offset(b_pair + 16 * q, 2 * sub + ((lane >> 3) & 1)));
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int j = 2 * q + h;
        if (j < kFragments) {
          mma_fp4<kSub>(acc[0][j], a[0], b[2 * h], b[2 * h + 1], sa0, sb[j]);
          mma_fp4<kSub>(acc[1][j], a[1], b[2 * h], b[2 * h + 1], sa1, sb[j]);
        }
      }
    }
  };

  const int fragments = min(max((count - pair0 + 7) / 8, 0), 8);
  auto main_loop = [&]<int kFragments>() {
    for (int index0 = 0; index0 < kStagesTotal; index0 += kStages) {
#pragma unroll
      for (int stage = 0; stage < kStages; ++stage) {
        const int index = index0 + stage;
        if (index < kStagesTotal) {
          cp_wait<kStages - 2>();
          __syncthreads();
          const int next = index + kStages - 1;
          if (next < kStagesTotal) load_stage(next, (stage + kStages - 1) % kStages);
          cp_commit();
          if constexpr (kFragments > 0) {
            const int step = index * kStageSteps;
            if ((stage & 1) == 0) {
              compute.template operator()<kFragments, 0>(step, stage, 0);
              compute.template operator()<kFragments, 1>(step + 1, stage, 1);
            } else {
              compute.template operator()<kFragments, 2>(step, stage, 0);
              compute.template operator()<kFragments, 3>(step + 1, stage, 1);
            }
          }
        }
      }
    }
  };
  switch (fragments) {
    case 0: main_loop.template operator()<0>(); break;
    case 1: main_loop.template operator()<1>(); break;
    case 2: main_loop.template operator()<2>(); break;
    case 3: main_loop.template operator()<3>(); break;
    case 4: main_loop.template operator()<4>(); break;
    case 5: main_loop.template operator()<5>(); break;
    case 6: main_loop.template operator()<6>(); break;
    case 7: main_loop.template operator()<7>(); break;
    default: main_loop.template operator()<8>(); break;
  }
  cp_wait<0>();
  __syncthreads();

  uint16_t* staged_out = reinterpret_cast<uint16_t*>(smem);
  const float w_scale = __ldg(g.factors + e);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int pair = pair0 + j * 8 + 2 * t;
    const float s0 = __fmul_rn(pair_outer[pair], w_scale);
    const float s1 = __fmul_rn(pair_outer[pair + 1], w_scale);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int col = col0 + 16 * i + gr;
      staged_out[pair * kOutStride + col] = f2bf_bits(acc[i][j][0] * s0);
      staged_out[(pair + 1) * kOutStride + col] = f2bf_bits(acc[i][j][1] * s1);
      staged_out[pair * kOutStride + col + 8] = f2bf_bits(acc[i][j][2] * s0);
      staged_out[(pair + 1) * kOutStride + col + 8] = f2bf_bits(acc[i][j][3] * s1);
    }
  }
  __syncthreads();

  for (int unit = tid; unit < count * (kTileN / 8); unit += 256) {
    const int pair = unit / (kTileN / 8);
    const int part = unit % (kTileN / 8);
    const uint4 v = *reinterpret_cast<const uint4*>(staged_out + pair * kOutStride + part * 8);
    if constexpr (MODE == Mode::Up) {
      const uint32_t words[4] = {v.x, v.y, v.z, v.w};
      uint16_t act[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float gt = bf_lo(words[i]), u = bf_hi(words[i]);
        const float sig = 1.f / (1.f + expf(-gt));
        act[i] = f2bf_bits(__fmul_rn(__fmul_rn(softcap(gt, 4.f), sig), softcap(u, 25.f)));
      }
      *reinterpret_cast<uint2*>(g.out + size_t(pb + pair) * kInter + n0 / 2 + part * 4) =
          *reinterpret_cast<const uint2*>(act);
    } else {
      const int dst = g.w.pair_dst[pb + pair];
      *reinterpret_cast<uint4*>(g.out + size_t(dst) * kHidden + n0 + part * 8) = v;
    }
  }
}

template <Mode MODE, int K>
void launch_nvfp4(const Nvfp4GemmArgs& g, int grid, cudaStream_t stream) {
  static const cudaError_t attr =
      cudaFuncSetAttribute(wgemm_nvfp4_kernel<MODE, K>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  if (attr != cudaSuccess) throw std::runtime_error("prefill wgemm_nvfp4: shared memory attribute failed");
  wgemm_nvfp4_kernel<MODE, K><<<grid, 256, kSmemBytes, stream>>>(g);
}

bool w4a4_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("INGINE_PREFILL_W4A4");
    return !e || std::atoi(e) != 0;
  }();
  return on;
}

}  // namespace nvfp4_tc

}  // namespace

void moe_gate_up(const MoeExperts& e, const __nv_bfloat16* x, int T, const MoeWork& w, __nv_bfloat16* act,
                 cudaStream_t stream, const MoeQuantScratch* qscratch) {
  if (e.fmt == 0 && qscratch && qscratch->xq && T >= 64 && nvfp4_tc::w4a4_enabled()) {
    nvfp4_tc::quantize_nvfp4_kernel<<<(T + 7) / 8, 256, 0, stream>>>(
        x, T, kHidden, qscratch->xq, qscratch->xscale, qscratch->xouter);
    nvfp4_tc::Nvfp4GemmArgs g;
    g.codes = static_cast<const uint8_t*>(e.w13);
    g.scales = static_cast<const uint8_t*>(e.w13_scale);
    g.factors = e.w13_factor;
    g.xq = qscratch->xq;
    g.xscale = qscratch->xscale;
    g.xouter = qscratch->xouter;
    g.w = w;
    g.out = reinterpret_cast<uint16_t*>(act);
    g.stage_first = e.stage_first;
    g.stage_n = e.stage_n;
    g.stage_codes = static_cast<const uint8_t*>(e.stage_w13);
    g.stage_scales = static_cast<const uint8_t*>(e.stage_w13_scale);
    nvfp4_tc::launch_nvfp4<Mode::Up, kHidden>(g, (2 * kInter / TN) * moe_max_tiles(T), stream);
    return;
  }
  GemmArgs g = expert_args(e.w13, e.w13_scale, e.w13_factor, x, w, act);
  g.stage_first = e.stage_first, g.stage_n = e.stage_n;
  g.stage_codes = static_cast<const uint8_t*>(e.stage_w13), g.stage_scales = static_cast<const uint8_t*>(e.stage_w13_scale);
  if (e.fmt == 0) launch_rows<FMT_NVFP4, Mode::Up, kHidden>(g, moe_tile_rows(T), 2 * kInter / TN, moe_max_tiles(T), stream);
  else launch_rows<FMT_MXFP8, Mode::Up, kHidden>(g, moe_tile_rows(T), 2 * kInter / TN, moe_max_tiles(T), stream);
}

void moe_down(const MoeExperts& e, const __nv_bfloat16* act, int T, const MoeWork& w, __nv_bfloat16* down,
              cudaStream_t stream, const MoeQuantScratch* qscratch) {
  if (e.fmt == 0 && qscratch && qscratch->xq && T >= 64 && nvfp4_tc::w4a4_enabled()) {
    const int pairs = T * kTopK;
    nvfp4_tc::quantize_nvfp4_kernel<<<(pairs + 7) / 8, 256, 0, stream>>>(
        act, pairs, kInter, qscratch->xq, qscratch->xscale, qscratch->xouter);
    nvfp4_tc::Nvfp4GemmArgs g;
    g.codes = static_cast<const uint8_t*>(e.w2);
    g.scales = static_cast<const uint8_t*>(e.w2_scale);
    g.factors = e.w2_factor;
    g.xq = qscratch->xq;
    g.xscale = qscratch->xscale;
    g.xouter = qscratch->xouter;
    g.w = w;
    g.out = reinterpret_cast<uint16_t*>(down);
    g.stage_first = e.stage_first;
    g.stage_n = e.stage_n;
    g.stage_codes = static_cast<const uint8_t*>(e.stage_w2);
    g.stage_scales = static_cast<const uint8_t*>(e.stage_w2_scale);
    nvfp4_tc::launch_nvfp4<Mode::Down, kInter>(g, (kHidden / TN) * moe_max_tiles(T), stream);
    return;
  }
  GemmArgs g = expert_args(e.w2, e.w2_scale, e.w2_factor, act, w, down);
  g.stage_first = e.stage_first, g.stage_n = e.stage_n;
  g.stage_codes = static_cast<const uint8_t*>(e.stage_w2), g.stage_scales = static_cast<const uint8_t*>(e.stage_w2_scale);
  if (e.fmt == 0) launch_rows<FMT_NVFP4, Mode::Down, kInter>(g, moe_tile_rows(T), kHidden / TN, moe_max_tiles(T), stream);
  else launch_rows<FMT_MXFP8, Mode::Down, kInter>(g, moe_tile_rows(T), kHidden / TN, moe_max_tiles(T), stream);
}

}  // namespace ingine::kernels::prefill
