// Routed experts stored as GGUF quant blocks (ISTA-DASLab GSQ-RCO packs: IQ1_S .. IQ3_S gate/up, Q2_0 or
// IQ4_NL down) for decode widths 1..8. Activations are quantized to Q8_1 per 32 values and dotted with the
// weight blocks by llama.cpp's integer dot products (kernels/moe/gguf_dot.cuh), as llama.cpp evaluates
// these packs. The SwiGLU-softcap, routing-weight sum and static conv epilogues match moe.cu.
//
// Layout (GGUF ffn_gate_up_exps / ffn_down_exps): expert e's row r starts at (e * rows + r) * row_bytes;
// gate/up rows 0..I-1 are the gate, I..2I-1 the up projection.
#include "kernels/moe/moe.h"
#include "kernels/moe/gguf_dot.cuh"
#include "kernels/moe/gguf_unit.cuh"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ingine::kernels::moe {
namespace {

constexpr int H = 2304, I = 1152, TOPK = 8, MAXM = 8;
#ifndef GGUF_UP_PAIRS
#define GGUF_UP_PAIRS 2
#endif
#ifndef GGUF_BATCH
#define GGUF_BATCH 4
#endif
#ifndef GGUF_MINB
#define GGUF_MINB 2
#endif
constexpr int UP_PAIRS = GGUF_UP_PAIRS;     // gate/up row pairs per warp
constexpr int UP_ROWS = 8 * UP_PAIRS;  // gate rows (and the matching up rows) per block
#ifndef GGUF_DOWN_RPW
#define GGUF_DOWN_RPW 2
#endif
constexpr int DOWN_RPW = GGUF_DOWN_RPW;     // output rows per warp
constexpr int DOWN_TILE = 8 * DOWN_RPW;  // output rows per block
constexpr int THREADS = 256;    // 8 warps

typedef float (*VecDot)(const void*, const block_q8_1*, const int&, const int&);

template <int T>
struct Traits;
#define TW_TRAITS(T_, QK_, QI_, VDR_, BLOCK_, FN_)                                    \
    template <>                                                                       \
    struct Traits<T_> {                                                               \
        static constexpr int qk = QK_, qi = QI_, vdr = VDR_, bs = sizeof(BLOCK_);     \
        static __device__ __forceinline__ float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { \
            return FN_(v, y, kbx, iqs);                                               \
        }                                                                             \
    };
TW_TRAITS(16, QK_K, QI2_XXS, VDR_IQ2_XXS_Q8_1_MMVQ, block_iq2_xxs, vec_dot_iq2_xxs_q8_1)
TW_TRAITS(17, QK_K, QI2_XS, VDR_IQ2_XS_Q8_1_MMVQ, block_iq2_xs, vec_dot_iq2_xs_q8_1)
TW_TRAITS(18, QK_K, QI3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ, block_iq3_xxs, vec_dot_iq3_xxs_q8_1)
TW_TRAITS(19, QK_K, QI1_S, VDR_IQ1_S_Q8_1_MMVQ, block_iq1_s, vec_dot_iq1_s_q8_1)
TW_TRAITS(20, QK4_NL, QI4_NL, VDR_IQ4_NL_Q8_1_MMVQ, block_iq4_nl, vec_dot_iq4_nl_q8_1)
TW_TRAITS(21, QK_K, QI3_S, VDR_IQ3_S_Q8_1_MMVQ, block_iq3_s, vec_dot_iq3_s_q8_1)
TW_TRAITS(22, QK_K, QI2_S, VDR_IQ2_S_Q8_1_MMVQ, block_iq2_s, vec_dot_iq2_s_q8_1)
TW_TRAITS(42, QK2_0, QI2_0, VDR_Q2_0_Q8_1_MMVQ, block_q2_0, vec_dot_q2_0_q8_1)
#undef TW_TRAITS

__device__ __forceinline__ float bf16_round(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Slots 0..63 (token * 8 + k) routed to expert e, from a warp-wide view of idx (as moe.cu's slot_mask).
__device__ __forceinline__ uint64_t slot_mask(int e, int lo, int hi) {
    const uint32_t m0 = __ballot_sync(0xffffffffu, lo == e);
    const uint32_t m1 = __ballot_sync(0xffffffffu, hi == e);
    return uint64_t(m0) | (uint64_t(m1) << 32);
}

// One warp: dots of weight row `wrow` (K values) with the Q8_1 rows of the tokens in `tokens` (bit n: token n),
// summed over the row's quant units; lane 0 returns the full sums in s[].
template <int T, int K>
__device__ __forceinline__ void row_dots(const uint8_t* wrow, const block_q8_1* xq, int xstride, uint32_t tokens,
                                         int lane, float (&s)[MAXM]) {
    using Tr = Traits<T>;
    constexpr int nb = K / Tr::qk, units = Tr::qi / Tr::vdr, total = nb * units;
#pragma unroll
    for (int n = 0; n < MAXM; ++n) s[n] = 0.f;
    for (int u = lane; u < total; u += 32) {
        const int kbx = u / units, kqs = Tr::vdr * (u % units);
        const int kby = kbx * (Tr::qk / QK8_1);
#pragma unroll
        for (int n = 0; n < MAXM; ++n)
            if ((tokens >> n) & 1u) s[n] += Tr::dot(wrow, xq + size_t(n) * xstride + kby, kbx, kqs);
    }
#pragma unroll
    for (int n = 0; n < MAXM; ++n) s[n] = warp_sum(s[n]);
}

// bf16 [rows, K] -> Q8_1 [rows, K / 32] (llama.cpp quantize_q8_1: d = amax / 127, q = round(x / d), ds = (d, d * sum q))
__global__ void quantize_q8_1_kernel(const __nv_bfloat16* __restrict__ x, int K, block_q8_1* __restrict__ y) {
    const int row = blockIdx.y, i0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i0 >= K) return;
    const float xi = __bfloat162float(x[size_t(row) * K + i0]);
    float amax = fabsf(xi);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : int8_t(roundf(xi / d));
    int sum = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    block_q8_1& b = y[size_t(row) * (K / QK8_1) + i0 / QK8_1];
    b.qs[i0 % QK8_1] = q;
    if (i0 % QK8_1 == 0) b.ds = make_half2(d, d * sum);
}

// ------------------------------------------------------------------------------------- gate/up
// grid (I / UP_ROWS, M * TOPK): block (tile, slot) computes gate rows tile*8..+7 and the matching up rows of
// the expert of `slot` for every token routed to it (an earlier slot with the same expert owns it). Warp w
// takes gate row tile*8 + w and up row I + tile*8 + w; each weight unit is decoded once for all tokens.
template <int T, int G>
__global__ void __launch_bounds__(THREADS, GGUF_MINB) moe_up_gguf_kernel(const block_q8_1* __restrict__ xq, const int* __restrict__ idx,
                                                             const uint8_t* __restrict__ w, size_t row_bytes, int M,
                                                             __nv_bfloat16* __restrict__ out) {
    constexpr int XS = H / QK8_1;
    constexpr int ROWS = UP_ROWS * G;  // gate rows per block
    __shared__ float acc[2 * ROWS][MAXM];
    __shared__ block_q8_1 xs[MAXM][XS];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tile = blockIdx.x, slot = blockIdx.y, nslots = M * TOPK;
    const int lo = lane < nslots ? idx[lane] : -1;
    const int hi = lane + 32 < nslots ? idx[lane + 32] : -1;
    const int e = idx[slot];
    const uint64_t mask = slot_mask(e, lo, hi);
    if (__ffsll(static_cast<long long>(mask)) - 1 != slot) return;
    uint32_t tokens = 0;
#pragma unroll
    for (int n = 0; n < MAXM; ++n)
        if (n < M && ((mask >> (8 * n)) & 0xffu)) tokens |= 1u << n;
    {
        const int4* src = reinterpret_cast<const int4*>(xq);
        int4* dst = reinterpret_cast<int4*>(&xs[0][0]);
        constexpr int per_token = XS * sizeof(block_q8_1) / 16;  // 162
        for (int i = threadIdx.x; i < M * per_token; i += THREADS) dst[i] = src[i];
    }
    __syncthreads();

    using U = gguf::Unit<T>;
    constexpr int nb = H / U::qk, units = U::qi / U::vdr, total = nb * units;
#pragma unroll 1
    for (int grp = 0; grp < G; ++grp) {
    const int c0 = tile * ROWS + grp * UP_ROWS + warp * UP_PAIRS;
    float sg[UP_PAIRS][MAXM], su[UP_PAIRS][MAXM];
#pragma unroll
    for (int p = 0; p < UP_PAIRS; ++p)
#pragma unroll
        for (int n = 0; n < MAXM; ++n) sg[p][n] = 0.f, su[p][n] = 0.f;
    // lanes run over (pair, unit) jointly; B items per lane are loaded together so their dependent table
    // lookups overlap (the kernel is load-latency bound)
    constexpr int B = GGUF_BATCH;
    for (int base = lane; base < UP_PAIRS * total; base += 32 * B) {
        U ug[B], uu[B];
        int prs[B], kbys[B];
#pragma unroll
        for (int j = 0; j < B; ++j) {
            const int item = base + 32 * j;
            prs[j] = -1;
            if (item < UP_PAIRS * total) {
                const int pr = item / total, q = item % total;
                const int kbx = q / units, kqs = U::vdr * (q % units);
                kbys[j] = kbx * (U::qk / QK8_1);
                prs[j] = pr;
                ug[j].load(w + (size_t(e) * 2 * I + c0 + pr) * row_bytes, kbx, kqs);
                uu[j].load(w + (size_t(e) * 2 * I + I + c0 + pr) * row_bytes, kbx, kqs);
            }
        }
#pragma unroll
        for (int j = 0; j < B; ++j) {
            if (prs[j] < 0) continue;
#pragma unroll
            for (int n = 0; n < MAXM; ++n)
                if ((tokens >> n) & 1u) {
                    const float g = ug[j].dot(&xs[n][kbys[j]]), u = uu[j].dot(&xs[n][kbys[j]]);
#pragma unroll
                    for (int p = 0; p < UP_PAIRS; ++p)
                        if (p == prs[j]) sg[p][n] += g, su[p][n] += u;
                }
        }
    }
#pragma unroll
    for (int p = 0; p < UP_PAIRS; ++p)
#pragma unroll
        for (int n = 0; n < MAXM; ++n) {
            const float g = warp_sum(sg[p][n]), u = warp_sum(su[p][n]);
            if (lane == 0) acc[grp * UP_ROWS + warp * UP_PAIRS + p][n] = g, acc[ROWS + grp * UP_ROWS + warp * UP_PAIRS + p][n] = u;
        }
    }
    __syncthreads();
    for (int t = threadIdx.x; t < ROWS * MAXM; t += THREADS) {
        const int p = t % ROWS, n = t / ROWS;
        uint32_t kmask = n < M ? uint32_t(mask >> (8 * n)) & 0xffu : 0u;
        if (kmask) {
            const float gate = bf16_round(acc[p][n]);
            const float up = bf16_round(acc[ROWS + p][n]);
            const float a = tanhf(gate / 4.f) * 4.f * (1.f / (1.f + expf(-gate))) * (tanhf(up / 25.f) * 25.f);
            const __nv_bfloat16 ab = __float2bfloat16_rn(a);
            while (kmask) {
                const int k = __ffs(kmask) - 1;
                kmask &= kmask - 1;
                out[(size_t(n) * TOPK + k) * I + tile * ROWS + p] = ab;
            }
        }
    }
}

// ------------------------------------------------------------------------------------- down
// Stage 1, grid (H / DOWN_TILE, M * TOPK): the owner slot of each unique expert computes its 16 output rows for
// the tokens routed to it, BF16-rounded into y[slot][token][row] (fp32 [64, 8, H]). Warp w takes rows 2w, 2w+1.
template <int T, int G>
__global__ void __launch_bounds__(THREADS, GGUF_MINB) moe_down_gguf_kernel(const block_q8_1* __restrict__ aq, const int* __restrict__ idx,
                                                               const uint8_t* __restrict__ w, size_t row_bytes, int M,
                                                               float* __restrict__ y) {
    constexpr int XS = I / QK8_1;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tile = blockIdx.x, slot = blockIdx.y, nslots = M * TOPK;
    const int lo = lane < nslots ? idx[lane] : -1;
    const int hi = lane + 32 < nslots ? idx[lane + 32] : -1;
    const int e = idx[slot];
    const uint64_t mask = slot_mask(e, lo, hi);
    if (__ffsll(static_cast<long long>(mask)) - 1 != slot) return;
    uint32_t tokens = 0;
    int first[MAXM];
#pragma unroll
    for (int n = 0; n < MAXM; ++n) {
        const uint32_t km = n < M ? uint32_t(mask >> (8 * n)) & 0xffu : 0u;
        first[n] = km ? n * TOPK + (__ffs(km) - 1) : 0;
        if (km) tokens |= 1u << n;
    }
    using U = gguf::Unit<T>;
    constexpr int nb = I / U::qk, units = U::qi / U::vdr, total = nb * units;
#pragma unroll 1
    for (int grp = 0; grp < G; ++grp) {
    const int r0 = (tile * G + grp) * DOWN_TILE + warp * DOWN_RPW;
    float sr[DOWN_RPW][MAXM];
#pragma unroll
    for (int r = 0; r < DOWN_RPW; ++r)
#pragma unroll
        for (int n = 0; n < MAXM; ++n) sr[r][n] = 0.f;
    constexpr int B = GGUF_BATCH;
    for (int base = lane; base < DOWN_RPW * total; base += 32 * B) {
        U a[B];
        int rrs[B], kbys[B];
#pragma unroll
        for (int j = 0; j < B; ++j) {
            const int item = base + 32 * j;
            rrs[j] = -1;
            if (item < DOWN_RPW * total) {
                const int rr = item / total, q = item % total;
                const int kbx = q / units, kqs = U::vdr * (q % units);
                kbys[j] = kbx * (U::qk / QK8_1);
                rrs[j] = rr;
                a[j].load(w + (size_t(e) * H + r0 + rr) * row_bytes, kbx, kqs);
            }
        }
#pragma unroll
        for (int j = 0; j < B; ++j) {
            if (rrs[j] < 0) continue;
#pragma unroll
            for (int n = 0; n < MAXM; ++n)
                if ((tokens >> n) & 1u) {
                    const float v = a[j].dot(aq + size_t(first[n]) * XS + kbys[j]);
#pragma unroll
                    for (int r = 0; r < DOWN_RPW; ++r)
                        if (r == rrs[j]) sr[r][n] += v;
                }
        }
    }
#pragma unroll
    for (int r = 0; r < DOWN_RPW; ++r)
#pragma unroll
        for (int n = 0; n < MAXM; ++n) {
            const float v = warp_sum(sr[r][n]);
            if (lane == 0 && ((tokens >> n) & 1u)) y[(size_t(slot) * MAXM + n) * H + r0 + r] = bf16_round(v);
        }
    }
}

// Stage 2, grid (H / DOWN_TILE): routing-weight sum over each token's 8 slots (the owner slot's rows) and the
// static conv epilogue of moe.cu's moe_down.
__global__ void __launch_bounds__(THREADS) moe_down_combine_kernel(const float* __restrict__ y, const int* __restrict__ idx,
                                                                  const float* __restrict__ wts, int M,
                                                                  const float* __restrict__ conv_w,
                                                                  float* __restrict__ conv_state, float* __restrict__ out,
                                                                  float* __restrict__ conv_hist) {
    __shared__ uint8_t owner[TOPK * MAXM];
    __shared__ float sbuf[MAXM][DOWN_TILE];
    const int tile = blockIdx.x, nslots = M * TOPK;
    if (threadIdx.x < nslots) {  // owner = first slot routed to the same expert
        const int sl = threadIdx.x, e = idx[sl];
        int o = sl;
        for (int j = 0; j < sl; ++j)
            if (idx[j] == e) {
                o = j;
                break;
            }
        owner[sl] = uint8_t(o);
    }
    __syncthreads();
    if (threadIdx.x < DOWN_TILE * M) {
        const int r = threadIdx.x % DOWN_TILE, n = threadIdx.x / DOWN_TILE;
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            const int sl = n * TOPK + k;
            s = __fadd_rn(s, __fmul_rn(wts[sl], y[(size_t(owner[sl]) * MAXM + n) * H + tile * DOWN_TILE + r]));
        }
        sbuf[n][r] = bf16_round(s);
    }
    __syncthreads();
    if (threadIdx.x < DOWN_TILE) {
        const int ch = tile * DOWN_TILE + threadIdx.x;
        const float4 cw = *reinterpret_cast<const float4*>(conv_w + 4 * ch);
        float ext[MAXM + 3];
        ext[0] = conv_state[ch];
        ext[1] = conv_state[H + ch];
        ext[2] = conv_state[2 * H + ch];
#pragma unroll
        for (int m = 0; m < MAXM; ++m) ext[m + 3] = m < M ? sbuf[m][threadIdx.x] : 0.f;
        constexpr float inv_sqrt5 = 1.f / 2.2360679774997896f;
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m < M) {
                float a = __fmul_rn(cw.x, ext[m]);
                a = __fadd_rn(a, __fmul_rn(cw.y, ext[m + 1]));
                a = __fadd_rn(a, __fmul_rn(cw.z, ext[m + 2]));
                a = __fadd_rn(a, __fmul_rn(cw.w, ext[m + 3]));
                out[size_t(m) * H + ch] = __fmul_rn(__fadd_rn(a, ext[m + 3]), inv_sqrt5);
                if (conv_hist) conv_hist[size_t(m) * H + ch] = ext[m + 3];
            }
        }
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int m = 1; m <= MAXM; ++m)
                if (m == M) conv_state[j * H + ch] = ext[m + j];
    }
}

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("moe_gguf: ") + what);
}

template <template <int> class Kernel, typename... Args>
void dispatch(int type, dim3 grid, cudaStream_t stream, Args... args) {
    switch (type) {
#define TW_CASE(T_)                                                         \
    case T_: Kernel<T_>::launch(grid, stream, args...); break;
        TW_CASE(16) TW_CASE(17) TW_CASE(18) TW_CASE(19) TW_CASE(20) TW_CASE(21) TW_CASE(22) TW_CASE(42)
#undef TW_CASE
        default: require(false, "unsupported GGUF expert type");
    }
    CUDA_CHECK(cudaGetLastError());
}
int env_int(const char* name, int fallback) {
    const char* e = std::getenv(name);
    return e && *e ? std::atoi(e) : fallback;
}
// rows-per-block multipliers (runtime-selectable for tuning: GGUF_UP_G, GGUF_DOWN_G in {1, 2, 4, 8})
int up_groups() {
    static const int g = env_int("GGUF_UP_G", 4);
    return g;
}
int down_groups() {
    static const int g = env_int("GGUF_DOWN_G", 4);
    return g;
}
template <int T>
struct UpLaunch {
    template <typename... A>
    static void launch(dim3 g, cudaStream_t s, A... a) {
        switch (up_groups()) {
            case 1: moe_up_gguf_kernel<T, 1><<<g, THREADS, 0, s>>>(a...); break;
            case 2: g.x /= 2; moe_up_gguf_kernel<T, 2><<<g, THREADS, 0, s>>>(a...); break;
            case 8: g.x /= 8; moe_up_gguf_kernel<T, 8><<<g, THREADS, 0, s>>>(a...); break;
            default: g.x /= 4; moe_up_gguf_kernel<T, 4><<<g, THREADS, 0, s>>>(a...); break;
        }
    }
};
template <int T>
struct DownLaunch {
    template <typename... A>
    static void launch(dim3 g, cudaStream_t s, A... a) {
        switch (down_groups()) {
            case 1: moe_down_gguf_kernel<T, 1><<<g, THREADS, 0, s>>>(a...); break;
            case 2: g.x /= 2; moe_down_gguf_kernel<T, 2><<<g, THREADS, 0, s>>>(a...); break;
            case 8: g.x /= 8; moe_down_gguf_kernel<T, 8><<<g, THREADS, 0, s>>>(a...); break;
            default: g.x /= 4; moe_down_gguf_kernel<T, 4><<<g, THREADS, 0, s>>>(a...); break;
        }
    }
};

}  // namespace

std::size_t gguf_row_bytes(int type, int K) {
    switch (type) {
        case 16: return K / QK_K * sizeof(block_iq2_xxs);
        case 17: return K / QK_K * sizeof(block_iq2_xs);
        case 18: return K / QK_K * sizeof(block_iq3_xxs);
        case 19: return K / QK_K * sizeof(block_iq1_s);
        case 20: return K / QK4_NL * sizeof(block_iq4_nl);
        case 21: return K / QK_K * sizeof(block_iq3_s);
        case 22: return K / QK_K * sizeof(block_iq2_s);
        case 42: return K / QK2_0 * sizeof(block_q2_0);
        default: return 0;
    }
}

std::size_t q8_1_bytes(int rows, int K) { return std::size_t(rows) * (K / QK8_1) * sizeof(block_q8_1); }

void quantize_q8_1(const __nv_bfloat16* x, int rows, int K, void* q8, cudaStream_t stream) {
    require(K % QK8_1 == 0 && rows > 0, "quantize_q8_1: bad shape");
    quantize_q8_1_kernel<<<dim3((K + 255) / 256, rows), 256, 0, stream>>>(x, K, static_cast<block_q8_1*>(q8));
    CUDA_CHECK(cudaGetLastError());
}

void moe_up_gguf(const void* xq8, int M, const int* idx, int type, const void* w13, std::size_t row_bytes,
                 __nv_bfloat16* out, cudaStream_t stream) {
    require(M >= 1 && M <= MAXM && xq8 && idx && w13 && out, "moe_up_gguf: bad arguments");
    require(row_bytes == gguf_row_bytes(type, H), "moe_up_gguf: row size does not match the type");
    dispatch<UpLaunch>(type, dim3(I / UP_ROWS, M * TOPK), stream, static_cast<const block_q8_1*>(xq8), idx,
                       static_cast<const uint8_t*>(w13), row_bytes, M, out);
}

void moe_down_gguf(const void* act_q8, int M, const int* idx, const float* wts, int type, const void* w2,
                   std::size_t row_bytes, float* y_scratch, const float* conv_w, float* conv_state, float* out,
                   float* conv_hist, cudaStream_t stream) {
    require(M >= 1 && M <= MAXM && act_q8 && idx && wts && w2 && y_scratch && conv_w && conv_state && out,
            "moe_down_gguf: bad arguments");
    require(row_bytes == gguf_row_bytes(type, I), "moe_down_gguf: row size does not match the type");
    dispatch<DownLaunch>(type, dim3(H / DOWN_TILE, M * TOPK), stream, static_cast<const block_q8_1*>(act_q8), idx,
                         static_cast<const uint8_t*>(w2), row_bytes, M, y_scratch);
    moe_down_combine_kernel<<<H / DOWN_TILE, THREADS, 0, stream>>>(y_scratch, idx, wts, M, conv_w, conv_state, out,
                                                                  conv_hist);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t down_scratch_bytes() { return std::size_t(TOPK * MAXM) * MAXM * H * sizeof(float); }

}  // namespace ingine::kernels::moe
