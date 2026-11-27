// GGUF expert blocks -> TWinkling's NVFP4 expert layout, for the prefill's grouped NVFP4 GEMMs.
// Dequantization follows llama.cpp's dequantize_row_* (ggml-quants.c, MIT); NVFP4 packing uses one fp32
// factor per expert (amax / (448 * 6)), an E4M3 scale per 16 values and round-to-nearest E2M1 codes,
// i.e. value = e2m1(code) * e4m3(scale) * factor as kernels/moe/dequant.cu decodes it.
#include "kernels/moe/moe.h"
#include "kernels/moe/gguf_dot.cuh"

#include "core/device.h"

#include <cuda_fp8.h>

#include <stdexcept>
#include <string>

namespace ingine::kernels::moe {
namespace {

constexpr int IH = 1152;  // I: gate rows 0..I-1, up rows I..2I-1 in GGUF ffn_gate_up_exps

__device__ __forceinline__ float h2f(ggml_half h) { return __half2float(h); }

// 32 consecutive dequantized values of a GGUF row: values sub*32 .. sub*32+31
template <int T>
__device__ void dequant32(const uint8_t* row, int sub, float (&y)[32]);

template <>
__device__ void dequant32<16>(const uint8_t* row, int sub, float (&y)[32]) {  // IQ2_XXS
    const block_iq2_xxs& b = reinterpret_cast<const block_iq2_xxs*>(row)[sub / 8];
    const int ib32 = sub % 8;
    uint32_t aux32[2];
    memcpy(aux32, b.qs + 4 * ib32, 8);
    const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
    const float db = h2f(b.d) * (0.5f + (aux32[1] >> 28)) * 0.25f;
    for (int l = 0; l < 4; ++l) {
        const uint8_t* grid = reinterpret_cast<const uint8_t*>(iq2xxs_grid + aux8[l]);
        const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
        for (int j = 0; j < 8; ++j) y[8 * l + j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    }
}
template <>
__device__ void dequant32<17>(const uint8_t* row, int sub, float (&y)[32]) {  // IQ2_XS
    const block_iq2_xs& b = reinterpret_cast<const block_iq2_xs*>(row)[sub / 8];
    const int ib32 = sub % 8;
    const float d = h2f(b.d);
    const float db[2] = {d * (0.5f + (b.scales[ib32] & 0xf)) * 0.25f, d * (0.5f + (b.scales[ib32] >> 4)) * 0.25f};
    for (int l = 0; l < 4; ++l) {
        const uint8_t* grid = reinterpret_cast<const uint8_t*>(iq2xs_grid + (b.qs[4 * ib32 + l] & 511));
        const uint8_t signs = ksigns_iq2xs[b.qs[4 * ib32 + l] >> 9];
        for (int j = 0; j < 8; ++j) y[8 * l + j] = db[l / 2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    }
}
template <>
__device__ void dequant32<22>(const uint8_t* row, int sub, float (&y)[32]) {  // IQ2_S
    const block_iq2_s& b = reinterpret_cast<const block_iq2_s*>(row)[sub / 8];
    const int ib32 = sub % 8;
    const float d = h2f(b.d);
    const uint8_t* qs = b.qs + 4 * ib32;
    const uint8_t* signs = b.qs + QK_K / 8 + 4 * ib32;
    const float db[2] = {d * (0.5f + (b.scales[ib32] & 0xf)) * 0.25f, d * (0.5f + (b.scales[ib32] >> 4)) * 0.25f};
    for (int l = 0; l < 4; ++l) {
        const uint8_t* grid = reinterpret_cast<const uint8_t*>(iq2s_grid + (qs[l] | (b.qh[ib32] << (8 - 2 * l) & 0x300)));
        for (int j = 0; j < 8; ++j) y[8 * l + j] = db[l / 2] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
    }
}
template <>
__device__ void dequant32<18>(const uint8_t* row, int sub, float (&y)[32]) {  // IQ3_XXS
    const block_iq3_xxs& b = reinterpret_cast<const block_iq3_xxs*>(row)[sub / 8];
    const int ib32 = sub % 8;
    const uint8_t* qs = b.qs + 8 * ib32;
    uint32_t aux32;
    memcpy(&aux32, b.qs + QK_K / 4 + 4 * ib32, 4);
    const float db = h2f(b.d) * (0.5f + (aux32 >> 28)) * 0.5f;
    for (int l = 0; l < 4; ++l) {
        const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
        const uint8_t* g1 = reinterpret_cast<const uint8_t*>(iq3xxs_grid + qs[2 * l + 0]);
        const uint8_t* g2 = reinterpret_cast<const uint8_t*>(iq3xxs_grid + qs[2 * l + 1]);
        for (int j = 0; j < 4; ++j) {
            y[8 * l + j] = db * g1[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            y[8 * l + j + 4] = db * g2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
    }
}
template <>
__device__ void dequant32<21>(const uint8_t* row, int sub, float (&y)[32]) {  // IQ3_S
    const block_iq3_s& b = reinterpret_cast<const block_iq3_s*>(row)[sub / 8];
    const int ib32 = sub % 8;  // pairs (2k, 2k+1) share scales[k] and qh[...] as dequantize_row_iq3_s
    const float d = h2f(b.d);
    const float db = d * (1 + 2 * ((b.scales[ib32 / 2] >> (4 * (ib32 % 2))) & 0xf));
    const uint8_t* qs = b.qs + 8 * ib32;
    const uint8_t qh = b.qh[ib32];
    const uint8_t* signs = b.signs + 4 * ib32;
    for (int l = 0; l < 4; ++l) {
        const uint8_t* g1 = reinterpret_cast<const uint8_t*>(iq3s_grid + (qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256)));
        const uint8_t* g2 = reinterpret_cast<const uint8_t*>(iq3s_grid + (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
        for (int j = 0; j < 4; ++j) {
            y[8 * l + j] = db * g1[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
            y[8 * l + j + 4] = db * g2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
    }
}
template <>
__device__ void dequant32<19>(const uint8_t* row, int sub, float (&y)[32]) {  // IQ1_S
    const block_iq1_s& b = reinterpret_cast<const block_iq1_s*>(row)[sub / 8];
    const int ib = sub % 8;
    const float d = h2f(b.d);
    const uint8_t* qs = b.qs + 4 * ib;
    const uint16_t qh = b.qh[ib];
    const float dl = d * (2 * ((qh >> 12) & 7) + 1);
    const float delta = qh & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
    for (int l = 0; l < 4; ++l) {
        // iq1s_grid_gpu packs grid + 1 as nibbles: elements 0..3 in the low nibbles, 4..7 in the high ones
        const uint32_t grid = iq1s_grid_gpu[qs[l] | (((qh >> 3 * l) & 7) << 8)];
        for (int j = 0; j < 8; ++j) {
            const int g = int((grid >> (8 * (j & 3) + 4 * (j >> 2))) & 0xF) - 1;
            y[8 * l + j] = dl * (float(g) + delta);
        }
    }
}
template <>
__device__ void dequant32<20>(const uint8_t* row, int sub, float (&y)[32]) {  // IQ4_NL
    const block_iq4_nl& b = reinterpret_cast<const block_iq4_nl*>(row)[sub];
    const float d = h2f(b.d);
    for (int j = 0; j < QK4_NL / 2; ++j) {
        y[j] = d * kvalues_iq4nl[b.qs[j] & 0xf];
        y[j + QK4_NL / 2] = d * kvalues_iq4nl[b.qs[j] >> 4];
    }
}
template <>
__device__ void dequant32<42>(const uint8_t* row, int sub, float (&y)[32]) {  // Q2_0: 00=-1 01=0 10=+1 11=+2
    const block_q2_0& b = reinterpret_cast<const block_q2_0*>(row)[sub / 2];
    const float d = h2f(b.d);
    const int base = 32 * (sub % 2);
    for (int j = 0; j < 32; ++j) {
        const int jj = base + j;
        const int q = (b.qs[jj / 4] >> ((jj % 4) * 2)) & 3;
        y[j] = float(q - 1) * d;
    }
}

__device__ __forceinline__ int e2m1_code(float x) {
    // nearest of {0, .5, 1, 1.5, 2, 3, 4, 6} (ties to the even code as the cvt instruction would)
    const float a = fabsf(x);
    int c;
    if (a <= 0.25f) c = 0;
    else if (a < 0.75f) c = 1;
    else if (a <= 1.25f) c = 2;
    else if (a < 1.75f) c = 3;
    else if (a <= 2.5f) c = 4;
    else if (a < 3.5f) c = 5;
    else if (a <= 5.f) c = 6;
    else c = 7;
    return (x < 0.f && c) ? (c | 8) : c;
}

// thread per (expert, NVFP4 row, 32-value group); src row of NVFP4 row r: gate/up interleave on w13
template <int T>
__global__ void amax_kernel(const uint8_t* __restrict__ w, size_t row_bytes, int rows, int K, int E,
                            unsigned* __restrict__ amax) {
    const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const int subs = K / 32;
    if (t >= (long long)E * rows * subs) return;
    const int e = int(t / ((long long)rows * subs)), r = int(t / subs % rows), sub = int(t % subs);
    float y[32];
    dequant32<T>(w + (size_t(e) * rows + r) * row_bytes, sub, y);
    float m = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) m = fmaxf(m, fabsf(y[j]));
    atomicMax(amax + e, __float_as_uint(m));
}

template <int T>
__global__ void pack_kernel(const uint8_t* __restrict__ w, size_t row_bytes, int rows, int K, int E, bool interleave,
                            const unsigned* __restrict__ amax, uint8_t* __restrict__ codes,
                            __nv_fp8_e4m3* __restrict__ scales, float* __restrict__ factor) {
    const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const int subs = K / 32;
    if (t >= (long long)E * rows * subs) return;
    const int e = int(t / ((long long)rows * subs)), r = int(t / subs % rows), sub = int(t % subs);
    const int src = interleave ? ((r & 1) ? IH + r / 2 : r / 2) : r;
    const float f = fmaxf(__uint_as_float(amax[e]), 1e-30f) / (448.f * 6.f);
    if (r == 0 && sub == 0) factor[e] = f;
    float y[32];
    dequant32<T>(w + (size_t(e) * rows + src) * row_bytes, sub, y);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        float m = 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j) m = fmaxf(m, fabsf(y[16 * h + j]));
        const __nv_fp8_e4m3 s(m / 6.f / f);
        const float sd = float(s) * f;
        const size_t blk = (size_t(e) * rows + r) * (K / 16) + sub * 2 + h;
        scales[blk] = s;
        uint32_t packed[2] = {0u, 0u};
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int c = sd > 0.f ? e2m1_code(y[16 * h + j] / sd) : 0;
            packed[j / 8] |= uint32_t(c) << (4 * (j % 8));
        }
        *reinterpret_cast<uint2*>(codes + blk * 8) = make_uint2(packed[0], packed[1]);
    }
}

template <int T>
__global__ void dequant_rows_kernel(const uint8_t* __restrict__ w, size_t row_bytes, int K, const int* __restrict__ rows,
                                    int n, float* __restrict__ out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x, subs = K / 32;
    if (t >= n * subs) return;
    const int i = t / subs, sub = t % subs;
    float y[32];
    dequant32<T>(w + size_t(rows[i]) * row_bytes, sub, y);
    for (int j = 0; j < 32; ++j) out[size_t(i) * K + sub * 32 + j] = y[j];
}

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("gguf_transcode: ") + what);
}

}  // namespace

void gguf_to_nvfp4(int type, const void* w, std::size_t row_bytes, int rows, int K, int E, bool interleave_gate_up,
                   void* amax_scratch, void* codes, void* scales, float* factor, cudaStream_t stream) {
    require(row_bytes == gguf_row_bytes(type, K) && K % 32 == 0 && rows > 0 && E > 0, "bad shape");
    const long long n = (long long)E * rows * (K / 32);
    const int threads = 128, blocks = int((n + threads - 1) / threads);
    auto* am = static_cast<unsigned*>(amax_scratch);
    CUDA_CHECK(cudaMemsetAsync(am, 0, sizeof(unsigned) * E, stream));
    const auto* src = static_cast<const uint8_t*>(w);
    auto* c = static_cast<uint8_t*>(codes);
    auto* s = static_cast<__nv_fp8_e4m3*>(scales);
    switch (type) {
#define TW_CASE(T_)                                                                                     \
    case T_:                                                                                            \
        amax_kernel<T_><<<blocks, threads, 0, stream>>>(src, row_bytes, rows, K, E, am);               \
        pack_kernel<T_><<<blocks, threads, 0, stream>>>(src, row_bytes, rows, K, E, interleave_gate_up, am, c, s, factor); \
        break;
        TW_CASE(16) TW_CASE(17) TW_CASE(18) TW_CASE(19) TW_CASE(20) TW_CASE(21) TW_CASE(22) TW_CASE(42)
#undef TW_CASE
        default: require(false, "unsupported GGUF expert type");
    }
    CUDA_CHECK(cudaGetLastError());
}

void gguf_dequant_rows(int type, const void* w, std::size_t row_bytes, int K, const int* rows, int n, float* out,
                       cudaStream_t stream) {
    const int threads = 128, blocks = (n * (K / 32) + threads - 1) / threads;
    const auto* src = static_cast<const uint8_t*>(w);
    switch (type) {
#define TW_CASE(T_) case T_: dequant_rows_kernel<T_><<<blocks, threads, 0, stream>>>(src, row_bytes, K, rows, n, out); break;
        TW_CASE(16) TW_CASE(17) TW_CASE(18) TW_CASE(19) TW_CASE(20) TW_CASE(21) TW_CASE(22) TW_CASE(42)
#undef TW_CASE
        default: require(false, "unsupported GGUF expert type");
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::moe
