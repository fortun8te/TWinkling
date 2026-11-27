// GGUF weight units decoded once and dotted with several Q8_1 tokens. Each Unit<T> splits llama.cpp's
// vec_dot_<type>_q8_1 (kernels/moe/gguf_dot.cuh, MIT) into load() (weight unit -> packed int8 values and
// its scales) and dot() (one token's Q8_1 blocks at the unit's kby) with the same integer arithmetic, so
// dot() returns exactly vec_dot(..., kbx, iqs) for that token.
#pragma once
#include "kernels/moe/gguf_dot.cuh"

namespace ingine::kernels::moe::gguf {

template <int T>
struct Unit;

template <>
struct Unit<16> {  // IQ2_XXS: iqs even, 32 values, y block iqs/2
    static constexpr int qk = QK_K, qi = QI2_XXS, vdr = 2;
    int w[8], ls, iq;
    float d;
    __device__ __forceinline__ void load(const void* vbq, int kbx, int iqs) {
        const block_iq2_xxs* b = (const block_iq2_xxs*)vbq + kbx;
        const int q2 = get_int_b2(b->qs, iqs);
        const uint8_t* aux8 = (const uint8_t*)&q2;
        const uint32_t aux32 = get_int_b2(b->qs, iqs + 1);
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 g = ((const uint2*)iq2xxs_grid)[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int s0 = __vcmpne4(signs & 0x08040201, 0), s1 = __vcmpne4(signs & 0x80402010, 0);
            w[k0] = __vsub4(g.x ^ s0, s0);
            w[k0 + 1] = __vsub4(g.y ^ s1, s1);
        }
        ls = aux32 >> 27 | 1;
        d = __half2float(b->d);
        iq = iqs;
    }
    __device__ __forceinline__ float dot(const block_q8_1* y) const {
        const block_q8_1& yb = y[iq / 2];
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = __dp4a(w[j], get_int_b4(yb.qs, j), sumi);
        sumi = sumi * ls / 8;
        return d * __low2float(yb.ds) * sumi;
    }
};

template <>
struct Unit<17> {  // IQ2_XS
    static constexpr int qk = QK_K, qi = QI2_XS, vdr = 2;
    int w[8], ls0, ls1, iq;
    float d;
    __device__ __forceinline__ void load(const void* vbq, int kbx, int iqs) {
        const block_iq2_xs* b = (const block_iq2_xs*)vbq + kbx;
        const int2 qp = make_int2(get_int_b2(b->qs, iqs + 0), get_int_b2(b->qs, iqs + 1));
        const uint16_t* q2 = (const uint16_t*)&qp;
        ls0 = b->scales[iqs / 2] & 0x0F;
        ls1 = b->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint2 g = ((const uint2*)iq2xs_grid)[q2[l0 / 2] & 0x1FF];
            const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
            const int s0 = __vcmpne4(signs & 0x08040201, 0), s1 = __vcmpne4(signs & 0x80402010, 0);
            w[l0] = __vsub4(g.x ^ s0, s0);
            w[l0 + 1] = __vsub4(g.y ^ s1, s1);
        }
        d = __half2float(b->d);
        iq = iqs;
    }
    __device__ __forceinline__ float dot(const block_q8_1* y) const {
        const block_q8_1& yb = y[iq / 2];
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) s0 = __dp4a(w[j], get_int_b4(yb.qs, j), s0);
#pragma unroll
        for (int j = 4; j < 8; ++j) s1 = __dp4a(w[j], get_int_b4(yb.qs, j), s1);
        const int sumi = (s0 * ls0 + s1 * ls1 + (s0 + s1) / 2) / 4;
        return d * __low2float(yb.ds) * sumi;
    }
};

template <>
struct Unit<22> {  // IQ2_S
    static constexpr int qk = QK_K, qi = QI2_S, vdr = 2;
    int w[8], ls0, ls1, iq;
    float d;
    __device__ __forceinline__ void load(const void* vbq, int kbx, int iqs) {
        const block_iq2_s* b = (const block_iq2_s*)vbq + kbx;
        const int qsp = get_int_b2(b->qs, iqs / 2);
        const uint8_t* qs = (const uint8_t*)&qsp;
        const int qh = b->qh[iqs / 2];
        const int sp32 = get_int_b2(b->qs, QK_K / 32 + iqs / 2);
        const uint8_t* sp = (const uint8_t*)&sp32;
        ls0 = b->scales[iqs / 2] & 0x0F;
        ls1 = b->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int* g = (const int*)(iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
            const int s0 = __vcmpne4(((sp[l0 / 2] & 0x03) << 7) | ((sp[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int s1 = __vcmpne4(((sp[l0 / 2] & 0x30) << 3) | ((sp[l0 / 2] & 0xC0) << 17), 0x00000000);
            w[l0] = __vsub4(g[0] ^ s0, s0);
            w[l0 + 1] = __vsub4(g[1] ^ s1, s1);
        }
        d = __half2float(b->d);
        iq = iqs;
    }
    __device__ __forceinline__ float dot(const block_q8_1* y) const {
        const block_q8_1& yb = y[iq / 2];
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) s0 = __dp4a(w[j], get_int_b4(yb.qs, j), s0);
#pragma unroll
        for (int j = 4; j < 8; ++j) s1 = __dp4a(w[j], get_int_b4(yb.qs, j), s1);
        const int sumi = (s0 * ls0 + s1 * ls1 + (s0 + s1) / 2) / 4;
        return d * __low2float(yb.ds) * sumi;
    }
};

template <>
struct Unit<18> {  // IQ3_XXS
    static constexpr int qk = QK_K, qi = QI3_XXS, vdr = 2;
    int w[8], ls, iq;
    float d;
    __device__ __forceinline__ void load(const void* vbq, int kbx, int iqs) {
        const block_iq3_xxs* b = (const block_iq3_xxs*)vbq + kbx;
        const int2 qp = make_int2(get_int_b2(b->qs, iqs), get_int_b2(b->qs, iqs + 1));
        const uint8_t* q3 = (const uint8_t*)&qp;
        const uint32_t aux32 = get_int_b2(b->qs, QK_K / 16 + iqs / 2);
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 g = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int s0 = __vcmpne4(signs & 0x08040201, 0), s1 = __vcmpne4(signs & 0x80402010, 0);
            w[l0] = __vsub4(g.x ^ s0, s0);
            w[l0 + 1] = __vsub4(g.y ^ s1, s1);
        }
        ls = aux32 >> 28;
        d = __half2float(b->d);
        iq = iqs;
    }
    __device__ __forceinline__ float dot(const block_q8_1* y) const {
        const block_q8_1& yb = y[iq / 2];
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = __dp4a(w[j], get_int_b4(yb.qs, j), sumi);
        sumi = (ls * sumi + sumi / 2) / 2;
        return d * __low2float(yb.ds) * sumi;
    }
};

template <>
struct Unit<21> {  // IQ3_S
    static constexpr int qk = QK_K, qi = QI3_S, vdr = 2;
    int w[8], sc, iq;
    float d;
    __device__ __forceinline__ void load(const void* vbq, int kbx, int iqs) {
        const block_iq3_s* b = (const block_iq3_s*)vbq + kbx;
        const int2 qp = make_int2(get_int_b2(b->qs, iqs + 0), get_int_b2(b->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*)&qp;
        const int qh = b->qh[iqs / 2];
        const int sp32 = get_int_b2(b->signs, iqs / 2);
        const uint8_t* sp = (const uint8_t*)&sp32;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 g = make_int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                     iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            const int s0 = __vcmpne4(((sp[l0 / 2] & 0x03) << 7) | ((sp[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int s1 = __vcmpne4(((sp[l0 / 2] & 0x30) << 3) | ((sp[l0 / 2] & 0xC0) << 17), 0x00000000);
            w[l0] = __vsub4(g.x ^ s0, s0);
            w[l0 + 1] = __vsub4(g.y ^ s1, s1);
        }
        sc = 1 + 2 * ((b->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        d = __half2float(b->d);
        iq = iqs;
    }
    __device__ __forceinline__ float dot(const block_q8_1* y) const {
        const block_q8_1& yb = y[iq / 2];
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = __dp4a(w[j], get_int_b4(yb.qs, j), sumi);
        sumi *= sc;
        return d * __low2float(yb.ds) * sumi;
    }
};

template <>
struct Unit<19> {  // IQ1_S: iqs 0..7 selects the 32-block (and y block iqs)
    static constexpr int qk = QK_K, qi = QI1_S, vdr = 1;
    int w[8], iq;
    float d1q, delta;
    __device__ __forceinline__ void load(const void* vbq, int kbx, int iqs) {
        const block_iq1_s* b = (const block_iq1_s*)vbq + kbx;
        const int qsp = get_int_b2(b->qs, iqs);
        const uint8_t* qs = (const uint8_t*)&qsp;
        const int qh = b->qh[iqs];
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int g = iq1s_grid_gpu[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
            w[l0] = (g >> 0) & 0x0F0F0F0F;
            w[l0 + 1] = (g >> 4) & 0x0F0F0F0F;
        }
        d1q = __half2float(b->d) * (((qh >> 11) & 0x0E) + 1);
        delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
        iq = iqs;
    }
    __device__ __forceinline__ float dot(const block_q8_1* y) const {
        const block_q8_1& yb = y[iq];
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = __dp4a(w[j], get_int_b4(yb.qs, j), sumi);
        const float2 ds = __half22float2(yb.ds);
        return d1q * (ds.x * sumi + ds.y * delta);
    }
};

template <>
struct Unit<20> {  // IQ4_NL: iqs 0 or 2 within the 32-block; ints iqs, iqs+1 (low) and iqs+4, iqs+5 (high)
    static constexpr int qk = QK4_NL, qi = QI4_NL, vdr = 2;
    int w[4], iq;
    float d;
    __device__ __forceinline__ void load(const void* vbq, int kbx, int iqs) {
        const block_iq4_nl* b = (const block_iq4_nl*)vbq + kbx;
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            const int2 v = get_int_from_table_16(get_int_b2(b->qs, iqs + l), kvalues_iq4nl);
            w[l] = v.x;
            w[2 + l] = v.y;
        }
        d = __half2float(b->d);
        iq = iqs;
    }
    __device__ __forceinline__ float dot(const block_q8_1* y) const {
        const int* q8 = (const int*)y->qs + iq;
        int sumi = 0;
        sumi = __dp4a(w[0], q8[0], sumi);
        sumi = __dp4a(w[2], q8[4], sumi);
        sumi = __dp4a(w[1], q8[1], sumi);
        sumi = __dp4a(w[3], q8[5], sumi);
        return d * __low2float(y->ds) * sumi;
    }
};

template <>
struct Unit<42> {  // Q2_0: iqs 0/1 selects the 32-value half (y block iqs)
    static constexpr int qk = QK2_0, qi = QI2_0, vdr = 1;
    int w[8], iq;
    float d;
    __device__ __forceinline__ void load(const void* vbq, int kbx, int iqs) {
        const block_q2_0* b = (const block_q2_0*)vbq + kbx;
        const int16_t* qs = (const int16_t*)b->qs + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
            const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
            w[2 * j] = __byte_perm(qe, qo, 0x5140);
            w[2 * j + 1] = __byte_perm(qe, qo, 0x7362);
        }
        d = __half2float(b->d);
        iq = iqs;
    }
    __device__ __forceinline__ float dot(const block_q8_1* y) const {
        const block_q8_1& yb = y[iq];
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = __dp4a(get_int_b4(yb.qs, j), w[j], sumi);
        return d * __low2float(yb.ds) * sumi;
    }
};

}  // namespace ingine::kernels::moe::gguf
